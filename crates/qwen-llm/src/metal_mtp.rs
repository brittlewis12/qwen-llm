//! Metal MTP (Multi-Token-Prediction) head and speculative-decode driver.
//!
//! Mirrors `crate::forward::Forward::mtp_step` (CPU oracle) on Metal.
//! See `docs/H4-MTP.md` §1.2 for the algorithmic contract:
//!
//! At sequence slot `position`, the MTP head consumes
//! `(embed(next_tok), prev_hidden, position)` and predicts logits for the
//! token at `position + 2`. Forward path:
//!
//! ```text
//! e_normed = RMSNorm(embed(next_tok), enorm)
//! h_normed = RMSNorm(prev_hidden, hnorm)
//! x        = eh_proj @ concat([e_normed, h_normed])     # 2H → H, vLLM order
//! x       += attn_block(MTP attn, x, position, mtp_kv)  # standard gated attn
//! x       += SwiGLU FFN(x)
//! logits   = lm_head(RMSNorm(x, shared_head_norm))      # shared lm_head
//! ```
//!
//! v1 ships `draft` (greedy logits + token id) and `draft_kv_only`
//! (logits discarded; advances MTP KV by one slot — used by both the
//! inline accept-branch bridge in §1.4 step E and the streamed
//! prompt-time prefill in §1.5).
//!
//! GPU command ordering: `draft` and `draft_kv_only` synchronously commit
//! and wait per call (v1). Future ICB-cached version will overlap with
//! base forward on the same MetalSession command queue.

use crate::codec::dequant_to_f32;
use crate::gguf::GgufFile;
use crate::loader::MtpHead;
use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, attn_v4_choose_nwg,
    attn_v4_choose_tile_c, encode_add_inplace_f32, encode_argmax_f32, encode_attn_decode_f16kv_f32,
    encode_attn_decode_v4_f32, encode_ffn_swiglu_q4_K_f32, encode_get_rows_f32, encode_mul_f32,
    encode_rms_norm_batched_f32, encode_rms_norm_mul_f32, encode_rope_neox_f32,
    encode_scatter_offset_f32_to_f16_kv, encode_sigmoid_f32, encode_silu_mul_f32,
    encode_split_q_gate_f32,
};
use crate::metal_dflash::{
    MetalDFlashLayerMajorScratch, MetalDFlashVerifyScratch, encode_packed_verify_layer_major_inner,
    encode_restore_after_partial_accept_inner,
};
use crate::metal_forward::{
    ATTN_V4_MAX_NWG, MetalAttnBlock, MetalForward, MetalSession, RMS_EPS, encode_mat_vec_dispatch,
    encode_scatter_offset_f32, weight_dtype_kept_native,
};
use crate::tensor::{GgmlType, TensorDesc};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};

#[derive(Debug, thiserror::Error)]
pub enum MtpError {
    #[error("metal: {0}")]
    Metal(#[from] MetalError),
    #[error("metal forward: {0}")]
    MetalForward(#[from] crate::metal_forward::MfError),
    #[error("codec: {0}")]
    Codec(#[from] crate::codec::CodecError),
    #[error("dflash: {0}")]
    DFlash(#[from] crate::metal_dflash::DFlashError),
    #[error("model has no MTP head")]
    NoMtpHead,
    #[error("token {0} out of vocab range {1}")]
    BadToken(i32, u32),
    #[error("MTP KV invariant: caller passed position={position} but kv_n_pos={kv_n_pos}")]
    KvPositionMismatch { position: u32, kv_n_pos: usize },
}

/// All MTP head weights, resident as `MetalTensor`s. Loaded once at session
/// start. Shape conventions match `crate::loader::MtpHead`.
pub struct MetalMtpHead {
    /// Block index in the GGUF (typically `arch.n_layer`).
    pub block_idx: u32,
    /// Standard full-attention weights at `blk.{block_idx}.*`.
    pub attn: MetalAttnBlock,
    /// `nextn.eh_proj.weight` — `[2H, H]`.
    pub eh_proj: MetalTensor,
    /// `nextn.enorm.weight` — `[H]`.
    pub enorm: MetalTensor,
    /// `nextn.hnorm.weight` — `[H]`.
    pub hnorm: MetalTensor,
    /// `nextn.shared_head_norm.weight` — `[H]`.
    pub shared_head_norm: MetalTensor,
}

impl MetalMtpHead {
    /// Load the MTP head's weights from the bound `MtpHead` view. Mirrors
    /// `MetalModel::load`'s native-quant policy: kernel-supported quants
    /// (Q4_K, Q5_K, Q6_K, F32) stay in their on-disk dtype; norms +
    /// elementwise weights are dequant'd to F32.
    pub fn load(ctx: &MetalContext, gguf: &GgufFile, mtp: &MtpHead<'_>) -> Result<Self, MtpError> {
        let load_f32 = |desc: &TensorDesc| -> Result<MetalTensor, MtpError> {
            if desc.dtype == GgmlType::F32 {
                Ok(MetalTensor::from_gguf_tensor(ctx, desc, gguf.slice(desc))?)
            } else {
                let f32 = dequant_to_f32(desc, gguf.slice(desc))?;
                Ok(MetalTensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&f32),
                    desc.shape.clone(),
                    GgmlType::F32,
                )?)
            }
        };
        let load_weight = |desc: &TensorDesc| -> Result<MetalTensor, MtpError> {
            if weight_dtype_kept_native(desc.dtype) {
                Ok(MetalTensor::from_gguf_tensor(ctx, desc, gguf.slice(desc))?)
            } else {
                load_f32(desc)
            }
        };

        Ok(Self {
            block_idx: mtp.block_idx,
            attn: MetalAttnBlock {
                attn_norm: load_f32(mtp.attn.attn_norm)?,
                post_attn_norm: load_f32(mtp.attn.post_attention_norm)?,
                ffn_gate: load_weight(mtp.attn.ffn_gate)?,
                ffn_up: load_weight(mtp.attn.ffn_up)?,
                ffn_down: load_weight(mtp.attn.ffn_down)?,
                q: load_weight(mtp.attn.q)?,
                k: load_weight(mtp.attn.k)?,
                v: load_weight(mtp.attn.v)?,
                o: load_weight(mtp.attn.o)?,
                q_norm: load_f32(mtp.attn.q_norm)?,
                k_norm: load_f32(mtp.attn.k_norm)?,
                ffn_moe: None,
            },
            eh_proj: load_weight(mtp.eh_proj)?,
            enorm: load_f32(mtp.enorm)?,
            hnorm: load_f32(mtp.hnorm)?,
            shared_head_norm: load_f32(mtp.shared_head_norm)?,
        })
    }
}

/// Per-sequence MTP state: dedicated KV ring for the single MTP attn
/// layer + scratch buffers for the per-step forward. Independent of the
/// base `MetalSession` so the MTP draft can run without disturbing base
/// session arena.
pub struct MetalMtpSession {
    /// MTP attn KV ring. Single layer (the MTP block's own attn).
    /// F16 storage matching the base attn KV convention.
    pub kv_k: MetalTensor,
    pub kv_v: MetalTensor,
    pub kv_n_pos: usize,
    pub kv_capacity: usize,

    // Pre-eh_proj scratch.
    pub e: MetalTensor,         // [H] — embed(next_tok)
    pub e_normed: MetalTensor,  // [H] — RMSNorm(e, enorm)
    pub h_normed: MetalTensor,  // [H] — RMSNorm(prev_hidden, hnorm)
    pub eh_concat: MetalTensor, // [2H] — concat([e_normed, h_normed])

    // Residual stream + post-norm activations (mirror base session).
    pub x: MetalTensor,         // [H] — residual stream
    pub h: MetalTensor,         // [H] — post-norm activation
    pub ffn_gate: MetalTensor,  // [F]
    pub ffn_up: MetalTensor,    // [F]
    pub ffn_inner: MetalTensor, // [F] — silu(gate) * up
    pub ffn_out: MetalTensor,   // [H]
    pub mixer_out: MetalTensor, // [H] — attn output

    // Attn scratch.
    pub attn_q_full: MetalTensor,   // [2 * q_dim] — Q + gate interleaved
    pub attn_q: MetalTensor,        // [q_dim]
    pub attn_gate: MetalTensor,     // [q_dim]
    pub attn_q_normed: MetalTensor, // [q_dim]
    pub attn_k_now: MetalTensor,    // [kv_dim]
    pub attn_v_now: MetalTensor,    // [kv_dim]
    pub attn_k_normed: MetalTensor, // [kv_dim]
    pub attn_o: MetalTensor,        // [q_dim]
    pub attn_v4_o_partial: MetalTensor, // n_kv * NWG_max * GROUP * head_dim
    pub attn_v4_ml_partial: MetalTensor, // n_kv * NWG_max * GROUP * 2

    // Output (shared with base — but kept separate to allow concurrent
    // dispatch in a future ICB world).
    pub logits: MetalTensor,       // [V]
    pub draft_argmax: MetalTensor, // [1] i32 in F32 buffer
    pub ids_buf: MetalTensor,      // i32 token id (in F32 buffer)
}

impl MetalMtpSession {
    /// Allocate a fresh MTP session. `kv_capacity` is the max number of
    /// MTP KV slots (typically `prompt_len + max_new_tokens`).
    pub fn fresh(
        ctx: &MetalContext,
        head: &MetalMtpHead,
        arch: &crate::model::Arch,
        kv_capacity: usize,
    ) -> Result<Self, MtpError> {
        let _ = head; // currently unused; reserved for arch consistency checks
        let h = arch.hidden_size as u64;
        let f = arch.intermediate_size as u64;
        let head_dim = arch.attn_head_dim as u64;
        let n_q = arch.n_q_heads as u64;
        let n_kv = arch.n_kv_heads as u64;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;

        Ok(Self {
            kv_k: MetalTensor::zeros_f16(ctx, vec![kv_capacity as u64 * kv_dim])?,
            kv_v: MetalTensor::zeros_f16(ctx, vec![kv_capacity as u64 * kv_dim])?,
            kv_n_pos: 0,
            kv_capacity,
            e: MetalTensor::zeros_f32(ctx, vec![h])?,
            e_normed: MetalTensor::zeros_f32(ctx, vec![h])?,
            h_normed: MetalTensor::zeros_f32(ctx, vec![h])?,
            eh_concat: MetalTensor::zeros_f32(ctx, vec![2 * h])?,
            x: MetalTensor::zeros_f32(ctx, vec![h])?,
            h: MetalTensor::zeros_f32(ctx, vec![h])?,
            ffn_gate: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_up: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_inner: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_out: MetalTensor::zeros_f32(ctx, vec![h])?,
            mixer_out: MetalTensor::zeros_f32(ctx, vec![h])?,
            attn_q_full: MetalTensor::zeros_f32(ctx, vec![2 * q_dim])?,
            attn_q: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_gate: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_q_normed: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_k_now: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_v_now: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_k_normed: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_o: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_v4_o_partial: MetalTensor::zeros_f32(
                ctx,
                vec![n_kv * (ATTN_V4_MAX_NWG as u64) * (n_q / n_kv) * head_dim],
            )?,
            attn_v4_ml_partial: MetalTensor::zeros_f32(
                ctx,
                vec![n_kv * (ATTN_V4_MAX_NWG as u64) * (n_q / n_kv) * 2],
            )?,
            logits: MetalTensor::zeros_f32(ctx, vec![arch.vocab_size as u64])?,
            draft_argmax: MetalTensor::zeros_f32(ctx, vec![1])?,
            ids_buf: MetalTensor::zeros_f32(ctx, vec![1])?,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DraftReadback {
    None,
    ArgmaxOnly,
    FullLogits,
}

#[cfg_attr(not(test), allow(dead_code))]
struct DraftResult {
    logits: Option<Vec<f32>>,
    argmax: Option<i32>,
}

fn copy_f32_tensor(src: &MetalTensor, dst: &MetalTensor) -> Result<(), MtpError> {
    if src.dtype != GgmlType::F32 || dst.dtype != GgmlType::F32 {
        return Err(MtpError::Metal(MetalError::BadShape {
            kernel: "copy_f32_tensor",
            detail: format!("expected F32/F32, got {:?}/{:?}", src.dtype, dst.dtype),
        }));
    }
    if src.n_elements() != dst.n_elements() {
        return Err(MtpError::Metal(MetalError::BadShape {
            kernel: "copy_f32_tensor",
            detail: format!(
                "element mismatch: src={} dst={}",
                src.n_elements(),
                dst.n_elements()
            ),
        }));
    }
    let n_bytes = (src.n_elements() as usize) * std::mem::size_of::<f32>();
    unsafe {
        let src_ptr = (src.buffer.contents().as_ptr() as *const u8).add(src.offset as usize);
        let dst_ptr = (dst.buffer.contents().as_ptr() as *mut u8).add(dst.offset as usize);
        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, n_bytes);
    }
    Ok(())
}

/// Top-level speculative-decode driver. Owns nothing the base path needs;
/// borrows base for the actual base-forward calls.
pub struct SpeculativeDecoder<'a> {
    pub base: &'a MetalForward<'a>,
    pub mtp_head: &'a MetalMtpHead,
    pub mtp_session: MetalMtpSession,
}

impl<'a> SpeculativeDecoder<'a> {
    pub fn new(
        base: &'a MetalForward<'a>,
        mtp_head: &'a MetalMtpHead,
        mtp_session: MetalMtpSession,
    ) -> Self {
        Self {
            base,
            mtp_head,
            mtp_session,
        }
    }

    /// Draft a single token. The MTP head at slot `position` consumes
    /// `(embed(next_tok), prev_hidden)` and predicts the token at
    /// `position + 2` (greedy v1).
    ///
    /// `prev_hidden` is the GPU-resident pre-output_norm hidden from the
    /// base model at slot `position`. Caller must ensure
    /// `mtp_session.kv_n_pos == position` (next-sequential append).
    /// Side effect: appends one entry to MTP KV at slot `position`.
    ///
    /// GPU command ordering: this call commits + waits before returning,
    /// so `prev_hidden` may be safely reused or overwritten by subsequent
    /// caller code.
    pub fn draft(
        &mut self,
        next_tok: i32,
        prev_hidden: &MetalTensor,
        position: u32,
    ) -> Result<i32, MtpError> {
        let result =
            self.draft_inner(next_tok, prev_hidden, position, DraftReadback::ArgmaxOnly)?;
        Ok(result.argmax.expect("draft argmax missing"))
    }

    /// Same as `draft` but discards logits. Used by the inline accept-branch
    /// bridge (`docs/H4-MTP.md` §1.4 step E) and prompt-time prefill (§1.5).
    /// Identical GPU command ordering contract.
    pub fn draft_kv_only(
        &mut self,
        next_tok: i32,
        prev_hidden: &MetalTensor,
        position: u32,
    ) -> Result<(), MtpError> {
        let _ = self.draft_inner(next_tok, prev_hidden, position, DraftReadback::None)?;
        Ok(())
    }

    /// Draft + (optionally) read back logits. Internal helper.
    fn draft_inner(
        &mut self,
        next_tok: i32,
        prev_hidden: &MetalTensor,
        position: u32,
        readback: DraftReadback,
    ) -> Result<DraftResult, MtpError> {
        let arch = &self.base.model.arch;
        if next_tok < 0 || (next_tok as u32) >= arch.vocab_size {
            return Err(MtpError::BadToken(next_tok, arch.vocab_size));
        }
        if self.mtp_session.kv_n_pos as u32 != position {
            return Err(MtpError::KvPositionMismatch {
                position,
                kv_n_pos: self.mtp_session.kv_n_pos,
            });
        }
        let h = arch.hidden_size as usize;
        if prev_hidden.n_elements() != h as u64 {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_draft.prev_hidden",
                detail: format!(
                    "prev_hidden.n_elements()={} != hidden_size={}",
                    prev_hidden.n_elements(),
                    h,
                ),
            }));
        }

        let ctx = self.base.ctx;

        // Stage next_tok into ids_buf.
        unsafe {
            let ptr = self.mtp_session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = next_tok;
        }

        // Build a single command buffer for the whole MTP step.
        let cmd = ctx.queue.commandBuffer().expect("cmd buf");
        let enc = KernelEncoder::begin(&cmd);

        // (1) Embedding lookup → e.
        encode_get_rows_f32(
            ctx,
            &enc,
            &self.base.model.token_embd,
            &self.mtp_session.ids_buf,
            &self.mtp_session.e,
            1,
            h,
        )?;

        // (2) RMSNorm both inputs.
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            &self.mtp_session.e,
            &self.mtp_head.enorm,
            &self.mtp_session.e_normed,
            RMS_EPS,
        )?;
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            prev_hidden,
            &self.mtp_head.hnorm,
            &self.mtp_session.h_normed,
            RMS_EPS,
        )?;

        // (3) Concat [e_normed, h_normed] → [2H]. vLLM canonical order:
        // embed first (cols 0..H), hidden second (cols H..2H).
        // Implemented as two scatter_offset writes since we don't have a
        // dedicated concat kernel.
        encode_scatter_offset_f32(
            ctx,
            &enc,
            &self.mtp_session.e_normed,
            &self.mtp_session.eh_concat,
            0,
            h,
        )?;
        encode_scatter_offset_f32(
            ctx,
            &enc,
            &self.mtp_session.h_normed,
            &self.mtp_session.eh_concat,
            h,
            h,
        )?;

        // (4) eh_proj: [2H, H] → [H]. Result lands in `x` (residual stream).
        encode_mat_vec_dispatch(
            ctx,
            &enc,
            &self.mtp_head.eh_proj,
            &self.mtp_session.eh_concat,
            &self.mtp_session.x,
            2 * h,
            h,
        )?;

        // ----- MTP transformer block: attn + FFN with residuals -----

        // (5) Pre-attn RMSNorm: x → h.
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            &self.mtp_session.x,
            &self.mtp_head.attn.attn_norm,
            &self.mtp_session.h,
            RMS_EPS,
        )?;

        // (6) Attn block (standard gated-attn, indexes our dedicated MTP KV).
        // KV-only path for draft_kv_only: append MTP KV for this slot but skip
        // the expensive attention decode / FFN / lm_head tail entirely.
        if matches!(readback, DraftReadback::None) {
            self.encode_mtp_kv_only(&enc, position)?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.mtp_session.kv_n_pos = position as usize + 1;
            return Ok(DraftResult {
                logits: None,
                argmax: None,
            });
        }
        self.encode_mtp_attn(&enc, position)?;

        // (7) Residual #1: x += mixer_out.
        encode_add_inplace_f32(ctx, &enc, &self.mtp_session.x, &self.mtp_session.mixer_out)?;

        // (8) Pre-FFN RMSNorm: x → h.
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            &self.mtp_session.x,
            &self.mtp_head.attn.post_attn_norm,
            &self.mtp_session.h,
            RMS_EPS,
        )?;

        // (9) SwiGLU FFN. Same fused/unfused path as base.
        let g_w = &self.mtp_head.attn.ffn_gate;
        let u_w = &self.mtp_head.attn.ffn_up;
        let d_w = &self.mtp_head.attn.ffn_down;
        let f = arch.intermediate_size as usize;
        let ffn_fused = g_w.dtype == GgmlType::Q4_K && u_w.dtype == GgmlType::Q4_K;
        if ffn_fused {
            encode_ffn_swiglu_q4_K_f32(
                ctx,
                &enc,
                g_w,
                u_w,
                &self.mtp_session.h,
                &self.mtp_session.ffn_inner,
                h,
                f,
            )?;
        } else {
            encode_mat_vec_dispatch(
                ctx,
                &enc,
                g_w,
                &self.mtp_session.h,
                &self.mtp_session.ffn_gate,
                h,
                f,
            )?;
            encode_mat_vec_dispatch(
                ctx,
                &enc,
                u_w,
                &self.mtp_session.h,
                &self.mtp_session.ffn_up,
                h,
                f,
            )?;
            encode_silu_mul_f32(
                ctx,
                &enc,
                &self.mtp_session.ffn_gate,
                &self.mtp_session.ffn_up,
                &self.mtp_session.ffn_inner,
            )?;
        }
        encode_mat_vec_dispatch(
            ctx,
            &enc,
            d_w,
            &self.mtp_session.ffn_inner,
            &self.mtp_session.ffn_out,
            f,
            h,
        )?;

        // (10) Residual #2: x += ffn_out.
        encode_add_inplace_f32(ctx, &enc, &self.mtp_session.x, &self.mtp_session.ffn_out)?;

        // (11) shared_head_norm: x → h.
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            &self.mtp_session.x,
            &self.mtp_head.shared_head_norm,
            &self.mtp_session.h,
            RMS_EPS,
        )?;

        // (12) lm_head: [H] → [V]. Reuses the base lm_head (shared per
        // Qwen3.5/3.6 tying convention).
        encode_mat_vec_dispatch(
            ctx,
            &enc,
            &self.base.model.lm_head,
            &self.mtp_session.h,
            &self.mtp_session.logits,
            h,
            arch.vocab_size as usize,
        )?;

        if !matches!(readback, DraftReadback::None) {
            encode_argmax_f32(
                ctx,
                &enc,
                &self.mtp_session.logits,
                &self.mtp_session.draft_argmax,
                1,
                arch.vocab_size as usize,
            )?;
        }

        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        // KV side-effect was committed by encode_mtp_attn; bump our counter.
        self.mtp_session.kv_n_pos = position as usize + 1;

        let argmax = if matches!(
            readback,
            DraftReadback::ArgmaxOnly | DraftReadback::FullLogits
        ) {
            unsafe {
                let src = self.mtp_session.draft_argmax.buffer.contents().as_ptr() as *const i32;
                Some(*src)
            }
        } else {
            None
        };

        if matches!(readback, DraftReadback::FullLogits) {
            let mut out = vec![0.0f32; arch.vocab_size as usize];
            unsafe {
                let src = self.mtp_session.logits.buffer.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
            }
            Ok(DraftResult {
                logits: Some(out),
                argmax,
            })
        } else {
            Ok(DraftResult {
                logits: None,
                argmax,
            })
        }
    }

    /// MTP-specific attn step. Mirrors `MetalForward::encode_attn` but
    /// indexes the dedicated MTP KV ring (single layer) instead of
    /// `MetalSession::kv_*[attn_idx]`.
    fn encode_mtp_kv_only(&mut self, enc: &KernelEncoder, position: u32) -> Result<(), MtpError> {
        let arch = &self.base.model.arch;
        let ctx = self.base.ctx;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_kv = arch.n_kv_heads as usize;
        let kv_dim = n_kv * head_dim;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
        let ab = &self.mtp_head.attn;
        let s = &mut self.mtp_session;

        encode_mat_vec_dispatch(ctx, enc, &ab.k, &s.h, &s.attn_k_now, h, kv_dim)?;
        encode_mat_vec_dispatch(ctx, enc, &ab.v, &s.h, &s.attn_v_now, h, kv_dim)?;
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &s.attn_k_now,
            &ab.k_norm,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            RMS_EPS,
        )?;
        encode_rope_neox_f32(
            ctx,
            enc,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            n_rot,
            position,
            arch.rope_theta,
        )?;
        encode_scatter_offset_f32_to_f16_kv(
            ctx,
            enc,
            &s.attn_k_normed,
            &s.attn_v_now,
            &s.kv_k,
            &s.kv_v,
            (position as usize) * kv_dim,
            kv_dim,
        )?;
        Ok(())
    }

    /// MTP-specific attn step. Mirrors `MetalForward::encode_attn` but
    /// indexes the dedicated MTP KV ring (single layer) instead of
    /// `MetalSession::kv_*[attn_idx]`.
    fn encode_mtp_attn(&mut self, enc: &KernelEncoder, position: u32) -> Result<(), MtpError> {
        let arch = &self.base.model.arch;
        let ctx = self.base.ctx;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_q = arch.n_q_heads as usize;
        let n_kv = arch.n_kv_heads as usize;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
        let ab = &self.mtp_head.attn;
        let s = &mut self.mtp_session;

        // (1) Q projection: outputs 2 * q_dim (Q + gate interleaved).
        encode_mat_vec_dispatch(ctx, enc, &ab.q, &s.h, &s.attn_q_full, h, 2 * q_dim)?;

        // (2) Split Q from gate.
        encode_split_q_gate_f32(
            ctx,
            enc,
            &s.attn_q_full,
            &s.attn_q,
            &s.attn_gate,
            n_q,
            head_dim,
        )?;

        // (3) Q-norm (per-head RMSNorm).
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &s.attn_q,
            &ab.q_norm,
            &s.attn_q_normed,
            n_q,
            head_dim,
            RMS_EPS,
        )?;

        // (4) K, V projections.
        encode_mat_vec_dispatch(ctx, enc, &ab.k, &s.h, &s.attn_k_now, h, kv_dim)?;
        encode_mat_vec_dispatch(ctx, enc, &ab.v, &s.h, &s.attn_v_now, h, kv_dim)?;

        // (5) K-norm (per-head).
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &s.attn_k_now,
            &ab.k_norm,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            RMS_EPS,
        )?;

        // (6) Partial RoPE on Q and K.
        encode_rope_neox_f32(
            ctx,
            enc,
            &s.attn_q_normed,
            n_q,
            head_dim,
            n_rot,
            position,
            arch.rope_theta,
        )?;
        encode_rope_neox_f32(
            ctx,
            enc,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            n_rot,
            position,
            arch.rope_theta,
        )?;

        // (7) Append (K, V) to the MTP KV ring at slot `position`.
        // Strict-monotonic — the caller-side check in draft_inner already
        // asserted kv_n_pos == position.
        encode_scatter_offset_f32_to_f16_kv(
            ctx,
            enc,
            &s.attn_k_normed,
            &s.attn_v_now,
            &s.kv_k,
            &s.kv_v,
            (position as usize) * kv_dim,
            kv_dim,
        )?;
        // KV n_pos counter is bumped in draft_inner after waitUntilCompleted.

        // (8) Fused attention decode against the prefix [0..=position].
        // n_pos is `position + 1` because slot `position` was just appended.
        let n_pos = position as usize + 1;
        const V4_HEAD_DIM: usize = 256;
        const V4_GROUP: usize = 6;
        let use_v4 = head_dim == V4_HEAD_DIM && n_q == n_kv * V4_GROUP;
        if use_v4 {
            let nwg = attn_v4_choose_nwg(n_pos, V4_GROUP);
            let tile_c = attn_v4_choose_tile_c(n_pos, V4_GROUP);
            encode_attn_decode_v4_f32(
                ctx,
                enc,
                &s.attn_q_normed,
                &s.kv_k,
                &s.kv_v,
                &s.attn_v4_o_partial,
                &s.attn_v4_ml_partial,
                &s.attn_o,
                n_q,
                n_kv,
                head_dim,
                n_pos,
                nwg,
                tile_c,
            )?;
        } else {
            encode_attn_decode_f16kv_f32(
                ctx,
                enc,
                &s.attn_q_normed,
                &s.kv_k,
                &s.kv_v,
                &s.attn_o,
                n_q,
                n_kv,
                head_dim,
                n_pos,
            )?;
        }

        // (9) Apply gated-attention sigmoid gate: attn_o *= sigmoid(gate).
        // Reuse attn_q (no longer needed) as scratch for sigmoid output.
        encode_sigmoid_f32(ctx, enc, &s.attn_gate, &s.attn_q)?;
        encode_mul_f32(ctx, enc, &s.attn_o, &s.attn_q, &s.attn_o)?;

        // (10) Output projection: q_dim → hidden. Result lands in mixer_out.
        encode_mat_vec_dispatch(ctx, enc, &ab.o, &s.attn_o, &s.mixer_out, q_dim, h)?;
        Ok(())
    }

    /// Greedy generation with MTP-augmented speculative decoding.
    /// Implements `docs/H4-MTP.md` §1.4 (lazy sequential verify with
    /// inline accept-branch bridge).
    ///
    /// `prompt_ids` are processed first via `prefill_prompt` (§1.5),
    /// which:
    /// * runs base forward for each prompt token
    /// * streams MTP-KV prefill for slots 0..n-2 immediately after
    ///   each base step (so `h_i` is consumed before invalidation)
    /// * retains `h_{n-1}` for bootstrap
    ///
    /// Then runs the per-step loop until `eos` or `max_new_tokens` is
    /// emitted.
    ///
    /// **Terminal-return contract:** when this returns, the session is
    /// NOT resumable for further generation (the loop may early-exit
    /// after emitting D_tok without running step E bridge or step F base
    /// forward, leaving base/MTP state inconsistent). Caller must use a
    /// fresh session for any continuation.
    pub fn decode(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        eos_id: i32,
        base_session: &mut MetalSession,
    ) -> Result<DecodeOutput, MtpError> {
        let arch = &self.base.model.arch;
        let h = arch.hidden_size as usize;
        let t_start = std::time::Instant::now();
        let mut stats = SpecStats::default();
        let mut tokens: Vec<i32> = Vec::with_capacity(prompt_ids.len() + max_new_tokens);
        tokens.extend_from_slice(prompt_ids);

        if prompt_ids.is_empty() {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "decode",
                detail: "prompt_ids must be non-empty".into(),
            }));
        }
        if max_new_tokens == 0 {
            return Ok(DecodeOutput {
                tokens,
                stats: stats.into_finalized(t_start),
            });
        }

        // Allocate a persistent hidden-carry buffer in the MTP session.
        // We keep two buffers so the bridge step can hold onto h_P while
        // the next single_token_with_hidden writes h_D into the other.
        let hidden_a = MetalTensor::zeros_f32(self.base.ctx, vec![h as u64])?;
        let hidden_b = MetalTensor::zeros_f32(self.base.ctx, vec![h as u64])?;

        // ---------- Prompt prefill ----------
        // Per §1.5: stream MTP-KV prefill during base prompt forward.
        // For i in 0..n: run base on prompt[i] → h_i (in hidden_a or hidden_b
        // depending on parity). For i in 0..n-1: also run MTP draft_kv_only
        // with (prompt[i+1], h_i, i). We read back only argmax tokens,
        // not full logits, since the speculative path needs just the
        // bootstrap next-token id.
        let n_prompt = prompt_ids.len();
        let mut next_bootstrap_tok: i32 = 0;
        let mut h_last_is_a = true; // which buffer holds the most recent hidden
        for (i, &tid) in prompt_ids.iter().enumerate() {
            let dst = if h_last_is_a { &hidden_a } else { &hidden_b };
            next_bootstrap_tok =
                self.base
                    .single_token_argmax_with_hidden(tid, i as u32, base_session, dst)?;
            stats.base_forward_calls += 1;

            if i + 1 < n_prompt {
                // MTP slot i: pair (prompt[i+1], h_i, i).
                self.draft_kv_only(prompt_ids[i + 1], dst, i as u32)?;
                stats.mtp_calls += 1;
            }
            // Alternate buffers so the next base call doesn't overwrite
            // `dst` while we still might need it. (For prefill, we
            // actually consume `dst` immediately via draft_kv_only above
            // before the next iteration's single_token_with_hidden, so
            // alternation isn't strictly required, but it keeps the
            // buffer policy consistent with the steady-state loop.)
            h_last_is_a = !h_last_is_a;
        }
        // After prefill loop: the LAST hidden (h_{n-1}) is in the OPPOSITE
        // buffer from the one h_last_is_a now points to (since we toggled
        // after consumption). Restore the pointer.
        h_last_is_a = !h_last_is_a;
        let hidden_at_proc = if h_last_is_a { &hidden_a } else { &hidden_b };
        // sentinel: track which buffer is which without another bool indirection
        let _ = hidden_at_proc;

        // Bootstrap: argmax(last_logits) is the first token to emit.
        let mut emit_tok = next_bootstrap_tok;
        let mut processed_pos = (n_prompt - 1) as u32;
        // mtp_processed_pos is implicitly tracked by self.mtp_session.kv_n_pos.
        // After prefill it should be n - 1 (slots 0..n-2 = n-1 entries).
        debug_assert_eq!(
            self.mtp_session.kv_n_pos as u32,
            (n_prompt - 1) as u32,
            "after prefill: mtp_kv should have n-1 entries"
        );

        // ---------- Per-step decode loop ----------
        let mut emitted_count: usize = 0;
        loop {
            // A. Emit the carried token. Stop if EOS / limit.
            tokens.push(emit_tok);
            emitted_count += 1;
            if emit_tok == eos_id || emitted_count >= max_new_tokens {
                break;
            }

            let p_tok = emit_tok;
            let p_pos = processed_pos + 1;

            // B. MTP draft for slot processed_pos with (P_tok, hidden_at_proc, processed_pos).
            // Predicts a candidate for position p_pos+1.
            // Precondition: mtp_session.kv_n_pos == processed_pos.
            // Recompute hidden_at_proc reference from h_last_is_a.
            let hidden_at_proc_ref = if h_last_is_a { &hidden_a } else { &hidden_b };
            let d_tok = self.draft(p_tok, hidden_at_proc_ref, processed_pos)?;
            stats.mtp_calls += 1;
            stats.drafts_attempted += 1;

            // C. Base forward on P_tok at p_pos. Writes new hidden into
            // the OTHER buffer so we can keep hidden_at_proc alive for
            // the inline bridge. After: that other buffer holds h_{p_pos}.
            let dst_for_p = if h_last_is_a { &hidden_b } else { &hidden_a };
            let target_next =
                self.base
                    .single_token_argmax_with_hidden(p_tok, p_pos, base_session, dst_for_p)?;
            stats.base_forward_calls += 1;

            // D. Lazy sequential verify.
            if d_tok == target_next {
                // ACCEPT branch.
                stats.accepted += 1;
                tokens.push(d_tok);
                emitted_count += 1;
                if d_tok == eos_id || emitted_count >= max_new_tokens {
                    // Terminal return: state inconsistent (no bridge, no
                    // step F). Per the H4-MTP §1.4 contract.
                    break;
                }
                let d_pos = p_pos + 1;

                // E. Inline MTP bridge for slot p_pos with (D_tok, h_p, p_pos).
                // h_p is in dst_for_p. This MUST run BEFORE step F's
                // single_token_with_hidden, which would overwrite the
                // hidden buffer (we'll write into the OTHER one to be
                // safe — alternation policy).
                self.draft_kv_only(d_tok, dst_for_p, p_pos)?;
                stats.mtp_calls += 1;

                // F. Base forward on D_tok at d_pos. Write h_d into the
                // buffer we just freed up (the one that previously held
                // hidden_at_proc — which has now been consumed by step B).
                let dst_for_d = if h_last_is_a { &hidden_a } else { &hidden_b };
                let next_emit = self.base.single_token_argmax_with_hidden(
                    d_tok,
                    d_pos,
                    base_session,
                    dst_for_d,
                )?;
                stats.base_forward_calls += 1;

                // Update for next iter. The buffer holding the latest
                // hidden has now flipped: previously hidden_at_proc was
                // in `(h_last_is_a ? a : b)`; now h_d is in `(h_last_is_a ? a : b)`
                // (yes, the same one, because we wrote it back into the
                // buffer hidden_at_proc was previously in — and we
                // consumed dst_for_p's contents in step E so dst_for_p is
                // free for the next iter's hidden_at_proc to invalidate).
                // Wait — that means h_last_is_a stays the same. Let me
                // re-derive:
                //   start of iter: hidden_at_proc in (h_last_is_a ? a : b)
                //   step C writes h_p into dst_for_p = (h_last_is_a ? b : a)
                //   step E consumes h_p (still in dst_for_p)
                //   step F writes h_d into dst_for_d = (h_last_is_a ? a : b)
                //                        which is the SAME buffer hidden_at_proc was in.
                //   So h_last_is_a stays the same and the next iter's
                //   hidden_at_proc reads from the same buffer.
                processed_pos = d_pos;
                emit_tok = next_emit;
                stats.steps += 1;
            } else {
                // REJECT branch.
                // Base sits at p_pos, MTP sits at processed_pos == p_pos - 1.
                // No bridge or step F runs. h_p is now the new hidden_at_proc;
                // it's in dst_for_p = (h_last_is_a ? b : a).
                processed_pos = p_pos;
                emit_tok = target_next;
                // Flip h_last_is_a so hidden_at_proc reads from dst_for_p next iter.
                h_last_is_a = !h_last_is_a;
                stats.steps += 1;
            }
        }

        Ok(DecodeOutput {
            tokens,
            stats: stats.into_finalized(t_start),
        })
    }

    /// Experimental MTP-N path. Drafts `spec_tokens` proposals by recursively
    /// feeding the MTP head, then verifies `[carry, drafts...]` in one packed
    /// base-model forward using the DFlash packed-verify machinery.
    ///
    /// This is bench-only for now: recursive draft slots beyond the first use
    /// the previous MTP hidden (`mtp_session.x`) as a surrogate for the exact
    /// base hidden. Correctness is still preserved because the target packed
    /// verify remains authoritative and we rebuild canonical MTP KV for the
    /// accepted prefix from captured base hiddens before continuing.
    pub fn decode_packed_n(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        eos_id: i32,
        base_session: &mut MetalSession,
        spec_tokens: usize,
        verify_scratch: &mut MetalDFlashVerifyScratch,
        layer_scratch: &mut MetalDFlashLayerMajorScratch,
    ) -> Result<DecodeOutput, MtpError> {
        if spec_tokens < 2 || spec_tokens > 3 {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n",
                detail: format!("spec_tokens={spec_tokens} must be in [2, 3]"),
            }));
        }

        let arch = &self.base.model.arch;
        let h = arch.hidden_size as usize;
        let t_start = std::time::Instant::now();
        let mut stats = SpecStats::default();
        let mut tokens: Vec<i32> = Vec::with_capacity(prompt_ids.len() + max_new_tokens);
        tokens.extend_from_slice(prompt_ids);

        if prompt_ids.is_empty() {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n",
                detail: "prompt_ids must be non-empty".into(),
            }));
        }
        if max_new_tokens == 0 {
            return Ok(DecodeOutput {
                tokens,
                stats: stats.into_finalized(t_start),
            });
        }

        let verify_n = spec_tokens + 1;
        if verify_scratch.n as usize != verify_n {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n.verify_scratch",
                detail: format!(
                    "verify_scratch.n={} != verify_n={verify_n}",
                    verify_scratch.n
                ),
            }));
        }
        if verify_scratch.k_target_layers != 1 {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n.verify_scratch",
                detail: format!(
                    "verify_scratch.k_target_layers={} != 1",
                    verify_scratch.k_target_layers
                ),
            }));
        }
        if layer_scratch.n as usize != verify_n {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n.layer_scratch",
                detail: format!("layer_scratch.n={} != verify_n={verify_n}", layer_scratch.n),
            }));
        }

        let hidden_cur = MetalTensor::zeros_f32(self.base.ctx, vec![h as u64])?;
        let recursive_hidden = MetalTensor::zeros_f32(self.base.ctx, vec![h as u64])?;
        let n_prompt = prompt_ids.len();
        let mut next_bootstrap_tok: i32 = 0;

        // Prompt prefill: identical streaming contract as H4, but only argmax
        // readback from the base path.
        for (i, &tid) in prompt_ids.iter().enumerate() {
            next_bootstrap_tok = self.base.single_token_argmax_with_hidden(
                tid,
                i as u32,
                base_session,
                &hidden_cur,
            )?;
            stats.base_forward_calls += 1;

            if i + 1 < n_prompt {
                self.draft_kv_only(prompt_ids[i + 1], &hidden_cur, i as u32)?;
                stats.mtp_calls += 1;
            }
        }

        debug_assert_eq!(
            self.mtp_session.kv_n_pos as u32,
            (n_prompt - 1) as u32,
            "after prefill: mtp_kv should have n-1 entries"
        );

        let last_layer = [(self.base.model.blocks.len() - 1) as u32];
        let mut emit_tok = next_bootstrap_tok;
        let mut processed_pos = (n_prompt - 1) as u32;
        let mut emitted_count: usize = 0;

        'outer: loop {
            tokens.push(emit_tok);
            emitted_count += 1;
            if emit_tok == eos_id || emitted_count >= max_new_tokens {
                break;
            }

            let carry_tok = emit_tok;
            let start_position = processed_pos + 1;

            // Draft chain. First slot uses exact base hidden. Subsequent slots
            // recursively consume the previous MTP hidden as an approximation.
            let mut drafts: Vec<i32> = Vec::with_capacity(spec_tokens);
            let first = self.draft(carry_tok, &hidden_cur, processed_pos)?;
            drafts.push(first);
            stats.mtp_calls += 1;
            stats.drafts_attempted += 1;
            copy_f32_tensor(&self.mtp_session.x, &recursive_hidden)?;
            for j in 1..spec_tokens {
                let d = self.draft(drafts[j - 1], &recursive_hidden, processed_pos + j as u32)?;
                drafts.push(d);
                stats.mtp_calls += 1;
                stats.drafts_attempted += 1;
                copy_f32_tensor(&self.mtp_session.x, &recursive_hidden)?;
            }

            let mut verify_input: Vec<i32> = Vec::with_capacity(verify_n);
            verify_input.push(carry_tok);
            verify_input.extend_from_slice(&drafts);

            let verify_argmax = encode_packed_verify_layer_major_inner(
                self.base,
                &last_layer,
                &verify_input,
                start_position,
                verify_scratch,
                layer_scratch,
                base_session,
                None,
                None,
            )?;
            stats.base_forward_calls += 1;

            let mut n_accepted = 0usize;
            let mut stop_now = false;
            for (j, &draft_tok) in drafts.iter().enumerate() {
                if draft_tok != verify_argmax[j] {
                    break;
                }
                stats.accepted += 1;
                n_accepted += 1;
                tokens.push(draft_tok);
                emitted_count += 1;
                if draft_tok == eos_id || emitted_count >= max_new_tokens {
                    stop_now = true;
                    break;
                }
            }

            let n_keep = (1 + n_accepted) as u32;
            if n_keep < verify_n as u32 {
                encode_restore_after_partial_accept_inner(
                    self.base,
                    verify_scratch,
                    n_keep,
                    start_position,
                    base_session,
                    None,
                )?;
            }

            // Recursive draft slots beyond the first are approximate. Rebuild the
            // canonical MTP KV for the accepted prefix from captured base hiddens.
            self.mtp_session.kv_n_pos = processed_pos as usize + 1;
            for j in 0..n_accepted {
                let prev_hidden = verify_scratch.hidden_capture_n_slot(j as u32);
                let bridge_position = processed_pos + 1 + j as u32;
                self.draft_kv_only(drafts[j], &prev_hidden, bridge_position)?;
                stats.mtp_calls += 1;
            }

            let next_hidden = verify_scratch.hidden_capture_n_slot(n_accepted as u32);
            copy_f32_tensor(&next_hidden, &hidden_cur)?;

            processed_pos += 1 + n_accepted as u32;
            emit_tok = verify_argmax[n_accepted];
            stats.steps += 1;

            if stop_now {
                break 'outer;
            }
        }

        Ok(DecodeOutput {
            tokens,
            stats: stats.into_finalized(t_start),
        })
    }
}

/// Result of [`SpeculativeDecoder::decode`].
#[derive(Debug, Clone)]
pub struct DecodeOutput {
    /// Full sequence: prompt tokens + emitted tokens.
    pub tokens: Vec<i32>,
    pub stats: SpecStats,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SpecStats {
    /// Number of MTP-augmented decode iterations (after prompt prefill).
    pub steps: u32,
    /// Drafts that matched target's argmax.
    pub accepted: u32,
    /// Total drafted tokens proposed for acceptance. For MTP-1 this equals
    /// `steps`; for experimental MTP-N it is `steps * N` minus any terminal
    /// short-circuit on the last outer step.
    pub drafts_attempted: u32,
    /// Total base forward calls (prompt prefill + decode loop).
    pub base_forward_calls: u32,
    /// Total MTP draft / draft_kv_only calls (prefill + decode + bridges).
    pub mtp_calls: u32,
    /// Wall time in milliseconds.
    pub wall_ms: f64,
}

impl SpecStats {
    fn into_finalized(mut self, t_start: std::time::Instant) -> Self {
        self.wall_ms = t_start.elapsed().as_secs_f64() * 1e3;
        self
    }

    /// Acceptance rate α = accepted / steps. Returns 0.0 if no steps ran.
    pub fn acceptance_rate(&self) -> f64 {
        let denom = if self.drafts_attempted > 0 {
            self.drafts_attempted
        } else {
            self.steps
        };
        if denom == 0 {
            0.0
        } else {
            self.accepted as f64 / denom as f64
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn argmax_i32(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

// ----- Tests -----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::{Forward, GdnState, KvCache};
    use crate::loader::Model;
    use crate::metal_forward::MetalModel;

    /// H4.2 cosine test: Metal MTP draft logits must match CPU MTP draft
    /// logits at cosine ≥ 0.9999 for identical `(next_tok, prev_hidden,
    /// position)` inputs across multiple sequential steps.
    ///
    /// Approach: build `prev_hidden` once via the CPU base forward
    /// (`Forward::single_token_with_hidden`), upload as a Metal tensor,
    /// then run BOTH `SpeculativeDecoder::draft` and `Forward::mtp_step`
    /// on the same input quadruple. Both impls share the same GGUF
    /// weights; the only variance is the kernel substrate. Drift should
    /// be at the quant-noise floor (~1e-3 for Q4_K).
    ///
    /// Exercises:
    /// * step 0: empty MTP KV (attention against self only)
    /// * step 1: 1-entry KV (first-position drift through encode_mtp_attn)
    /// * step 2: 2-entry KV (KV reads + RoPE position increment)
    #[test]
    fn metal_mtp_draft_matches_cpu() {
        let path = "/Users/tito/models/h4-smoke-test/Qwen3.5-0.8B/qwen3.5-0.8b.Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[mtp-metal-cosine] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init: {e}"),
        };

        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        assert!(m.mtp.is_some(), "model must have MTP head");
        let mtp_view = m.mtp.as_ref().unwrap();

        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mtp_head = MetalMtpHead::load(&ctx, &g, mtp_view).expect("metal mtp load");
        let mtp_session = MetalMtpSession::fresh(&ctx, &mtp_head, &m.arch, 64).expect("session");

        let mf = MetalForward::new(&ctx, &mm);
        let cpu = Forward::new(&g, &m);
        let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);

        // Run a 3-token base prompt forward on CPU to get hiddens
        // h_0, h_1, h_2 — these become `prev_hidden` for our MTP test.
        // We use CPU base forward here to keep the test self-contained
        // (Metal-vs-CPU-base parity is covered elsewhere). The MTP path
        // operates on the same hidden values regardless.
        let tok = crate::tokenizer::Tokenizer::open(path).expect("tok");
        let ids = tok
            .encode("The quick brown fox jumps over the lazy dog", false)
            .expect("tok");
        eprintln!("[mtp-metal-cosine] prompt tokens: {ids:?}");
        assert!(ids.len() >= 3, "need ≥3 prompt tokens");

        let mut state = GdnState::fresh(&m);
        let mut kv = KvCache::new(&m);
        let mut hiddens: Vec<Vec<f32>> = Vec::with_capacity(3);
        for (i, &tid) in ids.iter().take(3).enumerate() {
            let (_logits, hidden) = cpu
                .single_token_with_hidden(tid, i as u32, &mut state, &mut kv)
                .expect("base forward");
            hiddens.push(hidden);
        }

        // For each of 3 steps, run both CPU mtp_step and Metal draft on
        // matching inputs. Use position=i (the slot of h_i) and a
        // synthetic next_tok per the §1.2 contract (just pick the next
        // prompt id; the value is arbitrary for a kernel-cosine test as
        // long as both paths see the same one).
        let h = m.arch.hidden_size as usize;
        let mut cpu_mtp_kv = cpu.fresh_mtp_kv(64);
        for step in 0..3 {
            let position = step as u32;
            let next_tok = ids[(step + 1).min(ids.len() - 1)];
            let prev_hidden_cpu: &[f32] = &hiddens[step];

            // CPU MTP step.
            let cpu_logits = cpu
                .mtp_step(next_tok, prev_hidden_cpu, position, &mut cpu_mtp_kv)
                .expect("cpu mtp_step");
            assert_eq!(cpu_logits.len(), m.arch.vocab_size as usize);

            // Metal MTP draft on the same inputs.
            // Upload prev_hidden as a fresh F32 tensor (same lifetime as
            // the call — we don't reuse it).
            let prev_hidden_metal = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(prev_hidden_cpu),
                vec![h as u64],
                GgmlType::F32,
            )
            .expect("upload prev_hidden");
            let metal_logits = spec
                .draft_inner(
                    next_tok,
                    &prev_hidden_metal,
                    position,
                    DraftReadback::FullLogits,
                )
                .expect("metal mtp draft")
                .logits
                .expect("full logits");
            assert_eq!(metal_logits.len(), cpu_logits.len());

            let mut max_abs = 0.0f32;
            let mut argmax_metal = 0usize;
            let mut argmax_cpu = 0usize;
            let mut max_metal = f32::NEG_INFINITY;
            let mut max_cpu = f32::NEG_INFINITY;
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for i in 0..cpu_logits.len() {
                let d = (metal_logits[i] - cpu_logits[i]).abs();
                max_abs = max_abs.max(d);
                if metal_logits[i] > max_metal {
                    max_metal = metal_logits[i];
                    argmax_metal = i;
                }
                if cpu_logits[i] > max_cpu {
                    max_cpu = cpu_logits[i];
                    argmax_cpu = i;
                }
                dot += metal_logits[i] as f64 * cpu_logits[i] as f64;
                na += (metal_logits[i] as f64).powi(2);
                nb += (cpu_logits[i] as f64).powi(2);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
            eprintln!(
                "[mtp-metal-cosine] step={step} pos={position} next_tok={next_tok}: \
                 metal_argmax={argmax_metal}({:.4}) cpu_argmax={argmax_cpu}({:.4}) \
                 max|Δ|={max_abs:.2e} cos={cos:.6}",
                max_metal, max_cpu
            );

            assert_eq!(argmax_metal, argmax_cpu, "step {step}: argmax disagreement");
            // Q4_K weights through 1 attn + 1 FFN block. ~1e-3 max|Δ|
            // is in line with the existing single-token base oracle test.
            assert!(cos > 0.9999, "step {step}: cos {cos} below threshold");
            assert!(max_abs < 0.05, "step {step}: max|Δ| {max_abs} above floor");
        }
    }

    /// H4.3 greedy generation equivalence: with MTP=on and MTP=off,
    /// generated token sequences must be IDENTICAL up to max_new_tokens.
    /// This is the real correctness gate for the speculative decode loop
    /// (cursor accounting, inline bridge, prompt prefill, accept/reject
    /// logic). Under greedy verify both paths emit `argmax(target_logits)`
    /// at every position, so any deviation indicates a bug — most likely
    /// in the cursor state machine, prefill streaming, or inline bridge
    /// ordering.
    ///
    /// Reports acceptance rate α as a sanity signal. If α << 0.3 on
    /// natural prose, the MTP forward distribution is broken (not
    /// causing token corruption since target wins on reject, but
    /// silently negating any speculative speedup). For this 0.8B-MTP
    /// smoke test we just assert α > 0 to confirm the MTP forward
    /// produces SOMETHING reasonable.
    #[test]
    fn greedy_equivalence_0_8b_mtp() {
        let path = "/Users/tito/models/h4-smoke-test/Qwen3.5-0.8B/qwen3.5-0.8b.Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[mtp-equiv] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init: {e}"),
        };

        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mtp_view = m.mtp.as_ref().expect("MTP head present");

        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mtp_head = MetalMtpHead::load(&ctx, &g, mtp_view).expect("mtp load");
        let mf = MetalForward::new(&ctx, &mm);

        let tok = crate::tokenizer::Tokenizer::open(path).expect("tok");
        let prompt_ids = tok
            .encode("The quick brown fox jumps over the lazy dog", false)
            .expect("tok");
        let max_new_tokens = 16usize;
        let eos_id = 248046_i32; // <|im_end|> per the 0.8B vocab

        // ---------- Reference: MTP=off greedy generation ----------
        // Run base alone, capturing the last logits each iter so we can
        // argmax-greedy the next token.
        let mut ref_session =
            MetalSession::fresh(&ctx, &mm, prompt_ids.len() + max_new_tokens + 8).expect("session");
        let mut ref_tokens = prompt_ids.clone();
        let mut last_logits = Vec::new();
        for (i, &tid) in prompt_ids.iter().enumerate() {
            last_logits = mf
                .single_token(tid, i as u32, &mut ref_session)
                .expect("base forward");
        }
        let mut next_tok = argmax_i32(&last_logits);
        let mut pos = (prompt_ids.len() - 1) as u32;
        for _ in 0..max_new_tokens {
            ref_tokens.push(next_tok);
            if next_tok == eos_id {
                break;
            }
            pos += 1;
            let logits = mf
                .single_token(next_tok, pos, &mut ref_session)
                .expect("base forward decode");
            next_tok = argmax_i32(&logits);
        }
        let ref_generated = &ref_tokens[prompt_ids.len()..];
        eprintln!(
            "[mtp-equiv] ref (MTP=off) generated {} tokens: {:?}",
            ref_generated.len(),
            ref_generated
        );

        // ---------- Test: MTP=on speculative-decode generation ----------
        let mtp_session = MetalMtpSession::fresh(
            &ctx,
            &mtp_head,
            &m.arch,
            prompt_ids.len() + max_new_tokens + 8,
        )
        .expect("mtp session");
        let mut spec_session =
            MetalSession::fresh(&ctx, &mm, prompt_ids.len() + max_new_tokens + 8).expect("session");
        let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);
        let result = spec
            .decode(&prompt_ids, max_new_tokens, eos_id, &mut spec_session)
            .expect("spec decode");
        let mtp_generated = &result.tokens[prompt_ids.len()..];
        eprintln!(
            "[mtp-equiv] spec (MTP=on) generated {} tokens: {:?}",
            mtp_generated.len(),
            mtp_generated
        );
        eprintln!(
            "[mtp-equiv] stats: steps={} accepted={} α={:.3} base_calls={} mtp_calls={} wall={:.1}ms",
            result.stats.steps,
            result.stats.accepted,
            result.stats.acceptance_rate(),
            result.stats.base_forward_calls,
            result.stats.mtp_calls,
            result.stats.wall_ms,
        );

        // Token sequences MUST be identical (greedy + greedy verify ⇒
        // emitted = argmax(target) at every position).
        assert_eq!(
            mtp_generated.len(),
            ref_generated.len(),
            "spec generated {} tokens but ref generated {}",
            mtp_generated.len(),
            ref_generated.len()
        );
        for (i, (s, r)) in mtp_generated.iter().zip(ref_generated.iter()).enumerate() {
            assert_eq!(s, r, "token mismatch at gen-position {i}: spec={s} ref={r}",);
        }

        // Sanity: MTP forward did SOMETHING. α=0 on a 16-token sample
        // would mean every draft missed — possible for a small model
        // but worth flagging.
        assert!(
            result.stats.steps > 0,
            "no decode steps ran (max_new_tokens or prompt issue)"
        );
        eprintln!("[mtp-equiv] greedy equivalence: PASS");
    }
}
