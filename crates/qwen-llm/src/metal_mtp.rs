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
    attn_v4_choose_nwg, attn_v4_choose_tile_c, encode_add_inplace_f32,
    encode_attn_decode_f16kv_f32, encode_attn_decode_v4_f32, encode_ffn_swiglu_q4_K_f32,
    encode_get_rows_f32, encode_mul_f32, encode_rms_norm_batched_f32, encode_rms_norm_mul_f32,
    encode_rope_neox_f32, encode_scatter_offset_f32_to_f16_kv, encode_sigmoid_f32,
    encode_silu_mul_f32, encode_split_q_gate_f32, KernelEncoder, MetalContext, MetalError,
    MetalTensor,
};
use crate::metal_forward::{
    encode_mat_vec_dispatch, encode_scatter_offset_f32, weight_dtype_kept_native, MetalAttnBlock,
    MetalForward, ATTN_V4_MAX_NWG, RMS_EPS,
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
    pub logits: MetalTensor,  // [V]
    pub ids_buf: MetalTensor, // i32 token id (in F32 buffer)
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
            ids_buf: MetalTensor::zeros_f32(ctx, vec![1])?,
        })
    }
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
        let logits =
            self.draft_inner(next_tok, prev_hidden, position, /*want_logits=*/ true)?;
        // Greedy argmax.
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as i32;
        Ok(argmax)
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
        let _ = self.draft_inner(next_tok, prev_hidden, position, /*want_logits=*/ false)?;
        Ok(())
    }

    /// Draft + (optionally) read back logits. Internal helper.
    fn draft_inner(
        &mut self,
        next_tok: i32,
        prev_hidden: &MetalTensor,
        position: u32,
        want_logits: bool,
    ) -> Result<Vec<f32>, MtpError> {
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

        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        // KV side-effect was committed by encode_mtp_attn; bump our counter.
        self.mtp_session.kv_n_pos = position as usize + 1;

        if want_logits {
            let mut out = vec![0.0f32; arch.vocab_size as usize];
            unsafe {
                let src = self.mtp_session.logits.buffer.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
            }
            Ok(out)
        } else {
            Ok(Vec::new())
        }
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
            let nwg = attn_v4_choose_nwg(n_pos);
            let tile_c = attn_v4_choose_tile_c(n_pos);
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
                .draft_inner(next_tok, &prev_hidden_metal, position, true)
                .expect("metal mtp draft");
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
}
