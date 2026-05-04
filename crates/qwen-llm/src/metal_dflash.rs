//! Metal DFlash drafter and speculative-decode driver.
//!
//! Mirrors `crate::forward::Forward::dflash_draft` (CPU oracle) on Metal.
//! See `docs/H5-DFLASH.md` §1.3 for the algorithmic contract.
//!
//! **v1 posture (correctness-first; perf is H5.3+).**  This module
//! produces bit-tight cosine vs the CPU oracle but is intentionally
//! NOT a perf path. v1 dispatches Metal-resident kernels for projections
//! (mat-vec), RMSNorms, and RoPE — the bulk of the BW-bound work — and
//! falls back to CPU for the asymmetric SWA-masked attention and the
//! SwiGLU `silu(gate) * up` reduction. The packed mat-mat path (H5.3)
//! removes both fallbacks.
//!
//! Why CPU attention/FFN-mid in v1:
//! * The drafter's attention is asymmetric (Q from N noise tokens; K/V
//!   from `concat(target_ctx, noise)`) with a per-layer SWA mask.
//!   Existing flash-attn-v4 doesn't support this shape.
//! * SwiGLU silu_mul wants two `[F]`-sized buffers (gate + up) for the
//!   per-row reduction; a clean fix is `[N, F]` session scratch for
//!   each, but it adds 2 · N · F floats of GPU memory (~2 MB). Worth
//!   doing for H5.3 perf; not yet.
//!
//! H5.1.5 validates this v1 against the CPU oracle (cosine ≥ 0.9999
//! per noise-position logits). If that gate passes, we know the
//! Metal-resident kernels (projections + norms + RoPE) are correct. The
//! CPU fallback steps are identical bytes between Metal and CPU paths,
//! so they don't bias the cosine.

use crate::codec::dequant_to_f32;
use crate::gguf::GgufFile;
use crate::loader::{DFlashHead, DFlashLayer};
use crate::metal::{
    encode_add_inplace_f32, encode_get_rows_f32, encode_rms_norm_batched_f32,
    encode_rms_norm_mul_f32, encode_rope_neox_f32, KernelEncoder, MetalContext, MetalError,
    MetalTensor,
};
use crate::metal_forward::{
    encode_mat_vec_dispatch, encode_scatter_offset_f32, weight_dtype_kept_native, MetalForward,
    RMS_EPS,
};
use crate::tensor::{GgmlType, TensorDesc};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};

#[derive(Debug, thiserror::Error)]
pub enum DFlashError {
    #[error("metal: {0}")]
    Metal(#[from] MetalError),
    #[error("metal forward: {0}")]
    MetalForward(#[from] crate::metal_forward::MfError),
    #[error("codec: {0}")]
    Codec(#[from] crate::codec::CodecError),
    #[error("token {0} out of vocab range {1}")]
    BadToken(i32, u32),
    #[error("ctx_len {0} > capacity {1}")]
    CtxOverflow(usize, usize),
}

/// All DFlash drafter weights resident on Metal. Loaded once at session
/// start; tensor shapes mirror `crate::loader::DFlashHead`.
///
/// Drafter weight TensorDescs index into the DRAFTER GGUF mmap, never
/// the target's. The H5.1 cross-GGUF dequant bug taught the lesson —
/// `MetalDFlashHead::load` takes the drafter GGUF explicitly and never
/// touches `Forward::dequant` (which is target-scoped). All drafter
/// dequants happen at load time; per-call kernels read from the
/// pre-allocated `MetalTensor` buffers.
pub struct MetalDFlashHead {
    pub config: crate::loader::DFlashConfig,
    pub target_layer_ids: Vec<u32>,
    pub fc: MetalTensor,
    pub hidden_norm: MetalTensor,
    pub output_norm: MetalTensor,
    pub layers: Vec<MetalDFlashLayer>,
}

pub struct MetalDFlashLayer {
    pub attn_norm: MetalTensor,
    pub q: MetalTensor,
    pub k: MetalTensor,
    pub v: MetalTensor,
    pub o: MetalTensor,
    pub q_norm: MetalTensor,
    pub k_norm: MetalTensor,
    pub post_attention_norm: MetalTensor,
    pub ffn_gate: MetalTensor,
    pub ffn_up: MetalTensor,
    pub ffn_down: MetalTensor,
    pub is_swa: bool,
}

impl MetalDFlashHead {
    pub fn load(
        ctx: &MetalContext,
        drafter_gguf: &GgufFile,
        head: &DFlashHead<'_>,
    ) -> Result<Self, DFlashError> {
        let load_f32 = |desc: &TensorDesc| -> Result<MetalTensor, DFlashError> {
            if desc.dtype == GgmlType::F32 {
                Ok(MetalTensor::from_gguf_tensor(
                    ctx,
                    desc,
                    drafter_gguf.slice(desc),
                )?)
            } else {
                let f32 = dequant_to_f32(desc, drafter_gguf.slice(desc))?;
                Ok(MetalTensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&f32),
                    desc.shape.clone(),
                    GgmlType::F32,
                )?)
            }
        };
        let load_weight = |desc: &TensorDesc| -> Result<MetalTensor, DFlashError> {
            if weight_dtype_kept_native(desc.dtype) {
                Ok(MetalTensor::from_gguf_tensor(
                    ctx,
                    desc,
                    drafter_gguf.slice(desc),
                )?)
            } else {
                load_f32(desc)
            }
        };

        let layers: Result<Vec<MetalDFlashLayer>, DFlashError> = head
            .layers
            .iter()
            .map(|l: &DFlashLayer| {
                Ok(MetalDFlashLayer {
                    attn_norm: load_f32(l.attn_norm)?,
                    q: load_weight(l.q)?,
                    k: load_weight(l.k)?,
                    v: load_weight(l.v)?,
                    o: load_weight(l.o)?,
                    q_norm: load_f32(l.q_norm)?,
                    k_norm: load_f32(l.k_norm)?,
                    post_attention_norm: load_f32(l.post_attention_norm)?,
                    ffn_gate: load_weight(l.ffn_gate)?,
                    ffn_up: load_weight(l.ffn_up)?,
                    ffn_down: load_weight(l.ffn_down)?,
                    is_swa: l.is_swa,
                })
            })
            .collect();

        Ok(Self {
            config: head.config,
            target_layer_ids: head.target_layer_ids.clone(),
            fc: load_weight(head.fc)?,
            hidden_norm: load_f32(head.hidden_norm)?,
            output_norm: load_f32(head.output_norm)?,
            layers: layers?,
        })
    }
}

/// Per-step DFlash session state.
pub struct MetalDFlashSession {
    /// Cross-context K-target-layer hiddens stacked: `[K · H_target, ctx_capacity]`,
    /// row-major. Each column holds K target hiddens at one committed
    /// sequence position.
    pub target_ctx_stacked: MetalTensor,
    pub target_ctx_n: usize,
    pub target_ctx_capacity: usize,

    /// Per-context-column absolute target sequence position, `[ctx_capacity]` i32
    /// stored in F32 buffer for layout consistency.
    pub pos_ctx: MetalTensor,

    /// Projected cross-context: `[ctx_capacity, H_drafter]` (row-major over
    /// columns). Recomputed per `draft_block` call from
    /// `target_ctx_stacked` via `dflash_fc + hidden_norm`.
    pub ctx_h: MetalTensor,

    /// Per-step block input `[N]` i32 in F32 buffer (carry + (N-1) MASK).
    pub noise_ids: MetalTensor,

    /// Drafter scratch — all `[N, ...]`-shaped.
    pub x: MetalTensor, // [N, H_drafter]
    pub h: MetalTensor,         // [N, H_drafter]
    pub q_buf: MetalTensor,     // [N, n_q · head_dim]
    pub k_noise: MetalTensor,   // [N, n_kv · head_dim]
    pub v_noise: MetalTensor,   // [N, n_kv · head_dim]
    pub k_ctx_buf: MetalTensor, // [ctx_capacity, n_kv · head_dim]
    pub v_ctx_buf: MetalTensor, // [ctx_capacity, n_kv · head_dim]
    pub attn_o: MetalTensor,    // [N, n_q · head_dim]
    pub mixer_out: MetalTensor, // [N, H_drafter] — attn or ffn output

    /// Final logits buffer for the entire noise block: `[N, V_target]`.
    pub draft_logits: MetalTensor,
}

impl MetalDFlashSession {
    pub fn fresh(
        ctx: &MetalContext,
        head: &MetalDFlashHead,
        target_h: u64,
        vocab: u64,
        ctx_capacity: usize,
    ) -> Result<Self, DFlashError> {
        let cfg = &head.config;
        let n = cfg.block_size as u64;
        let h = cfg.hidden_size as u64;
        let q_dim = (cfg.n_q_heads * cfg.head_dim) as u64;
        let kv_dim = (cfg.n_kv_heads * cfg.head_dim) as u64;
        let k_layers = head.target_layer_ids.len() as u64;
        let n_target_features = k_layers * target_h;
        let cc = ctx_capacity as u64;
        Ok(Self {
            target_ctx_stacked: MetalTensor::zeros_f32(ctx, vec![n_target_features * cc])?,
            target_ctx_n: 0,
            target_ctx_capacity: ctx_capacity,
            pos_ctx: MetalTensor::zeros_f32(ctx, vec![cc])?,
            ctx_h: MetalTensor::zeros_f32(ctx, vec![h * cc])?,
            noise_ids: MetalTensor::zeros_f32(ctx, vec![n])?,
            x: MetalTensor::zeros_f32(ctx, vec![n * h])?,
            h: MetalTensor::zeros_f32(ctx, vec![n * h])?,
            q_buf: MetalTensor::zeros_f32(ctx, vec![n * q_dim])?,
            k_noise: MetalTensor::zeros_f32(ctx, vec![n * kv_dim])?,
            v_noise: MetalTensor::zeros_f32(ctx, vec![n * kv_dim])?,
            k_ctx_buf: MetalTensor::zeros_f32(ctx, vec![cc * kv_dim])?,
            v_ctx_buf: MetalTensor::zeros_f32(ctx, vec![cc * kv_dim])?,
            attn_o: MetalTensor::zeros_f32(ctx, vec![n * q_dim])?,
            mixer_out: MetalTensor::zeros_f32(ctx, vec![n * h])?,
            draft_logits: MetalTensor::zeros_f32(ctx, vec![n * vocab])?,
        })
    }

    /// Append one column of K-target-layer hiddens at the next free slot.
    /// `hidden_block` is `[K · H_target]`. Caller obtains it from
    /// `MetalForward::single_token_with_multi_hidden`.
    pub fn append_target_ctx_column(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        hidden_block: &MetalTensor,
        position: u32,
        n_target_features: usize,
    ) -> Result<(), DFlashError> {
        if self.target_ctx_n >= self.target_ctx_capacity {
            return Err(DFlashError::CtxOverflow(
                self.target_ctx_n,
                self.target_ctx_capacity,
            ));
        }
        if hidden_block.n_elements() as usize != n_target_features {
            return Err(DFlashError::Metal(MetalError::BadShape {
                kernel: "append_target_ctx_column.hidden_block",
                detail: format!(
                    "expected {n_target_features} elements, got {}",
                    hidden_block.n_elements()
                ),
            }));
        }
        let dst_off = self.target_ctx_n * n_target_features;
        encode_scatter_offset_f32(
            ctx,
            enc,
            hidden_block,
            &self.target_ctx_stacked,
            dst_off,
            n_target_features,
        )?;
        unsafe {
            let ptr = self.pos_ctx.buffer.contents().as_ptr() as *mut i32;
            *ptr.add(self.target_ctx_n) = position as i32;
        }
        self.target_ctx_n += 1;
        Ok(())
    }
}

/// Top-level DFlash speculative-decode driver.
pub struct DFlashDecoder<'a> {
    pub base: &'a MetalForward<'a>,
    pub head: &'a MetalDFlashHead,
    pub session: MetalDFlashSession,
}

impl<'a> DFlashDecoder<'a> {
    pub fn new(
        base: &'a MetalForward<'a>,
        head: &'a MetalDFlashHead,
        session: MetalDFlashSession,
    ) -> Self {
        Self {
            base,
            head,
            session,
        }
    }

    /// Run drafter forward for one outer step. Produces `[N]` greedy
    /// argmax tokens. Position 0 of the returned vector is the carry
    /// seed's argmax (conventionally discarded); positions `1..N` are
    /// the `D = N-1` draft candidates.
    ///
    /// `noise_start_pos` = absolute sequence position of `carry_tok`
    /// (i.e., `processed_pos + 1`).
    pub fn draft_block(
        &mut self,
        carry_tok: i32,
        noise_start_pos: u32,
    ) -> Result<Vec<i32>, DFlashError> {
        let arch = &self.base.model.arch;
        let cfg = self.head.config;
        if carry_tok < 0 || (carry_tok as u32) >= arch.vocab_size {
            return Err(DFlashError::BadToken(carry_tok, arch.vocab_size));
        }
        let n = cfg.block_size as usize;
        let h = cfg.hidden_size as usize;
        let f = cfg.intermediate_size as usize;
        let head_dim = cfg.head_dim as usize;
        let n_q = cfg.n_q_heads as usize;
        let n_kv = cfg.n_kv_heads as usize;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let group = n_q / n_kv;
        let n_rot = head_dim;
        let theta = cfg.rope_theta;
        let ctx_len = self.session.target_ctx_n;
        let v = arch.vocab_size as usize;
        let k_layers = self.head.target_layer_ids.len();
        let n_target_features = k_layers * arch.hidden_size as usize;

        // Stage noise_ids: [carry_tok, MASK × (N-1)].
        unsafe {
            let ptr = self.session.noise_ids.buffer.contents().as_ptr() as *mut i32;
            *ptr = carry_tok;
            for i in 1..n {
                *ptr.add(i) = cfg.mask_token_id;
            }
        }

        let ctx_metal = self.base.ctx;

        // ----- Phase 1 (Metal): cross-context fc + hidden_norm -----
        // Per-column mat-vec into ctx_h, then per-column RMSNorm.
        if ctx_len > 0 {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for c in 0..ctx_len {
                let src = self.session.target_ctx_stacked.view_subrange(
                    (c * n_target_features) as u64,
                    vec![n_target_features as u64],
                );
                let dst = self
                    .session
                    .ctx_h
                    .view_subrange((c * h) as u64, vec![h as u64]);
                encode_mat_vec_dispatch(
                    ctx_metal,
                    &enc,
                    &self.head.fc,
                    &src,
                    &dst,
                    n_target_features,
                    h,
                )?;
            }
            // RMSNorm per column. Reads x then writes y in two passes per
            // threadgroup, so x==y aliasing is safe (rms_norm.metal:38-61).
            for c in 0..ctx_len {
                let view = self
                    .session
                    .ctx_h
                    .view_subrange((c * h) as u64, vec![h as u64]);
                encode_rms_norm_mul_f32(
                    ctx_metal,
                    &enc,
                    &view,
                    &self.head.hidden_norm,
                    &view,
                    RMS_EPS,
                )?;
            }
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        // ----- Phase 2 (Metal): noise embed + per-layer fwd through
        //     pre-attn-norm, Q/K/V projections, per-head Q/K-norm, RoPE.
        //     Then we read back to CPU for asymmetric SWA-masked
        //     attention + ffn (correctness first; H5.3 swaps to packed
        //     Metal).
        let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        encode_get_rows_f32(
            ctx_metal,
            &enc,
            &self.base.model.token_embd,
            &self.session.noise_ids,
            &self.session.x,
            n,
            h,
        )?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        // Read pos_ctx once (used by RoPE on K_ctx and SWA mask).
        let mut pos_ctx_cpu = vec![0i32; ctx_len];
        if ctx_len > 0 {
            unsafe {
                let src = self.session.pos_ctx.buffer.contents().as_ptr() as *const i32;
                std::ptr::copy_nonoverlapping(src, pos_ctx_cpu.as_mut_ptr(), ctx_len);
            }
        }

        // Borrow weights as F32 slices directly into the Metal shared-
        // storage buffer (no copy). All drafter weights are F32 post-load
        // since `MetalDFlashHead::load` dequants Q8_0 at load time. The
        // returned slice is valid for the lifetime of the draft_block
        // call (no concurrent writes).
        //
        // Codex partner session caught this: a previous version copied
        // ~1 GB of weights per layer per call into fresh Vecs; with 5
        // layers per draft and N=16 noise rows, that was ~5 GB of
        // pointless memcpy per outer step. Borrowing as &[f32] gives
        // the same data with zero copy.
        let borrow_f32_tensor = |t: &MetalTensor| -> &[f32] {
            debug_assert_eq!(
                t.dtype,
                GgmlType::F32,
                "v1 CPU fallback expects F32 weights"
            );
            let n_elem = t.n_elements() as usize;
            unsafe {
                let ptr = t.buffer.contents().as_ptr() as *const f32;
                std::slice::from_raw_parts(ptr, n_elem)
            }
        };
        // Activation readbacks (small per-call buffers; copy is fine).
        let read_f32_activation = |t: &MetalTensor, n_elem: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; n_elem];
            unsafe {
                let src = t.buffer.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n_elem);
            }
            out
        };

        for layer in &self.head.layers {
            // Pre-attn norm: x → h (Metal).
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_rms_norm_batched_f32(
                ctx_metal,
                &enc,
                &self.session.x,
                &layer.attn_norm,
                &self.session.h,
                n,
                h,
                RMS_EPS,
            )?;
            // Q proj per noise row.
            for i in 0..n {
                let row_in = self.session.h.view_subrange((i * h) as u64, vec![h as u64]);
                let row_out = self
                    .session
                    .q_buf
                    .view_subrange((i * q_dim) as u64, vec![q_dim as u64]);
                encode_mat_vec_dispatch(ctx_metal, &enc, &layer.q, &row_in, &row_out, h, q_dim)?;
            }
            // K, V proj on noise rows.
            for i in 0..n {
                let row_in = self.session.h.view_subrange((i * h) as u64, vec![h as u64]);
                let k_row = self
                    .session
                    .k_noise
                    .view_subrange((i * kv_dim) as u64, vec![kv_dim as u64]);
                let v_row = self
                    .session
                    .v_noise
                    .view_subrange((i * kv_dim) as u64, vec![kv_dim as u64]);
                encode_mat_vec_dispatch(ctx_metal, &enc, &layer.k, &row_in, &k_row, h, kv_dim)?;
                encode_mat_vec_dispatch(ctx_metal, &enc, &layer.v, &row_in, &v_row, h, kv_dim)?;
            }
            // K, V proj on cross-context rows.
            for c in 0..ctx_len {
                let row_in = self
                    .session
                    .ctx_h
                    .view_subrange((c * h) as u64, vec![h as u64]);
                let k_row = self
                    .session
                    .k_ctx_buf
                    .view_subrange((c * kv_dim) as u64, vec![kv_dim as u64]);
                let v_row = self
                    .session
                    .v_ctx_buf
                    .view_subrange((c * kv_dim) as u64, vec![kv_dim as u64]);
                encode_mat_vec_dispatch(ctx_metal, &enc, &layer.k, &row_in, &k_row, h, kv_dim)?;
                encode_mat_vec_dispatch(ctx_metal, &enc, &layer.v, &row_in, &v_row, h, kv_dim)?;
            }
            // Per-head Q/K-norm.
            encode_rms_norm_batched_f32(
                ctx_metal,
                &enc,
                &self.session.q_buf,
                &layer.q_norm,
                &self.session.q_buf,
                n * n_q,
                head_dim,
                RMS_EPS,
            )?;
            encode_rms_norm_batched_f32(
                ctx_metal,
                &enc,
                &self.session.k_noise,
                &layer.k_norm,
                &self.session.k_noise,
                n * n_kv,
                head_dim,
                RMS_EPS,
            )?;
            if ctx_len > 0 {
                encode_rms_norm_batched_f32(
                    ctx_metal,
                    &enc,
                    &self.session.k_ctx_buf,
                    &layer.k_norm,
                    &self.session.k_ctx_buf,
                    ctx_len * n_kv,
                    head_dim,
                    RMS_EPS,
                )?;
            }
            // RoPE Q at noise positions.
            for i in 0..n {
                let row = self
                    .session
                    .q_buf
                    .view_subrange((i * q_dim) as u64, vec![q_dim as u64]);
                encode_rope_neox_f32(
                    ctx_metal,
                    &enc,
                    &row,
                    n_q,
                    head_dim,
                    n_rot,
                    noise_start_pos + i as u32,
                    theta,
                )?;
            }
            // RoPE K_noise.
            for i in 0..n {
                let row = self
                    .session
                    .k_noise
                    .view_subrange((i * kv_dim) as u64, vec![kv_dim as u64]);
                encode_rope_neox_f32(
                    ctx_metal,
                    &enc,
                    &row,
                    n_kv,
                    head_dim,
                    n_rot,
                    noise_start_pos + i as u32,
                    theta,
                )?;
            }
            // RoPE K_ctx at pos_ctx[c].
            for c in 0..ctx_len {
                let pos = pos_ctx_cpu[c] as u32;
                let row = self
                    .session
                    .k_ctx_buf
                    .view_subrange((c * kv_dim) as u64, vec![kv_dim as u64]);
                encode_rope_neox_f32(ctx_metal, &enc, &row, n_kv, head_dim, n_rot, pos, theta)?;
            }
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();

            // ----- Phase 3 (CPU): attention with SWA mask + FFN -----
            // Read q/k/v back, run scalar attention with mask, run scalar
            // SwiGLU FFN, write x += attn_proj_residual + ffn_proj_residual
            // back to GPU.
            // Activation readbacks (small enough to copy; sizes ≪ weights).
            let q_full = read_f32_activation(&self.session.q_buf, n * q_dim);
            let k_noise_cpu = read_f32_activation(&self.session.k_noise, n * kv_dim);
            let v_noise_cpu = read_f32_activation(&self.session.v_noise, n * kv_dim);
            let k_ctx_cpu = if ctx_len > 0 {
                let mut out = vec![0.0f32; ctx_len * kv_dim];
                unsafe {
                    let src = self.session.k_ctx_buf.buffer.contents().as_ptr() as *const f32;
                    std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
                }
                out
            } else {
                Vec::new()
            };
            let v_ctx_cpu = if ctx_len > 0 {
                let mut out = vec![0.0f32; ctx_len * kv_dim];
                unsafe {
                    let src = self.session.v_ctx_buf.buffer.contents().as_ptr() as *const f32;
                    std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
                }
                out
            } else {
                Vec::new()
            };
            // Read x (residual stream).
            let mut x_cpu = read_f32_activation(&self.session.x, n * h);
            // Read post-norm h for FFN.
            // (Will be overwritten by next layer's norm; capture now after attn.)

            // Attention.
            let n_kv_total = ctx_len + n;
            let mut k_full = vec![0.0f32; n_kv_total * kv_dim];
            let mut v_full = vec![0.0f32; n_kv_total * kv_dim];
            if ctx_len > 0 {
                k_full[..ctx_len * kv_dim].copy_from_slice(&k_ctx_cpu);
                v_full[..ctx_len * kv_dim].copy_from_slice(&v_ctx_cpu);
            }
            k_full[ctx_len * kv_dim..].copy_from_slice(&k_noise_cpu);
            v_full[ctx_len * kv_dim..].copy_from_slice(&v_noise_cpu);
            let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
            let swa_window = cfg.swa_window;
            let mut attn_out = vec![0.0f32; n * q_dim];
            for q_idx in 0..n {
                let q_pos = noise_start_pos + q_idx as u32;
                for qh in 0..n_q {
                    let kvh = qh / group;
                    let q_off = q_idx * q_dim + qh * head_dim;
                    let q_slice = &q_full[q_off..q_off + head_dim];
                    let mut scores = vec![f32::NEG_INFINITY; n_kv_total];
                    for k_idx in 0..n_kv_total {
                        let allowed = if k_idx < ctx_len {
                            let k_pos = pos_ctx_cpu[k_idx] as u32;
                            if !layer.is_swa {
                                true
                            } else {
                                q_pos.saturating_sub(k_pos) <= swa_window
                            }
                        } else {
                            (k_idx - ctx_len) <= q_idx
                        };
                        if !allowed {
                            continue;
                        }
                        let k_off = k_idx * kv_dim + kvh * head_dim;
                        let k_slice = &k_full[k_off..k_off + head_dim];
                        let mut s = 0.0f32;
                        for d in 0..head_dim {
                            s += q_slice[d] * k_slice[d];
                        }
                        scores[k_idx] = s * kq_scale;
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    let inv = 1.0 / sum;
                    for s in scores.iter_mut() {
                        *s *= inv;
                    }
                    let out_off = q_idx * q_dim + qh * head_dim;
                    let out_slice = &mut attn_out[out_off..out_off + head_dim];
                    for k_idx in 0..n_kv_total {
                        let w = scores[k_idx];
                        if !w.is_finite() || w == 0.0 {
                            continue;
                        }
                        let v_off = k_idx * kv_dim + kvh * head_dim;
                        for d in 0..head_dim {
                            out_slice[d] += w * v_full[v_off + d];
                        }
                    }
                }
            }
            // O proj per row (CPU, since we already have attn_out on CPU).
            // Weight borrowed (zero-copy view of Metal shared storage).
            let o_w = borrow_f32_tensor(&layer.o);
            for i in 0..n {
                let row_in = &attn_out[i * q_dim..(i + 1) * q_dim];
                let row_out = mat_vec_cpu(o_w, q_dim, h, row_in);
                // Residual #1: x += row_out.
                for d in 0..h {
                    x_cpu[i * h + d] += row_out[d];
                }
            }

            // Pre-FFN RMSNorm on CPU (cheap).
            let post_w = borrow_f32_tensor(&layer.post_attention_norm);
            let mut h_post = vec![0.0f32; n * h];
            for i in 0..n {
                let s = i * h;
                let row = &x_cpu[s..s + h];
                let normed = rms_norm_cpu(row, post_w, RMS_EPS);
                h_post[s..s + h].copy_from_slice(&normed);
            }

            // SwiGLU FFN per row. Weights borrowed.
            let g_w = borrow_f32_tensor(&layer.ffn_gate);
            let u_w = borrow_f32_tensor(&layer.ffn_up);
            let d_w = borrow_f32_tensor(&layer.ffn_down);
            for i in 0..n {
                let row = &h_post[i * h..(i + 1) * h];
                let gate = mat_vec_cpu(g_w, h, f, row);
                let up = mat_vec_cpu(u_w, h, f, row);
                let mut inner = vec![0.0f32; f];
                for j in 0..f {
                    let g = gate[j];
                    let silu_g = g / (1.0 + (-g).exp());
                    inner[j] = silu_g * up[j];
                }
                let down = mat_vec_cpu(d_w, f, h, &inner);
                // Residual #2.
                for d in 0..h {
                    x_cpu[i * h + d] += down[d];
                }
            }

            // Write x_cpu back to GPU for the next layer (or final norm).
            unsafe {
                let dst = self.session.x.buffer.contents().as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(x_cpu.as_ptr(), dst, x_cpu.len());
            }
        }

        // ----- Phase 4 (Metal): final norm + lm_head per noise row -----
        let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        encode_rms_norm_batched_f32(
            ctx_metal,
            &enc,
            &self.session.x,
            &self.head.output_norm,
            &self.session.h,
            n,
            h,
            RMS_EPS,
        )?;
        for i in 0..n {
            let row_in = self.session.h.view_subrange((i * h) as u64, vec![h as u64]);
            let row_out = self
                .session
                .draft_logits
                .view_subrange((i * v) as u64, vec![v as u64]);
            encode_mat_vec_dispatch(
                ctx_metal,
                &enc,
                &self.base.model.lm_head,
                &row_in,
                &row_out,
                h,
                v,
            )?;
        }
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        // Argmax per row (CPU). H5.3 swaps to GPU argmax via top-1 reduction.
        let mut logits = vec![0.0f32; n * v];
        unsafe {
            let src = self.session.draft_logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, logits.as_mut_ptr(), logits.len());
        }
        let mut argmaxes = Vec::with_capacity(n);
        for i in 0..n {
            let row = &logits[i * v..(i + 1) * v];
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(j, _)| j as i32)
                .unwrap_or(0);
            argmaxes.push(argmax);
        }
        Ok(argmaxes)
    }

    /// Same as `draft_block` but returns full `[N, V]` logits (CPU readback)
    /// for cosine validation against `Forward::dflash_draft`. Used by the
    /// H5.1.5 cosine gate.
    pub fn draft_block_with_logits(
        &mut self,
        carry_tok: i32,
        noise_start_pos: u32,
    ) -> Result<Vec<f32>, DFlashError> {
        let _argmaxes = self.draft_block(carry_tok, noise_start_pos)?;
        let n = self.head.config.block_size as usize;
        let v = self.base.model.arch.vocab_size as usize;
        let mut logits = vec![0.0f32; n * v];
        unsafe {
            let src = self.session.draft_logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, logits.as_mut_ptr(), logits.len());
        }
        Ok(logits)
    }
}

fn mat_vec_cpu(w: &[f32], n_in: usize, n_out: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; n_out];
    for o in 0..n_out {
        let mut s = 0.0f32;
        for i in 0..n_in {
            s += x[i] * w[o * n_in + i];
        }
        y[o] = s;
    }
    y
}

fn rms_norm_cpu(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * scale * wi)
        .collect()
}
