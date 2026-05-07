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
    BlitEncoder, KernelEncoder, MetalContext, MetalError, MetalTensor, encode_add_inplace_f32,
    encode_argmax_f32, encode_copy_offset_f32, encode_dflash_attn_f32, encode_gdn_alpha_chain_f32,
    encode_get_rows_f32, encode_rms_norm_batched_f32, encode_rms_norm_mul_f32,
    encode_rope_neox_f32, encode_scatter_offset_f32_to_f16_kv, encode_sigmoid_f32,
    encode_silu_mul_f32, encode_split_q_gate_f32,
};
use crate::metal_forward::{
    MetalBlock, MetalForward, MetalSession, RMS_EPS, encode_mat_mat_dispatch,
    encode_mat_vec_dispatch, encode_scatter_offset_f32, weight_dtype_kept_native,
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
    /// columns). Computed incrementally per `draft_block` call from
    /// `target_ctx_stacked` via `dflash_fc + hidden_norm`. **v0.74.0**:
    /// rows `[0, ctx_h_ready_n)` are already projected and remain valid
    /// across outer steps because `target_ctx_stacked` is append-only;
    /// `draft_block` re-projects only the new delta `[ctx_h_ready_n,
    /// target_ctx_n)`. Pre-v0.74.0 path re-projected the entire ctx
    /// every outer step (linear-in-ctx work, dominant at ctx ≥ 256).
    pub ctx_h: MetalTensor,
    /// **v0.74.0** Watermark for `ctx_h`: number of cross-context columns
    /// that have already been passed through `dflash_fc + hidden_norm`.
    /// `ctx_h[0..ctx_h_ready_n]` is valid, `ctx_h[ctx_h_ready_n..target_ctx_n]`
    /// is the delta to project this `draft_block` call. Append-only;
    /// monotonically non-decreasing within a session. Reset to 0 in
    /// `fresh()` and (defensively) clamped to `target_ctx_n` to
    /// preserve the "valid prefix" invariant if an external caller
    /// ever mutates `target_ctx_n` non-monotonically (currently no
    /// such caller exists; documented for future-proofing).
    pub ctx_h_ready_n: usize,

    /// **v0.74.1** Per-drafter-layer post-norm post-RoPE K cache for
    /// the cross-context. One `[ctx_capacity, kv_dim]` F32 tensor per
    /// drafter layer (5 for Qwen3.6 DFlash). Rows `[0, kv_ctx_ready_n)`
    /// hold the final (K_proj → K_norm → RoPE) result for the
    /// corresponding column of `target_ctx_stacked`; phase 3 attn
    /// reads from these caches instead of the shared `k_ctx_buf`.
    /// Replaces the linear-in-ctx phase 2 redundant work documented in
    /// rev 13: at ctx=1455, the per-layer ctx K/V proj + RoPE was
    /// 115 ms / outer step (5 × ~23 ms / layer); caching reduces
    /// per-step work to projecting only the appended positions.
    pub k_ctx_cache: Vec<MetalTensor>,
    /// **v0.74.1** Per-drafter-layer post-norm V cache. Same shape as
    /// `k_ctx_cache[i]` but V doesn't get RoPE'd (RoPE is K-only in
    /// the drafter forward), so this is just `(V_proj)` per row.
    pub v_ctx_cache: Vec<MetalTensor>,
    /// **v0.74.1** Watermark for the per-layer K/V caches. Number of
    /// cross-context positions that have already been projected +
    /// norm'd + RoPE'd (K only) through ALL drafter layers. Append-
    /// only; monotonically non-decreasing.
    ///
    /// Invariant: `kv_ctx_ready_n <= ctx_h_ready_n <= target_ctx_n`.
    /// kv_ctx_ready_n can lag ctx_h_ready_n if phase 1 ran but phase 2
    /// hasn't caught up (it doesn't today — the two phases always run
    /// in the same `draft_block` call — but the lag is harmless if it
    /// ever happens; phase 2 just projects `[kv_ctx_ready_n, ctx_len)`
    /// regardless of what phase 1 did).
    pub kv_ctx_ready_n: usize,

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
    /// `[N]` i32 (in F32 buffer) — drafter argmax destination, written
    /// by the GPU argmax kernel after the batched lm_head. Avoids the
    /// per-row CPU readback + scalar-loop argmax that v0.71's draft_block
    /// did. v0.72.0 codex-recommended port from packed_verify's batched
    /// tail.
    pub draft_argmax: MetalTensor,

    // ---- v0.72.1 Metal phase 3 buffers ----
    /// `[(ctx_capacity + N) * kv_dim]` F32 — concatenated K (ctx rows
    /// followed by noise rows). Built per-layer per-outer-step in
    /// `draft_block` by scatter-copying from `k_ctx_buf` and `k_noise`,
    /// then passed to `kernel_dflash_attn_f32`. Replaces the v0.71 CPU
    /// concat that was part of the readback.
    pub k_full: MetalTensor,
    /// `[(ctx_capacity + N) * kv_dim]` F32 — same shape as `k_full`, V.
    pub v_full: MetalTensor,
    /// `[(ctx_capacity + N)]` i32 — absolute K positions for `k_full`.
    /// Built on host per outer step and uploaded once.
    pub pos_k: MetalTensor,
    /// `[N * (n_q · head_dim)]` F32 — drafter attention output. Replaces
    /// the v0.71 per-row CPU `attn_out` Vec.
    pub attn_o_full: MetalTensor,
    /// `[N * F_drafter]` F32 — FFN gate output. v0.72.1 Metal phase 3.
    pub ffn_gate_buf: MetalTensor,
    /// `[N * F_drafter]` F32 — FFN up output.
    pub ffn_up_buf: MetalTensor,
    /// `[N * F_drafter]` F32 — silu(gate) * up.
    pub ffn_inner_buf: MetalTensor,
    /// `[N * H_drafter]` F32 — FFN final output. Added to `x` in residual #2.
    pub ffn_out_buf: MetalTensor,

    // ---- v0.72.3 lightweight phase timers ----
    /// When true, `draft_block` reads `cmd.GPUStartTime/EndTime` after
    /// each commit-wait and accumulates per-phase ms into
    /// `phase_timings`. Off by default; flip from the bench harness.
    pub enable_phase_timers: bool,
    /// Per-phase GPU time (ms), keyed by phase name. Repeated keys
    /// (e.g. one entry per layer) are summed by the bench reporter.
    /// Populated when `enable_phase_timers = true`. Cleared by the
    /// caller between bench runs.
    pub phase_timings: Vec<(String, f64)>,
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
            ctx_h_ready_n: 0,
            // v0.74.1: one K/V cache pair per drafter layer (cfg.n_layer).
            // Sized for ctx_capacity * kv_dim. Memory cost on M4 Max
            // 128 GB: 5 layers × ctx_capacity × kv_dim × 2 × 4 B. At
            // ctx_capacity=8K, kv_dim=1024 → 320 MB. Tractable.
            k_ctx_cache: (0..cfg.n_layer)
                .map(|_| MetalTensor::zeros_f32(ctx, vec![cc * kv_dim]))
                .collect::<Result<Vec<_>, _>>()?,
            v_ctx_cache: (0..cfg.n_layer)
                .map(|_| MetalTensor::zeros_f32(ctx, vec![cc * kv_dim]))
                .collect::<Result<Vec<_>, _>>()?,
            kv_ctx_ready_n: 0,
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
            draft_argmax: MetalTensor::zeros_f32(ctx, vec![n])?,
            // v0.72.1 phase 3 buffers
            k_full: MetalTensor::zeros_f32(ctx, vec![(cc + n) * kv_dim])?,
            v_full: MetalTensor::zeros_f32(ctx, vec![(cc + n) * kv_dim])?,
            pos_k: MetalTensor::zeros_f32(ctx, vec![cc + n])?,
            attn_o_full: MetalTensor::zeros_f32(ctx, vec![n * q_dim])?,
            ffn_gate_buf: MetalTensor::zeros_f32(ctx, vec![n * (cfg.intermediate_size as u64)])?,
            ffn_up_buf: MetalTensor::zeros_f32(ctx, vec![n * (cfg.intermediate_size as u64)])?,
            ffn_inner_buf: MetalTensor::zeros_f32(ctx, vec![n * (cfg.intermediate_size as u64)])?,
            ffn_out_buf: MetalTensor::zeros_f32(ctx, vec![n * h])?,
            enable_phase_timers: false,
            phase_timings: Vec::new(),
        })
    }

    /// Enable v0.72.3 lightweight phase timers; clears any prior
    /// timing buffer.
    pub fn enable_phase_timers(&mut self) {
        self.enable_phase_timers = true;
        self.phase_timings.clear();
    }

    /// Take the accumulated timings and reset the buffer.
    pub fn take_phase_timings(&mut self) -> Vec<(String, f64)> {
        std::mem::take(&mut self.phase_timings)
    }

    /// v0.72.3 helper: append `(name, gpu_ms)` to phase_timings if
    /// timing is enabled. Read AFTER `cmd.waitUntilCompleted()`.
    /// Pulls GPU time directly from the cmd buffer (not wall time;
    /// mirrors `single_token_phase_profiled` in metal_forward.rs).
    pub(crate) fn maybe_record(
        &mut self,
        name: &str,
        cmd: &objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        >,
    ) {
        if !self.enable_phase_timers {
            return;
        }
        let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        self.phase_timings.push((name.to_string(), gpu_ms));
    }

    /// Convenience wrapper: builds its own command buffer and waits.
    /// Use when caller doesn't already own a `KernelEncoder` (e.g. CLI
    /// tools that don't depend on `objc2-metal` directly).
    pub fn append_target_ctx_column_now(
        &mut self,
        ctx: &MetalContext,
        hidden_block: &MetalTensor,
        position: u32,
        n_target_features: usize,
    ) -> Result<(), DFlashError> {
        let cmd = ctx.queue.commandBuffer().expect("cmd buffer");
        let enc = KernelEncoder::begin(&cmd);
        self.append_target_ctx_column(ctx, &enc, hidden_block, position, n_target_features)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
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

    /// **v0.75.1** Contiguous-batch append: scatters a flat `[count *
    /// n_target_features]` F32 source (laid out as `count` consecutive
    /// `[K · H_target]` rows) into `target_ctx_stacked` at positions
    /// `[start_pos, start_pos + count)`, using ONE scatter dispatch +
    /// ONE wait.
    ///
    /// This is the data-flow match for `prefill_tokens_with_multi_hidden`:
    /// the prefill body writes per-token hidden captures into a
    /// contiguous flat tensor `[T, K · H_target]`, and this helper
    /// appends them to the drafter's cross-context in one shot. The
    /// alternative (per-token `append_target_ctx_column_now` in a loop)
    /// pays T encoder + commit + wait costs which add up at ctx ≥ 1K.
    ///
    /// `src` must have at least `count * n_target_features` elements;
    /// only the first `count * n_target_features` are read. The
    /// destination must have capacity for `count` more columns; otherwise
    /// returns `DFlashError::CtxOverflow`.
    ///
    /// Positions are stamped sequentially: `pos_ctx[target_ctx_n + i]
    /// = start_pos + i` for `i in 0..count`.
    pub fn append_target_ctx_columns_contiguous_now(
        &mut self,
        ctx: &MetalContext,
        src: &MetalTensor,
        start_pos: u32,
        count: usize,
        n_target_features: usize,
    ) -> Result<(), DFlashError> {
        if count == 0 {
            return Ok(());
        }
        if self.target_ctx_n + count > self.target_ctx_capacity {
            return Err(DFlashError::CtxOverflow(
                self.target_ctx_n + count,
                self.target_ctx_capacity,
            ));
        }
        let n_elems = count * n_target_features;
        if (src.n_elements() as usize) < n_elems {
            return Err(DFlashError::Metal(MetalError::BadShape {
                kernel: "append_target_ctx_columns_contiguous_now.src",
                detail: format!(
                    "expected ≥ {n_elems} elements (count={count} × n_target_features={n_target_features}), got {}",
                    src.n_elements()
                ),
            }));
        }
        // The destination layout is contiguous: rows are stored back-to-
        // back at `target_ctx_n * n_target_features`. So a single
        // scatter copies the whole [count * n_target_features] slab.
        // We use a sized view of src to satisfy encode_scatter_offset_f32's
        // "src.n == n" precondition.
        let cmd = ctx.queue.commandBuffer().expect("cmd buffer");
        let enc = KernelEncoder::begin(&cmd);
        let src_view = src.view_subrange(0, vec![n_elems as u64]);
        let dst_off = self.target_ctx_n * n_target_features;
        encode_scatter_offset_f32(
            ctx,
            &enc,
            &src_view,
            &self.target_ctx_stacked,
            dst_off,
            n_elems,
        )?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        // Stamp positions on the host side (pos_ctx is shared-storage F32
        // typed buffer holding i32; same convention as
        // append_target_ctx_column).
        unsafe {
            let ptr = self.pos_ctx.buffer.contents().as_ptr() as *mut i32;
            for i in 0..count {
                *ptr.add(self.target_ctx_n + i) = (start_pos as i32) + i as i32;
            }
        }
        self.target_ctx_n += count;
        Ok(())
    }

    /// **v0.74.3** Batched-commit version of `append_target_ctx_column_now`:
    /// runs `columns.len()` appends through ONE command buffer + ONE
    /// commit + ONE wait, instead of N waits.
    ///
    /// Used by the DFlash hot decode loop after greedy accept-prefix
    /// emits 1..N committed positions per outer step. The single-column
    /// `_now` variant (still preserved for prefill where we need
    /// per-token sequencing) creates its own command buffer + waits
    /// per call, costing N CPU/GPU sync points per outer step. At
    /// α_chain=5.2 drafts/step typical for code prompts that's ~6
    /// waits per outer step that this batched helper collapses to 1.
    ///
    /// Each `(hidden_block, position)` pair must satisfy the same
    /// per-call validation as `append_target_ctx_column`. If any
    /// individual append fails (capacity overflow, shape mismatch),
    /// no further appends are attempted but the partially-progressed
    /// session state is left as-is — the failing call returns Err
    /// and the caller should treat session state as inconsistent
    /// (which mirrors the per-call `_now` failure semantics).
    pub fn append_target_ctx_columns_now(
        &mut self,
        ctx: &MetalContext,
        columns: &[(&MetalTensor, u32)],
        n_target_features: usize,
    ) -> Result<(), DFlashError> {
        if columns.is_empty() {
            return Ok(());
        }
        let cmd = ctx.queue.commandBuffer().expect("cmd buffer");
        let enc = KernelEncoder::begin(&cmd);
        for &(hidden_block, position) in columns {
            self.append_target_ctx_column(ctx, &enc, hidden_block, position, n_target_features)?;
        }
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }
}

// =============================================================================
// MetalDFlashVerifyScratch — H5.3a packed verify scratch (PRODUCTION shape)
// =============================================================================
//
// Owns ALL N-shaped state needed by `MetalForward::packed_forward` (H5.3a).
// Allocated once per `DFlashDecoder`; threaded `&mut` through `packed_forward`.
//
// Codex partner-session refinements baked in here:
//
//   * **No `x_pack` / `h_pack` / etc.** Naive H5.3a runs N successive
//     single-token paths inside one command buffer; GPU writes sequence
//     within a command buffer, so reusing `MetalSession::x` / `h` /
//     `ffn_inner` etc. across the N tokens is correct (kernel N+1 reads
//     what kernel N wrote, by Metal's per-encoder ordering guarantee).
//     H5.3b tiled mat-mat will need N-wide activation buffers; that's
//     a separate scratch type when we get there.
//
//   * **`packed_ids_buf: [N]` is the one buffer that MUST be N-wide
//     in H5.3a.** `MetalSession::ids_buf` is mutated by HOST CPU in
//     between encoded kernels — re-using it across N tokens means
//     every queued `get_rows` reads the LAST-written CPU value
//     (predicted bug; Q7 mitigation). Each token reads
//     `packed_ids_buf.view_subrange(n, [1])`.
//
//   * **No `[N, V]` `debug_logits` in the production struct.** Per-token
//     argmax runs on the reused `MetalSession::logits` and writes into
//     `verify_argmax[n]` via `encode_argmax_f32` with `n_rows=1`. Saves
//     15.9 MB per outer step that we'd otherwise allocate for nothing
//     in production. The `_with_logits` debug variant uses
//     `MetalDFlashDebugScratch` (below) which adds the `[N, V]` buffer.
//
//   * **One backing `MetalTensor` per checkpoint class with
//     `slot_view(layer, n) -> MetalTensor` helpers.** Avoids 1536
//     `Retained` clones at construction; keeps `BlitEncoder::copy_tensor`
//     ergonomics at blit time (the per-call clone cost is irrelevant
//     since we only call `slot_view` during encode).
//
// Sizes (Qwen3.6-27B target, N=16, K=5 target_layer_ids, V=248320,
// n_gdn=48):
//   verify_argmax:   N · 4 B          =        64 B
//   hidden_capture:  K · N · H · 4 B  =     1.6 MB     (5 · 16 · 5120 · 4)
//   gdn_ckpt:        n_gdn · N · ssm  =     2.3 GiB    (48 · 16 · 3 MiB)
//   conv_ckpt:       n_gdn · N · conv =      90 MiB    (48 · 16 · 120 KiB)
//   packed_ids_buf:  N · 4 B          =        64 B
pub struct MetalDFlashVerifyScratch {
    /// `[N]` i32 — packed verify input tokens. Each block reads from
    /// `view_subrange(n, [1])`. Filled by `packed_forward` from `tokens`.
    pub packed_ids_buf: MetalTensor,

    /// `[N]` i32 — GPU-computed argmax tokens, one per packed position.
    /// Written into via `encode_argmax_f32` after each block's lm_head.
    pub verify_argmax: MetalTensor,

    /// `[K, N, H]` F32 — multi-layer hidden capture. Layer `target_layer_ids[k]`
    /// after token n in the packed batch lives at offset `(k * N + n) * H`.
    pub hidden_capture: MetalTensor,

    /// `[n_gdn, N, ssm_state_elems]` F32 — per-GDN-layer per-token SSM
    /// checkpoint. `gdn_ckpt_slot(layer, n)` returns the `[ssm_state_elems]`
    /// view; blit dest after the layer's `gdn_step` for token n.
    pub gdn_ckpt: MetalTensor,

    /// `[n_gdn, N, conv_state_elems]` F32 — per-GDN-layer per-token conv
    /// state checkpoint. `conv_ckpt_slot(layer, n)` returns the view.
    pub conv_ckpt: MetalTensor,

    // -- Cached dimensions (so slot helpers don't have to take a model ref) --
    pub n: u32,
    pub k_target_layers: u32,
    pub n_gdn_layers: u32,
    pub hidden_size: u64,
    pub ssm_state_elems: u64,
    pub conv_state_elems: u64,
}

impl MetalDFlashVerifyScratch {
    /// Allocate scratch for one DFlash outer step.
    ///
    /// `block_size` (= N) and `target_layer_ids.len()` (= K) come from the
    /// drafter config; everything else is pulled from the target model
    /// arch + layer schedule (so we never carry contradictions between
    /// what we allocate and what packed_forward expects).
    pub fn fresh(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
        k_target_layers: u32,
    ) -> Result<Self, MetalError> {
        let arch = &target_model.arch;
        let n = block_size as u64;
        let k = k_target_layers as u64;
        let h = arch.hidden_size as u64;

        // Count GDN layers from the layer schedule (matches MetalSession::fresh).
        let n_gdn_layers = target_model
            .blocks
            .iter()
            .filter(|b| matches!(b, crate::metal_forward::MetalBlock::Gdn(_)))
            .count() as u64;

        // SSM state: n_v_heads · head_dim · head_dim (F32).
        let ssm_state_elems =
            (arch.gdn_n_v_heads as u64) * (arch.gdn_head_dim as u64) * (arch.gdn_head_dim as u64);

        // Conv state: (kernel - 1) · conv_dim where
        // conv_dim = (2 * n_k + n_v) * head_dim.
        let conv_dim = (2 * (arch.gdn_n_k_heads as u64) + (arch.gdn_n_v_heads as u64))
            * (arch.gdn_head_dim as u64);
        let conv_state_elems = ((arch.gdn_conv_kernel as u64) - 1) * conv_dim;

        Ok(Self {
            packed_ids_buf: MetalTensor::zeros_f32(ctx, vec![n])?, // i32 in F32 buf
            verify_argmax: MetalTensor::zeros_f32(ctx, vec![n])?,  // i32 in F32 buf
            // Layout: [N, K, H] (NOT [K, N, H] as in v0.57). Each per-N
            // slot is `K * H` contiguous floats — exactly what
            // `MetalDFlashSession::append_target_ctx_column_now` expects
            // as a single `[K * H]` hidden_block per column. Per-block
            // writes during the layer loop now scatter at offset
            // `(n * K + k) * H` instead of `(k * N + n) * H`. Wins
            // because reads-by-n (during target_ctx append, hot path
            // in the H5.5 outer decode loop) are contiguous; writes
            // (per-block, K times per outer step) stay cheap.
            hidden_capture: MetalTensor::zeros_f32(ctx, vec![n, k, h])?,
            gdn_ckpt: MetalTensor::zeros_f32(ctx, vec![n_gdn_layers, n, ssm_state_elems])?,
            conv_ckpt: MetalTensor::zeros_f32(ctx, vec![n_gdn_layers, n, conv_state_elems])?,
            n: block_size,
            k_target_layers,
            n_gdn_layers: n_gdn_layers as u32,
            hidden_size: h,
            ssm_state_elems,
            conv_state_elems,
        })
    }

    /// Zero-copy view of GDN SSM checkpoint slot `(layer, n)` ∈
    /// `[0, n_gdn) × [0, N)`. Returned shape: `[ssm_state_elems]`.
    /// Used as a blit destination after the layer's gdn_step for token n,
    /// or as a blit source on rollback.
    ///
    /// **Runtime-asserts** bounds (NOT debug_assert) per codex H5.3a
    /// review: the failure mode if `(layer, n)` is OOB is silent
    /// out-of-bounds bytes written via blit on release builds. Cheap
    /// guard; compile into release.
    pub fn gdn_ckpt_slot(&self, layer: u32, n: u32) -> MetalTensor {
        assert!(
            layer < self.n_gdn_layers,
            "gdn_ckpt_slot OOB: layer={layer} >= n_gdn_layers={}",
            self.n_gdn_layers
        );
        assert!(
            n < self.n,
            "gdn_ckpt_slot OOB: n={n} >= scratch.n={}",
            self.n
        );
        let elem_offset = (layer as u64 * self.n as u64 + n as u64) * self.ssm_state_elems;
        self.gdn_ckpt
            .view_subrange(elem_offset, vec![self.ssm_state_elems])
    }

    /// Zero-copy view of conv checkpoint slot `(layer, n)`. Returned shape:
    /// `[conv_state_elems]`.
    pub fn conv_ckpt_slot(&self, layer: u32, n: u32) -> MetalTensor {
        assert!(
            layer < self.n_gdn_layers,
            "conv_ckpt_slot OOB: layer={layer} >= n_gdn_layers={}",
            self.n_gdn_layers
        );
        assert!(
            n < self.n,
            "conv_ckpt_slot OOB: n={n} >= scratch.n={}",
            self.n
        );
        let elem_offset = (layer as u64 * self.n as u64 + n as u64) * self.conv_state_elems;
        self.conv_ckpt
            .view_subrange(elem_offset, vec![self.conv_state_elems])
    }

    /// Zero-copy view of hidden_capture slot `(k, n)`. Returned shape:
    /// `[hidden_size]`. Used as a scatter destination after the K-indexed
    /// target layer's residual for token n.
    ///
    /// **Storage layout: `[N, K, H]` row-major** (changed from `[K, N, H]`
    /// in v0.71). Slot `(k, n)` lives at offset `(n * K + k) * H`. The
    /// `[N, K, H]` layout makes per-N reads contiguous (`K*H` floats per
    /// token), which is exactly what
    /// `MetalDFlashSession::append_target_ctx_column_now` consumes during
    /// the H5.5 outer decode loop. Per-block writes (K times per outer
    /// step) stay cheap.
    pub fn hidden_capture_slot(&self, k: u32, n: u32) -> MetalTensor {
        assert!(
            k < self.k_target_layers,
            "hidden_capture_slot OOB: k={k} >= k_target_layers={}",
            self.k_target_layers
        );
        assert!(
            n < self.n,
            "hidden_capture_slot OOB: n={n} >= scratch.n={}",
            self.n
        );
        let elem_offset = (n as u64 * self.k_target_layers as u64 + k as u64) * self.hidden_size;
        self.hidden_capture
            .view_subrange(elem_offset, vec![self.hidden_size])
    }

    /// Zero-copy view of hidden_capture for ALL K layers at token `n`.
    /// Returned shape: `[K * H]`. Convenient for
    /// `MetalDFlashSession::append_target_ctx_column_now`, which consumes
    /// exactly this contiguous slab per appended column.
    ///
    /// Only valid under the `[N, K, H]` storage layout (which v0.71
    /// switched to). The N-row stride is `K * H` floats, contiguous.
    pub fn hidden_capture_n_slot(&self, n: u32) -> MetalTensor {
        assert!(
            n < self.n,
            "hidden_capture_n_slot OOB: n={n} >= scratch.n={}",
            self.n
        );
        let kh = self.k_target_layers as u64 * self.hidden_size;
        let elem_offset = n as u64 * kh;
        self.hidden_capture.view_subrange(elem_offset, vec![kh])
    }

    /// Zero-copy view of `packed_ids_buf[n..n+1]`. Used as the
    /// `get_rows` token-id input for block n; required to avoid the
    /// shared-CPU-mutable `MetalSession::ids_buf` race that would
    /// silently corrupt N successive `get_rows` calls in one command
    /// buffer (codex Q7 — the bug we'd ship without this).
    pub fn token_slot(&self, n: u32) -> MetalTensor {
        assert!(n < self.n, "token_slot OOB: n={n} >= scratch.n={}", self.n);
        self.packed_ids_buf.view_subrange(n as u64, vec![1])
    }

    /// Zero-copy view of `verify_argmax[n..n+1]`. Used as the destination
    /// for `encode_argmax_f32` over block n's logits.
    pub fn argmax_slot(&self, n: u32) -> MetalTensor {
        assert!(n < self.n, "argmax_slot OOB: n={n} >= scratch.n={}", self.n);
        self.verify_argmax.view_subrange(n as u64, vec![1])
    }
}

// =============================================================================
// MetalDFlashDebugScratch — debug-only extension with [N, V] logits buffer
// =============================================================================
//
// Wraps a `MetalDFlashVerifyScratch` and adds an `[N, V]` F32 buffer for
// the H5.3a bit-exactness gate (G1: cosine ≥ 0.9999 vs N successive
// `single_token`). Production never allocates `debug_logits`; the
// `_with_logits` debug entrypoint uses this struct instead.
//
// Why two structs vs `Option<debug_logits>` (codex Q3): no dead `Option`
// paths in prod; allocation is explicit at the type level. Cost: small
// refactor footprint; debug variant takes `&mut MetalDFlashDebugScratch`
// and accesses `verify` for the N-shaped fields.
pub struct MetalDFlashDebugScratch {
    pub verify: MetalDFlashVerifyScratch,
    /// `[N, V]` F32 — full vocab logits per packed position, written by
    /// the debug variant of packed_forward. Used for bit-exactness gate
    /// only; never read on the production path.
    pub debug_logits: MetalTensor,
}

impl MetalDFlashDebugScratch {
    pub fn fresh(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
        k_target_layers: u32,
    ) -> Result<Self, MetalError> {
        let arch = &target_model.arch;
        let n = block_size as u64;
        let v = arch.vocab_size as u64;
        let verify =
            MetalDFlashVerifyScratch::fresh(ctx, target_model, block_size, k_target_layers)?;
        Ok(Self {
            verify,
            debug_logits: MetalTensor::zeros_f32(ctx, vec![n, v])?,
        })
    }

    /// Zero-copy view of `debug_logits[n, :]`. Used as the lm_head
    /// destination for block n's logits.
    pub fn logits_slot(&self, n: u32) -> MetalTensor {
        assert!(
            n < self.verify.n,
            "logits_slot OOB: n={n} >= scratch.n={}",
            self.verify.n
        );
        let v = self.debug_logits.shape[1];
        let elem_offset = n as u64 * v;
        self.debug_logits.view_subrange(elem_offset, vec![v])
    }
}

// =============================================================================
// MetalDFlashLayerMajorScratch — H5.3b.4-5 N-wide activation buffers
// =============================================================================
//
// Owns the [N, *] activation buffers needed by the layer-major
// packed_verify path (`encode_packed_verify_layer_major_inner`).
// Sits ALONGSIDE `MetalDFlashVerifyScratch` (which keeps owning
// outputs + checkpoints + packed_ids_buf). The token-major naive
// path does NOT allocate this struct — codex Q6 (parallel scratch)
// to keep oracle/debug paths cheap.
//
// All buffers are F32 row-major `[N, dim]`. Total size at
// Qwen3.6-27B with N=16:
//   x_pack            [N, H]            16·5120·4   =   320 KiB
//   h_pack            [N, H]            16·5120·4   =   320 KiB
//   mixer_out_pack    [N, H]            16·5120·4   =   320 KiB
//   attn_q_full_pack  [N, 2·q_dim]      16·12288·4  =   768 KiB  (gated Q)
//   attn_q_pack       [N, q_dim]        16·6144·4   =   384 KiB
//   attn_gate_pack    [N, q_dim]        16·6144·4   =   384 KiB
//   attn_q_normed_pack[N, q_dim]        16·6144·4   =   384 KiB
//   attn_k_now_pack   [N, kv_dim]       16·1024·4   =    64 KiB
//   attn_v_now_pack   [N, kv_dim]       16·1024·4   =    64 KiB
//   attn_k_normed_pack[N, kv_dim]       16·1024·4   =    64 KiB
//   attn_o_pack       [N, q_dim]        16·6144·4   =   384 KiB
//   ffn_gate_pack     [N, F]            16·17408·4  =  1088 KiB
//   ffn_up_pack       [N, F]            16·17408·4  =  1088 KiB
//   ffn_inner_pack    [N, F]            16·17408·4  =  1088 KiB
//   ffn_out_pack      [N, H]            16·5120·4   =   320 KiB
//                                                    ----------
//                                                    ~ 7.0 MiB
pub struct MetalDFlashLayerMajorScratch {
    /// `[N, H]` F32 — residual stream across N tokens.
    pub x_pack: MetalTensor,
    /// `[N, H]` F32 — post-norm activation across N tokens (reused for
    /// both pre-attn and pre-FFN norms).
    pub h_pack: MetalTensor,
    /// `[N, H]` F32 — mixer output (GDN or attn).
    pub mixer_out_pack: MetalTensor,

    // Attention scratch (only meaningful on attn layers).
    /// `[N, 2·q_dim]` F32 — gated Q projection (Q + gate interleaved).
    pub attn_q_full_pack: MetalTensor,
    /// `[N, q_dim]` F32 — Q after split.
    pub attn_q_pack: MetalTensor,
    /// `[N, q_dim]` F32 — gate after split.
    pub attn_gate_pack: MetalTensor,
    /// `[N, q_dim]` F32 — Q after per-head RMSNorm.
    pub attn_q_normed_pack: MetalTensor,
    /// `[N, kv_dim]` F32 — K projection.
    pub attn_k_now_pack: MetalTensor,
    /// `[N, kv_dim]` F32 — V projection.
    pub attn_v_now_pack: MetalTensor,
    /// `[N, kv_dim]` F32 — K after per-head RMSNorm.
    pub attn_k_normed_pack: MetalTensor,
    /// `[N, q_dim]` F32 — attention output (post softmax+V agg, post gate).
    pub attn_o_pack: MetalTensor,

    // FFN scratch.
    /// `[N, F]` F32 — FFN gate output (skipped when fused Q4_K SwiGLU is used).
    pub ffn_gate_pack: MetalTensor,
    /// `[N, F]` F32 — FFN up output.
    pub ffn_up_pack: MetalTensor,
    /// `[N, F]` F32 — silu(gate) * up.
    pub ffn_inner_pack: MetalTensor,
    /// `[N, H]` F32 — FFN final.
    pub ffn_out_pack: MetalTensor,

    /// `[N, V]` F32 — batched lm_head output (final logits across all N
    /// tokens). H5.3b.6: lifts lm_head out of the per-token mat-vec
    /// re-read loop. Allocation is ~16 MB at vocab=248320, N=16.
    /// Trivial overhead vs the GiB-scale checkpoint scratch already in
    /// MetalDFlashVerifyScratch.
    ///
    /// In the production path (`encode_packed_verify_layer_major_inner`
    /// with `debug_logits_dst = None`), this buffer holds the batched
    /// lm_head output, then GPU argmax reads from it row-by-row to
    /// produce `verify_argmax[N]`. The `[N, V]` bytes never leave the
    /// GPU — no CPU readback. The H5.3a anti-regression assertion
    /// (no per-step `[N, V]` spill) still holds.
    ///
    /// In the debug path (`Some(debug_logits_dst)`), the
    /// `MetalDFlashDebugScratch::debug_logits` buffer is used INSTEAD
    /// (it's already `[N, V]` shaped); this `final_logits_pack` is
    /// not touched by the debug variant.
    pub final_logits_pack: MetalTensor,

    // GDN batched-projection scratch (v0.73a). Selectively populated
    // by the layer-major path's GDN front-end and back-end mat-mat
    // dispatches when the layer's projections are mat-mat eligible
    // (`gdn_mat_mat_eligible`). The actual production 27B Q4_K_M GDN
    // dtype mix is:
    //
    //   in_proj_qkv: Q6_K [hidden, conv_dim]   — batched ⇒ gdn_qkv_pack
    //   in_proj_z:   Q4_K [hidden, v_dim]      — batched ⇒ gdn_z_pack
    //   beta_proj:   F32  [hidden, n_v]        — stays per-token mat-vec (small, F32)
    //   alpha_proj:  F32  [hidden, n_v]        — stays per-token mat-vec (small, F32)
    //   out_proj:    Q5_K [v_dim, hidden]      — batched ⇒ gdn_normed_pack → mixer_out_pack
    //
    // beta/alpha are F32 with n_out=48 (1 MB each); mat-mat dispatch
    // overhead exceeds the BW savings, so they stay per-token. If a
    // future GGUF quantizes them, the eligibility predicate widens.
    /// `[N, conv_dim]` F32 — batched in_proj_qkv output. conv_dim =
    /// (2*n_k + n_v) * head_dim. 27B: [16, 10240] = 640 KiB.
    pub gdn_qkv_pack: MetalTensor,
    /// `[N, v_dim]` F32 — batched in_proj_z output. v_dim = n_v * head_dim.
    /// 27B: [16, 6144] = 384 KiB.
    pub gdn_z_pack: MetalTensor,
    /// `[N, v_dim]` F32 — RMSNormGated output across N tokens; consumed by
    /// the batched out_proj mat-mat after the per-token recurrence loop.
    /// 27B: [16, 6144] = 384 KiB.
    pub gdn_normed_pack: MetalTensor,

    // Cached dims so callers don't have to re-derive.
    pub n: u32,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub q_dim: u64,
    pub kv_dim: u64,
    pub vocab_size: u64,
    /// GDN conv_dim = (2*n_k + n_v) * head_dim. Cached for layer-major
    /// GDN batching (v0.73a). Zero on architectures without GDN.
    pub gdn_conv_dim: u64,
    /// GDN v_dim = n_v * head_dim. Cached for layer-major GDN batching
    /// (v0.73a). Zero on architectures without GDN.
    pub gdn_v_dim: u64,
    /// GDN n_v_heads. Used to size beta/alpha projections.
    pub gdn_n_v: u64,
}

impl MetalDFlashLayerMajorScratch {
    pub fn fresh(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
    ) -> Result<Self, MetalError> {
        let arch = &target_model.arch;
        let n = block_size as u64;
        let h = arch.hidden_size as u64;
        let f = arch.intermediate_size as u64;
        let head_dim = arch.attn_head_dim as u64;
        let q_dim = (arch.n_q_heads as u64) * head_dim;
        let kv_dim = (arch.n_kv_heads as u64) * head_dim;
        let v = arch.vocab_size as u64;

        // GDN dims. Sized at 1 element when the arch has no GDN to keep
        // the buffers allocatable; the GDN layer-major path is gated on
        // `gdn_mat_mat_eligible` and never reads from these on non-GDN
        // archs.
        let gdn_head_dim = arch.gdn_head_dim as u64;
        let gdn_n_v = arch.gdn_n_v_heads as u64;
        let gdn_n_k = arch.gdn_n_k_heads as u64;
        let gdn_v_dim = (gdn_n_v * gdn_head_dim).max(1);
        let gdn_conv_dim = ((2 * gdn_n_k + gdn_n_v) * gdn_head_dim).max(1);

        Ok(Self {
            x_pack: MetalTensor::zeros_f32(ctx, vec![n, h])?,
            h_pack: MetalTensor::zeros_f32(ctx, vec![n, h])?,
            mixer_out_pack: MetalTensor::zeros_f32(ctx, vec![n, h])?,
            attn_q_full_pack: MetalTensor::zeros_f32(ctx, vec![n, 2 * q_dim])?,
            attn_q_pack: MetalTensor::zeros_f32(ctx, vec![n, q_dim])?,
            attn_gate_pack: MetalTensor::zeros_f32(ctx, vec![n, q_dim])?,
            attn_q_normed_pack: MetalTensor::zeros_f32(ctx, vec![n, q_dim])?,
            attn_k_now_pack: MetalTensor::zeros_f32(ctx, vec![n, kv_dim])?,
            attn_v_now_pack: MetalTensor::zeros_f32(ctx, vec![n, kv_dim])?,
            attn_k_normed_pack: MetalTensor::zeros_f32(ctx, vec![n, kv_dim])?,
            attn_o_pack: MetalTensor::zeros_f32(ctx, vec![n, q_dim])?,
            ffn_gate_pack: MetalTensor::zeros_f32(ctx, vec![n, f])?,
            ffn_up_pack: MetalTensor::zeros_f32(ctx, vec![n, f])?,
            ffn_inner_pack: MetalTensor::zeros_f32(ctx, vec![n, f])?,
            ffn_out_pack: MetalTensor::zeros_f32(ctx, vec![n, h])?,
            final_logits_pack: MetalTensor::zeros_f32(ctx, vec![n, v])?,
            gdn_qkv_pack: MetalTensor::zeros_f32(ctx, vec![n, gdn_conv_dim])?,
            gdn_z_pack: MetalTensor::zeros_f32(ctx, vec![n, gdn_v_dim])?,
            gdn_normed_pack: MetalTensor::zeros_f32(ctx, vec![n, gdn_v_dim])?,
            n: block_size,
            hidden_size: h,
            intermediate_size: f,
            vocab_size: v,
            q_dim,
            kv_dim,
            gdn_conv_dim,
            gdn_v_dim,
            gdn_n_v,
        })
    }

    /// Zero-copy view of row n of `x_pack` ([H] elements).
    pub fn x_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n, "x_row OOB: n={n} >= scratch.n={}", self.n);
        self.x_pack
            .view_subrange((n as u64) * self.hidden_size, vec![self.hidden_size])
    }

    /// Zero-copy view of row n of `h_pack`.
    pub fn h_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n);
        self.h_pack
            .view_subrange((n as u64) * self.hidden_size, vec![self.hidden_size])
    }

    /// Zero-copy view of row n of `mixer_out_pack`.
    pub fn mixer_out_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n);
        self.mixer_out_pack
            .view_subrange((n as u64) * self.hidden_size, vec![self.hidden_size])
    }

    /// Zero-copy view of row n of `attn_q_full_pack` (`[2*q_dim]`).
    pub fn attn_q_full_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n);
        let two_q = 2 * self.q_dim;
        self.attn_q_full_pack
            .view_subrange((n as u64) * two_q, vec![two_q])
    }

    /// Zero-copy view of row n of `attn_q_pack`, `attn_gate_pack`, etc.
    pub fn attn_q_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n);
        self.attn_q_pack
            .view_subrange((n as u64) * self.q_dim, vec![self.q_dim])
    }

    pub fn attn_gate_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n);
        self.attn_gate_pack
            .view_subrange((n as u64) * self.q_dim, vec![self.q_dim])
    }

    pub fn attn_q_normed_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n);
        self.attn_q_normed_pack
            .view_subrange((n as u64) * self.q_dim, vec![self.q_dim])
    }

    pub fn attn_k_now_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n);
        self.attn_k_now_pack
            .view_subrange((n as u64) * self.kv_dim, vec![self.kv_dim])
    }

    pub fn attn_v_now_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n);
        self.attn_v_now_pack
            .view_subrange((n as u64) * self.kv_dim, vec![self.kv_dim])
    }

    pub fn attn_k_normed_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n);
        self.attn_k_normed_pack
            .view_subrange((n as u64) * self.kv_dim, vec![self.kv_dim])
    }

    pub fn attn_o_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n);
        self.attn_o_pack
            .view_subrange((n as u64) * self.q_dim, vec![self.q_dim])
    }

    pub fn ffn_inner_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n);
        self.ffn_inner_pack.view_subrange(
            (n as u64) * self.intermediate_size,
            vec![self.intermediate_size],
        )
    }

    pub fn ffn_out_row(&self, n: u32) -> MetalTensor {
        assert!(n < self.n);
        self.ffn_out_pack
            .view_subrange((n as u64) * self.hidden_size, vec![self.hidden_size])
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

    // =====================================================================
    // packed_verify — H5.3a packed target verify forward (PRODUCTION path)
    // =====================================================================
    //
    // Naive H5.3a per plan rev 4 §H5.3a: N successive single-token paths
    // inside ONE command buffer; per-token GDN+conv state checkpoints
    // copied via blit between compute encoders; multi-layer hidden
    // capture inline at target_layer_ids; final logits → GPU argmax →
    // verify_argmax (no [N, V] CPU readback).
    //
    // Codex Q1 placement: this lives on `DFlashDecoder` (NOT `MetalForward`)
    // because the target driver should not bake in DFlash-specific
    // semantics. `encode_block` is bumped to `pub(crate)` to allow
    // re-use here without exposing it as a public API.
    //
    // Codex failure-mode mitigation: explicit dim guard at entry.
    // `MetalDFlashVerifyScratch::slot_*` use `debug_assert!`, which is
    // a no-op in release; without the guard wall here, a scratch
    // allocated for a different `block_size` / `target_layer_ids.len()`
    // / target model schedule would silently write bytes at wrong
    // offsets in production. The guard is the difference between
    // "panic on dev, corrupt on prod" and "fail-loudly always."
    //
    // Codex Q2 design Y (batched-end-of-token GDN/conv blits): each
    // `gdn_state[k]` is mutated only by GDN layer k's gdn_step and each
    // `gdn_conv[k]` only by GDN layer k's ssm_conv_silu, so after all
    // 64 blocks for token n complete, blitting `gdn_state[k]` and
    // `gdn_conv[k]` into the n-slot of the checkpoint captures the
    // correct post-token-n state. 32 encoder transitions per outer
    // step (16 tokens × 2 transitions) instead of 1536 (per-block).
    //
    // Codex Q4 hidden capture timing: MUST be inline in the per-token
    // compute encoder (not the post-token blit pass) because by the
    // next compute encoder runs, `session.x` will be overwritten with
    // token n+1's embedding.
    //
    // Codex Q5 GPU argmax timing: MUST be inline after lm_head (before
    // session.logits is reused for token n+1).
    //
    // Codex Q6 cmd buffer: ONE command buffer for all N tokens
    // (alternating compute / blit passes). Single commit + wait. The
    // simplest correctness-scaffold posture.
    /// Default packed verify path: **layer-major** (H5.3b.4-5).
    /// Uses batched RMSNorm + batched mat-mat for FFN/projections,
    /// per-token GDN/attn mixers. Requires `MetalDFlashLayerMajorScratch`.
    ///
    /// Token-major fallback `packed_verify_token_major` is preserved
    /// as the correctness oracle (per codex Q4 — the two are tested
    /// bit-exact against each other).
    pub fn packed_verify(
        &self,
        tokens: &[i32],
        start_position: u32,
        verify_scratch: &mut MetalDFlashVerifyScratch,
        layer_scratch: &mut MetalDFlashLayerMajorScratch,
        target_session: &mut MetalSession,
    ) -> Result<Vec<i32>, DFlashError> {
        encode_packed_verify_layer_major_inner(
            self.base,
            &self.head.target_layer_ids,
            tokens,
            start_position,
            verify_scratch,
            layer_scratch,
            target_session,
            None,
        )
    }

    /// Debug variant of layer-major `packed_verify` that ALSO writes
    /// `[N, V]` raw logits to `dbg_scratch.debug_logits` for the H5.3a
    /// cosine gate (G1 full). Production code MUST NOT call this — the
    /// extra `[N, V]` copy is 15.9 MB per outer step at 27B and negates
    /// the entire point of the GPU-argmax design.
    pub fn packed_verify_with_logits(
        &self,
        tokens: &[i32],
        start_position: u32,
        dbg_scratch: &mut MetalDFlashDebugScratch,
        layer_scratch: &mut MetalDFlashLayerMajorScratch,
        target_session: &mut MetalSession,
    ) -> Result<Vec<i32>, DFlashError> {
        let MetalDFlashDebugScratch {
            verify,
            debug_logits,
        } = dbg_scratch;
        encode_packed_verify_layer_major_inner(
            self.base,
            &self.head.target_layer_ids,
            tokens,
            start_position,
            verify,
            layer_scratch,
            target_session,
            Some(debug_logits),
        )
    }

    /// Token-major (naive H5.3a) packed verify path. Preserved as the
    /// correctness oracle for the layer-major rewrite — codex Q4 from
    /// the H5.3b.4-5 partner session: "ship both, default to layer-
    /// major, tests run BOTH on identical inputs and compare."
    ///
    /// Production callers should use `packed_verify` (layer-major) for
    /// throughput. This path is for bisecting / regression-locking the
    /// layer-major impl to the proven token-major one.
    pub fn packed_verify_token_major(
        &self,
        tokens: &[i32],
        start_position: u32,
        scratch: &mut MetalDFlashVerifyScratch,
        target_session: &mut MetalSession,
    ) -> Result<Vec<i32>, DFlashError> {
        encode_packed_verify_inner(
            self.base,
            &self.head.target_layer_ids,
            tokens,
            start_position,
            scratch,
            target_session,
        )
    }

    /// Token-major debug variant — preserved as oracle for the
    /// layer-major `_with_logits` variant.
    pub fn packed_verify_token_major_with_logits(
        &self,
        tokens: &[i32],
        start_position: u32,
        dbg_scratch: &mut MetalDFlashDebugScratch,
        target_session: &mut MetalSession,
    ) -> Result<Vec<i32>, DFlashError> {
        encode_packed_verify_with_logits_inner(
            self.base,
            &self.head.target_layer_ids,
            tokens,
            start_position,
            dbg_scratch,
            target_session,
        )
    }

    // =====================================================================
    // restore_after_partial_accept — H5.3a rollback primitive
    // =====================================================================
    //
    // Folded into H5.3a from H5.4 per plan rev 4 — the checkpoint
    // contract isn't testable without restore (gate G3 needs it). This
    // is the SECOND headline H5.3a feature.
    //
    // ## Indexing semantics — pinned brutally clearly per codex review.
    //
    // After `packed_verify(tokens[0..N], start_position)`, for each
    // n ∈ [0, N), the checkpoint slot `gdn_ckpt_slot(k, n)` holds
    // "state-after-token-n for GDN layer k". Same for `conv_ckpt_slot`.
    //
    // `restore_after_partial_accept(n_keep, start_position)` rolls the
    // session back to "as if exactly `n_keep` tokens were processed
    // starting at start_position." Concretely:
    //
    //   * `gdn_state[k]` ← `gdn_ckpt_slot(k, n_keep - 1)`
    //   * `gdn_conv[k]`  ← `conv_ckpt_slot(k, n_keep - 1)`
    //   * `kv_n_pos[i]`  := `start_position + n_keep`
    //   * KV slot bytes at [start_position + n_keep, ...) physically
    //     remain but become unreachable (next verify overwrites).
    //
    // ## Why `n_keep` instead of `n_accepted`?
    //
    // `n_accepted` is overloaded in the plan (acceptance count over
    // DRAFTS, not over the verify batch). The verify batch is
    // `[carry_tok, draft_0, draft_1, ..., draft_{D-1}]` of length N=D+1.
    // The carry is ALWAYS committed (it was selected in the previous
    // step's bonus); accepted drafts append to it. So:
    //
    //   tokens kept after this batch = 1 (carry) + n_accepted (drafts)
    //   n_keep                       = 1 + n_accepted ∈ [1, N]
    //
    // n_keep can never be 0 (the carry is always processed). n_keep=1
    // means full reject (carry only, no drafts accepted). n_keep=N
    // means full accept (carry + all D drafts) — the rollback is a
    // no-op but the call must be safe.
    //
    // Codex review: pin the API to `n_keep` so callers can't confuse
    // "accepted drafts" with "tokens to retain." The conversion lives
    // in the H5.5 outer loop, not here.
    pub fn restore_after_partial_accept(
        &self,
        scratch: &MetalDFlashVerifyScratch,
        n_keep: u32,
        start_position: u32,
        target_session: &mut MetalSession,
    ) -> Result<(), DFlashError> {
        encode_restore_after_partial_accept_inner(
            self.base,
            scratch,
            n_keep,
            start_position,
            target_session,
        )
    }
}

/// H5.3a packed verify forward — low-level entrypoint that takes
/// everything explicitly. `DFlashDecoder::packed_verify` is the
/// production wrapper; this exists so unit tests can exercise the
/// packed-verify algorithm without standing up a full DFlash drafter
/// (which requires real drafter GGUF weights). Signature mirrors
/// `MetalForward::single_token_with_multi_hidden`'s style.
pub(crate) fn encode_packed_verify_inner(
    base: &MetalForward<'_>,
    target_layer_ids: &[u32],
    tokens: &[i32],
    start_position: u32,
    scratch: &mut MetalDFlashVerifyScratch,
    target_session: &mut MetalSession,
) -> Result<Vec<i32>, DFlashError> {
    encode_packed_verify_inner_impl(
        base,
        target_layer_ids,
        tokens,
        start_position,
        scratch,
        target_session,
        None,
    )
}

/// Like `encode_packed_verify_inner` but ALSO writes raw `[N, V]` logits
/// to `dbg_scratch.debug_logits`. For correctness/cosine gate use only;
/// production paths must use `encode_packed_verify_inner` (no extra
/// vocab-sized buffer touched per token).
pub fn encode_packed_verify_with_logits_inner(
    base: &MetalForward<'_>,
    target_layer_ids: &[u32],
    tokens: &[i32],
    start_position: u32,
    dbg_scratch: &mut MetalDFlashDebugScratch,
    target_session: &mut MetalSession,
) -> Result<Vec<i32>, DFlashError> {
    // Validate the debug-logits buffer matches verify scratch dims.
    let n = dbg_scratch.verify.n;
    let v = base.model.arch.vocab_size as u64;
    if dbg_scratch.debug_logits.shape != vec![n as u64, v] {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify_with_logits",
            detail: format!(
                "debug_logits.shape={:?} != [N={n}, V={v}]",
                dbg_scratch.debug_logits.shape
            ),
        }));
    }
    // Borrow split: take &mut to verify scratch (for mutation) and
    // an immutable handle to debug_logits (for the logits scatter dst).
    // We can't borrow both fields of dbg_scratch at once via two &mut,
    // so split the borrow with explicit field access.
    let MetalDFlashDebugScratch {
        verify,
        debug_logits,
    } = dbg_scratch;
    encode_packed_verify_inner_impl(
        base,
        target_layer_ids,
        tokens,
        start_position,
        verify,
        target_session,
        Some(debug_logits),
    )
}

fn encode_packed_verify_inner_impl(
    base: &MetalForward<'_>,
    target_layer_ids: &[u32],
    tokens: &[i32],
    start_position: u32,
    scratch: &mut MetalDFlashVerifyScratch,
    target_session: &mut MetalSession,
    debug_logits_dst: Option<&MetalTensor>,
) -> Result<Vec<i32>, DFlashError> {
    let arch = &base.model.arch;

    // -- Codex failure-mode guard wall: validate ALL dims at entry
    // because the slot helpers use debug_assert (no-op in release).
    // If any of these mismatch, downstream blits would silently
    // write to wrong offsets in production builds. Fail loudly.
    let n = scratch.n as usize;
    if tokens.len() != n {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify",
            detail: format!(
                "tokens.len()={} != scratch.n={n} (scratch was allocated for a different block_size)",
                tokens.len()
            ),
        }));
    }
    let k = target_layer_ids.len();
    if scratch.k_target_layers as usize != k {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify",
            detail: format!(
                "scratch.k_target_layers={} != target_layer_ids.len()={k} (scratch allocated for a different drafter)",
                scratch.k_target_layers
            ),
        }));
    }
    if scratch.hidden_size != arch.hidden_size as u64 {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify",
            detail: format!(
                "scratch.hidden_size={} != arch.hidden_size={} (wrong target model)",
                scratch.hidden_size, arch.hidden_size
            ),
        }));
    }
    let n_gdn_actual = base
        .model
        .blocks
        .iter()
        .filter(|b| matches!(b, MetalBlock::Gdn(_)))
        .count() as u32;
    if scratch.n_gdn_layers != n_gdn_actual {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify",
            detail: format!(
                "scratch.n_gdn_layers={} != model n_gdn={n_gdn_actual} (scratch allocated for different layer schedule)",
                scratch.n_gdn_layers
            ),
        }));
    }
    // session must have matching state buffer counts.
    if target_session.gdn_state.len() != n_gdn_actual as usize
        || target_session.gdn_conv.len() != n_gdn_actual as usize
    {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify",
            detail: format!(
                "session.gdn_state.len={} / gdn_conv.len={} != model n_gdn={n_gdn_actual}",
                target_session.gdn_state.len(),
                target_session.gdn_conv.len(),
            ),
        }));
    }
    // KV capacity must accommodate start_position + N positions.
    // Use checked addition (codex review: avoid silent wrap on
    // pathological start_position values).
    let last_pos = (start_position as usize).checked_add(n).ok_or_else(|| {
        DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify",
            detail: format!("start_position={start_position} + N={n} overflows usize"),
        })
    })?;
    if last_pos > target_session.kv_capacity {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify",
            detail: format!(
                "start_position={} + N={n} = {last_pos} > kv_capacity={}",
                start_position, target_session.kv_capacity
            ),
        }));
    }
    // CRITICAL — kv_n_pos == start_position contract.
    //
    // The session must already represent the prefix ending at
    // `start_position`: per-attn-layer KV slots [0, start_position)
    // are populated and `kv_n_pos[i] == start_position` for every
    // attn layer i. Without this guard, a stale or misaligned
    // session silently attends over the wrong KV prefix —
    // `encode_attn` reads `s.kv_n_pos[attn_i]` (NOT `position`) for
    // the attention length argument, so a session with
    // `kv_n_pos=99` going through `packed_verify(start_position=0,
    // N=4)` would: write to slot 0 (correct), then attn would
    // attend over keys [0..1] AT POSITION 0, but BEFORE that slot
    // 0's K/V is what we just scattered (correct) — actually it
    // reads `s.kv_n_pos[i] = position+1 = 1` after the scatter
    // (correct for the FIRST token). But for a primed session
    // (kv_n_pos=99 going in), we'd write to slot 0 (overwriting),
    // attn at n_pos=1 (correct for re-priming) — so the bug is the
    // primed prefix is silently DISCARDED, not corrupted.
    //
    // Either way: the user expected a continuation at
    // start_position; what they got was a fresh session at slot 0.
    // Fail loudly. Codex called this the biggest miss in the v0.57
    // foundation review.
    for (i, &kp) in target_session.kv_n_pos.iter().enumerate() {
        if kp != start_position as usize {
            return Err(DFlashError::Metal(MetalError::BadShape {
                kernel: "packed_verify",
                detail: format!(
                    "kv_n_pos[{i}]={kp} != start_position={start_position} \
                     (session does not represent the prefix at the requested \
                     start position; either prime the session up to \
                     start_position or call with start_position=0 on a \
                     fresh session)"
                ),
            }));
        }
    }
    // Validate every token id and target_layer_id.
    for (i, &t) in tokens.iter().enumerate() {
        if t < 0 || (t as u32) >= arch.vocab_size {
            return Err(DFlashError::BadToken(t, arch.vocab_size));
        }
        // (i was just for debug if we wanted it; unused.)
        let _ = i;
    }
    for &lid in target_layer_ids {
        if (lid as usize) >= base.model.blocks.len() {
            return Err(DFlashError::Metal(MetalError::BadShape {
                kernel: "packed_verify.target_layer_ids",
                detail: format!("layer id {lid} >= n_layer {}", base.model.blocks.len()),
            }));
        }
    }

    let h = arch.hidden_size as usize;

    // -- Stage all N token ids into packed_ids_buf at once. This is
    // the codex-Q7 mitigation made concrete: each per-token block n
    // reads its OWN slot via `scratch.token_slot(n)`, never sharing a
    // CPU-mutable scalar buffer with another block.
    //
    // Apple Metal sync contract for StorageModeShared (per
    // https://developer.apple.com/documentation/Metal/resource-synchronization
    // and https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared):
    //   * Host writes must complete BEFORE `cmd_buf.commit()` for the
    //     GPU to observe them. Writing here (BEFORE commit, BEFORE
    //     even opening the first encoder) is well within that contract.
    //   * Host MUST NOT mutate the buffer while the cmd buffer is in
    //     flight. We don't; the next host access is the verify_argmax
    //     readback after waitUntilCompleted.
    //   * GPU writes are visible to the host after waitUntilCompleted.
    // The earlier comment "visible once we open the command encoder"
    // was wrong; ordering is anchored at commit, not encoder open.
    unsafe {
        let p = scratch.packed_ids_buf.buffer.contents().as_ptr() as *mut i32;
        for (i, &t) in tokens.iter().enumerate() {
            *p.add(i) = t;
        }
    }

    // -- One command buffer for the entire packed verify.
    let cmd_buf = base.ctx.queue.commandBuffer().expect("command buffer");

    for n_idx in 0..n {
        let position_n = start_position + n_idx as u32;

        // ===== Per-token COMPUTE pass =====
        let enc = KernelEncoder::begin(&cmd_buf);

        // Embed: read token id from packed_ids_buf[n_idx] → session.x.
        let tok_slot = scratch.token_slot(n_idx as u32);
        encode_get_rows_f32(
            base.ctx,
            &enc,
            &base.model.token_embd,
            &tok_slot,
            &target_session.x,
            1,
            h,
        )?;

        // Per-block forward, capturing target hiddens inline.
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in base.model.blocks.iter().enumerate() {
            base.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position_n,
                target_session,
            )?;
            // After this block's residual #2, scatter session.x
            // into hidden_capture[k_idx, n_idx, :] if this is one
            // of the target layers.
            for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                if lid as usize == il {
                    // hidden_capture is stored as [N, K, H] (v0.71
                    // layout change — see hidden_capture_slot doc).
                    // Slot (k, n) at offset (n * K + k) * H.
                    let dst_slot = scratch.hidden_capture_slot(k_idx as u32, n_idx as u32);
                    let elem_off = (n_idx as u64 * scratch.k_target_layers as u64 + k_idx as u64)
                        * scratch.hidden_size;
                    debug_assert_eq!(
                        dst_slot.offset,
                        elem_off * std::mem::size_of::<f32>() as u64
                    );
                    encode_scatter_offset_f32(
                        base.ctx,
                        &enc,
                        &target_session.x,
                        &scratch.hidden_capture,
                        elem_off as usize,
                        h,
                    )?;
                }
            }
        }

        // Final RMSNorm + lm_head → session.logits (reused per token).
        encode_rms_norm_mul_f32(
            base.ctx,
            &enc,
            &target_session.x,
            &base.model.output_norm,
            &target_session.h,
            RMS_EPS,
        )?;
        encode_mat_vec_dispatch(
            base.ctx,
            &enc,
            &base.model.lm_head,
            &target_session.h,
            &target_session.logits,
            h,
            arch.vocab_size as usize,
        )?;

        // Debug-only: spill session.logits → debug_logits[n_idx, :]
        // for the H5.3a cosine gate (G1 full). This is the ONLY new
        // dispatch on the debug path. MUST happen before the next
        // token's lm_head writes session.logits, AND before/after
        // argmax (both read session.logits). We do it before argmax
        // so the scatter can overlap with argmax's reduce.
        //
        // Production (debug_logits_dst = None) skips this entirely;
        // no per-vocab CPU readback is created. The 15.9 MB anti-
        // regression assertion still holds.
        if let Some(dst) = debug_logits_dst {
            let elem_off = (n_idx as u64) * (arch.vocab_size as u64);
            encode_scatter_offset_f32(
                base.ctx,
                &enc,
                &target_session.logits,
                dst,
                elem_off as usize,
                arch.vocab_size as usize,
            )?;
        }

        // GPU argmax over session.logits → verify_argmax[n_idx].
        // We pass argmax_slot (a [1]-shaped view) as the destination;
        // n_rows=1 so argmax dispatches a single threadgroup.
        // Codex-Q5: this MUST run before token n+1's lm_head writes
        // session.logits.
        let argmax_dst = scratch.argmax_slot(n_idx as u32);
        encode_argmax_f32(
            base.ctx,
            &enc,
            &target_session.logits,
            &argmax_dst,
            1,
            arch.vocab_size as usize,
        )?;

        enc.end();

        // ===== Per-token BLIT pass (GDN + conv state checkpoints) =====
        //
        // Codex-Q2 design Y: batch all checkpoint copies for token
        // n_idx into one blit pass. Each `gdn_state[k]` / `gdn_conv[k]`
        // was mutated by exactly one block above (GDN layer k); after
        // all blocks complete, those buffers contain the post-token-n
        // state we want to checkpoint.
        let blit = BlitEncoder::begin(&cmd_buf);
        for k in 0..n_gdn_actual {
            let ssm_dst = scratch.gdn_ckpt_slot(k, n_idx as u32);
            blit.copy_tensor(&target_session.gdn_state[k as usize], &ssm_dst);
            let conv_dst = scratch.conv_ckpt_slot(k, n_idx as u32);
            blit.copy_tensor(&target_session.gdn_conv[k as usize], &conv_dst);
        }
        blit.end();
    }

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    // Read back verify_argmax (only N i32 values; trivial).
    let mut out = vec![0i32; n];
    unsafe {
        let src = scratch.verify_argmax.buffer.contents().as_ptr() as *const i32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }
    Ok(out)
}

/// H5.3a rollback primitive — low-level entrypoint that takes
/// everything explicitly. `DFlashDecoder::restore_after_partial_accept`
/// is the production wrapper; this exists so unit tests can exercise
/// the rollback algorithm without standing up a full DFlash drafter.
///
/// See `DFlashDecoder::restore_after_partial_accept` for the indexing
/// spec and `n_keep` semantics — they are identical.
///
/// Algorithm:
///   1. Validate dims (n_keep ∈ [1, N], scratch matches model, etc.).
///   2. Open one MTLCommandBuffer + BlitEncoder.
///   3. For each GDN layer k:
///        gdn_state[k] ← gdn_ckpt_slot(k, n_keep - 1)
///        gdn_conv[k]  ← conv_ckpt_slot(k, n_keep - 1)
///   4. End blit encoder, commit, wait.
///   5. CPU update: kv_n_pos[i] := start_position + n_keep for every
///      attn layer i.
///
/// Step 5 is host-side because `MetalSession::kv_n_pos` is a
/// `Vec<usize>` on the host (matches the existing `encode_attn`
/// pattern where it's read at encode time, not GPU-side). KV slot
/// bytes at [start_position + n_keep, ...) physically remain but
/// become unreachable; next packed_verify call overwrites them.
// =============================================================================
// encode_packed_verify_layer_major_inner — H5.3b.4-5 layer-major path
// =============================================================================
//
// Per H5 plan rev 6 §H5.3b.4-5 (codex layer-major partner session, v0.65):
// rewrite packed_verify so that each layer's batched-batchable kernels
// (norms, projections, FFN) run ONCE across all N tokens, sharing weight
// loads. GDN/attn mixers stay sequential per token (recurrent state can't
// pack along time).
//
// Structural invariants (from codex partner session):
//   * `MetalDFlashLayerMajorScratch` owns N-wide activation buffers
//     (`x_pack`, `h_pack`, `mixer_out_pack`, `attn_*_pack`, `ffn_*_pack`).
//   * `MetalDFlashVerifyScratch` continues to own outputs + checkpoints
//     (`packed_ids_buf`, `verify_argmax`, `hidden_capture`, `gdn_ckpt`,
//      `conv_ckpt`).
//   * GDN/conv per-token checkpoint blits are inlined per-N inside the
//     mixer inner loop (Option A from codex Q1). Each GDN-layer iter:
//     compute → blit → compute. ~1536 transitions per outer step total.
//   * Attn `o_proj` is BATCHED mat-mat across N (Q4_K) — codex Q2.
//   * K/V projection fusion deferred — codex Q3.
//   * Dtype dispatch INSIDE this function (Q4_K vs F32 paths) — codex Q5.
//   * Token-major path stays as the oracle (Q4); this is the new
//     production path. The two are compared bit-exact in tests.
//
// Reuses `MetalForward::encode_gdn` and `MetalForward::encode_attn` for
// the per-token mixer code by writing per-row inputs into the existing
// `MetalSession` single-token scratch (`s.h`), running the unchanged
// mixer, then copying the per-row output back into `mixer_out_pack[n]`.
// 2 extra row-copy dispatches per token per mixer-bearing-layer; cheap
// (~20 KB per copy) and avoids re-implementing the mixer math.
//
// Mat-mat output layout note (verified bit-equivalent in H5.3b.0): the
// lifted `kernel_mul_mm_q4_K_f32` writes `dst[r + c*M]` which is byte-
// identical to row-major `[N, n_out]`. Downstream consumers
// (silu_mul, residual_add, chained mat-mat with this output as srcB)
// work without any transpose. The intermediate-layer cosine tests
// added in this phase verify this for every per-layer pack buffer.
pub fn encode_packed_verify_layer_major_inner(
    base: &MetalForward<'_>,
    target_layer_ids: &[u32],
    tokens: &[i32],
    start_position: u32,
    verify_scratch: &mut MetalDFlashVerifyScratch,
    layer_scratch: &mut MetalDFlashLayerMajorScratch,
    target_session: &mut MetalSession,
    debug_logits_dst: Option<&MetalTensor>,
) -> Result<Vec<i32>, DFlashError> {
    let arch = &base.model.arch;
    let n = verify_scratch.n as usize;
    let h = arch.hidden_size as usize;
    let f = arch.intermediate_size as usize;
    let v = arch.vocab_size as usize;

    // -- guard wall (mirrors token-major; same bug class) --
    if tokens.len() != n {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify_layer_major",
            detail: format!("tokens.len()={} != verify_scratch.n={n}", tokens.len()),
        }));
    }
    let k_target = target_layer_ids.len();
    if verify_scratch.k_target_layers as usize != k_target {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify_layer_major",
            detail: format!(
                "verify_scratch.k_target_layers={} != target_layer_ids.len()={k_target}",
                verify_scratch.k_target_layers
            ),
        }));
    }
    if verify_scratch.hidden_size != h as u64 {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify_layer_major",
            detail: format!(
                "verify_scratch.hidden_size={} != arch.hidden_size={h}",
                verify_scratch.hidden_size
            ),
        }));
    }
    if layer_scratch.n != verify_scratch.n
        || layer_scratch.hidden_size != verify_scratch.hidden_size
    {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify_layer_major",
            detail: format!(
                "layer_scratch ({} N × {} H) does not match verify_scratch ({} N × {} H)",
                layer_scratch.n,
                layer_scratch.hidden_size,
                verify_scratch.n,
                verify_scratch.hidden_size
            ),
        }));
    }
    let n_gdn_actual = base
        .model
        .blocks
        .iter()
        .filter(|b| matches!(b, MetalBlock::Gdn(_)))
        .count() as u32;
    if verify_scratch.n_gdn_layers != n_gdn_actual {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify_layer_major",
            detail: format!(
                "verify_scratch.n_gdn_layers={} != model n_gdn={n_gdn_actual}",
                verify_scratch.n_gdn_layers
            ),
        }));
    }
    let last_pos = (start_position as usize).checked_add(n).ok_or_else(|| {
        DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify_layer_major",
            detail: format!("start_position={start_position} + N={n} overflows usize"),
        })
    })?;
    if last_pos > target_session.kv_capacity {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify_layer_major",
            detail: format!(
                "start_position + N = {last_pos} > kv_capacity={}",
                target_session.kv_capacity
            ),
        }));
    }
    for (i, &kp) in target_session.kv_n_pos.iter().enumerate() {
        if kp != start_position as usize {
            return Err(DFlashError::Metal(MetalError::BadShape {
                kernel: "packed_verify_layer_major",
                detail: format!("kv_n_pos[{i}]={kp} != start_position={start_position}"),
            }));
        }
    }
    for &t in tokens.iter() {
        if t < 0 || (t as u32) >= arch.vocab_size {
            return Err(DFlashError::BadToken(t, arch.vocab_size));
        }
    }
    for &lid in target_layer_ids {
        if (lid as usize) >= base.model.blocks.len() {
            return Err(DFlashError::Metal(MetalError::BadShape {
                kernel: "packed_verify_layer_major.target_layer_ids",
                detail: format!("layer id {lid} >= n_layer {}", base.model.blocks.len()),
            }));
        }
    }
    if let Some(dst) = debug_logits_dst {
        if dst.shape != vec![n as u64, v as u64] {
            return Err(DFlashError::Metal(MetalError::BadShape {
                kernel: "packed_verify_layer_major.debug_logits_dst",
                detail: format!("expected [{n}, {v}], got {:?}", dst.shape),
            }));
        }
    }

    // -- Stage all N token ids into packed_ids_buf (host write before
    //    cmd_buf.commit; per Apple StorageModeShared contract). --
    unsafe {
        let p = verify_scratch.packed_ids_buf.buffer.contents().as_ptr() as *mut i32;
        for (i, &t) in tokens.iter().enumerate() {
            *p.add(i) = t;
        }
    }

    // -- One MTLCommandBuffer for the whole forward. We open and close
    //    compute encoders multiple times (alternating with blit encoders
    //    around the GDN-layer per-token checkpoint writes). --
    let cmd_buf = base.ctx.queue.commandBuffer().expect("command buffer");

    // === Phase 1: batched embed of all N tokens into x_pack [N, H]. ===
    {
        let enc = KernelEncoder::begin(&cmd_buf);
        encode_get_rows_f32(
            base.ctx,
            &enc,
            &base.model.token_embd,
            &verify_scratch.packed_ids_buf,
            &layer_scratch.x_pack,
            n,
            h,
        )?;
        enc.end();
    }

    // === Phase 2: layer loop. ===
    let mut gdn_idx = 0usize;
    let mut attn_idx = 0usize;
    for (il, block) in base.model.blocks.iter().enumerate() {
        // 2a: pre-mixer norm BATCHED across all N tokens. The kernel
        //     `kernel_rms_norm_batched_f32` already supports per-row
        //     RMSNorm with shared weight; we treat (n_heads = N,
        //     head_dim = H) which gives one RMSNorm per token row.
        let attn_norm = match block {
            MetalBlock::Gdn(g) => &g.attn_norm,
            MetalBlock::Attn(a) => &a.attn_norm,
        };
        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_rms_norm_batched_f32(
                base.ctx,
                &enc,
                &layer_scratch.x_pack,
                attn_norm,
                &layer_scratch.h_pack,
                n,
                h,
                RMS_EPS,
            )?;
            enc.end();
        }

        // 2b: mixer. Two paths.
        //
        //   GDN: per-token inner loop (recurrent; can't batch over
        //        time). Per-N: copy h_pack[n] → s.h, run encode_gdn,
        //        copy s.mixer_out → mixer_out_pack[n], then close
        //        compute encoder + blit gdn_state[gi] / gdn_conv[gi]
        //        → ckpt_slot(gi, n) + reopen compute encoder.
        //   Attn: per-token sequential — KV append + attn-v4 are
        //        per-token. No checkpoint writes (KV cache is a
        //        slot-indexed accumulator, not a recurrent state we
        //        roll back via blit). o_proj BATCHED via mat-mat.
        match block {
            MetalBlock::Gdn(g) => {
                let gi = gdn_idx;
                gdn_idx += 1;
                // v0.73a.1: GDN projection batching eligibility. Mirrors
                // the FFN dtype dispatch pattern at 2f. We batch
                // in_proj_qkv, in_proj_z, and out_proj as mat-mat across
                // N=16 when each is in {Q4_K, Q5_K, Q6_K} (codex review:
                // mirror FFN's eligibility set, not Q4_K-only). Production
                // 27B Q4_K_M is Q6_K/Q4_K/Q5_K respectively; F32 oracle
                // (0.8B) falls back per-token. beta_proj and alpha_proj
                // stay per-token mat-vec because production stores them
                // as F32 [hidden, n_v=48] — small, mat-mat dispatch
                // overhead exceeds BW savings (see docs/H5-DFLASH.md
                // rev 10).
                let gdn_mat_mat_eligible = |dtype: GgmlType| {
                    matches!(dtype, GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K)
                };
                let gdn_batched = gdn_mat_mat_eligible(g.in_proj_qkv.dtype)
                    && gdn_mat_mat_eligible(g.in_proj_z.dtype)
                    && gdn_mat_mat_eligible(g.out_proj.dtype);

                if gdn_batched {
                    // Step A: two batched front-end projections across all N tokens.
                    // Reads h_pack [N, H], writes gdn_qkv_pack [N, conv_dim] and
                    // gdn_z_pack [N, v_dim]. NR1=16 fast path fires automatically
                    // when N == 16 (current DFlash block size).
                    let n_k_u = arch.gdn_n_k_heads as usize;
                    let n_v = arch.gdn_n_v_heads as usize;
                    let head_dim_u = arch.gdn_head_dim as usize;
                    let conv_dim = (2 * n_k_u + n_v) * head_dim_u;
                    let v_dim = n_v * head_dim_u;
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &g.in_proj_qkv,
                            &layer_scratch.h_pack,
                            &layer_scratch.gdn_qkv_pack,
                            h,
                            conv_dim,
                            n,
                        )?;
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &g.in_proj_z,
                            &layer_scratch.h_pack,
                            &layer_scratch.gdn_z_pack,
                            h,
                            v_dim,
                            n,
                        )?;
                        enc.end();
                    }

                    // Per-token loop (recurrence is inherently sequential).
                    // Step B: per-token alpha/beta (F32 mat-vec; small) +
                    // post-projection recurrence body (encode_gdn_tail) +
                    // checkpoint blit.
                    let alpha_handle = target_session.gdn_alpha.clone();
                    let beta_handle = target_session.gdn_beta.clone();
                    for n_idx in 0..n {
                        // Compute pass.
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            // Per-row view of h_pack for the F32 beta/alpha mat-vecs.
                            let h_n = layer_scratch
                                .h_pack
                                .view_subrange((n_idx * h) as u64, vec![h as u64]);
                            // beta_proj (F32) → sigmoid → s.gdn_beta.
                            encode_mat_vec_dispatch(
                                base.ctx,
                                &enc,
                                &g.beta_proj,
                                &h_n,
                                &target_session.gdn_b,
                                h,
                                n_v,
                            )?;
                            encode_sigmoid_f32(
                                base.ctx,
                                &enc,
                                &target_session.gdn_b,
                                &target_session.gdn_beta,
                            )?;
                            // alpha_proj (F32) → fused softplus(a+dt_bias)*a_log → s.gdn_alpha.
                            encode_mat_vec_dispatch(
                                base.ctx,
                                &enc,
                                &g.alpha_proj,
                                &h_n,
                                &target_session.gdn_a,
                                h,
                                n_v,
                            )?;
                            encode_gdn_alpha_chain_f32(
                                base.ctx,
                                &enc,
                                &target_session.gdn_a,
                                &g.dt_bias,
                                &g.a_log,
                                &target_session.gdn_alpha,
                            )?;
                            // Per-row views of the batched pack buffers (zero-copy
                            // F32 view_subrange — F32 is supported, no super-block
                            // alignment needed).
                            let qkv_n = layer_scratch
                                .gdn_qkv_pack
                                .view_subrange((n_idx * conv_dim) as u64, vec![conv_dim as u64]);
                            let z_n = layer_scratch
                                .gdn_z_pack
                                .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                            let normed_n = layer_scratch
                                .gdn_normed_pack
                                .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                            base.encode_gdn_tail(
                                &enc,
                                g,
                                gi,
                                target_session,
                                &qkv_n,
                                &z_n,
                                &alpha_handle,
                                &beta_handle,
                                &normed_n,
                            )?;
                            enc.end();
                        }
                        // Blit pass: snapshot post-token-n state into ckpt slots.
                        {
                            let blit = BlitEncoder::begin(&cmd_buf);
                            let ssm_dst = verify_scratch.gdn_ckpt_slot(gi as u32, n_idx as u32);
                            blit.copy_tensor(&target_session.gdn_state[gi], &ssm_dst);
                            let conv_dst = verify_scratch.conv_ckpt_slot(gi as u32, n_idx as u32);
                            blit.copy_tensor(&target_session.gdn_conv[gi], &conv_dst);
                            blit.end();
                        }
                    }

                    // Step C: batched out_proj across all N. Reads
                    // gdn_normed_pack [N, v_dim], writes mixer_out_pack [N, H].
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &g.out_proj,
                            &layer_scratch.gdn_normed_pack,
                            &layer_scratch.mixer_out_pack,
                            v_dim,
                            h,
                            n,
                        )?;
                        enc.end();
                    }
                } else {
                    // F32 oracle / mixed-dtype fall-through: existing per-token
                    // encode_gdn pattern, unchanged. Keeps the 0.8B oracle path
                    // bit-exact.
                    for n_idx in 0..n {
                        // Compute pass: stage row, run mixer, capture row.
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_copy_offset_f32(
                                base.ctx,
                                &enc,
                                &layer_scratch.h_pack,
                                n_idx * h,
                                &target_session.h,
                                h,
                            )?;
                            base.encode_gdn(&enc, g, gi, target_session)?;
                            encode_scatter_offset_f32(
                                base.ctx,
                                &enc,
                                &target_session.mixer_out,
                                &layer_scratch.mixer_out_pack,
                                n_idx * h,
                                h,
                            )?;
                            enc.end();
                        }
                        // Blit pass: snapshot post-token-n state into ckpt slots.
                        {
                            let blit = BlitEncoder::begin(&cmd_buf);
                            let ssm_dst = verify_scratch.gdn_ckpt_slot(gi as u32, n_idx as u32);
                            blit.copy_tensor(&target_session.gdn_state[gi], &ssm_dst);
                            let conv_dst = verify_scratch.conv_ckpt_slot(gi as u32, n_idx as u32);
                            blit.copy_tensor(&target_session.gdn_conv[gi], &conv_dst);
                            blit.end();
                        }
                    }
                }
            }
            MetalBlock::Attn(a) => {
                let ai = attn_idx;
                attn_idx += 1;
                // v0.73c.1: attn projection batching, mirrors v0.73a.1 GDN
                // restructure. Production 27B Q4_K_M attn projections are
                // ALL Q4_K (q gated, k, v, output). Batch them as mat-mat
                // across N=16 in step A/C; per-token loop only does
                // RoPE + KV-scatter + attn-v4 + gate-sigmoid-mul (which
                // we batch into step C as a flat elementwise pair).
                let attn_mat_mat_eligible = |dtype: GgmlType| {
                    matches!(dtype, GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K)
                };
                let attn_batched = attn_mat_mat_eligible(a.q.dtype)
                    && attn_mat_mat_eligible(a.k.dtype)
                    && attn_mat_mat_eligible(a.v.dtype)
                    && attn_mat_mat_eligible(a.o.dtype);

                if attn_batched {
                    let arch = &base.model.arch;
                    let head_dim = arch.attn_head_dim as usize;
                    let n_q = arch.n_q_heads as usize;
                    let n_kv = arch.n_kv_heads as usize;
                    let q_dim = n_q * head_dim;
                    let kv_dim = n_kv * head_dim;
                    let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;

                    // Step A: batched front-end Q (gated) / K / V projections,
                    // batched Q-norm and K-norm. One encoder per layer.
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        // Q gated (Q + gate interleaved per head).
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &a.q,
                            &layer_scratch.h_pack,
                            &layer_scratch.attn_q_full_pack,
                            h,
                            2 * q_dim,
                            n,
                        )?;
                        // Split q + gate into separate packs. The
                        // single-token kernel deinterleaves per-head;
                        // pass `n_heads = N * n_q` so it processes all
                        // N rows in one dispatch (per-head layout
                        // repeats identically across rows).
                        encode_split_q_gate_f32(
                            base.ctx,
                            &enc,
                            &layer_scratch.attn_q_full_pack,
                            &layer_scratch.attn_q_pack,
                            &layer_scratch.attn_gate_pack,
                            n * n_q,
                            head_dim,
                        )?;
                        // K, V projections.
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &a.k,
                            &layer_scratch.h_pack,
                            &layer_scratch.attn_k_now_pack,
                            h,
                            kv_dim,
                            n,
                        )?;
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &a.v,
                            &layer_scratch.h_pack,
                            &layer_scratch.attn_v_now_pack,
                            h,
                            kv_dim,
                            n,
                        )?;
                        // Q-norm (per-head); n_heads = N * n_q.
                        encode_rms_norm_batched_f32(
                            base.ctx,
                            &enc,
                            &layer_scratch.attn_q_pack,
                            &a.q_norm,
                            &layer_scratch.attn_q_normed_pack,
                            n * n_q,
                            head_dim,
                            RMS_EPS,
                        )?;
                        // K-norm (per-head); n_heads = N * n_kv.
                        encode_rms_norm_batched_f32(
                            base.ctx,
                            &enc,
                            &layer_scratch.attn_k_now_pack,
                            &a.k_norm,
                            &layer_scratch.attn_k_normed_pack,
                            n * n_kv,
                            head_dim,
                            RMS_EPS,
                        )?;
                        enc.end();
                    }

                    // Per-token loop (KV append + softmax are inherently
                    // sequential per token; attn-v4 sees a different
                    // n_pos for each token and writes to a different
                    // KV cache slot).
                    for n_idx in 0..n {
                        let position_n = start_position + n_idx as u32;
                        let q_normed_n = layer_scratch
                            .attn_q_normed_pack
                            .view_subrange((n_idx * q_dim) as u64, vec![q_dim as u64]);
                        let k_normed_n = layer_scratch
                            .attn_k_normed_pack
                            .view_subrange((n_idx * kv_dim) as u64, vec![kv_dim as u64]);
                        let v_now_n = layer_scratch
                            .attn_v_now_pack
                            .view_subrange((n_idx * kv_dim) as u64, vec![kv_dim as u64]);
                        let attn_o_n = layer_scratch
                            .attn_o_pack
                            .view_subrange((n_idx * q_dim) as u64, vec![q_dim as u64]);
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            // RoPE on this row of Q and K.
                            encode_rope_neox_f32(
                                base.ctx,
                                &enc,
                                &q_normed_n,
                                n_q,
                                head_dim,
                                n_rot,
                                position_n,
                                arch.rope_theta,
                            )?;
                            encode_rope_neox_f32(
                                base.ctx,
                                &enc,
                                &k_normed_n,
                                n_kv,
                                head_dim,
                                n_rot,
                                position_n,
                                arch.rope_theta,
                            )?;
                            // KV scatter into F16 cache slot.
                            encode_scatter_offset_f32_to_f16_kv(
                                base.ctx,
                                &enc,
                                &k_normed_n,
                                &v_now_n,
                                &target_session.kv_k[ai],
                                &target_session.kv_v[ai],
                                (position_n as usize) * kv_dim,
                                kv_dim,
                            )?;
                            target_session.kv_n_pos[ai] = position_n as usize + 1;

                            // Fused attn-v4 (or naive fallback for non-matching shapes).
                            const V4_HEAD_DIM: usize = 256;
                            const V4_GROUP: usize = 6;
                            let use_v4 = head_dim == V4_HEAD_DIM && n_q == n_kv * V4_GROUP;
                            if use_v4 {
                                let nwg =
                                    crate::metal::attn_v4_choose_nwg(target_session.kv_n_pos[ai]);
                                let _tile_c = crate::metal::attn_v4_choose_tile_c(
                                    target_session.kv_n_pos[ai],
                                );
                                let tile_c = crate::metal::attn_v4_choose_tile_c(
                                    target_session.kv_n_pos[ai],
                                );
                                crate::metal::encode_attn_decode_v4_f32(
                                    base.ctx,
                                    &enc,
                                    &q_normed_n,
                                    &target_session.kv_k[ai],
                                    &target_session.kv_v[ai],
                                    &target_session.attn_v4_o_partial,
                                    &target_session.attn_v4_ml_partial,
                                    &attn_o_n,
                                    n_q,
                                    n_kv,
                                    head_dim,
                                    target_session.kv_n_pos[ai],
                                    nwg,
                                    tile_c,
                                )?;
                            } else {
                                crate::metal::encode_attn_decode_f16kv_f32(
                                    base.ctx,
                                    &enc,
                                    &q_normed_n,
                                    &target_session.kv_k[ai],
                                    &target_session.kv_v[ai],
                                    &attn_o_n,
                                    n_q,
                                    n_kv,
                                    head_dim,
                                    target_session.kv_n_pos[ai],
                                )?;
                            }
                            enc.end();
                        }
                    }

                    // Step C: gate-sigmoid + mul (flat elementwise on N*q_dim)
                    // followed by batched o_proj mat-mat. One encoder per layer.
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        // attn_gate_pack -> sigmoid into a scratch view of
                        // attn_q_pack (no longer needed; q_pack is dead after
                        // attn-v4). Same in-place reuse pattern as the
                        // single-token encode_attn (line 1399 of metal_forward.rs).
                        encode_sigmoid_f32(
                            base.ctx,
                            &enc,
                            &layer_scratch.attn_gate_pack,
                            &layer_scratch.attn_q_pack,
                        )?;
                        // attn_o_pack *= sigmoid(gate_pack), elementwise on N*q_dim.
                        crate::metal::encode_mul_f32(
                            base.ctx,
                            &enc,
                            &layer_scratch.attn_o_pack,
                            &layer_scratch.attn_q_pack,
                            &layer_scratch.attn_o_pack,
                        )?;
                        // Batched O projection: attn_o_pack [N, q_dim] -> mixer_out_pack [N, H].
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &a.o,
                            &layer_scratch.attn_o_pack,
                            &layer_scratch.mixer_out_pack,
                            q_dim,
                            h,
                            n,
                        )?;
                        enc.end();
                    }
                } else {
                    // F32 oracle / mixed-dtype fallback: existing per-token
                    // encode_attn pattern, unchanged. Keeps the 0.8B oracle
                    // path bit-exact and any future non-Q4_K attn weight
                    // working.
                    for n_idx in 0..n {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_copy_offset_f32(
                            base.ctx,
                            &enc,
                            &layer_scratch.h_pack,
                            n_idx * h,
                            &target_session.h,
                            h,
                        )?;
                        let position_n = start_position + n_idx as u32;
                        base.encode_attn(&enc, a, ai, position_n, target_session)?;
                        encode_scatter_offset_f32(
                            base.ctx,
                            &enc,
                            &target_session.mixer_out,
                            &layer_scratch.mixer_out_pack,
                            n_idx * h,
                            h,
                        )?;
                        enc.end();
                    }
                }
            }
        }

        // 2c: residual #1 — x_pack += mixer_out_pack (batched
        //     elementwise; encode_add_inplace_f32 just walks the flat
        //     N*H element count).
        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_add_inplace_f32(
                base.ctx,
                &enc,
                &layer_scratch.x_pack,
                &layer_scratch.mixer_out_pack,
            )?;
            enc.end();
        }

        // **v0.74.4 capture-point fix**: hidden_capture moved from
        // here (after residual #1, before FFN) to AFTER residual #2
        // (after FFN), matching:
        //   * `Forward::single_token_capture_layers` (forward.rs:211),
        //     the canonical CPU oracle for H5
        //   * `MetalForward::single_token_with_multi_hidden`
        //     (metal_forward.rs:534) — used for prefill bootstrap
        //   * `encode_packed_verify_inner_impl` token-major path
        //     (metal_dflash.rs:1532)
        //
        // The pre-fix layer-major path captured a DIFFERENT residual
        // stream snapshot than every other path. Hidden_capture feeds
        // the DFlash drafter's cross-context conditioning via dflash_fc;
        // the bug was latent because (a) the layer-major-vs-token-major
        // 27B test only compares LOGITS (which both paths compute from
        // the LAST layer's full residual #2 regardless of capture
        // timing), and (b) greedy equivalence vs DFlash=off held even
        // with mixed capture points in target_ctx_stacked (prefill
        // columns from the canonical post-FFN snapshot, decode columns
        // from the buggy pre-FFN snapshot). External code review
        // (codex pressure-test for v0.75 prefill) caught this. The
        // alpha measurements at all measured contexts (1.005x default,
        // 1.342x at 181, 1.482x at 363) are the floor — fix should
        // make them slightly better since the drafter now sees a
        // consistent input across prefill and decode.
        //
        // Capture inserted AFTER 2g (residual #2). Search 2d-fix below.

        // 2e: post-mixer norm BATCHED.
        let post_norm = match block {
            MetalBlock::Gdn(g) => &g.post_attn_norm,
            MetalBlock::Attn(a) => &a.post_attn_norm,
        };
        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_rms_norm_batched_f32(
                base.ctx,
                &enc,
                &layer_scratch.x_pack,
                post_norm,
                &layer_scratch.h_pack,
                n,
                h,
                RMS_EPS,
            )?;
            enc.end();
        }

        // 2f: SwiGLU FFN. Dtype dispatch (codex Q5) — the WIN.
        //   Q4_K weights → batched mat-mat (mat_mat_q4_k_f32) writing
        //                  ffn_gate_pack [N, F] then ffn_up_pack [N, F]
        //                  row-major (= mat-mat output bit-equivalent),
        //                  then silu_mul on flat N*F elements,
        //                  then mat-mat ffn_down → ffn_out_pack [N, H].
        //   F32 weights   → per-token mat-vec loop using the existing
        //                  fused encode_block path is wasteful for the
        //                  layer-major case; just call the existing
        //                  per-token encode_mat_vec_dispatch in a loop.
        //                  At 0.8B sizes this is still substantial weight
        //                  re-read but it's the F32 oracle path, NOT a
        //                  perf target.
        let (g_w, u_w, d_w) = match block {
            MetalBlock::Gdn(g) => (&g.ffn_gate, &g.ffn_up, &g.ffn_down),
            MetalBlock::Attn(a) => (&a.ffn_gate, &a.ffn_up, &a.ffn_down),
        };
        // Per-weight dtype dispatch (codex Q5: dispatch INSIDE the
        // function so rollback bugs stay localizable). Layer-major
        // wins via mat-mat for Q4_K and Q6_K weights; falls back to
        // the per-token mat-vec loop for F32 (0.8B oracle) or any
        // mixed/unsupported dtype.
        //
        // Production 27B Q4_K_M: ffn_gate / ffn_up are Q4_K, ffn_down
        // is Q6_K. Both legs hit the mat-mat fast path.
        let mat_mat_eligible = |dtype: GgmlType| matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K);
        let mat_mat_path = mat_mat_eligible(g_w.dtype)
            && mat_mat_eligible(u_w.dtype)
            && mat_mat_eligible(d_w.dtype);
        {
            let enc = KernelEncoder::begin(&cmd_buf);
            if mat_mat_path {
                encode_mat_mat_dispatch(
                    base.ctx,
                    &enc,
                    g_w,
                    &layer_scratch.h_pack,
                    &layer_scratch.ffn_gate_pack,
                    h,
                    f,
                    n,
                )?;
                encode_mat_mat_dispatch(
                    base.ctx,
                    &enc,
                    u_w,
                    &layer_scratch.h_pack,
                    &layer_scratch.ffn_up_pack,
                    h,
                    f,
                    n,
                )?;
                encode_silu_mul_f32(
                    base.ctx,
                    &enc,
                    &layer_scratch.ffn_gate_pack,
                    &layer_scratch.ffn_up_pack,
                    &layer_scratch.ffn_inner_pack,
                )?;
                encode_mat_mat_dispatch(
                    base.ctx,
                    &enc,
                    d_w,
                    &layer_scratch.ffn_inner_pack,
                    &layer_scratch.ffn_out_pack,
                    f,
                    h,
                    n,
                )?;
            } else {
                // F32 (or other non-mat-mat dtypes): per-token loop using
                // existing mat-vec-dispatch. Layer-major still wins here
                // through batched norms + scheduling, just not via mat-mat.
                for n_idx in 0..n {
                    let h_n = layer_scratch
                        .h_pack
                        .view_subrange((n_idx * h) as u64, vec![h as u64]);
                    let gate_n = layer_scratch
                        .ffn_gate_pack
                        .view_subrange((n_idx * f) as u64, vec![f as u64]);
                    let up_n = layer_scratch
                        .ffn_up_pack
                        .view_subrange((n_idx * f) as u64, vec![f as u64]);
                    let inner_n = layer_scratch
                        .ffn_inner_pack
                        .view_subrange((n_idx * f) as u64, vec![f as u64]);
                    let out_n = layer_scratch
                        .ffn_out_pack
                        .view_subrange((n_idx * h) as u64, vec![h as u64]);
                    encode_mat_vec_dispatch(base.ctx, &enc, g_w, &h_n, &gate_n, h, f)?;
                    encode_mat_vec_dispatch(base.ctx, &enc, u_w, &h_n, &up_n, h, f)?;
                    encode_silu_mul_f32(base.ctx, &enc, &gate_n, &up_n, &inner_n)?;
                    encode_mat_vec_dispatch(base.ctx, &enc, d_w, &inner_n, &out_n, f, h)?;
                }
            }
            // 2g: residual #2 — x_pack += ffn_out_pack.
            encode_add_inplace_f32(
                base.ctx,
                &enc,
                &layer_scratch.x_pack,
                &layer_scratch.ffn_out_pack,
            )?;

            // 2d-fix (v0.74.4): hidden capture AFTER residual #2,
            // matching single_token_capture_layers semantics. Inside
            // the same encoder as 2f-FFN + 2g-residual to avoid an
            // extra encoder transition. x_pack[n, :] is now post-FFN,
            // post-residual-#2 — bit-equivalent to what
            // `single_token_with_multi_hidden` writes into hidden_dst
            // for the same token (modulo mat-mat half-staging noise
            // when FFN takes the Q4_K mat-mat path; that's the same
            // noise the layer-major-vs-token-major test already
            // tolerates at cos≥0.999).
            for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                if lid as usize == il {
                    for n_idx in 0..n {
                        let elem_off = (n_idx as u64 * verify_scratch.k_target_layers as u64
                            + k_idx as u64)
                            * verify_scratch.hidden_size;
                        encode_scatter_offset_f32(
                            base.ctx,
                            &enc,
                            &layer_scratch
                                .x_pack
                                .view_subrange((n_idx * h) as u64, vec![h as u64]),
                            &verify_scratch.hidden_capture,
                            elem_off as usize,
                            h,
                        )?;
                    }
                }
            }

            enc.end();
        }
    }

    // === Phase 3: BATCHED tail (final norm + lm_head + argmax). ===
    //
    // H5.3b.6: lift lm_head from per-token mat-vec to a single mat-mat
    // across all N rows. lm_head is the largest weight in the model
    // (Q6_K [5120, 248320] = ~1 GiB); per-token mat-vec at N=16 would
    // re-read it 16 times = 16 GiB redundant traffic / outer step on
    // the LATENCY-CRITICAL tail path.
    //
    // Strategy:
    //   1. rms_norm_batched(x_pack, output_norm) → h_pack [N, H]
    //      (replaces per-token rms_norm_mul; trivial win)
    //   2. ONE mat-mat lm_head into final_logits_pack (or
    //      debug_logits_dst when provided — same shape, saves a copy)
    //   3. Batched argmax across all N rows → verify_argmax [N]
    //
    // Falls back to per-token mat-vec for non-mat-mat-eligible
    // lm_head dtypes (F32 0.8B oracle path).
    let lm_dtype = base.model.lm_head.dtype;
    let lm_mat_mat_path = matches!(lm_dtype, GgmlType::Q4_K | GgmlType::Q6_K);
    {
        let enc = KernelEncoder::begin(&cmd_buf);
        if lm_mat_mat_path {
            // Batched final norm: x_pack → h_pack.
            encode_rms_norm_batched_f32(
                base.ctx,
                &enc,
                &layer_scratch.x_pack,
                &base.model.output_norm,
                &layer_scratch.h_pack,
                n,
                h,
                RMS_EPS,
            )?;
            // Pick logits destination: debug_logits_dst if provided
            // (same [N, V] shape; saves a scatter), else
            // final_logits_pack.
            let logits_dst = match debug_logits_dst {
                Some(dst) => dst,
                None => &layer_scratch.final_logits_pack,
            };
            // Batched lm_head mat-mat.
            encode_mat_mat_dispatch(
                base.ctx,
                &enc,
                &base.model.lm_head,
                &layer_scratch.h_pack,
                logits_dst,
                h,
                v,
                n,
            )?;
            // Batched argmax across all N rows in ONE dispatch.
            encode_argmax_f32(
                base.ctx,
                &enc,
                logits_dst,
                &verify_scratch.verify_argmax,
                n,
                v,
            )?;
        } else {
            // F32 / unsupported lm_head: per-token mat-vec fallback
            // (the original layer-major tail). Layer-major still wins
            // through the batched final norm only.
            for n_idx in 0..n {
                encode_copy_offset_f32(
                    base.ctx,
                    &enc,
                    &layer_scratch.x_pack,
                    n_idx * h,
                    &target_session.x,
                    h,
                )?;
                encode_rms_norm_mul_f32(
                    base.ctx,
                    &enc,
                    &target_session.x,
                    &base.model.output_norm,
                    &target_session.h,
                    RMS_EPS,
                )?;
                encode_mat_vec_dispatch(
                    base.ctx,
                    &enc,
                    &base.model.lm_head,
                    &target_session.h,
                    &target_session.logits,
                    h,
                    v,
                )?;
                if let Some(dst) = debug_logits_dst {
                    let elem_off = (n_idx as u64) * (v as u64);
                    encode_scatter_offset_f32(
                        base.ctx,
                        &enc,
                        &target_session.logits,
                        dst,
                        elem_off as usize,
                        v,
                    )?;
                }
                let argmax_dst = verify_scratch.argmax_slot(n_idx as u32);
                encode_argmax_f32(base.ctx, &enc, &target_session.logits, &argmax_dst, 1, v)?;
            }
        }
        enc.end();
    }

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let mut out = vec![0i32; n];
    unsafe {
        let src = verify_scratch.verify_argmax.buffer.contents().as_ptr() as *const i32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }
    Ok(out)
}

/// **v0.75.1** Packed multi-token prefill.
///
/// Processes `token_ids` through chunks of P (= `layer_scratch.n`)
/// prompt tokens at a time, batching per-layer projections (Q/K/V,
/// out_proj, GDN in/out, FFN gate/up/down) as mat-mat across the
/// chunk while keeping per-token GDN recurrence and per-token
/// attn-v4 + KV scatter (which are inherently sequential because
/// each token's K/V cache slot must be written and observed by the
/// NEXT token's attention dispatch).
///
/// # Caller invariant
///
/// `target_session.kv_n_pos[ai] == start_position` for every attn
/// layer index `ai`. (I.e., the session has been advanced through
/// tokens `[0, start_position)` by some prior path; this prefill
/// extends KV state by `T = token_ids.len()` more positions, ending
/// with `kv_n_pos[ai] == start_position + T`.)
///
/// # Hidden capture layout
///
/// `hidden_dst` is `Some(tensor)` of shape `[T, K, H]` row-major
/// (`K = target_layer_ids.len()`, `H = arch.hidden_size`) for the
/// DFlash prefill case (drafter ctx bootstrap), or `None` for the
/// no-spec reference path (no hidden capture needed). When `Some`,
/// for token `t` (`0 ≤ t < T`) and capture index `k` (`0 ≤ k < K`),
/// the slice `hidden_dst[(t * K + k) * H .. (t * K + k + 1) * H]`
/// receives the post-residual-#2 hidden state at layer
/// `target_layer_ids[k]`. Matches the `[N, K, H]` layout used by
/// `packed_verify`'s `verify_scratch.hidden_capture`.
///
/// When `hidden_dst` is `None`, `target_layer_ids` MUST be empty.
///
/// # Tail
///
/// The LAST token (`token_ids[T-1]`) runs the full final RMSNorm,
/// lm_head, and CPU readback so the caller can seed the decode phase.
/// All other tokens skip the tail (matching v0.75.0 skip-tail semantics).
///
/// # Equivalence
///
/// Behaves like calling `single_token_with_multi_hidden` for every
/// prompt token in sequence, with cosine equivalence ≥ 0.999 on:
/// final logits, accumulated `hidden_dst`, KV cache prefix
/// `kv_k[ai] / kv_v[ai]` over `[0, start_position + T)`, GDN state
/// and conv tensors. NOT bit-exact: mat-mat half-staging differs
/// from per-token mat-vec summation order.
///
/// `kv_n_pos[ai] == start_position + T` exactly on return.
///
/// # Chunk sizing
///
/// `P = layer_scratch.n` (typically 16). The final chunk may have
/// fewer tokens (`chunk_p < P`); all encoder dispatches in that
/// chunk use `view_subrange`-sized tensors to satisfy host shape
/// validation. The mat-mat kernels handle partial-N via the generic
/// NR1 path (the NR1=16 fast path only fires when `n_query == 16`,
/// which is fine — short final chunks pay slight per-N overhead
/// but still amortize weight reads across the chunk).
pub fn prefill_tokens_with_multi_hidden(
    base: &MetalForward<'_>,
    token_ids: &[i32],
    start_position: u32,
    target_session: &mut MetalSession,
    layer_scratch: &mut MetalDFlashLayerMajorScratch,
    target_layer_ids: &[u32],
    hidden_dst: Option<&MetalTensor>,
) -> Result<Vec<f32>, DFlashError> {
    let arch = &base.model.arch;
    let total_n = token_ids.len();
    let h = arch.hidden_size as usize;
    let f = arch.intermediate_size as usize;
    let v = arch.vocab_size as usize;
    let k_target = target_layer_ids.len();
    let p_max = layer_scratch.n as usize;

    // ---- Public-entry validation ----
    if total_n == 0 {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "prefill_tokens_with_multi_hidden",
            detail: "token_ids is empty".into(),
        }));
    }
    if p_max == 0 {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "prefill_tokens_with_multi_hidden",
            detail: "layer_scratch.n is 0".into(),
        }));
    }
    if layer_scratch.hidden_size != h as u64 {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "prefill_tokens_with_multi_hidden",
            detail: format!(
                "layer_scratch.hidden_size={} != arch.hidden_size={h}",
                layer_scratch.hidden_size
            ),
        }));
    }
    let last_pos = (start_position as usize)
        .checked_add(total_n)
        .ok_or_else(|| {
            DFlashError::Metal(MetalError::BadShape {
                kernel: "prefill_tokens_with_multi_hidden",
                detail: format!("start_position={start_position} + T={total_n} overflows usize"),
            })
        })?;
    if last_pos > target_session.kv_capacity {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "prefill_tokens_with_multi_hidden",
            detail: format!(
                "start_position + T = {last_pos} > kv_capacity={}",
                target_session.kv_capacity
            ),
        }));
    }
    for (i, &kp) in target_session.kv_n_pos.iter().enumerate() {
        if kp != start_position as usize {
            return Err(DFlashError::Metal(MetalError::BadShape {
                kernel: "prefill_tokens_with_multi_hidden",
                detail: format!(
                    "kv_n_pos[{i}]={kp} != start_position={start_position} \
                     (caller must advance session through [0, start_position) before prefill)"
                ),
            }));
        }
    }
    for &t in token_ids.iter() {
        if t < 0 || (t as u32) >= arch.vocab_size {
            return Err(DFlashError::BadToken(t, arch.vocab_size));
        }
    }
    for &lid in target_layer_ids {
        if (lid as usize) >= base.model.blocks.len() {
            return Err(DFlashError::Metal(MetalError::BadShape {
                kernel: "prefill_tokens_with_multi_hidden.target_layer_ids",
                detail: format!("layer id {lid} >= n_layer {}", base.model.blocks.len()),
            }));
        }
    }
    match hidden_dst {
        Some(dst) => {
            let want_hidden_elems = total_n * k_target * h;
            if (dst.n_elements() as usize) != want_hidden_elems {
                return Err(DFlashError::Metal(MetalError::BadShape {
                    kernel: "prefill_tokens_with_multi_hidden.hidden_dst",
                    detail: format!(
                        "expected {want_hidden_elems} elements (T={total_n} × K={k_target} × H={h}), got {}",
                        dst.n_elements()
                    ),
                }));
            }
        }
        None => {
            if k_target != 0 {
                return Err(DFlashError::Metal(MetalError::BadShape {
                    kernel: "prefill_tokens_with_multi_hidden.hidden_dst",
                    detail: format!(
                        "hidden_dst=None requires target_layer_ids=[], got K={k_target}"
                    ),
                }));
            }
        }
    }
    let n_gdn_actual = base
        .model
        .blocks
        .iter()
        .filter(|b| matches!(b, MetalBlock::Gdn(_)))
        .count();
    if target_session.gdn_state.len() != n_gdn_actual
        || target_session.gdn_conv.len() != n_gdn_actual
    {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "prefill_tokens_with_multi_hidden",
            detail: format!(
                "target_session GDN state slots ({}/{}) != model n_gdn ({n_gdn_actual})",
                target_session.gdn_state.len(),
                target_session.gdn_conv.len()
            ),
        }));
    }

    // Cache mat-mat-eligible predicates once.
    let gdn_mat_mat_eligible =
        |dtype: GgmlType| matches!(dtype, GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K);
    let attn_mat_mat_eligible =
        |dtype: GgmlType| matches!(dtype, GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K);
    let ffn_mat_mat_eligible = |dtype: GgmlType| matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K);

    // Per-call ids buffer. P=16 i32 = 64 bytes; trivial alloc cost.
    // (Same "F32-typed buffer holding i32" convention as
    // verify_scratch.packed_ids_buf — see the noise_ids comment in
    // MetalDFlashSession.)
    let ids_buf =
        MetalTensor::zeros_f32(base.ctx, vec![p_max as u64]).map_err(DFlashError::Metal)?;

    let n_chunks = total_n.div_ceil(p_max);
    for chunk_idx in 0..n_chunks {
        let chunk_base = chunk_idx * p_max;
        let chunk_p = (total_n - chunk_base).min(p_max);
        let chunk_start = start_position + chunk_base as u32;
        let is_last_chunk = chunk_idx + 1 == n_chunks;

        // Per-chunk internal invariant guard. Catches off-by-ones in
        // the chunk-driver vs. the per-token attn-v4 advancement of
        // kv_n_pos — codex's predicted first-symptom-of-bug class.
        for (ai, &kp) in target_session.kv_n_pos.iter().enumerate() {
            if kp != chunk_start as usize {
                return Err(DFlashError::Metal(MetalError::BadShape {
                    kernel: "prefill_tokens_with_multi_hidden.chunk_invariant",
                    detail: format!(
                        "chunk {chunk_idx}: kv_n_pos[{ai}]={kp} != chunk_start={chunk_start} \
                         (expected {} after {chunk_idx} chunks of size up to {p_max})",
                        chunk_start
                    ),
                }));
            }
        }

        // Stage chunk_p token ids.
        unsafe {
            let p = ids_buf.buffer.contents().as_ptr() as *mut i32;
            for (i, &t) in token_ids[chunk_base..chunk_base + chunk_p]
                .iter()
                .enumerate()
            {
                *p.add(i) = t;
            }
        }

        // One MTLCommandBuffer per chunk. We need to commit + wait between
        // chunks because (a) ids_buf is reused across chunks and CPU writes
        // before each commit must be ordered behind GPU reads of prior
        // commits (codex Q3 hazard), and (b) layer_scratch is reused.
        let cmd_buf = base.ctx.queue.commandBuffer().expect("command buffer");

        // Sized views of layer_scratch sliced to chunk_p. Every encoder
        // dispatch's host-side n_elements() validation is against the
        // sized view, NOT the full [p_max] buffer. This is what keeps
        // chunk_p < p_max correct.
        let x_pack_p = layer_scratch
            .x_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack_p = layer_scratch
            .h_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);
        let mixer_out_pack_p = layer_scratch
            .mixer_out_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);

        // === Phase 1: batched embed of chunk_p tokens. ===
        {
            let enc = KernelEncoder::begin(&cmd_buf);
            let ids_view = ids_buf.view_subrange(0, vec![chunk_p as u64]);
            encode_get_rows_f32(
                base.ctx,
                &enc,
                &base.model.token_embd,
                &ids_view,
                &x_pack_p,
                chunk_p,
                h,
            )?;
            enc.end();
        }

        // === Phase 2: per-layer body. ===
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in base.model.blocks.iter().enumerate() {
            // 2a: pre-mixer norm batched across chunk_p rows.
            let attn_norm = match block {
                MetalBlock::Gdn(g) => &g.attn_norm,
                MetalBlock::Attn(a) => &a.attn_norm,
            };
            {
                let enc = KernelEncoder::begin(&cmd_buf);
                encode_rms_norm_batched_f32(
                    base.ctx, &enc, &x_pack_p, attn_norm, &h_pack_p, chunk_p, h, RMS_EPS,
                )?;
                enc.end();
            }

            // 2b: mixer (GDN or Attn). NO ckpt blits (prefill is
            // final-commit; no rollback machinery).
            match block {
                MetalBlock::Gdn(g) => {
                    let gi = gdn_idx;
                    gdn_idx += 1;
                    let n_k_u = arch.gdn_n_k_heads as usize;
                    let n_v_u = arch.gdn_n_v_heads as usize;
                    let head_dim_u = arch.gdn_head_dim as usize;
                    let conv_dim = (2 * n_k_u + n_v_u) * head_dim_u;
                    let v_dim = n_v_u * head_dim_u;

                    let gdn_batched = gdn_mat_mat_eligible(g.in_proj_qkv.dtype)
                        && gdn_mat_mat_eligible(g.in_proj_z.dtype)
                        && gdn_mat_mat_eligible(g.out_proj.dtype);

                    if gdn_batched {
                        // Sized views of GDN pack scratch.
                        let gdn_qkv_pack_p = layer_scratch
                            .gdn_qkv_pack
                            .view_subrange(0, vec![(chunk_p * conv_dim) as u64]);
                        let gdn_z_pack_p = layer_scratch
                            .gdn_z_pack
                            .view_subrange(0, vec![(chunk_p * v_dim) as u64]);
                        let gdn_normed_pack_p = layer_scratch
                            .gdn_normed_pack
                            .view_subrange(0, vec![(chunk_p * v_dim) as u64]);

                        // Step A: batched front-end QKV / Z projections.
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_mat_mat_dispatch(
                                base.ctx,
                                &enc,
                                &g.in_proj_qkv,
                                &h_pack_p,
                                &gdn_qkv_pack_p,
                                h,
                                conv_dim,
                                chunk_p,
                            )?;
                            encode_mat_mat_dispatch(
                                base.ctx,
                                &enc,
                                &g.in_proj_z,
                                &h_pack_p,
                                &gdn_z_pack_p,
                                h,
                                v_dim,
                                chunk_p,
                            )?;
                            enc.end();
                        }

                        // Per-token recurrence (GDN state mutates per token).
                        // NO ckpt blit — prefill never rolls back.
                        let alpha_handle = target_session.gdn_alpha.clone();
                        let beta_handle = target_session.gdn_beta.clone();
                        for n_idx in 0..chunk_p {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            let h_n = h_pack_p.view_subrange((n_idx * h) as u64, vec![h as u64]);
                            encode_mat_vec_dispatch(
                                base.ctx,
                                &enc,
                                &g.beta_proj,
                                &h_n,
                                &target_session.gdn_b,
                                h,
                                n_v_u,
                            )?;
                            encode_sigmoid_f32(
                                base.ctx,
                                &enc,
                                &target_session.gdn_b,
                                &target_session.gdn_beta,
                            )?;
                            encode_mat_vec_dispatch(
                                base.ctx,
                                &enc,
                                &g.alpha_proj,
                                &h_n,
                                &target_session.gdn_a,
                                h,
                                n_v_u,
                            )?;
                            encode_gdn_alpha_chain_f32(
                                base.ctx,
                                &enc,
                                &target_session.gdn_a,
                                &g.dt_bias,
                                &g.a_log,
                                &target_session.gdn_alpha,
                            )?;
                            let qkv_n = gdn_qkv_pack_p
                                .view_subrange((n_idx * conv_dim) as u64, vec![conv_dim as u64]);
                            let z_n = gdn_z_pack_p
                                .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                            let normed_n = gdn_normed_pack_p
                                .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                            base.encode_gdn_tail(
                                &enc,
                                g,
                                gi,
                                target_session,
                                &qkv_n,
                                &z_n,
                                &alpha_handle,
                                &beta_handle,
                                &normed_n,
                            )?;
                            enc.end();
                        }

                        // Step C: batched out_proj.
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_mat_mat_dispatch(
                                base.ctx,
                                &enc,
                                &g.out_proj,
                                &gdn_normed_pack_p,
                                &mixer_out_pack_p,
                                v_dim,
                                h,
                                chunk_p,
                            )?;
                            enc.end();
                        }
                    } else {
                        // F32 oracle / mixed-dtype fallback: per-token
                        // encode_gdn (no ckpt blit).
                        for n_idx in 0..chunk_p {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_copy_offset_f32(
                                base.ctx,
                                &enc,
                                &h_pack_p,
                                n_idx * h,
                                &target_session.h,
                                h,
                            )?;
                            base.encode_gdn(&enc, g, gi, target_session)?;
                            encode_scatter_offset_f32(
                                base.ctx,
                                &enc,
                                &target_session.mixer_out,
                                &mixer_out_pack_p,
                                n_idx * h,
                                h,
                            )?;
                            enc.end();
                        }
                    }
                }
                MetalBlock::Attn(a) => {
                    let ai = attn_idx;
                    attn_idx += 1;
                    let head_dim = arch.attn_head_dim as usize;
                    let n_q = arch.n_q_heads as usize;
                    let n_kv = arch.n_kv_heads as usize;
                    let q_dim = n_q * head_dim;
                    let kv_dim = n_kv * head_dim;
                    let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;

                    let attn_batched = attn_mat_mat_eligible(a.q.dtype)
                        && attn_mat_mat_eligible(a.k.dtype)
                        && attn_mat_mat_eligible(a.v.dtype)
                        && attn_mat_mat_eligible(a.o.dtype);

                    if attn_batched {
                        // Sized views of attn pack scratch.
                        let q_full_pack_p = layer_scratch
                            .attn_q_full_pack
                            .view_subrange(0, vec![(chunk_p * 2 * q_dim) as u64]);
                        let q_pack_p = layer_scratch
                            .attn_q_pack
                            .view_subrange(0, vec![(chunk_p * q_dim) as u64]);
                        let gate_pack_p = layer_scratch
                            .attn_gate_pack
                            .view_subrange(0, vec![(chunk_p * q_dim) as u64]);
                        let q_normed_pack_p = layer_scratch
                            .attn_q_normed_pack
                            .view_subrange(0, vec![(chunk_p * q_dim) as u64]);
                        let k_now_pack_p = layer_scratch
                            .attn_k_now_pack
                            .view_subrange(0, vec![(chunk_p * kv_dim) as u64]);
                        let v_now_pack_p = layer_scratch
                            .attn_v_now_pack
                            .view_subrange(0, vec![(chunk_p * kv_dim) as u64]);
                        let k_normed_pack_p = layer_scratch
                            .attn_k_normed_pack
                            .view_subrange(0, vec![(chunk_p * kv_dim) as u64]);
                        let attn_o_pack_p = layer_scratch
                            .attn_o_pack
                            .view_subrange(0, vec![(chunk_p * q_dim) as u64]);

                        // Step A: batched front-end Q gated / K / V projections + Q-norm + K-norm.
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_mat_mat_dispatch(
                                base.ctx,
                                &enc,
                                &a.q,
                                &h_pack_p,
                                &q_full_pack_p,
                                h,
                                2 * q_dim,
                                chunk_p,
                            )?;
                            encode_split_q_gate_f32(
                                base.ctx,
                                &enc,
                                &q_full_pack_p,
                                &q_pack_p,
                                &gate_pack_p,
                                chunk_p * n_q,
                                head_dim,
                            )?;
                            encode_mat_mat_dispatch(
                                base.ctx,
                                &enc,
                                &a.k,
                                &h_pack_p,
                                &k_now_pack_p,
                                h,
                                kv_dim,
                                chunk_p,
                            )?;
                            encode_mat_mat_dispatch(
                                base.ctx,
                                &enc,
                                &a.v,
                                &h_pack_p,
                                &v_now_pack_p,
                                h,
                                kv_dim,
                                chunk_p,
                            )?;
                            encode_rms_norm_batched_f32(
                                base.ctx,
                                &enc,
                                &q_pack_p,
                                &a.q_norm,
                                &q_normed_pack_p,
                                chunk_p * n_q,
                                head_dim,
                                RMS_EPS,
                            )?;
                            encode_rms_norm_batched_f32(
                                base.ctx,
                                &enc,
                                &k_now_pack_p,
                                &a.k_norm,
                                &k_normed_pack_p,
                                chunk_p * n_kv,
                                head_dim,
                                RMS_EPS,
                            )?;
                            enc.end();
                        }

                        // Per-token RoPE + KV scatter + attn-v4. This
                        // advances target_session.kv_n_pos[ai] from
                        // chunk_start to chunk_start + chunk_p.
                        for n_idx in 0..chunk_p {
                            let position_n = chunk_start + n_idx as u32;
                            let q_normed_n = q_normed_pack_p
                                .view_subrange((n_idx * q_dim) as u64, vec![q_dim as u64]);
                            let k_normed_n = k_normed_pack_p
                                .view_subrange((n_idx * kv_dim) as u64, vec![kv_dim as u64]);
                            let v_now_n = v_now_pack_p
                                .view_subrange((n_idx * kv_dim) as u64, vec![kv_dim as u64]);
                            let attn_o_n = attn_o_pack_p
                                .view_subrange((n_idx * q_dim) as u64, vec![q_dim as u64]);
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_rope_neox_f32(
                                base.ctx,
                                &enc,
                                &q_normed_n,
                                n_q,
                                head_dim,
                                n_rot,
                                position_n,
                                arch.rope_theta,
                            )?;
                            encode_rope_neox_f32(
                                base.ctx,
                                &enc,
                                &k_normed_n,
                                n_kv,
                                head_dim,
                                n_rot,
                                position_n,
                                arch.rope_theta,
                            )?;
                            encode_scatter_offset_f32_to_f16_kv(
                                base.ctx,
                                &enc,
                                &k_normed_n,
                                &v_now_n,
                                &target_session.kv_k[ai],
                                &target_session.kv_v[ai],
                                (position_n as usize) * kv_dim,
                                kv_dim,
                            )?;
                            target_session.kv_n_pos[ai] = position_n as usize + 1;

                            const V4_HEAD_DIM: usize = 256;
                            const V4_GROUP: usize = 6;
                            let use_v4 = head_dim == V4_HEAD_DIM && n_q == n_kv * V4_GROUP;
                            if use_v4 {
                                let nwg =
                                    crate::metal::attn_v4_choose_nwg(target_session.kv_n_pos[ai]);
                                let tile_c = crate::metal::attn_v4_choose_tile_c(
                                    target_session.kv_n_pos[ai],
                                );
                                crate::metal::encode_attn_decode_v4_f32(
                                    base.ctx,
                                    &enc,
                                    &q_normed_n,
                                    &target_session.kv_k[ai],
                                    &target_session.kv_v[ai],
                                    &target_session.attn_v4_o_partial,
                                    &target_session.attn_v4_ml_partial,
                                    &attn_o_n,
                                    n_q,
                                    n_kv,
                                    head_dim,
                                    target_session.kv_n_pos[ai],
                                    nwg,
                                    tile_c,
                                )?;
                            } else {
                                crate::metal::encode_attn_decode_f16kv_f32(
                                    base.ctx,
                                    &enc,
                                    &q_normed_n,
                                    &target_session.kv_k[ai],
                                    &target_session.kv_v[ai],
                                    &attn_o_n,
                                    n_q,
                                    n_kv,
                                    head_dim,
                                    target_session.kv_n_pos[ai],
                                )?;
                            }
                            enc.end();
                        }

                        // Step C: gate-sigmoid + mul + batched o_proj.
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_sigmoid_f32(base.ctx, &enc, &gate_pack_p, &q_pack_p)?;
                            crate::metal::encode_mul_f32(
                                base.ctx,
                                &enc,
                                &attn_o_pack_p,
                                &q_pack_p,
                                &attn_o_pack_p,
                            )?;
                            encode_mat_mat_dispatch(
                                base.ctx,
                                &enc,
                                &a.o,
                                &attn_o_pack_p,
                                &mixer_out_pack_p,
                                q_dim,
                                h,
                                chunk_p,
                            )?;
                            enc.end();
                        }
                    } else {
                        // F32 oracle / fallback: per-token encode_attn.
                        for n_idx in 0..chunk_p {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_copy_offset_f32(
                                base.ctx,
                                &enc,
                                &h_pack_p,
                                n_idx * h,
                                &target_session.h,
                                h,
                            )?;
                            let position_n = chunk_start + n_idx as u32;
                            base.encode_attn(&enc, a, ai, position_n, target_session)?;
                            encode_scatter_offset_f32(
                                base.ctx,
                                &enc,
                                &target_session.mixer_out,
                                &mixer_out_pack_p,
                                n_idx * h,
                                h,
                            )?;
                            enc.end();
                        }
                    }
                }
            }

            // 2c: residual #1 — x_pack += mixer_out_pack (batched).
            {
                let enc = KernelEncoder::begin(&cmd_buf);
                encode_add_inplace_f32(base.ctx, &enc, &x_pack_p, &mixer_out_pack_p)?;
                enc.end();
            }

            // 2e/f/g: post-norm + FFN + residual #2 + hidden_capture.
            let post_norm = match block {
                MetalBlock::Gdn(g) => &g.post_attn_norm,
                MetalBlock::Attn(a) => &a.post_attn_norm,
            };
            let (g_w, u_w, d_w) = match block {
                MetalBlock::Gdn(g) => (&g.ffn_gate, &g.ffn_up, &g.ffn_down),
                MetalBlock::Attn(a) => (&a.ffn_gate, &a.ffn_up, &a.ffn_down),
            };
            let mat_mat_path = ffn_mat_mat_eligible(g_w.dtype)
                && ffn_mat_mat_eligible(u_w.dtype)
                && ffn_mat_mat_eligible(d_w.dtype);

            // Sized FFN pack views.
            let ffn_gate_pack_p = layer_scratch
                .ffn_gate_pack
                .view_subrange(0, vec![(chunk_p * f) as u64]);
            let ffn_up_pack_p = layer_scratch
                .ffn_up_pack
                .view_subrange(0, vec![(chunk_p * f) as u64]);
            let ffn_inner_pack_p = layer_scratch
                .ffn_inner_pack
                .view_subrange(0, vec![(chunk_p * f) as u64]);
            let ffn_out_pack_p = layer_scratch
                .ffn_out_pack
                .view_subrange(0, vec![(chunk_p * h) as u64]);

            {
                let enc = KernelEncoder::begin(&cmd_buf);
                encode_rms_norm_batched_f32(
                    base.ctx, &enc, &x_pack_p, post_norm, &h_pack_p, chunk_p, h, RMS_EPS,
                )?;
                if mat_mat_path {
                    encode_mat_mat_dispatch(
                        base.ctx,
                        &enc,
                        g_w,
                        &h_pack_p,
                        &ffn_gate_pack_p,
                        h,
                        f,
                        chunk_p,
                    )?;
                    encode_mat_mat_dispatch(
                        base.ctx,
                        &enc,
                        u_w,
                        &h_pack_p,
                        &ffn_up_pack_p,
                        h,
                        f,
                        chunk_p,
                    )?;
                    encode_silu_mul_f32(
                        base.ctx,
                        &enc,
                        &ffn_gate_pack_p,
                        &ffn_up_pack_p,
                        &ffn_inner_pack_p,
                    )?;
                    encode_mat_mat_dispatch(
                        base.ctx,
                        &enc,
                        d_w,
                        &ffn_inner_pack_p,
                        &ffn_out_pack_p,
                        f,
                        h,
                        chunk_p,
                    )?;
                } else {
                    // F32 / non-mat-mat fallback: per-token mat-vec.
                    for n_idx in 0..chunk_p {
                        let h_n = h_pack_p.view_subrange((n_idx * h) as u64, vec![h as u64]);
                        let gate_n =
                            ffn_gate_pack_p.view_subrange((n_idx * f) as u64, vec![f as u64]);
                        let up_n = ffn_up_pack_p.view_subrange((n_idx * f) as u64, vec![f as u64]);
                        let inner_n =
                            ffn_inner_pack_p.view_subrange((n_idx * f) as u64, vec![f as u64]);
                        let out_n =
                            ffn_out_pack_p.view_subrange((n_idx * h) as u64, vec![h as u64]);
                        encode_mat_vec_dispatch(base.ctx, &enc, g_w, &h_n, &gate_n, h, f)?;
                        encode_mat_vec_dispatch(base.ctx, &enc, u_w, &h_n, &up_n, h, f)?;
                        encode_silu_mul_f32(base.ctx, &enc, &gate_n, &up_n, &inner_n)?;
                        encode_mat_vec_dispatch(base.ctx, &enc, d_w, &inner_n, &out_n, f, h)?;
                    }
                }
                // 2g: residual #2 — x_pack += ffn_out_pack (batched).
                encode_add_inplace_f32(base.ctx, &enc, &x_pack_p, &ffn_out_pack_p)?;

                // 2d-fix (matches v0.74.4 capture point): hidden capture
                // after residual #2. Writes into the GLOBAL hidden_dst at
                // offset ((global_idx * K + k_idx) * H), where
                // global_idx = chunk_base + n_idx. Skipped entirely when
                // hidden_dst is None (no-spec ref path).
                if let Some(dst) = hidden_dst {
                    for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                        if lid as usize == il {
                            for n_idx in 0..chunk_p {
                                let global_idx = chunk_base + n_idx;
                                let elem_off = (global_idx * k_target + k_idx) * h;
                                let row_view =
                                    x_pack_p.view_subrange((n_idx * h) as u64, vec![h as u64]);
                                encode_scatter_offset_f32(
                                    base.ctx, &enc, &row_view, dst, elem_off, h,
                                )?;
                            }
                        }
                    }
                }

                enc.end();
            }
        } // end per-layer loop

        // === Phase 3: tail. Only runs on the LAST chunk's LAST token. ===
        //
        // We DO NOT batch lm_head over the chunk: only the last prompt
        // token's logits are needed (to seed decode). Other tokens skip
        // the tail entirely (extending the v0.75.0 skip-tail lever to
        // multi-token prefill). Tail uses target_session.x / .h / .logits
        // (single-token scratch); we copy x_pack[chunk_p-1, :] into
        // session.h via final norm directly, skipping the session.x copy.
        if is_last_chunk {
            let enc = KernelEncoder::begin(&cmd_buf);
            let x_last = x_pack_p.view_subrange(((chunk_p - 1) * h) as u64, vec![h as u64]);
            encode_rms_norm_mul_f32(
                base.ctx,
                &enc,
                &x_last,
                &base.model.output_norm,
                &target_session.h,
                RMS_EPS,
            )?;
            encode_mat_vec_dispatch(
                base.ctx,
                &enc,
                &base.model.lm_head,
                &target_session.h,
                &target_session.logits,
                h,
                v,
            )?;
            enc.end();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            let mut last_logits = vec![0.0f32; v];
            unsafe {
                let src = target_session.logits.buffer.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, last_logits.as_mut_ptr(), v);
            }
            return Ok(last_logits);
        }

        // Non-last chunk: just commit + wait (no tail).
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }

    // unreachable: the last chunk always returns inside the loop. But
    // satisfy the type checker.
    unreachable!("prefill loop exits via last-chunk return");
}

pub fn encode_restore_after_partial_accept_inner(
    base: &MetalForward<'_>,
    scratch: &MetalDFlashVerifyScratch,
    n_keep: u32,
    start_position: u32,
    target_session: &mut MetalSession,
) -> Result<(), DFlashError> {
    // -- Validation guard wall (same discipline as packed_verify).
    let n = scratch.n;
    if n_keep == 0 || n_keep > n {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "restore_after_partial_accept",
            detail: format!(
                "n_keep={n_keep} must be in [1, N={n}] (n_keep=0 is impossible \
                 by construction — the carry token is always processed; \
                 see DFlashDecoder::restore_after_partial_accept docs)"
            ),
        }));
    }
    let n_gdn_actual = base
        .model
        .blocks
        .iter()
        .filter(|b| matches!(b, MetalBlock::Gdn(_)))
        .count() as u32;
    if scratch.n_gdn_layers != n_gdn_actual {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "restore_after_partial_accept",
            detail: format!(
                "scratch.n_gdn_layers={} != model n_gdn={n_gdn_actual} \
                 (scratch allocated for different layer schedule)",
                scratch.n_gdn_layers
            ),
        }));
    }
    if target_session.gdn_state.len() != n_gdn_actual as usize
        || target_session.gdn_conv.len() != n_gdn_actual as usize
    {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "restore_after_partial_accept",
            detail: format!(
                "session.gdn_state.len={} / gdn_conv.len={} != model n_gdn={n_gdn_actual}",
                target_session.gdn_state.len(),
                target_session.gdn_conv.len(),
            ),
        }));
    }
    // KV n_pos contract: every attn layer must currently have
    // kv_n_pos[i] == start_position + N (i.e., we just ran a full
    // packed_verify of length N and now want to roll back to
    // n_keep). Failing this means the caller is mismatching
    // packed_verify and restore.
    let expected_kv_pre = (start_position as usize)
        .checked_add(n as usize)
        .ok_or_else(|| {
            DFlashError::Metal(MetalError::BadShape {
                kernel: "restore_after_partial_accept",
                detail: format!("start_position={start_position} + N={n} overflows usize"),
            })
        })?;
    for (i, &kp) in target_session.kv_n_pos.iter().enumerate() {
        if kp != expected_kv_pre {
            return Err(DFlashError::Metal(MetalError::BadShape {
                kernel: "restore_after_partial_accept",
                detail: format!(
                    "kv_n_pos[{i}]={kp} != start_position + N = {expected_kv_pre}; \
                     restore must be called immediately after packed_verify(.., \
                     start_position) on the same session"
                ),
            }));
        }
    }

    // -- Encode all blits in one command buffer.
    //
    // Source: gdn_ckpt_slot(k, n_keep - 1) and conv_ckpt_slot(k, n_keep - 1)
    // for every GDN layer k. Each slot is a zero-copy view into the
    // big checkpoint buffer at the right offset.
    //
    // Destination: target_session.gdn_state[k] / target_session.gdn_conv[k].
    //
    // n_keep == N (full accept) edge case: source is gdn_ckpt_slot(k, N-1),
    // which holds state-after-token-(N-1) — exactly what's currently in
    // session.gdn_state[k]. The blit is a no-op in semantics but still
    // copies bytes. Optimization opportunity (skip the blit when n_keep == N)
    // is deferred — at v1 we want the simplest, most-defensive code path.
    let cmd_buf = base.ctx.queue.commandBuffer().expect("command buffer");
    let blit = BlitEncoder::begin(&cmd_buf);
    let ckpt_n = n_keep - 1; // checkpoint index to restore from
    for k in 0..n_gdn_actual {
        let ssm_src = scratch.gdn_ckpt_slot(k, ckpt_n);
        blit.copy_tensor(&ssm_src, &target_session.gdn_state[k as usize]);
        let conv_src = scratch.conv_ckpt_slot(k, ckpt_n);
        blit.copy_tensor(&conv_src, &target_session.gdn_conv[k as usize]);
    }
    blit.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    // -- Host-side: update kv_n_pos for every attn layer.
    //
    // KV slot bytes at [start_position + n_keep, ...) physically remain
    // but become unreachable; next verify will overwrite them. No need
    // to clear.
    let new_kv_pos = (start_position as usize) + (n_keep as usize);
    for i in 0..target_session.kv_n_pos.len() {
        target_session.kv_n_pos[i] = new_kv_pos;
    }

    Ok(())
}

impl<'a> DFlashDecoder<'a> {
    /// Run drafter forward for one outer step. Produces `[N]` greedy
    /// argmax tokens. Position 0 of the returned vector is the carry
    /// seed's argmax (conventionally discarded); positions `1..N` are
    /// the `D = N-1` draft candidates.
    ///
    /// `noise_start_pos` = absolute sequence position of `carry_tok`
    /// (i.e., `processed_pos + 1`).
    // v0.74.1 phase 2 ctx-cache RoPE delta loop:
    // `for c in phase2_ctx_start..ctx_len { pos_ctx_cpu[c]; view_subrange(c * kv_dim, ...) }`.
    // `c` is multi-purpose (positional arg + stride math).
    #[allow(clippy::needless_range_loop)]
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
        let _group = n_q / n_kv;
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

        // ----- Phase 1 (Metal, v0.74.0 cached): cross-context fc + hidden_norm -----
        // Per-column mat-vec into ctx_h, then per-column RMSNorm —
        // BUT only for the delta `[ctx_h_ready_n, ctx_len)`. The cache
        // exploits the append-only property of `target_ctx_stacked`:
        // rows `[0, ctx_h_ready_n)` were projected on a previous outer
        // step and the projection is a deterministic function of the
        // stacked input. Pre-v0.74.0 this loop was `0..ctx_len` and
        // grew linearly with prompt length (~830 ms / outer step at
        // ctx=1455).
        //
        // Defensive: clamp ctx_h_ready_n to ctx_len in case a future
        // caller ever resets target_ctx_n non-monotonically (no such
        // caller today; tracked under `ctx_h_ready_n` field doc).
        if self.session.ctx_h_ready_n > ctx_len {
            self.session.ctx_h_ready_n = ctx_len;
        }
        let phase1_start = self.session.ctx_h_ready_n;
        if ctx_len > phase1_start {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for c in phase1_start..ctx_len {
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
            for c in phase1_start..ctx_len {
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
            self.session.maybe_record("phase1_ctx_fc_norm", &cmd);
            // Cache watermark advances; phase 1 is complete for all
            // currently-stacked positions.
            self.session.ctx_h_ready_n = ctx_len;
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
        self.session.maybe_record("phase2_embed", &cmd);

        // Read pos_ctx once (used by RoPE on K_ctx and SWA mask).
        let mut pos_ctx_cpu = vec![0i32; ctx_len];
        if ctx_len > 0 {
            unsafe {
                let src = self.session.pos_ctx.buffer.contents().as_ptr() as *const i32;
                std::ptr::copy_nonoverlapping(src, pos_ctx_cpu.as_mut_ptr(), ctx_len);
            }
        }

        // v0.74.x notes:
        //   * Drafter weights have been native Q8_0 since v0.73b.1
        //     (`weight_dtype_kept_native` accepts Q8_0; the loader
        //     stops dequant'ing at load and the dispatchers route
        //     Q8_0 through `kernel_mat_vec_q8_0_f32` /
        //     `kernel_mat_mat_q8_0_f32`). Old "dequants Q8_0 at load
        //     time" comments removed; old `borrow_f32_tensor` /
        //     `read_f32_activation` closures (relics of the v0.71 CPU
        //     fallback before phase 3 was Metal-ized in v0.72.1)
        //     deleted as dead code.
        //   * Phase 1 (ctx_h) and per-layer K/V_ctx caches were added
        //     in v0.74.0 / v0.74.1; phase 3 lifted to Q8_0 mat-mat in
        //     v0.74.2.

        // v0.74.1: clamp the cross-context K/V cache watermark
        // defensively (mirrors ctx_h_ready_n behavior). The two
        // watermarks are independent so a future caller can't desync
        // them via target_ctx_n manipulation.
        if self.session.kv_ctx_ready_n > ctx_len {
            self.session.kv_ctx_ready_n = ctx_len;
        }
        let phase2_ctx_start = self.session.kv_ctx_ready_n;
        let phase2_ctx_delta = ctx_len.saturating_sub(phase2_ctx_start);

        for (layer_idx, layer) in self.head.layers.iter().enumerate() {
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
            // v0.74.1: K, V proj on cross-context rows — ONLY the new
            // delta `[phase2_ctx_start, ctx_len)`. Cached rows
            // `[0, phase2_ctx_start)` retain their post-norm post-RoPE
            // values from prior outer steps. Write into per-layer cache.
            for c in phase2_ctx_start..ctx_len {
                let row_in = self
                    .session
                    .ctx_h
                    .view_subrange((c * h) as u64, vec![h as u64]);
                let k_row = self.session.k_ctx_cache[layer_idx]
                    .view_subrange((c * kv_dim) as u64, vec![kv_dim as u64]);
                let v_row = self.session.v_ctx_cache[layer_idx]
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
            // v0.74.1: K_ctx norm — only the new delta. Pre-cached rows
            // were normed on a prior outer step's pass and retain their
            // post-norm values.
            if phase2_ctx_delta > 0 {
                let view = self.session.k_ctx_cache[layer_idx].view_subrange(
                    (phase2_ctx_start * kv_dim) as u64,
                    vec![(phase2_ctx_delta * kv_dim) as u64],
                );
                encode_rms_norm_batched_f32(
                    ctx_metal,
                    &enc,
                    &view,
                    &layer.k_norm,
                    &view,
                    phase2_ctx_delta * n_kv,
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
            // v0.74.1: RoPE K_ctx — only the new delta. Pre-cached
            // rows were RoPE'd on a prior outer step at their stable
            // pos_ctx[c] positions; positions don't change.
            for c in phase2_ctx_start..ctx_len {
                let pos = pos_ctx_cpu[c] as u32;
                let row = self.session.k_ctx_cache[layer_idx]
                    .view_subrange((c * kv_dim) as u64, vec![kv_dim as u64]);
                encode_rope_neox_f32(ctx_metal, &enc, &row, n_kv, head_dim, n_rot, pos, theta)?;
            }
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.session.maybe_record("phase2_proj_norm_rope", &cmd);

            // ----- Phase 3 (Metal, v0.72.1): attention + O proj + residual #1 +
            //       post-norm + SwiGLU FFN + residual #2. NO CPU readback. -----
            //
            // The v0.71 path read q/k/v + x back to CPU, ran scalar
            // attention with the SWA mask, did 4 mat-vecs per row × N
            // rows × 5 layers on CPU, and wrote x back. Cumulative:
            // ~12 s per outer step on Qwen3.6-27B-Q4_K_M.
            //
            // v0.72.1 replaces this with kernel_dflash_attn_f32
            // (custom small-N fused attention with per-layer SWA mask)
            // + per-row O proj + per-row FFN mat-vec on Metal. All in
            // one command buffer; no readback until the lm_head tail.
            //
            // Drafter weights are still F32 (dequant'd at load); v0.72.4
            // will switch to native Q8_0 mat-vec/mat-mat.
            let pos_k_uploaded;
            {
                // Build pos_k on host: pos_ctx (length ctx_len) ++
                // [noise_start_pos..noise_start_pos+N] (length N).
                let n_kv_total = ctx_len + n;
                pos_k_uploaded = n_kv_total;
                let mut pos_k_host: Vec<i32> = Vec::with_capacity(n_kv_total);
                pos_k_host.extend(pos_ctx_cpu.iter().take(ctx_len).copied());
                for i in 0..n {
                    pos_k_host.push((noise_start_pos + i as u32) as i32);
                }
                unsafe {
                    let dst = self.session.pos_k.buffer.contents().as_ptr() as *mut i32;
                    std::ptr::copy_nonoverlapping(pos_k_host.as_ptr(), dst, n_kv_total);
                }
            }

            let cmd = ctx_metal.queue.commandBuffer().expect("cmd phase3");
            let enc = KernelEncoder::begin(&cmd);

            // (a) Concat K_ctx + K_noise into k_full; same for V.
            //     k_full[0 .. ctx_len*kv_dim] <- k_ctx_cache[layer_idx][..ctx_len*kv_dim]
            //     k_full[ctx_len*kv_dim .. (ctx_len+N)*kv_dim] <- k_noise[..]
            // v0.74.1: read from per-layer K/V cache (post-norm post-RoPE).
            if ctx_len > 0 {
                let src_k_ctx = self.session.k_ctx_cache[layer_idx]
                    .view_subrange(0, vec![(ctx_len * kv_dim) as u64]);
                let src_v_ctx = self.session.v_ctx_cache[layer_idx]
                    .view_subrange(0, vec![(ctx_len * kv_dim) as u64]);
                encode_scatter_offset_f32(
                    ctx_metal,
                    &enc,
                    &src_k_ctx,
                    &self.session.k_full,
                    0,
                    ctx_len * kv_dim,
                )?;
                encode_scatter_offset_f32(
                    ctx_metal,
                    &enc,
                    &src_v_ctx,
                    &self.session.v_full,
                    0,
                    ctx_len * kv_dim,
                )?;
            }
            encode_scatter_offset_f32(
                ctx_metal,
                &enc,
                &self.session.k_noise,
                &self.session.k_full,
                ctx_len * kv_dim,
                n * kv_dim,
            )?;
            encode_scatter_offset_f32(
                ctx_metal,
                &enc,
                &self.session.v_noise,
                &self.session.v_full,
                ctx_len * kv_dim,
                n * kv_dim,
            )?;

            // (b) Slice the live regions of k_full/v_full/pos_k to the
            //     n_kv_total active rows. The rest is unused this layer.
            let n_kv_total = ctx_len + n;
            let k_view = self
                .session
                .k_full
                .view_subrange(0, vec![(n_kv_total * kv_dim) as u64]);
            let v_view = self
                .session
                .v_full
                .view_subrange(0, vec![(n_kv_total * kv_dim) as u64]);
            let pos_view = self.session.pos_k.view_subrange(0, vec![n_kv_total as u64]);

            // (c) Fused attention: writes attn_o_full [N, n_q*head_dim].
            let swa_window_arg = if layer.is_swa { cfg.swa_window } else { 0 };
            encode_dflash_attn_f32(
                ctx_metal,
                &enc,
                &self.session.q_buf,
                &k_view,
                &v_view,
                &pos_view,
                &self.session.attn_o_full,
                n,
                n_q,
                n_kv,
                head_dim,
                n_kv_total,
                ctx_len,
                noise_start_pos,
                swa_window_arg,
            )?;
            let _ = pos_k_uploaded;

            // (d) O proj — v0.74.2: batched mat-mat across all N noise
            //     rows when drafter weights are mat-mat eligible (Q8_0
            //     post-v0.73b.1; pre-v0.73b.1 was F32 dequant'd at load
            //     and per-row mat-vec was the only path). The reviewer's
            //     "you already built the kernels, now use them" find:
            //     `encode_mat_mat_dispatch` routes Q8_0 to the validated
            //     `kernel_mat_mat_q8_0_f32_n16` (A-lite gate 7.74×); the
            //     drafter weights have been native Q8_0 since v0.73b.1
            //     but draft_block phase 3 was still per-row mat-vec,
            //     dispatching 80 mat-vecs per outer step (5 layers × 16
            //     noise rows × 4 projections) where 4 mat-mats per layer
            //     suffice. Same playbook as v0.73a.1 GDN / v0.73c.1 attn:
            //     LOW correctness risk, reuses validated kernels.
            //
            //     Eligibility predicate matches the encode_mat_mat_dispatch
            //     supported set (Q4_K/Q5_K/Q6_K/Q8_0). F32 fall-through
            //     preserved for any non-batchable dtype (currently nothing
            //     in production hits it, but kept for the F32 oracle and
            //     future drafter quants).
            let drafter_mat_mat_eligible = |dtype: GgmlType| {
                matches!(
                    dtype,
                    GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0
                )
            };
            let phase3_batched = drafter_mat_mat_eligible(layer.o.dtype)
                && drafter_mat_mat_eligible(layer.ffn_gate.dtype)
                && drafter_mat_mat_eligible(layer.ffn_up.dtype)
                && drafter_mat_mat_eligible(layer.ffn_down.dtype);

            if phase3_batched {
                // Batched O proj: attn_o_full [N, q_dim] → ffn_out_buf [N, H].
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &layer.o,
                    &self.session.attn_o_full,
                    &self.session.ffn_out_buf,
                    q_dim,
                    h,
                    n,
                )?;
            } else {
                for i in 0..n {
                    let row_in = self
                        .session
                        .attn_o_full
                        .view_subrange((i * q_dim) as u64, vec![q_dim as u64]);
                    let row_out = self
                        .session
                        .ffn_out_buf
                        .view_subrange((i * h) as u64, vec![h as u64]);
                    encode_mat_vec_dispatch(
                        ctx_metal, &enc, &layer.o, &row_in, &row_out, q_dim, h,
                    )?;
                }
            }

            // (e) Residual #1: x += ffn_out_buf (reusing ffn_out_buf as
            //     a transient holder for the O proj output).
            encode_add_inplace_f32(ctx_metal, &enc, &self.session.x, &self.session.ffn_out_buf)?;

            // (f) Pre-FFN RMSNorm: x → h_buf (reuse session.h, batched).
            encode_rms_norm_batched_f32(
                ctx_metal,
                &enc,
                &self.session.x,
                &layer.post_attention_norm,
                &self.session.h,
                n,
                h,
                RMS_EPS,
            )?;

            // (g) SwiGLU FFN — v0.74.2: batched mat-mat ffn_gate / ffn_up
            //     / ffn_down when drafter weights are eligible. Same
            //     fall-through pattern as O-proj.
            if phase3_batched {
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &layer.ffn_gate,
                    &self.session.h,
                    &self.session.ffn_gate_buf,
                    h,
                    f,
                    n,
                )?;
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &layer.ffn_up,
                    &self.session.h,
                    &self.session.ffn_up_buf,
                    h,
                    f,
                    n,
                )?;
            } else {
                for i in 0..n {
                    let row_in = self.session.h.view_subrange((i * h) as u64, vec![h as u64]);
                    let gate_row = self
                        .session
                        .ffn_gate_buf
                        .view_subrange((i * f) as u64, vec![f as u64]);
                    let up_row = self
                        .session
                        .ffn_up_buf
                        .view_subrange((i * f) as u64, vec![f as u64]);
                    encode_mat_vec_dispatch(
                        ctx_metal,
                        &enc,
                        &layer.ffn_gate,
                        &row_in,
                        &gate_row,
                        h,
                        f,
                    )?;
                    encode_mat_vec_dispatch(
                        ctx_metal,
                        &enc,
                        &layer.ffn_up,
                        &row_in,
                        &up_row,
                        h,
                        f,
                    )?;
                }
            }
            // silu_mul over the entire N*F flat buffer (elementwise) —
            // unchanged whether the gate/up paths were batched or per-row;
            // the byte layout is bit-identical.
            encode_silu_mul_f32(
                ctx_metal,
                &enc,
                &self.session.ffn_gate_buf,
                &self.session.ffn_up_buf,
                &self.session.ffn_inner_buf,
            )?;
            // Batched ffn_down: ffn_inner_buf [N, F] → ffn_out_buf [N, H].
            if phase3_batched {
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &layer.ffn_down,
                    &self.session.ffn_inner_buf,
                    &self.session.ffn_out_buf,
                    f,
                    h,
                    n,
                )?;
            } else {
                for i in 0..n {
                    let row_in = self
                        .session
                        .ffn_inner_buf
                        .view_subrange((i * f) as u64, vec![f as u64]);
                    let row_out = self
                        .session
                        .ffn_out_buf
                        .view_subrange((i * h) as u64, vec![h as u64]);
                    encode_mat_vec_dispatch(
                        ctx_metal,
                        &enc,
                        &layer.ffn_down,
                        &row_in,
                        &row_out,
                        f,
                        h,
                    )?;
                }
            }
            // (h) Residual #2: x += ffn_out_buf.
            encode_add_inplace_f32(ctx_metal, &enc, &self.session.x, &self.session.ffn_out_buf)?;

            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.session
                .maybe_record("phase3_attn_oproj_ffn_residuals", &cmd);
        }

        // v0.74.1: all drafter layers now have post-norm post-RoPE
        // K/V cached for `[0, ctx_len)`. Advance the watermark so the
        // next outer step only projects the new appended positions.
        self.session.kv_ctx_ready_n = ctx_len;

        // ----- Phase 4 (Metal): batched final norm + lm_head + argmax -----
        //
        // v0.72.0 — port the H5.3b.6 batched-tail pattern from
        // packed_verify into draft_block. Codex Q3 from the v0.72-design
        // session: drafter shares target's lm_head (Q6_K), so the same
        // encode_mat_mat_dispatch + encode_argmax_f32 path applies.
        //
        // Replaces the per-row mat-vec lm_head loop (16 dispatches +
        // 16x weight re-read of ~1 GiB Q6_K = ~16 GiB redundant
        // traffic per outer step) with ONE mat-mat dispatch. Also
        // replaces the CPU `[N, V] -> argmax` readback (~16 MB
        // per outer step + scalar loop) with one GPU argmax dispatch
        // and an `[N]` i32 readback (64 B). Both wins compound.
        //
        // Falls back to per-row mat-vec for non-mat-mat-eligible
        // lm_head dtypes (F32 0.8B oracle path).
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
        let lm_dtype = self.base.model.lm_head.dtype;
        let lm_mat_mat_path = matches!(lm_dtype, GgmlType::Q4_K | GgmlType::Q6_K);
        if lm_mat_mat_path {
            encode_mat_mat_dispatch(
                ctx_metal,
                &enc,
                &self.base.model.lm_head,
                &self.session.h,
                &self.session.draft_logits,
                h,
                v,
                n,
            )?;
        } else {
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
        }
        encode_argmax_f32(
            ctx_metal,
            &enc,
            &self.session.draft_logits,
            &self.session.draft_argmax,
            n,
            v,
        )?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        self.session
            .maybe_record("phase4_tail_norm_lmhead_argmax", &cmd);

        // Read back `[N]` i32 argmaxes (64 B, vs the v0.71 per-token
        // `[V]` F32 readback = 16 MB/outer step at V=248320, N=16).
        let mut argmaxes = vec![0i32; n];
        unsafe {
            let src = self.session.draft_argmax.buffer.contents().as_ptr() as *const i32;
            std::ptr::copy_nonoverlapping(src, argmaxes.as_mut_ptr(), n);
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

// H5.1.5 metal_drafter_cosine_vs_cpu moved to tests/dflash_correctness.rs
// (slow: ~142s on 27B-Q4_K_M prefill + drafter forward; not a fast-
// feedback gate). Run with `cargo test --test dflash_correctness --release`.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::loader::Model;
    use crate::metal::MetalContext;
    use crate::metal_forward::{MetalModel, MetalSession};

    /// H5.3a foundation: verify `MetalDFlashVerifyScratch` allocates
    /// correctly-sized buffers, and that `slot_view` helpers land at
    /// the right offsets with the right shapes. Uses the 0.8B oracle
    /// (24 layers, all GDN — so n_gdn = n_layer = 24, smaller than
    /// 27B's 48). Loads in <1 s.
    #[test]
    fn dflash_verify_scratch_slots_and_offsets() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[dflash-scratch] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        // Pretend we have a DFlash drafter with N=8, K=3 (synthetic;
        // doesn't have to match a real drafter — we're only testing the
        // scratch struct's offset arithmetic against the 0.8B target arch).
        let n: u32 = 8;
        let k: u32 = 3;
        let scratch = MetalDFlashVerifyScratch::fresh(&ctx, &mm, n, k).expect("scratch alloc");

        // Sanity on cached dims.
        let arch = &mm.arch;
        assert_eq!(scratch.n, n);
        assert_eq!(scratch.k_target_layers, k);
        assert_eq!(scratch.hidden_size, arch.hidden_size as u64);
        let expected_ssm =
            (arch.gdn_n_v_heads as u64) * (arch.gdn_head_dim as u64) * (arch.gdn_head_dim as u64);
        let expected_conv = ((arch.gdn_conv_kernel as u64) - 1)
            * (2 * (arch.gdn_n_k_heads as u64) + (arch.gdn_n_v_heads as u64))
            * (arch.gdn_head_dim as u64);
        assert_eq!(scratch.ssm_state_elems, expected_ssm);
        assert_eq!(scratch.conv_state_elems, expected_conv);
        let expected_n_gdn = mm
            .blocks
            .iter()
            .filter(|b| matches!(b, crate::metal_forward::MetalBlock::Gdn(_)))
            .count() as u32;
        assert_eq!(scratch.n_gdn_layers, expected_n_gdn);
        eprintln!(
            "[dflash-scratch] H={} n_gdn={} ssm_elems={} conv_elems={}",
            scratch.hidden_size,
            scratch.n_gdn_layers,
            scratch.ssm_state_elems,
            scratch.conv_state_elems
        );

        // -- backing buffer sizes --
        let f32_size = std::mem::size_of::<f32>() as u64;
        assert_eq!(
            scratch.gdn_ckpt.shape,
            vec![scratch.n_gdn_layers as u64, n as u64, expected_ssm]
        );
        assert_eq!(
            scratch.gdn_ckpt.n_elements(),
            scratch.n_gdn_layers as u64 * n as u64 * expected_ssm
        );
        assert_eq!(
            scratch.conv_ckpt.shape,
            vec![scratch.n_gdn_layers as u64, n as u64, expected_conv]
        );
        // v0.71: layout switched from [K, N, H] to [N, K, H] for
        // contiguous-by-N reads (target_ctx append in H5.5 outer loop).
        assert_eq!(
            scratch.hidden_capture.shape,
            vec![n as u64, k as u64, scratch.hidden_size]
        );
        assert_eq!(scratch.packed_ids_buf.shape, vec![n as u64]);
        assert_eq!(scratch.verify_argmax.shape, vec![n as u64]);

        // -- gdn_ckpt_slot offsets --
        // Slot (layer, n) should land at offset (layer * N + n) * ssm_elems
        // F32 elements. View shape = [ssm_elems].
        for layer in 0..scratch.n_gdn_layers {
            for nn in 0..n {
                let slot = scratch.gdn_ckpt_slot(layer, nn);
                let expected_elem_off = (layer as u64 * n as u64 + nn as u64) * expected_ssm;
                let expected_byte_off = expected_elem_off * f32_size;
                assert_eq!(
                    slot.shape,
                    vec![expected_ssm],
                    "gdn slot ({layer},{nn}) shape"
                );
                assert_eq!(
                    slot.offset, expected_byte_off,
                    "gdn slot ({layer},{nn}) byte offset"
                );
                // Slot must share the underlying buffer with the parent.
                let slot_buf_ptr: *const _ = &*slot.buffer;
                let parent_buf_ptr: *const _ = &*scratch.gdn_ckpt.buffer;
                assert_eq!(
                    slot_buf_ptr, parent_buf_ptr,
                    "gdn slot does not share buffer with parent"
                );
            }
        }

        // -- conv_ckpt_slot offsets --
        for layer in 0..scratch.n_gdn_layers.min(4) {
            for nn in [0, n / 2, n - 1] {
                let slot = scratch.conv_ckpt_slot(layer, nn);
                let expected_elem_off = (layer as u64 * n as u64 + nn as u64) * expected_conv;
                assert_eq!(slot.shape, vec![expected_conv]);
                assert_eq!(slot.offset, expected_elem_off * f32_size);
            }
        }

        // -- hidden_capture_slot offsets (v0.71: [N, K, H] layout) --
        for kk in 0..k {
            for nn in 0..n {
                let slot = scratch.hidden_capture_slot(kk, nn);
                let expected_elem_off = (nn as u64 * k as u64 + kk as u64) * scratch.hidden_size;
                assert_eq!(slot.shape, vec![scratch.hidden_size]);
                assert_eq!(slot.offset, expected_elem_off * f32_size);
            }
        }
        // -- hidden_capture_n_slot (NEW v0.71): K*H contiguous per token --
        for nn in 0..n {
            let n_slot = scratch.hidden_capture_n_slot(nn);
            let kh = k as u64 * scratch.hidden_size;
            assert_eq!(n_slot.shape, vec![kh]);
            assert_eq!(n_slot.offset, (nn as u64) * kh * f32_size);
        }

        // -- token_slot / argmax_slot — single-element views --
        for nn in 0..n {
            let tok_slot = scratch.token_slot(nn);
            assert_eq!(tok_slot.shape, vec![1]);
            assert_eq!(tok_slot.offset, (nn as u64) * f32_size);
            let am_slot = scratch.argmax_slot(nn);
            assert_eq!(am_slot.shape, vec![1]);
            assert_eq!(am_slot.offset, (nn as u64) * f32_size);
        }

        // -- write/read round-trip via a slot, to confirm the underlying
        //    buffer offset actually addresses what we think it does. We
        //    write a sentinel through gdn_ckpt_slot(layer=2, n=3) and
        //    read it back through the parent's contents() pointer at
        //    the same byte offset.
        {
            let layer = 2u32;
            let nn = 3u32;
            let slot = scratch.gdn_ckpt_slot(layer, nn);
            // Write 'sentinel' as the FIRST element of the slot.
            unsafe {
                let p = (slot.buffer.contents().as_ptr() as *mut u8).add(slot.offset as usize)
                    as *mut f32;
                *p = 1234.5;
            }
            // Read through the PARENT buffer at the computed byte offset.
            let parent_byte_off = ((layer as u64 * n as u64 + nn as u64) * expected_ssm) * f32_size;
            unsafe {
                let p = (scratch.gdn_ckpt.buffer.contents().as_ptr() as *const u8)
                    .add(parent_byte_off as usize) as *const f32;
                assert!(
                    (*p - 1234.5).abs() < 1e-9,
                    "round-trip via slot got {} expected 1234.5",
                    *p
                );
            }
        }
    }

    /// H5.3a gate G1 (lite) + G5: packed_verify produces the SAME
    /// argmax tokens as N successive `single_token` calls from a fresh
    /// session. Headline correctness signal for the H5.3a scaffold —
    /// proves:
    ///   * packed semantics (residual stream evolution, KV append,
    ///     GDN+conv state evolution) match N single-token decode
    ///   * GPU argmax (lowest-index tie policy) matches CPU argmax
    ///   * packed_ids_buf reads the right slot per block (the codex Q7
    ///     mitigation; if this were broken, every get_rows would read
    ///     the same stale token id and all argmaxes would equal each
    ///     other or be silently wrong)
    ///   * codex Q2 design Y (batched-end-of-token blits) doesn't
    ///     break correctness — if the blit pass were perturbing later
    ///     tokens, the second/third token argmaxes would diverge
    ///
    /// Also pulls in gate G2 lite: post-packed `gdn_state[k]`,
    /// `gdn_conv[k]`, and `kv_n_pos` must match post-N-single-token
    /// session state (proves the per-token blits captured the same
    /// bytes the in-place updates produced).
    ///
    /// Does NOT yet verify (later H5.3a gates):
    ///   * checkpoint slot CONTENTS at intermediate n (G3 — needs
    ///     restore primitive to validate)
    ///   * hidden capture layout (G4 — separate test)
    ///   * cosine ≥ 0.9999 on raw logits (G1 full — needs _with_logits)
    ///
    /// 0.8B-F32, N=4. Loads in ~500 ms; total runtime ≤ 2 s on M4 Max.
    #[test]
    fn dflash_packed_verify_argmax_matches_n_single_tokens() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[dflash-packed-verify] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        // Baseline: N successive single_token calls on a fresh session.
        let n: u32 = 4;
        let start_position: u32 = 0;
        let tokens: Vec<i32> = vec![9419, 1, 5, 1234];
        assert_eq!(tokens.len() as u32, n);

        let mut single_session = MetalSession::fresh(&ctx, &mm, 64).expect("session 1");
        let mut single_argmaxes = Vec::with_capacity(n as usize);
        for (i, &tok) in tokens.iter().enumerate() {
            let logits = mf
                .single_token(tok, start_position + i as u32, &mut single_session)
                .expect("single token");
            // CPU argmax with lowest-index tie (matches kernel_argmax_f32).
            let mut best = f32::NEG_INFINITY;
            let mut idx: i32 = 0;
            for (j, &v) in logits.iter().enumerate() {
                if v > best {
                    best = v;
                    idx = j as i32;
                }
            }
            single_argmaxes.push(idx);
        }
        eprintln!("[dflash-packed-verify] single argmaxes: {single_argmaxes:?}");

        // Packed: one packed_verify call from a FRESH session.
        let target_layer_ids: Vec<u32> = vec![5, 15];
        let k_target_layers = target_layer_ids.len() as u32;
        let mut packed_session = MetalSession::fresh(&ctx, &mm, 64).expect("session 2");
        let mut scratch =
            MetalDFlashVerifyScratch::fresh(&ctx, &mm, n, k_target_layers).expect("scratch");

        let packed_argmaxes = encode_packed_verify_inner(
            &mf,
            &target_layer_ids,
            &tokens,
            start_position,
            &mut scratch,
            &mut packed_session,
        )
        .expect("packed verify");
        eprintln!("[dflash-packed-verify] packed argmaxes: {packed_argmaxes:?}");

        assert_eq!(
            packed_argmaxes, single_argmaxes,
            "G1/G5: packed_verify argmaxes must match N successive single_token argmaxes"
        );

        // Bonus: gate G2 lite — post-packed session state matches
        // post-N-single-token session state.
        //
        // **Bitwise equality, NOT a slack tolerance.** Per codex
        // open-ended review: this path is literally identical
        // dispatch order and identical state evolution (packed_verify
        // is N sequential single_token encodes inside one cmd
        // buffer; same kernels, same args, same bind order). If
        // bitwise eq fails, that is a real signal — not noise to
        // be papered over with `< 1e-5`. Keep the bar.
        for (i, (s_state, p_state)) in single_session
            .gdn_state
            .iter()
            .zip(packed_session.gdn_state.iter())
            .enumerate()
        {
            unsafe {
                let s = s_state.buffer.contents().as_ptr() as *const u32;
                let p = p_state.buffer.contents().as_ptr() as *const u32;
                let n_elems = s_state.n_elements() as usize;
                for j in 0..n_elems {
                    let sv = *s.add(j);
                    let pv = *p.add(j);
                    if sv != pv {
                        let sf = f32::from_bits(sv);
                        let pf = f32::from_bits(pv);
                        panic!(
                            "G2: gdn_state[{i}][{j}] bitwise mismatch: \
                             single={sf} (0x{sv:08x}) packed={pf} (0x{pv:08x}) \
                             Δ={}",
                            sf - pf
                        );
                    }
                }
            }
        }
        for (i, (s_conv, p_conv)) in single_session
            .gdn_conv
            .iter()
            .zip(packed_session.gdn_conv.iter())
            .enumerate()
        {
            unsafe {
                let s = s_conv.buffer.contents().as_ptr() as *const u32;
                let p = p_conv.buffer.contents().as_ptr() as *const u32;
                let n_elems = s_conv.n_elements() as usize;
                for j in 0..n_elems {
                    let sv = *s.add(j);
                    let pv = *p.add(j);
                    if sv != pv {
                        let sf = f32::from_bits(sv);
                        let pf = f32::from_bits(pv);
                        panic!(
                            "G2: gdn_conv[{i}][{j}] bitwise mismatch: \
                             single={sf} (0x{sv:08x}) packed={pf} (0x{pv:08x}) \
                             Δ={}",
                            sf - pf
                        );
                    }
                }
            }
        }
        assert_eq!(
            single_session.kv_n_pos, packed_session.kv_n_pos,
            "G2: kv_n_pos diverged"
        );
    }

    /// H5.3b.4-5 headline correctness gate: layer-major
    /// `packed_verify` produces BIT-EXACT identical argmaxes AND
    /// session state to the token-major oracle on the same inputs.
    ///
    /// The two paths use the same kernels with different scheduling
    /// (token-major: outer-loop over tokens, inner-loop over layers;
    /// layer-major: outer-loop over layers, inner-loop or batched
    /// across tokens). On F32 weights both should produce bit-
    /// identical bytes because the math is identical — only the
    /// dispatch order differs, and same-encoder same-stream Metal
    /// dispatches are deterministic.
    ///
    /// Per codex Q4 + the codex layer-major partner-session failure-
    /// mode prediction: "argmax + final-state gates can mask shape-
    /// only bugs on lucky logits." Therefore this test ALSO compares
    /// raw `[N, V]` logits row-by-row via the `_with_logits` debug
    /// variants, requiring bit-exact match.
    ///
    /// 0.8B-F32, M=2 prime + N=4 verify. ≤ 2 s.
    #[test]
    fn dflash_packed_verify_layer_major_matches_token_major() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        const M: u32 = 2;
        const N: u32 = 4;
        let prime_tokens: [i32; M as usize] = [9419, 1];
        let verify_tokens: [i32; N as usize] = [1234, 7, 999, 42];
        let target_layer_ids: Vec<u32> = vec![5, 15];
        let k_target = target_layer_ids.len() as u32;

        // Two identically-primed sessions.
        let mut sess_tok = MetalSession::fresh(&ctx, &mm, 64).expect("sess tok");
        let mut sess_lm = MetalSession::fresh(&ctx, &mm, 64).expect("sess lm");
        for (i, &tok) in prime_tokens.iter().enumerate() {
            mf.single_token(tok, i as u32, &mut sess_tok)
                .expect("prime tok");
            mf.single_token(tok, i as u32, &mut sess_lm)
                .expect("prime lm");
        }

        // Token-major path with logits.
        let mut dbg_tok =
            MetalDFlashDebugScratch::fresh(&ctx, &mm, N, k_target).expect("dbg scratch tok");
        let argmax_tok = encode_packed_verify_with_logits_inner(
            &mf,
            &target_layer_ids,
            &verify_tokens,
            M,
            &mut dbg_tok,
            &mut sess_tok,
        )
        .expect("token-major");

        // Layer-major path with logits.
        let mut dbg_lm =
            MetalDFlashDebugScratch::fresh(&ctx, &mm, N, k_target).expect("dbg scratch lm");
        let mut layer_scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, N).expect("layer scratch");
        let MetalDFlashDebugScratch {
            verify: lm_verify,
            debug_logits: lm_debug,
        } = &mut dbg_lm;
        let argmax_lm = encode_packed_verify_layer_major_inner(
            &mf,
            &target_layer_ids,
            &verify_tokens,
            M,
            lm_verify,
            &mut layer_scratch,
            &mut sess_lm,
            Some(lm_debug),
        )
        .expect("layer-major");

        eprintln!(
            "[layer-major-vs-token-major] argmax_tok={argmax_tok:?} \
             argmax_lm={argmax_lm:?}"
        );

        // Bit-exact argmax tokens.
        assert_eq!(
            argmax_tok, argmax_lm,
            "layer-major argmax tokens diverge from token-major"
        );

        // Bit-exact raw logits (codex's intermediate-layer paranoia
        // gate; argmax alone could pass even if intermediate layouts
        // were silently transposed for some shapes).
        let v = m.arch.vocab_size as usize;
        unsafe {
            let p_tok = dbg_tok.debug_logits.buffer.contents().as_ptr() as *const u32;
            let p_lm = dbg_lm.debug_logits.buffer.contents().as_ptr() as *const u32;
            for i in 0..(N as usize) * v {
                let t = *p_tok.add(i);
                let l = *p_lm.add(i);
                if t != l {
                    let n_idx = i / v;
                    let vocab_idx = i % v;
                    panic!(
                        "logits bit-mismatch at n_idx={n_idx} vocab_idx={vocab_idx}: \
                         token-major=0x{t:08x} (={}) layer-major=0x{l:08x} (={})",
                        f32::from_bits(t),
                        f32::from_bits(l)
                    );
                }
            }
        }

        // Bit-exact session state.
        for (i, (a, b)) in sess_tok
            .gdn_state
            .iter()
            .zip(sess_lm.gdn_state.iter())
            .enumerate()
        {
            unsafe {
                let pa = a.buffer.contents().as_ptr() as *const u32;
                let pb = b.buffer.contents().as_ptr() as *const u32;
                let n_elems = a.n_elements() as usize;
                for j in 0..n_elems {
                    if *pa.add(j) != *pb.add(j) {
                        panic!(
                            "gdn_state[{i}][{j}] diverges between token-major and \
                             layer-major after the same N-token batch"
                        );
                    }
                }
            }
        }
        for (i, (a, b)) in sess_tok
            .gdn_conv
            .iter()
            .zip(sess_lm.gdn_conv.iter())
            .enumerate()
        {
            unsafe {
                let pa = a.buffer.contents().as_ptr() as *const u32;
                let pb = b.buffer.contents().as_ptr() as *const u32;
                let n_elems = a.n_elements() as usize;
                for j in 0..n_elems {
                    if *pa.add(j) != *pb.add(j) {
                        panic!(
                            "gdn_conv[{i}][{j}] diverges between token-major and \
                             layer-major"
                        );
                    }
                }
            }
        }
        assert_eq!(
            sess_tok.kv_n_pos, sess_lm.kv_n_pos,
            "kv_n_pos diverges between token-major and layer-major"
        );
    }

    /// H5.3a guard-wall test (codex failure-mode mitigation): if the
    /// scratch was allocated with a different `block_size` /
    /// `target_layer_ids.len()` / model arch, packed_verify must
    /// FAIL LOUDLY at entry, not silently corrupt downstream blits.
    /// Catches the scratch/model/session dimensional drift class
    /// codex flagged.
    #[test]
    fn dflash_packed_verify_dim_guard_wall() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        let mut session = MetalSession::fresh(&ctx, &mm, 64).expect("session");
        let target_layer_ids: Vec<u32> = vec![5, 10, 15];
        let mut scratch = MetalDFlashVerifyScratch::fresh(&ctx, &mm, 4, 3).expect("scratch");

        // (a) tokens.len() != scratch.n
        let bad_tokens = vec![1i32, 2, 3];
        let err = encode_packed_verify_inner(
            &mf,
            &target_layer_ids,
            &bad_tokens,
            0,
            &mut scratch,
            &mut session,
        );
        match err {
            Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                assert!(detail.contains("tokens.len()"), "wrong error: {detail}");
            }
            other => panic!("expected BadShape on tokens.len mismatch, got {other:?}"),
        }

        // (b) target_layer_ids.len() != scratch.k_target_layers
        let bad_layers: Vec<u32> = vec![5, 15];
        let err = encode_packed_verify_inner(
            &mf,
            &bad_layers,
            &[1i32, 2, 3, 4],
            0,
            &mut scratch,
            &mut session,
        );
        match err {
            Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                assert!(detail.contains("k_target_layers"), "wrong error: {detail}");
            }
            other => panic!("expected BadShape on k mismatch, got {other:?}"),
        }

        // (c) start_position + N > kv_capacity
        let err = encode_packed_verify_inner(
            &mf,
            &target_layer_ids,
            &[1i32, 2, 3, 4],
            61, // 61 + 4 = 65 > 64
            &mut scratch,
            &mut session,
        );
        match err {
            Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                assert!(detail.contains("kv_capacity"), "wrong error: {detail}");
            }
            other => panic!("expected BadShape on kv overflow, got {other:?}"),
        }

        // (d) bad token id
        let bad_token: i32 = m.arch.vocab_size as i32 + 100;
        let err = encode_packed_verify_inner(
            &mf,
            &target_layer_ids,
            &[1i32, 2, bad_token, 4],
            0,
            &mut scratch,
            &mut session,
        );
        match err {
            Err(DFlashError::BadToken(t, _)) => assert_eq!(t, bad_token),
            other => panic!("expected BadToken, got {other:?}"),
        }

        // (e) target_layer_id out of range
        let bad_layers: Vec<u32> = vec![5, 999, 15]; // 999 > 0.8B's 24 layers
        let mut scratch2 = MetalDFlashVerifyScratch::fresh(&ctx, &mm, 4, 3).expect("scratch2");
        let err = encode_packed_verify_inner(
            &mf,
            &bad_layers,
            &[1i32, 2, 3, 4],
            0,
            &mut scratch2,
            &mut session,
        );
        match err {
            Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                assert!(detail.contains("layer id"), "wrong error: {detail}");
            }
            other => panic!("expected BadShape on layer id, got {other:?}"),
        }
    }

    /// H5.3a guard test (codex biggest miss): the session must
    /// represent the prefix ending at `start_position`. A misaligned
    /// session (kv_n_pos != start_position) must be rejected loudly,
    /// not silently produce wrong results.
    ///
    /// Two scenarios:
    ///   (a) Fresh session (kv_n_pos=0) called with start_position>0
    ///       — should fail.
    ///   (b) Stale session (kv_n_pos=K from prior decode) called with
    ///       start_position != K — should fail.
    #[test]
    fn dflash_packed_verify_kv_n_pos_guard() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        let target_layer_ids: Vec<u32> = vec![5, 15];
        let mut scratch = MetalDFlashVerifyScratch::fresh(&ctx, &mm, 4, 2).expect("scratch");

        // Scenario (a): fresh session (kv_n_pos all 0), start_position=5.
        let mut fresh_session = MetalSession::fresh(&ctx, &mm, 64).expect("session");
        let err = encode_packed_verify_inner(
            &mf,
            &target_layer_ids,
            &[1i32, 2, 3, 4],
            5,
            &mut scratch,
            &mut fresh_session,
        );
        match err {
            Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                assert!(detail.contains("kv_n_pos"), "wrong error: {detail}");
                assert!(detail.contains("start_position"), "wrong error: {detail}");
            }
            other => panic!("expected BadShape on fresh-session/start>0, got {other:?}"),
        }

        // 0.8B has all GDN layers — no attn — so `kv_n_pos` is empty.
        // Skip the stale-session check; it's better exercised at 27B.
        // But we DO want to confirm the guard exits cleanly when the
        // vec is empty (passes through trivially: no entries to
        // disagree). I.e. with no attn layers, fresh session at
        // start_position=0 passes the guard.
        eprintln!(
            "[dflash-kv-n-pos-guard] 0.8B fresh kv_n_pos.len={} (all GDN layers)",
            fresh_session.kv_n_pos.len()
        );
    }

    /// H5.3a G2++ via PRIMED session: prove packed_verify works
    /// correctly when the session is partway through a generation,
    /// i.e. the kv_n_pos==start_position guard isn't masking a bug
    /// where we silently DROP previously-encoded state.
    ///
    /// Setup:
    ///   1. Run M=2 single_token calls on session_A starting from
    ///      tokens[0..M]. Session_A.kv_n_pos == M after.
    ///   2. Run packed_verify(tokens[M..M+N], start_position=M)
    ///      against session_A. Expected: argmaxes match
    ///      tokens[M+1..M+N+1]'s argmax under continued single_token
    ///      decode.
    ///   3. Compare against single_token continued for N more steps
    ///      on session_B (also primed identically through M).
    ///
    /// This is the test codex specifically called out as more
    /// important than the cosine gate before writing restore.
    #[test]
    fn dflash_packed_verify_with_primed_session() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        // Prime BOTH sessions identically through M tokens.
        const M: u32 = 3; // priming length
        const N: u32 = 4; // packed verify length
        let prime_tokens: [i32; M as usize] = [9419, 1, 5];
        let verify_tokens: [i32; N as usize] = [1234, 7, 999, 42];

        let mut session_a = MetalSession::fresh(&ctx, &mm, 64).expect("session A");
        let mut session_b = MetalSession::fresh(&ctx, &mm, 64).expect("session B");
        for (i, &tok) in prime_tokens.iter().enumerate() {
            mf.single_token(tok, i as u32, &mut session_a)
                .expect("prime A");
            mf.single_token(tok, i as u32, &mut session_b)
                .expect("prime B");
        }
        // Note: 0.8B has no attn layers, so kv_n_pos is empty — the
        // guard trivially passes regardless of M. The test still
        // proves the GDN+conv state evolution is correct under
        // start_position > 0 on packed_verify.

        // Continue B with N single_token calls; collect argmaxes.
        let mut single_argmaxes = Vec::with_capacity(N as usize);
        for (i, &tok) in verify_tokens.iter().enumerate() {
            let logits = mf
                .single_token(tok, M + i as u32, &mut session_b)
                .expect("continue B");
            let mut best = f32::NEG_INFINITY;
            let mut idx: i32 = 0;
            for (j, &v) in logits.iter().enumerate() {
                if v > best {
                    best = v;
                    idx = j as i32;
                }
            }
            single_argmaxes.push(idx);
        }

        // Run packed_verify on A starting at start_position=M.
        let target_layer_ids: Vec<u32> = vec![5, 15];
        let mut scratch = MetalDFlashVerifyScratch::fresh(&ctx, &mm, N, 2).expect("scratch");
        let packed_argmaxes = encode_packed_verify_inner(
            &mf,
            &target_layer_ids,
            &verify_tokens,
            M,
            &mut scratch,
            &mut session_a,
        )
        .expect("packed verify with primed session");

        eprintln!(
            "[dflash-primed] M={M} N={N} \
             single_argmaxes={single_argmaxes:?} \
             packed_argmaxes={packed_argmaxes:?}"
        );
        assert_eq!(
            packed_argmaxes, single_argmaxes,
            "packed_verify on primed session must match continued single_token"
        );

        // Bitwise GDN+conv state equivalence post-packed vs post-single.
        for (i, (s_state, p_state)) in session_b
            .gdn_state
            .iter()
            .zip(session_a.gdn_state.iter())
            .enumerate()
        {
            unsafe {
                let s = s_state.buffer.contents().as_ptr() as *const u32;
                let p = p_state.buffer.contents().as_ptr() as *const u32;
                let n_elems = s_state.n_elements() as usize;
                for j in 0..n_elems {
                    if *s.add(j) != *p.add(j) {
                        panic!(
                            "primed-G2: gdn_state[{i}][{j}] bitwise mismatch \
                             after primed packed_verify"
                        );
                    }
                }
            }
        }
    }

    /// H5.3a gate G3 (checkpoint replay equivalence) + G6 (restore
    /// boundary cases): the headline correctness gate for the
    /// rollback primitive. Per codex H5.3a review: cosine is independent
    /// of restore; restore unblocks G3+G6, which prove the checkpoint
    /// CONTENTS at intermediate n are correct (not just final state).
    ///
    /// Setup: prime two fresh sessions identically through M tokens.
    /// Run packed_verify(verify_tokens, start_position=M) on session_A.
    /// For each n_keep ∈ {1, N/2, N}:
    ///   * Restore session_A to n_keep.
    ///   * Run one single_token at position M + n_keep on session_A
    ///     with a marker token.
    ///   * On session_B (separately primed), run n_keep single_tokens
    ///     of verify_tokens[0..n_keep], then one single_token of the
    ///     marker. session_A and session_B should now have BIT-EXACT
    ///     gdn_state, gdn_conv, kv_n_pos, AND argmax token.
    ///
    /// This is the strongest possible test of the rollback semantics.
    /// If checkpoint slot CONTENTS are wrong (e.g., off-by-one indexing),
    /// session_A's post-restore state diverges from session_B's
    /// "ground-truth" sequential state and we catch it.
    ///
    /// 0.8B-F32, M=2 prime + N=4 verify. ≤ 5 s on M4 Max.
    #[test]
    fn dflash_restore_after_partial_accept_replay_equivalence() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        const M: u32 = 2;
        const N: u32 = 4;
        let prime_tokens: [i32; M as usize] = [9419, 1];
        let verify_tokens: [i32; N as usize] = [1234, 7, 999, 42];
        let marker_token: i32 = 555; // post-restore single_token input

        let target_layer_ids: Vec<u32> = vec![5, 15];
        let k_target_layers = target_layer_ids.len() as u32;

        // G6 boundary set: n_keep = 1 (full reject; carry only),
        // n_keep = N/2 (typical partial), n_keep = N (full accept).
        for &n_keep in &[1u32, N / 2, N] {
            // -- session_A: prime + packed_verify + restore + one
            //    single_token at M + n_keep.
            let mut sess_a = MetalSession::fresh(&ctx, &mm, 64).expect("sess A");
            for (i, &tok) in prime_tokens.iter().enumerate() {
                mf.single_token(tok, i as u32, &mut sess_a)
                    .expect("prime A");
            }
            let mut scratch =
                MetalDFlashVerifyScratch::fresh(&ctx, &mm, N, k_target_layers).expect("scratch");
            let _packed = encode_packed_verify_inner(
                &mf,
                &target_layer_ids,
                &verify_tokens,
                M,
                &mut scratch,
                &mut sess_a,
            )
            .expect("packed verify");

            encode_restore_after_partial_accept_inner(&mf, &scratch, n_keep, M, &mut sess_a)
                .expect("restore");

            let logits_a = mf
                .single_token(marker_token, M + n_keep, &mut sess_a)
                .expect("marker on A");
            let mut argmax_a: i32 = 0;
            let mut best = f32::NEG_INFINITY;
            for (j, &v) in logits_a.iter().enumerate() {
                if v > best {
                    best = v;
                    argmax_a = j as i32;
                }
            }

            // -- session_B: prime + n_keep single_tokens through
            //    verify_tokens[0..n_keep] + one single_token of marker.
            //    This is the "ground truth" sequential trajectory.
            let mut sess_b = MetalSession::fresh(&ctx, &mm, 64).expect("sess B");
            for (i, &tok) in prime_tokens.iter().enumerate() {
                mf.single_token(tok, i as u32, &mut sess_b)
                    .expect("prime B");
            }
            for i in 0..n_keep {
                mf.single_token(verify_tokens[i as usize], M + i, &mut sess_b)
                    .expect("kept verify token on B");
            }
            let logits_b = mf
                .single_token(marker_token, M + n_keep, &mut sess_b)
                .expect("marker on B");
            let mut argmax_b: i32 = 0;
            let mut best = f32::NEG_INFINITY;
            for (j, &v) in logits_b.iter().enumerate() {
                if v > best {
                    best = v;
                    argmax_b = j as i32;
                }
            }

            eprintln!(
                "[restore-replay n_keep={n_keep}] argmax_A={argmax_a} \
                 argmax_B={argmax_b}"
            );

            // G3: argmax tokens must match (covers logits-after-restore
            // equivalence at the argmax-coarsened level).
            assert_eq!(
                argmax_a, argmax_b,
                "G3: argmax post-restore differs at n_keep={n_keep}: \
                 A={argmax_a} B={argmax_b}"
            );

            // G3 (stronger): bitwise equality on gdn_state / gdn_conv
            // after the marker single_token. The marker is processed
            // identically on both paths so divergence indicates a
            // restore bug, not a forward bug.
            for (i, (a_state, b_state)) in sess_a
                .gdn_state
                .iter()
                .zip(sess_b.gdn_state.iter())
                .enumerate()
            {
                unsafe {
                    let a = a_state.buffer.contents().as_ptr() as *const u32;
                    let b = b_state.buffer.contents().as_ptr() as *const u32;
                    let n_elems = a_state.n_elements() as usize;
                    for j in 0..n_elems {
                        if *a.add(j) != *b.add(j) {
                            let af = f32::from_bits(*a.add(j));
                            let bf = f32::from_bits(*b.add(j));
                            panic!(
                                "G3: gdn_state[{i}][{j}] post-restore-then-marker \
                                 differs at n_keep={n_keep}: A={af} B={bf}"
                            );
                        }
                    }
                }
            }
            for (i, (a_conv, b_conv)) in sess_a
                .gdn_conv
                .iter()
                .zip(sess_b.gdn_conv.iter())
                .enumerate()
            {
                unsafe {
                    let a = a_conv.buffer.contents().as_ptr() as *const u32;
                    let b = b_conv.buffer.contents().as_ptr() as *const u32;
                    let n_elems = a_conv.n_elements() as usize;
                    for j in 0..n_elems {
                        if *a.add(j) != *b.add(j) {
                            panic!(
                                "G3: gdn_conv[{i}][{j}] post-restore-then-marker \
                                 differs at n_keep={n_keep}"
                            );
                        }
                    }
                }
            }

            // kv_n_pos must equal M + n_keep + 1 on both (after marker
            // single_token).
            let expected_kv = (M as usize) + (n_keep as usize) + 1;
            for (i, &a_pos) in sess_a.kv_n_pos.iter().enumerate() {
                assert_eq!(
                    a_pos, expected_kv,
                    "G3: sess_A kv_n_pos[{i}]={a_pos} != expected {expected_kv}"
                );
                assert_eq!(
                    sess_b.kv_n_pos[i], expected_kv,
                    "G3: sess_B kv_n_pos[{i}] != expected {expected_kv}"
                );
            }
        }
    }

    /// H5.3a gate G1 (FULL): cosine ≥ 0.9999 between
    /// `packed_verify_with_logits` row-n logits and N successive
    /// `single_token` logits. The strongest correctness signal at
    /// the LOGITS layer (not just argmax-coarsened).
    ///
    /// G1 lite (in dflash_packed_verify_argmax_matches_n_single_tokens)
    /// only checks argmax tokens — it would pass if all rows shifted
    /// by a constant. G1 full catches:
    ///   * subtle accumulation differences in the lm_head mat-vec
    ///   * any per-vocab-row bias from a wrong scatter offset
    ///   * cosine that's strong but not perfect (e.g. F16 KV
    ///     accumulation paths in attn-v4) — G1 full's threshold of
    ///     0.9999 is the H5.3 plan gate per docs/H5-DFLASH.md §3 H5.3.
    ///
    /// 0.8B-F32, M=2 prime + N=4 verify. Uses
    /// MetalDFlashDebugScratch (allocates the [N, V] buffer; debug-
    /// only path).
    #[test]
    fn dflash_packed_verify_with_logits_cosine_match() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        const M: u32 = 2;
        const N: u32 = 4;
        let prime_tokens: [i32; M as usize] = [9419, 1];
        let verify_tokens: [i32; N as usize] = [1234, 7, 999, 42];
        let target_layer_ids: Vec<u32> = vec![5, 15];
        let k_target_layers = target_layer_ids.len() as u32;
        let v = m.arch.vocab_size as usize;

        // -- Reference: N successive single_token on a primed session.
        let mut sess_b = MetalSession::fresh(&ctx, &mm, 64).expect("sess B");
        for (i, &tok) in prime_tokens.iter().enumerate() {
            mf.single_token(tok, i as u32, &mut sess_b)
                .expect("prime B");
        }
        let mut reference: Vec<Vec<f32>> = Vec::with_capacity(N as usize);
        for (i, &tok) in verify_tokens.iter().enumerate() {
            let logits = mf
                .single_token(tok, M + i as u32, &mut sess_b)
                .expect("single token");
            reference.push(logits);
        }

        // -- Packed with logits dump.
        let mut sess_a = MetalSession::fresh(&ctx, &mm, 64).expect("sess A");
        for (i, &tok) in prime_tokens.iter().enumerate() {
            mf.single_token(tok, i as u32, &mut sess_a)
                .expect("prime A");
        }
        let mut dbg_scratch =
            MetalDFlashDebugScratch::fresh(&ctx, &mm, N, k_target_layers).expect("dbg scratch");
        let _ = encode_packed_verify_with_logits_inner(
            &mf,
            &target_layer_ids,
            &verify_tokens,
            M,
            &mut dbg_scratch,
            &mut sess_a,
        )
        .expect("packed verify w logits");

        // -- Per-row cosine vs reference. F32 path; we expect bit-exact
        //    actually, but the H5.3 plan threshold is 0.9999 because
        //    quantized paths will round-trip differently. Test both
        //    bounds.
        let dump_n_elems = dbg_scratch.debug_logits.n_elements() as usize;
        let mut packed_dump = vec![0.0f32; dump_n_elems];
        unsafe {
            let src = dbg_scratch.debug_logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, packed_dump.as_mut_ptr(), dump_n_elems);
        }

        for n in 0..N as usize {
            let packed_row = &packed_dump[n * v..(n + 1) * v];
            let ref_row = &reference[n];
            // Cosine.
            let mut dot = 0.0f64;
            let mut np = 0.0f64;
            let mut nr = 0.0f64;
            for i in 0..v {
                let p = packed_row[i] as f64;
                let r = ref_row[i] as f64;
                dot += p * r;
                np += p * p;
                nr += r * r;
            }
            let cos = dot / (np.sqrt() * nr.sqrt() + 1e-30);
            // Max abs diff.
            let mut max_abs = 0.0f32;
            for i in 0..v {
                let d = (packed_row[i] - ref_row[i]).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            // Bitwise equality count (sanity — F32 path should be
            // mostly bit-exact but some atomic ordering can diverge).
            let mut bit_eq = 0usize;
            for i in 0..v {
                if packed_row[i].to_bits() == ref_row[i].to_bits() {
                    bit_eq += 1;
                }
            }
            eprintln!(
                "[g1-full n={n}] cos={cos:.10} max|Δ|={max_abs:.3e} \
                 bit_eq={}/{} ({:.2}%)",
                bit_eq,
                v,
                100.0 * (bit_eq as f64) / (v as f64)
            );
            assert!(cos >= 0.9999, "G1 full: row {n} cosine {cos} < 0.9999");
        }
    }

    /// H5.3a gate G4: hidden capture LAYOUT.
    ///
    /// Codex flagged this gap: G1 / G2 / G3 all check argmax tokens
    /// or final session state, but `hidden_capture[k, n, :]` could
    /// have wrong dim-order (e.g., stored as [N, K, H] instead of
    /// [K, N, H]) and the rest of the test suite would still pass.
    /// The dim-order bug only surfaces downstream when the drafter
    /// reads target_ctx and produces garbage logits.
    ///
    /// Setup: prime fresh session through M tokens. Run packed_verify
    /// on session_A through N tokens; collect scratch.hidden_capture.
    /// Separately, run `single_token_with_multi_hidden` N times on
    /// session_B (primed identically), capturing per-token hiddens
    /// into a `[K, H]` buffer per call. Stack into a `[N, K, H]`
    /// reference. Compare against scratch.hidden_capture (which is
    /// layout `[K, N, H]`) under the documented permutation.
    ///
    /// Bitwise F32 match required. Catches:
    ///   * (k, n) → linear-index transpose bugs in slot_view
    ///   * scatter dst offset miscomputation in packed_verify
    ///   * the wrong target_layer being captured at index k
    ///
    /// 0.8B-F32, M=2 prime + N=4 verify, K=2 layers. ≤ 4 s.
    #[test]
    // Hidden-capture layout test: `n` is a multi-purpose row index used
    // for `reference[n][...]`, `(n * K + k) * H + i` stride math, and
    // failure-message position. Iterator rewrite would lose all three.
    #[allow(clippy::needless_range_loop)]
    fn dflash_packed_verify_hidden_capture_layout() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        const M: u32 = 2;
        const N: u32 = 4;
        let prime_tokens: [i32; M as usize] = [9419, 1];
        let verify_tokens: [i32; N as usize] = [1234, 7, 999, 42];

        // Pick target_layer_ids such that they MUST be captured
        // distinctly — different blocks (5, 15) on 0.8B's 24-layer
        // schedule. If layout is K↔N transposed, the two layers'
        // hiddens get confused at different (n, k) pairs.
        let target_layer_ids: Vec<u32> = vec![5, 15];
        let k_target_layers = target_layer_ids.len() as u32;
        let h = m.arch.hidden_size as usize;

        // -- session_A: packed_verify, capture into scratch.hidden_capture.
        let mut sess_a = MetalSession::fresh(&ctx, &mm, 64).expect("sess A");
        for (i, &tok) in prime_tokens.iter().enumerate() {
            mf.single_token(tok, i as u32, &mut sess_a)
                .expect("prime A");
        }
        let mut scratch =
            MetalDFlashVerifyScratch::fresh(&ctx, &mm, N, k_target_layers).expect("scratch");
        let _ = encode_packed_verify_inner(
            &mf,
            &target_layer_ids,
            &verify_tokens,
            M,
            &mut scratch,
            &mut sess_a,
        )
        .expect("packed verify");

        // -- session_B: prime identically, then for each n in 0..N call
        //    single_token_with_multi_hidden. The hidden_dst is shape
        //    [K, H], laid out as `[k * h .. (k+1) * h]` per layer
        //    (matches MetalForward::single_token_with_multi_hidden).
        let mut sess_b = MetalSession::fresh(&ctx, &mm, 64).expect("sess B");
        for (i, &tok) in prime_tokens.iter().enumerate() {
            mf.single_token(tok, i as u32, &mut sess_b)
                .expect("prime B");
        }
        let single_hidden_buf =
            MetalTensor::zeros_f32(&ctx, vec![k_target_layers as u64 * h as u64])
                .expect("hidden dst");
        // [N][K * H] — flat reference dump per token.
        let mut reference: Vec<Vec<f32>> = Vec::with_capacity(N as usize);
        for (i, &tok) in verify_tokens.iter().enumerate() {
            let _ = mf
                .single_token_with_multi_hidden(
                    tok,
                    M + i as u32,
                    &mut sess_b,
                    &target_layer_ids,
                    &single_hidden_buf,
                )
                .expect("single token w multi hidden");
            let n_elems = k_target_layers as usize * h;
            let mut row = vec![0.0f32; n_elems];
            unsafe {
                let src = single_hidden_buf.buffer.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, row.as_mut_ptr(), n_elems);
            }
            reference.push(row);
        }

        // -- Compare scratch.hidden_capture (layout [K, N, H]) against
        //    reference (layout [N, K * H]) under the documented
        //    permutation. For each (k, n): scratch[k*N*H + n*H + i]
        //    == reference[n][k*H + i].
        let scratch_buf_n_elems = scratch.hidden_capture.n_elements() as usize;
        let mut scratch_dump = vec![0.0f32; scratch_buf_n_elems];
        unsafe {
            let src = scratch.hidden_capture.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, scratch_dump.as_mut_ptr(), scratch_buf_n_elems);
        }

        // v0.71: layout is [N, K, H], so scratch index = (n * K + k) * H + i.
        for k in 0..k_target_layers as usize {
            for n in 0..N as usize {
                for i in 0..h {
                    let scratch_idx = (n * (k_target_layers as usize) + k) * h + i;
                    let ref_idx_in_row = k * h + i;
                    let s = scratch_dump[scratch_idx];
                    let r = reference[n][ref_idx_in_row];
                    if s.to_bits() != r.to_bits() {
                        panic!(
                            "G4: hidden_capture[n={n}, k={k}, i={i}] differs: \
                             scratch={s} (0x{:08x}) reference={r} (0x{:08x})",
                            s.to_bits(),
                            r.to_bits()
                        );
                    }
                }
            }
        }
        eprintln!(
            "[hidden-capture-layout] M={M} N={N} K={k_target_layers} \
             H={h}: bitwise match across all (k, n, i)"
        );

        // ALSO: confirm the two captured layers are NOT trivially
        // identical. If they were, a K↔N layout bug would silently
        // pass. We require the L2 distance between layer 5 and layer
        // 15 captures at n=0 to be substantial.
        let mut l2 = 0.0f64;
        for i in 0..h {
            let a = reference[0][i] as f64; // n=0, k=0 (layer 5)
            let b = reference[0][h + i] as f64; // n=0, k=1 (layer 15)
            l2 += (a - b).powi(2);
        }
        l2 = l2.sqrt();
        eprintln!(
            "[hidden-capture-layout] ||layer5_at_n0 - layer15_at_n0||_2 = {l2:.4} \
             (must be substantially nonzero or the test is degenerate)"
        );
        assert!(
            l2 > 0.1,
            "test is degenerate: the two captured layers are nearly identical, \
             a K↔N layout bug would pass silently. Pick more-different layers."
        );
    }

    /// H5.3a guard tests for restore primitive (codex failure-mode
    /// mitigation): n_keep=0 must fail loudly; n_keep > N must fail;
    /// stale kv_n_pos must fail.
    #[test]
    fn dflash_restore_after_partial_accept_guard_wall() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        const N: u32 = 4;
        let mut sess = MetalSession::fresh(&ctx, &mm, 64).expect("sess");
        let target_layer_ids: Vec<u32> = vec![5, 15];
        let mut scratch = MetalDFlashVerifyScratch::fresh(&ctx, &mm, N, 2).expect("scratch");

        // Run packed_verify so session is in the post-packed-verify state.
        let _ = encode_packed_verify_inner(
            &mf,
            &target_layer_ids,
            &[1i32, 2, 3, 4],
            0,
            &mut scratch,
            &mut sess,
        )
        .expect("packed verify");

        // (a) n_keep = 0
        let err = encode_restore_after_partial_accept_inner(&mf, &scratch, 0, 0, &mut sess);
        match err {
            Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                assert!(detail.contains("n_keep=0"), "wrong error: {detail}");
            }
            other => panic!("expected BadShape on n_keep=0, got {other:?}"),
        }

        // (b) n_keep > N
        let err = encode_restore_after_partial_accept_inner(&mf, &scratch, N + 1, 0, &mut sess);
        match err {
            Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                assert!(
                    detail.contains(&format!("n_keep={}", N + 1)),
                    "wrong error: {detail}"
                );
            }
            other => panic!("expected BadShape on n_keep>N, got {other:?}"),
        }

        // (c) wrong start_position (kv_n_pos contract violation).
        // 0.8B has no attn layers so kv_n_pos.len() == 0 — the loop
        // is trivially satisfied. Note in stderr; the contract is
        // exercised on 27B (different test).
        if !sess.kv_n_pos.is_empty() {
            let err = encode_restore_after_partial_accept_inner(&mf, &scratch, 2, 99, &mut sess);
            match err {
                Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                    assert!(detail.contains("kv_n_pos"), "wrong error: {detail}");
                }
                other => {
                    panic!("expected BadShape on kv_n_pos mismatch, got {other:?}")
                }
            }
        } else {
            eprintln!(
                "[restore-guard] 0.8B has no attn layers; \
                 kv_n_pos contract is exercised at 27B (separate test)"
            );
        }
    }

    /// Verify `MetalDFlashDebugScratch` builds correctly and `logits_slot`
    /// returns properly-aligned views into the [N, V] buffer.
    #[test]
    fn dflash_debug_scratch_logits_slots() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[dflash-debug-scratch] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        let n: u32 = 4;
        let k: u32 = 2;
        let dbg = MetalDFlashDebugScratch::fresh(&ctx, &mm, n, k).expect("debug scratch alloc");

        let v = m.arch.vocab_size as u64;
        let f32_size = std::mem::size_of::<f32>() as u64;
        assert_eq!(dbg.debug_logits.shape, vec![n as u64, v]);
        for nn in 0..n {
            let slot = dbg.logits_slot(nn);
            assert_eq!(slot.shape, vec![v]);
            assert_eq!(slot.offset, (nn as u64) * v * f32_size);
        }
        // Verify the wrapped verify scratch is independently usable.
        assert_eq!(dbg.verify.n, n);
        assert_eq!(dbg.verify.k_target_layers, k);
    }

    /// **v0.75.1 correctness gate** (fast, lib-loop variant on
    /// Qwen3.5-0.8B-F32). Oracle: a sequential `single_token_with_
    /// multi_hidden` loop. Experimental: one `prefill_tokens_with_
    /// multi_hidden` call.
    ///
    /// Cosine equivalence ≥ 0.999 required on:
    ///   * final logits (last prompt token's vocab vector)
    ///   * accumulated per-token multi-hidden capture
    ///   * GDN state + conv tensors per layer
    ///
    /// 0.8B-F32 has 36 GDN layers and ZERO attn layers, so this gates
    /// the GDN recurrence, FFN mat-mat (mat-mat-dispatch fall-through
    /// for F32 weights uses per-token mat-vec, but layer-major batched
    /// norms still cover their bookkeeping), and tail. Attn-path
    /// correctness is gated by a separate 27B integration test in
    /// `tests/dflash_correctness.rs` (slow; not in lib loop).
    ///
    /// Edge cases tested via subroutine: T<P, T==P, T==P+r, T==2P.
    #[test]
    fn prefill_tokens_matches_single_token_loop_0_8b() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[prefill-tokens-vs-single-token] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        let arch = &mm.arch;
        let h = arch.hidden_size as usize;

        // K=4 capture layers spanning the network. 0.8B-F32 has 24
        // layers so we hit GDN at varied depths (all-GDN model).
        let capture_layers: Vec<u32> = vec![0, 7, 14, 21];
        let k = capture_layers.len();

        // Run each scenario as a closure so the same comparator covers
        // T<P, T==P, T==P+r, T==2P, T=P+1 (chunk_p==1), and the
        // start_position > 0 case (extending an already-advanced session).
        //
        // `prefix_len` primes both sessions with `prefix_len` tokens via
        // sequential `single_token` first; prefill then runs from
        // `start_position = prefix_len` over `total_n` more tokens. The
        // total number of tokens forwarded by both paths is
        // `prefix_len + total_n`. The final logits / hidden capture
        // comparison is over the LAST `total_n` tokens.
        let run_scenario = |label: &str, total_n: usize, p: usize, prefix_len: usize| {
            assert!(total_n >= 1 && p >= 1, "{label}: need n,p ≥ 1");

            // Synthesize a deterministic prefix + suffix sequence.
            let n_total_with_prefix = prefix_len + total_n;
            let all_tokens: Vec<i32> = (0..n_total_with_prefix)
                .map(|i| ((i * 13 + 7) % (arch.vocab_size as usize - 1)) as i32 + 1)
                .collect();
            let prefix_tokens = &all_tokens[..prefix_len];
            let token_ids = &all_tokens[prefix_len..];

            // ---- Oracle path: sequential single_token_with_multi_hidden ----
            // (prefix uses single_token to advance state, suffix uses
            // single_token_with_multi_hidden so we capture the same
            // hidden states the prefill captures.)
            let cap = n_total_with_prefix + 4;
            let mut sess_a = MetalSession::fresh(&ctx, &mm, cap).expect("sess A");
            for (i, &tid) in prefix_tokens.iter().enumerate() {
                mf.single_token(tid, i as u32, &mut sess_a)
                    .expect("oracle prefix advance");
            }
            let h_dst_a = MetalTensor::zeros_f32(&ctx, vec![(k * h) as u64]).expect("h_dst_a");
            let mut accum_a = vec![0.0f32; total_n * k * h];
            let mut last_a = Vec::new();
            for (i, &tid) in token_ids.iter().enumerate() {
                last_a = mf
                    .single_token_with_multi_hidden(
                        tid,
                        (prefix_len + i) as u32,
                        &mut sess_a,
                        &capture_layers,
                        &h_dst_a,
                    )
                    .expect("oracle forward");
                unsafe {
                    let src = h_dst_a.buffer.contents().as_ptr() as *const f32;
                    std::ptr::copy_nonoverlapping(
                        src,
                        accum_a[i * k * h..(i + 1) * k * h].as_mut_ptr(),
                        k * h,
                    );
                }
            }

            // ---- Experimental path: prefill_tokens_with_multi_hidden ----
            // Same prefix advance via single_token, then one prefill call
            // starting at start_position = prefix_len.
            let mut sess_b = MetalSession::fresh(&ctx, &mm, cap).expect("sess B");
            for (i, &tid) in prefix_tokens.iter().enumerate() {
                mf.single_token(tid, i as u32, &mut sess_b)
                    .expect("experimental prefix advance");
            }
            let mut layer_scratch =
                MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, p as u32).expect("layer scratch");
            let h_dst_b =
                MetalTensor::zeros_f32(&ctx, vec![(total_n * k * h) as u64]).expect("h_dst_b");
            let last_b = prefill_tokens_with_multi_hidden(
                &mf,
                token_ids,
                prefix_len as u32,
                &mut sess_b,
                &mut layer_scratch,
                &capture_layers,
                Some(&h_dst_b),
            )
            .expect("prefill");

            // ---- Compare final logits (cos ≥ 0.999). ----
            assert_eq!(last_a.len(), last_b.len(), "{label}: logits len mismatch");
            let cos_logits = cosine_f32(&last_a, &last_b);
            eprintln!(
                "[prefill-vs-single] {label}: T={total_n} P={p} prefix={prefix_len} chunks={} cos(logits)={cos_logits:.6}",
                total_n.div_ceil(p)
            );
            assert!(
                cos_logits >= 0.999,
                "{label}: logits cos={cos_logits} < 0.999"
            );

            // ---- Compare accumulated multi-hidden (cos per token slot). ----
            let mut accum_b = vec![0.0f32; total_n * k * h];
            unsafe {
                let src = h_dst_b.buffer.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, accum_b.as_mut_ptr(), total_n * k * h);
            }
            // Compare per-token-per-capture-layer (most diagnostic) AND
            // overall (compact summary).
            let mut min_cos = f64::INFINITY;
            let mut worst_pos = (0usize, 0usize);
            for t in 0..total_n {
                for k_idx in 0..k {
                    let off = (t * k + k_idx) * h;
                    let a_slice = &accum_a[off..off + h];
                    let b_slice = &accum_b[off..off + h];
                    let c = cosine_f32(a_slice, b_slice);
                    if c < min_cos {
                        min_cos = c;
                        worst_pos = (t, k_idx);
                    }
                }
            }
            eprintln!(
                "[prefill-vs-single] {label}: hidden cos_min={min_cos:.6} \
                 (at token={}, capture_layer={})",
                worst_pos.0, worst_pos.1
            );
            assert!(
                min_cos >= 0.999,
                "{label}: hidden capture cos_min={min_cos} < 0.999 \
                 (worst at token={}, capture_layer={})",
                worst_pos.0,
                worst_pos.1
            );

            // ---- Compare GDN state + conv per layer (cos ≥ 0.999). ----
            assert_eq!(
                sess_a.gdn_state.len(),
                sess_b.gdn_state.len(),
                "GDN state vec length mismatch"
            );
            for gi in 0..sess_a.gdn_state.len() {
                let a_state = read_tensor_f32(&sess_a.gdn_state[gi]);
                let b_state = read_tensor_f32(&sess_b.gdn_state[gi]);
                let cs = cosine_f32(&a_state, &b_state);
                let a_conv = read_tensor_f32(&sess_a.gdn_conv[gi]);
                let b_conv = read_tensor_f32(&sess_b.gdn_conv[gi]);
                let cc = cosine_f32(&a_conv, &b_conv);
                assert!(cs >= 0.999, "{label}: GDN[{gi}] state cos={cs} < 0.999");
                assert!(cc >= 0.999, "{label}: GDN[{gi}] conv cos={cc} < 0.999");
            }

            // ---- KV state (only for attn layers; 0.8B has none). ----
            assert_eq!(sess_a.kv_n_pos.len(), sess_b.kv_n_pos.len());
            for ai in 0..sess_a.kv_n_pos.len() {
                assert_eq!(
                    sess_a.kv_n_pos[ai], sess_b.kv_n_pos[ai],
                    "{label}: kv_n_pos[{ai}] mismatch ({} vs {})",
                    sess_a.kv_n_pos[ai], sess_b.kv_n_pos[ai]
                );
            }
        };

        // Edge cases. Format: (label, total_n, p, prefix_len).
        run_scenario("T<P (T=3,P=8)", 3, 8, 0);
        run_scenario("T==P (T=8,P=8)", 8, 8, 0);
        run_scenario("T==P+r (T=11,P=8)", 11, 8, 0);
        run_scenario("T==2P (T=16,P=8)", 16, 8, 0);
        // chunk_p == 1 final chunk (codex pre-commit ask): T=9, P=8.
        run_scenario("T=P+1 chunk_p=1 (T=9,P=8)", 9, 8, 0);
        // start_position > 0 (codex pre-commit ask): prime with prefix=4
        // single_tokens, then run prefill at start_position=4 over T=10
        // suffix tokens. Exercises the "extend an already-advanced
        // session" public API guarantee that prior tests didn't hit.
        run_scenario("start_position>0 (prefix=4, T=10,P=8)", 10, 8, 4);

        // ---- T=0 must error. ----
        let mut sess_e = MetalSession::fresh(&ctx, &mm, 8).expect("sess E");
        let mut layer_scratch_e =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, 4).expect("scratch E");
        let h_dst_e = MetalTensor::zeros_f32(&ctx, vec![1]).expect("h_dst_e");
        let err = prefill_tokens_with_multi_hidden(
            &mf,
            &[],
            0,
            &mut sess_e,
            &mut layer_scratch_e,
            &capture_layers,
            Some(&h_dst_e),
        );
        match err {
            Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                assert!(
                    detail.contains("empty"),
                    "expected 'empty' in T=0 error: {detail}"
                );
            }
            other => panic!("expected BadShape on T=0, got {other:?}"),
        }
    }

    fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..a.len() {
            dot += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-30)
    }

    fn read_tensor_f32(t: &MetalTensor) -> Vec<f32> {
        let n = t.n_elements() as usize;
        let mut out = vec![0.0f32; n];
        unsafe {
            let src = t.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    }
}
