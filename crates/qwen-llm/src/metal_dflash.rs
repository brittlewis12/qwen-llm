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
    encode_add_inplace_f32, encode_argmax_f32, encode_get_rows_f32, encode_rms_norm_batched_f32,
    encode_rms_norm_mul_f32, encode_rope_neox_f32, BlitEncoder, KernelEncoder, MetalContext,
    MetalError, MetalTensor,
};
use crate::metal_forward::{
    encode_mat_vec_dispatch, encode_scatter_offset_f32, weight_dtype_kept_native, MetalBlock,
    MetalForward, MetalSession, RMS_EPS,
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
        unsafe { cmd.waitUntilCompleted() };
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
            hidden_capture: MetalTensor::zeros_f32(ctx, vec![k, n, h])?,
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
    pub fn gdn_ckpt_slot(&self, layer: u32, n: u32) -> MetalTensor {
        debug_assert!(layer < self.n_gdn_layers);
        debug_assert!(n < self.n);
        let elem_offset = (layer as u64 * self.n as u64 + n as u64) * self.ssm_state_elems;
        self.gdn_ckpt
            .view_subrange(elem_offset, vec![self.ssm_state_elems])
    }

    /// Zero-copy view of conv checkpoint slot `(layer, n)`. Returned shape:
    /// `[conv_state_elems]`.
    pub fn conv_ckpt_slot(&self, layer: u32, n: u32) -> MetalTensor {
        debug_assert!(layer < self.n_gdn_layers);
        debug_assert!(n < self.n);
        let elem_offset = (layer as u64 * self.n as u64 + n as u64) * self.conv_state_elems;
        self.conv_ckpt
            .view_subrange(elem_offset, vec![self.conv_state_elems])
    }

    /// Zero-copy view of hidden_capture slot `(k, n)`. Returned shape:
    /// `[hidden_size]`. Used as a scatter destination after the K-indexed
    /// target layer's residual for token n.
    pub fn hidden_capture_slot(&self, k: u32, n: u32) -> MetalTensor {
        debug_assert!(k < self.k_target_layers);
        debug_assert!(n < self.n);
        let elem_offset = (k as u64 * self.n as u64 + n as u64) * self.hidden_size;
        self.hidden_capture
            .view_subrange(elem_offset, vec![self.hidden_size])
    }

    /// Zero-copy view of `packed_ids_buf[n..n+1]`. Used as the
    /// `get_rows` token-id input for block n; required to avoid the
    /// shared-CPU-mutable `MetalSession::ids_buf` race that would
    /// silently corrupt N successive `get_rows` calls in one command
    /// buffer (codex Q7 — the bug we'd ship without this).
    pub fn token_slot(&self, n: u32) -> MetalTensor {
        debug_assert!(n < self.n);
        self.packed_ids_buf.view_subrange(n as u64, vec![1])
    }

    /// Zero-copy view of `verify_argmax[n..n+1]`. Used as the destination
    /// for `encode_argmax_f32` over block n's logits.
    pub fn argmax_slot(&self, n: u32) -> MetalTensor {
        debug_assert!(n < self.n);
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
        debug_assert!(n < self.verify.n);
        let v = self.debug_logits.shape[1];
        let elem_offset = n as u64 * v;
        self.debug_logits.view_subrange(elem_offset, vec![v])
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
    pub fn packed_verify(
        &self,
        tokens: &[i32],
        start_position: u32,
        scratch: &mut MetalDFlashVerifyScratch,
        target_session: &mut MetalSession,
    ) -> Result<Vec<i32>, DFlashError> {
        // Thin wrapper around the inner free function so that tests can
        // exercise packed_verify without constructing a real DFlash
        // drafter (DFlashDecoder requires real drafter weights). The
        // inner function takes everything explicitly as parameters and
        // is `pub(crate)` so it's not part of the public API.
        encode_packed_verify_inner(
            self.base,
            &self.head.target_layer_ids,
            tokens,
            start_position,
            scratch,
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
    let last_pos = start_position as usize + n;
    if last_pos > target_session.kv_capacity {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify",
            detail: format!(
                "start_position={} + N={n} = {last_pos} > kv_capacity={}",
                start_position, target_session.kv_capacity
            ),
        }));
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
    // CPU-mutable scalar buffer with another block. Buffer is
    // StorageModeShared so the host write is visible to the GPU once
    // we open the command encoder (the runtime synchronises on
    // `commandBuffer()` boundaries for shared-mode buffers).
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
                    // hidden_capture is stored as [K, N, H]; the
                    // slot view returns a [H]-shaped tensor at
                    // the right offset.
                    let dst_slot = scratch.hidden_capture_slot(k_idx as u32, n_idx as u32);
                    // We need scatter_offset_f32(src=x, dst=parent,
                    // dst_off=byte_off/4) — but the slot view IS the
                    // parent shifted by byte_off. We can't pass the
                    // slot to scatter_offset directly because that
                    // helper expects a parent tensor + element
                    // offset. Convert: dst_off = slot.offset / 4
                    // (F32 elem size), parent = scratch.hidden_capture.
                    let elem_off =
                        (k_idx as u64 * scratch.n as u64 + n_idx as u64) * scratch.hidden_size;
                    // Sanity: confirm the slot view we'd compute
                    // matches the elem_off arithmetic. Cheap; not
                    // in the hot path beyond once-per-target-layer.
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
    unsafe { cmd_buf.waitUntilCompleted() };

    // Read back verify_argmax (only N i32 values; trivial).
    let mut out = vec![0i32; n];
    unsafe {
        let src = scratch.verify_argmax.buffer.contents().as_ptr() as *const i32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }
    Ok(out)
}

impl<'a> DFlashDecoder<'a> {
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
                // Sub-view of just the populated rows (k_ctx_buf is
                // sized for ctx_capacity, but only first ctx_len * kv_dim
                // elements are valid this call).
                let view = self
                    .session
                    .k_ctx_buf
                    .view_subrange(0, vec![(ctx_len * kv_dim) as u64]);
                encode_rms_norm_batched_f32(
                    ctx_metal,
                    &enc,
                    &view,
                    &layer.k_norm,
                    &view,
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
        assert_eq!(
            scratch.hidden_capture.shape,
            vec![k as u64, n as u64, scratch.hidden_size]
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

        // -- hidden_capture_slot offsets --
        for kk in 0..k {
            for nn in 0..n {
                let slot = scratch.hidden_capture_slot(kk, nn);
                let expected_elem_off = (kk as u64 * n as u64 + nn as u64) * scratch.hidden_size;
                assert_eq!(slot.shape, vec![scratch.hidden_size]);
                assert_eq!(slot.offset, expected_elem_off * f32_size);
            }
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
        for (i, (s_state, p_state)) in single_session
            .gdn_state
            .iter()
            .zip(packed_session.gdn_state.iter())
            .enumerate()
        {
            unsafe {
                let s = s_state.buffer.contents().as_ptr() as *const f32;
                let p = p_state.buffer.contents().as_ptr() as *const f32;
                let n_elems = s_state.n_elements() as usize;
                let mut max_abs = 0.0f32;
                for j in 0..n_elems {
                    let d = (*s.add(j) - *p.add(j)).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
                assert!(
                    max_abs < 1e-5,
                    "G2: gdn_state[{i}] post-packed differs from post-single: max|Δ|={max_abs}"
                );
            }
        }
        for (i, (s_conv, p_conv)) in single_session
            .gdn_conv
            .iter()
            .zip(packed_session.gdn_conv.iter())
            .enumerate()
        {
            unsafe {
                let s = s_conv.buffer.contents().as_ptr() as *const f32;
                let p = p_conv.buffer.contents().as_ptr() as *const f32;
                let n_elems = s_conv.n_elements() as usize;
                let mut max_abs = 0.0f32;
                for j in 0..n_elems {
                    let d = (*s.add(j) - *p.add(j)).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
                assert!(
                    max_abs < 1e-5,
                    "G2: gdn_conv[{i}] post-packed differs from post-single: max|Δ|={max_abs}"
                );
            }
        }
        assert_eq!(
            single_session.kv_n_pos, packed_session.kv_n_pos,
            "G2: kv_n_pos diverged"
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
}
