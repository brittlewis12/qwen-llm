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

use crate::codec::{dequant_to_f32, dequant_to_f32_in_place};
use crate::gguf::GgufFile;
use crate::loader::{DFlashHead, DFlashLayer};
use crate::metal::{
    BlitEncoder, KernelEncoder, MetalContext, MetalError, MetalTensor, MetalTimestampSampleBuffer,
    encode_add_inplace_f32, encode_argmax_f32, encode_argmax_top2_f32, encode_axpy_rowwise_f32,
    encode_copy_offset_f32, encode_dflash_attn_f32, encode_dflash_attn_full_gqa_split4_f32,
    encode_dflash_attn_online_two_range_scan_f32, encode_dflash_attn_two_range_f32,
    encode_dflash2_conv_f32, encode_ffn_fused_swiglu_q4_k_mma8_f32, encode_fill_f32,
    encode_gdn_decay_chain_batched_f32, encode_gdn_decay_chain_f32,
    encode_gdn_prep_packed_ckpt_f32, encode_gdn_prep_packed_f32,
    encode_gdn_step_decay_packed_ckpt_f32, encode_gdn_step_decay_packed_f32, encode_get_rows_f32,
    encode_l2_norm_batched_f32, encode_l2_norm_pair_batched_f32, encode_mat_mat_f32_router_e8p32,
    encode_moe_down_iq4_xs_f32, encode_moe_down_q5_K_f32,
    encode_moe_down_weighted_sum_q5_K_f32_packed_slots, encode_moe_mat_vec_f32,
    encode_moe_swiglu_q4_K_f32_packed_slots, encode_moe_weighted_sum_f32,
    encode_qk_rms_norm_rope_f32_packed_consecutive, encode_rms_norm_batched_f32,
    encode_rms_norm_batched_src_strided_f32, encode_rms_norm_mul_f32, encode_rmsnorm_gated_f32,
    encode_rope_neox_f32, encode_rope_neox_f32_packed_consecutive,
    encode_rope_neox_pair_adaptive_f32_packed_consecutive, encode_scatter_offset_f32_to_f16_kv,
    encode_scatter_offset_f32_to_f16_kv_vt, encode_sigmoid_f32, encode_silu_mul_f32,
    encode_split_qkv_fused_f32, encode_topk_logits_softmax_dot_sigmoid_packed_f32,
    encode_topk16_f32, kernel_trace_begin, kernel_trace_snapshot, kernel_trace_take_delta,
};
#[cfg(feature = "dflash-k0s-diagnostics")]
use crate::metal::{
    dispatch_census_begin, dispatch_census_is_active, dispatch_census_tag_scope,
    dispatch_census_take,
};
use crate::metal_forward::{
    ATTN_V4_MAX_NWG, LmHeadTail, LmHeadTailEvidence, MetalBlock, MetalForward, MetalModel,
    MetalMoeFfn, MetalSession, MfError, RMS_EPS, checked_u64_add, checked_u64_double,
    checked_u64_mul, checked_u64_mul3, checked_u64_mul4, encode_mat_mat_dispatch,
    encode_mat_vec_dispatch, encode_scatter_offset_f32, weight_dtype_kept_native,
};
use crate::sampling::{Sampler, SamplingError, SparseProposal, WeightedCandidate};
use crate::tensor::{GgmlType, TensorDesc};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLResource,
    MTLStorageMode,
};
#[cfg(feature = "dflash-k0s-diagnostics")]
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct AttentionCaptureProvenance {
    pub position: usize,
    pub causal_length: usize,
    pub block: usize,
    pub kv_slot: usize,
    pub path: &'static str,
    pub online_matrix: bool,
    pub query_tiled: bool,
    pub query_rows: Option<usize>,
    pub packed_rows: Option<usize>,
    pub packed_qt: Option<usize>,
    pub nwg: Option<usize>,
    pub tile_c: Option<usize>,
    pub group_tile: Option<usize>,
    pub matrix_causal_skip: bool,
}

/// Maximum number of cross-context columns an all-SWA drafter can observe.
/// The served Qwen3.8-27B-DFlash2 head is all-SWA with `sliding_window = 2048`
/// (5/5 layers), and the F1 oracle (2026-08-20) proved bit-identical drafts
/// between a full-context session and one seeded with only the last 2,048
/// columns. Callers capture and store only this suffix.
pub const DFLASH_CAPTURE_WINDOW: usize = 2048;

/// Capture-window limit for a loaded head: the SWA window when every drafter
/// layer is SWA, else `usize::MAX` (full-context capture — any full-attention
/// layer can observe the whole history, so windowing would change drafts).
pub fn dflash_capture_window_limit(head: &MetalDFlashHead) -> usize {
    if head.layers.iter().all(|l| l.is_swa) && head.config.swa_window > 0 {
        head.config.swa_window as usize
    } else {
        usize::MAX
    }
}

/// Absolute start and length of the drafter capture window for a prompt of
/// `prompt_len` positions under `window_limit`. The window is the trailing
/// suffix; short prompts capture in full and return `(0, prompt_len)`.
pub fn dflash_capture_window_span(prompt_len: usize, window_limit: usize) -> (usize, usize) {
    let window = prompt_len.min(window_limit);
    (prompt_len - window, window)
}

/// Window-aware completeness predicate: the capture covers exactly
/// `[wstart, prompt_len)` with `captured` columns. Supersedes the old
/// `start == 0 && captured == prompt_len` invariant.
pub fn dflash_capture_window_complete(
    prompt_len: usize,
    capture_start: usize,
    captured: usize,
    window_limit: usize,
) -> bool {
    let (wstart, wlen) = dflash_capture_window_span(prompt_len, window_limit);
    capture_start == wstart && captured == wlen
}

pub struct AttentionCapture {
    pub blocks: Vec<usize>,
    pub positions: Vec<usize>,
    pub q: Vec<MetalTensor>,
    pub o: Vec<MetalTensor>,
    pub provenance: Vec<AttentionCaptureProvenance>,
    seen: Vec<bool>,
    q_dim: usize,
}

impl AttentionCapture {
    pub fn new(
        ctx: &MetalContext,
        blocks: Vec<usize>,
        positions: Vec<usize>,
        q_dim: usize,
    ) -> Result<Self, MetalError> {
        let elements = positions
            .len()
            .checked_mul(q_dim)
            .ok_or_else(|| MetalError::BadShape {
                kernel: "attention_capture",
                detail: "capture tensor size overflow".into(),
            })?;
        let mut q = Vec::with_capacity(blocks.len());
        let mut o = Vec::with_capacity(blocks.len());
        for _ in &blocks {
            let q_tensor = MetalTensor::zeros_f32(ctx, vec![elements as u64])?;
            let o_tensor = MetalTensor::zeros_f32(ctx, vec![elements as u64])?;
            poison_capture_tensor(&q_tensor)?;
            poison_capture_tensor(&o_tensor)?;
            q.push(q_tensor);
            o.push(o_tensor);
        }
        Ok(Self {
            seen: vec![false; blocks.len() * positions.len()],
            blocks,
            positions,
            q,
            o,
            provenance: Vec::new(),
            q_dim,
        })
    }

    pub fn validate_complete(&self) -> Result<(), MetalError> {
        if let Some(index) = self.seen.iter().position(|seen| !seen) {
            return Err(MetalError::BadShape {
                kernel: "attention_capture",
                detail: format!("capture row {index} was not produced"),
            });
        }
        for (block_index, tensor) in self.q.iter().chain(&self.o).enumerate() {
            let values = capture_tensor_f32(tensor)?;
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return Err(MetalError::BadShape {
                    kernel: "attention_capture",
                    detail: format!("capture tensor {block_index} row data {index} is non-finite"),
                });
            }
            if !values.iter().any(|&value| value != 0.0) {
                return Err(MetalError::BadShape {
                    kernel: "attention_capture",
                    detail: format!("capture tensor {block_index} is all zero"),
                });
            }
        }
        Ok(())
    }
}

fn capture_tensor_parts(tensor: &MetalTensor) -> Result<(*mut f32, usize), MetalError> {
    let offset = usize::try_from(tensor.offset).map_err(|_| MetalError::BadShape {
        kernel: "attention_capture",
        detail: "capture tensor offset does not fit usize".into(),
    })?;
    let elements = usize::try_from(tensor.n_elements()).map_err(|_| MetalError::BadShape {
        kernel: "attention_capture",
        detail: "capture tensor length does not fit usize".into(),
    })?;
    let byte_length = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: "attention_capture",
            detail: "capture tensor byte length overflow".into(),
        })?;
    let end = offset
        .checked_add(byte_length)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "attention_capture",
            detail: "capture tensor range overflow".into(),
        })?;
    if end > tensor.buffer.length() {
        return Err(MetalError::BadShape {
            kernel: "attention_capture",
            detail: "capture tensor exceeds backing buffer".into(),
        });
    }
    let base = unsafe { (tensor.buffer.contents().as_ptr() as *mut u8).add(offset) };
    Ok((base.cast::<f32>(), elements))
}

fn capture_tensor_f32(tensor: &MetalTensor) -> Result<&[f32], MetalError> {
    let (base, elements) = capture_tensor_parts(tensor)?;
    Ok(unsafe { std::slice::from_raw_parts(base.cast_const(), elements) })
}

fn poison_capture_tensor(tensor: &MetalTensor) -> Result<(), MetalError> {
    let (base, elements) = capture_tensor_parts(tensor)?;
    unsafe {
        std::ptr::write_bytes(base, 0xff, elements);
    }
    Ok(())
}

crate::env_flag!(default_on dense_packed_gdn_step_enabled, "QWEN_DENSE_GDN_STEP_PACKED");
crate::env_flag!(
    default_off mtp_attn_q2_shared_kv_enabled,
    "QWEN_MTP_ATTN_Q2_SHARED_KV"
);
// Generalizes the q2 shared-KV path to the whole verify chain
// (2 <= n <= 8): one chunked attention call instead of one per row.
// Equivalent by construction — the prefill v4 kernel masks with
// `k_pos <= base_pos + row`, so row i sees exactly `[0, start+i]`, the
// same set the per-token loop gives it; only FP summation order differs
// (E1 tier, like the shipped q2 path).
crate::env_flag!(
    default_off mtp_attn_qn_shared_kv_enabled,
    "QWEN_MTP_ATTN_QN_SHARED_KV"
);
crate::env_flag!(
    default_off mtp_attn_qn_matrix_enabled,
    "QWEN_MTP_ATTN_QN_MATRIX"
);
crate::env_flag!(
    default_on prefill_attn_gdn_scratch_overlay_enabled,
    "QWEN_PREFILL_ATTN_GDN_SCRATCH_OVERLAY"
);

pub fn ensure_prompt_lookup_n8_supported(model: &MetalModel) -> Result<(), String> {
    if model.arch != crate::model::QWEN3_27B {
        return Err(format!(
            "prompt lookup currently requires the dense 27B architecture; loaded {:?}",
            model.arch.kind
        ));
    }
    if model.lm_head.dtype != GgmlType::Q6_K {
        return Err(format!(
            "prompt lookup currently requires the validated Q4_K_M layout; lm_head is {:?}",
            model.lm_head.dtype
        ));
    }
    for (index, block) in model.blocks.iter().enumerate() {
        let (gate, up, down, moe) = match block {
            MetalBlock::Gdn(block) => (
                &block.ffn_gate,
                &block.ffn_up,
                &block.ffn_down,
                block.ffn_moe.as_ref(),
            ),
            MetalBlock::Attn(block) => (
                &block.ffn_gate,
                &block.ffn_up,
                &block.ffn_down,
                block.ffn_moe.as_ref(),
            ),
        };
        if moe.is_some()
            || gate.dtype != GgmlType::Q4_K
            || up.dtype != GgmlType::Q4_K
            || !matches!(down.dtype, GgmlType::Q4_K | GgmlType::Q6_K)
        {
            return Err(format!(
                "prompt lookup requires dense-27B Q4_K_M gate/up=Q4_K and down=Q4_K/Q6_K; block \
                 {index} has gate/up/down={:?}/{:?}/{:?}, moe={}",
                gate.dtype,
                up.dtype,
                down.dtype,
                moe.is_some()
            ));
        }
    }
    Ok(())
}

crate::env_flag!(default_on dflash_batched_proj_enabled, "QWEN_DFLASH_BATCHED_PROJ");

crate::env_flag!(default_off dflash_trace_phase3_split_enabled, "QWEN_DFLASH_TRACE_PHASE3_SPLIT");

crate::env_flag!(default_off dflash_attn_two_range_enabled, "QWEN_DFLASH_ATTN_TWO_RANGE");

// DFlash 2 ablation: take each position's top-1 candidate instead of the
// selector lattice walk (conv stays active). Isolates the selector's
// acceptance contribution. Bench-only; changes drafts, never correctness
// (verification still gates every token).
crate::env_flag!(default_off dflash2_selector_disabled, "QWEN_DFLASH2_NO_SELECTOR");

crate::env_flag!(
    default_on dflash_attn_online_two_range_enabled,
    "QWEN_DFLASH_ATTN_ONLINE_TWO_RANGE"
);

crate::env_flag!(default_on dflash_attn_swa_scan_enabled, "QWEN_DFLASH_ATTN_SWA_SCAN");

// SWA two-range split-K drafter attention; see
// `encode_dflash_attn_swa_split4_f32`.
crate::env_flag!(default_on dflash_attn_swa_split4_enabled, "QWEN_DFLASH_ATTN_SWA_SPLIT4");

fn dflash_swa_split4_eligible(
    layer_is_swa: bool,
    n: usize,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    ctx_len: usize,
    ctx_scan_start: usize,
    exact_visible_suffix: bool,
    swa_window: u32,
    selector_top_k: u32,
) -> bool {
    layer_is_swa
        && n == 8
        && n_q == 32
        && n_kv == 8
        && head_dim == 128
        && swa_window == 2048
        && selector_top_k == 16
        && ctx_scan_start <= ctx_len
        && ctx_len - ctx_scan_start == swa_window as usize
        && exact_visible_suffix
}

fn dflash_swa_exact_visible_suffix(
    pos_ctx: &[i32],
    ctx_len: usize,
    ctx_scan_start: usize,
    noise_start_pos: u32,
    swa_window: u32,
) -> bool {
    let Ok(window) = usize::try_from(swa_window) else {
        return false;
    };
    let Some(first_pos) = noise_start_pos.checked_sub(swa_window) else {
        return false;
    };
    if window == 0 || ctx_len > pos_ctx.len() || ctx_scan_start.checked_add(window) != Some(ctx_len)
    {
        return false;
    }
    pos_ctx[ctx_scan_start..ctx_len]
        .iter()
        .enumerate()
        .all(|(index, &position)| {
            u32::try_from(position).ok()
                == u32::try_from(index)
                    .ok()
                    .and_then(|index| first_pos.checked_add(index))
        })
}

// v0.77: encode the entire drafter forward into ONE command buffer with a
// single commit + waitUntilCompleted before the readback. The multi-buffer
// structure (up to 13 CPU/GPU round-trips per draft_block: phase 1, embed,
// 5× phase 2, 5× phase 3, phase 4) existed for per-phase GPU timing and a
// per-layer pos_k host write — the former is profiling-only, the latter is
// layer-invariant and hoisted. Profiled runs keep the multi-buffer path.
crate::env_flag!(default_on dflash_draft_single_cmd_enabled, "QWEN_DFLASH_DRAFT_SINGLE_CMD");

crate::env_flag!(
    default_on dflash_attn_full_gqa_split4_enabled,
    "QWEN_DFLASH_ATTN_FULL_GQA_SPLIT4"
);

fn dflash_swa_ctx_scan_start(
    pos_ctx: &[i32],
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
) -> usize {
    if ctx_len == 0 || swa_window == 0 {
        return 0;
    }
    let rows = &pos_ctx[..ctx_len];
    let mut prev = None;
    for &pos in rows {
        if pos < 0 {
            return 0;
        }
        if let Some(prev) = prev
            && pos < prev
        {
            return 0;
        }
        prev = Some(pos);
    }
    let min_pos = noise_start_pos.saturating_sub(swa_window);
    rows.partition_point(|&pos| (pos as u32) < min_pos)
}

crate::env_flag!(default_on prefill_gdn_batched_enabled, "QWEN_PREFILL_GDN_BATCHED");
crate::env_flag!(default_on prefill_rope_paired_enabled, "QWEN_PREFILL_ROPE_PAIRED");
crate::env_flag!(
    default_on prefill_qk_norm_rope_fused_flag,
    "QWEN_PREFILL_QK_NORM_ROPE_FUSED"
);

fn prefill_qk_norm_rope_fused_enabled(n_tokens: usize) -> bool {
    prefill_qk_norm_rope_fused_flag() && prefill_rope_paired_enabled() && n_tokens <= 48
}

fn encode_prefill_qk_rope(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    n_tokens: usize,
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    n_rot: usize,
    start_position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    if prefill_rope_paired_enabled() {
        encode_rope_neox_pair_adaptive_f32_packed_consecutive(
            ctx,
            enc,
            q,
            k,
            n_tokens,
            n_q_heads,
            n_k_heads,
            head_dim,
            n_rot,
            start_position,
            theta_base,
        )
    } else {
        encode_rope_neox_f32_packed_consecutive(
            ctx,
            enc,
            q,
            n_tokens,
            n_q_heads,
            head_dim,
            n_rot,
            start_position,
            theta_base,
        )?;
        encode_rope_neox_f32_packed_consecutive(
            ctx,
            enc,
            k,
            n_tokens,
            n_k_heads,
            head_dim,
            n_rot,
            start_position,
            theta_base,
        )
    }
}

// Qwen3.8 Q8 verifier arm: batch alpha/beta scheduling across the N rows
// while leaving recurrence and checkpoint semantics unchanged.
crate::env_flag!(
    default_on mtp_verify_q8_gdn_alpha_beta_batched_enabled,
    "QWEN_MTP_VERIFY_Q8_GDN_ALPHA_BETA_BATCHED"
);
crate::env_flag!(
    default_on dflash_verify_packed_gdn_enabled,
    "QWEN_DFLASH_VERIFY_PACKED_GDN"
);
crate::env_flag!(
    default_on dflash_verify_fused_ffn_q4_enabled,
    "QWEN_DFLASH_VERIFY_FUSED_FFN_Q4"
);

fn prefill_gdn_proj_oracle_layer_enabled(layer_idx: usize) -> bool {
    static LAYERS: OnceLock<Option<Vec<usize>>> = OnceLock::new();
    let layers = LAYERS.get_or_init(|| {
        let raw = std::env::var("QWEN_PREFILL_GDN_PROJ_ORACLE_LAYER").ok()?;
        if raw.trim() == "all" {
            return Some(vec![usize::MAX]);
        }
        let parsed: Vec<usize> = raw
            .split(',')
            .filter_map(|part| part.trim().parse().ok())
            .collect();
        (!parsed.is_empty()).then_some(parsed)
    });
    layers
        .as_ref()
        .is_some_and(|layers| layers.contains(&usize::MAX) || layers.contains(&layer_idx))
}

crate::env_flag!(default_off prefill_noop_ffn_enabled, "QWEN_PREFILL_NOOP_FFN");

thread_local! {
    static PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

pub fn with_prefill_dense_ffn_fused_swiglu_q4_override<R>(
    enabled: bool,
    f: impl FnOnce() -> R,
) -> R {
    let previous = PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4_OVERRIDE.with(|slot| {
        let previous = slot.get();
        slot.set(Some(enabled));
        previous
    });
    let out = f();
    PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4_OVERRIDE.with(|slot| slot.set(previous));
    out
}

fn prefill_dense_ffn_fused_swiglu_q4_enabled(hidden: usize) -> bool {
    if let Some(enabled) = PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    if let Some(enabled) = *ENV.get_or_init(|| {
        match std::env::var("QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4").as_deref() {
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") => Some(true),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO") => Some(false),
            _ => None,
        }
    }) {
        return enabled;
    }
    hidden <= 2048
}

fn prefill_mat_mat_dispatch_eligible(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::IQ2_S
            | GgmlType::IQ3_XXS
            | GgmlType::IQ3_S
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
            | GgmlType::IQ4_NL
            | GgmlType::IQ4_XS
    )
}

crate::env_flag!(default_on prefill_gdn_skinny_f32_e8p32_enabled, "QWEN_PREFILL_GDN_SKINNY_E8P32");

crate::env_flag!(default_on prefill_gdn_pair_l2_enabled, "QWEN_PREFILL_GDN_PAIR_L2");

fn prefill_gdn_matvec_proj_enabled(proj: &str) -> bool {
    static PROJS: OnceLock<Option<Vec<String>>> = OnceLock::new();
    let projs = PROJS.get_or_init(|| {
        let raw = std::env::var("QWEN_PREFILL_GDN_MATVEC_PROJ").ok()?;
        let parsed: Vec<String> = raw
            .split(',')
            .map(|part| part.trim().to_ascii_lowercase())
            .filter(|part| !part.is_empty())
            .collect();
        (!parsed.is_empty()).then_some(parsed)
    });
    let front = matches!(proj, "qkv" | "z" | "beta" | "alpha");
    projs.as_ref().is_some_and(|projs| {
        projs
            .iter()
            .any(|mode| mode == "all" || mode == proj || (mode == "front" && front))
    })
}

fn prefill_gdn_matvec_layer_enabled(layer_idx: usize) -> bool {
    static LAYERS: OnceLock<Option<Vec<usize>>> = OnceLock::new();
    let layers = LAYERS.get_or_init(|| {
        let raw = std::env::var("QWEN_PREFILL_GDN_MATVEC_LAYER").ok()?;
        if raw.trim() == "all" {
            return Some(vec![usize::MAX]);
        }
        let parsed: Vec<usize> = raw
            .split(',')
            .filter_map(|part| part.trim().parse().ok())
            .collect();
        (!parsed.is_empty()).then_some(parsed)
    });
    layers
        .as_ref()
        .is_none_or(|layers| layers.contains(&usize::MAX) || layers.contains(&layer_idx))
}

fn prefill_gdn_matvec_projection_enabled(proj: &str, layer_idx: usize) -> bool {
    prefill_gdn_matvec_proj_enabled(proj) && prefill_gdn_matvec_layer_enabled(layer_idx)
}

crate::env_flag!(default_on prefill_moe_packed_routed_enabled, "QWEN_PREFILL_MOE_PACKED_ROUTED");

crate::env_flag!(default_on prefill_moe_packed_route_enabled, "QWEN_PREFILL_MOE_PACKED_ROUTE");

crate::env_flag!(default_on prefill_moe_packed_down_sum_enabled, "QWEN_PREFILL_MOE_PACKED_DOWN_SUM");

crate::env_flag!(default_on prefill_moe_packed_shared_enabled, "QWEN_PREFILL_MOE_PACKED_SHARED");

crate::env_flag!(default_off prefill_noop_moe_shared_enabled, "QWEN_PREFILL_NOOP_MOE_SHARED");

crate::env_flag!(default_off prefill_noop_moe_routed_enabled, "QWEN_PREFILL_NOOP_MOE_ROUTED");

crate::env_flag!(default_off prefill_noop_moe_grouped_swiglu_enabled, "QWEN_PREFILL_NOOP_MOE_GROUPED_SWIGLU");

crate::env_flag!(default_off prefill_noop_moe_grouped_down_enabled, "QWEN_PREFILL_NOOP_MOE_GROUPED_DOWN");

crate::env_flag!(default_off prefill_noop_moe_grouped_reduce_enabled, "QWEN_PREFILL_NOOP_MOE_GROUPED_REDUCE");

crate::env_flag!(default_on prefill_moe_grouped_enabled, "QWEN_PREFILL_MOE_GROUPED");

crate::env_flag!(default_off mtp_moe_verify_grouped_ffn_enabled, "QWEN_MTP_MOE_VERIFY_GROUPED_FFN");

crate::env_flag!(default_on mtp_moe_verify_batched_mixer_enabled, "QWEN_MTP_MOE_VERIFY_BATCHED_MIXER");

crate::env_flag!(default_on mtp_moe_verify_concurrent_ffn_enabled, "QWEN_MTP_MOE_VERIFY_CONCURRENT_FFN");

crate::env_flag!(default_off mtp_verify_trace_counts_enabled, "QWEN_MTP_VERIFY_TRACE_COUNTS");

crate::env_flag!(default_on packed_verify_skip_final_ckpt_enabled, "QWEN_MTP_SKIP_FINAL_CKPT");

crate::env_flag!(default_on mtp_moe_verify_row_views_enabled, "QWEN_MTP_MOE_VERIFY_ROW_VIEWS");

crate::env_flag!(default_on mtp_moe_verify_batched_route_enabled, "QWEN_MTP_MOE_VERIFY_BATCHED_ROUTE");

fn prefill_moe_grouped_q5_gateup_enabled(h: usize, f_exp: usize, n_expert: usize) -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    match *MODE.get_or_init(|| env_mode("QWEN_PREFILL_MOE_GROUPED_Q5_GATEUP")) {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => h == 3072 && f_exp == 1024 && n_expert == 256,
    }
}

fn prefill_moe_grouped_q6_gateup_enabled(h: usize, f_exp: usize, n_expert: usize) -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    match *MODE.get_or_init(|| env_mode("QWEN_PREFILL_MOE_GROUPED_Q6_GATEUP")) {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => h == 2048 && f_exp == 512 && n_expert == 256,
    }
}

fn prefill_moe_grouped_q8_gateup_enabled(h: usize, f_exp: usize, n_expert: usize) -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    match *MODE.get_or_init(|| env_mode("QWEN_PREFILL_MOE_GROUPED_Q8_GATEUP")) {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => h == 2048 && f_exp == 512 && n_expert == 256,
    }
}

fn prefill_moe_grouped_f32_gateup_enabled() -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    match *MODE.get_or_init(|| env_mode("QWEN_PREFILL_MOE_GROUPED_F32_GATEUP")) {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff | PrefillEnvMode::Auto => false,
    }
}

fn prefill_moe_grouped_iq3_gateup_enabled(h: usize, f_exp: usize, n_expert: usize) -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    match *MODE.get_or_init(|| env_mode("QWEN_PREFILL_MOE_GROUPED_IQ3_GATEUP")) {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => h == 2048 && f_exp == 512 && n_expert == 256,
    }
}

fn prefill_moe_grouped_bf16_gateup_enabled(h: usize, f_exp: usize, n_expert: usize) -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    match *MODE.get_or_init(|| env_mode("QWEN_PREFILL_MOE_GROUPED_BF16_GATEUP")) {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => h == 2048 && f_exp == 512 && n_expert == 256,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillEnvMode {
    Auto,
    ForceOff,
    ForceOn,
}

fn env_mode(name: &str) -> PrefillEnvMode {
    match std::env::var(name).as_deref() {
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") => PrefillEnvMode::ForceOn,
        Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO") => PrefillEnvMode::ForceOff,
        _ => PrefillEnvMode::Auto,
    }
}

fn prefill_moe_grouped_hot_q4_n32_mode() -> PrefillEnvMode {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    *MODE.get_or_init(|| env_mode("QWEN_PREFILL_MOE_GROUPED_HOT_Q4_N32"))
}

fn prefill_moe_grouped_hot_q4_n32_enabled(chunk_p: usize) -> bool {
    match prefill_moe_grouped_hot_q4_n32_mode() {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => chunk_p >= 512,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillMoeGroupedQ4N32Mode {
    ForceOff,
    ForceOn,
}

fn prefill_moe_grouped_q4_n32_mode() -> PrefillMoeGroupedQ4N32Mode {
    static MODE: OnceLock<PrefillMoeGroupedQ4N32Mode> = OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("QWEN_PREFILL_MOE_GROUPED_Q4_N32").as_deref() {
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") => {
                PrefillMoeGroupedQ4N32Mode::ForceOn
            }
            _ => PrefillMoeGroupedQ4N32Mode::ForceOff,
        },
    )
}

fn prefill_moe_grouped_q4_n32_all_enabled(_arch: &crate::model::Arch, _chunk_p: usize) -> bool {
    match prefill_moe_grouped_q4_n32_mode() {
        PrefillMoeGroupedQ4N32Mode::ForceOn => true,
        PrefillMoeGroupedQ4N32Mode::ForceOff => false,
    }
}

crate::env_flag!(default_off prefill_moe_fused_finalizer_enabled, "QWEN_PREFILL_MOE_FUSED_FINALIZER");

crate::env_flag!(default_off prefill_moe_grouped_zero_fill_enabled, "QWEN_PREFILL_MOE_GROUPED_ZERO_FILL");

fn prefill_moe_tiny_down_r16_enabled(chunk_p: usize) -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    match *MODE.get_or_init(|| env_mode("QWEN_PREFILL_MOE_TINY8_DOWN_R16")) {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => chunk_p <= 768,
    }
}

fn prefill_moe_grouped_concurrent_tail_enabled(chunk_p: usize) -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    if chunk_p < 512 {
        return false;
    }
    match *MODE.get_or_init(|| env_mode("QWEN_PREFILL_MOE_GROUPED_CONCURRENT_TAIL")) {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => false,
    }
}

fn encode_prefill_moe_grouped_swiglu_q4(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    moe: &MetalMoeFfn,
    h_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    h: usize,
    f_exp: usize,
    n_expert: usize,
    topk: usize,
    chunk_p: usize,
    grouped_q4_n32_all: bool,
    hot_expert_min_slots: Option<usize>,
) -> Result<(), MetalError> {
    if grouped_q4_n32_all {
        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32(
            ctx,
            enc,
            &moe.gate_exps,
            &moe.up_exps,
            h_pack,
            counts,
            ids,
            inner,
            h,
            f_exp,
            n_expert,
            topk,
            chunk_p,
        )
    } else if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
        if let Some(hot_threshold) = hot_expert_min_slots {
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
                hot_threshold as u32,
                i32::MAX as u32,
            )?;
            if hot_threshold > 0 {
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
                    ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    h_pack,
                    counts,
                    ids,
                    inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                    0,
                    hot_threshold.saturating_sub(1) as u32,
                )?;
            }
            Ok(())
        } else {
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
        }
    } else {
        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
            ctx,
            enc,
            &moe.gate_exps,
            &moe.up_exps,
            h_pack,
            counts,
            ids,
            inner,
            h,
            f_exp,
            n_expert,
            topk,
            chunk_p,
        )
    }
}

fn encode_prefill_moe_grouped_swiglu(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    moe: &MetalMoeFfn,
    h_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    h: usize,
    f_exp: usize,
    n_expert: usize,
    topk: usize,
    chunk_p: usize,
    grouped_q4_n32_all: bool,
    hot_expert_min_slots: Option<usize>,
) -> Result<(), MetalError> {
    match (moe.gate_exps.dtype, moe.up_exps.dtype) {
        (GgmlType::Q4_K, GgmlType::Q4_K) => encode_prefill_moe_grouped_swiglu_q4(
            ctx,
            enc,
            moe,
            h_pack,
            counts,
            ids,
            inner,
            h,
            f_exp,
            n_expert,
            topk,
            chunk_p,
            grouped_q4_n32_all,
            hot_expert_min_slots,
        ),
        (GgmlType::Q5_K, GgmlType::Q5_K) => {
            crate::metal::encode_moe_swiglu_q5_K_f32_grouped_slots_n16(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
        }
        (GgmlType::Q6_K, GgmlType::Q6_K) => {
            crate::metal::encode_moe_swiglu_q6_K_f32_grouped_slots_n16(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
        }
        (GgmlType::Q8_0, GgmlType::Q8_0) => {
            crate::metal::encode_moe_swiglu_q8_0_f32_grouped_slots_n16(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
        }
        (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS) => {
            crate::metal::encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
        }
        (GgmlType::IQ3_S, GgmlType::IQ3_S) => {
            crate::metal::encode_moe_swiglu_iq3_s_f32_grouped_slots_n16(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
        }
        (GgmlType::F32, GgmlType::F32) => {
            crate::metal::encode_moe_swiglu_f32_f32_grouped_slots_n16(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
        }
        (GgmlType::BF16, GgmlType::BF16) => {
            crate::metal::encode_moe_swiglu_bf16_f32_grouped_slots_n16(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
        }
        other => Err(MetalError::BadShape {
            kernel: "prefill_moe_grouped_swiglu",
            detail: format!("unsupported grouped gate/up dtypes {other:?}"),
        }),
    }
}

fn encode_prefill_moe_grouped_swiglu_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    moe: &MetalMoeFfn,
    h_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    h: usize,
    f_exp: usize,
    n_expert: usize,
    topk: usize,
    chunk_p: usize,
    min_slots: u32,
    max_slots: u32,
    use_q4_n32: bool,
) -> Result<(), MetalError> {
    match (moe.gate_exps.dtype, moe.up_exps.dtype) {
        (GgmlType::Q4_K, GgmlType::Q4_K) if use_q4_n32 => {
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
                min_slots,
                max_slots,
            )
        }
        (GgmlType::Q4_K, GgmlType::Q4_K) => {
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
                min_slots,
                max_slots,
            )
        }
        (GgmlType::Q5_K, GgmlType::Q5_K) => {
            crate::metal::encode_moe_swiglu_q5_K_f32_grouped_slots_n16_range(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
                min_slots,
                max_slots,
            )
        }
        (GgmlType::Q6_K, GgmlType::Q6_K) => {
            crate::metal::encode_moe_swiglu_q6_K_f32_grouped_slots_n16_range(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
                min_slots,
                max_slots,
            )
        }
        (GgmlType::Q8_0, GgmlType::Q8_0) => {
            crate::metal::encode_moe_swiglu_q8_0_f32_grouped_slots_n16_range(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
                min_slots,
                max_slots,
            )
        }
        (GgmlType::BF16, GgmlType::BF16) => {
            crate::metal::encode_moe_swiglu_bf16_f32_grouped_slots_n16_range(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
                min_slots,
                max_slots,
            )
        }
        (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS) => {
            crate::metal::encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16_range(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
                min_slots,
                max_slots,
            )
        }
        (GgmlType::IQ3_S, GgmlType::IQ3_S) => {
            crate::metal::encode_moe_swiglu_iq3_s_f32_grouped_slots_n16_range(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                h_pack,
                counts,
                ids,
                inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
                min_slots,
                max_slots,
            )
        }
        other => Err(MetalError::BadShape {
            kernel: "prefill_moe_grouped_swiglu_range",
            detail: format!("unsupported grouped gate/up dtypes {other:?}"),
        }),
    }
}

fn encode_prefill_moe_grouped_down(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    down_exps: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    f_exp: usize,
    h: usize,
    n_expert: usize,
    chunk_p: usize,
) -> Result<(), MetalError> {
    match down_exps.dtype {
        GgmlType::Q5_K if prefill_moe_tiny_down_r16_enabled(chunk_p) => {
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots_tiny8_r16(
                ctx, enc, down_exps, inner, counts, ids, out, f_exp, h, n_expert, chunk_p, 1, 7,
            )?;
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots_range(
                ctx,
                enc,
                down_exps,
                inner,
                counts,
                ids,
                out,
                f_exp,
                h,
                n_expert,
                chunk_p,
                8,
                i32::MAX as u32,
            )
        }
        GgmlType::Q5_K => crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
            ctx, enc, down_exps, inner, counts, ids, out, f_exp, h, n_expert, chunk_p,
        ),
        GgmlType::Q6_K => crate::metal::encode_moe_down_q6_K_f32_grouped_slots(
            ctx, enc, down_exps, inner, counts, ids, out, f_exp, h, n_expert, chunk_p,
        ),
        GgmlType::Q8_0 => crate::metal::encode_moe_down_q8_0_f32_grouped_slots(
            ctx, enc, down_exps, inner, counts, ids, out, f_exp, h, n_expert, chunk_p,
        ),
        GgmlType::IQ4_XS => crate::metal::encode_moe_down_iq4_xs_f32_grouped_slots(
            ctx, enc, down_exps, inner, counts, ids, out, f_exp, h, n_expert, chunk_p,
        ),
        GgmlType::BF16 => crate::metal::encode_moe_down_bf16_f32_grouped_slots(
            ctx, enc, down_exps, inner, counts, ids, out, f_exp, h, n_expert, chunk_p,
        ),
        other => Err(MetalError::BadShape {
            kernel: "prefill_moe_grouped_down",
            detail: format!("unsupported grouped down dtype {other:?}"),
        }),
    }
}

fn prefill_moe_route_bucket_fused_mode() -> PrefillEnvMode {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    *MODE.get_or_init(|| env_mode("QWEN_PREFILL_MOE_ROUTE_BUCKET_FUSED"))
}

fn prefill_moe_route_bucket_fused_auto_enabled(h: usize, chunk_p: usize) -> bool {
    match prefill_moe_route_bucket_fused_mode() {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => {
            let min_chunk = if h <= 2048 { 128 } else { 512 };
            chunk_p >= min_chunk
        }
    }
}

fn prefill_moe_hot_expert_min_slots() -> Option<usize> {
    static MIN_SLOTS: OnceLock<Option<usize>> = OnceLock::new();
    *MIN_SLOTS.get_or_init(|| match std::env::var("QWEN_PREFILL_MOE_HOT_EXPERT_MIN") {
        Ok(v) => v.parse::<usize>().ok().filter(|&n| n > 0),
        Err(_)
            if !matches!(
                prefill_moe_grouped_hot_q4_n32_mode(),
                PrefillEnvMode::ForceOff
            ) =>
        {
            Some(48)
        }
        Err(_) => None,
    })
}

fn prefill_moe_route_logits_e8p32_mode() -> PrefillEnvMode {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    *MODE.get_or_init(|| env_mode("QWEN_PREFILL_MOE_ROUTE_LOGITS_E8P32"))
}

crate::env_flag!(default_off prefill_trace_labels_enabled, "QWEN_PREFILL_TRACE_LABELS");

crate::env_flag!(default_off prefill_trace_chunks_enabled, "QWEN_PREFILL_TRACE_CHUNKS");

crate::env_flag!(default_off prefill_trace_wall_enabled, "QWEN_PREFILL_TRACE_WALL");

crate::env_flag!(default_off prefill_trace_counts_enabled, "QWEN_PREFILL_TRACE_COUNTS");

crate::env_flag!(default_off prefill_trace_attn_phases_enabled, "QWEN_PREFILL_TRACE_ATTN_PHASES");

crate::env_flag!(default_off prefill_trace_layer_phases_enabled, "QWEN_PREFILL_TRACE_LAYER_PHASES");

crate::env_flag!(default_off prefill_trace_ffn_subphases_enabled, "QWEN_PREFILL_TRACE_FFN_SUBPHASES");

crate::env_flag!(default_off prefill_trace_moe_buckets_enabled, "QWEN_PREFILL_TRACE_MOE_BUCKETS");

crate::env_flag!(default_off prefill_trace_moe_bucket_bins_enabled, "QWEN_PREFILL_TRACE_MOE_BUCKET_BINS");

fn trace_prefill_moe_bucket_stats(
    enabled: bool,
    chunk_idx: usize,
    chunk_start: u32,
    layer_idx: usize,
    counts: &MetalTensor,
    n_expert: usize,
    chunk_p: usize,
    topk: usize,
    hot_expert_min_slots: Option<usize>,
) {
    if !enabled {
        return;
    }
    let counts_cpu = cpu_read_i32_f32buf(counts);
    let mut active: Vec<usize> = counts_cpu
        .iter()
        .take(n_expert)
        .filter_map(|&c| (c > 0).then_some(c as usize))
        .collect();
    active.sort_unstable();
    let total: usize = active.iter().sum();
    let p50 = active
        .get(active.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(0);
    let p90 = if active.is_empty() {
        0
    } else {
        active[(active.len() * 9 / 10).min(active.len() - 1)]
    };
    let max = active.last().copied().unwrap_or(0);
    let ge16 = active.iter().filter(|&&c| c >= 16).count();
    let ge32 = active.iter().filter(|&&c| c >= 32).count();
    let ge48 = active.iter().filter(|&&c| c >= 48).count();
    let bin_stats = |min_slots: usize, max_slots: usize| -> (usize, usize) {
        active
            .iter()
            .copied()
            .filter(|&c| c >= min_slots && c <= max_slots)
            .fold((0, 0), |(experts, slots), c| (experts + 1, slots + c))
    };
    let (e_lt8, s_lt8) = bin_stats(1, 7);
    let (e_8_15, s_8_15) = bin_stats(8, 15);
    let (e_lt16, s_lt16) = bin_stats(1, 15);
    let (e_16_31, s_16_31) = bin_stats(16, 31);
    let (e_32_47, s_32_47) = bin_stats(32, 47);
    let (e_48_63, s_48_63) = bin_stats(48, 63);
    let (e_ge64, s_ge64) = bin_stats(64, usize::MAX);
    let (hot_min, hot_experts, hot_slots) = if let Some(min_slots) = hot_expert_min_slots {
        let hot_experts = active.iter().filter(|&&c| c >= min_slots).count();
        let hot_slots = active.iter().filter(|&&c| c >= min_slots).sum::<usize>();
        (min_slots as isize, hot_experts, hot_slots)
    } else {
        (-1, 0, 0)
    };
    eprintln!(
        "[prefill-moe-buckets] chunk={chunk_idx} start={chunk_start} layer={layer_idx} total={total}/{} active={} p50={p50} p90={p90} max={max} ge16={ge16} ge32={ge32} ge48={ge48} hot_min={hot_min} hot_experts={hot_experts} hot_slots={hot_slots} e_lt8={e_lt8} s_lt8={s_lt8} e_8_15={e_8_15} s_8_15={s_8_15} e_lt16={e_lt16} s_lt16={s_lt16} e_16_31={e_16_31} s_16_31={s_16_31} e_32_47={e_32_47} s_32_47={s_32_47} e_48_63={e_48_63} s_48_63={s_48_63} e_ge64={e_ge64} s_ge64={s_ge64}",
        chunk_p * topk,
        active.len(),
    );
}

fn flush_prefill_phase(
    ctx: &MetalContext,
    cmd_buf: &mut Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    prefill_gpu_total_ms: &mut f64,
    enabled: bool,
    chunk_idx: usize,
    chunk_start: u32,
    layer_idx: usize,
    phase: &str,
) -> Result<(), DFlashError> {
    emit_prefill_count_phase(
        prefill_trace_counts_enabled(),
        chunk_idx,
        chunk_start,
        layer_idx,
        "attn-detail",
        phase,
    );
    if !enabled {
        return Ok(());
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    require_prefill_command_completed(cmd_buf)?;
    let gpu_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;
    *prefill_gpu_total_ms += gpu_ms;
    eprintln!(
        "[prefill-attn-phase] chunk={} start={} layer={} phase={} gpu_ms={:.2}",
        chunk_idx, chunk_start, layer_idx, phase, gpu_ms
    );
    *cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    Ok(())
}

fn require_prefill_command_completed(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<(), DFlashError> {
    let status = cmd_buf.status();
    let error = cmd_buf.error();
    if status != MTLCommandBufferStatus::Completed || error.is_some() {
        return Err(DFlashError::MetalForward(MfError::CommandBuffer {
            status: format!("{status:?}"),
            error: format!("{error:?}"),
        }));
    }
    Ok(())
}

fn emit_prefill_count_phase(
    enabled: bool,
    chunk_idx: usize,
    chunk_start: u32,
    layer_idx: usize,
    kind: &str,
    phase: &str,
) {
    if !enabled {
        return;
    }
    let delta = kernel_trace_take_delta();
    if delta.is_zero() {
        return;
    }
    eprintln!(
        "[prefill-count-phase] chunk={} start={} layer={} kind={} phase={} encoders={} concurrent_encoders={} dispatches={}",
        chunk_idx,
        chunk_start,
        layer_idx,
        kind,
        phase,
        delta.encoders,
        delta.concurrent_encoders,
        delta.dispatches,
    );
}

fn emit_mtp_verify_count_phase(enabled: bool, layer_idx: isize, kind: &str, phase: &str) {
    if !enabled {
        return;
    }
    let delta = kernel_trace_take_delta();
    if delta.is_zero() {
        return;
    }
    eprintln!(
        "[mtp-verify-count-phase] layer={} kind={} phase={} encoders={} concurrent_encoders={} dispatches={}",
        layer_idx, kind, phase, delta.encoders, delta.concurrent_encoders, delta.dispatches,
    );
}

fn flush_prefill_layer_phase(
    ctx: &MetalContext,
    cmd_buf: &mut Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    prefill_gpu_total_ms: &mut f64,
    enabled: bool,
    chunk_idx: usize,
    chunk_start: u32,
    layer_idx: usize,
    kind: &str,
    phase: &str,
) -> Result<(), DFlashError> {
    emit_prefill_count_phase(
        prefill_trace_counts_enabled(),
        chunk_idx,
        chunk_start,
        layer_idx,
        kind,
        phase,
    );
    if !enabled {
        return Ok(());
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    require_prefill_command_completed(cmd_buf)?;
    let gpu_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;
    *prefill_gpu_total_ms += gpu_ms;
    eprintln!(
        "[prefill-layer-phase] chunk={} start={} layer={} kind={} phase={} gpu_ms={:.2}",
        chunk_idx, chunk_start, layer_idx, kind, phase, gpu_ms
    );
    *cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    Ok(())
}

fn flush_prefill_layer_phase_accum(
    ctx: &MetalContext,
    cmd_buf: &mut Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    prefill_gpu_total_ms: &mut f64,
) -> Result<f64, DFlashError> {
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    require_prefill_command_completed(cmd_buf)?;
    let gpu_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;
    *prefill_gpu_total_ms += gpu_ms;
    *cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    Ok(gpu_ms)
}

fn diagnose_gdn_projection_matmat_vs_matvec(
    ctx: &MetalContext,
    cmd_buf: &mut Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    prefill_gpu_total_ms: &mut f64,
    chunk_idx: usize,
    chunk_start: u32,
    layer_idx: usize,
    proj: &str,
    weight: &MetalTensor,
    x_pack: &MetalTensor,
    matmat_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), DFlashError> {
    let matvec_out =
        MetalTensor::zeros_f32(ctx, vec![(n_query * n_out) as u64]).map_err(DFlashError::Metal)?;
    {
        let enc = KernelEncoder::begin(cmd_buf);
        for row in 0..n_query {
            let x_row = x_pack.view_subrange((row * n_in) as u64, vec![n_in as u64]);
            let y_row = matvec_out.view_subrange((row * n_out) as u64, vec![n_out as u64]);
            encode_mat_vec_dispatch(ctx, &enc, weight, &x_row, &y_row, n_in, n_out)?;
        }
        enc.end();
    }
    let gpu_ms = flush_prefill_layer_phase_accum(ctx, cmd_buf, prefill_gpu_total_ms)?;

    let mm = read_f32_tensor(matmat_out);
    let mv = read_f32_tensor(&matvec_out);
    let mut min_cos = f64::INFINITY;
    let mut worst_row = 0usize;
    let mut worst_max_abs = 0.0f32;
    for row in 0..n_query {
        let start = row * n_out;
        let end = start + n_out;
        let cos = cosine_f32(&mm[start..end], &mv[start..end]);
        if cos < min_cos {
            min_cos = cos;
            worst_row = row;
            worst_max_abs = max_abs_delta_f32(&mm[start..end], &mv[start..end]);
        }
    }

    eprintln!(
        "[prefill-gdn-proj-oracle] chunk={chunk_idx} start={chunk_start} \
         layer={layer_idx} proj={proj} dtype={:?} rows={n_query} n_in={n_in} \
         n_out={n_out} cos_min={min_cos:.6} worst_row={worst_row} \
         max|Δ|={worst_max_abs:.3e} flush_gpu_ms={gpu_ms:.2}",
        weight.dtype,
    );
    Ok(())
}

fn encode_packed_matvec_projection(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x_pack: &MetalTensor,
    y_pack: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), DFlashError> {
    for row in 0..n_query {
        let x_row = x_pack.view_subrange((row * n_in) as u64, vec![n_in as u64]);
        let y_row = y_pack.view_subrange((row * n_out) as u64, vec![n_out as u64]);
        encode_mat_vec_dispatch(ctx, enc, weight, &x_row, &y_row, n_in, n_out)?;
    }
    Ok(())
}

fn read_f32_tensor(t: &MetalTensor) -> Vec<f32> {
    assert_eq!(t.dtype, GgmlType::F32);
    let n = t.n_elements() as usize;
    let mut out = vec![0.0f32; n];
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize) as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }
    out
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

fn max_abs_delta_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn emit_prefill_layer_phase(
    chunk_idx: usize,
    chunk_start: u32,
    layer_idx: usize,
    kind: &str,
    phase: &str,
    gpu_ms: f64,
) {
    eprintln!(
        "[prefill-layer-phase] chunk={} start={} layer={} kind={} phase={} gpu_ms={:.2}",
        chunk_idx, chunk_start, layer_idx, kind, phase, gpu_ms
    );
}

fn prefill_attn_packed_g8_enabled(n_pos: usize, group: usize) -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    if group != 8 || n_pos < prefill_attn_packed_g8_min_pos() {
        return false;
    }
    match *MODE.get_or_init(|| env_mode("QWEN_PREFILL_ATTN_PACKED_G8")) {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => true,
    }
}

fn prefill_attn_packed_g16_enabled(n_pos: usize, group: usize) -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    if group != 16 || n_pos < prefill_attn_packed_g16_min_pos() {
        return false;
    }
    match *MODE.get_or_init(|| env_mode("QWEN_PREFILL_ATTN_PACKED_G16")) {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => true,
    }
}

fn prefill_attn_packed_g8_min_pos() -> usize {
    static MIN_POS: OnceLock<usize> = OnceLock::new();
    *MIN_POS.get_or_init(|| {
        std::env::var("QWEN_PREFILL_ATTN_PACKED_G8_MIN_POS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(128)
    })
}

fn prefill_attn_packed_g16_min_pos() -> usize {
    static MIN_POS: OnceLock<usize> = OnceLock::new();
    *MIN_POS.get_or_init(|| {
        std::env::var("QWEN_PREFILL_ATTN_PACKED_G16_MIN_POS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(128)
    })
}

fn prefill_attn_fused_qkv_g8_enabled(n_pos: usize, group: usize) -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    if group != 8 || n_pos < 4096 {
        return false;
    }
    match *MODE.get_or_init(|| env_mode("QWEN_PREFILL_ATTN_FUSED_QKV_G8")) {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => false,
    }
}

fn prefill_attn_matrix_g4_mode() -> PrefillEnvMode {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    *MODE.get_or_init(|| env_mode("QWEN_PREFILL_ATTN_MATRIX_G4"))
}

fn prefill_attn_matrix_g4_may_use() -> bool {
    !matches!(prefill_attn_matrix_g4_mode(), PrefillEnvMode::ForceOff)
}

fn prefill_attn_matrix_g8_mode() -> PrefillEnvMode {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    *MODE.get_or_init(|| env_mode("QWEN_PREFILL_ATTN_MATRIX_G8"))
}

fn prefill_attn_matrix_g8_may_use() -> bool {
    !matches!(prefill_attn_matrix_g8_mode(), PrefillEnvMode::ForceOff)
}

fn prefill_attn_matrix_g6_mode() -> PrefillEnvMode {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    *MODE.get_or_init(|| env_mode("QWEN_PREFILL_ATTN_MATRIX_G6"))
}

fn prefill_attn_matrix_g6_may_use() -> bool {
    !matches!(prefill_attn_matrix_g6_mode(), PrefillEnvMode::ForceOff)
}

fn prefill_attn_matrix_causal_skip_enabled() -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    !matches!(
        *MODE.get_or_init(|| env_mode("QWEN_PREFILL_ATTN_MATRIX_CAUSAL_SKIP")),
        PrefillEnvMode::ForceOff
    )
}

/// v0.439: two-pass online-softmax matrix attention (KQ folds the softmax
/// into its epilogue and stores F16 `P~` + an (m, l) sidecar; KQV normalizes
/// during staging). Deletes the separate softmax dispatch and drops score
/// traffic from 16 B/elem to 4 B/elem (microbench 1.22-1.37x on the matrix
/// body). Rollback: `QWEN_PREFILL_ATTN_MATRIX_ONLINE=0` restores the
/// three-kernel F32 sidecar (and its full-size F32 score scratch).
fn prefill_attn_matrix_online_enabled() -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    !matches!(
        *MODE.get_or_init(|| env_mode("QWEN_PREFILL_ATTN_MATRIX_ONLINE")),
        PrefillEnvMode::ForceOff
    )
}

fn parse_prefill_attn_matrix_query_cap(value: Option<&str>) -> Result<Option<usize>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let cap = value.parse::<usize>().map_err(|_| {
        format!("QWEN_PREFILL_ATTN_MATRIX_QUERY_CAP must be a positive integer, got {value:?}")
    })?;
    if cap == 0 {
        return Err("QWEN_PREFILL_ATTN_MATRIX_QUERY_CAP must be greater than zero".into());
    }
    Ok(Some(cap))
}

fn prefill_attn_matrix_query_cap() -> Result<Option<usize>, String> {
    static QUERY_CAP: OnceLock<Result<Option<usize>, String>> = OnceLock::new();
    QUERY_CAP
        .get_or_init(|| {
            let value = std::env::var("QWEN_PREFILL_ATTN_MATRIX_QUERY_CAP");
            match value {
                Ok(value) => parse_prefill_attn_matrix_query_cap(Some(&value)),
                Err(std::env::VarError::NotPresent) => parse_prefill_attn_matrix_query_cap(None),
                Err(std::env::VarError::NotUnicode(value)) => Err(format!(
                    "QWEN_PREFILL_ATTN_MATRIX_QUERY_CAP is not Unicode: {value:?}"
                )),
            }
        })
        .clone()
}

fn resolve_prefill_attn_matrix_query_cap(
    include_spec_packs: bool,
    config: Option<PrefillScratchConfig>,
) -> Result<Option<usize>, String> {
    resolve_prefill_attn_matrix_query_cap_with(
        include_spec_packs,
        config,
        prefill_attn_matrix_query_cap,
    )
}

fn resolve_prefill_attn_matrix_query_cap_with(
    include_spec_packs: bool,
    config: Option<PrefillScratchConfig>,
    environment: impl FnOnce() -> Result<Option<usize>, String>,
) -> Result<Option<usize>, String> {
    if include_spec_packs {
        return Ok(None);
    }
    match config.and_then(|value| value.matrix_query_cap) {
        Some(0) => Err("matrix query cap must be greater than zero".into()),
        Some(cap) => Ok(Some(cap)),
        None => environment(),
    }
}

fn prefill_scratch_overlay_diagnostic_mode_present() -> bool {
    const DIAGNOSTIC_ENV: &[&str] = &[
        "QWEN_PREFILL_GDN_PROJ_ORACLE_LAYER",
        "QWEN_PREFILL_GDN_MATVEC_PROJ",
        "QWEN_PREFILL_GDN_MATVEC_LAYER",
        "QWEN_PREFILL_GDN_SPLIT",
        "QWEN_PREFILL_NOOP_GDN_BODY",
        "QWEN_PREFILL_NOOP_ATTN_BODY",
        "QWEN_PREFILL_NOOP_FFN",
        "QWEN_PREFILL_NOOP_MOE_SHARED",
        "QWEN_PREFILL_NOOP_MOE_ROUTED",
        "QWEN_PREFILL_NOOP_MOE_GROUPED_SWIGLU",
        "QWEN_PREFILL_NOOP_MOE_GROUPED_DOWN",
        "QWEN_PREFILL_NOOP_MOE_GROUPED_REDUCE",
        "QWEN_PREFILL_ATTN_PACKED_G8_ORACLE",
        "QWEN_PREFILL_ATTN_PACKED_G16_ORACLE",
    ];
    DIAGNOSTIC_ENV
        .iter()
        .any(|name| std::env::var_os(name).is_some())
}

fn attn_matrix_query_tiles(n_rows: usize, max_rows: usize) -> impl Iterator<Item = (usize, usize)> {
    assert!(max_rows > 0, "matrix attention query tile must be nonzero");
    (0..n_rows)
        .step_by(max_rows)
        .map(move |row_base| (row_base, (n_rows - row_base).min(max_rows)))
}

pub(crate) fn attn_matrix_vt_prefix_rebuild_rows(
    use_matrix: bool,
    valid_until: usize,
    chunk_start: usize,
) -> Option<usize> {
    (use_matrix && valid_until < chunk_start).then_some(chunk_start)
}

fn prefill_attn_matrix_g16_mode() -> PrefillEnvMode {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    *MODE.get_or_init(|| env_mode("QWEN_PREFILL_ATTN_MATRIX_G16"))
}

fn prefill_attn_matrix_g16_may_use() -> bool {
    !matches!(prefill_attn_matrix_g16_mode(), PrefillEnvMode::ForceOff)
}

fn prefill_attn_matrix_max_pos() -> Option<usize> {
    static MAX_POS: OnceLock<Option<usize>> = OnceLock::new();
    *MAX_POS.get_or_init(|| {
        std::env::var("QWEN_PREFILL_ATTN_MATRIX_MAX_POS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
    })
}

fn prefill_attn_packed_g8_rows() -> usize {
    static ROWS: OnceLock<usize> = OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("QWEN_PREFILL_ATTN_PACKED_G8_ROWS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| (1..=ATTN_PREFILL_V4_PACKED_ROWS).contains(&n))
            .unwrap_or(4)
    })
}

fn prefill_attn_packed_g8_qt() -> usize {
    static QT: OnceLock<usize> = OnceLock::new();
    *QT.get_or_init(|| {
        std::env::var("QWEN_PREFILL_ATTN_PACKED_G8_QT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| matches!(n, 2 | 4))
            .unwrap_or(2)
    })
}

fn prefill_attn_packed_g8_nwg() -> usize {
    static NWG: OnceLock<usize> = OnceLock::new();
    *NWG.get_or_init(|| {
        std::env::var("QWEN_PREFILL_ATTN_PACKED_G8_NWG")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| matches!(n, 8 | 16 | 32 | 64))
            .unwrap_or(64)
    })
}

fn prefill_attn_packed_g16_rows() -> usize {
    static ROWS: OnceLock<usize> = OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("QWEN_PREFILL_ATTN_PACKED_G16_ROWS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| (1..=ATTN_PREFILL_V4_PACKED_ROWS).contains(&n))
            .unwrap_or(4)
    })
}

fn prefill_attn_packed_g16_qt() -> usize {
    static QT: OnceLock<usize> = OnceLock::new();
    *QT.get_or_init(|| {
        std::env::var("QWEN_PREFILL_ATTN_PACKED_G16_QT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| matches!(n, 2 | 4))
            .unwrap_or(2)
    })
}

fn prefill_attn_packed_g16_nwg() -> usize {
    static NWG: OnceLock<usize> = OnceLock::new();
    *NWG.get_or_init(|| {
        std::env::var("QWEN_PREFILL_ATTN_PACKED_G16_NWG")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| matches!(n, 8 | 16 | 32 | 64))
            .unwrap_or(32)
    })
}

crate::env_flag!(default_off prefill_attn_packed_g8_oracle_enabled, "QWEN_PREFILL_ATTN_PACKED_G8_ORACLE");

crate::env_flag!(default_off prefill_attn_packed_g16_oracle_enabled, "QWEN_PREFILL_ATTN_PACKED_G16_ORACLE");

fn label_prefill_encoder(enc: &KernelEncoder, layer: usize, label: &str) {
    if prefill_trace_labels_enabled() {
        let tag = format!("qwen-prefill-l{layer:03}-{label}");
        enc.set_label(&tag);
        enc.insert_debug_signpost(&tag);
    }
}

fn cosine_f32_slices(a: &[f32], b: &[f32]) -> f64 {
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

fn encode_moe_route_logits_dispatch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), crate::metal_forward::MfError> {
    let enabled = match prefill_moe_route_logits_e8p32_mode() {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff => false,
        PrefillEnvMode::Auto => {
            let min_query = if n_in <= 2048 { 128 } else { 512 };
            n_query >= min_query
        }
    };
    if enabled
        && weight.dtype == GgmlType::F32
        && n_in.is_multiple_of(4)
        && n_out.is_multiple_of(8)
        && n_query >= 32
    {
        Ok(encode_mat_mat_f32_router_e8p32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?)
    } else {
        encode_mat_mat_dispatch(ctx, enc, weight, x, y, n_in, n_out, n_query)
    }
}

#[derive(Clone, Copy, Debug)]
struct CpuExpertGroupRange {
    expert: usize,
    start: usize,
    len: usize,
}

fn cpu_read_i32_f32buf(t: &MetalTensor) -> Vec<i32> {
    assert_eq!(t.dtype, GgmlType::F32);
    let n = t.n_elements() as usize;
    let mut out = vec![0i32; n];
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const i32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }
    out
}

fn cpu_read_f32buf(t: &MetalTensor) -> Vec<f32> {
    assert_eq!(t.dtype, GgmlType::F32);
    let n = t.n_elements() as usize;
    let mut out = vec![0.0f32; n];
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }
    out
}

fn cpu_write_i32_f32buf(t: &MetalTensor, data: &[i32]) {
    assert_eq!(t.dtype, GgmlType::F32);
    assert_eq!(t.n_elements() as usize, data.len());
    unsafe {
        let dst = (t.buffer.contents().as_ptr() as *mut i32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }
}

fn cpu_write_i32buf(t: &MetalTensor, data: &[i32]) {
    assert_eq!(t.dtype, GgmlType::I32);
    assert_eq!(t.n_elements() as usize, data.len());
    unsafe {
        let dst = (t.buffer.contents().as_ptr() as *mut i32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }
}

fn cpu_write_f32buf(t: &MetalTensor, data: &[f32]) {
    assert_eq!(t.dtype, GgmlType::F32);
    assert_eq!(t.n_elements() as usize, data.len());
    unsafe {
        let dst = (t.buffer.contents().as_ptr() as *mut f32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }
}

fn build_expert_slot_groups_cpu(
    topk_idx: &[i32],
    topk_weight: &[f32],
    topk: usize,
    n_expert: usize,
) -> (Vec<CpuExpertGroupRange>, Vec<i32>, Vec<i32>, Vec<f32>) {
    let mut buckets: Vec<Vec<(i32, i32, f32)>> = vec![Vec::new(); n_expert];
    for (slot, &expert_i) in topk_idx.iter().enumerate() {
        if expert_i < 0 {
            continue;
        }
        let expert = expert_i as usize;
        if expert >= n_expert {
            continue;
        }
        let token = (slot / topk) as i32;
        buckets[expert].push((slot as i32, token, topk_weight[slot]));
    }

    let mut ranges = Vec::new();
    let mut slot_ids = Vec::with_capacity(topk_idx.len());
    let mut token_ids = Vec::with_capacity(topk_idx.len());
    let mut weights = Vec::with_capacity(topk_idx.len());
    for (expert, bucket) in buckets.into_iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        let start = slot_ids.len();
        for (slot_id, token_id, weight) in bucket {
            slot_ids.push(slot_id);
            token_ids.push(token_id);
            weights.push(weight);
        }
        ranges.push(CpuExpertGroupRange {
            expert,
            start,
            len: slot_ids.len() - start,
        });
    }
    (ranges, slot_ids, token_ids, weights)
}

crate::env_flag!(default_off prefill_noop_gdn_body_enabled, "QWEN_PREFILL_NOOP_GDN_BODY");

crate::env_flag!(default_off prefill_noop_attn_body_enabled, "QWEN_PREFILL_NOOP_ATTN_BODY");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillGdnSplitMode {
    None,
    SkipAll,
    OutOnly,
    PrepOut,
    PrepStepOut,
}

#[derive(Clone, Copy)]
enum PrefillTailMode<'a> {
    ReadLogits,
    SkipTail,
    Supplied(LmHeadTail<'a>),
}

impl PrefillGdnSplitMode {
    fn from_env() -> Self {
        if prefill_noop_gdn_body_enabled() {
            return Self::SkipAll;
        }
        match std::env::var("QWEN_PREFILL_GDN_SPLIT").as_deref() {
            Ok("skip_all") => Self::SkipAll,
            Ok("out_only") => Self::OutOnly,
            Ok("prep_out") => Self::PrepOut,
            Ok("prep_step_out") => Self::PrepStepOut,
            _ => Self::None,
        }
    }

    fn run_prep(self) -> bool {
        matches!(self, Self::None | Self::PrepOut | Self::PrepStepOut)
    }

    fn run_step(self) -> bool {
        matches!(self, Self::None | Self::PrepStepOut)
    }

    fn run_gated(self) -> bool {
        matches!(self, Self::None)
    }

    fn needs_zero_normed(self) -> bool {
        matches!(self, Self::OutOnly | Self::PrepOut | Self::PrepStepOut)
    }
}

fn prefill_gdn_split_mode() -> PrefillGdnSplitMode {
    static MODE: OnceLock<PrefillGdnSplitMode> = OnceLock::new();
    *MODE.get_or_init(PrefillGdnSplitMode::from_env)
}

fn zero_f32_tensor(t: &MetalTensor) {
    // Always-on (was debug_assert): a release-build dtype mismatch here
    // would short-zero a non-F32 tensor without any complaint.
    assert_eq!(
        t.dtype,
        GgmlType::F32,
        "zero_f32_tensor on non-F32 tensor: dtype={:?}",
        t.dtype
    );
    unsafe {
        let ptr = (t.buffer.contents().as_ptr() as *mut u8).add(t.offset as usize);
        std::ptr::write_bytes(ptr, 0, t.n_bytes() as usize);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DFlashError {
    #[error("metal: {0}")]
    Metal(#[from] MetalError),
    #[error("metal forward: {0}")]
    MetalForward(#[from] crate::metal_forward::MfError),
    #[error("codec: {0}")]
    Codec(#[from] crate::codec::CodecError),
    #[error("proposal sampling: {0}")]
    Sampling(#[from] SamplingError),
    #[error("token {0} out of vocab range {1}")]
    BadToken(i32, u32),
    #[error("ctx_len {0} > capacity {1}")]
    CtxOverflow(usize, usize),
    #[error("dflash2 drafter: {0}")]
    BadDrafter(&'static str),
    #[error("dflash2 selector diagnostic is disabled by QWEN_DFLASH2_NO_SELECTOR")]
    SelectorDiagnosticDisabled,
    #[error(
        "dflash2 selector diagnostic mismatch at depth {depth}: production={production}, replay={reconstructed}"
    )]
    SelectorDiagnosticMismatch {
        depth: usize,
        production: i32,
        reconstructed: i32,
    },
    #[cfg(feature = "dflash-k0s-diagnostics")]
    #[error("dflash K0-S diagnostic: {0}")]
    K0sDiagnostic(String),
    #[cfg(feature = "dflash-k0s-diagnostics")]
    #[error(
        "dflash K0-S observation event {event_sequence} is still live; extract or finish it before another draft"
    )]
    K0sObservationLive { event_sequence: u64 },
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
    /// DFlash 2 path-selector state. `Some` iff the drafter GGUF carries
    /// selector tensors (`config.selector_top_k > 0`).
    pub selector: Option<MetalDFlash2Selector>,
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
    /// DFlash 2 two-tap dynamic conv weights (GPU-resident). `None` for
    /// DFlash 1 drafters.
    pub conv: Option<MetalDFlash2Conv>,
}

/// DFlash 2 per-layer conv weights on Metal. Base kernels are tiny F32
/// (`[H, kernel, 2]`); projections stay native (Q8_0) and route through
/// `encode_mat_mat_dispatch` like every other drafter projection.
pub struct MetalDFlash2Conv {
    pub attn_base: MetalTensor,
    pub attn_proj: MetalTensor,
    pub ffn_base: MetalTensor,
    pub ffn_proj: MetalTensor,
}

/// DFlash 2 selector state. `hidden` (the context-gate projection
/// `[H, rank]`) lives on the GPU — it runs as one small mat-mat in the
/// phase-4 tail. The two token-embedding codebooks (`[rank, V]`, ~64 MB
/// each at Q8_0) stay on the CPU: the greedy path walk only ever touches
/// ~16 rows per draft position, so per-row dequant beats a 254 MB F32
/// upload or a GPU gather kernel.
pub struct MetalDFlash2Selector {
    pub hidden: MetalTensor,
    pub predecessor: DFlash2Codebook,
    pub successor: DFlash2Codebook,
    pub rank: usize,
    pub top_k: usize,
    #[cfg(feature = "dflash-k0s-diagnostics")]
    hidden_provenance: DFlashK0sTensorProvenance,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
pub const DFLASH_K0S_BLOCK_SIZE: usize = 8;
#[cfg(feature = "dflash-k0s-diagnostics")]
pub const DFLASH_K0S_TOP_K: usize = 16;
#[cfg(feature = "dflash-k0s-diagnostics")]
pub const DFLASH_K0S_RANK: usize = 256;
#[cfg(feature = "dflash-k0s-diagnostics")]
pub const DFLASH_K0S_HIDDEN: usize = 5120;
#[cfg(feature = "dflash-k0s-diagnostics")]
pub const DFLASH_K0S_VOCAB: usize = 248_320;
#[cfg(feature = "dflash-k0s-diagnostics")]
pub const DFLASH_K0S_LATTICE_ROWS: usize = 97;
#[cfg(feature = "dflash-k0s-diagnostics")]
pub const DFLASH_K0S_DISPATCH_CENSUS_MAX: usize = 256;
#[cfg(feature = "dflash-k0s-diagnostics")]
const DFLASH_K0S_SELECTOR_DISPATCH_TAG: &str = "dflash_k0s.selector_hidden_projection.v1";

#[cfg(feature = "dflash-k0s-diagnostics")]
pub fn dflash_k0s_embedded_metallib_bytes() -> &'static [u8] {
    crate::KERNELS_METALLIB
}

#[cfg(feature = "dflash-k0s-diagnostics")]
pub fn dflash_k0s_embedded_metallib_identity() -> DFlashK0sEmbeddedMetallibIdentity {
    DFlashK0sEmbeddedMetallibIdentity {
        byte_count: crate::KERNELS_METALLIB.len(),
        sha256: Sha256::digest(crate::KERNELS_METALLIB).into(),
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug)]
pub struct DFlashK0sTensorProvenance {
    pub descriptor: TensorDesc,
    pub full_tensor_sha256: [u8; 32],
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DFlashK0sCodebookSide {
    Predecessor,
    Successor,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug)]
pub struct DFlashK0sRawRow {
    pub side: DFlashK0sCodebookSide,
    pub depth: usize,
    pub predecessor_slot: Option<usize>,
    pub candidate_slot: Option<usize>,
    pub token_id: i32,
    pub bytes: Vec<u8>,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DFlashK0sNonFiniteClass {
    PositiveInfinity,
    NegativeInfinity,
    Nan,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DFlashK0sSlotIssue {
    DuplicateId {
        candidate_slot: usize,
        first_slot: usize,
        token_id: i32,
    },
    Sentinel {
        candidate_slot: usize,
        token_id: i32,
    },
    NonFiniteScore {
        candidate_slot: usize,
        token_id: i32,
        score_bits: u32,
        class: DFlashK0sNonFiniteClass,
    },
    NoValidChoice,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug)]
pub struct DFlashK0sSlot {
    pub candidate_slot: usize,
    pub token_id: i32,
    pub unary_bits: u32,
    pub score_bits: Option<u32>,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug)]
pub struct DFlashK0sLatticeRow {
    /// Global positional row index in the canonical 97-row packet, `0..=96`.
    pub row_index: usize,
    pub depth: usize,
    pub predecessor_slot: Option<usize>,
    pub predecessor_token: i32,
    pub slots: Vec<DFlashK0sSlot>,
    pub issues: Vec<DFlashK0sSlotIssue>,
    pub greedy_slot: usize,
    pub has_valid_choice: bool,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DFlashK0sTopKIssue {
    NonFiniteLogit {
        token_id: usize,
        bits: u32,
    },
    IdMismatch {
        slot: usize,
        expected: i32,
        observed: i32,
    },
    UnaryMismatch {
        slot: usize,
        expected_bits: u32,
        observed_bits: u32,
    },
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug)]
pub struct DFlashK0sDepth {
    pub depth: usize,
    pub position: u32,
    pub full_logits_bits: Vec<u32>,
    pub top_k_ids: Vec<i32>,
    pub unary_bits: Vec<u32>,
    pub selector_hidden_bits: Vec<u32>,
    pub top_k_issues: Vec<DFlashK0sTopKIssue>,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DFlashK0sChainEvent {
    InvalidCarry {
        depth: usize,
        token_id: i32,
    },
    MissingPredecessorRow {
        depth: usize,
        token_id: i32,
        predecessor_slot: Option<usize>,
    },
    SlotZeroTermination {
        depth: usize,
        token_id: i32,
        slot: usize,
    },
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug)]
pub struct DFlashK0sChain {
    pub requested_slots: Vec<usize>,
    pub visited_row_indices: Vec<usize>,
    pub tokens: Vec<i32>,
    pub event: Option<DFlashK0sChainEvent>,
    pub terminated: bool,
    pub packet_geometry_valid: bool,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DFlashK0sDispatchCensusRow {
    pub family: String,
    pub tag: Option<String>,
    pub encoder_ordinal: u64,
    pub encoder_concurrent: bool,
    pub kernel: String,
    pub grid: [u64; 3],
    pub threads: [u64; 3],
    pub grid_threadgroups: u64,
    pub threadgroup_threads: u64,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug)]
pub struct DFlashK0sStateIdentity {
    pub carry_token: i32,
    pub noise_start_position: u32,
    pub target_context_len: usize,
    pub context_hidden_watermark: usize,
    pub kv_context_watermark: usize,
    pub draft_tokens_sha256: [u8; 32],
    pub noise_input_sha256: [u8; 32],
    pub synchronized_event_sha256: [u8; 32],
    pub diagnostic_state_sha256: [u8; 32],
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DFlashK0sScalarContractCase {
    pub name: &'static str,
    pub a_bits: Vec<u32>,
    pub z_bits: Vec<u32>,
    pub successor_bits: Vec<u32>,
    pub unary_bits: u32,
    pub score_bits: u32,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DFlashK0sScalarContractFixture {
    pub cases: Vec<DFlashK0sScalarContractCase>,
    pub fixture_sha256: [u8; 32],
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug)]
pub struct DFlashK0sProvenance {
    pub selector_hidden: DFlashK0sTensorProvenance,
    pub predecessor: DFlashK0sTensorProvenance,
    pub successor: DFlashK0sTensorProvenance,
    pub embedded_metallib_sha256: [u8; 32],
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug)]
pub struct DFlashK0sCapture {
    pub draft_tokens: Vec<i32>,
    pub draft_token_bits: Vec<u32>,
    pub depths: Vec<DFlashK0sDepth>,
    pub lattice: Vec<DFlashK0sLatticeRow>,
    pub production_chain: DFlashK0sChain,
    pub raw_rows: Vec<DFlashK0sRawRow>,
    pub provenance: DFlashK0sProvenance,
    pub dispatch_census: Vec<DFlashK0sDispatchCensusRow>,
    pub selector_hidden_dispatch: DFlashK0sDispatchCensusRow,
    pub kernel_trace: crate::metal::KernelTraceCounters,
    pub state_identity: DFlashK0sStateIdentity,
    pub capture_sha256: [u8; 32],
    pub content_sha256: [u8; 32],
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DFlashK0sSelectorInputIdentity {
    pub full_logits_count: usize,
    pub full_logits_sha256_f32le: [u8; 32],
    pub top_k_ids_count: usize,
    pub top_k_ids_sha256_i32le: [u8; 32],
    pub unary_count: usize,
    pub unary_sha256_f32le: [u8; 32],
    pub selector_hidden_count: usize,
    pub selector_hidden_sha256_f32le: [u8; 32],
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DFlashK0sSelectorDispatchIdentity {
    pub weight_dtype: GgmlType,
    pub input_dtype: GgmlType,
    pub output_dtype: GgmlType,
    pub block_size: usize,
    pub hidden_size: usize,
    pub selector_rank: usize,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DFlashK0sEmbeddedMetallibIdentity {
    pub byte_count: usize,
    pub sha256: [u8; 32],
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DFlashK0sParitySummary {
    pub draft_tokens: Vec<i32>,
    pub dispatch_census: Vec<DFlashK0sDispatchCensusRow>,
    pub kernel_trace: [u64; 3],
    pub selector_inputs: DFlashK0sSelectorInputIdentity,
    pub selector_dispatch: DFlashK0sSelectorDispatchIdentity,
    pub diagnostic_state_sha256: [u8; 32],
    pub carry_token: i32,
    pub noise_start_position: u32,
    pub session_binding_sha256: [u8; 32],
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DFlashK0sObservationSummary {
    pub parity: DFlashK0sParitySummary,
    pub event_sequence: u64,
    pub event_envelope_sha256: [u8; 32],
}

#[cfg(feature = "dflash-k0s-diagnostics")]
pub struct DFlashK0sProductionObservation {
    summary: DFlashK0sObservationSummary,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
impl DFlashK0sProductionObservation {
    pub fn summary(&self) -> &DFlashK0sObservationSummary {
        &self.summary
    }
}

/// Read-only evidence from one DFlash 2 greedy selector walk.
#[derive(Clone, Debug)]
pub struct DFlash2SelectorDiagnostic {
    /// Tokens returned by the production `draft_block` call.
    pub draft_tokens: Vec<i32>,
    /// One record for each causal selector row, `1..block_size`.
    pub depths: Vec<DFlash2SelectorDepthDiagnostic>,
}

/// One sampled DFlash 2 path plus the realized sparse proposal distribution
/// for each draft token. `draft_tokens[0]` is the unused anchor slot, while
/// `proposals[i]` generated `draft_tokens[i + 1]`.
#[derive(Clone, Debug)]
pub struct DFlash2SparseProposalBlock {
    pub draft_tokens: Vec<i32>,
    pub proposals: Vec<SparseProposal>,
}

/// Exact scalar inputs and outputs for one causal depth of the selector.
#[derive(Clone, Debug)]
pub struct DFlash2SelectorDepthDiagnostic {
    pub depth: usize,
    pub predecessor_token: Option<i32>,
    /// Top-k slot that selected the predecessor at the preceding depth;
    /// `None` at depth 1, where the predecessor is the carry token.
    pub predecessor_choice_index: Option<usize>,
    pub top_k_ids: Vec<i32>,
    pub unary_logits: Vec<f32>,
    /// Predecessor-conditioned `unary + dot` scores. Sentinel IDs have no
    /// score; non-finite computed scores are retained exactly.
    pub final_scores: Vec<Option<f32>>,
    pub greedy_score: Option<f32>,
    pub greedy_index: Option<usize>,
    pub greedy_token: Option<i32>,
    pub issues: Vec<DFlash2SelectorIssue>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DFlash2SelectorIssue {
    Sentinel {
        candidate_index: usize,
        token_id: i32,
    },
    DuplicateId {
        candidate_index: usize,
        first_index: usize,
        token_id: i32,
    },
    NonFiniteScore {
        candidate_index: usize,
        token_id: i32,
        score: f32,
    },
    NoValidChoice,
}

/// CPU-side row-dequantizable copy of a selector codebook tensor
/// (`[rank, n_rows]` GGUF layout — one `rank`-wide row per token id).
pub struct DFlash2Codebook {
    /// Four-byte-aligned owned storage. Q4_K's reference dequantizer reads
    /// typed blocks, and every validated Q4_K row starts at a 4-byte boundary.
    raw_words: Vec<u32>,
    raw_len: usize,
    dtype: GgmlType,
    rank: usize,
    n_rows: usize,
    row_bytes: usize,
    row_desc: TensorDesc,
    #[cfg(feature = "dflash-k0s-diagnostics")]
    original_desc: TensorDesc,
    #[cfg(feature = "dflash-k0s-diagnostics")]
    full_tensor_sha256: [u8; 32],
}

impl DFlash2Codebook {
    fn from_gguf(desc: &TensorDesc, bytes: &[u8]) -> Result<Self, DFlashError> {
        let &[rank, n_rows] = desc.shape.as_slice() else {
            return Err(DFlashError::BadDrafter("selector codebook must be 2-D"));
        };
        if rank == 0 || n_rows == 0 {
            return Err(DFlashError::BadDrafter(
                "selector codebook dimensions must be nonzero",
            ));
        }
        let rank = usize::try_from(rank)
            .map_err(|_| DFlashError::BadDrafter("selector codebook rank overflows usize"))?;
        let n_rows = usize::try_from(n_rows)
            .map_err(|_| DFlashError::BadDrafter("selector codebook row count overflows usize"))?;
        match desc.dtype {
            GgmlType::Q4_K | GgmlType::Q8_0 | GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {}
            _ => {
                return Err(DFlashError::BadDrafter(
                    "selector codebook dtype must be Q4_K/Q8_0/F32/F16/BF16",
                ));
            }
        }

        let (block_elements, block_bytes) = desc.dtype.storage_layout().ok_or(
            DFlashError::BadDrafter("selector codebook dtype has no storage layout"),
        )?;
        let rank_u64 = u64::try_from(rank)
            .map_err(|_| DFlashError::BadDrafter("selector codebook rank overflows u64"))?;
        if !rank_u64.is_multiple_of(block_elements) {
            return Err(DFlashError::BadDrafter(
                "selector codebook rank is not block-aligned",
            ));
        }
        if desc.dtype == GgmlType::Q4_K && rank != 256 {
            return Err(DFlashError::BadDrafter(
                "Q4_K selector codebook rank must be 256",
            ));
        }
        let row_bytes_u64 = rank_u64
            .checked_div(block_elements)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(DFlashError::BadDrafter(
                "selector codebook row size overflows",
            ))?;
        let expected_u64 = row_bytes_u64
            .checked_mul(u64::try_from(n_rows).map_err(|_| {
                DFlashError::BadDrafter("selector codebook row count overflows u64")
            })?)
            .ok_or(DFlashError::BadDrafter(
                "selector codebook payload size overflows",
            ))?;
        let row_bytes = usize::try_from(row_bytes_u64)
            .map_err(|_| DFlashError::BadDrafter("selector codebook row size overflows usize"))?;
        let expected = usize::try_from(expected_u64).map_err(|_| {
            DFlashError::BadDrafter("selector codebook payload size overflows usize")
        })?;
        if bytes.len() != expected {
            return Err(DFlashError::BadDrafter(
                "selector codebook payload size mismatch",
            ));
        }
        if desc.dtype == GgmlType::Q4_K && !row_bytes.is_multiple_of(4) {
            return Err(DFlashError::BadDrafter(
                "Q4_K selector codebook rows must be 4-byte aligned",
            ));
        }

        let word_count = bytes.len().checked_add(3).ok_or(DFlashError::BadDrafter(
            "selector codebook aligned storage size overflows",
        ))? / 4;
        let mut raw_words = vec![0u32; word_count];
        bytemuck::cast_slice_mut::<u32, u8>(&mut raw_words)[..bytes.len()].copy_from_slice(bytes);
        let row_desc = TensorDesc {
            name: format!("{}.selector_row", desc.name),
            shape: vec![rank_u64],
            dtype: desc.dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: row_bytes_u64,
        };
        Ok(Self {
            raw_words,
            raw_len: bytes.len(),
            dtype: desc.dtype,
            rank,
            n_rows,
            row_bytes,
            row_desc,
            #[cfg(feature = "dflash-k0s-diagnostics")]
            original_desc: desc.clone(),
            #[cfg(feature = "dflash-k0s-diagnostics")]
            full_tensor_sha256: Sha256::digest(bytes).into(),
        })
    }

    /// Dequantize one token's embedding row into `out` (`len == rank`).
    fn dequant_row(&self, row: usize, out: &mut [f32]) -> Result<(), DFlashError> {
        if out.len() != self.rank {
            return Err(DFlashError::BadDrafter(
                "selector codebook output rank mismatch",
            ));
        }
        if row >= self.n_rows {
            return Err(DFlashError::BadDrafter(
                "selector codebook row out of range",
            ));
        }
        let base = row
            .checked_mul(self.row_bytes)
            .ok_or(DFlashError::BadDrafter(
                "selector codebook row offset overflows",
            ))?;
        let end = base
            .checked_add(self.row_bytes)
            .ok_or(DFlashError::BadDrafter(
                "selector codebook row end overflows",
            ))?;
        let raw = &bytemuck::cast_slice::<u32, u8>(&self.raw_words)[..self.raw_len];
        let row_bytes = raw.get(base..end).ok_or(DFlashError::BadDrafter(
            "selector codebook row exceeds payload",
        ))?;
        match self.dtype {
            GgmlType::Q4_K => {
                dequant_to_f32_in_place(&self.row_desc, row_bytes, out)?;
            }
            GgmlType::Q8_0 => {
                // block_q8_0: f16 scale + 32 * i8, 34 bytes / 32 elems.
                let blocks_per_row = self.rank / 32;
                for b in 0..blocks_per_row {
                    let blk = &row_bytes[b * 34..b * 34 + 34];
                    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    for (j, &q) in blk[2..34].iter().enumerate() {
                        out[b * 32 + j] = d * (q as i8) as f32;
                    }
                }
            }
            GgmlType::F32 => {
                for (j, slot) in out.iter_mut().enumerate() {
                    let o = j * 4;
                    *slot = f32::from_le_bytes([
                        row_bytes[o],
                        row_bytes[o + 1],
                        row_bytes[o + 2],
                        row_bytes[o + 3],
                    ]);
                }
            }
            GgmlType::F16 => {
                for (j, slot) in out.iter_mut().enumerate() {
                    let o = j * 2;
                    *slot = half::f16::from_le_bytes([row_bytes[o], row_bytes[o + 1]]).to_f32();
                }
            }
            GgmlType::BF16 => {
                for (j, slot) in out.iter_mut().enumerate() {
                    let o = j * 2;
                    *slot = f32::from_bits(
                        (u16::from_le_bytes([row_bytes[o], row_bytes[o + 1]]) as u32) << 16,
                    );
                }
            }
            _ => unreachable!("dtype validated in from_gguf"),
        }
        Ok(())
    }

    #[cfg(feature = "dflash-k0s-diagnostics")]
    fn raw_row(&self, row: usize) -> Result<Vec<u8>, DFlashError> {
        let base = row
            .checked_mul(self.row_bytes)
            .ok_or_else(|| DFlashError::K0sDiagnostic("codebook raw-row offset overflow".into()))?;
        let end = base
            .checked_add(self.row_bytes)
            .ok_or_else(|| DFlashError::K0sDiagnostic("codebook raw-row end overflow".into()))?;
        let raw = &bytemuck::cast_slice::<u32, u8>(&self.raw_words)[..self.raw_len];
        raw.get(base..end)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| DFlashError::K0sDiagnostic("codebook raw row is out of range".into()))
    }
}

fn checked_selector_read_layout(
    dtype: GgmlType,
    expected_dtype: GgmlType,
    elements: u64,
    expected_elements: usize,
    offset: u64,
    backing_len: usize,
    element_size: usize,
    element_align: usize,
    label: &'static str,
) -> Result<(usize, usize), DFlashError> {
    let bad_shape = |detail: String| {
        DFlashError::Metal(MetalError::BadShape {
            kernel: "dflash2_selector_diagnostic_read",
            detail: format!("{label}: {detail}"),
        })
    };
    if dtype != expected_dtype {
        return Err(bad_shape(format!(
            "expected dtype {expected_dtype:?}, got {dtype:?}"
        )));
    }
    let expected_elements_u64 = u64::try_from(expected_elements)
        .map_err(|_| bad_shape("expected element count does not fit u64".into()))?;
    if elements != expected_elements_u64 {
        return Err(bad_shape(format!(
            "expected {expected_elements} elements, got {elements}"
        )));
    }
    let byte_len = expected_elements
        .checked_mul(element_size)
        .ok_or_else(|| bad_shape("byte length overflow".into()))?;
    let offset = usize::try_from(offset)
        .map_err(|_| bad_shape("tensor offset does not fit usize".into()))?;
    if element_align == 0 || !offset.is_multiple_of(element_align) {
        return Err(bad_shape(format!(
            "offset {offset} is not aligned to {element_align} bytes"
        )));
    }
    let end = offset
        .checked_add(byte_len)
        .ok_or_else(|| bad_shape("offset + byte length overflow".into()))?;
    if end > backing_len {
        return Err(bad_shape(format!(
            "byte range {offset}..{end} exceeds backing buffer length {backing_len}"
        )));
    }
    Ok((offset, byte_len))
}

fn read_shared_selector_tensor<T: Copy>(
    tensor: &MetalTensor,
    expected_dtype: GgmlType,
    expected_elements: usize,
    label: &'static str,
) -> Result<Vec<T>, DFlashError> {
    if tensor.buffer.storageMode() != MTLStorageMode::Shared {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "dflash2_selector_diagnostic_read",
            detail: format!("{label}: backing buffer is not StorageModeShared"),
        }));
    }
    let (offset, _) = checked_selector_read_layout(
        tensor.dtype,
        expected_dtype,
        tensor.n_elements(),
        expected_elements,
        tensor.offset,
        tensor.buffer.length(),
        std::mem::size_of::<T>(),
        std::mem::align_of::<T>(),
        label,
    )?;
    let mut out = Vec::<T>::with_capacity(expected_elements);
    unsafe {
        let src = (tensor.buffer.contents().as_ptr() as *const u8)
            .add(offset)
            .cast::<T>();
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), expected_elements);
        out.set_len(expected_elements);
    }
    Ok(out)
}

fn verify_dflash2_selector_replay(
    production: &[i32],
    reconstructed: &[i32],
) -> Result<(), DFlashError> {
    if production.len() != reconstructed.len() {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "dflash2_selector_diagnostic_replay",
            detail: format!(
                "production token count {} != replay token count {}",
                production.len(),
                reconstructed.len()
            ),
        }));
    }
    if let Some((depth, (&production, &reconstructed))) = production
        .iter()
        .zip(reconstructed)
        .enumerate()
        .find(|(_, (production, reconstructed))| production != reconstructed)
    {
        return Err(DFlashError::SelectorDiagnosticMismatch {
            depth,
            production,
            reconstructed,
        });
    }
    Ok(())
}

fn diagnose_dflash2_selector_walk(
    predecessor: &DFlash2Codebook,
    successor: &DFlash2Codebook,
    top_k: usize,
    rank: usize,
    carry_tok: i32,
    n: usize,
    ids: &[i32],
    vals: &[f32],
    sel_h: &[f32],
) -> Result<(Vec<i32>, Vec<DFlash2SelectorDepthDiagnostic>), DFlashError> {
    let bad_replay_shape = |detail: String| {
        DFlashError::Metal(MetalError::BadShape {
            kernel: "dflash2_selector_diagnostic_replay",
            detail,
        })
    };
    if n == 0 || top_k == 0 {
        return Err(bad_replay_shape(
            "block size and selector top-k must be nonzero".into(),
        ));
    }
    let candidate_elements = n
        .checked_mul(top_k)
        .ok_or_else(|| bad_replay_shape("candidate element count overflow".into()))?;
    let gate_elements = n
        .checked_mul(rank)
        .ok_or_else(|| bad_replay_shape("gate element count overflow".into()))?;
    if ids.len() != candidate_elements
        || vals.len() != candidate_elements
        || sel_h.len() != gate_elements
    {
        return Err(bad_replay_shape(format!(
            "expected ids/vals/gate lengths {candidate_elements}/{candidate_elements}/{gate_elements}, got {}/{}/{}",
            ids.len(),
            vals.len(),
            sel_h.len()
        )));
    }

    let mut reconstructed = vec![ids[0]; n];
    let mut depths = Vec::with_capacity(n.saturating_sub(1));
    let mut predecessor_token = Some(carry_tok);
    let mut predecessor_choice_index = None;
    let mut pred = vec![0f32; rank];
    let mut succ = vec![0f32; rank];
    let mut gate = vec![0f32; rank];
    predecessor.dequant_row(carry_tok as usize, &mut pred)?;

    for depth in 1..n {
        let row_ids = ids[depth * top_k..(depth + 1) * top_k].to_vec();
        let unary_logits = vals[depth * top_k..(depth + 1) * top_k].to_vec();
        let mut final_scores = vec![None; top_k];
        let mut issues = Vec::new();
        let mut best_index = None;
        let mut best_score = f32::NEG_INFINITY;

        if predecessor_token.is_some() {
            let hrow = &sel_h[depth * rank..(depth + 1) * rank];
            for r in 0..rank {
                gate[r] = pred[r] * hrow[r];
            }
        }
        for candidate_index in 0..top_k {
            let token_id = row_ids[candidate_index];
            if let Some(first_index) = row_ids[..candidate_index]
                .iter()
                .position(|&prior| prior == token_id)
            {
                issues.push(DFlash2SelectorIssue::DuplicateId {
                    candidate_index,
                    first_index,
                    token_id,
                });
            }
            if token_id < 0 || token_id as usize >= successor.n_rows {
                issues.push(DFlash2SelectorIssue::Sentinel {
                    candidate_index,
                    token_id,
                });
                continue;
            }
            if predecessor_token.is_some() {
                successor.dequant_row(token_id as usize, &mut succ)?;
                let mut dot = 0f32;
                for r in 0..rank {
                    dot += gate[r] * succ[r];
                }
                let score = unary_logits[candidate_index] + dot;
                final_scores[candidate_index] = Some(score);
                if !score.is_finite() {
                    issues.push(DFlash2SelectorIssue::NonFiniteScore {
                        candidate_index,
                        token_id,
                        score,
                    });
                }
                // Match the production selector exactly: strict comparison
                // preserves top-k order on ties and naturally rejects NaN.
                if score > best_score {
                    best_score = score;
                    best_index = Some(candidate_index);
                }
            }
        }

        let no_valid_choice = best_index.is_none();
        if no_valid_choice {
            issues.push(DFlash2SelectorIssue::NoValidChoice);
        }
        // Production initializes its choice index to zero. Preserve that
        // fallback in the reconstructed tokens while explicitly diagnosing
        // that no score won the strict comparison.
        let greedy_index = if top_k == 0 {
            None
        } else {
            Some(best_index.unwrap_or(0))
        };
        let greedy_token = greedy_index.map(|index| row_ids[index]);
        depths.push(DFlash2SelectorDepthDiagnostic {
            depth,
            predecessor_token,
            predecessor_choice_index,
            top_k_ids: row_ids,
            unary_logits,
            final_scores,
            greedy_score: greedy_index.map(|_| best_score),
            greedy_index,
            greedy_token,
            issues,
        });

        let index = greedy_index.expect("top_k checked nonzero");
        let token = greedy_token.expect("top_k checked nonzero");
        reconstructed[depth] = token;
        // Production performs this dequant unconditionally, including its
        // slot-zero fallback when no score wins. Propagate the same failure
        // rather than manufacturing records for unreachable later depths.
        predecessor.dequant_row(token as usize, &mut pred)?;
        predecessor_token = Some(token);
        predecessor_choice_index = Some(index);
    }

    Ok((reconstructed, depths))
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_error(detail: impl Into<String>) -> DFlashError {
    DFlashError::K0sDiagnostic(detail.into())
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_require_no_live_event(live_event: Option<u64>) -> Result<(), DFlashError> {
    match live_event {
        Some(event_sequence) => Err(DFlashError::K0sObservationLive { event_sequence }),
        None => Ok(()),
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_check_dispatch_census_len(rows: usize) -> Result<(), DFlashError> {
    if rows > DFLASH_K0S_DISPATCH_CENSUS_MAX {
        return Err(dflash_k0s_error(format!(
            "dispatch census has {rows} rows, exceeding the v1 cap of {}",
            DFLASH_K0S_DISPATCH_CENSUS_MAX
        )));
    }
    Ok(())
}

#[cfg(feature = "dflash-k0s-diagnostics")]
struct DFlashK0sObserverGuard {
    baseline: [usize; 3],
    trace: Option<crate::metal::KernelTraceGuard>,
    census_active: bool,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
impl DFlashK0sObserverGuard {
    fn begin() -> Result<Self, DFlashError> {
        let baseline = crate::metal::diagnostics_observer_active_counts();
        if baseline[0] != 0 || baseline[1] != 0 || dispatch_census_is_active() {
            return Err(dflash_k0s_error(
                "dispatch census or kernel trace is already active",
            ));
        }
        dispatch_census_begin();
        let trace = kernel_trace_begin();
        Ok(Self {
            baseline,
            trace: Some(trace),
            census_active: true,
        })
    }

    fn finish(
        mut self,
    ) -> Result<
        (
            Vec<crate::metal::DispatchCensusRow>,
            crate::metal::KernelTraceCounters,
        ),
        DFlashError,
    > {
        let counters = kernel_trace_snapshot();
        drop(self.trace.take());
        let rows = dispatch_census_take();
        self.census_active = false;
        if crate::metal::diagnostics_observer_active_counts() != self.baseline {
            return Err(dflash_k0s_error(
                "diagnostic observers did not return to their baseline",
            ));
        }
        Ok((rows, counters))
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
impl Drop for DFlashK0sObserverGuard {
    fn drop(&mut self) {
        drop(self.trace.take());
        if self.census_active {
            let _ = dispatch_census_take();
            self.census_active = false;
        }
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_positions(noise_start_pos: u32) -> Result<[u32; 7], DFlashError> {
    let mut positions = [0u32; 7];
    for (offset, position) in positions.iter_mut().enumerate() {
        *position = noise_start_pos
            .checked_add((offset + 1) as u32)
            .ok_or_else(|| dflash_k0s_error("K0-S depth position overflows u32"))?;
    }
    Ok(positions)
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_nonfinite_class(value: f32) -> DFlashK0sNonFiniteClass {
    if value.is_nan() {
        DFlashK0sNonFiniteClass::Nan
    } else if value.is_sign_positive() {
        DFlashK0sNonFiniteClass::PositiveInfinity
    } else {
        DFlashK0sNonFiniteClass::NegativeInfinity
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_reconstruct_top_k(
    logits: &[f32],
    observed_ids: &[i32],
    observed_unary: &[f32],
) -> Vec<DFlashK0sTopKIssue> {
    let mut issues = Vec::new();
    for (token_id, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            issues.push(DFlashK0sTopKIssue::NonFiniteLogit {
                token_id,
                bits: value.to_bits(),
            });
        }
    }
    if !issues.is_empty() {
        return issues;
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        logits[right]
            .partial_cmp(&logits[left])
            .expect("finite logits have an ordinary ordering")
            .then_with(|| left.cmp(&right))
    });
    for (slot, &expected) in order.iter().take(DFLASH_K0S_TOP_K).enumerate() {
        if observed_ids.get(slot).copied() != Some(expected as i32) {
            issues.push(DFlashK0sTopKIssue::IdMismatch {
                slot,
                expected: expected as i32,
                observed: observed_ids.get(slot).copied().unwrap_or(-1),
            });
        }
        let expected_bits = logits[expected].to_bits();
        let observed_bits = observed_unary
            .get(slot)
            .map(|v| v.to_bits())
            .unwrap_or(u32::MAX);
        if observed_bits != expected_bits {
            issues.push(DFlashK0sTopKIssue::UnaryMismatch {
                slot,
                expected_bits,
                observed_bits,
            });
        }
    }
    issues
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_classify_slots(
    token_ids: &[i32],
    scores: &[Option<f32>],
    vocab: usize,
) -> (Vec<DFlashK0sSlotIssue>, usize, bool) {
    let mut issues = Vec::new();
    let mut best_score = f32::NEG_INFINITY;
    let mut best_slot = 0usize;
    let mut has_valid_choice = false;
    for candidate_slot in 0..token_ids.len() {
        let token_id = token_ids[candidate_slot];
        if let Some(first_slot) = token_ids[..candidate_slot]
            .iter()
            .position(|&prior| prior == token_id)
        {
            issues.push(DFlashK0sSlotIssue::DuplicateId {
                candidate_slot,
                first_slot,
                token_id,
            });
        }
        if token_id < 0 || token_id as usize >= vocab {
            issues.push(DFlashK0sSlotIssue::Sentinel {
                candidate_slot,
                token_id,
            });
        } else if let Some(score) = scores[candidate_slot] {
            if !score.is_finite() {
                issues.push(DFlashK0sSlotIssue::NonFiniteScore {
                    candidate_slot,
                    token_id,
                    score_bits: score.to_bits(),
                    class: dflash_k0s_nonfinite_class(score),
                });
            }
            if score > best_score {
                best_score = score;
                best_slot = candidate_slot;
                has_valid_choice = true;
            }
        }
    }
    if !has_valid_choice {
        issues.push(DFlashK0sSlotIssue::NoValidChoice);
    }
    (issues, best_slot, has_valid_choice)
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_scalar_score(gate: &[f32], successor: &[f32], unary: f32) -> f32 {
    let mut dot = 0.0f32;
    for rank in 0..gate.len() {
        let product = gate[rank] * successor[rank];
        dot += product;
    }
    unary + dot
}

#[cfg(feature = "dflash-k0s-diagnostics")]
trait DFlashK0sRowProvider {
    fn rank(&self) -> usize;
    fn row_count(&self) -> usize;
    fn dequant_row_for_k0s(&self, row: usize, out: &mut [f32]) -> Result<(), DFlashError>;
    fn raw_row_for_k0s(&self, row: usize) -> Result<Vec<u8>, DFlashError>;
}

#[cfg(feature = "dflash-k0s-diagnostics")]
impl DFlashK0sRowProvider for DFlash2Codebook {
    fn rank(&self) -> usize {
        self.rank
    }

    fn row_count(&self) -> usize {
        self.n_rows
    }

    fn dequant_row_for_k0s(&self, row: usize, out: &mut [f32]) -> Result<(), DFlashError> {
        self.dequant_row(row, out)
    }

    fn raw_row_for_k0s(&self, row: usize) -> Result<Vec<u8>, DFlashError> {
        self.raw_row(row)
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_build_lattice(
    predecessor: &DFlash2Codebook,
    successor: &DFlash2Codebook,
    carry_token: i32,
    ids: &[i32],
    unary: &[f32],
    selector_hidden: &[f32],
) -> Result<(Vec<DFlashK0sLatticeRow>, Vec<DFlashK0sRawRow>), DFlashError> {
    if predecessor.rank != DFLASH_K0S_RANK
        || successor.rank != DFLASH_K0S_RANK
        || predecessor.n_rows != DFLASH_K0S_VOCAB
        || successor.n_rows != DFLASH_K0S_VOCAB
    {
        return Err(dflash_k0s_error("malformed K0-S codebook geometry"));
    }
    dflash_k0s_build_lattice_core(
        predecessor,
        successor,
        carry_token,
        ids,
        unary,
        selector_hidden,
    )
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_build_lattice_core<P: DFlashK0sRowProvider, S: DFlashK0sRowProvider>(
    predecessor: &P,
    successor: &S,
    carry_token: i32,
    ids: &[i32],
    unary: &[f32],
    selector_hidden: &[f32],
) -> Result<(Vec<DFlashK0sLatticeRow>, Vec<DFlashK0sRawRow>), DFlashError> {
    let candidate_elements = DFLASH_K0S_BLOCK_SIZE
        .checked_mul(DFLASH_K0S_TOP_K)
        .ok_or_else(|| dflash_k0s_error("candidate geometry overflow"))?;
    let hidden_elements = DFLASH_K0S_BLOCK_SIZE
        .checked_mul(DFLASH_K0S_RANK)
        .ok_or_else(|| dflash_k0s_error("selector-hidden geometry overflow"))?;
    if ids.len() != candidate_elements
        || unary.len() != candidate_elements
        || selector_hidden.len() != hidden_elements
    {
        return Err(dflash_k0s_error("malformed K0-S input geometry"));
    }
    if predecessor.rank() != DFLASH_K0S_RANK
        || successor.rank() != DFLASH_K0S_RANK
        || predecessor.row_count() != DFLASH_K0S_VOCAB
        || successor.row_count() != DFLASH_K0S_VOCAB
    {
        return Err(dflash_k0s_error("malformed K0-S codebook geometry"));
    }
    if carry_token < 0 || carry_token as usize >= DFLASH_K0S_VOCAB {
        return Err(dflash_k0s_error("invalid K0-S carry token"));
    }

    let mut rows = Vec::with_capacity(DFLASH_K0S_LATTICE_ROWS);
    let mut raw_rows = Vec::new();
    let mut pred = vec![0.0f32; DFLASH_K0S_RANK];
    let mut succ = vec![0.0f32; DFLASH_K0S_RANK];
    let mut gate = vec![0.0f32; DFLASH_K0S_RANK];
    for depth in 1..DFLASH_K0S_BLOCK_SIZE {
        let predecessor_count = if depth == 1 { 1 } else { DFLASH_K0S_TOP_K };
        for predecessor_position in 0..predecessor_count {
            let predecessor_slot = (depth > 1).then_some(predecessor_position);
            let predecessor_token = if depth == 1 {
                carry_token
            } else {
                ids[(depth - 1) * DFLASH_K0S_TOP_K + predecessor_position]
            };
            let predecessor_valid =
                predecessor_token >= 0 && (predecessor_token as usize) < DFLASH_K0S_VOCAB;
            if predecessor_valid {
                predecessor.dequant_row_for_k0s(predecessor_token as usize, &mut pred)?;
                raw_rows.push(DFlashK0sRawRow {
                    side: DFlashK0sCodebookSide::Predecessor,
                    depth,
                    predecessor_slot,
                    candidate_slot: None,
                    token_id: predecessor_token,
                    bytes: predecessor.raw_row_for_k0s(predecessor_token as usize)?,
                });
                let z = &selector_hidden[depth * DFLASH_K0S_RANK..(depth + 1) * DFLASH_K0S_RANK];
                for rank in 0..DFLASH_K0S_RANK {
                    gate[rank] = pred[rank] * z[rank];
                }
            }

            let row_ids = &ids[depth * DFLASH_K0S_TOP_K..(depth + 1) * DFLASH_K0S_TOP_K];
            let row_unary = &unary[depth * DFLASH_K0S_TOP_K..(depth + 1) * DFLASH_K0S_TOP_K];
            let mut slots = Vec::with_capacity(DFLASH_K0S_TOP_K);
            let mut scores = Vec::with_capacity(DFLASH_K0S_TOP_K);
            for candidate_slot in 0..DFLASH_K0S_TOP_K {
                let token_id = row_ids[candidate_slot];
                let candidate_valid = token_id >= 0 && (token_id as usize) < DFLASH_K0S_VOCAB;
                let score_bits = if !candidate_valid {
                    None
                } else if predecessor_valid {
                    successor.dequant_row_for_k0s(token_id as usize, &mut succ)?;
                    raw_rows.push(DFlashK0sRawRow {
                        side: DFlashK0sCodebookSide::Successor,
                        depth,
                        predecessor_slot,
                        candidate_slot: Some(candidate_slot),
                        token_id,
                        bytes: successor.raw_row_for_k0s(token_id as usize)?,
                    });
                    let score = dflash_k0s_scalar_score(&gate, &succ, row_unary[candidate_slot]);
                    Some(score.to_bits())
                } else {
                    None
                };
                scores.push(score_bits.map(f32::from_bits));
                slots.push(DFlashK0sSlot {
                    candidate_slot,
                    token_id,
                    unary_bits: row_unary[candidate_slot].to_bits(),
                    score_bits,
                });
            }
            let (issues, best_slot, has_valid_choice) =
                dflash_k0s_classify_slots(row_ids, &scores, DFLASH_K0S_VOCAB);
            rows.push(DFlashK0sLatticeRow {
                row_index: rows.len(),
                depth,
                predecessor_slot,
                predecessor_token,
                slots,
                issues,
                greedy_slot: best_slot,
                has_valid_choice,
            });
        }
    }
    if rows.len() != DFLASH_K0S_LATTICE_ROWS {
        return Err(dflash_k0s_error("K0-S lattice did not produce 97 rows"));
    }
    Ok((rows, raw_rows))
}

#[cfg(feature = "dflash-k0s-diagnostics")]
pub fn dflash_k0s_traverse_slots(
    rows: &[DFlashK0sLatticeRow],
    carry_token: i32,
    requested_slots: &[usize],
) -> DFlashK0sChain {
    dflash_k0s_traverse_slots_mode(
        rows,
        carry_token,
        requested_slots,
        DFlashK0sChainMode::Fixed,
    )
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DFlashK0sChainMode {
    Fixed,
    Production,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_traverse_slots_mode(
    rows: &[DFlashK0sLatticeRow],
    carry_token: i32,
    requested_slots: &[usize],
    mode: DFlashK0sChainMode,
) -> DFlashK0sChain {
    let packet_geometry_valid = rows.len() == DFLASH_K0S_LATTICE_ROWS
        && (1..DFLASH_K0S_BLOCK_SIZE).all(|depth| {
            let expected = if depth == 1 { 1 } else { DFLASH_K0S_TOP_K };
            rows.iter().filter(|row| row.depth == depth).count() == expected
                && (0..expected).all(|position| {
                    let slot = (depth > 1).then_some(position);
                    rows.iter()
                        .filter(|row| row.depth == depth && row.predecessor_slot == slot)
                        .count()
                        == 1
                })
        });
    let mut chain = DFlashK0sChain {
        requested_slots: requested_slots.to_vec(),
        visited_row_indices: Vec::new(),
        tokens: Vec::new(),
        event: None,
        terminated: false,
        packet_geometry_valid,
    };
    if carry_token < 0 || carry_token as usize >= DFLASH_K0S_VOCAB {
        chain.event = Some(DFlashK0sChainEvent::InvalidCarry {
            depth: 1,
            token_id: carry_token,
        });
        chain.terminated = true;
        return chain;
    }
    let mut predecessor_slot = None;
    let mut predecessor_token = carry_token;
    for (offset, &slot) in requested_slots.iter().enumerate() {
        let depth = offset + 1;
        let Some(row) = rows
            .iter()
            .find(|row| row.depth == depth && row.predecessor_slot == predecessor_slot)
        else {
            chain.event = Some(DFlashK0sChainEvent::MissingPredecessorRow {
                depth,
                token_id: predecessor_token,
                predecessor_slot,
            });
            chain.terminated = true;
            return chain;
        };
        chain.visited_row_indices.push(row.row_index);
        let Some(candidate) = row.slots.get(slot) else {
            chain.event = Some(DFlashK0sChainEvent::MissingPredecessorRow {
                depth,
                token_id: predecessor_token,
                predecessor_slot,
            });
            chain.terminated = true;
            return chain;
        };
        let candidate_valid =
            candidate.token_id >= 0 && (candidate.token_id as usize) < DFLASH_K0S_VOCAB;
        if mode == DFlashK0sChainMode::Production
            && !row.has_valid_choice
            && slot == 0
            && !candidate_valid
        {
            chain.event = Some(DFlashK0sChainEvent::SlotZeroTermination {
                depth,
                token_id: candidate.token_id,
                slot,
            });
            chain.terminated = true;
            return chain;
        }
        if !candidate_valid {
            chain.terminated = true;
            return chain;
        }
        chain.tokens.push(candidate.token_id);
        predecessor_slot = Some(slot);
        predecessor_token = candidate.token_id;
    }
    chain
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_verify_production_replay(
    draft_tokens: &[i32],
    chain: &DFlashK0sChain,
) -> Result<(), DFlashError> {
    if draft_tokens.len() != DFLASH_K0S_BLOCK_SIZE
        || chain.terminated
        || chain.event.is_some()
        || chain.tokens.len() != DFLASH_K0S_BLOCK_SIZE - 1
        || chain.tokens.as_slice() != &draft_tokens[1..]
    {
        return Err(dflash_k0s_error(
            "production draft tokens differ from the complete K0-S greedy replay",
        ));
    }
    Ok(())
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_hash_bytes(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_hash_dispatch(hash: &mut Sha256, row: &DFlashK0sDispatchCensusRow) {
    dflash_k0s_hash_bytes(hash, row.family.as_bytes());
    match &row.tag {
        Some(tag) => {
            hash.update([1]);
            dflash_k0s_hash_bytes(hash, tag.as_bytes());
        }
        None => hash.update([0]),
    }
    hash.update(row.encoder_ordinal.to_le_bytes());
    hash.update([u8::from(row.encoder_concurrent)]);
    dflash_k0s_hash_bytes(hash, row.kernel.as_bytes());
    for value in row.grid.into_iter().chain(row.threads) {
        hash.update(value.to_le_bytes());
    }
    hash.update(row.grid_threadgroups.to_le_bytes());
    hash.update(row.threadgroup_threads.to_le_bytes());
}

/// Canonical K0-S capture digest. The domain is the literal ASCII string
/// `qwen.dflash_k0s.capture.v1`. Integers and f32 bit patterns are fixed-width
/// little-endian; booleans and option tags are one byte; byte strings are
/// prefixed by a little-endian u64 length. Vectors are prefixed by a u64 count
/// and retain capture order. `capture_sha256` itself is not hashed.
#[cfg(feature = "dflash-k0s-diagnostics")]
pub fn dflash_k0s_capture_sha256(capture: &DFlashK0sCapture) -> [u8; 32] {
    dflash_k0s_capture_digest(capture, true)
}

/// Internal binary material digest excluding the carry/position and synchronized
/// event-envelope digest. This is not the reducer's canonical JSON projection;
/// the CLI/reducer constructs and validates that separately.
#[cfg(feature = "dflash-k0s-diagnostics")]
pub fn dflash_k0s_capture_content_sha256(capture: &DFlashK0sCapture) -> [u8; 32] {
    dflash_k0s_capture_digest(capture, false)
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_capture_digest(capture: &DFlashK0sCapture, include_event_envelope: bool) -> [u8; 32] {
    let mut hash = Sha256::new();
    if include_event_envelope {
        hash.update(b"qwen.dflash_k0s.capture.v1");
        hash.update(capture.state_identity.carry_token.to_le_bytes());
        hash.update(capture.state_identity.noise_start_position.to_le_bytes());
    } else {
        hash.update(b"qwen.dflash_k0s.capture_content.v1");
    }
    for value in [
        capture.state_identity.target_context_len,
        capture.state_identity.context_hidden_watermark,
        capture.state_identity.kv_context_watermark,
    ] {
        hash.update((value as u64).to_le_bytes());
    }
    hash.update(capture.state_identity.draft_tokens_sha256);
    hash.update(capture.state_identity.noise_input_sha256);
    if include_event_envelope {
        hash.update(capture.state_identity.synchronized_event_sha256);
    }
    hash.update(capture.state_identity.diagnostic_state_sha256);
    hash.update((capture.draft_token_bits.len() as u64).to_le_bytes());
    for bits in &capture.draft_token_bits {
        hash.update(bits.to_le_bytes());
    }
    hash.update((capture.depths.len() as u64).to_le_bytes());
    for depth in &capture.depths {
        hash.update((depth.depth as u64).to_le_bytes());
        hash.update(depth.position.to_le_bytes());
        hash.update((depth.full_logits_bits.len() as u64).to_le_bytes());
        for bits in &depth.full_logits_bits {
            hash.update(bits.to_le_bytes());
        }
        hash.update((depth.top_k_ids.len() as u64).to_le_bytes());
        for token in &depth.top_k_ids {
            hash.update(token.to_le_bytes());
        }
        hash.update((depth.unary_bits.len() as u64).to_le_bytes());
        for bits in &depth.unary_bits {
            hash.update(bits.to_le_bytes());
        }
        hash.update((depth.selector_hidden_bits.len() as u64).to_le_bytes());
        for bits in &depth.selector_hidden_bits {
            hash.update(bits.to_le_bytes());
        }
    }
    hash.update((capture.dispatch_census.len() as u64).to_le_bytes());
    for row in &capture.dispatch_census {
        dflash_k0s_hash_dispatch(&mut hash, row);
    }
    dflash_k0s_hash_dispatch(&mut hash, &capture.selector_hidden_dispatch);
    hash.update(capture.kernel_trace.encoders.to_le_bytes());
    hash.update(capture.kernel_trace.concurrent_encoders.to_le_bytes());
    hash.update(capture.kernel_trace.dispatches.to_le_bytes());
    hash.update(capture.provenance.selector_hidden.full_tensor_sha256);
    hash.update(capture.provenance.predecessor.full_tensor_sha256);
    hash.update(capture.provenance.successor.full_tensor_sha256);
    hash.update(capture.provenance.embedded_metallib_sha256);
    hash.finalize().into()
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_hash_state_tensor_material(
    hash: &mut Sha256,
    label: &str,
    dtype: GgmlType,
    shape: &[u64],
    bytes: &[u8],
) {
    dflash_k0s_hash_bytes(hash, label.as_bytes());
    hash.update((dtype as u32).to_le_bytes());
    hash.update((shape.len() as u64).to_le_bytes());
    for dimension in shape {
        hash.update(dimension.to_le_bytes());
    }
    let elements = shape
        .iter()
        .try_fold(1u64, |product, dimension| product.checked_mul(*dimension))
        .unwrap_or(u64::MAX);
    hash.update(elements.to_le_bytes());
    dflash_k0s_hash_bytes(hash, bytes);
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_hash_state_tensor(
    hash: &mut Sha256,
    label: &str,
    tensor: &MetalTensor,
) -> Result<(), DFlashError> {
    if tensor.buffer.storageMode() != MTLStorageMode::Shared {
        return Err(dflash_k0s_error(format!(
            "state tensor {label} is not shared"
        )));
    }
    let element_bytes = match tensor.dtype {
        GgmlType::F32 | GgmlType::I32 => 4usize,
        GgmlType::F16 | GgmlType::BF16 => 2usize,
        _ => return Err(dflash_k0s_error("unsupported state-digest tensor dtype")),
    };
    let elements = usize::try_from(tensor.n_elements())
        .map_err(|_| dflash_k0s_error("state-digest tensor length overflow"))?;
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| dflash_k0s_error("state-digest byte count overflow"))?;
    let offset = usize::try_from(tensor.offset)
        .map_err(|_| dflash_k0s_error("state-digest tensor offset overflow"))?;
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| dflash_k0s_error("state-digest tensor range overflow"))?;
    if end > tensor.buffer.length() {
        return Err(dflash_k0s_error("state-digest tensor exceeds its buffer"));
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (tensor.buffer.contents().as_ptr() as *const u8).add(offset),
            bytes,
        )
    };
    dflash_k0s_hash_state_tensor_material(hash, label, tensor.dtype, &tensor.shape, bytes);
    Ok(())
}

#[cfg(feature = "dflash-k0s-diagnostics")]
const DFLASH_K0S_REQUIRED_STATE_TENSORS: &[&str] = &[
    "target_ctx_stacked",
    "pos_ctx",
    "ctx_h",
    "noise_ids",
    "x",
    "h",
    "q_buf",
    "k_noise",
    "v_noise",
    "k_ctx_buf",
    "v_ctx_buf",
    "attn_o",
    "mixer_out",
    "draft_logits",
    "draft_argmax",
    "k_full",
    "v_full",
    "pos_k",
    "attn_o_full",
    "ffn_gate_buf",
    "ffn_up_buf",
    "ffn_inner_buf",
    "ffn_out_buf",
];

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_state_sha256(session: &MetalDFlashSession) -> Result<[u8; 32], DFlashError> {
    let mut hash = Sha256::new();
    hash.update(b"qwen.dflash_k0s.state.v2");
    for value in [
        session.target_ctx_n,
        session.target_ctx_capacity,
        session.ctx_h_ready_n,
        session.kv_ctx_ready_n,
    ] {
        hash.update((value as u64).to_le_bytes());
    }
    hash.update([u8::from(session.enable_phase_timers)]);
    hash.update((session.phase_timings.len() as u64).to_le_bytes());
    for (name, milliseconds) in &session.phase_timings {
        dflash_k0s_hash_bytes(&mut hash, name.as_bytes());
        hash.update(milliseconds.to_bits().to_le_bytes());
    }

    for (index, (label, tensor)) in [
        ("target_ctx_stacked", &session.target_ctx_stacked),
        ("pos_ctx", &session.pos_ctx),
        ("ctx_h", &session.ctx_h),
        ("noise_ids", &session.noise_ids),
        ("x", &session.x),
        ("h", &session.h),
        ("q_buf", &session.q_buf),
        ("k_noise", &session.k_noise),
        ("v_noise", &session.v_noise),
        ("k_ctx_buf", &session.k_ctx_buf),
        ("v_ctx_buf", &session.v_ctx_buf),
        ("attn_o", &session.attn_o),
        ("mixer_out", &session.mixer_out),
        ("draft_logits", &session.draft_logits),
        ("draft_argmax", &session.draft_argmax),
        ("k_full", &session.k_full),
        ("v_full", &session.v_full),
        ("pos_k", &session.pos_k),
        ("attn_o_full", &session.attn_o_full),
        ("ffn_gate_buf", &session.ffn_gate_buf),
        ("ffn_up_buf", &session.ffn_up_buf),
        ("ffn_inner_buf", &session.ffn_inner_buf),
        ("ffn_out_buf", &session.ffn_out_buf),
    ]
    .into_iter()
    .enumerate()
    {
        debug_assert_eq!(label, DFLASH_K0S_REQUIRED_STATE_TENSORS[index]);
        dflash_k0s_hash_state_tensor(&mut hash, label, tensor)?;
    }
    for (index, tensor) in session.k_ctx_cache.iter().enumerate() {
        dflash_k0s_hash_state_tensor(&mut hash, &format!("k_ctx_cache.{index}"), tensor)?;
    }
    for (index, tensor) in session.v_ctx_cache.iter().enumerate() {
        dflash_k0s_hash_state_tensor(&mut hash, &format!("v_ctx_cache.{index}"), tensor)?;
    }
    for (label, tensor) in [
        ("conv_buf", session.conv_buf.as_ref()),
        ("conv_dyn_attn", session.conv_dyn_attn.as_ref()),
        ("conv_dyn_ffn", session.conv_dyn_ffn.as_ref()),
        ("topk_ids", session.topk_ids.as_ref()),
        ("topk_vals", session.topk_vals.as_ref()),
        ("sel_h", session.sel_h.as_ref()),
    ] {
        dflash_k0s_hash_bytes(&mut hash, label.as_bytes());
        hash.update([u8::from(tensor.is_some())]);
        if let Some(tensor) = tensor {
            dflash_k0s_hash_state_tensor(&mut hash, label, tensor)?;
        }
    }
    Ok(hash.finalize().into())
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_session_binding_sha256(
    session: &MetalDFlashSession,
) -> Result<[u8; 32], DFlashError> {
    let mut hash = Sha256::new();
    hash.update(b"qwen.dflash_k0s.session_binding.v2");
    hash.update(session.k0s_session_sequence.to_le_bytes());
    let topk_ids = session
        .topk_ids
        .as_ref()
        .ok_or_else(|| dflash_k0s_error("K0-S top-k IDs buffer is absent"))?;
    let topk_vals = session
        .topk_vals
        .as_ref()
        .ok_or_else(|| dflash_k0s_error("K0-S unary buffer is absent"))?;
    let sel_h = session
        .sel_h
        .as_ref()
        .ok_or_else(|| dflash_k0s_error("K0-S selector-hidden buffer is absent"))?;
    for tensor in [
        &session.draft_logits,
        topk_ids,
        topk_vals,
        sel_h,
        &session.noise_ids,
    ] {
        let identity = Retained::as_ptr(&tensor.buffer) as *const () as usize as u64;
        hash.update(identity.to_le_bytes());
        hash.update(tensor.offset.to_le_bytes());
    }
    Ok(hash.finalize().into())
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_allocate_session_sequence_from(
    next: &std::sync::atomic::AtomicU64,
) -> Result<u64, DFlashError> {
    next.fetch_update(
        std::sync::atomic::Ordering::Relaxed,
        std::sync::atomic::Ordering::Relaxed,
        |value| {
            if value == 0 {
                None
            } else {
                value.checked_add(1)
            }
        },
    )
    .map_err(|_| dflash_k0s_error("K0-S session sequence overflow"))
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_allocate_session_sequence() -> Result<u64, DFlashError> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    dflash_k0s_allocate_session_sequence_from(&NEXT)
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_hash_f32le(domain: &[u8], values: &[f32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    hash.finalize().into()
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_hash_i32le(domain: &[u8], values: &[i32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(value.to_le_bytes());
    }
    hash.finalize().into()
}

#[cfg(feature = "dflash-k0s-diagnostics")]
pub fn dflash_k0s_hash_full_logits_f32le(values: &[f32]) -> [u8; 32] {
    dflash_k0s_hash_f32le(b"qwen.dflash_k0s.full_logits.f32le.v1", values)
}

#[cfg(feature = "dflash-k0s-diagnostics")]
pub fn dflash_k0s_hash_top_k_ids_i32le(values: &[i32]) -> [u8; 32] {
    dflash_k0s_hash_i32le(b"qwen.dflash_k0s.top_k_ids.i32le.v1", values)
}

#[cfg(feature = "dflash-k0s-diagnostics")]
pub fn dflash_k0s_hash_unary_f32le(values: &[f32]) -> [u8; 32] {
    dflash_k0s_hash_f32le(b"qwen.dflash_k0s.unary.f32le.v1", values)
}

#[cfg(feature = "dflash-k0s-diagnostics")]
pub fn dflash_k0s_hash_selector_hidden_f32le(values: &[f32]) -> [u8; 32] {
    dflash_k0s_hash_f32le(b"qwen.dflash_k0s.selector_hidden.f32le.v1", values)
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_selector_input_identity(
    session: &MetalDFlashSession,
) -> Result<DFlashK0sSelectorInputIdentity, DFlashError> {
    let active_logits = (DFLASH_K0S_BLOCK_SIZE - 1) * DFLASH_K0S_VOCAB;
    let active_candidates = (DFLASH_K0S_BLOCK_SIZE - 1) * DFLASH_K0S_TOP_K;
    let active_hidden = (DFLASH_K0S_BLOCK_SIZE - 1) * DFLASH_K0S_RANK;
    let logits = read_shared_selector_tensor::<f32>(
        &session
            .draft_logits
            .view_subrange(DFLASH_K0S_VOCAB as u64, vec![active_logits as u64]),
        GgmlType::F32,
        active_logits,
        "k0s_observation_full_logits",
    )?;
    let ids = read_shared_selector_tensor::<i32>(
        &session
            .topk_ids
            .as_ref()
            .ok_or_else(|| dflash_k0s_error("K0-S top-k IDs buffer is absent"))?
            .view_subrange(DFLASH_K0S_TOP_K as u64, vec![active_candidates as u64]),
        GgmlType::I32,
        active_candidates,
        "k0s_observation_topk_ids",
    )?;
    let unary = read_shared_selector_tensor::<f32>(
        &session
            .topk_vals
            .as_ref()
            .ok_or_else(|| dflash_k0s_error("K0-S unary buffer is absent"))?
            .view_subrange(DFLASH_K0S_TOP_K as u64, vec![active_candidates as u64]),
        GgmlType::F32,
        active_candidates,
        "k0s_observation_unary",
    )?;
    let selector_hidden = read_shared_selector_tensor::<f32>(
        &session
            .sel_h
            .as_ref()
            .ok_or_else(|| dflash_k0s_error("K0-S selector-hidden buffer is absent"))?
            .view_subrange(DFLASH_K0S_RANK as u64, vec![active_hidden as u64]),
        GgmlType::F32,
        active_hidden,
        "k0s_observation_selector_hidden",
    )?;
    Ok(DFlashK0sSelectorInputIdentity {
        full_logits_count: logits.len(),
        full_logits_sha256_f32le: dflash_k0s_hash_full_logits_f32le(&logits),
        top_k_ids_count: ids.len(),
        top_k_ids_sha256_i32le: dflash_k0s_hash_top_k_ids_i32le(&ids),
        unary_count: unary.len(),
        unary_sha256_f32le: dflash_k0s_hash_unary_f32le(&unary),
        selector_hidden_count: selector_hidden.len(),
        selector_hidden_sha256_f32le: dflash_k0s_hash_selector_hidden_f32le(&selector_hidden),
    })
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_selector_dispatch_identity(
    head: &MetalDFlashHead,
    session: &MetalDFlashSession,
) -> Result<DFlashK0sSelectorDispatchIdentity, DFlashError> {
    let selector = head
        .selector
        .as_ref()
        .ok_or_else(|| dflash_k0s_error("K0-S selector is absent"))?;
    let output = session
        .sel_h
        .as_ref()
        .ok_or_else(|| dflash_k0s_error("K0-S selector-hidden buffer is absent"))?;
    Ok(DFlashK0sSelectorDispatchIdentity {
        weight_dtype: selector.hidden.dtype,
        input_dtype: session.h.dtype,
        output_dtype: output.dtype,
        block_size: head.config.block_size as usize,
        hidden_size: head.config.hidden_size as usize,
        selector_rank: selector.rank,
    })
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_parity_summary_from_parts(
    head: &MetalDFlashHead,
    session: &MetalDFlashSession,
    carry_token: i32,
    noise_start_position: u32,
    draft_tokens: &[i32],
    dispatch_census: &[DFlashK0sDispatchCensusRow],
    kernel_trace: crate::metal::KernelTraceCounters,
) -> Result<DFlashK0sParitySummary, DFlashError> {
    dflash_k0s_positions(noise_start_position)?;
    dflash_k0s_check_dispatch_census_len(dispatch_census.len())?;
    Ok(DFlashK0sParitySummary {
        draft_tokens: draft_tokens.to_vec(),
        dispatch_census: dispatch_census.to_vec(),
        kernel_trace: [
            kernel_trace.encoders,
            kernel_trace.concurrent_encoders,
            kernel_trace.dispatches,
        ],
        selector_inputs: dflash_k0s_selector_input_identity(session)?,
        selector_dispatch: dflash_k0s_selector_dispatch_identity(head, session)?,
        diagnostic_state_sha256: dflash_k0s_state_sha256(session)?,
        carry_token,
        noise_start_position,
        session_binding_sha256: dflash_k0s_session_binding_sha256(session)?,
    })
}

#[cfg(feature = "dflash-k0s-diagnostics")]
/// SHA-256 over `qwen.dflash_k0s.event_envelope.v1`, followed by sequence,
/// session binding, carry, position, length-prefixed draft tokens and dispatch
/// rows, counters, selector dtypes/N/H/R, selector-input counts/hashes, and
/// state digest. Integers are fixed-width little-endian; strings use the same
/// u64-length framing as [`dflash_k0s_capture_sha256`]; option/bool tags are one
/// byte. Dispatch and vector order are preserved exactly.
pub fn dflash_k0s_event_envelope_sha256(
    parity: &DFlashK0sParitySummary,
    sequence: u64,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"qwen.dflash_k0s.event_envelope.v1");
    hash.update(sequence.to_le_bytes());
    hash.update(parity.session_binding_sha256);
    hash.update(parity.carry_token.to_le_bytes());
    hash.update(parity.noise_start_position.to_le_bytes());
    hash.update((parity.draft_tokens.len() as u64).to_le_bytes());
    for token in &parity.draft_tokens {
        hash.update(token.to_le_bytes());
    }
    hash.update((parity.dispatch_census.len() as u64).to_le_bytes());
    for row in &parity.dispatch_census {
        dflash_k0s_hash_dispatch(&mut hash, row);
    }
    for counter in parity.kernel_trace {
        hash.update(counter.to_le_bytes());
    }
    hash.update((parity.selector_dispatch.weight_dtype as u32).to_le_bytes());
    hash.update((parity.selector_dispatch.input_dtype as u32).to_le_bytes());
    hash.update((parity.selector_dispatch.output_dtype as u32).to_le_bytes());
    hash.update((parity.selector_dispatch.block_size as u64).to_le_bytes());
    hash.update((parity.selector_dispatch.hidden_size as u64).to_le_bytes());
    hash.update((parity.selector_dispatch.selector_rank as u64).to_le_bytes());
    hash.update((parity.selector_inputs.full_logits_count as u64).to_le_bytes());
    hash.update(parity.selector_inputs.full_logits_sha256_f32le);
    hash.update((parity.selector_inputs.top_k_ids_count as u64).to_le_bytes());
    hash.update(parity.selector_inputs.top_k_ids_sha256_i32le);
    hash.update((parity.selector_inputs.unary_count as u64).to_le_bytes());
    hash.update(parity.selector_inputs.unary_sha256_f32le);
    hash.update((parity.selector_inputs.selector_hidden_count as u64).to_le_bytes());
    hash.update(parity.selector_inputs.selector_hidden_sha256_f32le);
    hash.update(parity.diagnostic_state_sha256);
    hash.finalize().into()
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_next_event_sequence() -> Result<u64, DFlashError> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_update(
        std::sync::atomic::Ordering::Relaxed,
        std::sync::atomic::Ordering::Relaxed,
        |value| value.checked_add(1),
    )
    .map_err(|_| dflash_k0s_error("K0-S event sequence overflow"))
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_consume_observation(
    head: &MetalDFlashHead,
    session: &MetalDFlashSession,
    vocab: usize,
    live_event: &mut Option<u64>,
    observation: DFlashK0sProductionObservation,
) -> Result<DFlashK0sCapture, DFlashError> {
    let summary = dflash_k0s_finish_observation(head, session, live_event, observation)?;
    let parity = summary.parity;
    let counters = crate::metal::KernelTraceCounters {
        encoders: parity.kernel_trace[0],
        concurrent_encoders: parity.kernel_trace[1],
        dispatches: parity.kernel_trace[2],
    };
    DFlashDecoder::extract_k0s_post_sync(
        head,
        session,
        vocab,
        parity.carry_token,
        parity.noise_start_position,
        parity.draft_tokens,
        parity.dispatch_census,
        counters,
    )
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn dflash_k0s_finish_observation(
    head: &MetalDFlashHead,
    session: &MetalDFlashSession,
    live_event: &mut Option<u64>,
    observation: DFlashK0sProductionObservation,
) -> Result<DFlashK0sObservationSummary, DFlashError> {
    let DFlashK0sObservationSummary {
        parity,
        event_sequence,
        event_envelope_sha256,
    } = observation.summary;
    if *live_event != Some(event_sequence) {
        return Err(dflash_k0s_error(
            "stale or already-consumed K0-S observation",
        ));
    }
    *live_event = None;
    if dflash_k0s_event_envelope_sha256(&parity, event_sequence) != event_envelope_sha256 {
        return Err(dflash_k0s_error("K0-S observation event envelope mismatch"));
    }
    let counters = crate::metal::KernelTraceCounters {
        encoders: parity.kernel_trace[0],
        concurrent_encoders: parity.kernel_trace[1],
        dispatches: parity.kernel_trace[2],
    };
    let current = dflash_k0s_parity_summary_from_parts(
        head,
        session,
        parity.carry_token,
        parity.noise_start_position,
        &parity.draft_tokens,
        &parity.dispatch_census,
        counters,
    )?;
    if current != parity {
        return Err(dflash_k0s_error(
            "K0-S observation is stale or bound to a different session/state",
        ));
    }
    Ok(DFlashK0sObservationSummary {
        parity,
        event_sequence,
        event_envelope_sha256,
    })
}

/// Executes the compiled rank-256 scalar graph used by K0-S: first `A * z`
/// in rank order, then separate `gate * B` and accumulator additions, then
/// unary addition. The digest domain is the literal ASCII string
/// `qwen.dflash_k0s.scalar_contract_fixture.v2`, followed by the case count as
/// little-endian u64. Each case hashes its u64-length-prefixed ASCII name,
/// then each rank-256 A/z/successor vector as a u64 count and little-endian
/// u32 bits, followed by unary and score u32 bits. Case and rank order are
/// fixed as returned.
#[cfg(feature = "dflash-k0s-diagnostics")]
pub fn dflash_k0s_scalar_contract_fixture() -> DFlashK0sScalarContractFixture {
    let mut inputs = Vec::with_capacity(6);
    let blank = || {
        (
            [0.0f32; DFLASH_K0S_RANK],
            [1.0f32; DFLASH_K0S_RANK],
            [0.0f32; DFLASH_K0S_RANK],
            0.0f32,
        )
    };

    let mut cancellation = blank();
    cancellation.0[0] = -1.0;
    cancellation.2[0] = 1.0;
    cancellation.0[1] = f32::from_bits(0x3f80_0001);
    cancellation.2[1] = f32::from_bits(0x3f7f_ffff);
    inputs.push(("fma_sensitive_cancellation", cancellation));

    let mut subnormal = blank();
    subnormal.0[0] = f32::from_bits(1);
    subnormal.2[0] = 1.0;
    subnormal.0[1] = f32::from_bits(2);
    subnormal.2[1] = -1.0;
    inputs.push(("subnormal_signed_result", subnormal));

    let mut signed_zero = blank();
    signed_zero.3 = -0.0;
    signed_zero.2.fill(-1.0);
    inputs.push(("signed_zero", signed_zero));

    let mut adjacent = blank();
    adjacent.0[0] = f32::MAX;
    adjacent.0[1] = f32::MAX;
    adjacent.1[0] = 0.5;
    adjacent.1[1] = 0.5;
    adjacent.2[0] = 1.0;
    adjacent.2[1] = 1.0;
    inputs.push(("overflow_adjacent_finite", adjacent));

    let mut rank_order = blank();
    rank_order.0[0] = 1.0e20;
    rank_order.2[0] = 1.0;
    rank_order.0[1] = -1.0e20;
    rank_order.2[1] = 1.0;
    rank_order.0[2] = 3.25;
    rank_order.2[2] = 1.0;
    inputs.push(("rank_order_cancellation", rank_order));

    let mut halfway = blank();
    halfway.0[0] = 1.0;
    halfway.2[0] = 1.0;
    halfway.0[1] = f32::from_bits(0x3380_0000);
    halfway.2[1] = 1.0;
    inputs.push(("halfway_round_to_even", halfway));

    let mut hash = Sha256::new();
    hash.update(b"qwen.dflash_k0s.scalar_contract_fixture.v2");
    hash.update((inputs.len() as u64).to_le_bytes());
    let mut cases = Vec::with_capacity(inputs.len());
    for (name, (a, z, successor, unary)) in inputs {
        let mut gate = [0.0f32; DFLASH_K0S_RANK];
        for rank in 0..DFLASH_K0S_RANK {
            gate[rank] = a[rank] * z[rank];
        }
        let score = dflash_k0s_scalar_score(&gate, &successor, unary);
        let a_bits: Vec<u32> = a.iter().map(|value| value.to_bits()).collect();
        let z_bits: Vec<u32> = z.iter().map(|value| value.to_bits()).collect();
        let successor_bits: Vec<u32> = successor.iter().map(|value| value.to_bits()).collect();
        dflash_k0s_hash_bytes(&mut hash, name.as_bytes());
        for bits in [&a_bits, &z_bits, &successor_bits] {
            hash.update((bits.len() as u64).to_le_bytes());
            for value in bits {
                hash.update(value.to_le_bytes());
            }
        }
        hash.update(unary.to_bits().to_le_bytes());
        hash.update(score.to_bits().to_le_bytes());
        cases.push(DFlashK0sScalarContractCase {
            name,
            a_bits,
            z_bits,
            successor_bits,
            unary_bits: unary.to_bits(),
            score_bits: score.to_bits(),
        });
    }
    DFlashK0sScalarContractFixture {
        cases,
        fixture_sha256: hash.finalize().into(),
    }
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
                let conv = match l.conv.as_ref() {
                    Some(c) => Some(MetalDFlash2Conv {
                        attn_base: load_f32(c.attn_base)?,
                        attn_proj: load_weight(c.attn_proj)?,
                        ffn_base: load_f32(c.ffn_base)?,
                        ffn_proj: load_weight(c.ffn_proj)?,
                    }),
                    None => None,
                };
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
                    conv,
                })
            })
            .collect();

        let selector = match head.selector.as_ref() {
            Some(sel) => {
                // kernel_topk16_f32 is specialized to k=16; the released
                // DFlash 2 drafters all ship selector_top_k=16.
                if head.config.selector_top_k != 16 {
                    return Err(DFlashError::BadDrafter(
                        "only selector_top_k=16 is supported",
                    ));
                }
                Some(MetalDFlash2Selector {
                    hidden: load_weight(sel.hidden)?,
                    predecessor: DFlash2Codebook::from_gguf(
                        sel.predecessor,
                        drafter_gguf.slice(sel.predecessor),
                    )?,
                    successor: DFlash2Codebook::from_gguf(
                        sel.successor,
                        drafter_gguf.slice(sel.successor),
                    )?,
                    rank: head.config.selector_rank as usize,
                    top_k: head.config.selector_top_k as usize,
                    #[cfg(feature = "dflash-k0s-diagnostics")]
                    hidden_provenance: DFlashK0sTensorProvenance {
                        descriptor: sel.hidden.clone(),
                        full_tensor_sha256: Sha256::digest(drafter_gguf.slice(sel.hidden)).into(),
                    },
                })
            }
            None => None,
        };

        Ok(Self {
            config: head.config,
            target_layer_ids: head.target_layer_ids.clone(),
            fc: load_weight(head.fc)?,
            hidden_norm: load_f32(head.hidden_norm)?,
            output_norm: load_f32(head.output_norm)?,
            layers: layers?,
            selector,
        })
    }
}

/// Per-step DFlash session state.
pub struct MetalDFlashSession {
    #[cfg(feature = "dflash-k0s-diagnostics")]
    k0s_session_sequence: u64,

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

    /// Per-step block input `[N]` I32 (carry + (N-1) MASK).
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
    /// `[N]` I32 — drafter argmax destination, written
    /// by the GPU argmax kernel after the batched lm_head. Avoids the
    /// per-row CPU readback + scalar-loop argmax that v0.71's draft_block
    /// did. v0.72.0 codex-recommended port from packed_verify's batched
    /// tail.
    pub draft_argmax: MetalTensor,

    // ---- DFlash 2 buffers (Some iff config.selector_top_k > 0) ----
    /// `[N, H]` F32 — conv output scratch. Side-0 conv writes here and the
    /// following projections read it; side-1 conv writes here and the
    /// residual add reads it. Never aliases its input (taps read
    /// neighboring rows).
    pub conv_buf: Option<MetalTensor>,
    /// `[N, 2·kernel·n_groups]` F32 — attention conv dynamic coefficients
    /// (computed from the pre-conv normed input; used by both sides).
    pub conv_dyn_attn: Option<MetalTensor>,
    /// `[N, 2·kernel·n_groups]` F32 — FFN conv dynamic coefficients.
    pub conv_dyn_ffn: Option<MetalTensor>,
    /// `[N, 16]` I32 — per-position top-16 candidate token ids.
    pub topk_ids: Option<MetalTensor>,
    /// `[N, 16]` F32 — matching top-16 logits (the selector's unary term).
    pub topk_vals: Option<MetalTensor>,
    /// `[N, rank]` F32 — context gate `W_h · h` per position.
    pub sel_h: Option<MetalTensor>,

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

#[derive(Clone, Copy, Debug)]
struct DFlashSessionGeometry {
    n: u64,
    n_target_features: u64,
    cc: u64,
    ctx_h_elems: u64,
    kv_ctx_elems: u64,
    x_elems: u64,
    q_elems: u64,
    kv_noise_elems: u64,
    logits_elems: u64,
    full_ctx_tokens: u64,
    kv_full_elems: u64,
    ffn_elems: u64,
}

impl DFlashSessionGeometry {
    fn new(
        cfg: &crate::loader::DFlashConfig,
        k_layers: usize,
        target_h: u64,
        vocab: u64,
        ctx_capacity: usize,
    ) -> Result<Self, DFlashError> {
        let n = cfg.block_size as u64;
        let h = cfg.hidden_size as u64;
        let q_dim = checked_u64_mul(
            cfg.n_q_heads as u64,
            cfg.head_dim as u64,
            "dflash q_dim overflow",
        )?;
        let kv_dim = checked_u64_mul(
            cfg.n_kv_heads as u64,
            cfg.head_dim as u64,
            "dflash kv_dim overflow",
        )?;
        let n_target_features = checked_u64_mul(
            k_layers as u64,
            target_h,
            "dflash n_target_features overflow",
        )?;
        let cc = ctx_capacity as u64;
        let ctx_h_elems = checked_u64_mul(h, cc, "dflash ctx_h size overflow")?;
        let kv_ctx_elems = checked_u64_mul(cc, kv_dim, "dflash kv ctx cache size overflow")?;
        let x_elems = checked_u64_mul(n, h, "dflash x size overflow")?;
        let q_elems = checked_u64_mul(n, q_dim, "dflash q size overflow")?;
        let kv_noise_elems = checked_u64_mul(n, kv_dim, "dflash kv noise size overflow")?;
        let logits_elems = checked_u64_mul(n, vocab, "dflash logits size overflow")?;
        let full_ctx_tokens = checked_u64_add(cc, n, "dflash cc + n overflow")?;
        let kv_full_elems =
            checked_u64_mul(full_ctx_tokens, kv_dim, "dflash full kv size overflow")?;
        let ffn_elems = checked_u64_mul(
            n,
            cfg.intermediate_size as u64,
            "dflash ffn scratch size overflow",
        )?;
        Ok(Self {
            n,
            n_target_features,
            cc,
            ctx_h_elems,
            kv_ctx_elems,
            x_elems,
            q_elems,
            kv_noise_elems,
            logits_elems,
            full_ctx_tokens,
            kv_full_elems,
            ffn_elems,
        })
    }

    fn context_allocation_elements(&self, drafter_layers: u32) -> Result<Vec<u64>, DFlashError> {
        let target_ctx = checked_u64_mul(
            self.n_target_features,
            self.cc,
            "dflash target context size overflow",
        )?;
        let mut elements = vec![target_ctx, self.cc, self.ctx_h_elems];
        for _ in 0..drafter_layers {
            elements.push(self.kv_ctx_elems);
            elements.push(self.kv_ctx_elems);
        }
        elements.extend([
            self.kv_ctx_elems,
            self.kv_ctx_elems,
            self.kv_full_elems,
            self.kv_full_elems,
            self.full_ctx_tokens,
        ]);
        Ok(elements)
    }

    #[cfg(test)]
    fn context_logical_bytes(&self, drafter_layers: u32) -> Result<u64, DFlashError> {
        self.context_allocation_elements(drafter_layers)?
            .into_iter()
            .try_fold(0u64, |total, elements| {
                let bytes = checked_u64_mul(elements, 4, "dflash context byte size overflow")?;
                Ok(checked_u64_add(
                    total,
                    bytes,
                    "dflash context byte total overflow",
                )?)
            })
    }
}

struct DFlashPhase3SplitRecord {
    name: &'static str,
    start_sample: usize,
    end_sample: usize,
}

struct DFlashPhase3SplitRecorder {
    samples: MetalTimestampSampleBuffer,
    next_sample: usize,
    records: Vec<DFlashPhase3SplitRecord>,
}

impl DFlashPhase3SplitRecorder {
    fn new(ctx: &MetalContext, n_stages: usize) -> Result<Self, DFlashError> {
        let sample_count = n_stages.checked_mul(2).ok_or_else(|| {
            DFlashError::Metal(MetalError::Counter(
                "dflash sample count overflow".to_string(),
            ))
        })?;
        Ok(Self {
            samples: ctx.timestamp_sample_buffer(sample_count)?,
            next_sample: 0,
            records: Vec::with_capacity(n_stages),
        })
    }

    fn begin(
        &mut self,
        cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        name: &'static str,
    ) -> Result<KernelEncoder, DFlashError> {
        let start_sample = self.next_sample;
        let end_sample = start_sample + 1;
        if end_sample >= self.samples.sample_count() {
            return Err(DFlashError::Metal(MetalError::Counter(format!(
                "dflash phase3 timestamp buffer exhausted at sample {end_sample}"
            ))));
        }
        self.next_sample += 2;
        self.records.push(DFlashPhase3SplitRecord {
            name,
            start_sample,
            end_sample,
        });
        Ok(KernelEncoder::begin_sampled(
            cmd,
            &self.samples,
            start_sample,
            end_sample,
            false,
        ))
    }

    fn record(
        self,
        ctx: &MetalContext,
        session: &mut MetalDFlashSession,
        total_gpu_ms: f64,
    ) -> Result<(), DFlashError> {
        let timestamps = ctx.resolve_timestamp_samples(&self.samples, self.next_sample)?;
        let sampled_span_ticks = match (self.records.first(), self.records.last()) {
            (Some(first), Some(last)) => {
                timestamps[last.end_sample].saturating_sub(timestamps[first.start_sample])
            }
            _ => 0,
        };
        let scale_ms_per_tick = if sampled_span_ticks > 0 {
            total_gpu_ms / sampled_span_ticks as f64
        } else {
            0.0
        };
        let mut recorded_ms = 0.0f64;
        for record in self.records {
            let ticks =
                timestamps[record.end_sample].saturating_sub(timestamps[record.start_sample]);
            let ms = ticks as f64 * scale_ms_per_tick;
            recorded_ms += ms;
            session.phase_timings.push((record.name.to_string(), ms));
        }
        let unattributed_ms = (total_gpu_ms - recorded_ms).max(0.0);
        if unattributed_ms > 0.0 {
            session
                .phase_timings
                .push(("phase3_split_unattributed".to_string(), unattributed_ms));
        }
        Ok(())
    }
}

impl MetalDFlashSession {
    /// Priced bytes for every session allocation that grows with context
    /// capacity. Fixed block/verify/layer scratch is intentionally excluded so
    /// callers can cover it with a separate reserve.
    pub fn context_capacity_priced_bytes(
        ctx: &MetalContext,
        head: &MetalDFlashHead,
        target_h: u64,
        ctx_capacity: usize,
    ) -> Result<u64, DFlashError> {
        let geometry = DFlashSessionGeometry::new(
            &head.config,
            head.target_layer_ids.len(),
            target_h,
            0,
            ctx_capacity,
        )?;
        geometry
            .context_allocation_elements(head.config.n_layer)?
            .into_iter()
            .try_fold(0u64, |total, elements| {
                let logical = checked_u64_mul(elements, 4, "dflash context byte size overflow")?;
                let priced = ctx.shared_buffer_size_and_align(logical)?.size;
                checked_u64_add(total, priced, "dflash context priced byte total overflow")
                    .map_err(DFlashError::from)
            })
    }

    /// Priced bytes for every Metal buffer allocated by [`Self::fresh`].
    /// Keep this list in constructor order so admission cannot silently omit
    /// fixed draft scratch while accounting only for context-sized buffers.
    pub fn priced_bytes(
        ctx: &MetalContext,
        head: &MetalDFlashHead,
        target_h: u64,
        vocab: u64,
        ctx_capacity: usize,
    ) -> Result<u64, DFlashError> {
        let cfg = &head.config;
        let geometry = DFlashSessionGeometry::new(
            cfg,
            head.target_layer_ids.len(),
            target_h,
            vocab,
            ctx_capacity,
        )?;
        let mut allocations = geometry.context_allocation_elements(cfg.n_layer)?;
        allocations.extend([
            geometry.n,
            geometry.x_elems,
            geometry.x_elems,
            geometry.q_elems,
            geometry.kv_noise_elems,
            geometry.kv_noise_elems,
            geometry.q_elems,
            geometry.x_elems,
            geometry.logits_elems,
            geometry.n,
            geometry.q_elems,
            geometry.ffn_elems,
            geometry.ffn_elems,
            geometry.ffn_elems,
            geometry.x_elems,
        ]);
        if cfg.selector_top_k > 0 {
            let n_groups = (cfg.hidden_size / cfg.conv_group_size) as u64;
            let dyn_dim = checked_u64_mul(
                2 * cfg.conv_kernel_size as u64,
                n_groups,
                "dflash2 conv dyn dim overflow",
            )?;
            let dyn_elems = checked_u64_mul(geometry.n, dyn_dim, "dflash2 conv dyn size overflow")?;
            let topk_elems = checked_u64_mul(
                geometry.n,
                cfg.selector_top_k as u64,
                "dflash2 topk size overflow",
            )?;
            let sel_elems = checked_u64_mul(
                geometry.n,
                cfg.selector_rank as u64,
                "dflash2 sel_h size overflow",
            )?;
            allocations.extend([
                geometry.x_elems,
                dyn_elems,
                dyn_elems,
                topk_elems,
                topk_elems,
                sel_elems,
            ]);
        }
        allocations.into_iter().try_fold(0u64, |total, elements| {
            let logical = checked_u64_mul(elements, 4, "dflash session byte size overflow")?;
            let priced = ctx.shared_buffer_size_and_align(logical)?.size;
            Ok(checked_u64_add(
                total,
                priced,
                "dflash session priced byte total overflow",
            )?)
        })
    }

    pub fn fresh(
        ctx: &MetalContext,
        head: &MetalDFlashHead,
        target_h: u64,
        vocab: u64,
        ctx_capacity: usize,
    ) -> Result<Self, DFlashError> {
        let cfg = &head.config;
        let geometry = DFlashSessionGeometry::new(
            cfg,
            head.target_layer_ids.len(),
            target_h,
            vocab,
            ctx_capacity,
        )?;
        let DFlashSessionGeometry {
            n,
            n_target_features,
            cc,
            ctx_h_elems,
            kv_ctx_elems,
            x_elems,
            q_elems,
            kv_noise_elems,
            logits_elems,
            full_ctx_tokens,
            kv_full_elems,
            ffn_elems,
        } = geometry;
        // DFlash 2 scratch (conv + selector), sized from the GGUF conv
        // metadata. ~90 KB total at N=8, H=5120 — negligible.
        let (conv_buf, conv_dyn_attn, conv_dyn_ffn, topk_ids, topk_vals, sel_h) =
            if cfg.selector_top_k > 0 {
                let n_groups = (cfg.hidden_size / cfg.conv_group_size) as u64;
                let dyn_dim = checked_u64_mul(
                    2 * cfg.conv_kernel_size as u64,
                    n_groups,
                    "dflash2 conv dyn dim overflow",
                )?;
                let dyn_elems = checked_u64_mul(n, dyn_dim, "dflash2 conv dyn size overflow")?;
                let topk_elems =
                    checked_u64_mul(n, cfg.selector_top_k as u64, "dflash2 topk size overflow")?;
                let sel_elems =
                    checked_u64_mul(n, cfg.selector_rank as u64, "dflash2 sel_h size overflow")?;
                (
                    Some(MetalTensor::zeros_f32(ctx, vec![x_elems])?),
                    Some(MetalTensor::zeros_f32(ctx, vec![dyn_elems])?),
                    Some(MetalTensor::zeros_f32(ctx, vec![dyn_elems])?),
                    Some(MetalTensor::zeros_i32(ctx, vec![topk_elems])?),
                    Some(MetalTensor::zeros_f32(ctx, vec![topk_elems])?),
                    Some(MetalTensor::zeros_f32(ctx, vec![sel_elems])?),
                )
            } else {
                (None, None, None, None, None, None)
            };
        Ok(Self {
            target_ctx_stacked: MetalTensor::zeros_f32(
                ctx,
                vec![checked_u64_mul(
                    n_target_features,
                    cc,
                    "dflash target_ctx_stacked size overflow",
                )?],
            )?,
            target_ctx_n: 0,
            target_ctx_capacity: ctx_capacity,
            pos_ctx: MetalTensor::zeros_f32(ctx, vec![cc])?,
            ctx_h: MetalTensor::zeros_f32(ctx, vec![ctx_h_elems])?,
            ctx_h_ready_n: 0,
            // v0.74.1: one K/V cache pair per drafter layer (cfg.n_layer).
            // Sized for ctx_capacity * kv_dim. Memory cost on M4 Max
            // 128 GB: 5 layers × ctx_capacity × kv_dim × 2 × 4 B. At
            // ctx_capacity=8K, kv_dim=1024 → 320 MB. Tractable.
            k_ctx_cache: (0..cfg.n_layer)
                .map(|_| MetalTensor::zeros_f32(ctx, vec![kv_ctx_elems]))
                .collect::<Result<Vec<_>, _>>()?,
            v_ctx_cache: (0..cfg.n_layer)
                .map(|_| MetalTensor::zeros_f32(ctx, vec![kv_ctx_elems]))
                .collect::<Result<Vec<_>, _>>()?,
            kv_ctx_ready_n: 0,
            noise_ids: MetalTensor::zeros_i32(ctx, vec![n])?,
            x: MetalTensor::zeros_f32(ctx, vec![x_elems])?,
            h: MetalTensor::zeros_f32(ctx, vec![x_elems])?,
            q_buf: MetalTensor::zeros_f32(ctx, vec![q_elems])?,
            k_noise: MetalTensor::zeros_f32(ctx, vec![kv_noise_elems])?,
            v_noise: MetalTensor::zeros_f32(ctx, vec![kv_noise_elems])?,
            k_ctx_buf: MetalTensor::zeros_f32(ctx, vec![kv_ctx_elems])?,
            v_ctx_buf: MetalTensor::zeros_f32(ctx, vec![kv_ctx_elems])?,
            attn_o: MetalTensor::zeros_f32(ctx, vec![q_elems])?,
            mixer_out: MetalTensor::zeros_f32(ctx, vec![x_elems])?,
            draft_logits: MetalTensor::zeros_f32(ctx, vec![logits_elems])?,
            draft_argmax: MetalTensor::zeros_i32(ctx, vec![n])?,
            conv_buf,
            conv_dyn_attn,
            conv_dyn_ffn,
            topk_ids,
            topk_vals,
            sel_h,
            // v0.72.1 phase 3 buffers
            k_full: MetalTensor::zeros_f32(ctx, vec![kv_full_elems])?,
            v_full: MetalTensor::zeros_f32(ctx, vec![kv_full_elems])?,
            pos_k: MetalTensor::zeros_f32(ctx, vec![full_ctx_tokens])?,
            attn_o_full: MetalTensor::zeros_f32(ctx, vec![q_elems])?,
            ffn_gate_buf: MetalTensor::zeros_f32(ctx, vec![ffn_elems])?,
            ffn_up_buf: MetalTensor::zeros_f32(ctx, vec![ffn_elems])?,
            ffn_inner_buf: MetalTensor::zeros_f32(ctx, vec![ffn_elems])?,
            ffn_out_buf: MetalTensor::zeros_f32(ctx, vec![x_elems])?,
            enable_phase_timers: false,
            phase_timings: Vec::new(),
            #[cfg(feature = "dflash-k0s-diagnostics")]
            k0s_session_sequence: dflash_k0s_allocate_session_sequence()?,
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

    /// `[N]` F32 — per-row (top1 - top2) argmax gap, written alongside
    /// `verify_argmax` by `encode_argmax_top2_f32`. Drives the
    /// margin-guarded exact fallback in the speculative loop.
    pub verify_gap: MetalTensor,

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

    /// `[n_gdn, ssm_state_elems]` F32 — pre-block GDN SSM state capture,
    /// blitted before the per-row loop so a margin-guarded fallback can
    /// roll the whole block back to its start.
    pub pre_gdn_ckpt: MetalTensor,

    /// `[n_gdn, conv_state_elems]` F32 — pre-block GDN conv state capture.
    pub pre_conv_ckpt: MetalTensor,

    // -- Cached dimensions (so slot helpers don't have to take a model ref) --
    pub n: u32,
    pub k_target_layers: u32,
    pub n_gdn_layers: u32,
    pub hidden_size: u64,
    pub ssm_state_elems: u64,
    pub conv_state_elems: u64,
}

impl MetalDFlashVerifyScratch {
    /// Priced bytes for every Metal buffer allocated by [`Self::fresh`].
    pub fn priced_bytes(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
        k_target_layers: u32,
    ) -> Result<u64, DFlashError> {
        let arch = &target_model.arch;
        let n = block_size as u64;
        let k = k_target_layers as u64;
        let h = arch.hidden_size as u64;
        let n_gdn_layers = u64::try_from(
            target_model
                .blocks
                .iter()
                .filter(|block| matches!(block, crate::metal_forward::MetalBlock::Gdn(_)))
                .count(),
        )
        .map_err(|_| {
            DFlashError::Metal(MetalError::BadShape {
                kernel: "dflash_verify_scratch",
                detail: "GDN layer count does not fit u64".into(),
            })
        })?;
        let ssm_state_elems = checked_u64_mul3(
            arch.gdn_n_v_heads as u64,
            arch.gdn_head_dim as u64,
            arch.gdn_head_dim as u64,
            "dflash verify ssm_state_elems overflow",
        )?;
        let conv_heads = checked_u64_add(
            checked_u64_double(
                arch.gdn_n_k_heads as u64,
                "dflash verify 2 * gdn_n_k_heads overflow",
            )?,
            arch.gdn_n_v_heads as u64,
            "dflash verify conv heads overflow",
        )?;
        let conv_dim = checked_u64_mul(
            conv_heads,
            arch.gdn_head_dim as u64,
            "dflash verify conv_dim overflow",
        )?;
        let conv_state_elems = checked_u64_mul(
            (arch.gdn_conv_kernel as u64).saturating_sub(1),
            conv_dim,
            "dflash verify conv_state_elems overflow",
        )?;
        let allocations = [
            n,
            n,
            n,
            checked_u64_mul3(n, k, h, "dflash verify hidden capture overflow")?,
            checked_u64_mul3(
                n_gdn_layers,
                n,
                ssm_state_elems,
                "dflash verify GDN checkpoint overflow",
            )?,
            checked_u64_mul3(
                n_gdn_layers,
                n,
                conv_state_elems,
                "dflash verify conv checkpoint overflow",
            )?,
            checked_u64_mul3(
                n_gdn_layers,
                1,
                ssm_state_elems,
                "dflash verify pre-block GDN checkpoint overflow",
            )?,
            checked_u64_mul3(
                n_gdn_layers,
                1,
                conv_state_elems,
                "dflash verify pre-block conv checkpoint overflow",
            )?,
        ];
        allocations.into_iter().try_fold(0u64, |total, elements| {
            let logical = checked_u64_mul(elements, 4, "dflash verify byte size overflow")?;
            let priced = ctx.shared_buffer_size_and_align(logical)?.size;
            Ok(checked_u64_add(
                total,
                priced,
                "dflash verify priced byte total overflow",
            )?)
        })
    }

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
        let ssm_state_elems = checked_u64_mul3(
            arch.gdn_n_v_heads as u64,
            arch.gdn_head_dim as u64,
            arch.gdn_head_dim as u64,
            "dflash verify ssm_state_elems overflow",
        )?;

        // Conv state: (kernel - 1) · conv_dim where
        // conv_dim = (2 * n_k + n_v) * head_dim.
        let conv_heads = checked_u64_add(
            checked_u64_double(
                arch.gdn_n_k_heads as u64,
                "dflash verify 2 * gdn_n_k_heads overflow",
            )?,
            arch.gdn_n_v_heads as u64,
            "dflash verify conv heads overflow",
        )?;
        let conv_dim = checked_u64_mul(
            conv_heads,
            arch.gdn_head_dim as u64,
            "dflash verify conv_dim overflow",
        )?;
        let conv_state_elems = checked_u64_mul(
            (arch.gdn_conv_kernel as u64).saturating_sub(1),
            conv_dim,
            "dflash verify conv_state_elems overflow",
        )?;

        Ok(Self {
            packed_ids_buf: MetalTensor::zeros_i32(ctx, vec![n])?,
            verify_argmax: MetalTensor::zeros_i32(ctx, vec![n])?,
            verify_gap: MetalTensor::zeros_f32(ctx, vec![n])?,
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
            pre_gdn_ckpt: MetalTensor::zeros_f32(ctx, vec![n_gdn_layers, ssm_state_elems])?,
            pre_conv_ckpt: MetalTensor::zeros_f32(ctx, vec![n_gdn_layers, conv_state_elems])?,
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

    /// Zero-copy view of `verify_gap[n..n+1]`.
    pub fn gap_slot(&self, n: u32) -> MetalTensor {
        assert!(n < self.n, "gap_slot OOB: n={n} >= scratch.n={}", self.n);
        self.verify_gap.view_subrange(n as u64, vec![1])
    }

    /// Zero-copy view of the pre-block GDN SSM capture for `layer`.
    pub fn pre_gdn_slot(&self, layer: u32) -> MetalTensor {
        assert!(
            layer < self.n_gdn_layers,
            "pre_gdn_slot OOB: layer={layer} >= n_gdn_layers={}",
            self.n_gdn_layers
        );
        self.pre_gdn_ckpt.view_subrange(
            (layer as u64) * self.ssm_state_elems,
            vec![self.ssm_state_elems],
        )
    }

    /// Zero-copy view of the pre-block GDN conv capture for `layer`.
    pub fn pre_conv_slot(&self, layer: u32) -> MetalTensor {
        assert!(
            layer < self.n_gdn_layers,
            "pre_conv_slot OOB: layer={layer} >= n_gdn_layers={}",
            self.n_gdn_layers
        );
        self.pre_conv_ckpt.view_subrange(
            (layer as u64) * self.conv_state_elems,
            vec![self.conv_state_elems],
        )
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
//   attn_q_pack       stub (v0.447)     4 B  (split path removed v0.432)
//   attn_gate_pack    stub (v0.447)     4 B  (split path removed v0.432)
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
const PREFILL_SCRATCH_OVERLAY_ALIGNMENT: u64 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OverlayRange {
    offset: u64,
    bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrefillScratchOverlayLayout {
    scores_h: OverlayRange,
    ml: OverlayRange,
    gdn_qkv: OverlayRange,
    gdn_z: OverlayRange,
    gdn_beta: OverlayRange,
    gdn_alpha: OverlayRange,
    gdn_q_norm: OverlayRange,
    gdn_k_norm: OverlayRange,
    gdn_v: OverlayRange,
    gdn_out: OverlayRange,
    gdn_normed: OverlayRange,
    attention_bytes: u64,
    gdn_bytes: u64,
    backing_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillScratchOverlayStats {
    pub backing_bytes: u64,
    pub attention_bytes: u64,
    pub gdn_bytes: u64,
    /// Savings between the two 256-byte-aligned alternative layouts and
    /// their shared backing. Product profile dimensions add no alignment pad.
    pub saved_bytes: u64,
}

struct PrefillScratchOverlayViews {
    scores_h: MetalTensor,
    ml: MetalTensor,
    gdn_qkv: MetalTensor,
    gdn_z: MetalTensor,
    gdn_beta: MetalTensor,
    gdn_alpha: MetalTensor,
    gdn_q_norm: MetalTensor,
    gdn_k_norm: MetalTensor,
    gdn_v: MetalTensor,
    gdn_out: MetalTensor,
    gdn_normed: MetalTensor,
}

fn checked_align_overlay(value: u64) -> Result<u64, MetalError> {
    let mask = PREFILL_SCRATCH_OVERLAY_ALIGNMENT - 1;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "prefill_scratch_overlay",
            detail: format!("cannot align byte offset {value}"),
        })
}

fn checked_overlay_range(cursor: &mut u64, bytes: u64) -> Result<OverlayRange, MetalError> {
    let offset = checked_align_overlay(*cursor)?;
    *cursor = offset
        .checked_add(bytes)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "prefill_scratch_overlay",
            detail: format!("byte range {offset}+{bytes} overflows"),
        })?;
    Ok(OverlayRange { offset, bytes })
}

fn prefill_scratch_overlay_layout(
    scores_h_elems: u64,
    ml_elems: u64,
    gdn_shapes: [u64; 9],
) -> Result<PrefillScratchOverlayLayout, MetalError> {
    let score_bytes = checked_u64_mul(scores_h_elems, 2, "overlay score bytes overflow")?;
    let ml_bytes = checked_u64_mul(ml_elems, 4, "overlay ml bytes overflow")?;
    let mut attention_cursor = 0;
    let scores_h = checked_overlay_range(&mut attention_cursor, score_bytes)?;
    let ml = checked_overlay_range(&mut attention_cursor, ml_bytes)?;
    let attention_bytes = checked_align_overlay(attention_cursor)?;

    let mut gdn_cursor = 0;
    let mut next_gdn = |elems: u64| {
        let bytes = checked_u64_mul(elems, 4, "overlay GDN bytes overflow")?;
        checked_overlay_range(&mut gdn_cursor, bytes)
    };
    let gdn_qkv = next_gdn(gdn_shapes[0])?;
    let gdn_z = next_gdn(gdn_shapes[1])?;
    let gdn_beta = next_gdn(gdn_shapes[2])?;
    let gdn_alpha = next_gdn(gdn_shapes[3])?;
    let gdn_q_norm = next_gdn(gdn_shapes[4])?;
    let gdn_k_norm = next_gdn(gdn_shapes[5])?;
    let gdn_v = next_gdn(gdn_shapes[6])?;
    let gdn_out = next_gdn(gdn_shapes[7])?;
    let gdn_normed = next_gdn(gdn_shapes[8])?;
    let gdn_bytes = checked_align_overlay(gdn_cursor)?;

    Ok(PrefillScratchOverlayLayout {
        scores_h,
        ml,
        gdn_qkv,
        gdn_z,
        gdn_beta,
        gdn_alpha,
        gdn_q_norm,
        gdn_k_norm,
        gdn_v,
        gdn_out,
        gdn_normed,
        attention_bytes,
        gdn_bytes,
        backing_bytes: attention_bytes.max(gdn_bytes),
    })
}

fn checked_overlay_tensor(
    backing: &MetalTensor,
    range: OverlayRange,
    shape: Vec<u64>,
    dtype: GgmlType,
) -> Result<MetalTensor, MetalError> {
    let elem_bytes = match dtype {
        GgmlType::F16 => 2u64,
        GgmlType::F32 => 4u64,
        other => {
            return Err(MetalError::BadShape {
                kernel: "prefill_scratch_overlay",
                detail: format!("unsupported overlay dtype {other:?}"),
            });
        }
    };
    let elements = shape.iter().try_fold(1u64, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| MetalError::BadShape {
            kernel: "prefill_scratch_overlay",
            detail: format!("shape {shape:?} overflows"),
        })
    })?;
    let bytes = elements
        .checked_mul(elem_bytes)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "prefill_scratch_overlay",
            detail: format!("shape {shape:?} byte size overflows"),
        })?;
    let end = range
        .offset
        .checked_add(bytes)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "prefill_scratch_overlay",
            detail: "view end overflows".into(),
        })?;
    let backing_bytes = backing.buffer.length() as u64;
    if bytes != range.bytes
        || !range.offset.is_multiple_of(elem_bytes)
        || !range
            .offset
            .is_multiple_of(PREFILL_SCRATCH_OVERLAY_ALIGNMENT)
        || end > backing_bytes
    {
        return Err(MetalError::BadShape {
            kernel: "prefill_scratch_overlay",
            detail: format!(
                concat!(
                    "invalid {:?} view offset={} bytes={} ",
                    "range_bytes={} backing={}"
                ),
                dtype, range.offset, bytes, range.bytes, backing_bytes
            ),
        });
    }
    Ok(MetalTensor {
        buffer: backing.buffer.clone(),
        offset: range.offset,
        shape,
        dtype,
        provenance: backing.provenance,
    })
}

/// Explicit production-prefill topology overrides. Absent fields retain the
/// existing environment-selected behavior.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefillScratchConfig {
    pub matrix_query_cap: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillScratchAllocation {
    name: &'static str,
    logical_bytes: u64,
    dtype: GgmlType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillScratchPlan {
    allocations: Vec<PrefillScratchAllocation>,
    deferred_allocations: Vec<PrefillScratchAllocation>,
    logical_bytes: u64,
    deferred_logical_bytes: u64,
    block_size: u32,
    matrix_max_pos: u64,
    matrix_query_rows: u32,
    overlay: Option<PrefillScratchOverlayStats>,
    modes: PrefillScratchPlanModes,
}

impl PrefillScratchPlan {
    pub fn allocations(&self) -> &[PrefillScratchAllocation] {
        &self.allocations
    }

    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    pub fn deferred_allocations(&self) -> &[PrefillScratchAllocation] {
        &self.deferred_allocations
    }

    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    pub fn maximum_logical_bytes(&self) -> Result<u64, MetalError> {
        self.logical_bytes
            .checked_add(self.deferred_logical_bytes)
            .ok_or_else(|| MetalError::BadShape {
                kernel: "prefill_scratch_plan",
                detail: "maximum logical byte total overflow".into(),
            })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn matrix_max_pos(&self) -> u64 {
        self.matrix_max_pos
    }

    pub fn matrix_query_rows(&self) -> u32 {
        self.matrix_query_rows
    }

    pub fn overlay(&self) -> Option<PrefillScratchOverlayStats> {
        self.overlay
    }

    fn validate_deferred(
        &self,
        name: &'static str,
        logical_bytes: u64,
        dtype: GgmlType,
    ) -> Result<(), MetalError> {
        if self.deferred_allocations.iter().any(|allocation| {
            allocation.name == name
                && allocation.logical_bytes == logical_bytes
                && allocation.dtype == dtype
        }) {
            return Ok(());
        }
        Err(MetalError::BadShape {
            kernel: "prefill_scratch_plan",
            detail: format!("unplanned deferred allocation {name}={logical_bytes}"),
        })
    }

    pub fn priced_upper_bound(
        &self,
        mut price: impl FnMut(u64) -> Result<u64, MetalError>,
    ) -> Result<u64, MetalError> {
        self.allocations
            .iter()
            .chain(&self.deferred_allocations)
            .try_fold(0u64, |total, allocation| {
                total
                    .checked_add(price(allocation.logical_bytes)?)
                    .ok_or_else(|| MetalError::BadShape {
                        kernel: "prefill_scratch_plan",
                        detail: "priced allocation total overflow".into(),
                    })
            })
    }
}

impl PrefillScratchAllocation {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    pub fn dtype(&self) -> GgmlType {
        self.dtype
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PrefillScratchPlanModes {
    enable_attn_packed: bool,
    enable_attn_fused_qkv_g8: bool,
    enable_attn_matrix: bool,
    attn_matrix_max_pos: u64,
    attn_matrix_online: bool,
    attn_matrix_query_cap: Option<usize>,
    overlay_allowed: bool,
}

struct PrefillScratchPlanBuilder {
    allocations: Vec<PrefillScratchAllocation>,
    logical_bytes: u64,
}

impl PrefillScratchPlanBuilder {
    fn new() -> Self {
        Self {
            allocations: Vec::new(),
            logical_bytes: 0,
        }
    }

    fn add(
        &mut self,
        name: &'static str,
        elements: u64,
        dtype: GgmlType,
    ) -> Result<(), MetalError> {
        let element_bytes = match dtype {
            GgmlType::F16 => 2,
            GgmlType::F32 | GgmlType::I32 => 4,
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "prefill_scratch_plan",
                    detail: format!("unsupported planned dtype {dtype:?}"),
                });
            }
        };
        let logical_bytes = checked_u64_mul(
            elements,
            element_bytes,
            "prefill scratch allocation bytes overflow",
        )?;
        self.logical_bytes = checked_u64_add(
            self.logical_bytes,
            logical_bytes,
            "prefill scratch total bytes overflow",
        )?;
        self.allocations.push(PrefillScratchAllocation {
            name,
            logical_bytes,
            dtype,
        });
        Ok(())
    }

    fn f32(&mut self, name: &'static str, elements: u64) -> Result<(), MetalError> {
        self.add(name, elements, GgmlType::F32)
    }

    fn f16(&mut self, name: &'static str, elements: u64) -> Result<(), MetalError> {
        self.add(name, elements, GgmlType::F16)
    }

    fn i32(&mut self, name: &'static str, elements: u64) -> Result<(), MetalError> {
        self.add(name, elements, GgmlType::I32)
    }
}

struct PrefillScratchAllocator<'a> {
    plan: &'a PrefillScratchPlan,
    next: usize,
}

impl<'a> PrefillScratchAllocator<'a> {
    fn new(plan: &'a PrefillScratchPlan) -> Self {
        Self { plan, next: 0 }
    }

    fn allocate(
        &mut self,
        ctx: &MetalContext,
        name: &'static str,
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> Result<MetalTensor, MetalError> {
        let element_bytes = match dtype {
            GgmlType::F16 => 2,
            GgmlType::F32 | GgmlType::I32 => 4,
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "prefill_scratch_plan",
                    detail: format!("unsupported planned dtype {dtype:?}"),
                });
            }
        };
        let elements = shape.iter().try_fold(1u64, |total, &dim| {
            checked_u64_mul(total, dim, "prefill planned tensor shape overflow")
        })?;
        let logical_bytes = checked_u64_mul(
            elements,
            element_bytes,
            "prefill planned tensor byte size overflow",
        )?;
        let expected =
            self.plan
                .allocations
                .get(self.next)
                .ok_or_else(|| MetalError::BadShape {
                    kernel: "prefill_scratch_plan",
                    detail: format!("unexpected allocation {name} after plan end"),
                })?;
        if expected.name != name
            || expected.logical_bytes != logical_bytes
            || expected.dtype != dtype
        {
            return Err(MetalError::BadShape {
                kernel: "prefill_scratch_plan",
                detail: format!(
                    "allocation {name}={logical_bytes}/{dtype:?} does not match {}={}/{:?}",
                    expected.name, expected.logical_bytes, expected.dtype
                ),
            });
        }
        self.next += 1;
        match dtype {
            GgmlType::F16 => MetalTensor::zeros_f16(ctx, shape),
            GgmlType::F32 => MetalTensor::zeros_f32(ctx, shape),
            GgmlType::I32 => MetalTensor::zeros_i32(ctx, shape),
            _ => unreachable!(),
        }
    }

    fn f32(
        &mut self,
        ctx: &MetalContext,
        name: &'static str,
        shape: Vec<u64>,
    ) -> Result<MetalTensor, MetalError> {
        self.allocate(ctx, name, shape, GgmlType::F32)
    }

    fn f16(
        &mut self,
        ctx: &MetalContext,
        name: &'static str,
        shape: Vec<u64>,
    ) -> Result<MetalTensor, MetalError> {
        self.allocate(ctx, name, shape, GgmlType::F16)
    }

    fn i32(
        &mut self,
        ctx: &MetalContext,
        name: &'static str,
        shape: Vec<u64>,
    ) -> Result<MetalTensor, MetalError> {
        self.allocate(ctx, name, shape, GgmlType::I32)
    }

    fn finish(self) -> Result<(), MetalError> {
        if self.next != self.plan.allocations.len() {
            return Err(MetalError::BadShape {
                kernel: "prefill_scratch_plan",
                detail: format!(
                    "constructor used {} of {} planned allocations",
                    self.next,
                    self.plan.allocations.len()
                ),
            });
        }
        Ok(())
    }
}

fn build_prefill_scratch_plan_from_arch(
    arch: &crate::model::Arch,
    n_attn_layers: u64,
    has_gdn: bool,
    block_size: u32,
    include_spec_packs: bool,
    modes: PrefillScratchPlanModes,
) -> Result<PrefillScratchPlan, MetalError> {
    let n = u64::from(block_size);
    let h = u64::from(arch.hidden_size);
    let f = u64::from(arch.intermediate_size);
    let head_dim = u64::from(arch.attn_head_dim);
    let n_q = u64::from(arch.n_q_heads);
    let n_kv = u64::from(arch.n_kv_heads);
    let expert_count = u64::from(arch.expert_count).max(1);
    let q_dim = checked_u64_mul(n_q, head_dim, "prefill plan q_dim overflow")?;
    let kv_dim = checked_u64_mul(n_kv, head_dim, "prefill plan kv_dim overflow")?;
    let attn_q_full_dim = checked_u64_double(q_dim, "prefill plan 2*q_dim overflow")?;
    let attn_qkv_fused_dim = checked_u64_add(
        attn_q_full_dim,
        checked_u64_double(kv_dim, "prefill plan 2*kv_dim overflow")?,
        "prefill plan fused qkv dim overflow",
    )?;
    let moe_topk = u64::from(arch.expert_used_count.min(arch.expert_count)).max(1);
    let moe_f_exp = u64::from(arch.expert_feed_forward_length).max(1);
    let moe_f_shared = u64::from(arch.expert_shared_feed_forward_length).max(1);
    let moe_slot_elems = checked_u64_mul(n, moe_topk, "prefill plan moe slots overflow")?;
    let moe_inner_elems =
        checked_u64_mul(moe_slot_elems, moe_f_exp, "prefill plan moe inner overflow")?;
    let moe_out_elems = checked_u64_mul(moe_slot_elems, h, "prefill plan moe output overflow")?;
    let moe_group_ids_elems =
        checked_u64_mul(expert_count, n, "prefill plan moe group ids overflow")?;
    let attn_group = n_q.checked_div(n_kv.max(1)).unwrap_or(1).max(1);
    let attn_packed_rows = if include_spec_packs && attn_group == 6 && block_size == 2 {
        n
    } else {
        ATTN_PREFILL_V4_PACKED_ROWS as u64
    };
    let attn_partial_group = checked_u64_mul(
        attn_group,
        head_dim,
        "prefill plan partial attention group overflow",
    )?;
    let attn_prefill_v4_o_partial_elems = checked_u64_mul4(
        attn_packed_rows,
        n_kv.max(1),
        ATTN_V4_MAX_NWG as u64,
        attn_partial_group,
        "prefill plan attention partial overflow",
    )?;
    let attn_prefill_v4_ml_partial_elems = checked_u64_mul4(
        attn_packed_rows,
        n_kv.max(1),
        ATTN_V4_MAX_NWG as u64,
        checked_u64_double(attn_group, "prefill plan attention ml group overflow")?,
        "prefill plan attention ml partial overflow",
    )?;

    let query_cap = if include_spec_packs {
        None
    } else {
        modes.attn_matrix_query_cap
    };
    if query_cap == Some(0) {
        return Err(MetalError::BadShape {
            kernel: "prefill_scratch_plan",
            detail: "matrix query cap must be greater than zero".into(),
        });
    }
    if modes.enable_attn_matrix && !modes.attn_matrix_online && query_cap.is_some() {
        return Err(MetalError::BadShape {
            kernel: "prefill_scratch_plan",
            detail: "matrix query cap requires online matrix attention".into(),
        });
    }
    let matrix_max_pos = if modes.enable_attn_matrix {
        modes.attn_matrix_max_pos.max(n)
    } else {
        0
    };
    let query_rows = query_cap.map(|cap| n.min(cap as u64)).unwrap_or(n).max(1);
    let matrix_scores_elems = if modes.enable_attn_matrix && !modes.attn_matrix_online {
        checked_u64_mul3(
            n,
            n_q,
            matrix_max_pos,
            "prefill plan matrix scores overflow",
        )?
    } else {
        1
    };
    let matrix_scores_h_elems = if modes.enable_attn_matrix && modes.attn_matrix_online {
        checked_u64_mul3(
            query_rows,
            n_q,
            matrix_max_pos,
            "prefill plan online matrix scores overflow",
        )?
    } else {
        1
    };
    let matrix_ml_elems = if modes.enable_attn_matrix && modes.attn_matrix_online {
        checked_u64_mul3(
            query_rows,
            n_q,
            checked_u64_double(
                matrix_max_pos.div_ceil(64),
                "prefill plan matrix ml tiles overflow",
            )?,
            "prefill plan matrix ml overflow",
        )?
    } else {
        1
    };
    let matrix_vt_elems = if modes.enable_attn_matrix {
        checked_u64_mul4(
            n_attn_layers.max(1),
            n_kv,
            head_dim,
            matrix_max_pos,
            "prefill plan matrix vt overflow",
        )?
    } else {
        1
    };

    let gdn_head_dim = u64::from(arch.gdn_head_dim);
    let gdn_n_v = u64::from(arch.gdn_n_v_heads);
    let gdn_n_k = u64::from(arch.gdn_n_k_heads);
    let gdn_v_dim = checked_u64_mul(
        gdn_n_v.max(1),
        gdn_head_dim.max(1),
        "prefill plan gdn v dim overflow",
    )?;
    let gdn_conv_heads = checked_u64_add(
        checked_u64_double(gdn_n_k.max(1), "prefill plan gdn 2*n_k overflow")?,
        gdn_n_v.max(1),
        "prefill plan gdn conv heads overflow",
    )?;
    let gdn_conv_dim = checked_u64_mul(
        gdn_conv_heads,
        gdn_head_dim.max(1),
        "prefill plan gdn conv dim overflow",
    )?;
    let gdn_k_dim = checked_u64_mul(
        gdn_n_k.max(1),
        gdn_head_dim.max(1),
        "prefill plan gdn k dim overflow",
    )?;
    let gdn_shapes = [
        checked_u64_mul(n, gdn_conv_dim, "prefill plan gdn qkv overflow")?,
        checked_u64_mul(n, gdn_v_dim, "prefill plan gdn z overflow")?,
        checked_u64_mul(n, gdn_n_v.max(1), "prefill plan gdn beta overflow")?,
        checked_u64_mul(n, gdn_n_v.max(1), "prefill plan gdn alpha overflow")?,
        checked_u64_mul(n, gdn_k_dim, "prefill plan gdn q norm overflow")?,
        checked_u64_mul(n, gdn_k_dim, "prefill plan gdn k norm overflow")?,
        checked_u64_mul(n, gdn_v_dim, "prefill plan gdn v overflow")?,
        checked_u64_mul(n, gdn_v_dim, "prefill plan gdn out overflow")?,
        checked_u64_mul(n, gdn_v_dim, "prefill plan gdn normed overflow")?,
    ];
    let overlay_layout = (modes.overlay_allowed
        && !include_spec_packs
        && modes.enable_attn_matrix
        && modes.attn_matrix_online
        && query_cap.is_some()
        && has_gdn)
        .then(|| prefill_scratch_overlay_layout(matrix_scores_h_elems, matrix_ml_elems, gdn_shapes))
        .transpose()?;
    let overlay = if let Some(layout) = overlay_layout {
        let saved_bytes = layout
            .attention_bytes
            .checked_add(layout.gdn_bytes)
            .and_then(|sum| sum.checked_sub(layout.backing_bytes))
            .ok_or_else(|| MetalError::BadShape {
                kernel: "prefill_scratch_plan",
                detail: "overlay savings arithmetic overflowed".into(),
            })?;
        Some(PrefillScratchOverlayStats {
            backing_bytes: layout.backing_bytes,
            attention_bytes: layout.attention_bytes,
            gdn_bytes: layout.gdn_bytes,
            saved_bytes,
        })
    } else {
        None
    };

    let mut builder = PrefillScratchPlanBuilder::new();
    if let Some(stats) = overlay {
        if !stats.backing_bytes.is_multiple_of(4) {
            return Err(MetalError::BadShape {
                kernel: "prefill_scratch_plan",
                detail: format!(
                    "overlay backing size {} is not F32-aligned",
                    stats.backing_bytes
                ),
            });
        }
        builder.f32("attn_gdn_overlay_backing", stats.backing_bytes / 4)?;
    } else {
        builder.f16("attn_matrix_scores_h_pack", matrix_scores_h_elems)?;
        builder.f32("attn_matrix_ml_pack", matrix_ml_elems)?;
        for (name, elements) in [
            "gdn_qkv_pack",
            "gdn_z_pack",
            "gdn_beta_pack",
            "gdn_alpha_pack",
            "gdn_q_norm_pack",
            "gdn_k_norm_pack",
            "gdn_v_pack",
            "gdn_out_pack",
            "gdn_normed_pack",
        ]
        .into_iter()
        .zip(gdn_shapes)
        {
            builder.f32(name, elements)?;
        }
    }

    let nh = checked_u64_mul(n, h, "prefill plan n*h overflow")?;
    let nf = checked_u64_mul(n, f, "prefill plan n*f overflow")?;
    let nq = checked_u64_mul(n, q_dim, "prefill plan n*q overflow")?;
    let nkv = checked_u64_mul(n, kv_dim, "prefill plan n*kv overflow")?;
    let n_q_full = checked_u64_mul(n, attn_q_full_dim, "prefill plan n*q_full overflow")?;
    let n_qkv_fused = checked_u64_mul(n, attn_qkv_fused_dim, "prefill plan n*qkv_fused overflow")?;
    let n_experts = checked_u64_mul(n, expert_count, "prefill plan n*experts overflow")?;
    let n_shared = checked_u64_mul(n, moe_f_shared, "prefill plan n*shared overflow")?;
    for (name, elements) in [
        ("x_pack", nh),
        ("h_pack", nh),
        ("mixer_out_pack", nh),
        (
            "attn_qkv_fused_pack",
            if modes.enable_attn_fused_qkv_g8 {
                n_qkv_fused
            } else {
                1
            },
        ),
        ("attn_q_full_pack", n_q_full),
        ("attn_q_pack", 1),
        ("attn_gate_pack", 1),
        ("attn_q_normed_pack", nq),
        ("attn_k_now_pack", nkv),
        ("attn_v_now_pack", nkv),
        ("attn_k_normed_pack", nkv),
        ("attn_o_pack", nq),
        (
            "attn_prefill_v4_o_partial_pack",
            if modes.enable_attn_packed {
                attn_prefill_v4_o_partial_elems
            } else {
                1
            },
        ),
        (
            "attn_prefill_v4_ml_partial_pack",
            if modes.enable_attn_packed {
                attn_prefill_v4_ml_partial_elems
            } else {
                1
            },
        ),
        ("attn_matrix_scores_pack", matrix_scores_elems),
    ] {
        builder.f32(name, elements)?;
    }
    builder.f16("attn_matrix_vt_pack", matrix_vt_elems)?;
    for (name, elements) in [
        ("ffn_gate_pack", nf),
        ("ffn_up_pack", nf),
        ("ffn_inner_pack", nf),
        ("ffn_out_pack", nh),
        ("moe_topk_idx_pack", moe_slot_elems),
        ("moe_router_probs_pack", n_experts),
        ("moe_topk_weight_pack", moe_slot_elems),
        ("moe_shared_gate_pack", n),
        (
            "moe_inner_pack",
            if include_spec_packs {
                moe_inner_elems
            } else {
                1
            },
        ),
        (
            "moe_expert_out_pack",
            if include_spec_packs { moe_out_elems } else { 1 },
        ),
        ("moe_group_slot_idx_pack", moe_slot_elems),
        ("moe_group_count_pack", expert_count),
        ("moe_group_ids_pack", moe_group_ids_elems),
        ("moe_group_token_idx_pack", moe_slot_elems),
        ("moe_group_weight_pack", moe_slot_elems),
        ("moe_group_inner_pack", moe_inner_elems),
        ("moe_group_out_pack", moe_out_elems),
        ("moe_shared_ffn_gate_pack", n_shared),
        ("moe_shared_ffn_up_pack", n_shared),
        ("moe_shared_ffn_inner_pack", n_shared),
        ("moe_shared_ffn_out_pack", nh),
        (
            "final_logits_pack",
            if include_spec_packs {
                checked_u64_mul(
                    n,
                    u64::from(arch.vocab_size),
                    "prefill plan final logits overflow",
                )?
            } else {
                1
            },
        ),
    ] {
        if name == "moe_group_slot_idx_pack" {
            builder.i32(name, elements)?;
        } else {
            builder.f32(name, elements)?;
        }
    }
    let mut deferred = PrefillScratchPlanBuilder::new();
    if !include_spec_packs {
        deferred.f32("moe_inner_pack_fallback_growth", moe_inner_elems)?;
        deferred.f32("moe_expert_out_pack_fallback_growth", moe_out_elems)?;
    }
    let matrix_query_rows = u32::try_from(query_rows).map_err(|_| MetalError::BadShape {
        kernel: "prefill_scratch_plan",
        detail: format!("matrix query rows {query_rows} do not fit u32"),
    })?;
    Ok(PrefillScratchPlan {
        allocations: builder.allocations,
        deferred_allocations: deferred.allocations,
        logical_bytes: builder.logical_bytes,
        deferred_logical_bytes: deferred.logical_bytes,
        block_size,
        matrix_max_pos,
        matrix_query_rows,
        overlay,
        modes,
    })
}

fn resolve_prefill_scratch_plan_modes(
    arch: &crate::model::Arch,
    block_size: u32,
    include_spec_packs: bool,
    matrix_max_pos_override: Option<usize>,
    config: Option<PrefillScratchConfig>,
) -> Result<PrefillScratchPlanModes, MetalError> {
    let head_dim = arch.attn_head_dim as usize;
    let n_kv = arch.n_kv_heads.max(1);
    let attn_group = arch.n_q_heads.checked_div(n_kv).unwrap_or(1).max(1) as usize;
    let enable_attn_packed = matches!(
        std::env::var("QWEN_PREFILL_ATTN_PACKED_G8").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    ) || matches!(
        std::env::var("QWEN_PREFILL_ATTN_PACKED_G16").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    ) || (head_dim == 256 && matches!(attn_group, 8 | 16))
        || (include_spec_packs
            && block_size == 2
            && head_dim == 256
            && attn_group == 6
            && mtp_attn_q2_shared_kv_enabled())
        // V1: same packs, any verify chain length up to the pack's row
        // capacity (ATTN_PREFILL_V4_PACKED_ROWS = 8, which is what the
        // non-`block_size == 2` sizing branch already allocates).
        || (include_spec_packs
            && head_dim == 256
            && attn_group == 6
            && mtp_attn_qn_shared_kv_enabled());
    let enable_attn_fused_qkv_g8 = matches!(
        std::env::var("QWEN_PREFILL_ATTN_FUSED_QKV_G8").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    );
    let enable_attn_matrix = head_dim == 256
        && ((prefill_attn_matrix_g4_may_use()
            && arch.n_kv_heads.checked_mul(4) == Some(arch.n_q_heads))
            || (prefill_attn_matrix_g8_may_use() && arch.n_q_heads == 16 && arch.n_kv_heads == 2)
            || (prefill_attn_matrix_g6_may_use() && arch.n_q_heads == 24 && arch.n_kv_heads == 4)
            || (prefill_attn_matrix_g16_may_use() && arch.n_q_heads == 32 && arch.n_kv_heads == 2));
    let attn_matrix_max_pos = if enable_attn_matrix {
        prefill_attn_matrix_max_pos()
            .or(matrix_max_pos_override)
            .unwrap_or(block_size as usize)
            .max(block_size as usize) as u64
    } else {
        0
    };
    let attn_matrix_query_cap = resolve_prefill_attn_matrix_query_cap(include_spec_packs, config)
        .map_err(|detail| MetalError::BadShape {
        kernel: "prefill_attn_matrix_query_cap",
        detail,
    })?;
    Ok(PrefillScratchPlanModes {
        enable_attn_packed,
        enable_attn_fused_qkv_g8,
        enable_attn_matrix,
        attn_matrix_max_pos,
        attn_matrix_online: prefill_attn_matrix_online_enabled(),
        attn_matrix_query_cap,
        overlay_allowed: prefill_attn_gdn_scratch_overlay_enabled()
            && !prefill_scratch_overlay_diagnostic_mode_present(),
    })
}

pub fn plan_prefill_scratch_with_matrix_max_pos_configured(
    target_model: &crate::metal_forward::MetalModel,
    block_size: u32,
    matrix_max_pos: usize,
    config: PrefillScratchConfig,
) -> Result<PrefillScratchPlan, MetalError> {
    let n_attn_layers = u64::try_from(
        target_model
            .blocks
            .iter()
            .filter(|block| matches!(block, MetalBlock::Attn(_)))
            .count()
            .max(1),
    )
    .map_err(|_| MetalError::BadShape {
        kernel: "prefill_scratch_plan",
        detail: "attention layer count does not fit u64".into(),
    })?;
    let has_gdn = target_model
        .blocks
        .iter()
        .any(|block| matches!(block, MetalBlock::Gdn(_)));
    let modes = resolve_prefill_scratch_plan_modes(
        &target_model.arch,
        block_size,
        false,
        Some(matrix_max_pos),
        Some(config),
    )?;
    build_prefill_scratch_plan_from_arch(
        &target_model.arch,
        n_attn_layers,
        has_gdn,
        block_size,
        false,
        modes,
    )
}

pub struct MetalDFlashLayerMajorScratch {
    /// `[N, H]` F32 — residual stream across N tokens.
    pub x_pack: MetalTensor,
    /// `[N, H]` F32 — post-norm activation across N tokens (reused for
    /// both pre-attn and pre-FFN norms).
    pub h_pack: MetalTensor,
    /// `[N, H]` F32 — mixer output (GDN or attn).
    pub mixer_out_pack: MetalTensor,

    // Attention scratch (only meaningful on attn layers).
    /// `[N, 2·q_dim + 2·kv_dim]` F32 — fused QKV front projection for the
    /// experimental group-8 packed-prefill path.
    pub attn_qkv_fused_pack: MetalTensor,
    /// `[N, 2·q_dim]` F32 — gated Q projection (Q + gate interleaved).
    pub attn_q_full_pack: MetalTensor,
    /// v0.447: stubbed at 1 element. The strided q-norm + fused gate
    /// epilogue (v0.432) read `attn_q_full_pack` in place; no production
    /// path splits Q/gate into these packs anymore (the only remaining
    /// split-path user is a test-module profile helper with local
    /// buffers).
    pub attn_q_pack: MetalTensor,
    /// v0.447: stubbed at 1 element (see `attn_q_pack`).
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
    /// `[R, n_kv, NWG, group, head_dim]` F32 — packed-prompt attention partials
    /// for the experimental A3B/group-8 prompt-native microproof.
    pub attn_prefill_v4_o_partial_pack: MetalTensor,
    /// `[R, n_kv, NWG, group, 2]` F32 — packed-prompt attention `(m, l)`
    /// partials for the same microproof.
    pub attn_prefill_v4_ml_partial_pack: MetalTensor,
    /// `[N * n_q_heads, matrix_max_pos]` F32 — score/prob scratch for the
    /// three-kernel matrix attention sidecar. Stubbed at 1 element when the
    /// two-pass online path is enabled (the default).
    pub attn_matrix_scores_pack: MetalTensor,
    /// `[matrix_query_rows * n_q_heads, matrix_max_pos]` F16 —
    /// `P~ = exp2(s*scale - m_tile)` scratch for the two-pass online matrix
    /// attention path. Stubbed at 1 element when the online path is disabled.
    pub attn_matrix_scores_h_pack: MetalTensor,
    /// `[matrix_query_rows * n_q_heads, ceil(matrix_max_pos/64), 2]` F32 —
    /// per-(query, tile) (m, l) sidecar for online matrix attention.
    pub attn_matrix_ml_pack: MetalTensor,
    /// `[n_attn_layers, n_kv_heads, head_dim, matrix_max_pos]` F16 — persistent
    /// transposed V-cache view used by both matrix attention variants.
    pub attn_matrix_vt_pack: MetalTensor,

    // FFN scratch.
    /// `[N, F]` F32 — FFN gate output (skipped when fused Q4_K SwiGLU is used).
    pub ffn_gate_pack: MetalTensor,
    /// `[N, F]` F32 — FFN up output.
    pub ffn_up_pack: MetalTensor,
    /// `[N, F]` F32 — silu(gate) * up.
    pub ffn_inner_pack: MetalTensor,
    /// `[N, H]` F32 — FFN final.
    pub ffn_out_pack: MetalTensor,

    /// `[N * topk]` i32-in-F32 buffer — packed routed expert ids per token.
    pub moe_topk_idx_pack: MetalTensor,
    /// `[N, n_expert]` F32 — packed router logits per token.
    pub moe_router_probs_pack: MetalTensor,
    /// `[N * topk]` F32 — packed routed expert weights per token.
    pub moe_topk_weight_pack: MetalTensor,
    /// `[N]` F32 — packed shared expert gate per token.
    pub moe_shared_gate_pack: MetalTensor,
    /// `[N * topk, F_exp]` F32 — packed routed expert inner activations.
    ///
    /// v0.431: only the packed-slot FALLBACK branches (exotic quant
    /// combinations, `QWEN_PREFILL_MOE_HOT_*` / packed-down-sum env
    /// overrides) read this; the default grouped path uses
    /// `moe_group_inner_pack`. Production prefill (`fresh_prefill*`)
    /// therefore allocates a 1-element stub and lazily grows it via
    /// [`Self::ensure_moe_packed_fallback`] the first time a fallback
    /// branch runs — saving ~17 MB (A3B) / ~34 MB (A10B) resident per
    /// scratch. Test/spec constructors (`fresh`) keep the full
    /// allocation so direct field access in tests stays valid.
    pub moe_inner_pack: MetalTensor,
    /// `[N * topk, H]` F32 — packed routed expert outputs before tokenwise
    /// reduction. Same fallback-only story as `moe_inner_pack`: production
    /// prefill stubs it (~67 MB A3B / ~128 MB A10B saved) and lazily grows
    /// on first fallback use.
    pub moe_expert_out_pack: MetalTensor,
    /// Full sizes for the two fallback packs above, kept so
    /// `ensure_moe_packed_fallback` can grow stubs without re-deriving
    /// arch math.
    moe_inner_full_elems: u64,
    moe_out_full_elems: u64,
    /// `[N * topk]` I32 — grouped routed slot ids used for row gathers.
    pub moe_group_slot_idx_pack: MetalTensor,
    /// `[n_expert]` i32-in-F32 buffer — grouped routed slot counts per expert.
    pub moe_group_count_pack: MetalTensor,
    /// `[n_expert * N]` i32-in-F32 buffer — grouped routed slot ids per expert.
    pub moe_group_ids_pack: MetalTensor,
    /// `[N * topk]` i32-in-F32 buffer — grouped destination token ids.
    pub moe_group_token_idx_pack: MetalTensor,
    /// `[N * topk]` F32 — grouped routed expert weights.
    pub moe_group_weight_pack: MetalTensor,
    /// `[N * topk, F_exp]` F32 — grouped routed expert inner rows.
    pub moe_group_inner_pack: MetalTensor,
    /// `[N * topk, H]` F32 — grouped routed expert down outputs.
    pub moe_group_out_pack: MetalTensor,
    /// `[N, F_shared]` F32 — packed shared-expert gate projection.
    pub moe_shared_ffn_gate_pack: MetalTensor,
    /// `[N, F_shared]` F32 — packed shared-expert up projection.
    pub moe_shared_ffn_up_pack: MetalTensor,
    /// `[N, F_shared]` F32 — packed shared-expert SwiGLU inner activations.
    pub moe_shared_ffn_inner_pack: MetalTensor,
    /// `[N, H]` F32 — packed shared-expert down projection before rowwise gate.
    pub moe_shared_ffn_out_pack: MetalTensor,

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
    /// `[N, n_v]` F32 — batched beta projection after sigmoid.
    pub gdn_beta_pack: MetalTensor,
    /// `[N, n_v]` F32 — batched alpha decay exp(g).
    pub gdn_alpha_pack: MetalTensor,
    /// `[N, n_k * head_dim]` F32 — packed l2-normed Q rows for step packing.
    pub gdn_q_norm_pack: MetalTensor,
    /// `[N, n_k * head_dim]` F32 — packed l2-normed K rows for step packing.
    pub gdn_k_norm_pack: MetalTensor,
    /// `[N, v_dim]` F32 — packed V rows for step packing.
    pub gdn_v_pack: MetalTensor,
    /// `[N, v_dim]` F32 — packed recurrence outputs before rmsnorm_gated.
    pub gdn_out_pack: MetalTensor,
    /// `[N, v_dim]` F32 — RMSNormGated output across N tokens; consumed by
    /// the batched out_proj mat-mat after the per-token recurrence loop.
    /// 27B: [16, 6144] = 384 KiB.
    pub gdn_normed_pack: MetalTensor,

    // Cached dims so callers don't have to re-derive.
    pub n: u32,
    /// Maximum query rows backed by online matrix-attention score scratch.
    /// This can be smaller than `n`; every other packed prefill buffer retains
    /// the outer chunk width.
    pub(crate) attn_matrix_query_rows: u32,
    attn_matrix_tiled_layer_calls: u64,
    attn_matrix_query_tile_calls: u64,
    scratch_overlay: Option<PrefillScratchOverlayStats>,
    scratch_plan: PrefillScratchPlan,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub q_dim: u64,
    pub kv_dim: u64,
    pub vocab_size: u64,
    /// Maximum `n_pos` covered by the experimental matrix-attention scratch.
    /// Zero when the sidecar is not allocated.
    pub attn_matrix_max_pos: u64,
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
    fn mutable_tensors(&self) -> impl Iterator<Item = &MetalTensor> {
        [
            &self.x_pack,
            &self.h_pack,
            &self.mixer_out_pack,
            &self.attn_qkv_fused_pack,
            &self.attn_q_full_pack,
            &self.attn_q_pack,
            &self.attn_gate_pack,
            &self.attn_q_normed_pack,
            &self.attn_k_now_pack,
            &self.attn_v_now_pack,
            &self.attn_k_normed_pack,
            &self.attn_o_pack,
            &self.attn_prefill_v4_o_partial_pack,
            &self.attn_prefill_v4_ml_partial_pack,
            &self.attn_matrix_scores_pack,
            &self.attn_matrix_scores_h_pack,
            &self.attn_matrix_ml_pack,
            &self.attn_matrix_vt_pack,
            &self.ffn_gate_pack,
            &self.ffn_up_pack,
            &self.ffn_inner_pack,
            &self.ffn_out_pack,
            &self.moe_topk_idx_pack,
            &self.moe_router_probs_pack,
            &self.moe_topk_weight_pack,
            &self.moe_shared_gate_pack,
            &self.moe_inner_pack,
            &self.moe_expert_out_pack,
            &self.moe_group_slot_idx_pack,
            &self.moe_group_count_pack,
            &self.moe_group_ids_pack,
            &self.moe_group_token_idx_pack,
            &self.moe_group_weight_pack,
            &self.moe_group_inner_pack,
            &self.moe_group_out_pack,
            &self.moe_shared_ffn_gate_pack,
            &self.moe_shared_ffn_up_pack,
            &self.moe_shared_ffn_inner_pack,
            &self.moe_shared_ffn_out_pack,
            &self.final_logits_pack,
            &self.gdn_qkv_pack,
            &self.gdn_z_pack,
            &self.gdn_beta_pack,
            &self.gdn_alpha_pack,
            &self.gdn_q_norm_pack,
            &self.gdn_k_norm_pack,
            &self.gdn_v_pack,
            &self.gdn_out_pack,
            &self.gdn_normed_pack,
        ]
        .into_iter()
    }

    fn aliases_mutable_buffer(&self, candidate: &MetalTensor) -> bool {
        let candidate_id = Retained::as_ptr(&candidate.buffer) as *const () as usize;
        self.mutable_tensors()
            .any(|tensor| Retained::as_ptr(&tensor.buffer) as *const () as usize == candidate_id)
    }

    pub fn mutable_buffer_ids(&self) -> Vec<usize> {
        let mut ids = self
            .mutable_tensors()
            .map(|tensor| Retained::as_ptr(&tensor.buffer) as *const () as usize)
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    pub fn aliases_scratch(&self, other: &Self) -> bool {
        let own = self.mutable_buffer_ids();
        let other = other.mutable_buffer_ids();
        own.iter().any(|id| other.binary_search(id).is_ok())
    }

    fn fresh_inner(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
        // True for spec-decode/test scratch (`fresh`): allocate the full
        // `[N, V]` logits pack AND the packed-slot MoE fallback packs
        // eagerly (tests poke the fields directly). False for production
        // prompt prefill (`fresh_prefill*`): logits pack is a stub (the
        // tail uses per-token mat-vec) and the MoE fallback packs start
        // as stubs, lazily grown by `ensure_moe_packed_fallback` only if
        // a fallback branch actually runs.
        include_spec_packs: bool,
        attn_matrix_max_pos_override: Option<usize>,
        config: Option<PrefillScratchConfig>,
        plan_override: Option<PrefillScratchPlan>,
    ) -> Result<Self, MetalError> {
        let arch = &target_model.arch;
        let n = block_size as u64;
        let h = arch.hidden_size as u64;
        let f = arch.intermediate_size as u64;
        let head_dim = arch.attn_head_dim as u64;
        let q_dim = checked_u64_mul(
            arch.n_q_heads as u64,
            head_dim,
            "layer-major q_dim overflow",
        )?;
        let kv_dim = checked_u64_mul(
            arch.n_kv_heads as u64,
            head_dim,
            "layer-major kv_dim overflow",
        )?;
        let v = arch.vocab_size as u64;
        let n_attn_layers = target_model
            .blocks
            .iter()
            .filter(|b| matches!(b, crate::metal_forward::MetalBlock::Attn(_)))
            .count()
            .max(1) as u64;
        let moe_topk = arch.expert_used_count.min(arch.expert_count).max(1) as u64;
        let moe_f_exp = arch.expert_feed_forward_length.max(1) as u64;
        let moe_f_shared = arch.expert_shared_feed_forward_length.max(1) as u64;
        let attn_q_full_dim = checked_u64_double(q_dim, "layer-major 2 * q_dim overflow")?;
        let attn_qkv_fused_dim = checked_u64_add(
            attn_q_full_dim,
            checked_u64_double(kv_dim, "layer-major 2 * kv_dim overflow")?,
            "layer-major attn qkv fused dim overflow",
        )?;
        let moe_slot_elems = checked_u64_mul(n, moe_topk, "layer-major moe slot count overflow")?;
        let moe_inner_elems = checked_u64_mul(
            moe_slot_elems,
            moe_f_exp,
            "layer-major moe inner size overflow",
        )?;
        let moe_out_elems =
            checked_u64_mul(moe_slot_elems, h, "layer-major moe out size overflow")?;
        let moe_group_ids_elems = checked_u64_mul(
            (arch.expert_count as u64).max(1),
            n,
            "layer-major moe group ids size overflow",
        )?;
        let attn_group = (arch.n_q_heads as u64)
            .checked_div((arch.n_kv_heads as u64).max(1))
            .unwrap_or(1)
            .max(1);
        let attn_packed_rows = if include_spec_packs && attn_group == 6 && block_size == 2 {
            n
        } else {
            ATTN_PREFILL_V4_PACKED_ROWS as u64
        };
        let attn_prefill_v4_o_partial_elems = checked_u64_mul4(
            attn_packed_rows,
            (arch.n_kv_heads as u64).max(1),
            ATTN_V4_MAX_NWG as u64,
            checked_u64_mul(
                attn_group,
                head_dim,
                "layer-major attn prefill partial group*head overflow",
            )?,
            "layer-major attn prefill partial size overflow",
        )?;
        let attn_prefill_v4_ml_partial_elems = checked_u64_mul4(
            attn_packed_rows,
            (arch.n_kv_heads as u64).max(1),
            ATTN_V4_MAX_NWG as u64,
            checked_u64_mul(
                attn_group,
                2,
                "layer-major attn prefill ml partial group*2 overflow",
            )?,
            "layer-major attn prefill ml partial size overflow",
        )?;
        let has_gdn = target_model
            .blocks
            .iter()
            .any(|block| matches!(block, MetalBlock::Gdn(_)));
        let (modes, scratch_plan) = if let Some(plan) = plan_override {
            if include_spec_packs || plan.block_size != block_size {
                return Err(MetalError::BadShape {
                    kernel: "prefill_scratch_plan",
                    detail: "prefill scratch plan does not match constructor mode or width".into(),
                });
            }
            let expected = build_prefill_scratch_plan_from_arch(
                arch,
                n_attn_layers,
                has_gdn,
                block_size,
                false,
                plan.modes,
            )?;
            if expected != plan {
                return Err(MetalError::BadShape {
                    kernel: "prefill_scratch_plan",
                    detail: "prefill scratch plan does not match target model".into(),
                });
            }
            (plan.modes, plan)
        } else {
            let modes = resolve_prefill_scratch_plan_modes(
                arch,
                block_size,
                include_spec_packs,
                attn_matrix_max_pos_override,
                config,
            )?;
            let plan = build_prefill_scratch_plan_from_arch(
                arch,
                n_attn_layers,
                has_gdn,
                block_size,
                include_spec_packs,
                modes,
            )?;
            (modes, plan)
        };
        let enable_attn_packed = modes.enable_attn_packed;
        let enable_attn_fused_qkv_g8 = modes.enable_attn_fused_qkv_g8;
        let enable_attn_matrix = modes.enable_attn_matrix;
        let attn_matrix_max_pos = modes.attn_matrix_max_pos;
        // The F32 scores buffer backs the three-kernel sidecar (rollback);
        // the F16 P~ + (m, l) pair backs the default two-pass online path.
        // Whichever variant the process-level env selects is allocated in
        // full and the other is stubbed at 1 element.
        let attn_matrix_online = modes.attn_matrix_online;
        let attn_matrix_query_cap = modes.attn_matrix_query_cap;
        if enable_attn_matrix && !attn_matrix_online && attn_matrix_query_cap.is_some() {
            return Err(MetalError::BadShape {
                kernel: "prefill_attn_matrix_query_cap",
                detail: "QWEN_PREFILL_ATTN_MATRIX_QUERY_CAP requires online matrix attention"
                    .into(),
            });
        }
        let attn_matrix_query_rows = attn_matrix_query_cap
            .map(|cap| n.min(cap as u64))
            .unwrap_or(n)
            .max(1);
        let attn_matrix_scores_elems = if enable_attn_matrix && !attn_matrix_online {
            checked_u64_mul3(
                n,
                arch.n_q_heads as u64,
                attn_matrix_max_pos,
                "layer-major attn matrix scores size overflow",
            )?
        } else {
            1
        };
        let attn_matrix_scores_h_elems = if enable_attn_matrix && attn_matrix_online {
            checked_u64_mul3(
                attn_matrix_query_rows,
                arch.n_q_heads as u64,
                attn_matrix_max_pos,
                "layer-major attn matrix online scores size overflow",
            )?
        } else {
            1
        };
        let attn_matrix_ml_elems = if enable_attn_matrix && attn_matrix_online {
            checked_u64_mul3(
                attn_matrix_query_rows,
                arch.n_q_heads as u64,
                checked_u64_double(
                    attn_matrix_max_pos.div_ceil(64),
                    "layer-major attn matrix ml tiles overflow",
                )?,
                "layer-major attn matrix ml size overflow",
            )?
        } else {
            1
        };
        let attn_matrix_vt_elems = if enable_attn_matrix {
            checked_u64_mul(
                n_attn_layers,
                checked_u64_mul3(
                    arch.n_kv_heads as u64,
                    head_dim,
                    attn_matrix_max_pos,
                    "layer-major attn matrix vt layer size overflow",
                )?,
                "layer-major attn matrix vt bank size overflow",
            )?
        } else {
            1
        };

        // GDN dims. Sized at 1 element when the arch has no GDN to keep
        // the buffers allocatable; the GDN layer-major path is gated on
        // `gdn_mat_mat_eligible` and never reads from these on non-GDN
        // archs.
        let gdn_head_dim = arch.gdn_head_dim as u64;
        let gdn_n_v = arch.gdn_n_v_heads as u64;
        let gdn_n_k = arch.gdn_n_k_heads as u64;
        let gdn_v_dim = checked_u64_mul(
            gdn_n_v.max(1),
            gdn_head_dim.max(1),
            "layer-major gdn_v_dim overflow",
        )?;
        let gdn_conv_heads = checked_u64_add(
            checked_u64_double(gdn_n_k.max(1), "layer-major 2 * gdn_n_k overflow")?,
            gdn_n_v.max(1),
            "layer-major gdn conv heads overflow",
        )?;
        let gdn_conv_dim = checked_u64_mul(
            gdn_conv_heads,
            gdn_head_dim.max(1),
            "layer-major gdn_conv_dim overflow",
        )?;
        let gdn_k_dim = checked_u64_mul(
            gdn_n_k.max(1),
            gdn_head_dim.max(1),
            "layer-major gdn_k_dim overflow",
        )?;
        let final_logits_shape = if include_spec_packs {
            vec![n, v]
        } else {
            vec![1]
        };
        // v0.431: packed-slot MoE fallback packs are stubbed on the
        // production prefill path (grouped default never touches them)
        // and lazily grown on first fallback use. ~84 MB (A3B) /
        // ~160 MB (A10B) resident savings per prefill scratch.
        let moe_inner_shape = if include_spec_packs {
            vec![moe_inner_elems]
        } else {
            vec![1]
        };
        let moe_out_shape = if include_spec_packs {
            vec![moe_out_elems]
        } else {
            vec![1]
        };
        let gdn_shapes = [
            checked_u64_mul(n, gdn_conv_dim, "overlay gdn qkv elements overflow")?,
            checked_u64_mul(n, gdn_v_dim, "overlay gdn z elements overflow")?,
            checked_u64_mul(n, gdn_n_v.max(1), "overlay gdn beta elements overflow")?,
            checked_u64_mul(n, gdn_n_v.max(1), "overlay gdn alpha elements overflow")?,
            checked_u64_mul(n, gdn_k_dim, "overlay gdn q norm elements overflow")?,
            checked_u64_mul(n, gdn_k_dim, "overlay gdn k norm elements overflow")?,
            checked_u64_mul(n, gdn_v_dim, "overlay gdn v elements overflow")?,
            checked_u64_mul(n, gdn_v_dim, "overlay gdn out elements overflow")?,
            checked_u64_mul(n, gdn_v_dim, "overlay gdn normed elements overflow")?,
        ];
        let mut allocator = PrefillScratchAllocator::new(&scratch_plan);
        let overlay_eligible = scratch_plan.overlay.is_some();
        let (overlay_views, scratch_overlay) = if overlay_eligible {
            let layout = prefill_scratch_overlay_layout(
                attn_matrix_scores_h_elems,
                attn_matrix_ml_elems,
                gdn_shapes,
            )?;
            let backing_elems = layout.backing_bytes / 4;
            let backing = allocator.f32(ctx, "attn_gdn_overlay_backing", vec![backing_elems])?;
            let views = PrefillScratchOverlayViews {
                scores_h: checked_overlay_tensor(
                    &backing,
                    layout.scores_h,
                    vec![attn_matrix_scores_h_elems],
                    GgmlType::F16,
                )?,
                ml: checked_overlay_tensor(
                    &backing,
                    layout.ml,
                    vec![attn_matrix_ml_elems],
                    GgmlType::F32,
                )?,
                gdn_qkv: checked_overlay_tensor(
                    &backing,
                    layout.gdn_qkv,
                    vec![n, gdn_conv_dim],
                    GgmlType::F32,
                )?,
                gdn_z: checked_overlay_tensor(
                    &backing,
                    layout.gdn_z,
                    vec![n, gdn_v_dim],
                    GgmlType::F32,
                )?,
                gdn_beta: checked_overlay_tensor(
                    &backing,
                    layout.gdn_beta,
                    vec![n, gdn_n_v.max(1)],
                    GgmlType::F32,
                )?,
                gdn_alpha: checked_overlay_tensor(
                    &backing,
                    layout.gdn_alpha,
                    vec![n, gdn_n_v.max(1)],
                    GgmlType::F32,
                )?,
                gdn_q_norm: checked_overlay_tensor(
                    &backing,
                    layout.gdn_q_norm,
                    vec![n, gdn_k_dim],
                    GgmlType::F32,
                )?,
                gdn_k_norm: checked_overlay_tensor(
                    &backing,
                    layout.gdn_k_norm,
                    vec![n, gdn_k_dim],
                    GgmlType::F32,
                )?,
                gdn_v: checked_overlay_tensor(
                    &backing,
                    layout.gdn_v,
                    vec![n, gdn_v_dim],
                    GgmlType::F32,
                )?,
                gdn_out: checked_overlay_tensor(
                    &backing,
                    layout.gdn_out,
                    vec![n, gdn_v_dim],
                    GgmlType::F32,
                )?,
                gdn_normed: checked_overlay_tensor(
                    &backing,
                    layout.gdn_normed,
                    vec![n, gdn_v_dim],
                    GgmlType::F32,
                )?,
            };
            let stats = scratch_plan.overlay.ok_or_else(|| MetalError::BadShape {
                kernel: "prefill_scratch_overlay",
                detail: "overlay layout missing from scratch plan".into(),
            })?;
            debug_assert_eq!(stats.backing_bytes, layout.backing_bytes);
            debug_assert_eq!(stats.attention_bytes, layout.attention_bytes);
            debug_assert_eq!(stats.gdn_bytes, layout.gdn_bytes);
            (views, Some(stats))
        } else {
            (
                PrefillScratchOverlayViews {
                    scores_h: allocator.f16(
                        ctx,
                        "attn_matrix_scores_h_pack",
                        vec![attn_matrix_scores_h_elems],
                    )?,
                    ml: allocator.f32(ctx, "attn_matrix_ml_pack", vec![attn_matrix_ml_elems])?,
                    gdn_qkv: allocator.f32(ctx, "gdn_qkv_pack", vec![n, gdn_conv_dim])?,
                    gdn_z: allocator.f32(ctx, "gdn_z_pack", vec![n, gdn_v_dim])?,
                    gdn_beta: allocator.f32(ctx, "gdn_beta_pack", vec![n, gdn_n_v.max(1)])?,
                    gdn_alpha: allocator.f32(ctx, "gdn_alpha_pack", vec![n, gdn_n_v.max(1)])?,
                    gdn_q_norm: allocator.f32(ctx, "gdn_q_norm_pack", vec![n, gdn_k_dim])?,
                    gdn_k_norm: allocator.f32(ctx, "gdn_k_norm_pack", vec![n, gdn_k_dim])?,
                    gdn_v: allocator.f32(ctx, "gdn_v_pack", vec![n, gdn_v_dim])?,
                    gdn_out: allocator.f32(ctx, "gdn_out_pack", vec![n, gdn_v_dim])?,
                    gdn_normed: allocator.f32(ctx, "gdn_normed_pack", vec![n, gdn_v_dim])?,
                },
                None,
            )
        };

        let scratch = Self {
            x_pack: allocator.f32(ctx, "x_pack", vec![n, h])?,
            h_pack: allocator.f32(ctx, "h_pack", vec![n, h])?,
            mixer_out_pack: allocator.f32(ctx, "mixer_out_pack", vec![n, h])?,
            attn_qkv_fused_pack: allocator.f32(
                ctx,
                "attn_qkv_fused_pack",
                if enable_attn_fused_qkv_g8 {
                    vec![n, attn_qkv_fused_dim]
                } else {
                    vec![1]
                },
            )?,
            attn_q_full_pack: allocator.f32(ctx, "attn_q_full_pack", vec![n, attn_q_full_dim])?,
            attn_q_pack: allocator.f32(ctx, "attn_q_pack", vec![1])?,
            attn_gate_pack: allocator.f32(ctx, "attn_gate_pack", vec![1])?,
            attn_q_normed_pack: allocator.f32(ctx, "attn_q_normed_pack", vec![n, q_dim])?,
            attn_k_now_pack: allocator.f32(ctx, "attn_k_now_pack", vec![n, kv_dim])?,
            attn_v_now_pack: allocator.f32(ctx, "attn_v_now_pack", vec![n, kv_dim])?,
            attn_k_normed_pack: allocator.f32(ctx, "attn_k_normed_pack", vec![n, kv_dim])?,
            attn_o_pack: allocator.f32(ctx, "attn_o_pack", vec![n, q_dim])?,
            attn_prefill_v4_o_partial_pack: allocator.f32(
                ctx,
                "attn_prefill_v4_o_partial_pack",
                if enable_attn_packed {
                    vec![attn_prefill_v4_o_partial_elems]
                } else {
                    vec![1]
                },
            )?,
            attn_prefill_v4_ml_partial_pack: allocator.f32(
                ctx,
                "attn_prefill_v4_ml_partial_pack",
                if enable_attn_packed {
                    vec![attn_prefill_v4_ml_partial_elems]
                } else {
                    vec![1]
                },
            )?,
            attn_matrix_scores_pack: allocator.f32(
                ctx,
                "attn_matrix_scores_pack",
                vec![attn_matrix_scores_elems],
            )?,
            attn_matrix_scores_h_pack: overlay_views.scores_h,
            attn_matrix_ml_pack: overlay_views.ml,
            attn_matrix_vt_pack: allocator.f16(
                ctx,
                "attn_matrix_vt_pack",
                vec![attn_matrix_vt_elems],
            )?,
            ffn_gate_pack: allocator.f32(ctx, "ffn_gate_pack", vec![n, f])?,
            ffn_up_pack: allocator.f32(ctx, "ffn_up_pack", vec![n, f])?,
            ffn_inner_pack: allocator.f32(ctx, "ffn_inner_pack", vec![n, f])?,
            ffn_out_pack: allocator.f32(ctx, "ffn_out_pack", vec![n, h])?,
            moe_topk_idx_pack: allocator.f32(ctx, "moe_topk_idx_pack", vec![moe_slot_elems])?,
            moe_router_probs_pack: allocator.f32(
                ctx,
                "moe_router_probs_pack",
                vec![n, (arch.expert_count as u64).max(1)],
            )?,
            moe_topk_weight_pack: allocator.f32(
                ctx,
                "moe_topk_weight_pack",
                vec![moe_slot_elems],
            )?,
            moe_shared_gate_pack: allocator.f32(ctx, "moe_shared_gate_pack", vec![n])?,
            moe_inner_pack: allocator.f32(ctx, "moe_inner_pack", moe_inner_shape)?,
            moe_expert_out_pack: allocator.f32(ctx, "moe_expert_out_pack", moe_out_shape)?,
            moe_inner_full_elems: moe_inner_elems,
            moe_out_full_elems: moe_out_elems,
            moe_group_slot_idx_pack: allocator.i32(
                ctx,
                "moe_group_slot_idx_pack",
                vec![moe_slot_elems],
            )?,
            moe_group_count_pack: allocator.f32(
                ctx,
                "moe_group_count_pack",
                vec![(arch.expert_count as u64).max(1)],
            )?,
            moe_group_ids_pack: allocator.f32(
                ctx,
                "moe_group_ids_pack",
                vec![moe_group_ids_elems],
            )?,
            moe_group_token_idx_pack: allocator.f32(
                ctx,
                "moe_group_token_idx_pack",
                vec![moe_slot_elems],
            )?,
            moe_group_weight_pack: allocator.f32(
                ctx,
                "moe_group_weight_pack",
                vec![moe_slot_elems],
            )?,
            moe_group_inner_pack: allocator.f32(
                ctx,
                "moe_group_inner_pack",
                vec![moe_inner_elems],
            )?,
            moe_group_out_pack: allocator.f32(ctx, "moe_group_out_pack", vec![moe_out_elems])?,
            moe_shared_ffn_gate_pack: allocator.f32(
                ctx,
                "moe_shared_ffn_gate_pack",
                vec![n, moe_f_shared],
            )?,
            moe_shared_ffn_up_pack: allocator.f32(
                ctx,
                "moe_shared_ffn_up_pack",
                vec![n, moe_f_shared],
            )?,
            moe_shared_ffn_inner_pack: allocator.f32(
                ctx,
                "moe_shared_ffn_inner_pack",
                vec![n, moe_f_shared],
            )?,
            moe_shared_ffn_out_pack: allocator.f32(ctx, "moe_shared_ffn_out_pack", vec![n, h])?,
            final_logits_pack: allocator.f32(ctx, "final_logits_pack", final_logits_shape)?,
            gdn_qkv_pack: overlay_views.gdn_qkv,
            gdn_z_pack: overlay_views.gdn_z,
            gdn_beta_pack: overlay_views.gdn_beta,
            gdn_alpha_pack: overlay_views.gdn_alpha,
            gdn_q_norm_pack: overlay_views.gdn_q_norm,
            gdn_k_norm_pack: overlay_views.gdn_k_norm,
            gdn_v_pack: overlay_views.gdn_v,
            gdn_out_pack: overlay_views.gdn_out,
            gdn_normed_pack: overlay_views.gdn_normed,
            n: block_size,
            attn_matrix_query_rows: u32::try_from(attn_matrix_query_rows).map_err(|_| {
                MetalError::BadShape {
                    kernel: "prefill_attn_matrix_query_cap",
                    detail: format!("matrix query rows {attn_matrix_query_rows} do not fit u32"),
                }
            })?,
            attn_matrix_tiled_layer_calls: 0,
            attn_matrix_query_tile_calls: 0,
            scratch_overlay,
            scratch_plan: scratch_plan.clone(),
            hidden_size: h,
            intermediate_size: f,
            vocab_size: v,
            q_dim,
            kv_dim,
            attn_matrix_max_pos,
            gdn_conv_dim,
            gdn_v_dim,
            gdn_n_v,
        };
        allocator.finish()?;
        Ok(scratch)
    }

    pub fn fresh(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
    ) -> Result<Self, MetalError> {
        Self::fresh_inner(ctx, target_model, block_size, true, None, None, None)
    }

    /// Priced bytes for the speculative layer-major constructor, including
    /// full logits and MoE fallback packs.
    pub fn speculative_priced_bytes(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
    ) -> Result<u64, MetalError> {
        let n_attn_layers = u64::try_from(
            target_model
                .blocks
                .iter()
                .filter(|block| matches!(block, MetalBlock::Attn(_)))
                .count()
                .max(1),
        )
        .map_err(|_| MetalError::BadShape {
            kernel: "dflash_layer_scratch",
            detail: "attention layer count does not fit u64".into(),
        })?;
        let has_gdn = target_model
            .blocks
            .iter()
            .any(|block| matches!(block, MetalBlock::Gdn(_)));
        let modes =
            resolve_prefill_scratch_plan_modes(&target_model.arch, block_size, true, None, None)?;
        let plan = build_prefill_scratch_plan_from_arch(
            &target_model.arch,
            n_attn_layers,
            has_gdn,
            block_size,
            true,
            modes,
        )?;
        plan.priced_upper_bound(|logical_bytes| {
            Ok(ctx.shared_buffer_size_and_align(logical_bytes)?.size)
        })
    }

    pub fn fresh_prefill(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
    ) -> Result<Self, MetalError> {
        Self::fresh_inner(ctx, target_model, block_size, false, None, None, None)
    }

    pub fn fresh_prefill_with_matrix_max_pos(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
        matrix_max_pos: usize,
    ) -> Result<Self, MetalError> {
        Self::fresh_inner(
            ctx,
            target_model,
            block_size,
            false,
            Some(matrix_max_pos),
            None,
            None,
        )
    }

    pub fn fresh_prefill_configured(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
        config: PrefillScratchConfig,
    ) -> Result<Self, MetalError> {
        Self::fresh_inner(
            ctx,
            target_model,
            block_size,
            false,
            None,
            Some(config),
            None,
        )
    }

    pub fn fresh_prefill_with_matrix_max_pos_configured(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
        matrix_max_pos: usize,
        config: PrefillScratchConfig,
    ) -> Result<Self, MetalError> {
        Self::fresh_inner(
            ctx,
            target_model,
            block_size,
            false,
            Some(matrix_max_pos),
            Some(config),
            None,
        )
    }

    pub fn fresh_prefill_from_plan(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        plan: PrefillScratchPlan,
    ) -> Result<Self, MetalError> {
        let block_size = plan.block_size;
        Self::fresh_inner(ctx, target_model, block_size, false, None, None, Some(plan))
    }

    /// Grow the packed-slot MoE fallback packs to full size if this scratch
    /// was built by a `fresh_prefill*` constructor (which stubs them; see
    /// `fresh_inner`). Called at the top of every packed-slot fallback
    /// branch in the prefill MoE tail. Idempotent; the dropped stub was
    /// never encoded into any command buffer, so replacing it is safe.
    pub fn ensure_moe_packed_fallback(&mut self, ctx: &MetalContext) -> Result<(), MetalError> {
        if self.moe_inner_pack.n_elements() < self.moe_inner_full_elems {
            let bytes = checked_u64_mul(
                self.moe_inner_full_elems,
                4,
                "moe inner fallback allocation bytes overflow",
            )?;
            self.scratch_plan.validate_deferred(
                "moe_inner_pack_fallback_growth",
                bytes,
                GgmlType::F32,
            )?;
            self.moe_inner_pack = MetalTensor::zeros_f32(ctx, vec![self.moe_inner_full_elems])?;
        }
        if self.moe_expert_out_pack.n_elements() < self.moe_out_full_elems {
            let bytes = checked_u64_mul(
                self.moe_out_full_elems,
                4,
                "moe output fallback allocation bytes overflow",
            )?;
            self.scratch_plan.validate_deferred(
                "moe_expert_out_pack_fallback_growth",
                bytes,
                GgmlType::F32,
            )?;
            self.moe_expert_out_pack = MetalTensor::zeros_f32(ctx, vec![self.moe_out_full_elems])?;
        }
        Ok(())
    }

    pub fn attn_matrix_query_rows(&self) -> usize {
        self.attn_matrix_query_rows as usize
    }

    pub fn attn_matrix_tiled_layer_calls(&self) -> u64 {
        self.attn_matrix_tiled_layer_calls
    }

    pub fn attn_matrix_query_tile_calls(&self) -> u64 {
        self.attn_matrix_query_tile_calls
    }

    pub fn prefill_scratch_overlay_stats(&self) -> Option<PrefillScratchOverlayStats> {
        self.scratch_overlay
    }

    pub fn prefill_scratch_plan(&self) -> &PrefillScratchPlan {
        &self.scratch_plan
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

    /// Zero-copy view of row n of `attn_q_normed_pack`, etc.
    /// (v0.447: `attn_q_row`/`attn_gate_row` deleted with their stubbed
    /// packs — the strided kernels read `attn_q_full_pack` directly.)
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
    #[cfg(feature = "dflash-k0s-diagnostics")]
    k0s_live_event: Option<u64>,
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
            #[cfg(feature = "dflash-k0s-diagnostics")]
            k0s_live_event: None,
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
    // The per-row slot helpers on `MetalDFlashVerifyScratch` and the
    // underlying `view_subrange` both now use `assert!` (promoted from
    // `debug_assert!` after the H5.3a release-build OOB hazard review),
    // but this entry guard is still cheaper than the per-row checks
    // and produces a single clear error message when a scratch
    // allocated for a different `block_size` / `target_layer_ids.len()`
    // / target model schedule reaches this path.
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
            None, // n_eff_override
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
            None, // n_eff_override (debug variant always uses full N)
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
    // After `packed_verify(tokens[0..N], start_position)`, checkpoint slots
    // n ∈ [0, N-1) hold "state-after-token-n for GDN layer k". The final
    // state already lives in the session and is never a rollback source.
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
    // means full accept (carry + all D drafts) — restore returns without a
    // copy because the session already holds that state.
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
            None, // n_eff_override (default; v0.76 adaptive will pass per step)
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

    // -- Codex failure-mode guard wall: validate ALL dims at entry.
    // The per-row slot helpers and view_subrange both now use `assert!`
    // (release-safe), but this single entry check produces a clearer
    // error than a downstream slot-OOB panic when a scratch was
    // allocated for the wrong shape.
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
        let gap_dst = scratch.gap_slot(n_idx as u32);
        encode_argmax_top2_f32(
            base.ctx,
            &enc,
            &target_session.logits,
            &argmax_dst,
            &gap_dst,
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
        if !packed_verify_skip_final_ckpt_enabled() || n_idx + 1 < n {
            let blit = BlitEncoder::begin(&cmd_buf);
            for k in 0..n_gdn_actual {
                let ssm_dst = scratch.gdn_ckpt_slot(k, n_idx as u32);
                blit.copy_tensor(&target_session.gdn_state[k as usize], &ssm_dst);
                let conv_dst = scratch.conv_ckpt_slot(k, n_idx as u32);
                blit.copy_tensor(&target_session.gdn_conv[k as usize], &conv_dst);
            }
            blit.end();
        }
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

// H5.3a rollback primitive — low-level entrypoint that takes
// everything explicitly. `DFlashDecoder::restore_after_partial_accept`
// is the production wrapper; this exists so unit tests can exercise
// the rollback algorithm without standing up a full DFlash drafter.
//
// See `DFlashDecoder::restore_after_partial_accept` for the indexing
// spec and `n_keep` semantics — they are identical.
//
// Algorithm:
//   1. Validate dims (n_keep ∈ [1, N], scratch matches model, etc.).
//   2. Open one MTLCommandBuffer + BlitEncoder.
//   3. For each GDN layer k:
//        gdn_state[k] ← gdn_ckpt_slot(k, n_keep - 1)
//        gdn_conv[k]  ← conv_ckpt_slot(k, n_keep - 1)
//   4. End blit encoder, commit, wait.
//   5. CPU update: kv_n_pos[i] := start_position + n_keep for every
//      attn layer i.
//
// Step 5 is host-side because `MetalSession::kv_n_pos` is a
// `Vec<usize>` on the host (matches the existing `encode_attn`
// pattern where it's read at encode time, not GPU-side). KV slot
// bytes at [start_position + n_keep, ...) physically remain but
// become unreachable; next packed_verify call overwrites them.
// =============================================================================
// encode_packed_verify_layer_major_inner — H5.3b.4-5 layer-major path
// =============================================================================

fn encode_packed_verify_moe_grouped_ffn_after_mixer(
    base: &MetalForward<'_>,
    enc: &KernelEncoder,
    block: &MetalBlock,
    x_pack: &MetalTensor,
    h_pack: &MetalTensor,
    layer_scratch: &MetalDFlashLayerMajorScratch,
    n: usize,
    h: usize,
) -> Result<bool, DFlashError> {
    let arch = &base.model.arch;
    let (ffn_gate, ffn_up, ffn_down, moe) = match block {
        MetalBlock::Gdn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
        MetalBlock::Attn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
    };
    let Some(moe) = moe else {
        return Ok(false);
    };

    let router_mat_mat_eligible = |dtype: GgmlType| {
        matches!(
            dtype,
            GgmlType::F32
                | GgmlType::F16
                | GgmlType::BF16
                | GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::Q8_0
        )
    };

    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let f_shared = arch.expert_shared_feed_forward_length as usize;
    if topk == 0
        || topk > 16
        || n_expert == 0
        || n_expert > 256
        || !h.is_multiple_of(256)
        || !f_exp.is_multiple_of(256)
    {
        return Ok(false);
    }

    let grouped_gate_up_dtype_eligible = match (moe.gate_exps.dtype, moe.up_exps.dtype) {
        (GgmlType::Q4_K, GgmlType::Q4_K) => true,
        (GgmlType::Q5_K, GgmlType::Q5_K) => {
            prefill_moe_grouped_q5_gateup_enabled(h, f_exp, n_expert)
        }
        (GgmlType::Q6_K, GgmlType::Q6_K) => {
            prefill_moe_grouped_q6_gateup_enabled(h, f_exp, n_expert)
        }
        (GgmlType::Q8_0, GgmlType::Q8_0) => {
            prefill_moe_grouped_q8_gateup_enabled(h, f_exp, n_expert)
        }
        (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS) | (GgmlType::IQ3_S, GgmlType::IQ3_S) => {
            n >= 32 && prefill_moe_grouped_iq3_gateup_enabled(h, f_exp, n_expert)
        }
        (GgmlType::F32, GgmlType::F32) => n >= 32 && prefill_moe_grouped_f32_gateup_enabled(),
        (GgmlType::BF16, GgmlType::BF16) => {
            n >= 32 && prefill_moe_grouped_bf16_gateup_enabled(h, f_exp, n_expert)
        }
        _ => false,
    };
    let grouped_down_dtype_eligible = matches!(
        moe.down_exps.dtype,
        GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0 | GgmlType::IQ4_XS | GgmlType::BF16
    );
    if !(prefill_moe_grouped_enabled()
        && grouped_gate_up_dtype_eligible
        && grouped_down_dtype_eligible)
    {
        return Ok(false);
    }
    if !(prefill_moe_packed_route_enabled()
        && router_mat_mat_eligible(moe.gate_inp.dtype)
        && moe.gate_inp_shexp.dtype == GgmlType::F32)
    {
        return Ok(false);
    }
    let packed_shared_path = f_shared == 0
        || (prefill_moe_packed_shared_enabled()
            && router_mat_mat_eligible(ffn_gate.dtype)
            && router_mat_mat_eligible(ffn_up.dtype)
            && router_mat_mat_eligible(ffn_down.dtype));
    if !packed_shared_path {
        return Ok(false);
    }

    let moe_topk_idx_pack = layer_scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(n * topk) as u64]);
    let moe_router_probs_pack = layer_scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(n * n_expert) as u64]);
    let moe_topk_weight_pack = layer_scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(n * topk) as u64]);
    let moe_shared_gate_pack = layer_scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![n as u64]);
    let moe_group_count_pack = layer_scratch
        .moe_group_count_pack
        .view_subrange(0, vec![n_expert as u64]);
    let moe_group_ids_pack = layer_scratch
        .moe_group_ids_pack
        .view_subrange(0, vec![(n_expert * n) as u64]);
    let moe_group_inner_pack = layer_scratch
        .moe_group_inner_pack
        .view_subrange(0, vec![(n * topk * f_exp) as u64]);
    let moe_group_out_pack = layer_scratch
        .moe_group_out_pack
        .view_subrange(0, vec![(n * topk * h) as u64]);
    let ffn_delta_pack = layer_scratch
        .mixer_out_pack
        .view_subrange(0, vec![(n * h) as u64]);
    let shared_gate_pack = layer_scratch
        .moe_shared_ffn_gate_pack
        .view_subrange(0, vec![(n * f_shared) as u64]);
    let shared_up_pack = layer_scratch
        .moe_shared_ffn_up_pack
        .view_subrange(0, vec![(n * f_shared) as u64]);
    let shared_inner_pack = layer_scratch
        .moe_shared_ffn_inner_pack
        .view_subrange(0, vec![(n * f_shared) as u64]);
    let shared_out_pack = layer_scratch
        .moe_shared_ffn_out_pack
        .view_subrange(0, vec![(n * h) as u64]);

    encode_moe_route_logits_dispatch(
        base.ctx,
        enc,
        &moe.gate_inp,
        h_pack,
        &moe_router_probs_pack,
        h,
        n_expert,
        n,
    )?;
    encode_fill_f32(base.ctx, enc, &moe_group_count_pack, 0.0)?;
    crate::metal::encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
        base.ctx,
        enc,
        &moe_router_probs_pack,
        &moe.gate_inp_shexp,
        h_pack,
        &moe_topk_idx_pack,
        &moe_topk_weight_pack,
        &moe_shared_gate_pack,
        &moe_group_count_pack,
        &moe_group_ids_pack,
        n_expert,
        topk,
        h,
        n,
    )?;
    encode_prefill_moe_grouped_swiglu(
        base.ctx,
        enc,
        moe,
        h_pack,
        &moe_group_count_pack,
        &moe_group_ids_pack,
        &moe_group_inner_pack,
        h,
        f_exp,
        n_expert,
        topk,
        n,
        prefill_moe_grouped_q4_n32_all_enabled(arch, n),
        prefill_moe_hot_expert_min_slots(),
    )?;
    encode_prefill_moe_grouped_down(
        base.ctx,
        enc,
        &moe.down_exps,
        &moe_group_inner_pack,
        &moe_group_count_pack,
        &moe_group_ids_pack,
        &moe_group_out_pack,
        f_exp,
        h,
        n_expert,
        n,
    )?;
    crate::metal::encode_moe_weighted_sum_packed_f32(
        base.ctx,
        enc,
        &moe_group_out_pack,
        &moe_topk_weight_pack,
        &ffn_delta_pack,
        h,
        topk,
        n,
    )?;

    if f_shared > 0 {
        encode_mat_mat_dispatch(
            base.ctx,
            enc,
            ffn_gate,
            h_pack,
            &shared_gate_pack,
            h,
            f_shared,
            n,
        )?;
        encode_mat_mat_dispatch(
            base.ctx,
            enc,
            ffn_up,
            h_pack,
            &shared_up_pack,
            h,
            f_shared,
            n,
        )?;
        encode_silu_mul_f32(
            base.ctx,
            enc,
            &shared_gate_pack,
            &shared_up_pack,
            &shared_inner_pack,
        )?;
        encode_mat_mat_dispatch(
            base.ctx,
            enc,
            ffn_down,
            &shared_inner_pack,
            &shared_out_pack,
            f_shared,
            h,
            n,
        )?;
        encode_axpy_rowwise_f32(
            base.ctx,
            enc,
            &shared_out_pack,
            &moe_shared_gate_pack,
            &ffn_delta_pack,
            h,
            n,
        )?;
    }
    encode_add_inplace_f32(base.ctx, enc, x_pack, &ffn_delta_pack)?;
    Ok(true)
}
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
    n_eff_override: Option<u32>,
) -> Result<Vec<i32>, DFlashError> {
    let arch = &base.model.arch;
    // `n_block` is the scratch allocation size (verify_scratch.n,
    // layer_scratch.n; both must agree). `n` is the EFFECTIVE chain
    // length used by THIS call — `n_block` if no override, else
    // `n_eff_override` for v0.76 adaptive-N back-off. Encoders that
    // strict-equal-check `n_elements()` against `n * dim` are passed
    // `view_subrange`-sized scratch views below; ckpt slot indices
    // `[0, n)` are written, indices `[n, n_block)` remain stale from
    // any prior call (they are never read by `restore` when called
    // with the SAME `n_eff_override`).
    let n_block = verify_scratch.n as usize;
    let n = n_eff_override.map(|v| v as usize).unwrap_or(n_block);
    let h = arch.hidden_size as usize;
    let f = arch.intermediate_size as usize;
    let v = arch.vocab_size as usize;

    // -- guard wall (mirrors token-major; same bug class) --
    if n == 0 || n > n_block {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify_layer_major",
            detail: format!(
                "n_eff_override={:?} resolves to n={n} which must be in [1, n_block={n_block}]",
                n_eff_override
            ),
        }));
    }
    if tokens.len() != n {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify_layer_major",
            detail: format!("tokens.len()={} != n_eff={n}", tokens.len()),
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
    if let Some(dst) = debug_logits_dst
        && dst.shape != vec![n as u64, v as u64]
    {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "packed_verify_layer_major.debug_logits_dst",
            detail: format!("expected [{n}, {v}], got {:?}", dst.shape),
        }));
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

    // -- v0.76 sized-view bindings for adaptive-N back-off --
    //
    // When `n_eff_override` truncates the verify chain below the
    // scratch allocation size `n_block`, every encoder dispatch
    // that strict-equals against `n * dim` would fail host-side
    // validation if we passed the full `[n_block, dim]` scratch
    // tensors. We size views to exactly `[n, dim]` here and use
    // them throughout the body. Per-block tensors with dims that
    // depend on arch (GDN conv_dim/v_dim, attn q_dim/kv_dim) are
    // sized inside each block's match arm where the dims are
    // already computed.
    //
    // `hidden_capture` ([n_block, K, H]) and `gdn_ckpt`/`conv_ckpt`
    // ([n_gdn, n_block, ...]) are NOT view-sized: they're written
    // at absolute slot offsets `(n_idx * K + k_idx) * H` etc., so
    // any subset of n_idx values writes to disjoint regions.
    // Slots `[n_eff, n_block)` remain stale (or zero from
    // construction); they are never read by `restore` or `bench`
    // when called with the SAME `n_eff_override` for the same
    // outer step.
    let x_pack = layer_scratch.x_pack.view_subrange(0, vec![(n * h) as u64]);
    let h_pack = layer_scratch.h_pack.view_subrange(0, vec![(n * h) as u64]);
    let mixer_out_pack = layer_scratch
        .mixer_out_pack
        .view_subrange(0, vec![(n * h) as u64]);
    let ffn_gate_pack = layer_scratch
        .ffn_gate_pack
        .view_subrange(0, vec![(n * f) as u64]);
    let ffn_up_pack = layer_scratch
        .ffn_up_pack
        .view_subrange(0, vec![(n * f) as u64]);
    let ffn_inner_pack = layer_scratch
        .ffn_inner_pack
        .view_subrange(0, vec![(n * f) as u64]);
    let ffn_out_pack = layer_scratch
        .ffn_out_pack
        .view_subrange(0, vec![(n * h) as u64]);
    let final_logits_pack = layer_scratch
        .final_logits_pack
        .view_subrange(0, vec![(n * v) as u64]);
    let packed_ids_buf = verify_scratch
        .packed_ids_buf
        .view_subrange(0, vec![n as u64]);
    let verify_argmax_view = verify_scratch
        .verify_argmax
        .view_subrange(0, vec![n as u64]);
    let verify_gap_view = verify_scratch.verify_gap.view_subrange(0, vec![n as u64]);

    let trace_counts = mtp_verify_trace_counts_enabled();
    let _kernel_trace_guard = trace_counts.then(kernel_trace_begin);

    let cmd_buf = base.ctx.queue.commandBuffer().expect("command buffer");

    // === Phase 0: pre-block GDN state capture (one blit pass). ===
    // Lets the margin-guarded fallback roll the whole block back to its
    // start (encode_restore_to_pre_block). Cost is one checkpoint set
    // per verify step.
    {
        let n_gdn_actual = base
            .model
            .blocks
            .iter()
            .filter(|b| matches!(b, MetalBlock::Gdn(_)))
            .count() as u32;
        let blit = BlitEncoder::begin(&cmd_buf);
        for k in 0..n_gdn_actual {
            let pre_ssm = verify_scratch.pre_gdn_slot(k);
            blit.copy_tensor(&target_session.gdn_state[k as usize], &pre_ssm);
            let pre_conv = verify_scratch.pre_conv_slot(k);
            blit.copy_tensor(&target_session.gdn_conv[k as usize], &pre_conv);
        }
        blit.end();
    }

    // === Phase 1: batched embed of all N tokens into x_pack [N, H]. ===
    {
        let enc = KernelEncoder::begin(&cmd_buf);
        encode_get_rows_f32(
            base.ctx,
            &enc,
            &base.model.token_embd,
            &packed_ids_buf,
            &x_pack,
            n,
            h,
        )?;
        enc.end();
    }
    emit_mtp_verify_count_phase(trace_counts, -1, "embed", "get_rows");

    // === Phase 2: layer loop. ===
    let mut gdn_idx = 0usize;
    let mut attn_idx = 0usize;
    for (il, block) in base.model.blocks.iter().enumerate() {
        let block_kind = match block {
            MetalBlock::Gdn(_) => "gdn",
            MetalBlock::Attn(_) => "attn",
        };
        if arch.kind == crate::model::ArchKind::Moe && !mtp_moe_verify_batched_mixer_enabled() {
            let gdn_ckpt_idx = match block {
                MetalBlock::Gdn(_) => {
                    let gi = gdn_idx;
                    gdn_idx += 1;
                    Some(gi)
                }
                MetalBlock::Attn(_) => {
                    attn_idx += 1;
                    None
                }
            };
            for n_idx in 0..n {
                let position_n = start_position + n_idx as u32;
                {
                    let enc = KernelEncoder::begin(&cmd_buf);
                    encode_copy_offset_f32(
                        base.ctx,
                        &enc,
                        &x_pack,
                        n_idx * h,
                        &target_session.x,
                        h,
                    )?;
                    base.encode_moe_mixer_prep_by_index(&enc, il, position_n, target_session)?;
                    encode_scatter_offset_f32(
                        base.ctx,
                        &enc,
                        &target_session.x,
                        &x_pack,
                        n_idx * h,
                        h,
                    )?;
                    encode_scatter_offset_f32(
                        base.ctx,
                        &enc,
                        &target_session.h,
                        &h_pack,
                        n_idx * h,
                        h,
                    )?;
                    enc.end();
                }
                if let Some(gi) = gdn_ckpt_idx
                    && (!packed_verify_skip_final_ckpt_enabled() || n_idx + 1 < n)
                {
                    let blit = BlitEncoder::begin(&cmd_buf);
                    let ssm_dst = verify_scratch.gdn_ckpt_slot(gi as u32, n_idx as u32);
                    blit.copy_tensor(&target_session.gdn_state[gi], &ssm_dst);
                    let conv_dst = verify_scratch.conv_ckpt_slot(gi as u32, n_idx as u32);
                    blit.copy_tensor(&target_session.gdn_conv[gi], &conv_dst);
                    blit.end();
                }
            }
            let ffn_batched = if mtp_moe_verify_grouped_ffn_enabled() {
                let enc = KernelEncoder::begin(&cmd_buf);
                let done = encode_packed_verify_moe_grouped_ffn_after_mixer(
                    base,
                    &enc,
                    block,
                    &x_pack,
                    &h_pack,
                    layer_scratch,
                    n,
                    h,
                )?;
                enc.end();
                emit_mtp_verify_count_phase(trace_counts, il as isize, "moe", "ffn_grouped");
                done
            } else {
                false
            };
            if !ffn_batched {
                for n_idx in 0..n {
                    let enc = KernelEncoder::begin(&cmd_buf);
                    encode_copy_offset_f32(
                        base.ctx,
                        &enc,
                        &x_pack,
                        n_idx * h,
                        &target_session.x,
                        h,
                    )?;
                    encode_copy_offset_f32(
                        base.ctx,
                        &enc,
                        &h_pack,
                        n_idx * h,
                        &target_session.h,
                        h,
                    )?;
                    base.encode_moe_ffn_after_mixer_by_index(&enc, il, target_session)?;
                    encode_scatter_offset_f32(
                        base.ctx,
                        &enc,
                        &target_session.x,
                        &x_pack,
                        n_idx * h,
                        h,
                    )?;
                    enc.end();
                }
            }
            for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                if lid as usize == il {
                    let enc = KernelEncoder::begin(&cmd_buf);
                    for n_idx in 0..n {
                        let elem_off = (n_idx as u64 * verify_scratch.k_target_layers as u64
                            + k_idx as u64)
                            * verify_scratch.hidden_size;
                        let x_n = x_pack.view_subrange((n_idx * h) as u64, vec![h as u64]);
                        encode_scatter_offset_f32(
                            base.ctx,
                            &enc,
                            &x_n,
                            &verify_scratch.hidden_capture,
                            elem_off as usize,
                            h,
                        )?;
                    }
                    enc.end();
                }
            }
            emit_mtp_verify_count_phase(trace_counts, il as isize, "moe_legacy", "layer");
            continue;
        }
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
                base.ctx, &enc, &x_pack, attn_norm, &h_pack, n, h, RMS_EPS,
            )?;
            enc.end();
        }
        emit_mtp_verify_count_phase(trace_counts, il as isize, block_kind, "pre_norm");
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
                // N=16 when each dtype is supported by encode_mat_mat_dispatch.
                // Production 27B Q4_K_M is Q6_K/Q4_K/Q5_K respectively.
                // beta_proj and alpha_proj
                // stay per-token mat-vec because production stores them
                // as F32 [hidden, n_v=48] — small, mat-mat dispatch
                // overhead exceeds BW savings (see docs/H5-DFLASH.md
                // rev 10).
                let gdn_mat_mat_eligible = prefill_mat_mat_dispatch_eligible;
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
                    // v0.76: sized views for adaptive-N back-off.
                    let gdn_qkv_pack = layer_scratch
                        .gdn_qkv_pack
                        .view_subrange(0, vec![(n * conv_dim) as u64]);
                    let gdn_z_pack = layer_scratch
                        .gdn_z_pack
                        .view_subrange(0, vec![(n * v_dim) as u64]);
                    let gdn_normed_pack = layer_scratch
                        .gdn_normed_pack
                        .view_subrange(0, vec![(n * v_dim) as u64]);
                    let gdn_q_norm_pack = layer_scratch
                        .gdn_q_norm_pack
                        .view_subrange(0, vec![(n * n_k_u * head_dim_u) as u64]);
                    let gdn_k_norm_pack = layer_scratch
                        .gdn_k_norm_pack
                        .view_subrange(0, vec![(n * n_k_u * head_dim_u) as u64]);
                    let gdn_v_pack = layer_scratch
                        .gdn_v_pack
                        .view_subrange(0, vec![(n * v_dim) as u64]);
                    let gdn_out_pack = layer_scratch
                        .gdn_out_pack
                        .view_subrange(0, vec![(n * v_dim) as u64]);
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &g.in_proj_qkv,
                            &h_pack,
                            &gdn_qkv_pack,
                            h,
                            conv_dim,
                            n,
                        )?;
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &g.in_proj_z,
                            &h_pack,
                            &gdn_z_pack,
                            h,
                            v_dim,
                            n,
                        )?;
                        enc.end();
                    }
                    emit_mtp_verify_count_phase(trace_counts, il as isize, "gdn", "front");

                    // Optional Q8 arm: keep the exact Q8 mat-vec projection
                    // kernels, but schedule alpha/beta for all rows before
                    // entering the sequential recurrence loop. This deletes
                    // repeated projection/transform encoder structure while
                    // leaving recurrence and rollback unchanged.
                    let packed_gdn = dflash_verify_packed_gdn_enabled()
                        && n > 1
                        && head_dim_u == 128
                        && n_k_u > 0
                        && n_v.is_multiple_of(n_k_u)
                        && g.conv1d.n_elements() as usize == 4 * conv_dim
                        && target_session.gdn_conv[gi].n_elements() as usize == 3 * conv_dim
                        && target_session.gdn_state[gi].n_elements() as usize
                            == n_v * head_dim_u * head_dim_u;
                    let q8_alpha_beta_batched = n > 1
                        && (packed_gdn
                            || (mtp_verify_q8_gdn_alpha_beta_batched_enabled()
                                && g.beta_proj.dtype == GgmlType::Q8_0
                                && g.alpha_proj.dtype == GgmlType::Q8_0));
                    let gdn_beta_pack = layer_scratch
                        .gdn_beta_pack
                        .view_subrange(0, vec![(n * n_v) as u64]);
                    let gdn_alpha_pack = layer_scratch
                        .gdn_alpha_pack
                        .view_subrange(0, vec![(n * n_v) as u64]);
                    if q8_alpha_beta_batched {
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_packed_matvec_projection(
                                base.ctx,
                                &enc,
                                &g.beta_proj,
                                &h_pack,
                                &gdn_beta_pack,
                                h,
                                n_v,
                                n,
                            )?;
                            encode_packed_matvec_projection(
                                base.ctx,
                                &enc,
                                &g.alpha_proj,
                                &h_pack,
                                &gdn_alpha_pack,
                                h,
                                n_v,
                                n,
                            )?;
                            enc.end();
                        }
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_sigmoid_f32(base.ctx, &enc, &gdn_beta_pack, &gdn_beta_pack)?;
                            encode_gdn_decay_chain_batched_f32(
                                base.ctx,
                                &enc,
                                &gdn_alpha_pack,
                                &g.dt_bias,
                                &g.a_log,
                                &gdn_alpha_pack,
                                n,
                                n_v,
                            )?;
                            enc.end();
                        }
                    }

                    // Per-token loop (recurrence is inherently sequential).
                    // Step B: alpha/beta (per-token by default, or packed Q8
                    // views above) + post-projection recurrence body +
                    // checkpoint blit.
                    if packed_gdn {
                        let n_checkpoints = if packed_verify_skip_final_ckpt_enabled() {
                            n - 1
                        } else {
                            n
                        };
                        let state_elems = verify_scratch.ssm_state_elems as usize;
                        let conv_elems = verify_scratch.conv_state_elems as usize;
                        let state_ckpt = verify_scratch.gdn_ckpt.view_subrange(
                            (gi * verify_scratch.n as usize * state_elems) as u64,
                            vec![(n_checkpoints * state_elems) as u64],
                        );
                        let conv_ckpt = verify_scratch.conv_ckpt.view_subrange(
                            (gi * verify_scratch.n as usize * conv_elems) as u64,
                            vec![(n_checkpoints * conv_elems) as u64],
                        );
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_gdn_prep_packed_ckpt_f32(
                            base.ctx,
                            &enc,
                            &gdn_qkv_pack,
                            &target_session.gdn_conv[gi],
                            &g.conv1d,
                            &gdn_q_norm_pack,
                            &gdn_k_norm_pack,
                            &gdn_v_pack,
                            &conv_ckpt,
                            n,
                            n_checkpoints,
                            n_k_u,
                            n_v,
                            head_dim_u,
                        )?;
                        if prefill_gdn_pair_l2_enabled() {
                            encode_l2_norm_pair_batched_f32(
                                base.ctx,
                                &enc,
                                &gdn_q_norm_pack,
                                &gdn_q_norm_pack,
                                &gdn_k_norm_pack,
                                &gdn_k_norm_pack,
                                n * n_k_u,
                                head_dim_u,
                                RMS_EPS,
                            )?;
                        } else {
                            encode_l2_norm_batched_f32(
                                base.ctx,
                                &enc,
                                &gdn_q_norm_pack,
                                &gdn_q_norm_pack,
                                n * n_k_u,
                                head_dim_u,
                                RMS_EPS,
                            )?;
                            encode_l2_norm_batched_f32(
                                base.ctx,
                                &enc,
                                &gdn_k_norm_pack,
                                &gdn_k_norm_pack,
                                n * n_k_u,
                                head_dim_u,
                                RMS_EPS,
                            )?;
                        }
                        encode_gdn_step_decay_packed_ckpt_f32(
                            base.ctx,
                            &enc,
                            &gdn_q_norm_pack,
                            &gdn_k_norm_pack,
                            &gdn_v_pack,
                            &gdn_alpha_pack,
                            &gdn_beta_pack,
                            &target_session.gdn_state[gi],
                            &gdn_out_pack,
                            &state_ckpt,
                            n_checkpoints,
                            n,
                            n_v,
                            n_k_u,
                            head_dim_u,
                        )?;
                        encode_rmsnorm_gated_f32(
                            base.ctx,
                            &enc,
                            &gdn_out_pack,
                            &g.norm,
                            &gdn_z_pack,
                            &gdn_normed_pack,
                            n * n_v,
                            head_dim_u,
                            RMS_EPS * head_dim_u as f32,
                        )?;
                        enc.end();
                    } else {
                        for n_idx in 0..n {
                            // Compute pass.
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                let (alpha_n, beta_n) = if q8_alpha_beta_batched {
                                    (
                                        gdn_alpha_pack
                                            .view_subrange((n_idx * n_v) as u64, vec![n_v as u64]),
                                        gdn_beta_pack
                                            .view_subrange((n_idx * n_v) as u64, vec![n_v as u64]),
                                    )
                                } else {
                                    // Per-row view of h_pack for the default
                                    // alpha/beta mat-vecs.
                                    let h_n = layer_scratch
                                        .h_pack
                                        .view_subrange((n_idx * h) as u64, vec![h as u64]);
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
                                    encode_mat_vec_dispatch(
                                        base.ctx,
                                        &enc,
                                        &g.alpha_proj,
                                        &h_n,
                                        &target_session.gdn_a,
                                        h,
                                        n_v,
                                    )?;
                                    encode_gdn_decay_chain_f32(
                                        base.ctx,
                                        &enc,
                                        &target_session.gdn_a,
                                        &g.dt_bias,
                                        &g.a_log,
                                        &target_session.gdn_alpha,
                                    )?;
                                    (
                                        target_session.gdn_alpha.clone(),
                                        target_session.gdn_beta.clone(),
                                    )
                                };
                                // Per-row views of the batched pack buffers (zero-copy
                                // F32 view_subrange — F32 is supported, no super-block
                                // alignment needed).
                                let qkv_n = layer_scratch.gdn_qkv_pack.view_subrange(
                                    (n_idx * conv_dim) as u64,
                                    vec![conv_dim as u64],
                                );
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
                                    &alpha_n,
                                    &beta_n,
                                    &normed_n,
                                )?;
                                enc.end();
                            }
                            // The final row already resides in the live session and
                            // cannot be a partial-restore source.
                            if !packed_verify_skip_final_ckpt_enabled() || n_idx + 1 < n {
                                let blit = BlitEncoder::begin(&cmd_buf);
                                let ssm_dst = verify_scratch.gdn_ckpt_slot(gi as u32, n_idx as u32);
                                blit.copy_tensor(&target_session.gdn_state[gi], &ssm_dst);
                                let conv_dst =
                                    verify_scratch.conv_ckpt_slot(gi as u32, n_idx as u32);
                                blit.copy_tensor(&target_session.gdn_conv[gi], &conv_dst);
                                blit.end();
                            }
                        }
                    }
                    emit_mtp_verify_count_phase(trace_counts, il as isize, "gdn", "tail_ckpt");

                    // Step C: batched out_proj across all N. Reads
                    // gdn_normed_pack [N, v_dim], writes mixer_out_pack [N, H].
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &g.out_proj,
                            &gdn_normed_pack,
                            &mixer_out_pack,
                            v_dim,
                            h,
                            n,
                        )?;
                        enc.end();
                    }
                    emit_mtp_verify_count_phase(trace_counts, il as isize, "gdn", "out");
                } else {
                    // Mixed-dtype fall-through: existing per-token
                    // encode_gdn pattern, unchanged. NOTE (v0.425): since
                    // v0.154 added F32 to the mat-mat eligibility set, F32
                    // models take the batched branch above, NOT this one —
                    // the 0.8B "oracle path" is no longer bit-exact vs
                    // token-major (FP32 reduction order differs; see the
                    // layer-major-vs-token-major test doc).
                    for n_idx in 0..n {
                        // Compute pass: stage row, run mixer, capture row.
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_copy_offset_f32(
                                base.ctx,
                                &enc,
                                &h_pack,
                                n_idx * h,
                                &target_session.h,
                                h,
                            )?;
                            base.encode_gdn(&enc, g, gi, target_session)?;
                            encode_scatter_offset_f32(
                                base.ctx,
                                &enc,
                                &target_session.mixer_out,
                                &mixer_out_pack,
                                n_idx * h,
                                h,
                            )?;
                            enc.end();
                        }
                        // The final row already resides in the live session and
                        // cannot be a partial-restore source.
                        if !packed_verify_skip_final_ckpt_enabled() || n_idx + 1 < n {
                            let blit = BlitEncoder::begin(&cmd_buf);
                            let ssm_dst = verify_scratch.gdn_ckpt_slot(gi as u32, n_idx as u32);
                            blit.copy_tensor(&target_session.gdn_state[gi], &ssm_dst);
                            let conv_dst = verify_scratch.conv_ckpt_slot(gi as u32, n_idx as u32);
                            blit.copy_tensor(&target_session.gdn_conv[gi], &conv_dst);
                            blit.end();
                        }
                    }
                    emit_mtp_verify_count_phase(trace_counts, il as isize, "gdn", "fallback_tail");
                }
            }
            MetalBlock::Attn(a) => {
                let ai = attn_idx;
                attn_idx += 1;
                // v0.73c.1: attn projection batching, mirrors v0.73a.1 GDN
                // restructure. Production 27B Q4_K_M attn projections are
                // ALL Q4_K (q gated, k, v, output). Batch them as mat-mat
                // across N=16 in step A/C. RoPE runs either in the fused
                // Step A post-processing kernel or immediately before
                // KV-scatter and attention; gate-sigmoid-mul is batched in C.
                let attn_mat_mat_eligible = prefill_mat_mat_dispatch_eligible;
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
                    let fused_qk_norm_rope = prefill_qk_norm_rope_fused_enabled(n);

                    // v0.76: sized views for adaptive-N back-off.
                    let attn_q_full_pack = layer_scratch
                        .attn_q_full_pack
                        .view_subrange(0, vec![(n * 2 * q_dim) as u64]);
                    // v0.432: attn_q_pack / attn_gate_pack views deleted —
                    // the strided q-norm and strided gate epilogue read the
                    // interleaved attn_q_full_pack directly.
                    let attn_q_normed_pack = layer_scratch
                        .attn_q_normed_pack
                        .view_subrange(0, vec![(n * q_dim) as u64]);
                    let attn_k_now_pack = layer_scratch
                        .attn_k_now_pack
                        .view_subrange(0, vec![(n * kv_dim) as u64]);
                    let attn_v_now_pack = layer_scratch
                        .attn_v_now_pack
                        .view_subrange(0, vec![(n * kv_dim) as u64]);
                    let attn_k_normed_pack = layer_scratch
                        .attn_k_normed_pack
                        .view_subrange(0, vec![(n * kv_dim) as u64]);
                    let attn_o_pack = layer_scratch
                        .attn_o_pack
                        .view_subrange(0, vec![(n * q_dim) as u64]);

                    // Step A: batched Q/K/V projections and Q/K post-processing.
                    // The fused arm includes RoPE; the rollback arm stops after norms.
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        // Q gated (Q + gate interleaved per head).
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &a.q,
                            &h_pack,
                            &attn_q_full_pack,
                            h,
                            2 * q_dim,
                            n,
                        )?;
                        // v0.432: no split_q_gate — the strided q-norm below
                        // reads the Q halves of the interleave directly
                        // (n_heads = N * n_q rows at stride 2*head_dim), and
                        // Step C's strided sigmoid_mul reads the gate halves.
                        // K, V projections.
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &a.k,
                            &h_pack,
                            &attn_k_now_pack,
                            h,
                            kv_dim,
                            n,
                        )?;
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &a.v,
                            &h_pack,
                            &attn_v_now_pack,
                            h,
                            kv_dim,
                            n,
                        )?;
                        if fused_qk_norm_rope {
                            encode_qk_rms_norm_rope_f32_packed_consecutive(
                                base.ctx,
                                &enc,
                                &attn_q_full_pack,
                                &a.q_norm,
                                &attn_q_normed_pack,
                                &attn_k_now_pack,
                                &a.k_norm,
                                &attn_k_normed_pack,
                                n,
                                n_q,
                                n_kv,
                                head_dim,
                                n_rot,
                                start_position,
                                RMS_EPS,
                                arch.rope_theta,
                            )?;
                        } else {
                            // Q-norm (per-head, strided source); n_heads = N * n_q.
                            encode_rms_norm_batched_src_strided_f32(
                                base.ctx,
                                &enc,
                                &attn_q_full_pack,
                                &a.q_norm,
                                &attn_q_normed_pack,
                                n * n_q,
                                head_dim,
                                2 * head_dim,
                                0,
                                RMS_EPS,
                            )?;
                            // K-norm (per-head); n_heads = N * n_kv.
                            encode_rms_norm_batched_f32(
                                base.ctx,
                                &enc,
                                &attn_k_now_pack,
                                &a.k_norm,
                                &attn_k_normed_pack,
                                n * n_kv,
                                head_dim,
                                RMS_EPS,
                            )?;
                        }
                        enc.end();
                    }
                    emit_mtp_verify_count_phase(trace_counts, il as isize, "attn", "front");

                    // V1 (v0.77): shared-KV chunk path, via either the
                    // legacy n==2 gate or the generalized 2..=8 chain.
                    // Both resolve to one `nwg` for a single chunked call
                    // over `[0, start_position + n)`.
                    let shared_kv_shape_ok = head_dim == 256
                        && n_q == 24
                        && n_kv == 4
                        && target_session.kv_k[ai].dtype == GgmlType::F16
                        && target_session.kv_v[ai].dtype == GgmlType::F16;
                    let packed_q2_nwg = if mtp_attn_q2_shared_kv_enabled()
                        && n == 2
                        && start_position as usize + n >= 16_384
                        && shared_kv_shape_ok
                    {
                        let n_pos0 = start_position as usize + 1;
                        let n_pos1 = start_position as usize + 2;
                        let nwg0 = crate::metal::attn_v4_choose_nwg(n_pos0, 6);
                        let nwg1 = crate::metal::attn_v4_choose_nwg(n_pos1, 6);
                        let tile0 = crate::metal::attn_v4_choose_tile_c(n_pos0, 6);
                        let tile1 = crate::metal::attn_v4_choose_tile_c(n_pos1, 6);
                        (nwg0 == nwg1
                            && n_pos0.div_ceil(nwg0) == n_pos1.div_ceil(nwg1)
                            && tile0 == 32
                            && tile1 == 32)
                            .then_some(nwg1)
                    } else if mtp_attn_qn_shared_kv_enabled()
                        && (2..=ATTN_PREFILL_V4_PACKED_ROWS).contains(&n)
                        && shared_kv_shape_ok
                    {
                        // One call covers every row, so only the final
                        // extent's schedule matters. The c32 kernel is a
                        // fixed template: require the heuristic to agree
                        // that 32 is the right tile (same guard the q2
                        // path uses). nwg follows the retuned selector
                        // (2026-08-22: 128/512 tiers for group 4|6); the
                        // encoder and partials support up to
                        // ATTN_V4_MAX_NWG.
                        let n_pos = start_position as usize + n;
                        (crate::metal::attn_v4_choose_tile_c(n_pos, 6) == 32)
                            .then(|| crate::metal::attn_v4_choose_nwg(n_pos, 6))
                    } else {
                        None
                    };

                    if let Some(nwg) = packed_q2_nwg {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        if !fused_qk_norm_rope {
                            encode_prefill_qk_rope(
                                base.ctx,
                                &enc,
                                &attn_q_normed_pack,
                                &attn_k_normed_pack,
                                n,
                                n_q,
                                n_kv,
                                head_dim,
                                n_rot,
                                start_position,
                                arch.rope_theta,
                            )?;
                        }
                        encode_scatter_offset_f32_to_f16_kv(
                            base.ctx,
                            &enc,
                            &attn_k_normed_pack,
                            &attn_v_now_pack,
                            &target_session.kv_k[ai],
                            &target_session.kv_v[ai],
                            start_position as usize * kv_dim,
                            n * kv_dim,
                        )?;
                        target_session.kv_n_pos[ai] = start_position as usize + n;
                        let group = n_q / n_kv;
                        let matrix_mode = mtp_attn_qn_matrix_enabled()
                            && layer_scratch.scratch_plan.modes.enable_attn_matrix
                            && !layer_scratch.scratch_plan.modes.attn_matrix_online
                            && layer_scratch.attn_matrix_max_pos as usize
                                >= start_position as usize + n;
                        if matrix_mode {
                            // Tier-3 matrix reader: KQ (MMA, per-row causal)
                            // -> softmax -> direct-V KQV. No V transpose.
                            let n_pos = start_position as usize + n;
                            let scores = layer_scratch
                                .attn_matrix_scores_pack
                                .view_subrange(0, vec![(n * n_q * n_pos) as u64]);
                            crate::metal::encode_attn_matrix_kq_f32(
                                base.ctx,
                                &enc,
                                &attn_q_normed_pack,
                                &target_session.kv_k[ai],
                                &scores,
                                n,
                                start_position as usize,
                                n_pos,
                                n_kv * head_dim,
                                n_q,
                                n_kv,
                                group,
                                head_dim,
                                true,
                            )?;
                            crate::metal::encode_attn_matrix_softmax_f32(
                                base.ctx,
                                &enc,
                                &scores,
                                n,
                                start_position as usize,
                                n_pos,
                                n_q,
                                n_kv,
                                group,
                                head_dim,
                            )?;
                            crate::metal::encode_attn_matrix_kqv_direct_v_f32(
                                base.ctx,
                                &enc,
                                &scores,
                                &target_session.kv_v[ai],
                                &attn_o_pack,
                                n,
                                start_position as usize,
                                n_pos,
                                n_kv * head_dim,
                                n_q,
                                n_kv,
                                group,
                                head_dim,
                                true,
                            )?;
                        } else {
                            crate::metal::encode_attn_prefill_v4_g6_q2_c32_f32(
                                base.ctx,
                                &enc,
                                &attn_q_normed_pack,
                                &target_session.kv_k[ai],
                                &target_session.kv_v[ai],
                                &layer_scratch.attn_prefill_v4_o_partial_pack,
                                &layer_scratch.attn_prefill_v4_ml_partial_pack,
                                &attn_o_pack,
                                n,
                                start_position as usize,
                                nwg,
                                true,
                            )?;
                        }
                        enc.end();
                    } else {
                        // The generic path appends and attends one causal row at
                        // a time because each row sees a different KV extent.
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
                                if !fused_qk_norm_rope {
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
                                }
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
                                let group = n_q / n_kv;
                                let use_v4 =
                                    head_dim == V4_HEAD_DIM && matches!(group, 4 | 6 | 8 | 16);
                                if use_v4 {
                                    let nwg = crate::metal::attn_v4_choose_nwg(
                                        target_session.kv_n_pos[ai],
                                        group,
                                    );
                                    let tile_c = crate::metal::attn_v4_choose_tile_c(
                                        target_session.kv_n_pos[ai],
                                        group,
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
                    }
                    emit_mtp_verify_count_phase(trace_counts, il as isize, "attn", "body");

                    // Step C: gate-sigmoid + mul (flat elementwise on N*q_dim)
                    // followed by batched o_proj mat-mat. One encoder per layer.
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        // v0.432: fused strided gate epilogue — reads the gate
                        // halves of the interleaved q_proj output in place and
                        // fuses sigmoid+mul (was: split + sigmoid-into-temp +
                        // mul). Matches the single-token default gate math.
                        crate::metal::encode_sigmoid_mul_gate_strided_f32(
                            base.ctx,
                            &enc,
                            &attn_q_full_pack,
                            &attn_o_pack,
                            &attn_o_pack,
                            n * n_q,
                            head_dim,
                            2 * head_dim,
                            head_dim,
                        )?;
                        // Batched O projection: attn_o_pack [N, q_dim] -> mixer_out_pack [N, H].
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            &a.o,
                            &attn_o_pack,
                            &mixer_out_pack,
                            q_dim,
                            h,
                            n,
                        )?;
                        enc.end();
                    }
                    emit_mtp_verify_count_phase(trace_counts, il as isize, "attn", "out");
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
                            &h_pack,
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
                            &mixer_out_pack,
                            n_idx * h,
                            h,
                        )?;
                        enc.end();
                    }
                    emit_mtp_verify_count_phase(trace_counts, il as isize, "attn", "fallback_body");
                }
            }
        }

        // 2c: residual #1 — x_pack += mixer_out_pack (batched
        //     elementwise; encode_add_inplace_f32 just walks the flat
        //     N*H element count).
        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_add_inplace_f32(base.ctx, &enc, &x_pack, &mixer_out_pack)?;
            enc.end();
        }
        emit_mtp_verify_count_phase(trace_counts, il as isize, block_kind, "residual1");

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
                base.ctx, &enc, &x_pack, post_norm, &h_pack, n, h, RMS_EPS,
            )?;
            enc.end();
        }
        emit_mtp_verify_count_phase(trace_counts, il as isize, block_kind, "post_norm");

        if arch.kind == crate::model::ArchKind::Moe {
            let (ffn_gate, ffn_up, ffn_down, moe) = match block {
                MetalBlock::Gdn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
                MetalBlock::Attn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
            };
            let moe = moe.ok_or_else(|| {
                DFlashError::Metal(MetalError::BadShape {
                    kernel: "packed_verify_moe_batched_mixer",
                    detail: "MoE verifier block has no ffn_moe".into(),
                })
            })?;
            let concurrent_ffn = mtp_moe_verify_concurrent_ffn_enabled()
                && moe.gate_exps.dtype == GgmlType::Q4_K
                && moe.up_exps.dtype == GgmlType::Q4_K;
            let topk = arch.expert_used_count.min(arch.expert_count) as usize;
            let n_expert = arch.expert_count as usize;
            let router_mat_mat_eligible = matches!(
                moe.gate_inp.dtype,
                GgmlType::F32
                    | GgmlType::F16
                    | GgmlType::BF16
                    | GgmlType::Q4_K
                    | GgmlType::Q5_K
                    | GgmlType::Q6_K
                    | GgmlType::Q8_0
            );
            let use_batched_route = mtp_moe_verify_batched_route_enabled()
                && mtp_moe_verify_row_views_enabled()
                && topk > 0
                && topk <= 16
                && n_expert > 0
                && n_expert <= 256
                && router_mat_mat_eligible
                && moe.gate_inp_shexp.dtype == GgmlType::F32;
            let ffn_batched = if mtp_moe_verify_grouped_ffn_enabled() {
                let enc = KernelEncoder::begin(&cmd_buf);
                let done = encode_packed_verify_moe_grouped_ffn_after_mixer(
                    base,
                    &enc,
                    block,
                    &x_pack,
                    &h_pack,
                    layer_scratch,
                    n,
                    h,
                )?;
                enc.end();
                emit_mtp_verify_count_phase(trace_counts, il as isize, "moe", "ffn_grouped");
                done
            } else {
                false
            };
            if !ffn_batched {
                let packed_route = if use_batched_route {
                    let router_probs_pack = layer_scratch
                        .moe_router_probs_pack
                        .view_subrange(0, vec![(n * n_expert) as u64]);
                    let topk_idx_pack = layer_scratch
                        .moe_topk_idx_pack
                        .view_subrange(0, vec![(n * topk) as u64]);
                    let topk_weight_pack = layer_scratch
                        .moe_topk_weight_pack
                        .view_subrange(0, vec![(n * topk) as u64]);
                    let shared_gate_pack = layer_scratch
                        .moe_shared_gate_pack
                        .view_subrange(0, vec![n as u64]);
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_moe_route_logits_dispatch(
                            base.ctx,
                            &enc,
                            &moe.gate_inp,
                            &h_pack,
                            &router_probs_pack,
                            h,
                            n_expert,
                            n,
                        )?;
                        encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                            base.ctx,
                            &enc,
                            &router_probs_pack,
                            &moe.gate_inp_shexp,
                            &h_pack,
                            &topk_idx_pack,
                            &topk_weight_pack,
                            &shared_gate_pack,
                            n_expert,
                            topk,
                            h,
                            n,
                        )?;
                        enc.end();
                    }
                    emit_mtp_verify_count_phase(trace_counts, il as isize, "moe", "route_pack");
                    Some((topk_idx_pack, topk_weight_pack, shared_gate_pack))
                } else {
                    None
                };
                if mtp_moe_verify_row_views_enabled() {
                    for n_idx in 0..n {
                        let row_x = x_pack.view_subrange((n_idx * h) as u64, vec![h as u64]);
                        let row_h = h_pack.view_subrange((n_idx * h) as u64, vec![h as u64]);
                        let old_x = std::mem::replace(&mut target_session.x, row_x);
                        let old_h = std::mem::replace(&mut target_session.h, row_h);
                        let route_old = if let Some((idx_pack, weight_pack, shared_pack)) =
                            packed_route.as_ref()
                        {
                            let row_idx =
                                idx_pack.view_subrange((n_idx * topk) as u64, vec![topk as u64]);
                            let row_weight =
                                weight_pack.view_subrange((n_idx * topk) as u64, vec![topk as u64]);
                            let row_shared = shared_pack.view_subrange(n_idx as u64, vec![1_u64]);
                            Some((
                                std::mem::replace(&mut target_session.moe_topk_idx, row_idx),
                                std::mem::replace(&mut target_session.moe_topk_weight, row_weight),
                                std::mem::replace(&mut target_session.moe_shared_gate, row_shared),
                            ))
                        } else {
                            None
                        };
                        let row_result = (|| -> Result<(), DFlashError> {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            if route_old.is_none() {
                                base.encode_moe_route_prepare_by_index(&enc, il, target_session)?;
                            }
                            if concurrent_ffn {
                                enc.end();
                                base.encode_moe_ffn_apply_gpu_concurrent_shared(
                                    &cmd_buf,
                                    target_session,
                                    ffn_gate,
                                    ffn_up,
                                    ffn_down,
                                    moe,
                                )?;
                            } else {
                                base.encode_moe_ffn_apply_gpu(
                                    &enc,
                                    target_session,
                                    ffn_gate,
                                    ffn_up,
                                    ffn_down,
                                    moe,
                                )?;
                                enc.end();
                            }
                            Ok(())
                        })();
                        target_session.x = old_x;
                        target_session.h = old_h;
                        if let Some((old_idx, old_weight, old_shared)) = route_old {
                            target_session.moe_topk_idx = old_idx;
                            target_session.moe_topk_weight = old_weight;
                            target_session.moe_shared_gate = old_shared;
                        }
                        row_result?;
                    }
                } else {
                    for n_idx in 0..n {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_copy_offset_f32(
                            base.ctx,
                            &enc,
                            &x_pack,
                            n_idx * h,
                            &target_session.x,
                            h,
                        )?;
                        encode_copy_offset_f32(
                            base.ctx,
                            &enc,
                            &h_pack,
                            n_idx * h,
                            &target_session.h,
                            h,
                        )?;
                        base.encode_moe_route_prepare_by_index(&enc, il, target_session)?;
                        if concurrent_ffn {
                            enc.end();
                            base.encode_moe_ffn_apply_gpu_concurrent_shared(
                                &cmd_buf,
                                target_session,
                                ffn_gate,
                                ffn_up,
                                ffn_down,
                                moe,
                            )?;
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_scatter_offset_f32(
                                base.ctx,
                                &enc,
                                &target_session.x,
                                &x_pack,
                                n_idx * h,
                                h,
                            )?;
                            enc.end();
                        } else {
                            base.encode_moe_ffn_apply_gpu(
                                &enc,
                                target_session,
                                ffn_gate,
                                ffn_up,
                                ffn_down,
                                moe,
                            )?;
                            encode_scatter_offset_f32(
                                base.ctx,
                                &enc,
                                &target_session.x,
                                &x_pack,
                                n_idx * h,
                                h,
                            )?;
                            enc.end();
                        }
                    }
                }
                emit_mtp_verify_count_phase(trace_counts, il as isize, "moe", "ffn_row_loop");
            }
            for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                if lid as usize == il {
                    let enc = KernelEncoder::begin(&cmd_buf);
                    for n_idx in 0..n {
                        let elem_off = (n_idx as u64 * verify_scratch.k_target_layers as u64
                            + k_idx as u64)
                            * verify_scratch.hidden_size;
                        let x_n = x_pack.view_subrange((n_idx * h) as u64, vec![h as u64]);
                        encode_scatter_offset_f32(
                            base.ctx,
                            &enc,
                            &x_n,
                            &verify_scratch.hidden_capture,
                            elem_off as usize,
                            h,
                        )?;
                    }
                    enc.end();
                }
            }
            emit_mtp_verify_count_phase(trace_counts, il as isize, "moe", "capture");
            continue;
        }

        // 2f: SwiGLU FFN. Dtype dispatch (codex Q5) — the WIN.
        //   Q4_K weights → batched mat-mat (mat_mat_q4_k_f32) writing
        //                  ffn_gate_pack [N, F] then ffn_up_pack [N, F]
        //                  row-major (= mat-mat output bit-equivalent),
        //                  then silu_mul on flat N*F elements,
        //                  then mat-mat ffn_down → ffn_out_pack [N, H].
        //   F32 weights   → since v0.154 F32 is mat-mat eligible and
        //                  takes the batched path like the K-quants; the
        //                  per-token mat-vec loop below is only the
        //                  fall-through for dtypes without a mat-mat
        //                  kernel. (Historically F32 was kept per-token
        //                  as a bit-exact oracle path; that guarantee is
        //                  gone — see the layer-major-vs-token-major
        //                  test doc.)
        let (g_w, u_w, d_w) = match block {
            MetalBlock::Gdn(g) => (&g.ffn_gate, &g.ffn_up, &g.ffn_down),
            MetalBlock::Attn(a) => (&a.ffn_gate, &a.ffn_up, &a.ffn_down),
        };
        // Per-weight dtype dispatch (codex Q5: dispatch INSIDE the
        // function so rollback bugs stay localizable). Layer-major
        // wins via mat-mat for all dtypes supported by encode_mat_mat_dispatch;
        // unsupported dtypes fall back to the per-token mat-vec loop.
        //
        // Production 27B Q4_K_M: ffn_gate / ffn_up are Q4_K, ffn_down
        // is Q6_K. Both legs hit the mat-mat fast path.
        let mat_mat_eligible = prefill_mat_mat_dispatch_eligible;
        let mat_mat_path = mat_mat_eligible(g_w.dtype)
            && mat_mat_eligible(u_w.dtype)
            && mat_mat_eligible(d_w.dtype);
        {
            let enc = KernelEncoder::begin(&cmd_buf);
            if mat_mat_path {
                let fused_q4 = dflash_verify_fused_ffn_q4_enabled()
                    && n == 8
                    && h.is_multiple_of(256)
                    && f.is_multiple_of(8)
                    && g_w.dtype == GgmlType::Q4_K
                    && u_w.dtype == GgmlType::Q4_K;
                if fused_q4 {
                    encode_ffn_fused_swiglu_q4_k_mma8_f32(
                        base.ctx,
                        &enc,
                        g_w,
                        u_w,
                        &h_pack,
                        &ffn_inner_pack,
                        h,
                        f,
                    )?;
                } else {
                    encode_mat_mat_dispatch(base.ctx, &enc, g_w, &h_pack, &ffn_gate_pack, h, f, n)?;
                    encode_mat_mat_dispatch(base.ctx, &enc, u_w, &h_pack, &ffn_up_pack, h, f, n)?;
                    encode_silu_mul_f32(
                        base.ctx,
                        &enc,
                        &ffn_gate_pack,
                        &ffn_up_pack,
                        &ffn_inner_pack,
                    )?;
                }
                encode_mat_mat_dispatch(
                    base.ctx,
                    &enc,
                    d_w,
                    &ffn_inner_pack,
                    &ffn_out_pack,
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
            encode_add_inplace_f32(base.ctx, &enc, &x_pack, &ffn_out_pack)?;

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
        emit_mtp_verify_count_phase(trace_counts, il as isize, block_kind, "dense_ffn");
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
    // Falls back to per-token mat-vec for dtypes without a mat-mat kernel.
    let lm_dtype = base.model.lm_head.dtype;
    let lm_mat_mat_path = prefill_mat_mat_dispatch_eligible(lm_dtype);
    {
        let enc = KernelEncoder::begin(&cmd_buf);
        if lm_mat_mat_path {
            // Batched final norm: x_pack → h_pack.
            encode_rms_norm_batched_f32(
                base.ctx,
                &enc,
                &x_pack,
                &base.model.output_norm,
                &h_pack,
                n,
                h,
                RMS_EPS,
            )?;
            // Pick logits destination: debug_logits_dst if provided
            // (same [N, V] shape; saves a scatter), else
            // final_logits_pack.
            let logits_dst = match debug_logits_dst {
                Some(dst) => dst,
                None => &final_logits_pack,
            };
            // Batched lm_head mat-mat.
            encode_mat_mat_dispatch(
                base.ctx,
                &enc,
                &base.model.lm_head,
                &h_pack,
                logits_dst,
                h,
                v,
                n,
            )?;
            // Batched argmax across all N rows in ONE dispatch, plus the
            // per-row (top1 - top2) gap for the margin-guarded fallback.
            encode_argmax_top2_f32(
                base.ctx,
                &enc,
                logits_dst,
                &verify_argmax_view,
                &verify_gap_view,
                n,
                v,
            )?;
        } else {
            // F32 / unsupported lm_head: per-token mat-vec fallback
            // (the original layer-major tail). Layer-major still wins
            // through the batched final norm only.
            for n_idx in 0..n {
                encode_copy_offset_f32(base.ctx, &enc, &x_pack, n_idx * h, &target_session.x, h)?;
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
                let gap_dst = verify_scratch.gap_slot(n_idx as u32);
                encode_argmax_top2_f32(
                    base.ctx,
                    &enc,
                    &target_session.logits,
                    &argmax_dst,
                    &gap_dst,
                    1,
                    v,
                )?;
            }
        }
        enc.end();
    }
    emit_mtp_verify_count_phase(trace_counts, -1, "tail", "lm_head_argmax");

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
    let (logits, _) = prefill_tokens_with_multi_hidden_profiled(
        base,
        token_ids,
        start_position,
        target_session,
        layer_scratch,
        target_layer_ids,
        hidden_dst,
    )?;
    Ok(logits)
}

pub fn prefill_tokens_with_multi_hidden_profiled(
    base: &MetalForward<'_>,
    token_ids: &[i32],
    start_position: u32,
    target_session: &mut MetalSession,
    layer_scratch: &mut MetalDFlashLayerMajorScratch,
    target_layer_ids: &[u32],
    hidden_dst: Option<&MetalTensor>,
) -> Result<(Vec<f32>, f64), DFlashError> {
    let (logits, gpu_ms, _) = prefill_tokens_with_multi_hidden_profiled_inner(
        base,
        token_ids,
        start_position,
        target_session,
        layer_scratch,
        target_layer_ids,
        hidden_dst,
        PrefillTailMode::ReadLogits,
        None,
    )?;
    let logits = logits.ok_or_else(|| {
        DFlashError::Metal(MetalError::BadShape {
            kernel: "prefill_tokens_with_multi_hidden_profiled",
            detail: "internal tail mode returned no logits".into(),
        })
    })?;
    Ok((logits, gpu_ms))
}

pub fn prefill_tokens_prompt_only_profiled(
    base: &MetalForward<'_>,
    token_ids: &[i32],
    start_position: u32,
    target_session: &mut MetalSession,
    layer_scratch: &mut MetalDFlashLayerMajorScratch,
) -> Result<f64, DFlashError> {
    let (_, gpu_ms, _) = prefill_tokens_with_multi_hidden_profiled_inner(
        base,
        token_ids,
        start_position,
        target_session,
        layer_scratch,
        &[],
        None,
        PrefillTailMode::SkipTail,
        None,
    )?;
    Ok(gpu_ms)
}

/// Packed prompt prefill with post-block capture and no final norm, LM head,
/// logits readback, or decode-tail work.
pub fn prefill_tokens_with_multi_hidden_prompt_only_profiled(
    base: &MetalForward<'_>,
    token_ids: &[i32],
    start_position: u32,
    target_session: &mut MetalSession,
    layer_scratch: &mut MetalDFlashLayerMajorScratch,
    target_layer_ids: &[u32],
    hidden_dst: &MetalTensor,
) -> Result<f64, DFlashError> {
    let (_, gpu_ms, _) = prefill_tokens_with_multi_hidden_profiled_inner(
        base,
        token_ids,
        start_position,
        target_session,
        layer_scratch,
        target_layer_ids,
        Some(hidden_dst),
        PrefillTailMode::SkipTail,
        None,
    )?;
    Ok(gpu_ms)
}

#[doc(hidden)]
pub fn prefill_tokens_profiled_with_tail(
    base: &MetalForward<'_>,
    token_ids: &[i32],
    start_position: u32,
    target_session: &mut MetalSession,
    layer_scratch: &mut MetalDFlashLayerMajorScratch,
    tail: LmHeadTail<'_>,
) -> Result<(f64, LmHeadTailEvidence), DFlashError> {
    let (_, gpu_ms, evidence) = prefill_tokens_with_multi_hidden_profiled_inner(
        base,
        token_ids,
        start_position,
        target_session,
        layer_scratch,
        &[],
        None,
        PrefillTailMode::Supplied(tail),
        None,
    )?;
    Ok((
        gpu_ms,
        evidence.ok_or_else(|| {
            DFlashError::Metal(MetalError::BadShape {
                kernel: "prefill_tokens_profiled_with_tail",
                detail: "supplied tail returned no evidence".into(),
            })
        })?,
    ))
}

/// Bench-only prefill entry point. The ordinary prefill wrappers always pass no capture.
pub fn prefill_tokens_attention_capture(
    base: &MetalForward<'_>,
    token_ids: &[i32],
    start_position: u32,
    target_session: &mut MetalSession,
    layer_scratch: &mut MetalDFlashLayerMajorScratch,
    capture: &mut AttentionCapture,
) -> Result<f64, DFlashError> {
    let (_, gpu_ms, _) = prefill_tokens_with_multi_hidden_profiled_inner(
        base,
        token_ids,
        start_position,
        target_session,
        layer_scratch,
        &[],
        None,
        PrefillTailMode::SkipTail,
        Some(capture),
    )?;
    capture.validate_complete()?;
    Ok(gpu_ms)
}

type ProfiledPrefillResult = (Option<Vec<f32>>, f64, Option<LmHeadTailEvidence>);

fn prefill_tokens_with_multi_hidden_profiled_inner<'a>(
    base: &MetalForward<'_>,
    token_ids: &[i32],
    start_position: u32,
    target_session: &mut MetalSession,
    layer_scratch: &mut MetalDFlashLayerMajorScratch,
    target_layer_ids: &[u32],
    hidden_dst: Option<&MetalTensor>,
    tail_mode: PrefillTailMode<'a>,
    mut attention_capture: Option<&mut AttentionCapture>,
) -> Result<ProfiledPrefillResult, DFlashError> {
    let arch = &base.model.arch;
    let total_n = token_ids.len();
    let h = arch.hidden_size as usize;
    let f = arch.intermediate_size as usize;
    let v = arch.vocab_size as usize;
    let k_target = target_layer_ids.len();
    let mut tail_evidence = match tail_mode {
        PrefillTailMode::Supplied(tail) => {
            if let LmHeadTail::CompactQ6K { weight, output } = tail
                && (layer_scratch.aliases_mutable_buffer(weight)
                    || layer_scratch.aliases_mutable_buffer(output))
            {
                return Err(DFlashError::Metal(MetalError::BadShape {
                    kernel: "prefill_lm_head_tail",
                    detail: "compact tail aliases mutable packed-prefill scratch".into(),
                }));
            }
            Some(base.validate_lm_head_tail(target_session, tail)?)
        }
        PrefillTailMode::ReadLogits | PrefillTailMode::SkipTail => None,
    };
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
    let attn_matrix_g4_force_on = prefill_attn_matrix_g4_mode() == PrefillEnvMode::ForceOn
        && arch.attn_head_dim as usize == 256
        && arch.n_q_heads == arch.n_kv_heads * 4;
    let attn_matrix_g8_force_on = prefill_attn_matrix_g8_mode() == PrefillEnvMode::ForceOn
        && arch.attn_head_dim as usize == 256
        && arch.n_q_heads == 16
        && arch.n_kv_heads == 2;
    let attn_matrix_g6_force_on = prefill_attn_matrix_g6_mode() == PrefillEnvMode::ForceOn
        && arch.attn_head_dim as usize == 256
        && arch.n_q_heads == 24
        && arch.n_kv_heads == 4;
    let attn_matrix_g16_force_on = prefill_attn_matrix_g16_mode() == PrefillEnvMode::ForceOn
        && arch.attn_head_dim as usize == 256
        && arch.n_q_heads == 32
        && arch.n_kv_heads == 2;
    if (attn_matrix_g4_force_on
        || attn_matrix_g8_force_on
        || attn_matrix_g6_force_on
        || attn_matrix_g16_force_on)
        && layer_scratch.attn_matrix_max_pos < last_pos as u64
    {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "prefill_attn_matrix",
            detail: format!(
                "matrix scratch max_pos={} < required last_pos={last_pos}; set QWEN_PREFILL_ATTN_MATRIX_MAX_POS before scratch allocation",
                layer_scratch.attn_matrix_max_pos
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
    let n_attn_actual = base
        .model
        .blocks
        .iter()
        .filter(|b| matches!(b, MetalBlock::Attn(_)))
        .count();
    let mut attn_matrix_vt_valid_until = vec![0usize; n_attn_actual];

    // Cache mat-mat-eligible predicates once.
    let gdn_mat_mat_eligible = prefill_mat_mat_dispatch_eligible;
    let attn_mat_mat_eligible = prefill_mat_mat_dispatch_eligible;
    let ffn_mat_mat_eligible = prefill_mat_mat_dispatch_eligible;
    // Callsite-scoped specialization for GDN beta/alpha prompt prefill. Do not
    // reuse this helper as a generic F32 skinny mat-mat dispatcher without a new
    // shape/correctness gate.
    let gdn_skinny_mat_mat = |enc: &KernelEncoder,
                              weight: &MetalTensor,
                              x: &MetalTensor,
                              y: &MetalTensor,
                              n_in: usize,
                              n_out: usize,
                              n_query: usize|
     -> Result<(), DFlashError> {
        if prefill_gdn_skinny_f32_e8p32_enabled()
            && weight.dtype == GgmlType::F32
            && n_in.is_multiple_of(4)
            && n_out.is_multiple_of(8)
        {
            crate::metal::encode_mat_mat_f32_router_e8p32(
                base.ctx, enc, weight, x, y, n_in, n_out, n_query,
            )?;
        } else {
            encode_mat_mat_dispatch(base.ctx, enc, weight, x, y, n_in, n_out, n_query)?;
        }
        Ok(())
    };

    // Per-call IDs buffer. P=16 I32 = 64 bytes; trivial allocation cost.
    let ids_buf =
        MetalTensor::zeros_i32(base.ctx, vec![p_max as u64]).map_err(DFlashError::Metal)?;

    let n_chunks = total_n.div_ceil(p_max);
    let mut prefill_gpu_total_ms = 0.0f64;
    for chunk_idx in 0..n_chunks {
        let chunk_wall = Instant::now();
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
        let mut cmd_buf = base.ctx.queue.commandBuffer().expect("command buffer");
        let trace_layer_phases = prefill_trace_layer_phases_enabled();
        let trace_wall = prefill_trace_wall_enabled();
        let trace_counts = prefill_trace_counts_enabled();
        let _kernel_trace_guard = (trace_wall || trace_counts).then(kernel_trace_begin);
        let trace_moe_buckets = trace_layer_phases && prefill_trace_moe_buckets_enabled();

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
        let chunk_encode_start = Instant::now();

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
        emit_prefill_count_phase(trace_counts, chunk_idx, chunk_start, 0, "chunk", "embed");

        // === Phase 2: per-layer body. ===
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in base.model.blocks.iter().enumerate() {
            let mut apply_mixer_residual = true;
            let block_kind = match block {
                MetalBlock::Gdn(_) => "gdn",
                MetalBlock::Attn(_) => "attn",
            };
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
            flush_prefill_layer_phase(
                base.ctx,
                &mut cmd_buf,
                &mut prefill_gpu_total_ms,
                trace_layer_phases,
                chunk_idx,
                chunk_start,
                il,
                block_kind,
                "pre_norm",
            )?;

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

                    let gdn_batched = prefill_gdn_batched_enabled()
                        && gdn_mat_mat_eligible(g.in_proj_qkv.dtype)
                        && gdn_mat_mat_eligible(g.in_proj_z.dtype)
                        && gdn_mat_mat_eligible(g.out_proj.dtype);
                    let gdn_split = prefill_gdn_split_mode();

                    if gdn_batched {
                        // Sized views of GDN pack scratch.
                        let gdn_qkv_pack_p = layer_scratch
                            .gdn_qkv_pack
                            .view_subrange(0, vec![(chunk_p * conv_dim) as u64]);
                        let gdn_z_pack_p = layer_scratch
                            .gdn_z_pack
                            .view_subrange(0, vec![(chunk_p * v_dim) as u64]);
                        let gdn_beta_pack_p = layer_scratch
                            .gdn_beta_pack
                            .view_subrange(0, vec![(chunk_p * n_v_u) as u64]);
                        let gdn_alpha_pack_p = layer_scratch
                            .gdn_alpha_pack
                            .view_subrange(0, vec![(chunk_p * n_v_u) as u64]);
                        let gdn_q_norm_pack_p = layer_scratch
                            .gdn_q_norm_pack
                            .view_subrange(0, vec![(chunk_p * n_k_u * head_dim_u) as u64]);
                        let gdn_k_norm_pack_p = layer_scratch
                            .gdn_k_norm_pack
                            .view_subrange(0, vec![(chunk_p * n_k_u * head_dim_u) as u64]);
                        let gdn_v_pack_p = layer_scratch
                            .gdn_v_pack
                            .view_subrange(0, vec![(chunk_p * v_dim) as u64]);
                        let gdn_out_pack_p = layer_scratch
                            .gdn_out_pack
                            .view_subrange(0, vec![(chunk_p * v_dim) as u64]);
                        let gdn_normed_pack_p = layer_scratch
                            .gdn_normed_pack
                            .view_subrange(0, vec![(chunk_p * v_dim) as u64]);

                        // Step A: batched front-end QKV / Z projections.
                        if trace_layer_phases {
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                if prefill_gdn_matvec_projection_enabled("qkv", il) {
                                    encode_packed_matvec_projection(
                                        base.ctx,
                                        &enc,
                                        &g.in_proj_qkv,
                                        &h_pack_p,
                                        &gdn_qkv_pack_p,
                                        h,
                                        conv_dim,
                                        chunk_p,
                                    )?;
                                } else {
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
                                }
                                enc.end();
                            }
                            flush_prefill_layer_phase(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                trace_layer_phases,
                                chunk_idx,
                                chunk_start,
                                il,
                                "gdn",
                                "gdn_qkv",
                            )?;
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                if prefill_gdn_matvec_projection_enabled("z", il) {
                                    encode_packed_matvec_projection(
                                        base.ctx,
                                        &enc,
                                        &g.in_proj_z,
                                        &h_pack_p,
                                        &gdn_z_pack_p,
                                        h,
                                        v_dim,
                                        chunk_p,
                                    )?;
                                } else {
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
                                }
                                enc.end();
                            }
                            flush_prefill_layer_phase(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                trace_layer_phases,
                                chunk_idx,
                                chunk_start,
                                il,
                                "gdn",
                                "gdn_z",
                            )?;
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                if prefill_gdn_matvec_projection_enabled("beta", il) {
                                    encode_packed_matvec_projection(
                                        base.ctx,
                                        &enc,
                                        &g.beta_proj,
                                        &h_pack_p,
                                        &gdn_beta_pack_p,
                                        h,
                                        n_v_u,
                                        chunk_p,
                                    )?;
                                } else {
                                    gdn_skinny_mat_mat(
                                        &enc,
                                        &g.beta_proj,
                                        &h_pack_p,
                                        &gdn_beta_pack_p,
                                        h,
                                        n_v_u,
                                        chunk_p,
                                    )?;
                                }
                                if prefill_gdn_matvec_projection_enabled("alpha", il) {
                                    encode_packed_matvec_projection(
                                        base.ctx,
                                        &enc,
                                        &g.alpha_proj,
                                        &h_pack_p,
                                        &gdn_alpha_pack_p,
                                        h,
                                        n_v_u,
                                        chunk_p,
                                    )?;
                                } else {
                                    gdn_skinny_mat_mat(
                                        &enc,
                                        &g.alpha_proj,
                                        &h_pack_p,
                                        &gdn_alpha_pack_p,
                                        h,
                                        n_v_u,
                                        chunk_p,
                                    )?;
                                }
                                enc.end();
                            }
                            flush_prefill_layer_phase(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                trace_layer_phases,
                                chunk_idx,
                                chunk_start,
                                il,
                                "gdn",
                                "gdn_beta_alpha",
                            )?;
                        } else {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            if prefill_gdn_matvec_projection_enabled("qkv", il) {
                                encode_packed_matvec_projection(
                                    base.ctx,
                                    &enc,
                                    &g.in_proj_qkv,
                                    &h_pack_p,
                                    &gdn_qkv_pack_p,
                                    h,
                                    conv_dim,
                                    chunk_p,
                                )?;
                            } else {
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
                            }
                            if prefill_gdn_matvec_projection_enabled("z", il) {
                                encode_packed_matvec_projection(
                                    base.ctx,
                                    &enc,
                                    &g.in_proj_z,
                                    &h_pack_p,
                                    &gdn_z_pack_p,
                                    h,
                                    v_dim,
                                    chunk_p,
                                )?;
                            } else {
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
                            }
                            if prefill_gdn_matvec_projection_enabled("beta", il) {
                                encode_packed_matvec_projection(
                                    base.ctx,
                                    &enc,
                                    &g.beta_proj,
                                    &h_pack_p,
                                    &gdn_beta_pack_p,
                                    h,
                                    n_v_u,
                                    chunk_p,
                                )?;
                            } else {
                                gdn_skinny_mat_mat(
                                    &enc,
                                    &g.beta_proj,
                                    &h_pack_p,
                                    &gdn_beta_pack_p,
                                    h,
                                    n_v_u,
                                    chunk_p,
                                )?;
                            }
                            if prefill_gdn_matvec_projection_enabled("alpha", il) {
                                encode_packed_matvec_projection(
                                    base.ctx,
                                    &enc,
                                    &g.alpha_proj,
                                    &h_pack_p,
                                    &gdn_alpha_pack_p,
                                    h,
                                    n_v_u,
                                    chunk_p,
                                )?;
                            } else {
                                gdn_skinny_mat_mat(
                                    &enc,
                                    &g.alpha_proj,
                                    &h_pack_p,
                                    &gdn_alpha_pack_p,
                                    h,
                                    n_v_u,
                                    chunk_p,
                                )?;
                            }
                            enc.end();
                        }

                        if prefill_gdn_proj_oracle_layer_enabled(il) {
                            diagnose_gdn_projection_matmat_vs_matvec(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                chunk_idx,
                                chunk_start,
                                il,
                                "qkv",
                                &g.in_proj_qkv,
                                &h_pack_p,
                                &gdn_qkv_pack_p,
                                h,
                                conv_dim,
                                chunk_p,
                            )?;
                            diagnose_gdn_projection_matmat_vs_matvec(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                chunk_idx,
                                chunk_start,
                                il,
                                "z",
                                &g.in_proj_z,
                                &h_pack_p,
                                &gdn_z_pack_p,
                                h,
                                v_dim,
                                chunk_p,
                            )?;
                            diagnose_gdn_projection_matmat_vs_matvec(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                chunk_idx,
                                chunk_start,
                                il,
                                "beta",
                                &g.beta_proj,
                                &h_pack_p,
                                &gdn_beta_pack_p,
                                h,
                                n_v_u,
                                chunk_p,
                            )?;
                            diagnose_gdn_projection_matmat_vs_matvec(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                chunk_idx,
                                chunk_start,
                                il,
                                "alpha",
                                &g.alpha_proj,
                                &h_pack_p,
                                &gdn_alpha_pack_p,
                                h,
                                n_v_u,
                                chunk_p,
                            )?;
                        }

                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_sigmoid_f32(base.ctx, &enc, &gdn_beta_pack_p, &gdn_beta_pack_p)?;
                            encode_gdn_decay_chain_batched_f32(
                                base.ctx,
                                &enc,
                                &gdn_alpha_pack_p,
                                &g.dt_bias,
                                &g.a_log,
                                &gdn_alpha_pack_p,
                                chunk_p,
                                n_v_u,
                            )?;
                            enc.end();
                        }
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            "gdn",
                            "gdn_alpha_beta",
                        )?;

                        if matches!(gdn_split, PrefillGdnSplitMode::SkipAll) {
                            apply_mixer_residual = false;
                        } else if dense_packed_gdn_step_enabled() {
                            if gdn_split.run_prep() {
                                if trace_layer_phases {
                                    {
                                        let enc = KernelEncoder::begin(&cmd_buf);
                                        encode_gdn_prep_packed_f32(
                                            base.ctx,
                                            &enc,
                                            &gdn_qkv_pack_p,
                                            &target_session.gdn_conv[gi],
                                            &g.conv1d,
                                            &gdn_q_norm_pack_p,
                                            &gdn_k_norm_pack_p,
                                            &gdn_v_pack_p,
                                            chunk_p,
                                            n_k_u,
                                            n_v_u,
                                            head_dim_u,
                                        )?;
                                        enc.end();
                                    }
                                    flush_prefill_layer_phase(
                                        base.ctx,
                                        &mut cmd_buf,
                                        &mut prefill_gpu_total_ms,
                                        trace_layer_phases,
                                        chunk_idx,
                                        chunk_start,
                                        il,
                                        "gdn",
                                        "gdn_prep_conv",
                                    )?;

                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    if prefill_gdn_pair_l2_enabled() {
                                        encode_l2_norm_pair_batched_f32(
                                            base.ctx,
                                            &enc,
                                            &gdn_q_norm_pack_p,
                                            &gdn_q_norm_pack_p,
                                            &gdn_k_norm_pack_p,
                                            &gdn_k_norm_pack_p,
                                            chunk_p * n_k_u,
                                            head_dim_u,
                                            RMS_EPS,
                                        )?;
                                    } else {
                                        encode_l2_norm_batched_f32(
                                            base.ctx,
                                            &enc,
                                            &gdn_q_norm_pack_p,
                                            &gdn_q_norm_pack_p,
                                            chunk_p * n_k_u,
                                            head_dim_u,
                                            RMS_EPS,
                                        )?;
                                        encode_l2_norm_batched_f32(
                                            base.ctx,
                                            &enc,
                                            &gdn_k_norm_pack_p,
                                            &gdn_k_norm_pack_p,
                                            chunk_p * n_k_u,
                                            head_dim_u,
                                            RMS_EPS,
                                        )?;
                                    }
                                    enc.end();
                                    flush_prefill_layer_phase(
                                        base.ctx,
                                        &mut cmd_buf,
                                        &mut prefill_gpu_total_ms,
                                        trace_layer_phases,
                                        chunk_idx,
                                        chunk_start,
                                        il,
                                        "gdn",
                                        "gdn_prep_l2",
                                    )?;
                                } else {
                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    encode_gdn_prep_packed_f32(
                                        base.ctx,
                                        &enc,
                                        &gdn_qkv_pack_p,
                                        &target_session.gdn_conv[gi],
                                        &g.conv1d,
                                        &gdn_q_norm_pack_p,
                                        &gdn_k_norm_pack_p,
                                        &gdn_v_pack_p,
                                        chunk_p,
                                        n_k_u,
                                        n_v_u,
                                        head_dim_u,
                                    )?;
                                    if prefill_gdn_pair_l2_enabled() {
                                        encode_l2_norm_pair_batched_f32(
                                            base.ctx,
                                            &enc,
                                            &gdn_q_norm_pack_p,
                                            &gdn_q_norm_pack_p,
                                            &gdn_k_norm_pack_p,
                                            &gdn_k_norm_pack_p,
                                            chunk_p * n_k_u,
                                            head_dim_u,
                                            RMS_EPS,
                                        )?;
                                    } else {
                                        encode_l2_norm_batched_f32(
                                            base.ctx,
                                            &enc,
                                            &gdn_q_norm_pack_p,
                                            &gdn_q_norm_pack_p,
                                            chunk_p * n_k_u,
                                            head_dim_u,
                                            RMS_EPS,
                                        )?;
                                        encode_l2_norm_batched_f32(
                                            base.ctx,
                                            &enc,
                                            &gdn_k_norm_pack_p,
                                            &gdn_k_norm_pack_p,
                                            chunk_p * n_k_u,
                                            head_dim_u,
                                            RMS_EPS,
                                        )?;
                                    }
                                    enc.end();
                                    flush_prefill_layer_phase(
                                        base.ctx,
                                        &mut cmd_buf,
                                        &mut prefill_gpu_total_ms,
                                        trace_layer_phases,
                                        chunk_idx,
                                        chunk_start,
                                        il,
                                        "gdn",
                                        "gdn_prep",
                                    )?;
                                }
                            }

                            if gdn_split.run_step() {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                encode_gdn_step_decay_packed_f32(
                                    base.ctx,
                                    &enc,
                                    &gdn_q_norm_pack_p,
                                    &gdn_k_norm_pack_p,
                                    &gdn_v_pack_p,
                                    &gdn_alpha_pack_p,
                                    &gdn_beta_pack_p,
                                    &target_session.gdn_state[gi],
                                    &gdn_out_pack_p,
                                    chunk_p,
                                    n_v_u,
                                    n_k_u,
                                    head_dim_u,
                                )?;
                                enc.end();
                                flush_prefill_layer_phase(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                    trace_layer_phases,
                                    chunk_idx,
                                    chunk_start,
                                    il,
                                    "gdn",
                                    "gdn_step",
                                )?;
                            }

                            if gdn_split.run_gated() {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                encode_rmsnorm_gated_f32(
                                    base.ctx,
                                    &enc,
                                    &gdn_out_pack_p,
                                    &g.norm,
                                    &gdn_z_pack_p,
                                    &gdn_normed_pack_p,
                                    chunk_p * n_v_u,
                                    head_dim_u,
                                    RMS_EPS * head_dim_u as f32,
                                )?;
                                enc.end();
                                flush_prefill_layer_phase(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                    trace_layer_phases,
                                    chunk_idx,
                                    chunk_start,
                                    il,
                                    "gdn",
                                    "gdn_gated",
                                )?;
                            } else if gdn_split.needs_zero_normed() {
                                zero_f32_tensor(&gdn_normed_pack_p);
                            }
                        } else {
                            // Per-token recurrence (GDN state mutates per token).
                            // NO ckpt blit — prefill never rolls back.
                            for n_idx in 0..chunk_p {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                let qkv_n = gdn_qkv_pack_p.view_subrange(
                                    (n_idx * conv_dim) as u64,
                                    vec![conv_dim as u64],
                                );
                                let z_n = gdn_z_pack_p
                                    .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                                let beta_n = gdn_beta_pack_p
                                    .view_subrange((n_idx * n_v_u) as u64, vec![n_v_u as u64]);
                                let alpha_n = gdn_alpha_pack_p
                                    .view_subrange((n_idx * n_v_u) as u64, vec![n_v_u as u64]);
                                let normed_n = gdn_normed_pack_p
                                    .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                                base.encode_gdn_tail(
                                    &enc,
                                    g,
                                    gi,
                                    target_session,
                                    &qkv_n,
                                    &z_n,
                                    &alpha_n,
                                    &beta_n,
                                    &normed_n,
                                )?;
                                enc.end();
                            }
                            flush_prefill_layer_phase(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                trace_layer_phases,
                                chunk_idx,
                                chunk_start,
                                il,
                                "gdn",
                                "gdn_tail",
                            )?;
                        }

                        if apply_mixer_residual {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            if prefill_gdn_matvec_projection_enabled("out", il) {
                                encode_packed_matvec_projection(
                                    base.ctx,
                                    &enc,
                                    &g.out_proj,
                                    &gdn_normed_pack_p,
                                    &mixer_out_pack_p,
                                    v_dim,
                                    h,
                                    chunk_p,
                                )?;
                            } else {
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
                            }
                            enc.end();
                            if prefill_gdn_proj_oracle_layer_enabled(il)
                                && !prefill_gdn_matvec_projection_enabled("out", il)
                            {
                                diagnose_gdn_projection_matmat_vs_matvec(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                    chunk_idx,
                                    chunk_start,
                                    il,
                                    "out",
                                    &g.out_proj,
                                    &gdn_normed_pack_p,
                                    &mixer_out_pack_p,
                                    v_dim,
                                    h,
                                    chunk_p,
                                )?;
                            } else {
                                flush_prefill_layer_phase(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                    trace_layer_phases,
                                    chunk_idx,
                                    chunk_start,
                                    il,
                                    "gdn",
                                    "gdn_back",
                                )?;
                            }
                        }
                    } else {
                        // F32 oracle / mixed-dtype fallback: per-token
                        // encode_gdn (no ckpt blit).
                        if prefill_noop_gdn_body_enabled() {
                            apply_mixer_residual = false;
                        } else {
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
                            flush_prefill_layer_phase(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                trace_layer_phases,
                                chunk_idx,
                                chunk_start,
                                il,
                                "gdn",
                                "gdn_fallback",
                            )?;
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
                    let trace_attn_phases = prefill_trace_attn_phases_enabled();
                    let attn_q_full_dim = 2 * q_dim;

                    if attn_batched {
                        // Sized views of attn pack scratch.
                        let q_full_pack_p = layer_scratch
                            .attn_q_full_pack
                            .view_subrange(0, vec![(chunk_p * 2 * q_dim) as u64]);
                        // v0.432: attn_q_pack / attn_gate_pack views deleted —
                        // the strided q-norm and strided gate epilogue read
                        // the interleaved q_full pack directly.
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
                        let use_fused_qkv =
                            layer_scratch.scratch_plan.modes.enable_attn_fused_qkv_g8
                                && prefill_attn_fused_qkv_g8_enabled(
                                    chunk_start as usize + chunk_p,
                                    n_q / n_kv,
                                )
                                && a.qkv_fused.is_some();
                        let fused_qk_norm_rope = prefill_qk_norm_rope_fused_enabled(chunk_p)
                            && !prefill_noop_attn_body_enabled();
                        let qkv_fused_dim = attn_q_full_dim + 2 * kv_dim;

                        // Step A: batched Q/K/V projections and Q/K post-processing.
                        if trace_attn_phases {
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                label_prefill_encoder(
                                    &enc,
                                    il,
                                    if use_fused_qkv {
                                        "attn-qkv-proj"
                                    } else {
                                        "attn-proj"
                                    },
                                );
                                if use_fused_qkv {
                                    let fused_qkv_pack = layer_scratch
                                        .attn_qkv_fused_pack
                                        .view_subrange(0, vec![(chunk_p * qkv_fused_dim) as u64]);
                                    encode_mat_mat_dispatch(
                                        base.ctx,
                                        &enc,
                                        a.qkv_fused.as_ref().expect("qkv_fused"),
                                        &h_pack_p,
                                        &fused_qkv_pack,
                                        h,
                                        qkv_fused_dim,
                                        chunk_p,
                                    )?;
                                    encode_split_qkv_fused_f32(
                                        base.ctx,
                                        &enc,
                                        &fused_qkv_pack,
                                        &q_full_pack_p,
                                        &k_now_pack_p,
                                        &v_now_pack_p,
                                        chunk_p,
                                        attn_q_full_dim,
                                        kv_dim,
                                    )?;
                                } else {
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
                                }
                                // v0.432: no split_q_gate — the q-norm below
                                // reads the interleaved Q halves directly and
                                // the attn epilogue reads the gate halves.
                                enc.end();
                            }
                            flush_prefill_phase(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                trace_attn_phases,
                                chunk_idx,
                                chunk_start,
                                il,
                                if use_fused_qkv { "qkv_matmul" } else { "proj" },
                            )?;
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                label_prefill_encoder(
                                    &enc,
                                    il,
                                    if fused_qk_norm_rope {
                                        "attn-norm-rope"
                                    } else {
                                        "attn-norm"
                                    },
                                );
                                if fused_qk_norm_rope {
                                    encode_qk_rms_norm_rope_f32_packed_consecutive(
                                        base.ctx,
                                        &enc,
                                        &q_full_pack_p,
                                        &a.q_norm,
                                        &q_normed_pack_p,
                                        &k_now_pack_p,
                                        &a.k_norm,
                                        &k_normed_pack_p,
                                        chunk_p,
                                        n_q,
                                        n_kv,
                                        head_dim,
                                        n_rot,
                                        chunk_start,
                                        RMS_EPS,
                                        arch.rope_theta,
                                    )?;
                                } else {
                                    encode_rms_norm_batched_src_strided_f32(
                                        base.ctx,
                                        &enc,
                                        &q_full_pack_p,
                                        &a.q_norm,
                                        &q_normed_pack_p,
                                        chunk_p * n_q,
                                        head_dim,
                                        2 * head_dim,
                                        0,
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
                                }
                                enc.end();
                            }
                            flush_prefill_phase(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                trace_attn_phases,
                                chunk_idx,
                                chunk_start,
                                il,
                                if fused_qk_norm_rope {
                                    "norm_rope"
                                } else {
                                    "norm"
                                },
                            )?;
                        } else {
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                label_prefill_encoder(
                                    &enc,
                                    il,
                                    if fused_qk_norm_rope {
                                        "attn-front-norm-rope"
                                    } else {
                                        "attn-front"
                                    },
                                );
                                if use_fused_qkv {
                                    let fused_qkv_pack = layer_scratch
                                        .attn_qkv_fused_pack
                                        .view_subrange(0, vec![(chunk_p * qkv_fused_dim) as u64]);
                                    encode_mat_mat_dispatch(
                                        base.ctx,
                                        &enc,
                                        a.qkv_fused.as_ref().expect("qkv_fused"),
                                        &h_pack_p,
                                        &fused_qkv_pack,
                                        h,
                                        qkv_fused_dim,
                                        chunk_p,
                                    )?;
                                    encode_split_qkv_fused_f32(
                                        base.ctx,
                                        &enc,
                                        &fused_qkv_pack,
                                        &q_full_pack_p,
                                        &k_now_pack_p,
                                        &v_now_pack_p,
                                        chunk_p,
                                        attn_q_full_dim,
                                        kv_dim,
                                    )?;
                                } else {
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
                                }
                                // v0.432: strided q-norm replaces split_q_gate
                                // + compact q-norm (gate halves are read by the
                                // strided sigmoid_mul at the attn epilogue).
                                if fused_qk_norm_rope {
                                    encode_qk_rms_norm_rope_f32_packed_consecutive(
                                        base.ctx,
                                        &enc,
                                        &q_full_pack_p,
                                        &a.q_norm,
                                        &q_normed_pack_p,
                                        &k_now_pack_p,
                                        &a.k_norm,
                                        &k_normed_pack_p,
                                        chunk_p,
                                        n_q,
                                        n_kv,
                                        head_dim,
                                        n_rot,
                                        chunk_start,
                                        RMS_EPS,
                                        arch.rope_theta,
                                    )?;
                                } else {
                                    encode_rms_norm_batched_src_strided_f32(
                                        base.ctx,
                                        &enc,
                                        &q_full_pack_p,
                                        &a.q_norm,
                                        &q_normed_pack_p,
                                        chunk_p * n_q,
                                        head_dim,
                                        2 * head_dim,
                                        0,
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
                                }
                                enc.end();
                            }
                        }

                        if prefill_noop_attn_body_enabled() {
                            apply_mixer_residual = false;
                            target_session.kv_n_pos[ai] = chunk_start as usize + chunk_p;
                        } else {
                            let use_packed_g8 = layer_scratch.scratch_plan.modes.enable_attn_packed
                                && prefill_attn_packed_g8_enabled(
                                    chunk_start as usize + chunk_p,
                                    n_q / n_kv,
                                )
                                && target_session.kv_k[ai].dtype == GgmlType::F16
                                && target_session.kv_v[ai].dtype == GgmlType::F16
                                && head_dim == 256
                                && n_q == 16
                                && n_kv == 2;
                            let use_packed_g16 =
                                layer_scratch.scratch_plan.modes.enable_attn_packed
                                    && prefill_attn_packed_g16_enabled(
                                        chunk_start as usize + chunk_p,
                                        n_q / n_kv,
                                    )
                                    && target_session.kv_k[ai].dtype == GgmlType::F16
                                    && target_session.kv_v[ai].dtype == GgmlType::F16
                                    && head_dim == 256
                                    && n_q == 32
                                    && n_kv == 2;
                            let matrix_scratch_covers_chunk = layer_scratch.attn_matrix_max_pos
                                >= chunk_start as u64 + chunk_p as u64;
                            let use_matrix_g8 = use_packed_g8
                                && layer_scratch.scratch_plan.modes.enable_attn_matrix
                                && prefill_attn_matrix_g8_may_use()
                                && matrix_scratch_covers_chunk;
                            let use_matrix_g4 = layer_scratch.scratch_plan.modes.enable_attn_matrix
                                && prefill_attn_matrix_g4_may_use()
                                && target_session.kv_k[ai].dtype == GgmlType::F16
                                && target_session.kv_v[ai].dtype == GgmlType::F16
                                && head_dim == 256
                                && n_q == n_kv * 4
                                && matrix_scratch_covers_chunk;
                            let use_matrix_g6 = layer_scratch.scratch_plan.modes.enable_attn_matrix
                                && prefill_attn_matrix_g6_may_use()
                                && target_session.kv_k[ai].dtype == GgmlType::F16
                                && target_session.kv_v[ai].dtype == GgmlType::F16
                                && head_dim == 256
                                && n_q == 24
                                && n_kv == 4
                                && matrix_scratch_covers_chunk;
                            let use_matrix_g16 = use_packed_g16
                                && layer_scratch.scratch_plan.modes.enable_attn_matrix
                                && prefill_attn_matrix_g16_may_use()
                                && matrix_scratch_covers_chunk;
                            let use_matrix =
                                use_matrix_g4 || use_matrix_g8 || use_matrix_g6 || use_matrix_g16;
                            let n_pos = chunk_start as usize + chunk_p;
                            let matrix_vt_prefix_rebuild_rows = attn_matrix_vt_prefix_rebuild_rows(
                                use_matrix,
                                attn_matrix_vt_valid_until[ai],
                                chunk_start as usize,
                            );
                            let rebuild_matrix_vt_prefix = matrix_vt_prefix_rebuild_rows.is_some();
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                label_prefill_encoder(
                                    &enc,
                                    il,
                                    if fused_qk_norm_rope {
                                        "attn-scatter"
                                    } else {
                                        "attn-rope-scatter"
                                    },
                                );
                                if !fused_qk_norm_rope {
                                    encode_prefill_qk_rope(
                                        base.ctx,
                                        &enc,
                                        &q_normed_pack_p,
                                        &k_normed_pack_p,
                                        chunk_p,
                                        n_q,
                                        n_kv,
                                        head_dim,
                                        n_rot,
                                        chunk_start,
                                        arch.rope_theta,
                                    )?;
                                }
                                if use_matrix {
                                    let vt_stride = layer_scratch.attn_matrix_max_pos as usize;
                                    let per_attn_vt = n_kv * head_dim * vt_stride;
                                    let v_t = layer_scratch.attn_matrix_vt_pack.view_subrange(
                                        (ai * per_attn_vt) as u64,
                                        vec![per_attn_vt as u64],
                                    );
                                    encode_scatter_offset_f32_to_f16_kv_vt(
                                        base.ctx,
                                        &enc,
                                        &k_normed_pack_p,
                                        &v_now_pack_p,
                                        &target_session.kv_k[ai],
                                        &target_session.kv_v[ai],
                                        &v_t,
                                        (chunk_start as usize) * kv_dim,
                                        chunk_p * kv_dim,
                                        chunk_start as usize,
                                        kv_dim,
                                        head_dim,
                                        vt_stride,
                                    )?;
                                } else {
                                    encode_scatter_offset_f32_to_f16_kv(
                                        base.ctx,
                                        &enc,
                                        &k_normed_pack_p,
                                        &v_now_pack_p,
                                        &target_session.kv_k[ai],
                                        &target_session.kv_v[ai],
                                        (chunk_start as usize) * kv_dim,
                                        chunk_p * kv_dim,
                                    )?;
                                }
                                enc.end();
                            }
                            flush_prefill_phase(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                trace_attn_phases,
                                chunk_idx,
                                chunk_start,
                                il,
                                if fused_qk_norm_rope {
                                    "scatter"
                                } else {
                                    "rope_scatter"
                                },
                            )?;

                            let mut traced_matrix_subphases = false;
                            if use_packed_g8 || use_packed_g16 || use_matrix {
                                let group = n_q / n_kv;
                                let use_packed = use_packed_g8 || use_packed_g16;
                                let packed_rows = if use_packed_g8 {
                                    prefill_attn_packed_g8_rows()
                                } else if use_packed_g16 {
                                    prefill_attn_packed_g16_rows()
                                } else {
                                    1
                                };
                                let packed_qt = if use_packed_g8 {
                                    prefill_attn_packed_g8_qt()
                                } else if use_packed_g16 {
                                    prefill_attn_packed_g16_qt()
                                } else {
                                    1
                                };
                                let attn_packed_oracle = if use_packed_g8 {
                                    prefill_attn_packed_g8_oracle_enabled()
                                } else if use_packed_g16 {
                                    prefill_attn_packed_g16_oracle_enabled()
                                } else {
                                    false
                                };
                                let nwg = if use_packed_g8 {
                                    prefill_attn_packed_g8_nwg()
                                } else if use_packed_g16 {
                                    prefill_attn_packed_g16_nwg()
                                } else {
                                    1
                                };
                                if trace_attn_phases && use_packed {
                                    let row_groups = chunk_p.div_ceil(packed_rows);
                                    let partial_bytes = chunk_p
                                        * n_kv
                                        * nwg
                                        * group
                                        * (head_dim + 2)
                                        * std::mem::size_of::<f32>();
                                    eprintln!(
                                        "[prefill-attn-packed-shape] layer={} chunk_start={} chunk_p={} group={} packed_rows={} qt={} row_groups={} dispatches={} nwg={} partial_rw_mib={:.2}",
                                        il,
                                        chunk_start,
                                        chunk_p,
                                        group,
                                        packed_rows,
                                        packed_qt,
                                        row_groups,
                                        row_groups * 2,
                                        nwg,
                                        (partial_bytes as f64 * 2.0) / (1024.0 * 1024.0),
                                    );
                                }
                                if trace_attn_phases && use_matrix {
                                    let vt_stride = layer_scratch.attn_matrix_max_pos as usize;
                                    let matrix_query_rows =
                                        chunk_p.min(layer_scratch.attn_matrix_query_rows as usize);
                                    // Online path stores F16 P~ plus the (m, l) sidecar;
                                    // the legacy sidecar stores F32 scores.
                                    let scores_bytes = if layer_scratch
                                        .scratch_plan
                                        .modes
                                        .attn_matrix_online
                                    {
                                        matrix_query_rows * n_q * n_pos * std::mem::size_of::<u16>()
                                            + crate::metal::attn_matrix_ml_elems(
                                                matrix_query_rows,
                                                n_q,
                                                n_pos,
                                            ) * std::mem::size_of::<f32>()
                                    } else {
                                        chunk_p * n_q * n_pos * std::mem::size_of::<f32>()
                                    };
                                    let vt_bytes =
                                        n_kv * head_dim * vt_stride * std::mem::size_of::<u16>();
                                    let (vt_base, vt_rows) = if rebuild_matrix_vt_prefix {
                                        (0, n_pos)
                                    } else {
                                        (chunk_start as usize, chunk_p)
                                    };
                                    let vt_update_bytes =
                                        n_kv * head_dim * vt_rows * std::mem::size_of::<u16>();
                                    let vt_rebuild_rows =
                                        matrix_vt_prefix_rebuild_rows.unwrap_or(0);
                                    let vt_rebuild_bytes = n_kv
                                        * head_dim
                                        * vt_rebuild_rows
                                        * std::mem::size_of::<u16>();
                                    eprintln!(
                                        concat!(
                                            "[prefill-attn-matrix-g{}-shape] layer={} ",
                                            "chunk_start={} chunk_p={} query_rows={} ",
                                            "query_tiles={} n_pos={} score_tile_scratch_mib={:.2} ",
                                            "vt_stride={} vt_layer_mib={:.2} vt_update_base={} ",
                                            "vt_update_rows={} vt_update_mib={:.2} ",
                                            "vt_rebuild_rows={} vt_rebuild_bytes={} ",
                                            "vt_rebuild_mib={:.2}"
                                        ),
                                        group,
                                        il,
                                        chunk_start,
                                        chunk_p,
                                        matrix_query_rows,
                                        chunk_p.div_ceil(matrix_query_rows),
                                        n_pos,
                                        scores_bytes as f64 / (1024.0 * 1024.0),
                                        vt_stride,
                                        vt_bytes as f64 / (1024.0 * 1024.0),
                                        vt_base,
                                        vt_rows,
                                        vt_update_bytes as f64 / (1024.0 * 1024.0),
                                        vt_rebuild_rows,
                                        vt_rebuild_bytes,
                                        vt_rebuild_bytes as f64 / (1024.0 * 1024.0),
                                    );
                                }
                                let matrix_query_tiled =
                                    layer_scratch.scratch_plan.modes.attn_matrix_online
                                        && (layer_scratch.attn_matrix_query_rows as usize)
                                            < chunk_p;
                                let trace_matrix_subphases = use_matrix
                                    && trace_attn_phases
                                    && !attn_packed_oracle
                                    && !matrix_query_tiled;
                                traced_matrix_subphases = trace_matrix_subphases;
                                if trace_matrix_subphases {
                                    let vt_stride = layer_scratch.attn_matrix_max_pos as usize;
                                    let per_attn_vt = n_kv * head_dim * vt_stride;
                                    let matrix_online =
                                        layer_scratch.scratch_plan.modes.attn_matrix_online;
                                    let v_t = layer_scratch.attn_matrix_vt_pack.view_subrange(
                                        (ai * per_attn_vt) as u64,
                                        vec![per_attn_vt as u64],
                                    );

                                    if let Some(prefix_rows) = matrix_vt_prefix_rebuild_rows {
                                        let enc = KernelEncoder::begin(&cmd_buf);
                                        label_prefill_encoder(
                                            &enc,
                                            il,
                                            if use_matrix_g4 {
                                                "attn-prefill-g4-matrix-vt"
                                            } else if use_matrix_g6 {
                                                "attn-prefill-g6-matrix-vt"
                                            } else if use_matrix_g16 {
                                                "attn-prefill-g16-matrix-vt"
                                            } else {
                                                "attn-prefill-g8-matrix-vt"
                                            },
                                        );
                                        crate::metal::encode_attn_matrix_transpose_v_f16(
                                            base.ctx,
                                            &enc,
                                            &target_session.kv_v[ai],
                                            &v_t,
                                            0,
                                            prefix_rows,
                                            n_pos,
                                            n_kv * head_dim,
                                            vt_stride,
                                            n_kv,
                                            head_dim,
                                        )?;
                                        enc.end();
                                        flush_prefill_phase(
                                            base.ctx,
                                            &mut cmd_buf,
                                            &mut prefill_gpu_total_ms,
                                            trace_attn_phases,
                                            chunk_idx,
                                            chunk_start,
                                            il,
                                            "body_matrix_vt_rebuild",
                                        )?;
                                    }

                                    {
                                        let enc = KernelEncoder::begin(&cmd_buf);
                                        label_prefill_encoder(
                                            &enc,
                                            il,
                                            if use_matrix_g4 {
                                                "attn-prefill-g4-matrix-kq"
                                            } else if use_matrix_g6 {
                                                "attn-prefill-g6-matrix-kq"
                                            } else if use_matrix_g16 {
                                                "attn-prefill-g16-matrix-kq"
                                            } else {
                                                "attn-prefill-g8-matrix-kq"
                                            },
                                        );
                                        if matrix_online {
                                            let scores_h = layer_scratch
                                                .attn_matrix_scores_h_pack
                                                .view_subrange(
                                                    0,
                                                    vec![(chunk_p * n_q * n_pos) as u64],
                                                );
                                            let ml =
                                                layer_scratch.attn_matrix_ml_pack.view_subrange(
                                                    0,
                                                    vec![crate::metal::attn_matrix_ml_elems(
                                                        chunk_p, n_q, n_pos,
                                                    )
                                                        as u64],
                                                );
                                            crate::metal::encode_attn_matrix_kq_online_f32(
                                                base.ctx,
                                                &enc,
                                                &q_normed_pack_p,
                                                &target_session.kv_k[ai],
                                                &scores_h,
                                                &ml,
                                                chunk_p,
                                                chunk_start as usize,
                                                n_pos,
                                                n_kv * head_dim,
                                                n_q,
                                                n_kv,
                                                group,
                                                head_dim,
                                                prefill_attn_matrix_causal_skip_enabled(),
                                            )?;
                                        } else {
                                            let scores = layer_scratch
                                                .attn_matrix_scores_pack
                                                .view_subrange(
                                                    0,
                                                    vec![(chunk_p * n_q * n_pos) as u64],
                                                );
                                            crate::metal::encode_attn_matrix_kq_f32(
                                                base.ctx,
                                                &enc,
                                                &q_normed_pack_p,
                                                &target_session.kv_k[ai],
                                                &scores,
                                                chunk_p,
                                                chunk_start as usize,
                                                n_pos,
                                                n_kv * head_dim,
                                                n_q,
                                                n_kv,
                                                group,
                                                head_dim,
                                                // v0.430: causal tile skip is group-generic in the
                                                // kernels (row_last/group + base_pos); enable for all
                                                // matrix groups, not only dense G6. Rollback:
                                                // QWEN_PREFILL_ATTN_MATRIX_CAUSAL_SKIP=0.
                                                prefill_attn_matrix_causal_skip_enabled(),
                                            )?;
                                        }
                                        enc.end();
                                    }
                                    flush_prefill_phase(
                                        base.ctx,
                                        &mut cmd_buf,
                                        &mut prefill_gpu_total_ms,
                                        trace_attn_phases,
                                        chunk_idx,
                                        chunk_start,
                                        il,
                                        "body_matrix_kq",
                                    )?;

                                    if !matrix_online {
                                        {
                                            let enc = KernelEncoder::begin(&cmd_buf);
                                            label_prefill_encoder(
                                                &enc,
                                                il,
                                                if use_matrix_g4 {
                                                    "attn-prefill-g4-matrix-softmax"
                                                } else if use_matrix_g6 {
                                                    "attn-prefill-g6-matrix-softmax"
                                                } else if use_matrix_g16 {
                                                    "attn-prefill-g16-matrix-softmax"
                                                } else {
                                                    "attn-prefill-g8-matrix-softmax"
                                                },
                                            );
                                            let scores = layer_scratch
                                                .attn_matrix_scores_pack
                                                .view_subrange(
                                                    0,
                                                    vec![(chunk_p * n_q * n_pos) as u64],
                                                );
                                            crate::metal::encode_attn_matrix_softmax_f32(
                                                base.ctx,
                                                &enc,
                                                &scores,
                                                chunk_p,
                                                chunk_start as usize,
                                                n_pos,
                                                n_q,
                                                n_kv,
                                                group,
                                                head_dim,
                                            )?;
                                            enc.end();
                                        }
                                        flush_prefill_phase(
                                            base.ctx,
                                            &mut cmd_buf,
                                            &mut prefill_gpu_total_ms,
                                            trace_attn_phases,
                                            chunk_idx,
                                            chunk_start,
                                            il,
                                            "body_matrix_softmax",
                                        )?;
                                    }

                                    {
                                        let enc = KernelEncoder::begin(&cmd_buf);
                                        label_prefill_encoder(
                                            &enc,
                                            il,
                                            if use_matrix_g4 {
                                                "attn-prefill-g4-matrix-kqv"
                                            } else if use_matrix_g6 {
                                                "attn-prefill-g6-matrix-kqv"
                                            } else if use_matrix_g16 {
                                                "attn-prefill-g16-matrix-kqv"
                                            } else {
                                                "attn-prefill-g8-matrix-kqv"
                                            },
                                        );
                                        if matrix_online {
                                            let scores_h = layer_scratch
                                                .attn_matrix_scores_h_pack
                                                .view_subrange(
                                                    0,
                                                    vec![(chunk_p * n_q * n_pos) as u64],
                                                );
                                            let ml =
                                                layer_scratch.attn_matrix_ml_pack.view_subrange(
                                                    0,
                                                    vec![crate::metal::attn_matrix_ml_elems(
                                                        chunk_p, n_q, n_pos,
                                                    )
                                                        as u64],
                                                );
                                            crate::metal::encode_attn_matrix_kqv_norm_f32(
                                                base.ctx,
                                                &enc,
                                                &scores_h,
                                                &ml,
                                                &v_t,
                                                &attn_o_pack_p,
                                                chunk_p,
                                                chunk_start as usize,
                                                n_pos,
                                                vt_stride,
                                                n_q,
                                                n_kv,
                                                group,
                                                head_dim,
                                                prefill_attn_matrix_causal_skip_enabled(),
                                            )?;
                                        } else {
                                            let scores = layer_scratch
                                                .attn_matrix_scores_pack
                                                .view_subrange(
                                                    0,
                                                    vec![(chunk_p * n_q * n_pos) as u64],
                                                );
                                            crate::metal::encode_attn_matrix_kqv_f32(
                                                base.ctx,
                                                &enc,
                                                &scores,
                                                &v_t,
                                                &attn_o_pack_p,
                                                chunk_p,
                                                chunk_start as usize,
                                                n_pos,
                                                vt_stride,
                                                n_q,
                                                n_kv,
                                                group,
                                                head_dim,
                                                // v0.430: causal tile skip is group-generic in the
                                                // kernels (row_last/group + base_pos); enable for all
                                                // matrix groups, not only dense G6. Rollback:
                                                // QWEN_PREFILL_ATTN_MATRIX_CAUSAL_SKIP=0.
                                                prefill_attn_matrix_causal_skip_enabled(),
                                            )?;
                                        }
                                        enc.end();
                                    }
                                    flush_prefill_phase(
                                        base.ctx,
                                        &mut cmd_buf,
                                        &mut prefill_gpu_total_ms,
                                        trace_attn_phases,
                                        chunk_idx,
                                        chunk_start,
                                        il,
                                        "body_matrix_kqv",
                                    )?;

                                    target_session.kv_n_pos[ai] = chunk_start as usize + chunk_p;
                                    attn_matrix_vt_valid_until[ai] = n_pos;
                                } else {
                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    label_prefill_encoder(
                                        &enc,
                                        il,
                                        if use_matrix_g4 {
                                            "attn-prefill-g4-matrix"
                                        } else if use_matrix_g6 {
                                            "attn-prefill-g6-matrix"
                                        } else if use_matrix_g16 {
                                            "attn-prefill-g16-matrix"
                                        } else if use_matrix_g8 {
                                            "attn-prefill-g8-matrix"
                                        } else if use_packed_g8 {
                                            "attn-prefill-g8-packed"
                                        } else {
                                            "attn-prefill-g16-packed"
                                        },
                                    );
                                    if use_matrix {
                                        let vt_stride = layer_scratch.attn_matrix_max_pos as usize;
                                        let per_attn_vt = n_kv * head_dim * vt_stride;
                                        let v_t = layer_scratch.attn_matrix_vt_pack.view_subrange(
                                            (ai * per_attn_vt) as u64,
                                            vec![per_attn_vt as u64],
                                        );
                                        if let Some(prefix_rows) = matrix_vt_prefix_rebuild_rows {
                                            crate::metal::encode_attn_matrix_transpose_v_f16(
                                                base.ctx,
                                                &enc,
                                                &target_session.kv_v[ai],
                                                &v_t,
                                                0,
                                                prefix_rows,
                                                n_pos,
                                                n_kv * head_dim,
                                                vt_stride,
                                                n_kv,
                                                head_dim,
                                            )?;
                                        }
                                        if layer_scratch.scratch_plan.modes.attn_matrix_online {
                                            // v0.439: two-pass online-softmax matrix attention.
                                            // Rollback: QWEN_PREFILL_ATTN_MATRIX_ONLINE=0.
                                            let query_rows =
                                                layer_scratch.attn_matrix_query_rows as usize;
                                            let query_tiles = chunk_p.div_ceil(query_rows);
                                            if query_tiles > 1 {
                                                layer_scratch.attn_matrix_tiled_layer_calls += 1;
                                                layer_scratch.attn_matrix_query_tile_calls +=
                                                    query_tiles as u64;
                                            }
                                            for (row_base, rows_n) in
                                                attn_matrix_query_tiles(chunk_p, query_rows)
                                            {
                                                let q_rows = q_normed_pack_p.view_subrange(
                                                    (row_base * q_dim) as u64,
                                                    vec![(rows_n * q_dim) as u64],
                                                );
                                                let attn_o_rows = attn_o_pack_p.view_subrange(
                                                    (row_base * q_dim) as u64,
                                                    vec![(rows_n * q_dim) as u64],
                                                );
                                                let scores_h = layer_scratch
                                                    .attn_matrix_scores_h_pack
                                                    .view_subrange(
                                                        0,
                                                        vec![(rows_n * n_q * n_pos) as u64],
                                                    );
                                                let ml = layer_scratch
                                                    .attn_matrix_ml_pack
                                                    .view_subrange(
                                                        0,
                                                        vec![crate::metal::attn_matrix_ml_elems(
                                                            rows_n, n_q, n_pos,
                                                        )
                                                            as u64],
                                                    );
                                                let base_pos = chunk_start as usize + row_base;
                                                crate::metal::encode_attn_matrix_kq_online_f32(
                                                    base.ctx,
                                                    &enc,
                                                    &q_rows,
                                                    &target_session.kv_k[ai],
                                                    &scores_h,
                                                    &ml,
                                                    rows_n,
                                                    base_pos,
                                                    n_pos,
                                                    n_kv * head_dim,
                                                    n_q,
                                                    n_kv,
                                                    group,
                                                    head_dim,
                                                    prefill_attn_matrix_causal_skip_enabled(),
                                                )?;
                                                crate::metal::encode_attn_matrix_kqv_norm_f32(
                                                    base.ctx,
                                                    &enc,
                                                    &scores_h,
                                                    &ml,
                                                    &v_t,
                                                    &attn_o_rows,
                                                    rows_n,
                                                    base_pos,
                                                    n_pos,
                                                    vt_stride,
                                                    n_q,
                                                    n_kv,
                                                    group,
                                                    head_dim,
                                                    prefill_attn_matrix_causal_skip_enabled(),
                                                )?;
                                            }
                                        } else {
                                            let scores = layer_scratch
                                                .attn_matrix_scores_pack
                                                .view_subrange(
                                                    0,
                                                    vec![(chunk_p * n_q * n_pos) as u64],
                                                );
                                            crate::metal::encode_attn_matrix_kq_f32(
                                                base.ctx,
                                                &enc,
                                                &q_normed_pack_p,
                                                &target_session.kv_k[ai],
                                                &scores,
                                                chunk_p,
                                                chunk_start as usize,
                                                n_pos,
                                                n_kv * head_dim,
                                                n_q,
                                                n_kv,
                                                group,
                                                head_dim,
                                                // v0.430: causal tile skip is group-generic in the
                                                // kernels (row_last/group + base_pos); enable for all
                                                // matrix groups, not only dense G6. Rollback:
                                                // QWEN_PREFILL_ATTN_MATRIX_CAUSAL_SKIP=0.
                                                prefill_attn_matrix_causal_skip_enabled(),
                                            )?;
                                            crate::metal::encode_attn_matrix_softmax_f32(
                                                base.ctx,
                                                &enc,
                                                &scores,
                                                chunk_p,
                                                chunk_start as usize,
                                                n_pos,
                                                n_q,
                                                n_kv,
                                                group,
                                                head_dim,
                                            )?;
                                            crate::metal::encode_attn_matrix_kqv_f32(
                                                base.ctx,
                                                &enc,
                                                &scores,
                                                &v_t,
                                                &attn_o_pack_p,
                                                chunk_p,
                                                chunk_start as usize,
                                                n_pos,
                                                vt_stride,
                                                n_q,
                                                n_kv,
                                                group,
                                                head_dim,
                                                // v0.430: causal tile skip is group-generic in the
                                                // kernels (row_last/group + base_pos); enable for all
                                                // matrix groups, not only dense G6. Rollback:
                                                // QWEN_PREFILL_ATTN_MATRIX_CAUSAL_SKIP=0.
                                                prefill_attn_matrix_causal_skip_enabled(),
                                            )?;
                                        }
                                    } else {
                                        for row_base in (0..chunk_p).step_by(packed_rows) {
                                            let rows_n = (chunk_p - row_base).min(packed_rows);
                                            let q_rows = q_normed_pack_p.view_subrange(
                                                (row_base * q_dim) as u64,
                                                vec![(rows_n * q_dim) as u64],
                                            );
                                            let attn_o_rows = attn_o_pack_p.view_subrange(
                                                (row_base * q_dim) as u64,
                                                vec![(rows_n * q_dim) as u64],
                                            );
                                            let o_partial_rows = layer_scratch
                                                .attn_prefill_v4_o_partial_pack
                                                .view_subrange(
                                                    0,
                                                    vec![
                                                        (rows_n
                                                            * n_kv
                                                            * ATTN_V4_MAX_NWG
                                                            * group
                                                            * head_dim)
                                                            as u64,
                                                    ],
                                                );
                                            let ml_partial_rows = layer_scratch
                                                .attn_prefill_v4_ml_partial_pack
                                                .view_subrange(
                                                    0,
                                                    vec![
                                                        (rows_n
                                                            * n_kv
                                                            * ATTN_V4_MAX_NWG
                                                            * group
                                                            * 2)
                                                            as u64,
                                                    ],
                                                );
                                            if use_packed_g8 {
                                                if packed_qt == 4 {
                                                    crate::metal::encode_attn_prefill_v4_g8_t2_q4_c64_f32(
                                                base.ctx,
                                                &enc,
                                                &q_rows,
                                                &target_session.kv_k[ai],
                                                &target_session.kv_v[ai],
                                                &o_partial_rows,
                                                &ml_partial_rows,
                                                &attn_o_rows,
                                                rows_n,
                                                chunk_start as usize + row_base,
                                                nwg,
                                            )?;
                                                } else {
                                                    crate::metal::encode_attn_prefill_v4_g8_t2_q2_c64_f32(
                                                base.ctx,
                                                &enc,
                                                &q_rows,
                                                &target_session.kv_k[ai],
                                                &target_session.kv_v[ai],
                                                &o_partial_rows,
                                                &ml_partial_rows,
                                                &attn_o_rows,
                                                rows_n,
                                                chunk_start as usize + row_base,
                                                nwg,
                                            )?;
                                                }
                                            } else {
                                                if packed_qt == 4 {
                                                    crate::metal::encode_attn_prefill_v4_g16_t4_q4_c64_f32(
                                                base.ctx,
                                                &enc,
                                                &q_rows,
                                                &target_session.kv_k[ai],
                                                &target_session.kv_v[ai],
                                                &o_partial_rows,
                                                &ml_partial_rows,
                                                &attn_o_rows,
                                                rows_n,
                                                chunk_start as usize + row_base,
                                                nwg,
                                            )?;
                                                } else {
                                                    crate::metal::encode_attn_prefill_v4_g16_t4_q2_c64_f32(
                                                base.ctx,
                                                &enc,
                                                &q_rows,
                                                &target_session.kv_k[ai],
                                                &target_session.kv_v[ai],
                                                &o_partial_rows,
                                                &ml_partial_rows,
                                                &attn_o_rows,
                                                rows_n,
                                                chunk_start as usize + row_base,
                                                nwg,
                                            )?;
                                                }
                                            }
                                        }
                                    }
                                    enc.end();
                                    if attn_packed_oracle {
                                        cmd_buf.commit();
                                        cmd_buf.waitUntilCompleted();
                                        require_prefill_command_completed(&cmd_buf)?;
                                        prefill_gpu_total_ms +=
                                            (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

                                        let attn_oracle_pack = MetalTensor::zeros_f32(
                                            base.ctx,
                                            vec![(chunk_p * q_dim) as u64],
                                        )?;
                                        let oracle_cmd = base
                                            .ctx
                                            .queue
                                            .commandBuffer()
                                            .expect("oracle command buffer");
                                        for n_idx in 0..chunk_p {
                                            let q_normed_n = q_normed_pack_p.view_subrange(
                                                (n_idx * q_dim) as u64,
                                                vec![q_dim as u64],
                                            );
                                            let attn_o_n = attn_oracle_pack.view_subrange(
                                                (n_idx * q_dim) as u64,
                                                vec![q_dim as u64],
                                            );
                                            let enc = KernelEncoder::begin(&oracle_cmd);
                                            let position_n = chunk_start + n_idx as u32;
                                            let nwg = crate::metal::attn_v4_choose_nwg(
                                                position_n as usize + 1,
                                                group,
                                            );
                                            let tile_c = crate::metal::attn_v4_choose_tile_c(
                                                position_n as usize + 1,
                                                group,
                                            );
                                            let group_tile =
                                                crate::metal::attn_v4_choose_group_tile_prefill(
                                                    position_n as usize + 1,
                                                    group,
                                                );
                                            crate::metal::with_attn_v4_group_tile_override(
                                                group_tile,
                                                || {
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
                                                        position_n as usize + 1,
                                                        nwg,
                                                        tile_c,
                                                    )
                                                },
                                            )?;
                                            enc.end();
                                        }
                                        oracle_cmd.commit();
                                        oracle_cmd.waitUntilCompleted();
                                        require_prefill_command_completed(&oracle_cmd)?;

                                        let packed = cpu_read_f32buf(&attn_o_pack_p);
                                        let oracle = cpu_read_f32buf(&attn_oracle_pack);
                                        let cos = cosine_f32_slices(&packed, &oracle);
                                        let packed_nonfinite =
                                            packed.iter().filter(|v| !v.is_finite()).count();
                                        let oracle_nonfinite =
                                            oracle.iter().filter(|v| !v.is_finite()).count();
                                        let mut max_abs = 0.0f32;
                                        let mut worst = 0usize;
                                        for i in 0..packed.len() {
                                            let d = (packed[i] - oracle[i]).abs();
                                            if d > max_abs {
                                                max_abs = d;
                                                worst = i;
                                            }
                                        }
                                        eprintln!(
                                            "[prefill-attn-packed-g{}-oracle] layer={} chunk_start={} chunk_p={} cos={:.6} max_abs={:.3e} worst_idx={} nonfinite={}/{}",
                                            group,
                                            il,
                                            chunk_start,
                                            chunk_p,
                                            cos,
                                            max_abs,
                                            worst,
                                            packed_nonfinite,
                                            oracle_nonfinite,
                                        );
                                        let cos_limit = if use_matrix { 0.9999 } else { 0.99999 };
                                        let max_abs_limit = if use_matrix { 2e-2 } else { 2e-3 };
                                        if packed_nonfinite != 0
                                            || oracle_nonfinite != 0
                                            || !cos.is_finite()
                                            || cos < cos_limit
                                            || max_abs > max_abs_limit
                                        {
                                            return Err(DFlashError::Metal(MetalError::BadShape {
                                                kernel: if use_packed_g8 {
                                                    "prefill_attn_packed_g8_oracle"
                                                } else {
                                                    "prefill_attn_packed_g16_oracle"
                                                },
                                                detail: format!(
                                                    "layer={il} chunk_start={chunk_start} chunk_p={chunk_p} cos={cos:.6} max_abs={max_abs:.3e} nonfinite={packed_nonfinite}/{oracle_nonfinite}"
                                                ),
                                            }));
                                        }
                                        cmd_buf =
                                            base.ctx.queue.commandBuffer().expect("command buffer");
                                    }
                                    target_session.kv_n_pos[ai] = chunk_start as usize + chunk_p;
                                    if use_matrix {
                                        attn_matrix_vt_valid_until[ai] = n_pos;
                                    }
                                }
                            } else {
                                for n_idx in 0..chunk_p {
                                    let position_n = chunk_start + n_idx as u32;
                                    let q_normed_n = q_normed_pack_p
                                        .view_subrange((n_idx * q_dim) as u64, vec![q_dim as u64]);
                                    let attn_o_n = attn_o_pack_p
                                        .view_subrange((n_idx * q_dim) as u64, vec![q_dim as u64]);
                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    target_session.kv_n_pos[ai] = position_n as usize + 1;

                                    const V4_HEAD_DIM: usize = 256;
                                    let group = n_q / n_kv;
                                    let use_v4 =
                                        head_dim == V4_HEAD_DIM && matches!(group, 4 | 6 | 8 | 16);
                                    if use_v4 {
                                        let nwg = crate::metal::attn_v4_choose_nwg(
                                            target_session.kv_n_pos[ai],
                                            group,
                                        );
                                        let tile_c = crate::metal::attn_v4_choose_tile_c(
                                            target_session.kv_n_pos[ai],
                                            group,
                                        );
                                        let group_tile =
                                            crate::metal::attn_v4_choose_group_tile_prefill(
                                                target_session.kv_n_pos[ai],
                                                group,
                                            );
                                        crate::metal::with_attn_v4_group_tile_override(
                                            group_tile,
                                            || {
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
                                                )
                                            },
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
                            if !traced_matrix_subphases {
                                flush_prefill_phase(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                    trace_attn_phases,
                                    chunk_idx,
                                    chunk_start,
                                    il,
                                    "body",
                                )?;
                            }

                            if let Some(capture) = attention_capture.as_deref_mut()
                                && let Some(block_index) =
                                    capture.blocks.iter().position(|&block| block == il)
                            {
                                let capture_packed =
                                    (use_packed_g8 || use_packed_g16) && !use_matrix;
                                let capture_packed_rows = if use_packed_g8 {
                                    prefill_attn_packed_g8_rows()
                                } else if use_packed_g16 {
                                    prefill_attn_packed_g16_rows()
                                } else {
                                    0
                                };
                                let capture_packed_qt = if use_packed_g8 {
                                    prefill_attn_packed_g8_qt()
                                } else if use_packed_g16 {
                                    prefill_attn_packed_g16_qt()
                                } else {
                                    0
                                };
                                let capture_nwg = if use_packed_g8 {
                                    prefill_attn_packed_g8_nwg()
                                } else if use_packed_g16 {
                                    prefill_attn_packed_g16_nwg()
                                } else {
                                    0
                                };
                                let capture_group = n_q / n_kv;
                                let rows: Vec<(usize, usize)> = capture
                                    .positions
                                    .iter()
                                    .copied()
                                    .enumerate()
                                    .filter_map(|(capture_row, position)| {
                                        let in_chunk = position >= chunk_start as usize
                                            && position < chunk_start as usize + chunk_p;
                                        in_chunk.then_some((capture_row, position))
                                    })
                                    .collect();
                                if !rows.is_empty() {
                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    for (capture_row, position) in rows {
                                        let seen_index =
                                            block_index * capture.positions.len() + capture_row;
                                        if capture.seen[seen_index] {
                                            return Err(DFlashError::Metal(MetalError::BadShape {
                                                kernel: "attention_capture",
                                                detail: format!(
                                                    "duplicate block={il} \
                                                             position={position}"
                                                ),
                                            }));
                                        }
                                        let source_row = position - chunk_start as usize;
                                        let q_dst = capture.q[block_index].view_subrange(
                                            (capture_row * capture.q_dim) as u64,
                                            vec![capture.q_dim as u64],
                                        );
                                        let o_dst = capture.o[block_index].view_subrange(
                                            (capture_row * capture.q_dim) as u64,
                                            vec![capture.q_dim as u64],
                                        );
                                        encode_copy_offset_f32(
                                            base.ctx,
                                            &enc,
                                            &q_normed_pack_p,
                                            source_row * capture.q_dim,
                                            &q_dst,
                                            capture.q_dim,
                                        )?;
                                        encode_copy_offset_f32(
                                            base.ctx,
                                            &enc,
                                            &attn_o_pack_p,
                                            source_row * capture.q_dim,
                                            &o_dst,
                                            capture.q_dim,
                                        )?;
                                        let causal_length = position + 1;
                                        let capture_fallback = !use_matrix && !capture_packed;
                                        let fallback_nwg = capture_fallback.then(|| {
                                            crate::metal::attn_v4_choose_nwg(
                                                causal_length,
                                                capture_group,
                                            )
                                        });
                                        let fallback_tile_c = capture_fallback.then(|| {
                                            crate::metal::attn_v4_choose_tile_c(
                                                causal_length,
                                                capture_group,
                                            )
                                        });
                                        let fallback_group_tile = capture_fallback.then(|| {
                                            crate::metal::attn_v4_choose_group_tile_prefill(
                                                causal_length,
                                                capture_group,
                                            )
                                        });
                                        capture.provenance.push(AttentionCaptureProvenance {
                                            position,
                                            causal_length,
                                            block: il,
                                            kv_slot: ai,
                                            path: if use_matrix {
                                                "matrix"
                                            } else if capture_packed {
                                                "packed"
                                            } else {
                                                "decode_fallback"
                                            },
                                            online_matrix: use_matrix
                                                && layer_scratch
                                                    .scratch_plan
                                                    .modes
                                                    .attn_matrix_online,
                                            query_tiled: use_matrix
                                                && layer_scratch.attn_matrix_query_rows
                                                    < chunk_p as u32,
                                            query_rows: use_matrix.then_some(
                                                layer_scratch.attn_matrix_query_rows as usize,
                                            ),
                                            packed_rows: capture_packed
                                                .then_some(capture_packed_rows),
                                            packed_qt: capture_packed.then_some(capture_packed_qt),
                                            nwg: capture_packed
                                                .then_some(capture_nwg)
                                                .or(fallback_nwg),
                                            tile_c: fallback_tile_c,
                                            group_tile: fallback_group_tile,
                                            matrix_causal_skip: use_matrix
                                                && prefill_attn_matrix_causal_skip_enabled(),
                                        });
                                        capture.seen[seen_index] = true;
                                    }
                                    enc.end();
                                }
                            }

                            let enc = KernelEncoder::begin(&cmd_buf);
                            // v0.432: fused strided gate epilogue — reads the
                            // gate halves of the interleaved q_proj output in
                            // place (no split_q_gate) and fuses sigmoid+mul
                            // (was: sigmoid into q_pack temp, then mul).
                            crate::metal::encode_sigmoid_mul_gate_strided_f32(
                                base.ctx,
                                &enc,
                                &q_full_pack_p,
                                &attn_o_pack_p,
                                &attn_o_pack_p,
                                chunk_p * n_q,
                                head_dim,
                                2 * head_dim,
                                head_dim,
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
                            flush_prefill_layer_phase(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                trace_layer_phases,
                                chunk_idx,
                                chunk_start,
                                il,
                                "attn",
                                if trace_attn_phases {
                                    "attn_back"
                                } else {
                                    "attn"
                                },
                            )?;
                        }
                    } else {
                        // F32 oracle / fallback: per-token encode_attn.
                        if prefill_noop_attn_body_enabled() {
                            apply_mixer_residual = false;
                            target_session.kv_n_pos[ai] = chunk_start as usize + chunk_p;
                        } else {
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
                            flush_prefill_layer_phase(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                trace_layer_phases,
                                chunk_idx,
                                chunk_start,
                                il,
                                "attn",
                                "attn_fallback",
                            )?;
                        }
                    }
                }
            }

            // 2c: residual #1 — x_pack += mixer_out_pack (batched).
            if apply_mixer_residual {
                let enc = KernelEncoder::begin(&cmd_buf);
                encode_add_inplace_f32(base.ctx, &enc, &x_pack_p, &mixer_out_pack_p)?;
                enc.end();
                flush_prefill_layer_phase(
                    base.ctx,
                    &mut cmd_buf,
                    &mut prefill_gpu_total_ms,
                    trace_layer_phases,
                    chunk_idx,
                    chunk_start,
                    il,
                    block_kind,
                    "mixer_resid",
                )?;
            }

            // 2e/f/g: post-norm + FFN/MoE tail + residual #2 + hidden_capture.
            let post_norm = match block {
                MetalBlock::Gdn(g) => &g.post_attn_norm,
                MetalBlock::Attn(a) => &a.post_attn_norm,
            };
            let (g_w, u_w, d_w, moe) = match block {
                MetalBlock::Gdn(g) => (&g.ffn_gate, &g.ffn_up, &g.ffn_down, g.ffn_moe.as_ref()),
                MetalBlock::Attn(a) => (&a.ffn_gate, &a.ffn_up, &a.ffn_down, a.ffn_moe.as_ref()),
            };
            let skip_ffn = prefill_noop_ffn_enabled();

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

            if let Some(moe) = moe {
                {
                    let enc = KernelEncoder::begin(&cmd_buf);
                    encode_rms_norm_batched_f32(
                        base.ctx, &enc, &x_pack_p, post_norm, &h_pack_p, chunk_p, h, RMS_EPS,
                    )?;
                    enc.end();
                }
                flush_prefill_layer_phase(
                    base.ctx,
                    &mut cmd_buf,
                    &mut prefill_gpu_total_ms,
                    trace_layer_phases,
                    chunk_idx,
                    chunk_start,
                    il,
                    "moe",
                    "post_norm",
                )?;

                let router_mat_mat_eligible = |dtype: GgmlType| {
                    matches!(
                        dtype,
                        GgmlType::F32
                            | GgmlType::F16
                            | GgmlType::BF16
                            | GgmlType::Q4_K
                            | GgmlType::Q5_K
                            | GgmlType::Q6_K
                            | GgmlType::Q8_0
                    )
                };
                let packed_routed_path = prefill_moe_packed_routed_enabled()
                    && moe.gate_exps.dtype == GgmlType::Q4_K
                    && moe.up_exps.dtype == GgmlType::Q4_K
                    && moe.down_exps.dtype == GgmlType::Q5_K;
                let grouped_down_dtype_eligible = matches!(
                    moe.down_exps.dtype,
                    GgmlType::Q5_K
                        | GgmlType::Q6_K
                        | GgmlType::Q8_0
                        | GgmlType::IQ4_XS
                        | GgmlType::BF16
                );
                let topk = arch.expert_used_count.min(arch.expert_count) as usize;
                let n_expert = arch.expert_count as usize;
                let f_exp = arch.expert_feed_forward_length as usize;
                let f_shared = arch.expert_shared_feed_forward_length as usize;
                let grouped_gate_up_dtype_eligible = match (moe.gate_exps.dtype, moe.up_exps.dtype)
                {
                    (GgmlType::Q4_K, GgmlType::Q4_K) => true,
                    (GgmlType::Q5_K, GgmlType::Q5_K) => {
                        prefill_moe_grouped_q5_gateup_enabled(h, f_exp, n_expert)
                    }
                    (GgmlType::Q6_K, GgmlType::Q6_K) => {
                        prefill_moe_grouped_q6_gateup_enabled(h, f_exp, n_expert)
                    }
                    (GgmlType::Q8_0, GgmlType::Q8_0) => {
                        prefill_moe_grouped_q8_gateup_enabled(h, f_exp, n_expert)
                    }
                    (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS) | (GgmlType::IQ3_S, GgmlType::IQ3_S) => {
                        chunk_p >= 32 && prefill_moe_grouped_iq3_gateup_enabled(h, f_exp, n_expert)
                    }
                    (GgmlType::F32, GgmlType::F32) => {
                        chunk_p >= 32 && prefill_moe_grouped_f32_gateup_enabled()
                    }
                    (GgmlType::BF16, GgmlType::BF16) => {
                        chunk_p >= 32 && prefill_moe_grouped_bf16_gateup_enabled(h, f_exp, n_expert)
                    }
                    _ => false,
                };
                let grouped_routed_path = prefill_moe_grouped_enabled()
                    && grouped_gate_up_dtype_eligible
                    && grouped_down_dtype_eligible
                    && h.is_multiple_of(256)
                    && f_exp.is_multiple_of(256);

                if !skip_ffn && (packed_routed_path || grouped_routed_path) {
                    let skip_moe_routed = prefill_noop_moe_routed_enabled();
                    let skip_moe_shared = prefill_noop_moe_shared_enabled();
                    let noop_grouped_swiglu = prefill_noop_moe_grouped_swiglu_enabled();
                    let noop_grouped_down = prefill_noop_moe_grouped_down_enabled();
                    let noop_grouped_reduce = prefill_noop_moe_grouped_reduce_enabled();
                    let hot_expert_min_slots = prefill_moe_hot_expert_min_slots();
                    let packed_route_path = prefill_moe_packed_route_enabled()
                        && router_mat_mat_eligible(moe.gate_inp.dtype)
                        && moe.gate_inp_shexp.dtype == GgmlType::F32
                        && n_expert <= 256
                        && (1..=16).contains(&topk);
                    let packed_shared_path = prefill_moe_packed_shared_enabled()
                        && f_shared > 0
                        && router_mat_mat_eligible(g_w.dtype)
                        && router_mat_mat_eligible(u_w.dtype)
                        && router_mat_mat_eligible(d_w.dtype);
                    let fused_grouped_finalizer = grouped_routed_path
                        && prefill_moe_fused_finalizer_enabled()
                        && (skip_moe_shared || packed_shared_path);
                    let fused_route_bucket = grouped_routed_path
                        && packed_route_path
                        && prefill_moe_route_bucket_fused_auto_enabled(h, chunk_p);
                    let moe_topk_idx_pack_p = layer_scratch
                        .moe_topk_idx_pack
                        .view_subrange(0, vec![(chunk_p * topk) as u64]);
                    let moe_router_probs_pack_p = layer_scratch
                        .moe_router_probs_pack
                        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
                    let moe_topk_weight_pack_p = layer_scratch
                        .moe_topk_weight_pack
                        .view_subrange(0, vec![(chunk_p * topk) as u64]);
                    let moe_shared_gate_pack_p = layer_scratch
                        .moe_shared_gate_pack
                        .view_subrange(0, vec![chunk_p as u64]);
                    // v0.431: moe_inner_pack / moe_expert_out_pack views are
                    // built lazily inside the packed-slot fallback branches
                    // below (production prefill stubs those packs; the
                    // grouped default never reads them).
                    let moe_mixer_out_pack_p = layer_scratch
                        .mixer_out_pack
                        .view_subrange(0, vec![(chunk_p * h) as u64]);
                    let moe_group_slot_idx_pack_p = layer_scratch
                        .moe_group_slot_idx_pack
                        .view_subrange(0, vec![(chunk_p * topk) as u64]);
                    let moe_group_count_pack = layer_scratch
                        .moe_group_count_pack
                        .view_subrange(0, vec![n_expert as u64]);
                    let moe_group_ids_pack = layer_scratch
                        .moe_group_ids_pack
                        .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
                    let moe_group_token_idx_pack_p = layer_scratch
                        .moe_group_token_idx_pack
                        .view_subrange(0, vec![(chunk_p * topk) as u64]);
                    let moe_group_weight_pack_p = layer_scratch
                        .moe_group_weight_pack
                        .view_subrange(0, vec![(chunk_p * topk) as u64]);
                    let moe_group_inner_pack_p = layer_scratch
                        .moe_group_inner_pack
                        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
                    let moe_group_out_pack_p = layer_scratch
                        .moe_group_out_pack
                        .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
                    let moe_shared_ffn_gate_pack_p = layer_scratch
                        .moe_shared_ffn_gate_pack
                        .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
                    let moe_shared_ffn_up_pack_p = layer_scratch
                        .moe_shared_ffn_up_pack
                        .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
                    let moe_shared_ffn_inner_pack_p = layer_scratch
                        .moe_shared_ffn_inner_pack
                        .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
                    let moe_shared_ffn_out_pack_p = layer_scratch
                        .moe_shared_ffn_out_pack
                        .view_subrange(0, vec![(chunk_p * h) as u64]);

                    if fused_route_bucket {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        label_prefill_encoder(&enc, il, "moe-route-fused");
                        encode_moe_route_logits_dispatch(
                            base.ctx,
                            &enc,
                            &moe.gate_inp,
                            &h_pack_p,
                            &moe_router_probs_pack_p,
                            h,
                            n_expert,
                            chunk_p,
                        )?;
                        encode_fill_f32(base.ctx, &enc, &moe_group_count_pack, 0.0)?;
                        crate::metal::encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
                            base.ctx,
                            &enc,
                            &moe_router_probs_pack_p,
                            &moe.gate_inp_shexp,
                            &h_pack_p,
                            &moe_topk_idx_pack_p,
                            &moe_topk_weight_pack_p,
                            &moe_shared_gate_pack_p,
                            &moe_group_count_pack,
                            &moe_group_ids_pack,
                            n_expert,
                            topk,
                            h,
                            chunk_p,
                        )?;
                        enc.end();
                    } else if packed_route_path {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        label_prefill_encoder(&enc, il, "moe-route-packed");
                        encode_moe_route_logits_dispatch(
                            base.ctx,
                            &enc,
                            &moe.gate_inp,
                            &h_pack_p,
                            &moe_router_probs_pack_p,
                            h,
                            n_expert,
                            chunk_p,
                        )?;
                        encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                            base.ctx,
                            &enc,
                            &moe_router_probs_pack_p,
                            &moe.gate_inp_shexp,
                            &h_pack_p,
                            &moe_topk_idx_pack_p,
                            &moe_topk_weight_pack_p,
                            &moe_shared_gate_pack_p,
                            n_expert,
                            topk,
                            h,
                            chunk_p,
                        )?;
                        enc.end();
                    } else {
                        for n_idx in 0..chunk_p {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            label_prefill_encoder(&enc, il, "moe-route-token-loop");
                            encode_copy_offset_f32(
                                base.ctx,
                                &enc,
                                &h_pack_p,
                                n_idx * h,
                                &target_session.h,
                                h,
                            )?;
                            base.encode_moe_route_prepare(&enc, target_session, moe)?;
                            encode_scatter_offset_f32(
                                base.ctx,
                                &enc,
                                &target_session.moe_topk_idx,
                                &moe_topk_idx_pack_p,
                                n_idx * topk,
                                topk,
                            )?;
                            encode_scatter_offset_f32(
                                base.ctx,
                                &enc,
                                &target_session.moe_topk_weight,
                                &moe_topk_weight_pack_p,
                                n_idx * topk,
                                topk,
                            )?;
                            encode_scatter_offset_f32(
                                base.ctx,
                                &enc,
                                &target_session.moe_shared_gate,
                                &moe_shared_gate_pack_p,
                                n_idx,
                                1,
                            )?;
                            enc.end();
                        }
                    }

                    flush_prefill_layer_phase(
                        base.ctx,
                        &mut cmd_buf,
                        &mut prefill_gpu_total_ms,
                        trace_layer_phases,
                        chunk_idx,
                        chunk_start,
                        il,
                        "moe",
                        if fused_route_bucket {
                            "route_fused"
                        } else if packed_route_path {
                            "route_packed"
                        } else {
                            "route_token_loop"
                        },
                    )?;
                    if grouped_routed_path && fused_route_bucket {
                        trace_prefill_moe_bucket_stats(
                            trace_moe_buckets,
                            chunk_idx,
                            chunk_start,
                            il,
                            &moe_group_count_pack,
                            n_expert,
                            chunk_p,
                            topk,
                            hot_expert_min_slots,
                        );
                    }

                    let concurrent_grouped_shared = grouped_routed_path
                        && packed_shared_path
                        && !skip_moe_routed
                        && !skip_moe_shared
                        && prefill_moe_grouped_concurrent_tail_enabled(chunk_p);

                    if concurrent_grouped_shared {
                        let zero_grouped_buffers = prefill_moe_grouped_zero_fill_enabled();
                        let grouped_q4_n32_all =
                            prefill_moe_grouped_q4_n32_all_enabled(arch, chunk_p);
                        let enc = KernelEncoder::begin_concurrent(&cmd_buf);
                        label_prefill_encoder(&enc, il, "moe-tail-concurrent");
                        if !fused_route_bucket {
                            crate::metal::encode_moe_route_bucket_slots_f32(
                                base.ctx,
                                &enc,
                                &moe_topk_idx_pack_p,
                                &moe_group_count_pack,
                                &moe_group_ids_pack,
                                n_expert,
                                chunk_p,
                                topk,
                            )?;
                        }
                        if zero_grouped_buffers {
                            encode_fill_f32(base.ctx, &enc, &moe_group_inner_pack_p, 0.0)?;
                        }
                        if noop_grouped_swiglu {
                            encode_fill_f32(base.ctx, &enc, &moe_group_inner_pack_p, 0.0)?;
                        } else {
                            encode_prefill_moe_grouped_swiglu(
                                base.ctx,
                                &enc,
                                moe,
                                &h_pack_p,
                                &moe_group_count_pack,
                                &moe_group_ids_pack,
                                &moe_group_inner_pack_p,
                                h,
                                f_exp,
                                n_expert,
                                topk,
                                chunk_p,
                                grouped_q4_n32_all,
                                hot_expert_min_slots,
                            )?;
                        }
                        if zero_grouped_buffers {
                            encode_fill_f32(base.ctx, &enc, &moe_group_out_pack_p, 0.0)?;
                        }
                        if noop_grouped_down {
                            encode_fill_f32(base.ctx, &enc, &moe_group_out_pack_p, 0.0)?;
                        } else {
                            encode_prefill_moe_grouped_down(
                                base.ctx,
                                &enc,
                                &moe.down_exps,
                                &moe_group_inner_pack_p,
                                &moe_group_count_pack,
                                &moe_group_ids_pack,
                                &moe_group_out_pack_p,
                                f_exp,
                                h,
                                n_expert,
                                chunk_p,
                            )?;
                        }
                        if noop_grouped_down || noop_grouped_reduce {
                            encode_fill_f32(base.ctx, &enc, &moe_mixer_out_pack_p, 0.0)?;
                        } else {
                            crate::metal::encode_moe_weighted_sum_packed_f32(
                                base.ctx,
                                &enc,
                                &moe_group_out_pack_p,
                                &moe_topk_weight_pack_p,
                                &moe_mixer_out_pack_p,
                                h,
                                topk,
                                chunk_p,
                            )?;
                        }
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            g_w,
                            &h_pack_p,
                            &moe_shared_ffn_gate_pack_p,
                            h,
                            f_shared,
                            chunk_p,
                        )?;
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            u_w,
                            &h_pack_p,
                            &moe_shared_ffn_up_pack_p,
                            h,
                            f_shared,
                            chunk_p,
                        )?;
                        encode_silu_mul_f32(
                            base.ctx,
                            &enc,
                            &moe_shared_ffn_gate_pack_p,
                            &moe_shared_ffn_up_pack_p,
                            &moe_shared_ffn_inner_pack_p,
                        )?;
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            d_w,
                            &moe_shared_ffn_inner_pack_p,
                            &moe_shared_ffn_out_pack_p,
                            f_shared,
                            h,
                            chunk_p,
                        )?;
                        enc.end();
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "tail_concurrent",
                        )?;

                        let enc = KernelEncoder::begin(&cmd_buf);
                        label_prefill_encoder(&enc, il, "moe-tail-concurrent-final");
                        encode_axpy_rowwise_f32(
                            base.ctx,
                            &enc,
                            &moe_shared_ffn_out_pack_p,
                            &moe_shared_gate_pack_p,
                            &moe_mixer_out_pack_p,
                            h,
                            chunk_p,
                        )?;
                        encode_add_inplace_f32(base.ctx, &enc, &x_pack_p, &moe_mixer_out_pack_p)?;
                        enc.end();
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "tail_concurrent_final",
                        )?;
                    } else if skip_moe_routed {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        label_prefill_encoder(&enc, il, "moe-routed-skip");
                        encode_fill_f32(base.ctx, &enc, &moe_mixer_out_pack_p, 0.0)?;
                        enc.end();
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "routed_skip",
                        )?;
                    } else if grouped_routed_path {
                        let zero_grouped_buffers = prefill_moe_grouped_zero_fill_enabled();
                        let grouped_q4_n32_all =
                            prefill_moe_grouped_q4_n32_all_enabled(arch, chunk_p);
                        if trace_layer_phases {
                            if !fused_route_bucket {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                crate::metal::encode_moe_route_bucket_slots_f32(
                                    base.ctx,
                                    &enc,
                                    &moe_topk_idx_pack_p,
                                    &moe_group_count_pack,
                                    &moe_group_ids_pack,
                                    n_expert,
                                    chunk_p,
                                    topk,
                                )?;
                                enc.end();
                                flush_prefill_layer_phase(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                    trace_layer_phases,
                                    chunk_idx,
                                    chunk_start,
                                    il,
                                    "moe",
                                    "route_bucket",
                                )?;
                                trace_prefill_moe_bucket_stats(
                                    trace_moe_buckets,
                                    chunk_idx,
                                    chunk_start,
                                    il,
                                    &moe_group_count_pack,
                                    n_expert,
                                    chunk_p,
                                    topk,
                                    hot_expert_min_slots,
                                );
                            }
                            {
                                let q4_hot_swiglu_bins = matches!(
                                    (moe.gate_exps.dtype, moe.up_exps.dtype),
                                    (GgmlType::Q4_K, GgmlType::Q4_K)
                                ) && !grouped_q4_n32_all
                                    && prefill_moe_grouped_hot_q4_n32_enabled(chunk_p)
                                    && hot_expert_min_slots == Some(48);
                                let split_swiglu_bins = prefill_trace_moe_bucket_bins_enabled()
                                    && match (moe.gate_exps.dtype, moe.up_exps.dtype) {
                                        (GgmlType::Q4_K, GgmlType::Q4_K) => q4_hot_swiglu_bins,
                                        (GgmlType::Q5_K, GgmlType::Q5_K)
                                        | (GgmlType::Q6_K, GgmlType::Q6_K)
                                        | (GgmlType::Q8_0, GgmlType::Q8_0)
                                        | (GgmlType::BF16, GgmlType::BF16)
                                        | (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS)
                                        | (GgmlType::IQ3_S, GgmlType::IQ3_S) => true,
                                        _ => false,
                                    };
                                if split_swiglu_bins {
                                    let bins: [(&str, u32, u32); 6] = [
                                        ("routed_swiglu_lt8", 0, 7),
                                        ("routed_swiglu_8_15", 8, 15),
                                        ("routed_swiglu_16_31", 16, 31),
                                        ("routed_swiglu_32_47", 32, 47),
                                        ("routed_swiglu_48_63", 48, 63),
                                        ("routed_swiglu_ge64", 64, i32::MAX as u32),
                                    ];
                                    for (bin_idx, (phase, min_slots, max_slots)) in
                                        bins.into_iter().enumerate()
                                    {
                                        let enc = KernelEncoder::begin(&cmd_buf);
                                        label_prefill_encoder(&enc, il, phase);
                                        if zero_grouped_buffers && bin_idx == 0 {
                                            encode_fill_f32(
                                                base.ctx,
                                                &enc,
                                                &moe_group_inner_pack_p,
                                                0.0,
                                            )?;
                                        }
                                        encode_prefill_moe_grouped_swiglu_range(
                                            base.ctx,
                                            &enc,
                                            moe,
                                            &h_pack_p,
                                            &moe_group_count_pack,
                                            &moe_group_ids_pack,
                                            &moe_group_inner_pack_p,
                                            h,
                                            f_exp,
                                            n_expert,
                                            topk,
                                            chunk_p,
                                            min_slots,
                                            max_slots,
                                            q4_hot_swiglu_bins && min_slots >= 48,
                                        )?;
                                        enc.end();
                                        flush_prefill_layer_phase(
                                            base.ctx,
                                            &mut cmd_buf,
                                            &mut prefill_gpu_total_ms,
                                            trace_layer_phases,
                                            chunk_idx,
                                            chunk_start,
                                            il,
                                            "moe",
                                            phase,
                                        )?;
                                    }
                                } else {
                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    label_prefill_encoder(&enc, il, "moe-routed-grouped-swiglu");
                                    if zero_grouped_buffers {
                                        encode_fill_f32(
                                            base.ctx,
                                            &enc,
                                            &moe_group_inner_pack_p,
                                            0.0,
                                        )?;
                                    }
                                    if noop_grouped_swiglu {
                                        encode_fill_f32(
                                            base.ctx,
                                            &enc,
                                            &moe_group_inner_pack_p,
                                            0.0,
                                        )?;
                                    } else {
                                        encode_prefill_moe_grouped_swiglu(
                                            base.ctx,
                                            &enc,
                                            moe,
                                            &h_pack_p,
                                            &moe_group_count_pack,
                                            &moe_group_ids_pack,
                                            &moe_group_inner_pack_p,
                                            h,
                                            f_exp,
                                            n_expert,
                                            topk,
                                            chunk_p,
                                            grouped_q4_n32_all,
                                            hot_expert_min_slots,
                                        )?;
                                    }
                                    enc.end();
                                    flush_prefill_layer_phase(
                                        base.ctx,
                                        &mut cmd_buf,
                                        &mut prefill_gpu_total_ms,
                                        trace_layer_phases,
                                        chunk_idx,
                                        chunk_start,
                                        il,
                                        "moe",
                                        "routed_swiglu",
                                    )?;
                                }
                            }
                            {
                                let split_down_bins = prefill_trace_moe_bucket_bins_enabled()
                                    && matches!(
                                        moe.down_exps.dtype,
                                        GgmlType::Q5_K | GgmlType::BF16
                                    );
                                if split_down_bins {
                                    let bins: [(&str, u32, u32); 6] = [
                                        ("routed_down_lt8", 0, 7),
                                        ("routed_down_8_15", 8, 15),
                                        ("routed_down_16_31", 16, 31),
                                        ("routed_down_32_47", 32, 47),
                                        ("routed_down_48_63", 48, 63),
                                        ("routed_down_ge64", 64, i32::MAX as u32),
                                    ];
                                    for (bin_idx, (phase, min_slots, max_slots)) in
                                        bins.into_iter().enumerate()
                                    {
                                        let enc = KernelEncoder::begin(&cmd_buf);
                                        label_prefill_encoder(&enc, il, phase);
                                        if zero_grouped_buffers && bin_idx == 0 {
                                            encode_fill_f32(
                                                base.ctx,
                                                &enc,
                                                &moe_group_out_pack_p,
                                                0.0,
                                            )?;
                                        }
                                        if moe.down_exps.dtype == GgmlType::BF16 {
                                            crate::metal::encode_moe_down_bf16_f32_grouped_slots_range(
                                                base.ctx,
                                                &enc,
                                                &moe.down_exps,
                                                &moe_group_inner_pack_p,
                                                &moe_group_count_pack,
                                                &moe_group_ids_pack,
                                                &moe_group_out_pack_p,
                                                f_exp,
                                                h,
                                                n_expert,
                                                chunk_p,
                                                min_slots,
                                                max_slots,
                                            )?;
                                        } else if prefill_moe_tiny_down_r16_enabled(chunk_p)
                                            && min_slots == 0
                                            && max_slots == 7
                                        {
                                            crate::metal::encode_moe_down_q5_K_f32_grouped_slots_tiny8_r16(
                                                base.ctx,
                                                &enc,
                                                &moe.down_exps,
                                                &moe_group_inner_pack_p,
                                                &moe_group_count_pack,
                                                &moe_group_ids_pack,
                                                &moe_group_out_pack_p,
                                                f_exp,
                                                h,
                                                n_expert,
                                                chunk_p,
                                                1,
                                                7,
                                            )?;
                                        } else {
                                            crate::metal::encode_moe_down_q5_K_f32_grouped_slots_range(
                                                base.ctx,
                                                &enc,
                                                &moe.down_exps,
                                                &moe_group_inner_pack_p,
                                                &moe_group_count_pack,
                                                &moe_group_ids_pack,
                                                &moe_group_out_pack_p,
                                                f_exp,
                                                h,
                                                n_expert,
                                                chunk_p,
                                                min_slots,
                                                max_slots,
                                            )?;
                                        }
                                        enc.end();
                                        flush_prefill_layer_phase(
                                            base.ctx,
                                            &mut cmd_buf,
                                            &mut prefill_gpu_total_ms,
                                            trace_layer_phases,
                                            chunk_idx,
                                            chunk_start,
                                            il,
                                            "moe",
                                            phase,
                                        )?;
                                    }
                                } else {
                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    label_prefill_encoder(&enc, il, "moe-routed-grouped-down");
                                    if zero_grouped_buffers {
                                        encode_fill_f32(
                                            base.ctx,
                                            &enc,
                                            &moe_group_out_pack_p,
                                            0.0,
                                        )?;
                                    }
                                    if noop_grouped_down {
                                        encode_fill_f32(
                                            base.ctx,
                                            &enc,
                                            &moe_group_out_pack_p,
                                            0.0,
                                        )?;
                                    } else {
                                        encode_prefill_moe_grouped_down(
                                            base.ctx,
                                            &enc,
                                            &moe.down_exps,
                                            &moe_group_inner_pack_p,
                                            &moe_group_count_pack,
                                            &moe_group_ids_pack,
                                            &moe_group_out_pack_p,
                                            f_exp,
                                            h,
                                            n_expert,
                                            chunk_p,
                                        )?;
                                    }
                                    enc.end();
                                    flush_prefill_layer_phase(
                                        base.ctx,
                                        &mut cmd_buf,
                                        &mut prefill_gpu_total_ms,
                                        trace_layer_phases,
                                        chunk_idx,
                                        chunk_start,
                                        il,
                                        "moe",
                                        "routed_down",
                                    )?;
                                }
                            }
                            if !fused_grouped_finalizer {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                if noop_grouped_down || noop_grouped_reduce {
                                    encode_fill_f32(base.ctx, &enc, &moe_mixer_out_pack_p, 0.0)?;
                                } else {
                                    crate::metal::encode_moe_weighted_sum_packed_f32(
                                        base.ctx,
                                        &enc,
                                        &moe_group_out_pack_p,
                                        &moe_topk_weight_pack_p,
                                        &moe_mixer_out_pack_p,
                                        h,
                                        topk,
                                        chunk_p,
                                    )?;
                                }
                                enc.end();
                                flush_prefill_layer_phase(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                    trace_layer_phases,
                                    chunk_idx,
                                    chunk_start,
                                    il,
                                    "moe",
                                    "routed_reduce",
                                )?;
                            }
                        } else {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            label_prefill_encoder(&enc, il, "moe-routed-grouped");
                            if !fused_route_bucket {
                                crate::metal::encode_moe_route_bucket_slots_f32(
                                    base.ctx,
                                    &enc,
                                    &moe_topk_idx_pack_p,
                                    &moe_group_count_pack,
                                    &moe_group_ids_pack,
                                    n_expert,
                                    chunk_p,
                                    topk,
                                )?;
                            }
                            if zero_grouped_buffers {
                                encode_fill_f32(base.ctx, &enc, &moe_group_inner_pack_p, 0.0)?;
                            }
                            if noop_grouped_swiglu {
                                encode_fill_f32(base.ctx, &enc, &moe_group_inner_pack_p, 0.0)?;
                            } else {
                                encode_prefill_moe_grouped_swiglu(
                                    base.ctx,
                                    &enc,
                                    moe,
                                    &h_pack_p,
                                    &moe_group_count_pack,
                                    &moe_group_ids_pack,
                                    &moe_group_inner_pack_p,
                                    h,
                                    f_exp,
                                    n_expert,
                                    topk,
                                    chunk_p,
                                    grouped_q4_n32_all,
                                    hot_expert_min_slots,
                                )?;
                            }
                            if zero_grouped_buffers {
                                encode_fill_f32(base.ctx, &enc, &moe_group_out_pack_p, 0.0)?;
                            }
                            if noop_grouped_down {
                                encode_fill_f32(base.ctx, &enc, &moe_group_out_pack_p, 0.0)?;
                            } else {
                                encode_prefill_moe_grouped_down(
                                    base.ctx,
                                    &enc,
                                    &moe.down_exps,
                                    &moe_group_inner_pack_p,
                                    &moe_group_count_pack,
                                    &moe_group_ids_pack,
                                    &moe_group_out_pack_p,
                                    f_exp,
                                    h,
                                    n_expert,
                                    chunk_p,
                                )?;
                            }
                            if !fused_grouped_finalizer {
                                if noop_grouped_down || noop_grouped_reduce {
                                    encode_fill_f32(base.ctx, &enc, &moe_mixer_out_pack_p, 0.0)?;
                                } else {
                                    crate::metal::encode_moe_weighted_sum_packed_f32(
                                        base.ctx,
                                        &enc,
                                        &moe_group_out_pack_p,
                                        &moe_topk_weight_pack_p,
                                        &moe_mixer_out_pack_p,
                                        h,
                                        topk,
                                        chunk_p,
                                    )?;
                                }
                            }
                            enc.end();
                            flush_prefill_layer_phase(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                trace_layer_phases,
                                chunk_idx,
                                chunk_start,
                                il,
                                "moe",
                                "routed_grouped",
                            )?;
                        }
                    } else if let Some(hot_threshold) = hot_expert_min_slots {
                        layer_scratch.ensure_moe_packed_fallback(base.ctx)?;
                        let moe_inner_pack_p = layer_scratch
                            .moe_inner_pack
                            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
                        let enc = KernelEncoder::begin(&cmd_buf);
                        label_prefill_encoder(&enc, il, "moe-routed-cpu-hot");
                        encode_moe_swiglu_q4_K_f32_packed_slots(
                            base.ctx,
                            &enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &h_pack_p,
                            &moe_topk_idx_pack_p,
                            &moe_inner_pack_p,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                        )?;
                        enc.end();

                        cmd_buf.commit();
                        cmd_buf.waitUntilCompleted();
                        require_prefill_command_completed(&cmd_buf)?;
                        prefill_gpu_total_ms +=
                            (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

                        let topk_idx_cpu = cpu_read_i32_f32buf(&moe_topk_idx_pack_p);
                        let topk_weight_cpu = cpu_read_f32buf(&moe_topk_weight_pack_p);
                        let (groups, slot_ids, token_ids, weights) = build_expert_slot_groups_cpu(
                            &topk_idx_cpu,
                            &topk_weight_cpu,
                            topk,
                            n_expert,
                        );
                        let hot_groups: Vec<_> = groups
                            .iter()
                            .copied()
                            .filter(|g| g.len >= hot_threshold)
                            .collect();
                        let mut cold_idx = topk_idx_cpu.clone();
                        for group in &hot_groups {
                            for &slot_id in &slot_ids[group.start..group.start + group.len] {
                                cold_idx[slot_id as usize] = -1;
                            }
                        }
                        cpu_write_i32buf(&moe_group_slot_idx_pack_p, &slot_ids);
                        cpu_write_i32_f32buf(&moe_group_token_idx_pack_p, &token_ids);
                        cpu_write_f32buf(&moe_group_weight_pack_p, &weights);
                        cpu_write_i32_f32buf(&moe_topk_idx_pack_p, &cold_idx);

                        cmd_buf = base.ctx.queue.commandBuffer().expect("command buffer");
                        let enc = KernelEncoder::begin(&cmd_buf);
                        label_prefill_encoder(&enc, il, "moe-routed-cpu-hot-down");
                        encode_fill_f32(base.ctx, &enc, &moe_mixer_out_pack_p, 0.0)?;
                        for group in &hot_groups {
                            let slot_ids_n = moe_group_slot_idx_pack_p
                                .view_subrange(group.start as u64, vec![group.len as u64]);
                            let token_ids_n = moe_group_token_idx_pack_p
                                .view_subrange(group.start as u64, vec![group.len as u64]);
                            let weights_n = moe_group_weight_pack_p
                                .view_subrange(group.start as u64, vec![group.len as u64]);
                            let inner_n = moe_group_inner_pack_p.view_subrange(
                                (group.start * f_exp) as u64,
                                vec![(group.len * f_exp) as u64],
                            );
                            let out_n = moe_group_out_pack_p.view_subrange(
                                (group.start * h) as u64,
                                vec![(group.len * h) as u64],
                            );
                            encode_get_rows_f32(
                                base.ctx,
                                &enc,
                                &moe_inner_pack_p,
                                &slot_ids_n,
                                &inner_n,
                                group.len,
                                f_exp,
                            )?;
                            let expert_bytes = moe.down_exps.n_bytes() / n_expert as u64;
                            let expert_w = moe.down_exps.view_bytes(
                                group.expert as u64 * expert_bytes,
                                vec![(h * f_exp) as u64],
                            );
                            encode_mat_mat_dispatch(
                                base.ctx, &enc, &expert_w, &inner_n, &out_n, f_exp, h, group.len,
                            )?;
                            crate::metal::encode_scatter_axpy_rows_unique_f32(
                                base.ctx,
                                &enc,
                                &out_n,
                                &token_ids_n,
                                &weights_n,
                                &moe_mixer_out_pack_p,
                                h,
                                group.len,
                            )?;
                        }
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                            base.ctx,
                            &enc,
                            &moe.down_exps,
                            &moe_inner_pack_p,
                            &moe_topk_idx_pack_p,
                            &moe_topk_weight_pack_p,
                            &moe_shared_ffn_out_pack_p,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            chunk_p,
                        )?;
                        encode_add_inplace_f32(
                            base.ctx,
                            &enc,
                            &moe_mixer_out_pack_p,
                            &moe_shared_ffn_out_pack_p,
                        )?;
                        enc.end();
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "routed_cpu_hot_down",
                        )?;
                    } else if prefill_moe_packed_down_sum_enabled() {
                        layer_scratch.ensure_moe_packed_fallback(base.ctx)?;
                        let moe_inner_pack_p = layer_scratch
                            .moe_inner_pack
                            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
                        let enc = KernelEncoder::begin(&cmd_buf);
                        label_prefill_encoder(&enc, il, "moe-routed-packed-down-sum");
                        encode_moe_swiglu_q4_K_f32_packed_slots(
                            base.ctx,
                            &enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &h_pack_p,
                            &moe_topk_idx_pack_p,
                            &moe_inner_pack_p,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                        )?;
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                            base.ctx,
                            &enc,
                            &moe.down_exps,
                            &moe_inner_pack_p,
                            &moe_topk_idx_pack_p,
                            &moe_topk_weight_pack_p,
                            &moe_mixer_out_pack_p,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            chunk_p,
                        )?;
                        enc.end();
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "routed_packed_down_sum",
                        )?;
                    } else {
                        layer_scratch.ensure_moe_packed_fallback(base.ctx)?;
                        let moe_inner_pack_p = layer_scratch
                            .moe_inner_pack
                            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
                        let moe_expert_out_pack_p = layer_scratch
                            .moe_expert_out_pack
                            .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
                        let enc = KernelEncoder::begin(&cmd_buf);
                        label_prefill_encoder(&enc, il, "moe-routed-token-loop");
                        encode_moe_swiglu_q4_K_f32_packed_slots(
                            base.ctx,
                            &enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &h_pack_p,
                            &moe_topk_idx_pack_p,
                            &moe_inner_pack_p,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                        )?;
                        for n_idx in 0..chunk_p {
                            let inner_n = moe_inner_pack_p.view_subrange(
                                (n_idx * topk * f_exp) as u64,
                                vec![(topk * f_exp) as u64],
                            );
                            let idx_n = moe_topk_idx_pack_p
                                .view_subrange((n_idx * topk) as u64, vec![topk as u64]);
                            let weight_n = moe_topk_weight_pack_p
                                .view_subrange((n_idx * topk) as u64, vec![topk as u64]);
                            let expert_out_n = moe_expert_out_pack_p
                                .view_subrange((n_idx * topk * h) as u64, vec![(topk * h) as u64]);
                            let mixer_n = moe_mixer_out_pack_p
                                .view_subrange((n_idx * h) as u64, vec![h as u64]);
                            encode_moe_down_q5_K_f32(
                                base.ctx,
                                &enc,
                                &moe.down_exps,
                                &inner_n,
                                &idx_n,
                                &expert_out_n,
                                f_exp,
                                h,
                                n_expert,
                                topk,
                            )?;
                            encode_moe_weighted_sum_f32(
                                base.ctx,
                                &enc,
                                &expert_out_n,
                                &weight_n,
                                &mixer_n,
                                h,
                                topk,
                            )?;
                        }
                        enc.end();
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "routed_token_loop",
                        )?;
                    }

                    if concurrent_grouped_shared {
                    } else if skip_moe_shared {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        label_prefill_encoder(&enc, il, "moe-shared-skip");
                        if fused_grouped_finalizer {
                            encode_fill_f32(base.ctx, &enc, &moe_shared_ffn_out_pack_p, 0.0)?;
                            crate::metal::encode_moe_grouped_finalizer_f32(
                                base.ctx,
                                &enc,
                                &moe_group_out_pack_p,
                                &moe_topk_weight_pack_p,
                                &moe_shared_gate_pack_p,
                                &moe_shared_ffn_out_pack_p,
                                &x_pack_p,
                                h,
                                topk,
                                chunk_p,
                            )?;
                        } else {
                            encode_add_inplace_f32(
                                base.ctx,
                                &enc,
                                &x_pack_p,
                                &moe_mixer_out_pack_p,
                            )?;
                        }
                        enc.end();
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "shared_skip",
                        )?;
                    } else if packed_shared_path {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        label_prefill_encoder(&enc, il, "moe-shared-packed");
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            g_w,
                            &h_pack_p,
                            &moe_shared_ffn_gate_pack_p,
                            h,
                            f_shared,
                            chunk_p,
                        )?;
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            u_w,
                            &h_pack_p,
                            &moe_shared_ffn_up_pack_p,
                            h,
                            f_shared,
                            chunk_p,
                        )?;
                        encode_silu_mul_f32(
                            base.ctx,
                            &enc,
                            &moe_shared_ffn_gate_pack_p,
                            &moe_shared_ffn_up_pack_p,
                            &moe_shared_ffn_inner_pack_p,
                        )?;
                        encode_mat_mat_dispatch(
                            base.ctx,
                            &enc,
                            d_w,
                            &moe_shared_ffn_inner_pack_p,
                            &moe_shared_ffn_out_pack_p,
                            f_shared,
                            h,
                            chunk_p,
                        )?;
                        if fused_grouped_finalizer {
                            crate::metal::encode_moe_grouped_finalizer_f32(
                                base.ctx,
                                &enc,
                                &moe_group_out_pack_p,
                                &moe_topk_weight_pack_p,
                                &moe_shared_gate_pack_p,
                                &moe_shared_ffn_out_pack_p,
                                &x_pack_p,
                                h,
                                topk,
                                chunk_p,
                            )?;
                        } else {
                            encode_axpy_rowwise_f32(
                                base.ctx,
                                &enc,
                                &moe_shared_ffn_out_pack_p,
                                &moe_shared_gate_pack_p,
                                &moe_mixer_out_pack_p,
                                h,
                                chunk_p,
                            )?;
                            encode_add_inplace_f32(
                                base.ctx,
                                &enc,
                                &x_pack_p,
                                &moe_mixer_out_pack_p,
                            )?;
                        }
                        enc.end();
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "shared_packed",
                        )?;
                    } else {
                        for n_idx in 0..chunk_p {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            label_prefill_encoder(&enc, il, "moe-shared-token-loop");
                            encode_copy_offset_f32(
                                base.ctx,
                                &enc,
                                &x_pack_p,
                                n_idx * h,
                                &target_session.x,
                                h,
                            )?;
                            encode_copy_offset_f32(
                                base.ctx,
                                &enc,
                                &h_pack_p,
                                n_idx * h,
                                &target_session.h,
                                h,
                            )?;
                            let mixer_n = moe_mixer_out_pack_p
                                .view_subrange((n_idx * h) as u64, vec![h as u64]);
                            encode_copy_offset_f32(
                                base.ctx,
                                &enc,
                                &mixer_n,
                                0,
                                &target_session.mixer_out,
                                h,
                            )?;
                            let shared_gate_n =
                                moe_shared_gate_pack_p.view_subrange(n_idx as u64, vec![1]);
                            encode_copy_offset_f32(
                                base.ctx,
                                &enc,
                                &shared_gate_n,
                                0,
                                &target_session.moe_shared_gate,
                                1,
                            )?;
                            base.encode_moe_shared_ffn_gpu(&enc, target_session, g_w, u_w, d_w)?;
                            encode_add_inplace_f32(
                                base.ctx,
                                &enc,
                                &target_session.x,
                                &target_session.mixer_out,
                            )?;
                            encode_scatter_offset_f32(
                                base.ctx,
                                &enc,
                                &target_session.x,
                                &x_pack_p,
                                n_idx * h,
                                h,
                            )?;
                            enc.end();
                        }
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "shared_token_loop",
                        )?;
                    }
                } else if !skip_ffn {
                    if trace_layer_phases {
                        let mut copy_ms = 0.0f64;
                        let mut route_ms = 0.0f64;
                        let mut routed_gate_ms = 0.0f64;
                        let mut routed_up_ms = 0.0f64;
                        let mut routed_silu_ms = 0.0f64;
                        let mut routed_down_ms = 0.0f64;
                        let mut routed_weighted_sum_ms = 0.0f64;
                        let mut routed_other_ms = 0.0f64;
                        let mut shared_ms = 0.0f64;
                        let mut residual_scatter_ms = 0.0f64;

                        for n_idx in 0..chunk_p {
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                encode_copy_offset_f32(
                                    base.ctx,
                                    &enc,
                                    &x_pack_p,
                                    n_idx * h,
                                    &target_session.x,
                                    h,
                                )?;
                                encode_copy_offset_f32(
                                    base.ctx,
                                    &enc,
                                    &h_pack_p,
                                    n_idx * h,
                                    &target_session.h,
                                    h,
                                )?;
                                enc.end();
                            }
                            copy_ms += flush_prefill_layer_phase_accum(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                            )?;

                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                base.encode_moe_route_prepare(&enc, target_session, moe)?;
                                enc.end();
                            }
                            route_ms += flush_prefill_layer_phase_accum(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                            )?;

                            if matches!(
                                (moe.gate_exps.dtype, moe.up_exps.dtype, moe.down_exps.dtype),
                                (GgmlType::F32, GgmlType::F32, GgmlType::IQ4_XS)
                            ) {
                                let moe_inner = target_session
                                    .moe_inner
                                    .view_subrange(0, vec![(topk * f_exp) as u64]);
                                let moe_expert_out = target_session
                                    .moe_expert_out
                                    .view_subrange(0, vec![(topk * h) as u64]);
                                let gate_pack = target_session
                                    .moe_expert_out
                                    .view_subrange(0, vec![(topk * f_exp) as u64]);
                                let up_pack = target_session.moe_expert_out.view_subrange(
                                    (topk * f_exp) as u64,
                                    vec![(topk * f_exp) as u64],
                                );
                                let topk_idx = target_session
                                    .moe_topk_idx
                                    .view_subrange(0, vec![topk as u64]);
                                let topk_w = target_session
                                    .moe_topk_weight
                                    .view_subrange(0, vec![topk as u64]);

                                {
                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    encode_moe_mat_vec_f32(
                                        base.ctx,
                                        &enc,
                                        &moe.gate_exps,
                                        &target_session.h,
                                        &topk_idx,
                                        &gate_pack,
                                        h,
                                        f_exp,
                                        n_expert,
                                        topk,
                                    )?;
                                    enc.end();
                                }
                                routed_gate_ms += flush_prefill_layer_phase_accum(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                )?;

                                {
                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    encode_moe_mat_vec_f32(
                                        base.ctx,
                                        &enc,
                                        &moe.up_exps,
                                        &target_session.h,
                                        &topk_idx,
                                        &up_pack,
                                        h,
                                        f_exp,
                                        n_expert,
                                        topk,
                                    )?;
                                    enc.end();
                                }
                                routed_up_ms += flush_prefill_layer_phase_accum(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                )?;

                                {
                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    encode_silu_mul_f32(
                                        base.ctx, &enc, &gate_pack, &up_pack, &moe_inner,
                                    )?;
                                    enc.end();
                                }
                                routed_silu_ms += flush_prefill_layer_phase_accum(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                )?;

                                {
                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    encode_moe_down_iq4_xs_f32(
                                        base.ctx,
                                        &enc,
                                        &moe.down_exps,
                                        &moe_inner,
                                        &topk_idx,
                                        &moe_expert_out,
                                        f_exp,
                                        h,
                                        n_expert,
                                        topk,
                                    )?;
                                    enc.end();
                                }
                                routed_down_ms += flush_prefill_layer_phase_accum(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                )?;

                                {
                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    encode_moe_weighted_sum_f32(
                                        base.ctx,
                                        &enc,
                                        &moe_expert_out,
                                        &topk_w,
                                        &target_session.mixer_out,
                                        h,
                                        topk,
                                    )?;
                                    enc.end();
                                }
                                routed_weighted_sum_ms += flush_prefill_layer_phase_accum(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                )?;
                            } else {
                                {
                                    let enc = KernelEncoder::begin(&cmd_buf);
                                    base.encode_moe_routed_ffn_gpu(&enc, target_session, moe)?;
                                    enc.end();
                                }
                                routed_other_ms += flush_prefill_layer_phase_accum(
                                    base.ctx,
                                    &mut cmd_buf,
                                    &mut prefill_gpu_total_ms,
                                )?;
                            }

                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                base.encode_moe_shared_ffn_gpu(
                                    &enc,
                                    target_session,
                                    g_w,
                                    u_w,
                                    d_w,
                                )?;
                                enc.end();
                            }
                            shared_ms += flush_prefill_layer_phase_accum(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                            )?;

                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                encode_add_inplace_f32(
                                    base.ctx,
                                    &enc,
                                    &target_session.x,
                                    &target_session.mixer_out,
                                )?;
                                encode_scatter_offset_f32(
                                    base.ctx,
                                    &enc,
                                    &target_session.x,
                                    &x_pack_p,
                                    n_idx * h,
                                    h,
                                )?;
                                enc.end();
                            }
                            residual_scatter_ms += flush_prefill_layer_phase_accum(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                            )?;
                        }

                        emit_prefill_layer_phase(
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "fallback_copy_token_loop",
                            copy_ms,
                        );
                        emit_prefill_layer_phase(
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "fallback_route_token_loop",
                            route_ms,
                        );
                        emit_prefill_layer_phase(
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "fallback_routed_gate_f32_token_loop",
                            routed_gate_ms,
                        );
                        emit_prefill_layer_phase(
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "fallback_routed_up_f32_token_loop",
                            routed_up_ms,
                        );
                        emit_prefill_layer_phase(
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "fallback_routed_silu_token_loop",
                            routed_silu_ms,
                        );
                        emit_prefill_layer_phase(
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "fallback_routed_down_iq4_xs_token_loop",
                            routed_down_ms,
                        );
                        emit_prefill_layer_phase(
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "fallback_routed_weighted_sum_token_loop",
                            routed_weighted_sum_ms,
                        );
                        emit_prefill_layer_phase(
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "fallback_routed_other_token_loop",
                            routed_other_ms,
                        );
                        emit_prefill_layer_phase(
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "fallback_shared_token_loop",
                            shared_ms,
                        );
                        emit_prefill_layer_phase(
                            chunk_idx,
                            chunk_start,
                            il,
                            "moe",
                            "fallback_residual_scatter_token_loop",
                            residual_scatter_ms,
                        );
                    } else {
                        for n_idx in 0..chunk_p {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_copy_offset_f32(
                                base.ctx,
                                &enc,
                                &x_pack_p,
                                n_idx * h,
                                &target_session.x,
                                h,
                            )?;
                            encode_copy_offset_f32(
                                base.ctx,
                                &enc,
                                &h_pack_p,
                                n_idx * h,
                                &target_session.h,
                                h,
                            )?;
                            base.encode_moe_route_prepare(&enc, target_session, moe)?;
                            base.encode_moe_ffn_apply_gpu(
                                &enc,
                                target_session,
                                g_w,
                                u_w,
                                d_w,
                                moe,
                            )?;
                            encode_scatter_offset_f32(
                                base.ctx,
                                &enc,
                                &target_session.x,
                                &x_pack_p,
                                n_idx * h,
                                h,
                            )?;
                            enc.end();
                        }
                    }
                }

                if let Some(dst) = hidden_dst {
                    let enc = KernelEncoder::begin(&cmd_buf);
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
                    enc.end();
                }
            } else {
                let mat_mat_path = ffn_mat_mat_eligible(g_w.dtype)
                    && ffn_mat_mat_eligible(u_w.dtype)
                    && ffn_mat_mat_eligible(d_w.dtype);
                if trace_layer_phases && !skip_ffn && mat_mat_path {
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_rms_norm_batched_f32(
                            base.ctx, &enc, &x_pack_p, post_norm, &h_pack_p, chunk_p, h, RMS_EPS,
                        )?;
                        enc.end();
                    }
                    flush_prefill_layer_phase(
                        base.ctx,
                        &mut cmd_buf,
                        &mut prefill_gpu_total_ms,
                        trace_layer_phases,
                        chunk_idx,
                        chunk_start,
                        il,
                        block_kind,
                        "ffn_norm",
                    )?;

                    let use_fused_swiglu = prefill_dense_ffn_fused_swiglu_q4_enabled(h)
                        && g_w.dtype == GgmlType::Q4_K
                        && u_w.dtype == GgmlType::Q4_K
                        && h.is_multiple_of(256)
                        && chunk_p >= 32;
                    let split_ffn_subphases =
                        prefill_trace_ffn_subphases_enabled() && !use_fused_swiglu && !skip_ffn;
                    if split_ffn_subphases {
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
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
                            enc.end();
                        }
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            block_kind,
                            "ffn_gate",
                        )?;

                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
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
                            enc.end();
                        }
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            block_kind,
                            "ffn_up",
                        )?;

                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_silu_mul_f32(
                                base.ctx,
                                &enc,
                                &ffn_gate_pack_p,
                                &ffn_up_pack_p,
                                &ffn_inner_pack_p,
                            )?;
                            enc.end();
                        }
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            block_kind,
                            "ffn_swiglu",
                        )?;
                    } else {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        if use_fused_swiglu {
                            crate::metal::encode_ffn_fused_swiglu_q4_K_mm_f32(
                                base.ctx,
                                &enc,
                                g_w,
                                u_w,
                                &h_pack_p,
                                &ffn_inner_pack_p,
                                h,
                                f,
                                chunk_p,
                            )?;
                        } else {
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
                        }
                        enc.end();
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            block_kind,
                            if use_fused_swiglu {
                                "ffn_fused_gate_up_swiglu"
                            } else {
                                "ffn_gate_up_swiglu"
                            },
                        )?;
                    }

                    if split_ffn_subphases {
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
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
                            enc.end();
                        }
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            block_kind,
                            "ffn_down",
                        )?;

                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
                            encode_add_inplace_f32(base.ctx, &enc, &x_pack_p, &ffn_out_pack_p)?;
                            if let Some(dst) = hidden_dst {
                                for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                                    if lid as usize == il {
                                        for n_idx in 0..chunk_p {
                                            let global_idx = chunk_base + n_idx;
                                            let elem_off = (global_idx * k_target + k_idx) * h;
                                            let row_view = x_pack_p
                                                .view_subrange((n_idx * h) as u64, vec![h as u64]);
                                            encode_scatter_offset_f32(
                                                base.ctx, &enc, &row_view, dst, elem_off, h,
                                            )?;
                                        }
                                    }
                                }
                            }
                            enc.end();
                        }
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            block_kind,
                            "ffn_resid",
                        )?;
                    } else {
                        {
                            let enc = KernelEncoder::begin(&cmd_buf);
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
                            encode_add_inplace_f32(base.ctx, &enc, &x_pack_p, &ffn_out_pack_p)?;
                            if let Some(dst) = hidden_dst {
                                for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                                    if lid as usize == il {
                                        for n_idx in 0..chunk_p {
                                            let global_idx = chunk_base + n_idx;
                                            let elem_off = (global_idx * k_target + k_idx) * h;
                                            let row_view = x_pack_p
                                                .view_subrange((n_idx * h) as u64, vec![h as u64]);
                                            encode_scatter_offset_f32(
                                                base.ctx, &enc, &row_view, dst, elem_off, h,
                                            )?;
                                        }
                                    }
                                }
                            }
                            enc.end();
                        }
                        flush_prefill_layer_phase(
                            base.ctx,
                            &mut cmd_buf,
                            &mut prefill_gpu_total_ms,
                            trace_layer_phases,
                            chunk_idx,
                            chunk_start,
                            il,
                            block_kind,
                            "ffn_down_resid",
                        )?;
                    }
                } else {
                    let enc = KernelEncoder::begin(&cmd_buf);
                    encode_rms_norm_batched_f32(
                        base.ctx, &enc, &x_pack_p, post_norm, &h_pack_p, chunk_p, h, RMS_EPS,
                    )?;
                    if skip_ffn {
                        // profiling only: leave x_pack unchanged after post-norm so a
                        // production-shape run can report the direct FFN wall delta.
                    } else if mat_mat_path {
                        if prefill_dense_ffn_fused_swiglu_q4_enabled(h)
                            && g_w.dtype == GgmlType::Q4_K
                            && u_w.dtype == GgmlType::Q4_K
                            && h.is_multiple_of(256)
                            && chunk_p >= 32
                        {
                            crate::metal::encode_ffn_fused_swiglu_q4_K_mm_f32(
                                base.ctx,
                                &enc,
                                g_w,
                                u_w,
                                &h_pack_p,
                                &ffn_inner_pack_p,
                                h,
                                f,
                                chunk_p,
                            )?;
                        } else {
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
                        }
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
                        encode_add_inplace_f32(base.ctx, &enc, &x_pack_p, &ffn_out_pack_p)?;
                    } else {
                        // F32 / non-mat-mat fallback: per-token mat-vec.
                        for n_idx in 0..chunk_p {
                            let h_n = h_pack_p.view_subrange((n_idx * h) as u64, vec![h as u64]);
                            let gate_n =
                                ffn_gate_pack_p.view_subrange((n_idx * f) as u64, vec![f as u64]);
                            let up_n =
                                ffn_up_pack_p.view_subrange((n_idx * f) as u64, vec![f as u64]);
                            let inner_n =
                                ffn_inner_pack_p.view_subrange((n_idx * f) as u64, vec![f as u64]);
                            let out_n =
                                ffn_out_pack_p.view_subrange((n_idx * h) as u64, vec![h as u64]);
                            encode_mat_vec_dispatch(base.ctx, &enc, g_w, &h_n, &gate_n, h, f)?;
                            encode_mat_vec_dispatch(base.ctx, &enc, u_w, &h_n, &up_n, h, f)?;
                            encode_silu_mul_f32(base.ctx, &enc, &gate_n, &up_n, &inner_n)?;
                            encode_mat_vec_dispatch(base.ctx, &enc, d_w, &inner_n, &out_n, f, h)?;
                        }
                        encode_add_inplace_f32(base.ctx, &enc, &x_pack_p, &ffn_out_pack_p)?;
                    }

                    // 2d-fix (matches v0.74.4 capture point): hidden capture
                    // after residual #2. Writes into the GLOBAL hidden_dst at
                    // offset ((global_idx * K + k_idx) * H), where
                    // global_idx = chunk_base + n_idx. Skipped entirely when
                    // hidden_dst is None (no-spec ref path).
                    //
                    // v0.432: one strided-row copy per capture layer instead
                    // of chunk_p scatter dispatches — dst rows for
                    // consecutive n_idx are uniformly strided by K * H.
                    if let Some(dst) = hidden_dst {
                        for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                            if lid as usize == il {
                                let dst_base = (chunk_base * k_target + k_idx) * h;
                                crate::metal_forward::encode_copy_rows_dst_strided_f32(
                                    base.ctx,
                                    &enc,
                                    &x_pack_p,
                                    dst,
                                    chunk_p,
                                    h,
                                    k_target * h,
                                    dst_base,
                                )?;
                            }
                        }
                    }

                    enc.end();
                    flush_prefill_layer_phase(
                        base.ctx,
                        &mut cmd_buf,
                        &mut prefill_gpu_total_ms,
                        trace_layer_phases,
                        chunk_idx,
                        chunk_start,
                        il,
                        block_kind,
                        "ffn",
                    )?;
                }
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
            if !matches!(tail_mode, PrefillTailMode::SkipTail) {
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
                let tail = match tail_mode {
                    PrefillTailMode::ReadLogits => LmHeadTail::Resident,
                    PrefillTailMode::Supplied(tail) => tail,
                    PrefillTailMode::SkipTail => unreachable!("skip tail was excluded"),
                };
                base.encode_lm_head_tail(&enc, target_session, tail)?;
                enc.end();
            }
            emit_prefill_count_phase(trace_counts, chunk_idx, chunk_start, 0, "chunk", "tail");
            let before_commit = Instant::now();
            cmd_buf.commit();
            let after_commit = Instant::now();
            cmd_buf.waitUntilCompleted();
            let after_wait = Instant::now();
            if let Some(evidence) = tail_evidence.as_mut() {
                let status = cmd_buf.status();
                let error = cmd_buf.error();
                evidence.command_completed = status == MTLCommandBufferStatus::Completed;
                evidence.command_error_none = error.is_none();
            }
            require_prefill_command_completed(&cmd_buf)?;
            let chunk_gpu_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;
            prefill_gpu_total_ms += chunk_gpu_ms;
            let mut tail_readback_ms = 0.0f64;
            let result_logits = if matches!(tail_mode, PrefillTailMode::ReadLogits) {
                let readback_start = Instant::now();
                let mut last_logits = vec![0.0f32; v];
                unsafe {
                    let src = target_session.logits.buffer.contents().as_ptr() as *const f32;
                    std::ptr::copy_nonoverlapping(src, last_logits.as_mut_ptr(), v);
                }
                tail_readback_ms = readback_start.elapsed().as_secs_f64() * 1e3;
                Some(last_logits)
            } else {
                None
            };
            let chunk_wall_ms = chunk_wall.elapsed().as_secs_f64() * 1e3;
            if prefill_trace_chunks_enabled() {
                eprintln!(
                    "[prefill-chunk] idx={} start={} tokens={} gpu_ms={:.2} wall_ms={:.2} ms_per_tok={:.4} cumulative_gpu_ms={:.2}",
                    chunk_idx,
                    chunk_start,
                    chunk_p,
                    chunk_gpu_ms,
                    chunk_wall_ms,
                    chunk_gpu_ms / chunk_p as f64,
                    prefill_gpu_total_ms,
                );
            }
            if trace_wall {
                let kernel_trace = kernel_trace_snapshot();
                eprintln!(
                    "[prefill-wall] idx={} start={} tokens={} last=1 tail={} setup_ms={:.3} encode_ms={:.3} commit_ms={:.3} wait_ms={:.3} readback_ms={:.3} encoders={} concurrent_encoders={} dispatches={} gpu_ms={:.3} wall_ms={:.3} cumulative_gpu_ms={:.3}",
                    chunk_idx,
                    chunk_start,
                    chunk_p,
                    match tail_mode {
                        PrefillTailMode::ReadLogits => "read",
                        PrefillTailMode::SkipTail => "skip",
                        PrefillTailMode::Supplied(_) => "supplied",
                    },
                    (chunk_encode_start - chunk_wall).as_secs_f64() * 1e3,
                    (before_commit - chunk_encode_start).as_secs_f64() * 1e3,
                    (after_commit - before_commit).as_secs_f64() * 1e3,
                    (after_wait - after_commit).as_secs_f64() * 1e3,
                    tail_readback_ms,
                    kernel_trace.encoders,
                    kernel_trace.concurrent_encoders,
                    kernel_trace.dispatches,
                    chunk_gpu_ms,
                    chunk_wall_ms,
                    prefill_gpu_total_ms,
                );
            }
            return Ok((result_logits, prefill_gpu_total_ms, tail_evidence));
        }

        // Non-last chunk: just commit + wait (no tail).
        let before_commit = Instant::now();
        cmd_buf.commit();
        let after_commit = Instant::now();
        cmd_buf.waitUntilCompleted();
        let after_wait = Instant::now();
        require_prefill_command_completed(&cmd_buf)?;
        let chunk_gpu_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;
        let chunk_wall_ms = chunk_wall.elapsed().as_secs_f64() * 1e3;
        prefill_gpu_total_ms += chunk_gpu_ms;
        if prefill_trace_chunks_enabled() {
            eprintln!(
                "[prefill-chunk] idx={} start={} tokens={} gpu_ms={:.2} wall_ms={:.2} ms_per_tok={:.4} cumulative_gpu_ms={:.2}",
                chunk_idx,
                chunk_start,
                chunk_p,
                chunk_gpu_ms,
                chunk_wall_ms,
                chunk_gpu_ms / chunk_p as f64,
                prefill_gpu_total_ms,
            );
        }
        if trace_wall {
            let kernel_trace = kernel_trace_snapshot();
            eprintln!(
                "[prefill-wall] idx={} start={} tokens={} last=0 tail=skip setup_ms={:.3} encode_ms={:.3} commit_ms={:.3} wait_ms={:.3} readback_ms=0.000 encoders={} concurrent_encoders={} dispatches={} gpu_ms={:.3} wall_ms={:.3} cumulative_gpu_ms={:.3}",
                chunk_idx,
                chunk_start,
                chunk_p,
                (chunk_encode_start - chunk_wall).as_secs_f64() * 1e3,
                (before_commit - chunk_encode_start).as_secs_f64() * 1e3,
                (after_commit - before_commit).as_secs_f64() * 1e3,
                (after_wait - after_commit).as_secs_f64() * 1e3,
                kernel_trace.encoders,
                kernel_trace.concurrent_encoders,
                kernel_trace.dispatches,
                chunk_gpu_ms,
                chunk_wall_ms,
                prefill_gpu_total_ms,
            );
        }
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
    n_eff_override: Option<u32>,
) -> Result<(), DFlashError> {
    // -- Validation guard wall (same discipline as packed_verify).
    //
    // `n_block` is the scratch allocation size. `n` is the EFFECTIVE
    // chain length used by the most recent packed_verify call (which
    // wrote ckpt slots [0, n) and advanced kv_n_pos to start_position +
    // n). `restore` must be passed the SAME `n_eff_override` value as
    // the packed_verify call it follows — otherwise the bounds checks
    // and kv_n_pos contract are wrong.
    let n_block = scratch.n;
    let n = n_eff_override.unwrap_or(n_block);
    if n == 0 || n > n_block {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "restore_after_partial_accept",
            detail: format!(
                "n_eff_override={:?} resolves to n={n} which must be in [1, n_block={n_block}]",
                n_eff_override
            ),
        }));
    }
    if n_keep == 0 || n_keep > n {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "restore_after_partial_accept",
            detail: format!(
                "n_keep={n_keep} must be in [1, n_eff={n}] (n_keep=0 is impossible \
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
    // Full accept needs no copy: the live session already holds the final state,
    // and packed verification intentionally does not publish the unreachable
    // final checkpoint slot.
    if packed_verify_skip_final_ckpt_enabled() && n_keep == n {
        return Ok(());
    }
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

/// Roll a just-executed packed verify back to its PRE-BLOCK state using the
/// phase-0 capture. Used by the margin-guarded exact fallback: when a
/// committed row's argmax gap is too small to trust the batched arithmetic,
/// the whole block is replayed through token-major forwards.
///
/// Contract mirrors [`encode_restore_after_partial_accept_inner`]: call
/// immediately after `encode_packed_verify_layer_major_inner(..,
/// start_position, .., n_eff_override)` on the same session; kv_n_pos must
/// still be `start_position + n`. KV bytes at `[start_position, ..)` remain
/// physically but become unreachable and are overwritten by the replay.
pub fn encode_restore_to_pre_block(
    base: &MetalForward<'_>,
    scratch: &MetalDFlashVerifyScratch,
    start_position: u32,
    target_session: &mut MetalSession,
    n_eff_override: Option<u32>,
) -> Result<(), DFlashError> {
    let n_block = scratch.n;
    let n = n_eff_override.unwrap_or(n_block);
    if n == 0 || n > n_block {
        return Err(DFlashError::Metal(MetalError::BadShape {
            kernel: "restore_to_pre_block",
            detail: format!(
                "n_eff_override={:?} resolves to n={n} which must be in [1, n_block={n_block}]",
                n_eff_override
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
            kernel: "restore_to_pre_block",
            detail: format!(
                "scratch.n_gdn_layers={} != model n_gdn={n_gdn_actual} \
                 (scratch allocated for different layer schedule)",
                scratch.n_gdn_layers
            ),
        }));
    }
    let expected_kv_pre = (start_position as usize)
        .checked_add(n as usize)
        .ok_or_else(|| {
            DFlashError::Metal(MetalError::BadShape {
                kernel: "restore_to_pre_block",
                detail: format!("start_position={start_position} + N={n} overflows usize"),
            })
        })?;
    for (i, &kp) in target_session.kv_n_pos.iter().enumerate() {
        if kp != expected_kv_pre {
            return Err(DFlashError::Metal(MetalError::BadShape {
                kernel: "restore_to_pre_block",
                detail: format!(
                    "kv_n_pos[{i}]={kp} != start_position + N = {expected_kv_pre}; \
                     restore must be called immediately after packed_verify(.., \
                     start_position) on the same session"
                ),
            }));
        }
    }

    let cmd_buf = base.ctx.queue.commandBuffer().expect("command buffer");
    let blit = BlitEncoder::begin(&cmd_buf);
    for k in 0..n_gdn_actual {
        let ssm_src = scratch.pre_gdn_slot(k);
        blit.copy_tensor(&ssm_src, &target_session.gdn_state[k as usize]);
        let conv_src = scratch.pre_conv_slot(k);
        blit.copy_tensor(&conv_src, &target_session.gdn_conv[k as usize]);
    }
    blit.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    for i in 0..target_session.kv_n_pos.len() {
        target_session.kv_n_pos[i] = start_position as usize;
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
        #[cfg(feature = "dflash-k0s-diagnostics")]
        {
            dflash_k0s_require_no_live_event(self.k0s_live_event)?;
        }
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
        // DFlash 2 conv dims (0 / unused for DFlash 1 drafters).
        let conv_kernel = cfg.conv_kernel_size as usize;
        let conv_group = cfg.conv_group_size as usize;
        let conv_dyn_dim = if cfg.selector_top_k > 0 {
            2 * conv_kernel * (h / conv_group)
        } else {
            0
        };

        // Stage noise_ids: [carry_tok, MASK × (N-1)].
        unsafe {
            let ptr = self.session.noise_ids.buffer.contents().as_ptr() as *mut i32;
            *ptr = carry_tok;
            for i in 1..n {
                *ptr.add(i) = cfg.mask_token_id;
            }
        }

        let ctx_metal = self.base.ctx;

        // v0.77 single-command-buffer mode: no data dependency requires
        // CPU intervention between phases (all per-layer constants are
        // known up front; pos_k is layer-invariant and staged below), so
        // everything can queue into one command buffer with one wait at
        // phase 4. Profiled runs keep per-phase buffers for attribution.
        // Rollback: QWEN_DFLASH_DRAFT_SINGLE_CMD=0.
        let single_cmd_mode =
            dflash_draft_single_cmd_enabled() && !self.session.enable_phase_timers;
        let shared_cmd = if single_cmd_mode {
            Some(ctx_metal.queue.commandBuffer().expect("cmd draft"))
        } else {
            None
        };

        // Read pos_ctx once (used by RoPE on K_ctx, the SWA mask, and the
        // pos_k staging below). CPU-visible and synced: all writers ran
        // under previous calls' waits.
        let mut pos_ctx_cpu = vec![0i32; ctx_len];
        if ctx_len > 0 {
            unsafe {
                let src = self.session.pos_ctx.buffer.contents().as_ptr() as *const i32;
                std::ptr::copy_nonoverlapping(src, pos_ctx_cpu.as_mut_ptr(), ctx_len);
            }
        }

        // v0.77: pos_k staging hoisted out of the per-layer loop — the
        // per-layer host write into the shared buffer is what forced a
        // CPU sync between layers. Layout is layer-invariant: ctx
        // positions followed by noise positions. Two-range kernels read
        // only the first ctx_len entries; the legacy concat kernel reads
        // all ctx_len + n.
        {
            let n_kv_total = ctx_len + n;
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
            let phase1_delta = ctx_len - phase1_start;
            let cmd = shared_cmd
                .clone()
                .unwrap_or_else(|| ctx_metal.queue.commandBuffer().expect("cmd"));
            let enc = KernelEncoder::begin(&cmd);
            if dflash_batched_proj_enabled() && phase1_delta > 1 {
                let src = self.session.target_ctx_stacked.view_subrange(
                    (phase1_start * n_target_features) as u64,
                    vec![(phase1_delta * n_target_features) as u64],
                );
                let dst = self
                    .session
                    .ctx_h
                    .view_subrange((phase1_start * h) as u64, vec![(phase1_delta * h) as u64]);
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &self.head.fc,
                    &src,
                    &dst,
                    n_target_features,
                    h,
                    phase1_delta,
                )?;
                encode_rms_norm_batched_f32(
                    ctx_metal,
                    &enc,
                    &dst,
                    &self.head.hidden_norm,
                    &dst,
                    phase1_delta,
                    h,
                    RMS_EPS,
                )?;
            } else {
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
            }
            enc.end();
            if shared_cmd.is_none() {
                cmd.commit();
                cmd.waitUntilCompleted();
                self.session.maybe_record("phase1_ctx_fc_norm", &cmd);
            }
            // ctx_h_ready_n watermark advances after the final wait (see
            // the phase-4 tail) — in single-cmd mode nothing has executed
            // yet at this point.
        }

        // ----- Phase 2 (Metal): noise embed + per-layer fwd through
        //     pre-attn-norm, Q/K/V projections, per-head Q/K-norm, RoPE.
        //     v0.527 batches the regular projection/RoPE work; phase 3
        //     handles asymmetric SWA attention and the FFN fully on Metal.
        let cmd = shared_cmd
            .clone()
            .unwrap_or_else(|| ctx_metal.queue.commandBuffer().expect("cmd"));
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
        if shared_cmd.is_none() {
            cmd.commit();
            cmd.waitUntilCompleted();
            self.session.maybe_record("phase2_embed", &cmd);
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
        let batched_proj = dflash_batched_proj_enabled();

        for (layer_idx, layer) in self.head.layers.iter().enumerate() {
            // Pre-attn norm: x → h (Metal).
            let cmd = shared_cmd
                .clone()
                .unwrap_or_else(|| ctx_metal.queue.commandBuffer().expect("cmd"));
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
            // DFlash 2 attention conv: dynamic coefficients from the
            // PRE-conv normed input (shared by both sides), then the
            // side-0 conv. Q/K/V projections read the conv'd result.
            let attn_qkv_src: &MetalTensor = if let Some(cv) = layer.conv.as_ref() {
                let dyn_attn = self
                    .session
                    .conv_dyn_attn
                    .as_ref()
                    .expect("dflash2 dyn_attn");
                let conv_buf = self.session.conv_buf.as_ref().expect("dflash2 conv_buf");
                if batched_proj {
                    encode_mat_mat_dispatch(
                        ctx_metal,
                        &enc,
                        &cv.attn_proj,
                        &self.session.h,
                        dyn_attn,
                        h,
                        conv_dyn_dim,
                        n,
                    )?;
                } else {
                    for i in 0..n {
                        let row_in = self.session.h.view_subrange((i * h) as u64, vec![h as u64]);
                        let row_out = dyn_attn
                            .view_subrange((i * conv_dyn_dim) as u64, vec![conv_dyn_dim as u64]);
                        encode_mat_vec_dispatch(
                            ctx_metal,
                            &enc,
                            &cv.attn_proj,
                            &row_in,
                            &row_out,
                            h,
                            conv_dyn_dim,
                        )?;
                    }
                }
                encode_dflash2_conv_f32(
                    ctx_metal,
                    &enc,
                    &self.session.h,
                    dyn_attn,
                    &cv.attn_base,
                    conv_buf,
                    n,
                    h,
                    conv_kernel,
                    conv_group,
                    0,
                )?;
                conv_buf
            } else {
                &self.session.h
            };
            if batched_proj {
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &layer.q,
                    attn_qkv_src,
                    &self.session.q_buf,
                    h,
                    q_dim,
                    n,
                )?;
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &layer.k,
                    attn_qkv_src,
                    &self.session.k_noise,
                    h,
                    kv_dim,
                    n,
                )?;
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &layer.v,
                    attn_qkv_src,
                    &self.session.v_noise,
                    h,
                    kv_dim,
                    n,
                )?;
            } else {
                // Q proj per noise row.
                for i in 0..n {
                    let row_in = attn_qkv_src.view_subrange((i * h) as u64, vec![h as u64]);
                    let row_out = self
                        .session
                        .q_buf
                        .view_subrange((i * q_dim) as u64, vec![q_dim as u64]);
                    encode_mat_vec_dispatch(
                        ctx_metal, &enc, &layer.q, &row_in, &row_out, h, q_dim,
                    )?;
                }
                // K, V proj on noise rows.
                for i in 0..n {
                    let row_in = attn_qkv_src.view_subrange((i * h) as u64, vec![h as u64]);
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
            }
            // v0.74.1: K, V proj on cross-context rows — ONLY the new
            // delta `[phase2_ctx_start, ctx_len)`. Cached rows
            // `[0, phase2_ctx_start)` retain their post-norm post-RoPE
            // values from prior outer steps. Write into per-layer cache.
            if batched_proj && phase2_ctx_delta > 1 {
                let row_in = self.session.ctx_h.view_subrange(
                    (phase2_ctx_start * h) as u64,
                    vec![(phase2_ctx_delta * h) as u64],
                );
                let k_rows = self.session.k_ctx_cache[layer_idx].view_subrange(
                    (phase2_ctx_start * kv_dim) as u64,
                    vec![(phase2_ctx_delta * kv_dim) as u64],
                );
                let v_rows = self.session.v_ctx_cache[layer_idx].view_subrange(
                    (phase2_ctx_start * kv_dim) as u64,
                    vec![(phase2_ctx_delta * kv_dim) as u64],
                );
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &layer.k,
                    &row_in,
                    &k_rows,
                    h,
                    kv_dim,
                    phase2_ctx_delta,
                )?;
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &layer.v,
                    &row_in,
                    &v_rows,
                    h,
                    kv_dim,
                    phase2_ctx_delta,
                )?;
            } else {
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
            if batched_proj {
                encode_prefill_qk_rope(
                    ctx_metal,
                    &enc,
                    &self.session.q_buf,
                    &self.session.k_noise,
                    n,
                    n_q,
                    n_kv,
                    head_dim,
                    n_rot,
                    noise_start_pos,
                    theta,
                )?;
            } else {
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
            }
            // v0.74.1: RoPE K_ctx — only the new delta. Pre-cached
            // rows were RoPE'd on a prior outer step at their stable
            // pos_ctx[c] positions; positions don't change.
            let ctx_positions_are_consecutive = phase2_ctx_delta > 0
                && pos_ctx_cpu[phase2_ctx_start] >= 0
                && (0..phase2_ctx_delta).all(|i| {
                    pos_ctx_cpu[phase2_ctx_start + i] == pos_ctx_cpu[phase2_ctx_start] + i as i32
                });
            if batched_proj && ctx_positions_are_consecutive {
                let row = self.session.k_ctx_cache[layer_idx].view_subrange(
                    (phase2_ctx_start * kv_dim) as u64,
                    vec![(phase2_ctx_delta * kv_dim) as u64],
                );
                encode_rope_neox_f32_packed_consecutive(
                    ctx_metal,
                    &enc,
                    &row,
                    phase2_ctx_delta,
                    n_kv,
                    head_dim,
                    n_rot,
                    pos_ctx_cpu[phase2_ctx_start] as u32,
                    theta,
                )?;
            } else {
                for c in phase2_ctx_start..ctx_len {
                    let pos = pos_ctx_cpu[c] as u32;
                    let row = self.session.k_ctx_cache[layer_idx]
                        .view_subrange((c * kv_dim) as u64, vec![kv_dim as u64]);
                    encode_rope_neox_f32(ctx_metal, &enc, &row, n_kv, head_dim, n_rot, pos, theta)?;
                }
            }
            enc.end();
            if shared_cmd.is_none() {
                cmd.commit();
                cmd.waitUntilCompleted();
                self.session.maybe_record("phase2_proj_norm_rope", &cmd);
            }

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
            // Drafter projection/FFN weights stay native where the GGUF dtype
            // is supported; the dispatchers route Q8_0 and K-quants directly.
            let swa_window_arg = if layer.is_swa { cfg.swa_window } else { 0 };
            let ctx_scan_start = if layer.is_swa && dflash_attn_swa_scan_enabled() {
                dflash_swa_ctx_scan_start(&pos_ctx_cpu, ctx_len, noise_start_pos, swa_window_arg)
            } else {
                0
            };
            let exact_visible_suffix = dflash_swa_exact_visible_suffix(
                &pos_ctx_cpu,
                ctx_len,
                ctx_scan_start,
                noise_start_pos,
                swa_window_arg,
            );
            let full_gqa_split4_attn = dflash_attn_full_gqa_split4_enabled()
                && !layer.is_swa
                && n == 16
                && n_q == 32
                && n_kv == 8
                && head_dim == 128
                && (7986..=8241).contains(&ctx_len);
            // The retained measurement covers DFlash 2 at its saturated SWA
            // window. Keep short windows and the unmeasured N16 drafter on the
            // incumbent online path.
            let swa_split4_attn = dflash_attn_swa_split4_enabled()
                && !full_gqa_split4_attn
                && dflash_swa_split4_eligible(
                    layer.is_swa,
                    n,
                    n_q,
                    n_kv,
                    head_dim,
                    ctx_len,
                    ctx_scan_start,
                    exact_visible_suffix,
                    cfg.swa_window,
                    cfg.selector_top_k,
                );
            let online_two_range_attn = dflash_attn_online_two_range_enabled();
            let two_range_attn = full_gqa_split4_attn
                || swa_split4_attn
                || online_two_range_attn
                || dflash_attn_two_range_enabled();
            // pos_k staging hoisted to the top of draft_block (v0.77) —
            // the full ctx ‖ noise layout serves both the two-range
            // kernels (read the first ctx_len entries) and the legacy
            // concat kernel (reads all of it).

            let cmd = shared_cmd
                .clone()
                .unwrap_or_else(|| ctx_metal.queue.commandBuffer().expect("cmd phase3"));
            let phase3_attention_label = if layer.is_swa {
                "phase3_split_attention_swa"
            } else {
                "phase3_split_attention_full"
            };
            let mut phase3_split =
                if self.session.enable_phase_timers && dflash_trace_phase3_split_enabled() {
                    Some(DFlashPhase3SplitRecorder::new(ctx_metal, 7)?)
                } else {
                    None
                };
            let mut enc = if let Some(recorder) = phase3_split.as_mut() {
                recorder.begin(&cmd, phase3_attention_label)?
            } else {
                KernelEncoder::begin(&cmd)
            };

            // (a) Fused attention: writes attn_o_full [N, n_q*head_dim].
            let n_kv_total = ctx_len + n;
            if full_gqa_split4_attn {
                const O_PARTIAL_ELEMS: u64 = 16 * 8 * 4 * 4 * 128;
                const ML_PARTIAL_ELEMS: u64 = 16 * 8 * 4 * 4 * 2;
                let o_partial = self.session.k_full.view_subrange(0, vec![O_PARTIAL_ELEMS]);
                let ml_partial = self.session.v_full.view_subrange(0, vec![ML_PARTIAL_ELEMS]);
                encode_dflash_attn_full_gqa_split4_f32(
                    ctx_metal,
                    &enc,
                    &self.session.q_buf,
                    &self.session.k_ctx_cache[layer_idx],
                    &self.session.v_ctx_cache[layer_idx],
                    &self.session.k_noise,
                    &self.session.v_noise,
                    &o_partial,
                    &ml_partial,
                    &self.session.attn_o_full,
                    n,
                    n_q,
                    n_kv,
                    head_dim,
                    ctx_len,
                )?;
            } else if swa_split4_attn {
                // Same borrowed partials as the split4 path (k_full/v_full
                // are unused when a two-range kernel runs). Size the views
                // from the active block rather than assuming N16 capacity.
                let partial_groups = n as u64 * 8 * 4 * 4;
                let o_partial = self
                    .session
                    .k_full
                    .view_subrange(0, vec![partial_groups * 128]);
                let ml_partial = self
                    .session
                    .v_full
                    .view_subrange(0, vec![partial_groups * 2]);
                let pos_ctx_view = self.session.pos_k.view_subrange(0, vec![ctx_len as u64]);
                crate::metal::encode_dflash_attn_swa_split4_f32(
                    ctx_metal,
                    &enc,
                    &self.session.q_buf,
                    &self.session.k_ctx_cache[layer_idx],
                    &self.session.v_ctx_cache[layer_idx],
                    &self.session.k_noise,
                    &self.session.v_noise,
                    &pos_ctx_view,
                    &o_partial,
                    &ml_partial,
                    &self.session.attn_o_full,
                    n,
                    ctx_len,
                    noise_start_pos,
                    swa_window_arg,
                    ctx_scan_start,
                )?;
            } else if online_two_range_attn {
                let pos_ctx_view = self.session.pos_k.view_subrange(0, vec![ctx_len as u64]);
                encode_dflash_attn_online_two_range_scan_f32(
                    ctx_metal,
                    &enc,
                    &self.session.q_buf,
                    &self.session.k_ctx_cache[layer_idx],
                    &self.session.v_ctx_cache[layer_idx],
                    &self.session.k_noise,
                    &self.session.v_noise,
                    &pos_ctx_view,
                    &self.session.attn_o_full,
                    n,
                    n_q,
                    n_kv,
                    head_dim,
                    ctx_len,
                    noise_start_pos,
                    swa_window_arg,
                    ctx_scan_start,
                )?;
            } else if two_range_attn {
                let pos_ctx_view = self.session.pos_k.view_subrange(0, vec![ctx_len as u64]);
                encode_dflash_attn_two_range_f32(
                    ctx_metal,
                    &enc,
                    &self.session.q_buf,
                    &self.session.k_ctx_cache[layer_idx],
                    &self.session.v_ctx_cache[layer_idx],
                    &self.session.k_noise,
                    &self.session.v_noise,
                    &pos_ctx_view,
                    &self.session.attn_o_full,
                    n,
                    n_q,
                    n_kv,
                    head_dim,
                    ctx_len,
                    noise_start_pos,
                    swa_window_arg,
                )?;
            } else {
                // Concat K_ctx + K_noise into k_full; same for V.
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

                let k_view = self
                    .session
                    .k_full
                    .view_subrange(0, vec![(n_kv_total * kv_dim) as u64]);
                let v_view = self
                    .session
                    .v_full
                    .view_subrange(0, vec![(n_kv_total * kv_dim) as u64]);
                let pos_view = self.session.pos_k.view_subrange(0, vec![n_kv_total as u64]);
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
            }

            if let Some(recorder) = phase3_split.as_mut() {
                enc.end();
                enc = recorder.begin(&cmd, "phase3_split_o_proj")?;
            }

            // (b) O proj — v0.74.2: batched mat-mat across all N noise
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
            let drafter_mat_mat_eligible = prefill_mat_mat_dispatch_eligible;
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

            if let Some(recorder) = phase3_split.as_mut() {
                enc.end();
                enc = recorder.begin(&cmd, "phase3_split_resid1_post_norm")?;
            }

            // (e) Residual #1: x += O-proj output (DFlash 2: the attention
            //     side-1 conv sits between the O proj and the residual).
            if let Some(cv) = layer.conv.as_ref() {
                let dyn_attn = self
                    .session
                    .conv_dyn_attn
                    .as_ref()
                    .expect("dflash2 dyn_attn");
                let conv_buf = self.session.conv_buf.as_ref().expect("dflash2 conv_buf");
                encode_dflash2_conv_f32(
                    ctx_metal,
                    &enc,
                    &self.session.ffn_out_buf,
                    dyn_attn,
                    &cv.attn_base,
                    conv_buf,
                    n,
                    h,
                    conv_kernel,
                    conv_group,
                    1,
                )?;
                encode_add_inplace_f32(ctx_metal, &enc, &self.session.x, conv_buf)?;
            } else {
                encode_add_inplace_f32(
                    ctx_metal,
                    &enc,
                    &self.session.x,
                    &self.session.ffn_out_buf,
                )?;
            }

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
            // DFlash 2 FFN conv: dynamic coefficients from the pre-conv
            // normed input, then side-0. gate/up read the conv'd result.
            let ffn_src: &MetalTensor = if let Some(cv) = layer.conv.as_ref() {
                let dyn_ffn = self.session.conv_dyn_ffn.as_ref().expect("dflash2 dyn_ffn");
                let conv_buf = self.session.conv_buf.as_ref().expect("dflash2 conv_buf");
                if phase3_batched {
                    encode_mat_mat_dispatch(
                        ctx_metal,
                        &enc,
                        &cv.ffn_proj,
                        &self.session.h,
                        dyn_ffn,
                        h,
                        conv_dyn_dim,
                        n,
                    )?;
                } else {
                    for i in 0..n {
                        let row_in = self.session.h.view_subrange((i * h) as u64, vec![h as u64]);
                        let row_out = dyn_ffn
                            .view_subrange((i * conv_dyn_dim) as u64, vec![conv_dyn_dim as u64]);
                        encode_mat_vec_dispatch(
                            ctx_metal,
                            &enc,
                            &cv.ffn_proj,
                            &row_in,
                            &row_out,
                            h,
                            conv_dyn_dim,
                        )?;
                    }
                }
                encode_dflash2_conv_f32(
                    ctx_metal,
                    &enc,
                    &self.session.h,
                    dyn_ffn,
                    &cv.ffn_base,
                    conv_buf,
                    n,
                    h,
                    conv_kernel,
                    conv_group,
                    0,
                )?;
                conv_buf
            } else {
                &self.session.h
            };

            if let Some(recorder) = phase3_split.as_mut() {
                enc.end();
                enc = recorder.begin(&cmd, "phase3_split_gate_up")?;
            }

            // (g) SwiGLU FFN — v0.74.2: batched mat-mat ffn_gate / ffn_up
            //     / ffn_down when drafter weights are eligible. Same
            //     fall-through pattern as O-proj.
            if phase3_batched {
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &layer.ffn_gate,
                    ffn_src,
                    &self.session.ffn_gate_buf,
                    h,
                    f,
                    n,
                )?;
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &layer.ffn_up,
                    ffn_src,
                    &self.session.ffn_up_buf,
                    h,
                    f,
                    n,
                )?;
            } else {
                for i in 0..n {
                    let row_in = ffn_src.view_subrange((i * h) as u64, vec![h as u64]);
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

            if let Some(recorder) = phase3_split.as_mut() {
                enc.end();
                enc = recorder.begin(&cmd, "phase3_split_silu_mul")?;
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
            if let Some(recorder) = phase3_split.as_mut() {
                enc.end();
                enc = recorder.begin(&cmd, "phase3_split_down")?;
            }
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
            if let Some(recorder) = phase3_split.as_mut() {
                enc.end();
                enc = recorder.begin(&cmd, "phase3_split_resid2")?;
            }
            // (h) Residual #2: x += FFN output (DFlash 2: the FFN side-1
            //     conv sits between ffn_down and the residual).
            if let Some(cv) = layer.conv.as_ref() {
                let dyn_ffn = self.session.conv_dyn_ffn.as_ref().expect("dflash2 dyn_ffn");
                let conv_buf = self.session.conv_buf.as_ref().expect("dflash2 conv_buf");
                encode_dflash2_conv_f32(
                    ctx_metal,
                    &enc,
                    &self.session.ffn_out_buf,
                    dyn_ffn,
                    &cv.ffn_base,
                    conv_buf,
                    n,
                    h,
                    conv_kernel,
                    conv_group,
                    1,
                )?;
                encode_add_inplace_f32(ctx_metal, &enc, &self.session.x, conv_buf)?;
            } else {
                encode_add_inplace_f32(
                    ctx_metal,
                    &enc,
                    &self.session.x,
                    &self.session.ffn_out_buf,
                )?;
            }

            enc.end();
            if shared_cmd.is_none() {
                cmd.commit();
                cmd.waitUntilCompleted();
                let phase3_gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                self.session
                    .maybe_record("phase3_attn_oproj_ffn_residuals", &cmd);
                if let Some(recorder) = phase3_split {
                    recorder.record(ctx_metal, &mut self.session, phase3_gpu_ms)?;
                }
            }
        }

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
        let cmd = shared_cmd
            .clone()
            .unwrap_or_else(|| ctx_metal.queue.commandBuffer().expect("cmd"));
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
        let lm_mat_mat_path = prefill_mat_mat_dispatch_eligible(lm_dtype);
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
        if let Some(sel) = self.head.selector.as_ref() {
            // DFlash 2 tail: top-16 candidates + logits per position, and
            // the selector context gate `W_h · h`. The greedy path walk
            // happens on the CPU after the wait (≤20 KB readback).
            encode_topk16_f32(
                ctx_metal,
                &enc,
                &self.session.draft_logits,
                self.session.topk_ids.as_ref().expect("dflash2 topk_ids"),
                self.session.topk_vals.as_ref().expect("dflash2 topk_vals"),
                n,
                v,
            )?;
            {
                #[cfg(feature = "dflash-k0s-diagnostics")]
                let _selector_dispatch_tag =
                    dispatch_census_tag_scope(|| DFLASH_K0S_SELECTOR_DISPATCH_TAG.to_owned());
                encode_mat_mat_dispatch(
                    ctx_metal,
                    &enc,
                    &sel.hidden,
                    &self.session.h,
                    self.session.sel_h.as_ref().expect("dflash2 sel_h"),
                    h,
                    sel.rank,
                    n,
                )?;
            }
        } else {
            encode_argmax_f32(
                ctx_metal,
                &enc,
                &self.session.draft_logits,
                &self.session.draft_argmax,
                n,
                v,
            )?;
        }
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        self.session
            .maybe_record("phase4_tail_norm_lmhead_argmax", &cmd);

        // Watermarks advance only after ALL GPU work is complete — in
        // single-cmd mode nothing executed before the commit above, and
        // in multi-buffer mode every earlier phase already waited. An
        // early error return leaves the watermarks untouched, so a
        // retried call re-projects the delta instead of trusting
        // never-executed work.
        self.session.ctx_h_ready_n = ctx_len;
        self.session.kv_ctx_ready_n = ctx_len;

        if let Some(sel) = self.head.selector.as_ref() {
            return self.select_draft_path(sel, carry_tok, n);
        }

        // Read back `[N]` i32 argmaxes (64 B, vs the v0.71 per-token
        // `[V]` F32 readback = 16 MB/outer step at V=248320, N=16).
        let mut argmaxes = vec![0i32; n];
        unsafe {
            let src = self.session.draft_argmax.buffer.contents().as_ptr() as *const i32;
            std::ptr::copy_nonoverlapping(src, argmaxes.as_mut_ptr(), n);
        }
        Ok(argmaxes)
    }

    /// DFlash 2 greedy path selection — the CPU tail of `draft_block`.
    ///
    /// Mirrors llama.cpp PR 27342's lattice walk at T=0: starting from the
    /// carry token (block position 0), at each draft position pick the
    /// candidate `b` maximizing `U(b) + ⟨A(prev) ⊙ (W_h·h_pos), B(b)⟩`,
    /// where `U` is the drafter's own logit, `A`/`B` are the predecessor/
    /// successor codebook embeddings, and `h_pos` is the position's final
    /// (post-`output_norm`) hidden. Returns `[N]` tokens laid out like the
    /// DFlash 1 argmax vector: slot 0 is the anchor position's top-1
    /// (unused by the driver), slots 1..N are the drafts.
    fn select_draft_path(
        &self,
        sel: &MetalDFlash2Selector,
        carry_tok: i32,
        n: usize,
    ) -> Result<Vec<i32>, DFlashError> {
        let top_k = sel.top_k;
        let rank = sel.rank;
        let ids_t = self.session.topk_ids.as_ref().expect("dflash2 topk_ids");
        let vals_t = self.session.topk_vals.as_ref().expect("dflash2 topk_vals");
        let sel_h_t = self.session.sel_h.as_ref().expect("dflash2 sel_h");
        let mut ids = vec![0i32; n * top_k];
        let mut vals = vec![0f32; n * top_k];
        let mut sel_h = vec![0f32; n * rank];
        unsafe {
            std::ptr::copy_nonoverlapping(
                ids_t.buffer.contents().as_ptr() as *const i32,
                ids.as_mut_ptr(),
                ids.len(),
            );
            std::ptr::copy_nonoverlapping(
                vals_t.buffer.contents().as_ptr() as *const f32,
                vals.as_mut_ptr(),
                vals.len(),
            );
            std::ptr::copy_nonoverlapping(
                sel_h_t.buffer.contents().as_ptr() as *const f32,
                sel_h.as_mut_ptr(),
                sel_h.len(),
            );
        }

        let mut out = vec![0i32; n];
        out[0] = ids[0]; // anchor top-1; the driver never reads slot 0
        if dflash2_selector_disabled() {
            // Ablation: per-position top-1, no path selection.
            for pos in 1..n {
                out[pos] = ids[pos * top_k];
            }
            return Ok(out);
        }
        let mut pred = vec![0f32; rank];
        let mut succ = vec![0f32; rank];
        let mut gate = vec![0f32; rank];
        sel.predecessor.dequant_row(carry_tok as usize, &mut pred)?;
        for pos in 1..n {
            let hrow = &sel_h[pos * rank..(pos + 1) * rank];
            for r in 0..rank {
                gate[r] = pred[r] * hrow[r];
            }
            let mut best_b = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for b in 0..top_k {
                let cand = ids[pos * top_k + b];
                if cand < 0 || (cand as usize) >= sel.successor.n_rows {
                    continue; // unfilled top-k sentinel
                }
                sel.successor.dequant_row(cand as usize, &mut succ)?;
                let mut dot = 0f32;
                for r in 0..rank {
                    dot += gate[r] * succ[r];
                }
                // Strict `>` keeps the FIRST (highest-unary-rank) candidate
                // on ties — matches std::max_element in the reference.
                let score = vals[pos * top_k + b] + dot;
                if score > best_score {
                    best_score = score;
                    best_b = b;
                }
            }
            let chosen = ids[pos * top_k + best_b];
            out[pos] = chosen;
            sel.predecessor.dequant_row(chosen as usize, &mut pred)?;
        }
        Ok(out)
    }

    /// Run DFlash 2 and sample its predecessor-conditioned sparse selector
    /// distribution at each draft position. The proposal sampler must apply
    /// only positive temperature; target-side filters belong to verification.
    pub fn draft_block_sampled(
        &mut self,
        carry_tok: i32,
        noise_start_pos: u32,
        proposal_sampler: &mut Sampler,
    ) -> Result<DFlash2SparseProposalBlock, DFlashError> {
        if self.head.selector.is_none() {
            return Err(DFlashError::BadDrafter(
                "sampled path requires a DFlash 2 selector",
            ));
        }
        let proposal_config = proposal_sampler.config();
        if proposal_config.temperature <= 0.0
            || proposal_config.top_k != 0
            || proposal_config.top_p != 1.0
            || proposal_config.min_p != 0.0
        {
            return Err(DFlashError::BadDrafter(
                "DFlash 2 proposal sampler must use temperature without target filters",
            ));
        }
        if dflash2_selector_disabled() {
            return Err(DFlashError::SelectorDiagnosticDisabled);
        }

        let _greedy_path = self.draft_block(carry_tok, noise_start_pos)?;
        let selector = self.head.selector.as_ref().expect("selector checked above");
        let n = self.head.config.block_size as usize;
        let top_k = selector.top_k;
        let rank = selector.rank;
        let candidate_elements = n.checked_mul(top_k).ok_or_else(|| {
            DFlashError::Metal(MetalError::BadShape {
                kernel: "dflash2_sampled_selector_read",
                detail: "candidate element count overflow".into(),
            })
        })?;
        let gate_elements = n.checked_mul(rank).ok_or_else(|| {
            DFlashError::Metal(MetalError::BadShape {
                kernel: "dflash2_sampled_selector_read",
                detail: "gate element count overflow".into(),
            })
        })?;
        let ids = read_shared_selector_tensor::<i32>(
            self.session
                .topk_ids
                .as_ref()
                .ok_or(DFlashError::BadDrafter(
                    "sampled selector missing topk_ids buffer",
                ))?,
            GgmlType::I32,
            candidate_elements,
            "topk_ids",
        )?;
        let vals = read_shared_selector_tensor::<f32>(
            self.session
                .topk_vals
                .as_ref()
                .ok_or(DFlashError::BadDrafter(
                    "sampled selector missing topk_vals buffer",
                ))?,
            GgmlType::F32,
            candidate_elements,
            "topk_vals",
        )?;
        let selector_hidden = read_shared_selector_tensor::<f32>(
            self.session.sel_h.as_ref().ok_or(DFlashError::BadDrafter(
                "sampled selector missing sel_h buffer",
            ))?,
            GgmlType::F32,
            gate_elements,
            "sel_h",
        )?;

        let mut draft_tokens = vec![ids[0]; n];
        let mut proposals = Vec::with_capacity(n.saturating_sub(1));
        let mut predecessor = vec![0.0f32; rank];
        let mut successor = vec![0.0f32; rank];
        let mut gate = vec![0.0f32; rank];
        selector
            .predecessor
            .dequant_row(carry_tok as usize, &mut predecessor)?;

        for position in 1..n {
            let hidden_row = &selector_hidden[position * rank..(position + 1) * rank];
            for r in 0..rank {
                gate[r] = predecessor[r] * hidden_row[r];
            }

            let candidate_ids = &ids[position * top_k..(position + 1) * top_k];
            let unary_logits = &vals[position * top_k..(position + 1) * top_k];
            let mut scores = vec![f32::NEG_INFINITY; top_k];
            for (candidate_index, &token) in candidate_ids.iter().enumerate() {
                if token < 0 || token as usize >= selector.successor.n_rows {
                    continue;
                }
                selector
                    .successor
                    .dequant_row(token as usize, &mut successor)?;
                let mut dot = 0.0f32;
                for r in 0..rank {
                    dot += gate[r] * successor[r];
                }
                scores[candidate_index] = unary_logits[candidate_index] + dot;
            }

            let distribution = proposal_sampler.sample_with_distribution(&scores)?;
            let selected_index = usize::try_from(distribution.sampled.token).map_err(|_| {
                DFlashError::BadDrafter("sampled selector returned a negative candidate index")
            })?;
            let selected_token =
                *candidate_ids
                    .get(selected_index)
                    .ok_or(DFlashError::BadDrafter(
                        "sampled selector candidate index is out of range",
                    ))?;
            if selected_token < 0 || selected_token as usize >= selector.predecessor.n_rows {
                return Err(DFlashError::BadDrafter(
                    "sampled selector returned an invalid token",
                ));
            }

            let candidates = distribution
                .candidates
                .into_iter()
                .filter(|candidate| candidate.weight > 0.0)
                .map(|candidate| {
                    let index = candidate.token as usize;
                    WeightedCandidate {
                        token: candidate_ids[index],
                        weight: candidate.weight,
                    }
                })
                .collect();
            proposals.push(SparseProposal {
                token: selected_token,
                candidates,
            });
            draft_tokens[position] = selected_token;
            selector
                .predecessor
                .dequant_row(selected_token as usize, &mut predecessor)?;
        }

        Ok(DFlash2SparseProposalBlock {
            draft_tokens,
            proposals,
        })
    }

    /// Run the production DFlash 2 draft path, then re-read its synchronized
    /// selector buffers and replay the scalar greedy walk as diagnostic
    /// evidence. This does not alter selector inputs or production choices.
    pub fn draft_block_with_selector_diagnostic(
        &mut self,
        carry_tok: i32,
        noise_start_pos: u32,
    ) -> Result<DFlash2SelectorDiagnostic, DFlashError> {
        if dflash2_selector_disabled() {
            return Err(DFlashError::SelectorDiagnosticDisabled);
        }
        if self.head.selector.is_none() {
            return Err(DFlashError::BadDrafter(
                "selector diagnostic requires a DFlash 2 selector",
            ));
        }
        let draft_tokens = self.draft_block(carry_tok, noise_start_pos)?;
        let sel = self.head.selector.as_ref().expect("selector checked above");
        let n = self.head.config.block_size as usize;
        let candidate_elements = n.checked_mul(sel.top_k).ok_or_else(|| {
            DFlashError::Metal(MetalError::BadShape {
                kernel: "dflash2_selector_diagnostic_read",
                detail: "candidate element count overflow".into(),
            })
        })?;
        let gate_elements = n.checked_mul(sel.rank).ok_or_else(|| {
            DFlashError::Metal(MetalError::BadShape {
                kernel: "dflash2_selector_diagnostic_read",
                detail: "gate element count overflow".into(),
            })
        })?;
        let ids = read_shared_selector_tensor::<i32>(
            self.session
                .topk_ids
                .as_ref()
                .ok_or(DFlashError::BadDrafter(
                    "selector diagnostic missing topk_ids buffer",
                ))?,
            GgmlType::I32,
            candidate_elements,
            "topk_ids",
        )?;
        let vals = read_shared_selector_tensor::<f32>(
            self.session
                .topk_vals
                .as_ref()
                .ok_or(DFlashError::BadDrafter(
                    "selector diagnostic missing topk_vals buffer",
                ))?,
            GgmlType::F32,
            candidate_elements,
            "topk_vals",
        )?;
        let sel_h = read_shared_selector_tensor::<f32>(
            self.session.sel_h.as_ref().ok_or(DFlashError::BadDrafter(
                "selector diagnostic missing sel_h buffer",
            ))?,
            GgmlType::F32,
            gate_elements,
            "sel_h",
        )?;
        let (reconstructed, depths) = diagnose_dflash2_selector_walk(
            &sel.predecessor,
            &sel.successor,
            sel.top_k,
            sel.rank,
            carry_tok,
            n,
            &ids,
            &vals,
            &sel_h,
        )?;
        verify_dflash2_selector_replay(&draft_tokens, &reconstructed)?;
        Ok(DFlash2SelectorDiagnostic {
            draft_tokens,
            depths,
        })
    }

    /// Hash every CPU-readable DFlash session tensor and relevant CPU state
    /// without encoding, committing, or waiting for Metal work. The domain is
    /// `qwen.dflash_k0s.state.v2`; integers and f64 timer bits are
    /// little-endian, strings/bytes have u64 lengths, and each tensor commits
    /// its deterministic label, GGML dtype, shape rank/dimensions, element
    /// count, byte count, and complete logical bytes. Vector tensors retain
    /// layer order and optional tensors commit presence before material.
    /// Callers must invoke this only after the production synchronization they
    /// intend to compare. Non-shared tensors fail closed.
    #[cfg(feature = "dflash-k0s-diagnostics")]
    pub fn dflash_k0s_diagnostic_state_sha256(&self) -> Result<[u8; 32], DFlashError> {
        dflash_k0s_state_sha256(&self.session)
    }

    #[cfg(feature = "dflash-k0s-diagnostics")]
    pub fn draft_block_with_k0s_diagnostic(
        &mut self,
        carry_tok: i32,
        noise_start_pos: u32,
    ) -> Result<DFlashK0sCapture, DFlashError> {
        let observation = self.draft_block_with_k0s_observation(carry_tok, noise_start_pos)?;
        self.extract_k0s_observation(observation)
    }

    #[cfg(feature = "dflash-k0s-diagnostics")]
    pub fn draft_block_with_k0s_observation(
        &mut self,
        carry_tok: i32,
        noise_start_pos: u32,
    ) -> Result<DFlashK0sProductionObservation, DFlashError> {
        dflash_k0s_positions(noise_start_pos)?;
        if dflash2_selector_disabled() {
            return Err(DFlashError::SelectorDiagnosticDisabled);
        }
        let (selector_top_k, selector_rank, predecessor_rows, successor_rows) = {
            let sel = self
                .head
                .selector
                .as_ref()
                .ok_or_else(|| dflash_k0s_error("K0-S requires a DFlash 2 selector"))?;
            (
                sel.top_k,
                sel.rank,
                sel.predecessor.n_rows,
                sel.successor.n_rows,
            )
        };
        let cfg = self.head.config;
        let vocab = self.base.model.arch.vocab_size as usize;
        if cfg.block_size as usize != DFLASH_K0S_BLOCK_SIZE
            || selector_top_k != DFLASH_K0S_TOP_K
            || selector_rank != DFLASH_K0S_RANK
            || cfg.hidden_size as usize != DFLASH_K0S_HIDDEN
            || vocab != DFLASH_K0S_VOCAB
            || predecessor_rows != DFLASH_K0S_VOCAB
            || successor_rows != DFLASH_K0S_VOCAB
        {
            return Err(dflash_k0s_error(format!(
                "required geometry is N=8,K=16,R=256,H=5120,V=248320; got N={},K={},R={},H={},V={},A={},B={}",
                cfg.block_size,
                selector_top_k,
                selector_rank,
                cfg.hidden_size,
                vocab,
                predecessor_rows,
                successor_rows
            )));
        }
        let (draft_tokens, dispatch_census, kernel_trace) =
            self.observe_k0s_production_draft(carry_tok, noise_start_pos)?;
        let parity = dflash_k0s_parity_summary_from_parts(
            self.head,
            &self.session,
            carry_tok,
            noise_start_pos,
            &draft_tokens,
            &dispatch_census,
            kernel_trace,
        )?;
        let event_sequence = dflash_k0s_next_event_sequence()?;
        self.k0s_live_event = Some(event_sequence);
        let event_envelope_sha256 = dflash_k0s_event_envelope_sha256(&parity, event_sequence);
        Ok(DFlashK0sProductionObservation {
            summary: DFlashK0sObservationSummary {
                parity,
                event_sequence,
                event_envelope_sha256,
            },
        })
    }

    #[cfg(feature = "dflash-k0s-diagnostics")]
    pub fn extract_k0s_observation(
        &mut self,
        observation: DFlashK0sProductionObservation,
    ) -> Result<DFlashK0sCapture, DFlashError> {
        dflash_k0s_consume_observation(
            self.head,
            &self.session,
            self.base.model.arch.vocab_size as usize,
            &mut self.k0s_live_event,
            observation,
        )
    }

    #[cfg(feature = "dflash-k0s-diagnostics")]
    pub fn finish_k0s_observation_without_extraction(
        &mut self,
        observation: DFlashK0sProductionObservation,
    ) -> Result<DFlashK0sObservationSummary, DFlashError> {
        dflash_k0s_finish_observation(
            self.head,
            &self.session,
            &mut self.k0s_live_event,
            observation,
        )
    }

    #[cfg(feature = "dflash-k0s-diagnostics")]
    pub fn dflash_k0s_parity_summary(
        &self,
        carry_token: i32,
        noise_start_position: u32,
        draft_tokens: &[i32],
        dispatch_census: &[DFlashK0sDispatchCensusRow],
        kernel_trace: crate::metal::KernelTraceCounters,
    ) -> Result<DFlashK0sParitySummary, DFlashError> {
        dflash_k0s_parity_summary_from_parts(
            self.head,
            &self.session,
            carry_token,
            noise_start_position,
            draft_tokens,
            dispatch_census,
            kernel_trace,
        )
    }

    #[cfg(feature = "dflash-k0s-diagnostics")]
    fn observe_k0s_production_draft(
        &mut self,
        carry_tok: i32,
        noise_start_pos: u32,
    ) -> Result<
        (
            Vec<i32>,
            Vec<DFlashK0sDispatchCensusRow>,
            crate::metal::KernelTraceCounters,
        ),
        DFlashError,
    > {
        let observer_guard = DFlashK0sObserverGuard::begin()?;
        let draft_result = self.draft_block(carry_tok, noise_start_pos);
        let (dispatch_rows, kernel_trace) = observer_guard.finish()?;
        dflash_k0s_check_dispatch_census_len(dispatch_rows.len())?;
        let dispatch_census: Vec<DFlashK0sDispatchCensusRow> = dispatch_rows
            .into_iter()
            .map(|row| DFlashK0sDispatchCensusRow {
                family: row.family.to_owned(),
                tag: row.tag,
                encoder_ordinal: row.encoder_ordinal,
                encoder_concurrent: row.encoder_concurrent,
                kernel: row.kernel,
                grid: [row.grid_width, row.grid_height, row.grid_depth],
                threads: [row.threads_width, row.threads_height, row.threads_depth],
                grid_threadgroups: row.grid_tgs,
                threadgroup_threads: row.tg_threads,
            })
            .collect();
        let draft_tokens = draft_result?;
        Ok((draft_tokens, dispatch_census, kernel_trace))
    }

    #[cfg(feature = "dflash-k0s-diagnostics")]
    fn extract_k0s_post_sync(
        head: &MetalDFlashHead,
        session: &MetalDFlashSession,
        vocab: usize,
        carry_tok: i32,
        noise_start_pos: u32,
        draft_tokens: Vec<i32>,
        dispatch_census: Vec<DFlashK0sDispatchCensusRow>,
        kernel_trace: crate::metal::KernelTraceCounters,
    ) -> Result<DFlashK0sCapture, DFlashError> {
        let depth_positions = dflash_k0s_positions(noise_start_pos)?;
        let sel = head
            .selector
            .as_ref()
            .ok_or_else(|| dflash_k0s_error("K0-S requires a DFlash 2 selector"))?;
        let cfg = head.config;
        if cfg.block_size as usize != DFLASH_K0S_BLOCK_SIZE
            || sel.top_k != DFLASH_K0S_TOP_K
            || sel.rank != DFLASH_K0S_RANK
            || cfg.hidden_size as usize != DFLASH_K0S_HIDDEN
            || vocab != DFLASH_K0S_VOCAB
            || sel.predecessor.n_rows != DFLASH_K0S_VOCAB
            || sel.successor.n_rows != DFLASH_K0S_VOCAB
        {
            return Err(dflash_k0s_error("malformed K0-S extraction geometry"));
        }
        dflash_k0s_check_dispatch_census_len(dispatch_census.len())?;
        let active_logits = (DFLASH_K0S_BLOCK_SIZE - 1)
            .checked_mul(DFLASH_K0S_VOCAB)
            .ok_or_else(|| dflash_k0s_error("active-logit geometry overflow"))?;
        let active_candidates = (DFLASH_K0S_BLOCK_SIZE - 1)
            .checked_mul(DFLASH_K0S_TOP_K)
            .ok_or_else(|| dflash_k0s_error("active-candidate geometry overflow"))?;
        let active_hidden = (DFLASH_K0S_BLOCK_SIZE - 1)
            .checked_mul(DFLASH_K0S_RANK)
            .ok_or_else(|| dflash_k0s_error("active-selector-hidden geometry overflow"))?;
        let tagged_dispatches: Vec<&DFlashK0sDispatchCensusRow> = dispatch_census
            .iter()
            .filter(|row| row.tag.as_deref() == Some(DFLASH_K0S_SELECTOR_DISPATCH_TAG))
            .collect();
        if tagged_dispatches.len() != 1 {
            return Err(dflash_k0s_error(format!(
                "expected exactly one tagged selector-hidden dispatch, observed {}",
                tagged_dispatches.len()
            )));
        }
        let selector_hidden_dispatch = (*tagged_dispatches[0]).clone();

        let logits_view = session
            .draft_logits
            .view_subrange(DFLASH_K0S_VOCAB as u64, vec![active_logits as u64]);
        let ids_view = session
            .topk_ids
            .as_ref()
            .ok_or_else(|| dflash_k0s_error("K0-S top-k IDs buffer is absent"))?
            .view_subrange(DFLASH_K0S_TOP_K as u64, vec![active_candidates as u64]);
        let unary_view = session
            .topk_vals
            .as_ref()
            .ok_or_else(|| dflash_k0s_error("K0-S unary buffer is absent"))?
            .view_subrange(DFLASH_K0S_TOP_K as u64, vec![active_candidates as u64]);
        let hidden_view = session
            .sel_h
            .as_ref()
            .ok_or_else(|| dflash_k0s_error("K0-S selector-hidden buffer is absent"))?
            .view_subrange(DFLASH_K0S_RANK as u64, vec![active_hidden as u64]);
        let active_logits_values = read_shared_selector_tensor::<f32>(
            &logits_view,
            GgmlType::F32,
            active_logits,
            "k0s_full_logits",
        )?;
        let active_ids = read_shared_selector_tensor::<i32>(
            &ids_view,
            GgmlType::I32,
            active_candidates,
            "k0s_topk_ids",
        )?;
        let active_unary = read_shared_selector_tensor::<f32>(
            &unary_view,
            GgmlType::F32,
            active_candidates,
            "k0s_unary",
        )?;
        let active_z = read_shared_selector_tensor::<f32>(
            &hidden_view,
            GgmlType::F32,
            active_hidden,
            "k0s_selector_hidden",
        )?;

        let mut depths = Vec::with_capacity(DFLASH_K0S_BLOCK_SIZE - 1);
        for depth in 1..DFLASH_K0S_BLOCK_SIZE {
            let active_depth = depth - 1;
            let logits = &active_logits_values
                [active_depth * DFLASH_K0S_VOCAB..(active_depth + 1) * DFLASH_K0S_VOCAB];
            let ids =
                &active_ids[active_depth * DFLASH_K0S_TOP_K..(active_depth + 1) * DFLASH_K0S_TOP_K];
            let unary = &active_unary
                [active_depth * DFLASH_K0S_TOP_K..(active_depth + 1) * DFLASH_K0S_TOP_K];
            let z = &active_z[active_depth * DFLASH_K0S_RANK..(active_depth + 1) * DFLASH_K0S_RANK];
            depths.push(DFlashK0sDepth {
                depth,
                position: depth_positions[active_depth],
                full_logits_bits: logits.iter().map(|value| value.to_bits()).collect(),
                top_k_ids: ids.to_vec(),
                unary_bits: unary.iter().map(|value| value.to_bits()).collect(),
                selector_hidden_bits: z.iter().map(|value| value.to_bits()).collect(),
                top_k_issues: dflash_k0s_reconstruct_top_k(logits, ids, unary),
            });
        }

        let total_candidates = DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_TOP_K;
        let total_hidden = DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_RANK;
        let mut ids = vec![0i32; total_candidates];
        let mut unary = vec![0.0f32; total_candidates];
        let mut z = vec![0.0f32; total_hidden];
        ids[DFLASH_K0S_TOP_K..].copy_from_slice(&active_ids);
        unary[DFLASH_K0S_TOP_K..].copy_from_slice(&active_unary);
        z[DFLASH_K0S_RANK..].copy_from_slice(&active_z);
        let (lattice, raw_rows) = dflash_k0s_build_lattice(
            &sel.predecessor,
            &sel.successor,
            carry_tok,
            &ids,
            &unary,
            &z,
        )?;

        let mut greedy_slots = Vec::with_capacity(DFLASH_K0S_BLOCK_SIZE - 1);
        let mut predecessor_slot = None;
        for depth in 1..DFLASH_K0S_BLOCK_SIZE {
            let row = lattice
                .iter()
                .find(|row| row.depth == depth && row.predecessor_slot == predecessor_slot)
                .ok_or_else(|| dflash_k0s_error("production chain predecessor row is absent"))?;
            greedy_slots.push(row.greedy_slot);
            predecessor_slot = Some(row.greedy_slot);
        }
        let production_chain = dflash_k0s_traverse_slots_mode(
            &lattice,
            carry_tok,
            &greedy_slots,
            DFlashK0sChainMode::Production,
        );
        dflash_k0s_verify_production_replay(&draft_tokens, &production_chain)?;

        let mut token_hasher = Sha256::new();
        for token in &draft_tokens {
            token_hasher.update(token.to_le_bytes());
        }
        let mut noise_hasher = Sha256::new();
        noise_hasher.update(carry_tok.to_le_bytes());
        for _ in 1..DFLASH_K0S_BLOCK_SIZE {
            noise_hasher.update(cfg.mask_token_id.to_le_bytes());
        }
        let noise_input_sha256 = noise_hasher.finalize().into();
        let mut event_hasher = Sha256::new();
        event_hasher.update(carry_tok.to_le_bytes());
        event_hasher.update(noise_start_pos.to_le_bytes());
        event_hasher.update((session.target_ctx_n as u64).to_le_bytes());
        event_hasher.update((session.ctx_h_ready_n as u64).to_le_bytes());
        event_hasher.update((session.kv_ctx_ready_n as u64).to_le_bytes());
        event_hasher.update(noise_input_sha256);
        for token_id in &active_ids {
            event_hasher.update(token_id.to_le_bytes());
        }
        for value in active_unary.iter().chain(&active_z) {
            event_hasher.update(value.to_bits().to_le_bytes());
        }
        let synchronized_event_sha256 = event_hasher.finalize().into();
        let diagnostic_state_sha256 = dflash_k0s_state_sha256(session)?;
        let draft_token_bits: Vec<u32> = draft_tokens.iter().map(|token| *token as u32).collect();
        let provenance = DFlashK0sProvenance {
            selector_hidden: sel.hidden_provenance.clone(),
            predecessor: DFlashK0sTensorProvenance {
                descriptor: sel.predecessor.original_desc.clone(),
                full_tensor_sha256: sel.predecessor.full_tensor_sha256,
            },
            successor: DFlashK0sTensorProvenance {
                descriptor: sel.successor.original_desc.clone(),
                full_tensor_sha256: sel.successor.full_tensor_sha256,
            },
            embedded_metallib_sha256: Sha256::digest(crate::KERNELS_METALLIB).into(),
        };
        let mut capture = DFlashK0sCapture {
            draft_tokens,
            draft_token_bits,
            depths,
            lattice,
            production_chain,
            raw_rows,
            provenance,
            dispatch_census,
            selector_hidden_dispatch,
            kernel_trace,
            state_identity: DFlashK0sStateIdentity {
                carry_token: carry_tok,
                noise_start_position: noise_start_pos,
                target_context_len: session.target_ctx_n,
                context_hidden_watermark: session.ctx_h_ready_n,
                kv_context_watermark: session.kv_ctx_ready_n,
                draft_tokens_sha256: token_hasher.finalize().into(),
                noise_input_sha256,
                synchronized_event_sha256,
                diagnostic_state_sha256,
            },
            capture_sha256: [0; 32],
            content_sha256: [0; 32],
        };
        capture.content_sha256 = dflash_k0s_capture_content_sha256(&capture);
        capture.capture_sha256 = dflash_k0s_capture_sha256(&capture);
        Ok(capture)
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
const ATTN_PREFILL_V4_PACKED_ROWS: usize = 8;

// H5.1.5 metal_drafter_cosine_vs_cpu moved to tests/dflash_correctness.rs
// (slow: ~142s on 27B-Q4_K_M prefill + drafter forward; not a fast-
// feedback gate). Run with `cargo test --test dflash_correctness --release`.

#[cfg(test)]
mod tests;
