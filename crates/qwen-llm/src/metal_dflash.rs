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
    encode_argmax_f32, encode_axpy_rowwise_f32, encode_copy_offset_f32, encode_dflash_attn_f32,
    encode_fill_f32, encode_gdn_decay_chain_batched_f32, encode_gdn_decay_chain_f32,
    encode_gdn_prep_packed_f32, encode_gdn_step_decay_packed_f32, encode_get_rows_f32,
    encode_l2_norm_batched_f32, encode_l2_norm_pair_batched_f32, encode_mat_mat_f32_router_e8p32,
    encode_moe_down_iq4_xs_f32, encode_moe_down_q5_K_f32,
    encode_moe_down_weighted_sum_q5_K_f32_packed_slots, encode_moe_mat_vec_f32,
    encode_moe_swiglu_q4_K_f32_packed_slots, encode_moe_weighted_sum_f32,
    encode_rms_norm_batched_f32, encode_rms_norm_mul_f32, encode_rmsnorm_gated_f32,
    encode_rope_neox_f32, encode_rope_neox_f32_packed_consecutive,
    encode_scatter_offset_f32_to_f16_kv, encode_scatter_offset_f32_to_f16_kv_vt,
    encode_sigmoid_f32, encode_silu_mul_f32, encode_split_q_gate_f32, encode_split_qkv_fused_f32,
    encode_topk_logits_softmax_dot_sigmoid_packed_f32,
};
use crate::metal_forward::{
    ATTN_V4_MAX_NWG, MetalBlock, MetalForward, MetalMoeFfn, MetalSession, RMS_EPS, checked_u64_add,
    checked_u64_double, checked_u64_mul, checked_u64_mul3, checked_u64_mul4,
    encode_mat_mat_dispatch, encode_mat_vec_dispatch, encode_scatter_offset_f32,
    weight_dtype_kept_native,
};
use crate::tensor::{GgmlType, TensorDesc};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use std::cell::Cell;
use std::sync::OnceLock;
use std::time::Instant;

fn env_flag_enabled(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn dense_packed_gdn_step_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_DENSE_GDN_STEP_PACKED").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn prefill_gdn_batched_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_PREFILL_GDN_BATCHED").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

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

fn prefill_noop_ffn_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_NOOP_FFN"))
}

thread_local! {
    static PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4_OVERRIDE: Cell<Option<bool>> = Cell::new(None);
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
    hidden <= 1536
}

fn prefill_mat_mat_dispatch_eligible(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q2_K
            | GgmlType::Q3_K
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

fn prefill_gdn_skinny_f32_e8p32_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_PREFILL_GDN_SKINNY_E8P32").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn prefill_gdn_pair_l2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_PREFILL_GDN_PAIR_L2").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

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
    layers.as_ref().map_or(true, |layers| {
        layers.contains(&usize::MAX) || layers.contains(&layer_idx)
    })
}

fn prefill_gdn_matvec_projection_enabled(proj: &str, layer_idx: usize) -> bool {
    prefill_gdn_matvec_proj_enabled(proj) && prefill_gdn_matvec_layer_enabled(layer_idx)
}

fn prefill_moe_packed_routed_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_PREFILL_MOE_PACKED_ROUTED").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn prefill_moe_packed_route_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_PREFILL_MOE_PACKED_ROUTE").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn prefill_moe_packed_down_sum_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_PREFILL_MOE_PACKED_DOWN_SUM").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn prefill_moe_packed_shared_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_PREFILL_MOE_PACKED_SHARED").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn prefill_noop_moe_shared_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_NOOP_MOE_SHARED"))
}

fn prefill_noop_moe_routed_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_NOOP_MOE_ROUTED"))
}

fn prefill_moe_grouped_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_PREFILL_MOE_GROUPED").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

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

fn prefill_moe_grouped_iq3_gateup_enabled() -> bool {
    static MODE: OnceLock<PrefillEnvMode> = OnceLock::new();
    match *MODE.get_or_init(|| env_mode("QWEN_PREFILL_MOE_GROUPED_IQ3_GATEUP")) {
        PrefillEnvMode::ForceOn => true,
        PrefillEnvMode::ForceOff | PrefillEnvMode::Auto => false,
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

fn prefill_moe_fused_finalizer_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_MOE_FUSED_FINALIZER"))
}

fn prefill_moe_grouped_zero_fill_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_MOE_GROUPED_ZERO_FILL"))
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
        other => Err(MetalError::BadShape {
            kernel: "prefill_moe_grouped_swiglu",
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

fn prefill_trace_labels_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_TRACE_LABELS"))
}

fn prefill_trace_chunks_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_TRACE_CHUNKS"))
}

fn prefill_trace_attn_phases_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_TRACE_ATTN_PHASES"))
}

fn prefill_trace_layer_phases_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_TRACE_LAYER_PHASES"))
}

fn prefill_trace_ffn_subphases_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_TRACE_FFN_SUBPHASES"))
}

fn prefill_trace_moe_buckets_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_TRACE_MOE_BUCKETS"))
}

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
    let (hot_min, hot_experts, hot_slots) = if let Some(min_slots) = hot_expert_min_slots {
        let hot_experts = active.iter().filter(|&&c| c >= min_slots).count();
        let hot_slots = active.iter().filter(|&&c| c >= min_slots).sum::<usize>();
        (min_slots as isize, hot_experts, hot_slots)
    } else {
        (-1, 0, 0)
    };
    eprintln!(
        "[prefill-moe-buckets] chunk={chunk_idx} start={chunk_start} layer={layer_idx} total={total}/{} active={} p50={p50} p90={p90} max={max} ge16={ge16} ge32={ge32} ge48={ge48} hot_min={hot_min} hot_experts={hot_experts} hot_slots={hot_slots}",
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
) {
    if !enabled {
        return;
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    let gpu_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;
    *prefill_gpu_total_ms += gpu_ms;
    eprintln!(
        "[prefill-attn-phase] chunk={} start={} layer={} phase={} gpu_ms={:.2}",
        chunk_idx, chunk_start, layer_idx, phase, gpu_ms
    );
    *cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
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
) {
    if !enabled {
        return;
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    let gpu_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;
    *prefill_gpu_total_ms += gpu_ms;
    eprintln!(
        "[prefill-layer-phase] chunk={} start={} layer={} kind={} phase={} gpu_ms={:.2}",
        chunk_idx, chunk_start, layer_idx, kind, phase, gpu_ms
    );
    *cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
}

fn flush_prefill_layer_phase_accum(
    ctx: &MetalContext,
    cmd_buf: &mut Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    prefill_gpu_total_ms: &mut f64,
) -> f64 {
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    let gpu_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;
    *prefill_gpu_total_ms += gpu_ms;
    *cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    gpu_ms
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
    let gpu_ms = flush_prefill_layer_phase_accum(ctx, cmd_buf, prefill_gpu_total_ms);

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

fn prefill_attn_packed_g8_oracle_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_ATTN_PACKED_G8_ORACLE"))
}

fn prefill_attn_packed_g16_oracle_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_ATTN_PACKED_G16_ORACLE"))
}

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
    if enabled && weight.dtype == GgmlType::F32 && n_in % 4 == 0 && n_out % 8 == 0 && n_query >= 32
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

fn prefill_noop_gdn_body_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_NOOP_GDN_BODY"))
}

fn prefill_noop_attn_body_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("QWEN_PREFILL_NOOP_ATTN_BODY"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillGdnSplitMode {
    None,
    SkipAll,
    OutOnly,
    PrepOut,
    PrepStepOut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillTailMode {
    ReadLogits,
    SkipTail,
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
        let k_layers = head.target_layer_ids.len() as u64;
        let n_target_features =
            checked_u64_mul(k_layers, target_h, "dflash n_target_features overflow")?;
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
            noise_ids: MetalTensor::zeros_f32(ctx, vec![n])?,
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
            draft_argmax: MetalTensor::zeros_f32(ctx, vec![n])?,
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
    /// `[N, 2·q_dim + 2·kv_dim]` F32 — fused QKV front projection for the
    /// experimental group-8 packed-prefill path.
    pub attn_qkv_fused_pack: MetalTensor,
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
    /// `[R, n_kv, NWG, group, head_dim]` F32 — packed-prompt attention partials
    /// for the experimental A3B/group-8 prompt-native microproof.
    pub attn_prefill_v4_o_partial_pack: MetalTensor,
    /// `[R, n_kv, NWG, group, 2]` F32 — packed-prompt attention `(m, l)`
    /// partials for the same microproof.
    pub attn_prefill_v4_ml_partial_pack: MetalTensor,
    /// `[N * n_q_heads, matrix_max_pos]` F32 — experimental non-flash matrix
    /// attention score/prob scratch for the A3B/group-8 sidecar.
    pub attn_matrix_scores_pack: MetalTensor,
    /// `[n_attn_layers, n_kv_heads, head_dim, matrix_max_pos]` F16 — persistent
    /// transposed V-cache view used by the experimental non-flash matrix
    /// attention sidecar.
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
    pub moe_inner_pack: MetalTensor,
    /// `[N * topk, H]` F32 — packed routed expert outputs before tokenwise reduction.
    pub moe_expert_out_pack: MetalTensor,
    /// `[N * topk]` i32-in-F32 buffer — grouped routed slot ids.
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
    fn fresh_inner(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
        include_final_logits_pack: bool,
        attn_matrix_max_pos_override: Option<usize>,
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
        let attn_prefill_v4_o_partial_elems = checked_u64_mul4(
            ATTN_PREFILL_V4_PACKED_ROWS as u64,
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
            ATTN_PREFILL_V4_PACKED_ROWS as u64,
            (arch.n_kv_heads as u64).max(1),
            ATTN_V4_MAX_NWG as u64,
            checked_u64_mul(
                attn_group,
                2,
                "layer-major attn prefill ml partial group*2 overflow",
            )?,
            "layer-major attn prefill ml partial size overflow",
        )?;
        let attn_group = attn_group as usize;
        let enable_attn_packed = matches!(
            std::env::var("QWEN_PREFILL_ATTN_PACKED_G8").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ) || matches!(
            std::env::var("QWEN_PREFILL_ATTN_PACKED_G16").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ) || (head_dim as usize == 256 && matches!(attn_group, 8 | 16));
        let enable_attn_fused_qkv_g8 = matches!(
            std::env::var("QWEN_PREFILL_ATTN_FUSED_QKV_G8").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        );
        let enable_attn_matrix = head_dim as usize == 256
            && ((prefill_attn_matrix_g4_may_use() && arch.n_q_heads == arch.n_kv_heads * 4)
                || (prefill_attn_matrix_g8_may_use()
                    && arch.n_q_heads == 16
                    && arch.n_kv_heads == 2)
                || (prefill_attn_matrix_g6_may_use()
                    && arch.n_q_heads == 24
                    && arch.n_kv_heads == 4)
                || (prefill_attn_matrix_g16_may_use()
                    && arch.n_q_heads == 32
                    && arch.n_kv_heads == 2));
        let attn_matrix_max_pos = if enable_attn_matrix {
            prefill_attn_matrix_max_pos()
                .or(attn_matrix_max_pos_override)
                .unwrap_or(block_size as usize)
                .max(block_size as usize) as u64
        } else {
            0
        };
        let attn_matrix_scores_elems = if enable_attn_matrix {
            checked_u64_mul3(
                n,
                arch.n_q_heads as u64,
                attn_matrix_max_pos,
                "layer-major attn matrix scores size overflow",
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
        let final_logits_shape = if include_final_logits_pack {
            vec![n, v]
        } else {
            vec![1]
        };

        Ok(Self {
            x_pack: MetalTensor::zeros_f32(ctx, vec![n, h])?,
            h_pack: MetalTensor::zeros_f32(ctx, vec![n, h])?,
            mixer_out_pack: MetalTensor::zeros_f32(ctx, vec![n, h])?,
            attn_qkv_fused_pack: MetalTensor::zeros_f32(
                ctx,
                if enable_attn_fused_qkv_g8 {
                    vec![n, attn_qkv_fused_dim]
                } else {
                    vec![1]
                },
            )?,
            attn_q_full_pack: MetalTensor::zeros_f32(ctx, vec![n, attn_q_full_dim])?,
            attn_q_pack: MetalTensor::zeros_f32(ctx, vec![n, q_dim])?,
            attn_gate_pack: MetalTensor::zeros_f32(ctx, vec![n, q_dim])?,
            attn_q_normed_pack: MetalTensor::zeros_f32(ctx, vec![n, q_dim])?,
            attn_k_now_pack: MetalTensor::zeros_f32(ctx, vec![n, kv_dim])?,
            attn_v_now_pack: MetalTensor::zeros_f32(ctx, vec![n, kv_dim])?,
            attn_k_normed_pack: MetalTensor::zeros_f32(ctx, vec![n, kv_dim])?,
            attn_o_pack: MetalTensor::zeros_f32(ctx, vec![n, q_dim])?,
            attn_prefill_v4_o_partial_pack: MetalTensor::zeros_f32(
                ctx,
                if enable_attn_packed {
                    vec![attn_prefill_v4_o_partial_elems]
                } else {
                    vec![1]
                },
            )?,
            attn_prefill_v4_ml_partial_pack: MetalTensor::zeros_f32(
                ctx,
                if enable_attn_packed {
                    vec![attn_prefill_v4_ml_partial_elems]
                } else {
                    vec![1]
                },
            )?,
            attn_matrix_scores_pack: MetalTensor::zeros_f32(ctx, vec![attn_matrix_scores_elems])?,
            attn_matrix_vt_pack: MetalTensor::zeros_f16(ctx, vec![attn_matrix_vt_elems])?,
            ffn_gate_pack: MetalTensor::zeros_f32(ctx, vec![n, f])?,
            ffn_up_pack: MetalTensor::zeros_f32(ctx, vec![n, f])?,
            ffn_inner_pack: MetalTensor::zeros_f32(ctx, vec![n, f])?,
            ffn_out_pack: MetalTensor::zeros_f32(ctx, vec![n, h])?,
            moe_topk_idx_pack: MetalTensor::zeros_f32(ctx, vec![moe_slot_elems])?,
            moe_router_probs_pack: MetalTensor::zeros_f32(
                ctx,
                vec![n, (arch.expert_count as u64).max(1)],
            )?,
            moe_topk_weight_pack: MetalTensor::zeros_f32(ctx, vec![moe_slot_elems])?,
            moe_shared_gate_pack: MetalTensor::zeros_f32(ctx, vec![n])?,
            moe_inner_pack: MetalTensor::zeros_f32(ctx, vec![moe_inner_elems])?,
            moe_expert_out_pack: MetalTensor::zeros_f32(ctx, vec![moe_out_elems])?,
            moe_group_slot_idx_pack: MetalTensor::zeros_f32(ctx, vec![moe_slot_elems])?,
            moe_group_count_pack: MetalTensor::zeros_f32(
                ctx,
                vec![(arch.expert_count as u64).max(1)],
            )?,
            moe_group_ids_pack: MetalTensor::zeros_f32(ctx, vec![moe_group_ids_elems])?,
            moe_group_token_idx_pack: MetalTensor::zeros_f32(ctx, vec![moe_slot_elems])?,
            moe_group_weight_pack: MetalTensor::zeros_f32(ctx, vec![moe_slot_elems])?,
            moe_group_inner_pack: MetalTensor::zeros_f32(ctx, vec![moe_inner_elems])?,
            moe_group_out_pack: MetalTensor::zeros_f32(ctx, vec![moe_out_elems])?,
            moe_shared_ffn_gate_pack: MetalTensor::zeros_f32(ctx, vec![n, moe_f_shared])?,
            moe_shared_ffn_up_pack: MetalTensor::zeros_f32(ctx, vec![n, moe_f_shared])?,
            moe_shared_ffn_inner_pack: MetalTensor::zeros_f32(ctx, vec![n, moe_f_shared])?,
            moe_shared_ffn_out_pack: MetalTensor::zeros_f32(ctx, vec![n, h])?,
            final_logits_pack: MetalTensor::zeros_f32(ctx, final_logits_shape)?,
            gdn_qkv_pack: MetalTensor::zeros_f32(ctx, vec![n, gdn_conv_dim])?,
            gdn_z_pack: MetalTensor::zeros_f32(ctx, vec![n, gdn_v_dim])?,
            gdn_beta_pack: MetalTensor::zeros_f32(ctx, vec![n, gdn_n_v.max(1)])?,
            gdn_alpha_pack: MetalTensor::zeros_f32(ctx, vec![n, gdn_n_v.max(1)])?,
            gdn_q_norm_pack: MetalTensor::zeros_f32(ctx, vec![n, gdn_k_dim])?,
            gdn_k_norm_pack: MetalTensor::zeros_f32(ctx, vec![n, gdn_k_dim])?,
            gdn_v_pack: MetalTensor::zeros_f32(ctx, vec![n, gdn_v_dim])?,
            gdn_out_pack: MetalTensor::zeros_f32(ctx, vec![n, gdn_v_dim])?,
            gdn_normed_pack: MetalTensor::zeros_f32(ctx, vec![n, gdn_v_dim])?,
            n: block_size,
            hidden_size: h,
            intermediate_size: f,
            vocab_size: v,
            q_dim,
            kv_dim,
            attn_matrix_max_pos,
            gdn_conv_dim,
            gdn_v_dim,
            gdn_n_v,
        })
    }

    pub fn fresh(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
    ) -> Result<Self, MetalError> {
        Self::fresh_inner(ctx, target_model, block_size, true, None)
    }

    pub fn fresh_prefill(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
    ) -> Result<Self, MetalError> {
        Self::fresh_inner(ctx, target_model, block_size, false, None)
    }

    pub fn fresh_prefill_with_matrix_max_pos(
        ctx: &MetalContext,
        target_model: &crate::metal_forward::MetalModel,
        block_size: u32,
        matrix_max_pos: usize,
    ) -> Result<Self, MetalError> {
        Self::fresh_inner(ctx, target_model, block_size, false, Some(matrix_max_pos))
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

    let cmd_buf = base.ctx.queue.commandBuffer().expect("command buffer");

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
                base.ctx, &enc, &x_pack, attn_norm, &h_pack, n, h, RMS_EPS,
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
                            // alpha_proj (F32) -> fused exp(softplus(a+dt_bias)*a_log).
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
                            &gdn_normed_pack,
                            &mixer_out_pack,
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

                    // v0.76: sized views for adaptive-N back-off.
                    let attn_q_full_pack = layer_scratch
                        .attn_q_full_pack
                        .view_subrange(0, vec![(n * 2 * q_dim) as u64]);
                    let attn_q_pack = layer_scratch
                        .attn_q_pack
                        .view_subrange(0, vec![(n * q_dim) as u64]);
                    let attn_gate_pack = layer_scratch
                        .attn_gate_pack
                        .view_subrange(0, vec![(n * q_dim) as u64]);
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

                    // Step A: batched front-end Q (gated) / K / V projections,
                    // batched Q-norm and K-norm. One encoder per layer.
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
                        // Split q + gate into separate packs. The
                        // single-token kernel deinterleaves per-head;
                        // pass `n_heads = N * n_q` so it processes all
                        // N rows in one dispatch (per-head layout
                        // repeats identically across rows).
                        encode_split_q_gate_f32(
                            base.ctx,
                            &enc,
                            &attn_q_full_pack,
                            &attn_q_pack,
                            &attn_gate_pack,
                            n * n_q,
                            head_dim,
                        )?;
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
                        // Q-norm (per-head); n_heads = N * n_q.
                        encode_rms_norm_batched_f32(
                            base.ctx,
                            &enc,
                            &attn_q_pack,
                            &a.q_norm,
                            &attn_q_normed_pack,
                            n * n_q,
                            head_dim,
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
                            let group = n_q / n_kv;
                            let use_v4 = head_dim == V4_HEAD_DIM && matches!(group, 4 | 6 | 8 | 16);
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

                    // Step C: gate-sigmoid + mul (flat elementwise on N*q_dim)
                    // followed by batched o_proj mat-mat. One encoder per layer.
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        // attn_gate_pack -> sigmoid into a scratch view of
                        // attn_q_pack (no longer needed; q_pack is dead after
                        // attn-v4). Same in-place reuse pattern as the
                        // single-token encode_attn (line 1399 of metal_forward.rs).
                        encode_sigmoid_f32(base.ctx, &enc, &attn_gate_pack, &attn_q_pack)?;
                        // attn_o_pack *= sigmoid(gate_pack), elementwise on N*q_dim.
                        crate::metal::encode_mul_f32(
                            base.ctx,
                            &enc,
                            &attn_o_pack,
                            &attn_q_pack,
                            &attn_o_pack,
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
                encode_mat_mat_dispatch(base.ctx, &enc, g_w, &h_pack, &ffn_gate_pack, h, f, n)?;
                encode_mat_mat_dispatch(base.ctx, &enc, u_w, &h_pack, &ffn_up_pack, h, f, n)?;
                encode_silu_mul_f32(
                    base.ctx,
                    &enc,
                    &ffn_gate_pack,
                    &ffn_up_pack,
                    &ffn_inner_pack,
                )?;
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
            // Batched argmax across all N rows in ONE dispatch.
            encode_argmax_f32(base.ctx, &enc, logits_dst, &verify_argmax_view, n, v)?;
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
    let (logits, gpu_ms) = prefill_tokens_with_multi_hidden_profiled_inner(
        base,
        token_ids,
        start_position,
        target_session,
        layer_scratch,
        target_layer_ids,
        hidden_dst,
        PrefillTailMode::ReadLogits,
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
    let (_, gpu_ms) = prefill_tokens_with_multi_hidden_profiled_inner(
        base,
        token_ids,
        start_position,
        target_session,
        layer_scratch,
        &[],
        None,
        PrefillTailMode::SkipTail,
    )?;
    Ok(gpu_ms)
}

fn prefill_tokens_with_multi_hidden_profiled_inner(
    base: &MetalForward<'_>,
    token_ids: &[i32],
    start_position: u32,
    target_session: &mut MetalSession,
    layer_scratch: &mut MetalDFlashLayerMajorScratch,
    target_layer_ids: &[u32],
    hidden_dst: Option<&MetalTensor>,
    tail_mode: PrefillTailMode,
) -> Result<(Option<Vec<f32>>, f64), DFlashError> {
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
    if attn_matrix_g4_force_on
        || attn_matrix_g8_force_on
        || attn_matrix_g6_force_on
        || attn_matrix_g16_force_on
    {
        if layer_scratch.attn_matrix_max_pos < last_pos as u64 {
            return Err(DFlashError::Metal(MetalError::BadShape {
                kernel: "prefill_attn_matrix",
                detail: format!(
                    "matrix scratch max_pos={} < required last_pos={last_pos}; set QWEN_PREFILL_ATTN_MATRIX_MAX_POS before scratch allocation",
                    layer_scratch.attn_matrix_max_pos
                ),
            }));
        }
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
            && n_in % 4 == 0
            && n_out % 8 == 0
        {
            crate::metal::encode_mat_mat_f32_router_e8p32(
                base.ctx, enc, weight, x, y, n_in, n_out, n_query,
            )?;
        } else {
            encode_mat_mat_dispatch(base.ctx, enc, weight, x, y, n_in, n_out, n_query)?;
        }
        Ok(())
    };

    // Per-call ids buffer. P=16 i32 = 64 bytes; trivial alloc cost.
    // (Same "F32-typed buffer holding i32" convention as
    // verify_scratch.packed_ids_buf — see the noise_ids comment in
    // MetalDFlashSession.)
    let ids_buf =
        MetalTensor::zeros_f32(base.ctx, vec![p_max as u64]).map_err(DFlashError::Metal)?;

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
            );

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
                            );
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
                            );
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
                            );
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
                        );

                        if matches!(gdn_split, PrefillGdnSplitMode::SkipAll) {
                            apply_mixer_residual = false;
                        } else if dense_packed_gdn_step_enabled() {
                            if gdn_split.run_prep() {
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
                                );
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
                                );
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
                                    RMS_EPS,
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
                                );
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
                            );
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
                                );
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
                            );
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
                        let use_fused_qkv = prefill_attn_fused_qkv_g8_enabled(
                            chunk_start as usize + chunk_p,
                            n_q / n_kv,
                        ) && a.qkv_fused.is_some();
                        let qkv_fused_dim = attn_q_full_dim + 2 * kv_dim;

                        // Step A: batched front-end Q gated / K / V projections + Q-norm + K-norm.
                        if trace_attn_phases {
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
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
                                if !use_fused_qkv {
                                    encode_split_q_gate_f32(
                                        base.ctx,
                                        &enc,
                                        &q_full_pack_p,
                                        &q_pack_p,
                                        &gate_pack_p,
                                        chunk_p * n_q,
                                        head_dim,
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
                                if use_fused_qkv { "qkv_matmul" } else { "proj" },
                            );
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                if use_fused_qkv {
                                    let fused_qkv_pack = layer_scratch
                                        .attn_qkv_fused_pack
                                        .view_subrange(0, vec![(chunk_p * qkv_fused_dim) as u64]);
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
                                }
                                encode_split_q_gate_f32(
                                    base.ctx,
                                    &enc,
                                    &q_full_pack_p,
                                    &q_pack_p,
                                    &gate_pack_p,
                                    chunk_p * n_q,
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
                                if use_fused_qkv {
                                    "split"
                                } else {
                                    "split_q_gate"
                                },
                            );
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
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
                            flush_prefill_phase(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                                trace_attn_phases,
                                chunk_idx,
                                chunk_start,
                                il,
                                "norm",
                            );
                        } else {
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
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
                                encode_split_q_gate_f32(
                                    base.ctx,
                                    &enc,
                                    &q_full_pack_p,
                                    &q_pack_p,
                                    &gate_pack_p,
                                    chunk_p * n_q,
                                    head_dim,
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
                        }

                        if prefill_noop_attn_body_enabled() {
                            apply_mixer_residual = false;
                            target_session.kv_n_pos[ai] = chunk_start as usize + chunk_p;
                        } else {
                            let use_packed_g8 = prefill_attn_packed_g8_enabled(
                                chunk_start as usize + chunk_p,
                                n_q / n_kv,
                            ) && target_session.kv_k[ai].dtype == GgmlType::F16
                                && target_session.kv_v[ai].dtype == GgmlType::F16
                                && head_dim == 256
                                && n_q == 16
                                && n_kv == 2;
                            let use_packed_g16 = prefill_attn_packed_g16_enabled(
                                chunk_start as usize + chunk_p,
                                n_q / n_kv,
                            ) && target_session.kv_k[ai].dtype
                                == GgmlType::F16
                                && target_session.kv_v[ai].dtype == GgmlType::F16
                                && head_dim == 256
                                && n_q == 32
                                && n_kv == 2;
                            let matrix_scratch_covers_chunk = layer_scratch.attn_matrix_max_pos
                                >= chunk_start as u64 + chunk_p as u64;
                            let use_matrix_g8 = use_packed_g8
                                && prefill_attn_matrix_g8_may_use()
                                && matrix_scratch_covers_chunk;
                            let use_matrix_g4 = prefill_attn_matrix_g4_may_use()
                                && target_session.kv_k[ai].dtype == GgmlType::F16
                                && target_session.kv_v[ai].dtype == GgmlType::F16
                                && head_dim == 256
                                && n_q == n_kv * 4
                                && matrix_scratch_covers_chunk;
                            let use_matrix_g6 = prefill_attn_matrix_g6_may_use()
                                && target_session.kv_k[ai].dtype == GgmlType::F16
                                && target_session.kv_v[ai].dtype == GgmlType::F16
                                && head_dim == 256
                                && n_q == 24
                                && n_kv == 4
                                && matrix_scratch_covers_chunk;
                            let use_matrix_g16 = use_packed_g16
                                && prefill_attn_matrix_g16_may_use()
                                && matrix_scratch_covers_chunk;
                            let use_matrix =
                                use_matrix_g4 || use_matrix_g8 || use_matrix_g6 || use_matrix_g16;
                            let n_pos = chunk_start as usize + chunk_p;
                            let rebuild_matrix_vt_prefix =
                                use_matrix && attn_matrix_vt_valid_until[ai] < chunk_start as usize;
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                encode_rope_neox_f32_packed_consecutive(
                                    base.ctx,
                                    &enc,
                                    &q_normed_pack_p,
                                    chunk_p,
                                    n_q,
                                    head_dim,
                                    n_rot,
                                    chunk_start,
                                    arch.rope_theta,
                                )?;
                                encode_rope_neox_f32_packed_consecutive(
                                    base.ctx,
                                    &enc,
                                    &k_normed_pack_p,
                                    chunk_p,
                                    n_kv,
                                    head_dim,
                                    n_rot,
                                    chunk_start,
                                    arch.rope_theta,
                                )?;
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
                                "rope_scatter",
                            );

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
                                    let scores_bytes =
                                        chunk_p * n_q * n_pos * std::mem::size_of::<f32>();
                                    let vt_bytes =
                                        n_kv * head_dim * vt_stride * std::mem::size_of::<u16>();
                                    let (vt_base, vt_rows) = if rebuild_matrix_vt_prefix {
                                        (0, n_pos)
                                    } else {
                                        (chunk_start as usize, chunk_p)
                                    };
                                    let vt_update_bytes =
                                        n_kv * head_dim * vt_rows * std::mem::size_of::<u16>();
                                    eprintln!(
                                        "[prefill-attn-matrix-g{}-shape] layer={} chunk_start={} chunk_p={} n_pos={} scores_mib={:.2} vt_stride={} vt_layer_mib={:.2} vt_update_base={} vt_update_rows={} vt_update_mib={:.2}",
                                        group,
                                        il,
                                        chunk_start,
                                        chunk_p,
                                        n_pos,
                                        scores_bytes as f64 / (1024.0 * 1024.0),
                                        vt_stride,
                                        vt_bytes as f64 / (1024.0 * 1024.0),
                                        vt_base,
                                        vt_rows,
                                        vt_update_bytes as f64 / (1024.0 * 1024.0),
                                    );
                                }
                                let trace_matrix_subphases =
                                    use_matrix && trace_attn_phases && !attn_packed_oracle;
                                traced_matrix_subphases = trace_matrix_subphases;
                                if trace_matrix_subphases {
                                    let rebuild_vt_prefix = rebuild_matrix_vt_prefix;
                                    let vt_stride = layer_scratch.attn_matrix_max_pos as usize;
                                    let per_attn_vt = n_kv * head_dim * vt_stride;
                                    let scores = layer_scratch
                                        .attn_matrix_scores_pack
                                        .view_subrange(0, vec![(chunk_p * n_q * n_pos) as u64]);
                                    let v_t = layer_scratch.attn_matrix_vt_pack.view_subrange(
                                        (ai * per_attn_vt) as u64,
                                        vec![per_attn_vt as u64],
                                    );

                                    if rebuild_vt_prefix {
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
                                            n_pos,
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
                                        );
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
                                            prefill_attn_matrix_causal_skip_enabled()
                                                && use_matrix_g6,
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
                                        "body_matrix_kq",
                                    );

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
                                    );

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
                                            prefill_attn_matrix_causal_skip_enabled()
                                                && use_matrix_g6,
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
                                        "body_matrix_kqv",
                                    );

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
                                        let rebuild_vt_prefix = rebuild_matrix_vt_prefix;
                                        let vt_stride = layer_scratch.attn_matrix_max_pos as usize;
                                        let per_attn_vt = n_kv * head_dim * vt_stride;
                                        let scores = layer_scratch
                                            .attn_matrix_scores_pack
                                            .view_subrange(0, vec![(chunk_p * n_q * n_pos) as u64]);
                                        let v_t = layer_scratch.attn_matrix_vt_pack.view_subrange(
                                            (ai * per_attn_vt) as u64,
                                            vec![per_attn_vt as u64],
                                        );
                                        if rebuild_vt_prefix {
                                            crate::metal::encode_attn_matrix_transpose_v_f16(
                                                base.ctx,
                                                &enc,
                                                &target_session.kv_v[ai],
                                                &v_t,
                                                0,
                                                n_pos,
                                                n_pos,
                                                n_kv * head_dim,
                                                vt_stride,
                                                n_kv,
                                                head_dim,
                                            )?;
                                        }
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
                                            prefill_attn_matrix_causal_skip_enabled()
                                                && use_matrix_g6,
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
                                            prefill_attn_matrix_causal_skip_enabled()
                                                && use_matrix_g6,
                                        )?;
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
                                );
                            }

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
                            );
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
                            );
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
                );
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
                );

                let router_mat_mat_eligible = |dtype: GgmlType| {
                    matches!(
                        dtype,
                        GgmlType::F32
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
                    GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0 | GgmlType::IQ4_XS
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
                    (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS) => {
                        chunk_p >= 32 && prefill_moe_grouped_iq3_gateup_enabled()
                    }
                    (GgmlType::F32, GgmlType::F32) => {
                        chunk_p >= 32 && prefill_moe_grouped_f32_gateup_enabled()
                    }
                    _ => false,
                };
                let grouped_routed_path = prefill_moe_grouped_enabled()
                    && grouped_gate_up_dtype_eligible
                    && grouped_down_dtype_eligible
                    && h % 256 == 0
                    && f_exp % 256 == 0;

                if !skip_ffn && (packed_routed_path || grouped_routed_path) {
                    let skip_moe_routed = prefill_noop_moe_routed_enabled();
                    let skip_moe_shared = prefill_noop_moe_shared_enabled();
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
                    let moe_inner_pack_p = layer_scratch
                        .moe_inner_pack
                        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
                    let moe_mixer_out_pack_p = layer_scratch
                        .mixer_out_pack
                        .view_subrange(0, vec![(chunk_p * h) as u64]);
                    let moe_expert_out_pack_p = layer_scratch
                        .moe_expert_out_pack
                        .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
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
                    );
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
                        if zero_grouped_buffers {
                            encode_fill_f32(base.ctx, &enc, &moe_group_out_pack_p, 0.0)?;
                        }
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
                        );

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
                        );
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
                        );
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
                                );
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
                                let enc = KernelEncoder::begin(&cmd_buf);
                                label_prefill_encoder(&enc, il, "moe-routed-grouped-swiglu");
                                if zero_grouped_buffers {
                                    encode_fill_f32(base.ctx, &enc, &moe_group_inner_pack_p, 0.0)?;
                                }
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
                                );
                            }
                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                label_prefill_encoder(&enc, il, "moe-routed-grouped-down");
                                if zero_grouped_buffers {
                                    encode_fill_f32(base.ctx, &enc, &moe_group_out_pack_p, 0.0)?;
                                }
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
                                );
                            }
                            if !fused_grouped_finalizer {
                                let enc = KernelEncoder::begin(&cmd_buf);
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
                                );
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
                            if zero_grouped_buffers {
                                encode_fill_f32(base.ctx, &enc, &moe_group_out_pack_p, 0.0)?;
                            }
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
                            if !fused_grouped_finalizer {
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
                                "routed_grouped",
                            );
                        }
                    } else if let Some(hot_threshold) = hot_expert_min_slots {
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
                        cpu_write_i32_f32buf(&moe_group_slot_idx_pack_p, &slot_ids);
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
                        );
                    } else if prefill_moe_packed_down_sum_enabled() {
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
                        );
                    } else {
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
                        );
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
                        );
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
                        );
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
                        );
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
                            );

                            {
                                let enc = KernelEncoder::begin(&cmd_buf);
                                base.encode_moe_route_prepare(&enc, target_session, moe)?;
                                enc.end();
                            }
                            route_ms += flush_prefill_layer_phase_accum(
                                base.ctx,
                                &mut cmd_buf,
                                &mut prefill_gpu_total_ms,
                            );

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
                                );

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
                                );

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
                                );

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
                                );

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
                                );
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
                                );
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
                            );

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
                            );
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
                    );

                    let use_fused_swiglu = prefill_dense_ffn_fused_swiglu_q4_enabled(h)
                        && g_w.dtype == GgmlType::Q4_K
                        && u_w.dtype == GgmlType::Q4_K
                        && h % 256 == 0
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
                        );

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
                        );

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
                        );
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
                        );
                    }

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
                    );
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
                            && h % 256 == 0
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
                    );
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
            if matches!(tail_mode, PrefillTailMode::ReadLogits) {
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
            }
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
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
            if matches!(tail_mode, PrefillTailMode::ReadLogits) {
                let mut last_logits = vec![0.0f32; v];
                unsafe {
                    let src = target_session.logits.buffer.contents().as_ptr() as *const f32;
                    std::ptr::copy_nonoverlapping(src, last_logits.as_mut_ptr(), v);
                }
                return Ok((Some(last_logits), prefill_gpu_total_ms));
            }
            return Ok((None, prefill_gpu_total_ms));
        }

        // Non-last chunk: just commit + wait (no tail).
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
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
    use crate::metal::{
        MetalContext, encode_gdn_step_decay_f32, encode_l2_norm_batched_f32,
        encode_rmsnorm_gated_f32, encode_ssm_conv_silu_f32,
    };
    use crate::metal_forward::{MetalModel, MetalSession};
    use std::time::Instant;

    fn write_tensor_f32(t: &MetalTensor, data: &[f32]) {
        assert_eq!(t.dtype, GgmlType::F32);
        assert_eq!(t.n_elements() as usize, data.len());
        unsafe {
            let dst = (t.buffer.contents().as_ptr() as *mut f32).add((t.offset / 4) as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }

    fn read_tensor_i32_f32buf(t: &MetalTensor) -> Vec<i32> {
        assert_eq!(t.dtype, GgmlType::F32);
        let n = t.n_elements() as usize;
        let mut out = vec![0i32; n];
        unsafe {
            let src = (t.buffer.contents().as_ptr() as *const i32).add((t.offset / 4) as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    }

    fn write_tensor_i32_f32buf(t: &MetalTensor, data: &[i32]) {
        assert_eq!(t.dtype, GgmlType::F32);
        assert_eq!(t.n_elements() as usize, data.len());
        unsafe {
            let dst = (t.buffer.contents().as_ptr() as *mut i32).add((t.offset / 4) as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct ExpertGroupRange {
        expert: usize,
        start: usize,
        len: usize,
    }

    fn build_expert_slot_groups(
        topk_idx: &[i32],
        topk_weight: &[f32],
        topk: usize,
        n_expert: usize,
    ) -> (Vec<ExpertGroupRange>, Vec<i32>, Vec<i32>, Vec<f32>) {
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
            ranges.push(ExpertGroupRange {
                expert,
                start,
                len: slot_ids.len() - start,
            });
        }
        (ranges, slot_ids, token_ids, weights)
    }

    fn timed_gpu_cmd<F>(ctx: &MetalContext, f: F) -> f64
    where
        F: FnOnce(&KernelEncoder),
    {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        f(&enc);
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3
    }

    fn run_packed_moe_tail_profile(model_path: &str, label: &str, chunk_p: usize, n_runs: usize) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[packed-moe-tail-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, g_w, u_w, d_w, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => (
                &g.post_attn_norm,
                &g.ffn_gate,
                &g.ffn_up,
                &g.ffn_down,
                g.ffn_moe.as_ref().expect("moe block"),
            ),
            crate::metal_forward::MetalBlock::Attn(a) => (
                &a.post_attn_norm,
                &a.ffn_gate,
                &a.ffn_up,
                &a.ffn_down,
                a.ffn_moe.as_ref().expect("moe block"),
            ),
        };

        let mut session = MetalSession::fresh(&ctx, &mm, chunk_p + 16).expect("session");
        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        // Warmup pipeline cache.
        write_tensor_f32(&x_pack, &x_init);
        {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_rms_norm_batched_f32(
                &ctx,
                &enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("warmup postnorm");
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        for n_idx in 0..chunk_p.min(2) {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_copy_offset_f32(&ctx, &enc, &x_pack, n_idx * h, &session.x, h).expect("copy x");
            encode_copy_offset_f32(&ctx, &enc, &h_pack, n_idx * h, &session.h, h).expect("copy h");
            mf.encode_moe_route_prepare(&enc, &mut session, moe)
                .expect("route");
            mf.encode_moe_routed_ffn_gpu(&enc, &mut session, moe)
                .expect("routed");
            mf.encode_moe_shared_ffn_gpu(&enc, &mut session, g_w, u_w, d_w)
                .expect("shared");
            encode_add_inplace_f32(&ctx, &enc, &session.x, &session.mixer_out).expect("resid");
            encode_scatter_offset_f32(&ctx, &enc, &session.x, &x_pack, n_idx * h, h)
                .expect("scatter");
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let mut postnorm_ms = 0.0f64;
        let mut route_ms = 0.0f64;
        let mut routed_ms = 0.0f64;
        let mut resid_scatter_ms = 0.0f64;
        let mut total_wall_ms = 0.0f64;

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);

            let wall = Instant::now();
            {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                encode_rms_norm_batched_f32(
                    &ctx,
                    &enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
                enc.end();
                cmd.commit();
                cmd.waitUntilCompleted();
                postnorm_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            }
            for n_idx in 0..chunk_p {
                {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    encode_copy_offset_f32(&ctx, &enc, &x_pack, n_idx * h, &session.x, h)
                        .expect("copy x");
                    encode_copy_offset_f32(&ctx, &enc, &h_pack, n_idx * h, &session.h, h)
                        .expect("copy h");
                    mf.encode_moe_route_prepare(&enc, &mut session, moe)
                        .expect("route");
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    route_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                }
                {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    mf.encode_moe_routed_ffn_gpu(&enc, &mut session, moe)
                        .expect("routed");
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    routed_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                }
                {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    mf.encode_moe_shared_ffn_gpu(&enc, &mut session, g_w, u_w, d_w)
                        .expect("shared");
                    encode_add_inplace_f32(&ctx, &enc, &session.x, &session.mixer_out)
                        .expect("resid");
                    encode_scatter_offset_f32(&ctx, &enc, &session.x, &x_pack, n_idx * h, h)
                        .expect("scatter");
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    resid_scatter_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                }
            }
            total_wall_ms += wall.elapsed().as_secs_f64() * 1e3;
        }

        let denom = n_runs as f64;
        let postnorm_ms = postnorm_ms / denom;
        let route_ms = route_ms / denom;
        let routed_ms = routed_ms / denom;
        let resid_scatter_ms = resid_scatter_ms / denom;
        let total_gpu = postnorm_ms + route_ms + routed_ms + resid_scatter_ms;
        eprintln!(
            "[packed-moe-tail-{label}] chunk_p={chunk_p} avg wall={:.2} ms",
            total_wall_ms / denom
        );
        eprintln!(
            "[packed-moe-tail-{label}]   postnorm          {:6.2} ms ({:5.1}%)",
            postnorm_ms,
            postnorm_ms / total_gpu * 100.0
        );
        eprintln!(
            "[packed-moe-tail-{label}]   route+copy        {:6.2} ms ({:5.1}%)",
            route_ms,
            route_ms / total_gpu * 100.0
        );
        eprintln!(
            "[packed-moe-tail-{label}]   routed_ffn        {:6.2} ms ({:5.1}%)",
            routed_ms,
            routed_ms / total_gpu * 100.0
        );
        eprintln!(
            "[packed-moe-tail-{label}]   shared+resid+copy {:6.2} ms ({:5.1}%)",
            resid_scatter_ms,
            resid_scatter_ms / total_gpu * 100.0
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_packed_moe_tail_profile() {
        run_packed_moe_tail_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            8,
            4,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_packed_moe_tail_profile() {
        run_packed_moe_tail_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            8,
            3,
        );
    }

    fn run_packed_moe_tail_ab_profile(
        model_path: &str,
        label: &str,
        chunk_ps: &[usize],
        n_runs: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[moe-tail-ab-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, g_w, u_w, d_w, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => (
                &g.post_attn_norm,
                &g.ffn_gate,
                &g.ffn_up,
                &g.ffn_down,
                g.ffn_moe.as_ref().expect("moe block"),
            ),
            crate::metal_forward::MetalBlock::Attn(a) => (
                &a.post_attn_norm,
                &a.ffn_gate,
                &a.ffn_up,
                &a.ffn_down,
                a.ffn_moe.as_ref().expect("moe block"),
            ),
        };

        for &chunk_p in chunk_ps {
            let mut session = MetalSession::fresh(&ctx, &mm, chunk_p + 16).expect("session");
            let scratch =
                MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
            let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
            let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
            let router_probs_pack = scratch
                .moe_router_probs_pack
                .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
            let topk_idx_pack = scratch
                .moe_topk_idx_pack
                .view_subrange(0, vec![(chunk_p * topk) as u64]);
            let topk_weight_pack = scratch
                .moe_topk_weight_pack
                .view_subrange(0, vec![(chunk_p * topk) as u64]);
            let shared_gate_pack = scratch
                .moe_shared_gate_pack
                .view_subrange(0, vec![chunk_p as u64]);
            let moe_inner_pack = scratch
                .moe_inner_pack
                .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
            let mixer_out_pack = scratch
                .mixer_out_pack
                .view_subrange(0, vec![(chunk_p * h) as u64]);
            let x_init: Vec<f32> = (0..chunk_p * h)
                .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
                .collect();

            let mut old_postnorm = 0.0f64;
            let mut old_route = 0.0f64;
            let mut old_routed = 0.0f64;
            let mut old_shared = 0.0f64;
            let mut old_wall = 0.0f64;
            let mut new_postnorm = 0.0f64;
            let mut new_route = 0.0f64;
            let mut new_swiglu = 0.0f64;
            let mut new_down = 0.0f64;
            let mut new_shared = 0.0f64;
            let mut new_wall = 0.0f64;
            let mut old_one_cb = 0.0f64;
            let mut new_one_cb = 0.0f64;

            for run_idx in 0..=n_runs {
                let sample = run_idx > 0;

                write_tensor_f32(&x_pack, &x_init);
                let old_wall_start = Instant::now();
                let ms = timed_gpu_cmd(&ctx, |enc| {
                    encode_rms_norm_batched_f32(
                        &ctx,
                        enc,
                        &x_pack,
                        post_norm,
                        &h_pack,
                        chunk_p,
                        h,
                        crate::metal_forward::RMS_EPS,
                    )
                    .expect("old postnorm");
                });
                if sample {
                    old_postnorm += ms;
                }
                for n_idx in 0..chunk_p {
                    let ms = timed_gpu_cmd(&ctx, |enc| {
                        encode_copy_offset_f32(&ctx, enc, &x_pack, n_idx * h, &session.x, h)
                            .expect("old copy x");
                        encode_copy_offset_f32(&ctx, enc, &h_pack, n_idx * h, &session.h, h)
                            .expect("old copy h");
                        mf.encode_moe_route_prepare(enc, &mut session, moe)
                            .expect("old route");
                    });
                    if sample {
                        old_route += ms;
                    }
                    let ms = timed_gpu_cmd(&ctx, |enc| {
                        mf.encode_moe_routed_ffn_gpu(enc, &mut session, moe)
                            .expect("old routed");
                    });
                    if sample {
                        old_routed += ms;
                    }
                    let ms = timed_gpu_cmd(&ctx, |enc| {
                        mf.encode_moe_shared_ffn_gpu(enc, &mut session, g_w, u_w, d_w)
                            .expect("old shared");
                        encode_add_inplace_f32(&ctx, enc, &session.x, &session.mixer_out)
                            .expect("old resid");
                        encode_scatter_offset_f32(&ctx, enc, &session.x, &x_pack, n_idx * h, h)
                            .expect("old scatter");
                    });
                    if sample {
                        old_shared += ms;
                    }
                }
                if sample {
                    old_wall += old_wall_start.elapsed().as_secs_f64() * 1e3;
                }

                write_tensor_f32(&x_pack, &x_init);
                let new_wall_start = Instant::now();
                let ms = timed_gpu_cmd(&ctx, |enc| {
                    encode_rms_norm_batched_f32(
                        &ctx,
                        enc,
                        &x_pack,
                        post_norm,
                        &h_pack,
                        chunk_p,
                        h,
                        crate::metal_forward::RMS_EPS,
                    )
                    .expect("new postnorm");
                });
                if sample {
                    new_postnorm += ms;
                }
                let ms = timed_gpu_cmd(&ctx, |enc| {
                    encode_mat_mat_dispatch(
                        &ctx,
                        enc,
                        &moe.gate_inp,
                        &h_pack,
                        &router_probs_pack,
                        h,
                        n_expert,
                        chunk_p,
                    )
                    .expect("new router matmat");
                    encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                        &ctx,
                        enc,
                        &router_probs_pack,
                        &moe.gate_inp_shexp,
                        &h_pack,
                        &topk_idx_pack,
                        &topk_weight_pack,
                        &shared_gate_pack,
                        n_expert,
                        topk,
                        h,
                        chunk_p,
                    )
                    .expect("new packed topk/shared gate");
                });
                if sample {
                    new_route += ms;
                }
                let ms = timed_gpu_cmd(&ctx, |enc| {
                    encode_moe_swiglu_q4_K_f32_packed_slots(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &topk_idx_pack,
                        &moe_inner_pack,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("new packed swiglu");
                });
                if sample {
                    new_swiglu += ms;
                }
                let ms = timed_gpu_cmd(&ctx, |enc| {
                    encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                        &ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner_pack,
                        &topk_idx_pack,
                        &topk_weight_pack,
                        &mixer_out_pack,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("new packed down+sum");
                });
                if sample {
                    new_down += ms;
                }
                for n_idx in 0..chunk_p {
                    let mixer_n = mixer_out_pack.view_subrange((n_idx * h) as u64, vec![h as u64]);
                    let shared_gate_n = shared_gate_pack.view_subrange(n_idx as u64, vec![1]);
                    let ms = timed_gpu_cmd(&ctx, |enc| {
                        encode_copy_offset_f32(&ctx, enc, &x_pack, n_idx * h, &session.x, h)
                            .expect("new copy x");
                        encode_copy_offset_f32(&ctx, enc, &h_pack, n_idx * h, &session.h, h)
                            .expect("new copy h");
                        encode_copy_offset_f32(&ctx, enc, &mixer_n, 0, &session.mixer_out, h)
                            .expect("new copy routed out");
                        encode_copy_offset_f32(
                            &ctx,
                            enc,
                            &shared_gate_n,
                            0,
                            &session.moe_shared_gate,
                            1,
                        )
                        .expect("new copy shared gate");
                        mf.encode_moe_shared_ffn_gpu(enc, &mut session, g_w, u_w, d_w)
                            .expect("new shared");
                        encode_add_inplace_f32(&ctx, enc, &session.x, &session.mixer_out)
                            .expect("new resid");
                        encode_scatter_offset_f32(&ctx, enc, &session.x, &x_pack, n_idx * h, h)
                            .expect("new scatter");
                    });
                    if sample {
                        new_shared += ms;
                    }
                }
                if sample {
                    new_wall += new_wall_start.elapsed().as_secs_f64() * 1e3;
                }

                write_tensor_f32(&x_pack, &x_init);
                let ms = timed_gpu_cmd(&ctx, |enc| {
                    encode_rms_norm_batched_f32(
                        &ctx,
                        enc,
                        &x_pack,
                        post_norm,
                        &h_pack,
                        chunk_p,
                        h,
                        crate::metal_forward::RMS_EPS,
                    )
                    .expect("old-cb postnorm");
                    for n_idx in 0..chunk_p {
                        encode_copy_offset_f32(&ctx, enc, &x_pack, n_idx * h, &session.x, h)
                            .expect("old-cb copy x");
                        encode_copy_offset_f32(&ctx, enc, &h_pack, n_idx * h, &session.h, h)
                            .expect("old-cb copy h");
                        mf.encode_moe_route_prepare(enc, &mut session, moe)
                            .expect("old-cb route");
                        mf.encode_moe_routed_ffn_gpu(enc, &mut session, moe)
                            .expect("old-cb routed");
                        mf.encode_moe_shared_ffn_gpu(enc, &mut session, g_w, u_w, d_w)
                            .expect("old-cb shared");
                        encode_add_inplace_f32(&ctx, enc, &session.x, &session.mixer_out)
                            .expect("old-cb resid");
                        encode_scatter_offset_f32(&ctx, enc, &session.x, &x_pack, n_idx * h, h)
                            .expect("old-cb scatter");
                    }
                });
                if sample {
                    old_one_cb += ms;
                }

                write_tensor_f32(&x_pack, &x_init);
                let ms = timed_gpu_cmd(&ctx, |enc| {
                    encode_rms_norm_batched_f32(
                        &ctx,
                        enc,
                        &x_pack,
                        post_norm,
                        &h_pack,
                        chunk_p,
                        h,
                        crate::metal_forward::RMS_EPS,
                    )
                    .expect("new-cb postnorm");
                    encode_mat_mat_dispatch(
                        &ctx,
                        enc,
                        &moe.gate_inp,
                        &h_pack,
                        &router_probs_pack,
                        h,
                        n_expert,
                        chunk_p,
                    )
                    .expect("new-cb router matmat");
                    encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                        &ctx,
                        enc,
                        &router_probs_pack,
                        &moe.gate_inp_shexp,
                        &h_pack,
                        &topk_idx_pack,
                        &topk_weight_pack,
                        &shared_gate_pack,
                        n_expert,
                        topk,
                        h,
                        chunk_p,
                    )
                    .expect("new-cb packed topk/shared gate");
                    encode_moe_swiglu_q4_K_f32_packed_slots(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &topk_idx_pack,
                        &moe_inner_pack,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("new-cb packed swiglu");
                    encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                        &ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner_pack,
                        &topk_idx_pack,
                        &topk_weight_pack,
                        &mixer_out_pack,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("new-cb packed down+sum");
                    for n_idx in 0..chunk_p {
                        let mixer_n =
                            mixer_out_pack.view_subrange((n_idx * h) as u64, vec![h as u64]);
                        let shared_gate_n = shared_gate_pack.view_subrange(n_idx as u64, vec![1]);
                        encode_copy_offset_f32(&ctx, enc, &x_pack, n_idx * h, &session.x, h)
                            .expect("new-cb copy x");
                        encode_copy_offset_f32(&ctx, enc, &h_pack, n_idx * h, &session.h, h)
                            .expect("new-cb copy h");
                        encode_copy_offset_f32(&ctx, enc, &mixer_n, 0, &session.mixer_out, h)
                            .expect("new-cb copy routed out");
                        encode_copy_offset_f32(
                            &ctx,
                            enc,
                            &shared_gate_n,
                            0,
                            &session.moe_shared_gate,
                            1,
                        )
                        .expect("new-cb copy shared gate");
                        mf.encode_moe_shared_ffn_gpu(enc, &mut session, g_w, u_w, d_w)
                            .expect("new-cb shared");
                        encode_add_inplace_f32(&ctx, enc, &session.x, &session.mixer_out)
                            .expect("new-cb resid");
                        encode_scatter_offset_f32(&ctx, enc, &session.x, &x_pack, n_idx * h, h)
                            .expect("new-cb scatter");
                    }
                });
                if sample {
                    new_one_cb += ms;
                }
            }

            let denom = n_runs as f64;
            let old_postnorm = old_postnorm / denom;
            let old_route = old_route / denom;
            let old_routed = old_routed / denom;
            let old_shared = old_shared / denom;
            let old_wall = old_wall / denom;
            let new_postnorm = new_postnorm / denom;
            let new_route = new_route / denom;
            let new_swiglu = new_swiglu / denom;
            let new_down = new_down / denom;
            let new_shared = new_shared / denom;
            let new_wall = new_wall / denom;
            let old_one_cb = old_one_cb / denom;
            let new_one_cb = new_one_cb / denom;
            let old_gpu = old_postnorm + old_route + old_routed + old_shared;
            let new_gpu = new_postnorm + new_route + new_swiglu + new_down + new_shared;
            eprintln!(
                "[moe-tail-ab-{label}] P={chunk_p} split_old_gpu={old_gpu:.2} ms split_new_gpu={new_gpu:.2} ms split_speedup={:.3} old_wall={old_wall:.2} ms new_wall={new_wall:.2} ms",
                old_gpu / new_gpu
            );
            eprintln!(
                "[moe-tail-ab-{label}]   one_cb old={old_one_cb:.2} ms new={new_one_cb:.2} ms speedup={:.3}",
                old_one_cb / new_one_cb
            );
            eprintln!(
                "[moe-tail-ab-{label}]   old postnorm={old_postnorm:.2} route+copy={old_route:.2} routed={old_routed:.2} shared+resid+copy={old_shared:.2}"
            );
            eprintln!(
                "[moe-tail-ab-{label}]   new postnorm={new_postnorm:.2} packed_route={new_route:.2} swiglu={new_swiglu:.2} down_sum={new_down:.2} shared+resid+copy={new_shared:.2}"
            );
        }
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_packed_moe_tail_ab_profile() {
        run_packed_moe_tail_ab_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            &[8, 16, 64, 128, 320],
            3,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_packed_moe_tail_ab_profile() {
        run_packed_moe_tail_ab_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            &[8, 64, 128, 320],
            2,
        );
    }

    fn run_live_packed_moe_tail_phase_profile(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        n_runs: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[live-packed-moe-tail-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let f_shared = arch.expert_shared_feed_forward_length as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, g_w, u_w, d_w, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => (
                &g.post_attn_norm,
                &g.ffn_gate,
                &g.ffn_up,
                &g.ffn_down,
                g.ffn_moe.as_ref().expect("moe block"),
            ),
            crate::metal_forward::MetalBlock::Attn(a) => (
                &a.post_attn_norm,
                &a.ffn_gate,
                &a.ffn_up,
                &a.ffn_down,
                a.ffn_moe.as_ref().expect("moe block"),
            ),
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let moe_inner_pack = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let mixer_out_pack = scratch
            .mixer_out_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);
        let shared_ffn_gate_pack = scratch
            .moe_shared_ffn_gate_pack
            .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
        let shared_ffn_up_pack = scratch
            .moe_shared_ffn_up_pack
            .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
        let shared_ffn_inner_pack = scratch
            .moe_shared_ffn_inner_pack
            .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
        let shared_ffn_out_pack = scratch
            .moe_shared_ffn_out_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("warmup postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("warmup route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("warmup topk/shared");
            encode_moe_swiglu_q4_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &topk_idx_pack,
                &moe_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("warmup swiglu");
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_inner_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &mixer_out_pack,
                f_exp,
                h,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("warmup down");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                g_w,
                &h_pack,
                &shared_ffn_gate_pack,
                h,
                f_shared,
                chunk_p,
            )
            .expect("warmup shared gate");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                u_w,
                &h_pack,
                &shared_ffn_up_pack,
                h,
                f_shared,
                chunk_p,
            )
            .expect("warmup shared up");
            encode_silu_mul_f32(
                &ctx,
                enc,
                &shared_ffn_gate_pack,
                &shared_ffn_up_pack,
                &shared_ffn_inner_pack,
            )
            .expect("warmup shared silu");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                d_w,
                &shared_ffn_inner_pack,
                &shared_ffn_out_pack,
                f_shared,
                h,
                chunk_p,
            )
            .expect("warmup shared down");
            encode_axpy_rowwise_f32(
                &ctx,
                enc,
                &shared_ffn_out_pack,
                &shared_gate_pack,
                &mixer_out_pack,
                h,
                chunk_p,
            )
            .expect("warmup shared axpy");
            encode_add_inplace_f32(&ctx, enc, &x_pack, &mixer_out_pack).expect("warmup resid");
        });

        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("hist postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("hist route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("hist topk/shared");
        });
        let idxs = read_tensor_i32_f32buf(&topk_idx_pack);
        let mut counts = vec![0usize; n_expert];
        for &idx in &idxs {
            if idx >= 0 && (idx as usize) < n_expert {
                counts[idx as usize] += 1;
            }
        }
        let active = counts.iter().filter(|&&c| c > 0).count();
        let mut nonzero: Vec<usize> = counts.iter().copied().filter(|&c| c > 0).collect();
        nonzero.sort_unstable();
        let max_slots = nonzero.last().copied().unwrap_or(0);
        let p50 = nonzero.get(nonzero.len() / 2).copied().unwrap_or(0);
        let p90 = nonzero
            .get((nonzero.len().saturating_sub(1) * 9) / 10)
            .copied()
            .unwrap_or(0);
        let mean_active = if active > 0 {
            idxs.len() as f64 / active as f64
        } else {
            0.0
        };
        eprintln!(
            "[live-packed-moe-tail-{label}] expert bucket stats: slots={} active_experts={} mean_active={mean_active:.2} p50={} p90={} max={}",
            idxs.len(),
            active,
            p50,
            p90,
            max_slots
        );

        let mut postnorm_ms = 0.0f64;
        let mut route_ms = 0.0f64;
        let mut routed_swiglu_ms = 0.0f64;
        let mut routed_down_ms = 0.0f64;
        let mut shared_gateup_silu_ms = 0.0f64;
        let mut shared_down_ms = 0.0f64;
        let mut shared_axpy_resid_ms = 0.0f64;
        let mut one_cb_ms = 0.0f64;
        let mut wall_ms = 0.0f64;

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);
            let wall = Instant::now();

            postnorm_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
            });

            route_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("route logits");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared gate");
            });

            routed_swiglu_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_moe_swiglu_q4_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &topk_idx_pack,
                    &moe_inner_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("routed swiglu");
            });

            routed_down_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &mixer_out_pack,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("routed down");
            });

            shared_gateup_silu_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    g_w,
                    &h_pack,
                    &shared_ffn_gate_pack,
                    h,
                    f_shared,
                    chunk_p,
                )
                .expect("shared gate");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    u_w,
                    &h_pack,
                    &shared_ffn_up_pack,
                    h,
                    f_shared,
                    chunk_p,
                )
                .expect("shared up");
                encode_silu_mul_f32(
                    &ctx,
                    enc,
                    &shared_ffn_gate_pack,
                    &shared_ffn_up_pack,
                    &shared_ffn_inner_pack,
                )
                .expect("shared silu");
            });

            shared_down_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    d_w,
                    &shared_ffn_inner_pack,
                    &shared_ffn_out_pack,
                    f_shared,
                    h,
                    chunk_p,
                )
                .expect("shared down");
            });

            shared_axpy_resid_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_axpy_rowwise_f32(
                    &ctx,
                    enc,
                    &shared_ffn_out_pack,
                    &shared_gate_pack,
                    &mixer_out_pack,
                    h,
                    chunk_p,
                )
                .expect("shared axpy");
                encode_add_inplace_f32(&ctx, enc, &x_pack, &mixer_out_pack).expect("resid add");
            });

            one_cb_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("one-cb postnorm");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("one-cb route logits");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("one-cb topk/shared gate");
                encode_moe_swiglu_q4_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &topk_idx_pack,
                    &moe_inner_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("one-cb routed swiglu");
                encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &mixer_out_pack,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("one-cb routed down");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    g_w,
                    &h_pack,
                    &shared_ffn_gate_pack,
                    h,
                    f_shared,
                    chunk_p,
                )
                .expect("one-cb shared gate");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    u_w,
                    &h_pack,
                    &shared_ffn_up_pack,
                    h,
                    f_shared,
                    chunk_p,
                )
                .expect("one-cb shared up");
                encode_silu_mul_f32(
                    &ctx,
                    enc,
                    &shared_ffn_gate_pack,
                    &shared_ffn_up_pack,
                    &shared_ffn_inner_pack,
                )
                .expect("one-cb shared silu");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    d_w,
                    &shared_ffn_inner_pack,
                    &shared_ffn_out_pack,
                    f_shared,
                    h,
                    chunk_p,
                )
                .expect("one-cb shared down");
                encode_axpy_rowwise_f32(
                    &ctx,
                    enc,
                    &shared_ffn_out_pack,
                    &shared_gate_pack,
                    &mixer_out_pack,
                    h,
                    chunk_p,
                )
                .expect("one-cb shared axpy");
                encode_add_inplace_f32(&ctx, enc, &x_pack, &mixer_out_pack).expect("one-cb resid");
            });

            wall_ms += wall.elapsed().as_secs_f64() * 1e3;
        }

        let denom = n_runs as f64;
        let postnorm_ms = postnorm_ms / denom;
        let route_ms = route_ms / denom;
        let routed_swiglu_ms = routed_swiglu_ms / denom;
        let routed_down_ms = routed_down_ms / denom;
        let shared_gateup_silu_ms = shared_gateup_silu_ms / denom;
        let shared_down_ms = shared_down_ms / denom;
        let shared_axpy_resid_ms = shared_axpy_resid_ms / denom;
        let one_cb_ms = one_cb_ms / denom;
        let wall_ms = wall_ms / denom;
        let split_gpu = postnorm_ms
            + route_ms
            + routed_swiglu_ms
            + routed_down_ms
            + shared_gateup_silu_ms
            + shared_down_ms
            + shared_axpy_resid_ms;
        eprintln!(
            "[live-packed-moe-tail-{label}] chunk_p={chunk_p} avg wall={wall_ms:.2} ms split_gpu={split_gpu:.2} ms one_cb={one_cb_ms:.2} ms"
        );
        eprintln!(
            "[live-packed-moe-tail-{label}]   postnorm           {:6.2} ms ({:5.1}%)",
            postnorm_ms,
            postnorm_ms / split_gpu * 100.0
        );
        eprintln!(
            "[live-packed-moe-tail-{label}]   route+topk+sgate   {:6.2} ms ({:5.1}%)",
            route_ms,
            route_ms / split_gpu * 100.0
        );
        eprintln!(
            "[live-packed-moe-tail-{label}]   routed_swiglu      {:6.2} ms ({:5.1}%)",
            routed_swiglu_ms,
            routed_swiglu_ms / split_gpu * 100.0
        );
        eprintln!(
            "[live-packed-moe-tail-{label}]   routed_down_sum    {:6.2} ms ({:5.1}%)",
            routed_down_ms,
            routed_down_ms / split_gpu * 100.0
        );
        eprintln!(
            "[live-packed-moe-tail-{label}]   shared_gateup_silu {:6.2} ms ({:5.1}%)",
            shared_gateup_silu_ms,
            shared_gateup_silu_ms / split_gpu * 100.0
        );
        eprintln!(
            "[live-packed-moe-tail-{label}]   shared_down        {:6.2} ms ({:5.1}%)",
            shared_down_ms,
            shared_down_ms / split_gpu * 100.0
        );
        eprintln!(
            "[live-packed-moe-tail-{label}]   shared_axpy+resid  {:6.2} ms ({:5.1}%)",
            shared_axpy_resid_ms,
            shared_axpy_resid_ms / split_gpu * 100.0
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_live_packed_moe_tail_phase_profile() {
        run_live_packed_moe_tail_phase_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            320,
            3,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_live_packed_moe_tail_phase_profile() {
        run_live_packed_moe_tail_phase_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            320,
            2,
        );
    }

    fn run_live_grouped_moe_tail_phase_profile(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        n_runs: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[live-grouped-moe-tail-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let f_shared = arch.expert_shared_feed_forward_length as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, g_w, u_w, d_w, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => (
                &g.post_attn_norm,
                &g.ffn_gate,
                &g.ffn_up,
                &g.ffn_down,
                g.ffn_moe.as_ref().expect("moe block"),
            ),
            crate::metal_forward::MetalBlock::Attn(a) => (
                &a.post_attn_norm,
                &a.ffn_gate,
                &a.ffn_up,
                &a.ffn_down,
                a.ffn_moe.as_ref().expect("moe block"),
            ),
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let moe_group_count_pack = scratch
            .moe_group_count_pack
            .view_subrange(0, vec![n_expert as u64]);
        let moe_group_ids_pack = scratch
            .moe_group_ids_pack
            .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
        let moe_group_inner_pack = scratch
            .moe_group_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let moe_group_out_pack = scratch
            .moe_group_out_pack
            .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
        let mixer_out_pack = scratch
            .mixer_out_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);
        let shared_ffn_gate_pack = scratch
            .moe_shared_ffn_gate_pack
            .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
        let shared_ffn_up_pack = scratch
            .moe_shared_ffn_up_pack
            .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
        let shared_ffn_inner_pack = scratch
            .moe_shared_ffn_inner_pack
            .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
        let shared_ffn_out_pack = scratch
            .moe_shared_ffn_out_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();
        let fused_route_bucket = prefill_moe_route_bucket_fused_auto_enabled(h, chunk_p);

        let mut postnorm_ms = 0.0f64;
        let mut route_logits_ms = 0.0f64;
        let mut route_select_ms = 0.0f64;
        let mut route_bucket_ms = 0.0f64;
        let mut grouped_swiglu_ms = 0.0f64;
        let mut grouped_down_ms = 0.0f64;
        let mut grouped_reduce_ms = 0.0f64;
        let mut shared_gateup_silu_ms = 0.0f64;
        let mut shared_down_ms = 0.0f64;
        let mut shared_axpy_resid_ms = 0.0f64;
        let mut wall_ms = 0.0f64;

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);
            let wall = Instant::now();

            postnorm_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
            });

            route_logits_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_moe_route_logits_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("route logits");
            });

            if fused_route_bucket {
                route_select_ms += timed_gpu_cmd(&ctx, |enc| {
                    encode_fill_f32(&ctx, enc, &moe_group_count_pack, 0.0)
                        .expect("zero fused route counts");
                    crate::metal::encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
                        &ctx,
                        enc,
                        &router_probs_pack,
                        &moe.gate_inp_shexp,
                        &h_pack,
                        &topk_idx_pack,
                        &topk_weight_pack,
                        &shared_gate_pack,
                        &moe_group_count_pack,
                        &moe_group_ids_pack,
                        n_expert,
                        topk,
                        h,
                        chunk_p,
                    )
                    .expect("fused topk+bucket");
                });
            } else {
                route_select_ms += timed_gpu_cmd(&ctx, |enc| {
                    encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                        &ctx,
                        enc,
                        &router_probs_pack,
                        &moe.gate_inp_shexp,
                        &h_pack,
                        &topk_idx_pack,
                        &topk_weight_pack,
                        &shared_gate_pack,
                        n_expert,
                        topk,
                        h,
                        chunk_p,
                    )
                    .expect("topk/shared gate");
                });
                route_bucket_ms += timed_gpu_cmd(&ctx, |enc| {
                    crate::metal::encode_moe_route_bucket_slots_f32(
                        &ctx,
                        enc,
                        &topk_idx_pack,
                        &moe_group_count_pack,
                        &moe_group_ids_pack,
                        n_expert,
                        chunk_p,
                        topk,
                    )
                    .expect("bucket slots");
                });
            }

            grouped_swiglu_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &moe_group_inner_pack, 0.0).expect("zero grouped inner");
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &moe_group_count_pack,
                    &moe_group_ids_pack,
                    &moe_group_inner_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("grouped swiglu");
            });

            grouped_down_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &moe_group_out_pack, 0.0).expect("zero grouped out");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_group_inner_pack,
                    &moe_group_count_pack,
                    &moe_group_ids_pack,
                    &moe_group_out_pack,
                    f_exp,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("grouped down");
            });

            grouped_reduce_ms += timed_gpu_cmd(&ctx, |enc| {
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &moe_group_out_pack,
                    &topk_weight_pack,
                    &mixer_out_pack,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("grouped reduce");
            });

            shared_gateup_silu_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    g_w,
                    &h_pack,
                    &shared_ffn_gate_pack,
                    h,
                    f_shared,
                    chunk_p,
                )
                .expect("shared gate");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    u_w,
                    &h_pack,
                    &shared_ffn_up_pack,
                    h,
                    f_shared,
                    chunk_p,
                )
                .expect("shared up");
                encode_silu_mul_f32(
                    &ctx,
                    enc,
                    &shared_ffn_gate_pack,
                    &shared_ffn_up_pack,
                    &shared_ffn_inner_pack,
                )
                .expect("shared silu");
            });

            shared_down_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    d_w,
                    &shared_ffn_inner_pack,
                    &shared_ffn_out_pack,
                    f_shared,
                    h,
                    chunk_p,
                )
                .expect("shared down");
            });

            shared_axpy_resid_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_axpy_rowwise_f32(
                    &ctx,
                    enc,
                    &shared_ffn_out_pack,
                    &shared_gate_pack,
                    &mixer_out_pack,
                    h,
                    chunk_p,
                )
                .expect("shared axpy");
                encode_add_inplace_f32(&ctx, enc, &x_pack, &mixer_out_pack).expect("resid");
            });

            wall_ms += wall.elapsed().as_secs_f64() * 1e3;
        }

        let denom = n_runs as f64;
        let total = postnorm_ms
            + route_logits_ms
            + route_select_ms
            + route_bucket_ms
            + grouped_swiglu_ms
            + grouped_down_ms
            + grouped_reduce_ms
            + shared_gateup_silu_ms
            + shared_down_ms
            + shared_axpy_resid_ms;
        eprintln!(
            "[live-grouped-moe-tail-{label}] chunk_p={chunk_p} avg wall={:.2} ms split_gpu={:.2} ms",
            wall_ms / denom,
            total / denom,
        );
        eprintln!(
            "[live-grouped-moe-tail-{label}]   postnorm           {:6.2} ms ({:5.1}%)",
            postnorm_ms / denom,
            postnorm_ms / total * 100.0,
        );
        eprintln!(
            "[live-grouped-moe-tail-{label}]   route_logits       {:6.2} ms ({:5.1}%)",
            route_logits_ms / denom,
            route_logits_ms / total * 100.0,
        );
        eprintln!(
            "[live-grouped-moe-tail-{label}]   route_select       {:6.2} ms ({:5.1}%)",
            route_select_ms / denom,
            route_select_ms / total * 100.0,
        );
        eprintln!(
            "[live-grouped-moe-tail-{label}]   route_bucket       {:6.2} ms ({:5.1}%)",
            route_bucket_ms / denom,
            route_bucket_ms / total * 100.0,
        );
        eprintln!(
            "[live-grouped-moe-tail-{label}]   grouped_swiglu     {:6.2} ms ({:5.1}%)",
            grouped_swiglu_ms / denom,
            grouped_swiglu_ms / total * 100.0,
        );
        eprintln!(
            "[live-grouped-moe-tail-{label}]   grouped_down       {:6.2} ms ({:5.1}%)",
            grouped_down_ms / denom,
            grouped_down_ms / total * 100.0,
        );
        eprintln!(
            "[live-grouped-moe-tail-{label}]   grouped_reduce     {:6.2} ms ({:5.1}%)",
            grouped_reduce_ms / denom,
            grouped_reduce_ms / total * 100.0,
        );
        eprintln!(
            "[live-grouped-moe-tail-{label}]   shared_gateup_silu {:6.2} ms ({:5.1}%)",
            shared_gateup_silu_ms / denom,
            shared_gateup_silu_ms / total * 100.0,
        );
        eprintln!(
            "[live-grouped-moe-tail-{label}]   shared_down        {:6.2} ms ({:5.1}%)",
            shared_down_ms / denom,
            shared_down_ms / total * 100.0,
        );
        eprintln!(
            "[live-grouped-moe-tail-{label}]   shared_axpy+resid  {:6.2} ms ({:5.1}%)",
            shared_axpy_resid_ms / denom,
            shared_axpy_resid_ms / total * 100.0,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_live_grouped_moe_tail_phase_profile_320() {
        run_live_grouped_moe_tail_phase_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-320",
            320,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_live_grouped_moe_tail_phase_profile_320() {
        run_live_grouped_moe_tail_phase_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-320",
            320,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_live_grouped_moe_tail_phase_profile_512() {
        run_live_grouped_moe_tail_phase_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-512",
            512,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_live_grouped_moe_tail_phase_profile_512() {
        run_live_grouped_moe_tail_phase_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-512",
            512,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_live_grouped_moe_tail_phase_profile_1024() {
        run_live_grouped_moe_tail_phase_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-1024",
            1024,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_live_grouped_moe_tail_phase_profile_1024() {
        run_live_grouped_moe_tail_phase_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-1024",
            1024,
            2,
        );
    }

    fn run_moe_route_bucket_fused_oracle(model_path: &str, label: &str, chunk_p: usize) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[moe-route-bucket-fused-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };
        assert_eq!(moe.gate_exps.dtype, GgmlType::Q4_K, "gate dtype");
        assert_eq!(moe.up_exps.dtype, GgmlType::Q4_K, "up dtype");
        assert_eq!(moe.down_exps.dtype, GgmlType::Q5_K, "down dtype");

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let split_idx = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let split_w = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let split_gate = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let split_counts = scratch
            .moe_group_count_pack
            .view_subrange(0, vec![n_expert as u64]);
        let split_ids = scratch
            .moe_group_ids_pack
            .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
        let fused_idx = scratch
            .moe_group_slot_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let fused_w = scratch
            .moe_group_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let fused_gate = scratch
            .moe_group_token_idx_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let fused_counts =
            MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("fused_counts");
        let fused_ids =
            MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("fused_ids");
        let f_exp = arch.expert_feed_forward_length as usize;
        let split_inner = scratch
            .moe_group_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let fused_inner = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let split_out = scratch
            .moe_group_out_pack
            .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
        let fused_out = scratch
            .moe_expert_out_pack
            .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
        let split_reduced =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("split_reduced");
        let fused_reduced =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("fused_reduced");
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();
        write_tensor_f32(&x_pack, &x_init);

        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route logits");
        });

        let _ = timed_gpu_cmd(&ctx, |enc| {
            crate::metal::encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &split_idx,
                &split_w,
                &split_gate,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("split topk/shared");
            crate::metal::encode_moe_route_bucket_slots_f32(
                &ctx,
                enc,
                &split_idx,
                &split_counts,
                &split_ids,
                n_expert,
                chunk_p,
                topk,
            )
            .expect("split bucket");
        });

        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &fused_counts, 0.0).expect("zero fused counts");
            crate::metal::encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &fused_idx,
                &fused_w,
                &fused_gate,
                &fused_counts,
                &fused_ids,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("fused topk+bucket");
        });

        let split_idx_cpu = read_tensor_i32_f32buf(&split_idx);
        let fused_idx_cpu = read_tensor_i32_f32buf(&fused_idx);
        let split_w_cpu = read_tensor_f32(&split_w);
        let fused_w_cpu = read_tensor_f32(&fused_w);
        let split_gate_cpu = read_tensor_f32(&split_gate);
        let fused_gate_cpu = read_tensor_f32(&fused_gate);
        let split_counts_cpu = cpu_read_i32_f32buf(&split_counts);
        let fused_counts_cpu = cpu_read_i32_f32buf(&fused_counts);
        let split_ids_cpu = cpu_read_i32_f32buf(&split_ids);
        let fused_ids_cpu = cpu_read_i32_f32buf(&fused_ids);

        assert_eq!(split_idx_cpu, fused_idx_cpu, "topk idx mismatch");
        assert_eq!(split_counts_cpu, fused_counts_cpu, "bucket counts mismatch");
        let slot_count = chunk_p * topk;
        let total_count: usize = split_counts_cpu.iter().map(|&c| c.max(0) as usize).sum();
        let active_experts = split_counts_cpu.iter().filter(|&&c| c > 0).count();
        for expert in 0..n_expert {
            let count = split_counts_cpu[expert] as usize;
            let base = expert * chunk_p;
            let split_slice = &split_ids_cpu[base..base + count];
            let fused_slice = &fused_ids_cpu[base..base + count];
            for &slot in fused_slice {
                assert!(
                    slot >= 0 && (slot as usize) < slot_count,
                    "fused bucket slot out of range: expert={expert} slot={slot} slot_count={slot_count}"
                );
                assert_eq!(
                    fused_idx_cpu[slot as usize], expert as i32,
                    "fused bucket expert mismatch: expert={expert} slot={slot} slot_expert={} count={count}",
                    fused_idx_cpu[slot as usize]
                );
            }
            let mut split_sorted = split_slice.to_vec();
            let mut fused_sorted = fused_slice.to_vec();
            split_sorted.sort_unstable();
            fused_sorted.sort_unstable();
            assert_eq!(
                split_sorted,
                fused_sorted,
                "bucket content mismatch for expert {expert}: split={:?} fused={:?}",
                &split_sorted[..split_sorted.len().min(16)],
                &fused_sorted[..fused_sorted.len().min(16)]
            );
        }
        let cos_w = cosine_f32(&split_w_cpu, &fused_w_cpu);
        let cos_gate = cosine_f32(&split_gate_cpu, &fused_gate_cpu);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &split_inner, 0.0).expect("zero split inner");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &split_counts,
                &split_ids,
                &split_inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("split swiglu");
            encode_fill_f32(&ctx, enc, &split_out, 0.0).expect("zero split out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &split_inner,
                &split_counts,
                &split_ids,
                &split_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("split down");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &split_out,
                &split_w,
                &split_reduced,
                h,
                topk,
                chunk_p,
            )
            .expect("split reduce");
        });
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &fused_inner, 0.0).expect("zero fused inner");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &fused_counts,
                &fused_ids,
                &fused_inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("fused swiglu");
            encode_fill_f32(&ctx, enc, &fused_out, 0.0).expect("zero fused out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &fused_inner,
                &fused_counts,
                &fused_ids,
                &fused_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("fused down");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &fused_out,
                &fused_w,
                &fused_reduced,
                h,
                topk,
                chunk_p,
            )
            .expect("fused reduce");
        });
        let split_reduced_cpu = read_tensor_f32(&split_reduced);
        let fused_reduced_cpu = read_tensor_f32(&fused_reduced);
        let cos_reduced = cosine_f32(&split_reduced_cpu, &fused_reduced_cpu);
        eprintln!(
            "[moe-route-bucket-fused-{label}] total_count={} active_experts={} cos(topk_w)={cos_w:.6} cos(shared_gate)={cos_gate:.6} cos(reduced)={cos_reduced:.6}",
            total_count, active_experts,
        );
        assert!(cos_w > 0.999999, "topk_w cos too low: {cos_w}");
        assert!(cos_gate > 0.999999, "shared_gate cos too low: {cos_gate}");
        assert!(cos_reduced > 0.999999, "reduced cos too low: {cos_reduced}");
    }

    fn run_moe_route_logits_e8p32_oracle(model_path: &str, label: &str, chunk_p: usize) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[moe-route-e8p32-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };
        assert_eq!(
            moe.gate_inp.dtype,
            GgmlType::F32,
            "route logits oracle expects F32 router"
        );
        assert_eq!(
            n_expert % 8,
            0,
            "route logits oracle expects expert_count % 8 == 0"
        );
        assert_eq!(h % 4, 0, "route logits oracle expects hidden % 4 == 0");

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let probs_generic = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let probs_e8 =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * n_expert) as u64]).expect("probs_e8");
        let idx_generic = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let w_generic = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let gate_generic = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let idx_e8 = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk) as u64]).expect("idx_e8");
        let w_e8 = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk) as u64]).expect("w_e8");
        let gate_e8 = MetalTensor::zeros_f32(&ctx, vec![chunk_p as u64]).expect("gate_e8");
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();
        write_tensor_f32(&x_pack, &x_init);

        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
        });

        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &probs_generic,
                h,
                n_expert,
                chunk_p,
            )
            .expect("generic route logits");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &probs_generic,
                &moe.gate_inp_shexp,
                &h_pack,
                &idx_generic,
                &w_generic,
                &gate_generic,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("generic topk");
        });

        let _ = timed_gpu_cmd(&ctx, |enc| {
            crate::metal::encode_mat_mat_f32_router_e8p32(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &probs_e8,
                h,
                n_expert,
                chunk_p,
            )
            .expect("e8p32 route logits");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &probs_e8,
                &moe.gate_inp_shexp,
                &h_pack,
                &idx_e8,
                &w_e8,
                &gate_e8,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("e8p32 topk");
        });

        let probs_generic_cpu = read_tensor_f32(&probs_generic);
        let probs_e8_cpu = read_tensor_f32(&probs_e8);
        let idx_generic_cpu = read_tensor_i32_f32buf(&idx_generic);
        let idx_e8_cpu = read_tensor_i32_f32buf(&idx_e8);
        let w_generic_cpu = read_tensor_f32(&w_generic);
        let w_e8_cpu = read_tensor_f32(&w_e8);
        let gate_generic_cpu = read_tensor_f32(&gate_generic);
        let gate_e8_cpu = read_tensor_f32(&gate_e8);

        let probs_cos = cosine_f32(&probs_generic_cpu, &probs_e8_cpu);
        let w_cos = cosine_f32(&w_generic_cpu, &w_e8_cpu);
        let gate_cos = cosine_f32(&gate_generic_cpu, &gate_e8_cpu);
        let mismatch_count = idx_generic_cpu
            .iter()
            .zip(&idx_e8_cpu)
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "[moe-route-e8p32-{label}] probs_cos={probs_cos:.6} topk_mismatches={mismatch_count} w_cos={w_cos:.6} gate_cos={gate_cos:.6}"
        );
        assert_eq!(mismatch_count, 0, "topk mismatch count {mismatch_count}");
        assert!(
            probs_cos > 0.999999,
            "router probs cos too low: {probs_cos}"
        );
        assert!(w_cos > 0.999999, "topk weight cos too low: {w_cos}");
        assert!(gate_cos > 0.999999, "shared gate cos too low: {gate_cos}");
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_moe_route_bucket_fused_oracle_128() {
        run_moe_route_bucket_fused_oracle(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-128",
            128,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_moe_route_bucket_fused_oracle_320() {
        run_moe_route_bucket_fused_oracle(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-320",
            320,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_moe_route_bucket_fused_oracle_512() {
        run_moe_route_bucket_fused_oracle(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            512,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_moe_route_bucket_fused_oracle_512() {
        run_moe_route_bucket_fused_oracle(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            512,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_moe_route_logits_e8p32_oracle_128() {
        run_moe_route_logits_e8p32_oracle(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-128",
            128,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_moe_route_logits_e8p32_oracle_320() {
        run_moe_route_logits_e8p32_oracle(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-320",
            320,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_moe_route_logits_e8p32_oracle_512() {
        run_moe_route_logits_e8p32_oracle(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-512",
            512,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_moe_route_logits_e8p32_oracle_1024() {
        run_moe_route_logits_e8p32_oracle(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-1024",
            1024,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_moe_route_logits_e8p32_oracle_320() {
        run_moe_route_logits_e8p32_oracle(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-320",
            320,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_moe_route_logits_e8p32_oracle_512() {
        run_moe_route_logits_e8p32_oracle(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-512",
            512,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_moe_route_logits_e8p32_oracle_1024() {
        run_moe_route_logits_e8p32_oracle(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-1024",
            1024,
        );
    }

    fn run_gpu_compacted_grouped_down_profile(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        n_runs: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[gpu-grouped-down-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let slot_count = chunk_p * topk;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![slot_count as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![slot_count as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let moe_inner_pack = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(slot_count * f_exp) as u64]);
        let grouped_slot_out = scratch
            .moe_expert_out_pack
            .view_subrange(0, vec![(slot_count * h) as u64]);
        let slot_ref =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("slot ref");
        let current_reduced =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("current reduced");
        let grouped_reduced =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("grouped reduced");
        let counts = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("counts");
        let ids = MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids");
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        let mut current_ms = 0.0f64;
        let mut map_ms = 0.0f64;
        let mut grouped_down_ms = 0.0f64;
        let mut grouped_reduce_ms = 0.0f64;
        let mut grouped_total_ms = 0.0f64;
        let mut active_experts = 0usize;
        let mut max_count = 0i32;
        let mut total_count = 0i32;
        let mut covered_slots = 0usize;
        let mut count_mismatches = 0usize;
        let mut id_mismatches = 0usize;
        let mut cos_min = f64::INFINITY;
        let mut max_abs = 0.0f32;
        let mut slot_cos_min = f64::INFINITY;
        let mut slot_max_abs = 0.0f32;
        let mut lane_dot = [0.0f64; 4];
        let mut lane_na = [0.0f64; 4];
        let mut lane_nb = [0.0f64; 4];

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("route");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared");
                encode_moe_swiglu_q4_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &topk_idx_pack,
                    &moe_inner_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("swiglu");
            });

            current_ms += timed_gpu_cmd(&ctx, |enc| {
                crate::metal::encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &current_reduced,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("current down");
            });

            let total_start = Instant::now();
            map_ms += timed_gpu_cmd(&ctx, |enc| {
                crate::metal::encode_moe_route_bucket_slots_f32(
                    &ctx,
                    enc,
                    &topk_idx_pack,
                    &counts,
                    &ids,
                    n_expert,
                    chunk_p,
                    topk,
                )
                .expect("bucket slots");
            });
            let counts_cpu = read_tensor_i32_f32buf(&counts);
            active_experts = counts_cpu.iter().filter(|&&c| c > 0).count();
            max_count = counts_cpu.iter().copied().max().unwrap_or(0);
            total_count = counts_cpu.iter().sum();
            let ids_cpu = read_tensor_i32_f32buf(&ids);
            let topk_idx_cpu = read_tensor_i32_f32buf(&topk_idx_pack);
            let topk_weight_cpu = read_tensor_f32(&topk_weight_pack);
            let (cpu_groups, cpu_slot_ids, _, _) =
                build_expert_slot_groups_cpu(&topk_idx_cpu, &topk_weight_cpu, topk, n_expert);
            let mut cpu_counts = vec![0usize; n_expert];
            for group in &cpu_groups {
                cpu_counts[group.expert] = group.len;
            }
            count_mismatches = (0..n_expert)
                .filter(|&e| cpu_counts[e] as i32 != counts_cpu[e])
                .count();
            let mut cpu_ids = vec![-1i32; n_expert * chunk_p];
            for group in &cpu_groups {
                let dst = &mut cpu_ids[group.expert * chunk_p..group.expert * chunk_p + group.len];
                let src = &cpu_slot_ids[group.start..group.start + group.len];
                dst.copy_from_slice(src);
            }
            id_mismatches = 0;
            for expert in 0..n_expert {
                let count = counts_cpu[expert].max(0) as usize;
                for j in 0..count.min(chunk_p) {
                    if cpu_ids[expert * chunk_p + j] != ids_cpu[expert * chunk_p + j] {
                        id_mismatches += 1;
                    }
                }
            }
            let mut seen = vec![false; slot_count];
            for expert in 0..n_expert {
                let count = counts_cpu[expert].max(0) as usize;
                for j in 0..count.min(chunk_p) {
                    let slot = ids_cpu[expert * chunk_p + j];
                    if slot >= 0 && (slot as usize) < slot_count {
                        seen[slot as usize] = true;
                    }
                }
            }
            covered_slots = seen.iter().filter(|&&b| b).count();
            grouped_down_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &grouped_slot_out, 0.0).expect("zero grouped slot out");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner_pack,
                    &counts,
                    &ids,
                    &grouped_slot_out,
                    f_exp,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("grouped down slots");
            });
            grouped_reduce_ms += timed_gpu_cmd(&ctx, |enc| {
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &grouped_slot_out,
                    &topk_weight_pack,
                    &grouped_reduced,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("grouped reduce");
            });

            let _ = timed_gpu_cmd(&ctx, |enc| {
                for token in 0..chunk_p {
                    let inner_n = moe_inner_pack
                        .view_subrange((token * topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                    let idx_n =
                        topk_idx_pack.view_subrange((token * topk) as u64, vec![topk as u64]);
                    let out_n =
                        slot_ref.view_subrange((token * topk * h) as u64, vec![(topk * h) as u64]);
                    crate::metal::encode_moe_down_q5_K_f32(
                        &ctx,
                        enc,
                        &moe.down_exps,
                        &inner_n,
                        &idx_n,
                        &out_n,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )
                    .expect("slot ref down");
                }
            });
            grouped_total_ms += total_start.elapsed().as_secs_f64() * 1e3;

            let cur = read_tensor_f32(&current_reduced);
            let grp = read_tensor_f32(&grouped_reduced);
            cos_min = cos_min.min(cosine_f32(&cur, &grp));
            for i in 0..cur.len() {
                max_abs = max_abs.max((cur[i] - grp[i]).abs());
            }
            let slot_cur = read_tensor_f32(&slot_ref);
            let slot_grp = read_tensor_f32(&grouped_slot_out);
            slot_cos_min = slot_cos_min.min(cosine_f32(&slot_cur, &slot_grp));
            for i in 0..slot_cur.len() {
                slot_max_abs = slot_max_abs.max((slot_cur[i] - slot_grp[i]).abs());
            }
            for expert in 0..n_expert {
                let count = counts_cpu[expert].max(0) as usize;
                for j in 0..count.min(chunk_p) {
                    let slot = ids_cpu[expert * chunk_p + j] as usize;
                    if slot >= slot_count {
                        continue;
                    }
                    let lane = j % 4;
                    let off = slot * h;
                    for c in 0..h {
                        let a = slot_cur[off + c] as f64;
                        let b = slot_grp[off + c] as f64;
                        lane_dot[lane] += a * b;
                        lane_na[lane] += a * a;
                        lane_nb[lane] += b * b;
                    }
                }
            }
        }

        let lane_cos = |lane: usize| -> f64 {
            lane_dot[lane] / (lane_na[lane].sqrt() * lane_nb[lane].sqrt() + 1e-30)
        };

        let denom = n_runs as f64;
        eprintln!(
            "[gpu-grouped-down-{label}] chunk_p={chunk_p} active_experts={} max_count={} total_count={} covered_slots={} count_mismatches={} id_mismatches={} current={:.2} ms map={:.2} ms grouped_down={:.2} ms grouped_reduce={:.2} ms grouped_total={:.2} ms speedup={:.3} cos_min={:.6} max_abs={:.3e} slot_cos_min={:.6} slot_max_abs={:.3e} lane_cos=[{:.4},{:.4},{:.4},{:.4}]",
            active_experts,
            max_count,
            total_count,
            covered_slots,
            count_mismatches,
            id_mismatches,
            current_ms / denom,
            map_ms / denom,
            grouped_down_ms / denom,
            grouped_reduce_ms / denom,
            grouped_total_ms / denom,
            (current_ms / denom) / (grouped_total_ms / denom),
            cos_min,
            max_abs,
            slot_cos_min,
            slot_max_abs,
            lane_cos(0),
            lane_cos(1),
            lane_cos(2),
            lane_cos(3)
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_gpu_compacted_grouped_down_profile() {
        run_gpu_compacted_grouped_down_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            320,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_gpu_compacted_grouped_down_profile() {
        run_gpu_compacted_grouped_down_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            320,
            2,
        );
    }

    fn run_grouped_q5_down_vs_matmat_oracle(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        min_group: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[grouped-q5-oracle-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let moe_inner_pack = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared");
            encode_moe_swiglu_q4_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &topk_idx_pack,
                &moe_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("swiglu");
        });

        let topk_idx_cpu = read_tensor_i32_f32buf(&topk_idx_pack);
        let topk_weight_cpu = read_tensor_f32(&topk_weight_pack);
        let (groups, slot_ids, _, _) =
            build_expert_slot_groups_cpu(&topk_idx_cpu, &topk_weight_cpu, topk, n_expert);
        let group = groups
            .iter()
            .copied()
            .max_by_key(|g| g.len)
            .expect("at least one expert group");
        assert!(group.len >= min_group, "largest group too small");

        let count_t = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("count_t");
        let ids_t =
            MetalTensor::zeros_f32(&ctx, vec![(n_expert * group.len) as u64]).expect("ids_t");
        let inner_t =
            MetalTensor::zeros_f32(&ctx, vec![(group.len * f_exp) as u64]).expect("inner_t");
        let grouped_out_t =
            MetalTensor::zeros_f32(&ctx, vec![(group.len * h) as u64]).expect("grouped_out_t");
        let matmat_out_t =
            MetalTensor::zeros_f32(&ctx, vec![(group.len * h) as u64]).expect("matmat_out_t");
        let slot_ids_t = MetalTensor::zeros_f32(&ctx, vec![group.len as u64]).expect("slot_ids_t");

        let mut counts = vec![0i32; n_expert];
        counts[group.expert] = group.len as i32;
        let mut ids = vec![-1i32; n_expert * group.len];
        for j in 0..group.len {
            ids[group.expert * group.len + j] = j as i32;
        }
        cpu_write_i32_f32buf(&count_t, &counts);
        cpu_write_i32_f32buf(&ids_t, &ids);
        cpu_write_i32_f32buf(&slot_ids_t, &slot_ids[group.start..group.start + group.len]);

        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_get_rows_f32(
                &ctx,
                enc,
                &moe_inner_pack,
                &slot_ids_t,
                &inner_t,
                group.len,
                f_exp,
            )
            .expect("gather inner");
        });

        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &grouped_out_t, 0.0).expect("zero grouped out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &inner_t,
                &count_t,
                &ids_t,
                &grouped_out_t,
                f_exp,
                h,
                n_expert,
                group.len,
            )
            .expect("grouped slots");
        });

        let expert_bytes = moe.down_exps.n_bytes() / n_expert as u64;
        let expert_w = moe
            .down_exps
            .view_bytes(group.expert as u64 * expert_bytes, vec![(h * f_exp) as u64]);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &expert_w,
                &inner_t,
                &matmat_out_t,
                f_exp,
                h,
                group.len,
            )
            .expect("matmat oracle");
        });

        let grouped = read_tensor_f32(&grouped_out_t);
        let oracle = read_tensor_f32(&matmat_out_t);
        let cos = cosine_f32(&grouped, &oracle);
        let mut grouped_t = vec![0.0f32; grouped.len()];
        for q in 0..group.len {
            for r in 0..h {
                grouped_t[q * h + r] = grouped[r * group.len + q];
            }
        }
        let cos_t = cosine_f32(&grouped_t, &oracle);
        let mut max_abs = 0.0f32;
        for i in 0..grouped.len() {
            max_abs = max_abs.max((grouped[i] - oracle[i]).abs());
        }
        let tile_rows = h.min(64);
        let tile_cols = group.len.min(32);
        let mut row_mod8_abs = [0.0f64; 8];
        let mut row_mod8_n = [0usize; 8];
        let mut col_mod4_abs = [0.0f64; 4];
        let mut col_mod4_n = [0usize; 4];
        for q in 0..tile_cols {
            for r in 0..tile_rows {
                let d = (grouped[q * h + r] - oracle[q * h + r]).abs() as f64;
                row_mod8_abs[r % 8] += d;
                row_mod8_n[r % 8] += 1;
                col_mod4_abs[q % 4] += d;
                col_mod4_n[q % 4] += 1;
            }
        }
        let row_stat = |i: usize| row_mod8_abs[i] / row_mod8_n[i].max(1) as f64;
        let col_stat = |i: usize| col_mod4_abs[i] / col_mod4_n[i].max(1) as f64;
        eprintln!(
            "[grouped-q5-oracle-{label}] expert={} count={} cos={cos:.6} cos_t={cos_t:.6} max_abs={max_abs:.3e} row_mod8=[{:.2e},{:.2e},{:.2e},{:.2e},{:.2e},{:.2e},{:.2e},{:.2e}] col_mod4=[{:.2e},{:.2e},{:.2e},{:.2e}]",
            group.expert,
            group.len,
            row_stat(0),
            row_stat(1),
            row_stat(2),
            row_stat(3),
            row_stat(4),
            row_stat(5),
            row_stat(6),
            row_stat(7),
            col_stat(0),
            col_stat(1),
            col_stat(2),
            col_stat(3),
        );

        for &probe_n in &[17usize, 32] {
            if group.len < probe_n {
                continue;
            }
            let count_probe =
                MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("count_probe");
            let ids_probe =
                MetalTensor::zeros_f32(&ctx, vec![(n_expert * probe_n) as u64]).expect("ids_probe");
            let inner_probe = inner_t.view_subrange(0, vec![(probe_n * f_exp) as u64]);
            let grouped_probe =
                MetalTensor::zeros_f32(&ctx, vec![(probe_n * h) as u64]).expect("grouped_probe");
            let oracle_probe =
                MetalTensor::zeros_f32(&ctx, vec![(probe_n * h) as u64]).expect("oracle_probe");
            let mut probe_counts = vec![0i32; n_expert];
            probe_counts[group.expert] = probe_n as i32;
            let mut probe_ids = vec![-1i32; n_expert * probe_n];
            for j in 0..probe_n {
                probe_ids[group.expert * probe_n + j] = j as i32;
            }
            cpu_write_i32_f32buf(&count_probe, &probe_counts);
            cpu_write_i32_f32buf(&ids_probe, &probe_ids);
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &grouped_probe, 0.0).expect("zero grouped probe");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &inner_probe,
                    &count_probe,
                    &ids_probe,
                    &grouped_probe,
                    f_exp,
                    h,
                    n_expert,
                    probe_n,
                )
                .expect("grouped probe");
            });
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &expert_w,
                    &inner_probe,
                    &oracle_probe,
                    f_exp,
                    h,
                    probe_n,
                )
                .expect("oracle probe");
            });
            let gp = read_tensor_f32(&grouped_probe);
            let op = read_tensor_f32(&oracle_probe);
            let probe_cos = cosine_f32(&gp, &op);
            let split16_cos = if probe_n >= 32 {
                let mut a0 = Vec::with_capacity(16 * h);
                let mut b0 = Vec::with_capacity(16 * h);
                let mut a1 = Vec::with_capacity(16 * h);
                let mut b1 = Vec::with_capacity(16 * h);
                for q in 0..16 {
                    a0.extend_from_slice(&gp[q * h..(q + 1) * h]);
                    b0.extend_from_slice(&op[q * h..(q + 1) * h]);
                }
                for q in 16..32 {
                    a1.extend_from_slice(&gp[q * h..(q + 1) * h]);
                    b1.extend_from_slice(&op[q * h..(q + 1) * h]);
                }
                format!(
                    " first16={:.6} second16={:.6}",
                    cosine_f32(&a0, &b0),
                    cosine_f32(&a1, &b1)
                )
            } else {
                String::new()
            };
            let mut probe_max = 0.0f32;
            for i in 0..gp.len() {
                probe_max = probe_max.max((gp[i] - op[i]).abs());
            }
            eprintln!(
                "[grouped-q5-oracle-{label}] probe_n={} cos={probe_cos:.6} max_abs={probe_max:.3e}{}",
                probe_n, split16_cos,
            );
        }
    }

    fn run_grouped_q5_swiglu_vs_matmat_oracle(
        model_path: &str,
        label: &str,
        layer_idx: usize,
        chunk_p: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[grouped-q5-swiglu-oracle-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;

        let block = mf.model.blocks.get(layer_idx).expect("layer exists");
        let moe = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => g.ffn_moe.as_ref().expect("moe block"),
            crate::metal_forward::MetalBlock::Attn(a) => a.ffn_moe.as_ref().expect("moe block"),
        };
        assert_eq!(moe.gate_exps.dtype, GgmlType::Q5_K, "gate dtype");
        assert_eq!(moe.up_exps.dtype, GgmlType::Q5_K, "up dtype");

        let slot_count = chunk_p * topk;
        let mut count_plan = vec![1usize, 15, 16, 17, 31, 32, 33, 48, 64, 64, 64, 64];
        let used: usize = count_plan.iter().sum();
        assert!(used <= slot_count, "count plan too large");
        count_plan.push(slot_count - used);
        assert!(
            count_plan.iter().all(|&c| c <= chunk_p),
            "count plan exceeds grouped id stride"
        );

        let selected_experts: Vec<usize> = (0..count_plan.len())
            .map(|i| (17 * i + 3) % n_expert)
            .collect();
        let mut counts = vec![0i32; n_expert];
        let mut ids = vec![-1i32; n_expert * chunk_p];
        let slots: Vec<i32> = (0..slot_count)
            .map(|i| ((i * 37) % slot_count) as i32)
            .collect();
        let mut cursor = 0usize;
        let mut expert_slots: Vec<(usize, Vec<i32>)> = Vec::new();
        for (&expert, &count) in selected_experts.iter().zip(count_plan.iter()) {
            counts[expert] = count as i32;
            let mut group_slots = Vec::with_capacity(count);
            for j in 0..count {
                let slot = slots[cursor + j];
                ids[expert * chunk_p + j] = slot;
                group_slots.push(slot);
            }
            cursor += count;
            expert_slots.push((expert, group_slots));
        }
        assert_eq!(cursor, slot_count);

        let h_pack = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("h_pack");
        let counts_t = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("counts");
        let ids_t = MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids");
        let actual_inner =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("actual_inner");
        let h_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| (((i * 13 + 7) % 97) as f32 - 48.0) * 0.0075)
            .collect();
        write_tensor_f32(&h_pack, &h_init);
        cpu_write_i32_f32buf(&counts_t, &counts);
        cpu_write_i32_f32buf(&ids_t, &ids);

        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &actual_inner, -777.0).expect("poison actual");
            crate::metal::encode_moe_swiglu_q5_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &counts_t,
                &ids_t,
                &actual_inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("grouped q5 swiglu");
        });

        let gate_bytes = moe.gate_exps.n_bytes() / n_expert as u64;
        let up_bytes = moe.up_exps.n_bytes() / n_expert as u64;
        let mut expected = vec![0.0f32; slot_count * f_exp];
        for (expert, group_slots) in &expert_slots {
            if group_slots.is_empty() {
                continue;
            }
            let n = group_slots.len();
            let token_ids: Vec<i32> = group_slots.iter().map(|slot| slot / topk as i32).collect();
            let token_ids_t = MetalTensor::zeros_f32(&ctx, vec![n as u64]).expect("token_ids");
            let h_group = MetalTensor::zeros_f32(&ctx, vec![(n * h) as u64]).expect("h_group");
            let gate_out =
                MetalTensor::zeros_f32(&ctx, vec![(n * f_exp) as u64]).expect("gate_out");
            let up_out = MetalTensor::zeros_f32(&ctx, vec![(n * f_exp) as u64]).expect("up_out");
            let inner_group =
                MetalTensor::zeros_f32(&ctx, vec![(n * f_exp) as u64]).expect("inner_group");
            cpu_write_i32_f32buf(&token_ids_t, &token_ids);
            let gate_w = moe
                .gate_exps
                .view_bytes(*expert as u64 * gate_bytes, vec![(h * f_exp) as u64]);
            let up_w = moe
                .up_exps
                .view_bytes(*expert as u64 * up_bytes, vec![(h * f_exp) as u64]);
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_get_rows_f32(&ctx, enc, &h_pack, &token_ids_t, &h_group, n, h)
                    .expect("gather h");
                encode_mat_mat_dispatch(&ctx, enc, &gate_w, &h_group, &gate_out, h, f_exp, n)
                    .expect("gate oracle");
                encode_mat_mat_dispatch(&ctx, enc, &up_w, &h_group, &up_out, h, f_exp, n)
                    .expect("up oracle");
                encode_silu_mul_f32(&ctx, enc, &gate_out, &up_out, &inner_group)
                    .expect("silu oracle");
            });
            let group_cpu = read_tensor_f32(&inner_group);
            for (j, &slot) in group_slots.iter().enumerate() {
                let dst = slot as usize * f_exp;
                let src = j * f_exp;
                expected[dst..dst + f_exp].copy_from_slice(&group_cpu[src..src + f_exp]);
            }
        }

        let actual = read_tensor_f32(&actual_inner);
        let cos = cosine_f32(&actual, &expected);
        let mut max_abs = 0.0f32;
        let mut poison_count = 0usize;
        for (a, e) in actual.iter().zip(expected.iter()) {
            max_abs = max_abs.max((a - e).abs());
            if *a == -777.0 {
                poison_count += 1;
            }
        }
        eprintln!(
            "[grouped-q5-swiglu-oracle-{label}] layer={layer_idx} chunk_p={chunk_p} experts={} slots={} cos={cos:.6} max_abs={max_abs:.3e} poison_count={poison_count}",
            expert_slots.len(),
            slot_count,
        );
        assert_eq!(poison_count, 0, "grouped q5 swiglu left poisoned slots");
        assert!(cos > 0.999, "grouped q5 swiglu cos too low: {cos}");
        assert!(
            max_abs < 2.5e-1,
            "grouped q5 swiglu max_abs too high: {max_abs}"
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_q5_down_vs_matmat_oracle() {
        run_grouped_q5_down_vs_matmat_oracle(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            320,
            32,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_q5_swiglu_vs_matmat_oracle_layer46() {
        run_grouped_q5_swiglu_vs_matmat_oracle(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-layer46",
            46,
            64,
        );
    }

    fn run_grouped_q4_swiglu_vs_packed_oracle(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        min_group: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[grouped-q4-oracle-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let moe_inner_pack = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared");
            encode_moe_swiglu_q4_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &topk_idx_pack,
                &moe_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("packed swiglu");
        });

        let topk_idx_cpu = read_tensor_i32_f32buf(&topk_idx_pack);
        let topk_weight_cpu = read_tensor_f32(&topk_weight_pack);
        let (groups, slot_ids, _, _) =
            build_expert_slot_groups_cpu(&topk_idx_cpu, &topk_weight_cpu, topk, n_expert);
        let group = groups
            .iter()
            .copied()
            .max_by_key(|g| g.len)
            .expect("at least one expert group");
        assert!(group.len >= min_group, "largest group too small");

        let count_t = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("count_t");
        let ids_t = MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids_t");
        let grouped_inner_out = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk * f_exp) as u64])
            .expect("grouped_inner_out");
        let slot_ids_t = MetalTensor::zeros_f32(&ctx, vec![group.len as u64]).expect("slot_ids_t");
        let packed_probe =
            MetalTensor::zeros_f32(&ctx, vec![(group.len * f_exp) as u64]).expect("packed_probe");
        let grouped_probe =
            MetalTensor::zeros_f32(&ctx, vec![(group.len * f_exp) as u64]).expect("grouped_probe");

        let mut counts = vec![0i32; n_expert];
        counts[group.expert] = group.len as i32;
        let mut ids = vec![-1i32; n_expert * chunk_p];
        for j in 0..group.len {
            ids[group.expert * chunk_p + j] = slot_ids[group.start + j];
        }
        cpu_write_i32_f32buf(&count_t, &counts);
        cpu_write_i32_f32buf(&ids_t, &ids);
        cpu_write_i32_f32buf(&slot_ids_t, &slot_ids[group.start..group.start + group.len]);

        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &grouped_inner_out, 0.0).expect("zero grouped q4 out");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &count_t,
                &ids_t,
                &grouped_inner_out,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("grouped q4 swiglu");
            encode_get_rows_f32(
                &ctx,
                enc,
                &moe_inner_pack,
                &slot_ids_t,
                &packed_probe,
                group.len,
                f_exp,
            )
            .expect("gather packed probe");
            encode_get_rows_f32(
                &ctx,
                enc,
                &grouped_inner_out,
                &slot_ids_t,
                &grouped_probe,
                group.len,
                f_exp,
            )
            .expect("gather grouped probe");
        });

        let packed = read_tensor_f32(&packed_probe);
        let grouped = read_tensor_f32(&grouped_probe);
        let cos = cosine_f32(&packed, &grouped);
        let mut max_abs = 0.0f32;
        for i in 0..packed.len() {
            max_abs = max_abs.max((packed[i] - grouped[i]).abs());
        }
        eprintln!(
            "[grouped-q4-oracle-{label}] expert={} count={} cos={cos:.6} max_abs={max_abs:.3e}",
            group.expert, group.len,
        );

        for &probe_n in &[16usize, 17, 32] {
            if group.len < probe_n {
                continue;
            }
            let count_probe =
                MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("count_probe");
            let ids_probe =
                MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids_probe");
            let grouped_probe_all =
                MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk * f_exp) as u64])
                    .expect("grouped_probe_all");
            let packed_probe_n = MetalTensor::zeros_f32(&ctx, vec![(probe_n * f_exp) as u64])
                .expect("packed_probe_n");
            let grouped_probe_n = MetalTensor::zeros_f32(&ctx, vec![(probe_n * f_exp) as u64])
                .expect("grouped_probe_n");
            let slot_ids_probe =
                MetalTensor::zeros_f32(&ctx, vec![probe_n as u64]).expect("slot_ids_probe");
            let mut probe_counts = vec![0i32; n_expert];
            probe_counts[group.expert] = probe_n as i32;
            let mut probe_ids = vec![-1i32; n_expert * chunk_p];
            for j in 0..probe_n {
                probe_ids[group.expert * chunk_p + j] = slot_ids[group.start + j];
            }
            cpu_write_i32_f32buf(&count_probe, &probe_counts);
            cpu_write_i32_f32buf(&ids_probe, &probe_ids);
            cpu_write_i32_f32buf(
                &slot_ids_probe,
                &slot_ids[group.start..group.start + probe_n],
            );

            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &grouped_probe_all, 0.0)
                    .expect("zero grouped q4 probe all");
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &count_probe,
                    &ids_probe,
                    &grouped_probe_all,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("grouped q4 probe");
                encode_get_rows_f32(
                    &ctx,
                    enc,
                    &moe_inner_pack,
                    &slot_ids_probe,
                    &packed_probe_n,
                    probe_n,
                    f_exp,
                )
                .expect("gather packed probe n");
                encode_get_rows_f32(
                    &ctx,
                    enc,
                    &grouped_probe_all,
                    &slot_ids_probe,
                    &grouped_probe_n,
                    probe_n,
                    f_exp,
                )
                .expect("gather grouped probe n");
            });

            let packed_n = read_tensor_f32(&packed_probe_n);
            let grouped_n = read_tensor_f32(&grouped_probe_n);
            let probe_cos = cosine_f32(&packed_n, &grouped_n);
            let mut probe_max = 0.0f32;
            for i in 0..packed_n.len() {
                probe_max = probe_max.max((packed_n[i] - grouped_n[i]).abs());
            }
            eprintln!(
                "[grouped-q4-oracle-{label}] probe_n={} cos={probe_cos:.6} max_abs={probe_max:.3e}",
                probe_n,
            );
        }
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_q4_swiglu_vs_packed_oracle() {
        run_grouped_q4_swiglu_vs_packed_oracle(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            320,
            32,
        );
    }

    fn run_grouped_swiglu_down_backend_profile(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        n_runs: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[grouped-swiglu-down-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let slot_count = chunk_p * topk;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let atomic_topk_idx_pack = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk) as u64])
            .expect("atomic_topk_idx_pack");
        let atomic_topk_weight_pack = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk) as u64])
            .expect("atomic_topk_weight_pack");
        let atomic_shared_gate_pack =
            MetalTensor::zeros_f32(&ctx, vec![chunk_p as u64]).expect("atomic_shared_gate_pack");
        let moe_inner_pack = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let packed_reduced =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("packed_reduced");
        let counts = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("counts");
        let ids = MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids");
        let atomic_counts =
            MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("atomic_counts");
        let atomic_ids =
            MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("atomic_ids");
        let live_grouped_inner = MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64])
            .expect("live_grouped_inner");
        let live_grouped_slot_out = MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64])
            .expect("live_grouped_slot_out");
        let live_grouped_reduced =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("live_grouped_reduced");
        let atomic_grouped_inner = MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64])
            .expect("atomic_grouped_inner");
        let atomic_grouped_slot_out = MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64])
            .expect("atomic_grouped_slot_out");
        let atomic_grouped_reduced = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64])
            .expect("atomic_grouped_reduced");
        let fused_gate_up =
            fuse_q4k_gate_up_expert_banks(&ctx, &moe.gate_exps, &moe.up_exps, h, f_exp, n_expert);
        let fused_grouped_inner = MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64])
            .expect("fused_grouped_inner");
        let fused_grouped_slot_out = MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64])
            .expect("fused_grouped_slot_out");
        let fused_grouped_reduced = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64])
            .expect("fused_grouped_reduced");
        let grouped_gate =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("grouped_gate");
        let grouped_up =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("grouped_up");
        let grouped_inner =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("grouped_inner");
        let grouped_slot_out =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("grouped_slot_out");
        let grouped_reduced =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("grouped_reduced");
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        let mut packed_tail_ms = 0.0f64;
        let mut live_grouped_tail_ms = 0.0f64;
        let mut atomic_grouped_tail_ms = 0.0f64;
        let mut fused_grouped_tail_ms = 0.0f64;
        let mut map_ms = 0.0f64;
        let mut grouped_swiglu_gpu_ms = 0.0f64;
        let mut grouped_down_ms = 0.0f64;
        let mut grouped_reduce_ms = 0.0f64;
        let mut grouped_total_wall_ms = 0.0f64;
        let mut cos_min = f64::INFINITY;
        let mut max_abs = 0.0f32;
        let mut atomic_cos_min = f64::INFINITY;
        let mut atomic_max_abs = 0.0f32;
        let mut fused_cos_min = f64::INFINITY;
        let mut fused_max_abs = 0.0f32;
        let mut active_experts = 0usize;
        let mut max_count = 0usize;
        let mut p50_count = 0usize;
        let mut p90_count = 0usize;
        let mut experts_ge16 = 0usize;
        let mut experts_ge32 = 0usize;
        let mut experts_ge48 = 0usize;
        let mut scan_back_edges = 0usize;
        let mut scan_edges = 0usize;
        let mut atomic_back_edges = 0usize;
        let mut atomic_edges = 0usize;

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("route");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared");
                encode_fill_f32(&ctx, enc, &atomic_counts, 0.0).expect("zero atomic counts");
                crate::metal::encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &atomic_topk_idx_pack,
                    &atomic_topk_weight_pack,
                    &atomic_shared_gate_pack,
                    &atomic_counts,
                    &atomic_ids,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared fused bucket");
            });

            packed_tail_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_moe_swiglu_q4_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &topk_idx_pack,
                    &moe_inner_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("current swiglu");
                crate::metal::encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &packed_reduced,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("current down");
            });

            map_ms += timed_gpu_cmd(&ctx, |enc| {
                crate::metal::encode_moe_route_bucket_slots_f32(
                    &ctx,
                    enc,
                    &topk_idx_pack,
                    &counts,
                    &ids,
                    n_expert,
                    chunk_p,
                    topk,
                )
                .expect("bucket slots");
            });

            live_grouped_tail_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &live_grouped_inner, 0.0)
                    .expect("zero live grouped inner");
                if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
                    if let Some(hot_threshold) = prefill_moe_hot_expert_min_slots() {
                        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &h_pack,
                            &counts,
                            &ids,
                            &live_grouped_inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            hot_threshold as u32,
                            i32::MAX as u32,
                        )
                        .expect("live grouped hot n32");
                        if hot_threshold > 0 {
                            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
                                &ctx,
                                enc,
                                &moe.gate_exps,
                                &moe.up_exps,
                                &h_pack,
                                &counts,
                                &ids,
                                &live_grouped_inner,
                                h,
                                f_exp,
                                n_expert,
                                topk,
                                chunk_p,
                                0,
                                hot_threshold.saturating_sub(1) as u32,
                            )
                            .expect("live grouped cold n16");
                        }
                    } else {
                        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &h_pack,
                            &counts,
                            &ids,
                            &live_grouped_inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                        )
                        .expect("live grouped n16 fallback");
                    }
                } else {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &live_grouped_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("live grouped n16");
                }
                encode_fill_f32(&ctx, enc, &live_grouped_slot_out, 0.0)
                    .expect("zero live grouped slot out");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &live_grouped_inner,
                    &counts,
                    &ids,
                    &live_grouped_slot_out,
                    f_exp,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("live grouped down");
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &live_grouped_slot_out,
                    &topk_weight_pack,
                    &live_grouped_reduced,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("live grouped reduce");
            });

            atomic_grouped_tail_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &atomic_grouped_inner, 0.0)
                    .expect("zero atomic grouped inner");
                if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
                    if let Some(hot_threshold) = prefill_moe_hot_expert_min_slots() {
                        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &h_pack,
                            &atomic_counts,
                            &atomic_ids,
                            &atomic_grouped_inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            hot_threshold as u32,
                            i32::MAX as u32,
                        )
                        .expect("atomic grouped hot n32");
                        if hot_threshold > 0 {
                            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
                                &ctx,
                                enc,
                                &moe.gate_exps,
                                &moe.up_exps,
                                &h_pack,
                                &atomic_counts,
                                &atomic_ids,
                                &atomic_grouped_inner,
                                h,
                                f_exp,
                                n_expert,
                                topk,
                                chunk_p,
                                0,
                                hot_threshold.saturating_sub(1) as u32,
                            )
                            .expect("atomic grouped cold n16");
                        }
                    } else {
                        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &h_pack,
                            &atomic_counts,
                            &atomic_ids,
                            &atomic_grouped_inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                        )
                        .expect("atomic grouped n16 fallback");
                    }
                } else {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &atomic_counts,
                        &atomic_ids,
                        &atomic_grouped_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("atomic grouped n16");
                }
                encode_fill_f32(&ctx, enc, &atomic_grouped_slot_out, 0.0)
                    .expect("zero atomic grouped slot out");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &atomic_grouped_inner,
                    &atomic_counts,
                    &atomic_ids,
                    &atomic_grouped_slot_out,
                    f_exp,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("atomic grouped down");
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &atomic_grouped_slot_out,
                    &atomic_topk_weight_pack,
                    &atomic_grouped_reduced,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("atomic grouped reduce");
            });

            fused_grouped_tail_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &fused_grouped_inner, 0.0)
                    .expect("zero fused grouped inner");
                if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
                    if let Some(hot_threshold) = prefill_moe_hot_expert_min_slots() {
                        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n32_range(
                            &ctx,
                            enc,
                            &fused_gate_up,
                            &h_pack,
                            &counts,
                            &ids,
                            &fused_grouped_inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            hot_threshold as u32,
                            i32::MAX as u32,
                        )
                        .expect("fused grouped hot n32");
                        if hot_threshold > 0 {
                            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n16_range(
                                &ctx,
                                enc,
                                &fused_gate_up,
                                &h_pack,
                                &counts,
                                &ids,
                                &fused_grouped_inner,
                                h,
                                f_exp,
                                n_expert,
                                topk,
                                chunk_p,
                                0,
                                hot_threshold.saturating_sub(1) as u32,
                            )
                            .expect("fused grouped cold n16");
                        }
                    } else {
                        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n16_range(
                            &ctx,
                            enc,
                            &fused_gate_up,
                            &h_pack,
                            &counts,
                            &ids,
                            &fused_grouped_inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            0,
                            i32::MAX as u32,
                        )
                        .expect("fused grouped n16 fallback");
                    }
                } else {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n16_range(
                        &ctx,
                        enc,
                        &fused_gate_up,
                        &h_pack,
                        &counts,
                        &ids,
                        &fused_grouped_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                        0,
                        i32::MAX as u32,
                    )
                    .expect("fused grouped n16");
                }
                encode_fill_f32(&ctx, enc, &fused_grouped_slot_out, 0.0)
                    .expect("zero fused grouped slot out");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &fused_grouped_inner,
                    &counts,
                    &ids,
                    &fused_grouped_slot_out,
                    f_exp,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("fused grouped down");
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &fused_grouped_slot_out,
                    &topk_weight_pack,
                    &fused_grouped_reduced,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("fused grouped reduce");
            });

            let wall = Instant::now();
            let counts_cpu = read_tensor_i32_f32buf(&counts);
            let ids_cpu = read_tensor_i32_f32buf(&ids);
            let atomic_counts_cpu = read_tensor_i32_f32buf(&atomic_counts);
            let atomic_ids_cpu = read_tensor_i32_f32buf(&atomic_ids);
            let mut active_counts: Vec<usize> = counts_cpu
                .iter()
                .filter_map(|&c| (c > 0).then_some(c as usize))
                .collect();
            active_counts.sort_unstable();
            active_experts = active_counts.len();
            max_count = counts_cpu.iter().copied().max().unwrap_or(0).max(0) as usize;
            if !active_counts.is_empty() {
                p50_count = active_counts[active_counts.len() / 2];
                p90_count =
                    active_counts[(active_counts.len() * 9 / 10).min(active_counts.len() - 1)];
            }
            experts_ge16 = active_counts.iter().filter(|&&c| c >= 16).count();
            experts_ge32 = active_counts.iter().filter(|&&c| c >= 32).count();
            experts_ge48 = active_counts.iter().filter(|&&c| c >= 48).count();
            let bucket_back_edges = |counts: &[i32], ids: &[i32]| -> (usize, usize) {
                let mut back = 0usize;
                let mut edges = 0usize;
                for expert in 0..n_expert {
                    let count = counts[expert].max(0) as usize;
                    let mut prev_token = None;
                    for j in 0..count.min(chunk_p) {
                        let slot = ids[expert * chunk_p + j].max(0) as usize;
                        let token = slot / topk;
                        if let Some(prev) = prev_token {
                            edges += 1;
                            if token < prev {
                                back += 1;
                            }
                        }
                        prev_token = Some(token);
                    }
                }
                (back, edges)
            };
            (scan_back_edges, scan_edges) = bucket_back_edges(&counts_cpu, &ids_cpu);
            (atomic_back_edges, atomic_edges) =
                bucket_back_edges(&atomic_counts_cpu, &atomic_ids_cpu);

            grouped_swiglu_gpu_ms += timed_gpu_cmd(&ctx, |enc| {
                if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
                    if let Some(hot_threshold) = prefill_moe_hot_expert_min_slots() {
                        crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n32_range(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &h_pack,
                            &counts,
                            &ids,
                            &grouped_gate,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            hot_threshold as u32,
                            i32::MAX as u32,
                        )
                        .expect("split gate hot n32");
                        crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n32_range(
                            &ctx,
                            enc,
                            &moe.up_exps,
                            &h_pack,
                            &counts,
                            &ids,
                            &grouped_up,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            hot_threshold as u32,
                            i32::MAX as u32,
                        )
                        .expect("split up hot n32");
                        if hot_threshold > 0 {
                            crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n16_range(
                                &ctx,
                                enc,
                                &moe.gate_exps,
                                &h_pack,
                                &counts,
                                &ids,
                                &grouped_gate,
                                h,
                                f_exp,
                                n_expert,
                                topk,
                                chunk_p,
                                0,
                                hot_threshold.saturating_sub(1) as u32,
                            )
                            .expect("split gate cold n16");
                            crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n16_range(
                                &ctx,
                                enc,
                                &moe.up_exps,
                                &h_pack,
                                &counts,
                                &ids,
                                &grouped_up,
                                h,
                                f_exp,
                                n_expert,
                                topk,
                                chunk_p,
                                0,
                                hot_threshold.saturating_sub(1) as u32,
                            )
                            .expect("split up cold n16");
                        }
                    } else {
                        crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n16(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &h_pack,
                            &counts,
                            &ids,
                            &grouped_gate,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                        )
                        .expect("split gate n16 fallback");
                        crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n16(
                            &ctx,
                            enc,
                            &moe.up_exps,
                            &h_pack,
                            &counts,
                            &ids,
                            &grouped_up,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                        )
                        .expect("split up n16 fallback");
                    }
                } else {
                    crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n16(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &grouped_gate,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("split gate n16");
                    crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n16(
                        &ctx,
                        enc,
                        &moe.up_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &grouped_up,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("split up n16");
                }
                encode_silu_mul_f32(&ctx, enc, &grouped_gate, &grouped_up, &grouped_inner)
                    .expect("grouped silu");
            });

            grouped_down_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &grouped_slot_out, 0.0).expect("zero grouped slot out");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &grouped_inner,
                    &counts,
                    &ids,
                    &grouped_slot_out,
                    f_exp,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("grouped down slots");
            });
            grouped_reduce_ms += timed_gpu_cmd(&ctx, |enc| {
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &grouped_slot_out,
                    &topk_weight_pack,
                    &grouped_reduced,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("grouped reduce");
            });
            grouped_total_wall_ms += wall.elapsed().as_secs_f64() * 1e3;

            let cur = read_tensor_f32(&live_grouped_reduced);
            let grp = read_tensor_f32(&grouped_reduced);
            cos_min = cos_min.min(cosine_f32(&cur, &grp));
            for i in 0..cur.len() {
                max_abs = max_abs.max((cur[i] - grp[i]).abs());
            }
            let atomic = read_tensor_f32(&atomic_grouped_reduced);
            atomic_cos_min = atomic_cos_min.min(cosine_f32(&cur, &atomic));
            for i in 0..cur.len() {
                atomic_max_abs = atomic_max_abs.max((cur[i] - atomic[i]).abs());
            }
            let fus = read_tensor_f32(&fused_grouped_reduced);
            fused_cos_min = fused_cos_min.min(cosine_f32(&cur, &fus));
            for i in 0..cur.len() {
                fused_max_abs = fused_max_abs.max((cur[i] - fus[i]).abs());
            }
        }

        let denom = n_runs as f64;
        let split_down_reduce_ms = (grouped_down_ms + grouped_reduce_ms) / denom;
        eprintln!(
            "[grouped-swiglu-down-{label}] chunk_p={chunk_p} active_experts={} p50_count={} p90_count={} max_count={} experts_ge16={} experts_ge32={} experts_ge48={} scan_back_edges={}/{} atomic_back_edges={}/{} packed_tail={:.2} ms live_grouped_tail={:.2} ms atomic_grouped_tail={:.2} ms fused_grouped_tail={:.2} ms map={:.2} ms split_gate_up={:.2} ms split_down={:.2} ms split_reduce={:.2} ms split_down_reduce={:.2} ms split_wall={:.2} vs_live={:.3} vs_atomic_live={:.3} vs_fused_live={:.3} vs_packed={:.3} cos_min={:.6} max_abs={:.3e} atomic_cos_min={:.6} atomic_max_abs={:.3e} fused_cos_min={:.6} fused_max_abs={:.3e}",
            active_experts,
            p50_count,
            p90_count,
            max_count,
            experts_ge16,
            experts_ge32,
            experts_ge48,
            scan_back_edges,
            scan_edges,
            atomic_back_edges,
            atomic_edges,
            packed_tail_ms / denom,
            live_grouped_tail_ms / denom,
            atomic_grouped_tail_ms / denom,
            fused_grouped_tail_ms / denom,
            map_ms / denom,
            grouped_swiglu_gpu_ms / denom,
            grouped_down_ms / denom,
            grouped_reduce_ms / denom,
            split_down_reduce_ms,
            grouped_total_wall_ms / denom,
            (live_grouped_tail_ms / denom) / (grouped_total_wall_ms / denom),
            (live_grouped_tail_ms / denom) / (atomic_grouped_tail_ms / denom),
            (live_grouped_tail_ms / denom) / (fused_grouped_tail_ms / denom),
            (packed_tail_ms / denom) / (grouped_total_wall_ms / denom),
            cos_min,
            max_abs,
            atomic_cos_min,
            atomic_max_abs,
            fused_cos_min,
            fused_max_abs,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_swiglu_down_backend_profile() {
        run_grouped_swiglu_down_backend_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            320,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_swiglu_down_backend_profile() {
        run_grouped_swiglu_down_backend_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            320,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_swiglu_down_backend_profile_512() {
        run_grouped_swiglu_down_backend_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-512",
            512,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_swiglu_down_backend_profile_512() {
        run_grouped_swiglu_down_backend_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-512",
            512,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_swiglu_down_backend_profile_1024() {
        run_grouped_swiglu_down_backend_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-1024",
            1024,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_swiglu_down_backend_profile_1024() {
        run_grouped_swiglu_down_backend_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-1024",
            1024,
            2,
        );
    }

    fn run_grouped_swiglu_fused_bank_profile(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        n_runs: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[grouped-swiglu-fused-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };
        assert_eq!(moe.gate_exps.dtype, GgmlType::Q4_K);
        assert_eq!(moe.up_exps.dtype, GgmlType::Q4_K);

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let counts = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("counts");
        let ids = MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids");
        let current_inner = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk * f_exp) as u64])
            .expect("current_inner");
        let fused_inner = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk * f_exp) as u64])
            .expect("fused_inner");
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();
        let hot_threshold = prefill_moe_hot_expert_min_slots().unwrap_or(1) as u32;
        let fused_gate_up =
            fuse_q4k_gate_up_expert_banks(&ctx, &moe.gate_exps, &moe.up_exps, h, f_exp, n_expert);

        let mut current_ms = 0.0f64;
        let mut fused_ms = 0.0f64;
        let mut cos_min = f64::INFINITY;
        let mut max_abs = 0.0f32;
        let mut hot_experts = 0usize;
        let mut hot_slots = 0usize;
        let mut max_count = 0usize;

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("route");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared");
            });
            let _ = timed_gpu_cmd(&ctx, |enc| {
                crate::metal::encode_moe_route_bucket_slots_f32(
                    &ctx,
                    enc,
                    &topk_idx_pack,
                    &counts,
                    &ids,
                    n_expert,
                    chunk_p,
                    topk,
                )
                .expect("bucket slots");
            });

            let counts_cpu = read_tensor_i32_f32buf(&counts);
            hot_experts = counts_cpu
                .iter()
                .filter(|&&c| c >= hot_threshold as i32)
                .count();
            hot_slots = counts_cpu
                .iter()
                .filter(|&&c| c >= hot_threshold as i32)
                .map(|&c| c as usize)
                .sum();
            max_count = counts_cpu.iter().copied().max().unwrap_or(0).max(0) as usize;
            if hot_experts == 0 {
                eprintln!(
                    "[grouped-swiglu-fused-{label}] no experts above threshold={hot_threshold}"
                );
                return;
            }

            current_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &current_inner, 0.0).expect("zero current inner");
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &counts,
                    &ids,
                    &current_inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                    hot_threshold,
                    i32::MAX as u32,
                )
                .expect("current hot n32");
                if hot_threshold > 0 {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &current_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                        0,
                        hot_threshold.saturating_sub(1),
                    )
                    .expect("current cold n16");
                }
            });

            fused_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &fused_inner, 0.0).expect("zero fused inner");
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n32_range(
                    &ctx,
                    enc,
                    &fused_gate_up,
                    &h_pack,
                    &counts,
                    &ids,
                    &fused_inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                    hot_threshold,
                    i32::MAX as u32,
                )
                .expect("fused hot n32");
                if hot_threshold > 0 {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n16_range(
                        &ctx,
                        enc,
                        &fused_gate_up,
                        &h_pack,
                        &counts,
                        &ids,
                        &fused_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                        0,
                        hot_threshold.saturating_sub(1),
                    )
                    .expect("fused cold n16");
                }
            });

            let cur = read_tensor_f32(&current_inner);
            let fus = read_tensor_f32(&fused_inner);
            cos_min = cos_min.min(cosine_f32(&cur, &fus));
            for i in 0..cur.len() {
                max_abs = max_abs.max((cur[i] - fus[i]).abs());
            }
        }

        let denom = n_runs as f64;
        eprintln!(
            "[grouped-swiglu-fused-{label}] chunk_p={chunk_p} hot_threshold={} hot_experts={} hot_slots={} max_count={} current_total={:.2} ms fused_total={:.2} ms speedup={:.3} cos_min={:.6} max_abs={:.3e}",
            hot_threshold,
            hot_experts,
            hot_slots,
            max_count,
            current_ms / denom,
            fused_ms / denom,
            (current_ms / denom) / (fused_ms / denom),
            cos_min,
            max_abs,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_swiglu_fused_bank_profile_320() {
        run_grouped_swiglu_fused_bank_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-320",
            320,
            3,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_swiglu_fused_bank_profile_320() {
        run_grouped_swiglu_fused_bank_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-320",
            320,
            3,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_swiglu_fused_bank_profile_512() {
        run_grouped_swiglu_fused_bank_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-512",
            512,
            3,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_swiglu_fused_bank_profile_512() {
        run_grouped_swiglu_fused_bank_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-512",
            512,
            3,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_swiglu_fused_bank_profile_1024() {
        run_grouped_swiglu_fused_bank_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-1024",
            1024,
            3,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_swiglu_fused_bank_profile_1024() {
        run_grouped_swiglu_fused_bank_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-1024",
            1024,
            3,
        );
    }

    fn assert_moe_grouped_slot_coverage(
        label: &str,
        counts: &[i32],
        ids: &[i32],
        topk_idx: &[i32],
        n_expert: usize,
        topk: usize,
        chunk_p: usize,
    ) {
        assert_eq!(counts.len(), n_expert, "{label}: counts length");
        assert_eq!(ids.len(), n_expert * chunk_p, "{label}: ids length");
        assert_eq!(topk_idx.len(), chunk_p * topk, "{label}: topk length");

        let mut per_token_seen = vec![false; n_expert];
        for token in 0..chunk_p {
            per_token_seen.fill(false);
            for k in 0..topk {
                let slot = token * topk + k;
                let expert = topk_idx[slot];
                assert!(expert >= 0, "{label}: negative expert at slot {slot}");
                let expert = expert as usize;
                assert!(
                    expert < n_expert,
                    "{label}: expert {expert} out of range at slot {slot}"
                );
                assert!(
                    !per_token_seen[expert],
                    "{label}: duplicate expert {expert} for token {token}"
                );
                per_token_seen[expert] = true;
            }
        }

        let slot_count = chunk_p * topk;
        let mut seen = vec![0u8; slot_count];
        let mut total = 0usize;
        for expert in 0..n_expert {
            let count = counts[expert];
            assert!(count >= 0, "{label}: negative count for expert {expert}");
            let count = count as usize;
            assert!(
                count <= chunk_p,
                "{label}: count {count} exceeds chunk_p for expert {expert}"
            );
            total += count;
            let base = expert * chunk_p;
            for j in 0..count {
                let slot = ids[base + j];
                assert!(slot >= 0, "{label}: negative id for expert {expert} j={j}");
                let slot = slot as usize;
                assert!(
                    slot < slot_count,
                    "{label}: id {slot} out of slot range for expert {expert} j={j}"
                );
                seen[slot] = seen[slot].saturating_add(1);
            }
        }
        assert_eq!(total, slot_count, "{label}: total routed slot count");
        for (slot, &n) in seen.iter().enumerate() {
            assert_eq!(n, 1, "{label}: slot {slot} appears {n} times");
        }
    }

    fn run_grouped_zero_fill_coverage_oracle(model_path: &str, label: &str, chunk_p: usize) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[grouped-zero-fill-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let slot_count = chunk_p * topk;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };
        assert_eq!(moe.gate_exps.dtype, GgmlType::Q4_K, "gate dtype");
        assert_eq!(moe.up_exps.dtype, GgmlType::Q4_K, "up dtype");
        assert_eq!(moe.down_exps.dtype, GgmlType::Q5_K, "down dtype");

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let counts = scratch
            .moe_group_count_pack
            .view_subrange(0, vec![n_expert as u64]);
        let ids = scratch
            .moe_group_ids_pack
            .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
        let inner_zero =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("inner_zero");
        let out_zero =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("out_zero");
        let reduced_zero =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("reduced_zero");
        let inner_poison =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("inner_poison");
        let out_poison =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("out_poison");
        let reduced_poison =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("reduced_poison");

        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();
        write_tensor_f32(&x_pack, &x_init);

        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_moe_route_logits_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route logits");
            encode_fill_f32(&ctx, enc, &counts, 0.0).expect("zero route counts");
            crate::metal::encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                &counts,
                &ids,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk+bucket");
        });

        let counts_cpu = read_tensor_i32_f32buf(&counts);
        let ids_cpu = read_tensor_i32_f32buf(&ids);
        let topk_idx_cpu = read_tensor_i32_f32buf(&topk_idx_pack);
        assert_moe_grouped_slot_coverage(
            label,
            &counts_cpu,
            &ids_cpu,
            &topk_idx_cpu,
            n_expert,
            topk,
            chunk_p,
        );

        let run_grouped =
            |inner: &MetalTensor, out: &MetalTensor, reduced: &MetalTensor, fill: f32| {
                timed_gpu_cmd(&ctx, |enc| {
                    encode_fill_f32(&ctx, enc, inner, fill).expect("fill grouped inner");
                    if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
                        if let Some(hot_threshold) = prefill_moe_hot_expert_min_slots() {
                            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
                                &ctx,
                                enc,
                                &moe.gate_exps,
                                &moe.up_exps,
                                &h_pack,
                                &counts,
                                &ids,
                                inner,
                                h,
                                f_exp,
                                n_expert,
                                topk,
                                chunk_p,
                                hot_threshold as u32,
                                i32::MAX as u32,
                            )
                            .expect("hot grouped swiglu");
                            if hot_threshold > 0 {
                                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
                                    &ctx,
                                    enc,
                                    &moe.gate_exps,
                                    &moe.up_exps,
                                    &h_pack,
                                    &counts,
                                    &ids,
                                    inner,
                                    h,
                                    f_exp,
                                    n_expert,
                                    topk,
                                    chunk_p,
                                    0,
                                    hot_threshold.saturating_sub(1) as u32,
                                )
                                .expect("cold grouped swiglu");
                            }
                        } else {
                            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                                &ctx,
                                enc,
                                &moe.gate_exps,
                                &moe.up_exps,
                                &h_pack,
                                &counts,
                                &ids,
                                inner,
                                h,
                                f_exp,
                                n_expert,
                                topk,
                                chunk_p,
                            )
                            .expect("grouped swiglu");
                        }
                    } else {
                        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &h_pack,
                            &counts,
                            &ids,
                            inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                        )
                        .expect("grouped swiglu");
                    }
                    encode_fill_f32(&ctx, enc, out, fill).expect("fill grouped out");
                    crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                        &ctx,
                        enc,
                        &moe.down_exps,
                        inner,
                        &counts,
                        &ids,
                        out,
                        f_exp,
                        h,
                        n_expert,
                        chunk_p,
                    )
                    .expect("grouped down");
                    crate::metal::encode_moe_weighted_sum_packed_f32(
                        &ctx,
                        enc,
                        out,
                        &topk_weight_pack,
                        reduced,
                        h,
                        topk,
                        chunk_p,
                    )
                    .expect("weighted sum");
                })
            };

        let zero_ms = run_grouped(&inner_zero, &out_zero, &reduced_zero, 0.0);
        let poison_ms = run_grouped(&inner_poison, &out_poison, &reduced_poison, f32::NAN);
        let zero = read_tensor_f32(&reduced_zero);
        let poison = read_tensor_f32(&reduced_poison);
        assert!(
            zero.iter().all(|v| v.is_finite()),
            "{label}: zero output has non-finite"
        );
        assert!(
            poison.iter().all(|v| v.is_finite()),
            "{label}: poison output has non-finite"
        );
        let cos = cosine_f32(&zero, &poison);
        let mut max_abs = 0.0f32;
        for i in 0..zero.len() {
            max_abs = max_abs.max((zero[i] - poison[i]).abs());
        }
        eprintln!(
            "[grouped-zero-fill-{label}] chunk_p={chunk_p} zero={zero_ms:.2} ms poison={poison_ms:.2} ms cos={cos:.9} max_abs={max_abs:.3e}"
        );
        assert!(cos > 0.999999, "{label}: cosine {cos:.9} below gate");
        assert!(max_abs < 1e-5, "{label}: max_abs {max_abs:.3e} above gate");
    }

    fn run_grouped_moe_overlap_falsifier(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        n_runs: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[grouped-overlap-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let f_shared = arch.expert_shared_feed_forward_length as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, g_w, u_w, d_w, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => (
                &g.post_attn_norm,
                &g.ffn_gate,
                &g.ffn_up,
                &g.ffn_down,
                g.ffn_moe.as_ref().expect("moe block"),
            ),
            crate::metal_forward::MetalBlock::Attn(a) => (
                &a.post_attn_norm,
                &a.ffn_gate,
                &a.ffn_up,
                &a.ffn_down,
                a.ffn_moe.as_ref().expect("moe block"),
            ),
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let counts = scratch
            .moe_group_count_pack
            .view_subrange(0, vec![n_expert as u64]);
        let ids = scratch
            .moe_group_ids_pack
            .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
        let routed_inner = scratch
            .moe_group_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let routed_out = scratch
            .moe_group_out_pack
            .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
        let mixer_out =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("mixer_out");
        let shared_gate_ffn = scratch
            .moe_shared_ffn_gate_pack
            .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
        let shared_up_ffn = scratch
            .moe_shared_ffn_up_pack
            .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
        let shared_inner_ffn = scratch
            .moe_shared_ffn_inner_pack
            .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
        let shared_out_ffn = scratch
            .moe_shared_ffn_out_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);
        let serial_final =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("serial_final");
        let concurrent_final =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("concurrent_final");
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        let _ = timed_gpu_cmd(&ctx, |enc| {
            write_tensor_f32(&x_pack, &x_init);
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_moe_route_logits_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route logits");
            encode_fill_f32(&ctx, enc, &counts, 0.0).expect("zero counts");
            crate::metal::encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                &counts,
                &ids,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("route+bucket");
        });

        let encode_routed = |enc: &KernelEncoder| {
            if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
                if let Some(hot_threshold) = prefill_moe_hot_expert_min_slots() {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &routed_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                        hot_threshold as u32,
                        i32::MAX as u32,
                    )
                    .expect("routed hot n32");
                    if hot_threshold > 0 {
                        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &h_pack,
                            &counts,
                            &ids,
                            &routed_inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            0,
                            hot_threshold.saturating_sub(1) as u32,
                        )
                        .expect("routed cold n16");
                    }
                } else {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &routed_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("routed n16 fallback");
                }
            } else {
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &counts,
                    &ids,
                    &routed_inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("routed n16");
            }
            encode_fill_f32(&ctx, enc, &routed_out, 0.0).expect("zero routed out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &routed_inner,
                &counts,
                &ids,
                &routed_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("routed down");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &routed_out,
                &topk_weight_pack,
                &mixer_out,
                h,
                topk,
                chunk_p,
            )
            .expect("routed reduce");
        };

        let encode_shared = |enc: &KernelEncoder| {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                g_w,
                &h_pack,
                &shared_gate_ffn,
                h,
                f_shared,
                chunk_p,
            )
            .expect("shared gate");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                u_w,
                &h_pack,
                &shared_up_ffn,
                h,
                f_shared,
                chunk_p,
            )
            .expect("shared up");
            encode_silu_mul_f32(
                &ctx,
                enc,
                &shared_gate_ffn,
                &shared_up_ffn,
                &shared_inner_ffn,
            )
            .expect("shared silu");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                d_w,
                &shared_inner_ffn,
                &shared_out_ffn,
                f_shared,
                h,
                chunk_p,
            )
            .expect("shared down");
        };

        let mut serial_gpu = 0.0f64;
        let mut concurrent_gpu = 0.0f64;
        let mut serial_wall = 0.0f64;
        let mut concurrent_wall = 0.0f64;
        let mut cos_min = f64::INFINITY;
        let mut max_abs = 0.0f32;

        for _ in 0..n_runs {
            write_tensor_f32(&serial_final, &x_init);
            let wall = Instant::now();
            let cmd = ctx.queue.commandBuffer().expect("serial cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_fill_f32(&ctx, &enc, &mixer_out, 0.0).expect("zero mixer");
            encode_routed(&enc);
            encode_shared(&enc);
            encode_axpy_rowwise_f32(
                &ctx,
                &enc,
                &shared_out_ffn,
                &shared_gate_pack,
                &mixer_out,
                h,
                chunk_p,
            )
            .expect("shared axpy");
            encode_add_inplace_f32(&ctx, &enc, &serial_final, &mixer_out).expect("serial add");
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            serial_gpu += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            serial_wall += wall.elapsed().as_secs_f64() * 1e3;

            write_tensor_f32(&concurrent_final, &x_init);
            let wall = Instant::now();
            let cmd = ctx.queue.commandBuffer().expect("concurrent cmd");
            {
                let enc = KernelEncoder::begin(&cmd);
                encode_fill_f32(&ctx, &enc, &mixer_out, 0.0).expect("zero mixer concurrent");
                enc.end();
            }
            {
                let enc = KernelEncoder::begin_concurrent(&cmd);
                encode_routed(&enc);
                encode_shared(&enc);
                enc.end();
            }
            {
                let enc = KernelEncoder::begin(&cmd);
                encode_axpy_rowwise_f32(
                    &ctx,
                    &enc,
                    &shared_out_ffn,
                    &shared_gate_pack,
                    &mixer_out,
                    h,
                    chunk_p,
                )
                .expect("shared axpy concurrent");
                encode_add_inplace_f32(&ctx, &enc, &concurrent_final, &mixer_out)
                    .expect("concurrent add");
                enc.end();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            concurrent_gpu += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            concurrent_wall += wall.elapsed().as_secs_f64() * 1e3;

            let serial = read_tensor_f32(&serial_final);
            let concurrent = read_tensor_f32(&concurrent_final);
            cos_min = cos_min.min(cosine_f32(&serial, &concurrent));
            for i in 0..serial.len() {
                max_abs = max_abs.max((serial[i] - concurrent[i]).abs());
            }
        }

        let denom = n_runs as f64;
        eprintln!(
            "[grouped-overlap-{label}] chunk_p={chunk_p} serial_gpu={:.2} ms concurrent_gpu={:.2} ms serial_wall={:.2} ms concurrent_wall={:.2} ms speedup_gpu={:.3} speedup_wall={:.3} cos_min={:.6} max_abs={:.3e}",
            serial_gpu / denom,
            concurrent_gpu / denom,
            serial_wall / denom,
            concurrent_wall / denom,
            (serial_gpu / denom) / (concurrent_gpu / denom),
            (serial_wall / denom) / (concurrent_wall / denom),
            cos_min,
            max_abs,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_overlap_falsifier_512() {
        run_grouped_moe_overlap_falsifier(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-512",
            512,
            3,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_overlap_falsifier_512() {
        run_grouped_moe_overlap_falsifier(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "a10b-512",
            512,
            3,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_zero_fill_coverage_oracle_512() {
        run_grouped_zero_fill_coverage_oracle(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-512",
            512,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_zero_fill_coverage_oracle_512() {
        run_grouped_zero_fill_coverage_oracle(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "a10b-512",
            512,
        );
    }

    fn run_grouped_q4_n32_proof(model_path: &str, label: &str, chunk_p: usize, n_runs: usize) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[grouped-q4-n32-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let group_count_pack = scratch
            .moe_group_count_pack
            .view_subrange(0, vec![n_expert as u64]);
        let group_ids_pack = scratch
            .moe_group_ids_pack
            .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
        let inner_n16 = scratch
            .moe_group_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let inner_n32 = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let slot_out = scratch
            .moe_group_out_pack
            .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
        let reduced_n16 =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("reduced_n16");
        let reduced_n32 =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("reduced_n32");
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        let mut n16_gpu = 0.0f64;
        let mut n32_gpu = 0.0f64;
        let mut cos_min = f64::INFINITY;
        let mut max_abs = 0.0f32;
        let mut active_experts = 0usize;
        let mut max_count = 0usize;

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("route");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared");
                crate::metal::encode_moe_route_bucket_slots_f32(
                    &ctx,
                    enc,
                    &topk_idx_pack,
                    &group_count_pack,
                    &group_ids_pack,
                    n_expert,
                    chunk_p,
                    topk,
                )
                .expect("bucket slots");
            });
            let count_cpu = cpu_read_i32_f32buf(&group_count_pack);
            active_experts = count_cpu.iter().filter(|&&c| c > 0).count();
            max_count = count_cpu.iter().copied().max().unwrap_or(0).max(0) as usize;

            n16_gpu += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &inner_n16, 0.0).expect("zero inner n16");
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &group_count_pack,
                    &group_ids_pack,
                    &inner_n16,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("grouped swiglu n16");
                encode_fill_f32(&ctx, enc, &slot_out, 0.0).expect("zero slot out");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &inner_n16,
                    &group_count_pack,
                    &group_ids_pack,
                    &slot_out,
                    f_exp,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("grouped down n16");
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &slot_out,
                    &topk_weight_pack,
                    &reduced_n16,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("grouped reduce n16");
            });

            n32_gpu += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &inner_n32, 0.0).expect("zero inner n32");
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &group_count_pack,
                    &group_ids_pack,
                    &inner_n32,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("grouped swiglu n32");
                encode_fill_f32(&ctx, enc, &slot_out, 0.0).expect("zero slot out 2");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &inner_n32,
                    &group_count_pack,
                    &group_ids_pack,
                    &slot_out,
                    f_exp,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("grouped down n32");
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &slot_out,
                    &topk_weight_pack,
                    &reduced_n32,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("grouped reduce n32");
            });

            let cur = read_tensor_f32(&reduced_n16);
            let alt = read_tensor_f32(&reduced_n32);
            cos_min = cos_min.min(cosine_f32(&cur, &alt));
            for i in 0..cur.len() {
                max_abs = max_abs.max((cur[i] - alt[i]).abs());
            }
        }

        let denom = n_runs as f64;
        eprintln!(
            "[grouped-q4-n32-{label}] chunk_p={chunk_p} active_experts={} max_count={} n16_gpu={:.2} ms n32_gpu={:.2} ms speedup={:.3} cos_min={:.6} max_abs={:.3e}",
            active_experts,
            max_count,
            n16_gpu / denom,
            n32_gpu / denom,
            (n16_gpu / denom) / (n32_gpu / denom),
            cos_min,
            max_abs,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_q4_n32_proof_512() {
        run_grouped_q4_n32_proof(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            512,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_q4_n32_proof_512() {
        run_grouped_q4_n32_proof(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            512,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_q4_n32_proof_1024() {
        run_grouped_q4_n32_proof(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b-1024",
            1024,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_q4_n32_proof_1024() {
        run_grouped_q4_n32_proof(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b-1024",
            1024,
            2,
        );
    }

    fn run_grouped_q4_hot_n32_proof(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        threshold: usize,
        n_runs: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[grouped-q4-hot-n32-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let count_pack = scratch
            .moe_group_count_pack
            .view_subrange(0, vec![n_expert as u64]);
        let ids_pack = scratch
            .moe_group_ids_pack
            .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
        let hot_count_pack =
            MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("hot_count");
        let cold_count_pack =
            MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("cold_count");
        let hot_ids_pack =
            MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("hot_ids");
        let cold_ids_pack =
            MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("cold_ids");
        let inner_n16 = scratch
            .moe_group_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let inner_mix = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let slot_out = scratch
            .moe_group_out_pack
            .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
        let reduced_n16 =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("reduced_n16");
        let reduced_mix =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("reduced_mix");
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        let mut n16_gpu = 0.0f64;
        let mut mix_gpu = 0.0f64;
        let mut cpu_split_ms = 0.0f64;
        let mut cos_min = f64::INFINITY;
        let mut max_abs = 0.0f32;
        let mut hot_experts = 0usize;

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("route");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared");
                crate::metal::encode_moe_route_bucket_slots_f32(
                    &ctx,
                    enc,
                    &topk_idx_pack,
                    &count_pack,
                    &ids_pack,
                    n_expert,
                    chunk_p,
                    topk,
                )
                .expect("bucket slots");
            });

            let t_cpu = Instant::now();
            let counts_cpu = cpu_read_i32_f32buf(&count_pack);
            let ids_cpu = cpu_read_i32_f32buf(&ids_pack);
            let mut hot_counts = vec![0i32; n_expert];
            let mut cold_counts = vec![0i32; n_expert];
            let mut hot_ids = vec![-1i32; n_expert * chunk_p];
            let mut cold_ids = vec![-1i32; n_expert * chunk_p];
            hot_experts = 0;
            for expert in 0..n_expert {
                let len = counts_cpu[expert].max(0) as usize;
                if len >= threshold {
                    hot_experts += 1;
                    hot_counts[expert] = len as i32;
                    hot_ids[expert * chunk_p..expert * chunk_p + len]
                        .copy_from_slice(&ids_cpu[expert * chunk_p..expert * chunk_p + len]);
                } else if len > 0 {
                    cold_counts[expert] = len as i32;
                    cold_ids[expert * chunk_p..expert * chunk_p + len]
                        .copy_from_slice(&ids_cpu[expert * chunk_p..expert * chunk_p + len]);
                }
            }
            cpu_write_i32_f32buf(&hot_count_pack, &hot_counts);
            cpu_write_i32_f32buf(&cold_count_pack, &cold_counts);
            cpu_write_i32_f32buf(&hot_ids_pack, &hot_ids);
            cpu_write_i32_f32buf(&cold_ids_pack, &cold_ids);
            cpu_split_ms += t_cpu.elapsed().as_secs_f64() * 1e3;

            n16_gpu += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &inner_n16, 0.0).expect("zero inner n16");
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &count_pack,
                    &ids_pack,
                    &inner_n16,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("grouped swiglu n16");
                encode_fill_f32(&ctx, enc, &slot_out, 0.0).expect("zero slot out");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &inner_n16,
                    &count_pack,
                    &ids_pack,
                    &slot_out,
                    f_exp,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("grouped down n16");
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &slot_out,
                    &topk_weight_pack,
                    &reduced_n16,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("grouped reduce n16");
            });

            mix_gpu += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &inner_mix, 0.0).expect("zero inner mix");
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &hot_count_pack,
                    &hot_ids_pack,
                    &inner_mix,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("grouped swiglu n32 hot");
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &cold_count_pack,
                    &cold_ids_pack,
                    &inner_mix,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("grouped swiglu n16 cold");
                encode_fill_f32(&ctx, enc, &slot_out, 0.0).expect("zero slot out 2");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &inner_mix,
                    &count_pack,
                    &ids_pack,
                    &slot_out,
                    f_exp,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("grouped down mix");
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &slot_out,
                    &topk_weight_pack,
                    &reduced_mix,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("grouped reduce mix");
            });

            let cur = read_tensor_f32(&reduced_n16);
            let alt = read_tensor_f32(&reduced_mix);
            cos_min = cos_min.min(cosine_f32(&cur, &alt));
            for i in 0..cur.len() {
                max_abs = max_abs.max((cur[i] - alt[i]).abs());
            }
        }

        let denom = n_runs as f64;
        eprintln!(
            "[grouped-q4-hot-n32-{label}] chunk_p={chunk_p} threshold={threshold} hot_experts={} n16_gpu={:.2} ms mix_gpu={:.2} ms cpu_split={:.2} ms speedup={:.3} cos_min={:.6} max_abs={:.3e}",
            hot_experts,
            n16_gpu / denom,
            mix_gpu / denom,
            cpu_split_ms / denom,
            (n16_gpu / denom) / (mix_gpu / denom),
            cos_min,
            max_abs,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_q4_hot_n32_proof_512() {
        run_grouped_q4_hot_n32_proof(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            512,
            32,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_q4_hot_n32_proof_512() {
        run_grouped_q4_hot_n32_proof(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            512,
            32,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_q4_hot_n32_threshold_scan_512() {
        for &threshold in &[48usize, 64, 96, 128] {
            run_grouped_q4_hot_n32_proof(
                "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                &format!("a3b-th{threshold}"),
                512,
                threshold,
                2,
            );
        }
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_q4_hot_n32_threshold_scan_512() {
        for &threshold in &[48usize, 64, 96, 128] {
            run_grouped_q4_hot_n32_proof(
                "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
                &format!("122b-th{threshold}"),
                512,
                threshold,
                2,
            );
        }
    }

    fn run_gpu_owned_grouped_routed_backend_profile(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        n_runs: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[gpu-owned-grouped-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let moe_inner_pack = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let grouped_reduced =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("grouped_reduced");
        let current_reduced =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("current_reduced");
        let counts = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("counts");
        let ids = MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids");
        let grouped_inner_slot_major =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk * f_exp) as u64])
                .expect("grouped_inner_slot_major");
        let grouped_slot_out = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk * h) as u64])
            .expect("grouped_slot_out");
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        let mut current_tail_ms = 0.0f64;
        let mut map_ms = 0.0f64;
        let mut grouped_tail_ms = 0.0f64;
        let mut grouped_wall_ms = 0.0f64;
        let mut cos_min = f64::INFINITY;
        let mut max_abs = 0.0f32;

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("route");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared");
            });

            current_tail_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_moe_swiglu_q4_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &topk_idx_pack,
                    &moe_inner_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("current swiglu");
                crate::metal::encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &current_reduced,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("current down");
            });

            let wall = Instant::now();
            map_ms += timed_gpu_cmd(&ctx, |enc| {
                crate::metal::encode_moe_route_bucket_slots_f32(
                    &ctx,
                    enc,
                    &topk_idx_pack,
                    &counts,
                    &ids,
                    n_expert,
                    chunk_p,
                    topk,
                )
                .expect("bucket slots");
            });

            grouped_tail_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &grouped_inner_slot_major, 0.0)
                    .expect("zero grouped inner slot-major");
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &counts,
                    &ids,
                    &grouped_inner_slot_major,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("grouped swiglu n16");
                encode_fill_f32(&ctx, enc, &grouped_slot_out, 0.0).expect("zero grouped slot out");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &grouped_inner_slot_major,
                    &counts,
                    &ids,
                    &grouped_slot_out,
                    f_exp,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("grouped down slots");
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &grouped_slot_out,
                    &topk_weight_pack,
                    &grouped_reduced,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("grouped reduce");
            });
            grouped_wall_ms += wall.elapsed().as_secs_f64() * 1e3;

            let cur = read_tensor_f32(&current_reduced);
            let grp = read_tensor_f32(&grouped_reduced);
            cos_min = cos_min.min(cosine_f32(&cur, &grp));
            for i in 0..cur.len() {
                max_abs = max_abs.max((cur[i] - grp[i]).abs());
            }
        }

        let denom = n_runs as f64;
        eprintln!(
            "[gpu-owned-grouped-{label}] chunk_p={chunk_p} current_tail={:.2} ms map={:.2} ms grouped_tail={:.2} ms grouped_wall={:.2} ms speedup={:.3} cos_min={:.6} max_abs={:.3e}",
            current_tail_ms / denom,
            map_ms / denom,
            grouped_tail_ms / denom,
            grouped_wall_ms / denom,
            (current_tail_ms / denom) / (grouped_wall_ms / denom),
            cos_min,
            max_abs,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_gpu_owned_grouped_routed_backend_profile() {
        run_gpu_owned_grouped_routed_backend_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            320,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_gpu_owned_grouped_routed_backend_profile() {
        run_gpu_owned_grouped_routed_backend_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            320,
            2,
        );
    }

    fn run_grouped_routed_down_accum_proof(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        n_runs: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[grouped-down-accum-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let moe_group_count_pack = scratch
            .moe_group_count_pack
            .view_subrange(0, vec![n_expert as u64]);
        let moe_group_ids_pack = scratch
            .moe_group_ids_pack
            .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
        let moe_group_slot_idx_pack = scratch
            .moe_group_slot_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let moe_group_token_idx_pack = scratch
            .moe_group_token_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let moe_group_weight_pack = scratch
            .moe_group_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let moe_group_inner_pack = scratch
            .moe_group_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let moe_group_out_pack = scratch
            .moe_group_out_pack
            .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
        let current_reduced =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("current_reduced");
        let accum_reduced =
            MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("accum_reduced");
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        let mut current_gpu_ms = 0.0f64;
        let mut accum_gpu_ms = 0.0f64;
        let mut cpu_group_ms = 0.0f64;
        let mut accum_wall_ms = 0.0f64;
        let mut cos_min = f64::INFINITY;
        let mut max_abs = 0.0f32;

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("route");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared");
                crate::metal::encode_moe_route_bucket_slots_f32(
                    &ctx,
                    enc,
                    &topk_idx_pack,
                    &moe_group_count_pack,
                    &moe_group_ids_pack,
                    n_expert,
                    chunk_p,
                    topk,
                )
                .expect("bucket slots");
                encode_fill_f32(&ctx, enc, &moe_group_inner_pack, 0.0).expect("zero grouped inner");
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &moe_group_count_pack,
                    &moe_group_ids_pack,
                    &moe_group_inner_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("grouped swiglu");
            });

            current_gpu_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &moe_group_out_pack, 0.0).expect("zero grouped out");
                crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_group_inner_pack,
                    &moe_group_count_pack,
                    &moe_group_ids_pack,
                    &moe_group_out_pack,
                    f_exp,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("grouped down");
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &moe_group_out_pack,
                    &topk_weight_pack,
                    &current_reduced,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("grouped reduce");
            });

            let t_cpu = Instant::now();
            let topk_idx_cpu = cpu_read_i32_f32buf(&topk_idx_pack);
            let topk_weight_cpu = cpu_read_f32buf(&topk_weight_pack);
            let count_cpu = cpu_read_i32_f32buf(&moe_group_count_pack);
            let ids_cpu = cpu_read_i32_f32buf(&moe_group_ids_pack);
            let (groups, slot_ids, token_ids, weights) =
                build_expert_slot_groups_cpu(&topk_idx_cpu, &topk_weight_cpu, topk, n_expert);
            let mut gpu_order = Vec::new();
            for expert in 0..n_expert {
                let len = count_cpu[expert].max(0) as usize;
                for j in 0..len.min(chunk_p) {
                    gpu_order.push(ids_cpu[expert * chunk_p + j]);
                }
            }
            assert_eq!(slot_ids, gpu_order, "grouped slot ordering diverged");
            cpu_write_i32_f32buf(&moe_group_slot_idx_pack, &slot_ids);
            cpu_write_i32_f32buf(&moe_group_token_idx_pack, &token_ids);
            cpu_write_f32buf(&moe_group_weight_pack, &weights);
            cpu_group_ms += t_cpu.elapsed().as_secs_f64() * 1e3;

            let wall = Instant::now();
            accum_gpu_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &accum_reduced, 0.0).expect("zero accum reduced");
                for group in &groups {
                    let token_ids_n = moe_group_token_idx_pack
                        .view_subrange(group.start as u64, vec![group.len as u64]);
                    let weights_n = moe_group_weight_pack
                        .view_subrange(group.start as u64, vec![group.len as u64]);
                    let inner_n = moe_group_inner_pack.view_subrange(
                        (group.start * f_exp) as u64,
                        vec![(group.len * f_exp) as u64],
                    );
                    let out_n = moe_group_out_pack
                        .view_subrange((group.start * h) as u64, vec![(group.len * h) as u64]);
                    let expert_bytes = moe.down_exps.n_bytes() / n_expert as u64;
                    let expert_w = moe
                        .down_exps
                        .view_bytes(group.expert as u64 * expert_bytes, vec![(h * f_exp) as u64]);
                    encode_mat_mat_dispatch(
                        &ctx, enc, &expert_w, &inner_n, &out_n, f_exp, h, group.len,
                    )
                    .expect("expert grouped down");
                    crate::metal::encode_scatter_axpy_rows_unique_f32(
                        &ctx,
                        enc,
                        &out_n,
                        &token_ids_n,
                        &weights_n,
                        &accum_reduced,
                        h,
                        group.len,
                    )
                    .expect("scatter accum");
                }
            });
            accum_wall_ms += wall.elapsed().as_secs_f64() * 1e3;

            let cur = read_tensor_f32(&current_reduced);
            let acc = read_tensor_f32(&accum_reduced);
            cos_min = cos_min.min(cosine_f32(&cur, &acc));
            for i in 0..cur.len() {
                max_abs = max_abs.max((cur[i] - acc[i]).abs());
            }
        }

        let denom = n_runs as f64;
        eprintln!(
            "[grouped-down-accum-{label}] chunk_p={chunk_p} current_gpu={:.2} ms accum_gpu={:.2} ms cpu_group={:.2} ms accum_wall={:.2} ms gpu_speedup={:.3} cos_min={:.6} max_abs={:.3e}",
            current_gpu_ms / denom,
            accum_gpu_ms / denom,
            cpu_group_ms / denom,
            accum_wall_ms / denom,
            (current_gpu_ms / denom) / (accum_gpu_ms / denom),
            cos_min,
            max_abs,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_routed_down_accum_proof_512() {
        run_grouped_routed_down_accum_proof(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            512,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_routed_down_accum_proof_512() {
        run_grouped_routed_down_accum_proof(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            512,
            2,
        );
    }

    fn run_fused_routed_tail_profile(model_path: &str, label: &str, chunk_p: usize, n_runs: usize) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[fused-routed-tail-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let moe_inner_pack = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let current_out = scratch
            .mixer_out_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);
        let fused_out = scratch
            .moe_shared_ffn_out_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("warmup postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("warmup route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("warmup topk/shared");
            encode_moe_swiglu_q4_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &topk_idx_pack,
                &moe_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("warmup swiglu");
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_inner_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &current_out,
                f_exp,
                h,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("warmup current down");
            encode_fill_f32(&ctx, enc, &fused_out, 0.0).expect("warmup fused zero");
            crate::metal::encode_moe_fused_routed_q4q5_token_f32(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &moe.down_exps,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &fused_out,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("warmup fused routed");
        });

        let mut current_ms = 0.0f64;
        let mut fused_ms = 0.0f64;
        let mut cos_min = f64::INFINITY;
        let mut max_abs = 0.0f32;

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("route");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared");
            });

            current_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_moe_swiglu_q4_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &topk_idx_pack,
                    &moe_inner_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("current swiglu");
                encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &current_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("current down");
            });

            fused_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &fused_out, 0.0).expect("fused zero");
                crate::metal::encode_moe_fused_routed_q4q5_token_f32(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &moe.down_exps,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &fused_out,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("fused routed");
            });

            let cur = read_tensor_f32(&current_out);
            let fus = read_tensor_f32(&fused_out);
            let cos = cosine_f32(&cur, &fus);
            cos_min = cos_min.min(cos);
            for i in 0..cur.len() {
                max_abs = max_abs.max((cur[i] - fus[i]).abs());
            }
        }

        let denom = n_runs as f64;
        eprintln!(
            "[fused-routed-tail-{label}] chunk_p={chunk_p} current={:.2} ms fused={:.2} ms speedup={:.3} cos_min={:.6} max_abs={:.3e}",
            current_ms / denom,
            fused_ms / denom,
            (current_ms / denom) / (fused_ms / denom),
            cos_min,
            max_abs
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_fused_routed_tail_profile() {
        run_fused_routed_tail_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            320,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_fused_routed_tail_profile() {
        run_fused_routed_tail_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            320,
            2,
        );
    }

    fn run_grouped_routed_down_experiment_profile(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        n_runs: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[grouped-down-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let slot_count = chunk_p * topk;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(slot_count) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(slot_count) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let moe_inner_pack = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(slot_count * f_exp) as u64]);
        let mixer_out_pack = scratch
            .mixer_out_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);
        let token_major_out = scratch
            .moe_expert_out_pack
            .view_subrange(0, vec![(slot_count * h) as u64]);
        let group_slot_ids =
            MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("group ids");
        let group_token_ids =
            MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("group tokens");
        let group_expert_ids =
            MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("group experts");
        let group_weights =
            MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("group weights");
        let grouped_inner =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("grouped inner");
        let grouped_out =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("grouped out");
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        let mut current_down_gpu = 0.0f64;
        let mut grouped_cpu_ms = 0.0f64;
        let mut grouped_gpu_ms = 0.0f64;
        let mut grouped_wall_ms = 0.0f64;
        let mut group_count = 0usize;
        let mut max_group = 0usize;

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("route");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared");
                encode_moe_swiglu_q4_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &topk_idx_pack,
                    &moe_inner_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("swiglu");
            });

            current_down_gpu += timed_gpu_cmd(&ctx, |enc| {
                encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &mixer_out_pack,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("current down");
            });

            let prep_start = Instant::now();
            let topk_idx = read_tensor_i32_f32buf(&topk_idx_pack);
            let topk_weight = read_tensor_f32(&topk_weight_pack);
            let (groups, slot_ids, token_ids, weights) =
                build_expert_slot_groups(&topk_idx, &topk_weight, topk, n_expert);
            let mut expert_ids = Vec::with_capacity(slot_ids.len());
            for group in &groups {
                expert_ids.extend(std::iter::repeat_n(group.expert as i32, group.len));
            }
            group_count = groups.len();
            max_group = groups.iter().map(|g| g.len).max().unwrap_or(0);
            write_tensor_i32_f32buf(&group_slot_ids, &slot_ids);
            write_tensor_i32_f32buf(&group_token_ids, &token_ids);
            write_tensor_i32_f32buf(&group_expert_ids, &expert_ids);
            write_tensor_f32(&group_weights, &weights);
            grouped_cpu_ms += prep_start.elapsed().as_secs_f64() * 1e3;

            let wall = Instant::now();
            grouped_gpu_ms += timed_gpu_cmd(&ctx, |enc| {
                crate::metal::encode_get_rows_f32(
                    &ctx,
                    enc,
                    &moe_inner_pack,
                    &group_slot_ids,
                    &grouped_inner,
                    slot_count,
                    f_exp,
                )
                .expect("gather grouped inner");
                crate::metal::encode_moe_down_q5_K_f32_grouped_rows(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &grouped_inner,
                    &group_expert_ids,
                    &grouped_out,
                    f_exp,
                    h,
                    n_expert,
                    slot_count,
                )
                .expect("grouped down matmat");
                crate::metal::encode_scatter_rows_f32_unique(
                    &ctx,
                    enc,
                    &grouped_out,
                    &group_slot_ids,
                    &token_major_out,
                    h,
                    slot_count,
                )
                .expect("scatter grouped rows");
                crate::metal::encode_moe_weighted_sum_packed_f32(
                    &ctx,
                    enc,
                    &token_major_out,
                    &topk_weight_pack,
                    &mixer_out_pack,
                    h,
                    topk,
                    chunk_p,
                )
                .expect("packed weighted sum");
            });
            grouped_wall_ms += wall.elapsed().as_secs_f64() * 1e3;
        }

        let denom = n_runs as f64;
        eprintln!(
            "[grouped-down-{label}] chunk_p={chunk_p} groups={} max_group={} current_down_gpu={:.2} ms grouped_cpu={:.2} ms grouped_gpu={:.2} ms grouped_total={:.2} ms grouped_wall={:.2} ms speedup={:.3}",
            group_count,
            max_group,
            current_down_gpu / denom,
            grouped_cpu_ms / denom,
            grouped_gpu_ms / denom,
            (grouped_cpu_ms + grouped_gpu_ms) / denom,
            grouped_wall_ms / denom,
            (current_down_gpu / denom) / ((grouped_cpu_ms + grouped_gpu_ms) / denom)
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_grouped_routed_down_experiment_profile() {
        run_grouped_routed_down_experiment_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            320,
            2,
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_grouped_routed_down_experiment_profile() {
        run_grouped_routed_down_experiment_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            320,
            2,
        );
    }

    fn run_hot_expert_routed_down_hybrid_profile(
        model_path: &str,
        label: &str,
        chunk_p: usize,
        hot_threshold: usize,
        n_runs: usize,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[hot-hybrid-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        assert_eq!(arch.kind, crate::model::ArchKind::Moe);
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let n_expert = arch.expert_count as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let slot_count = chunk_p * topk;

        let block = &mf.model.blocks[0];
        let (post_norm, moe) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
            }
        };

        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![slot_count as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![slot_count as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let moe_inner_pack = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(slot_count * f_exp) as u64]);
        let hot_mixer_out = scratch
            .mixer_out_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);
        let group_slot_ids =
            MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("group ids");
        let group_token_ids =
            MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("group tokens");
        let group_weights =
            MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("group weights");
        let grouped_inner =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("grouped inner");
        let grouped_out =
            MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("grouped out");
        let cold_idx_pack =
            MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("cold idx");
        let cold_out = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("cold out");
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        let mut current_down_gpu = 0.0f64;
        let mut hybrid_cpu_ms = 0.0f64;
        let mut hybrid_gpu_ms = 0.0f64;
        let mut hybrid_wall_ms = 0.0f64;
        let mut hot_groups_n = 0usize;
        let mut hot_slots_n = 0usize;

        for _ in 0..n_runs {
            write_tensor_f32(&x_pack, &x_init);
            let _ = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("postnorm");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("route");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared");
                encode_moe_swiglu_q4_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &topk_idx_pack,
                    &moe_inner_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("swiglu");
            });

            current_down_gpu += timed_gpu_cmd(&ctx, |enc| {
                encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &hot_mixer_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("current down");
            });

            let prep_start = Instant::now();
            let topk_idx = read_tensor_i32_f32buf(&topk_idx_pack);
            let topk_weight = read_tensor_f32(&topk_weight_pack);
            let (groups, slot_ids, token_ids, weights) =
                build_expert_slot_groups(&topk_idx, &topk_weight, topk, n_expert);
            let hot_groups: Vec<_> = groups
                .iter()
                .copied()
                .filter(|g| g.len >= hot_threshold)
                .collect();
            hot_groups_n = hot_groups.len();
            hot_slots_n = hot_groups.iter().map(|g| g.len).sum();
            let mut cold_idx = topk_idx.clone();
            for group in &hot_groups {
                for &slot_id in &slot_ids[group.start..group.start + group.len] {
                    cold_idx[slot_id as usize] = -1;
                }
            }
            write_tensor_i32_f32buf(&group_slot_ids, &slot_ids);
            write_tensor_i32_f32buf(&group_token_ids, &token_ids);
            write_tensor_f32(&group_weights, &weights);
            write_tensor_i32_f32buf(&cold_idx_pack, &cold_idx);
            hybrid_cpu_ms += prep_start.elapsed().as_secs_f64() * 1e3;

            let wall = Instant::now();
            hybrid_gpu_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &hot_mixer_out, 0.0).expect("zero hot");
                for group in &hot_groups {
                    let slot_ids_n =
                        group_slot_ids.view_subrange(group.start as u64, vec![group.len as u64]);
                    let token_ids_n =
                        group_token_ids.view_subrange(group.start as u64, vec![group.len as u64]);
                    let weights_n =
                        group_weights.view_subrange(group.start as u64, vec![group.len as u64]);
                    let inner_n = grouped_inner.view_subrange(
                        (group.start * f_exp) as u64,
                        vec![(group.len * f_exp) as u64],
                    );
                    let out_n = grouped_out
                        .view_subrange((group.start * h) as u64, vec![(group.len * h) as u64]);
                    crate::metal::encode_get_rows_f32(
                        &ctx,
                        enc,
                        &moe_inner_pack,
                        &slot_ids_n,
                        &inner_n,
                        group.len,
                        f_exp,
                    )
                    .expect("gather hot inner");
                    let expert_bytes = moe.down_exps.n_bytes() / n_expert as u64;
                    let expert_w = moe
                        .down_exps
                        .view_bytes(group.expert as u64 * expert_bytes, vec![(h * f_exp) as u64]);
                    encode_mat_mat_dispatch(
                        &ctx, enc, &expert_w, &inner_n, &out_n, f_exp, h, group.len,
                    )
                    .expect("hot grouped matmat");
                    crate::metal::encode_scatter_axpy_rows_unique_f32(
                        &ctx,
                        enc,
                        &out_n,
                        &token_ids_n,
                        &weights_n,
                        &hot_mixer_out,
                        h,
                        group.len,
                    )
                    .expect("scatter hot out");
                }
                encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner_pack,
                    &cold_idx_pack,
                    &topk_weight_pack,
                    &cold_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("cold down");
                encode_add_inplace_f32(&ctx, enc, &hot_mixer_out, &cold_out)
                    .expect("merge hot/cold");
            });
            hybrid_wall_ms += wall.elapsed().as_secs_f64() * 1e3;
        }

        let denom = n_runs as f64;
        eprintln!(
            "[hot-hybrid-{label}] chunk_p={chunk_p} threshold={hot_threshold} hot_groups={} hot_slots={} current_down_gpu={:.2} ms hybrid_cpu={:.2} ms hybrid_gpu={:.2} ms hybrid_total={:.2} ms hybrid_wall={:.2} ms speedup={:.3}",
            hot_groups_n,
            hot_slots_n,
            current_down_gpu / denom,
            hybrid_cpu_ms / denom,
            hybrid_gpu_ms / denom,
            (hybrid_cpu_ms + hybrid_gpu_ms) / denom,
            hybrid_wall_ms / denom,
            (current_down_gpu / denom) / ((hybrid_cpu_ms + hybrid_gpu_ms) / denom)
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_hot_expert_routed_down_hybrid_profile() {
        for threshold in [64usize, 96, 128] {
            run_hot_expert_routed_down_hybrid_profile(
                "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
                "122b",
                320,
                threshold,
                2,
            );
        }
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_hot_expert_routed_down_hybrid_profile() {
        run_hot_expert_routed_down_hybrid_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            320,
            64,
            2,
        );
    }

    fn run_packed_dense_prefill_phase_profile(model_path: &str, prompt: &str, chunk_p: usize) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[packed-prefill-phase] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        assert_eq!(m.arch.kind, crate::model::ArchKind::Dense);
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok.encode(prompt, false).expect("tokenize");
        let total_n = ids.len();
        let arch = &mm.arch;
        let h = arch.hidden_size as usize;
        let f = arch.intermediate_size as usize;
        let cap = total_n + 16;
        let mut sess = MetalSession::fresh(&ctx, &mm, cap).expect("session");
        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");

        let ids_buf = MetalTensor::zeros_f32(&ctx, vec![chunk_p as u64]).expect("ids buf");
        unsafe {
            let p = ids_buf.buffer.contents().as_ptr() as *mut i32;
            for (i, &t) in ids.iter().enumerate() {
                *p.add(i) = t;
            }
        }

        let x_pack_p = scratch.x_pack.view_subrange(0, vec![(total_n * h) as u64]);
        let h_pack_p = scratch.h_pack.view_subrange(0, vec![(total_n * h) as u64]);
        let mixer_out_pack_p = scratch
            .mixer_out_pack
            .view_subrange(0, vec![(total_n * h) as u64]);

        let timed =
            |label: &str,
             cb: &mut dyn FnMut(&KernelEncoder) -> Result<(), crate::metal_forward::MfError>|
             -> f64 {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                cb(&enc).expect(label);
                enc.end();
                cmd.commit();
                cmd.waitUntilCompleted();
                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3
            };

        let mut embed_ms = 0.0f64;
        let mut gdn_front_ms = 0.0f64;
        let mut gdn_alpha_beta_ms = 0.0f64;
        let mut gdn_tail_ms = 0.0f64;
        let mut gdn_back_ms = 0.0f64;
        let mut attn_front_ms = 0.0f64;
        let mut attn_decode_ms = 0.0f64;
        let mut attn_back_ms = 0.0f64;
        let mut ffn_ms = 0.0f64;
        let mut tail_ms = 0.0f64;

        embed_ms += timed("embed", &mut |enc| {
            let ids_view = ids_buf.view_subrange(0, vec![total_n as u64]);
            crate::metal::encode_get_rows_f32(
                &ctx,
                enc,
                &mf.model.token_embd,
                &ids_view,
                &x_pack_p,
                total_n,
                h,
            )
            .map_err(crate::metal_forward::MfError::from)
        });

        let gdn_mat_mat_eligible = prefill_mat_mat_dispatch_eligible;
        let attn_mat_mat_eligible = prefill_mat_mat_dispatch_eligible;
        let ffn_mat_mat_eligible = prefill_mat_mat_dispatch_eligible;

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &mf.model.blocks {
            let attn_norm = match block {
                crate::metal_forward::MetalBlock::Gdn(g) => &g.attn_norm,
                crate::metal_forward::MetalBlock::Attn(a) => &a.attn_norm,
            };
            let post_norm = match block {
                crate::metal_forward::MetalBlock::Gdn(g) => &g.post_attn_norm,
                crate::metal_forward::MetalBlock::Attn(a) => &a.post_attn_norm,
            };
            let (g_w, u_w, d_w) = match block {
                crate::metal_forward::MetalBlock::Gdn(g) => (&g.ffn_gate, &g.ffn_up, &g.ffn_down),
                crate::metal_forward::MetalBlock::Attn(a) => (&a.ffn_gate, &a.ffn_up, &a.ffn_down),
            };

            timed("pre_mixer_norm", &mut |enc| {
                crate::metal::encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack_p,
                    attn_norm,
                    &h_pack_p,
                    total_n,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .map_err(crate::metal_forward::MfError::from)
            });

            match block {
                crate::metal_forward::MetalBlock::Gdn(g) => {
                    let gi = gdn_idx;
                    gdn_idx += 1;
                    let n_k_u = arch.gdn_n_k_heads as usize;
                    let n_v_u = arch.gdn_n_v_heads as usize;
                    let head_dim_u = arch.gdn_head_dim as usize;
                    let conv_dim = (2 * n_k_u + n_v_u) * head_dim_u;
                    let v_dim = n_v_u * head_dim_u;
                    let gdn_qkv_pack_p = scratch
                        .gdn_qkv_pack
                        .view_subrange(0, vec![(total_n * conv_dim) as u64]);
                    let gdn_z_pack_p = scratch
                        .gdn_z_pack
                        .view_subrange(0, vec![(total_n * v_dim) as u64]);
                    let gdn_beta_pack_p = scratch
                        .gdn_beta_pack
                        .view_subrange(0, vec![(total_n * n_v_u) as u64]);
                    let gdn_alpha_pack_p = scratch
                        .gdn_alpha_pack
                        .view_subrange(0, vec![(total_n * n_v_u) as u64]);
                    let gdn_q_norm_pack_p = scratch
                        .gdn_q_norm_pack
                        .view_subrange(0, vec![(total_n * n_k_u * head_dim_u) as u64]);
                    let gdn_k_norm_pack_p = scratch
                        .gdn_k_norm_pack
                        .view_subrange(0, vec![(total_n * n_k_u * head_dim_u) as u64]);
                    let gdn_v_pack_p = scratch
                        .gdn_v_pack
                        .view_subrange(0, vec![(total_n * v_dim) as u64]);
                    let gdn_out_pack_p = scratch
                        .gdn_out_pack
                        .view_subrange(0, vec![(total_n * v_dim) as u64]);
                    let gdn_normed_pack_p = scratch
                        .gdn_normed_pack
                        .view_subrange(0, vec![(total_n * v_dim) as u64]);
                    let gdn_batched = gdn_mat_mat_eligible(g.in_proj_qkv.dtype)
                        && gdn_mat_mat_eligible(g.in_proj_z.dtype)
                        && gdn_mat_mat_eligible(g.out_proj.dtype);
                    if gdn_batched {
                        gdn_front_ms += timed("gdn_front", &mut |enc| {
                            encode_mat_mat_dispatch(
                                &ctx,
                                enc,
                                &g.in_proj_qkv,
                                &h_pack_p,
                                &gdn_qkv_pack_p,
                                h,
                                conv_dim,
                                total_n,
                            )?;
                            encode_mat_mat_dispatch(
                                &ctx,
                                enc,
                                &g.in_proj_z,
                                &h_pack_p,
                                &gdn_z_pack_p,
                                h,
                                v_dim,
                                total_n,
                            )?;
                            encode_mat_mat_dispatch(
                                &ctx,
                                enc,
                                &g.beta_proj,
                                &h_pack_p,
                                &gdn_beta_pack_p,
                                h,
                                n_v_u,
                                total_n,
                            )?;
                            encode_mat_mat_dispatch(
                                &ctx,
                                enc,
                                &g.alpha_proj,
                                &h_pack_p,
                                &gdn_alpha_pack_p,
                                h,
                                n_v_u,
                                total_n,
                            )
                        });
                        gdn_alpha_beta_ms += timed("gdn_alpha_beta", &mut |enc| {
                            crate::metal::encode_sigmoid_f32(
                                &ctx,
                                enc,
                                &gdn_beta_pack_p,
                                &gdn_beta_pack_p,
                            )?;
                            encode_gdn_decay_chain_batched_f32(
                                &ctx,
                                enc,
                                &gdn_alpha_pack_p,
                                &g.dt_bias,
                                &g.a_log,
                                &gdn_alpha_pack_p,
                                total_n,
                                n_v_u,
                            )
                            .map_err(crate::metal_forward::MfError::from)
                        });
                        if dense_packed_gdn_step_enabled() {
                            let mut conv_stage_ms = 0.0f64;
                            let mut step_stage_ms = 0.0f64;
                            let mut rms_stage_ms = 0.0f64;
                            conv_stage_ms += timed("gdn_tail_pre_step", &mut |enc| {
                                for n_idx in 0..total_n {
                                    let qkv_n = gdn_qkv_pack_p.view_subrange(
                                        (n_idx * conv_dim) as u64,
                                        vec![conv_dim as u64],
                                    );
                                    encode_ssm_conv_silu_f32(
                                        &ctx,
                                        enc,
                                        &qkv_n,
                                        &sess.gdn_conv[gi],
                                        &g.conv1d,
                                        &sess.gdn_qkv_conv,
                                        conv_dim,
                                    )?;
                                    let q_view = sess
                                        .gdn_qkv_conv
                                        .view_subrange(0, vec![(n_k_u * head_dim_u) as u64]);
                                    let k_view = sess.gdn_qkv_conv.view_subrange(
                                        (n_k_u * head_dim_u) as u64,
                                        vec![(n_k_u * head_dim_u) as u64],
                                    );
                                    let v_view = sess.gdn_qkv_conv.view_subrange(
                                        (2 * n_k_u * head_dim_u) as u64,
                                        vec![v_dim as u64],
                                    );
                                    encode_l2_norm_batched_f32(
                                        &ctx,
                                        enc,
                                        &q_view,
                                        &sess.gdn_q_norm,
                                        n_k_u,
                                        head_dim_u,
                                        crate::metal_forward::RMS_EPS,
                                    )?;
                                    encode_l2_norm_batched_f32(
                                        &ctx,
                                        enc,
                                        &k_view,
                                        &sess.gdn_k_norm,
                                        n_k_u,
                                        head_dim_u,
                                        crate::metal_forward::RMS_EPS,
                                    )?;
                                    encode_scatter_offset_f32(
                                        &ctx,
                                        enc,
                                        &sess.gdn_q_norm,
                                        &gdn_q_norm_pack_p,
                                        n_idx * n_k_u * head_dim_u,
                                        n_k_u * head_dim_u,
                                    )?;
                                    encode_scatter_offset_f32(
                                        &ctx,
                                        enc,
                                        &sess.gdn_k_norm,
                                        &gdn_k_norm_pack_p,
                                        n_idx * n_k_u * head_dim_u,
                                        n_k_u * head_dim_u,
                                    )?;
                                    encode_scatter_offset_f32(
                                        &ctx,
                                        enc,
                                        &v_view,
                                        &gdn_v_pack_p,
                                        n_idx * v_dim,
                                        v_dim,
                                    )?;
                                }
                                Ok(())
                            });
                            step_stage_ms += timed("gdn_tail_step", &mut |enc| {
                                encode_gdn_step_decay_packed_f32(
                                    &ctx,
                                    enc,
                                    &gdn_q_norm_pack_p,
                                    &gdn_k_norm_pack_p,
                                    &gdn_v_pack_p,
                                    &gdn_alpha_pack_p,
                                    &gdn_beta_pack_p,
                                    &sess.gdn_state[gi],
                                    &gdn_out_pack_p,
                                    total_n,
                                    n_v_u,
                                    n_k_u,
                                    head_dim_u,
                                )
                                .map_err(crate::metal_forward::MfError::from)
                            });
                            rms_stage_ms += timed("gdn_tail_post_step", &mut |enc| {
                                for n_idx in 0..total_n {
                                    let out_n = gdn_out_pack_p
                                        .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                                    let z_n = gdn_z_pack_p
                                        .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                                    let normed_n = gdn_normed_pack_p
                                        .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                                    encode_rmsnorm_gated_f32(
                                        &ctx,
                                        enc,
                                        &out_n,
                                        &g.norm,
                                        &z_n,
                                        &normed_n,
                                        n_v_u,
                                        head_dim_u,
                                        crate::metal_forward::RMS_EPS,
                                    )?;
                                }
                                Ok(())
                            });
                            gdn_tail_ms += conv_stage_ms + step_stage_ms + rms_stage_ms;
                        } else {
                            gdn_tail_ms += timed("gdn_tail", &mut |enc| {
                                for n_idx in 0..total_n {
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
                                    mf.encode_gdn_tail(
                                        enc, g, gi, &mut sess, &qkv_n, &z_n, &alpha_n, &beta_n,
                                        &normed_n,
                                    )?;
                                }
                                Ok(())
                            });
                        }
                        gdn_back_ms += timed("gdn_back", &mut |enc| {
                            encode_mat_mat_dispatch(
                                &ctx,
                                enc,
                                &g.out_proj,
                                &gdn_normed_pack_p,
                                &mixer_out_pack_p,
                                v_dim,
                                h,
                                total_n,
                            )?;
                            crate::metal::encode_add_inplace_f32(
                                &ctx,
                                enc,
                                &x_pack_p,
                                &mixer_out_pack_p,
                            )
                            .map_err(crate::metal_forward::MfError::from)
                        });
                    } else {
                        gdn_tail_ms += timed("gdn_fallback", &mut |enc| {
                            for n_idx in 0..total_n {
                                encode_copy_offset_f32(
                                    &ctx,
                                    enc,
                                    &h_pack_p,
                                    n_idx * h,
                                    &sess.h,
                                    h,
                                )?;
                                mf.encode_gdn(enc, g, gi, &mut sess)?;
                                crate::metal_forward::encode_scatter_offset_f32(
                                    &ctx,
                                    enc,
                                    &sess.mixer_out,
                                    &mixer_out_pack_p,
                                    n_idx * h,
                                    h,
                                )?;
                            }
                            Ok(())
                        });
                        gdn_back_ms += timed("gdn_resid", &mut |enc| {
                            crate::metal::encode_add_inplace_f32(
                                &ctx,
                                enc,
                                &x_pack_p,
                                &mixer_out_pack_p,
                            )
                            .map_err(crate::metal_forward::MfError::from)
                        });
                    }
                }
                crate::metal_forward::MetalBlock::Attn(a) => {
                    let ai = attn_idx;
                    attn_idx += 1;
                    let head_dim = arch.attn_head_dim as usize;
                    let n_q = arch.n_q_heads as usize;
                    let n_kv = arch.n_kv_heads as usize;
                    let q_dim = n_q * head_dim;
                    let kv_dim = n_kv * head_dim;
                    let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
                    let q_full_pack_p = scratch
                        .attn_q_full_pack
                        .view_subrange(0, vec![(total_n * 2 * q_dim) as u64]);
                    let q_pack_p = scratch
                        .attn_q_pack
                        .view_subrange(0, vec![(total_n * q_dim) as u64]);
                    let gate_pack_p = scratch
                        .attn_gate_pack
                        .view_subrange(0, vec![(total_n * q_dim) as u64]);
                    let q_normed_pack_p = scratch
                        .attn_q_normed_pack
                        .view_subrange(0, vec![(total_n * q_dim) as u64]);
                    let k_now_pack_p = scratch
                        .attn_k_now_pack
                        .view_subrange(0, vec![(total_n * kv_dim) as u64]);
                    let v_now_pack_p = scratch
                        .attn_v_now_pack
                        .view_subrange(0, vec![(total_n * kv_dim) as u64]);
                    let k_normed_pack_p = scratch
                        .attn_k_normed_pack
                        .view_subrange(0, vec![(total_n * kv_dim) as u64]);
                    let attn_o_pack_p = scratch
                        .attn_o_pack
                        .view_subrange(0, vec![(total_n * q_dim) as u64]);
                    let attn_batched = attn_mat_mat_eligible(a.q.dtype)
                        && attn_mat_mat_eligible(a.k.dtype)
                        && attn_mat_mat_eligible(a.v.dtype)
                        && attn_mat_mat_eligible(a.o.dtype);
                    if attn_batched {
                        attn_front_ms += timed("attn_front", &mut |enc| {
                            encode_mat_mat_dispatch(
                                &ctx,
                                enc,
                                &a.q,
                                &h_pack_p,
                                &q_full_pack_p,
                                h,
                                2 * q_dim,
                                total_n,
                            )?;
                            crate::metal::encode_split_q_gate_f32(
                                &ctx,
                                enc,
                                &q_full_pack_p,
                                &q_pack_p,
                                &gate_pack_p,
                                total_n * n_q,
                                head_dim,
                            )?;
                            encode_mat_mat_dispatch(
                                &ctx,
                                enc,
                                &a.k,
                                &h_pack_p,
                                &k_now_pack_p,
                                h,
                                kv_dim,
                                total_n,
                            )?;
                            encode_mat_mat_dispatch(
                                &ctx,
                                enc,
                                &a.v,
                                &h_pack_p,
                                &v_now_pack_p,
                                h,
                                kv_dim,
                                total_n,
                            )?;
                            crate::metal::encode_rms_norm_batched_f32(
                                &ctx,
                                enc,
                                &q_pack_p,
                                &a.q_norm,
                                &q_normed_pack_p,
                                total_n * n_q,
                                head_dim,
                                crate::metal_forward::RMS_EPS,
                            )?;
                            crate::metal::encode_rms_norm_batched_f32(
                                &ctx,
                                enc,
                                &k_now_pack_p,
                                &a.k_norm,
                                &k_normed_pack_p,
                                total_n * n_kv,
                                head_dim,
                                crate::metal_forward::RMS_EPS,
                            )
                            .map_err(crate::metal_forward::MfError::from)
                        });
                        attn_decode_ms += timed("attn_decode", &mut |enc| {
                            crate::metal::encode_rope_neox_f32_packed_consecutive(
                                &ctx,
                                enc,
                                &q_normed_pack_p,
                                total_n,
                                n_q,
                                head_dim,
                                n_rot,
                                0,
                                arch.rope_theta,
                            )?;
                            crate::metal::encode_rope_neox_f32_packed_consecutive(
                                &ctx,
                                enc,
                                &k_normed_pack_p,
                                total_n,
                                n_kv,
                                head_dim,
                                n_rot,
                                0,
                                arch.rope_theta,
                            )?;
                            crate::metal::encode_scatter_offset_f32_to_f16_kv(
                                &ctx,
                                enc,
                                &k_normed_pack_p,
                                &v_now_pack_p,
                                &sess.kv_k[ai],
                                &sess.kv_v[ai],
                                0,
                                total_n * kv_dim,
                            )?;
                            for n_idx in 0..total_n {
                                let q_normed_n = q_normed_pack_p
                                    .view_subrange((n_idx * q_dim) as u64, vec![q_dim as u64]);
                                let attn_o_n = attn_o_pack_p
                                    .view_subrange((n_idx * q_dim) as u64, vec![q_dim as u64]);
                                sess.kv_n_pos[ai] = n_idx + 1;
                                let group = n_q / n_kv;
                                let nwg =
                                    crate::metal::attn_v4_choose_nwg(sess.kv_n_pos[ai], group);
                                let tile_c =
                                    crate::metal::attn_v4_choose_tile_c(sess.kv_n_pos[ai], group);
                                crate::metal::encode_attn_decode_v4_f32(
                                    &ctx,
                                    enc,
                                    &q_normed_n,
                                    &sess.kv_k[ai],
                                    &sess.kv_v[ai],
                                    &sess.attn_v4_o_partial,
                                    &sess.attn_v4_ml_partial,
                                    &attn_o_n,
                                    n_q,
                                    n_kv,
                                    head_dim,
                                    sess.kv_n_pos[ai],
                                    nwg,
                                    tile_c,
                                )?;
                            }
                            Ok(())
                        });
                        attn_back_ms += timed("attn_back", &mut |enc| {
                            crate::metal::encode_sigmoid_f32(&ctx, enc, &gate_pack_p, &q_pack_p)?;
                            crate::metal::encode_mul_f32(
                                &ctx,
                                enc,
                                &attn_o_pack_p,
                                &q_pack_p,
                                &attn_o_pack_p,
                            )?;
                            encode_mat_mat_dispatch(
                                &ctx,
                                enc,
                                &a.o,
                                &attn_o_pack_p,
                                &mixer_out_pack_p,
                                q_dim,
                                h,
                                total_n,
                            )?;
                            crate::metal::encode_add_inplace_f32(
                                &ctx,
                                enc,
                                &x_pack_p,
                                &mixer_out_pack_p,
                            )
                            .map_err(crate::metal_forward::MfError::from)
                        });
                    } else {
                        attn_decode_ms += timed("attn_fallback", &mut |enc| {
                            for n_idx in 0..total_n {
                                encode_copy_offset_f32(
                                    &ctx,
                                    enc,
                                    &h_pack_p,
                                    n_idx * h,
                                    &sess.h,
                                    h,
                                )?;
                                mf.encode_attn(enc, a, ai, n_idx as u32, &mut sess)?;
                                crate::metal_forward::encode_scatter_offset_f32(
                                    &ctx,
                                    enc,
                                    &sess.mixer_out,
                                    &mixer_out_pack_p,
                                    n_idx * h,
                                    h,
                                )?;
                            }
                            Ok(())
                        });
                        attn_back_ms += timed("attn_resid", &mut |enc| {
                            crate::metal::encode_add_inplace_f32(
                                &ctx,
                                enc,
                                &x_pack_p,
                                &mixer_out_pack_p,
                            )
                            .map_err(crate::metal_forward::MfError::from)
                        });
                    }
                }
            }

            let mat_mat_path = ffn_mat_mat_eligible(g_w.dtype)
                && ffn_mat_mat_eligible(u_w.dtype)
                && ffn_mat_mat_eligible(d_w.dtype);
            let ffn_gate_pack_p = scratch
                .ffn_gate_pack
                .view_subrange(0, vec![(total_n * f) as u64]);
            let ffn_up_pack_p = scratch
                .ffn_up_pack
                .view_subrange(0, vec![(total_n * f) as u64]);
            let ffn_inner_pack_p = scratch
                .ffn_inner_pack
                .view_subrange(0, vec![(total_n * f) as u64]);
            let ffn_out_pack_p = scratch
                .ffn_out_pack
                .view_subrange(0, vec![(total_n * h) as u64]);
            ffn_ms += timed("ffn", &mut |enc| {
                crate::metal::encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack_p,
                    post_norm,
                    &h_pack_p,
                    total_n,
                    h,
                    crate::metal_forward::RMS_EPS,
                )?;
                if mat_mat_path {
                    encode_mat_mat_dispatch(
                        &ctx,
                        enc,
                        g_w,
                        &h_pack_p,
                        &ffn_gate_pack_p,
                        h,
                        f,
                        total_n,
                    )?;
                    encode_mat_mat_dispatch(
                        &ctx,
                        enc,
                        u_w,
                        &h_pack_p,
                        &ffn_up_pack_p,
                        h,
                        f,
                        total_n,
                    )?;
                    crate::metal::encode_silu_mul_f32(
                        &ctx,
                        enc,
                        &ffn_gate_pack_p,
                        &ffn_up_pack_p,
                        &ffn_inner_pack_p,
                    )?;
                    encode_mat_mat_dispatch(
                        &ctx,
                        enc,
                        d_w,
                        &ffn_inner_pack_p,
                        &ffn_out_pack_p,
                        f,
                        h,
                        total_n,
                    )?;
                } else {
                    for n_idx in 0..total_n {
                        let h_n = h_pack_p.view_subrange((n_idx * h) as u64, vec![h as u64]);
                        let gate_n =
                            ffn_gate_pack_p.view_subrange((n_idx * f) as u64, vec![f as u64]);
                        let up_n = ffn_up_pack_p.view_subrange((n_idx * f) as u64, vec![f as u64]);
                        let inner_n =
                            ffn_inner_pack_p.view_subrange((n_idx * f) as u64, vec![f as u64]);
                        let out_n =
                            ffn_out_pack_p.view_subrange((n_idx * h) as u64, vec![h as u64]);
                        crate::metal_forward::encode_mat_vec_dispatch(
                            &ctx, enc, g_w, &h_n, &gate_n, h, f,
                        )?;
                        crate::metal_forward::encode_mat_vec_dispatch(
                            &ctx, enc, u_w, &h_n, &up_n, h, f,
                        )?;
                        crate::metal::encode_silu_mul_f32(&ctx, enc, &gate_n, &up_n, &inner_n)?;
                        crate::metal_forward::encode_mat_vec_dispatch(
                            &ctx, enc, d_w, &inner_n, &out_n, f, h,
                        )?;
                    }
                }
                crate::metal::encode_add_inplace_f32(&ctx, enc, &x_pack_p, &ffn_out_pack_p)
                    .map_err(crate::metal_forward::MfError::from)
            });
        }

        tail_ms += timed("tail", &mut |enc| {
            let x_last = x_pack_p.view_subrange(((total_n - 1) * h) as u64, vec![h as u64]);
            crate::metal::encode_rms_norm_mul_f32(
                &ctx,
                enc,
                &x_last,
                &mf.model.output_norm,
                &sess.h,
                crate::metal_forward::RMS_EPS,
            )?;
            crate::metal_forward::encode_mat_vec_dispatch(
                &ctx,
                enc,
                &mf.model.lm_head,
                &sess.h,
                &sess.logits,
                h,
                arch.vocab_size as usize,
            )
        });

        let total = embed_ms
            + gdn_front_ms
            + gdn_alpha_beta_ms
            + gdn_tail_ms
            + gdn_back_ms
            + attn_front_ms
            + attn_decode_ms
            + attn_back_ms
            + ffn_ms
            + tail_ms;
        eprintln!(
            "[packed-prefill-phase] prompt_tokens={} chunk_p={chunk_p} phase_sum={total:.2} ms",
            total_n
        );
        eprintln!(
            "[packed-prefill-phase]   embed         {:7.2} ms ({:5.1}%)",
            embed_ms,
            embed_ms / total * 100.0
        );
        eprintln!(
            "[packed-prefill-phase]   gdn_front     {:7.2} ms ({:5.1}%)",
            gdn_front_ms,
            gdn_front_ms / total * 100.0
        );
        eprintln!(
            "[packed-prefill-phase]   gdn_alpha_beta{:7.2} ms ({:5.1}%)",
            gdn_alpha_beta_ms,
            gdn_alpha_beta_ms / total * 100.0
        );
        eprintln!(
            "[packed-prefill-phase]   gdn_tail      {:7.2} ms ({:5.1}%)",
            gdn_tail_ms,
            gdn_tail_ms / total * 100.0
        );
        eprintln!(
            "[packed-prefill-phase]   gdn_back      {:7.2} ms ({:5.1}%)",
            gdn_back_ms,
            gdn_back_ms / total * 100.0
        );
        eprintln!(
            "[packed-prefill-phase]   attn_front    {:7.2} ms ({:5.1}%)",
            attn_front_ms,
            attn_front_ms / total * 100.0
        );
        eprintln!(
            "[packed-prefill-phase]   attn_decode   {:7.2} ms ({:5.1}%)",
            attn_decode_ms,
            attn_decode_ms / total * 100.0
        );
        eprintln!(
            "[packed-prefill-phase]   attn_back     {:7.2} ms ({:5.1}%)",
            attn_back_ms,
            attn_back_ms / total * 100.0
        );
        eprintln!(
            "[packed-prefill-phase]   ffn           {:7.2} ms ({:5.1}%)",
            ffn_ms,
            ffn_ms / total * 100.0
        );
        eprintln!(
            "[packed-prefill-phase]   tail          {:7.2} ms ({:5.1}%)",
            tail_ms,
            tail_ms / total * 100.0
        );
    }

    #[test]
    #[ignore]
    fn metal_27b_packed_prefill_phase_profile() {
        let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(32);
        run_packed_dense_prefill_phase_profile(
            "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf",
            &prompt,
            321,
        );
    }

    #[test]
    #[ignore]
    fn metal_27b_packed_gdn_tail_profile() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[packed-gdn-tail] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok
            .encode(
                &"The quick brown fox jumps over the lazy dog. ".repeat(32),
                false,
            )
            .expect("tokenize");
        let total_n = ids.len();
        let arch = &mm.arch;
        let h = arch.hidden_size as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let n_v = arch.gdn_n_v_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        let conv_dim = (2 * n_k + n_v) * head_dim;
        let v_dim = n_v * head_dim;
        let g = match &mf.model.blocks[0] {
            crate::metal_forward::MetalBlock::Gdn(g) => g,
            _ => panic!("expected block 0 gdn"),
        };

        let sess = MetalSession::fresh(&ctx, &mm, total_n + 16).expect("session");
        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, total_n as u32).expect("scratch");
        let ids_buf = MetalTensor::zeros_f32(&ctx, vec![total_n as u64]).expect("ids buf");
        unsafe {
            let p = ids_buf.buffer.contents().as_ptr() as *mut i32;
            for (i, &t) in ids.iter().enumerate() {
                *p.add(i) = t;
            }
        }

        let x_pack = scratch.x_pack.view_subrange(0, vec![(total_n * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(total_n * h) as u64]);
        let gdn_qkv_pack = scratch
            .gdn_qkv_pack
            .view_subrange(0, vec![(total_n * conv_dim) as u64]);
        let gdn_z_pack = scratch
            .gdn_z_pack
            .view_subrange(0, vec![(total_n * v_dim) as u64]);
        let gdn_beta_pack = scratch
            .gdn_beta_pack
            .view_subrange(0, vec![(total_n * n_v) as u64]);
        let gdn_alpha_pack = scratch
            .gdn_alpha_pack
            .view_subrange(0, vec![(total_n * n_v) as u64]);
        let gdn_normed_pack = scratch
            .gdn_normed_pack
            .view_subrange(0, vec![(total_n * v_dim) as u64]);

        let timed =
            |label: &str,
             cb: &mut dyn FnMut(&KernelEncoder) -> Result<(), crate::metal_forward::MfError>|
             -> f64 {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                cb(&enc).expect(label);
                enc.end();
                cmd.commit();
                cmd.waitUntilCompleted();
                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3
            };

        // Stage packed inputs for one representative GDN block.
        let _ = timed("warm_embed", &mut |enc| {
            let ids_view = ids_buf.view_subrange(0, vec![total_n as u64]);
            encode_get_rows_f32(
                &ctx,
                enc,
                &mf.model.token_embd,
                &ids_view,
                &x_pack,
                total_n,
                h,
            )
            .map_err(crate::metal_forward::MfError::from)
        });
        let _ = timed("warm_norm", &mut |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                &g.attn_norm,
                &h_pack,
                total_n,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .map_err(crate::metal_forward::MfError::from)
        });
        let _ = timed("warm_front", &mut |enc| {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &g.in_proj_qkv,
                &h_pack,
                &gdn_qkv_pack,
                h,
                conv_dim,
                total_n,
            )?;
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &g.in_proj_z,
                &h_pack,
                &gdn_z_pack,
                h,
                v_dim,
                total_n,
            )?;
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &g.beta_proj,
                &h_pack,
                &gdn_beta_pack,
                h,
                n_v,
                total_n,
            )?;
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &g.alpha_proj,
                &h_pack,
                &gdn_alpha_pack,
                h,
                n_v,
                total_n,
            )
        });
        let _ = timed("warm_alpha_beta", &mut |enc| {
            encode_sigmoid_f32(&ctx, enc, &gdn_beta_pack, &gdn_beta_pack)?;
            encode_gdn_decay_chain_batched_f32(
                &ctx,
                enc,
                &gdn_alpha_pack,
                &g.dt_bias,
                &g.a_log,
                &gdn_alpha_pack,
                total_n,
                n_v,
            )
            .map_err(crate::metal_forward::MfError::from)
        });

        let mut conv_ms = 0.0f64;
        let mut l2_ms = 0.0f64;
        let mut step_ms = 0.0f64;
        let mut rms_ms = 0.0f64;
        let mut out_ms = 0.0f64;
        for n_idx in 0..total_n {
            let qkv_n =
                gdn_qkv_pack.view_subrange((n_idx * conv_dim) as u64, vec![conv_dim as u64]);
            let z_n = gdn_z_pack.view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
            let beta_n = gdn_beta_pack.view_subrange((n_idx * n_v) as u64, vec![n_v as u64]);
            let alpha_n = gdn_alpha_pack.view_subrange((n_idx * n_v) as u64, vec![n_v as u64]);
            let normed_n =
                gdn_normed_pack.view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);

            conv_ms += timed("conv", &mut |enc| {
                encode_ssm_conv_silu_f32(
                    &ctx,
                    enc,
                    &qkv_n,
                    &sess.gdn_conv[0],
                    &g.conv1d,
                    &sess.gdn_qkv_conv,
                    conv_dim,
                )
                .map_err(crate::metal_forward::MfError::from)
            });
            let q_view = sess
                .gdn_qkv_conv
                .view_subrange(0, vec![(n_k * head_dim) as u64]);
            let k_view = sess
                .gdn_qkv_conv
                .view_subrange((n_k * head_dim) as u64, vec![(n_k * head_dim) as u64]);
            let v_view = sess
                .gdn_qkv_conv
                .view_subrange((2 * n_k * head_dim) as u64, vec![(n_v * head_dim) as u64]);
            l2_ms += timed("l2", &mut |enc| {
                encode_l2_norm_batched_f32(
                    &ctx,
                    enc,
                    &q_view,
                    &sess.gdn_q_norm,
                    n_k,
                    head_dim,
                    crate::metal_forward::RMS_EPS,
                )?;
                encode_l2_norm_batched_f32(
                    &ctx,
                    enc,
                    &k_view,
                    &sess.gdn_k_norm,
                    n_k,
                    head_dim,
                    crate::metal_forward::RMS_EPS,
                )
                .map_err(crate::metal_forward::MfError::from)
            });
            step_ms += timed("step", &mut |enc| {
                encode_gdn_step_decay_f32(
                    &ctx,
                    enc,
                    &sess.gdn_q_norm,
                    &sess.gdn_k_norm,
                    &v_view,
                    &alpha_n,
                    &beta_n,
                    &sess.gdn_state[0],
                    &sess.gdn_out,
                    n_v,
                    n_k,
                    head_dim,
                )
                .map_err(crate::metal_forward::MfError::from)
            });
            rms_ms += timed("rmsnorm_gated", &mut |enc| {
                encode_rmsnorm_gated_f32(
                    &ctx,
                    enc,
                    &sess.gdn_out,
                    &g.norm,
                    &z_n,
                    &normed_n,
                    n_v,
                    head_dim,
                    crate::metal_forward::RMS_EPS,
                )
                .map_err(crate::metal_forward::MfError::from)
            });
            out_ms += timed("out_proj", &mut |enc| {
                crate::metal_forward::encode_mat_vec_dispatch(
                    &ctx,
                    enc,
                    &g.out_proj,
                    &normed_n,
                    &sess.mixer_out,
                    v_dim,
                    h,
                )
            });
        }

        let total = conv_ms + l2_ms + step_ms + rms_ms + out_ms;
        eprintln!(
            "[packed-gdn-tail] prompt_tokens={} one-layer total={total:.2} ms",
            total_n
        );
        eprintln!(
            "[packed-gdn-tail]   conv           {:7.2} ms ({:5.1}%)",
            conv_ms,
            conv_ms / total * 100.0
        );
        eprintln!(
            "[packed-gdn-tail]   l2             {:7.2} ms ({:5.1}%)",
            l2_ms,
            l2_ms / total * 100.0
        );
        eprintln!(
            "[packed-gdn-tail]   step_decay     {:7.2} ms ({:5.1}%)",
            step_ms,
            step_ms / total * 100.0
        );
        eprintln!(
            "[packed-gdn-tail]   rmsnorm_gated  {:7.2} ms ({:5.1}%)",
            rms_ms,
            rms_ms / total * 100.0
        );
        eprintln!(
            "[packed-gdn-tail]   out_proj       {:7.2} ms ({:5.1}%)",
            out_ms,
            out_ms / total * 100.0
        );
    }

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
            None, // n_eff_override (test always uses full N)
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

    /// **v0.76 adaptive-N back-off correctness gate**: confirm that
    /// `encode_packed_verify_layer_major_inner` with `n_eff_override
    /// = Some(n_eff)` produces argmaxes EQUAL to a fresh full-N=block
    /// run on the first `n_eff` tokens, AND advances session state
    /// (gdn_state, gdn_conv, kv_n_pos for attn layers) consistently
    /// with running a single_token loop on those `n_eff` tokens.
    ///
    /// The greedy-equivalence story for adaptive N hinges on this: the
    /// verify math doesn't change, we just process fewer tokens. The
    /// argmax tokens accepted should be identical OVER THE FIRST n_eff
    /// SLOTS regardless of whether N=16 or N=8 was used.
    ///
    /// Uses 0.8B-F32 (lib loop, fast). 27B integration test follows in
    /// dflash_correctness.rs.
    #[test]
    fn dflash_packed_verify_n_eff_override_equiv() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[n_eff-equiv] skipped — fixture missing");
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

        // Two identically-primed sessions: M=2 priming tokens.
        const M: u32 = 2;
        const N_BLOCK: u32 = 16;
        let prime_tokens: [i32; M as usize] = [9419, 1];
        // 8 verify tokens (we'll run them via n_eff_override=Some(8)).
        let verify_tokens_8: [i32; 8] = [1234, 7, 999, 42, 11, 22, 33, 44];

        let mut sess_full = MetalSession::fresh(&ctx, &mm, 64).expect("sess full");
        let mut sess_eff = MetalSession::fresh(&ctx, &mm, 64).expect("sess eff");
        for (i, &tok) in prime_tokens.iter().enumerate() {
            mf.single_token(tok, i as u32, &mut sess_full)
                .expect("prime full");
            mf.single_token(tok, i as u32, &mut sess_eff)
                .expect("prime eff");
        }

        let target_layer_ids: Vec<u32> = vec![5, 15];
        let k_target = target_layer_ids.len() as u32;

        // Path A: full N=8 scratch, no override (baseline behavior).
        let mut verify_8 =
            MetalDFlashVerifyScratch::fresh(&ctx, &mm, 8, k_target).expect("verify_8");
        let mut layer_8 = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, 8).expect("layer_8");
        let argmax_full_8 = encode_packed_verify_layer_major_inner(
            &mf,
            &target_layer_ids,
            &verify_tokens_8,
            M,
            &mut verify_8,
            &mut layer_8,
            &mut sess_full,
            None,
            None, // no override; verify N=8 from scratch shape
        )
        .expect("packed_verify N=8 full");

        // Path B: N=16 scratch, n_eff_override=Some(8), only 8 verify tokens.
        let mut verify_16 =
            MetalDFlashVerifyScratch::fresh(&ctx, &mm, N_BLOCK, k_target).expect("verify_16");
        let mut layer_16 =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, N_BLOCK).expect("layer_16");
        let argmax_eff_8 = encode_packed_verify_layer_major_inner(
            &mf,
            &target_layer_ids,
            &verify_tokens_8,
            M,
            &mut verify_16,
            &mut layer_16,
            &mut sess_eff,
            None,
            Some(8), // override truncates effective chain to 8
        )
        .expect("packed_verify N=16 scratch with n_eff=8 override");

        eprintln!("[n_eff-equiv] argmax_full_8={argmax_full_8:?} argmax_eff_8={argmax_eff_8:?}");

        // Bit-exact argmax — same math, just allocated differently.
        assert_eq!(
            argmax_full_8, argmax_eff_8,
            "n_eff=8 override produces different argmaxes than fresh N=8 scratch"
        );
        // Returned vec length matches n_eff, NOT n_block.
        assert_eq!(argmax_eff_8.len(), 8);

        // Bit-exact session state (GDN state, conv, kv_n_pos).
        for (i, (a, b)) in sess_full
            .gdn_state
            .iter()
            .zip(sess_eff.gdn_state.iter())
            .enumerate()
        {
            unsafe {
                let pa = a.buffer.contents().as_ptr() as *const u32;
                let pb = b.buffer.contents().as_ptr() as *const u32;
                let n_elems = a.n_elements() as usize;
                for j in 0..n_elems {
                    if *pa.add(j) != *pb.add(j) {
                        panic!("gdn_state[{i}][{j}] diverges between full-N=8 and N=16+override=8");
                    }
                }
            }
        }
        for (i, (a, b)) in sess_full
            .gdn_conv
            .iter()
            .zip(sess_eff.gdn_conv.iter())
            .enumerate()
        {
            unsafe {
                let pa = a.buffer.contents().as_ptr() as *const u32;
                let pb = b.buffer.contents().as_ptr() as *const u32;
                let n_elems = a.n_elements() as usize;
                for j in 0..n_elems {
                    if *pa.add(j) != *pb.add(j) {
                        panic!("gdn_conv[{i}][{j}] diverges between full-N=8 and N=16+override=8");
                    }
                }
            }
        }
        assert_eq!(
            sess_full.kv_n_pos, sess_eff.kv_n_pos,
            "kv_n_pos diverges between full-N=8 and N=16+override=8"
        );

        // Edge cases on the override itself.
        // n_eff=0 → error.
        let err = encode_packed_verify_layer_major_inner(
            &mf,
            &target_layer_ids,
            &[],
            M + 8, // start_position past the prior call
            &mut verify_16,
            &mut layer_16,
            &mut sess_eff,
            None,
            Some(0),
        );
        match err {
            Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                assert!(
                    detail.contains("n=0") || detail.contains("[1, n_block"),
                    "expected n_eff=0 → BadShape with range error: {detail}"
                );
            }
            other => panic!("expected BadShape on n_eff=0, got {other:?}"),
        }

        // n_eff > n_block → error.
        let err = encode_packed_verify_layer_major_inner(
            &mf,
            &target_layer_ids,
            &[1i32; 17],
            M + 8,
            &mut verify_16,
            &mut layer_16,
            &mut sess_eff,
            None,
            Some(17), // > N_BLOCK=16
        );
        match err {
            Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                assert!(
                    detail.contains("[1, n_block"),
                    "expected n_eff=17 → BadShape with range error: {detail}"
                );
            }
            other => panic!("expected BadShape on n_eff>n_block, got {other:?}"),
        }
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

            encode_restore_after_partial_accept_inner(&mf, &scratch, n_keep, M, &mut sess_a, None)
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
        let err = encode_restore_after_partial_accept_inner(&mf, &scratch, 0, 0, &mut sess, None);
        match err {
            Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                assert!(detail.contains("n_keep=0"), "wrong error: {detail}");
            }
            other => panic!("expected BadShape on n_keep=0, got {other:?}"),
        }

        // (b) n_keep > N
        let err =
            encode_restore_after_partial_accept_inner(&mf, &scratch, N + 1, 0, &mut sess, None);
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
            let err =
                encode_restore_after_partial_accept_inner(&mf, &scratch, 2, 99, &mut sess, None);
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
    /// 0.8B-F32 has both GDN and attention blocks, so this gates the GDN
    /// recurrence, dense FFN mat-mat, attention projection mat-mat, and tail.
    /// The larger 27B integration test in `tests/dflash_correctness.rs` still
    /// gates production-shape attention profiling.
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
                MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
                    &ctx,
                    &mm,
                    p as u32,
                    n_total_with_prefix,
                )
                .expect("layer scratch");
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

    /// **MoE prefill correctness gate** (Qwen3.6-35B-A3B-UD-Q4_K_M, with
    /// gate/up expert banks in Q4_K and down expert banks in a mixed Q5_K/Q6_K
    /// set). Oracle: a sequential `single_token` loop. Experimental: one
    /// `prefill_tokens_with_multi_hidden` call
    /// (with empty capture-layers since the per-layer hidden-capture
    /// helper returns `UnsupportedMoe` on MoE arches; the lr1 / packed
    /// MoE bugs we're guarding against still propagate to final logits
    /// and GDN/KV session state, so dropping the per-layer capture
    /// loses debug locality but does not weaken the gate).
    ///
    /// `prefill_moe_grouped_enabled()` and `prefill_moe_packed_routed_
    /// enabled()` both latch into process-wide `OnceLock`s on first
    /// call, so a single test invocation can only validate whichever
    /// branch wins per process. To exercise both:
    /// ```bash
    /// cargo test ..._35b_a3b_moe --release -- --ignored --nocapture           # grouped (default)
    /// QWEN_PREFILL_MOE_GROUPED=0 cargo test ..._35b_a3b_moe --release -- --ignored --nocapture  # packed-routed
    /// ```
    /// Without this gate, the grouped path was default-on but had no
    /// non-`#[ignore]` MoE prefill correctness coverage; the May 2026
    /// Q4 grouped kernel `lr1` clamp bug (`kernels/moe.metal:686`,
    /// missing clamp to `nr1 - 1`) would have been caught here.
    ///
    /// Cosine equivalence ≥ 0.999 required on logits, GDN state+conv
    /// per layer, and exact equality on KV pos counters.
    ///
    /// `#[ignore]` because A3B-Q4 dequant + 30B-param prefill is several
    /// minutes per scenario.
    #[test]
    #[ignore]
    fn prefill_tokens_matches_single_token_loop_35b_a3b_moe() {
        let path = "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[moe-prefill-vs-single] skipped — fixture missing");
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
        assert_eq!(
            m.arch.kind,
            crate::model::ArchKind::Moe,
            "expected A3B MoE arch; got {:?}",
            m.arch.kind
        );
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        let arch = &mm.arch;

        // Branch labeling is informational: see test-level doc — the
        // MoE-prefill path is locked per-process by OnceLock on first
        // call. The label here reflects whichever env-flag state was
        // active at process start.
        let active_branch = if std::env::var("QWEN_PREFILL_MOE_GROUPED")
            .as_deref()
            .map(|v| matches!(v, "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(false)
        {
            "packed-routed (grouped disabled via env)"
        } else {
            "grouped (default)"
        };
        eprintln!("[moe-prefill-vs-single] active branch: {active_branch}");

        // NOTE: `single_token_with_multi_hidden` returns UnsupportedMoe,
        // so the oracle for MoE has to be plain `single_token` and the
        // experimental path passes `&[]` to skip multi-hidden capture.
        // The lr1 clamp bug we're guarding against propagates from the
        // MoE swiglu output → next layer residual → final logits and
        // GDN/KV state — so dropping the per-layer hidden capture
        // loses some debug locality but does NOT weaken the gate.

        let run_scenario = |scenario_label: &str, total_n: usize, p: usize, prefix_len: usize| {
            let branch_label = format!("{scenario_label} | {active_branch}");
            assert!(total_n >= 1 && p >= 1, "{branch_label}: need n,p ≥ 1");

            // Deterministic synthetic token sequence (skip 0 to avoid
            // any specialness around <bos>; vocab on A3B is ~150k).
            let n_total_with_prefix = prefix_len + total_n;
            let all_tokens: Vec<i32> = (0..n_total_with_prefix)
                .map(|i| ((i * 13 + 7) % (arch.vocab_size as usize - 1)) as i32 + 1)
                .collect();
            let prefix_tokens = &all_tokens[..prefix_len];
            let token_ids = &all_tokens[prefix_len..];

            // ---- Oracle path: sequential single_token loop. ----
            let cap = n_total_with_prefix + 4;
            let mut sess_a = MetalSession::fresh(&ctx, &mm, cap).expect("sess A");
            for (i, &tid) in prefix_tokens.iter().enumerate() {
                mf.single_token(tid, i as u32, &mut sess_a)
                    .expect("oracle prefix advance");
            }
            let mut last_a = Vec::new();
            for (i, &tid) in token_ids.iter().enumerate() {
                last_a = mf
                    .single_token(tid, (prefix_len + i) as u32, &mut sess_a)
                    .expect("oracle forward");
            }

            // ---- Experimental path: one prefill_tokens call. ----
            let mut sess_b = MetalSession::fresh(&ctx, &mm, cap).expect("sess B");
            for (i, &tid) in prefix_tokens.iter().enumerate() {
                mf.single_token(tid, i as u32, &mut sess_b)
                    .expect("experimental prefix advance");
            }
            let mut layer_scratch =
                MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
                    &ctx,
                    &mm,
                    p as u32,
                    n_total_with_prefix,
                )
                .expect("layer scratch");
            let last_b = prefill_tokens_with_multi_hidden(
                &mf,
                token_ids,
                prefix_len as u32,
                &mut sess_b,
                &mut layer_scratch,
                &[],
                None,
            )
            .expect("prefill");

            // ---- Compare final logits (cos ≥ 0.999). ----
            assert_eq!(
                last_a.len(),
                last_b.len(),
                "{branch_label}: logits len mismatch"
            );
            let cos_logits = cosine_f32(&last_a, &last_b);
            eprintln!(
                "[moe-prefill-vs-single] {branch_label}: T={total_n} P={p} prefix={prefix_len} chunks={} cos(logits)={cos_logits:.6}",
                total_n.div_ceil(p)
            );
            assert!(
                cos_logits >= 0.999,
                "{branch_label}: logits cos={cos_logits} < 0.999"
            );

            // ---- Compare GDN state + conv per layer (cos ≥ 0.999). ----
            assert_eq!(
                sess_a.gdn_state.len(),
                sess_b.gdn_state.len(),
                "{branch_label}: GDN state vec length mismatch"
            );
            let mut min_gdn_cos = f64::INFINITY;
            let mut worst_gdn = 0usize;
            for gi in 0..sess_a.gdn_state.len() {
                let a_state = read_tensor_f32(&sess_a.gdn_state[gi]);
                let b_state = read_tensor_f32(&sess_b.gdn_state[gi]);
                let cs = cosine_f32(&a_state, &b_state);
                let a_conv = read_tensor_f32(&sess_a.gdn_conv[gi]);
                let b_conv = read_tensor_f32(&sess_b.gdn_conv[gi]);
                let cc = cosine_f32(&a_conv, &b_conv);
                let layer_min = cs.min(cc);
                if layer_min < min_gdn_cos {
                    min_gdn_cos = layer_min;
                    worst_gdn = gi;
                }
                assert!(
                    cs >= 0.999,
                    "{branch_label}: GDN[{gi}] state cos={cs} < 0.999"
                );
                assert!(
                    cc >= 0.999,
                    "{branch_label}: GDN[{gi}] conv cos={cc} < 0.999"
                );
            }
            eprintln!(
                "[moe-prefill-vs-single] {branch_label}: gdn cos_min={min_gdn_cos:.6} (layer={worst_gdn})"
            );

            // ---- KV pos counters (A3B has attn layers). ----
            assert_eq!(sess_a.kv_n_pos.len(), sess_b.kv_n_pos.len());
            for ai in 0..sess_a.kv_n_pos.len() {
                assert_eq!(
                    sess_a.kv_n_pos[ai], sess_b.kv_n_pos[ai],
                    "{branch_label}: kv_n_pos[{ai}] mismatch ({} vs {})",
                    sess_a.kv_n_pos[ai], sess_b.kv_n_pos[ai]
                );
            }
        };

        // Scenarios: kept small because A3B prefill is heavy (~30B
        // active during MoE swiglu/down across all 8-of-128 experts).
        // P=8 matches the production drafter block_size for A3B.
        run_scenario("T=5 single-chunk", 5, 8, 0);
        run_scenario("T=11/P=8 multi-chunk", 11, 8, 0);
        run_scenario("start_position>0 (prefix=2, T=6)", 6, 8, 2);
        run_scenario(
            "A3B packed-attn active-shape (prefix=4096, T=4, P=8)",
            4,
            8,
            4096,
        );
        run_scenario(
            "A3B packed-attn threshold-cross (prefix=4095, T=2, P=8)",
            2,
            8,
            4095,
        );
        run_scenario(
            "A3B packed-attn single-row (prefix=4096, T=1, P=8)",
            1,
            8,
            4096,
        );
        run_scenario(
            "A3B packed-attn full-tile (prefix=4096, T=8, P=8)",
            8,
            8,
            4096,
        );
        run_scenario(
            "A3B packed-attn multi-tile (prefix=4096, T=16, P=16)",
            16,
            16,
            4096,
        );
        run_scenario(
            "A3B packed-attn late-prefix (prefix=8191, T=2, P=8)",
            2,
            8,
            8191,
        );
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

    fn read_tensor_bytes(t: &MetalTensor) -> Vec<u8> {
        let n = t.n_bytes() as usize;
        let mut out = vec![0u8; n];
        unsafe {
            let src = (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    }

    fn fuse_q4k_gate_up_expert_banks(
        ctx: &MetalContext,
        gate: &MetalTensor,
        up: &MetalTensor,
        n_hidden: usize,
        n_ffn: usize,
        n_expert: usize,
    ) -> MetalTensor {
        const Q4K_BYTES: usize = 144;
        let gate_bytes = read_tensor_bytes(gate);
        let up_bytes = read_tensor_bytes(up);
        let row_bytes = (n_hidden / 256) * Q4K_BYTES;
        let blocks_per_row = row_bytes / Q4K_BYTES;
        let per_expert_bytes = row_bytes * n_ffn;
        assert_eq!(gate_bytes.len(), per_expert_bytes * n_expert);
        assert_eq!(up_bytes.len(), per_expert_bytes * n_expert);
        let mut fused = Vec::with_capacity(gate_bytes.len() * 2);
        for expert in 0..n_expert {
            let g_exp = &gate_bytes[expert * per_expert_bytes..(expert + 1) * per_expert_bytes];
            let u_exp = &up_bytes[expert * per_expert_bytes..(expert + 1) * per_expert_bytes];
            for row in 0..n_ffn {
                let row_off = row * row_bytes;
                for block in 0..blocks_per_row {
                    let off = row_off + block * Q4K_BYTES;
                    fused.extend_from_slice(&g_exp[off..off + Q4K_BYTES]);
                    fused.extend_from_slice(&u_exp[off..off + Q4K_BYTES]);
                }
            }
        }
        let fused_elems = (fused.len() / Q4K_BYTES) * 256;
        MetalTensor::from_bytes(ctx, &fused, vec![fused_elems as u64], GgmlType::Q4_K)
            .expect("fused q4k gate/up")
    }
}
const ATTN_PREFILL_V4_PACKED_ROWS: usize = 8;
