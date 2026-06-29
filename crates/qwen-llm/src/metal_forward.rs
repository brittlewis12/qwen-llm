//! End-to-end Metal forward pass driver.
//!
//! Mirrors `crate::forward::Forward` but with all heavy ops on Metal,
//! orchestrated through one `MTLCommandBuffer` per token. The CPU oracle
//! (`forward.rs`) stays as the validation reference.
//!
//! v1 scope:
//!
//! * Single-token decode (no batched prefill).
//! * F32-only first (Qwen3.5-0.8B.F32). Quantized lift comes after the
//!   F32 path validates against the oracle — every kernel that takes
//!   `mat_vec_f32` will swap to `mat_vec_q4_K` / `mat_vec_q6_K` based on
//!   tensor `dtype`.
//! * Full-attn layers built from existing kernels (q-proj, k-proj,
//!   v-proj, q-norm, k-norm, RoPE, KV append, scoring, softmax,
//!   V-aggregate, gate, output proj). No fused attn block yet — the
//!   composition lands first; a fused version is a v2 perf project.
//! * Persistent `MetalTensor`s for all weights; one `MetalSession` owns
//!   the per-sequence state (GDN conv buffers + SSM states + KV cache).
//! * Activation arena: per-step scratch buffers held in `MetalSession`,
//!   reused across layers.
//!
//! v2 lift after the F32 forward validates:
//!
//! * Quantized weight path — change `MetalModel::load` to use
//!   `from_gguf_tensor` directly (currently dequants F32 via codec).
//! * Indirect Command Buffer (ICB) — encode the per-token sequence once,
//!   replay it. The encode-only API is already designed for this.
//! * Fused full-attn block.

use crate::gguf::GgufFile;
use crate::loader::{Block, Model, MoeFfn};
use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, attn_v4_choose_nwg,
    attn_v4_choose_tile_c, encode_add_inplace_f32, encode_argmax_f32, encode_attn_decode_f16kv_f32,
    encode_attn_decode_v4_f32, encode_axpy_f32, encode_axpy_scalar_f32, encode_dot_sigmoid_f32,
    encode_ffn_swiglu_q4_K_f32, encode_fill_f32, encode_gdn_decay_chain_f32,
    encode_gdn_step_decay_f32, encode_get_rows_f32, encode_l2_norm_batched_f32, encode_mat_vec_f32,
    encode_mat_vec_q4_k_f32, encode_mat_vec_q5_k_f32, encode_mat_vec_q6_k_f32,
    encode_moe_down_bf16_f32, encode_moe_down_iq4_xs_f32, encode_moe_down_q5_K_f32,
    encode_moe_down_weighted_sum_q5_K_f32_packed_slots,
    encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2,
    encode_moe_down_weighted_sum_q6_K_f32, encode_moe_mat_vec_bf16_f32, encode_moe_mat_vec_f32,
    encode_moe_mat_vec_iq3_s_f32, encode_moe_mat_vec_iq3_xxs_f32, encode_moe_mat_vec_q5_K_f32,
    encode_moe_shared_accum_resid_f32, encode_moe_swiglu_q4_K_f32, encode_moe_weighted_sum_f32,
    encode_mul_f32, encode_rms_norm_batched_f32, encode_rms_norm_mul_f32, encode_rmsnorm_gated_f32,
    encode_rope_neox_f32, encode_scatter_offset_f32_to_f16_kv,
    encode_scatter_offset_f32_to_q8_0_kv, encode_shared_swiglu_q8_0_f32, encode_sigmoid_f32,
    encode_sigmoid_mul_f32, encode_silu_mul_f32, encode_split_q_gate_f32, encode_ssm_conv_silu_f32,
    encode_topk_logits_softmax_dot_sigmoid_f32, encode_topk_logits_softmax_f32,
};
use crate::model::ArchKind;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use std::{cell::Cell, sync::OnceLock};

/// Max NWG (split-K partitions) the v4 dispatcher will ever request.
/// Sets the size of session-resident partial buffers; see
/// `attn_v4_choose_nwg` for the selection heuristic.
pub const ATTN_V4_MAX_NWG: usize = 64;
use crate::tensor::{GgmlType, TensorDesc};

/// Single source of truth for which weight dtypes the loader keeps in
/// their native form (vs. dequant'ing to F32). Used by both `load_weight`
/// in `MetalModel::load` and the byte-ledger diagnostic, so they stay
/// in sync. If you add a new native quant kernel, list its dtype here.
///
/// Q8_0 added v0.73b.1 — DFlash drafter switches from F32-dequant
/// resident (~7.4 GB) to native Q8_0 (~1.85 GB). F16/BF16 stay native once
/// their primitive mat-vec/mat-mat/get_rows kernels are available.
pub fn weight_dtype_kept_native(dtype: GgmlType) -> bool {
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

fn moe_iq3_expert_native_enabled(desc: &TensorDesc) -> bool {
    let truthy = |v: &str| matches!(v, "1" | "true" | "TRUE" | "yes" | "YES");
    let falsey = |v: &str| matches!(v, "0" | "false" | "FALSE" | "no" | "NO");
    if let Ok(v) = std::env::var("QWEN_MOE_IQ3_EXPERT_NATIVE") {
        if truthy(&v) {
            return true;
        }
        if falsey(&v) {
            return false;
        }
    }
    if let Ok(v) = std::env::var("QWEN_PREFILL_MOE_GROUPED_IQ3_GATEUP") {
        if truthy(&v) {
            return true;
        }
        if falsey(&v) {
            return false;
        }
    }
    desc.shape.len() >= 3 && desc.shape[0..3] == [2048, 512, 256]
}

fn alloc_shape_error(detail: &'static str) -> MetalError {
    MetalError::BadShape {
        kernel: "session_alloc",
        detail: detail.into(),
    }
}

pub(crate) fn checked_u64_mul(a: u64, b: u64, detail: &'static str) -> Result<u64, MetalError> {
    a.checked_mul(b).ok_or_else(|| alloc_shape_error(detail))
}

pub(crate) fn checked_u64_add(a: u64, b: u64, detail: &'static str) -> Result<u64, MetalError> {
    a.checked_add(b).ok_or_else(|| alloc_shape_error(detail))
}

pub(crate) fn checked_u64_mul3(
    a: u64,
    b: u64,
    c: u64,
    detail: &'static str,
) -> Result<u64, MetalError> {
    checked_u64_mul(checked_u64_mul(a, b, detail)?, c, detail)
}

pub(crate) fn checked_u64_mul4(
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    detail: &'static str,
) -> Result<u64, MetalError> {
    checked_u64_mul(checked_u64_mul3(a, b, c, detail)?, d, detail)
}

pub(crate) fn checked_u64_double(a: u64, detail: &'static str) -> Result<u64, MetalError> {
    checked_u64_mul(a, 2, detail)
}

pub(crate) fn checked_u64_div_exact(
    numerator: u64,
    denominator: u64,
    detail: &'static str,
) -> Result<u64, MetalError> {
    if denominator == 0 {
        return Err(alloc_shape_error(detail));
    }
    if numerator % denominator != 0 {
        return Err(alloc_shape_error(detail));
    }
    Ok(numerator / denominator)
}

fn kv_cache_dtype_for_arch(arch: &crate::model::Arch) -> GgmlType {
    static KV_Q8: OnceLock<bool> = OnceLock::new();
    let enabled = *KV_Q8.get_or_init(|| {
        matches!(
            std::env::var("QWEN_KV_Q8").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    });
    let group = (arch.n_q_heads / arch.n_kv_heads.max(1)) as usize;
    if enabled && arch.kind == ArchKind::Dense && arch.attn_head_dim == 256 && group == 6 {
        GgmlType::Q8_0
    } else {
        GgmlType::F16
    }
}

fn concurrent_gdn_moe_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_DECODE_MOE_CONCURRENT_GDN").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn concurrent_gdn_dense_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_DECODE_DENSE_CONCURRENT_GDN").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn concurrent_shared_moe_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_DECODE_MOE_CONCURRENT_SHARED").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn decode_shared_swiglu_q8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_DECODE_SHARED_SWIGLU_Q8").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn decode_moe_q5_down_fused_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_DECODE_MOE_Q5_DOWN_FUSED").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn decode_moe_q5_down_k512_r2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_DECODE_MOE_Q5_DOWN_K512_R2").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn decode_moe_fused_finalizer_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_DECODE_MOE_FUSED_FINALIZER").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn decode_attn_sigmoid_mul_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_DECODE_ATTN_SIGMOID_MUL").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

fn decode_gdn_noop_front_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_DECODE_GDN_NOOP_FRONT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

fn decode_gdn_noop_out_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_DECODE_GDN_NOOP_OUT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

fn decode_gdn_noop_qkv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    decode_gdn_noop_front_enabled()
        || *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QWEN_DECODE_GDN_NOOP_QKV").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        })
}

fn decode_gdn_noop_z_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    decode_gdn_noop_front_enabled()
        || *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QWEN_DECODE_GDN_NOOP_Z").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        })
}

fn decode_gdn_noop_beta_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    decode_gdn_noop_front_enabled()
        || *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QWEN_DECODE_GDN_NOOP_BETA").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        })
}

fn decode_gdn_noop_alpha_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    decode_gdn_noop_front_enabled()
        || *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QWEN_DECODE_GDN_NOOP_ALPHA").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        })
}

fn phase_moe_ffn_split_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_PHASE_MOE_FFN_SPLIT").as_deref(),
            Ok("1")
                | Ok("2")
                | Ok("deep")
                | Ok("DEEP")
                | Ok("true")
                | Ok("TRUE")
                | Ok("yes")
                | Ok("YES")
        )
    })
}

fn phase_moe_ffn_deep_split_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_PHASE_MOE_FFN_SPLIT").as_deref(),
            Ok("2") | Ok("deep") | Ok("DEEP")
        )
    })
}

fn phase_gdn_proj_split_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_PHASE_GDN_PROJ_SPLIT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

fn decode_moe_noop_routed_gateup_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_DECODE_MOE_NOOP_ROUTED_GATEUP").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

fn decode_moe_noop_routed_down_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_DECODE_MOE_NOOP_ROUTED_DOWN").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

thread_local! {
    static MATMAT_BF16_BFLOAT_ACT_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

pub fn with_matmat_bf16_bfloat_act_override<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    let previous = MATMAT_BF16_BFLOAT_ACT_OVERRIDE.with(|slot| {
        let previous = slot.get();
        slot.set(Some(enabled));
        previous
    });
    let out = f();
    MATMAT_BF16_BFLOAT_ACT_OVERRIDE.with(|slot| slot.set(previous));
    out
}

fn matmat_bf16_bfloat_act_enabled() -> bool {
    if let Some(enabled) = MATMAT_BF16_BFLOAT_ACT_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("QWEN_MATMAT_BF16_BFLOAT_ACT").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

/// Return type for [`MetalForward::single_token_phase_profiled`]:
/// `(logits, wall_with_artifact_ms, per-phase GPU ms map)`. The
/// per-phase entries are `(phase_name, gpu_ms)`.
pub type PhaseProfileOutput = (Vec<f32>, f64, Vec<(String, f64)>);

use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState};

#[derive(Debug, thiserror::Error)]
pub enum MfError {
    #[error("metal: {0}")]
    Metal(#[from] MetalError),
    #[error("codec: {0}")]
    Codec(#[from] crate::codec::CodecError),
    #[error("token {0} out of vocab range {1}")]
    BadToken(i32, u32),
    #[error("qwen35moe Metal path not implemented yet")]
    UnsupportedMoe,
    #[error("v1 driver requires F32 weights; tensor {name} is {dtype:?}")]
    UnsupportedDtype { name: String, dtype: GgmlType },
}

/// All weight tensors, resident as `MetalTensor`s. Loaded once at session
/// start. v1: only F32 weights are supported here; quantized path comes
/// in v2 by switching `MetalModel::load` to use `MetalTensor::from_gguf_tensor`
/// directly (instead of going through the F32 codec) and the kernel
/// dispatchers to pick `_q4_k`/`_q6_k` based on dtype.
pub struct MetalModel {
    /// Reference back to the loader's bound model. Carries `arch`, the
    /// layer schedule (GDN vs Attn), tied-embedding flag.
    pub arch: crate::model::Arch,
    pub tied_embeddings: bool,

    pub token_embd: MetalTensor,
    pub output_norm: MetalTensor,
    pub lm_head: MetalTensor,

    pub blocks: Vec<MetalBlock>,
}

pub enum MetalBlock {
    Gdn(MetalGdnBlock),
    Attn(MetalAttnBlock),
}

pub struct MetalGdnBlock {
    pub attn_norm: MetalTensor,
    pub post_attn_norm: MetalTensor,
    pub ffn_gate: MetalTensor,
    pub ffn_up: MetalTensor,
    pub ffn_down: MetalTensor,
    pub in_proj_qkv: MetalTensor,
    pub in_proj_z: MetalTensor,
    pub beta_proj: MetalTensor,
    pub alpha_proj: MetalTensor,
    pub a_log: MetalTensor,
    pub dt_bias: MetalTensor,
    pub conv1d: MetalTensor,
    pub norm: MetalTensor,
    pub out_proj: MetalTensor,
    pub ffn_moe: Option<MetalMoeFfn>,
}

pub struct MetalAttnBlock {
    pub attn_norm: MetalTensor,
    pub post_attn_norm: MetalTensor,
    pub ffn_gate: MetalTensor,
    pub ffn_up: MetalTensor,
    pub ffn_down: MetalTensor,
    pub q: MetalTensor, // outputs 2x q_dim — Q + gate
    pub k: MetalTensor,
    pub v: MetalTensor,
    pub qkv_fused: Option<MetalTensor>,
    pub o: MetalTensor,
    pub q_norm: MetalTensor,
    pub k_norm: MetalTensor,
    pub ffn_moe: Option<MetalMoeFfn>,
}

pub struct MetalMoeFfn {
    pub gate_inp: MetalTensor,
    pub gate_exps: MetalTensor,
    pub up_exps: MetalTensor,
    pub down_exps: MetalTensor,
    pub gate_inp_shexp: MetalTensor,
    pub gate_inp_cpu: Vec<f32>,
    pub gate_inp_shexp_cpu: Vec<f32>,
}

#[derive(Clone, Copy)]
enum MixerSlot {
    Gdn(usize),
    Attn(usize),
}

#[allow(dead_code)]
struct MoeRouteDecision {
    ranked: Vec<(usize, f32)>,
    shared_gate_scalar: f32,
}

impl MetalModel {
    /// Load weights from an `loader::Model` view. Native-quant path:
    /// keeps weight tensors at their on-disk dtype (Q4_K, Q6_K, F32,
    /// etc.) and the kernel dispatchers pick the right `encode_mat_vec_*`
    /// based on dtype.
    ///
    /// For weights that aren't matmul'd by a quant-supporting kernel
    /// (e.g. norms, ssm_a, dt_bias — they need F32 for the elementwise
    /// kernels), we dequant via the codec at load time. The big tensors
    /// (mat_vec inputs, embeddings, lm_head) keep their native dtype.
    pub fn load(ctx: &MetalContext, gguf: &GgufFile, model: &Model<'_>) -> Result<Self, MfError> {
        // Helper: load a tensor that *must* be F32 in memory (used by
        // elementwise kernels, norms, etc.). Dequants via codec if needed.
        let load_f32 = |desc: &TensorDesc| -> Result<MetalTensor, MfError> {
            if desc.dtype == GgmlType::F32 {
                Ok(MetalTensor::from_gguf_tensor(ctx, desc, gguf.slice(desc))?)
            } else {
                let f32 = crate::codec::dequant_to_f32(desc, gguf.slice(desc))?;
                Ok(MetalTensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&f32),
                    desc.shape.clone(),
                    GgmlType::F32,
                )?)
            }
        };
        // Helper: load a tensor that's a mat_vec/mat_mat weight. Keeps native
        // dtype for dtypes covered by the primitive dispatchers; falls back to
        // F32 conversion for types we don't have native kernels for yet.
        let load_weight = |desc: &TensorDesc| -> Result<MetalTensor, MfError> {
            if weight_dtype_kept_native(desc.dtype) {
                Ok(MetalTensor::from_gguf_tensor(ctx, desc, gguf.slice(desc))?)
            } else {
                eprintln!(
                    "[metal-load] {} is {:?}; dequanting to F32 (no active native path)",
                    desc.name, desc.dtype
                );
                load_f32(desc)
            }
        };
        let load_embedding = |desc: &TensorDesc| -> Result<MetalTensor, MfError> {
            if matches!(desc.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
                Ok(MetalTensor::from_gguf_tensor(ctx, desc, gguf.slice(desc))?)
            } else {
                load_f32(desc)
            }
        };
        // Existing alias for the call sites below.
        let load_tensor = load_f32;

        // Embedding has native get_rows kernels for F32/F16/BF16. Quantized
        // embeddings still dequant to F32 until they get native get_rows.
        let token_embd = load_embedding(model.token_embd)?;
        let output_norm = load_f32(model.output_norm)?;
        let lm_head = load_weight(model.lm_head)?;

        let load_moe_expert = |desc: &TensorDesc| -> Result<MetalTensor, MfError> {
            if matches!(desc.dtype, GgmlType::IQ3_XXS | GgmlType::IQ3_S)
                && moe_iq3_expert_native_enabled(desc)
            {
                Ok(MetalTensor::from_gguf_tensor(ctx, desc, gguf.slice(desc))?)
            } else {
                load_weight(desc)
            }
        };
        let load_moe = |moe: &MoeFfn<'_>| -> Result<MetalMoeFfn, MfError> {
            Ok(MetalMoeFfn {
                gate_inp: load_f32(moe.gate_inp)?,
                gate_exps: load_moe_expert(moe.gate_exps)?,
                up_exps: load_moe_expert(moe.up_exps)?,
                down_exps: load_weight(moe.down_exps)?,
                gate_inp_shexp: load_f32(moe.gate_inp_shexp)?,
                gate_inp_cpu: crate::codec::dequant_to_f32(moe.gate_inp, gguf.slice(moe.gate_inp))?,
                gate_inp_shexp_cpu: crate::codec::dequant_to_f32(
                    moe.gate_inp_shexp,
                    gguf.slice(moe.gate_inp_shexp),
                )?,
            })
        };

        let load_attn_qkv_fused = |q: &MetalTensor,
                                   k: &MetalTensor,
                                   v: &MetalTensor|
         -> Result<Option<MetalTensor>, MfError> {
            let enabled = matches!(
                std::env::var("QWEN_PREFILL_ATTN_FUSED_QKV_G8").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            );
            if !enabled {
                return Ok(None);
            }
            if !(q.dtype == k.dtype
                && q.dtype == v.dtype
                && q.dtype == GgmlType::Q8_0
                && q.shape.len() == 2
                && k.shape.len() == 2
                && v.shape.len() == 2
                && q.shape[0] == k.shape[0]
                && q.shape[0] == v.shape[0])
            {
                return Ok(None);
            }
            let read_bytes = |t: &MetalTensor| -> Vec<u8> {
                let n = t.n_bytes() as usize;
                let mut out = vec![0u8; n];
                unsafe {
                    let src = (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize);
                    std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
                }
                out
            };
            let qb = read_bytes(q);
            let kb = read_bytes(k);
            let vb = read_bytes(v);
            let mut bytes = Vec::with_capacity(qb.len() + kb.len() + vb.len());
            bytes.extend_from_slice(&qb);
            bytes.extend_from_slice(&kb);
            bytes.extend_from_slice(&vb);
            let out_dim = checked_u64_add(
                checked_u64_add(q.shape[1], k.shape[1], "attn q+k out dim overflow")?,
                v.shape[1],
                "attn qkv fused out dim overflow",
            )?;
            Ok(Some(MetalTensor::from_bytes(
                ctx,
                &bytes,
                vec![q.shape[0], out_dim],
                q.dtype,
            )?))
        };

        let mut blocks = Vec::with_capacity(model.blocks.len());
        for b in &model.blocks {
            match b {
                Block::Gdn(g) => {
                    blocks.push(MetalBlock::Gdn(MetalGdnBlock {
                        attn_norm: load_f32(g.attn_norm)?,
                        post_attn_norm: load_f32(g.post_attention_norm)?,
                        ffn_gate: load_weight(g.ffn_gate)?,
                        ffn_up: load_weight(g.ffn_up)?,
                        ffn_down: load_weight(g.ffn_down)?,
                        in_proj_qkv: load_weight(g.in_proj_qkv)?,
                        in_proj_z: load_weight(g.in_proj_z)?,
                        beta_proj: load_weight(g.beta_proj)?,
                        alpha_proj: load_weight(g.alpha_proj)?,
                        a_log: load_f32(g.a_log)?,
                        dt_bias: load_f32(g.dt_bias)?,
                        conv1d: load_f32(g.conv1d)?,
                        norm: load_f32(g.norm)?,
                        out_proj: load_weight(g.out_proj)?,
                        ffn_moe: g.ffn_moe.as_ref().map(&load_moe).transpose()?,
                    }));
                }
                Block::Attn(a) => {
                    let q = load_weight(a.q)?;
                    let k = load_weight(a.k)?;
                    let v = load_weight(a.v)?;
                    blocks.push(MetalBlock::Attn(MetalAttnBlock {
                        attn_norm: load_f32(a.attn_norm)?,
                        post_attn_norm: load_f32(a.post_attention_norm)?,
                        ffn_gate: load_weight(a.ffn_gate)?,
                        ffn_up: load_weight(a.ffn_up)?,
                        ffn_down: load_weight(a.ffn_down)?,
                        qkv_fused: load_attn_qkv_fused(&q, &k, &v)?,
                        q,
                        k,
                        v,
                        o: load_weight(a.o)?,
                        q_norm: load_f32(a.q_norm)?,
                        k_norm: load_f32(a.k_norm)?,
                        ffn_moe: a.ffn_moe.as_ref().map(&load_moe).transpose()?,
                    }));
                }
            }
        }
        let _ = load_tensor; // suppress unused-warning if all sites switched

        Ok(Self {
            arch: model.arch,
            tied_embeddings: model.tied_embeddings,
            token_embd,
            output_norm,
            lm_head,
            blocks,
        })
    }
}

/// Per-sequence state: GDN conv buffers + SSM states (one set per GDN
/// layer), KV cache (one set per attn layer), and a small pool of
/// scratch activation tensors that are reused across layers.
pub struct MetalSession {
    /// (kernel-1) * conv_dim per GDN layer, F32, contiguous.
    pub gdn_conv: Vec<MetalTensor>,
    /// n_v_heads * head_dim * head_dim per GDN layer, F32.
    pub gdn_state: Vec<MetalTensor>,

    /// `[capacity_tokens, n_kv_heads, head_dim]` per attn layer, F32.
    pub kv_k: Vec<MetalTensor>,
    pub kv_v: Vec<MetalTensor>,
    pub kv_n_pos: Vec<usize>,
    pub kv_capacity: usize,

    // Scratch arena — F32 buffers reused across layers within a single
    // forward step. Sized for the largest transient at each role.
    pub x: MetalTensor,            // hidden_size — residual stream
    pub h: MetalTensor,            // hidden_size — post-norm activation
    pub ffn_gate: MetalTensor,     // intermediate_size — FFN gate output
    pub ffn_up: MetalTensor,       // intermediate_size — FFN up output
    pub ffn_inner: MetalTensor,    // intermediate_size — silu(gate)*up
    pub ffn_out: MetalTensor,      // hidden_size — FFN final
    pub gdn_qkv: MetalTensor,      // conv_dim
    pub gdn_qkv_conv: MetalTensor, // conv_dim — post-conv
    pub gdn_z: MetalTensor,        // n_v * head_dim
    pub gdn_b: MetalTensor,        // n_v   (β source, pre-sigmoid)
    pub gdn_beta: MetalTensor,     // n_v   (post-sigmoid)
    pub gdn_a: MetalTensor,        // n_v   (α source, pre-softplus)
    pub gdn_alpha: MetalTensor,    // n_v   (per-head decay exp(g))
    // gdn_q/k/v removed in v0.31: q/k/v are now zero-copy views into
    // gdn_qkv_conv via MetalTensor::view_subrange; no scratch buffers needed.
    pub gdn_q_norm: MetalTensor, // n_k * head_dim — l2-normed
    pub gdn_k_norm: MetalTensor, // n_k * head_dim — l2-normed
    pub gdn_out: MetalTensor,    // n_v * head_dim — recurrence output
    pub gdn_normed: MetalTensor, // n_v * head_dim — RMSNormGated output
    pub gdn_proj: MetalTensor,   // hidden_size — out_proj output
    pub mixer_out: MetalTensor,  // hidden_size — mixer output (GDN or attn)

    // Attention scratch.
    pub attn_q_full: MetalTensor,   // 2 * q_dim — Q + gate interleaved
    pub attn_q: MetalTensor,        // q_dim — Q only
    pub attn_gate: MetalTensor,     // q_dim — sigmoid'd gate
    pub attn_q_normed: MetalTensor, // q_dim
    pub attn_k_now: MetalTensor,    // kv_dim — current step K
    pub attn_v_now: MetalTensor,    // kv_dim — current step V
    pub attn_k_normed: MetalTensor, // kv_dim
    pub attn_scores: MetalTensor,   // capacity_tokens — scores for current step
    pub attn_o: MetalTensor,        // q_dim — attention output
    // v4 flash-attn split-K partials. Sized for ATTN_V4_MAX_NWG; the
    // dispatcher passes the chosen NWG ≤ this value.
    pub attn_v4_o_partial: MetalTensor, // n_kv * NWG_max * GROUP * head_dim
    pub attn_v4_ml_partial: MetalTensor, // n_kv * NWG_max * GROUP * 2

    pub logits: MetalTensor,           // vocab_size
    pub argmax_tok: MetalTensor,       // [1] i32 in F32 buffer
    pub moe_router_probs: MetalTensor, // [n_expert] F32 (or [1] on dense models)
    pub moe_topk_idx: MetalTensor,     // [top_k] i32 in F32 buffer (or [1] on dense models)
    pub moe_topk_weight: MetalTensor,  // [top_k] F32 (or [1] on dense models)
    pub moe_shared_gate: MetalTensor,  // [1] F32
    pub moe_inner: MetalTensor,        // [top_k, expert_ffn] F32 (or [1] on dense models)
    pub moe_expert_out: MetalTensor,   // [top_k, hidden] F32 (or [1] on dense models)
    pub ids_buf: MetalTensor, // 1-element scratch for the input token id (i32 in an F32 buf)
}

impl MetalSession {
    pub fn fresh(
        ctx: &MetalContext,
        model: &MetalModel,
        kv_capacity: usize,
    ) -> Result<Self, MetalError> {
        let arch = &model.arch;
        let h = arch.hidden_size as u64;
        let f = if arch.kind == ArchKind::Moe {
            arch.expert_shared_feed_forward_length
                .max(arch.expert_count)
                .max(arch.expert_feed_forward_length) as u64
        } else {
            arch.intermediate_size as u64
        };
        let head_dim = arch.attn_head_dim as u64;
        let n_q = arch.n_q_heads as u64;
        let n_kv = arch.n_kv_heads as u64;
        let q_dim = checked_u64_mul(n_q, head_dim, "q_dim overflow")?;
        let kv_dim = checked_u64_mul(n_kv, head_dim, "kv_dim overflow")?;
        let q_group = checked_u64_div_exact(n_q, n_kv, "n_q_heads / n_kv_heads invalid")?;
        let moe_router_n = if arch.kind == ArchKind::Moe {
            arch.expert_count.max(1) as u64
        } else {
            1
        };
        let moe_topk_n = if arch.kind == ArchKind::Moe {
            arch.expert_used_count.max(1).min(arch.expert_count.max(1)) as u64
        } else {
            1
        };
        let moe_inner_n = if arch.kind == ArchKind::Moe {
            checked_u64_mul(
                moe_topk_n,
                arch.expert_feed_forward_length.max(1) as u64,
                "moe_inner_n overflow",
            )?
        } else {
            1
        };
        let moe_expert_out_n = if arch.kind == ArchKind::Moe {
            checked_u64_mul(moe_topk_n, h, "moe_expert_out_n overflow")?
        } else {
            1
        };
        let vh = arch.gdn_head_dim as u64;
        let n_v = arch.gdn_n_v_heads as u64;
        let n_k = arch.gdn_n_k_heads as u64;
        let conv_heads = checked_u64_add(
            checked_u64_double(n_k, "2 * gdn_n_k_heads overflow")?,
            n_v,
            "2 * gdn_n_k_heads + gdn_n_v_heads overflow",
        )?;
        let conv_dim = checked_u64_mul(conv_heads, vh, "conv_dim overflow")?;
        let conv_kernel = arch.gdn_conv_kernel as u64;
        let v_dim = checked_u64_mul(n_v, vh, "v_dim overflow")?;
        let k_dim = checked_u64_mul(n_k, vh, "k_dim overflow")?;
        let kv_shape = vec![checked_u64_mul(
            kv_capacity as u64,
            kv_dim,
            "kv cache size overflow",
        )?];
        let gdn_conv_elems = checked_u64_mul(
            conv_kernel.saturating_sub(1),
            conv_dim,
            "gdn conv scratch size overflow",
        )?;
        let gdn_state_elems = checked_u64_mul3(n_v, vh, vh, "gdn state size overflow")?;
        let attn_q_full_elems = checked_u64_double(q_dim, "2 * q_dim overflow")?;
        let attn_v4_o_partial_elems = checked_u64_mul4(
            n_kv,
            ATTN_V4_MAX_NWG as u64,
            q_group,
            head_dim,
            "attn_v4_o_partial size overflow",
        )?;
        let attn_v4_ml_partial_elems = checked_u64_mul4(
            n_kv,
            ATTN_V4_MAX_NWG as u64,
            q_group,
            2,
            "attn_v4_ml_partial size overflow",
        )?;

        let mut gdn_conv = Vec::new();
        let mut gdn_state = Vec::new();
        for b in &model.blocks {
            if matches!(b, MetalBlock::Gdn(_)) {
                gdn_conv.push(MetalTensor::zeros_f32(ctx, vec![gdn_conv_elems])?);
                gdn_state.push(MetalTensor::zeros_f32(ctx, vec![gdn_state_elems])?);
            }
        }

        // KV cache defaults to F16 storage (matches llama.cpp's default
        // --cache-type-k f16). Experimental dense long-context path can
        // switch to Q8_0 via `QWEN_KV_Q8=1`; this is currently limited to the
        // dense group6 / head_dim=256 shape.
        let kv_dtype = kv_cache_dtype_for_arch(arch);
        let mut kv_k = Vec::new();
        let mut kv_v = Vec::new();
        let mut kv_n_pos = Vec::new();
        for b in &model.blocks {
            if matches!(b, MetalBlock::Attn(_)) {
                let shape = kv_shape.clone();
                kv_k.push(match kv_dtype {
                    GgmlType::F16 => MetalTensor::zeros_f16(ctx, shape.clone())?,
                    GgmlType::Q8_0 => MetalTensor::zeros_q8_0(ctx, shape.clone())?,
                    _ => unreachable!(),
                });
                kv_v.push(match kv_dtype {
                    GgmlType::F16 => MetalTensor::zeros_f16(ctx, shape.clone())?,
                    GgmlType::Q8_0 => MetalTensor::zeros_q8_0(ctx, shape.clone())?,
                    _ => unreachable!(),
                });
                kv_n_pos.push(0);
            }
        }

        Ok(Self {
            gdn_conv,
            gdn_state,
            kv_k,
            kv_v,
            kv_n_pos,
            kv_capacity,
            x: MetalTensor::zeros_f32(ctx, vec![h])?,
            h: MetalTensor::zeros_f32(ctx, vec![h])?,
            ffn_gate: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_up: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_inner: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_out: MetalTensor::zeros_f32(ctx, vec![h])?,
            gdn_qkv: MetalTensor::zeros_f32(ctx, vec![conv_dim])?,
            gdn_qkv_conv: MetalTensor::zeros_f32(ctx, vec![conv_dim])?,
            gdn_z: MetalTensor::zeros_f32(ctx, vec![v_dim])?,
            gdn_b: MetalTensor::zeros_f32(ctx, vec![n_v])?,
            gdn_beta: MetalTensor::zeros_f32(ctx, vec![n_v])?,
            gdn_a: MetalTensor::zeros_f32(ctx, vec![n_v])?,
            gdn_alpha: MetalTensor::zeros_f32(ctx, vec![n_v])?,
            gdn_q_norm: MetalTensor::zeros_f32(ctx, vec![k_dim])?,
            gdn_k_norm: MetalTensor::zeros_f32(ctx, vec![k_dim])?,
            gdn_out: MetalTensor::zeros_f32(ctx, vec![v_dim])?,
            gdn_normed: MetalTensor::zeros_f32(ctx, vec![v_dim])?,
            gdn_proj: MetalTensor::zeros_f32(ctx, vec![h])?,
            mixer_out: MetalTensor::zeros_f32(ctx, vec![h])?,
            attn_q_full: MetalTensor::zeros_f32(ctx, vec![attn_q_full_elems])?,
            attn_q: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_gate: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_q_normed: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_k_now: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_v_now: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_k_normed: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_scores: MetalTensor::zeros_f32(ctx, vec![kv_capacity as u64])?,
            attn_o: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            // v4 partials: n_kv * NWG_max * GROUP * head_dim (and *2 for ml).
            attn_v4_o_partial: MetalTensor::zeros_f32(ctx, vec![attn_v4_o_partial_elems])?,
            attn_v4_ml_partial: MetalTensor::zeros_f32(ctx, vec![attn_v4_ml_partial_elems])?,
            logits: MetalTensor::zeros_f32(ctx, vec![arch.vocab_size as u64])?,
            argmax_tok: MetalTensor::zeros_f32(ctx, vec![1])?,
            moe_router_probs: MetalTensor::zeros_f32(ctx, vec![moe_router_n])?,
            moe_topk_idx: MetalTensor::zeros_f32(ctx, vec![moe_topk_n])?,
            moe_topk_weight: MetalTensor::zeros_f32(ctx, vec![moe_topk_n])?,
            moe_shared_gate: MetalTensor::zeros_f32(ctx, vec![1])?,
            moe_inner: MetalTensor::zeros_f32(ctx, vec![moe_inner_n])?,
            moe_expert_out: MetalTensor::zeros_f32(ctx, vec![moe_expert_out_n])?,
            ids_buf: MetalTensor::zeros_f32(ctx, vec![1])?,
        })
    }
}

/// End-to-end Metal forward driver.
pub struct MetalForward<'a> {
    pub ctx: &'a MetalContext,
    pub model: &'a MetalModel,
}

impl<'a> MetalForward<'a> {
    pub fn new(ctx: &'a MetalContext, model: &'a MetalModel) -> Self {
        Self { ctx, model }
    }

    #[allow(dead_code)]
    fn route_moe_block(&self, h_tensor: &MetalTensor, moe: &MetalMoeFfn) -> MoeRouteDecision {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let n_expert = arch.expert_count as usize;
        let n_expert_used = arch.expert_used_count.min(arch.expert_count) as usize;
        let h_cpu = unsafe {
            std::slice::from_raw_parts(
                (h_tensor.buffer.contents().as_ptr() as *const f32)
                    .add((h_tensor.offset / 4) as usize),
                h,
            )
        };
        let mut probs = crate::forward::mat_vec_pub(&moe.gate_inp_cpu, h, n_expert, h_cpu);
        let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in &mut probs {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum.max(1e-20);
        for v in &mut probs {
            *v *= inv;
        }
        let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(n_expert_used);

        let shared_gate_scalar = {
            let s: f32 = h_cpu[..h]
                .iter()
                .zip(&moe.gate_inp_shexp_cpu[..h])
                .map(|(a, b)| a * b)
                .sum();
            1.0 / (1.0 + (-s).exp())
        };
        MoeRouteDecision {
            ranked,
            shared_gate_scalar,
        }
    }

    #[allow(dead_code)]
    fn read_moe_route_result(&self, session: &MetalSession, topk: usize) -> MoeRouteDecision {
        let mut ranked = Vec::with_capacity(topk);
        unsafe {
            let idx_ptr = (session.moe_topk_idx.buffer.contents().as_ptr() as *const i32)
                .add((session.moe_topk_idx.offset / 4) as usize);
            let w_ptr = (session.moe_topk_weight.buffer.contents().as_ptr() as *const f32)
                .add((session.moe_topk_weight.offset / 4) as usize);
            for i in 0..topk {
                ranked.push((*idx_ptr.add(i) as usize, *w_ptr.add(i)));
            }
            let shared_gate_scalar = *((session.moe_shared_gate.buffer.contents().as_ptr()
                as *const f32)
                .add((session.moe_shared_gate.offset / 4) as usize));
            MoeRouteDecision {
                ranked,
                shared_gate_scalar,
            }
        }
    }

    pub(crate) fn encode_moe_route_prepare(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        self.encode_moe_router_logits(enc, session, moe)?;
        self.encode_moe_topk_and_shared_from_logits(enc, session, moe)?;
        Ok(())
    }

    fn encode_moe_router_logits(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let n_expert = arch.expert_count as usize;

        let router_probs = session
            .moe_router_probs
            .view_subrange(0, vec![n_expert as u64]);

        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            &moe.gate_inp,
            &session.h,
            &router_probs,
            h,
            n_expert,
        )?;
        Ok(())
    }

    fn encode_moe_topk_from_logits(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;

        let router_probs = session
            .moe_router_probs
            .view_subrange(0, vec![n_expert as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        encode_topk_logits_softmax_f32(
            self.ctx,
            enc,
            &router_probs,
            &topk_idx,
            &topk_w,
            n_expert,
            topk,
        )?;
        Ok(())
    }

    fn encode_moe_topk_and_shared_from_logits(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;

        if n_expert > 256 || topk > 16 {
            self.encode_moe_topk_from_logits(enc, session)?;
            self.encode_moe_shared_gate(enc, session, moe)?;
            return Ok(());
        }

        let router_probs = session
            .moe_router_probs
            .view_subrange(0, vec![n_expert as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        encode_topk_logits_softmax_dot_sigmoid_f32(
            self.ctx,
            enc,
            &router_probs,
            &moe.gate_inp_shexp,
            &session.h,
            &topk_idx,
            &topk_w,
            &session.moe_shared_gate,
            n_expert,
            topk,
            h,
        )?;
        Ok(())
    }

    fn encode_moe_shared_gate(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;

        encode_dot_sigmoid_f32(
            self.ctx,
            enc,
            &moe.gate_inp_shexp,
            &session.h,
            &session.moe_shared_gate,
            h,
        )?;
        Ok(())
    }

    #[allow(dead_code)]
    fn encode_moe_ffn_apply(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        ffn_down: &MetalTensor,
        moe: &MetalMoeFfn,
        route: &MoeRouteDecision,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let f_shared = arch.expert_shared_feed_forward_length as usize;
        let n_expert = arch.expert_count as usize;

        // 2^-14 (FP16 minimum normal): matches forward.rs MoE normalization floor.
        let weight_sum = route
            .ranked
            .iter()
            .map(|(_, p)| *p)
            .sum::<f32>()
            .max(f32::from_bits(0x3880_0000));

        unsafe {
            let dst = (session.mixer_out.buffer.contents().as_ptr() as *mut u8)
                .add(session.mixer_out.offset as usize);
            std::ptr::write_bytes(dst, 0, h * std::mem::size_of::<f32>());
        }

        let gate_tmp = session.ffn_gate.view_subrange(0, vec![f_exp as u64]);
        let up_tmp = session.ffn_up.view_subrange(0, vec![f_exp as u64]);
        let inner_tmp = session.ffn_inner.view_subrange(0, vec![f_exp as u64]);
        let out_tmp = session.ffn_out.view_subrange(0, vec![h as u64]);
        let shared_gate_tmp = session.ffn_gate.view_subrange(0, vec![f_shared as u64]);
        let shared_up_tmp = session.ffn_up.view_subrange(0, vec![f_shared as u64]);
        let shared_inner_tmp = session.ffn_inner.view_subrange(0, vec![f_shared as u64]);

        let per_gate_bytes = moe.gate_exps.n_bytes() / n_expert as u64;
        let per_up_bytes = moe.up_exps.n_bytes() / n_expert as u64;
        let per_down_bytes = moe.down_exps.n_bytes() / n_expert as u64;
        let routed_fused =
            moe.gate_exps.dtype == GgmlType::Q4_K && moe.up_exps.dtype == GgmlType::Q4_K;
        for (expert_idx, prob) in route.ranked.iter().copied() {
            let weight = prob / weight_sum;
            let gate_w = moe.gate_exps.view_bytes(
                per_gate_bytes * expert_idx as u64,
                vec![h as u64, f_exp as u64],
            );
            let up_w = moe.up_exps.view_bytes(
                per_up_bytes * expert_idx as u64,
                vec![h as u64, f_exp as u64],
            );
            let down_w = moe.down_exps.view_bytes(
                per_down_bytes * expert_idx as u64,
                vec![f_exp as u64, h as u64],
            );
            if routed_fused {
                encode_ffn_swiglu_q4_K_f32(
                    self.ctx, enc, &gate_w, &up_w, &session.h, &inner_tmp, h, f_exp,
                )?;
            } else {
                encode_mat_vec_dispatch(self.ctx, enc, &gate_w, &session.h, &gate_tmp, h, f_exp)?;
                encode_mat_vec_dispatch(self.ctx, enc, &up_w, &session.h, &up_tmp, h, f_exp)?;
                encode_silu_mul_f32(self.ctx, enc, &gate_tmp, &up_tmp, &inner_tmp)?;
            }
            encode_mat_vec_dispatch(self.ctx, enc, &down_w, &inner_tmp, &out_tmp, f_exp, h)?;
            encode_axpy_f32(self.ctx, enc, &out_tmp, &session.mixer_out, weight)?;
        }

        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            ffn_gate,
            &session.h,
            &shared_gate_tmp,
            h,
            f_shared,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            ffn_up,
            &session.h,
            &shared_up_tmp,
            h,
            f_shared,
        )?;
        encode_silu_mul_f32(
            self.ctx,
            enc,
            &shared_gate_tmp,
            &shared_up_tmp,
            &shared_inner_tmp,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            ffn_down,
            &shared_inner_tmp,
            &out_tmp,
            f_shared,
            h,
        )?;
        encode_axpy_f32(
            self.ctx,
            enc,
            &out_tmp,
            &session.mixer_out,
            route.shared_gate_scalar,
        )?;
        encode_add_inplace_f32(self.ctx, enc, &session.x, &session.mixer_out)?;
        Ok(())
    }

    pub(crate) fn encode_moe_routed_ffn_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;

        let gate_up_supported = matches!(
            (moe.gate_exps.dtype, moe.up_exps.dtype),
            (GgmlType::Q4_K, GgmlType::Q4_K)
                | (GgmlType::Q5_K, GgmlType::Q5_K)
                | (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS)
                | (GgmlType::IQ3_S, GgmlType::IQ3_S)
                | (GgmlType::BF16, GgmlType::BF16)
                | (GgmlType::F32, GgmlType::F32)
        );
        if !gate_up_supported {
            return Err(MfError::UnsupportedDtype {
                name: "MoE routed gate/up expert banks".into(),
                dtype: moe.gate_exps.dtype,
            });
        }
        if !matches!(
            moe.down_exps.dtype,
            GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::IQ4_XS | GgmlType::BF16
        ) {
            return Err(MfError::UnsupportedDtype {
                name: "MoE routed down expert bank".into(),
                dtype: moe.down_exps.dtype,
            });
        }

        let moe_inner = session
            .moe_inner
            .view_subrange(0, vec![(topk * f_exp) as u64]);
        let moe_expert_out = session
            .moe_expert_out
            .view_subrange(0, vec![(topk * h) as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        match moe.gate_exps.dtype {
            GgmlType::Q4_K => encode_moe_swiglu_q4_K_f32(
                self.ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &session.h,
                &topk_idx,
                &moe_inner,
                h,
                f_exp,
                n_expert,
                topk,
            )?,
            GgmlType::Q5_K => {
                let gate_pack = session
                    .moe_expert_out
                    .view_subrange(0, vec![(topk * f_exp) as u64]);
                let up_pack = session
                    .moe_expert_out
                    .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                encode_moe_mat_vec_q5_K_f32(
                    self.ctx,
                    enc,
                    &moe.gate_exps,
                    &session.h,
                    &topk_idx,
                    &gate_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_moe_mat_vec_q5_K_f32(
                    self.ctx,
                    enc,
                    &moe.up_exps,
                    &session.h,
                    &topk_idx,
                    &up_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_silu_mul_f32(self.ctx, enc, &gate_pack, &up_pack, &moe_inner)?;
            }
            GgmlType::IQ3_XXS => {
                let gate_pack = session
                    .moe_expert_out
                    .view_subrange(0, vec![(topk * f_exp) as u64]);
                let up_pack = session
                    .moe_expert_out
                    .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                encode_moe_mat_vec_iq3_xxs_f32(
                    self.ctx,
                    enc,
                    &moe.gate_exps,
                    &session.h,
                    &topk_idx,
                    &gate_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_moe_mat_vec_iq3_xxs_f32(
                    self.ctx,
                    enc,
                    &moe.up_exps,
                    &session.h,
                    &topk_idx,
                    &up_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_silu_mul_f32(self.ctx, enc, &gate_pack, &up_pack, &moe_inner)?;
            }
            GgmlType::IQ3_S => {
                let gate_pack = session
                    .moe_expert_out
                    .view_subrange(0, vec![(topk * f_exp) as u64]);
                let up_pack = session
                    .moe_expert_out
                    .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                encode_moe_mat_vec_iq3_s_f32(
                    self.ctx,
                    enc,
                    &moe.gate_exps,
                    &session.h,
                    &topk_idx,
                    &gate_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_moe_mat_vec_iq3_s_f32(
                    self.ctx,
                    enc,
                    &moe.up_exps,
                    &session.h,
                    &topk_idx,
                    &up_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_silu_mul_f32(self.ctx, enc, &gate_pack, &up_pack, &moe_inner)?;
            }
            GgmlType::F32 => {
                let gate_pack = session
                    .moe_expert_out
                    .view_subrange(0, vec![(topk * f_exp) as u64]);
                let up_pack = session
                    .moe_expert_out
                    .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                encode_moe_mat_vec_f32(
                    self.ctx,
                    enc,
                    &moe.gate_exps,
                    &session.h,
                    &topk_idx,
                    &gate_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_moe_mat_vec_f32(
                    self.ctx,
                    enc,
                    &moe.up_exps,
                    &session.h,
                    &topk_idx,
                    &up_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_silu_mul_f32(self.ctx, enc, &gate_pack, &up_pack, &moe_inner)?;
            }
            GgmlType::BF16 => {
                let gate_pack = session
                    .moe_expert_out
                    .view_subrange(0, vec![(topk * f_exp) as u64]);
                let up_pack = session
                    .moe_expert_out
                    .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                encode_moe_mat_vec_bf16_f32(
                    self.ctx,
                    enc,
                    &moe.gate_exps,
                    &session.h,
                    &topk_idx,
                    &gate_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_moe_mat_vec_bf16_f32(
                    self.ctx,
                    enc,
                    &moe.up_exps,
                    &session.h,
                    &topk_idx,
                    &up_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_silu_mul_f32(self.ctx, enc, &gate_pack, &up_pack, &moe_inner)?;
            }
            _ => unreachable!(),
        }
        match moe.down_exps.dtype {
            GgmlType::Q5_K => {
                if decode_moe_q5_down_fused_enabled() {
                    encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                        self.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &topk_w,
                        &session.mixer_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                        1,
                    )?;
                } else {
                    encode_moe_down_q5_K_f32(
                        self.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )?;
                    encode_moe_weighted_sum_f32(
                        self.ctx,
                        enc,
                        &moe_expert_out,
                        &topk_w,
                        &session.mixer_out,
                        h,
                        topk,
                    )?;
                }
            }
            GgmlType::Q6_K => encode_moe_down_weighted_sum_q6_K_f32(
                self.ctx,
                enc,
                &moe.down_exps,
                &moe_inner,
                &topk_idx,
                &topk_w,
                &session.mixer_out,
                f_exp,
                h,
                n_expert,
                topk,
            )?,
            GgmlType::IQ4_XS => {
                encode_moe_down_iq4_xs_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                encode_moe_weighted_sum_f32(
                    self.ctx,
                    enc,
                    &moe_expert_out,
                    &topk_w,
                    &session.mixer_out,
                    h,
                    topk,
                )?;
            }
            GgmlType::BF16 => {
                encode_moe_down_bf16_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                encode_moe_weighted_sum_f32(
                    self.ctx,
                    enc,
                    &moe_expert_out,
                    &topk_w,
                    &session.mixer_out,
                    h,
                    topk,
                )?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    pub(crate) fn encode_moe_shared_ffn_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        ffn_down: &MetalTensor,
    ) -> Result<(), MfError> {
        self.encode_moe_shared_ffn_core_gpu(enc, session, ffn_gate, ffn_up, ffn_down)?;
        self.encode_moe_shared_ffn_accumulate_gpu(enc, session)
    }

    fn encode_moe_shared_ffn_core_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        ffn_down: &MetalTensor,
    ) -> Result<(), MfError> {
        let fused_inner = self.encode_moe_shared_ffn_gate_up_gpu(enc, session, ffn_gate, ffn_up)?;
        if !fused_inner {
            self.encode_moe_shared_ffn_silu_gpu(enc, session)?;
        }
        self.encode_moe_shared_ffn_down_gpu(enc, session, ffn_down)
    }

    fn encode_moe_shared_ffn_gate_up_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
    ) -> Result<bool, MfError> {
        let h = self.model.arch.hidden_size as usize;
        let f_shared = self.model.arch.expert_shared_feed_forward_length as usize;
        if decode_shared_swiglu_q8_enabled()
            && ffn_gate.dtype == GgmlType::Q8_0
            && ffn_up.dtype == GgmlType::Q8_0
        {
            let shared_inner_tmp = session.ffn_inner.view_subrange(0, vec![f_shared as u64]);
            encode_shared_swiglu_q8_0_f32(
                self.ctx,
                enc,
                ffn_gate,
                ffn_up,
                &session.h,
                &shared_inner_tmp,
                h,
                f_shared,
            )?;
            return Ok(true);
        }
        let shared_gate_tmp = session.ffn_gate.view_subrange(0, vec![f_shared as u64]);
        let shared_up_tmp = session.ffn_up.view_subrange(0, vec![f_shared as u64]);
        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            ffn_gate,
            &session.h,
            &shared_gate_tmp,
            h,
            f_shared,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            ffn_up,
            &session.h,
            &shared_up_tmp,
            h,
            f_shared,
        )?;
        Ok(false)
    }

    fn encode_moe_shared_ffn_silu_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let f_shared = self.model.arch.expert_shared_feed_forward_length as usize;
        let shared_gate_tmp = session.ffn_gate.view_subrange(0, vec![f_shared as u64]);
        let shared_up_tmp = session.ffn_up.view_subrange(0, vec![f_shared as u64]);
        let shared_inner_tmp = session.ffn_inner.view_subrange(0, vec![f_shared as u64]);
        encode_silu_mul_f32(
            self.ctx,
            enc,
            &shared_gate_tmp,
            &shared_up_tmp,
            &shared_inner_tmp,
        )?;
        Ok(())
    }

    fn encode_moe_shared_ffn_down_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        ffn_down: &MetalTensor,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        let f_shared = self.model.arch.expert_shared_feed_forward_length as usize;
        let shared_inner_tmp = session.ffn_inner.view_subrange(0, vec![f_shared as u64]);
        let shared_out_tmp = session.ffn_out.view_subrange(0, vec![h as u64]);
        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            ffn_down,
            &shared_inner_tmp,
            &shared_out_tmp,
            f_shared,
            h,
        )?;
        Ok(())
    }

    fn encode_moe_shared_ffn_accumulate_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        let shared_out_tmp = session.ffn_out.view_subrange(0, vec![h as u64]);
        encode_axpy_scalar_f32(
            self.ctx,
            enc,
            &shared_out_tmp,
            &session.moe_shared_gate,
            &session.mixer_out,
        )?;
        Ok(())
    }

    fn encode_moe_final_residual_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        let shared_out_tmp = session.ffn_out.view_subrange(0, vec![h as u64]);
        if decode_moe_fused_finalizer_enabled() {
            encode_moe_shared_accum_resid_f32(
                self.ctx,
                enc,
                &shared_out_tmp,
                &session.moe_shared_gate,
                &session.mixer_out,
                &session.x,
            )?;
        } else {
            self.encode_moe_shared_ffn_accumulate_gpu(enc, session)?;
            encode_add_inplace_f32(self.ctx, enc, &session.x, &session.mixer_out)?;
        }
        Ok(())
    }

    pub(crate) fn encode_moe_ffn_apply_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        ffn_down: &MetalTensor,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        self.encode_moe_routed_ffn_gpu(enc, session, moe)?;
        self.encode_moe_shared_ffn_core_gpu(enc, session, ffn_gate, ffn_up, ffn_down)?;
        self.encode_moe_final_residual_gpu(enc, session)?;
        Ok(())
    }

    fn encode_moe_routed_gate_up_q4_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let moe_inner = session
            .moe_inner
            .view_subrange(0, vec![(topk * f_exp) as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);

        if decode_moe_noop_routed_gateup_enabled() {
            encode_fill_f32(self.ctx, enc, &moe_inner, 0.0)?;
        } else {
            encode_moe_swiglu_q4_K_f32(
                self.ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &session.h,
                &topk_idx,
                &moe_inner,
                h,
                f_exp,
                n_expert,
                topk,
            )?;
        }
        Ok(())
    }

    fn encode_moe_routed_down_only_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<bool, MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let moe_inner = session
            .moe_inner
            .view_subrange(0, vec![(topk * f_exp) as u64]);
        let moe_expert_out = session
            .moe_expert_out
            .view_subrange(0, vec![(topk * h) as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        if decode_moe_noop_routed_down_enabled() {
            encode_fill_f32(self.ctx, enc, &session.mixer_out, 0.0)?;
            return Ok(false);
        }

        let pending = match moe.down_exps.dtype {
            GgmlType::Q5_K => {
                if decode_moe_q5_down_fused_enabled() {
                    if f_exp == 512 && decode_moe_q5_down_k512_r2_enabled() {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                            self.ctx,
                            enc,
                            &moe.down_exps,
                            &moe_inner,
                            &topk_idx,
                            &topk_w,
                            &session.mixer_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            1,
                        )?;
                    } else {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                            self.ctx,
                            enc,
                            &moe.down_exps,
                            &moe_inner,
                            &topk_idx,
                            &topk_w,
                            &session.mixer_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            1,
                        )?;
                    }
                    false
                } else {
                    encode_moe_down_q5_K_f32(
                        self.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )?;
                    true
                }
            }
            GgmlType::Q6_K => {
                encode_moe_down_weighted_sum_q6_K_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &topk_w,
                    &session.mixer_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                false
            }
            GgmlType::IQ4_XS => {
                encode_moe_down_iq4_xs_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                true
            }
            GgmlType::BF16 => {
                encode_moe_down_bf16_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                true
            }
            _ => unreachable!(),
        };
        Ok(pending)
    }

    fn encode_moe_ffn_gate_up_wave_gpu(
        &self,
        cmd_buf: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        moe: &MetalMoeFfn,
    ) -> Result<bool, MfError> {
        let enc = KernelEncoder::begin_concurrent(cmd_buf);
        self.encode_moe_routed_gate_up_q4_gpu(&enc, session, moe)?;
        let shared_inner_fused =
            self.encode_moe_shared_ffn_gate_up_gpu(&enc, session, ffn_gate, ffn_up)?;
        enc.end();
        Ok(shared_inner_fused)
    }

    fn encode_moe_ffn_down_wave_gpu(
        &self,
        cmd_buf: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        session: &mut MetalSession,
        ffn_down: &MetalTensor,
        moe: &MetalMoeFfn,
    ) -> Result<bool, MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let moe_inner = session
            .moe_inner
            .view_subrange(0, vec![(topk * f_exp) as u64]);
        let moe_expert_out = session
            .moe_expert_out
            .view_subrange(0, vec![(topk * h) as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        if decode_moe_noop_routed_down_enabled() {
            let enc = KernelEncoder::begin_concurrent(cmd_buf);
            encode_fill_f32(self.ctx, &enc, &session.mixer_out, 0.0)?;
            self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
            enc.end();
            return Ok(false);
        }

        let pending = match moe.down_exps.dtype {
            GgmlType::Q5_K => {
                let enc = KernelEncoder::begin_concurrent(cmd_buf);
                let pending = if decode_moe_q5_down_fused_enabled() {
                    if f_exp == 512 && decode_moe_q5_down_k512_r2_enabled() {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                            self.ctx,
                            &enc,
                            &moe.down_exps,
                            &moe_inner,
                            &topk_idx,
                            &topk_w,
                            &session.mixer_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            1,
                        )?;
                    } else {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                            self.ctx,
                            &enc,
                            &moe.down_exps,
                            &moe_inner,
                            &topk_idx,
                            &topk_w,
                            &session.mixer_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            1,
                        )?;
                    }
                    false
                } else {
                    encode_moe_down_q5_K_f32(
                        self.ctx,
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
                    true
                };
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                pending
            }
            GgmlType::Q6_K => {
                let enc = KernelEncoder::begin_concurrent(cmd_buf);
                encode_moe_down_weighted_sum_q6_K_f32(
                    self.ctx,
                    &enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &topk_w,
                    &session.mixer_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                false
            }
            GgmlType::IQ4_XS => {
                let enc = KernelEncoder::begin_concurrent(cmd_buf);
                encode_moe_down_iq4_xs_f32(
                    self.ctx,
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
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                true
            }
            GgmlType::BF16 => {
                let enc = KernelEncoder::begin_concurrent(cmd_buf);
                encode_moe_down_bf16_f32(
                    self.ctx,
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
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                true
            }
            _ => unreachable!(),
        };
        Ok(pending)
    }

    fn encode_moe_ffn_final_wave_gpu(
        &self,
        cmd_buf: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        session: &mut MetalSession,
        routed_weighted_sum_is_pending: bool,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        let topk = self
            .model
            .arch
            .expert_used_count
            .min(self.model.arch.expert_count) as usize;
        let moe_expert_out = session
            .moe_expert_out
            .view_subrange(0, vec![(topk * h) as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        let enc = KernelEncoder::begin(cmd_buf);
        if routed_weighted_sum_is_pending {
            encode_moe_weighted_sum_f32(
                self.ctx,
                &enc,
                &moe_expert_out,
                &topk_w,
                &session.mixer_out,
                h,
                topk,
            )?;
        }
        self.encode_moe_final_residual_gpu(&enc, session)?;
        enc.end();
        Ok(())
    }

    fn encode_moe_ffn_apply_gpu_concurrent_shared(
        &self,
        cmd_buf: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        ffn_down: &MetalTensor,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        if moe.gate_exps.dtype != GgmlType::Q4_K || moe.up_exps.dtype != GgmlType::Q4_K {
            let enc = KernelEncoder::begin(cmd_buf);
            self.encode_moe_ffn_apply_gpu(&enc, session, ffn_gate, ffn_up, ffn_down, moe)?;
            enc.end();
            return Ok(());
        }

        let shared_inner_fused =
            self.encode_moe_ffn_gate_up_wave_gpu(cmd_buf, session, ffn_gate, ffn_up, moe)?;
        if !shared_inner_fused {
            let enc = KernelEncoder::begin(cmd_buf);
            self.encode_moe_shared_ffn_silu_gpu(&enc, session)?;
            enc.end();
        }
        let routed_weighted_sum_is_pending =
            self.encode_moe_ffn_down_wave_gpu(cmd_buf, session, ffn_down, moe)?;
        self.encode_moe_ffn_final_wave_gpu(cmd_buf, session, routed_weighted_sum_is_pending)?;
        Ok(())
    }

    fn encode_moe_block_gpu(
        &self,
        enc: &KernelEncoder,
        block: &MetalBlock,
        mixer_slot: MixerSlot,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        self.encode_moe_mixer_prep(enc, block, mixer_slot, position, session)?;
        let (ffn_gate, ffn_up, ffn_down, moe) = match block {
            MetalBlock::Gdn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
            MetalBlock::Attn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
        };
        let moe = moe.ok_or(MfError::UnsupportedMoe)?;
        self.encode_moe_route_prepare(enc, session, moe)?;
        self.encode_moe_ffn_apply_gpu(enc, session, ffn_gate, ffn_up, ffn_down, moe)
    }

    fn encode_moe_mixer_prep(
        &self,
        enc: &KernelEncoder,
        block: &MetalBlock,
        mixer_slot: MixerSlot,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let (attn_norm, post_norm) = match block {
            MetalBlock::Gdn(b) => (&b.attn_norm, &b.post_attn_norm),
            MetalBlock::Attn(b) => (&b.attn_norm, &b.post_attn_norm),
        };
        encode_rms_norm_mul_f32(self.ctx, enc, &session.x, attn_norm, &session.h, RMS_EPS)?;
        match (block, mixer_slot) {
            (MetalBlock::Gdn(g), MixerSlot::Gdn(idx)) => self.encode_gdn(enc, g, idx, session)?,
            (MetalBlock::Attn(a), MixerSlot::Attn(idx)) => {
                self.encode_attn(enc, a, idx, position, session)?
            }
            _ => {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "single_token_moe",
                    detail: "mixer slot type did not match block kind".into(),
                }));
            }
        }
        encode_add_inplace_f32(self.ctx, enc, &session.x, &session.mixer_out)?;
        encode_rms_norm_mul_f32(self.ctx, enc, &session.x, post_norm, &session.h, RMS_EPS)?;
        Ok(())
    }

    /// Run a single token through the model. Encodes all kernels into
    /// one command buffer, commits, waits, reads back logits.
    ///
    /// `position` is the 0-indexed sequence position (used by RoPE for
    /// the full-attn layers; ignored by GDN layers).
    pub fn single_token(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<Vec<f32>, MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return self.single_token_moe(token_id, position, session);
        }
        let (logits, _) = self.single_token_profiled(token_id, position, session)?;
        Ok(logits)
    }

    pub fn single_token_profiled_concurrent_gdn_dense(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Dense {
            return Err(MfError::UnsupportedMoe);
        }
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_get_rows_f32(
                self.ctx,
                &enc,
                &self.model.token_embd,
                &session.ids_buf,
                &session.x,
                1,
                h,
            )?;
            enc.end();
        }

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &self.model.blocks {
            match block {
                MetalBlock::Gdn(g) => {
                    let i = gdn_idx;
                    gdn_idx += 1;
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin_concurrent(&cmd_buf);
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        self.encode_gdn_after_projections(&enc, g, i, session)?;
                        self.encode_post_mixer_ffn(&enc, block, session)?;
                        enc.end();
                    }
                }
                MetalBlock::Attn(_) => {
                    let enc = KernelEncoder::begin(&cmd_buf);
                    self.encode_block(
                        &enc,
                        0,
                        block,
                        &mut gdn_idx,
                        &mut attn_idx,
                        position,
                        session,
                    )?;
                    enc.end();
                }
            }
        }

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                arch.hidden_size as usize,
                arch.vocab_size as usize,
            )?;
            enc.end();
        }

        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            out,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    pub fn single_token_argmax_profiled_concurrent_gdn_dense(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(i32, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Dense {
            return Err(MfError::UnsupportedMoe);
        }
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let ids_buf = session.ids_buf.clone();
        let argmax_tok = session.argmax_tok.clone();

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_get_rows_f32(
                self.ctx,
                &enc,
                &self.model.token_embd,
                &ids_buf,
                &session.x,
                1,
                h,
            )?;
            enc.end();
        }

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &self.model.blocks {
            match block {
                MetalBlock::Gdn(g) => {
                    let i = gdn_idx;
                    gdn_idx += 1;
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin_concurrent(&cmd_buf);
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        self.encode_gdn_after_projections(&enc, g, i, session)?;
                        self.encode_post_mixer_ffn(&enc, block, session)?;
                        enc.end();
                    }
                }
                MetalBlock::Attn(_) => {
                    let enc = KernelEncoder::begin(&cmd_buf);
                    self.encode_block(
                        &enc,
                        0,
                        block,
                        &mut gdn_idx,
                        &mut attn_idx,
                        position,
                        session,
                    )?;
                    enc.end();
                }
            }
        }

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                h,
                arch.vocab_size as usize,
            )?;
            encode_argmax_f32(
                self.ctx,
                &enc,
                &session.logits,
                &argmax_tok,
                1,
                arch.vocab_size as usize,
            )?;
            enc.end();
        }

        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let argmax = unsafe {
            let src = argmax_tok.buffer.contents().as_ptr() as *const i32;
            *src
        };
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            argmax,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    pub fn single_token_profiled_concurrent_gdn_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_get_rows_f32(
                self.ctx,
                &enc,
                &self.model.token_embd,
                &session.ids_buf,
                &session.x,
                1,
                h,
            )?;
            enc.end();
        }

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &self.model.blocks {
            match block {
                MetalBlock::Gdn(g) => {
                    let i = gdn_idx;
                    gdn_idx += 1;
                    let moe = g.ffn_moe.as_ref().ok_or(MfError::UnsupportedMoe)?;
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin_concurrent(&cmd_buf);
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        self.encode_gdn_after_projections(&enc, g, i, session)?;
                        encode_add_inplace_f32(self.ctx, &enc, &session.x, &session.mixer_out)?;
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.post_attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        self.encode_moe_route_prepare(&enc, session, moe)?;
                        if !concurrent_shared_moe_decode_enabled() {
                            self.encode_moe_ffn_apply_gpu(
                                &enc,
                                session,
                                &g.ffn_gate,
                                &g.ffn_up,
                                &g.ffn_down,
                                moe,
                            )?;
                        }
                        enc.end();
                    }
                    if concurrent_shared_moe_decode_enabled() {
                        self.encode_moe_ffn_apply_gpu_concurrent_shared(
                            &cmd_buf,
                            session,
                            &g.ffn_gate,
                            &g.ffn_up,
                            &g.ffn_down,
                            moe,
                        )?;
                    }
                }
                MetalBlock::Attn(a) => {
                    let slot = MixerSlot::Attn(attn_idx);
                    attn_idx += 1;
                    let enc = KernelEncoder::begin(&cmd_buf);
                    self.encode_moe_mixer_prep(&enc, block, slot, position, session)?;
                    let moe = a.ffn_moe.as_ref().ok_or(MfError::UnsupportedMoe)?;
                    self.encode_moe_route_prepare(&enc, session, moe)?;
                    if !concurrent_shared_moe_decode_enabled() {
                        self.encode_moe_ffn_apply_gpu(
                            &enc,
                            session,
                            &a.ffn_gate,
                            &a.ffn_up,
                            &a.ffn_down,
                            moe,
                        )?;
                    }
                    enc.end();
                    if concurrent_shared_moe_decode_enabled() {
                        self.encode_moe_ffn_apply_gpu_concurrent_shared(
                            &cmd_buf,
                            session,
                            &a.ffn_gate,
                            &a.ffn_up,
                            &a.ffn_down,
                            moe,
                        )?;
                    }
                }
            }
        }

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                h,
                arch.vocab_size as usize,
            )?;
            enc.end();
        }

        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            out,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    pub fn single_token_argmax_profiled_concurrent_gdn_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(i32, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let ids_buf = session.ids_buf.clone();
        let argmax_tok = session.argmax_tok.clone();

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_get_rows_f32(
                self.ctx,
                &enc,
                &self.model.token_embd,
                &ids_buf,
                &session.x,
                1,
                h,
            )?;
            enc.end();
        }

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &self.model.blocks {
            match block {
                MetalBlock::Gdn(g) => {
                    let i = gdn_idx;
                    gdn_idx += 1;
                    let moe = g.ffn_moe.as_ref().ok_or(MfError::UnsupportedMoe)?;
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin_concurrent(&cmd_buf);
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        self.encode_gdn_after_projections(&enc, g, i, session)?;
                        encode_add_inplace_f32(self.ctx, &enc, &session.x, &session.mixer_out)?;
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.post_attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        self.encode_moe_route_prepare(&enc, session, moe)?;
                        if !concurrent_shared_moe_decode_enabled() {
                            self.encode_moe_ffn_apply_gpu(
                                &enc,
                                session,
                                &g.ffn_gate,
                                &g.ffn_up,
                                &g.ffn_down,
                                moe,
                            )?;
                        }
                        enc.end();
                    }
                    if concurrent_shared_moe_decode_enabled() {
                        self.encode_moe_ffn_apply_gpu_concurrent_shared(
                            &cmd_buf,
                            session,
                            &g.ffn_gate,
                            &g.ffn_up,
                            &g.ffn_down,
                            moe,
                        )?;
                    }
                }
                MetalBlock::Attn(a) => {
                    let slot = MixerSlot::Attn(attn_idx);
                    attn_idx += 1;
                    let enc = KernelEncoder::begin(&cmd_buf);
                    self.encode_moe_mixer_prep(&enc, block, slot, position, session)?;
                    let moe = a.ffn_moe.as_ref().ok_or(MfError::UnsupportedMoe)?;
                    self.encode_moe_route_prepare(&enc, session, moe)?;
                    if !concurrent_shared_moe_decode_enabled() {
                        self.encode_moe_ffn_apply_gpu(
                            &enc,
                            session,
                            &a.ffn_gate,
                            &a.ffn_up,
                            &a.ffn_down,
                            moe,
                        )?;
                    }
                    enc.end();
                    if concurrent_shared_moe_decode_enabled() {
                        self.encode_moe_ffn_apply_gpu_concurrent_shared(
                            &cmd_buf,
                            session,
                            &a.ffn_gate,
                            &a.ffn_up,
                            &a.ffn_down,
                            moe,
                        )?;
                    }
                }
            }
        }

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                h,
                arch.vocab_size as usize,
            )?;
            encode_argmax_f32(
                self.ctx,
                &enc,
                &session.logits,
                &argmax_tok,
                1,
                arch.vocab_size as usize,
            )?;
            enc.end();
        }

        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let argmax = unsafe {
            let src = argmax_tok.buffer.contents().as_ptr() as *const i32;
            *src
        };
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            argmax,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    pub fn single_token_profiled_concurrent_attn_dense(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Dense {
            return Err(MfError::UnsupportedMoe);
        }
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_get_rows_f32(
                self.ctx,
                &enc,
                &self.model.token_embd,
                &session.ids_buf,
                &session.x,
                1,
                h,
            )?;
            enc.end();
        }

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &self.model.blocks {
            match block {
                MetalBlock::Gdn(_) => {
                    let enc = KernelEncoder::begin(&cmd_buf);
                    self.encode_block(
                        &enc,
                        0,
                        block,
                        &mut gdn_idx,
                        &mut attn_idx,
                        position,
                        session,
                    )?;
                    enc.end();
                }
                MetalBlock::Attn(a) => {
                    let i = attn_idx;
                    attn_idx += 1;
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &a.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin_concurrent(&cmd_buf);
                        self.encode_attn_front_projections(&enc, a, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        self.encode_attn_after_projections(&enc, a, i, position, session)?;
                        self.encode_post_mixer_ffn(&enc, block, session)?;
                        enc.end();
                    }
                }
            }
        }

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                arch.hidden_size as usize,
                arch.vocab_size as usize,
            )?;
            enc.end();
        }

        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            out,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    pub fn single_token_profiled_concurrent_gdn_attn_dense(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Dense {
            return Err(MfError::UnsupportedMoe);
        }
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_get_rows_f32(
                self.ctx,
                &enc,
                &self.model.token_embd,
                &session.ids_buf,
                &session.x,
                1,
                h,
            )?;
            enc.end();
        }

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &self.model.blocks {
            match block {
                MetalBlock::Gdn(g) => {
                    let i = gdn_idx;
                    gdn_idx += 1;
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin_concurrent(&cmd_buf);
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        self.encode_gdn_after_projections(&enc, g, i, session)?;
                        self.encode_post_mixer_ffn(&enc, block, session)?;
                        enc.end();
                    }
                }
                MetalBlock::Attn(a) => {
                    let i = attn_idx;
                    attn_idx += 1;
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &a.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin_concurrent(&cmd_buf);
                        self.encode_attn_front_projections(&enc, a, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        self.encode_attn_after_projections(&enc, a, i, position, session)?;
                        self.encode_post_mixer_ffn(&enc, block, session)?;
                        enc.end();
                    }
                }
            }
        }

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                arch.hidden_size as usize,
                arch.vocab_size as usize,
            )?;
            enc.end();
        }

        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            out,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    fn single_token_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<Vec<f32>, MfError> {
        let (logits, _) = if concurrent_gdn_moe_decode_enabled() {
            self.single_token_profiled_concurrent_gdn_moe(token_id, position, session)?
        } else {
            self.single_token_profiled_moe(token_id, position, session)?
        };
        Ok(logits)
    }

    pub fn single_token_argmax(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<i32, MfError> {
        let (argmax, _) = self.single_token_argmax_profiled(token_id, position, session)?;
        Ok(argmax)
    }

    fn single_token_argmax_profiled_dense_serial(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(i32, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }

        let t_total = std::time::Instant::now();
        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }
        let ids_buf = session.ids_buf.clone();
        let argmax_tok = session.argmax_tok.clone();

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        self.encode_single_token_argmax_dense(&enc, position, session, &ids_buf, &argmax_tok)?;

        enc.end();
        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;

        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let argmax = unsafe {
            let src = argmax_tok.buffer.contents().as_ptr() as *const i32;
            *src
        };
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            argmax,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    pub fn single_token_argmax_profiled(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(i32, TokenProfile), MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return if concurrent_gdn_moe_decode_enabled() {
                self.single_token_argmax_profiled_concurrent_gdn_moe(token_id, position, session)
            } else {
                self.single_token_argmax_profiled_moe(token_id, position, session)
            };
        }
        if concurrent_gdn_dense_decode_enabled() {
            return self
                .single_token_argmax_profiled_concurrent_gdn_dense(token_id, position, session);
        }
        self.single_token_argmax_profiled_dense_serial(token_id, position, session)
    }

    pub fn encode_single_token_argmax(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
    ) -> Result<(), MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return self
                .encode_single_token_argmax_moe(enc, position, session, ids_buf, argmax_tok);
        }
        self.encode_single_token_argmax_dense(enc, position, session, ids_buf, argmax_tok)
    }

    pub fn encode_single_token_argmax_dense(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;

        encode_get_rows_f32(
            self.ctx,
            enc,
            &self.model.token_embd,
            ids_buf,
            &session.x,
            1,
            arch.hidden_size as usize,
        )?;

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
        }

        encode_rms_norm_mul_f32(
            self.ctx,
            enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            arch.hidden_size as usize,
            arch.vocab_size as usize,
        )?;
        encode_argmax_f32(
            self.ctx,
            enc,
            &session.logits,
            argmax_tok,
            1,
            arch.vocab_size as usize,
        )?;
        Ok(())
    }

    fn encode_single_token_argmax_moe(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;

        encode_get_rows_f32(
            self.ctx,
            enc,
            &self.model.token_embd,
            ids_buf,
            &session.x,
            1,
            h,
        )?;

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &self.model.blocks {
            let slot = match block {
                MetalBlock::Gdn(_) => {
                    let s = MixerSlot::Gdn(gdn_idx);
                    gdn_idx += 1;
                    s
                }
                MetalBlock::Attn(_) => {
                    let s = MixerSlot::Attn(attn_idx);
                    attn_idx += 1;
                    s
                }
            };
            self.encode_moe_block_gpu(enc, block, slot, position, session)?;
        }

        encode_rms_norm_mul_f32(
            self.ctx,
            enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            h,
            arch.vocab_size as usize,
        )?;
        encode_argmax_f32(
            self.ctx,
            enc,
            &session.logits,
            argmax_tok,
            1,
            arch.vocab_size as usize,
        )?;
        Ok(())
    }

    fn single_token_profiled_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            h,
        )?;

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &self.model.blocks {
            let slot = match block {
                MetalBlock::Gdn(_) => {
                    let s = MixerSlot::Gdn(gdn_idx);
                    gdn_idx += 1;
                    s
                }
                MetalBlock::Attn(_) => {
                    let s = MixerSlot::Attn(attn_idx);
                    attn_idx += 1;
                    s
                }
            };
            self.encode_moe_block_gpu(&enc, block, slot, position, session)?;
        }

        encode_rms_norm_mul_f32(
            self.ctx,
            &enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            &enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            h,
            arch.vocab_size as usize,
        )?;
        enc.end();
        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;

        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let mut logits = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, logits.as_mut_ptr(), logits.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            logits,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    fn single_token_argmax_profiled_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(i32, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let ids_buf = session.ids_buf.clone();
        let argmax_tok = session.argmax_tok.clone();

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);
        self.encode_single_token_argmax_moe(&enc, position, session, &ids_buf, &argmax_tok)?;
        enc.end();
        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;

        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let argmax = unsafe {
            let src = argmax_tok.buffer.contents().as_ptr() as *const i32;
            *src
        };
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            argmax,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    /// Same as [`single_token`] but ALSO copies the pre-output_norm hidden
    /// state (the residual stream right before the final RMSNorm + lm_head)
    /// into `hidden_dst`. This is the input the MTP head's `prev_hidden`
    /// argument expects per `docs/H4-MTP.md` §1.2.
    ///
    /// `hidden_dst` must be a zero-copy F32 tensor of shape `[H]`. It's
    /// kept GPU-resident so the next MTP draft call can consume it
    /// without a CPU readback. The copy happens inside the same command
    /// buffer as the forward, so `hidden_dst` is up-to-date by the time
    /// this call returns (which commits + waits).
    ///
    /// Caller must ensure `hidden_dst` is not aliased with any tensor
    /// the next forward call writes (typically allocate it as part of
    /// the MTP session arena).
    /// Multi-layer hidden-state capture variant of [`single_token`].
    /// Captures the residual stream `s.x` (post-FFN, post-residual) at
    /// each layer index in `target_layer_ids`, writing into the
    /// caller-supplied `hidden_dst` of shape `[K · H]` where
    /// `K = target_layer_ids.len()`.
    ///
    /// Captured layout: `hidden_dst[k * H .. (k+1) * H]` holds the
    /// residual after `target_layer_ids[k]` runs (in the same K order
    /// the caller specified, NOT sorted by layer index).
    ///
    /// All scatters happen inside the same command buffer as the
    /// forward, so `hidden_dst` is up-to-date by the time this returns.
    /// `target_layer_ids` may be empty (degenerate case: produces no
    /// hidden capture; equivalent to `single_token`).
    ///
    /// Used by H5 to bootstrap `target_ctx` from prompt prefill: per
    /// docs/H5-DFLASH.md §1.1, the DFlash drafter consumes K=5 target
    /// layer hiddens fused via `dflash_fc`. Caller is responsible for
    /// stacking these K hiddens into the final
    /// `[K · H_target, ctx_len]` cross-context buffer.
    pub fn single_token_with_multi_hidden(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        target_layer_ids: &[u32],
        hidden_dst: &MetalTensor,
    ) -> Result<Vec<f32>, MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let k = target_layer_ids.len();
        if hidden_dst.n_elements() as usize != k * h {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "single_token_with_multi_hidden.hidden_dst",
                detail: format!(
                    "expected {} elements (K={k} layers × H={h}), got {}",
                    k * h,
                    hidden_dst.n_elements()
                ),
            }));
        }
        for &lid in target_layer_ids {
            if (lid as usize) >= self.model.blocks.len() {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "single_token_with_multi_hidden.target_layer_ids",
                    detail: format!("layer id {lid} >= n_layer {}", self.model.blocks.len()),
                }));
            }
        }

        // Stage token id.
        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        // Embed → s.x.
        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            h,
        )?;

        // Per-block, capturing at requested layer indices AFTER each
        // block's residual #2 (s.x is the post-FFN residual, exactly
        // matching the CPU `single_token_capture_layers` capture point).
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
            // Capture at any (possibly multiple) target_layer_ids slot
            // matching this block. Scatters run inline with the rest of
            // the command buffer; reads s.x BEFORE the next block writes
            // it, which is required since s.x is reused per layer.
            for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                if lid as usize == il {
                    encode_scatter_offset_f32(
                        self.ctx,
                        &enc,
                        &session.x,
                        hidden_dst,
                        k_idx * h,
                        h,
                    )?;
                }
            }
        }

        // Final RMSNorm + lm_head — produces final logits as usual.
        encode_rms_norm_mul_f32(
            self.ctx,
            &enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            &enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            h,
            arch.vocab_size as usize,
        )?;

        enc.end();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let mut logits = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, logits.as_mut_ptr(), logits.len());
        }
        Ok(logits)
    }

    /// Skip-tail variant of [`single_token`] for prefill loops.
    ///
    /// Encodes embedding + per-block forward into one command buffer,
    /// commits, waits — but DOES NOT run final RMSNorm, lm_head, or
    /// readback logits. Returns `Ok(())` on success.
    ///
    /// The session state (KV cache, GDN state, position counters) is
    /// advanced exactly as if [`single_token`] had been called. Only
    /// `session.h` and `session.logits` are left in an unspecified
    /// state (downstream consumers must treat them as scratch). Callers
    /// MUST run a non-no-tail variant for the LAST prompt token to
    /// produce the bootstrap logits for the decode phase.
    ///
    /// v0.75.0: shipped to skip ~2 ms/token of lm_head Q6_K mat-vec +
    /// readback during prompt prefill. Estimated ~5% TTFT win at
    /// ctx ≥ 181. Establishes the API shape for v0.75.1's packed
    /// multi-token prefill (which replaces the body wholesale).
    pub fn single_token_no_tail(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            h,
        )?;

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
        }

        // SKIP final RMSNorm + lm_head + readback (the "tail").
        enc.end();
        cmd_buf.commit();
        // Codex Q3: keep wait. Skipping the wait introduces a
        // session.ids_buf reuse hazard — the next prefill iteration
        // CPU-writes ids_buf for the next token, and Metal cmd-buffer
        // ordering does NOT order CPU writes to shared buffers after
        // commit. If get_rows for token i hasn't run yet, it would
        // read the overwritten id. Defer real async pipelining to
        // v0.75.1 where packed prefill restructures this.
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Skip-tail variant of [`single_token_with_multi_hidden`] for
    /// DFlash prefill loops. Captures the K layer hiddens into
    /// `hidden_dst` exactly as [`single_token_with_multi_hidden`] does
    /// (those go on to feed `target_ctx_stacked` via the bench's
    /// `append_target_ctx_column_now`), but skips final RMSNorm,
    /// lm_head, and logits readback.
    ///
    /// See [`single_token_no_tail`] for the rationale and constraints.
    pub fn single_token_with_multi_hidden_no_tail(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        target_layer_ids: &[u32],
        hidden_dst: &MetalTensor,
    ) -> Result<(), MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let k = target_layer_ids.len();
        if hidden_dst.n_elements() as usize != k * h {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "single_token_with_multi_hidden_no_tail.hidden_dst",
                detail: format!(
                    "expected {} elements (K={k} layers × H={h}), got {}",
                    k * h,
                    hidden_dst.n_elements()
                ),
            }));
        }
        for &lid in target_layer_ids {
            if (lid as usize) >= self.model.blocks.len() {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "single_token_with_multi_hidden_no_tail.target_layer_ids",
                    detail: format!("layer id {lid} >= n_layer {}", self.model.blocks.len()),
                }));
            }
        }

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            h,
        )?;

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
            // Capture post-residual-#2 hidden at any matching layer
            // (matches v0.74.4 capture-point semantics). Runs inside
            // the same command buffer, before the next block writes
            // session.x.
            for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                if lid as usize == il {
                    encode_scatter_offset_f32(
                        self.ctx,
                        &enc,
                        &session.x,
                        hidden_dst,
                        k_idx * h,
                        h,
                    )?;
                }
            }
        }

        // SKIP final RMSNorm + lm_head + readback.
        enc.end();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    pub fn single_token_with_hidden(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        hidden_dst: &MetalTensor,
    ) -> Result<Vec<f32>, MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        if hidden_dst.n_elements() as usize != h {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "single_token_with_hidden.hidden_dst",
                detail: format!("expected {h} elements, got {}", hidden_dst.n_elements()),
            }));
        }

        // Stage token id into the ids buffer.
        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        // (1) Embedding lookup → s.x.
        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            h,
        )?;

        // (2) Per-block.
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
        }

        // (3) Final RMSNorm over residual stream. session.x is read,
        // session.h is written. After this point s.x is still untouched
        // (output_norm doesn't write back to its input).
        encode_rms_norm_mul_f32(
            self.ctx,
            &enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;

        // (4) LM head → logits.
        encode_mat_vec_dispatch(
            self.ctx,
            &enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            h,
            arch.vocab_size as usize,
        )?;

        // (5) Capture pre-output_norm hidden into the caller-supplied dst.
        // Runs LAST in the command buffer to avoid any chance of
        // interleaving with the rms_norm + lm_head reads of session.x.
        // session.x has not been mutated since step (2) ended; the read
        // here pulls the same bytes RMSNorm read.
        encode_scatter_offset_f32(self.ctx, &enc, &session.x, hidden_dst, 0, h)?;

        enc.end();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        // Read back logits to CPU.
        let mut logits = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, logits.as_mut_ptr(), logits.len());
        }
        Ok(logits)
    }

    /// Same forward as [`single_token_with_hidden`] but reads back only the
    /// argmax token id instead of the full logits row. Used by the MTP
    /// speculative path, which needs the greedy next token and the hidden
    /// carry but never consumes full-vocab logits on CPU.
    pub fn single_token_argmax_with_hidden(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        hidden_dst: &MetalTensor,
    ) -> Result<i32, MfError> {
        let arch = &self.model.arch;
        if arch.kind == ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        if hidden_dst.n_elements() as usize != h {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "single_token_argmax_with_hidden.hidden_dst",
                detail: format!("expected {h} elements, got {}", hidden_dst.n_elements()),
            }));
        }

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            h,
        )?;

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
        }

        encode_rms_norm_mul_f32(
            self.ctx,
            &enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            &enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            h,
            arch.vocab_size as usize,
        )?;
        encode_scatter_offset_f32(self.ctx, &enc, &session.x, hidden_dst, 0, h)?;
        encode_argmax_f32(
            self.ctx,
            &enc,
            &session.logits,
            &session.argmax_tok,
            1,
            arch.vocab_size as usize,
        )?;

        enc.end();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let argmax = unsafe {
            let src = session.argmax_tok.buffer.contents().as_ptr() as *const i32;
            *src
        };
        Ok(argmax)
    }

    /// Phase-resolved profiling: splits the per-token forward across
    /// MANY command buffers (one per block, plus embedding and lm_head)
    /// so we can attribute GPU time to logical phases. ★ ARTIFACT WARNING:
    /// the returned `wall_with_artifact_ms` is the WALL CLOCK of this
    /// split execution and includes ~10-15 ms of per-phase command-buffer
    /// overhead (commit + waitUntilCompleted + setup) NOT present in
    /// production single-token decode. Use the per-phase GPU times
    /// (sum-of-phases) for proportional reasoning, NOT the wall number,
    /// when comparing to production.
    ///
    /// For production-realistic ms/token, use [`single_token_profiled`]
    /// instead — that uses one command buffer per token (matching
    /// production) and returns true wall + true CPU encode + true GPU
    /// kernel times.
    ///
    /// Returns: (logits, wall_with_artifact_ms, per-phase GPU ms map)
    pub fn single_token_phase_profiled(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<PhaseProfileOutput, MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return self.single_token_phase_profiled_moe(token_id, position, session);
        }
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }

        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let mut phases: Vec<(String, f64)> = Vec::new();

        // Phase: embedding lookup.
        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_get_rows_f32(
                self.ctx,
                &enc,
                &self.model.token_embd,
                &session.ids_buf,
                &session.x,
                1,
                arch.hidden_size as usize,
            )?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            phases.push(("embedding".into(), ms));
        }

        // One command buffer per block. We aggregate by class (gdn vs attn)
        // so the report is digestible.
        let mut gdn_total_ms = 0.0f64;
        let mut attn_total_ms = 0.0f64;
        let mut gdn_count = 0usize;
        let mut attn_count = 0usize;
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            match block {
                MetalBlock::Gdn(_) => {
                    gdn_total_ms += ms;
                    gdn_count += 1;
                }
                MetalBlock::Attn(_) => {
                    attn_total_ms += ms;
                    attn_count += 1;
                }
            }
        }
        phases.push((format!("gdn layers (×{gdn_count})"), gdn_total_ms));
        phases.push((format!("attn layers (×{attn_count})"), attn_total_ms));

        // Phase: final norm.
        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            phases.push(("final norm".into(), ms));
        }

        // Phase: lm head.
        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                arch.hidden_size as usize,
                arch.vocab_size as usize,
            )?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            phases.push(("lm head".into(), ms));
        }

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((out, total_ms, phases))
    }

    fn single_token_phase_profiled_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<PhaseProfileOutput, MfError> {
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let t_total = std::time::Instant::now();
        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let mut phases: Vec<(String, f64)> = Vec::new();
        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_get_rows_f32(
                self.ctx,
                &enc,
                &self.model.token_embd,
                &session.ids_buf,
                &session.x,
                1,
                h,
            )?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            phases.push((
                "embedding".into(),
                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
            ));
        }

        let mut gdn_pre_norm_total_ms = 0.0f64;
        let mut gdn_front_proj_total_ms = 0.0f64;
        let mut gdn_qkv_proj_total_ms = 0.0f64;
        let mut gdn_z_proj_total_ms = 0.0f64;
        let mut gdn_beta_proj_total_ms = 0.0f64;
        let mut gdn_alpha_proj_total_ms = 0.0f64;
        let mut gdn_alpha_beta_total_ms = 0.0f64;
        let mut gdn_tail_total_ms = 0.0f64;
        let mut gdn_out_proj_total_ms = 0.0f64;
        let mut gdn_resid_post_total_ms = 0.0f64;
        let mut attn_mixer_total_ms = 0.0f64;
        let mut route_total_ms = 0.0f64;
        let mut ffn_apply_total_ms = 0.0f64;
        let mut ffn_gate_up_wave_total_ms = 0.0f64;
        let mut ffn_shared_silu_total_ms = 0.0f64;
        let mut ffn_down_wave_total_ms = 0.0f64;
        let mut ffn_finalizer_total_ms = 0.0f64;
        let mut ffn_routed_gate_up_total_ms = 0.0f64;
        let mut ffn_routed_down_total_ms = 0.0f64;
        let mut ffn_shared_gate_up_total_ms = 0.0f64;
        let mut ffn_shared_down_total_ms = 0.0f64;
        let mut ffn_fallback_routed_total_ms = 0.0f64;
        let mut ffn_fallback_shared_total_ms = 0.0f64;
        let mut gdn_count = 0usize;
        let mut attn_count = 0usize;
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        let split_gdn_proj = phase_gdn_proj_split_enabled();
        let split_ffn_apply = phase_moe_ffn_split_enabled();
        let deep_split_ffn_apply = phase_moe_ffn_deep_split_enabled();
        for block in &self.model.blocks {
            let slot = match block {
                MetalBlock::Gdn(_) => {
                    let s = MixerSlot::Gdn(gdn_idx);
                    gdn_idx += 1;
                    gdn_count += 1;
                    s
                }
                MetalBlock::Attn(_) => {
                    let s = MixerSlot::Attn(attn_idx);
                    attn_idx += 1;
                    attn_count += 1;
                    s
                }
            };
            let (ffn_gate, ffn_up, ffn_down, moe) = match block {
                MetalBlock::Gdn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
                MetalBlock::Attn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
            };
            let moe = moe.ok_or(MfError::UnsupportedMoe)?;

            match (block, slot) {
                (MetalBlock::Gdn(g), MixerSlot::Gdn(gdn_i)) => {
                    let v_dim = self.model.arch.gdn_n_v_heads as usize
                        * self.model.arch.gdn_head_dim as usize;

                    {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        gdn_pre_norm_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                    if split_gdn_proj {
                        let n_v = self.model.arch.gdn_n_v_heads as usize;
                        let n_k = self.model.arch.gdn_n_k_heads as usize;
                        let head_dim = self.model.arch.gdn_head_dim as usize;
                        let conv_dim = (2 * n_k + n_v) * head_dim;
                        {
                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            if decode_gdn_noop_qkv_enabled() {
                                encode_fill_f32(self.ctx, &enc, &session.gdn_qkv, 0.0)?;
                            } else {
                                encode_mat_vec_dispatch(
                                    self.ctx,
                                    &enc,
                                    &g.in_proj_qkv,
                                    &session.h,
                                    &session.gdn_qkv,
                                    h,
                                    conv_dim,
                                )?;
                            }
                            enc.end();
                            cmd.commit();
                            cmd.waitUntilCompleted();
                            gdn_qkv_proj_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        }
                        {
                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            if decode_gdn_noop_z_enabled() {
                                encode_fill_f32(self.ctx, &enc, &session.gdn_z, 0.0)?;
                            } else {
                                encode_mat_vec_dispatch(
                                    self.ctx,
                                    &enc,
                                    &g.in_proj_z,
                                    &session.h,
                                    &session.gdn_z,
                                    h,
                                    v_dim,
                                )?;
                            }
                            enc.end();
                            cmd.commit();
                            cmd.waitUntilCompleted();
                            gdn_z_proj_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        }
                        {
                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            if decode_gdn_noop_beta_enabled() {
                                encode_fill_f32(self.ctx, &enc, &session.gdn_b, 0.0)?;
                            } else {
                                encode_mat_vec_dispatch(
                                    self.ctx,
                                    &enc,
                                    &g.beta_proj,
                                    &session.h,
                                    &session.gdn_b,
                                    h,
                                    n_v,
                                )?;
                            }
                            enc.end();
                            cmd.commit();
                            cmd.waitUntilCompleted();
                            gdn_beta_proj_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        }
                        {
                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            if decode_gdn_noop_alpha_enabled() {
                                encode_fill_f32(self.ctx, &enc, &session.gdn_a, 0.0)?;
                            } else {
                                encode_mat_vec_dispatch(
                                    self.ctx,
                                    &enc,
                                    &g.alpha_proj,
                                    &session.h,
                                    &session.gdn_a,
                                    h,
                                    n_v,
                                )?;
                            }
                            enc.end();
                            cmd.commit();
                            cmd.waitUntilCompleted();
                            gdn_alpha_proj_total_ms +=
                                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        }
                    } else {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        gdn_front_proj_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                    {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        encode_sigmoid_f32(self.ctx, &enc, &session.gdn_b, &session.gdn_beta)?;
                        encode_gdn_decay_chain_f32(
                            self.ctx,
                            &enc,
                            &session.gdn_a,
                            &g.dt_bias,
                            &g.a_log,
                            &session.gdn_alpha,
                        )?;
                        enc.end();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        gdn_alpha_beta_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                    {
                        let gdn_qkv = session.gdn_qkv.clone();
                        let gdn_z = session.gdn_z.clone();
                        let gdn_alpha = session.gdn_alpha.clone();
                        let gdn_beta = session.gdn_beta.clone();
                        let gdn_normed = session.gdn_normed.clone();
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        self.encode_gdn_tail(
                            &enc,
                            g,
                            gdn_i,
                            session,
                            &gdn_qkv,
                            &gdn_z,
                            &gdn_alpha,
                            &gdn_beta,
                            &gdn_normed,
                        )?;
                        enc.end();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        gdn_tail_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                    {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        encode_mat_vec_dispatch(
                            self.ctx,
                            &enc,
                            &g.out_proj,
                            &session.gdn_normed,
                            &session.mixer_out,
                            v_dim,
                            h,
                        )?;
                        enc.end();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        gdn_out_proj_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                    {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        encode_add_inplace_f32(self.ctx, &enc, &session.x, &session.mixer_out)?;
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.post_attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        gdn_resid_post_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                }
                (MetalBlock::Attn(_), MixerSlot::Attn(_)) => {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_mixer_prep(&enc, block, slot, position, session)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    attn_mixer_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                }
                _ => {
                    return Err(MfError::Metal(MetalError::BadShape {
                        kernel: "single_token_phase_profiled_moe",
                        detail: "mixer slot type did not match block kind".into(),
                    }));
                }
            }

            {
                let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                self.encode_moe_route_prepare(&enc, session, moe)?;
                enc.end();
                cmd.commit();
                cmd.waitUntilCompleted();
                route_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            }

            if split_ffn_apply {
                if deep_split_ffn_apply
                    && moe.gate_exps.dtype == GgmlType::Q4_K
                    && moe.up_exps.dtype == GgmlType::Q4_K
                {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_routed_gate_up_q4_gpu(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_routed_gate_up_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    let routed_weighted_sum_is_pending =
                        self.encode_moe_routed_down_only_gpu(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_routed_down_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    let shared_inner_fused =
                        self.encode_moe_shared_ffn_gate_up_gpu(&enc, session, ffn_gate, ffn_up)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_shared_gate_up_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    if !shared_inner_fused {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        self.encode_moe_shared_ffn_silu_gpu(&enc, session)?;
                        enc.end();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        ffn_shared_silu_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_shared_down_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    self.encode_moe_ffn_final_wave_gpu(
                        &cmd,
                        session,
                        routed_weighted_sum_is_pending,
                    )?;
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_finalizer_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                } else if concurrent_shared_moe_decode_enabled()
                    && moe.gate_exps.dtype == GgmlType::Q4_K
                    && moe.up_exps.dtype == GgmlType::Q4_K
                {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let shared_inner_fused =
                        self.encode_moe_ffn_gate_up_wave_gpu(&cmd, session, ffn_gate, ffn_up, moe)?;
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_gate_up_wave_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    if !shared_inner_fused {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        self.encode_moe_shared_ffn_silu_gpu(&enc, session)?;
                        enc.end();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        ffn_shared_silu_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let routed_weighted_sum_is_pending =
                        self.encode_moe_ffn_down_wave_gpu(&cmd, session, ffn_down, moe)?;
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_down_wave_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    self.encode_moe_ffn_final_wave_gpu(
                        &cmd,
                        session,
                        routed_weighted_sum_is_pending,
                    )?;
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_finalizer_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                } else {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_routed_ffn_gpu(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_fallback_routed_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_shared_ffn_core_gpu(&enc, session, ffn_gate, ffn_up, ffn_down)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_fallback_shared_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_final_residual_gpu(&enc, session)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_finalizer_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                }
            } else {
                let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                if concurrent_shared_moe_decode_enabled() {
                    self.encode_moe_ffn_apply_gpu_concurrent_shared(
                        &cmd, session, ffn_gate, ffn_up, ffn_down, moe,
                    )?;
                } else {
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_ffn_apply_gpu(&enc, session, ffn_gate, ffn_up, ffn_down, moe)?;
                    enc.end();
                }
                cmd.commit();
                cmd.waitUntilCompleted();
                ffn_apply_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            }
        }
        phases.push((
            format!("gdn pre_norm (x{gdn_count})"),
            gdn_pre_norm_total_ms,
        ));
        if split_gdn_proj {
            phases.push((
                format!("gdn qkv proj (x{gdn_count})"),
                gdn_qkv_proj_total_ms,
            ));
            phases.push((format!("gdn z proj (x{gdn_count})"), gdn_z_proj_total_ms));
            phases.push((
                format!("gdn beta proj (x{gdn_count})"),
                gdn_beta_proj_total_ms,
            ));
            phases.push((
                format!("gdn alpha proj (x{gdn_count})"),
                gdn_alpha_proj_total_ms,
            ));
        } else {
            phases.push((
                format!("gdn front proj (x{gdn_count})"),
                gdn_front_proj_total_ms,
            ));
        }
        phases.push((
            format!("gdn alpha/beta (x{gdn_count})"),
            gdn_alpha_beta_total_ms,
        ));
        phases.push((format!("gdn tail (x{gdn_count})"), gdn_tail_total_ms));
        phases.push((
            format!("gdn out_proj (x{gdn_count})"),
            gdn_out_proj_total_ms,
        ));
        phases.push((
            format!("gdn resid/post (x{gdn_count})"),
            gdn_resid_post_total_ms,
        ));
        phases.push((format!("attn mixer (x{attn_count})"), attn_mixer_total_ms));
        phases.push(("moe route".into(), route_total_ms));
        if split_ffn_apply {
            if ffn_routed_gate_up_total_ms > 0.0 {
                phases.push(("moe ffn routed gate/up".into(), ffn_routed_gate_up_total_ms));
            }
            if ffn_routed_down_total_ms > 0.0 {
                phases.push(("moe ffn routed down".into(), ffn_routed_down_total_ms));
            }
            if ffn_shared_gate_up_total_ms > 0.0 {
                phases.push(("moe ffn shared gate/up".into(), ffn_shared_gate_up_total_ms));
            }
            if ffn_gate_up_wave_total_ms > 0.0 {
                phases.push(("moe ffn gate/up wave".into(), ffn_gate_up_wave_total_ms));
            }
            if ffn_shared_silu_total_ms > 0.0 {
                phases.push(("moe ffn shared silu".into(), ffn_shared_silu_total_ms));
            }
            if ffn_shared_down_total_ms > 0.0 {
                phases.push(("moe ffn shared down".into(), ffn_shared_down_total_ms));
            }
            if ffn_down_wave_total_ms > 0.0 {
                phases.push(("moe ffn down wave".into(), ffn_down_wave_total_ms));
            }
            if ffn_fallback_routed_total_ms > 0.0 {
                phases.push((
                    "moe ffn fallback routed".into(),
                    ffn_fallback_routed_total_ms,
                ));
            }
            if ffn_fallback_shared_total_ms > 0.0 {
                phases.push((
                    "moe ffn fallback shared".into(),
                    ffn_fallback_shared_total_ms,
                ));
            }
            phases.push(("moe ffn finalizer".into(), ffn_finalizer_total_ms));
        } else {
            phases.push(("moe ffn apply".into(), ffn_apply_total_ms));
        }

        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            phases.push((
                "final norm".into(),
                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
            ));
        }
        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                h,
                arch.vocab_size as usize,
            )?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            phases.push((
                "lm head".into(),
                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
            ));
        }

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((out, total_ms, phases))
    }

    fn single_token_profiled_dense_serial(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }

        let t_total = std::time::Instant::now();

        // Stage the token id into the ids_buf (i32 view of the F32 buffer).
        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        // (1) Embedding lookup → x.
        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            arch.hidden_size as usize,
        )?;

        // (2) Per-block.
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
        }

        // (3) Final RMSNorm over residual stream.
        encode_rms_norm_mul_f32(
            self.ctx,
            &enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;

        // (4) LM head → logits. Dispatch on dtype (Q4_K, Q6_K, F32).
        encode_mat_vec_dispatch(
            self.ctx,
            &enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            arch.hidden_size as usize,
            arch.vocab_size as usize,
        )?;

        enc.end();
        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;

        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;

        // GPU-reported wall-clock execution time (CFTimeInterval seconds).
        let gpu_start = cmd_buf.GPUStartTime();
        let gpu_end = cmd_buf.GPUEndTime();
        let gpu_kernel_ms = (gpu_end - gpu_start) * 1e3;

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;

        Ok((
            out,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    pub fn single_token_profiled(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return if concurrent_gdn_moe_decode_enabled() {
                self.single_token_profiled_concurrent_gdn_moe(token_id, position, session)
            } else {
                self.single_token_profiled_moe(token_id, position, session)
            };
        }
        if concurrent_gdn_dense_decode_enabled() {
            return self.single_token_profiled_concurrent_gdn_dense(token_id, position, session);
        }
        self.single_token_profiled_dense_serial(token_id, position, session)
    }

    pub fn encode_block(
        &self,
        enc: &KernelEncoder,
        _il: usize,
        block: &MetalBlock,
        gdn_idx: &mut usize,
        attn_idx: &mut usize,
        position: u32,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        // Pre-mixer norm.
        let attn_norm = match block {
            MetalBlock::Gdn(g) => &g.attn_norm,
            MetalBlock::Attn(a) => &a.attn_norm,
        };
        encode_rms_norm_mul_f32(self.ctx, enc, &s.x, attn_norm, &s.h, RMS_EPS)?;

        // Mixer (GDN or Attn) → mixer_out.
        match block {
            MetalBlock::Gdn(g) => {
                let i = *gdn_idx;
                *gdn_idx += 1;
                self.encode_gdn(enc, g, i, s)?;
            }
            MetalBlock::Attn(a) => {
                let i = *attn_idx;
                *attn_idx += 1;
                self.encode_attn(enc, a, i, position, s)?;
            }
        }

        self.encode_post_mixer_ffn(enc, block, s)?;
        Ok(())
    }

    fn encode_post_mixer_ffn(
        &self,
        enc: &KernelEncoder,
        block: &MetalBlock,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;

        // Residual #1: x += mixer_out.
        encode_add_inplace_f32(self.ctx, enc, &s.x, &s.mixer_out)?;

        // Pre-FFN norm.
        let post_norm = match block {
            MetalBlock::Gdn(g) => &g.post_attn_norm,
            MetalBlock::Attn(a) => &a.post_attn_norm,
        };
        encode_rms_norm_mul_f32(self.ctx, enc, &s.x, post_norm, &s.h, RMS_EPS)?;

        // SwiGLU FFN.
        let (g_w, u_w, d_w) = match block {
            MetalBlock::Gdn(g) => (&g.ffn_gate, &g.ffn_up, &g.ffn_down),
            MetalBlock::Attn(a) => (&a.ffn_gate, &a.ffn_up, &a.ffn_down),
        };
        let f = arch.intermediate_size as usize;
        // Fused SwiGLU when both gate and up are Q4_K (27B production).
        // Falls back to 3-dispatch path for F32 weights (0.8B) or other dtypes.
        // Per Jeff & Sanjay: amortize boundary crossings + eliminate
        // intermediate materialization (no separate gate/up writes).
        let ffn_fused = g_w.dtype == GgmlType::Q4_K && u_w.dtype == GgmlType::Q4_K;
        if ffn_fused {
            encode_ffn_swiglu_q4_K_f32(self.ctx, enc, g_w, u_w, &s.h, &s.ffn_inner, h, f)?;
        } else {
            encode_mat_vec_dispatch(self.ctx, enc, g_w, &s.h, &s.ffn_gate, h, f)?;
            encode_mat_vec_dispatch(self.ctx, enc, u_w, &s.h, &s.ffn_up, h, f)?;
            encode_silu_mul_f32(self.ctx, enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)?;
        }
        encode_mat_vec_dispatch(self.ctx, enc, d_w, &s.ffn_inner, &s.ffn_out, f, h)?;

        // Residual #2: x += ffn_out.
        encode_add_inplace_f32(self.ctx, enc, &s.x, &s.ffn_out)?;
        Ok(())
    }

    fn encode_gdn_front_projections(
        &self,
        enc: &KernelEncoder,
        gb: &MetalGdnBlock,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        let conv_dim = (2 * n_k + n_v) * head_dim;
        let v_dim = n_v * head_dim;

        if decode_gdn_noop_qkv_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_qkv, 0.0)?;
        } else {
            encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &gb.in_proj_qkv,
                &s.h,
                &s.gdn_qkv,
                h,
                conv_dim,
            )?;
        }
        if decode_gdn_noop_z_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_z, 0.0)?;
        } else {
            encode_mat_vec_dispatch(self.ctx, enc, &gb.in_proj_z, &s.h, &s.gdn_z, h, v_dim)?;
        }
        if decode_gdn_noop_beta_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_b, 0.0)?;
        } else {
            encode_mat_vec_dispatch(self.ctx, enc, &gb.beta_proj, &s.h, &s.gdn_b, h, n_v)?;
        }
        if decode_gdn_noop_alpha_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_a, 0.0)?;
        } else {
            encode_mat_vec_dispatch(self.ctx, enc, &gb.alpha_proj, &s.h, &s.gdn_a, h, n_v)?;
        }
        Ok(())
    }

    fn encode_gdn_after_projections(
        &self,
        enc: &KernelEncoder,
        gb: &MetalGdnBlock,
        gdn_i: usize,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let v_dim = arch.gdn_n_v_heads as usize * arch.gdn_head_dim as usize;
        let h = arch.hidden_size as usize;
        let gdn_qkv = s.gdn_qkv.clone();
        let gdn_z = s.gdn_z.clone();
        let gdn_alpha = s.gdn_alpha.clone();
        let gdn_beta = s.gdn_beta.clone();
        let gdn_normed = s.gdn_normed.clone();

        encode_sigmoid_f32(self.ctx, enc, &s.gdn_b, &s.gdn_beta)?;
        encode_gdn_decay_chain_f32(
            self.ctx,
            enc,
            &s.gdn_a,
            &gb.dt_bias,
            &gb.a_log,
            &s.gdn_alpha,
        )?;
        self.encode_gdn_tail(
            enc,
            gb,
            gdn_i,
            s,
            &gdn_qkv,
            &gdn_z,
            &gdn_alpha,
            &gdn_beta,
            &gdn_normed,
        )?;
        if decode_gdn_noop_out_enabled() {
            encode_fill_f32(self.ctx, enc, &s.mixer_out, 0.0)?;
        } else {
            encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &gb.out_proj,
                &gdn_normed,
                &s.mixer_out,
                v_dim,
                h,
            )?;
        }
        Ok(())
    }

    fn encode_attn_front_projections(
        &self,
        enc: &KernelEncoder,
        ab: &MetalAttnBlock,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_q = arch.n_q_heads as usize;
        let n_kv = arch.n_kv_heads as usize;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;

        encode_mat_vec_dispatch(self.ctx, enc, &ab.q, &s.h, &s.attn_q_full, h, 2 * q_dim)?;
        encode_mat_vec_dispatch(self.ctx, enc, &ab.k, &s.h, &s.attn_k_now, h, kv_dim)?;
        encode_mat_vec_dispatch(self.ctx, enc, &ab.v, &s.h, &s.attn_v_now, h, kv_dim)?;
        Ok(())
    }

    fn encode_attn_after_projections(
        &self,
        enc: &KernelEncoder,
        ab: &MetalAttnBlock,
        attn_i: usize,
        position: u32,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_q = arch.n_q_heads as usize;
        let n_kv = arch.n_kv_heads as usize;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;

        encode_split_q_gate_f32(
            self.ctx,
            enc,
            &s.attn_q_full,
            &s.attn_q,
            &s.attn_gate,
            n_q,
            head_dim,
        )?;
        encode_rms_norm_batched_f32(
            self.ctx,
            enc,
            &s.attn_q,
            &ab.q_norm,
            &s.attn_q_normed,
            n_q,
            head_dim,
            RMS_EPS,
        )?;
        encode_rms_norm_batched_f32(
            self.ctx,
            enc,
            &s.attn_k_now,
            &ab.k_norm,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            RMS_EPS,
        )?;
        encode_rope_neox_f32(
            self.ctx,
            enc,
            &s.attn_q_normed,
            n_q,
            head_dim,
            n_rot,
            position,
            arch.rope_theta,
        )?;
        encode_rope_neox_f32(
            self.ctx,
            enc,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            n_rot,
            position,
            arch.rope_theta,
        )?;
        let kv_dst_off = usize::try_from(checked_u64_mul(
            position as u64,
            kv_dim as u64,
            "kv dst offset overflow",
        )?)
        .map_err(|_| MetalError::BadShape {
            kernel: "attn_step",
            detail: "kv dst offset does not fit usize".into(),
        })?;
        match s.kv_k[attn_i].dtype {
            GgmlType::F16 => encode_scatter_offset_f32_to_f16_kv(
                self.ctx,
                enc,
                &s.attn_k_normed,
                &s.attn_v_now,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                kv_dst_off,
                kv_dim,
            )?,
            GgmlType::Q8_0 => encode_scatter_offset_f32_to_q8_0_kv(
                self.ctx,
                enc,
                &s.attn_k_normed,
                &s.attn_v_now,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                kv_dst_off,
                kv_dim,
            )?,
            other => {
                return Err(MfError::UnsupportedDtype {
                    name: "attention KV cache".into(),
                    dtype: other,
                });
            }
        }
        s.kv_n_pos[attn_i] = position as usize + 1;

        const V4_HEAD_DIM: usize = 256;
        let group = n_q / n_kv;
        let use_v4 = head_dim == V4_HEAD_DIM && matches!(group, 4 | 6 | 8 | 16);
        if use_v4 {
            let nwg = attn_v4_choose_nwg(s.kv_n_pos[attn_i], group);
            let tile_c = attn_v4_choose_tile_c(s.kv_n_pos[attn_i], group);
            encode_attn_decode_v4_f32(
                self.ctx,
                enc,
                &s.attn_q_normed,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                &s.attn_v4_o_partial,
                &s.attn_v4_ml_partial,
                &s.attn_o,
                n_q,
                n_kv,
                head_dim,
                s.kv_n_pos[attn_i],
                nwg,
                tile_c,
            )?;
        } else {
            encode_attn_decode_f16kv_f32(
                self.ctx,
                enc,
                &s.attn_q_normed,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                &s.attn_o,
                n_q,
                n_kv,
                head_dim,
                s.kv_n_pos[attn_i],
            )?;
        }

        if decode_attn_sigmoid_mul_enabled() {
            encode_sigmoid_mul_f32(self.ctx, enc, &s.attn_gate, &s.attn_o, &s.attn_o)?;
        } else {
            encode_sigmoid_f32(self.ctx, enc, &s.attn_gate, &s.attn_q)?;
            encode_mul_f32(self.ctx, enc, &s.attn_o, &s.attn_q, &s.attn_o)?;
        }
        encode_mat_vec_dispatch(self.ctx, enc, &ab.o, &s.attn_o, &s.mixer_out, q_dim, h)?;
        Ok(())
    }

    pub fn encode_gdn(
        &self,
        enc: &KernelEncoder,
        gb: &MetalGdnBlock,
        gdn_i: usize,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        let conv_dim = (2 * n_k + n_v) * head_dim;
        let v_dim = n_v * head_dim;

        if decode_gdn_noop_qkv_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_qkv, 0.0)?;
        } else {
            // QKV input projection.
            encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &gb.in_proj_qkv,
                &s.h,
                &s.gdn_qkv,
                h,
                conv_dim,
            )?;
        }
        if decode_gdn_noop_z_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_z, 0.0)?;
        } else {
            // z projection.
            encode_mat_vec_dispatch(self.ctx, enc, &gb.in_proj_z, &s.h, &s.gdn_z, h, v_dim)?;
        }
        if decode_gdn_noop_beta_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_b, 0.0)?;
        } else {
            // beta source projection.
            encode_mat_vec_dispatch(self.ctx, enc, &gb.beta_proj, &s.h, &s.gdn_b, h, n_v)?;
        }
        if decode_gdn_noop_alpha_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_a, 0.0)?;
        } else {
            // α source projection.
            encode_mat_vec_dispatch(self.ctx, enc, &gb.alpha_proj, &s.h, &s.gdn_a, h, n_v)?;
        }
        encode_sigmoid_f32(self.ctx, enc, &s.gdn_b, &s.gdn_beta)?;
        // Decay-chain fusion: gdn_alpha stores exp(softplus(gdn_a + dt_bias) * a_log).
        // Replaces add_inplace + softplus + mul + per-row exp with one
        // per-head fused kernel.
        encode_gdn_decay_chain_f32(
            self.ctx,
            enc,
            &s.gdn_a,
            &gb.dt_bias,
            &gb.a_log,
            &s.gdn_alpha,
        )?;
        // Now `gdn_alpha` is the per-head decay exp(g), reused by every state row.

        // Conv1d step + SiLU. Mutates the conv buffer in place.
        encode_ssm_conv_silu_f32(
            self.ctx,
            enc,
            &s.gdn_qkv,
            &s.gdn_conv[gdn_i],
            &gb.conv1d,
            &s.gdn_qkv_conv,
            conv_dim,
        )?;

        // Split conv output into Q, K, V via zero-copy views (no dispatch).
        // Per Jeff & Sanjay (avoid copies / use indices instead of pointers):
        // the previous code did 3 copy_offset dispatches per layer × 32
        // GDN layers = 96 dispatches/token just to alias subranges.
        // view_subrange returns a sub-tensor pointing at the same MTLBuffer
        // with shifted offset, consumed by the next kernel directly.
        let q_view = s
            .gdn_qkv_conv
            .view_subrange(0, vec![(n_k * head_dim) as u64]);
        let k_view = s
            .gdn_qkv_conv
            .view_subrange((n_k * head_dim) as u64, vec![(n_k * head_dim) as u64]);
        let v_view = s
            .gdn_qkv_conv
            .view_subrange((2 * n_k * head_dim) as u64, vec![v_dim as u64]);

        // Per-head L2-norm of Q and K.
        encode_l2_norm_batched_f32(
            self.ctx,
            enc,
            &q_view,
            &s.gdn_q_norm,
            n_k,
            head_dim,
            RMS_EPS,
        )?;
        encode_l2_norm_batched_f32(
            self.ctx,
            enc,
            &k_view,
            &s.gdn_k_norm,
            n_k,
            head_dim,
            RMS_EPS,
        )?;

        // Recurrence step (kernel does the head-repeat internally).
        encode_gdn_step_decay_f32(
            self.ctx,
            enc,
            &s.gdn_q_norm,
            &s.gdn_k_norm,
            &v_view,
            &s.gdn_alpha,
            &s.gdn_beta,
            &s.gdn_state[gdn_i],
            &s.gdn_out,
            n_v,
            n_k,
            head_dim,
        )?;

        // RMSNormGated: y = norm(o) * silu(z), per head.
        encode_rmsnorm_gated_f32(
            self.ctx,
            enc,
            &s.gdn_out,
            &gb.norm,
            &s.gdn_z,
            &s.gdn_normed,
            n_v,
            head_dim,
            RMS_EPS,
        )?;

        // Output projection: [v_dim, hidden] → mixer_out.
        if decode_gdn_noop_out_enabled() {
            encode_fill_f32(self.ctx, enc, &s.mixer_out, 0.0)?;
        } else {
            encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &gb.out_proj,
                &s.gdn_normed,
                &s.mixer_out,
                v_dim,
                h,
            )?;
        }
        Ok(())
    }

    /// GDN per-token recurrence body, factored out for v0.73a.1 layer-major
    /// batching. Takes pre-computed inputs as zero-copy F32 views (one
    /// row of N-shaped pack buffers) and writes the per-head normed
    /// output to `gdn_normed_out` (also a row view).
    ///
    /// Performs in order: ssm_conv1d+silu (mutates `s.gdn_conv[gdn_i]`)
    /// → l2_norm Q/K → gdn_step (mutates `s.gdn_state[gdn_i]`) →
    /// rmsnorm_gated. Bit-exact with the corresponding inner part of
    /// `encode_gdn` when given the same inputs (validated by
    /// `gdn_tail_matches_inline`).
    ///
    /// The `_qkv_in` / `z_in` arguments alias rows of the layer-major
    /// pack buffers (`gdn_qkv_pack`, `gdn_z_pack`); `alpha_in` /
    /// `beta_in` come from the per-token session scratch (`s.gdn_alpha`,
    /// `s.gdn_beta`) populated by per-token alpha/beta mat-vec +
    /// sigmoid + decay-chain because production beta_proj/alpha_proj
    /// are F32 (small, mat-mat dispatch overhead > BW savings; see
    /// docs/H5-DFLASH.md rev 10).
    ///
    /// Caller's responsibility: per-token sequencing of `s.gdn_conv[gdn_i]`
    /// and `s.gdn_state[gdn_i]` (the recurrence is inherently
    /// per-token-sequential), and checkpoint blits between calls.
    pub fn encode_gdn_tail(
        &self,
        enc: &KernelEncoder,
        gb: &MetalGdnBlock,
        gdn_i: usize,
        s: &mut MetalSession,
        qkv_in: &MetalTensor,         // [conv_dim] F32 — one row of gdn_qkv_pack
        z_in: &MetalTensor,           // [v_dim] F32   — one row of gdn_z_pack
        alpha_in: &MetalTensor,       // [n_v] F32     — pre-computed decay exp(g)
        beta_in: &MetalTensor,        // [n_v] F32     — pre-computed sigmoid(beta)
        gdn_normed_out: &MetalTensor, // [v_dim] F32 — one row of gdn_normed_pack
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        let conv_dim = (2 * n_k + n_v) * head_dim;

        encode_ssm_conv_silu_f32(
            self.ctx,
            enc,
            qkv_in,
            &s.gdn_conv[gdn_i],
            &gb.conv1d,
            &s.gdn_qkv_conv,
            conv_dim,
        )?;
        let q_view = s
            .gdn_qkv_conv
            .view_subrange(0, vec![(n_k * head_dim) as u64]);
        let k_view = s
            .gdn_qkv_conv
            .view_subrange((n_k * head_dim) as u64, vec![(n_k * head_dim) as u64]);
        let v_view = s
            .gdn_qkv_conv
            .view_subrange((2 * n_k * head_dim) as u64, vec![(n_v * head_dim) as u64]);
        encode_l2_norm_batched_f32(
            self.ctx,
            enc,
            &q_view,
            &s.gdn_q_norm,
            n_k,
            head_dim,
            RMS_EPS,
        )?;
        encode_l2_norm_batched_f32(
            self.ctx,
            enc,
            &k_view,
            &s.gdn_k_norm,
            n_k,
            head_dim,
            RMS_EPS,
        )?;
        encode_gdn_step_decay_f32(
            self.ctx,
            enc,
            &s.gdn_q_norm,
            &s.gdn_k_norm,
            &v_view,
            alpha_in,
            beta_in,
            &s.gdn_state[gdn_i],
            &s.gdn_out,
            n_v,
            n_k,
            head_dim,
        )?;
        encode_rmsnorm_gated_f32(
            self.ctx,
            enc,
            &s.gdn_out,
            &gb.norm,
            z_in,
            gdn_normed_out,
            n_v,
            head_dim,
            RMS_EPS,
        )?;
        Ok(())
    }

    pub fn encode_attn(
        &self,
        enc: &KernelEncoder,
        ab: &MetalAttnBlock,
        attn_i: usize,
        position: u32,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_q = arch.n_q_heads as usize;
        let n_kv = arch.n_kv_heads as usize;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;

        // (1) Q projection: outputs 2 * q_dim (Q + gate interleaved per head).
        encode_mat_vec_dispatch(self.ctx, enc, &ab.q, &s.h, &s.attn_q_full, h, 2 * q_dim)?;

        // (2) Split into Q and gate.
        encode_split_q_gate_f32(
            self.ctx,
            enc,
            &s.attn_q_full,
            &s.attn_q,
            &s.attn_gate,
            n_q,
            head_dim,
        )?;

        // (3) Q-norm (per-head RMSNorm, shared per-channel weight).
        encode_rms_norm_batched_f32(
            self.ctx,
            enc,
            &s.attn_q,
            &ab.q_norm,
            &s.attn_q_normed,
            n_q,
            head_dim,
            RMS_EPS,
        )?;

        // (4) K, V projections.
        encode_mat_vec_dispatch(self.ctx, enc, &ab.k, &s.h, &s.attn_k_now, h, kv_dim)?;
        encode_mat_vec_dispatch(self.ctx, enc, &ab.v, &s.h, &s.attn_v_now, h, kv_dim)?;

        // (5) K-norm (per-head). Reuses Q-norm weight tensor type but
        // points at K's weight.
        encode_rms_norm_batched_f32(
            self.ctx,
            enc,
            &s.attn_k_now,
            &ab.k_norm,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            RMS_EPS,
        )?;

        // (6) Partial RoPE on Q (in `attn_q_normed`) and K (in `attn_k_normed`).
        encode_rope_neox_f32(
            self.ctx,
            enc,
            &s.attn_q_normed,
            n_q,
            head_dim,
            n_rot,
            position,
            arch.rope_theta,
        )?;
        encode_rope_neox_f32(
            self.ctx,
            enc,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            n_rot,
            position,
            arch.rope_theta,
        )?;

        // (7) KV cache append. v1 enforces strict-monotonic-from-zero;
        // we copy K and V at slot `position` directly via copy_offset
        // running in reverse direction. Since we don't yet have a
        // "scatter" kernel, we synchronously fill the KV cache slot via
        // CPU-visible memory under StorageModeShared. This requires a
        // wait-on-encoder boundary, which is the kind of CPU/GPU
        // synchronization codex warned against. **Acceptable for v1
        // single-token decode** because the readback is one row of
        // `kv_dim` floats (4 KB), and the next dispatch will see the
        // updated bytes. We just need to make sure the encoder is split
        // so the previous K/V projection has completed before we read.
        //
        // For v2 we'll add a `kv_append_f32` kernel that writes into the
        // cache slot using a small dispatch (no CPU sync needed).
        //
        // For now: emit a `copy_offset_f32` from the encoded K/V into
        // the right cache slot. The cache is `[capacity, n_kv_heads, head_dim]`
        // row-major, so slot `position` starts at `position * kv_stride`
        // bytes (here, in elements). We use copy_offset's offset arg
        // *inverted*: copy_offset reads from src+off into dst[0..n].
        // We need the opposite: copy from src[0..n] into dst+off. So
        // we add a small "scatter_offset" kernel below.
        // KV cache append: F32 source → F16 destination (cache is F16 to
        // halve attention bandwidth at long context). Fused K+V scatter
        // (Bulk-API principle): one dispatch writes both, saving 16 dispatches/token.
        let kv_dst_off = usize::try_from(checked_u64_mul(
            position as u64,
            kv_dim as u64,
            "kv dst offset overflow",
        )?)
        .map_err(|_| MetalError::BadShape {
            kernel: "attn_step_q8",
            detail: "kv dst offset does not fit usize".into(),
        })?;
        match s.kv_k[attn_i].dtype {
            GgmlType::F16 => encode_scatter_offset_f32_to_f16_kv(
                self.ctx,
                enc,
                &s.attn_k_normed,
                &s.attn_v_now,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                kv_dst_off,
                kv_dim,
            )?,
            GgmlType::Q8_0 => encode_scatter_offset_f32_to_q8_0_kv(
                self.ctx,
                enc,
                &s.attn_k_normed,
                &s.attn_v_now,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                kv_dst_off,
                kv_dim,
            )?,
            other => {
                return Err(MfError::UnsupportedDtype {
                    name: "attention KV cache".into(),
                    dtype: other,
                });
            }
        }
        s.kv_n_pos[attn_i] = position as usize + 1;

        // (8) Fused attention decode: scoring + softmax + V-aggregate.
        //
        // Selection: v4 (GQA-dedup + online softmax + split-K) when the
        // shape matches its hardcoded constants (head_dim=256, GROUP in
        // {4,6,8,16}). Falls back to f16kv naive kernel for other shapes.
        //
        // v4 gives 2-8× speedup over naive on the 27B shape AND removes
        // the n_pos ≤ ~7000 correctness cliff (naive's threadgroup-mem
        // scores buffer caps out around there).
        const V4_HEAD_DIM: usize = 256;
        let group = n_q / n_kv;
        let use_v4 = head_dim == V4_HEAD_DIM && matches!(group, 4 | 6 | 8 | 16);
        if use_v4 {
            let nwg = attn_v4_choose_nwg(s.kv_n_pos[attn_i], group);
            let tile_c = attn_v4_choose_tile_c(s.kv_n_pos[attn_i], group);
            encode_attn_decode_v4_f32(
                self.ctx,
                enc,
                &s.attn_q_normed,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                &s.attn_v4_o_partial,
                &s.attn_v4_ml_partial,
                &s.attn_o,
                n_q,
                n_kv,
                head_dim,
                s.kv_n_pos[attn_i],
                nwg,
                tile_c,
            )?;
        } else {
            encode_attn_decode_f16kv_f32(
                self.ctx,
                enc,
                &s.attn_q_normed,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                &s.attn_o,
                n_q,
                n_kv,
                head_dim,
                s.kv_n_pos[attn_i],
            )?;
        }

        // (9) Apply gated-attention sigmoid gate: attn_o *= sigmoid(gate).
        if decode_attn_sigmoid_mul_enabled() {
            encode_sigmoid_mul_f32(self.ctx, enc, &s.attn_gate, &s.attn_o, &s.attn_o)?;
        } else {
            encode_sigmoid_f32(self.ctx, enc, &s.attn_gate, &s.attn_q)?;
            encode_mul_f32(self.ctx, enc, &s.attn_o, &s.attn_q, &s.attn_o)?;
        }

        // (10) Output projection: q_dim → hidden.
        encode_mat_vec_dispatch(self.ctx, enc, &ab.o, &s.attn_o, &s.mixer_out, q_dim, h)?;
        Ok(())
    }
}

/// Helper: scatter `n` floats from `src[0..n]` into `dst[off..off+n]`.
/// Inverse of `copy_offset` (which gathers). Used to write into the KV
/// cache slot for the current position, and (via the metal_mtp module)
/// to assemble the `[e_normed, h_normed]` concat for the eh_proj input.
pub fn encode_scatter_offset_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    dst_off: usize,
    n: usize,
) -> Result<(), MetalError> {
    if src.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset",
            detail: format!("src.n={} != n={n}", src.n_elements()),
        });
    }
    let dst_end = dst_off.checked_add(n).ok_or_else(|| MetalError::BadShape {
        kernel: "scatter_offset",
        detail: format!("dst_off={dst_off} + n={n} overflows usize"),
    })?;
    if dst_end as u64 > dst.n_elements() {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset",
            detail: format!("dst_off+n={dst_end} > dst.n={}", dst.n_elements()),
        });
    }
    let n_u32 = u32::try_from(n).map_err(|_| MetalError::BadShape {
        kernel: "scatter_offset",
        detail: format!("n={n} does not fit u32 kernel args"),
    })?;
    let dst_off_u32 = u32::try_from(dst_off).map_err(|_| MetalError::BadShape {
        kernel: "scatter_offset",
        detail: format!("dst_off={dst_off} does not fit u32 kernel args"),
    })?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        dst_off: u32,
    }
    let pso = ctx.pipeline("kernel_scatter_offset_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n_u32,
            dst_off: dst_off_u32,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, dst);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        objc2_metal::MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Per-token timing profile.
///
/// * `cpu_encode_ms` — time spent in `KernelEncoder::begin` through
///   `enc.end()`. This is the CPU-side cost of encoding all dispatches
///   into the command buffer. ICB will collapse this to ~0.
/// * `gpu_kernel_ms` — `GPUEndTime - GPUStartTime`, the wall-clock the
///   GPU spent actually executing kernels. This is the floor a
///   correctness-preserving optimization can reach.
/// * `cpu_to_gpu_complete_ms` — `commit() + waitUntilCompleted()` wall
///   clock. Difference vs `gpu_kernel_ms` is mostly driver/queue
///   submission + completion handler overhead.
/// * `total_ms` — the user-visible per-token latency (incl. logits
///   readback).
#[derive(Debug, Clone, Copy)]
pub struct TokenProfile {
    pub cpu_encode_ms: f64,
    pub cpu_to_gpu_complete_ms: f64,
    pub gpu_kernel_ms: f64,
    pub total_ms: f64,
    pub moe_cpu_route_ms: f64,
    pub moe_cmd_count: u32,
}

pub const RMS_EPS: f32 = 1e-6;

/// Dispatch the right `encode_mat_vec_*` based on `weight.dtype`. This
/// is the single seam that lets the same MetalForward driver run on
/// F32, Q4_K_M, Q6_K, etc. weights. New quant types plug in here.
pub fn encode_mat_vec_dispatch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MfError> {
    match weight.dtype {
        GgmlType::F32 => Ok(encode_mat_vec_f32(ctx, enc, weight, x, y, n_in, n_out)?),
        GgmlType::F16 => Ok(crate::metal::encode_mat_vec_f16_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::BF16 => Ok(crate::metal::encode_mat_vec_bf16_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q2_K => Ok(crate::metal::encode_mat_vec_q2_k_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q3_K => Ok(crate::metal::encode_mat_vec_q3_k_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::IQ2_S => Ok(crate::metal::encode_mat_vec_iq2_s_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::IQ3_XXS => Ok(crate::metal::encode_mat_vec_iq3_xxs_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::IQ3_S => Ok(crate::metal::encode_mat_vec_iq3_s_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q4_0 => Ok(crate::metal::encode_mat_vec_q4_0_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q4_1 => Ok(crate::metal::encode_mat_vec_q4_1_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q4_K => Ok(encode_mat_vec_q4_k_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q5_K => Ok(encode_mat_vec_q5_k_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q6_K => Ok(encode_mat_vec_q6_k_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q8_0 => Ok(crate::metal::encode_mat_vec_q8_0_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::IQ4_NL => Ok(crate::metal::encode_mat_vec_iq4_nl_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::IQ4_XS => Ok(crate::metal::encode_mat_vec_iq4_xs_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        other => Err(MfError::UnsupportedDtype {
            name: "(weight at mat_vec dispatch)".to_string(),
            dtype: other,
        }),
    }
}

/// Mat-mat dispatch routing for the H5.3b layer-major path. Picks the
/// right `kernel_mul_mm_*` lift based on weight dtype. Output is
/// row-major `[n_query, n_out]` (codex H5.3b mid-impl review verified
/// the col-major framing in the lifted llama kernels is bit-identical
/// to row-major storage at this stride).
///
/// Production 27B Q4_K_M reaches several weight dtypes via mat-mat:
///   * F32/F16/BF16 (full-precision and mixed GGUF variants)
///   * Q2_K/Q3_K (low-bit K-quant compatibility)
///   * Q4_0/Q4_1 (legacy quant compatibility)
///   * Q4_K (ffn_gate, ffn_up, attn projections)
///   * Q5_K (GDN out_proj — added by v0.73a.0)
///   * Q6_K (ffn_down, lm_head)
///   * Q8_0 (DFlash drafter projections, lm_head — added by v0.73b.0)
///   * IQ4_NL/IQ4_XS (IQ quant compatibility)
pub fn encode_mat_mat_dispatch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MfError> {
    match weight.dtype {
        GgmlType::Q4_K => Ok(crate::metal::encode_mat_mat_q4_k_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::F32 => Ok(crate::metal::encode_mat_mat_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::F16 => Ok(crate::metal::encode_mat_mat_f16_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::BF16 if matmat_bf16_bfloat_act_enabled() && n_in % 32 == 0 && n_query >= 16 => {
            Ok(crate::metal::encode_mat_mat_bf16_bfloat_act_f32(
                ctx, enc, weight, x, y, n_in, n_out, n_query,
            )?)
        }
        GgmlType::BF16 => Ok(crate::metal::encode_mat_mat_bf16_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q2_K => Ok(crate::metal::encode_mat_mat_q2_k_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q3_K => Ok(crate::metal::encode_mat_mat_q3_k_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::IQ2_S => Ok(crate::metal::encode_mat_mat_iq2_s_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::IQ3_XXS => Ok(crate::metal::encode_mat_mat_iq3_xxs_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::IQ3_S => Ok(crate::metal::encode_mat_mat_iq3_s_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q4_0 => Ok(crate::metal::encode_mat_mat_q4_0_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q4_1 => Ok(crate::metal::encode_mat_mat_q4_1_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q5_K => Ok(crate::metal::encode_mat_mat_q5_k_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q6_K => Ok(crate::metal::encode_mat_mat_q6_k_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q8_0 => Ok(crate::metal::encode_mat_mat_q8_0_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::IQ4_NL => Ok(crate::metal::encode_mat_mat_iq4_nl_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::IQ4_XS => Ok(crate::metal::encode_mat_mat_iq4_xs_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        other => Err(MfError::UnsupportedDtype {
            name: "(weight at mat_mat dispatch)".to_string(),
            dtype: other,
        }),
    }
}

/// Public single-block GDN dispatcher for end-to-end validation. Encodes
/// one GDN block (norm → mixer → residual → post_norm → FFN → residual)
/// and reads back the resulting `x` (residual stream).
///
/// Mirrors what `single_token` does for one block but skips the global
/// embedding/lm_head, so we can validate one block at a time against the
/// CPU oracle. The GDN block is the most complex piece in the
/// architecture; if this is bit-tight, the rest of the driver is glue.
impl<'a> MetalForward<'a> {
    pub fn run_one_gdn_block_for_test(
        &self,
        block_idx: usize,
        gdn_idx_in_session: usize,
        s: &mut MetalSession,
    ) -> Result<Vec<f32>, MfError> {
        let block = &self.model.blocks[block_idx];
        let gb = match block {
            MetalBlock::Gdn(g) => g,
            MetalBlock::Attn(_) => {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "run_one_gdn_block",
                    detail: format!("block {block_idx} is not a GDN block"),
                }));
            }
        };

        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        // Pre-mixer norm (s.x → s.h).
        encode_rms_norm_mul_f32(self.ctx, &enc, &s.x, &gb.attn_norm, &s.h, RMS_EPS)?;

        // GDN mixer (s.h → s.mixer_out).
        self.encode_gdn(&enc, gb, gdn_idx_in_session, s)?;

        // Residual #1: s.x += s.mixer_out.
        encode_add_inplace_f32(self.ctx, &enc, &s.x, &s.mixer_out)?;

        // Pre-FFN norm (s.x → s.h).
        encode_rms_norm_mul_f32(self.ctx, &enc, &s.x, &gb.post_attn_norm, &s.h, RMS_EPS)?;

        // SwiGLU FFN.
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f = arch.intermediate_size as usize;
        encode_mat_vec_dispatch(self.ctx, &enc, &gb.ffn_gate, &s.h, &s.ffn_gate, h, f)?;
        encode_mat_vec_dispatch(self.ctx, &enc, &gb.ffn_up, &s.h, &s.ffn_up, h, f)?;
        encode_silu_mul_f32(self.ctx, &enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)?;
        encode_mat_vec_dispatch(self.ctx, &enc, &gb.ffn_down, &s.ffn_inner, &s.ffn_out, f, h)?;

        // Residual #2: s.x += s.ffn_out.
        encode_add_inplace_f32(self.ctx, &enc, &s.x, &s.ffn_out)?;

        enc.end();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let mut out = vec![0.0f32; h];
        unsafe {
            let src = s.x.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), h);
        }
        Ok(out)
    }

    /// Test helper: write `data` into `s.x` (the residual stream).
    pub fn set_residual_for_test(&self, s: &mut MetalSession, data: &[f32]) {
        unsafe {
            let dst = s.x.buffer.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }

    /// Test helper: run one full-attn block end-to-end (norm → attn →
    /// residual → post_norm → FFN → residual) and read back the residual
    /// stream. Mirrors `run_one_gdn_block_for_test` for the attn path.
    pub fn run_one_attn_block_for_test(
        &self,
        block_idx: usize,
        attn_idx_in_session: usize,
        position: u32,
        s: &mut MetalSession,
    ) -> Result<Vec<f32>, MfError> {
        let block = &self.model.blocks[block_idx];
        let ab = match block {
            MetalBlock::Attn(a) => a,
            MetalBlock::Gdn(_) => {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "run_one_attn_block",
                    detail: format!("block {block_idx} is not an attn block"),
                }));
            }
        };

        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        encode_rms_norm_mul_f32(self.ctx, &enc, &s.x, &ab.attn_norm, &s.h, RMS_EPS)?;
        self.encode_attn(&enc, ab, attn_idx_in_session, position, s)?;
        encode_add_inplace_f32(self.ctx, &enc, &s.x, &s.mixer_out)?;

        encode_rms_norm_mul_f32(self.ctx, &enc, &s.x, &ab.post_attn_norm, &s.h, RMS_EPS)?;

        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f = arch.intermediate_size as usize;
        encode_mat_vec_dispatch(self.ctx, &enc, &ab.ffn_gate, &s.h, &s.ffn_gate, h, f)?;
        encode_mat_vec_dispatch(self.ctx, &enc, &ab.ffn_up, &s.h, &s.ffn_up, h, f)?;
        encode_silu_mul_f32(self.ctx, &enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)?;
        encode_mat_vec_dispatch(self.ctx, &enc, &ab.ffn_down, &s.ffn_inner, &s.ffn_out, f, h)?;

        encode_add_inplace_f32(self.ctx, &enc, &s.x, &s.ffn_out)?;

        enc.end();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let mut out = vec![0.0f32; h];
        unsafe {
            let src = s.x.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), h);
        }
        Ok(out)
    }
}

// ============================================================================
// Prefix-cache snapshots (H2): packed-arena state capture for warm-restart.
// ============================================================================
//
// Per the H2 design (validated bit-exact in v0.34's correctness spike):
// snapshot the full per-sequence state at a token prefix boundary so a
// later request sharing that prefix can restore-and-resume instead of
// cold-prefilling.
//
// Storage layout (CPU-side `Vec<u8>` arenas, NOT Metal-managed buffers):
//   * Four packed arenas: kv_k, kv_v, gdn_conv, gdn_state.
//   * One CPU vec each, contiguous across layers (NOT 4*n_layers small
//     allocations -- per codex review, that's a long-term mistake).
//   * Final logits stored for the exact-hit case (request == cached
//     prefix exactly).
//
// Lifecycle:
//   * `MetalSession::snapshot(ctx, identity, prefix_tokens)` builds
//     a `SessionSnapshot` by reading the live MTLBuffer.contents() of
//     each session field via raw memcpy. Shared-storage UMA makes this
//     safe and fast (no command-buffer round-trip); Apple docs
//     guarantee the producer's writes are visible after that command
//     buffer completes.
//   * `MetalSession::restore_from(snap)` validates identity matches,
//     then memcpys arena bytes back into session buffers and copies
//     `kv_n_pos`. Subsequent forward passes see the restored state.
//
// Why CPU-side `Vec<u8>` instead of `MetalTensor` arenas (which would
// also be shared-storage on UMA): LRU eviction and cross-session
// disk persistence get cleaner; system memory pressure handler sees
// the cost; Metal's buffer pool stays uncluttered.

/// Identity tag for a snapshot. Checked at restore to refuse silent
/// corruption from model drift / dtype change / layout-version bump.
///
/// `model_id` is content-addressable (typically a hash of the GGUF
/// metadata + tensor descriptor table). `layout_version` is a manual
/// counter bumped whenever the `MetalSession` field layout changes
/// in a way that would invalidate prior snapshots.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SnapshotIdentity {
    pub model_id: u64,
    pub tokenizer_id: u64,
    pub layout_version: u32,
    pub n_attn_layers: u32,
    pub n_gdn_layers: u32,
    pub kv_dim_elements: u32,
    pub kv_bytes_per_token: u32,
    pub gdn_state_elements_per_layer: u32,
    pub gdn_conv_elements_per_layer: u32,
}

/// Bump this when MetalSession's per-layer state shape changes.
pub const SNAPSHOT_LAYOUT_VERSION: u32 = 2;

/// Captured state at the end of prefilling `prefix_tokens` through a
/// fresh session. Restoring into a fresh session and running additional
/// tokens is bit-equivalent to cold prefill of the full sequence
/// (validated by `h2_prefix_cache_correctness_spike`).
#[derive(Clone, Debug)]
pub struct SessionSnapshot {
    pub identity: SnapshotIdentity,
    /// Tokens consumed up to the snapshot boundary. Used as cache key.
    pub prefix_tokens: Vec<i32>,
    /// `kv_n_pos[attn_layer]` after prefill. Same value across layers
    /// for our forward pass (single-stream); kept per-layer for safety.
    pub kv_n_pos: Vec<usize>,
    /// Packed K cache slice: `n_attn × prefix_len × kv_bytes_per_token` bytes.
    /// Sized exactly to the prefix; doesn't carry the unused tail of the
    /// session's full-capacity KV buffer.
    pub kv_k_arena: Vec<u8>,
    /// Packed V cache slice (same shape).
    pub kv_v_arena: Vec<u8>,
    /// Packed GDN conv buffers: `n_gdn × gdn_conv_elements_per_layer × 4 bytes` (F32).
    pub gdn_conv_arena: Vec<u8>,
    /// Packed GDN recurrent state: `n_gdn × gdn_state_elements_per_layer × 4 bytes` (F32).
    pub gdn_state_arena: Vec<u8>,
    /// Logits at the last prefix token (vocab_size F32). Lets a
    /// subsequent exact-hit (request == cached prefix) sample directly
    /// without a forward pass. None if not stored at snapshot time.
    pub final_logits: Option<Vec<f32>>,
}

impl SessionSnapshot {
    /// Total in-memory cost of this snapshot, in bytes.
    pub fn n_bytes(&self) -> u64 {
        (self.kv_k_arena.len()
            + self.kv_v_arena.len()
            + self.gdn_conv_arena.len()
            + self.gdn_state_arena.len()
            + self.final_logits.as_ref().map_or(0, |v| v.len() * 4)
            + self.prefix_tokens.len() * 4
            + self.kv_n_pos.len() * 8) as u64
    }

    /// Number of tokens consumed up to this snapshot.
    pub fn prefix_len(&self) -> usize {
        self.prefix_tokens.len()
    }
}

/// Copy raw bytes FROM a shared-storage MetalTensor's MTLBuffer INTO an
/// existing destination slice (single memcpy, no intermediate alloc).
/// Caller must ensure any prior GPU write is complete (i.e., the command
/// buffer that wrote this tensor was committed AND waitUntilCompleted'd).
fn read_tensor_into(dst: &mut [u8], t: &MetalTensor) {
    // Always-on bounds check: this guards an unchecked memcpy. A debug-only
    // assert would let release builds silently read past the MTLBuffer end.
    let end = (dst.len() as u64)
        .checked_add(t.offset)
        .expect("read offset+len overflow");
    assert!(
        end <= t.buffer.length() as u64,
        "read OOB: offset={} + n={} > buffer.len={}",
        t.offset,
        dst.len(),
        t.buffer.length()
    );
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize);
        std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
    }
}

/// Write raw bytes into a shared-storage MetalTensor's MTLBuffer at offset.
/// Caller must ensure any in-flight GPU read of this tensor has completed
/// before calling. Subsequent GPU work will see the written bytes.
fn write_tensor_bytes(t: &MetalTensor, bytes: &[u8]) {
    // Always-on bounds check: see read_tensor_into above.
    let end = (bytes.len() as u64)
        .checked_add(t.offset)
        .expect("write offset+len overflow");
    assert!(
        end <= t.buffer.length() as u64,
        "write OOB: offset={} + n={} > buffer.len={}",
        t.offset,
        bytes.len(),
        t.buffer.length()
    );
    unsafe {
        let dst = (t.buffer.contents().as_ptr() as *mut u8).add(t.offset as usize);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
    }
}

impl MetalSession {
    /// Compute the identity tag for snapshots produced by this session
    /// shape under the given model. Stable across runs of the same
    /// (model, tokenizer, layout) tuple.
    pub fn snapshot_identity(&self, model_id: u64, tokenizer_id: u64) -> SnapshotIdentity {
        SnapshotIdentity {
            model_id,
            tokenizer_id,
            layout_version: SNAPSHOT_LAYOUT_VERSION,
            n_attn_layers: self.kv_k.len() as u32,
            n_gdn_layers: self.gdn_state.len() as u32,
            kv_dim_elements: (self.kv_k.first().map(|t| t.n_elements()).unwrap_or(0)
                / self.kv_capacity as u64) as u32,
            kv_bytes_per_token: self
                .kv_k
                .first()
                .map(|t| t.n_bytes() / self.kv_capacity as u64)
                .unwrap_or(0) as u32,
            gdn_state_elements_per_layer: self
                .gdn_state
                .first()
                .map(|t| t.n_elements())
                .unwrap_or(0) as u32,
            gdn_conv_elements_per_layer: self.gdn_conv.first().map(|t| t.n_elements()).unwrap_or(0)
                as u32,
        }
    }

    /// Build a `SessionSnapshot` from the current session state.
    ///
    /// `prefix_tokens` is the token sequence that was just consumed
    /// (used as the cache key). `final_logits` is the logits at the
    /// last consumed token (stored for exact-hit lookups; pass `None`
    /// to skip).
    ///
    /// The caller must ensure all prior forward-pass work on this
    /// session has completed (i.e., the last `single_token` call has
    /// returned, which implies its command buffer was waited on).
    /// Reads bytes from MTLBuffer.contents() directly via memcpy --
    /// safe on shared-storage UMA per Apple docs once writes are
    /// scheduled and complete.
    pub fn snapshot(
        &self,
        identity: SnapshotIdentity,
        prefix_tokens: Vec<i32>,
        final_logits: Option<Vec<f32>>,
    ) -> SessionSnapshot {
        let prefix_len = prefix_tokens.len();
        let n_attn = self.kv_k.len();
        let n_gdn = self.gdn_state.len();
        // KV: per-layer slice is exactly prefix_len rows of the active KV dtype.
        let kv_slice_bytes = prefix_len * identity.kv_bytes_per_token as usize;
        let kv_arena_bytes = n_attn * kv_slice_bytes;

        // Allocate each arena UNINITIALIZED (no zero-init), then memcpy
        // directly from MTLBuffer.contents() into slices. ONE write per byte.
        // Zero-init costs almost as much as the actual copy at this scale
        // (158 MB of writes), so skipping it ~halves wall time.
        // Safety: the entire allocation is overwritten by read_tensor_into
        // before any read; no uninitialized bytes ever escape.
        let mut kv_k_arena: Vec<u8> = Vec::with_capacity(kv_arena_bytes);
        let mut kv_v_arena: Vec<u8> = Vec::with_capacity(kv_arena_bytes);
        // SAFETY: capacity is exactly arena_bytes; we will fully overwrite
        // before any read; u8 has no Drop and no validity invariants.
        unsafe {
            kv_k_arena.set_len(kv_arena_bytes);
            kv_v_arena.set_len(kv_arena_bytes);
        }
        for i in 0..n_attn {
            let off = i * kv_slice_bytes;
            read_tensor_into(&mut kv_k_arena[off..off + kv_slice_bytes], &self.kv_k[i]);
            read_tensor_into(&mut kv_v_arena[off..off + kv_slice_bytes], &self.kv_v[i]);
        }

        // GDN: each layer's full buffer (size doesn't depend on prefix_len).
        let gdn_conv_per = (identity.gdn_conv_elements_per_layer as usize) * 4;
        let gdn_state_per = (identity.gdn_state_elements_per_layer as usize) * 4;
        let gdn_conv_total = n_gdn * gdn_conv_per;
        let gdn_state_total = n_gdn * gdn_state_per;
        let mut gdn_conv_arena: Vec<u8> = Vec::with_capacity(gdn_conv_total);
        let mut gdn_state_arena: Vec<u8> = Vec::with_capacity(gdn_state_total);
        // SAFETY: same as above.
        unsafe {
            gdn_conv_arena.set_len(gdn_conv_total);
            gdn_state_arena.set_len(gdn_state_total);
        }
        for i in 0..n_gdn {
            let off_c = i * gdn_conv_per;
            let off_s = i * gdn_state_per;
            read_tensor_into(
                &mut gdn_conv_arena[off_c..off_c + gdn_conv_per],
                &self.gdn_conv[i],
            );
            read_tensor_into(
                &mut gdn_state_arena[off_s..off_s + gdn_state_per],
                &self.gdn_state[i],
            );
        }

        SessionSnapshot {
            identity,
            prefix_tokens,
            kv_n_pos: self.kv_n_pos.clone(),
            kv_k_arena,
            kv_v_arena,
            gdn_conv_arena,
            gdn_state_arena,
            final_logits,
        }
    }

    /// Restore a session to the state captured in `snap`. The session
    /// MUST have been freshly created with the same architecture as
    /// the one that produced `snap` (validated by identity check).
    /// Returns Err on identity mismatch (to avoid silent corruption).
    ///
    /// The caller must ensure no in-flight GPU work is reading these
    /// session buffers (i.e., this should be called after
    /// `MetalSession::fresh` and before the first `single_token`).
    pub fn restore_from(&mut self, snap: &SessionSnapshot) -> Result<(), MfError> {
        let want = self.snapshot_identity(snap.identity.model_id, snap.identity.tokenizer_id);
        if want != snap.identity {
            return Err(MfError::Metal(crate::metal::MetalError::BadShape {
                kernel: "snapshot_restore",
                detail: format!(
                    "identity mismatch: snapshot={:?}, session-shape={:?}",
                    snap.identity, want
                ),
            }));
        }
        let n_attn = self.kv_k.len();
        let n_gdn = self.gdn_state.len();
        let prefix_len = snap.prefix_len();
        let kv_slice_bytes = prefix_len * snap.identity.kv_bytes_per_token as usize;

        for i in 0..n_attn {
            let off = i * kv_slice_bytes;
            write_tensor_bytes(&self.kv_k[i], &snap.kv_k_arena[off..off + kv_slice_bytes]);
            write_tensor_bytes(&self.kv_v[i], &snap.kv_v_arena[off..off + kv_slice_bytes]);
        }
        self.kv_n_pos.copy_from_slice(&snap.kv_n_pos);

        let gdn_conv_per = (snap.identity.gdn_conv_elements_per_layer as usize) * 4;
        let gdn_state_per = (snap.identity.gdn_state_elements_per_layer as usize) * 4;
        for i in 0..n_gdn {
            let off_c = i * gdn_conv_per;
            let off_s = i * gdn_state_per;
            write_tensor_bytes(
                &self.gdn_conv[i],
                &snap.gdn_conv_arena[off_c..off_c + gdn_conv_per],
            );
            write_tensor_bytes(
                &self.gdn_state[i],
                &snap.gdn_state_arena[off_s..off_s + gdn_state_per],
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::Forward;
    use crate::gguf::GgufFile;
    use crate::loader::Model;

    #[test]
    fn checked_u64_div_exact_rejects_zero_or_remainder() {
        assert!(checked_u64_div_exact(12, 3, "ok").is_ok());
        assert!(checked_u64_div_exact(12, 0, "zero").is_err());
        assert!(checked_u64_div_exact(13, 3, "remainder").is_err());
    }

    fn argmax_i32_local(xs: &[f32]) -> i32 {
        xs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i32)
            .unwrap_or(0)
    }

    fn run_argmax_chain_equivalence(model_path: &str, label: &str, cos_floor: f64) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[argmax-chain-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let mut ids = tok
            .encode("The quick brown fox jumps over the lazy dog", false)
            .expect("tokenize");
        ids.truncate(ids.len().min(4));
        assert!(!ids.is_empty(), "tokenizer returned empty prompt");

        let mf = MetalForward::new(&ctx, &mm);
        let cap = ids.len() + 8;
        let mut s_full = MetalSession::fresh(&ctx, &mm, cap).expect("full session");
        let mut s_arg = MetalSession::fresh(&ctx, &mm, cap).expect("arg session");
        let mut last_full = Vec::new();

        for (i, &tid) in ids.iter().enumerate() {
            last_full = mf
                .single_token(tid, i as u32, &mut s_full)
                .expect("full forward");
            let arg = mf
                .single_token_argmax(tid, i as u32, &mut s_arg)
                .expect("argmax forward");
            assert_eq!(
                arg,
                argmax_i32_local(&last_full),
                "[argmax-chain-{label}] prompt-step argmax mismatch at position {i}"
            );
        }

        let next_tok = argmax_i32_local(&last_full);
        let logits_full = mf
            .single_token(next_tok, ids.len() as u32, &mut s_full)
            .expect("full follow-up");
        let logits_arg = mf
            .single_token(next_tok, ids.len() as u32, &mut s_arg)
            .expect("argmax follow-up");

        let mut max_abs = 0.0f32;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (a, b) in logits_full.iter().zip(logits_arg.iter()) {
            max_abs = max_abs.max((a - b).abs());
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64).powi(2);
            nb += (*b as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        let arg_full = argmax_i32_local(&logits_full);
        let arg_arg = argmax_i32_local(&logits_arg);
        eprintln!(
            "[argmax-chain-{label}] cos={cos:.6} max|Δ|={max_abs:.4} argmax full={arg_full} argmax-path={arg_arg}"
        );
        assert_eq!(
            arg_full, arg_arg,
            "[argmax-chain-{label}] follow-up argmax mismatch"
        );
        assert!(
            cos > cos_floor,
            "[argmax-chain-{label}] cos={cos} below floor {cos_floor}"
        );
    }

    fn run_concurrent_gdn_moe_equivalence(
        model_path: &str,
        label: &str,
        max_prompt_tokens: usize,
        cos_floor: f64,
    ) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[moe-concurrent-gdn-{label}] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        assert_eq!(m.arch.kind, ArchKind::Moe);
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let mut ids = tok
            .encode("The quick brown fox jumps over the lazy dog", false)
            .expect("tokenize");
        ids.truncate(ids.len().min(max_prompt_tokens));
        assert!(!ids.is_empty(), "tokenizer returned empty prompt");

        let mf = MetalForward::new(&ctx, &mm);
        let cap = ids.len() + 8;
        let mut s_serial = MetalSession::fresh(&ctx, &mm, cap).expect("serial session");
        let mut s_conc = MetalSession::fresh(&ctx, &mm, cap).expect("concurrent session");
        let mut last_serial = Vec::new();

        let compare = |lhs: &[f32], rhs: &[f32], where_label: &str| {
            let mut max_abs = 0.0f32;
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for i in 0..lhs.len() {
                max_abs = max_abs.max((lhs[i] - rhs[i]).abs());
                dot += lhs[i] as f64 * rhs[i] as f64;
                na += (lhs[i] as f64).powi(2);
                nb += (rhs[i] as f64).powi(2);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
            let arg_l = argmax_i32_local(lhs);
            let arg_r = argmax_i32_local(rhs);
            eprintln!(
                "[moe-concurrent-gdn-{label}] {where_label}: argmax serial={arg_l} conc={arg_r} max|Δ|={max_abs:.4} cos={cos:.6}"
            );
            assert_eq!(arg_l, arg_r, "[{where_label}] argmax disagreement");
            assert!(
                cos >= cos_floor,
                "[{where_label}] cos={cos} below floor {cos_floor}"
            );
        };

        for (i, &tid) in ids.iter().enumerate() {
            let (serial, _) = mf
                .single_token_profiled_moe(tid, i as u32, &mut s_serial)
                .expect("serial prompt step");
            let (concurrent, _) = mf
                .single_token_profiled_concurrent_gdn_moe(tid, i as u32, &mut s_conc)
                .expect("concurrent prompt step");
            compare(&serial, &concurrent, &format!("prompt-step-{i}"));
            last_serial = serial;
        }

        let next_tok = argmax_i32_local(&last_serial);
        let (serial, _) = mf
            .single_token_profiled_moe(next_tok, ids.len() as u32, &mut s_serial)
            .expect("serial follow-up");
        let (concurrent, _) = mf
            .single_token_profiled_concurrent_gdn_moe(next_tok, ids.len() as u32, &mut s_conc)
            .expect("concurrent follow-up");
        compare(&serial, &concurrent, "follow-up");
    }

    /// Validate a single GDN block end-to-end on Metal vs the CPU oracle.
    /// Uses block 0 of Qwen3.5-0.8B-F32 (the first GDN block, n_v=n_k=16).
    /// Compares the residual stream (s.x) after the block against what
    /// the CPU forward produces after running just block 0.
    #[test]
    fn metal_gdn_block_matches_cpu() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[metal-gdn] skipped — model missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        // Build inputs for block 0:
        //   * residual stream = the embedding row for token "Hello" (id 9419)
        // The CPU oracle and Metal driver both run from this same starting state.
        let token_id = 9419usize;
        let h = m.arch.hidden_size as usize;
        let embed =
            crate::codec::dequant_to_f32(m.token_embd, g.slice(m.token_embd)).expect("embed");
        let initial_x: Vec<f32> = embed[token_id * h..(token_id + 1) * h].to_vec();

        // CPU oracle: run Forward through one block manually.
        // We reuse Forward::single_token but only after the embedding
        // step; easier path is to just replicate the block in CPU code,
        // matching what forward.rs does. But that's a duplication risk.
        // Instead: call the public `Forward::single_token` and intercept
        // by limiting blocks. Forward doesn't expose that, so we carve
        // out a CPU-block helper here that mirrors forward.rs:single_token's
        // inner block loop for block 0 only.
        //
        // For the validation we rely on Forward::single_token computing
        // the same x state (post-block-0) — but it doesn't expose that.
        // Simplest: inline the block 0 computation here using the same
        // primitives. Given block 0 of 0.8B is a GDN block, this reads
        // exactly like the GDN inner of forward.rs.
        let cpu_x_after_block0 = run_cpu_block0_for_test(&g, &m, &initial_x);

        // Metal: build a fresh session, plant initial_x in the residual
        // stream, run block 0.
        let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
        let mf = MetalForward::new(&ctx, &mm);
        mf.set_residual_for_test(&mut s, &initial_x);
        let metal_x = mf
            .run_one_gdn_block_for_test(0, 0, &mut s)
            .expect("metal block 0");

        let max_abs = metal_x
            .iter()
            .zip(cpu_x_after_block0.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = metal_x
            .iter()
            .zip(cpu_x_after_block0.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = metal_x.iter().map(|v| (*v as f64).powi(2)).sum();
        let nb: f64 = cpu_x_after_block0.iter().map(|v| (*v as f64).powi(2)).sum();
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("[metal-gdn-block0] hidden={h} max|Δ|={max_abs:.2e} cos={cos:.6}");
        assert!(max_abs < 1e-3, "block-0 drift {max_abs}");
        assert!(cos > 0.9999, "block-0 cos {cos}");
    }

    /// **End-to-end Metal forward** validated against `llm`/`llama_core`'s
    /// snapshot dump (which uses llama.cpp under the hood and is what
    /// the CPU oracle is also validated against). One token, one
    /// command buffer, all 24 blocks of Qwen3.5-0.8B-F32 chained.
    #[test]
    fn metal_single_token_matches_cpu_oracle() {
        let model_path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        let oracle_path = "/tmp/qwen-oracle/hello_t0.f32";
        if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists()
        {
            eprintln!("[metal-e2e] skipped — fixtures missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
        let n = oracle_bytes.len() / 4;
        let oracle: Vec<f32> = (0..n)
            .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        assert_eq!(oracle.len(), m.arch.vocab_size as usize);

        // Tokenize "Hello" → 9419.
        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok.encode("Hello", false).expect("tokenize");
        eprintln!("[metal-e2e] 'Hello' -> {ids:?}");
        assert_eq!(ids.len(), 1);

        let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
        let mf = MetalForward::new(&ctx, &mm);
        let t = std::time::Instant::now();
        let logits = mf.single_token(ids[0], 0, &mut s).expect("forward");
        let ms = t.elapsed().as_secs_f64() * 1e3;

        let mut max_abs = 0.0f32;
        let mut argmax_ours = 0usize;
        let mut argmax_oracle = 0usize;
        let mut max_ours = f32::NEG_INFINITY;
        let mut max_oracle = f32::NEG_INFINITY;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..n {
            let d = (logits[i] - oracle[i]).abs();
            max_abs = max_abs.max(d);
            if logits[i] > max_ours {
                max_ours = logits[i];
                argmax_ours = i;
            }
            if oracle[i] > max_oracle {
                max_oracle = oracle[i];
                argmax_oracle = i;
            }
            dot += logits[i] as f64 * oracle[i] as f64;
            na += (logits[i] as f64).powi(2);
            nb += (oracle[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[metal-e2e] {ms:.1}ms — argmax: ours={argmax_ours} ({:.4}) | oracle={argmax_oracle} ({:.4}) | max|Δ|={max_abs:.4} cos={cos:.6}",
            max_ours, max_oracle
        );
        assert_eq!(argmax_ours, argmax_oracle, "argmax disagreement");
        assert!(cos > 0.9999, "cos={cos} below threshold");
        assert!(max_abs < 0.05, "max|Δ|={max_abs} above noise floor");
    }

    #[test]
    fn metal_single_token_concurrent_gdn_matches_serial() {
        let model_path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[metal-concurrent-gdn] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok.encode("Hello", false).expect("tokenize");
        assert_eq!(ids.len(), 1);

        let mf = MetalForward::new(&ctx, &mm);
        let mut s_serial = MetalSession::fresh(&ctx, &mm, 256).expect("session-serial");
        let mut s_conc = MetalSession::fresh(&ctx, &mm, 256).expect("session-concurrent");

        let (serial, _) = mf
            .single_token_profiled_dense_serial(ids[0], 0, &mut s_serial)
            .expect("serial");
        let (concurrent, _) = mf
            .single_token_profiled_concurrent_gdn_dense(ids[0], 0, &mut s_conc)
            .expect("concurrent");

        let mut max_abs = 0.0f32;
        let mut argmax_serial = 0usize;
        let mut argmax_conc = 0usize;
        let mut max_serial = f32::NEG_INFINITY;
        let mut max_conc = f32::NEG_INFINITY;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..serial.len() {
            let d = (serial[i] - concurrent[i]).abs();
            max_abs = max_abs.max(d);
            if serial[i] > max_serial {
                max_serial = serial[i];
                argmax_serial = i;
            }
            if concurrent[i] > max_conc {
                max_conc = concurrent[i];
                argmax_conc = i;
            }
            dot += serial[i] as f64 * concurrent[i] as f64;
            na += (serial[i] as f64).powi(2);
            nb += (concurrent[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[metal-concurrent-gdn] argmax serial={argmax_serial} conc={argmax_conc} max|Δ|={max_abs:.4} cos={cos:.6}"
        );
        assert_eq!(argmax_serial, argmax_conc, "argmax disagreement");
        assert!(cos > 0.9999, "cos={cos} below threshold");
        assert!(max_abs < 0.05, "max|Δ|={max_abs} above noise floor");

        let mut s_serial_argmax =
            MetalSession::fresh(&ctx, &mm, 256).expect("session-serial-argmax");
        let mut s_conc_argmax =
            MetalSession::fresh(&ctx, &mm, 256).expect("session-concurrent-argmax");
        let (serial_argmax, _) = mf
            .single_token_argmax_profiled_dense_serial(ids[0], 0, &mut s_serial_argmax)
            .expect("serial argmax");
        let (concurrent_argmax, _) = mf
            .single_token_argmax_profiled_concurrent_gdn_dense(ids[0], 0, &mut s_conc_argmax)
            .expect("concurrent argmax");
        assert_eq!(serial_argmax, concurrent_argmax, "GPU argmax disagreement");
    }

    #[test]
    fn metal_single_token_concurrent_gdn_attn_matches_serial() {
        let model_path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[metal-concurrent-gdn-attn] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok.encode("Hello", false).expect("tokenize");
        assert_eq!(ids.len(), 1);

        let mf = MetalForward::new(&ctx, &mm);
        let mut s_serial = MetalSession::fresh(&ctx, &mm, 256).expect("session-serial");
        let mut s_conc = MetalSession::fresh(&ctx, &mm, 256).expect("session-concurrent");

        let (serial, _) = mf
            .single_token_profiled_dense_serial(ids[0], 0, &mut s_serial)
            .expect("serial");
        let (concurrent, _) = mf
            .single_token_profiled_concurrent_gdn_attn_dense(ids[0], 0, &mut s_conc)
            .expect("concurrent");

        let mut max_abs = 0.0f32;
        let mut argmax_serial = 0usize;
        let mut argmax_conc = 0usize;
        let mut max_serial = f32::NEG_INFINITY;
        let mut max_conc = f32::NEG_INFINITY;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..serial.len() {
            let d = (serial[i] - concurrent[i]).abs();
            max_abs = max_abs.max(d);
            if serial[i] > max_serial {
                max_serial = serial[i];
                argmax_serial = i;
            }
            if concurrent[i] > max_conc {
                max_conc = concurrent[i];
                argmax_conc = i;
            }
            dot += serial[i] as f64 * concurrent[i] as f64;
            na += (serial[i] as f64).powi(2);
            nb += (concurrent[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[metal-concurrent-gdn-attn] argmax serial={argmax_serial} conc={argmax_conc} max|Δ|={max_abs:.4} cos={cos:.6}"
        );
        assert_eq!(argmax_serial, argmax_conc, "argmax disagreement");
        assert!(cos > 0.9999, "cos={cos} below threshold");
        assert!(max_abs < 0.05, "max|Δ|={max_abs} above noise floor");
    }

    #[test]
    fn metal_single_token_concurrent_gdn_moe_matches_serial_a3b() {
        run_concurrent_gdn_moe_equivalence(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            4,
            0.995,
        );
    }

    #[test]
    fn metal_single_token_concurrent_gdn_moe_matches_serial_a10b_smoke() {
        run_concurrent_gdn_moe_equivalence(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",
            "a10b",
            2,
            0.995,
        );
    }

    #[test]
    fn metal_35b_a3b_moe_matches_cpu_smoke() {
        let model_path = "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[metal-moe-a3b] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        assert_eq!(m.arch.kind, ArchKind::Moe);
        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok.encode("Hello", false).expect("tokenize");
        assert_eq!(ids.len(), 1);

        let f = Forward::new(&g, &m);
        let mut cpu_state = crate::forward::GdnState::fresh(&m);
        let mut cpu_kv = crate::forward::KvCache::with_capacity(&m, 8);
        let cpu = f
            .single_token(ids[0], 0, &mut cpu_state, &mut cpu_kv)
            .expect("cpu forward");

        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let mut s = MetalSession::fresh(&ctx, &mm, 8).expect("session");
        let (metal, prof) = mf
            .single_token_profiled(ids[0], 0, &mut s)
            .expect("metal forward");

        let mut argmax_cpu = 0usize;
        let mut argmax_metal = 0usize;
        let mut max_cpu = f32::NEG_INFINITY;
        let mut max_metal = f32::NEG_INFINITY;
        let mut max_abs = 0.0f32;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..cpu.len() {
            let d = (metal[i] - cpu[i]).abs();
            max_abs = max_abs.max(d);
            if cpu[i] > max_cpu {
                max_cpu = cpu[i];
                argmax_cpu = i;
            }
            if metal[i] > max_metal {
                max_metal = metal[i];
                argmax_metal = i;
            }
            dot += metal[i] as f64 * cpu[i] as f64;
            na += (metal[i] as f64).powi(2);
            nb += (cpu[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[metal-moe-a3b] total={:.2}ms gpu={:.2}ms cmd_bufs={} argmax metal={argmax_metal}({max_metal:.4}) cpu={argmax_cpu}({max_cpu:.4}) max|Δ|={max_abs:.4} cos={cos:.6}",
            prof.total_ms, prof.gpu_kernel_ms, prof.moe_cmd_count
        );
        assert_eq!(argmax_metal, argmax_cpu, "argmax disagreement");
        assert!(cos > 0.995, "cos={cos} below threshold");
    }

    #[test]
    fn metal_argmax_chain_matches_full_logits_dense() {
        run_argmax_chain_equivalence(
            "/Users/tito/models/Qwen3.5-0.8B.F32.gguf",
            "dense-0p8b",
            0.9999,
        );
    }

    #[test]
    fn metal_argmax_chain_matches_full_logits_moe() {
        run_argmax_chain_equivalence(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "moe-a3b",
            0.995,
        );
    }

    /// **H5.2** — Metal multi-layer hidden capture matches CPU oracle.
    /// Reads hiddens at K layer indices via Metal in one command buffer,
    /// then captures the same hiddens via the CPU
    /// `single_token_capture_layers` reference. Cosine ≥ 0.9999 per
    /// captured layer (Q4_K_M + Q6_K weights through F32 norms +
    /// elementwise residual chain — same noise floor as the single-token
    /// oracle test above).
    ///
    /// Skipped on the F32 0.8B model since its layer count (24) does
    /// fit `target_layer_ids = [1, 16, 31, 46, 61]` which is a 27B-style
    /// list. We use [1, 5, 10, 15, 23] for the 0.8B variant — different
    /// indices, same shape K=5.
    #[test]
    fn metal_multi_hidden_matches_cpu() {
        let model_path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[metal-multi-hidden] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        let h = m.arch.hidden_size as usize;
        let target_layer_ids: Vec<u32> = vec![1, 5, 10, 15, 23];
        let k = target_layer_ids.len();
        let token_id = 9419i32; // "Hello"
        let position = 0u32;

        // Metal capture.
        let mf = MetalForward::new(&ctx, &mm);
        let mut sm = MetalSession::fresh(&ctx, &mm, 256).expect("session-m");
        let hidden_dst = MetalTensor::zeros_f32(&ctx, vec![(k * h) as u64]).expect("hidden_dst");
        let _logits_metal = mf
            .single_token_with_multi_hidden(
                token_id,
                position,
                &mut sm,
                &target_layer_ids,
                &hidden_dst,
            )
            .expect("metal multi-hidden");
        // Read back hidden_dst.
        let mut metal_hidden = vec![0.0f32; k * h];
        unsafe {
            let src = hidden_dst.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, metal_hidden.as_mut_ptr(), metal_hidden.len());
        }

        // CPU capture via the existing oracle.
        let cpu = crate::forward::Forward::new(&g, &m);
        let mut cpu_state = crate::forward::GdnState::fresh(&m);
        let mut cpu_kv = crate::forward::KvCache::new(&m);
        let cpu_hidden = cpu
            .single_token_capture_layers(
                token_id,
                position,
                &mut cpu_state,
                &mut cpu_kv,
                &target_layer_ids,
            )
            .expect("cpu multi-hidden");
        assert_eq!(metal_hidden.len(), cpu_hidden.len());

        // Compare per-layer cosine + max|Δ|.
        for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
            let off = k_idx * h;
            let mh = &metal_hidden[off..off + h];
            let ch = &cpu_hidden[off..off + h];
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            let mut max_abs = 0.0f32;
            for i in 0..h {
                dot += mh[i] as f64 * ch[i] as f64;
                na += (mh[i] as f64).powi(2);
                nb += (ch[i] as f64).powi(2);
                max_abs = max_abs.max((mh[i] - ch[i]).abs());
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
            eprintln!(
                "[metal-multi-hidden] k={k_idx} (layer {lid}): cos={cos:.6} max|Δ|={max_abs:.4}"
            );
            assert!(
                cos > 0.9999,
                "k={k_idx} layer {lid}: cos {cos} below threshold"
            );
            assert!(
                mh.iter().all(|x| x.is_finite()),
                "k={k_idx} layer {lid}: NaN/Inf in metal hidden"
            );
            // 0.8B-F32 has effectively zero quant noise; bound tight.
            assert!(
                max_abs < 1e-2,
                "k={k_idx} layer {lid}: max|Δ| {max_abs} above F32 noise floor"
            );
        }

        // Sanity: hiddens at different layers must NOT be identical
        // (catches a layout bug where we'd accidentally write the same
        // layer's residual into all K slots).
        for k_idx in 0..k - 1 {
            let a = &metal_hidden[k_idx * h..(k_idx + 1) * h];
            let b = &metal_hidden[(k_idx + 1) * h..(k_idx + 2) * h];
            let identical = a.iter().zip(b.iter()).all(|(x, y)| x == y);
            assert!(
                !identical,
                "captured hiddens at k={k_idx} and k={} are identical — multi-hidden layout bug",
                k_idx + 1
            );
        }
    }

    /// Same as `metal_gdn_block_matches_cpu` but for 27B-Q4_K_M block 0.
    /// Tests the dispatch-by-dtype path on real Q4_K + Q6_K weights.
    #[test]
    #[ignore]
    fn metal_27b_gdn_block0_matches_cpu() {
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        // Same harness as metal_gdn_block_matches_cpu, but with token id
        // and arch dims pulled from 27B.
        let token_id = 9419usize; // "Hello"
        let h = m.arch.hidden_size as usize;
        let embed =
            crate::codec::dequant_to_f32(m.token_embd, g.slice(m.token_embd)).expect("embed");
        let initial_x: Vec<f32> = embed[token_id * h..(token_id + 1) * h].to_vec();

        let cpu_x = run_cpu_block0_for_test(&g, &m, &initial_x);

        let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
        let mf = MetalForward::new(&ctx, &mm);
        mf.set_residual_for_test(&mut s, &initial_x);
        let metal_x = mf
            .run_one_gdn_block_for_test(0, 0, &mut s)
            .expect("metal block 0");

        let max_abs = metal_x
            .iter()
            .zip(cpu_x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = metal_x
            .iter()
            .zip(cpu_x.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = metal_x.iter().map(|v| (*v as f64).powi(2)).sum();
        let nb: f64 = cpu_x.iter().map(|v| (*v as f64).powi(2)).sum();
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        // Inspect ssm_a values for the first GDN block.
        let block0 = &m.blocks[0];
        let ssm_a_desc = match block0 {
            crate::loader::Block::Gdn(g) => g.a_log,
            _ => panic!("not gdn"),
        };
        let ssm_a = crate::codec::dequant_to_f32(ssm_a_desc, g.slice(ssm_a_desc)).unwrap();
        eprintln!(
            "[metal-27b-gdn0] ssm_a[0..8]={:?}",
            &ssm_a[..8.min(ssm_a.len())]
        );
        eprintln!(
            "[metal-27b-gdn0] ssm_a min={:.4} max={:.4}",
            ssm_a.iter().cloned().fold(f32::INFINITY, f32::min),
            ssm_a.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );

        let nm: f32 = metal_x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nc: f32 = cpu_x.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "[metal-27b-gdn0] hidden={h} ||metal||={nm:.4e} ||cpu||={nc:.4e} max|Δ|={max_abs:.4} cos={cos:.6}"
        );
        eprintln!(
            "[metal-27b-gdn0] metal[0..4]={:?}\n              cpu[0..4]={:?}",
            &metal_x[..4.min(metal_x.len())],
            &cpu_x[..4.min(cpu_x.len())]
        );
        // Q4_K + Q6_K: relax noise floor a bit.
        assert!(cos > 0.999, "27B block 0 cos={cos}");
    }

    /// **End-to-end Metal forward on the 27B Q4_K_M target.** Validates
    /// the quantized weight path: native Q4_K and Q6_K mat-vec kernels
    /// dispatched based on tensor dtype, no per-call dequant.
    ///
    /// This is the test that proves we can run the full production
    /// 27B target on Metal with bit-tight correctness vs llm/llama_core.
    /// Once this passes, we benchmark vs llama-bench.
    ///
    /// Marked #[ignore] because (1) loading 16.8 GB of weights through
    /// the loader takes a few seconds and (2) the codec-fallback path
    /// for unsupported quants (Q5_K, etc.) might dequant some tensors,
    /// and we want to flag that explicitly when run.
    #[test]
    #[ignore]
    fn metal_27b_q4_k_m_matches_oracle() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        let oracle_path = "/tmp/qwen-oracle/hello_27b_q4km.f32";
        if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists()
        {
            eprintln!("[metal-27b] skipped — fixtures missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };

        let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
        let n = oracle_bytes.len() / 4;
        let oracle: Vec<f32> = (0..n)
            .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok.encode("Hello", false).expect("tokenize");
        eprintln!("[metal-27b] 'Hello' -> {ids:?}");

        let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
        let mf = MetalForward::new(&ctx, &mm);

        // Warmup: first call compiles all 20+ kernel pipeline state objects.
        let _ = mf.single_token(ids[0], 0, &mut s).expect("warmup");
        // Reset session for the timed run.
        let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session2");

        let t = std::time::Instant::now();
        let logits = mf.single_token(ids[0], 0, &mut s).expect("forward");
        let ms = t.elapsed().as_secs_f64() * 1e3;

        let mut max_abs = 0.0f32;
        let mut argmax_ours = 0usize;
        let mut argmax_oracle = 0usize;
        let mut max_ours = f32::NEG_INFINITY;
        let mut max_oracle = f32::NEG_INFINITY;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..n {
            let d = (logits[i] - oracle[i]).abs();
            max_abs = max_abs.max(d);
            if logits[i] > max_ours {
                max_ours = logits[i];
                argmax_ours = i;
            }
            if oracle[i] > max_oracle {
                max_oracle = oracle[i];
                argmax_oracle = i;
            }
            dot += logits[i] as f64 * oracle[i] as f64;
            na += (logits[i] as f64).powi(2);
            nb += (oracle[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[metal-27b] {ms:.1}ms — argmax: ours={argmax_ours} ({:.4}) | oracle={argmax_oracle} ({:.4}) | max|Δ|={max_abs:.4} cos={cos:.6}",
            max_ours, max_oracle
        );
        eprintln!(
            "[metal-27b] effective decode tok/s (single-token, single-shot): {:.2}",
            1000.0 / ms
        );
        assert_eq!(argmax_ours, argmax_oracle, "argmax disagreement");
        assert!(cos > 0.999, "cos={cos} below threshold");
    }

    /// **Bench the Q5_K → F32 fallback cost.** Replay all 48 GDN layers'
    /// `ssm_out.weight` mat-vecs in F32 (current state), measure GPU
    /// kernel time. Then estimate the native-Q5_K time as
    /// `f32_time × (q5_bytes / f32_bytes)` and report the delta.
    #[test]
    #[ignore]
    fn metal_27b_q5_fallback_bench() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");

        // Find all blk.*.ssm_out.weight tensors with Q5_K dtype.
        let q5_tensors: Vec<&TensorDesc> = g
            .tensors
            .iter()
            .filter(|t| {
                t.name.ends_with(".ssm_out.weight")
                    && t.dtype == GgmlType::Q5_K
                    && t.shape.len() == 2
            })
            .collect();
        eprintln!("[q5-bench] {} Q5_K ssm_out tensors", q5_tensors.len());
        if q5_tensors.is_empty() {
            return;
        }

        // For each one: current path is dequant-to-F32 then F32 mat_vec.
        // Build the F32 weight buffers, plus a constant input vector and
        // an output buffer. Replay all 48 mat_vecs in one command buffer
        // and time it.
        let n_in = q5_tensors[0].shape[0] as usize; // 6144 for 27B GDN
        let n_out = q5_tensors[0].shape[1] as usize; // 5120
        let total_q5_bytes: u64 = q5_tensors.iter().map(|t| t.n_bytes).sum();
        let total_f32_bytes: u64 = q5_tensors
            .iter()
            .map(|t| (t.shape.iter().product::<u64>()) * 4)
            .sum();

        eprintln!("[q5-bench] shape [{n_in}, {n_out}], 48 layers");
        eprintln!(
            "[q5-bench] total Q5_K bytes: {:.2} MiB",
            total_q5_bytes as f64 / (1024.0 * 1024.0)
        );
        eprintln!(
            "[q5-bench] total F32 bytes:  {:.2} MiB (current resident)",
            total_f32_bytes as f64 / (1024.0 * 1024.0)
        );

        // Dequant all to F32 + upload as MetalTensor.
        let f32_weights: Vec<MetalTensor> = q5_tensors
            .iter()
            .map(|t| {
                let f = crate::codec::dequant_to_f32(t, g.slice(t)).unwrap();
                MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&f),
                    t.shape.clone(),
                    GgmlType::F32,
                )
                .unwrap()
            })
            .collect();

        // Input + output buffers.
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 1e-3).sin()).collect();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .unwrap();
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();

        // Warmup.
        for _ in 0..3 {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for w in &f32_weights {
                crate::metal::encode_mat_vec_f32(&ctx, &enc, w, &x_t, &y_t, n_in, n_out).unwrap();
            }
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        // Timed replay.
        const ITERS: usize = 30;
        let t = std::time::Instant::now();
        let mut gpu_sum_ms = 0.0f64;
        for _ in 0..ITERS {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for w in &f32_weights {
                crate::metal::encode_mat_vec_f32(&ctx, &enc, w, &x_t, &y_t, n_in, n_out).unwrap();
            }
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            gpu_sum_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        }
        let total_ms = t.elapsed().as_secs_f64() * 1e3;
        let per_iter_total = total_ms / ITERS as f64;
        let per_iter_gpu = gpu_sum_ms / ITERS as f64;

        let bytes_per_iter_f32 = total_f32_bytes as f64;
        let bytes_per_iter_q5 = total_q5_bytes as f64;
        let bw_f32 = bytes_per_iter_f32 / (per_iter_gpu / 1000.0) / 1e9;
        let predicted_q5_ms = per_iter_gpu * (bytes_per_iter_q5 / bytes_per_iter_f32);

        eprintln!(
            "[q5-bench] {ITERS} iters: total {per_iter_total:.2} ms/iter, gpu {per_iter_gpu:.2} ms/iter"
        );
        eprintln!("[q5-bench]   F32 BW achieved:    {bw_f32:.0} GB/s");
        eprintln!("[q5-bench]   F32 mat_vec cost (current):  {per_iter_gpu:.2} ms/token");
        eprintln!(
            "[q5-bench]   estimated native Q5_K cost:  {predicted_q5_ms:.2} ms/token  (BW-scaled)"
        );
        eprintln!(
            "[q5-bench]   POTENTIAL SAVINGS:           {:.2} ms/token",
            per_iter_gpu - predicted_q5_ms
        );
        eprintln!(
            "[q5-bench]   we're at 51.26 ms total; saving this would put us at {:.2} ms = {:.2} t/s",
            51.26 - (per_iter_gpu - predicted_q5_ms),
            1000.0 / (51.26 - (per_iter_gpu - predicted_q5_ms))
        );
    }

    /// **Per-tensor byte ledger.** Audit what's actually loaded into Metal
    /// memory vs what came out of the GGUF. Specifically: which tensors
    /// got native dtype, which got dequant-fallback to F32, and how many
    /// bytes per category. Run before optimization to ground decisions.
    #[test]
    #[ignore]
    fn metal_27b_byte_ledger() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");

        // Walk loader::Model and classify each tensor.
        // Loader emits: token_embd, output_norm, lm_head, then per-block.
        // For each tensor, we ask: would MetalModel::load() preserve native
        // or fallback to F32?
        // Match the policy in MetalModel::load:
        //   * load_f32 (ALWAYS dequant to F32): norms, ssm_a, ssm_dt, conv1d,
        //     ssm_norm, q_norm, k_norm, output_norm, token_embd
        //   * load_weight (preserves F32/Q4_K/Q6_K, falls back to F32 for
        //     others): all the mat_vec weights — lm_head, ffn_*, attn_q/k/v/o,
        //     attn_qkv, attn_gate, in_proj_qkv, in_proj_z, beta_proj,
        //     alpha_proj, out_proj
        let mut stats: std::collections::BTreeMap<String, (u64, u64, u64)> =
            std::collections::BTreeMap::new(); // role -> (gguf_bytes, metal_bytes, count)
        let bump = |stats: &mut std::collections::BTreeMap<String, (u64, u64, u64)>,
                    role: &str,
                    gguf_b: u64,
                    metal_b: u64| {
            let e = stats.entry(role.into()).or_insert((0, 0, 0));
            e.0 += gguf_b;
            e.1 += metal_b;
            e.2 += 1;
        };
        let f32_size = |shape: &[u64]| -> u64 { shape.iter().product::<u64>() * 4 };

        // Top-level tensors.
        bump(
            &mut stats,
            "token_embd (load_f32)",
            m.token_embd.n_bytes,
            f32_size(&m.token_embd.shape),
        );
        bump(
            &mut stats,
            "output_norm (load_f32)",
            m.output_norm.n_bytes,
            f32_size(&m.output_norm.shape),
        );
        let lm_head_kept = weight_dtype_kept_native(m.lm_head.dtype);
        bump(
            &mut stats,
            if lm_head_kept {
                "lm_head (native)"
            } else {
                "lm_head (FALLBACK F32)"
            },
            m.lm_head.n_bytes,
            if lm_head_kept {
                m.lm_head.n_bytes
            } else {
                f32_size(&m.lm_head.shape)
            },
        );

        for b in &m.blocks {
            match b {
                crate::loader::Block::Gdn(g) => {
                    let f32_descs: &[&TensorDesc] = &[
                        g.attn_norm,
                        g.post_attention_norm,
                        g.a_log,
                        g.dt_bias,
                        g.conv1d,
                        g.norm,
                    ];
                    for d in f32_descs {
                        bump(
                            &mut stats,
                            "gdn f32-required",
                            d.n_bytes,
                            f32_size(&d.shape),
                        );
                    }
                    let weight_descs: &[(&TensorDesc, &str)] = &[
                        (g.in_proj_qkv, "gdn in_proj_qkv"),
                        (g.in_proj_z, "gdn in_proj_z"),
                        (g.beta_proj, "gdn beta_proj"),
                        (g.alpha_proj, "gdn alpha_proj"),
                        (g.out_proj, "gdn out_proj"),
                        (g.ffn_gate, "gdn ffn_gate"),
                        (g.ffn_up, "gdn ffn_up"),
                        (g.ffn_down, "gdn ffn_down"),
                    ];
                    for (d, role) in weight_descs {
                        let kept = weight_dtype_kept_native(d.dtype);
                        let key = format!(
                            "{role} ({:?}{})",
                            d.dtype,
                            if kept { "" } else { " FALLBACK→F32" }
                        );
                        bump(
                            &mut stats,
                            &key,
                            d.n_bytes,
                            if kept { d.n_bytes } else { f32_size(&d.shape) },
                        );
                    }
                }
                crate::loader::Block::Attn(a) => {
                    let f32_descs: &[&TensorDesc] =
                        &[a.attn_norm, a.post_attention_norm, a.q_norm, a.k_norm];
                    for d in f32_descs {
                        bump(
                            &mut stats,
                            "attn f32-required",
                            d.n_bytes,
                            f32_size(&d.shape),
                        );
                    }
                    let weight_descs: &[(&TensorDesc, &str)] = &[
                        (a.q, "attn q"),
                        (a.k, "attn k"),
                        (a.v, "attn v"),
                        (a.o, "attn o"),
                        (a.ffn_gate, "attn ffn_gate"),
                        (a.ffn_up, "attn ffn_up"),
                        (a.ffn_down, "attn ffn_down"),
                    ];
                    for (d, role) in weight_descs {
                        let kept = weight_dtype_kept_native(d.dtype);
                        let key = format!(
                            "{role} ({:?}{})",
                            d.dtype,
                            if kept { "" } else { " FALLBACK→F32" }
                        );
                        bump(
                            &mut stats,
                            &key,
                            d.n_bytes,
                            if kept { d.n_bytes } else { f32_size(&d.shape) },
                        );
                    }
                }
            }
        }

        eprintln!("[ledger] role  count  gguf_MB  metal_MB  delta_MB");
        let mut total_gguf = 0u64;
        let mut total_metal = 0u64;
        for (role, (gguf_b, metal_b, count)) in &stats {
            let dg = *gguf_b as f64 / (1024.0 * 1024.0);
            let dm = *metal_b as f64 / (1024.0 * 1024.0);
            let delta = dm - dg;
            eprintln!(
                "[ledger]   {role:60} {count:4}  {dg:8.2}  {dm:8.2}  {:+.2}",
                delta
            );
            total_gguf += gguf_b;
            total_metal += metal_b;
        }
        let total_gguf_gb = total_gguf as f64 / (1024.0 * 1024.0 * 1024.0);
        let total_metal_gb = total_metal as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!("[ledger] === TOTALS ===");
        eprintln!("[ledger]   gguf  bytes: {total_gguf_gb:.2} GiB");
        eprintln!("[ledger]   metal bytes: {total_metal_gb:.2} GiB");
        eprintln!(
            "[ledger]   inflation:    {:+.2} GiB ({:+.1}% from quant fallbacks)",
            total_metal_gb - total_gguf_gb,
            (total_metal_gb / total_gguf_gb - 1.0) * 100.0
        );
        let bw_floor_native = total_gguf_gb * 1024.0 / 546.0; // ms at peak BW (note: GiB->GB unit fudge but consistent)
        let bw_floor_metal = total_metal_gb * 1024.0 / 546.0;
        eprintln!("[ledger]   bandwidth floor at GGUF native bytes: {bw_floor_native:.2} ms");
        eprintln!("[ledger]   bandwidth floor at Metal bytes:       {bw_floor_metal:.2} ms");
        eprintln!(
            "[ledger]   estimated cost of fallbacks: {:+.2} ms",
            bw_floor_metal - bw_floor_native
        );
    }

    /// **MTP tensor inventory**: scan the 27B GGUF for `mtp.*` tensors
    /// to see what speculative-decoding state is shipped in the file.
    /// Per the Qwen3.5/3.6 spec, the MTP head is a single decoder layer
    /// with shared `embed_tokens` + `lm_head`. The released checkpoint
    /// includes the trained MTP weights even though HF transformers
    /// ignores them.
    #[test]
    #[ignore]
    fn mtp_tensor_inventory() {
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = GgufFile::open(path).expect("open");
        let mtp_tensors: Vec<_> = g
            .tensors
            .iter()
            .filter(|t| t.name.starts_with("mtp") || t.name.contains(".mtp"))
            .collect();
        eprintln!("[mtp] found {} MTP-prefixed tensors:", mtp_tensors.len());
        let mut total_bytes = 0u64;
        for t in &mtp_tensors {
            eprintln!(
                "[mtp]   {:40} {:?}  shape={:?}  ({} bytes)",
                t.name, t.dtype, t.shape, t.n_bytes
            );
            total_bytes += t.n_bytes;
        }
        eprintln!(
            "[mtp] total MTP weight bytes: {:.2} MiB",
            total_bytes as f64 / (1024.0 * 1024.0)
        );
        // For comparison: also list the canonical "next" architecture key.
        for k in g
            .model
            .metadata()
            .keys()
            .filter(|k| k.contains("mtp") || k.contains("next") || k.contains("speculative"))
        {
            eprintln!("[mtp] metadata key: {k}");
        }
    }

    /// **Attn intra-layer profile**: same idea as the GDN intra
    /// profiler but for full-attn blocks. Critical for the long-context
    /// regression — tells us whether the cost lives in the score loop,
    /// softmax, or V-aggregate inside attn_decode.
    fn attn_intra_profile_single_block(
        mf: &MetalForward,
        attn_block_idx: usize,
        attn_idx_in_session: usize,
        position: u32,
        s: &mut MetalSession,
    ) -> Result<Vec<(String, f64)>, MfError> {
        let ab = match &mf.model.blocks[attn_block_idx] {
            MetalBlock::Attn(a) => a,
            _ => {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "attn_intra",
                    detail: format!("block {attn_block_idx} is not attn"),
                }));
            }
        };
        let arch = &mf.model.arch;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_q = arch.n_q_heads as usize;
        let n_kv = arch.n_kv_heads as usize;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;

        let mut phases: Vec<(String, f64)> = Vec::new();
        let timed = |label: &str,
                     cb: &dyn Fn(&KernelEncoder) -> Result<(), MfError>,
                     phases: &mut Vec<(String, f64)>|
         -> Result<(), MfError> {
            let cmd = mf.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            cb(&enc)?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            phases.push((label.into(), ms));
            Ok(())
        };

        // Pre-mixer norm.
        timed(
            "pre_norm (rms_norm)",
            &|enc| {
                encode_rms_norm_mul_f32(mf.ctx, enc, &s.x, &ab.attn_norm, &s.h, RMS_EPS)
                    .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // Q projection.
        timed(
            "q_proj_2x (mat_vec)",
            &|enc| encode_mat_vec_dispatch(mf.ctx, enc, &ab.q, &s.h, &s.attn_q_full, h, 2 * q_dim),
            &mut phases,
        )?;
        // Split Q + gate.
        timed(
            "split_q_gate",
            &|enc| {
                encode_split_q_gate_f32(
                    mf.ctx,
                    enc,
                    &s.attn_q_full,
                    &s.attn_q,
                    &s.attn_gate,
                    n_q,
                    head_dim,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // Q-norm.
        timed(
            "q_norm (batched rms)",
            &|enc| {
                encode_rms_norm_batched_f32(
                    mf.ctx,
                    enc,
                    &s.attn_q,
                    &ab.q_norm,
                    &s.attn_q_normed,
                    n_q,
                    head_dim,
                    RMS_EPS,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // K, V projections.
        timed(
            "k_proj (mat_vec)",
            &|enc| encode_mat_vec_dispatch(mf.ctx, enc, &ab.k, &s.h, &s.attn_k_now, h, kv_dim),
            &mut phases,
        )?;
        timed(
            "v_proj (mat_vec)",
            &|enc| encode_mat_vec_dispatch(mf.ctx, enc, &ab.v, &s.h, &s.attn_v_now, h, kv_dim),
            &mut phases,
        )?;
        // K-norm.
        timed(
            "k_norm (batched rms)",
            &|enc| {
                encode_rms_norm_batched_f32(
                    mf.ctx,
                    enc,
                    &s.attn_k_now,
                    &ab.k_norm,
                    &s.attn_k_normed,
                    n_kv,
                    head_dim,
                    RMS_EPS,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // RoPE Q.
        timed(
            "rope Q",
            &|enc| {
                encode_rope_neox_f32(
                    mf.ctx,
                    enc,
                    &s.attn_q_normed,
                    n_q,
                    head_dim,
                    n_rot,
                    position,
                    arch.rope_theta,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // RoPE K.
        timed(
            "rope K",
            &|enc| {
                encode_rope_neox_f32(
                    mf.ctx,
                    enc,
                    &s.attn_k_normed,
                    n_kv,
                    head_dim,
                    n_rot,
                    position,
                    arch.rope_theta,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // KV scatter (fused K+V, 1 dispatch).
        timed(
            "kv scatter (fused)",
            &|enc| {
                encode_scatter_offset_f32_to_f16_kv(
                    mf.ctx,
                    enc,
                    &s.attn_k_normed,
                    &s.attn_v_now,
                    &s.kv_k[attn_idx_in_session],
                    &s.kv_v[attn_idx_in_session],
                    (position as usize) * kv_dim,
                    kv_dim,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        s.kv_n_pos[attn_idx_in_session] = position as usize + 1;
        // Attn decode: mirror the production dispatcher so the profiler
        // tracks the kernel path we actually ship.
        const V4_HEAD_DIM: usize = 256;
        let group = n_q / n_kv;
        let n_pos = s.kv_n_pos[attn_idx_in_session];
        let use_v4 = head_dim == V4_HEAD_DIM && matches!(group, 4 | 6 | 8 | 16);
        if use_v4 {
            let nwg = attn_v4_choose_nwg(n_pos, group);
            let tile_c = attn_v4_choose_tile_c(n_pos, group);
            timed(
                "attn_decode_v4_main",
                &|enc| {
                    crate::metal::encode_attn_decode_v4_main_only_f32(
                        mf.ctx,
                        enc,
                        &s.attn_q_normed,
                        &s.kv_k[attn_idx_in_session],
                        &s.kv_v[attn_idx_in_session],
                        &s.attn_v4_o_partial,
                        &s.attn_v4_ml_partial,
                        n_q,
                        n_kv,
                        head_dim,
                        n_pos,
                        nwg,
                        tile_c,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
            timed(
                "attn_decode_v4_reduce",
                &|enc| {
                    crate::metal::encode_attn_decode_v4_reduce_only_f32(
                        mf.ctx,
                        enc,
                        &s.attn_v4_o_partial,
                        &s.attn_v4_ml_partial,
                        &s.attn_o,
                        n_q,
                        n_kv,
                        head_dim,
                        nwg,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        } else {
            timed(
                "attn_decode_f16kv",
                &|enc| {
                    encode_attn_decode_f16kv_f32(
                        mf.ctx,
                        enc,
                        &s.attn_q_normed,
                        &s.kv_k[attn_idx_in_session],
                        &s.kv_v[attn_idx_in_session],
                        &s.attn_o,
                        n_q,
                        n_kv,
                        head_dim,
                        n_pos,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
        // Sigmoid + mul (gated-attn).
        timed(
            "gate sigmoid + mul",
            &|enc| {
                if decode_attn_sigmoid_mul_enabled() {
                    encode_sigmoid_mul_f32(mf.ctx, enc, &s.attn_gate, &s.attn_o, &s.attn_o)
                        .map_err(MfError::from)
                } else {
                    encode_sigmoid_f32(mf.ctx, enc, &s.attn_gate, &s.attn_q)?;
                    encode_mul_f32(mf.ctx, enc, &s.attn_o, &s.attn_q, &s.attn_o)
                        .map_err(MfError::from)
                }
            },
            &mut phases,
        )?;
        // Output proj.
        timed(
            "o_proj (mat_vec)",
            &|enc| encode_mat_vec_dispatch(mf.ctx, enc, &ab.o, &s.attn_o, &s.mixer_out, q_dim, h),
            &mut phases,
        )?;
        // Residual #1.
        timed(
            "residual_add #1",
            &|enc| encode_add_inplace_f32(mf.ctx, enc, &s.x, &s.mixer_out).map_err(MfError::from),
            &mut phases,
        )?;
        // Pre-FFN norm.
        timed(
            "post_norm (rms_norm)",
            &|enc| {
                encode_rms_norm_mul_f32(mf.ctx, enc, &s.x, &ab.post_attn_norm, &s.h, RMS_EPS)
                    .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // FFN — mirror the production fused-or-fallback path from
        // encode_block, so the intra-profiler measurements track what
        // actually runs at decode. Q4_K + Q4_K weights take the fused
        // SwiGLU path (1 dispatch); other dtypes fall back to the
        // 3-dispatch sequence.
        let ffn_fused = ab.ffn_gate.dtype == GgmlType::Q4_K && ab.ffn_up.dtype == GgmlType::Q4_K;
        if ffn_fused {
            timed(
                "ffn_swiglu_q4_K (fused gate+up+silu_mul)",
                &|enc| {
                    encode_ffn_swiglu_q4_K_f32(
                        mf.ctx,
                        enc,
                        &ab.ffn_gate,
                        &ab.ffn_up,
                        &s.h,
                        &s.ffn_inner,
                        h,
                        arch.intermediate_size as usize,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        } else {
            timed(
                "ffn_gate (mat_vec) [unfused fallback]",
                &|enc| {
                    encode_mat_vec_dispatch(
                        mf.ctx,
                        enc,
                        &ab.ffn_gate,
                        &s.h,
                        &s.ffn_gate,
                        h,
                        arch.intermediate_size as usize,
                    )
                },
                &mut phases,
            )?;
            timed(
                "ffn_up (mat_vec) [unfused fallback]",
                &|enc| {
                    encode_mat_vec_dispatch(
                        mf.ctx,
                        enc,
                        &ab.ffn_up,
                        &s.h,
                        &s.ffn_up,
                        h,
                        arch.intermediate_size as usize,
                    )
                },
                &mut phases,
            )?;
            timed(
                "silu_mul [unfused fallback]",
                &|enc| {
                    encode_silu_mul_f32(mf.ctx, enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)
                        .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
        timed(
            "ffn_down (mat_vec)",
            &|enc| {
                encode_mat_vec_dispatch(
                    mf.ctx,
                    enc,
                    &ab.ffn_down,
                    &s.ffn_inner,
                    &s.ffn_out,
                    arch.intermediate_size as usize,
                    h,
                )
            },
            &mut phases,
        )?;
        timed(
            "residual_add #2",
            &|enc| encode_add_inplace_f32(mf.ctx, enc, &s.x, &s.ffn_out).map_err(MfError::from),
            &mut phases,
        )?;
        Ok(phases)
    }

    /// **GDN intra-layer profile**: split a single GDN block across its
    /// 8+ logical sub-phases so we can attribute the ~0.64 ms/layer cost
    /// to which sub-kernels. Free fn (not on MetalForward) because it
    /// lives in the test module.
    fn gdn_intra_profile_single_block(
        mf: &MetalForward,
        gdn_block_idx: usize,
        gdn_idx_in_session: usize,
        s: &mut MetalSession,
    ) -> Result<Vec<(String, f64)>, MfError> {
        let gb = match &mf.model.blocks[gdn_block_idx] {
            MetalBlock::Gdn(g) => g,
            _ => {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "gdn_intra_profile",
                    detail: format!("block {gdn_block_idx} is not GDN"),
                }));
            }
        };
        let arch = &mf.model.arch;
        let h = arch.hidden_size as usize;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        let conv_dim = (2 * n_k + n_v) * head_dim;
        let v_dim = n_v * head_dim;

        let mut phases: Vec<(String, f64)> = Vec::new();
        let timed = |label: &str,
                     cb: &dyn Fn(&KernelEncoder) -> Result<(), MfError>,
                     phases: &mut Vec<(String, f64)>|
         -> Result<(), MfError> {
            let cmd = mf.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            cb(&enc)?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            phases.push((label.into(), ms));
            Ok(())
        };

        // Pre-mixer norm.
        timed(
            "pre_norm (rms_norm)",
            &|enc| {
                encode_rms_norm_mul_f32(mf.ctx, enc, &s.x, &gb.attn_norm, &s.h, RMS_EPS)
                    .map_err(MfError::from)
            },
            &mut phases,
        )?;

        // QKV projection.
        timed(
            "in_proj_qkv (mat_vec)",
            &|enc| {
                encode_mat_vec_dispatch(mf.ctx, enc, &gb.in_proj_qkv, &s.h, &s.gdn_qkv, h, conv_dim)
            },
            &mut phases,
        )?;
        // z projection.
        timed(
            "in_proj_z (mat_vec)",
            &|enc| encode_mat_vec_dispatch(mf.ctx, enc, &gb.in_proj_z, &s.h, &s.gdn_z, h, v_dim),
            &mut phases,
        )?;
        // beta proj + sigmoid.
        timed(
            "beta_proj+sigmoid",
            &|enc| {
                encode_mat_vec_dispatch(mf.ctx, enc, &gb.beta_proj, &s.h, &s.gdn_b, h, n_v)?;
                encode_sigmoid_f32(mf.ctx, enc, &s.gdn_b, &s.gdn_beta).map_err(MfError::from)
            },
            &mut phases,
        )?;
        // alpha proj + decay-chain, matching production.
        timed(
            "alpha_proj+decay_chain",
            &|enc| {
                encode_mat_vec_dispatch(mf.ctx, enc, &gb.alpha_proj, &s.h, &s.gdn_a, h, n_v)?;
                encode_gdn_decay_chain_f32(
                    mf.ctx,
                    enc,
                    &s.gdn_a,
                    &gb.dt_bias,
                    &gb.a_log,
                    &s.gdn_alpha,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // ssm_conv (with internal silu).
        timed(
            "ssm_conv_silu",
            &|enc| {
                encode_ssm_conv_silu_f32(
                    mf.ctx,
                    enc,
                    &s.gdn_qkv,
                    &s.gdn_conv[gdn_idx_in_session],
                    &gb.conv1d,
                    &s.gdn_qkv_conv,
                    conv_dim,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // qkv split via zero-copy views (no dispatch). Per Jeff & Sanjay:
        // avoid copies / use indices instead of pointers.
        let q_view = s
            .gdn_qkv_conv
            .view_subrange(0, vec![(n_k * head_dim) as u64]);
        let k_view = s
            .gdn_qkv_conv
            .view_subrange((n_k * head_dim) as u64, vec![(n_k * head_dim) as u64]);
        let v_view = s
            .gdn_qkv_conv
            .view_subrange((2 * n_k * head_dim) as u64, vec![v_dim as u64]);
        timed(
            "l2_norm_qk (2 batched)",
            &|enc| {
                encode_l2_norm_batched_f32(
                    mf.ctx,
                    enc,
                    &q_view,
                    &s.gdn_q_norm,
                    n_k,
                    head_dim,
                    RMS_EPS,
                )?;
                encode_l2_norm_batched_f32(
                    mf.ctx,
                    enc,
                    &k_view,
                    &s.gdn_k_norm,
                    n_k,
                    head_dim,
                    RMS_EPS,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // gdn_step_decay: production recurrence with precomputed decay.
        timed(
            "gdn_step_decay (recurrence)",
            &|enc| {
                encode_gdn_step_decay_f32(
                    mf.ctx,
                    enc,
                    &s.gdn_q_norm,
                    &s.gdn_k_norm,
                    &v_view,
                    &s.gdn_alpha,
                    &s.gdn_beta,
                    &s.gdn_state[gdn_idx_in_session],
                    &s.gdn_out,
                    n_v,
                    n_k,
                    head_dim,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // rmsnorm_gated.
        timed(
            "rmsnorm_gated",
            &|enc| {
                encode_rmsnorm_gated_f32(
                    mf.ctx,
                    enc,
                    &s.gdn_out,
                    &gb.norm,
                    &s.gdn_z,
                    &s.gdn_normed,
                    n_v,
                    head_dim,
                    RMS_EPS,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // out_proj.
        timed(
            "out_proj (mat_vec)",
            &|enc| {
                encode_mat_vec_dispatch(
                    mf.ctx,
                    enc,
                    &gb.out_proj,
                    &s.gdn_normed,
                    &s.mixer_out,
                    v_dim,
                    h,
                )
            },
            &mut phases,
        )?;
        // residual #1.
        timed(
            "residual_add #1",
            &|enc| encode_add_inplace_f32(mf.ctx, enc, &s.x, &s.mixer_out).map_err(MfError::from),
            &mut phases,
        )?;
        // post-FFN norm.
        timed(
            "post_norm (rms_norm)",
            &|enc| {
                encode_rms_norm_mul_f32(mf.ctx, enc, &s.x, &gb.post_attn_norm, &s.h, RMS_EPS)
                    .map_err(MfError::from)
            },
            &mut phases,
        )?;
        // FFN — mirror the production fused-or-fallback path.
        let ffn_fused = gb.ffn_gate.dtype == GgmlType::Q4_K && gb.ffn_up.dtype == GgmlType::Q4_K;
        if ffn_fused {
            timed(
                "ffn_swiglu_q4_K (fused gate+up+silu_mul)",
                &|enc| {
                    encode_ffn_swiglu_q4_K_f32(
                        mf.ctx,
                        enc,
                        &gb.ffn_gate,
                        &gb.ffn_up,
                        &s.h,
                        &s.ffn_inner,
                        h,
                        arch.intermediate_size as usize,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        } else {
            timed(
                "ffn_gate (mat_vec) [unfused fallback]",
                &|enc| {
                    encode_mat_vec_dispatch(
                        mf.ctx,
                        enc,
                        &gb.ffn_gate,
                        &s.h,
                        &s.ffn_gate,
                        h,
                        arch.intermediate_size as usize,
                    )
                },
                &mut phases,
            )?;
            timed(
                "ffn_up (mat_vec) [unfused fallback]",
                &|enc| {
                    encode_mat_vec_dispatch(
                        mf.ctx,
                        enc,
                        &gb.ffn_up,
                        &s.h,
                        &s.ffn_up,
                        h,
                        arch.intermediate_size as usize,
                    )
                },
                &mut phases,
            )?;
            timed(
                "silu_mul [unfused fallback]",
                &|enc| {
                    encode_silu_mul_f32(mf.ctx, enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)
                        .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
        timed(
            "ffn_down (mat_vec)",
            &|enc| {
                encode_mat_vec_dispatch(
                    mf.ctx,
                    enc,
                    &gb.ffn_down,
                    &s.ffn_inner,
                    &s.ffn_out,
                    arch.intermediate_size as usize,
                    h,
                )
            },
            &mut phases,
        )?;
        // residual #2.
        timed(
            "residual_add #2",
            &|enc| encode_add_inplace_f32(mf.ctx, enc, &s.x, &s.ffn_out).map_err(MfError::from),
            &mut phases,
        )?;
        Ok(phases)
    }

    /// Split one MoE block into mixer, route, routed expert FFN, shared FFN,
    /// and residual pieces. This is profiler-only; production keeps these in a
    /// single command buffer for normal decode.
    fn moe_intra_profile_single_block(
        mf: &MetalForward,
        block_idx: usize,
        mixer_slot: MixerSlot,
        position: u32,
        s: &mut MetalSession,
    ) -> Result<Vec<(String, f64)>, MfError> {
        let block = &mf.model.blocks[block_idx];
        let (ffn_gate, ffn_up, ffn_down, moe) = match block {
            MetalBlock::Gdn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
            MetalBlock::Attn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
        };
        let moe = moe.ok_or(MfError::UnsupportedMoe)?;
        let arch = &mf.model.arch;
        let h = arch.hidden_size as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let f_shared = arch.expert_shared_feed_forward_length as usize;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;

        let mut phases: Vec<(String, f64)> = Vec::new();
        let timed = |label: &str,
                     cb: &dyn Fn(&KernelEncoder) -> Result<(), MfError>,
                     phases: &mut Vec<(String, f64)>|
         -> Result<(), MfError> {
            let cmd = mf.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            cb(&enc)?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            phases.push((label.into(), ms));
            Ok(())
        };
        let timed_mut = |label: &str,
                         cb: &mut dyn FnMut(&KernelEncoder) -> Result<(), MfError>,
                         phases: &mut Vec<(String, f64)>|
         -> Result<(), MfError> {
            let cmd = mf.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            cb(&enc)?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            phases.push((label.into(), ms));
            Ok(())
        };

        timed_mut(
            "mixer_prep (norm+mixer+resid+postnorm)",
            &mut |enc| mf.encode_moe_mixer_prep(enc, block, mixer_slot, position, s),
            &mut phases,
        )?;
        timed_mut(
            "route_prepare (router+topk+shared_gate)",
            &mut |enc| mf.encode_moe_route_prepare(enc, s, moe),
            &mut phases,
        )?;

        let moe_inner = s.moe_inner.view_subrange(0, vec![(topk * f_exp) as u64]);
        let moe_expert_out = s.moe_expert_out.view_subrange(0, vec![(topk * h) as u64]);
        let topk_idx = s.moe_topk_idx.view_subrange(0, vec![topk as u64]);
        let topk_w = s.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        match moe.gate_exps.dtype {
            GgmlType::Q4_K => {
                timed(
                    "routed_gate_up_swiglu_q4_K",
                    &|enc| {
                        encode_moe_swiglu_q4_K_f32(
                            mf.ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &s.h,
                            &topk_idx,
                            &moe_inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                        )
                        .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
            }
            GgmlType::Q5_K => {
                let gate_pack = s
                    .moe_expert_out
                    .view_subrange(0, vec![(topk * f_exp) as u64]);
                let up_pack = s
                    .moe_expert_out
                    .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                timed(
                    "routed_gate_q5_K",
                    &|enc| {
                        encode_moe_mat_vec_q5_K_f32(
                            mf.ctx,
                            enc,
                            &moe.gate_exps,
                            &s.h,
                            &topk_idx,
                            &gate_pack,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                        )
                        .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
                timed(
                    "routed_up_q5_K",
                    &|enc| {
                        encode_moe_mat_vec_q5_K_f32(
                            mf.ctx,
                            enc,
                            &moe.up_exps,
                            &s.h,
                            &topk_idx,
                            &up_pack,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                        )
                        .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
                timed(
                    "routed_silu_mul",
                    &|enc| {
                        encode_silu_mul_f32(mf.ctx, enc, &gate_pack, &up_pack, &moe_inner)
                            .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
            }
            GgmlType::F32 => {
                let gate_pack = s
                    .moe_expert_out
                    .view_subrange(0, vec![(topk * f_exp) as u64]);
                let up_pack = s
                    .moe_expert_out
                    .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                timed(
                    "routed_gate_f32",
                    &|enc| {
                        encode_moe_mat_vec_f32(
                            mf.ctx,
                            enc,
                            &moe.gate_exps,
                            &s.h,
                            &topk_idx,
                            &gate_pack,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                        )
                        .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
                timed(
                    "routed_up_f32",
                    &|enc| {
                        encode_moe_mat_vec_f32(
                            mf.ctx,
                            enc,
                            &moe.up_exps,
                            &s.h,
                            &topk_idx,
                            &up_pack,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                        )
                        .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
                timed(
                    "routed_silu_mul",
                    &|enc| {
                        encode_silu_mul_f32(mf.ctx, enc, &gate_pack, &up_pack, &moe_inner)
                            .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
            }
            dtype => {
                return Err(MfError::UnsupportedDtype {
                    name: "MoE routed gate/up expert banks".into(),
                    dtype,
                });
            }
        }

        match moe.down_exps.dtype {
            GgmlType::Q5_K => {
                if decode_moe_q5_down_fused_enabled() {
                    timed(
                        "routed_down_weighted_sum_q5_K",
                        &|enc| {
                            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                                mf.ctx,
                                enc,
                                &moe.down_exps,
                                &moe_inner,
                                &topk_idx,
                                &topk_w,
                                &s.mixer_out,
                                f_exp,
                                h,
                                n_expert,
                                topk,
                                1,
                            )
                            .map_err(MfError::from)
                        },
                        &mut phases,
                    )?;
                } else {
                    timed(
                        "routed_down_q5_K",
                        &|enc| {
                            encode_moe_down_q5_K_f32(
                                mf.ctx,
                                enc,
                                &moe.down_exps,
                                &moe_inner,
                                &topk_idx,
                                &moe_expert_out,
                                f_exp,
                                h,
                                n_expert,
                                topk,
                            )
                            .map_err(MfError::from)
                        },
                        &mut phases,
                    )?;
                    timed(
                        "routed_weighted_sum",
                        &|enc| {
                            encode_moe_weighted_sum_f32(
                                mf.ctx,
                                enc,
                                &moe_expert_out,
                                &topk_w,
                                &s.mixer_out,
                                h,
                                topk,
                            )
                            .map_err(MfError::from)
                        },
                        &mut phases,
                    )?;
                }
            }
            GgmlType::Q6_K => {
                timed(
                    "routed_down_weighted_sum_q6_K",
                    &|enc| {
                        encode_moe_down_weighted_sum_q6_K_f32(
                            mf.ctx,
                            enc,
                            &moe.down_exps,
                            &moe_inner,
                            &topk_idx,
                            &topk_w,
                            &s.mixer_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                        )
                        .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
            }
            GgmlType::IQ4_XS => {
                timed(
                    "routed_down_iq4_xs",
                    &|enc| {
                        encode_moe_down_iq4_xs_f32(
                            mf.ctx,
                            enc,
                            &moe.down_exps,
                            &moe_inner,
                            &topk_idx,
                            &moe_expert_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                        )
                        .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
                timed(
                    "routed_weighted_sum",
                    &|enc| {
                        encode_moe_weighted_sum_f32(
                            mf.ctx,
                            enc,
                            &moe_expert_out,
                            &topk_w,
                            &s.mixer_out,
                            h,
                            topk,
                        )
                        .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
            }
            dtype => {
                return Err(MfError::UnsupportedDtype {
                    name: "MoE routed down expert bank".into(),
                    dtype,
                });
            }
        }

        let shared_gate_tmp = s.ffn_gate.view_subrange(0, vec![f_shared as u64]);
        let shared_up_tmp = s.ffn_up.view_subrange(0, vec![f_shared as u64]);
        let shared_inner_tmp = s.ffn_inner.view_subrange(0, vec![f_shared as u64]);
        let shared_out_tmp = s.ffn_out.view_subrange(0, vec![h as u64]);
        timed(
            "shared_gate (mat_vec)",
            &|enc| {
                encode_mat_vec_dispatch(mf.ctx, enc, ffn_gate, &s.h, &shared_gate_tmp, h, f_shared)
            },
            &mut phases,
        )?;
        timed(
            "shared_up (mat_vec)",
            &|enc| encode_mat_vec_dispatch(mf.ctx, enc, ffn_up, &s.h, &shared_up_tmp, h, f_shared),
            &mut phases,
        )?;
        timed(
            "shared_silu_mul",
            &|enc| {
                encode_silu_mul_f32(
                    mf.ctx,
                    enc,
                    &shared_gate_tmp,
                    &shared_up_tmp,
                    &shared_inner_tmp,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        timed(
            "shared_down (mat_vec)",
            &|enc| {
                encode_mat_vec_dispatch(
                    mf.ctx,
                    enc,
                    ffn_down,
                    &shared_inner_tmp,
                    &shared_out_tmp,
                    f_shared,
                    h,
                )
            },
            &mut phases,
        )?;
        timed(
            "shared_axpy_scalar",
            &|enc| {
                encode_axpy_scalar_f32(
                    mf.ctx,
                    enc,
                    &shared_out_tmp,
                    &s.moe_shared_gate,
                    &s.mixer_out,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        timed(
            "residual_add #2",
            &|enc| encode_add_inplace_f32(mf.ctx, enc, &s.x, &s.mixer_out).map_err(MfError::from),
            &mut phases,
        )?;

        Ok(phases)
    }

    /// **Phase-resolved profile**: at each context length we care about,
    /// run a phase-split forward to attribute GPU time to logical phases.
    /// This is the experiment that should tell us whether the long-context
    /// regression lives in attn layers, GDN layers, or somewhere else.
    #[test]
    #[ignore]
    fn metal_27b_phase_profile() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        let max_n = 4096usize;
        let mf = MetalForward::new(&ctx, &mm);
        // Single warmup session for pipeline cache.
        {
            let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("warmup");
            for i in 0..3 {
                let _ = mf.single_token(0, i as u32, &mut s).expect("warmup");
            }
        }

        for &target in &[1usize, 1024, 4096] {
            let mut s = MetalSession::fresh(&ctx, &mm, max_n + 16).expect("session");
            // Ramp to `target` positions (no timing).
            for p in 0..(target as u32) {
                let _ = mf.single_token(0, p, &mut s).expect("ramp");
            }
            // Now do a phase-profiled call at position `target`.
            let (_logits, wall_with_artifact_ms, phases) = mf
                .single_token_phase_profiled(0, target as u32, &mut s)
                .expect("phase");

            let phase_sum_ms: f64 = phases.iter().map(|p| p.1).sum();
            // ★ phase_sum is the production-realistic GPU time; the wall
            // includes ~12 ms of per-phase cmdbuf overhead (artifact).
            // For production ms/token use metal_27b_context_sweep instead.
            eprintln!(
                "[phase ctx={target:>5}] phase_sum {phase_sum_ms:.2} ms (production-realistic) | \
                 wall_with_artifact {wall_with_artifact_ms:.2} ms (DO NOT use for prod ms/token)"
            );
            for (name, ms) in &phases {
                let pct = ms / phase_sum_ms * 100.0;
                eprintln!("[phase ctx={target:>5}]   {name:25} {ms:7.2} ms  ({pct:5.1}%)");
            }
        }
    }

    /// **Attn intra-layer breakdown across context lengths**. Tells us
    /// which sub-phase of attention scales with n_pos. Critical: only
    /// `attn_decode` should grow with context; everything else should
    /// be flat. If something else grows, that's an unexpected scaling
    /// problem.
    #[test]
    #[ignore]
    fn metal_27b_attn_intra_profile() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        // Warmup.
        {
            let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
            for i in 0..3 {
                let _ = mf.single_token(0, i as u32, &mut s).expect("warmup");
            }
        }

        // Find first attn block index (0.8B has it at block 3 — let me find it
        // for 27B). 27B has pattern [GDN×3, attn×1] × 16, so block 3 is the
        // first attn block; 27B has 16 attn blocks total.
        let attn_block_idx = 3usize;

        for &target in &[1usize, 1024, 4096] {
            let mut s = MetalSession::fresh(&ctx, &mm, target + 32).expect("session");
            // Ramp to `target`.
            for p in 0..(target as u32) {
                let _ = mf.single_token(0, p, &mut s).expect("ramp");
            }
            // Profile one attn block in isolation, averaging 5 runs.
            let n_runs = 5usize;
            let mut agg: std::collections::BTreeMap<String, f64> =
                std::collections::BTreeMap::new();
            let mut order: Vec<String> = Vec::new();
            for run in 0..n_runs {
                // Position must increment with each call (for KV scatter).
                let pos = target as u32 + run as u32;
                let phases = attn_intra_profile_single_block(&mf, attn_block_idx, 0, pos, &mut s)
                    .expect("attn-intra");
                for (name, ms) in phases {
                    if run == 0 {
                        order.push(name.clone());
                    }
                    *agg.entry(name).or_default() += ms;
                }
            }
            let total: f64 = agg.values().sum::<f64>() / n_runs as f64;
            eprintln!(
                "[attn-intra ctx={target}] one ATTN layer total: {total:.3} ms (×16 = {:.2} ms)",
                total * 16.0
            );
            for name in &order {
                let avg_ms = agg[name] / n_runs as f64;
                let pct = avg_ms / total * 100.0;
                eprintln!("[attn-intra ctx={target}]   {name:35} {avg_ms:6.3} ms  ({pct:5.1}%)");
            }
            eprintln!();
        }
    }

    /// **GDN intra-layer breakdown**: profile a single GDN block by
    /// sub-phase. Tells us where the ~0.64 ms/layer cost lives —
    /// which sub-kernels are big, which are negligible, and which
    /// fusion targets are worth pursuing.
    #[test]
    #[ignore]
    fn metal_27b_gdn_intra_profile() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        // Warmup pipeline cache.
        let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
        for i in 0..3 {
            let _ = mf.single_token(0, i as u32, &mut s).expect("warmup");
        }

        // Run a real forward to populate state, then profile block 0 (a
        // GDN block).
        let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
        let _ = mf.single_token(9419, 0, &mut s).expect("p0");

        // Profile just one GDN block in isolation. Aggregate across
        // 5 runs to get noise-floor stable numbers.
        let n_runs = 5usize;
        let mut agg: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
        let mut order: Vec<String> = Vec::new();
        for run in 0..n_runs {
            let phases = gdn_intra_profile_single_block(&mf, 0, 0, &mut s).expect("intra");
            for (name, ms) in phases {
                if run == 0 {
                    order.push(name.clone());
                }
                *agg.entry(name).or_default() += ms;
            }
        }
        let total: f64 = agg.values().sum::<f64>() / n_runs as f64;
        eprintln!("[gdn-intra] === per-sub-phase breakdown (avg of {n_runs} runs) ===");
        eprintln!("[gdn-intra] one GDN layer total: {total:.3} ms");
        for name in &order {
            let avg_ms = agg[name] / n_runs as f64;
            let pct = avg_ms / total * 100.0;
            eprintln!("[gdn-intra]   {name:35} {avg_ms:6.3} ms  ({pct:5.1}%)");
        }
        eprintln!(
            "[gdn-intra] extrapolated to 48 layers: {:.2} ms",
            total * 48.0
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_gdn_intra_profile() {
        let model_path = "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
        for i in 0..3 {
            let _ = mf.single_token(0, i as u32, &mut s).expect("warmup");
        }

        let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
        let _ = mf.single_token(0, 0, &mut s).expect("p0");

        let n_runs = 8usize;
        let mut agg: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
        let mut order: Vec<String> = Vec::new();
        for run in 0..n_runs {
            let phases = gdn_intra_profile_single_block(&mf, 0, 0, &mut s).expect("intra");
            for (name, ms) in phases {
                if run == 0 {
                    order.push(name.clone());
                }
                *agg.entry(name).or_default() += ms;
            }
        }
        let total: f64 = agg.values().sum::<f64>() / n_runs as f64;
        eprintln!("[gdn-intra-a3b] === per-sub-phase breakdown (avg of {n_runs} runs) ===");
        eprintln!("[gdn-intra-a3b] one GDN layer total: {total:.3} ms");
        for name in &order {
            let avg_ms = agg[name] / n_runs as f64;
            let pct = avg_ms / total * 100.0;
            eprintln!("[gdn-intra-a3b]   {name:35} {avg_ms:6.3} ms  ({pct:5.1}%)");
        }
        eprintln!(
            "[gdn-intra-a3b] extrapolated to 30 layers: {:.2} ms",
            total * 30.0
        );
    }

    fn run_moe_intra_profile(model_path: &str, label: &str, n_runs: usize) {
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
        for i in 0..3 {
            let _ = mf.single_token(0, i as u32, &mut s).expect("warmup");
        }

        let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
        let _ = mf.single_token(0, 0, &mut s).expect("p0");
        let slot = match &mf.model.blocks[0] {
            MetalBlock::Gdn(_) => MixerSlot::Gdn(0),
            MetalBlock::Attn(_) => MixerSlot::Attn(0),
        };

        let mut agg: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
        let mut order: Vec<String> = Vec::new();
        for run in 0..n_runs {
            let phases = moe_intra_profile_single_block(&mf, 0, slot, run as u32, &mut s)
                .expect("moe-intra");
            for (name, ms) in phases {
                if run == 0 {
                    order.push(name.clone());
                }
                *agg.entry(name).or_default() += ms;
            }
        }
        let total: f64 = agg.values().sum::<f64>() / n_runs as f64;
        eprintln!("[moe-intra-{label}] === per-sub-phase breakdown (avg of {n_runs} runs) ===");
        eprintln!("[moe-intra-{label}] one MoE block total: {total:.3} ms");
        for name in &order {
            let avg_ms = agg[name] / n_runs as f64;
            let pct = avg_ms / total * 100.0;
            eprintln!("[moe-intra-{label}]   {name:42} {avg_ms:6.3} ms  ({pct:5.1}%)");
        }
        eprintln!(
            "[moe-intra-{label}] extrapolated to {} blocks: {:.2} ms",
            mf.model.blocks.len(),
            total * mf.model.blocks.len() as f64
        );
    }

    #[test]
    #[ignore]
    fn metal_35b_a3b_moe_intra_profile() {
        run_moe_intra_profile(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "a3b",
            6,
        );
    }

    #[test]
    #[ignore]
    fn metal_122b_a10b_moe_intra_profile() {
        run_moe_intra_profile(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
            "122b",
            4,
        );
    }

    /// **Context-length sweep**: how does decode throughput scale as the
    /// KV cache and GDN state grow? llama-bench's `tg128` is at fixed
    /// position 0..127. We sweep further to see where the cliffs are.
    ///
    /// Drives 1, 64, 256, 1024, 4096 tokens and reports per-token cost
    /// at each prefix length. The KV cache grows linearly with context
    /// (16 attn layers × 64 KB / token), so attn_decode kernel
    /// time should grow linearly too. GDN state is fixed-size so GDN
    /// layer cost is invariant.
    #[test]
    #[ignore]
    fn metal_27b_context_sweep() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        // Sweep targets. Naive attn_decode_f32 / f16kv had a hard cap at
        // ~7000 positions (28KB threadgroup memory ÷ 4 B/score). v4
        // (online softmax + split-K) unlocks arbitrary context.
        // 16384 ramp + window costs ~5 minutes; trim if quick iteration is
        // needed.
        let checkpoints = [1usize, 64, 256, 1024, 4096, 8192, 16384];

        let max_n = *checkpoints.iter().max().unwrap();
        let mut s = MetalSession::fresh(&ctx, &mm, max_n + 16).expect("session");
        let mf = MetalForward::new(&ctx, &mm);

        // Warmup pipeline state cache.
        for i in 0..3 {
            let _ = mf.single_token(0, i as u32, &mut s).expect("warmup");
        }
        let mut s = MetalSession::fresh(&ctx, &mm, max_n + 16).expect("session2");
        // One pre-warmed token at position 0 to populate everything.
        let _ = mf.single_token(0, 0, &mut s).expect("p0");

        eprintln!("[ctx-sweep] === per-token decode cost vs context ===");
        eprintln!("[ctx-sweep] context  total_ms  gpu_ms  cpu_enc_ms  t/s   GB/s   %peak");

        let mut prev_pos = 1u32;
        for &target in &checkpoints {
            // Ramp KV cache + GDN state to `target` positions.
            // For positions 1..target we don't need to time; just need them
            // populated. Use any token id (0).
            for p in prev_pos..(target as u32) {
                let _ = mf.single_token(0, p, &mut s).expect("ramp");
            }
            prev_pos = target as u32;

            // Time a window at this context length.
            const WINDOW: usize = 5;
            let mut samples = Vec::with_capacity(WINDOW);
            for i in 0..WINDOW {
                let pos = prev_pos + i as u32;
                let (_, p) = mf.single_token_profiled(0, pos, &mut s).expect("timed");
                samples.push(p);
            }
            prev_pos += WINDOW as u32;

            let avg_total = samples.iter().map(|p| p.total_ms).sum::<f64>() / WINDOW as f64;
            let avg_gpu = samples.iter().map(|p| p.gpu_kernel_ms).sum::<f64>() / WINDOW as f64;
            let avg_enc = samples.iter().map(|p| p.cpu_encode_ms).sum::<f64>() / WINDOW as f64;
            let bw = 16.8_f64 / (avg_gpu / 1000.0); // model-only bytes
            eprintln!(
                "[ctx-sweep] {target:>7}  {avg_total:>8.2}  {avg_gpu:>6.2}  {avg_enc:>10.2}  {:>4.1}  {bw:>5.0}   {:>4.0}%",
                1000.0 / avg_total,
                bw / 5.46
            );
        }
    }

    /// **Per-token profiling on 27B-Q4_K_M.** Runs N steady-state tokens,
    /// reports the CPU-encode / GPU-kernel / total-wall split, and the
    /// dispatch count. The data we feed to optimization decisions.
    #[test]
    #[ignore]
    fn metal_27b_perf_profile() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok
            .encode(
                "The quick brown fox jumps over the lazy dog and runs into the field where",
                false,
            )
            .expect("tokenize");
        eprintln!("[perf-27b] {} prompt tokens", ids.len());

        let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 32).expect("session");
        let mf = MetalForward::new(&ctx, &mm);

        // Warmup: enough tokens to fully populate pipeline cache.
        for (i, &tid) in ids.iter().take(3).enumerate() {
            let _ = mf.single_token(tid, i as u32, &mut s).expect("warmup");
        }
        // Reset session for a clean steady-state run.
        let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 32).expect("session2");
        // Re-warmup PSO cache by running once.
        let _ = mf.single_token(ids[0], 0, &mut s).expect("warmup2");
        let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 32).expect("session3");

        let mut profiles: Vec<TokenProfile> = Vec::new();
        for (i, &tid) in ids.iter().enumerate() {
            let (_, p) = mf
                .single_token_profiled(tid, i as u32, &mut s)
                .expect("forward");
            profiles.push(p);
        }

        // Skip the first to avoid first-call jitter.
        let steady = &profiles[1..];
        let avg = |f: fn(&TokenProfile) -> f64| -> f64 {
            steady.iter().map(f).sum::<f64>() / steady.len() as f64
        };
        let total = avg(|p| p.total_ms);
        let cpu_enc = avg(|p| p.cpu_encode_ms);
        let gpu_kern = avg(|p| p.gpu_kernel_ms);
        let cpu_gpu = avg(|p| p.cpu_to_gpu_complete_ms);
        let queue_overhead = cpu_gpu - gpu_kern;
        let readback_etc = total - cpu_enc - cpu_gpu;

        eprintln!(
            "[perf-27b] === avg over {} steady-state tokens ===",
            steady.len()
        );
        eprintln!(
            "[perf-27b]   total wall:           {total:.2} ms = {:.2} t/s",
            1000.0 / total
        );
        eprintln!(
            "[perf-27b]   cpu encode:           {cpu_enc:.2} ms ({:.0}%)",
            cpu_enc / total * 100.0
        );
        eprintln!(
            "[perf-27b]   gpu kernels:          {gpu_kern:.2} ms ({:.0}%)",
            gpu_kern / total * 100.0
        );
        eprintln!(
            "[perf-27b]   queue/sched overhead: {queue_overhead:.2} ms ({:.0}%)",
            queue_overhead / total * 100.0
        );
        eprintln!(
            "[perf-27b]   readback + misc:      {readback_etc:.2} ms ({:.0}%)",
            readback_etc / total * 100.0
        );

        // Theoretical bandwidth-bound floor for this model: 16.8 GB / 546 GB/s
        // = 30.7 ms. So gpu_kernel_ms tells us how close we are to the BW wall.
        let gb = 16.8_f64;
        let peak = 546.0_f64;
        let bw_floor = gb / peak * 1000.0;
        eprintln!(
            "[perf-27b]   bandwidth floor:      {bw_floor:.2} ms ({:.0} GB/s peak; we're at {:.0} GB/s = {:.0}%)",
            peak,
            gb / (gpu_kern / 1000.0),
            gb / (gpu_kern / 1000.0) / peak * 100.0
        );
        // llama.cpp clean baseline: 21.21 t/s = 47.1 ms/token.
        eprintln!("[perf-27b]   llama.cpp baseline:   47.15 ms (21.21 t/s)");
        eprintln!(
            "[perf-27b]   our headroom to BW floor: {:.2} ms",
            gpu_kern - bw_floor
        );
        eprintln!(
            "[perf-27b]   our headroom to llama.cpp: {:.2} ms ({:+.1} t/s)",
            total - 47.15,
            1000.0 / total - 21.21
        );

        eprintln!("[perf-27b] per-token profiles:");
        for (i, p) in profiles.iter().enumerate() {
            eprintln!(
                "[perf-27b]   t{i}: total={:.2} cpu_enc={:.2} gpu={:.2} q={:.2}",
                p.total_ms,
                p.cpu_encode_ms,
                p.gpu_kernel_ms,
                p.cpu_to_gpu_complete_ms - p.gpu_kernel_ms
            );
        }
    }

    /// **27B Q4_K_M, multi-token**: validates position > 0 + steady-state
    /// throughput. The headline number we've been working toward.
    #[test]
    #[ignore]
    fn metal_27b_multi_token_perf() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        let oracle_path = "/tmp/qwen-oracle/longprompt_27b.f32";
        if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists()
        {
            eprintln!("[metal-27b-multi] skipped — fixtures missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };

        let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
        let n = oracle_bytes.len() / 4;
        let oracle: Vec<f32> = (0..n)
            .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok
            .encode("The quick brown fox jumps over the lazy dog", false)
            .expect("tokenize");
        eprintln!("[metal-27b-multi] {} tokens: {ids:?}", ids.len());
        assert_eq!(ids.len(), 9);

        let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session");
        let mf = MetalForward::new(&ctx, &mm);

        // Warmup pass to compile pipeline state objects + warm caches.
        let _ = mf.single_token(ids[0], 0, &mut s).expect("warmup");
        // Reset session for the actual run.
        let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session2");

        let t = std::time::Instant::now();
        let mut last = vec![];
        let mut per_token_ms: Vec<f64> = Vec::new();
        for (i, &tid) in ids.iter().enumerate() {
            let tt = std::time::Instant::now();
            last = mf.single_token(tid, i as u32, &mut s).expect("forward");
            per_token_ms.push(tt.elapsed().as_secs_f64() * 1e3);
        }
        let total_ms = t.elapsed().as_secs_f64() * 1e3;

        let mut max_abs = 0.0f32;
        let mut argmax_ours = 0usize;
        let mut argmax_oracle = 0usize;
        let mut max_ours = f32::NEG_INFINITY;
        let mut max_oracle = f32::NEG_INFINITY;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..n {
            let d = (last[i] - oracle[i]).abs();
            max_abs = max_abs.max(d);
            if last[i] > max_ours {
                max_ours = last[i];
                argmax_ours = i;
            }
            if oracle[i] > max_oracle {
                max_oracle = oracle[i];
                argmax_oracle = i;
            }
            dot += last[i] as f64 * oracle[i] as f64;
            na += (last[i] as f64).powi(2);
            nb += (oracle[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[metal-27b-multi] {total_ms:.1}ms total, {:.1}ms/token (avg) — argmax: ours={argmax_ours} ({:.4}) | oracle={argmax_oracle} ({:.4}) | max|Δ|={max_abs:.4} cos={cos:.6}",
            total_ms / ids.len() as f64,
            max_ours,
            max_oracle
        );
        eprintln!("[metal-27b-multi] per-token (ms): {per_token_ms:?}");
        let avg_excl_first =
            per_token_ms[1..].iter().sum::<f64>() / (per_token_ms.len() - 1) as f64;
        eprintln!(
            "[metal-27b-multi] steady-state (excl. first): {avg_excl_first:.1} ms/token = {:.2} t/s",
            1000.0 / avg_excl_first
        );
        assert_eq!(argmax_ours, argmax_oracle, "argmax disagreement");
        assert!(cos > 0.999, "cos={cos} below threshold");
    }

    /// **End-to-end Metal forward, multi-token**. Exercises position > 0
    /// in the attn block (RoPE, KV cache reads at multiple positions).
    /// Oracle: llm/llama_core's snapshot dump for "The quick brown fox
    /// jumps over the lazy dog" (9 tokens), Qwen3.5-0.8B-F32.
    #[test]
    fn metal_multi_token_matches_cpu_oracle() {
        let model_path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        let oracle_path = "/tmp/qwen-oracle/longprompt_t0.f32";
        if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists()
        {
            eprintln!("[metal-e2e-multi] skipped — fixtures missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
        let n = oracle_bytes.len() / 4;
        let oracle: Vec<f32> = (0..n)
            .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok
            .encode("The quick brown fox jumps over the lazy dog", false)
            .expect("tokenize");
        eprintln!("[metal-e2e-multi] {} tokens: {ids:?}", ids.len());
        assert_eq!(ids.len(), 9);

        let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session");
        let mf = MetalForward::new(&ctx, &mm);
        let t = std::time::Instant::now();
        let mut last = vec![];
        for (i, &tid) in ids.iter().enumerate() {
            last = mf.single_token(tid, i as u32, &mut s).expect("forward");
        }
        let total_ms = t.elapsed().as_secs_f64() * 1e3;

        let mut max_abs = 0.0f32;
        let mut argmax_ours = 0usize;
        let mut argmax_oracle = 0usize;
        let mut max_ours = f32::NEG_INFINITY;
        let mut max_oracle = f32::NEG_INFINITY;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..n {
            let d = (last[i] - oracle[i]).abs();
            max_abs = max_abs.max(d);
            if last[i] > max_ours {
                max_ours = last[i];
                argmax_ours = i;
            }
            if oracle[i] > max_oracle {
                max_oracle = oracle[i];
                argmax_oracle = i;
            }
            dot += last[i] as f64 * oracle[i] as f64;
            na += (last[i] as f64).powi(2);
            nb += (oracle[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[metal-e2e-multi] {total_ms:.1}ms ({:.1}ms/token) — argmax: ours={argmax_ours} ({:.4}) | oracle={argmax_oracle} ({:.4}) | max|Δ|={max_abs:.4} cos={cos:.6}",
            total_ms / ids.len() as f64,
            max_ours,
            max_oracle
        );
        assert_eq!(argmax_ours, argmax_oracle, "argmax disagreement");
        assert!(cos > 0.9999, "cos={cos} below threshold");
    }

    /// **v0.75.0 correctness gate**: skip-tail prefill must produce
    /// bit-identical session state to the full-tail path. Two sessions
    /// run the same 9-token prompt: session A goes through `single_token`
    /// for every token, session B goes through `single_token_no_tail`
    /// for tokens [0..n-1) and `single_token` for the last token. Final
    /// logits MUST match bit-exactly (same forward path through embed +
    /// blocks + final norm + lm_head; the no_tail path just skips work
    /// that doesn't feed back into the next iteration).
    ///
    /// Equally critical: the `target_layer_ids` capture path
    /// (`single_token_with_multi_hidden_no_tail`) must produce
    /// bit-identical hidden_dst on every prefill step. We accumulate
    /// the per-token captures across the prompt and compare.
    #[test]
    fn no_tail_prefill_matches_full_tail() {
        let model_path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[no-tail-prefill] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);

        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok
            .encode("The quick brown fox jumps over the lazy dog", false)
            .expect("tokenize");
        let n = ids.len();
        assert!(n >= 2, "need ≥2 tokens to test the skip-tail path");

        // Path A: full-tail every token.
        let mut s_a = MetalSession::fresh(&ctx, &mm, n + 4).expect("session A");
        let mut last_a = vec![];
        for (i, &tid) in ids.iter().enumerate() {
            last_a = mf.single_token(tid, i as u32, &mut s_a).expect("forward A");
        }

        // Path B: no_tail for [0..n-1), full tail for the last token.
        let mut s_b = MetalSession::fresh(&ctx, &mm, n + 4).expect("session B");
        let mut last_b = vec![];
        for (i, &tid) in ids.iter().enumerate() {
            if i + 1 < n {
                mf.single_token_no_tail(tid, i as u32, &mut s_b)
                    .expect("forward B no_tail");
            } else {
                last_b = mf
                    .single_token(tid, i as u32, &mut s_b)
                    .expect("forward B tail");
            }
        }

        // Bit-exact match required: same kernels in the same order with
        // same inputs (no mat-mat half-staging on the no_tail path).
        assert_eq!(
            last_a.len(),
            last_b.len(),
            "logits length mismatch ({} vs {})",
            last_a.len(),
            last_b.len()
        );
        let mut max_abs = 0.0f32;
        for i in 0..last_a.len() {
            max_abs = max_abs.max((last_a[i] - last_b[i]).abs());
        }
        eprintln!("[no-tail-prefill] full-tail vs no-tail final logits max|Δ|={max_abs:.6e}");
        assert_eq!(
            max_abs, 0.0,
            "logits must match BIT-EXACTLY (max|Δ|={max_abs:e})"
        );

        // Multi-hidden capture must also be bit-exact across all prefill
        // positions. We accumulate the per-position hidden captures into
        // a host-side buffer (mirroring the bench's
        // `append_target_ctx_column_now` pattern) and compare path A vs
        // path B's accumulations.
        let arch = &mm.arch;
        let h = arch.hidden_size as usize;
        // Pick a few capture layers spanning the network (the H5 drafter
        // captures K=5; the 0.8B-F32 oracle has 36 blocks so 5 evenly
        // spaced layers exercises a realistic K).
        let capture_layers: Vec<u32> = vec![
            0,
            (mm.blocks.len() / 4) as u32,
            (mm.blocks.len() / 2) as u32,
            (3 * mm.blocks.len() / 4) as u32,
            (mm.blocks.len() - 1) as u32,
        ];
        let k = capture_layers.len();

        let h_dst_a = MetalTensor::zeros_f32(&ctx, vec![(k * h) as u64]).expect("h_dst_a");
        let h_dst_b = MetalTensor::zeros_f32(&ctx, vec![(k * h) as u64]).expect("h_dst_b");

        // Per-position accumulator: [n, k * h]
        let mut accum_a = vec![0.0f32; n * k * h];
        let mut accum_b = vec![0.0f32; n * k * h];

        let mut s2_a = MetalSession::fresh(&ctx, &mm, n + 4).expect("session2 A");
        for (i, &tid) in ids.iter().enumerate() {
            mf.single_token_with_multi_hidden(tid, i as u32, &mut s2_a, &capture_layers, &h_dst_a)
                .expect("multi_hidden A");
            unsafe {
                let src = h_dst_a.buffer.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(
                    src,
                    accum_a[i * k * h..(i + 1) * k * h].as_mut_ptr(),
                    k * h,
                );
            }
        }

        let mut s2_b = MetalSession::fresh(&ctx, &mm, n + 4).expect("session2 B");
        for (i, &tid) in ids.iter().enumerate() {
            if i + 1 < n {
                mf.single_token_with_multi_hidden_no_tail(
                    tid,
                    i as u32,
                    &mut s2_b,
                    &capture_layers,
                    &h_dst_b,
                )
                .expect("multi_hidden B no_tail");
            } else {
                mf.single_token_with_multi_hidden(
                    tid,
                    i as u32,
                    &mut s2_b,
                    &capture_layers,
                    &h_dst_b,
                )
                .expect("multi_hidden B tail");
            }
            unsafe {
                let src = h_dst_b.buffer.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(
                    src,
                    accum_b[i * k * h..(i + 1) * k * h].as_mut_ptr(),
                    k * h,
                );
            }
        }

        let mut max_abs_h = 0.0f32;
        let mut first_pos = usize::MAX;
        for i in 0..(n * k * h) {
            let d = (accum_a[i] - accum_b[i]).abs();
            if d > max_abs_h {
                max_abs_h = d;
                first_pos = i;
            }
        }
        eprintln!(
            "[no-tail-prefill] accumulated multi-hidden max|Δ|={max_abs_h:.6e} (first nonzero idx={first_pos})"
        );
        assert_eq!(
            max_abs_h, 0.0,
            "accumulated multi-hidden must match BIT-EXACTLY (max|Δ|={max_abs_h:e})"
        );
    }

    /// Validate a single full-attention block end-to-end on Metal vs the
    /// CPU oracle. Uses block 3 of Qwen3.5-0.8B-F32 (the first attn block,
    /// n_q=8, n_kv=2, head_dim=256, 4:1 GQA).
    #[test]
    fn metal_attn_block_matches_cpu() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[metal-attn] skipped — model missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        // Inputs: a synthetic but fixed residual stream (no need for a
        // real model state for block-level validation; we just need
        // identical inputs to CPU and GPU paths).
        let h = m.arch.hidden_size as usize;
        let initial_x: Vec<f32> = (0..h).map(|i| ((i % 31) as f32 - 15.0) * 0.02).collect();
        let position: u32 = 0;
        let attn_block_idx = 3usize; // first attn block in 0.8B
        // attn_idx_in_session is the 0-indexed count among ATTN blocks
        // before this one. block 3 is the first attn block, so 0.
        let attn_idx_in_session = 0usize;

        // CPU reference: replicate exactly what forward.rs:attn_step does.
        let cpu_x = run_cpu_attn_block_for_test(&g, &m, &initial_x, attn_block_idx, position);

        // Metal.
        let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
        let mf = MetalForward::new(&ctx, &mm);
        mf.set_residual_for_test(&mut s, &initial_x);
        let metal_x = mf
            .run_one_attn_block_for_test(attn_block_idx, attn_idx_in_session, position, &mut s)
            .expect("metal attn block");

        let max_abs = metal_x
            .iter()
            .zip(cpu_x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = metal_x
            .iter()
            .zip(cpu_x.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = metal_x.iter().map(|v| (*v as f64).powi(2)).sum();
        let nb: f64 = cpu_x.iter().map(|v| (*v as f64).powi(2)).sum();
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("[metal-attn-block3] hidden={h} max|Δ|={max_abs:.2e} cos={cos:.6}");
        assert!(max_abs < 1e-3, "attn block-3 drift {max_abs}");
        assert!(cos > 0.9999, "attn block-3 cos {cos}");
    }

    /// CPU reference: full-attn block (norm → attn → residual → post_norm
    /// → FFN → residual) replicated inline. Mirrors forward.rs's
    /// single_token block flow for an attn block.
    fn run_cpu_attn_block_for_test(
        gguf: &GgufFile,
        model: &Model<'_>,
        initial_x: &[f32],
        block_idx: usize,
        position: u32,
    ) -> Vec<f32> {
        let block = &model.blocks[block_idx];
        let ab = match block {
            crate::loader::Block::Attn(a) => a,
            _ => panic!("not an attn block"),
        };
        let arch = &model.arch;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_q = arch.n_q_heads as usize;
        let n_kv = arch.n_kv_heads as usize;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let group = n_q / n_kv;
        let theta = arch.rope_theta;

        let mut x = initial_x.to_vec();

        // Pre-mixer norm.
        let attn_norm_w =
            crate::codec::dequant_to_f32(ab.attn_norm, gguf.slice(ab.attn_norm)).unwrap();
        let cur = crate::forward::rms_norm_pub(&x, &attn_norm_w, super::RMS_EPS);

        // Q projection (2 * q_dim) → split.
        let q_w = crate::codec::dequant_to_f32(ab.q, gguf.slice(ab.q)).unwrap();
        let q_full = crate::forward::mat_vec_pub(&q_w, h, 2 * q_dim, &cur);
        let mut qcur = vec![0.0f32; q_dim];
        let mut gate = vec![0.0f32; q_dim];
        for hi in 0..n_q {
            let src = &q_full[hi * 2 * head_dim..(hi + 1) * 2 * head_dim];
            qcur[hi * head_dim..(hi + 1) * head_dim].copy_from_slice(&src[..head_dim]);
            gate[hi * head_dim..(hi + 1) * head_dim].copy_from_slice(&src[head_dim..]);
        }

        // Q-norm.
        let qnorm_w = crate::codec::dequant_to_f32(ab.q_norm, gguf.slice(ab.q_norm)).unwrap();
        for hi in 0..n_q {
            let s = hi * head_dim;
            let n = crate::forward::rms_norm_pub(&qcur[s..s + head_dim], &qnorm_w, super::RMS_EPS);
            qcur[s..s + head_dim].copy_from_slice(&n);
        }

        // K, V.
        let k_w = crate::codec::dequant_to_f32(ab.k, gguf.slice(ab.k)).unwrap();
        let v_w = crate::codec::dequant_to_f32(ab.v, gguf.slice(ab.v)).unwrap();
        let mut kcur = crate::forward::mat_vec_pub(&k_w, h, kv_dim, &cur);
        let vcur = crate::forward::mat_vec_pub(&v_w, h, kv_dim, &cur);

        // K-norm.
        let knorm_w = crate::codec::dequant_to_f32(ab.k_norm, gguf.slice(ab.k_norm)).unwrap();
        for hi in 0..n_kv {
            let s = hi * head_dim;
            let n = crate::forward::rms_norm_pub(&kcur[s..s + head_dim], &knorm_w, super::RMS_EPS);
            kcur[s..s + head_dim].copy_from_slice(&n);
        }

        // RoPE on Q and K (NEOX/IMROPE pairing for text positions).
        rope_in_place_local(&mut qcur, n_q, head_dim, n_rot, position, theta);
        rope_in_place_local(&mut kcur, n_kv, head_dim, n_rot, position, theta);

        // Single-token cache: KV is just the current step.
        // Attention with one position (position itself).
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; q_dim];
        for qh in 0..n_q {
            let kvh = qh / group;
            let q_slice = &qcur[qh * head_dim..(qh + 1) * head_dim];
            let k_slice = &kcur[kvh * head_dim..(kvh + 1) * head_dim];
            let v_slice = &vcur[kvh * head_dim..(kvh + 1) * head_dim];

            let mut s = 0.0f32;
            for i in 0..head_dim {
                s += q_slice[i] * k_slice[i];
            }
            let _score = s * scale;
            // softmax over a single value = 1.0 → out = v
            for i in 0..head_dim {
                attn_out[qh * head_dim + i] = v_slice[i];
            }
        }

        // Apply gated-attention sigmoid gate.
        for i in 0..attn_out.len() {
            let sg = 1.0 / (1.0 + (-gate[i]).exp());
            attn_out[i] *= sg;
        }

        // Output projection.
        let o_w = crate::codec::dequant_to_f32(ab.o, gguf.slice(ab.o)).unwrap();
        let mixer_out = crate::forward::mat_vec_pub(&o_w, q_dim, h, &attn_out);

        // Residual #1.
        for (xi, mo) in x.iter_mut().zip(mixer_out.iter()) {
            *xi += *mo;
        }

        // Pre-FFN norm.
        let post_norm_w = crate::codec::dequant_to_f32(
            ab.post_attention_norm,
            gguf.slice(ab.post_attention_norm),
        )
        .unwrap();
        let cur = crate::forward::rms_norm_pub(&x, &post_norm_w, super::RMS_EPS);

        // FFN.
        let f = arch.intermediate_size as usize;
        let g_w = crate::codec::dequant_to_f32(ab.ffn_gate, gguf.slice(ab.ffn_gate)).unwrap();
        let u_w = crate::codec::dequant_to_f32(ab.ffn_up, gguf.slice(ab.ffn_up)).unwrap();
        let d_w = crate::codec::dequant_to_f32(ab.ffn_down, gguf.slice(ab.ffn_down)).unwrap();
        let gated = crate::forward::mat_vec_pub(&g_w, h, f, &cur);
        let upped = crate::forward::mat_vec_pub(&u_w, h, f, &cur);
        let mut hidden = vec![0.0f32; f];
        for i in 0..f {
            let g = gated[i];
            let silu_g = g / (1.0 + (-g).exp());
            hidden[i] = silu_g * upped[i];
        }
        let ffn_out = crate::forward::mat_vec_pub(&d_w, f, h, &hidden);

        // Residual #2.
        for (xi, fo) in x.iter_mut().zip(ffn_out.iter()) {
            *xi += *fo;
        }
        x
    }

    fn rope_in_place_local(
        buf: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        n_rot: usize,
        position: u32,
        theta_base: f32,
    ) {
        let pos = position as f32;
        let half = n_rot / 2;
        for hi in 0..n_heads {
            let h_off = hi * head_dim;
            for i in 0..half {
                let exponent = (2 * i) as f32 / n_rot as f32;
                let freq = pos / theta_base.powf(exponent);
                let (s, c) = freq.sin_cos();
                let a = buf[h_off + i];
                let b = buf[h_off + i + half];
                buf[h_off + i] = a * c - b * s;
                buf[h_off + i + half] = a * s + b * c;
            }
        }
    }

    /// CPU reference: replicate exactly what forward.rs:single_token does
    /// for block 0 of Qwen3.5-0.8B (a GDN block), starting from
    /// `initial_x` (the token embedding).
    fn run_cpu_block0_for_test(gguf: &GgufFile, model: &Model<'_>, initial_x: &[f32]) -> Vec<f32> {
        let f = Forward::new(gguf, model);
        // Use Forward's public single_token but at position 0 with token
        // id derived from initial_x: too indirect. Easier path: take the
        // raw GDN-block computation and replicate it inline.
        //
        // forward.rs's `single_token` already does exactly this. The
        // simplest validation is to run it in full and compare logits —
        // but that requires the full attn block, which Metal doesn't
        // have yet. So instead we replicate just block 0 here.
        //
        // Block 0 in 0.8B is a GDN block. The CPU computation is in
        // forward.rs lines 290-510-ish. We recreate the same flow with
        // the public helpers in `forward`.
        let mut x = initial_x.to_vec();
        let block = &model.blocks[0];
        let gb = match block {
            crate::loader::Block::Gdn(g) => g,
            _ => panic!("block 0 is not GDN"),
        };

        // Pre-mixer norm.
        let attn_norm_w =
            crate::codec::dequant_to_f32(gb.attn_norm, gguf.slice(gb.attn_norm)).unwrap();
        let cur = crate::forward::rms_norm_pub(&x, &attn_norm_w, super::RMS_EPS);

        // GDN inner. The cleanest way to replicate is to hand-call
        // Forward's gdn_step. It takes a GdnState; we'll build a fresh
        // one and ignore the state output.
        let mut state = crate::forward::GdnState::fresh(model);
        let mixer_out = call_gdn_step_directly(&f, gb, &cur, &mut state);

        // Residual #1.
        for (xi, mi) in x.iter_mut().zip(mixer_out.iter()) {
            *xi += *mi;
        }

        // Pre-FFN norm.
        let post_norm_w = crate::codec::dequant_to_f32(
            gb.post_attention_norm,
            gguf.slice(gb.post_attention_norm),
        )
        .unwrap();
        let cur = crate::forward::rms_norm_pub(&x, &post_norm_w, super::RMS_EPS);

        // FFN.
        let arch = &model.arch;
        let h = arch.hidden_size as usize;
        let fdim = arch.intermediate_size as usize;
        let g_w = crate::codec::dequant_to_f32(gb.ffn_gate, gguf.slice(gb.ffn_gate)).unwrap();
        let u_w = crate::codec::dequant_to_f32(gb.ffn_up, gguf.slice(gb.ffn_up)).unwrap();
        let d_w = crate::codec::dequant_to_f32(gb.ffn_down, gguf.slice(gb.ffn_down)).unwrap();

        let gated = crate::forward::mat_vec_pub(&g_w, h, fdim, &cur);
        let upped = crate::forward::mat_vec_pub(&u_w, h, fdim, &cur);
        let mut hidden = vec![0.0f32; fdim];
        for i in 0..fdim {
            let g = gated[i];
            let silu_g = g / (1.0 + (-g).exp());
            hidden[i] = silu_g * upped[i];
        }
        let ffn_out = crate::forward::mat_vec_pub(&d_w, fdim, h, &hidden);

        // Residual #2.
        for (xi, fo) in x.iter_mut().zip(ffn_out.iter()) {
            *xi += *fo;
        }
        x
    }

    /// Use a private CPU-only path to run forward::Forward's gdn_step on
    /// block 0. We can't call it directly because it's private to
    /// Forward; the test re-implements the same math inline.
    // CPU GDN inline re-impl: strided 2D state access
    // `state.ssm[0][s_off + dv * head_dim + dk]`. Iterator rewrite
    // hides the stride math (the whole point of this reference).
    #[allow(clippy::needless_range_loop)]
    fn call_gdn_step_directly(
        f: &Forward,
        gb: &crate::loader::GdnBlock,
        x: &[f32],
        state: &mut crate::forward::GdnState,
    ) -> Vec<f32> {
        // Public re-exports added below in forward.rs would let us avoid
        // this. For now, since Forward's gdn_step is private, we run the
        // *full* Forward::single_token and extract the post-block-0
        // residual. That requires a hook in Forward we don't have.
        //
        // Pragmatic shortcut: re-implement gdn_step here using the public
        // mat_vec_pub and matching the exact CPU forward path. This is
        // ~50 lines of duplication but isolates the test from Forward's
        // internals.
        let arch = &f.model.arch;
        let h = arch.hidden_size as usize;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        let conv_dim = 2 * n_k * head_dim + n_v * head_dim;
        let conv_kernel = arch.gdn_conv_kernel as usize;
        let v_dim = n_v * head_dim;

        let qkv_w =
            crate::codec::dequant_to_f32(gb.in_proj_qkv, f.gguf.slice(gb.in_proj_qkv)).unwrap();
        let qkv = crate::forward::mat_vec_pub(&qkv_w, h, conv_dim, x);
        let z_w = crate::codec::dequant_to_f32(gb.in_proj_z, f.gguf.slice(gb.in_proj_z)).unwrap();
        let z = crate::forward::mat_vec_pub(&z_w, h, v_dim, x);

        let beta_w =
            crate::codec::dequant_to_f32(gb.beta_proj, f.gguf.slice(gb.beta_proj)).unwrap();
        let mut beta = crate::forward::mat_vec_pub(&beta_w, h, n_v, x);
        for v in beta.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
        let alpha_w =
            crate::codec::dequant_to_f32(gb.alpha_proj, f.gguf.slice(gb.alpha_proj)).unwrap();
        let mut alpha = crate::forward::mat_vec_pub(&alpha_w, h, n_v, x);
        let dt = crate::codec::dequant_to_f32(gb.dt_bias, f.gguf.slice(gb.dt_bias)).unwrap();
        for (a, &dti) in alpha.iter_mut().zip(dt.iter()) {
            *a += dti;
        }
        let a_log = crate::codec::dequant_to_f32(gb.a_log, f.gguf.slice(gb.a_log)).unwrap();
        let mut g = vec![0.0f32; n_v];
        for i in 0..n_v {
            let sp = if alpha[i] > 20.0 {
                alpha[i]
            } else if alpha[i] < -20.0 {
                alpha[i].exp()
            } else {
                (1.0 + alpha[i].exp()).ln()
            };
            g[i] = sp * a_log[i];
        }

        let conv_w = crate::codec::dequant_to_f32(gb.conv1d, f.gguf.slice(gb.conv1d)).unwrap();
        let kmin1 = conv_kernel - 1;
        let mut conv_input = vec![0.0f32; conv_kernel * conv_dim];
        for t in 0..kmin1 {
            conv_input[t * conv_dim..(t + 1) * conv_dim]
                .copy_from_slice(&state.conv[0][t * conv_dim..(t + 1) * conv_dim]);
        }
        conv_input[kmin1 * conv_dim..].copy_from_slice(&qkv);

        let mut conv_out = vec![0.0f32; conv_dim];
        for c in 0..conv_dim {
            let mut sm = 0.0f32;
            for k in 0..conv_kernel {
                sm += conv_w[c * conv_kernel + k] * conv_input[k * conv_dim + c];
            }
            conv_out[c] = sm / (1.0 + (-sm).exp());
        }
        // (slide buffer; not asked for in test, just for completeness it
        // would happen here, but state.conv is local to this fn here)
        for t in 0..kmin1 - 1 {
            for i in 0..conv_dim {
                state.conv[0][t * conv_dim + i] = state.conv[0][(t + 1) * conv_dim + i];
            }
        }
        let last = (kmin1 - 1) * conv_dim;
        state.conv[0][last..last + conv_dim].copy_from_slice(&qkv);

        // Split conv_out → q,k,v.
        let q_full = conv_out[0..n_k * head_dim].to_vec();
        let k_full = conv_out[n_k * head_dim..2 * n_k * head_dim].to_vec();
        let v_full = conv_out[2 * n_k * head_dim..].to_vec();

        // Per-head L2 norm of Q and K.
        let mut q_full = q_full.clone();
        let mut k_full = k_full.clone();
        for hi in 0..n_k {
            let off = hi * head_dim;
            let sq: f32 = q_full[off..off + head_dim].iter().map(|v| v * v).sum();
            let scale = 1.0 / sq.sqrt().max(super::RMS_EPS);
            for i in 0..head_dim {
                q_full[off + i] *= scale;
            }
            let sq: f32 = k_full[off..off + head_dim].iter().map(|v| v * v).sum();
            let scale = 1.0 / sq.sqrt().max(super::RMS_EPS);
            for i in 0..head_dim {
                k_full[off + i] *= scale;
            }
        }

        // Recurrence.
        let mut o = vec![0.0f32; v_dim];
        for hi in 0..n_v {
            let hk = hi % n_k;
            let s_off = hi * head_dim * head_dim;
            let q_h = &q_full[hk * head_dim..(hk + 1) * head_dim];
            let k_h = &k_full[hk * head_dim..(hk + 1) * head_dim];
            let v_h = &v_full[hi * head_dim..(hi + 1) * head_dim];
            let g_h = g[hi].exp();
            let b_h = beta[hi];
            for j in 0..head_dim * head_dim {
                state.ssm[0][s_off + j] *= g_h;
            }
            let mut sk = vec![0.0f32; head_dim];
            for dv in 0..head_dim {
                let mut sm = 0.0f32;
                for dk in 0..head_dim {
                    sm += state.ssm[0][s_off + dv * head_dim + dk] * k_h[dk];
                }
                sk[dv] = sm;
            }
            for dv in 0..head_dim {
                let coeff = b_h * (v_h[dv] - sk[dv]);
                for dk in 0..head_dim {
                    state.ssm[0][s_off + dv * head_dim + dk] += coeff * k_h[dk];
                }
            }
            let scale = 1.0 / (head_dim as f32).sqrt();
            for dv in 0..head_dim {
                let mut sm = 0.0f32;
                for dk in 0..head_dim {
                    sm += state.ssm[0][s_off + dv * head_dim + dk] * q_h[dk];
                }
                o[hi * head_dim + dv] = sm * scale;
            }
        }

        // RMSNormGated: norm(o) * silu(z), per-head.
        let norm_w = crate::codec::dequant_to_f32(gb.norm, f.gguf.slice(gb.norm)).unwrap();
        let mut gated = vec![0.0f32; v_dim];
        for hi in 0..n_v {
            let off = hi * head_dim;
            let normed =
                crate::forward::rms_norm_pub(&o[off..off + head_dim], &norm_w, super::RMS_EPS);
            for i in 0..head_dim {
                let zi = z[off + i];
                let silu_z = zi / (1.0 + (-zi).exp());
                gated[off + i] = normed[i] * silu_z;
            }
        }

        // Output projection.
        let out_w = crate::codec::dequant_to_f32(gb.out_proj, f.gguf.slice(gb.out_proj)).unwrap();
        crate::forward::mat_vec_pub(&out_w, v_dim, h, &gated)
    }

    /// **H2.0 — prefix-cache correctness spike.** Validates that
    /// snapshot+restore at a prefix boundary yields the same final
    /// logits as cold prefill of the full sequence.
    ///
    /// Mechanism (deliberately minimal — no public API yet, just to
    /// prove the principle):
    ///   1. Prefill prompt into session A (cold path).
    ///   2. Capture last-position logits from A.
    ///   3. Make a fresh session B, prefill ONLY the prefix into B.
    ///   4. Snapshot B's state by raw-byte-cloning all six per-layer
    ///      MTLBuffers via `MTLBuffer.contents()` → `Vec<u8>`.
    ///   5. Make a fresh session C, restore the snapshot bytes into C's
    ///      buffers (also via raw `contents()` write).
    ///   6. Run the suffix tokens through C, capture last-position logits.
    ///   7. Assert cos(A_logits, C_logits) ≥ 0.99999 and same argmax.
    ///
    /// If this passes, snapshot mechanism is sound and we can build the
    /// real packed-arena LRU on top. If it fails, H2 is dead and we
    /// pivot to backend sampler / attention-surround fusions.
    ///
    /// Falsifiable kill criterion (per codex's H2 review): cos < 0.99999
    /// or argmax mismatch at any tested prefix length.
    #[test]
    #[ignore]
    fn h2_prefix_cache_correctness_spike() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[h2-spike] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let mf = MetalForward::new(&ctx, &mm);

        // Use a longer prompt so we can split it into prefix+suffix.
        let prompt =
            "The quick brown fox jumps over the lazy dog and then runs into the deep forest";
        let ids = tok.encode(prompt, false).expect("tokenize");
        eprintln!("[h2-spike] {} prompt tokens: {ids:?}", ids.len());
        // Test multiple prefix split points (per codex's kill criteria).
        let prefix_lens = [3usize, 5, 8, ids.len() - 1];

        // Read all bytes from a MetalTensor's MTLBuffer (shared storage).
        let read_bytes = |t: &MetalTensor| -> Vec<u8> {
            let n = t.n_bytes() as usize;
            let mut out = vec![0u8; n];
            unsafe {
                let src = (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize);
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
            }
            out
        };
        let write_bytes = |t: &MetalTensor, bytes: &[u8]| {
            assert_eq!(
                bytes.len() as u64,
                t.n_bytes(),
                "snapshot byte size mismatch"
            );
            unsafe {
                let dst = (t.buffer.contents().as_ptr() as *mut u8).add(t.offset as usize);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
            }
        };

        for &prefix_len in &prefix_lens {
            assert!(prefix_len < ids.len() && prefix_len > 0);
            let suffix_len = ids.len() - prefix_len;
            eprintln!("[h2-spike] === prefix_len={prefix_len} suffix_len={suffix_len} ===");

            // ---- 1+2: Cold prefill of full prompt. Capture last logits. ----
            let mut sess_cold = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session A");
            let mut logits_cold = vec![];
            for (i, &tid) in ids.iter().enumerate() {
                logits_cold = mf
                    .single_token(tid, i as u32, &mut sess_cold)
                    .expect("cold");
            }
            let n = logits_cold.len();

            // ---- 3: Fresh session, prefill only the prefix. ----
            let mut sess_pre = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session B");
            for (i, &tid) in ids.iter().take(prefix_len).enumerate() {
                let _ = mf.single_token(tid, i as u32, &mut sess_pre).expect("pre");
            }

            // ---- 4: Snapshot all six per-layer state bytes. ----
            let snap_kv_k: Vec<Vec<u8>> = sess_pre.kv_k.iter().map(read_bytes).collect();
            let snap_kv_v: Vec<Vec<u8>> = sess_pre.kv_v.iter().map(read_bytes).collect();
            let snap_kv_n_pos = sess_pre.kv_n_pos.clone();
            let snap_gdn_conv: Vec<Vec<u8>> = sess_pre.gdn_conv.iter().map(read_bytes).collect();
            let snap_gdn_state: Vec<Vec<u8>> = sess_pre.gdn_state.iter().map(read_bytes).collect();

            let total_snap_bytes: usize = snap_kv_k.iter().map(|v| v.len()).sum::<usize>()
                + snap_kv_v.iter().map(|v| v.len()).sum::<usize>()
                + snap_gdn_conv.iter().map(|v| v.len()).sum::<usize>()
                + snap_gdn_state.iter().map(|v| v.len()).sum::<usize>();
            eprintln!(
                "[h2-spike]   snapshot size: {:.1} MB ({} attn KV + {} GDN conv + {} GDN state buffers)",
                total_snap_bytes as f64 / 1e6,
                snap_kv_k.len() * 2,
                snap_gdn_conv.len(),
                snap_gdn_state.len()
            );

            // ---- 5: Fresh session, restore the snapshot bytes. ----
            let mut sess_restored =
                MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session C");
            for (i, src) in snap_kv_k.iter().enumerate() {
                write_bytes(&sess_restored.kv_k[i], src);
            }
            for (i, src) in snap_kv_v.iter().enumerate() {
                write_bytes(&sess_restored.kv_v[i], src);
            }
            sess_restored.kv_n_pos.copy_from_slice(&snap_kv_n_pos);
            for (i, src) in snap_gdn_conv.iter().enumerate() {
                write_bytes(&sess_restored.gdn_conv[i], src);
            }
            for (i, src) in snap_gdn_state.iter().enumerate() {
                write_bytes(&sess_restored.gdn_state[i], src);
            }

            // ---- 6: Run suffix tokens through restored session. ----
            let mut logits_restored = vec![];
            for k in 0..suffix_len {
                let pos = (prefix_len + k) as u32;
                let tid = ids[prefix_len + k];
                logits_restored = mf
                    .single_token(tid, pos, &mut sess_restored)
                    .expect("restored forward");
            }

            // ---- 7: Compare last-position logits. ----
            assert_eq!(
                logits_cold.len(),
                logits_restored.len(),
                "logits len mismatch"
            );
            let mut max_abs = 0.0f32;
            let mut argmax_cold = 0usize;
            let mut argmax_restored = 0usize;
            let mut max_cold = f32::NEG_INFINITY;
            let mut max_restored = f32::NEG_INFINITY;
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for i in 0..n {
                let d = (logits_cold[i] - logits_restored[i]).abs();
                max_abs = max_abs.max(d);
                if logits_cold[i] > max_cold {
                    max_cold = logits_cold[i];
                    argmax_cold = i;
                }
                if logits_restored[i] > max_restored {
                    max_restored = logits_restored[i];
                    argmax_restored = i;
                }
                dot += logits_cold[i] as f64 * logits_restored[i] as f64;
                na += (logits_cold[i] as f64).powi(2);
                nb += (logits_restored[i] as f64).powi(2);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
            eprintln!(
                "[h2-spike]   cos={cos:.7} max|Δ|={max_abs:.4} argmax: cold={argmax_cold} restored={argmax_restored} {}",
                if argmax_cold == argmax_restored {
                    "✓"
                } else {
                    "✗ MISMATCH"
                }
            );
            assert_eq!(
                argmax_cold, argmax_restored,
                "argmax mismatch at prefix_len={prefix_len}: cold={argmax_cold} restored={argmax_restored}"
            );
            assert!(
                cos > 0.99999,
                "cos={cos} below 0.99999 at prefix_len={prefix_len}"
            );
        }
        eprintln!("[h2-spike] ALL prefix splits passed — H2 mechanism validated.");
    }

    /// **H2.1 — packed-arena snapshot via the production API.**
    /// Same correctness guarantee as the H2.0 spike, but uses the new
    /// `MetalSession::snapshot` / `restore_from` methods on top of
    /// `SessionSnapshot` arenas. Validates the production API matches
    /// the bit-exact spike.
    ///
    /// Also reports snapshot/restore wall-clock so we can compare to
    /// the bandwidth model (codex predicted ~0.9-2 ms; let's see).
    #[test]
    #[ignore]
    fn h2_packed_arena_snapshot_matches_cold() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[h2-arena] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => panic!("init failed: {e}"),
        };

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let mf = MetalForward::new(&ctx, &mm);

        // Larger prompt for more meaningful prefix lengths.
        let prompt = "The quick brown fox jumps over the lazy dog and then runs into the deep dark forest where it meets a wise old owl who teaches it the meaning of life";
        let ids = tok.encode(prompt, false).expect("tokenize");
        eprintln!("[h2-arena] {} prompt tokens", ids.len());

        for &prefix_len in &[1usize, 5, 16, ids.len() - 1] {
            assert!(prefix_len < ids.len() && prefix_len > 0);
            let suffix_len = ids.len() - prefix_len;
            eprintln!("[h2-arena] === prefix_len={prefix_len} suffix_len={suffix_len} ===");

            // Cold reference.
            let mut sess_cold = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("A");
            let mut logits_cold = vec![];
            for (i, &tid) in ids.iter().enumerate() {
                logits_cold = mf
                    .single_token(tid, i as u32, &mut sess_cold)
                    .expect("cold");
            }

            // Build a snapshot via the production API.
            let mut sess_pre = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("B");
            let mut last_pre_logits = vec![];
            for (i, &tid) in ids.iter().take(prefix_len).enumerate() {
                last_pre_logits = mf.single_token(tid, i as u32, &mut sess_pre).expect("pre");
            }
            let identity = sess_pre.snapshot_identity(0xDEADBEEF, 0xCAFE);
            let prefix_tokens: Vec<i32> = ids[..prefix_len].to_vec();

            let t = std::time::Instant::now();
            let snap = sess_pre.snapshot(identity, prefix_tokens, Some(last_pre_logits));
            let snap_ms = t.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "[h2-arena]   snapshot: {:.2} MB in {:.2} ms = {:.1} GB/s",
                snap.n_bytes() as f64 / 1e6,
                snap_ms,
                (snap.n_bytes() as f64 / 1e9) / (snap_ms / 1e3)
            );

            // Restore into a fresh session via the production API.
            let mut sess_restored = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("C");
            let t = std::time::Instant::now();
            sess_restored.restore_from(&snap).expect("restore");
            let restore_ms = t.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "[h2-arena]   restore:  {:.2} ms = {:.1} GB/s",
                restore_ms,
                (snap.n_bytes() as f64 / 1e9) / (restore_ms / 1e3)
            );

            // Run suffix tokens through restored session.
            let mut logits_restored = vec![];
            for k in 0..suffix_len {
                let pos = (prefix_len + k) as u32;
                let tid = ids[prefix_len + k];
                logits_restored = mf
                    .single_token(tid, pos, &mut sess_restored)
                    .expect("restored forward");
            }

            // Compare last-position logits.
            let n = logits_cold.len();
            assert_eq!(n, logits_restored.len());
            let mut max_abs = 0.0f32;
            let mut argmax_cold = 0usize;
            let mut argmax_restored = 0usize;
            let mut max_cold = f32::NEG_INFINITY;
            let mut max_restored = f32::NEG_INFINITY;
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for i in 0..n {
                let d = (logits_cold[i] - logits_restored[i]).abs();
                max_abs = max_abs.max(d);
                if logits_cold[i] > max_cold {
                    max_cold = logits_cold[i];
                    argmax_cold = i;
                }
                if logits_restored[i] > max_restored {
                    max_restored = logits_restored[i];
                    argmax_restored = i;
                }
                dot += logits_cold[i] as f64 * logits_restored[i] as f64;
                na += (logits_cold[i] as f64).powi(2);
                nb += (logits_restored[i] as f64).powi(2);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
            eprintln!(
                "[h2-arena]   cos={cos:.7} max|Δ|={max_abs:.4} argmax: cold={argmax_cold} restored={argmax_restored} {}",
                if argmax_cold == argmax_restored {
                    "✓"
                } else {
                    "✗ MISMATCH"
                }
            );
            assert_eq!(argmax_cold, argmax_restored);
            assert!(cos > 0.99999, "cos={cos} below threshold");
        }

        // Identity-mismatch refusal.
        {
            let sess = MetalSession::fresh(&ctx, &mm, 64).expect("ident");
            let bogus_identity = SnapshotIdentity {
                model_id: 0,
                tokenizer_id: 0,
                layout_version: 999,
                n_attn_layers: 0,
                n_gdn_layers: 0,
                kv_dim_elements: 0,
                kv_bytes_per_token: 0,
                gdn_state_elements_per_layer: 0,
                gdn_conv_elements_per_layer: 0,
            };
            let bad_snap = SessionSnapshot {
                identity: bogus_identity,
                prefix_tokens: vec![],
                kv_n_pos: vec![],
                kv_k_arena: vec![],
                kv_v_arena: vec![],
                gdn_conv_arena: vec![],
                gdn_state_arena: vec![],
                final_logits: None,
            };
            let mut s2 = sess;
            assert!(
                s2.restore_from(&bad_snap).is_err(),
                "identity mismatch must error"
            );
            eprintln!("[h2-arena]   identity-mismatch refusal: ✓");
        }

        eprintln!("[h2-arena] all prefix splits passed via packed-arena API.");
    }
}
