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
    Buffer, GgufBackingEligibility, KernelEncoder, MetalContext, MetalError, MetalGgufBacking,
    MetalTensor, MetalTensorProvenance, MetalTimestampSampleBuffer, RetainedStorageDisposition,
    RetainedStorageFallback, RetainedStoragePlan, attn_v4_choose_nwg, attn_v4_choose_tile_c,
    encode_add_inplace_f32, encode_argmax_f32, encode_argmax_f32_greedy,
    encode_attn_decode_f16kv_f32, encode_attn_decode_v4_f32, encode_axpy_f32,
    encode_axpy_scalar_f32, encode_dot_sigmoid_f32, encode_ffn_swiglu_q4_K_f32, encode_fill_f32,
    encode_gdn_decay_chain_f32, encode_gdn_step_decay_f32, encode_get_rows_f32,
    encode_l2_norm_batched_f32, encode_l2_norm_pair_batched_f32, encode_mat_vec_f32,
    encode_mat_vec_f32_sigmoid, encode_mat_vec_q4_k_f32, encode_mat_vec_q5_k_f32,
    encode_mat_vec_q6_k_f32, encode_moe_down_bf16_f32, encode_moe_down_f32_f32,
    encode_moe_down_iq4_xs_f32, encode_moe_down_iq4_xs_f32_fast, encode_moe_down_q4_K_f32,
    encode_moe_down_q5_K_f32, encode_moe_down_weighted_sum_q5_K_f32_packed_slots,
    encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2,
    encode_moe_down_weighted_sum_q6_K_f32, encode_moe_down_weighted_sum_q8_0_f32,
    encode_moe_grouped_finalizer_f32, encode_moe_mat_vec_bf16_f32, encode_moe_mat_vec_f32,
    encode_moe_mat_vec_iq3_s_f32, encode_moe_mat_vec_iq3_xxs_f32, encode_moe_mat_vec_q5_K_f32,
    encode_moe_shared_accum_resid_f32, encode_moe_swiglu_iq3_s_f32,
    encode_moe_swiglu_iq3_s_f32_fast, encode_moe_swiglu_iq3_xxs_f32,
    encode_moe_swiglu_iq3_xxs_f32_fast, encode_moe_swiglu_q4_K_f32, encode_moe_swiglu_q6_K_f32,
    encode_moe_swiglu_q8_0_f32, encode_moe_weighted_sum_f32, encode_mul_f32,
    encode_residual_rms_norm_mul_f32, encode_rms_norm_batched_f32,
    encode_rms_norm_batched_src_strided_f32, encode_rms_norm_mul_f32, encode_rmsnorm_gated_f32,
    encode_rope_neox_f32, encode_rope_neox_pair_f32, encode_scatter_offset_f32_to_f16_kv,
    encode_scatter_offset_f32_to_q8_0_kv, encode_shared_swiglu_q8_0_f32, encode_sigmoid_f32,
    encode_sigmoid_mul_gate_strided_f32, encode_silu_mul_f32, encode_split_q_gate_f32,
    encode_ssm_conv_silu_f32, encode_topk_logits_softmax_dot_sigmoid_f32,
    encode_topk_logits_softmax_f32, encode_topk_logits_softmax_parallel_f32,
    evaluate_metal_memory_admission, host_page_size_bytes, plan_retained_storage,
};
use crate::model::{Arch, ArchKind};
use crate::sampling::{BoundedTopKEvidence, GreedySelection, SampledToken, Sampler, SamplingError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    mem::MaybeUninit,
    sync::OnceLock,
};

/// Max NWG (split-K partitions) the v4 dispatcher will ever request.
/// Sets the size of session-resident partial buffers; see
/// `attn_v4_choose_nwg` for the selection heuristic.
pub const ATTN_V4_MAX_NWG: usize = 1024;
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

fn prefill_attn_fused_qkv_g8_enabled() -> bool {
    matches!(
        std::env::var("QWEN_PREFILL_ATTN_FUSED_QKV_G8").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
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
    if !numerator.is_multiple_of(denominator) {
        return Err(alloc_shape_error(detail));
    }
    Ok(numerator / denominator)
}

crate::env_flag!(default_off kv_q8_flag, "QWEN_KV_Q8");

pub(crate) fn kv_cache_dtype_for_arch(arch: &crate::model::Arch) -> GgmlType {
    let enabled = kv_q8_flag();
    let group = (arch.n_q_heads / arch.n_kv_heads.max(1)) as usize;
    if enabled
        && arch.attn_head_dim == 256
        && ((arch.kind == ArchKind::Dense && group == 6)
            || (arch.kind == ArchKind::Moe && group == 8))
    {
        GgmlType::Q8_0
    } else {
        GgmlType::F16
    }
}

// Cached QWEN_* boolean knobs. Polarity is part of the declaration:
// `default_on` flags ship enabled and the env var is a rollback lever;
// `default_off` flags are opt-in diagnostics/experiments. Semantics are
// identical to the hand-rolled OnceLock blocks these replace (latch on
// first read; unrecognized values resolve to the default). See
// `crate::env_flag` for the shared parser and the macro definition.
crate::env_flag!(default_on concurrent_gdn_moe_decode_enabled, "QWEN_DECODE_MOE_CONCURRENT_GDN");
crate::env_flag!(default_on concurrent_gdn_dense_decode_enabled, "QWEN_DECODE_DENSE_CONCURRENT_GDN");
crate::env_flag!(default_on concurrent_shared_moe_decode_enabled, "QWEN_DECODE_MOE_CONCURRENT_SHARED");
crate::env_flag!(default_on decode_shared_swiglu_q8_enabled, "QWEN_DECODE_SHARED_SWIGLU_Q8");
crate::env_flag!(default_on decode_moe_iq3_fused_swiglu_enabled, "QWEN_DECODE_MOE_IQ3_FUSED_SWIGLU");
crate::env_flag!(default_on decode_moe_iq3_fast_swiglu_enabled, "QWEN_DECODE_MOE_IQ3_FAST_SWIGLU");
crate::env_flag!(default_on decode_moe_q5_down_fused_enabled, "QWEN_DECODE_MOE_Q5_DOWN_FUSED");
crate::env_flag!(default_on decode_moe_iq4_down_fast_enabled, "QWEN_DECODE_MOE_IQ4_DOWN_FAST");
crate::env_flag!(default_on decode_moe_q5_down_k512_r2_enabled, "QWEN_DECODE_MOE_Q5_DOWN_K512_R2");
crate::env_flag!(default_on decode_moe_fused_finalizer_enabled, "QWEN_DECODE_MOE_FUSED_FINALIZER");
crate::env_flag!(default_on decode_moe_grouped_finalizer_enabled, "QWEN_DECODE_MOE_GROUPED_FINALIZER");
crate::env_flag!(default_on decode_attn_sigmoid_mul_enabled, "QWEN_DECODE_ATTN_SIGMOID_MUL");
crate::env_flag!(default_off moe_router_f16_enabled, "QWEN_MOE_ROUTER_F16");
crate::env_flag!(default_off decode_gdn_noop_front_enabled, "QWEN_DECODE_GDN_NOOP_FRONT");
crate::env_flag!(default_off decode_gdn_noop_out_enabled, "QWEN_DECODE_GDN_NOOP_OUT");
crate::env_flag!(default_on decode_gdn_fused_beta_proj_enabled, "QWEN_DECODE_GDN_FUSED_BETA_PROJ");
crate::env_flag!(default_on decode_rope_pair_enabled, "QWEN_DECODE_ROPE_PAIR");

fn gdn_beta_projection_fused(gb: &MetalGdnBlock) -> bool {
    decode_gdn_fused_beta_proj_enabled()
        && gb.beta_proj.dtype == GgmlType::F32
        && !decode_gdn_noop_beta_enabled()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeQuantEmbeddingMode {
    Auto,
    Forced,
    Disabled,
    Invalid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeQuantEmbeddingSelection {
    AutoPromoted,
    Forced,
    AutoUnpromoted,
    RollbackDisabled,
    InvalidDisabled,
    Unsupported,
}

impl NativeQuantEmbeddingSelection {
    fn uses_native(self) -> bool {
        matches!(self, Self::AutoPromoted | Self::Forced)
    }

    fn label(self) -> &'static str {
        match self {
            Self::AutoPromoted => "auto-promoted",
            Self::Forced => "forced",
            Self::AutoUnpromoted => "auto-unpromoted",
            Self::RollbackDisabled => "rollback-disabled",
            Self::InvalidDisabled => "invalid-disabled",
            Self::Unsupported => "unsupported",
        }
    }
}

fn parse_native_quant_embedding_mode(value: Option<&str>) -> NativeQuantEmbeddingMode {
    match value {
        None => NativeQuantEmbeddingMode::Auto,
        Some(value) if crate::env_flag::env_value_truthy(value) => NativeQuantEmbeddingMode::Forced,
        Some(value) if crate::env_flag::env_value_falsy(value) => {
            NativeQuantEmbeddingMode::Disabled
        }
        Some(_) => NativeQuantEmbeddingMode::Invalid,
    }
}

fn native_quant_embedding_mode() -> NativeQuantEmbeddingMode {
    static MODE: std::sync::OnceLock<NativeQuantEmbeddingMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("QWEN_NATIVE_QUANT_EMBED") {
            Ok(value) => parse_native_quant_embedding_mode(Some(&value)),
            Err(std::env::VarError::NotPresent) => NativeQuantEmbeddingMode::Auto,
            Err(std::env::VarError::NotUnicode(_)) => NativeQuantEmbeddingMode::Invalid,
        };
        if mode == NativeQuantEmbeddingMode::Invalid {
            eprintln!(
                "[metal-load] invalid QWEN_NATIVE_QUANT_EMBED value; disabling native embeddings"
            );
        }
        mode
    })
}

fn native_quant_embedding_supported(dtype: GgmlType, shape: &[u64]) -> bool {
    shape.len() == 2
        && shape[0] > 0
        && shape[1] > 0
        && ((dtype == GgmlType::Q4_K && shape[0].is_multiple_of(256))
            || (dtype == GgmlType::Q8_0 && shape[0].is_multiple_of(32)))
}

fn native_quant_embedding_default_promoted(
    arch: &crate::model::Arch,
    tied_embeddings: bool,
    mtp_present: bool,
    dtype: GgmlType,
    shape: &[u64],
) -> bool {
    if tied_embeddings || mtp_present || shape != [arch.hidden_size as u64, arch.vocab_size as u64]
    {
        return false;
    }
    let common = arch.vocab_size == 248_320
        && arch.full_attention_interval == 4
        && arch.attn_head_dim == 256
        && arch.rope_theta == 10_000_000.0
        && arch.partial_rotary_factor == 0.25
        && arch.gdn_n_k_heads == 16
        && arch.gdn_head_dim == 128
        && arch.gdn_conv_kernel == 4
        && arch.mtp_n_hidden_layers == 0;
    common
        && ((dtype == GgmlType::Q4_K
            && arch.kind == ArchKind::Dense
            && arch.n_layer == 64
            && arch.hidden_size == 5120
            && arch.intermediate_size == 17408
            && arch.n_q_heads == 24
            && arch.n_kv_heads == 4
            && arch.gdn_n_v_heads == 48
            && arch.expert_count == 0
            && arch.expert_used_count == 0
            && arch.expert_feed_forward_length == 0
            && arch.expert_shared_feed_forward_length == 0)
            || (dtype == GgmlType::Q8_0
                && arch.kind == ArchKind::Moe
                && arch.n_layer == 40
                && arch.hidden_size == 2048
                && arch.intermediate_size == 0
                && arch.n_q_heads == 16
                && arch.n_kv_heads == 2
                && arch.gdn_n_v_heads == 32
                && arch.expert_count == 256
                && arch.expert_used_count == 8
                && arch.expert_feed_forward_length == 512
                && arch.expert_shared_feed_forward_length == 512))
}

fn resolve_native_quant_embedding(
    mode: NativeQuantEmbeddingMode,
    supported: bool,
    promoted: bool,
) -> NativeQuantEmbeddingSelection {
    if !supported {
        return NativeQuantEmbeddingSelection::Unsupported;
    }
    match mode {
        NativeQuantEmbeddingMode::Auto if promoted => NativeQuantEmbeddingSelection::AutoPromoted,
        NativeQuantEmbeddingMode::Forced => NativeQuantEmbeddingSelection::Forced,
        NativeQuantEmbeddingMode::Auto => NativeQuantEmbeddingSelection::AutoUnpromoted,
        NativeQuantEmbeddingMode::Disabled => NativeQuantEmbeddingSelection::RollbackDisabled,
        NativeQuantEmbeddingMode::Invalid => NativeQuantEmbeddingSelection::InvalidDisabled,
    }
}

const GGUF_NO_COPY_ALIGNMENT: usize = 32;
const GGUF_NO_COPY_27B_LAYOUT_DIGEST: u64 = 0xd116_405f_d99f_54d9;
const GGUF_NO_COPY_27B_MAPPED_BYTES: usize = 16_817_244_384;
const GGUF_NO_COPY_27B_DESCRIPTOR_COUNT: usize = 851;
const GGUF_NO_COPY_27B_SOURCE_BYTES: u64 = 16_806_250_496;
const GGUF_NO_COPY_27B_VIEW_COUNT: usize = 850;
const GGUF_NO_COPY_27B_VIEW_BYTES: u64 = 16_806_230_016;
const GGUF_NO_COPY_27B_TAIL_NAME: &str = "blk.63.post_attention_norm.weight";
const GGUF_NO_COPY_27B_TAIL_BYTES: u64 = 20_480;
const GGUF_OWNED_A3B_MAPPED_BYTES: usize = 22_134_528_992;
const GGUF_OWNED_A3B_LAYOUT_DIGEST: u64 = 0x5ae6_45df_5cf7_d568;
const GGUF_OWNED_A3B_INVENTORY_DIGEST: &str =
    "f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5";
const GGUF_OWNED_A3B_PLAN_DIGEST: &str =
    "fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af";
const GGUF_OWNED_A3B_REQUESTS: usize = 733;
const GGUF_OWNED_A3B_SOURCE_BYTES: u64 = 22_123_538_944;
const GGUF_OWNED_A3B_VIEWS: usize = 732;
const GGUF_OWNED_A3B_VIEW_BYTES: u64 = 22_123_530_752;
const GGUF_OWNED_A3B_WINDOW_BYTES: u64 = 22_123_544_576;
const GGUF_OWNED_A3B_GAP_BYTES: u64 = 13_824;
const GGUF_OWNED_A3B_FALLBACK_BYTES: u64 = 8_192;
const GGUF_OWNED_A3B_PHYSICAL_BYTES: u64 = 22_123_552_768;
const GGUF_PAGE_ROUNDED_A3B_ALLOCATED_BYTES: u64 = 22_126_297_088;
const GGUF_PAGE_ROUNDED_A3B_PADDING_BYTES: u64 = 2_758_144;
const GGUF_PAGE_ROUNDED_A3B_PADDED_RESOURCES: usize = 232;
const GGUF_OWNED_WORKERS: usize = 4;
const A3B_PARALLEL_COPY_AUTO_DEVICE: &str = "Apple M4 Max";
const A3B_PARALLEL_COPY_AUTO_MIN_MEMORY: u64 = 128 * 1024 * 1024 * 1024;
const A10B_PARALLEL_PREAD_REQUIRED_HEADROOM_BYTES: u64 = 85_608_931_328;
const A3B_PARALLEL_COPY_AUTO_OVERRIDE_ENVS: [&str; 8] = [
    "QWEN_GGUF_NO_COPY",
    "QWEN_GGUF_OWNED_ARENA",
    "QWEN_GGUF_NO_COPY_PREFAULT",
    "QWEN_NATIVE_QUANT_EMBED",
    "QWEN_MOE_ROUTER_F16",
    "QWEN_MOE_IQ3_EXPERT_NATIVE",
    "QWEN_PREFILL_MOE_GROUPED_IQ3_GATEUP",
    "QWEN_PREFILL_ATTN_FUSED_QKV_G8",
];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ParallelCopyProfileId {
    A3bQ4kmV1,
    A10bQ4xlV1,
    Dense27bQ4kmV1,
}

impl ParallelCopyProfileId {
    fn label(self) -> &'static str {
        match self {
            Self::A3bQ4kmV1 => "a3b-q4km-v1",
            Self::A10bQ4xlV1 => "a10b-q4xl-v1",
            Self::Dense27bQ4kmV1 => "dense27b-q4km-v1",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParallelCopyDeviceConstraint {
    UnifiedAnyName,
    ExactUnified(&'static str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParallelCopyAuthentication {
    A3bRetainedPlan,
    A10bPlannerFree,
    DensePlannerFree,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParallelCopyMarkerContract {
    A3b,
    A10bSchema2,
    DenseSchema2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ParallelCopyScheduleIdentity {
    request_index: usize,
    name: &'static str,
    shard_idx: usize,
    source_offset: u64,
    source_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ParallelCopyScheduleBoundary {
    first: ParallelCopyScheduleIdentity,
    last: ParallelCopyScheduleIdentity,
}

#[derive(Debug)]
struct ParallelCopyProfile {
    id: ParallelCopyProfileId,
    architecture_label: Option<&'static str>,
    arch: Arch,
    tied_embeddings: bool,
    mtp_present: bool,
    shard_mapped_lengths: &'static [usize],
    descriptor_layout_digest: u64,
    inventory_digest: &'static str,
    embedding_dtype: GgmlType,
    embedding_shape: &'static [u64],
    device_constraint: ParallelCopyDeviceConstraint,
    authentication: ParallelCopyAuthentication,
    marker_contract: ParallelCopyMarkerContract,
    supports_direct_pread: bool,
    request_count: usize,
    source_bytes: u64,
    cuts: [usize; GGUF_OWNED_WORKERS - 1],
    task_counts: [usize; GGUF_OWNED_WORKERS],
    worker_bytes: [u64; GGUF_OWNED_WORKERS],
    boundaries: [ParallelCopyScheduleBoundary; GGUF_OWNED_WORKERS],
}

const A3B_PARALLEL_COPY_ARCH: Arch = Arch {
    kind: ArchKind::Moe,
    n_layer: 40,
    hidden_size: 2048,
    intermediate_size: 0,
    vocab_size: 248_320,
    full_attention_interval: 4,
    n_q_heads: 16,
    n_kv_heads: 2,
    attn_head_dim: 256,
    rope_theta: 10_000_000.0,
    partial_rotary_factor: 0.25,
    gdn_n_v_heads: 32,
    gdn_n_k_heads: 16,
    gdn_head_dim: 128,
    gdn_conv_kernel: 4,
    expert_count: 256,
    expert_used_count: 8,
    expert_feed_forward_length: 512,
    expert_shared_feed_forward_length: 512,
    mtp_n_hidden_layers: 0,
};

const DENSE27B_PARALLEL_COPY_ARCH: Arch = Arch {
    kind: ArchKind::Dense,
    n_layer: 64,
    hidden_size: 5120,
    intermediate_size: 17_408,
    vocab_size: 248_320,
    full_attention_interval: 4,
    n_q_heads: 24,
    n_kv_heads: 4,
    attn_head_dim: 256,
    rope_theta: 10_000_000.0,
    partial_rotary_factor: 0.25,
    gdn_n_v_heads: 48,
    gdn_n_k_heads: 16,
    gdn_head_dim: 128,
    gdn_conv_kernel: 4,
    expert_count: 0,
    expert_used_count: 0,
    expert_feed_forward_length: 0,
    expert_shared_feed_forward_length: 0,
    mtp_n_hidden_layers: 0,
};

const A10B_PARALLEL_COPY_ARCH: Arch = Arch {
    kind: ArchKind::Moe,
    n_layer: 48,
    hidden_size: 3072,
    intermediate_size: 0,
    vocab_size: 248_320,
    full_attention_interval: 4,
    n_q_heads: 32,
    n_kv_heads: 2,
    attn_head_dim: 256,
    rope_theta: 10_000_000.0,
    partial_rotary_factor: 0.25,
    gdn_n_v_heads: 64,
    gdn_n_k_heads: 16,
    gdn_head_dim: 128,
    gdn_conv_kernel: 4,
    expert_count: 256,
    expert_used_count: 8,
    expert_feed_forward_length: 1024,
    expert_shared_feed_forward_length: 1024,
    mtp_n_hidden_layers: 0,
};

const A3B_PARALLEL_COPY_PROFILE: ParallelCopyProfile = ParallelCopyProfile {
    id: ParallelCopyProfileId::A3bQ4kmV1,
    architecture_label: None,
    arch: A3B_PARALLEL_COPY_ARCH,
    tied_embeddings: false,
    mtp_present: false,
    shard_mapped_lengths: &[GGUF_OWNED_A3B_MAPPED_BYTES],
    descriptor_layout_digest: GGUF_OWNED_A3B_LAYOUT_DIGEST,
    inventory_digest: GGUF_OWNED_A3B_INVENTORY_DIGEST,
    embedding_dtype: GgmlType::Q8_0,
    embedding_shape: &[2048, 248_320],
    device_constraint: ParallelCopyDeviceConstraint::UnifiedAnyName,
    authentication: ParallelCopyAuthentication::A3bRetainedPlan,
    marker_contract: ParallelCopyMarkerContract::A3b,
    supports_direct_pread: true,
    request_count: GGUF_OWNED_A3B_REQUESTS,
    source_bytes: GGUF_OWNED_A3B_SOURCE_BYTES,
    cuts: [155, 359, 539],
    task_counts: [155, 204, 180, 194],
    worker_bytes: [5_532_746_240, 5_462_315_776, 5_595_522_304, 5_532_954_624],
    boundaries: [
        ParallelCopyScheduleBoundary {
            first: ParallelCopyScheduleIdentity {
                request_index: 2,
                name: "output.weight",
                shard_idx: 0,
                source_offset: 10_990_048,
                source_bytes: 417_177_600,
            },
            last: ParallelCopyScheduleIdentity {
                request_index: 164,
                name: "blk.8.ffn_gate_exps.weight",
                shard_idx: 0,
                source_offset: 5_392_741_344,
                source_bytes: 150_994_944,
            },
        },
        ParallelCopyScheduleBoundary {
            first: ParallelCopyScheduleIdentity {
                request_index: 163,
                name: "blk.8.ffn_gate_inp.weight",
                shard_idx: 0,
                source_offset: 5_543_736_288,
                source_bytes: 2_097_152,
            },
            last: ParallelCopyScheduleIdentity {
                request_index: 354,
                name: "blk.19.attn_v.weight",
                shard_idx: 0,
                source_offset: 11_004_937_952,
                source_bytes: 1_114_112,
            },
        },
        ParallelCopyScheduleBoundary {
            first: ParallelCopyScheduleIdentity {
                request_index: 366,
                name: "blk.19.ffn_down_exps.weight",
                shard_idx: 0,
                source_offset: 11_006_052_064,
                source_bytes: 184_549_376,
            },
            last: ParallelCopyScheduleIdentity {
                request_index: 548,
                name: "blk.29.ffn_gate_exps.weight",
                shard_idx: 0,
                source_offset: 16_450_579_424,
                source_bytes: 150_994_944,
            },
        },
        ParallelCopyScheduleBoundary {
            first: ParallelCopyScheduleIdentity {
                request_index: 547,
                name: "blk.29.ffn_gate_inp.weight",
                shard_idx: 0,
                source_offset: 16_601_574_368,
                source_bytes: 2_097_152,
            },
            last: ParallelCopyScheduleIdentity {
                request_index: 721,
                name: "blk.39.post_attention_norm.weight",
                shard_idx: 0,
                source_offset: 22_134_520_800,
                source_bytes: 8_192,
            },
        },
    ],
};

const DENSE27B_PARALLEL_COPY_PROFILE: ParallelCopyProfile = ParallelCopyProfile {
    id: ParallelCopyProfileId::Dense27bQ4kmV1,
    architecture_label: Some("qwen35"),
    arch: DENSE27B_PARALLEL_COPY_ARCH,
    tied_embeddings: false,
    mtp_present: false,
    shard_mapped_lengths: &[GGUF_NO_COPY_27B_MAPPED_BYTES],
    descriptor_layout_digest: GGUF_NO_COPY_27B_LAYOUT_DIGEST,
    inventory_digest: "50e9af4e4f590fc85687a71f5602ce035e7fdf0e2a31e928b2c7a2be10458a07",
    embedding_dtype: GgmlType::Q4_K,
    embedding_shape: &[5120, 248_320],
    device_constraint: ParallelCopyDeviceConstraint::ExactUnified("Apple M4 Max"),
    authentication: ParallelCopyAuthentication::DensePlannerFree,
    marker_contract: ParallelCopyMarkerContract::DenseSchema2,
    supports_direct_pread: true,
    request_count: GGUF_NO_COPY_27B_DESCRIPTOR_COUNT,
    source_bytes: GGUF_NO_COPY_27B_SOURCE_BYTES,
    cuts: [136, 377, 618],
    task_counts: [136, 241, 241, 233],
    worker_bytes: [4_194_110_464, 4_214_375_808, 4_204_933_376, 4_192_830_848],
    boundaries: [
        ParallelCopyScheduleBoundary {
            first: ParallelCopyScheduleIdentity {
                request_index: 2,
                name: "output.weight",
                shard_idx: 0,
                source_offset: 10_993_888,
                source_bytes: 1_042_944_000,
            },
            last: ParallelCopyScheduleIdentity {
                request_index: 135,
                name: "blk.9.ssm_norm.weight",
                shard_idx: 0,
                source_offset: 4_205_103_840,
                source_bytes: 512,
            },
        },
        ParallelCopyScheduleBoundary {
            first: ParallelCopyScheduleIdentity {
                request_index: 136,
                name: "blk.9.ssm_out.weight",
                shard_idx: 0,
                source_offset: 4_205_104_352,
                source_bytes: 21_626_880,
            },
            last: ParallelCopyScheduleIdentity {
                request_index: 379,
                name: "blk.28.attn_qkv.weight",
                shard_idx: 0,
                source_offset: 8_376_472_160,
                source_bytes: 43_008_000,
            },
        },
        ParallelCopyScheduleBoundary {
            first: ParallelCopyScheduleIdentity {
                request_index: 378,
                name: "blk.28.ffn_down.weight",
                shard_idx: 0,
                source_offset: 8_419_480_160,
                source_bytes: 73_113_600,
            },
            last: ParallelCopyScheduleIdentity {
                request_index: 618,
                name: "blk.46.ffn_down.weight",
                shard_idx: 0,
                source_offset: 12_551_299_936,
                source_bytes: 73_113_600,
            },
        },
        ParallelCopyScheduleBoundary {
            first: ParallelCopyScheduleIdentity {
                request_index: 616,
                name: "blk.46.ffn_gate.weight",
                shard_idx: 0,
                source_offset: 12_624_413_536,
                source_bytes: 50_135_040,
            },
            last: ParallelCopyScheduleIdentity {
                request_index: 844,
                name: "blk.63.post_attention_norm.weight",
                shard_idx: 0,
                source_offset: 16_817_223_904,
                source_bytes: 20_480,
            },
        },
    ],
};

const A10B_PARALLEL_PREAD_PROFILE: ParallelCopyProfile = ParallelCopyProfile {
    id: ParallelCopyProfileId::A10bQ4xlV1,
    architecture_label: Some("qwen35moe"),
    arch: A10B_PARALLEL_COPY_ARCH,
    tied_embeddings: false,
    mtp_present: false,
    shard_mapped_lengths: &[10_943_552, 49_640_779_424, 27_378_273_056],
    descriptor_layout_digest: 0x3eb2_9091_5bec_2041,
    inventory_digest: "b331c475123dbee3bc862a495266dee3996c5f3adabcd6fbeaff9bbabd71a4f8",
    embedding_dtype: GgmlType::Q8_0,
    embedding_shape: &[3072, 248_320],
    device_constraint: ParallelCopyDeviceConstraint::ExactUnified("Apple M4 Max"),
    authentication: ParallelCopyAuthentication::A10bPlannerFree,
    marker_contract: ParallelCopyMarkerContract::A10bSchema2,
    supports_direct_pread: true,
    request_count: 879,
    source_bytes: 77_018_996_736,
    cuts: [214, 435, 658],
    task_counts: [214, 221, 223, 221],
    worker_bytes: [
        19_474_295_808,
        19_228_744_704,
        19_231_902_720,
        19_084_053_504,
    ],
    boundaries: [
        ParallelCopyScheduleBoundary {
            first: ParallelCopyScheduleIdentity {
                request_index: 2,
                name: "output.weight",
                shard_idx: 1,
                source_offset: 35_488,
                source_bytes: 810_516_480,
            },
            last: ParallelCopyScheduleIdentity {
                request_index: 220,
                name: "blk.11.ffn_down_exps.weight",
                shard_idx: 1,
                source_offset: 18_920_683_168,
                source_bytes: 553_648_128,
            },
        },
        ParallelCopyScheduleBoundary {
            first: ParallelCopyScheduleIdentity {
                request_index: 213,
                name: "blk.11.ffn_down_shexp.weight",
                shard_idx: 1,
                source_offset: 19_474_331_296,
                source_bytes: 3_342_336,
            },
            last: ParallelCopyScheduleIdentity {
                request_index: 437,
                name: "blk.23.ffn_gate_exps.weight",
                shard_idx: 1,
                source_offset: 38_250_091_168,
                source_bytes: 452_984_832,
            },
        },
        ParallelCopyScheduleBoundary {
            first: ParallelCopyScheduleIdentity {
                request_index: 436,
                name: "blk.23.ffn_gate_inp.weight",
                shard_idx: 1,
                source_offset: 38_703_076_000,
                source_bytes: 3_145_728,
            },
            last: ParallelCopyScheduleIdentity {
                request_index: 657,
                name: "blk.35.ffn_up_exps.weight",
                shard_idx: 2,
                source_offset: 7_841_234_720,
                source_bytes: 452_984_832,
            },
        },
        ParallelCopyScheduleBoundary {
            first: ParallelCopyScheduleIdentity {
                request_index: 650,
                name: "blk.35.ffn_up_shexp.weight",
                shard_idx: 2,
                source_offset: 8_294_219_552,
                source_bytes: 3_342_336,
            },
            last: ParallelCopyScheduleIdentity {
                request_index: 867,
                name: "blk.47.post_attention_norm.weight",
                shard_idx: 2,
                source_offset: 27_378_260_768,
                source_bytes: 12_288,
            },
        },
    ],
};

static PARALLEL_COPY_PROFILES: [&ParallelCopyProfile; 3] = [
    &A3B_PARALLEL_COPY_PROFILE,
    &A10B_PARALLEL_PREAD_PROFILE,
    &DENSE27B_PARALLEL_COPY_PROFILE,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GgufNoCopyMode {
    Disabled,
    Forced,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GgufOwnedArenaMode {
    Disabled,
    Forced,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GgufParallelCopyMode {
    Auto,
    Disabled,
    ForcedCopy,
    ForcedPread,
    ForcedPageRoundedCopy,
}

impl GgufParallelCopyMode {
    fn is_forced(self) -> bool {
        matches!(
            self,
            Self::ForcedCopy | Self::ForcedPread | Self::ForcedPageRoundedCopy
        )
    }

    fn forced_configuration(self) -> Option<(ParallelPopulationMethod, ParallelDestinationLength)> {
        match self {
            Self::ForcedCopy => Some((
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::LogicalExact,
            )),
            Self::ForcedPread => Some((
                ParallelPopulationMethod::Pread,
                ParallelDestinationLength::LogicalExact,
            )),
            Self::ForcedPageRoundedCopy => Some((
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::PageRounded16K,
            )),
            Self::Auto | Self::Disabled => None,
        }
    }
}

fn parse_gguf_owned_arena_mode(value: Option<&str>) -> Result<GgufOwnedArenaMode, MfError> {
    match value {
        None => Ok(GgufOwnedArenaMode::Disabled),
        Some(value) if crate::env_flag::env_value_truthy(value) => Ok(GgufOwnedArenaMode::Forced),
        Some(value) if crate::env_flag::env_value_falsy(value) => Ok(GgufOwnedArenaMode::Disabled),
        Some(value) => Err(MfError::LoadPolicy(format!(
            "invalid QWEN_GGUF_OWNED_ARENA value {value:?}"
        ))),
    }
}

fn gguf_owned_arena_mode() -> Result<GgufOwnedArenaMode, MfError> {
    match std::env::var("QWEN_GGUF_OWNED_ARENA") {
        Ok(value) => parse_gguf_owned_arena_mode(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_gguf_owned_arena_mode(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(MfError::LoadPolicy(
            "QWEN_GGUF_OWNED_ARENA is not valid Unicode".to_string(),
        )),
    }
}

fn parse_gguf_parallel_copy_mode(value: Option<&str>) -> Result<GgufParallelCopyMode, MfError> {
    match value {
        None => Ok(GgufParallelCopyMode::Auto),
        Some(value) if value.eq_ignore_ascii_case("pread") => Ok(GgufParallelCopyMode::ForcedPread),
        Some(value) if value.eq_ignore_ascii_case("page-rounded-copy") => {
            Ok(GgufParallelCopyMode::ForcedPageRoundedCopy)
        }
        Some(value) if crate::env_flag::env_value_truthy(value) => {
            Ok(GgufParallelCopyMode::ForcedCopy)
        }
        Some(value) if crate::env_flag::env_value_falsy(value) => {
            Ok(GgufParallelCopyMode::Disabled)
        }
        Some(value) => Err(MfError::LoadPolicy(format!(
            "invalid QWEN_GGUF_PARALLEL_COPY value {value:?}"
        ))),
    }
}

fn gguf_parallel_copy_mode() -> Result<GgufParallelCopyMode, MfError> {
    match std::env::var("QWEN_GGUF_PARALLEL_COPY") {
        Ok(value) => parse_gguf_parallel_copy_mode(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_gguf_parallel_copy_mode(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(MfError::LoadPolicy(
            "QWEN_GGUF_PARALLEL_COPY is not valid Unicode".to_string(),
        )),
    }
}

fn auto_parallel_copy_a3b_enabled(
    admission_enabled: bool,
    parallel_mode: GgufParallelCopyMode,
    explicit_override_present: bool,
) -> bool {
    admission_enabled && parallel_mode == GgufParallelCopyMode::Auto && !explicit_override_present
}

fn auto_parallel_copy_a3b_override_present(mut is_present: impl FnMut(&str) -> bool) -> bool {
    A3B_PARALLEL_COPY_AUTO_OVERRIDE_ENVS
        .iter()
        .copied()
        .any(&mut is_present)
}

fn validate_parallel_copy_policy(
    parallel_mode: GgufParallelCopyMode,
    no_copy_mode: GgufNoCopyMode,
    owned_mode: GgufOwnedArenaMode,
    prefault_present: bool,
    native_embedding_present: bool,
    router_f16: Option<&str>,
) -> Result<(), MfError> {
    if !parallel_mode.is_forced() {
        return Ok(());
    }
    if no_copy_mode == GgufNoCopyMode::Forced {
        return Err(MfError::LoadPolicy(
            "parallel copy and retained no-copy are mutually exclusive".to_string(),
        ));
    }
    if owned_mode == GgufOwnedArenaMode::Forced {
        return Err(MfError::LoadPolicy(
            "parallel copy and owned arena are mutually exclusive".to_string(),
        ));
    }
    if prefault_present {
        return Err(MfError::LoadPolicy(
            "QWEN_GGUF_NO_COPY_PREFAULT is invalid with parallel copy".to_string(),
        ));
    }
    if native_embedding_present {
        return Err(MfError::LoadPolicy(
            "parallel copy requires production-auto native embedding selection".to_string(),
        ));
    }
    match router_f16 {
        None => Ok(()),
        Some(value) if crate::env_flag::env_value_falsy(value) => Ok(()),
        Some(value) if crate::env_flag::env_value_truthy(value) => Err(MfError::LoadPolicy(
            "QWEN_MOE_ROUTER_F16 is invalid with parallel copy".to_string(),
        )),
        Some(value) => Err(MfError::LoadPolicy(format!(
            "invalid QWEN_MOE_ROUTER_F16 value {value:?} with parallel copy"
        ))),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GgufNoCopyPrefaultMode {
    Default,
    Enabled,
    Disabled,
}

fn parse_gguf_no_copy_mode(value: Option<&str>) -> Result<GgufNoCopyMode, MfError> {
    match value {
        None => Ok(GgufNoCopyMode::Disabled),
        Some(value) if crate::env_flag::env_value_truthy(value) => Ok(GgufNoCopyMode::Forced),
        Some(value) if crate::env_flag::env_value_falsy(value) => Ok(GgufNoCopyMode::Disabled),
        Some(value) => Err(MfError::LoadPolicy(format!(
            "invalid QWEN_GGUF_NO_COPY value {value:?}"
        ))),
    }
}

fn gguf_no_copy_mode() -> Result<GgufNoCopyMode, MfError> {
    match std::env::var("QWEN_GGUF_NO_COPY") {
        Ok(value) => parse_gguf_no_copy_mode(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_gguf_no_copy_mode(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(MfError::LoadPolicy(
            "QWEN_GGUF_NO_COPY is not valid Unicode".to_string(),
        )),
    }
}

fn parse_gguf_no_copy_prefault(value: Option<&str>) -> Result<GgufNoCopyPrefaultMode, MfError> {
    match value {
        None => Ok(GgufNoCopyPrefaultMode::Default),
        Some(value) if crate::env_flag::env_value_truthy(value) => {
            Ok(GgufNoCopyPrefaultMode::Enabled)
        }
        Some(value) if crate::env_flag::env_value_falsy(value) => {
            Ok(GgufNoCopyPrefaultMode::Disabled)
        }
        Some(value) => Err(MfError::LoadPolicy(format!(
            "invalid QWEN_GGUF_NO_COPY_PREFAULT value {value:?}"
        ))),
    }
}

fn gguf_no_copy_prefault_mode() -> Result<GgufNoCopyPrefaultMode, MfError> {
    match std::env::var("QWEN_GGUF_NO_COPY_PREFAULT") {
        Ok(value) => parse_gguf_no_copy_prefault(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_gguf_no_copy_prefault(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(MfError::LoadPolicy(
            "QWEN_GGUF_NO_COPY_PREFAULT is not valid Unicode".to_string(),
        )),
    }
}

fn hash_layout_bytes(hash: &mut u64, bytes: &[u8]) {
    const PRIME: u64 = 0x100000001b3;
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(PRIME);
    }
}

fn hash_layout_u64(hash: &mut u64, value: u64) {
    hash_layout_bytes(hash, &value.to_le_bytes());
}

pub fn gguf_descriptor_layout_digest(gguf: &GgufFile) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    hash_layout_bytes(&mut hash, b"qwen-gguf-layout-v1");
    hash_layout_u64(&mut hash, gguf.shard_count() as u64);
    for shard in &gguf.shards {
        hash_layout_u64(&mut hash, shard.mmap.len() as u64);
    }
    hash_layout_u64(&mut hash, gguf.tensors.len() as u64);
    for desc in &gguf.tensors {
        hash_layout_u64(&mut hash, desc.name.len() as u64);
        hash_layout_bytes(&mut hash, desc.name.as_bytes());
        hash_layout_u64(&mut hash, desc.dtype as i32 as u32 as u64);
        hash_layout_u64(&mut hash, desc.shard_idx as u64);
        hash_layout_u64(&mut hash, desc.data_offset);
        hash_layout_u64(&mut hash, desc.n_bytes);
        hash_layout_u64(&mut hash, desc.shape.len() as u64);
        for dim in &desc.shape {
            hash_layout_u64(&mut hash, *dim);
        }
    }
    hash
}

// Per-projection GDN no-op ablations: each is its own opt-in flag, OR'd
// with the broad `..NOOP_FRONT` umbrella flag above.
crate::env_flag!(default_off decode_gdn_noop_qkv_flag, "QWEN_DECODE_GDN_NOOP_QKV");
crate::env_flag!(default_off decode_gdn_noop_z_flag, "QWEN_DECODE_GDN_NOOP_Z");
crate::env_flag!(default_off decode_gdn_noop_beta_flag, "QWEN_DECODE_GDN_NOOP_BETA");
crate::env_flag!(default_off decode_gdn_noop_alpha_flag, "QWEN_DECODE_GDN_NOOP_ALPHA");
crate::env_flag!(default_on decode_gdn_pair_l2_enabled, "QWEN_DECODE_GDN_PAIR_L2");

fn decode_gdn_noop_qkv_enabled() -> bool {
    decode_gdn_noop_front_enabled() || decode_gdn_noop_qkv_flag()
}

fn decode_gdn_noop_z_enabled() -> bool {
    decode_gdn_noop_front_enabled() || decode_gdn_noop_z_flag()
}

fn decode_gdn_noop_beta_enabled() -> bool {
    decode_gdn_noop_front_enabled() || decode_gdn_noop_beta_flag()
}

fn decode_gdn_noop_alpha_enabled() -> bool {
    decode_gdn_noop_front_enabled() || decode_gdn_noop_alpha_flag()
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

crate::env_flag!(default_off phase_gdn_proj_split_enabled, "QWEN_PHASE_GDN_PROJ_SPLIT");
crate::env_flag!(default_off phase_gdn_tail_split_enabled, "QWEN_PHASE_GDN_TAIL_SPLIT");

fn phase_moe_route_split_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_PHASE_MOE_ROUTE_SPLIT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

fn phase_moe_route_deep_split_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_PHASE_MOE_ROUTE_SPLIT").as_deref(),
            Ok("2") | Ok("deep") | Ok("DEEP")
        )
    })
}

crate::env_flag!(default_off phase_moe_cpu_route_enabled, "QWEN_PHASE_MOE_CPU_ROUTE");
crate::env_flag!(default_off phase_moe_route_replay_enabled, "QWEN_PHASE_MOE_ROUTE_REPLAY");
crate::env_flag!(default_off phase_lm_argmax_enabled, "QWEN_PHASE_LM_ARGMAX");
crate::env_flag!(default_off decode_fused_residual_rmsnorm_enabled, "QWEN_DECODE_FUSED_RESIDUAL_RMSNORM");
crate::env_flag!(default_off decode_moe_noop_route_enabled, "QWEN_DECODE_MOE_NOOP_ROUTE");
crate::env_flag!(default_off decode_moe_noop_routed_gateup_enabled, "QWEN_DECODE_MOE_NOOP_ROUTED_GATEUP");
crate::env_flag!(default_off decode_moe_noop_routed_down_enabled, "QWEN_DECODE_MOE_NOOP_ROUTED_DOWN");

fn moe_routed_gate_up_decode_supported(gate: GgmlType, up: GgmlType) -> bool {
    matches!(
        (gate, up),
        (GgmlType::Q4_K, GgmlType::Q4_K)
            | (GgmlType::Q5_K, GgmlType::Q5_K)
            | (GgmlType::Q6_K, GgmlType::Q6_K)
            | (GgmlType::Q8_0, GgmlType::Q8_0)
            | (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS)
            | (GgmlType::IQ3_S, GgmlType::IQ3_S)
            | (GgmlType::BF16, GgmlType::BF16)
            | (GgmlType::F32, GgmlType::F32)
    )
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
    *ENABLED.get_or_init(|| crate::env_flag::read_default_on("QWEN_MATMAT_BF16_BFLOAT_ACT"))
}

/// Return type for [`MetalForward::single_token_phase_profiled`]:
/// `(logits, wall_with_artifact_ms, per-phase GPU ms map)`. The
/// per-phase entries are `(phase_name, gpu_ms)`.
pub type PhaseProfileOutput = (Vec<f32>, f64, Vec<(String, f64)>);

// --- T9 bench-only FFN-input capture (docs/bench/2026-07-20-trellis3-
// t9-ldlq-pilot/). When installed, `encode_post_mixer_ffn` scatters the
// post-norm FFN input (`s.h`) and the SwiGLU intermediate
// (`s.ffn_inner`) into caller-owned buffers for the registered per-token
// FFN-call indices (== absolute block index on the dense decode paths,
// which invoke that fn exactly once per block in layer order). The
// caller MUST call [`t9_ffn_capture_reset_token`] before each token.
// Zero cost when not installed; never installed by production paths.

thread_local! {
    static T9_FFN_CAPTURE: std::cell::RefCell<Option<Vec<(usize, MetalTensor, MetalTensor)>>> =
        const { std::cell::RefCell::new(None) };
    static T9_FFN_CALL_IDX: Cell<usize> = const { Cell::new(0) };
}

/// Install capture slots: (ffn_call_index, h_dst, inner_dst) triples.
pub fn t9_ffn_capture_install(slots: Vec<(usize, MetalTensor, MetalTensor)>) {
    T9_FFN_CAPTURE.with(|c| *c.borrow_mut() = Some(slots));
    T9_FFN_CALL_IDX.with(|c| c.set(0));
}

/// Reset the per-token FFN call counter (call before every token).
pub fn t9_ffn_capture_reset_token() {
    T9_FFN_CALL_IDX.with(|c| c.set(0));
}

/// Uninstall capture.
pub fn t9_ffn_capture_uninstall() {
    T9_FFN_CAPTURE.with(|c| *c.borrow_mut() = None);
}

fn t9_ffn_capture_slots_for_current_call() -> Option<(MetalTensor, MetalTensor)> {
    T9_FFN_CAPTURE.with(|c| {
        let borrow = c.borrow();
        let slots = borrow.as_ref()?;
        let idx = T9_FFN_CALL_IDX.with(|i| {
            let v = i.get();
            i.set(v + 1);
            v
        });
        slots
            .iter()
            .find(|(want, _, _)| *want == idx)
            .map(|(_, h, inner)| (h.clone(), inner.clone()))
    })
}

use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLCPUCacheMode, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLHazardTrackingMode, MTLResidencySet,
    MTLResidencySetDescriptor, MTLResource, MTLStorageMode,
};

#[cfg(test)]
thread_local! {
    static METAL_LOAD_TEST_LINES: std::cell::RefCell<Option<Vec<String>>> =
        const { std::cell::RefCell::new(None) };
}

fn emit_metal_load_line(arguments: std::fmt::Arguments<'_>) {
    #[cfg(not(test))]
    eprintln!("{arguments}");

    #[cfg(test)]
    {
        let line = arguments.to_string();
        eprintln!("{line}");
        METAL_LOAD_TEST_LINES.with(|lines| {
            if let Some(lines) = lines.borrow_mut().as_mut() {
                lines.push(line);
            }
        });
    }
}

#[cfg(test)]
fn capture_metal_load_lines<T>(run: impl FnOnce() -> T) -> (T, Vec<String>) {
    struct CaptureGuard;

    impl Drop for CaptureGuard {
        fn drop(&mut self) {
            METAL_LOAD_TEST_LINES.with(|lines| {
                lines.borrow_mut().take();
            });
        }
    }

    METAL_LOAD_TEST_LINES.with(|lines| {
        assert!(lines.borrow().is_none(), "nested Metal load-line capture");
        *lines.borrow_mut() = Some(Vec::new());
    });
    let guard = CaptureGuard;
    let value = run();
    let lines = METAL_LOAD_TEST_LINES.with(|lines| {
        lines
            .borrow_mut()
            .take()
            .expect("Metal load-line capture disappeared")
    });
    drop(guard);
    (value, lines)
}

fn emit_native_quant_embedding_policy(
    model: &Model<'_>,
    embedding_selection: NativeQuantEmbeddingSelection,
) {
    emit_metal_load_line(format_args!(
        "[metal-load] native quantized token embedding policy: {} ({:?} {:?})",
        embedding_selection.label(),
        model.token_embd.dtype,
        model.token_embd.shape,
    ));
}

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
    #[error("metal load policy: {0}")]
    LoadPolicy(String),
    #[error("snapshot validation: {0}")]
    Snapshot(#[from] SnapshotValidationError),
    #[error("Metal command buffer failed: status={status} error={error}")]
    CommandBuffer { status: String, error: String },
    #[error("Metal session is poisoned: {reason}")]
    SessionPoisoned { reason: &'static str },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArgmaxReduction {
    SpeculativeLowest,
    GreedyTotal,
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum LmHeadTail<'a> {
    Resident,
    CompactQ6K {
        weight: &'a MetalTensor,
        output: &'a MetalTensor,
    },
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LmHeadTailKind {
    Resident,
    CompactQ6K,
}

impl LmHeadTailKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::CompactQ6K => "compact_q6_k",
        }
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LmHeadTailEvidence {
    pub kind: LmHeadTailKind,
    pub n_in: usize,
    pub n_out: usize,
    pub weight_offset: u64,
    pub output_offset: u64,
    pub tail_dispatches: u32,
    pub full_head_dispatches: u32,
    pub command_completed: bool,
    pub command_error_none: bool,
}

fn lm_head_tail_error(detail: impl Into<String>) -> MfError {
    MfError::Metal(MetalError::BadShape {
        kernel: "lm_head_tail",
        detail: detail.into(),
    })
}

fn checked_tail_range(
    tensor: &MetalTensor,
    logical_bytes: usize,
    label: &str,
) -> Result<(usize, usize), MfError> {
    let start = usize::try_from(tensor.offset)
        .map_err(|_| lm_head_tail_error(format!("{label} offset does not fit usize")))?;
    let end = start
        .checked_add(logical_bytes)
        .ok_or_else(|| lm_head_tail_error(format!("{label} endpoint overflow")))?;
    if end > tensor.buffer.length() {
        return Err(lm_head_tail_error(format!(
            "{label} range [{start}..{end}) exceeds buffer length {}",
            tensor.buffer.length()
        )));
    }
    Ok((start, end))
}

fn tail_ranges_overlap(
    left: &MetalTensor,
    left_range: (usize, usize),
    right: &MetalTensor,
    right_range: (usize, usize),
) -> bool {
    Retained::as_ptr(&left.buffer) == Retained::as_ptr(&right.buffer)
        && left_range.0 < right_range.1
        && right_range.0 < left_range.1
}

fn encode_argmax_reduction(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    output: &MetalTensor,
    n_rows: usize,
    vocab: usize,
    reduction: ArgmaxReduction,
) -> Result<(), MetalError> {
    match reduction {
        ArgmaxReduction::SpeculativeLowest => {
            encode_argmax_f32(ctx, enc, logits, output, n_rows, vocab)
        }
        ArgmaxReduction::GreedyTotal => {
            encode_argmax_f32_greedy(ctx, enc, logits, output, n_rows, vocab)
        }
    }
}

/// All weight tensors, resident as `MetalTensor`s. Loaded once at session
/// start. v1: only F32 weights are supported here; quantized path comes
/// in v2 by switching `MetalModel::load` to use `MetalTensor::from_gguf_tensor`
/// directly (instead of going through the F32 codec) and the kernel
/// dispatchers to pick `_q4_k`/`_q6_k` based on dtype.
pub struct MetalModel {
    // Fields drop in declaration order. Remove the residency set before any
    // allocation it references can be released.
    _residency_set: Option<MetalModelResidencySetGuard>,

    /// Reference back to the loader's bound model. Carries `arch`, the
    /// layer schedule (GDN vs Attn), tied-embedding flag.
    pub arch: crate::model::Arch,
    pub tied_embeddings: bool,

    pub token_embd: MetalTensor,
    pub output_norm: MetalTensor,
    pub lm_head: MetalTensor,

    pub blocks: Vec<MetalBlock>,
}

impl MetalModel {
    /// Whether this model relies on a residency set attached to its load queue.
    /// Such a model cannot be submitted through unrelated command queues until
    /// those queues receive matching residency-set ownership and teardown.
    #[doc(hidden)]
    pub fn has_queue_scoped_residency_set(&self) -> bool {
        self._residency_set.is_some()
    }

    pub(crate) fn attach_residency_to_queue(
        &self,
        queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Option<MetalAdditionalQueueResidencySetGuard> {
        let model_guard = self._residency_set.as_ref()?;
        queue.addResidencySet(&model_guard.set);
        Some(MetalAdditionalQueueResidencySetGuard {
            queue: queue.clone(),
            set: model_guard.set.clone(),
        })
    }
}

pub(crate) struct MetalAdditionalQueueResidencySetGuard {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

impl Drop for MetalAdditionalQueueResidencySetGuard {
    fn drop(&mut self) {
        self.queue.removeResidencySet(&self.set);
    }
}

struct MetalModelResidencySetGuard {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

impl Drop for MetalModelResidencySetGuard {
    fn drop(&mut self) {
        let started = std::time::Instant::now();
        let allocations = self.set.allocationCount();
        let allocated_bytes = self.set.allocatedSize();
        self.queue.removeResidencySet(&self.set);
        self.set.endResidency();
        let _ = std::io::Write::write_fmt(
            &mut std::io::stderr().lock(),
            format_args!(
                "[metal-residency] API teardown returned allocations={allocations} allocated_bytes={allocated_bytes} elapsed_ms={:.3}\n",
                started.elapsed().as_secs_f64() * 1e3,
            ),
        );
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MetalModelLoadOptions {
    pub auto_parallel_copy_a3b: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MetalLoadPrefetchAdvice {
    PreserveConfiguredPolicy,
    SuppressColdOnlyAuthenticatedA3bDirectPread,
}

#[derive(Clone, Copy, Debug)]
struct ResolvedStoragePolicy {
    no_copy_mode: GgufNoCopyMode,
    prefault_enabled: bool,
    owned_mode: GgufOwnedArenaMode,
    parallel_mode: GgufParallelCopyMode,
}

#[derive(Clone, Copy, Debug)]
struct ResolvedWeightLoadChoices {
    embedding_selection: NativeQuantEmbeddingSelection,
    router_f16: bool,
    fused_qkv_g8: bool,
}

#[derive(Clone, Copy)]
enum PreparedParallelCopyProof {
    A3bRetainedPlan,
    A10bPlannerFree,
    DensePlannerFree,
}

struct PreparedParallelCopiedProfile {
    profile: &'static ParallelCopyProfile,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
    expected_identities: Vec<ModelWeightStorageIdentity>,
    sorted_request_indices: Vec<usize>,
    _proof: PreparedParallelCopyProof,
}

enum PreparedAutoSelection {
    NotEligible,
    NoMatch,
    Selected(PreparedParallelCopiedProfile),
}

impl PreparedAutoSelection {
    fn prefetch_advice(&self) -> MetalLoadPrefetchAdvice {
        match self {
            Self::Selected(prepared)
                if prepared.profile.id == ParallelCopyProfileId::A3bQ4kmV1
                    && prepared.population == ParallelPopulationMethod::Pread
                    && prepared.destination_length == ParallelDestinationLength::LogicalExact
                    && matches!(&prepared._proof, PreparedParallelCopyProof::A3bRetainedPlan) =>
            {
                MetalLoadPrefetchAdvice::SuppressColdOnlyAuthenticatedA3bDirectPread
            }
            Self::NotEligible | Self::NoMatch | Self::Selected(_) => {
                MetalLoadPrefetchAdvice::PreserveConfiguredPolicy
            }
        }
    }
}

pub(crate) struct PreparedMetalModelLoad<'ctx, 'gguf, 'model> {
    ctx: &'ctx MetalContext,
    gguf: &'gguf GgufFile,
    model: &'model Model<'gguf>,
    storage: ResolvedStoragePolicy,
    choices: ResolvedWeightLoadChoices,
    expected: Vec<ModelWeightStorageRequest<'gguf>>,
    auto: PreparedAutoSelection,
}

impl PreparedMetalModelLoad<'_, '_, '_> {
    pub(crate) fn prefetch_advice(&self) -> MetalLoadPrefetchAdvice {
        self.auto.prefetch_advice()
    }
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

#[derive(Clone, Debug)]
pub struct MoeRouteReplayRow {
    pub topk_idx: Vec<i32>,
    pub topk_weight: Vec<f32>,
    pub hidden: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceMaterialization {
    DirectCopy,
    DirectView,
    DirectAlias,
    TailFallback,
    ConvertedF32,
    ConvertedF16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelWeightStorageKind {
    Direct,
    ConvertedF32,
    ConvertedF16,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelWeightStorageRequest<'a> {
    pub desc: &'a TensorDesc,
    pub kind: ModelWeightStorageKind,
    pub resident_bytes: u64,
}

fn storage_digest_records(records: impl IntoIterator<Item = String>) -> String {
    let mut hasher = Sha256::new();
    for record in records {
        hasher.update((record.len() as u64).to_be_bytes());
        hasher.update(record.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn model_weight_storage_inventory_digest(requests: &[ModelWeightStorageRequest<'_>]) -> String {
    storage_digest_records(requests.iter().map(|request| {
        format!(
            "{}\0{}\0{}\0{}\0{:?}\0{:?}\0{:?}\0{}",
            request.desc.name,
            request.desc.shard_idx,
            request.desc.data_offset,
            request.desc.n_bytes,
            request.desc.dtype,
            request.desc.shape,
            request.kind,
            request.resident_bytes
        )
    }))
}

pub fn retained_storage_plan_digest(plan: &RetainedStoragePlan) -> String {
    let mut records = vec![format!(
        "header\0{}\0{}\0{}\0{}",
        plan.page_size, plan.max_buffer_length, plan.usable_window_length, plan.required_alignment
    )];
    records.extend(plan.windows.iter().enumerate().map(|(index, window)| {
        format!(
            "window\0{index}\0{}\0{}\0{}",
            window.shard_idx, window.mmap_offset, window.length
        )
    }));
    records.extend(plan.entries.iter().map(|entry| {
        format!(
            "entry\0{}\0{}\0{}\0{}\0{}\0{:?}",
            entry.request_index,
            entry.name,
            entry.shard_idx,
            entry.data_offset,
            entry.n_bytes,
            entry.disposition
        )
    }));
    storage_digest_records(records)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ModelWeightStorageIdentity {
    name: String,
    shard_idx: usize,
    data_offset: u64,
    source_bytes: u64,
    dtype: GgmlType,
    shape: Vec<u64>,
    kind: ModelWeightStorageKind,
    resident_bytes: u64,
}

fn expected_model_weight_identity(
    request: &ModelWeightStorageRequest<'_>,
) -> ModelWeightStorageIdentity {
    ModelWeightStorageIdentity {
        name: request.desc.name.clone(),
        shard_idx: request.desc.shard_idx,
        data_offset: request.desc.data_offset,
        source_bytes: request.desc.n_bytes,
        dtype: request.desc.dtype,
        shape: request.desc.shape.clone(),
        kind: request.kind,
        resident_bytes: request.resident_bytes,
    }
}

fn validate_model_weight_request_sequence(
    actual: &[ModelWeightStorageIdentity],
    expected: &[ModelWeightStorageRequest<'_>],
) -> Result<(), MfError> {
    if actual.len() != expected.len() {
        return Err(MfError::LoadPolicy(format!(
            "model storage request count drift: actual={} expected={}",
            actual.len(),
            expected.len()
        )));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let expected = expected_model_weight_identity(expected);
        if *actual != expected {
            return Err(MfError::LoadPolicy(format!(
                "model storage request drift at index {index}: actual={actual:?} expected={expected:?}"
            )));
        }
    }
    Ok(())
}

fn push_model_weight_request<'a>(
    requests: &mut Vec<ModelWeightStorageRequest<'a>>,
    desc: &'a TensorDesc,
    kind: ModelWeightStorageKind,
) -> Result<(), MfError> {
    let elements = desc.checked_n_elements().ok_or_else(|| {
        MfError::LoadPolicy(format!("tensor {:?} element count overflow", desc.name))
    })?;
    let resident_bytes = match kind {
        ModelWeightStorageKind::Direct => desc.n_bytes,
        ModelWeightStorageKind::ConvertedF32 => elements
            .checked_mul(4)
            .ok_or_else(|| MfError::LoadPolicy("F32 conversion size overflow".to_string()))?,
        ModelWeightStorageKind::ConvertedF16 => elements
            .checked_mul(2)
            .ok_or_else(|| MfError::LoadPolicy("F16 conversion size overflow".to_string()))?,
    };
    requests.push(ModelWeightStorageRequest {
        desc,
        kind,
        resident_bytes,
    });
    Ok(())
}

fn push_f32_weight_request<'a>(
    requests: &mut Vec<ModelWeightStorageRequest<'a>>,
    desc: &'a TensorDesc,
) -> Result<(), MfError> {
    let kind = if desc.dtype == GgmlType::F32 {
        ModelWeightStorageKind::Direct
    } else {
        ModelWeightStorageKind::ConvertedF32
    };
    push_model_weight_request(requests, desc, kind)
}

fn push_native_weight_request<'a>(
    requests: &mut Vec<ModelWeightStorageRequest<'a>>,
    desc: &'a TensorDesc,
) -> Result<(), MfError> {
    if weight_dtype_kept_native(desc.dtype) {
        push_model_weight_request(requests, desc, ModelWeightStorageKind::Direct)
    } else {
        push_f32_weight_request(requests, desc)
    }
}

fn push_moe_weight_requests<'a>(
    requests: &mut Vec<ModelWeightStorageRequest<'a>>,
    moe: &MoeFfn<'a>,
    router_f16: bool,
) -> Result<(), MfError> {
    if router_f16 {
        push_model_weight_request(requests, moe.gate_inp, ModelWeightStorageKind::ConvertedF16)?;
    } else {
        push_f32_weight_request(requests, moe.gate_inp)?;
    }
    push_native_weight_request(requests, moe.gate_exps)?;
    push_native_weight_request(requests, moe.up_exps)?;
    push_native_weight_request(requests, moe.down_exps)?;
    push_f32_weight_request(requests, moe.gate_inp_shexp)
}

pub fn native_quant_embedding_storage_supported(model: &Model<'_>) -> bool {
    native_quant_embedding_supported(model.token_embd.dtype, &model.token_embd.shape)
}

pub fn production_native_quant_embedding_storage_enabled(model: &Model<'_>) -> bool {
    native_quant_embedding_storage_supported(model)
        && native_quant_embedding_default_promoted(
            &model.arch,
            model.tied_embeddings,
            model.mtp.is_some(),
            model.token_embd.dtype,
            &model.token_embd.shape,
        )
}

pub fn model_weight_storage_requests<'a>(
    model: &Model<'a>,
    native_quant_embedding: bool,
    router_f16: bool,
) -> Result<Vec<ModelWeightStorageRequest<'a>>, MfError> {
    if native_quant_embedding && !native_quant_embedding_storage_supported(model) {
        return Err(MfError::LoadPolicy(format!(
            "native token embedding is unsupported for {:?} {:?}",
            model.token_embd.dtype, model.token_embd.shape
        )));
    }
    let mut requests = Vec::new();
    let embedding_direct = matches!(
        model.token_embd.dtype,
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
    ) || native_quant_embedding;
    if embedding_direct {
        push_model_weight_request(
            &mut requests,
            model.token_embd,
            ModelWeightStorageKind::Direct,
        )?;
    } else {
        push_f32_weight_request(&mut requests, model.token_embd)?;
    }
    push_f32_weight_request(&mut requests, model.output_norm)?;
    push_native_weight_request(&mut requests, model.lm_head)?;

    for block in &model.blocks {
        match block {
            Block::Gdn(gdn) => {
                push_f32_weight_request(&mut requests, gdn.attn_norm)?;
                push_f32_weight_request(&mut requests, gdn.post_attention_norm)?;
                push_native_weight_request(&mut requests, gdn.ffn_gate)?;
                push_native_weight_request(&mut requests, gdn.ffn_up)?;
                push_native_weight_request(&mut requests, gdn.ffn_down)?;
                push_native_weight_request(&mut requests, gdn.in_proj_qkv)?;
                push_native_weight_request(&mut requests, gdn.in_proj_z)?;
                push_native_weight_request(&mut requests, gdn.beta_proj)?;
                push_native_weight_request(&mut requests, gdn.alpha_proj)?;
                push_f32_weight_request(&mut requests, gdn.a_log)?;
                push_f32_weight_request(&mut requests, gdn.dt_bias)?;
                push_f32_weight_request(&mut requests, gdn.conv1d)?;
                push_f32_weight_request(&mut requests, gdn.norm)?;
                push_native_weight_request(&mut requests, gdn.out_proj)?;
                if let Some(moe) = gdn.ffn_moe.as_ref() {
                    push_moe_weight_requests(&mut requests, moe, router_f16)?;
                }
            }
            Block::Attn(attn) => {
                push_native_weight_request(&mut requests, attn.q)?;
                push_native_weight_request(&mut requests, attn.k)?;
                push_native_weight_request(&mut requests, attn.v)?;
                push_f32_weight_request(&mut requests, attn.attn_norm)?;
                push_f32_weight_request(&mut requests, attn.post_attention_norm)?;
                push_native_weight_request(&mut requests, attn.ffn_gate)?;
                push_native_weight_request(&mut requests, attn.ffn_up)?;
                push_native_weight_request(&mut requests, attn.ffn_down)?;
                push_native_weight_request(&mut requests, attn.o)?;
                push_f32_weight_request(&mut requests, attn.q_norm)?;
                push_f32_weight_request(&mut requests, attn.k_norm)?;
                if let Some(moe) = attn.ffn_moe.as_ref() {
                    push_moe_weight_requests(&mut requests, moe, router_f16)?;
                }
            }
        }
    }
    Ok(requests)
}

pub fn mtp_weight_source_descriptors<'a>(model: &Model<'a>) -> Vec<&'a TensorDesc> {
    let Some(mtp) = model.mtp.as_ref() else {
        return Vec::new();
    };
    let attn = &mtp.attn;
    let mut descriptors = vec![
        attn.q,
        attn.k,
        attn.v,
        attn.attn_norm,
        attn.post_attention_norm,
        attn.ffn_gate,
        attn.ffn_up,
        attn.ffn_down,
        attn.o,
        attn.q_norm,
        attn.k_norm,
    ];
    if let Some(moe) = attn.ffn_moe.as_ref() {
        descriptors.extend([
            moe.gate_inp,
            moe.gate_exps,
            moe.up_exps,
            moe.down_exps,
            moe.gate_inp_shexp,
        ]);
    }
    descriptors.extend([mtp.eh_proj, mtp.enorm, mtp.hnorm, mtp.shared_head_norm]);
    descriptors
}

#[derive(Default)]
struct WeightLoadLedger {
    source_descriptors: usize,
    source_bytes: u64,
    direct_copy_descriptors: usize,
    direct_copy_bytes: u64,
    direct_view_descriptors: usize,
    direct_view_bytes: u64,
    direct_alias_descriptors: usize,
    direct_alias_bytes: u64,
    tail_fallback_descriptors: usize,
    tail_fallback_bytes: u64,
    converted_descriptors: usize,
    converted_source_bytes: u64,
    converted_resident_bytes: u64,
    derived_allocations: usize,
    derived_bytes: u64,
    requests: Vec<ModelWeightStorageIdentity>,
}

impl WeightLoadLedger {
    fn record_source(
        &mut self,
        desc: &TensorDesc,
        materialization: SourceMaterialization,
        resident_bytes: u64,
    ) -> Result<(), MfError> {
        self.source_descriptors = self
            .source_descriptors
            .checked_add(1)
            .ok_or_else(|| MfError::LoadPolicy("source descriptor count overflow".to_string()))?;
        self.source_bytes = self
            .source_bytes
            .checked_add(desc.n_bytes)
            .ok_or_else(|| MfError::LoadPolicy("source byte ledger overflow".to_string()))?;
        let (count, bytes) = match materialization {
            SourceMaterialization::DirectCopy => (
                &mut self.direct_copy_descriptors,
                &mut self.direct_copy_bytes,
            ),
            SourceMaterialization::DirectView => (
                &mut self.direct_view_descriptors,
                &mut self.direct_view_bytes,
            ),
            SourceMaterialization::DirectAlias => (
                &mut self.direct_alias_descriptors,
                &mut self.direct_alias_bytes,
            ),
            SourceMaterialization::TailFallback => (
                &mut self.tail_fallback_descriptors,
                &mut self.tail_fallback_bytes,
            ),
            SourceMaterialization::ConvertedF32 | SourceMaterialization::ConvertedF16 => {
                self.converted_resident_bytes = self
                    .converted_resident_bytes
                    .checked_add(resident_bytes)
                    .ok_or_else(|| {
                        MfError::LoadPolicy("converted resident byte ledger overflow".to_string())
                    })?;
                (
                    &mut self.converted_descriptors,
                    &mut self.converted_source_bytes,
                )
            }
        };
        *count = count
            .checked_add(1)
            .ok_or_else(|| MfError::LoadPolicy("materialization count overflow".to_string()))?;
        *bytes = bytes
            .checked_add(desc.n_bytes)
            .ok_or_else(|| MfError::LoadPolicy("materialization byte overflow".to_string()))?;
        let kind = match materialization {
            SourceMaterialization::DirectCopy
            | SourceMaterialization::DirectView
            | SourceMaterialization::DirectAlias
            | SourceMaterialization::TailFallback => ModelWeightStorageKind::Direct,
            SourceMaterialization::ConvertedF32 => ModelWeightStorageKind::ConvertedF32,
            SourceMaterialization::ConvertedF16 => ModelWeightStorageKind::ConvertedF16,
        };
        self.requests.push(ModelWeightStorageIdentity {
            name: desc.name.clone(),
            shard_idx: desc.shard_idx,
            data_offset: desc.data_offset,
            source_bytes: desc.n_bytes,
            dtype: desc.dtype,
            shape: desc.shape.clone(),
            kind,
            resident_bytes,
        });
        Ok(())
    }

    fn record_derived(&mut self, tensor: &MetalTensor) -> Result<(), MfError> {
        self.derived_allocations = self
            .derived_allocations
            .checked_add(1)
            .ok_or_else(|| MfError::LoadPolicy("derived allocation count overflow".to_string()))?;
        self.derived_bytes = self
            .derived_bytes
            .checked_add(tensor.n_bytes())
            .ok_or_else(|| MfError::LoadPolicy("derived allocation byte overflow".to_string()))?;
        Ok(())
    }
}

struct PlannedRetainedStorage {
    plan: RetainedStoragePlan,
    windows: Vec<MetalGgufBacking>,
    realized: Vec<Option<MetalTensor>>,
    cursor: usize,
}

impl PlannedRetainedStorage {
    fn load_direct(
        &mut self,
        ctx: &MetalContext,
        gguf: &GgufFile,
        desc: &TensorDesc,
    ) -> Result<(MetalTensor, SourceMaterialization), MfError> {
        let entry = self.plan.entries.get(self.cursor).ok_or_else(|| {
            MfError::LoadPolicy(format!(
                "retained storage received unexpected direct tensor {:?} at index {}",
                desc.name, self.cursor
            ))
        })?;
        if entry.request_index != self.cursor
            || entry.name != desc.name
            || entry.shard_idx != desc.shard_idx
            || entry.data_offset != desc.data_offset
            || entry.n_bytes != desc.n_bytes
        {
            return Err(MfError::LoadPolicy(format!(
                "retained storage request drift at direct index {}: entry={entry:?} desc={desc:?}",
                self.cursor
            )));
        }
        let (tensor, materialization) = match entry.disposition {
            RetainedStorageDisposition::View {
                window_index,
                buffer_offset,
            } => {
                let backing = self.windows.get(window_index).ok_or_else(|| {
                    MfError::LoadPolicy(format!(
                        "retained storage window index {window_index} is missing"
                    ))
                })?;
                let (eligibility, tensor) = backing.tensor(desc)?;
                let tensor = match (eligibility, tensor) {
                    (GgufBackingEligibility::Eligible, Some(tensor)) => tensor,
                    (reason, _) => {
                        return Err(MfError::LoadPolicy(format!(
                            "planned retained tensor {:?} failed realization: {reason:?}",
                            desc.name
                        )));
                    }
                };
                if tensor.offset != buffer_offset {
                    return Err(MfError::LoadPolicy(format!(
                        "retained tensor {:?} offset drift: actual={} planned={buffer_offset}",
                        desc.name, tensor.offset
                    )));
                }
                (tensor, SourceMaterialization::DirectView)
            }
            RetainedStorageDisposition::Alias {
                source_request_index,
            } => {
                if source_request_index >= self.cursor {
                    return Err(MfError::LoadPolicy(format!(
                        "retained alias {:?} has non-prior source {source_request_index}",
                        desc.name
                    )));
                }
                let tensor = self
                    .realized
                    .get(source_request_index)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        MfError::LoadPolicy(format!(
                            "retained alias {:?} source {source_request_index} is unrealized",
                            desc.name
                        ))
                    })?
                    .clone();
                (tensor, SourceMaterialization::DirectAlias)
            }
            RetainedStorageDisposition::CopyFallback { reason } => {
                if reason != RetainedStorageFallback::FinalPartialPage {
                    return Err(MfError::LoadPolicy(format!(
                        "retained tensor {:?} has disallowed copy fallback {reason:?}",
                        desc.name
                    )));
                }
                let tensor = MetalTensor::from_gguf_tensor(ctx, desc, gguf.slice(desc))?;
                (tensor, SourceMaterialization::TailFallback)
            }
        };
        self.realized[self.cursor] = Some(tensor.clone());
        self.cursor += 1;
        Ok((tensor, materialization))
    }

    fn validate_complete(&self, ledger: &WeightLoadLedger) -> Result<(), MfError> {
        if self.cursor != self.plan.entries.len() || self.realized.iter().any(Option::is_none) {
            return Err(MfError::LoadPolicy(format!(
                "retained storage consumption mismatch: consumed={} planned={} realized={}",
                self.cursor,
                self.plan.entries.len(),
                self.realized
                    .iter()
                    .filter(|tensor| tensor.is_some())
                    .count()
            )));
        }
        let view_count = self
            .plan
            .entries
            .iter()
            .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
            .count();
        let alias_count = self
            .plan
            .entries
            .iter()
            .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::Alias { .. }))
            .count();
        let fallback_count = self
            .plan
            .entries
            .iter()
            .filter(|entry| {
                matches!(
                    entry.disposition,
                    RetainedStorageDisposition::CopyFallback { .. }
                )
            })
            .count();
        if ledger.direct_copy_descriptors != 0
            || ledger.direct_view_descriptors != view_count
            || ledger.direct_view_bytes != self.plan.unique_view_bytes
            || ledger.direct_alias_descriptors != alias_count
            || ledger.direct_alias_bytes != self.plan.alias_bytes
            || ledger.tail_fallback_descriptors != fallback_count
            || ledger.tail_fallback_bytes != self.plan.unique_fallback_bytes
        {
            return Err(MfError::LoadPolicy(format!(
                concat!(
                    "retained storage ledger mismatch: copy={}/{} view={}/{} ",
                    "alias={}/{} fallback={}/{} planned_view={}/{} ",
                    "planned_alias={}/{} planned_fallback={}/{}"
                ),
                ledger.direct_copy_descriptors,
                ledger.direct_copy_bytes,
                ledger.direct_view_descriptors,
                ledger.direct_view_bytes,
                ledger.direct_alias_descriptors,
                ledger.direct_alias_bytes,
                ledger.tail_fallback_descriptors,
                ledger.tail_fallback_bytes,
                view_count,
                self.plan.unique_view_bytes,
                alias_count,
                self.plan.alias_bytes,
                fallback_count,
                self.plan.unique_fallback_bytes,
            )));
        }
        Ok(())
    }
}

struct PlannedOwnedStorage {
    plan: RetainedStoragePlan,
    resources: Vec<Buffer>,
    fallback_resources: HashMap<usize, usize>,
    realized: Vec<Option<MetalTensor>>,
    cursor: usize,
}

struct ParallelCopyTask<'a> {
    source: &'a [u8],
    destination: &'a mut [u8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParallelPopulationMethod {
    MmapCopy,
    Pread,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParallelDestinationLength {
    LogicalExact,
    PageRounded16K,
}

struct ParallelPreadTask<'a> {
    shard_idx: usize,
    source_offset: u64,
    destination: &'a mut [u8],
}

enum ParallelPopulationTask<'a> {
    MmapCopy(ParallelCopyTask<'a>),
    Pread(ParallelPreadTask<'a>),
}

#[derive(Clone, Copy, Debug)]
struct ParallelCopyUsage {
    minor_faults: i64,
    major_faults: i64,
    user_time_us: i64,
    system_time_us: i64,
}

#[derive(Clone, Copy, Debug)]
struct ParallelCopyProcUsage {
    instructions: u64,
    cycles: u64,
}

#[derive(Clone, Copy, Debug)]
struct ParallelCopyEndpointAccounting {
    user_cpu_us: u64,
    system_cpu_us: u64,
    total_cpu_us: u64,
    timer_minor_faults: u64,
    timer_major_faults: u64,
    instructions_delta_raw: u64,
    cycles_delta_raw: u64,
}

#[derive(Clone, Copy, Debug)]
struct ParallelCopyTiming {
    allocation_us: u64,
    source_us: u64,
    copy_us: u64,
    binding_us: u64,
    ready_us: u64,
}

fn parallel_copy_timeval_us(value: libc::timeval) -> Result<i64, MfError> {
    value
        .tv_sec
        .checked_mul(1_000_000)
        .and_then(|seconds| seconds.checked_add(i64::from(value.tv_usec)))
        .ok_or_else(|| MfError::LoadPolicy("parallel-copy CPU time overflow".to_string()))
}

fn capture_parallel_copy_usage() -> Result<ParallelCopyUsage, MfError> {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy getrusage failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    let usage = unsafe { usage.assume_init() };
    Ok(ParallelCopyUsage {
        minor_faults: usage.ru_minflt,
        major_faults: usage.ru_majflt,
        user_time_us: parallel_copy_timeval_us(usage.ru_utime)?,
        system_time_us: parallel_copy_timeval_us(usage.ru_stime)?,
    })
}

fn capture_parallel_copy_proc_usage() -> Result<ParallelCopyProcUsage, MfError> {
    let mut usage = MaybeUninit::<libc::rusage_info_v4>::zeroed();
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            usage.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    if rc != 0 {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy proc_pid_rusage v4 failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    let usage = unsafe { usage.assume_init() };
    Ok(ParallelCopyProcUsage {
        instructions: usage.ri_instructions,
        cycles: usage.ri_cycles,
    })
}

fn parallel_copy_i64_delta(after: i64, before: i64, label: &str) -> Result<u64, MfError> {
    let delta = after
        .checked_sub(before)
        .ok_or_else(|| MfError::LoadPolicy(format!("parallel-copy {label} counter underflow")))?;
    u64::try_from(delta)
        .map_err(|_| MfError::LoadPolicy(format!("parallel-copy {label} counter regressed")))
}

fn parallel_copy_u64_delta(after: u64, before: u64, label: &str) -> Result<u64, MfError> {
    after
        .checked_sub(before)
        .ok_or_else(|| MfError::LoadPolicy(format!("parallel-copy {label} counter regressed")))
}

fn finish_parallel_copy_accounting(
    before: ParallelCopyUsage,
    after: ParallelCopyUsage,
    proc_before: ParallelCopyProcUsage,
    proc_after: ParallelCopyProcUsage,
) -> Result<ParallelCopyEndpointAccounting, MfError> {
    let user_cpu_us = parallel_copy_i64_delta(after.user_time_us, before.user_time_us, "user CPU")?;
    let system_cpu_us =
        parallel_copy_i64_delta(after.system_time_us, before.system_time_us, "system CPU")?;
    let total_cpu_us = user_cpu_us
        .checked_add(system_cpu_us)
        .ok_or_else(|| MfError::LoadPolicy("parallel-copy total CPU overflow".to_string()))?;
    Ok(ParallelCopyEndpointAccounting {
        user_cpu_us,
        system_cpu_us,
        total_cpu_us,
        timer_minor_faults: parallel_copy_i64_delta(
            after.minor_faults,
            before.minor_faults,
            "minor fault",
        )?,
        timer_major_faults: parallel_copy_i64_delta(
            after.major_faults,
            before.major_faults,
            "major fault",
        )?,
        instructions_delta_raw: parallel_copy_u64_delta(
            proc_after.instructions,
            proc_before.instructions,
            "instruction",
        )?,
        cycles_delta_raw: parallel_copy_u64_delta(proc_after.cycles, proc_before.cycles, "cycle")?,
    })
}

fn validate_parallel_population(
    profile: &ParallelCopyProfile,
    population: ParallelPopulationMethod,
) -> Result<(), MfError> {
    if profile.id == ParallelCopyProfileId::A10bQ4xlV1
        && population != ParallelPopulationMethod::Pread
    {
        return Err(MfError::LoadPolicy(
            "A10B parallel population requires direct pread".to_string(),
        ));
    }
    if population == ParallelPopulationMethod::Pread && !profile.supports_direct_pread {
        return Err(MfError::LoadPolicy(format!(
            "parallel pread rejects unauthenticated profile {}",
            profile.id.label()
        )));
    }
    Ok(())
}

fn validate_parallel_destination_length(
    profile: &ParallelCopyProfile,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
) -> Result<(), MfError> {
    validate_parallel_population(profile, population)?;
    if destination_length == ParallelDestinationLength::PageRounded16K
        && (profile.id != ParallelCopyProfileId::A3bQ4kmV1
            || population != ParallelPopulationMethod::MmapCopy)
    {
        return Err(MfError::LoadPolicy(
            "page-rounded parallel copy requires authenticated A3B mmap population".to_string(),
        ));
    }
    Ok(())
}

fn parallel_destination_resource_length(
    logical_bytes: u64,
    destination_length: ParallelDestinationLength,
    max_buffer_length: usize,
) -> Result<usize, MfError> {
    let logical = usize::try_from(logical_bytes).map_err(|_| {
        MfError::LoadPolicy("parallel-copy resource length does not fit usize".to_string())
    })?;
    if logical == 0 {
        return Err(MfError::LoadPolicy(
            "parallel-copy resource length is zero".to_string(),
        ));
    }
    let allocated = match destination_length {
        ParallelDestinationLength::LogicalExact => logical,
        ParallelDestinationLength::PageRounded16K => logical
            .checked_add(16_383)
            .map(|value| value & !16_383)
            .ok_or_else(|| {
                MfError::LoadPolicy("page-rounded resource length overflow".to_string())
            })?,
    };
    if allocated > max_buffer_length {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy resource length {allocated} exceeds max buffer {max_buffer_length}"
        )));
    }
    Ok(allocated)
}

fn parallel_destination_accounting(
    expected: &[ModelWeightStorageIdentity],
    destination_length: ParallelDestinationLength,
    max_buffer_length: usize,
) -> Result<(u64, usize), MfError> {
    let mut allocated_bytes = 0u64;
    let mut padded_resources = 0usize;
    for identity in expected {
        let allocated = parallel_destination_resource_length(
            identity.source_bytes,
            destination_length,
            max_buffer_length,
        )?;
        allocated_bytes = allocated_bytes
            .checked_add(allocated as u64)
            .ok_or_else(|| {
                MfError::LoadPolicy("parallel-copy allocated byte overflow".to_string())
            })?;
        padded_resources += usize::from(allocated as u64 != identity.source_bytes);
    }
    Ok((allocated_bytes, padded_resources))
}

fn parallel_copy_marker_label(
    profile: &ParallelCopyProfile,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
) -> Result<&'static str, MfError> {
    validate_parallel_destination_length(profile, population, destination_length)?;
    Ok(match (population, destination_length) {
        (ParallelPopulationMethod::MmapCopy, ParallelDestinationLength::LogicalExact) => {
            "[metal-gguf-parallel-copied]"
        }
        (ParallelPopulationMethod::Pread, ParallelDestinationLength::LogicalExact) => {
            "[metal-gguf-parallel-pread]"
        }
        (ParallelPopulationMethod::MmapCopy, ParallelDestinationLength::PageRounded16K) => {
            "[metal-gguf-parallel-page-rounded]"
        }
        (ParallelPopulationMethod::Pread, ParallelDestinationLength::PageRounded16K) => {
            unreachable!("page-rounded pread is rejected above")
        }
    })
}

fn emit_parallel_copy_marker(
    profile: &ParallelCopyProfile,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
    timing: ParallelCopyTiming,
    accounting: Option<ParallelCopyEndpointAccounting>,
) -> Result<(), MfError> {
    let marker = parallel_copy_marker_label(profile, population, destination_length)?;
    match profile.marker_contract {
        ParallelCopyMarkerContract::A3b => {
            if accounting.is_some() {
                return Err(MfError::LoadPolicy(
                    "A3B parallel-copy marker received dense accounting".to_string(),
                ));
            }
            match destination_length {
                ParallelDestinationLength::LogicalExact => {
                    emit_metal_load_line(format_args!(
                        concat!(
                            "{} schema=1 resources=733 bytes=22123538944 ",
                            "workers=4 cuts=155,359,539 tasks=155,204,180,194 ",
                            "worker_bytes=5532746240,5462315776,5595522304,5532954624 ",
                            "first_offsets=10990048,5543736288,11006052064,16601574368 ",
                            "last_offsets=5392741344,11004937952,16450579424,22134520800 ",
                            "create=shared,default_cache,default observed=shared,default_cache,tracked ",
                            "page=16384 alignment=32 max_buffer=77309411328 mapped=22134528992 ",
                            "layout=0x5ae645df5cf7d568 ",
                            "inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5 ",
                            "plan=fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af ",
                            "allocation_us={} source_us={} copy_us={} binding_us={} ready_us={}"
                        ),
                        marker,
                        timing.allocation_us,
                        timing.source_us,
                        timing.copy_us,
                        timing.binding_us,
                        timing.ready_us,
                    ));
                }
                ParallelDestinationLength::PageRounded16K => {
                    emit_metal_load_line(format_args!(
                        concat!(
                            "{} schema=2 resources=733 logical_bytes=22123538944 ",
                            "allocated_bytes=22126297088 padding_bytes=2758144 ",
                            "padded_resources=232 workers=4 cuts=155,359,539 ",
                            "tasks=155,204,180,194 ",
                            "worker_bytes=5532746240,5462315776,5595522304,5532954624 ",
                            "first_offsets=10990048,5543736288,11006052064,16601574368 ",
                            "last_offsets=5392741344,11004937952,16450579424,22134520800 ",
                            "create=shared,default_cache,default observed=shared,default_cache,tracked ",
                            "page=16384 alignment=32 max_buffer=77309411328 mapped=22134528992 ",
                            "layout=0x5ae645df5cf7d568 ",
                            "inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5 ",
                            "plan=fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af ",
                            "allocation_us={} source_us={} copy_us={} binding_us={} ready_us={}"
                        ),
                        marker,
                        timing.allocation_us,
                        timing.source_us,
                        timing.copy_us,
                        timing.binding_us,
                        timing.ready_us,
                    ));
                }
            }
        }
        ParallelCopyMarkerContract::A10bSchema2 | ParallelCopyMarkerContract::DenseSchema2 => {
            let accounting = accounting.ok_or_else(|| {
                MfError::LoadPolicy(
                    "profile parallel-copy marker is missing endpoint accounting".to_string(),
                )
            })?;
            let mapped_bytes =
                profile
                    .shard_mapped_lengths
                    .iter()
                    .try_fold(0u64, |total, &length| {
                        total.checked_add(length as u64).ok_or_else(|| {
                            MfError::LoadPolicy("parallel-copy mapped byte overflow".to_string())
                        })
                    })?;
            let boundary = profile.boundaries;
            emit_metal_load_line(format_args!(
                concat!(
                    "{} schema=2 profile={} ",
                    "resources={} bytes={} workers=4 cuts={},{},{} ",
                    "tasks={},{},{},{} worker_bytes={},{},{},{} ",
                    "w0_first={},{},{},{},{} w0_last={},{},{},{},{} ",
                    "w1_first={},{},{},{},{} w1_last={},{},{},{},{} ",
                    "w2_first={},{},{},{},{} w2_last={},{},{},{},{} ",
                    "w3_first={},{},{},{},{} w3_last={},{},{},{},{} ",
                    "create=shared,default_cache,default ",
                    "observed=shared,default_cache,tracked ",
                    "page=16384 alignment=32 max_buffer=77309411328 ",
                    "mapped={} layout={:#018x} ",
                    "inventory={} allocation_us={} source_us={} copy_us={} ",
                    "binding_us={} ready_us={} user_cpu_us={} system_cpu_us={} ",
                    "total_cpu_us={} timer_minor_faults={} timer_major_faults={} ",
                    "instructions_delta_raw={} cycles_delta_raw={}"
                ),
                marker,
                profile.id.label(),
                profile.request_count,
                profile.source_bytes,
                profile.cuts[0],
                profile.cuts[1],
                profile.cuts[2],
                profile.task_counts[0],
                profile.task_counts[1],
                profile.task_counts[2],
                profile.task_counts[3],
                profile.worker_bytes[0],
                profile.worker_bytes[1],
                profile.worker_bytes[2],
                profile.worker_bytes[3],
                boundary[0].first.request_index,
                boundary[0].first.name,
                boundary[0].first.shard_idx,
                boundary[0].first.source_offset,
                boundary[0].first.source_bytes,
                boundary[0].last.request_index,
                boundary[0].last.name,
                boundary[0].last.shard_idx,
                boundary[0].last.source_offset,
                boundary[0].last.source_bytes,
                boundary[1].first.request_index,
                boundary[1].first.name,
                boundary[1].first.shard_idx,
                boundary[1].first.source_offset,
                boundary[1].first.source_bytes,
                boundary[1].last.request_index,
                boundary[1].last.name,
                boundary[1].last.shard_idx,
                boundary[1].last.source_offset,
                boundary[1].last.source_bytes,
                boundary[2].first.request_index,
                boundary[2].first.name,
                boundary[2].first.shard_idx,
                boundary[2].first.source_offset,
                boundary[2].first.source_bytes,
                boundary[2].last.request_index,
                boundary[2].last.name,
                boundary[2].last.shard_idx,
                boundary[2].last.source_offset,
                boundary[2].last.source_bytes,
                boundary[3].first.request_index,
                boundary[3].first.name,
                boundary[3].first.shard_idx,
                boundary[3].first.source_offset,
                boundary[3].first.source_bytes,
                boundary[3].last.request_index,
                boundary[3].last.name,
                boundary[3].last.shard_idx,
                boundary[3].last.source_offset,
                boundary[3].last.source_bytes,
                mapped_bytes,
                profile.descriptor_layout_digest,
                profile.inventory_digest,
                timing.allocation_us,
                timing.source_us,
                timing.copy_us,
                timing.binding_us,
                timing.ready_us,
                accounting.user_cpu_us,
                accounting.system_cpu_us,
                accounting.total_cpu_us,
                accounting.timer_minor_faults,
                accounting.timer_major_faults,
                accounting.instructions_delta_raw,
                accounting.cycles_delta_raw,
            ));
        }
    }
    Ok(())
}

struct PlannedParallelCopiedStorage {
    profile: &'static ParallelCopyProfile,
    destination_length: ParallelDestinationLength,
    expected: Vec<ModelWeightStorageIdentity>,
    sorted_request_indices: Vec<usize>,
    resources: Vec<Buffer>,
    tensors: Vec<MetalTensor>,
    cursor: usize,
}

fn frozen_parallel_copy_order(
    profile: &ParallelCopyProfile,
    expected: &[ModelWeightStorageIdentity],
) -> Result<Vec<usize>, MfError> {
    if expected.len() != profile.request_count
        || expected.iter().any(|identity| {
            identity.kind != ModelWeightStorageKind::Direct
                || identity.source_bytes == 0
                || identity.resident_bytes != identity.source_bytes
        })
    {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy {} request inventory is not all-direct and nonempty",
            profile.id.label()
        )));
    }

    let mut sorted_request_indices = (0..expected.len()).collect::<Vec<_>>();
    sorted_request_indices.sort_by_key(|&request_index| {
        let identity = &expected[request_index];
        (identity.shard_idx, identity.data_offset, request_index)
    });
    let mut permutation_check = sorted_request_indices.clone();
    permutation_check.sort_unstable();
    if permutation_check.iter().copied().ne(0..expected.len()) {
        return Err(MfError::LoadPolicy(
            "parallel-copy schedule is not a complete permutation".to_string(),
        ));
    }

    let boundaries = [
        0,
        profile.cuts[0],
        profile.cuts[1],
        profile.cuts[2],
        expected.len(),
    ];
    if boundaries[0] != 0
        || boundaries[GGUF_OWNED_WORKERS] != expected.len()
        || boundaries.windows(2).any(|pair| pair[0] >= pair[1])
        || boundaries
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .ne(profile.task_counts)
        || profile.task_counts.iter().sum::<usize>() != expected.len()
    {
        return Err(MfError::LoadPolicy(
            "parallel-copy frozen boundaries drifted".to_string(),
        ));
    }
    let mut total_bytes = 0u64;
    for worker in 0..GGUF_OWNED_WORKERS {
        let partition = &sorted_request_indices[boundaries[worker]..boundaries[worker + 1]];
        let worker_bytes = partition.iter().try_fold(0u64, |total, &request_index| {
            total
                .checked_add(expected[request_index].source_bytes)
                .ok_or_else(|| {
                    MfError::LoadPolicy("parallel-copy partition byte overflow".to_string())
                })
        })?;
        let first = &expected[partition[0]];
        let first_index = partition[0];
        let last_index = *partition.last().expect("partition is nonempty");
        let last = &expected[last_index];
        let frozen = profile.boundaries[worker];
        let first_matches = first_index == frozen.first.request_index
            && first.name == frozen.first.name
            && first.shard_idx == frozen.first.shard_idx
            && first.data_offset == frozen.first.source_offset
            && first.source_bytes == frozen.first.source_bytes;
        let last_matches = last_index == frozen.last.request_index
            && last.name == frozen.last.name
            && last.shard_idx == frozen.last.shard_idx
            && last.data_offset == frozen.last.source_offset
            && last.source_bytes == frozen.last.source_bytes;
        if partition.len() != profile.task_counts[worker]
            || worker_bytes != profile.worker_bytes[worker]
            || !first_matches
            || !last_matches
        {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy frozen partition {worker} drifted"
            )));
        }
        total_bytes = total_bytes.checked_add(worker_bytes).ok_or_else(|| {
            MfError::LoadPolicy("parallel-copy schedule byte overflow".to_string())
        })?;
    }
    if total_bytes != profile.source_bytes {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy schedule bytes drifted: {total_bytes}"
        )));
    }
    Ok(sorted_request_indices)
}

unsafe fn exclusive_buffer_bytes_mut(buffer: &mut Buffer, logical_length: usize) -> &mut [u8] {
    // SAFETY: the caller proves this logical prefix is nonempty and within the
    // CPU-accessible resource, pairwise disjoint from every other destination,
    // disjoint from all immutable sources, and exclusively borrowed until the
    // slice dies. Any physical padding remains uninitialized and inaccessible.
    unsafe {
        std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u8>(), logical_length)
    }
}

fn validate_parallel_copied_topology(
    profile: &ParallelCopyProfile,
    destination_length: ParallelDestinationLength,
    expected: &[ModelWeightStorageIdentity],
    resources: &[Buffer],
    tensors: &[MetalTensor],
) -> Result<(), MfError> {
    if expected.len() != profile.request_count
        || resources.len() != expected.len()
        || tensors.len() != expected.len()
    {
        return Err(MfError::LoadPolicy(
            "parallel-copy topology count drifted".to_string(),
        ));
    }
    let mut resource_identities = HashSet::with_capacity(resources.len());
    let mut resource_bytes = 0u64;
    for (index, ((identity, resource), tensor)) in
        expected.iter().zip(resources).zip(tensors).enumerate()
    {
        let resource_identity = Retained::as_ptr(resource) as *const () as usize;
        let expected_resource_length = parallel_destination_resource_length(
            identity.source_bytes,
            destination_length,
            usize::MAX,
        )?;
        resource_bytes = resource_bytes
            .checked_add(resource.length() as u64)
            .ok_or_else(|| MfError::LoadPolicy("parallel-copy byte overflow".to_string()))?;
        if !resource_identities.insert(resource_identity)
            || resource.length() != expected_resource_length
            || resource.storageMode() != MTLStorageMode::Shared
            || resource.cpuCacheMode() != MTLCPUCacheMode::DefaultCache
            || resource.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
            || Retained::as_ptr(&tensor.buffer) != Retained::as_ptr(resource)
            || tensor.offset != 0
            || tensor.n_bytes() != identity.resident_bytes
            || tensor.dtype != identity.dtype
            || tensor.shape != identity.shape
            || tensor.provenance() != MetalTensorProvenance::OwnedWeightReadOnly
        {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy resource or tensor {index} drifted"
            )));
        }
    }
    let (expected_resource_bytes, padded_resources) =
        parallel_destination_accounting(expected, destination_length, usize::MAX)?;
    let accounting_matches = match destination_length {
        ParallelDestinationLength::LogicalExact => {
            expected_resource_bytes == profile.source_bytes && padded_resources == 0
        }
        ParallelDestinationLength::PageRounded16K => {
            expected_resource_bytes == GGUF_PAGE_ROUNDED_A3B_ALLOCATED_BYTES
                && expected_resource_bytes.checked_sub(profile.source_bytes)
                    == Some(GGUF_PAGE_ROUNDED_A3B_PADDING_BYTES)
                && padded_resources == GGUF_PAGE_ROUNDED_A3B_PADDED_RESOURCES
        }
    };
    if resource_bytes != expected_resource_bytes || !accounting_matches {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy resource accounting drifted: actual={resource_bytes} expected={expected_resource_bytes} padded={padded_resources}"
        )));
    }
    Ok(())
}

fn owned_arena_four_worker_boundaries(
    length: usize,
    page_size: usize,
) -> Result<[usize; 5], MfError> {
    if page_size == 0 || !length.is_multiple_of(page_size) {
        return Err(MfError::LoadPolicy(
            "owned arena range is not page aligned".to_string(),
        ));
    }
    let pages = length / page_size;
    if pages < GGUF_OWNED_WORKERS {
        return Err(MfError::LoadPolicy(
            "owned arena range has fewer than four pages".to_string(),
        ));
    }
    let boundaries =
        std::array::from_fn(|worker| page_size * (worker * pages / GGUF_OWNED_WORKERS));
    if boundaries[0] != 0 || boundaries[GGUF_OWNED_WORKERS] != length {
        return Err(MfError::LoadPolicy(
            "owned arena worker boundaries do not cover the range".to_string(),
        ));
    }
    if boundaries.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(MfError::LoadPolicy(
            "owned arena worker boundaries overlap or are empty".to_string(),
        ));
    }
    Ok(boundaries)
}

fn copy_owned_arena_four_workers(
    source: &[u8],
    destination: &Buffer,
    page_size: usize,
) -> Result<(), MfError> {
    if source.len() != destination.length() {
        return Err(MfError::LoadPolicy(format!(
            "owned arena source {} differs from destination {}",
            source.len(),
            destination.length()
        )));
    }
    let boundaries = owned_arena_four_worker_boundaries(source.len(), page_size)?;
    // SAFETY: the anonymous shared buffer is retained by the caller, has exactly
    // source.len() bytes, and no typed views exist until all scoped workers join.
    let destination = unsafe {
        std::slice::from_raw_parts_mut(destination.contents().as_ptr().cast::<u8>(), source.len())
    };
    std::thread::scope(|scope| {
        let mut source_tail = source;
        let mut destination_tail = destination;
        let mut previous = 0usize;
        for &boundary in boundaries.iter().skip(1) {
            let length = boundary - previous;
            let (source_chunk, next_source) = source_tail.split_at(length);
            let (destination_chunk, next_destination) = destination_tail.split_at_mut(length);
            scope.spawn(move || destination_chunk.copy_from_slice(source_chunk));
            source_tail = next_source;
            destination_tail = next_destination;
            previous = boundary;
        }
    });
    Ok(())
}

fn copy_owned_arena_serial(source: &[u8], destination: &Buffer) -> Result<(), MfError> {
    if source.len() != destination.length() {
        return Err(MfError::LoadPolicy(format!(
            "owned fallback source {} differs from destination {}",
            source.len(),
            destination.length()
        )));
    }
    // SAFETY: the fallback buffer is retained, exact-sized, and has no live views.
    let destination = unsafe {
        std::slice::from_raw_parts_mut(destination.contents().as_ptr().cast::<u8>(), source.len())
    };
    destination.copy_from_slice(source);
    Ok(())
}

impl PlannedOwnedStorage {
    fn load_direct(
        &mut self,
        desc: &TensorDesc,
    ) -> Result<(MetalTensor, SourceMaterialization), MfError> {
        let entry = self.plan.entries.get(self.cursor).ok_or_else(|| {
            MfError::LoadPolicy(format!(
                "owned storage received unexpected tensor {:?} at index {}",
                desc.name, self.cursor
            ))
        })?;
        if entry.request_index != self.cursor
            || entry.name != desc.name
            || entry.shard_idx != desc.shard_idx
            || entry.data_offset != desc.data_offset
            || entry.n_bytes != desc.n_bytes
        {
            return Err(MfError::LoadPolicy(format!(
                "owned storage request drift at index {}: entry={entry:?} desc={desc:?}",
                self.cursor
            )));
        }
        let (tensor, materialization) = match entry.disposition {
            RetainedStorageDisposition::View {
                window_index,
                buffer_offset,
            } => {
                let buffer = self.resources.get(window_index).ok_or_else(|| {
                    MfError::LoadPolicy(format!("owned storage window {window_index} is missing"))
                })?;
                let tensor = MetalTensor::owned_weight_view(
                    buffer.clone(),
                    buffer_offset,
                    desc.shape.clone(),
                    desc.dtype,
                    GGUF_NO_COPY_ALIGNMENT,
                )?;
                (tensor, SourceMaterialization::DirectCopy)
            }
            RetainedStorageDisposition::Alias {
                source_request_index,
            } => {
                if source_request_index >= self.cursor {
                    return Err(MfError::LoadPolicy(format!(
                        "owned alias {:?} has non-prior source {source_request_index}",
                        desc.name
                    )));
                }
                let tensor = self
                    .realized
                    .get(source_request_index)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        MfError::LoadPolicy(format!(
                            "owned alias {:?} source {source_request_index} is unrealized",
                            desc.name
                        ))
                    })?
                    .clone();
                (tensor, SourceMaterialization::DirectAlias)
            }
            RetainedStorageDisposition::CopyFallback { reason } => {
                if reason != RetainedStorageFallback::FinalPartialPage {
                    return Err(MfError::LoadPolicy(format!(
                        "owned tensor {:?} has disallowed fallback {reason:?}",
                        desc.name
                    )));
                }
                let resource_index = *self
                    .fallback_resources
                    .get(&entry.request_index)
                    .ok_or_else(|| {
                        MfError::LoadPolicy(format!(
                            "owned fallback resource for {:?} is missing",
                            desc.name
                        ))
                    })?;
                let tensor = MetalTensor::owned_weight_view(
                    self.resources[resource_index].clone(),
                    0,
                    desc.shape.clone(),
                    desc.dtype,
                    GGUF_NO_COPY_ALIGNMENT,
                )?;
                (tensor, SourceMaterialization::TailFallback)
            }
        };
        if tensor.provenance() != MetalTensorProvenance::OwnedWeightReadOnly {
            return Err(MfError::LoadPolicy(format!(
                "owned tensor {:?} has writable or retained provenance",
                desc.name
            )));
        }
        self.realized[self.cursor] = Some(tensor.clone());
        self.cursor += 1;
        Ok((tensor, materialization))
    }

    fn validate_complete(&self, ledger: &WeightLoadLedger) -> Result<(), MfError> {
        if self.cursor != self.plan.entries.len() || self.realized.iter().any(Option::is_none) {
            return Err(MfError::LoadPolicy(format!(
                "owned storage consumption mismatch: consumed={} planned={} realized={}",
                self.cursor,
                self.plan.entries.len(),
                self.realized
                    .iter()
                    .filter(|tensor| tensor.is_some())
                    .count()
            )));
        }
        if self.resources.len() != 2
            || self.resources[0].length() as u64 != GGUF_OWNED_A3B_WINDOW_BYTES
            || self.resources[1].length() as u64 != GGUF_OWNED_A3B_FALLBACK_BYTES
            || self
                .resources
                .iter()
                .any(|buffer| buffer.storageMode() != MTLStorageMode::Shared)
        {
            return Err(MfError::LoadPolicy(
                "owned storage physical resource ledger drifted".to_string(),
            ));
        }
        let fallback_entry = self
            .plan
            .entries
            .iter()
            .find(|entry| {
                matches!(
                    entry.disposition,
                    RetainedStorageDisposition::CopyFallback { .. }
                )
            })
            .ok_or_else(|| MfError::LoadPolicy("owned fallback entry is missing".to_string()))?;
        if self.fallback_resources.len() != 1
            || self.fallback_resources.get(&fallback_entry.request_index) != Some(&1)
        {
            return Err(MfError::LoadPolicy(
                "owned fallback resource map drifted".to_string(),
            ));
        }
        for (entry, tensor) in self.plan.entries.iter().zip(self.realized.iter().flatten()) {
            let (resource_index, offset) = match entry.disposition {
                RetainedStorageDisposition::View {
                    window_index,
                    buffer_offset,
                } => (window_index, buffer_offset),
                RetainedStorageDisposition::CopyFallback { .. } => (1, 0),
                RetainedStorageDisposition::Alias { .. } => {
                    return Err(MfError::LoadPolicy(
                        "owned A3B sentinel unexpectedly contains an alias".to_string(),
                    ));
                }
            };
            if Retained::as_ptr(&tensor.buffer) != Retained::as_ptr(&self.resources[resource_index])
                || tensor.offset != offset
            {
                return Err(MfError::LoadPolicy(format!(
                    "owned tensor {:?} resource identity drifted",
                    entry.name
                )));
            }
        }
        if ledger.source_descriptors != GGUF_OWNED_A3B_REQUESTS
            || ledger.source_bytes != GGUF_OWNED_A3B_SOURCE_BYTES
            || ledger.direct_copy_descriptors != GGUF_OWNED_A3B_VIEWS
            || ledger.direct_copy_bytes != GGUF_OWNED_A3B_VIEW_BYTES
            || ledger.tail_fallback_descriptors != 1
            || ledger.tail_fallback_bytes != GGUF_OWNED_A3B_FALLBACK_BYTES
            || ledger.direct_view_descriptors != 0
            || ledger.direct_view_bytes != 0
            || ledger.direct_alias_descriptors != 0
            || ledger.direct_alias_bytes != 0
            || ledger.converted_descriptors != 0
            || ledger.converted_source_bytes != 0
            || ledger.converted_resident_bytes != 0
            || self
                .realized
                .iter()
                .flatten()
                .any(|tensor| tensor.provenance() != MetalTensorProvenance::OwnedWeightReadOnly)
        {
            return Err(MfError::LoadPolicy(format!(
                concat!(
                    "owned storage logical ledger drift: source={}/{} copy={}/{} ",
                    "tail={}/{} view={}/{} alias={}/{} converted={}/{}/{}"
                ),
                ledger.source_descriptors,
                ledger.source_bytes,
                ledger.direct_copy_descriptors,
                ledger.direct_copy_bytes,
                ledger.tail_fallback_descriptors,
                ledger.tail_fallback_bytes,
                ledger.direct_view_descriptors,
                ledger.direct_view_bytes,
                ledger.direct_alias_descriptors,
                ledger.direct_alias_bytes,
                ledger.converted_descriptors,
                ledger.converted_source_bytes,
                ledger.converted_resident_bytes,
            )));
        }
        Ok(())
    }
}

impl PlannedParallelCopiedStorage {
    fn load_direct(
        &mut self,
        desc: &TensorDesc,
    ) -> Result<(MetalTensor, SourceMaterialization), MfError> {
        let expected = self.expected.get(self.cursor).ok_or_else(|| {
            MfError::LoadPolicy(format!(
                "parallel-copy storage received unexpected tensor {:?} at index {}",
                desc.name, self.cursor
            ))
        })?;
        let actual = ModelWeightStorageIdentity {
            name: desc.name.clone(),
            shard_idx: desc.shard_idx,
            data_offset: desc.data_offset,
            source_bytes: desc.n_bytes,
            dtype: desc.dtype,
            shape: desc.shape.clone(),
            kind: ModelWeightStorageKind::Direct,
            resident_bytes: desc.n_bytes,
        };
        if actual != *expected {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy request drift at index {}: actual={actual:?} expected={expected:?}",
                self.cursor
            )));
        }
        let tensor = self.tensors.get(self.cursor).ok_or_else(|| {
            MfError::LoadPolicy(format!("parallel-copy tensor {} is missing", self.cursor))
        })?;
        if Retained::as_ptr(&tensor.buffer) != Retained::as_ptr(&self.resources[self.cursor])
            || tensor.offset != 0
            || tensor.n_bytes() != desc.n_bytes
            || tensor.dtype != desc.dtype
            || tensor.shape != desc.shape
            || tensor.provenance() != MetalTensorProvenance::OwnedWeightReadOnly
        {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy tensor {} changed before consumption",
                self.cursor
            )));
        }
        self.cursor += 1;
        Ok((tensor.clone(), SourceMaterialization::DirectCopy))
    }

    fn validate_complete(&self, ledger: &WeightLoadLedger) -> Result<(), MfError> {
        if self.cursor != self.expected.len() {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy storage consumption mismatch: consumed={} expected={}",
                self.cursor,
                self.expected.len()
            )));
        }
        let schedule = frozen_parallel_copy_order(self.profile, &self.expected)?;
        if schedule != self.sorted_request_indices {
            return Err(MfError::LoadPolicy(
                "parallel-copy schedule attestation drifted".to_string(),
            ));
        }
        validate_parallel_copied_topology(
            self.profile,
            self.destination_length,
            &self.expected,
            &self.resources,
            &self.tensors,
        )?;
        if ledger.source_descriptors != self.profile.request_count
            || ledger.source_bytes != self.profile.source_bytes
            || ledger.direct_copy_descriptors != self.profile.request_count
            || ledger.direct_copy_bytes != self.profile.source_bytes
            || ledger.direct_view_descriptors != 0
            || ledger.direct_view_bytes != 0
            || ledger.direct_alias_descriptors != 0
            || ledger.direct_alias_bytes != 0
            || ledger.tail_fallback_descriptors != 0
            || ledger.tail_fallback_bytes != 0
            || ledger.converted_descriptors != 0
            || ledger.converted_source_bytes != 0
            || ledger.converted_resident_bytes != 0
            || ledger.derived_allocations != 0
            || ledger.derived_bytes != 0
        {
            return Err(MfError::LoadPolicy(format!(
                concat!(
                    "parallel-copy logical ledger drift: source={}/{} copy={}/{} ",
                    "view={}/{} alias={}/{} tail={}/{} converted={}/{}/{} derived={}/{}"
                ),
                ledger.source_descriptors,
                ledger.source_bytes,
                ledger.direct_copy_descriptors,
                ledger.direct_copy_bytes,
                ledger.direct_view_descriptors,
                ledger.direct_view_bytes,
                ledger.direct_alias_descriptors,
                ledger.direct_alias_bytes,
                ledger.tail_fallback_descriptors,
                ledger.tail_fallback_bytes,
                ledger.converted_descriptors,
                ledger.converted_source_bytes,
                ledger.converted_resident_bytes,
                ledger.derived_allocations,
                ledger.derived_bytes,
            )));
        }
        Ok(())
    }

    #[cfg(test)]
    fn validate_source_bytes(
        &self,
        gguf: &GgufFile,
        expected: &[ModelWeightStorageRequest<'_>],
    ) -> Result<(), MfError> {
        if expected.len() != self.expected.len() {
            return Err(MfError::LoadPolicy(
                "parallel-copy byte audit request count drifted".to_string(),
            ));
        }
        for (index, ((identity, buffer), request)) in self
            .expected
            .iter()
            .zip(&self.resources)
            .zip(expected)
            .enumerate()
        {
            if expected_model_weight_identity(request) != *identity {
                return Err(MfError::LoadPolicy(format!(
                    "parallel-copy byte audit identity {index} drifted"
                )));
            }
            let source = gguf.try_slice(request.desc).map_err(|error| {
                MfError::LoadPolicy(format!("parallel-copy byte audit source {index}: {error}"))
            })?;
            // SAFETY: the resource is CPU-accessible and retained by self. The
            // logical prefix has the exact source length, and no mutable CPU
            // access exists after construction. Physical padding is excluded.
            let actual = unsafe {
                std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), source.len())
            };
            if actual != source {
                return Err(MfError::LoadPolicy(format!(
                    "parallel-copy byte audit resource {index} differs from source"
                )));
            }
        }
        Ok(())
    }
}

enum DirectStorage {
    Copied,
    ForcedExact27B(MetalGgufBacking),
    ForcedPlanned(PlannedRetainedStorage),
    ForcedOwned(PlannedOwnedStorage),
    ForcedParallelCopied(PlannedParallelCopiedStorage),
}

fn create_a10b_parallel_residency_set(
    ctx: &MetalContext,
    direct_storage: &DirectStorage,
) -> Result<Option<MetalModelResidencySetGuard>, MfError> {
    let DirectStorage::ForcedParallelCopied(storage) = direct_storage else {
        return Ok(None);
    };
    if storage.profile.id != ParallelCopyProfileId::A10bQ4xlV1 {
        return Ok(None);
    }
    if storage.resources.len() != storage.profile.request_count {
        return Err(MfError::LoadPolicy(format!(
            "A10B residency resource count drifted: {}/{}",
            storage.resources.len(),
            storage.profile.request_count
        )));
    }

    let descriptor = MTLResidencySetDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str("qwen-a10b-parallel-pread")));
    // SAFETY: initialCapacity is advisory and equals the authenticated resource count.
    unsafe { descriptor.setInitialCapacity(storage.resources.len()) };
    let set = ctx
        .device
        .newResidencySetWithDescriptor_error(&descriptor)
        .map_err(|error| {
            let error: Retained<NSError> = error;
            MfError::LoadPolicy(format!(
                "A10B residency set creation failed: {}",
                error.localizedDescription()
            ))
        })?;
    let mut expected_allocated_bytes = 0u64;
    for buffer in &storage.resources {
        let allocation: &ProtocolObject<dyn MTLAllocation> = ProtocolObject::from_ref(&**buffer);
        let allocated_bytes = u64::try_from(allocation.allocatedSize()).map_err(|_| {
            MfError::LoadPolicy("A10B residency allocation size does not fit u64".to_string())
        })?;
        expected_allocated_bytes = expected_allocated_bytes
            .checked_add(allocated_bytes)
            .ok_or_else(|| {
                MfError::LoadPolicy("A10B residency allocated byte overflow".to_string())
            })?;
        set.addAllocation(allocation);
    }
    set.commit();
    if set.allocationCount() != storage.resources.len()
        || set.allocatedSize() != expected_allocated_bytes
    {
        let actual_count = set.allocationCount();
        let actual_bytes = set.allocatedSize();
        set.endResidency();
        return Err(MfError::LoadPolicy(format!(
            concat!(
                "A10B residency set commitment drifted: allocations={}/{} ",
                "bytes={}/{}"
            ),
            actual_count,
            storage.resources.len(),
            actual_bytes,
            expected_allocated_bytes,
        )));
    }
    let started = std::time::Instant::now();
    set.requestResidency();
    let residency_ms = started.elapsed().as_secs_f64() * 1e3;
    ctx.queue.addResidencySet(&set);
    emit_metal_load_line(format_args!(
        concat!(
            "[metal-gguf-parallel-residency] profile={} allocations={} ",
            "allocated_bytes={} request_ms={:.3} rollback=QWEN_GGUF_PARALLEL_COPY=0"
        ),
        storage.profile.id.label(),
        set.allocationCount(),
        set.allocatedSize(),
        residency_ms,
    ));
    Ok(Some(MetalModelResidencySetGuard {
        queue: ctx.queue.clone(),
        set,
    }))
}

struct MetalWeightLoader<'a> {
    ctx: &'a MetalContext,
    gguf: &'a GgufFile,
    direct_storage: DirectStorage,
    router_f16: bool,
    seen_forced: HashSet<(usize, u64, u64)>,
    ledger: WeightLoadLedger,
}

impl<'a> MetalWeightLoader<'a> {
    fn new(
        ctx: &'a MetalContext,
        gguf: &'a GgufFile,
        direct_storage: DirectStorage,
        router_f16: bool,
    ) -> Self {
        Self {
            ctx,
            gguf,
            direct_storage,
            router_f16,
            seen_forced: HashSet::new(),
            ledger: WeightLoadLedger::default(),
        }
    }

    fn is_forced_exact_27b(&self) -> bool {
        matches!(self.direct_storage, DirectStorage::ForcedExact27B(_))
    }

    fn record_source(
        &mut self,
        desc: &TensorDesc,
        materialization: SourceMaterialization,
        resident_bytes: u64,
    ) -> Result<(), MfError> {
        if self.is_forced_exact_27b()
            && !self
                .seen_forced
                .insert((desc.shard_idx, desc.data_offset, desc.n_bytes))
        {
            return Err(MfError::LoadPolicy(format!(
                "forced no-copy materialized tensor {:?} more than once",
                desc.name
            )));
        }
        self.ledger
            .record_source(desc, materialization, resident_bytes)
    }

    fn load_direct(&mut self, desc: &TensorDesc) -> Result<MetalTensor, MfError> {
        let (tensor, materialization) = match &mut self.direct_storage {
            DirectStorage::Copied => {
                let tensor = MetalTensor::from_gguf_tensor(self.ctx, desc, self.gguf.slice(desc))?;
                (tensor, SourceMaterialization::DirectCopy)
            }
            DirectStorage::ForcedExact27B(backing) => {
                let (eligibility, tensor) = backing.tensor(desc)?;
                match (eligibility, tensor) {
                    (GgufBackingEligibility::Eligible, Some(tensor)) => {
                        (tensor, SourceMaterialization::DirectView)
                    }
                    (GgufBackingEligibility::FinalPartialPage, None)
                        if desc.name == GGUF_NO_COPY_27B_TAIL_NAME
                            && desc.n_bytes == GGUF_NO_COPY_27B_TAIL_BYTES =>
                    {
                        let tensor =
                            MetalTensor::from_gguf_tensor(self.ctx, desc, self.gguf.slice(desc))?;
                        (tensor, SourceMaterialization::TailFallback)
                    }
                    (reason, _) => Err(MfError::LoadPolicy(format!(
                        "forced no-copy rejected direct tensor {:?}: {reason:?}",
                        desc.name
                    )))?,
                }
            }
            DirectStorage::ForcedPlanned(storage) => {
                storage.load_direct(self.ctx, self.gguf, desc)?
            }
            DirectStorage::ForcedOwned(storage) => storage.load_direct(desc)?,
            DirectStorage::ForcedParallelCopied(storage) => storage.load_direct(desc)?,
        };
        self.record_source(desc, materialization, tensor.n_bytes())?;
        Ok(tensor)
    }

    fn load_f32(&mut self, desc: &TensorDesc) -> Result<MetalTensor, MfError> {
        if desc.dtype == GgmlType::F32 {
            return self.load_direct(desc);
        }
        let f32 = crate::codec::dequant_to_f32(desc, self.gguf.slice(desc))?;
        let tensor = MetalTensor::from_bytes(
            self.ctx,
            bytemuck::cast_slice(&f32),
            desc.shape.clone(),
            GgmlType::F32,
        )?;
        self.record_source(desc, SourceMaterialization::ConvertedF32, tensor.n_bytes())?;
        Ok(tensor)
    }

    fn load_weight(&mut self, desc: &TensorDesc) -> Result<MetalTensor, MfError> {
        if weight_dtype_kept_native(desc.dtype) {
            return self.load_direct(desc);
        }
        eprintln!(
            "[metal-load] {} is {:?}; dequanting to F32 (no active native path)",
            desc.name, desc.dtype
        );
        self.load_f32(desc)
    }

    fn load_embedding(
        &mut self,
        desc: &TensorDesc,
        native_quant: bool,
    ) -> Result<MetalTensor, MfError> {
        if matches!(desc.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) || native_quant {
            self.load_direct(desc)
        } else {
            self.load_f32(desc)
        }
    }

    fn load_router_weight(&mut self, desc: &TensorDesc) -> Result<MetalTensor, MfError> {
        if !self.router_f16 {
            return self.load_f32(desc);
        }
        let f32 = crate::codec::dequant_to_f32(desc, self.gguf.slice(desc))?;
        let f16: Vec<half::f16> = f32.iter().copied().map(half::f16::from_f32).collect();
        let tensor = MetalTensor::from_bytes(
            self.ctx,
            bytemuck::cast_slice(&f16),
            desc.shape.clone(),
            GgmlType::F16,
        )?;
        self.record_source(desc, SourceMaterialization::ConvertedF16, tensor.n_bytes())?;
        Ok(tensor)
    }

    fn load_moe_expert(&mut self, desc: &TensorDesc) -> Result<MetalTensor, MfError> {
        if matches!(desc.dtype, GgmlType::IQ3_XXS | GgmlType::IQ3_S)
            && moe_iq3_expert_native_enabled(desc)
        {
            self.load_direct(desc)
        } else {
            self.load_weight(desc)
        }
    }

    fn load_moe(&mut self, moe: &MoeFfn<'_>) -> Result<MetalMoeFfn, MfError> {
        Ok(MetalMoeFfn {
            gate_inp: self.load_router_weight(moe.gate_inp)?,
            gate_exps: self.load_moe_expert(moe.gate_exps)?,
            up_exps: self.load_moe_expert(moe.up_exps)?,
            down_exps: self.load_weight(moe.down_exps)?,
            gate_inp_shexp: self.load_f32(moe.gate_inp_shexp)?,
            gate_inp_cpu: crate::codec::dequant_to_f32(
                moe.gate_inp,
                self.gguf.slice(moe.gate_inp),
            )?,
            gate_inp_shexp_cpu: crate::codec::dequant_to_f32(
                moe.gate_inp_shexp,
                self.gguf.slice(moe.gate_inp_shexp),
            )?,
        })
    }

    fn record_derived(&mut self, tensor: Option<&MetalTensor>) -> Result<(), MfError> {
        if let Some(tensor) = tensor {
            self.ledger.record_derived(tensor)?;
        }
        Ok(())
    }

    fn finish(
        self,
        exact_sentinel: bool,
        expected: &[ModelWeightStorageRequest<'_>],
    ) -> Result<(), MfError> {
        let forced_exact_27b = self.is_forced_exact_27b();
        if let DirectStorage::ForcedPlanned(storage) = &self.direct_storage {
            storage.validate_complete(&self.ledger)?;
        }
        if let DirectStorage::ForcedOwned(storage) = &self.direct_storage {
            storage.validate_complete(&self.ledger)?;
        }
        if let DirectStorage::ForcedParallelCopied(storage) = &self.direct_storage {
            storage.validate_complete(&self.ledger)?;
        }
        let seen_forced = self.seen_forced.len();
        let ledger = self.ledger;
        validate_model_weight_request_sequence(&ledger.requests, expected)?;
        let accounted_source_bytes = ledger
            .direct_copy_bytes
            .checked_add(ledger.direct_view_bytes)
            .and_then(|bytes| bytes.checked_add(ledger.direct_alias_bytes))
            .and_then(|bytes| bytes.checked_add(ledger.tail_fallback_bytes))
            .and_then(|bytes| bytes.checked_add(ledger.converted_source_bytes))
            .ok_or_else(|| MfError::LoadPolicy("source accounting overflow".to_string()))?;
        if accounted_source_bytes != ledger.source_bytes {
            return Err(MfError::LoadPolicy(format!(
                "source accounting mismatch: categories={accounted_source_bytes} total={}",
                ledger.source_bytes
            )));
        }
        let expected_source_bytes = expected.iter().try_fold(0u64, |total, request| {
            total
                .checked_add(request.desc.n_bytes)
                .ok_or_else(|| MfError::LoadPolicy("expected source byte overflow".to_string()))
        })?;
        let expected_direct = expected
            .iter()
            .filter(|request| request.kind == ModelWeightStorageKind::Direct)
            .collect::<Vec<_>>();
        let expected_direct_bytes = expected_direct.iter().try_fold(0u64, |total, request| {
            total
                .checked_add(request.desc.n_bytes)
                .ok_or_else(|| MfError::LoadPolicy("expected direct byte overflow".to_string()))
        })?;
        let expected_converted = expected
            .iter()
            .filter(|request| request.kind != ModelWeightStorageKind::Direct)
            .collect::<Vec<_>>();
        let expected_converted_source_bytes =
            expected_converted.iter().try_fold(0u64, |total, request| {
                total.checked_add(request.desc.n_bytes).ok_or_else(|| {
                    MfError::LoadPolicy("expected converted source byte overflow".to_string())
                })
            })?;
        let expected_converted_resident_bytes =
            expected_converted.iter().try_fold(0u64, |total, request| {
                total.checked_add(request.resident_bytes).ok_or_else(|| {
                    MfError::LoadPolicy("expected converted resident byte overflow".to_string())
                })
            })?;
        let actual_direct_count = ledger
            .direct_copy_descriptors
            .checked_add(ledger.direct_view_descriptors)
            .and_then(|count| count.checked_add(ledger.direct_alias_descriptors))
            .and_then(|count| count.checked_add(ledger.tail_fallback_descriptors))
            .ok_or_else(|| MfError::LoadPolicy("direct descriptor overflow".to_string()))?;
        let actual_direct_bytes = ledger
            .direct_copy_bytes
            .checked_add(ledger.direct_view_bytes)
            .and_then(|bytes| bytes.checked_add(ledger.direct_alias_bytes))
            .and_then(|bytes| bytes.checked_add(ledger.tail_fallback_bytes))
            .ok_or_else(|| MfError::LoadPolicy("direct byte overflow".to_string()))?;
        if ledger.source_descriptors != expected.len()
            || ledger.source_bytes != expected_source_bytes
            || actual_direct_count != expected_direct.len()
            || actual_direct_bytes != expected_direct_bytes
            || ledger.converted_descriptors != expected_converted.len()
            || ledger.converted_source_bytes != expected_converted_source_bytes
            || ledger.converted_resident_bytes != expected_converted_resident_bytes
        {
            return Err(MfError::LoadPolicy(format!(
                concat!(
                    "model storage request drift: source={}/{} bytes={}/{} ",
                    "direct={}/{} bytes={}/{} converted={}/{} source_bytes={}/{} ",
                    "resident_bytes={}/{}"
                ),
                ledger.source_descriptors,
                expected.len(),
                ledger.source_bytes,
                expected_source_bytes,
                actual_direct_count,
                expected_direct.len(),
                actual_direct_bytes,
                expected_direct_bytes,
                ledger.converted_descriptors,
                expected_converted.len(),
                ledger.converted_source_bytes,
                expected_converted_source_bytes,
                ledger.converted_resident_bytes,
                expected_converted_resident_bytes,
            )));
        }
        if exact_sentinel
            && (ledger.source_descriptors != GGUF_NO_COPY_27B_DESCRIPTOR_COUNT
                || ledger.source_bytes != GGUF_NO_COPY_27B_SOURCE_BYTES)
        {
            return Err(MfError::LoadPolicy(format!(
                "exact 27B source ledger mismatch: descriptors={} bytes={}",
                ledger.source_descriptors, ledger.source_bytes
            )));
        }
        if forced_exact_27b
            && (seen_forced != GGUF_NO_COPY_27B_DESCRIPTOR_COUNT
                || ledger.direct_view_descriptors != GGUF_NO_COPY_27B_VIEW_COUNT
                || ledger.direct_view_bytes != GGUF_NO_COPY_27B_VIEW_BYTES
                || ledger.direct_alias_descriptors != 0
                || ledger.direct_alias_bytes != 0
                || ledger.tail_fallback_descriptors != 1
                || ledger.tail_fallback_bytes != GGUF_NO_COPY_27B_TAIL_BYTES
                || ledger.direct_copy_descriptors != 0
                || ledger.converted_descriptors != 0
                || ledger.derived_allocations != 0)
        {
            return Err(MfError::LoadPolicy(format!(
                concat!(
                    "forced no-copy ledger mismatch: seen={} view={}/{} alias={}/{} ",
                    "tail={}/{} copy={}/{} converted={}/{}/{} derived={}/{}"
                ),
                seen_forced,
                ledger.direct_view_descriptors,
                ledger.direct_view_bytes,
                ledger.direct_alias_descriptors,
                ledger.direct_alias_bytes,
                ledger.tail_fallback_descriptors,
                ledger.tail_fallback_bytes,
                ledger.direct_copy_descriptors,
                ledger.direct_copy_bytes,
                ledger.converted_descriptors,
                ledger.converted_source_bytes,
                ledger.converted_resident_bytes,
                ledger.derived_allocations,
                ledger.derived_bytes,
            )));
        }
        emit_metal_load_line(format_args!(
            concat!(
                "[metal-load-ledger] source={}/{} direct_copy={}/{} direct_view={}/{} ",
                "direct_alias={}/{} tail_fallback={}/{} converted={}/{}/{} derived={}/{}"
            ),
            ledger.source_descriptors,
            ledger.source_bytes,
            ledger.direct_copy_descriptors,
            ledger.direct_copy_bytes,
            ledger.direct_view_descriptors,
            ledger.direct_view_bytes,
            ledger.direct_alias_descriptors,
            ledger.direct_alias_bytes,
            ledger.tail_fallback_descriptors,
            ledger.tail_fallback_bytes,
            ledger.converted_descriptors,
            ledger.converted_source_bytes,
            ledger.converted_resident_bytes,
            ledger.derived_allocations,
            ledger.derived_bytes,
        ));
        Ok(())
    }
}

fn matches_no_copy_27b_sentinel(gguf: &GgufFile, model: &Model<'_>) -> bool {
    gguf.shard_count() == 1
        && gguf.total_mapped_len() == GGUF_NO_COPY_27B_MAPPED_BYTES
        && gguf.tensors.len() == GGUF_NO_COPY_27B_DESCRIPTOR_COUNT
        && gguf_descriptor_layout_digest(gguf) == GGUF_NO_COPY_27B_LAYOUT_DIGEST
        && !model.tied_embeddings
        && model.mtp.is_none()
        && native_quant_embedding_default_promoted(
            &model.arch,
            model.tied_embeddings,
            model.mtp.is_some(),
            model.token_embd.dtype,
            &model.token_embd.shape,
        )
}

fn matches_owned_a3b_arch(model: &Model<'_>) -> bool {
    let arch = model.arch;
    arch.kind == ArchKind::Moe
        && arch.n_layer == 40
        && arch.hidden_size == 2048
        && arch.intermediate_size == 0
        && arch.vocab_size == 248_320
        && arch.full_attention_interval == 4
        && arch.n_q_heads == 16
        && arch.n_kv_heads == 2
        && arch.attn_head_dim == 256
        && arch.rope_theta == 10_000_000.0
        && arch.partial_rotary_factor == 0.25
        && arch.gdn_n_v_heads == 32
        && arch.gdn_n_k_heads == 16
        && arch.gdn_head_dim == 128
        && arch.gdn_conv_kernel == 4
        && arch.expert_count == 256
        && arch.expert_used_count == 8
        && arch.expert_feed_forward_length == 512
        && arch.expert_shared_feed_forward_length == 512
        && arch.mtp_n_hidden_layers == 0
}

fn select_unique_parallel_copy_profile<F>(
    profiles: &[&'static ParallelCopyProfile],
    mut matches: F,
) -> Result<&'static ParallelCopyProfile, MfError>
where
    F: FnMut(&ParallelCopyProfile) -> Result<bool, MfError>,
{
    let mut ids = HashSet::with_capacity(profiles.len());
    for profile in profiles {
        if !ids.insert(profile.id) {
            return Err(MfError::LoadPolicy(format!(
                "duplicate parallel-copy profile id {}",
                profile.id.label()
            )));
        }
    }
    let mut selected = None;
    for profile in profiles {
        if !matches(profile)? {
            continue;
        }
        if selected.is_some() {
            return Err(MfError::LoadPolicy(
                "parallel-copy profile match is ambiguous".to_string(),
            ));
        }
        selected = Some(*profile);
    }
    selected.ok_or_else(|| {
        MfError::LoadPolicy("forced parallel copy rejects unsupported geometry".to_string())
    })
}

fn parallel_copy_profile_matches(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
    profile: &ParallelCopyProfile,
) -> Result<bool, MfError> {
    if !ctx.device.hasUnifiedMemory() {
        return Ok(false);
    }
    if let ParallelCopyDeviceConstraint::ExactUnified(name) = profile.device_constraint
        && ctx.device.name().to_string() != name
    {
        return Ok(false);
    }
    let architecture = gguf.architecture();
    if profile
        .architecture_label
        .is_some_and(|expected| architecture.as_deref() != Some(expected))
    {
        return Ok(false);
    }
    let shard_lengths = gguf.shard_mapped_lengths();
    let embedding_qualified = match profile.id {
        ParallelCopyProfileId::A10bQ4xlV1 => {
            embedding_selection == NativeQuantEmbeddingSelection::Forced
        }
        ParallelCopyProfileId::A3bQ4kmV1 | ParallelCopyProfileId::Dense27bQ4kmV1 => {
            embedding_selection == NativeQuantEmbeddingSelection::AutoPromoted
                && native_quant_embedding_default_promoted(
                    &model.arch,
                    model.tied_embeddings,
                    model.mtp.is_some(),
                    model.token_embd.dtype,
                    &model.token_embd.shape,
                )
        }
    };
    if shard_lengths.as_slice() != profile.shard_mapped_lengths
        || gguf_descriptor_layout_digest(gguf) != profile.descriptor_layout_digest
        || model.arch != profile.arch
        || model.tied_embeddings != profile.tied_embeddings
        || model.mtp.is_some() != profile.mtp_present
        || model.token_embd.dtype != profile.embedding_dtype
        || model.token_embd.shape.as_slice() != profile.embedding_shape
        || !embedding_qualified
        || host_page_size_bytes()? != 16_384
        || ctx.max_buffer_length() != 77_309_411_328
        || expected.len() != profile.request_count
        || model_weight_storage_inventory_digest(expected) != profile.inventory_digest
    {
        return Ok(false);
    }
    let mut source_bytes = 0u64;
    for request in expected {
        let Some(shard_len) = profile.shard_mapped_lengths.get(request.desc.shard_idx) else {
            return Ok(false);
        };
        let Some(source_end) = request.desc.data_offset.checked_add(request.desc.n_bytes) else {
            return Ok(false);
        };
        if request.kind != ModelWeightStorageKind::Direct
            || request.desc.n_bytes == 0
            || request.resident_bytes != request.desc.n_bytes
            || source_end > *shard_len as u64
            || request.desc.n_bytes > ctx.max_buffer_length() as u64
            || usize::try_from(request.desc.n_bytes).is_err()
        {
            return Ok(false);
        }
        source_bytes = source_bytes
            .checked_add(request.desc.n_bytes)
            .ok_or_else(|| {
                MfError::LoadPolicy("parallel-copy profile source byte overflow".to_string())
            })?;
    }
    Ok(source_bytes == profile.source_bytes)
}

fn select_parallel_copy_profile(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
) -> Result<&'static ParallelCopyProfile, MfError> {
    select_unique_parallel_copy_profile(&PARALLEL_COPY_PROFILES, |profile| {
        parallel_copy_profile_matches(ctx, gguf, model, expected, embedding_selection, profile)
    })
}

fn host_physical_memory_bytes() -> Option<u64> {
    let mut bytes = 0u64;
    let mut size = std::mem::size_of::<u64>();
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (result == 0 && size == std::mem::size_of::<u64>()).then_some(bytes)
}

fn a3b_parallel_copy_auto_host_supported(
    unified_memory: bool,
    device_name: &str,
    physical_memory_bytes: Option<u64>,
) -> bool {
    unified_memory
        && device_name == A3B_PARALLEL_COPY_AUTO_DEVICE
        && physical_memory_bytes.is_some_and(|bytes| bytes >= A3B_PARALLEL_COPY_AUTO_MIN_MEMORY)
}

fn select_auto_parallel_copy_profile(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
) -> Result<Option<&'static ParallelCopyProfile>, MfError> {
    if !a3b_parallel_copy_auto_host_supported(
        ctx.device.hasUnifiedMemory(),
        &ctx.device.name().to_string(),
        host_physical_memory_bytes(),
    ) {
        return Ok(None);
    }
    parallel_copy_profile_matches(
        ctx,
        gguf,
        model,
        expected,
        embedding_selection,
        &A3B_PARALLEL_COPY_PROFILE,
    )
    .map(|matched| matched.then_some(&A3B_PARALLEL_COPY_PROFILE))
}

fn auto_parallel_copy_population(
    profile: ParallelCopyProfileId,
) -> Option<ParallelPopulationMethod> {
    match profile {
        ParallelCopyProfileId::A3bQ4kmV1 => Some(ParallelPopulationMethod::Pread),
        ParallelCopyProfileId::A10bQ4xlV1 => None,
        ParallelCopyProfileId::Dense27bQ4kmV1 => None,
    }
}

fn forced_a10b_parallel_pread_embedding_qualified(
    parallel_mode: GgufParallelCopyMode,
    gguf: &GgufFile,
    model: &Model<'_>,
) -> bool {
    if parallel_mode != GgufParallelCopyMode::ForcedPread {
        return false;
    }
    let profile = &A10B_PARALLEL_PREAD_PROFILE;
    let architecture = gguf.architecture();
    architecture.as_deref() == profile.architecture_label
        && gguf.shard_mapped_lengths().as_slice() == profile.shard_mapped_lengths
        && gguf_descriptor_layout_digest(gguf) == profile.descriptor_layout_digest
        && model.arch == profile.arch
        && model.tied_embeddings == profile.tied_embeddings
        && model.mtp.is_some() == profile.mtp_present
        && model.token_embd.dtype == profile.embedding_dtype
        && model.token_embd.shape.as_slice() == profile.embedding_shape
}

fn authenticated_a3b_storage_plan(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
) -> Result<RetainedStoragePlan, MfError> {
    if !ctx.device.hasUnifiedMemory() {
        return Err(MfError::LoadPolicy(
            "forced A3B storage requires unified memory".to_string(),
        ));
    }
    let expected_source_bytes = expected.iter().try_fold(0u64, |total, request| {
        total
            .checked_add(request.desc.n_bytes)
            .ok_or_else(|| MfError::LoadPolicy("A3B source byte overflow".to_string()))
    })?;
    if gguf.shard_count() != 1
        || gguf.total_mapped_len() != GGUF_OWNED_A3B_MAPPED_BYTES
        || gguf_descriptor_layout_digest(gguf) != GGUF_OWNED_A3B_LAYOUT_DIGEST
        || !matches_owned_a3b_arch(model)
        || model.tied_embeddings
        || model.mtp.is_some()
        || embedding_selection != NativeQuantEmbeddingSelection::AutoPromoted
        || expected.len() != GGUF_OWNED_A3B_REQUESTS
        || expected_source_bytes != GGUF_OWNED_A3B_SOURCE_BYTES
        || expected
            .iter()
            .any(|request| request.kind != ModelWeightStorageKind::Direct)
        || model_weight_storage_inventory_digest(expected) != GGUF_OWNED_A3B_INVENTORY_DIGEST
    {
        return Err(MfError::LoadPolicy(
            "forced A3B storage rejects non-sentinel layout".to_string(),
        ));
    }

    let direct = expected
        .iter()
        .map(|request| request.desc)
        .collect::<Vec<_>>();
    let page_size = host_page_size_bytes()?;
    let plan = plan_retained_storage(
        &gguf.shard_mapped_lengths(),
        &direct,
        page_size,
        ctx.max_buffer_length(),
        GGUF_NO_COPY_ALIGNMENT,
    )?;
    let window_bytes = plan.windows.iter().try_fold(0u64, |total, window| {
        total
            .checked_add(window.length as u64)
            .ok_or_else(|| MfError::LoadPolicy("A3B window byte overflow".to_string()))
    })?;
    let view_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
        .count();
    let alias_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::Alias { .. }))
        .count();
    let fallback_entries = plan
        .entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.disposition,
                RetainedStorageDisposition::CopyFallback { .. }
            )
        })
        .collect::<Vec<_>>();
    if page_size != 16_384
        || ctx.max_buffer_length() != 77_309_411_328
        || retained_storage_plan_digest(&plan) != GGUF_OWNED_A3B_PLAN_DIGEST
        || plan.windows.len() != 1
        || window_bytes != GGUF_OWNED_A3B_WINDOW_BYTES
        || view_count != GGUF_OWNED_A3B_VIEWS
        || plan.unique_view_bytes != GGUF_OWNED_A3B_VIEW_BYTES
        || plan.logical_view_bytes != GGUF_OWNED_A3B_VIEW_BYTES
        || alias_count != 0
        || plan.alias_bytes != 0
        || fallback_entries.len() != 1
        || plan.unique_fallback_bytes != GGUF_OWNED_A3B_FALLBACK_BYTES
        || window_bytes - plan.unique_view_bytes != GGUF_OWNED_A3B_GAP_BYTES
        || !matches!(
            fallback_entries[0].disposition,
            RetainedStorageDisposition::CopyFallback {
                reason: RetainedStorageFallback::FinalPartialPage
            }
        )
    {
        return Err(MfError::LoadPolicy(
            "forced A3B storage planner geometry drifted".to_string(),
        ));
    }
    Ok(plan)
}

fn planned_owned_storage_for_load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
) -> Result<PlannedOwnedStorage, MfError> {
    let plan = authenticated_a3b_storage_plan(ctx, gguf, model, expected, embedding_selection)?;
    let direct = expected
        .iter()
        .map(|request| request.desc)
        .collect::<Vec<_>>();
    let fallback_entries = plan
        .entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.disposition,
                RetainedStorageDisposition::CopyFallback { .. }
            )
        })
        .collect::<Vec<_>>();
    let window_bytes = plan.windows[0].length as u64;

    let ready_started = std::time::Instant::now();
    let allocation_started = std::time::Instant::now();
    let resources = vec![
        ctx.buffer_uninit(plan.windows[0].length)?,
        ctx.buffer_uninit(GGUF_OWNED_A3B_FALLBACK_BYTES as usize)?,
    ];
    let allocation_ms = allocation_started.elapsed().as_secs_f64() * 1e3;
    let copy_started = std::time::Instant::now();
    let window = &plan.windows[0];
    let window_source = gguf
        .try_shard_range(window.shard_idx, window.mmap_offset, window.length)
        .map_err(|error| MfError::LoadPolicy(format!("owned window source: {error}")))?;
    copy_owned_arena_four_workers(window_source, &resources[0], plan.page_size)?;
    let fallback = fallback_entries[0];
    let fallback_desc = direct.get(fallback.request_index).ok_or_else(|| {
        MfError::LoadPolicy("owned fallback request index is out of bounds".to_string())
    })?;
    copy_owned_arena_serial(gguf.slice(fallback_desc), &resources[1])?;
    let copy_ms = copy_started.elapsed().as_secs_f64() * 1e3;
    let ready_ms = ready_started.elapsed().as_secs_f64() * 1e3;
    let physical_bytes = resources.iter().try_fold(0u64, |total, buffer| {
        total
            .checked_add(buffer.length() as u64)
            .ok_or_else(|| MfError::LoadPolicy("owned physical byte overflow".to_string()))
    })?;
    if resources.len() != 2
        || physical_bytes != GGUF_OWNED_A3B_PHYSICAL_BYTES
        || resources
            .iter()
            .any(|buffer| buffer.storageMode() != MTLStorageMode::Shared)
    {
        return Err(MfError::LoadPolicy(
            "forced owned arena physical realization drifted".to_string(),
        ));
    }
    eprintln!(
        concat!(
            "[metal-gguf-owned] windows=1 window_bytes={} gaps={} fallback=1/{} ",
            "resources=2/{} workers={} page={} alignment={} allocation_ms={:.3} ",
            "copy_ms={:.3} ready_ms={:.3}"
        ),
        window_bytes,
        GGUF_OWNED_A3B_GAP_BYTES,
        GGUF_OWNED_A3B_FALLBACK_BYTES,
        physical_bytes,
        GGUF_OWNED_WORKERS,
        plan.page_size,
        plan.required_alignment,
        allocation_ms,
        copy_ms,
        ready_ms,
    );
    let mut fallback_resources = HashMap::new();
    fallback_resources.insert(fallback.request_index, 1);
    Ok(PlannedOwnedStorage {
        realized: vec![None; plan.entries.len()],
        plan,
        resources,
        fallback_resources,
        cursor: 0,
    })
}

fn planned_parallel_copied_storage_for_load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
) -> Result<PlannedParallelCopiedStorage, MfError> {
    let profile = select_parallel_copy_profile(ctx, gguf, model, expected, embedding_selection)?;
    let prepared = prepare_parallel_copied_profile(
        ctx,
        gguf,
        model,
        expected,
        embedding_selection,
        profile,
        population,
        destination_length,
    )?;
    realize_parallel_copied_profile(ctx, gguf, expected, prepared)
}

fn validate_a10b_parallel_memory_admission(
    ctx: &MetalContext,
    profile: &ParallelCopyProfile,
    phase: &str,
) -> Result<(), MfError> {
    if profile.id != ParallelCopyProfileId::A10bQ4xlV1 {
        return Ok(());
    }
    let admission = evaluate_metal_memory_admission(
        A10B_PARALLEL_PREAD_REQUIRED_HEADROOM_BYTES,
        0,
        ctx.memory_signals(),
        true,
    );
    emit_metal_load_line(format_args!(
        concat!(
            "[metal-gguf-parallel-memory] profile={} phase={} admitted={} reason={} ",
            "required={} recommended={} current={} process_remaining={:?}"
        ),
        profile.id.label(),
        phase,
        admission.admitted,
        admission.reason.as_str(),
        A10B_PARALLEL_PREAD_REQUIRED_HEADROOM_BYTES,
        admission.signals.recommended_max_bytes,
        admission.signals.current_allocated_bytes,
        admission.signals.process_limit_remaining_bytes,
    ));
    if !admission.admitted {
        return Err(MfError::LoadPolicy(format!(
            "A10B parallel pread memory admission failed during {phase}: {}",
            admission.reason.as_str()
        )));
    }
    Ok(())
}

fn prepare_parallel_copied_profile(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
    profile: &'static ParallelCopyProfile,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
) -> Result<PreparedParallelCopiedProfile, MfError> {
    validate_parallel_destination_length(profile, population, destination_length)?;
    validate_a10b_parallel_memory_admission(ctx, profile, "prepare")?;
    let proof = match profile.authentication {
        ParallelCopyAuthentication::A3bRetainedPlan => {
            authenticated_a3b_storage_plan(ctx, gguf, model, expected, embedding_selection)?;
            PreparedParallelCopyProof::A3bRetainedPlan
        }
        ParallelCopyAuthentication::A10bPlannerFree => PreparedParallelCopyProof::A10bPlannerFree,
        ParallelCopyAuthentication::DensePlannerFree => PreparedParallelCopyProof::DensePlannerFree,
    };
    let expected_identities = expected
        .iter()
        .map(expected_model_weight_identity)
        .collect::<Vec<_>>();
    let sorted_request_indices = frozen_parallel_copy_order(profile, &expected_identities)?;
    Ok(PreparedParallelCopiedProfile {
        profile,
        population,
        destination_length,
        expected_identities,
        sorted_request_indices,
        _proof: proof,
    })
}

fn realize_parallel_copied_profile(
    ctx: &MetalContext,
    gguf: &GgufFile,
    expected: &[ModelWeightStorageRequest<'_>],
    prepared: PreparedParallelCopiedProfile,
) -> Result<PlannedParallelCopiedStorage, MfError> {
    let PreparedParallelCopiedProfile {
        profile,
        population,
        destination_length,
        expected_identities,
        sorted_request_indices,
        _proof,
    } = prepared;
    validate_model_weight_request_sequence(&expected_identities, expected)?;
    validate_a10b_parallel_memory_admission(ctx, profile, "realize")?;
    let source_stamps = if profile.id == ParallelCopyProfileId::A10bQ4xlV1 {
        Some(gguf.revalidate_retained_shard_stamps().map_err(|error| {
            MfError::LoadPolicy(format!("A10B pread source preflight failed: {error}"))
        })?)
    } else {
        None
    };
    let usage_before = match profile.marker_contract {
        ParallelCopyMarkerContract::A3b => None,
        ParallelCopyMarkerContract::A10bSchema2 | ParallelCopyMarkerContract::DenseSchema2 => {
            Some(capture_parallel_copy_usage()?)
        }
    };
    let proc_before = match profile.marker_contract {
        ParallelCopyMarkerContract::A3b => None,
        ParallelCopyMarkerContract::A10bSchema2 | ParallelCopyMarkerContract::DenseSchema2 => {
            Some(capture_parallel_copy_proc_usage()?)
        }
    };
    let ready_started = std::time::Instant::now();
    let resources_result = expected
        .iter()
        .map(|request| {
            let length = parallel_destination_resource_length(
                request.desc.n_bytes,
                destination_length,
                ctx.max_buffer_length(),
            )?;
            ctx.buffer_uninit(length).map_err(MfError::from)
        })
        .collect::<Result<Vec<_>, _>>();
    let mut resources = resources_result?;
    let allocation_finished = std::time::Instant::now();

    let sources = match population {
        ParallelPopulationMethod::MmapCopy => Some(
            expected
                .iter()
                .enumerate()
                .map(|(index, request)| {
                    gguf.try_slice(request.desc).map_err(|error| {
                        MfError::LoadPolicy(format!(
                            "parallel-copy source resolution failed at {index}: {error}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        ParallelPopulationMethod::Pread => None,
    };

    let mut resource_identities = HashSet::with_capacity(resources.len());
    let mut destination_ranges = Vec::with_capacity(resources.len());
    let mut destination_bytes = 0u64;
    for (index, (resource, request)) in resources.iter().zip(expected).enumerate() {
        let identity = Retained::as_ptr(resource) as *const () as usize;
        let start = resource.contents().as_ptr().cast::<u8>() as usize;
        let end = start.checked_add(resource.length()).ok_or_else(|| {
            MfError::LoadPolicy("parallel-copy destination range overflow".to_string())
        })?;
        destination_bytes = destination_bytes
            .checked_add(resource.length() as u64)
            .ok_or_else(|| {
                MfError::LoadPolicy("parallel-copy destination byte overflow".to_string())
            })?;
        let expected_resource_length = parallel_destination_resource_length(
            request.desc.n_bytes,
            destination_length,
            ctx.max_buffer_length(),
        )?;
        if !resource_identities.insert(identity)
            || resource.length() != expected_resource_length
            || resource.length() == 0
            || start == 0
            || resource.storageMode() != MTLStorageMode::Shared
            || resource.cpuCacheMode() != MTLCPUCacheMode::DefaultCache
            || resource.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
        {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy destination resource {index} drifted"
            )));
        }
        destination_ranges.push((start, end));
    }
    let (expected_destination_bytes, padded_resources) = parallel_destination_accounting(
        &expected_identities,
        destination_length,
        ctx.max_buffer_length(),
    )?;
    let destination_accounting_matches = match destination_length {
        ParallelDestinationLength::LogicalExact => {
            expected_destination_bytes == profile.source_bytes && padded_resources == 0
        }
        ParallelDestinationLength::PageRounded16K => {
            expected_destination_bytes == GGUF_PAGE_ROUNDED_A3B_ALLOCATED_BYTES
                && expected_destination_bytes.checked_sub(profile.source_bytes)
                    == Some(GGUF_PAGE_ROUNDED_A3B_PADDING_BYTES)
                && padded_resources == GGUF_PAGE_ROUNDED_A3B_PADDED_RESOURCES
        }
    };
    if destination_bytes != expected_destination_bytes || !destination_accounting_matches {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy destination accounting drifted: actual={destination_bytes} expected={expected_destination_bytes} padded={padded_resources}"
        )));
    }
    let mut ranges_by_address = destination_ranges.clone();
    ranges_by_address.sort_unstable();
    if ranges_by_address
        .windows(2)
        .any(|pair| pair[0].1 > pair[1].0)
    {
        return Err(MfError::LoadPolicy(
            "parallel-copy destination resources overlap".to_string(),
        ));
    }
    if let Some(sources) = sources.as_ref() {
        for (index, source) in sources.iter().enumerate() {
            let source_start = source.as_ptr() as usize;
            let source_end = source_start.checked_add(source.len()).ok_or_else(|| {
                MfError::LoadPolicy("parallel-copy source range overflow".to_string())
            })?;
            if source.is_empty()
                || source_start == 0
                || source.len() as u64 != expected[index].desc.n_bytes
                || source.len() > resources[index].length()
                || destination_ranges
                    .iter()
                    .any(|&(start, end)| source_start < end && start < source_end)
            {
                return Err(MfError::LoadPolicy(format!(
                    "parallel-copy source range {index} is invalid or overlaps a destination"
                )));
            }
        }
    }

    let mut tasks_by_request = resources
        .iter_mut()
        .zip(expected)
        .enumerate()
        .map(|(request_index, (resource, request))| {
            // SAFETY: every prerequisite in exclusive_buffer_bytes_mut's
            // contract was established for the complete resource set above.
            let logical_length = usize::try_from(request.desc.n_bytes).map_err(|_| {
                MfError::LoadPolicy(
                    "parallel-copy logical destination length does not fit usize".to_string(),
                )
            })?;
            let destination = unsafe { exclusive_buffer_bytes_mut(resource, logical_length) };
            Ok(Some(match population {
                ParallelPopulationMethod::MmapCopy => {
                    let source = *sources
                        .as_ref()
                        .expect("mmap-copy population must resolve sources")
                        .get(request_index)
                        .expect("source inventory matches destination inventory");
                    ParallelPopulationTask::MmapCopy(ParallelCopyTask {
                        source,
                        destination,
                    })
                }
                ParallelPopulationMethod::Pread => {
                    ParallelPopulationTask::Pread(ParallelPreadTask {
                        shard_idx: request.desc.shard_idx,
                        source_offset: request.desc.data_offset,
                        destination,
                    })
                }
            }))
        })
        .collect::<Result<Vec<_>, MfError>>()?;
    let mut tasks = Vec::with_capacity(tasks_by_request.len());
    for &request_index in &sorted_request_indices {
        tasks.push(tasks_by_request[request_index].take().ok_or_else(|| {
            MfError::LoadPolicy(format!(
                "parallel-copy request {request_index} was assigned more than once"
            ))
        })?);
    }
    if tasks_by_request.iter().any(Option::is_some) {
        return Err(MfError::LoadPolicy(
            "parallel-copy task union is incomplete".to_string(),
        ));
    }
    let source_finished = std::time::Instant::now();

    let copy_result = std::thread::scope(|scope| {
        let mut task_tail = tasks.as_mut_slice();
        let mut handles = Vec::with_capacity(GGUF_OWNED_WORKERS);
        let mut spawn_error = None;
        let mut partition_error = None;
        let mut assigned_tasks = 0usize;
        for worker in 0..GGUF_OWNED_WORKERS {
            let count = profile.task_counts[worker];
            if count == 0 || count > task_tail.len() {
                partition_error = Some(worker);
                break;
            }
            let (worker_tasks, remaining) = task_tail.split_at_mut(count);
            match std::thread::Builder::new().spawn_scoped(scope, move || {
                for task in worker_tasks {
                    match task {
                        ParallelPopulationTask::MmapCopy(task) => {
                            task.destination.copy_from_slice(task.source);
                        }
                        ParallelPopulationTask::Pread(task) => {
                            gguf.read_shard_exact_at(
                                task.shard_idx,
                                task.source_offset,
                                task.destination,
                            )
                            .map_err(|error| {
                                MfError::LoadPolicy(format!(
                                    "parallel-pread worker {worker} read failed: {error}"
                                ))
                            })?;
                        }
                    }
                }
                Ok::<(), MfError>(())
            }) {
                Ok(handle) => handles.push((worker, handle)),
                Err(error) => {
                    spawn_error = Some((worker, error));
                    break;
                }
            }
            task_tail = remaining;
            assigned_tasks += count;
        }

        let mut worker_error = None;
        let mut panicked_worker = None;
        for (worker, handle) in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) if worker_error.is_none() => {
                    worker_error = Some((worker, error));
                }
                Ok(Err(_)) => {}
                Err(_) if panicked_worker.is_none() => panicked_worker = Some(worker),
                Err(_) => {}
            }
        }
        if let Some((worker, error)) = spawn_error {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy worker {worker} spawn failed: {error}"
            )));
        }
        if let Some(worker) = partition_error {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy worker {worker} partition is invalid"
            )));
        }
        if let Some((worker, error)) = worker_error {
            return Err(MfError::LoadPolicy(format!(
                "parallel population worker {worker} failed: {error}"
            )));
        }
        if let Some(worker) = panicked_worker {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy worker {worker} panicked"
            )));
        }
        if assigned_tasks != profile.request_count {
            return Err(MfError::LoadPolicy(
                "parallel-copy workers did not consume every task".to_string(),
            ));
        }
        Ok(())
    });
    copy_result?;
    if let Some(before) = source_stamps {
        let after = gguf.revalidate_retained_shard_stamps().map_err(|error| {
            MfError::LoadPolicy(format!("A10B pread source postflight failed: {error}"))
        })?;
        if after != before {
            return Err(MfError::LoadPolicy(
                "A10B retained source stamps changed during population".to_string(),
            ));
        }
    }
    let copy_finished = std::time::Instant::now();
    drop(tasks);
    drop(tasks_by_request);
    drop(sources);

    let tensors = resources
        .iter()
        .zip(expected)
        .map(|(resource, request)| {
            MetalTensor::owned_weight_view(
                resource.clone(),
                0,
                request.desc.shape.clone(),
                request.desc.dtype,
                GGUF_NO_COPY_ALIGNMENT,
            )
            .map_err(MfError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_parallel_copied_topology(
        profile,
        destination_length,
        &expected_identities,
        &resources,
        &tensors,
    )?;
    let binding_finished = std::time::Instant::now();
    let usage_after = match profile.marker_contract {
        ParallelCopyMarkerContract::A3b => None,
        ParallelCopyMarkerContract::A10bSchema2 | ParallelCopyMarkerContract::DenseSchema2 => {
            Some(capture_parallel_copy_usage()?)
        }
    };
    let proc_after = match profile.marker_contract {
        ParallelCopyMarkerContract::A3b => None,
        ParallelCopyMarkerContract::A10bSchema2 | ParallelCopyMarkerContract::DenseSchema2 => {
            Some(capture_parallel_copy_proc_usage()?)
        }
    };

    let allocation_us = allocation_finished
        .duration_since(ready_started)
        .as_micros() as u64;
    let source_us = source_finished
        .duration_since(allocation_finished)
        .as_micros() as u64;
    let copy_us = copy_finished.duration_since(source_finished).as_micros() as u64;
    let binding_us = binding_finished.duration_since(copy_finished).as_micros() as u64;
    let ready_us = binding_finished.duration_since(ready_started).as_micros() as u64;
    let phase_us = allocation_us
        .checked_add(source_us)
        .and_then(|total| total.checked_add(copy_us))
        .and_then(|total| total.checked_add(binding_us))
        .ok_or_else(|| MfError::LoadPolicy("parallel-copy phase time overflow".to_string()))?;
    if ready_us == 0 || ready_us.abs_diff(phase_us) > 4 {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy timing reconciliation drifted: ready={ready_us} phases={phase_us}"
        )));
    }

    let accounting = match (usage_before, usage_after, proc_before, proc_after) {
        (None, None, None, None) => None,
        (Some(before), Some(after), Some(proc_before), Some(proc_after)) => Some(
            finish_parallel_copy_accounting(before, after, proc_before, proc_after)?,
        ),
        _ => {
            return Err(MfError::LoadPolicy(
                "parallel-copy endpoint accounting capture is incomplete".to_string(),
            ));
        }
    };
    emit_parallel_copy_marker(
        profile,
        population,
        destination_length,
        ParallelCopyTiming {
            allocation_us,
            source_us,
            copy_us,
            binding_us,
            ready_us,
        },
        accounting,
    )?;

    Ok(PlannedParallelCopiedStorage {
        profile,
        destination_length,
        expected: expected_identities,
        sorted_request_indices,
        resources,
        tensors,
        cursor: 0,
    })
}

fn planned_retained_storage_for_load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    expected: &[ModelWeightStorageRequest<'_>],
    prefault_enabled: bool,
) -> Result<PlannedRetainedStorage, MfError> {
    let direct = expected
        .iter()
        .filter(|request| request.kind == ModelWeightStorageKind::Direct)
        .map(|request| request.desc)
        .collect::<Vec<_>>();
    if direct.is_empty() {
        return Err(MfError::LoadPolicy(
            "forced retained storage requires at least one direct tensor".to_string(),
        ));
    }
    let page_size = host_page_size_bytes()?;
    let max_buffer_length = ctx.device.maxBufferLength();
    let plan = plan_retained_storage(
        &gguf.shard_mapped_lengths(),
        &direct,
        page_size,
        max_buffer_length,
        GGUF_NO_COPY_ALIGNMENT,
    )?;
    for entry in &plan.entries {
        if let RetainedStorageDisposition::CopyFallback { reason } = entry.disposition
            && reason != RetainedStorageFallback::FinalPartialPage
        {
            return Err(MfError::LoadPolicy(format!(
                "forced retained storage rejects {:?} fallback for {:?}",
                reason, entry.name
            )));
        }
    }

    let mut windows = Vec::with_capacity(plan.windows.len());
    let mut window_bytes = 0u64;
    let mut prefault_pages = 0usize;
    let mut prefault_bytes = 0usize;
    let mut prefault_ms = 0.0;
    let mut prefault_checksum = 0u64;
    for window in &plan.windows {
        let mmap = gguf.retained_shard_mmap(window.shard_idx).ok_or_else(|| {
            MfError::LoadPolicy(format!(
                "retained storage window references missing shard {}",
                window.shard_idx
            ))
        })?;
        let mmap_offset = usize::try_from(window.mmap_offset).map_err(|_| {
            MfError::LoadPolicy(format!(
                "retained storage window offset {} does not fit usize",
                window.mmap_offset
            ))
        })?;
        let backing = ctx.gguf_no_copy_window(
            mmap,
            window.shard_idx,
            mmap_offset,
            window.length,
            GGUF_NO_COPY_ALIGNMENT,
        )?;
        if backing.mmap_offset() != mmap_offset
            || backing.exposed_len() != window.length
            || backing.required_alignment() != GGUF_NO_COPY_ALIGNMENT
        {
            return Err(MfError::LoadPolicy(format!(
                "retained storage window realization drift at shard {} offset {}",
                window.shard_idx, window.mmap_offset
            )));
        }
        window_bytes = window_bytes
            .checked_add(window.length as u64)
            .ok_or_else(|| MfError::LoadPolicy("retained window bytes overflow".to_string()))?;
        if prefault_enabled {
            let report = backing.prefault_read();
            let expected_pages = backing.exposed_len() / backing.page_size();
            if report.page_count != expected_pages || report.covered_bytes != backing.exposed_len()
            {
                return Err(MfError::LoadPolicy(format!(
                    "retained prefault mismatch: pages={}/{} covered={}/{}",
                    report.page_count,
                    expected_pages,
                    report.covered_bytes,
                    backing.exposed_len(),
                )));
            }
            prefault_pages = prefault_pages
                .checked_add(report.page_count)
                .ok_or_else(|| {
                    MfError::LoadPolicy("retained prefault page count overflow".to_string())
                })?;
            prefault_bytes = prefault_bytes
                .checked_add(report.covered_bytes)
                .ok_or_else(|| {
                    MfError::LoadPolicy("retained prefault byte count overflow".to_string())
                })?;
            prefault_ms += report.wall_ms;
            prefault_checksum = prefault_checksum.rotate_left(7) ^ report.checksum;
        }
        windows.push(backing);
    }
    let view_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
        .count();
    let alias_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::Alias { .. }))
        .count();
    let fallback_count = plan
        .entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.disposition,
                RetainedStorageDisposition::CopyFallback { .. }
            )
        })
        .count();
    eprintln!(
        concat!(
            "[metal-gguf-retained] windows={} window_bytes={} direct={} view={}/{} ",
            "alias={}/{} fallback={}/{} page={} max_buffer={} alignment={} ",
            "prefault={} prefault_pages={} prefault_bytes={} prefault_ms={:.3} ",
            "checksum={:#018x}"
        ),
        plan.windows.len(),
        window_bytes,
        plan.entries.len(),
        view_count,
        plan.unique_view_bytes,
        alias_count,
        plan.alias_bytes,
        fallback_count,
        plan.unique_fallback_bytes,
        plan.page_size,
        plan.max_buffer_length,
        plan.required_alignment,
        if prefault_enabled {
            "enabled"
        } else {
            "disabled"
        },
        prefault_pages,
        prefault_bytes,
        prefault_ms,
        prefault_checksum,
    );
    Ok(PlannedRetainedStorage {
        realized: vec![None; plan.entries.len()],
        plan,
        windows,
        cursor: 0,
    })
}

fn direct_storage_for_load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    mode: GgufNoCopyMode,
    prefault_enabled: bool,
    owned_mode: GgufOwnedArenaMode,
    parallel_mode: GgufParallelCopyMode,
    prepared_auto: PreparedAutoSelection,
    embedding_selection: NativeQuantEmbeddingSelection,
) -> Result<(DirectStorage, bool), MfError> {
    let exact_sentinel = matches_no_copy_27b_sentinel(gguf, model);
    if let Some((population, destination_length)) = parallel_mode.forced_configuration() {
        let storage = planned_parallel_copied_storage_for_load(
            ctx,
            gguf,
            model,
            expected,
            embedding_selection,
            population,
            destination_length,
        )?;
        return Ok((DirectStorage::ForcedParallelCopied(storage), false));
    }
    if owned_mode == GgufOwnedArenaMode::Forced {
        if mode != GgufNoCopyMode::Disabled {
            return Err(MfError::LoadPolicy(
                "owned arena and retained no-copy are mutually exclusive".to_string(),
            ));
        }
        let storage =
            planned_owned_storage_for_load(ctx, gguf, model, expected, embedding_selection)?;
        return Ok((DirectStorage::ForcedOwned(storage), false));
    }
    if let PreparedAutoSelection::Selected(prepared) = prepared_auto {
        let storage = realize_parallel_copied_profile(ctx, gguf, expected, prepared)?;
        return Ok((DirectStorage::ForcedParallelCopied(storage), false));
    }
    if mode == GgufNoCopyMode::Disabled {
        return Ok((DirectStorage::Copied, exact_sentinel));
    }
    if !ctx.device.hasUnifiedMemory() {
        return Err(MfError::LoadPolicy(
            "forced no-copy requires a unified-memory Metal device".to_string(),
        ));
    }
    if exact_sentinel
        && expected
            .first()
            .is_none_or(|request| request.kind != ModelWeightStorageKind::Direct)
    {
        return Err(MfError::LoadPolicy(
            "exact 27B no-copy requires native token embedding residency".to_string(),
        ));
    }
    if !exact_sentinel {
        let storage = planned_retained_storage_for_load(ctx, gguf, expected, prefault_enabled)?;
        return Ok((DirectStorage::ForcedPlanned(storage), false));
    }
    let mmap = gguf.retained_shard_mmap(0).ok_or_else(|| {
        MfError::LoadPolicy("exact no-copy sentinel is missing shard 0".to_string())
    })?;
    let backing = ctx.gguf_no_copy_backing(mmap, 0, GGUF_NO_COPY_ALIGNMENT)?;
    let expected_pages = backing.exposed_len() / backing.page_size();
    if backing.required_alignment() != GGUF_NO_COPY_ALIGNMENT {
        return Err(MfError::LoadPolicy(format!(
            "forced no-copy alignment mismatch: {}/{}",
            backing.required_alignment(),
            GGUF_NO_COPY_ALIGNMENT,
        )));
    }
    let prefault = if prefault_enabled {
        let report = backing.prefault_read();
        if report.page_count != expected_pages || report.covered_bytes != backing.exposed_len() {
            return Err(MfError::LoadPolicy(format!(
                "forced no-copy prefault mismatch: pages={}/{} covered={}/{}",
                report.page_count,
                expected_pages,
                report.covered_bytes,
                backing.exposed_len(),
            )));
        }
        Some(report)
    } else {
        None
    };
    eprintln!(
        concat!(
            "[metal-gguf-no-copy] mapped={} exposed={} suffix={} page={} pages={} ",
            "alignment={} prefault={} prefault_pages={} prefault_ms={:.3} ",
            "checksum={:#018x}"
        ),
        backing.mapped_len(),
        backing.exposed_len(),
        backing.mapped_len() - backing.exposed_len(),
        backing.page_size(),
        expected_pages,
        backing.required_alignment(),
        if prefault_enabled {
            "enabled"
        } else {
            "disabled"
        },
        prefault.map_or(0, |report| report.page_count),
        prefault.map_or(0.0, |report| report.wall_ms),
        prefault.map_or(0, |report| report.checksum),
    );
    Ok((DirectStorage::ForcedExact27B(backing), true))
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
    /// (mat_vec inputs and lm_head) keep their native dtype. Q4_K/Q8_0
    /// embeddings can opt into native residency once their row kernels apply.
    pub fn load(ctx: &MetalContext, gguf: &GgufFile, model: &Model<'_>) -> Result<Self, MfError> {
        Self::load_with_options(ctx, gguf, model, MetalModelLoadOptions::default())
    }

    pub(crate) fn load_with_options(
        ctx: &MetalContext,
        gguf: &GgufFile,
        model: &Model<'_>,
        options: MetalModelLoadOptions,
    ) -> Result<Self, MfError> {
        Self::load_prepared(Self::prepare_load_with_options(ctx, gguf, model, options)?)
    }

    pub(crate) fn prepare_load_with_options<'ctx, 'gguf, 'model>(
        ctx: &'ctx MetalContext,
        gguf: &'gguf GgufFile,
        model: &'model Model<'gguf>,
        options: MetalModelLoadOptions,
    ) -> Result<PreparedMetalModelLoad<'ctx, 'gguf, 'model>, MfError> {
        let no_copy_mode = gguf_no_copy_mode()?;
        let owned_mode = gguf_owned_arena_mode()?;
        let parallel_mode = gguf_parallel_copy_mode()?;
        let prefault_mode = gguf_no_copy_prefault_mode()?;
        let prefault_present = std::env::var_os("QWEN_GGUF_NO_COPY_PREFAULT").is_some();
        let native_embedding_present = std::env::var_os("QWEN_NATIVE_QUANT_EMBED").is_some();
        let router_f16_value = match std::env::var("QWEN_MOE_ROUTER_F16") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) if parallel_mode.is_forced() => {
                return Err(MfError::LoadPolicy(
                    "QWEN_MOE_ROUTER_F16 is not valid Unicode with parallel copy".to_string(),
                ));
            }
            Err(std::env::VarError::NotUnicode(_)) => None,
        };
        validate_parallel_copy_policy(
            parallel_mode,
            no_copy_mode,
            owned_mode,
            prefault_present,
            native_embedding_present,
            router_f16_value.as_deref(),
        )?;
        if owned_mode == GgufOwnedArenaMode::Forced && no_copy_mode == GgufNoCopyMode::Forced {
            return Err(MfError::LoadPolicy(
                "QWEN_GGUF_OWNED_ARENA and QWEN_GGUF_NO_COPY are mutually exclusive".to_string(),
            ));
        }
        if owned_mode == GgufOwnedArenaMode::Forced
            && prefault_mode != GgufNoCopyPrefaultMode::Default
        {
            return Err(MfError::LoadPolicy(
                "QWEN_GGUF_NO_COPY_PREFAULT is invalid with owned arena".to_string(),
            ));
        }
        if owned_mode == GgufOwnedArenaMode::Disabled
            && no_copy_mode == GgufNoCopyMode::Disabled
            && prefault_mode != GgufNoCopyPrefaultMode::Default
        {
            return Err(MfError::LoadPolicy(
                "QWEN_GGUF_NO_COPY_PREFAULT requires QWEN_GGUF_NO_COPY=1".to_string(),
            ));
        }
        let prefault_enabled = prefault_mode != GgufNoCopyPrefaultMode::Disabled;
        let explicit_override_present =
            auto_parallel_copy_a3b_override_present(|name| std::env::var_os(name).is_some());
        let auto_parallel_copy_a3b = auto_parallel_copy_a3b_enabled(
            options.auto_parallel_copy_a3b,
            parallel_mode,
            explicit_override_present,
        );
        let embedding_mode = native_quant_embedding_mode();
        if (owned_mode == GgufOwnedArenaMode::Forced || parallel_mode.is_forced())
            && embedding_mode != NativeQuantEmbeddingMode::Auto
        {
            return Err(MfError::LoadPolicy(
                "forced owned or parallel storage requires production-auto native embedding selection"
                    .to_string(),
            ));
        }
        let forced_a10b_embedding = embedding_mode == NativeQuantEmbeddingMode::Auto
            && forced_a10b_parallel_pread_embedding_qualified(parallel_mode, gguf, model);
        let embedding_selection = if forced_a10b_embedding {
            NativeQuantEmbeddingSelection::Forced
        } else {
            resolve_native_quant_embedding(
                embedding_mode,
                native_quant_embedding_supported(model.token_embd.dtype, &model.token_embd.shape),
                native_quant_embedding_default_promoted(
                    &model.arch,
                    model.tied_embeddings,
                    model.mtp.is_some(),
                    model.token_embd.dtype,
                    &model.token_embd.shape,
                ),
            )
        };
        emit_native_quant_embedding_policy(model, embedding_selection);
        let router_f16 = moe_router_f16_enabled();
        let expected =
            model_weight_storage_requests(model, embedding_selection.uses_native(), router_f16)?;
        let auto = if auto_parallel_copy_a3b {
            match select_auto_parallel_copy_profile(
                ctx,
                gguf,
                model,
                &expected,
                embedding_selection,
            )? {
                Some(profile) => match auto_parallel_copy_population(profile.id) {
                    Some(population) => {
                        let prepared = prepare_parallel_copied_profile(
                            ctx,
                            gguf,
                            model,
                            &expected,
                            embedding_selection,
                            profile,
                            population,
                            ParallelDestinationLength::LogicalExact,
                        )?;
                        emit_metal_load_line(format_args!(
                            "[metal-gguf-parallel-policy] mode=auto profile={}",
                            profile.id.label()
                        ));
                        PreparedAutoSelection::Selected(prepared)
                    }
                    None => PreparedAutoSelection::NoMatch,
                },
                None => PreparedAutoSelection::NoMatch,
            }
        } else {
            PreparedAutoSelection::NotEligible
        };
        Ok(PreparedMetalModelLoad {
            ctx,
            gguf,
            model,
            storage: ResolvedStoragePolicy {
                no_copy_mode,
                prefault_enabled,
                owned_mode,
                parallel_mode,
            },
            choices: ResolvedWeightLoadChoices {
                embedding_selection,
                router_f16,
                fused_qkv_g8: prefill_attn_fused_qkv_g8_enabled(),
            },
            expected,
            auto,
        })
    }

    pub(crate) fn load_prepared(
        prepared: PreparedMetalModelLoad<'_, '_, '_>,
    ) -> Result<Self, MfError> {
        let PreparedMetalModelLoad {
            ctx,
            gguf,
            model,
            storage,
            choices,
            expected,
            auto,
        } = prepared;
        let (direct_storage, exact_sentinel) = direct_storage_for_load(
            ctx,
            gguf,
            model,
            &expected,
            storage.no_copy_mode,
            storage.prefault_enabled,
            storage.owned_mode,
            storage.parallel_mode,
            auto,
            choices.embedding_selection,
        )?;
        Self::load_with_direct_storage(
            ctx,
            gguf,
            model,
            choices,
            &expected,
            direct_storage,
            exact_sentinel,
        )
    }

    #[cfg(test)]
    fn load_with_no_copy_policy(
        ctx: &MetalContext,
        gguf: &GgufFile,
        model: &Model<'_>,
        no_copy_mode: GgufNoCopyMode,
        prefault_enabled: bool,
    ) -> Result<Self, MfError> {
        Self::load_with_storage_policy(
            ctx,
            gguf,
            model,
            no_copy_mode,
            prefault_enabled,
            GgufOwnedArenaMode::Disabled,
            GgufParallelCopyMode::Disabled,
            false,
        )
    }

    #[cfg(test)]
    fn load_with_storage_policy(
        ctx: &MetalContext,
        gguf: &GgufFile,
        model: &Model<'_>,
        no_copy_mode: GgufNoCopyMode,
        prefault_enabled: bool,
        owned_mode: GgufOwnedArenaMode,
        parallel_mode: GgufParallelCopyMode,
        auto_parallel_copy_a3b: bool,
    ) -> Result<Self, MfError> {
        if auto_parallel_copy_a3b {
            return Err(MfError::LoadPolicy(
                "test storage-policy helper cannot select Auto parallel copy".to_string(),
            ));
        }
        let embedding_mode = native_quant_embedding_mode();
        if (owned_mode == GgufOwnedArenaMode::Forced || parallel_mode.is_forced())
            && embedding_mode != NativeQuantEmbeddingMode::Auto
        {
            return Err(MfError::LoadPolicy(
                "forced owned or parallel storage requires production-auto native embedding selection"
                    .to_string(),
            ));
        }
        let embedding_selection = resolve_native_quant_embedding(
            embedding_mode,
            native_quant_embedding_supported(model.token_embd.dtype, &model.token_embd.shape),
            native_quant_embedding_default_promoted(
                &model.arch,
                model.tied_embeddings,
                model.mtp.is_some(),
                model.token_embd.dtype,
                &model.token_embd.shape,
            ),
        );
        emit_native_quant_embedding_policy(model, embedding_selection);
        let router_f16 = moe_router_f16_enabled();
        let expected_storage_requests =
            model_weight_storage_requests(model, embedding_selection.uses_native(), router_f16)?;
        let (direct_storage, exact_sentinel) = direct_storage_for_load(
            ctx,
            gguf,
            model,
            &expected_storage_requests,
            no_copy_mode,
            prefault_enabled,
            owned_mode,
            parallel_mode,
            PreparedAutoSelection::NotEligible,
            embedding_selection,
        )?;
        Self::load_with_direct_storage(
            ctx,
            gguf,
            model,
            ResolvedWeightLoadChoices {
                embedding_selection,
                router_f16,
                fused_qkv_g8: prefill_attn_fused_qkv_g8_enabled(),
            },
            &expected_storage_requests,
            direct_storage,
            exact_sentinel,
        )
    }

    fn load_with_direct_storage(
        ctx: &MetalContext,
        gguf: &GgufFile,
        model: &Model<'_>,
        choices: ResolvedWeightLoadChoices,
        expected_storage_requests: &[ModelWeightStorageRequest<'_>],
        direct_storage: DirectStorage,
        exact_sentinel: bool,
    ) -> Result<Self, MfError> {
        let residency_set = create_a10b_parallel_residency_set(ctx, &direct_storage)?;
        let mut loader = MetalWeightLoader::new(ctx, gguf, direct_storage, choices.router_f16);
        let token_embd =
            loader.load_embedding(model.token_embd, choices.embedding_selection.uses_native())?;
        let output_norm = loader.load_f32(model.output_norm)?;
        let lm_head = loader.load_weight(model.lm_head)?;

        let load_attn_qkv_fused = |q: &MetalTensor,
                                   k: &MetalTensor,
                                   v: &MetalTensor|
         -> Result<Option<MetalTensor>, MfError> {
            if !choices.fused_qkv_g8 {
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
                    let attn_norm = loader.load_f32(g.attn_norm)?;
                    let post_attn_norm = loader.load_f32(g.post_attention_norm)?;
                    let ffn_gate = loader.load_weight(g.ffn_gate)?;
                    let ffn_up = loader.load_weight(g.ffn_up)?;
                    let ffn_down = loader.load_weight(g.ffn_down)?;
                    let in_proj_qkv = loader.load_weight(g.in_proj_qkv)?;
                    let in_proj_z = loader.load_weight(g.in_proj_z)?;
                    let beta_proj = loader.load_weight(g.beta_proj)?;
                    let alpha_proj = loader.load_weight(g.alpha_proj)?;
                    let a_log = loader.load_f32(g.a_log)?;
                    let dt_bias = loader.load_f32(g.dt_bias)?;
                    let conv1d = loader.load_f32(g.conv1d)?;
                    let norm = loader.load_f32(g.norm)?;
                    let out_proj = loader.load_weight(g.out_proj)?;
                    let ffn_moe = match g.ffn_moe.as_ref() {
                        Some(moe) => Some(loader.load_moe(moe)?),
                        None => None,
                    };
                    blocks.push(MetalBlock::Gdn(MetalGdnBlock {
                        attn_norm,
                        post_attn_norm,
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                        in_proj_qkv,
                        in_proj_z,
                        beta_proj,
                        alpha_proj,
                        a_log,
                        dt_bias,
                        conv1d,
                        norm,
                        out_proj,
                        ffn_moe,
                    }));
                }
                Block::Attn(a) => {
                    let q = loader.load_weight(a.q)?;
                    let k = loader.load_weight(a.k)?;
                    let v = loader.load_weight(a.v)?;
                    let attn_norm = loader.load_f32(a.attn_norm)?;
                    let post_attn_norm = loader.load_f32(a.post_attention_norm)?;
                    let ffn_gate = loader.load_weight(a.ffn_gate)?;
                    let ffn_up = loader.load_weight(a.ffn_up)?;
                    let ffn_down = loader.load_weight(a.ffn_down)?;
                    let qkv_fused = load_attn_qkv_fused(&q, &k, &v)?;
                    loader.record_derived(qkv_fused.as_ref())?;
                    let o = loader.load_weight(a.o)?;
                    let q_norm = loader.load_f32(a.q_norm)?;
                    let k_norm = loader.load_f32(a.k_norm)?;
                    let ffn_moe = match a.ffn_moe.as_ref() {
                        Some(moe) => Some(loader.load_moe(moe)?),
                        None => None,
                    };
                    blocks.push(MetalBlock::Attn(MetalAttnBlock {
                        attn_norm,
                        post_attn_norm,
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                        qkv_fused,
                        q,
                        k,
                        v,
                        o,
                        q_norm,
                        k_norm,
                        ffn_moe,
                    }));
                }
            }
        }
        loader.finish(exact_sentinel, expected_storage_requests)?;

        Ok(Self {
            arch: model.arch,
            tied_embeddings: model.tied_embeddings,
            token_embd,
            output_norm,
            lm_head,
            blocks,
            _residency_set: residency_set,
        })
    }
}

/// Per-sequence state: GDN conv buffers + SSM states (one set per GDN
/// layer), KV cache (one set per attn layer), and a small pool of
/// scratch activation tensors that are reused across layers.
pub struct MetalSession {
    poison_reason: Option<&'static str>,
    /// (kernel-1) * conv_dim per GDN layer, F32, contiguous.
    pub gdn_conv: Vec<MetalTensor>,
    /// n_v_heads * head_dim * head_dim per GDN layer, F32.
    pub gdn_state: Vec<MetalTensor>,

    /// `[capacity_tokens, n_kv_heads, head_dim]` per attention layer in the
    /// active KV storage type (normally F16, optionally Q8_0).
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
    pub mixer_out: MetalTensor,  // hidden_size — mixer output (GDN or attn)

    // Attention scratch.
    pub attn_q_full: MetalTensor,   // 2 * q_dim — Q + gate interleaved
    pub attn_q: MetalTensor,        // q_dim — Q only
    pub attn_gate: MetalTensor,     // q_dim — sigmoid'd gate
    pub attn_q_normed: MetalTensor, // q_dim
    pub attn_k_now: MetalTensor,    // kv_dim — current step K
    pub attn_v_now: MetalTensor,    // kv_dim — current step V
    pub attn_k_normed: MetalTensor, // kv_dim
    pub attn_o: MetalTensor,        // q_dim — attention output
    // v4 flash-attn split-K partials. Sized for ATTN_V4_MAX_NWG; the
    // dispatcher passes the chosen NWG ≤ this value.
    pub attn_v4_o_partial: MetalTensor, // n_kv * NWG_max * GROUP * head_dim
    pub attn_v4_ml_partial: MetalTensor, // n_kv * NWG_max * GROUP * 2

    pub logits: MetalTensor,           // vocab_size
    pub argmax_tok: MetalTensor,       // [1] I32
    pub moe_router_probs: MetalTensor, // [n_expert] F32 (or [1] on dense models)
    pub moe_topk_idx: MetalTensor,     // [top_k] i32 in F32 buffer (or [1] on dense models)
    pub moe_topk_weight: MetalTensor,  // [top_k] F32 (or [1] on dense models)
    pub moe_shared_gate: MetalTensor,  // [1] F32
    pub moe_inner: MetalTensor,        // [top_k, expert_ffn] F32 (or [1] on dense models)
    pub moe_expert_out: MetalTensor,   // [top_k, hidden] F32 (or [1] on dense models)
    pub ids_buf: MetalTensor,          // [1] I32 input-token scratch
}

impl MetalSession {
    pub(crate) fn ensure_usable(&self) -> Result<(), MfError> {
        match self.poison_reason {
            Some(reason) => Err(MfError::SessionPoisoned { reason }),
            None => Ok(()),
        }
    }

    pub(crate) fn poison(&mut self, reason: &'static str) {
        self.poison_reason = Some(reason);
    }

    fn any_mutable_tensor(&self, mut predicate: impl FnMut(&MetalTensor) -> bool) -> bool {
        self.gdn_conv.iter().any(&mut predicate)
            || self.gdn_state.iter().any(&mut predicate)
            || self.kv_k.iter().any(&mut predicate)
            || self.kv_v.iter().any(&mut predicate)
            || [
                &self.x,
                &self.h,
                &self.ffn_gate,
                &self.ffn_up,
                &self.ffn_inner,
                &self.ffn_out,
                &self.gdn_qkv,
                &self.gdn_qkv_conv,
                &self.gdn_z,
                &self.gdn_b,
                &self.gdn_beta,
                &self.gdn_a,
                &self.gdn_alpha,
                &self.gdn_q_norm,
                &self.gdn_k_norm,
                &self.gdn_out,
                &self.gdn_normed,
                &self.mixer_out,
                &self.attn_q_full,
                &self.attn_q,
                &self.attn_gate,
                &self.attn_q_normed,
                &self.attn_k_now,
                &self.attn_v_now,
                &self.attn_k_normed,
                &self.attn_o,
                &self.attn_v4_o_partial,
                &self.attn_v4_ml_partial,
                &self.logits,
                &self.argmax_tok,
                &self.moe_router_probs,
                &self.moe_topk_idx,
                &self.moe_topk_weight,
                &self.moe_shared_gate,
                &self.moe_inner,
                &self.moe_expert_out,
                &self.ids_buf,
            ]
            .into_iter()
            .any(predicate)
    }

    pub(crate) fn aliases_mutable_buffer(&self, candidate: &MetalTensor) -> bool {
        self.any_mutable_tensor(|tensor| {
            Retained::as_ptr(&candidate.buffer) == Retained::as_ptr(&tensor.buffer)
        })
    }

    pub(crate) fn aliases_mutable_session(&self, other: &Self) -> bool {
        other.any_mutable_tensor(|tensor| self.aliases_mutable_buffer(tensor))
    }

    pub(crate) fn has_internal_mutable_alias(&self) -> bool {
        let mut seen = std::collections::HashSet::new();
        self.any_mutable_tensor(|tensor| {
            !seen.insert(Retained::as_ptr(&tensor.buffer) as *const () as usize)
        })
    }

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
            poison_reason: None,
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
            mixer_out: MetalTensor::zeros_f32(ctx, vec![h])?,
            attn_q_full: MetalTensor::zeros_f32(ctx, vec![attn_q_full_elems])?,
            attn_q: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_gate: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_q_normed: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_k_now: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_v_now: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_k_normed: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_o: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            // v4 partials: n_kv * NWG_max * GROUP * head_dim (and *2 for ml).
            attn_v4_o_partial: MetalTensor::zeros_f32(ctx, vec![attn_v4_o_partial_elems])?,
            attn_v4_ml_partial: MetalTensor::zeros_f32(ctx, vec![attn_v4_ml_partial_elems])?,
            logits: MetalTensor::zeros_f32(ctx, vec![arch.vocab_size as u64])?,
            argmax_tok: MetalTensor::zeros_i32(ctx, vec![1])?,
            moe_router_probs: MetalTensor::zeros_f32(ctx, vec![moe_router_n])?,
            moe_topk_idx: MetalTensor::zeros_f32(ctx, vec![moe_topk_n])?,
            moe_topk_weight: MetalTensor::zeros_f32(ctx, vec![moe_topk_n])?,
            moe_shared_gate: MetalTensor::zeros_f32(ctx, vec![1])?,
            moe_inner: MetalTensor::zeros_f32(ctx, vec![moe_inner_n])?,
            moe_expert_out: MetalTensor::zeros_f32(ctx, vec![moe_expert_out_n])?,
            ids_buf: MetalTensor::zeros_i32(ctx, vec![1])?,
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

    fn write_moe_route_result(&self, session: &mut MetalSession, route: &MoeRouteDecision) {
        unsafe {
            let idx_ptr = (session.moe_topk_idx.buffer.contents().as_ptr() as *mut i32)
                .add((session.moe_topk_idx.offset / 4) as usize);
            let w_ptr = (session.moe_topk_weight.buffer.contents().as_ptr() as *mut f32)
                .add((session.moe_topk_weight.offset / 4) as usize);
            for (i, &(expert, weight)) in route.ranked.iter().enumerate() {
                *idx_ptr.add(i) = expert as i32;
                *w_ptr.add(i) = weight;
            }
            let shared_ptr = (session.moe_shared_gate.buffer.contents().as_ptr() as *mut f32)
                .add((session.moe_shared_gate.offset / 4) as usize);
            *shared_ptr = route.shared_gate_scalar;
        }
    }

    pub(crate) fn encode_moe_route_prepare(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        if decode_moe_noop_route_enabled() {
            return Ok(());
        }
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

    fn encode_moe_topk_parallel_from_logits(
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

        encode_topk_logits_softmax_parallel_f32(
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

        let gate_up_supported =
            moe_routed_gate_up_decode_supported(moe.gate_exps.dtype, moe.up_exps.dtype);
        if !gate_up_supported {
            return Err(MfError::UnsupportedDtype {
                name: "MoE routed gate/up expert banks".into(),
                dtype: moe.gate_exps.dtype,
            });
        }
        if !matches!(
            moe.down_exps.dtype,
            GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::Q8_0
                | GgmlType::IQ4_XS
                | GgmlType::BF16
                | GgmlType::F32
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

        self.encode_moe_routed_gate_up_gpu(enc, session, moe)?;
        if decode_moe_noop_routed_down_enabled() {
            encode_fill_f32(self.ctx, enc, &session.mixer_out, 0.0)?;
            return Ok(());
        }
        match moe.down_exps.dtype {
            GgmlType::Q4_K => {
                encode_moe_down_q4_K_f32(
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
            GgmlType::Q8_0 => encode_moe_down_weighted_sum_q8_0_f32(
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
                if decode_moe_iq4_down_fast_enabled() {
                    encode_moe_down_iq4_xs_f32_fast(
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
                } else {
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
                }
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
            GgmlType::F32 => {
                encode_moe_down_f32_f32(
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

    fn encode_moe_routed_gate_up_gpu(
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
            return Ok(());
        }

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
            GgmlType::Q6_K => encode_moe_swiglu_q6_K_f32(
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
            GgmlType::Q8_0 => encode_moe_swiglu_q8_0_f32(
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
            GgmlType::IQ3_XXS => {
                if decode_moe_iq3_fast_swiglu_enabled() {
                    encode_moe_swiglu_iq3_xxs_f32_fast(
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
                } else if decode_moe_iq3_fused_swiglu_enabled() {
                    encode_moe_swiglu_iq3_xxs_f32(
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
                } else {
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
            }
            GgmlType::IQ3_S => {
                if decode_moe_iq3_fast_swiglu_enabled() {
                    encode_moe_swiglu_iq3_s_f32_fast(
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
                } else if decode_moe_iq3_fused_swiglu_enabled() {
                    encode_moe_swiglu_iq3_s_f32(
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
                } else {
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
            GgmlType::Q4_K => {
                encode_moe_down_q4_K_f32(
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
            GgmlType::Q8_0 => {
                encode_moe_down_weighted_sum_q8_0_f32(
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
                if decode_moe_iq4_down_fast_enabled() {
                    encode_moe_down_iq4_xs_f32_fast(
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
                } else {
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
                }
                true
            }
            GgmlType::F32 => {
                encode_moe_down_f32_f32(
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
        stage_recorder: Option<&mut DecodeStageRecorder>,
        stage_meta: Option<DecodeStageMeta>,
    ) -> Result<bool, MfError> {
        let enc = begin_decode_stage(cmd_buf, stage_recorder, stage_meta, true)?;
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
        mut stage_recorder: Option<&mut DecodeStageRecorder>,
        stage_meta: Option<DecodeStageMeta>,
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
            let enc = begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
            encode_fill_f32(self.ctx, &enc, &session.mixer_out, 0.0)?;
            self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
            enc.end();
            return Ok(false);
        }

        let pending = match moe.down_exps.dtype {
            GgmlType::Q4_K => {
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
                encode_moe_down_q4_K_f32(
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
            GgmlType::Q5_K => {
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
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
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
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
            GgmlType::Q8_0 => {
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
                encode_moe_down_weighted_sum_q8_0_f32(
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
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
                if decode_moe_iq4_down_fast_enabled() {
                    encode_moe_down_iq4_xs_f32_fast(
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
                } else {
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
                }
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                true
            }
            GgmlType::F32 => {
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
                encode_moe_down_f32_f32(
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
                let enc = begin_decode_stage(cmd_buf, stage_recorder, stage_meta, true)?;
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
        stage_recorder: Option<&mut DecodeStageRecorder>,
        stage_meta: Option<DecodeStageMeta>,
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

        let enc = begin_decode_stage(cmd_buf, stage_recorder, stage_meta, false)?;
        if routed_weighted_sum_is_pending {
            if decode_moe_grouped_finalizer_enabled() {
                let shared_out = session.ffn_out.view_subrange(0, vec![h as u64]);
                encode_moe_grouped_finalizer_f32(
                    self.ctx,
                    &enc,
                    &moe_expert_out,
                    &topk_w,
                    &session.moe_shared_gate,
                    &shared_out,
                    &session.x,
                    h,
                    topk,
                    1,
                )?;
            } else {
                encode_moe_weighted_sum_f32(
                    self.ctx,
                    &enc,
                    &moe_expert_out,
                    &topk_w,
                    &session.mixer_out,
                    h,
                    topk,
                )?;
                self.encode_moe_final_residual_gpu(&enc, session)?;
            }
        } else {
            self.encode_moe_final_residual_gpu(&enc, session)?;
        }
        enc.end();
        Ok(())
    }

    pub(crate) fn encode_moe_ffn_apply_gpu_concurrent_shared(
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

        let shared_inner_fused = self
            .encode_moe_ffn_gate_up_wave_gpu(cmd_buf, session, ffn_gate, ffn_up, moe, None, None)?;
        if !shared_inner_fused {
            let enc = KernelEncoder::begin(cmd_buf);
            self.encode_moe_shared_ffn_silu_gpu(&enc, session)?;
            enc.end();
        }
        let routed_weighted_sum_is_pending =
            self.encode_moe_ffn_down_wave_gpu(cmd_buf, session, ffn_down, moe, None, None)?;
        self.encode_moe_ffn_final_wave_gpu(
            cmd_buf,
            session,
            routed_weighted_sum_is_pending,
            None,
            None,
        )?;
        Ok(())
    }

    fn encode_stage_profiled_moe_ffn_waves(
        &self,
        cmd_buf: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        recorder: &mut DecodeStageRecorder,
        block_kind: &'static str,
        block_idx: usize,
        local_idx: usize,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        ffn_down: &MetalTensor,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        if moe.gate_exps.dtype != GgmlType::Q4_K || moe.up_exps.dtype != GgmlType::Q4_K {
            let enc = begin_decode_stage(
                cmd_buf,
                Some(&mut *recorder),
                Some(DecodeStageMeta {
                    family: "moe_fallback_ffn",
                    block_kind,
                    block_index: Some(block_idx),
                    local_index: Some(local_idx),
                }),
                false,
            )?;
            self.encode_moe_ffn_apply_gpu(&enc, session, ffn_gate, ffn_up, ffn_down, moe)?;
            enc.end();
            return Ok(());
        }

        let shared_inner_fused = self.encode_moe_ffn_gate_up_wave_gpu(
            cmd_buf,
            session,
            ffn_gate,
            ffn_up,
            moe,
            Some(&mut *recorder),
            Some(DecodeStageMeta {
                family: "moe_gate_up_wave",
                block_kind,
                block_index: Some(block_idx),
                local_index: Some(local_idx),
            }),
        )?;
        if !shared_inner_fused {
            let enc = begin_decode_stage(
                cmd_buf,
                Some(&mut *recorder),
                Some(DecodeStageMeta {
                    family: "moe_shared_silu",
                    block_kind,
                    block_index: Some(block_idx),
                    local_index: Some(local_idx),
                }),
                false,
            )?;
            self.encode_moe_shared_ffn_silu_gpu(&enc, session)?;
            enc.end();
        }
        let routed_weighted_sum_is_pending = self.encode_moe_ffn_down_wave_gpu(
            cmd_buf,
            session,
            ffn_down,
            moe,
            Some(&mut *recorder),
            Some(DecodeStageMeta {
                family: "moe_down_wave",
                block_kind,
                block_index: Some(block_idx),
                local_index: Some(local_idx),
            }),
        )?;
        self.encode_moe_ffn_final_wave_gpu(
            cmd_buf,
            session,
            routed_weighted_sum_is_pending,
            Some(&mut *recorder),
            Some(DecodeStageMeta {
                family: "moe_final_wave",
                block_kind,
                block_index: Some(block_idx),
                local_index: Some(local_idx),
            }),
        )
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

    fn moe_block_slot_by_index(
        &self,
        block_idx: usize,
    ) -> Result<(&MetalBlock, MixerSlot), MfError> {
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (i, block) in self.model.blocks.iter().enumerate() {
            match block {
                MetalBlock::Gdn(_) => {
                    if i == block_idx {
                        return Ok((block, MixerSlot::Gdn(gdn_idx)));
                    }
                    gdn_idx += 1;
                }
                MetalBlock::Attn(_) => {
                    if i == block_idx {
                        return Ok((block, MixerSlot::Attn(attn_idx)));
                    }
                    attn_idx += 1;
                }
            }
        }
        Err(MfError::Metal(MetalError::BadShape {
            kernel: "encode_moe_block_by_index",
            detail: format!(
                "block index {block_idx} >= n_layer {}",
                self.model.blocks.len()
            ),
        }))
    }

    /// Bench hook: encode one MoE block by absolute block index.
    ///
    /// This keeps the public surface free of the internal `MixerSlot` enum while
    /// allowing block-slice microbenchmarks to execute normal attention/MoE work.
    pub fn encode_moe_block_by_index(
        &self,
        enc: &KernelEncoder,
        block_idx: usize,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let (block, slot) = self.moe_block_slot_by_index(block_idx)?;
        self.encode_moe_block_gpu(enc, block, slot, position, session)
    }

    /// Bench hook: run only the mixer + post-mixer norm for an absolute block.
    /// The caller may inspect `session.h` before running route/FFN.
    pub fn encode_moe_mixer_prep_by_index(
        &self,
        enc: &KernelEncoder,
        block_idx: usize,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let (block, slot) = self.moe_block_slot_by_index(block_idx)?;
        self.encode_moe_mixer_prep(enc, block, slot, position, session)
    }

    /// Bench hook: run only the route preparation for an absolute block.
    pub fn encode_moe_route_prepare_by_index(
        &self,
        enc: &KernelEncoder,
        block_idx: usize,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let (block, _) = self.moe_block_slot_by_index(block_idx)?;
        let moe = match block {
            MetalBlock::Gdn(b) => b.ffn_moe.as_ref(),
            MetalBlock::Attn(b) => b.ffn_moe.as_ref(),
        };
        self.encode_moe_route_prepare(enc, session, moe.ok_or(MfError::UnsupportedMoe)?)
    }

    /// Bench hook: run the route + MoE FFN tail after a caller has already
    /// computed mixer residual and post-mixer norm for this block.
    pub fn encode_moe_ffn_after_mixer_by_index(
        &self,
        enc: &KernelEncoder,
        block_idx: usize,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let (block, _) = self.moe_block_slot_by_index(block_idx)?;
        let (ffn_gate, ffn_up, ffn_down, moe) = match block {
            MetalBlock::Gdn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
            MetalBlock::Attn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
        };
        let moe = moe.ok_or(MfError::UnsupportedMoe)?;
        self.encode_moe_route_prepare(enc, session, moe)?;
        self.encode_moe_ffn_apply_gpu(enc, session, ffn_gate, ffn_up, ffn_down, moe)
    }

    /// Bench hook: run the route and MoE FFN tail with the production
    /// concurrent-shared policy after a caller has computed the mixer and norm.
    #[doc(hidden)]
    pub fn encode_moe_ffn_after_mixer_production_by_index(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        block_idx: usize,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let (block, _) = self.moe_block_slot_by_index(block_idx)?;
        let (ffn_gate, ffn_up, ffn_down, moe) = match block {
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
        let moe = moe.ok_or(MfError::UnsupportedMoe)?;
        if concurrent_shared_moe_decode_enabled() {
            let encoder = KernelEncoder::begin(command);
            self.encode_moe_route_prepare(&encoder, session, moe)?;
            encoder.end();
            self.encode_moe_ffn_apply_gpu_concurrent_shared(
                command, session, ffn_gate, ffn_up, ffn_down, moe,
            )
        } else {
            let encoder = KernelEncoder::begin(command);
            self.encode_moe_route_prepare(&encoder, session, moe)?;
            self.encode_moe_ffn_apply_gpu(&encoder, session, ffn_gate, ffn_up, ffn_down, moe)?;
            encoder.end();
            Ok(())
        }
    }

    /// Bench hook: encode only the shared-expert gate/up half of the production
    /// concurrent wave after route preparation. The routed inner may be supplied
    /// by an exact cross-lane kernel before the remaining tail is encoded.
    #[doc(hidden)]
    pub fn encode_moe_shared_gate_up_by_index(
        &self,
        encoder: &KernelEncoder,
        block_idx: usize,
        session: &mut MetalSession,
    ) -> Result<bool, MfError> {
        let (block, _) = self.moe_block_slot_by_index(block_idx)?;
        let (ffn_gate, ffn_up) = match block {
            MetalBlock::Gdn(block) => (&block.ffn_gate, &block.ffn_up),
            MetalBlock::Attn(block) => (&block.ffn_gate, &block.ffn_up),
        };
        self.encode_moe_shared_ffn_gate_up_gpu(encoder, session, ffn_gate, ffn_up)
    }

    /// Bench hook: finish production shared/routed down and final waves after a
    /// caller has supplied the exact routed inner and encoded shared gate/up.
    #[doc(hidden)]
    pub fn encode_moe_ffn_after_external_routed_inner_by_index(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        block_idx: usize,
        session: &mut MetalSession,
        shared_inner_fused: bool,
    ) -> Result<(), MfError> {
        let (block, _) = self.moe_block_slot_by_index(block_idx)?;
        let (ffn_down, moe) = match block {
            MetalBlock::Gdn(block) => (&block.ffn_down, block.ffn_moe.as_ref()),
            MetalBlock::Attn(block) => (&block.ffn_down, block.ffn_moe.as_ref()),
        };
        let moe = moe.ok_or(MfError::UnsupportedMoe)?;
        if !shared_inner_fused {
            let encoder = KernelEncoder::begin(command);
            self.encode_moe_shared_ffn_silu_gpu(&encoder, session)?;
            encoder.end();
        }
        let routed_weighted_sum_is_pending =
            self.encode_moe_ffn_down_wave_gpu(command, session, ffn_down, moe, None, None)?;
        self.encode_moe_ffn_final_wave_gpu(
            command,
            session,
            routed_weighted_sum_is_pending,
            None,
            None,
        )
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
        if decode_fused_residual_rmsnorm_enabled() {
            encode_residual_rms_norm_mul_f32(
                self.ctx,
                enc,
                &session.x,
                &session.mixer_out,
                post_norm,
                &session.h,
                RMS_EPS,
            )?;
        } else {
            encode_add_inplace_f32(self.ctx, enc, &session.x, &session.mixer_out)?;
            encode_rms_norm_mul_f32(self.ctx, enc, &session.x, post_norm, &session.h, RMS_EPS)?;
        }
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
        session.ensure_usable()?;
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
        self.single_token_argmax_profiled_concurrent_gdn_dense_with_reduction(
            token_id,
            position,
            session,
            ArgmaxReduction::SpeculativeLowest,
        )
    }

    fn single_token_argmax_profiled_concurrent_gdn_dense_with_reduction(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        reduction: ArgmaxReduction,
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
            encode_argmax_reduction(
                self.ctx,
                &enc,
                &session.logits,
                &argmax_tok,
                1,
                arch.vocab_size as usize,
                reduction,
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

    fn resident_lm_head_tail_evidence(&self, session: &MetalSession) -> LmHeadTailEvidence {
        LmHeadTailEvidence {
            kind: LmHeadTailKind::Resident,
            n_in: self.model.arch.hidden_size as usize,
            n_out: self.model.arch.vocab_size as usize,
            weight_offset: self.model.lm_head.offset,
            output_offset: session.logits.offset,
            tail_dispatches: 1,
            full_head_dispatches: 1,
            command_completed: false,
            command_error_none: false,
        }
    }

    pub(crate) fn validate_lm_head_tail(
        &self,
        session: &MetalSession,
        tail: LmHeadTail<'_>,
    ) -> Result<LmHeadTailEvidence, MfError> {
        let h = self.model.arch.hidden_size as usize;
        let vocab = self.model.arch.vocab_size as usize;
        match tail {
            LmHeadTail::Resident => {
                if self.model.lm_head.shape.as_slice() != [h as u64, vocab as u64] {
                    return Err(lm_head_tail_error("resident head shape mismatch"));
                }
                if session.logits.dtype != GgmlType::F32
                    || session.logits.shape.as_slice() != [vocab as u64]
                    || !session.logits.is_writable()
                {
                    return Err(lm_head_tail_error("resident logits shape mismatch"));
                }
                let output_bytes = vocab
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| lm_head_tail_error("resident logits bytes overflow"))?;
                checked_tail_range(&session.logits, output_bytes, "resident logits")?;
                Ok(self.resident_lm_head_tail_evidence(session))
            }
            LmHeadTail::CompactQ6K { weight, output } => {
                if weight.dtype != GgmlType::Q6_K
                    || weight.shape.len() != 2
                    || weight.shape[0] != h as u64
                    || weight.shape[1] == 0
                    || weight.shape[1] > 17
                    || weight.provenance() != MetalTensorProvenance::OwnedWeightReadOnly
                    || weight.is_writable()
                {
                    return Err(lm_head_tail_error("compact Q6_K weight contract mismatch"));
                }
                let n_out = usize::try_from(weight.shape[1])
                    .map_err(|_| lm_head_tail_error("compact width does not fit usize"))?;
                if !h.is_multiple_of(256) {
                    return Err(lm_head_tail_error(
                        "compact input width is not Q6_K block aligned",
                    ));
                }
                if output.dtype != GgmlType::F32
                    || output.shape.as_slice() != [n_out as u64]
                    || !output.is_writable()
                    || output.offset % std::mem::align_of::<f32>() as u64 != 0
                    || weight.offset % 32 != 0
                {
                    return Err(lm_head_tail_error("compact output contract mismatch"));
                }
                if session.h.dtype != GgmlType::F32
                    || session.h.shape.as_slice() != [h as u64]
                    || !session.h.is_writable()
                    || !session
                        .h
                        .offset
                        .is_multiple_of(std::mem::align_of::<f32>() as u64)
                    || session.x.dtype != GgmlType::F32
                    || session.x.shape.as_slice() != [h as u64]
                    || !session.x.is_writable()
                    || !session
                        .x
                        .offset
                        .is_multiple_of(std::mem::align_of::<f32>() as u64)
                {
                    return Err(lm_head_tail_error("session hidden contract mismatch"));
                }
                if session.aliases_mutable_buffer(weight) {
                    return Err(lm_head_tail_error(
                        "compact weight aliases mutable session storage",
                    ));
                }
                if session.aliases_mutable_buffer(output) {
                    return Err(lm_head_tail_error(
                        "compact output aliases mutable session storage",
                    ));
                }
                let weight_bytes = h
                    .checked_div(256)
                    .and_then(|blocks| blocks.checked_mul(210))
                    .and_then(|row| row.checked_mul(n_out))
                    .ok_or_else(|| lm_head_tail_error("compact weight bytes overflow"))?;
                let output_bytes = n_out
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| lm_head_tail_error("compact output bytes overflow"))?;
                let hidden_bytes = h
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| lm_head_tail_error("hidden bytes overflow"))?;
                let weight_range = checked_tail_range(weight, weight_bytes, "compact weight")?;
                let output_range = checked_tail_range(output, output_bytes, "compact output")?;
                let hidden_range = checked_tail_range(&session.h, hidden_bytes, "session.h")?;
                let residual_range = checked_tail_range(&session.x, hidden_bytes, "session.x")?;
                if tail_ranges_overlap(weight, weight_range, output, output_range)
                    || tail_ranges_overlap(&session.h, hidden_range, output, output_range)
                    || tail_ranges_overlap(&session.x, residual_range, output, output_range)
                {
                    return Err(lm_head_tail_error("compact tail buffers overlap"));
                }
                Ok(LmHeadTailEvidence {
                    kind: LmHeadTailKind::CompactQ6K,
                    n_in: h,
                    n_out,
                    weight_offset: weight.offset,
                    output_offset: output.offset,
                    tail_dispatches: 1,
                    full_head_dispatches: 0,
                    command_completed: false,
                    command_error_none: false,
                })
            }
        }
    }

    pub(crate) fn encode_lm_head_tail(
        &self,
        enc: &KernelEncoder,
        session: &MetalSession,
        tail: LmHeadTail<'_>,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        match tail {
            LmHeadTail::Resident => encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                h,
                self.model.arch.vocab_size as usize,
            )?,
            LmHeadTail::CompactQ6K { weight, output } => encode_mat_vec_dispatch(
                self.ctx,
                enc,
                weight,
                &session.h,
                output,
                h,
                usize::try_from(weight.shape[1])
                    .map_err(|_| lm_head_tail_error("compact width does not fit usize"))?,
            )?,
        }
        Ok(())
    }

    fn encode_single_token_concurrent_gdn_moe_body(
        &self,
        cmd_buf: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        {
            let enc = KernelEncoder::begin(cmd_buf);
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
                        let enc = KernelEncoder::begin(cmd_buf);
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
                        let enc = KernelEncoder::begin_concurrent(cmd_buf);
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(cmd_buf);
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
                            cmd_buf,
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
                    let enc = KernelEncoder::begin(cmd_buf);
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
                            cmd_buf,
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
        Ok(())
    }

    fn single_token_profiled_concurrent_gdn_moe_tail_inner(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        tail: LmHeadTail<'_>,
        require_completed_command: bool,
    ) -> Result<(TokenProfile, LmHeadTailEvidence, std::time::Instant), MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let mut evidence = if require_completed_command {
            self.validate_lm_head_tail(session, tail)?
        } else {
            if !matches!(tail, LmHeadTail::Resident) {
                return Err(lm_head_tail_error(
                    "unchecked tail execution is resident-only",
                ));
            }
            self.resident_lm_head_tail_evidence(session)
        };
        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        self.encode_single_token_concurrent_gdn_moe_body(&cmd_buf, position, session)?;
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
            self.encode_lm_head_tail(&enc, session, tail)?;
            enc.end();
        }

        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;
        if require_completed_command {
            let status = cmd_buf.status();
            let error = cmd_buf.error();
            evidence.command_completed = status == MTLCommandBufferStatus::Completed;
            evidence.command_error_none = error.is_none();
            if !evidence.command_completed || !evidence.command_error_none {
                return Err(MfError::CommandBuffer {
                    status: format!("{status:?}"),
                    error: format!("{error:?}"),
                });
            }
        }

        Ok((
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms: t_total.elapsed().as_secs_f64() * 1e3,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
            evidence,
            t_total,
        ))
    }

    #[doc(hidden)]
    pub fn single_token_profiled_concurrent_gdn_moe_with_tail(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        tail: LmHeadTail<'_>,
    ) -> Result<(TokenProfile, LmHeadTailEvidence), MfError> {
        let (profile, evidence, _) = self.single_token_profiled_concurrent_gdn_moe_tail_inner(
            token_id, position, session, tail, true,
        )?;
        Ok((profile, evidence))
    }

    pub fn single_token_profiled_concurrent_gdn_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        let (mut profile, evidence, t_total) = self
            .single_token_profiled_concurrent_gdn_moe_tail_inner(
                token_id,
                position,
                session,
                LmHeadTail::Resident,
                false,
            )?;
        debug_assert_eq!(evidence.kind, LmHeadTailKind::Resident);
        let mut out = vec![0.0f32; self.model.arch.vocab_size as usize];
        unsafe {
            let src = (session.logits.buffer.contents().as_ptr() as *const u8)
                .add(session.logits.offset as usize) as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        profile.total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((out, profile))
    }

    /// Run the production concurrent-MoE full-logit transition with opt-in
    /// attribution around only the existing host destination allocation and
    /// Shared-buffer copy.
    pub fn single_token_sampled_attribution(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile, LogitsReadbackProfile), MfError> {
        if !concurrent_gdn_moe_decode_enabled() {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "sampling_attribution",
                detail: "requires the production concurrent-GDN MoE decode path".into(),
            }));
        }
        let (mut profile, evidence, t_total) = self
            .single_token_profiled_concurrent_gdn_moe_tail_inner(
                token_id,
                position,
                session,
                LmHeadTail::Resident,
                false,
            )?;
        debug_assert_eq!(evidence.kind, LmHeadTailKind::Resident);

        let allocation_t0 = std::time::Instant::now();
        let mut out = vec![0.0f32; self.model.arch.vocab_size as usize];
        let allocation_zero_fill_ms = allocation_t0.elapsed().as_secs_f64() * 1e3;
        if session.logits.dtype != GgmlType::F32 || session.logits.n_elements() < out.len() as u64 {
            return Err(lm_head_tail_error(
                "sampled logits readback requires a complete F32 logits row",
            ));
        }
        let bytes = out
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| lm_head_tail_error("sampled logits byte size overflow"))?;
        let (source_start, _) =
            checked_tail_range(&session.logits, bytes, "sampled logits readback")?;
        if source_start % std::mem::align_of::<f32>() != 0 {
            return Err(lm_head_tail_error(
                "sampled logits source is not aligned for F32 readback",
            ));
        }

        let copy_t0 = std::time::Instant::now();
        unsafe {
            let src = (session.logits.buffer.contents().as_ptr() as *const u8).add(source_start)
                as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let copy_ms = copy_t0.elapsed().as_secs_f64() * 1e3;
        profile.total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            out,
            profile,
            LogitsReadbackProfile {
                timer_spans: 2,
                bytes,
                allocation_zero_fill_ms,
                copy_ms,
            },
        ))
    }

    /// Validate the architecture and decode organization required by scoped
    /// resident-logit sampling before a request starts prefill.
    pub fn ensure_sampled_structural_supported(&self) -> Result<(), MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Moe
            || arch.n_layer != 40
            || arch.hidden_size != 2_048
            || arch.vocab_size != 248_320
            || arch.n_q_heads != 16
            || arch.n_kv_heads != 2
            || arch.attn_head_dim != 256
            || arch.full_attention_interval != 4
            || arch.partial_rotary_factor.to_bits() != 0.25f32.to_bits()
            || arch.gdn_n_k_heads != 16
            || arch.gdn_n_v_heads != 32
            || arch.gdn_head_dim != 128
            || arch.gdn_conv_kernel != 4
            || arch.expert_count != 256
            || arch.expert_used_count != 8
            || arch.expert_feed_forward_length != 512
            || arch.expert_shared_feed_forward_length != 512
            || arch.mtp_n_hidden_layers != 0
            || self.model.blocks.len() != arch.n_layer as usize
            || self.model.lm_head.dtype != GgmlType::Q6_K
            || self.model.lm_head.shape.as_slice() != [2_048, 248_320]
        {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "sampled_structural",
                detail: "requires the frozen Qwen3.6 35B A3B resident-head geometry".into(),
            }));
        }
        if !concurrent_gdn_moe_decode_enabled() {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "sampled_structural",
                detail: "requires the production concurrent-GDN MoE decode path".into(),
            }));
        }
        Ok(())
    }

    /// Validate the exact session-row contract before a flagged request starts
    /// prompt prefill.
    pub fn ensure_sampled_structural_session_supported(
        &self,
        session: &MetalSession,
    ) -> Result<(), MfError> {
        self.ensure_sampled_structural_supported()?;
        self.validate_sampled_structural_logits(session)?;
        Ok(())
    }

    /// Execute the production concurrent-MoE transition and expose its
    /// synchronized Shared logits row only to the bounded CPU sampler.
    pub fn single_token_sampled_structural(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        sampler: &mut Sampler,
    ) -> Result<SampledStructuralOutcome, MfError> {
        self.single_token_sampled_structural_scoped(token_id, position, session, |row| {
            sampler.sample_bounded_top_k(row)
        })
    }

    fn single_token_sampled_structural_scoped<R, F>(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        consume: F,
    ) -> Result<(R, TokenProfile, StructuralRowEvidence), MfError>
    where
        F: for<'row> FnOnce(&'row [f32]) -> R,
    {
        self.ensure_sampled_structural_supported()?;
        let (mut profile, evidence, t_total) = self
            .single_token_profiled_concurrent_gdn_moe_tail_inner(
                token_id,
                position,
                session,
                LmHeadTail::Resident,
                false,
            )?;
        debug_assert_eq!(evidence.kind, LmHeadTailKind::Resident);
        let mut structural_evidence = StructuralRowEvidence {
            resident_head_wait_calls: 1,
            ..StructuralRowEvidence::default()
        };
        let validated = self.validate_sampled_structural_logits(session)?;
        structural_evidence.validated_shared_row_calls = 1;
        let row = unsafe { std::slice::from_raw_parts(validated.source.as_ptr(), validated.len) };
        let result = consume(row);
        profile.total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((result, profile, structural_evidence))
    }

    fn validate_sampled_structural_logits(
        &self,
        session: &MetalSession,
    ) -> Result<ValidatedSharedLogits, MfError> {
        let vocab = usize::try_from(self.model.arch.vocab_size)
            .map_err(|_| lm_head_tail_error("sampled logits vocabulary does not fit usize"))?;
        if session.logits.dtype != GgmlType::F32
            || session.logits.shape != [self.model.arch.vocab_size as u64]
            || session.logits.provenance() != MetalTensorProvenance::OwnedWritable
            || session.logits.buffer.storageMode() != MTLStorageMode::Shared
        {
            return Err(lm_head_tail_error(
                "sampled structural logits require exact writable Shared F32 [vocab] storage",
            ));
        }
        let bytes = vocab
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| lm_head_tail_error("sampled structural logits byte size overflow"))?;
        let (source_start, _) =
            checked_tail_range(&session.logits, bytes, "sampled structural logits")?;
        let base = std::ptr::NonNull::new(session.logits.buffer.contents().as_ptr() as *mut u8)
            .ok_or_else(|| {
                lm_head_tail_error("sampled structural logits have null host contents")
            })?;
        let source = unsafe { base.as_ptr().add(source_start) };
        if !(source as usize).is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(lm_head_tail_error(
                "sampled structural logits source is not aligned for F32 access",
            ));
        }
        let source = std::ptr::NonNull::new(source.cast::<f32>())
            .ok_or_else(|| lm_head_tail_error("sampled structural logits have null F32 source"))?;
        Ok(ValidatedSharedLogits { source, len: vocab })
    }

    pub fn single_token_argmax_profiled_concurrent_gdn_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(i32, TokenProfile), MfError> {
        self.single_token_argmax_profiled_concurrent_gdn_moe_with_reduction(
            token_id,
            position,
            session,
            ArgmaxReduction::SpeculativeLowest,
        )
    }

    fn single_token_argmax_profiled_concurrent_gdn_moe_with_reduction(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        reduction: ArgmaxReduction,
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
            encode_argmax_reduction(
                self.ctx,
                &enc,
                &session.logits,
                &argmax_tok,
                1,
                arch.vocab_size as usize,
                reduction,
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

    pub fn single_token_argmax_stage_profiled_concurrent_gdn_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        split_attn_route: bool,
        split_attn_detail: bool,
        split_gdn_after: bool,
    ) -> Result<(i32, DecodeStageProfile), MfError> {
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
        let sample_count = 2 * (2 + self.model.blocks.len() * 16);
        let mut recorder = DecodeStageRecorder::new(self.ctx, sample_count)?;

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");

        {
            let enc = begin_decode_stage(
                &cmd_buf,
                Some(&mut recorder),
                Some(DecodeStageMeta {
                    family: "embed",
                    block_kind: "tail",
                    block_index: None,
                    local_index: None,
                }),
                false,
            )?;
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
        for (block_idx, block) in self.model.blocks.iter().enumerate() {
            match block {
                MetalBlock::Gdn(g) => {
                    let local_idx = gdn_idx;
                    gdn_idx += 1;
                    let moe = g.ffn_moe.as_ref().ok_or(MfError::UnsupportedMoe)?;
                    {
                        let enc = begin_decode_stage(
                            &cmd_buf,
                            Some(&mut recorder),
                            Some(DecodeStageMeta {
                                family: "gdn_pre_norm",
                                block_kind: "gdn",
                                block_index: Some(block_idx),
                                local_index: Some(local_idx),
                            }),
                            false,
                        )?;
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
                        let enc = begin_decode_stage(
                            &cmd_buf,
                            Some(&mut recorder),
                            Some(DecodeStageMeta {
                                family: "gdn_front",
                                block_kind: "gdn",
                                block_index: Some(block_idx),
                                local_index: Some(local_idx),
                            }),
                            true,
                        )?;
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                    }
                    if split_gdn_after {
                        let v_dim = arch.gdn_n_v_heads as usize * arch.gdn_head_dim as usize;
                        let gdn_qkv = session.gdn_qkv.clone();
                        let gdn_z = session.gdn_z.clone();
                        let gdn_alpha = session.gdn_alpha.clone();
                        let gdn_beta = session.gdn_beta.clone();
                        let gdn_normed = session.gdn_normed.clone();

                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "gdn_beta_alpha",
                                    block_kind: "gdn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            if !gdn_beta_projection_fused(g) {
                                encode_sigmoid_f32(
                                    self.ctx,
                                    &enc,
                                    &session.gdn_b,
                                    &session.gdn_beta,
                                )?;
                            }
                            encode_gdn_decay_chain_f32(
                                self.ctx,
                                &enc,
                                &session.gdn_a,
                                &g.dt_bias,
                                &g.a_log,
                                &session.gdn_alpha,
                            )?;
                            enc.end();
                        }
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "gdn_tail",
                                    block_kind: "gdn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            self.encode_gdn_tail(
                                &enc,
                                g,
                                local_idx,
                                session,
                                &gdn_qkv,
                                &gdn_z,
                                &gdn_alpha,
                                &gdn_beta,
                                &gdn_normed,
                            )?;
                            enc.end();
                        }
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "gdn_out_proj",
                                    block_kind: "gdn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            if decode_gdn_noop_out_enabled() {
                                encode_fill_f32(self.ctx, &enc, &session.mixer_out, 0.0)?;
                            } else {
                                encode_mat_vec_dispatch(
                                    self.ctx,
                                    &enc,
                                    &g.out_proj,
                                    &gdn_normed,
                                    &session.mixer_out,
                                    v_dim,
                                    h,
                                )?;
                            }
                            enc.end();
                        }
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "gdn_resid_post_norm",
                                    block_kind: "gdn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
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
                        }
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: if concurrent_shared_moe_decode_enabled() {
                                        "gdn_route"
                                    } else {
                                        "gdn_route_ffn_serial"
                                    },
                                    block_kind: "gdn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
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
                    } else {
                        let enc = begin_decode_stage(
                            &cmd_buf,
                            Some(&mut recorder),
                            Some(DecodeStageMeta {
                                family: if concurrent_shared_moe_decode_enabled() {
                                    "gdn_after_route"
                                } else {
                                    "gdn_after_route_ffn_serial"
                                },
                                block_kind: "gdn",
                                block_index: Some(block_idx),
                                local_index: Some(local_idx),
                            }),
                            false,
                        )?;
                        self.encode_gdn_after_projections(&enc, g, local_idx, session)?;
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
                        self.encode_stage_profiled_moe_ffn_waves(
                            &cmd_buf,
                            &mut recorder,
                            "gdn",
                            block_idx,
                            local_idx,
                            session,
                            &g.ffn_gate,
                            &g.ffn_up,
                            &g.ffn_down,
                            moe,
                        )?;
                    }
                }
                MetalBlock::Attn(a) => {
                    let local_idx = attn_idx;
                    attn_idx += 1;
                    let slot = MixerSlot::Attn(local_idx);
                    let moe = a.ffn_moe.as_ref().ok_or(MfError::UnsupportedMoe)?;
                    if split_attn_detail {
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "attn_pre_norm",
                                    block_kind: "attn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
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
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "attn_front_proj",
                                    block_kind: "attn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            self.encode_attn_front_projections(&enc, a, session)?;
                            enc.end();
                        }
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "attn_body_out",
                                    block_kind: "attn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            self.encode_attn_after_projections(
                                &enc, a, local_idx, position, session,
                            )?;
                            enc.end();
                        }
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "attn_resid_post_norm",
                                    block_kind: "attn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            encode_add_inplace_f32(self.ctx, &enc, &session.x, &session.mixer_out)?;
                            encode_rms_norm_mul_f32(
                                self.ctx,
                                &enc,
                                &session.x,
                                &a.post_attn_norm,
                                &session.h,
                                RMS_EPS,
                            )?;
                            enc.end();
                        }
                        let enc = begin_decode_stage(
                            &cmd_buf,
                            Some(&mut recorder),
                            Some(DecodeStageMeta {
                                family: if concurrent_shared_moe_decode_enabled() {
                                    "attn_route"
                                } else {
                                    "attn_route_ffn_serial"
                                },
                                block_kind: "attn",
                                block_index: Some(block_idx),
                                local_index: Some(local_idx),
                            }),
                            false,
                        )?;
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
                    } else if split_attn_route {
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "attn_mixer",
                                    block_kind: "attn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            self.encode_moe_mixer_prep(&enc, block, slot, position, session)?;
                            enc.end();
                        }
                        let enc = begin_decode_stage(
                            &cmd_buf,
                            Some(&mut recorder),
                            Some(DecodeStageMeta {
                                family: if concurrent_shared_moe_decode_enabled() {
                                    "attn_route"
                                } else {
                                    "attn_route_ffn_serial"
                                },
                                block_kind: "attn",
                                block_index: Some(block_idx),
                                local_index: Some(local_idx),
                            }),
                            false,
                        )?;
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
                    } else {
                        let enc = begin_decode_stage(
                            &cmd_buf,
                            Some(&mut recorder),
                            Some(DecodeStageMeta {
                                family: if concurrent_shared_moe_decode_enabled() {
                                    "attn_mixer_route"
                                } else {
                                    "attn_mixer_route_ffn_serial"
                                },
                                block_kind: "attn",
                                block_index: Some(block_idx),
                                local_index: Some(local_idx),
                            }),
                            false,
                        )?;
                        self.encode_moe_mixer_prep(&enc, block, slot, position, session)?;
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
                    }
                    if concurrent_shared_moe_decode_enabled() {
                        self.encode_stage_profiled_moe_ffn_waves(
                            &cmd_buf,
                            &mut recorder,
                            "attn",
                            block_idx,
                            local_idx,
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
            let enc = begin_decode_stage(
                &cmd_buf,
                Some(&mut recorder),
                Some(DecodeStageMeta {
                    family: "tail_lm_head_argmax",
                    block_kind: "tail",
                    block_index: None,
                    local_index: None,
                }),
                false,
            )?;
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
        let token = TokenProfile {
            cpu_encode_ms,
            cpu_to_gpu_complete_ms,
            gpu_kernel_ms,
            total_ms,
            moe_cpu_route_ms: 0.0,
            moe_cmd_count: 1,
        };
        let profile = recorder.resolve(self.ctx, token)?;
        Ok((argmax, profile))
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
        session.ensure_usable()?;
        let (argmax, _) = self.single_token_argmax_profiled(token_id, position, session)?;
        Ok(argmax)
    }

    pub fn single_token_greedy(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<GreedySelection, MfError> {
        session.ensure_usable()?;
        let (raw, _) = self.single_token_reduced_profiled(
            token_id,
            position,
            session,
            ArgmaxReduction::GreedyTotal,
        )?;
        Ok(GreedySelection::from_encoded(raw))
    }

    fn single_token_argmax_profiled_dense_serial(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        reduction: ArgmaxReduction,
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

        self.encode_single_token_argmax_dense_with_reduction(
            &enc,
            position,
            session,
            &ids_buf,
            &argmax_tok,
            reduction,
        )?;

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
        session.ensure_usable()?;
        self.single_token_reduced_profiled(
            token_id,
            position,
            session,
            ArgmaxReduction::SpeculativeLowest,
        )
    }

    fn single_token_reduced_profiled(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        reduction: ArgmaxReduction,
    ) -> Result<(i32, TokenProfile), MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return if concurrent_gdn_moe_decode_enabled() {
                self.single_token_argmax_profiled_concurrent_gdn_moe_with_reduction(
                    token_id, position, session, reduction,
                )
            } else {
                self.single_token_argmax_profiled_moe(token_id, position, session, reduction)
            };
        }
        if concurrent_gdn_dense_decode_enabled() {
            return self.single_token_argmax_profiled_concurrent_gdn_dense_with_reduction(
                token_id, position, session, reduction,
            );
        }
        self.single_token_argmax_profiled_dense_serial(token_id, position, session, reduction)
    }

    pub fn encode_single_token_argmax(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
    ) -> Result<(), MfError> {
        self.encode_single_token_argmax_with_reduction(
            enc,
            position,
            session,
            ids_buf,
            argmax_tok,
            ArgmaxReduction::SpeculativeLowest,
        )
    }

    pub(crate) fn encode_single_token_greedy(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
    ) -> Result<(), MfError> {
        session.ensure_usable()?;
        self.encode_single_token_argmax_with_reduction(
            enc,
            position,
            session,
            ids_buf,
            argmax_tok,
            ArgmaxReduction::GreedyTotal,
        )
    }

    fn encode_single_token_argmax_with_reduction(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
        reduction: ArgmaxReduction,
    ) -> Result<(), MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return self.encode_single_token_argmax_moe_with_reduction(
                enc, position, session, ids_buf, argmax_tok, reduction,
            );
        }
        self.encode_single_token_argmax_dense_with_reduction(
            enc, position, session, ids_buf, argmax_tok, reduction,
        )
    }

    pub fn encode_single_token_argmax_dense(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
    ) -> Result<(), MfError> {
        self.encode_single_token_argmax_dense_with_reduction(
            enc,
            position,
            session,
            ids_buf,
            argmax_tok,
            ArgmaxReduction::SpeculativeLowest,
        )
    }

    fn encode_single_token_argmax_dense_with_reduction(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
        reduction: ArgmaxReduction,
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
        encode_argmax_reduction(
            self.ctx,
            enc,
            &session.logits,
            argmax_tok,
            1,
            arch.vocab_size as usize,
            reduction,
        )?;
        Ok(())
    }

    fn encode_single_token_argmax_moe_with_reduction(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
        reduction: ArgmaxReduction,
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
        encode_argmax_reduction(
            self.ctx,
            enc,
            &session.logits,
            argmax_tok,
            1,
            arch.vocab_size as usize,
            reduction,
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
        reduction: ArgmaxReduction,
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
        self.encode_single_token_argmax_moe_with_reduction(
            &enc,
            position,
            session,
            &ids_buf,
            &argmax_tok,
            reduction,
        )?;
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

    /// Same as [`single_token`] but ALSO copies the hidden state for the MTP
    /// carry into `hidden_dst`. By default this is the pre-output_norm residual;
    /// `post_norm_hidden` instead captures the final RMSNorm output.
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
        post_norm_hidden: bool,
    ) -> Result<i32, MfError> {
        let arch = &self.model.arch;
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

        if arch.kind == ArchKind::Moe && concurrent_gdn_moe_decode_enabled() {
            // D1 fix (2026-07-20, packets w0b/w0c-econ): this path must use
            // the same encoder organization as production A3B decode. Keep
            // that invariant structurally by sharing the production body.
            self.encode_single_token_concurrent_gdn_moe_body(&cmd_buf, position, session)?;
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
            let hidden_src = if post_norm_hidden {
                &session.h
            } else {
                &session.x
            };
            encode_scatter_offset_f32(self.ctx, &enc, hidden_src, hidden_dst, 0, h)?;
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
            return Ok(argmax);
        }

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
            if arch.kind == ArchKind::Moe {
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
            } else {
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
        let hidden_src = if post_norm_hidden {
            &session.h
        } else {
            &session.x
        };
        encode_scatter_offset_f32(self.ctx, &enc, hidden_src, hidden_dst, 0, h)?;
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

        if phase_lm_argmax_enabled() {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_argmax_f32(
                self.ctx,
                &enc,
                &session.logits,
                &session.argmax_tok,
                1,
                arch.vocab_size as usize,
            )?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            phases.push(("lm argmax".into(), ms));
        }

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((out, total_ms, phases))
    }

    /// Decode one MoE token and return per-layer post-norm hidden vectors plus
    /// routed expert ids/weights after each real route kernel. This is bench-only
    /// instrumentation for replaying realistic route patterns in isolated MoE
    /// microbenches.
    pub fn capture_moe_gateup_replay_for_token(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<Vec<MoeRouteReplayRow>, MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

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
        }

        let mut routes = Vec::new();
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
            let (ffn_gate, ffn_up, ffn_down, moe) = match block {
                MetalBlock::Gdn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
                MetalBlock::Attn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
            };
            let moe = moe.ok_or(MfError::UnsupportedMoe)?;

            {
                let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                self.encode_moe_mixer_prep(&enc, block, slot, position, session)?;
                enc.end();
                cmd.commit();
                cmd.waitUntilCompleted();
            }
            {
                let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                self.encode_moe_route_prepare(&enc, session, moe)?;
                enc.end();
                cmd.commit();
                cmd.waitUntilCompleted();
            }

            let route = self.read_moe_route_result(session, topk);
            let hidden = unsafe {
                let src = (session.h.buffer.contents().as_ptr() as *const f32)
                    .add((session.h.offset / 4) as usize);
                std::slice::from_raw_parts(src, h).to_vec()
            };
            let topk_idx = route
                .ranked
                .iter()
                .map(|&(expert, _)| expert as i32)
                .collect();
            let topk_weight = route.ranked.iter().map(|&(_, weight)| weight).collect();
            routes.push(MoeRouteReplayRow {
                topk_idx,
                topk_weight,
                hidden,
            });

            {
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
            }
        }
        Ok(routes)
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
        let mut gdn_tail_conv_total_ms = 0.0f64;
        let mut gdn_tail_l2_total_ms = 0.0f64;
        let mut gdn_tail_step_total_ms = 0.0f64;
        let mut gdn_tail_norm_total_ms = 0.0f64;
        let mut gdn_out_proj_total_ms = 0.0f64;
        let mut gdn_resid_post_total_ms = 0.0f64;
        let mut attn_mixer_total_ms = 0.0f64;
        let mut route_total_ms = 0.0f64;
        let mut route_logits_total_ms = 0.0f64;
        let mut route_topk_total_ms = 0.0f64;
        let mut route_shared_total_ms = 0.0f64;
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
        let split_gdn_tail = phase_gdn_tail_split_enabled();
        let route_replay = phase_moe_route_replay_enabled();
        let split_route_deep = !route_replay && phase_moe_route_deep_split_enabled();
        let split_route = !route_replay && (phase_moe_route_split_enabled() || split_route_deep);
        let cpu_route = phase_moe_cpu_route_enabled();
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
                            } else if gdn_beta_projection_fused(g) {
                                encode_mat_vec_f32_sigmoid(
                                    self.ctx,
                                    &enc,
                                    &g.beta_proj,
                                    &session.h,
                                    &session.gdn_beta,
                                    h,
                                    n_v,
                                )?;
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
                        if !gdn_beta_projection_fused(g) {
                            encode_sigmoid_f32(self.ctx, &enc, &session.gdn_b, &session.gdn_beta)?;
                        }
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
                        if split_gdn_tail {
                            let n_v = self.model.arch.gdn_n_v_heads as usize;
                            let n_k = self.model.arch.gdn_n_k_heads as usize;
                            let head_dim = self.model.arch.gdn_head_dim as usize;
                            let conv_dim = (2 * n_k + n_v) * head_dim;
                            let q_view = session
                                .gdn_qkv_conv
                                .view_subrange(0, vec![(n_k * head_dim) as u64]);
                            let k_view = session.gdn_qkv_conv.view_subrange(
                                (n_k * head_dim) as u64,
                                vec![(n_k * head_dim) as u64],
                            );
                            let v_view = session.gdn_qkv_conv.view_subrange(
                                (2 * n_k * head_dim) as u64,
                                vec![(n_v * head_dim) as u64],
                            );
                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            encode_ssm_conv_silu_f32(
                                self.ctx,
                                &enc,
                                &gdn_qkv,
                                &session.gdn_conv[gdn_i],
                                &g.conv1d,
                                &session.gdn_qkv_conv,
                                conv_dim,
                            )?;
                            enc.end();
                            cmd.commit();
                            cmd.waitUntilCompleted();
                            gdn_tail_conv_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            encode_l2_norm_batched_f32(
                                self.ctx,
                                &enc,
                                &q_view,
                                &session.gdn_q_norm,
                                n_k,
                                head_dim,
                                RMS_EPS,
                            )?;
                            encode_l2_norm_batched_f32(
                                self.ctx,
                                &enc,
                                &k_view,
                                &session.gdn_k_norm,
                                n_k,
                                head_dim,
                                RMS_EPS,
                            )?;
                            enc.end();
                            cmd.commit();
                            cmd.waitUntilCompleted();
                            gdn_tail_l2_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            encode_gdn_step_decay_f32(
                                self.ctx,
                                &enc,
                                &session.gdn_q_norm,
                                &session.gdn_k_norm,
                                &v_view,
                                &gdn_alpha,
                                &gdn_beta,
                                &session.gdn_state[gdn_i],
                                &session.gdn_out,
                                n_v,
                                n_k,
                                head_dim,
                            )?;
                            enc.end();
                            cmd.commit();
                            cmd.waitUntilCompleted();
                            gdn_tail_step_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            encode_rmsnorm_gated_f32(
                                self.ctx,
                                &enc,
                                &session.gdn_out,
                                &g.norm,
                                &gdn_z,
                                &gdn_normed,
                                n_v,
                                head_dim,
                                RMS_EPS * head_dim as f32,
                            )?;
                            enc.end();
                            cmd.commit();
                            cmd.waitUntilCompleted();
                            gdn_tail_norm_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        } else {
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
                if cpu_route {
                    let route = self.route_moe_block(&session.h, moe);
                    self.write_moe_route_result(session, &route);
                } else if route_replay {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_route_prepare(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                } else if split_route_deep {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_router_logits(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    route_logits_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_topk_parallel_from_logits(&enc, session)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    route_topk_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_shared_gate(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    route_shared_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                } else if split_route {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_router_logits(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    route_logits_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_topk_and_shared_from_logits(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    route_topk_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                } else {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_route_prepare(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    route_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                }
            }

            if split_ffn_apply {
                if deep_split_ffn_apply
                    && moe_routed_gate_up_decode_supported(moe.gate_exps.dtype, moe.up_exps.dtype)
                    && matches!(
                        moe.down_exps.dtype,
                        GgmlType::Q5_K
                            | GgmlType::Q6_K
                            | GgmlType::Q8_0
                            | GgmlType::IQ4_XS
                            | GgmlType::BF16
                    )
                {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_routed_gate_up_gpu(&enc, session, moe)?;
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
                        None,
                        None,
                    )?;
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_finalizer_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                } else if concurrent_shared_moe_decode_enabled()
                    && moe.gate_exps.dtype == GgmlType::Q4_K
                    && moe.up_exps.dtype == GgmlType::Q4_K
                {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let shared_inner_fused = self.encode_moe_ffn_gate_up_wave_gpu(
                        &cmd, session, ffn_gate, ffn_up, moe, None, None,
                    )?;
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
                    let routed_weighted_sum_is_pending = self
                        .encode_moe_ffn_down_wave_gpu(&cmd, session, ffn_down, moe, None, None)?;
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    ffn_down_wave_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    self.encode_moe_ffn_final_wave_gpu(
                        &cmd,
                        session,
                        routed_weighted_sum_is_pending,
                        None,
                        None,
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
        if split_gdn_tail {
            phases.push((
                format!("gdn tail conv (x{gdn_count})"),
                gdn_tail_conv_total_ms,
            ));
            phases.push((format!("gdn tail l2 (x{gdn_count})"), gdn_tail_l2_total_ms));
            phases.push((
                format!("gdn tail step (x{gdn_count})"),
                gdn_tail_step_total_ms,
            ));
            phases.push((
                format!("gdn tail norm (x{gdn_count})"),
                gdn_tail_norm_total_ms,
            ));
        } else {
            phases.push((format!("gdn tail (x{gdn_count})"), gdn_tail_total_ms));
        }
        phases.push((
            format!("gdn out_proj (x{gdn_count})"),
            gdn_out_proj_total_ms,
        ));
        phases.push((
            format!("gdn resid/post (x{gdn_count})"),
            gdn_resid_post_total_ms,
        ));
        phases.push((format!("attn mixer (x{attn_count})"), attn_mixer_total_ms));
        if route_replay {
            phases.push(("moe route replay (excluded)".into(), route_total_ms));
        } else if split_route {
            phases.push(("moe route logits".into(), route_logits_total_ms));
            if split_route_deep {
                phases.push(("moe route topk".into(), route_topk_total_ms));
                phases.push(("moe route shared gate".into(), route_shared_total_ms));
            } else {
                phases.push(("moe route topk/shared".into(), route_topk_total_ms));
            }
        } else {
            phases.push(("moe route".into(), route_total_ms));
        }
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

        if phase_lm_argmax_enabled() {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_argmax_f32(
                self.ctx,
                &enc,
                &session.logits,
                &session.argmax_tok,
                1,
                arch.vocab_size as usize,
            )?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            phases.push((
                "lm argmax".into(),
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

        // Stage the token id into the I32 ids buffer.
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
        session.ensure_usable()?;
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
        s.ensure_usable()?;
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

    /// Apply the active production residual/post-norm organization to a
    /// caller-provided mixer row. Batch executors use this seam so rollback
    /// flags cannot silently diverge from singleton decode.
    #[doc(hidden)]
    pub fn encode_post_mixer_norm(
        &self,
        enc: &KernelEncoder,
        residual: &MetalTensor,
        mixer_output: &MetalTensor,
        norm: &MetalTensor,
        normalized: &MetalTensor,
    ) -> Result<(), MfError> {
        if decode_fused_residual_rmsnorm_enabled() {
            encode_residual_rms_norm_mul_f32(
                self.ctx,
                enc,
                residual,
                mixer_output,
                norm,
                normalized,
                RMS_EPS,
            )?;
        } else {
            encode_add_inplace_f32(self.ctx, enc, residual, mixer_output)?;
            encode_rms_norm_mul_f32(self.ctx, enc, residual, norm, normalized, RMS_EPS)?;
        }
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

        // Pre-FFN norm.
        let post_norm = match block {
            MetalBlock::Gdn(g) => &g.post_attn_norm,
            MetalBlock::Attn(a) => &a.post_attn_norm,
        };
        self.encode_post_mixer_norm(enc, &s.x, &s.mixer_out, post_norm, &s.h)?;

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
        // T9 bench-only capture (no-op unless installed by the capture
        // harness; see t9_ffn_capture_install).
        if let Some((h_dst, inner_dst)) = t9_ffn_capture_slots_for_current_call() {
            encode_scatter_offset_f32(self.ctx, enc, &s.h, &h_dst, 0, h)?;
            encode_scatter_offset_f32(self.ctx, enc, &s.ffn_inner, &inner_dst, 0, f)?;
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
        } else if gdn_beta_projection_fused(gb) {
            encode_mat_vec_f32_sigmoid(self.ctx, enc, &gb.beta_proj, &s.h, &s.gdn_beta, h, n_v)?;
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

        if !gdn_beta_projection_fused(gb) {
            encode_sigmoid_f32(self.ctx, enc, &s.gdn_b, &s.gdn_beta)?;
        }
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

    /// Complete the stateful GDN body after a batch executor has produced
    /// QKV and Z rows. Small alpha/beta projections remain sequence-private;
    /// the caller may batch the final output projection from `normed_output`.
    #[doc(hidden)]
    pub fn encode_gdn_after_batched_front(
        &self,
        enc: &KernelEncoder,
        gb: &MetalGdnBlock,
        gdn_i: usize,
        session: &mut MetalSession,
        qkv: &MetalTensor,
        z: &MetalTensor,
        normed_output: &MetalTensor,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        let n_v = self.model.arch.gdn_n_v_heads as usize;
        if decode_gdn_noop_beta_enabled() {
            encode_fill_f32(self.ctx, enc, &session.gdn_b, 0.0)?;
        } else if gdn_beta_projection_fused(gb) {
            encode_mat_vec_f32_sigmoid(
                self.ctx,
                enc,
                &gb.beta_proj,
                &session.h,
                &session.gdn_beta,
                h,
                n_v,
            )?;
        } else {
            encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &gb.beta_proj,
                &session.h,
                &session.gdn_b,
                h,
                n_v,
            )?;
        }
        if decode_gdn_noop_alpha_enabled() {
            encode_fill_f32(self.ctx, enc, &session.gdn_a, 0.0)?;
        } else {
            encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &gb.alpha_proj,
                &session.h,
                &session.gdn_a,
                h,
                n_v,
            )?;
        }
        if !gdn_beta_projection_fused(gb) {
            encode_sigmoid_f32(self.ctx, enc, &session.gdn_b, &session.gdn_beta)?;
        }
        encode_gdn_decay_chain_f32(
            self.ctx,
            enc,
            &session.gdn_a,
            &gb.dt_bias,
            &gb.a_log,
            &session.gdn_alpha,
        )?;
        let alpha = session.gdn_alpha.clone();
        let beta = session.gdn_beta.clone();
        self.encode_gdn_tail(
            enc,
            gb,
            gdn_i,
            session,
            qkv,
            z,
            &alpha,
            &beta,
            normed_output,
        )?;
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

    /// Complete one attention mixer from already-populated Q/K/V projection
    /// buffers. Exposed for diagnostics that batch only the immutable-weight
    /// front projections while retaining sequence-private KV state.
    #[doc(hidden)]
    pub fn encode_attn_after_projections(
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

        // v0.432: the default path reads the interleaved q_proj output
        // ([head_dim Q, head_dim gate] per head) directly — Q halves via
        // the strided q-norm here, gate halves via the strided sigmoid_mul
        // at the attention epilogue — deleting the split_q_gate layout
        // copy (one dispatch + a 2*q_dim round-trip per attn layer). The
        // QWEN_DECODE_ATTN_SIGMOID_MUL=0 rollback branch still consumes a
        // compact attn_gate, so it keeps the split.
        if decode_attn_sigmoid_mul_enabled() {
            encode_rms_norm_batched_src_strided_f32(
                self.ctx,
                enc,
                &s.attn_q_full,
                &ab.q_norm,
                &s.attn_q_normed,
                n_q,
                head_dim,
                2 * head_dim,
                0,
                RMS_EPS,
            )?;
        } else {
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
        }
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
        if decode_rope_pair_enabled() {
            encode_rope_neox_pair_f32(
                self.ctx,
                enc,
                &s.attn_q_normed,
                &s.attn_k_normed,
                n_q,
                n_kv,
                head_dim,
                n_rot,
                position,
                arch.rope_theta,
            )?;
        } else {
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
        }
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
            // Strided gate read from the interleaved q_proj output (see the
            // v0.432 comment at the q-norm above).
            encode_sigmoid_mul_gate_strided_f32(
                self.ctx,
                enc,
                &s.attn_q_full,
                &s.attn_o,
                &s.attn_o,
                n_q,
                head_dim,
                2 * head_dim,
                head_dim,
            )?;
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
        let fuse_beta_proj = gdn_beta_projection_fused(gb);
        if decode_gdn_noop_beta_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_b, 0.0)?;
        } else if fuse_beta_proj {
            encode_mat_vec_f32_sigmoid(self.ctx, enc, &gb.beta_proj, &s.h, &s.gdn_beta, h, n_v)?;
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
        if !fuse_beta_proj {
            encode_sigmoid_f32(self.ctx, enc, &s.gdn_b, &s.gdn_beta)?;
        }
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

        if decode_gdn_pair_l2_enabled() {
            encode_l2_norm_pair_batched_f32(
                self.ctx,
                enc,
                &q_view,
                &s.gdn_q_norm,
                &k_view,
                &s.gdn_k_norm,
                n_k,
                head_dim,
                RMS_EPS,
            )?;
        } else {
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
        }

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
            RMS_EPS * head_dim as f32,
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
        if decode_gdn_pair_l2_enabled() {
            encode_l2_norm_pair_batched_f32(
                self.ctx,
                enc,
                &q_view,
                &s.gdn_q_norm,
                &k_view,
                &s.gdn_k_norm,
                n_k,
                head_dim,
                RMS_EPS,
            )?;
        } else {
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
        }
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
            RMS_EPS * head_dim as f32,
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

        // (2)+(3) Q-norm reading the interleaved q_proj output directly
        // (v0.432: replaces split_q_gate + compact q-norm on the default
        // path; the QWEN_DECODE_ATTN_SIGMOID_MUL=0 rollback keeps the
        // split because its gate consumer needs a compact attn_gate).
        if decode_attn_sigmoid_mul_enabled() {
            encode_rms_norm_batched_src_strided_f32(
                self.ctx,
                enc,
                &s.attn_q_full,
                &ab.q_norm,
                &s.attn_q_normed,
                n_q,
                head_dim,
                2 * head_dim,
                0,
                RMS_EPS,
            )?;
        } else {
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
        }

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
        if decode_rope_pair_enabled() {
            encode_rope_neox_pair_f32(
                self.ctx,
                enc,
                &s.attn_q_normed,
                &s.attn_k_normed,
                n_q,
                n_kv,
                head_dim,
                n_rot,
                position,
                arch.rope_theta,
            )?;
        } else {
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
        }

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
            encode_sigmoid_mul_gate_strided_f32(
                self.ctx,
                enc,
                &s.attn_q_full,
                &s.attn_o,
                &s.attn_o,
                n_q,
                head_dim,
                2 * head_dim,
                head_dim,
            )?;
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
/// Copy `n_rows` contiguous F32 source rows of `row_len` elements into
/// `dst` rows at `dst_base + row * dst_stride` (v0.432). One dispatch
/// replaces a per-row `encode_scatter_offset_f32` loop in the DFlash
/// prefill hidden-capture tap (chunk_p dispatches per capture layer per
/// chunk -> 1).
pub fn encode_copy_rows_dst_strided_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    n_rows: usize,
    row_len: usize,
    dst_stride: usize,
    dst_base: usize,
) -> Result<(), MetalError> {
    if n_rows == 0 || row_len == 0 {
        return Err(MetalError::BadShape {
            kernel: "copy_rows_dst_strided",
            detail: "n_rows/row_len must be nonzero".to_string(),
        });
    }
    let total = (n_rows * row_len) as u64;
    if src.n_elements() < total {
        return Err(MetalError::BadShape {
            kernel: "copy_rows_dst_strided",
            detail: format!("src has {} elements, needs >= {total}", src.n_elements()),
        });
    }
    let dst_need = dst_base as u64 + (n_rows as u64 - 1) * dst_stride as u64 + row_len as u64;
    if dst.n_elements() < dst_need {
        return Err(MetalError::BadShape {
            kernel: "copy_rows_dst_strided",
            detail: format!("dst has {} elements, needs >= {dst_need}", dst.n_elements()),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_rows: u32,
        row_len: u32,
        dst_stride: u32,
        dst_base: u32,
    }
    let pso = ctx.pipeline("kernel_copy_rows_dst_strided_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_rows: n_rows as u32,
            row_len: row_len as u32,
            dst_stride: dst_stride as u32,
            dst_base: dst_base as u32,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, dst);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        objc2_metal::MTLSize {
            width: (total as usize).div_ceil(tg_threads),
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

#[derive(Debug, Clone, Copy)]
pub struct LogitsReadbackProfile {
    pub timer_spans: u32,
    pub bytes: usize,
    pub allocation_zero_fill_ms: f64,
    pub copy_ms: f64,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct StructuralRowEvidence {
    pub resident_head_wait_calls: u64,
    pub validated_shared_row_calls: u64,
    pub transition_logits_copy_bytes: u64,
    pub extra_command_buffers: u64,
    pub gpu_sampling_dispatches: u64,
}

pub type SampledStructuralOutcome = (
    Result<(SampledToken, BoundedTopKEvidence), SamplingError>,
    TokenProfile,
    StructuralRowEvidence,
);

struct ValidatedSharedLogits {
    source: std::ptr::NonNull<f32>,
    len: usize,
}

#[derive(Debug, Clone)]
pub struct DecodeStageTiming {
    pub family: String,
    pub block_kind: String,
    pub block_index: Option<usize>,
    pub local_index: Option<usize>,
    pub concurrent: bool,
    pub start_sample: usize,
    pub end_sample: usize,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub duration_ticks: u64,
    pub duration_ms_scaled: f64,
    pub fraction_of_gpu: f64,
}

#[derive(Debug, Clone)]
pub struct DecodeStageProfile {
    pub token: TokenProfile,
    pub stages: Vec<DecodeStageTiming>,
    pub sampled_span_ticks: u64,
    pub raw_span_ms_assuming_ns: f64,
    pub raw_coverage_assuming_ns: f64,
}

#[derive(Clone, Copy)]
struct DecodeStageMeta {
    family: &'static str,
    block_kind: &'static str,
    block_index: Option<usize>,
    local_index: Option<usize>,
}

struct DecodeStageRecord {
    meta: DecodeStageMeta,
    concurrent: bool,
    start_sample: usize,
    end_sample: usize,
}

struct DecodeStageRecorder {
    samples: MetalTimestampSampleBuffer,
    next_sample: usize,
    records: Vec<DecodeStageRecord>,
}

impl DecodeStageRecorder {
    fn new(ctx: &MetalContext, sample_count: usize) -> Result<Self, MfError> {
        Ok(Self {
            samples: ctx.timestamp_sample_buffer(sample_count)?,
            next_sample: 0,
            records: Vec::new(),
        })
    }

    fn begin(
        &mut self,
        cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        meta: DecodeStageMeta,
        concurrent: bool,
    ) -> Result<KernelEncoder, MfError> {
        let start_sample = self.next_sample;
        let end_sample = start_sample + 1;
        if end_sample >= self.samples.sample_count() {
            return Err(MfError::Metal(MetalError::Counter(format!(
                "decode stage timestamp buffer exhausted at sample {end_sample}"
            ))));
        }
        self.next_sample += 2;
        self.records.push(DecodeStageRecord {
            meta,
            concurrent,
            start_sample,
            end_sample,
        });
        Ok(KernelEncoder::begin_sampled(
            cmd,
            &self.samples,
            start_sample,
            end_sample,
            concurrent,
        ))
    }

    fn resolve(
        self,
        ctx: &MetalContext,
        token: TokenProfile,
    ) -> Result<DecodeStageProfile, MfError> {
        let timestamps = ctx.resolve_timestamp_samples(&self.samples, self.next_sample)?;
        let sampled_span_ticks = match (self.records.first(), self.records.last()) {
            (Some(first), Some(last)) => {
                timestamps[last.end_sample].saturating_sub(timestamps[first.start_sample])
            }
            _ => 0,
        };
        let scale_ms_per_tick = if sampled_span_ticks > 0 {
            token.gpu_kernel_ms / sampled_span_ticks as f64
        } else {
            0.0
        };
        let stages = self
            .records
            .into_iter()
            .map(|record| {
                let start_timestamp = timestamps[record.start_sample];
                let end_timestamp = timestamps[record.end_sample];
                let duration_ticks = end_timestamp.saturating_sub(start_timestamp);
                let duration_ms_scaled = duration_ticks as f64 * scale_ms_per_tick;
                let fraction_of_gpu = if token.gpu_kernel_ms > 0.0 {
                    duration_ms_scaled / token.gpu_kernel_ms
                } else {
                    0.0
                };
                DecodeStageTiming {
                    family: record.meta.family.to_string(),
                    block_kind: record.meta.block_kind.to_string(),
                    block_index: record.meta.block_index,
                    local_index: record.meta.local_index,
                    concurrent: record.concurrent,
                    start_sample: record.start_sample,
                    end_sample: record.end_sample,
                    start_timestamp,
                    end_timestamp,
                    duration_ticks,
                    duration_ms_scaled,
                    fraction_of_gpu,
                }
            })
            .collect();
        let raw_span_ms_assuming_ns = sampled_span_ticks as f64 * 1e-6;
        let raw_coverage_assuming_ns = if token.gpu_kernel_ms > 0.0 {
            raw_span_ms_assuming_ns / token.gpu_kernel_ms
        } else {
            0.0
        };
        Ok(DecodeStageProfile {
            token,
            stages,
            sampled_span_ticks,
            raw_span_ms_assuming_ns,
            raw_coverage_assuming_ns,
        })
    }
}

fn begin_decode_stage(
    cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    recorder: Option<&mut DecodeStageRecorder>,
    meta: Option<DecodeStageMeta>,
    concurrent: bool,
) -> Result<KernelEncoder, MfError> {
    if let Some(meta) = &meta {
        // bench-only dispatch census label; no-op unless a census is active
        qwen_llm_dispatch_census_set_family(meta.family);
    }
    if let (Some(recorder), Some(meta)) = (recorder, meta) {
        recorder.begin(cmd, meta, concurrent)
    } else if concurrent {
        Ok(KernelEncoder::begin_concurrent(cmd))
    } else {
        Ok(KernelEncoder::begin(cmd))
    }
}

use crate::metal::dispatch_census_set_family as qwen_llm_dispatch_census_set_family;

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
    // Debug-only concurrent-pass hazard tracking (no-op on serial encoders
    // and in release builds). This is the chokepoint for the concurrent
    // GDN/attention front-projection encoders; a future edit that makes one
    // projection consume another's output inside the same Concurrent pass
    // will panic here instead of racing on the GPU.
    enc.note_read(weight);
    enc.note_read(x);
    enc.note_write(y);
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
        GgmlType::IQ2_XS => Ok(crate::metal::encode_mat_vec_iq2_xs_f32(
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
        GgmlType::MXFP4 => Ok(crate::metal::encode_mat_vec_mxfp4_f32(
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

crate::env_flag!(default_on matmat_smalln_table_enabled, "QWEN_MATMAT_SMALLN_TABLE");

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
    // v0.501: small-N best-kernel table from the v0.500 selection sweep
    // (PERF-LOG v0.500; the pre-v0.501 selection fell through to the
    // GENERIC 32-wide tile at every N < 16 and lost 2-5x). Only fires
    // where the sweep measured a clear win AND the buffer contract is
    // drop-in (x = [n_query, n_in], y = [n_query, n_out], exact).
    // Exactness: nc is E0 per column vs mat-vec (tighter than the E1
    // half-staged tile it replaces); mma8v is E1 with F32 activations
    // (cos 1.000000 at rel_rms ~2e-4 on all swept shapes). End-to-end
    // greedy equivalence re-gated at v0.501. Rollback:
    // QWEN_MATMAT_SMALLN_TABLE=0.
    if matmat_smalln_table_enabled()
        && matches!(weight.dtype, GgmlType::Q4_K | GgmlType::Q6_K)
        && n_in.is_multiple_of(256)
    {
        match n_query {
            2 if weight.dtype == GgmlType::Q4_K => {
                // nc2rp4: c(2) 1.53-1.72 vs generic 5.1-8.0.
                return Ok(crate::metal::encode_mat_vec_q4_k_nc2_rp4_f32(
                    ctx, enc, weight, x, y, n_in, n_out,
                )?);
            }
            2 | 4 => {
                // nc2 (Q6_K) / nc4: c 1.6-3.3 vs generic 5.1-8.3.
                return Ok(crate::metal::encode_mat_vec_nc_dispatch(
                    ctx, enc, weight, x, y, n_in, n_out, n_query,
                )?);
            }
            8 if weight.dtype == GgmlType::Q6_K && n_out.is_multiple_of(8) => {
                // r1c1k128: flat c ~1.6-1.8 across N on Q6_K shapes.
                return Ok(crate::metal::encode_mat_mat_mma8_variant(
                    ctx, enc, weight, x, y, n_in, n_out, "r1c1k128",
                )?);
            }
            8 if weight.dtype == GgmlType::Q4_K && n_out.is_multiple_of(16) => {
                // r1c1k64_sg2: the Q4_K N=8 all-rounder (2.25-2.78,
                // never worst) vs generic 5.1-8.3.
                return Ok(crate::metal::encode_mat_mat_mma8_variant(
                    ctx,
                    enc,
                    weight,
                    x,
                    y,
                    n_in,
                    n_out,
                    "r1c1k64_sg2",
                )?);
            }
            16 if n_out.is_multiple_of(16) && n_out < 100_000 => {
                // r2c2k64 beats n16 by 8-35% on ffn/gdn/attn shapes;
                // n16 retained for lm_head-class (n_out >= 100k) where
                // it still wins (2.35 vs 2.81).
                return Ok(crate::metal::encode_mat_mat_mma8_variant(
                    ctx, enc, weight, x, y, n_in, n_out, "r2c2k64",
                )?);
            }
            _ => {}
        }
    }
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
        GgmlType::BF16
            if matmat_bf16_bfloat_act_enabled() && n_in.is_multiple_of(32) && n_query >= 16 =>
        {
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
        GgmlType::IQ2_XS => Ok(crate::metal::encode_mat_mat_iq2_xs_f32(
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
//     a checked `SessionSnapshot` by reading the live MTLBuffer.contents() of
//     each session field via raw memcpy. Shared-storage UMA makes this
//     safe and fast (no command-buffer round-trip); Apple docs
//     guarantee the producer's writes are visible after that command
//     buffer completes.
//   * `MetalSession::restore_from(snap, identity)` validates identity,
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
/// `model_id` is a caller-supplied compatibility fingerprint (typically a hash
/// of GGUF metadata + tensor descriptors for in-process caches). Persistent or
/// cross-process caches need a stronger content identity. `layout_version` is a
/// manual counter bumped whenever the `MetalSession` field layout changes in a
/// way that would invalidate prior snapshots.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SnapshotIdentity {
    pub model_id: u64,
    pub tokenizer_id: u64,
    pub layout_version: u32,
    pub n_attn_layers: u32,
    pub n_gdn_layers: u32,
    pub kv_dim_elements: u32,
    pub kv_bytes_per_token: u32,
    pub kv_storage_kind: SnapshotKvStorageKind,
    pub gdn_state_elements_per_layer: u32,
    pub gdn_conv_elements_per_layer: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SnapshotAbi {
    pub layout_version: u32,
    pub n_attn_layers: u32,
    pub n_gdn_layers: u32,
    pub kv_dim_elements: u32,
    pub kv_bytes_per_token: u32,
    pub kv_storage_kind: SnapshotKvStorageKind,
    pub gdn_state_elements_per_layer: u32,
    pub gdn_conv_elements_per_layer: u32,
}

impl SnapshotIdentity {
    pub fn abi(&self) -> SnapshotAbi {
        SnapshotAbi {
            layout_version: self.layout_version,
            n_attn_layers: self.n_attn_layers,
            n_gdn_layers: self.n_gdn_layers,
            kv_dim_elements: self.kv_dim_elements,
            kv_bytes_per_token: self.kv_bytes_per_token,
            kv_storage_kind: self.kv_storage_kind,
            gdn_state_elements_per_layer: self.gdn_state_elements_per_layer,
            gdn_conv_elements_per_layer: self.gdn_conv_elements_per_layer,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum SnapshotKvStorageKind {
    None = 0,
    F16 = 1,
    Q8_0 = 2,
}

/// Bump this when MetalSession's per-layer state shape changes.
pub const SNAPSHOT_LAYOUT_VERSION: u32 = 4;

/// Captured state at the end of prefilling `prefix_tokens` through a
/// fresh session. Restoring into a fresh session and running additional
/// tokens is bit-equivalent to cold prefill of the full sequence
/// (validated by `h2_prefix_cache_correctness_spike`).
#[derive(Clone, Debug)]
pub struct SessionSnapshot {
    pub identity: SnapshotIdentity,
    /// Tokens consumed into the captured KV/GDN state.
    pub prefix_tokens: Vec<i32>,
    /// A terminal token selected and emitted from `final_logits`, but not yet
    /// consumed into model state. Prefix matching includes this token; restore
    /// resumes from it so a completed request never pays an unused transition.
    pub pending_token: Option<i32>,
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

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum SnapshotValidationError {
    #[error("snapshot identity mismatch: expected {expected:?}, got {actual:?}")]
    IdentityMismatch {
        expected: SnapshotIdentity,
        actual: SnapshotIdentity,
    },
    #[error("snapshot prefix length {prefix_len} exceeds capacity {capacity}")]
    PrefixCapacity { prefix_len: usize, capacity: usize },
    #[error("snapshot {section} length arithmetic overflow")]
    LengthOverflow { section: &'static str },
    #[error("snapshot {section} length {actual} != expected {expected}")]
    SectionLength {
        section: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("snapshot {section} allocation of {bytes} bytes failed")]
    AllocationFailed { section: &'static str, bytes: usize },
    #[error(
        "snapshot {section} tensor at layer {layer} has {available} bytes available, needs {required}"
    )]
    TensorBounds {
        section: &'static str,
        layer: usize,
        available: u64,
        required: usize,
    },
    #[error("snapshot KV position at layer {layer} is {actual} != prefix length {expected}")]
    KvPosition {
        layer: usize,
        actual: usize,
        expected: usize,
    },
    #[error("snapshot token {token} at index {index} is outside vocabulary size {vocab_size}")]
    TokenOutOfRange {
        index: usize,
        token: i32,
        vocab_size: usize,
    },
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
            + self.pending_token.map_or(0, |_| 4)
            + self.kv_n_pos.len() * 8) as u64
    }

    /// Number of tokens consumed up to this snapshot.
    pub fn prefix_len(&self) -> usize {
        self.prefix_tokens.len()
    }

    /// Number of canonical prefix tokens represented by this checkpoint.
    pub fn matched_prefix_len(&self) -> usize {
        self.prefix_len() + usize::from(self.pending_token.is_some())
    }

    /// Validate every shape and offset premise used by restore before any
    /// session buffer is mutated. Persistent readers must call this after
    /// decoding their bounded sections and before handing the snapshot to
    /// Metal; the production restore path also calls it defensively.
    pub fn validate_for_restore(
        &self,
        expected_identity: &SnapshotIdentity,
        max_context_tokens: usize,
        expected_vocab_size: Option<usize>,
    ) -> Result<(), SnapshotValidationError> {
        if &self.identity != expected_identity {
            return Err(SnapshotValidationError::IdentityMismatch {
                expected: expected_identity.clone(),
                actual: self.identity.clone(),
            });
        }

        let prefix_len = self.prefix_len();
        if prefix_len > max_context_tokens {
            return Err(SnapshotValidationError::PrefixCapacity {
                prefix_len,
                capacity: max_context_tokens,
            });
        }

        let n_attn = self.identity.n_attn_layers as usize;
        let n_gdn = self.identity.n_gdn_layers as usize;
        require_snapshot_len("kv_n_pos", self.kv_n_pos.len(), n_attn)?;
        for (layer, &actual) in self.kv_n_pos.iter().enumerate() {
            if actual != prefix_len {
                return Err(SnapshotValidationError::KvPosition {
                    layer,
                    actual,
                    expected: prefix_len,
                });
            }
        }

        let kv_per_layer = checked_snapshot_product(
            "kv_arena",
            &[prefix_len, self.identity.kv_bytes_per_token as usize],
        )?;
        let kv_total = checked_snapshot_product("kv_arena", &[n_attn, kv_per_layer])?;
        require_snapshot_len("kv_k_arena", self.kv_k_arena.len(), kv_total)?;
        require_snapshot_len("kv_v_arena", self.kv_v_arena.len(), kv_total)?;

        let gdn_conv_total = checked_snapshot_product(
            "gdn_conv_arena",
            &[
                n_gdn,
                self.identity.gdn_conv_elements_per_layer as usize,
                std::mem::size_of::<f32>(),
            ],
        )?;
        require_snapshot_len("gdn_conv_arena", self.gdn_conv_arena.len(), gdn_conv_total)?;
        let gdn_state_total = checked_snapshot_product(
            "gdn_state_arena",
            &[
                n_gdn,
                self.identity.gdn_state_elements_per_layer as usize,
                std::mem::size_of::<f32>(),
            ],
        )?;
        require_snapshot_len(
            "gdn_state_arena",
            self.gdn_state_arena.len(),
            gdn_state_total,
        )?;

        if let Some(vocab_size) = expected_vocab_size {
            for (index, &token) in self.prefix_tokens.iter().enumerate() {
                if token < 0 || token as usize >= vocab_size {
                    return Err(SnapshotValidationError::TokenOutOfRange {
                        index,
                        token,
                        vocab_size,
                    });
                }
            }
            if let Some(token) = self.pending_token
                && (token < 0 || token as usize >= vocab_size)
            {
                return Err(SnapshotValidationError::TokenOutOfRange {
                    index: prefix_len,
                    token,
                    vocab_size,
                });
            }
            if let Some(logits) = self.final_logits.as_ref() {
                require_snapshot_len("final_logits", logits.len(), vocab_size)?;
            }
        }
        Ok(())
    }
}

fn checked_snapshot_product(
    section: &'static str,
    factors: &[usize],
) -> Result<usize, SnapshotValidationError> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or(SnapshotValidationError::LengthOverflow { section })
    })
}

fn require_snapshot_len(
    section: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), SnapshotValidationError> {
    if actual != expected {
        return Err(SnapshotValidationError::SectionLength {
            section,
            actual,
            expected,
        });
    }
    Ok(())
}

fn allocate_snapshot_arena(
    section: &'static str,
    bytes: usize,
) -> Result<Vec<u8>, SnapshotValidationError> {
    let mut arena = Vec::new();
    arena
        .try_reserve_exact(bytes)
        .map_err(|_| SnapshotValidationError::AllocationFailed { section, bytes })?;
    // SAFETY: the caller writes every byte before exposing the arena. u8 has
    // no validity invariant and no destructor.
    unsafe {
        arena.set_len(bytes);
    }
    Ok(arena)
}

fn validate_snapshot_tensor_span(
    section: &'static str,
    layer: usize,
    tensor: &MetalTensor,
    required: usize,
) -> Result<(), SnapshotValidationError> {
    let available = (tensor.buffer.length() as u64).saturating_sub(tensor.offset);
    if tensor.offset > tensor.buffer.length() as u64 || required as u64 > available {
        return Err(SnapshotValidationError::TensorBounds {
            section,
            layer,
            available,
            required,
        });
    }
    Ok(())
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
    pub fn snapshot_abi(&self) -> SnapshotAbi {
        SnapshotAbi {
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
            kv_storage_kind: match self.kv_k.first().map(|tensor| tensor.dtype) {
                None => SnapshotKvStorageKind::None,
                Some(GgmlType::F16) => SnapshotKvStorageKind::F16,
                Some(GgmlType::Q8_0) => SnapshotKvStorageKind::Q8_0,
                Some(_) => unreachable!("unsupported snapshot KV storage"),
            },
            gdn_state_elements_per_layer: self
                .gdn_state
                .first()
                .map(|t| t.n_elements())
                .unwrap_or(0) as u32,
            gdn_conv_elements_per_layer: self.gdn_conv.first().map(|t| t.n_elements()).unwrap_or(0)
                as u32,
        }
    }

    /// Compute the identity tag for snapshots produced by this session
    /// shape under the given model. Stable across runs of the same
    /// (model, tokenizer, layout) tuple.
    pub fn snapshot_identity(&self, model_id: u64, tokenizer_id: u64) -> SnapshotIdentity {
        let abi = self.snapshot_abi();
        SnapshotIdentity {
            model_id,
            tokenizer_id,
            layout_version: abi.layout_version,
            n_attn_layers: abi.n_attn_layers,
            n_gdn_layers: abi.n_gdn_layers,
            kv_dim_elements: abi.kv_dim_elements,
            kv_bytes_per_token: abi.kv_bytes_per_token,
            kv_storage_kind: abi.kv_storage_kind,
            gdn_state_elements_per_layer: abi.gdn_state_elements_per_layer,
            gdn_conv_elements_per_layer: abi.gdn_conv_elements_per_layer,
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
    ) -> Result<SessionSnapshot, MfError> {
        self.ensure_usable()?;
        let prefix_len = prefix_tokens.len();
        let n_attn = self.kv_k.len();
        let n_gdn = self.gdn_state.len();
        require_snapshot_len("kv_v_layers", self.kv_v.len(), n_attn)?;
        require_snapshot_len("gdn_conv_layers", self.gdn_conv.len(), n_gdn)?;
        let expected_identity = self.snapshot_identity(identity.model_id, identity.tokenizer_id);
        let vocab_size = usize::try_from(self.logits.n_elements()).map_err(|_| {
            SnapshotValidationError::LengthOverflow {
                section: "final_logits",
            }
        })?;
        if identity != expected_identity {
            return Err(SnapshotValidationError::IdentityMismatch {
                expected: expected_identity,
                actual: identity,
            }
            .into());
        }
        if prefix_len > self.kv_capacity {
            return Err(SnapshotValidationError::PrefixCapacity {
                prefix_len,
                capacity: self.kv_capacity,
            }
            .into());
        }
        require_snapshot_len("kv_n_pos", self.kv_n_pos.len(), n_attn)?;
        for (layer, &actual) in self.kv_n_pos.iter().enumerate() {
            if actual != prefix_len {
                return Err(SnapshotValidationError::KvPosition {
                    layer,
                    actual,
                    expected: prefix_len,
                }
                .into());
            }
        }
        for (index, &token) in prefix_tokens.iter().enumerate() {
            if token < 0 || token as usize >= vocab_size {
                return Err(SnapshotValidationError::TokenOutOfRange {
                    index,
                    token,
                    vocab_size,
                }
                .into());
            }
        }
        if let Some(logits) = final_logits.as_ref() {
            require_snapshot_len("final_logits", logits.len(), vocab_size)?;
        }
        // KV: per-layer slice is exactly prefix_len rows of the active KV dtype.
        let kv_slice_bytes = checked_snapshot_product(
            "kv_arena",
            &[prefix_len, identity.kv_bytes_per_token as usize],
        )?;
        let kv_arena_bytes = checked_snapshot_product("kv_arena", &[n_attn, kv_slice_bytes])?;
        for (layer, (k, v)) in self.kv_k.iter().zip(&self.kv_v).enumerate() {
            validate_snapshot_tensor_span("kv_k", layer, k, kv_slice_bytes)?;
            validate_snapshot_tensor_span("kv_v", layer, v, kv_slice_bytes)?;
        }

        // Allocate each arena UNINITIALIZED (no zero-init), then memcpy
        // directly from MTLBuffer.contents() into slices. ONE write per byte.
        // Zero-init costs almost as much as the actual copy at this scale
        // (158 MB of writes), so skipping it ~halves wall time.
        // Safety: the entire allocation is overwritten by read_tensor_into
        // before any read; no uninitialized bytes ever escape.
        let mut kv_k_arena = allocate_snapshot_arena("kv_k_arena", kv_arena_bytes)?;
        let mut kv_v_arena = allocate_snapshot_arena("kv_v_arena", kv_arena_bytes)?;
        for i in 0..n_attn {
            let off = i * kv_slice_bytes;
            read_tensor_into(&mut kv_k_arena[off..off + kv_slice_bytes], &self.kv_k[i]);
            read_tensor_into(&mut kv_v_arena[off..off + kv_slice_bytes], &self.kv_v[i]);
        }

        // GDN: each layer's full buffer (size doesn't depend on prefix_len).
        let gdn_conv_per = checked_snapshot_product(
            "gdn_conv_arena",
            &[
                identity.gdn_conv_elements_per_layer as usize,
                std::mem::size_of::<f32>(),
            ],
        )?;
        let gdn_state_per = checked_snapshot_product(
            "gdn_state_arena",
            &[
                identity.gdn_state_elements_per_layer as usize,
                std::mem::size_of::<f32>(),
            ],
        )?;
        let gdn_conv_total = checked_snapshot_product("gdn_conv_arena", &[n_gdn, gdn_conv_per])?;
        let gdn_state_total = checked_snapshot_product("gdn_state_arena", &[n_gdn, gdn_state_per])?;
        for (layer, (conv, state)) in self.gdn_conv.iter().zip(&self.gdn_state).enumerate() {
            validate_snapshot_tensor_span("gdn_conv", layer, conv, gdn_conv_per)?;
            validate_snapshot_tensor_span("gdn_state", layer, state, gdn_state_per)?;
        }
        let mut gdn_conv_arena = allocate_snapshot_arena("gdn_conv_arena", gdn_conv_total)?;
        let mut gdn_state_arena = allocate_snapshot_arena("gdn_state_arena", gdn_state_total)?;
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

        Ok(SessionSnapshot {
            identity,
            prefix_tokens,
            pending_token: None,
            kv_n_pos: self.kv_n_pos.clone(),
            kv_k_arena,
            kv_v_arena,
            gdn_conv_arena,
            gdn_state_arena,
            final_logits,
        })
    }

    /// Restore a session to the state captured in `snap`. The session
    /// MUST have been freshly created with the same architecture as
    /// the one that produced `snap` (validated by identity check).
    /// Returns Err on identity mismatch (to avoid silent corruption).
    ///
    /// The caller must ensure no in-flight GPU work is reading these
    /// session buffers (i.e., this should be called after
    /// `MetalSession::fresh` and before the first `single_token`).
    pub fn restore_from(
        &mut self,
        snap: &SessionSnapshot,
        expected_identity: &SnapshotIdentity,
    ) -> Result<(), MfError> {
        let want =
            self.snapshot_identity(expected_identity.model_id, expected_identity.tokenizer_id);
        if &want != expected_identity {
            return Err(SnapshotValidationError::IdentityMismatch {
                expected: want,
                actual: expected_identity.clone(),
            }
            .into());
        }
        let vocab_size = usize::try_from(self.logits.n_elements()).map_err(|_| {
            SnapshotValidationError::LengthOverflow {
                section: "final_logits",
            }
        })?;
        snap.validate_for_restore(expected_identity, self.kv_capacity, Some(vocab_size))?;
        let n_attn = self.kv_k.len();
        let n_gdn = self.gdn_state.len();
        require_snapshot_len("kv_v_layers", self.kv_v.len(), n_attn)?;
        require_snapshot_len("gdn_conv_layers", self.gdn_conv.len(), n_gdn)?;
        let prefix_len = snap.prefix_len();
        let kv_slice_bytes = prefix_len * snap.identity.kv_bytes_per_token as usize;

        for (layer, (k, v)) in self.kv_k.iter().zip(&self.kv_v).enumerate() {
            validate_snapshot_tensor_span("kv_k", layer, k, kv_slice_bytes)?;
            validate_snapshot_tensor_span("kv_v", layer, v, kv_slice_bytes)?;
        }
        let gdn_conv_per = (snap.identity.gdn_conv_elements_per_layer as usize) * 4;
        let gdn_state_per = (snap.identity.gdn_state_elements_per_layer as usize) * 4;
        for (layer, (conv, state)) in self.gdn_conv.iter().zip(&self.gdn_state).enumerate() {
            validate_snapshot_tensor_span("gdn_conv", layer, conv, gdn_conv_per)?;
            validate_snapshot_tensor_span("gdn_state", layer, state, gdn_state_per)?;
        }

        for i in 0..n_attn {
            let off = i * kv_slice_bytes;
            write_tensor_bytes(&self.kv_k[i], &snap.kv_k_arena[off..off + kv_slice_bytes]);
            write_tensor_bytes(&self.kv_v[i], &snap.kv_v_arena[off..off + kv_slice_bytes]);
        }
        self.kv_n_pos.copy_from_slice(&snap.kv_n_pos);

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
        self.poison_reason = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::Forward;
    use crate::gguf::GgufFile;
    use crate::loader::Model;
    use crate::sampling::{Sampler, SamplingConfig};

    fn metal_test_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(ctx) => Some(ctx),
            Err(MetalError::EmptyLibrary | MetalError::NoDevice) => {
                let required = matches!(
                    std::env::var("QWEN_REQUIRE_METAL_TESTS").as_deref(),
                    Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
                );
                if required {
                    panic!("Metal is required but unavailable");
                }
                None
            }
            Err(error) => panic!("Metal context: {error}"),
        }
    }

    fn snapshot_validation_fixture() -> SessionSnapshot {
        SessionSnapshot {
            identity: SnapshotIdentity {
                model_id: 11,
                tokenizer_id: 12,
                layout_version: SNAPSHOT_LAYOUT_VERSION,
                n_attn_layers: 2,
                n_gdn_layers: 3,
                kv_dim_elements: 4,
                kv_bytes_per_token: 8,
                kv_storage_kind: SnapshotKvStorageKind::F16,
                gdn_state_elements_per_layer: 5,
                gdn_conv_elements_per_layer: 6,
            },
            prefix_tokens: vec![1, 2],
            pending_token: None,
            kv_n_pos: vec![2, 2],
            kv_k_arena: vec![0; 32],
            kv_v_arena: vec![0; 32],
            gdn_conv_arena: vec![0; 72],
            gdn_state_arena: vec![0; 60],
            final_logits: Some(vec![0.0; 4]),
        }
    }

    #[test]
    fn snapshot_validation_accepts_only_complete_consistent_sections() {
        let valid = snapshot_validation_fixture();
        valid
            .validate_for_restore(&valid.identity, 8, Some(4))
            .expect("valid fixture");

        let mut without_logits = valid.clone();
        without_logits.final_logits = None;
        without_logits
            .validate_for_restore(&without_logits.identity, 8, Some(4))
            .expect("logits are an optional snapshot capability");

        let mut with_pending = valid.clone();
        with_pending.pending_token = Some(3);
        assert_eq!(with_pending.matched_prefix_len(), 3);
        with_pending
            .validate_for_restore(&with_pending.identity, 8, Some(4))
            .expect("valid pending token");
        with_pending.pending_token = Some(-1);
        assert!(matches!(
            with_pending.validate_for_restore(&with_pending.identity, 8, Some(4)),
            Err(SnapshotValidationError::TokenOutOfRange { index: 2, .. })
        ));
        with_pending.pending_token = Some(4);
        assert!(matches!(
            with_pending.validate_for_restore(&with_pending.identity, 8, Some(4)),
            Err(SnapshotValidationError::TokenOutOfRange { index: 2, .. })
        ));

        let mut bad = valid.clone();
        bad.kv_n_pos.pop();
        assert!(matches!(
            bad.validate_for_restore(&bad.identity, 8, Some(4)),
            Err(SnapshotValidationError::SectionLength {
                section: "kv_n_pos",
                ..
            })
        ));

        let mut bad = valid.clone();
        bad.kv_n_pos[1] = 1;
        assert!(matches!(
            bad.validate_for_restore(&bad.identity, 8, Some(4)),
            Err(SnapshotValidationError::KvPosition { layer: 1, .. })
        ));

        for section in [
            "kv_k_arena",
            "kv_v_arena",
            "gdn_conv_arena",
            "gdn_state_arena",
            "final_logits",
        ] {
            let mut bad = valid.clone();
            match section {
                "kv_k_arena" => bad.kv_k_arena.pop(),
                "kv_v_arena" => bad.kv_v_arena.pop(),
                "gdn_conv_arena" => bad.gdn_conv_arena.pop(),
                "gdn_state_arena" => bad.gdn_state_arena.pop(),
                "final_logits" => {
                    bad.final_logits.as_mut().expect("logits").pop();
                    Some(0)
                }
                _ => unreachable!(),
            };
            assert!(matches!(
                bad.validate_for_restore(&bad.identity, 8, Some(4)),
                Err(SnapshotValidationError::SectionLength {
                    section: actual,
                    ..
                }) if actual == section
            ));
        }

        let mut bad = valid.clone();
        bad.prefix_tokens[1] = 4;
        assert!(matches!(
            bad.validate_for_restore(&bad.identity, 8, Some(4)),
            Err(SnapshotValidationError::TokenOutOfRange { index: 1, .. })
        ));
        let mut bad = valid.clone();
        bad.prefix_tokens[0] = -1;
        assert!(matches!(
            bad.validate_for_restore(&bad.identity, 8, Some(4)),
            Err(SnapshotValidationError::TokenOutOfRange { index: 0, .. })
        ));
        assert!(matches!(
            valid.validate_for_restore(&valid.identity, 1, Some(4)),
            Err(SnapshotValidationError::PrefixCapacity { .. })
        ));

        let mut wrong_identity = valid.identity.clone();
        wrong_identity.layout_version += 1;
        assert!(matches!(
            valid.validate_for_restore(&wrong_identity, 8, Some(4)),
            Err(SnapshotValidationError::IdentityMismatch { .. })
        ));
    }

    #[test]
    fn snapshot_validation_accepts_empty_and_single_state_families() {
        let mut empty = snapshot_validation_fixture();
        empty.prefix_tokens.clear();
        empty.kv_n_pos.fill(0);
        empty.kv_k_arena.clear();
        empty.kv_v_arena.clear();
        empty
            .validate_for_restore(&empty.identity, 8, Some(4))
            .expect("empty prefix");

        let mut pure_gdn = empty.clone();
        pure_gdn.identity.n_attn_layers = 0;
        pure_gdn.kv_n_pos.clear();
        pure_gdn
            .validate_for_restore(&pure_gdn.identity, 8, Some(4))
            .expect("pure GDN");

        let mut pure_attention = empty;
        pure_attention.identity.n_gdn_layers = 0;
        pure_attention.gdn_conv_arena.clear();
        pure_attention.gdn_state_arena.clear();
        pure_attention
            .validate_for_restore(&pure_attention.identity, 8, Some(4))
            .expect("pure attention");
    }

    #[test]
    fn snapshot_validation_rejects_oversized_sections() {
        let valid = snapshot_validation_fixture();
        for section in [
            "kv_k_arena",
            "kv_v_arena",
            "gdn_conv_arena",
            "gdn_state_arena",
            "final_logits",
        ] {
            let mut bad = valid.clone();
            match section {
                "kv_k_arena" => bad.kv_k_arena.push(0),
                "kv_v_arena" => bad.kv_v_arena.push(0),
                "gdn_conv_arena" => bad.gdn_conv_arena.push(0),
                "gdn_state_arena" => bad.gdn_state_arena.push(0),
                "final_logits" => bad.final_logits.as_mut().expect("logits").push(0.0),
                _ => unreachable!(),
            }
            assert!(matches!(
                bad.validate_for_restore(&bad.identity, 8, Some(4)),
                Err(SnapshotValidationError::SectionLength {
                    section: actual,
                    ..
                }) if actual == section
            ));
        }
    }

    #[test]
    fn snapshot_validation_rejects_section_length_overflow() {
        let mut snapshot = snapshot_validation_fixture();
        snapshot.identity.n_attn_layers = 0;
        snapshot.identity.n_gdn_layers = u32::MAX;
        snapshot.identity.gdn_conv_elements_per_layer = u32::MAX;
        snapshot.kv_n_pos.clear();
        snapshot.kv_k_arena.clear();
        snapshot.kv_v_arena.clear();
        assert!(matches!(
            snapshot.validate_for_restore(&snapshot.identity, 8, Some(4)),
            Err(SnapshotValidationError::LengthOverflow {
                section: "gdn_conv_arena"
            })
        ));

        assert!(matches!(
            checked_snapshot_product("kv_arena", &[usize::MAX, 2]),
            Err(SnapshotValidationError::LengthOverflow {
                section: "kv_arena"
            })
        ));
    }

    #[test]
    fn gguf_no_copy_mode_is_strict_and_default_off() {
        assert_eq!(
            parse_gguf_owned_arena_mode(None).unwrap(),
            GgufOwnedArenaMode::Disabled
        );
        for value in ["1", "true", "TRUE", "yes", "YES"] {
            assert_eq!(
                parse_gguf_owned_arena_mode(Some(value)).unwrap(),
                GgufOwnedArenaMode::Forced
            );
        }
        for value in ["0", "false", "FALSE", "no", "NO"] {
            assert_eq!(
                parse_gguf_owned_arena_mode(Some(value)).unwrap(),
                GgufOwnedArenaMode::Disabled
            );
        }
        assert!(parse_gguf_owned_arena_mode(Some("enabled")).is_err());
        assert!(parse_gguf_owned_arena_mode(Some("")).is_err());

        assert_eq!(
            parse_gguf_no_copy_mode(None).unwrap(),
            GgufNoCopyMode::Disabled
        );
        for value in ["1", "true", "TRUE", "yes", "YES"] {
            assert_eq!(
                parse_gguf_no_copy_mode(Some(value)).unwrap(),
                GgufNoCopyMode::Forced
            );
        }
        for value in ["0", "false", "FALSE", "no", "NO"] {
            assert_eq!(
                parse_gguf_no_copy_mode(Some(value)).unwrap(),
                GgufNoCopyMode::Disabled
            );
        }
        assert!(parse_gguf_no_copy_mode(Some("enabled")).is_err());
        assert!(parse_gguf_no_copy_mode(Some("")).is_err());

        assert_eq!(
            parse_gguf_no_copy_prefault(None).unwrap(),
            GgufNoCopyPrefaultMode::Default
        );
        for value in ["1", "true", "TRUE", "yes", "YES"] {
            assert_eq!(
                parse_gguf_no_copy_prefault(Some(value)).unwrap(),
                GgufNoCopyPrefaultMode::Enabled
            );
        }
        for value in ["0", "false", "FALSE", "no", "NO"] {
            assert_eq!(
                parse_gguf_no_copy_prefault(Some(value)).unwrap(),
                GgufNoCopyPrefaultMode::Disabled
            );
        }
        assert!(parse_gguf_no_copy_prefault(Some("enabled")).is_err());
        assert!(parse_gguf_no_copy_prefault(Some("")).is_err());
    }

    #[test]
    fn parallel_copy_policy_is_strict_and_conflict_complete() {
        assert_eq!(
            parse_gguf_parallel_copy_mode(None).unwrap(),
            GgufParallelCopyMode::Auto
        );
        for value in ["1", "true", "TRUE", "yes", "YES"] {
            assert_eq!(
                parse_gguf_parallel_copy_mode(Some(value)).unwrap(),
                GgufParallelCopyMode::ForcedCopy
            );
        }
        assert_eq!(
            parse_gguf_parallel_copy_mode(Some("pread")).unwrap(),
            GgufParallelCopyMode::ForcedPread
        );
        assert_eq!(
            parse_gguf_parallel_copy_mode(Some("PREAD")).unwrap(),
            GgufParallelCopyMode::ForcedPread
        );
        assert_eq!(
            parse_gguf_parallel_copy_mode(Some("page-rounded-copy")).unwrap(),
            GgufParallelCopyMode::ForcedPageRoundedCopy
        );
        assert_eq!(
            parse_gguf_parallel_copy_mode(Some("PAGE-ROUNDED-COPY")).unwrap(),
            GgufParallelCopyMode::ForcedPageRoundedCopy
        );
        for value in ["0", "false", "FALSE", "no", "NO"] {
            assert_eq!(
                parse_gguf_parallel_copy_mode(Some(value)).unwrap(),
                GgufParallelCopyMode::Disabled
            );
        }
        assert!(parse_gguf_parallel_copy_mode(Some("enabled")).is_err());
        assert!(parse_gguf_parallel_copy_mode(Some("")).is_err());

        let valid = |router_f16| {
            validate_parallel_copy_policy(
                GgufParallelCopyMode::ForcedCopy,
                GgufNoCopyMode::Disabled,
                GgufOwnedArenaMode::Disabled,
                false,
                false,
                router_f16,
            )
        };
        assert!(valid(None).is_ok());
        assert!(valid(Some("0")).is_ok());
        assert!(valid(Some("false")).is_ok());
        assert!(valid(Some("1")).is_err());
        assert!(valid(Some("invalid")).is_err());
        assert!(
            validate_parallel_copy_policy(
                GgufParallelCopyMode::ForcedPageRoundedCopy,
                GgufNoCopyMode::Disabled,
                GgufOwnedArenaMode::Disabled,
                false,
                false,
                None,
            )
            .is_ok()
        );
        assert!(
            validate_parallel_copy_policy(
                GgufParallelCopyMode::ForcedPread,
                GgufNoCopyMode::Disabled,
                GgufOwnedArenaMode::Disabled,
                false,
                false,
                None,
            )
            .is_ok()
        );
        assert!(
            validate_parallel_copy_policy(
                GgufParallelCopyMode::ForcedPread,
                GgufNoCopyMode::Forced,
                GgufOwnedArenaMode::Disabled,
                false,
                false,
                None,
            )
            .is_err()
        );
        assert!(
            validate_parallel_copy_policy(
                GgufParallelCopyMode::ForcedCopy,
                GgufNoCopyMode::Forced,
                GgufOwnedArenaMode::Disabled,
                false,
                false,
                None,
            )
            .is_err()
        );
        assert!(
            validate_parallel_copy_policy(
                GgufParallelCopyMode::ForcedCopy,
                GgufNoCopyMode::Disabled,
                GgufOwnedArenaMode::Forced,
                false,
                false,
                None,
            )
            .is_err()
        );
        assert!(
            validate_parallel_copy_policy(
                GgufParallelCopyMode::ForcedCopy,
                GgufNoCopyMode::Disabled,
                GgufOwnedArenaMode::Disabled,
                true,
                false,
                None,
            )
            .is_err()
        );
        assert!(
            validate_parallel_copy_policy(
                GgufParallelCopyMode::ForcedCopy,
                GgufNoCopyMode::Disabled,
                GgufOwnedArenaMode::Disabled,
                false,
                true,
                None,
            )
            .is_err()
        );
        assert!(
            validate_parallel_copy_policy(
                GgufParallelCopyMode::Auto,
                GgufNoCopyMode::Forced,
                GgufOwnedArenaMode::Forced,
                true,
                true,
                Some("invalid"),
            )
            .is_ok()
        );
    }

    #[test]
    fn parallel_copy_auto_requires_scope_and_yields_to_overrides() {
        assert!(auto_parallel_copy_a3b_enabled(
            true,
            GgufParallelCopyMode::Auto,
            false,
        ));
        assert!(!auto_parallel_copy_a3b_enabled(
            false,
            GgufParallelCopyMode::Auto,
            false,
        ));
        assert!(!auto_parallel_copy_a3b_enabled(
            true,
            GgufParallelCopyMode::Disabled,
            false,
        ));
        assert!(!auto_parallel_copy_a3b_enabled(
            true,
            GgufParallelCopyMode::ForcedCopy,
            false,
        ));
        assert!(!auto_parallel_copy_a3b_enabled(
            true,
            GgufParallelCopyMode::ForcedPageRoundedCopy,
            false,
        ));
        assert!(!auto_parallel_copy_a3b_enabled(
            true,
            GgufParallelCopyMode::Auto,
            true,
        ));
        assert!(!auto_parallel_copy_a3b_override_present(|_| false));
        for expected in A3B_PARALLEL_COPY_AUTO_OVERRIDE_ENVS {
            assert!(auto_parallel_copy_a3b_override_present(
                |name| name == expected
            ));
        }
        assert!(!MetalModelLoadOptions::default().auto_parallel_copy_a3b);
    }

    #[test]
    fn parallel_copy_auto_population_is_exact_and_narrow() {
        assert_eq!(
            auto_parallel_copy_population(ParallelCopyProfileId::A3bQ4kmV1),
            Some(ParallelPopulationMethod::Pread)
        );
        assert_eq!(
            auto_parallel_copy_population(ParallelCopyProfileId::A10bQ4xlV1),
            None
        );
        assert_eq!(
            auto_parallel_copy_population(ParallelCopyProfileId::Dense27bQ4kmV1),
            None
        );
    }

    #[test]
    fn gguf_parallel_pread_capability_table_is_exact() {
        let observed = PARALLEL_COPY_PROFILES
            .iter()
            .map(|profile| (profile.id, profile.supports_direct_pread))
            .collect::<Vec<_>>();
        assert_eq!(
            observed,
            vec![
                (ParallelCopyProfileId::A3bQ4kmV1, true),
                (ParallelCopyProfileId::A10bQ4xlV1, true),
                (ParallelCopyProfileId::Dense27bQ4kmV1, true),
            ]
        );
        for profile in PARALLEL_COPY_PROFILES {
            if profile.id == ParallelCopyProfileId::A10bQ4xlV1 {
                assert!(
                    validate_parallel_population(profile, ParallelPopulationMethod::MmapCopy)
                        .is_err()
                );
            } else {
                validate_parallel_population(profile, ParallelPopulationMethod::MmapCopy)
                    .expect("mmap population capability");
            }
            validate_parallel_population(profile, ParallelPopulationMethod::Pread)
                .expect("pread population capability");
        }
    }

    #[test]
    fn gguf_parallel_pread_dense_marker_table_is_exact() {
        assert_eq!(
            parallel_copy_marker_label(
                &DENSE27B_PARALLEL_COPY_PROFILE,
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::LogicalExact,
            )
            .unwrap(),
            "[metal-gguf-parallel-copied]"
        );
        assert_eq!(
            parallel_copy_marker_label(
                &DENSE27B_PARALLEL_COPY_PROFILE,
                ParallelPopulationMethod::Pread,
                ParallelDestinationLength::LogicalExact,
            )
            .unwrap(),
            "[metal-gguf-parallel-pread]"
        );
    }

    #[test]
    fn page_rounded_parallel_copy_contract_is_narrow_and_checked() {
        assert_eq!(
            GgufParallelCopyMode::ForcedCopy.forced_configuration(),
            Some((
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::LogicalExact,
            ))
        );
        assert_eq!(
            GgufParallelCopyMode::ForcedPread.forced_configuration(),
            Some((
                ParallelPopulationMethod::Pread,
                ParallelDestinationLength::LogicalExact,
            ))
        );
        assert_eq!(
            GgufParallelCopyMode::ForcedPageRoundedCopy.forced_configuration(),
            Some((
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::PageRounded16K,
            ))
        );
        assert!(GgufParallelCopyMode::Auto.forced_configuration().is_none());
        assert!(
            GgufParallelCopyMode::Disabled
                .forced_configuration()
                .is_none()
        );

        assert!(
            validate_parallel_destination_length(
                &A3B_PARALLEL_COPY_PROFILE,
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::PageRounded16K,
            )
            .is_ok()
        );
        assert!(
            validate_parallel_destination_length(
                &A3B_PARALLEL_COPY_PROFILE,
                ParallelPopulationMethod::Pread,
                ParallelDestinationLength::PageRounded16K,
            )
            .is_err()
        );
        assert!(
            validate_parallel_destination_length(
                &DENSE27B_PARALLEL_COPY_PROFILE,
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::PageRounded16K,
            )
            .is_err()
        );
        assert_eq!(
            parallel_copy_marker_label(
                &A3B_PARALLEL_COPY_PROFILE,
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::PageRounded16K,
            )
            .unwrap(),
            "[metal-gguf-parallel-page-rounded]"
        );

        assert_eq!(
            parallel_destination_resource_length(
                4,
                ParallelDestinationLength::PageRounded16K,
                16_384,
            )
            .unwrap(),
            16_384
        );
        assert_eq!(
            parallel_destination_resource_length(
                16_384,
                ParallelDestinationLength::PageRounded16K,
                16_384,
            )
            .unwrap(),
            16_384
        );
        assert_eq!(
            parallel_destination_resource_length(
                16_385,
                ParallelDestinationLength::PageRounded16K,
                32_768,
            )
            .unwrap(),
            32_768
        );
        assert!(
            parallel_destination_resource_length(
                16_385,
                ParallelDestinationLength::PageRounded16K,
                32_767,
            )
            .is_err()
        );
        assert!(
            parallel_destination_resource_length(
                usize::MAX as u64,
                ParallelDestinationLength::PageRounded16K,
                usize::MAX,
            )
            .is_err()
        );
        assert!(
            parallel_destination_resource_length(
                0,
                ParallelDestinationLength::LogicalExact,
                usize::MAX,
            )
            .is_err()
        );

        let identity = |index: usize, source_bytes: u64| ModelWeightStorageIdentity {
            name: format!("weight.{index}"),
            shard_idx: 0,
            data_offset: index as u64 * 65_536,
            source_bytes,
            dtype: GgmlType::F32,
            shape: vec![source_bytes / 4],
            kind: ModelWeightStorageKind::Direct,
            resident_bytes: source_bytes,
        };
        let identities = [identity(0, 4), identity(1, 16_384), identity(2, 16_388)];
        assert_eq!(
            parallel_destination_accounting(
                &identities,
                ParallelDestinationLength::LogicalExact,
                usize::MAX,
            )
            .unwrap(),
            (32_776, 0)
        );
        assert_eq!(
            parallel_destination_accounting(
                &identities,
                ParallelDestinationLength::PageRounded16K,
                usize::MAX,
            )
            .unwrap(),
            (65_536, 2)
        );
    }

    #[test]
    fn gguf_parallel_pread_non_a3b_auto_remains_none() {
        assert_eq!(
            auto_parallel_copy_population(ParallelCopyProfileId::Dense27bQ4kmV1),
            None
        );
        assert_eq!(
            auto_parallel_copy_population(ParallelCopyProfileId::A10bQ4xlV1),
            None
        );
        assert_eq!(
            auto_parallel_copy_population(ParallelCopyProfileId::A3bQ4kmV1),
            Some(ParallelPopulationMethod::Pread)
        );
    }

    #[test]
    fn gguf_parallel_pread_a10b_profile_is_exact_and_force_only() {
        let profile = &A10B_PARALLEL_PREAD_PROFILE;
        assert_eq!(profile.id.label(), "a10b-q4xl-v1");
        assert_eq!(profile.architecture_label, Some("qwen35moe"));
        assert_eq!(profile.arch, A10B_PARALLEL_COPY_ARCH);
        assert_eq!(
            profile.shard_mapped_lengths,
            [10_943_552, 49_640_779_424, 27_378_273_056]
        );
        assert_eq!(profile.descriptor_layout_digest, 0x3eb2_9091_5bec_2041);
        assert_eq!(
            profile.inventory_digest,
            "b331c475123dbee3bc862a495266dee3996c5f3adabcd6fbeaff9bbabd71a4f8"
        );
        assert_eq!(profile.embedding_dtype, GgmlType::Q8_0);
        assert_eq!(profile.embedding_shape, [3072, 248_320]);
        assert_eq!(profile.request_count, 879);
        assert_eq!(profile.source_bytes, 77_018_996_736);
        assert_eq!(profile.cuts, [214, 435, 658]);
        assert_eq!(profile.task_counts, [214, 221, 223, 221]);
        assert_eq!(
            profile.worker_bytes,
            [
                19_474_295_808,
                19_228_744_704,
                19_231_902_720,
                19_084_053_504
            ]
        );
        assert!(validate_parallel_population(profile, ParallelPopulationMethod::Pread).is_ok());
        assert!(validate_parallel_population(profile, ParallelPopulationMethod::MmapCopy).is_err());
        assert_eq!(
            parallel_copy_marker_label(
                profile,
                ParallelPopulationMethod::Pread,
                ParallelDestinationLength::LogicalExact,
            )
            .unwrap(),
            "[metal-gguf-parallel-pread]"
        );
    }

    #[test]
    fn gguf_parallel_pread_dense_advice_preserves_configured_policy() {
        let selected = PreparedAutoSelection::Selected(PreparedParallelCopiedProfile {
            profile: &DENSE27B_PARALLEL_COPY_PROFILE,
            population: ParallelPopulationMethod::Pread,
            destination_length: ParallelDestinationLength::LogicalExact,
            expected_identities: Vec::new(),
            sorted_request_indices: Vec::new(),
            _proof: PreparedParallelCopyProof::DensePlannerFree,
        });
        assert_eq!(
            selected.prefetch_advice(),
            MetalLoadPrefetchAdvice::PreserveConfiguredPolicy
        );
        assert_eq!(
            PreparedAutoSelection::NotEligible.prefetch_advice(),
            MetalLoadPrefetchAdvice::PreserveConfiguredPolicy
        );
    }

    #[test]
    fn prepared_auto_prefetch_advice_selector_table_is_fail_closed() {
        #[derive(Clone, Copy)]
        enum Selection {
            A3bPread,
            A3bCopy,
            A3bPageRoundedCopy,
            A3bPreadWrongProof,
            DenseCopy,
            NoMatch,
        }

        let selected = |profile: &'static ParallelCopyProfile,
                        population: ParallelPopulationMethod,
                        destination_length: ParallelDestinationLength,
                        proof: PreparedParallelCopyProof| {
            PreparedAutoSelection::Selected(PreparedParallelCopiedProfile {
                profile,
                population,
                destination_length,
                expected_identities: Vec::new(),
                sorted_request_indices: Vec::new(),
                _proof: proof,
            })
        };
        let selection = |case| match case {
            Selection::A3bPread => selected(
                &A3B_PARALLEL_COPY_PROFILE,
                ParallelPopulationMethod::Pread,
                ParallelDestinationLength::LogicalExact,
                PreparedParallelCopyProof::A3bRetainedPlan,
            ),
            Selection::A3bCopy => selected(
                &A3B_PARALLEL_COPY_PROFILE,
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::LogicalExact,
                PreparedParallelCopyProof::A3bRetainedPlan,
            ),
            Selection::A3bPageRoundedCopy => selected(
                &A3B_PARALLEL_COPY_PROFILE,
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::PageRounded16K,
                PreparedParallelCopyProof::A3bRetainedPlan,
            ),
            Selection::A3bPreadWrongProof => selected(
                &A3B_PARALLEL_COPY_PROFILE,
                ParallelPopulationMethod::Pread,
                ParallelDestinationLength::LogicalExact,
                PreparedParallelCopyProof::DensePlannerFree,
            ),
            Selection::DenseCopy => selected(
                &DENSE27B_PARALLEL_COPY_PROFILE,
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::LogicalExact,
                PreparedParallelCopyProof::DensePlannerFree,
            ),
            Selection::NoMatch => PreparedAutoSelection::NoMatch,
        };
        let cases = [
            (
                "disposable-auto-authenticated-a3b-pread",
                true,
                GgufParallelCopyMode::Auto,
                false,
                Selection::A3bPread,
                MetalLoadPrefetchAdvice::SuppressColdOnlyAuthenticatedA3bDirectPread,
            ),
            (
                "force-only",
                false,
                GgufParallelCopyMode::Auto,
                false,
                Selection::A3bPread,
                MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
            ),
            (
                "forced-copy",
                true,
                GgufParallelCopyMode::ForcedCopy,
                false,
                Selection::A3bPread,
                MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
            ),
            (
                "forced-pread",
                true,
                GgufParallelCopyMode::ForcedPread,
                false,
                Selection::A3bPread,
                MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
            ),
            (
                "parallel-copy-disabled",
                true,
                GgufParallelCopyMode::Disabled,
                false,
                Selection::A3bPread,
                MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
            ),
            (
                "explicit-override",
                true,
                GgufParallelCopyMode::Auto,
                true,
                Selection::A3bPread,
                MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
            ),
            (
                "no-profile-match",
                true,
                GgufParallelCopyMode::Auto,
                false,
                Selection::NoMatch,
                MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
            ),
            (
                "mmap-copy-population",
                true,
                GgufParallelCopyMode::Auto,
                false,
                Selection::A3bCopy,
                MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
            ),
            (
                "page-rounded-mmap-copy-population",
                true,
                GgufParallelCopyMode::Auto,
                false,
                Selection::A3bPageRoundedCopy,
                MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
            ),
            (
                "authentication-proof-mismatch",
                true,
                GgufParallelCopyMode::Auto,
                false,
                Selection::A3bPreadWrongProof,
                MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
            ),
            (
                "other-profile",
                true,
                GgufParallelCopyMode::Auto,
                false,
                Selection::DenseCopy,
                MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
            ),
        ];

        for (name, admission, mode, override_present, selected, expected) in cases {
            let auto = if auto_parallel_copy_a3b_enabled(admission, mode, override_present) {
                selection(selected)
            } else {
                PreparedAutoSelection::NotEligible
            };
            assert_eq!(auto.prefetch_advice(), expected, "{name}");
        }
    }

    #[test]
    fn parallel_copy_auto_host_gate_is_exact() {
        assert!(a3b_parallel_copy_auto_host_supported(
            true,
            "Apple M4 Max",
            Some(128 * 1024 * 1024 * 1024),
        ));
        assert!(a3b_parallel_copy_auto_host_supported(
            true,
            "Apple M4 Max",
            Some(192 * 1024 * 1024 * 1024),
        ));
        assert!(!a3b_parallel_copy_auto_host_supported(
            false,
            "Apple M4 Max",
            Some(128 * 1024 * 1024 * 1024),
        ));
        assert!(!a3b_parallel_copy_auto_host_supported(
            true,
            "Apple M4 Pro",
            Some(128 * 1024 * 1024 * 1024),
        ));
        assert!(!a3b_parallel_copy_auto_host_supported(
            true,
            "Apple M4 Max",
            Some(128 * 1024 * 1024 * 1024 - 1),
        ));
        assert!(!a3b_parallel_copy_auto_host_supported(
            true,
            "Apple M4 Max",
            None,
        ));
    }

    #[test]
    fn parallel_copy_profile_selection_is_exactly_one() {
        let selected = select_unique_parallel_copy_profile(&PARALLEL_COPY_PROFILES, |profile| {
            Ok(profile.id == ParallelCopyProfileId::Dense27bQ4kmV1)
        })
        .unwrap();
        assert_eq!(selected.id, ParallelCopyProfileId::Dense27bQ4kmV1);

        assert!(
            select_unique_parallel_copy_profile(&PARALLEL_COPY_PROFILES, |_| Ok(false)).is_err()
        );
        assert!(
            select_unique_parallel_copy_profile(&PARALLEL_COPY_PROFILES, |_| Ok(true)).is_err()
        );
        assert!(
            select_unique_parallel_copy_profile(
                &[&A3B_PARALLEL_COPY_PROFILE, &A3B_PARALLEL_COPY_PROFILE],
                |_| Ok(true),
            )
            .is_err()
        );
    }

    #[test]
    fn owned_arena_worker_boundaries_cover_nondivisible_page_counts() {
        let page = 16_384;
        let boundaries =
            owned_arena_four_worker_boundaries(11 * page, page).expect("valid boundaries");
        assert_eq!(boundaries, [0, 2 * page, 5 * page, 8 * page, 11 * page]);
        assert!(owned_arena_four_worker_boundaries(3 * page, page).is_err());
        assert!(owned_arena_four_worker_boundaries(4 * page + 1, page).is_err());
    }

    #[test]
    fn model_weight_request_sequence_rejects_equal_aggregate_drift() {
        let direct_desc = TensorDesc {
            name: "direct".to_string(),
            shape: vec![8],
            dtype: GgmlType::F32,
            shard_idx: 0,
            data_offset: 32,
            n_bytes: 32,
        };
        let converted_desc = TensorDesc {
            name: "converted".to_string(),
            shape: vec![8],
            dtype: GgmlType::F32,
            shard_idx: 0,
            data_offset: 64,
            n_bytes: 32,
        };
        let expected = vec![
            ModelWeightStorageRequest {
                desc: &direct_desc,
                kind: ModelWeightStorageKind::Direct,
                resident_bytes: 32,
            },
            ModelWeightStorageRequest {
                desc: &converted_desc,
                kind: ModelWeightStorageKind::ConvertedF32,
                resident_bytes: 32,
            },
        ];
        let actual = expected
            .iter()
            .map(expected_model_weight_identity)
            .collect::<Vec<_>>();
        validate_model_weight_request_sequence(&actual, &expected).unwrap();

        let mut reordered = actual.clone();
        reordered.swap(0, 1);
        assert!(validate_model_weight_request_sequence(&reordered, &expected).is_err());

        let mut swapped_kinds = actual.clone();
        swapped_kinds[0].kind = ModelWeightStorageKind::ConvertedF32;
        swapped_kinds[1].kind = ModelWeightStorageKind::Direct;
        assert!(validate_model_weight_request_sequence(&swapped_kinds, &expected).is_err());

        let mut duplicated = actual;
        duplicated[0] = duplicated[1].clone();
        assert!(validate_model_weight_request_sequence(&duplicated, &expected).is_err());
    }

    struct GenericRetainedArmResult {
        prefill_logits: Vec<f32>,
        prefill_snapshot: SessionSnapshot,
        next_token: i32,
        decode_logits: Vec<f32>,
        decode_snapshot: SessionSnapshot,
    }

    #[derive(Clone, Copy)]
    struct GenericRetainedContract {
        windows: usize,
        window_bytes: u64,
        direct: usize,
        views: usize,
        view_bytes: u64,
        aliases: usize,
        alias_bytes: u64,
        fallbacks: usize,
        fallback_bytes: u64,
    }

    fn assert_generic_retained_contract(
        ctx: &MetalContext,
        gguf: &GgufFile,
        model: &Model<'_>,
        native_embedding: bool,
        contract: GenericRetainedContract,
    ) {
        let expected = model_weight_storage_requests(model, native_embedding, false)
            .expect("generic fixture storage requests");
        let direct = expected
            .iter()
            .filter(|request| request.kind == ModelWeightStorageKind::Direct)
            .map(|request| request.desc)
            .collect::<Vec<_>>();
        let mut storage = planned_retained_storage_for_load(ctx, gguf, &expected, false)
            .expect("generic fixture retained storage");
        let window_bytes = storage
            .plan
            .windows
            .iter()
            .map(|window| window.length as u64)
            .sum::<u64>();
        let views = storage
            .plan
            .entries
            .iter()
            .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
            .count();
        let aliases = storage
            .plan
            .entries
            .iter()
            .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::Alias { .. }))
            .count();
        let fallbacks = storage
            .plan
            .entries
            .iter()
            .filter(|entry| {
                matches!(
                    entry.disposition,
                    RetainedStorageDisposition::CopyFallback { .. }
                )
            })
            .count();
        assert_eq!(storage.plan.windows.len(), contract.windows);
        assert_eq!(window_bytes, contract.window_bytes);
        assert_eq!(storage.plan.entries.len(), contract.direct);
        assert_eq!(views, contract.views);
        assert_eq!(storage.plan.unique_view_bytes, contract.view_bytes);
        assert_eq!(aliases, contract.aliases);
        assert_eq!(storage.plan.alias_bytes, contract.alias_bytes);
        assert_eq!(fallbacks, contract.fallbacks);
        assert_eq!(storage.plan.unique_fallback_bytes, contract.fallback_bytes);

        let mut ledger = WeightLoadLedger::default();
        for desc in &direct {
            let (tensor, materialization) = storage
                .load_direct(ctx, gguf, desc)
                .expect("materialize generic fixture tensor");
            ledger
                .record_source(desc, materialization, tensor.n_bytes())
                .expect("record generic fixture tensor");
        }
        storage
            .validate_complete(&ledger)
            .expect("complete generic fixture realization");
        for (index, entry) in storage.plan.entries.iter().enumerate() {
            if let RetainedStorageDisposition::Alias {
                source_request_index,
            } = entry.disposition
            {
                let source = storage.realized[source_request_index]
                    .as_ref()
                    .expect("realized alias source");
                let alias = storage.realized[index]
                    .as_ref()
                    .expect("realized alias tensor");
                assert_eq!(source.offset, alias.offset);
                assert_eq!(
                    Retained::as_ptr(&source.buffer),
                    Retained::as_ptr(&alias.buffer)
                );
            }
        }
    }

    fn run_generic_retained_arm(
        ctx: &MetalContext,
        gguf: &GgufFile,
        model: &Model<'_>,
        tokens: &[i32],
        mode: GgufNoCopyMode,
        owned_mode: GgufOwnedArenaMode,
        forced_next: Option<i32>,
    ) -> GenericRetainedArmResult {
        let metal_model = MetalModel::load_with_storage_policy(
            ctx,
            gguf,
            model,
            mode,
            false,
            owned_mode,
            GgufParallelCopyMode::Disabled,
            false,
        )
        .expect("load generic storage arm");
        run_loaded_model_arm(ctx, metal_model, tokens, forced_next)
    }

    fn run_loaded_model_arm(
        ctx: &MetalContext,
        metal_model: MetalModel,
        tokens: &[i32],
        forced_next: Option<i32>,
    ) -> GenericRetainedArmResult {
        use crate::metal_dflash::{
            MetalDFlashLayerMajorScratch, PrefillScratchConfig,
            plan_prefill_scratch_with_matrix_max_pos_configured, prefill_tokens_with_multi_hidden,
        };

        let forward = MetalForward::new(ctx, &metal_model);
        let capacity = 64;
        let mut session =
            MetalSession::fresh(ctx, &metal_model, capacity).expect("fresh generic session");
        let plan = plan_prefill_scratch_with_matrix_max_pos_configured(
            &metal_model,
            tokens.len() as u32,
            capacity,
            PrefillScratchConfig::default(),
        )
        .expect("generic prefill scratch plan");
        let mut scratch =
            MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(ctx, &metal_model, plan)
                .expect("generic prefill scratch");
        let prefill_logits = prefill_tokens_with_multi_hidden(
            &forward,
            tokens,
            0,
            &mut session,
            &mut scratch,
            &[],
            None,
        )
        .expect("generic packed prefill");
        let identity = session.snapshot_identity(0x594, 0x594);
        let prefill_snapshot = session
            .snapshot(
                identity.clone(),
                tokens.to_vec(),
                Some(prefill_logits.clone()),
            )
            .expect("generic prefill snapshot");
        let next_token = forced_next.unwrap_or_else(|| {
            prefill_logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                    if value > best.1 { (index, value) } else { best }
                })
                .0 as i32
        });
        let decode_logits = forward
            .single_token(next_token, tokens.len() as u32, &mut session)
            .expect("generic forced decode transition");
        let mut consumed = tokens.to_vec();
        consumed.push(next_token);
        let decode_snapshot = session
            .snapshot(identity, consumed, Some(decode_logits.clone()))
            .expect("generic decode snapshot");
        GenericRetainedArmResult {
            prefill_logits,
            prefill_snapshot,
            next_token,
            decode_logits,
            decode_snapshot,
        }
    }

    fn assert_generic_retained_f32_bits(label: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "{label} length");
        for (index, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{label} bit mismatch at {index}");
        }
    }

    fn assert_generic_retained_snapshot(label: &str, a: &SessionSnapshot, b: &SessionSnapshot) {
        assert_eq!(a.identity, b.identity, "{label} identity");
        assert_eq!(a.prefix_tokens, b.prefix_tokens, "{label} tokens");
        assert_eq!(a.kv_n_pos, b.kv_n_pos, "{label} KV positions");
        assert_eq!(a.kv_k_arena, b.kv_k_arena, "{label} K arena");
        assert_eq!(a.kv_v_arena, b.kv_v_arena, "{label} V arena");
        assert_eq!(a.gdn_conv_arena, b.gdn_conv_arena, "{label} conv arena");
        assert_eq!(a.gdn_state_arena, b.gdn_state_arena, "{label} state arena");
        match (&a.final_logits, &b.final_logits) {
            (Some(a), Some(b)) => {
                assert_generic_retained_f32_bits(&format!("{label} logits"), a, b)
            }
            (None, None) => {}
            _ => panic!("{label} final-logits presence mismatch"),
        }
    }

    fn assert_generic_retained_model_exact(
        model_path: &str,
        expect_tied: bool,
        embedding_mode: NativeQuantEmbeddingMode,
        native_embedding: bool,
        contract: GenericRetainedContract,
    ) {
        assert_eq!(
            native_quant_embedding_mode(),
            embedding_mode,
            "native embedding environment does not match the fixture contract"
        );
        assert!(
            std::path::Path::new(model_path).is_file(),
            "missing generic retained fixture {model_path}"
        );
        let ctx = MetalContext::new().expect("Metal context");
        let gguf = GgufFile::open(model_path).expect("open generic retained fixture");
        let model = Model::from_gguf(&gguf).expect("bind generic retained fixture");
        assert_eq!(model.tied_embeddings, expect_tied);
        assert!(!matches_no_copy_27b_sentinel(&gguf, &model));
        assert_generic_retained_contract(&ctx, &gguf, &model, native_embedding, contract);
        let tokenizer = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
        let tokens = tokenizer
            .encode("Retained storage must preserve this state.", true)
            .expect("tokenize generic retained prompt");
        assert!(tokens.len() < 64);

        let copied = run_generic_retained_arm(
            &ctx,
            &gguf,
            &model,
            &tokens,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            None,
        );
        let retained = run_generic_retained_arm(
            &ctx,
            &gguf,
            &model,
            &tokens,
            GgufNoCopyMode::Forced,
            GgufOwnedArenaMode::Disabled,
            Some(copied.next_token),
        );
        let retained_argmax = retained
            .prefill_logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                if value > best.1 { (index, value) } else { best }
            })
            .0 as i32;
        assert_eq!(copied.next_token, retained_argmax);
        assert_eq!(retained.next_token, retained_argmax);
        assert_generic_retained_f32_bits(
            "generic prefill logits",
            &copied.prefill_logits,
            &retained.prefill_logits,
        );
        assert_generic_retained_snapshot(
            "generic prefill snapshot",
            &copied.prefill_snapshot,
            &retained.prefill_snapshot,
        );
        assert_generic_retained_f32_bits(
            "generic decode logits",
            &copied.decode_logits,
            &retained.decode_logits,
        );
        assert_generic_retained_snapshot(
            "generic decode snapshot",
            &copied.decode_snapshot,
            &retained.decode_snapshot,
        );
    }

    #[test]
    #[ignore = "requires local tied 0.8B Q8_0 fixture and explicit native embedding"]
    fn gguf_no_copy_generic_tied_q8_is_bit_exact() {
        assert_generic_retained_model_exact(
            "/Users/tito/models/Qwen3.5-0.8B-Q8_0.gguf",
            true,
            NativeQuantEmbeddingMode::Forced,
            true,
            GenericRetainedContract {
                windows: 1,
                window_bytes: 800_882_688,
                direct: 321,
                views: 319,
                view_bytes: 800_877_824,
                aliases: 1,
                alias_bytes: 270_172_160,
                fallbacks: 1,
                fallback_bytes: 4_096,
            },
        );
    }

    #[test]
    #[ignore = "requires local tied 0.8B Q8_0 fixture and no native embedding override"]
    fn gguf_no_copy_generic_tied_q8_converted_embedding_is_bit_exact() {
        assert_generic_retained_model_exact(
            "/Users/tito/models/Qwen3.5-0.8B-Q8_0.gguf",
            true,
            NativeQuantEmbeddingMode::Auto,
            false,
            GenericRetainedContract {
                windows: 1,
                window_bytes: 800_882_688,
                direct: 320,
                views: 319,
                view_bytes: 800_877_824,
                aliases: 0,
                alias_bytes: 0,
                fallbacks: 1,
                fallback_bytes: 4_096,
            },
        );
    }

    #[test]
    #[ignore = "requires local Qwen3.6 A3B Q4 fixture and explicit native embedding"]
    fn gguf_no_copy_generic_a3b_q4_is_bit_exact() {
        assert_generic_retained_model_exact(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            false,
            NativeQuantEmbeddingMode::Forced,
            true,
            GenericRetainedContract {
                windows: 1,
                window_bytes: 22_123_544_576,
                direct: 733,
                views: 732,
                view_bytes: 22_123_530_752,
                aliases: 0,
                alias_bytes: 0,
                fallbacks: 1,
                fallback_bytes: 8_192,
            },
        );
    }

    #[test]
    #[ignore = "requires local Qwen3.6 A3B Q4 fixture"]
    fn gguf_owned_arena_a3b_q4_is_bit_exact() {
        let model_path = "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
        assert_eq!(
            native_quant_embedding_mode(),
            NativeQuantEmbeddingMode::Auto,
            "owned fixture requires production-auto embedding selection"
        );
        let ctx = MetalContext::new().expect("Metal context");
        let gguf = GgufFile::open(model_path).expect("open owned A3B fixture");
        let model = Model::from_gguf(&gguf).expect("bind owned A3B fixture");
        let expected =
            model_weight_storage_requests(&model, true, false).expect("owned A3B storage requests");
        let direct = expected
            .iter()
            .map(|request| request.desc)
            .collect::<Vec<_>>();
        let mut storage = planned_owned_storage_for_load(
            &ctx,
            &gguf,
            &model,
            &expected,
            NativeQuantEmbeddingSelection::AutoPromoted,
        )
        .expect("realize owned A3B storage");
        assert_eq!(storage.resources.len(), 2);
        assert_ne!(
            Retained::as_ptr(&storage.resources[0]),
            Retained::as_ptr(&storage.resources[1])
        );
        let mut ledger = WeightLoadLedger::default();
        for desc in &direct {
            let (tensor, materialization) = storage
                .load_direct(desc)
                .expect("materialize owned A3B tensor");
            assert_eq!(
                tensor.provenance(),
                MetalTensorProvenance::OwnedWeightReadOnly
            );
            assert!(!tensor.is_writable());
            ledger
                .record_source(desc, materialization, tensor.n_bytes())
                .expect("record owned A3B tensor");
        }
        storage
            .validate_complete(&ledger)
            .expect("complete owned A3B realization");
        for (index, entry) in storage.plan.entries.iter().enumerate() {
            let tensor = storage.realized[index]
                .as_ref()
                .expect("realized owned tensor");
            let resource_index = match entry.disposition {
                RetainedStorageDisposition::View { window_index, .. } => window_index,
                RetainedStorageDisposition::CopyFallback { .. } => 1,
                RetainedStorageDisposition::Alias { .. } => {
                    panic!("owned A3B sentinel must not contain aliases")
                }
            };
            assert_eq!(
                Retained::as_ptr(&tensor.buffer),
                Retained::as_ptr(&storage.resources[resource_index])
            );
        }
        drop(storage);

        let tokenizer = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
        let tokens = tokenizer
            .encode("Owned storage must preserve this state.", true)
            .expect("tokenize owned prompt");
        assert!(tokens.len() < 64);
        let copied = run_generic_retained_arm(
            &ctx,
            &gguf,
            &model,
            &tokens,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            None,
        );
        let owned = run_generic_retained_arm(
            &ctx,
            &gguf,
            &model,
            &tokens,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Forced,
            Some(copied.next_token),
        );
        let owned_argmax = owned
            .prefill_logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                if value > best.1 { (index, value) } else { best }
            })
            .0 as i32;
        assert_eq!(copied.next_token, owned_argmax);
        assert_eq!(owned.next_token, owned_argmax);
        assert_generic_retained_f32_bits(
            "owned prefill logits",
            &copied.prefill_logits,
            &owned.prefill_logits,
        );
        assert_generic_retained_snapshot(
            "owned prefill snapshot",
            &copied.prefill_snapshot,
            &owned.prefill_snapshot,
        );
        assert_generic_retained_f32_bits(
            "owned decode logits",
            &copied.decode_logits,
            &owned.decode_logits,
        );
        assert_generic_retained_snapshot(
            "owned decode snapshot",
            &copied.decode_snapshot,
            &owned.decode_snapshot,
        );
    }

    fn assert_gguf_parallel_a3b_q4_is_bit_exact(
        population: ParallelPopulationMethod,
        destination_length: ParallelDestinationLength,
        marker: &str,
    ) {
        let model_path = "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
        assert_eq!(
            native_quant_embedding_mode(),
            NativeQuantEmbeddingMode::Auto,
            "parallel-copy fixture requires production-auto embedding selection"
        );
        assert!(!moe_router_f16_enabled());
        let ctx = MetalContext::new().expect("Metal context");
        let gguf = GgufFile::open(model_path).expect("open parallel-copy A3B fixture");
        let model = Model::from_gguf(&gguf).expect("bind parallel-copy A3B fixture");
        let tokens = [
            7734, 264, 12654, 709, 310, 12204, 279, 76938, 8240, 5199, 7638, 13,
        ];

        let ((reference, parallel), load_lines) = capture_metal_load_lines(|| {
            let run_parallel = |population, destination_length, expected_next_token| {
                let embedding_selection = NativeQuantEmbeddingSelection::AutoPromoted;
                emit_native_quant_embedding_policy(&model, embedding_selection);
                let expected = model_weight_storage_requests(&model, true, false)
                    .expect("parallel-copy A3B storage requests");
                let storage = planned_parallel_copied_storage_for_load(
                    &ctx,
                    &gguf,
                    &model,
                    &expected,
                    embedding_selection,
                    population,
                    destination_length,
                )
                .expect("realize parallel A3B storage");
                storage
                    .validate_source_bytes(&gguf, &expected)
                    .expect("audit every parallel logical source byte");
                validate_parallel_copied_topology(
                    storage.profile,
                    storage.destination_length,
                    &storage.expected,
                    &storage.resources,
                    &storage.tensors,
                )
                .expect("validate parallel topology before model construction");
                assert_eq!(
                    frozen_parallel_copy_order(storage.profile, &storage.expected)
                        .expect("frozen schedule"),
                    storage.sorted_request_indices
                );
                assert_eq!(storage.destination_length, destination_length);
                let (allocated_bytes, padded_resources) = parallel_destination_accounting(
                    &storage.expected,
                    destination_length,
                    ctx.max_buffer_length(),
                )
                .expect("parallel destination accounting");
                match destination_length {
                    ParallelDestinationLength::LogicalExact => {
                        assert_eq!(allocated_bytes, GGUF_OWNED_A3B_SOURCE_BYTES);
                        assert_eq!(padded_resources, 0);
                    }
                    ParallelDestinationLength::PageRounded16K => {
                        assert_eq!(allocated_bytes, GGUF_PAGE_ROUNDED_A3B_ALLOCATED_BYTES);
                        assert_eq!(padded_resources, GGUF_PAGE_ROUNDED_A3B_PADDED_RESOURCES);
                        assert_eq!(
                            allocated_bytes - GGUF_OWNED_A3B_SOURCE_BYTES,
                            GGUF_PAGE_ROUNDED_A3B_PADDING_BYTES
                        );
                        assert!(
                            storage
                                .resources
                                .iter()
                                .all(|resource| resource.length() % 16_384 == 0)
                        );
                    }
                }

                let command = ctx.queue.commandBuffer().expect("write guard command");
                let encoder = crate::metal::KernelEncoder::begin(&command);
                let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    encoder.note_write(&storage.tensors[0]);
                }));
                assert!(
                    write_result.is_err(),
                    "parallel weights must reject compute writes"
                );
                encoder.end();

                let command = ctx.queue.commandBuffer().expect("blit guard command");
                let blit = crate::metal::BlitEncoder::begin(&command);
                let blit_destination = storage
                    .tensors
                    .iter()
                    .find(|tensor| tensor.dtype == GgmlType::F32 && tensor.n_bytes() <= 1_048_576)
                    .expect("small direct F32 candidate weight");
                let writable_source = MetalTensor::zeros_f32(&ctx, blit_destination.shape.clone())
                    .expect("equal-sized blit source");
                assert_eq!(writable_source.n_bytes(), blit_destination.n_bytes());
                let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    blit.copy_tensor(&writable_source, blit_destination);
                }));
                assert!(
                    write_result.is_err(),
                    "parallel weights must reject blit writes"
                );
                blit.end();

                let parallel_model = MetalModel::load_with_direct_storage(
                    &ctx,
                    &gguf,
                    &model,
                    ResolvedWeightLoadChoices {
                        embedding_selection,
                        router_f16: false,
                        fused_qkv_g8: false,
                    },
                    &expected,
                    DirectStorage::ForcedParallelCopied(storage),
                    false,
                )
                .expect("construct model from audited parallel storage");
                run_loaded_model_arm(&ctx, parallel_model, &tokens, expected_next_token)
            };

            let reference = match destination_length {
                ParallelDestinationLength::LogicalExact => run_generic_retained_arm(
                    &ctx,
                    &gguf,
                    &model,
                    &tokens,
                    GgufNoCopyMode::Disabled,
                    GgufOwnedArenaMode::Disabled,
                    None,
                ),
                ParallelDestinationLength::PageRounded16K => run_parallel(
                    ParallelPopulationMethod::MmapCopy,
                    ParallelDestinationLength::LogicalExact,
                    None,
                ),
            };
            let parallel = run_parallel(population, destination_length, Some(reference.next_token));
            (reference, parallel)
        });
        let parallel_argmax = parallel
            .prefill_logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                if value > best.1 { (index, value) } else { best }
            })
            .0 as i32;
        assert_eq!(reference.next_token, parallel_argmax);
        assert_eq!(parallel.next_token, parallel_argmax);
        assert_generic_retained_f32_bits(
            "parallel prefill logits",
            &reference.prefill_logits,
            &parallel.prefill_logits,
        );
        assert_generic_retained_snapshot(
            "parallel prefill snapshot",
            &reference.prefill_snapshot,
            &parallel.prefill_snapshot,
        );
        assert_generic_retained_f32_bits(
            "parallel decode logits",
            &reference.decode_logits,
            &parallel.decode_logits,
        );
        assert_generic_retained_snapshot(
            "parallel decode snapshot",
            &reference.decode_snapshot,
            &parallel.decode_snapshot,
        );

        let native_line = format!(
            "[metal-load] native quantized token embedding policy: auto-promoted ({:?} {:?})",
            model.token_embd.dtype, model.token_embd.shape
        );
        let ledger_line = concat!(
            "[metal-load-ledger] source=733/22123538944 ",
            "direct_copy=733/22123538944 direct_view=0/0 direct_alias=0/0 ",
            "tail_fallback=0/0 converted=0/0/0 derived=0/0"
        );
        let candidate_marker_index = match destination_length {
            ParallelDestinationLength::LogicalExact => {
                assert_eq!(load_lines.len(), 5, "recognized load-line count");
                assert_eq!(load_lines[0], native_line, "A native policy line");
                assert_eq!(load_lines[1], ledger_line, "A copied ledger line");
                assert_eq!(load_lines[2], native_line, "B native policy line");
                assert_eq!(load_lines[4], ledger_line, "B copied ledger line");
                3
            }
            ParallelDestinationLength::PageRounded16K => {
                assert_eq!(load_lines.len(), 6, "recognized load-line count");
                assert_eq!(load_lines[0], native_line, "A native policy line");
                assert!(
                    load_lines[1].starts_with("[metal-gguf-parallel-copied] schema=1 "),
                    "A exact-parallel marker"
                );
                assert_eq!(load_lines[2], ledger_line, "A copied ledger line");
                assert_eq!(load_lines[3], native_line, "B native policy line");
                assert_eq!(load_lines[5], ledger_line, "B copied ledger line");
                4
            }
        };
        assert_eq!(
            load_lines
                .iter()
                .filter(|line| line.starts_with(marker))
                .count(),
            1,
            "candidate marker count"
        );
        assert_eq!(
            load_lines
                .iter()
                .filter(|line| *line == ledger_line)
                .count(),
            2,
            "copied ledger count"
        );

        let marker_contract = match destination_length {
            ParallelDestinationLength::LogicalExact => concat!(
                " schema=1 resources=733 bytes=22123538944 ",
                "workers=4 cuts=155,359,539 tasks=155,204,180,194 ",
                "worker_bytes=5532746240,5462315776,5595522304,5532954624 ",
                "first_offsets=10990048,5543736288,11006052064,16601574368 ",
                "last_offsets=5392741344,11004937952,16450579424,22134520800 ",
                "create=shared,default_cache,default observed=shared,default_cache,tracked ",
                "page=16384 alignment=32 max_buffer=77309411328 mapped=22134528992 ",
                "layout=0x5ae645df5cf7d568 ",
                "inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5 ",
                "plan=fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af "
            ),
            ParallelDestinationLength::PageRounded16K => concat!(
                " schema=2 resources=733 logical_bytes=22123538944 ",
                "allocated_bytes=22126297088 padding_bytes=2758144 ",
                "padded_resources=232 workers=4 cuts=155,359,539 ",
                "tasks=155,204,180,194 ",
                "worker_bytes=5532746240,5462315776,5595522304,5532954624 ",
                "first_offsets=10990048,5543736288,11006052064,16601574368 ",
                "last_offsets=5392741344,11004937952,16450579424,22134520800 ",
                "create=shared,default_cache,default observed=shared,default_cache,tracked ",
                "page=16384 alignment=32 max_buffer=77309411328 mapped=22134528992 ",
                "layout=0x5ae645df5cf7d568 ",
                "inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5 ",
                "plan=fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af "
            ),
        };
        let marker_prefix = format!("{marker}{marker_contract}");
        let timing_suffix = load_lines[candidate_marker_index]
            .strip_prefix(&marker_prefix)
            .expect("exact candidate marker prefix and field order");
        let timing_fields = timing_suffix.split(' ').collect::<Vec<_>>();
        assert_eq!(timing_fields.len(), 5, "candidate timing field count");
        let parse_timing = |index: usize, name: &str| {
            let value = timing_fields[index]
                .strip_prefix(name)
                .expect("candidate timing field name");
            assert!(
                !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
                "candidate timing must be unsigned decimal"
            );
            let parsed = value.parse::<u64>().expect("candidate timing value");
            assert_eq!(parsed.to_string(), value, "candidate timing canonical form");
            parsed
        };
        let allocation_us = parse_timing(0, "allocation_us=");
        let source_us = parse_timing(1, "source_us=");
        let copy_us = parse_timing(2, "copy_us=");
        let binding_us = parse_timing(3, "binding_us=");
        let ready_us = parse_timing(4, "ready_us=");
        let phase_us = allocation_us + source_us + copy_us + binding_us;
        assert!(
            ready_us.abs_diff(phase_us) <= 4,
            "candidate timing reconciliation"
        );
    }

    #[test]
    #[ignore = "requires local Qwen3.6 A3B Q4 fixture"]
    fn gguf_parallel_copied_a3b_q4_is_bit_exact() {
        assert_gguf_parallel_a3b_q4_is_bit_exact(
            ParallelPopulationMethod::MmapCopy,
            ParallelDestinationLength::LogicalExact,
            "[metal-gguf-parallel-copied]",
        );
    }

    #[test]
    #[ignore = "requires local Qwen3.6 A3B Q4 fixture"]
    fn gguf_parallel_pread_a3b_q4_is_bit_exact() {
        assert_gguf_parallel_a3b_q4_is_bit_exact(
            ParallelPopulationMethod::Pread,
            ParallelDestinationLength::LogicalExact,
            "[metal-gguf-parallel-pread]",
        );
    }

    #[test]
    #[ignore = "requires local Qwen3.6 A3B Q4 fixture"]
    fn gguf_parallel_page_rounded_a3b_q4_is_bit_exact() {
        assert_gguf_parallel_a3b_q4_is_bit_exact(
            ParallelPopulationMethod::MmapCopy,
            ParallelDestinationLength::PageRounded16K,
            "[metal-gguf-parallel-page-rounded]",
        );
    }

    fn assert_gguf_parallel_dense27b_q4_is_bit_exact(
        population: ParallelPopulationMethod,
        marker: &str,
    ) {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        assert_eq!(
            native_quant_embedding_mode(),
            NativeQuantEmbeddingMode::Auto,
            "parallel-copy fixture requires production-auto embedding selection"
        );
        assert!(!moe_router_f16_enabled());
        let ctx = MetalContext::new().expect("Metal context");
        let gguf = GgufFile::open(model_path).expect("open parallel-copy dense fixture");
        let model = Model::from_gguf(&gguf).expect("bind parallel-copy dense fixture");
        let profile = &DENSE27B_PARALLEL_COPY_PROFILE;
        assert_eq!(gguf.architecture().as_deref(), profile.architecture_label);
        assert_eq!(gguf.shard_mapped_lengths(), profile.shard_mapped_lengths);
        assert_eq!(
            gguf_descriptor_layout_digest(&gguf),
            profile.descriptor_layout_digest
        );
        assert_eq!(model.arch, profile.arch);
        assert_eq!(model.tied_embeddings, profile.tied_embeddings);
        assert_eq!(model.mtp.is_some(), profile.mtp_present);
        assert_eq!(model.token_embd.dtype, profile.embedding_dtype);
        assert_eq!(model.token_embd.shape, profile.embedding_shape);
        assert_eq!(ctx.device.name().to_string(), "Apple M4 Max");
        assert_eq!(host_page_size_bytes().unwrap(), 16_384);
        assert_eq!(ctx.max_buffer_length(), 77_309_411_328);
        let expected_probe =
            model_weight_storage_requests(&model, true, false).expect("dense profile requests");
        assert_eq!(expected_probe.len(), profile.request_count);
        assert_eq!(
            model_weight_storage_inventory_digest(&expected_probe),
            profile.inventory_digest
        );
        assert!(
            parallel_copy_profile_matches(
                &ctx,
                &gguf,
                &model,
                &expected_probe,
                NativeQuantEmbeddingSelection::AutoPromoted,
                profile,
            )
            .expect("dense profile match")
        );
        drop(expected_probe);
        let tokens = [
            7734, 264, 12654, 709, 310, 12204, 279, 76938, 8240, 5199, 7638, 13,
        ];

        let ((copied, parallel), load_lines) = capture_metal_load_lines(|| {
            let copied = run_generic_retained_arm(
                &ctx,
                &gguf,
                &model,
                &tokens,
                GgufNoCopyMode::Disabled,
                GgufOwnedArenaMode::Disabled,
                None,
            );

            let embedding_selection = NativeQuantEmbeddingSelection::AutoPromoted;
            emit_native_quant_embedding_policy(&model, embedding_selection);
            let expected = model_weight_storage_requests(&model, true, false)
                .expect("parallel-copy dense storage requests");
            let storage = planned_parallel_copied_storage_for_load(
                &ctx,
                &gguf,
                &model,
                &expected,
                embedding_selection,
                population,
                ParallelDestinationLength::LogicalExact,
            )
            .expect("realize parallel dense storage");
            assert_eq!(storage.profile.id, ParallelCopyProfileId::Dense27bQ4kmV1);
            storage
                .validate_source_bytes(&gguf, &expected)
                .expect("audit every parallel resource byte");
            validate_parallel_copied_topology(
                storage.profile,
                storage.destination_length,
                &storage.expected,
                &storage.resources,
                &storage.tensors,
            )
            .expect("validate parallel-copy topology before model construction");
            assert_eq!(
                frozen_parallel_copy_order(storage.profile, &storage.expected)
                    .expect("frozen schedule"),
                storage.sorted_request_indices
            );

            let command = ctx.queue.commandBuffer().expect("write guard command");
            let encoder = crate::metal::KernelEncoder::begin(&command);
            let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                encoder.note_write(&storage.tensors[0]);
            }));
            assert!(
                write_result.is_err(),
                "parallel weights must reject compute writes"
            );
            encoder.end();

            let command = ctx.queue.commandBuffer().expect("blit guard command");
            let blit = crate::metal::BlitEncoder::begin(&command);
            let blit_destination = storage
                .tensors
                .iter()
                .find(|tensor| tensor.dtype == GgmlType::F32 && tensor.n_bytes() <= 1_048_576)
                .expect("small direct F32 candidate weight");
            let writable_source = MetalTensor::zeros_f32(&ctx, blit_destination.shape.clone())
                .expect("equal-sized blit source");
            assert_eq!(writable_source.n_bytes(), blit_destination.n_bytes());
            let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                blit.copy_tensor(&writable_source, blit_destination);
            }));
            assert!(
                write_result.is_err(),
                "parallel weights must reject blit writes"
            );
            blit.end();

            let parallel_model = MetalModel::load_with_direct_storage(
                &ctx,
                &gguf,
                &model,
                ResolvedWeightLoadChoices {
                    embedding_selection,
                    router_f16: false,
                    fused_qkv_g8: false,
                },
                &expected,
                DirectStorage::ForcedParallelCopied(storage),
                false,
            )
            .expect("construct model from audited parallel-copy storage");
            let parallel =
                run_loaded_model_arm(&ctx, parallel_model, &tokens, Some(copied.next_token));
            (copied, parallel)
        });
        let parallel_argmax = parallel
            .prefill_logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                if value > best.1 { (index, value) } else { best }
            })
            .0 as i32;
        assert_eq!(copied.next_token, parallel_argmax);
        assert_eq!(parallel.next_token, parallel_argmax);
        assert_generic_retained_f32_bits(
            "parallel prefill logits",
            &copied.prefill_logits,
            &parallel.prefill_logits,
        );
        assert_generic_retained_snapshot(
            "parallel prefill snapshot",
            &copied.prefill_snapshot,
            &parallel.prefill_snapshot,
        );
        assert_generic_retained_f32_bits(
            "parallel decode logits",
            &copied.decode_logits,
            &parallel.decode_logits,
        );
        assert_generic_retained_snapshot(
            "parallel decode snapshot",
            &copied.decode_snapshot,
            &parallel.decode_snapshot,
        );

        let native_line = format!(
            "[metal-load] native quantized token embedding policy: auto-promoted ({:?} {:?})",
            model.token_embd.dtype, model.token_embd.shape
        );
        let ledger_line = concat!(
            "[metal-load-ledger] source=851/16806250496 ",
            "direct_copy=851/16806250496 direct_view=0/0 direct_alias=0/0 ",
            "tail_fallback=0/0 converted=0/0/0 derived=0/0"
        );
        assert_eq!(load_lines.len(), 5, "recognized load-line count");
        assert_eq!(load_lines[0], native_line, "A native policy line");
        assert_eq!(load_lines[1], ledger_line, "A copied ledger line");
        assert_eq!(load_lines[2], native_line, "B native policy line");
        assert_eq!(load_lines[4], ledger_line, "B copied ledger line");
        assert_eq!(
            load_lines
                .iter()
                .filter(|line| line.starts_with(marker))
                .count(),
            1,
            "candidate marker count"
        );
        assert_eq!(
            load_lines
                .iter()
                .filter(|line| *line == ledger_line)
                .count(),
            2,
            "copied ledger count"
        );

        let marker_prefix = format!(
            "{}{}",
            marker,
            concat!(
                " schema=2 profile=dense27b-q4km-v1 ",
                "resources=851 bytes=16806250496 workers=4 cuts=136,377,618 ",
                "tasks=136,241,241,233 ",
                "worker_bytes=4194110464,4214375808,4204933376,4192830848 ",
                "w0_first=2,output.weight,0,10993888,1042944000 ",
                "w0_last=135,blk.9.ssm_norm.weight,0,4205103840,512 ",
                "w1_first=136,blk.9.ssm_out.weight,0,4205104352,21626880 ",
                "w1_last=379,blk.28.attn_qkv.weight,0,8376472160,43008000 ",
                "w2_first=378,blk.28.ffn_down.weight,0,8419480160,73113600 ",
                "w2_last=618,blk.46.ffn_down.weight,0,12551299936,73113600 ",
                "w3_first=616,blk.46.ffn_gate.weight,0,12624413536,50135040 ",
                "w3_last=844,blk.63.post_attention_norm.weight,0,16817223904,20480 ",
                "create=shared,default_cache,default ",
                "observed=shared,default_cache,tracked ",
                "page=16384 alignment=32 max_buffer=77309411328 ",
                "mapped=16817244384 layout=0xd116405fd99f54d9 ",
                "inventory=50e9af4e4f590fc85687a71f5602ce035e7fdf0e2a31e928b2c7a2be10458a07 "
            )
        );
        assert!(!load_lines[3].contains(" plan="));
        let timing_suffix = load_lines[3]
            .strip_prefix(&marker_prefix)
            .expect("exact dense candidate marker prefix and field order");
        let timing_fields = timing_suffix.split(' ').collect::<Vec<_>>();
        let timing_names = [
            "allocation_us",
            "source_us",
            "copy_us",
            "binding_us",
            "ready_us",
            "user_cpu_us",
            "system_cpu_us",
            "total_cpu_us",
            "timer_minor_faults",
            "timer_major_faults",
            "instructions_delta_raw",
            "cycles_delta_raw",
        ];
        assert_eq!(
            timing_fields.len(),
            timing_names.len(),
            "candidate dynamic field count"
        );
        let mut timing_values = Vec::with_capacity(timing_names.len());
        for (field, name) in timing_fields.iter().zip(timing_names) {
            let value = field
                .strip_prefix(&format!("{name}="))
                .expect("candidate dynamic field name");
            assert!(
                !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
                "candidate dynamic field must be unsigned decimal"
            );
            let parsed = value.parse::<u64>().expect("candidate dynamic value");
            assert_eq!(
                parsed.to_string(),
                value,
                "candidate dynamic canonical form"
            );
            timing_values.push(parsed);
        }
        assert!(
            timing_values[4] > 0,
            "candidate ready time must be positive"
        );
        assert!(
            timing_values[4].abs_diff(timing_values[..4].iter().sum()) <= 4,
            "candidate timing reconciliation"
        );
        assert_eq!(
            timing_values[7],
            timing_values[5] + timing_values[6],
            "candidate CPU reconciliation"
        );
    }

    #[test]
    #[ignore = "requires local Qwen3.6 dense 27B Q4 fixture"]
    fn gguf_parallel_copied_dense27b_q4_is_bit_exact() {
        assert_gguf_parallel_dense27b_q4_is_bit_exact(
            ParallelPopulationMethod::MmapCopy,
            "[metal-gguf-parallel-copied]",
        );
    }

    #[test]
    #[ignore = "requires local Qwen3.6 dense 27B Q4 fixture"]
    fn gguf_parallel_pread_dense27b_q4_is_bit_exact() {
        assert_gguf_parallel_dense27b_q4_is_bit_exact(
            ParallelPopulationMethod::Pread,
            "[metal-gguf-parallel-pread]",
        );
    }

    #[test]
    #[ignore = "requires local split A10B fixture and substantial virtual Metal residency"]
    fn gguf_no_copy_split_a10b_resources_outlive_loader() {
        let model_path = concat!(
            "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/",
            "Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf"
        );
        assert!(std::path::Path::new(model_path).is_file());
        let ctx = MetalContext::new().expect("Metal context");
        let weak_mmaps = objc2::rc::autoreleasepool(|_| {
            let gguf = GgufFile::open(model_path).expect("open split A10B fixture");
            let model = Model::from_gguf(&gguf).expect("bind split A10B fixture");
            assert_eq!(gguf.shard_count(), 3);
            assert!(native_quant_embedding_storage_supported(&model));
            let expected = model_weight_storage_requests(&model, true, false)
                .expect("split A10B storage requests");
            let direct = expected
                .iter()
                .filter(|request| request.kind == ModelWeightStorageKind::Direct)
                .map(|request| request.desc)
                .collect::<Vec<_>>();
            let mut storage = planned_retained_storage_for_load(&ctx, &gguf, &expected, false)
                .expect("realize split A10B retained storage");
            assert_eq!(storage.plan.windows.len(), 2);
            assert_eq!(
                storage
                    .plan
                    .windows
                    .iter()
                    .map(|window| window.length as u64)
                    .sum::<u64>(),
                77_015_662_592
            );
            assert_eq!(storage.plan.entries.len(), 879);
            assert_eq!(storage.plan.unique_view_bytes, 77_015_642_112);
            assert_eq!(storage.plan.alias_bytes, 0);
            assert_eq!(storage.plan.unique_fallback_bytes, 3_354_624);
            assert_eq!(
                storage
                    .plan
                    .entries
                    .iter()
                    .filter(|entry| {
                        matches!(entry.disposition, RetainedStorageDisposition::View { .. })
                    })
                    .count(),
                877
            );
            assert_eq!(
                storage
                    .plan
                    .entries
                    .iter()
                    .filter(|entry| {
                        matches!(
                            entry.disposition,
                            RetainedStorageDisposition::CopyFallback { .. }
                        )
                    })
                    .count(),
                2
            );
            let active_shards = storage
                .plan
                .windows
                .iter()
                .map(|window| window.shard_idx)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(active_shards.len(), 2);

            let mut ledger = WeightLoadLedger::default();
            for desc in &direct {
                let (tensor, materialization) = storage
                    .load_direct(&ctx, &gguf, desc)
                    .expect("materialize split A10B tensor");
                ledger
                    .record_source(desc, materialization, tensor.n_bytes())
                    .expect("record split A10B tensor");
            }
            storage
                .validate_complete(&ledger)
                .expect("complete split A10B realization");

            let mut sample_indices = std::collections::BTreeSet::new();
            for window_index in 0..storage.plan.windows.len() {
                let indices = storage
                    .plan
                    .entries
                    .iter()
                    .enumerate()
                    .filter_map(|(index, entry)| match entry.disposition {
                        RetainedStorageDisposition::View {
                            window_index: entry_window,
                            ..
                        } if entry_window == window_index => Some(index),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert!(!indices.is_empty());
                sample_indices.insert(indices[0]);
                sample_indices.insert(indices[indices.len() / 2]);
                sample_indices.insert(indices[indices.len() - 1]);
            }
            let mut samples = Vec::new();
            for index in sample_indices {
                let desc = direct[index];
                let tensor = storage.realized[index]
                    .as_ref()
                    .expect("realized sample tensor")
                    .clone();
                let sample_len = usize::try_from(desc.n_bytes.min(256)).unwrap();
                let tail_offset = desc.n_bytes - sample_len as u64;
                for byte_offset in [0, tail_offset] {
                    let start = byte_offset as usize;
                    let expected_bytes = gguf.slice(desc)[start..start + sample_len].to_vec();
                    samples.push((tensor.clone(), byte_offset, expected_bytes));
                }
            }
            let weak_mmaps = active_shards
                .iter()
                .map(|&shard_idx| {
                    let mmap = gguf
                        .retained_shard_mmap(shard_idx)
                        .expect("active split A10B shard mmap");
                    std::sync::Arc::downgrade(&mmap)
                })
                .collect::<Vec<_>>();
            drop(direct);
            drop(expected);
            drop(model);
            drop(gguf);
            assert!(weak_mmaps.iter().all(|weak| weak.upgrade().is_some()));
            drop(storage);
            assert!(weak_mmaps.iter().all(|weak| weak.upgrade().is_some()));

            let outputs = samples
                .iter()
                .map(|(_, _, expected)| ctx.buffer_uninit(expected.len()))
                .collect::<Result<Vec<_>, _>>()
                .expect("sample output buffers");
            let command = ctx
                .queue
                .commandBuffer()
                .expect("split A10B sample command");
            let blit = crate::metal::BlitEncoder::begin(&command);
            for ((tensor, byte_offset, expected_bytes), output) in samples.iter().zip(&outputs) {
                blit.copy_buffer(
                    &tensor.buffer,
                    tensor.offset + byte_offset,
                    output,
                    0,
                    expected_bytes.len() as u64,
                );
            }
            blit.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "split A10B sample blit failed");
            for ((_, _, expected), output) in samples.iter().zip(&outputs) {
                let actual = unsafe {
                    std::slice::from_raw_parts(
                        output.contents().as_ptr().cast::<u8>(),
                        expected.len(),
                    )
                };
                assert_eq!(actual, expected);
            }
            weak_mmaps
        });
        assert!(weak_mmaps.iter().all(|weak| weak.upgrade().is_none()));
    }

    #[test]
    #[ignore = "requires QWEN_GGUF_NO_COPY_MODEL local exact 27B fixture"]
    fn gguf_no_copy_layout_digest_probe() {
        let path = std::env::var("QWEN_GGUF_NO_COPY_MODEL")
            .expect("set QWEN_GGUF_NO_COPY_MODEL to the exact 27B GGUF");
        let gguf = GgufFile::open(&path).expect("open exact 27B GGUF");
        eprintln!(
            "[gguf-no-copy-layout] model={path} digest={:#018x}",
            gguf_descriptor_layout_digest(&gguf)
        );
    }

    #[test]
    #[ignore = "requires local exact 27B model and frozen Reva prompt"]
    fn gguf_no_copy_27b_prefill_and_continuation_are_bit_exact() {
        use crate::metal_dflash::{
            MetalDFlashLayerMajorScratch, PrefillScratchConfig,
            plan_prefill_scratch_with_matrix_max_pos_configured, prefill_tokens_with_multi_hidden,
        };

        struct ArmResult {
            prefill_logits: Vec<f32>,
            prefill_snapshot: SessionSnapshot,
            next_token: i32,
            decode_logits: Vec<f32>,
            decode_snapshot: SessionSnapshot,
        }

        fn run_arm(
            ctx: &MetalContext,
            gguf: &GgufFile,
            model: &Model<'_>,
            tokens: &[i32],
            mode: GgufNoCopyMode,
            prefault_enabled: bool,
            forced_next: Option<i32>,
        ) -> ArmResult {
            let metal_model =
                MetalModel::load_with_no_copy_policy(ctx, gguf, model, mode, prefault_enabled)
                    .expect("load exactness arm");
            let forward = MetalForward::new(ctx, &metal_model);
            let capacity = 512;
            let mut session =
                MetalSession::fresh(ctx, &metal_model, capacity).expect("fresh session");
            let chunk = tokens.len() as u32;
            let plan = plan_prefill_scratch_with_matrix_max_pos_configured(
                &metal_model,
                chunk,
                capacity,
                PrefillScratchConfig::default(),
            )
            .expect("prefill scratch plan");
            let mut scratch =
                MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(ctx, &metal_model, plan)
                    .expect("prefill scratch");
            let prefill_logits = prefill_tokens_with_multi_hidden(
                &forward,
                tokens,
                0,
                &mut session,
                &mut scratch,
                &[],
                None,
            )
            .expect("packed prefill");
            let identity = session.snapshot_identity(0x591, 0x27b);
            let prefill_snapshot = session
                .snapshot(
                    identity.clone(),
                    tokens.to_vec(),
                    Some(prefill_logits.clone()),
                )
                .expect("prefill snapshot");
            let next_token = forced_next.unwrap_or_else(|| {
                prefill_logits
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                        if value > best.1 { (index, value) } else { best }
                    })
                    .0 as i32
            });
            let decode_logits = forward
                .single_token(next_token, tokens.len() as u32, &mut session)
                .expect("forced decode transition");
            let mut consumed = tokens.to_vec();
            consumed.push(next_token);
            let decode_snapshot = session
                .snapshot(identity, consumed, Some(decode_logits.clone()))
                .expect("decode snapshot");
            ArmResult {
                prefill_logits,
                prefill_snapshot,
                next_token,
                decode_logits,
                decode_snapshot,
            }
        }

        fn assert_f32_bits(label: &str, a: &[f32], b: &[f32]) {
            assert_eq!(a.len(), b.len(), "{label} length");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{label} bit mismatch at {index}");
            }
        }

        fn assert_snapshot(label: &str, a: &SessionSnapshot, b: &SessionSnapshot) {
            assert_eq!(a.identity, b.identity, "{label} identity");
            assert_eq!(a.prefix_tokens, b.prefix_tokens, "{label} tokens");
            assert_eq!(a.kv_n_pos, b.kv_n_pos, "{label} KV positions");
            assert_eq!(a.kv_k_arena, b.kv_k_arena, "{label} K arena");
            assert_eq!(a.kv_v_arena, b.kv_v_arena, "{label} V arena");
            assert_eq!(a.gdn_conv_arena, b.gdn_conv_arena, "{label} conv arena");
            assert_eq!(a.gdn_state_arena, b.gdn_state_arena, "{label} state arena");
            match (&a.final_logits, &b.final_logits) {
                (Some(a), Some(b)) => assert_f32_bits(&format!("{label} logits"), a, b),
                (None, None) => {}
                _ => panic!("{label} final-logits presence mismatch"),
            }
        }

        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        let prompt_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt");
        assert!(
            std::path::Path::new(model_path).is_file(),
            "missing exact 27B fixture {model_path}"
        );
        assert!(
            prompt_path.is_file(),
            "missing frozen prompt {}",
            prompt_path.display()
        );
        let ctx = MetalContext::new().expect("Metal context");
        let gguf = GgufFile::open(model_path).expect("open exact 27B GGUF");
        let model = Model::from_gguf(&gguf).expect("bind exact 27B model");
        assert!(matches_no_copy_27b_sentinel(&gguf, &model));
        let converted_embedding_requests =
            model_weight_storage_requests(&model, false, false).expect("converted request plan");
        let rollback_error = match direct_storage_for_load(
            &ctx,
            &gguf,
            &model,
            &converted_embedding_requests,
            GgufNoCopyMode::Forced,
            false,
            GgufOwnedArenaMode::Disabled,
            GgufParallelCopyMode::Disabled,
            PreparedAutoSelection::NotEligible,
            NativeQuantEmbeddingSelection::AutoUnpromoted,
        ) {
            Ok(_) => panic!("exact 27B rollback must fail before resource realization"),
            Err(error) => error,
        };
        assert!(
            rollback_error
                .to_string()
                .contains("requires native token embedding residency")
        );
        let tokenizer = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
        let prompt = std::fs::read_to_string(prompt_path).expect("read frozen prompt");
        let tokens = tokenizer
            .encode(&prompt, true)
            .expect("tokenize frozen prompt");
        assert_eq!(tokens.len(), 419, "frozen Reva token count drifted");

        let copied = run_arm(
            &ctx,
            &gguf,
            &model,
            &tokens,
            GgufNoCopyMode::Disabled,
            true,
            None,
        );
        let retained = run_arm(
            &ctx,
            &gguf,
            &model,
            &tokens,
            GgufNoCopyMode::Forced,
            false,
            Some(copied.next_token),
        );
        assert_eq!(copied.next_token, retained.next_token);
        assert_f32_bits(
            "prefill logits",
            &copied.prefill_logits,
            &retained.prefill_logits,
        );
        assert_snapshot(
            "prefill snapshot",
            &copied.prefill_snapshot,
            &retained.prefill_snapshot,
        );
        assert_f32_bits(
            "decode logits",
            &copied.decode_logits,
            &retained.decode_logits,
        );
        assert_snapshot(
            "decode snapshot",
            &copied.decode_snapshot,
            &retained.decode_snapshot,
        );
    }

    #[test]
    fn native_quant_embedding_support_requires_an_aligned_matrix() {
        assert!(native_quant_embedding_supported(
            GgmlType::Q4_K,
            &[5120, 248_320]
        ));
        assert!(native_quant_embedding_supported(
            GgmlType::Q8_0,
            &[2048, 248_320]
        ));
        assert!(native_quant_embedding_supported(
            GgmlType::Q8_0,
            &[3072, 248_320]
        ));
        assert!(!native_quant_embedding_supported(
            GgmlType::Q6_K,
            &[5120, 248_320]
        ));
        assert!(!native_quant_embedding_supported(
            GgmlType::Q4_K,
            &[5119, 248_320]
        ));
        assert!(!native_quant_embedding_supported(
            GgmlType::Q8_0,
            &[2047, 248_320]
        ));
        assert!(!native_quant_embedding_supported(GgmlType::Q8_0, &[2048]));
        assert!(!native_quant_embedding_supported(
            GgmlType::Q8_0,
            &[0, 248_320]
        ));
    }

    #[test]
    fn native_quant_embedding_mode_is_strict_and_tri_state() {
        assert_eq!(
            parse_native_quant_embedding_mode(None),
            NativeQuantEmbeddingMode::Auto
        );
        for value in ["1", "true", "TRUE", "yes", "YES"] {
            assert_eq!(
                parse_native_quant_embedding_mode(Some(value)),
                NativeQuantEmbeddingMode::Forced
            );
        }
        for value in ["0", "false", "FALSE", "no", "NO"] {
            assert_eq!(
                parse_native_quant_embedding_mode(Some(value)),
                NativeQuantEmbeddingMode::Disabled
            );
        }
        for value in ["", "on", "off", "ture", "2"] {
            assert_eq!(
                parse_native_quant_embedding_mode(Some(value)),
                NativeQuantEmbeddingMode::Invalid
            );
        }
    }

    #[test]
    fn native_quant_embedding_resolution_preserves_force_and_rollback() {
        use NativeQuantEmbeddingMode::{Auto, Disabled, Forced, Invalid};
        use NativeQuantEmbeddingSelection::Unsupported;
        use NativeQuantEmbeddingSelection::{
            AutoPromoted, AutoUnpromoted, Forced as On, InvalidDisabled, RollbackDisabled,
        };

        assert_eq!(
            resolve_native_quant_embedding(Auto, true, true),
            AutoPromoted
        );
        assert_eq!(
            resolve_native_quant_embedding(Auto, true, false),
            AutoUnpromoted
        );
        assert_eq!(resolve_native_quant_embedding(Forced, true, false), On);
        assert_eq!(
            resolve_native_quant_embedding(Disabled, true, true),
            RollbackDisabled
        );
        assert_eq!(
            resolve_native_quant_embedding(Invalid, true, true),
            InvalidDisabled
        );
        assert_eq!(
            resolve_native_quant_embedding(Forced, false, true),
            Unsupported
        );
    }

    #[test]
    fn native_quant_embedding_defaults_only_on_promoted_fingerprints() {
        let mut dense = crate::model::QWEN3_27B;
        dense.mtp_n_hidden_layers = 0;
        assert!(native_quant_embedding_default_promoted(
            &dense,
            false,
            false,
            GgmlType::Q4_K,
            &[5120, 248_320],
        ));
        assert!(!native_quant_embedding_default_promoted(
            &dense,
            true,
            false,
            GgmlType::Q4_K,
            &[5120, 248_320],
        ));
        assert!(!native_quant_embedding_default_promoted(
            &dense,
            false,
            true,
            GgmlType::Q4_K,
            &[5120, 248_320],
        ));
        assert!(!native_quant_embedding_default_promoted(
            &dense,
            false,
            false,
            GgmlType::Q8_0,
            &[5120, 248_320],
        ));
        dense.mtp_n_hidden_layers = 1;
        assert!(!native_quant_embedding_default_promoted(
            &dense,
            false,
            false,
            GgmlType::Q4_K,
            &[5120, 248_320],
        ));

        let a3b = crate::model::Arch {
            kind: ArchKind::Moe,
            n_layer: 40,
            hidden_size: 2048,
            intermediate_size: 0,
            vocab_size: 248_320,
            full_attention_interval: 4,
            n_q_heads: 16,
            n_kv_heads: 2,
            attn_head_dim: 256,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            gdn_n_v_heads: 32,
            gdn_n_k_heads: 16,
            gdn_head_dim: 128,
            gdn_conv_kernel: 4,
            expert_count: 256,
            expert_used_count: 8,
            expert_feed_forward_length: 512,
            expert_shared_feed_forward_length: 512,
            mtp_n_hidden_layers: 0,
        };
        assert!(native_quant_embedding_default_promoted(
            &a3b,
            false,
            false,
            GgmlType::Q8_0,
            &[2048, 248_320],
        ));
        assert!(!native_quant_embedding_default_promoted(
            &a3b,
            true,
            false,
            GgmlType::Q8_0,
            &[2048, 248_320],
        ));
        assert!(!native_quant_embedding_default_promoted(
            &a3b,
            false,
            true,
            GgmlType::Q8_0,
            &[2048, 248_320],
        ));
        assert!(!native_quant_embedding_default_promoted(
            &a3b,
            false,
            false,
            GgmlType::Q4_K,
            &[2048, 248_320],
        ));
        assert!(!native_quant_embedding_default_promoted(
            &a3b,
            false,
            false,
            GgmlType::Q8_0,
            &[3072, 248_320],
        ));
    }

    #[test]
    #[ignore = "requires QWEN_EMBED_RESIDENCY_MODEL local fixture"]
    fn quantized_embedding_residency_load_probe() {
        let path = std::env::var("QWEN_EMBED_RESIDENCY_MODEL")
            .expect("set QWEN_EMBED_RESIDENCY_MODEL to a local GGUF");
        let ctx = MetalContext::new().expect("metal context");
        let g = GgufFile::open(&path).expect("open fixture");
        let m = Model::from_gguf(&g).expect("bind model");
        let source = m.token_embd;
        assert!(
            !matches!(source.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16),
            "probe requires a quantized token embedding"
        );
        let f32_bytes = source.n_elements().checked_mul(4).expect("F32 byte size");
        let allocated_before = ctx.current_allocated_size();
        let started = std::time::Instant::now();
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let load_ms = started.elapsed().as_secs_f64() * 1e3;
        let allocated_after = ctx.current_allocated_size();
        let allocated_delta = allocated_after.saturating_sub(allocated_before);
        let resident_bytes = mm.token_embd.n_bytes();
        let backing_bytes = mm.token_embd.buffer.length() as u64;
        let selection = resolve_native_quant_embedding(
            native_quant_embedding_mode(),
            native_quant_embedding_supported(source.dtype, &source.shape),
            native_quant_embedding_default_promoted(
                &m.arch,
                m.tied_embeddings,
                m.mtp.is_some(),
                source.dtype,
                &source.shape,
            ),
        );
        let expect_native = selection.uses_native();

        assert_eq!(mm.token_embd.dtype == source.dtype, expect_native);
        assert!(
            mm.token_embd.offset + resident_bytes <= backing_bytes,
            "logical embedding range must fit its backing buffer"
        );
        if expect_native {
            assert_eq!(resident_bytes, source.n_bytes);
            let theoretical_savings = f32_bytes - source.n_bytes;
            let observed_savings = f32_bytes - resident_bytes;
            assert!(observed_savings * 100 >= theoretical_savings * 95);
        } else if !matches!(source.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            assert_eq!(mm.token_embd.dtype, GgmlType::F32);
            assert_eq!(resident_bytes, f32_bytes);
        }
        eprintln!(
            concat!(
                "[embed-residency] model={} source={:?} resident={:?} ",
                "source_bytes={} resident_bytes={} backing_bytes={} f32_bytes={} device_delta={} ",
                "load_ms={:.3}"
            ),
            path,
            source.dtype,
            mm.token_embd.dtype,
            source.n_bytes,
            resident_bytes,
            backing_bytes,
            f32_bytes,
            allocated_delta,
            load_ms,
        );
    }

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
        let Some(ctx) = metal_test_context() else {
            return;
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

    fn run_exact_greedy_chain_equivalence(model_path: &str, label: &str) {
        if !std::path::Path::new(model_path).exists() {
            eprintln!("[greedy-chain-{label}] skipped - fixture missing");
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
        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
        let mut prompt = tok
            .encode("The quick brown fox jumps over the lazy dog", false)
            .expect("tokenize");
        prompt.truncate(prompt.len().min(4));
        assert!(!prompt.is_empty(), "tokenizer returned empty prompt");

        let select = |logits: &[f32]| {
            let mut sampler =
                crate::sampling::Sampler::new(crate::sampling::SamplingConfig::default())
                    .expect("greedy sampler");
            sampler.sample(logits).expect("valid logits").token
        };
        let mf = MetalForward::new(&ctx, &mm);
        let capacity = prompt.len() + 8;
        let mut full = MetalSession::fresh(&ctx, &mm, capacity).expect("full session");
        let mut greedy = MetalSession::fresh(&ctx, &mm, capacity).expect("greedy session");
        let mut consumed = Vec::new();
        let mut full_logits = Vec::new();

        for (position, &token) in prompt.iter().enumerate() {
            full_logits = mf
                .single_token(token, position as u32, &mut full)
                .expect("full prompt step");
            let selected = mf
                .single_token_greedy(token, position as u32, &mut greedy)
                .expect("greedy prompt step")
                .into_token()
                .expect("finite greedy prompt logits");
            assert_eq!(
                selected,
                select(&full_logits),
                "[greedy-chain-{label}] prompt selection at {position}"
            );
            consumed.push(token);
        }

        for step in 0..4 {
            let token = select(&full_logits);
            let position = consumed.len() as u32;
            full_logits = mf
                .single_token(token, position, &mut full)
                .expect("full generation step");
            let selected = mf
                .single_token_greedy(token, position, &mut greedy)
                .expect("greedy generation step")
                .into_token()
                .expect("finite greedy generation logits");
            assert_eq!(
                selected,
                select(&full_logits),
                "[greedy-chain-{label}] generation selection at {step}"
            );
            consumed.push(token);
        }

        let token = select(&full_logits);
        let position = consumed.len() as u32;
        let full_continuation = mf
            .single_token(token, position, &mut full)
            .expect("full continuation");
        let greedy_continuation = mf
            .single_token(token, position, &mut greedy)
            .expect("greedy continuation");
        consumed.push(token);
        assert!(
            full_continuation
                .iter()
                .zip(&greedy_continuation)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "[greedy-chain-{label}] continuation logits differ"
        );

        let identity = full.snapshot_identity(1, 2);
        let full_snapshot = full
            .snapshot(identity.clone(), consumed.clone(), None)
            .expect("full snapshot");
        let greedy_snapshot = greedy
            .snapshot(identity, consumed, None)
            .expect("greedy snapshot");
        assert_eq!(full_snapshot.identity, greedy_snapshot.identity);
        assert_eq!(full_snapshot.prefix_tokens, greedy_snapshot.prefix_tokens);
        assert_eq!(full_snapshot.pending_token, greedy_snapshot.pending_token);
        assert_eq!(full_snapshot.kv_n_pos, greedy_snapshot.kv_n_pos);
        assert_eq!(full_snapshot.kv_k_arena, greedy_snapshot.kv_k_arena);
        assert_eq!(full_snapshot.kv_v_arena, greedy_snapshot.kv_v_arena);
        assert_eq!(full_snapshot.gdn_conv_arena, greedy_snapshot.gdn_conv_arena);
        assert_eq!(
            full_snapshot.gdn_state_arena,
            greedy_snapshot.gdn_state_arena
        );
        eprintln!("[greedy-chain-{label}] exact-state PASS");
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
        let Some(ctx) = metal_test_context() else {
            return;
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
            .single_token_argmax_profiled_dense_serial(
                ids[0],
                0,
                &mut s_serial_argmax,
                ArgmaxReduction::SpeculativeLowest,
            )
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
    #[ignore = "requires the 22 GB A3B fixture and Metal GPU"]
    fn metal_sampled_attribution_matches_production_a3b() {
        let model_path = "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
        let metadata = std::fs::metadata(model_path).expect("required A3B fixture is missing");
        assert_eq!(metadata.len(), 22_134_528_992, "A3B fixture size changed");
        let mut file = std::fs::File::open(model_path).expect("open A3B for authentication");
        let mut digest = Sha256::new();
        let mut bytes = vec![0u8; 16 * 1024 * 1024];
        loop {
            let count =
                std::io::Read::read(&mut file, &mut bytes).expect("hash authenticated A3B fixture");
            if count == 0 {
                break;
            }
            digest.update(&bytes[..count]);
        }
        assert_eq!(
            format!("{:x}", digest.finalize()),
            "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
            "A3B fixture SHA-256 changed"
        );
        let ctx = MetalContext::new().expect("Metal context");
        let gguf = GgufFile::open(model_path).expect("open A3B");
        let model = Model::from_gguf(&gguf).expect("parse A3B");
        let metal = MetalModel::load(&ctx, &gguf, &model).expect("load A3B");
        let tokenizer = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
        let ids = tokenizer.encode("Hello", false).expect("tokenize");
        assert_eq!(ids.len(), 1);

        let forward = MetalForward::new(&ctx, &metal);
        let mut ordinary = MetalSession::fresh(&ctx, &metal, 4).expect("ordinary session");
        let mut profiled = MetalSession::fresh(&ctx, &metal, 4).expect("profiled session");
        let (ordinary_logits, _) = forward
            .single_token_profiled_concurrent_gdn_moe(ids[0], 0, &mut ordinary)
            .expect("ordinary transition");
        let (profiled_logits, _, readback) = forward
            .single_token_sampled_attribution(ids[0], 0, &mut profiled)
            .expect("profiled transition");
        assert_eq!(readback.timer_spans, 2);
        assert_eq!(
            readback.bytes,
            ordinary_logits.len() * std::mem::size_of::<f32>()
        );
        assert!(readback.allocation_zero_fill_ms >= 0.0 && readback.copy_ms >= 0.0);
        assert!(
            ordinary_logits
                .iter()
                .zip(&profiled_logits)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "attributed transition logits differ"
        );

        let identity = ordinary.snapshot_identity(1, 2);
        let ordinary_snapshot = ordinary
            .snapshot(identity.clone(), ids.clone(), None)
            .expect("ordinary snapshot");
        let profiled_snapshot = profiled
            .snapshot(identity.clone(), ids.clone(), None)
            .expect("profiled snapshot");
        assert_eq!(ordinary_snapshot.kv_n_pos, profiled_snapshot.kv_n_pos);
        assert_eq!(ordinary_snapshot.kv_k_arena, profiled_snapshot.kv_k_arena);
        assert_eq!(ordinary_snapshot.kv_v_arena, profiled_snapshot.kv_v_arena);
        assert_eq!(
            ordinary_snapshot.gdn_conv_arena,
            profiled_snapshot.gdn_conv_arena
        );
        assert_eq!(
            ordinary_snapshot.gdn_state_arena,
            profiled_snapshot.gdn_state_arena
        );

        let next = argmax_i32_local(&ordinary_logits);
        let ordinary_continuation = forward
            .single_token_profiled_concurrent_gdn_moe(next, 1, &mut ordinary)
            .expect("ordinary continuation")
            .0;
        let profiled_continuation = forward
            .single_token_profiled_concurrent_gdn_moe(next, 1, &mut profiled)
            .expect("profiled continuation")
            .0;
        assert!(
            ordinary_continuation
                .iter()
                .zip(&profiled_continuation)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "continuation logits differ"
        );
        let prefix = vec![ids[0], next];
        let ordinary_snapshot = ordinary
            .snapshot(identity.clone(), prefix.clone(), None)
            .expect("ordinary continuation snapshot");
        let profiled_snapshot = profiled
            .snapshot(identity, prefix, None)
            .expect("profiled continuation snapshot");
        assert_eq!(ordinary_snapshot.kv_n_pos, profiled_snapshot.kv_n_pos);
        assert_eq!(ordinary_snapshot.kv_k_arena, profiled_snapshot.kv_k_arena);
        assert_eq!(ordinary_snapshot.kv_v_arena, profiled_snapshot.kv_v_arena);
        assert_eq!(
            ordinary_snapshot.gdn_conv_arena,
            profiled_snapshot.gdn_conv_arena
        );
        assert_eq!(
            ordinary_snapshot.gdn_state_arena,
            profiled_snapshot.gdn_state_arena
        );
        eprintln!("[sampling-attribution-a3b] exact-state PASS");
    }

    #[test]
    #[ignore = "requires the 22 GB A3B fixture, frozen prompt, and Metal GPU"]
    fn metal_sampled_structural_matches_copied_a3b() {
        use crate::metal_dflash::{
            MetalDFlashLayerMajorScratch, PrefillScratchConfig,
            plan_prefill_scratch_with_matrix_max_pos_configured, prefill_tokens_with_multi_hidden,
        };

        let model_path = "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
        let metadata = std::fs::metadata(model_path).expect("required A3B fixture is missing");
        assert_eq!(metadata.len(), 22_134_528_992, "A3B fixture size changed");
        let mut file = std::fs::File::open(model_path).expect("open A3B for authentication");
        let mut digest = Sha256::new();
        let mut bytes = vec![0u8; 16 * 1024 * 1024];
        loop {
            let count =
                std::io::Read::read(&mut file, &mut bytes).expect("hash authenticated A3B fixture");
            if count == 0 {
                break;
            }
            digest.update(&bytes[..count]);
        }
        assert_eq!(
            format!("{:x}", digest.finalize()),
            "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
            "A3B fixture SHA-256 changed"
        );

        let prompt_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt");
        let prompt = std::fs::read_to_string(&prompt_path).expect("read frozen Reva prompt");
        assert_eq!(prompt.len(), 1_891, "frozen prompt length changed");
        assert_eq!(
            format!("{:x}", Sha256::digest(prompt.as_bytes())),
            "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474",
            "frozen prompt SHA-256 changed"
        );

        let ctx = MetalContext::new().expect("Metal context");
        let gguf = GgufFile::open(model_path).expect("open A3B");
        let model = Model::from_gguf(&gguf).expect("parse A3B");
        let metal = MetalModel::load(&ctx, &gguf, &model).expect("load A3B");
        let tokenizer = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
        let ids = tokenizer
            .encode(&prompt, true)
            .expect("tokenize frozen prompt");
        assert_eq!(ids.len(), 419, "frozen prompt token count changed");
        assert_eq!(
            crate::tokenizer::token_ids_sha256_i32le(&ids),
            "fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f",
            "frozen prompt token identity changed"
        );

        let forward = MetalForward::new(&ctx, &metal);
        forward
            .ensure_sampled_structural_supported()
            .expect("sampled structural support");
        let capacity = 1_024;
        let mut ordinary = MetalSession::fresh(&ctx, &metal, capacity).expect("ordinary session");
        let mut structural =
            MetalSession::fresh(&ctx, &metal, capacity).expect("structural session");
        forward
            .ensure_sampled_structural_session_supported(&ordinary)
            .expect("ordinary session row support");
        forward
            .ensure_sampled_structural_session_supported(&structural)
            .expect("structural session row support");
        let mut invalid = MetalSession::fresh(&ctx, &metal, capacity).expect("validation session");
        let valid_logits = invalid.logits.clone();
        invalid.logits.dtype = GgmlType::F16;
        assert!(
            forward
                .ensure_sampled_structural_session_supported(&invalid)
                .is_err(),
            "wrong logits dtype must fail preflight"
        );
        invalid.logits = valid_logits.clone();
        invalid.logits.shape = vec![248_319];
        assert!(
            forward
                .ensure_sampled_structural_session_supported(&invalid)
                .is_err(),
            "wrong logits shape must fail preflight"
        );
        invalid.logits = valid_logits.clone();
        invalid.logits.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        assert!(
            forward
                .ensure_sampled_structural_session_supported(&invalid)
                .is_err(),
            "read-only logits provenance must fail preflight"
        );
        invalid.logits = valid_logits.clone();
        invalid.logits.offset = std::mem::size_of::<f32>() as u64;
        assert!(
            forward
                .ensure_sampled_structural_session_supported(&invalid)
                .is_err(),
            "out-of-bounds logits range must fail preflight"
        );
        let mut misaligned = MetalTensor::zeros_f32(&ctx, vec![248_321])
            .expect("oversized logits alignment fixture");
        misaligned.shape = vec![248_320];
        misaligned.offset = 1;
        invalid.logits = misaligned;
        assert!(
            forward
                .ensure_sampled_structural_session_supported(&invalid)
                .is_err(),
            "misaligned logits address must fail preflight"
        );
        let plan = plan_prefill_scratch_with_matrix_max_pos_configured(
            &metal,
            1_024,
            capacity,
            PrefillScratchConfig::default(),
        )
        .expect("prefill scratch plan");
        let mut ordinary_scratch =
            MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(&ctx, &metal, plan.clone())
                .expect("ordinary prefill scratch");
        let mut structural_scratch =
            MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(&ctx, &metal, plan)
                .expect("structural prefill scratch");
        let ordinary_prompt_logits = prefill_tokens_with_multi_hidden(
            &forward,
            &ids,
            0,
            &mut ordinary,
            &mut ordinary_scratch,
            &[],
            None,
        )
        .expect("ordinary prompt prefill");
        let structural_prompt_logits = prefill_tokens_with_multi_hidden(
            &forward,
            &ids,
            0,
            &mut structural,
            &mut structural_scratch,
            &[],
            None,
        )
        .expect("structural prompt prefill");
        assert_generic_retained_f32_bits(
            "sampled structural prompt logits",
            &ordinary_prompt_logits,
            &structural_prompt_logits,
        );

        let config = SamplingConfig::qwen_chat(42);
        assert_eq!(config.temperature.to_bits(), 0.7f32.to_bits());
        assert_eq!(config.top_k, 200);
        assert_eq!(config.top_p.to_bits(), 1.0f32.to_bits());
        assert_eq!(config.min_p.to_bits(), 0.05f32.to_bits());
        assert_eq!(config.seed, 42);
        let mut ordinary_sampler = Sampler::new(config).expect("ordinary sampler");
        let mut structural_sampler = ordinary_sampler.clone();
        let ordinary_initial = ordinary_sampler
            .sample(&ordinary_prompt_logits)
            .expect("ordinary prompt sample");
        let (structural_initial, prompt_evidence) = structural_sampler
            .sample_bounded_top_k(&structural_prompt_logits)
            .expect("structural prompt sample");
        assert!(prompt_evidence.used_bounded_path);
        assert_eq!(ordinary_initial, structural_initial);
        assert_eq!(ordinary_sampler.draws(), structural_sampler.draws());

        let mut current = ordinary_initial;
        let mut consumed = ids.clone();
        let mut resident_head_wait_calls = 0u64;
        let mut validated_shared_row_calls = 0u64;
        for step in 0..127usize {
            let position = ids.len() + step;
            let (ordinary_logits, _) = forward
                .single_token_profiled_concurrent_gdn_moe(
                    current.token,
                    position as u32,
                    &mut ordinary,
                )
                .expect("ordinary sampled transition");
            let ordinary_next = ordinary_sampler
                .sample(&ordinary_logits)
                .expect("ordinary transition sample");
            let (structural_next, _, evidence) = forward
                .single_token_sampled_structural_scoped(
                    current.token,
                    position as u32,
                    &mut structural,
                    |row| {
                        assert_generic_retained_f32_bits(
                            "sampled structural transition logits",
                            &ordinary_logits,
                            row,
                        );
                        structural_sampler.sample_bounded_top_k(row)
                    },
                )
                .expect("structural sampled transition");
            let (structural_next, bounded) = structural_next.expect("structural transition sample");
            assert!(bounded.used_bounded_path);
            assert_eq!(evidence.resident_head_wait_calls, 1);
            assert_eq!(evidence.validated_shared_row_calls, 1);
            assert_eq!(evidence.transition_logits_copy_bytes, 0);
            assert_eq!(evidence.extra_command_buffers, 0);
            assert_eq!(evidence.gpu_sampling_dispatches, 0);
            assert_eq!(ordinary_next, structural_next, "sample mismatch at {step}");
            assert_eq!(
                ordinary_sampler.draws(),
                structural_sampler.draws(),
                "draw mismatch at {step}"
            );
            resident_head_wait_calls += evidence.resident_head_wait_calls;
            validated_shared_row_calls += evidence.validated_shared_row_calls;
            consumed.push(current.token);
            current = ordinary_next;
        }
        assert_eq!(resident_head_wait_calls, 127);
        assert_eq!(validated_shared_row_calls, 127);
        assert_eq!(consumed.len(), ids.len() + 127);

        let identity = ordinary.snapshot_identity(0x660, 0x660);
        let ordinary_snapshot = ordinary
            .snapshot(identity.clone(), consumed.clone(), None)
            .expect("ordinary sampled snapshot");
        let structural_snapshot = structural
            .snapshot(identity.clone(), consumed.clone(), None)
            .expect("structural sampled snapshot");
        assert_generic_retained_snapshot(
            "sampled structural state",
            &ordinary_snapshot,
            &structural_snapshot,
        );

        let continuation_position = consumed.len() as u32;
        let ordinary_continuation = forward
            .single_token_profiled_concurrent_gdn_moe(
                current.token,
                continuation_position,
                &mut ordinary,
            )
            .expect("ordinary continuation")
            .0;
        let structural_continuation = forward
            .single_token_profiled_concurrent_gdn_moe(
                current.token,
                continuation_position,
                &mut structural,
            )
            .expect("structural continuation")
            .0;
        assert_generic_retained_f32_bits(
            "sampled structural continuation logits",
            &ordinary_continuation,
            &structural_continuation,
        );
        consumed.push(current.token);
        let ordinary_snapshot = ordinary
            .snapshot(identity.clone(), consumed.clone(), None)
            .expect("ordinary continuation snapshot");
        let structural_snapshot = structural
            .snapshot(identity, consumed, None)
            .expect("structural continuation snapshot");
        assert_generic_retained_snapshot(
            "sampled structural continuation state",
            &ordinary_snapshot,
            &structural_snapshot,
        );
        eprintln!(
            "[sampled-structural-a3b] exact rows/state PASS \
             prompt_token_sha256=fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f"
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

    #[test]
    fn metal_exact_greedy_chain_matches_full_logits_dense() {
        run_exact_greedy_chain_equivalence(
            "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf",
            "dense-27b",
        );
    }

    #[test]
    fn metal_exact_greedy_chain_matches_full_logits_moe() {
        run_exact_greedy_chain_equivalence(
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "moe-a3b",
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
        //     ssm_norm, q_norm, k_norm, output_norm; token_embd is F32 by
        //     default and Q4_K/Q8_0-native under QWEN_NATIVE_QUANT_EMBED.
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
            "token_embd (F32 default; Q4_K/Q8_0 native opt-in)",
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
        // v0.432: default path skips split_q_gate (strided q-norm +
        // strided gate sigmoid_mul read the interleave directly); the
        // phase name is kept only for the rollback branch.
        if !decode_attn_sigmoid_mul_enabled() {
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
        }
        // Q-norm.
        timed(
            "q_norm (batched rms)",
            &|enc| {
                if decode_attn_sigmoid_mul_enabled() {
                    encode_rms_norm_batched_src_strided_f32(
                        mf.ctx,
                        enc,
                        &s.attn_q_full,
                        &ab.q_norm,
                        &s.attn_q_normed,
                        n_q,
                        head_dim,
                        2 * head_dim,
                        0,
                        RMS_EPS,
                    )
                    .map_err(MfError::from)
                } else {
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
                }
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
                    encode_sigmoid_mul_gate_strided_f32(
                        mf.ctx,
                        enc,
                        &s.attn_q_full,
                        &s.attn_o,
                        &s.attn_o,
                        n_q,
                        head_dim,
                        2 * head_dim,
                        head_dim,
                    )
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
                    RMS_EPS * head_dim as f32,
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
            GgmlType::Q4_K => {
                timed(
                    "routed_down_q4_K",
                    &|enc| {
                        encode_moe_down_q4_K_f32(
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
            GgmlType::F32 => {
                timed(
                    "routed_down_f32",
                    &|enc| {
                        encode_moe_down_f32_f32(
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
        let Some(ctx) = metal_test_context() else {
            return;
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
            for dv in 0..head_dim {
                let mut sm = 0.0f32;
                for dk in 0..head_dim {
                    sm += state.ssm[0][s_off + dv * head_dim + dk] * q_h[dk];
                }
                o[hi * head_dim + dv] = sm;
            }
        }

        // RMSNormGated: norm(o) * silu(z), per-head.
        let norm_w = crate::codec::dequant_to_f32(gb.norm, f.gguf.slice(gb.norm)).unwrap();
        let mut gated = vec![0.0f32; v_dim];
        for hi in 0..n_v {
            let off = hi * head_dim;
            let normed = crate::forward::rms_norm_pub(
                &o[off..off + head_dim],
                &norm_w,
                super::RMS_EPS * head_dim as f32,
            );
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
            let snap = sess_pre
                .snapshot(identity.clone(), prefix_tokens, Some(last_pre_logits))
                .expect("snapshot");
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
            sess_restored
                .restore_from(&snap, &identity)
                .expect("restore");
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
                kv_storage_kind: SnapshotKvStorageKind::None,
                gdn_state_elements_per_layer: 0,
                gdn_conv_elements_per_layer: 0,
            };
            let bad_snap = SessionSnapshot {
                identity: bogus_identity,
                prefix_tokens: vec![],
                pending_token: None,
                kv_n_pos: vec![],
                kv_k_arena: vec![],
                kv_v_arena: vec![],
                gdn_conv_arena: vec![],
                gdn_state_arena: vec![],
                final_logits: None,
            };
            let mut s2 = sess;
            assert!(
                s2.restore_from(&bad_snap, &s2.snapshot_identity(1, 1))
                    .is_err(),
                "identity mismatch must error"
            );
            eprintln!("[h2-arena]   identity-mismatch refusal: ✓");
        }

        eprintln!("[h2-arena] all prefix splits passed via packed-arena API.");
    }
}
