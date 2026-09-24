//! Product fixed-width MoE decode selected from model capabilities.
//!
//! This intentionally mirrors the measured B=16 composition rather than
//! extending the dense batching machinery. Causal state, routing decisions,
//! expert slot order, and all down/final waves remain sequence-owned.

use crate::env_flag::read_default_off;
use crate::metal::{
    BlitEncoder, KernelEncoder, MetalContext, MetalError, MetalTensor, encode_argmax_f32_greedy,
    encode_get_rows_f32, encode_mat_vec_q6_k_batch_f32, encode_mat_vec_q8_0_batch_f32,
    encode_moe_swiglu_q4_K_f32_packed_slots, encode_rms_norm_mul_f32, mat_vec_q8_0_lcpp_enabled,
};
use crate::metal_forward::{
    ATTN_V4_MAX_NWG, MetalBlock, MetalForward, MetalGdnBlock, MetalModel, MetalSession, MfError,
    RMS_EPS, encode_mat_vec_dispatch,
};
use crate::model::{Arch, ArchKind};
use crate::sampling::{GreedySelection, SamplingError};
use crate::tensor::{GgmlType, ggml_type_layout};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};

pub const MOE_BATCH16_WIDTH: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GdnMixerMode {
    BatchedQ8,
    PerLane,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutedGateUpMode {
    PackedQ4,
    PerLane,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeadMode {
    BatchedQ6,
    BatchedQ8,
    PerLane,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeBatch16PlanTelemetry {
    pub q8_batched_gdn_blocks: usize,
    pub packed_q4_gate_up_blocks: usize,
    pub per_lane_gdn_blocks: usize,
    pub per_lane_gate_up_blocks: usize,
    pub per_lane_iq3_gate_up_blocks: usize,
    pub per_lane_other_gate_up_blocks: usize,
    pub head_mode: HeadMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MixerMode {
    Gdn(GdnMixerMode),
    Attention,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BlockExecutionPlan {
    mixer: MixerMode,
    routed: RoutedGateUpMode,
    routed_dtype: GgmlType,
}

#[derive(Debug)]
struct MoeBatch16ExecutionPlan {
    blocks: Box<[BlockExecutionPlan]>,
    head: HeadMode,
    telemetry: MoeBatch16PlanTelemetry,
}

#[derive(Debug, thiserror::Error)]
pub enum MoeBatch16Error {
    #[error("MoE B=16 unsupported: {0}")]
    Unsupported(String),
    #[error("MoE B=16 validation: {0}")]
    Validation(String),
    #[error("MoE B=16 cancelled before commit")]
    CancelledBeforeCommit,
    #[error("MoE B=16 executor is poisoned after a committed command failure")]
    Poisoned,
    #[error("MoE B=16 command failed: status={status} error={error}")]
    CommandBuffer { status: String, error: String },
    #[error("MoE B=16 metal: {0}")]
    Metal(#[from] MetalError),
    #[error("MoE B=16 forward: {0}")]
    Forward(#[from] MfError),
    #[error("MoE B=16 greedy selection failed in slot {slot}: {source}")]
    GreedySelection {
        slot: usize,
        #[source]
        source: SamplingError,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeBatch16Step {
    pub argmax_ids: [i32; MOE_BATCH16_WIDTH],
    pub gpu_ms: Option<f64>,
}

fn checked_mul(values: &[usize], label: &str) -> Result<usize, MoeBatch16Error> {
    values.iter().try_fold(1usize, |total, value| {
        total
            .checked_mul(*value)
            .ok_or_else(|| MoeBatch16Error::Validation(format!("{label} overflow")))
    })
}

fn geometry(arch: &Arch) -> Result<(usize, usize, usize, usize, usize), MoeBatch16Error> {
    let hidden = arch.hidden_size as usize;
    let topk = arch.expert_used_count as usize;
    let experts = arch.expert_count as usize;
    let expert_ffn = arch.expert_feed_forward_length as usize;
    let shared_ffn = arch.expert_shared_feed_forward_length as usize;
    if hidden == 0
        || experts == 0
        || topk == 0
        || topk > experts
        || expert_ffn == 0
        || shared_ffn == 0
    {
        return Err(MoeBatch16Error::Unsupported(
            "expert geometry must be nonzero and top-k must not exceed expert count".into(),
        ));
    }
    checked_mul(&[experts, hidden, expert_ffn], "expert bank geometry")?;
    checked_mul(&[topk, expert_ffn], "routed inner geometry")?;
    let conv_heads = (arch.gdn_n_k_heads as usize)
        .checked_mul(2)
        .and_then(|value| value.checked_add(arch.gdn_n_v_heads as usize))
        .ok_or_else(|| MoeBatch16Error::Unsupported("GDN head geometry overflow".into()))?;
    let conv_dim = conv_heads
        .checked_mul(arch.gdn_head_dim as usize)
        .ok_or_else(|| MoeBatch16Error::Unsupported("GDN convolution geometry overflow".into()))?;
    let value_dim = (arch.gdn_n_v_heads as usize)
        .checked_mul(arch.gdn_head_dim as usize)
        .ok_or_else(|| MoeBatch16Error::Unsupported("GDN value geometry overflow".into()))?;
    if conv_dim == 0 || value_dim == 0 {
        return Err(MoeBatch16Error::Unsupported(
            "GDN geometry must be nonzero".into(),
        ));
    }
    Ok((hidden, topk, experts, expert_ffn, value_dim))
}

fn moe_batch16_scratch_logical_buffers(arch: &Arch) -> Result<Vec<u64>, MoeBatch16Error> {
    let (hidden, topk, _, expert_ffn, value_dim) = geometry(arch)?;
    let conv_heads = (arch.gdn_n_k_heads as usize)
        .checked_mul(2)
        .and_then(|value| value.checked_add(arch.gdn_n_v_heads as usize))
        .ok_or_else(|| MoeBatch16Error::Validation("GDN scratch geometry overflow".into()))?;
    let conv_dim = checked_mul(
        &[conv_heads, arch.gdn_head_dim as usize],
        "GDN scratch geometry",
    )?;
    let f32 = |width, label| {
        batch_elements(width, label)?
            .checked_mul(4)
            .ok_or_else(|| MoeBatch16Error::Validation(format!("{label} bytes overflow")))
    };
    let i32 = |width, label| {
        batch_elements(width, label)?
            .checked_mul(4)
            .ok_or_else(|| MoeBatch16Error::Validation(format!("{label} bytes overflow")))
    };
    Ok(vec![
        i32(1, "ID scratch")?,
        f32(hidden, "row scratch")?,
        f32(arch.vocab_size as usize, "logit scratch")?,
        i32(1, "selection scratch")?,
        i32(topk, "top-k scratch")?,
        f32(topk * expert_ffn, "routed-inner scratch")?,
        f32(hidden, "GDN H scratch")?,
        f32(conv_dim, "GDN QKV scratch")?,
        f32(value_dim, "GDN Z scratch")?,
        f32(value_dim, "GDN normed scratch")?,
        f32(hidden, "GDN out scratch")?,
    ])
}

/// Logical bytes allocated by the executor's persistent B=16 scratch arena.
/// This excludes model weights and sequence-owned causal state.
pub fn moe_batch16_scratch_bytes(arch: &Arch) -> Result<u64, MoeBatch16Error> {
    moe_batch16_scratch_logical_buffers(arch)?
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| MoeBatch16Error::Validation("scratch byte count overflow".into()))
        })
}

/// Metal-allocation upper bound for the executor's independent scratch buffers.
pub fn moe_batch16_scratch_upper_bytes(
    ctx: &MetalContext,
    arch: &Arch,
) -> Result<u64, MoeBatch16Error> {
    moe_batch16_scratch_logical_buffers(arch)?
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            let priced = ctx.shared_buffer_size_and_align(bytes)?.size;
            total.checked_add(priced).ok_or_else(|| {
                MoeBatch16Error::Validation("priced scratch byte count overflow".into())
            })
        })
}

struct GdnScratch {
    h: MetalTensor,
    qkv: MetalTensor,
    z: MetalTensor,
    normed: MetalTensor,
    out: MetalTensor,
}

struct MoeBatch16Scratch {
    ids: MetalTensor,
    rows: MetalTensor,
    logits: MetalTensor,
    selections: MetalTensor,
    topk_idx: MetalTensor,
    routed_inner: MetalTensor,
    gdn: GdnScratch,
    bytes: u64,
}

fn batch_elements(width: usize, label: &str) -> Result<u64, MoeBatch16Error> {
    checked_mul(&[MOE_BATCH16_WIDTH, width], label).and_then(|value| {
        u64::try_from(value)
            .map_err(|_| MoeBatch16Error::Validation(format!("{label} exceeds u64")))
    })
}

impl MoeBatch16Scratch {
    fn new(ctx: &MetalContext, model: &MetalModel) -> Result<Self, MoeBatch16Error> {
        let arch = &model.arch;
        let (hidden, topk, _, expert_ffn, value_dim) = geometry(arch)?;
        let conv_dim = ((2 * arch.gdn_n_k_heads as usize) + arch.gdn_n_v_heads as usize)
            .checked_mul(arch.gdn_head_dim as usize)
            .ok_or_else(|| MoeBatch16Error::Validation("GDN scratch width overflow".into()))?;
        let bytes = moe_batch16_scratch_bytes(arch)?;
        Ok(Self {
            ids: MetalTensor::zeros_i32(ctx, vec![MOE_BATCH16_WIDTH as u64])?,
            rows: MetalTensor::zeros_f32(ctx, vec![batch_elements(hidden, "row scratch")?])?,
            logits: MetalTensor::zeros_f32(
                ctx,
                vec![batch_elements(arch.vocab_size as usize, "logit scratch")?],
            )?,
            selections: MetalTensor::zeros_i32(ctx, vec![MOE_BATCH16_WIDTH as u64])?,
            topk_idx: MetalTensor::zeros_i32(ctx, vec![batch_elements(topk, "top-k scratch")?])?,
            routed_inner: MetalTensor::zeros_f32(
                ctx,
                vec![batch_elements(topk * expert_ffn, "routed-inner scratch")?],
            )?,
            gdn: GdnScratch {
                h: MetalTensor::zeros_f32(ctx, vec![batch_elements(hidden, "GDN H scratch")?])?,
                qkv: MetalTensor::zeros_f32(
                    ctx,
                    vec![batch_elements(conv_dim, "GDN QKV scratch")?],
                )?,
                z: MetalTensor::zeros_f32(ctx, vec![batch_elements(value_dim, "GDN Z scratch")?])?,
                normed: MetalTensor::zeros_f32(
                    ctx,
                    vec![batch_elements(value_dim, "GDN normed scratch")?],
                )?,
                out: MetalTensor::zeros_f32(ctx, vec![batch_elements(hidden, "GDN out scratch")?])?,
            },
            bytes,
        })
    }
}

fn matvec_dtype(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::IQ2_XS
            | GgmlType::IQ2_S
            | GgmlType::IQ3_XXS
            | GgmlType::IQ3_S
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::MXFP4
            | GgmlType::Q8_0
            | GgmlType::IQ4_NL
            | GgmlType::IQ4_XS
    )
}

fn validate_matvec_contract(
    dtype: GgmlType,
    n_in: usize,
    n_out: usize,
    label: &str,
) -> Result<(), MoeBatch16Error> {
    if !matvec_dtype(dtype) {
        return Err(MoeBatch16Error::Unsupported(format!(
            "{label} dtype {dtype:?} has no production mat-vec contract"
        )));
    }
    let block = ggml_type_layout(dtype)
        .and_then(|(block, _)| usize::try_from(block).ok())
        .filter(|block| *block > 0)
        .ok_or_else(|| {
            MoeBatch16Error::Unsupported(format!(
                "{label} dtype {dtype:?} has no valid block layout"
            ))
        })?;
    if n_in == 0
        || n_out == 0
        || !n_in.is_multiple_of(block)
        || u32::try_from(n_in).is_err()
        || u32::try_from(n_out).is_err()
        || n_in.checked_mul(n_out).is_none()
    {
        return Err(MoeBatch16Error::Unsupported(format!(
            "{label} dimensions {n_in}x{n_out} are invalid for {dtype:?} block {block}"
        )));
    }
    Ok(())
}

fn embedding_dtype(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q4_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
    )
}

fn routed_gate_up_dtype(gate: GgmlType, up: GgmlType) -> bool {
    gate == up
        && matches!(
            gate,
            GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::Q8_0
                | GgmlType::IQ3_XXS
                | GgmlType::IQ3_S
                | GgmlType::BF16
                | GgmlType::F32
        )
}

fn dimensions_fit_u32(dimensions: &[usize]) -> bool {
    dimensions.iter().all(|value| u32::try_from(*value).is_ok())
        && checked_mul(dimensions, "kernel dimensions").is_ok()
}

fn select_gdn_mode(
    qkv: GgmlType,
    z: GgmlType,
    out: GgmlType,
    hidden: usize,
    conv_dim: usize,
    value_dim: usize,
) -> Result<GdnMixerMode, MoeBatch16Error> {
    validate_matvec_contract(qkv, hidden, conv_dim, "GDN QKV projection")?;
    validate_matvec_contract(z, hidden, value_dim, "GDN Z projection")?;
    validate_matvec_contract(out, value_dim, hidden, "GDN output projection")?;
    let batched_dimensions = dimensions_fit_u32(&[hidden, conv_dim, value_dim])
        && hidden.is_multiple_of(32)
        && value_dim.is_multiple_of(32);
    Ok(
        if qkv == GgmlType::Q8_0
            && z == GgmlType::Q8_0
            && out == GgmlType::Q8_0
            && batched_dimensions
            && mat_vec_q8_0_lcpp_enabled()
        {
            GdnMixerMode::BatchedQ8
        } else {
            GdnMixerMode::PerLane
        },
    )
}

fn select_routed_mode(
    gate: GgmlType,
    up: GgmlType,
    hidden: usize,
    expert_ffn: usize,
    experts: usize,
    topk: usize,
) -> Result<RoutedGateUpMode, MoeBatch16Error> {
    if !routed_gate_up_dtype(gate, up) {
        return Err(MoeBatch16Error::Unsupported(format!(
            "routed gate/up dtype contract is unsupported: {gate:?}/{up:?}"
        )));
    }
    validate_matvec_contract(gate, hidden, expert_ffn, "routed gate projection")?;
    validate_matvec_contract(up, hidden, expert_ffn, "routed up projection")?;
    if matches!(
        gate,
        GgmlType::Q5_K | GgmlType::IQ3_XXS | GgmlType::IQ3_S | GgmlType::BF16 | GgmlType::F32
    ) && expert_ffn
        .checked_mul(2)
        .is_none_or(|packed_width| packed_width > hidden)
    {
        return Err(MoeBatch16Error::Unsupported(
            "routed gate/up fallback scratch exceeds the production session contract".into(),
        ));
    }
    let packed_dimensions = hidden.is_multiple_of(256)
        && dimensions_fit_u32(&[hidden, expert_ffn, experts, topk, MOE_BATCH16_WIDTH])
        && checked_mul(
            &[MOE_BATCH16_WIDTH, topk, expert_ffn],
            "packed routed output",
        )
        .is_ok();
    Ok(if gate == GgmlType::Q4_K && packed_dimensions {
        RoutedGateUpMode::PackedQ4
    } else {
        RoutedGateUpMode::PerLane
    })
}

fn select_head_mode(
    dtype: GgmlType,
    hidden: usize,
    vocab: usize,
) -> Result<HeadMode, MoeBatch16Error> {
    validate_matvec_contract(dtype, hidden, vocab, "LM head")?;
    Ok(
        if dtype == GgmlType::Q6_K
            && hidden.is_multiple_of(256)
            && dimensions_fit_u32(&[hidden, vocab, MOE_BATCH16_WIDTH])
        {
            HeadMode::BatchedQ6
        } else if dtype == GgmlType::Q8_0
            && hidden.is_multiple_of(32)
            && dimensions_fit_u32(&[hidden, vocab, MOE_BATCH16_WIDTH])
            && mat_vec_q8_0_lcpp_enabled()
        {
            HeadMode::BatchedQ8
        } else {
            HeadMode::PerLane
        },
    )
}

fn has_meaningful_batched_stage(
    gdn_modes: impl IntoIterator<Item = GdnMixerMode>,
    routed_modes: impl IntoIterator<Item = RoutedGateUpMode>,
    head: HeadMode,
) -> bool {
    matches!(head, HeadMode::BatchedQ6 | HeadMode::BatchedQ8)
        || gdn_modes
            .into_iter()
            .any(|mode| mode == GdnMixerMode::BatchedQ8)
        || routed_modes
            .into_iter()
            .any(|mode| mode == RoutedGateUpMode::PackedQ4)
}

fn validate_architecture(arch: &Arch) -> Result<(), MoeBatch16Error> {
    if arch.kind != ArchKind::Moe {
        return Err(MoeBatch16Error::Unsupported(
            "model architecture is not Qwen MoE".into(),
        ));
    }
    geometry(arch)?;
    if arch.n_layer == 0 {
        return Err(MoeBatch16Error::Unsupported(
            "model must contain at least one base layer".into(),
        ));
    }
    if arch.gdn_head_dim != 128 || arch.gdn_conv_kernel != 4 {
        return Err(MoeBatch16Error::Unsupported(
            "GDN requires head_dim 128 and convolution kernel 4".into(),
        ));
    }
    if arch.gdn_n_k_heads == 0
        || arch.gdn_n_v_heads == 0
        || !arch.gdn_n_v_heads.is_multiple_of(arch.gdn_n_k_heads)
    {
        return Err(MoeBatch16Error::Unsupported(
            "GDN value heads must be an integral multiple of key heads".into(),
        ));
    }
    let attention_group = if arch.n_kv_heads > 0 {
        arch.n_q_heads.checked_div(arch.n_kv_heads)
    } else {
        None
    };
    if arch.attn_head_dim != 256
        || arch.n_kv_heads == 0
        || arch.n_q_heads == 0
        || !arch.n_q_heads.is_multiple_of(arch.n_kv_heads)
        || !matches!(attention_group, Some(4 | 6 | 8 | 16))
    {
        return Err(MoeBatch16Error::Unsupported(
            "attention requires head_dim 256 and GQA group 4, 6, 8, or 16".into(),
        ));
    }
    if !arch.rope_theta.is_finite() || arch.rope_theta <= 0.0 {
        return Err(MoeBatch16Error::Unsupported(
            "RoPE theta must be finite and positive".into(),
        ));
    }
    let rotary_width = arch.attn_head_dim as f32 * arch.partial_rotary_factor;
    if !arch.partial_rotary_factor.is_finite()
        || arch.partial_rotary_factor <= 0.0
        || !rotary_width.is_finite()
        || rotary_width.fract() != 0.0
        || rotary_width < 2.0
        || rotary_width > arch.attn_head_dim as f32
        || !(rotary_width as usize).is_multiple_of(2)
    {
        return Err(MoeBatch16Error::Unsupported(
            "partial rotary factor must produce a positive even width within the attention head"
                .into(),
        ));
    }
    if arch.expert_count > 256 || arch.expert_used_count > 16 {
        return Err(MoeBatch16Error::Unsupported(
            "production routing requires at most 256 experts and top-k at most 16".into(),
        ));
    }
    let hidden = arch.hidden_size as usize;
    let conv_heads = (arch.gdn_n_k_heads as usize)
        .checked_mul(2)
        .and_then(|value| value.checked_add(arch.gdn_n_v_heads as usize))
        .ok_or_else(|| MoeBatch16Error::Unsupported("GDN head geometry overflow".into()))?;
    for (label, dimensions) in [
        ("GDN", vec![conv_heads, arch.gdn_head_dim as usize]),
        (
            "attention Q",
            vec![arch.n_q_heads as usize, arch.attn_head_dim as usize],
        ),
        (
            "attention KV",
            vec![arch.n_kv_heads as usize, arch.attn_head_dim as usize],
        ),
        ("embedding", vec![hidden, arch.vocab_size as usize]),
    ] {
        checked_mul(&dimensions, label)?;
        if !dimensions_fit_u32(&dimensions) {
            return Err(MoeBatch16Error::Unsupported(format!(
                "{label} dimensions exceed kernel ABI"
            )));
        }
    }
    Ok(())
}

fn down_dtype(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
            | GgmlType::IQ4_XS
            | GgmlType::BF16
            | GgmlType::F32
    )
}

fn require_weight_shape(
    tensor: &MetalTensor,
    name: &str,
    dimensions: &[usize],
) -> Result<(), MoeBatch16Error> {
    checked_mul(dimensions, name)?;
    let expected: Vec<u64> = dimensions
        .iter()
        .copied()
        .map(|dimension| {
            u64::try_from(dimension).map_err(|_| {
                MoeBatch16Error::Unsupported(format!(
                    "{name} dimension {dimension} does not fit u64"
                ))
            })
        })
        .collect::<Result<_, _>>()?;
    if tensor.shape != expected {
        return Err(MoeBatch16Error::Unsupported(format!(
            "{name} has shape {:?}, expected {expected:?}",
            tensor.shape
        )));
    }
    Ok(())
}

fn require_weight_shape_one_of(
    tensor: &MetalTensor,
    name: &str,
    alternatives: &[&[usize]],
) -> Result<(), MoeBatch16Error> {
    for dimensions in alternatives {
        checked_mul(dimensions, name)?;
        let matches = tensor.shape.len() == dimensions.len()
            && tensor
                .shape
                .iter()
                .zip(*dimensions)
                .all(|(actual, expected)| *actual == *expected as u64);
        if matches {
            return Ok(());
        }
    }
    Err(MoeBatch16Error::Unsupported(format!(
        "{name} has unsupported shape {:?}",
        tensor.shape
    )))
}

fn build_execution_plan(model: &MetalModel) -> Result<MoeBatch16ExecutionPlan, MoeBatch16Error> {
    let arch = &model.arch;
    validate_architecture(arch)?;
    if model.blocks.is_empty() || model.blocks.len() != arch.n_layer as usize {
        return Err(MoeBatch16Error::Unsupported(
            "block inventory does not match architecture".into(),
        ));
    }
    if model.has_queue_scoped_residency_set() {
        return Err(MoeBatch16Error::Unsupported(
            "model residency is scoped to another command queue".into(),
        ));
    }
    for flag in [
        "QWEN_DECODE_GDN_NOOP_FRONT",
        "QWEN_DECODE_GDN_NOOP_OUT",
        "QWEN_DECODE_GDN_NOOP_QKV",
        "QWEN_DECODE_GDN_NOOP_Z",
        "QWEN_DECODE_GDN_NOOP_BETA",
        "QWEN_DECODE_GDN_NOOP_ALPHA",
        "QWEN_DECODE_MOE_NOOP_ROUTE",
        "QWEN_DECODE_MOE_NOOP_ROUTED_GATEUP",
        "QWEN_DECODE_MOE_NOOP_ROUTED_DOWN",
    ] {
        if read_default_off(flag) {
            return Err(MoeBatch16Error::Unsupported(format!(
                "diagnostic flag {flag} changes the production graph"
            )));
        }
    }
    let hidden = arch.hidden_size as usize;
    let vocab = arch.vocab_size as usize;
    let experts = arch.expert_count as usize;
    let expert_ffn = arch.expert_feed_forward_length as usize;
    let shared_ffn = arch.expert_shared_feed_forward_length as usize;
    let conv_heads = checked_mul(&[2, arch.gdn_n_k_heads as usize], "GDN head geometry")?
        .checked_add(arch.gdn_n_v_heads as usize)
        .ok_or_else(|| MoeBatch16Error::Validation("GDN head geometry overflow".into()))?;
    let conv_dim = checked_mul(
        &[conv_heads, arch.gdn_head_dim as usize],
        "GDN convolution geometry",
    )?;
    let value_dim = checked_mul(
        &[arch.gdn_n_v_heads as usize, arch.gdn_head_dim as usize],
        "GDN value geometry",
    )?;
    let q_dim = checked_mul(
        &[arch.n_q_heads as usize, arch.attn_head_dim as usize],
        "attention Q geometry",
    )?;
    let kv_dim = checked_mul(
        &[arch.n_kv_heads as usize, arch.attn_head_dim as usize],
        "attention KV geometry",
    )?;
    if !embedding_dtype(model.token_embd.dtype) {
        return Err(MoeBatch16Error::Unsupported(format!(
            "token embedding dtype {:?} has no batched get-rows contract",
            model.token_embd.dtype
        )));
    }
    validate_matvec_contract(model.token_embd.dtype, hidden, vocab, "token embedding")?;
    require_weight_shape(&model.token_embd, "token embedding", &[hidden, vocab])?;
    require_weight_shape(&model.output_norm, "output norm", &[hidden])?;
    require_weight_shape(&model.lm_head, "LM head", &[hidden, vocab])?;
    let head = select_head_mode(model.lm_head.dtype, hidden, vocab)?;
    let mut blocks = Vec::with_capacity(model.blocks.len());
    for (index, block) in model.blocks.iter().enumerate() {
        let (mixer, shared_gate, shared_up, shared_down, moe) = match block {
            MetalBlock::Gdn(block) => {
                let mode = select_gdn_mode(
                    block.in_proj_qkv.dtype,
                    block.in_proj_z.dtype,
                    block.out_proj.dtype,
                    hidden,
                    conv_dim,
                    value_dim,
                )?;
                for (name, tensor) in [("beta", &block.beta_proj), ("alpha", &block.alpha_proj)] {
                    validate_matvec_contract(
                        tensor.dtype,
                        hidden,
                        arch.gdn_n_v_heads as usize,
                        &format!("block {index} GDN {name} projection"),
                    )?;
                }
                require_weight_shape(
                    &block.in_proj_qkv,
                    &format!("block {index} GDN QKV"),
                    &[hidden, conv_dim],
                )?;
                require_weight_shape(
                    &block.in_proj_z,
                    &format!("block {index} GDN Z"),
                    &[hidden, value_dim],
                )?;
                require_weight_shape(
                    &block.out_proj,
                    &format!("block {index} GDN output"),
                    &[value_dim, hidden],
                )?;
                for (name, tensor, dimensions) in [
                    ("attention norm", &block.attn_norm, vec![hidden]),
                    ("post-attention norm", &block.post_attn_norm, vec![hidden]),
                    (
                        "beta projection",
                        &block.beta_proj,
                        vec![hidden, arch.gdn_n_v_heads as usize],
                    ),
                    (
                        "alpha projection",
                        &block.alpha_proj,
                        vec![hidden, arch.gdn_n_v_heads as usize],
                    ),
                    ("A log", &block.a_log, vec![arch.gdn_n_v_heads as usize]),
                    ("DT bias", &block.dt_bias, vec![arch.gdn_n_v_heads as usize]),
                    (
                        "convolution",
                        &block.conv1d,
                        vec![arch.gdn_conv_kernel as usize, conv_dim],
                    ),
                    ("state norm", &block.norm, vec![arch.gdn_head_dim as usize]),
                ] {
                    require_weight_shape(
                        tensor,
                        &format!("block {index} GDN {name}"),
                        &dimensions,
                    )?;
                }
                (
                    MixerMode::Gdn(mode),
                    &block.ffn_gate,
                    &block.ffn_up,
                    &block.ffn_down,
                    block.ffn_moe.as_ref(),
                )
            }
            MetalBlock::Attn(block) => {
                let q_full_dim = checked_mul(&[2, q_dim], "attention Q projection width")?;
                for (name, tensor, n_in, n_out) in [
                    ("Q", &block.q, hidden, q_full_dim),
                    ("K", &block.k, hidden, kv_dim),
                    ("V", &block.v, hidden, kv_dim),
                    ("O", &block.o, q_dim, hidden),
                ] {
                    validate_matvec_contract(
                        tensor.dtype,
                        n_in,
                        n_out,
                        &format!("block {index} attention {name}"),
                    )?;
                }
                for (name, tensor, dimensions) in [
                    ("Q", &block.q, [hidden, q_full_dim]),
                    ("K", &block.k, [hidden, kv_dim]),
                    ("V", &block.v, [hidden, kv_dim]),
                    ("O", &block.o, [q_dim, hidden]),
                ] {
                    require_weight_shape(
                        tensor,
                        &format!("block {index} attention {name}"),
                        &dimensions,
                    )?;
                }
                for (name, tensor, dimensions) in [
                    ("attention norm", &block.attn_norm, vec![hidden]),
                    ("post-attention norm", &block.post_attn_norm, vec![hidden]),
                    ("Q norm", &block.q_norm, vec![arch.attn_head_dim as usize]),
                    ("K norm", &block.k_norm, vec![arch.attn_head_dim as usize]),
                ] {
                    require_weight_shape(tensor, &format!("block {index} {name}"), &dimensions)?;
                }
                (
                    MixerMode::Attention,
                    &block.ffn_gate,
                    &block.ffn_up,
                    &block.ffn_down,
                    block.ffn_moe.as_ref(),
                )
            }
        };
        let moe = moe.ok_or_else(|| {
            MoeBatch16Error::Unsupported(format!("block {index} has no MoE weights"))
        })?;
        let routed = select_routed_mode(
            moe.gate_exps.dtype,
            moe.up_exps.dtype,
            hidden,
            expert_ffn,
            experts,
            arch.expert_used_count as usize,
        )?;
        if moe.gate_inp_shexp.dtype != GgmlType::F32 {
            return Err(MoeBatch16Error::Unsupported(format!(
                "block {index} router/shared-gate dtype contract is unsupported"
            )));
        }
        validate_matvec_contract(
            moe.gate_inp.dtype,
            hidden,
            experts,
            &format!("block {index} router"),
        )?;
        require_weight_shape(
            &moe.gate_inp,
            &format!("block {index} router"),
            &[hidden, experts],
        )?;
        require_weight_shape_one_of(
            &moe.gate_inp_shexp,
            &format!("block {index} shared gate"),
            &[&[hidden], &[hidden, 1]],
        )?;
        for (name, tensor) in [("routed gate", &moe.gate_exps), ("routed up", &moe.up_exps)] {
            require_weight_shape(
                tensor,
                &format!("block {index} {name}"),
                &[hidden, expert_ffn, experts],
            )?;
        }
        require_weight_shape(
            &moe.down_exps,
            &format!("block {index} routed down"),
            &[expert_ffn, hidden, experts],
        )?;
        if !down_dtype(moe.down_exps.dtype) {
            return Err(MoeBatch16Error::Unsupported(format!(
                "block {index} routed down dtype {:?} is unsupported",
                moe.down_exps.dtype
            )));
        }
        validate_matvec_contract(
            moe.down_exps.dtype,
            expert_ffn,
            hidden,
            &format!("block {index} routed down"),
        )?;
        for (name, tensor, n_in, n_out) in [
            ("shared gate", shared_gate, hidden, shared_ffn),
            ("shared up", shared_up, hidden, shared_ffn),
            ("shared down", shared_down, shared_ffn, hidden),
        ] {
            validate_matvec_contract(tensor.dtype, n_in, n_out, &format!("block {index} {name}"))?;
        }
        require_weight_shape(
            shared_gate,
            &format!("block {index} shared gate projection"),
            &[hidden, shared_ffn],
        )?;
        require_weight_shape(
            shared_up,
            &format!("block {index} shared up projection"),
            &[hidden, shared_ffn],
        )?;
        require_weight_shape(
            shared_down,
            &format!("block {index} shared down projection"),
            &[shared_ffn, hidden],
        )?;
        blocks.push(BlockExecutionPlan {
            mixer,
            routed,
            routed_dtype: moe.gate_exps.dtype,
        });
    }
    let telemetry = MoeBatch16PlanTelemetry {
        q8_batched_gdn_blocks: blocks
            .iter()
            .filter(|block| block.mixer == MixerMode::Gdn(GdnMixerMode::BatchedQ8))
            .count(),
        packed_q4_gate_up_blocks: blocks
            .iter()
            .filter(|block| block.routed == RoutedGateUpMode::PackedQ4)
            .count(),
        per_lane_gdn_blocks: blocks
            .iter()
            .filter(|block| block.mixer == MixerMode::Gdn(GdnMixerMode::PerLane))
            .count(),
        per_lane_gate_up_blocks: blocks
            .iter()
            .filter(|block| block.routed == RoutedGateUpMode::PerLane)
            .count(),
        per_lane_iq3_gate_up_blocks: blocks
            .iter()
            .filter(|block| {
                block.routed == RoutedGateUpMode::PerLane
                    && matches!(block.routed_dtype, GgmlType::IQ3_XXS | GgmlType::IQ3_S)
            })
            .count(),
        per_lane_other_gate_up_blocks: blocks
            .iter()
            .filter(|block| {
                block.routed == RoutedGateUpMode::PerLane
                    && !matches!(block.routed_dtype, GgmlType::IQ3_XXS | GgmlType::IQ3_S)
            })
            .count(),
        head_mode: head,
    };
    if !has_meaningful_batched_stage(
        blocks.iter().filter_map(|block| match block.mixer {
            MixerMode::Gdn(mode) => Some(mode),
            MixerMode::Attention => None,
        }),
        blocks.iter().map(|block| block.routed),
        head,
    ) {
        return Err(MoeBatch16Error::Unsupported(
            "execution plan has no meaningful batched compute stage".into(),
        ));
    }
    Ok(MoeBatch16ExecutionPlan {
        blocks: blocks.into_boxed_slice(),
        head,
        telemetry,
    })
}

pub(crate) fn inspect_execution_plan(
    model: &MetalModel,
) -> Result<MoeBatch16PlanTelemetry, MoeBatch16Error> {
    Ok(build_execution_plan(model)?.telemetry)
}

pub(crate) struct MoeBatch16Executor<'a> {
    forward: MetalForward<'a>,
    plan: MoeBatch16ExecutionPlan,
    scratch: MoeBatch16Scratch,
    poisoned: bool,
}

impl<'a> MoeBatch16Executor<'a> {
    pub(crate) fn new(
        ctx: &'a MetalContext,
        model: &'a MetalModel,
    ) -> Result<Self, MoeBatch16Error> {
        // All capability checks deliberately precede the large logits/inner allocations.
        let plan = build_execution_plan(model)?;
        Ok(Self {
            forward: MetalForward::new(ctx, model),
            plan,
            scratch: MoeBatch16Scratch::new(ctx, model)?,
            poisoned: false,
        })
    }

    pub(crate) fn scratch_bytes(&self) -> u64 {
        self.scratch.bytes
    }

    pub(crate) fn plan_telemetry(&self) -> MoeBatch16PlanTelemetry {
        self.plan.telemetry
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub(crate) fn validate_refs(
        &self,
        token_ids: [i32; MOE_BATCH16_WIDTH],
        positions: [u32; MOE_BATCH16_WIDTH],
        sessions: &mut [&mut MetalSession; MOE_BATCH16_WIDTH],
    ) -> Result<(), MoeBatch16Error> {
        if self.poisoned {
            return Err(MoeBatch16Error::Poisoned);
        }
        self.validate_step(&token_ids, positions, sessions)
    }

    pub(crate) fn step_greedy_refs(
        &mut self,
        token_ids: [i32; MOE_BATCH16_WIDTH],
        positions: [u32; MOE_BATCH16_WIDTH],
        sessions: &mut [&mut MetalSession; MOE_BATCH16_WIDTH],
        cancelled: impl Fn() -> bool,
    ) -> Result<MoeBatch16Step, MoeBatch16Error> {
        if self.poisoned {
            return Err(MoeBatch16Error::Poisoned);
        }
        if cancelled() {
            return Err(MoeBatch16Error::CancelledBeforeCommit);
        }
        self.validate_step(&token_ids, positions, sessions)?;
        // IDs are staged before any command encoding can mutate host frontiers.
        self.write_ids(&token_ids)?;
        if cancelled() {
            return Err(MoeBatch16Error::CancelledBeforeCommit);
        }
        let command = self
            .forward
            .ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| MoeBatch16Error::Validation("command buffer unavailable".into()))?;
        let encoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.encode_step(&command, positions, sessions)?;
            Ok::<bool, MoeBatch16Error>(cancelled())
        }));
        match encoded {
            Ok(Ok(false)) => {}
            Ok(Ok(true)) => {
                Self::restore_frontiers(sessions, positions);
                return Err(MoeBatch16Error::CancelledBeforeCommit);
            }
            Ok(Err(error)) => {
                Self::restore_frontiers(sessions, positions);
                return Err(error);
            }
            Err(payload) => {
                Self::restore_frontiers(sessions, positions);
                std::panic::resume_unwind(payload);
            }
        }
        command.commit();
        crate::metal::wait_unchecked(&command);
        let status = command.status();
        let error = command.error();
        if status != MTLCommandBufferStatus::Completed || error.is_some() {
            self.poisoned = true;
            Self::poison_sessions(sessions, "a committed MoE B=16 command failed");
            return Err(MoeBatch16Error::CommandBuffer {
                status: format!("{status:?}"),
                error: format!("{error:?}"),
            });
        }
        let mut argmax_ids = match self.read_selections() {
            Ok(ids) => ids,
            Err(error) => {
                self.poisoned = true;
                Self::poison_sessions(sessions, "MoE B=16 greedy readback failed");
                return Err(error);
            }
        };
        for (slot, raw) in argmax_ids.iter_mut().enumerate() {
            *raw = match GreedySelection::from_encoded(*raw).into_token() {
                Ok(token) => token,
                Err(source) => {
                    self.poisoned = true;
                    Self::poison_sessions(sessions, "MoE B=16 greedy selection failed");
                    return Err(MoeBatch16Error::GreedySelection { slot, source });
                }
            };
        }
        let start = command.GPUStartTime();
        let end = command.GPUEndTime();
        let gpu_ms = (start.is_finite() && end.is_finite() && start > 0.0 && end > start)
            .then_some((end - start) * 1e3);
        Ok(MoeBatch16Step { argmax_ids, gpu_ms })
    }

    fn validate_step(
        &self,
        token_ids: &[i32; MOE_BATCH16_WIDTH],
        positions: [u32; MOE_BATCH16_WIDTH],
        sessions: &[&mut MetalSession; MOE_BATCH16_WIDTH],
    ) -> Result<(), MoeBatch16Error> {
        let arch = &self.forward.model.arch;
        let hidden = arch.hidden_size as usize;
        let topk = arch.expert_used_count as usize;
        let experts = arch.expert_count as usize;
        let expert_ffn = arch.expert_feed_forward_length as usize;
        let shared_ffn = arch.expert_shared_feed_forward_length as usize;
        let value_dim = checked_mul(
            &[arch.gdn_n_v_heads as usize, arch.gdn_head_dim as usize],
            "GDN value geometry",
        )?;
        let key_dim = checked_mul(
            &[arch.gdn_n_k_heads as usize, arch.gdn_head_dim as usize],
            "GDN key geometry",
        )?;
        let conv_heads = (arch.gdn_n_k_heads as usize)
            .checked_mul(2)
            .and_then(|value| value.checked_add(arch.gdn_n_v_heads as usize))
            .ok_or_else(|| MoeBatch16Error::Validation("GDN head geometry overflow".into()))?;
        let conv_dim = checked_mul(
            &[conv_heads, arch.gdn_head_dim as usize],
            "GDN convolution geometry",
        )?;
        let conv_history = (arch.gdn_conv_kernel as usize)
            .checked_sub(1)
            .ok_or_else(|| MoeBatch16Error::Validation("GDN kernel must be nonzero".into()))?;
        let conv_state = conv_history
            .checked_mul(conv_dim)
            .ok_or_else(|| MoeBatch16Error::Validation("GDN state geometry overflow".into()))?;
        let recurrent_state = checked_mul(
            &[value_dim, arch.gdn_head_dim as usize],
            "GDN recurrent geometry",
        )?;
        let q_dim = checked_mul(
            &[arch.n_q_heads as usize, arch.attn_head_dim as usize],
            "attention Q geometry",
        )?;
        let kv_dim = checked_mul(
            &[arch.n_kv_heads as usize, arch.attn_head_dim as usize],
            "attention KV geometry",
        )?;
        let group = (arch.n_q_heads as usize)
            .checked_div(arch.n_kv_heads as usize)
            .filter(|group| *group > 0)
            .ok_or_else(|| MoeBatch16Error::Validation("invalid Q/KV head ratio".into()))?;
        let partial_o = checked_mul(
            &[
                arch.n_kv_heads as usize,
                ATTN_V4_MAX_NWG,
                group,
                arch.attn_head_dim as usize,
            ],
            "attention O partial geometry",
        )?;
        let partial_ml = checked_mul(
            &[arch.n_kv_heads as usize, ATTN_V4_MAX_NWG, group, 2],
            "attention ML partial geometry",
        )?;
        let expected_gdn = self
            .forward
            .model
            .blocks
            .iter()
            .filter(|block| matches!(block, MetalBlock::Gdn(_)))
            .count();
        let expected_attn = self.forward.model.blocks.len() - expected_gdn;
        let capacity = sessions[0].kv_capacity;
        for (slot, token) in token_ids.iter().copied().enumerate() {
            if token < 0 || token as u32 >= arch.vocab_size {
                return Err(MoeBatch16Error::Validation(format!(
                    "slot {slot} token {token} is outside vocab {}",
                    arch.vocab_size
                )));
            }
        }
        for (slot, session) in sessions.iter().enumerate() {
            let session = &**session;
            let position = positions[slot] as usize;
            session.ensure_usable()?;
            if session.has_internal_mutable_alias() {
                return Err(MoeBatch16Error::Validation(format!(
                    "slot {slot} aliases mutable storage internally"
                )));
            }
            if session.kv_capacity != capacity {
                return Err(MoeBatch16Error::Validation(format!(
                    "slot {slot} capacity {} != cohort capacity {capacity}",
                    session.kv_capacity
                )));
            }
            if position >= capacity {
                return Err(MoeBatch16Error::Validation(format!(
                    "slot {slot} position {position} exceeds capacity {capacity}"
                )));
            }
            if session.kv_n_pos.iter().any(|actual| *actual != position) {
                return Err(MoeBatch16Error::Validation(format!(
                    "slot {slot} does not share cohort frontier {position}"
                )));
            }
            if session.gdn_state.len() != expected_gdn
                || session.gdn_conv.len() != expected_gdn
                || session.kv_k.len() != expected_attn
                || session.kv_v.len() != expected_attn
                || session.kv_n_pos.len() != expected_attn
            {
                return Err(MoeBatch16Error::Validation(format!(
                    "slot {slot} state inventory does not match model"
                )));
            }
            let kv_dtype = GgmlType::F16;
            if !session
                .kv_k
                .iter()
                .chain(&session.kv_v)
                .all(|tensor| tensor.dtype == kv_dtype)
            {
                return Err(MoeBatch16Error::Unsupported(format!(
                    "slot {slot} requires qualified F16 KV storage"
                )));
            }
            for (layer, tensor) in session.gdn_conv.iter().enumerate() {
                require_tensor(
                    tensor,
                    &format!("slot {slot} gdn_conv[{layer}]"),
                    GgmlType::F32,
                    conv_state,
                )?;
            }
            for (layer, tensor) in session.gdn_state.iter().enumerate() {
                require_tensor(
                    tensor,
                    &format!("slot {slot} gdn_state[{layer}]"),
                    GgmlType::F32,
                    recurrent_state,
                )?;
            }
            let kv_elements = capacity
                .checked_mul(kv_dim)
                .ok_or_else(|| MoeBatch16Error::Validation("KV geometry overflow".into()))?;
            for (layer, tensor) in session.kv_k.iter().chain(&session.kv_v).enumerate() {
                require_tensor(
                    tensor,
                    &format!("slot {slot} KV[{layer}]"),
                    kv_dtype,
                    kv_elements,
                )?;
            }
            for (name, tensor, elements, dtype) in [
                ("x", &session.x, hidden, GgmlType::F32),
                ("h", &session.h, hidden, GgmlType::F32),
                (
                    "ffn_gate",
                    &session.ffn_gate,
                    shared_ffn.max(experts).max(expert_ffn),
                    GgmlType::F32,
                ),
                (
                    "ffn_up",
                    &session.ffn_up,
                    shared_ffn.max(experts).max(expert_ffn),
                    GgmlType::F32,
                ),
                (
                    "ffn_inner",
                    &session.ffn_inner,
                    shared_ffn.max(experts).max(expert_ffn),
                    GgmlType::F32,
                ),
                ("ffn_out", &session.ffn_out, hidden, GgmlType::F32),
                ("gdn_qkv", &session.gdn_qkv, conv_dim, GgmlType::F32),
                (
                    "gdn_qkv_conv",
                    &session.gdn_qkv_conv,
                    conv_dim,
                    GgmlType::F32,
                ),
                ("gdn_z", &session.gdn_z, value_dim, GgmlType::F32),
                (
                    "gdn_b",
                    &session.gdn_b,
                    arch.gdn_n_v_heads as usize,
                    GgmlType::F32,
                ),
                (
                    "gdn_beta",
                    &session.gdn_beta,
                    arch.gdn_n_v_heads as usize,
                    GgmlType::F32,
                ),
                (
                    "gdn_a",
                    &session.gdn_a,
                    arch.gdn_n_v_heads as usize,
                    GgmlType::F32,
                ),
                (
                    "gdn_alpha",
                    &session.gdn_alpha,
                    arch.gdn_n_v_heads as usize,
                    GgmlType::F32,
                ),
                ("gdn_q_norm", &session.gdn_q_norm, key_dim, GgmlType::F32),
                ("gdn_k_norm", &session.gdn_k_norm, key_dim, GgmlType::F32),
                ("gdn_out", &session.gdn_out, value_dim, GgmlType::F32),
                ("gdn_normed", &session.gdn_normed, value_dim, GgmlType::F32),
                ("mixer_out", &session.mixer_out, hidden, GgmlType::F32),
                (
                    "attn_q_full",
                    &session.attn_q_full,
                    2 * q_dim,
                    GgmlType::F32,
                ),
                ("attn_q", &session.attn_q, q_dim, GgmlType::F32),
                ("attn_gate", &session.attn_gate, q_dim, GgmlType::F32),
                (
                    "attn_q_normed",
                    &session.attn_q_normed,
                    q_dim,
                    GgmlType::F32,
                ),
                ("attn_k_now", &session.attn_k_now, kv_dim, GgmlType::F32),
                ("attn_v_now", &session.attn_v_now, kv_dim, GgmlType::F32),
                (
                    "attn_k_normed",
                    &session.attn_k_normed,
                    kv_dim,
                    GgmlType::F32,
                ),
                ("attn_o", &session.attn_o, q_dim, GgmlType::F32),
                (
                    "logits",
                    &session.logits,
                    arch.vocab_size as usize,
                    GgmlType::F32,
                ),
                ("argmax_tok", &session.argmax_tok, 1, GgmlType::I32),
                (
                    "moe_router_probs",
                    &session.moe_router_probs,
                    experts,
                    GgmlType::F32,
                ),
                ("moe_topk_idx", &session.moe_topk_idx, topk, GgmlType::F32),
                (
                    "moe_topk_weight",
                    &session.moe_topk_weight,
                    topk,
                    GgmlType::F32,
                ),
                (
                    "moe_shared_gate",
                    &session.moe_shared_gate,
                    1,
                    GgmlType::F32,
                ),
                (
                    "moe_inner",
                    &session.moe_inner,
                    topk * expert_ffn,
                    GgmlType::F32,
                ),
                (
                    "moe_expert_out",
                    &session.moe_expert_out,
                    topk * hidden,
                    GgmlType::F32,
                ),
                (
                    "attn_v4_o_partial",
                    &session.attn_v4_o_partial,
                    partial_o,
                    GgmlType::F32,
                ),
                (
                    "attn_v4_ml_partial",
                    &session.attn_v4_ml_partial,
                    partial_ml,
                    GgmlType::F32,
                ),
                ("ids_buf", &session.ids_buf, 1, GgmlType::I32),
            ] {
                require_tensor(tensor, &format!("slot {slot} {name}"), dtype, elements)?;
            }
        }
        for left in 0..MOE_BATCH16_WIDTH {
            for right in left + 1..MOE_BATCH16_WIDTH {
                if sessions[left].aliases_mutable_session(sessions[right]) {
                    return Err(MoeBatch16Error::Validation(format!(
                        "slots {left} and {right} alias mutable session storage"
                    )));
                }
            }
        }
        Ok(())
    }

    fn write_ids(&self, ids: &[i32; MOE_BATCH16_WIDTH]) -> Result<(), MoeBatch16Error> {
        require_tensor(
            &self.scratch.ids,
            "input IDs",
            GgmlType::I32,
            MOE_BATCH16_WIDTH,
        )?;
        let ptr = self.scratch.ids.buffer.contents().as_ptr().cast::<i32>();
        if ptr.is_null() {
            return Err(MoeBatch16Error::Validation(
                "input IDs are not CPU visible".into(),
            ));
        }
        let offset = usize::try_from(self.scratch.ids.offset / 4)
            .map_err(|_| MoeBatch16Error::Validation("input ID offset exceeds usize".into()))?;
        unsafe {
            std::ptr::copy_nonoverlapping(ids.as_ptr(), ptr.add(offset), MOE_BATCH16_WIDTH);
        }
        Ok(())
    }

    fn read_selections(&self) -> Result<[i32; MOE_BATCH16_WIDTH], MoeBatch16Error> {
        require_tensor(
            &self.scratch.selections,
            "greedy selections",
            GgmlType::I32,
            MOE_BATCH16_WIDTH,
        )?;
        let ptr = self
            .scratch
            .selections
            .buffer
            .contents()
            .as_ptr()
            .cast::<i32>();
        if ptr.is_null() {
            return Err(MoeBatch16Error::Validation(
                "greedy selections are not CPU visible".into(),
            ));
        }
        let offset = usize::try_from(self.scratch.selections.offset / 4).map_err(|_| {
            MoeBatch16Error::Validation("greedy selection offset exceeds usize".into())
        })?;
        let mut ids = [0; MOE_BATCH16_WIDTH];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.add(offset), ids.as_mut_ptr(), MOE_BATCH16_WIDTH);
        }
        Ok(ids)
    }

    fn encode_step(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        positions: [u32; MOE_BATCH16_WIDTH],
        sessions: &mut [&mut MetalSession; MOE_BATCH16_WIDTH],
    ) -> Result<(), MoeBatch16Error> {
        let arch = &self.forward.model.arch;
        let hidden = arch.hidden_size as usize;
        let vocab = arch.vocab_size as usize;
        let conv_heads = checked_mul(&[2, arch.gdn_n_k_heads as usize], "GDN head geometry")?
            .checked_add(arch.gdn_n_v_heads as usize)
            .ok_or_else(|| MoeBatch16Error::Validation("GDN head geometry overflow".into()))?;
        let conv_dim = checked_mul(
            &[conv_heads, arch.gdn_head_dim as usize],
            "GDN convolution geometry",
        )?;
        let value_dim = checked_mul(
            &[arch.gdn_n_v_heads as usize, arch.gdn_head_dim as usize],
            "GDN value geometry",
        )?;
        let encoder = KernelEncoder::begin(command);
        encode_get_rows_f32(
            self.forward.ctx,
            &encoder,
            &self.forward.model.token_embd,
            &self.scratch.ids,
            &self.scratch.rows,
            MOE_BATCH16_WIDTH,
            hidden,
        )?;
        encoder.end();
        self.blit_rows_to_sessions(command, &self.scratch.rows, sessions, |session| &session.x)?;

        let mut gdn_index = 0usize;
        for (block_index, block) in self.forward.model.blocks.iter().enumerate() {
            let block_plan = self.plan.blocks[block_index];
            match (block, block_plan.mixer) {
                (MetalBlock::Gdn(block), MixerMode::Gdn(GdnMixerMode::BatchedQ8)) => {
                    self.encode_gdn(
                        command, block, gdn_index, sessions, hidden, conv_dim, value_dim,
                    )?;
                    gdn_index += 1;
                }
                (MetalBlock::Gdn(_), MixerMode::Gdn(GdnMixerMode::PerLane)) => {
                    let encoder = KernelEncoder::begin(command);
                    for (slot, session) in sessions.iter_mut().enumerate() {
                        self.forward.encode_moe_mixer_prep_by_index(
                            &encoder,
                            block_index,
                            positions[slot],
                            session,
                        )?;
                    }
                    encoder.end();
                    gdn_index += 1;
                }
                (MetalBlock::Attn(_), MixerMode::Attention) => {
                    let encoder = KernelEncoder::begin(command);
                    for (slot, session) in sessions.iter_mut().enumerate() {
                        self.forward.encode_moe_mixer_prep_by_index(
                            &encoder,
                            block_index,
                            positions[slot],
                            session,
                        )?;
                    }
                    encoder.end();
                }
                _ => {
                    return Err(MoeBatch16Error::Validation(format!(
                        "block {block_index} execution plan no longer matches model"
                    )));
                }
            }
            match block_plan.routed {
                RoutedGateUpMode::PackedQ4 => {
                    self.encode_packed_ffn(command, block, block_index, sessions, hidden)?;
                }
                RoutedGateUpMode::PerLane => {
                    for session in sessions.iter_mut() {
                        self.forward
                            .encode_moe_ffn_after_mixer_production_by_index(
                                command,
                                block_index,
                                session,
                            )?;
                    }
                }
            }
        }

        let encoder = KernelEncoder::begin(command);
        for session in sessions.iter() {
            encode_rms_norm_mul_f32(
                self.forward.ctx,
                &encoder,
                &session.x,
                &self.forward.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
        }
        encoder.end();
        match self.plan.head {
            HeadMode::BatchedQ6 => {
                self.blit_rows_from_sessions(command, sessions, &self.scratch.rows, |session| {
                    &session.h
                })?;
                let encoder = KernelEncoder::begin(command);
                encode_mat_vec_q6_k_batch_f32(
                    self.forward.ctx,
                    &encoder,
                    &self.forward.model.lm_head,
                    &self.scratch.rows,
                    &self.scratch.logits,
                    hidden,
                    vocab,
                    MOE_BATCH16_WIDTH,
                )?;
                encode_argmax_f32_greedy(
                    self.forward.ctx,
                    &encoder,
                    &self.scratch.logits,
                    &self.scratch.selections,
                    MOE_BATCH16_WIDTH,
                    vocab,
                )?;
                encoder.end();
                self.blit_logits(command, sessions, vocab)?;
            }
            HeadMode::BatchedQ8 => {
                self.blit_rows_from_sessions(command, sessions, &self.scratch.rows, |session| {
                    &session.h
                })?;
                let encoder = KernelEncoder::begin(command);
                encode_mat_vec_q8_0_batch_f32(
                    self.forward.ctx,
                    &encoder,
                    &self.forward.model.lm_head,
                    &self.scratch.rows,
                    &self.scratch.logits,
                    hidden,
                    vocab,
                    MOE_BATCH16_WIDTH,
                )?;
                encode_argmax_f32_greedy(
                    self.forward.ctx,
                    &encoder,
                    &self.scratch.logits,
                    &self.scratch.selections,
                    MOE_BATCH16_WIDTH,
                    vocab,
                )?;
                encoder.end();
                self.blit_logits(command, sessions, vocab)?;
            }
            HeadMode::PerLane => {
                let encoder = KernelEncoder::begin(command);
                for session in sessions.iter() {
                    encode_mat_vec_dispatch(
                        self.forward.ctx,
                        &encoder,
                        &self.forward.model.lm_head,
                        &session.h,
                        &session.logits,
                        hidden,
                        vocab,
                    )?;
                }
                encoder.end();
                self.blit_session_logits_to_pack(command, sessions, vocab)?;
                let encoder = KernelEncoder::begin(command);
                encode_argmax_f32_greedy(
                    self.forward.ctx,
                    &encoder,
                    &self.scratch.logits,
                    &self.scratch.selections,
                    MOE_BATCH16_WIDTH,
                    vocab,
                )?;
                encoder.end();
            }
        }
        Ok(())
    }

    fn encode_gdn(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        block: &MetalGdnBlock,
        gdn_index: usize,
        sessions: &mut [&mut MetalSession; MOE_BATCH16_WIDTH],
        hidden: usize,
        conv_dim: usize,
        value_dim: usize,
    ) -> Result<(), MoeBatch16Error> {
        let encoder = KernelEncoder::begin(command);
        for session in sessions.iter() {
            encode_rms_norm_mul_f32(
                self.forward.ctx,
                &encoder,
                &session.x,
                &block.attn_norm,
                &session.h,
                RMS_EPS,
            )?;
        }
        encoder.end();
        self.blit_rows_from_sessions(command, sessions, &self.scratch.gdn.h, |session| &session.h)?;
        let encoder = KernelEncoder::begin(command);
        encode_mat_vec_q8_0_batch_f32(
            self.forward.ctx,
            &encoder,
            &block.in_proj_qkv,
            &self.scratch.gdn.h,
            &self.scratch.gdn.qkv,
            hidden,
            conv_dim,
            MOE_BATCH16_WIDTH,
        )?;
        encode_mat_vec_q8_0_batch_f32(
            self.forward.ctx,
            &encoder,
            &block.in_proj_z,
            &self.scratch.gdn.h,
            &self.scratch.gdn.z,
            hidden,
            value_dim,
            MOE_BATCH16_WIDTH,
        )?;
        for (slot, session) in sessions.iter_mut().enumerate() {
            let qkv = self
                .scratch
                .gdn
                .qkv
                .view_subrange((slot * conv_dim) as u64, vec![conv_dim as u64]);
            let z = self
                .scratch
                .gdn
                .z
                .view_subrange((slot * value_dim) as u64, vec![value_dim as u64]);
            let normed = self
                .scratch
                .gdn
                .normed
                .view_subrange((slot * value_dim) as u64, vec![value_dim as u64]);
            self.forward.encode_gdn_after_batched_front(
                &encoder, block, gdn_index, session, &qkv, &z, &normed,
            )?;
        }
        encode_mat_vec_q8_0_batch_f32(
            self.forward.ctx,
            &encoder,
            &block.out_proj,
            &self.scratch.gdn.normed,
            &self.scratch.gdn.out,
            value_dim,
            hidden,
            MOE_BATCH16_WIDTH,
        )?;
        for (slot, session) in sessions.iter().enumerate() {
            let output = self
                .scratch
                .gdn
                .out
                .view_subrange((slot * hidden) as u64, vec![hidden as u64]);
            self.forward.encode_post_mixer_norm(
                &encoder,
                &session.x,
                &output,
                &block.post_attn_norm,
                &session.h,
            )?;
        }
        encoder.end();
        Ok(())
    }

    fn encode_packed_ffn(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        block: &MetalBlock,
        block_index: usize,
        sessions: &mut [&mut MetalSession; MOE_BATCH16_WIDTH],
        hidden: usize,
    ) -> Result<(), MoeBatch16Error> {
        let moe = match block {
            MetalBlock::Gdn(block) => block.ffn_moe.as_ref(),
            MetalBlock::Attn(block) => block.ffn_moe.as_ref(),
        }
        .ok_or_else(|| {
            MoeBatch16Error::Validation(format!("block {block_index} lost MoE weights"))
        })?;
        let topk = self.forward.model.arch.expert_used_count as usize;
        let expert_ffn = self.forward.model.arch.expert_feed_forward_length as usize;
        let experts = self.forward.model.arch.expert_count as usize;
        let encoder = KernelEncoder::begin(command);
        for session in sessions.iter_mut() {
            self.forward
                .encode_moe_route_prepare_by_index(&encoder, block_index, session)?;
        }
        encoder.end();
        self.blit_rows_from_sessions(command, sessions, &self.scratch.rows, |session| &session.h)?;
        let idx_bytes = (topk * std::mem::size_of::<i32>()) as u64;
        let blit = BlitEncoder::begin(command);
        for (slot, session) in sessions.iter().enumerate() {
            blit.copy_buffer(
                &session.moe_topk_idx.buffer,
                session.moe_topk_idx.offset,
                &self.scratch.topk_idx.buffer,
                self.scratch.topk_idx.offset + slot as u64 * idx_bytes,
                idx_bytes,
            );
        }
        blit.end();
        let encoder = KernelEncoder::begin_concurrent(command);
        encode_moe_swiglu_q4_K_f32_packed_slots(
            self.forward.ctx,
            &encoder,
            &moe.gate_exps,
            &moe.up_exps,
            &self.scratch.rows,
            &self.scratch.topk_idx,
            &self.scratch.routed_inner,
            hidden,
            expert_ffn,
            experts,
            topk,
            MOE_BATCH16_WIDTH,
        )?;
        let mut shared_fused = [false; MOE_BATCH16_WIDTH];
        for (slot, session) in sessions.iter_mut().enumerate() {
            shared_fused[slot] =
                self.forward
                    .encode_moe_shared_gate_up_by_index(&encoder, block_index, session)?;
        }
        encoder.end();
        let inner_bytes = (topk * expert_ffn * std::mem::size_of::<f32>()) as u64;
        let blit = BlitEncoder::begin(command);
        for (slot, session) in sessions.iter().enumerate() {
            blit.copy_buffer(
                &self.scratch.routed_inner.buffer,
                self.scratch.routed_inner.offset + slot as u64 * inner_bytes,
                &session.moe_inner.buffer,
                session.moe_inner.offset,
                inner_bytes,
            );
        }
        blit.end();
        for (slot, session) in sessions.iter_mut().enumerate() {
            self.forward
                .encode_moe_ffn_after_external_routed_inner_by_index(
                    command,
                    block_index,
                    session,
                    shared_fused[slot],
                )?;
        }
        Ok(())
    }

    fn blit_rows_to_sessions(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        source: &MetalTensor,
        sessions: &[&mut MetalSession; MOE_BATCH16_WIDTH],
        destination: impl Fn(&MetalSession) -> &MetalTensor,
    ) -> Result<(), MoeBatch16Error> {
        let row_bytes = u64::try_from(checked_mul(
            &[self.forward.model.arch.hidden_size as usize, 4],
            "row byte width",
        )?)
        .map_err(|_| MoeBatch16Error::Validation("row byte width overflow".into()))?;
        let blit = BlitEncoder::begin(command);
        for (slot, session) in sessions.iter().enumerate() {
            let destination = destination(session);
            blit.copy_buffer(
                &source.buffer,
                source.offset + slot as u64 * row_bytes,
                &destination.buffer,
                destination.offset,
                row_bytes,
            );
        }
        blit.end();
        Ok(())
    }

    fn blit_rows_from_sessions(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        sessions: &[&mut MetalSession; MOE_BATCH16_WIDTH],
        destination: &MetalTensor,
        source: impl Fn(&MetalSession) -> &MetalTensor,
    ) -> Result<(), MoeBatch16Error> {
        let row_bytes = u64::try_from(checked_mul(
            &[self.forward.model.arch.hidden_size as usize, 4],
            "row byte width",
        )?)
        .map_err(|_| MoeBatch16Error::Validation("row byte width overflow".into()))?;
        let blit = BlitEncoder::begin(command);
        for (slot, session) in sessions.iter().enumerate() {
            let source = source(session);
            blit.copy_buffer(
                &source.buffer,
                source.offset,
                &destination.buffer,
                destination.offset + slot as u64 * row_bytes,
                row_bytes,
            );
        }
        blit.end();
        Ok(())
    }

    fn blit_logits(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        sessions: &[&mut MetalSession; MOE_BATCH16_WIDTH],
        vocab: usize,
    ) -> Result<(), MoeBatch16Error> {
        let row_bytes = u64::try_from(checked_mul(&[vocab, 4], "logit byte width")?)
            .map_err(|_| MoeBatch16Error::Validation("logit byte width overflow".into()))?;
        let blit = BlitEncoder::begin(command);
        for (slot, session) in sessions.iter().enumerate() {
            blit.copy_buffer(
                &self.scratch.logits.buffer,
                self.scratch.logits.offset + slot as u64 * row_bytes,
                &session.logits.buffer,
                session.logits.offset,
                row_bytes,
            );
        }
        blit.end();
        Ok(())
    }

    fn blit_session_logits_to_pack(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        sessions: &[&mut MetalSession; MOE_BATCH16_WIDTH],
        vocab: usize,
    ) -> Result<(), MoeBatch16Error> {
        let row_bytes = u64::try_from(checked_mul(&[vocab, 4], "logit byte width")?)
            .map_err(|_| MoeBatch16Error::Validation("logit byte width exceeds u64".into()))?;
        let blit = BlitEncoder::begin(command);
        for (slot, session) in sessions.iter().enumerate() {
            blit.copy_buffer(
                &session.logits.buffer,
                session.logits.offset,
                &self.scratch.logits.buffer,
                self.scratch.logits.offset + slot as u64 * row_bytes,
                row_bytes,
            );
        }
        blit.end();
        Ok(())
    }

    fn restore_frontiers(
        sessions: &mut [&mut MetalSession; MOE_BATCH16_WIDTH],
        positions: [u32; MOE_BATCH16_WIDTH],
    ) {
        for (slot, session) in sessions.iter_mut().enumerate() {
            session.kv_n_pos.fill(positions[slot] as usize);
        }
    }

    fn poison_sessions(
        sessions: &mut [&mut MetalSession; MOE_BATCH16_WIDTH],
        reason: &'static str,
    ) {
        for session in sessions {
            session.poison(reason);
        }
    }
}

fn require_tensor(
    tensor: &MetalTensor,
    name: &str,
    dtype: GgmlType,
    elements: usize,
) -> Result<(), MoeBatch16Error> {
    if tensor.dtype != dtype || tensor.n_elements() as usize != elements || !tensor.is_writable() {
        return Err(MoeBatch16Error::Validation(format!(
            "{name} contract mismatch: dtype={:?} elements={} writable={} expected_dtype={dtype:?} expected_elements={elements}",
            tensor.dtype,
            tensor.n_elements(),
            tensor.is_writable()
        )));
    }
    let alignment = match dtype {
        GgmlType::F16 => 2,
        GgmlType::F32 | GgmlType::I32 => 4,
        GgmlType::Q8_0 => 32,
        _ => 1,
    };
    let end = tensor
        .offset
        .checked_add(tensor.n_bytes())
        .ok_or_else(|| MoeBatch16Error::Validation(format!("{name} byte range overflow")))?;
    if !tensor.offset.is_multiple_of(alignment) || end > tensor.buffer.length() as u64 {
        return Err(MoeBatch16Error::Validation(format!(
            "{name} byte range is unaligned or exceeds its buffer"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a3b_arch() -> Arch {
        Arch {
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
        }
    }

    #[test]
    fn scratch_accounting_is_stable_for_a3b() {
        assert_eq!(moe_batch16_scratch_bytes(&a3b_arch()).unwrap(), 17_597_056);
    }

    #[test]
    fn scratch_accounting_rejects_invalid_expert_geometry() {
        let mut arch = a3b_arch();
        arch.expert_used_count = arch.expert_count + 1;
        assert!(matches!(
            moe_batch16_scratch_bytes(&arch),
            Err(MoeBatch16Error::Unsupported(_))
        ));
    }

    #[test]
    fn scratch_accounting_detects_geometry_overflow() {
        let mut arch = a3b_arch();
        arch.expert_count = u32::MAX;
        arch.expert_feed_forward_length = u32::MAX;
        arch.hidden_size = u32::MAX;
        assert!(moe_batch16_scratch_bytes(&arch).is_err());
    }

    #[test]
    fn a3b_selects_all_exact_batched_modes() {
        let arch = a3b_arch();
        validate_architecture(&arch).unwrap();
        assert_eq!(
            select_gdn_mode(
                GgmlType::Q8_0,
                GgmlType::Q8_0,
                GgmlType::Q8_0,
                2048,
                8192,
                4096
            )
            .unwrap(),
            GdnMixerMode::BatchedQ8
        );
        assert_eq!(
            select_routed_mode(GgmlType::Q4_K, GgmlType::Q4_K, 2048, 512, 256, 8).unwrap(),
            RoutedGateUpMode::PackedQ4
        );
        assert_eq!(
            select_head_mode(GgmlType::Q6_K, 2048, 248_320).unwrap(),
            HeadMode::BatchedQ6
        );
    }

    #[test]
    fn mtp_metadata_does_not_change_base_eligibility() {
        let mut arch = a3b_arch();
        arch.mtp_n_hidden_layers = 2;
        validate_architecture(&arch).unwrap();
    }

    #[test]
    fn a10b_geometry_and_mixed_routed_modes_are_supported() {
        let mut arch = a3b_arch();
        arch.n_layer = 48;
        arch.hidden_size = 3072;
        arch.n_q_heads = 32;
        arch.gdn_n_v_heads = 64;
        arch.expert_feed_forward_length = 1024;
        arch.expert_shared_feed_forward_length = 1024;
        validate_architecture(&arch).unwrap();
        assert_eq!(
            select_gdn_mode(
                GgmlType::Q8_0,
                GgmlType::Q8_0,
                GgmlType::Q8_0,
                3072,
                12_288,
                8192,
            )
            .unwrap(),
            GdnMixerMode::BatchedQ8
        );
        assert_eq!(
            select_routed_mode(GgmlType::Q4_K, GgmlType::Q4_K, 3072, 1024, 256, 8).unwrap(),
            RoutedGateUpMode::PackedQ4
        );
        assert_eq!(
            select_routed_mode(GgmlType::Q5_K, GgmlType::Q5_K, 3072, 1024, 256, 8).unwrap(),
            RoutedGateUpMode::PerLane
        );
        assert_eq!(
            select_head_mode(GgmlType::Q8_0, 3072, 248_320).unwrap(),
            HeadMode::BatchedQ8
        );
    }

    #[test]
    fn iq3_routed_falls_back_while_q8_head_batches() {
        assert_eq!(
            select_routed_mode(GgmlType::IQ3_S, GgmlType::IQ3_S, 3072, 1024, 256, 8,).unwrap(),
            RoutedGateUpMode::PerLane
        );
        assert_eq!(
            select_head_mode(GgmlType::Q8_0, 3072, 248_320).unwrap(),
            HeadMode::BatchedQ8
        );
    }

    #[test]
    fn unsupported_modes_and_no_batched_stage_are_rejected() {
        assert!(select_routed_mode(GgmlType::F16, GgmlType::F16, 2048, 512, 256, 8).is_err());
        assert!(select_head_mode(GgmlType::I32, 2048, 1024).is_err());
        assert!(!has_meaningful_batched_stage(
            [GdnMixerMode::PerLane],
            [RoutedGateUpMode::PerLane],
            HeadMode::PerLane,
        ));
    }

    #[test]
    fn architecture_rejects_overflow_and_true_mixer_abi_failures() {
        let mut overflow = a3b_arch();
        overflow.hidden_size = u32::MAX;
        overflow.expert_count = u32::MAX;
        overflow.expert_feed_forward_length = u32::MAX;
        assert!(validate_architecture(&overflow).is_err());

        let mut bad_gdn = a3b_arch();
        bad_gdn.gdn_head_dim = 64;
        assert!(validate_architecture(&bad_gdn).is_err());
        bad_gdn = a3b_arch();
        bad_gdn.gdn_n_v_heads = 31;
        assert!(validate_architecture(&bad_gdn).is_err());

        let mut bad_attention = a3b_arch();
        bad_attention.attn_head_dim = 128;
        assert!(validate_architecture(&bad_attention).is_err());
        bad_attention = a3b_arch();
        bad_attention.n_q_heads = 15;
        assert!(validate_architecture(&bad_attention).is_err());

        let mut zero_layers = a3b_arch();
        zero_layers.n_layer = 0;
        assert!(validate_architecture(&zero_layers).is_err());

        let mut fractional_rope = a3b_arch();
        fractional_rope.partial_rotary_factor = 64.5 / 256.0;
        assert!(validate_architecture(&fractional_rope).is_err());

        assert!(
            validate_matvec_contract(GgmlType::Q8_0, 2048, u32::MAX as usize + 1, "test").is_err()
        );
    }
}
