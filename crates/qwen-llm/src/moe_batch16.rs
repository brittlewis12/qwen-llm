//! Product fixed-width MoE decode for the Qwen 35B-A3B execution profile.
//!
//! This intentionally mirrors the measured B=16 composition rather than
//! extending the dense batching machinery. Causal state, routing decisions,
//! expert slot order, and all down/final waves remain sequence-owned.

use crate::env_flag::read_default_off;
use crate::metal::{
    BlitEncoder, KernelEncoder, MetalContext, MetalError, MetalTensor, encode_argmax_f32_greedy,
    encode_get_rows_f32, encode_mat_vec_q6_k_batch_f32, encode_mat_vec_q8_0_batch_f32,
    encode_moe_swiglu_q4_K_f32_packed_slots, encode_rms_norm_mul_f32,
};
use crate::metal_forward::{
    ATTN_V4_MAX_NWG, MetalBlock, MetalForward, MetalGdnBlock, MetalModel, MetalSession, MfError,
    RMS_EPS,
};
use crate::model::{Arch, ArchKind};
use crate::sampling::{GreedySelection, SamplingError};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};

pub const MOE_BATCH16_WIDTH: usize = 16;

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

/// Logical bytes allocated by the executor's persistent B=16 scratch arena.
/// This excludes model weights and sequence-owned causal state.
pub fn moe_batch16_scratch_bytes(arch: &Arch) -> Result<u64, MoeBatch16Error> {
    let (hidden, topk, _, expert_ffn, value_dim) = geometry(arch)?;
    let conv_heads = (arch.gdn_n_k_heads as usize)
        .checked_mul(2)
        .and_then(|value| value.checked_add(arch.gdn_n_v_heads as usize))
        .ok_or_else(|| MoeBatch16Error::Validation("GDN scratch geometry overflow".into()))?;
    let conv_dim = checked_mul(
        &[conv_heads, arch.gdn_head_dim as usize],
        "GDN scratch geometry",
    )?;
    let vocab = arch.vocab_size as usize;
    let f32_elements = [
        hidden, // embedding/final-head rows
        vocab,  // logits
        topk * expert_ffn,
        hidden, // GDN normalized input rows
        conv_dim,
        value_dim,
        value_dim,
        hidden,
    ]
    .into_iter()
    .try_fold(0usize, |sum, width| {
        let elements = checked_mul(&[MOE_BATCH16_WIDTH, width], "scratch elements")?;
        sum.checked_add(elements)
            .ok_or_else(|| MoeBatch16Error::Validation("scratch element sum overflow".into()))
    })?;
    let i32_elements = checked_mul(&[MOE_BATCH16_WIDTH, 2 + topk], "I32 scratch")?;
    let bytes = (f32_elements as u128)
        .checked_mul(4)
        .and_then(|value| value.checked_add((i32_elements as u128) * 4))
        .ok_or_else(|| MoeBatch16Error::Validation("scratch byte count overflow".into()))?;
    u64::try_from(bytes)
        .map_err(|_| MoeBatch16Error::Validation("scratch byte count exceeds u64".into()))
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

fn validate_capabilities(model: &MetalModel) -> Result<(), MoeBatch16Error> {
    let arch = &model.arch;
    if arch.kind != ArchKind::Moe {
        return Err(MoeBatch16Error::Unsupported(
            "model architecture is not Qwen MoE".into(),
        ));
    }
    geometry(arch)?;
    // The production qualification currently covers the 35B-A3B backbone.
    if arch.n_layer != 40
        || arch.hidden_size != 2048
        || arch.intermediate_size != 0
        || arch.vocab_size != 248_320
        || arch.full_attention_interval != 4
        || arch.n_q_heads != 16
        || arch.n_kv_heads != 2
        || arch.attn_head_dim != 256
        || arch.rope_theta.to_bits() != 10_000_000.0f32.to_bits()
        || arch.partial_rotary_factor.to_bits() != 0.25f32.to_bits()
        || arch.gdn_n_v_heads != 32
        || arch.gdn_n_k_heads != 16
        || arch.gdn_head_dim != 128
        || arch.gdn_conv_kernel != 4
        || arch.expert_count != 256
        || arch.expert_used_count != 8
        || arch.expert_feed_forward_length != 512
        || arch.expert_shared_feed_forward_length != 512
        || arch.mtp_n_hidden_layers != 0
    {
        return Err(MoeBatch16Error::Unsupported(
            "only the qualified Qwen 35B-A3B geometry is currently supported".into(),
        ));
    }
    if model.blocks.len() != arch.n_layer as usize {
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
    if model.lm_head.dtype != GgmlType::Q6_K {
        return Err(MoeBatch16Error::Unsupported(format!(
            "LM head must be Q6_K, got {:?}",
            model.lm_head.dtype
        )));
    }
    let hidden = arch.hidden_size as usize;
    let vocab = arch.vocab_size as usize;
    let experts = arch.expert_count as usize;
    let expert_ffn = arch.expert_feed_forward_length as usize;
    let shared_ffn = arch.expert_shared_feed_forward_length as usize;
    let conv_dim = (2 * arch.gdn_n_k_heads as usize + arch.gdn_n_v_heads as usize)
        * arch.gdn_head_dim as usize;
    let value_dim = arch.gdn_n_v_heads as usize * arch.gdn_head_dim as usize;
    let q_dim = arch.n_q_heads as usize * arch.attn_head_dim as usize;
    let kv_dim = arch.n_kv_heads as usize * arch.attn_head_dim as usize;
    if model.token_embd.dtype != GgmlType::Q8_0 {
        return Err(MoeBatch16Error::Unsupported(format!(
            "token embedding must be Q8_0, got {:?}",
            model.token_embd.dtype
        )));
    }
    require_weight_shape(&model.token_embd, "token embedding", &[hidden, vocab])?;
    require_weight_shape(&model.output_norm, "output norm", &[hidden])?;
    require_weight_shape(&model.lm_head, "LM head", &[hidden, vocab])?;
    for (index, block) in model.blocks.iter().enumerate() {
        let (shared_gate, shared_up, shared_down, moe) = match block {
            MetalBlock::Gdn(block) => {
                for (name, tensor) in [
                    ("qkv", &block.in_proj_qkv),
                    ("z", &block.in_proj_z),
                    ("out", &block.out_proj),
                ] {
                    if tensor.dtype != GgmlType::Q8_0 {
                        return Err(MoeBatch16Error::Unsupported(format!(
                            "block {index} GDN {name} projection must be Q8_0, got {:?}",
                            tensor.dtype
                        )));
                    }
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
                (
                    &block.ffn_gate,
                    &block.ffn_up,
                    &block.ffn_down,
                    block.ffn_moe.as_ref(),
                )
            }
            MetalBlock::Attn(block) => {
                for (name, tensor) in [
                    ("Q", &block.q),
                    ("K", &block.k),
                    ("V", &block.v),
                    ("O", &block.o),
                ] {
                    if !matvec_dtype(tensor.dtype) {
                        return Err(MoeBatch16Error::Unsupported(format!(
                            "block {index} attention {name} dtype {:?} has no production mat-vec contract",
                            tensor.dtype
                        )));
                    }
                }
                for (name, tensor, dimensions) in [
                    ("Q", &block.q, [hidden, 2 * q_dim]),
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
                (
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
        if moe.gate_exps.dtype != GgmlType::Q4_K || moe.up_exps.dtype != GgmlType::Q4_K {
            return Err(MoeBatch16Error::Unsupported(format!(
                "block {index} routed gate/up banks must both be Q4_K"
            )));
        }
        if !matches!(moe.gate_inp.dtype, GgmlType::F32 | GgmlType::F16)
            || moe.gate_inp_shexp.dtype != GgmlType::F32
        {
            return Err(MoeBatch16Error::Unsupported(format!(
                "block {index} router/shared-gate dtype contract is unsupported"
            )));
        }
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
        for (name, tensor) in [
            ("shared gate", shared_gate),
            ("shared up", shared_up),
            ("shared down", shared_down),
        ] {
            if !matvec_dtype(tensor.dtype) {
                return Err(MoeBatch16Error::Unsupported(format!(
                    "block {index} {name} dtype {:?} is unsupported",
                    tensor.dtype
                )));
            }
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
    }
    Ok(())
}

pub(crate) struct MoeBatch16Executor<'a> {
    forward: MetalForward<'a>,
    scratch: MoeBatch16Scratch,
    poisoned: bool,
}

impl<'a> MoeBatch16Executor<'a> {
    pub(crate) fn new(
        ctx: &'a MetalContext,
        model: &'a MetalModel,
    ) -> Result<Self, MoeBatch16Error> {
        // All capability checks deliberately precede the large logits/inner allocations.
        validate_capabilities(model)?;
        Ok(Self {
            forward: MetalForward::new(ctx, model),
            scratch: MoeBatch16Scratch::new(ctx, model)?,
            poisoned: false,
        })
    }

    pub(crate) fn scratch_bytes(&self) -> u64 {
        self.scratch.bytes
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub(crate) fn validate_refs(
        &self,
        token_ids: [i32; MOE_BATCH16_WIDTH],
        position: u32,
        sessions: &mut [&mut MetalSession; MOE_BATCH16_WIDTH],
    ) -> Result<(), MoeBatch16Error> {
        if self.poisoned {
            return Err(MoeBatch16Error::Poisoned);
        }
        self.validate_step(&token_ids, position, sessions)
    }

    pub(crate) fn step_greedy_refs(
        &mut self,
        token_ids: [i32; MOE_BATCH16_WIDTH],
        position: u32,
        sessions: &mut [&mut MetalSession; MOE_BATCH16_WIDTH],
        cancelled: impl Fn() -> bool,
    ) -> Result<MoeBatch16Step, MoeBatch16Error> {
        if self.poisoned {
            return Err(MoeBatch16Error::Poisoned);
        }
        if cancelled() {
            return Err(MoeBatch16Error::CancelledBeforeCommit);
        }
        self.validate_step(&token_ids, position, sessions)?;
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
            self.encode_step(&command, position, sessions)?;
            Ok::<bool, MoeBatch16Error>(cancelled())
        }));
        match encoded {
            Ok(Ok(false)) => {}
            Ok(Ok(true)) => {
                Self::restore_frontiers(sessions, position as usize);
                return Err(MoeBatch16Error::CancelledBeforeCommit);
            }
            Ok(Err(error)) => {
                Self::restore_frontiers(sessions, position as usize);
                return Err(error);
            }
            Err(payload) => {
                Self::restore_frontiers(sessions, position as usize);
                std::panic::resume_unwind(payload);
            }
        }
        command.commit();
        command.waitUntilCompleted();
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
        position: u32,
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
        let position = position as usize;
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
        position: u32,
        sessions: &mut [&mut MetalSession; MOE_BATCH16_WIDTH],
    ) -> Result<(), MoeBatch16Error> {
        let arch = &self.forward.model.arch;
        let hidden = arch.hidden_size as usize;
        let vocab = arch.vocab_size as usize;
        let conv_dim = (2 * arch.gdn_n_k_heads as usize + arch.gdn_n_v_heads as usize)
            * arch.gdn_head_dim as usize;
        let value_dim = arch.gdn_n_v_heads as usize * arch.gdn_head_dim as usize;
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
            match block {
                MetalBlock::Gdn(block) => {
                    self.encode_gdn(
                        command, block, gdn_index, sessions, hidden, conv_dim, value_dim,
                    )?;
                    gdn_index += 1;
                }
                MetalBlock::Attn(_) => {
                    let encoder = KernelEncoder::begin(command);
                    for session in sessions.iter_mut() {
                        self.forward.encode_moe_mixer_prep_by_index(
                            &encoder,
                            block_index,
                            position,
                            session,
                        )?;
                    }
                    encoder.end();
                }
            }
            self.encode_packed_ffn(command, block, block_index, sessions, hidden)?;
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
        self.blit_rows_from_sessions(command, sessions, &self.scratch.rows, |session| &session.h)?;
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
        let row_bytes = u64::try_from(self.forward.model.arch.hidden_size as usize * 4)
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
        let row_bytes = u64::try_from(self.forward.model.arch.hidden_size as usize * 4)
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
        let row_bytes = u64::try_from(vocab * 4)
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

    fn restore_frontiers(sessions: &mut [&mut MetalSession; MOE_BATCH16_WIDTH], position: usize) {
        for session in sessions {
            session.kv_n_pos.fill(position);
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
}
