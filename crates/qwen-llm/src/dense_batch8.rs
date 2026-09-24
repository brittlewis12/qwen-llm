//! Fixed-width dense decode with cross-sequence weight reuse.
//!
//! This is deliberately not a generic batching engine. It is the measured
//! eight-slot dense-Qwen backend: mutable recurrent/KV state remains private
//! while immutable-weight projections and the LM head run as B=8 matrices.

use crate::env_flag::read_default_off;
use crate::metal::{
    BlitEncoder, KernelEncoder, MetalContext, MetalError, MetalTensor, encode_add_inplace_f32,
    encode_argmax_f32, encode_argmax_f32_greedy, encode_get_rows_f32, encode_rms_norm_mul_f32,
    encode_silu_mul_f32,
};
use crate::metal_forward::{
    ATTN_V4_MAX_NWG, MetalAttnBlock, MetalBlock, MetalForward, MetalGdnBlock, MetalModel,
    MetalSession, MfError, RMS_EPS, encode_mat_mat_dispatch,
};
use crate::model::{Arch, ArchKind};
use crate::sampling::{GreedySelection, SamplingError};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};

pub const DENSE_BATCH8_WIDTH: usize = 8;

#[derive(Debug, thiserror::Error)]
pub enum DenseBatch8Error {
    #[error("dense B=8 unsupported: {0}")]
    Unsupported(String),
    #[error("dense B=8 validation: {0}")]
    Validation(String),
    #[error("dense B=8 cancelled before commit")]
    CancelledBeforeCommit,
    #[error("dense B=8 executor is poisoned after a committed command failure")]
    Poisoned,
    #[error("dense B=8 command failed: status={status} error={error}")]
    CommandBuffer { status: String, error: String },
    #[error("dense B=8 metal: {0}")]
    Metal(#[from] MetalError),
    #[error("dense B=8 forward: {0}")]
    Forward(#[from] MfError),
    #[error("dense B=8 greedy selection failed in slot {slot}: {source}")]
    GreedySelection {
        slot: usize,
        #[source]
        source: SamplingError,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct DenseBatch8Step {
    pub argmax_ids: [i32; DENSE_BATCH8_WIDTH],
    pub gpu_ms: Option<f64>,
}

#[derive(Clone, Copy)]
enum Reduction {
    LowestIndex,
    GreedyTotal,
}

struct GdnScratch {
    h: MetalTensor,
    qkv: MetalTensor,
    z: MetalTensor,
    normed: MetalTensor,
    out: MetalTensor,
}

struct FfnScratch {
    h: MetalTensor,
    gate: MetalTensor,
    up: MetalTensor,
    inner: MetalTensor,
    out: MetalTensor,
}

struct AttnScratch {
    h: MetalTensor,
    q: MetalTensor,
    k: MetalTensor,
    v: MetalTensor,
}

struct DenseBatch8Scratch {
    ids: MetalTensor,
    rows: MetalTensor,
    logits: MetalTensor,
    argmax: MetalTensor,
    gdn: GdnScratch,
    ffn: FfnScratch,
    attn: AttnScratch,
}

fn checked_elements(width: usize, label: &str) -> Result<u64, DenseBatch8Error> {
    DENSE_BATCH8_WIDTH
        .checked_mul(width)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| DenseBatch8Error::Validation(format!("{label} scratch size overflow")))
}

fn dense_batch8_scratch_logical_buffers(arch: &Arch) -> Result<Vec<u64>, DenseBatch8Error> {
    let hidden = arch.hidden_size as usize;
    let ffn = arch.intermediate_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = n_k
        .checked_mul(2)
        .and_then(|value| value.checked_add(n_v))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| DenseBatch8Error::Validation("GDN scratch width overflow".into()))?;
    let value_dim = n_v
        .checked_mul(head_dim)
        .ok_or_else(|| DenseBatch8Error::Validation("GDN value width overflow".into()))?;
    let q_dim = (arch.n_q_heads as usize)
        .checked_mul(arch.attn_head_dim as usize)
        .ok_or_else(|| DenseBatch8Error::Validation("attention Q width overflow".into()))?;
    let kv_dim = (arch.n_kv_heads as usize)
        .checked_mul(arch.attn_head_dim as usize)
        .ok_or_else(|| DenseBatch8Error::Validation("attention KV width overflow".into()))?;
    let f32 = |width, label| {
        checked_elements(width, label)?
            .checked_mul(4)
            .ok_or_else(|| DenseBatch8Error::Validation(format!("{label} bytes overflow")))
    };
    Ok(vec![
        u64::try_from(DENSE_BATCH8_WIDTH * std::mem::size_of::<i32>())
            .map_err(|_| DenseBatch8Error::Validation("ID scratch overflow".into()))?,
        f32(hidden, "hidden")?,
        f32(arch.vocab_size as usize, "logits")?,
        u64::try_from(DENSE_BATCH8_WIDTH * std::mem::size_of::<i32>())
            .map_err(|_| DenseBatch8Error::Validation("argmax scratch overflow".into()))?,
        f32(hidden, "GDN hidden")?,
        f32(conv_dim, "GDN QKV")?,
        f32(value_dim, "GDN Z")?,
        f32(value_dim, "GDN normalized")?,
        f32(hidden, "GDN output")?,
        f32(hidden, "FFN hidden")?,
        f32(ffn, "FFN gate")?,
        f32(ffn, "FFN up")?,
        f32(ffn, "FFN inner")?,
        f32(hidden, "FFN output")?,
        f32(hidden, "attention hidden")?,
        f32(2 * q_dim, "attention Q")?,
        f32(kv_dim, "attention K")?,
        f32(kv_dim, "attention V")?,
    ])
}

/// Logical bytes allocated by the executor's persistent B=8 scratch tensors.
pub fn dense_batch8_scratch_bytes(arch: &Arch) -> Result<u64, DenseBatch8Error> {
    dense_batch8_scratch_logical_buffers(arch)?
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total.checked_add(bytes).ok_or_else(|| {
                DenseBatch8Error::Validation("dense B=8 scratch bytes overflow".into())
            })
        })
}

/// Metal-allocation upper bound for the executor's independent scratch buffers.
pub fn dense_batch8_scratch_upper_bytes(
    ctx: &MetalContext,
    arch: &Arch,
) -> Result<u64, DenseBatch8Error> {
    dense_batch8_scratch_logical_buffers(arch)?
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            let priced = ctx.shared_buffer_size_and_align(bytes)?.size;
            total.checked_add(priced).ok_or_else(|| {
                DenseBatch8Error::Validation("dense B=8 priced scratch overflow".into())
            })
        })
}

fn require_writable_tensor(
    tensor: &MetalTensor,
    name: &str,
    dtype: GgmlType,
    elements: usize,
) -> Result<(), DenseBatch8Error> {
    if tensor.dtype != dtype || tensor.n_elements() as usize != elements || !tensor.is_writable() {
        return Err(DenseBatch8Error::Validation(format!(
            "{name} contract mismatch: dtype={:?} elements={} writable={} expected_dtype={dtype:?} expected_elements={elements}",
            tensor.dtype,
            tensor.n_elements(),
            tensor.is_writable(),
        )));
    }
    let alignment = match dtype {
        GgmlType::F16 => 2,
        GgmlType::F32 | GgmlType::I32 => 4,
        _ => 1,
    };
    let end = tensor
        .offset
        .checked_add(tensor.n_bytes())
        .ok_or_else(|| DenseBatch8Error::Validation(format!("{name} byte range overflow")))?;
    if !tensor.offset.is_multiple_of(alignment) || end > tensor.buffer.length() as u64 {
        return Err(DenseBatch8Error::Validation(format!(
            "{name} byte range is unaligned or exceeds its buffer"
        )));
    }
    Ok(())
}

impl DenseBatch8Scratch {
    fn new(ctx: &MetalContext, model: &MetalModel) -> Result<Self, DenseBatch8Error> {
        let arch = &model.arch;
        let hidden = arch.hidden_size as usize;
        let ffn = arch.intermediate_size as usize;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let gdn_head_dim = arch.gdn_head_dim as usize;
        let conv_dim = (2 * n_k + n_v)
            .checked_mul(gdn_head_dim)
            .ok_or_else(|| DenseBatch8Error::Validation("GDN width overflow".into()))?;
        let value_dim = n_v
            .checked_mul(gdn_head_dim)
            .ok_or_else(|| DenseBatch8Error::Validation("GDN value width overflow".into()))?;
        let q_dim = (arch.n_q_heads as usize)
            .checked_mul(arch.attn_head_dim as usize)
            .ok_or_else(|| DenseBatch8Error::Validation("attention Q width overflow".into()))?;
        let kv_dim = (arch.n_kv_heads as usize)
            .checked_mul(arch.attn_head_dim as usize)
            .ok_or_else(|| DenseBatch8Error::Validation("attention KV width overflow".into()))?;
        let vocab = arch.vocab_size as usize;
        dense_batch8_scratch_bytes(arch)?;
        Ok(Self {
            ids: MetalTensor::zeros_i32(ctx, vec![DENSE_BATCH8_WIDTH as u64])?,
            rows: MetalTensor::zeros_f32(ctx, vec![checked_elements(hidden, "hidden")?])?,
            logits: MetalTensor::zeros_f32(ctx, vec![checked_elements(vocab, "logits")?])?,
            argmax: MetalTensor::zeros_i32(ctx, vec![DENSE_BATCH8_WIDTH as u64])?,
            gdn: GdnScratch {
                h: MetalTensor::zeros_f32(ctx, vec![checked_elements(hidden, "GDN hidden")?])?,
                qkv: MetalTensor::zeros_f32(ctx, vec![checked_elements(conv_dim, "GDN QKV")?])?,
                z: MetalTensor::zeros_f32(ctx, vec![checked_elements(value_dim, "GDN Z")?])?,
                normed: MetalTensor::zeros_f32(
                    ctx,
                    vec![checked_elements(value_dim, "GDN normalized")?],
                )?,
                out: MetalTensor::zeros_f32(ctx, vec![checked_elements(hidden, "GDN output")?])?,
            },
            ffn: FfnScratch {
                h: MetalTensor::zeros_f32(ctx, vec![checked_elements(hidden, "FFN hidden")?])?,
                gate: MetalTensor::zeros_f32(ctx, vec![checked_elements(ffn, "FFN gate")?])?,
                up: MetalTensor::zeros_f32(ctx, vec![checked_elements(ffn, "FFN up")?])?,
                inner: MetalTensor::zeros_f32(ctx, vec![checked_elements(ffn, "FFN inner")?])?,
                out: MetalTensor::zeros_f32(ctx, vec![checked_elements(hidden, "FFN output")?])?,
            },
            attn: AttnScratch {
                h: MetalTensor::zeros_f32(
                    ctx,
                    vec![checked_elements(hidden, "attention hidden")?],
                )?,
                q: MetalTensor::zeros_f32(ctx, vec![checked_elements(2 * q_dim, "attention Q")?])?,
                k: MetalTensor::zeros_f32(ctx, vec![checked_elements(kv_dim, "attention K")?])?,
                v: MetalTensor::zeros_f32(ctx, vec![checked_elements(kv_dim, "attention V")?])?,
            },
        })
    }
}

pub struct DenseBatch8Executor<'a> {
    forward: MetalForward<'a>,
    scratch: DenseBatch8Scratch,
    poisoned: bool,
}

pub fn inspect_dense_batch8(model: &MetalModel) -> Result<(), DenseBatch8Error> {
    if model.arch.kind != ArchKind::Dense {
        return Err(DenseBatch8Error::Unsupported(
            "model architecture is not dense Qwen".into(),
        ));
    }
    if model.has_queue_scoped_residency_set() {
        return Err(DenseBatch8Error::Unsupported(
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
    ] {
        if read_default_off(flag) {
            return Err(DenseBatch8Error::Unsupported(format!(
                "diagnostic flag {flag} changes the production graph"
            )));
        }
    }
    dense_batch8_scratch_bytes(&model.arch)?;
    Ok(())
}

impl<'a> DenseBatch8Executor<'a> {
    pub fn new(ctx: &'a MetalContext, model: &'a MetalModel) -> Result<Self, DenseBatch8Error> {
        inspect_dense_batch8(model)?;
        Ok(Self {
            forward: MetalForward::new(ctx, model),
            scratch: DenseBatch8Scratch::new(ctx, model)?,
            poisoned: false,
        })
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn step(
        &mut self,
        token_ids: [i32; DENSE_BATCH8_WIDTH],
        position: u32,
        sessions: &mut [MetalSession],
    ) -> Result<DenseBatch8Step, DenseBatch8Error> {
        self.step_with_cancel(token_ids, position, sessions, || false)
    }

    pub fn step_with_cancel(
        &mut self,
        token_ids: [i32; DENSE_BATCH8_WIDTH],
        position: u32,
        sessions: &mut [MetalSession],
        cancelled: impl Fn() -> bool,
    ) -> Result<DenseBatch8Step, DenseBatch8Error> {
        if self.poisoned {
            return Err(DenseBatch8Error::Poisoned);
        }
        if cancelled() {
            return Err(DenseBatch8Error::CancelledBeforeCommit);
        }
        if sessions.len() != DENSE_BATCH8_WIDTH {
            return Err(DenseBatch8Error::Validation(format!(
                "session width {} != {DENSE_BATCH8_WIDTH}",
                sessions.len()
            )));
        }
        let sessions: &mut [MetalSession; DENSE_BATCH8_WIDTH] = sessions
            .try_into()
            .map_err(|_| DenseBatch8Error::Validation("session width changed".into()))?;
        let [s0, s1, s2, s3, s4, s5, s6, s7] = sessions;
        let mut refs = [s0, s1, s2, s3, s4, s5, s6, s7];
        self.step_refs_with_cancel(
            token_ids,
            [position; DENSE_BATCH8_WIDTH],
            &mut refs,
            Reduction::LowestIndex,
            cancelled,
        )
    }

    pub fn step_greedy_refs(
        &mut self,
        token_ids: [i32; DENSE_BATCH8_WIDTH],
        positions: [u32; DENSE_BATCH8_WIDTH],
        sessions: &mut [&mut MetalSession; DENSE_BATCH8_WIDTH],
        cancelled: impl Fn() -> bool,
    ) -> Result<DenseBatch8Step, DenseBatch8Error> {
        self.step_refs_with_cancel(
            token_ids,
            positions,
            sessions,
            Reduction::GreedyTotal,
            cancelled,
        )
    }

    pub fn validate_refs(
        &self,
        token_ids: [i32; DENSE_BATCH8_WIDTH],
        positions: [u32; DENSE_BATCH8_WIDTH],
        sessions: &mut [&mut MetalSession; DENSE_BATCH8_WIDTH],
    ) -> Result<(), DenseBatch8Error> {
        if self.poisoned {
            return Err(DenseBatch8Error::Poisoned);
        }
        self.validate_step(&token_ids, positions, sessions)
    }

    fn step_refs_with_cancel(
        &mut self,
        token_ids: [i32; DENSE_BATCH8_WIDTH],
        positions: [u32; DENSE_BATCH8_WIDTH],
        sessions: &mut [&mut MetalSession; DENSE_BATCH8_WIDTH],
        reduction: Reduction,
        cancelled: impl Fn() -> bool,
    ) -> Result<DenseBatch8Step, DenseBatch8Error> {
        if self.poisoned {
            return Err(DenseBatch8Error::Poisoned);
        }
        if cancelled() {
            return Err(DenseBatch8Error::CancelledBeforeCommit);
        }
        self.validate_step(&token_ids, positions, sessions)?;
        self.write_ids(&token_ids)?;
        let command = self
            .forward
            .ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| DenseBatch8Error::Validation("command buffer unavailable".into()))?;
        if let Err(error) = self.encode_step(&command, positions, sessions, reduction) {
            Self::restore_frontiers(sessions, positions);
            return Err(error);
        }
        if cancelled() {
            Self::restore_frontiers(sessions, positions);
            return Err(DenseBatch8Error::CancelledBeforeCommit);
        }
        command.commit();
        crate::metal::wait_unchecked(&command);
        let status = command.status();
        let error = command.error();
        if status != MTLCommandBufferStatus::Completed || error.is_some() {
            self.poisoned = true;
            Self::poison_sessions(sessions, "a committed dense B=8 command failed");
            return Err(DenseBatch8Error::CommandBuffer {
                status: format!("{status:?}"),
                error: format!("{error:?}"),
            });
        }
        let mut argmax_ids = match self.read_argmax() {
            Ok(ids) => ids,
            Err(error) => {
                self.poisoned = true;
                Self::poison_sessions(sessions, "dense B=8 argmax readback failed");
                return Err(error);
            }
        };
        if matches!(reduction, Reduction::GreedyTotal) {
            for (slot, raw) in argmax_ids.iter_mut().enumerate() {
                match GreedySelection::from_encoded(*raw).into_token() {
                    Ok(token) => *raw = token,
                    Err(source) => {
                        self.poisoned = true;
                        Self::poison_sessions(sessions, "dense B=8 greedy selection failed");
                        return Err(DenseBatch8Error::GreedySelection { slot, source });
                    }
                }
            }
        }
        let start = command.GPUStartTime();
        let end = command.GPUEndTime();
        let gpu_ms = (start.is_finite() && end.is_finite() && start > 0.0 && end > start)
            .then_some((end - start) * 1e3);
        Ok(DenseBatch8Step { argmax_ids, gpu_ms })
    }

    fn validate_step(
        &self,
        token_ids: &[i32; DENSE_BATCH8_WIDTH],
        positions: [u32; DENSE_BATCH8_WIDTH],
        sessions: &[&mut MetalSession; DENSE_BATCH8_WIDTH],
    ) -> Result<(), DenseBatch8Error> {
        if sessions.len() != DENSE_BATCH8_WIDTH {
            return Err(DenseBatch8Error::Validation(format!(
                "session width {} != {DENSE_BATCH8_WIDTH}",
                sessions.len()
            )));
        }
        let vocab = self.forward.model.arch.vocab_size;
        for (slot, token) in token_ids.iter().copied().enumerate() {
            if token < 0 || token as u32 >= vocab {
                return Err(DenseBatch8Error::Validation(format!(
                    "slot {slot} token {token} is outside vocab {vocab}"
                )));
            }
        }
        let expected_gdn = self
            .forward
            .model
            .blocks
            .iter()
            .filter(|block| matches!(block, MetalBlock::Gdn(_)))
            .count();
        let expected_attn = self.forward.model.blocks.len() - expected_gdn;
        let arch = &self.forward.model.arch;
        let hidden = arch.hidden_size as usize;
        let ffn = arch.intermediate_size as usize;
        let gdn_head_dim = arch.gdn_head_dim as usize;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let conv_dim = (2 * n_k + n_v) * gdn_head_dim;
        let value_dim = n_v * gdn_head_dim;
        let key_dim = n_k * gdn_head_dim;
        let conv_history = (arch.gdn_conv_kernel as usize)
            .checked_sub(1)
            .ok_or_else(|| DenseBatch8Error::Validation("GDN kernel must be nonzero".into()))?;
        let conv_state = conv_history
            .checked_mul(conv_dim)
            .ok_or_else(|| DenseBatch8Error::Validation("GDN conv state overflow".into()))?;
        let recurrent_state = value_dim * gdn_head_dim;
        let q_dim = arch.n_q_heads as usize * arch.attn_head_dim as usize;
        let kv_dim = arch.n_kv_heads as usize * arch.attn_head_dim as usize;
        let q_group = (arch.n_q_heads as usize)
            .checked_div(arch.n_kv_heads as usize)
            .filter(|group| *group > 0)
            .ok_or_else(|| DenseBatch8Error::Validation("invalid Q/KV head ratio".into()))?;
        let partial_o = (arch.n_kv_heads as usize)
            .checked_mul(ATTN_V4_MAX_NWG)
            .and_then(|value| value.checked_mul(q_group))
            .and_then(|value| value.checked_mul(arch.attn_head_dim as usize))
            .ok_or_else(|| DenseBatch8Error::Validation("attention O partial overflow".into()))?;
        let partial_ml = (arch.n_kv_heads as usize)
            .checked_mul(ATTN_V4_MAX_NWG)
            .and_then(|value| value.checked_mul(q_group))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| DenseBatch8Error::Validation("attention ML partial overflow".into()))?;
        let vocab = arch.vocab_size as usize;
        for (slot, session) in sessions.iter().enumerate() {
            let position = positions[slot] as usize;
            let session = &**session;
            session.ensure_usable()?;
            if session.has_internal_mutable_alias() {
                return Err(DenseBatch8Error::Validation(format!(
                    "slot {slot} aliases mutable storage internally"
                )));
            }
            if position >= session.kv_capacity {
                return Err(DenseBatch8Error::Validation(format!(
                    "slot {slot} position {position} exceeds capacity {}",
                    session.kv_capacity
                )));
            }
            for (layer, actual) in session.kv_n_pos.iter().copied().enumerate() {
                if actual != position {
                    return Err(DenseBatch8Error::Validation(format!(
                        "slot {slot} attention layer {layer} frontier {actual} != {position}"
                    )));
                }
            }
            if session.gdn_state.len() != expected_gdn
                || session.gdn_conv.len() != expected_gdn
                || session.kv_k.len() != expected_attn
                || session.kv_v.len() != expected_attn
                || session.kv_n_pos.len() != expected_attn
            {
                return Err(DenseBatch8Error::Validation(format!(
                    "slot {slot} state inventory does not match model"
                )));
            }
            if !session
                .kv_k
                .iter()
                .chain(&session.kv_v)
                .all(|tensor| tensor.dtype == GgmlType::F16)
            {
                return Err(DenseBatch8Error::Unsupported(
                    "only F16 KV is qualified for dense B=8".into(),
                ));
            }
            for (layer, tensor) in session.gdn_conv.iter().enumerate() {
                require_writable_tensor(
                    tensor,
                    &format!("slot {slot} gdn_conv[{layer}]"),
                    GgmlType::F32,
                    conv_state,
                )?;
            }
            for (layer, tensor) in session.gdn_state.iter().enumerate() {
                require_writable_tensor(
                    tensor,
                    &format!("slot {slot} gdn_state[{layer}]"),
                    GgmlType::F32,
                    recurrent_state,
                )?;
            }
            let kv_elements = session.kv_capacity.checked_mul(kv_dim).ok_or_else(|| {
                DenseBatch8Error::Validation(format!("slot {slot} KV size overflow"))
            })?;
            for (layer, tensor) in session.kv_k.iter().chain(&session.kv_v).enumerate() {
                require_writable_tensor(
                    tensor,
                    &format!("slot {slot} KV[{layer}]"),
                    GgmlType::F16,
                    kv_elements,
                )?;
            }
            for (name, tensor, elements) in [
                ("x", &session.x, hidden),
                ("h", &session.h, hidden),
                ("ffn_gate", &session.ffn_gate, ffn),
                ("ffn_up", &session.ffn_up, ffn),
                ("ffn_inner", &session.ffn_inner, ffn),
                ("ffn_out", &session.ffn_out, hidden),
                ("gdn_qkv", &session.gdn_qkv, conv_dim),
                ("gdn_qkv_conv", &session.gdn_qkv_conv, conv_dim),
                ("gdn_z", &session.gdn_z, value_dim),
                ("gdn_b", &session.gdn_b, n_v),
                ("gdn_beta", &session.gdn_beta, n_v),
                ("gdn_a", &session.gdn_a, n_v),
                ("gdn_alpha", &session.gdn_alpha, n_v),
                ("gdn_q_norm", &session.gdn_q_norm, key_dim),
                ("gdn_k_norm", &session.gdn_k_norm, key_dim),
                ("gdn_out", &session.gdn_out, value_dim),
                ("gdn_normed", &session.gdn_normed, value_dim),
                ("mixer_out", &session.mixer_out, hidden),
                ("attn_q_full", &session.attn_q_full, 2 * q_dim),
                ("attn_q", &session.attn_q, q_dim),
                ("attn_gate", &session.attn_gate, q_dim),
                ("attn_q_normed", &session.attn_q_normed, q_dim),
                ("attn_k_now", &session.attn_k_now, kv_dim),
                ("attn_v_now", &session.attn_v_now, kv_dim),
                ("attn_k_normed", &session.attn_k_normed, kv_dim),
                ("attn_o", &session.attn_o, q_dim),
                ("logits", &session.logits, vocab),
            ] {
                require_writable_tensor(
                    tensor,
                    &format!("slot {slot} {name}"),
                    GgmlType::F32,
                    elements,
                )?;
            }
            require_writable_tensor(
                &session.argmax_tok,
                &format!("slot {slot} argmax_tok"),
                GgmlType::I32,
                1,
            )?;
            require_writable_tensor(
                &session.attn_v4_o_partial,
                &format!("slot {slot} attn_v4_o_partial"),
                GgmlType::F32,
                partial_o,
            )?;
            require_writable_tensor(
                &session.attn_v4_ml_partial,
                &format!("slot {slot} attn_v4_ml_partial"),
                GgmlType::F32,
                partial_ml,
            )?;
        }
        for left in 0..DENSE_BATCH8_WIDTH {
            for right in left + 1..DENSE_BATCH8_WIDTH {
                if sessions[left].aliases_mutable_session(sessions[right]) {
                    return Err(DenseBatch8Error::Validation(format!(
                        "slots {left} and {right} alias mutable session storage"
                    )));
                }
            }
        }
        Ok(())
    }

    fn restore_frontiers(
        sessions: &mut [&mut MetalSession; DENSE_BATCH8_WIDTH],
        positions: [u32; DENSE_BATCH8_WIDTH],
    ) {
        for (slot, session) in sessions.iter_mut().enumerate() {
            session.kv_n_pos.fill(positions[slot] as usize);
        }
    }

    fn poison_sessions(
        sessions: &mut [&mut MetalSession; DENSE_BATCH8_WIDTH],
        reason: &'static str,
    ) {
        for session in sessions {
            session.poison(reason);
        }
    }

    fn write_ids(&self, token_ids: &[i32; DENSE_BATCH8_WIDTH]) -> Result<(), DenseBatch8Error> {
        if self.scratch.ids.dtype != GgmlType::I32
            || !self.scratch.ids.offset.is_multiple_of(4)
            || self.scratch.ids.n_elements() as usize != DENSE_BATCH8_WIDTH
        {
            return Err(DenseBatch8Error::Validation(
                "input token scratch contract mismatch".into(),
            ));
        }
        let ptr = self.scratch.ids.buffer.contents().as_ptr().cast::<i32>();
        if ptr.is_null() {
            return Err(DenseBatch8Error::Validation(
                "input token scratch is not CPU visible".into(),
            ));
        }
        let offset = usize::try_from(self.scratch.ids.offset / 4)
            .map_err(|_| DenseBatch8Error::Validation("input offset exceeds usize".into()))?;
        unsafe {
            std::ptr::copy_nonoverlapping(token_ids.as_ptr(), ptr.add(offset), DENSE_BATCH8_WIDTH);
        }
        Ok(())
    }

    fn read_argmax(&self) -> Result<[i32; DENSE_BATCH8_WIDTH], DenseBatch8Error> {
        let ptr = self.scratch.argmax.buffer.contents().as_ptr().cast::<i32>();
        if ptr.is_null() || !self.scratch.argmax.offset.is_multiple_of(4) {
            return Err(DenseBatch8Error::Validation(
                "argmax scratch is not aligned CPU-visible I32".into(),
            ));
        }
        let offset = usize::try_from(self.scratch.argmax.offset / 4)
            .map_err(|_| DenseBatch8Error::Validation("argmax offset exceeds usize".into()))?;
        let mut output = [0i32; DENSE_BATCH8_WIDTH];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.add(offset), output.as_mut_ptr(), DENSE_BATCH8_WIDTH);
        }
        Ok(output)
    }

    fn encode_step(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        positions: [u32; DENSE_BATCH8_WIDTH],
        sessions: &mut [&mut MetalSession; DENSE_BATCH8_WIDTH],
        reduction: Reduction,
    ) -> Result<(), DenseBatch8Error> {
        let arch = &self.forward.model.arch;
        let hidden = arch.hidden_size as usize;
        let ffn = arch.intermediate_size as usize;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let gdn_head_dim = arch.gdn_head_dim as usize;
        let conv_dim = (2 * n_k + n_v) * gdn_head_dim;
        let value_dim = n_v * gdn_head_dim;
        let q_dim = arch.n_q_heads as usize * arch.attn_head_dim as usize;
        let kv_dim = arch.n_kv_heads as usize * arch.attn_head_dim as usize;
        let vocab = arch.vocab_size as usize;

        let encoder = KernelEncoder::begin(command);
        encode_get_rows_f32(
            self.forward.ctx,
            &encoder,
            &self.forward.model.token_embd,
            &self.scratch.ids,
            &self.scratch.rows,
            DENSE_BATCH8_WIDTH,
            hidden,
        )?;
        encoder.end();
        self.blit_rows_to_sessions(command, &self.scratch.rows, sessions, |session| &session.x)?;

        let mut gdn_index = 0usize;
        let mut attn_index = 0usize;
        for block in &self.forward.model.blocks {
            match block {
                MetalBlock::Gdn(block) => {
                    self.encode_gdn_block(
                        command, block, gdn_index, sessions, hidden, conv_dim, value_dim, ffn,
                    )?;
                    gdn_index += 1;
                }
                MetalBlock::Attn(block) => {
                    self.encode_attn_block(
                        command, block, attn_index, positions, sessions, hidden, q_dim, kv_dim, ffn,
                    )?;
                    attn_index += 1;
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
        self.blit_rows_from_sessions(command, sessions, &self.scratch.rows, |session| &session.h)?;

        let encoder = KernelEncoder::begin(command);
        encode_mat_mat_dispatch(
            self.forward.ctx,
            &encoder,
            &self.forward.model.lm_head,
            &self.scratch.rows,
            &self.scratch.logits,
            hidden,
            vocab,
            DENSE_BATCH8_WIDTH,
        )?;
        match reduction {
            Reduction::LowestIndex => encode_argmax_f32(
                self.forward.ctx,
                &encoder,
                &self.scratch.logits,
                &self.scratch.argmax,
                DENSE_BATCH8_WIDTH,
                vocab,
            )?,
            Reduction::GreedyTotal => encode_argmax_f32_greedy(
                self.forward.ctx,
                &encoder,
                &self.scratch.logits,
                &self.scratch.argmax,
                DENSE_BATCH8_WIDTH,
                vocab,
            )?,
        }
        encoder.end();
        self.blit_logits_to_sessions(command, sessions, vocab)?;
        Ok(())
    }

    fn blit_rows_to_sessions(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        source: &MetalTensor,
        sessions: &[&mut MetalSession; DENSE_BATCH8_WIDTH],
        destination: impl Fn(&MetalSession) -> &MetalTensor,
    ) -> Result<(), DenseBatch8Error> {
        let width = self.forward.model.arch.hidden_size as usize;
        let row_bytes = u64::try_from(width * std::mem::size_of::<f32>())
            .map_err(|_| DenseBatch8Error::Validation("hidden row bytes overflow".into()))?;
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
        sessions: &[&mut MetalSession; DENSE_BATCH8_WIDTH],
        destination: &MetalTensor,
        source: impl Fn(&MetalSession) -> &MetalTensor,
    ) -> Result<(), DenseBatch8Error> {
        let width = self.forward.model.arch.hidden_size as usize;
        let row_bytes = u64::try_from(width * std::mem::size_of::<f32>())
            .map_err(|_| DenseBatch8Error::Validation("hidden row bytes overflow".into()))?;
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

    fn encode_gdn_block(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        block: &MetalGdnBlock,
        gdn_index: usize,
        sessions: &mut [&mut MetalSession; DENSE_BATCH8_WIDTH],
        hidden: usize,
        conv_dim: usize,
        value_dim: usize,
        ffn: usize,
    ) -> Result<(), DenseBatch8Error> {
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
        encode_mat_mat_dispatch(
            self.forward.ctx,
            &encoder,
            &block.in_proj_qkv,
            &self.scratch.gdn.h,
            &self.scratch.gdn.qkv,
            hidden,
            conv_dim,
            DENSE_BATCH8_WIDTH,
        )?;
        encode_mat_mat_dispatch(
            self.forward.ctx,
            &encoder,
            &block.in_proj_z,
            &self.scratch.gdn.h,
            &self.scratch.gdn.z,
            hidden,
            value_dim,
            DENSE_BATCH8_WIDTH,
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
        encode_mat_mat_dispatch(
            self.forward.ctx,
            &encoder,
            &block.out_proj,
            &self.scratch.gdn.normed,
            &self.scratch.gdn.out,
            value_dim,
            hidden,
            DENSE_BATCH8_WIDTH,
        )?;
        for (slot, session) in sessions.iter().enumerate() {
            let out = self
                .scratch
                .gdn
                .out
                .view_subrange((slot * hidden) as u64, vec![hidden as u64]);
            self.forward.encode_post_mixer_norm(
                &encoder,
                &session.x,
                &out,
                &block.post_attn_norm,
                &session.h,
            )?;
        }
        encoder.end();
        self.encode_ffn(
            command,
            &block.ffn_gate,
            &block.ffn_up,
            &block.ffn_down,
            sessions,
            hidden,
            ffn,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_attn_block(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        block: &MetalAttnBlock,
        attn_index: usize,
        positions: [u32; DENSE_BATCH8_WIDTH],
        sessions: &mut [&mut MetalSession; DENSE_BATCH8_WIDTH],
        hidden: usize,
        q_dim: usize,
        kv_dim: usize,
        ffn: usize,
    ) -> Result<(), DenseBatch8Error> {
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
        self.blit_rows_from_sessions(command, sessions, &self.scratch.attn.h, |session| {
            &session.h
        })?;

        let encoder = KernelEncoder::begin(command);
        encode_mat_mat_dispatch(
            self.forward.ctx,
            &encoder,
            &block.q,
            &self.scratch.attn.h,
            &self.scratch.attn.q,
            hidden,
            2 * q_dim,
            DENSE_BATCH8_WIDTH,
        )?;
        encode_mat_mat_dispatch(
            self.forward.ctx,
            &encoder,
            &block.k,
            &self.scratch.attn.h,
            &self.scratch.attn.k,
            hidden,
            kv_dim,
            DENSE_BATCH8_WIDTH,
        )?;
        encode_mat_mat_dispatch(
            self.forward.ctx,
            &encoder,
            &block.v,
            &self.scratch.attn.h,
            &self.scratch.attn.v,
            hidden,
            kv_dim,
            DENSE_BATCH8_WIDTH,
        )?;
        encoder.end();

        let q_row_bytes = u64::try_from(2 * q_dim * std::mem::size_of::<f32>())
            .map_err(|_| DenseBatch8Error::Validation("attention Q row bytes overflow".into()))?;
        let kv_row_bytes = u64::try_from(kv_dim * std::mem::size_of::<f32>())
            .map_err(|_| DenseBatch8Error::Validation("attention KV row bytes overflow".into()))?;
        let blit = BlitEncoder::begin(command);
        for (slot, session) in sessions.iter().enumerate() {
            blit.copy_buffer(
                &self.scratch.attn.q.buffer,
                self.scratch.attn.q.offset + slot as u64 * q_row_bytes,
                &session.attn_q_full.buffer,
                session.attn_q_full.offset,
                q_row_bytes,
            );
            blit.copy_buffer(
                &self.scratch.attn.k.buffer,
                self.scratch.attn.k.offset + slot as u64 * kv_row_bytes,
                &session.attn_k_now.buffer,
                session.attn_k_now.offset,
                kv_row_bytes,
            );
            blit.copy_buffer(
                &self.scratch.attn.v.buffer,
                self.scratch.attn.v.offset + slot as u64 * kv_row_bytes,
                &session.attn_v_now.buffer,
                session.attn_v_now.offset,
                kv_row_bytes,
            );
        }
        blit.end();

        let encoder = KernelEncoder::begin(command);
        for (slot, session) in sessions.iter_mut().enumerate() {
            self.forward.encode_attn_after_projections(
                &encoder,
                block,
                attn_index,
                positions[slot],
                session,
            )?;
            self.forward.encode_post_mixer_norm(
                &encoder,
                &session.x,
                &session.mixer_out,
                &block.post_attn_norm,
                &session.h,
            )?;
        }
        encoder.end();
        self.encode_ffn(
            command,
            &block.ffn_gate,
            &block.ffn_up,
            &block.ffn_down,
            sessions,
            hidden,
            ffn,
        )
    }

    fn encode_ffn(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        gate_weight: &MetalTensor,
        up_weight: &MetalTensor,
        down_weight: &MetalTensor,
        sessions: &[&mut MetalSession; DENSE_BATCH8_WIDTH],
        hidden: usize,
        ffn: usize,
    ) -> Result<(), DenseBatch8Error> {
        self.blit_rows_from_sessions(command, sessions, &self.scratch.ffn.h, |session| &session.h)?;
        let encoder = KernelEncoder::begin(command);
        encode_mat_mat_dispatch(
            self.forward.ctx,
            &encoder,
            gate_weight,
            &self.scratch.ffn.h,
            &self.scratch.ffn.gate,
            hidden,
            ffn,
            DENSE_BATCH8_WIDTH,
        )?;
        encode_mat_mat_dispatch(
            self.forward.ctx,
            &encoder,
            up_weight,
            &self.scratch.ffn.h,
            &self.scratch.ffn.up,
            hidden,
            ffn,
            DENSE_BATCH8_WIDTH,
        )?;
        encode_silu_mul_f32(
            self.forward.ctx,
            &encoder,
            &self.scratch.ffn.gate,
            &self.scratch.ffn.up,
            &self.scratch.ffn.inner,
        )?;
        encode_mat_mat_dispatch(
            self.forward.ctx,
            &encoder,
            down_weight,
            &self.scratch.ffn.inner,
            &self.scratch.ffn.out,
            ffn,
            hidden,
            DENSE_BATCH8_WIDTH,
        )?;
        for (slot, session) in sessions.iter().enumerate() {
            let out = self
                .scratch
                .ffn
                .out
                .view_subrange((slot * hidden) as u64, vec![hidden as u64]);
            encode_add_inplace_f32(self.forward.ctx, &encoder, &session.x, &out)?;
        }
        encoder.end();
        Ok(())
    }

    fn blit_logits_to_sessions(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        sessions: &[&mut MetalSession; DENSE_BATCH8_WIDTH],
        vocab: usize,
    ) -> Result<(), DenseBatch8Error> {
        let row_bytes = u64::try_from(vocab * std::mem::size_of::<f32>())
            .map_err(|_| DenseBatch8Error::Validation("logit row bytes overflow".into()))?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::QWEN3_0_8B;

    #[test]
    fn scratch_accounting_is_stable_for_dense_0_8b() {
        assert_eq!(dense_batch8_scratch_bytes(&QWEN3_0_8B).unwrap(), 8_978_496);
    }
}
