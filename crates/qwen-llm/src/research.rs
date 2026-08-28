//! Safe, narrow instrumentation surface for workspace-lens research.
//!
//! This module deliberately returns owned CPU data and opaque linear IDs. It
//! does not expose resident model buffers, command encoders, or mutable Metal
//! session state across the public API boundary.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, RmsNormVjpRule, SwiGluVjpRule,
    encode_add_f32, encode_frozen_linear_vjp_f32, encode_rms_norm_mul_f32,
    encode_rms_norm_mul_vjp_broadcast_f32, encode_silu_mul_vjp_broadcast_f32,
};
use crate::metal_forward::{MetalBlock, MfError, RMS_EPS, encode_mat_vec_dispatch};
use crate::model::{Arch, ArchKind};
use crate::runtime::{LoadedModel, RuntimeError, Sequence};
use crate::tensor::GgmlType;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};

/// Lightweight locator identity derived from model metadata, shard paths, and
/// file stamps. It is useful within one machine, but is not a content digest.
pub const RESEARCH_IDENTITY_SCHEME: &str = "qwen_llm_model_locator_v1";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ResearchModelIdentity {
    pub model_locator_id: u64,
    pub tokenizer_metadata_id: u64,
    pub content_authenticated: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResearchLinear {
    LmHead,
    Layer { index: u32, role: LinearRole },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LinearRole {
    FfnGate,
    FfnUp,
    FfnDown,
    GdnQkv,
    GdnZ,
    GdnBeta,
    GdnAlpha,
    GdnOut,
    AttentionQAndGate,
    AttentionK,
    AttentionV,
    AttentionOut,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResearchLinearInfo {
    pub id: ResearchLinear,
    pub dtype: GgmlType,
    /// GGUF axis order: `[n_in, n_out]`.
    pub shape: [usize; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActivationCapture {
    pub layer_ids: Vec<u32>,
    pub hidden_size: usize,
    /// Caller-layer order, flattened as `[K, H]`.
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResearchForward {
    pub position: usize,
    pub token_id: i32,
    pub logits: Vec<f32>,
    pub capture: ActivationCapture,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseFfnActivationCapture {
    pub layer_ids: Vec<u32>,
    pub hidden_size: usize,
    /// Post-mixer residuals immediately before post-attention RMSNorm,
    /// flattened in caller layer order as `[K, H]`.
    pub pre_ffn_residuals: Vec<f32>,
    /// Post-FFN, post-residual block outputs, flattened as `[K, H]`.
    pub post_block_residuals: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResearchDenseFfnForward {
    pub position: usize,
    pub token_id: i32,
    pub logits: Vec<f32>,
    pub capture: DenseFfnActivationCapture,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseFfnVjpRule {
    /// Ordinary activation Jacobian used by J-lens.
    Jacobian,
    /// R-lens/RelP rules: detached RMS denominator and SwiGLU identity/half.
    Relp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseFfnVjp {
    pub layer: u32,
    pub n_query: usize,
    pub hidden_size: usize,
    /// Query-major cotangents of the pre-FFN residual, flattened as `[Q, H]`.
    pub values: Vec<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum ResearchError {
    #[error("runtime: {0}")]
    Runtime(#[from] RuntimeError),
    #[error("forward: {0}")]
    Forward(#[from] MfError),
    #[error("metal: {0}")]
    Metal(#[from] MetalError),
    #[error("workspace-lens research currently requires a dense model, got {0:?}")]
    UnsupportedArchitecture(ArchKind),
    #[error("layer {layer} is out of range for {n_layers} layers")]
    InvalidLayer { layer: u32, n_layers: u32 },
    #[error("linear role {role:?} is not present on layer {layer}")]
    InvalidLinearRole { layer: u32, role: LinearRole },
    #[error("linear {id:?} has shape {shape:?}, expected a two-dimensional row bank")]
    InvalidLinearShape { id: ResearchLinear, shape: Vec<u64> },
    #[error(
        "linear {id:?} uses {dtype:?}; the native activation VJP supports Q8_0, BF16, F16, and F32"
    )]
    UnsupportedLinearDtype { id: ResearchLinear, dtype: GgmlType },
    #[error("layer {layer} {role:?} has shape {got:?}, expected {expected:?}")]
    InvalidDenseFfnShape {
        layer: u32,
        role: LinearRole,
        got: [usize; 2],
        expected: [usize; 2],
    },
    #[error("layer {layer} post-attention norm must be F32 [{expected}], got {dtype:?} {shape:?}")]
    InvalidDenseFfnNorm {
        layer: u32,
        dtype: GgmlType,
        shape: Vec<u64>,
        expected: usize,
    },
    #[error("{name} length {got} does not match expected length {expected}")]
    ActivationSize {
        name: &'static str,
        got: usize,
        expected: usize,
    },
    #[error("n_query must be nonzero")]
    EmptyQueryBatch,
    #[error("cotangent length {got} does not match n_query={n_query} x n_out={n_out} ({expected})")]
    CotangentSize {
        got: usize,
        expected: usize,
        n_query: usize,
        n_out: usize,
    },
    #[error("research tensor size overflow")]
    SizeOverflow,
    #[error("sequence position {0} exceeds the ordinary-Qwen u32 position contract")]
    PositionOverflow(usize),
    #[error("Metal did not provide a command buffer")]
    MissingCommandBuffer,
    #[error("Metal command buffer failed: status={status} error={error}")]
    CommandBuffer { status: String, error: String },
}

pub struct ResearchSession<'model, 'sequence> {
    model: &'model LoadedModel,
    sequence: &'sequence mut Sequence,
}

impl LoadedModel {
    pub fn research_identity(&self) -> ResearchModelIdentity {
        let (model_locator_id, tokenizer_metadata_id) = self.lightweight_identity_parts();
        ResearchModelIdentity {
            model_locator_id,
            tokenizer_metadata_id,
            content_authenticated: false,
        }
    }

    pub fn research_session<'model, 'sequence>(
        &'model self,
        sequence: &'sequence mut Sequence,
    ) -> Result<ResearchSession<'model, 'sequence>, ResearchError> {
        self.ensure_owns(sequence)?;
        if self.arch().kind != ArchKind::Dense {
            return Err(ResearchError::UnsupportedArchitecture(self.arch().kind));
        }
        Ok(ResearchSession {
            model: self,
            sequence,
        })
    }
}

impl ResearchSession<'_, '_> {
    pub fn arch(&self) -> Arch {
        self.model.arch()
    }

    pub fn identity(&self) -> ResearchModelIdentity {
        self.model.research_identity()
    }

    pub fn linear_info(&self, id: ResearchLinear) -> Result<ResearchLinearInfo, ResearchError> {
        let tensor = self.resolve_linear(id)?;
        let shape = linear_shape(id, tensor)?;
        Ok(ResearchLinearInfo {
            id,
            dtype: tensor.dtype,
            shape,
        })
    }

    pub fn linears(&self) -> Result<Vec<ResearchLinearInfo>, ResearchError> {
        let mut ids = vec![ResearchLinear::LmHead];
        for (index, block) in self.model.metal_model().blocks.iter().enumerate() {
            let index = u32::try_from(index).map_err(|_| ResearchError::SizeOverflow)?;
            for role in [LinearRole::FfnGate, LinearRole::FfnUp, LinearRole::FfnDown] {
                ids.push(ResearchLinear::Layer { index, role });
            }
            match block {
                MetalBlock::Gdn(_) => {
                    for role in [
                        LinearRole::GdnQkv,
                        LinearRole::GdnZ,
                        LinearRole::GdnBeta,
                        LinearRole::GdnAlpha,
                        LinearRole::GdnOut,
                    ] {
                        ids.push(ResearchLinear::Layer { index, role });
                    }
                }
                MetalBlock::Attn(_) => {
                    for role in [
                        LinearRole::AttentionQAndGate,
                        LinearRole::AttentionK,
                        LinearRole::AttentionV,
                        LinearRole::AttentionOut,
                    ] {
                        ids.push(ResearchLinear::Layer { index, role });
                    }
                }
            }
        }
        ids.into_iter().map(|id| self.linear_info(id)).collect()
    }

    /// Advance one token and return post-block residuals in caller layer order.
    pub fn forward_token(
        &mut self,
        token_id: i32,
        capture_layers: &[u32],
    ) -> Result<ResearchForward, ResearchError> {
        self.sequence.ensure_can_append(1)?;
        validate_capture_layers(self.arch().n_layer, capture_layers)?;
        let position = self.sequence.position();
        let position_u32 =
            u32::try_from(position).map_err(|_| ResearchError::PositionOverflow(position))?;
        let hidden_size = self.arch().hidden_size as usize;
        let capture_len = capture_layers
            .len()
            .checked_mul(hidden_size)
            .ok_or(ResearchError::SizeOverflow)?;

        let forward = self.model.forward();
        let state = unsafe { self.sequence.metal_session_mut() };
        state.ensure_usable()?;
        let (logits, values) = if capture_layers.is_empty() {
            (
                forward.single_token(token_id, position_u32, state)?,
                Vec::new(),
            )
        } else {
            let capture = MetalTensor::zeros_f32(self.model.context(), vec![capture_len as u64])?;
            let logits = forward.single_token_with_multi_hidden(
                token_id,
                position_u32,
                state,
                capture_layers,
                &capture,
            )?;
            (logits, read_f32(&capture, capture_len))
        };
        self.sequence.advance_by(1)?;

        Ok(ResearchForward {
            position,
            token_id,
            logits,
            capture: ActivationCapture {
                layer_ids: capture_layers.to_vec(),
                hidden_size,
                values,
            },
        })
    }

    /// Advance one token while capturing both sides of selected dense FFNs.
    pub fn forward_token_with_dense_ffn_capture(
        &mut self,
        token_id: i32,
        capture_layers: &[u32],
    ) -> Result<ResearchDenseFfnForward, ResearchError> {
        self.sequence.ensure_can_append(1)?;
        validate_capture_layers(self.arch().n_layer, capture_layers)?;
        let position = self.sequence.position();
        let position_u32 =
            u32::try_from(position).map_err(|_| ResearchError::PositionOverflow(position))?;
        let hidden_size = self.arch().hidden_size as usize;
        let capture_len = capture_layers
            .len()
            .checked_mul(hidden_size)
            .ok_or(ResearchError::SizeOverflow)?;

        let forward = self.model.forward();
        let state = unsafe { self.sequence.metal_session_mut() };
        state.ensure_usable()?;
        let (logits, pre_ffn_residuals, post_block_residuals) = if capture_layers.is_empty() {
            (
                forward.single_token(token_id, position_u32, state)?,
                Vec::new(),
                Vec::new(),
            )
        } else {
            let pre_ffn = MetalTensor::zeros_f32(
                self.model.context(),
                vec![hidden_size as u64, capture_layers.len() as u64],
            )?;
            let post_block = MetalTensor::zeros_f32(
                self.model.context(),
                vec![hidden_size as u64, capture_layers.len() as u64],
            )?;
            let logits = forward.single_token_with_dense_ffn_capture(
                token_id,
                position_u32,
                state,
                capture_layers,
                &pre_ffn,
                &post_block,
            )?;
            (
                logits,
                read_f32(&pre_ffn, capture_len),
                read_f32(&post_block, capture_len),
            )
        };
        self.sequence.advance_by(1)?;

        Ok(ResearchDenseFfnForward {
            position,
            token_id,
            logits,
            capture: DenseFfnActivationCapture {
                layer_ids: capture_layers.to_vec(),
                hidden_size,
                pre_ffn_residuals,
                post_block_residuals,
            },
        })
    }

    /// Apply the exact activation VJP of a supported frozen resident linear map.
    pub fn frozen_linear_vjp(
        &mut self,
        id: ResearchLinear,
        grad_output: &[f32],
        n_query: usize,
    ) -> Result<Vec<f32>, ResearchError> {
        if n_query == 0 {
            return Err(ResearchError::EmptyQueryBatch);
        }
        let weight = self.resolve_linear(id)?;
        let [n_in, n_out] = linear_shape(id, weight)?;
        if !matches!(
            weight.dtype,
            GgmlType::Q8_0 | GgmlType::BF16 | GgmlType::F16 | GgmlType::F32
        ) {
            return Err(ResearchError::UnsupportedLinearDtype {
                id,
                dtype: weight.dtype,
            });
        }
        let expected = n_query
            .checked_mul(n_out)
            .ok_or(ResearchError::SizeOverflow)?;
        if grad_output.len() != expected {
            return Err(ResearchError::CotangentSize {
                got: grad_output.len(),
                expected,
                n_query,
                n_out,
            });
        }
        let grad_output = MetalTensor::from_bytes(
            self.model.context(),
            bytemuck::cast_slice(grad_output),
            vec![n_out as u64, n_query as u64],
            GgmlType::F32,
        )?;
        let grad_input =
            MetalTensor::zeros_f32(self.model.context(), vec![n_in as u64, n_query as u64])?;
        let command = self
            .model
            .context()
            .queue
            .commandBuffer()
            .ok_or(ResearchError::MissingCommandBuffer)?;
        let encoder = KernelEncoder::begin(&command);
        let encode_result = encode_frozen_linear_vjp_f32(
            self.model.context(),
            &encoder,
            weight,
            &grad_output,
            &grad_input,
            n_in,
            n_out,
            n_query,
        );
        encoder.end();
        encode_result?;
        command.commit();
        command.waitUntilCompleted();
        let status = command.status();
        let error = command.error();
        if status != MTLCommandBufferStatus::Completed || error.is_some() {
            return Err(ResearchError::CommandBuffer {
                status: format!("{status:?}"),
                error: format!("{error:?}"),
            });
        }
        let output_len = n_query
            .checked_mul(n_in)
            .ok_or(ResearchError::SizeOverflow)?;
        Ok(read_f32(&grad_input, output_len))
    }

    /// Reverse one dense FFN residual update from post-block cotangents to the
    /// post-mixer, pre-FFN residual captured during the matching forward.
    ///
    /// The method recomputes RMSNorm, gate, and up primals from `pre_ffn_residual`
    /// and keeps all resident weights opaque. `grad_output` and the result are
    /// query-major `[n_query, H]`. The unchanged residual branch is included.
    /// All three FFN linears must be resident as Q8_0, BF16, F16, or F32.
    pub fn dense_ffn_vjp(
        &self,
        layer: u32,
        pre_ffn_residual: &[f32],
        grad_output: &[f32],
        n_query: usize,
        rule: DenseFfnVjpRule,
    ) -> Result<DenseFfnVjp, ResearchError> {
        let (post_norm, gate, up, down) = self.resolve_dense_ffn(layer)?;
        let hidden_size = self.arch().hidden_size as usize;
        let intermediate_size = self.arch().intermediate_size as usize;
        let values = dense_ffn_vjp_readback(
            self.model.context(),
            layer,
            hidden_size,
            intermediate_size,
            pre_ffn_residual,
            post_norm,
            gate,
            up,
            down,
            grad_output,
            n_query,
            rule,
        )?;
        Ok(DenseFfnVjp {
            layer,
            n_query,
            hidden_size,
            values,
        })
    }

    fn resolve_dense_ffn(
        &self,
        layer: u32,
    ) -> Result<(&MetalTensor, &MetalTensor, &MetalTensor, &MetalTensor), ResearchError> {
        let block = self.model.metal_model().blocks.get(layer as usize).ok_or(
            ResearchError::InvalidLayer {
                layer,
                n_layers: self.arch().n_layer,
            },
        )?;
        Ok(match block {
            MetalBlock::Gdn(block) => (
                &block.post_attn_norm,
                &block.ffn_gate,
                &block.ffn_up,
                &block.ffn_down,
            ),
            MetalBlock::Attn(block) => (
                &block.post_attn_norm,
                &block.ffn_gate,
                &block.ffn_up,
                &block.ffn_down,
            ),
        })
    }

    fn resolve_linear(&self, id: ResearchLinear) -> Result<&MetalTensor, ResearchError> {
        let ResearchLinear::Layer { index, role } = id else {
            return Ok(&self.model.metal_model().lm_head);
        };
        let block = self.model.metal_model().blocks.get(index as usize).ok_or(
            ResearchError::InvalidLayer {
                layer: index,
                n_layers: self.arch().n_layer,
            },
        )?;
        let tensor = match (block, role) {
            (MetalBlock::Gdn(block), LinearRole::FfnGate) => &block.ffn_gate,
            (MetalBlock::Gdn(block), LinearRole::FfnUp) => &block.ffn_up,
            (MetalBlock::Gdn(block), LinearRole::FfnDown) => &block.ffn_down,
            (MetalBlock::Gdn(block), LinearRole::GdnQkv) => &block.in_proj_qkv,
            (MetalBlock::Gdn(block), LinearRole::GdnZ) => &block.in_proj_z,
            (MetalBlock::Gdn(block), LinearRole::GdnBeta) => &block.beta_proj,
            (MetalBlock::Gdn(block), LinearRole::GdnAlpha) => &block.alpha_proj,
            (MetalBlock::Gdn(block), LinearRole::GdnOut) => &block.out_proj,
            (MetalBlock::Attn(block), LinearRole::FfnGate) => &block.ffn_gate,
            (MetalBlock::Attn(block), LinearRole::FfnUp) => &block.ffn_up,
            (MetalBlock::Attn(block), LinearRole::FfnDown) => &block.ffn_down,
            (MetalBlock::Attn(block), LinearRole::AttentionQAndGate) => &block.q,
            (MetalBlock::Attn(block), LinearRole::AttentionK) => &block.k,
            (MetalBlock::Attn(block), LinearRole::AttentionV) => &block.v,
            (MetalBlock::Attn(block), LinearRole::AttentionOut) => &block.o,
            _ => return Err(ResearchError::InvalidLinearRole { layer: index, role }),
        };
        Ok(tensor)
    }
}

#[allow(clippy::too_many_arguments)]
fn dense_ffn_vjp_readback(
    context: &MetalContext,
    layer: u32,
    hidden_size: usize,
    intermediate_size: usize,
    pre_ffn_residual: &[f32],
    post_norm: &MetalTensor,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    down_weight: &MetalTensor,
    grad_output: &[f32],
    n_query: usize,
    rule: DenseFfnVjpRule,
) -> Result<Vec<f32>, ResearchError> {
    if n_query == 0 {
        return Err(ResearchError::EmptyQueryBatch);
    }
    let gate_id = ResearchLinear::Layer {
        index: layer,
        role: LinearRole::FfnGate,
    };
    let up_id = ResearchLinear::Layer {
        index: layer,
        role: LinearRole::FfnUp,
    };
    let down_id = ResearchLinear::Layer {
        index: layer,
        role: LinearRole::FfnDown,
    };
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnGate,
        linear_shape(gate_id, gate_weight)?,
        [hidden_size, intermediate_size],
    )?;
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnUp,
        linear_shape(up_id, up_weight)?,
        [hidden_size, intermediate_size],
    )?;
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnDown,
        linear_shape(down_id, down_weight)?,
        [intermediate_size, hidden_size],
    )?;
    if post_norm.dtype != GgmlType::F32 || post_norm.shape != [hidden_size as u64] {
        return Err(ResearchError::InvalidDenseFfnNorm {
            layer,
            dtype: post_norm.dtype,
            shape: post_norm.shape.clone(),
            expected: hidden_size,
        });
    }
    for (id, weight) in [
        (gate_id, gate_weight),
        (up_id, up_weight),
        (down_id, down_weight),
    ] {
        validate_vjp_dtype(id, weight)?;
    }
    if pre_ffn_residual.len() != hidden_size {
        return Err(ResearchError::ActivationSize {
            name: "pre-FFN residual",
            got: pre_ffn_residual.len(),
            expected: hidden_size,
        });
    }
    let hidden_query_elements = n_query
        .checked_mul(hidden_size)
        .ok_or(ResearchError::SizeOverflow)?;
    if grad_output.len() != hidden_query_elements {
        return Err(ResearchError::CotangentSize {
            got: grad_output.len(),
            expected: hidden_query_elements,
            n_query,
            n_out: hidden_size,
        });
    }
    let n_query_u64 = u64::try_from(n_query).map_err(|_| ResearchError::SizeOverflow)?;
    let hidden_u64 = u64::try_from(hidden_size).map_err(|_| ResearchError::SizeOverflow)?;
    let intermediate_u64 =
        u64::try_from(intermediate_size).map_err(|_| ResearchError::SizeOverflow)?;

    let pre_ffn_residual = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(pre_ffn_residual),
        vec![hidden_u64],
        GgmlType::F32,
    )?;
    let normalized = MetalTensor::zeros_f32(context, vec![hidden_u64])?;
    let gate = MetalTensor::zeros_f32(context, vec![intermediate_u64])?;
    let up = MetalTensor::zeros_f32(context, vec![intermediate_u64])?;
    let grad_output = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(grad_output),
        vec![hidden_u64, n_query_u64],
        GgmlType::F32,
    )?;
    let intermediate_query_shape = vec![intermediate_u64, n_query_u64];
    let hidden_query_shape = vec![hidden_u64, n_query_u64];
    let grad_inner = MetalTensor::zeros_f32(context, intermediate_query_shape.clone())?;
    let grad_gate = MetalTensor::zeros_f32(context, intermediate_query_shape.clone())?;
    let grad_up = MetalTensor::zeros_f32(context, intermediate_query_shape)?;
    let grad_norm_gate = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_norm_up = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_norm = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_ffn_input = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(context, hidden_query_shape)?;

    let command = context
        .queue
        .commandBuffer()
        .ok_or(ResearchError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), ResearchError> {
        encode_rms_norm_mul_f32(
            context,
            &encoder,
            &pre_ffn_residual,
            post_norm,
            &normalized,
            RMS_EPS,
        )?;
        encode_mat_vec_dispatch(
            context,
            &encoder,
            gate_weight,
            &normalized,
            &gate,
            hidden_size,
            intermediate_size,
        )?;
        encode_mat_vec_dispatch(
            context,
            &encoder,
            up_weight,
            &normalized,
            &up,
            hidden_size,
            intermediate_size,
        )?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            down_weight,
            &grad_output,
            &grad_inner,
            intermediate_size,
            hidden_size,
            n_query,
        )?;
        let (rms_rule, swiglu_rule) = match rule {
            DenseFfnVjpRule::Jacobian => (RmsNormVjpRule::Jacobian, SwiGluVjpRule::Jacobian),
            DenseFfnVjpRule::Relp => (
                RmsNormVjpRule::RelpDetachedScale,
                SwiGluVjpRule::RelpIdentityHalf,
            ),
        };
        encode_silu_mul_vjp_broadcast_f32(
            context,
            &encoder,
            &gate,
            &up,
            &grad_inner,
            &grad_gate,
            &grad_up,
            n_query,
            intermediate_size,
            swiglu_rule,
        )?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            gate_weight,
            &grad_gate,
            &grad_norm_gate,
            hidden_size,
            intermediate_size,
            n_query,
        )?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            up_weight,
            &grad_up,
            &grad_norm_up,
            hidden_size,
            intermediate_size,
            n_query,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_norm_gate,
            &grad_norm_up,
            &grad_norm,
        )?;
        encode_rms_norm_mul_vjp_broadcast_f32(
            context,
            &encoder,
            &pre_ffn_residual,
            post_norm,
            &grad_norm,
            &grad_ffn_input,
            n_query,
            hidden_size,
            RMS_EPS,
            rms_rule,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_output,
            &grad_ffn_input,
            &grad_input,
        )?;
        Ok(())
    })();
    encoder.end();
    encode_result?;
    command.commit();
    command.waitUntilCompleted();
    let status = command.status();
    let error = command.error();
    if status != MTLCommandBufferStatus::Completed || error.is_some() {
        return Err(ResearchError::CommandBuffer {
            status: format!("{status:?}"),
            error: format!("{error:?}"),
        });
    }
    Ok(read_f32(&grad_input, hidden_query_elements))
}

fn validate_dense_ffn_shape(
    layer: u32,
    role: LinearRole,
    got: [usize; 2],
    expected: [usize; 2],
) -> Result<(), ResearchError> {
    if got != expected {
        return Err(ResearchError::InvalidDenseFfnShape {
            layer,
            role,
            got,
            expected,
        });
    }
    Ok(())
}

fn validate_vjp_dtype(id: ResearchLinear, weight: &MetalTensor) -> Result<(), ResearchError> {
    if !matches!(
        weight.dtype,
        GgmlType::Q8_0 | GgmlType::BF16 | GgmlType::F16 | GgmlType::F32
    ) {
        return Err(ResearchError::UnsupportedLinearDtype {
            id,
            dtype: weight.dtype,
        });
    }
    Ok(())
}

fn linear_shape(id: ResearchLinear, tensor: &MetalTensor) -> Result<[usize; 2], ResearchError> {
    let [n_in, n_out] = tensor.shape.as_slice() else {
        return Err(ResearchError::InvalidLinearShape {
            id,
            shape: tensor.shape.clone(),
        });
    };
    Ok([
        usize::try_from(*n_in).map_err(|_| ResearchError::SizeOverflow)?,
        usize::try_from(*n_out).map_err(|_| ResearchError::SizeOverflow)?,
    ])
}

fn validate_capture_layers(n_layers: u32, capture_layers: &[u32]) -> Result<(), ResearchError> {
    if let Some(&layer) = capture_layers.iter().find(|&&layer| layer >= n_layers) {
        return Err(ResearchError::InvalidLayer { layer, n_layers });
    }
    Ok(())
}

fn read_f32(tensor: &MetalTensor, len: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; len];
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<f32>();
        std::ptr::copy_nonoverlapping(source, output.as_mut_ptr(), len);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_tensor(context: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        MetalTensor::from_bytes(context, bytemuck::cast_slice(values), shape, GgmlType::F32)
            .unwrap()
    }

    fn cpu_dense_ffn_vjp(
        residual: &[f32],
        norm: &[f32],
        gate_weight: &[f32],
        up_weight: &[f32],
        down_weight: &[f32],
        grad_output: &[f32],
        n_query: usize,
        intermediate_size: usize,
        rule: DenseFfnVjpRule,
    ) -> Vec<f32> {
        let hidden_size = residual.len();
        let sumsq: f32 = residual.iter().map(|value| value * value).sum();
        let scale = (sumsq / hidden_size as f32 + RMS_EPS).sqrt().recip();
        let normalized: Vec<f32> = residual
            .iter()
            .zip(norm)
            .map(|(value, weight)| value * scale * weight)
            .collect();
        let project = |weight: &[f32], n_in: usize, n_out: usize, input: &[f32]| {
            (0..n_out)
                .map(|output| {
                    (0..n_in)
                        .map(|input_index| weight[output * n_in + input_index] * input[input_index])
                        .sum::<f32>()
                })
                .collect::<Vec<_>>()
        };
        let gate = project(gate_weight, hidden_size, intermediate_size, &normalized);
        let up = project(up_weight, hidden_size, intermediate_size, &normalized);
        let mut result = vec![0.0f32; n_query * hidden_size];
        for query in 0..n_query {
            let incoming = &grad_output[query * hidden_size..(query + 1) * hidden_size];
            let mut grad_inner = vec![0.0f32; intermediate_size];
            for input in 0..intermediate_size {
                grad_inner[input] = (0..hidden_size)
                    .map(|output| {
                        down_weight[output * intermediate_size + input] * incoming[output]
                    })
                    .sum();
            }
            let mut grad_gate = vec![0.0f32; intermediate_size];
            let mut grad_up = vec![0.0f32; intermediate_size];
            for index in 0..intermediate_size {
                let sigmoid = 1.0 / (1.0 + (-gate[index]).exp());
                let silu = gate[index] * sigmoid;
                match rule {
                    DenseFfnVjpRule::Jacobian => {
                        let derivative = sigmoid * (1.0 + gate[index] * (1.0 - sigmoid));
                        grad_gate[index] = grad_inner[index] * up[index] * derivative;
                        grad_up[index] = grad_inner[index] * silu;
                    }
                    DenseFfnVjpRule::Relp => {
                        grad_gate[index] = 0.5 * grad_inner[index] * up[index] * sigmoid;
                        grad_up[index] = 0.5 * grad_inner[index] * silu;
                    }
                }
            }
            let mut grad_norm = vec![0.0f32; hidden_size];
            for input in 0..hidden_size {
                grad_norm[input] = (0..intermediate_size)
                    .map(|output| {
                        gate_weight[output * hidden_size + input] * grad_gate[output]
                            + up_weight[output * hidden_size + input] * grad_up[output]
                    })
                    .sum();
            }
            let dot: f32 = (0..hidden_size)
                .map(|index| residual[index] * grad_norm[index] * norm[index])
                .sum();
            let correction = dot * scale * scale * scale / hidden_size as f32;
            for index in 0..hidden_size {
                let direct = grad_norm[index] * norm[index] * scale;
                let ffn_branch = match rule {
                    DenseFfnVjpRule::Jacobian => direct - residual[index] * correction,
                    DenseFfnVjpRule::Relp => direct,
                };
                result[query * hidden_size + index] = incoming[index] + ffn_branch;
            }
        }
        result
    }

    #[test]
    fn capture_layer_validation_preserves_unsorted_duplicates() {
        let layers = [7, 1, 7, 0];
        validate_capture_layers(8, &layers).unwrap();
        assert_eq!(layers, [7, 1, 7, 0]);
    }

    #[test]
    fn capture_layer_validation_rejects_first_out_of_range_layer() {
        let error = validate_capture_layers(8, &[1, 8, 9]).unwrap_err();
        assert!(matches!(
            error,
            ResearchError::InvalidLayer {
                layer: 8,
                n_layers: 8
            }
        ));
    }

    #[test]
    fn dense_ffn_vjp_composes_jacobian_and_relp_rules() {
        let context = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const HIDDEN: usize = 11;
        const INTERMEDIATE: usize = 17;
        let residual: Vec<f32> = (0..HIDDEN)
            .map(|index| ((index * 7 + 3) % 19) as f32 * 0.041 - 0.31)
            .collect();
        let norm: Vec<f32> = (0..HIDDEN)
            .map(|index| 0.61 + (index % 5) as f32 * 0.08)
            .collect();
        let gate_weight: Vec<f32> = (0..HIDDEN * INTERMEDIATE)
            .map(|index| ((index * 11 + 5) % 37) as f32 * 0.006 - 0.097)
            .collect();
        let up_weight: Vec<f32> = (0..HIDDEN * INTERMEDIATE)
            .map(|index| ((index * 13 + 2) % 41) as f32 * 0.005 - 0.083)
            .collect();
        let down_weight: Vec<f32> = (0..INTERMEDIATE * HIDDEN)
            .map(|index| ((index * 17 + 1) % 43) as f32 * 0.004 - 0.071)
            .collect();
        let norm_tensor = f32_tensor(&context, &norm, vec![HIDDEN as u64]);
        let gate_tensor = f32_tensor(
            &context,
            &gate_weight,
            vec![HIDDEN as u64, INTERMEDIATE as u64],
        );
        let up_tensor = f32_tensor(
            &context,
            &up_weight,
            vec![HIDDEN as u64, INTERMEDIATE as u64],
        );
        let down_tensor = f32_tensor(
            &context,
            &down_weight,
            vec![INTERMEDIATE as u64, HIDDEN as u64],
        );

        for n_query in [1usize, 2, 8] {
            let grad_output: Vec<f32> = (0..n_query * HIDDEN)
                .map(|index| ((index * 19 + 4) % 47) as f32 * 0.009 - 0.18)
                .collect();
            for rule in [DenseFfnVjpRule::Jacobian, DenseFfnVjpRule::Relp] {
                let expected = cpu_dense_ffn_vjp(
                    &residual,
                    &norm,
                    &gate_weight,
                    &up_weight,
                    &down_weight,
                    &grad_output,
                    n_query,
                    INTERMEDIATE,
                    rule,
                );
                let actual = dense_ffn_vjp_readback(
                    &context,
                    3,
                    HIDDEN,
                    INTERMEDIATE,
                    &residual,
                    &norm_tensor,
                    &gate_tensor,
                    &up_tensor,
                    &down_tensor,
                    &grad_output,
                    n_query,
                    rule,
                )
                .unwrap();
                let max_abs = actual
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs < 4e-5,
                    "rule={rule:?} n_query={n_query}: max error {max_abs}"
                );
            }
        }
        let mismatch = dense_ffn_vjp_readback(
            &context,
            3,
            HIDDEN + 1,
            INTERMEDIATE,
            &residual,
            &norm_tensor,
            &gate_tensor,
            &up_tensor,
            &down_tensor,
            &vec![0.0; HIDDEN],
            1,
            DenseFfnVjpRule::Jacobian,
        )
        .expect_err("architecture/weight shape mismatch must fail");
        assert!(matches!(
            mismatch,
            ResearchError::InvalidDenseFfnShape {
                role: LinearRole::FfnGate,
                ..
            }
        ));
    }

    #[test]
    fn dense_ffn_vjp_preserves_the_identity_residual() {
        let context = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const HIDDEN: usize = 7;
        const INTERMEDIATE: usize = 9;
        const N_QUERY: usize = 2;
        let residual = vec![0.25f32; HIDDEN];
        let norm = vec![1.0f32; HIDDEN];
        let gate_weight = vec![0.03f32; HIDDEN * INTERMEDIATE];
        let up_weight = vec![-0.02f32; HIDDEN * INTERMEDIATE];
        let down_weight = vec![0.0f32; INTERMEDIATE * HIDDEN];
        let grad_output: Vec<f32> = (0..N_QUERY * HIDDEN)
            .map(|index| index as f32 * 0.017 - 0.09)
            .collect();
        let norm = f32_tensor(&context, &norm, vec![HIDDEN as u64]);
        let gate = f32_tensor(
            &context,
            &gate_weight,
            vec![HIDDEN as u64, INTERMEDIATE as u64],
        );
        let up = f32_tensor(
            &context,
            &up_weight,
            vec![HIDDEN as u64, INTERMEDIATE as u64],
        );
        let down = f32_tensor(
            &context,
            &down_weight,
            vec![INTERMEDIATE as u64, HIDDEN as u64],
        );
        for rule in [DenseFfnVjpRule::Jacobian, DenseFfnVjpRule::Relp] {
            let actual = dense_ffn_vjp_readback(
                &context,
                0,
                HIDDEN,
                INTERMEDIATE,
                &residual,
                &norm,
                &gate,
                &up,
                &down,
                &grad_output,
                N_QUERY,
                rule,
            )
            .unwrap();
            assert_eq!(actual, grad_output, "identity branch changed in {rule:?}");
        }
    }
}
