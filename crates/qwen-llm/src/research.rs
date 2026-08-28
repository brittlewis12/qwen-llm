//! Safe, narrow instrumentation surface for workspace-lens research.
//!
//! This module deliberately returns owned CPU data and opaque linear IDs. It
//! does not expose resident model buffers, command encoders, or mutable Metal
//! session state across the public API boundary.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, RmsNormVjpRule, SwiGluVjpRule,
    encode_add_f32, encode_fill_f32, encode_frozen_linear_vjp_f32,
    encode_gdn_decay_chain_batched_f32, encode_gdn_decay_chain_vjp_f32,
    encode_gdn_prep_packed_ckpt_f32, encode_gdn_step_decay_packed_ckpt_f32,
    encode_gdn_step_decay_packed_vjp_f32, encode_l2_norm_batched_f32,
    encode_l2_norm_vjp_batched_f32, encode_rms_norm_batched_f32, encode_rms_norm_mul_f32,
    encode_rms_norm_mul_rows_f32, encode_rms_norm_mul_vjp_broadcast_f32,
    encode_rms_norm_mul_vjp_rows_f32, encode_rmsnorm_gated_f32, encode_rmsnorm_gated_vjp_f32,
    encode_sigmoid_f32, encode_sigmoid_output_vjp_f32, encode_silu_mul_vjp_broadcast_f32,
    encode_silu_mul_vjp_f32, encode_split_q_gate_f32, encode_ssm_conv_silu_split_packed_vjp_f32,
};
use crate::metal_forward::{
    MetalAttnBlock, MetalBlock, MetalGdnBlock, MfError, RMS_EPS, encode_mat_vec_dispatch,
};
use crate::model::{Arch, ArchKind};
use crate::runtime::{LoadedModel, RuntimeError, Sequence};
use crate::tensor::GgmlType;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};

/// Lightweight locator identity derived from model metadata, shard paths, and
/// file stamps. It is useful within one machine, but is not a content digest.
pub const RESEARCH_IDENTITY_SCHEME: &str = "qwen_llm_model_locator_v1";
pub const MAX_RESEARCH_GDN_TOKENS: usize = 16;
pub const MAX_RESEARCH_ATTN_TOKENS: usize = 16;
pub const MAX_RESEARCH_WORKSPACE_TOKENS: usize = 16;
pub const MAX_RESEARCH_WORKSPACE_DIM_BATCH: usize = 32;

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

#[derive(Clone, Debug, PartialEq)]
pub struct ResearchGdnForward {
    identity: ResearchModelIdentity,
    owner_token_id: u64,
    layer: u32,
    start_position: usize,
    token_ids: Vec<i32>,
    hidden_size: usize,
    final_logits: Vec<f32>,
    input_residuals: Vec<f32>,
    post_mixer_residuals: Vec<f32>,
    post_block_residuals: Vec<f32>,
    initial_conv_state: Vec<f32>,
    initial_recurrence_state: Vec<f32>,
    final_conv_state: Vec<f32>,
    final_recurrence_state: Vec<f32>,
}

impl ResearchGdnForward {
    pub fn identity(&self) -> ResearchModelIdentity {
        self.identity
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    pub fn start_position(&self) -> usize {
        self.start_position
    }

    pub fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn final_logits(&self) -> &[f32] {
        &self.final_logits
    }

    pub fn n_tokens(&self) -> usize {
        self.token_ids.len()
    }

    /// Real residual-stream inputs to the selected layer, flattened `[T, H]`.
    pub fn input_residuals(&self) -> &[f32] {
        &self.input_residuals
    }

    /// Real post-mixer residuals `input + mixer(input)`, flattened `[T, H]`.
    pub fn post_mixer_residuals(&self) -> &[f32] {
        &self.post_mixer_residuals
    }

    /// Real post-FFN block outputs, flattened `[T, H]`.
    pub fn post_block_residuals(&self) -> &[f32] {
        &self.post_block_residuals
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GdnMixerVjpRule {
    /// Ordinary pre-mixer RMSNorm Jacobian used by J-lens.
    Jacobian,
    /// Released Qwen R-lens rule: detach the pre-mixer RMS denominator.
    Relp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResearchGdnVjp {
    pub layer: u32,
    pub n_tokens: usize,
    pub hidden_size: usize,
    /// Mixer-branch input cotangents, flattened `[T, H]`. The residual identity
    /// branch is intentionally not included.
    pub values: Vec<f32>,
    pub grad_initial_conv_state: Vec<f32>,
    pub grad_initial_recurrence_state: Vec<f32>,
    pub replay_mixer_outputs: Vec<f32>,
    pub residual_replay_max_abs_error: f32,
    pub final_conv_state_max_abs_error: f32,
    pub final_recurrence_state_max_abs_error: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GdnBlockVjpRule {
    /// Ordinary Jacobians throughout the block.
    Jacobian,
    /// Released Qwen R-lens rules on both residual-stream RMSNorms and the
    /// FFN gate/product. GDN-internal normalization remains an ordinary
    /// Jacobian.
    Relp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResearchGdnBlockVjp {
    pub layer: u32,
    pub n_tokens: usize,
    pub hidden_size: usize,
    /// Full block-input cotangents, including both residual identities.
    pub values: Vec<f32>,
    /// Cotangents after reversing the FFN residual update and before the mixer
    /// residual update, flattened `[T, H]`.
    pub grad_post_mixer_residuals: Vec<f32>,
    pub mixer: ResearchGdnVjp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResearchAttnForward {
    identity: ResearchModelIdentity,
    owner_token_id: u64,
    layer: u32,
    token_ids: Vec<i32>,
    hidden_size: usize,
    final_logits: Vec<f32>,
    input_residuals: Vec<f32>,
    post_mixer_residuals: Vec<f32>,
    post_block_residuals: Vec<f32>,
}

impl ResearchAttnForward {
    pub fn identity(&self) -> ResearchModelIdentity {
        self.identity
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    pub fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }

    pub fn n_tokens(&self) -> usize {
        self.token_ids.len()
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn final_logits(&self) -> &[f32] {
        &self.final_logits
    }

    pub fn input_residuals(&self) -> &[f32] {
        &self.input_residuals
    }

    pub fn post_mixer_residuals(&self) -> &[f32] {
        &self.post_mixer_residuals
    }

    pub fn post_block_residuals(&self) -> &[f32] {
        &self.post_block_residuals
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttnBlockVjpRule {
    /// Ordinary Jacobians throughout the block.
    Jacobian,
    /// Released Qwen R-lens rules on residual-stream RMSNorms and the FFN.
    /// Attention, Q/K normalization, RoPE, softmax, and gating stay ordinary.
    Relp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResearchAttnBlockVjp {
    pub layer: u32,
    pub n_tokens: usize,
    pub hidden_size: usize,
    pub values: Vec<f32>,
    pub grad_post_mixer_residuals: Vec<f32>,
    pub replay_mixer_outputs: Vec<f32>,
    /// Drift between the F32 model-level oracle replay and the production
    /// residual, whose KV path rounds K/V to F16 or Q8.
    pub residual_replay_max_abs_error: f32,
}

/// One fresh prompt's post-block residual boundaries and post-mixer residuals.
/// Banks are owned CPU F32 data in layer-major, token-major order. Layer IDs
/// follow the Hugging Face hook convention: layer `l` is the output of block
/// `l`, so fitting from target `t` to source `s` reverses blocks `t..s+1`.
#[derive(Clone, Debug, PartialEq)]
pub struct ResearchWorkspaceForward {
    identity: ResearchModelIdentity,
    owner_token_id: u64,
    token_ids: Vec<i32>,
    n_layers: u32,
    hidden_size: usize,
    post_mixer_residuals: Vec<f32>,
    post_block_residuals: Vec<f32>,
}

impl ResearchWorkspaceForward {
    pub fn identity(&self) -> ResearchModelIdentity {
        self.identity
    }

    pub fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }

    pub fn n_tokens(&self) -> usize {
        self.token_ids.len()
    }

    pub fn n_layers(&self) -> u32 {
        self.n_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Real `input + mixer(input)` residuals for one layer, flattened `[T,H]`.
    pub fn post_mixer_residuals(&self, layer: u32) -> Option<&[f32]> {
        self.layer_bank(&self.post_mixer_residuals, layer)
    }

    /// Real post-FFN block outputs for one layer, flattened `[T,H]`.
    pub fn post_block_residuals(&self, layer: u32) -> Option<&[f32]> {
        self.layer_bank(&self.post_block_residuals, layer)
    }

    /// Real block inputs, flattened `[T,H]`. Layer zero's embedding input is
    /// intentionally absent because the reference lens only fits source layers
    /// strictly below a target layer and never reverses through block zero.
    pub fn input_residuals(&self, layer: u32) -> Option<&[f32]> {
        layer
            .checked_sub(1)
            .and_then(|previous| self.post_block_residuals(previous))
    }

    fn layer_bank<'a>(&self, values: &'a [f32], layer: u32) -> Option<&'a [f32]> {
        if layer >= self.n_layers {
            return None;
        }
        let layer_elements = self.n_tokens().checked_mul(self.hidden_size)?;
        let start = usize::try_from(layer).ok()?.checked_mul(layer_elements)?;
        values.get(start..start.checked_add(layer_elements)?)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkspaceLensRule {
    /// Ordinary activation Jacobians throughout every traversed block.
    Jacobian,
    /// Released Qwen R-lens transport rules at residual RMSNorm and SwiGLU
    /// sites; attention and GDN internals retain ordinary Jacobians.
    Relp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResearchWorkspaceBlockKind {
    Gdn,
    Attention,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResearchWorkspaceReplayDiagnostic {
    pub layer: u32,
    pub kind: ResearchWorkspaceBlockKind,
    /// Drift between replayed `input + mixer(input)` and the production
    /// residual. Attention replay is an F32 model-level oracle and therefore
    /// does not differentiate production F16/Q8 KV conversion.
    pub residual_replay_max_abs_error: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResearchWorkspaceVjp {
    pub target_layer: u32,
    /// Caller order, including duplicates.
    pub source_layers: Vec<u32>,
    pub n_tokens: usize,
    pub hidden_size: usize,
    /// Source-layer order, flattened `[K,T,H]`.
    pub values: Vec<f32>,
    /// Reverse traversal order, from the target block toward the earliest
    /// requested source layer.
    pub diagnostics: Vec<ResearchWorkspaceReplayDiagnostic>,
}

impl ResearchWorkspaceVjp {
    pub fn source_values(&self, slot: usize) -> Option<&[f32]> {
        let source_elements = self.n_tokens.checked_mul(self.hidden_size)?;
        let start = slot.checked_mul(source_elements)?;
        self.values.get(start..start.checked_add(source_elements)?)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResearchWorkspaceVjpBatch {
    pub target_layer: u32,
    /// Caller order, including duplicates.
    pub source_layers: Vec<u32>,
    pub n_query: usize,
    pub n_tokens: usize,
    pub hidden_size: usize,
    /// Source-layer major, then query major: `[K,Q,T,H]`.
    pub values: Vec<f32>,
    /// Reverse traversal order, from the target block toward the earliest
    /// requested source layer.
    pub diagnostics: Vec<ResearchWorkspaceReplayDiagnostic>,
}

impl ResearchWorkspaceVjpBatch {
    pub fn source_values(&self, slot: usize) -> Option<&[f32]> {
        let trajectory_elements = self.n_tokens.checked_mul(self.hidden_size)?;
        let source_elements = self.n_query.checked_mul(trajectory_elements)?;
        let start = slot.checked_mul(source_elements)?;
        self.values.get(start..start.checked_add(source_elements)?)
    }

    pub fn source_query_values(&self, source_slot: usize, query_slot: usize) -> Option<&[f32]> {
        if query_slot >= self.n_query {
            return None;
        }
        let trajectory_elements = self.n_tokens.checked_mul(self.hidden_size)?;
        let source_elements = self.n_query.checked_mul(trajectory_elements)?;
        let start = source_slot
            .checked_mul(source_elements)?
            .checked_add(query_slot.checked_mul(trajectory_elements)?)?;
        self.values
            .get(start..start.checked_add(trajectory_elements)?)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResearchWorkspaceRows {
    pub target_layer: u32,
    pub source_layers: Vec<u32>,
    /// Target/output coordinates in caller order.
    pub output_rows: Vec<u32>,
    pub n_tokens: usize,
    pub n_valid_positions: usize,
    pub hidden_size: usize,
    /// Source-layer major, then output-row major: `[K,R,H]`.
    pub values: Vec<f32>,
    /// Maximum replay drift observed for each traversed block across all rows.
    pub diagnostics: Vec<ResearchWorkspaceReplayDiagnostic>,
}

impl ResearchWorkspaceRows {
    pub fn source_values(&self, slot: usize) -> Option<&[f32]> {
        let source_elements = self.output_rows.len().checked_mul(self.hidden_size)?;
        let start = slot.checked_mul(source_elements)?;
        self.values.get(start..start.checked_add(source_elements)?)
    }

    pub fn row_values(&self, source_slot: usize, row_slot: usize) -> Option<&[f32]> {
        if row_slot >= self.output_rows.len() {
            return None;
        }
        let source_elements = self.output_rows.len().checked_mul(self.hidden_size)?;
        let start = source_slot
            .checked_mul(source_elements)?
            .checked_add(row_slot.checked_mul(self.hidden_size)?)?;
        self.values.get(start..start.checked_add(self.hidden_size)?)
    }
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
    #[error("layer {layer} is not a GDN layer")]
    NotGdnLayer { layer: u32 },
    #[error("GDN prompt capture requires layer > 0 so its real input residual can be observed")]
    GdnCaptureRequiresPreviousLayer,
    #[error("GDN prompt capture requires at least one token")]
    EmptyGdnPrompt,
    #[error("GDN prompt capture length {got} exceeds the bounded isolated-sequence limit {max}")]
    GdnPromptTooLong { got: usize, max: usize },
    #[error(
        "layer {layer} GDN geometry must use head_dim=128 with nonzero V/K heads and V divisible by K; got n_v={n_v} n_k={n_k} head_dim={head_dim}"
    )]
    UnsupportedGdnGeometry {
        layer: u32,
        n_v: usize,
        n_k: usize,
        head_dim: usize,
    },
    #[error("layer {layer} {role:?} has shape {got:?}, expected {expected:?}")]
    InvalidGdnLinearShape {
        layer: u32,
        role: LinearRole,
        got: [usize; 2],
        expected: [usize; 2],
    },
    #[error(
        "layer {layer} GDN tensor {name} must be F32 with {expected_elements} elements, got {dtype:?} {shape:?}"
    )]
    InvalidGdnTensor {
        layer: u32,
        name: &'static str,
        dtype: GgmlType,
        shape: Vec<u64>,
        expected_elements: usize,
    },
    #[error("GDN capture belongs to a different model identity")]
    GdnCaptureModelMismatch,
    #[error("GDN capture belongs to a different loaded-model owner")]
    GdnCaptureOwnerMismatch,
    #[error("layer {layer} is not a full-attention layer")]
    NotAttentionLayer { layer: u32 },
    #[error("attention prompt capture requires layer > 0")]
    AttnCaptureRequiresPreviousLayer,
    #[error("attention prompt capture requires a fresh sequence at position zero, got {0}")]
    AttnCaptureRequiresFreshSequence(usize),
    #[error("attention prompt capture requires at least one token")]
    EmptyAttnPrompt,
    #[error("attention prompt capture length {got} exceeds the bounded F32-oracle limit {max}")]
    AttnPromptTooLong { got: usize, max: usize },
    #[error("attention capture belongs to a different model identity")]
    AttnCaptureModelMismatch,
    #[error("attention capture belongs to a different loaded-model owner")]
    AttnCaptureOwnerMismatch,
    #[error("layer {layer} {role:?} has shape {got:?}, expected {expected:?}")]
    InvalidAttnLinearShape {
        layer: u32,
        role: LinearRole,
        got: [usize; 2],
        expected: [usize; 2],
    },
    #[error(
        "layer {layer} attention tensor {name} must be F32 with {expected_elements} elements, got {dtype:?} {shape:?}"
    )]
    InvalidAttnTensor {
        layer: u32,
        name: &'static str,
        dtype: GgmlType,
        shape: Vec<u64>,
        expected_elements: usize,
    },
    #[error("workspace prompt capture requires a fresh sequence at position zero, got {0}")]
    WorkspaceCaptureRequiresFreshSequence(usize),
    #[error("workspace prompt capture requires at least one token")]
    EmptyWorkspacePrompt,
    #[error("workspace prompt capture length {got} exceeds the bounded limit {max}")]
    WorkspacePromptTooLong { got: usize, max: usize },
    #[error("workspace capture belongs to a different model identity")]
    WorkspaceCaptureModelMismatch,
    #[error("workspace capture belongs to a different loaded-model owner")]
    WorkspaceCaptureOwnerMismatch,
    #[error("workspace VJP requires at least one source layer")]
    EmptyWorkspaceSourceLayers,
    #[error(
        "workspace source layer {source_layer} must be strictly below target layer {target_layer}"
    )]
    WorkspaceSourceNotBeforeTarget {
        source_layer: u32,
        target_layer: u32,
    },
    #[error("workspace row fit requires at least one output coordinate")]
    EmptyWorkspaceOutputRows,
    #[error("workspace output rows must be strictly increasing and unique")]
    WorkspaceOutputRowsNotStrict,
    #[error("workspace output coordinate {row} is out of range for hidden size {hidden_size}")]
    WorkspaceOutputRowOutOfRange { row: u32, hidden_size: usize },
    #[error(
        "workspace prompt length {n_tokens} leaves no valid positions with skip_first={skip_first}; require length >= skip_first + 2"
    )]
    WorkspaceNoValidPositions { n_tokens: usize, skip_first: usize },
    #[error("workspace replay diagnostic schedule changed across fitted rows")]
    WorkspaceDiagnosticScheduleMismatch,
    #[error("workspace query batch {got} exceeds the bounded limit {max}")]
    WorkspaceQueryBatchTooLarge { got: usize, max: usize },
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

    fn forward_token_with_dense_ffn_capture_no_tail(
        &mut self,
        token_id: i32,
        capture_layers: &[u32],
    ) -> Result<DenseFfnActivationCapture, ResearchError> {
        self.sequence.ensure_can_append(1)?;
        validate_capture_layers(self.arch().n_layer, capture_layers)?;
        let position = self.sequence.position();
        let position_u32 =
            u32::try_from(position).map_err(|_| ResearchError::PositionOverflow(position))?;
        let hidden_size = self.arch().hidden_size as usize;
        let capture_len = checked_product(capture_layers.len(), hidden_size)?;
        let forward = self.model.forward();
        let state = unsafe { self.sequence.metal_session_mut() };
        state.ensure_usable()?;
        let (pre_ffn_residuals, post_block_residuals) = if capture_layers.is_empty() {
            forward.single_token_no_tail(token_id, position_u32, state)?;
            (Vec::new(), Vec::new())
        } else {
            let shape = vec![hidden_size as u64, capture_layers.len() as u64];
            let pre_ffn = MetalTensor::zeros_f32(self.model.context(), shape.clone())?;
            let post_block = MetalTensor::zeros_f32(self.model.context(), shape)?;
            forward.single_token_with_dense_ffn_capture_no_tail(
                token_id,
                position_u32,
                state,
                capture_layers,
                &pre_ffn,
                &post_block,
            )?;
            (
                read_f32(&pre_ffn, capture_len),
                read_f32(&post_block, capture_len),
            )
        };
        self.sequence.advance_by(1)?;
        Ok(DenseFfnActivationCapture {
            layer_ids: capture_layers.to_vec(),
            hidden_size,
            pre_ffn_residuals,
            post_block_residuals,
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

    /// Advance a bounded prompt while capturing the real input trajectory and
    /// recurrent boundary states for one GDN layer.
    ///
    /// The selected layer must be nonzero. A command failure after an earlier
    /// token succeeds can leave that successful prefix consumed, matching the
    /// existing token-at-a-time research forward contract.
    pub fn forward_prompt_with_gdn_capture(
        &mut self,
        token_ids: &[i32],
        layer: u32,
    ) -> Result<ResearchGdnForward, ResearchError> {
        if token_ids.is_empty() {
            return Err(ResearchError::EmptyGdnPrompt);
        }
        if token_ids.len() > MAX_RESEARCH_GDN_TOKENS {
            return Err(ResearchError::GdnPromptTooLong {
                got: token_ids.len(),
                max: MAX_RESEARCH_GDN_TOKENS,
            });
        }
        if layer == 0 {
            return Err(ResearchError::GdnCaptureRequiresPreviousLayer);
        }
        for &token_id in token_ids {
            if token_id < 0 || token_id as u32 >= self.arch().vocab_size {
                return Err(MfError::BadToken(token_id, self.arch().vocab_size).into());
            }
        }
        self.sequence.ensure_can_append(token_ids.len())?;
        let start_position = self.sequence.position();
        let last_position = start_position
            .checked_add(token_ids.len() - 1)
            .ok_or(ResearchError::SizeOverflow)?;
        u32::try_from(last_position).map_err(|_| ResearchError::PositionOverflow(last_position))?;
        let (gdn_index, geometry) = {
            let (block, gdn_index, geometry) = self.resolve_gdn(layer)?;
            validate_gdn_weights(layer, block, geometry)?;
            (gdn_index, geometry)
        };
        let (initial_conv_state, initial_recurrence_state) = {
            let state = unsafe { self.sequence.metal_session_mut() };
            state.ensure_usable()?;
            let conv = state
                .gdn_conv
                .get(gdn_index)
                .ok_or(ResearchError::SizeOverflow)?;
            let recurrence = state
                .gdn_state
                .get(gdn_index)
                .ok_or(ResearchError::SizeOverflow)?;
            (
                read_f32(conv, geometry.conv_state_elements),
                read_f32(recurrence, geometry.state_elements),
            )
        };

        let hidden_elements = token_ids
            .len()
            .checked_mul(geometry.hidden_size)
            .ok_or(ResearchError::SizeOverflow)?;
        let mut input_residuals = Vec::with_capacity(hidden_elements);
        let mut post_mixer_residuals = Vec::with_capacity(hidden_elements);
        let mut post_block_residuals = Vec::with_capacity(hidden_elements);
        let mut final_logits = Vec::new();
        let capture_layers = [layer - 1, layer];
        for &token_id in token_ids {
            let forward = match self.forward_token_with_dense_ffn_capture(token_id, &capture_layers)
            {
                Ok(forward) => forward,
                Err(error) => {
                    let state = unsafe { self.sequence.metal_session_mut() };
                    state.poison("GDN prompt capture forward failed");
                    return Err(error);
                }
            };
            let hidden = geometry.hidden_size;
            input_residuals.extend_from_slice(&forward.capture.post_block_residuals[..hidden]);
            post_mixer_residuals
                .extend_from_slice(&forward.capture.pre_ffn_residuals[hidden..2 * hidden]);
            post_block_residuals
                .extend_from_slice(&forward.capture.post_block_residuals[hidden..2 * hidden]);
            final_logits = forward.logits;
        }
        let (final_conv_state, final_recurrence_state) = {
            let state = unsafe { self.sequence.metal_session_mut() };
            state.ensure_usable()?;
            (
                read_f32(&state.gdn_conv[gdn_index], geometry.conv_state_elements),
                read_f32(&state.gdn_state[gdn_index], geometry.state_elements),
            )
        };
        Ok(ResearchGdnForward {
            identity: self.identity(),
            owner_token_id: self.model.owner_token_id(),
            layer,
            start_position,
            token_ids: token_ids.to_vec(),
            hidden_size: geometry.hidden_size,
            final_logits,
            input_residuals,
            post_mixer_residuals,
            post_block_residuals,
            initial_conv_state,
            initial_recurrence_state,
            final_conv_state,
            final_recurrence_state,
        })
    }

    /// Reverse the isolated mixer branch represented by a matching GDN prompt
    /// capture. The result stops at the selected layer's input residual. If the
    /// incoming cotangent is on `input + mixer(input)`, callers add that same
    /// cotangent as the residual identity branch. Terminal convolution and
    /// recurrence-state cotangents are fixed to zero, so captures are isolated
    /// sequences and cannot be stitched into a longer reverse pass.
    pub fn gdn_mixer_vjp(
        &self,
        forward: &ResearchGdnForward,
        grad_mixer_output: &[f32],
        rule: GdnMixerVjpRule,
    ) -> Result<ResearchGdnVjp, ResearchError> {
        if forward.identity != self.identity() {
            return Err(ResearchError::GdnCaptureModelMismatch);
        }
        if forward.owner_token_id != self.model.owner_token_id() {
            return Err(ResearchError::GdnCaptureOwnerMismatch);
        }
        let (block, _, geometry) = self.resolve_gdn(forward.layer)?;
        validate_gdn_weights(forward.layer, block, geometry)?;
        let n_tokens = forward.n_tokens();
        if n_tokens == 0 || n_tokens > MAX_RESEARCH_GDN_TOKENS {
            return Err(ResearchError::ActivationSize {
                name: "GDN capture token count",
                got: n_tokens,
                expected: 1,
            });
        }
        let hidden_elements = n_tokens
            .checked_mul(geometry.hidden_size)
            .ok_or(ResearchError::SizeOverflow)?;
        for (name, values, expected) in [
            (
                "GDN captured input residuals",
                forward.input_residuals.as_slice(),
                hidden_elements,
            ),
            (
                "GDN captured post-mixer residuals",
                forward.post_mixer_residuals.as_slice(),
                hidden_elements,
            ),
            (
                "GDN captured post-block residuals",
                forward.post_block_residuals.as_slice(),
                hidden_elements,
            ),
            (
                "GDN captured initial conv state",
                forward.initial_conv_state.as_slice(),
                geometry.conv_state_elements,
            ),
            (
                "GDN captured initial recurrence state",
                forward.initial_recurrence_state.as_slice(),
                geometry.state_elements,
            ),
            (
                "GDN captured final conv state",
                forward.final_conv_state.as_slice(),
                geometry.conv_state_elements,
            ),
            (
                "GDN captured final recurrence state",
                forward.final_recurrence_state.as_slice(),
                geometry.state_elements,
            ),
        ] {
            if values.len() != expected {
                return Err(ResearchError::ActivationSize {
                    name,
                    got: values.len(),
                    expected,
                });
            }
        }
        if forward.hidden_size != geometry.hidden_size {
            return Err(ResearchError::ActivationSize {
                name: "GDN capture hidden size",
                got: forward.hidden_size,
                expected: geometry.hidden_size,
            });
        }
        if grad_mixer_output.len() != hidden_elements {
            return Err(ResearchError::ActivationSize {
                name: "GDN mixer cotangent",
                got: grad_mixer_output.len(),
                expected: hidden_elements,
            });
        }
        let replay = gdn_mixer_replay_vjp_readback(
            self.model.context(),
            geometry,
            GdnMixerWeights::from(block),
            &forward.input_residuals,
            &forward.initial_conv_state,
            &forward.initial_recurrence_state,
            grad_mixer_output,
            n_tokens,
            rule,
            true,
        )?;
        let residual_replay_max_abs_error = forward
            .input_residuals
            .iter()
            .zip(&replay.mixer_outputs)
            .zip(&forward.post_mixer_residuals)
            .map(|((&input, &mixer), &observed)| finite_abs_difference(input + mixer, observed))
            .fold(0.0f32, f32::max);
        Ok(ResearchGdnVjp {
            layer: forward.layer,
            n_tokens,
            hidden_size: geometry.hidden_size,
            values: replay.grad_input,
            grad_initial_conv_state: replay.grad_initial_conv_state,
            grad_initial_recurrence_state: replay.grad_initial_recurrence_state,
            replay_mixer_outputs: replay.mixer_outputs,
            residual_replay_max_abs_error,
            final_conv_state_max_abs_error: max_abs_difference(
                &replay.final_conv_state,
                &forward.final_conv_state,
            ),
            final_recurrence_state_max_abs_error: max_abs_difference(
                &replay.final_recurrence_state,
                &forward.final_recurrence_state,
            ),
        })
    }

    /// Reverse a complete GDN block over the captured prompt trajectory.
    /// This composes the rowwise dense FFN VJP, its residual identity, the
    /// temporal GDN mixer VJP, and the mixer residual identity.
    pub fn gdn_block_vjp(
        &self,
        forward: &ResearchGdnForward,
        grad_block_output: &[f32],
        rule: GdnBlockVjpRule,
    ) -> Result<ResearchGdnBlockVjp, ResearchError> {
        if forward.identity != self.identity() {
            return Err(ResearchError::GdnCaptureModelMismatch);
        }
        if forward.owner_token_id != self.model.owner_token_id() {
            return Err(ResearchError::GdnCaptureOwnerMismatch);
        }
        let n_tokens = forward.n_tokens();
        let hidden_size = self.arch().hidden_size as usize;
        let expected = checked_product(n_tokens, hidden_size)?;
        if grad_block_output.len() != expected {
            return Err(ResearchError::ActivationSize {
                name: "GDN block cotangent",
                got: grad_block_output.len(),
                expected,
            });
        }
        if forward.post_mixer_residuals.len() != expected {
            return Err(ResearchError::ActivationSize {
                name: "GDN captured post-mixer residuals",
                got: forward.post_mixer_residuals.len(),
                expected,
            });
        }
        let (post_norm, gate, up, down) = self.resolve_dense_ffn(forward.layer)?;
        let composition = compose_gdn_block_vjp(
            self.model.context(),
            forward.layer,
            hidden_size,
            self.arch().intermediate_size as usize,
            &forward.post_mixer_residuals,
            post_norm,
            gate,
            up,
            down,
            grad_block_output,
            n_tokens,
            rule,
            |grad_post_mixer, mixer_rule| self.gdn_mixer_vjp(forward, grad_post_mixer, mixer_rule),
        )?;
        Ok(ResearchGdnBlockVjp {
            layer: forward.layer,
            n_tokens,
            hidden_size,
            values: composition.values,
            grad_post_mixer_residuals: composition.grad_post_mixer_residuals,
            mixer: composition.mixer,
        })
    }

    /// Advance a fresh bounded prompt while capturing one full-attention block.
    /// The replay derivative is the model-level F32 attention Jacobian; it does
    /// not differentiate the production F16/Q8 KV storage conversion.
    pub fn forward_prompt_with_attn_capture(
        &mut self,
        token_ids: &[i32],
        layer: u32,
    ) -> Result<ResearchAttnForward, ResearchError> {
        if token_ids.is_empty() {
            return Err(ResearchError::EmptyAttnPrompt);
        }
        if token_ids.len() > MAX_RESEARCH_ATTN_TOKENS {
            return Err(ResearchError::AttnPromptTooLong {
                got: token_ids.len(),
                max: MAX_RESEARCH_ATTN_TOKENS,
            });
        }
        if layer == 0 {
            return Err(ResearchError::AttnCaptureRequiresPreviousLayer);
        }
        if self.sequence.position() != 0 {
            return Err(ResearchError::AttnCaptureRequiresFreshSequence(
                self.sequence.position(),
            ));
        }
        for &token_id in token_ids {
            if token_id < 0 || token_id as u32 >= self.arch().vocab_size {
                return Err(MfError::BadToken(token_id, self.arch().vocab_size).into());
            }
        }
        self.sequence.ensure_can_append(token_ids.len())?;
        let geometry = {
            let (block, geometry) = self.resolve_attn(layer)?;
            validate_attn_weights(layer, block, geometry)?;
            geometry
        };
        let hidden_elements = checked_product(token_ids.len(), geometry.hidden_size)?;
        let mut input_residuals = Vec::with_capacity(hidden_elements);
        let mut post_mixer_residuals = Vec::with_capacity(hidden_elements);
        let mut post_block_residuals = Vec::with_capacity(hidden_elements);
        let mut final_logits = Vec::new();
        let capture_layers = [layer - 1, layer];
        for &token_id in token_ids {
            let forward = match self.forward_token_with_dense_ffn_capture(token_id, &capture_layers)
            {
                Ok(forward) => forward,
                Err(error) => {
                    let state = unsafe { self.sequence.metal_session_mut() };
                    state.poison("attention prompt capture forward failed");
                    return Err(error);
                }
            };
            let hidden = geometry.hidden_size;
            input_residuals.extend_from_slice(&forward.capture.post_block_residuals[..hidden]);
            post_mixer_residuals
                .extend_from_slice(&forward.capture.pre_ffn_residuals[hidden..2 * hidden]);
            post_block_residuals
                .extend_from_slice(&forward.capture.post_block_residuals[hidden..2 * hidden]);
            final_logits = forward.logits;
        }
        Ok(ResearchAttnForward {
            identity: self.identity(),
            owner_token_id: self.model.owner_token_id(),
            layer,
            token_ids: token_ids.to_vec(),
            hidden_size: geometry.hidden_size,
            final_logits,
            input_residuals,
            post_mixer_residuals,
            post_block_residuals,
        })
    }

    /// Reverse one captured full-attention block using an ordinary F32 causal
    /// attention Jacobian and the selected residual-stream/FFN lens rule.
    pub fn attn_block_vjp(
        &self,
        forward: &ResearchAttnForward,
        grad_block_output: &[f32],
        rule: AttnBlockVjpRule,
    ) -> Result<ResearchAttnBlockVjp, ResearchError> {
        if forward.identity != self.identity() {
            return Err(ResearchError::AttnCaptureModelMismatch);
        }
        if forward.owner_token_id != self.model.owner_token_id() {
            return Err(ResearchError::AttnCaptureOwnerMismatch);
        }
        let (block, geometry) = self.resolve_attn(forward.layer)?;
        validate_attn_weights(forward.layer, block, geometry)?;
        let n_tokens = forward.n_tokens();
        if n_tokens == 0 || n_tokens > MAX_RESEARCH_ATTN_TOKENS {
            return Err(ResearchError::AttnPromptTooLong {
                got: n_tokens,
                max: MAX_RESEARCH_ATTN_TOKENS,
            });
        }
        let hidden_elements = checked_product(n_tokens, geometry.hidden_size)?;
        for (name, values) in [
            (
                "attention captured inputs",
                forward.input_residuals.as_slice(),
            ),
            (
                "attention captured post-mixer residuals",
                forward.post_mixer_residuals.as_slice(),
            ),
            (
                "attention captured post-block residuals",
                forward.post_block_residuals.as_slice(),
            ),
            ("attention block cotangent", grad_block_output),
        ] {
            if values.len() != hidden_elements {
                return Err(ResearchError::ActivationSize {
                    name,
                    got: values.len(),
                    expected: hidden_elements,
                });
            }
        }
        if forward.hidden_size != geometry.hidden_size {
            return Err(ResearchError::ActivationSize {
                name: "attention capture hidden size",
                got: forward.hidden_size,
                expected: geometry.hidden_size,
            });
        }
        let grad_post_mixer_residuals = dense_ffn_vjp_rows_readback(
            self.model.context(),
            forward.layer,
            geometry.hidden_size,
            self.arch().intermediate_size as usize,
            &forward.post_mixer_residuals,
            &block.post_attn_norm,
            &block.ffn_gate,
            &block.ffn_up,
            &block.ffn_down,
            grad_block_output,
            n_tokens,
            match rule {
                AttnBlockVjpRule::Jacobian => DenseFfnVjpRule::Jacobian,
                AttnBlockVjpRule::Relp => DenseFfnVjpRule::Relp,
            },
        )?;
        let mixer = attn_mixer_replay_vjp_readback(
            self.model.context(),
            geometry,
            AttnMixerWeights::from(block),
            &forward.input_residuals,
            &grad_post_mixer_residuals,
            n_tokens,
            rule,
        )?;
        let values = grad_post_mixer_residuals
            .iter()
            .zip(&mixer.grad_input)
            .map(|(&identity, &branch)| identity + branch)
            .collect();
        let residual_replay_max_abs_error = forward
            .input_residuals
            .iter()
            .zip(&mixer.mixer_outputs)
            .zip(&forward.post_mixer_residuals)
            .map(|((&input, &mixer), &observed)| finite_abs_difference(input + mixer, observed))
            .fold(0.0f32, f32::max);
        Ok(ResearchAttnBlockVjp {
            layer: forward.layer,
            n_tokens,
            hidden_size: geometry.hidden_size,
            values,
            grad_post_mixer_residuals,
            replay_mixer_outputs: mixer.mixer_outputs,
            residual_replay_max_abs_error,
        })
    }

    /// Advance one fresh bounded prompt once while capturing every block's
    /// post-mixer and post-block residuals. The final RMSNorm and LM head are
    /// skipped because reference J/R-lens fitting targets a block residual. A
    /// failure after a successful token poisons the partially consumed
    /// sequence rather than exposing stitchable state.
    pub fn forward_prompt_with_workspace_capture(
        &mut self,
        token_ids: &[i32],
    ) -> Result<ResearchWorkspaceForward, ResearchError> {
        if token_ids.is_empty() {
            return Err(ResearchError::EmptyWorkspacePrompt);
        }
        if token_ids.len() > MAX_RESEARCH_WORKSPACE_TOKENS {
            return Err(ResearchError::WorkspacePromptTooLong {
                got: token_ids.len(),
                max: MAX_RESEARCH_WORKSPACE_TOKENS,
            });
        }
        if self.sequence.position() != 0 {
            return Err(ResearchError::WorkspaceCaptureRequiresFreshSequence(
                self.sequence.position(),
            ));
        }
        let arch = self.arch();
        for &token_id in token_ids {
            if token_id < 0 || token_id as u32 >= arch.vocab_size {
                return Err(MfError::BadToken(token_id, arch.vocab_size).into());
            }
        }
        self.sequence.ensure_can_append(token_ids.len())?;
        let last_position = token_ids.len() - 1;
        u32::try_from(last_position).map_err(|_| ResearchError::PositionOverflow(last_position))?;
        self.validate_workspace_weights()?;

        let n_layers = usize::try_from(arch.n_layer).map_err(|_| ResearchError::SizeOverflow)?;
        let hidden_size = arch.hidden_size as usize;
        let layer_elements = checked_product(token_ids.len(), hidden_size)?;
        let bank_elements = checked_product(n_layers, layer_elements)?;
        let capture_layers: Vec<u32> = (0..arch.n_layer).collect();
        let mut post_mixer_residuals = vec![0.0f32; bank_elements];
        let mut post_block_residuals = vec![0.0f32; bank_elements];
        for (token, &token_id) in token_ids.iter().enumerate() {
            let capture = match self
                .forward_token_with_dense_ffn_capture_no_tail(token_id, &capture_layers)
            {
                Ok(capture) => capture,
                Err(error) => {
                    let state = unsafe { self.sequence.metal_session_mut() };
                    state.poison("workspace prompt capture forward failed");
                    return Err(error);
                }
            };
            copy_workspace_token_capture(
                &mut post_mixer_residuals,
                &capture.pre_ffn_residuals,
                token,
                token_ids.len(),
                n_layers,
                hidden_size,
            )?;
            copy_workspace_token_capture(
                &mut post_block_residuals,
                &capture.post_block_residuals,
                token,
                token_ids.len(),
                n_layers,
                hidden_size,
            )?;
        }
        Ok(ResearchWorkspaceForward {
            identity: self.identity(),
            owner_token_id: self.model.owner_token_id(),
            token_ids: token_ids.to_vec(),
            n_layers: arch.n_layer,
            hidden_size,
            post_mixer_residuals,
            post_block_residuals,
        })
    }

    /// Apply one target-layer cotangent trajectory to every requested source
    /// layer using the reference current-and-future-position VJP semantics.
    /// The caller performs any source-position reduction (for example, the
    /// paper's mean over valid positions) on the returned `[K,T,H]` values.
    pub fn workspace_vjp(
        &self,
        forward: &ResearchWorkspaceForward,
        target_layer: u32,
        source_layers: &[u32],
        target_cotangent: &[f32],
        rule: WorkspaceLensRule,
    ) -> Result<ResearchWorkspaceVjp, ResearchError> {
        let (arch, n_tokens, hidden_elements) =
            self.validate_workspace_vjp_forward(forward, target_layer)?;
        if target_cotangent.len() != hidden_elements {
            return Err(ResearchError::ActivationSize {
                name: "workspace target cotangent",
                got: target_cotangent.len(),
                expected: hidden_elements,
            });
        }
        let (values, diagnostics) = compose_workspace_vjp(
            target_layer,
            source_layers,
            hidden_elements,
            target_cotangent,
            |layer, grad_output| {
                let input = forward.input_residuals(layer).ok_or(
                    ResearchError::WorkspaceSourceNotBeforeTarget {
                        source_layer: layer,
                        target_layer,
                    },
                )?;
                let post_mixer =
                    forward
                        .post_mixer_residuals(layer)
                        .ok_or(ResearchError::InvalidLayer {
                            layer,
                            n_layers: arch.n_layer,
                        })?;
                self.workspace_block_vjp(layer, input, post_mixer, grad_output, n_tokens, rule)
            },
        )?;
        Ok(ResearchWorkspaceVjp {
            target_layer,
            source_layers: source_layers.to_vec(),
            n_tokens,
            hidden_size: forward.hidden_size,
            values,
            diagnostics,
        })
    }

    /// Apply a query-major bank of target-layer cotangent trajectories while
    /// sharing each block's dense-FFN primal replay. Mixer reverse passes are
    /// currently isolated per query. Returned values use `[K,Q,T,H]` order.
    pub fn workspace_vjp_batch(
        &self,
        forward: &ResearchWorkspaceForward,
        target_layer: u32,
        source_layers: &[u32],
        target_cotangents: &[f32],
        n_query: usize,
        rule: WorkspaceLensRule,
    ) -> Result<ResearchWorkspaceVjpBatch, ResearchError> {
        if n_query == 0 {
            return Err(ResearchError::EmptyQueryBatch);
        }
        if n_query > MAX_RESEARCH_WORKSPACE_DIM_BATCH {
            return Err(ResearchError::WorkspaceQueryBatchTooLarge {
                got: n_query,
                max: MAX_RESEARCH_WORKSPACE_DIM_BATCH,
            });
        }
        let (_arch, n_tokens, hidden_elements) =
            self.validate_workspace_vjp_forward(forward, target_layer)?;
        let query_elements = checked_product(n_query, hidden_elements)?;
        if target_cotangents.len() != query_elements {
            return Err(ResearchError::ActivationSize {
                name: "workspace target cotangent query bank",
                got: target_cotangents.len(),
                expected: query_elements,
            });
        }
        let (values, diagnostics) = compose_workspace_vjp(
            target_layer,
            source_layers,
            query_elements,
            target_cotangents,
            |layer, grad_outputs| {
                let input = forward.input_residuals(layer).ok_or(
                    ResearchError::WorkspaceSourceNotBeforeTarget {
                        source_layer: layer,
                        target_layer,
                    },
                )?;
                let post_mixer =
                    forward
                        .post_mixer_residuals(layer)
                        .ok_or(ResearchError::InvalidLayer {
                            layer,
                            n_layers: forward.n_layers,
                        })?;
                self.workspace_block_vjp_batch(
                    layer,
                    input,
                    post_mixer,
                    grad_outputs,
                    n_tokens,
                    n_query,
                    rule,
                )
            },
        )?;
        Ok(ResearchWorkspaceVjpBatch {
            target_layer,
            source_layers: source_layers.to_vec(),
            n_query,
            n_tokens,
            hidden_size: forward.hidden_size,
            values,
            diagnostics,
        })
    }

    /// Fit selected rows of the reference current-and-future-position
    /// transport estimator for one captured prompt.
    ///
    /// For each output coordinate, the cotangent is one at every valid target
    /// position `skip_first..T-1` (the final prompt position is excluded). The
    /// resulting source trajectories are averaged over those same positions,
    /// with no second normalization over target positions.
    pub fn workspace_fit_rows(
        &self,
        forward: &ResearchWorkspaceForward,
        target_layer: u32,
        source_layers: &[u32],
        output_rows: &[u32],
        skip_first: usize,
        rule: WorkspaceLensRule,
    ) -> Result<ResearchWorkspaceRows, ResearchError> {
        if output_rows.is_empty() {
            return Err(ResearchError::EmptyWorkspaceOutputRows);
        }
        if output_rows.windows(2).any(|rows| rows[0] >= rows[1]) {
            return Err(ResearchError::WorkspaceOutputRowsNotStrict);
        }
        let hidden_size = forward.hidden_size();
        if let Some(&row) = output_rows.iter().find(|&&row| row as usize >= hidden_size) {
            return Err(ResearchError::WorkspaceOutputRowOutOfRange { row, hidden_size });
        }
        let valid_positions = workspace_valid_position_range(forward.n_tokens(), skip_first)?;
        let n_valid_positions = valid_positions.len();
        let hidden_elements = checked_product(forward.n_tokens(), hidden_size)?;
        let source_row_elements = checked_product(output_rows.len(), hidden_size)?;
        let output_elements = checked_product(source_layers.len(), source_row_elements)?;
        let mut values = vec![0.0f32; output_elements];
        let mut diagnostics = Vec::new();
        let mut target_cotangent = vec![0.0f32; hidden_elements];
        for (row_slot, &row) in output_rows.iter().enumerate() {
            target_cotangent.fill(0.0);
            for position in valid_positions.clone() {
                let offset = checked_product(position, hidden_size)?
                    .checked_add(row as usize)
                    .ok_or(ResearchError::SizeOverflow)?;
                target_cotangent[offset] = 1.0;
            }
            let vjp = self.workspace_vjp(
                forward,
                target_layer,
                source_layers,
                &target_cotangent,
                rule,
            )?;
            merge_workspace_diagnostics(&mut diagnostics, &vjp.diagnostics)?;
            for source_slot in 0..source_layers.len() {
                let source =
                    vjp.source_values(source_slot)
                        .ok_or(ResearchError::ActivationSize {
                            name: "workspace fitted source trajectory",
                            got: vjp.values.len(),
                            expected: checked_product(source_layers.len(), hidden_elements)?,
                        })?;
                let destination_row = checked_product(source_slot, output_rows.len())?
                    .checked_add(row_slot)
                    .ok_or(ResearchError::SizeOverflow)?;
                let destination_start = checked_product(destination_row, hidden_size)?;
                let destination_end = destination_start
                    .checked_add(hidden_size)
                    .ok_or(ResearchError::SizeOverflow)?;
                reduce_workspace_source_positions(
                    source,
                    forward.n_tokens(),
                    hidden_size,
                    valid_positions.clone(),
                    &mut values[destination_start..destination_end],
                )?;
            }
        }
        Ok(ResearchWorkspaceRows {
            target_layer,
            source_layers: source_layers.to_vec(),
            output_rows: output_rows.to_vec(),
            n_tokens: forward.n_tokens(),
            n_valid_positions,
            hidden_size,
            values,
            diagnostics,
        })
    }

    /// Fit rows in query-major execution batches while preserving the public
    /// row-shard orientation `[K,R,H]` and the reference estimator exactly.
    pub fn workspace_fit_rows_batched(
        &self,
        forward: &ResearchWorkspaceForward,
        target_layer: u32,
        source_layers: &[u32],
        output_rows: &[u32],
        skip_first: usize,
        dim_batch: usize,
        rule: WorkspaceLensRule,
    ) -> Result<ResearchWorkspaceRows, ResearchError> {
        if dim_batch == 0 {
            return Err(ResearchError::EmptyQueryBatch);
        }
        if dim_batch > MAX_RESEARCH_WORKSPACE_DIM_BATCH {
            return Err(ResearchError::WorkspaceQueryBatchTooLarge {
                got: dim_batch,
                max: MAX_RESEARCH_WORKSPACE_DIM_BATCH,
            });
        }
        if output_rows.is_empty() {
            return Err(ResearchError::EmptyWorkspaceOutputRows);
        }
        if output_rows.windows(2).any(|rows| rows[0] >= rows[1]) {
            return Err(ResearchError::WorkspaceOutputRowsNotStrict);
        }
        let hidden_size = forward.hidden_size();
        if let Some(&row) = output_rows.iter().find(|&&row| row as usize >= hidden_size) {
            return Err(ResearchError::WorkspaceOutputRowOutOfRange { row, hidden_size });
        }
        let valid_positions = workspace_valid_position_range(forward.n_tokens(), skip_first)?;
        let n_valid_positions = valid_positions.len();
        let hidden_elements = checked_product(forward.n_tokens(), hidden_size)?;
        let source_row_elements = checked_product(output_rows.len(), hidden_size)?;
        let output_elements = checked_product(source_layers.len(), source_row_elements)?;
        let mut values = vec![0.0f32; output_elements];
        let mut diagnostics = Vec::new();
        let mut first_row_slot = 0usize;
        for rows in output_rows.chunks(dim_batch) {
            let n_query = rows.len();
            let mut target_cotangents = vec![0.0f32; checked_product(n_query, hidden_elements)?];
            for (query_slot, &row) in rows.iter().enumerate() {
                let query_start = checked_product(query_slot, hidden_elements)?;
                for position in valid_positions.clone() {
                    let offset = query_start
                        .checked_add(checked_product(position, hidden_size)?)
                        .and_then(|offset| offset.checked_add(row as usize))
                        .ok_or(ResearchError::SizeOverflow)?;
                    target_cotangents[offset] = 1.0;
                }
            }
            let vjp = self.workspace_vjp_batch(
                forward,
                target_layer,
                source_layers,
                &target_cotangents,
                n_query,
                rule,
            )?;
            merge_workspace_diagnostics(&mut diagnostics, &vjp.diagnostics)?;
            for source_slot in 0..source_layers.len() {
                for query_slot in 0..n_query {
                    let source = vjp.source_query_values(source_slot, query_slot).ok_or(
                        ResearchError::ActivationSize {
                            name: "workspace fitted source trajectory query",
                            got: vjp.values.len(),
                            expected: checked_product(
                                source_layers.len(),
                                checked_product(n_query, hidden_elements)?,
                            )?,
                        },
                    )?;
                    let row_slot = first_row_slot
                        .checked_add(query_slot)
                        .ok_or(ResearchError::SizeOverflow)?;
                    let destination_row = checked_product(source_slot, output_rows.len())?
                        .checked_add(row_slot)
                        .ok_or(ResearchError::SizeOverflow)?;
                    let destination_start = checked_product(destination_row, hidden_size)?;
                    let destination_end = destination_start
                        .checked_add(hidden_size)
                        .ok_or(ResearchError::SizeOverflow)?;
                    reduce_workspace_source_positions(
                        source,
                        forward.n_tokens(),
                        hidden_size,
                        valid_positions.clone(),
                        &mut values[destination_start..destination_end],
                    )?;
                }
            }
            first_row_slot = first_row_slot
                .checked_add(n_query)
                .ok_or(ResearchError::SizeOverflow)?;
        }
        Ok(ResearchWorkspaceRows {
            target_layer,
            source_layers: source_layers.to_vec(),
            output_rows: output_rows.to_vec(),
            n_tokens: forward.n_tokens(),
            n_valid_positions,
            hidden_size,
            values,
            diagnostics,
        })
    }

    fn validate_workspace_vjp_forward(
        &self,
        forward: &ResearchWorkspaceForward,
        target_layer: u32,
    ) -> Result<(Arch, usize, usize), ResearchError> {
        if forward.identity != self.identity() {
            return Err(ResearchError::WorkspaceCaptureModelMismatch);
        }
        if forward.owner_token_id != self.model.owner_token_id() {
            return Err(ResearchError::WorkspaceCaptureOwnerMismatch);
        }
        let arch = self.arch();
        if target_layer >= arch.n_layer {
            return Err(ResearchError::InvalidLayer {
                layer: target_layer,
                n_layers: arch.n_layer,
            });
        }
        let n_tokens = forward.n_tokens();
        if n_tokens == 0 || n_tokens > MAX_RESEARCH_WORKSPACE_TOKENS {
            return Err(ResearchError::WorkspacePromptTooLong {
                got: n_tokens,
                max: MAX_RESEARCH_WORKSPACE_TOKENS,
            });
        }
        if forward.n_layers != arch.n_layer {
            return Err(ResearchError::ActivationSize {
                name: "workspace capture layer count",
                got: forward.n_layers as usize,
                expected: arch.n_layer as usize,
            });
        }
        if forward.hidden_size != arch.hidden_size as usize {
            return Err(ResearchError::ActivationSize {
                name: "workspace capture hidden size",
                got: forward.hidden_size,
                expected: arch.hidden_size as usize,
            });
        }
        let hidden_elements = checked_product(n_tokens, forward.hidden_size)?;
        let bank_elements = checked_product(arch.n_layer as usize, hidden_elements)?;
        for (name, values) in [
            (
                "workspace post-mixer residual bank",
                forward.post_mixer_residuals.as_slice(),
            ),
            (
                "workspace post-block residual bank",
                forward.post_block_residuals.as_slice(),
            ),
        ] {
            if values.len() != bank_elements {
                return Err(ResearchError::ActivationSize {
                    name,
                    got: values.len(),
                    expected: bank_elements,
                });
            }
        }
        Ok((arch, n_tokens, hidden_elements))
    }

    fn validate_workspace_weights(&self) -> Result<(), ResearchError> {
        let arch = self.arch();
        if self.model.metal_model().blocks.len() != arch.n_layer as usize {
            return Err(ResearchError::ActivationSize {
                name: "resident workspace block schedule",
                got: self.model.metal_model().blocks.len(),
                expected: arch.n_layer as usize,
            });
        }
        for (layer, block) in self.model.metal_model().blocks.iter().enumerate() {
            let layer = u32::try_from(layer).map_err(|_| ResearchError::SizeOverflow)?;
            let (post_norm, gate, up, down) = match block {
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
            };
            validate_dense_ffn_weights(
                layer,
                arch.hidden_size as usize,
                arch.intermediate_size as usize,
                post_norm,
                gate,
                up,
                down,
            )?;
            match block {
                MetalBlock::Gdn(block) => {
                    let geometry = GdnGeometry::new(layer, arch)?;
                    validate_gdn_weights(layer, block, geometry)?;
                }
                MetalBlock::Attn(block) => {
                    let geometry = AttnGeometry::new(arch)?;
                    validate_attn_weights(layer, block, geometry)?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn workspace_block_vjp(
        &self,
        layer: u32,
        input_residuals: &[f32],
        post_mixer_residuals: &[f32],
        grad_block_output: &[f32],
        n_tokens: usize,
        rule: WorkspaceLensRule,
    ) -> Result<(Vec<f32>, ResearchWorkspaceReplayDiagnostic), ResearchError> {
        let arch = self.arch();
        let hidden_size = arch.hidden_size as usize;
        let hidden_elements = checked_product(n_tokens, hidden_size)?;
        for (name, values) in [
            ("workspace block inputs", input_residuals),
            ("workspace post-mixer residuals", post_mixer_residuals),
            ("workspace block cotangent", grad_block_output),
        ] {
            if values.len() != hidden_elements {
                return Err(ResearchError::ActivationSize {
                    name,
                    got: values.len(),
                    expected: hidden_elements,
                });
            }
        }
        let block = self.model.metal_model().blocks.get(layer as usize).ok_or(
            ResearchError::InvalidLayer {
                layer,
                n_layers: arch.n_layer,
            },
        )?;
        let (post_norm, gate, up, down) = match block {
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
        };
        let grad_post_mixer = dense_ffn_vjp_rows_readback(
            self.model.context(),
            layer,
            hidden_size,
            arch.intermediate_size as usize,
            post_mixer_residuals,
            post_norm,
            gate,
            up,
            down,
            grad_block_output,
            n_tokens,
            match rule {
                WorkspaceLensRule::Jacobian => DenseFfnVjpRule::Jacobian,
                WorkspaceLensRule::Relp => DenseFfnVjpRule::Relp,
            },
        )?;
        let (mixer_outputs, grad_mixer_input, kind) = match block {
            MetalBlock::Gdn(block) => {
                let geometry = GdnGeometry::new(layer, arch)?;
                validate_gdn_weights(layer, block, geometry)?;
                let initial_conv_state = vec![0.0f32; geometry.conv_state_elements];
                let initial_recurrence_state = vec![0.0f32; geometry.state_elements];
                let replay = gdn_mixer_replay_vjp_readback(
                    self.model.context(),
                    geometry,
                    GdnMixerWeights::from(block),
                    input_residuals,
                    &initial_conv_state,
                    &initial_recurrence_state,
                    &grad_post_mixer,
                    n_tokens,
                    match rule {
                        WorkspaceLensRule::Jacobian => GdnMixerVjpRule::Jacobian,
                        WorkspaceLensRule::Relp => GdnMixerVjpRule::Relp,
                    },
                    false,
                )?;
                (
                    replay.mixer_outputs,
                    replay.grad_input,
                    ResearchWorkspaceBlockKind::Gdn,
                )
            }
            MetalBlock::Attn(block) => {
                let geometry = AttnGeometry::new(arch)?;
                validate_attn_weights(layer, block, geometry)?;
                let replay = attn_mixer_replay_vjp_readback(
                    self.model.context(),
                    geometry,
                    AttnMixerWeights::from(block),
                    input_residuals,
                    &grad_post_mixer,
                    n_tokens,
                    match rule {
                        WorkspaceLensRule::Jacobian => AttnBlockVjpRule::Jacobian,
                        WorkspaceLensRule::Relp => AttnBlockVjpRule::Relp,
                    },
                )?;
                (
                    replay.mixer_outputs,
                    replay.grad_input,
                    ResearchWorkspaceBlockKind::Attention,
                )
            }
        };
        let residual_replay_max_abs_error = input_residuals
            .iter()
            .zip(&mixer_outputs)
            .zip(post_mixer_residuals)
            .map(|((&input, &mixer), &observed)| finite_abs_difference(input + mixer, observed))
            .fold(0.0f32, f32::max);
        if grad_mixer_input.len() != grad_post_mixer.len() {
            return Err(ResearchError::ActivationSize {
                name: "workspace mixer branch cotangent",
                got: grad_mixer_input.len(),
                expected: grad_post_mixer.len(),
            });
        }
        let values = grad_post_mixer
            .iter()
            .zip(grad_mixer_input)
            .map(|(&identity, branch)| identity + branch)
            .collect();
        Ok((
            values,
            ResearchWorkspaceReplayDiagnostic {
                layer,
                kind,
                residual_replay_max_abs_error,
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn workspace_block_vjp_batch(
        &self,
        layer: u32,
        input_residuals: &[f32],
        post_mixer_residuals: &[f32],
        grad_block_outputs: &[f32],
        n_tokens: usize,
        n_query: usize,
        rule: WorkspaceLensRule,
    ) -> Result<(Vec<f32>, ResearchWorkspaceReplayDiagnostic), ResearchError> {
        if n_query == 0 {
            return Err(ResearchError::EmptyQueryBatch);
        }
        let arch = self.arch();
        let hidden_size = arch.hidden_size as usize;
        let hidden_elements = checked_product(n_tokens, hidden_size)?;
        let query_elements = checked_product(n_query, hidden_elements)?;
        for (name, values, expected) in [
            ("workspace block inputs", input_residuals, hidden_elements),
            (
                "workspace post-mixer residuals",
                post_mixer_residuals,
                hidden_elements,
            ),
            (
                "workspace block cotangent query bank",
                grad_block_outputs,
                query_elements,
            ),
        ] {
            if values.len() != expected {
                return Err(ResearchError::ActivationSize {
                    name,
                    got: values.len(),
                    expected,
                });
            }
        }
        let block = self.model.metal_model().blocks.get(layer as usize).ok_or(
            ResearchError::InvalidLayer {
                layer,
                n_layers: arch.n_layer,
            },
        )?;
        let (post_norm, gate, up, down) = match block {
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
        };
        let grad_post_mixer = dense_ffn_vjp_query_rows_readback(
            self.model.context(),
            layer,
            hidden_size,
            arch.intermediate_size as usize,
            post_mixer_residuals,
            post_norm,
            gate,
            up,
            down,
            grad_block_outputs,
            n_tokens,
            n_query,
            match rule {
                WorkspaceLensRule::Jacobian => DenseFfnVjpRule::Jacobian,
                WorkspaceLensRule::Relp => DenseFfnVjpRule::Relp,
            },
        )?;

        let mut grad_mixer_input = Vec::with_capacity(query_elements);
        let mut residual_replay_max_abs_error = 0.0f32;
        let kind = match block {
            MetalBlock::Gdn(block) => {
                let geometry = GdnGeometry::new(layer, arch)?;
                validate_gdn_weights(layer, block, geometry)?;
                let initial_conv_state = vec![0.0f32; geometry.conv_state_elements];
                let initial_recurrence_state = vec![0.0f32; geometry.state_elements];
                for grad_query in grad_post_mixer.chunks_exact(hidden_elements) {
                    let replay = gdn_mixer_replay_vjp_readback(
                        self.model.context(),
                        geometry,
                        GdnMixerWeights::from(block),
                        input_residuals,
                        &initial_conv_state,
                        &initial_recurrence_state,
                        grad_query,
                        n_tokens,
                        match rule {
                            WorkspaceLensRule::Jacobian => GdnMixerVjpRule::Jacobian,
                            WorkspaceLensRule::Relp => GdnMixerVjpRule::Relp,
                        },
                        false,
                    )?;
                    residual_replay_max_abs_error = residual_replay_max_abs_error.max(
                        input_residuals
                            .iter()
                            .zip(&replay.mixer_outputs)
                            .zip(post_mixer_residuals)
                            .map(|((&input, &mixer), &observed)| {
                                finite_abs_difference(input + mixer, observed)
                            })
                            .fold(0.0f32, f32::max),
                    );
                    if replay.grad_input.len() != hidden_elements {
                        return Err(ResearchError::ActivationSize {
                            name: "workspace GDN mixer branch cotangent",
                            got: replay.grad_input.len(),
                            expected: hidden_elements,
                        });
                    }
                    grad_mixer_input.extend(replay.grad_input);
                }
                ResearchWorkspaceBlockKind::Gdn
            }
            MetalBlock::Attn(block) => {
                let geometry = AttnGeometry::new(arch)?;
                validate_attn_weights(layer, block, geometry)?;
                for grad_query in grad_post_mixer.chunks_exact(hidden_elements) {
                    let replay = attn_mixer_replay_vjp_readback(
                        self.model.context(),
                        geometry,
                        AttnMixerWeights::from(block),
                        input_residuals,
                        grad_query,
                        n_tokens,
                        match rule {
                            WorkspaceLensRule::Jacobian => AttnBlockVjpRule::Jacobian,
                            WorkspaceLensRule::Relp => AttnBlockVjpRule::Relp,
                        },
                    )?;
                    residual_replay_max_abs_error = residual_replay_max_abs_error.max(
                        input_residuals
                            .iter()
                            .zip(&replay.mixer_outputs)
                            .zip(post_mixer_residuals)
                            .map(|((&input, &mixer), &observed)| {
                                finite_abs_difference(input + mixer, observed)
                            })
                            .fold(0.0f32, f32::max),
                    );
                    if replay.grad_input.len() != hidden_elements {
                        return Err(ResearchError::ActivationSize {
                            name: "workspace attention mixer branch cotangent",
                            got: replay.grad_input.len(),
                            expected: hidden_elements,
                        });
                    }
                    grad_mixer_input.extend(replay.grad_input);
                }
                ResearchWorkspaceBlockKind::Attention
            }
        };
        if grad_mixer_input.len() != query_elements {
            return Err(ResearchError::ActivationSize {
                name: "workspace mixer branch cotangent query bank",
                got: grad_mixer_input.len(),
                expected: query_elements,
            });
        }
        let values = grad_post_mixer
            .into_iter()
            .zip(grad_mixer_input)
            .map(|(identity, branch)| identity + branch)
            .collect();
        Ok((
            values,
            ResearchWorkspaceReplayDiagnostic {
                layer,
                kind,
                residual_replay_max_abs_error,
            },
        ))
    }

    fn resolve_attn(&self, layer: u32) -> Result<(&MetalAttnBlock, AttnGeometry), ResearchError> {
        let block = self.model.metal_model().blocks.get(layer as usize).ok_or(
            ResearchError::InvalidLayer {
                layer,
                n_layers: self.arch().n_layer,
            },
        )?;
        let MetalBlock::Attn(block) = block else {
            return Err(ResearchError::NotAttentionLayer { layer });
        };
        Ok((block, AttnGeometry::new(self.arch())?))
    }

    fn resolve_gdn(
        &self,
        layer: u32,
    ) -> Result<(&MetalGdnBlock, usize, GdnGeometry), ResearchError> {
        let block = self.model.metal_model().blocks.get(layer as usize).ok_or(
            ResearchError::InvalidLayer {
                layer,
                n_layers: self.arch().n_layer,
            },
        )?;
        let MetalBlock::Gdn(block) = block else {
            return Err(ResearchError::NotGdnLayer { layer });
        };
        let gdn_index = self.model.metal_model().blocks[..layer as usize]
            .iter()
            .filter(|block| matches!(block, MetalBlock::Gdn(_)))
            .count();
        Ok((block, gdn_index, GdnGeometry::new(layer, self.arch())?))
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

#[derive(Clone, Copy)]
struct GdnGeometry {
    hidden_size: usize,
    n_v_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    qk_elements: usize,
    v_elements: usize,
    conv_dim: usize,
    state_elements: usize,
    conv_state_elements: usize,
}

#[derive(Clone, Copy)]
struct AttnGeometry {
    hidden_size: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_rot: usize,
    q_elements: usize,
    q_full_elements: usize,
    kv_elements: usize,
    rope_theta: f32,
}

impl AttnGeometry {
    fn new(arch: Arch) -> Result<Self, ResearchError> {
        let hidden_size = arch.hidden_size as usize;
        let n_q_heads = arch.n_q_heads as usize;
        let n_kv_heads = arch.n_kv_heads as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
        if hidden_size == 0
            || n_q_heads == 0
            || n_kv_heads == 0
            || !n_q_heads.is_multiple_of(n_kv_heads)
            || head_dim == 0
            || n_rot == 0
            || n_rot > head_dim
            || !n_rot.is_multiple_of(2)
            || !arch.rope_theta.is_finite()
            || arch.rope_theta <= 0.0
        {
            return Err(ResearchError::SizeOverflow);
        }
        let q_elements = checked_product(n_q_heads, head_dim)?;
        Ok(Self {
            hidden_size,
            n_q_heads,
            n_kv_heads,
            head_dim,
            n_rot,
            q_elements,
            q_full_elements: checked_product(q_elements, 2)?,
            kv_elements: checked_product(n_kv_heads, head_dim)?,
            rope_theta: arch.rope_theta,
        })
    }
}

struct CpuCausalAttentionForward {
    attention_output: Vec<f32>,
    gated_output: Vec<f32>,
}

struct CpuCausalAttentionVjp {
    grad_q: Vec<f32>,
    grad_k: Vec<f32>,
    grad_v: Vec<f32>,
    grad_gate: Vec<f32>,
}

fn rope_neox_rows_in_place(
    values: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    start_position: u32,
    rope_theta: f32,
    transpose: bool,
) -> Result<(), ResearchError> {
    let expected = checked_product(checked_product(n_tokens, n_heads)?, head_dim)?;
    if values.len() != expected
        || n_tokens == 0
        || n_heads == 0
        || head_dim == 0
        || n_rot == 0
        || n_rot > head_dim
        || !n_rot.is_multiple_of(2)
        || !rope_theta.is_finite()
        || rope_theta <= 0.0
    {
        return Err(ResearchError::ActivationSize {
            name: "RoPE row bank",
            got: values.len(),
            expected,
        });
    }
    let half = n_rot / 2;
    for token in 0..n_tokens {
        let position = start_position
            .checked_add(u32::try_from(token).map_err(|_| ResearchError::SizeOverflow)?)
            .ok_or(ResearchError::SizeOverflow)? as f32;
        for head in 0..n_heads {
            let base = (token * n_heads + head) * head_dim;
            for index in 0..half {
                let exponent = (2 * index) as f32 / n_rot as f32;
                let angle = position / rope_theta.powf(exponent);
                let (sin, cos) = angle.sin_cos();
                let left = values[base + index];
                let right = values[base + index + half];
                if transpose {
                    values[base + index] = left * cos + right * sin;
                    values[base + index + half] = -left * sin + right * cos;
                } else {
                    values[base + index] = left * cos - right * sin;
                    values[base + index + half] = left * sin + right * cos;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::needless_range_loop)]
fn cpu_causal_gated_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    n_tokens: usize,
    geometry: AttnGeometry,
) -> Result<CpuCausalAttentionForward, ResearchError> {
    let q_total = checked_product(n_tokens, geometry.q_elements)?;
    let kv_total = checked_product(n_tokens, geometry.kv_elements)?;
    for (name, values, expected) in [
        ("attention Q", q, q_total),
        ("attention K", k, kv_total),
        ("attention V", v, kv_total),
        ("attention gate", gate, q_total),
    ] {
        if values.len() != expected {
            return Err(ResearchError::ActivationSize {
                name,
                got: values.len(),
                expected,
            });
        }
    }
    let group = geometry.n_q_heads / geometry.n_kv_heads;
    let scale = (geometry.head_dim as f32).sqrt().recip();
    let mut attention_output = vec![0.0f32; q_total];
    for token in 0..n_tokens {
        for q_head in 0..geometry.n_q_heads {
            let kv_head = q_head / group;
            let q_base = (token * geometry.n_q_heads + q_head) * geometry.head_dim;
            let mut scores = vec![0.0f32; token + 1];
            for key_token in 0..=token {
                let k_base = (key_token * geometry.n_kv_heads + kv_head) * geometry.head_dim;
                let mut score = 0.0f32;
                for index in 0..geometry.head_dim {
                    score += q[q_base + index] * k[k_base + index];
                }
                scores[key_token] = score * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0f32;
            for score in &mut scores {
                *score = (*score - max).exp();
                denominator += *score;
            }
            for score in &mut scores {
                *score /= denominator;
            }
            for key_token in 0..=token {
                let v_base = (key_token * geometry.n_kv_heads + kv_head) * geometry.head_dim;
                let probability = scores[key_token];
                for index in 0..geometry.head_dim {
                    attention_output[q_base + index] += probability * v[v_base + index];
                }
            }
        }
    }
    let gated_output = attention_output
        .iter()
        .zip(gate)
        .map(|(&attention, &gate)| attention * (1.0 + (-gate).exp()).recip())
        .collect();
    Ok(CpuCausalAttentionForward {
        attention_output,
        gated_output,
    })
}

#[allow(clippy::needless_range_loop)]
fn cpu_causal_gated_attention_vjp(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    grad_gated_output: &[f32],
    n_tokens: usize,
    geometry: AttnGeometry,
) -> Result<CpuCausalAttentionVjp, ResearchError> {
    let forward = cpu_causal_gated_attention_forward(q, k, v, gate, n_tokens, geometry)?;
    let q_total = checked_product(n_tokens, geometry.q_elements)?;
    let kv_total = checked_product(n_tokens, geometry.kv_elements)?;
    if grad_gated_output.len() != q_total {
        return Err(ResearchError::ActivationSize {
            name: "gated attention cotangent",
            got: grad_gated_output.len(),
            expected: q_total,
        });
    }
    let group = geometry.n_q_heads / geometry.n_kv_heads;
    let scale = (geometry.head_dim as f32).sqrt().recip();
    let mut grad_q = vec![0.0f32; q_total];
    let mut grad_k = vec![0.0f32; kv_total];
    let mut grad_v = vec![0.0f32; kv_total];
    let mut grad_gate = vec![0.0f32; q_total];
    for token in 0..n_tokens {
        for q_head in 0..geometry.n_q_heads {
            let kv_head = q_head / group;
            let q_base = (token * geometry.n_q_heads + q_head) * geometry.head_dim;
            let mut grad_attention = vec![0.0f32; geometry.head_dim];
            for index in 0..geometry.head_dim {
                let sigmoid = (1.0 + (-gate[q_base + index]).exp()).recip();
                grad_attention[index] = grad_gated_output[q_base + index] * sigmoid;
                grad_gate[q_base + index] = grad_gated_output[q_base + index]
                    * forward.attention_output[q_base + index]
                    * sigmoid
                    * (1.0 - sigmoid);
            }
            let mut scores = vec![0.0f32; token + 1];
            for key_token in 0..=token {
                let k_base = (key_token * geometry.n_kv_heads + kv_head) * geometry.head_dim;
                let mut score = 0.0f32;
                for index in 0..geometry.head_dim {
                    score += q[q_base + index] * k[k_base + index];
                }
                scores[key_token] = score * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0f32;
            for score in &mut scores {
                *score = (*score - max).exp();
                denominator += *score;
            }
            for score in &mut scores {
                *score /= denominator;
            }
            let mut grad_probability = vec![0.0f32; token + 1];
            for key_token in 0..=token {
                let v_base = (key_token * geometry.n_kv_heads + kv_head) * geometry.head_dim;
                for index in 0..geometry.head_dim {
                    grad_probability[key_token] += grad_attention[index] * v[v_base + index];
                    grad_v[v_base + index] += scores[key_token] * grad_attention[index];
                }
            }
            let probability_dot = scores
                .iter()
                .zip(&grad_probability)
                .map(|(&probability, &gradient)| probability * gradient)
                .sum::<f32>();
            for key_token in 0..=token {
                let grad_score =
                    scores[key_token] * (grad_probability[key_token] - probability_dot);
                let k_base = (key_token * geometry.n_kv_heads + kv_head) * geometry.head_dim;
                for index in 0..geometry.head_dim {
                    grad_q[q_base + index] += scale * grad_score * k[k_base + index];
                    grad_k[k_base + index] += scale * grad_score * q[q_base + index];
                }
            }
        }
    }
    Ok(CpuCausalAttentionVjp {
        grad_q,
        grad_k,
        grad_v,
        grad_gate,
    })
}

#[derive(Clone, Copy)]
struct AttnMixerWeights<'a> {
    attn_norm: &'a MetalTensor,
    q: &'a MetalTensor,
    k: &'a MetalTensor,
    v: &'a MetalTensor,
    o: &'a MetalTensor,
    q_norm: &'a MetalTensor,
    k_norm: &'a MetalTensor,
}

impl<'a> From<&'a MetalAttnBlock> for AttnMixerWeights<'a> {
    fn from(block: &'a MetalAttnBlock) -> Self {
        Self {
            attn_norm: &block.attn_norm,
            q: &block.q,
            k: &block.k,
            v: &block.v,
            o: &block.o,
            q_norm: &block.q_norm,
            k_norm: &block.k_norm,
        }
    }
}

fn validate_attn_weights(
    layer: u32,
    block: &MetalAttnBlock,
    geometry: AttnGeometry,
) -> Result<(), ResearchError> {
    let weights = AttnMixerWeights::from(block);
    for (role, weight, expected) in [
        (
            LinearRole::AttentionQAndGate,
            weights.q,
            [geometry.hidden_size, geometry.q_full_elements],
        ),
        (
            LinearRole::AttentionK,
            weights.k,
            [geometry.hidden_size, geometry.kv_elements],
        ),
        (
            LinearRole::AttentionV,
            weights.v,
            [geometry.hidden_size, geometry.kv_elements],
        ),
        (
            LinearRole::AttentionOut,
            weights.o,
            [geometry.q_elements, geometry.hidden_size],
        ),
    ] {
        let id = ResearchLinear::Layer { index: layer, role };
        let got = linear_shape(id, weight)?;
        if got != expected {
            return Err(ResearchError::InvalidAttnLinearShape {
                layer,
                role,
                got,
                expected,
            });
        }
        validate_vjp_dtype(id, weight)?;
    }
    for (name, tensor, expected_elements) in [
        ("pre-mixer norm", weights.attn_norm, geometry.hidden_size),
        ("Q norm", weights.q_norm, geometry.head_dim),
        ("K norm", weights.k_norm, geometry.head_dim),
    ] {
        if tensor.dtype != GgmlType::F32 || tensor.n_elements() as usize != expected_elements {
            return Err(ResearchError::InvalidAttnTensor {
                layer,
                name,
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                expected_elements,
            });
        }
    }
    Ok(())
}

struct AttnReplayFrontTensors {
    input: MetalTensor,
    normalized: MetalTensor,
    q_full: MetalTensor,
    q_raw: MetalTensor,
    gate: MetalTensor,
    k_raw: MetalTensor,
    v: MetalTensor,
    q_normed: MetalTensor,
    k_normed: MetalTensor,
}

impl AttnReplayFrontTensors {
    fn new(
        context: &MetalContext,
        geometry: AttnGeometry,
        input: &[f32],
        n_tokens: usize,
    ) -> Result<Self, ResearchError> {
        let hidden_total = checked_product(n_tokens, geometry.hidden_size)?;
        if input.len() != hidden_total {
            return Err(ResearchError::ActivationSize {
                name: "attention replay input",
                got: input.len(),
                expected: hidden_total,
            });
        }
        let q_total = checked_product(n_tokens, geometry.q_elements)?;
        let kv_total = checked_product(n_tokens, geometry.kv_elements)?;
        let q_full_total = checked_product(n_tokens, geometry.q_full_elements)?;
        let hidden_shape = row_shape(geometry.hidden_size, n_tokens)?;
        Ok(Self {
            input: MetalTensor::from_bytes(
                context,
                bytemuck::cast_slice(input),
                hidden_shape.clone(),
                GgmlType::F32,
            )?,
            normalized: MetalTensor::zeros_f32(context, hidden_shape)?,
            q_full: flat_f32(context, q_full_total)?,
            q_raw: flat_f32(context, q_total)?,
            gate: flat_f32(context, q_total)?,
            k_raw: flat_f32(context, kv_total)?,
            v: flat_f32(context, kv_total)?,
            q_normed: flat_f32(context, q_total)?,
            k_normed: flat_f32(context, kv_total)?,
        })
    }

    fn encode(
        &self,
        context: &MetalContext,
        encoder: &KernelEncoder,
        geometry: AttnGeometry,
        weights: AttnMixerWeights<'_>,
        n_tokens: usize,
    ) -> Result<(), ResearchError> {
        encode_rms_norm_mul_rows_f32(
            context,
            encoder,
            &self.input,
            weights.attn_norm,
            &self.normalized,
            n_tokens,
            geometry.hidden_size,
            RMS_EPS,
        )?;
        for token in 0..n_tokens {
            let hidden = row_view(&self.normalized, token, geometry.hidden_size);
            let q_full = row_view(&self.q_full, token, geometry.q_full_elements);
            let q = row_view(&self.q_raw, token, geometry.q_elements);
            let gate = row_view(&self.gate, token, geometry.q_elements);
            let k = row_view(&self.k_raw, token, geometry.kv_elements);
            let v = row_view(&self.v, token, geometry.kv_elements);
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.q,
                &hidden,
                &q_full,
                geometry.hidden_size,
                geometry.q_full_elements,
            )?;
            encode_split_q_gate_f32(
                context,
                encoder,
                &q_full,
                &q,
                &gate,
                geometry.n_q_heads,
                geometry.head_dim,
            )?;
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.k,
                &hidden,
                &k,
                geometry.hidden_size,
                geometry.kv_elements,
            )?;
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.v,
                &hidden,
                &v,
                geometry.hidden_size,
                geometry.kv_elements,
            )?;
        }
        encode_rms_norm_batched_f32(
            context,
            encoder,
            &self.q_raw,
            weights.q_norm,
            &self.q_normed,
            checked_product(n_tokens, geometry.n_q_heads)?,
            geometry.head_dim,
            RMS_EPS,
        )?;
        encode_rms_norm_batched_f32(
            context,
            encoder,
            &self.k_raw,
            weights.k_norm,
            &self.k_normed,
            checked_product(n_tokens, geometry.n_kv_heads)?,
            geometry.head_dim,
            RMS_EPS,
        )?;
        Ok(())
    }
}

fn cpu_weighted_rms_vjp_rows(
    x: &[f32],
    weight: &[f32],
    grad_output: &[f32],
    n_rows: usize,
    width: usize,
    detach_scale: bool,
) -> Result<Vec<f32>, ResearchError> {
    let expected = checked_product(n_rows, width)?;
    if x.len() != expected || grad_output.len() != expected || weight.len() != width {
        return Err(ResearchError::ActivationSize {
            name: "CPU weighted RMSNorm VJP",
            got: x.len().min(grad_output.len()),
            expected,
        });
    }
    let mut result = vec![0.0f32; expected];
    for row in 0..n_rows {
        let base = row * width;
        let sum_squares = x[base..base + width]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let scale = (sum_squares / width as f32 + RMS_EPS).sqrt().recip();
        let dot = (0..width)
            .map(|index| x[base + index] * grad_output[base + index] * weight[index])
            .sum::<f32>();
        let correction = dot * scale * scale * scale / width as f32;
        for index in 0..width {
            let direct = grad_output[base + index] * weight[index] * scale;
            result[base + index] = if detach_scale {
                direct
            } else {
                direct - x[base + index] * correction
            };
        }
    }
    Ok(result)
}

struct AttnMixerVjpReadback {
    mixer_outputs: Vec<f32>,
    grad_input: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn attn_mixer_replay_vjp_readback(
    context: &MetalContext,
    geometry: AttnGeometry,
    weights: AttnMixerWeights<'_>,
    input: &[f32],
    grad_mixer_output: &[f32],
    n_tokens: usize,
    rule: AttnBlockVjpRule,
) -> Result<AttnMixerVjpReadback, ResearchError> {
    if n_tokens == 0 || n_tokens > MAX_RESEARCH_ATTN_TOKENS {
        return Err(ResearchError::AttnPromptTooLong {
            got: n_tokens,
            max: MAX_RESEARCH_ATTN_TOKENS,
        });
    }
    let hidden_total = checked_product(n_tokens, geometry.hidden_size)?;
    let q_total = checked_product(n_tokens, geometry.q_elements)?;
    let kv_total = checked_product(n_tokens, geometry.kv_elements)?;
    if input.len() != hidden_total || grad_mixer_output.len() != hidden_total {
        return Err(ResearchError::ActivationSize {
            name: "attention mixer input/cotangent",
            got: input.len().min(grad_mixer_output.len()),
            expected: hidden_total,
        });
    }
    let front = AttnReplayFrontTensors::new(context, geometry, input, n_tokens)?;
    let command = context
        .queue
        .commandBuffer()
        .ok_or(ResearchError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = front.encode(context, &encoder, geometry, weights, n_tokens);
    encoder.end();
    encode_result?;
    command.commit();
    command.waitUntilCompleted();
    validate_completed_command(&command)?;

    let q_raw = read_f32(&front.q_raw, q_total);
    let k_raw = read_f32(&front.k_raw, kv_total);
    let gate = read_f32(&front.gate, q_total);
    let v = read_f32(&front.v, kv_total);
    let mut q = read_f32(&front.q_normed, q_total);
    let mut k = read_f32(&front.k_normed, kv_total);
    rope_neox_rows_in_place(
        &mut q,
        n_tokens,
        geometry.n_q_heads,
        geometry.head_dim,
        geometry.n_rot,
        0,
        geometry.rope_theta,
        false,
    )?;
    rope_neox_rows_in_place(
        &mut k,
        n_tokens,
        geometry.n_kv_heads,
        geometry.head_dim,
        geometry.n_rot,
        0,
        geometry.rope_theta,
        false,
    )?;
    let attention = cpu_causal_gated_attention_forward(&q, &k, &v, &gate, n_tokens, geometry)?;

    let gated_output = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(&attention.gated_output),
        row_shape(geometry.q_elements, n_tokens)?,
        GgmlType::F32,
    )?;
    let mixer_output = MetalTensor::zeros_f32(context, row_shape(geometry.hidden_size, n_tokens)?)?;
    let grad_mixer = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(grad_mixer_output),
        row_shape(geometry.hidden_size, n_tokens)?,
        GgmlType::F32,
    )?;
    let grad_gated = MetalTensor::zeros_f32(context, row_shape(geometry.q_elements, n_tokens)?)?;
    let command = context
        .queue
        .commandBuffer()
        .ok_or(ResearchError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), ResearchError> {
        for token in 0..n_tokens {
            let gated = row_view(&gated_output, token, geometry.q_elements);
            let mixer = row_view(&mixer_output, token, geometry.hidden_size);
            encode_mat_vec_dispatch(
                context,
                &encoder,
                weights.o,
                &gated,
                &mixer,
                geometry.q_elements,
                geometry.hidden_size,
            )?;
        }
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            weights.o,
            &grad_mixer,
            &grad_gated,
            geometry.q_elements,
            geometry.hidden_size,
            n_tokens,
        )?;
        Ok(())
    })();
    encoder.end();
    encode_result?;
    command.commit();
    command.waitUntilCompleted();
    validate_completed_command(&command)?;
    let mixer_outputs = read_f32(&mixer_output, hidden_total);
    let grad_gated = read_f32(&grad_gated, q_total);

    let mut attention_vjp =
        cpu_causal_gated_attention_vjp(&q, &k, &v, &gate, &grad_gated, n_tokens, geometry)?;
    rope_neox_rows_in_place(
        &mut attention_vjp.grad_q,
        n_tokens,
        geometry.n_q_heads,
        geometry.head_dim,
        geometry.n_rot,
        0,
        geometry.rope_theta,
        true,
    )?;
    rope_neox_rows_in_place(
        &mut attention_vjp.grad_k,
        n_tokens,
        geometry.n_kv_heads,
        geometry.head_dim,
        geometry.n_rot,
        0,
        geometry.rope_theta,
        true,
    )?;
    let q_norm_weight = read_f32(weights.q_norm, geometry.head_dim);
    let k_norm_weight = read_f32(weights.k_norm, geometry.head_dim);
    let grad_q_raw = cpu_weighted_rms_vjp_rows(
        &q_raw,
        &q_norm_weight,
        &attention_vjp.grad_q,
        checked_product(n_tokens, geometry.n_q_heads)?,
        geometry.head_dim,
        false,
    )?;
    let grad_k_raw = cpu_weighted_rms_vjp_rows(
        &k_raw,
        &k_norm_weight,
        &attention_vjp.grad_k,
        checked_product(n_tokens, geometry.n_kv_heads)?,
        geometry.head_dim,
        false,
    )?;
    let mut grad_q_full = vec![0.0f32; checked_product(n_tokens, geometry.q_full_elements)?];
    for token in 0..n_tokens {
        for head in 0..geometry.n_q_heads {
            let source = (token * geometry.n_q_heads + head) * geometry.head_dim;
            let destination = (token * geometry.n_q_heads + head) * 2 * geometry.head_dim;
            grad_q_full[destination..destination + geometry.head_dim]
                .copy_from_slice(&grad_q_raw[source..source + geometry.head_dim]);
            grad_q_full[destination + geometry.head_dim..destination + 2 * geometry.head_dim]
                .copy_from_slice(&attention_vjp.grad_gate[source..source + geometry.head_dim]);
        }
    }

    let grad_q_full = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(&grad_q_full),
        row_shape(geometry.q_full_elements, n_tokens)?,
        GgmlType::F32,
    )?;
    let grad_k = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(&grad_k_raw),
        row_shape(geometry.kv_elements, n_tokens)?,
        GgmlType::F32,
    )?;
    let grad_v = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(&attention_vjp.grad_v),
        row_shape(geometry.kv_elements, n_tokens)?,
        GgmlType::F32,
    )?;
    let hidden_shape = row_shape(geometry.hidden_size, n_tokens)?;
    let grad_hidden_q = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_k = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_v = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_qk = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(context, hidden_shape)?;
    let command = context
        .queue
        .commandBuffer()
        .ok_or(ResearchError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), ResearchError> {
        for (weight, grad_output, grad_hidden, n_out) in [
            (
                weights.q,
                &grad_q_full,
                &grad_hidden_q,
                geometry.q_full_elements,
            ),
            (weights.k, &grad_k, &grad_hidden_k, geometry.kv_elements),
            (weights.v, &grad_v, &grad_hidden_v, geometry.kv_elements),
        ] {
            encode_frozen_linear_vjp_f32(
                context,
                &encoder,
                weight,
                grad_output,
                grad_hidden,
                geometry.hidden_size,
                n_out,
                n_tokens,
            )?;
        }
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_q,
            &grad_hidden_k,
            &grad_hidden_qk,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_qk,
            &grad_hidden_v,
            &grad_hidden,
        )?;
        encode_rms_norm_mul_vjp_rows_f32(
            context,
            &encoder,
            &front.input,
            weights.attn_norm,
            &grad_hidden,
            &grad_input,
            n_tokens,
            geometry.hidden_size,
            RMS_EPS,
            match rule {
                AttnBlockVjpRule::Jacobian => RmsNormVjpRule::Jacobian,
                AttnBlockVjpRule::Relp => RmsNormVjpRule::RelpDetachedScale,
            },
        )?;
        Ok(())
    })();
    encoder.end();
    encode_result?;
    command.commit();
    command.waitUntilCompleted();
    validate_completed_command(&command)?;
    Ok(AttnMixerVjpReadback {
        mixer_outputs,
        grad_input: read_f32(&grad_input, hidden_total),
    })
}

impl GdnGeometry {
    fn new(layer: u32, arch: Arch) -> Result<Self, ResearchError> {
        let hidden_size = arch.hidden_size as usize;
        let n_v_heads = arch.gdn_n_v_heads as usize;
        let n_k_heads = arch.gdn_n_k_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        if hidden_size == 0
            || head_dim != 128
            || n_v_heads == 0
            || n_k_heads == 0
            || !n_v_heads.is_multiple_of(n_k_heads)
        {
            return Err(ResearchError::UnsupportedGdnGeometry {
                layer,
                n_v: n_v_heads,
                n_k: n_k_heads,
                head_dim,
            });
        }
        let qk_elements = n_k_heads
            .checked_mul(head_dim)
            .ok_or(ResearchError::SizeOverflow)?;
        let v_elements = n_v_heads
            .checked_mul(head_dim)
            .ok_or(ResearchError::SizeOverflow)?;
        let conv_dim = qk_elements
            .checked_mul(2)
            .and_then(|value| value.checked_add(v_elements))
            .ok_or(ResearchError::SizeOverflow)?;
        let state_elements = v_elements
            .checked_mul(head_dim)
            .ok_or(ResearchError::SizeOverflow)?;
        let conv_state_elements = conv_dim.checked_mul(3).ok_or(ResearchError::SizeOverflow)?;
        Ok(Self {
            hidden_size,
            n_v_heads,
            n_k_heads,
            head_dim,
            qk_elements,
            v_elements,
            conv_dim,
            state_elements,
            conv_state_elements,
        })
    }
}

#[derive(Clone, Copy)]
struct GdnMixerWeights<'a> {
    attn_norm: &'a MetalTensor,
    in_proj_qkv: &'a MetalTensor,
    in_proj_z: &'a MetalTensor,
    beta_proj: &'a MetalTensor,
    alpha_proj: &'a MetalTensor,
    a_log: &'a MetalTensor,
    dt_bias: &'a MetalTensor,
    conv1d: &'a MetalTensor,
    norm: &'a MetalTensor,
    out_proj: &'a MetalTensor,
}

impl<'a> From<&'a MetalGdnBlock> for GdnMixerWeights<'a> {
    fn from(block: &'a MetalGdnBlock) -> Self {
        Self {
            attn_norm: &block.attn_norm,
            in_proj_qkv: &block.in_proj_qkv,
            in_proj_z: &block.in_proj_z,
            beta_proj: &block.beta_proj,
            alpha_proj: &block.alpha_proj,
            a_log: &block.a_log,
            dt_bias: &block.dt_bias,
            conv1d: &block.conv1d,
            norm: &block.norm,
            out_proj: &block.out_proj,
        }
    }
}

fn validate_gdn_weights(
    layer: u32,
    block: &MetalGdnBlock,
    geometry: GdnGeometry,
) -> Result<(), ResearchError> {
    let weights = GdnMixerWeights::from(block);
    for (role, weight, expected) in [
        (
            LinearRole::GdnQkv,
            weights.in_proj_qkv,
            [geometry.hidden_size, geometry.conv_dim],
        ),
        (
            LinearRole::GdnZ,
            weights.in_proj_z,
            [geometry.hidden_size, geometry.v_elements],
        ),
        (
            LinearRole::GdnBeta,
            weights.beta_proj,
            [geometry.hidden_size, geometry.n_v_heads],
        ),
        (
            LinearRole::GdnAlpha,
            weights.alpha_proj,
            [geometry.hidden_size, geometry.n_v_heads],
        ),
        (
            LinearRole::GdnOut,
            weights.out_proj,
            [geometry.v_elements, geometry.hidden_size],
        ),
    ] {
        let id = ResearchLinear::Layer { index: layer, role };
        let got = linear_shape(id, weight)?;
        if got != expected {
            return Err(ResearchError::InvalidGdnLinearShape {
                layer,
                role,
                got,
                expected,
            });
        }
        validate_vjp_dtype(id, weight)?;
    }
    for (name, tensor, expected_elements) in [
        ("pre-mixer norm", weights.attn_norm, geometry.hidden_size),
        ("A log", weights.a_log, geometry.n_v_heads),
        ("timestep bias", weights.dt_bias, geometry.n_v_heads),
        (
            "conv1d",
            weights.conv1d,
            geometry
                .conv_dim
                .checked_mul(4)
                .ok_or(ResearchError::SizeOverflow)?,
        ),
        ("internal norm", weights.norm, geometry.head_dim),
    ] {
        if tensor.dtype != GgmlType::F32 || tensor.n_elements() as usize != expected_elements {
            return Err(ResearchError::InvalidGdnTensor {
                layer,
                name,
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                expected_elements,
            });
        }
    }
    Ok(())
}

struct GdnReplayTensors {
    input: MetalTensor,
    normalized: MetalTensor,
    qkv_source: MetalTensor,
    z: MetalTensor,
    beta_source: MetalTensor,
    beta: MetalTensor,
    alpha_source: MetalTensor,
    decay: MetalTensor,
    initial_conv_state: MetalTensor,
    conv_state: MetalTensor,
    conv_checkpoints: MetalTensor,
    q_raw: MetalTensor,
    k_raw: MetalTensor,
    v: MetalTensor,
    q: MetalTensor,
    k: MetalTensor,
    initial_recurrence_state: MetalTensor,
    recurrence_state: MetalTensor,
    recurrence_checkpoints: MetalTensor,
    recurrence_output: MetalTensor,
    normed: MetalTensor,
    mixer_output: MetalTensor,
}

impl GdnReplayTensors {
    fn new(
        context: &MetalContext,
        geometry: GdnGeometry,
        input: &[f32],
        initial_conv_state: &[f32],
        initial_recurrence_state: &[f32],
        n_tokens: usize,
    ) -> Result<Self, ResearchError> {
        let hidden_elements = checked_product(n_tokens, geometry.hidden_size)?;
        let qkv_elements = checked_product(n_tokens, geometry.conv_dim)?;
        let qk_elements = checked_product(n_tokens, geometry.qk_elements)?;
        let v_elements = checked_product(n_tokens, geometry.v_elements)?;
        let scalar_elements = checked_product(n_tokens, geometry.n_v_heads)?;
        let conv_checkpoint_elements = checked_product(n_tokens, geometry.conv_state_elements)?;
        let recurrence_checkpoint_elements = checked_product(n_tokens, geometry.state_elements)?;
        for (name, values, expected) in [
            ("GDN replay input", input, hidden_elements),
            (
                "GDN initial conv state",
                initial_conv_state,
                geometry.conv_state_elements,
            ),
            (
                "GDN initial recurrence state",
                initial_recurrence_state,
                geometry.state_elements,
            ),
        ] {
            if values.len() != expected {
                return Err(ResearchError::ActivationSize {
                    name,
                    got: values.len(),
                    expected,
                });
            }
        }
        let hidden_shape = row_shape(geometry.hidden_size, n_tokens)?;
        Ok(Self {
            input: MetalTensor::from_bytes(
                context,
                bytemuck::cast_slice(input),
                hidden_shape.clone(),
                GgmlType::F32,
            )?,
            normalized: MetalTensor::zeros_f32(context, hidden_shape.clone())?,
            qkv_source: flat_f32(context, qkv_elements)?,
            z: flat_f32(context, v_elements)?,
            beta_source: flat_f32(context, scalar_elements)?,
            beta: flat_f32(context, scalar_elements)?,
            alpha_source: flat_f32(context, scalar_elements)?,
            decay: flat_f32(context, scalar_elements)?,
            initial_conv_state: f32_from_slice(context, initial_conv_state)?,
            conv_state: f32_from_slice(context, initial_conv_state)?,
            conv_checkpoints: flat_f32(context, conv_checkpoint_elements)?,
            q_raw: flat_f32(context, qk_elements)?,
            k_raw: flat_f32(context, qk_elements)?,
            v: flat_f32(context, v_elements)?,
            q: flat_f32(context, qk_elements)?,
            k: flat_f32(context, qk_elements)?,
            initial_recurrence_state: f32_from_slice(context, initial_recurrence_state)?,
            recurrence_state: f32_from_slice(context, initial_recurrence_state)?,
            recurrence_checkpoints: flat_f32(context, recurrence_checkpoint_elements)?,
            recurrence_output: flat_f32(context, v_elements)?,
            normed: flat_f32(context, v_elements)?,
            mixer_output: flat_f32(context, hidden_elements)?,
        })
    }

    fn encode_forward(
        &self,
        context: &MetalContext,
        encoder: &KernelEncoder,
        geometry: GdnGeometry,
        weights: GdnMixerWeights<'_>,
        n_tokens: usize,
    ) -> Result<(), ResearchError> {
        encode_rms_norm_mul_rows_f32(
            context,
            encoder,
            &self.input,
            weights.attn_norm,
            &self.normalized,
            n_tokens,
            geometry.hidden_size,
            RMS_EPS,
        )?;
        for token in 0..n_tokens {
            let hidden = row_view(&self.normalized, token, geometry.hidden_size);
            let qkv = row_view(&self.qkv_source, token, geometry.conv_dim);
            let z = row_view(&self.z, token, geometry.v_elements);
            let beta = row_view(&self.beta_source, token, geometry.n_v_heads);
            let alpha = row_view(&self.alpha_source, token, geometry.n_v_heads);
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.in_proj_qkv,
                &hidden,
                &qkv,
                geometry.hidden_size,
                geometry.conv_dim,
            )?;
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.in_proj_z,
                &hidden,
                &z,
                geometry.hidden_size,
                geometry.v_elements,
            )?;
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.beta_proj,
                &hidden,
                &beta,
                geometry.hidden_size,
                geometry.n_v_heads,
            )?;
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.alpha_proj,
                &hidden,
                &alpha,
                geometry.hidden_size,
                geometry.n_v_heads,
            )?;
        }
        encode_sigmoid_f32(context, encoder, &self.beta_source, &self.beta)?;
        encode_gdn_decay_chain_batched_f32(
            context,
            encoder,
            &self.alpha_source,
            weights.dt_bias,
            weights.a_log,
            &self.decay,
            n_tokens,
            geometry.n_v_heads,
        )?;
        encode_gdn_prep_packed_ckpt_f32(
            context,
            encoder,
            &self.qkv_source,
            &self.conv_state,
            weights.conv1d,
            &self.q_raw,
            &self.k_raw,
            &self.v,
            &self.conv_checkpoints,
            n_tokens,
            n_tokens,
            geometry.n_k_heads,
            geometry.n_v_heads,
            geometry.head_dim,
        )?;
        encode_l2_norm_batched_f32(
            context,
            encoder,
            &self.q_raw,
            &self.q,
            checked_product(n_tokens, geometry.n_k_heads)?,
            geometry.head_dim,
            RMS_EPS,
        )?;
        encode_l2_norm_batched_f32(
            context,
            encoder,
            &self.k_raw,
            &self.k,
            checked_product(n_tokens, geometry.n_k_heads)?,
            geometry.head_dim,
            RMS_EPS,
        )?;
        encode_gdn_step_decay_packed_ckpt_f32(
            context,
            encoder,
            &self.q,
            &self.k,
            &self.v,
            &self.decay,
            &self.beta,
            &self.recurrence_state,
            &self.recurrence_output,
            &self.recurrence_checkpoints,
            n_tokens,
            n_tokens,
            geometry.n_v_heads,
            geometry.n_k_heads,
            geometry.head_dim,
        )?;
        encode_rmsnorm_gated_f32(
            context,
            encoder,
            &self.recurrence_output,
            weights.norm,
            &self.z,
            &self.normed,
            checked_product(n_tokens, geometry.n_v_heads)?,
            geometry.head_dim,
            RMS_EPS * geometry.head_dim as f32,
        )?;
        for token in 0..n_tokens {
            let normed = row_view(&self.normed, token, geometry.v_elements);
            let mixer = row_view(&self.mixer_output, token, geometry.hidden_size);
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.out_proj,
                &normed,
                &mixer,
                geometry.v_elements,
                geometry.hidden_size,
            )?;
        }
        Ok(())
    }
}

struct GdnReplayVjpReadback {
    mixer_outputs: Vec<f32>,
    final_conv_state: Vec<f32>,
    final_recurrence_state: Vec<f32>,
    grad_input: Vec<f32>,
    grad_initial_conv_state: Vec<f32>,
    grad_initial_recurrence_state: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn gdn_mixer_replay_vjp_readback(
    context: &MetalContext,
    geometry: GdnGeometry,
    weights: GdnMixerWeights<'_>,
    input: &[f32],
    initial_conv_state: &[f32],
    initial_recurrence_state: &[f32],
    grad_mixer_output: &[f32],
    n_tokens: usize,
    rule: GdnMixerVjpRule,
    read_state_diagnostics: bool,
) -> Result<GdnReplayVjpReadback, ResearchError> {
    if n_tokens == 0 || n_tokens > MAX_RESEARCH_GDN_TOKENS {
        return Err(ResearchError::GdnPromptTooLong {
            got: n_tokens,
            max: MAX_RESEARCH_GDN_TOKENS,
        });
    }
    let hidden_elements = checked_product(n_tokens, geometry.hidden_size)?;
    let qkv_elements = checked_product(n_tokens, geometry.conv_dim)?;
    let qk_elements = checked_product(n_tokens, geometry.qk_elements)?;
    let v_elements = checked_product(n_tokens, geometry.v_elements)?;
    let scalar_elements = checked_product(n_tokens, geometry.n_v_heads)?;
    if grad_mixer_output.len() != hidden_elements {
        return Err(ResearchError::ActivationSize {
            name: "GDN mixer cotangent",
            got: grad_mixer_output.len(),
            expected: hidden_elements,
        });
    }
    let replay = GdnReplayTensors::new(
        context,
        geometry,
        input,
        initial_conv_state,
        initial_recurrence_state,
        n_tokens,
    )?;
    let grad_mixer = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(grad_mixer_output),
        row_shape(geometry.hidden_size, n_tokens)?,
        GgmlType::F32,
    )?;
    let grad_normed = MetalTensor::zeros_f32(context, row_shape(geometry.v_elements, n_tokens)?)?;
    let grad_recurrence_output = flat_f32(context, v_elements)?;
    let grad_z = flat_f32(context, v_elements)?;
    let grad_q = flat_f32(context, qk_elements)?;
    let grad_k = flat_f32(context, qk_elements)?;
    let grad_v = flat_f32(context, v_elements)?;
    let grad_decay = flat_f32(context, scalar_elements)?;
    let grad_beta = flat_f32(context, scalar_elements)?;
    let zero_final_recurrence_state = flat_f32(context, geometry.state_elements)?;
    let grad_initial_recurrence_state = flat_f32(context, geometry.state_elements)?;
    let recurrence_state_scratch_a = flat_f32(context, geometry.state_elements)?;
    let recurrence_state_scratch_b = flat_f32(context, geometry.state_elements)?;
    let correction_scratch = flat_f32(context, geometry.v_elements)?;
    let residual_scratch = flat_f32(context, geometry.v_elements)?;
    let grad_q_raw = flat_f32(context, qk_elements)?;
    let grad_k_raw = flat_f32(context, qk_elements)?;
    let grad_qkv = flat_f32(context, qkv_elements)?;
    let zero_final_conv_state = flat_f32(context, geometry.conv_state_elements)?;
    let grad_initial_conv_state = flat_f32(context, geometry.conv_state_elements)?;
    let conv_state_scratch_a = flat_f32(context, geometry.conv_state_elements)?;
    let conv_state_scratch_b = flat_f32(context, geometry.conv_state_elements)?;
    let grad_alpha_source = flat_f32(context, scalar_elements)?;
    let grad_beta_source = flat_f32(context, scalar_elements)?;
    let hidden_shape = row_shape(geometry.hidden_size, n_tokens)?;
    let grad_hidden_qkv = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_z = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_alpha = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_beta = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_sum_a = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_sum_b = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(context, hidden_shape)?;

    let command = context
        .queue
        .commandBuffer()
        .ok_or(ResearchError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), ResearchError> {
        replay.encode_forward(context, &encoder, geometry, weights, n_tokens)?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            weights.out_proj,
            &grad_mixer,
            &grad_normed,
            geometry.v_elements,
            geometry.hidden_size,
            n_tokens,
        )?;
        let grad_normed_flat = grad_normed.view_subrange(0, vec![v_elements as u64]);
        encode_rmsnorm_gated_vjp_f32(
            context,
            &encoder,
            &replay.recurrence_output,
            weights.norm,
            &replay.z,
            &grad_normed_flat,
            &grad_recurrence_output,
            &grad_z,
            checked_product(n_tokens, geometry.n_v_heads)?,
            geometry.head_dim,
            RMS_EPS * geometry.head_dim as f32,
        )?;
        encode_fill_f32(context, &encoder, &zero_final_recurrence_state, 0.0)?;
        encode_gdn_step_decay_packed_vjp_f32(
            context,
            &encoder,
            &replay.q,
            &replay.k,
            &replay.v,
            &replay.decay,
            &replay.beta,
            &replay.initial_recurrence_state,
            &replay.recurrence_checkpoints,
            n_tokens,
            &grad_recurrence_output,
            &zero_final_recurrence_state,
            &grad_q,
            &grad_k,
            &grad_v,
            &grad_decay,
            &grad_beta,
            &grad_initial_recurrence_state,
            &recurrence_state_scratch_a,
            &recurrence_state_scratch_b,
            &correction_scratch,
            &residual_scratch,
            n_tokens,
            geometry.n_v_heads,
            geometry.n_k_heads,
            geometry.head_dim,
        )?;
        let packed_k_heads = checked_product(n_tokens, geometry.n_k_heads)?;
        encode_l2_norm_vjp_batched_f32(
            context,
            &encoder,
            &replay.q_raw,
            &grad_q,
            &grad_q_raw,
            packed_k_heads,
            geometry.head_dim,
            RMS_EPS,
        )?;
        encode_l2_norm_vjp_batched_f32(
            context,
            &encoder,
            &replay.k_raw,
            &grad_k,
            &grad_k_raw,
            packed_k_heads,
            geometry.head_dim,
            RMS_EPS,
        )?;
        encode_fill_f32(context, &encoder, &zero_final_conv_state, 0.0)?;
        let conv_weight = weights
            .conv1d
            .view_subrange(0, vec![(geometry.conv_dim * 4) as u64]);
        encode_ssm_conv_silu_split_packed_vjp_f32(
            context,
            &encoder,
            &replay.qkv_source,
            &replay.initial_conv_state,
            &replay.conv_checkpoints,
            n_tokens,
            &conv_weight,
            &grad_q_raw,
            &grad_k_raw,
            &grad_v,
            &zero_final_conv_state,
            &grad_qkv,
            &grad_initial_conv_state,
            &conv_state_scratch_a,
            &conv_state_scratch_b,
            n_tokens,
            geometry.n_k_heads,
            geometry.n_v_heads,
            geometry.head_dim,
        )?;
        for token in 0..n_tokens {
            let alpha_source = row_view(&replay.alpha_source, token, geometry.n_v_heads);
            let decay = row_view(&replay.decay, token, geometry.n_v_heads);
            let grad_decay_row = row_view(&grad_decay, token, geometry.n_v_heads);
            let grad_alpha_row = row_view(&grad_alpha_source, token, geometry.n_v_heads);
            encode_gdn_decay_chain_vjp_f32(
                context,
                &encoder,
                &alpha_source,
                weights.dt_bias,
                weights.a_log,
                &decay,
                &grad_decay_row,
                &grad_alpha_row,
            )?;
        }
        encode_sigmoid_output_vjp_f32(
            context,
            &encoder,
            &replay.beta,
            &grad_beta,
            &grad_beta_source,
        )?;
        let grad_qkv_rows = grad_qkv.view_subrange(0, row_shape(geometry.conv_dim, n_tokens)?);
        let grad_z_rows = grad_z.view_subrange(0, row_shape(geometry.v_elements, n_tokens)?);
        let grad_alpha_rows =
            grad_alpha_source.view_subrange(0, row_shape(geometry.n_v_heads, n_tokens)?);
        let grad_beta_rows =
            grad_beta_source.view_subrange(0, row_shape(geometry.n_v_heads, n_tokens)?);
        for (weight, grad_output, grad_hidden_output, n_out) in [
            (
                weights.in_proj_qkv,
                &grad_qkv_rows,
                &grad_hidden_qkv,
                geometry.conv_dim,
            ),
            (
                weights.in_proj_z,
                &grad_z_rows,
                &grad_hidden_z,
                geometry.v_elements,
            ),
            (
                weights.alpha_proj,
                &grad_alpha_rows,
                &grad_hidden_alpha,
                geometry.n_v_heads,
            ),
            (
                weights.beta_proj,
                &grad_beta_rows,
                &grad_hidden_beta,
                geometry.n_v_heads,
            ),
        ] {
            encode_frozen_linear_vjp_f32(
                context,
                &encoder,
                weight,
                grad_output,
                grad_hidden_output,
                geometry.hidden_size,
                n_out,
                n_tokens,
            )?;
        }
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_qkv,
            &grad_hidden_z,
            &grad_hidden_sum_a,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_alpha,
            &grad_hidden_beta,
            &grad_hidden_sum_b,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_sum_a,
            &grad_hidden_sum_b,
            &grad_hidden,
        )?;
        encode_rms_norm_mul_vjp_rows_f32(
            context,
            &encoder,
            &replay.input,
            weights.attn_norm,
            &grad_hidden,
            &grad_input,
            n_tokens,
            geometry.hidden_size,
            RMS_EPS,
            match rule {
                GdnMixerVjpRule::Jacobian => RmsNormVjpRule::Jacobian,
                GdnMixerVjpRule::Relp => RmsNormVjpRule::RelpDetachedScale,
            },
        )?;
        Ok(())
    })();
    encoder.end();
    encode_result?;
    command.commit();
    command.waitUntilCompleted();
    validate_completed_command(&command)?;
    let (
        final_conv_state,
        final_recurrence_state,
        grad_initial_conv_state,
        grad_initial_recurrence_state,
    ) = if read_state_diagnostics {
        (
            read_f32(&replay.conv_state, geometry.conv_state_elements),
            read_f32(&replay.recurrence_state, geometry.state_elements),
            read_f32(&grad_initial_conv_state, geometry.conv_state_elements),
            read_f32(&grad_initial_recurrence_state, geometry.state_elements),
        )
    } else {
        (Vec::new(), Vec::new(), Vec::new(), Vec::new())
    };
    Ok(GdnReplayVjpReadback {
        mixer_outputs: read_f32(&replay.mixer_output, hidden_elements),
        final_conv_state,
        final_recurrence_state,
        grad_input: read_f32(&grad_input, hidden_elements),
        grad_initial_conv_state,
        grad_initial_recurrence_state,
    })
}

fn copy_workspace_token_capture(
    destination: &mut [f32],
    token_capture: &[f32],
    token: usize,
    n_tokens: usize,
    n_layers: usize,
    hidden_size: usize,
) -> Result<(), ResearchError> {
    let token_elements = checked_product(n_layers, hidden_size)?;
    let layer_elements = checked_product(n_tokens, hidden_size)?;
    let bank_elements = checked_product(n_layers, layer_elements)?;
    if token_capture.len() != token_elements {
        return Err(ResearchError::ActivationSize {
            name: "workspace token capture",
            got: token_capture.len(),
            expected: token_elements,
        });
    }
    if destination.len() != bank_elements {
        return Err(ResearchError::ActivationSize {
            name: "workspace layer-major destination",
            got: destination.len(),
            expected: bank_elements,
        });
    }
    if token >= n_tokens {
        return Err(ResearchError::ActivationSize {
            name: "workspace token index",
            got: token,
            expected: n_tokens,
        });
    }
    for layer in 0..n_layers {
        let source = checked_product(layer, hidden_size)?;
        let source_end = source
            .checked_add(hidden_size)
            .ok_or(ResearchError::SizeOverflow)?;
        let destination_row = checked_product(layer, n_tokens)?
            .checked_add(token)
            .ok_or(ResearchError::SizeOverflow)?;
        let destination_start = checked_product(destination_row, hidden_size)?;
        let destination_end = destination_start
            .checked_add(hidden_size)
            .ok_or(ResearchError::SizeOverflow)?;
        destination[destination_start..destination_end]
            .copy_from_slice(&token_capture[source..source_end]);
    }
    Ok(())
}

fn compose_workspace_vjp(
    target_layer: u32,
    source_layers: &[u32],
    hidden_elements: usize,
    target_cotangent: &[f32],
    mut reverse_block: impl FnMut(
        u32,
        &[f32],
    ) -> Result<
        (Vec<f32>, ResearchWorkspaceReplayDiagnostic),
        ResearchError,
    >,
) -> Result<(Vec<f32>, Vec<ResearchWorkspaceReplayDiagnostic>), ResearchError> {
    if source_layers.is_empty() {
        return Err(ResearchError::EmptyWorkspaceSourceLayers);
    }
    for &source in source_layers {
        if source >= target_layer {
            return Err(ResearchError::WorkspaceSourceNotBeforeTarget {
                source_layer: source,
                target_layer,
            });
        }
    }
    if target_cotangent.len() != hidden_elements {
        return Err(ResearchError::ActivationSize {
            name: "workspace target cotangent",
            got: target_cotangent.len(),
            expected: hidden_elements,
        });
    }
    let output_elements = checked_product(source_layers.len(), hidden_elements)?;
    let mut values = vec![0.0f32; output_elements];
    let mut diagnostics = Vec::new();
    let earliest_source = source_layers
        .iter()
        .copied()
        .min()
        .ok_or(ResearchError::EmptyWorkspaceSourceLayers)?;
    let first_block = earliest_source
        .checked_add(1)
        .ok_or(ResearchError::SizeOverflow)?;
    let mut gradient = target_cotangent.to_vec();
    for layer in (first_block..=target_layer).rev() {
        let (next_gradient, diagnostic) = reverse_block(layer, &gradient)?;
        if next_gradient.len() != hidden_elements {
            return Err(ResearchError::ActivationSize {
                name: "workspace reversed block cotangent",
                got: next_gradient.len(),
                expected: hidden_elements,
            });
        }
        gradient = next_gradient;
        diagnostics.push(diagnostic);
        let crossed_source = layer - 1;
        for (slot, &source) in source_layers.iter().enumerate() {
            if source == crossed_source {
                let start = checked_product(slot, hidden_elements)?;
                let end = start
                    .checked_add(hidden_elements)
                    .ok_or(ResearchError::SizeOverflow)?;
                values[start..end].copy_from_slice(&gradient);
            }
        }
    }
    Ok((values, diagnostics))
}

pub fn workspace_valid_position_range(
    n_tokens: usize,
    skip_first: usize,
) -> Result<std::ops::Range<usize>, ResearchError> {
    let minimum = skip_first
        .checked_add(2)
        .ok_or(ResearchError::SizeOverflow)?;
    if n_tokens < minimum {
        return Err(ResearchError::WorkspaceNoValidPositions {
            n_tokens,
            skip_first,
        });
    }
    Ok(skip_first..n_tokens - 1)
}

fn reduce_workspace_source_positions(
    source: &[f32],
    n_tokens: usize,
    hidden_size: usize,
    valid_positions: std::ops::Range<usize>,
    destination: &mut [f32],
) -> Result<(), ResearchError> {
    let expected = checked_product(n_tokens, hidden_size)?;
    if source.len() != expected {
        return Err(ResearchError::ActivationSize {
            name: "workspace source trajectory reduction",
            got: source.len(),
            expected,
        });
    }
    if destination.len() != hidden_size {
        return Err(ResearchError::ActivationSize {
            name: "workspace fitted row destination",
            got: destination.len(),
            expected: hidden_size,
        });
    }
    let count = valid_positions.len();
    if count == 0 || valid_positions.end > n_tokens {
        return Err(ResearchError::WorkspaceNoValidPositions {
            n_tokens,
            skip_first: valid_positions.start,
        });
    }
    destination.fill(0.0);
    for position in valid_positions {
        let start = checked_product(position, hidden_size)?;
        let end = start
            .checked_add(hidden_size)
            .ok_or(ResearchError::SizeOverflow)?;
        for (sum, &value) in destination.iter_mut().zip(&source[start..end]) {
            *sum += value;
        }
    }
    let scale = (count as f32).recip();
    for value in destination {
        *value *= scale;
    }
    Ok(())
}

fn merge_workspace_diagnostics(
    aggregate: &mut Vec<ResearchWorkspaceReplayDiagnostic>,
    current: &[ResearchWorkspaceReplayDiagnostic],
) -> Result<(), ResearchError> {
    if aggregate.is_empty() {
        aggregate.extend_from_slice(current);
        return Ok(());
    }
    if aggregate.len() != current.len() {
        return Err(ResearchError::WorkspaceDiagnosticScheduleMismatch);
    }
    for (aggregate, current) in aggregate.iter_mut().zip(current) {
        if aggregate.layer != current.layer || aggregate.kind != current.kind {
            return Err(ResearchError::WorkspaceDiagnosticScheduleMismatch);
        }
        aggregate.residual_replay_max_abs_error = aggregate
            .residual_replay_max_abs_error
            .max(current.residual_replay_max_abs_error);
    }
    Ok(())
}

fn checked_product(left: usize, right: usize) -> Result<usize, ResearchError> {
    left.checked_mul(right).ok_or(ResearchError::SizeOverflow)
}

fn row_shape(width: usize, rows: usize) -> Result<Vec<u64>, ResearchError> {
    Ok(vec![
        u64::try_from(width).map_err(|_| ResearchError::SizeOverflow)?,
        u64::try_from(rows).map_err(|_| ResearchError::SizeOverflow)?,
    ])
}

fn flat_f32(context: &MetalContext, elements: usize) -> Result<MetalTensor, ResearchError> {
    Ok(MetalTensor::zeros_f32(
        context,
        vec![u64::try_from(elements).map_err(|_| ResearchError::SizeOverflow)?],
    )?)
}

fn f32_from_slice(context: &MetalContext, values: &[f32]) -> Result<MetalTensor, ResearchError> {
    Ok(MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(values),
        vec![u64::try_from(values.len()).map_err(|_| ResearchError::SizeOverflow)?],
        GgmlType::F32,
    )?)
}

fn row_view(tensor: &MetalTensor, row: usize, width: usize) -> MetalTensor {
    tensor.view_subrange((row * width) as u64, vec![width as u64])
}

fn validate_completed_command(
    command: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>>,
) -> Result<(), ResearchError> {
    let status = command.status();
    let error = command.error();
    if status != MTLCommandBufferStatus::Completed || error.is_some() {
        return Err(ResearchError::CommandBuffer {
            status: format!("{status:?}"),
            error: format!("{error:?}"),
        });
    }
    Ok(())
}

fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return f32::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(&left, &right)| finite_abs_difference(left, right))
        .fold(0.0f32, f32::max)
}

fn finite_abs_difference(left: f32, right: f32) -> f32 {
    if !left.is_finite() || !right.is_finite() {
        return f32::INFINITY;
    }
    let difference = (left - right).abs();
    if difference.is_finite() {
        difference
    } else {
        f32::INFINITY
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

#[allow(clippy::too_many_arguments)]
fn dense_ffn_vjp_rows_readback(
    context: &MetalContext,
    layer: u32,
    hidden_size: usize,
    intermediate_size: usize,
    pre_ffn_residuals: &[f32],
    post_norm: &MetalTensor,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    down_weight: &MetalTensor,
    grad_outputs: &[f32],
    n_rows: usize,
    rule: DenseFfnVjpRule,
) -> Result<Vec<f32>, ResearchError> {
    if n_rows == 0 {
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
    let hidden_elements = checked_product(n_rows, hidden_size)?;
    checked_product(n_rows, intermediate_size)?;
    for (name, values) in [
        ("pre-FFN residual rows", pre_ffn_residuals),
        ("post-block cotangent rows", grad_outputs),
    ] {
        if values.len() != hidden_elements {
            return Err(ResearchError::ActivationSize {
                name,
                got: values.len(),
                expected: hidden_elements,
            });
        }
    }
    let hidden_shape = row_shape(hidden_size, n_rows)?;
    let intermediate_shape = row_shape(intermediate_size, n_rows)?;
    let residuals = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(pre_ffn_residuals),
        hidden_shape.clone(),
        GgmlType::F32,
    )?;
    let normalized = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let gate = MetalTensor::zeros_f32(context, intermediate_shape.clone())?;
    let up = MetalTensor::zeros_f32(context, intermediate_shape.clone())?;
    let grad_output = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(grad_outputs),
        hidden_shape.clone(),
        GgmlType::F32,
    )?;
    let grad_inner = MetalTensor::zeros_f32(context, intermediate_shape.clone())?;
    let grad_gate = MetalTensor::zeros_f32(context, intermediate_shape.clone())?;
    let grad_up = MetalTensor::zeros_f32(context, intermediate_shape)?;
    let grad_norm_gate = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_norm_up = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_norm = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_ffn_input = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(context, hidden_shape)?;

    let command = context
        .queue
        .commandBuffer()
        .ok_or(ResearchError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), ResearchError> {
        encode_rms_norm_mul_rows_f32(
            context,
            &encoder,
            &residuals,
            post_norm,
            &normalized,
            n_rows,
            hidden_size,
            RMS_EPS,
        )?;
        for row in 0..n_rows {
            let normalized_row = row_view(&normalized, row, hidden_size);
            let gate_row = row_view(&gate, row, intermediate_size);
            let up_row = row_view(&up, row, intermediate_size);
            encode_mat_vec_dispatch(
                context,
                &encoder,
                gate_weight,
                &normalized_row,
                &gate_row,
                hidden_size,
                intermediate_size,
            )?;
            encode_mat_vec_dispatch(
                context,
                &encoder,
                up_weight,
                &normalized_row,
                &up_row,
                hidden_size,
                intermediate_size,
            )?;
        }
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            down_weight,
            &grad_output,
            &grad_inner,
            intermediate_size,
            hidden_size,
            n_rows,
        )?;
        let (rms_rule, swiglu_rule) = match rule {
            DenseFfnVjpRule::Jacobian => (RmsNormVjpRule::Jacobian, SwiGluVjpRule::Jacobian),
            DenseFfnVjpRule::Relp => (
                RmsNormVjpRule::RelpDetachedScale,
                SwiGluVjpRule::RelpIdentityHalf,
            ),
        };
        encode_silu_mul_vjp_f32(
            context,
            &encoder,
            &gate,
            &up,
            &grad_inner,
            &grad_gate,
            &grad_up,
            n_rows,
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
            n_rows,
        )?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            up_weight,
            &grad_up,
            &grad_norm_up,
            hidden_size,
            intermediate_size,
            n_rows,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_norm_gate,
            &grad_norm_up,
            &grad_norm,
        )?;
        encode_rms_norm_mul_vjp_rows_f32(
            context,
            &encoder,
            &residuals,
            post_norm,
            &grad_norm,
            &grad_ffn_input,
            n_rows,
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
    validate_completed_command(&command)?;
    Ok(read_f32(&grad_input, hidden_elements))
}

#[allow(clippy::too_many_arguments)]
fn dense_ffn_vjp_query_rows_readback(
    context: &MetalContext,
    layer: u32,
    hidden_size: usize,
    intermediate_size: usize,
    pre_ffn_residuals: &[f32],
    post_norm: &MetalTensor,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    down_weight: &MetalTensor,
    grad_outputs: &[f32],
    n_rows: usize,
    n_query_batches: usize,
    rule: DenseFfnVjpRule,
) -> Result<Vec<f32>, ResearchError> {
    if n_rows == 0 || n_query_batches == 0 {
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

    let hidden_elements = checked_product(n_rows, hidden_size)?;
    checked_product(n_rows, intermediate_size)?;
    let query_rows = checked_product(n_query_batches, n_rows)?;
    let hidden_query_elements = checked_product(query_rows, hidden_size)?;
    checked_product(query_rows, intermediate_size)?;
    if pre_ffn_residuals.len() != hidden_elements {
        return Err(ResearchError::ActivationSize {
            name: "pre-FFN residual rows",
            got: pre_ffn_residuals.len(),
            expected: hidden_elements,
        });
    }
    if grad_outputs.len() != hidden_query_elements {
        return Err(ResearchError::ActivationSize {
            name: "post-block cotangent query rows",
            got: grad_outputs.len(),
            expected: hidden_query_elements,
        });
    }

    let hidden_shape = row_shape(hidden_size, n_rows)?;
    let intermediate_shape = row_shape(intermediate_size, n_rows)?;
    let hidden_query_shape = row_shape(hidden_size, query_rows)?;
    let intermediate_query_shape = row_shape(intermediate_size, query_rows)?;
    let residuals = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(pre_ffn_residuals),
        hidden_shape.clone(),
        GgmlType::F32,
    )?;
    let normalized = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let gate = MetalTensor::zeros_f32(context, intermediate_shape.clone())?;
    let up = MetalTensor::zeros_f32(context, intermediate_shape.clone())?;
    let grad_output = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(grad_outputs),
        hidden_query_shape.clone(),
        GgmlType::F32,
    )?;
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
        encode_rms_norm_mul_rows_f32(
            context,
            &encoder,
            &residuals,
            post_norm,
            &normalized,
            n_rows,
            hidden_size,
            RMS_EPS,
        )?;
        for row in 0..n_rows {
            let normalized_row = row_view(&normalized, row, hidden_size);
            let gate_row = row_view(&gate, row, intermediate_size);
            let up_row = row_view(&up, row, intermediate_size);
            encode_mat_vec_dispatch(
                context,
                &encoder,
                gate_weight,
                &normalized_row,
                &gate_row,
                hidden_size,
                intermediate_size,
            )?;
            encode_mat_vec_dispatch(
                context,
                &encoder,
                up_weight,
                &normalized_row,
                &up_row,
                hidden_size,
                intermediate_size,
            )?;
        }
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            down_weight,
            &grad_output,
            &grad_inner,
            intermediate_size,
            hidden_size,
            query_rows,
        )?;
        let (rms_rule, swiglu_rule) = match rule {
            DenseFfnVjpRule::Jacobian => (RmsNormVjpRule::Jacobian, SwiGluVjpRule::Jacobian),
            DenseFfnVjpRule::Relp => (
                RmsNormVjpRule::RelpDetachedScale,
                SwiGluVjpRule::RelpIdentityHalf,
            ),
        };
        let intermediate_query_elements = checked_product(n_rows, intermediate_size)?;
        for query in 0..n_query_batches {
            let offset = u64::try_from(checked_product(query, intermediate_query_elements)?)
                .map_err(|_| ResearchError::SizeOverflow)?;
            let grad_inner_query = grad_inner.view_subrange(offset, intermediate_shape.clone());
            let grad_gate_query = grad_gate.view_subrange(offset, intermediate_shape.clone());
            let grad_up_query = grad_up.view_subrange(offset, intermediate_shape.clone());
            encode_silu_mul_vjp_f32(
                context,
                &encoder,
                &gate,
                &up,
                &grad_inner_query,
                &grad_gate_query,
                &grad_up_query,
                n_rows,
                intermediate_size,
                swiglu_rule,
            )?;
        }
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            gate_weight,
            &grad_gate,
            &grad_norm_gate,
            hidden_size,
            intermediate_size,
            query_rows,
        )?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            up_weight,
            &grad_up,
            &grad_norm_up,
            hidden_size,
            intermediate_size,
            query_rows,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_norm_gate,
            &grad_norm_up,
            &grad_norm,
        )?;
        for query in 0..n_query_batches {
            let offset = u64::try_from(checked_product(query, hidden_elements)?)
                .map_err(|_| ResearchError::SizeOverflow)?;
            let grad_norm_query = grad_norm.view_subrange(offset, hidden_shape.clone());
            let grad_ffn_input_query = grad_ffn_input.view_subrange(offset, hidden_shape.clone());
            encode_rms_norm_mul_vjp_rows_f32(
                context,
                &encoder,
                &residuals,
                post_norm,
                &grad_norm_query,
                &grad_ffn_input_query,
                n_rows,
                hidden_size,
                RMS_EPS,
                rms_rule,
            )?;
        }
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
    validate_completed_command(&command)?;
    Ok(read_f32(&grad_input, hidden_query_elements))
}

struct GdnBlockComposition {
    values: Vec<f32>,
    grad_post_mixer_residuals: Vec<f32>,
    mixer: ResearchGdnVjp,
}

#[allow(clippy::too_many_arguments)]
fn compose_gdn_block_vjp(
    context: &MetalContext,
    layer: u32,
    hidden_size: usize,
    intermediate_size: usize,
    post_mixer_residuals: &[f32],
    post_norm: &MetalTensor,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    down_weight: &MetalTensor,
    grad_block_output: &[f32],
    n_rows: usize,
    rule: GdnBlockVjpRule,
    mixer_vjp: impl FnOnce(&[f32], GdnMixerVjpRule) -> Result<ResearchGdnVjp, ResearchError>,
) -> Result<GdnBlockComposition, ResearchError> {
    let grad_post_mixer_residuals = dense_ffn_vjp_rows_readback(
        context,
        layer,
        hidden_size,
        intermediate_size,
        post_mixer_residuals,
        post_norm,
        gate_weight,
        up_weight,
        down_weight,
        grad_block_output,
        n_rows,
        match rule {
            GdnBlockVjpRule::Jacobian => DenseFfnVjpRule::Jacobian,
            GdnBlockVjpRule::Relp => DenseFfnVjpRule::Relp,
        },
    )?;
    let mixer = mixer_vjp(
        &grad_post_mixer_residuals,
        match rule {
            GdnBlockVjpRule::Jacobian => GdnMixerVjpRule::Jacobian,
            GdnBlockVjpRule::Relp => GdnMixerVjpRule::Relp,
        },
    )?;
    if mixer.values.len() != grad_post_mixer_residuals.len() {
        return Err(ResearchError::ActivationSize {
            name: "GDN mixer branch cotangent",
            got: mixer.values.len(),
            expected: grad_post_mixer_residuals.len(),
        });
    }
    let values = grad_post_mixer_residuals
        .iter()
        .zip(&mixer.values)
        .map(|(&identity, &branch)| identity + branch)
        .collect();
    Ok(GdnBlockComposition {
        values,
        grad_post_mixer_residuals,
        mixer,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_dense_ffn_weights(
    layer: u32,
    hidden_size: usize,
    intermediate_size: usize,
    post_norm: &MetalTensor,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    down_weight: &MetalTensor,
) -> Result<(), ResearchError> {
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
    Ok(())
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
    fn workspace_capture_transposes_tokens_into_layer_major_banks() {
        const TOKENS: usize = 3;
        const LAYERS: usize = 2;
        const HIDDEN: usize = 2;
        let token_captures = [
            [0.0, 1.0, 10.0, 11.0],
            [2.0, 3.0, 12.0, 13.0],
            [4.0, 5.0, 14.0, 15.0],
        ];
        let mut bank = vec![0.0; TOKENS * LAYERS * HIDDEN];
        for (token, capture) in token_captures.iter().enumerate() {
            copy_workspace_token_capture(&mut bank, capture, token, TOKENS, LAYERS, HIDDEN)
                .unwrap();
        }
        assert_eq!(
            bank,
            [
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0
            ]
        );
    }

    #[test]
    fn workspace_batch_accessors_preserve_source_then_query_layout() {
        let batch = ResearchWorkspaceVjpBatch {
            target_layer: 3,
            source_layers: vec![0, 2],
            n_query: 3,
            n_tokens: 2,
            hidden_size: 2,
            values: (0..24).map(|value| value as f32).collect(),
            diagnostics: Vec::new(),
        };
        assert_eq!(
            batch.source_values(1).unwrap(),
            (12..24).map(|value| value as f32).collect::<Vec<_>>()
        );
        assert_eq!(
            batch.source_query_values(1, 2).unwrap(),
            [20.0, 21.0, 22.0, 23.0]
        );
        assert!(batch.source_values(2).is_none());
        assert!(batch.source_query_values(0, 3).is_none());
    }

    #[test]
    fn workspace_reference_positions_exclude_prefix_and_final_token() {
        assert_eq!(workspace_valid_position_range(8, 4).unwrap(), 4..7);
        assert!(matches!(
            workspace_valid_position_range(5, 4).unwrap_err(),
            ResearchError::WorkspaceNoValidPositions {
                n_tokens: 5,
                skip_first: 4
            }
        ));

        let source = [0.0f32, 10.0, 2.0, 12.0, 4.0, 14.0, 6.0, 16.0, 100.0, 200.0];
        let mut row = [0.0f32; 2];
        reduce_workspace_source_positions(&source, 5, 2, 1..4, &mut row).unwrap();
        assert_eq!(row, [4.0, 14.0]);
    }

    #[test]
    fn workspace_row_diagnostics_merge_by_schedule_and_maximum() {
        let mut aggregate = vec![ResearchWorkspaceReplayDiagnostic {
            layer: 3,
            kind: ResearchWorkspaceBlockKind::Attention,
            residual_replay_max_abs_error: 0.1,
        }];
        merge_workspace_diagnostics(
            &mut aggregate,
            &[ResearchWorkspaceReplayDiagnostic {
                layer: 3,
                kind: ResearchWorkspaceBlockKind::Attention,
                residual_replay_max_abs_error: 0.2,
            }],
        )
        .unwrap();
        assert_eq!(aggregate[0].residual_replay_max_abs_error, 0.2);
        assert!(matches!(
            merge_workspace_diagnostics(
                &mut aggregate,
                &[ResearchWorkspaceReplayDiagnostic {
                    layer: 2,
                    kind: ResearchWorkspaceBlockKind::Gdn,
                    residual_replay_max_abs_error: 0.0,
                }],
            )
            .unwrap_err(),
            ResearchError::WorkspaceDiagnosticScheduleMismatch
        ));
    }

    #[test]
    fn workspace_vjp_crosses_sources_in_caller_order_without_reversing_them() {
        let source_layers = [0, 3, 1, 3];
        let target_cotangent = [1.0f32, 2.0, 3.0];
        let mut traversed = Vec::new();
        let (values, diagnostics) = compose_workspace_vjp(
            4,
            &source_layers,
            target_cotangent.len(),
            &target_cotangent,
            |layer, gradient| {
                traversed.push(layer);
                Ok((
                    gradient.iter().map(|value| value * layer as f32).collect(),
                    ResearchWorkspaceReplayDiagnostic {
                        layer,
                        kind: if layer == 4 {
                            ResearchWorkspaceBlockKind::Attention
                        } else {
                            ResearchWorkspaceBlockKind::Gdn
                        },
                        residual_replay_max_abs_error: layer as f32 * 1e-6,
                    },
                ))
            },
        )
        .unwrap();
        assert_eq!(traversed, [4, 3, 2, 1]);
        assert_eq!(
            diagnostics
                .iter()
                .map(|diagnostic| diagnostic.layer)
                .collect::<Vec<_>>(),
            traversed
        );
        assert_eq!(&values[0..3], &[24.0, 48.0, 72.0]);
        assert_eq!(&values[3..6], &[4.0, 8.0, 12.0]);
        assert_eq!(&values[6..9], &[24.0, 48.0, 72.0]);
        assert_eq!(&values[9..12], &[4.0, 8.0, 12.0]);

        let error = compose_workspace_vjp(
            4,
            &[4],
            target_cotangent.len(),
            &target_cotangent,
            |_, _| unreachable!(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ResearchError::WorkspaceSourceNotBeforeTarget {
                source_layer: 4,
                target_layer: 4
            }
        ));
    }

    #[test]
    fn replay_diagnostics_fail_closed_on_non_finite_values() {
        assert_eq!(finite_abs_difference(f32::NAN, 0.0), f32::INFINITY);
        assert_eq!(finite_abs_difference(0.0, f32::INFINITY), f32::INFINITY);
        assert_eq!(
            max_abs_difference(&[0.0, f32::NAN], &[0.0, 0.0]),
            f32::INFINITY
        );
    }

    #[test]
    fn causal_gated_attention_and_rope_vjps_match_finite_differences() {
        const TOKENS: usize = 4;
        const N_Q: usize = 4;
        const N_KV: usize = 2;
        const HEAD_DIM: usize = 8;
        const N_ROT: usize = 4;
        let geometry = AttnGeometry {
            hidden_size: 13,
            n_q_heads: N_Q,
            n_kv_heads: N_KV,
            head_dim: HEAD_DIM,
            n_rot: N_ROT,
            q_elements: N_Q * HEAD_DIM,
            q_full_elements: 2 * N_Q * HEAD_DIM,
            kv_elements: N_KV * HEAD_DIM,
            rope_theta: 10_000.0,
        };
        let values = |len: usize, stride: usize, scale: f32, offset: f32| {
            (0..len)
                .map(|index| ((index * stride + 3) % 47) as f32 * scale + offset)
                .collect::<Vec<_>>()
        };
        let q = values(TOKENS * geometry.q_elements, 5, 0.013, -0.27);
        let k = values(TOKENS * geometry.kv_elements, 7, 0.011, -0.23);
        let v = values(TOKENS * geometry.kv_elements, 11, 0.017, -0.35);
        let gate = values(TOKENS * geometry.q_elements, 13, 0.029, -0.61);
        let grad = values(TOKENS * geometry.q_elements, 17, 0.019, -0.41);
        let actual =
            cpu_causal_gated_attention_vjp(&q, &k, &v, &gate, &grad, TOKENS, geometry).unwrap();
        let objective = |q: &[f32], k: &[f32], v: &[f32], gate: &[f32]| {
            cpu_causal_gated_attention_forward(q, k, v, gate, TOKENS, geometry)
                .unwrap()
                .gated_output
                .iter()
                .zip(&grad)
                .map(|(&value, &gradient)| f64::from(value) * f64::from(gradient))
                .sum::<f64>()
        };
        let epsilon = 2e-3f32;
        let finite_difference = |values: &[f32], index: usize, evaluate: &dyn Fn(&[f32]) -> f64| {
            let mut plus = values.to_vec();
            let mut minus = values.to_vec();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            (evaluate(&plus) - evaluate(&minus)) / (2.0 * f64::from(epsilon))
        };
        for &index in &[
            2 * geometry.q_elements,
            3 * geometry.q_elements + HEAD_DIM - 1,
        ] {
            let fd = finite_difference(&q, index, &|candidate| objective(candidate, &k, &v, &gate));
            assert!((fd - f64::from(actual.grad_q[index])).abs() < 2e-3);
        }
        for &index in &[0usize, geometry.kv_elements + 3, k.len() - 1] {
            let fd = finite_difference(&k, index, &|candidate| objective(&q, candidate, &v, &gate));
            assert!((fd - f64::from(actual.grad_k[index])).abs() < 2e-3);
            let fd = finite_difference(&v, index, &|candidate| objective(&q, &k, candidate, &gate));
            assert!((fd - f64::from(actual.grad_v[index])).abs() < 2e-3);
        }
        for &index in &[0usize, 2 * geometry.q_elements + 5, gate.len() - 1] {
            let fd = finite_difference(&gate, index, &|candidate| objective(&q, &k, &v, candidate));
            assert!((fd - f64::from(actual.grad_gate[index])).abs() < 2e-3);
        }
        let direction = |len: usize, stride: usize| {
            (0..len)
                .map(|index| ((index * stride + 1) % 31) as f32 * 0.0013 - 0.019)
                .collect::<Vec<_>>()
        };
        let dq = direction(q.len(), 19);
        let dk = direction(k.len(), 23);
        let dv = direction(v.len(), 29);
        let dg = direction(gate.len(), 31);
        let shift = |base: &[f32], tangent: &[f32], amount: f32| {
            base.iter()
                .zip(tangent)
                .map(|(&base, &tangent)| base + amount * tangent)
                .collect::<Vec<_>>()
        };
        let plus = objective(
            &shift(&q, &dq, epsilon),
            &shift(&k, &dk, epsilon),
            &shift(&v, &dv, epsilon),
            &shift(&gate, &dg, epsilon),
        );
        let minus = objective(
            &shift(&q, &dq, -epsilon),
            &shift(&k, &dk, -epsilon),
            &shift(&v, &dv, -epsilon),
            &shift(&gate, &dg, -epsilon),
        );
        let forward_directional = (plus - minus) / (2.0 * f64::from(epsilon));
        let inner = |gradient: &[f32], tangent: &[f32]| {
            gradient
                .iter()
                .zip(tangent)
                .map(|(&gradient, &tangent)| f64::from(gradient) * f64::from(tangent))
                .sum::<f64>()
        };
        let reverse_directional = inner(&actual.grad_q, &dq)
            + inner(&actual.grad_k, &dk)
            + inner(&actual.grad_v, &dv)
            + inner(&actual.grad_gate, &dg);
        assert!((forward_directional - reverse_directional).abs() < 2e-3);

        let rope_input = values(TOKENS * N_Q * HEAD_DIM, 37, 0.021, -0.44);
        let rope_grad = values(TOKENS * N_Q * HEAD_DIM, 41, 0.018, -0.39);
        let mut rotated = rope_input.clone();
        rope_neox_rows_in_place(
            &mut rotated,
            TOKENS,
            N_Q,
            HEAD_DIM,
            N_ROT,
            7,
            geometry.rope_theta,
            false,
        )
        .unwrap();
        let mut transposed = rope_grad.clone();
        rope_neox_rows_in_place(
            &mut transposed,
            TOKENS,
            N_Q,
            HEAD_DIM,
            N_ROT,
            7,
            geometry.rope_theta,
            true,
        )
        .unwrap();
        let left = inner(&rotated, &rope_grad);
        let right = inner(&rope_input, &transposed);
        assert!((left - right).abs() < 2e-5);
        for token in 0..TOKENS {
            for head in 0..N_Q {
                let base = (token * N_Q + head) * HEAD_DIM;
                for index in N_ROT..HEAD_DIM {
                    assert_eq!(
                        rotated[base + index].to_bits(),
                        rope_input[base + index].to_bits()
                    );
                    assert_eq!(
                        transposed[base + index].to_bits(),
                        rope_grad[base + index].to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn hybrid_attention_mixer_vjp_matches_directional_finite_differences() {
        let context = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const TOKENS: usize = 3;
        const HIDDEN: usize = 16;
        const N_Q: usize = 2;
        const N_KV: usize = 1;
        const HEAD_DIM: usize = 8;
        let geometry = AttnGeometry {
            hidden_size: HIDDEN,
            n_q_heads: N_Q,
            n_kv_heads: N_KV,
            head_dim: HEAD_DIM,
            n_rot: 4,
            q_elements: N_Q * HEAD_DIM,
            q_full_elements: 2 * N_Q * HEAD_DIM,
            kv_elements: N_KV * HEAD_DIM,
            rope_theta: 10_000.0,
        };
        let values = |len: usize, stride: usize, scale: f32, offset: f32| {
            (0..len)
                .map(|index| ((index * stride + 3) % 47) as f32 * scale + offset)
                .collect::<Vec<_>>()
        };
        let attn_norm = f32_tensor(
            &context,
            &values(HIDDEN, 5, 0.011, 0.67),
            vec![HIDDEN as u64],
        );
        let q_weight_values = values(HIDDEN * 2 * geometry.q_elements, 7, 0.0017, -0.037);
        let q_weight = f32_tensor(
            &context,
            &q_weight_values,
            vec![HIDDEN as u64, (2 * geometry.q_elements) as u64],
        );
        let k_weight_values = values(HIDDEN * geometry.kv_elements, 11, 0.0021, -0.043);
        let k_weight = f32_tensor(
            &context,
            &k_weight_values,
            vec![HIDDEN as u64, geometry.kv_elements as u64],
        );
        let v_weight_values = values(HIDDEN * geometry.kv_elements, 13, 0.0023, -0.047);
        let v_weight = f32_tensor(
            &context,
            &v_weight_values,
            vec![HIDDEN as u64, geometry.kv_elements as u64],
        );
        let o_weight_values = values(geometry.q_elements * HIDDEN, 17, 0.0019, -0.039);
        let o_weight = f32_tensor(
            &context,
            &o_weight_values,
            vec![geometry.q_elements as u64, HIDDEN as u64],
        );
        let q_norm = f32_tensor(
            &context,
            &values(HEAD_DIM, 19, 0.017, 0.59),
            vec![HEAD_DIM as u64],
        );
        let k_norm = f32_tensor(
            &context,
            &values(HEAD_DIM, 23, 0.019, 0.57),
            vec![HEAD_DIM as u64],
        );
        let weights = AttnMixerWeights {
            attn_norm: &attn_norm,
            q: &q_weight,
            k: &k_weight,
            v: &v_weight,
            o: &o_weight,
            q_norm: &q_norm,
            k_norm: &k_norm,
        };
        let input = values(TOKENS * HIDDEN, 29, 0.009, -0.19);
        let grad = values(TOKENS * HIDDEN, 31, 0.013, -0.27);
        let run = |input: &[f32], grad: &[f32], rule| {
            attn_mixer_replay_vjp_readback(&context, geometry, weights, input, grad, TOKENS, rule)
                .unwrap()
        };
        let actual = run(&input, &grad, AttnBlockVjpRule::Jacobian);
        assert!(actual.grad_input.iter().all(|value| value.is_finite()));
        assert!(actual.grad_input.iter().any(|value| *value != 0.0));
        let objective = |candidate: &[f32]| {
            run(candidate, &grad, AttnBlockVjpRule::Jacobian)
                .mixer_outputs
                .iter()
                .zip(&grad)
                .map(|(&value, &gradient)| f64::from(value) * f64::from(gradient))
                .sum::<f64>()
        };
        let epsilon = 2e-2f32;
        for index in [0usize, HIDDEN, input.len() - 1] {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let fd = (objective(&plus) - objective(&minus)) / (2.0 * f64::from(epsilon));
            assert!(
                (fd - f64::from(actual.grad_input[index])).abs() < 2e-3,
                "coordinate {index} fd={fd} reverse={}",
                actual.grad_input[index]
            );
        }
        let direction = values(input.len(), 37, 0.0013, -0.021);
        let shift = |amount: f32| {
            input
                .iter()
                .zip(&direction)
                .map(|(&value, &direction)| value + amount * direction)
                .collect::<Vec<_>>()
        };
        let forward_directional =
            (objective(&shift(epsilon)) - objective(&shift(-epsilon))) / (2.0 * f64::from(epsilon));
        let reverse_directional = actual
            .grad_input
            .iter()
            .zip(&direction)
            .map(|(&gradient, &direction)| f64::from(gradient) * f64::from(direction))
            .sum::<f64>();
        assert!(
            (forward_directional - reverse_directional).abs() < 2e-3,
            "hybrid attention directional mismatch forward={forward_directional} reverse={reverse_directional}"
        );
        let zero = run(&input, &vec![0.0; grad.len()], AttnBlockVjpRule::Jacobian);
        assert!(zero.grad_input.iter().all(|value| *value == 0.0));
        let relp = run(&input, &grad, AttnBlockVjpRule::Relp);
        assert!(
            relp.grad_input
                .iter()
                .zip(&actual.grad_input)
                .any(|(relp, jacobian)| relp.to_bits() != jacobian.to_bits())
        );
    }

    #[test]
    fn gdn_mixer_replay_vjp_matches_directional_finite_differences() {
        let context = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const N_TOKENS: usize = 3;
        const HIDDEN: usize = 32;
        const N_K: usize = 1;
        const N_V: usize = 2;
        const HEAD_DIM: usize = 128;
        let qk_elements = N_K * HEAD_DIM;
        let v_elements = N_V * HEAD_DIM;
        let conv_dim = 2 * qk_elements + v_elements;
        let state_elements = v_elements * HEAD_DIM;
        let conv_state_elements = 3 * conv_dim;
        let geometry = GdnGeometry {
            hidden_size: HIDDEN,
            n_v_heads: N_V,
            n_k_heads: N_K,
            head_dim: HEAD_DIM,
            qk_elements,
            v_elements,
            conv_dim,
            state_elements,
            conv_state_elements,
        };
        let values = |len: usize, stride: usize, scale: f32, offset: f32| {
            (0..len)
                .map(|index| ((index * stride + 3) % 47) as f32 * scale + offset)
                .collect::<Vec<_>>()
        };
        let attn_norm = f32_tensor(
            &context,
            &values(HIDDEN, 5, 0.013, 0.63),
            vec![HIDDEN as u64],
        );
        let qkv_weight_values = values(HIDDEN * conv_dim, 7, 0.0007, -0.015);
        let qkv_weight = f32_tensor(
            &context,
            &qkv_weight_values,
            vec![HIDDEN as u64, conv_dim as u64],
        );
        let z_weight_values = values(HIDDEN * v_elements, 11, 0.0008, -0.017);
        let z_weight = f32_tensor(
            &context,
            &z_weight_values,
            vec![HIDDEN as u64, v_elements as u64],
        );
        let beta_weight_values = values(HIDDEN * N_V, 13, 0.0011, -0.021);
        let beta_weight = f32_tensor(
            &context,
            &beta_weight_values,
            vec![HIDDEN as u64, N_V as u64],
        );
        let alpha_weight_values = values(HIDDEN * N_V, 17, 0.0013, -0.024);
        let alpha_weight = f32_tensor(
            &context,
            &alpha_weight_values,
            vec![HIDDEN as u64, N_V as u64],
        );
        let a_log = f32_tensor(&context, &[-0.08, -0.13], vec![N_V as u64]);
        let dt_bias = f32_tensor(&context, &[0.07, -0.11], vec![N_V as u64]);
        let conv_weight_values = values(4 * conv_dim, 19, 0.0019, -0.041);
        let conv_weight = f32_tensor(&context, &conv_weight_values, vec![4, conv_dim as u64]);
        let internal_norm = f32_tensor(
            &context,
            &values(HEAD_DIM, 23, 0.009, 0.58),
            vec![HEAD_DIM as u64],
        );
        let out_weight_values = values(v_elements * HIDDEN, 29, 0.0009, -0.019);
        let out_weight = f32_tensor(
            &context,
            &out_weight_values,
            vec![v_elements as u64, HIDDEN as u64],
        );
        let weights = GdnMixerWeights {
            attn_norm: &attn_norm,
            in_proj_qkv: &qkv_weight,
            in_proj_z: &z_weight,
            beta_proj: &beta_weight,
            alpha_proj: &alpha_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            conv1d: &conv_weight,
            norm: &internal_norm,
            out_proj: &out_weight,
        };
        let input = values(N_TOKENS * HIDDEN, 31, 0.007, -0.15);
        let initial_conv_state = values(conv_state_elements, 37, 0.0008, -0.018);
        let initial_recurrence_state = values(state_elements, 41, 0.00017, -0.004);
        let grad_output = values(N_TOKENS * HIDDEN, 43, 0.006, -0.13);
        let run = |input: &[f32], grad_output: &[f32], rule| {
            gdn_mixer_replay_vjp_readback(
                &context,
                geometry,
                weights,
                input,
                &initial_conv_state,
                &initial_recurrence_state,
                grad_output,
                N_TOKENS,
                rule,
                true,
            )
            .unwrap()
        };
        let actual = run(&input, &grad_output, GdnMixerVjpRule::Jacobian);
        assert!(actual.grad_input.iter().all(|value| value.is_finite()));
        assert!(actual.mixer_outputs.iter().all(|value| value.is_finite()));
        assert!(actual.grad_input.iter().any(|value| *value != 0.0));
        assert_eq!(actual.final_conv_state.len(), conv_state_elements);
        assert_eq!(actual.final_recurrence_state.len(), state_elements);

        let objective = |candidate: &[f32]| {
            run(candidate, &grad_output, GdnMixerVjpRule::Jacobian)
                .mixer_outputs
                .iter()
                .zip(&grad_output)
                .map(|(&value, &gradient)| f64::from(value) * f64::from(gradient))
                .sum::<f64>()
        };
        let epsilon = 2e-2f32;
        for index in [0usize, HIDDEN, input.len() - 1] {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let finite_difference =
                (objective(&plus) - objective(&minus)) / (2.0 * f64::from(epsilon));
            assert!(
                (finite_difference - f64::from(actual.grad_input[index])).abs() < 8e-4,
                "coordinate {index} finite_difference={finite_difference} reverse={}",
                actual.grad_input[index]
            );
        }
        let direction = values(input.len(), 47, 0.0011, -0.023);
        let shift = |amount: f32| {
            input
                .iter()
                .zip(&direction)
                .map(|(&value, &direction)| value + amount * direction)
                .collect::<Vec<_>>()
        };
        let forward_directional =
            (objective(&shift(epsilon)) - objective(&shift(-epsilon))) / (2.0 * f64::from(epsilon));
        let reverse_directional = actual
            .grad_input
            .iter()
            .zip(&direction)
            .map(|(&gradient, &direction)| f64::from(gradient) * f64::from(direction))
            .sum::<f64>();
        assert!(
            (forward_directional - reverse_directional).abs() < 1e-3,
            "directional mismatch forward={forward_directional} reverse={reverse_directional}"
        );

        let zeros = vec![0.0f32; grad_output.len()];
        let zero = run(&input, &zeros, GdnMixerVjpRule::Jacobian);
        assert!(zero.grad_input.iter().all(|value| *value == 0.0));
        assert!(
            zero.grad_initial_conv_state
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            zero.grad_initial_recurrence_state
                .iter()
                .all(|value| *value == 0.0)
        );
        let relp = run(&input, &grad_output, GdnMixerVjpRule::Relp);
        assert!(
            relp.grad_input
                .iter()
                .zip(&actual.grad_input)
                .any(|(relp, jacobian)| relp.to_bits() != jacobian.to_bits())
        );
    }

    #[test]
    fn dense_ffn_query_rows_match_serial_batches_and_do_not_cross_talk() {
        let context = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const ROWS: usize = 3;
        const HIDDEN: usize = 11;
        const INTERMEDIATE: usize = 17;
        let hidden_elements = ROWS * HIDDEN;
        let residuals: Vec<f32> = (0..hidden_elements)
            .map(|index| ((index * 7 + 3) % 29) as f32 * 0.027 - 0.36)
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

        for rule in [DenseFfnVjpRule::Jacobian, DenseFfnVjpRule::Relp] {
            for query_batches in [1, 2, 8] {
                let grad_outputs: Vec<f32> = (0..query_batches * hidden_elements)
                    .map(|index| {
                        let query = index / hidden_elements;
                        let local = index % hidden_elements;
                        ((local * 19 + query * 7 + 4) % 47) as f32 * 0.009 - 0.18
                            + query as f32 * 0.013
                    })
                    .collect();
                let batched = dense_ffn_vjp_query_rows_readback(
                    &context,
                    3,
                    HIDDEN,
                    INTERMEDIATE,
                    &residuals,
                    &norm_tensor,
                    &gate_tensor,
                    &up_tensor,
                    &down_tensor,
                    &grad_outputs,
                    ROWS,
                    query_batches,
                    rule,
                )
                .unwrap();
                for query in 0..query_batches {
                    let start = query * hidden_elements;
                    let end = start + hidden_elements;
                    let serial = dense_ffn_vjp_rows_readback(
                        &context,
                        3,
                        HIDDEN,
                        INTERMEDIATE,
                        &residuals,
                        &norm_tensor,
                        &gate_tensor,
                        &up_tensor,
                        &down_tensor,
                        &grad_outputs[start..end],
                        ROWS,
                        rule,
                    )
                    .unwrap();
                    let max_abs = batched[start..end]
                        .iter()
                        .zip(&serial)
                        .map(|(&batched, &serial)| (batched - serial).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_abs < 2e-5,
                        "{rule:?} batch {query_batches} query {query} error {max_abs}"
                    );
                }
            }

            const QUERY_BATCHES: usize = 4;
            const ACTIVE_QUERY: usize = 2;
            let active_grad: Vec<f32> = (0..hidden_elements)
                .map(|index| ((index * 23 + 9) % 53) as f32 * 0.007 - 0.16)
                .collect();
            let mut isolated_grad = vec![0.0f32; QUERY_BATCHES * hidden_elements];
            let active_start = ACTIVE_QUERY * hidden_elements;
            isolated_grad[active_start..active_start + hidden_elements]
                .copy_from_slice(&active_grad);
            let isolated = dense_ffn_vjp_query_rows_readback(
                &context,
                3,
                HIDDEN,
                INTERMEDIATE,
                &residuals,
                &norm_tensor,
                &gate_tensor,
                &up_tensor,
                &down_tensor,
                &isolated_grad,
                ROWS,
                QUERY_BATCHES,
                rule,
            )
            .unwrap();
            let serial = dense_ffn_vjp_rows_readback(
                &context,
                3,
                HIDDEN,
                INTERMEDIATE,
                &residuals,
                &norm_tensor,
                &gate_tensor,
                &up_tensor,
                &down_tensor,
                &active_grad,
                ROWS,
                rule,
            )
            .unwrap();
            for query in 0..QUERY_BATCHES {
                let start = query * hidden_elements;
                let end = start + hidden_elements;
                if query == ACTIVE_QUERY {
                    let max_abs = isolated[start..end]
                        .iter()
                        .zip(&serial)
                        .map(|(&batched, &serial)| (batched - serial).abs())
                        .fold(0.0f32, f32::max);
                    assert!(max_abs < 2e-5, "{rule:?} isolated query error {max_abs}");
                } else {
                    assert!(
                        isolated[start..end].iter().all(|value| *value == 0.0),
                        "{rule:?} query {query} received another query's cotangent"
                    );
                }
            }
        }
    }

    #[test]
    fn gdn_block_composition_matches_distinct_row_and_temporal_oracles() {
        let context = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const ROWS: usize = 3;
        const HIDDEN: usize = 11;
        const INTERMEDIATE: usize = 17;
        let residuals: Vec<f32> = (0..ROWS * HIDDEN)
            .map(|index| ((index * 7 + 3) % 29) as f32 * 0.027 - 0.36)
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
        let grad_output: Vec<f32> = (0..ROWS * HIDDEN)
            .map(|index| ((index * 19 + 4) % 47) as f32 * 0.009 - 0.18)
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
        for rule in [DenseFfnVjpRule::Jacobian, DenseFfnVjpRule::Relp] {
            let actual = dense_ffn_vjp_rows_readback(
                &context,
                3,
                HIDDEN,
                INTERMEDIATE,
                &residuals,
                &norm_tensor,
                &gate_tensor,
                &up_tensor,
                &down_tensor,
                &grad_output,
                ROWS,
                rule,
            )
            .unwrap();
            let mut expected = Vec::with_capacity(ROWS * HIDDEN);
            for row in 0..ROWS {
                expected.extend(cpu_dense_ffn_vjp(
                    &residuals[row * HIDDEN..(row + 1) * HIDDEN],
                    &norm,
                    &gate_weight,
                    &up_weight,
                    &down_weight,
                    &grad_output[row * HIDDEN..(row + 1) * HIDDEN],
                    1,
                    INTERMEDIATE,
                    rule,
                ));
            }
            let max_abs = actual
                .iter()
                .zip(&expected)
                .map(|(&actual, &expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs < 2e-5, "{rule:?} rowwise FFN error {max_abs}");

            let block_rule = match rule {
                DenseFfnVjpRule::Jacobian => GdnBlockVjpRule::Jacobian,
                DenseFfnVjpRule::Relp => GdnBlockVjpRule::Relp,
            };
            let expected_mixer_rule = match rule {
                DenseFfnVjpRule::Jacobian => GdnMixerVjpRule::Jacobian,
                DenseFfnVjpRule::Relp => GdnMixerVjpRule::Relp,
            };
            let composition = compose_gdn_block_vjp(
                &context,
                3,
                HIDDEN,
                INTERMEDIATE,
                &residuals,
                &norm_tensor,
                &gate_tensor,
                &up_tensor,
                &down_tensor,
                &grad_output,
                ROWS,
                block_rule,
                |incoming, mixer_rule| {
                    assert_eq!(mixer_rule, expected_mixer_rule);
                    let incoming_error = incoming
                        .iter()
                        .zip(&expected)
                        .map(|(&incoming, &expected)| (incoming - expected).abs())
                        .fold(0.0f32, f32::max);
                    assert!(incoming_error < 2e-5);
                    let mut branch = vec![0.0f32; incoming.len()];
                    for row in 0..ROWS {
                        for column in 0..HIDDEN {
                            let index = row * HIDDEN + column;
                            branch[index] = 0.35 * incoming[index]
                                + if row + 1 < ROWS {
                                    0.2 * incoming[(row + 1) * HIDDEN + column]
                                } else {
                                    0.0
                                };
                        }
                    }
                    Ok(ResearchGdnVjp {
                        layer: 3,
                        n_tokens: ROWS,
                        hidden_size: HIDDEN,
                        values: branch,
                        grad_initial_conv_state: Vec::new(),
                        grad_initial_recurrence_state: Vec::new(),
                        replay_mixer_outputs: Vec::new(),
                        residual_replay_max_abs_error: 0.0,
                        final_conv_state_max_abs_error: 0.0,
                        final_recurrence_state_max_abs_error: 0.0,
                    })
                },
            )
            .unwrap();
            let mut expected_full = expected.clone();
            for row in 0..ROWS {
                for column in 0..HIDDEN {
                    let index = row * HIDDEN + column;
                    expected_full[index] += 0.35 * expected[index]
                        + if row + 1 < ROWS {
                            0.2 * expected[(row + 1) * HIDDEN + column]
                        } else {
                            0.0
                        };
                }
            }
            let post_mixer_error = composition
                .grad_post_mixer_residuals
                .iter()
                .zip(&expected)
                .map(|(&actual, &expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            let full_error = composition
                .values
                .iter()
                .zip(&expected_full)
                .map(|(&actual, &expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(post_mixer_error < 2e-5);
            assert!(full_error < 4e-5, "{rule:?} full block error {full_error}");
        }

        let zero_gate = f32_tensor(
            &context,
            &vec![0.0; HIDDEN * INTERMEDIATE],
            vec![HIDDEN as u64, INTERMEDIATE as u64],
        );
        let zero_up = f32_tensor(
            &context,
            &vec![0.0; HIDDEN * INTERMEDIATE],
            vec![HIDDEN as u64, INTERMEDIATE as u64],
        );
        let zero_down = f32_tensor(
            &context,
            &vec![0.0; INTERMEDIATE * HIDDEN],
            vec![INTERMEDIATE as u64, HIDDEN as u64],
        );
        let identity_only = compose_gdn_block_vjp(
            &context,
            3,
            HIDDEN,
            INTERMEDIATE,
            &residuals,
            &norm_tensor,
            &zero_gate,
            &zero_up,
            &zero_down,
            &grad_output,
            ROWS,
            GdnBlockVjpRule::Jacobian,
            |incoming, _| {
                Ok(ResearchGdnVjp {
                    layer: 3,
                    n_tokens: ROWS,
                    hidden_size: HIDDEN,
                    values: vec![0.0; incoming.len()],
                    grad_initial_conv_state: Vec::new(),
                    grad_initial_recurrence_state: Vec::new(),
                    replay_mixer_outputs: Vec::new(),
                    residual_replay_max_abs_error: 0.0,
                    final_conv_state_max_abs_error: 0.0,
                    final_recurrence_state_max_abs_error: 0.0,
                })
            },
        )
        .unwrap();
        assert_eq!(identity_only.grad_post_mixer_residuals, grad_output);
        assert_eq!(identity_only.values, grad_output);
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
