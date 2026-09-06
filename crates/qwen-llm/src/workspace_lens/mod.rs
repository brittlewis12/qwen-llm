//! Safe, narrow instrumentation surface for workspace lenses.
//!
//! This module deliberately returns owned CPU data and opaque linear IDs. It
//! does not expose resident model buffers, command encoders, or mutable Metal
//! session state across the public API boundary.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalMemoryAdmissionReason, MetalTensor,
    RmsNormVjpRule, SwiGluVjpRule, encode_add_f32, encode_copy_offset_f32, encode_fill_f32,
    encode_frozen_linear_vjp_f32, encode_gdn_decay_chain_batched_f32,
    encode_gdn_decay_chain_vjp_f32, encode_gdn_prep_packed_ckpt_f32,
    encode_gdn_step_decay_packed_ckpt_f32, encode_gdn_step_decay_packed_vjp_f32,
    encode_get_rows_f32, encode_l2_norm_batched_f32, encode_l2_norm_vjp_batched_f32,
    encode_mask_row_indices_f32, encode_mat_mat_f16_f32, encode_mat_vec_f16_f32,
    encode_mps_topk16_f32, encode_rms_norm_batched_f32, encode_rms_norm_mul_f32,
    encode_rms_norm_mul_rows_f32, encode_rms_norm_mul_vjp_broadcast_f32,
    encode_rms_norm_mul_vjp_rows_f32, encode_rmsnorm_gated_f32, encode_rmsnorm_gated_vjp_f32,
    encode_sigmoid_f32, encode_sigmoid_output_vjp_f32, encode_silu_mul_vjp_broadcast_f32,
    encode_silu_mul_vjp_f32, encode_split_q_gate_f32, encode_ssm_conv_silu_split_packed_vjp_f32,
    evaluate_metal_memory_admission,
};
use crate::metal_dflash::{
    DFlashError, MetalDFlashLayerMajorScratch, PrefillScratchConfig,
    plan_prefill_scratch_with_matrix_max_pos_configured,
    prefill_tokens_with_multi_hidden_prompt_only_profiled,
};
use crate::metal_forward::{
    MetalAttnBlock, MetalBlock, MetalGdnBlock, MfError, RMS_EPS, encode_mat_mat_dispatch,
    encode_mat_vec_dispatch,
};
use crate::model::{Arch, ArchKind};
use crate::runtime::{LoadedModel, RuntimeError, Sequence};
use crate::tensor::GgmlType;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};
use std::time::Instant;

mod attn;
mod dense_ffn;
mod error;
mod fit;
mod gdn;
mod readout;
mod session;
mod support;
#[cfg(test)]
mod tests;
#[allow(unused_imports)]
use attn::*;
#[allow(unused_imports)]
use dense_ffn::*;
pub use error::*;
#[allow(unused_imports)]
use fit::*;
#[allow(unused_imports)]
use gdn::*;
#[allow(unused_imports)]
use readout::*;
#[allow(unused_imports)]
use session::*;
pub use support::*;

/// Lightweight locator identity derived from model metadata, shard paths, and
/// file stamps. It is useful within one machine, but is not a content digest.
pub const WORKSPACE_LENS_IDENTITY_SCHEME: &str = "qwen_llm_model_locator_v1";
pub const MAX_WORKSPACE_LENS_GDN_TOKENS: usize = 128;
pub const MAX_WORKSPACE_LENS_ATTN_TOKENS: usize = 128;
pub const MAX_WORKSPACE_LENS_TOKENS: usize = 128;
pub const MAX_WORKSPACE_LENS_DIM_BATCH: usize = 32;
pub const MAX_WORKSPACE_LENS_PACKED_READOUT_POSITIONS: usize = 128;
const PACKED_FULL_READOUT_CHUNK_SIZE: usize = 16;
const MPS_FULL_READOUT_TOP_K: usize = 16;
const FULL_READOUT_CANDIDATE_COUNT: usize = 2 * MPS_FULL_READOUT_TOP_K;
const MAX_FULL_READOUT_TOP_K: usize = 25;
const F16_TRANSPORT_READOUT_LIVE_TRANSPORT_BANKS: usize = 2;
const F16_TRANSPORT_READOUT_LIVE_COVECTOR_BANKS: usize = 4;
/// Maximum peak host bytes attributable to a newly materialized workspace-lens
/// result and its immediate fitting/readout workspaces. 256 MiB keeps selected
/// experimental banks practical while preventing accidental multi-GiB jobs.
pub const MAX_WORKSPACE_LENS_OWNED_RESULT_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorkspaceLensModelIdentity {
    pub model_locator_id: u64,
    pub tokenizer_metadata_id: u64,
    pub content_authenticated: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WorkspaceLensLinear {
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
pub struct WorkspaceLensLinearInfo {
    pub id: WorkspaceLensLinear,
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
pub struct WorkspaceLensForward {
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
pub struct WorkspaceLensDenseFfnForward {
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
pub struct WorkspaceLensGdnForward {
    identity: WorkspaceLensModelIdentity,
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

impl WorkspaceLensGdnForward {
    pub fn identity(&self) -> WorkspaceLensModelIdentity {
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
pub struct WorkspaceLensGdnVjp {
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
pub struct WorkspaceLensGdnBlockVjp {
    pub layer: u32,
    pub n_tokens: usize,
    pub hidden_size: usize,
    /// Full block-input cotangents, including both residual identities.
    pub values: Vec<f32>,
    /// Cotangents after reversing the FFN residual update and before the mixer
    /// residual update, flattened `[T, H]`.
    pub grad_post_mixer_residuals: Vec<f32>,
    pub mixer: WorkspaceLensGdnVjp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensAttnForward {
    identity: WorkspaceLensModelIdentity,
    owner_token_id: u64,
    layer: u32,
    token_ids: Vec<i32>,
    hidden_size: usize,
    final_logits: Vec<f32>,
    input_residuals: Vec<f32>,
    post_mixer_residuals: Vec<f32>,
    post_block_residuals: Vec<f32>,
}

impl WorkspaceLensAttnForward {
    pub fn identity(&self) -> WorkspaceLensModelIdentity {
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
pub struct WorkspaceLensAttnBlockVjp {
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
pub struct WorkspaceLensPromptForward {
    identity: WorkspaceLensModelIdentity,
    owner_token_id: u64,
    token_ids: Vec<i32>,
    n_layers: u32,
    hidden_size: usize,
    post_mixer_residuals: Vec<f32>,
    post_block_residuals: Vec<f32>,
}

impl WorkspaceLensPromptForward {
    pub fn identity(&self) -> WorkspaceLensModelIdentity {
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
pub enum WorkspaceLensBlockKind {
    Gdn,
    Attention,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensReplayDiagnostic {
    pub layer: u32,
    pub kind: WorkspaceLensBlockKind,
    /// Drift between replayed `input + mixer(input)` and the production
    /// residual. Attention replay is an F32 model-level oracle and therefore
    /// does not differentiate production F16/Q8 KV conversion.
    pub residual_replay_max_abs_error: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensVjp {
    pub target_layer: u32,
    /// Caller order, including duplicates.
    pub source_layers: Vec<u32>,
    pub n_tokens: usize,
    pub hidden_size: usize,
    /// Source-layer order, flattened `[K,T,H]`.
    pub values: Vec<f32>,
    /// Reverse traversal order, from the target block toward the earliest
    /// requested source layer.
    pub diagnostics: Vec<WorkspaceLensReplayDiagnostic>,
}

impl WorkspaceLensVjp {
    pub fn source_values(&self, slot: usize) -> Option<&[f32]> {
        let source_elements = self.n_tokens.checked_mul(self.hidden_size)?;
        let start = slot.checked_mul(source_elements)?;
        self.values.get(start..start.checked_add(source_elements)?)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensVjpBatch {
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
    pub diagnostics: Vec<WorkspaceLensReplayDiagnostic>,
}

impl WorkspaceLensVjpBatch {
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
pub struct WorkspaceLensRows {
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
    pub diagnostics: Vec<WorkspaceLensReplayDiagnostic>,
}

impl WorkspaceLensRows {
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

/// Arbitrary target-covector workspace fits. Values are owned CPU F32 data in
/// source-layer-major, query-major order `[K,Q,H]`.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensReadouts {
    pub target_layer: u32,
    pub source_layers: Vec<u32>,
    pub n_query: usize,
    pub n_tokens: usize,
    pub n_valid_positions: usize,
    pub hidden_size: usize,
    pub values: Vec<f32>,
    /// Maximum replay drift observed for each traversed block across all queries.
    pub diagnostics: Vec<WorkspaceLensReplayDiagnostic>,
}

impl WorkspaceLensReadouts {
    pub fn source_values(&self, source_slot: usize) -> Option<&[f32]> {
        let source_elements = self.n_query.checked_mul(self.hidden_size)?;
        let start = source_slot.checked_mul(source_elements)?;
        self.values.get(start..start.checked_add(source_elements)?)
    }

    pub fn query_values(&self, source_slot: usize, query_slot: usize) -> Option<&[f32]> {
        if query_slot >= self.n_query {
            return None;
        }
        let source_elements = self.n_query.checked_mul(self.hidden_size)?;
        let start = source_slot
            .checked_mul(source_elements)?
            .checked_add(query_slot.checked_mul(self.hidden_size)?)?;
        self.values.get(start..start.checked_add(self.hidden_size)?)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkspaceLensTokenCovectorKind {
    RawLmHead,
    DeployedLogitNumerator,
}

/// Selected vocabulary target covectors gathered from resident LM-head rows.
/// `covector_kind` records whether final RMSNorm gamma was folded into them.
/// No RMS denominator or softmax is applied.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensTokenReadouts {
    pub covector_kind: WorkspaceLensTokenCovectorKind,
    pub token_ids: Vec<u32>,
    pub hidden_size: usize,
    pub lm_head_dtype: GgmlType,
    /// Native GGUF axis order `[H,V]`.
    pub lm_head_shape: [usize; 2],
    pub output_norm_dtype: GgmlType,
    pub output_norm_shape: Vec<u64>,
    /// Token-major score covectors, flattened `[V_selected,H]`.
    pub values: Vec<f32>,
}

impl WorkspaceLensTokenReadouts {
    pub fn token_values(&self, slot: usize) -> Option<&[f32]> {
        let start = slot.checked_mul(self.hidden_size)?;
        self.values.get(start..start.checked_add(self.hidden_size)?)
    }
}

pub struct WorkspaceLensPreparedF16Transport<'model> {
    model: &'model LoadedModel,
    hidden_size: usize,
    tensor: MetalTensor,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensPromptLastCapture {
    pub position: usize,
    pub token_id: i32,
    pub capture: ActivationCapture,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensVocabularyScore {
    pub token_id: u32,
    pub logit: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensFullVocabularyReadout {
    /// Diagnostic F64 host recomputation, rounded to F32. The deployed Metal
    /// RMSNorm performs its own F32 parallel reduction for the actual logits.
    pub rms_denominator_f64_recomputed: f32,
    pub scores: Vec<WorkspaceLensVocabularyScore>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensFullVocabularyReadoutWithVector {
    pub readout: WorkspaceLensFullVocabularyReadout,
    /// Transported target-coordinate residual before output RMSNorm.
    pub transported_values: Vec<f32>,
}

/// Model-bound GPU storage for repeated full-vocabulary readout rows.
///
/// The capacity is expressed in rows so callers can later tile positions or
/// flatten prompt batches without changing the allocation contract.
pub struct WorkspaceLensFullReadoutWorkspace<'model> {
    model: &'model LoadedModel,
    row_capacity: usize,
    transport_bound: bool,
    transport: MetalTensor,
    source: MetalTensor,
    transported: MetalTensor,
    normalized: MetalTensor,
    logits: MetalTensor,
    first_ids: MetalTensor,
    first_values: MetalTensor,
    second_ids: MetalTensor,
    second_values: MetalTensor,
}

struct PackedFullReadoutValidation {
    layer_slot: usize,
    position_count: usize,
    hidden_size: usize,
    vocab_size: usize,
    transported_position_rows: Vec<usize>,
}

impl WorkspaceLensFullReadoutWorkspace<'_> {
    pub fn apply_row_f16_transport_topk(
        &mut self,
        transport_bytes: &[u8],
        source_residual: &[f32],
        top_k: usize,
    ) -> Result<WorkspaceLensFullVocabularyReadout, WorkspaceLensError> {
        Ok(self
            .apply_row_f16_transport_topk_with_vector(transport_bytes, source_residual, top_k)?
            .readout)
    }

    pub fn row_capacity(&self) -> usize {
        self.row_capacity
    }
}

/// Opaque packed post-block residual capture owned by one loaded model.
/// The resident `[T,K,H]` Metal tensor is intentionally private.
pub struct WorkspaceLensPackedPostBlockCapture<'model> {
    model: &'model LoadedModel,
    start_position: usize,
    token_ids: Vec<i32>,
    layer_ids: Vec<u32>,
    hidden_size: usize,
    packed_prefill_gpu_ms: f64,
    packed_prefill_wall_ms: f64,
    values: MetalTensor,
}

impl WorkspaceLensPackedPostBlockCapture<'_> {
    pub fn start_position(&self) -> usize {
        self.start_position
    }

    pub fn position_count(&self) -> usize {
        self.token_ids.len()
    }

    pub fn end_position(&self) -> usize {
        self.start_position + self.token_ids.len()
    }

    pub fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }

    /// Unique zero-based block indices in the caller's requested order.
    pub fn layer_ids(&self) -> &[u32] {
        &self.layer_ids
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn packed_prefill_wall_ms(&self) -> f64 {
        self.packed_prefill_wall_ms
    }

    pub fn packed_prefill_gpu_ms(&self) -> f64 {
        self.packed_prefill_gpu_ms
    }

    #[cfg(test)]
    fn row_for_test(
        &self,
        position_row: usize,
        source_layer: u32,
    ) -> Result<Vec<f32>, WorkspaceLensError> {
        let layer_slot = self.layer_slot(source_layer)?;
        if position_row >= self.position_count() {
            return Err(WorkspaceLensError::ActivationSize {
                name: "packed diagnostic capture row",
                got: position_row,
                expected: self.position_count(),
            });
        }
        let offset = checked_product(
            checked_product(position_row, self.layer_ids.len())?
                .checked_add(layer_slot)
                .ok_or(WorkspaceLensError::SizeOverflow)?,
            self.hidden_size,
        )?;
        let row = self.values.view_subrange(
            u64::try_from(offset).map_err(|_| WorkspaceLensError::SizeOverflow)?,
            vec![self.hidden_size as u64],
        );
        read_f32_fallible(&row, self.hidden_size, "packed diagnostic capture row")
    }

    fn layer_slot(&self, source_layer: u32) -> Result<usize, WorkspaceLensError> {
        self.layer_ids
            .iter()
            .position(|&layer| layer == source_layer)
            .ok_or(WorkspaceLensError::PackedCaptureLayerNotFound { source_layer })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensPackedVocabularyPosition {
    /// Zero-based absolute position of the captured prompt token.
    pub source_position: usize,
    pub source_token_id: i32,
    /// The transported residual at `source_position` predicts this position.
    pub predicts_position: usize,
    pub scores: Vec<WorkspaceLensVocabularyScore>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensPackedTransportedVector {
    /// Zero-based absolute position of the captured prompt token.
    pub source_position: usize,
    pub source_token_id: i32,
    /// The transported residual at `source_position` predicts this position.
    pub predicts_position: usize,
    /// Target-coordinate F32 values before the deployed output RMSNorm and LM head.
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceLensPackedFullVocabularyReadout {
    /// Zero-based block index captured after its second residual update.
    pub source_layer: u32,
    pub start_position: usize,
    pub position_count: usize,
    pub top_k: usize,
    /// GPU intervals accumulated across packed prompt chunks.
    pub packed_prefill_gpu_ms: f64,
    /// Wall time spent in the packed prompt forward, including its required wait.
    pub packed_prefill_wall_ms: f64,
    /// GPU interval for transport, output norm, LM head, and top-k.
    pub readout_gpu_ms: f64,
    /// Wall time through completion of the readout command buffer.
    pub readout_wall_ms: f64,
    pub positions: Vec<WorkspaceLensPackedVocabularyPosition>,
    /// Caller-selected transported rows in caller request order.
    pub transported_vectors: Vec<WorkspaceLensPackedTransportedVector>,
}

pub struct WorkspaceLensSession<'model, 'sequence> {
    model: &'model LoadedModel,
    sequence: &'sequence mut Sequence,
}

impl LoadedModel {
    pub fn workspace_lens_identity(&self) -> WorkspaceLensModelIdentity {
        let (model_locator_id, tokenizer_metadata_id) = self.lightweight_identity_parts();
        WorkspaceLensModelIdentity {
            model_locator_id,
            tokenizer_metadata_id,
            content_authenticated: false,
        }
    }

    pub fn workspace_lens_session<'model, 'sequence>(
        &'model self,
        sequence: &'sequence mut Sequence,
    ) -> Result<WorkspaceLensSession<'model, 'sequence>, WorkspaceLensError> {
        self.ensure_owns(sequence)?;
        if self.arch().kind != ArchKind::Dense {
            return Err(WorkspaceLensError::UnsupportedArchitecture(
                self.arch().kind,
            ));
        }
        Ok(WorkspaceLensSession {
            model: self,
            sequence,
        })
    }

    pub fn workspace_lens_packed_capture_scratch_upper_bytes(
        &self,
        matrix_max_position: usize,
    ) -> Result<u64, WorkspaceLensError> {
        if matrix_max_position == 0 {
            return Err(WorkspaceLensError::EmptyFullReadoutPrompt);
        }
        let plan = plan_prefill_scratch_with_matrix_max_pos_configured(
            self.metal_model(),
            PACKED_FULL_READOUT_CHUNK_SIZE as u32,
            matrix_max_position,
            PrefillScratchConfig::default(),
        )?;
        Ok(plan.priced_upper_bound(|logical_bytes| {
            Ok(self
                .context()
                .shared_buffer_size_and_align(logical_bytes)?
                .size)
        })?)
    }
}
