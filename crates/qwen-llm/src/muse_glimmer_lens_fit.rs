//! Bounded one-block replay and VJP for Muse attention blocks.
//!
//! Production capture uses the scalar text path's F16 KV cache. This module
//! intentionally replays and differentiates the smooth F32 model-level graph,
//! matching the ordinary-Qwen research semantics rather than applying an STE
//! through F16 KV conversion. Replay diagnostics quantify that deployment
//! difference.

use crate::metal::{
    KernelEncoder, MetalContext, MetalTensor, encode_add_f32, encode_frozen_linear_vjp_bank_f32,
    encode_frozen_linear_vjp_f32, encode_rms_norm_batched_f32, encode_rms_norm_mul_rows_f32,
    encode_rms_norm_mul_vjp_broadcast_f32, encode_rms_norm_mul_vjp_periodic_f32,
    encode_rms_norm_mul_vjp_rows_f32, encode_silu_mul_f32, encode_silu_mul_vjp_broadcast_f32,
    encode_silu_mul_vjp_f32, encode_silu_mul_vjp_periodic_f32, tensor_ranges_overlap,
};
use crate::metal_forward::encode_mat_vec_dispatch;
use crate::muse_glimmer::MuseGlimmerConfig;
use crate::muse_glimmer_lens::{
    MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS, MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS,
    MuseGlimmerLensCapture, MuseGlimmerLensCaptureBank, MuseGlimmerLensError, MuseGlimmerLensRule,
    MuseGlimmerRmsNormSite, MuseGlimmerSelectedTokenCovectors,
};
use crate::muse_glimmer_metal::{
    encode_muse_glimmer_causal_gqa_vjp_bank_f32,
    encode_muse_glimmer_rope_adjacent_pair_periodic_in_place_f32,
};
use crate::muse_glimmer_residency::{MuseGlimmerMetalLayerWeights, MuseGlimmerMetalModelWeights};
use crate::tensor::GgmlType;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLDevice, MTLResource,
};
use rayon::prelude::*;
use std::time::{Duration, Instant};

pub const MUSE_GLIMMER_FULL_R_MAX_DIM_BATCH: usize = 32;
pub const MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD: usize = 256;
pub const MUSE_GLIMMER_QUERY_BATCH_MAX: usize = 32;

#[allow(clippy::too_many_arguments)]
fn encode_muse_glimmer_one_attention_block_vjp_bank_feed_forward(
    ctx: &MetalContext,
    encoder: &KernelEncoder,
    config: &MuseGlimmerConfig,
    layer: &MuseGlimmerMetalLayerWeights<'_>,
    tensors: &ReplayTensors,
    geometry: MuseAttentionGeometry,
    n_tokens: usize,
    bank_rows: usize,
    grad_post_block: &MetalTensor,
    scratch: &MuseGlimmerOneBlockVjpBankScratch,
    rule: MuseGlimmerLensRule,
) -> Result<(), MuseGlimmerLensError> {
    encode_rms_norm_mul_vjp_periodic_f32(
        ctx,
        encoder,
        &tensors.ffn_branch_raw,
        layer.post_feed_forward_norm,
        grad_post_block,
        &scratch.hidden[0],
        bank_rows,
        n_tokens,
        geometry.hidden,
        config.post_norm_epsilon,
        rule.rms_norm_rule(MuseGlimmerRmsNormSite::FeedForwardBranchPostNorm),
    )?;
    encode_frozen_linear_vjp_bank_f32(
        ctx,
        encoder,
        layer.feed_forward_down,
        &scratch.hidden[0],
        &scratch.feed_forward[0],
        geometry.feed_forward,
        geometry.hidden,
        bank_rows,
    )?;
    encode_silu_mul_vjp_periodic_f32(
        ctx,
        encoder,
        &tensors.ffn_gate_raw,
        &tensors.ffn_up,
        &scratch.feed_forward[0],
        &scratch.feed_forward[1],
        &scratch.feed_forward[2],
        bank_rows,
        n_tokens,
        geometry.feed_forward,
        rule.swiglu_rule(),
    )?;
    encode_frozen_linear_vjp_bank_f32(
        ctx,
        encoder,
        layer.feed_forward_gate,
        &scratch.feed_forward[1],
        &scratch.hidden[1],
        geometry.hidden,
        geometry.feed_forward,
        bank_rows,
    )?;
    encode_frozen_linear_vjp_bank_f32(
        ctx,
        encoder,
        layer.feed_forward_up,
        &scratch.feed_forward[2],
        &scratch.hidden[2],
        geometry.hidden,
        geometry.feed_forward,
        bank_rows,
    )?;
    encode_add_f32(
        ctx,
        encoder,
        &scratch.hidden[1],
        &scratch.hidden[2],
        &scratch.hidden[3],
    )?;
    encode_rms_norm_mul_vjp_periodic_f32(
        ctx,
        encoder,
        &tensors.post_attention,
        layer.feed_forward_norm,
        &scratch.hidden[3],
        &scratch.hidden[1],
        bank_rows,
        n_tokens,
        geometry.hidden,
        config.rms_epsilon,
        rule.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreFeedForward),
    )?;
    encode_add_f32(
        ctx,
        encoder,
        grad_post_block,
        &scratch.hidden[1],
        &scratch.hidden[2],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_muse_glimmer_one_attention_block_vjp_bank_attention_output(
    ctx: &MetalContext,
    encoder: &KernelEncoder,
    config: &MuseGlimmerConfig,
    layer: &MuseGlimmerMetalLayerWeights<'_>,
    tensors: &ReplayTensors,
    geometry: MuseAttentionGeometry,
    n_tokens: usize,
    bank_rows: usize,
    scratch: &MuseGlimmerOneBlockVjpBankScratch,
    rule: MuseGlimmerLensRule,
) -> Result<(), MuseGlimmerLensError> {
    encode_rms_norm_mul_vjp_periodic_f32(
        ctx,
        encoder,
        &tensors.attention_branch_raw,
        layer.post_attention_norm,
        &scratch.hidden[2],
        &scratch.hidden[0],
        bank_rows,
        n_tokens,
        geometry.hidden,
        config.post_norm_epsilon,
        rule.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionBranchPostNorm),
    )?;
    encode_frozen_linear_vjp_bank_f32(
        ctx,
        encoder,
        layer.attention_output,
        &scratch.hidden[0],
        &scratch.query[0],
        geometry.query,
        geometry.hidden,
        bank_rows,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_muse_glimmer_one_attention_block_vjp_bank_causal_gqa(
    ctx: &MetalContext,
    encoder: &KernelEncoder,
    layer: &MuseGlimmerMetalLayerWeights<'_>,
    tensors: &ReplayTensors,
    geometry: MuseAttentionGeometry,
    basis_count: usize,
    n_tokens: usize,
    bank_rows: usize,
    scratch: &MuseGlimmerOneBlockVjpBankScratch,
) -> Result<(), MuseGlimmerLensError> {
    encode_muse_glimmer_causal_gqa_vjp_bank_f32(
        ctx,
        encoder,
        &tensors.q,
        &tensors.k,
        &tensors.v,
        &tensors.attention_gate,
        &tensors.attention_output,
        &tensors.attention_probabilities,
        &scratch.query[0],
        &scratch.query[1],
        &scratch.partial_grad_key,
        &scratch.partial_grad_value,
        &scratch.kv[0],
        &scratch.kv[1],
        &scratch.query[2],
        basis_count,
        n_tokens,
        geometry.q_heads,
        geometry.kv_heads,
        geometry.head_dim,
    )?;
    if layer.sliding_attention {
        encode_muse_glimmer_rope_adjacent_pair_periodic_in_place_f32(
            ctx,
            encoder,
            &scratch.query[1],
            &scratch.kv[0],
            geometry.q_heads,
            geometry.kv_heads,
            geometry.head_dim,
            bank_rows,
            n_tokens,
            geometry.rope_theta,
            true,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_muse_glimmer_one_attention_block_vjp_bank_attention_input(
    ctx: &MetalContext,
    encoder: &KernelEncoder,
    config: &MuseGlimmerConfig,
    layer: &MuseGlimmerMetalLayerWeights<'_>,
    tensors: &ReplayTensors,
    geometry: MuseAttentionGeometry,
    n_tokens: usize,
    bank_rows: usize,
    grad_input: &MetalTensor,
    scratch: &MuseGlimmerOneBlockVjpBankScratch,
    rule: MuseGlimmerLensRule,
) -> Result<(), MuseGlimmerLensError> {
    encode_frozen_linear_vjp_bank_f32(
        ctx,
        encoder,
        layer.attention_value,
        &scratch.kv[1],
        &scratch.hidden[0],
        geometry.hidden,
        geometry.kv,
        bank_rows,
    )?;
    encode_frozen_linear_vjp_bank_f32(
        ctx,
        encoder,
        layer.attention_gate,
        &scratch.query[2],
        &scratch.hidden[1],
        geometry.hidden,
        geometry.query,
        bank_rows,
    )?;
    encode_add_f32(
        ctx,
        encoder,
        &scratch.hidden[0],
        &scratch.hidden[1],
        &scratch.hidden[3],
    )?;

    encode_rms_norm_mul_vjp_periodic_f32(
        ctx,
        encoder,
        &tensors.q_raw_heads,
        layer.query_norm,
        &scratch.query_heads[1],
        &scratch.query_heads[0],
        checked_mul(bank_rows, geometry.q_heads, "query cotangent head rows")?,
        checked_mul(n_tokens, geometry.q_heads, "query primal head rows")?,
        geometry.head_dim,
        config.rms_epsilon,
        rule.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionQuery),
    )?;
    encode_frozen_linear_vjp_bank_f32(
        ctx,
        encoder,
        layer.attention_query,
        &scratch.query[0],
        &scratch.hidden[0],
        geometry.hidden,
        geometry.query,
        bank_rows,
    )?;
    encode_add_f32(
        ctx,
        encoder,
        &scratch.hidden[3],
        &scratch.hidden[0],
        &scratch.hidden[1],
    )?;

    encode_rms_norm_mul_vjp_periodic_f32(
        ctx,
        encoder,
        &tensors.k_raw_heads,
        layer.key_norm,
        &scratch.kv_heads[0],
        &scratch.kv_heads[1],
        checked_mul(bank_rows, geometry.kv_heads, "key cotangent head rows")?,
        checked_mul(n_tokens, geometry.kv_heads, "key primal head rows")?,
        geometry.head_dim,
        config.rms_epsilon,
        rule.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionKey),
    )?;
    encode_frozen_linear_vjp_bank_f32(
        ctx,
        encoder,
        layer.attention_key,
        &scratch.kv[1],
        &scratch.hidden[0],
        geometry.hidden,
        geometry.kv,
        bank_rows,
    )?;
    encode_add_f32(
        ctx,
        encoder,
        &scratch.hidden[1],
        &scratch.hidden[0],
        &scratch.hidden[3],
    )?;
    encode_rms_norm_mul_vjp_periodic_f32(
        ctx,
        encoder,
        &tensors.input,
        layer.attention_norm,
        &scratch.hidden[3],
        &scratch.hidden[0],
        bank_rows,
        n_tokens,
        geometry.hidden,
        config.rms_epsilon,
        rule.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention),
    )?;
    encode_add_f32(
        ctx,
        encoder,
        &scratch.hidden[2],
        &scratch.hidden[0],
        grad_input,
    )?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerOneBlockVjp {
    pub target_block: u32,
    pub rule: MuseGlimmerLensRule,
    pub n_tokens: usize,
    pub hidden_size: usize,
    /// Full cotangent at the selected block input, flattened `[T,H]`.
    pub input_cotangent: Vec<f32>,
    /// F32 replay versus production capture, whose attention KV path is F16.
    pub post_attention_replay_max_abs_error: f32,
    /// F32 replay versus production capture after the complete block.
    pub post_block_replay_max_abs_error: f32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MuseGlimmerQueryBatchVjpTimings {
    pub replay: Duration,
    pub full_attention_bank_command: Duration,
    pub sliding_attention_bank_command: Duration,
    pub feed_forward_reverse: Duration,
    pub attention_output_reverse: Duration,
    pub attention_cpu_reverse: Duration,
    pub attention_input_reverse: Duration,
    pub total: Duration,
}

impl MuseGlimmerQueryBatchVjpTimings {
    fn add_components_assign(&mut self, other: Self) {
        self.replay += other.replay;
        self.full_attention_bank_command += other.full_attention_bank_command;
        self.sliding_attention_bank_command += other.sliding_attention_bank_command;
        self.feed_forward_reverse += other.feed_forward_reverse;
        self.attention_output_reverse += other.attention_output_reverse;
        self.attention_cpu_reverse += other.attention_cpu_reverse;
        self.attention_input_reverse += other.attention_input_reverse;
    }

    fn add_assign(&mut self, other: Self) {
        self.add_components_assign(other);
        self.total += other.total;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerQueryBatchOneBlockVjp {
    pub target_block: u32,
    pub rule: MuseGlimmerLensRule,
    pub query_count: usize,
    pub n_tokens: usize,
    pub hidden_size: usize,
    /// Query-major cotangents at the selected block input, flattened `[Q,T,H]`.
    pub input_cotangents: Vec<f32>,
    pub post_attention_replay_max_abs_error: f32,
    pub post_block_replay_max_abs_error: f32,
    pub timings: MuseGlimmerQueryBatchVjpTimings,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerQueryBatchComposedVjp {
    pub source_layers: Vec<u32>,
    pub target_block: u32,
    pub method: MuseGlimmerLensRule,
    pub query_count: usize,
    pub n_tokens: usize,
    pub hidden_size: usize,
    /// Source-major, query-major cotangents, flattened `[S,Q,T,H]`.
    pub values: Vec<f32>,
    pub diagnostics: Vec<MuseGlimmerBlockReplayDiagnostic>,
    pub timings: MuseGlimmerQueryBatchVjpTimings,
}

impl MuseGlimmerQueryBatchComposedVjp {
    pub fn source_values(&self, source_slot: usize) -> Option<&[f32]> {
        let elements = self
            .query_count
            .checked_mul(self.n_tokens)?
            .checked_mul(self.hidden_size)?;
        let start = source_slot.checked_mul(elements)?;
        self.values.get(start..start.checked_add(elements)?)
    }

    pub fn source_query_values(&self, source_slot: usize, query_slot: usize) -> Option<&[f32]> {
        if query_slot >= self.query_count {
            return None;
        }
        let query_elements = self.n_tokens.checked_mul(self.hidden_size)?;
        let source_elements = self.query_count.checked_mul(query_elements)?;
        let start = source_slot
            .checked_mul(source_elements)?
            .checked_add(query_slot.checked_mul(query_elements)?)?;
        self.values.get(start..start.checked_add(query_elements)?)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerAdjacentSelectedTokenFit {
    pub source_block: u32,
    pub target_block: u32,
    pub rule: MuseGlimmerLensRule,
    pub token_ids: Vec<u32>,
    pub n_valid_positions: usize,
    pub hidden_size: usize,
    /// Token-major fitted directions, flattened `[K,H]`.
    pub values: Vec<f32>,
    pub post_attention_replay_max_abs_error: f32,
    pub post_block_replay_max_abs_error: f32,
}

impl MuseGlimmerAdjacentSelectedTokenFit {
    pub fn token_values(&self, slot: usize) -> Option<&[f32]> {
        let start = slot.checked_mul(self.hidden_size)?;
        self.values.get(start..start.checked_add(self.hidden_size)?)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerAdjacentRowSlab {
    pub source_block: u32,
    pub target_block: u32,
    pub rule: MuseGlimmerLensRule,
    pub row_start: u32,
    pub row_end: u32,
    pub n_tokens: usize,
    pub n_valid_positions: usize,
    pub hidden_size: usize,
    /// Target-row-major fitted directions, flattened `[R,H]`.
    pub values: Vec<f32>,
    pub post_attention_replay_max_abs_error: f32,
    pub post_block_replay_max_abs_error: f32,
}

impl MuseGlimmerAdjacentRowSlab {
    pub fn row_values(&self, row: u32) -> Option<&[f32]> {
        let slot = row.checked_sub(self.row_start)? as usize;
        if row >= self.row_end {
            return None;
        }
        let start = slot.checked_mul(self.hidden_size)?;
        self.values.get(start..start.checked_add(self.hidden_size)?)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MuseGlimmerAttentionBlockKind {
    Full,
    Sliding,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerBlockReplayDiagnostic {
    pub block: u32,
    pub kind: MuseGlimmerAttentionBlockKind,
    pub post_attention_replay_max_abs_error: f32,
    pub post_block_replay_max_abs_error: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerMultiSourceSelectedTokenFit {
    pub source_layers: Vec<u32>,
    pub target_block: u32,
    pub method: MuseGlimmerLensRule,
    pub token_ids: Vec<u32>,
    pub n_valid_positions: usize,
    pub hidden_size: usize,
    /// Source-major, token-major fitted directions, flattened `[S,K,H]`.
    pub values: Vec<f32>,
    /// Reverse traversal order from target toward the earliest source.
    pub diagnostics: Vec<MuseGlimmerBlockReplayDiagnostic>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerFullTransportRowFit {
    pub source_layers: Vec<u32>,
    pub target_block: u32,
    pub method: MuseGlimmerLensRule,
    pub output_row_ids: Vec<u32>,
    pub n_valid_positions: usize,
    pub hidden_size: usize,
    /// Source-major, output-row-major fitted rows, flattened `[S,R,H]`.
    pub values: Vec<f32>,
    /// Reverse traversal order from target toward the earliest source.
    pub diagnostics: Vec<MuseGlimmerBlockReplayDiagnostic>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerBatchedFullTransportRowFit {
    pub source_layers: Vec<u32>,
    pub target_block: u32,
    pub method: MuseGlimmerLensRule,
    pub output_row_ids: Vec<u32>,
    pub n_valid_positions: usize,
    pub hidden_size: usize,
    pub query_batch_size: usize,
    /// Source-major, output-row-major fitted rows, flattened `[S,R,H]`.
    pub values: Vec<f32>,
    pub diagnostics: Vec<MuseGlimmerBlockReplayDiagnostic>,
    pub timings: MuseGlimmerQueryBatchVjpTimings,
}

impl MuseGlimmerFullTransportRowFit {
    pub fn source_values(&self, source_slot: usize) -> Option<&[f32]> {
        let elements = self.output_row_ids.len().checked_mul(self.hidden_size)?;
        let start = source_slot.checked_mul(elements)?;
        self.values.get(start..start.checked_add(elements)?)
    }

    pub fn source_row_values(&self, source_slot: usize, row_slot: usize) -> Option<&[f32]> {
        if row_slot >= self.output_row_ids.len() {
            return None;
        }
        let source_elements = self.output_row_ids.len().checked_mul(self.hidden_size)?;
        let start = source_slot
            .checked_mul(source_elements)?
            .checked_add(row_slot.checked_mul(self.hidden_size)?)?;
        self.values.get(start..start.checked_add(self.hidden_size)?)
    }
}

impl MuseGlimmerBatchedFullTransportRowFit {
    pub fn source_values(&self, source_slot: usize) -> Option<&[f32]> {
        let elements = self.output_row_ids.len().checked_mul(self.hidden_size)?;
        let start = source_slot.checked_mul(elements)?;
        self.values.get(start..start.checked_add(elements)?)
    }

    pub fn source_row_values(&self, source_slot: usize, row_slot: usize) -> Option<&[f32]> {
        if row_slot >= self.output_row_ids.len() {
            return None;
        }
        let source_elements = self.output_row_ids.len().checked_mul(self.hidden_size)?;
        let start = source_slot
            .checked_mul(source_elements)?
            .checked_add(row_slot.checked_mul(self.hidden_size)?)?;
        self.values.get(start..start.checked_add(self.hidden_size)?)
    }
}

impl MuseGlimmerMultiSourceSelectedTokenFit {
    pub fn source_values(&self, source_slot: usize) -> Option<&[f32]> {
        let elements = self.token_ids.len().checked_mul(self.hidden_size)?;
        let start = source_slot.checked_mul(elements)?;
        self.values.get(start..start.checked_add(elements)?)
    }

    pub fn source_token_values(&self, source_slot: usize, token_slot: usize) -> Option<&[f32]> {
        if token_slot >= self.token_ids.len() {
            return None;
        }
        let source_elements = self.token_ids.len().checked_mul(self.hidden_size)?;
        let start = source_slot
            .checked_mul(source_elements)?
            .checked_add(token_slot.checked_mul(self.hidden_size)?)?;
        self.values.get(start..start.checked_add(self.hidden_size)?)
    }
}

pub(crate) fn muse_glimmer_fit_selected_tokens_to_sources(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    captures: &MuseGlimmerLensCaptureBank,
    target_block: u32,
    source_layers: &[u32],
    covectors: &MuseGlimmerSelectedTokenCovectors,
    skip_first: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerMultiSourceSelectedTokenFit, MuseGlimmerLensError> {
    let traversal = validate_composition_request(
        weights,
        captures,
        target_block,
        source_layers,
        covectors.hidden_size(),
        covectors.token_ids().len(),
        Some(MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS),
    )?;
    let valid_positions = adjacent_fit_position_range(captures.n_tokens(), skip_first)?;
    let n_valid_positions = valid_positions.len();
    let mut diagnostics = composition_diagnostics(weights, &traversal);
    let values = compose_covector_rows_to_sources(
        source_layers,
        &traversal,
        covectors.values(),
        covectors.token_ids().len(),
        captures.n_tokens(),
        captures.hidden_size(),
        valid_positions,
        &mut diagnostics,
        |block, current| {
            let capture = captures.block_capture(block).ok_or_else(|| {
                MuseGlimmerLensError::Invalid(format!(
                    "capture bank is missing traversed block {block}"
                ))
            })?;
            muse_glimmer_one_attention_block_vjp(ctx, weights, &capture, current, rule)
        },
    )?;
    require_finite("multi-source selected-token fit", &values)?;
    Ok(MuseGlimmerMultiSourceSelectedTokenFit {
        source_layers: source_layers.to_vec(),
        target_block,
        method: rule,
        token_ids: covectors.token_ids().to_vec(),
        n_valid_positions,
        hidden_size: captures.hidden_size(),
        values,
        diagnostics,
    })
}

pub(crate) fn muse_glimmer_fit_full_transport_rows_to_sources(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    captures: &MuseGlimmerLensCaptureBank,
    target_block: u32,
    source_layers: &[u32],
    output_row_ids: &[u32],
    skip_first: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerFullTransportRowFit, MuseGlimmerLensError> {
    let hidden_size = captures.hidden_size();
    validate_output_row_ids(output_row_ids, hidden_size)?;
    let traversal = validate_composition_request(
        weights,
        captures,
        target_block,
        source_layers,
        hidden_size,
        output_row_ids.len(),
        None,
    )?;
    let valid_positions = adjacent_fit_position_range(captures.n_tokens(), skip_first)?;
    let n_valid_positions = valid_positions.len();
    let row_elements = checked_mul(output_row_ids.len(), hidden_size, "basis covector elements")?;
    let mut basis_covectors = vec![0.0_f32; row_elements];
    for (slot, &row) in output_row_ids.iter().enumerate() {
        basis_covectors[slot * hidden_size + row as usize] = 1.0;
    }
    let mut diagnostics = composition_diagnostics(weights, &traversal);
    let values = compose_covector_rows_to_sources(
        source_layers,
        &traversal,
        &basis_covectors,
        output_row_ids.len(),
        captures.n_tokens(),
        hidden_size,
        valid_positions,
        &mut diagnostics,
        |block, current| {
            let capture = captures.block_capture(block).ok_or_else(|| {
                MuseGlimmerLensError::Invalid(format!(
                    "capture bank is missing traversed block {block}"
                ))
            })?;
            muse_glimmer_one_attention_block_vjp(ctx, weights, &capture, current, rule)
        },
    )?;
    require_finite("full-transport row fit", &values)?;
    Ok(MuseGlimmerFullTransportRowFit {
        source_layers: source_layers.to_vec(),
        target_block,
        method: rule,
        output_row_ids: output_row_ids.to_vec(),
        n_valid_positions,
        hidden_size,
        values,
        diagnostics,
    })
}

pub(crate) fn muse_glimmer_composed_vjp_query_batch(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    captures: &MuseGlimmerLensCaptureBank,
    target_block: u32,
    source_layers: &[u32],
    target_cotangents: &[f32],
    query_count: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerQueryBatchComposedVjp, MuseGlimmerLensError> {
    let hidden_size = captures.hidden_size();
    let traversal = validate_composition_request(
        weights,
        captures,
        target_block,
        source_layers,
        hidden_size,
        query_count,
        Some(MUSE_GLIMMER_QUERY_BATCH_MAX),
    )?;
    let query_elements = checked_mul(captures.n_tokens(), hidden_size, "composed query elements")?;
    validate_len(
        "composed query-major target cotangents",
        target_cotangents,
        checked_mul(query_count, query_elements, "composed target elements")?,
    )?;
    require_finite("composed query-major target cotangents", target_cotangents)?;
    let source_elements = checked_mul(query_count, query_elements, "composed source elements")?;
    let mut values = vec![
        0.0_f32;
        checked_mul(
            source_layers.len(),
            source_elements,
            "composed output elements"
        )?
    ];
    let mut diagnostics = composition_diagnostics(weights, &traversal);
    let mut timings = MuseGlimmerQueryBatchVjpTimings::default();
    let mut current = target_cotangents.to_vec();
    for (diagnostic_slot, &block) in traversal.iter().enumerate() {
        let capture = captures.block_capture(block).ok_or_else(|| {
            MuseGlimmerLensError::Invalid(format!(
                "capture bank is missing traversed block {block}"
            ))
        })?;
        let vjp = if weights.layers[block as usize].sliding_attention {
            muse_glimmer_one_attention_block_vjp_query_batch(
                ctx,
                weights,
                &capture,
                &current,
                query_count,
                rule,
            )?
        } else {
            muse_glimmer_one_full_attention_block_vjp_query_batch(
                ctx,
                weights,
                &capture,
                &current,
                query_count,
                rule,
            )?
        };
        current = vjp.input_cotangents;
        timings.add_assign(vjp.timings);
        let diagnostic = &mut diagnostics[diagnostic_slot];
        diagnostic.post_attention_replay_max_abs_error = vjp.post_attention_replay_max_abs_error;
        diagnostic.post_block_replay_max_abs_error = vjp.post_block_replay_max_abs_error;
        if let Ok(source_slot) = source_layers.binary_search(&(block - 1)) {
            let destination = source_slot * source_elements;
            values[destination..destination + source_elements].copy_from_slice(&current);
        }
    }
    require_finite("composed query-batch source cotangents", &values)?;
    Ok(MuseGlimmerQueryBatchComposedVjp {
        source_layers: source_layers.to_vec(),
        target_block,
        method: rule,
        query_count,
        n_tokens: captures.n_tokens(),
        hidden_size,
        values,
        diagnostics,
        timings,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn muse_glimmer_fit_full_transport_rows_to_sources_batched(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    captures: &MuseGlimmerLensCaptureBank,
    target_block: u32,
    source_layers: &[u32],
    output_row_ids: &[u32],
    skip_first: usize,
    query_batch_size: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerBatchedFullTransportRowFit, MuseGlimmerLensError> {
    let total_started = Instant::now();
    let hidden_size = captures.hidden_size();
    validate_output_row_ids(output_row_ids, hidden_size)?;
    if query_batch_size == 0 || query_batch_size > MUSE_GLIMMER_QUERY_BATCH_MAX {
        return invalid(format!(
            "full-transport query batch size must be in 1..={MUSE_GLIMMER_QUERY_BATCH_MAX}, got {query_batch_size}"
        ));
    }
    let traversal = validate_composition_request(
        weights,
        captures,
        target_block,
        source_layers,
        hidden_size,
        output_row_ids.len(),
        None,
    )?;
    let valid_positions = adjacent_fit_position_range(captures.n_tokens(), skip_first)?;
    let n_valid_positions = valid_positions.len();
    let source_elements = checked_mul(output_row_ids.len(), hidden_size, "batched fit source")?;
    let mut values =
        vec![0.0_f32; checked_mul(source_layers.len(), source_elements, "batched fit output")?];
    let mut diagnostics = composition_diagnostics(weights, &traversal);
    let mut timings = MuseGlimmerQueryBatchVjpTimings::default();
    let query_elements = checked_mul(captures.n_tokens(), hidden_size, "fit query elements")?;
    let mut banks = Vec::with_capacity(output_row_ids.len().div_ceil(query_batch_size));
    for (chunk_slot, rows) in output_row_ids.chunks(query_batch_size).enumerate() {
        let mut targets = vec![0.0_f32; checked_mul(rows.len(), query_elements, "fit targets")?];
        for (query, &row) in rows.iter().enumerate() {
            for position in valid_positions.clone() {
                targets[query * query_elements + position * hidden_size + row as usize] = 1.0;
            }
        }
        banks.push(FullTransportQueryBank {
            first_row_slot: chunk_slot * query_batch_size,
            query_count: rows.len(),
            current: targets,
        });
    }

    for (diagnostic_slot, &block) in traversal.iter().enumerate() {
        let capture = captures.block_capture(block).ok_or_else(|| {
            MuseGlimmerLensError::Invalid(format!(
                "capture bank is missing traversed block {block}"
            ))
        })?;
        let replay_started = Instant::now();
        let prepared = if weights.layers[block as usize].sliding_attention {
            MuseGlimmerPreparedMetalAttentionBlock::new_sliding(ctx, weights, &capture)?
        } else {
            MuseGlimmerPreparedMetalAttentionBlock::new(ctx, weights, &capture)?
        };
        timings.replay += replay_started.elapsed();
        let (post_attention, post_block) = prepared.replay_max_abs_errors();
        let diagnostic = &mut diagnostics[diagnostic_slot];
        diagnostic.post_attention_replay_max_abs_error = post_attention;
        diagnostic.post_block_replay_max_abs_error = post_block;
        for bank in &mut banks {
            let vjp = reverse_prepared_metal_attention_block_vjp_query_batch(
                ctx,
                &prepared,
                &bank.current,
                bank.query_count,
                rule,
            )?;
            timings.add_components_assign(vjp.timings);
            bank.current = vjp.input_cotangents;
        }

        if let Ok(source_slot) = source_layers.binary_search(&(block - 1)) {
            for bank in &banks {
                for query in 0..bank.query_count {
                    let query_start = query * query_elements;
                    let source_query = &bank.current[query_start..query_start + query_elements];
                    let reduced = mean_reduce_position_rows(
                        source_query,
                        captures.n_tokens(),
                        hidden_size,
                        valid_positions.clone(),
                    )?;
                    let row_slot = bank.first_row_slot + query;
                    let destination = source_slot * source_elements + row_slot * hidden_size;
                    values[destination..destination + hidden_size].copy_from_slice(&reduced);
                }
            }
        }
    }
    require_finite("batched full-transport row fit", &values)?;
    timings.total = total_started.elapsed();
    Ok(MuseGlimmerBatchedFullTransportRowFit {
        source_layers: source_layers.to_vec(),
        target_block,
        method: rule,
        output_row_ids: output_row_ids.to_vec(),
        n_valid_positions,
        hidden_size,
        query_batch_size,
        values,
        diagnostics,
        timings,
    })
}

struct FullTransportQueryBank {
    first_row_slot: usize,
    query_count: usize,
    current: Vec<f32>,
}

fn validate_composition_request(
    weights: &MuseGlimmerMetalModelWeights<'_>,
    captures: &MuseGlimmerLensCaptureBank,
    target_block: u32,
    source_layers: &[u32],
    covector_hidden_size: usize,
    covector_count: usize,
    max_covectors: Option<usize>,
) -> Result<Vec<u32>, MuseGlimmerLensError> {
    if target_block == 0 || target_block as usize >= weights.layers.len() {
        return invalid(format!("invalid composition target block {target_block}"));
    }
    let traversal = composition_traversal(target_block, source_layers)?;
    if captures.hidden_size() != weights.config.hidden_size as usize
        || covector_hidden_size != captures.hidden_size()
        || captures.token_ids().is_empty()
        || covector_count == 0
        || max_covectors.is_some_and(|maximum| covector_count > maximum)
    {
        return invalid("capture/covector dimensions or counts do not match composition contract");
    }
    if let Some(&missing) = traversal
        .iter()
        .find(|&&block| captures.block_slot(block).is_none())
    {
        return invalid(format!(
            "capture bank is missing required traversed block {missing}"
        ));
    }
    Ok(traversal)
}

fn validate_output_row_ids(
    output_row_ids: &[u32],
    hidden_size: usize,
) -> Result<(), MuseGlimmerLensError> {
    if output_row_ids.is_empty() {
        return invalid("full-transport fit requires at least one output row");
    }
    if output_row_ids.len() > MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD {
        return invalid(format!(
            "full-transport fit supports at most {MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD} rows per shard, got {}",
            output_row_ids.len()
        ));
    }
    if output_row_ids.windows(2).any(|rows| rows[0] >= rows[1]) {
        return invalid("output row IDs must be strictly increasing");
    }
    if let Some(&row) = output_row_ids
        .iter()
        .find(|&&row| row as usize >= hidden_size)
    {
        return invalid(format!(
            "output row {row} is outside hidden size {hidden_size}"
        ));
    }
    Ok(())
}

fn composition_diagnostics(
    weights: &MuseGlimmerMetalModelWeights<'_>,
    traversal: &[u32],
) -> Vec<MuseGlimmerBlockReplayDiagnostic> {
    traversal
        .iter()
        .map(|&block| MuseGlimmerBlockReplayDiagnostic {
            block,
            kind: if weights.layers[block as usize].sliding_attention {
                MuseGlimmerAttentionBlockKind::Sliding
            } else {
                MuseGlimmerAttentionBlockKind::Full
            },
            post_attention_replay_max_abs_error: 0.0,
            post_block_replay_max_abs_error: 0.0,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn compose_covector_rows_to_sources<F>(
    source_layers: &[u32],
    traversal: &[u32],
    covectors: &[f32],
    n_covectors: usize,
    n_tokens: usize,
    hidden_size: usize,
    valid_positions: std::ops::Range<usize>,
    diagnostics: &mut [MuseGlimmerBlockReplayDiagnostic],
    mut reverse_block: F,
) -> Result<Vec<f32>, MuseGlimmerLensError>
where
    F: FnMut(u32, &[f32]) -> Result<MuseGlimmerOneBlockVjp, MuseGlimmerLensError>,
{
    let source_elements = checked_mul(n_covectors, hidden_size, "source fit elements")?;
    validate_len("target covector rows", covectors, source_elements)?;
    if diagnostics.len() != traversal.len() {
        return invalid("composition diagnostic schedule differs from traversal");
    }
    let output_elements = checked_mul(source_layers.len(), source_elements, "source fit output")?;
    let mut values = vec![0.0_f32; output_elements];
    let positions = valid_positions.clone().collect::<Vec<_>>();
    for covector_slot in 0..n_covectors {
        let start = covector_slot * hidden_size;
        let mut current = muse_glimmer_positioned_target_cotangent(
            &covectors[start..start + hidden_size],
            n_tokens,
            hidden_size,
            &positions,
        )?;
        for (diagnostic_slot, &block) in traversal.iter().enumerate() {
            let vjp = reverse_block(block, &current)?;
            current = vjp.input_cotangent;
            let diagnostic = &mut diagnostics[diagnostic_slot];
            diagnostic.post_attention_replay_max_abs_error = diagnostic
                .post_attention_replay_max_abs_error
                .max(vjp.post_attention_replay_max_abs_error);
            diagnostic.post_block_replay_max_abs_error = diagnostic
                .post_block_replay_max_abs_error
                .max(vjp.post_block_replay_max_abs_error);
            let reached_source = block - 1;
            if let Ok(source_slot) = source_layers.binary_search(&reached_source) {
                let reduced = mean_reduce_position_rows(
                    &current,
                    n_tokens,
                    hidden_size,
                    valid_positions.clone(),
                )?;
                let destination = source_slot * source_elements + covector_slot * hidden_size;
                values[destination..destination + hidden_size].copy_from_slice(&reduced);
            }
        }
    }
    Ok(values)
}

fn composition_traversal(
    target_block: u32,
    source_layers: &[u32],
) -> Result<Vec<u32>, MuseGlimmerLensError> {
    if source_layers.is_empty() {
        return invalid("multi-source fit requires at least one source layer");
    }
    for (slot, &source) in source_layers.iter().enumerate() {
        if source >= target_block {
            return invalid(format!(
                "source layer {source} must be below target block {target_block}"
            ));
        }
        if slot > 0 && source_layers[slot - 1] >= source {
            return invalid("source layers must be strictly increasing");
        }
    }
    let earliest = source_layers[0];
    Ok(((earliest + 1)..=target_block).rev().collect())
}

pub(crate) fn muse_glimmer_fit_adjacent_full_attention_selected_tokens(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    capture: &MuseGlimmerLensCapture,
    covectors: &MuseGlimmerSelectedTokenCovectors,
    skip_first: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerAdjacentSelectedTokenFit, MuseGlimmerLensError> {
    let target_block = capture.target_block();
    let source_block = target_block.checked_sub(1).ok_or_else(|| {
        MuseGlimmerLensError::Invalid("adjacent fit requires a nonzero target block".into())
    })?;
    if covectors.token_ids().is_empty()
        || covectors.token_ids().len() > MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS
    {
        return invalid(format!(
            "adjacent fit selected-token count must be in 1..={MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS}, got {}",
            covectors.token_ids().len()
        ));
    }
    if covectors.hidden_size() != capture.hidden_size() {
        return invalid(format!(
            "covector hidden size {} differs from capture hidden size {}",
            covectors.hidden_size(),
            capture.hidden_size()
        ));
    }
    let valid_positions = adjacent_fit_position_range(capture.n_tokens(), skip_first)?;
    let n_valid_positions = valid_positions.len();
    let positions = valid_positions.clone().collect::<Vec<_>>();
    let mut values = Vec::with_capacity(covectors.token_ids().len() * capture.hidden_size());
    let mut post_attention_replay_max_abs_error = 0.0_f32;
    let mut post_block_replay_max_abs_error = 0.0_f32;
    for slot in 0..covectors.token_ids().len() {
        let covector = covectors.token_values(slot).ok_or_else(|| {
            MuseGlimmerLensError::Invalid(format!("selected-token covector slot {slot} is missing"))
        })?;
        let target_cotangent = muse_glimmer_positioned_target_cotangent(
            covector,
            capture.n_tokens(),
            capture.hidden_size(),
            &positions,
        )?;
        let vjp = muse_glimmer_one_full_attention_block_vjp(
            ctx,
            weights,
            capture,
            &target_cotangent,
            rule,
        )?;
        values.extend(mean_reduce_position_rows(
            &vjp.input_cotangent,
            capture.n_tokens(),
            capture.hidden_size(),
            valid_positions.clone(),
        )?);
        post_attention_replay_max_abs_error =
            post_attention_replay_max_abs_error.max(vjp.post_attention_replay_max_abs_error);
        post_block_replay_max_abs_error =
            post_block_replay_max_abs_error.max(vjp.post_block_replay_max_abs_error);
    }
    require_finite("adjacent selected-token fit", &values)?;
    Ok(MuseGlimmerAdjacentSelectedTokenFit {
        source_block,
        target_block,
        rule,
        token_ids: covectors.token_ids().to_vec(),
        n_valid_positions,
        hidden_size: capture.hidden_size(),
        values,
        post_attention_replay_max_abs_error,
        post_block_replay_max_abs_error,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn muse_glimmer_fit_adjacent_full_attention_rows_batched(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    capture: &MuseGlimmerLensCapture,
    rows: std::ops::Range<u32>,
    skip_first: usize,
    dim_batch: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerAdjacentRowSlab, MuseGlimmerLensError> {
    let hidden_size = capture.hidden_size();
    let hidden_size_u32 = u32::try_from(hidden_size)
        .map_err(|_| MuseGlimmerLensError::Invalid("hidden size exceeds u32".into()))?;
    if rows.start >= rows.end || rows.end > hidden_size_u32 {
        return invalid(format!(
            "adjacent row range {}..{} must be nonempty and within hidden size {hidden_size}",
            rows.start, rows.end
        ));
    }
    if dim_batch == 0 || dim_batch > MUSE_GLIMMER_FULL_R_MAX_DIM_BATCH {
        return invalid(format!(
            "adjacent row dim_batch must be in 1..={MUSE_GLIMMER_FULL_R_MAX_DIM_BATCH}, got {dim_batch}"
        ));
    }
    let valid_positions = adjacent_fit_position_range(capture.n_tokens(), skip_first)?;
    let n_valid_positions = valid_positions.len();
    let prepared = MuseGlimmerPreparedMetalAttentionBlock::new(ctx, weights, capture)?;
    let row_count = usize::try_from(rows.end - rows.start)
        .map_err(|_| MuseGlimmerLensError::Invalid("row count exceeds usize".into()))?;
    let basis_count = dim_batch.min(row_count);
    let trajectory_elements = checked_mul(
        checked_mul(basis_count, capture.n_tokens(), "row slab trajectories")?,
        hidden_size,
        "row slab trajectory elements",
    )?;
    let shape = row_shape(hidden_size, basis_count * capture.n_tokens())?;
    let current = MetalTensor::zeros_f32(ctx, shape.clone())?;
    let next = MetalTensor::zeros_f32(ctx, shape)?;
    let mut workspace = MuseGlimmerMetalAttentionBankWorkspace::new(ctx, &prepared, basis_count)?;
    let mut current_values = vec![0.0_f32; trajectory_elements];
    let value_count = checked_mul(row_count, hidden_size, "adjacent row slab values")?;
    let mut values = Vec::with_capacity(value_count);

    let mut chunk_start = rows.start;
    while chunk_start < rows.end {
        let chunk_end = rows.end.min(chunk_start.saturating_add(basis_count as u32));
        let active_basis = usize::try_from(chunk_end - chunk_start)
            .map_err(|_| MuseGlimmerLensError::Invalid("active row count exceeds usize".into()))?;
        current_values.fill(0.0);
        for basis in 0..active_basis {
            let row = usize::try_from(chunk_start)
                .ok()
                .and_then(|start| start.checked_add(basis))
                .ok_or_else(|| MuseGlimmerLensError::Invalid("target row overflow".into()))?;
            for position in valid_positions.clone() {
                let index = (basis * capture.n_tokens() + position) * hidden_size + row;
                current_values[index] = 1.0;
            }
        }
        write_f32(&current, &current_values)?;
        run_command(ctx, |encoder| {
            prepared.encode_bank(ctx, encoder, &current, &next, &mut workspace, rule)
        })?;
        let output = read_f32(&next);
        let trajectory_stride = checked_mul(
            capture.n_tokens(),
            hidden_size,
            "row slab trajectory stride",
        )?;
        for basis in 0..active_basis {
            let start = checked_mul(basis, trajectory_stride, "row slab output offset")?;
            values.extend(mean_reduce_position_rows(
                &output[start..start + trajectory_stride],
                capture.n_tokens(),
                hidden_size,
                valid_positions.clone(),
            )?);
        }
        chunk_start = chunk_end;
    }
    require_finite("adjacent row slab", &values)?;
    validate_len("adjacent row slab", &values, value_count)?;
    let (post_attention_replay_max_abs_error, post_block_replay_max_abs_error) =
        prepared.replay_max_abs_errors();
    Ok(MuseGlimmerAdjacentRowSlab {
        source_block: prepared.target_block() - 1,
        target_block: prepared.target_block(),
        rule,
        row_start: rows.start,
        row_end: rows.end,
        n_tokens: prepared.n_tokens(),
        n_valid_positions,
        hidden_size: prepared.hidden_size(),
        values,
        post_attention_replay_max_abs_error,
        post_block_replay_max_abs_error,
    })
}

fn adjacent_fit_position_range(
    n_tokens: usize,
    skip_first: usize,
) -> Result<std::ops::Range<usize>, MuseGlimmerLensError> {
    let final_position = n_tokens.checked_sub(1).ok_or_else(|| {
        MuseGlimmerLensError::Invalid("adjacent fit requires at least two prompt tokens".into())
    })?;
    if skip_first >= final_position {
        return invalid(format!(
            "skip_first {skip_first} leaves no positions before final prompt position {final_position}"
        ));
    }
    Ok(skip_first..final_position)
}

fn mean_reduce_position_rows(
    values: &[f32],
    n_tokens: usize,
    hidden_size: usize,
    positions: std::ops::Range<usize>,
) -> Result<Vec<f32>, MuseGlimmerLensError> {
    validate_len(
        "source cotangent bank",
        values,
        checked_mul(n_tokens, hidden_size, "source cotangent elements")?,
    )?;
    if positions.is_empty() || positions.end > n_tokens {
        return invalid("source reduction positions are empty or out of range");
    }
    let count = positions.len();
    let mut reduced = vec![0.0_f32; hidden_size];
    for position in positions {
        let row = &values[position * hidden_size..(position + 1) * hidden_size];
        for (destination, &value) in reduced.iter_mut().zip(row) {
            *destination += value;
        }
    }
    let scale = (count as f32).recip();
    for value in &mut reduced {
        *value *= scale;
    }
    require_finite("mean-reduced source cotangent", &reduced)?;
    Ok(reduced)
}

/// Place one `[H]` score covector at selected rows of a zeroed `[T,H]` bank.
pub fn muse_glimmer_positioned_target_cotangent(
    covector: &[f32],
    n_tokens: usize,
    hidden_size: usize,
    positions: &[usize],
) -> Result<Vec<f32>, MuseGlimmerLensError> {
    if hidden_size == 0 || covector.len() != hidden_size {
        return invalid(format!(
            "target covector length {} differs from nonzero hidden size {hidden_size}",
            covector.len()
        ));
    }
    if n_tokens == 0 || n_tokens > MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS || positions.is_empty() {
        return invalid(format!(
            "target bank requires 1..={MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS} tokens and at least one position"
        ));
    }
    require_finite("target covector", covector)?;
    let mut values = vec![0.0_f32; checked_mul(n_tokens, hidden_size, "target bank elements")?];
    for (slot, &position) in positions.iter().enumerate() {
        if position >= n_tokens {
            return invalid(format!(
                "target position {position} is outside token count {n_tokens}"
            ));
        }
        if positions[..slot].contains(&position) {
            return invalid(format!("target position {position} occurs more than once"));
        }
        let start = position * hidden_size;
        values[start..start + hidden_size].copy_from_slice(covector);
    }
    Ok(values)
}

#[derive(Clone, Copy)]
struct MuseAttentionGeometry {
    hidden: usize,
    feed_forward: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query: usize,
    kv: usize,
    rope_theta: f32,
}

impl MuseAttentionGeometry {
    fn from_weights(
        weights: &MuseGlimmerMetalModelWeights<'_>,
    ) -> Result<Self, MuseGlimmerLensError> {
        let config = weights.config;
        let geometry = Self {
            hidden: config.hidden_size as usize,
            feed_forward: config.feed_forward_size as usize,
            q_heads: config.query_head_count as usize,
            kv_heads: config.kv_head_count as usize,
            head_dim: config.key_head_dim as usize,
            query: usize::try_from(config.query_width()?)
                .map_err(|_| MuseGlimmerLensError::Invalid("query width exceeds usize".into()))?,
            kv: usize::try_from(config.kv_width()?)
                .map_err(|_| MuseGlimmerLensError::Invalid("KV width exceeds usize".into()))?,
            rope_theta: config.rope_theta,
        };
        if geometry.hidden == 0
            || geometry.feed_forward == 0
            || geometry.q_heads == 0
            || geometry.kv_heads == 0
            || geometry.head_dim == 0
            || !geometry.q_heads.is_multiple_of(geometry.kv_heads)
            || geometry.query != geometry.q_heads * geometry.head_dim
            || geometry.kv != geometry.kv_heads * geometry.head_dim
            || !geometry.rope_theta.is_finite()
            || geometry.rope_theta <= 0.0
        {
            return invalid("inconsistent Muse attention geometry");
        }
        Ok(geometry)
    }
}

struct MuseAttentionForward {
    attention_output: Vec<f32>,
    gated_output: Vec<f32>,
    probabilities: Vec<f32>,
}

struct MuseAttentionVjp {
    grad_q: Vec<f32>,
    grad_k: Vec<f32>,
    grad_v: Vec<f32>,
    grad_gate: Vec<f32>,
}

struct MuseAttentionQueryBatchVjp {
    /// Primal-head-major/query-minor rows `[T,QH,Q,D]`.
    grad_q_head_query: Vec<f32>,
    /// Primal-head-major/query-minor rows `[T,KVH,Q,D]`.
    grad_k_head_query: Vec<f32>,
    /// Token-major/query-minor rows `[T,Q,KV]`.
    grad_v_token_query: Vec<f32>,
    /// Token-major/query-minor rows `[T,Q,QW]`.
    grad_gate_token_query: Vec<f32>,
}

fn adjacent_pair_rope_rows_in_place(
    values: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    inverse: bool,
) -> Result<(), MuseGlimmerLensError> {
    if n_tokens == 0
        || n_heads == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || !theta.is_finite()
        || theta <= 0.0
    {
        return invalid(
            "adjacent-pair RoPE requires nonzero rows/heads, even head_dim, and positive finite theta",
        );
    }
    validate_len(
        "adjacent-pair RoPE rows",
        values,
        checked_mul(
            checked_mul(n_tokens, n_heads, "RoPE token-head count")?,
            head_dim,
            "RoPE element count",
        )?,
    )?;
    for token in 0..n_tokens {
        for head in 0..n_heads {
            let base = (token * n_heads + head) * head_dim;
            for pair in 0..head_dim / 2 {
                let relative = 2 * pair;
                let angle = token as f32 * theta.powf(-(relative as f32) / head_dim as f32);
                let (mut sine, cosine) = angle.sin_cos();
                if inverse {
                    sine = -sine;
                }
                let first = values[base + relative];
                let second = values[base + relative + 1];
                values[base + relative] = first * cosine - second * sine;
                values[base + relative + 1] = first * sine + second * cosine;
            }
        }
    }
    require_finite("adjacent-pair RoPE output", values)
}

#[allow(clippy::needless_range_loop)]
fn cpu_causal_gqa_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    n_tokens: usize,
    geometry: MuseAttentionGeometry,
) -> Result<MuseAttentionForward, MuseGlimmerLensError> {
    let q_total = checked_mul(n_tokens, geometry.query, "Q elements")?;
    let kv_total = checked_mul(n_tokens, geometry.kv, "KV elements")?;
    validate_len("attention Q", q, q_total)?;
    validate_len("attention K", k, kv_total)?;
    validate_len("attention V", v, kv_total)?;
    validate_len("attention gate", gate, q_total)?;
    require_finite("attention Q", q)?;
    require_finite("attention K", k)?;
    require_finite("attention V", v)?;
    require_finite("attention gate", gate)?;

    let group = geometry.q_heads / geometry.kv_heads;
    let scale = (geometry.head_dim as f32).sqrt().recip();
    let mut attention_output = vec![0.0_f32; q_total];
    let probability_rows = checked_mul(n_tokens, geometry.q_heads, "probability rows")?;
    let mut probability_bank =
        vec![0.0_f32; checked_mul(probability_rows, n_tokens, "probability elements")?];
    for token in 0..n_tokens {
        for q_head in 0..geometry.q_heads {
            let kv_head = q_head / group;
            let q_base = (token * geometry.q_heads + q_head) * geometry.head_dim;
            let mut probabilities = vec![0.0_f32; token + 1];
            for key_token in 0..=token {
                let k_base = (key_token * geometry.kv_heads + kv_head) * geometry.head_dim;
                let mut score = 0.0_f32;
                for dim in 0..geometry.head_dim {
                    score += q[q_base + dim] * k[k_base + dim];
                }
                probabilities[key_token] = score * scale;
            }
            softmax_in_place(&mut probabilities);
            let probability_base = (token * geometry.q_heads + q_head) * n_tokens;
            probability_bank[probability_base..probability_base + probabilities.len()]
                .copy_from_slice(&probabilities);
            for key_token in 0..=token {
                let v_base = (key_token * geometry.kv_heads + kv_head) * geometry.head_dim;
                for dim in 0..geometry.head_dim {
                    attention_output[q_base + dim] += probabilities[key_token] * v[v_base + dim];
                }
            }
        }
    }
    let gated_output = attention_output
        .iter()
        .zip(gate)
        .map(|(&attention, &gate)| attention * sigmoid(gate))
        .collect::<Vec<_>>();
    require_finite("attention output", &attention_output)?;
    require_finite("gated attention output", &gated_output)?;
    require_finite("attention probabilities", &probability_bank)?;
    Ok(MuseAttentionForward {
        attention_output,
        gated_output,
        probabilities: probability_bank,
    })
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn cpu_causal_gqa_vjp(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    grad_gated: &[f32],
    n_tokens: usize,
    geometry: MuseAttentionGeometry,
    forward: &MuseAttentionForward,
) -> Result<MuseAttentionVjp, MuseGlimmerLensError> {
    let q_total = checked_mul(n_tokens, geometry.query, "Q elements")?;
    let kv_total = checked_mul(n_tokens, geometry.kv, "KV elements")?;
    validate_len("attention cotangent", grad_gated, q_total)?;
    validate_len("attention forward", &forward.attention_output, q_total)?;
    let group = geometry.q_heads / geometry.kv_heads;
    let scale = (geometry.head_dim as f32).sqrt().recip();
    let mut grad_q = vec![0.0_f32; q_total];
    let mut grad_k = vec![0.0_f32; kv_total];
    let mut grad_v = vec![0.0_f32; kv_total];
    let mut grad_gate = vec![0.0_f32; q_total];
    for token in 0..n_tokens {
        for q_head in 0..geometry.q_heads {
            let kv_head = q_head / group;
            let q_base = (token * geometry.q_heads + q_head) * geometry.head_dim;
            let mut grad_attention = vec![0.0_f32; geometry.head_dim];
            for dim in 0..geometry.head_dim {
                let gate_value = sigmoid(gate[q_base + dim]);
                grad_attention[dim] = grad_gated[q_base + dim] * gate_value;
                grad_gate[q_base + dim] = grad_gated[q_base + dim]
                    * forward.attention_output[q_base + dim]
                    * gate_value
                    * (1.0 - gate_value);
            }
            let mut probabilities = vec![0.0_f32; token + 1];
            for key_token in 0..=token {
                let k_base = (key_token * geometry.kv_heads + kv_head) * geometry.head_dim;
                let mut score = 0.0_f32;
                for dim in 0..geometry.head_dim {
                    score += q[q_base + dim] * k[k_base + dim];
                }
                probabilities[key_token] = score * scale;
            }
            softmax_in_place(&mut probabilities);
            let mut grad_probability = vec![0.0_f32; token + 1];
            for key_token in 0..=token {
                let v_base = (key_token * geometry.kv_heads + kv_head) * geometry.head_dim;
                for dim in 0..geometry.head_dim {
                    grad_probability[key_token] += grad_attention[dim] * v[v_base + dim];
                    grad_v[v_base + dim] += probabilities[key_token] * grad_attention[dim];
                }
            }
            let probability_dot = probabilities
                .iter()
                .zip(&grad_probability)
                .map(|(&probability, &gradient)| probability * gradient)
                .sum::<f32>();
            for key_token in 0..=token {
                let grad_score =
                    probabilities[key_token] * (grad_probability[key_token] - probability_dot);
                let k_base = (key_token * geometry.kv_heads + kv_head) * geometry.head_dim;
                for dim in 0..geometry.head_dim {
                    grad_q[q_base + dim] += scale * grad_score * k[k_base + dim];
                    grad_k[k_base + dim] += scale * grad_score * q[q_base + dim];
                }
            }
        }
    }
    for (name, values) in [
        ("attention grad Q", grad_q.as_slice()),
        ("attention grad K", grad_k.as_slice()),
        ("attention grad V", grad_v.as_slice()),
        ("attention grad gate", grad_gate.as_slice()),
    ] {
        require_finite(name, values)?;
    }
    Ok(MuseAttentionVjp {
        grad_q,
        grad_k,
        grad_v,
        grad_gate,
    })
}

#[allow(clippy::too_many_arguments)]
fn cpu_causal_gqa_vjp_query_batch(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    grad_gated_token_query: &[f32],
    query_count: usize,
    n_tokens: usize,
    geometry: MuseAttentionGeometry,
    forward: &MuseAttentionForward,
    inverse_rope: bool,
) -> Result<MuseAttentionQueryBatchVjp, MuseGlimmerLensError> {
    let tq = checked_mul(n_tokens, query_count, "attention batch rows")?;
    validate_len(
        "batched attention cotangent",
        grad_gated_token_query,
        checked_mul(tq, geometry.query, "batched attention cotangent elements")?,
    )?;
    let q_head_rows = checked_mul(
        checked_mul(n_tokens, geometry.q_heads, "Q primal head rows")?,
        query_count,
        "Q query head rows",
    )?;
    let k_head_rows = checked_mul(
        checked_mul(n_tokens, geometry.kv_heads, "K primal head rows")?,
        query_count,
        "K query head rows",
    )?;
    let mut output = MuseAttentionQueryBatchVjp {
        grad_q_head_query: vec![0.0; checked_mul(q_head_rows, geometry.head_dim, "Q batch")?],
        grad_k_head_query: vec![0.0; checked_mul(k_head_rows, geometry.head_dim, "K batch")?],
        grad_v_token_query: vec![0.0; checked_mul(tq, geometry.kv, "V batch")?],
        grad_gate_token_query: vec![0.0; checked_mul(tq, geometry.query, "gate batch")?],
    };
    let queries = (0..query_count)
        .into_par_iter()
        .map(|query_slot| {
            let mut one_grad_gated = vec![0.0_f32; n_tokens * geometry.query];
            for token in 0..n_tokens {
                let source = (token * query_count + query_slot) * geometry.query;
                let destination = token * geometry.query;
                one_grad_gated[destination..destination + geometry.query]
                    .copy_from_slice(&grad_gated_token_query[source..source + geometry.query]);
            }
            let mut one =
                cpu_causal_gqa_vjp(q, k, v, gate, &one_grad_gated, n_tokens, geometry, forward)?;
            if inverse_rope {
                adjacent_pair_rope_rows_in_place(
                    &mut one.grad_q,
                    n_tokens,
                    geometry.q_heads,
                    geometry.head_dim,
                    geometry.rope_theta,
                    true,
                )?;
                adjacent_pair_rope_rows_in_place(
                    &mut one.grad_k,
                    n_tokens,
                    geometry.kv_heads,
                    geometry.head_dim,
                    geometry.rope_theta,
                    true,
                )?;
            }
            Ok::<_, MuseGlimmerLensError>(one)
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (query_slot, one) in queries.into_iter().enumerate() {
        for token in 0..n_tokens {
            for head in 0..geometry.q_heads {
                let source = (token * geometry.q_heads + head) * geometry.head_dim;
                let destination = ((token * geometry.q_heads + head) * query_count + query_slot)
                    * geometry.head_dim;
                output.grad_q_head_query[destination..destination + geometry.head_dim]
                    .copy_from_slice(&one.grad_q[source..source + geometry.head_dim]);
            }
            for head in 0..geometry.kv_heads {
                let source = (token * geometry.kv_heads + head) * geometry.head_dim;
                let destination = ((token * geometry.kv_heads + head) * query_count + query_slot)
                    * geometry.head_dim;
                output.grad_k_head_query[destination..destination + geometry.head_dim]
                    .copy_from_slice(&one.grad_k[source..source + geometry.head_dim]);
            }
            for (width, source_values, destination_values) in [
                (
                    geometry.kv,
                    one.grad_v.as_slice(),
                    &mut output.grad_v_token_query,
                ),
                (
                    geometry.query,
                    one.grad_gate.as_slice(),
                    &mut output.grad_gate_token_query,
                ),
            ] {
                let source = token * width;
                let destination = (token * query_count + query_slot) * width;
                destination_values[destination..destination + width]
                    .copy_from_slice(&source_values[source..source + width]);
            }
        }
    }
    Ok(output)
}

fn query_major_to_token_query(
    values: &[f32],
    query_count: usize,
    n_tokens: usize,
    width: usize,
) -> Result<Vec<f32>, MuseGlimmerLensError> {
    validate_len(
        "query-major row bank",
        values,
        checked_mul(
            checked_mul(query_count, n_tokens, "query-major row count")?,
            width,
            "query-major row elements",
        )?,
    )?;
    let mut output = vec![0.0_f32; values.len()];
    for query in 0..query_count {
        for token in 0..n_tokens {
            let source = (query * n_tokens + token) * width;
            let destination = (token * query_count + query) * width;
            output[destination..destination + width]
                .copy_from_slice(&values[source..source + width]);
        }
    }
    Ok(output)
}

fn token_query_to_query_major(
    values: &[f32],
    query_count: usize,
    n_tokens: usize,
    width: usize,
) -> Result<Vec<f32>, MuseGlimmerLensError> {
    validate_len(
        "token-query row bank",
        values,
        checked_mul(
            checked_mul(query_count, n_tokens, "token-query row count")?,
            width,
            "token-query row elements",
        )?,
    )?;
    let mut output = vec![0.0_f32; values.len()];
    for token in 0..n_tokens {
        for query in 0..query_count {
            let source = (token * query_count + query) * width;
            let destination = (query * n_tokens + token) * width;
            output[destination..destination + width]
                .copy_from_slice(&values[source..source + width]);
        }
    }
    Ok(output)
}

fn head_query_to_token_query(
    values: &[f32],
    query_count: usize,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, MuseGlimmerLensError> {
    let width = checked_mul(n_heads, head_dim, "head-query width")?;
    validate_len(
        "head-query row bank",
        values,
        checked_mul(
            checked_mul(query_count, n_tokens, "head-query token rows")?,
            width,
            "head-query elements",
        )?,
    )?;
    let mut output = vec![0.0_f32; values.len()];
    for token in 0..n_tokens {
        for head in 0..n_heads {
            for query in 0..query_count {
                let source = ((token * n_heads + head) * query_count + query) * head_dim;
                let destination = (token * query_count + query) * width + head * head_dim;
                output[destination..destination + head_dim]
                    .copy_from_slice(&values[source..source + head_dim]);
            }
        }
    }
    Ok(output)
}

struct ReplayTensors {
    input: MetalTensor,
    attention_normed: MetalTensor,
    q_raw: MetalTensor,
    q_raw_heads: MetalTensor,
    q: MetalTensor,
    k_raw: MetalTensor,
    k_raw_heads: MetalTensor,
    k: MetalTensor,
    v: MetalTensor,
    attention_gate: MetalTensor,
    attention_output: MetalTensor,
    attention_probabilities: MetalTensor,
    gated_attention: MetalTensor,
    attention_branch_raw: MetalTensor,
    attention_branch: MetalTensor,
    post_attention: MetalTensor,
    ffn_normed: MetalTensor,
    ffn_gate_raw: MetalTensor,
    ffn_up: MetalTensor,
    ffn_inner: MetalTensor,
    ffn_branch_raw: MetalTensor,
    ffn_branch: MetalTensor,
    post_block: MetalTensor,
}

impl ReplayTensors {
    fn new(
        ctx: &MetalContext,
        input: &[f32],
        n_tokens: usize,
        geometry: MuseAttentionGeometry,
    ) -> Result<Self, MuseGlimmerLensError> {
        let hidden_shape = row_shape(geometry.hidden, n_tokens)?;
        let query_shape = row_shape(geometry.query, n_tokens)?;
        let kv_shape = row_shape(geometry.kv, n_tokens)?;
        let ffn_shape = row_shape(geometry.feed_forward, n_tokens)?;
        let probability_shape = vec![n_tokens as u64, geometry.q_heads as u64, n_tokens as u64];
        let q_raw = MetalTensor::zeros_f32(ctx, query_shape.clone())?;
        let q_raw_heads = q_raw.view_subrange(
            0,
            row_shape(geometry.head_dim, n_tokens * geometry.q_heads)?,
        );
        let k_raw = MetalTensor::zeros_f32(ctx, kv_shape.clone())?;
        let k_raw_heads = k_raw.view_subrange(
            0,
            row_shape(geometry.head_dim, n_tokens * geometry.kv_heads)?,
        );
        Ok(Self {
            input: from_f32(ctx, input, hidden_shape.clone())?,
            attention_normed: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            q_raw,
            q_raw_heads,
            q: MetalTensor::zeros_f32(ctx, query_shape.clone())?,
            k_raw,
            k_raw_heads,
            k: MetalTensor::zeros_f32(ctx, kv_shape.clone())?,
            v: MetalTensor::zeros_f32(ctx, kv_shape)?,
            attention_gate: MetalTensor::zeros_f32(ctx, query_shape.clone())?,
            attention_output: MetalTensor::zeros_f32(ctx, query_shape.clone())?,
            attention_probabilities: MetalTensor::zeros_f32(ctx, probability_shape)?,
            gated_attention: MetalTensor::zeros_f32(ctx, query_shape)?,
            attention_branch_raw: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            attention_branch: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            post_attention: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            ffn_normed: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            ffn_gate_raw: MetalTensor::zeros_f32(ctx, ffn_shape.clone())?,
            ffn_up: MetalTensor::zeros_f32(ctx, ffn_shape.clone())?,
            ffn_inner: MetalTensor::zeros_f32(ctx, ffn_shape)?,
            ffn_branch_raw: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            ffn_branch: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            post_block: MetalTensor::zeros_f32(ctx, hidden_shape)?,
        })
    }
}

struct ReplayState {
    tensors: ReplayTensors,
    n_tokens: usize,
    geometry: MuseAttentionGeometry,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention_gate: Vec<f32>,
    attention: MuseAttentionForward,
}

struct MuseGlimmerPreparedAttentionBlock<'weights> {
    config: &'weights MuseGlimmerConfig,
    target_block: u32,
    layer: MuseGlimmerMetalLayerWeights<'weights>,
    replay: ReplayState,
    post_attention_replay_max_abs_error: f32,
    post_block_replay_max_abs_error: f32,
}

impl<'weights> MuseGlimmerPreparedAttentionBlock<'weights> {
    fn new(
        ctx: &MetalContext,
        weights: &MuseGlimmerMetalModelWeights<'weights>,
        capture: &MuseGlimmerLensCapture,
    ) -> Result<Self, MuseGlimmerLensError> {
        let geometry = MuseAttentionGeometry::from_weights(weights)?;
        let layer = *validate_replay_request(
            weights,
            capture.target_block(),
            capture.input_residuals(),
            capture.n_tokens(),
            geometry,
        )?;
        if capture.hidden_size() != geometry.hidden {
            return invalid(format!(
                "capture hidden size {} differs from model hidden size {}",
                capture.hidden_size(),
                geometry.hidden
            ));
        }
        let replay = replay_state(
            ctx,
            weights,
            &layer,
            capture.input_residuals(),
            capture.n_tokens(),
            geometry,
        )?;
        let replay_readback = replay_readback(&replay)?;
        let post_attention_replay_max_abs_error = max_abs_difference(
            &replay_readback.post_attention_residuals,
            capture.post_attention_residuals(),
        )?;
        let post_block_replay_max_abs_error = max_abs_difference(
            &replay_readback.post_block_residuals,
            capture.post_block_residuals(),
        )?;
        Ok(Self {
            config: weights.config,
            target_block: capture.target_block(),
            layer,
            replay,
            post_attention_replay_max_abs_error,
            post_block_replay_max_abs_error,
        })
    }
}

struct MuseGlimmerOneBlockVjpBankScratch {
    basis_count: usize,
    n_tokens: usize,
    geometry: MuseAttentionGeometry,
    hidden: [MetalTensor; 4],
    feed_forward: [MetalTensor; 3],
    query: [MetalTensor; 3],
    query_heads: [MetalTensor; 3],
    kv: [MetalTensor; 2],
    kv_heads: [MetalTensor; 2],
    partial_grad_key: MetalTensor,
    partial_grad_value: MetalTensor,
}

impl MuseGlimmerOneBlockVjpBankScratch {
    fn new(
        ctx: &MetalContext,
        basis_count: usize,
        n_tokens: usize,
        geometry: MuseAttentionGeometry,
    ) -> Result<Self, MuseGlimmerLensError> {
        if basis_count == 0 || n_tokens == 0 || n_tokens > MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS {
            return invalid(format!(
                "one-block VJP bank requires B>0 and T in 1..={MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS}, got B={basis_count} T={n_tokens}"
            ));
        }
        let bank_rows = checked_mul(basis_count, n_tokens, "VJP bank rows")?;
        let hidden_shape = row_shape(geometry.hidden, bank_rows)?;
        let ffn_shape = row_shape(geometry.feed_forward, bank_rows)?;
        let query_shape = row_shape(geometry.query, bank_rows)?;
        let kv_shape = row_shape(geometry.kv, bank_rows)?;
        let query_head_rows = checked_mul(bank_rows, geometry.q_heads, "query head rows")?;
        let kv_head_rows = checked_mul(bank_rows, geometry.kv_heads, "KV head rows")?;

        let query_0 = MetalTensor::zeros_f32(ctx, query_shape.clone())?;
        let query_1 = MetalTensor::zeros_f32(ctx, query_shape.clone())?;
        let query_2 = MetalTensor::zeros_f32(ctx, query_shape)?;
        let query_heads = [
            query_0.view_subrange(0, row_shape(geometry.head_dim, query_head_rows)?),
            query_1.view_subrange(0, row_shape(geometry.head_dim, query_head_rows)?),
            query_2.view_subrange(0, row_shape(geometry.head_dim, query_head_rows)?),
        ];
        let kv_0 = MetalTensor::zeros_f32(ctx, kv_shape.clone())?;
        let kv_1 = MetalTensor::zeros_f32(ctx, kv_shape)?;
        let kv_heads = [
            kv_0.view_subrange(0, row_shape(geometry.head_dim, kv_head_rows)?),
            kv_1.view_subrange(0, row_shape(geometry.head_dim, kv_head_rows)?),
        ];
        let partial_shape = vec![
            geometry.head_dim as u64,
            n_tokens as u64,
            geometry.q_heads as u64,
            basis_count as u64,
        ];

        Ok(Self {
            basis_count,
            n_tokens,
            geometry,
            hidden: [
                MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
                MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
                MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
                MetalTensor::zeros_f32(ctx, hidden_shape)?,
            ],
            feed_forward: [
                MetalTensor::zeros_f32(ctx, ffn_shape.clone())?,
                MetalTensor::zeros_f32(ctx, ffn_shape.clone())?,
                MetalTensor::zeros_f32(ctx, ffn_shape)?,
            ],
            query: [query_0, query_1, query_2],
            query_heads,
            kv: [kv_0, kv_1],
            kv_heads,
            partial_grad_key: MetalTensor::zeros_f32(ctx, partial_shape.clone())?,
            partial_grad_value: MetalTensor::zeros_f32(ctx, partial_shape)?,
        })
    }
}

pub(crate) struct MuseGlimmerPreparedMetalAttentionBlock<'weights> {
    config: &'weights MuseGlimmerConfig,
    device_registry_id: u64,
    target_block: u32,
    layer: MuseGlimmerMetalLayerWeights<'weights>,
    replay: ReplayState,
    post_attention_replay_max_abs_error: f32,
    post_block_replay_max_abs_error: f32,
}

impl<'weights> MuseGlimmerPreparedMetalAttentionBlock<'weights> {
    pub(crate) fn new(
        ctx: &MetalContext,
        weights: &MuseGlimmerMetalModelWeights<'weights>,
        capture: &MuseGlimmerLensCapture,
    ) -> Result<Self, MuseGlimmerLensError> {
        require_full_attention_block(weights, capture.target_block())?;
        Self::new_any(ctx, weights, capture)
    }

    fn new_sliding(
        ctx: &MetalContext,
        weights: &MuseGlimmerMetalModelWeights<'weights>,
        capture: &MuseGlimmerLensCapture,
    ) -> Result<Self, MuseGlimmerLensError> {
        require_sliding_attention_block(weights, capture.target_block())?;
        Self::new_any(ctx, weights, capture)
    }

    fn new_any(
        ctx: &MetalContext,
        weights: &MuseGlimmerMetalModelWeights<'weights>,
        capture: &MuseGlimmerLensCapture,
    ) -> Result<Self, MuseGlimmerLensError> {
        let geometry = MuseAttentionGeometry::from_weights(weights)?;
        let layer = *validate_replay_request(
            weights,
            capture.target_block(),
            capture.input_residuals(),
            capture.n_tokens(),
            geometry,
        )?;
        if capture.hidden_size() != geometry.hidden {
            return invalid(format!(
                "capture hidden size {} differs from model hidden size {}",
                capture.hidden_size(),
                geometry.hidden
            ));
        }
        let replay = replay_state(
            ctx,
            weights,
            &layer,
            capture.input_residuals(),
            capture.n_tokens(),
            geometry,
        )?;
        let replay_readback = replay_readback(&replay)?;
        let post_attention_replay_max_abs_error = max_abs_difference(
            &replay_readback.post_attention_residuals,
            capture.post_attention_residuals(),
        )?;
        let post_block_replay_max_abs_error = max_abs_difference(
            &replay_readback.post_block_residuals,
            capture.post_block_residuals(),
        )?;
        Ok(Self {
            config: weights.config,
            device_registry_id: ctx.device.registryID(),
            target_block: capture.target_block(),
            layer,
            replay,
            post_attention_replay_max_abs_error,
            post_block_replay_max_abs_error,
        })
    }

    pub(crate) fn target_block(&self) -> u32 {
        self.target_block
    }

    pub(crate) fn n_tokens(&self) -> usize {
        self.replay.n_tokens
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.replay.geometry.hidden
    }

    pub(crate) fn replay_max_abs_errors(&self) -> (f32, f32) {
        (
            self.post_attention_replay_max_abs_error,
            self.post_block_replay_max_abs_error,
        )
    }

    pub(crate) fn encode_bank(
        &self,
        ctx: &MetalContext,
        encoder: &KernelEncoder,
        current: &MetalTensor,
        next: &MetalTensor,
        workspace: &mut MuseGlimmerMetalAttentionBankWorkspace,
        rule: MuseGlimmerLensRule,
    ) -> Result<(), MuseGlimmerLensError> {
        let expected_device = self.device_registry_id;
        let context_device = ctx.device.registryID();
        let encoder_device = encoder.parent_command_buffer().device().registryID();
        let workspace_device = workspace.device_registry_id;
        let current_device = current.buffer.device().registryID();
        let next_device = next.buffer.device().registryID();
        if context_device != expected_device
            || encoder_device != expected_device
            || workspace_device != expected_device
            || current_device != expected_device
            || next_device != expected_device
        {
            return invalid(format!(
                "one-block VJP bank device mismatch prepared={expected_device} context={context_device} encoder={encoder_device} workspace={workspace_device} current={current_device} next={next_device}"
            ));
        }
        encode_muse_glimmer_one_attention_block_vjp_bank(
            ctx,
            encoder,
            self.config,
            &self.layer,
            &self.replay,
            current,
            next,
            &workspace.scratch,
            rule,
        )
    }
}

/// Scratch may not be reused by another command until its owning command has
/// completed.
pub(crate) struct MuseGlimmerMetalAttentionBankWorkspace {
    device_registry_id: u64,
    scratch: MuseGlimmerOneBlockVjpBankScratch,
}

impl MuseGlimmerMetalAttentionBankWorkspace {
    pub(crate) fn new(
        ctx: &MetalContext,
        prepared: &MuseGlimmerPreparedMetalAttentionBlock<'_>,
        basis_count: usize,
    ) -> Result<Self, MuseGlimmerLensError> {
        let device_registry_id = ctx.device.registryID();
        if device_registry_id != prepared.device_registry_id {
            return invalid(format!(
                "one-block VJP workspace device {device_registry_id} differs from prepared device {}",
                prepared.device_registry_id
            ));
        }
        Ok(Self {
            device_registry_id,
            scratch: MuseGlimmerOneBlockVjpBankScratch::new(
                ctx,
                basis_count,
                prepared.replay.n_tokens,
                prepared.replay.geometry,
            )?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_muse_glimmer_one_attention_block_vjp_bank(
    ctx: &MetalContext,
    encoder: &KernelEncoder,
    config: &MuseGlimmerConfig,
    layer: &MuseGlimmerMetalLayerWeights<'_>,
    prepared: &ReplayState,
    grad_post_block: &MetalTensor,
    grad_input: &MetalTensor,
    scratch: &MuseGlimmerOneBlockVjpBankScratch,
    rule: MuseGlimmerLensRule,
) -> Result<(), MuseGlimmerLensError> {
    if encoder.is_concurrent() {
        return invalid("one-block VJP bank requires a serial encoder");
    }
    let geometry = prepared.geometry;
    let n_tokens = prepared.n_tokens;
    let basis_count = scratch.basis_count;
    let bank_rows = checked_mul(basis_count, n_tokens, "one-block VJP bank rows")?;
    if scratch.n_tokens != n_tokens
        || scratch.geometry.hidden != geometry.hidden
        || scratch.geometry.feed_forward != geometry.feed_forward
        || scratch.geometry.q_heads != geometry.q_heads
        || scratch.geometry.kv_heads != geometry.kv_heads
        || scratch.geometry.head_dim != geometry.head_dim
        || grad_post_block.dtype != GgmlType::F32
        || grad_input.dtype != GgmlType::F32
        || !grad_input.is_writable()
        || grad_post_block.shape != scratch.hidden[0].shape
        || grad_input.shape != scratch.hidden[0].shape
        || tensor_ranges_overlap(grad_post_block, grad_input)
    {
        return invalid("one-block VJP bank scratch or cotangent shape mismatch");
    }
    let tensors = &prepared.tensors;

    encode_muse_glimmer_one_attention_block_vjp_bank_feed_forward(
        ctx,
        encoder,
        config,
        layer,
        tensors,
        geometry,
        n_tokens,
        bank_rows,
        grad_post_block,
        scratch,
        rule,
    )?;
    encode_muse_glimmer_one_attention_block_vjp_bank_attention_output(
        ctx, encoder, config, layer, tensors, geometry, n_tokens, bank_rows, scratch, rule,
    )?;
    encode_muse_glimmer_one_attention_block_vjp_bank_causal_gqa(
        ctx,
        encoder,
        layer,
        tensors,
        geometry,
        basis_count,
        n_tokens,
        bank_rows,
        scratch,
    )?;
    encode_muse_glimmer_one_attention_block_vjp_bank_attention_input(
        ctx, encoder, config, layer, tensors, geometry, n_tokens, bank_rows, grad_input, scratch,
        rule,
    )?;

    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MuseGlimmerOneBlockReplay {
    pub post_attention_residuals: Vec<f32>,
    pub post_block_residuals: Vec<f32>,
}

/// Test seam for evaluating the smooth F32 model-level block graph on an
/// arbitrary `[T,H]` input. It deliberately does not round K/V through the
/// production F16 cache.
#[cfg(test)]
pub(crate) fn muse_glimmer_replay_one_full_attention_block(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    target_block: u32,
    input: &[f32],
    n_tokens: usize,
) -> Result<MuseGlimmerOneBlockReplay, MuseGlimmerLensError> {
    require_full_attention_block(weights, target_block)?;
    muse_glimmer_replay_one_attention_block(ctx, weights, target_block, input, n_tokens)
}

#[cfg(test)]
pub(crate) fn muse_glimmer_replay_one_attention_block(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    target_block: u32,
    input: &[f32],
    n_tokens: usize,
) -> Result<MuseGlimmerOneBlockReplay, MuseGlimmerLensError> {
    let geometry = MuseAttentionGeometry::from_weights(weights)?;
    let layer = validate_replay_request(weights, target_block, input, n_tokens, geometry)?;
    let state = replay_state(ctx, weights, layer, input, n_tokens, geometry)?;
    replay_readback(&state)
}

fn replay_readback(state: &ReplayState) -> Result<MuseGlimmerOneBlockReplay, MuseGlimmerLensError> {
    let post_attention_residuals = read_f32(&state.tensors.post_attention);
    let post_block_residuals = read_f32(&state.tensors.post_block);
    require_finite(
        "replayed post-attention residual",
        &post_attention_residuals,
    )?;
    require_finite("replayed post-block residual", &post_block_residuals)?;
    Ok(MuseGlimmerOneBlockReplay {
        post_attention_residuals,
        post_block_residuals,
    })
}

fn replay_state(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    layer: &MuseGlimmerMetalLayerWeights<'_>,
    input: &[f32],
    n_tokens: usize,
    geometry: MuseAttentionGeometry,
) -> Result<ReplayState, MuseGlimmerLensError> {
    let tensors = ReplayTensors::new(ctx, input, n_tokens, geometry)?;
    run_command(ctx, |encoder| {
        encode_rms_norm_mul_rows_f32(
            ctx,
            encoder,
            &tensors.input,
            layer.attention_norm,
            &tensors.attention_normed,
            n_tokens,
            geometry.hidden,
            weights.config.rms_epsilon,
        )?;
        for token in 0..n_tokens {
            let input = row_view(&tensors.attention_normed, token, geometry.hidden);
            for (weight, output, width) in [
                (layer.attention_query, &tensors.q_raw, geometry.query),
                (layer.attention_key, &tensors.k_raw, geometry.kv),
                (layer.attention_value, &tensors.v, geometry.kv),
                (
                    layer.attention_gate,
                    &tensors.attention_gate,
                    geometry.query,
                ),
            ] {
                encode_mat_vec_dispatch(
                    ctx,
                    encoder,
                    weight,
                    &input,
                    &row_view(output, token, width),
                    geometry.hidden,
                    width,
                )?;
            }
        }
        encode_rms_norm_batched_f32(
            ctx,
            encoder,
            &tensors.q_raw,
            layer.query_norm,
            &tensors.q,
            n_tokens * geometry.q_heads,
            geometry.head_dim,
            weights.config.rms_epsilon,
        )?;
        encode_rms_norm_batched_f32(
            ctx,
            encoder,
            &tensors.k_raw,
            layer.key_norm,
            &tensors.k,
            n_tokens * geometry.kv_heads,
            geometry.head_dim,
            weights.config.rms_epsilon,
        )?;
        Ok(())
    })?;

    let mut q = read_f32(&tensors.q);
    let mut k = read_f32(&tensors.k);
    let v = read_f32(&tensors.v);
    let attention_gate = read_f32(&tensors.attention_gate);
    if layer.sliding_attention {
        adjacent_pair_rope_rows_in_place(
            &mut q,
            n_tokens,
            geometry.q_heads,
            geometry.head_dim,
            geometry.rope_theta,
            false,
        )?;
        adjacent_pair_rope_rows_in_place(
            &mut k,
            n_tokens,
            geometry.kv_heads,
            geometry.head_dim,
            geometry.rope_theta,
            false,
        )?;
        write_f32(&tensors.q, &q)?;
        write_f32(&tensors.k, &k)?;
    }
    let attention = cpu_causal_gqa_forward(&q, &k, &v, &attention_gate, n_tokens, geometry)?;
    write_f32(&tensors.attention_output, &attention.attention_output)?;
    write_f32(&tensors.attention_probabilities, &attention.probabilities)?;
    write_f32(&tensors.gated_attention, &attention.gated_output)?;

    run_command(ctx, |encoder| {
        for token in 0..n_tokens {
            encode_mat_vec_dispatch(
                ctx,
                encoder,
                layer.attention_output,
                &row_view(&tensors.gated_attention, token, geometry.query),
                &row_view(&tensors.attention_branch_raw, token, geometry.hidden),
                geometry.query,
                geometry.hidden,
            )?;
        }
        encode_rms_norm_mul_rows_f32(
            ctx,
            encoder,
            &tensors.attention_branch_raw,
            layer.post_attention_norm,
            &tensors.attention_branch,
            n_tokens,
            geometry.hidden,
            weights.config.post_norm_epsilon,
        )?;
        encode_add_f32(
            ctx,
            encoder,
            &tensors.input,
            &tensors.attention_branch,
            &tensors.post_attention,
        )?;
        encode_rms_norm_mul_rows_f32(
            ctx,
            encoder,
            &tensors.post_attention,
            layer.feed_forward_norm,
            &tensors.ffn_normed,
            n_tokens,
            geometry.hidden,
            weights.config.rms_epsilon,
        )?;
        for token in 0..n_tokens {
            let input = row_view(&tensors.ffn_normed, token, geometry.hidden);
            for (weight, output) in [
                (layer.feed_forward_gate, &tensors.ffn_gate_raw),
                (layer.feed_forward_up, &tensors.ffn_up),
            ] {
                encode_mat_vec_dispatch(
                    ctx,
                    encoder,
                    weight,
                    &input,
                    &row_view(output, token, geometry.feed_forward),
                    geometry.hidden,
                    geometry.feed_forward,
                )?;
            }
        }
        encode_silu_mul_f32(
            ctx,
            encoder,
            &tensors.ffn_gate_raw,
            &tensors.ffn_up,
            &tensors.ffn_inner,
        )?;
        for token in 0..n_tokens {
            encode_mat_vec_dispatch(
                ctx,
                encoder,
                layer.feed_forward_down,
                &row_view(&tensors.ffn_inner, token, geometry.feed_forward),
                &row_view(&tensors.ffn_branch_raw, token, geometry.hidden),
                geometry.feed_forward,
                geometry.hidden,
            )?;
        }
        encode_rms_norm_mul_rows_f32(
            ctx,
            encoder,
            &tensors.ffn_branch_raw,
            layer.post_feed_forward_norm,
            &tensors.ffn_branch,
            n_tokens,
            geometry.hidden,
            weights.config.post_norm_epsilon,
        )?;
        encode_add_f32(
            ctx,
            encoder,
            &tensors.post_attention,
            &tensors.ffn_branch,
            &tensors.post_block,
        )?;
        Ok(())
    })?;
    Ok(ReplayState {
        tensors,
        n_tokens,
        geometry,
        q,
        k,
        v,
        attention_gate,
        attention,
    })
}

pub(crate) fn muse_glimmer_one_full_attention_block_vjp(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    capture: &MuseGlimmerLensCapture,
    target_cotangent: &[f32],
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerOneBlockVjp, MuseGlimmerLensError> {
    require_full_attention_block(weights, capture.target_block())?;
    muse_glimmer_one_attention_block_vjp(ctx, weights, capture, target_cotangent, rule)
}

pub(crate) fn muse_glimmer_one_attention_block_vjp(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    capture: &MuseGlimmerLensCapture,
    target_cotangent: &[f32],
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerOneBlockVjp, MuseGlimmerLensError> {
    let geometry = MuseAttentionGeometry::from_weights(weights)?;
    let target_block = capture.target_block();
    let layer = validate_replay_request(
        weights,
        target_block,
        capture.input_residuals(),
        capture.n_tokens(),
        geometry,
    )?;
    validate_request(capture, target_cotangent, geometry.hidden)?;
    let n_tokens = capture.n_tokens();
    let state = replay_state(
        ctx,
        weights,
        layer,
        capture.input_residuals(),
        n_tokens,
        geometry,
    )?;
    let replay = replay_readback(&state)?;
    let ReplayState {
        tensors,
        q,
        k,
        v,
        attention_gate,
        attention,
        ..
    } = state;
    let post_attention_replay_max_abs_error = max_abs_difference(
        &replay.post_attention_residuals,
        capture.post_attention_residuals(),
    )?;
    let post_block_replay_max_abs_error =
        max_abs_difference(&replay.post_block_residuals, capture.post_block_residuals())?;

    let hidden_shape = row_shape(geometry.hidden, n_tokens)?;
    let ffn_shape = row_shape(geometry.feed_forward, n_tokens)?;
    let query_shape = row_shape(geometry.query, n_tokens)?;
    let kv_shape = row_shape(geometry.kv, n_tokens)?;
    let grad_post_block = from_f32(ctx, target_cotangent, hidden_shape.clone())?;
    let grad_ffn_raw = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_ffn_inner = MetalTensor::zeros_f32(ctx, ffn_shape.clone())?;
    let grad_ffn_gate = MetalTensor::zeros_f32(ctx, ffn_shape.clone())?;
    let grad_ffn_up = MetalTensor::zeros_f32(ctx, ffn_shape)?;
    let grad_ffn_norm_gate = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_ffn_norm_up = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_ffn_norm = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_post_attention_branch = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_post_attention = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    run_command(ctx, |encoder| {
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &tensors.ffn_branch_raw,
            layer.post_feed_forward_norm,
            &grad_post_block,
            &grad_ffn_raw,
            n_tokens,
            geometry.hidden,
            weights.config.post_norm_epsilon,
            rule.rms_norm_rule(MuseGlimmerRmsNormSite::FeedForwardBranchPostNorm),
        )?;
        encode_frozen_linear_vjp_f32(
            ctx,
            encoder,
            layer.feed_forward_down,
            &grad_ffn_raw,
            &grad_ffn_inner,
            geometry.feed_forward,
            geometry.hidden,
            n_tokens,
        )?;
        encode_silu_mul_vjp_f32(
            ctx,
            encoder,
            &tensors.ffn_gate_raw,
            &tensors.ffn_up,
            &grad_ffn_inner,
            &grad_ffn_gate,
            &grad_ffn_up,
            n_tokens,
            geometry.feed_forward,
            rule.swiglu_rule(),
        )?;
        encode_frozen_linear_vjp_f32(
            ctx,
            encoder,
            layer.feed_forward_gate,
            &grad_ffn_gate,
            &grad_ffn_norm_gate,
            geometry.hidden,
            geometry.feed_forward,
            n_tokens,
        )?;
        encode_frozen_linear_vjp_f32(
            ctx,
            encoder,
            layer.feed_forward_up,
            &grad_ffn_up,
            &grad_ffn_norm_up,
            geometry.hidden,
            geometry.feed_forward,
            n_tokens,
        )?;
        encode_add_f32(
            ctx,
            encoder,
            &grad_ffn_norm_gate,
            &grad_ffn_norm_up,
            &grad_ffn_norm,
        )?;
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &tensors.post_attention,
            layer.feed_forward_norm,
            &grad_ffn_norm,
            &grad_post_attention_branch,
            n_tokens,
            geometry.hidden,
            weights.config.rms_epsilon,
            rule.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreFeedForward),
        )?;
        encode_add_f32(
            ctx,
            encoder,
            &grad_post_block,
            &grad_post_attention_branch,
            &grad_post_attention,
        )?;
        Ok(())
    })?;

    let grad_attention_raw = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_gated = MetalTensor::zeros_f32(ctx, query_shape.clone())?;
    run_command(ctx, |encoder| {
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &tensors.attention_branch_raw,
            layer.post_attention_norm,
            &grad_post_attention,
            &grad_attention_raw,
            n_tokens,
            geometry.hidden,
            weights.config.post_norm_epsilon,
            rule.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionBranchPostNorm),
        )?;
        encode_frozen_linear_vjp_f32(
            ctx,
            encoder,
            layer.attention_output,
            &grad_attention_raw,
            &grad_gated,
            geometry.query,
            geometry.hidden,
            n_tokens,
        )?;
        Ok(())
    })?;
    let mut attention_vjp = cpu_causal_gqa_vjp(
        &q,
        &k,
        &v,
        &attention_gate,
        &read_f32(&grad_gated),
        n_tokens,
        geometry,
        &attention,
    )?;
    if layer.sliding_attention {
        adjacent_pair_rope_rows_in_place(
            &mut attention_vjp.grad_q,
            n_tokens,
            geometry.q_heads,
            geometry.head_dim,
            geometry.rope_theta,
            true,
        )?;
        adjacent_pair_rope_rows_in_place(
            &mut attention_vjp.grad_k,
            n_tokens,
            geometry.kv_heads,
            geometry.head_dim,
            geometry.rope_theta,
            true,
        )?;
    }

    let grad_q = from_f32(ctx, &attention_vjp.grad_q, query_shape.clone())?;
    let grad_k = from_f32(ctx, &attention_vjp.grad_k, kv_shape.clone())?;
    let grad_v = from_f32(ctx, &attention_vjp.grad_v, kv_shape)?;
    let grad_gate = from_f32(ctx, &attention_vjp.grad_gate, query_shape)?;
    let grad_q_raw = MetalTensor::zeros_f32(ctx, row_shape(geometry.query, n_tokens)?)?;
    let grad_k_raw = MetalTensor::zeros_f32(ctx, row_shape(geometry.kv, n_tokens)?)?;
    let grad_norm_q = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_norm_k = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_norm_v = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_norm_gate = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_norm_qk = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_norm_qkv = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_norm = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_input_branch = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(ctx, hidden_shape)?;
    let grad_q_heads = grad_q.view_subrange(
        0,
        row_shape(geometry.head_dim, n_tokens * geometry.q_heads)?,
    );
    let grad_q_raw_heads = grad_q_raw.view_subrange(
        0,
        row_shape(geometry.head_dim, n_tokens * geometry.q_heads)?,
    );
    let grad_k_heads = grad_k.view_subrange(
        0,
        row_shape(geometry.head_dim, n_tokens * geometry.kv_heads)?,
    );
    let grad_k_raw_heads = grad_k_raw.view_subrange(
        0,
        row_shape(geometry.head_dim, n_tokens * geometry.kv_heads)?,
    );
    run_command(ctx, |encoder| {
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &tensors.q_raw_heads,
            layer.query_norm,
            &grad_q_heads,
            &grad_q_raw_heads,
            n_tokens * geometry.q_heads,
            geometry.head_dim,
            weights.config.rms_epsilon,
            rule.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionQuery),
        )?;
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &tensors.k_raw_heads,
            layer.key_norm,
            &grad_k_heads,
            &grad_k_raw_heads,
            n_tokens * geometry.kv_heads,
            geometry.head_dim,
            weights.config.rms_epsilon,
            rule.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionKey),
        )?;
        for (weight, grad_output, grad_hidden, n_out) in [
            (
                layer.attention_query,
                &grad_q_raw,
                &grad_norm_q,
                geometry.query,
            ),
            (layer.attention_key, &grad_k_raw, &grad_norm_k, geometry.kv),
            (layer.attention_value, &grad_v, &grad_norm_v, geometry.kv),
            (
                layer.attention_gate,
                &grad_gate,
                &grad_norm_gate,
                geometry.query,
            ),
        ] {
            encode_frozen_linear_vjp_f32(
                ctx,
                encoder,
                weight,
                grad_output,
                grad_hidden,
                geometry.hidden,
                n_out,
                n_tokens,
            )?;
        }
        encode_add_f32(ctx, encoder, &grad_norm_q, &grad_norm_k, &grad_norm_qk)?;
        encode_add_f32(ctx, encoder, &grad_norm_qk, &grad_norm_v, &grad_norm_qkv)?;
        encode_add_f32(ctx, encoder, &grad_norm_qkv, &grad_norm_gate, &grad_norm)?;
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &tensors.input,
            layer.attention_norm,
            &grad_norm,
            &grad_input_branch,
            n_tokens,
            geometry.hidden,
            weights.config.rms_epsilon,
            rule.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention),
        )?;
        encode_add_f32(
            ctx,
            encoder,
            &grad_post_attention,
            &grad_input_branch,
            &grad_input,
        )?;
        Ok(())
    })?;
    let input_cotangent = read_f32(&grad_input);
    require_finite("input cotangent", &input_cotangent)?;
    Ok(MuseGlimmerOneBlockVjp {
        target_block,
        rule,
        n_tokens,
        hidden_size: geometry.hidden,
        input_cotangent,
        post_attention_replay_max_abs_error,
        post_block_replay_max_abs_error,
    })
}

fn reverse_prepared_metal_attention_block_vjp_query_batch(
    ctx: &MetalContext,
    prepared: &MuseGlimmerPreparedMetalAttentionBlock<'_>,
    target_cotangents: &[f32],
    query_count: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerQueryBatchOneBlockVjp, MuseGlimmerLensError> {
    let total_started = Instant::now();
    let geometry = prepared.replay.geometry;
    validate_query_bank_values(
        target_cotangents,
        query_count,
        prepared.n_tokens(),
        geometry.hidden,
    )?;
    let shape = row_shape(
        geometry.hidden,
        checked_mul(query_count, prepared.n_tokens(), "full bank rows")?,
    )?;
    let current = from_f32(ctx, target_cotangents, shape.clone())?;
    let next = MetalTensor::zeros_f32(ctx, shape)?;
    let mut workspace = MuseGlimmerMetalAttentionBankWorkspace::new(ctx, prepared, query_count)?;

    let reverse_started = Instant::now();
    run_command(ctx, |encoder| {
        prepared.encode_bank(ctx, encoder, &current, &next, &mut workspace, rule)
    })?;
    let reverse_elapsed = reverse_started.elapsed();
    let input_cotangents = read_f32(&next);
    require_finite("Metal attention bank input cotangents", &input_cotangents)?;
    let (post_attention_replay_max_abs_error, post_block_replay_max_abs_error) =
        prepared.replay_max_abs_errors();
    let (full_attention_bank_command, sliding_attention_bank_command) =
        if prepared.layer.sliding_attention {
            (Duration::ZERO, reverse_elapsed)
        } else {
            (reverse_elapsed, Duration::ZERO)
        };
    Ok(MuseGlimmerQueryBatchOneBlockVjp {
        target_block: prepared.target_block(),
        rule,
        query_count,
        n_tokens: prepared.n_tokens(),
        hidden_size: prepared.hidden_size(),
        input_cotangents,
        post_attention_replay_max_abs_error,
        post_block_replay_max_abs_error,
        timings: MuseGlimmerQueryBatchVjpTimings {
            full_attention_bank_command,
            sliding_attention_bank_command,
            total: total_started.elapsed(),
            ..Default::default()
        },
    })
}

pub(crate) fn muse_glimmer_one_full_attention_block_vjp_query_batch(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    capture: &MuseGlimmerLensCapture,
    target_cotangents: &[f32],
    query_count: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerQueryBatchOneBlockVjp, MuseGlimmerLensError> {
    let total_started = Instant::now();
    require_full_attention_block(weights, capture.target_block())?;
    let geometry = MuseAttentionGeometry::from_weights(weights)?;
    validate_query_batch_request(capture, target_cotangents, query_count, geometry.hidden)?;
    let replay_started = Instant::now();
    let prepared = MuseGlimmerPreparedMetalAttentionBlock::new(ctx, weights, capture)?;
    let replay_elapsed = replay_started.elapsed();
    let mut result = reverse_prepared_metal_attention_block_vjp_query_batch(
        ctx,
        &prepared,
        target_cotangents,
        query_count,
        rule,
    )?;
    result.timings.replay = replay_elapsed;
    result.timings.total = total_started.elapsed();
    Ok(result)
}

#[cfg(test)]
fn muse_glimmer_one_sliding_attention_block_vjp_metal_query_batch(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    capture: &MuseGlimmerLensCapture,
    target_cotangents: &[f32],
    query_count: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerQueryBatchOneBlockVjp, MuseGlimmerLensError> {
    let total_started = Instant::now();
    require_sliding_attention_block(weights, capture.target_block())?;
    let geometry = MuseAttentionGeometry::from_weights(weights)?;
    validate_query_batch_request(capture, target_cotangents, query_count, geometry.hidden)?;
    let replay_started = Instant::now();
    let prepared = MuseGlimmerPreparedMetalAttentionBlock::new_sliding(ctx, weights, capture)?;
    let replay_elapsed = replay_started.elapsed();
    let mut result = reverse_prepared_metal_attention_block_vjp_query_batch(
        ctx,
        &prepared,
        target_cotangents,
        query_count,
        rule,
    )?;
    result.timings.replay = replay_elapsed;
    result.timings.total = total_started.elapsed();
    Ok(result)
}

fn reverse_prepared_attention_block_vjp_query_batch(
    ctx: &MetalContext,
    prepared: &MuseGlimmerPreparedAttentionBlock<'_>,
    target_cotangents: &[f32],
    query_count: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerQueryBatchOneBlockVjp, MuseGlimmerLensError> {
    reverse_attention_block_vjp_query_batch_from_replay(
        ctx,
        prepared.config,
        prepared.target_block,
        &prepared.layer,
        &prepared.replay,
        prepared.post_attention_replay_max_abs_error,
        prepared.post_block_replay_max_abs_error,
        target_cotangents,
        query_count,
        rule,
    )
}

#[allow(clippy::too_many_arguments)]
fn reverse_attention_block_vjp_query_batch_from_replay(
    ctx: &MetalContext,
    config: &MuseGlimmerConfig,
    target_block: u32,
    layer: &MuseGlimmerMetalLayerWeights<'_>,
    replay: &ReplayState,
    post_attention_replay_max_abs_error: f32,
    post_block_replay_max_abs_error: f32,
    target_cotangents: &[f32],
    query_count: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerQueryBatchOneBlockVjp, MuseGlimmerLensError> {
    let total_started = Instant::now();
    let geometry = replay.geometry;
    validate_query_bank_values(
        target_cotangents,
        query_count,
        replay.n_tokens,
        geometry.hidden,
    )?;
    let n_tokens = replay.n_tokens;
    let batch_rows = checked_mul(n_tokens, query_count, "query-batch reverse rows")?;
    let tensors = &replay.tensors;
    let q = replay.q.as_slice();
    let k = replay.k.as_slice();
    let v = replay.v.as_slice();
    let attention_gate = replay.attention_gate.as_slice();
    let attention = &replay.attention;

    validate_layer_shapes(layer, geometry)?;

    // Public banks are query-major. Reverse-linear kernels consume the same
    // independent rows in token-major/query-minor order so one primal token
    // row can be broadcast across its compact Q-row group.
    let target_token_query =
        query_major_to_token_query(target_cotangents, query_count, n_tokens, geometry.hidden)?;
    let hidden_batch_shape = row_shape(geometry.hidden, batch_rows)?;
    let ffn_batch_shape = row_shape(geometry.feed_forward, batch_rows)?;
    let query_batch_shape = row_shape(geometry.query, batch_rows)?;
    let kv_batch_shape = row_shape(geometry.kv, batch_rows)?;

    let feed_forward_started = Instant::now();
    let grad_post_block = from_f32(ctx, &target_token_query, hidden_batch_shape.clone())?;
    let grad_ffn_raw = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_ffn_inner = MetalTensor::zeros_f32(ctx, ffn_batch_shape.clone())?;
    let grad_ffn_gate = MetalTensor::zeros_f32(ctx, ffn_batch_shape.clone())?;
    let grad_ffn_up = MetalTensor::zeros_f32(ctx, ffn_batch_shape)?;
    let grad_ffn_norm_gate = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_ffn_norm_up = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_ffn_norm = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_post_attention_branch = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_post_attention = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    run_command(ctx, |encoder| {
        for token in 0..n_tokens {
            encode_rms_norm_mul_vjp_broadcast_f32(
                ctx,
                encoder,
                &row_view(&tensors.ffn_branch_raw, token, geometry.hidden),
                layer.post_feed_forward_norm,
                &batch_token_view(&grad_post_block, token, query_count, geometry.hidden),
                &batch_token_view(&grad_ffn_raw, token, query_count, geometry.hidden),
                query_count,
                geometry.hidden,
                config.post_norm_epsilon,
                rule.rms_norm_rule(MuseGlimmerRmsNormSite::FeedForwardBranchPostNorm),
            )?;
        }
        encode_frozen_linear_vjp_bank_f32(
            ctx,
            encoder,
            layer.feed_forward_down,
            &grad_ffn_raw,
            &grad_ffn_inner,
            geometry.feed_forward,
            geometry.hidden,
            batch_rows,
        )?;
        for token in 0..n_tokens {
            encode_silu_mul_vjp_broadcast_f32(
                ctx,
                encoder,
                &row_view(&tensors.ffn_gate_raw, token, geometry.feed_forward),
                &row_view(&tensors.ffn_up, token, geometry.feed_forward),
                &batch_token_view(&grad_ffn_inner, token, query_count, geometry.feed_forward),
                &batch_token_view(&grad_ffn_gate, token, query_count, geometry.feed_forward),
                &batch_token_view(&grad_ffn_up, token, query_count, geometry.feed_forward),
                query_count,
                geometry.feed_forward,
                rule.swiglu_rule(),
            )?;
        }
        encode_frozen_linear_vjp_bank_f32(
            ctx,
            encoder,
            layer.feed_forward_gate,
            &grad_ffn_gate,
            &grad_ffn_norm_gate,
            geometry.hidden,
            geometry.feed_forward,
            batch_rows,
        )?;
        encode_frozen_linear_vjp_bank_f32(
            ctx,
            encoder,
            layer.feed_forward_up,
            &grad_ffn_up,
            &grad_ffn_norm_up,
            geometry.hidden,
            geometry.feed_forward,
            batch_rows,
        )?;
        encode_add_f32(
            ctx,
            encoder,
            &grad_ffn_norm_gate,
            &grad_ffn_norm_up,
            &grad_ffn_norm,
        )?;
        for token in 0..n_tokens {
            encode_rms_norm_mul_vjp_broadcast_f32(
                ctx,
                encoder,
                &row_view(&tensors.post_attention, token, geometry.hidden),
                layer.feed_forward_norm,
                &batch_token_view(&grad_ffn_norm, token, query_count, geometry.hidden),
                &batch_token_view(
                    &grad_post_attention_branch,
                    token,
                    query_count,
                    geometry.hidden,
                ),
                query_count,
                geometry.hidden,
                config.rms_epsilon,
                rule.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreFeedForward),
            )?;
        }
        encode_add_f32(
            ctx,
            encoder,
            &grad_post_block,
            &grad_post_attention_branch,
            &grad_post_attention,
        )?;
        Ok(())
    })?;
    let feed_forward_elapsed = feed_forward_started.elapsed();

    let attention_output_started = Instant::now();
    let grad_attention_raw = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_gated = MetalTensor::zeros_f32(ctx, query_batch_shape.clone())?;
    run_command(ctx, |encoder| {
        for token in 0..n_tokens {
            encode_rms_norm_mul_vjp_broadcast_f32(
                ctx,
                encoder,
                &row_view(&tensors.attention_branch_raw, token, geometry.hidden),
                layer.post_attention_norm,
                &batch_token_view(&grad_post_attention, token, query_count, geometry.hidden),
                &batch_token_view(&grad_attention_raw, token, query_count, geometry.hidden),
                query_count,
                geometry.hidden,
                config.post_norm_epsilon,
                rule.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionBranchPostNorm),
            )?;
        }
        encode_frozen_linear_vjp_bank_f32(
            ctx,
            encoder,
            layer.attention_output,
            &grad_attention_raw,
            &grad_gated,
            geometry.query,
            geometry.hidden,
            batch_rows,
        )?;
        Ok(())
    })?;
    let attention_output_elapsed = attention_output_started.elapsed();

    let attention_cpu_started = Instant::now();
    let attention_vjp = cpu_causal_gqa_vjp_query_batch(
        q,
        k,
        v,
        attention_gate,
        &read_f32(&grad_gated),
        query_count,
        n_tokens,
        geometry,
        attention,
        layer.sliding_attention,
    )?;
    let attention_cpu_elapsed = attention_cpu_started.elapsed();

    let attention_input_started = Instant::now();
    let q_primal_head_rows = checked_mul(n_tokens, geometry.q_heads, "Q primal head rows")?;
    let k_primal_head_rows = checked_mul(n_tokens, geometry.kv_heads, "K primal head rows")?;
    let q_head_rows = checked_mul(q_primal_head_rows, query_count, "Q head rows")?;
    let k_head_rows = checked_mul(k_primal_head_rows, query_count, "K head rows")?;
    let grad_q_heads = from_f32(
        ctx,
        &attention_vjp.grad_q_head_query,
        row_shape(geometry.head_dim, q_head_rows)?,
    )?;
    let grad_k_heads = from_f32(
        ctx,
        &attention_vjp.grad_k_head_query,
        row_shape(geometry.head_dim, k_head_rows)?,
    )?;
    let grad_q_raw_heads = MetalTensor::zeros_f32(ctx, row_shape(geometry.head_dim, q_head_rows)?)?;
    let grad_k_raw_heads = MetalTensor::zeros_f32(ctx, row_shape(geometry.head_dim, k_head_rows)?)?;
    run_command(ctx, |encoder| {
        for primal_head in 0..q_primal_head_rows {
            encode_rms_norm_mul_vjp_broadcast_f32(
                ctx,
                encoder,
                &row_view(&tensors.q_raw, primal_head, geometry.head_dim),
                layer.query_norm,
                &batch_token_view(&grad_q_heads, primal_head, query_count, geometry.head_dim),
                &batch_token_view(
                    &grad_q_raw_heads,
                    primal_head,
                    query_count,
                    geometry.head_dim,
                ),
                query_count,
                geometry.head_dim,
                config.rms_epsilon,
                rule.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionQuery),
            )?;
        }
        for primal_head in 0..k_primal_head_rows {
            encode_rms_norm_mul_vjp_broadcast_f32(
                ctx,
                encoder,
                &row_view(&tensors.k_raw, primal_head, geometry.head_dim),
                layer.key_norm,
                &batch_token_view(&grad_k_heads, primal_head, query_count, geometry.head_dim),
                &batch_token_view(
                    &grad_k_raw_heads,
                    primal_head,
                    query_count,
                    geometry.head_dim,
                ),
                query_count,
                geometry.head_dim,
                config.rms_epsilon,
                rule.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionKey),
            )?;
        }
        Ok(())
    })?;

    // Q/K normalization needs `[primal_head,Q,D]`; projection VJPs need the
    // ordinary flattened `[T,Q,H*D]` row bank. This bounded conversion is the
    // only extra Q/K temporary required by the existing kernel contracts.
    let grad_q_raw_token_query = head_query_to_token_query(
        &read_f32(&grad_q_raw_heads),
        query_count,
        n_tokens,
        geometry.q_heads,
        geometry.head_dim,
    )?;
    let grad_k_raw_token_query = head_query_to_token_query(
        &read_f32(&grad_k_raw_heads),
        query_count,
        n_tokens,
        geometry.kv_heads,
        geometry.head_dim,
    )?;
    let grad_q_raw = from_f32(ctx, &grad_q_raw_token_query, query_batch_shape)?;
    let grad_k_raw = from_f32(ctx, &grad_k_raw_token_query, kv_batch_shape.clone())?;
    let grad_v = from_f32(ctx, &attention_vjp.grad_v_token_query, kv_batch_shape)?;
    let grad_gate = from_f32(
        ctx,
        &attention_vjp.grad_gate_token_query,
        row_shape(geometry.query, batch_rows)?,
    )?;
    let grad_norm_q = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_norm_k = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_norm_v = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_norm_gate = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_norm_qk = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_norm_qkv = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_norm = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_input_branch = MetalTensor::zeros_f32(ctx, hidden_batch_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(ctx, hidden_batch_shape)?;
    run_command(ctx, |encoder| {
        for (weight, grad_output, grad_hidden, n_out) in [
            (
                layer.attention_query,
                &grad_q_raw,
                &grad_norm_q,
                geometry.query,
            ),
            (layer.attention_key, &grad_k_raw, &grad_norm_k, geometry.kv),
            (layer.attention_value, &grad_v, &grad_norm_v, geometry.kv),
            (
                layer.attention_gate,
                &grad_gate,
                &grad_norm_gate,
                geometry.query,
            ),
        ] {
            encode_frozen_linear_vjp_bank_f32(
                ctx,
                encoder,
                weight,
                grad_output,
                grad_hidden,
                geometry.hidden,
                n_out,
                batch_rows,
            )?;
        }
        encode_add_f32(ctx, encoder, &grad_norm_q, &grad_norm_k, &grad_norm_qk)?;
        encode_add_f32(ctx, encoder, &grad_norm_qk, &grad_norm_v, &grad_norm_qkv)?;
        encode_add_f32(ctx, encoder, &grad_norm_qkv, &grad_norm_gate, &grad_norm)?;
        for token in 0..n_tokens {
            encode_rms_norm_mul_vjp_broadcast_f32(
                ctx,
                encoder,
                &row_view(&tensors.input, token, geometry.hidden),
                layer.attention_norm,
                &batch_token_view(&grad_norm, token, query_count, geometry.hidden),
                &batch_token_view(&grad_input_branch, token, query_count, geometry.hidden),
                query_count,
                geometry.hidden,
                config.rms_epsilon,
                rule.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention),
            )?;
        }
        encode_add_f32(
            ctx,
            encoder,
            &grad_post_attention,
            &grad_input_branch,
            &grad_input,
        )?;
        Ok(())
    })?;
    let input_cotangents = token_query_to_query_major(
        &read_f32(&grad_input),
        query_count,
        n_tokens,
        geometry.hidden,
    )?;
    require_finite("query-batch input cotangents", &input_cotangents)?;
    let attention_input_elapsed = attention_input_started.elapsed();
    Ok(MuseGlimmerQueryBatchOneBlockVjp {
        target_block,
        rule,
        query_count,
        n_tokens,
        hidden_size: geometry.hidden,
        input_cotangents,
        post_attention_replay_max_abs_error,
        post_block_replay_max_abs_error,
        timings: MuseGlimmerQueryBatchVjpTimings {
            replay: Duration::ZERO,
            full_attention_bank_command: Duration::ZERO,
            sliding_attention_bank_command: Duration::ZERO,
            feed_forward_reverse: feed_forward_elapsed,
            attention_output_reverse: attention_output_elapsed,
            attention_cpu_reverse: attention_cpu_elapsed,
            attention_input_reverse: attention_input_elapsed,
            total: total_started.elapsed(),
        },
    })
}

pub(crate) fn muse_glimmer_one_attention_block_vjp_query_batch(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    capture: &MuseGlimmerLensCapture,
    target_cotangents: &[f32],
    query_count: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerQueryBatchOneBlockVjp, MuseGlimmerLensError> {
    let total_started = Instant::now();
    let geometry = MuseAttentionGeometry::from_weights(weights)?;
    validate_query_batch_request(capture, target_cotangents, query_count, geometry.hidden)?;
    let replay_started = Instant::now();
    let prepared = MuseGlimmerPreparedAttentionBlock::new(ctx, weights, capture)?;
    let replay_elapsed = replay_started.elapsed();
    let mut result = reverse_prepared_attention_block_vjp_query_batch(
        ctx,
        &prepared,
        target_cotangents,
        query_count,
        rule,
    )?;
    result.timings.replay = replay_elapsed;
    result.timings.total = total_started.elapsed();
    Ok(result)
}

fn validate_replay_request<'a, 'weights>(
    weights: &'a MuseGlimmerMetalModelWeights<'weights>,
    target_block: u32,
    input: &[f32],
    n_tokens: usize,
    geometry: MuseAttentionGeometry,
) -> Result<&'a MuseGlimmerMetalLayerWeights<'weights>, MuseGlimmerLensError> {
    if target_block == 0 {
        return invalid("one-block replay requires a nonzero target block");
    }
    let layer = weights.layers.get(target_block as usize).ok_or_else(|| {
        MuseGlimmerLensError::Invalid(format!("target block {target_block} is out of range"))
    })?;
    if n_tokens == 0 || n_tokens > MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS {
        return invalid(format!(
            "replay token count must be in 1..={MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS}, got {n_tokens}"
        ));
    }
    validate_len(
        "replay input",
        input,
        checked_mul(n_tokens, geometry.hidden, "replay input elements")?,
    )?;
    require_finite("replay input", input)?;
    validate_layer_shapes(layer, geometry)?;
    Ok(layer)
}

fn validate_request(
    capture: &MuseGlimmerLensCapture,
    target_cotangent: &[f32],
    hidden_size: usize,
) -> Result<(), MuseGlimmerLensError> {
    if capture.target_block() == 0 {
        return invalid("one-block VJP requires a nonzero target block");
    }
    if capture.n_tokens() == 0 || capture.n_tokens() > MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS {
        return invalid(format!(
            "capture token count must be in 1..={MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS}, got {}",
            capture.n_tokens()
        ));
    }
    if capture.hidden_size() != hidden_size {
        return invalid(format!(
            "capture hidden size {} differs from model hidden size {hidden_size}",
            capture.hidden_size()
        ));
    }
    let expected = checked_mul(capture.n_tokens(), hidden_size, "capture elements")?;
    for (name, values) in [
        ("input residuals", capture.input_residuals()),
        (
            "post-attention residuals",
            capture.post_attention_residuals(),
        ),
        ("post-block residuals", capture.post_block_residuals()),
        ("target cotangent", target_cotangent),
    ] {
        validate_len(name, values, expected)?;
        require_finite(name, values)?;
    }
    Ok(())
}

fn validate_query_batch_request(
    capture: &MuseGlimmerLensCapture,
    target_cotangents: &[f32],
    query_count: usize,
    hidden_size: usize,
) -> Result<(), MuseGlimmerLensError> {
    validate_query_bank_values(
        target_cotangents,
        query_count,
        capture.n_tokens(),
        hidden_size,
    )?;
    let one_query_elements = checked_mul(capture.n_tokens(), hidden_size, "one query elements")?;
    validate_request(
        capture,
        &target_cotangents[..one_query_elements],
        hidden_size,
    )
}

fn validate_query_bank_values(
    target_cotangents: &[f32],
    query_count: usize,
    n_tokens: usize,
    hidden_size: usize,
) -> Result<(), MuseGlimmerLensError> {
    if query_count == 0 || query_count > MUSE_GLIMMER_QUERY_BATCH_MAX {
        return invalid(format!(
            "query count must be in 1..={MUSE_GLIMMER_QUERY_BATCH_MAX}, got {query_count}"
        ));
    }
    let one_query_elements = checked_mul(n_tokens, hidden_size, "one query elements")?;
    validate_len(
        "query-major target cotangents",
        target_cotangents,
        checked_mul(
            query_count,
            one_query_elements,
            "query-batch target elements",
        )?,
    )?;
    require_finite("query-major target cotangents", target_cotangents)
}

fn require_full_attention_block(
    weights: &MuseGlimmerMetalModelWeights<'_>,
    target_block: u32,
) -> Result<(), MuseGlimmerLensError> {
    let layer = weights.layers.get(target_block as usize).ok_or_else(|| {
        MuseGlimmerLensError::Invalid(format!("target block {target_block} is out of range"))
    })?;
    if layer.sliding_attention {
        return invalid(format!(
            "target block {target_block} uses sliding attention; this API requires full attention"
        ));
    }
    Ok(())
}

fn require_sliding_attention_block(
    weights: &MuseGlimmerMetalModelWeights<'_>,
    target_block: u32,
) -> Result<(), MuseGlimmerLensError> {
    let layer = weights.layers.get(target_block as usize).ok_or_else(|| {
        MuseGlimmerLensError::Invalid(format!("target block {target_block} is out of range"))
    })?;
    if !layer.sliding_attention {
        return invalid(format!(
            "target block {target_block} uses full attention; this API requires sliding attention"
        ));
    }
    Ok(())
}

fn validate_layer_shapes(
    layer: &MuseGlimmerMetalLayerWeights<'_>,
    geometry: MuseAttentionGeometry,
) -> Result<(), MuseGlimmerLensError> {
    for (name, tensor, shape) in [
        (
            "attention query",
            layer.attention_query,
            [geometry.hidden, geometry.query],
        ),
        (
            "attention key",
            layer.attention_key,
            [geometry.hidden, geometry.kv],
        ),
        (
            "attention value",
            layer.attention_value,
            [geometry.hidden, geometry.kv],
        ),
        (
            "attention gate",
            layer.attention_gate,
            [geometry.hidden, geometry.query],
        ),
        (
            "attention output",
            layer.attention_output,
            [geometry.query, geometry.hidden],
        ),
        (
            "FFN gate",
            layer.feed_forward_gate,
            [geometry.hidden, geometry.feed_forward],
        ),
        (
            "FFN up",
            layer.feed_forward_up,
            [geometry.hidden, geometry.feed_forward],
        ),
        (
            "FFN down",
            layer.feed_forward_down,
            [geometry.feed_forward, geometry.hidden],
        ),
    ] {
        if tensor.shape.as_slice() != [shape[0] as u64, shape[1] as u64] {
            return invalid(format!(
                "{name} shape {:?} differs from {shape:?}",
                tensor.shape
            ));
        }
    }
    Ok(())
}

fn run_command<F>(ctx: &MetalContext, encode: F) -> Result<(), MuseGlimmerLensError>
where
    F: FnOnce(&KernelEncoder) -> Result<(), MuseGlimmerLensError>,
{
    let command = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| MuseGlimmerLensError::CommandBuffer("allocation failed".into()))?;
    let encoder = KernelEncoder::begin(&command);
    let result = encode(&encoder);
    encoder.end();
    result?;
    command.commit();
    command.waitUntilCompleted();
    let status = command.status();
    let error = command.error().map(|error| error.to_string());
    if status != MTLCommandBufferStatus::Completed || error.is_some() {
        return Err(MuseGlimmerLensError::CommandBuffer(format!(
            "status={status:?}, error={error:?}"
        )));
    }
    Ok(())
}

fn row_shape(width: usize, rows: usize) -> Result<Vec<u64>, MuseGlimmerLensError> {
    checked_mul(width, rows, "row elements")?;
    Ok(vec![width as u64, rows as u64])
}

fn row_view(tensor: &MetalTensor, row: usize, width: usize) -> MetalTensor {
    tensor.view_subrange((row * width) as u64, vec![width as u64])
}

fn batch_token_view(
    tensor: &MetalTensor,
    primal_row: usize,
    query_count: usize,
    width: usize,
) -> MetalTensor {
    tensor.view_subrange(
        (primal_row * query_count * width) as u64,
        vec![width as u64, query_count as u64],
    )
}

fn from_f32(
    ctx: &MetalContext,
    values: &[f32],
    shape: Vec<u64>,
) -> Result<MetalTensor, MuseGlimmerLensError> {
    Ok(MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(values),
        shape,
        GgmlType::F32,
    )?)
}

fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
    let mut values = vec![0.0_f32; tensor.n_elements() as usize];
    unsafe {
        std::ptr::copy_nonoverlapping(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>(),
            values.as_mut_ptr(),
            values.len(),
        );
    }
    values
}

fn write_f32(tensor: &MetalTensor, values: &[f32]) -> Result<(), MuseGlimmerLensError> {
    validate_len("Metal tensor write", values, tensor.n_elements() as usize)?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>(),
            values.len(),
        );
    }
    Ok(())
}

fn max_abs_difference(left: &[f32], right: &[f32]) -> Result<f32, MuseGlimmerLensError> {
    validate_len("replay comparison", left, right.len())?;
    let mut maximum = 0.0_f32;
    for (&left, &right) in left.iter().zip(right) {
        let difference = (left - right).abs();
        if !difference.is_finite() {
            return invalid("non-finite replay error");
        }
        maximum = maximum.max(difference);
    }
    Ok(maximum)
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn sigmoid(value: f32) -> f32 {
    (1.0 + (-value).exp()).recip()
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, MuseGlimmerLensError> {
    left.checked_mul(right)
        .ok_or_else(|| MuseGlimmerLensError::Invalid(format!("{name} overflow")))
}

fn validate_len(name: &str, values: &[f32], expected: usize) -> Result<(), MuseGlimmerLensError> {
    if values.len() != expected {
        return invalid(format!(
            "{name} length {} differs from expected {expected}",
            values.len()
        ));
    }
    Ok(())
}

fn require_finite(name: &str, values: &[f32]) -> Result<(), MuseGlimmerLensError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return invalid(format!(
            "{name} contains a non-finite value at index {index}"
        ));
    }
    Ok(())
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, MuseGlimmerLensError> {
    Err(MuseGlimmerLensError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests;
