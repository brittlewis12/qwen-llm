//! Correctness-first text execution for Muse Glimmer 30B.
//!
//! The initial path uses native Q8_0/BF16 projections and a contiguous F16 KV
//! cache. Sliding layers use a suffix view while full-attention layers use the
//! entire prefix. Ordinary prefill batches superchunks of up to 128 tokens on
//! a 16-token quantum while retaining the scalar path for decode, tails,
//! captures, and interventions.

#[cfg(test)]
use crate::metal::encode_mat_mat_q8_0_f32;
use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalMemoryAdmission, MetalMemoryAdmissionReason,
    MetalTensor, PostBlockIntervention, encode_add_inplace_f32, encode_copy_offset_f32,
    encode_get_rows_f32, encode_mask_row_indices_f32, encode_mat_vec_f16_f32,
    encode_mat_vec_q8_0_batch_f32, encode_post_block_intervention_f32, encode_rms_norm_batched_f32,
    encode_rms_norm_mul_f32, encode_rms_norm_mul_rows_f32, encode_scatter_offset_f32_to_f16_kv,
    encode_sigmoid_mul_f32, encode_silu_mul_f32, encode_topk16_f32,
    evaluate_metal_memory_admission, evaluate_metal_memory_admission_with_cpu_bytes,
};
use crate::metal_forward::{MfError, encode_mat_vec_dispatch};
use crate::muse_glimmer::{MuseGlimmerConfig, MuseGlimmerError};
use crate::muse_glimmer_lens::{
    MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS, MuseGlimmerLensCapture, MuseGlimmerLensCaptureBank,
    MuseGlimmerLensError, MuseGlimmerSelectedTokenCovectors,
};
use crate::muse_glimmer_lens_fit::{
    MuseGlimmerAdjacentRowSlab, MuseGlimmerAdjacentSelectedTokenFit,
    MuseGlimmerBatchedFullTransportRowFit, MuseGlimmerFullTransportRowFit,
    MuseGlimmerMultiSourceSelectedTokenFit, MuseGlimmerOneBlockVjp,
    MuseGlimmerQueryBatchComposedVjp, MuseGlimmerQueryBatchOneBlockVjp,
    muse_glimmer_composed_vjp_query_batch, muse_glimmer_fit_adjacent_full_attention_rows_batched,
    muse_glimmer_fit_adjacent_full_attention_selected_tokens,
    muse_glimmer_fit_full_transport_rows_to_sources,
    muse_glimmer_fit_full_transport_rows_to_sources_batched,
    muse_glimmer_fit_selected_tokens_to_sources, muse_glimmer_one_attention_block_vjp_query_batch,
    muse_glimmer_one_full_attention_block_vjp,
    muse_glimmer_one_full_attention_block_vjp_query_batch,
};
use crate::muse_glimmer_metal::{
    MUSE_GLIMMER_MATERIALIZED_ATTENTION_MAX_POSITIONS, encode_muse_glimmer_attn_decode_f16kv_f32,
    encode_muse_glimmer_attn_prefill_f16kv_f32, encode_muse_glimmer_logit_softcap_f32,
    encode_muse_glimmer_rope_adjacent_pair_in_place_f32,
    encode_muse_glimmer_rope_adjacent_pair_rows_in_place_f32,
};
use crate::muse_glimmer_residency::{
    MuseGlimmerMetalModelWeights, MuseGlimmerMetalWeights, MuseGlimmerResidencyError,
};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLDevice, MTLResource,
};

pub const MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
pub const MUSE_GLIMMER_MAX_LIVE_CAPTURE_LAYERS: usize = 64;
pub const MUSE_GLIMMER_PACKED_PREFILL_QUANTUM: usize = 16;
pub const MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS: usize = 128;
const MUSE_GLIMMER_FULL_READOUT_PASS_K: usize = 16;
pub const MUSE_GLIMMER_FULL_READOUT_MAX_TOP_K: usize = MUSE_GLIMMER_FULL_READOUT_PASS_K * 2;
pub const MUSE_GLIMMER_FULL_READOUT_MAX_ROWS: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MuseGlimmerFullReadoutHeadMode {
    Q8Batch,
    Bf16ScalarRows,
}

fn muse_glimmer_full_readout_head_mode(
    dtype: GgmlType,
    q8_lcpp_enabled: bool,
) -> Option<MuseGlimmerFullReadoutHeadMode> {
    match dtype {
        GgmlType::Q8_0 if q8_lcpp_enabled => Some(MuseGlimmerFullReadoutHeadMode::Q8Batch),
        GgmlType::BF16 => Some(MuseGlimmerFullReadoutHeadMode::Bf16ScalarRows),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerPostBlockForward {
    pub position: usize,
    pub token_id: u32,
    pub layer_ids: Vec<u32>,
    pub hidden_size: usize,
    pub logits: Vec<f32>,
    /// Layer-major post-block residuals, flattened `[L,H]`.
    pub post_block_residuals: Vec<f32>,
}

impl MuseGlimmerPostBlockForward {
    pub fn layer_values(&self, slot: usize) -> Option<&[f32]> {
        let start = slot.checked_mul(self.hidden_size)?;
        self.post_block_residuals
            .get(start..start.checked_add(self.hidden_size)?)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MuseGlimmerTextSessionError {
    #[error(transparent)]
    Config(#[from] MuseGlimmerError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Forward(#[from] MfError),
    #[error(transparent)]
    Residency(#[from] MuseGlimmerResidencyError),
    #[error("invalid Muse Glimmer text-session contract: {0}")]
    Invalid(String),
    #[error("Muse Glimmer command buffer failed: {0}")]
    CommandBuffer(String),
    #[error(
        "Muse Glimmer full-readout workspace admission denied: reason={reason:?} required={required_bytes:?} working_set_headroom={working_set_headroom_bytes:?} process_remaining={process_remaining_bytes:?}"
    )]
    FullReadoutMemoryAdmissionDenied {
        reason: MetalMemoryAdmissionReason,
        required_bytes: Option<u64>,
        working_set_headroom_bytes: Option<u64>,
        process_remaining_bytes: Option<u64>,
    },
    #[error("Muse Glimmer prefill checkpoint failed: {0}")]
    Checkpoint(String),
    #[error("Muse Glimmer text session is poisoned: {0}")]
    Poisoned(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MuseGlimmerTextGeometry {
    layer_count: usize,
    hidden_size: usize,
    feed_forward_size: usize,
    vocab_size: usize,
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    query_width: usize,
    kv_width: usize,
    sliding_window: usize,
    capacity: usize,
}

impl MuseGlimmerTextGeometry {
    pub fn from_config(
        config: &MuseGlimmerConfig,
        capacity: usize,
    ) -> Result<Self, MuseGlimmerTextSessionError> {
        config.validate_release_profile()?;
        if capacity == 0 || capacity > config.context_length as usize {
            return invalid(format!(
                "capacity must be in 1..={} for the released model context, got {capacity}",
                config.context_length
            ));
        }
        let query_width = usize::try_from(config.query_width()?).map_err(|_| {
            MuseGlimmerTextSessionError::Invalid("query width exceeds usize".into())
        })?;
        let kv_width = usize::try_from(config.kv_width()?)
            .map_err(|_| MuseGlimmerTextSessionError::Invalid("KV width exceeds usize".into()))?;
        let geometry = Self {
            layer_count: config.layer_count as usize,
            hidden_size: config.hidden_size as usize,
            feed_forward_size: config.feed_forward_size as usize,
            vocab_size: config.vocab_size as usize,
            query_head_count: config.query_head_count as usize,
            kv_head_count: config.kv_head_count as usize,
            head_dim: config.key_head_dim as usize,
            query_width,
            kv_width,
            sliding_window: config.sliding_window as usize,
            capacity,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    fn validate(&self) -> Result<(), MuseGlimmerTextSessionError> {
        if self.query_width
            != self
                .query_head_count
                .checked_mul(self.head_dim)
                .ok_or_else(|| {
                    MuseGlimmerTextSessionError::Invalid("query geometry overflow".into())
                })?
            || self.kv_width
                != self
                    .kv_head_count
                    .checked_mul(self.head_dim)
                    .ok_or_else(|| {
                        MuseGlimmerTextSessionError::Invalid("KV geometry overflow".into())
                    })?
            || !self.query_head_count.is_multiple_of(self.kv_head_count)
            || self.sliding_window == 0
        {
            return invalid("inconsistent Muse Glimmer attention geometry");
        }
        self.cache_elements()?;
        for (name, value) in [
            ("hidden size", self.hidden_size),
            ("feed-forward size", self.feed_forward_size),
            ("vocabulary size", self.vocab_size),
            ("query width", self.query_width),
            ("KV width", self.kv_width),
        ] {
            if value == 0 || u32::try_from(value).is_err() {
                return invalid(format!("{name} must fit nonzero u32, got {value}"));
            }
        }
        Ok(())
    }

    fn cache_elements(&self) -> Result<usize, MuseGlimmerTextSessionError> {
        self.layer_count
            .checked_mul(self.capacity)
            .and_then(|value| value.checked_mul(self.kv_width))
            .ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid("KV cache element count overflow".into())
            })
    }

    fn cache_write_offset(
        &self,
        layer: usize,
        position: usize,
    ) -> Result<usize, MuseGlimmerTextSessionError> {
        if layer >= self.layer_count || position >= self.capacity {
            return invalid(format!(
                "cache write layer/position {layer}/{position} exceeds {}/{}",
                self.layer_count, self.capacity
            ));
        }
        layer
            .checked_mul(self.capacity)
            .and_then(|value| value.checked_add(position))
            .and_then(|value| value.checked_mul(self.kv_width))
            .ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid("KV cache write offset overflow".into())
            })
    }

    fn visible_cache_range(
        &self,
        layer: usize,
        position: usize,
        sliding: bool,
    ) -> Result<(usize, usize), MuseGlimmerTextSessionError> {
        if layer >= self.layer_count || position >= self.capacity {
            return invalid(format!(
                "visible cache layer/position {layer}/{position} exceeds {}/{}",
                self.layer_count, self.capacity
            ));
        }
        let end = position + 1;
        let start = if sliding {
            end.saturating_sub(self.sliding_window)
        } else {
            0
        };
        let count = end - start;
        let element_offset = layer
            .checked_mul(self.capacity)
            .and_then(|value| value.checked_add(start))
            .and_then(|value| value.checked_mul(self.kv_width))
            .ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid("visible KV offset overflow".into())
            })?;
        let element_count = count.checked_mul(self.kv_width).ok_or_else(|| {
            MuseGlimmerTextSessionError::Invalid("visible KV length overflow".into())
        })?;
        Ok((element_offset, element_count))
    }

    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn feed_forward_size(&self) -> usize {
        self.feed_forward_size
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MuseGlimmerTextSessionAllocation {
    pub name: String,
    pub logical_bytes: u64,
    pub priced_bytes: u64,
    pub alignment: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MuseGlimmerTextSessionMemoryPlan {
    logical_bytes: u64,
    priced_upper_bytes: u64,
    allocations: Vec<MuseGlimmerTextSessionAllocation>,
}

impl MuseGlimmerTextSessionMemoryPlan {
    pub fn for_geometry(
        ctx: &MetalContext,
        geometry: &MuseGlimmerTextGeometry,
    ) -> Result<Self, MuseGlimmerTextSessionError> {
        let mut allocations = Vec::new();
        let mut logical_bytes = 0_u64;
        let mut priced_upper_bytes = 0_u64;
        for (name, logical) in session_allocation_specs(geometry)? {
            let priced = ctx.price_shared_buffer_upper(logical).map_err(|error| {
                MuseGlimmerTextSessionError::Invalid(format!("session allocation {name:?} {error}"))
                })?;
            let (priced_bytes, alignment) = (priced.priced_upper_bytes, priced.alignment);
            logical_bytes = logical_bytes.checked_add(logical).ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid("session logical byte total overflow".into())
            })?;
            priced_upper_bytes = priced_upper_bytes
                .checked_add(priced_bytes)
                .ok_or_else(|| {
                    MuseGlimmerTextSessionError::Invalid(
                        "session priced byte total overflow".into(),
                    )
                })?;
            allocations.push(MuseGlimmerTextSessionAllocation {
                name,
                logical_bytes: logical,
                priced_bytes,
                alignment,
            });
        }
        if priced_upper_bytes < logical_bytes {
            return invalid("priced session bytes are below logical bytes");
        }
        Ok(Self {
            logical_bytes,
            priced_upper_bytes,
            allocations,
        })
    }

    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    pub fn priced_upper_bytes(&self) -> u64 {
        self.priced_upper_bytes
    }

    pub fn allocations(&self) -> &[MuseGlimmerTextSessionAllocation] {
        &self.allocations
    }

    pub fn admission(&self, signals: crate::metal::MetalMemorySignals) -> MetalMemoryAdmission {
        evaluate_metal_memory_admission(
            self.priced_upper_bytes,
            MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES,
            signals,
            true,
        )
    }

    fn reconcile(
        &self,
        allocated_before: u64,
        allocated_after: u64,
    ) -> Result<u64, MuseGlimmerTextSessionError> {
        let observed = allocated_after.saturating_sub(allocated_before);
        if observed > self.priced_upper_bytes {
            return invalid(format!(
                "observed session allocation {observed} exceeds priced upper bound {}",
                self.priced_upper_bytes
            ));
        }
        Ok(observed)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MuseGlimmerPrefillPlan {
    packed_chunks: usize,
    packed_tokens: usize,
    scalar_tail: usize,
}

impl MuseGlimmerPrefillPlan {
    fn for_tokens(token_count: usize) -> Self {
        let packed_tokens =
            token_count / MUSE_GLIMMER_PACKED_PREFILL_QUANTUM * MUSE_GLIMMER_PACKED_PREFILL_QUANTUM;
        Self {
            packed_chunks: packed_tokens / MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS
                + usize::from(
                    !packed_tokens.is_multiple_of(MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS),
                ),
            packed_tokens,
            scalar_tail: token_count - packed_tokens,
        }
    }

    fn packed_tokens(self) -> usize {
        self.packed_tokens
    }
}

struct MuseGlimmerPackedPrefillWorkspace {
    ids: MetalTensor,
    residual: MetalTensor,
    normed: MetalTensor,
    branch_raw: MetalTensor,
    branch_normed: MetalTensor,
    query_raw: MetalTensor,
    query: MetalTensor,
    key_raw: MetalTensor,
    key: MetalTensor,
    value: MetalTensor,
    attention_gate: MetalTensor,
    attention_output: MetalTensor,
    feed_forward_gate: MetalTensor,
    feed_forward_up: MetalTensor,
}

struct MuseGlimmerPackedPrefillViews {
    ids: MetalTensor,
    residual: MetalTensor,
    normed: MetalTensor,
    branch_raw: MetalTensor,
    branch_normed: MetalTensor,
    query_raw: MetalTensor,
    query: MetalTensor,
    key_raw: MetalTensor,
    key: MetalTensor,
    value: MetalTensor,
    attention_gate: MetalTensor,
    attention_output: MetalTensor,
    feed_forward_gate: MetalTensor,
    feed_forward_up: MetalTensor,
}

impl MuseGlimmerPackedPrefillWorkspace {
    fn new(
        ctx: &MetalContext,
        geometry: &MuseGlimmerTextGeometry,
    ) -> Result<Self, MuseGlimmerTextSessionError> {
        let rows = MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS as u64;
        let hidden = geometry.hidden_size as u64;
        let query = geometry.query_width as u64;
        let kv = geometry.kv_width as u64;
        let feed_forward = geometry.feed_forward_size as u64;
        Ok(Self {
            ids: MetalTensor::zeros_i32(ctx, vec![rows])?,
            residual: MetalTensor::zeros_f32(ctx, vec![hidden, rows])?,
            normed: MetalTensor::zeros_f32(ctx, vec![hidden, rows])?,
            branch_raw: MetalTensor::zeros_f32(ctx, vec![hidden, rows])?,
            branch_normed: MetalTensor::zeros_f32(ctx, vec![hidden, rows])?,
            query_raw: MetalTensor::zeros_f32(ctx, vec![query, rows])?,
            query: MetalTensor::zeros_f32(ctx, vec![query, rows])?,
            key_raw: MetalTensor::zeros_f32(ctx, vec![kv, rows])?,
            key: MetalTensor::zeros_f32(ctx, vec![kv, rows])?,
            value: MetalTensor::zeros_f32(ctx, vec![kv, rows])?,
            attention_gate: MetalTensor::zeros_f32(ctx, vec![query, rows])?,
            attention_output: MetalTensor::zeros_f32(ctx, vec![query, rows])?,
            feed_forward_gate: MetalTensor::zeros_f32(ctx, vec![feed_forward, rows])?,
            feed_forward_up: MetalTensor::zeros_f32(ctx, vec![feed_forward, rows])?,
        })
    }

    fn tensors(&self) -> [&MetalTensor; 14] {
        [
            &self.ids,
            &self.residual,
            &self.normed,
            &self.branch_raw,
            &self.branch_normed,
            &self.query_raw,
            &self.query,
            &self.key_raw,
            &self.key,
            &self.value,
            &self.attention_gate,
            &self.attention_output,
            &self.feed_forward_gate,
            &self.feed_forward_up,
        ]
    }

    fn write_tokens(&self, tokens: &[u32]) -> Result<(), MuseGlimmerTextSessionError> {
        if tokens.len() < MUSE_GLIMMER_PACKED_PREFILL_QUANTUM
            || tokens.len() > MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS
            || !tokens
                .len()
                .is_multiple_of(MUSE_GLIMMER_PACKED_PREFILL_QUANTUM)
        {
            return invalid(format!(
                "packed prefill requires a multiple of {MUSE_GLIMMER_PACKED_PREFILL_QUANTUM} tokens in {}..={}, got {}",
                MUSE_GLIMMER_PACKED_PREFILL_QUANTUM,
                MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS,
                tokens.len()
            ));
        }
        let pointer = unsafe {
            self.ids
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(self.ids.offset as usize)
                .cast::<i32>()
        };
        for (index, &token) in tokens.iter().enumerate() {
            let token = i32::try_from(token).map_err(|_| {
                MuseGlimmerTextSessionError::Invalid(format!(
                    "packed token {token} exceeds signed 32-bit row-id range"
                ))
            })?;
            unsafe { pointer.add(index).write(token) };
        }
        Ok(())
    }

    fn views(
        &self,
        geometry: &MuseGlimmerTextGeometry,
        rows: usize,
    ) -> Result<MuseGlimmerPackedPrefillViews, MuseGlimmerTextSessionError> {
        if rows < MUSE_GLIMMER_PACKED_PREFILL_QUANTUM
            || rows > MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS
            || !rows.is_multiple_of(MUSE_GLIMMER_PACKED_PREFILL_QUANTUM)
        {
            return invalid(format!(
                "packed workspace rows must be a multiple of {MUSE_GLIMMER_PACKED_PREFILL_QUANTUM} in {}..={}, got {rows}",
                MUSE_GLIMMER_PACKED_PREFILL_QUANTUM, MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS,
            ));
        }
        let view = |tensor: &MetalTensor, width: usize| {
            tensor.view_subrange(0, vec![width as u64, rows as u64])
        };
        Ok(MuseGlimmerPackedPrefillViews {
            ids: self.ids.view_subrange(0, vec![rows as u64]),
            residual: view(&self.residual, geometry.hidden_size),
            normed: view(&self.normed, geometry.hidden_size),
            branch_raw: view(&self.branch_raw, geometry.hidden_size),
            branch_normed: view(&self.branch_normed, geometry.hidden_size),
            query_raw: view(&self.query_raw, geometry.query_width),
            query: view(&self.query, geometry.query_width),
            key_raw: view(&self.key_raw, geometry.kv_width),
            key: view(&self.key, geometry.kv_width),
            value: view(&self.value, geometry.kv_width),
            attention_gate: view(&self.attention_gate, geometry.query_width),
            attention_output: view(&self.attention_output, geometry.query_width),
            feed_forward_gate: view(&self.feed_forward_gate, geometry.feed_forward_size),
            feed_forward_up: view(&self.feed_forward_up, geometry.feed_forward_size),
        })
    }
}

pub struct MuseGlimmerTextSession {
    geometry: MuseGlimmerTextGeometry,
    memory_plan: MuseGlimmerTextSessionMemoryPlan,
    admission: MetalMemoryAdmission,
    observed_allocation_delta: u64,
    device_registry_id: u64,
    next_position: usize,
    poison_reason: Option<String>,
    ids: MetalTensor,
    embedding_norm_weight: MetalTensor,
    residual: MetalTensor,
    normed: MetalTensor,
    branch_raw: MetalTensor,
    branch_normed: MetalTensor,
    query_raw: MetalTensor,
    query: MetalTensor,
    key_raw: MetalTensor,
    key: MetalTensor,
    value: MetalTensor,
    attention_gate: MetalTensor,
    attention_output: MetalTensor,
    feed_forward_gate: MetalTensor,
    feed_forward_up: MetalTensor,
    packed: MuseGlimmerPackedPrefillWorkspace,
    logits: MetalTensor,
    key_cache: MetalTensor,
    value_cache: MetalTensor,
}

impl MuseGlimmerTextSession {
    pub fn new(
        ctx: &MetalContext,
        config: &MuseGlimmerConfig,
        capacity: usize,
    ) -> Result<Self, MuseGlimmerTextSessionError> {
        let geometry = MuseGlimmerTextGeometry::from_config(config, capacity)?;
        let memory_plan = MuseGlimmerTextSessionMemoryPlan::for_geometry(ctx, &geometry)?;
        let _allocation_transaction = ctx.begin_allocation_transaction();
        let admission = memory_plan.admission(ctx.memory_signals());
        if !admission.admitted {
            return invalid(format!(
                "Metal session admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                admission.reason.as_str(),
                admission.required_bytes,
                admission.working_set_headroom_bytes,
                admission.signals.process_limit_remaining_bytes
            ));
        }

        let allocated_before = ctx.current_allocated_size();
        let hidden = geometry.hidden_size as u64;
        let query = geometry.query_width as u64;
        let kv = geometry.kv_width as u64;
        let feed_forward = geometry.feed_forward_size as u64;
        let cache_elements = geometry.cache_elements()? as u64;
        let embedding_norm_weight = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&vec![1.0_f32; geometry.hidden_size]),
            vec![hidden],
            GgmlType::F32,
        )?;
        let packed = MuseGlimmerPackedPrefillWorkspace::new(ctx, &geometry)?;
        let session = Self {
            geometry,
            memory_plan,
            admission,
            observed_allocation_delta: 0,
            device_registry_id: ctx.device.registryID(),
            next_position: 0,
            poison_reason: None,
            ids: MetalTensor::zeros_i32(ctx, vec![1])?,
            embedding_norm_weight,
            residual: MetalTensor::zeros_f32(ctx, vec![hidden])?,
            normed: MetalTensor::zeros_f32(ctx, vec![hidden])?,
            branch_raw: MetalTensor::zeros_f32(ctx, vec![hidden])?,
            branch_normed: MetalTensor::zeros_f32(ctx, vec![hidden])?,
            query_raw: MetalTensor::zeros_f32(ctx, vec![query])?,
            query: MetalTensor::zeros_f32(ctx, vec![query])?,
            key_raw: MetalTensor::zeros_f32(ctx, vec![kv])?,
            key: MetalTensor::zeros_f32(ctx, vec![kv])?,
            value: MetalTensor::zeros_f32(ctx, vec![kv])?,
            attention_gate: MetalTensor::zeros_f32(ctx, vec![query])?,
            attention_output: MetalTensor::zeros_f32(ctx, vec![query])?,
            feed_forward_gate: MetalTensor::zeros_f32(ctx, vec![feed_forward])?,
            feed_forward_up: MetalTensor::zeros_f32(ctx, vec![feed_forward])?,
            packed,
            logits: MetalTensor::zeros_f32(ctx, vec![config.vocab_size as u64])?,
            key_cache: MetalTensor::zeros_f16(ctx, vec![cache_elements])?,
            value_cache: MetalTensor::zeros_f16(ctx, vec![cache_elements])?,
        };
        let allocated_after = ctx.current_allocated_size();
        let observed_allocation_delta = session
            .memory_plan
            .reconcile(allocated_before, allocated_after)?;
        Ok(Self {
            observed_allocation_delta,
            ..session
        })
    }

    pub fn geometry(&self) -> &MuseGlimmerTextGeometry {
        &self.geometry
    }

    pub fn memory_plan(&self) -> &MuseGlimmerTextSessionMemoryPlan {
        &self.memory_plan
    }

    pub fn admission(&self) -> MetalMemoryAdmission {
        self.admission
    }

    pub fn observed_allocation_delta(&self) -> u64 {
        self.observed_allocation_delta
    }

    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn remaining_forwards(&self) -> usize {
        self.geometry.capacity - self.next_position
    }

    pub fn reset(&mut self) -> Result<(), MuseGlimmerTextSessionError> {
        self.rewind_prefix(0)
    }

    /// Discard a suffix of the current causal KV history without copying it.
    /// All layers retain absolute-position KV, including sliding-attention layers.
    /// Scratch/logits are not restored; callers must forward again before readout.
    pub fn rewind_prefix(&mut self, position: usize) -> Result<(), MuseGlimmerTextSessionError> {
        self.ensure_usable()?;
        if position > self.next_position {
            return invalid(format!(
                "cannot rewind from {} to future position {position}",
                self.next_position
            ));
        }
        self.next_position = position;
        Ok(())
    }

    fn ensure_usable(&self) -> Result<(), MuseGlimmerTextSessionError> {
        if let Some(reason) = &self.poison_reason {
            return Err(MuseGlimmerTextSessionError::Poisoned(reason.clone()));
        }
        Ok(())
    }

    fn aliases_mutable_buffer(&self, candidate: &MetalTensor) -> bool {
        let scalar_alias = [
            &self.ids,
            &self.embedding_norm_weight,
            &self.residual,
            &self.normed,
            &self.branch_raw,
            &self.branch_normed,
            &self.query_raw,
            &self.query,
            &self.key_raw,
            &self.key,
            &self.value,
            &self.attention_gate,
            &self.attention_output,
            &self.feed_forward_gate,
            &self.feed_forward_up,
            &self.logits,
            &self.key_cache,
            &self.value_cache,
        ]
        .into_iter()
        .any(|tensor| Retained::as_ptr(&candidate.buffer) == Retained::as_ptr(&tensor.buffer));
        scalar_alias
            || self.packed.tensors().into_iter().any(|tensor| {
                Retained::as_ptr(&candidate.buffer) == Retained::as_ptr(&tensor.buffer)
            })
    }

    fn write_token(&self, token: i32) {
        unsafe {
            let pointer = self.ids.buffer.contents().as_ptr().cast::<i32>();
            pointer
                .add(self.ids.offset as usize / std::mem::size_of::<i32>())
                .write(token);
        }
    }

    fn write_residual(&self, residual: &[f32]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                residual.as_ptr(),
                self.residual
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(self.residual.offset as usize)
                    .cast::<f32>(),
                residual.len(),
            );
        }
    }

    fn cache_views(
        &self,
        layer: usize,
        position: usize,
        sliding: bool,
    ) -> Result<(MetalTensor, MetalTensor, usize), MuseGlimmerTextSessionError> {
        let (offset, elements) = self
            .geometry
            .visible_cache_range(layer, position, sliding)?;
        let count = elements / self.geometry.kv_width;
        Ok((
            self.key_cache
                .view_subrange(offset as u64, vec![elements as u64]),
            self.value_cache
                .view_subrange(offset as u64, vec![elements as u64]),
            count,
        ))
    }

    fn cache_prefix_views(
        &self,
        layer: usize,
        end_position: usize,
    ) -> Result<(MetalTensor, MetalTensor), MuseGlimmerTextSessionError> {
        if layer >= self.geometry.layer_count
            || end_position == 0
            || end_position > self.geometry.capacity
        {
            return invalid(format!(
                "cache prefix layer/end {layer}/{end_position} exceeds {}/{}",
                self.geometry.layer_count, self.geometry.capacity
            ));
        }
        let offset = layer
            .checked_mul(self.geometry.capacity)
            .and_then(|value| value.checked_mul(self.geometry.kv_width))
            .ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid("cache prefix offset overflow".into())
            })?;
        let elements = end_position
            .checked_mul(self.geometry.kv_width)
            .ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid("cache prefix length overflow".into())
            })?;
        Ok((
            self.key_cache
                .view_subrange(offset as u64, vec![elements as u64]),
            self.value_cache
                .view_subrange(offset as u64, vec![elements as u64]),
        ))
    }

    fn read_logits(&self) -> Vec<f32> {
        let mut logits = vec![0.0_f32; self.geometry.vocab_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.logits
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<f32>()
                    .add(self.logits.offset as usize / std::mem::size_of::<f32>()),
                logits.as_mut_ptr(),
                logits.len(),
            );
        }
        logits
    }
}

pub struct MuseGlimmerTextForward<'ctx, 'model> {
    ctx: &'ctx MetalContext,
    weights: MuseGlimmerMetalModelWeights<'model>,
    #[cfg(test)]
    packed_q8_mat_mat: bool,
}

pub struct MuseGlimmerPreparedF16Transport {
    tensor: MetalTensor,
    hidden_size: usize,
    device_registry_id: u64,
}

/// Reusable GPU storage for transport/output-tail readout over independent rows.
/// Rows need not belong to one prompt, which leaves batching policy with callers.
pub struct MuseGlimmerFullReadoutWorkspace {
    device_registry_id: u64,
    row_capacity: usize,
    hidden_size: usize,
    vocab_size: usize,
    source: MetalTensor,
    transported: MetalTensor,
    normalized: MetalTensor,
    logits: MetalTensor,
    first_ids: MetalTensor,
    first_values: MetalTensor,
    second_ids: MetalTensor,
    second_values: MetalTensor,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MuseGlimmerFullReadoutWorkspacePlan {
    row_capacity: usize,
    logical_bytes: u64,
    priced_upper_bytes: u64,
    prepared_transport_reserve_bytes: u64,
    host_transport_reserve_bytes: u64,
}

impl MuseGlimmerFullReadoutWorkspacePlan {
    pub fn for_model(
        ctx: &MetalContext,
        hidden_size: usize,
        vocab_size: usize,
        row_capacity: usize,
    ) -> Result<Self, MuseGlimmerTextSessionError> {
        if row_capacity == 0 || row_capacity > MUSE_GLIMMER_FULL_READOUT_MAX_ROWS {
            return invalid(format!(
                "full-readout workspace row capacity must be in 1..={}, got {row_capacity}",
                MUSE_GLIMMER_FULL_READOUT_MAX_ROWS
            ));
        }
        let rows = row_capacity as u64;
        let hidden = hidden_size as u64;
        let vocab = vocab_size as u64;
        let f32_bytes = std::mem::size_of::<f32>() as u64;
        let i32_bytes = std::mem::size_of::<i32>() as u64;
        let row_hidden_bytes = rows
            .checked_mul(hidden)
            .and_then(|elements| elements.checked_mul(f32_bytes))
            .ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid("workspace size overflow".into())
            })?;
        let logits_bytes = rows
            .checked_mul(vocab)
            .and_then(|elements| elements.checked_mul(f32_bytes))
            .ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid("workspace size overflow".into())
            })?;
        let compact_elements = rows
            .checked_mul(MUSE_GLIMMER_FULL_READOUT_PASS_K as u64)
            .ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid("workspace size overflow".into())
            })?;
        let specs = [
            row_hidden_bytes,
            row_hidden_bytes,
            row_hidden_bytes,
            logits_bytes,
            compact_elements * i32_bytes,
            compact_elements * f32_bytes,
            compact_elements * i32_bytes,
            compact_elements * f32_bytes,
        ];
        let mut logical_bytes = 0_u64;
        let mut priced_upper_bytes = 0_u64;
        for logical in specs {
            let aligned = ctx
                .price_shared_buffer_upper(logical)
                .map_err(|error| {
                    MuseGlimmerTextSessionError::Invalid(format!(
                        "full-readout workspace allocation {error}"
                    ))
                })?
                .priced_upper_bytes;
            logical_bytes = logical_bytes.checked_add(logical).ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid("workspace size overflow".into())
            })?;
            priced_upper_bytes = priced_upper_bytes.checked_add(aligned).ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid("workspace pricing overflow".into())
            })?;
        }
        let transport_logical = hidden
            .checked_mul(hidden)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<half::f16>() as u64))
            .ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid("transport reserve overflow".into())
            })?;
        let prepared_transport_reserve_bytes = ctx
            .price_shared_buffer_upper(transport_logical)
            .map_err(|error| {
                MuseGlimmerTextSessionError::Invalid(format!("prepared-transport reserve {error}"))
            })?
            .priced_upper_bytes;
        Ok(Self {
            row_capacity,
            logical_bytes,
            priced_upper_bytes,
            prepared_transport_reserve_bytes,
            host_transport_reserve_bytes: transport_logical,
        })
    }

    pub fn row_capacity(&self) -> usize {
        self.row_capacity
    }

    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    pub fn priced_upper_bytes(&self) -> u64 {
        self.priced_upper_bytes
    }

    pub fn prepared_transport_reserve_bytes(&self) -> u64 {
        self.prepared_transport_reserve_bytes
    }

    pub fn host_transport_reserve_bytes(&self) -> u64 {
        self.host_transport_reserve_bytes
    }

    pub fn admission(&self, ctx: &MetalContext) -> MetalMemoryAdmission {
        evaluate_metal_memory_admission_with_cpu_bytes(
            self.priced_upper_bytes,
            self.host_transport_reserve_bytes,
            self.prepared_transport_reserve_bytes,
            ctx.memory_signals(),
            true,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerFullReadoutScore {
    pub token_id: u32,
    pub logit: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerFullReadoutRow {
    pub row: usize,
    pub scores: Vec<MuseGlimmerFullReadoutScore>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerTransportedRow {
    pub row: usize,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerBatchedFullReadout {
    pub row_count: usize,
    pub top_k: usize,
    pub gpu_ms: f64,
    pub command_wall_ms: f64,
    pub transport_gpu_ms: f64,
    pub transport_wall_ms: f64,
    pub output_tail_gpu_ms: f64,
    pub output_tail_wall_ms: f64,
    pub rows: Vec<MuseGlimmerFullReadoutRow>,
    pub transported_rows: Vec<MuseGlimmerTransportedRow>,
}

impl<'ctx, 'model> MuseGlimmerTextForward<'ctx, 'model> {
    pub fn new(
        ctx: &'ctx MetalContext,
        resident: &'model MuseGlimmerMetalWeights,
    ) -> Result<Self, MuseGlimmerTextSessionError> {
        resident.validate_context(ctx)?;
        let weights = MuseGlimmerMetalModelWeights::bind(resident)?;
        Ok(Self {
            ctx,
            weights,
            #[cfg(test)]
            packed_q8_mat_mat: false,
        })
    }

    #[cfg(test)]
    fn new_with_packed_q8_mat_mat(
        ctx: &'ctx MetalContext,
        resident: &'model MuseGlimmerMetalWeights,
        packed_q8_mat_mat: bool,
    ) -> Result<Self, MuseGlimmerTextSessionError> {
        let mut forward = Self::new(ctx, resident)?;
        forward.packed_q8_mat_mat = packed_q8_mat_mat;
        Ok(forward)
    }

    pub fn forward_token(
        &self,
        token: u32,
        session: &mut MuseGlimmerTextSession,
    ) -> Result<Vec<f32>, MuseGlimmerTextSessionError> {
        self.forward_token_with_post_block_interventions(token, &[], session)
    }

    pub fn forward_token_with_post_block_interventions(
        &self,
        token: u32,
        interventions: &[PostBlockIntervention<'_>],
        session: &mut MuseGlimmerTextSession,
    ) -> Result<Vec<f32>, MuseGlimmerTextSessionError> {
        self.validate_token_and_session(token, session)?;
        self.execute_token_with_sink(token, session, true, None, interventions)?
            .ok_or_else(|| MuseGlimmerTextSessionError::Invalid("logits were not produced".into()))
    }

    pub(crate) fn deployed_logits_from_post_block_residual(
        &self,
        residual: &[f32],
        session: &mut MuseGlimmerTextSession,
    ) -> Result<Vec<f32>, MuseGlimmerTextSessionError> {
        session.ensure_usable()?;
        self.validate_session_geometry(session)?;
        validate_deployed_output_residual(residual, session.geometry.hidden_size)?;
        session.write_residual(residual);

        let command = self.ctx.queue.commandBuffer().ok_or_else(|| {
            MuseGlimmerTextSessionError::CommandBuffer("allocation failed".into())
        })?;
        let encode_result = (|| {
            let encoder = KernelEncoder::begin(&command);
            self.encode_deployed_output_tail(&encoder, session)?;
            encoder.end();
            Ok::<(), MuseGlimmerTextSessionError>(())
        })();
        encode_result?;

        command.commit();
        command.waitUntilCompleted();
        let status = command.status();
        let command_error = command.error().map(|error| error.to_string());
        if status != MTLCommandBufferStatus::Completed || command_error.is_some() {
            let reason = format!("status={status:?}, error={command_error:?}");
            session.poison_reason = Some(reason.clone());
            return Err(MuseGlimmerTextSessionError::CommandBuffer(reason));
        }
        Ok(session.read_logits())
    }

    pub(crate) fn apply_f16_post_block_transport(
        &self,
        transport_bytes: &[u8],
        source_residual: &[f32],
    ) -> Result<Vec<f32>, MuseGlimmerTextSessionError> {
        let transport = self.prepare_f16_post_block_transport(transport_bytes)?;
        self.apply_prepared_f16_post_block_transport(&transport, source_residual)
    }

    pub(crate) fn prepare_f16_post_block_transport(
        &self,
        transport_bytes: &[u8],
    ) -> Result<MuseGlimmerPreparedF16Transport, MuseGlimmerTextSessionError> {
        let hidden = self.weights.config.hidden_size as usize;
        validate_f16_transport(transport_bytes, hidden)?;
        let tensor = MetalTensor::from_bytes(
            self.ctx,
            transport_bytes,
            vec![hidden as u64, hidden as u64],
            GgmlType::F16,
        )?;
        Ok(MuseGlimmerPreparedF16Transport {
            tensor,
            hidden_size: hidden,
            device_registry_id: self.ctx.device.registryID(),
        })
    }

    pub(crate) fn full_readout_workspace_plan(
        &self,
        row_capacity: usize,
    ) -> Result<MuseGlimmerFullReadoutWorkspacePlan, MuseGlimmerTextSessionError> {
        MuseGlimmerFullReadoutWorkspacePlan::for_model(
            self.ctx,
            self.weights.config.hidden_size as usize,
            self.weights.config.vocab_size as usize,
            row_capacity,
        )
    }

    fn full_readout_head_mode(&self) -> Option<MuseGlimmerFullReadoutHeadMode> {
        muse_glimmer_full_readout_head_mode(
            self.weights.output.dtype,
            crate::metal::mat_vec_q8_0_lcpp_enabled(),
        )
    }

    pub(crate) fn supports_command_batched_full_readout(&self) -> bool {
        self.full_readout_head_mode().is_some()
    }

    pub(crate) fn create_full_readout_workspace(
        &self,
        row_capacity: usize,
    ) -> Result<MuseGlimmerFullReadoutWorkspace, MuseGlimmerTextSessionError> {
        let hidden_size = self.weights.config.hidden_size as usize;
        let vocab_size = self.weights.config.vocab_size as usize;
        let plan = self.full_readout_workspace_plan(row_capacity)?;
        let _allocation_transaction = self.ctx.begin_allocation_transaction();
        let admission = plan.admission(self.ctx);
        if !admission.admitted {
            return Err(
                MuseGlimmerTextSessionError::FullReadoutMemoryAdmissionDenied {
                    reason: admission.reason,
                    required_bytes: admission.required_bytes,
                    working_set_headroom_bytes: admission.working_set_headroom_bytes,
                    process_remaining_bytes: admission.signals.process_limit_remaining_bytes,
                },
            );
        }
        let rows = row_capacity as u64;
        let pass_k = MUSE_GLIMMER_FULL_READOUT_PASS_K as u64;
        Ok(MuseGlimmerFullReadoutWorkspace {
            device_registry_id: self.ctx.device.registryID(),
            row_capacity,
            hidden_size,
            vocab_size,
            source: MetalTensor::zeros_f32(self.ctx, vec![rows, hidden_size as u64])?,
            transported: MetalTensor::zeros_f32(self.ctx, vec![rows, hidden_size as u64])?,
            normalized: MetalTensor::zeros_f32(self.ctx, vec![rows, hidden_size as u64])?,
            logits: MetalTensor::zeros_f32(self.ctx, vec![rows, vocab_size as u64])?,
            first_ids: MetalTensor::zeros_i32(self.ctx, vec![rows, pass_k])?,
            first_values: MetalTensor::zeros_f32(self.ctx, vec![rows, pass_k])?,
            second_ids: MetalTensor::zeros_i32(self.ctx, vec![rows, pass_k])?,
            second_values: MetalTensor::zeros_f32(self.ctx, vec![rows, pass_k])?,
        })
    }

    /// Batch independent F32 source rows through one prepared F16 transport,
    /// the deployed output tail, and exact compact top-k. Only requested
    /// transported rows and the compact score set cross back to the host.
    pub(crate) fn apply_prepared_f16_transport_topk_rows(
        &self,
        workspace: &mut MuseGlimmerFullReadoutWorkspace,
        transport: &MuseGlimmerPreparedF16Transport,
        source_rows: &[f32],
        top_k: usize,
        transported_rows: &[usize],
    ) -> Result<MuseGlimmerBatchedFullReadout, MuseGlimmerTextSessionError> {
        let hidden_size = self.weights.config.hidden_size as usize;
        let vocab_size = self.weights.config.vocab_size as usize;
        if workspace.hidden_size != hidden_size || workspace.vocab_size != vocab_size {
            return invalid("full-readout workspace does not match the loaded model");
        }
        let device_registry_id = self.ctx.device.registryID();
        if workspace.device_registry_id != device_registry_id {
            return invalid("full-readout workspace belongs to a different Metal device");
        }
        if transport.hidden_size != hidden_size
            || transport.device_registry_id != device_registry_id
        {
            return invalid("prepared F16 transport does not match the loaded model");
        }
        if top_k == 0 || top_k > MUSE_GLIMMER_FULL_READOUT_MAX_TOP_K {
            return invalid(format!(
                "full-readout top-k must be in 1..={}, got {top_k}",
                MUSE_GLIMMER_FULL_READOUT_MAX_TOP_K
            ));
        }
        if top_k > vocab_size {
            return invalid(format!(
                "full-readout top-k {top_k} exceeds vocabulary size {vocab_size}"
            ));
        }
        if source_rows.is_empty() || !source_rows.len().is_multiple_of(hidden_size) {
            return invalid(format!(
                "full-readout source length {} is not a positive multiple of hidden size {hidden_size}",
                source_rows.len()
            ));
        }
        if let Some(index) = source_rows.iter().position(|value| !value.is_finite()) {
            return invalid(format!(
                "full-readout source has non-finite value at index {index}"
            ));
        }
        let row_count = source_rows.len() / hidden_size;
        if row_count > workspace.row_capacity {
            return invalid(format!(
                "full-readout row count {row_count} exceeds workspace capacity {}",
                workspace.row_capacity
            ));
        }
        for (slot, &row) in transported_rows.iter().enumerate() {
            if row >= row_count {
                return invalid(format!(
                    "transported row request {row} at slot {slot} is outside {row_count} rows"
                ));
            }
            if transported_rows[..slot].contains(&row) {
                return invalid(format!("transported row request {row} is duplicated"));
            }
        }
        write_f32_prefix(&workspace.source, source_rows)?;

        let hidden_elements = row_count * hidden_size;
        let logits_elements = row_count * vocab_size;
        let source = workspace
            .source
            .view_subrange(0, vec![row_count as u64, hidden_size as u64]);
        let transported = workspace
            .transported
            .view_subrange(0, vec![row_count as u64, hidden_size as u64]);
        let normalized = workspace
            .normalized
            .view_subrange(0, vec![row_count as u64, hidden_size as u64]);
        let logits = workspace
            .logits
            .view_subrange(0, vec![row_count as u64, vocab_size as u64]);
        let first_ids = workspace.first_ids.view_subrange(
            0,
            vec![row_count as u64, MUSE_GLIMMER_FULL_READOUT_PASS_K as u64],
        );
        let first_values = workspace.first_values.view_subrange(
            0,
            vec![row_count as u64, MUSE_GLIMMER_FULL_READOUT_PASS_K as u64],
        );
        let second_ids = workspace.second_ids.view_subrange(
            0,
            vec![row_count as u64, MUSE_GLIMMER_FULL_READOUT_PASS_K as u64],
        );
        let second_values = workspace.second_values.view_subrange(
            0,
            vec![row_count as u64, MUSE_GLIMMER_FULL_READOUT_PASS_K as u64],
        );
        debug_assert_eq!(source.n_elements() as usize, hidden_elements);
        debug_assert_eq!(logits.n_elements() as usize, logits_elements);

        let transport_started = std::time::Instant::now();
        let transport_command = self.ctx.queue.commandBuffer().ok_or_else(|| {
            MuseGlimmerTextSessionError::CommandBuffer("allocation failed".into())
        })?;
        let transport_encoder = KernelEncoder::begin(&transport_command);
        let transport_encode_result = (|| {
            for row in 0..row_count {
                let source_row = packed_row(&source, row, hidden_size);
                let transported_row = packed_row(&transported, row, hidden_size);
                encode_mat_vec_f16_f32(
                    self.ctx,
                    &transport_encoder,
                    &transport.tensor,
                    &source_row,
                    &transported_row,
                    hidden_size,
                    hidden_size,
                )?;
            }
            Ok::<(), MuseGlimmerTextSessionError>(())
        })();
        transport_encoder.end();
        transport_encode_result?;
        transport_command.commit();
        transport_command.waitUntilCompleted();
        let transport_status = transport_command.status();
        let transport_error = transport_command.error().map(|error| error.to_string());
        if transport_status != MTLCommandBufferStatus::Completed || transport_error.is_some() {
            return Err(MuseGlimmerTextSessionError::CommandBuffer(format!(
                "transport status={transport_status:?}, error={transport_error:?}"
            )));
        }
        let transport_gpu_ms =
            (transport_command.GPUEndTime() - transport_command.GPUStartTime()) * 1e3;
        let mut selected_transported = Vec::with_capacity(transported_rows.len());
        for &row in transported_rows {
            let values = read_f32(&packed_row(&transported, row, hidden_size));
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return invalid(format!(
                    "transported row {row} has non-finite value at component {index}"
                ));
            }
            selected_transported.push(MuseGlimmerTransportedRow { row, values });
        }
        let transport_wall_ms = transport_started.elapsed().as_secs_f64() * 1e3;

        let output_tail_started = std::time::Instant::now();
        let output_tail_command = self.ctx.queue.commandBuffer().ok_or_else(|| {
            MuseGlimmerTextSessionError::CommandBuffer("allocation failed".into())
        })?;
        let output_tail_encoder = KernelEncoder::begin(&output_tail_command);
        let output_tail_encode_result = (|| {
            for row in 0..row_count {
                let transported_row = packed_row(&transported, row, hidden_size);
                let normalized_row = packed_row(&normalized, row, hidden_size);
                encode_rms_norm_mul_f32(
                    self.ctx,
                    &output_tail_encoder,
                    &transported_row,
                    self.weights.output_norm,
                    &normalized_row,
                    self.weights.config.rms_epsilon,
                )?;
            }
            match self.full_readout_head_mode() {
                Some(MuseGlimmerFullReadoutHeadMode::Q8Batch) => {
                    encode_mat_vec_q8_0_batch_f32(
                        self.ctx,
                        &output_tail_encoder,
                        self.weights.output,
                        &normalized,
                        &logits,
                        hidden_size,
                        vocab_size,
                        row_count,
                    )?;
                }
                Some(MuseGlimmerFullReadoutHeadMode::Bf16ScalarRows) => {
                    for row in 0..row_count {
                        encode_mat_vec_dispatch(
                            self.ctx,
                            &output_tail_encoder,
                            self.weights.output,
                            &packed_row(&normalized, row, hidden_size),
                            &packed_row(&logits, row, vocab_size),
                            hidden_size,
                            vocab_size,
                        )?;
                    }
                }
                None => return invalid("command-batched full readout is unavailable"),
            }
            encode_muse_glimmer_logit_softcap_f32(
                self.ctx,
                &output_tail_encoder,
                &logits,
                &logits,
                self.weights.config.logit_scale,
                self.weights.config.final_logit_softcap,
            )?;
            encode_topk16_f32(
                self.ctx,
                &output_tail_encoder,
                &logits,
                &first_ids,
                &first_values,
                row_count,
                vocab_size,
            )?;
            if top_k > MUSE_GLIMMER_FULL_READOUT_PASS_K {
                encode_mask_row_indices_f32(
                    self.ctx,
                    &output_tail_encoder,
                    &logits,
                    &first_ids,
                    row_count,
                    vocab_size,
                    MUSE_GLIMMER_FULL_READOUT_PASS_K,
                )?;
                encode_topk16_f32(
                    self.ctx,
                    &output_tail_encoder,
                    &logits,
                    &second_ids,
                    &second_values,
                    row_count,
                    vocab_size,
                )?;
            }
            Ok::<(), MuseGlimmerTextSessionError>(())
        })();
        output_tail_encoder.end();
        output_tail_encode_result?;
        output_tail_command.commit();
        output_tail_command.waitUntilCompleted();
        let output_tail_status = output_tail_command.status();
        let output_tail_error = output_tail_command.error().map(|error| error.to_string());
        if output_tail_status != MTLCommandBufferStatus::Completed || output_tail_error.is_some() {
            return Err(MuseGlimmerTextSessionError::CommandBuffer(format!(
                "output-tail status={output_tail_status:?}, error={output_tail_error:?}"
            )));
        }
        let output_tail_gpu_ms =
            (output_tail_command.GPUEndTime() - output_tail_command.GPUStartTime()) * 1e3;
        let first_ids = read_i32(&first_ids);
        let first_values = read_f32(&first_values);
        let (second_ids, second_values) = if top_k > MUSE_GLIMMER_FULL_READOUT_PASS_K {
            (read_i32(&second_ids), read_f32(&second_values))
        } else {
            (Vec::new(), Vec::new())
        };
        let rows = build_full_readout_rows(
            row_count,
            vocab_size,
            top_k,
            &first_ids,
            &first_values,
            &second_ids,
            &second_values,
        )?;
        let output_tail_wall_ms = output_tail_started.elapsed().as_secs_f64() * 1e3;
        Ok(MuseGlimmerBatchedFullReadout {
            row_count,
            top_k,
            gpu_ms: transport_gpu_ms + output_tail_gpu_ms,
            command_wall_ms: transport_wall_ms + output_tail_wall_ms,
            transport_gpu_ms,
            transport_wall_ms,
            output_tail_gpu_ms,
            output_tail_wall_ms,
            rows,
            transported_rows: selected_transported,
        })
    }

    pub(crate) fn apply_prepared_f16_post_block_transport(
        &self,
        transport: &MuseGlimmerPreparedF16Transport,
        source_residual: &[f32],
    ) -> Result<Vec<f32>, MuseGlimmerTextSessionError> {
        let hidden = self.weights.config.hidden_size as usize;
        if transport.hidden_size != hidden
            || transport.device_registry_id != self.ctx.device.registryID()
        {
            return invalid(format!(
                "prepared F16 transport hidden size/device {}/{} != model {hidden}/{}",
                transport.hidden_size,
                transport.device_registry_id,
                self.ctx.device.registryID()
            ));
        }
        validate_deployed_output_residual(source_residual, hidden)?;
        let source = MetalTensor::from_bytes(
            self.ctx,
            bytemuck::cast_slice(source_residual),
            vec![hidden as u64],
            GgmlType::F32,
        )?;
        let transported = MetalTensor::zeros_f32(self.ctx, vec![hidden as u64])?;
        let command = self.ctx.queue.commandBuffer().ok_or_else(|| {
            MuseGlimmerTextSessionError::CommandBuffer("allocation failed".into())
        })?;
        let encoder = KernelEncoder::begin(&command);
        let encode_result = encode_mat_vec_f16_f32(
            self.ctx,
            &encoder,
            &transport.tensor,
            &source,
            &transported,
            hidden,
            hidden,
        );
        encoder.end();
        encode_result?;
        command.commit();
        command.waitUntilCompleted();
        let status = command.status();
        let command_error = command.error().map(|error| error.to_string());
        if status != MTLCommandBufferStatus::Completed || command_error.is_some() {
            return Err(MuseGlimmerTextSessionError::CommandBuffer(format!(
                "status={status:?}, error={command_error:?}"
            )));
        }
        let values = read_f32(&transported);
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return invalid(format!(
                "F16 post-block transport produced a non-finite value at index {index}"
            ));
        }
        Ok(values)
    }

    pub fn forward_token_capture_post_blocks(
        &self,
        token: u32,
        layer_ids: &[u32],
        session: &mut MuseGlimmerTextSession,
    ) -> Result<MuseGlimmerPostBlockForward, MuseGlimmerTextSessionError> {
        self.forward_token_capture_post_blocks_with_interventions(token, layer_ids, &[], session)
    }

    pub fn forward_token_capture_post_blocks_with_interventions(
        &self,
        token: u32,
        layer_ids: &[u32],
        interventions: &[PostBlockIntervention<'_>],
        session: &mut MuseGlimmerTextSession,
    ) -> Result<MuseGlimmerPostBlockForward, MuseGlimmerTextSessionError> {
        self.validate_token_and_session(token, session)?;
        validate_live_capture_layers(layer_ids, self.weights.layers.len())?;
        let position = session.next_position;
        let hidden_size = session.geometry.hidden_size;
        let captured =
            MetalTensor::zeros_f32(self.ctx, vec![hidden_size as u64, layer_ids.len() as u64])?;
        let destination = MuseGlimmerPostBlockCaptureDestination {
            layer_ids,
            captured: &captured,
        };
        let logits = self
            .execute_token_with_sink(
                token,
                session,
                true,
                Some(MuseGlimmerProductionCaptureSink::PostBlock(&destination)),
                interventions,
            )?
            .ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid(
                    "capture forward did not produce logits".into(),
                )
            })?;
        Ok(MuseGlimmerPostBlockForward {
            position,
            token_id: token,
            layer_ids: layer_ids.to_vec(),
            hidden_size,
            logits,
            post_block_residuals: read_f32(&captured),
        })
    }

    pub fn prefill(
        &self,
        tokens: &[u32],
        session: &mut MuseGlimmerTextSession,
    ) -> Result<Vec<f32>, MuseGlimmerTextSessionError> {
        self.prefill_with_command_checkpoint(tokens, session, || Ok(()))
    }

    pub fn prefill_with_command_checkpoint<F>(
        &self,
        tokens: &[u32],
        session: &mut MuseGlimmerTextSession,
        mut checkpoint: F,
    ) -> Result<Vec<f32>, MuseGlimmerTextSessionError>
    where
        F: FnMut() -> Result<(), MuseGlimmerTextSessionError>,
    {
        session.ensure_usable()?;
        if tokens.is_empty() {
            return invalid("prefill requires at least one token");
        }
        if tokens.len() > session.remaining_forwards() {
            return invalid(format!(
                "prefill of {} tokens exceeds {} remaining session positions",
                tokens.len(),
                session.remaining_forwards()
            ));
        }
        for &token in tokens {
            self.validate_token_and_session(token, session)?;
        }
        let plan = MuseGlimmerPrefillPlan::for_tokens(tokens.len());
        let (packed_tokens, scalar_tokens) = tokens.split_at(plan.packed_tokens());
        let mut logits = None;
        let mut chunk_start = 0;
        let mut chunk_index = 0;
        while chunk_start < packed_tokens.len() {
            let chunk_end = chunk_start
                .checked_add(MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS)
                .unwrap_or(usize::MAX)
                .min(packed_tokens.len());
            let chunk = &packed_tokens[chunk_start..chunk_end];
            checkpoint()?;
            let produce_logits = plan.scalar_tail == 0 && chunk_index + 1 == plan.packed_chunks;
            logits = self.execute_packed_chunk(chunk, session, produce_logits)?;
            chunk_start = chunk_end;
            chunk_index += 1;
        }
        debug_assert_eq!(chunk_index, plan.packed_chunks);
        for (index, &token) in scalar_tokens.iter().enumerate() {
            checkpoint()?;
            logits = self.execute_token(token, session, index + 1 == scalar_tokens.len())?;
        }
        logits.ok_or_else(|| {
            MuseGlimmerTextSessionError::Invalid("prefill endpoint logits were not produced".into())
        })
    }

    pub(crate) fn capture_fresh_lens_prompt(
        &self,
        tokens: &[u32],
        target_block: u32,
        session: &mut MuseGlimmerTextSession,
    ) -> Result<MuseGlimmerLensCapture, MuseGlimmerTextSessionError> {
        self.capture_fresh_lens_prompt_blocks(tokens, &[target_block], session)?
            .block_capture(target_block)
            .ok_or_else(|| {
                MuseGlimmerTextSessionError::Invalid(
                    "single-block capture was absent from capture bank".into(),
                )
            })
    }

    pub(crate) fn capture_fresh_lens_prompt_blocks(
        &self,
        tokens: &[u32],
        target_blocks: &[u32],
        session: &mut MuseGlimmerTextSession,
    ) -> Result<MuseGlimmerLensCaptureBank, MuseGlimmerTextSessionError> {
        session.ensure_usable()?;
        validate_multi_lens_capture_request(
            tokens.len(),
            target_blocks,
            self.weights.layers.len(),
            session.next_position,
            session.geometry.capacity,
        )?;
        for &token in tokens {
            self.validate_token_and_session(token, session)?;
        }

        let hidden_size = session.geometry.hidden_size;
        let bank_shape = vec![
            hidden_size as u64,
            tokens.len() as u64,
            target_blocks.len() as u64,
        ];
        let input = MetalTensor::zeros_f32(self.ctx, bank_shape.clone())?;
        let post_attention = MetalTensor::zeros_f32(self.ctx, bank_shape.clone())?;
        let post_block = MetalTensor::zeros_f32(self.ctx, bank_shape)?;
        for (token_slot, &token) in tokens.iter().enumerate() {
            let destination = MuseGlimmerLensCaptureDestination {
                target_blocks,
                token_slot,
                n_tokens: tokens.len(),
                input: &input,
                post_attention: &post_attention,
                post_block: &post_block,
            };
            self.execute_token_with_capture(token, session, &destination)?;
        }

        Ok(MuseGlimmerLensCaptureBank::new(
            target_blocks.to_vec(),
            tokens.to_vec(),
            hidden_size,
            read_f32(&input),
            read_f32(&post_attention),
            read_f32(&post_block),
        ))
    }

    pub(crate) fn lens_one_full_attention_block_vjp(
        &self,
        capture: &MuseGlimmerLensCapture,
        target_cotangent: &[f32],
        rule: crate::muse_glimmer_lens::MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerOneBlockVjp, MuseGlimmerLensError> {
        muse_glimmer_one_full_attention_block_vjp(
            self.ctx,
            &self.weights,
            capture,
            target_cotangent,
            rule,
        )
    }

    pub(crate) fn lens_one_attention_block_vjp_query_batch(
        &self,
        capture: &MuseGlimmerLensCapture,
        target_cotangents: &[f32],
        query_count: usize,
        rule: crate::muse_glimmer_lens::MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerQueryBatchOneBlockVjp, MuseGlimmerLensError> {
        muse_glimmer_one_attention_block_vjp_query_batch(
            self.ctx,
            &self.weights,
            capture,
            target_cotangents,
            query_count,
            rule,
        )
    }

    pub(crate) fn lens_one_full_attention_block_vjp_query_batch(
        &self,
        capture: &MuseGlimmerLensCapture,
        target_cotangents: &[f32],
        query_count: usize,
        rule: crate::muse_glimmer_lens::MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerQueryBatchOneBlockVjp, MuseGlimmerLensError> {
        muse_glimmer_one_full_attention_block_vjp_query_batch(
            self.ctx,
            &self.weights,
            capture,
            target_cotangents,
            query_count,
            rule,
        )
    }

    pub(crate) fn lens_composed_vjp_query_batch(
        &self,
        captures: &MuseGlimmerLensCaptureBank,
        target_block: u32,
        source_layers: &[u32],
        target_cotangents: &[f32],
        query_count: usize,
        rule: crate::muse_glimmer_lens::MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerQueryBatchComposedVjp, MuseGlimmerLensError> {
        muse_glimmer_composed_vjp_query_batch(
            self.ctx,
            &self.weights,
            captures,
            target_block,
            source_layers,
            target_cotangents,
            query_count,
            rule,
        )
    }

    pub(crate) fn fit_adjacent_full_attention_selected_tokens(
        &self,
        capture: &MuseGlimmerLensCapture,
        covectors: &MuseGlimmerSelectedTokenCovectors,
        skip_first: usize,
        rule: crate::muse_glimmer_lens::MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerAdjacentSelectedTokenFit, MuseGlimmerLensError> {
        muse_glimmer_fit_adjacent_full_attention_selected_tokens(
            self.ctx,
            &self.weights,
            capture,
            covectors,
            skip_first,
            rule,
        )
    }

    pub(crate) fn fit_adjacent_full_attention_rows_batched(
        &self,
        capture: &MuseGlimmerLensCapture,
        rows: std::ops::Range<u32>,
        skip_first: usize,
        dim_batch: usize,
        rule: crate::muse_glimmer_lens::MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerAdjacentRowSlab, MuseGlimmerLensError> {
        muse_glimmer_fit_adjacent_full_attention_rows_batched(
            self.ctx,
            &self.weights,
            capture,
            rows,
            skip_first,
            dim_batch,
            rule,
        )
    }

    pub(crate) fn fit_selected_tokens_to_sources(
        &self,
        captures: &MuseGlimmerLensCaptureBank,
        target_block: u32,
        source_layers: &[u32],
        covectors: &MuseGlimmerSelectedTokenCovectors,
        skip_first: usize,
        rule: crate::muse_glimmer_lens::MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerMultiSourceSelectedTokenFit, MuseGlimmerLensError> {
        muse_glimmer_fit_selected_tokens_to_sources(
            self.ctx,
            &self.weights,
            captures,
            target_block,
            source_layers,
            covectors,
            skip_first,
            rule,
        )
    }

    pub(crate) fn fit_full_transport_rows_to_sources(
        &self,
        captures: &MuseGlimmerLensCaptureBank,
        target_block: u32,
        source_layers: &[u32],
        output_row_ids: &[u32],
        skip_first: usize,
        rule: crate::muse_glimmer_lens::MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerFullTransportRowFit, MuseGlimmerLensError> {
        muse_glimmer_fit_full_transport_rows_to_sources(
            self.ctx,
            &self.weights,
            captures,
            target_block,
            source_layers,
            output_row_ids,
            skip_first,
            rule,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fit_full_transport_rows_to_sources_batched(
        &self,
        captures: &MuseGlimmerLensCaptureBank,
        target_block: u32,
        source_layers: &[u32],
        output_row_ids: &[u32],
        skip_first: usize,
        query_batch_size: usize,
        rule: crate::muse_glimmer_lens::MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerBatchedFullTransportRowFit, MuseGlimmerLensError> {
        muse_glimmer_fit_full_transport_rows_to_sources_batched(
            self.ctx,
            &self.weights,
            captures,
            target_block,
            source_layers,
            output_row_ids,
            skip_first,
            query_batch_size,
            rule,
        )
    }

    fn validate_token_and_session(
        &self,
        token: u32,
        session: &MuseGlimmerTextSession,
    ) -> Result<(), MuseGlimmerTextSessionError> {
        session.ensure_usable()?;
        self.validate_session_geometry(session)?;
        if token >= self.weights.config.vocab_size {
            return invalid(format!(
                "token {token} is outside vocabulary {}",
                self.weights.config.vocab_size
            ));
        }
        if session.next_position >= session.geometry.capacity {
            return invalid(format!(
                "session capacity {} is exhausted",
                session.geometry.capacity
            ));
        }
        Ok(())
    }

    fn validate_session_geometry(
        &self,
        session: &MuseGlimmerTextSession,
    ) -> Result<(), MuseGlimmerTextSessionError> {
        if session.geometry.hidden_size != self.weights.config.hidden_size as usize
            || session.geometry.feed_forward_size != self.weights.config.feed_forward_size as usize
            || session.geometry.vocab_size != self.weights.config.vocab_size as usize
            || session.geometry.layer_count != self.weights.layers.len()
        {
            return invalid("session geometry differs from resident model weights");
        }
        if session.device_registry_id != self.ctx.device.registryID() {
            return invalid(format!(
                "session belongs to Metal device registry {}, forward context is {}",
                session.device_registry_id,
                self.ctx.device.registryID()
            ));
        }
        Ok(())
    }

    fn execute_token(
        &self,
        token: u32,
        session: &mut MuseGlimmerTextSession,
        produce_logits: bool,
    ) -> Result<Option<Vec<f32>>, MuseGlimmerTextSessionError> {
        self.execute_token_with_sink(token, session, produce_logits, None, &[])
    }

    fn execute_packed_chunk(
        &self,
        tokens: &[u32],
        session: &mut MuseGlimmerTextSession,
        produce_logits: bool,
    ) -> Result<Option<Vec<f32>>, MuseGlimmerTextSessionError> {
        session.ensure_usable()?;
        self.validate_session_geometry(session)?;
        if tokens.len() < MUSE_GLIMMER_PACKED_PREFILL_QUANTUM
            || tokens.len() > MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS
            || !tokens
                .len()
                .is_multiple_of(MUSE_GLIMMER_PACKED_PREFILL_QUANTUM)
        {
            return invalid(format!(
                "packed prefill requires a multiple of {MUSE_GLIMMER_PACKED_PREFILL_QUANTUM} tokens in {}..={}, got {}",
                MUSE_GLIMMER_PACKED_PREFILL_QUANTUM,
                MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS,
                tokens.len()
            ));
        }
        if tokens.len() > session.remaining_forwards() {
            return invalid(format!(
                "packed prefill of {} tokens exceeds {} remaining session positions",
                tokens.len(),
                session.remaining_forwards()
            ));
        }
        for &token in tokens {
            if token >= self.weights.config.vocab_size {
                return invalid(format!(
                    "token {token} is outside vocabulary {}",
                    self.weights.config.vocab_size
                ));
            }
        }

        let start_position = session.next_position;
        let end_position = start_position.checked_add(tokens.len()).ok_or_else(|| {
            MuseGlimmerTextSessionError::Invalid("packed prefill position range overflow".into())
        })?;
        u32::try_from(end_position - 1).map_err(|_| {
            MuseGlimmerTextSessionError::Invalid("packed prefill position exceeds u32".into())
        })?;
        session.packed.write_tokens(tokens)?;

        let command = self.ctx.queue.commandBuffer().ok_or_else(|| {
            MuseGlimmerTextSessionError::CommandBuffer("allocation failed".into())
        })?;
        let encode_result = (|| {
            let encoder = KernelEncoder::begin(&command);
            self.encode_packed_chunk_graph(
                &encoder,
                start_position,
                tokens.len(),
                session,
                produce_logits,
            )?;
            encoder.end();
            Ok::<(), MuseGlimmerTextSessionError>(())
        })();
        encode_result?;

        command.commit();
        command.waitUntilCompleted();
        let status = command.status();
        let command_error = command.error().map(|error| error.to_string());
        if status != MTLCommandBufferStatus::Completed || command_error.is_some() {
            let reason = format!("status={status:?}, error={command_error:?}");
            session.poison_reason = Some(reason.clone());
            return Err(MuseGlimmerTextSessionError::CommandBuffer(reason));
        }
        session.next_position = end_position;
        Ok(produce_logits.then(|| session.read_logits()))
    }

    fn encode_packed_chunk_graph(
        &self,
        encoder: &KernelEncoder,
        start_position: usize,
        rows: usize,
        session: &MuseGlimmerTextSession,
        produce_logits: bool,
    ) -> Result<(), MuseGlimmerTextSessionError> {
        let geometry = &session.geometry;
        let packed = session.packed.views(geometry, rows)?;

        encode_get_rows_f32(
            self.ctx,
            encoder,
            self.weights.token_embedding,
            &packed.ids,
            &packed.normed,
            rows,
            geometry.hidden_size,
        )?;
        encode_rms_norm_mul_rows_f32(
            self.ctx,
            encoder,
            &packed.normed,
            &session.embedding_norm_weight,
            &packed.residual,
            rows,
            geometry.hidden_size,
            self.weights.config.rms_epsilon,
        )?;

        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            encode_rms_norm_mul_rows_f32(
                self.ctx,
                encoder,
                &packed.residual,
                layer.attention_norm,
                &packed.normed,
                rows,
                geometry.hidden_size,
                self.weights.config.rms_epsilon,
            )?;
            self.encode_packed_projection(
                encoder,
                layer.attention_query,
                &packed.normed,
                &packed.query_raw,
                geometry.hidden_size,
                geometry.query_width,
                rows,
            )?;
            self.encode_packed_projection(
                encoder,
                layer.attention_key,
                &packed.normed,
                &packed.key_raw,
                geometry.hidden_size,
                geometry.kv_width,
                rows,
            )?;
            self.encode_packed_projection(
                encoder,
                layer.attention_value,
                &packed.normed,
                &packed.value,
                geometry.hidden_size,
                geometry.kv_width,
                rows,
            )?;
            self.encode_packed_projection(
                encoder,
                layer.attention_gate,
                &packed.normed,
                &packed.attention_gate,
                geometry.hidden_size,
                geometry.query_width,
                rows,
            )?;
            encode_rms_norm_batched_f32(
                self.ctx,
                encoder,
                &packed.query_raw,
                layer.query_norm,
                &packed.query,
                rows * geometry.query_head_count,
                geometry.head_dim,
                self.weights.config.rms_epsilon,
            )?;
            encode_rms_norm_batched_f32(
                self.ctx,
                encoder,
                &packed.key_raw,
                layer.key_norm,
                &packed.key,
                rows * geometry.kv_head_count,
                geometry.head_dim,
                self.weights.config.rms_epsilon,
            )?;
            if layer.sliding_attention {
                encode_muse_glimmer_rope_adjacent_pair_rows_in_place_f32(
                    self.ctx,
                    encoder,
                    &packed.query,
                    &packed.key,
                    geometry.query_head_count,
                    geometry.kv_head_count,
                    geometry.head_dim,
                    rows,
                    start_position,
                    self.weights.config.rope_theta,
                )?;
            }

            let cache_write = geometry.cache_write_offset(layer_index, start_position)?;
            encode_scatter_offset_f32_to_f16_kv(
                self.ctx,
                encoder,
                &packed.key,
                &packed.value,
                &session.key_cache,
                &session.value_cache,
                cache_write,
                rows * geometry.kv_width,
            )?;
            let end_position = start_position + rows;
            let maximum_visible = if layer.sliding_attention {
                end_position.min(geometry.sliding_window)
            } else {
                end_position
            };
            if maximum_visible <= MUSE_GLIMMER_MATERIALIZED_ATTENTION_MAX_POSITIONS {
                let (key_cache, value_cache) =
                    session.cache_prefix_views(layer_index, end_position)?;
                encode_muse_glimmer_attn_prefill_f16kv_f32(
                    self.ctx,
                    encoder,
                    &packed.query,
                    &key_cache,
                    &value_cache,
                    &packed.attention_output,
                    rows,
                    start_position,
                    geometry.query_head_count,
                    geometry.kv_head_count,
                    geometry.head_dim,
                    layer.sliding_attention.then_some(geometry.sliding_window),
                )?;
            } else {
                for row in 0..rows {
                    let position = start_position + row;
                    let query = packed_row(&packed.query, row, geometry.query_width);
                    let attention_output =
                        packed_row(&packed.attention_output, row, geometry.query_width);
                    let (key_cache, value_cache, visible_positions) =
                        session.cache_views(layer_index, position, layer.sliding_attention)?;
                    encode_muse_glimmer_attn_decode_f16kv_f32(
                        self.ctx,
                        encoder,
                        &query,
                        &key_cache,
                        &value_cache,
                        &attention_output,
                        geometry.query_head_count,
                        geometry.kv_head_count,
                        geometry.head_dim,
                        visible_positions,
                    )?;
                }
            }
            encode_sigmoid_mul_f32(
                self.ctx,
                encoder,
                &packed.attention_gate,
                &packed.attention_output,
                &packed.attention_output,
            )?;
            self.encode_packed_projection(
                encoder,
                layer.attention_output,
                &packed.attention_output,
                &packed.branch_raw,
                geometry.query_width,
                geometry.hidden_size,
                rows,
            )?;
            encode_rms_norm_mul_rows_f32(
                self.ctx,
                encoder,
                &packed.branch_raw,
                layer.post_attention_norm,
                &packed.branch_normed,
                rows,
                geometry.hidden_size,
                self.weights.config.post_norm_epsilon,
            )?;
            encode_add_inplace_f32(self.ctx, encoder, &packed.residual, &packed.branch_normed)?;
            encode_rms_norm_mul_rows_f32(
                self.ctx,
                encoder,
                &packed.residual,
                layer.feed_forward_norm,
                &packed.normed,
                rows,
                geometry.hidden_size,
                self.weights.config.rms_epsilon,
            )?;
            self.encode_packed_projection(
                encoder,
                layer.feed_forward_gate,
                &packed.normed,
                &packed.feed_forward_gate,
                geometry.hidden_size,
                geometry.feed_forward_size,
                rows,
            )?;
            self.encode_packed_projection(
                encoder,
                layer.feed_forward_up,
                &packed.normed,
                &packed.feed_forward_up,
                geometry.hidden_size,
                geometry.feed_forward_size,
                rows,
            )?;
            encode_silu_mul_f32(
                self.ctx,
                encoder,
                &packed.feed_forward_gate,
                &packed.feed_forward_up,
                &packed.feed_forward_gate,
            )?;
            self.encode_packed_projection(
                encoder,
                layer.feed_forward_down,
                &packed.feed_forward_gate,
                &packed.branch_raw,
                geometry.feed_forward_size,
                geometry.hidden_size,
                rows,
            )?;
            encode_rms_norm_mul_rows_f32(
                self.ctx,
                encoder,
                &packed.branch_raw,
                layer.post_feed_forward_norm,
                &packed.branch_normed,
                rows,
                geometry.hidden_size,
                self.weights.config.post_norm_epsilon,
            )?;
            encode_add_inplace_f32(self.ctx, encoder, &packed.residual, &packed.branch_normed)?;
        }

        if produce_logits {
            encode_copy_offset_f32(
                self.ctx,
                encoder,
                &packed.residual,
                (rows - 1) * geometry.hidden_size,
                &session.residual,
                geometry.hidden_size,
            )?;
            self.encode_deployed_output_tail(encoder, session)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_packed_projection(
        &self,
        encoder: &KernelEncoder,
        weight: &MetalTensor,
        input: &MetalTensor,
        output: &MetalTensor,
        n_in: usize,
        n_out: usize,
        rows: usize,
    ) -> Result<(), MuseGlimmerTextSessionError> {
        if weight.dtype == GgmlType::Q8_0 {
            #[cfg(test)]
            if self.packed_q8_mat_mat {
                encode_mat_mat_q8_0_f32(
                    self.ctx, encoder, weight, input, output, n_in, n_out, rows,
                )?;
                return Ok(());
            }
            encode_mat_vec_q8_0_batch_f32(
                self.ctx, encoder, weight, input, output, n_in, n_out, rows,
            )?;
            return Ok(());
        }
        for row in 0..rows {
            let input = packed_row(input, row, n_in);
            let output = packed_row(output, row, n_out);
            encode_mat_vec_dispatch(self.ctx, encoder, weight, &input, &output, n_in, n_out)?;
        }
        Ok(())
    }

    fn execute_token_with_sink(
        &self,
        token: u32,
        session: &mut MuseGlimmerTextSession,
        produce_logits: bool,
        capture: Option<MuseGlimmerProductionCaptureSink<'_>>,
        interventions: &[PostBlockIntervention<'_>],
    ) -> Result<Option<Vec<f32>>, MuseGlimmerTextSessionError> {
        validate_post_block_interventions(
            session,
            interventions,
            self.weights.layers.len(),
            session.geometry.hidden_size,
        )?;
        let position = session.next_position;
        let position_u32 = u32::try_from(position).map_err(|_| {
            MuseGlimmerTextSessionError::Invalid("session position exceeds u32".into())
        })?;
        session.write_token(token as i32);
        let command = self.ctx.queue.commandBuffer().ok_or_else(|| {
            MuseGlimmerTextSessionError::CommandBuffer("allocation failed".into())
        })?;

        let encode_result = (|| {
            let encoder = KernelEncoder::begin(&command);
            self.encode_token_graph_inner(
                &encoder,
                position,
                position_u32,
                session,
                produce_logits,
                capture,
                interventions,
            )?;
            encoder.end();
            Ok::<(), MuseGlimmerTextSessionError>(())
        })();
        if let Err(error) = encode_result {
            return Err(error);
        }

        command.commit();
        command.waitUntilCompleted();
        let status = command.status();
        let command_error = command.error().map(|error| error.to_string());
        if status != MTLCommandBufferStatus::Completed || command_error.is_some() {
            let reason = format!("status={status:?}, error={command_error:?}");
            session.poison_reason = Some(reason.clone());
            return Err(MuseGlimmerTextSessionError::CommandBuffer(reason));
        }
        session.next_position += 1;
        Ok(produce_logits.then(|| session.read_logits()))
    }

    fn execute_token_with_capture(
        &self,
        token: u32,
        session: &mut MuseGlimmerTextSession,
        capture: &MuseGlimmerLensCaptureDestination,
    ) -> Result<(), MuseGlimmerTextSessionError> {
        self.execute_token_with_sink(
            token,
            session,
            false,
            Some(MuseGlimmerProductionCaptureSink::Lens(capture)),
            &[],
        )?;
        Ok(())
    }

    fn encode_token_graph_inner(
        &self,
        encoder: &KernelEncoder,
        position: usize,
        position_u32: u32,
        session: &MuseGlimmerTextSession,
        produce_logits: bool,
        capture: Option<MuseGlimmerProductionCaptureSink<'_>>,
        interventions: &[PostBlockIntervention<'_>],
    ) -> Result<(), MuseGlimmerTextSessionError> {
        let geometry = &session.geometry;
        encode_get_rows_f32(
            self.ctx,
            encoder,
            self.weights.token_embedding,
            &session.ids,
            &session.normed,
            1,
            geometry.hidden_size,
        )?;
        encode_rms_norm_mul_f32(
            self.ctx,
            encoder,
            &session.normed,
            &session.embedding_norm_weight,
            &session.residual,
            self.weights.config.rms_epsilon,
        )?;

        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            if let Some(MuseGlimmerProductionCaptureSink::Lens(capture)) = capture
                && let Ok(block_slot) = capture.target_blocks.binary_search(&(layer_index as u32))
            {
                encode_copy_offset_f32(
                    self.ctx,
                    encoder,
                    &session.residual,
                    0,
                    &capture.block_token_view(capture.input, block_slot, geometry.hidden_size),
                    geometry.hidden_size,
                )?;
            }
            encode_rms_norm_mul_f32(
                self.ctx,
                encoder,
                &session.residual,
                layer.attention_norm,
                &session.normed,
                self.weights.config.rms_epsilon,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                encoder,
                layer.attention_query,
                &session.normed,
                &session.query_raw,
                geometry.hidden_size,
                geometry.query_width,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                encoder,
                layer.attention_key,
                &session.normed,
                &session.key_raw,
                geometry.hidden_size,
                geometry.kv_width,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                encoder,
                layer.attention_value,
                &session.normed,
                &session.value,
                geometry.hidden_size,
                geometry.kv_width,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                encoder,
                layer.attention_gate,
                &session.normed,
                &session.attention_gate,
                geometry.hidden_size,
                geometry.query_width,
            )?;
            encode_rms_norm_batched_f32(
                self.ctx,
                encoder,
                &session.query_raw,
                layer.query_norm,
                &session.query,
                geometry.query_head_count,
                geometry.head_dim,
                self.weights.config.rms_epsilon,
            )?;
            encode_rms_norm_batched_f32(
                self.ctx,
                encoder,
                &session.key_raw,
                layer.key_norm,
                &session.key,
                geometry.kv_head_count,
                geometry.head_dim,
                self.weights.config.rms_epsilon,
            )?;
            if layer.sliding_attention {
                encode_muse_glimmer_rope_adjacent_pair_in_place_f32(
                    self.ctx,
                    encoder,
                    &session.query,
                    &session.key,
                    geometry.query_head_count,
                    geometry.kv_head_count,
                    geometry.head_dim,
                    position_u32,
                    self.weights.config.rope_theta,
                )?;
            }
            let cache_write = geometry.cache_write_offset(layer_index, position)?;
            encode_scatter_offset_f32_to_f16_kv(
                self.ctx,
                encoder,
                &session.key,
                &session.value,
                &session.key_cache,
                &session.value_cache,
                cache_write,
                geometry.kv_width,
            )?;
            let (key_cache, value_cache, visible_positions) =
                session.cache_views(layer_index, position, layer.sliding_attention)?;
            encode_muse_glimmer_attn_decode_f16kv_f32(
                self.ctx,
                encoder,
                &session.query,
                &key_cache,
                &value_cache,
                &session.attention_output,
                geometry.query_head_count,
                geometry.kv_head_count,
                geometry.head_dim,
                visible_positions,
            )?;
            encode_sigmoid_mul_f32(
                self.ctx,
                encoder,
                &session.attention_gate,
                &session.attention_output,
                &session.attention_output,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                encoder,
                layer.attention_output,
                &session.attention_output,
                &session.branch_raw,
                geometry.query_width,
                geometry.hidden_size,
            )?;
            encode_rms_norm_mul_f32(
                self.ctx,
                encoder,
                &session.branch_raw,
                layer.post_attention_norm,
                &session.branch_normed,
                self.weights.config.post_norm_epsilon,
            )?;
            encode_add_inplace_f32(self.ctx, encoder, &session.residual, &session.branch_normed)?;
            if let Some(MuseGlimmerProductionCaptureSink::Lens(capture)) = capture
                && let Ok(block_slot) = capture.target_blocks.binary_search(&(layer_index as u32))
            {
                encode_copy_offset_f32(
                    self.ctx,
                    encoder,
                    &session.residual,
                    0,
                    &capture.block_token_view(
                        capture.post_attention,
                        block_slot,
                        geometry.hidden_size,
                    ),
                    geometry.hidden_size,
                )?;
            }
            encode_rms_norm_mul_f32(
                self.ctx,
                encoder,
                &session.residual,
                layer.feed_forward_norm,
                &session.normed,
                self.weights.config.rms_epsilon,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                encoder,
                layer.feed_forward_gate,
                &session.normed,
                &session.feed_forward_gate,
                geometry.hidden_size,
                geometry.feed_forward_size,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                encoder,
                layer.feed_forward_up,
                &session.normed,
                &session.feed_forward_up,
                geometry.hidden_size,
                geometry.feed_forward_size,
            )?;
            encode_silu_mul_f32(
                self.ctx,
                encoder,
                &session.feed_forward_gate,
                &session.feed_forward_up,
                &session.feed_forward_gate,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                encoder,
                layer.feed_forward_down,
                &session.feed_forward_gate,
                &session.branch_raw,
                geometry.feed_forward_size,
                geometry.hidden_size,
            )?;
            encode_rms_norm_mul_f32(
                self.ctx,
                encoder,
                &session.branch_raw,
                layer.post_feed_forward_norm,
                &session.branch_normed,
                self.weights.config.post_norm_epsilon,
            )?;
            encode_add_inplace_f32(self.ctx, encoder, &session.residual, &session.branch_normed)?;
            for intervention in interventions
                .iter()
                .filter(|intervention| intervention_layer(intervention) as usize == layer_index)
            {
                encode_post_block_intervention_f32(
                    self.ctx,
                    encoder,
                    &session.residual,
                    intervention,
                )?;
            }
            if let Some(MuseGlimmerProductionCaptureSink::Lens(capture)) = capture
                && let Ok(block_slot) = capture.target_blocks.binary_search(&(layer_index as u32))
            {
                encode_copy_offset_f32(
                    self.ctx,
                    encoder,
                    &session.residual,
                    0,
                    &capture.block_token_view(capture.post_block, block_slot, geometry.hidden_size),
                    geometry.hidden_size,
                )?;
            }
            if let Some(MuseGlimmerProductionCaptureSink::PostBlock(capture)) = capture
                && let Ok(slot) = capture.layer_ids.binary_search(&(layer_index as u32))
            {
                encode_copy_offset_f32(
                    self.ctx,
                    encoder,
                    &session.residual,
                    0,
                    &capture.captured.view_subrange(
                        (slot * geometry.hidden_size) as u64,
                        vec![geometry.hidden_size as u64],
                    ),
                    geometry.hidden_size,
                )?;
            }
        }

        if produce_logits {
            self.encode_deployed_output_tail(encoder, session)?;
        }
        Ok(())
    }

    fn encode_deployed_output_tail(
        &self,
        encoder: &KernelEncoder,
        session: &MuseGlimmerTextSession,
    ) -> Result<(), MuseGlimmerTextSessionError> {
        encode_rms_norm_mul_f32(
            self.ctx,
            encoder,
            &session.residual,
            self.weights.output_norm,
            &session.normed,
            self.weights.config.rms_epsilon,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            encoder,
            self.weights.output,
            &session.normed,
            &session.logits,
            session.geometry.hidden_size,
            session.geometry.vocab_size,
        )?;
        encode_muse_glimmer_logit_softcap_f32(
            self.ctx,
            encoder,
            &session.logits,
            &session.logits,
            self.weights.config.logit_scale,
            self.weights.config.final_logit_softcap,
        )?;
        Ok(())
    }
}

struct MuseGlimmerLensCaptureDestination<'a> {
    target_blocks: &'a [u32],
    token_slot: usize,
    n_tokens: usize,
    input: &'a MetalTensor,
    post_attention: &'a MetalTensor,
    post_block: &'a MetalTensor,
}

impl MuseGlimmerLensCaptureDestination<'_> {
    fn block_token_view(
        &self,
        bank: &MetalTensor,
        block_slot: usize,
        hidden_size: usize,
    ) -> MetalTensor {
        let offset = (block_slot * self.n_tokens + self.token_slot) * hidden_size;
        bank.view_subrange(offset as u64, vec![hidden_size as u64])
    }
}

struct MuseGlimmerPostBlockCaptureDestination<'a> {
    layer_ids: &'a [u32],
    captured: &'a MetalTensor,
}

#[derive(Clone, Copy)]
enum MuseGlimmerProductionCaptureSink<'a> {
    Lens(&'a MuseGlimmerLensCaptureDestination<'a>),
    PostBlock(&'a MuseGlimmerPostBlockCaptureDestination<'a>),
}

fn intervention_layer(intervention: &PostBlockIntervention<'_>) -> u32 {
    match intervention {
        PostBlockIntervention::Fixed { layer, .. }
        | PostBlockIntervention::ResidualL2Relative { layer, .. }
        | PostBlockIntervention::Projection { layer, .. }
        | PostBlockIntervention::SourceToTarget { layer, .. } => *layer,
    }
}

fn validate_post_block_interventions(
    session: &MuseGlimmerTextSession,
    interventions: &[PostBlockIntervention<'_>],
    layer_count: usize,
    hidden_size: usize,
) -> Result<(), MuseGlimmerTextSessionError> {
    for (index, intervention) in interventions.iter().enumerate() {
        let (layer, coefficient) = match intervention {
            PostBlockIntervention::Fixed {
                layer, coefficient, ..
            }
            | PostBlockIntervention::ResidualL2Relative {
                layer, coefficient, ..
            }
            | PostBlockIntervention::Projection {
                layer, coefficient, ..
            }
            | PostBlockIntervention::SourceToTarget {
                layer, coefficient, ..
            } => (*layer, *coefficient),
        };
        if layer as usize >= layer_count {
            return invalid(format!(
                "intervention {index} layer {layer} is outside layer count {layer_count}"
            ));
        }
        if !coefficient.is_finite() || coefficient == 0.0 {
            return invalid(format!(
                "intervention {index} coefficient must be finite and nonzero, got {coefficient}"
            ));
        }
        match intervention {
            PostBlockIntervention::Fixed { direction, .. }
            | PostBlockIntervention::ResidualL2Relative { direction, .. }
            | PostBlockIntervention::Projection { direction, .. } => {
                validate_intervention_tensor(session, direction, hidden_size, index, "direction")?;
            }
            PostBlockIntervention::SourceToTarget { source, target, .. } => {
                validate_intervention_tensor(session, source, hidden_size, index, "source")?;
                validate_intervention_tensor(session, target, hidden_size, index, "target")?;
            }
        }
    }
    Ok(())
}

fn validate_intervention_tensor(
    session: &MuseGlimmerTextSession,
    tensor: &MetalTensor,
    hidden_size: usize,
    index: usize,
    role: &str,
) -> Result<(), MuseGlimmerTextSessionError> {
    let byte_count = hidden_size
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            MuseGlimmerTextSessionError::Invalid(format!(
                "intervention {index} {role} byte count overflow"
            ))
        })?;
    let end = tensor
        .offset
        .checked_add(byte_count as u64)
        .ok_or_else(|| {
            MuseGlimmerTextSessionError::Invalid(format!(
                "intervention {index} {role} endpoint overflow"
            ))
        })?;
    if tensor.dtype != GgmlType::F32
        || tensor.shape != [hidden_size as u64]
        || tensor.n_elements() as usize != hidden_size
        || !tensor
            .offset
            .is_multiple_of(std::mem::align_of::<f32>() as u64)
        || end > tensor.buffer.length() as u64
    {
        return invalid(format!(
            "intervention {index} {role} must be aligned F32 [{hidden_size}], got {:?} {:?} offset={} buffer_bytes={}",
            tensor.dtype,
            tensor.shape,
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    let device = tensor.buffer.device().registryID();
    if device != session.device_registry_id {
        return invalid(format!(
            "intervention {index} {role} belongs to Metal device {device}, expected {}",
            session.device_registry_id
        ));
    }
    if session.aliases_mutable_buffer(tensor) {
        return invalid(format!(
            "intervention {index} {role} aliases mutable session storage"
        ));
    }
    Ok(())
}

fn validate_live_capture_layers(
    layer_ids: &[u32],
    layer_count: usize,
) -> Result<(), MuseGlimmerTextSessionError> {
    if layer_ids.is_empty() || layer_ids.len() > MUSE_GLIMMER_MAX_LIVE_CAPTURE_LAYERS {
        return invalid(format!(
            "live capture layer count must be in 1..={MUSE_GLIMMER_MAX_LIVE_CAPTURE_LAYERS}, got {}",
            layer_ids.len()
        ));
    }
    for (slot, &layer) in layer_ids.iter().enumerate() {
        if layer as usize >= layer_count {
            return invalid(format!(
                "live capture layer {layer} is outside layer count {layer_count}"
            ));
        }
        if slot > 0 && layer_ids[slot - 1] >= layer {
            return invalid("live capture layers must be sorted and unique");
        }
    }
    Ok(())
}

fn validate_multi_lens_capture_request(
    token_count: usize,
    target_blocks: &[u32],
    layer_count: usize,
    next_position: usize,
    capacity: usize,
) -> Result<(), MuseGlimmerTextSessionError> {
    if next_position != 0 {
        return invalid(format!(
            "lens capture requires a fresh session at position zero, got {next_position}"
        ));
    }
    if token_count == 0 || token_count > MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS {
        return invalid(format!(
            "lens capture prompt length must be in 1..={MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS}, got {token_count}"
        ));
    }
    if token_count > capacity {
        return invalid(format!(
            "lens capture of {token_count} tokens exceeds session capacity {capacity}"
        ));
    }
    if target_blocks.is_empty() || target_blocks.len() > layer_count {
        return invalid(format!(
            "lens capture block count must be in 1..={layer_count}, got {}",
            target_blocks.len()
        ));
    }
    for (slot, &block) in target_blocks.iter().enumerate() {
        if block == 0 || block as usize >= layer_count {
            return invalid(format!(
                "lens capture target block must be in 1..{layer_count}, got {block}"
            ));
        }
        if slot > 0 && target_blocks[slot - 1] >= block {
            return invalid("lens capture target blocks must be sorted and unique");
        }
    }
    Ok(())
}

fn packed_row(tensor: &MetalTensor, row: usize, width: usize) -> MetalTensor {
    tensor.view_subrange((row * width) as u64, vec![width as u64])
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

fn read_i32(tensor: &MetalTensor) -> Vec<i32> {
    let mut values = vec![0_i32; tensor.n_elements() as usize];
    unsafe {
        std::ptr::copy_nonoverlapping(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<i32>(),
            values.as_mut_ptr(),
            values.len(),
        );
    }
    values
}

fn write_f32_prefix(
    tensor: &MetalTensor,
    values: &[f32],
) -> Result<(), MuseGlimmerTextSessionError> {
    if values.len() > tensor.n_elements() as usize {
        return invalid(format!(
            "Metal tensor prefix write length {} exceeds capacity {}",
            values.len(),
            tensor.n_elements()
        ));
    }
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

fn build_full_readout_rows(
    row_count: usize,
    vocab_size: usize,
    top_k: usize,
    first_ids: &[i32],
    first_values: &[f32],
    second_ids: &[i32],
    second_values: &[f32],
) -> Result<Vec<MuseGlimmerFullReadoutRow>, MuseGlimmerTextSessionError> {
    if top_k == 0 || top_k > MUSE_GLIMMER_FULL_READOUT_MAX_TOP_K || top_k > vocab_size {
        return invalid(format!(
            "full-readout top-k {top_k} is outside the supported vocabulary bound"
        ));
    }
    let expected = row_count
        .checked_mul(MUSE_GLIMMER_FULL_READOUT_PASS_K)
        .ok_or_else(|| {
            MuseGlimmerTextSessionError::Invalid("full-readout result size overflow".into())
        })?;
    if first_ids.len() != expected || first_values.len() != expected {
        return invalid("full-readout first-pass result size is inconsistent");
    }
    let second_expected = if top_k > MUSE_GLIMMER_FULL_READOUT_PASS_K {
        expected
    } else {
        0
    };
    if second_ids.len() != second_expected || second_values.len() != second_expected {
        return invalid("full-readout second-pass result size is inconsistent");
    }
    let mut rows = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let base = row * MUSE_GLIMMER_FULL_READOUT_PASS_K;
        let candidate_capacity = if second_expected == 0 {
            MUSE_GLIMMER_FULL_READOUT_PASS_K
        } else {
            MUSE_GLIMMER_FULL_READOUT_MAX_TOP_K
        };
        let mut scores = Vec::with_capacity(candidate_capacity);
        append_full_readout_pass(
            row,
            vocab_size,
            "first",
            &first_ids[base..base + MUSE_GLIMMER_FULL_READOUT_PASS_K],
            &first_values[base..base + MUSE_GLIMMER_FULL_READOUT_PASS_K],
            &mut scores,
        )?;
        if second_expected != 0 {
            append_full_readout_pass(
                row,
                vocab_size,
                "second",
                &second_ids[base..base + MUSE_GLIMMER_FULL_READOUT_PASS_K],
                &second_values[base..base + MUSE_GLIMMER_FULL_READOUT_PASS_K],
                &mut scores,
            )?;
        }
        scores.sort_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        scores.truncate(top_k);
        rows.push(MuseGlimmerFullReadoutRow { row, scores });
    }
    Ok(rows)
}

fn append_full_readout_pass(
    row: usize,
    vocab_size: usize,
    pass: &str,
    ids: &[i32],
    values: &[f32],
    scores: &mut Vec<MuseGlimmerFullReadoutScore>,
) -> Result<(), MuseGlimmerTextSessionError> {
    debug_assert_eq!(ids.len(), MUSE_GLIMMER_FULL_READOUT_PASS_K);
    debug_assert_eq!(values.len(), MUSE_GLIMMER_FULL_READOUT_PASS_K);
    let mut previous: Option<MuseGlimmerFullReadoutScore> = None;
    for (&token_id, &logit) in ids.iter().zip(values) {
        if token_id < 0 || token_id as usize >= vocab_size {
            return invalid(format!(
                "full-readout row {row} {pass} pass returned invalid token ID {token_id}"
            ));
        }
        if !logit.is_finite() {
            return invalid(format!(
                "full-readout row {row} {pass} pass returned non-finite compact logit"
            ));
        }
        let score = MuseGlimmerFullReadoutScore {
            token_id: token_id as u32,
            logit,
        };
        if scores
            .iter()
            .any(|existing| existing.token_id == score.token_id)
        {
            return invalid(format!(
                "full-readout row {row} returned duplicate token ID {token_id}"
            ));
        }
        if previous.as_ref().is_some_and(|prior| {
            let order = prior.logit.total_cmp(&score.logit);
            order.is_lt() || (order.is_eq() && prior.token_id > score.token_id)
        }) {
            return invalid(format!(
                "full-readout row {row} {pass} pass is not deterministically ordered"
            ));
        }
        previous = Some(score.clone());
        scores.push(score);
    }
    Ok(())
}

fn session_allocation_specs(
    geometry: &MuseGlimmerTextGeometry,
) -> Result<Vec<(String, u64)>, MuseGlimmerTextSessionError> {
    let mut specs = Vec::with_capacity(32);
    specs.push(("session.ids".into(), std::mem::size_of::<i32>() as u64));
    for name in [
        "session.embedding_norm_weight",
        "session.residual",
        "session.normed",
        "session.branch_raw",
        "session.branch_normed",
    ] {
        specs.push((
            name.into(),
            checked_bytes(geometry.hidden_size, std::mem::size_of::<f32>(), name)?,
        ));
    }
    for name in [
        "session.query_raw",
        "session.query",
        "session.attention_gate",
        "session.attention_output",
    ] {
        specs.push((
            name.into(),
            checked_bytes(geometry.query_width, std::mem::size_of::<f32>(), name)?,
        ));
    }
    for name in ["session.key_raw", "session.key", "session.value"] {
        specs.push((
            name.into(),
            checked_bytes(geometry.kv_width, std::mem::size_of::<f32>(), name)?,
        ));
    }
    for name in ["session.feed_forward_gate", "session.feed_forward_up"] {
        specs.push((
            name.into(),
            checked_bytes(geometry.feed_forward_size, std::mem::size_of::<f32>(), name)?,
        ));
    }
    specs.push((
        "session.packed.ids".into(),
        checked_bytes(
            MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS,
            std::mem::size_of::<i32>(),
            "session.packed.ids",
        )?,
    ));
    for name in [
        "session.packed.residual",
        "session.packed.normed",
        "session.packed.branch_raw",
        "session.packed.branch_normed",
    ] {
        specs.push((
            name.into(),
            checked_packed_f32_bytes(geometry.hidden_size, name)?,
        ));
    }
    for name in [
        "session.packed.query_raw",
        "session.packed.query",
        "session.packed.attention_gate",
        "session.packed.attention_output",
    ] {
        specs.push((
            name.into(),
            checked_packed_f32_bytes(geometry.query_width, name)?,
        ));
    }
    for name in [
        "session.packed.key_raw",
        "session.packed.key",
        "session.packed.value",
    ] {
        specs.push((
            name.into(),
            checked_packed_f32_bytes(geometry.kv_width, name)?,
        ));
    }
    for name in [
        "session.packed.feed_forward_gate",
        "session.packed.feed_forward_up",
    ] {
        specs.push((
            name.into(),
            checked_packed_f32_bytes(geometry.feed_forward_size, name)?,
        ));
    }
    specs.push((
        "session.logits".into(),
        checked_bytes(
            geometry.vocab_size,
            std::mem::size_of::<f32>(),
            "session.logits",
        )?,
    ));
    let cache_bytes = checked_bytes(
        geometry.cache_elements()?,
        std::mem::size_of::<u16>(),
        "session F16 cache",
    )?;
    specs.push(("session.key_cache".into(), cache_bytes));
    specs.push(("session.value_cache".into(), cache_bytes));
    Ok(specs)
}

fn checked_packed_f32_bytes(
    row_width: usize,
    label: &str,
) -> Result<u64, MuseGlimmerTextSessionError> {
    let elements = row_width
        .checked_mul(MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS)
        .ok_or_else(|| {
            MuseGlimmerTextSessionError::Invalid(format!("{label} packed element count overflow"))
        })?;
    checked_bytes(elements, std::mem::size_of::<f32>(), label)
}

fn checked_bytes(
    elements: usize,
    element_size: usize,
    label: &str,
) -> Result<u64, MuseGlimmerTextSessionError> {
    elements
        .checked_mul(element_size)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| MuseGlimmerTextSessionError::Invalid(format!("{label} byte count overflow")))
}

fn validate_deployed_output_residual(
    residual: &[f32],
    hidden_size: usize,
) -> Result<(), MuseGlimmerTextSessionError> {
    if residual.len() != hidden_size {
        return invalid(format!(
            "deployed output residual must contain exactly {hidden_size} F32 values, got {}",
            residual.len()
        ));
    }
    if let Some(index) = residual.iter().position(|value| !value.is_finite()) {
        return invalid(format!(
            "deployed output residual contains a non-finite F32 value at index {index}"
        ));
    }
    Ok(())
}

fn validate_f16_transport(
    transport_bytes: &[u8],
    hidden_size: usize,
) -> Result<(), MuseGlimmerTextSessionError> {
    let expected = hidden_size
        .checked_mul(hidden_size)
        .and_then(|values| values.checked_mul(2))
        .ok_or_else(|| {
            MuseGlimmerTextSessionError::Invalid("F16 transport byte count overflow".into())
        })?;
    if transport_bytes.len() != expected {
        return invalid(format!(
            "F16 post-block transport must contain exactly {expected} bytes, got {}",
            transport_bytes.len()
        ));
    }
    for (index, chunk) in transport_bytes.chunks_exact(2).enumerate() {
        let value = half::f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap()));
        if !value.is_finite() {
            return invalid(format!(
                "F16 post-block transport contains a non-finite value at index {index}"
            ));
        }
    }
    Ok(())
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, MuseGlimmerTextSessionError> {
    Err(MuseGlimmerTextSessionError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::muse_glimmer_residency::MuseGlimmerMetalWeightPlan;
    use crate::tokenizer::LlamaCppTokenizer;
    use sha2::{Digest, Sha256};

    fn scalar_top_k(logits: &[f32], top_k: usize) -> Vec<MuseGlimmerFullReadoutScore> {
        let mut scores = logits
            .iter()
            .copied()
            .enumerate()
            .map(|(token_id, logit)| MuseGlimmerFullReadoutScore {
                token_id: token_id as u32,
                logit,
            })
            .collect::<Vec<_>>();
        scores.sort_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        scores.truncate(top_k);
        scores
    }

    #[test]
    fn deterministic_gpu_topk_and_compaction_match_scalar_rows_exactly() {
        let mut signed_zero = vec![-5.0; 64];
        signed_zero[0] = -0.0;
        signed_zero[1] = 0.0;
        let logits = [
            (0..64)
                .map(|token| ((token * 37 + 11) % 101) as f32 - token as f32 * 0.001)
                .collect::<Vec<_>>(),
            (0..64)
                .map(|token| if token < 20 { 9.0 } else { -(token as f32) })
                .collect::<Vec<_>>(),
            signed_zero,
        ];
        let flattened = logits.concat();
        let ctx = MetalContext::new().unwrap();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&flattened),
            vec![3, 64],
            GgmlType::F32,
        )
        .unwrap();
        let first_ids = MetalTensor::zeros_i32(&ctx, vec![3, 16]).unwrap();
        let first_values = MetalTensor::zeros_f32(&ctx, vec![3, 16]).unwrap();
        let second_ids = MetalTensor::zeros_i32(&ctx, vec![3, 16]).unwrap();
        let second_values = MetalTensor::zeros_f32(&ctx, vec![3, 16]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_topk16_f32(&ctx, &encoder, &input, &first_ids, &first_values, 3, 64).unwrap();
        encode_mask_row_indices_f32(&ctx, &encoder, &input, &first_ids, 3, 64, 16).unwrap();
        encode_topk16_f32(&ctx, &encoder, &input, &second_ids, &second_values, 3, 64).unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let first_ids = read_i32(&first_ids);
        let first_values = read_f32(&first_values);
        let second_ids = read_i32(&second_ids);
        let second_values = read_f32(&second_values);
        for top_k in [16, 17, 25, 32] {
            let (second_ids, second_values) = if top_k > 16 {
                (second_ids.as_slice(), second_values.as_slice())
            } else {
                (&[][..], &[][..])
            };
            let actual = build_full_readout_rows(
                3,
                64,
                top_k,
                &first_ids,
                &first_values,
                second_ids,
                second_values,
            )
            .unwrap();
            for (row, source) in actual.iter().zip(&logits) {
                assert_eq!(row.scores, scalar_top_k(source, top_k));
            }
        }
    }

    #[test]
    fn full_readout_compaction_rejects_non_deterministic_order() {
        let ids = (0..MUSE_GLIMMER_FULL_READOUT_PASS_K as i32).collect::<Vec<_>>();
        let values = (0..MUSE_GLIMMER_FULL_READOUT_PASS_K)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert!(build_full_readout_rows(1, 64, 8, &ids, &values, &[], &[]).is_err());
    }

    #[test]
    fn full_readout_workspace_plan_is_bounded_and_prices_every_buffer() {
        let ctx = MetalContext::new().unwrap();
        assert!(MuseGlimmerFullReadoutWorkspacePlan::for_model(&ctx, 64, 256, 0).is_err());
        assert!(
            MuseGlimmerFullReadoutWorkspacePlan::for_model(
                &ctx,
                64,
                256,
                MUSE_GLIMMER_FULL_READOUT_MAX_ROWS + 1,
            )
            .is_err()
        );
        let plan = MuseGlimmerFullReadoutWorkspacePlan::for_model(
            &ctx,
            64,
            256,
            MUSE_GLIMMER_FULL_READOUT_MAX_ROWS,
        )
        .unwrap();
        assert_eq!(plan.row_capacity(), MUSE_GLIMMER_FULL_READOUT_MAX_ROWS);
        assert_eq!(
            plan.logical_bytes(),
            (3 * 64 * MUSE_GLIMMER_FULL_READOUT_MAX_ROWS * std::mem::size_of::<f32>()
                + 256 * MUSE_GLIMMER_FULL_READOUT_MAX_ROWS * std::mem::size_of::<f32>()
                + 2 * MUSE_GLIMMER_FULL_READOUT_PASS_K
                    * MUSE_GLIMMER_FULL_READOUT_MAX_ROWS
                    * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>())) as u64
        );
        assert!(plan.priced_upper_bytes() >= plan.logical_bytes());
        assert!(plan.prepared_transport_reserve_bytes() >= 64 * 64 * 2);
        assert_eq!(plan.host_transport_reserve_bytes(), 64 * 64 * 2);
        let admission = plan.admission(&ctx);
        assert_eq!(
            admission.required_bytes,
            Some(
                plan.priced_upper_bytes()
                    + plan.prepared_transport_reserve_bytes()
                    + plan.host_transport_reserve_bytes()
            )
        );
        assert!(admission.admitted);
    }

    #[test]
    fn full_readout_head_modes_batch_bf16_and_q8_and_fallback_other_dtypes() {
        assert_eq!(
            muse_glimmer_full_readout_head_mode(GgmlType::BF16, false),
            Some(MuseGlimmerFullReadoutHeadMode::Bf16ScalarRows)
        );
        assert_eq!(
            muse_glimmer_full_readout_head_mode(GgmlType::Q8_0, true),
            Some(MuseGlimmerFullReadoutHeadMode::Q8Batch)
        );
        assert_eq!(
            muse_glimmer_full_readout_head_mode(GgmlType::Q8_0, false),
            None,
            "Q8 must fall back when the scalar _lcpp mode is disabled"
        );
        for dtype in [GgmlType::F16, GgmlType::F32, GgmlType::Q4_K] {
            assert_eq!(muse_glimmer_full_readout_head_mode(dtype, true), None);
        }
    }

    #[test]
    fn bf16_head_rows_match_separate_scalar_commands_bitwise() {
        let ctx = MetalContext::new().unwrap();
        let n_in = 64;
        let n_out = 7;
        let rows = 3;
        let weights = (0..n_in * n_out)
            .map(|index| half::bf16::from_f32(((index * 13 % 41) as f32 - 20.0) * 0.01))
            .collect::<Vec<_>>();
        let inputs = (0..rows * n_in)
            .map(|index| ((index * 17 % 53) as f32 - 26.0) * 0.02)
            .collect::<Vec<_>>();
        let weight = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&weights),
            vec![n_in as u64, n_out as u64],
            GgmlType::BF16,
        )
        .unwrap();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&inputs),
            vec![rows as u64, n_in as u64],
            GgmlType::F32,
        )
        .unwrap();
        let batched = MetalTensor::zeros_f32(&ctx, vec![rows as u64, n_out as u64]).unwrap();
        let scalar = MetalTensor::zeros_f32(&ctx, vec![rows as u64, n_out as u64]).unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for row in 0..rows {
            encode_mat_vec_dispatch(
                &ctx,
                &encoder,
                &weight,
                &packed_row(&input, row, n_in),
                &packed_row(&batched, row, n_out),
                n_in,
                n_out,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        for row in 0..rows {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_mat_vec_dispatch(
                &ctx,
                &encoder,
                &weight,
                &packed_row(&input, row, n_in),
                &packed_row(&scalar, row, n_out),
                n_in,
                n_out,
            )
            .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        }
        assert_eq!(
            read_f32(&batched)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            read_f32(&scalar)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn release_geometry_accepts_model_context_and_pins_cache_windows() {
        let config = MuseGlimmerConfig::unsloth_release_reference();
        let capacity = config.context_length as usize;
        let geometry = MuseGlimmerTextGeometry::from_config(&config, capacity).unwrap();
        assert_eq!(geometry.query_width, 4_096);
        assert_eq!(geometry.kv_width, 256);
        assert_eq!(geometry.layer_count, 52);
        assert_eq!(
            geometry
                .visible_cache_range(3, capacity - 1, false)
                .unwrap()
                .1,
            capacity * 256
        );
        let (offset, elements) = geometry.visible_cache_range(2, capacity - 1, true).unwrap();
        assert_eq!(offset, (2 * capacity + capacity - 2_048) * 256);
        assert_eq!(elements, 2_048 * 256);
        assert!(MuseGlimmerTextGeometry::from_config(&config, 7_169).is_ok());
        assert!(MuseGlimmerTextGeometry::from_config(&config, 22_612).is_ok());
        assert!(MuseGlimmerTextGeometry::from_config(&config, 0).is_err());
        assert!(MuseGlimmerTextGeometry::from_config(&config, capacity + 1).is_err());
    }

    #[test]
    fn session_memory_plan_accounts_for_both_full_f16_caches() {
        let ctx = MetalContext::new().unwrap();
        let config = MuseGlimmerConfig::unsloth_release_reference();
        let geometry = MuseGlimmerTextGeometry::from_config(&config, 2_048).unwrap();
        let plan = MuseGlimmerTextSessionMemoryPlan::for_geometry(&ctx, &geometry).unwrap();
        let cache_bytes = 2_u64 * 52 * 2_048 * 256 * 2;
        assert!(plan.logical_bytes() > cache_bytes);
        assert!(plan.logical_bytes() < cache_bytes + 64 * 1024 * 1024);
        assert!(plan.priced_upper_bytes() >= plan.logical_bytes());
        assert_eq!(plan.allocations().len(), 32);
        assert!(
            plan.allocations()
                .iter()
                .any(|allocation| allocation.name == "session.packed.feed_forward_up")
        );
    }

    #[test]
    fn allocates_and_resets_a_small_reference_session() {
        let ctx = MetalContext::new().unwrap();
        let config = MuseGlimmerConfig::unsloth_release_reference();
        let mut session = MuseGlimmerTextSession::new(&ctx, &config, 8).unwrap();
        assert_eq!(session.next_position(), 0);
        assert_eq!(session.remaining_forwards(), 8);
        assert_eq!(session.memory_plan().allocations().len(), 32);
        assert!(session.admission().admitted);
        session.reset().unwrap();
    }

    #[test]
    fn prefill_plan_uses_128_token_superchunks_and_sixteen_token_quantum() {
        for (tokens, packed_chunks, packed_tokens, scalar_tail) in [
            (0, 0, 0, 0),
            (1, 0, 0, 1),
            (15, 0, 0, 15),
            (16, 1, 16, 0),
            (17, 1, 16, 1),
            (31, 1, 16, 15),
            (32, 1, 32, 0),
            (127, 1, 112, 15),
            (128, 1, 128, 0),
            (144, 2, 144, 0),
            (256, 2, 256, 0),
            (6_229, 49, 6_224, 5),
        ] {
            let plan = MuseGlimmerPrefillPlan::for_tokens(tokens);
            assert_eq!(plan.packed_chunks, packed_chunks, "tokens={tokens}");
            assert_eq!(plan.packed_tokens(), packed_tokens, "tokens={tokens}");
            assert_eq!(plan.scalar_tail, scalar_tail, "tokens={tokens}");
            assert_eq!(
                plan.packed_tokens() + plan.scalar_tail,
                tokens,
                "tokens={tokens}"
            );
        }
    }

    #[test]
    fn packed_q8_projection_rows_match_singleton_accumulation_bitwise() {
        let ctx = MetalContext::new().unwrap();
        let n_in = 64;
        let n_out = 7;
        let rows = MUSE_GLIMMER_PACKED_PREFILL_QUANTUM;
        let mut weight_bytes = Vec::with_capacity(n_out * (n_in / 32) * 34);
        for output in 0..n_out {
            for block in 0..n_in / 32 {
                let scale = half::f16::from_f32(0.003 * (1 + output + block) as f32);
                weight_bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
                for lane in 0..32 {
                    let quant = ((output * 11 + block * 7 + lane * 3) % 31) as i8 - 15;
                    weight_bytes.push(quant as u8);
                }
            }
        }
        let weight = MetalTensor::from_bytes(
            &ctx,
            &weight_bytes,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q8_0,
        )
        .unwrap();
        let input_values = (0..rows * n_in)
            .map(|index| ((index * 17 % 97) as f32 - 48.0) * 0.0025)
            .collect::<Vec<_>>();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input_values),
            vec![n_in as u64, rows as u64],
            GgmlType::F32,
        )
        .unwrap();
        let packed = MetalTensor::zeros_f32(&ctx, vec![n_out as u64, rows as u64]).unwrap();
        let singleton = MetalTensor::zeros_f32(&ctx, vec![n_out as u64, rows as u64]).unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_mat_vec_q8_0_batch_f32(&ctx, &encoder, &weight, &input, &packed, n_in, n_out, rows)
            .unwrap();
        for row in 0..rows {
            crate::metal::encode_mat_vec_q8_0_f32(
                &ctx,
                &encoder,
                &weight,
                &packed_row(&input, row, n_in),
                &packed_row(&singleton, row, n_out),
                n_in,
                n_out,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let packed = read_f32(&packed);
        let singleton = read_f32(&singleton);
        assert_eq!(
            packed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            singleton
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn live_capture_layers_require_nonempty_sorted_unique_in_range_selection() {
        validate_live_capture_layers(&[0, 3, 51], 52).unwrap();
        assert!(validate_live_capture_layers(&[], 52).is_err());
        assert!(validate_live_capture_layers(&[3, 3], 52).is_err());
        assert!(validate_live_capture_layers(&[4, 3], 52).is_err());
        assert!(validate_live_capture_layers(&[52], 52).is_err());
        assert!(validate_live_capture_layers(&vec![0; 65], 52).is_err());
    }

    #[test]
    fn deployed_output_residual_requires_exact_finite_hidden_row() {
        validate_deployed_output_residual(&[0.0, -1.5, 2.0], 3).unwrap();
        assert!(validate_deployed_output_residual(&[0.0, 1.0], 3).is_err());
        assert!(validate_deployed_output_residual(&[0.0, 1.0, 2.0, 3.0], 3).is_err());
        assert!(validate_deployed_output_residual(&[0.0, f32::NAN, 2.0], 3).is_err());
        assert!(validate_deployed_output_residual(&[0.0, f32::INFINITY, 2.0], 3).is_err());
    }

    #[test]
    fn f16_transport_validation_requires_exact_finite_square_matrix() {
        let one = half::f16::ONE.to_bits().to_le_bytes();
        validate_f16_transport(&one.repeat(4), 2).unwrap();
        assert!(validate_f16_transport(&one.repeat(3), 2).is_err());
        let mut non_finite = one.repeat(4);
        non_finite[2..4].copy_from_slice(&half::f16::NAN.to_bits().to_le_bytes());
        assert!(validate_f16_transport(&non_finite, 2).is_err());
    }

    #[test]
    fn post_block_interventions_require_bounded_non_aliasing_f32_rows() {
        let ctx = MetalContext::new().unwrap();
        let config = MuseGlimmerConfig::unsloth_release_reference();
        let session = MuseGlimmerTextSession::new(&ctx, &config, 1).unwrap();
        let hidden = config.hidden_size as usize;
        let direction = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&vec![1.0_f32; hidden]),
            vec![hidden as u64],
            GgmlType::F32,
        )
        .unwrap();
        let valid = [
            PostBlockIntervention::Fixed {
                layer: 0,
                direction: &direction,
                coefficient: 1.0,
            },
            PostBlockIntervention::ResidualL2Relative {
                layer: 1,
                direction: &direction,
                coefficient: 0.1,
            },
            PostBlockIntervention::Projection {
                layer: 2,
                direction: &direction,
                coefficient: 1.0,
            },
            PostBlockIntervention::SourceToTarget {
                layer: 51,
                source: &direction,
                target: &direction,
                coefficient: 0.5,
            },
        ];
        validate_post_block_interventions(&session, &valid, 52, hidden).unwrap();

        let bad_layer = [PostBlockIntervention::Fixed {
            layer: 52,
            direction: &direction,
            coefficient: 1.0,
        }];
        assert!(validate_post_block_interventions(&session, &bad_layer, 52, hidden).is_err());
        let bad_coefficient = [PostBlockIntervention::Projection {
            layer: 1,
            direction: &direction,
            coefficient: 0.0,
        }];
        assert!(validate_post_block_interventions(&session, &bad_coefficient, 52, hidden).is_err());
        let short = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[1.0_f32]),
            vec![1],
            GgmlType::F32,
        )
        .unwrap();
        let bad_shape = [PostBlockIntervention::Fixed {
            layer: 1,
            direction: &short,
            coefficient: 1.0,
        }];
        assert!(validate_post_block_interventions(&session, &bad_shape, 52, hidden).is_err());
        let alias = [PostBlockIntervention::Fixed {
            layer: 1,
            direction: &session.residual,
            coefficient: 1.0,
        }];
        assert!(validate_post_block_interventions(&session, &alias, 52, hidden).is_err());
    }

    #[test]
    fn multi_lens_capture_requires_fresh_sorted_unique_nonzero_blocks() {
        validate_multi_lens_capture_request(3, &[1, 50, 51], 52, 0, 3).unwrap();
        assert!(validate_multi_lens_capture_request(3, &[], 52, 0, 3).is_err());
        assert!(validate_multi_lens_capture_request(3, &[0], 52, 0, 3).is_err());
        assert!(validate_multi_lens_capture_request(3, &[50, 50], 52, 0, 3).is_err());
        assert!(validate_multi_lens_capture_request(3, &[51, 50], 52, 0, 3).is_err());
        assert!(validate_multi_lens_capture_request(3, &[52], 52, 0, 3).is_err());
        assert!(validate_multi_lens_capture_request(3, &[50], 52, 1, 3).is_err());
        assert!(validate_multi_lens_capture_request(17, &[50], 52, 0, 17).is_err());
    }

    #[test]
    #[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
    fn packed_q8_prefill_endpoint_and_scalar_continuation_match_scalar_bitwise() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF")
            .unwrap_or_else(|_| crate::test_fixtures::MUSE_GLIMMER_Q8_0.path().into());
        let gguf = GgufFile::open(&path).expect("open Muse Q8 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf)
            .expect("qualify and plan Muse Q8 target");
        let admitted = plan
            .admit(ctx.memory_signals())
            .expect("admit Muse Q8 residency");
        let realized = MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted)
            .expect("realize Muse Q8 weights");
        let weights = realized.into_weights();
        let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse forward");
        let mut scalar = MuseGlimmerTextSession::new(&ctx, weights.config(), 17)
            .expect("allocate scalar session");
        let mut packed = MuseGlimmerTextSession::new(&ctx, weights.config(), 17)
            .expect("allocate packed session");
        let mut tokens = vec![weights.config().bos_token_id];
        tokens.extend((0..16).map(|index| 1_000 + index));

        let mut scalar_endpoint = None;
        for (index, &token) in tokens[..16].iter().enumerate() {
            scalar_endpoint = forward
                .execute_token(token, &mut scalar, index == 15)
                .expect("run scalar prefill token");
        }
        let scalar_endpoint = scalar_endpoint.expect("produce scalar endpoint logits");
        let packed_endpoint = forward
            .prefill(&tokens[..16], &mut packed)
            .expect("run packed prefill chunk");
        assert_logits_bitwise_equal("packed endpoint", &packed_endpoint, &scalar_endpoint);
        assert_eq!(scalar.next_position(), 16);
        assert_eq!(packed.next_position(), 16);

        let scalar_continuation = forward
            .forward_token(tokens[16], &mut scalar)
            .expect("run scalar baseline continuation");
        let packed_continuation = forward
            .forward_token(tokens[16], &mut packed)
            .expect("run packed-state continuation");
        assert_logits_bitwise_equal(
            "packed scalar continuation",
            &packed_continuation,
            &scalar_continuation,
        );
    }

    #[test]
    #[ignore = "serial Metal, Muse live-prefix rewind and branch exactness"]
    fn live_prefix_rewind_matches_fresh_logits_and_all_active_kv() {
        let gguf = GgufFile::open(crate::test_fixtures::MUSE_GLIMMER_Q8_0.path()).unwrap();
        let ctx = MetalContext::new().unwrap();
        let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).unwrap();
        let admitted = plan.admit(ctx.memory_signals()).unwrap();
        let weights = MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted)
            .unwrap()
            .into_weights();
        let forward = MuseGlimmerTextForward::new(&ctx, &weights).unwrap();
        let mut live = MuseGlimmerTextSession::new(&ctx, weights.config(), 160).unwrap();
        let mut fresh = MuseGlimmerTextSession::new(&ctx, weights.config(), 160).unwrap();
        let root: Vec<u32> = (0..145).map(|i| 1000 + i).collect();
        forward.prefill(&root, &mut live).unwrap();
        let mut history = root.clone();
        let mut branch = root[..33].to_vec();
        branch[16] += 1;
        let mut unrelated = root[..129].to_vec();
        unrelated[0] += 1;
        let mut extension = unrelated.clone();
        extension.extend_from_slice(&root[129..]);
        for prompt in [
            root.clone(),
            root[..128].to_vec(),
            root[..17].to_vec(),
            branch,
            root[..1].to_vec(),
            unrelated,
            extension,
        ] {
            let reused = history
                .iter()
                .zip(&prompt)
                .take_while(|(a, b)| a == b)
                .count()
                .min(prompt.len() - 1);
            live.rewind_prefix(reused).unwrap();
            let actual = forward.prefill(&prompt[reused..], &mut live).unwrap();
            fresh.reset().unwrap();
            let expected = forward.prefill(&prompt, &mut fresh).unwrap();
            assert_logits_bitwise_equal("rewound prompt", &actual, &expected);
            assert_eq!(live.next_position(), prompt.len());
            for layer in 0..live.geometry.layer_count {
                let offset = live.geometry.cache_write_offset(layer, 0).unwrap() * 2;
                let bytes = prompt.len() * live.geometry.kv_width * 2;
                for (a, b) in [
                    (&live.key_cache, &fresh.key_cache),
                    (&live.value_cache, &fresh.value_cache),
                ] {
                    unsafe {
                        let a = std::slice::from_raw_parts(
                            (a.buffer.contents().as_ptr() as *const u8).add(offset),
                            bytes,
                        );
                        let b = std::slice::from_raw_parts(
                            (b.buffer.contents().as_ptr() as *const u8).add(offset),
                            bytes,
                        );
                        assert_eq!(a, b, "active KV differs at layer {layer}");
                    }
                }
            }
            let token = greedy_argmax(&expected);
            let actual = forward.forward_token(token, &mut live).unwrap();
            let expected = forward.forward_token(token, &mut fresh).unwrap();
            assert_logits_bitwise_equal("rewound continuation", &actual, &expected);
            history = prompt;
            history.push(token);
        }
        let position = live.next_position();
        assert!(live.rewind_prefix(position + 1).is_err());
        assert_eq!(live.next_position(), position);
        live.poison_reason = Some("injected command failure".into());
        assert!(live.rewind_prefix(0).is_err());
        assert!(live.reset().is_err());
        assert_eq!(live.next_position(), position);
    }

    #[test]
    #[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
    fn packed_q8_prefill_n144_superchunk_state_matches_scalar_bitwise() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF")
            .unwrap_or_else(|_| crate::test_fixtures::MUSE_GLIMMER_Q8_0.path().into());
        let gguf = GgufFile::open(&path).expect("open Muse Q8 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf)
            .expect("qualify and plan Muse Q8 target");
        let admitted = plan
            .admit(ctx.memory_signals())
            .expect("admit Muse Q8 residency");
        let realized = MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted)
            .expect("realize Muse Q8 weights");
        let weights = realized.into_weights();
        let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse forward");
        let mut scalar = MuseGlimmerTextSession::new(&ctx, weights.config(), 145)
            .expect("allocate scalar session");
        let mut packed = MuseGlimmerTextSession::new(&ctx, weights.config(), 145)
            .expect("allocate packed session");
        let tokens = std::iter::once(weights.config().bos_token_id)
            .chain((1..145).map(|index| 1_000 + index))
            .collect::<Vec<_>>();

        let mut scalar_endpoint = None;
        for (index, &token) in tokens[..144].iter().enumerate() {
            scalar_endpoint = forward
                .execute_token(token, &mut scalar, index == 143)
                .expect("run scalar prefill token");
        }
        let scalar_endpoint = scalar_endpoint.expect("produce scalar endpoint logits");
        let packed_endpoint = forward
            .prefill(&tokens[..144], &mut packed)
            .expect("run 128-plus-16 packed prefill");
        assert_logits_bitwise_equal("N=144 packed endpoint", &packed_endpoint, &scalar_endpoint);
        assert_eq!(scalar.next_position(), 144);
        assert_eq!(packed.next_position(), 144);

        let scalar_continuation = forward
            .forward_token(tokens[144], &mut scalar)
            .expect("run scalar baseline continuation");
        let packed_continuation = forward
            .forward_token(tokens[144], &mut packed)
            .expect("run packed-state continuation");
        assert_logits_bitwise_equal(
            "N=144 packed scalar continuation",
            &packed_continuation,
            &scalar_continuation,
        );
    }

    #[test]
    #[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
    fn packed_q8_prefill_n128_wall_screen() {
        use std::time::Instant;

        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF")
            .unwrap_or_else(|_| crate::test_fixtures::MUSE_GLIMMER_Q8_0.path().into());
        let gguf = GgufFile::open(&path).expect("open Muse Q8 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf)
            .expect("qualify and plan Muse Q8 target");
        let admitted = plan
            .admit(ctx.memory_signals())
            .expect("admit Muse Q8 residency");
        let realized = MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted)
            .expect("realize Muse Q8 weights");
        let weights = realized.into_weights();
        let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse forward");
        let tokens = std::iter::once(weights.config().bos_token_id)
            .chain((1..128).map(|index| 1_000 + index))
            .collect::<Vec<_>>();
        let mut warm = MuseGlimmerTextSession::new(&ctx, weights.config(), 1)
            .expect("allocate warmup session");
        forward
            .forward_token(tokens[0], &mut warm)
            .expect("warm resident weights");
        let mut scalar = MuseGlimmerTextSession::new(&ctx, weights.config(), tokens.len())
            .expect("allocate scalar screen session");
        let mut packed = MuseGlimmerTextSession::new(&ctx, weights.config(), tokens.len())
            .expect("allocate packed screen session");

        let mut run_scalar = || {
            scalar.reset().expect("reset scalar screen session");
            let started = Instant::now();
            let mut logits = None;
            for (index, &token) in tokens.iter().enumerate() {
                logits = forward
                    .execute_token(token, &mut scalar, index + 1 == tokens.len())
                    .expect("run scalar screen token");
            }
            (
                started.elapsed().as_secs_f64() * 1e3,
                logits.expect("produce scalar screen logits"),
            )
        };
        let mut run_packed = || {
            packed.reset().expect("reset packed screen session");
            let started = Instant::now();
            let logits = forward
                .prefill(&tokens, &mut packed)
                .expect("run packed screen prompt");
            (started.elapsed().as_secs_f64() * 1e3, logits)
        };

        let (scalar_first_ms, scalar_first) = run_scalar();
        let (packed_first_ms, packed_first) = run_packed();
        let (packed_second_ms, packed_second) = run_packed();
        let (scalar_second_ms, scalar_second) = run_scalar();
        for (label, logits) in [
            ("packed first", &packed_first),
            ("packed second", &packed_second),
            ("scalar replay", &scalar_second),
        ] {
            assert_logits_bitwise_equal(label, logits, &scalar_first);
        }
        let scalar_ms = (scalar_first_ms + scalar_second_ms) * 0.5;
        let packed_ms = (packed_first_ms + packed_second_ms) * 0.5;
        let ratio = packed_ms / scalar_ms;
        eprintln!(
            "Muse packed N=128 wall screen: scalar={scalar_ms:.2} ms packed={packed_ms:.2} ms ratio={ratio:.3} scalar_tps={:.2} packed_tps={:.2}",
            tokens.len() as f64 / (scalar_ms / 1e3),
            tokens.len() as f64 / (packed_ms / 1e3),
        );
        assert!(
            ratio <= 0.85,
            "packed N=128 wall ratio {ratio:.3} exceeds 0.85 promotion gate"
        );
    }

    #[test]
    #[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
    fn packed_q8_mat_mat_n128_numerical_and_wall_screen() {
        use std::time::Instant;

        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF")
            .unwrap_or_else(|_| crate::test_fixtures::MUSE_GLIMMER_Q8_0.path().into());
        let tokenizer = LlamaCppTokenizer::open(&path).expect("open Muse tokenizer");
        let gguf = GgufFile::open(&path).expect("open Muse Q8 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf)
            .expect("qualify and plan Muse Q8 target");
        let admitted = plan
            .admit(ctx.memory_signals())
            .expect("admit Muse Q8 residency");
        let realized = MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted)
            .expect("realize Muse Q8 weights");
        let weights = realized.into_weights();
        let exact_forward =
            MuseGlimmerTextForward::new_with_packed_q8_mat_mat(&ctx, &weights, false)
                .expect("bind exact Muse forward");
        let matrix_forward =
            MuseGlimmerTextForward::new_with_packed_q8_mat_mat(&ctx, &weights, true)
                .expect("bind matrix Muse forward");
        let prompt = "The quick brown fox jumps over the lazy dog while the patient astronomer records each changing constellation. ".repeat(32);
        let mut tokens = tokenizer
            .encode(&prompt, true)
            .expect("tokenize natural Muse screen prompt")
            .into_iter()
            .map(|token| u32::try_from(token).expect("nonnegative Muse token"))
            .collect::<Vec<_>>();
        assert!(tokens.len() >= 128);
        tokens.truncate(128);

        let capacity = tokens.len() + 16;
        let mut warm = MuseGlimmerTextSession::new(&ctx, weights.config(), 1)
            .expect("allocate warmup session");
        exact_forward
            .forward_token(tokens[0], &mut warm)
            .expect("warm resident weights");
        let mut exact = MuseGlimmerTextSession::new(&ctx, weights.config(), capacity)
            .expect("allocate exact screen session");
        let mut matrix = MuseGlimmerTextSession::new(&ctx, weights.config(), capacity)
            .expect("allocate matrix screen session");

        let mut run_exact = || {
            exact.reset().expect("reset exact screen session");
            let started = Instant::now();
            let logits = exact_forward
                .prefill(&tokens, &mut exact)
                .expect("run exact packed screen prompt");
            (started.elapsed().as_secs_f64() * 1e3, logits)
        };
        let mut run_matrix = || {
            matrix.reset().expect("reset matrix screen session");
            let started = Instant::now();
            let logits = matrix_forward
                .prefill(&tokens, &mut matrix)
                .expect("run matrix packed screen prompt");
            (started.elapsed().as_secs_f64() * 1e3, logits)
        };
        let (exact_first_ms, exact_first) = run_exact();
        let (matrix_first_ms, matrix_first) = run_matrix();
        let (matrix_second_ms, matrix_second) = run_matrix();
        let (exact_second_ms, exact_second) = run_exact();
        assert_logits_bitwise_equal("exact replay", &exact_second, &exact_first);
        assert_logits_bitwise_equal("matrix replay", &matrix_second, &matrix_first);

        let endpoint = compare_logits(&matrix_second, &exact_second);
        let exact_ms = (exact_first_ms + exact_second_ms) * 0.5;
        let matrix_ms = (matrix_first_ms + matrix_second_ms) * 0.5;
        let ratio = matrix_ms / exact_ms;
        eprintln!(
            "Muse Q8 matrix N=128 screen: exact={exact_ms:.2} ms matrix={matrix_ms:.2} ms ratio={ratio:.3} exact_tps={:.2} matrix_tps={:.2} endpoint={endpoint:?}",
            tokens.len() as f64 / (exact_ms / 1e3),
            tokens.len() as f64 / (matrix_ms / 1e3),
        );
        assert!(
            ratio <= 0.75,
            "matrix N=128 wall ratio {ratio:.3} exceeds 0.75 diagnostic performance gate"
        );
        assert!(endpoint.cosine > 0.999_99, "endpoint {endpoint:?}");
        assert!(endpoint.relative_rms < 0.002, "endpoint {endpoint:?}");
        assert!(endpoint.max_abs < 0.1, "endpoint {endpoint:?}");
        assert_eq!(
            endpoint.candidate_argmax, endpoint.reference_argmax,
            "endpoint {endpoint:?}"
        );

        let packed = matrix
            .packed
            .views(&matrix.geometry, tokens.len())
            .expect("bind matrix stage views");
        let (projection_wall_ms, projection_gpu_ms) = measure_gpu_chain(&ctx, |encoder| {
            for layer in &matrix_forward.weights.layers {
                matrix_forward.encode_packed_projection(
                    encoder,
                    layer.attention_query,
                    &packed.normed,
                    &packed.query_raw,
                    matrix.geometry.hidden_size,
                    matrix.geometry.query_width,
                    tokens.len(),
                )?;
                matrix_forward.encode_packed_projection(
                    encoder,
                    layer.attention_key,
                    &packed.normed,
                    &packed.key_raw,
                    matrix.geometry.hidden_size,
                    matrix.geometry.kv_width,
                    tokens.len(),
                )?;
                matrix_forward.encode_packed_projection(
                    encoder,
                    layer.attention_value,
                    &packed.normed,
                    &packed.value,
                    matrix.geometry.hidden_size,
                    matrix.geometry.kv_width,
                    tokens.len(),
                )?;
                matrix_forward.encode_packed_projection(
                    encoder,
                    layer.attention_gate,
                    &packed.normed,
                    &packed.attention_gate,
                    matrix.geometry.hidden_size,
                    matrix.geometry.query_width,
                    tokens.len(),
                )?;
                matrix_forward.encode_packed_projection(
                    encoder,
                    layer.attention_output,
                    &packed.attention_output,
                    &packed.branch_raw,
                    matrix.geometry.query_width,
                    matrix.geometry.hidden_size,
                    tokens.len(),
                )?;
                matrix_forward.encode_packed_projection(
                    encoder,
                    layer.feed_forward_gate,
                    &packed.normed,
                    &packed.feed_forward_gate,
                    matrix.geometry.hidden_size,
                    matrix.geometry.feed_forward_size,
                    tokens.len(),
                )?;
                matrix_forward.encode_packed_projection(
                    encoder,
                    layer.feed_forward_up,
                    &packed.normed,
                    &packed.feed_forward_up,
                    matrix.geometry.hidden_size,
                    matrix.geometry.feed_forward_size,
                    tokens.len(),
                )?;
                matrix_forward.encode_packed_projection(
                    encoder,
                    layer.feed_forward_down,
                    &packed.feed_forward_gate,
                    &packed.branch_raw,
                    matrix.geometry.feed_forward_size,
                    matrix.geometry.hidden_size,
                    tokens.len(),
                )?;
            }
            Ok(())
        });
        let (attention_wall_ms, attention_gpu_ms) = measure_gpu_chain(&ctx, |encoder| {
            for (layer_index, layer) in matrix_forward.weights.layers.iter().enumerate() {
                for row in 0..tokens.len() {
                    let query = packed_row(&packed.query, row, matrix.geometry.query_width);
                    let output =
                        packed_row(&packed.attention_output, row, matrix.geometry.query_width);
                    let (key_cache, value_cache, visible_positions) =
                        matrix.cache_views(layer_index, row, layer.sliding_attention)?;
                    encode_muse_glimmer_attn_decode_f16kv_f32(
                        &ctx,
                        encoder,
                        &query,
                        &key_cache,
                        &value_cache,
                        &output,
                        matrix.geometry.query_head_count,
                        matrix.geometry.kv_head_count,
                        matrix.geometry.head_dim,
                        visible_positions,
                    )?;
                }
            }
            Ok(())
        });
        let (rope_wall_ms, rope_gpu_ms) = measure_gpu_chain(&ctx, |encoder| {
            for layer in &matrix_forward.weights.layers {
                if !layer.sliding_attention {
                    continue;
                }
                for row in 0..tokens.len() {
                    let query = packed_row(&packed.query, row, matrix.geometry.query_width);
                    let key = packed_row(&packed.key, row, matrix.geometry.kv_width);
                    encode_muse_glimmer_rope_adjacent_pair_in_place_f32(
                        &ctx,
                        encoder,
                        &query,
                        &key,
                        matrix.geometry.query_head_count,
                        matrix.geometry.kv_head_count,
                        matrix.geometry.head_dim,
                        row as u32,
                        matrix_forward.weights.config.rope_theta,
                    )?;
                }
            }
            Ok(())
        });
        eprintln!(
            "Muse Q8 matrix N=128 isolated synthetic chains: projections wall/gpu={projection_wall_ms:.2}/{projection_gpu_ms:.2} ms attention={attention_wall_ms:.2}/{attention_gpu_ms:.2} ms rope={rope_wall_ms:.2}/{rope_gpu_ms:.2} ms nonadditive_wall_sum={:.2} ms full_wall={matrix_ms:.2} ms",
            projection_wall_ms + attention_wall_ms + rope_wall_ms,
        );

        let mut exact_logits = exact_second;
        let mut matrix_logits = matrix_second;
        let mut minimum_cosine = endpoint.cosine;
        let mut maximum_relative_rms = endpoint.relative_rms;
        let mut maximum_absolute = endpoint.max_abs;
        let mut greedy_ids = Vec::with_capacity(16);
        for step in 0..16 {
            let exact_token = greedy_argmax(&exact_logits);
            let matrix_token = greedy_argmax(&matrix_logits);
            assert_eq!(
                matrix_token, exact_token,
                "greedy token mismatch at step {step}"
            );
            greedy_ids.push(exact_token);
            exact_logits = exact_forward
                .forward_token(exact_token, &mut exact)
                .expect("run exact scalar continuation");
            matrix_logits = matrix_forward
                .forward_token(matrix_token, &mut matrix)
                .expect("run matrix scalar continuation");
            let comparison = compare_logits(&matrix_logits, &exact_logits);
            minimum_cosine = minimum_cosine.min(comparison.cosine);
            maximum_relative_rms = maximum_relative_rms.max(comparison.relative_rms);
            maximum_absolute = maximum_absolute.max(comparison.max_abs);
        }
        eprintln!(
            "Muse Q8 matrix continuation: greedy_ids={greedy_ids:?} min_cosine={minimum_cosine:.9} max_relative_rms={maximum_relative_rms:.6e} max_abs={maximum_absolute:.6e}"
        );
        assert!(minimum_cosine > 0.999_99);
        assert!(maximum_relative_rms < 0.006);
        assert!(maximum_absolute < 0.3);
    }

    fn measure_gpu_chain<F>(ctx: &MetalContext, mut encode: F) -> (f64, f64)
    where
        F: FnMut(&KernelEncoder) -> Result<(), MuseGlimmerTextSessionError>,
    {
        let mut walls = Vec::with_capacity(3);
        let mut gpu = Vec::with_capacity(3);
        for iteration in 0..5 {
            let started = std::time::Instant::now();
            let command = ctx.queue.commandBuffer().expect("allocate stage command");
            let encoder = KernelEncoder::begin(&command);
            encode(&encoder).expect("encode stage chain");
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            if iteration >= 2 {
                walls.push(started.elapsed().as_secs_f64() * 1e3);
                gpu.push((command.GPUEndTime() - command.GPUStartTime()) * 1e3);
            }
        }
        walls.sort_by(f64::total_cmp);
        gpu.sort_by(f64::total_cmp);
        (walls[1], gpu[1])
    }

    #[derive(Clone, Copy, Debug)]
    struct LogitComparison {
        cosine: f64,
        relative_rms: f64,
        max_abs: f32,
        candidate_argmax: u32,
        reference_argmax: u32,
    }

    fn compare_logits(candidate: &[f32], reference: &[f32]) -> LogitComparison {
        assert_eq!(candidate.len(), reference.len());
        let mut dot = 0.0_f64;
        let mut candidate_sq = 0.0_f64;
        let mut reference_sq = 0.0_f64;
        let mut difference_sq = 0.0_f64;
        let mut max_abs = 0.0_f32;
        for (&candidate, &reference) in candidate.iter().zip(reference) {
            assert!(candidate.is_finite() && reference.is_finite());
            dot += f64::from(candidate) * f64::from(reference);
            candidate_sq += f64::from(candidate) * f64::from(candidate);
            reference_sq += f64::from(reference) * f64::from(reference);
            let difference = candidate - reference;
            difference_sq += f64::from(difference) * f64::from(difference);
            max_abs = max_abs.max(difference.abs());
        }
        LogitComparison {
            cosine: dot / (candidate_sq.sqrt() * reference_sq.sqrt()),
            relative_rms: (difference_sq / reference_sq).sqrt(),
            max_abs,
            candidate_argmax: greedy_argmax(candidate),
            reference_argmax: greedy_argmax(reference),
        }
    }

    fn greedy_argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index as u32)
            .expect("nonempty logits")
    }

    fn assert_logits_bitwise_equal(label: &str, left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len(), "{label} logit length");
        let mismatches = left
            .iter()
            .zip(right)
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        let max_abs = left
            .iter()
            .zip(right)
            .map(|(&left, &right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert_eq!(mismatches, 0, "{label}: max_abs={max_abs}");
    }

    #[test]
    #[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
    fn capture_enabled_prefill_and_decode_logits_match_ordinary_forward_bitwise() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF")
            .unwrap_or_else(|_| crate::test_fixtures::MUSE_GLIMMER_Q8_0.path().into());
        let gguf = GgufFile::open(&path).expect("open Muse Q8 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf)
            .expect("qualify and plan Muse Q8 target");
        let admitted = plan
            .admit(ctx.memory_signals())
            .expect("admit Muse Q8 residency");
        let realized = MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted)
            .expect("realize Muse Q8 weights");
        let weights = realized.into_weights();
        let mut session = MuseGlimmerTextSession::new(&ctx, weights.config(), 2)
            .expect("allocate Muse text session");
        let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse forward");
        let tokens = [weights.config().bos_token_id, 19_873];
        let ordinary_prefill = forward
            .forward_token(tokens[0], &mut session)
            .expect("ordinary prefill token");
        let ordinary_decode = forward
            .forward_token(tokens[1], &mut session)
            .expect("ordinary decode token");
        session.reset().expect("reset identical state");
        let captured_prefill = forward
            .forward_token_capture_post_blocks(tokens[0], &[0, 3, 50, 51], &mut session)
            .expect("capture prefill token");
        let captured_decode = forward
            .forward_token_capture_post_blocks(tokens[1], &[0, 3, 50, 51], &mut session)
            .expect("capture decode token");
        assert_eq!(
            ordinary_prefill
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            captured_prefill
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ordinary_decode
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            captured_decode
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        for capture in [&captured_prefill, &captured_decode] {
            assert_eq!(capture.layer_ids, [0, 3, 50, 51]);
            assert_eq!(capture.post_block_residuals.len(), 4 * 6_656);
            assert!(
                capture
                    .post_block_residuals
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
        assert_eq!(captured_prefill.position, 0);
        assert_eq!(captured_decode.position, 1);
        assert_eq!(session.next_position(), 2);

        let baseline = captured_prefill.layer_values(2).unwrap().to_vec();
        let mut unit = vec![0.0_f32; weights.config().hidden_size as usize];
        unit[0] = 1.0;
        let direction = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&unit),
            vec![unit.len() as u64],
            GgmlType::F32,
        )
        .unwrap();
        let fixed = PostBlockIntervention::Fixed {
            layer: 50,
            direction: &direction,
            coefficient: 1.0,
        };
        let projection = PostBlockIntervention::Projection {
            layer: 50,
            direction: &direction,
            coefficient: 0.5,
        };
        session.reset().unwrap();
        let fixed_then_projection = forward
            .forward_token_capture_post_blocks_with_interventions(
                tokens[0],
                &[50],
                &[fixed, projection],
                &mut session,
            )
            .unwrap();
        session.reset().unwrap();
        let projection_then_fixed = forward
            .forward_token_capture_post_blocks_with_interventions(
                tokens[0],
                &[50],
                &[projection, fixed],
                &mut session,
            )
            .unwrap();
        let mut expected_fixed_then_projection = baseline.clone();
        expected_fixed_then_projection[0] += 1.0;
        expected_fixed_then_projection[0] -= 0.5 * expected_fixed_then_projection[0];
        let mut expected_projection_then_fixed = baseline;
        expected_projection_then_fixed[0] -= 0.5 * expected_projection_then_fixed[0];
        expected_projection_then_fixed[0] += 1.0;
        for (actual, expected) in [
            (
                fixed_then_projection.layer_values(0).unwrap(),
                expected_fixed_then_projection.as_slice(),
            ),
            (
                projection_then_fixed.layer_values(0).unwrap(),
                expected_projection_then_fixed.as_slice(),
            ),
        ] {
            let max_abs = actual
                .iter()
                .zip(expected)
                .map(|(&left, &right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_abs < 1e-5, "ordered intervention max_abs={max_abs}");
        }
        assert!(
            (fixed_then_projection.layer_values(0).unwrap()[0]
                - projection_then_fixed.layer_values(0).unwrap()[0])
                .abs()
                > 0.49
        );
        assert!(
            fixed_then_projection
                .logits
                .iter()
                .zip(&projection_then_fixed.logits)
                .any(|(&left, &right)| left.to_bits() != right.to_bits())
        );
        assert!(
            fixed_then_projection
                .logits
                .iter()
                .chain(&projection_then_fixed.logits)
                .all(|value| value.is_finite())
        );
    }

    #[test]
    #[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
    fn runs_pinned_q8_first_token_after_dropping_gguf_owner() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF")
            .unwrap_or_else(|_| crate::test_fixtures::MUSE_GLIMMER_Q8_0.path().into());
        let tokenizer = LlamaCppTokenizer::open(&path).expect("open Muse tokenizer");
        let gguf = GgufFile::open(&path).expect("open Muse Q8 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf)
            .expect("qualify and plan Muse Q8 target");
        let admitted = plan
            .admit(ctx.memory_signals())
            .expect("admit Muse Q8 residency");
        let realized = MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted)
            .expect("realize Muse Q8 weights");
        drop(gguf);
        let weights = realized.into_weights();
        let mut session = MuseGlimmerTextSession::new(&ctx, weights.config(), 8)
            .expect("allocate Muse text session");
        let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse text forward");
        let logits = forward
            .forward_token(weights.config().bos_token_id, &mut session)
            .expect("run Muse BOS token");
        assert_eq!(logits.len(), weights.config().vocab_size as usize);
        assert!(logits.iter().all(|value| value.is_finite()));
        assert!(logits.iter().all(|value| value.abs() <= 20.000_1));
        assert_eq!(session.next_position(), 1);
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap();
        let mut digest = Sha256::new();
        for value in &logits {
            digest.update(value.to_le_bytes());
        }
        eprintln!(
            "Muse Q8 BOS: argmax={argmax} logit={} sha256_f32le={:x}",
            logits[argmax],
            digest.finalize()
        );

        session.reset().unwrap();
        let prompt_tokens = tokenizer
            .encode("Hello", true)
            .expect("tokenize raw prompt");
        assert_eq!(prompt_tokens.first().copied(), tokenizer.bos());
        let prompt_tokens = prompt_tokens
            .into_iter()
            .map(|token| u32::try_from(token).expect("nonnegative Muse token"))
            .collect::<Vec<_>>();
        let logits = forward
            .prefill(&prompt_tokens, &mut session)
            .expect("prefill raw Hello prompt");
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap();
        let piece = tokenizer
            .try_decode_piece_bytes_exact(argmax as i32)
            .expect("decode Muse argmax");
        eprintln!(
            "Muse Q8 raw Hello: tokens={prompt_tokens:?} argmax={argmax} piece={:?}",
            String::from_utf8_lossy(&piece)
        );
        assert_eq!(piece, b",");

        let oracle_path = std::env::var("MUSE_GLIMMER_Q8_HELLO_ORACLE")
            .unwrap_or_else(|_| "/tmp/muse-glimmer-oracle/hello-q8.f32".into());
        let oracle_bytes = std::fs::read(&oracle_path).expect("read llama.cpp-rs Hello logits");
        assert_eq!(
            oracle_bytes.len(),
            logits.len() * std::mem::size_of::<f32>()
        );
        let oracle = oracle_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let oracle_argmax = oracle
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap();
        let mut dot = 0.0_f64;
        let mut ours_sq = 0.0_f64;
        let mut oracle_sq = 0.0_f64;
        let mut diff_sq = 0.0_f64;
        let mut max_abs = 0.0_f32;
        for (&ours, &reference) in logits.iter().zip(&oracle) {
            dot += f64::from(ours) * f64::from(reference);
            ours_sq += f64::from(ours) * f64::from(ours);
            oracle_sq += f64::from(reference) * f64::from(reference);
            let difference = ours - reference;
            diff_sq += f64::from(difference) * f64::from(difference);
            max_abs = max_abs.max(difference.abs());
        }
        let cosine = dot / (ours_sq.sqrt() * oracle_sq.sqrt());
        let relative_rms = (diff_sq / oracle_sq).sqrt();
        eprintln!(
            "Muse Q8 raw Hello oracle: argmax={argmax}/{oracle_argmax} cosine={cosine:.9} relative_rms={relative_rms:.6e} max_abs={max_abs:.6e} top_logit={}/{}",
            logits[argmax], oracle[oracle_argmax]
        );
        assert_eq!(argmax, oracle_argmax);
        assert!(cosine > 0.999_999, "Muse/llama.cpp logits cosine {cosine}");
        assert!(
            relative_rms < 1e-4,
            "Muse/llama.cpp relative RMS {relative_rms}"
        );
        assert!(
            max_abs < 0.002,
            "Muse/llama.cpp maximum absolute error {max_abs}"
        );
    }
}
