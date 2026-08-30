//! Correctness-first scalar text execution for Muse Glimmer 30B.
//!
//! The initial path uses native Q8_0/BF16 projections and a contiguous F16 KV
//! cache. Its reference attention kernel supports at most 7,168 visible tokens;
//! sliding layers use a suffix view while full-attention layers use the entire
//! prefix. Packed prefill and long-context streaming attention are separate
//! follow-on optimizations.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalMemoryAdmission, MetalTensor,
    PostBlockIntervention, encode_add_inplace_f32, encode_attn_decode_f16kv_f32,
    encode_copy_offset_f32, encode_get_rows_f32, encode_mat_vec_f16_f32,
    encode_post_block_intervention_f32, encode_rms_norm_batched_f32, encode_rms_norm_mul_f32,
    encode_scatter_offset_f32_to_f16_kv, encode_sigmoid_mul_f32, encode_silu_mul_f32,
    evaluate_metal_memory_admission, host_page_size_bytes,
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
    encode_muse_glimmer_logit_softcap_f32, encode_muse_glimmer_rope_adjacent_pair_in_place_f32,
};
use crate::muse_glimmer_residency::{
    MuseGlimmerMetalModelWeights, MuseGlimmerMetalWeights, MuseGlimmerResidencyError,
};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLDevice, MTLResource,
};

pub const MUSE_GLIMMER_REFERENCE_ATTENTION_CAPACITY: usize = 7_168;
pub const MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
pub const MUSE_GLIMMER_MAX_LIVE_CAPTURE_LAYERS: usize = 64;

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
        if capacity == 0
            || capacity > config.context_length as usize
            || capacity > MUSE_GLIMMER_REFERENCE_ATTENTION_CAPACITY
        {
            return invalid(format!(
                "capacity must be in 1..={MUSE_GLIMMER_REFERENCE_ATTENTION_CAPACITY} for the reference attention kernel and no larger than model context {}, got {capacity}",
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
        let page_size = host_page_size_bytes()? as u64;
        let max_buffer_length = u64::try_from(ctx.max_buffer_length()).map_err(|_| {
            MuseGlimmerTextSessionError::Invalid("Metal maximum buffer length exceeds u64".into())
        })?;
        let mut allocations = Vec::new();
        let mut logical_bytes = 0_u64;
        let mut priced_upper_bytes = 0_u64;
        for (name, logical) in session_allocation_specs(geometry)? {
            if logical == 0 || logical > max_buffer_length {
                return invalid(format!(
                    "session allocation {name:?} requires {logical} bytes, outside 1..={max_buffer_length}"
                ));
            }
            let priced = ctx.shared_buffer_size_and_align(logical)?;
            if priced.size < logical || priced.alignment == 0 || !priced.alignment.is_power_of_two()
            {
                return invalid(format!(
                    "invalid Metal pricing for {name:?}: logical={logical} priced={} alignment={}",
                    priced.size, priced.alignment
                ));
            }
            let alignment = priced.alignment.max(page_size);
            let priced_bytes = priced
                .size
                .checked_add(alignment - 1)
                .map(|bytes| bytes / alignment * alignment)
                .ok_or_else(|| {
                    MuseGlimmerTextSessionError::Invalid(format!(
                        "aligned Metal pricing for {name:?} overflows u64"
                    ))
                })?;
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
        self.ensure_usable()?;
        self.next_position = 0;
        Ok(())
    }

    fn ensure_usable(&self) -> Result<(), MuseGlimmerTextSessionError> {
        if let Some(reason) = &self.poison_reason {
            return Err(MuseGlimmerTextSessionError::Poisoned(reason.clone()));
        }
        Ok(())
    }

    fn aliases_mutable_buffer(&self, candidate: &MetalTensor) -> bool {
        [
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
        .any(|tensor| Retained::as_ptr(&candidate.buffer) == Retained::as_ptr(&tensor.buffer))
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
}

impl<'ctx, 'model> MuseGlimmerTextForward<'ctx, 'model> {
    pub fn new(
        ctx: &'ctx MetalContext,
        resident: &'model MuseGlimmerMetalWeights,
    ) -> Result<Self, MuseGlimmerTextSessionError> {
        resident.validate_context(ctx)?;
        let weights = MuseGlimmerMetalModelWeights::bind(resident)?;
        Ok(Self { ctx, weights })
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
        let hidden = self.weights.config.hidden_size as usize;
        validate_f16_transport(transport_bytes, hidden)?;
        validate_deployed_output_residual(source_residual, hidden)?;
        let transport = MetalTensor::from_bytes(
            self.ctx,
            transport_bytes,
            vec![hidden as u64, hidden as u64],
            GgmlType::F16,
        )?;
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
            &transport,
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
        let final_index = tokens.len() - 1;
        let mut logits = None;
        for (index, &token) in tokens.iter().enumerate() {
            checkpoint()?;
            logits = self.execute_token(token, session, index == final_index)?;
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
            encode_attn_decode_f16kv_f32(
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

fn session_allocation_specs(
    geometry: &MuseGlimmerTextGeometry,
) -> Result<Vec<(String, u64)>, MuseGlimmerTextSessionError> {
    let mut specs = Vec::with_capacity(18);
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

    #[test]
    fn release_geometry_pins_reference_attention_limit_and_cache_windows() {
        let config = MuseGlimmerConfig::unsloth_release_reference();
        let geometry = MuseGlimmerTextGeometry::from_config(
            &config,
            MUSE_GLIMMER_REFERENCE_ATTENTION_CAPACITY,
        )
        .unwrap();
        assert_eq!(geometry.query_width, 4_096);
        assert_eq!(geometry.kv_width, 256);
        assert_eq!(geometry.layer_count, 52);
        assert_eq!(
            geometry.visible_cache_range(3, 4_095, false).unwrap().1,
            4_096 * 256
        );
        let (offset, elements) = geometry.visible_cache_range(2, 4_095, true).unwrap();
        assert_eq!(offset, (2 * 7_168 + 2_048) * 256);
        assert_eq!(elements, 2_048 * 256);
        assert!(MuseGlimmerTextGeometry::from_config(&config, 7_169).is_err());
    }

    #[test]
    fn session_memory_plan_accounts_for_both_full_f16_caches() {
        let ctx = MetalContext::new().unwrap();
        let config = MuseGlimmerConfig::unsloth_release_reference();
        let geometry = MuseGlimmerTextGeometry::from_config(&config, 2_048).unwrap();
        let plan = MuseGlimmerTextSessionMemoryPlan::for_geometry(&ctx, &geometry).unwrap();
        let cache_bytes = 2_u64 * 52 * 2_048 * 256 * 2;
        assert!(plan.logical_bytes() > cache_bytes);
        assert!(plan.logical_bytes() < cache_bytes + 10 * 1024 * 1024);
        assert!(plan.priced_upper_bytes() >= plan.logical_bytes());
        assert_eq!(plan.allocations().len(), 18);
    }

    #[test]
    fn allocates_and_resets_a_small_reference_session() {
        let ctx = MetalContext::new().unwrap();
        let config = MuseGlimmerConfig::unsloth_release_reference();
        let mut session = MuseGlimmerTextSession::new(&ctx, &config, 8).unwrap();
        assert_eq!(session.next_position(), 0);
        assert_eq!(session.remaining_forwards(), 8);
        assert_eq!(session.memory_plan().allocations().len(), 18);
        assert!(session.admission().admitted);
        session.reset().unwrap();
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
    fn capture_enabled_prefill_and_decode_logits_match_ordinary_forward_bitwise() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF").unwrap_or_else(|_| {
            "/Users/tito/models/muse-glimmer/Muse-Glimmer-30B-Q8_0.gguf".into()
        });
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
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF").unwrap_or_else(|_| {
            "/Users/tito/models/muse-glimmer/Muse-Glimmer-30B-Q8_0.gguf".into()
        });
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
