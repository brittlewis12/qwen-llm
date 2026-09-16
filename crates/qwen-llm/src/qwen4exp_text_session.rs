//! Full one-token text decode session for Qwen3.8-Flash-Next.

use crate::metal::{
    KernelEncoder, MetalBufferSizeAndAlign, MetalContext, MetalError, MetalMemoryAdmission,
    MetalMemorySignals, MetalTensor, MetalTensorProvenance, MetalTimestampSampleBuffer,
    encode_axpy_f32, encode_copy_offset_f32, encode_get_rows_f32, evaluate_metal_memory_admission,
    host_page_size_bytes,
};
use crate::metal_forward::{MfError, encode_mat_vec_dispatch};
use crate::qwen4exp::{MixerKind, PleHistory, Qwen4ExpConfig, Qwen4ExpError};
use crate::qwen4exp_gdn::{
    GatedDeltaNetMetalGeometry, GatedDeltaNetPackedScratch, Qwen4ExpGdnError,
};
use crate::qwen4exp_layers_zero_one::{
    Qwen4ExpLayersZeroOneError, Qwen4ExpLayersZeroOneMetalGeometry,
    Qwen4ExpLayersZeroOneMetalWeights, Qwen4ExpLayersZeroOneMetalWorkspace,
    encode_qwen4exp_layers_zero_one_packed_staged, encode_qwen4exp_layers_zero_one_staged,
};
use crate::qwen4exp_metal::{
    GatedResidualMetalReadWeights, GatedResidualMetalScratch, GatedResidualPackedScratch,
    Qwen4ExpMetalError, encode_hc_repeat_packed, validate_and_preflight_final_gated_residual_mix,
    validate_and_preflight_hc_repeat_packed,
};
use crate::qwen4exp_moe::{
    Qwen4ExpMoeError, Qwen4ExpMoeMetalGeometry, Qwen4ExpMoePackedMotorScratch,
};
use crate::qwen4exp_ple::{PleGatherError, PleIq4NlTable};
use crate::qwen4exp_ple_metal::{Qwen4ExpPleMetalError, Qwen4ExpPlePackedMotorScratch};
use crate::qwen4exp_post_ple_block::{
    Qwen4ExpPostPleBlockError, Qwen4ExpPostPleBlockMetalGeometry, Qwen4ExpPostPleBlockMetalWeights,
    Qwen4ExpPostPleBlockMetalWorkspace, Qwen4ExpPostPleMixerMetalGeometry,
    encode_qwen4exp_post_ple_block, encode_qwen4exp_post_ple_block_packed,
    encode_qwen4exp_post_ple_block_packed_profiled,
    encode_qwen4exp_post_ple_block_packed_stage_sampled,
};
use crate::qwen4exp_profile::{
    Qwen4ExpPackedProfileLabel, Qwen4ExpPackedProfileRecorder, Qwen4ExpPackedProfileSpan,
    begin_optional, end_optional, is_stage_profiled_layer, packed_stage_sample_count,
    packed_stage_span_count, stage_profile_count,
};
use crate::qwen4exp_qsa::{
    Qwen4ExpQsaError, QwenSparseAttentionMetalGeometry, QwenSparseAttentionPackedScratch,
};
use crate::qwen4exp_residency::{
    Qwen4ExpMetalWeightMemoryPlan, Qwen4ExpMetalWeights, Qwen4ExpResidencyError,
};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLComputePipelineState, MTLDevice,
    MTLResource,
};
use std::fmt;
use std::mem::size_of;
use std::time::Instant;

#[cfg(test)]
mod checkpoint;

#[inline(always)]
fn with_diagnostic_execution_range<R>(
    start_position: usize,
    rows: usize,
    f: impl FnOnce() -> R,
) -> R {
    #[cfg(test)]
    {
        return crate::qwen4exp_composition_trace::with_qwen4exp_diagnostic_execution_range(
            start_position,
            rows,
            f,
        );
    }
    #[cfg(not(test))]
    {
        let _ = (start_position, rows);
        f()
    }
}

#[cfg(test)]
fn reject_active_sampled_diagnostics(kind: &str) -> Result<(), Qwen4ExpTextSessionError> {
    if crate::qwen4exp_composition_trace::qwen4exp_composition_trace_active() {
        return invalid(format!(
            "composition tracing is unavailable in sampled {kind} profiles"
        ));
    }
    if crate::qwen4exp_qsa::qwen4exp_qsa_decision_capture_active() {
        return invalid(format!(
            "QSA decision capture is unavailable in sampled {kind} profiles"
        ));
    }
    if crate::qwen4exp_metal::qwen4exp_hc_packed_projection_override_active() {
        return invalid(format!(
            "HC projection overrides are unavailable in sampled {kind} profiles"
        ));
    }
    Ok(())
}

pub const QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpTextSessionError {
    #[error(transparent)]
    Config(#[from] Qwen4ExpError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Forward(#[from] MfError),
    #[error(transparent)]
    Residual(#[from] Qwen4ExpMetalError),
    #[error(transparent)]
    LayersZeroOne(#[from] Qwen4ExpLayersZeroOneError),
    #[error(transparent)]
    PostPleBlock(#[from] Qwen4ExpPostPleBlockError),
    #[error(transparent)]
    GatedDeltaNet(#[from] Qwen4ExpGdnError),
    #[error(transparent)]
    Moe(#[from] Qwen4ExpMoeError),
    #[error(transparent)]
    Ple(#[from] Qwen4ExpPleMetalError),
    #[error(transparent)]
    PleGather(#[from] PleGatherError),
    #[error(transparent)]
    QwenSparseAttention(#[from] Qwen4ExpQsaError),
    #[error(transparent)]
    Residency(#[from] Qwen4ExpResidencyError),
    #[error("invalid Qwen3.8-Flash-Next text-session contract: {0}")]
    Invalid(String),
    #[error("Qwen3.8-Flash-Next text-session command buffer failed: {0}")]
    CommandBuffer(String),
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Qwen4ExpPackedEncodeCpuTiming {
    pub preflight_ms: f64,
    pub stage_inputs_ms: f64,
    pub graph_encode_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Qwen4ExpTextSessionReleaseTiming {
    pub root_wait_ms: f64,
    pub child_publication_ms: f64,
    pub root_publish_ms: f64,
    pub release_total_ms: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4ExpTextSessionMetalGeometry {
    zero_one: Qwen4ExpLayersZeroOneMetalGeometry,
    post_ple: Vec<Qwen4ExpPostPleBlockMetalGeometry>,
    qsa_layers: Vec<u32>,
    capacity: usize,
    vocab_size: usize,
}

impl Qwen4ExpTextSessionMetalGeometry {
    pub fn from_config(
        config: &Qwen4ExpConfig,
        capacity: usize,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        config.validate()?;
        if config.layer_count < 2 {
            return invalid("text session requires at least two layers");
        }
        let zero_one = Qwen4ExpLayersZeroOneMetalGeometry::from_config(config)?;
        let mut post_ple = Vec::with_capacity(config.layer_count.saturating_sub(2) as usize);
        let mut qsa_layers = Vec::new();
        for layer in 2..config.layer_count {
            let qsa_capacity = match config.mixer_kind(layer) {
                Some(MixerKind::GatedDeltaNet) => None,
                Some(MixerKind::QwenSparseAttention) => {
                    qsa_layers.push(layer);
                    Some(capacity)
                }
                None => return invalid(format!("layer {layer} has no mixer schedule entry")),
            };
            post_ple.push(Qwen4ExpPostPleBlockMetalGeometry::from_config(
                config,
                layer,
                qsa_capacity,
            )?);
        }
        let geometry = Self {
            zero_one,
            post_ple,
            qsa_layers,
            capacity,
            vocab_size: config.vocab_size as usize,
        };
        geometry.validate(config.layer_count as usize)?;
        Ok(geometry)
    }

    fn validate(&self, layer_count: usize) -> Result<(), Qwen4ExpTextSessionError> {
        if self.capacity == 0 || self.vocab_size == 0 {
            return invalid("session capacity and vocabulary size must be nonzero");
        }
        if self.post_ple.len() != layer_count.saturating_sub(2) {
            return invalid("post-PLE block count differs from the model layer count");
        }
        for (offset, block) in self.post_ple.iter().enumerate() {
            let expected_layer = offset as u32 + 2;
            if block.layer() != expected_layer
                || block.branch_count() != self.zero_one.branch_count()
                || block.hidden_size() != self.zero_one.hidden_size()
                || block.low_rank() != self.zero_one.low_rank()
                || block.eps().to_bits() != self.zero_one.eps().to_bits()
            {
                return invalid(format!(
                    "post-PLE block {expected_layer} differs from session geometry"
                ));
            }
            if let Some(qsa) = block.mixer().qsa()
                && qsa.capacity() != self.capacity
            {
                return invalid(format!(
                    "QSA layer {expected_layer} capacity {} differs from session {}",
                    qsa.capacity(),
                    self.capacity
                ));
            }
        }
        let observed_qsa = self
            .post_ple
            .iter()
            .filter(|block| block.mixer().kind() == MixerKind::QwenSparseAttention)
            .map(|block| block.layer())
            .collect::<Vec<_>>();
        if observed_qsa != self.qsa_layers {
            return invalid("QSA layer index does not match the post-PLE schedule");
        }
        if self.qsa_layers.is_empty() {
            return invalid("text session requires at least one QSA layer");
        }
        for (name, value) in [
            ("capacity", self.capacity),
            ("vocabulary size", self.vocab_size),
            ("hidden size", self.hidden_size()),
            ("hyper width", self.hyper_width()),
        ] {
            if u32::try_from(value).is_err() {
                return invalid(format!("{name} {value} exceeds u32"));
            }
        }
        Ok(())
    }

    pub fn zero_one(&self) -> Qwen4ExpLayersZeroOneMetalGeometry {
        self.zero_one
    }

    pub fn post_ple(&self) -> &[Qwen4ExpPostPleBlockMetalGeometry] {
        &self.post_ple
    }

    pub fn qsa_layers(&self) -> &[u32] {
        &self.qsa_layers
    }

    pub fn layer_count(&self) -> usize {
        self.post_ple.len() + 2
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn branch_count(&self) -> usize {
        self.zero_one.branch_count()
    }

    pub fn hidden_size(&self) -> usize {
        self.zero_one.hidden_size()
    }

    pub fn low_rank(&self) -> usize {
        self.zero_one.low_rank()
    }

    pub fn hyper_width(&self) -> usize {
        self.zero_one.hyper_width()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn eps(&self) -> f32 {
        self.zero_one.eps()
    }

    fn packed_qsa_geometry(
        &self,
    ) -> Result<QwenSparseAttentionMetalGeometry, Qwen4ExpTextSessionError> {
        let geometry = self
            .post_ple
            .iter()
            .find_map(|block| block.mixer().qsa())
            .ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid(
                    "text session has no QSA geometry for packed scratch".into(),
                )
            })?;
        if self
            .post_ple
            .iter()
            .filter_map(|block| block.mixer().qsa())
            .any(|candidate| candidate != geometry)
        {
            return invalid("QSA layers require distinct packed scratch geometries");
        }
        Ok(geometry)
    }

    fn packed_capacity(&self) -> Result<usize, Qwen4ExpTextSessionError> {
        Ok(self
            .capacity
            .min(self.packed_qsa_geometry()?.token_budget()))
    }

    fn packed_selected_capable_for_extent(
        &self,
        prompt_extent: usize,
    ) -> Result<bool, Qwen4ExpTextSessionError> {
        if prompt_extent > self.capacity {
            return invalid(format!(
                "packed prompt extent {prompt_extent} exceeds session capacity {}",
                self.capacity
            ));
        }
        Ok(prompt_extent > self.packed_qsa_geometry()?.output_width())
    }
}

pub struct Qwen4ExpTextSessionMetalWeights<'a> {
    pub geometry: Qwen4ExpTextSessionMetalGeometry,
    pub zero_one: Qwen4ExpLayersZeroOneMetalWeights<'a>,
    pub post_ple: Vec<Qwen4ExpPostPleBlockMetalWeights<'a>>,
    pub final_read: GatedResidualMetalReadWeights<'a>,
    pub output: &'a MetalTensor,
}

impl<'a> Qwen4ExpTextSessionMetalWeights<'a> {
    pub fn bind(
        weights: &'a Qwen4ExpMetalWeights,
        capacity: usize,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let geometry = Qwen4ExpTextSessionMetalGeometry::from_config(weights.config(), capacity)?;
        let mut post_ple = Vec::with_capacity(geometry.post_ple.len());
        for block in &geometry.post_ple {
            post_ple.push(Qwen4ExpPostPleBlockMetalWeights::bind(
                weights,
                block.layer(),
                block.mixer().qsa().map(|qsa| qsa.capacity()),
            )?);
        }
        Ok(Self {
            geometry,
            zero_one: Qwen4ExpLayersZeroOneMetalWeights::bind(weights)?,
            post_ple,
            final_read: GatedResidualMetalReadWeights {
                norm: weights.require_tensor("output_hc_norm.weight")?,
                down: weights.require_tensor("output_hc_down.weight")?,
                up: weights.require_tensor("output_hc_up.weight")?,
            },
            output: weights.require_tensor("output.weight")?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen4ExpTextSessionAllocation {
    pub name: String,
    pub logical_bytes: u64,
    pub priced_bytes: u64,
    pub alignment: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen4ExpTextSessionMemoryPlan {
    residency_priced_upper_bytes: u64,
    session_logical_bytes: u64,
    session_priced_upper_bytes: u64,
    packed_capacity: Option<usize>,
    packed_selected_capable: bool,
    split_decode: bool,
    allocations: Vec<Qwen4ExpTextSessionAllocation>,
}

impl Qwen4ExpTextSessionMemoryPlan {
    pub fn for_geometry(
        ctx: &MetalContext,
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        residency: &Qwen4ExpMetalWeightMemoryPlan,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        Self::for_geometry_with_residency_bytes(ctx, geometry, residency.priced_upper_bytes())
    }

    /// Reserve the maximum packed plan implied by the full session geometry.
    /// Use the prompt-aware plan when the actual prompt extent is known.
    pub fn for_geometry_with_packed_prefill(
        ctx: &MetalContext,
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        residency: &Qwen4ExpMetalWeightMemoryPlan,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let packed_selected_capable =
            geometry.packed_selected_capable_for_extent(geometry.capacity())?;
        Self::for_geometry_with_options(
            ctx,
            geometry,
            residency.priced_upper_bytes(),
            Some(geometry.packed_capacity()?),
            packed_selected_capable,
        )
    }

    fn for_geometry_with_residency_bytes(
        ctx: &MetalContext,
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        residency_priced_upper_bytes: u64,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        Self::for_geometry_with_options(ctx, geometry, residency_priced_upper_bytes, None, false)
    }

    #[cfg(test)]
    fn for_geometry_with_packed_residency_bytes(
        ctx: &MetalContext,
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        residency_priced_upper_bytes: u64,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let packed_selected_capable =
            geometry.packed_selected_capable_for_extent(geometry.capacity())?;
        Self::for_geometry_with_options(
            ctx,
            geometry,
            residency_priced_upper_bytes,
            Some(geometry.packed_capacity()?),
            packed_selected_capable,
        )
    }

    fn for_geometry_with_options(
        ctx: &MetalContext,
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        residency_priced_upper_bytes: u64,
        packed_capacity: Option<usize>,
        packed_selected_capable: bool,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        Self::for_geometry_with_split_options(
            ctx,
            geometry,
            residency_priced_upper_bytes,
            packed_capacity,
            packed_selected_capable,
            false,
        )
    }

    fn for_geometry_with_split_options(
        ctx: &MetalContext,
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        residency_priced_upper_bytes: u64,
        packed_capacity: Option<usize>,
        packed_selected_capable: bool,
        split_decode: bool,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        if packed_capacity.is_none() && packed_selected_capable {
            return invalid("selected packed QSA scratch requires packed prefill");
        }
        let mut builder = AllocationBuilder::default();
        if split_decode {
            if !geometry.packed_qsa_geometry()?.supports_split_decode() {
                return invalid("split decode requires released QSA geometry");
            }
            crate::qwen4exp_qsa::split_decode::preflight(ctx)?;
            builder.f32(
                "session.qsa_split",
                crate::qwen4exp_qsa::split_decode::SCRATCH_FLOATS,
            )?;
        }
        add_zero_one_allocations(&mut builder, geometry.zero_one)?;
        if let Some(capacity) = packed_capacity {
            let maximum = geometry.packed_capacity()?;
            if !(2..=maximum).contains(&capacity) {
                return invalid(format!(
                    "packed prefill capacity {capacity} is outside 2..={maximum}"
                ));
            }
            add_packed_allocations(&mut builder, geometry, capacity, packed_selected_capable)?;
        }
        builder.f32("session.hyper_residual", geometry.hyper_width())?;
        for block in &geometry.post_ple {
            add_post_ple_allocations(&mut builder, *block)?;
        }
        add_residual_allocations(
            &mut builder,
            "session.final_read",
            geometry.branch_count(),
            geometry.hidden_size(),
            geometry.low_rank(),
        )?;
        builder.f32("session.logits", geometry.vocab_size())?;
        let mut allocations = Vec::with_capacity(builder.allocations.len());
        let mut session_logical_bytes = 0_u64;
        let mut session_priced_upper_bytes = 0_u64;
        let host_page_size = host_page_size_bytes()? as u64;
        let max_buffer_length = u64::try_from(ctx.max_buffer_length()).map_err(|_| {
            Qwen4ExpTextSessionError::Invalid("Metal maximum buffer length exceeds u64".into())
        })?;
        for (name, logical_bytes) in builder.allocations {
            let priced = ctx.shared_buffer_size_and_align(logical_bytes)?;
            let (priced_upper_bytes, alignment) = price_session_allocation(
                &name,
                logical_bytes,
                priced,
                host_page_size,
                max_buffer_length,
            )?;
            session_logical_bytes = session_logical_bytes
                .checked_add(logical_bytes)
                .ok_or_else(|| {
                    Qwen4ExpTextSessionError::Invalid("session logical byte total overflow".into())
                })?;
            session_priced_upper_bytes = session_priced_upper_bytes
                .checked_add(priced_upper_bytes)
                .ok_or_else(|| {
                    Qwen4ExpTextSessionError::Invalid("session priced byte total overflow".into())
                })?;
            allocations.push(Qwen4ExpTextSessionAllocation {
                name,
                logical_bytes,
                priced_bytes: priced_upper_bytes,
                alignment,
            });
        }
        Ok(Self {
            residency_priced_upper_bytes,
            session_logical_bytes,
            session_priced_upper_bytes,
            packed_capacity,
            packed_selected_capable,
            split_decode,
            allocations,
        })
    }

    pub fn residency_priced_upper_bytes(&self) -> u64 {
        self.residency_priced_upper_bytes
    }

    pub fn session_logical_bytes(&self) -> u64 {
        self.session_logical_bytes
    }

    pub fn session_priced_upper_bytes(&self) -> u64 {
        self.session_priced_upper_bytes
    }

    pub fn allocations(&self) -> &[Qwen4ExpTextSessionAllocation] {
        &self.allocations
    }

    pub fn packed_prefill_capacity(&self) -> Option<usize> {
        self.packed_capacity
    }

    pub fn packed_selected_capable(&self) -> bool {
        self.packed_selected_capable
    }

    pub fn split_decode_enabled(&self) -> bool {
        self.split_decode
    }

    pub fn priced_upper_bytes_for_sessions(
        &self,
        session_count: usize,
    ) -> Result<u64, Qwen4ExpTextSessionError> {
        if session_count == 0 {
            return invalid("memory admission requires at least one session");
        }
        let count = u64::try_from(session_count)
            .map_err(|_| Qwen4ExpTextSessionError::Invalid("session count exceeds u64".into()))?;
        self.session_priced_upper_bytes
            .checked_mul(count)
            .and_then(|sessions| self.residency_priced_upper_bytes.checked_add(sessions))
            .ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid(
                    "residency plus session priced byte total overflow".into(),
                )
            })
    }

    pub fn admission_before_residency(
        &self,
        signals: MetalMemorySignals,
        session_count: usize,
    ) -> Result<MetalMemoryAdmission, Qwen4ExpTextSessionError> {
        Ok(evaluate_metal_memory_admission(
            self.priced_upper_bytes_for_sessions(session_count)?,
            QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
            signals,
            true,
        ))
    }

    pub fn admission_after_residency(&self, signals: MetalMemorySignals) -> MetalMemoryAdmission {
        evaluate_metal_memory_admission(
            self.session_priced_upper_bytes,
            QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
            signals,
            true,
        )
    }

    pub fn reconcile_session(
        &self,
        allocated_before: u64,
        allocated_after: u64,
    ) -> Result<u64, Qwen4ExpTextSessionError> {
        let observed = allocated_after.saturating_sub(allocated_before);
        if observed > self.session_priced_upper_bytes {
            return invalid(format!(
                "observed session allocation {observed} exceeds priced upper bound {}",
                self.session_priced_upper_bytes
            ));
        }
        Ok(observed)
    }
}

impl fmt::Display for Qwen4ExpTextSessionMemoryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "residency_priced={} session_allocations={} session_logical={} session_priced={} packed_capacity={:?} packed_selected_capable={} reserve={}",
            self.residency_priced_upper_bytes,
            self.allocations.len(),
            self.session_logical_bytes,
            self.session_priced_upper_bytes,
            self.packed_capacity,
            self.packed_selected_capable,
            QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES
        )
    }
}

pub struct Qwen4ExpTextSessionPlan {
    geometry: Qwen4ExpTextSessionMetalGeometry,
    memory: Qwen4ExpTextSessionMemoryPlan,
    device_registry_id: u64,
}

impl Qwen4ExpTextSessionPlan {
    pub fn with_split_decode(
        mut self,
        ctx: &MetalContext,
        enabled: bool,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        if self.device_registry_id != ctx.device.registryID() {
            return invalid("split decode plan device mismatch");
        }
        if self.memory.split_decode_enabled() == enabled {
            return Ok(self);
        }
        self.memory = Qwen4ExpTextSessionMemoryPlan::for_geometry_with_split_options(
            ctx,
            &self.geometry,
            self.memory.residency_priced_upper_bytes,
            self.memory.packed_capacity,
            self.memory.packed_selected_capable,
            enabled,
        )?;
        Ok(self)
    }
    pub fn for_config(
        ctx: &MetalContext,
        config: &Qwen4ExpConfig,
        capacity: usize,
        residency: &Qwen4ExpMetalWeightMemoryPlan,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        Self::for_config_with_options(ctx, config, capacity, residency, None, false)
    }

    /// Reserve packed scratch against the full session geometry when no prompt
    /// extent is available.
    pub fn for_config_with_packed_prefill(
        ctx: &MetalContext,
        config: &Qwen4ExpConfig,
        capacity: usize,
        residency: &Qwen4ExpMetalWeightMemoryPlan,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let geometry = Qwen4ExpTextSessionMetalGeometry::from_config(config, capacity)?;
        let packed_capacity = geometry.packed_capacity()?;
        let packed_selected_capable =
            geometry.packed_selected_capable_for_extent(geometry.capacity())?;
        Self::from_geometry(
            ctx,
            geometry,
            residency,
            Some(packed_capacity),
            packed_selected_capable,
        )
    }

    /// Reserve an explicit dense-only packed capacity. Selected-range callers
    /// must use the prompt-aware constructor.
    pub fn for_config_with_packed_prefill_capacity(
        ctx: &MetalContext,
        config: &Qwen4ExpConfig,
        capacity: usize,
        residency: &Qwen4ExpMetalWeightMemoryPlan,
        packed_capacity: usize,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        Self::for_config_with_options(
            ctx,
            config,
            capacity,
            residency,
            Some(packed_capacity),
            false,
        )
    }

    /// Reserve reusable packed rows from prompt length and add selected-range
    /// scratch only when the full prompt crosses the QSA dense limit.
    pub fn for_config_with_packed_prefill_tokens(
        ctx: &MetalContext,
        config: &Qwen4ExpConfig,
        capacity: usize,
        residency: &Qwen4ExpMetalWeightMemoryPlan,
        prompt_tokens: usize,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let geometry = Qwen4ExpTextSessionMetalGeometry::from_config(config, capacity)?;
        let (packed_capacity, packed_selected_capable) =
            Self::packed_prefill_options_for_prompt(&geometry, prompt_tokens)?;
        Self::from_geometry(
            ctx,
            geometry,
            residency,
            Some(packed_capacity),
            packed_selected_capable,
        )
    }

    fn packed_prefill_options_for_prompt(
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        prompt_tokens: usize,
    ) -> Result<(usize, bool), Qwen4ExpTextSessionError> {
        Ok((
            prompt_tokens.min(geometry.packed_capacity()?),
            geometry.packed_selected_capable_for_extent(prompt_tokens)?,
        ))
    }

    fn for_config_with_options(
        ctx: &MetalContext,
        config: &Qwen4ExpConfig,
        capacity: usize,
        residency: &Qwen4ExpMetalWeightMemoryPlan,
        packed_capacity: Option<usize>,
        packed_selected_capable: bool,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let geometry = Qwen4ExpTextSessionMetalGeometry::from_config(config, capacity)?;
        Self::from_geometry(
            ctx,
            geometry,
            residency,
            packed_capacity,
            packed_selected_capable,
        )
    }

    fn from_geometry(
        ctx: &MetalContext,
        geometry: Qwen4ExpTextSessionMetalGeometry,
        residency: &Qwen4ExpMetalWeightMemoryPlan,
        packed_capacity: Option<usize>,
        packed_selected_capable: bool,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let memory = Qwen4ExpTextSessionMemoryPlan::for_geometry_with_options(
            ctx,
            &geometry,
            residency.priced_upper_bytes(),
            packed_capacity,
            packed_selected_capable,
        )?;
        Ok(Self {
            geometry,
            memory,
            device_registry_id: ctx.device.registryID(),
        })
    }

    pub fn geometry(&self) -> &Qwen4ExpTextSessionMetalGeometry {
        &self.geometry
    }

    pub fn memory_plan(&self) -> &Qwen4ExpTextSessionMemoryPlan {
        &self.memory
    }

    pub fn admit_after_residency(
        self,
        weights: &Qwen4ExpMetalWeights,
        signals: MetalMemorySignals,
    ) -> Result<Qwen4ExpAdmittedTextSessionPlan, Qwen4ExpTextSessionError> {
        if weights.device_registry_id() != self.device_registry_id {
            return invalid(format!(
                "resident weights belong to Metal device registry {}, session plan is for {}",
                weights.device_registry_id(),
                self.device_registry_id
            ));
        }
        if Qwen4ExpTextSessionMetalGeometry::from_config(weights.config(), self.geometry.capacity)?
            != self.geometry
        {
            return invalid("resident weights differ from the text-session geometry");
        }
        if weights.memory_plan().priced_upper_bytes() != self.memory.residency_priced_upper_bytes {
            return invalid("resident weights differ from the session residency memory plan");
        }
        let admission = self.memory.admission_after_residency(signals);
        if !admission.admitted {
            return invalid(format!(
                "text-session memory admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                admission.reason.as_str(),
                admission.required_bytes,
                admission.working_set_headroom_bytes,
                admission.signals.process_limit_remaining_bytes
            ));
        }
        Ok(Qwen4ExpAdmittedTextSessionPlan {
            plan: self,
            admission,
        })
    }
}

pub struct Qwen4ExpAdmittedTextSessionPlan {
    plan: Qwen4ExpTextSessionPlan,
    admission: MetalMemoryAdmission,
}

impl Qwen4ExpAdmittedTextSessionPlan {
    pub fn admission(&self) -> MetalMemoryAdmission {
        self.admission
    }

    pub fn memory_plan(&self) -> &Qwen4ExpTextSessionMemoryPlan {
        &self.plan.memory
    }
}

struct Qwen4ExpTextPackedScratch {
    capacity: usize,
    token_ids: MetalTensor,
    embedding: MetalTensor,
    ple_packed_rows: MetalTensor,
    ple_local_row_ids: MetalTensor,
    ple_embedding: MetalTensor,
    hyper_residual: MetalTensor,
    bridge: MetalTensor,
    residual: GatedResidualPackedScratch,
    gdn: GatedDeltaNetPackedScratch,
    ple: Qwen4ExpPlePackedMotorScratch,
    moe: Qwen4ExpMoePackedMotorScratch,
    qsa: QwenSparseAttentionPackedScratch,
}

struct Qwen4ExpTextPackedViews {
    token_ids: MetalTensor,
    embedding: MetalTensor,
    ple_packed_rows: MetalTensor,
    ple_local_row_ids: MetalTensor,
    ple_embedding: MetalTensor,
    hyper_residual: MetalTensor,
    bridge: MetalTensor,
}

impl Qwen4ExpTextPackedScratch {
    fn new(
        ctx: &MetalContext,
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        capacity: usize,
        selected_capable: bool,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let maximum = geometry.packed_capacity()?;
        if !(2..=maximum).contains(&capacity) {
            return invalid(format!(
                "packed scratch capacity {capacity} is outside 2..={maximum}"
            ));
        }
        let zero_one = geometry.zero_one();
        let hidden = geometry.hidden_size();
        let hyper = geometry.hyper_width();
        let ple_geometry = zero_one.ple();
        let lookup_capacity = ple_geometry
            .head_count()
            .checked_mul(capacity)
            .ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid("packed PLE lookup capacity overflow".into())
            })?;
        let packed_bytes = ple_geometry
            .packed_staging_bytes()
            .checked_mul(capacity)
            .ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid("packed PLE staging byte count overflow".into())
            })?;
        let local_row_ids = (0..lookup_capacity)
            .map(|row| {
                i32::try_from(row).map_err(|_| {
                    Qwen4ExpTextSessionError::Invalid("packed PLE local row ID exceeds i32".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let qsa_geometry = geometry.packed_qsa_geometry()?;
        let qsa = if selected_capable {
            QwenSparseAttentionPackedScratch::new_with_selected_capability(
                ctx,
                qsa_geometry,
                capacity,
                true,
            )?
        } else {
            QwenSparseAttentionPackedScratch::new(ctx, qsa_geometry, capacity)?
        };
        let shape = |width: usize| vec![width as u64, capacity as u64];
        Ok(Self {
            capacity,
            token_ids: MetalTensor::zeros_i32(ctx, vec![capacity as u64])?,
            embedding: MetalTensor::zeros_f32(ctx, shape(hidden))?,
            ple_packed_rows: MetalTensor::from_bytes(
                ctx,
                &vec![0_u8; packed_bytes],
                vec![ple_geometry.head_dim() as u64, lookup_capacity as u64],
                GgmlType::IQ4_NL,
            )?,
            ple_local_row_ids: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&local_row_ids),
                vec![lookup_capacity as u64],
                GgmlType::I32,
            )?,
            ple_embedding: MetalTensor::zeros_f32(ctx, shape(hidden))?,
            hyper_residual: MetalTensor::zeros_f32(ctx, shape(hyper))?,
            bridge: MetalTensor::zeros_f32(ctx, shape(hidden))?,
            residual: GatedResidualPackedScratch::new(
                ctx,
                geometry.branch_count(),
                hidden,
                geometry.low_rank(),
                capacity,
            )?,
            gdn: GatedDeltaNetPackedScratch::new(ctx, zero_one.layer_zero().gdn(), capacity)?,
            ple: Qwen4ExpPlePackedMotorScratch::new(ctx, ple_geometry, capacity)?,
            moe: Qwen4ExpMoePackedMotorScratch::new(ctx, zero_one.layer_zero().moe(), capacity)?,
            qsa,
        })
    }

    fn views(
        &self,
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        tokens: usize,
    ) -> Result<Qwen4ExpTextPackedViews, Qwen4ExpTextSessionError> {
        if tokens <= 1 || tokens > self.capacity {
            return invalid(format!(
                "packed text token count {tokens} is outside 2..={}",
                self.capacity
            ));
        }
        let ple = geometry.zero_one().ple();
        let lookups = ple.head_count().checked_mul(tokens).ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("packed PLE lookup count overflow".into())
        })?;
        Ok(Qwen4ExpTextPackedViews {
            token_ids: self.token_ids.view_subrange(0, vec![tokens as u64]),
            embedding: packed_prefix_view(
                "packed token embedding",
                &self.embedding,
                geometry.hidden_size(),
                tokens,
                self.capacity,
            )?,
            ple_packed_rows: self.ple_packed_rows.clone(),
            ple_local_row_ids: self
                .ple_local_row_ids
                .view_subrange(0, vec![lookups as u64]),
            ple_embedding: packed_prefix_view(
                "packed PLE embedding",
                &self.ple_embedding,
                geometry.hidden_size(),
                tokens,
                self.capacity,
            )?,
            hyper_residual: packed_prefix_view(
                "packed hyper residual",
                &self.hyper_residual,
                geometry.hyper_width(),
                tokens,
                self.capacity,
            )?,
            bridge: packed_prefix_view(
                "packed block bridge",
                &self.bridge,
                geometry.hidden_size(),
                tokens,
                self.capacity,
            )?,
        })
    }
}

#[cfg(test)]
use crate::qwen4exp_metal::encode_final_gated_residual_mix;

pub struct Qwen4ExpTextSessionMetalWorkspace {
    geometry: Qwen4ExpTextSessionMetalGeometry,
    split_decode_scratch: Option<MetalTensor>,
    zero_one: Qwen4ExpLayersZeroOneMetalWorkspace,
    packed: Option<Qwen4ExpTextPackedScratch>,
    hyper_residual: MetalTensor,
    post_ple: Vec<Qwen4ExpPostPleBlockMetalWorkspace>,
    final_read: GatedResidualMetalScratch,
    logits: MetalTensor,
    memory: Qwen4ExpTextSessionMemoryPlan,
    admission: MetalMemoryAdmission,
    observed_allocation_delta: u64,
    committed_length: usize,
    pending_length: Option<usize>,
    active_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    state_poisoned: bool,
    encode_failed: bool,
    logits_ready: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct Qwen4ExpFixedHyperAddTensor<'a> {
    pub direction: &'a MetalTensor,
    pub coefficient: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Qwen4ExpPostLayerHyperProbe<'a> {
    pub layer: u32,
    pub capture: &'a MetalTensor,
    pub fixed_add: Option<Qwen4ExpFixedHyperAddTensor<'a>>,
}

impl Qwen4ExpTextSessionMetalWorkspace {
    pub fn from_admitted(
        ctx: &MetalContext,
        admitted: Qwen4ExpAdmittedTextSessionPlan,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        if admitted.plan.device_registry_id != ctx.device.registryID() {
            return invalid(format!(
                "session plan belongs to Metal device registry {}, context is {}",
                admitted.plan.device_registry_id,
                ctx.device.registryID()
            ));
        }
        let _allocation_transaction = ctx.begin_allocation_transaction();
        let refreshed = admitted
            .plan
            .memory
            .admission_after_residency(ctx.memory_signals());
        if !refreshed.admitted {
            return invalid(format!(
                "text-session memory admission changed before allocation: reason={} required={:?}",
                refreshed.reason.as_str(),
                refreshed.required_bytes
            ));
        }
        Self::allocate(ctx, admitted.plan.geometry, admitted.plan.memory, refreshed)
    }

    fn allocate(
        ctx: &MetalContext,
        geometry: Qwen4ExpTextSessionMetalGeometry,
        memory: Qwen4ExpTextSessionMemoryPlan,
        admission: MetalMemoryAdmission,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let allocated_before = ctx.current_allocated_size();
        let split_decode_scratch = if memory.split_decode_enabled() {
            Some(MetalTensor::zeros_f32(
                ctx,
                vec![crate::qwen4exp_qsa::split_decode::SCRATCH_FLOATS as u64],
            )?)
        } else {
            None
        };
        let mut post_ple = Vec::with_capacity(geometry.post_ple.len());
        for block in &geometry.post_ple {
            post_ple.push(Qwen4ExpPostPleBlockMetalWorkspace::new(ctx, *block)?);
        }
        for block in &post_ple {
            block.validate_split_binding(ctx, split_decode_scratch.as_ref())?;
        }
        for block in &mut post_ple {
            block.bind_split_scratch(split_decode_scratch.as_ref());
        }
        let packed = if let Some(planned_capacity) = memory.packed_prefill_capacity() {
            let scratch = Qwen4ExpTextPackedScratch::new(
                ctx,
                &geometry,
                planned_capacity,
                memory.packed_selected_capable(),
            )?;
            if scratch.capacity != planned_capacity {
                return invalid(format!(
                    "packed scratch capacity {} differs from admitted {planned_capacity}",
                    scratch.capacity
                ));
            }
            if scratch.qsa.selected_capable() != memory.packed_selected_capable() {
                return invalid("packed selected QSA scratch differs from admitted plan");
            }
            Some(scratch)
        } else {
            None
        };
        let workspace = Self {
            split_decode_scratch,
            zero_one: Qwen4ExpLayersZeroOneMetalWorkspace::new(ctx, geometry.zero_one)?,
            packed,
            hyper_residual: MetalTensor::zeros_f32(ctx, vec![geometry.hyper_width() as u64])?,
            final_read: GatedResidualMetalScratch::new(
                ctx,
                geometry.branch_count(),
                geometry.hidden_size(),
                geometry.low_rank(),
            )?,
            logits: MetalTensor::zeros_f32(ctx, vec![geometry.vocab_size() as u64])?,
            geometry,
            post_ple,
            memory,
            admission,
            observed_allocation_delta: 0,
            committed_length: 0,
            pending_length: None,
            active_command: None,
            state_poisoned: false,
            encode_failed: false,
            logits_ready: false,
        };
        let allocated_after = ctx.current_allocated_size();
        let observed = workspace
            .memory
            .reconcile_session(allocated_before, allocated_after)?;
        Ok(Self {
            observed_allocation_delta: observed,
            ..workspace
        })
    }

    #[cfg(test)]
    pub(crate) fn new_for_tests(
        ctx: &MetalContext,
        geometry: Qwen4ExpTextSessionMetalGeometry,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        Self::new_for_tests_with_options(ctx, geometry, None)
    }

    #[cfg(test)]
    pub(crate) fn new_for_tests_with_packed(
        ctx: &MetalContext,
        geometry: Qwen4ExpTextSessionMetalGeometry,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let selected_capable = geometry.packed_selected_capable_for_extent(geometry.capacity())?;
        Self::new_for_tests_with_options(ctx, geometry, Some(selected_capable))
    }

    #[cfg(test)]
    pub(crate) fn new_for_tests_with_dense_packed(
        ctx: &MetalContext,
        geometry: Qwen4ExpTextSessionMetalGeometry,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        Self::new_for_tests_with_options(ctx, geometry, Some(false))
    }

    #[cfg(test)]
    fn new_for_tests_with_options(
        ctx: &MetalContext,
        geometry: Qwen4ExpTextSessionMetalGeometry,
        packed_selected_capable: Option<bool>,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let _allocation_transaction = ctx.begin_allocation_transaction();
        let memory = match packed_selected_capable {
            Some(selected_capable) => Qwen4ExpTextSessionMemoryPlan::for_geometry_with_options(
                ctx,
                &geometry,
                0,
                Some(geometry.packed_capacity()?),
                selected_capable,
            )?,
            None => {
                Qwen4ExpTextSessionMemoryPlan::for_geometry_with_residency_bytes(ctx, &geometry, 0)?
            }
        };
        let admission = memory.admission_after_residency(MetalMemorySignals {
            recommended_max_bytes: u64::MAX,
            current_allocated_bytes: ctx.current_allocated_size(),
            process_limit_remaining_bytes: Some(u64::MAX),
        });
        if !admission.admitted {
            return invalid(format!(
                "test session memory admission denied: {}",
                admission.reason.as_str()
            ));
        }
        Self::allocate(ctx, geometry, memory, admission)
    }

    pub fn geometry(&self) -> &Qwen4ExpTextSessionMetalGeometry {
        &self.geometry
    }

    pub fn split_decode_enabled(&self) -> bool {
        self.split_decode_scratch.is_some()
    }

    pub fn hc_up_mix_enabled(&self) -> bool {
        self.final_read.hc_up_mix_enabled()
    }

    pub fn guarded_topk_enabled(&self) -> bool {
        self.zero_one.guarded_topk_enabled()
    }

    pub(crate) fn configure_guarded_topk(
        &mut self,
        ctx: &MetalContext,
        enabled: bool,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        self.require_idle()?;
        if self.state_poisoned || self.encode_failed || self.pending_length.is_some() {
            return invalid("top-k configuration requires a healthy released session");
        }
        self.zero_one.validate_topk_binding(ctx)?;
        for block in &self.post_ple {
            block.validate_topk_binding(ctx)?;
        }
        if enabled {
            crate::qwen4exp_moe::guarded_topk::preflight(ctx)?;
        }
        self.zero_one.bind_guarded_topk(enabled);
        for block in &mut self.post_ple {
            block.bind_guarded_topk(enabled);
        }
        Ok(())
    }

    pub(crate) fn configure_hc_up_mix(
        &mut self,
        ctx: &MetalContext,
        enabled: bool,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        self.require_idle()?;
        if self.state_poisoned || self.encode_failed || self.pending_length.is_some() {
            return invalid("HC configuration requires a healthy released session");
        }
        self.zero_one.validate_hc_up_binding(ctx)?;
        for block in &self.post_ple {
            block.validate_hc_up_binding(ctx)?;
        }
        self.final_read.validate_hc_up_binding(ctx)?;
        if enabled {
            crate::qwen4exp_metal::hc_up::preflight(ctx)?;
        }
        self.zero_one.bind_hc_up_mix(enabled);
        for block in &mut self.post_ple {
            block.bind_hc_up_mix(enabled);
        }
        self.final_read.bind_hc_up_mix(enabled);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn set_split_decode_for_tests(
        &mut self,
        ctx: &MetalContext,
        enabled: bool,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        self.require_idle()?;
        if self.state_poisoned || self.encode_failed || self.pending_length.is_some() {
            return invalid("split binding switch requires a healthy released session");
        }
        let scratch = if enabled {
            Some(self.split_decode_scratch.as_ref().ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid("split scratch was not admitted".into())
            })?)
        } else {
            None
        };
        for block in &self.post_ple {
            block.validate_split_binding(ctx, scratch)?;
        }
        for block in &mut self.post_ple {
            block.bind_split_scratch(scratch);
        }
        Ok(())
    }

    pub fn memory_plan(&self) -> &Qwen4ExpTextSessionMemoryPlan {
        &self.memory
    }

    pub fn admission(&self) -> MetalMemoryAdmission {
        self.admission
    }

    pub fn packed_prefill_capacity(&self) -> Option<usize> {
        self.memory.packed_prefill_capacity()
    }

    pub fn packed_selected_capable(&self) -> bool {
        self.memory.packed_selected_capable()
    }

    pub fn observed_allocation_delta(&self) -> u64 {
        self.observed_allocation_delta
    }

    pub fn committed_length(&self) -> usize {
        self.committed_length
    }

    pub fn qsa_committed_lengths(&self) -> Vec<(u32, usize)> {
        self.geometry
            .post_ple
            .iter()
            .zip(&self.post_ple)
            .filter_map(|(geometry, workspace)| {
                workspace
                    .mixer_committed_length()
                    .map(|length| (geometry.layer(), length))
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn persistent_state_tensors(&self) -> Vec<MetalTensor> {
        let mut tensors = self.zero_one.persistent_state_tensors();
        for block in &self.post_ple {
            tensors.extend(block.persistent_state_tensors());
        }
        tensors
    }

    #[cfg(test)]
    pub(crate) fn qsa_persistent_state_tensors(&self) -> Vec<(u32, Vec<MetalTensor>)> {
        self.geometry
            .post_ple
            .iter()
            .zip(&self.post_ple)
            .filter_map(|(geometry, workspace)| {
                (geometry.mixer().kind() == MixerKind::QwenSparseAttention)
                    .then(|| (geometry.layer(), workspace.persistent_state_tensors()))
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn mark_pending_encode_failed_for_tests(&mut self) {
        assert!(self.active_command.is_some());
        assert!(self.pending_length.is_some());
        self.encode_failed = true;
        self.state_poisoned = true;
    }

    pub fn ple_prior_tokens(&self) -> &[u32] {
        self.zero_one.prior_tokens()
    }

    pub fn is_poisoned(&self) -> bool {
        self.state_poisoned
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpTextSessionError> {
        self.require_idle()?;
        if self.pending_length.is_some() {
            return invalid("cannot reset while a token update is pending");
        }
        self.committed_length = 0;
        self.state_poisoned = true;
        self.encode_failed = true;
        self.logits_ready = false;
        self.zero_one.reset()?;
        for block in &mut self.post_ple {
            block.reset()?;
        }
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    pub fn release_after(&mut self) -> Result<(), Qwen4ExpTextSessionError> {
        self.release_after_inner(None)
    }

    pub(crate) fn release_after_timed(
        &mut self,
    ) -> Result<Qwen4ExpTextSessionReleaseTiming, Qwen4ExpTextSessionError> {
        let started = Instant::now();
        let mut timing = Qwen4ExpTextSessionReleaseTiming::default();
        let result = self.release_after_inner(Some(&mut timing));
        timing.release_total_ms = started.elapsed().as_secs_f64() * 1e3;
        result?;
        Ok(timing)
    }

    fn release_after_inner(
        &mut self,
        mut timing: Option<&mut Qwen4ExpTextSessionReleaseTiming>,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        let Some(command) = self.active_command.clone() else {
            if self.pending_length.is_some() {
                return invalid("token length is pending without an owning command");
            }
            return Ok(());
        };
        let status = command.status();
        if matches!(
            status,
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued
        ) {
            return invalid(format!(
                "workspace owner is not committed (status {status:?}); commit it or abandon the uncommitted command"
            ));
        }
        let wait_started = timing.is_some().then(Instant::now);
        command.waitUntilCompleted();
        if let (Some(timing), Some(started)) = (timing.as_deref_mut(), wait_started) {
            timing.root_wait_ms = started.elapsed().as_secs_f64() * 1e3;
        }
        let status = command.status();
        let command_error = command.error().map(|error| error.to_string());
        let mut child_errors = Vec::new();
        let children_started = timing.is_some().then(Instant::now);
        if let Err(error) = self.zero_one.release_after() {
            child_errors.push(format!("layers zero-one: {error}"));
        }
        for (geometry, block) in self.geometry.post_ple.iter().zip(&mut self.post_ple) {
            if let Err(error) = block.release_after() {
                child_errors.push(format!("layer {}: {error}", geometry.layer()));
            }
        }
        if let Err(error) = self.final_read.release_after() {
            child_errors.push(format!("final HC read: {error}"));
        }
        if let (Some(timing), Some(started)) = (timing.as_deref_mut(), children_started) {
            timing.child_publication_ms = started.elapsed().as_secs_f64() * 1e3;
        }
        let publish_started = timing.is_some().then(Instant::now);
        self.active_command = None;
        let pending = self.pending_length.take();
        let pending_present = pending.is_some();
        let mut result = None;
        if let Some(expected) = pending {
            if self.zero_one.next_position() != Some(expected as u64) {
                child_errors.push(format!(
                    "PLE history position {:?}, expected {expected}",
                    self.zero_one.next_position()
                ));
            }
            for (geometry, block) in self.geometry.post_ple.iter().zip(&self.post_ple) {
                if geometry.mixer().kind() == MixerKind::QwenSparseAttention
                    && block.mixer_committed_length() != Some(expected)
                {
                    child_errors.push(format!(
                        "QSA layer {} committed length {:?}, expected {expected}",
                        geometry.layer(),
                        block.mixer_committed_length()
                    ));
                }
            }
            if status == MTLCommandBufferStatus::Completed
                && command_error.is_none()
                && child_errors.is_empty()
                && !self.encode_failed
            {
                self.committed_length = expected;
                self.encode_failed = false;
                self.logits_ready = true;
                result = Some(Ok(()));
            }
        }
        let result = result.unwrap_or_else(|| {
            self.state_poisoned = true;
            self.logits_ready = false;
            Err(Qwen4ExpTextSessionError::CommandBuffer(format!(
                "status={status:?}, error={command_error:?}, encode_failed={}, pending_length={pending_present}, children={child_errors:?}",
                self.encode_failed
            )))
        });
        if let (Some(timing), Some(started)) = (timing.as_deref_mut(), publish_started) {
            timing.root_publish_ms = started.elapsed().as_secs_f64() * 1e3;
        }
        result
    }

    /// Release a full session from a command that will never be committed.
    ///
    /// # Safety
    ///
    /// The caller must end and permanently discard every reference to the
    /// owning command. Committing it later may mutate all layer states and
    /// logits after another token acquires this session.
    pub unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpTextSessionError> {
        let Some(command) = self.active_command.as_ref() else {
            return Ok(());
        };
        let status = command.status();
        if status != MTLCommandBufferStatus::NotEnqueued {
            return invalid(format!(
                "only a NotEnqueued workspace owner can be abandoned, got {status:?}"
            ));
        }
        let mut child_errors = Vec::new();
        if let Err(error) = unsafe { self.zero_one.abandon_uncommitted() } {
            child_errors.push(format!("layers zero-one: {error}"));
        }
        for (geometry, block) in self.geometry.post_ple.iter().zip(&mut self.post_ple) {
            if let Err(error) = unsafe { block.abandon_uncommitted() } {
                child_errors.push(format!("layer {}: {error}", geometry.layer()));
            }
        }
        if let Err(error) = unsafe { self.final_read.abandon_uncommitted() } {
            child_errors.push(format!("final HC read: {error}"));
        }
        if !child_errors.is_empty() {
            return invalid(format!(
                "could not abandon every text-session child: {child_errors:?}"
            ));
        }
        self.active_command = None;
        self.pending_length = None;
        self.state_poisoned = false;
        self.encode_failed = false;
        self.logits_ready = self.committed_length > 0;
        Ok(())
    }

    pub fn logits(&self) -> Result<Qwen4ExpCompletedLogits<'_>, Qwen4ExpTextSessionError> {
        if self.active_command.is_some() || !self.logits_ready || self.state_poisoned {
            return invalid("completed logits are unavailable");
        }
        Ok(Qwen4ExpCompletedLogits { workspace: self })
    }

    #[cfg(test)]
    pub(crate) fn final_hidden_tensor(&self) -> &MetalTensor {
        self.final_read.mixed_tensor()
    }

    fn require_idle(&self) -> Result<(), Qwen4ExpTextSessionError> {
        if self.active_command.is_some() {
            invalid("workspace is still owned by a command buffer")
        } else {
            Ok(())
        }
    }

    fn packed_scratch(&self) -> Result<&Qwen4ExpTextPackedScratch, Qwen4ExpTextSessionError> {
        self.packed.as_ref().ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid(
                "packed prefill was not admitted for this session".into(),
            )
        })
    }
}

#[must_use = "end and commit the command, then release the text session"]
pub struct Qwen4ExpTextSessionPending<'a> {
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
    position: usize,
}

impl Qwen4ExpTextSessionPending<'_> {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn vocab_size(&self) -> usize {
        self.workspace.geometry.vocab_size()
    }
}

pub struct Qwen4ExpCompletedLogits<'a> {
    workspace: &'a Qwen4ExpTextSessionMetalWorkspace,
}

impl Qwen4ExpCompletedLogits<'_> {
    pub fn n_elements(&self) -> u64 {
        self.workspace.geometry.vocab_size() as u64
    }

    pub fn dtype(&self) -> GgmlType {
        GgmlType::F32
    }

    pub fn position(&self) -> usize {
        self.workspace.committed_length - 1
    }

    pub fn as_slice(&self) -> &[f32] {
        let offset = (self.workspace.logits.offset / size_of::<f32>() as u64) as usize;
        // SAFETY: session logits are an owned, shared F32 buffer. The owning
        // command completed before this handle became available, and the
        // immutable workspace borrow prevents another token from overwriting
        // the buffer while the returned slice is live.
        unsafe {
            std::slice::from_raw_parts(
                self.workspace
                    .logits
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<f32>()
                    .add(offset),
                self.workspace.geometry.vocab_size(),
            )
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn encode_qwen4exp_text_token<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    position: usize,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
) -> Result<Qwen4ExpTextSessionPending<'a>, Qwen4ExpTextSessionError> {
    encode_qwen4exp_text_token_inner(
        ctx, enc, token_id, position, table, weights, workspace, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen4exp_text_token_with_post_layer_hyper_probe<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    position: usize,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
    probe: &Qwen4ExpPostLayerHyperProbe<'_>,
) -> Result<Qwen4ExpTextSessionPending<'a>, Qwen4ExpTextSessionError> {
    encode_qwen4exp_text_token_inner(
        ctx,
        enc,
        token_id,
        position,
        table,
        weights,
        workspace,
        Some(probe),
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen4exp_text_token_inner<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    position: usize,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
    probe: Option<&Qwen4ExpPostLayerHyperProbe<'_>>,
) -> Result<Qwen4ExpTextSessionPending<'a>, Qwen4ExpTextSessionError> {
    if let Some(probe) = probe {
        validate_post_layer_hyper_probe(
            ctx,
            &workspace.geometry,
            &workspace.hyper_residual,
            probe,
        )?;
    }
    let next_history =
        prepare_qwen4exp_text_token(ctx, enc, token_id, position, table, weights, workspace)?;
    if let Err(error) = encode_step(
        ctx,
        enc,
        token_id,
        position,
        next_history,
        weights,
        workspace,
        probe,
    ) {
        workspace.encode_failed = true;
        workspace.state_poisoned = true;
        return Err(error);
    }
    Ok(Qwen4ExpTextSessionPending {
        workspace,
        position,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn encode_qwen4exp_text_packed<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_ids: &[u32],
    start_position: usize,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
) -> Result<Qwen4ExpTextSessionPending<'a>, Qwen4ExpTextSessionError> {
    encode_qwen4exp_text_packed_inner(
        ctx,
        enc,
        token_ids,
        start_position,
        table,
        weights,
        workspace,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen4exp_text_packed_profiled<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_ids: &[u32],
    start_position: usize,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
    recorder: &mut Qwen4ExpPackedProfileRecorder<'_>,
    cpu_timing: &mut Qwen4ExpPackedEncodeCpuTiming,
) -> Result<Qwen4ExpTextSessionPending<'a>, Qwen4ExpTextSessionError> {
    #[cfg(test)]
    if crate::qwen4exp_moe::qwen4exp_iq3_gate_up_capture_active() {
        return invalid("IQ3 gate/up capture is unavailable in profiled packed prefill");
    }
    encode_qwen4exp_text_packed_inner(
        ctx,
        enc,
        token_ids,
        start_position,
        table,
        weights,
        workspace,
        Some(recorder),
        Some(cpu_timing),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen4exp_text_packed_layer_sampled<'a>(
    ctx: &MetalContext,
    command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    samples: &MetalTimestampSampleBuffer,
    token_ids: &[u32],
    start_position: usize,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
    cpu_timing: &mut Qwen4ExpPackedEncodeCpuTiming,
) -> Result<
    (
        Qwen4ExpTextSessionPending<'a>,
        Vec<Qwen4ExpPackedProfileSpan>,
    ),
    Qwen4ExpTextSessionError,
> {
    #[cfg(test)]
    reject_active_sampled_diagnostics("packed")?;
    #[cfg(test)]
    if crate::qwen4exp_moe::qwen4exp_moe_route_count_capture_active() {
        return invalid("route-count capture is unavailable in sampled packed profiles");
    }
    #[cfg(test)]
    if crate::qwen4exp_moe::qwen4exp_iq3_gate_up_capture_active() {
        return invalid("IQ3 gate/up capture is unavailable in sampled packed profiles");
    }
    if token_ids.len() <= 1 {
        return invalid("sampled packed profile requires at least two tokens");
    }
    let expected_samples = packed_stage_sample_count(weights.post_ple.len())?;
    if samples.sample_count() != expected_samples {
        return invalid(format!(
            "packed layer profile has {} timestamp samples, expected {expected_samples}",
            samples.sample_count()
        ));
    }
    let detailed_blocks = weights
        .post_ple
        .iter()
        .filter(|weights| {
            is_stage_profiled_layer(weights.geometry.layer(), weights.geometry.mixer().kind())
        })
        .count();
    if detailed_blocks != 2 {
        return invalid(format!(
            "packed layer profile found {detailed_blocks} detailed blocks, expected 2"
        ));
    }
    let first = sampled_stage_encoder(command, samples, 0)?;
    let preflight_started = Instant::now();
    let (next_history, row_ids) = validate_and_preflight_packed(
        ctx,
        &first,
        token_ids,
        start_position,
        table,
        weights,
        workspace,
    )?;
    cpu_timing.preflight_ms = preflight_started.elapsed().as_secs_f64() * 1e3;
    let stage_started = Instant::now();
    stage_packed_inputs(token_ids, &row_ids, table, workspace)?;
    cpu_timing.stage_inputs_ms = stage_started.elapsed().as_secs_f64() * 1e3;
    let pending_length = start_position
        .checked_add(token_ids.len())
        .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("packed length overflow".into()))?;
    reserve_command(workspace, &first, pending_length)?;
    workspace.logits_ready = false;
    let graph_started = Instant::now();
    let spans = unsafe {
        encode_packed_step_layer_sampled(
            ctx,
            command,
            samples,
            first,
            start_position,
            next_history,
            weights,
            workspace,
            token_ids.len(),
        )
    };
    let spans = match spans {
        Ok(spans) => spans,
        Err(error) => {
            workspace.encode_failed = true;
            workspace.state_poisoned = true;
            return Err(error);
        }
    };
    cpu_timing.graph_encode_ms = graph_started.elapsed().as_secs_f64() * 1e3;
    Ok((
        Qwen4ExpTextSessionPending {
            workspace,
            position: pending_length - 1,
        },
        spans,
    ))
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen4exp_text_packed_inner<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_ids: &[u32],
    start_position: usize,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
    profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
    mut cpu_timing: Option<&mut Qwen4ExpPackedEncodeCpuTiming>,
) -> Result<Qwen4ExpTextSessionPending<'a>, Qwen4ExpTextSessionError> {
    if token_ids.len() == 1 {
        return encode_qwen4exp_text_token(
            ctx,
            enc,
            token_ids[0],
            start_position,
            table,
            weights,
            workspace,
        );
    }
    let encode = || {
        let preflight_started = cpu_timing.is_some().then(Instant::now);
        let (next_history, row_ids) = validate_and_preflight_packed(
            ctx,
            enc,
            token_ids,
            start_position,
            table,
            weights,
            workspace,
        )?;
        if let (Some(timing), Some(started)) = (cpu_timing.as_deref_mut(), preflight_started) {
            timing.preflight_ms = started.elapsed().as_secs_f64() * 1e3;
        }
        let stage_started = cpu_timing.is_some().then(Instant::now);
        stage_packed_inputs(token_ids, &row_ids, table, workspace)?;
        if let (Some(timing), Some(started)) = (cpu_timing.as_deref_mut(), stage_started) {
            timing.stage_inputs_ms = started.elapsed().as_secs_f64() * 1e3;
        }
        let pending_length = start_position
            .checked_add(token_ids.len())
            .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("packed length overflow".into()))?;
        reserve_command(workspace, enc, pending_length)?;
        workspace.logits_ready = false;
        let graph_started = cpu_timing.is_some().then(Instant::now);
        if let Err(error) = unsafe {
            encode_packed_step(
                ctx,
                enc,
                start_position,
                next_history,
                weights,
                workspace,
                token_ids.len(),
                profile,
            )
        } {
            workspace.encode_failed = true;
            workspace.state_poisoned = true;
            return Err(error);
        }
        if let (Some(timing), Some(started)) = (cpu_timing.as_deref_mut(), graph_started) {
            timing.graph_encode_ms = started.elapsed().as_secs_f64() * 1e3;
        }
        Ok(Qwen4ExpTextSessionPending {
            workspace,
            position: pending_length - 1,
        })
    };
    #[cfg(test)]
    {
        let has_diagnostic_range =
            !token_ids.is_empty() && start_position.checked_add(token_ids.len()).is_some();
        if has_diagnostic_range {
            return with_diagnostic_execution_range(start_position, token_ids.len(), encode);
        }
    }
    encode()
}

#[allow(clippy::too_many_arguments)]
pub fn encode_qwen4exp_text_token_layer_sampled<'a>(
    ctx: &MetalContext,
    command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    samples: &MetalTimestampSampleBuffer,
    token_id: u32,
    position: usize,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
) -> Result<Qwen4ExpTextSessionPending<'a>, Qwen4ExpTextSessionError> {
    #[cfg(test)]
    reject_active_sampled_diagnostics("token")?;
    #[cfg(test)]
    if crate::qwen4exp_moe::qwen4exp_moe_route_count_capture_active() {
        return invalid("route-count capture is unavailable in sampled token profiles");
    }
    #[cfg(test)]
    if crate::qwen4exp_moe::qwen4exp_iq3_gate_up_capture_active() {
        return invalid("IQ3 gate/up capture is unavailable in sampled token profiles");
    }
    let stage_count = weights
        .post_ple
        .len()
        .checked_add(2)
        .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("stage count overflow".into()))?;
    let expected_samples = stage_count
        .checked_mul(2)
        .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("stage sample count overflow".into()))?;
    if samples.sample_count() != expected_samples {
        return invalid(format!(
            "layer profile has {} timestamp samples, expected {expected_samples}",
            samples.sample_count()
        ));
    }

    let first = sampled_stage_encoder(command, samples, 0)?;
    let next_history =
        prepare_qwen4exp_text_token(ctx, &first, token_id, position, table, weights, workspace)?;
    let encoded = (|| {
        encode_zero_one_stage(
            ctx,
            &first,
            token_id,
            position,
            next_history,
            weights,
            workspace,
        )?;
        first.end();
        for index in 0..weights.post_ple.len() {
            let encoder = sampled_stage_encoder(command, samples, index + 1)?;
            encode_post_ple_stage(ctx, &encoder, position, index, weights, workspace)?;
            encoder.end();
        }
        let tail = sampled_stage_encoder(command, samples, stage_count - 1)?;
        let hyper_residual = workspace.hyper_residual.clone();
        encode_tail_stage(ctx, &tail, &hyper_residual, weights, workspace, true)?;
        tail.end();
        Ok(())
    })();
    if let Err(error) = encoded {
        workspace.encode_failed = true;
        workspace.state_poisoned = true;
        return Err(error);
    }
    Ok(Qwen4ExpTextSessionPending {
        workspace,
        position,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_qwen4exp_text_token(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    position: usize,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
) -> Result<PleHistory, Qwen4ExpTextSessionError> {
    validate_encoder(ctx, enc)?;
    validate_and_preflight(ctx, enc, token_id, position, weights, workspace)?;
    let next_history = crate::qwen4exp_layers_zero_one::stage_ple_rows(
        token_id,
        position as u64,
        table,
        weights.zero_one,
        &mut workspace.zero_one,
    )?;
    for block in &mut workspace.post_ple {
        block.prepare_for_parent(position)?;
    }
    reserve_command(workspace, enc, position + 1)?;
    workspace.logits_ready = false;
    Ok(next_history)
}

#[allow(clippy::too_many_arguments)]
fn validate_and_preflight_packed(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_ids: &[u32],
    start_position: usize,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &Qwen4ExpTextSessionMetalWorkspace,
) -> Result<(PleHistory, Vec<u32>), Qwen4ExpTextSessionError> {
    validate_encoder(ctx, enc)?;
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if workspace.pending_length.is_some() {
        return invalid("workspace has a token update without a command owner");
    }
    if weights.geometry != workspace.geometry {
        return invalid("packed text-session weight and workspace geometry differ");
    }
    if weights.post_ple.len() != workspace.post_ple.len() {
        return invalid("packed post-PLE weight and workspace counts differ");
    }
    if start_position != workspace.committed_length {
        return invalid(format!(
            "packed start position {start_position} differs from committed length {}",
            workspace.committed_length
        ));
    }
    let tokens = token_ids.len();
    let packed = workspace.packed_scratch()?;
    if tokens <= 1 || tokens > packed.capacity {
        return invalid(format!(
            "packed text token count {tokens} is outside 2..={}",
            packed.capacity
        ));
    }
    let end_position = start_position.checked_add(tokens).ok_or_else(|| {
        Qwen4ExpTextSessionError::Invalid("packed position range overflow".into())
    })?;
    if end_position > workspace.geometry.capacity {
        return invalid(format!(
            "packed position range {start_position}..{end_position} exceeds session capacity {}",
            workspace.geometry.capacity
        ));
    }
    if let Some((index, token_id)) = token_ids.iter().copied().enumerate().find(|(_, token_id)| {
        *token_id as usize >= workspace.geometry.vocab_size() || i32::try_from(*token_id).is_err()
    }) {
        return invalid(format!(
            "token ID {token_id} at packed row {index} is outside the Metal vocabulary contract"
        ));
    }
    let ple_geometry = workspace.geometry.zero_one().ple();
    if table.row_width() != ple_geometry.head_dim() {
        return invalid(format!(
            "PLE table row width {} differs from packed head width {}",
            table.row_width(),
            ple_geometry.head_dim()
        ));
    }

    let views = packed.views(&workspace.geometry, tokens)?;
    require_projection(
        "packed token embedding",
        weights.zero_one.layer_zero.token_embedding,
        workspace.geometry.hidden_size(),
        workspace.geometry.vocab_size(),
        &[GgmlType::Q8_0],
    )?;
    require_tensor(
        "packed token IDs",
        &views.token_ids,
        GgmlType::I32,
        &[tokens as u64],
        true,
    )?;
    let lookup_count = ple_geometry
        .head_count()
        .checked_mul(tokens)
        .ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("packed PLE lookup count overflow".into())
        })?;
    require_tensor(
        "packed PLE rows",
        &views.ple_packed_rows,
        GgmlType::IQ4_NL,
        &[
            ple_geometry.head_dim() as u64,
            (ple_geometry.head_count() * packed.capacity) as u64,
        ],
        true,
    )?;
    require_tensor(
        "packed PLE local row IDs",
        &views.ple_local_row_ids,
        GgmlType::I32,
        &[lookup_count as u64],
        false,
    )?;
    require_tensor(
        "packed PLE embedding",
        &views.ple_embedding,
        GgmlType::F32,
        &[workspace.geometry.hidden_size() as u64, tokens as u64],
        true,
    )?;
    require_read_only_weights(&[(
        "packed token embedding",
        weights.zero_one.layer_zero.token_embedding,
    )])?;
    let top_level = [
        ("packed token IDs", &views.token_ids),
        ("packed token embedding", &views.embedding),
        ("packed PLE rows", &views.ple_packed_rows),
        ("packed PLE local row IDs", &views.ple_local_row_ids),
        ("packed PLE embedding", &views.ple_embedding),
        ("packed hyper residual", &views.hyper_residual),
        ("packed block bridge", &views.bridge),
        (
            "packed token embedding weights",
            weights.zero_one.layer_zero.token_embedding,
        ),
    ];
    require_same_device(ctx, &top_level)?;
    require_disjoint(&top_level)?;
    for kernel in ["kernel_get_rows_q8_0_f32", "kernel_get_rows_iq4_nl_f32"] {
        let pipeline = ctx.pipeline(kernel)?;
        if pipeline.threadExecutionWidth() != 32 || pipeline.maxTotalThreadsPerThreadgroup() < 32 {
            return invalid(format!(
                "packed row pipeline {kernel} requires SIMD width and capacity 32, got width={} capacity={}",
                pipeline.threadExecutionWidth(),
                pipeline.maxTotalThreadsPerThreadgroup()
            ));
        }
    }
    validate_and_preflight_hc_repeat_packed(
        ctx,
        &views.embedding,
        &views.hyper_residual,
        workspace.geometry.branch_count(),
        workspace.geometry.hidden_size(),
        tokens,
    )?;
    let (next_history, row_ids) = crate::qwen4exp_layers_zero_one::validate_and_preflight_packed(
        ctx,
        enc,
        token_ids,
        start_position as u64,
        &views.ple_embedding,
        &views.hyper_residual,
        &views.bridge,
        weights.zero_one,
        &workspace.zero_one,
        &packed.residual,
        &packed.gdn,
        &packed.ple,
        &packed.moe,
    )?;
    for ((geometry, weights), block) in workspace
        .geometry
        .post_ple
        .iter()
        .zip(&weights.post_ple)
        .zip(&workspace.post_ple)
    {
        if weights.geometry != *geometry {
            return invalid(format!(
                "layer {} packed weight geometry differs from the session",
                geometry.layer()
            ));
        }
        crate::qwen4exp_post_ple_block::validate_and_preflight_packed(
            ctx,
            enc,
            start_position,
            &views.hyper_residual,
            &views.bridge,
            *weights,
            block,
            &packed.residual,
            &packed.gdn,
            &packed.qsa,
            &packed.moe,
            tokens,
        )?;
    }
    let last_hyper = views.hyper_residual.view_subrange(
        ((tokens - 1) * workspace.geometry.hyper_width()) as u64,
        vec![workspace.geometry.hyper_width() as u64],
    );
    validate_final_contract(ctx, &last_hyper, weights, workspace)?;
    Ok((next_history, row_ids))
}

fn stage_packed_inputs(
    token_ids: &[u32],
    row_ids: &[u32],
    table: PleIq4NlTable<'_>,
    workspace: &Qwen4ExpTextSessionMetalWorkspace,
) -> Result<(), Qwen4ExpTextSessionError> {
    let packed = workspace.packed_scratch()?;
    write_i32_prefix(&packed.token_ids, token_ids)?;
    let mut packed_rows = vec![0_u8; table.packed_staging_bytes(row_ids.len())?];
    table.gather_packed_into(row_ids, &mut packed_rows)?;
    write_tensor_prefix_bytes(&packed.ple_packed_rows, &packed_rows)
}

/// Encode a dense packed prefix into the scalar session's persistent state.
///
/// # Safety
///
/// The caller must retain and serialize the owning command and workspace until
/// completion or permanent abandonment.
#[allow(clippy::too_many_arguments)]
unsafe fn encode_packed_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    start_position: usize,
    next_history: PleHistory,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
    tokens: usize,
    mut profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
) -> Result<(), Qwen4ExpTextSessionError> {
    let views = workspace
        .packed_scratch()?
        .views(&workspace.geometry, tokens)?;
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::coarse("input_embedding", None, None),
    )?;
    encode_get_rows_f32(
        ctx,
        enc,
        weights.zero_one.layer_zero.token_embedding,
        &views.token_ids,
        &views.embedding,
        tokens,
        workspace.geometry.hidden_size(),
    )?;
    end_optional(&mut profile, enc, marker)?;
    let ple_geometry = workspace.geometry.zero_one().ple();
    let lookup_count = ple_geometry.head_count() * tokens;
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::coarse("ple_dequant", None, None),
    )?;
    encode_get_rows_f32(
        ctx,
        enc,
        &views.ple_packed_rows,
        &views.ple_local_row_ids,
        &views.ple_embedding,
        lookup_count,
        ple_geometry.head_dim(),
    )?;
    end_optional(&mut profile, enc, marker)?;
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::coarse("hc_bootstrap", None, None),
    )?;
    encode_hc_repeat_packed(
        ctx,
        enc,
        &views.embedding,
        &views.hyper_residual,
        workspace.geometry.branch_count(),
        workspace.geometry.hidden_size(),
        tokens,
    )?;
    end_optional(&mut profile, enc, marker)?;

    let Qwen4ExpTextSessionMetalWorkspace {
        packed,
        zero_one,
        post_ple,
        ..
    } = workspace;
    let packed = packed.as_mut().ok_or_else(|| {
        Qwen4ExpTextSessionError::Invalid("packed prefill was not admitted for this session".into())
    })?;
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::coarse("layers_zero_one", None, None),
    )?;
    unsafe {
        encode_qwen4exp_layers_zero_one_packed_staged(
            ctx,
            enc,
            &views.ple_embedding,
            &views.hyper_residual,
            &views.bridge,
            weights.zero_one,
            next_history,
            zero_one,
            &mut packed.residual,
            &packed.gdn,
            &packed.ple,
            &packed.moe,
            tokens,
        )
    }?;
    end_optional(&mut profile, enc, marker)?;
    for index in 0..weights.post_ple.len() {
        let block_weights = weights.post_ple.get(index).copied().ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid(format!(
                "packed post-PLE weight index {index} is absent"
            ))
        })?;
        let block = post_ple.get_mut(index).ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid(format!(
                "packed post-PLE workspace index {index} is absent"
            ))
        })?;
        let layer = block_weights.geometry.layer();
        let mixer = block_weights.geometry.mixer().kind();
        let marker = begin_optional(
            &mut profile,
            enc,
            Qwen4ExpPackedProfileLabel::coarse("post_ple_layer", Some(layer), Some(mixer)),
        )?;
        if let Some(recorder) = profile.as_deref_mut() {
            unsafe {
                encode_qwen4exp_post_ple_block_packed_profiled(
                    ctx,
                    enc,
                    start_position,
                    &views.hyper_residual,
                    &views.bridge,
                    block_weights,
                    block,
                    &mut packed.residual,
                    &packed.gdn,
                    &packed.qsa,
                    &packed.moe,
                    tokens,
                    recorder,
                )
            }?;
        } else {
            unsafe {
                encode_qwen4exp_post_ple_block_packed(
                    ctx,
                    enc,
                    start_position,
                    &views.hyper_residual,
                    &views.bridge,
                    block_weights,
                    block,
                    &mut packed.residual,
                    &packed.gdn,
                    &packed.qsa,
                    &packed.moe,
                    tokens,
                )
            }?;
        }
        end_optional(&mut profile, enc, marker)?;
    }
    let last_hyper = views.hyper_residual.view_subrange(
        ((tokens - 1) * workspace.geometry.hyper_width()) as u64,
        vec![workspace.geometry.hyper_width() as u64],
    );
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::coarse("tail", None, None),
    )?;
    encode_tail_stage(ctx, enc, &last_hyper, weights, workspace, false)?;
    end_optional(&mut profile, enc, marker)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_packed_step_layer_sampled(
    ctx: &MetalContext,
    command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    samples: &MetalTimestampSampleBuffer,
    first: KernelEncoder,
    start_position: usize,
    next_history: PleHistory,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
    tokens: usize,
) -> Result<Vec<Qwen4ExpPackedProfileSpan>, Qwen4ExpTextSessionError> {
    let views = workspace
        .packed_scratch()?
        .views(&workspace.geometry, tokens)?;
    encode_get_rows_f32(
        ctx,
        &first,
        weights.zero_one.layer_zero.token_embedding,
        &views.token_ids,
        &views.embedding,
        tokens,
        workspace.geometry.hidden_size(),
    )?;
    let ple_geometry = workspace.geometry.zero_one().ple();
    encode_get_rows_f32(
        ctx,
        &first,
        &views.ple_packed_rows,
        &views.ple_local_row_ids,
        &views.ple_embedding,
        ple_geometry.head_count() * tokens,
        ple_geometry.head_dim(),
    )?;
    encode_hc_repeat_packed(
        ctx,
        &first,
        &views.embedding,
        &views.hyper_residual,
        workspace.geometry.branch_count(),
        workspace.geometry.hidden_size(),
        tokens,
    )?;
    let Qwen4ExpTextSessionMetalWorkspace {
        packed,
        zero_one,
        post_ple,
        ..
    } = workspace;
    let packed = packed.as_mut().ok_or_else(|| {
        Qwen4ExpTextSessionError::Invalid("packed prefill was not admitted for this session".into())
    })?;
    unsafe {
        encode_qwen4exp_layers_zero_one_packed_staged(
            ctx,
            &first,
            &views.ple_embedding,
            &views.hyper_residual,
            &views.bridge,
            weights.zero_one,
            next_history,
            zero_one,
            &mut packed.residual,
            &packed.gdn,
            &packed.ple,
            &packed.moe,
            tokens,
        )
    }?;
    first.end();
    let expected_spans = packed_stage_span_count(weights.post_ple.len())?;
    let mut spans = Vec::with_capacity(expected_spans);
    spans.push(Qwen4ExpPackedProfileSpan {
        label: Qwen4ExpPackedProfileLabel::coarse("bootstrap_layers_zero_one", None, None),
        depth: 0,
        start_sample: 0,
        end_sample: 1,
    });
    let mut next_stage = 1;
    for index in 0..weights.post_ple.len() {
        let block_weights = weights.post_ple.get(index).copied().ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid(format!(
                "packed post-PLE weight index {index} is absent"
            ))
        })?;
        let block = post_ple.get_mut(index).ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid(format!(
                "packed post-PLE workspace index {index} is absent"
            ))
        })?;
        let layer = block_weights.geometry.layer();
        let mixer = block_weights.geometry.mixer().kind();
        if let Some(stage_count) = stage_profile_count(layer, mixer) {
            spans.extend(unsafe {
                encode_qwen4exp_post_ple_block_packed_stage_sampled(
                    ctx,
                    command,
                    samples,
                    next_stage,
                    start_position,
                    &views.hyper_residual,
                    &views.bridge,
                    block_weights,
                    block,
                    &mut packed.residual,
                    &packed.gdn,
                    &packed.qsa,
                    &packed.moe,
                    tokens,
                )
            }?);
            next_stage = next_stage.checked_add(stage_count).ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid("packed profile stage cursor overflow".into())
            })?;
        } else {
            let encoder = sampled_stage_encoder(command, samples, next_stage)?;
            unsafe {
                encode_qwen4exp_post_ple_block_packed(
                    ctx,
                    &encoder,
                    start_position,
                    &views.hyper_residual,
                    &views.bridge,
                    block_weights,
                    block,
                    &mut packed.residual,
                    &packed.gdn,
                    &packed.qsa,
                    &packed.moe,
                    tokens,
                )
            }?;
            encoder.end();
            spans.push(Qwen4ExpPackedProfileSpan {
                label: Qwen4ExpPackedProfileLabel::coarse(
                    "post_ple_layer",
                    Some(layer),
                    Some(mixer),
                ),
                depth: 0,
                start_sample: next_stage * 2,
                end_sample: next_stage * 2 + 1,
            });
            next_stage = next_stage.checked_add(1).ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid("packed profile stage cursor overflow".into())
            })?;
        }
    }
    let tail_stage = next_stage;
    if (tail_stage + 1) * 2 != samples.sample_count() {
        return invalid(format!(
            "packed profile used {} stages before a {}-sample tail",
            tail_stage + 1,
            samples.sample_count()
        ));
    }
    let tail = sampled_stage_encoder(command, samples, tail_stage)?;
    let last_hyper = views.hyper_residual.view_subrange(
        ((tokens - 1) * workspace.geometry.hyper_width()) as u64,
        vec![workspace.geometry.hyper_width() as u64],
    );
    encode_tail_stage(ctx, &tail, &last_hyper, weights, workspace, false)?;
    tail.end();
    spans.push(Qwen4ExpPackedProfileSpan {
        label: Qwen4ExpPackedProfileLabel::coarse("tail", None, None),
        depth: 0,
        start_sample: tail_stage * 2,
        end_sample: tail_stage * 2 + 1,
    });
    if spans.len() != expected_spans {
        return invalid(format!(
            "packed profile produced {} spans, expected {expected_spans}",
            spans.len()
        ));
    }
    Ok(spans)
}

fn validate_and_preflight(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    position: usize,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &Qwen4ExpTextSessionMetalWorkspace,
) -> Result<(), Qwen4ExpTextSessionError> {
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if workspace.pending_length.is_some() {
        return invalid("workspace has a token update without a command owner");
    }
    if weights.geometry != workspace.geometry {
        return invalid("text-session weight and workspace geometry differ");
    }
    if weights.post_ple.len() != workspace.post_ple.len() {
        return invalid("post-PLE weight and workspace counts differ");
    }
    if position != workspace.committed_length {
        return invalid(format!(
            "position {position} differs from committed length {}",
            workspace.committed_length
        ));
    }
    if position >= workspace.geometry.capacity {
        return invalid(format!(
            "text-session capacity {} is exhausted",
            workspace.geometry.capacity
        ));
    }
    crate::qwen4exp_layers_zero_one::validate_and_preflight(
        ctx,
        enc,
        token_id,
        position as u64,
        weights.zero_one,
        &workspace.zero_one,
    )?;
    for ((geometry, weights), block) in workspace
        .geometry
        .post_ple
        .iter()
        .zip(&weights.post_ple)
        .zip(&workspace.post_ple)
    {
        if weights.geometry != *geometry {
            return invalid(format!(
                "layer {} weight geometry differs from the session",
                geometry.layer()
            ));
        }
        crate::qwen4exp_post_ple_block::validate_and_preflight(
            ctx,
            position,
            &workspace.hyper_residual,
            *weights,
            block,
        )?;
    }
    validate_final_contract(ctx, &workspace.hyper_residual, weights, workspace)?;
    Ok(())
}

fn validate_final_contract(
    ctx: &MetalContext,
    hyper_residual: &MetalTensor,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &Qwen4ExpTextSessionMetalWorkspace,
) -> Result<(), Qwen4ExpTextSessionError> {
    let g = &workspace.geometry;
    require_tensor(
        "session hyper residual",
        hyper_residual,
        GgmlType::F32,
        &[g.hyper_width() as u64],
        true,
    )?;
    if workspace.final_read.branch_count() != g.branch_count()
        || workspace.final_read.hidden_size() != g.hidden_size()
        || workspace.final_read.low_rank() != g.low_rank()
    {
        return invalid("final HC scratch geometry differs from the session");
    }
    validate_and_preflight_final_gated_residual_mix(
        ctx,
        hyper_residual,
        g.eps(),
        weights.final_read,
        &workspace.final_read,
    )?;
    require_projection(
        "output projection",
        weights.output,
        g.hidden_size(),
        g.vocab_size(),
        &[GgmlType::F32, GgmlType::Q6_K],
    )?;
    require_tensor(
        "session logits",
        &workspace.logits,
        GgmlType::F32,
        &[g.vocab_size() as u64],
        true,
    )?;
    let final_weights = [
        ("final HC norm", weights.final_read.norm),
        ("final HC down", weights.final_read.down),
        ("final HC up", weights.final_read.up),
        ("output projection", weights.output),
    ];
    require_read_only_weights(&final_weights)?;
    let tensors = [
        ("session hyper residual", hyper_residual),
        ("final HC norm", weights.final_read.norm),
        ("final HC down", weights.final_read.down),
        ("final HC up", weights.final_read.up),
        ("final HC mixed output", workspace.final_read.mixed_tensor()),
        ("output projection", weights.output),
        ("session logits", &workspace.logits),
    ];
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)?;
    match weights.output.dtype {
        GgmlType::F32 => {
            ctx.pipeline("kernel_mat_vec_f32_f32")?;
            ctx.pipeline("kernel_mat_vec_f32_f32_lcpp_r2")?;
        }
        GgmlType::Q6_K => {
            ctx.pipeline("kernel_mat_vec_q6_K_f32")?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn encode_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    position: usize,
    next_history: PleHistory,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
    probe: Option<&Qwen4ExpPostLayerHyperProbe<'_>>,
) -> Result<(), Qwen4ExpTextSessionError> {
    encode_zero_one_stage(
        ctx,
        enc,
        token_id,
        position,
        next_history,
        weights,
        workspace,
    )?;
    encode_post_layer_hyper_probe(ctx, enc, 1, &workspace.hyper_residual, probe)?;
    for index in 0..weights.post_ple.len() {
        encode_post_ple_stage(ctx, enc, position, index, weights, workspace)?;
        encode_post_layer_hyper_probe(
            ctx,
            enc,
            index as u32 + 2,
            &workspace.hyper_residual,
            probe,
        )?;
    }
    let hyper_residual = workspace.hyper_residual.clone();
    encode_tail_stage(ctx, enc, &hyper_residual, weights, workspace, true)
}

fn validate_post_layer_hyper_probe(
    ctx: &MetalContext,
    geometry: &Qwen4ExpTextSessionMetalGeometry,
    hyper_residual: &MetalTensor,
    probe: &Qwen4ExpPostLayerHyperProbe<'_>,
) -> Result<(), Qwen4ExpTextSessionError> {
    let layer_count = geometry.layer_count();
    if probe.layer == 0 || probe.layer as usize >= layer_count {
        return invalid(format!(
            "post-layer hyper probe layer {} is outside 1..{}",
            probe.layer, layer_count
        ));
    }
    let shape = [geometry.hyper_width() as u64];
    require_tensor(
        "session hyper residual",
        hyper_residual,
        GgmlType::F32,
        &shape,
        true,
    )?;
    require_tensor(
        "post-layer hyper capture",
        probe.capture,
        GgmlType::F32,
        &shape,
        true,
    )?;
    let mut tensors = vec![
        ("session hyper residual", hyper_residual),
        ("post-layer hyper capture", probe.capture),
    ];
    if let Some(fixed_add) = probe.fixed_add {
        if !fixed_add.coefficient.is_finite() || fixed_add.coefficient == 0.0 {
            return invalid(format!(
                "post-layer hyper fixed-add coefficient must be finite and nonzero, got {}",
                fixed_add.coefficient
            ));
        }
        require_tensor(
            "post-layer hyper fixed-add direction",
            fixed_add.direction,
            GgmlType::F32,
            &shape,
            false,
        )?;
        tensors.push(("post-layer hyper fixed-add direction", fixed_add.direction));
        ctx.pipeline("kernel_axpy_f32")?;
    }
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)?;
    ctx.pipeline("kernel_copy_offset_f32")?;
    Ok(())
}

fn encode_post_layer_hyper_probe(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    layer: u32,
    hyper_residual: &MetalTensor,
    probe: Option<&Qwen4ExpPostLayerHyperProbe<'_>>,
) -> Result<(), Qwen4ExpTextSessionError> {
    let Some(probe) = probe.filter(|probe| probe.layer == layer) else {
        return Ok(());
    };
    if let Some(fixed_add) = probe.fixed_add {
        encode_axpy_f32(
            ctx,
            enc,
            fixed_add.direction,
            hyper_residual,
            fixed_add.coefficient,
        )?;
    }
    encode_copy_offset_f32(
        ctx,
        enc,
        hyper_residual,
        0,
        probe.capture,
        hyper_residual.n_elements() as usize,
    )?;
    Ok(())
}

fn encode_zero_one_stage(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    position: usize,
    next_history: PleHistory,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
) -> Result<(), Qwen4ExpTextSessionError> {
    let zero_one = with_diagnostic_execution_range(position, 1, || {
        encode_qwen4exp_layers_zero_one_staged(
            ctx,
            enc,
            token_id,
            weights.zero_one,
            next_history,
            &mut workspace.zero_one,
        )
    })?;
    zero_one
        .output()
        .encode_copy_to(ctx, enc, &workspace.hyper_residual)?;
    drop(zero_one);
    Ok(())
}

fn encode_post_ple_stage(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    position: usize,
    index: usize,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
) -> Result<(), Qwen4ExpTextSessionError> {
    let block_weights = weights.post_ple.get(index).copied().ok_or_else(|| {
        Qwen4ExpTextSessionError::Invalid(format!("post-PLE weight index {index} is absent"))
    })?;
    let block = workspace.post_ple.get_mut(index).ok_or_else(|| {
        Qwen4ExpTextSessionError::Invalid(format!("post-PLE workspace index {index} is absent"))
    })?;
    let read = with_diagnostic_execution_range(position, 1, || {
        encode_qwen4exp_post_ple_block(
            ctx,
            enc,
            position,
            &workspace.hyper_residual,
            block_weights,
            block,
        )
    })?;
    drop(read);
    Ok(())
}

fn encode_tail_stage(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hyper_residual: &MetalTensor,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
    allow_hc_up_mix: bool,
) -> Result<(), Qwen4ExpTextSessionError> {
    let final_read = crate::qwen4exp_metal::encode_final_gated_residual_mix_with_policy(
        ctx,
        enc,
        hyper_residual,
        workspace.geometry.eps(),
        weights.final_read,
        &mut workspace.final_read,
        allow_hc_up_mix,
    )?;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.output,
        final_read.mixed(),
        &workspace.logits,
        workspace.geometry.hidden_size(),
        workspace.geometry.vocab_size(),
    )?;
    drop(final_read);
    Ok(())
}

fn sampled_stage_encoder(
    command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    samples: &MetalTimestampSampleBuffer,
    stage: usize,
) -> Result<KernelEncoder, Qwen4ExpTextSessionError> {
    let start_sample = stage
        .checked_mul(2)
        .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("stage sample index overflow".into()))?;
    let end_sample = start_sample + 1;
    if end_sample >= samples.sample_count() {
        return invalid(format!(
            "stage {stage} timestamp index {end_sample} exceeds {} samples",
            samples.sample_count()
        ));
    }
    Ok(KernelEncoder::try_begin_sampled(
        command,
        samples,
        start_sample,
        end_sample,
        false,
    )?)
}

fn validate_encoder(
    ctx: &MetalContext,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpTextSessionError> {
    let command = enc.parent_command_buffer();
    let actual = command.device().registryID();
    let expected = ctx.device.registryID();
    if actual != expected {
        return invalid(format!(
            "encoder belongs to Metal device registry {actual}, context is {expected}"
        ));
    }
    if enc.is_concurrent() {
        return invalid("text-session dependent dispatches require a serial encoder");
    }
    let status = command.status();
    if status != MTLCommandBufferStatus::NotEnqueued {
        return invalid(format!(
            "text-session encoding requires a NotEnqueued command buffer, got {status:?}"
        ));
    }
    Ok(())
}

fn reserve_command(
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
    enc: &KernelEncoder,
    pending_length: usize,
) -> Result<(), Qwen4ExpTextSessionError> {
    workspace.require_idle()?;
    workspace.active_command = Some(enc.parent_command_buffer());
    workspace.pending_length = Some(pending_length);
    workspace.encode_failed = false;
    Ok(())
}

fn packed_prefix_view(
    name: &str,
    tensor: &MetalTensor,
    width: usize,
    tokens: usize,
    capacity: usize,
) -> Result<MetalTensor, Qwen4ExpTextSessionError> {
    if tokens == 0 || tokens > capacity {
        return invalid(format!(
            "{name} token count {tokens} is outside capacity {capacity}"
        ));
    }
    require_tensor(
        name,
        tensor,
        GgmlType::F32,
        &[width as u64, capacity as u64],
        true,
    )?;
    let elements = width.checked_mul(tokens).ok_or_else(|| {
        Qwen4ExpTextSessionError::Invalid(format!("{name} element count overflow"))
    })?;
    let view = tensor.view_subrange(0, vec![width as u64, tokens as u64]);
    if view.n_elements() as usize != elements {
        return invalid(format!("{name} prefix view has the wrong element count"));
    }
    Ok(view)
}

fn write_i32_prefix(tensor: &MetalTensor, values: &[u32]) -> Result<(), Qwen4ExpTextSessionError> {
    if tensor.dtype != GgmlType::I32 || !tensor.is_writable() {
        return invalid("packed token staging must be writable I32");
    }
    let bytes = values.len().checked_mul(size_of::<i32>()).ok_or_else(|| {
        Qwen4ExpTextSessionError::Invalid("packed token staging byte count overflow".into())
    })?;
    let available = tensor
        .buffer
        .length()
        .checked_sub(tensor.offset as usize)
        .ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("packed token staging offset is invalid".into())
        })?;
    if bytes > available || !tensor.offset.is_multiple_of(size_of::<i32>() as u64) {
        return invalid("packed token staging exceeds or misaligns its buffer");
    }
    let offset = tensor.offset as usize / size_of::<i32>();
    let destination = tensor.buffer.contents().as_ptr().cast::<i32>();
    for (index, &value) in values.iter().enumerate() {
        let value = i32::try_from(value).map_err(|_| {
            Qwen4ExpTextSessionError::Invalid(format!(
                "packed token ID {value} exceeds signed Metal indexing"
            ))
        })?;
        // SAFETY: byte capacity and alignment are checked above, and this
        // workspace is exclusively borrowed before command ownership begins.
        unsafe { destination.add(offset + index).write(value) };
    }
    Ok(())
}

fn write_tensor_prefix_bytes(
    tensor: &MetalTensor,
    bytes: &[u8],
) -> Result<(), Qwen4ExpTextSessionError> {
    if !tensor.is_writable() {
        return invalid("packed PLE staging tensor must be writable");
    }
    let end = tensor
        .offset
        .checked_add(bytes.len() as u64)
        .ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("packed PLE staging range overflow".into())
        })?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "packed PLE staging requires {} bytes beyond offset {}, buffer has {}",
            bytes.len(),
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    // SAFETY: the destination range is checked above and remains exclusively
    // owned until the subsequently encoded command completes.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize),
            bytes.len(),
        );
    }
    Ok(())
}

fn require_projection(
    name: &str,
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
    dtypes: &[GgmlType],
) -> Result<(), Qwen4ExpTextSessionError> {
    if tensor.shape != [n_in as u64, n_out as u64] || !dtypes.contains(&tensor.dtype) {
        return invalid(format!(
            "{name} must use {dtypes:?} with shape [{n_in}, {n_out}], got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    let (block, _) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpTextSessionError::Invalid(format!("unsupported dtype {:?}", tensor.dtype))
    })?;
    if !(n_in as u64).is_multiple_of(block) {
        return invalid(format!(
            "{name} input width {n_in} is not aligned to {block} elements"
        ));
    }
    require_range(name, tensor)
}

fn require_tensor(
    name: &str,
    tensor: &MetalTensor,
    dtype: GgmlType,
    shape: &[u64],
    writable: bool,
) -> Result<(), Qwen4ExpTextSessionError> {
    if tensor.dtype != dtype || tensor.shape != shape {
        return invalid(format!(
            "{name} must be {dtype:?} with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    if writable && !tensor.is_writable() {
        return invalid(format!("{name} must be writable"));
    }
    require_range(name, tensor)
}

fn storage_bytes(tensor: &MetalTensor) -> Result<u64, Qwen4ExpTextSessionError> {
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("tensor element count overflow".into()))?;
    let (block, bytes) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpTextSessionError::Invalid(format!("unsupported dtype {:?}", tensor.dtype))
    })?;
    if block == 0 || !elements.is_multiple_of(block) {
        return invalid(format!(
            "tensor shape {:?} is not block-aligned for {:?}",
            tensor.shape, tensor.dtype
        ));
    }
    elements
        .checked_div(block)
        .and_then(|units| units.checked_mul(bytes))
        .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("tensor byte count overflow".into()))
}

fn require_range(name: &str, tensor: &MetalTensor) -> Result<(), Qwen4ExpTextSessionError> {
    let alignment = match tensor.dtype {
        GgmlType::F32 | GgmlType::I32 => 4,
        _ => 2,
    };
    if !tensor.offset.is_multiple_of(alignment) {
        return invalid(format!(
            "{name} offset {} is not {alignment}-byte aligned",
            tensor.offset
        ));
    }
    let bytes = storage_bytes(tensor)?;
    let end = tensor
        .offset
        .checked_add(bytes)
        .ok_or_else(|| Qwen4ExpTextSessionError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range offset={} bytes={bytes} exceeds buffer={}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

fn require_read_only_weights(
    tensors: &[(&str, &MetalTensor)],
) -> Result<(), Qwen4ExpTextSessionError> {
    for (name, tensor) in tensors {
        if tensor.provenance() == MetalTensorProvenance::OwnedWritable {
            return invalid(format!("{name} must have read-only weight provenance"));
        }
    }
    Ok(())
}

fn require_same_device(
    ctx: &MetalContext,
    tensors: &[(&str, &MetalTensor)],
) -> Result<(), Qwen4ExpTextSessionError> {
    let expected = ctx.device.registryID();
    for (name, tensor) in tensors {
        let actual = tensor.buffer.device().registryID();
        if actual != expected {
            return invalid(format!(
                "{name} belongs to Metal device registry {actual}, expected {expected}"
            ));
        }
    }
    Ok(())
}

fn require_disjoint(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpTextSessionError> {
    for left in 0..tensors.len() {
        let left_bytes = storage_bytes(tensors[left].1)?;
        for right in left + 1..tensors.len() {
            if Retained::as_ptr(&tensors[left].1.buffer)
                != Retained::as_ptr(&tensors[right].1.buffer)
            {
                continue;
            }
            let right_bytes = storage_bytes(tensors[right].1)?;
            let left_end = tensors[left].1.offset.saturating_add(left_bytes);
            let right_end = tensors[right].1.offset.saturating_add(right_bytes);
            if tensors[left].1.offset < right_end && tensors[right].1.offset < left_end {
                return invalid(format!("{} overlaps {}", tensors[left].0, tensors[right].0));
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct AllocationBuilder {
    allocations: Vec<(String, u64)>,
}

impl AllocationBuilder {
    fn bytes(
        &mut self,
        name: impl Into<String>,
        bytes: usize,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        let bytes = u64::try_from(bytes).map_err(|_| {
            Qwen4ExpTextSessionError::Invalid("allocation byte count exceeds u64".into())
        })?;
        if bytes == 0 {
            return invalid("session allocation must be nonzero");
        }
        self.allocations.push((name.into(), bytes));
        Ok(())
    }

    fn f32(
        &mut self,
        name: impl Into<String>,
        elements: usize,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        self.bytes(
            name,
            elements.checked_mul(4).ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid("F32 allocation overflow".into())
            })?,
        )
    }

    fn i32(
        &mut self,
        name: impl Into<String>,
        elements: usize,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        self.f32(name, elements)
    }
}

fn add_packed_allocations(
    builder: &mut AllocationBuilder,
    geometry: &Qwen4ExpTextSessionMetalGeometry,
    capacity: usize,
    selected_capable: bool,
) -> Result<(), Qwen4ExpTextSessionError> {
    let zero_one = geometry.zero_one();
    let hidden = geometry.hidden_size();
    let hyper = geometry.hyper_width();
    let ple = zero_one.ple();
    let gdn = zero_one.layer_zero().gdn();
    let moe = zero_one.layer_zero().moe();
    let product = |name: &str, factors: &[usize]| {
        factors
            .iter()
            .try_fold(1_usize, |value, &factor| value.checked_mul(factor))
            .ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid(format!(
                    "packed {name} allocation element count overflow"
                ))
            })
    };

    builder.i32("packed.token_ids", capacity)?;
    builder.f32(
        "packed.embedding",
        product("embedding", &[hidden, capacity])?,
    )?;
    builder.bytes(
        "packed.ple_rows",
        product("PLE rows", &[ple.packed_staging_bytes(), capacity])?,
    )?;
    builder.i32(
        "packed.ple_local_row_ids",
        product("PLE local row IDs", &[ple.head_count(), capacity])?,
    )?;
    builder.f32(
        "packed.ple_embedding",
        product("PLE embedding", &[hidden, capacity])?,
    )?;
    builder.f32(
        "packed.hyper_residual",
        product("hyper residual", &[hyper, capacity])?,
    )?;
    builder.f32("packed.bridge", product("bridge", &[hidden, capacity])?)?;

    for (name, width) in [
        ("normalized", hyper),
        ("low", geometry.low_rank()),
        ("raw_gate", hyper),
        ("mixed", hidden),
        ("injection", geometry.branch_count()),
    ] {
        builder.f32(
            format!("packed.residual.{name}"),
            product("residual scratch", &[width, capacity])?,
        )?;
    }
    for (name, width) in [
        ("qkv", gdn.conv_width()),
        ("gate", gdn.value_width()),
        ("beta", gdn.value_heads()),
        ("alpha", gdn.value_heads()),
        ("decay", gdn.value_heads()),
        ("query", gdn.key_width()),
        ("key", gdn.key_width()),
        ("value", gdn.value_width()),
        ("query_norm", gdn.key_width()),
        ("key_norm", gdn.key_width()),
        ("recurrent", gdn.value_width()),
        ("normalized", gdn.value_width()),
        ("output", hidden),
    ] {
        builder.f32(
            format!("packed.gdn.{name}"),
            product("GDN scratch", &[width, capacity])?,
        )?;
    }
    for (name, width) in [
        ("key", hyper),
        ("value", hidden),
        ("key_norm", hyper),
        ("query_norm", hyper),
        ("gated", hyper),
        ("conv_input", hyper),
        ("conv_raw", hyper),
        ("output", hyper),
    ] {
        builder.f32(
            format!("packed.ple.{name}"),
            product("PLE scratch", &[width, capacity])?,
        )?;
    }

    let moe_allocations = [
        ("router_logits", vec![moe.expert_count(), capacity]),
        ("topk_ids", vec![moe.experts_per_token(), capacity]),
        ("topk_weights", vec![moe.experts_per_token(), capacity]),
        ("shared_scale", vec![capacity]),
        ("route_counts", vec![moe.expert_count()]),
        ("route_slots", vec![moe.expert_count(), capacity]),
        (
            "routed_inner",
            vec![
                moe.routed_intermediate_size(),
                moe.experts_per_token(),
                capacity,
            ],
        ),
        (
            "routed_expert_output",
            vec![hidden, moe.experts_per_token(), capacity],
        ),
        (
            "shared_gate_projection",
            vec![moe.shared_intermediate_size(), capacity],
        ),
        (
            "shared_up_projection",
            vec![moe.shared_intermediate_size(), capacity],
        ),
        (
            "shared_inner",
            vec![moe.shared_intermediate_size(), capacity],
        ),
        ("shared_output", vec![hidden, capacity]),
        ("output", vec![hidden, capacity]),
    ];
    for (name, factors) in moe_allocations {
        let elements = product("MoE scratch", &factors)?;
        if matches!(name, "topk_ids" | "route_counts" | "route_slots") {
            builder.i32(format!("packed.moe.{name}"), elements)?;
        } else {
            builder.f32(format!("packed.moe.{name}"), elements)?;
        }
    }
    for (index, bytes) in geometry
        .packed_qsa_geometry()?
        .packed_scratch_logical_allocations(capacity)?
        .into_iter()
        .enumerate()
    {
        builder.bytes(format!("packed.qsa.{index}"), bytes)?;
    }
    if selected_capable {
        let names = [
            "index_query_raw",
            "index_query",
            "scores",
            "visible_blocks",
            "selected_blocks",
            "selected_count",
            "selector_status",
            "token_ids",
            "attention_logits",
        ];
        let allocations = geometry
            .packed_qsa_geometry()?
            .selected_packed_scratch_logical_allocations(capacity)?;
        if allocations.len() != names.len() {
            return invalid(format!(
                "selected packed QSA allocation count {} differs from {} names",
                allocations.len(),
                names.len()
            ));
        }
        for (name, bytes) in names.into_iter().zip(allocations) {
            builder.bytes(format!("packed.qsa.selected.{name}"), bytes)?;
        }
    }
    Ok(())
}

fn add_zero_one_allocations(
    builder: &mut AllocationBuilder,
    geometry: Qwen4ExpLayersZeroOneMetalGeometry,
) -> Result<(), Qwen4ExpTextSessionError> {
    let layer = geometry.layer_zero();
    builder.i32("zero_one.layer_zero.token_id", 1)?;
    builder.f32("zero_one.layer_zero.embedding", geometry.hidden_size())?;
    builder.f32("zero_one.layer_zero.hyper_residual", geometry.hyper_width())?;
    builder.f32("zero_one.layer_zero.mixer_output", geometry.hidden_size())?;
    builder.f32("zero_one.layer_zero.moe_output", geometry.hidden_size())?;
    add_residual_allocations(
        builder,
        "zero_one.layer_zero.residual",
        geometry.branch_count(),
        geometry.hidden_size(),
        geometry.low_rank(),
    )?;
    add_gdn_allocations(builder, "zero_one.layer_zero.gdn", layer.gdn())?;
    add_moe_allocations(builder, "zero_one.layer_zero.moe", layer.moe())?;

    builder.f32("zero_one.hyper_residual", geometry.hyper_width())?;
    add_ple_allocations(builder, "zero_one.ple", geometry.ple())?;
    builder.f32("zero_one.mixer_output", geometry.hidden_size())?;
    builder.f32("zero_one.moe_output", geometry.hidden_size())?;
    add_residual_allocations(
        builder,
        "zero_one.residual",
        geometry.branch_count(),
        geometry.hidden_size(),
        geometry.low_rank(),
    )?;
    add_gdn_allocations(builder, "zero_one.layer_one_gdn", layer.gdn())?;
    add_moe_allocations(builder, "zero_one.layer_one_moe", layer.moe())
}

fn add_post_ple_allocations(
    builder: &mut AllocationBuilder,
    geometry: Qwen4ExpPostPleBlockMetalGeometry,
) -> Result<(), Qwen4ExpTextSessionError> {
    let prefix = format!("block.{}", geometry.layer());
    builder.f32(format!("{prefix}.mixer_output"), geometry.hidden_size())?;
    builder.f32(format!("{prefix}.moe_output"), geometry.hidden_size())?;
    add_residual_allocations(
        builder,
        &format!("{prefix}.residual"),
        geometry.branch_count(),
        geometry.hidden_size(),
        geometry.low_rank(),
    )?;
    match geometry.mixer() {
        Qwen4ExpPostPleMixerMetalGeometry::GatedDeltaNet(gdn) => {
            add_gdn_allocations(builder, &format!("{prefix}.gdn"), gdn)?;
        }
        Qwen4ExpPostPleMixerMetalGeometry::QwenSparseAttention(qsa) => {
            for (index, bytes) in qsa.workspace_logical_allocations()?.into_iter().enumerate() {
                builder.bytes(format!("{prefix}.qsa.{index}"), bytes)?;
            }
        }
    }
    add_moe_allocations(builder, &format!("{prefix}.moe"), geometry.moe())
}

fn add_residual_allocations(
    builder: &mut AllocationBuilder,
    prefix: &str,
    branches: usize,
    hidden: usize,
    rank: usize,
) -> Result<(), Qwen4ExpTextSessionError> {
    let hyper = branches
        .checked_mul(hidden)
        .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("residual width overflow".into()))?;
    builder.f32(format!("{prefix}.normalized"), hyper)?;
    builder.f32(format!("{prefix}.low"), rank)?;
    builder.f32(format!("{prefix}.raw_gate"), hyper)?;
    builder.f32(format!("{prefix}.mixed"), hidden)?;
    builder.f32(format!("{prefix}.injection"), branches)
}

fn add_gdn_allocations(
    builder: &mut AllocationBuilder,
    prefix: &str,
    geometry: GatedDeltaNetMetalGeometry,
) -> Result<(), Qwen4ExpTextSessionError> {
    for (name, elements) in [
        ("conv_state", geometry.conv_state_elements()),
        ("delta_state", geometry.delta_state_elements()),
        ("qkv", geometry.conv_width()),
        ("qkv_conv", geometry.conv_width()),
        ("gate", geometry.value_width()),
        ("beta", geometry.value_heads()),
        ("alpha", geometry.value_heads()),
        ("decay", geometry.value_heads()),
        ("query_norm", geometry.key_width()),
        ("key_norm", geometry.key_width()),
        ("recurrent", geometry.value_width()),
        ("normalized", geometry.value_width()),
        ("output", geometry.hidden_size()),
    ] {
        builder.f32(format!("{prefix}.{name}"), elements)?;
    }
    Ok(())
}

fn add_moe_allocations(
    builder: &mut AllocationBuilder,
    prefix: &str,
    geometry: Qwen4ExpMoeMetalGeometry,
) -> Result<(), Qwen4ExpTextSessionError> {
    builder.f32(format!("{prefix}.router_logits"), geometry.expert_count())?;
    builder.i32(format!("{prefix}.topk_ids"), geometry.experts_per_token())?;
    builder.f32(
        format!("{prefix}.topk_weights"),
        geometry.experts_per_token(),
    )?;
    builder.f32(format!("{prefix}.shared_gate"), 1)?;
    builder.f32(
        format!("{prefix}.routed_inner"),
        geometry
            .experts_per_token()
            .checked_mul(geometry.routed_intermediate_size())
            .ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid("routed inner size overflow".into())
            })?,
    )?;
    builder.f32(
        format!("{prefix}.routed_expert_output"),
        geometry
            .experts_per_token()
            .checked_mul(geometry.hidden_size())
            .ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid("routed output size overflow".into())
            })?,
    )?;
    builder.f32(
        format!("{prefix}.shared_inner"),
        geometry.shared_intermediate_size(),
    )?;
    builder.f32(format!("{prefix}.shared_output"), geometry.hidden_size())?;
    builder.f32(format!("{prefix}.output"), geometry.hidden_size())
}

fn add_ple_allocations(
    builder: &mut AllocationBuilder,
    prefix: &str,
    geometry: crate::qwen4exp_ple_metal::Qwen4ExpPleMetalGeometry,
) -> Result<(), Qwen4ExpTextSessionError> {
    builder.bytes(
        format!("{prefix}.packed_rows"),
        geometry.packed_staging_bytes(),
    )?;
    builder.i32(format!("{prefix}.local_row_ids"), geometry.head_count())?;
    builder.f32(format!("{prefix}.embedding"), geometry.hidden_size())?;
    builder.f32(format!("{prefix}.key"), geometry.hyper_width())?;
    builder.f32(format!("{prefix}.value"), geometry.hidden_size())?;
    builder.f32(format!("{prefix}.key_norm"), geometry.hyper_width())?;
    builder.f32(format!("{prefix}.query_norm"), geometry.hyper_width())?;
    builder.f32(format!("{prefix}.gated"), geometry.hyper_width())?;
    builder.f32(format!("{prefix}.conv_input"), geometry.hyper_width())?;
    builder.f32(
        format!("{prefix}.conv_state"),
        geometry
            .hyper_width()
            .checked_mul(geometry.history_len())
            .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("PLE state size overflow".into()))?,
    )?;
    builder.f32(format!("{prefix}.conv_raw"), geometry.hyper_width())?;
    builder.f32(format!("{prefix}.output"), geometry.hyper_width())
}

fn price_session_allocation(
    name: &str,
    logical_bytes: u64,
    priced: MetalBufferSizeAndAlign,
    host_page_size: u64,
    max_buffer_length: u64,
) -> Result<(u64, u64), Qwen4ExpTextSessionError> {
    crate::metal::price_shared_buffer_upper(
        logical_bytes,
        priced,
        host_page_size,
        max_buffer_length,
    )
    .map(|priced| (priced.priced_upper_bytes, priced.alignment))
    .map_err(|error| {
        Qwen4ExpTextSessionError::Invalid(format!("session allocation {name:?} {error}"))
    })
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpTextSessionError> {
    Err(Qwen4ExpTextSessionError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::metal::MetalMemoryAdmissionReason;
    use crate::qwen4exp_layer_zero::Qwen4ExpResidualMetalWeights;
    use crate::qwen4exp_layers_zero_one::encode_qwen4exp_layers_zero_one;
    use crate::qwen4exp_moe::Qwen4ExpMoeMetalWeights;
    use crate::qwen4exp_post_ple_block::Qwen4ExpPostPleMixerMetalWeights;
    use crate::qwen4exp_residency::Qwen4ExpMetalWeightPlan;
    use crate::qwen4exp_runtime::{Qwen4ExpSessionCapacity, forward_qwen4exp_text_token_sync};
    use objc2_metal::MTLCommandQueue;

    fn context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(ctx) => Some(ctx),
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => None,
            Err(error) => panic!("Metal initialization failed: {error}"),
        }
    }

    #[test]
    fn sampled_profiles_reject_command_mutating_diagnostics() {
        let Some(ctx) = context() else { return };
        crate::metal::dispatch_census_begin();
        let composition = crate::qwen4exp_composition_trace::Qwen4ExpCompositionTraceBanks::new(
            &ctx, 2_051, 8, 1,
        )
        .unwrap();
        let (result, records) = crate::qwen4exp_composition_trace::with_qwen4exp_composition_trace(
            &composition,
            || reject_active_sampled_diagnostics("packed"),
        );
        let error = result.unwrap_err();
        assert!(error.to_string().contains("composition tracing"));
        assert!(records.is_empty());

        let config = Qwen4ExpConfig::flash_next_reference();
        let qsa = crate::qwen4exp_qsa::Qwen4ExpQsaDecisionCaptureBanks::new(
            &ctx, &config, 2_052, 2_051, 1,
        )
        .unwrap();
        let (result, records) =
            crate::qwen4exp_qsa::with_qwen4exp_qsa_decision_capture(&qsa, || {
                reject_active_sampled_diagnostics("packed")
            });
        let error = result.unwrap_err();
        assert!(error.to_string().contains("QSA decision capture"));
        assert!(records.is_empty());

        let (result, records) = crate::qwen4exp_metal::with_qwen4exp_hc_packed_projection_override(
            crate::qwen4exp_metal::Qwen4ExpHcPackedProjectionArm::WideF32Down,
            2_051,
            2_048,
            || reject_active_sampled_diagnostics("packed"),
        );
        let error = result.unwrap_err();
        assert!(error.to_string().contains("HC projection overrides"));
        assert!(records.is_empty());
        assert!(crate::metal::dispatch_census_take().is_empty());
    }

    fn signals(available: u64) -> MetalMemorySignals {
        MetalMemorySignals {
            recommended_max_bytes: available,
            current_allocated_bytes: 0,
            process_limit_remaining_bytes: Some(available),
        }
    }

    #[test]
    #[ignore = "production lease; model-free optional QSA pricing and admission boundary"]
    fn qualified_defaults_optional_scratch_plan() {
        let _lease =
            crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
        let ctx = MetalContext::new().unwrap();
        let config = Qwen4ExpConfig::flash_next_reference();
        for packed in [false, true] {
            let geometry = Qwen4ExpTextSessionMetalGeometry::from_config(&config, 2588).unwrap();
            let memory = Qwen4ExpTextSessionMemoryPlan::for_geometry_with_options(
                &ctx,
                &geometry,
                0,
                packed.then_some(128),
                packed,
            )
            .unwrap();
            let baseline = memory.clone();
            assert!(!baseline.split_decode_enabled());
            let plan = Qwen4ExpTextSessionPlan {
                geometry,
                memory,
                device_registry_id: ctx.device.registryID(),
            };
            let plan = plan.with_split_decode(&ctx, false).unwrap();
            assert_eq!(plan.memory_plan(), &baseline);
            let plan = plan.with_split_decode(&ctx, true).unwrap();
            let optimized = plan.memory_plan();
            assert_eq!(
                optimized.session_logical_bytes() - baseline.session_logical_bytes(),
                1_585_152
            );
            assert_eq!(
                optimized.allocations().len(),
                baseline.allocations().len() + 1
            );
            assert_eq!(
                optimized
                    .allocations()
                    .iter()
                    .filter(|a| a.name == "session.qsa_split")
                    .count(),
                1
            );
            let available = baseline.priced_upper_bytes_for_sessions(1).unwrap()
                + QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES;
            assert!(
                baseline
                    .admission_before_residency(signals(available), 1)
                    .unwrap()
                    .admitted
            );
            assert!(
                !optimized
                    .admission_before_residency(signals(available), 1)
                    .unwrap()
                    .admitted
            );
            assert!(
                optimized
                    .admission_before_residency(
                        signals(
                            available + optimized.session_priced_upper_bytes()
                                - baseline.session_priced_upper_bytes()
                        ),
                        1
                    )
                    .unwrap()
                    .admitted
            );
            let fallback = plan.with_split_decode(&ctx, false).unwrap();
            assert_eq!(fallback.memory_plan(), &baseline);
            assert!(
                fallback
                    .memory_plan()
                    .admission_before_residency(signals(available), 1)
                    .unwrap()
                    .admitted
            );
            assert!(
                !fallback
                    .memory_plan()
                    .admission_before_residency(signals(available - 1), 1)
                    .unwrap()
                    .admitted
            );
        }
    }

    fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
        assert_eq!(tensor.dtype, GgmlType::F32);
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>();
            std::slice::from_raw_parts(source, tensor.n_elements() as usize).to_vec()
        }
    }

    #[test]
    fn post_layer_hyper_probe_validates_native_contract() {
        let Some(ctx) = context() else {
            return;
        };
        let config = Qwen4ExpConfig::flash_next_reference();
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, 1)
            .unwrap()
            .qsa_physical_capacity();
        let geometry = Qwen4ExpTextSessionMetalGeometry::from_config(&config, capacity).unwrap();
        let width = geometry.hyper_width();
        let hyper = MetalTensor::zeros_f32(&ctx, vec![width as u64]).unwrap();
        let capture = MetalTensor::zeros_f32(&ctx, vec![width as u64]).unwrap();
        let direction = MetalTensor::zeros_f32(&ctx, vec![width as u64]).unwrap();
        let valid = Qwen4ExpPostLayerHyperProbe {
            layer: 1,
            capture: &capture,
            fixed_add: Some(Qwen4ExpFixedHyperAddTensor {
                direction: &direction,
                coefficient: 0.25,
            }),
        };
        validate_post_layer_hyper_probe(&ctx, &geometry, &hyper, &valid).unwrap();

        for layer in [0, geometry.layer_count() as u32] {
            let invalid = Qwen4ExpPostLayerHyperProbe { layer, ..valid };
            let error = validate_post_layer_hyper_probe(&ctx, &geometry, &hyper, &invalid)
                .unwrap_err()
                .to_string();
            assert!(error.contains("outside 1.."), "{error}");
        }
        for coefficient in [0.0, f32::NAN, f32::INFINITY] {
            let invalid = Qwen4ExpPostLayerHyperProbe {
                fixed_add: Some(Qwen4ExpFixedHyperAddTensor {
                    direction: &direction,
                    coefficient,
                }),
                ..valid
            };
            let error = validate_post_layer_hyper_probe(&ctx, &geometry, &hyper, &invalid)
                .unwrap_err()
                .to_string();
            assert!(error.contains("finite and nonzero"), "{error}");
        }

        let short_direction = MetalTensor::zeros_f32(&ctx, vec![(width - 1) as u64]).unwrap();
        let invalid = Qwen4ExpPostLayerHyperProbe {
            fixed_add: Some(Qwen4ExpFixedHyperAddTensor {
                direction: &short_direction,
                coefficient: 0.25,
            }),
            ..valid
        };
        let error = validate_post_layer_hyper_probe(&ctx, &geometry, &hyper, &invalid)
            .unwrap_err()
            .to_string();
        assert!(error.contains("direction must be F32"), "{error}");

        let i32_direction = MetalTensor::zeros_i32(&ctx, vec![width as u64]).unwrap();
        let invalid = Qwen4ExpPostLayerHyperProbe {
            fixed_add: Some(Qwen4ExpFixedHyperAddTensor {
                direction: &i32_direction,
                coefficient: 0.25,
            }),
            ..valid
        };
        let error = validate_post_layer_hyper_probe(&ctx, &geometry, &hyper, &invalid)
            .unwrap_err()
            .to_string();
        assert!(error.contains("direction must be F32"), "{error}");

        let short_capture = MetalTensor::zeros_f32(&ctx, vec![(width - 1) as u64]).unwrap();
        let invalid = Qwen4ExpPostLayerHyperProbe {
            capture: &short_capture,
            ..valid
        };
        let error = validate_post_layer_hyper_probe(&ctx, &geometry, &hyper, &invalid)
            .unwrap_err()
            .to_string();
        assert!(error.contains("capture must be F32"), "{error}");

        let aliased = Qwen4ExpPostLayerHyperProbe {
            capture: &hyper,
            ..valid
        };
        let error = validate_post_layer_hyper_probe(&ctx, &geometry, &hyper, &aliased)
            .unwrap_err()
            .to_string();
        assert!(error.contains("overlaps"), "{error}");
    }

    #[test]
    fn post_layer_hyper_probe_applies_fixed_add_before_capture() {
        let Some(ctx) = context() else {
            return;
        };
        let config = Qwen4ExpConfig::flash_next_reference();
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, 1)
            .unwrap()
            .qsa_physical_capacity();
        let geometry = Qwen4ExpTextSessionMetalGeometry::from_config(&config, capacity).unwrap();
        let width = geometry.hyper_width();
        let initial = vec![1.0_f32; width];
        let mut direction_values = vec![0.0_f32; width];
        direction_values[17] = 1.0;
        let hyper = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&initial),
            vec![width as u64],
            GgmlType::F32,
        )
        .unwrap();
        let direction = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&direction_values),
            vec![width as u64],
            GgmlType::F32,
        )
        .unwrap();
        let capture = MetalTensor::zeros_f32(&ctx, vec![width as u64]).unwrap();
        let probe = Qwen4ExpPostLayerHyperProbe {
            layer: 23,
            capture: &capture,
            fixed_add: Some(Qwen4ExpFixedHyperAddTensor {
                direction: &direction,
                coefficient: 0.25,
            }),
        };
        validate_post_layer_hyper_probe(&ctx, &geometry, &hyper, &probe).unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_post_layer_hyper_probe(&ctx, &encoder, 23, &hyper, Some(&probe)).unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();

        let mut expected = initial;
        expected[17] = 1.25;
        assert_close("fixed-add hyper state", &read_f32(&hyper), &expected, 0.0);
        assert_close(
            "post-add hyper capture",
            &read_f32(&capture),
            &expected,
            0.0,
        );
    }

    fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        let mut max_error = 0.0_f32;
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(
                actual.is_finite() && expected.is_finite(),
                "{label} is non-finite"
            );
            max_error = max_error.max((actual - expected).abs());
        }
        assert!(
            max_error <= tolerance,
            "{label} max error {max_error} exceeds {tolerance}"
        );
    }

    fn assert_similarity(
        label: &str,
        actual: &[f32],
        expected: &[f32],
        maximum_relative_rms: f64,
        minimum_cosine: f64,
        maximum_absolute: f32,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        let mut error_square = 0.0_f64;
        let mut expected_square = 0.0_f64;
        let mut actual_square = 0.0_f64;
        let mut dot = 0.0_f64;
        let mut observed_max = 0.0_f32;
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite() && expected.is_finite(), "{label} finite");
            let error = (actual - expected) as f64;
            error_square += error * error;
            expected_square += expected as f64 * expected as f64;
            actual_square += actual as f64 * actual as f64;
            dot += actual as f64 * expected as f64;
            observed_max = observed_max.max((actual - expected).abs());
        }
        let relative_rms = (error_square / expected_square.max(1e-30)).sqrt();
        let cosine = dot / (actual_square * expected_square).sqrt().max(1e-30);
        eprintln!(
            "[{label}] relative_rms={relative_rms:.3e} cosine={cosine:.9} max_abs={observed_max:.3e}"
        );
        assert!(
            relative_rms <= maximum_relative_rms,
            "{label} relative RMS {relative_rms}"
        );
        assert!(cosine >= minimum_cosine, "{label} cosine {cosine}");
        assert!(
            observed_max <= maximum_absolute,
            "{label} max abs {observed_max}"
        );
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .unwrap()
    }

    fn assert_binding(actual: &MetalTensor, expected: &MetalTensor) {
        assert_eq!(
            Retained::as_ptr(&actual.buffer),
            Retained::as_ptr(&expected.buffer)
        );
        assert_eq!(actual.offset, expected.offset);
        assert_eq!(actual.shape, expected.shape);
        assert_eq!(actual.dtype, expected.dtype);
        assert_eq!(actual.provenance(), expected.provenance());
    }

    fn assert_named_binding(actual: &MetalTensor, resident: &Qwen4ExpMetalWeights, name: &str) {
        assert_binding(actual, resident.require_tensor(name).unwrap());
    }

    fn assert_residual_bindings(
        weights: Qwen4ExpResidualMetalWeights<'_>,
        resident: &Qwen4ExpMetalWeights,
        layer: u32,
        phase: &str,
    ) {
        for (actual, suffix) in [
            (weights.read.norm, format!("hc_{phase}_norm.weight")),
            (weights.read.down, format!("hc_{phase}_down.weight")),
            (weights.read.up, format!("hc_{phase}_up.weight")),
            (weights.inject, format!("hc_{phase}_inject.weight")),
        ] {
            assert_named_binding(actual, resident, &format!("blk.{layer}.{suffix}"));
        }
    }

    fn assert_moe_bindings(
        weights: Qwen4ExpMoeMetalWeights<'_>,
        resident: &Qwen4ExpMetalWeights,
        layer: u32,
    ) {
        for (actual, suffix) in [
            (weights.router, "ffn_gate_inp.weight"),
            (weights.routed_gate, "ffn_gate_exps.weight"),
            (weights.routed_up, "ffn_up_exps.weight"),
            (weights.routed_down, "ffn_down_exps.weight"),
            (weights.shared_router, "ffn_gate_inp_shexp.weight"),
            (weights.shared_gate, "ffn_gate_shexp.weight"),
            (weights.shared_up, "ffn_up_shexp.weight"),
            (weights.shared_down, "ffn_down_shexp.weight"),
        ] {
            assert_named_binding(actual, resident, &format!("blk.{layer}.{suffix}"));
        }
    }

    fn assert_post_ple_bindings(
        weights: Qwen4ExpPostPleBlockMetalWeights<'_>,
        resident: &Qwen4ExpMetalWeights,
        layer: u32,
    ) {
        assert_eq!(weights.geometry.layer(), layer);
        assert_residual_bindings(weights.attention_residual, resident, layer, "attn");
        match weights.mixer {
            Qwen4ExpPostPleMixerMetalWeights::GatedDeltaNet(weights) => {
                for (actual, suffix) in [
                    (weights.qkv, "attn_qkv.weight"),
                    (weights.gate, "attn_gate.weight"),
                    (weights.beta, "ssm_beta.weight"),
                    (weights.alpha, "ssm_alpha.weight"),
                    (weights.a, "ssm_a"),
                    (weights.dt_bias, "ssm_dt.bias"),
                    (weights.conv, "ssm_conv1d.weight"),
                    (weights.norm, "ssm_norm.weight"),
                    (weights.output, "ssm_out.weight"),
                ] {
                    assert_named_binding(actual, resident, &format!("blk.{layer}.{suffix}"));
                }
            }
            Qwen4ExpPostPleMixerMetalWeights::QwenSparseAttention(weights) => {
                for (actual, suffix) in [
                    (weights.query, "attn_q.weight"),
                    (weights.key, "attn_k.weight"),
                    (weights.value, "attn_v.weight"),
                    (weights.output, "attn_output.weight"),
                    (weights.query_norm, "attn_q_norm.weight"),
                    (weights.key_norm, "attn_k_norm.weight"),
                    (weights.index_query, "indexer.q_proj.weight"),
                    (weights.index_key, "indexer.k_proj.weight"),
                    (weights.index_query_norm, "indexer.q_norm.weight"),
                    (weights.index_key_norm, "indexer.k_norm.weight"),
                ] {
                    assert_named_binding(actual, resident, &format!("blk.{layer}.{suffix}"));
                }
            }
        }
        assert_residual_bindings(weights.ffn_residual, resident, layer, "ffn");
        assert_moe_bindings(weights.moe, resident, layer);
    }

    #[test]
    fn reference_memory_inventory_matches_all_workspace_allocations() {
        let Some(ctx) = context() else { return };
        let config = Qwen4ExpConfig::flash_next_reference();
        for capacity in [4, config.context_length as usize] {
            let geometry =
                Qwen4ExpTextSessionMetalGeometry::from_config(&config, capacity).unwrap();
            let plan = Qwen4ExpTextSessionMemoryPlan::for_geometry_with_residency_bytes(
                &ctx, &geometry, 0,
            )
            .unwrap();
            let expected = 143_207_764_u64 + 25_356_u64 * capacity as u64;
            assert_eq!(geometry.layer_count(), 48);
            assert_eq!(geometry.qsa_layers().len(), 12);
            assert_eq!(
                geometry.qsa_layers(),
                &[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
            );
            assert_eq!(plan.allocations().len(), 1_523);
            assert_eq!(plan.packed_prefill_capacity(), None);
            assert!(
                plan.allocations()
                    .iter()
                    .all(|allocation| !allocation.name.starts_with("packed."))
            );
            assert_eq!(plan.session_logical_bytes(), expected);
            assert!(plan.session_priced_upper_bytes() >= expected);
            assert!(
                plan.allocations()
                    .iter()
                    .all(|allocation| allocation.logical_bytes > 0
                        && allocation.priced_bytes >= allocation.logical_bytes
                        && allocation.alignment.is_power_of_two())
            );
        }
    }

    #[test]
    fn reference_packed_memory_inventory_is_explicit_and_exact() {
        let Some(ctx) = context() else { return };
        let config = Qwen4ExpConfig::flash_next_reference();
        for capacity in [4, config.context_length as usize] {
            let geometry =
                Qwen4ExpTextSessionMetalGeometry::from_config(&config, capacity).unwrap();
            let plan = Qwen4ExpTextSessionMemoryPlan::for_geometry_with_packed_residency_bytes(
                &ctx, &geometry, 0,
            )
            .unwrap();
            let packed_capacity = geometry.packed_capacity().unwrap();
            let (
                packed_expected,
                packed_selected_capable,
                total_allocations,
                expected_packed_allocations,
            ) = match packed_capacity {
                4 => (4_477_600_u64, false, 1_578, 55),
                2_048 => (1_917_948_672_u64, true, 1_587, 64),
                other => panic!("unexpected packed capacity {other}"),
            };
            let scalar_expected = 143_207_764_u64 + 25_356_u64 * capacity as u64;
            let packed_allocations = plan
                .allocations()
                .iter()
                .filter(|allocation| allocation.name.starts_with("packed."))
                .collect::<Vec<_>>();
            assert_eq!(plan.packed_prefill_capacity(), Some(packed_capacity));
            assert_eq!(plan.packed_selected_capable(), packed_selected_capable);
            assert_eq!(plan.allocations().len(), total_allocations);
            assert_eq!(packed_allocations.len(), expected_packed_allocations);
            assert_eq!(
                packed_allocations
                    .iter()
                    .map(|allocation| allocation.logical_bytes)
                    .sum::<u64>(),
                packed_expected
            );
            assert_eq!(
                plan.session_logical_bytes(),
                scalar_expected + packed_expected
            );
        }
    }

    #[test]
    fn packed_memory_inventory_uses_the_prompt_extent() {
        let Some(ctx) = context() else { return };
        let config = Qwen4ExpConfig::flash_next_reference();
        let geometry = Qwen4ExpTextSessionMetalGeometry::from_config(&config, 2_052).unwrap();
        assert_eq!(
            Qwen4ExpTextSessionPlan::packed_prefill_options_for_prompt(&geometry, 2_051).unwrap(),
            (2_048, false)
        );
        assert_eq!(
            Qwen4ExpTextSessionPlan::packed_prefill_options_for_prompt(&geometry, 2_052).unwrap(),
            (2_048, true)
        );
        let prompt_plan = Qwen4ExpTextSessionMemoryPlan::for_geometry_with_options(
            &ctx,
            &geometry,
            0,
            Some(18),
            false,
        )
        .unwrap();
        let maximum_plan = Qwen4ExpTextSessionMemoryPlan::for_geometry_with_options(
            &ctx,
            &geometry,
            0,
            Some(2_048),
            true,
        )
        .unwrap();
        assert_eq!(prompt_plan.packed_prefill_capacity(), Some(18));
        assert!(!prompt_plan.packed_selected_capable());
        assert!(maximum_plan.packed_selected_capable());
        assert_eq!(
            maximum_plan.allocations().len(),
            prompt_plan.allocations().len() + 9
        );
        assert!(
            prompt_plan
                .allocations()
                .iter()
                .all(|allocation| !allocation.name.starts_with("packed.qsa.selected."))
        );
        assert_eq!(
            maximum_plan
                .allocations()
                .iter()
                .filter(|allocation| allocation.name.starts_with("packed.qsa.selected."))
                .count(),
            9
        );
        assert!(prompt_plan.session_logical_bytes() < maximum_plan.session_logical_bytes());
        assert!(
            Qwen4ExpTextSessionMemoryPlan::for_geometry_with_options(
                &ctx,
                &geometry,
                0,
                Some(1),
                false,
            )
            .is_err()
        );
        assert!(
            Qwen4ExpTextSessionMemoryPlan::for_geometry_with_options(
                &ctx,
                &geometry,
                0,
                Some(2_049),
                false,
            )
            .is_err()
        );
        assert!(
            Qwen4ExpTextSessionMemoryPlan::for_geometry_with_options(
                &ctx, &geometry, 0, None, true,
            )
            .is_err()
        );
    }

    #[test]
    fn multi_session_admission_accounts_for_residency_and_reserve() {
        let Some(ctx) = context() else { return };
        let config = Qwen4ExpConfig::flash_next_reference();
        let geometry = Qwen4ExpTextSessionMetalGeometry::from_config(&config, 4).unwrap();
        let residency = 91_u64;
        let plan = Qwen4ExpTextSessionMemoryPlan::for_geometry_with_residency_bytes(
            &ctx, &geometry, residency,
        )
        .unwrap();
        let sessions = 2;
        let scratch = plan.priced_upper_bytes_for_sessions(sessions).unwrap();
        assert_eq!(
            scratch,
            residency + plan.session_priced_upper_bytes() * sessions as u64
        );
        let required = scratch + QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES;
        let admitted = plan
            .admission_before_residency(signals(required), sessions)
            .unwrap();
        assert!(admitted.admitted);
        assert_eq!(
            admitted.reason,
            MetalMemoryAdmissionReason::AdmittedWithProcessBudget
        );
        assert_eq!(admitted.required_bytes, Some(required));

        let denied = plan
            .admission_before_residency(signals(required - 1), sessions)
            .unwrap();
        assert!(!denied.admitted);
        assert_eq!(denied.reason, MetalMemoryAdmissionReason::BothInsufficient);
        assert!(plan.priced_upper_bytes_for_sessions(0).is_err());
        assert!(plan.priced_upper_bytes_for_sessions(usize::MAX).is_err());
    }

    #[test]
    fn session_reconciliation_tolerates_concurrent_release_but_rejects_overrun() {
        let Some(ctx) = context() else { return };
        let config = Qwen4ExpConfig::flash_next_reference();
        let geometry = Qwen4ExpTextSessionMetalGeometry::from_config(&config, 4).unwrap();
        let plan =
            Qwen4ExpTextSessionMemoryPlan::for_geometry_with_residency_bytes(&ctx, &geometry, 0)
                .unwrap();
        assert_eq!(plan.reconcile_session(7, 7).unwrap(), 0);
        assert_eq!(plan.reconcile_session(8, 7).unwrap(), 0);
        assert!(
            plan.reconcile_session(0, plan.session_priced_upper_bytes() + 1)
                .is_err()
        );
    }

    #[test]
    fn session_allocation_pricing_fails_closed_at_arithmetic_boundaries() {
        assert_eq!(
            price_session_allocation(
                "boundary",
                4_097,
                MetalBufferSizeAndAlign {
                    size: 4_352,
                    alignment: 256,
                },
                4_096,
                4_097,
            )
            .unwrap(),
            (8_192, 4_096)
        );
        for result in [
            price_session_allocation(
                "zero",
                0,
                MetalBufferSizeAndAlign {
                    size: 1,
                    alignment: 1,
                },
                4_096,
                u64::MAX,
            ),
            price_session_allocation(
                "too-large",
                4_098,
                MetalBufferSizeAndAlign {
                    size: 4_098,
                    alignment: 2,
                },
                4_096,
                4_097,
            ),
            price_session_allocation(
                "underpriced",
                4_097,
                MetalBufferSizeAndAlign {
                    size: 4_096,
                    alignment: 256,
                },
                4_096,
                u64::MAX,
            ),
            price_session_allocation(
                "bad-alignment",
                1,
                MetalBufferSizeAndAlign {
                    size: 1,
                    alignment: 3,
                },
                4_096,
                u64::MAX,
            ),
            price_session_allocation(
                "rounding-overflow",
                u64::MAX,
                MetalBufferSizeAndAlign {
                    size: u64::MAX,
                    alignment: 2,
                },
                4_096,
                u64::MAX,
            ),
        ] {
            assert!(result.is_err());
        }
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_TEXT_SESSION_GGUF to the pinned full release"]
    fn released_full_text_session_matches_separate_layer_commands() {
        let path = crate::test_fixtures::QWEN4EXP_Q3_K_XL.required();
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let ctx = MetalContext::new().expect("initialize Metal");
        let weight_plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
        let session_plan = Qwen4ExpTextSessionPlan::for_config(
            &ctx,
            weight_plan.config(),
            4,
            weight_plan.memory_plan(),
        )
        .unwrap();
        let aggregate = session_plan
            .memory_plan()
            .admission_before_residency(ctx.memory_signals(), 2)
            .unwrap();
        assert!(
            aggregate.admitted,
            "two-session aggregate admission failed: {aggregate:?}"
        );
        let admitted_weights = weight_plan.admit(ctx.memory_signals()).unwrap();
        let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted_weights).unwrap();
        let resident = realized.weights();
        let weights = Qwen4ExpTextSessionMetalWeights::bind(resident, 4).unwrap();
        let table = resident.ple_source().bind(&gguf).unwrap();
        assert_eq!(weights.post_ple.len(), 46);
        assert_eq!(weights.geometry.qsa_layers().len(), 12);
        assert_binding(
            weights.final_read.norm,
            resident.require_tensor("output_hc_norm.weight").unwrap(),
        );
        assert_binding(
            weights.final_read.down,
            resident.require_tensor("output_hc_down.weight").unwrap(),
        );
        assert_binding(
            weights.final_read.up,
            resident.require_tensor("output_hc_up.weight").unwrap(),
        );
        assert_binding(
            weights.output,
            resident.require_tensor("output.weight").unwrap(),
        );
        assert_eq!(weights.final_read.norm.dtype, GgmlType::F32);
        assert_eq!(weights.final_read.down.dtype, GgmlType::Q8_0);
        assert_eq!(weights.final_read.up.dtype, GgmlType::Q8_0);
        assert_eq!(weights.output.dtype, GgmlType::Q6_K);
        for (offset, block) in weights.post_ple.iter().enumerate() {
            let layer = offset as u32 + 2;
            assert_eq!(block.mixer.geometry(), block.geometry.mixer());
            assert_post_ple_bindings(*block, resident, layer);
        }

        let geometry = weights.geometry.clone();
        let mut control_zero_one =
            Qwen4ExpLayersZeroOneMetalWorkspace::new(&ctx, geometry.zero_one()).unwrap();
        let mut control_blocks = geometry
            .post_ple()
            .iter()
            .map(|geometry| Qwen4ExpPostPleBlockMetalWorkspace::new(&ctx, *geometry).unwrap())
            .collect::<Vec<_>>();
        let control_hyper =
            MetalTensor::zeros_f32(&ctx, vec![geometry.hyper_width() as u64]).unwrap();
        let mut control_final = GatedResidualMetalScratch::new(
            &ctx,
            geometry.branch_count(),
            geometry.hidden_size(),
            geometry.low_rank(),
        )
        .unwrap();
        let control_logits =
            MetalTensor::zeros_f32(&ctx, vec![geometry.vocab_size() as u64]).unwrap();
        let admitted_session = session_plan
            .admit_after_residency(resident, ctx.memory_signals())
            .unwrap();
        let mut integrated =
            Qwen4ExpTextSessionMetalWorkspace::from_admitted(&ctx, admitted_session).unwrap();

        for (position, token) in [35_u32, 201, 17, 89].into_iter().enumerate() {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let read = encode_qwen4exp_layers_zero_one(
                &ctx,
                &encoder,
                token,
                position as u64,
                table,
                weights.zero_one,
                &mut control_zero_one,
            )
            .unwrap();
            read.output()
                .encode_copy_to(&ctx, &encoder, &control_hyper)
                .unwrap();
            drop(read);
            encoder.end();
            command.commit();
            control_zero_one.release_after().unwrap();

            for (block_weights, block_workspace) in weights.post_ple.iter().zip(&mut control_blocks)
            {
                let command = ctx.queue.commandBuffer().unwrap();
                let encoder = KernelEncoder::begin(&command);
                let read = encode_qwen4exp_post_ple_block(
                    &ctx,
                    &encoder,
                    position,
                    &control_hyper,
                    *block_weights,
                    block_workspace,
                )
                .unwrap();
                drop(read);
                encoder.end();
                command.commit();
                block_workspace.release_after().unwrap();
            }

            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let final_read = encode_final_gated_residual_mix(
                &ctx,
                &encoder,
                &control_hyper,
                geometry.eps(),
                weights.final_read,
                &mut control_final,
            )
            .unwrap();
            encode_mat_vec_dispatch(
                &ctx,
                &encoder,
                weights.output,
                final_read.mixed(),
                &control_logits,
                geometry.hidden_size(),
                geometry.vocab_size(),
            )
            .unwrap();
            drop(final_read);
            encoder.end();
            command.commit();
            control_final.release_after().unwrap();

            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let pending = encode_qwen4exp_text_token(
                &ctx,
                &encoder,
                token,
                position,
                table,
                &weights,
                &mut integrated,
            )
            .unwrap();
            drop(pending);
            encoder.end();
            command.commit();
            integrated.release_after().unwrap();

            let expected_hidden = read_f32(control_final.mixed_tensor());
            let actual_hidden = read_f32(integrated.final_hidden_tensor());
            let expected_logits = read_f32(&control_logits);
            let actual_logits = integrated.logits().unwrap().to_vec();
            assert_close(
                "released full-stack final hidden",
                &actual_hidden,
                &expected_hidden,
                2e-4,
            );
            assert_close(
                "released full-stack logits",
                &actual_logits,
                &expected_logits,
                5e-3,
            );
            assert_eq!(argmax(&actual_logits), argmax(&expected_logits));
        }
        assert_eq!(integrated.committed_length(), 4);
        assert!(
            integrated
                .qsa_committed_lengths()
                .iter()
                .all(|(_, length)| *length == 4)
        );
        assert!(
            control_blocks
                .iter()
                .filter_map(|workspace| workspace.mixer_committed_length())
                .all(|length| length == 4)
        );
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_TEXT_SESSION_GGUF to the pinned full release"]
    fn released_packed_text_session_matches_scalar_and_continues() {
        const CAPACITY: usize = 12;
        let path = crate::test_fixtures::QWEN4EXP_Q3_K_XL.required();
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let ctx = MetalContext::new().expect("initialize Metal");
        let weight_plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
        let session_plan = Qwen4ExpTextSessionPlan::for_config_with_packed_prefill(
            &ctx,
            weight_plan.config(),
            CAPACITY,
            weight_plan.memory_plan(),
        )
        .unwrap();
        let aggregate = session_plan
            .memory_plan()
            .admission_before_residency(ctx.memory_signals(), 2)
            .unwrap();
        assert!(
            aggregate.admitted,
            "two-session aggregate admission failed: {aggregate:?}"
        );
        let admitted_weights = weight_plan.admit(ctx.memory_signals()).unwrap();
        let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted_weights).unwrap();
        let resident = realized.weights();
        let weights = Qwen4ExpTextSessionMetalWeights::bind(resident, CAPACITY).unwrap();
        let table = resident.ple_source().bind(&gguf).unwrap();
        let geometry = weights.geometry.clone();
        let mut scalar =
            Qwen4ExpTextSessionMetalWorkspace::new_for_tests(&ctx, geometry.clone()).unwrap();
        let mut packed =
            Qwen4ExpTextSessionMetalWorkspace::new_for_tests_with_packed(&ctx, geometry.clone())
                .unwrap();
        let tokens = [35_u32, 201, 17, 89, 5, 42, 7, 11];

        let mut scalar_logits = Vec::new();
        for &token in &tokens {
            scalar_logits =
                forward_qwen4exp_text_token_sync(&ctx, token, table, &weights, &mut scalar)
                    .unwrap()
                    .to_vec();
        }
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let pending =
            encode_qwen4exp_text_packed(&ctx, &encoder, &tokens, 0, table, &weights, &mut packed)
                .unwrap();
        drop(pending);
        encoder.end();
        command.commit();
        packed.release_after().unwrap();
        let packed_logits = packed.logits().unwrap().to_vec();
        assert_similarity(
            "released packed final hidden",
            &read_f32(packed.final_hidden_tensor()),
            &read_f32(scalar.final_hidden_tensor()),
            2e-2,
            0.999_85,
            1.2,
        );
        assert_similarity(
            "released packed logits",
            &packed_logits,
            &scalar_logits,
            1.5e-2,
            0.999_9,
            0.2,
        );
        assert_eq!(argmax(&packed_logits), argmax(&scalar_logits));
        assert_eq!(packed.committed_length(), tokens.len());
        assert!(
            packed
                .qsa_committed_lengths()
                .iter()
                .all(|(_, length)| *length == tokens.len())
        );

        let continuation = 13;
        let scalar_logits =
            forward_qwen4exp_text_token_sync(&ctx, continuation, table, &weights, &mut scalar)
                .unwrap()
                .to_vec();
        let packed_logits =
            forward_qwen4exp_text_token_sync(&ctx, continuation, table, &weights, &mut packed)
                .unwrap()
                .to_vec();
        assert_similarity(
            "released packed-to-scalar hidden",
            &read_f32(packed.final_hidden_tensor()),
            &read_f32(scalar.final_hidden_tensor()),
            2e-2,
            0.999_85,
            1.0,
        );
        assert_similarity(
            "released packed-to-scalar logits",
            &packed_logits,
            &scalar_logits,
            1.5e-2,
            0.999_9,
            0.2,
        );
        assert_eq!(argmax(&packed_logits), argmax(&scalar_logits));
        assert_eq!(packed.committed_length(), tokens.len() + 1);
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_TEXT_SESSION_GGUF to the pinned full release"]
    fn released_packed_2048_crosses_into_scalar_selection_without_migration() {
        const PACKED_TOKENS: usize = 2_048;
        const CAPACITY: usize = 2_052;
        let path = crate::test_fixtures::QWEN4EXP_Q3_K_XL.required();
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let ctx = MetalContext::new().expect("initialize Metal");
        let weight_plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
        let session_plan = Qwen4ExpTextSessionPlan::for_config_with_packed_prefill(
            &ctx,
            weight_plan.config(),
            CAPACITY,
            weight_plan.memory_plan(),
        )
        .unwrap();
        assert_eq!(
            session_plan.memory_plan().packed_prefill_capacity(),
            Some(PACKED_TOKENS)
        );
        let aggregate = session_plan
            .memory_plan()
            .admission_before_residency(ctx.memory_signals(), 1)
            .unwrap();
        assert!(
            aggregate.admitted,
            "packed aggregate admission failed: {aggregate:?}"
        );
        let admitted_weights = weight_plan.admit(ctx.memory_signals()).unwrap();
        let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted_weights).unwrap();
        let resident = realized.weights();
        let weights = Qwen4ExpTextSessionMetalWeights::bind(resident, CAPACITY).unwrap();
        let table = resident.ple_source().bind(&gguf).unwrap();
        let admitted_session = session_plan
            .admit_after_residency(resident, ctx.memory_signals())
            .unwrap();
        let mut workspace =
            Qwen4ExpTextSessionMetalWorkspace::from_admitted(&ctx, admitted_session).unwrap();
        let state_before = workspace.persistent_state_tensors();
        let source = [35_u32, 201, 17, 89, 5, 42, 7, 11];
        let tokens = (0..PACKED_TOKENS)
            .map(|index| source[index % source.len()])
            .collect::<Vec<_>>();

        let wall_started = std::time::Instant::now();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let pending = encode_qwen4exp_text_packed(
            &ctx,
            &encoder,
            &tokens,
            0,
            table,
            &weights,
            &mut workspace,
        )
        .unwrap();
        drop(pending);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        let wall_ms = wall_started.elapsed().as_secs_f64() * 1e3;
        let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        eprintln!(
            "released packed N={PACKED_TOKENS}: wall_ms={wall_ms:.3} gpu_ms={gpu_ms:.3} tok/s={:.3}",
            PACKED_TOKENS as f64 / (wall_ms / 1e3)
        );
        assert_eq!(workspace.committed_length(), PACKED_TOKENS);
        assert!(
            workspace
                .qsa_committed_lengths()
                .iter()
                .all(|(_, length)| *length == PACKED_TOKENS)
        );
        assert!(
            workspace
                .logits()
                .unwrap()
                .as_slice()
                .iter()
                .all(|v| v.is_finite())
        );
        let state_after = workspace.persistent_state_tensors();
        assert_eq!(state_before.len(), state_after.len());
        for (before, after) in state_before.iter().zip(&state_after) {
            assert_binding(after, before);
        }

        for token in [13_u32, 29, 53, 97] {
            let logits =
                forward_qwen4exp_text_token_sync(&ctx, token, table, &weights, &mut workspace)
                    .unwrap();
            assert!(logits.as_slice().iter().all(|value| value.is_finite()));
        }
        assert_eq!(workspace.committed_length(), CAPACITY);
        assert!(
            workspace
                .qsa_committed_lengths()
                .iter()
                .all(|(_, length)| *length == CAPACITY)
        );
        let state_after_selection = workspace.persistent_state_tensors();
        for (before, after) in state_before.iter().zip(&state_after_selection) {
            assert_binding(after, before);
        }
    }
}
