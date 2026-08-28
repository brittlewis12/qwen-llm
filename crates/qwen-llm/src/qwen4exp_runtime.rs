//! Synchronous single-session runtime for Qwen3.8-Flash-Next text generation.

use crate::gguf::GgufFile;
use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalMemoryAdmission, MetalTimestampSampleBuffer,
};
use crate::qwen4exp::{MixerKind, Qwen4ExpConfig, Qwen4ExpError};
use crate::qwen4exp_ple::PleIq4NlTable;
use crate::qwen4exp_profile::{
    QWEN4EXP_PACKED_PROFILE_GDN_LAYER, QWEN4EXP_PACKED_PROFILE_QSA_LAYER,
    QWEN4EXP_PACKED_PROFILE_SAMPLE_CAPACITY, Qwen4ExpPackedProfileRecorder,
    Qwen4ExpPackedProfileSpan, packed_stage_sample_count,
};
pub use crate::qwen4exp_profile::{Qwen4ExpPackedProfileLabel, Qwen4ExpPackedProfileScope};
use crate::qwen4exp_residency::{
    Qwen4ExpMetalWeightPlan, Qwen4ExpMetalWeights, Qwen4ExpResidencyError,
};
use crate::qwen4exp_text_session::{
    Qwen4ExpCompletedLogits, Qwen4ExpPackedEncodeCpuTiming, Qwen4ExpTextSessionError,
    Qwen4ExpTextSessionMetalWeights, Qwen4ExpTextSessionMetalWorkspace, Qwen4ExpTextSessionPending,
    Qwen4ExpTextSessionPlan, Qwen4ExpTextSessionReleaseTiming, encode_qwen4exp_text_packed,
    encode_qwen4exp_text_packed_layer_sampled, encode_qwen4exp_text_packed_profiled,
    encode_qwen4exp_text_token, encode_qwen4exp_text_token_layer_sampled,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLDevice};
use std::ops::Range;
use std::time::Instant;

crate::env_flag!(
    default_off configured_qwen4exp_packed_selected_qsa_enabled,
    "QWEN4EXP_PACKED_SELECTED_QSA"
);

#[cfg(test)]
thread_local! {
    static QWEN4EXP_PACKED_SELECTED_QSA_OVERRIDE: std::cell::Cell<Option<bool>> = const {
        std::cell::Cell::new(None)
    };
}

fn qwen4exp_packed_selected_qsa_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = QWEN4EXP_PACKED_SELECTED_QSA_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    configured_qwen4exp_packed_selected_qsa_enabled()
}

#[cfg(test)]
struct Qwen4ExpPackedSelectedQsaOverride {
    previous: Option<bool>,
}

#[cfg(test)]
impl Qwen4ExpPackedSelectedQsaOverride {
    fn set(enabled: bool) -> Self {
        let previous = QWEN4EXP_PACKED_SELECTED_QSA_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(enabled));
            previous
        });
        Self { previous }
    }
}

#[cfg(test)]
impl Drop for Qwen4ExpPackedSelectedQsaOverride {
    fn drop(&mut self) {
        QWEN4EXP_PACKED_SELECTED_QSA_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpRuntimeError {
    #[error(transparent)]
    Config(#[from] Qwen4ExpError),
    #[error(transparent)]
    Residency(#[from] Qwen4ExpResidencyError),
    #[error(transparent)]
    Session(#[from] Qwen4ExpTextSessionError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(
        "Qwen3.8-Flash-Next prefill failed after committing {committed} of {requested} tokens; reset before retrying the full prompt: {source}"
    )]
    Prefill {
        committed: usize,
        requested: usize,
        #[source]
        source: Box<Qwen4ExpRuntimeError>,
    },
    #[error("Qwen3.8-Flash-Next prefill checkpoint failed: {0}")]
    Checkpoint(String),
    #[error("invalid Qwen3.8-Flash-Next runtime contract: {0}")]
    Invalid(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen4ExpSessionCapacity {
    forward_limit: usize,
    qsa_physical_capacity: usize,
}

impl Qwen4ExpSessionCapacity {
    pub fn for_forward_limit(
        config: &Qwen4ExpConfig,
        forward_limit: usize,
    ) -> Result<Self, Qwen4ExpRuntimeError> {
        config.validate()?;
        if forward_limit == 0 {
            return invalid("forward limit must be nonzero");
        }
        let context_length = config.context_length as usize;
        if forward_limit > context_length {
            return invalid(format!(
                "forward limit {forward_limit} exceeds model context {context_length}"
            ));
        }
        let alignment = config
            .compress_ratios
            .iter()
            .copied()
            .filter(|ratio| *ratio != 0)
            .try_fold(1_usize, |alignment, ratio| {
                checked_lcm(alignment, ratio as usize)
            })?;
        if alignment == 1 {
            return invalid("text runtime requires at least one compressed QSA layer");
        }
        let rounded = forward_limit
            .checked_add(alignment - 1)
            .map(|value| value / alignment * alignment)
            .ok_or_else(|| {
                Qwen4ExpRuntimeError::Invalid("QSA physical capacity overflow".into())
            })?;
        let qsa_physical_capacity = if rounded <= context_length {
            rounded
        } else if context_length.is_multiple_of(alignment) {
            context_length
        } else {
            return invalid(format!(
                "model context {context_length} cannot hold aligned QSA capacity for {forward_limit} forwards"
            ));
        };
        Ok(Self {
            forward_limit,
            qsa_physical_capacity,
        })
    }

    pub fn forward_limit(self) -> usize {
        self.forward_limit
    }

    pub fn qsa_physical_capacity(self) -> usize {
        self.qsa_physical_capacity
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpRuntimeAdmission {
    pub aggregate: MetalMemoryAdmission,
    pub weights: MetalMemoryAdmission,
    pub session: MetalMemoryAdmission,
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpTokenTiming {
    pub position: usize,
    pub encode_cpu_ms: f64,
    pub completion_wait_ms: f64,
    pub gpu_ms: Option<f64>,
    pub total_wall_ms: f64,
}

impl Qwen4ExpTokenTiming {
    pub fn outside_gpu_ms(self) -> Option<f64> {
        self.gpu_ms
            .map(|gpu_ms| (self.total_wall_ms - gpu_ms).max(0.0))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpPrefillTiming {
    pub token_count: usize,
    /// Total tokens encoded by all packed commands in this prefill.
    pub packed_token_count: usize,
    /// Whether any packed command crossed the QSA dense boundary.
    pub contains_selection: bool,
    pub command_count: usize,
    pub encode_cpu_ms: f64,
    pub completion_wait_ms: f64,
    pub gpu_ms: f64,
    pub gpu_samples: usize,
    pub total_wall_ms: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Qwen4ExpPrefillExecutionPlan {
    packed_ranges: Vec<Range<usize>>,
    packed_token_count: usize,
    scalar_start: usize,
    contains_selection: bool,
}

fn plan_qwen4exp_prefill_execution(
    token_count: usize,
    packed_capacity: Option<usize>,
    selected_enabled: bool,
    dense_end: usize,
) -> Result<Qwen4ExpPrefillExecutionPlan, Qwen4ExpRuntimeError> {
    if token_count == 0 {
        return invalid("prefill execution plan requires at least one token");
    }
    if dense_end == 0 {
        return invalid("prefill execution plan requires a nonzero QSA dense end");
    }
    if packed_capacity.is_none() && selected_enabled {
        return invalid("selected packed execution requires packed scratch");
    }
    let Some(packed_capacity) = packed_capacity else {
        return Ok(Qwen4ExpPrefillExecutionPlan {
            packed_ranges: Vec::new(),
            packed_token_count: 0,
            scalar_start: 0,
            contains_selection: false,
        });
    };
    if packed_capacity < 2 {
        return invalid(format!(
            "packed prefill capacity {packed_capacity} is smaller than two tokens"
        ));
    }
    let packed_end = if selected_enabled {
        token_count
    } else {
        token_count.min(dense_end)
    };
    let mut packed_ranges = Vec::with_capacity(packed_end.div_ceil(packed_capacity));
    let mut cursor = 0_usize;
    while packed_end - cursor >= 2 {
        let mut rows = (packed_end - cursor).min(packed_capacity);
        let proposed_end = cursor
            .checked_add(rows)
            .ok_or_else(|| Qwen4ExpRuntimeError::Invalid("packed prefill range overflow".into()))?;
        if selected_enabled && cursor < dense_end && proposed_end > dense_end {
            let dense_rows = dense_end - cursor;
            if dense_rows >= 2 {
                rows = dense_rows;
            }
        }
        let end = cursor
            .checked_add(rows)
            .ok_or_else(|| Qwen4ExpRuntimeError::Invalid("packed prefill range overflow".into()))?;
        packed_ranges.push(cursor..end);
        cursor = end;
    }
    let contains_selection = packed_ranges.iter().any(|range| range.end > dense_end);
    Ok(Qwen4ExpPrefillExecutionPlan {
        packed_ranges,
        packed_token_count: cursor,
        scalar_start: cursor,
        contains_selection,
    })
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Qwen4ExpPackedProfileEncodeTiming {
    pub preflight_ms: f64,
    pub stage_inputs_ms: f64,
    pub graph_encode_ms: f64,
    pub unattributed_ms: f64,
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Qwen4ExpPackedProfileCommandTiming {
    pub commit_return_ms: f64,
    pub root_wait_ms: f64,
    pub child_publication_ms: f64,
    pub root_publish_ms: f64,
    pub release_total_ms: f64,
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Qwen4ExpPackedProfileStageTiming {
    pub label: Qwen4ExpPackedProfileLabel,
    pub depth: usize,
    pub start_sample: usize,
    pub end_sample: usize,
    pub duration_ticks: u64,
    pub gpu_ms: f64,
    pub fraction_of_gpu: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Qwen4ExpPackedProfileSampling {
    DispatchBoundary,
    EncoderStage,
}

impl Qwen4ExpPackedProfileSampling {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DispatchBoundary => "dispatch_boundary",
            Self::EncoderStage => "encoder_stage",
        }
    }
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Qwen4ExpPackedPrefillProfile {
    pub token: Qwen4ExpTokenTiming,
    pub encode: Qwen4ExpPackedProfileEncodeTiming,
    pub command: Qwen4ExpPackedProfileCommandTiming,
    pub sampling: Qwen4ExpPackedProfileSampling,
    pub sampling_fallback: Option<String>,
    pub stages: Vec<Qwen4ExpPackedProfileStageTiming>,
    pub sample_count: usize,
    pub sampled_span_ticks: u64,
    pub raw_span_ms_assuming_ns: f64,
    pub raw_coverage_assuming_ns: f64,
}

#[derive(Clone, Debug, thiserror::Error)]
#[error("Qwen3.8-Flash-Next packed-profile telemetry unavailable: {detail}")]
pub struct Qwen4ExpPackedProfileError {
    detail: String,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Qwen4ExpPackedProfileOutcome {
    pub token: Qwen4ExpTokenTiming,
    pub encode: Qwen4ExpPackedProfileEncodeTiming,
    pub command: Qwen4ExpPackedProfileCommandTiming,
    pub sampling: Qwen4ExpPackedProfileSampling,
    pub sampling_fallback: Option<String>,
    pub profile: Result<Qwen4ExpPackedPrefillProfile, Qwen4ExpPackedProfileError>,
}

impl Qwen4ExpPrefillTiming {
    fn new(token_count: usize, packed_token_count: usize, contains_selection: bool) -> Self {
        Self {
            token_count,
            packed_token_count,
            contains_selection,
            command_count: 0,
            encode_cpu_ms: 0.0,
            completion_wait_ms: 0.0,
            gpu_ms: 0.0,
            gpu_samples: 0,
            total_wall_ms: 0.0,
        }
    }

    fn record(&mut self, timing: Qwen4ExpTokenTiming) {
        self.command_count += 1;
        self.encode_cpu_ms += timing.encode_cpu_ms;
        self.completion_wait_ms += timing.completion_wait_ms;
        self.total_wall_ms += timing.total_wall_ms;
        if let Some(gpu_ms) = timing.gpu_ms {
            self.gpu_ms += gpu_ms;
            self.gpu_samples += 1;
        }
    }

    pub fn complete_gpu_ms(self) -> Option<f64> {
        (self.gpu_samples == self.command_count).then_some(self.gpu_ms)
    }

    pub fn outside_gpu_ms(self) -> Option<f64> {
        self.complete_gpu_ms()
            .map(|gpu_ms| (self.total_wall_ms - gpu_ms).max(0.0))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen4ExpLayerStage {
    LayersZeroOne,
    PostPle { layer: u32, mixer: MixerKind },
    Tail,
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpLayerStageTiming {
    pub stage: Qwen4ExpLayerStage,
    pub duration_ticks: u64,
    pub gpu_ms: f64,
    pub fraction_of_gpu: f64,
}

#[derive(Clone, Debug)]
pub struct Qwen4ExpLayerProfile {
    pub token: Qwen4ExpTokenTiming,
    pub stages: Vec<Qwen4ExpLayerStageTiming>,
    pub encoder_boundary_ms: f64,
    pub sampled_span_ticks: u64,
}

#[derive(Clone, Debug, thiserror::Error)]
#[error("Qwen3.8-Flash-Next layer-profile telemetry unavailable: {detail}")]
pub struct Qwen4ExpLayerProfileError {
    detail: String,
}

#[derive(Clone, Debug)]
pub struct Qwen4ExpLayerProfileOutcome {
    pub token: Qwen4ExpTokenTiming,
    pub profile: Result<Qwen4ExpLayerProfile, Qwen4ExpLayerProfileError>,
}

pub struct Qwen4ExpLoadedModel<'gguf> {
    weights: Qwen4ExpMetalWeights,
    ple_table: PleIq4NlTable<'gguf>,
    workspace: Option<Qwen4ExpTextSessionMetalWorkspace>,
    capacity: Qwen4ExpSessionCapacity,
    admission: Qwen4ExpRuntimeAdmission,
    observed_weight_bytes: u64,
    device_registry_id: u64,
}

impl<'gguf> Qwen4ExpLoadedModel<'gguf> {
    pub fn load(
        ctx: &MetalContext,
        gguf: &'gguf GgufFile,
        capacity: Qwen4ExpSessionCapacity,
    ) -> Result<Self, Qwen4ExpRuntimeError> {
        Self::load_with_options(ctx, gguf, capacity, None)
    }

    /// Admit reusable packed scratch for a prompt of this extent. The extent
    /// is an allocation hint, not a later request-length lock; longer requests
    /// remain valid within the forward limit but may scalarize unadmitted
    /// selected-range rows.
    pub fn load_with_packed_prefill(
        ctx: &MetalContext,
        gguf: &'gguf GgufFile,
        capacity: Qwen4ExpSessionCapacity,
        prompt_tokens: usize,
    ) -> Result<Self, Qwen4ExpRuntimeError> {
        if prompt_tokens < 2 {
            return invalid("packed prefill requires at least two prompt tokens");
        }
        if prompt_tokens > capacity.forward_limit {
            return invalid(format!(
                "packed prompt length {prompt_tokens} exceeds forward limit {}",
                capacity.forward_limit
            ));
        }
        Self::load_with_options(ctx, gguf, capacity, Some(prompt_tokens))
    }

    fn load_with_options(
        ctx: &MetalContext,
        gguf: &'gguf GgufFile,
        capacity: Qwen4ExpSessionCapacity,
        packed_prefill_tokens: Option<usize>,
    ) -> Result<Self, Qwen4ExpRuntimeError> {
        let weight_plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(ctx, gguf)?;
        let expected_capacity = Qwen4ExpSessionCapacity::for_forward_limit(
            weight_plan.config(),
            capacity.forward_limit,
        )?;
        if expected_capacity != capacity {
            return invalid("session capacity differs from the released model geometry");
        }
        let session_plan = if let Some(prompt_tokens) = packed_prefill_tokens {
            Qwen4ExpTextSessionPlan::for_config_with_packed_prefill_tokens(
                ctx,
                weight_plan.config(),
                capacity.qsa_physical_capacity,
                weight_plan.memory_plan(),
                prompt_tokens,
            )?
        } else {
            Qwen4ExpTextSessionPlan::for_config(
                ctx,
                weight_plan.config(),
                capacity.qsa_physical_capacity,
                weight_plan.memory_plan(),
            )?
        };

        let _allocation_transaction = ctx.begin_allocation_transaction();
        let aggregate = session_plan
            .memory_plan()
            .admission_before_residency(ctx.memory_signals(), 1)?;
        if !aggregate.admitted {
            return invalid(format!(
                "combined weight and session admission denied: reason={} required={:?}",
                aggregate.reason.as_str(),
                aggregate.required_bytes
            ));
        }
        let admitted_weights = weight_plan.admit(ctx.memory_signals())?;
        let realized = Qwen4ExpMetalWeights::realize(ctx, gguf, admitted_weights)?;
        let weight_admission = realized.admission();
        let observed_weight_bytes = realized.observed_allocation_delta();
        let weights = realized.into_weights();
        let ple_table = weights.ple_source().bind(gguf)?;
        let admitted_session =
            session_plan.admit_after_residency(&weights, ctx.memory_signals())?;
        let workspace = Qwen4ExpTextSessionMetalWorkspace::from_admitted(ctx, admitted_session)?;
        let session_admission = workspace.admission();

        Ok(Self {
            weights,
            ple_table,
            workspace: Some(workspace),
            capacity,
            admission: Qwen4ExpRuntimeAdmission {
                aggregate,
                weights: weight_admission,
                session: session_admission,
            },
            observed_weight_bytes,
            device_registry_id: ctx.device.registryID(),
        })
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        self.weights.config()
    }

    pub fn capacity(&self) -> Qwen4ExpSessionCapacity {
        self.capacity
    }

    pub fn admission(&self) -> Qwen4ExpRuntimeAdmission {
        self.admission
    }

    pub fn observed_weight_bytes(&self) -> u64 {
        self.observed_weight_bytes
    }

    pub fn observed_session_bytes(&self) -> u64 {
        self.workspace.as_ref().map_or(
            0,
            Qwen4ExpTextSessionMetalWorkspace::observed_allocation_delta,
        )
    }

    pub fn packed_prefill_capacity(&self) -> Option<usize> {
        self.workspace
            .as_ref()
            .and_then(Qwen4ExpTextSessionMetalWorkspace::packed_prefill_capacity)
    }

    pub fn packed_selected_capable(&self) -> bool {
        self.workspace
            .as_ref()
            .is_some_and(Qwen4ExpTextSessionMetalWorkspace::packed_selected_capable)
    }

    pub fn packed_selected_requested(&self) -> bool {
        qwen4exp_packed_selected_qsa_enabled()
    }

    pub fn packed_selected_active(&self) -> bool {
        self.packed_selected_requested() && self.packed_selected_capable()
    }

    pub fn create_runner<'ctx, 'model>(
        &'model mut self,
        ctx: &'ctx MetalContext,
    ) -> Result<Qwen4ExpTextRunner<'ctx, 'model, 'gguf>, Qwen4ExpRuntimeError> {
        if ctx.device.registryID() != self.device_registry_id {
            return invalid(format!(
                "loaded model belongs to Metal device registry {}, runner context is {}",
                self.device_registry_id,
                ctx.device.registryID()
            ));
        }
        let weights = Qwen4ExpTextSessionMetalWeights::bind(
            &self.weights,
            self.capacity.qsa_physical_capacity,
        )?;
        let workspace = self.workspace.take().ok_or_else(|| {
            Qwen4ExpRuntimeError::Invalid("loaded model session was consumed".into())
        })?;
        Ok(Qwen4ExpTextRunner {
            ctx,
            weights,
            ple_table: self.ple_table,
            workspace,
            capacity: self.capacity,
            last_token_timing: None,
            last_prefill_timing: None,
        })
    }
}

pub struct Qwen4ExpTextRunner<'ctx, 'model, 'gguf> {
    ctx: &'ctx MetalContext,
    weights: Qwen4ExpTextSessionMetalWeights<'model>,
    ple_table: PleIq4NlTable<'gguf>,
    workspace: Qwen4ExpTextSessionMetalWorkspace,
    capacity: Qwen4ExpSessionCapacity,
    last_token_timing: Option<Qwen4ExpTokenTiming>,
    last_prefill_timing: Option<Qwen4ExpPrefillTiming>,
}

impl Qwen4ExpTextRunner<'_, '_, '_> {
    pub fn capacity(&self) -> Qwen4ExpSessionCapacity {
        self.capacity
    }

    pub fn next_position(&self) -> usize {
        self.workspace.committed_length()
    }

    pub fn remaining_forwards(&self) -> usize {
        self.capacity
            .forward_limit
            .saturating_sub(self.next_position())
    }

    pub fn logits(&self) -> Result<Qwen4ExpCompletedLogits<'_>, Qwen4ExpRuntimeError> {
        Ok(self.workspace.logits()?)
    }

    pub fn last_token_timing(&self) -> Option<Qwen4ExpTokenTiming> {
        self.last_token_timing
    }

    pub fn last_prefill_timing(&self) -> Option<Qwen4ExpPrefillTiming> {
        self.last_prefill_timing
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
    ) -> Result<Qwen4ExpCompletedLogits<'_>, Qwen4ExpRuntimeError> {
        self.validate_token_id(token_id)?;
        if self.next_position() >= self.capacity.forward_limit {
            return invalid(format!(
                "logical forward limit {} is exhausted",
                self.capacity.forward_limit
            ));
        }
        let timing = execute_qwen4exp_text_token_sync(
            self.ctx,
            token_id,
            self.ple_table,
            &self.weights,
            &mut self.workspace,
        )?;
        self.last_token_timing = Some(timing);
        Ok(self.workspace.logits()?)
    }

    pub fn forward_token_layer_profiled(
        &mut self,
        token_id: u32,
    ) -> Result<Qwen4ExpLayerProfileOutcome, Qwen4ExpRuntimeError> {
        self.validate_token_id(token_id)?;
        if self.next_position() >= self.capacity.forward_limit {
            return invalid(format!(
                "logical forward limit {} is exhausted",
                self.capacity.forward_limit
            ));
        }
        let outcome = execute_qwen4exp_text_token_layer_profiled_sync(
            self.ctx,
            token_id,
            self.ple_table,
            &self.weights,
            &mut self.workspace,
        )?;
        self.last_token_timing = Some(outcome.token);
        Ok(outcome)
    }

    pub fn prefill(
        &mut self,
        token_ids: &[u32],
    ) -> Result<Qwen4ExpCompletedLogits<'_>, Qwen4ExpRuntimeError> {
        self.prefill_with_command_checkpoint(token_ids, || Ok(()))
    }

    pub fn prefill_with_command_checkpoint<F>(
        &mut self,
        token_ids: &[u32],
        mut checkpoint: F,
    ) -> Result<Qwen4ExpCompletedLogits<'_>, Qwen4ExpRuntimeError>
    where
        F: FnMut() -> Result<(), Qwen4ExpRuntimeError>,
    {
        self.last_prefill_timing = None;
        self.validate_prefill_request(token_ids)?;
        let start = self.next_position();
        let selected_enabled =
            self.workspace.packed_selected_capable() && qwen4exp_packed_selected_qsa_enabled();
        let plan = plan_qwen4exp_prefill_execution(
            token_ids.len(),
            self.workspace.packed_prefill_capacity(),
            selected_enabled,
            self.packed_qsa_dense_end()?,
        )?;
        debug_assert!(!plan.contains_selection || selected_enabled);
        let mut prefill_timing = Qwen4ExpPrefillTiming::new(
            token_ids.len(),
            plan.packed_token_count,
            plan.contains_selection,
        );
        for range in &plan.packed_ranges {
            self.run_prefill_checkpoint(start, token_ids.len(), &mut checkpoint)?;
            match execute_qwen4exp_text_packed_sync(
                self.ctx,
                &token_ids[range.clone()],
                self.ple_table,
                &self.weights,
                &mut self.workspace,
            ) {
                Ok(timing) => {
                    if self.next_position() != range.end {
                        return Err(Qwen4ExpRuntimeError::Prefill {
                            committed: self.next_position().saturating_sub(start),
                            requested: token_ids.len(),
                            source: Box::new(Qwen4ExpRuntimeError::Invalid(format!(
                                "packed command published position {}, expected {}",
                                self.next_position(),
                                range.end
                            ))),
                        });
                    }
                    self.last_token_timing = None;
                    prefill_timing.record(timing);
                }
                Err(source) => {
                    return Err(Qwen4ExpRuntimeError::Prefill {
                        committed: self.next_position().saturating_sub(start),
                        requested: token_ids.len(),
                        source: Box::new(source),
                    });
                }
            }
        }
        for (index, &token_id) in token_ids[plan.scalar_start..].iter().enumerate() {
            self.run_prefill_checkpoint(start, token_ids.len(), &mut checkpoint)?;
            match execute_qwen4exp_text_token_sync(
                self.ctx,
                token_id,
                self.ple_table,
                &self.weights,
                &mut self.workspace,
            ) {
                Ok(timing) => {
                    let expected = plan.scalar_start + index + 1;
                    if self.next_position() != expected {
                        return Err(Qwen4ExpRuntimeError::Prefill {
                            committed: self.next_position().saturating_sub(start),
                            requested: token_ids.len(),
                            source: Box::new(Qwen4ExpRuntimeError::Invalid(format!(
                                "scalar command published position {}, expected {expected}",
                                self.next_position()
                            ))),
                        });
                    }
                    self.last_token_timing = Some(timing);
                    prefill_timing.record(timing);
                }
                Err(source) => {
                    return Err(Qwen4ExpRuntimeError::Prefill {
                        committed: self.next_position().saturating_sub(start),
                        requested: token_ids.len(),
                        source: Box::new(source),
                    });
                }
            }
        }
        self.last_prefill_timing = Some(prefill_timing);
        self.logits()
    }

    pub fn prefill_packed_profiled(
        &mut self,
        token_ids: &[u32],
    ) -> Result<Qwen4ExpPackedProfileOutcome, Qwen4ExpRuntimeError> {
        self.last_prefill_timing = None;
        self.validate_prefill_request(token_ids)?;
        let packed_capacity = self.workspace.packed_prefill_capacity().ok_or_else(|| {
            Qwen4ExpRuntimeError::Invalid("packed prefill was not admitted for this runner".into())
        })?;
        if token_ids.len() < 2 || token_ids.len() > packed_capacity {
            return invalid(format!(
                "profiled packed prompt length {} is outside 2..={packed_capacity}",
                token_ids.len()
            ));
        }
        let start = self.next_position();
        let outcome = match execute_qwen4exp_text_packed_profiled_sync(
            self.ctx,
            token_ids,
            self.ple_table,
            &self.weights,
            &mut self.workspace,
        ) {
            Ok(outcome) => outcome,
            Err(source) => {
                return Err(Qwen4ExpRuntimeError::Prefill {
                    committed: self.next_position().saturating_sub(start),
                    requested: token_ids.len(),
                    source: Box::new(source),
                });
            }
        };
        self.last_token_timing = None;
        let mut timing = Qwen4ExpPrefillTiming::new(token_ids.len(), token_ids.len(), false);
        timing.record(outcome.token);
        self.last_prefill_timing = Some(timing);
        Ok(outcome)
    }

    fn validate_prefill_request(&self, token_ids: &[u32]) -> Result<(), Qwen4ExpRuntimeError> {
        if self.next_position() != 0 {
            return invalid(format!(
                "prefill requires a reset session at position zero, got position {}",
                self.next_position()
            ));
        }
        if token_ids.is_empty() {
            return invalid("prompt token sequence must be nonempty");
        }
        if token_ids.len() > self.remaining_forwards() {
            return invalid(format!(
                "prompt requires {} forwards but only {} remain",
                token_ids.len(),
                self.remaining_forwards()
            ));
        }
        for (index, &token_id) in token_ids.iter().enumerate() {
            self.validate_token_id(token_id).map_err(|source| {
                Qwen4ExpRuntimeError::Invalid(format!(
                    "prompt token {index} is invalid before prefill: {source}"
                ))
            })?;
        }
        Ok(())
    }

    fn packed_qsa_dense_end(&self) -> Result<usize, Qwen4ExpRuntimeError> {
        self.weights
            .geometry
            .post_ple()
            .iter()
            .filter_map(|block| block.mixer().qsa().map(|qsa| qsa.output_width()))
            .min()
            .ok_or_else(|| {
                Qwen4ExpRuntimeError::Invalid(
                    "packed prefill requires at least one QSA layer".into(),
                )
            })
    }

    fn run_prefill_checkpoint<F>(
        &self,
        start: usize,
        requested: usize,
        checkpoint: &mut F,
    ) -> Result<(), Qwen4ExpRuntimeError>
    where
        F: FnMut() -> Result<(), Qwen4ExpRuntimeError>,
    {
        if let Err(source) = checkpoint() {
            let committed = self.next_position().saturating_sub(start);
            if committed == 0 {
                return Err(source);
            }
            return Err(Qwen4ExpRuntimeError::Prefill {
                committed,
                requested,
                source: Box::new(source),
            });
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpRuntimeError> {
        self.workspace.reset()?;
        self.last_token_timing = None;
        self.last_prefill_timing = None;
        Ok(())
    }

    fn validate_token_id(&self, token_id: u32) -> Result<(), Qwen4ExpRuntimeError> {
        let vocab_size = self.weights.geometry.vocab_size();
        if token_id as usize >= vocab_size {
            return invalid(format!(
                "token ID {token_id} is outside vocabulary {vocab_size}"
            ));
        }
        Ok(())
    }
}

pub fn forward_qwen4exp_text_token_sync<'a>(
    ctx: &MetalContext,
    token_id: u32,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
) -> Result<Qwen4ExpCompletedLogits<'a>, Qwen4ExpRuntimeError> {
    execute_qwen4exp_text_token_sync(ctx, token_id, table, weights, workspace)?;
    Ok(workspace.logits()?)
}

fn execute_qwen4exp_text_packed_profiled_sync(
    ctx: &MetalContext,
    token_ids: &[u32],
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
) -> Result<Qwen4ExpPackedProfileOutcome, Qwen4ExpRuntimeError> {
    let start_position = workspace.committed_length();
    let position = start_position
        .checked_add(token_ids.len())
        .and_then(|end| end.checked_sub(1))
        .ok_or_else(|| {
            Qwen4ExpRuntimeError::Invalid(
                "profiled packed prefill position range is empty or overflows".into(),
            )
        })?;
    let stage_sample_count = packed_stage_sample_count(weights.post_ple.len())?;
    let (samples, sampling, sampling_fallback) =
        match ctx.timestamp_dispatch_sample_buffer(QWEN4EXP_PACKED_PROFILE_SAMPLE_CAPACITY) {
            Ok(samples) => (
                samples,
                Qwen4ExpPackedProfileSampling::DispatchBoundary,
                None,
            ),
            Err(error) => (
                ctx.timestamp_sample_buffer(stage_sample_count)?,
                Qwen4ExpPackedProfileSampling::EncoderStage,
                Some(error.to_string()),
            ),
        };
    let wall_started = Instant::now();
    let command = ctx.queue.commandBuffer().ok_or_else(|| {
        Qwen4ExpRuntimeError::Invalid("Metal command queue returned no command buffer".into())
    })?;
    let mut encode_detail = Qwen4ExpPackedEncodeCpuTiming::default();
    let encoded = match sampling {
        Qwen4ExpPackedProfileSampling::DispatchBoundary => {
            encode_qwen4exp_text_packed_dispatch_profiled(
                ctx,
                &command,
                &samples,
                token_ids,
                start_position,
                table,
                weights,
                workspace,
                &mut encode_detail,
            )
        }
        Qwen4ExpPackedProfileSampling::EncoderStage => encode_qwen4exp_text_packed_stage_profiled(
            ctx,
            &command,
            &samples,
            token_ids,
            start_position,
            table,
            weights,
            workspace,
            &mut encode_detail,
        ),
    };
    let (pending, sample_count, spans) = match encoded {
        Ok(recorded) => recorded,
        Err(error) => {
            let abandon = unsafe { workspace.abandon_uncommitted() };
            drop(command);
            if let Err(abandon_error) = abandon {
                return invalid(format!(
                    "profiled packed prefill encode failed ({error}); abandoning its command also failed ({abandon_error})"
                ));
            }
            return Err(error);
        }
    };
    drop(pending);
    let encode_cpu_ms = wall_started.elapsed().as_secs_f64() * 1e3;
    let wait_started = Instant::now();
    let commit_started = Instant::now();
    command.commit();
    let commit_return_ms = commit_started.elapsed().as_secs_f64() * 1e3;
    let release = workspace.release_after_timed()?;
    let completion_wait_ms = wait_started.elapsed().as_secs_f64() * 1e3;
    let gpu_start = command.GPUStartTime();
    let gpu_end = command.GPUEndTime();
    let gpu_ms =
        (gpu_start.is_finite() && gpu_end.is_finite() && gpu_start > 0.0 && gpu_end > gpu_start)
            .then_some((gpu_end - gpu_start) * 1e3);
    let token = Qwen4ExpTokenTiming {
        position,
        encode_cpu_ms,
        completion_wait_ms,
        gpu_ms,
        total_wall_ms: wall_started.elapsed().as_secs_f64() * 1e3,
    };
    let encode = Qwen4ExpPackedProfileEncodeTiming {
        preflight_ms: encode_detail.preflight_ms,
        stage_inputs_ms: encode_detail.stage_inputs_ms,
        graph_encode_ms: encode_detail.graph_encode_ms,
        unattributed_ms: (encode_cpu_ms
            - encode_detail.preflight_ms
            - encode_detail.stage_inputs_ms
            - encode_detail.graph_encode_ms)
            .max(0.0),
    };
    let command_timing = packed_profile_command_timing(commit_return_ms, release);
    let profile = ctx
        .resolve_timestamp_samples(&samples, sample_count)
        .map_err(|error| packed_profile_error(error.to_string()))
        .and_then(|timestamps| {
            resolve_qwen4exp_packed_profile(
                token,
                encode,
                command_timing,
                sampling,
                sampling_fallback.clone(),
                sample_count,
                &spans,
                &timestamps,
            )
        });
    Ok(Qwen4ExpPackedProfileOutcome {
        token,
        encode,
        command: command_timing,
        sampling,
        sampling_fallback,
        profile,
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen4exp_text_packed_dispatch_profiled<'a>(
    ctx: &MetalContext,
    command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    samples: &MetalTimestampSampleBuffer,
    token_ids: &[u32],
    start_position: usize,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
    encode_detail: &mut Qwen4ExpPackedEncodeCpuTiming,
) -> Result<
    (
        Qwen4ExpTextSessionPending<'a>,
        usize,
        Vec<Qwen4ExpPackedProfileSpan>,
    ),
    Qwen4ExpRuntimeError,
> {
    let detailed_gdn_layer = weights
        .geometry
        .post_ple()
        .iter()
        .find(|block| block.layer() == QWEN4EXP_PACKED_PROFILE_GDN_LAYER)
        .filter(|block| block.mixer().kind() == MixerKind::GatedDeltaNet)
        .map(|block| block.layer())
        .ok_or_else(|| {
            Qwen4ExpRuntimeError::Invalid("packed profile requires GDN layer 5".into())
        })?;
    let detailed_qsa_layer = weights
        .geometry
        .post_ple()
        .iter()
        .find(|block| block.layer() == QWEN4EXP_PACKED_PROFILE_QSA_LAYER)
        .filter(|block| block.mixer().kind() == MixerKind::QwenSparseAttention)
        .map(|block| block.layer())
        .ok_or_else(|| {
            Qwen4ExpRuntimeError::Invalid("packed profile requires QSA layer 7".into())
        })?;
    let mut recorder =
        Qwen4ExpPackedProfileRecorder::new(samples, detailed_gdn_layer, detailed_qsa_layer)?;
    let encoder = KernelEncoder::begin(command);
    let pending = encode_qwen4exp_text_packed_profiled(
        ctx,
        &encoder,
        token_ids,
        start_position,
        table,
        weights,
        workspace,
        &mut recorder,
        encode_detail,
    )?;
    let (sample_count, spans) = recorder.finish()?;
    encoder.end();
    Ok((pending, sample_count, spans))
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen4exp_text_packed_stage_profiled<'a>(
    ctx: &MetalContext,
    command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    samples: &MetalTimestampSampleBuffer,
    token_ids: &[u32],
    start_position: usize,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
    encode_detail: &mut Qwen4ExpPackedEncodeCpuTiming,
) -> Result<
    (
        Qwen4ExpTextSessionPending<'a>,
        usize,
        Vec<Qwen4ExpPackedProfileSpan>,
    ),
    Qwen4ExpRuntimeError,
> {
    let (pending, spans) = encode_qwen4exp_text_packed_layer_sampled(
        ctx,
        command,
        samples,
        token_ids,
        start_position,
        table,
        weights,
        workspace,
        encode_detail,
    )?;
    Ok((pending, samples.sample_count(), spans))
}

fn packed_profile_command_timing(
    commit_return_ms: f64,
    release: Qwen4ExpTextSessionReleaseTiming,
) -> Qwen4ExpPackedProfileCommandTiming {
    Qwen4ExpPackedProfileCommandTiming {
        commit_return_ms,
        root_wait_ms: release.root_wait_ms,
        child_publication_ms: release.child_publication_ms,
        root_publish_ms: release.root_publish_ms,
        release_total_ms: release.release_total_ms,
    }
}

fn execute_qwen4exp_text_packed_sync(
    ctx: &MetalContext,
    token_ids: &[u32],
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
) -> Result<Qwen4ExpTokenTiming, Qwen4ExpRuntimeError> {
    let start_position = workspace.committed_length();
    let position = start_position
        .checked_add(token_ids.len())
        .and_then(|end| end.checked_sub(1))
        .ok_or_else(|| {
            Qwen4ExpRuntimeError::Invalid(
                "packed prefill position range is empty or overflows".into(),
            )
        })?;
    let wall_started = Instant::now();
    let command = ctx.queue.commandBuffer().ok_or_else(|| {
        Qwen4ExpRuntimeError::Invalid("Metal command queue returned no command buffer".into())
    })?;
    let encoder = KernelEncoder::begin(&command);
    let pending = match encode_qwen4exp_text_packed(
        ctx,
        &encoder,
        token_ids,
        start_position,
        table,
        weights,
        workspace,
    ) {
        Ok(pending) => pending,
        Err(error) => {
            encoder.end();
            let abandon = unsafe { workspace.abandon_uncommitted() };
            drop(command);
            if let Err(abandon_error) = abandon {
                return invalid(format!(
                    "packed prefill encode failed ({error}); abandoning its command also failed ({abandon_error})"
                ));
            }
            return Err(error.into());
        }
    };
    drop(pending);
    encoder.end();
    let encode_cpu_ms = wall_started.elapsed().as_secs_f64() * 1e3;
    let wait_started = Instant::now();
    command.commit();
    workspace.release_after()?;
    let completion_wait_ms = wait_started.elapsed().as_secs_f64() * 1e3;
    let gpu_start = command.GPUStartTime();
    let gpu_end = command.GPUEndTime();
    let gpu_ms =
        (gpu_start.is_finite() && gpu_end.is_finite() && gpu_start > 0.0 && gpu_end > gpu_start)
            .then_some((gpu_end - gpu_start) * 1e3);
    Ok(Qwen4ExpTokenTiming {
        position,
        encode_cpu_ms,
        completion_wait_ms,
        gpu_ms,
        total_wall_ms: wall_started.elapsed().as_secs_f64() * 1e3,
    })
}

fn execute_qwen4exp_text_token_sync(
    ctx: &MetalContext,
    token_id: u32,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
) -> Result<Qwen4ExpTokenTiming, Qwen4ExpRuntimeError> {
    let position = workspace.committed_length();
    let wall_started = Instant::now();
    let command = ctx.queue.commandBuffer().ok_or_else(|| {
        Qwen4ExpRuntimeError::Invalid("Metal command queue returned no command buffer".into())
    })?;
    let encoder = KernelEncoder::begin(&command);
    let pending = match encode_qwen4exp_text_token(
        ctx, &encoder, token_id, position, table, weights, workspace,
    ) {
        Ok(pending) => pending,
        Err(error) => {
            encoder.end();
            let abandon = unsafe { workspace.abandon_uncommitted() };
            drop(command);
            if let Err(abandon_error) = abandon {
                return invalid(format!(
                    "token encode failed ({error}); abandoning its command also failed ({abandon_error})"
                ));
            }
            return Err(error.into());
        }
    };
    drop(pending);
    encoder.end();
    let encode_cpu_ms = wall_started.elapsed().as_secs_f64() * 1e3;
    let wait_started = Instant::now();
    command.commit();
    workspace.release_after()?;
    let completion_wait_ms = wait_started.elapsed().as_secs_f64() * 1e3;
    let gpu_start = command.GPUStartTime();
    let gpu_end = command.GPUEndTime();
    let gpu_ms =
        (gpu_start.is_finite() && gpu_end.is_finite() && gpu_start > 0.0 && gpu_end > gpu_start)
            .then_some((gpu_end - gpu_start) * 1e3);
    Ok(Qwen4ExpTokenTiming {
        position,
        encode_cpu_ms,
        completion_wait_ms,
        gpu_ms,
        total_wall_ms: wall_started.elapsed().as_secs_f64() * 1e3,
    })
}

fn execute_qwen4exp_text_token_layer_profiled_sync(
    ctx: &MetalContext,
    token_id: u32,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
) -> Result<Qwen4ExpLayerProfileOutcome, Qwen4ExpRuntimeError> {
    let position = workspace.committed_length();
    let stages = qwen4exp_layer_stages(weights);
    let sample_count = stages
        .len()
        .checked_mul(2)
        .ok_or_else(|| Qwen4ExpRuntimeError::Invalid("layer sample count overflow".into()))?;
    let samples = ctx.timestamp_sample_buffer(sample_count)?;
    let wall_started = Instant::now();
    let command = ctx.queue.commandBuffer().ok_or_else(|| {
        Qwen4ExpRuntimeError::Invalid("Metal command queue returned no command buffer".into())
    })?;
    let pending = match encode_qwen4exp_text_token_layer_sampled(
        ctx, &command, &samples, token_id, position, table, weights, workspace,
    ) {
        Ok(pending) => pending,
        Err(error) => {
            let abandon = unsafe { workspace.abandon_uncommitted() };
            drop(command);
            if let Err(abandon_error) = abandon {
                return invalid(format!(
                    "profiled token encode failed ({error}); abandoning its command also failed ({abandon_error})"
                ));
            }
            return Err(error.into());
        }
    };
    drop(pending);
    let encode_cpu_ms = wall_started.elapsed().as_secs_f64() * 1e3;
    let wait_started = Instant::now();
    command.commit();
    workspace.release_after()?;
    let completion_wait_ms = wait_started.elapsed().as_secs_f64() * 1e3;
    let gpu_start = command.GPUStartTime();
    let gpu_end = command.GPUEndTime();
    let gpu_ms =
        (gpu_start.is_finite() && gpu_end.is_finite() && gpu_start > 0.0 && gpu_end > gpu_start)
            .then_some((gpu_end - gpu_start) * 1e3);
    let token = Qwen4ExpTokenTiming {
        position,
        encode_cpu_ms,
        completion_wait_ms,
        gpu_ms,
        total_wall_ms: wall_started.elapsed().as_secs_f64() * 1e3,
    };
    let profile = ctx
        .resolve_timestamp_samples(&samples, sample_count)
        .map_err(|error| layer_profile_error(error.to_string()))
        .and_then(|timestamps| resolve_qwen4exp_layer_profile(token, &stages, &timestamps));
    Ok(Qwen4ExpLayerProfileOutcome { token, profile })
}

fn qwen4exp_layer_stages(weights: &Qwen4ExpTextSessionMetalWeights<'_>) -> Vec<Qwen4ExpLayerStage> {
    let mut stages = Vec::with_capacity(weights.post_ple.len() + 2);
    stages.push(Qwen4ExpLayerStage::LayersZeroOne);
    stages.extend(
        weights
            .geometry
            .post_ple()
            .iter()
            .map(|block| Qwen4ExpLayerStage::PostPle {
                layer: block.layer(),
                mixer: block.mixer().kind(),
            }),
    );
    stages.push(Qwen4ExpLayerStage::Tail);
    stages
}

fn resolve_qwen4exp_packed_profile(
    token: Qwen4ExpTokenTiming,
    encode: Qwen4ExpPackedProfileEncodeTiming,
    command: Qwen4ExpPackedProfileCommandTiming,
    sampling: Qwen4ExpPackedProfileSampling,
    sampling_fallback: Option<String>,
    sample_count: usize,
    spans: &[Qwen4ExpPackedProfileSpan],
    timestamps: &[u64],
) -> Result<Qwen4ExpPackedPrefillProfile, Qwen4ExpPackedProfileError> {
    if spans.is_empty() || sample_count == 0 || timestamps.len() != sample_count {
        return Err(packed_profile_error(format!(
            "packed profile has {} spans, {sample_count} samples, and {} timestamps",
            spans.len(),
            timestamps.len()
        )));
    }
    if timestamps.contains(&u64::MAX) {
        return Err(packed_profile_error(
            "packed profile contains a Metal counter error sentinel",
        ));
    }
    if timestamps.windows(2).any(|window| window[1] < window[0]) {
        return Err(packed_profile_error(
            "packed profile resolved timestamps are not globally monotonic",
        ));
    }
    let first_sample = spans
        .iter()
        .map(|span| span.start_sample)
        .min()
        .expect("nonempty packed spans have a first sample");
    let last_sample = spans
        .iter()
        .map(|span| span.end_sample)
        .max()
        .expect("nonempty packed spans have a final sample");
    if last_sample >= timestamps.len() {
        return Err(packed_profile_error(format!(
            "packed profile sample {last_sample} exceeds {} resolved timestamps",
            timestamps.len()
        )));
    }
    let sampled_span_ticks = timestamps[last_sample]
        .checked_sub(timestamps[first_sample])
        .ok_or_else(|| packed_profile_error("packed profile timestamps are not monotonic"))?;
    if sampled_span_ticks == 0 {
        return Err(packed_profile_error(
            "packed profile timestamp span is zero",
        ));
    }
    let command_gpu_ms = token
        .gpu_ms
        .ok_or_else(|| packed_profile_error("profiled packed command has no GPU interval"))?;
    let scale_ms_per_tick = command_gpu_ms / sampled_span_ticks as f64;
    let mut stages = Vec::with_capacity(spans.len());
    let mut coarse_ticks = 0_u64;
    for span in spans {
        if span.start_sample >= timestamps.len() || span.end_sample >= timestamps.len() {
            return Err(packed_profile_error(format!(
                "packed stage {:?} references samples {}..{} outside {} timestamps",
                span.label,
                span.start_sample,
                span.end_sample,
                timestamps.len()
            )));
        }
        let duration_ticks = timestamps[span.end_sample]
            .checked_sub(timestamps[span.start_sample])
            .ok_or_else(|| {
                packed_profile_error(format!(
                    "packed stage {:?} timestamps are not monotonic",
                    span.label
                ))
            })?;
        let gpu_ms = duration_ticks as f64 * scale_ms_per_tick;
        if span.label.scope == Qwen4ExpPackedProfileScope::Coarse {
            coarse_ticks = coarse_ticks
                .checked_add(duration_ticks)
                .ok_or_else(|| packed_profile_error("packed profile coarse duration overflow"))?;
        }
        stages.push(Qwen4ExpPackedProfileStageTiming {
            label: span.label,
            depth: span.depth,
            start_sample: span.start_sample,
            end_sample: span.end_sample,
            duration_ticks,
            gpu_ms,
            fraction_of_gpu: gpu_ms / command_gpu_ms,
        });
    }
    if coarse_ticks > sampled_span_ticks {
        return Err(packed_profile_error(format!(
            "packed profile coarse stages account for {coarse_ticks} ticks across a {sampled_span_ticks}-tick span"
        )));
    }
    stages.sort_by(|left, right| {
        left.start_sample
            .cmp(&right.start_sample)
            .then_with(|| right.end_sample.cmp(&left.end_sample))
    });
    let raw_span_ms_assuming_ns = sampled_span_ticks as f64 * 1e-6;
    Ok(Qwen4ExpPackedPrefillProfile {
        token,
        encode,
        command,
        sampling,
        sampling_fallback,
        stages,
        sample_count,
        sampled_span_ticks,
        raw_span_ms_assuming_ns,
        raw_coverage_assuming_ns: raw_span_ms_assuming_ns / command_gpu_ms,
    })
}

fn resolve_qwen4exp_layer_profile(
    token: Qwen4ExpTokenTiming,
    stages: &[Qwen4ExpLayerStage],
    timestamps: &[u64],
) -> Result<Qwen4ExpLayerProfile, Qwen4ExpLayerProfileError> {
    let expected_samples = stages
        .len()
        .checked_mul(2)
        .ok_or_else(|| layer_profile_error("layer sample count overflow"))?;
    if stages.is_empty() || timestamps.len() != expected_samples {
        return Err(layer_profile_error(format!(
            "layer profile has {} stages and {} samples; expected a nonempty 2:1 sample mapping",
            stages.len(),
            timestamps.len()
        )));
    }
    let command_gpu_ms = token
        .gpu_ms
        .ok_or_else(|| layer_profile_error("profiled command has no GPU interval"))?;
    let first_start = timestamps[0];
    let last_end = *timestamps
        .last()
        .expect("nonempty layer timestamps have a final sample");
    let sampled_span_ticks = last_end
        .checked_sub(first_start)
        .ok_or_else(|| layer_profile_error("layer timestamps are not monotonic"))?;
    if sampled_span_ticks == 0 {
        return Err(layer_profile_error("layer timestamp span is zero"));
    }
    let scale_ms_per_tick = command_gpu_ms / sampled_span_ticks as f64;
    let mut previous_end = None;
    let mut stage_gpu_ms = 0.0;
    let mut resolved = Vec::with_capacity(stages.len());
    for (index, &stage) in stages.iter().enumerate() {
        let start = timestamps[index * 2];
        let end = timestamps[index * 2 + 1];
        if end < start || previous_end.is_some_and(|previous| start < previous) {
            return Err(layer_profile_error(format!(
                "layer stage {index} timestamps are not monotonic: previous_end={previous_end:?} start={start} end={end}"
            )));
        }
        let duration_ticks = end - start;
        let gpu_ms = duration_ticks as f64 * scale_ms_per_tick;
        stage_gpu_ms += gpu_ms;
        resolved.push(Qwen4ExpLayerStageTiming {
            stage,
            duration_ticks,
            gpu_ms,
            fraction_of_gpu: gpu_ms / command_gpu_ms,
        });
        previous_end = Some(end);
    }
    Ok(Qwen4ExpLayerProfile {
        token,
        stages: resolved,
        encoder_boundary_ms: (command_gpu_ms - stage_gpu_ms).max(0.0),
        sampled_span_ticks,
    })
}

fn layer_profile_error(detail: impl Into<String>) -> Qwen4ExpLayerProfileError {
    Qwen4ExpLayerProfileError {
        detail: detail.into(),
    }
}

fn packed_profile_error(detail: impl Into<String>) -> Qwen4ExpPackedProfileError {
    Qwen4ExpPackedProfileError {
        detail: detail.into(),
    }
}

fn checked_lcm(left: usize, right: usize) -> Result<usize, Qwen4ExpRuntimeError> {
    let divisor = gcd(left, right);
    left.checked_div(divisor)
        .and_then(|value| value.checked_mul(right))
        .ok_or_else(|| Qwen4ExpRuntimeError::Invalid("QSA alignment overflow".into()))
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpRuntimeError> {
    Err(Qwen4ExpRuntimeError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::{DispatchCensusRow, evaluate_metal_memory_admission, host_page_size_bytes};
    use crate::qwen4exp_moe::{
        Qwen4ExpIq3GateUpCaptureBanks, Qwen4ExpIq3GateUpCaptureRecord, Qwen4ExpIq3GateUpProbeArm,
        encode_qwen4exp_iq3_gate_up_captured_arm, with_qwen4exp_iq3_gate_up_capture,
        with_qwen4exp_moe_iq3_fast_override, with_qwen4exp_moe_route_count_capture,
        with_qwen4exp_packed_router_e8p32_strict_override,
    };
    use crate::sampling::{Sampler, SamplingConfig};
    use crate::tensor::GgmlType;
    use crate::tokenizer::Tokenizer;
    use objc2_metal::{MTLBuffer, MTLCommandBufferStatus};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap()
    }

    fn report_logit_arms(
        label: &str,
        baseline: &[f32],
        candidate: &[f32],
    ) -> (bool, f64, f64, f32) {
        assert_eq!(baseline.len(), candidate.len());
        assert!(
            baseline
                .iter()
                .chain(candidate)
                .all(|value| value.is_finite())
        );
        let mut dot = 0.0_f64;
        let mut baseline_norm = 0.0_f64;
        let mut candidate_norm = 0.0_f64;
        let mut difference_norm = 0.0_f64;
        let mut max_abs = 0.0_f32;
        for (&left, &right) in baseline.iter().zip(candidate) {
            let left = left as f64;
            let right = right as f64;
            let difference = left - right;
            dot += left * right;
            baseline_norm += left * left;
            candidate_norm += right * right;
            difference_norm += difference * difference;
            max_abs = max_abs.max(difference.abs() as f32);
        }
        let cosine = dot / (baseline_norm.sqrt() * candidate_norm.sqrt()).max(f64::MIN_POSITIVE);
        let relative_rms = (difference_norm / baseline_norm.max(f64::MIN_POSITIVE)).sqrt();
        eprintln!(
            "{label}: baseline_argmax={} candidate_argmax={} cosine={cosine:.12} relative_rms={relative_rms:.6e} max_abs={max_abs:.6e}",
            argmax(baseline),
            argmax(candidate),
        );
        (
            argmax(baseline) == argmax(candidate),
            cosine,
            relative_rms,
            max_abs,
        )
    }

    fn assert_logit_arms_close(label: &str, baseline: &[f32], candidate: &[f32]) {
        let (argmax_equal, cosine, relative_rms, max_abs) =
            report_logit_arms(label, baseline, candidate);
        assert!(argmax_equal, "{label} argmax");
        assert!(cosine > 0.999_999_99, "{label} cosine {cosine}");
        assert!(relative_rms < 1e-4, "{label} relative RMS {relative_rms}");
        assert!(max_abs < 1e-3, "{label} maximum delta {max_abs}");
    }

    fn assert_released_packed_logit_arms_close(label: &str, baseline: &[f32], candidate: &[f32]) {
        let (argmax_equal, cosine, relative_rms, max_abs) =
            report_logit_arms(label, baseline, candidate);
        assert!(argmax_equal, "{label} argmax");
        assert!(cosine >= 0.999_9, "{label} cosine {cosine}");
        assert!(
            relative_rms <= 1.5e-2,
            "{label} relative RMS {relative_rms}"
        );
        assert!(max_abs <= 0.2, "{label} maximum delta {max_abs}");
    }

    fn assert_f32_bits_eq(label: &str, baseline: &[f32], candidate: &[f32]) {
        assert_eq!(baseline.len(), candidate.len(), "{label} length");
        if let Some((index, (&expected, &actual))) = baseline
            .iter()
            .zip(candidate)
            .enumerate()
            .find(|(_, (expected, actual))| expected.to_bits() != actual.to_bits())
        {
            panic!(
                "{label}[{index}] differs: baseline={expected:?} ({:#010x}) candidate={actual:?} ({:#010x})",
                expected.to_bits(),
                actual.to_bits(),
            );
        }
    }

    fn snapshot_persistent_state(runner: &Qwen4ExpTextRunner<'_, '_, '_>) -> Vec<Vec<u8>> {
        runner
            .workspace
            .persistent_state_tensors()
            .into_iter()
            .map(|tensor| unsafe {
                let source = tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize);
                std::slice::from_raw_parts(source, tensor.n_bytes() as usize).to_vec()
            })
            .collect()
    }

    fn zero_persistent_state(runner: &Qwen4ExpTextRunner<'_, '_, '_>) {
        for tensor in runner.workspace.persistent_state_tensors() {
            unsafe {
                let destination = tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize);
                std::slice::from_raw_parts_mut(destination, tensor.n_bytes() as usize).fill(0);
            }
        }
    }

    fn assert_state_bytes_eq(label: &str, baseline: &[Vec<u8>], candidate: &[Vec<u8>]) {
        assert_eq!(baseline.len(), candidate.len(), "{label} tensor count");
        for (tensor, (expected, actual)) in baseline.iter().zip(candidate).enumerate() {
            assert_eq!(
                expected.len(),
                actual.len(),
                "{label} tensor {tensor} byte length"
            );
            if let Some((byte, (&expected, &actual))) = expected
                .iter()
                .zip(actual)
                .enumerate()
                .find(|(_, (expected, actual))| expected != actual)
            {
                panic!(
                    "{label} tensor {tensor} byte {byte} differs: baseline={expected:#04x} candidate={actual:#04x}"
                );
            }
        }
    }

    fn assert_router_candidate_census(
        label: &str,
        tokens: usize,
        expected_substitutions: usize,
        baseline: &[DispatchCensusRow],
        candidate: &[DispatchCensusRow],
    ) {
        const GENERIC: &str = "kernel_mat_mat_f32_f32";
        const STRICT: &str = "kernel_mat_mat_f32_f32_router_e8p32_strict";
        assert_eq!(baseline.len(), candidate.len(), "{label} dispatch count");
        assert!(!baseline.is_empty(), "{label} baseline census");
        assert!(
            baseline
                .iter()
                .chain(candidate)
                .all(|row| row.encoder_ordinal == 0 && !row.encoder_concurrent),
            "{label} must use one serial encoder"
        );
        let mut substitutions = 0_usize;
        for (index, (baseline, candidate)) in baseline.iter().zip(candidate).enumerate() {
            assert_eq!(baseline.family, candidate.family, "{label} family {index}");
            assert_eq!(baseline.tag, candidate.tag, "{label} tag {index}");
            assert_eq!(
                baseline.encoder_ordinal, candidate.encoder_ordinal,
                "{label} encoder {index}"
            );
            assert_eq!(
                baseline.encoder_concurrent, candidate.encoder_concurrent,
                "{label} encoder mode {index}"
            );
            if candidate.kernel == STRICT {
                substitutions += 1;
                assert_eq!(baseline.kernel, GENERIC, "{label} substitution {index}");
                assert_eq!(baseline.grid_width, 512, "{label} baseline grid {index}");
                assert_eq!(candidate.grid_width, 64, "{label} candidate grid {index}");
                assert_eq!(baseline.grid_height, tokens.div_ceil(32) as u64);
                assert_eq!(candidate.grid_height, tokens.div_ceil(32) as u64);
                assert_eq!(baseline.grid_depth, 1);
                assert_eq!(candidate.grid_depth, 1);
                assert_eq!(baseline.threads_width, 32);
                assert_eq!(candidate.threads_width, 32);
                assert_eq!(baseline.threads_height, 1);
                assert_eq!(candidate.threads_height, 1);
                assert_eq!(baseline.threads_depth, 1);
                assert_eq!(candidate.threads_depth, 1);
                assert_eq!(baseline.tg_threads, candidate.tg_threads);
            } else {
                assert_eq!(baseline.kernel, candidate.kernel, "{label} kernel {index}");
                assert_eq!(
                    (
                        baseline.grid_width,
                        baseline.grid_height,
                        baseline.grid_depth,
                        baseline.threads_width,
                        baseline.threads_height,
                        baseline.threads_depth,
                        baseline.grid_tgs,
                        baseline.tg_threads,
                    ),
                    (
                        candidate.grid_width,
                        candidate.grid_height,
                        candidate.grid_depth,
                        candidate.threads_width,
                        candidate.threads_height,
                        candidate.threads_depth,
                        candidate.grid_tgs,
                        candidate.tg_threads,
                    ),
                    "{label} geometry {index}"
                );
            }
        }
        assert_eq!(
            substitutions, expected_substitutions,
            "{label} strict router substitutions"
        );
    }

    struct PackedRouterReplay {
        endpoint: Vec<f32>,
        continuation: Vec<f32>,
        prefill_state: Vec<Vec<u8>>,
        continuation_state: Vec<Vec<u8>>,
        prefill_qsa_lengths: Vec<(u32, usize)>,
        continuation_qsa_lengths: Vec<(u32, usize)>,
        prefill_ple_prior_tokens: Vec<u32>,
        continuation_ple_prior_tokens: Vec<u32>,
        census: Vec<DispatchCensusRow>,
    }

    fn run_packed_replay(
        runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
        tokens: &[u32],
        continuation_token: u32,
    ) -> PackedRouterReplay {
        crate::metal::dispatch_census_begin();
        let endpoint = runner.prefill(tokens).unwrap().to_vec();
        let census = crate::metal::dispatch_census_take();
        let timing = runner.last_prefill_timing().unwrap();
        assert_eq!(timing.packed_token_count, tokens.len());
        assert_eq!(timing.command_count, 1);
        let prefill_state = snapshot_persistent_state(runner);
        let prefill_qsa_lengths = runner.workspace.qsa_committed_lengths();
        let prefill_ple_prior_tokens = runner.workspace.ple_prior_tokens().to_vec();
        let continuation = runner.forward_token(continuation_token).unwrap().to_vec();
        PackedRouterReplay {
            endpoint,
            continuation,
            prefill_state,
            continuation_state: snapshot_persistent_state(runner),
            prefill_qsa_lengths,
            continuation_qsa_lengths: runner.workspace.qsa_committed_lengths(),
            prefill_ple_prior_tokens,
            continuation_ple_prior_tokens: runner.workspace.ple_prior_tokens().to_vec(),
            census,
        }
    }

    fn fill_i32_tensor(tensor: &crate::metal::MetalTensor, value: i32) {
        assert_eq!(tensor.dtype, GgmlType::I32);
        unsafe {
            let destination = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<i32>();
            std::slice::from_raw_parts_mut(destination, tensor.n_elements() as usize).fill(value);
        }
    }

    fn fill_f32_tensor_bits(tensor: &crate::metal::MetalTensor, bits: u32) {
        assert_eq!(tensor.dtype, GgmlType::F32);
        unsafe {
            let destination = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<u32>();
            std::slice::from_raw_parts_mut(destination, tensor.n_elements() as usize).fill(bits);
        }
    }

    fn read_f32_tensor_bits(tensor: &crate::metal::MetalTensor) -> Vec<u32> {
        assert_eq!(tensor.dtype, GgmlType::F32);
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<u32>();
            std::slice::from_raw_parts(source, tensor.n_elements() as usize).to_vec()
        }
    }

    fn sha256_metal_tensor_bytes(domain: &[u8], tensor: &crate::metal::MetalTensor) -> String {
        let bytes = unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize);
            std::slice::from_raw_parts(source, tensor.n_bytes() as usize)
        };
        let mut hash = Sha256::new();
        hash.update(domain);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
        format!("{:x}", hash.finalize())
    }

    fn read_i32_tensor(tensor: &crate::metal::MetalTensor) -> Vec<i32> {
        assert_eq!(tensor.dtype, GgmlType::I32);
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<i32>();
            std::slice::from_raw_parts(source, tensor.n_elements() as usize).to_vec()
        }
    }

    fn assert_route_count_capture_census(
        label: &str,
        baseline: &[DispatchCensusRow],
        captured: &[DispatchCensusRow],
    ) {
        const COPY: &str = "kernel_copy_offset_i32";
        let copies = captured.iter().filter(|row| row.kernel == COPY).count();
        assert_eq!(copies, 48, "{label} capture dispatches");
        for (index, row) in captured
            .iter()
            .enumerate()
            .filter(|(_, row)| row.kernel == COPY)
        {
            assert!(index > 0 && index + 1 < captured.len());
            assert_eq!(
                captured[index - 1].kernel,
                "kernel_moe_route_bucket_slots_f32",
                "{label} capture {index} must follow its bucket"
            );
            assert!(
                captured[index + 1].kernel.starts_with("kernel_moe_swiglu_"),
                "{label} capture {index} must precede routed gate/up"
            );
            assert_eq!(row.encoder_ordinal, 0);
            assert!(!row.encoder_concurrent);
        }
        let filtered = captured
            .iter()
            .filter(|row| row.kernel != COPY)
            .collect::<Vec<_>>();
        assert_eq!(baseline.len(), filtered.len(), "{label} filtered census");
        for (index, (expected, actual)) in baseline.iter().zip(filtered).enumerate() {
            assert_eq!(expected.family, actual.family, "{label} family {index}");
            assert_eq!(expected.tag, actual.tag, "{label} tag {index}");
            assert_eq!(
                expected.encoder_ordinal, actual.encoder_ordinal,
                "{label} encoder {index}"
            );
            assert_eq!(
                expected.encoder_concurrent, actual.encoder_concurrent,
                "{label} encoder mode {index}"
            );
            assert_eq!(expected.kernel, actual.kernel, "{label} kernel {index}");
            assert_eq!(
                (
                    expected.grid_width,
                    expected.grid_height,
                    expected.grid_depth,
                    expected.threads_width,
                    expected.threads_height,
                    expected.threads_depth,
                    expected.grid_tgs,
                    expected.tg_threads,
                ),
                (
                    actual.grid_width,
                    actual.grid_height,
                    actual.grid_depth,
                    actual.threads_width,
                    actual.threads_height,
                    actual.threads_depth,
                    actual.grid_tgs,
                    actual.tg_threads,
                ),
                "{label} geometry {index}"
            );
        }
    }

    fn assert_iq3_gate_up_capture_census(
        label: &str,
        records: &[Qwen4ExpIq3GateUpCaptureRecord],
        baseline: &[DispatchCensusRow],
        candidate: &[DispatchCensusRow],
    ) {
        const PREFIX: &str = "qwen4exp.iq3_gate_up_capture.";
        let expected_layers = (0_u32..48)
            .filter(|layer| !matches!(layer, 2 | 4 | 30 | 46 | 47))
            .collect::<Vec<_>>();
        assert_eq!(records.len(), expected_layers.len(), "{label} records");
        assert_eq!(candidate.len(), baseline.len() + expected_layers.len() * 3);
        let capture_rows = candidate
            .iter()
            .enumerate()
            .filter(|(_, row)| {
                row.tag
                    .as_deref()
                    .is_some_and(|tag| tag.starts_with(PREFIX))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            capture_rows.len(),
            expected_layers.len() * 3,
            "{label} rows"
        );
        for (ordinal, (&layer, record)) in expected_layers.iter().zip(records).enumerate() {
            assert_eq!(record.ordinal, ordinal);
            assert_eq!(record.layer, layer);
            let triple = &capture_rows[ordinal * 3..ordinal * 3 + 3];
            let first_index = triple[0].0;
            assert!(first_index > 0 && first_index + 3 < candidate.len());
            assert_eq!(triple[1].0, first_index + 1);
            assert_eq!(triple[2].0, first_index + 2);
            assert_eq!(
                candidate[first_index - 1].kernel,
                "kernel_moe_route_bucket_slots_f32"
            );
            let production = &candidate[first_index + 3];
            assert_eq!(
                production.kernel,
                "kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16"
            );
            assert!(
                production
                    .tag
                    .as_deref()
                    .is_none_or(|tag| !tag.starts_with(PREFIX))
            );
            for ((_, row), (kind, kernel)) in triple.iter().zip([
                ("input", "kernel_copy_offset_f32"),
                ("counts", "kernel_copy_offset_i32"),
                ("slots", "kernel_copy_offset_i32"),
            ]) {
                let expected_tag =
                    format!("qwen4exp.iq3_gate_up_capture.ordinal{ordinal}.layer{layer}.{kind}");
                assert_eq!(row.tag.as_deref(), Some(expected_tag.as_str()));
                assert_eq!(row.kernel, kernel);
                assert_eq!(row.encoder_ordinal, 0);
                assert!(!row.encoder_concurrent);
            }
        }
        let filtered = candidate
            .iter()
            .filter(|row| {
                row.tag
                    .as_deref()
                    .is_none_or(|tag| !tag.starts_with(PREFIX))
            })
            .collect::<Vec<_>>();
        assert_eq!(baseline.len(), filtered.len(), "{label} filtered census");
        for (index, (expected, actual)) in baseline.iter().zip(filtered).enumerate() {
            assert_eq!(expected.family, actual.family, "{label} family {index}");
            assert_eq!(expected.tag, actual.tag, "{label} tag {index}");
            assert_eq!(expected.kernel, actual.kernel, "{label} kernel {index}");
            assert_eq!(
                (
                    expected.encoder_ordinal,
                    expected.encoder_concurrent,
                    expected.grid_width,
                    expected.grid_height,
                    expected.grid_depth,
                    expected.threads_width,
                    expected.threads_height,
                    expected.threads_depth,
                    expected.grid_tgs,
                    expected.tg_threads,
                ),
                (
                    actual.encoder_ordinal,
                    actual.encoder_concurrent,
                    actual.grid_width,
                    actual.grid_height,
                    actual.grid_depth,
                    actual.threads_width,
                    actual.threads_height,
                    actual.threads_depth,
                    actual.grid_tgs,
                    actual.tg_threads,
                ),
                "{label} geometry {index}"
            );
        }
    }

    fn assert_iq3_gate_up_range_census(
        arm: Qwen4ExpIq3GateUpProbeArm,
        records: &[Qwen4ExpIq3GateUpCaptureRecord],
        census: &[DispatchCensusRow],
    ) {
        assert_eq!(census.len(), records.len());
        for (row, record) in census.iter().zip(records) {
            let expected_tag = format!(
                "qwen4exp.iq3_gate_up_range.{}.ordinal{}.layer{}",
                arm.as_str(),
                record.ordinal,
                record.layer
            );
            assert_eq!(row.tag.as_deref(), Some(expected_tag.as_str()));
            assert_eq!(
                row.kernel,
                "kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16"
            );
            assert_eq!(row.encoder_ordinal, 0);
            assert!(!row.encoder_concurrent);
            assert_eq!(
                (
                    row.grid_width,
                    row.grid_height,
                    row.grid_depth,
                    row.threads_width,
                    row.threads_height,
                    row.threads_depth,
                ),
                (128, 10, 512, 128, 1, 1)
            );
        }
    }

    fn sha256_u32_le(domain: &[u8], values: &[u32]) -> String {
        let mut hash = Sha256::new();
        hash.update(domain);
        for value in values {
            hash.update(value.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    fn sha256_i32_le(domain: &[u8], values: &[i32]) -> String {
        let mut hash = Sha256::new();
        hash.update(domain);
        for value in values {
            hash.update(value.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    fn sha256_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn sha256_file(path: &std::path::Path) -> String {
        use std::io::Read;

        let mut file = std::fs::File::open(path).unwrap();
        let mut hash = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let read = file.read(&mut buffer).unwrap();
            if read == 0 {
                break;
            }
            hash.update(&buffer[..read]);
        }
        format!("{:x}", hash.finalize())
    }

    fn sha256_f32_bits(domain: &[u8], values: &[f32]) -> String {
        let mut hash = Sha256::new();
        hash.update(domain);
        for value in values {
            hash.update(value.to_bits().to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    fn sha256_state_bytes(domain: &[u8], tensors: &[Vec<u8>]) -> String {
        let mut hash = Sha256::new();
        hash.update(domain);
        for tensor in tensors {
            hash.update((tensor.len() as u64).to_le_bytes());
            hash.update(tensor);
        }
        format!("{:x}", hash.finalize())
    }

    fn sha256_census_rows(domain: &[u8], rows: &[DispatchCensusRow], omit_capture: bool) -> String {
        let mut hash = Sha256::new();
        hash.update(domain);
        for row in rows
            .iter()
            .filter(|row| !omit_capture || row.kernel != "kernel_copy_offset_i32")
        {
            for value in [row.family, row.tag.as_deref().unwrap_or(""), &row.kernel] {
                hash.update((value.len() as u64).to_le_bytes());
                hash.update(value.as_bytes());
            }
            hash.update(row.encoder_ordinal.to_le_bytes());
            hash.update([u8::from(row.encoder_concurrent)]);
            for value in [
                row.grid_width,
                row.grid_height,
                row.grid_depth,
                row.threads_width,
                row.threads_height,
                row.threads_depth,
                row.grid_tgs,
                row.tg_threads,
            ] {
                hash.update(value.to_le_bytes());
            }
        }
        format!("{:x}", hash.finalize())
    }

    fn sha256_qsa_lengths(domain: &[u8], lengths: &[(u32, usize)]) -> String {
        let mut hash = Sha256::new();
        hash.update(domain);
        for &(layer, length) in lengths {
            hash.update(layer.to_le_bytes());
            hash.update((length as u64).to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    struct PackedReplayDigests {
        endpoint_logits: String,
        continuation_logits: String,
        prefill_state: String,
        continuation_state: String,
        prefill_qsa_lengths: String,
        continuation_qsa_lengths: String,
        prefill_ple_history: String,
        continuation_ple_history: String,
        census_raw: String,
        census_without_capture: String,
    }

    impl PackedReplayDigests {
        fn from_replay(replay: &PackedRouterReplay) -> Self {
            Self {
                endpoint_logits: sha256_f32_bits(
                    b"qwen4exp-packed-replay-endpoint-logits-f32le-v1\0",
                    &replay.endpoint,
                ),
                continuation_logits: sha256_f32_bits(
                    b"qwen4exp-packed-replay-continuation-logits-f32le-v1\0",
                    &replay.continuation,
                ),
                prefill_state: sha256_state_bytes(
                    b"qwen4exp-packed-replay-prefill-state-v1\0",
                    &replay.prefill_state,
                ),
                continuation_state: sha256_state_bytes(
                    b"qwen4exp-packed-replay-continuation-state-v1\0",
                    &replay.continuation_state,
                ),
                prefill_qsa_lengths: sha256_qsa_lengths(
                    b"qwen4exp-packed-replay-prefill-qsa-lengths-v1\0",
                    &replay.prefill_qsa_lengths,
                ),
                continuation_qsa_lengths: sha256_qsa_lengths(
                    b"qwen4exp-packed-replay-continuation-qsa-lengths-v1\0",
                    &replay.continuation_qsa_lengths,
                ),
                prefill_ple_history: sha256_u32_le(
                    b"qwen4exp-packed-replay-prefill-ple-history-u32le-v1\0",
                    &replay.prefill_ple_prior_tokens,
                ),
                continuation_ple_history: sha256_u32_le(
                    b"qwen4exp-packed-replay-continuation-ple-history-u32le-v1\0",
                    &replay.continuation_ple_prior_tokens,
                ),
                census_raw: sha256_census_rows(
                    b"qwen4exp-packed-replay-dispatch-census-v1\0",
                    &replay.census,
                    false,
                ),
                census_without_capture: sha256_census_rows(
                    b"qwen4exp-packed-replay-dispatch-census-v1\0",
                    &replay.census,
                    true,
                ),
            }
        }

        fn binding_sha256(&self, omit_capture: bool) -> String {
            let census = if omit_capture {
                &self.census_without_capture
            } else {
                &self.census_raw
            };
            let mut hash = Sha256::new();
            hash.update(b"qwen4exp-packed-replay-evidence-binding-v1\0");
            for (name, digest) in [
                ("endpoint_logits", &self.endpoint_logits),
                ("continuation_logits", &self.continuation_logits),
                ("prefill_state", &self.prefill_state),
                ("continuation_state", &self.continuation_state),
                ("prefill_qsa_lengths", &self.prefill_qsa_lengths),
                ("continuation_qsa_lengths", &self.continuation_qsa_lengths),
                ("prefill_ple_history", &self.prefill_ple_history),
                ("continuation_ple_history", &self.continuation_ple_history),
                ("dispatch_census", census),
            ] {
                hash.update((name.len() as u64).to_le_bytes());
                hash.update(name.as_bytes());
                hash.update(digest.as_bytes());
            }
            format!("{:x}", hash.finalize())
        }

        fn json(&self) -> serde_json::Value {
            serde_json::json!({
                "endpoint_logits_sha256_f32le": &self.endpoint_logits,
                "continuation_logits_sha256_f32le": &self.continuation_logits,
                "prefill_persistent_state_sha256": &self.prefill_state,
                "continuation_persistent_state_sha256": &self.continuation_state,
                "prefill_qsa_lengths_sha256": &self.prefill_qsa_lengths,
                "continuation_qsa_lengths_sha256": &self.continuation_qsa_lengths,
                "prefill_ple_history_sha256_u32le": &self.prefill_ple_history,
                "continuation_ple_history_sha256_u32le": &self.continuation_ple_history,
                "dispatch_census_sha256": &self.census_raw,
                "dispatch_census_without_capture_sha256": &self.census_without_capture,
                "raw_binding_sha256": self.binding_sha256(false),
                "without_capture_binding_sha256": self.binding_sha256(true)
            })
        }
    }

    fn is_lower_hex(value: &str, length: usize) -> bool {
        value.len() == length
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    fn route_count_band_json(counts: &[i32], min: i32, max: Option<i32>) -> serde_json::Value {
        let selected = counts
            .iter()
            .copied()
            .filter(|&count| count >= min && max.is_none_or(|maximum| count <= maximum));
        let (experts, routes) = selected.fold((0_usize, 0_usize), |(experts, routes), count| {
            (experts + 1, routes + count as usize)
        });
        serde_json::json!({ "experts": experts, "routes": routes })
    }

    fn route_count_panels_json(counts: &[i32], tokens: usize, width: usize) -> serde_json::Value {
        let routes = counts.iter().map(|&count| count as usize).sum::<usize>();
        let active_panels = counts
            .iter()
            .map(|&count| (count as usize).div_ceil(width))
            .sum::<usize>();
        let full_panels = counts.len() * tokens.div_ceil(width);
        let padded_columns = active_panels * width - routes;
        let occupancy = if active_panels == 0 {
            0.0
        } else {
            routes as f64 / (active_panels * width) as f64
        };
        serde_json::json!({
            "width": width,
            "active_panels": active_panels,
            "full_panels": full_panels,
            "early_return_panels": full_panels - active_panels,
            "padded_columns": padded_columns,
            "active_panel_lane_occupancy": occupancy
        })
    }

    fn route_count_layer_json(
        layer: usize,
        counts: &[i32],
        tokens: usize,
        gate_dtype: GgmlType,
        up_dtype: GgmlType,
        down_dtype: GgmlType,
    ) -> serde_json::Value {
        let mut histogram = BTreeMap::<i32, usize>::new();
        for &count in counts {
            *histogram.entry(count).or_default() += 1;
        }
        let route_sum = counts.iter().map(|&count| count as usize).sum::<usize>();
        serde_json::json!({
            "layer": layer,
            "gate_dtype": format!("{gate_dtype:?}"),
            "up_dtype": format!("{up_dtype:?}"),
            "down_dtype": format!("{down_dtype:?}"),
            "route_sum": route_sum,
            "active_experts": counts.iter().filter(|&&count| count != 0).count(),
            "max_count": counts.iter().copied().max().unwrap_or(0),
            "exact_histogram": histogram.into_iter().collect::<Vec<_>>(),
            "bands": {
                "zero": route_count_band_json(counts, 0, Some(0)),
                "one_to_8": route_count_band_json(counts, 1, Some(8)),
                "nine_to_16": route_count_band_json(counts, 9, Some(16)),
                "seventeen_to_32": route_count_band_json(counts, 17, Some(32)),
                "thirty_three_to_64": route_count_band_json(counts, 33, Some(64)),
                "sixty_five_plus": route_count_band_json(counts, 65, None)
            },
            "panels": {
                "8": route_count_panels_json(counts, tokens, 8),
                "16": route_count_panels_json(counts, tokens, 16),
                "32": route_count_panels_json(counts, tokens, 32)
            },
            "current_n16_grid": {
                "full_threadgroups": 10 * 512 * tokens.div_ceil(16),
                "active_threadgroups": 10 * counts.iter().map(|&count| (count as usize).div_ceil(16)).sum::<usize>()
            }
        })
    }

    fn route_count_aggregate_json(
        name: &str,
        layers: &[usize],
        rows: &[Vec<i32>],
        tokens: usize,
    ) -> serde_json::Value {
        let counts = layers
            .iter()
            .flat_map(|&layer| rows[layer].iter().copied())
            .collect::<Vec<_>>();
        serde_json::json!({
            "name": name,
            "layers": layers,
            "route_sum": counts.iter().map(|&count| count as usize).sum::<usize>(),
            "active_expert_instances": counts.iter().filter(|&&count| count != 0).count(),
            "bands": {
                "zero": route_count_band_json(&counts, 0, Some(0)),
                "one_to_8": route_count_band_json(&counts, 1, Some(8)),
                "nine_to_16": route_count_band_json(&counts, 9, Some(16)),
                "seventeen_to_32": route_count_band_json(&counts, 17, Some(32)),
                "thirty_three_to_64": route_count_band_json(&counts, 33, Some(64)),
                "sixty_five_plus": route_count_band_json(&counts, 65, None)
            },
            "panels": {
                "8": route_count_panels_json(&counts, tokens, 8),
                "16": route_count_panels_json(&counts, tokens, 16),
                "32": route_count_panels_json(&counts, tokens, 32)
            }
        })
    }

    fn assert_packed_replay_state_bits_eq(
        label: &str,
        baseline: &PackedRouterReplay,
        candidate: &PackedRouterReplay,
    ) {
        assert_f32_bits_eq(
            &format!("{label} endpoint logits"),
            &baseline.endpoint,
            &candidate.endpoint,
        );
        assert_f32_bits_eq(
            &format!("{label} continuation logits"),
            &baseline.continuation,
            &candidate.continuation,
        );
        assert_state_bytes_eq(
            &format!("{label} packed persistent state"),
            &baseline.prefill_state,
            &candidate.prefill_state,
        );
        assert_state_bytes_eq(
            &format!("{label} continuation persistent state"),
            &baseline.continuation_state,
            &candidate.continuation_state,
        );
        assert_eq!(candidate.prefill_qsa_lengths, baseline.prefill_qsa_lengths);
        assert_eq!(
            candidate.continuation_qsa_lengths,
            baseline.continuation_qsa_lengths
        );
        assert_eq!(
            candidate.prefill_ple_prior_tokens,
            baseline.prefill_ple_prior_tokens
        );
        assert_eq!(
            candidate.continuation_ple_prior_tokens,
            baseline.continuation_ple_prior_tokens
        );
    }

    fn assert_packed_replay_bits_eq(
        label: &str,
        baseline: &PackedRouterReplay,
        captured: &PackedRouterReplay,
    ) {
        assert_packed_replay_state_bits_eq(label, baseline, captured);
        assert_route_count_capture_census(label, &baseline.census, &captured.census);
    }

    struct Iq3GateUpRangeObservation {
        arm: Qwen4ExpIq3GateUpProbeArm,
        command_gpu_ms: f64,
        command_wall_ms: f64,
    }

    impl Iq3GateUpRangeObservation {
        fn json(&self) -> serde_json::Value {
            serde_json::json!({
                "arm": self.arm.as_str(),
                "command_gpu_ms": self.command_gpu_ms,
                "command_wall_ms": self.command_wall_ms
            })
        }
    }

    fn run_iq3_gate_up_range_command(
        ctx: &MetalContext,
        banks: &Qwen4ExpIq3GateUpCaptureBanks,
        records: &[Qwen4ExpIq3GateUpCaptureRecord],
        weights: &[crate::qwen4exp_moe::Qwen4ExpMoeMetalWeights<'_>],
        output: &crate::metal::MetalTensor,
        arm: Qwen4ExpIq3GateUpProbeArm,
        qualify_census: bool,
    ) -> (Iq3GateUpRangeObservation, Vec<DispatchCensusRow>) {
        let wall_started = Instant::now();
        if qualify_census {
            crate::metal::dispatch_census_begin();
        }
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_qwen4exp_iq3_gate_up_captured_arm(
            ctx, &encoder, banks, records, weights, output, arm,
        )
        .unwrap();
        let census = if qualify_census {
            crate::metal::dispatch_census_take()
        } else {
            Vec::new()
        };
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());
        let gpu_start = command.GPUStartTime();
        let gpu_end = command.GPUEndTime();
        assert!(gpu_start.is_finite() && gpu_start > 0.0);
        assert!(gpu_end.is_finite() && gpu_end > gpu_start);
        (
            Iq3GateUpRangeObservation {
                arm,
                command_gpu_ms: (gpu_end - gpu_start) * 1e3,
                command_wall_ms: wall_started.elapsed().as_secs_f64() * 1e3,
            },
            census,
        )
    }

    fn metal_signals_json(signals: crate::metal::MetalMemorySignals) -> serde_json::Value {
        serde_json::json!({
            "recommended_max_bytes": signals.recommended_max_bytes,
            "current_allocated_bytes": signals.current_allocated_bytes,
            "process_limit_remaining_bytes": signals.process_limit_remaining_bytes
        })
    }

    fn allocate_iq3_gate_up_capture(
        ctx: &MetalContext,
    ) -> (
        Qwen4ExpIq3GateUpCaptureBanks,
        crate::metal::MetalTensor,
        serde_json::Value,
    ) {
        const TOKENS: usize = 2_048;
        const LAYERS: usize = 43;
        const INPUT_BYTES: u64 = (LAYERS * 2_560 * TOKENS * 4) as u64;
        const SLOT_BYTES: u64 = (LAYERS * 512 * TOKENS * 4) as u64;
        const COUNT_BYTES: u64 = (LAYERS * 512 * 4) as u64;
        const OUTPUT_BYTES: u64 = (640 * 10 * TOKENS * 4) as u64;
        let specs = [
            ("inputs", INPUT_BYTES),
            ("slots", SLOT_BYTES),
            ("counts", COUNT_BYTES),
            ("output", OUTPUT_BYTES),
        ];
        let max_buffer_bytes = ctx.max_buffer_length() as u64;
        let page_bytes = host_page_size_bytes().unwrap() as u64;
        let mut priced_upper_bytes = 0_u64;
        let mut price_rows = Vec::new();
        for (name, logical_bytes) in specs {
            assert!(logical_bytes > 0 && logical_bytes <= max_buffer_bytes);
            let priced = ctx.shared_buffer_size_and_align(logical_bytes).unwrap();
            assert!(priced.size >= logical_bytes);
            assert!(priced.alignment.is_power_of_two());
            let alignment = priced.alignment.max(page_bytes);
            let upper_bytes = priced
                .size
                .checked_add(alignment - 1)
                .unwrap()
                .checked_div(alignment)
                .unwrap()
                .checked_mul(alignment)
                .unwrap();
            priced_upper_bytes = priced_upper_bytes.checked_add(upper_bytes).unwrap();
            price_rows.push(serde_json::json!({
                "name": name,
                "logical_bytes": logical_bytes,
                "metal_priced_bytes": priced.size,
                "metal_alignment_bytes": priced.alignment,
                "admission_alignment_bytes": alignment,
                "priced_upper_bytes": upper_bytes
            }));
        }
        let _transaction = ctx.begin_allocation_transaction();
        let before = ctx.memory_signals();
        let reserve_bytes =
            crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES;
        let admission =
            evaluate_metal_memory_admission(priced_upper_bytes, reserve_bytes, before, true);
        assert!(
            admission.admitted,
            "IQ3 gate/up capture admission denied: {}",
            admission.reason.as_str()
        );
        let inputs =
            crate::metal::MetalTensor::zeros_f32(ctx, vec![2_560, TOKENS as u64, LAYERS as u64])
                .unwrap();
        let slots =
            crate::metal::MetalTensor::zeros_i32(ctx, vec![TOKENS as u64, 512, LAYERS as u64])
                .unwrap();
        let counts = crate::metal::MetalTensor::zeros_i32(ctx, vec![512, LAYERS as u64]).unwrap();
        let output =
            crate::metal::MetalTensor::zeros_f32(ctx, vec![640, 10, TOKENS as u64]).unwrap();
        let after = ctx.memory_signals();
        let observed_bytes = after
            .current_allocated_bytes
            .saturating_sub(before.current_allocated_bytes);
        assert!(observed_bytes <= priced_upper_bytes);
        let banks = Qwen4ExpIq3GateUpCaptureBanks {
            inputs,
            counts,
            slots,
            tokens: TOKENS,
        };
        let evidence = serde_json::json!({
            "max_buffer_bytes": max_buffer_bytes,
            "host_page_bytes": page_bytes,
            "buffers": price_rows,
            "logical_bytes": INPUT_BYTES + SLOT_BYTES + COUNT_BYTES + OUTPUT_BYTES,
            "priced_upper_bytes": priced_upper_bytes,
            "reserve_bytes": reserve_bytes,
            "required_bytes": admission.required_bytes,
            "admission_reason": admission.reason.as_str(),
            "signals_before": metal_signals_json(before),
            "signals_after": metal_signals_json(after),
            "observed_allocation_delta_bytes": observed_bytes
        });
        (banks, output, evidence)
    }

    fn qualify_iq3_gate_up_capture_packets(
        banks: &Qwen4ExpIq3GateUpCaptureBanks,
        records: &[Qwen4ExpIq3GateUpCaptureRecord],
        expected_counts: &serde_json::Value,
    ) -> serde_json::Value {
        const EXPERTS: usize = 512;
        const TOP_K: usize = 10;

        let expected_counts = expected_counts.as_array().unwrap();
        assert_eq!(expected_counts.len(), 48);
        assert_eq!(records.len(), 43);
        let expected_routes = banks.tokens * TOP_K;
        let mut aggregate_routes = 0_u64;
        let mut aggregate_active_experts = 0_u64;
        let mut maximum_count = 0_usize;
        let mut layer_rows = Vec::with_capacity(records.len());
        for record in records {
            let counts = read_i32_tensor(&banks.counts_view(record.ordinal));
            let slots = read_i32_tensor(&banks.slots_view(record.ordinal));
            assert_eq!(counts.len(), EXPERTS);
            assert_eq!(slots.len(), EXPERTS * banks.tokens);
            let expected_layer = expected_counts[record.layer as usize].as_array().unwrap();
            assert_eq!(expected_layer.len(), EXPERTS);
            let mut seen_routes = vec![false; expected_routes];
            let mut layer_routes = 0_usize;
            let mut active_experts = 0_usize;
            let mut layer_maximum_count = 0_usize;
            for (expert, &raw_count) in counts.iter().enumerate() {
                let count = usize::try_from(raw_count).unwrap();
                assert!(
                    count <= banks.tokens,
                    "layer {} expert {expert} count",
                    record.layer
                );
                assert_eq!(
                    raw_count as i64,
                    expected_layer[expert].as_i64().unwrap(),
                    "layer {} expert {expert} imported count",
                    record.layer
                );
                layer_routes += count;
                active_experts += usize::from(count != 0);
                layer_maximum_count = layer_maximum_count.max(count);
                let start = expert * banks.tokens;
                for &raw_slot in &slots[start..start + count] {
                    let slot = usize::try_from(raw_slot).unwrap();
                    assert!(
                        slot < expected_routes,
                        "layer {} expert {expert} route slot {slot}",
                        record.layer
                    );
                    assert!(
                        !std::mem::replace(&mut seen_routes[slot], true),
                        "layer {} duplicate route slot {slot}",
                        record.layer
                    );
                }
            }
            assert_eq!(
                layer_routes, expected_routes,
                "layer {} route sum",
                record.layer
            );
            assert!(
                seen_routes.into_iter().all(|seen| seen),
                "layer {} route-slot permutation",
                record.layer
            );
            aggregate_routes += layer_routes as u64;
            aggregate_active_experts += active_experts as u64;
            maximum_count = maximum_count.max(layer_maximum_count);
            layer_rows.push(serde_json::json!({
                "ordinal": record.ordinal,
                "layer": record.layer,
                "route_sum": layer_routes,
                "active_experts": active_experts,
                "maximum_count": layer_maximum_count,
                "counts_equal_imported_census": true,
                "active_slots_are_exact_route_permutation": true
            }));
        }
        serde_json::json!({
            "records": records.len(),
            "tokens": banks.tokens,
            "aggregate_routes": aggregate_routes,
            "aggregate_active_experts": aggregate_active_experts,
            "maximum_count": maximum_count,
            "counts_equal_imported_census": true,
            "active_slots_are_exact_route_permutations": true,
            "hashes": {
                "inputs_sha256": sha256_metal_tensor_bytes(
                    b"qwen4exp-iq3-gate-up-capture-inputs-f32-v1\0",
                    &banks.inputs,
                ),
                "counts_sha256": sha256_metal_tensor_bytes(
                    b"qwen4exp-iq3-gate-up-capture-counts-i32-v1\0",
                    &banks.counts,
                ),
                "slots_sha256": sha256_metal_tensor_bytes(
                    b"qwen4exp-iq3-gate-up-capture-slots-i32-v1\0",
                    &banks.slots,
                )
            },
            "layers": layer_rows
        })
    }

    #[test]
    fn request_forward_limit_rounds_only_physical_qsa_rows() {
        let config = Qwen4ExpConfig::flash_next_reference();
        for (forward_limit, physical) in [
            (1, 4),
            (4, 4),
            (5, 8),
            (262_143, 262_144),
            (262_144, 262_144),
        ] {
            let capacity =
                Qwen4ExpSessionCapacity::for_forward_limit(&config, forward_limit).unwrap();
            assert_eq!(capacity.forward_limit(), forward_limit);
            assert_eq!(capacity.qsa_physical_capacity(), physical);
        }
        assert!(Qwen4ExpSessionCapacity::for_forward_limit(&config, 0).is_err());
        assert!(Qwen4ExpSessionCapacity::for_forward_limit(&config, 262_145).is_err());
    }

    #[test]
    fn prefill_plan_chunks_dense_and_selected_ranges_before_execution() {
        const C: usize = 2_048;
        const D: usize = 2_051;
        let cases = [
            (C, false, vec![0..C], C, false),
            (C + 1, false, vec![0..C], C, false),
            (
                2 * C - 1,
                true,
                vec![0..C, C..D, D..2 * C - 1],
                2 * C - 1,
                true,
            ),
            (2 * C, true, vec![0..C, C..D, D..2 * C], 2 * C, true),
            (
                2 * C + 1,
                true,
                vec![0..C, C..D, D..2 * C + 1],
                2 * C + 1,
                true,
            ),
            (D - 1, false, vec![0..C, C..D - 1], D - 1, false),
            (D, false, vec![0..C, C..D], D, false),
            (D + 1, false, vec![0..C, C..D], D, false),
            (D + 1, true, vec![0..C, C..D], D, false),
            (D + 2, true, vec![0..C, C..D, D..D + 2], D + 2, true),
            (D + C + 1, true, vec![0..C, C..D, D..D + C], D + C, true),
        ];
        for (tokens, selected_enabled, ranges, scalar_start, contains_selection) in cases {
            let plan =
                plan_qwen4exp_prefill_execution(tokens, Some(C), selected_enabled, D).unwrap();
            assert_eq!(plan.packed_ranges, ranges, "tokens={tokens}");
            assert_eq!(plan.packed_token_count, scalar_start, "tokens={tokens}");
            assert_eq!(plan.scalar_start, scalar_start, "tokens={tokens}");
            assert_eq!(
                plan.contains_selection, contains_selection,
                "tokens={tokens}"
            );
            assert!(plan.packed_ranges.iter().all(|range| {
                (2..=C).contains(&(range.end - range.start)) && (selected_enabled || range.end <= D)
            }));
        }

        let scalar = plan_qwen4exp_prefill_execution(18, None, false, D).unwrap();
        assert!(scalar.packed_ranges.is_empty());
        assert_eq!((scalar.packed_token_count, scalar.scalar_start), (0, 0));

        let under_hinted = plan_qwen4exp_prefill_execution(3_000, Some(100), false, D).unwrap();
        assert_eq!(under_hinted.packed_ranges.len(), 21);
        assert_eq!(under_hinted.packed_ranges.last(), Some(&(2_000..D)));
        assert_eq!(under_hinted.scalar_start, D);
        assert!(!under_hinted.contains_selection);

        assert!(plan_qwen4exp_prefill_execution(0, Some(C), true, D).is_err());
        assert!(plan_qwen4exp_prefill_execution(2, Some(1), false, D).is_err());
        assert!(plan_qwen4exp_prefill_execution(2, None, true, D).is_err());
        assert!(plan_qwen4exp_prefill_execution(2, Some(C), false, 0).is_err());
    }

    #[test]
    fn selected_prefill_execution_requires_explicit_runtime_opt_in() {
        let configured = qwen4exp_packed_selected_qsa_enabled();
        {
            let _override = Qwen4ExpPackedSelectedQsaOverride::set(false);
            assert!(!qwen4exp_packed_selected_qsa_enabled());
        }
        assert_eq!(qwen4exp_packed_selected_qsa_enabled(), configured);
        {
            let _override = Qwen4ExpPackedSelectedQsaOverride::set(true);
            assert!(qwen4exp_packed_selected_qsa_enabled());
        }
        assert_eq!(qwen4exp_packed_selected_qsa_enabled(), configured);
    }

    #[test]
    fn prefill_timing_retains_partial_gpu_coverage() {
        let mut timing = Qwen4ExpPrefillTiming::new(4_097, 4_096, false);
        timing.record(Qwen4ExpTokenTiming {
            position: 2_047,
            encode_cpu_ms: 1.0,
            completion_wait_ms: 4.0,
            gpu_ms: Some(3.0),
            total_wall_ms: 5.0,
        });
        timing.record(Qwen4ExpTokenTiming {
            position: 4_095,
            encode_cpu_ms: 1.0,
            completion_wait_ms: 4.0,
            gpu_ms: Some(3.5),
            total_wall_ms: 5.0,
        });
        timing.record(Qwen4ExpTokenTiming {
            position: 4_096,
            encode_cpu_ms: 1.0,
            completion_wait_ms: 4.0,
            gpu_ms: None,
            total_wall_ms: 5.0,
        });
        assert_eq!(timing.token_count, 4_097);
        assert_eq!(timing.packed_token_count, 4_096);
        assert!(!timing.contains_selection);
        assert_eq!(timing.token_count - timing.packed_token_count, 1);
        assert_eq!((timing.gpu_samples, timing.command_count), (2, 3));
        assert_eq!(timing.gpu_ms, 6.5);
        assert_eq!(timing.complete_gpu_ms(), None);
        assert_eq!(timing.outside_gpu_ms(), None);
    }

    #[test]
    fn packed_profile_resolves_nested_spans_without_double_counting() {
        let token = Qwen4ExpTokenTiming {
            position: 17,
            encode_cpu_ms: 2.0,
            completion_wait_ms: 6.0,
            gpu_ms: Some(5.0),
            total_wall_ms: 8.0,
        };
        let spans = [
            Qwen4ExpPackedProfileSpan {
                label: Qwen4ExpPackedProfileLabel::detail(
                    "moe.router",
                    2,
                    MixerKind::GatedDeltaNet,
                ),
                depth: 1,
                start_sample: 1,
                end_sample: 2,
            },
            Qwen4ExpPackedProfileSpan {
                label: Qwen4ExpPackedProfileLabel::coarse(
                    "post_ple_layer",
                    Some(2),
                    Some(MixerKind::GatedDeltaNet),
                ),
                depth: 0,
                start_sample: 0,
                end_sample: 3,
            },
        ];
        let profile = resolve_qwen4exp_packed_profile(
            token,
            Qwen4ExpPackedProfileEncodeTiming {
                preflight_ms: 0.5,
                stage_inputs_ms: 0.25,
                graph_encode_ms: 1.0,
                unattributed_ms: 0.25,
            },
            Qwen4ExpPackedProfileCommandTiming {
                commit_return_ms: 0.1,
                root_wait_ms: 5.0,
                child_publication_ms: 0.5,
                root_publish_ms: 0.1,
                release_total_ms: 5.6,
            },
            Qwen4ExpPackedProfileSampling::DispatchBoundary,
            None,
            4,
            &spans,
            &[10, 20, 25, 50],
        )
        .unwrap();
        assert_eq!(profile.sample_count, 4);
        assert_eq!(profile.sampled_span_ticks, 40);
        assert_eq!(profile.stages[0].label.name, "post_ple_layer");
        assert_eq!(profile.stages[1].label.name, "moe.router");
        assert_eq!(profile.stages[0].depth, 0);
        assert_eq!(profile.stages[1].depth, 1);
        assert!((profile.stages[0].gpu_ms - 5.0).abs() < 1e-12);
        assert!((profile.stages[1].gpu_ms - 0.625).abs() < 1e-12);
        assert!(
            resolve_qwen4exp_packed_profile(
                token,
                profile.encode,
                profile.command,
                profile.sampling,
                profile.sampling_fallback.clone(),
                4,
                &spans,
                &[10, 30, 20, 50],
            )
            .is_err()
        );
    }

    #[test]
    fn layer_profile_scales_stage_ticks_and_preserves_encoder_gaps() {
        let token = Qwen4ExpTokenTiming {
            position: 7,
            encode_cpu_ms: 1.0,
            completion_wait_ms: 6.0,
            gpu_ms: Some(5.0),
            total_wall_ms: 7.0,
        };
        let stages = [
            Qwen4ExpLayerStage::LayersZeroOne,
            Qwen4ExpLayerStage::PostPle {
                layer: 2,
                mixer: MixerKind::GatedDeltaNet,
            },
            Qwen4ExpLayerStage::Tail,
        ];
        let profile =
            resolve_qwen4exp_layer_profile(token, &stages, &[10, 20, 25, 45, 50, 60]).unwrap();
        assert_eq!(profile.token.position, 7);
        assert_eq!(profile.sampled_span_ticks, 50);
        assert_eq!(profile.stages.len(), 3);
        assert!((profile.stages[0].gpu_ms - 1.0).abs() < 1e-12);
        assert!((profile.stages[1].gpu_ms - 2.0).abs() < 1e-12);
        assert!((profile.stages[2].gpu_ms - 1.0).abs() < 1e-12);
        assert!((profile.encoder_boundary_ms - 1.0).abs() < 1e-12);
        assert!(resolve_qwen4exp_layer_profile(token, &stages, &[10, 9]).is_err());
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_RUNTIME_GGUF to the pinned full release"]
    fn released_runner_generates_expected_text_and_replays_prompt_logits() {
        let path = std::env::var_os("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF")
            .expect("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF must point to the first Q3 shard");
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let tokenizer = Tokenizer::from_gguf(&gguf).expect("load released tokenizer");
        let prompt = "<|im_start|>user\nReply with exactly: HELLO<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let prompt_tokens = tokenizer.encode(prompt, false).unwrap();
        assert_eq!(prompt_tokens.len(), 18);
        let prompt_tokens = prompt_tokens
            .into_iter()
            .map(|token| u32::try_from(token).unwrap())
            .collect::<Vec<_>>();
        let ctx = MetalContext::new().expect("initialize Metal");
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(
            &Qwen4ExpConfig::flash_next_reference(),
            prompt_tokens.len() + 7,
        )
        .unwrap();
        let mut loaded = Qwen4ExpLoadedModel::load(&ctx, &gguf, capacity).unwrap();
        let mut runner = loaded.create_runner(&ctx).unwrap();
        let invalid_prefill = match runner.prefill(&[prompt_tokens[0], 248_320]) {
            Ok(_) => panic!("invalid prefill token must fail before execution"),
            Err(error) => error.to_string(),
        };
        assert!(invalid_prefill.contains("prompt token 1 is invalid before prefill"));
        assert_eq!(runner.next_position(), 0);
        let continuation = [49_006_u32, 1_537];
        let baseline_rows = with_qwen4exp_moe_iq3_fast_override(false, || {
            let mut rows = vec![runner.prefill(&prompt_tokens).unwrap().to_vec()];
            for &token in &continuation {
                rows.push(runner.forward_token(token).unwrap().to_vec());
            }
            rows
        });
        let nonzero_prefill = match runner.prefill(&prompt_tokens) {
            Ok(_) => panic!("prefill must reject a session with committed state"),
            Err(error) => error.to_string(),
        };
        assert!(nonzero_prefill.contains("requires a reset session at position zero"));
        runner.reset().unwrap();
        let candidate_rows = with_qwen4exp_moe_iq3_fast_override(true, || {
            let mut rows = vec![runner.prefill(&prompt_tokens).unwrap().to_vec()];
            for &token in &continuation {
                rows.push(runner.forward_token(token).unwrap().to_vec());
            }
            rows
        });
        for (index, (baseline, candidate)) in baseline_rows.iter().zip(&candidate_rows).enumerate()
        {
            assert_logit_arms_close(&format!("released logits row {index}"), baseline, candidate);
        }
        runner.reset().unwrap();
        let first = with_qwen4exp_moe_iq3_fast_override(true, || {
            runner.prefill(&prompt_tokens).unwrap().to_vec()
        });
        let stop_tokens = gguf.stop_token_ids().unwrap();
        let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
        let mut logits = first.clone();
        let mut generated = Vec::new();
        let mut output = Vec::new();
        with_qwen4exp_moe_iq3_fast_override(true, || {
            for _ in 0..8 {
                let token = sampler.sample(&logits).unwrap().token;
                generated.push(token);
                if stop_tokens.contains(&token) {
                    break;
                }
                output.extend_from_slice(tokenizer.try_decode_piece_bytes_exact(token).unwrap());
                logits = runner
                    .forward_token(u32::try_from(token).unwrap())
                    .unwrap()
                    .to_vec();
            }
        });
        assert_eq!(generated, [49_006, 1_537, 248_046]);
        assert_eq!(output, b"HELLO");
        assert_eq!(runner.next_position(), prompt_tokens.len() + 2);
        runner.reset().unwrap();
        assert_eq!(runner.next_position(), 0);
        assert!(runner.logits().is_err());
        let (last, prefix) = prompt_tokens.split_last().unwrap();
        let outcome = with_qwen4exp_moe_iq3_fast_override(true, || {
            for &token in prefix {
                let _ = runner.forward_token(token).unwrap();
            }
            runner.forward_token_layer_profiled(*last).unwrap()
        });
        let profile = outcome.profile.unwrap();
        assert_eq!(profile.token.position, prompt_tokens.len() - 1);
        assert_eq!(profile.stages.len(), 48);
        assert!(profile.sampled_span_ticks > 0);
        assert!(profile.encoder_boundary_ms >= 0.0);
        let replay = runner.logits().unwrap().to_vec();
        assert_eq!(first, replay);
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_RUNTIME_GGUF to the pinned full release"]
    fn released_selected_prefill_scheduler_crosses_dense_boundary() {
        const PROMPT_LENGTHS: [usize; 3] = [2_053, 2_054, 2_055];
        const MAX_PROMPT_TOKENS: usize = 2_055;

        let path = std::env::var_os("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF")
            .expect("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF must point to the first Q3 shard");
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let tokenizer = Tokenizer::from_gguf(&gguf).expect("load released tokenizer");
        let seed = tokenizer
            .encode(
                "A careful systems test checks every causal boundary.\n",
                false,
            )
            .unwrap();
        assert!(seed.len() > 1);
        let tokens = seed
            .iter()
            .copied()
            .cycle()
            .take(MAX_PROMPT_TOKENS)
            .map(|token| u32::try_from(token).unwrap())
            .collect::<Vec<_>>();
        let ctx = MetalContext::new().expect("initialize Metal");
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(
            &Qwen4ExpConfig::flash_next_reference(),
            MAX_PROMPT_TOKENS + 1,
        )
        .unwrap();
        let mut loaded =
            Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, MAX_PROMPT_TOKENS)
                .unwrap();
        assert_eq!(loaded.packed_prefill_capacity(), Some(2_048));
        assert!(loaded.packed_selected_capable());
        {
            let _selected_override = Qwen4ExpPackedSelectedQsaOverride::set(false);
            assert!(!loaded.packed_selected_requested());
            assert!(!loaded.packed_selected_active());
        }
        {
            let _selected_override = Qwen4ExpPackedSelectedQsaOverride::set(true);
            assert!(loaded.packed_selected_requested());
            assert!(loaded.packed_selected_active());
        }
        let mut runner = loaded.create_runner(&ctx).unwrap();
        let selected_packet = [
            "kernel_qwen4exp_qsa_reset_selected_controls_i32",
            "kernel_qwen4exp_qsa_norm_rope_packed_f32",
            "kernel_qwen4exp_qsa_index_scores_packed_4x128_f16",
            "kernel_deepseek_v4_select_top_k_radix4_ids_f32",
            "kernel_qwen4exp_qsa_expand_ids_packed_i32",
            "kernel_qwen4exp_qsa_attention_logits_packed_f16",
            "kernel_qwen4exp_qsa_attention_softmax_value_packed_f16",
            "kernel_qwen4exp_qsa_audit_selected_i32",
        ];
        assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());

        for prompt_tokens in PROMPT_LENGTHS {
            runner.reset().unwrap();
            let prompt = &tokens[..prompt_tokens];
            let (default_logits, default_continuation) = {
                let _selected_override = Qwen4ExpPackedSelectedQsaOverride::set(false);
                crate::metal::dispatch_census_begin();
                let logits = runner.prefill(prompt).unwrap().to_vec();
                let census = crate::metal::dispatch_census_take();
                let timing = runner.last_prefill_timing().unwrap();
                assert_eq!(timing.token_count, prompt_tokens);
                assert_eq!(timing.packed_token_count, 2_051);
                assert!(!timing.contains_selection);
                assert_eq!(timing.command_count, 2 + prompt_tokens - 2_051);
                assert_eq!(
                    timing.token_count - timing.packed_token_count,
                    prompt_tokens - 2_051
                );
                assert!(runner.last_token_timing().is_some());
                assert_eq!(runner.next_position(), prompt_tokens);
                assert!(
                    runner
                        .workspace
                        .qsa_committed_lengths()
                        .iter()
                        .all(|(_, length)| *length == prompt_tokens)
                );
                assert!(!census.iter().any(|row| {
                    row.kernel == "kernel_qwen4exp_qsa_reset_selected_controls_i32"
                }));
                eprintln!(
                    "released default scheduler: tokens={prompt_tokens} commands={} packed_tokens={} wall_ms={:.3} gpu_ms={:?} tok/s={:.3}",
                    timing.command_count,
                    timing.packed_token_count,
                    timing.total_wall_ms,
                    timing.complete_gpu_ms(),
                    prompt_tokens as f64 / (timing.total_wall_ms / 1e3)
                );
                let continuation = runner.forward_token(tokens[0]).unwrap().to_vec();
                (logits, continuation)
            };

            runner.reset().unwrap();
            let (selected_logits, selected_continuation) = {
                let _selected_override = Qwen4ExpPackedSelectedQsaOverride::set(true);
                crate::metal::dispatch_census_begin();
                let logits = runner.prefill(prompt).unwrap().to_vec();
                let census = crate::metal::dispatch_census_take();
                let timing = runner.last_prefill_timing().unwrap();
                assert_eq!(timing.token_count, prompt_tokens);
                assert_eq!(timing.packed_token_count, prompt_tokens);
                assert!(timing.contains_selection);
                assert_eq!(timing.command_count, 3);
                assert_eq!(timing.token_count - timing.packed_token_count, 0);
                assert_eq!(runner.next_position(), prompt_tokens);
                assert!(
                    runner
                        .workspace
                        .qsa_committed_lengths()
                        .iter()
                        .all(|(_, length)| *length == prompt_tokens)
                );
                let selected_rows = prompt_tokens - 2_051;
                let selected_bands = selected_rows.div_ceil(32);
                assert_eq!(
                    census
                        .iter()
                        .filter(|row| {
                            row.kernel == "kernel_qwen4exp_qsa_reset_selected_controls_i32"
                        })
                        .count(),
                    12 * selected_bands
                );
                assert_eq!(
                    census
                        .iter()
                        .filter(|row| row.kernel == "kernel_qwen4exp_qsa_audit_selected_i32")
                        .count(),
                    12 * selected_bands
                );
                for ordinal in 0..selected_bands {
                    let tag = format!("qwen4exp.qsa.selected_band.{ordinal}");
                    let selected_dispatches = census
                        .iter()
                        .filter(|row| row.tag.as_deref() == Some(tag.as_str()))
                        .map(|row| row.kernel.as_str())
                        .collect::<Vec<_>>();
                    assert_eq!(selected_dispatches.len(), 12 * selected_packet.len());
                    assert!(
                        selected_dispatches
                            .chunks_exact(selected_packet.len())
                            .all(|packet| packet == selected_packet)
                    );
                }
                let exact_outputs = census
                    .iter()
                    .filter(|row| row.kernel == "kernel_mat_vec_q8_0_f32_lcpp_batch")
                    .collect::<Vec<_>>();
                assert_eq!(exact_outputs.len(), 12);
                assert!(
                    exact_outputs
                        .iter()
                        .all(|row| row.grid_height == selected_rows as u64)
                );
                eprintln!(
                    "released selected scheduler: tokens={prompt_tokens} commands={} wall_ms={:.3} gpu_ms={:?} tok/s={:.3}",
                    timing.command_count,
                    timing.total_wall_ms,
                    timing.complete_gpu_ms(),
                    prompt_tokens as f64 / (timing.total_wall_ms / 1e3)
                );
                let continuation = runner.forward_token(tokens[0]).unwrap().to_vec();
                assert_eq!(runner.next_position(), prompt_tokens + 1);
                assert!(
                    runner
                        .workspace
                        .qsa_committed_lengths()
                        .iter()
                        .all(|(_, length)| *length == prompt_tokens + 1)
                );
                (logits, continuation)
            };

            assert_released_packed_logit_arms_close(
                &format!("released selected/default N={prompt_tokens} endpoint"),
                &default_logits,
                &selected_logits,
            );
            assert_released_packed_logit_arms_close(
                &format!("released selected/default N={prompt_tokens} continuation"),
                &default_continuation,
                &selected_continuation,
            );
        }
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_RUNTIME_GGUF to the pinned full release"]
    fn released_packed_router_e8p32_strict_replays_exactly() {
        let path = std::env::var_os("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF")
            .expect("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF must point to the first Q3 shard");
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let tokenizer = Tokenizer::from_gguf(&gguf).expect("load released tokenizer");
        let prompt = "<|im_start|>user\nReply with exactly: HELLO<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let short_tokens = tokenizer
            .encode(prompt, false)
            .unwrap()
            .into_iter()
            .map(|token| u32::try_from(token).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(short_tokens.len(), 18);
        let marker = tokenizer.encode("<|im_start|>", false).unwrap();
        assert_eq!(marker.len(), 1);
        let marker = u32::try_from(marker[0]).unwrap();
        let long_tokens = vec![marker; 2_048];
        let ctx = MetalContext::new().expect("initialize Metal");
        assert_eq!(ctx.device.name().to_string(), "Apple M4 Max");
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(
            &Qwen4ExpConfig::flash_next_reference(),
            long_tokens.len() + 1,
        )
        .unwrap();
        let mut loaded =
            Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, long_tokens.len())
                .unwrap();
        let mut runner = loaded.create_runner(&ctx).unwrap();

        for (label, tokens) in [
            ("N=18", short_tokens.as_slice()),
            ("N=2048", long_tokens.as_slice()),
        ] {
            runner.reset().unwrap();
            zero_persistent_state(&runner);
            let baseline = with_qwen4exp_packed_router_e8p32_strict_override(false, || {
                crate::metal::dispatch_census_begin();
                let endpoint = runner.prefill(tokens).unwrap().to_vec();
                let census = crate::metal::dispatch_census_take();
                let timing = runner.last_prefill_timing().unwrap();
                assert_eq!(timing.packed_token_count, tokens.len(), "{label} baseline");
                assert_eq!(timing.command_count, 1, "{label} baseline commands");
                let prefill_state = snapshot_persistent_state(&runner);
                let prefill_qsa_lengths = runner.workspace.qsa_committed_lengths();
                let prefill_ple_prior_tokens = runner.workspace.ple_prior_tokens().to_vec();
                let continuation = runner.forward_token(marker).unwrap().to_vec();
                PackedRouterReplay {
                    endpoint,
                    continuation,
                    prefill_state,
                    continuation_state: snapshot_persistent_state(&runner),
                    prefill_qsa_lengths,
                    continuation_qsa_lengths: runner.workspace.qsa_committed_lengths(),
                    prefill_ple_prior_tokens,
                    continuation_ple_prior_tokens: runner.workspace.ple_prior_tokens().to_vec(),
                    census,
                }
            });
            assert_eq!(runner.next_position(), tokens.len() + 1);

            runner.reset().unwrap();
            zero_persistent_state(&runner);
            let candidate = with_qwen4exp_packed_router_e8p32_strict_override(true, || {
                crate::metal::dispatch_census_begin();
                let endpoint = runner.prefill(tokens).unwrap().to_vec();
                let census = crate::metal::dispatch_census_take();
                let timing = runner.last_prefill_timing().unwrap();
                assert_eq!(timing.packed_token_count, tokens.len(), "{label} candidate");
                assert_eq!(timing.command_count, 1, "{label} candidate commands");
                let prefill_state = snapshot_persistent_state(&runner);
                let prefill_qsa_lengths = runner.workspace.qsa_committed_lengths();
                let prefill_ple_prior_tokens = runner.workspace.ple_prior_tokens().to_vec();
                let continuation = runner.forward_token(marker).unwrap().to_vec();
                PackedRouterReplay {
                    endpoint,
                    continuation,
                    prefill_state,
                    continuation_state: snapshot_persistent_state(&runner),
                    prefill_qsa_lengths,
                    continuation_qsa_lengths: runner.workspace.qsa_committed_lengths(),
                    prefill_ple_prior_tokens,
                    continuation_ple_prior_tokens: runner.workspace.ple_prior_tokens().to_vec(),
                    census,
                }
            });
            assert_eq!(runner.next_position(), tokens.len() + 1);
            assert_f32_bits_eq(
                &format!("{label} endpoint logits"),
                &baseline.endpoint,
                &candidate.endpoint,
            );
            assert_f32_bits_eq(
                &format!("{label} continuation logits"),
                &baseline.continuation,
                &candidate.continuation,
            );
            assert_state_bytes_eq(
                &format!("{label} packed persistent state"),
                &baseline.prefill_state,
                &candidate.prefill_state,
            );
            assert_eq!(
                candidate.prefill_qsa_lengths, baseline.prefill_qsa_lengths,
                "{label} packed QSA lengths"
            );
            assert_eq!(
                candidate.prefill_ple_prior_tokens, baseline.prefill_ple_prior_tokens,
                "{label} packed PLE history"
            );
            assert_state_bytes_eq(
                &format!("{label} continuation persistent state"),
                &baseline.continuation_state,
                &candidate.continuation_state,
            );
            assert_eq!(
                candidate.continuation_qsa_lengths, baseline.continuation_qsa_lengths,
                "{label} continuation QSA lengths"
            );
            assert_eq!(
                candidate.continuation_ple_prior_tokens, baseline.continuation_ple_prior_tokens,
                "{label} continuation PLE history"
            );
            assert_router_candidate_census(
                label,
                tokens.len(),
                if tokens.len() == 2_048 { 48 } else { 0 },
                &baseline.census,
                &candidate.census,
            );
        }
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_RUNTIME_GGUF and QWEN4EXP_ROUTE_COUNT_CENSUS_OUT"]
    fn released_packed_route_count_capture_is_noop_and_complete() {
        const LAYERS: usize = 48;
        const EXPERTS: usize = 512;
        const TOP_K: usize = 10;
        const MODEL_REPOSITORY: &str = "unsloth/Qwen3.8-Flash-Next-GGUF";
        const MODEL_REVISION: &str = "8bdc666649440e9bdc97e16f3f75782c98478ff5";
        const MODEL_SHARDS: [(&str, u64, &str); 3] = [
            (
                "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
                10_946_624,
                "f2ef4328929d8b8c8930e2856eef52128dd4ce3425302f04bc3c657431cc4c49",
            ),
            (
                "Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf",
                49_983_253_824,
                "7d230e7c9421d868b89eebaf23033af0ea1a4e046956df00fb156814fb62346e",
            ),
            (
                "Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00003.gguf",
                39_992_153_376,
                "21d4f90f9cd7b7c3a1582667c20cb22f7b03de895b88a23bb20aaeaa44f2c199",
            ),
        ];

        let model_path = std::path::PathBuf::from(
            std::env::var_os("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF")
                .expect("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF must point to the first Q3 shard"),
        );
        let output_path = std::path::PathBuf::from(
            std::env::var_os("QWEN4EXP_ROUTE_COUNT_CENSUS_OUT")
                .expect("QWEN4EXP_ROUTE_COUNT_CENSUS_OUT must name the census JSON output"),
        );
        let declared_source_commit = std::env::var("QWEN4EXP_ROUTE_COUNT_CENSUS_SOURCE")
            .expect("QWEN4EXP_ROUTE_COUNT_CENSUS_SOURCE must be a lowercase full commit ID");
        assert!(
            is_lower_hex(&declared_source_commit, 40),
            "QWEN4EXP_ROUTE_COUNT_CENSUS_SOURCE must be 40 lowercase hex characters"
        );
        let declared_tracked_diff_sha256 = std::env::var("QWEN4EXP_ROUTE_COUNT_CENSUS_DIFF_SHA256")
            .expect("QWEN4EXP_ROUTE_COUNT_CENSUS_DIFF_SHA256 must identify the tracked diff");
        assert!(
            is_lower_hex(&declared_tracked_diff_sha256, 64),
            "QWEN4EXP_ROUTE_COUNT_CENSUS_DIFF_SHA256 must be 64 lowercase hex characters"
        );
        let test_executable = std::env::current_exe().expect("resolve current test executable");
        let test_executable_bytes = std::fs::metadata(&test_executable).unwrap().len();
        let test_executable_sha256 = sha256_file(&test_executable);
        let embedded_metallib_sha256 = sha256_bytes(crate::KERNELS_METALLIB);

        let gguf = GgufFile::open(&model_path).expect("open released UD-Q3_K_XL GGUF");
        assert_eq!(gguf.shards.len(), MODEL_SHARDS.len());
        let retained_shard_stamps = gguf
            .revalidate_retained_shard_stamps()
            .expect("revalidate retained model shards");
        assert_eq!(retained_shard_stamps.len(), MODEL_SHARDS.len());
        let mut expected_shard_manifest_domain =
            String::from("qwen4exp-route-count-expected-model-shard-manifest-v1\0");
        let mut local_shard_stamp_domain =
            String::from("qwen4exp-route-count-local-model-shard-stamps-v1\0");
        let model_shard_rows = gguf
            .shards
            .iter()
            .zip(MODEL_SHARDS)
            .enumerate()
            .map(
                |(index, (shard, (expected_name, expected_size, expected_lfs_sha256)))| {
                    let actual_name = shard
                        .path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap();
                    let actual_size = shard.mmap_len() as u64;
                    assert_eq!(actual_name, expected_name, "model shard {index} name");
                    assert_eq!(actual_size, expected_size, "model shard {index} size");
                    expected_shard_manifest_domain.push_str(&format!(
                        "{index}\t{expected_name}\t{expected_size}\t{expected_lfs_sha256}\n"
                    ));
                    let stamp = &retained_shard_stamps[index];
                    assert_eq!(stamp.shard_idx, index);
                    assert_eq!(stamp.path, shard.path);
                    local_shard_stamp_domain.push_str(&format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                        stamp.shard_idx,
                        stamp.device,
                        stamp.inode,
                        stamp.size,
                        stamp.mtime_sec,
                        stamp.mtime_nsec,
                        stamp.ctime_sec,
                        stamp.ctime_nsec,
                    ));
                    serde_json::json!({
                        "index": index,
                        "expected_release": {
                            "file_name": expected_name,
                            "size": expected_size,
                            "lfs_sha256": expected_lfs_sha256
                        },
                        "actual_local_file": {
                            "path": &stamp.path,
                            "file_name": actual_name,
                            "device": stamp.device,
                            "inode": stamp.inode,
                            "size": stamp.size,
                            "mtime_sec": stamp.mtime_sec,
                            "mtime_nsec": stamp.mtime_nsec,
                            "ctime_sec": stamp.ctime_sec,
                            "ctime_nsec": stamp.ctime_nsec
                        }
                    })
                },
            )
            .collect::<Vec<_>>();
        let expected_shard_manifest_sha256 =
            sha256_bytes(expected_shard_manifest_domain.as_bytes());
        let local_shard_stamp_sha256 = sha256_bytes(local_shard_stamp_domain.as_bytes());
        let tokenizer = Tokenizer::from_gguf(&gguf).expect("load released tokenizer");
        let prompt = "<|im_start|>user\nReply with exactly: HELLO<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let short_tokens = tokenizer
            .encode(prompt, false)
            .unwrap()
            .into_iter()
            .map(|token| u32::try_from(token).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(short_tokens.len(), 18);
        let marker = tokenizer.encode("<|im_start|>", false).unwrap();
        assert_eq!(marker.len(), 1);
        let marker = u32::try_from(marker[0]).unwrap();
        let long_tokens = vec![marker; 2_048];
        let natural_tokens = tokenizer
            .encode(include_str!("../../../docs/PERF-ROADMAP.md"), false)
            .unwrap()
            .into_iter()
            .take(2_048)
            .map(|token| u32::try_from(token).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(natural_tokens.len(), 2_048);

        let ctx = MetalContext::new().expect("initialize Metal");
        assert_eq!(ctx.device.name().to_string(), "Apple M4 Max");
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(
            &Qwen4ExpConfig::flash_next_reference(),
            long_tokens.len() + 1,
        )
        .unwrap();
        let mut loaded =
            Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, long_tokens.len())
                .unwrap();
        let released_config = loaded.config().clone();
        let mut runner = loaded.create_runner(&ctx).unwrap();

        let mut moe_weights = vec![
            runner.weights.zero_one.layer_zero.moe,
            runner.weights.zero_one.layer_one_moe,
        ];
        moe_weights.extend(runner.weights.post_ple.iter().map(|weights| weights.moe));
        assert_eq!(moe_weights.len(), LAYERS);
        let mut dtype_cohort_domain = String::from("qwen4exp-route-count-layer-dtype-cohort-v1\0");
        let layer_dtype_cohort = moe_weights
            .iter()
            .enumerate()
            .map(|(layer, weights)| {
                let mixer = released_config.mixer_kind(layer as u32).unwrap();
                let mixer = format!("{mixer:?}");
                let gate = format!("{:?}", weights.routed_gate.dtype);
                let up = format!("{:?}", weights.routed_up.dtype);
                let down = format!("{:?}", weights.routed_down.dtype);
                dtype_cohort_domain.push_str(&format!("{layer}\t{mixer}\t{gate}\t{up}\t{down}\n"));
                serde_json::json!({
                    "layer": layer,
                    "mixer": mixer,
                    "routed_gate": gate,
                    "routed_up": up,
                    "routed_down": down
                })
            })
            .collect::<Vec<_>>();
        let layer_dtype_cohort_sha256 = sha256_bytes(dtype_cohort_domain.as_bytes());
        let iq3_layers = moe_weights
            .iter()
            .enumerate()
            .filter_map(|(layer, weights)| {
                (weights.routed_gate.dtype == GgmlType::IQ3_XXS
                    && weights.routed_up.dtype == GgmlType::IQ3_XXS)
                    .then_some(layer)
            })
            .collect::<Vec<_>>();
        assert_eq!(iq3_layers.len(), 47);
        let credited_layers = (0..LAYERS)
            .filter(|layer| !matches!(layer, 2 | 4 | 30 | 46 | 47))
            .collect::<Vec<_>>();
        assert_eq!(credited_layers.len(), 43);

        let evidence_domain = format!(
            concat!(
                "qwen4exp-route-count-evidence-v2\0",
                "operator_declared_source_commit={declared_source_commit}\n",
                "operator_declared_tracked_diff_sha256={declared_tracked_diff_sha256}\n",
                "test_executable_sha256={test_executable_sha256}\n",
                "embedded_metallib_sha256={embedded_metallib_sha256}\n",
                "expected_model_repository={MODEL_REPOSITORY}\n",
                "expected_model_revision={MODEL_REVISION}\n",
                "expected_model_shard_manifest_sha256={expected_shard_manifest_sha256}\n",
                "actual_local_model_shard_stamps_sha256={local_shard_stamp_sha256}\n",
                "layer_dtype_cohort_sha256={layer_dtype_cohort_sha256}\n",
                "device_name={device_name}\n",
                "device_registry_id={device_registry_id}\n",
                "layers={LAYERS}\nexperts={EXPERTS}\ntop_k={TOP_K}\n"
            ),
            declared_source_commit = declared_source_commit,
            declared_tracked_diff_sha256 = declared_tracked_diff_sha256,
            test_executable_sha256 = test_executable_sha256,
            embedded_metallib_sha256 = embedded_metallib_sha256,
            MODEL_REPOSITORY = MODEL_REPOSITORY,
            MODEL_REVISION = MODEL_REVISION,
            expected_shard_manifest_sha256 = expected_shard_manifest_sha256,
            local_shard_stamp_sha256 = local_shard_stamp_sha256,
            layer_dtype_cohort_sha256 = layer_dtype_cohort_sha256,
            LAYERS = LAYERS,
            EXPERTS = EXPERTS,
            TOP_K = TOP_K,
            device_name = ctx.device.name(),
            device_registry_id = ctx.device.registryID()
        );
        let evidence_binding_sha256 = sha256_bytes(evidence_domain.as_bytes());

        let mut workload_rows = Vec::new();
        for (label, tokens, input_kind, classification) in [
            (
                "hello_n18",
                short_tokens.as_slice(),
                "handcrafted_canonical_dialogue",
                "canonical_interactive_correctness",
            ),
            (
                "repeated_im_start_n2048",
                long_tokens.as_slice(),
                "synthetic_repetition",
                "concentrated_route_stress",
            ),
            (
                "natural_roadmap_prefix_n2048",
                natural_tokens.as_slice(),
                "repository_document_prefix",
                "natural_technical_text",
            ),
        ] {
            runner.reset().unwrap();
            zero_persistent_state(&runner);
            let baseline = run_packed_replay(&mut runner, tokens, marker);
            assert_eq!(runner.next_position(), tokens.len() + 1);

            runner.reset().unwrap();
            zero_persistent_state(&runner);
            let capture =
                crate::metal::MetalTensor::zeros_i32(&ctx, vec![EXPERTS as u64, LAYERS as u64])
                    .unwrap();
            fill_i32_tensor(&capture, i32::MIN);
            let (captured, captured_layers) =
                with_qwen4exp_moe_route_count_capture(&capture, || {
                    run_packed_replay(&mut runner, tokens, marker)
                });
            assert_eq!(captured_layers, LAYERS, "{label} captured layers");
            assert_eq!(runner.next_position(), tokens.len() + 1);
            assert_packed_replay_bits_eq(label, &baseline, &captured);
            let baseline_digests = PackedReplayDigests::from_replay(&baseline);
            let captured_digests = PackedReplayDigests::from_replay(&captured);
            let baseline_binding = baseline_digests.binding_sha256(false);
            let captured_without_capture_binding = captured_digests.binding_sha256(true);
            assert_eq!(
                baseline_binding, captured_without_capture_binding,
                "{label} replay evidence after removing capture dispatches"
            );

            let counts = read_i32_tensor(&capture);
            assert_eq!(counts.len(), LAYERS * EXPERTS);
            assert!(!counts.contains(&i32::MIN));
            let rows = counts
                .chunks_exact(EXPERTS)
                .map(<[i32]>::to_vec)
                .collect::<Vec<_>>();
            assert_eq!(rows.len(), LAYERS);
            for (layer, row) in rows.iter().enumerate() {
                assert!(
                    row.iter()
                        .all(|&count| (0..=tokens.len() as i32).contains(&count)),
                    "{label} layer {layer} count range"
                );
                assert_eq!(
                    row.iter().sum::<i32>(),
                    (tokens.len() * TOP_K) as i32,
                    "{label} layer {layer} route sum"
                );
            }

            let layer_rows = rows
                .iter()
                .enumerate()
                .map(|(layer, row)| {
                    route_count_layer_json(
                        layer,
                        row,
                        tokens.len(),
                        moe_weights[layer].routed_gate.dtype,
                        moe_weights[layer].routed_up.dtype,
                        moe_weights[layer].routed_down.dtype,
                    )
                })
                .collect::<Vec<_>>();
            let all_layers = (0..LAYERS).collect::<Vec<_>>();
            let prompt_domain =
                format!("qwen4exp-route-count-prompt-u32le-v1;n={}\0", tokens.len());
            let prompt_sha256 = sha256_u32_le(prompt_domain.as_bytes(), tokens);
            let count_domain = format!(
                concat!(
                    "qwen4exp-route-count-matrix-i32le-v2\0",
                    "evidence_binding_sha256={evidence_binding_sha256}\n",
                    "label={label}\n",
                    "prompt_token_ids_sha256_u32le={prompt_sha256}\n",
                    "tokens={tokens}\n",
                    "layers={LAYERS}\nexperts={EXPERTS}\ntop_k={TOP_K}\n",
                    "layer_dtype_cohort_sha256={layer_dtype_cohort_sha256}\n",
                    "capture_off_replay_binding_sha256={baseline_binding}\n",
                    "capture_on_without_capture_binding_sha256={captured_without_capture_binding}\n",
                    "capture_on_raw_census_sha256={capture_on_raw_census_sha256}\n"
                ),
                evidence_binding_sha256 = evidence_binding_sha256,
                label = label,
                prompt_sha256 = prompt_sha256,
                LAYERS = LAYERS,
                EXPERTS = EXPERTS,
                TOP_K = TOP_K,
                layer_dtype_cohort_sha256 = layer_dtype_cohort_sha256,
                baseline_binding = baseline_binding,
                captured_without_capture_binding = captured_without_capture_binding,
                tokens = tokens.len(),
                capture_on_raw_census_sha256 = captured_digests.census_raw
            );
            workload_rows.push(serde_json::json!({
                "label": label,
                "tokens": tokens.len(),
                "input_kind": input_kind,
                "classification": classification,
                "population_representative": false,
                "lever_decision_scope": "single_sample",
                "prompt_domain_utf8": &prompt_domain,
                "prompt_token_ids": tokens,
                "prompt_token_ids_sha256_u32le": prompt_sha256,
                "payload_domain_utf8": &count_domain,
                "payload_sha256_i32le": sha256_i32_le(count_domain.as_bytes(), &counts),
                "replay_evidence": {
                    "capture_off": baseline_digests.json(),
                    "capture_on": captured_digests.json(),
                    "capture_off_equals_capture_on_without_capture_dispatches": true
                },
                "counts": rows,
                "layers": layer_rows,
                "aggregates": {
                    "all_48": route_count_aggregate_json("all_48", &all_layers, &rows, tokens.len()),
                    "iq3_gate_up_47": route_count_aggregate_json("iq3_gate_up_47", &iq3_layers, &rows, tokens.len()),
                    "credited_43": route_count_aggregate_json("credited_43", &credited_layers, &rows, tokens.len())
                }
            }));
        }

        let report = serde_json::json!({
            "schema_version": 2,
            "implementation": {
                "operator_declared_source_commit": declared_source_commit,
                "operator_declared_tracked_diff_sha256": declared_tracked_diff_sha256,
                "operator_declaration_verification": "lowercase hexadecimal syntax only; the test executable hash is the exact runnable identity",
                "tracked_diff_definition": "sha256(git diff --binary --no-ext-diff HEAD --)",
                "test_executable": {
                    "path": test_executable,
                    "size": test_executable_bytes,
                    "sha256": test_executable_sha256
                },
                "embedded_metallib": {
                    "size": crate::KERNELS_METALLIB.len(),
                    "sha256": embedded_metallib_sha256
                },
                "evidence_domain_utf8": evidence_domain,
                "evidence_binding_sha256": evidence_binding_sha256
            },
            "model": {
                "expected_release": {
                    "repository": MODEL_REPOSITORY,
                    "revision": MODEL_REVISION,
                    "quant": "UD-Q3_K_XL",
                    "qualification": "ordered local file names and byte sizes match this expected manifest; local bytes were not hashed against the expected LFS digests",
                    "shard_manifest_domain_utf8": expected_shard_manifest_domain,
                    "shard_manifest_sha256": expected_shard_manifest_sha256
                },
                "actual_local_files": {
                    "qualification": "retained file-descriptor stamps were revalidated after mmap and bind the exact local files, not their contents",
                    "stamp_domain_utf8": local_shard_stamp_domain,
                    "stamp_sha256": local_shard_stamp_sha256
                },
                "shards": model_shard_rows
            },
            "device": {
                "name": ctx.device.name().to_string(),
                "registry_id": ctx.device.registryID(),
                "max_threadgroup_memory_bytes": ctx.device.maxThreadgroupMemoryLength()
            },
            "geometry": {
                "layers": LAYERS,
                "experts": EXPERTS,
                "top_k": TOP_K,
                "hidden": 2_560,
                "routed_intermediate": 640,
                "layout": "layer_major_i32",
                "layer_dtype_cohort_domain_utf8": dtype_cohort_domain,
                "layer_dtype_cohort_sha256": layer_dtype_cohort_sha256,
                "layer_dtype_cohort": layer_dtype_cohort
            },
            "capture": {
                "dispatches": LAYERS,
                "kernel": "kernel_copy_offset_i32",
                "performance_eligible": false,
                "control_scope": "cfg(test) thread-local binding",
                "ordinary_runtime_capture_branch": false,
                "embedded_metallib_contains_dormant_copy_kernel": true,
                "capture_off_on_bits_equal": true,
                "non_capture_dispatch_topology_equal": true
            },
            "digest_encoding": {
                "integers": "little-endian after the emitted UTF-8 domain",
                "f32": "IEEE-754 bit patterns in little-endian order after the fixed field domain",
                "persistent_state": "fixed field domain followed by u64 little-endian tensor byte length and tensor bytes in session order",
                "dispatch_census": "fixed field domain followed by length-prefixed family/tag/kernel, encoder fields, and dispatch geometry in recorded order"
            },
            "workloads": workload_rows
        });
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output_path)
            .expect("census output path must not already exist");
        let mut report_bytes = serde_json::to_vec_pretty(&report).unwrap();
        report_bytes.push(b'\n');
        std::io::Write::write_all(&mut output, &report_bytes).unwrap();
        output.sync_all().unwrap();
        eprintln!(
            "wrote Qwen4Exp route-count census to {}",
            output_path.display()
        );
    }

    #[test]
    #[ignore = "set the released GGUF plus QWEN4EXP_IQ3_GATE_UP_PROBE_* evidence variables"]
    fn released_iq3_gate_up_range_probe_is_noop_and_measures_headroom() {
        const TOKENS: usize = 2_048;
        const MIRRORED_PAIRS: usize = 2;
        const LEAF_GATE: f64 = 0.10;
        const COMMAND_GATE: f64 = 0.01;
        const MAX_CONTROL_DRIFT: f64 = 0.02;
        const MAX_ADDITIVITY_RESIDUAL: f64 = 0.20;
        const PROBE_SENTINEL: u32 = 0x7fc0_38a1;

        let route_census_bytes = include_bytes!(
            "../../../docs/bench/2026-08-27-qwen4exp-iq3-gate-up-census/route-count-census.json"
        );
        let route_census_sha256 = sha256_bytes(route_census_bytes);
        assert_eq!(
            route_census_sha256,
            "ac48f12f0a0f5877e0c9261d69bdb0191103b2494b9d727a49f6bdb24d19ab08"
        );
        let route_census: serde_json::Value = serde_json::from_slice(route_census_bytes).unwrap();
        let natural_census = route_census["workloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|workload| workload["label"] == "natural_roadmap_prefix_n2048")
            .unwrap();
        let panels = &natural_census["aggregates"]["credited_43"]["panels"]["16"];
        let full_threadgroups = panels["full_panels"].as_u64().unwrap() * 10;
        let active_threadgroups = panels["active_panels"].as_u64().unwrap() * 10;
        let early_return_threadgroup_fraction =
            (full_threadgroups - active_threadgroups) as f64 / full_threadgroups as f64;

        let attribution_bytes = include_bytes!(
            "../../../docs/bench/2026-08-27-qwen4exp-packed-moe-attribution/results.json"
        );
        let attribution_sha256 = sha256_bytes(attribution_bytes);
        let attribution: serde_json::Value = serde_json::from_slice(attribution_bytes).unwrap();
        let credited_gate_up_command_fraction = attribution["workloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|workload| workload["tokens"] == TOKENS)
            .unwrap()["credited_command_share"]["routed_gate_up"]
            .as_f64()
            .unwrap();

        let model_path = std::path::PathBuf::from(
            std::env::var_os("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF")
                .expect("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF must point to the first Q3 shard"),
        );
        let output_path = std::path::PathBuf::from(
            std::env::var_os("QWEN4EXP_IQ3_GATE_UP_PROBE_OUT")
                .expect("QWEN4EXP_IQ3_GATE_UP_PROBE_OUT must name a new JSON report"),
        );
        let declared_source_commit = std::env::var("QWEN4EXP_IQ3_GATE_UP_PROBE_SOURCE")
            .expect("QWEN4EXP_IQ3_GATE_UP_PROBE_SOURCE must be a full commit ID");
        assert!(is_lower_hex(&declared_source_commit, 40));
        let declared_tracked_diff_sha256 = std::env::var("QWEN4EXP_IQ3_GATE_UP_PROBE_DIFF_SHA256")
            .expect("QWEN4EXP_IQ3_GATE_UP_PROBE_DIFF_SHA256 must identify the tracked diff");
        assert!(is_lower_hex(&declared_tracked_diff_sha256, 64));

        let test_executable = std::env::current_exe().unwrap();
        let test_executable_bytes = std::fs::metadata(&test_executable).unwrap().len();
        let test_executable_sha256 = sha256_file(&test_executable);
        let embedded_metallib_sha256 = sha256_bytes(crate::KERNELS_METALLIB);
        let gguf = GgufFile::open(&model_path).expect("open released UD-Q3_K_XL GGUF");
        let shard_stamps = gguf.revalidate_retained_shard_stamps().unwrap();
        assert_eq!(shard_stamps.len(), 3);
        let shard_rows = shard_stamps
            .iter()
            .map(|stamp| {
                serde_json::json!({
                    "index": stamp.shard_idx,
                    "path": &stamp.path,
                    "device": stamp.device,
                    "inode": stamp.inode,
                    "size": stamp.size,
                    "mtime_sec": stamp.mtime_sec,
                    "mtime_nsec": stamp.mtime_nsec,
                    "ctime_sec": stamp.ctime_sec,
                    "ctime_nsec": stamp.ctime_nsec
                })
            })
            .collect::<Vec<_>>();
        let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
        let tokens = natural_census["prompt_token_ids"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|token| u32::try_from(token.as_u64().unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(tokens.len(), TOKENS);
        let marker = tokenizer.encode("<|im_start|>", false).unwrap();
        assert_eq!(marker.len(), 1);
        let marker = u32::try_from(marker[0]).unwrap();
        let prompt_domain = format!("qwen4exp-iq3-gate-up-probe-prompt-u32le-v1;n={TOKENS}\0");
        let prompt_sha256 = sha256_u32_le(prompt_domain.as_bytes(), &tokens);

        let ctx = MetalContext::new().unwrap();
        assert_eq!(ctx.device.name().to_string(), "Apple M4 Max");
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(
            &Qwen4ExpConfig::flash_next_reference(),
            TOKENS + 1,
        )
        .unwrap();
        let mut loaded =
            Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, TOKENS).unwrap();
        let mut runner = loaded.create_runner(&ctx).unwrap();
        let mut moe_weights = vec![
            runner.weights.zero_one.layer_zero.moe,
            runner.weights.zero_one.layer_one_moe,
        ];
        moe_weights.extend(runner.weights.post_ple.iter().map(|weights| weights.moe));
        assert_eq!(moe_weights.len(), 48);
        let (capture_banks, replay_output, allocation_evidence) =
            allocate_iq3_gate_up_capture(&ctx);

        runner.reset().unwrap();
        zero_persistent_state(&runner);
        let _ = runner.prefill(&tokens).unwrap();
        let warm_timing = runner.last_prefill_timing().unwrap();

        runner.reset().unwrap();
        zero_persistent_state(&runner);
        let baseline = run_packed_replay(&mut runner, &tokens, marker);
        let baseline_timing = runner.last_prefill_timing().unwrap();
        let baseline_digests = PackedReplayDigests::from_replay(&baseline);

        runner.reset().unwrap();
        zero_persistent_state(&runner);
        let (captured, records) = with_qwen4exp_iq3_gate_up_capture(&capture_banks, || {
            run_packed_replay(&mut runner, &tokens, marker)
        });
        let capture_timing = runner.last_prefill_timing().unwrap();
        assert_packed_replay_state_bits_eq("capture", &baseline, &captured);
        assert_iq3_gate_up_capture_census("capture", &records, &baseline.census, &captured.census);
        let capture_digests = PackedReplayDigests::from_replay(&captured);
        let capture_packet_qualification = qualify_iq3_gate_up_capture_packets(
            &capture_banks,
            &records,
            &natural_census["counts"],
        );

        let mut qualification_rows = Vec::new();
        for arm in Qwen4ExpIq3GateUpProbeArm::ALL {
            fill_f32_tensor_bits(&replay_output, PROBE_SENTINEL);
            let (observation, census) = run_iq3_gate_up_range_command(
                &ctx,
                &capture_banks,
                &records,
                &moe_weights,
                &replay_output,
                arm,
                true,
            );
            assert_iq3_gate_up_range_census(arm, &records, &census);
            let no_work_output_unchanged = arm != Qwen4ExpIq3GateUpProbeArm::NoWork
                || read_f32_tensor_bits(&replay_output)
                    .iter()
                    .all(|&bits| bits == PROBE_SENTINEL);
            assert!(no_work_output_unchanged);
            qualification_rows.push(serde_json::json!({
                "arm": arm.as_str(),
                "standalone_dispatch_census_exact": true,
                "range_kernel_source_fixture": "qwen4exp_moe::tests::packed_common_moe_motor_matches_serial_rows_and_routes",
                "range_kernel_source_fixture_executed_by_this_test": false,
                "standalone_output_values_compared_by_this_test": false,
                "no_work_output_unchanged": no_work_output_unchanged,
                "dispatches": census.len(),
                "observation": observation.json()
            }));
        }

        let (standalone_warm, standalone_warm_census) = run_iq3_gate_up_range_command(
            &ctx,
            &capture_banks,
            &records,
            &moe_weights,
            &replay_output,
            Qwen4ExpIq3GateUpProbeArm::Full,
            false,
        );
        assert!(standalone_warm_census.is_empty());

        let forward = Qwen4ExpIq3GateUpProbeArm::ALL;
        let reverse = [
            Qwen4ExpIq3GateUpProbeArm::Full,
            Qwen4ExpIq3GateUpProbeArm::Count65Plus,
            Qwen4ExpIq3GateUpProbeArm::Count33To64,
            Qwen4ExpIq3GateUpProbeArm::Count17To32,
            Qwen4ExpIq3GateUpProbeArm::Count9To16,
            Qwen4ExpIq3GateUpProbeArm::Count1To8,
            Qwen4ExpIq3GateUpProbeArm::NoWork,
        ];
        let mut sequence_rows = Vec::new();
        let mut control_rows = Vec::new();
        let mut sequence_headroom = Vec::new();
        for pair in 0..MIRRORED_PAIRS {
            for (direction, order) in [("forward", forward), ("reverse", reverse)] {
                let sequence = sequence_rows.len();
                let (before, before_census) = run_iq3_gate_up_range_command(
                    &ctx,
                    &capture_banks,
                    &records,
                    &moe_weights,
                    &replay_output,
                    Qwen4ExpIq3GateUpProbeArm::Full,
                    false,
                );
                assert!(before_census.is_empty());
                control_rows.push(serde_json::json!({
                    "sequence": sequence,
                    "pair": pair,
                    "direction": direction,
                    "position": "before",
                    "observation": before.json()
                }));
                let observations = order
                    .into_iter()
                    .map(|arm| {
                        let (observation, census) = run_iq3_gate_up_range_command(
                            &ctx,
                            &capture_banks,
                            &records,
                            &moe_weights,
                            &replay_output,
                            arm,
                            false,
                        );
                        assert!(census.is_empty());
                        observation
                    })
                    .collect::<Vec<_>>();
                let (after, after_census) = run_iq3_gate_up_range_command(
                    &ctx,
                    &capture_banks,
                    &records,
                    &moe_weights,
                    &replay_output,
                    Qwen4ExpIq3GateUpProbeArm::Full,
                    false,
                );
                assert!(after_census.is_empty());
                control_rows.push(serde_json::json!({
                    "sequence": sequence,
                    "pair": pair,
                    "direction": direction,
                    "position": "after",
                    "observation": after.json()
                }));
                let gpu_ms = |arm| {
                    observations
                        .iter()
                        .find(|observation| observation.arm == arm)
                        .unwrap()
                        .command_gpu_ms
                };
                let no_work = gpu_ms(Qwen4ExpIq3GateUpProbeArm::NoWork);
                let full = gpu_ms(Qwen4ExpIq3GateUpProbeArm::Full);
                let optimistic_leaf_headroom = (no_work / full) * early_return_threadgroup_fraction;
                let optimistic_command_headroom =
                    optimistic_leaf_headroom * credited_gate_up_command_fraction;
                let band_increment_values = [
                    Qwen4ExpIq3GateUpProbeArm::Count1To8,
                    Qwen4ExpIq3GateUpProbeArm::Count9To16,
                    Qwen4ExpIq3GateUpProbeArm::Count17To32,
                    Qwen4ExpIq3GateUpProbeArm::Count33To64,
                    Qwen4ExpIq3GateUpProbeArm::Count65Plus,
                ]
                .into_iter()
                .map(|arm| (arm, gpu_ms(arm) - no_work))
                .collect::<Vec<_>>();
                let band_increments = band_increment_values
                    .iter()
                    .map(|&(arm, increment)| {
                        serde_json::json!({
                            "arm": arm.as_str(),
                            "gpu_ms_minus_no_work": increment,
                            "fraction_of_full": increment / full
                        })
                    })
                    .collect::<Vec<_>>();
                let additive_increment = band_increment_values
                    .iter()
                    .map(|&(_, increment)| increment)
                    .sum::<f64>();
                let full_increment = full - no_work;
                let control_mean = (before.command_gpu_ms + after.command_gpu_ms) * 0.5;
                let control_drift =
                    (after.command_gpu_ms - before.command_gpu_ms).abs() / control_mean;
                let full_index = observations
                    .iter()
                    .position(|observation| observation.arm == Qwen4ExpIq3GateUpProbeArm::Full)
                    .unwrap();
                let interpolation_fraction =
                    (full_index + 1) as f64 / (observations.len() + 1) as f64;
                let interpolated_full_control = before.command_gpu_ms
                    + (after.command_gpu_ms - before.command_gpu_ms) * interpolation_fraction;
                let full_control_agreement =
                    (full - interpolated_full_control).abs() / interpolated_full_control;
                let additivity_residual = additive_increment - full_increment;
                let additivity_residual_fraction = if full_increment > 0.0 {
                    additivity_residual.abs() / full_increment
                } else {
                    f64::INFINITY
                };
                let band_increments_nonnegative = band_increment_values
                    .iter()
                    .all(|&(_, increment)| increment >= 0.0);
                let sequence_valid = full_increment > 0.0
                    && band_increments_nonnegative
                    && additivity_residual_fraction <= MAX_ADDITIVITY_RESIDUAL
                    && control_drift <= MAX_CONTROL_DRIFT
                    && full_control_agreement <= MAX_CONTROL_DRIFT;
                sequence_headroom.push((
                    optimistic_leaf_headroom,
                    optimistic_command_headroom,
                    sequence_valid,
                ));
                sequence_rows.push(serde_json::json!({
                    "sequence": sequence,
                    "pair": pair,
                    "direction": direction,
                    "control_bracket_gpu_ms": [before.command_gpu_ms, after.command_gpu_ms],
                    "observations": observations.iter().map(Iq3GateUpRangeObservation::json).collect::<Vec<_>>(),
                    "derived": {
                        "no_work_gpu_ms": no_work,
                        "full_gpu_ms": full,
                        "full_minus_no_work_gpu_ms": full_increment,
                        "band_increments": band_increments,
                        "band_increments_nonnegative": band_increments_nonnegative,
                        "additivity_residual_gpu_ms": additivity_residual,
                        "absolute_additivity_residual_fraction_of_useful_increment": additivity_residual_fraction,
                        "control_drift_fraction": control_drift,
                        "full_position_between_controls": full_index,
                        "full_control_interpolation_fraction": interpolation_fraction,
                        "interpolated_full_control_gpu_ms": interpolated_full_control,
                        "full_control_agreement_fraction": full_control_agreement,
                        "sequence_valid": sequence_valid,
                        "optimistic_leaf_headroom_heuristic": optimistic_leaf_headroom,
                        "optimistic_command_headroom_heuristic": optimistic_command_headroom,
                        "clears_leaf_heuristic": optimistic_leaf_headroom >= LEAF_GATE,
                        "clears_command_heuristic": optimistic_command_headroom >= COMMAND_GATE
                    }
                }));
            }
        }
        let minimum_leaf_headroom = sequence_headroom
            .iter()
            .map(|&(leaf, _, _)| leaf)
            .fold(f64::INFINITY, f64::min);
        let minimum_command_headroom = sequence_headroom
            .iter()
            .map(|&(_, command, _)| command)
            .fold(f64::INFINITY, f64::min);
        let all_sequences_valid = sequence_headroom.iter().all(|&(_, _, valid)| valid);
        if !all_sequences_valid {
            panic!(
                "standalone IQ3 gate/up probe failed a validity gate:\n{}",
                serde_json::to_string_pretty(&sequence_rows).unwrap()
            );
        }
        let heuristic_prototype_triage_pass = all_sequences_valid
            && minimum_leaf_headroom >= LEAF_GATE
            && minimum_command_headroom >= COMMAND_GATE;

        let report = serde_json::json!({
            "schema_version": 2,
            "implementation": {
                "operator_declared_source_commit": declared_source_commit,
                "operator_declared_tracked_diff_sha256": declared_tracked_diff_sha256,
                "operator_declaration_verification": "lowercase hexadecimal syntax only; executable and metallib hashes identify the runnable artifacts",
                "tracked_diff_definition": "sha256(git diff --binary --no-ext-diff HEAD --)",
                "test_executable": {
                    "path": test_executable,
                    "size": test_executable_bytes,
                    "sha256": test_executable_sha256
                },
                "embedded_metallib": {
                    "size": crate::KERNELS_METALLIB.len(),
                    "sha256": embedded_metallib_sha256
                }
            },
            "imported_evidence": {
                "route_census": {
                    "path": "docs/bench/2026-08-27-qwen4exp-iq3-gate-up-census/route-count-census.json",
                    "sha256": route_census_sha256,
                    "evidence_binding_sha256": route_census["implementation"]["evidence_binding_sha256"],
                    "full_threadgroups": full_threadgroups,
                    "active_threadgroups": active_threadgroups,
                    "early_return_threadgroup_fraction": early_return_threadgroup_fraction
                },
                "packed_moe_attribution": {
                    "path": "docs/bench/2026-08-27-qwen4exp-packed-moe-attribution/results.json",
                    "sha256": attribution_sha256,
                    "source_commit": attribution["source_commit"],
                    "credited_gate_up_command_fraction": credited_gate_up_command_fraction,
                    "status": "bound imported attribution; not remeasured by this probe"
                }
            },
            "model": {
                "operator_expected_repository": "unsloth/Qwen3.8-Flash-Next-GGUF",
                "operator_expected_revision": "8bdc666649440e9bdc97e16f3f75782c98478ff5",
                "operator_expected_quant": "UD-Q3_K_XL",
                "identity_scope": "local files are bound only by revalidated retained descriptor stamps; this probe does not hash local bytes against expected LFS digests",
                "shards": shard_rows
            },
            "device": {
                "name": ctx.device.name().to_string(),
                "registry_id": ctx.device.registryID()
            },
            "workload": {
                "label": "natural_roadmap_prefix_n2048",
                "tokens": TOKENS,
                "population_representative": false,
                "lever_decision_scope": "single natural technical-text sample",
                "prompt_domain_utf8": prompt_domain,
                "prompt_token_ids": tokens,
                "prompt_token_ids_sha256_u32le": prompt_sha256
            },
            "protocol": {
                "purpose": "triage compact N16 active-panel descriptors before implementation",
                "performance_eligible": false,
                "observer": "supported MTLCommandBuffer GPUStartTime/GPUEndTime interval",
                "capture_once_then_replay": true,
                "capture_compilation_scope": "cfg(test); absent from non-test production builds; capture-off test execution retains a guarded TLS lookup",
                "dedicated_layer_major_capture_banks": true,
                "capture_dispatches_per_layer": 3,
                "standalone_dispatches_per_arm_command": records.len(),
                "one_arm_per_standalone_command": true,
                "production_dispatches_absent_from_standalone_commands": true,
                "same_command_arm_to_arm_warming_avoided": true,
                "cross_command_weight_and_output_residency_shared": true,
                "mirrored_pairs": MIRRORED_PAIRS,
                "measured_sequences": MIRRORED_PAIRS * 2,
                "credited_layers": 43,
                "early_return_threadgroup_fraction": early_return_threadgroup_fraction,
                "credited_gate_up_command_fraction": credited_gate_up_command_fraction,
                "optimistic_leaf_heuristic": LEAF_GATE,
                "optimistic_command_heuristic": COMMAND_GATE,
                "maximum_control_drift_fraction": MAX_CONTROL_DRIFT,
                "maximum_interpolated_full_control_disagreement_fraction": MAX_CONTROL_DRIFT,
                "maximum_absolute_additivity_residual_fraction_of_useful_increment": MAX_ADDITIVITY_RESIDUAL,
                "limitations": [
                    "no-work is the full direct grid filtered to perform no matrix writes",
                    "the headroom heuristic is optimistic, not a confidence bound or performance gate",
                    "subtraction does not predict descriptor-build or indirect-dispatch cost",
                    "fixed command and dispatch costs remain in both direct and future compact paths",
                    "standalone commands use dedicated capture addresses and may differ in TLB state from production",
                    "each command contains only 43 consecutive routed gate/up leaves rather than the intervening production graph"
                ]
            },
            "warmup": {
                "ordinary_model_prefill": {
                    "gpu_ms": warm_timing.gpu_ms,
                    "wall_ms": warm_timing.total_wall_ms
                },
                "standalone_full_command": standalone_warm.json()
            },
            "ordinary_baseline": {
                "gpu_ms": baseline_timing.gpu_ms,
                "wall_ms": baseline_timing.total_wall_ms,
                "replay_digests": baseline_digests.json()
            },
            "capture": {
                "allocation": allocation_evidence,
                "ordinary_replay_bits_equal": true,
                "ordinary_topology_after_filtering": true,
                "gpu_ms_with_capture": capture_timing.gpu_ms,
                "wall_ms_with_capture": capture_timing.total_wall_ms,
                "replay_digests": capture_digests.json(),
                "packets": capture_packet_qualification
            },
            "qualification": qualification_rows,
            "controls": control_rows,
            "sequences": sequence_rows,
            "decision": {
                "scope": "heuristic prototype triage only; cannot satisfy the candidate performance gate",
                "rule": "all four sequences must pass bracket drift, interpolated Full agreement, monotonic increment, and additivity checks; their minimum optimistic heuristics must clear both screens",
                "all_sequences_valid": all_sequences_valid,
                "minimum_optimistic_leaf_headroom_heuristic": minimum_leaf_headroom,
                "minimum_optimistic_command_headroom_heuristic": minimum_command_headroom,
                "heuristic_prototype_triage_pass": heuristic_prototype_triage_pass,
                "prototype_if_screened_in": "compact N16 active-panel descriptors plus indirect dispatch; retain arithmetic and 16 KiB TGM; require a separate paired candidate A/B"
            }
        });
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output_path)
            .expect("probe output path must not already exist");
        let mut report_bytes = serde_json::to_vec_pretty(&report).unwrap();
        report_bytes.push(b'\n');
        std::io::Write::write_all(&mut output, &report_bytes).unwrap();
        output.sync_all().unwrap();
        eprintln!("wrote IQ3 gate/up probe to {}", output_path.display());
    }
}
