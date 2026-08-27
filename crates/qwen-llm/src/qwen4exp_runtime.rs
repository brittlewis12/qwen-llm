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
use std::time::Instant;

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
    pub packed_token_count: usize,
    pub command_count: usize,
    pub encode_cpu_ms: f64,
    pub completion_wait_ms: f64,
    pub gpu_ms: f64,
    pub gpu_samples: usize,
    pub total_wall_ms: f64,
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
    fn new(token_count: usize, packed_token_count: usize) -> Self {
        Self {
            token_count,
            packed_token_count,
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
        let packed_tokens = self
            .workspace
            .packed_prefill_capacity()
            .unwrap_or(0)
            .min(token_ids.len());
        let packed_tokens = if packed_tokens >= 2 { packed_tokens } else { 0 };
        let mut prefill_timing = Qwen4ExpPrefillTiming::new(token_ids.len(), packed_tokens);
        if packed_tokens != 0 {
            self.run_prefill_checkpoint(start, token_ids.len(), &mut checkpoint)?;
            match execute_qwen4exp_text_packed_sync(
                self.ctx,
                &token_ids[..packed_tokens],
                self.ple_table,
                &self.weights,
                &mut self.workspace,
            ) {
                Ok(timing) => {
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
        for &token_id in &token_ids[packed_tokens..] {
            self.run_prefill_checkpoint(start, token_ids.len(), &mut checkpoint)?;
            match execute_qwen4exp_text_token_sync(
                self.ctx,
                token_id,
                self.ple_table,
                &self.weights,
                &mut self.workspace,
            ) {
                Ok(timing) => {
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
        let mut timing = Qwen4ExpPrefillTiming::new(token_ids.len(), token_ids.len());
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
    use crate::qwen4exp_moe::with_qwen4exp_moe_iq3_fast_override;
    use crate::sampling::{Sampler, SamplingConfig};
    use crate::tokenizer::Tokenizer;

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

    fn assert_logit_arms_close(label: &str, baseline: &[f32], candidate: &[f32]) {
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
        assert_eq!(argmax(baseline), argmax(candidate), "{label} argmax");
        assert!(cosine > 0.999_999_99, "{label} cosine {cosine}");
        assert!(relative_rms < 1e-4, "{label} relative RMS {relative_rms}");
        assert!(max_abs < 1e-3, "{label} maximum delta {max_abs}");
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
    fn prefill_timing_retains_partial_gpu_coverage() {
        let mut timing = Qwen4ExpPrefillTiming::new(2_050, 2_048);
        timing.record(Qwen4ExpTokenTiming {
            position: 2_047,
            encode_cpu_ms: 1.0,
            completion_wait_ms: 4.0,
            gpu_ms: Some(3.0),
            total_wall_ms: 5.0,
        });
        timing.record(Qwen4ExpTokenTiming {
            position: 2_048,
            encode_cpu_ms: 1.0,
            completion_wait_ms: 4.0,
            gpu_ms: None,
            total_wall_ms: 5.0,
        });
        assert_eq!(timing.token_count, 2_050);
        assert_eq!(timing.packed_token_count, 2_048);
        assert_eq!((timing.gpu_samples, timing.command_count), (1, 2));
        assert_eq!(timing.gpu_ms, 3.0);
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
                start_sample: 1,
                end_sample: 2,
            },
            Qwen4ExpPackedProfileSpan {
                label: Qwen4ExpPackedProfileLabel::coarse(
                    "post_ple_layer",
                    Some(2),
                    Some(MixerKind::GatedDeltaNet),
                ),
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
}
