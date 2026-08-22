use super::*;
use qwen_llm::dense_batch8::DENSE_BATCH8_WIDTH;
use qwen_llm::moe_batch16::{HeadMode, MOE_BATCH16_WIDTH, MoeBatch16PlanTelemetry};
use qwen_llm::runtime::{DenseBatch8SequenceExecutor, MoeBatch16SequenceExecutor, RuntimeError};

const FINISHED_LANE_FILL_TOKEN: i32 = 0;
const TRANSIENT_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const PREFIX_FANOUT_MIN_TOKENS: usize = 256;
const MIN_REQUESTED_TRANSITION_UTILIZATION_NUMERATOR: usize = 3;
const MIN_REQUESTED_TRANSITION_UTILIZATION_DENOMINATOR: usize = 4;
const REFILL_MIN_STEP_SAVINGS_NUMERATOR: usize = 1;
const REFILL_MIN_STEP_SAVINGS_DENOMINATOR: usize = 10;
const DEFAULT_REFILL_MAX_PROMPT_TOKENS: usize = 64;
const AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS: usize = 256;
const AUTOMATIC_DENSE_RAGGED_MIN_GENERATION_TOKENS: usize = 32;
const AUTOMATIC_DENSE_RAGGED_MAX_PREFILL_PER_TRANSITION: usize = 2;
const AUTOMATIC_DENSE_RAGGED_REFILL_MAX_GENERATION_TOKENS: usize = 40;
const AUTOMATIC_DENSE_RAGGED_REFILL_MIN_ARENAS: usize = 2;
const AUTOMATIC_DENSE_RAGGED_REFILL_MAX_PREFILL_PER_TRANSITION: usize = 9;
const AUTOMATIC_DENSE_RAGGED_REFILL_MAX_ARENA_PREFILL_PER_TRANSITION: usize = 32;
const AUTOMATIC_DENSE_RAGGED_REFILL_MIN_UTILIZATION_NUMERATOR: usize = 9;
const AUTOMATIC_DENSE_RAGGED_REFILL_MIN_UTILIZATION_DENOMINATOR: usize = 10;
const AUTOMATIC_DENSE_RAGGED_REFILL_MIN_SAVINGS_NUMERATOR: usize = 3;
const AUTOMATIC_DENSE_RAGGED_REFILL_MIN_SAVINGS_DENOMINATOR: usize = 20;
const DENSE_PREFIX_FANOUT_ENV: &str = "QWEN_DENSE_BATCH8_PREFIX_FANOUT";
const MOE_PREFIX_FANOUT_ENV: &str = "QWEN_MOE_BATCH16_PREFIX_FANOUT";
const PREFIX_PACKING_ENV: &str = "QWEN_FIXED_COHORT_PREFIX_PACKING";
const FILE_ROOT_FANOUT_ENV: &str = "QWEN_FIXED_COHORT_FILE_ROOT_FANOUT";
// Explicitly broadens or disables the charged automatic dense policy.
const RAGGED_PROMPTS_ENV: &str = "QWEN_FIXED_COHORT_RAGGED_PROMPTS";
// Charged only for dense B8; synchronous MoE B16 refill loses to B2.
const DENSE_REFILL_ENV: &str = "QWEN_DENSE_BATCH8_REFILL";

#[derive(Clone, Copy, Debug)]
struct FixedCohortStep<const WIDTH: usize> {
    argmax_ids: [i32; WIDTH],
    gpu_ms: Option<f64>,
}

trait FixedCohortExecutor<const WIDTH: usize> {
    const DISPLAY_NAME: &'static str;
    const PREFIX_FANOUT_ENV: &'static str;
    const COHORT_BACKEND: &'static str;
    const PLANNER_BACKEND: &'static str;
    const TELEMETRY_PREFIX: &'static str;
    const PLANNER_TELEMETRY_PREFIX: &'static str;
    const TELEMETRY_SCHEMA_VERSION: u32;
    const PLANNER_TELEMETRY_SCHEMA_VERSION: u32;
    const REFILL_TELEMETRY: bool = false;

    fn executor_scratch_upper_bytes(loaded: &LoadedModel) -> Result<u64>;

    fn validate(
        &self,
        token_ids: [i32; WIDTH],
        sequences: [&mut Sequence; WIDTH],
    ) -> Result<(), RuntimeError>;

    fn step_greedy(
        &mut self,
        token_ids: [i32; WIDTH],
        sequences: [&mut Sequence; WIDTH],
        cancelled: impl Fn() -> bool,
    ) -> Result<FixedCohortStep<WIDTH>, RuntimeError>;

    fn scratch_bytes(&self) -> Option<u64> {
        None
    }

    fn moe_plan_telemetry(&self) -> Option<MoeBatch16PlanTelemetry> {
        None
    }
}

impl FixedCohortExecutor<DENSE_BATCH8_WIDTH> for DenseBatch8SequenceExecutor<'_> {
    const DISPLAY_NAME: &'static str = "dense B=8";
    const PREFIX_FANOUT_ENV: &'static str = DENSE_PREFIX_FANOUT_ENV;
    const COHORT_BACKEND: &'static str = "dense_qwen_static_batch8_v6";
    const PLANNER_BACKEND: &'static str = "dense_qwen_fixed_cohort_planner_v9";
    const TELEMETRY_PREFIX: &'static str = "dense_batch8";
    const PLANNER_TELEMETRY_PREFIX: &'static str = "dense_batch8_planner";
    const TELEMETRY_SCHEMA_VERSION: u32 = 7;
    const PLANNER_TELEMETRY_SCHEMA_VERSION: u32 = 9;
    const REFILL_TELEMETRY: bool = true;

    fn executor_scratch_upper_bytes(loaded: &LoadedModel) -> Result<u64> {
        Ok(loaded.dense_batch8_scratch_upper_bytes()?)
    }

    fn validate(
        &self,
        token_ids: [i32; DENSE_BATCH8_WIDTH],
        sequences: [&mut Sequence; DENSE_BATCH8_WIDTH],
    ) -> Result<(), RuntimeError> {
        let [s0, s1, s2, s3, s4, s5, s6, s7] = sequences;
        DenseBatch8SequenceExecutor::validate(self, token_ids, [s0, s1, s2, s3, s4, s5, s6, s7])
    }

    fn step_greedy(
        &mut self,
        token_ids: [i32; DENSE_BATCH8_WIDTH],
        sequences: [&mut Sequence; DENSE_BATCH8_WIDTH],
        cancelled: impl Fn() -> bool,
    ) -> Result<FixedCohortStep<DENSE_BATCH8_WIDTH>, RuntimeError> {
        let [s0, s1, s2, s3, s4, s5, s6, s7] = sequences;
        let step = DenseBatch8SequenceExecutor::step_greedy(
            self,
            token_ids,
            [s0, s1, s2, s3, s4, s5, s6, s7],
            cancelled,
        )?;
        Ok(FixedCohortStep {
            argmax_ids: step.argmax_ids,
            gpu_ms: step.gpu_ms,
        })
    }
}

impl FixedCohortExecutor<MOE_BATCH16_WIDTH> for MoeBatch16SequenceExecutor<'_> {
    const DISPLAY_NAME: &'static str = "MoE B=16";
    const PREFIX_FANOUT_ENV: &'static str = MOE_PREFIX_FANOUT_ENV;
    const COHORT_BACKEND: &'static str = "qwen_moe_capability_batch16_v6";
    const PLANNER_BACKEND: &'static str = "qwen_moe_fixed_cohort_planner_v6";
    const TELEMETRY_PREFIX: &'static str = "moe_batch16";
    const PLANNER_TELEMETRY_PREFIX: &'static str = "moe_batch16_planner";
    const TELEMETRY_SCHEMA_VERSION: u32 = 8;
    const PLANNER_TELEMETRY_SCHEMA_VERSION: u32 = 6;

    fn executor_scratch_upper_bytes(loaded: &LoadedModel) -> Result<u64> {
        Ok(loaded.moe_batch16_scratch_upper_bytes()?)
    }

    fn validate(
        &self,
        token_ids: [i32; MOE_BATCH16_WIDTH],
        sequences: [&mut Sequence; MOE_BATCH16_WIDTH],
    ) -> Result<(), RuntimeError> {
        MoeBatch16SequenceExecutor::validate(self, token_ids, sequences)
    }

    fn step_greedy(
        &mut self,
        token_ids: [i32; MOE_BATCH16_WIDTH],
        sequences: [&mut Sequence; MOE_BATCH16_WIDTH],
        cancelled: impl Fn() -> bool,
    ) -> Result<FixedCohortStep<MOE_BATCH16_WIDTH>, RuntimeError> {
        let step = MoeBatch16SequenceExecutor::step_greedy(self, token_ids, sequences, cancelled)?;
        Ok(FixedCohortStep {
            argmax_ids: step.argmax_ids,
            gpu_ms: step.gpu_ms,
        })
    }

    fn scratch_bytes(&self) -> Option<u64> {
        Some(MoeBatch16SequenceExecutor::scratch_bytes(self))
    }

    fn moe_plan_telemetry(&self) -> Option<MoeBatch16PlanTelemetry> {
        Some(MoeBatch16SequenceExecutor::plan_telemetry(self))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PrefixFanoutPlan {
    common_prefix_tokens: usize,
    selected_prefix_tokens: usize,
    reason: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CohortCompatibility {
    EqualPromptTokens(usize),
    RaggedPrompts,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RaggedPromptPolicy {
    Disabled,
    Explicit,
    AutomaticDenseCharged,
}

impl RaggedPromptPolicy {
    fn enabled(self) -> bool {
        self != Self::Disabled
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Explicit => "explicit",
            Self::AutomaticDenseCharged => "automatic_dense_charged",
        }
    }
}

impl From<bool> for RaggedPromptPolicy {
    fn from(enabled: bool) -> Self {
        if enabled {
            Self::Explicit
        } else {
            Self::Disabled
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RaggedPromptDecision {
    Disabled,
    Explicit,
    AutomaticAdmitted,
    AutomaticEnvelopeRejected,
    AutomaticChargeRejected,
    AutomaticNoCohortGain,
    AutomaticRefillAdmitted,
    AutomaticRefillDisabled,
    AutomaticRefillEnvelopeRejected,
    AutomaticRefillChargeRejected,
    AutomaticRefillNoCohortGain,
}

impl RaggedPromptDecision {
    fn for_policy(policy: RaggedPromptPolicy) -> Self {
        match policy {
            RaggedPromptPolicy::Disabled => Self::Disabled,
            RaggedPromptPolicy::Explicit => Self::Explicit,
            RaggedPromptPolicy::AutomaticDenseCharged => Self::AutomaticAdmitted,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Explicit => "explicit",
            Self::AutomaticAdmitted => "automatic_dense_admitted",
            Self::AutomaticEnvelopeRejected => "automatic_dense_envelope_rejected",
            Self::AutomaticChargeRejected => "automatic_dense_charge_rejected",
            Self::AutomaticNoCohortGain => "automatic_dense_no_cohort_gain",
            Self::AutomaticRefillAdmitted => "automatic_dense_refill_admitted",
            Self::AutomaticRefillDisabled => "automatic_dense_refill_disabled",
            Self::AutomaticRefillEnvelopeRejected => "automatic_dense_refill_envelope_rejected",
            Self::AutomaticRefillChargeRejected => "automatic_dense_refill_charge_rejected",
            Self::AutomaticRefillNoCohortGain => "automatic_dense_refill_no_cohort_gain",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum RefillFallback<const WIDTH: usize> {
    Batch([usize; WIDTH]),
    Serial(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DenseRefillPolicy {
    Disabled,
    ShortSerialFallbackRescue,
    AutomaticChargedRagged,
    Forced,
}

impl DenseRefillPolicy {
    fn enabled(self) -> bool {
        self != Self::Disabled
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::ShortSerialFallbackRescue => "short_serial_fallback_rescue",
            Self::AutomaticChargedRagged => "automatic_charged_ragged",
            Self::Forced => "forced",
        }
    }

    fn default_max_prompt_tokens(self) -> Option<usize> {
        match self {
            Self::ShortSerialFallbackRescue => Some(DEFAULT_REFILL_MAX_PROMPT_TOKENS),
            Self::AutomaticChargedRagged => Some(AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS),
            Self::Disabled | Self::Forced => None,
        }
    }
}

impl From<bool> for DenseRefillPolicy {
    fn from(enabled: bool) -> Self {
        if enabled {
            Self::Forced
        } else {
            Self::Disabled
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RefillArenaPlan<const WIDTH: usize> {
    fallback_work: Vec<RefillFallback<WIDTH>>,
    request_indices: Vec<usize>,
    shared_capacity: usize,
    simulated_physical_steps: usize,
    simulated_productive_transitions: usize,
    requested_two_wave_steps: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PlannedWork<const WIDTH: usize> {
    Batch([usize; WIDTH]),
    RefillArena(RefillArenaPlan<WIDTH>),
    Serial(usize),
}

impl<const WIDTH: usize> PlannedWork<WIDTH> {
    fn first_request_index(&self) -> usize {
        match self {
            Self::Batch(indices) => *indices.iter().min().expect("nonempty fixed cohort"),
            Self::RefillArena(plan) => *plan
                .request_indices
                .iter()
                .min()
                .expect("nonempty refill arena"),
            Self::Serial(index) => *index,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RaggedPackingStrategy {
    Disabled,
    GenerationDepth,
    Capacity,
    Mixed,
}

impl RaggedPackingStrategy {
    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Disabled, value) | (value, Self::Disabled) => value,
            (left, right) if left == right => left,
            _ => Self::Mixed,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::GenerationDepth => "generation_depth",
            Self::Capacity => "capacity",
            Self::Mixed => "mixed",
        }
    }
}

#[derive(Debug)]
struct AutomaticRefillFallback<const WIDTH: usize> {
    work: Vec<PlannedWork<WIDTH>>,
    prefix_affinity_batches: Vec<[usize; WIDTH]>,
}

#[derive(Debug)]
struct CohortPlan<const WIDTH: usize> {
    work: Vec<PlannedWork<WIDTH>>,
    prefix_packing_configured: bool,
    prefix_packing_effective: bool,
    prefix_fanout_boundary_policy: QwenPrefixFanoutBoundaryPolicy,
    ragged_prompt_policy: RaggedPromptPolicy,
    ragged_prompt_plan_decision: RaggedPromptDecision,
    ragged_packing_strategy: RaggedPackingStrategy,
    compatibility_buckets: usize,
    candidate_cohorts: usize,
    prefix_plan_fallback_buckets: usize,
    full_cohorts: usize,
    economics_rejected_cohorts: usize,
    serial_fallback_requests: usize,
    prefix_affinity_batches: Vec<[usize; WIDTH]>,
    refill_policy: DenseRefillPolicy,
    refill_configured: bool,
    refill_effective: bool,
    refill_arenas: usize,
    refill_requests: usize,
    max_execution_capacity: usize,
    automatic_refill_fallback: Option<AutomaticRefillFallback<WIDTH>>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CohortPlanSummary {
    pub ragged_prompt_policy: Option<&'static str>,
    pub ragged_prompt_plan_decision: Option<&'static str>,
    pub refill_policy: Option<&'static str>,
    pub planned_refill_arenas: Option<usize>,
    pub planned_refill_requests: Option<usize>,
    pub full_cohorts: usize,
    pub economics_rejected_cohorts: usize,
    pub serial_fallback_requests: usize,
    pub max_execution_capacity: usize,
}

#[derive(Debug)]
struct LaneProgress {
    max_tokens: usize,
    generated: Vec<i32>,
    stop_reason: Option<StopReason>,
    logical_transitions: usize,
    padding_transitions: usize,
}

impl LaneProgress {
    fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            generated: Vec::with_capacity(max_tokens),
            stop_reason: None,
            logical_transitions: 0,
            padding_transitions: 0,
        }
    }

    fn is_active(&self) -> bool {
        self.stop_reason.is_none()
    }

    fn transition_token(&self) -> i32 {
        self.generated
            .last()
            .copied()
            .filter(|_| self.is_active())
            .unwrap_or(FINISHED_LANE_FILL_TOKEN)
    }

    /// Record one selected token and return whether its piece is visible.
    fn record_selection(&mut self, token: i32, stop_tokens: &[i32]) -> Result<bool> {
        ensure!(
            self.is_active(),
            "cannot select into a finished fixed-cohort lane"
        );
        ensure!(
            self.generated.len() < self.max_tokens,
            "fixed-cohort lane exceeded its token limit"
        );
        self.generated.push(token);
        if stop_tokens.contains(&token) {
            self.stop_reason = Some(StopReason::Eos);
            return Ok(false);
        }
        if self.generated.len() == self.max_tokens {
            self.stop_reason = Some(StopReason::TokenLimit);
        }
        Ok(true)
    }

    fn record_batch_step(
        &mut self,
        was_active: bool,
        selected: i32,
        stop_tokens: &[i32],
    ) -> Result<bool> {
        ensure!(
            was_active == self.is_active(),
            "fixed-cohort active-lane snapshot drifted"
        );
        if was_active {
            self.logical_transitions += 1;
            self.record_selection(selected, stop_tokens)
        } else {
            self.padding_transitions += 1;
            Ok(false)
        }
    }

    fn validate_request_complete(&self) -> Result<()> {
        ensure!(
            self.stop_reason.is_some(),
            "fixed-cohort lane did not terminate"
        );
        ensure!(
            self.logical_transitions.checked_add(1) == Some(self.generated.len()),
            "fixed-cohort lane violated N-1 transition semantics"
        );
        Ok(())
    }

    fn validate_complete(&self, physical_batch_steps: usize) -> Result<()> {
        self.validate_request_complete()?;
        ensure!(
            self.logical_transitions + self.padding_transitions == physical_batch_steps,
            "fixed-cohort physical transition accounting drifted"
        );
        Ok(())
    }
}

struct Lane {
    request_index: usize,
    id: String,
    prompt_tokens: usize,
    sequence: Sequence,
    progress: LaneProgress,
    generated_text: String,
}

#[derive(Debug, Serialize)]
struct CohortTelemetry {
    schema_version: u32,
    backend: &'static str,
    cohort_index: usize,
    width: usize,
    prompt_tokens_per_lane: Vec<usize>,
    min_prompt_tokens: usize,
    max_prompt_tokens: usize,
    requested_tokens_per_lane: Vec<usize>,
    shared_capacity_tokens: usize,
    generated_tokens_per_lane: Vec<usize>,
    productive_transitions_per_lane: Vec<usize>,
    padding_transitions_per_lane: Vec<usize>,
    generated_tokens: usize,
    productive_transitions: usize,
    padding_transitions: usize,
    physical_batch_steps: usize,
    physical_transition_utilization: f64,
    common_prefix_tokens: usize,
    prefix_fanout_tokens: usize,
    prefix_fanout_reason: &'static str,
    prefix_fanout_boundary_policy: &'static str,
    prefix_fanout_min_tokens: usize,
    prefix_snapshot_bytes: u64,
    prefix_prefill_ms: f64,
    prefix_snapshot_ms: f64,
    prefix_restore_ms: f64,
    suffix_prefill_ms: f64,
    private_suffix_singleton_max_tokens: usize,
    private_suffix_singleton_lanes: usize,
    private_suffix_singleton_tokens: usize,
    private_suffix_packed_lanes: usize,
    private_suffix_packed_tokens: usize,
    prefill_ms: f64,
    decode_ms: f64,
    batch_transition_ms: f64,
    executor_gpu_ms: Option<f64>,
    aggregate_generated_tps: f64,
    sequence_allocation_delta_bytes: u64,
    remaining_sequence_required_bytes: u64,
    prefix_snapshot_required_bytes: u64,
    memory_admission_incremental_bytes: u64,
    memory_admission_reserve_bytes: u64,
    memory_admission_required_bytes: Option<u64>,
    memory_admission_reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_root_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_root_restores: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_root_restore_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    executor_scratch_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    executor_scratch_incremental_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moe_q8_batched_gdn_blocks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moe_packed_q4_gate_up_blocks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moe_per_lane_gdn_blocks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moe_per_lane_gate_up_blocks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moe_per_lane_iq3_gate_up_blocks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moe_per_lane_other_gate_up_blocks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moe_head_mode: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct FixedFileRootTelemetry {
    schema_version: u32,
    backend: &'static str,
    enabled: bool,
    requests: usize,
    planned_cohorts: usize,
    common_prefix_tokens: usize,
    selected_prefix_tokens: usize,
    planned_avoided_prefix_evaluations: usize,
    planned_avoided_prompt_tokens: usize,
    minimum_tokens: usize,
    reason: &'static str,
    outcome: &'static str,
    actual_cohort_uses: usize,
    actual_avoided_prefix_evaluations: usize,
    actual_avoided_prompt_tokens: usize,
    excluded_requests: usize,
    snapshot_required_bytes: u64,
    max_cohort_snapshot_required_bytes: u64,
    additional_memory_bytes: u64,
    memory_admission_required_bytes: Option<u64>,
    memory_admission_reason: &'static str,
    snapshot_bytes: u64,
    prefill_ms: f64,
    snapshot_ms: f64,
}

#[derive(Debug, Serialize)]
struct RefillTelemetry {
    schema_version: u32,
    backend: &'static str,
    arena_index: usize,
    width: usize,
    requests: usize,
    shared_capacity_tokens: usize,
    simulated_physical_steps: usize,
    simulated_productive_transitions: usize,
    requested_two_wave_steps: usize,
    physical_batch_steps: usize,
    productive_transitions: usize,
    padding_transitions: usize,
    physical_transition_utilization: f64,
    replacements: usize,
    prefill_tokens: usize,
    initial_prefill_model_ms: f64,
    initial_prepare_wall_ms: f64,
    replacement_prefill_model_ms: f64,
    replacement_prepare_wall_ms: f64,
    arena_wall_ms: f64,
    transition_and_refill_wall_ms: f64,
    batch_transition_ms: f64,
    executor_gpu_ms: Option<f64>,
    generated_tokens: usize,
    productive_transition_tps: f64,
    arena_generated_tps: f64,
    output_order: &'static str,
}

#[derive(Debug, Serialize)]
struct PlannerTelemetry {
    schema_version: u32,
    backend: &'static str,
    requests: usize,
    prefix_packing_configured: bool,
    prefix_packing_effective: bool,
    prefix_fanout_boundary_policy: &'static str,
    ragged_prompts_enabled: bool,
    ragged_prompt_policy: &'static str,
    ragged_prompt_plan_decision: &'static str,
    ragged_packing_strategy: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    refill_policy: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refill_default_max_prompt_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refill_configured: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refill_effective: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    planned_refill_arenas: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    planned_refill_requests: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    realized_refill_arenas: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    realized_refill_requests: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    realized_refill_memory_denied_arenas: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    realized_refill_memory_fallback_arenas: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    automatic_refill_transaction_outcome: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    automatic_refill_min_arenas: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    automatic_refill_max_prompt_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    automatic_refill_max_generation_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    automatic_refill_max_file_prefill_per_transition: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    automatic_refill_max_arena_prefill_per_transition: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    automatic_refill_min_transition_utilization: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    automatic_refill_min_idealized_step_savings: Option<&'static str>,
    compatibility_buckets: usize,
    planned_candidate_cohorts: usize,
    realized_prefix_affinity_cohorts: usize,
    planned_prefix_fallback_buckets: usize,
    planned_full_cohorts: usize,
    planned_batched_requests: usize,
    realized_full_cohorts: usize,
    planned_economics_rejected_cohorts: usize,
    realized_memory_rejected_cohorts: usize,
    realized_batched_requests: usize,
    realized_serial_fallback_requests: usize,
    realized_estimated_physical_transition_slots: usize,
    realized_estimated_capacity_slots: usize,
    minimum_requested_transition_utilization: &'static str,
    packing_order: &'static str,
    output_order: &'static str,
}

fn common_prefix_tokens(requests: &[&PreparedJsonlRequest]) -> usize {
    let Some(first) = requests.first() else {
        return 0;
    };
    requests
        .iter()
        .skip(1)
        .fold(first.prompt_ids.len(), |len, request| {
            first.prompt_ids[..len]
                .iter()
                .zip(&request.prompt_ids)
                .take_while(|(left, right)| left == right)
                .count()
        })
}

fn plan_prefix_fanout(requests: &[&PreparedJsonlRequest], enabled: bool) -> PrefixFanoutPlan {
    let common_prefix_tokens = common_prefix_tokens(requests);
    if !enabled {
        return PrefixFanoutPlan {
            common_prefix_tokens,
            selected_prefix_tokens: 0,
            reason: "disabled",
        };
    }
    if common_prefix_tokens < PREFIX_FANOUT_MIN_TOKENS {
        return PrefixFanoutPlan {
            common_prefix_tokens,
            selected_prefix_tokens: 0,
            reason: "below_minimum",
        };
    }
    PrefixFanoutPlan {
        common_prefix_tokens,
        selected_prefix_tokens: common_prefix_tokens,
        reason: "selected",
    }
}

fn align_prefix_fanout(
    mut plan: PrefixFanoutPlan,
    prompt_tokens: usize,
    prefill_chunk: usize,
    boundary_policy: QwenPrefixFanoutBoundaryPolicy,
) -> PrefixFanoutPlan {
    if plan.selected_prefix_tokens == 0 || plan.selected_prefix_tokens == prompt_tokens {
        return plan;
    }
    let aligned = plan.selected_prefix_tokens / prefill_chunk * prefill_chunk;
    if boundary_policy == QwenPrefixFanoutBoundaryPolicy::ExactLcp {
        plan.reason = "selected_exact_lcp";
        return plan;
    }
    if boundary_policy == QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp
        && aligned >= PREFIX_FANOUT_MIN_TOKENS
        && prompt_tokens.saturating_sub(plan.selected_prefix_tokens)
            <= PRIVATE_SUFFIX_SINGLETON_MAX_TOKENS
    {
        plan.reason = "selected_tiny_suffix_exact_lcp";
        return plan;
    }
    if aligned < PREFIX_FANOUT_MIN_TOKENS {
        plan.selected_prefix_tokens = 0;
        plan.reason = "alignment_below_minimum";
    } else if aligned != plan.selected_prefix_tokens {
        plan.selected_prefix_tokens = aligned;
        plan.reason = "selected_chunk_aligned";
    }
    plan
}

fn parse_prefix_fanout_enabled(value: Option<&OsStr>) -> bool {
    let Some(value) = value.and_then(OsStr::to_str) else {
        return true;
    };
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn parse_prefix_packing_enabled(value: Option<&str>, default_enabled: bool) -> Result<bool> {
    let Some(value) = value else {
        return Ok(default_enabled);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{PREFIX_PACKING_ENV} must be a boolean"),
    }
}

fn prefix_packing_enabled(default_enabled: bool) -> Result<bool> {
    let value = std::env::var_os(PREFIX_PACKING_ENV)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("{PREFIX_PACKING_ENV} must be valid UTF-8"))
        })
        .transpose()?;
    parse_prefix_packing_enabled(value.as_deref(), default_enabled)
}

fn fixed_file_root_default_enabled<const WIDTH: usize>() -> bool {
    matches!(WIDTH, DENSE_BATCH8_WIDTH | MOE_BATCH16_WIDTH)
}

fn fixed_cohort_fanout_enabled<const WIDTH: usize>() -> bool {
    let env_name = match WIDTH {
        DENSE_BATCH8_WIDTH => DENSE_PREFIX_FANOUT_ENV,
        MOE_BATCH16_WIDTH => MOE_PREFIX_FANOUT_ENV,
        _ => return false,
    };
    parse_prefix_fanout_enabled(std::env::var_os(env_name).as_deref())
}

fn selected_cohort_prefix_tokens<const WIDTH: usize>(
    indices: &[usize; WIDTH],
    requests: &[PreparedJsonlRequest],
    args: &Args,
    boundary_policy: QwenPrefixFanoutBoundaryPolicy,
) -> usize {
    let PrefillChunkArg::Fixed(chunk) = args.prefill_chunk else {
        return 0;
    };
    let request_refs = indices.map(|index| &requests[index]);
    let max_prompt_tokens = request_refs
        .iter()
        .map(|request| request.prompt_ids.len())
        .max()
        .unwrap_or(0);
    align_prefix_fanout(
        plan_prefix_fanout(&request_refs, fixed_cohort_fanout_enabled::<WIDTH>()),
        max_prompt_tokens,
        chunk.min(max_prompt_tokens.max(1)),
        boundary_policy,
    )
    .selected_prefix_tokens
}

fn ragged_prompt_policy<const WIDTH: usize>(
    automatic_execution: bool,
) -> Result<RaggedPromptPolicy> {
    let default_policy = if automatic_execution && WIDTH == DENSE_BATCH8_WIDTH {
        RaggedPromptPolicy::AutomaticDenseCharged
    } else {
        RaggedPromptPolicy::Disabled
    };
    let value = std::env::var_os(RAGGED_PROMPTS_ENV)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("{RAGGED_PROMPTS_ENV} must be valid UTF-8"))
        })
        .transpose()?;
    parse_ragged_prompt_policy(value.as_deref(), default_policy)
}

fn parse_ragged_prompt_policy(
    value: Option<&str>,
    default_policy: RaggedPromptPolicy,
) -> Result<RaggedPromptPolicy> {
    let Some(value) = value else {
        return Ok(default_policy);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(RaggedPromptPolicy::Explicit),
        "0" | "false" | "no" | "off" => Ok(RaggedPromptPolicy::Disabled),
        _ => bail!("{RAGGED_PROMPTS_ENV} must be a boolean"),
    }
}

fn dense_refill_policy(
    ragged_prompts: bool,
    automatic_execution: bool,
) -> Result<DenseRefillPolicy> {
    let value = std::env::var_os(DENSE_REFILL_ENV)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("{DENSE_REFILL_ENV} must be valid UTF-8"))
        })
        .transpose()?;
    parse_dense_refill_policy(value.as_deref(), ragged_prompts, automatic_execution)
}

fn parse_dense_refill_policy(
    value: Option<&str>,
    ragged_prompts: bool,
    automatic_execution: bool,
) -> Result<DenseRefillPolicy> {
    let Some(value) = value else {
        return Ok(if ragged_prompts {
            DenseRefillPolicy::Disabled
        } else {
            DenseRefillPolicy::ShortSerialFallbackRescue
        });
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(if automatic_execution {
            if ragged_prompts {
                DenseRefillPolicy::Disabled
            } else {
                DenseRefillPolicy::ShortSerialFallbackRescue
            }
        } else {
            DenseRefillPolicy::Forced
        }),
        "0" | "false" | "no" | "off" => Ok(DenseRefillPolicy::Disabled),
        _ => bail!("{DENSE_REFILL_ENV} must be a boolean"),
    }
}

fn fixed_cohort_refill_policy<const WIDTH: usize>(
    ragged_prompts: bool,
    automatic_execution: bool,
) -> Result<DenseRefillPolicy> {
    if WIDTH == DENSE_BATCH8_WIDTH {
        dense_refill_policy(ragged_prompts, automatic_execution)
    } else {
        Ok(DenseRefillPolicy::Disabled)
    }
}

fn cohort_compatibility(
    request: &PreparedJsonlRequest,
    args: &Args,
    ragged_prompts: bool,
) -> Result<CohortCompatibility> {
    jsonl_generation_capacity(request, args)?;
    Ok(if ragged_prompts {
        CohortCompatibility::RaggedPrompts
    } else {
        CohortCompatibility::EqualPromptTokens(request.prompt_ids.len())
    })
}

#[derive(Debug, Eq, PartialEq)]
struct CohortGenerationPlan {
    prompt_tokens_per_lane: Vec<usize>,
    min_prompt_tokens: usize,
    max_prompt_tokens: usize,
    requested_tokens: Vec<usize>,
    shared_capacity: usize,
}

fn cohort_generation_plan<const WIDTH: usize>(
    requests: &[&PreparedJsonlRequest; WIDTH],
    args: &Args,
) -> Result<CohortGenerationPlan> {
    let prompt_tokens_per_lane = requests
        .iter()
        .map(|request| request.prompt_ids.len())
        .collect::<Vec<_>>();
    let min_prompt_tokens = prompt_tokens_per_lane.iter().copied().min().unwrap_or(0);
    let max_prompt_tokens = prompt_tokens_per_lane.iter().copied().max().unwrap_or(0);
    let mut requested_tokens = Vec::with_capacity(WIDTH);
    let mut shared_capacity = 0usize;
    for request in requests {
        let (requested, capacity) = jsonl_generation_capacity(request, args)?;
        requested_tokens.push(requested);
        shared_capacity = shared_capacity.max(capacity);
    }
    Ok(CohortGenerationPlan {
        prompt_tokens_per_lane,
        min_prompt_tokens,
        max_prompt_tokens,
        requested_tokens,
        shared_capacity,
    })
}

fn requested_transition_utilization_qualifies<const WIDTH: usize>(
    indices: &[usize; WIDTH],
    requested_tokens: &[usize],
) -> Result<bool> {
    let transitions = indices.map(|index| requested_tokens[index].saturating_sub(1));
    let max_transitions = transitions.into_iter().max().unwrap_or(0);
    if max_transitions == 0 {
        return Ok(false);
    }
    let productive = transitions.into_iter().try_fold(0usize, |sum, value| {
        sum.checked_add(value)
            .context("fixed-cohort productive transition estimate overflow")
    })?;
    let physical = max_transitions
        .checked_mul(WIDTH)
        .context("fixed-cohort physical transition estimate overflow")?;
    Ok(productive
        .checked_mul(MIN_REQUESTED_TRANSITION_UTILIZATION_DENOMINATOR)
        .context("fixed-cohort utilization comparison overflow")?
        >= physical
            .checked_mul(MIN_REQUESTED_TRANSITION_UTILIZATION_NUMERATOR)
            .context("fixed-cohort utilization comparison overflow")?)
}

#[derive(Debug)]
struct BucketWorkPlan<const WIDTH: usize> {
    work: Vec<PlannedWork<WIDTH>>,
    candidate_cohorts: usize,
    prefix_affinity_cohorts: usize,
    full_cohorts: usize,
    economics_rejected_cohorts: usize,
    serial_fallback_requests: usize,
    physical_transition_slots: usize,
    capacity_slots: usize,
    prefix_affinity_batches: Vec<[usize; WIDTH]>,
}

fn estimated_transition_slots<const WIDTH: usize>(
    work: &[PlannedWork<WIDTH>],
    requested_tokens: &[usize],
) -> Result<usize> {
    work.iter().try_fold(0usize, |total, item| {
        let slots = match item {
            PlannedWork::Batch(indices) => indices
                .iter()
                .map(|&index| requested_tokens[index].saturating_sub(1))
                .max()
                .unwrap_or(0)
                .checked_mul(WIDTH)
                .context("fixed-cohort batch transition estimate overflow")?,
            PlannedWork::RefillArena(plan) => plan
                .simulated_physical_steps
                .checked_mul(WIDTH)
                .context("refill arena transition estimate overflow")?,
            PlannedWork::Serial(index) => requested_tokens[*index].saturating_sub(1),
        };
        total
            .checked_add(slots)
            .context("fixed-cohort transition estimate overflow")
    })
}

fn estimate_capacity_slots<const WIDTH: usize>(
    work: &[PlannedWork<WIDTH>],
    capacities: &[usize],
) -> Result<usize> {
    work.iter().try_fold(0usize, |total, item| {
        let slots = match item {
            PlannedWork::Batch(indices) => indices
                .iter()
                .map(|&index| capacities[index])
                .max()
                .unwrap_or(0)
                .checked_mul(WIDTH)
                .context("fixed-cohort batch capacity estimate overflow")?,
            PlannedWork::RefillArena(plan) => plan
                .shared_capacity
                .checked_mul(WIDTH)
                .context("refill arena capacity estimate overflow")?,
            PlannedWork::Serial(index) => capacities[*index],
        };
        total
            .checked_add(slots)
            .context("fixed-cohort capacity estimate overflow")
    })
}

fn plan_ordered_bucket<const WIDTH: usize>(
    indices: Vec<usize>,
    requested_tokens: &[usize],
    capacities: &[usize],
) -> Result<BucketWorkPlan<WIDTH>> {
    let mut work = Vec::new();
    let mut candidate_cohorts = 0usize;
    let mut full_cohorts = 0usize;
    let mut economics_rejected_cohorts = 0usize;
    let mut serial_fallback_requests = 0usize;
    let mut cohorts = indices.chunks_exact(WIDTH);
    for cohort in &mut cohorts {
        let cohort: [usize; WIDTH] = cohort.try_into().expect("exact fixed-cohort planner chunk");
        candidate_cohorts += 1;
        if requested_transition_utilization_qualifies(&cohort, requested_tokens)? {
            work.push(PlannedWork::Batch(cohort));
            full_cohorts += 1;
        } else {
            economics_rejected_cohorts += 1;
            for index in cohort {
                work.push(PlannedWork::Serial(index));
                serial_fallback_requests += 1;
            }
        }
    }
    for &index in cohorts.remainder() {
        work.push(PlannedWork::Serial(index));
        serial_fallback_requests += 1;
    }
    let physical_transition_slots = estimated_transition_slots(&work, requested_tokens)?;
    let capacity_slots = estimate_capacity_slots(&work, capacities)?;
    Ok(BucketWorkPlan {
        work,
        candidate_cohorts,
        prefix_affinity_cohorts: 0,
        full_cohorts,
        economics_rejected_cohorts,
        serial_fallback_requests,
        physical_transition_slots,
        capacity_slots,
        prefix_affinity_batches: Vec::new(),
    })
}

fn plan_depth_bucket<const WIDTH: usize>(
    mut indices: Vec<usize>,
    requested_tokens: &[usize],
    capacities: &[usize],
) -> Result<BucketWorkPlan<WIDTH>> {
    indices.sort_by_key(|&index| (requested_tokens[index], capacities[index], index));
    plan_ordered_bucket(indices, requested_tokens, capacities)
}

fn plan_capacity_bucket<const WIDTH: usize>(
    mut indices: Vec<usize>,
    requested_tokens: &[usize],
    capacities: &[usize],
) -> Result<BucketWorkPlan<WIDTH>> {
    indices.sort_by_key(|&index| (capacities[index], requested_tokens[index], index));
    plan_ordered_bucket(indices, requested_tokens, capacities)
}

fn simulate_refill_steps<const WIDTH: usize>(
    request_indices: &[usize],
    requested_tokens: &[usize],
) -> Result<(usize, usize)> {
    ensure!(WIDTH > 0, "refill width must be nonzero");
    let mut loads = [0usize; WIDTH];
    let mut productive = 0usize;
    for &index in request_indices {
        let transitions = requested_tokens[index].saturating_sub(1);
        productive = productive
            .checked_add(transitions)
            .context("refill productive transition overflow")?;
        let slot = loads
            .iter()
            .enumerate()
            .min_by_key(|&(slot, load)| (*load, slot))
            .map(|(slot, _)| slot)
            .expect("nonzero refill width");
        loads[slot] = loads[slot]
            .checked_add(transitions)
            .context("refill slot transition overflow")?;
    }
    Ok((loads.into_iter().max().unwrap_or(0), productive))
}

struct DenseRefillRewrite<const WIDTH: usize> {
    work: Vec<PlannedWork<WIDTH>>,
    arenas: usize,
    requests: usize,
    absorbed_serial_requests: usize,
}

fn plan_dense_refill<const WIDTH: usize>(
    work: Vec<PlannedWork<WIDTH>>,
    requests: &[PreparedJsonlRequest],
    args: &Args,
    _prefix_fanout_boundary_policy: QwenPrefixFanoutBoundaryPolicy,
    requested_tokens: &[usize],
    capacities: &[usize],
    _prefix_affinity_batches: &[[usize; WIDTH]],
    policy: DenseRefillPolicy,
) -> Result<DenseRefillRewrite<WIDTH>> {
    if !policy.enabled() || WIDTH != DENSE_BATCH8_WIDTH {
        return Ok(DenseRefillRewrite {
            work,
            arenas: 0,
            requests: 0,
            absorbed_serial_requests: 0,
        });
    }
    let mut output = Vec::with_capacity(work.len());
    let mut arenas = 0usize;
    let mut refill_requests = 0usize;
    let mut absorbed_serial_requests = 0usize;
    let mut cursor = 0usize;
    while cursor < work.len() {
        let mut end = cursor;
        let mut request_count = 0usize;
        while end < work.len() && request_count < WIDTH * 2 {
            request_count += match &work[end] {
                PlannedWork::Batch(_) => WIDTH,
                PlannedWork::Serial(_) => 1,
                PlannedWork::RefillArena(_) => unreachable!("nested refill plan"),
            };
            end += 1;
        }
        if request_count != WIDTH * 2 {
            output.push(work[cursor].clone());
            cursor += 1;
            continue;
        }
        let fallback_slice = &work[cursor..end];
        if fallback_slice.iter().any(|item| {
            matches!(item, PlannedWork::Batch(indices) if _prefix_affinity_batches.contains(indices))
        }) {
            output.push(work[cursor].clone());
            cursor += 1;
            continue;
        }
        let mut request_indices = Vec::with_capacity(WIDTH * 2);
        let mut fallback_work = Vec::with_capacity(fallback_slice.len());
        let mut candidate_serial_requests = 0usize;
        for item in fallback_slice {
            match item {
                PlannedWork::Batch(indices) => {
                    if selected_cohort_prefix_tokens(
                        indices,
                        requests,
                        args,
                        _prefix_fanout_boundary_policy,
                    ) > 0
                    {
                        request_indices.clear();
                        break;
                    }
                    request_indices.extend(indices);
                    fallback_work.push(RefillFallback::Batch(*indices));
                }
                PlannedWork::Serial(index) => {
                    request_indices.push(*index);
                    fallback_work.push(RefillFallback::Serial(*index));
                    candidate_serial_requests += 1;
                }
                PlannedWork::RefillArena(_) => unreachable!("nested refill plan"),
            }
        }
        let default_rescue_ineligible = policy == DenseRefillPolicy::ShortSerialFallbackRescue
            && (candidate_serial_requests == 0
                || request_indices.iter().any(|&index| {
                    requests[index].prompt_ids.len() > DEFAULT_REFILL_MAX_PROMPT_TOKENS
                }));
        if request_indices.len() != WIDTH * 2 || default_rescue_ineligible {
            output.push(work[cursor].clone());
            cursor += 1;
            continue;
        }
        let mut requested_order = request_indices.clone();
        requested_order.sort_by_key(|&index| (requested_tokens[index], capacities[index], index));
        let requested_two_wave_steps = requested_order[..WIDTH]
            .iter()
            .map(|&index| requested_tokens[index].saturating_sub(1))
            .max()
            .unwrap_or(0)
            .checked_add(
                requested_order[WIDTH..]
                    .iter()
                    .map(|&index| requested_tokens[index].saturating_sub(1))
                    .max()
                    .unwrap_or(0),
            )
            .context("refill requested two-wave transition overflow")?;
        request_indices.sort_by_key(|&index| {
            (
                std::cmp::Reverse(requested_tokens[index]),
                capacities[index],
                index,
            )
        });
        let (simulated_physical_steps, simulated_productive_transitions) =
            simulate_refill_steps::<WIDTH>(&request_indices, requested_tokens)?;
        let physical_slots = simulated_physical_steps
            .checked_mul(WIDTH)
            .context("refill physical slot overflow")?;
        let utilization_qualifies = simulated_productive_transitions
            .checked_mul(MIN_REQUESTED_TRANSITION_UTILIZATION_DENOMINATOR)
            .context("refill utilization overflow")?
            >= physical_slots
                .checked_mul(MIN_REQUESTED_TRANSITION_UTILIZATION_NUMERATOR)
                .context("refill utilization overflow")?;
        let savings_qualifies = simulated_physical_steps
            .checked_mul(REFILL_MIN_STEP_SAVINGS_DENOMINATOR)
            .context("refill savings overflow")?
            <= requested_two_wave_steps
                .checked_mul(
                    REFILL_MIN_STEP_SAVINGS_DENOMINATOR - REFILL_MIN_STEP_SAVINGS_NUMERATOR,
                )
                .context("refill savings overflow")?;
        if simulated_physical_steps == 0 || !utilization_qualifies || !savings_qualifies {
            output.push(work[cursor].clone());
            cursor += 1;
            continue;
        }
        let required_refill_capacity = request_indices
            .iter()
            .map(|&index| {
                requests[index]
                    .prompt_ids
                    .len()
                    .checked_add(simulated_productive_transitions)
                    .and_then(|value| value.checked_add(1))
                    .context("refill shared capacity overflow")
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        if args
            .max_context_tokens
            .is_some_and(|explicit| required_refill_capacity > explicit)
        {
            output.push(work[cursor].clone());
            cursor += 1;
            continue;
        }
        let shared_capacity = required_refill_capacity.max(
            request_indices
                .iter()
                .map(|&index| capacities[index])
                .max()
                .unwrap_or(0),
        );
        arenas += 1;
        refill_requests += request_indices.len();
        absorbed_serial_requests += candidate_serial_requests;
        output.push(PlannedWork::RefillArena(RefillArenaPlan {
            fallback_work,
            request_indices,
            shared_capacity,
            simulated_physical_steps,
            simulated_productive_transitions,
            requested_two_wave_steps,
        }));
        cursor = end;
    }
    Ok(DenseRefillRewrite {
        work: output,
        arenas,
        requests: refill_requests,
        absorbed_serial_requests,
    })
}

fn automatic_dense_ragged_envelope_qualifies<const WIDTH: usize>(
    requests: &[PreparedJsonlRequest],
    args: &Args,
) -> bool {
    let max_prompt_tokens = requests
        .iter()
        .map(|request| request.prompt_ids.len())
        .max()
        .unwrap_or(0);
    let PrefillChunkArg::Fixed(prefill_chunk) = args.prefill_chunk else {
        return false;
    };
    let mut requested_tokens = requests
        .iter()
        .map(|request| request.request.tokens.unwrap_or(args.tokens));
    let Some(first_requested_tokens) = requested_tokens.clone().next() else {
        return false;
    };
    WIDTH == DENSE_BATCH8_WIDTH
        && requests.len() >= WIDTH
        && max_prompt_tokens <= AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS
        && prefill_chunk >= max_prompt_tokens
        && first_requested_tokens >= AUTOMATIC_DENSE_RAGGED_MIN_GENERATION_TOKENS
        && requested_tokens.all(|tokens| tokens == first_requested_tokens)
}

fn automatic_dense_ragged_refill_enabled() -> Result<bool> {
    let value = std::env::var_os(DENSE_REFILL_ENV)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("{DENSE_REFILL_ENV} must be valid UTF-8"))
        })
        .transpose()?;
    let Some(value) = value else {
        return Ok(true);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{DENSE_REFILL_ENV} must be a boolean"),
    }
}

fn automatic_dense_ragged_refill_applicable<const WIDTH: usize>(
    requests: &[PreparedJsonlRequest],
    args: &Args,
) -> bool {
    let mut requested_tokens = requests
        .iter()
        .map(|request| request.request.tokens.unwrap_or(args.tokens));
    let Some(first) = requested_tokens.next() else {
        return false;
    };
    WIDTH == DENSE_BATCH8_WIDTH
        && requests.len() >= WIDTH * 2 * AUTOMATIC_DENSE_RAGGED_REFILL_MIN_ARENAS
        && requested_tokens.any(|tokens| tokens != first)
}

fn automatic_dense_ragged_refill_envelope_qualifies<const WIDTH: usize>(
    requests: &[PreparedJsonlRequest],
    args: &Args,
) -> bool {
    let max_prompt_tokens = requests
        .iter()
        .map(|request| request.prompt_ids.len())
        .max()
        .unwrap_or(0);
    let PrefillChunkArg::Fixed(prefill_chunk) = args.prefill_chunk else {
        return false;
    };
    WIDTH == DENSE_BATCH8_WIDTH
        && requests.len() >= WIDTH * 2 * AUTOMATIC_DENSE_RAGGED_REFILL_MIN_ARENAS
        && max_prompt_tokens <= AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS
        && prefill_chunk >= max_prompt_tokens
        && requests.iter().all(|request| {
            request.request.tokens.unwrap_or(args.tokens)
                <= AUTOMATIC_DENSE_RAGGED_REFILL_MAX_GENERATION_TOKENS
        })
}

fn automatic_dense_ragged_refill_arena_schedule_qualifies<const WIDTH: usize>(
    arena: &RefillArenaPlan<WIDTH>,
    requests: &[PreparedJsonlRequest],
) -> Result<bool> {
    let physical_slots = arena
        .simulated_physical_steps
        .checked_mul(WIDTH)
        .context("automatic ragged refill physical slot overflow")?;
    let utilization_qualifies = arena
        .simulated_productive_transitions
        .checked_mul(AUTOMATIC_DENSE_RAGGED_REFILL_MIN_UTILIZATION_DENOMINATOR)
        .context("automatic ragged refill utilization overflow")?
        >= physical_slots
            .checked_mul(AUTOMATIC_DENSE_RAGGED_REFILL_MIN_UTILIZATION_NUMERATOR)
            .context("automatic ragged refill utilization overflow")?;
    let savings_qualifies = arena
        .simulated_physical_steps
        .checked_mul(AUTOMATIC_DENSE_RAGGED_REFILL_MIN_SAVINGS_DENOMINATOR)
        .context("automatic ragged refill savings overflow")?
        <= arena
            .requested_two_wave_steps
            .checked_mul(
                AUTOMATIC_DENSE_RAGGED_REFILL_MIN_SAVINGS_DENOMINATOR
                    - AUTOMATIC_DENSE_RAGGED_REFILL_MIN_SAVINGS_NUMERATOR,
            )
            .context("automatic ragged refill savings overflow")?;
    let charged_prefill_tokens =
        arena
            .request_indices
            .iter()
            .try_fold(0usize, |total, &index| {
                total
                    .checked_add(requests[index].prompt_ids.len())
                    .context("automatic ragged refill arena charge overflow")
            })?;
    let arena_charge_qualifies = arena.simulated_productive_transitions > 0
        && charged_prefill_tokens
            <= arena
                .simulated_productive_transitions
                .checked_mul(AUTOMATIC_DENSE_RAGGED_REFILL_MAX_ARENA_PREFILL_PER_TRANSITION)
                .context("automatic ragged refill arena charge comparison overflow")?;
    Ok(utilization_qualifies && savings_qualifies && arena_charge_qualifies)
}

fn automatic_dense_ragged_refill_file_charge_qualifies(
    charged_prefill_tokens: usize,
    productive_transitions: usize,
) -> Result<bool> {
    Ok(productive_transitions > 0
        && charged_prefill_tokens
            <= productive_transitions
                .checked_mul(AUTOMATIC_DENSE_RAGGED_REFILL_MAX_PREFILL_PER_TRANSITION)
                .context("automatic ragged refill charge comparison overflow")?)
}

fn automatic_dense_ragged_refill_charge_qualifies<const WIDTH: usize>(
    plan: &CohortPlan<WIDTH>,
    requests: &[PreparedJsonlRequest],
) -> Result<bool> {
    if plan.refill_arenas < AUTOMATIC_DENSE_RAGGED_REFILL_MIN_ARENAS
        || plan.refill_requests != requests.len()
        || plan.serial_fallback_requests > 0
        || plan
            .work
            .iter()
            .any(|work| !matches!(work, PlannedWork::RefillArena(_)))
    {
        return Ok(false);
    }
    let mut productive_transitions = 0usize;
    for work in &plan.work {
        let PlannedWork::RefillArena(arena) = work else {
            unreachable!("automatic ragged refill charge checked arena-only work")
        };
        if !automatic_dense_ragged_refill_arena_schedule_qualifies(arena, requests)? {
            return Ok(false);
        }
        productive_transitions = productive_transitions
            .checked_add(arena.simulated_productive_transitions)
            .context("automatic ragged refill transition charge overflow")?;
    }
    let charged_prefill_tokens = requests.iter().try_fold(0usize, |total, request| {
        total
            .checked_add(request.prompt_ids.len())
            .context("automatic ragged refill prefill charge overflow")
    })?;
    // Selection and publication are file-wide, and prompt prefill is additive
    // across arenas. Keep schedule quality local, but charge the measured whole
    // file so a shallow arena can be amortized by a deeper arena in the same
    // admitted transaction.
    automatic_dense_ragged_refill_file_charge_qualifies(
        charged_prefill_tokens,
        productive_transitions,
    )
}

fn automatic_dense_ragged_charge_qualifies<const WIDTH: usize>(
    work: &[PlannedWork<WIDTH>],
    requests: &[PreparedJsonlRequest],
    args: &Args,
) -> Result<bool> {
    let mut batches = 0usize;
    for item in work {
        let PlannedWork::Batch(indices) = item else {
            continue;
        };
        batches += 1;
        let charged_prefill_tokens = indices.iter().try_fold(0usize, |total, &index| {
            total
                .checked_add(requests[index].prompt_ids.len())
                .context("automatic ragged prefill charge overflow")
        })?;
        let productive_transitions = indices.iter().try_fold(0usize, |total, &index| {
            let (requested_tokens, _) = jsonl_generation_capacity(&requests[index], args)?;
            total
                .checked_add(requested_tokens.saturating_sub(1))
                .context("automatic ragged transition charge overflow")
        })?;
        if productive_transitions == 0
            || charged_prefill_tokens
                > productive_transitions
                    .checked_mul(AUTOMATIC_DENSE_RAGGED_MAX_PREFILL_PER_TRANSITION)
                    .context("automatic ragged charge comparison overflow")?
        {
            return Ok(false);
        }
    }
    Ok(batches > 0)
}

fn plan_request_work<const WIDTH: usize>(
    requests: &[PreparedJsonlRequest],
    args: &Args,
) -> Result<CohortPlan<WIDTH>> {
    let automatic_execution =
        args.execution_mode == Some(execution_selector::ExecutionModeArg::Auto);
    let ragged_prompt_policy = ragged_prompt_policy::<WIDTH>(automatic_execution)?;
    plan_request_work_with_policy(requests, args, ragged_prompt_policy)
}

fn plan_request_work_with_policy<const WIDTH: usize>(
    requests: &[PreparedJsonlRequest],
    args: &Args,
    ragged_prompt_policy: RaggedPromptPolicy,
) -> Result<CohortPlan<WIDTH>> {
    let automatic_execution =
        args.execution_mode == Some(execution_selector::ExecutionModeArg::Auto);
    let prefix_packing = prefix_packing_enabled(true)?;
    let prefix_fanout_boundary_policy = qwen_prefix_fanout_boundary_policy()?;
    if ragged_prompt_policy == RaggedPromptPolicy::AutomaticDenseCharged {
        let mut baseline = plan_request_work_configured(
            requests,
            args,
            prefix_packing,
            prefix_fanout_boundary_policy,
            RaggedPromptPolicy::Disabled,
            fixed_cohort_refill_policy::<WIDTH>(false, automatic_execution)?,
        )?;
        if automatic_dense_ragged_envelope_qualifies::<WIDTH>(requests, args) {
            let candidate = plan_request_work_configured(
                requests,
                args,
                prefix_packing,
                prefix_fanout_boundary_policy,
                ragged_prompt_policy,
                DenseRefillPolicy::Disabled,
            )?;
            if candidate.full_cohorts > baseline.full_cohorts
                && candidate.serial_fallback_requests == 0
            {
                if automatic_dense_ragged_charge_qualifies(&candidate.work, requests, args)? {
                    return Ok(candidate);
                }
                baseline.ragged_prompt_plan_decision =
                    RaggedPromptDecision::AutomaticChargeRejected;
            } else {
                baseline.ragged_prompt_plan_decision = RaggedPromptDecision::AutomaticNoCohortGain;
            }
        } else {
            baseline.ragged_prompt_plan_decision = RaggedPromptDecision::AutomaticEnvelopeRejected;
        }
        if !automatic_dense_ragged_refill_applicable::<WIDTH>(requests, args) {
            return Ok(baseline);
        }
        if !automatic_dense_ragged_refill_enabled()? {
            baseline.ragged_prompt_plan_decision = RaggedPromptDecision::AutomaticRefillDisabled;
            return Ok(baseline);
        }
        if !automatic_dense_ragged_refill_envelope_qualifies::<WIDTH>(requests, args) {
            baseline.ragged_prompt_plan_decision =
                RaggedPromptDecision::AutomaticRefillEnvelopeRejected;
            return Ok(baseline);
        }
        let mut candidate = plan_request_work_configured(
            requests,
            args,
            prefix_packing,
            prefix_fanout_boundary_policy,
            ragged_prompt_policy,
            DenseRefillPolicy::AutomaticChargedRagged,
        )?;
        if baseline.full_cohorts > 0
            || baseline.serial_fallback_requests != requests.len()
            || candidate.full_cohorts <= baseline.full_cohorts
            || candidate.serial_fallback_requests > 0
        {
            baseline.ragged_prompt_plan_decision =
                RaggedPromptDecision::AutomaticRefillNoCohortGain;
            return Ok(baseline);
        }
        if !automatic_dense_ragged_refill_charge_qualifies(&candidate, requests)? {
            baseline.ragged_prompt_plan_decision =
                RaggedPromptDecision::AutomaticRefillChargeRejected;
            return Ok(baseline);
        }
        candidate.ragged_prompt_plan_decision = RaggedPromptDecision::AutomaticRefillAdmitted;
        candidate.automatic_refill_fallback = Some(AutomaticRefillFallback {
            work: baseline.work,
            prefix_affinity_batches: baseline.prefix_affinity_batches,
        });
        return Ok(candidate);
    }
    plan_request_work_configured(
        requests,
        args,
        prefix_packing,
        prefix_fanout_boundary_policy,
        ragged_prompt_policy,
        fixed_cohort_refill_policy::<WIDTH>(ragged_prompt_policy.enabled(), automatic_execution)?,
    )
}

fn plan_request_work_configured<const WIDTH: usize>(
    requests: &[PreparedJsonlRequest],
    args: &Args,
    prefix_packing: bool,
    prefix_fanout_boundary_policy: QwenPrefixFanoutBoundaryPolicy,
    ragged_prompt_policy: impl Into<RaggedPromptPolicy>,
    refill_policy: impl Into<DenseRefillPolicy>,
) -> Result<CohortPlan<WIDTH>> {
    ensure!(WIDTH > 0, "fixed-cohort planner width must be nonzero");
    let ragged_prompt_policy = ragged_prompt_policy.into();
    let ragged_prompt_plan_decision = RaggedPromptDecision::for_policy(ragged_prompt_policy);
    let ragged_prompts = ragged_prompt_policy.enabled();
    let refill_policy = refill_policy.into();
    let generation = requests
        .iter()
        .map(|request| jsonl_generation_capacity(request, args))
        .collect::<Result<Vec<_>>>()?;
    let requested_tokens = generation
        .iter()
        .map(|&(tokens, _)| tokens)
        .collect::<Vec<_>>();
    let capacities = generation
        .iter()
        .map(|&(_, capacity)| capacity)
        .collect::<Vec<_>>();
    let mut buckets: Vec<(CohortCompatibility, Vec<usize>)> = Vec::new();
    for (index, request) in requests.iter().enumerate() {
        let key = cohort_compatibility(request, args, ragged_prompts)?;
        if let Some((_, indices)) = buckets.iter_mut().find(|(candidate, _)| *candidate == key) {
            indices.push(index);
        } else {
            buckets.push((key, vec![index]));
        }
    }

    let compatibility_buckets = buckets.len();
    let mut work = Vec::new();
    let mut candidate_cohorts = 0usize;
    let mut prefix_plan_fallback_buckets = 0usize;
    let mut full_cohorts = 0usize;
    let mut economics_rejected_cohorts = 0usize;
    let mut serial_fallback_requests = 0usize;
    let mut estimated_physical_transition_slots = 0usize;
    let mut estimated_capacity_slots = 0usize;
    let mut prefix_affinity_batches = Vec::new();
    let mut ragged_packing_strategy = RaggedPackingStrategy::Disabled;
    let mut refill_arenas = 0usize;
    let mut refill_requests = 0usize;
    for (_, indices) in buckets {
        let baseline = plan_depth_bucket::<WIDTH>(indices.clone(), &requested_tokens, &capacities)?;
        let mut bucket_ragged_strategy = RaggedPackingStrategy::Disabled;
        let selected = if ragged_prompts {
            let capacity = plan_capacity_bucket::<WIDTH>(indices, &requested_tokens, &capacities)?;
            if capacity.full_cohorts >= baseline.full_cohorts
                && capacity.physical_transition_slots <= baseline.physical_transition_slots
                && capacity.capacity_slots < baseline.capacity_slots
            {
                bucket_ragged_strategy = RaggedPackingStrategy::Capacity;
                capacity
            } else {
                bucket_ragged_strategy = RaggedPackingStrategy::GenerationDepth;
                baseline
            }
        } else if prefix_packing {
            let mut indices = indices;
            indices.sort_by(|&left, &right| {
                requests[left]
                    .prompt_ids
                    .cmp(&requests[right].prompt_ids)
                    .then_with(|| requested_tokens[left].cmp(&requested_tokens[right]))
                    .then_with(|| left.cmp(&right))
            });
            let mut prefix_work = Vec::new();
            let mut prefix_affinity_cohorts = 0usize;
            let mut remaining = Vec::new();
            let mut prefix_cohorts = indices.chunks_exact(WIDTH);
            for cohort in &mut prefix_cohorts {
                let cohort: [usize; WIDTH] = cohort
                    .try_into()
                    .expect("exact fixed-cohort prefix planner chunk");
                if selected_cohort_prefix_tokens(
                    &cohort,
                    requests,
                    args,
                    prefix_fanout_boundary_policy,
                ) > 0
                    && requested_transition_utilization_qualifies(&cohort, &requested_tokens)?
                {
                    prefix_work.push(PlannedWork::Batch(cohort));
                    prefix_affinity_cohorts += 1;
                } else {
                    remaining.extend(cohort);
                }
            }
            remaining.extend(prefix_cohorts.remainder());
            let depth = plan_depth_bucket::<WIDTH>(remaining, &requested_tokens, &capacities)?;
            let prefix_affinity_batches = prefix_work
                .iter()
                .filter_map(|work| match work {
                    PlannedWork::Batch(indices) => Some(*indices),
                    PlannedWork::RefillArena(_) | PlannedWork::Serial(_) => None,
                })
                .collect();
            let mut candidate_work = prefix_work;
            candidate_work.extend(depth.work);
            let candidate = BucketWorkPlan {
                physical_transition_slots: estimated_transition_slots(
                    &candidate_work,
                    &requested_tokens,
                )?,
                capacity_slots: estimate_capacity_slots(&candidate_work, &capacities)?,
                work: candidate_work,
                candidate_cohorts: prefix_affinity_cohorts + depth.candidate_cohorts,
                prefix_affinity_cohorts,
                full_cohorts: prefix_affinity_cohorts + depth.full_cohorts,
                economics_rejected_cohorts: depth.economics_rejected_cohorts,
                serial_fallback_requests: depth.serial_fallback_requests,
                prefix_affinity_batches,
            };
            let safe = candidate.prefix_affinity_cohorts > 0
                && (candidate.full_cohorts > baseline.full_cohorts
                    || (candidate.full_cohorts == baseline.full_cohorts
                        && candidate.physical_transition_slots
                            <= baseline.physical_transition_slots));
            if safe {
                candidate
            } else {
                if candidate.prefix_affinity_cohorts > 0 {
                    prefix_plan_fallback_buckets += 1;
                }
                baseline
            }
        } else {
            baseline
        };
        let refill = plan_dense_refill(
            selected.work,
            requests,
            args,
            prefix_fanout_boundary_policy,
            &requested_tokens,
            &capacities,
            &selected.prefix_affinity_batches,
            refill_policy,
        )?;
        let rescued_cohort_equivalents = refill
            .absorbed_serial_requests
            .checked_div(WIDTH)
            .context("refill width must be nonzero")?;
        candidate_cohorts += selected.candidate_cohorts;
        full_cohorts += selected.full_cohorts + rescued_cohort_equivalents;
        economics_rejected_cohorts += selected.economics_rejected_cohorts;
        serial_fallback_requests += selected
            .serial_fallback_requests
            .checked_sub(refill.absorbed_serial_requests)
            .context("refill absorbed more serial requests than planned")?;
        estimated_physical_transition_slots = estimated_physical_transition_slots
            .checked_add(estimated_transition_slots(&refill.work, &requested_tokens)?)
            .context("fixed-cohort plan transition estimate overflow")?;
        estimated_capacity_slots = estimated_capacity_slots
            .checked_add(estimate_capacity_slots(&refill.work, &capacities)?)
            .context("fixed-cohort plan capacity estimate overflow")?;
        refill_arenas += refill.arenas;
        refill_requests += refill.requests;
        prefix_affinity_batches.extend(selected.prefix_affinity_batches);
        work.extend(refill.work);
        ragged_packing_strategy = ragged_packing_strategy.merge(bucket_ragged_strategy);
    }
    work.sort_by_key(PlannedWork::first_request_index);
    let max_execution_capacity = work
        .iter()
        .filter_map(|item| match item {
            PlannedWork::RefillArena(plan) => Some(plan.shared_capacity),
            PlannedWork::Batch(_) | PlannedWork::Serial(_) => None,
        })
        .chain(capacities.iter().copied())
        .max()
        .unwrap_or(0);
    ensure!(
        full_cohorts
            .checked_mul(WIDTH)
            .and_then(|batched| batched.checked_add(serial_fallback_requests))
            == Some(requests.len()),
        "fixed-cohort planner request accounting drifted"
    );
    Ok(CohortPlan {
        work,
        prefix_packing_configured: prefix_packing,
        prefix_packing_effective: prefix_packing && !ragged_prompts,
        prefix_fanout_boundary_policy,
        ragged_prompt_policy,
        ragged_prompt_plan_decision,
        ragged_packing_strategy,
        compatibility_buckets,
        candidate_cohorts,
        prefix_plan_fallback_buckets,
        full_cohorts,
        economics_rejected_cohorts,
        serial_fallback_requests,
        prefix_affinity_batches,
        refill_policy,
        refill_configured: refill_policy.enabled(),
        refill_effective: refill_arenas > 0,
        refill_arenas,
        refill_requests,
        max_execution_capacity,
        automatic_refill_fallback: None,
    })
}

pub(super) fn plan_summary<const WIDTH: usize>(
    requests: &[PreparedJsonlRequest],
    args: &Args,
) -> Result<CohortPlanSummary> {
    let plan = plan_request_work::<WIDTH>(requests, args)?;
    let dense_refill = WIDTH == DENSE_BATCH8_WIDTH;
    Ok(CohortPlanSummary {
        ragged_prompt_policy: Some(plan.ragged_prompt_policy.as_str()),
        ragged_prompt_plan_decision: Some(plan.ragged_prompt_plan_decision.as_str()),
        refill_policy: dense_refill.then_some(plan.refill_policy.as_str()),
        planned_refill_arenas: dense_refill.then_some(plan.refill_arenas),
        planned_refill_requests: dense_refill.then_some(plan.refill_requests),
        full_cohorts: plan.full_cohorts,
        economics_rejected_cohorts: plan.economics_rejected_cohorts,
        serial_fallback_requests: plan.serial_fallback_requests,
        max_execution_capacity: plan.max_execution_capacity,
    })
}

pub(super) fn validate_cli(args: &Args, explicit: ExplicitCliOptions) -> Result<()> {
    let Some(batch_size) = args.batch_size else {
        return Ok(());
    };
    ensure!(
        matches!(batch_size, DENSE_BATCH8_WIDTH | MOE_BATCH16_WIDTH),
        "--batch-size supports only {DENSE_BATCH8_WIDTH} (dense Qwen) or {MOE_BATCH16_WIDTH} (Qwen MoE), got {batch_size}"
    );
    ensure!(!args.info, "--batch-size cannot be used with --info");
    let requests_path = args
        .requests_jsonl
        .as_deref()
        .context("--batch-size requires --requests-jsonl")?;
    ensure!(
        requests_path != Path::new("-"),
        "--batch-size requires a JSONL file; streaming stdin cohorts are not yet supported"
    );
    ensure!(
        matches!(args.prefill_chunk, PrefillChunkArg::Fixed(_)),
        "--batch-size currently requires a fixed --prefill-chunk"
    );
    ensure!(
        args.temperature == 0.0,
        "--batch-size currently requires greedy decoding (--temp 0)"
    );
    ensure!(
        !args.prompt_lookup,
        "--batch-size does not yet compose with --prompt-lookup"
    );
    ensure!(
        args.request_stats.is_none() && args.request_stats_jsonl.is_none(),
        "--batch-size does not yet emit per-request stats sidecars"
    );
    ensure!(
        args.trace_request.is_none(),
        "--batch-size does not yet emit request-trace rows"
    );
    ensure!(
        args.cache_prefix_tokens.is_none()
            && !explicit.prefix_cache_max_mib
            && !explicit.cache_prefix_auto_min_tokens,
        "--batch-size does not yet compose with prefix-cache configuration"
    );
    ensure!(
        args.durable_prefix_cache.is_none(),
        "--batch-size does not yet compose with durable prefix caching"
    );
    validate_greedy_gpu_mode(configured_greedy_gpu_argmax_mode(), args.batch_size)?;
    Ok(())
}

fn validate_greedy_gpu_mode(mode: GreedyGpuArgmaxMode, batch_size: Option<usize>) -> Result<()> {
    if mode == GreedyGpuArgmaxMode::ExplicitRollback {
        if batch_size == Some(DENSE_BATCH8_WIDTH) {
            bail!(
                "{GREEDY_GPU_ARGMAX_ENV}=0 disables --batch-size because fixed B=8 requires GPU greedy selection"
            );
        }
        bail!(
            "{GREEDY_GPU_ARGMAX_ENV}=0 disables --batch-size because MoE B=16 requires GPU greedy selection"
        );
    }
    Ok(())
}

pub(super) fn validate_model_family(
    batch_size: Option<usize>,
    model_family: Option<ModelFamily>,
) -> Result<()> {
    ensure!(
        match batch_size {
            None => true,
            Some(DENSE_BATCH8_WIDTH) => model_family == Some(ModelFamily::Qwen35),
            Some(MOE_BATCH16_WIDTH) => model_family == Some(ModelFamily::Qwen35Moe),
            Some(_) => false,
        },
        "unsupported --batch-size/model combination: batch size 8 requires qwen35 (dense), and batch size 16 requires qwen35moe"
    );
    Ok(())
}

pub(super) fn run_file(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    requests_path: &Path,
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
    stdout: &mut impl Write,
) -> Result<usize> {
    let batch_size = args
        .batch_size
        .context("fixed-cohort JSONL execution requires --batch-size")?;
    validate_greedy_gpu_mode(greedy_gpu_mode, Some(batch_size))?;
    let requests_metadata = std::fs::metadata(requests_path)
        .with_context(|| format!("inspect requests JSONL {}", requests_path.display()))?;
    ensure!(
        requests_metadata.is_file(),
        "--batch-size requires a regular JSONL file, got {}",
        requests_path.display()
    );
    let requests = prepare_jsonl_requests(requests_path, tokenizer, args)?;
    run_prepared(
        loaded,
        tokenizer,
        &requests,
        args,
        greedy_gpu_mode,
        stdout,
        batch_size,
    )
}

pub(super) fn run_prepared(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    requests: &[PreparedJsonlRequest],
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
    stdout: &mut impl Write,
    batch_size: usize,
) -> Result<usize> {
    validate_greedy_gpu_mode(greedy_gpu_mode, Some(batch_size))?;
    validate_requests(requests, args)?;
    match batch_size {
        DENSE_BATCH8_WIDTH => run_fixed_cohort_file(
            loaded,
            tokenizer,
            requests,
            args,
            greedy_gpu_mode,
            stdout,
            plan_request_work::<DENSE_BATCH8_WIDTH>(requests, args)?,
            || {
                loaded
                    .create_dense_batch8_executor()
                    .context("create dense B=8 executor")
            },
        ),
        MOE_BATCH16_WIDTH => run_fixed_cohort_file(
            loaded,
            tokenizer,
            requests,
            args,
            greedy_gpu_mode,
            stdout,
            plan_request_work::<MOE_BATCH16_WIDTH>(requests, args)?,
            || {
                loaded
                    .create_moe_batch16_executor()
                    .context("create MoE B=16 executor")
            },
        ),
        _ => bail!("unsupported fixed-cohort batch size {batch_size}"),
    }
}

fn fixed_cohort_memory_admission<const WIDTH: usize, E: FixedCohortExecutor<WIDTH>>(
    loaded: &LoadedModel,
    requests: &[&PreparedJsonlRequest; WIDTH],
    args: &Args,
    prefix_fanout_boundary_policy: QwenPrefixFanoutBoundaryPolicy,
) -> Result<qwen_llm::metal::MetalMemoryAdmission> {
    let generation = cohort_generation_plan(requests, args)?;
    let prefill_scratch_upper_bytes = requests
        .iter()
        .map(|request| concurrent_jsonl::prefill_scratch_upper_bytes(loaded, request, args))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("fixed cohort contained no prefill scratch estimate")?;
    let chunk = match args.prefill_chunk {
        PrefillChunkArg::Fixed(requested) => requested.min(generation.max_prompt_tokens.max(1)),
        PrefillChunkArg::Auto => bail!("{} requires a fixed prefill chunk", E::DISPLAY_NAME),
    };
    let prefix_fanout = align_prefix_fanout(
        plan_prefix_fanout(
            requests,
            parse_prefix_fanout_enabled(std::env::var_os(E::PREFIX_FANOUT_ENV).as_deref()),
        ),
        generation.max_prompt_tokens,
        chunk,
        prefix_fanout_boundary_policy,
    );
    let prefix_snapshot_required_bytes = if prefix_fanout.selected_prefix_tokens > 0 {
        loaded
            .estimate_checkpoint_boundary_sizes_unallocated(
                generation.shared_capacity,
                prefix_fanout.selected_prefix_tokens,
                false,
                false,
            )?
            .snapshot_bytes
    } else {
        0
    };
    let executor_scratch_upper_bytes = E::executor_scratch_upper_bytes(loaded)?;
    let admission = loaded
        .qwen_execution_memory_admission_with_additional_bytes(
            WIDTH,
            generation.shared_capacity,
            prefill_scratch_upper_bytes,
            executor_scratch_upper_bytes,
            prefix_snapshot_required_bytes,
        )
        .map_err(anyhow::Error::from)?;
    if admission.admitted || prefix_snapshot_required_bytes == 0 {
        return Ok(admission);
    }
    loaded
        .qwen_execution_memory_admission(
            WIDTH,
            generation.shared_capacity,
            prefill_scratch_upper_bytes,
            executor_scratch_upper_bytes,
        )
        .map_err(anyhow::Error::from)
}

fn refill_arena_memory_admission<const WIDTH: usize, E: FixedCohortExecutor<WIDTH>>(
    loaded: &LoadedModel,
    requests: &[PreparedJsonlRequest],
    args: &Args,
    plan: &RefillArenaPlan<WIDTH>,
) -> Result<qwen_llm::metal::MetalMemoryAdmission> {
    let prefill_scratch_upper_bytes = plan
        .request_indices
        .iter()
        .map(|&index| concurrent_jsonl::prefill_scratch_upper_bytes(loaded, &requests[index], args))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("refill arena contained no prefill scratch estimate")?;
    loaded
        .qwen_execution_memory_admission(
            WIDTH,
            plan.shared_capacity,
            prefill_scratch_upper_bytes,
            E::executor_scratch_upper_bytes(loaded)?,
        )
        .map_err(anyhow::Error::from)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AutomaticRefillAdmission {
    NotApplicable,
    Admitted,
    PlannerBaselineFallback { rejected_arenas: usize },
}

fn admit_automatic_refill_plan<const WIDTH: usize>(
    mut plan: CohortPlan<WIDTH>,
    mut admitted: impl FnMut(&RefillArenaPlan<WIDTH>) -> Result<bool>,
) -> Result<(CohortPlan<WIDTH>, AutomaticRefillAdmission)> {
    if plan.automatic_refill_fallback.is_none() {
        return Ok((plan, AutomaticRefillAdmission::NotApplicable));
    }
    for work in &plan.work {
        if let PlannedWork::RefillArena(arena) = work
            && !admitted(arena)?
        {
            let rejected_arenas = plan.refill_arenas;
            let fallback = plan
                .automatic_refill_fallback
                .take()
                .expect("automatic refill plan carried a planner baseline fallback");
            plan.work = fallback.work;
            plan.prefix_affinity_batches = fallback.prefix_affinity_batches;
            return Ok((
                plan,
                AutomaticRefillAdmission::PlannerBaselineFallback { rejected_arenas },
            ));
        }
    }
    plan.automatic_refill_fallback = None;
    Ok((plan, AutomaticRefillAdmission::Admitted))
}

fn rewrite_unadmitted_refill_arenas<const WIDTH: usize>(
    work: Vec<PlannedWork<WIDTH>>,
    mut admitted: impl FnMut(&RefillArenaPlan<WIDTH>) -> Result<bool>,
) -> Result<(Vec<PlannedWork<WIDTH>>, usize)> {
    let mut rewritten = Vec::with_capacity(work.len());
    let mut rejected = 0usize;
    for item in work {
        match item {
            PlannedWork::RefillArena(plan) if !admitted(&plan)? => {
                rejected += 1;
                rewritten.extend(plan.fallback_work.into_iter().map(|item| match item {
                    RefillFallback::Batch(indices) => PlannedWork::Batch(indices),
                    RefillFallback::Serial(index) => PlannedWork::Serial(index),
                }));
            }
            item => rewritten.push(item),
        }
    }
    rewritten.sort_by_key(PlannedWork::first_request_index);
    Ok((rewritten, rejected))
}

fn fixed_file_root_request_indices<const WIDTH: usize>(work: &[PlannedWork<WIDTH>]) -> Vec<usize> {
    work.iter()
        .filter_map(|work| match work {
            PlannedWork::Batch(indices) => Some(indices.as_slice()),
            PlannedWork::RefillArena(_) | PlannedWork::Serial(_) => None,
        })
        .flatten()
        .copied()
        .collect()
}

fn max_fixed_file_root_snapshot_bytes<const WIDTH: usize>(
    loaded: &LoadedModel,
    requests: &[PreparedJsonlRequest],
    work: &[PlannedWork<WIDTH>],
    args: &Args,
    boundary_policy: QwenPrefixFanoutBoundaryPolicy,
    root_tokens: usize,
    max_capacity: usize,
) -> Result<u64> {
    let mut maximum = 0u64;
    for item in work {
        let PlannedWork::Batch(indices) = item else {
            continue;
        };
        let prefix_tokens = selected_cohort_prefix_tokens(indices, requests, args, boundary_policy)
            .max(root_tokens);
        if prefix_tokens <= root_tokens {
            continue;
        }
        let bytes = loaded
            .estimate_checkpoint_boundary_sizes_unallocated(
                max_capacity,
                prefix_tokens,
                false,
                false,
            )
            .context("estimate cohort checkpoint above file root")?
            .snapshot_bytes;
        maximum = maximum.max(bytes);
    }
    Ok(maximum)
}

fn rewrite_unadmitted_batches<const WIDTH: usize>(
    work: Vec<PlannedWork<WIDTH>>,
    mut admitted: impl FnMut(&[usize; WIDTH]) -> Result<bool>,
) -> Result<(Vec<PlannedWork<WIDTH>>, Vec<[usize; WIDTH]>)> {
    let mut rewritten = Vec::with_capacity(work.len());
    let mut rejected = Vec::new();
    for item in work {
        match item {
            PlannedWork::Batch(indices) if !admitted(&indices)? => {
                rejected.push(indices);
                rewritten.extend(indices.map(PlannedWork::Serial));
            }
            item => rewritten.push(item),
        }
    }
    rewritten.sort_by_key(PlannedWork::first_request_index);
    Ok((rewritten, rejected))
}

#[allow(clippy::too_many_arguments)]
fn run_fixed_cohort_file<const WIDTH: usize, E: FixedCohortExecutor<WIDTH>>(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    requests: &[PreparedJsonlRequest],
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
    stdout: &mut impl Write,
    plan: CohortPlan<WIDTH>,
    executor_factory: impl FnOnce() -> Result<E>,
) -> Result<usize> {
    let stop_tokens = loaded
        .gguf()
        .stop_token_ids()
        .context("load producer-declared stop tokens")?;
    let mut denied_admissions = Vec::new();
    let mut denied_refill_admissions = Vec::new();
    let (plan, automatic_refill_admission) = admit_automatic_refill_plan(plan, |arena| {
        let admission = refill_arena_memory_admission::<WIDTH, E>(loaded, requests, args, arena)?;
        if !admission.admitted {
            denied_refill_admissions.push((
                arena.request_indices.first().copied().unwrap_or(0),
                admission,
            ));
        }
        Ok(admission.admitted)
    })?;
    let (planned_work, rejected_refill_arenas) = match automatic_refill_admission {
        AutomaticRefillAdmission::Admitted => (plan.work, 0),
        AutomaticRefillAdmission::PlannerBaselineFallback { rejected_arenas } => {
            (plan.work, rejected_arenas)
        }
        AutomaticRefillAdmission::NotApplicable => {
            rewrite_unadmitted_refill_arenas(plan.work, |arena| {
                let admission =
                    refill_arena_memory_admission::<WIDTH, E>(loaded, requests, args, arena)?;
                if !admission.admitted {
                    denied_refill_admissions.push((
                        arena.request_indices.first().copied().unwrap_or(0),
                        admission,
                    ));
                }
                Ok(admission.admitted)
            })?
        }
    };
    let (planned_work, rejected_batches) = rewrite_unadmitted_batches(planned_work, |indices| {
        let cohort = indices.map(|index| &requests[index]);
        let admission = fixed_cohort_memory_admission::<WIDTH, E>(
            loaded,
            &cohort,
            args,
            plan.prefix_fanout_boundary_policy,
        )?;
        if !admission.admitted {
            denied_admissions.push((*indices, admission));
        }
        Ok(admission.admitted)
    })?;
    let memory_rejected_cohorts = rejected_batches.len();
    let denied_refill_arena_count = denied_refill_admissions.len();
    for (indices, admission) in denied_admissions {
        eprintln!(
            "{}_admission: cohort_first_request={} outcome=serial_fallback reason={} required_bytes={:?} working_set_headroom_bytes={:?} process_limit_remaining_bytes={:?}",
            E::TELEMETRY_PREFIX,
            indices.iter().min().copied().unwrap_or(0),
            admission.reason.as_str(),
            admission.required_bytes,
            admission.working_set_headroom_bytes,
            admission.signals.process_limit_remaining_bytes,
        );
    }
    for (first_request, admission) in denied_refill_admissions {
        eprintln!(
            "dense_batch8_refill_admission: arena_first_request={first_request} outcome={} reason={} required_bytes={:?} working_set_headroom_bytes={:?} process_limit_remaining_bytes={:?}",
            if matches!(
                automatic_refill_admission,
                AutomaticRefillAdmission::PlannerBaselineFallback { .. }
            ) {
                "automatic_planner_baseline_fallback"
            } else {
                "cohort_fallback"
            },
            admission.reason.as_str(),
            admission.required_bytes,
            admission.working_set_headroom_bytes,
            admission.signals.process_limit_remaining_bytes,
        );
    }
    let realized_batched_requests = planned_work.iter().try_fold(0usize, |total, work| {
        let requests = match work {
            PlannedWork::Batch(_) => WIDTH,
            PlannedWork::RefillArena(plan) => plan.request_indices.len(),
            PlannedWork::Serial(_) => 0,
        };
        total
            .checked_add(requests)
            .context("fixed-cohort realized batch count overflow")
    })?;
    let realized_full_cohorts = realized_batched_requests / WIDTH;
    let realized_serial_fallback_requests = planned_work
        .iter()
        .filter(|work| matches!(work, PlannedWork::Serial(_)))
        .count();
    let realized_transition_slots = estimated_transition_slots(
        &planned_work,
        &requests
            .iter()
            .map(|request| jsonl_generation_capacity(request, args).map(|(tokens, _)| tokens))
            .collect::<Result<Vec<_>>>()?,
    )?;
    let capacities = requests
        .iter()
        .map(|request| jsonl_generation_capacity(request, args).map(|(_, capacity)| capacity))
        .collect::<Result<Vec<_>>>()?;
    let realized_capacity_slots = estimate_capacity_slots(&planned_work, &capacities)?;
    let fixed_cohort_count = planned_work
        .iter()
        .filter(|work| matches!(work, PlannedWork::Batch(_)))
        .count();
    let root_request_indices = fixed_file_root_request_indices(&planned_work);
    let root_prompt_refs = root_request_indices
        .iter()
        .map(|&index| requests[index].prompt_ids.as_slice())
        .collect::<Vec<_>>();
    let root_use_count = fixed_cohort_count;
    let root_plan = qwen_file_root::plan(
        &root_prompt_refs,
        args.prefill_chunk,
        root_use_count,
        qwen_file_root::enabled(
            FILE_ROOT_FANOUT_ENV,
            fixed_file_root_default_enabled::<WIDTH>(),
        )?,
    );
    let excluded_requests = requests.len().saturating_sub(root_prompt_refs.len());
    let mut file_root = None;
    let mut root_telemetry = FixedFileRootTelemetry {
        schema_version: 1,
        backend: "qwen_fixed_file_root_v1",
        enabled: root_plan.enabled,
        requests: requests.len(),
        planned_cohorts: root_use_count,
        common_prefix_tokens: root_plan.common_prefix_tokens,
        selected_prefix_tokens: root_plan.selected_prefix_tokens,
        planned_avoided_prefix_evaluations: root_plan.planned_avoided_prefix_evaluations,
        planned_avoided_prompt_tokens: root_plan.planned_avoided_prompt_tokens,
        minimum_tokens: qwen_file_root::MIN_TOKENS,
        reason: root_plan.reason,
        outcome: "not_selected",
        actual_cohort_uses: 0,
        actual_avoided_prefix_evaluations: 0,
        actual_avoided_prompt_tokens: 0,
        excluded_requests,
        snapshot_required_bytes: 0,
        max_cohort_snapshot_required_bytes: 0,
        additional_memory_bytes: 0,
        memory_admission_required_bytes: None,
        memory_admission_reason: "not_requested",
        snapshot_bytes: 0,
        prefill_ms: 0.0,
        snapshot_ms: 0.0,
    };
    if root_plan.selected_prefix_tokens > 0 {
        let max_capacity = planned_work
            .iter()
            .filter_map(|work| match work {
                PlannedWork::Batch(indices) => indices.iter().map(|&index| capacities[index]).max(),
                PlannedWork::RefillArena(_) | PlannedWork::Serial(_) => None,
            })
            .max()
            .context("file-root plan contained no batched capacity")?;
        let prefill_scratch_upper_bytes = root_request_indices
            .iter()
            .map(|&index| {
                concurrent_jsonl::prefill_scratch_upper_bytes(loaded, &requests[index], args)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let root_snapshot_required_bytes = loaded
            .estimate_checkpoint_boundary_sizes_unallocated(
                max_capacity,
                root_plan.selected_prefix_tokens,
                false,
                true,
            )
            .context("estimate fixed-cohort file root")?
            .snapshot_bytes;
        let max_cohort_snapshot_required_bytes = max_fixed_file_root_snapshot_bytes(
            loaded,
            requests,
            &planned_work,
            args,
            plan.prefix_fanout_boundary_policy,
            root_plan.selected_prefix_tokens,
            max_capacity,
        )?;
        let additional_memory_bytes = root_snapshot_required_bytes
            .checked_add(max_cohort_snapshot_required_bytes)
            .context("file root plus cohort checkpoint bytes overflow")?;
        let admission = loaded
            .qwen_execution_memory_admission_with_additional_bytes(
                WIDTH,
                max_capacity,
                prefill_scratch_upper_bytes,
                E::executor_scratch_upper_bytes(loaded)?,
                additional_memory_bytes,
            )
            .context("price fixed-cohort file root")?;
        root_telemetry.snapshot_required_bytes = root_snapshot_required_bytes;
        root_telemetry.max_cohort_snapshot_required_bytes = max_cohort_snapshot_required_bytes;
        root_telemetry.additional_memory_bytes = additional_memory_bytes;
        root_telemetry.memory_admission_required_bytes = admission.required_bytes;
        root_telemetry.memory_admission_reason = admission.reason.as_str();
        if admission.admitted {
            let root_prompt = root_prompt_refs
                .first()
                .copied()
                .context("file-root plan has no source prompt")?;
            let (prepared, prefill_ms, snapshot_ms) = qwen_file_root::prepare(
                loaded,
                root_prompt,
                args.prefill_chunk,
                root_plan,
                max_capacity,
                root_snapshot_required_bytes,
            )?;
            root_telemetry.outcome = "prepared";
            root_telemetry.snapshot_bytes = prepared.checkpoint.snapshot_bytes();
            root_telemetry.prefill_ms = prefill_ms;
            root_telemetry.snapshot_ms = snapshot_ms;
            file_root = Some(prepared);
        } else {
            root_telemetry.outcome = "memory_fallback";
            root_telemetry.additional_memory_bytes = 0;
        }
    }
    let mut executor = if realized_full_cohorts > 0 {
        Some(executor_factory()?)
    } else {
        None
    };
    let realized_refill_arenas = planned_work
        .iter()
        .filter(|work| matches!(work, PlannedWork::RefillArena(_)))
        .count();
    let realized_refill_requests = planned_work
        .iter()
        .filter_map(|work| match work {
            PlannedWork::RefillArena(plan) => Some(plan.request_indices.len()),
            _ => None,
        })
        .sum();
    let realized_prefix_affinity_cohorts = plan
        .prefix_affinity_batches
        .iter()
        .filter(|batch| {
            planned_work
                .iter()
                .any(|work| matches!(work, PlannedWork::Batch(indices) if indices == *batch))
        })
        .count();
    let refill_telemetry_enabled = E::REFILL_TELEMETRY;
    let planner_telemetry = PlannerTelemetry {
        schema_version: E::PLANNER_TELEMETRY_SCHEMA_VERSION,
        backend: E::PLANNER_BACKEND,
        requests: requests.len(),
        prefix_packing_configured: plan.prefix_packing_configured,
        prefix_packing_effective: plan.prefix_packing_effective,
        prefix_fanout_boundary_policy: plan.prefix_fanout_boundary_policy.as_str(),
        ragged_prompts_enabled: plan.ragged_prompt_policy.enabled(),
        ragged_prompt_policy: plan.ragged_prompt_policy.as_str(),
        ragged_prompt_plan_decision: plan.ragged_prompt_plan_decision.as_str(),
        ragged_packing_strategy: plan.ragged_packing_strategy.as_str(),
        refill_policy: refill_telemetry_enabled.then_some(plan.refill_policy.as_str()),
        refill_default_max_prompt_tokens: refill_telemetry_enabled
            .then(|| plan.refill_policy.default_max_prompt_tokens())
            .flatten(),
        refill_configured: refill_telemetry_enabled.then_some(plan.refill_configured),
        refill_effective: refill_telemetry_enabled.then_some(plan.refill_effective),
        planned_refill_arenas: refill_telemetry_enabled.then_some(plan.refill_arenas),
        planned_refill_requests: refill_telemetry_enabled.then_some(plan.refill_requests),
        realized_refill_arenas: refill_telemetry_enabled.then_some(realized_refill_arenas),
        realized_refill_requests: refill_telemetry_enabled.then_some(realized_refill_requests),
        realized_refill_memory_denied_arenas: refill_telemetry_enabled
            .then_some(denied_refill_arena_count),
        realized_refill_memory_fallback_arenas: refill_telemetry_enabled
            .then_some(rejected_refill_arenas),
        automatic_refill_transaction_outcome: match automatic_refill_admission {
            AutomaticRefillAdmission::NotApplicable => None,
            AutomaticRefillAdmission::Admitted => Some("admitted"),
            AutomaticRefillAdmission::PlannerBaselineFallback { .. } => {
                Some("planner_baseline_fallback")
            }
        },
        automatic_refill_min_arenas: refill_telemetry_enabled
            .then_some(AUTOMATIC_DENSE_RAGGED_REFILL_MIN_ARENAS),
        automatic_refill_max_prompt_tokens: refill_telemetry_enabled
            .then_some(AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS),
        automatic_refill_max_generation_tokens: refill_telemetry_enabled
            .then_some(AUTOMATIC_DENSE_RAGGED_REFILL_MAX_GENERATION_TOKENS),
        automatic_refill_max_file_prefill_per_transition: refill_telemetry_enabled
            .then_some(AUTOMATIC_DENSE_RAGGED_REFILL_MAX_PREFILL_PER_TRANSITION),
        automatic_refill_max_arena_prefill_per_transition: refill_telemetry_enabled
            .then_some(AUTOMATIC_DENSE_RAGGED_REFILL_MAX_ARENA_PREFILL_PER_TRANSITION),
        automatic_refill_min_transition_utilization: refill_telemetry_enabled.then_some("9/10"),
        automatic_refill_min_idealized_step_savings: refill_telemetry_enabled.then_some("3/20"),
        compatibility_buckets: plan.compatibility_buckets,
        planned_candidate_cohorts: plan.candidate_cohorts,
        realized_prefix_affinity_cohorts,
        planned_prefix_fallback_buckets: plan.prefix_plan_fallback_buckets,
        planned_full_cohorts: plan.full_cohorts,
        planned_batched_requests: plan.full_cohorts * WIDTH,
        realized_full_cohorts,
        planned_economics_rejected_cohorts: plan.economics_rejected_cohorts,
        realized_memory_rejected_cohorts: memory_rejected_cohorts,
        realized_batched_requests,
        realized_serial_fallback_requests,
        realized_estimated_physical_transition_slots: realized_transition_slots,
        realized_estimated_capacity_slots: realized_capacity_slots,
        minimum_requested_transition_utilization: "3/4",
        packing_order: if plan.ragged_prompt_policy.enabled() {
            "ragged_pareto"
        } else if realized_prefix_affinity_cohorts > 0 {
            "prefix_affinity_then_generation_depth"
        } else {
            "generation_depth"
        },
        output_order: "input",
    };
    let mut pending_outputs = std::iter::repeat_with(|| None)
        .take(requests.len())
        .collect::<Vec<Option<RequestOutput>>>();
    let mut next_output = 0usize;
    let mut completed = 0usize;
    let mut cohort_index = 0usize;
    let mut refill_arena_index = 0usize;
    for work in planned_work {
        shutdown::checkpoint()?;
        match work {
            PlannedWork::Batch(indices) => {
                let cohort = indices.map(|index| &requests[index]);
                let (outputs, telemetry) = run_cohort(
                    loaded,
                    tokenizer,
                    executor.as_mut().expect("planned fixed-cohort executor"),
                    &cohort,
                    args,
                    &stop_tokens,
                    cohort_index,
                    plan.prefix_fanout_boundary_policy,
                    file_root.as_ref(),
                )
                .with_context(|| format!("run {} cohort {cohort_index}", E::DISPLAY_NAME))?;
                for (index, output) in indices.into_iter().zip(outputs) {
                    ensure!(
                        pending_outputs[index].replace(output).is_none(),
                        "{} planner produced request {index} twice",
                        E::DISPLAY_NAME
                    );
                }
                eprintln!(
                    "{}: {}",
                    E::TELEMETRY_PREFIX,
                    serde_json::to_string(&telemetry)
                        .with_context(|| format!("serialize {} telemetry", E::DISPLAY_NAME))?
                );
                if telemetry.file_root_restores.unwrap_or(0) > 0 {
                    root_telemetry.actual_cohort_uses += 1;
                }
                cohort_index += 1;
            }
            PlannedWork::RefillArena(plan) => {
                let (outputs, telemetry) = run_refill_arena(
                    loaded,
                    tokenizer,
                    executor.as_mut().expect("planned dense refill executor"),
                    requests,
                    args,
                    &stop_tokens,
                    refill_arena_index,
                    &plan,
                )
                .with_context(|| format!("run dense refill arena {refill_arena_index}"))?;
                for (index, output) in outputs {
                    ensure!(
                        pending_outputs[index].replace(output).is_none(),
                        "refill planner produced request {index} twice"
                    );
                }
                eprintln!(
                    "dense_batch8_refill: {}",
                    serde_json::to_string(&telemetry)
                        .context("serialize dense B=8 refill telemetry")?
                );
                refill_arena_index += 1;
            }
            PlannedWork::Serial(index) => {
                let (output, _) =
                    run_jsonl_request(loaded, tokenizer, &requests[index], args, greedy_gpu_mode)
                        .with_context(|| {
                        format!("run {} serial fallback request {index}", E::DISPLAY_NAME)
                    })?;
                ensure!(
                    pending_outputs[index].replace(output).is_none(),
                    "{} planner produced request {index} twice",
                    E::DISPLAY_NAME
                );
            }
        }
        shutdown::checkpoint()?;
        completed += flush_ready_outputs(stdout, &mut pending_outputs, &mut next_output)?;
    }
    ensure!(
        completed == requests.len() && next_output == requests.len(),
        "{} planner completed {completed}/{} requests",
        E::DISPLAY_NAME,
        requests.len()
    );
    root_telemetry.actual_avoided_prefix_evaluations =
        root_telemetry.actual_cohort_uses.saturating_sub(1);
    root_telemetry.actual_avoided_prompt_tokens = root_telemetry
        .selected_prefix_tokens
        .saturating_mul(root_telemetry.actual_avoided_prefix_evaluations);
    eprintln!(
        "fixed_cohort_file_root: {}",
        serde_json::to_string(&root_telemetry)
            .context("serialize fixed-cohort file-root telemetry")?
    );
    eprintln!(
        "{}: {}",
        E::PLANNER_TELEMETRY_PREFIX,
        serde_json::to_string(&planner_telemetry)
            .with_context(|| format!("serialize {} planner telemetry", E::DISPLAY_NAME))?
    );
    Ok(completed)
}

fn flush_ready_outputs(
    stdout: &mut impl Write,
    pending: &mut [Option<RequestOutput>],
    next: &mut usize,
) -> Result<usize> {
    let mut encoded = Vec::new();
    let start = *next;
    while let Some(output) = pending.get_mut(*next).and_then(Option::take) {
        serde_json::to_writer(&mut encoded, &output).context("encode request output")?;
        encoded.push(b'\n');
        *next += 1;
    }
    if !encoded.is_empty() {
        stdout
            .write_all(&encoded)
            .context("write fixed-cohort ordered outputs")?;
        stdout.flush()?;
    }
    Ok(*next - start)
}

fn validate_requests(requests: &[PreparedJsonlRequest], args: &Args) -> Result<()> {
    ensure!(
        !requests.is_empty(),
        "--batch-size requires at least one request"
    );
    for request in requests {
        ensure!(
            request.request.cache_prefix_tokens.is_none(),
            "request {} sets cache_prefix_tokens, which is unsupported with --batch-size",
            request.id
        );
        ensure!(
            request.sampling.temperature == 0.0,
            "request {} uses temperature {}; --batch-size currently requires greedy decoding",
            request.id,
            request.sampling.temperature
        );
        ensure!(
            request.auto_cache_prefix_tokens.is_none() && request.auto_cache_future_hits == 0,
            "request {} carried prefix-cache lookahead into fixed-cohort admission",
            request.id
        );
        jsonl_generation_capacity(request, args)?;
    }
    Ok(())
}

fn prepare_refill_lane(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    scratch: &mut MetalDFlashLayerMajorScratch,
    request_index: usize,
    request: &PreparedJsonlRequest,
    capacity: usize,
    args: &Args,
    stop_tokens: &[i32],
) -> Result<(Lane, f64)> {
    shutdown::checkpoint()?;
    let mut sequence = loaded
        .create_sequence(SequenceConfig::new(capacity))
        .with_context(|| format!("allocate refill sequence for request {request_index}"))?;
    let forward = loaded.forward();
    let (logits, prefill_ms) =
        prefill_span(&forward, &mut sequence, scratch, &request.prompt_ids, 0)
            .with_context(|| format!("prefill refill request {request_index}"))?;
    let mut sampler = Sampler::new(request.sampling)
        .with_context(|| format!("initialize refill sampler for request {request_index}"))?;
    let first = sampler
        .sample(&logits)
        .with_context(|| format!("select refill first token for request {request_index}"))?
        .token;
    let (requested_tokens, _) = jsonl_generation_capacity(request, args)?;
    let mut progress = LaneProgress::new(requested_tokens);
    let visible = progress.record_selection(first, stop_tokens)?;
    let mut generated_text = String::new();
    if visible {
        generated_text.push_str(&tokenizer.decode_piece(first));
    }
    Ok((
        Lane {
            request_index,
            id: request.id.clone(),
            prompt_tokens: request.prompt_ids.len(),
            sequence,
            progress,
            generated_text,
        },
        prefill_ms,
    ))
}

fn finalize_refill_lane(lane: Lane) -> Result<(usize, RequestOutput)> {
    lane.progress
        .validate_request_complete()
        .with_context(|| format!("validate refill lane {}", lane.id))?;
    ensure!(
        lane.sequence.position()
            == lane.prompt_tokens
                + lane.progress.logical_transitions
                + lane.progress.padding_transitions,
        "refill lane {} sequence frontier drifted",
        lane.id
    );
    let stop_reason = lane
        .progress
        .stop_reason
        .expect("validated refill lane termination");
    Ok((
        lane.request_index,
        RequestOutput {
            id: lane.id,
            prompt_tokens: lane.prompt_tokens,
            generated_tokens: lane.progress.generated.len(),
            generated_token_sha256: generated_token_sha256(&lane.progress.generated),
            generated_text: lane.generated_text,
            stop_reason,
            terminal_token_target_transition_consumed: false,
            thinking_partition: None,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn run_refill_arena<const WIDTH: usize, E: FixedCohortExecutor<WIDTH>>(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    executor: &mut E,
    requests: &[PreparedJsonlRequest],
    args: &Args,
    stop_tokens: &[i32],
    arena_index: usize,
    plan: &RefillArenaPlan<WIDTH>,
) -> Result<(Vec<(usize, RequestOutput)>, RefillTelemetry)> {
    ensure!(
        WIDTH == DENSE_BATCH8_WIDTH,
        "synchronous refill is authorized only for dense B=8"
    );
    ensure!(
        plan.request_indices.len() > WIDTH && plan.request_indices.len() <= WIDTH * 2,
        "refill arena must contain one initial wave and at most one replacement wave"
    );
    let arena_t0 = Instant::now();
    let max_prompt_tokens = plan
        .request_indices
        .iter()
        .map(|&index| requests[index].prompt_ids.len())
        .max()
        .unwrap_or(0);
    let chunk = match args.prefill_chunk {
        PrefillChunkArg::Fixed(requested) => requested.min(max_prompt_tokens.max(1)),
        PrefillChunkArg::Auto => bail!("dense B=8 refill requires a fixed prefill chunk"),
    };
    let mut scratch = allocate_legacy_prefill_scratch(loaded, chunk, max_prompt_tokens)?;
    let mut waiting = plan.request_indices.iter().copied();
    let prefill_t0 = Instant::now();
    let mut lanes: [Option<Lane>; WIDTH] = std::array::from_fn(|_| None);
    let mut initial_prefill_ms = 0.0;
    let mut prefill_tokens = 0usize;
    for lane_slot in &mut lanes {
        let request_index = waiting.next().context("refill initial wave underfilled")?;
        let request = &requests[request_index];
        let (lane, ms) = prepare_refill_lane(
            loaded,
            tokenizer,
            &mut scratch,
            request_index,
            request,
            plan.shared_capacity,
            args,
            stop_tokens,
        )?;
        initial_prefill_ms += ms;
        prefill_tokens += request.prompt_ids.len();
        *lane_slot = Some(lane);
    }
    let initial_prepare_wall_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

    let mut outputs = Vec::with_capacity(plan.request_indices.len());
    let mut physical_batch_steps = 0usize;
    let mut productive_transitions = 0usize;
    let mut padding_transitions = 0usize;
    let mut replacements = 0usize;
    let mut replacement_prefill_ms = 0.0;
    let mut replacement_prepare_wall_ms = 0.0;
    let mut batch_transition_ms = 0.0;
    let mut executor_gpu_ms = Some(0.0);
    let decode_t0 = Instant::now();
    loop {
        for lane_slot in &mut lanes {
            while lane_slot
                .as_ref()
                .is_some_and(|lane| !lane.progress.is_active())
            {
                let Some(request_index) = waiting.next() else {
                    break;
                };
                let lane = lane_slot.take().expect("finished refill lane");
                outputs.push(finalize_refill_lane(lane)?);
                let request = &requests[request_index];
                let replacement_t0 = Instant::now();
                let (lane, ms) = prepare_refill_lane(
                    loaded,
                    tokenizer,
                    &mut scratch,
                    request_index,
                    request,
                    plan.shared_capacity,
                    args,
                    stop_tokens,
                )?;
                replacement_prefill_ms += ms;
                replacement_prepare_wall_ms += replacement_t0.elapsed().as_secs_f64() * 1e3;
                prefill_tokens += request.prompt_ids.len();
                replacements += 1;
                *lane_slot = Some(lane);
            }
        }
        if lanes
            .iter()
            .all(|lane| lane.as_ref().is_none_or(|lane| !lane.progress.is_active()))
        {
            break;
        }
        shutdown::checkpoint()?;
        let active: [bool; WIDTH] = std::array::from_fn(|slot| {
            lanes[slot]
                .as_ref()
                .is_some_and(|lane| lane.progress.is_active())
        });
        let token_ids: [i32; WIDTH] = std::array::from_fn(|slot| {
            lanes[slot]
                .as_ref()
                .map(|lane| lane.progress.transition_token())
                .unwrap_or(FINISHED_LANE_FILL_TOKEN)
        });
        let transition_t0 = Instant::now();
        let step = {
            let lane_refs = lanes.each_mut();
            executor.step_greedy(
                token_ids,
                lane_refs.map(|lane| {
                    &mut lane
                        .as_mut()
                        .expect("refill keeps one physical sequence in every slot")
                        .sequence
                }),
                || shutdown::checkpoint().is_err(),
            )?
        };
        shutdown::checkpoint()?;
        batch_transition_ms += transition_t0.elapsed().as_secs_f64() * 1e3;
        executor_gpu_ms = match (executor_gpu_ms, step.gpu_ms) {
            (Some(total), Some(step)) => Some(total + step),
            _ => None,
        };
        physical_batch_steps += 1;
        for slot in 0..WIDTH {
            let lane = lanes[slot].as_mut().expect("refill physical lane");
            if active[slot] {
                productive_transitions += 1;
            } else {
                padding_transitions += 1;
            }
            if lane
                .progress
                .record_batch_step(active[slot], step.argmax_ids[slot], stop_tokens)?
            {
                lane.generated_text
                    .push_str(&tokenizer.decode_piece(step.argmax_ids[slot]));
            }
        }
    }
    for lane in lanes.into_iter().flatten() {
        outputs.push(finalize_refill_lane(lane)?);
    }
    ensure!(
        outputs.len() == plan.request_indices.len(),
        "refill completed {}/{} requests",
        outputs.len(),
        plan.request_indices.len()
    );
    let transition_and_refill_wall_ms = decode_t0.elapsed().as_secs_f64() * 1e3;
    let arena_wall_ms = arena_t0.elapsed().as_secs_f64() * 1e3;
    let generated_tokens = outputs
        .iter()
        .map(|(_, output)| output.generated_tokens)
        .sum::<usize>();
    let physical_slots = physical_batch_steps
        .checked_mul(WIDTH)
        .context("refill physical slot overflow")?;
    ensure!(
        productive_transitions + padding_transitions == physical_slots,
        "refill transition accounting drifted"
    );
    let physical_transition_utilization = if physical_slots > 0 {
        productive_transitions as f64 / physical_slots as f64
    } else {
        0.0
    };
    let productive_transition_tps = if batch_transition_ms > 0.0 {
        productive_transitions as f64 / (batch_transition_ms / 1e3)
    } else {
        0.0
    };
    let arena_generated_tps = if arena_wall_ms > 0.0 {
        generated_tokens as f64 / (arena_wall_ms / 1e3)
    } else {
        0.0
    };
    Ok((
        outputs,
        RefillTelemetry {
            schema_version: 1,
            backend: "dense_qwen_refill_b8_v1",
            arena_index,
            width: WIDTH,
            requests: plan.request_indices.len(),
            shared_capacity_tokens: plan.shared_capacity,
            simulated_physical_steps: plan.simulated_physical_steps,
            simulated_productive_transitions: plan.simulated_productive_transitions,
            requested_two_wave_steps: plan.requested_two_wave_steps,
            physical_batch_steps,
            productive_transitions,
            padding_transitions,
            physical_transition_utilization,
            replacements,
            prefill_tokens,
            initial_prefill_model_ms: initial_prefill_ms,
            initial_prepare_wall_ms,
            replacement_prefill_model_ms: replacement_prefill_ms,
            replacement_prepare_wall_ms,
            arena_wall_ms,
            transition_and_refill_wall_ms,
            batch_transition_ms,
            executor_gpu_ms,
            generated_tokens,
            productive_transition_tps,
            arena_generated_tps,
            output_order: "input",
        },
    ))
}

fn run_cohort<const WIDTH: usize, E: FixedCohortExecutor<WIDTH>>(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    executor: &mut E,
    requests: &[&PreparedJsonlRequest; WIDTH],
    args: &Args,
    stop_tokens: &[i32],
    cohort_index: usize,
    prefix_fanout_boundary_policy: QwenPrefixFanoutBoundaryPolicy,
    file_root: Option<&qwen_file_root::Prepared>,
) -> Result<(Vec<RequestOutput>, CohortTelemetry)> {
    let generation_plan = cohort_generation_plan(requests, args)?;
    let prompt_tokens_per_lane = &generation_plan.prompt_tokens_per_lane;
    let min_prompt_tokens = generation_plan.min_prompt_tokens;
    let max_prompt_tokens = generation_plan.max_prompt_tokens;
    let capacity = generation_plan.shared_capacity;
    let chunk = match args.prefill_chunk {
        PrefillChunkArg::Fixed(requested) => requested.min(max_prompt_tokens.max(1)),
        PrefillChunkArg::Auto => bail!("{} requires a fixed prefill chunk", E::DISPLAY_NAME),
    };
    ensure!(
        chunk > 0,
        "{} prefill chunk must be nonzero",
        E::DISPLAY_NAME
    );
    let mut prefix_fanout = align_prefix_fanout(
        plan_prefix_fanout(
            requests,
            parse_prefix_fanout_enabled(std::env::var_os(E::PREFIX_FANOUT_ENV).as_deref()),
        ),
        max_prompt_tokens,
        chunk,
        prefix_fanout_boundary_policy,
    );
    if let Some(file_root) = file_root {
        let root_tokens = file_root.plan.selected_prefix_tokens;
        if prefix_fanout.selected_prefix_tokens < root_tokens {
            prefix_fanout.selected_prefix_tokens = root_tokens;
            prefix_fanout.reason = "selected_file_root";
        }
    }

    let mut scratch = allocate_legacy_prefill_scratch(loaded, chunk, max_prompt_tokens)?;
    let mut sequences = Vec::with_capacity(WIDTH);
    let before_first_sequence = loaded.context().current_allocated_size();
    sequences.push(
        loaded
            .create_sequence(SequenceConfig::new(capacity))
            .with_context(|| format!("allocate first {} sequence", E::DISPLAY_NAME))?,
    );
    let after_first_sequence = loaded.context().current_allocated_size();
    let sequence_allocation_delta_bytes = after_first_sequence
        .checked_sub(before_first_sequence)
        .filter(|&bytes| bytes > 0)
        .with_context(|| {
            format!(
                "{} sequence allocation did not produce a valid Metal byte delta",
                E::DISPLAY_NAME
            )
        })?;
    let remaining_sequence_required_bytes = sequence_allocation_delta_bytes
        .checked_mul((WIDTH - 1) as u64)
        .with_context(|| {
            format!(
                "{} remaining sequence byte estimate overflow",
                E::DISPLAY_NAME
            )
        })?;
    let file_root_tokens = file_root.map_or(0, |root| root.plan.selected_prefix_tokens);
    let mut prefix_snapshot_required_bytes =
        if prefix_fanout.selected_prefix_tokens > file_root_tokens {
            loaded
                .estimate_checkpoint_boundary_sizes(
                    &sequences[0],
                    prefix_fanout.selected_prefix_tokens,
                    false,
                    false,
                    0,
                )
                .with_context(|| format!("estimate {} prefix fanout snapshot", E::DISPLAY_NAME))?
                .snapshot_bytes
        } else {
            0
        };
    let mut memory_admission_incremental_bytes = remaining_sequence_required_bytes
        .checked_add(prefix_snapshot_required_bytes)
        .with_context(|| format!("{} fanout admission byte overflow", E::DISPLAY_NAME))?;
    let mut memory_admission = evaluate_metal_memory_admission_with_cpu_bytes(
        remaining_sequence_required_bytes,
        prefix_snapshot_required_bytes,
        TRANSIENT_RESERVE_BYTES,
        loaded.context().memory_signals(),
        true,
    );
    if !memory_admission.admitted && prefix_fanout.selected_prefix_tokens > 0 {
        let without_fanout = evaluate_metal_memory_admission(
            remaining_sequence_required_bytes,
            TRANSIENT_RESERVE_BYTES,
            loaded.context().memory_signals(),
            true,
        );
        if without_fanout.admitted {
            prefix_fanout.selected_prefix_tokens = 0;
            prefix_fanout.reason = "memory_fallback";
            prefix_snapshot_required_bytes = 0;
            memory_admission_incremental_bytes = remaining_sequence_required_bytes;
            memory_admission = without_fanout;
        }
    }
    ensure!(
        memory_admission.admitted,
        "{} memory admission denied: reason={} required_bytes={:?} working_set_headroom_bytes={:?} process_limit_remaining_bytes={:?}",
        E::DISPLAY_NAME,
        memory_admission.reason.as_str(),
        memory_admission.required_bytes,
        memory_admission.working_set_headroom_bytes,
        memory_admission.signals.process_limit_remaining_bytes,
    );
    for _ in 1..WIDTH {
        sequences.push(
            loaded
                .create_sequence(SequenceConfig::new(capacity))
                .with_context(|| format!("allocate admitted {} sequence", E::DISPLAY_NAME))?,
        );
    }

    let forward = loaded.forward();
    let prefill_t0 = Instant::now();
    let mut prompt_logits = Vec::with_capacity(WIDTH);
    let mut prefix_snapshot_bytes = 0u64;
    let mut prefix_prefill_ms = 0.0;
    let mut prefix_snapshot_ms = 0.0;
    let mut prefix_restore_ms = 0.0;
    let mut file_root_restores = 0usize;
    let mut file_root_restore_ms = 0.0;
    let mut suffix_prefill_ms = 0.0;
    let mut private_suffix_singleton_lanes = 0usize;
    let mut private_suffix_singleton_tokens = 0usize;
    let mut private_suffix_packed_lanes = 0usize;
    let mut private_suffix_packed_tokens = 0usize;
    if prefix_fanout.selected_prefix_tokens > 0 {
        let prefix_len = prefix_fanout.selected_prefix_tokens;
        let (prefix_logits, ms) = if let Some(file_root) = file_root {
            let restore_t0 = Instant::now();
            let restored = loaded
                .restore_prepared_checkpoint(
                    &file_root.checkpoint,
                    &mut sequences[0],
                    &requests[0].prompt_ids,
                )
                .with_context(|| format!("restore {} file root", E::DISPLAY_NAME))?;
            let restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
            prefix_restore_ms += restore_ms;
            file_root_restore_ms += restore_ms;
            file_root_restores += 1;
            ensure!(
                restored.matched_prefix_len == file_root_tokens
                    && restored.restored_prefix_len == file_root_tokens,
                "{} restored an unexpected file-root boundary",
                E::DISPLAY_NAME
            );
            if prefix_len == file_root_tokens {
                (
                    file_root
                        .checkpoint
                        .final_logits()
                        .context("fixed-cohort file root omitted final logits")?
                        .to_vec(),
                    0.0,
                )
            } else {
                prefill_span(
                    &forward,
                    &mut sequences[0],
                    &mut scratch,
                    &requests[0].prompt_ids[file_root_tokens..prefix_len],
                    file_root_tokens,
                )
                .with_context(|| format!("prefill {} prefix above file root", E::DISPLAY_NAME))?
            }
        } else {
            prefill_span(
                &forward,
                &mut sequences[0],
                &mut scratch,
                &requests[0].prompt_ids[..prefix_len],
                0,
            )
            .with_context(|| format!("prefill {} shared prefix", E::DISPLAY_NAME))?
        };
        prefix_prefill_ms = ms;

        let prepared = if prefix_len > file_root_tokens {
            let snapshot_t0 = Instant::now();
            let prepared = loaded
                .prepare_checkpoint_boundary(
                    &sequences[0],
                    requests[0].prompt_ids[..prefix_len].to_vec(),
                    None,
                    None,
                    None,
                    0,
                )
                .with_context(|| format!("capture {} shared prefix", E::DISPLAY_NAME))?;
            prefix_snapshot_ms = snapshot_t0.elapsed().as_secs_f64() * 1e3;
            prefix_snapshot_bytes = prepared.snapshot_bytes();
            ensure!(
                prefix_snapshot_bytes == prefix_snapshot_required_bytes,
                "{} prefix snapshot bytes {} != estimate {}",
                E::DISPLAY_NAME,
                prefix_snapshot_bytes,
                prefix_snapshot_required_bytes,
            );
            Some(prepared)
        } else {
            None
        };

        for slot in 0..WIDTH {
            shutdown::checkpoint()?;
            if slot > 0 {
                let restore_t0 = Instant::now();
                let checkpoint = prepared
                    .as_ref()
                    .or_else(|| file_root.map(|root| &root.checkpoint))
                    .expect("selected prefix requires a checkpoint");
                let restored = loaded
                    .restore_prepared_checkpoint(
                        checkpoint,
                        &mut sequences[slot],
                        &requests[slot].prompt_ids,
                    )
                    .with_context(|| {
                        format!("restore {} shared prefix into slot {slot}", E::DISPLAY_NAME)
                    })?;
                let restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
                prefix_restore_ms += restore_ms;
                if prepared.is_none() && file_root.is_some() {
                    file_root_restore_ms += restore_ms;
                    file_root_restores += 1;
                }
                ensure!(
                    restored.matched_prefix_len == prefix_len
                        && restored.restored_prefix_len == prefix_len,
                    "{} slot {slot} restored an unexpected prefix boundary",
                    E::DISPLAY_NAME
                );
            }
            let suffix = &requests[slot].prompt_ids[prefix_len..];
            if suffix.is_empty() {
                prompt_logits.push(prefix_logits.clone());
            } else {
                let result = prefill_private_suffix(
                    &forward,
                    &mut sequences[slot],
                    &mut scratch,
                    suffix,
                    prefix_len,
                )
                .with_context(|| {
                    format!("prefill {} cohort suffix slot {slot}", E::DISPLAY_NAME)
                })?;
                suffix_prefill_ms += result.ms;
                match result.mode {
                    PrivateSuffixExecutionMode::Packed => {
                        private_suffix_packed_lanes += 1;
                        private_suffix_packed_tokens += suffix.len();
                    }
                    PrivateSuffixExecutionMode::Singleton => {
                        private_suffix_singleton_lanes += 1;
                        private_suffix_singleton_tokens += suffix.len();
                    }
                }
                prompt_logits.push(result.logits);
            }
        }
    } else {
        for (slot, (request, sequence)) in requests.iter().zip(&mut sequences).enumerate() {
            shutdown::checkpoint()?;
            let (logits, ms) =
                prefill_span(&forward, sequence, &mut scratch, &request.prompt_ids, 0)
                    .with_context(|| format!("prefill {} cohort slot {slot}", E::DISPLAY_NAME))?;
            suffix_prefill_ms += ms;
            prompt_logits.push(logits);
        }
    }
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;
    drop(scratch);
    {
        let sequences: &mut [Sequence; WIDTH] = sequences
            .as_mut_slice()
            .try_into()
            .expect("fixed-cohort sequence width");
        executor
            .validate([FINISHED_LANE_FILL_TOKEN; WIDTH], sequences.each_mut())
            .with_context(|| format!("validate {} cohort backend", E::DISPLAY_NAME))?;
    }

    let decode_t0 = Instant::now();
    let mut lanes = Vec::with_capacity(WIDTH);
    for (slot, ((request, sequence), logits)) in requests
        .iter()
        .zip(sequences)
        .zip(prompt_logits)
        .enumerate()
    {
        let mut sampler = Sampler::new(request.sampling)
            .with_context(|| format!("initialize {} greedy sampler", E::DISPLAY_NAME))?;
        let first = sampler
            .sample(&logits)
            .with_context(|| format!("select {} first token", E::DISPLAY_NAME))?
            .token;
        let mut progress = LaneProgress::new(generation_plan.requested_tokens[slot]);
        let visible = progress.record_selection(first, stop_tokens)?;
        let mut generated_text = String::new();
        if visible {
            generated_text.push_str(&tokenizer.decode_piece(first));
        }
        lanes.push(Lane {
            request_index: slot,
            id: request.id.clone(),
            prompt_tokens: request.prompt_ids.len(),
            sequence,
            progress,
            generated_text,
        });
    }

    let mut physical_batch_steps = 0usize;
    let mut batch_transition_ms = 0.0;
    let mut executor_gpu_ms = Some(0.0);
    while lanes.iter().any(|lane| lane.progress.is_active()) {
        shutdown::checkpoint()?;
        let active: [bool; WIDTH] = std::array::from_fn(|slot| lanes[slot].progress.is_active());
        let token_ids: [i32; WIDTH] =
            std::array::from_fn(|slot| lanes[slot].progress.transition_token());
        let transition_t0 = Instant::now();
        let step = {
            let lanes: &mut [Lane; WIDTH] = lanes
                .as_mut_slice()
                .try_into()
                .expect("fixed-cohort lane width");
            executor.step_greedy(
                token_ids,
                lanes.each_mut().map(|lane| &mut lane.sequence),
                || shutdown::checkpoint().is_err(),
            )?
        };
        shutdown::checkpoint()?;
        batch_transition_ms += transition_t0.elapsed().as_secs_f64() * 1e3;
        executor_gpu_ms = match (executor_gpu_ms, step.gpu_ms) {
            (Some(total), Some(step)) => Some(total + step),
            _ => None,
        };
        physical_batch_steps += 1;

        for (slot, lane) in lanes.iter_mut().enumerate() {
            let token = step.argmax_ids[slot];
            if lane
                .progress
                .record_batch_step(active[slot], token, stop_tokens)?
            {
                lane.generated_text.push_str(&tokenizer.decode_piece(token));
            }
        }
    }
    let decode_ms = decode_t0.elapsed().as_secs_f64() * 1e3;

    let generated_tokens = lanes
        .iter()
        .map(|lane| lane.progress.generated.len())
        .sum::<usize>();
    let productive_transitions = lanes
        .iter()
        .map(|lane| lane.progress.logical_transitions)
        .sum::<usize>();
    let padding_transitions = lanes
        .iter()
        .map(|lane| lane.progress.padding_transitions)
        .sum::<usize>();
    let generated_tokens_per_lane = lanes
        .iter()
        .map(|lane| lane.progress.generated.len())
        .collect::<Vec<_>>();
    let productive_transitions_per_lane = lanes
        .iter()
        .map(|lane| lane.progress.logical_transitions)
        .collect::<Vec<_>>();
    let padding_transitions_per_lane = lanes
        .iter()
        .map(|lane| lane.progress.padding_transitions)
        .collect::<Vec<_>>();
    let physical_transition_slots = physical_batch_steps
        .checked_mul(WIDTH)
        .context("fixed-cohort realized transition slot overflow")?;
    let physical_transition_utilization = if physical_transition_slots > 0 {
        productive_transitions as f64 / physical_transition_slots as f64
    } else {
        0.0
    };
    let aggregate_generated_tps = if decode_ms > 0.0 {
        generated_tokens as f64 / (decode_ms / 1e3)
    } else {
        0.0
    };
    let mut outputs = Vec::with_capacity(WIDTH);
    for lane in lanes {
        lane.progress
            .validate_complete(physical_batch_steps)
            .with_context(|| format!("validate {} lane {}", E::DISPLAY_NAME, lane.id))?;
        ensure!(
            lane.sequence.position() == lane.prompt_tokens + physical_batch_steps,
            "{} lane {} sequence frontier drifted",
            E::DISPLAY_NAME,
            lane.id
        );
        let stop_reason = lane
            .progress
            .stop_reason
            .expect("validated fixed-cohort lane termination");
        outputs.push(RequestOutput {
            id: lane.id,
            prompt_tokens: lane.prompt_tokens,
            generated_tokens: lane.progress.generated.len(),
            generated_token_sha256: generated_token_sha256(&lane.progress.generated),
            generated_text: lane.generated_text,
            stop_reason,
            terminal_token_target_transition_consumed: false,
            thinking_partition: None,
        });
    }
    let moe_plan = executor.moe_plan_telemetry();
    Ok((
        outputs,
        CohortTelemetry {
            schema_version: E::TELEMETRY_SCHEMA_VERSION,
            backend: E::COHORT_BACKEND,
            cohort_index,
            width: WIDTH,
            prompt_tokens_per_lane: prompt_tokens_per_lane.clone(),
            min_prompt_tokens,
            max_prompt_tokens,
            requested_tokens_per_lane: generation_plan.requested_tokens,
            shared_capacity_tokens: generation_plan.shared_capacity,
            generated_tokens_per_lane,
            productive_transitions_per_lane,
            padding_transitions_per_lane,
            generated_tokens,
            productive_transitions,
            padding_transitions,
            physical_batch_steps,
            physical_transition_utilization,
            common_prefix_tokens: prefix_fanout.common_prefix_tokens,
            prefix_fanout_tokens: prefix_fanout.selected_prefix_tokens,
            prefix_fanout_reason: prefix_fanout.reason,
            prefix_fanout_boundary_policy: prefix_fanout_boundary_policy.as_str(),
            prefix_fanout_min_tokens: PREFIX_FANOUT_MIN_TOKENS,
            prefix_snapshot_bytes,
            prefix_prefill_ms,
            prefix_snapshot_ms,
            prefix_restore_ms,
            suffix_prefill_ms,
            private_suffix_singleton_max_tokens: PRIVATE_SUFFIX_SINGLETON_MAX_TOKENS,
            private_suffix_singleton_lanes,
            private_suffix_singleton_tokens,
            private_suffix_packed_lanes,
            private_suffix_packed_tokens,
            prefill_ms,
            decode_ms,
            batch_transition_ms,
            executor_gpu_ms,
            aggregate_generated_tps,
            sequence_allocation_delta_bytes,
            remaining_sequence_required_bytes,
            prefix_snapshot_required_bytes,
            memory_admission_incremental_bytes,
            memory_admission_reserve_bytes: memory_admission.reserve_bytes,
            memory_admission_required_bytes: memory_admission.required_bytes,
            memory_admission_reason: memory_admission.reason.as_str(),
            file_root_tokens: file_root.map(|_| file_root_tokens),
            file_root_restores: file_root.map(|_| file_root_restores),
            file_root_restore_ms: file_root.map(|_| file_root_restore_ms),
            executor_scratch_bytes: executor.scratch_bytes(),
            // The executor is created before the first sequence delta and this
            // admission snapshot, so its persistent Metal arena is already in
            // current_allocated_size and memory_signals. Adding it here would
            // charge the MoE scratch twice.
            executor_scratch_incremental_bytes: executor.scratch_bytes().map(|_| 0),
            moe_q8_batched_gdn_blocks: moe_plan.map(|plan| plan.q8_batched_gdn_blocks),
            moe_packed_q4_gate_up_blocks: moe_plan.map(|plan| plan.packed_q4_gate_up_blocks),
            moe_per_lane_gdn_blocks: moe_plan.map(|plan| plan.per_lane_gdn_blocks),
            moe_per_lane_gate_up_blocks: moe_plan.map(|plan| plan.per_lane_gate_up_blocks),
            moe_per_lane_iq3_gate_up_blocks: moe_plan.map(|plan| plan.per_lane_iq3_gate_up_blocks),
            moe_per_lane_other_gate_up_blocks: moe_plan
                .map(|plan| plan.per_lane_other_gate_up_blocks),
            moe_head_mode: moe_plan.map(|plan| match plan.head_mode {
                HeadMode::BatchedQ6 => "batched_q6",
                HeadMode::BatchedQ8 => "batched_q8",
                HeadMode::PerLane => "per_lane",
            }),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn test_args() -> Args {
        Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "requests.jsonl",
            "--batch-size",
            "8",
            "--tokens",
            "3",
        ])
        .expect("valid dense B=8 test arguments")
    }

    fn prepared(id: &str, tokens: &[i32]) -> PreparedJsonlRequest {
        PreparedJsonlRequest {
            request: JsonlRequest {
                id: Some(id.to_string()),
                prompt: None,
                prompt_file: None,
                tokens: None,
                cache_prefix_tokens: None,
                sampling: None,
            },
            id: id.to_string(),
            line: 1,
            prompt_ids: tokens.to_vec(),
            sampling: SamplingConfig::default(),
            auto_cache_prefix_tokens: None,
            auto_cache_future_hits: 0,
        }
    }

    fn output(id: &str) -> RequestOutput {
        RequestOutput {
            id: id.to_string(),
            prompt_tokens: 1,
            generated_tokens: 1,
            generated_token_sha256: id.to_string(),
            generated_text: id.to_string(),
            stop_reason: StopReason::TokenLimit,
            terminal_token_target_transition_consumed: false,
            thinking_partition: None,
        }
    }

    #[test]
    fn cli_contract_is_explicit_and_family_scoped() {
        let args = test_args();
        validate_cli(&args, ExplicitCliOptions::default()).unwrap();
        validate_model_family(args.batch_size, Some(ModelFamily::Qwen35)).unwrap();
        assert!(validate_model_family(args.batch_size, Some(ModelFamily::Qwen35Moe)).is_err());
        assert!(validate_model_family(args.batch_size, Some(ModelFamily::DeepSeek4)).is_err());
        validate_model_family(Some(16), Some(ModelFamily::Qwen35Moe)).unwrap();
        assert!(validate_model_family(Some(16), Some(ModelFamily::Qwen35)).is_err());
        assert!(validate_model_family(Some(16), Some(ModelFamily::DeepSeek4)).is_err());
        let mut moe_args = test_args();
        moe_args.batch_size = Some(MOE_BATCH16_WIDTH);
        validate_cli(&moe_args, ExplicitCliOptions::default()).unwrap();
        assert_eq!(
            parse_greedy_gpu_argmax_mode(Some(OsStr::new("0"))),
            GreedyGpuArgmaxMode::ExplicitRollback
        );
        assert!(
            validate_greedy_gpu_mode(
                GreedyGpuArgmaxMode::ExplicitRollback,
                Some(DENSE_BATCH8_WIDTH)
            )
            .is_err()
        );
        assert!(
            validate_greedy_gpu_mode(
                GreedyGpuArgmaxMode::ExplicitRollback,
                Some(MOE_BATCH16_WIDTH)
            )
            .is_err()
        );
        validate_greedy_gpu_mode(GreedyGpuArgmaxMode::DefaultOff, Some(DENSE_BATCH8_WIDTH))
            .unwrap();
        validate_greedy_gpu_mode(GreedyGpuArgmaxMode::ForceEnabled, Some(MOE_BATCH16_WIDTH))
            .unwrap();

        let mut invalid = test_args();
        invalid.batch_size = Some(4);
        assert!(validate_cli(&invalid, ExplicitCliOptions::default()).is_err());
        invalid = test_args();
        invalid.requests_jsonl = Some(PathBuf::from("-"));
        assert!(validate_cli(&invalid, ExplicitCliOptions::default()).is_err());
        invalid = test_args();
        invalid.temperature = 0.7;
        assert!(validate_cli(&invalid, ExplicitCliOptions::default()).is_err());
        invalid = test_args();
        invalid.info = true;
        assert!(validate_cli(&invalid, ExplicitCliOptions::default()).is_err());
        assert!(
            validate_cli(
                &test_args(),
                ExplicitCliOptions {
                    prefix_cache_max_mib: true,
                    ..ExplicitCliOptions::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn planner_forms_compatible_cohorts_and_serializes_underfill() {
        let args = test_args();
        let mut requests = (0..DENSE_BATCH8_WIDTH)
            .map(|slot| {
                let mut request = prepared(&format!("a-{slot}"), &[1, 2]);
                request.request.tokens = Some(24 + slot);
                request
            })
            .chain((0..DENSE_BATCH8_WIDTH).map(|slot| {
                let mut request = prepared(&format!("b-{slot}"), &[1, 2, 3]);
                request.request.tokens = Some(4);
                request
            }))
            .collect::<Vec<_>>();
        let mut a_underfill = prepared("a-underfill", &[1, 2]);
        a_underfill.request.tokens = Some(32);
        requests.push(a_underfill);
        requests.push(prepared("c-underfill", &[1, 2, 3, 4]));
        validate_requests(&requests, &args).unwrap();
        let plan = plan_request_work::<DENSE_BATCH8_WIDTH>(&requests, &args).unwrap();
        assert_eq!(plan.compatibility_buckets, 3);
        assert_eq!(plan.candidate_cohorts, 2);
        assert_eq!(plan.full_cohorts, 2);
        assert_eq!(plan.economics_rejected_cohorts, 0);
        assert_eq!(plan.serial_fallback_requests, 2);
        assert_eq!(
            plan.work,
            vec![
                PlannedWork::Batch([0, 1, 2, 3, 4, 5, 6, 7]),
                PlannedWork::Batch([8, 9, 10, 11, 12, 13, 14, 15]),
                PlannedWork::Serial(16),
                PlannedWork::Serial(17),
            ]
        );
        let underfill = plan_request_work::<DENSE_BATCH8_WIDTH>(&requests[..7], &args).unwrap();
        assert_eq!(underfill.full_cohorts, 0);
        assert_eq!(underfill.serial_fallback_requests, 7);

        requests[5].sampling.temperature = 0.1;
        assert!(validate_requests(&requests, &args).is_err());
        requests[5].sampling.temperature = 0.0;
        requests[6].request.cache_prefix_tokens = Some(1);
        assert!(validate_requests(&requests, &args).is_err());
    }

    #[test]
    fn ragged_prompt_policy_forms_one_cross_length_cohort() {
        let args = test_args();
        let requests = (0..DENSE_BATCH8_WIDTH)
            .map(|slot| {
                let mut request = prepared(&format!("slot-{slot}"), &vec![7; slot + 1]);
                request.request.tokens = Some(24);
                request
            })
            .collect::<Vec<_>>();
        let control = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            true,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            false,
        )
        .unwrap();
        assert_eq!(control.full_cohorts, 0);
        assert_eq!(control.serial_fallback_requests, DENSE_BATCH8_WIDTH);

        let ragged = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            true,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            true,
            false,
        )
        .unwrap();
        assert!(ragged.ragged_prompt_policy.enabled());
        assert_eq!(ragged.compatibility_buckets, 1);
        assert_eq!(ragged.full_cohorts, 1);
        assert_eq!(ragged.serial_fallback_requests, 0);
        assert_eq!(
            ragged.work,
            vec![PlannedWork::Batch([0, 1, 2, 3, 4, 5, 6, 7])]
        );
    }

    #[test]
    fn automatic_dense_ragged_is_charged_and_moe_remains_explicit() {
        let mut args = test_args();
        args.execution_mode = Some(execution_selector::ExecutionModeArg::Auto);
        args.prefill_chunk = PrefillChunkArg::Fixed(512);
        let make_requests = |requested_tokens: usize, longest_prompt: usize| {
            let measured_prompt_tokens = [10, 11, 13, 17, 25, 41, 73, longest_prompt];
            (0..DENSE_BATCH8_WIDTH)
                .map(|slot| {
                    let prompt_tokens = measured_prompt_tokens[slot];
                    let mut request = prepared(
                        &format!("auto-ragged-{requested_tokens}-{slot}"),
                        &vec![7; prompt_tokens],
                    );
                    request.request.tokens = Some(requested_tokens);
                    request
                })
                .collect::<Vec<_>>()
        };

        let admitted = make_requests(32, 137);
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &admitted,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_eq!(
            plan.ragged_prompt_policy,
            RaggedPromptPolicy::AutomaticDenseCharged
        );
        assert_eq!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticAdmitted
        );
        assert_eq!(plan.full_cohorts, 1);
        assert_eq!(plan.compatibility_buckets, 1);
        assert!(!plan.refill_configured);

        let too_little_decode = make_requests(16, 137);
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &too_little_decode,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_eq!(plan.ragged_prompt_policy, RaggedPromptPolicy::Disabled);
        assert_eq!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticEnvelopeRejected
        );
        assert_eq!(plan.full_cohorts, 0);
        assert!(plan.refill_configured);

        let mut mixed_limits = make_requests(64, 137);
        mixed_limits[0].request.tokens = Some(32);
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &mixed_limits,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_eq!(plan.ragged_prompt_policy, RaggedPromptPolicy::Disabled);
        assert_eq!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticEnvelopeRejected
        );

        let too_long = make_requests(64, AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS + 1);
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &too_long,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_eq!(plan.ragged_prompt_policy, RaggedPromptPolicy::Disabled);
        assert_eq!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticEnvelopeRejected
        );
        assert_eq!(plan.full_cohorts, 0);

        let mut undersized_chunk = test_args();
        undersized_chunk.execution_mode = Some(execution_selector::ExecutionModeArg::Auto);
        undersized_chunk.prefill_chunk = PrefillChunkArg::Fixed(64);
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &admitted,
            &undersized_chunk,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_eq!(plan.ragged_prompt_policy, RaggedPromptPolicy::Disabled);
        assert_eq!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticEnvelopeRejected
        );

        let moe_requests = (0..MOE_BATCH16_WIDTH)
            .map(|slot| {
                let mut request = prepared(&format!("moe-auto-ragged-{slot}"), &vec![7; 10 + slot]);
                request.request.tokens = Some(64);
                request
            })
            .collect::<Vec<_>>();
        let moe = plan_request_work::<MOE_BATCH16_WIDTH>(&moe_requests, &args).unwrap();
        assert_eq!(moe.ragged_prompt_policy, RaggedPromptPolicy::Disabled);
        assert_eq!(
            moe.ragged_prompt_plan_decision,
            RaggedPromptDecision::Disabled
        );
        assert_eq!(moe.full_cohorts, 0);

        let explicit = plan_request_work_with_policy::<MOE_BATCH16_WIDTH>(
            &moe_requests,
            &args,
            RaggedPromptPolicy::Explicit,
        )
        .unwrap();
        assert_eq!(explicit.ragged_prompt_policy, RaggedPromptPolicy::Explicit);
        assert_eq!(explicit.full_cohorts, 1);
    }

    fn automatic_ragged_refill_requests(
        max_prompt_tokens: usize,
        max_generation_tokens: usize,
    ) -> Vec<PreparedJsonlRequest> {
        let prompt_tokens = [14, 15, 17, 21, 29, 45, 77, max_prompt_tokens];
        let measured_limits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 28, 32, 40];
        measured_limits
            .into_iter()
            .cycle()
            .take(DENSE_BATCH8_WIDTH * 4)
            .enumerate()
            .map(|(index, limit)| {
                let mut request = prepared(
                    &format!("auto-refill-{index}"),
                    &vec![7; prompt_tokens[index % prompt_tokens.len()]],
                );
                request.request.tokens = Some(limit.min(max_generation_tokens));
                request
            })
            .collect()
    }

    #[test]
    fn automatic_dense_ragged_refill_is_bounded_and_charged() {
        let mut args = test_args();
        args.execution_mode = Some(execution_selector::ExecutionModeArg::Auto);
        args.prefill_chunk = PrefillChunkArg::Fixed(AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS);
        let requests = automatic_ragged_refill_requests(
            AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS,
            AUTOMATIC_DENSE_RAGGED_REFILL_MAX_GENERATION_TOKENS,
        );
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_eq!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticRefillAdmitted
        );
        assert_eq!(
            plan.refill_policy,
            DenseRefillPolicy::AutomaticChargedRagged
        );
        assert_eq!(plan.refill_arenas, 2);
        assert_eq!(plan.refill_requests, requests.len());
        assert_eq!(plan.serial_fallback_requests, 0);
        assert!(plan.automatic_refill_fallback.is_some());
        assert!(
            plan.work
                .iter()
                .all(|work| matches!(work, PlannedWork::RefillArena(_)))
        );

        let summary = plan_summary::<DENSE_BATCH8_WIDTH>(&requests, &args).unwrap();
        assert_eq!(
            summary.ragged_prompt_plan_decision,
            Some("automatic_dense_refill_admitted")
        );
        assert_eq!(summary.full_cohorts, 4);
        assert_eq!(summary.serial_fallback_requests, 0);

        let underfilled = automatic_ragged_refill_requests(
            AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS,
            AUTOMATIC_DENSE_RAGGED_REFILL_MAX_GENERATION_TOKENS,
        )
        .into_iter()
        .take(DENSE_BATCH8_WIDTH * 4 - 1)
        .collect::<Vec<_>>();
        assert!(!automatic_dense_ragged_refill_applicable::<
            DENSE_BATCH8_WIDTH,
        >(&underfilled, &args,));
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &underfilled,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_ne!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticRefillAdmitted
        );

        let mut too_long = automatic_ragged_refill_requests(
            AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS,
            AUTOMATIC_DENSE_RAGGED_REFILL_MAX_GENERATION_TOKENS,
        );
        too_long[7].prompt_ids.push(7);
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &too_long,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_eq!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticRefillEnvelopeRejected
        );

        let mut too_much_generation = automatic_ragged_refill_requests(
            AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS,
            AUTOMATIC_DENSE_RAGGED_REFILL_MAX_GENERATION_TOKENS,
        );
        too_much_generation[0].request.tokens =
            Some(AUTOMATIC_DENSE_RAGGED_REFILL_MAX_GENERATION_TOKENS + 1);
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &too_much_generation,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_eq!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticRefillEnvelopeRejected
        );

        let expensive =
            automatic_ragged_refill_requests(AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS, 8);
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &expensive,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_eq!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticRefillChargeRejected
        );
    }

    #[test]
    fn automatic_dense_ragged_refill_schedule_boundaries_are_inclusive() {
        let requests = (0..DENSE_BATCH8_WIDTH * 2)
            .map(|index| prepared(&format!("arena-boundary-{index}"), &[7; 18]))
            .collect::<Vec<_>>();
        let arena = |simulated_physical_steps, simulated_productive_transitions, two_wave| {
            RefillArenaPlan::<DENSE_BATCH8_WIDTH> {
                fallback_work: Vec::new(),
                request_indices: (0..DENSE_BATCH8_WIDTH * 2).collect(),
                shared_capacity: 64,
                simulated_physical_steps,
                simulated_productive_transitions,
                requested_two_wave_steps: two_wave,
            }
        };
        assert!(
            automatic_dense_ragged_refill_arena_schedule_qualifies(&arena(10, 72, 12), &requests,)
                .unwrap()
        );
        assert!(
            !automatic_dense_ragged_refill_arena_schedule_qualifies(&arena(10, 71, 12), &requests,)
                .unwrap()
        );
        assert!(
            automatic_dense_ragged_refill_arena_schedule_qualifies(&arena(17, 136, 20), &requests,)
                .unwrap()
        );
        assert!(
            !automatic_dense_ragged_refill_arena_schedule_qualifies(
                &arena(18, 144, 20),
                &requests,
            )
            .unwrap()
        );

        let maximum_arena_charge = (0..DENSE_BATCH8_WIDTH * 2)
            .map(|index| prepared(&format!("arena-charge-{index}"), &[7; 32]))
            .collect::<Vec<_>>();
        assert!(
            automatic_dense_ragged_refill_arena_schedule_qualifies(
                &arena(2, 16, 4),
                &maximum_arena_charge,
            )
            .unwrap()
        );
        let above_arena_charge = (0..DENSE_BATCH8_WIDTH * 2)
            .map(|index| {
                let length = if index == 0 { 33 } else { 32 };
                prepared(&format!("arena-charge-over-{index}"), &vec![7; length])
            })
            .collect::<Vec<_>>();
        assert!(
            !automatic_dense_ragged_refill_arena_schedule_qualifies(
                &arena(2, 16, 4),
                &above_arena_charge,
            )
            .unwrap()
        );

        assert!(automatic_dense_ragged_refill_file_charge_qualifies(90, 10).unwrap());
        assert!(!automatic_dense_ragged_refill_file_charge_qualifies(91, 10).unwrap());
        assert!(!automatic_dense_ragged_refill_file_charge_qualifies(0, 0).unwrap());
    }

    #[test]
    fn automatic_dense_ragged_refill_memory_denial_restores_planner_baseline() {
        let mut args = test_args();
        args.execution_mode = Some(execution_selector::ExecutionModeArg::Auto);
        args.prefill_chunk = PrefillChunkArg::Fixed(AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS);
        let requests = automatic_ragged_refill_requests(
            AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS,
            AUTOMATIC_DENSE_RAGGED_REFILL_MAX_GENERATION_TOKENS,
        );
        let candidate = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        let baseline = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            true,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            RaggedPromptPolicy::Disabled,
            DenseRefillPolicy::ShortSerialFallbackRescue,
        )
        .unwrap();
        let mut admission_calls = 0usize;
        let (fallback, admission) = admit_automatic_refill_plan(candidate, |_| {
            admission_calls += 1;
            Ok(admission_calls == 1)
        })
        .unwrap();
        assert_eq!(admission_calls, 2);
        assert_eq!(
            admission,
            AutomaticRefillAdmission::PlannerBaselineFallback { rejected_arenas: 2 }
        );
        assert_eq!(fallback.work, baseline.work);
        assert_eq!(fallback.refill_arenas, 2);
        assert_eq!(fallback.refill_requests, requests.len());
        assert_eq!(
            fallback.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticRefillAdmitted
        );
        assert!(fallback.automatic_refill_fallback.is_none());
    }

    #[test]
    fn automatic_dense_ragged_charges_every_proposed_cohort() {
        let mut args = test_args();
        args.execution_mode = Some(execution_selector::ExecutionModeArg::Auto);
        args.prefill_chunk = PrefillChunkArg::Fixed(512);
        let mut requests = (0..DENSE_BATCH8_WIDTH)
            .map(|slot| {
                let mut request = prepared(&format!("cheap-{slot}"), &vec![7; 8 + slot]);
                request.request.tokens = Some(32);
                request
            })
            .collect::<Vec<_>>();
        requests.extend((0..DENSE_BATCH8_WIDTH).map(|slot| {
            let mut request = prepared(&format!("expensive-{slot}"), &vec![7; 249 + slot]);
            request.request.tokens = Some(32);
            request
        }));
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_eq!(plan.ragged_prompt_policy, RaggedPromptPolicy::Disabled);
        assert_eq!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticChargeRejected
        );
        assert_eq!(plan.full_cohorts, 0);
        assert_eq!(plan.serial_fallback_requests, requests.len());
    }

    #[test]
    fn automatic_dense_ragged_does_not_strand_b2_remainders() {
        let mut args = test_args();
        args.execution_mode = Some(execution_selector::ExecutionModeArg::Auto);
        args.prefill_chunk = PrefillChunkArg::Fixed(512);
        let requests = (0..DENSE_BATCH8_WIDTH + 1)
            .map(|slot| {
                let mut request =
                    prepared(&format!("ragged-remainder-{slot}"), &vec![7; 10 + slot]);
                request.request.tokens = Some(64);
                request
            })
            .collect::<Vec<_>>();
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_eq!(plan.ragged_prompt_policy, RaggedPromptPolicy::Disabled);
        assert_eq!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticNoCohortGain
        );
        assert_eq!(plan.full_cohorts, 0);
        assert_eq!(plan.serial_fallback_requests, requests.len());
    }

    #[test]
    fn automatic_dense_ragged_never_replaces_an_equally_full_incumbent() {
        let mut args = test_args();
        args.execution_mode = Some(execution_selector::ExecutionModeArg::Auto);
        args.prefill_chunk = PrefillChunkArg::Fixed(512);
        let requests = (0..DENSE_BATCH8_WIDTH)
            .map(|slot| {
                let mut request = prepared(&format!("equal-{slot}"), &[7; 32]);
                request.request.tokens = Some(64);
                request
            })
            .collect::<Vec<_>>();
        let plan = plan_request_work_with_policy::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            RaggedPromptPolicy::AutomaticDenseCharged,
        )
        .unwrap();
        assert_eq!(plan.ragged_prompt_policy, RaggedPromptPolicy::Disabled);
        assert_eq!(
            plan.ragged_prompt_plan_decision,
            RaggedPromptDecision::AutomaticNoCohortGain
        );
        assert_eq!(plan.full_cohorts, 1);
        assert!(plan.refill_configured);
    }

    #[test]
    fn ragged_capacity_order_is_only_selected_when_it_is_pareto_safe() {
        let mut args = test_args();
        args.max_context_tokens = None;
        let mut requests = Vec::new();
        for group in 0..2 {
            for slot in 0..DENSE_BATCH8_WIDTH {
                let prompt_len = if slot % 2 == 0 {
                    8 + group
                } else {
                    4_000 + group
                };
                let mut request =
                    prepared(&format!("group-{group}-slot-{slot}"), &vec![7; prompt_len]);
                request.request.tokens = Some(if group == 0 { 32 } else { 96 });
                requests.push(request);
            }
        }
        let plan = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            true,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            true,
            false,
        )
        .unwrap();
        let generation = requests
            .iter()
            .map(|request| jsonl_generation_capacity(request, &args).unwrap())
            .collect::<Vec<_>>();
        let requested = generation
            .iter()
            .map(|&(tokens, _)| tokens)
            .collect::<Vec<_>>();
        let capacities = generation.iter().map(|&(_, cap)| cap).collect::<Vec<_>>();
        let depth = plan_depth_bucket::<DENSE_BATCH8_WIDTH>(
            (0..requests.len()).collect(),
            &requested,
            &capacities,
        )
        .unwrap();
        assert_eq!(plan.full_cohorts, depth.full_cohorts);
        assert!(estimate_capacity_slots(&plan.work, &capacities).unwrap() <= depth.capacity_slots);
        assert!(
            estimated_transition_slots(&plan.work, &requested).unwrap()
                <= depth.physical_transition_slots
        );
    }

    #[test]
    fn memory_denial_rewrites_only_the_rejected_batch_to_serial() {
        let first = PlannedWork::Batch([0, 1, 2, 3, 4, 5, 6, 7]);
        let second = PlannedWork::Batch([8, 9, 10, 11, 12, 13, 14, 15]);
        let (rewritten, rejected) =
            rewrite_unadmitted_batches(vec![first, second, PlannedWork::Serial(16)], |indices| {
                Ok(indices[0] != 0)
            })
            .unwrap();
        assert_eq!(rejected, vec![[0, 1, 2, 3, 4, 5, 6, 7]]);
        assert!(
            rewritten[..8]
                .iter()
                .all(|work| matches!(work, PlannedWork::Serial(_)))
        );
        assert!(matches!(rewritten[8], PlannedWork::Batch(indices) if indices[0] == 8));
        assert_eq!(rewritten[9], PlannedWork::Serial(16));
    }

    #[test]
    fn ragged_prefix_policy_uses_the_longest_lane_for_tiny_suffix_admission() {
        let shared = vec![7; 513];
        let requests = (0..DENSE_BATCH8_WIDTH)
            .map(|slot| {
                let mut prompt = shared.clone();
                prompt.extend(std::iter::repeat_n(1_000 + slot as i32, slot + 1));
                prepared(&format!("slot-{slot}"), &prompt)
            })
            .collect::<Vec<_>>();
        let refs = requests.iter().collect::<Vec<_>>();
        let max_prompt_tokens = requests
            .iter()
            .map(|request| request.prompt_ids.len())
            .max()
            .unwrap();
        assert_eq!(common_prefix_tokens(&refs), shared.len());
        assert_eq!(
            align_prefix_fanout(
                plan_prefix_fanout(&refs, true),
                max_prompt_tokens,
                512,
                QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp,
            ),
            PrefixFanoutPlan {
                common_prefix_tokens: 513,
                selected_prefix_tokens: 512,
                reason: "selected_chunk_aligned",
            }
        );
    }

    #[test]
    fn fixed_file_root_policy_is_strict_and_family_defaulted() {
        assert!(fixed_file_root_default_enabled::<DENSE_BATCH8_WIDTH>());
        assert!(fixed_file_root_default_enabled::<MOE_BATCH16_WIDTH>());
        assert!(
            qwen_file_root::plan(
                &[&[7; 1_500], &[7; 1_400]],
                PrefillChunkArg::Fixed(512),
                2,
                true,
            )
            .selected_prefix_tokens
                == 1_024
        );
    }

    #[test]
    fn fixed_file_root_participants_exclude_serial_and_refill_work() {
        let batch = [0, 1, 2, 3, 4, 5, 6, 7];
        let refill = RefillArenaPlan {
            fallback_work: Vec::new(),
            request_indices: (8..24).collect(),
            shared_capacity: 64,
            simulated_physical_steps: 8,
            simulated_productive_transitions: 64,
            requested_two_wave_steps: 16,
        };
        let work = [
            PlannedWork::Batch(batch),
            PlannedWork::RefillArena(refill),
            PlannedWork::Serial(24),
        ];
        let participants = fixed_file_root_request_indices(&work);
        assert_eq!(participants, batch);
    }

    #[test]
    fn dense_refill_policy_defaults_to_short_serial_rescue_and_moe_stays_closed() {
        assert_eq!(
            parse_dense_refill_policy(None, false, false).unwrap(),
            DenseRefillPolicy::ShortSerialFallbackRescue
        );
        assert_eq!(
            parse_dense_refill_policy(None, true, false).unwrap(),
            DenseRefillPolicy::Disabled
        );
        assert_eq!(
            parse_dense_refill_policy(Some("on"), false, false).unwrap(),
            DenseRefillPolicy::Forced
        );
        assert_eq!(
            parse_dense_refill_policy(Some("0"), false, false).unwrap(),
            DenseRefillPolicy::Disabled
        );
        assert!(parse_dense_refill_policy(Some("sometimes"), false, false).is_err());

        let args = test_args();
        let limits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 28, 32, 40]
            .into_iter()
            .cycle()
            .take(32)
            .collect::<Vec<_>>();
        let requests = limits
            .iter()
            .enumerate()
            .map(|(index, &limit)| {
                let mut request = prepared(&format!("slot-{index}"), &[1, 2]);
                request.request.tokens = Some(limit);
                request
            })
            .collect::<Vec<_>>();
        let dense = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            true,
        )
        .unwrap();
        assert!(dense.refill_effective);
        assert_eq!(dense.refill_arenas, 2);
        assert_eq!(dense.refill_requests, 32);
        let PlannedWork::RefillArena(arena) = &dense.work[0] else {
            panic!("expected one dense refill arena");
        };
        assert_eq!(arena.request_indices.len(), 16);
        assert!(arena.simulated_physical_steps * 10 <= arena.requested_two_wave_steps * 9);
        assert_eq!(arena.request_indices[0], 7);

        let rescue = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            DenseRefillPolicy::ShortSerialFallbackRescue,
        )
        .unwrap();
        assert!(rescue.refill_effective);
        assert_eq!(rescue.refill_arenas, 2);
        assert_eq!(rescue.refill_requests, 32);

        let moe = plan_request_work_configured::<MOE_BATCH16_WIDTH>(
            &requests,
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            true,
        )
        .unwrap();
        assert!(!moe.refill_effective);
        assert_eq!(moe.refill_arenas, 0);
    }

    fn planner_telemetry_for_policy(
        policy: Option<DenseRefillPolicy>,
        refill_telemetry: bool,
    ) -> PlannerTelemetry {
        let policy = refill_telemetry.then_some(policy).flatten();
        PlannerTelemetry {
            schema_version: if refill_telemetry { 9 } else { 6 },
            backend: if refill_telemetry {
                "dense_qwen_fixed_cohort_planner_v9"
            } else {
                "qwen_moe_fixed_cohort_planner_v6"
            },
            requests: 16,
            prefix_packing_configured: true,
            prefix_packing_effective: true,
            prefix_fanout_boundary_policy: "tiny_suffix_exact_lcp",
            ragged_prompts_enabled: false,
            ragged_prompt_policy: "disabled",
            ragged_prompt_plan_decision: "disabled",
            ragged_packing_strategy: "disabled",
            refill_policy: policy.map(DenseRefillPolicy::as_str),
            refill_default_max_prompt_tokens: policy
                .and_then(DenseRefillPolicy::default_max_prompt_tokens),
            refill_configured: policy.map(DenseRefillPolicy::enabled),
            refill_effective: refill_telemetry.then_some(false),
            planned_refill_arenas: refill_telemetry.then_some(0),
            planned_refill_requests: refill_telemetry.then_some(0),
            realized_refill_arenas: refill_telemetry.then_some(0),
            realized_refill_requests: refill_telemetry.then_some(0),
            realized_refill_memory_denied_arenas: refill_telemetry.then_some(0),
            realized_refill_memory_fallback_arenas: refill_telemetry.then_some(0),
            automatic_refill_transaction_outcome: None,
            automatic_refill_min_arenas: refill_telemetry
                .then_some(AUTOMATIC_DENSE_RAGGED_REFILL_MIN_ARENAS),
            automatic_refill_max_prompt_tokens: refill_telemetry
                .then_some(AUTOMATIC_DENSE_RAGGED_MAX_PROMPT_TOKENS),
            automatic_refill_max_generation_tokens: refill_telemetry
                .then_some(AUTOMATIC_DENSE_RAGGED_REFILL_MAX_GENERATION_TOKENS),
            automatic_refill_max_file_prefill_per_transition: refill_telemetry
                .then_some(AUTOMATIC_DENSE_RAGGED_REFILL_MAX_PREFILL_PER_TRANSITION),
            automatic_refill_max_arena_prefill_per_transition: refill_telemetry
                .then_some(AUTOMATIC_DENSE_RAGGED_REFILL_MAX_ARENA_PREFILL_PER_TRANSITION),
            automatic_refill_min_transition_utilization: refill_telemetry.then_some("9/10"),
            automatic_refill_min_idealized_step_savings: refill_telemetry.then_some("3/20"),
            compatibility_buckets: 1,
            planned_candidate_cohorts: 2,
            realized_prefix_affinity_cohorts: 0,
            planned_prefix_fallback_buckets: 0,
            planned_full_cohorts: 2,
            planned_batched_requests: 16,
            realized_full_cohorts: 2,
            planned_economics_rejected_cohorts: 0,
            realized_memory_rejected_cohorts: 0,
            realized_batched_requests: 16,
            realized_serial_fallback_requests: 0,
            realized_estimated_physical_transition_slots: 160,
            realized_estimated_capacity_slots: 608,
            minimum_requested_transition_utilization: "3/4",
            packing_order: "generation_depth",
            output_order: "input",
        }
    }

    #[test]
    fn refill_policy_telemetry_is_closed_and_family_scoped() {
        let bounded = serde_json::to_value(planner_telemetry_for_policy(
            Some(DenseRefillPolicy::ShortSerialFallbackRescue),
            true,
        ))
        .unwrap();
        assert_eq!(bounded["schema_version"], 9);
        assert_eq!(bounded["refill_policy"], "short_serial_fallback_rescue");
        assert_eq!(bounded["refill_default_max_prompt_tokens"], 64);

        let forced = serde_json::to_value(planner_telemetry_for_policy(
            Some(DenseRefillPolicy::Forced),
            true,
        ))
        .unwrap();
        assert_eq!(forced["refill_policy"], "forced");
        assert!(forced.get("refill_default_max_prompt_tokens").is_none());

        let disabled = serde_json::to_value(planner_telemetry_for_policy(
            Some(DenseRefillPolicy::Disabled),
            true,
        ))
        .unwrap();
        assert_eq!(disabled["refill_policy"], "disabled");
        assert!(disabled.get("refill_default_max_prompt_tokens").is_none());

        let moe = serde_json::to_value(planner_telemetry_for_policy(None, false)).unwrap();
        for key in [
            "refill_policy",
            "refill_default_max_prompt_tokens",
            "refill_configured",
            "refill_effective",
            "planned_refill_arenas",
            "planned_refill_requests",
            "realized_refill_arenas",
            "realized_refill_requests",
            "realized_refill_memory_denied_arenas",
            "realized_refill_memory_fallback_arenas",
            "automatic_refill_transaction_outcome",
            "automatic_refill_min_arenas",
            "automatic_refill_max_prompt_tokens",
            "automatic_refill_max_generation_tokens",
            "automatic_refill_max_file_prefill_per_transition",
            "automatic_refill_max_arena_prefill_per_transition",
            "automatic_refill_min_transition_utilization",
            "automatic_refill_min_idealized_step_savings",
        ] {
            assert!(moe.get(key).is_none(), "MoE telemetry retained {key}");
        }
    }

    #[test]
    fn automatic_refill_never_expands_to_forced_policy() {
        assert_eq!(
            parse_dense_refill_policy(None, false, true).unwrap(),
            DenseRefillPolicy::ShortSerialFallbackRescue
        );
        assert_eq!(
            parse_dense_refill_policy(Some("1"), false, true).unwrap(),
            DenseRefillPolicy::ShortSerialFallbackRescue
        );
        assert_eq!(
            parse_dense_refill_policy(Some("1"), true, true).unwrap(),
            DenseRefillPolicy::Disabled
        );
        assert_eq!(
            parse_dense_refill_policy(Some("0"), false, true).unwrap(),
            DenseRefillPolicy::Disabled
        );
        assert_eq!(
            parse_dense_refill_policy(Some("1"), false, false).unwrap(),
            DenseRefillPolicy::Forced
        );
    }

    #[test]
    fn automatic_dense_execution_uses_the_same_bounded_refill_policy() {
        let mut args = test_args();
        args.execution_mode = Some(execution_selector::ExecutionModeArg::Auto);
        let limits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 28, 32, 40]
            .into_iter()
            .cycle()
            .take(32)
            .collect::<Vec<_>>();
        let requests = limits
            .iter()
            .enumerate()
            .map(|(index, &limit)| {
                let mut request = prepared(&format!("auto-{index}"), &[1, 2]);
                request.request.tokens = Some(limit);
                request
            })
            .collect::<Vec<_>>();
        let plan = plan_request_work::<DENSE_BATCH8_WIDTH>(&requests, &args).unwrap();
        assert_eq!(
            plan.refill_policy,
            DenseRefillPolicy::ShortSerialFallbackRescue
        );
        assert!(plan.refill_configured);
        assert!(plan.refill_effective);
        assert_eq!(plan.refill_arenas, 2);
        assert_eq!(plan.refill_requests, 32);

        let summary = plan_summary::<DENSE_BATCH8_WIDTH>(&requests, &args).unwrap();
        assert_eq!(summary.full_cohorts, 4);
        assert_eq!(summary.serial_fallback_requests, 0);
        assert_eq!(summary.max_execution_capacity, 349);
        assert_eq!(
            fixed_cohort_refill_policy::<MOE_BATCH16_WIDTH>(false, true).unwrap(),
            DenseRefillPolicy::Disabled
        );
    }

    #[test]
    fn default_refill_rescues_only_at_the_measured_prompt_boundary() {
        let args = test_args();
        let limits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 28, 32, 40]
            .into_iter()
            .cycle()
            .take(32)
            .collect::<Vec<_>>();
        let make_requests = |prompt_tokens: usize| {
            limits
                .iter()
                .enumerate()
                .map(|(index, &limit)| {
                    let mut request = prepared(
                        &format!("boundary-{prompt_tokens}-{index}"),
                        &vec![1; prompt_tokens],
                    );
                    request.request.tokens = Some(limit);
                    request
                })
                .collect::<Vec<_>>()
        };

        let boundary = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &make_requests(DEFAULT_REFILL_MAX_PROMPT_TOKENS),
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            DenseRefillPolicy::ShortSerialFallbackRescue,
        )
        .unwrap();
        assert!(boundary.refill_effective);
        assert_eq!(boundary.refill_arenas, 2);

        let above = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &make_requests(DEFAULT_REFILL_MAX_PROMPT_TOKENS + 1),
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            DenseRefillPolicy::ShortSerialFallbackRescue,
        )
        .unwrap();
        assert!(!above.refill_effective);
        assert_eq!(above.full_cohorts, 2);
        assert_eq!(above.serial_fallback_requests, 16);

        let forced = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &make_requests(DEFAULT_REFILL_MAX_PROMPT_TOKENS + 1),
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            DenseRefillPolicy::Forced,
        )
        .unwrap();
        assert!(forced.refill_effective);
    }

    #[test]
    fn default_refill_leaves_fully_batched_threshold_work_static() {
        let args = test_args();
        let limits = [3, 3, 3, 3, 3, 3, 3, 3, 12, 13, 14, 15, 16, 17, 18, 19];
        let requests = limits
            .iter()
            .enumerate()
            .map(|(index, &limit)| {
                let mut request = prepared(&format!("threshold-{index}"), &[1, 2]);
                request.request.tokens = Some(limit);
                request
            })
            .collect::<Vec<_>>();
        let default = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            DenseRefillPolicy::ShortSerialFallbackRescue,
        )
        .unwrap();
        assert!(!default.refill_effective);
        assert_eq!(default.full_cohorts, 2);
        assert_eq!(default.serial_fallback_requests, 0);

        let forced = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            DenseRefillPolicy::Forced,
        )
        .unwrap();
        assert!(forced.refill_effective);
        assert_eq!(forced.refill_arenas, 1);
    }

    #[test]
    fn explicit_context_limit_rejects_refill_padding_expansion() {
        let mut args = test_args();
        args.max_context_tokens = Some(64);
        let limits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 28, 32, 40]
            .into_iter()
            .cycle()
            .take(32)
            .collect::<Vec<_>>();
        let requests = limits
            .iter()
            .enumerate()
            .map(|(index, &limit)| {
                let mut request = prepared(&format!("slot-{index}"), &[1, 2]);
                request.request.tokens = Some(limit);
                request
            })
            .collect::<Vec<_>>();
        let plan = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            true,
        )
        .unwrap();
        assert!(plan.refill_effective);
        assert_eq!(plan.refill_arenas, 1);
        let arenas = plan
            .work
            .iter()
            .filter_map(|work| match work {
                PlannedWork::RefillArena(arena) => Some(arena),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(arenas.len(), 1);
        assert!(arenas[0].shared_capacity <= 64);
        assert!(
            plan.work
                .iter()
                .any(|work| matches!(work, PlannedWork::Serial(_)))
        );
    }

    #[test]
    fn refill_admission_denial_restores_the_exact_static_plan() {
        let args = test_args();
        let limits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 28, 32, 40]
            .into_iter()
            .cycle()
            .take(32)
            .collect::<Vec<_>>();
        let requests = limits
            .iter()
            .enumerate()
            .map(|(index, &limit)| {
                let mut request = prepared(&format!("slot-{index}"), &[1, 2]);
                request.request.tokens = Some(limit);
                request
            })
            .collect::<Vec<_>>();
        let static_plan = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            false,
        )
        .unwrap();
        let refill_plan = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            true,
        )
        .unwrap();
        let (rewritten, rejected) =
            rewrite_unadmitted_refill_arenas(refill_plan.work, |_| Ok(false)).unwrap();
        assert_eq!(rejected, 2);
        assert_eq!(rewritten, static_plan.work);
    }

    #[test]
    fn ragged_prompt_policy_is_strict_and_rollbackable() {
        assert_eq!(
            parse_ragged_prompt_policy(None, RaggedPromptPolicy::AutomaticDenseCharged).unwrap(),
            RaggedPromptPolicy::AutomaticDenseCharged
        );
        assert_eq!(
            parse_ragged_prompt_policy(Some("on"), RaggedPromptPolicy::Disabled).unwrap(),
            RaggedPromptPolicy::Explicit
        );
        assert_eq!(
            parse_ragged_prompt_policy(Some("NO"), RaggedPromptPolicy::AutomaticDenseCharged,)
                .unwrap(),
            RaggedPromptPolicy::Disabled
        );
        assert!(
            parse_ragged_prompt_policy(Some("sometimes"), RaggedPromptPolicy::Disabled).is_err()
        );
    }

    #[test]
    fn ragged_generation_plan_uses_per_lane_lengths_and_shared_capacity() {
        let mut args = test_args();
        args.max_context_tokens = Some(64);
        let requests = (0..DENSE_BATCH8_WIDTH)
            .map(|slot| prepared(&format!("slot-{slot}"), &vec![7; slot + 1]))
            .collect::<Vec<_>>();
        let refs: [&PreparedJsonlRequest; DENSE_BATCH8_WIDTH] =
            std::array::from_fn(|slot| &requests[slot]);
        let plan = cohort_generation_plan(&refs, &args).unwrap();
        assert_eq!(plan.prompt_tokens_per_lane, (1..=8).collect::<Vec<_>>());
        assert_eq!(plan.min_prompt_tokens, 1);
        assert_eq!(plan.max_prompt_tokens, 8);
        assert_eq!(plan.shared_capacity, 64);
    }

    #[test]
    fn prefix_packing_recovers_interleaved_dense_cohorts() {
        let mut args = test_args();
        args.prefill_chunk = PrefillChunkArg::Fixed(256);
        let requests = (0..(DENSE_BATCH8_WIDTH * 2))
            .map(|slot| {
                let mut prompt = vec![if slot.is_multiple_of(2) { 1 } else { 2 }; 1_024];
                prompt.push(i32::try_from(slot).unwrap() + 10);
                let mut request = prepared(&format!("slot-{slot}"), &prompt);
                request.request.tokens = Some(32);
                request
            })
            .collect::<Vec<_>>();

        let control = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            false,
        )
        .unwrap();
        assert_eq!(control.full_cohorts, 2);
        assert_eq!(control.prefix_affinity_batches.len(), 0);

        let candidate = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            true,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            false,
        )
        .unwrap();
        assert_eq!(candidate.full_cohorts, 2);
        assert_eq!(candidate.prefix_affinity_batches.len(), 2);
        assert_eq!(
            candidate.work,
            vec![
                PlannedWork::Batch([0, 2, 4, 6, 8, 10, 12, 14]),
                PlannedWork::Batch([1, 3, 5, 7, 9, 11, 13, 15]),
            ]
        );
    }

    #[test]
    fn prefix_packing_policy_is_strict_and_rollbackable() {
        assert!(!parse_prefix_packing_enabled(None, false).unwrap());
        assert!(parse_prefix_packing_enabled(None, true).unwrap());
        assert!(parse_prefix_packing_enabled(Some("on"), false).unwrap());
        assert!(!parse_prefix_packing_enabled(Some("NO"), true).unwrap());
        assert!(parse_prefix_packing_enabled(Some("sometimes"), true).is_err());
    }

    #[test]
    fn prefix_packing_preserves_depth_groups_without_reusable_prefixes() {
        let args = test_args();
        let requests = (0..(DENSE_BATCH8_WIDTH * 2))
            .map(|slot| {
                let mut prompt = vec![i32::try_from(slot).unwrap() + 1; 32];
                prompt.push(99);
                let mut request = prepared(&format!("slot-{slot}"), &prompt);
                request.request.tokens = Some(24 + slot);
                request
            })
            .collect::<Vec<_>>();
        let control = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            false,
        )
        .unwrap();
        let candidate = plan_request_work_configured::<DENSE_BATCH8_WIDTH>(
            &requests,
            &args,
            true,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            false,
        )
        .unwrap();
        assert_eq!(candidate.work, control.work);
        assert_eq!(candidate.full_cohorts, control.full_cohorts);
        assert_eq!(candidate.economics_rejected_cohorts, 0);
        assert_eq!(candidate.prefix_affinity_batches.len(), 0);
    }

    fn assert_prefix_fragmentation_falls_back<const WIDTH: usize>() {
        let mut args = test_args();
        args.prefill_chunk = PrefillChunkArg::Fixed(256);
        let mut before_prefix = 1i32;
        let mut after_prefix = 200i32;
        let prefix_low = WIDTH / 4;
        let prefix_high_end = WIDTH * 2 + WIDTH * 3 / 4;
        let before_prefix_end = WIDTH + WIDTH / 4;
        let requests = (0..(WIDTH * 3))
            .map(|slot| {
                let prefix_family =
                    slot < prefix_low || (WIDTH * 2..prefix_high_end).contains(&slot);
                let first = if prefix_family {
                    100
                } else if (prefix_low..before_prefix_end).contains(&slot) {
                    let value = before_prefix;
                    before_prefix += 1;
                    value
                } else {
                    let value = after_prefix;
                    after_prefix += 1;
                    value
                };
                let mut prompt = vec![first; 300];
                prompt.push(i32::try_from(slot).unwrap() + 1_000);
                let mut request = prepared(&format!("slot-{slot}"), &prompt);
                request.request.tokens = Some(if slot < WIDTH {
                    2
                } else if slot < WIDTH * 2 {
                    5
                } else {
                    11
                });
                request
            })
            .collect::<Vec<_>>();
        let baseline = plan_request_work_configured::<WIDTH>(
            &requests,
            &args,
            false,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            false,
        )
        .unwrap();
        let candidate = plan_request_work_configured::<WIDTH>(
            &requests,
            &args,
            true,
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            false,
            false,
        )
        .unwrap();
        assert_eq!(baseline.full_cohorts, 3);
        assert_eq!(candidate.work, baseline.work);
        assert_eq!(candidate.full_cohorts, 3);
        assert_eq!(candidate.prefix_affinity_batches.len(), 0);
        assert_eq!(candidate.prefix_plan_fallback_buckets, 1);
        let requested_tokens = requests
            .iter()
            .map(|request| jsonl_generation_capacity(request, &args).unwrap().0)
            .collect::<Vec<_>>();
        assert_eq!(
            estimated_transition_slots(&candidate.work, &requested_tokens).unwrap(),
            estimated_transition_slots(&baseline.work, &requested_tokens).unwrap()
        );
    }

    #[test]
    fn dense_prefix_packing_falls_back_when_it_fragments_depth_cohorts() {
        assert_prefix_fragmentation_falls_back::<DENSE_BATCH8_WIDTH>();
    }

    #[test]
    fn moe_prefix_packing_falls_back_when_it_fragments_depth_cohorts() {
        assert_prefix_fragmentation_falls_back::<MOE_BATCH16_WIDTH>();
    }

    #[test]
    fn planner_forms_exact_sixteen_way_cohorts_and_serializes_remainders() {
        let mut args = test_args();
        args.batch_size = Some(MOE_BATCH16_WIDTH);
        let mut requests = (0..MOE_BATCH16_WIDTH)
            .map(|slot| {
                let mut request = prepared(&format!("full-{slot}"), &[1, 2, 3]);
                request.request.tokens = Some(24 + slot);
                request
            })
            .collect::<Vec<_>>();
        let mut heterogeneous = prepared("heterogeneous", &[1, 2, 3, 4]);
        heterogeneous.request.tokens = Some(7);
        requests.push(heterogeneous);
        let mut underfill = prepared("underfill", &[1, 2, 3]);
        underfill.request.tokens = Some(40);
        requests.push(underfill);

        let plan = plan_request_work::<MOE_BATCH16_WIDTH>(&requests, &args).unwrap();
        assert_eq!(plan.compatibility_buckets, 2);
        assert_eq!(plan.candidate_cohorts, 1);
        assert_eq!(plan.full_cohorts, 1);
        assert_eq!(plan.economics_rejected_cohorts, 0);
        assert_eq!(plan.serial_fallback_requests, 2);
        assert_eq!(
            plan.work,
            vec![
                PlannedWork::Batch([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                PlannedWork::Serial(16),
                PlannedWork::Serial(17),
            ]
        );

        let underfill = plan_request_work::<MOE_BATCH16_WIDTH>(&requests[..15], &args).unwrap();
        assert_eq!(underfill.full_cohorts, 0);
        assert_eq!(underfill.serial_fallback_requests, 15);
    }

    #[test]
    fn planner_rejects_padding_dominated_mixed_limits() {
        let args = test_args();
        let limits = [1, 2, 3, 4, 5, 6, 7, 40];
        let requests = limits
            .iter()
            .enumerate()
            .map(|(slot, &limit)| {
                let mut request = prepared(&format!("slot-{slot}"), &[1, 2]);
                request.request.tokens = Some(limit);
                request
            })
            .collect::<Vec<_>>();
        let plan = plan_request_work::<DENSE_BATCH8_WIDTH>(&requests, &args).unwrap();
        assert_eq!(plan.compatibility_buckets, 1);
        assert_eq!(plan.candidate_cohorts, 1);
        assert_eq!(plan.full_cohorts, 0);
        assert_eq!(plan.economics_rejected_cohorts, 1);
        assert_eq!(plan.serial_fallback_requests, DENSE_BATCH8_WIDTH);
        assert!(
            plan.work
                .iter()
                .all(|work| matches!(work, PlannedWork::Serial(_)))
        );
    }

    #[test]
    fn requested_transition_utilization_floor_is_inclusive() {
        let at_floor = [3, 3, 4, 4, 4, 4, 5, 5];
        let below_floor = [2, 3, 4, 4, 4, 4, 5, 5];
        let indices: [usize; DENSE_BATCH8_WIDTH] = std::array::from_fn(|index| index);
        assert!(requested_transition_utilization_qualifies(&indices, &at_floor).unwrap());
        assert!(!requested_transition_utilization_qualifies(&indices, &below_floor).unwrap());
    }

    #[test]
    fn cohort_capacity_uses_the_longest_limit_independent_of_lane_order() {
        let args = test_args();
        let limits = [1, 2, 3, 4, 5, 6, 7, 20];
        let requests = limits
            .iter()
            .enumerate()
            .map(|(slot, &limit)| {
                let mut request = prepared(&format!("slot-{slot}"), &[1, 2]);
                request.request.tokens = Some(limit);
                request
            })
            .collect::<Vec<_>>();
        let refs: [&PreparedJsonlRequest; DENSE_BATCH8_WIDTH] =
            std::array::from_fn(|slot| &requests[slot]);
        let plan = cohort_generation_plan(&refs, &args).unwrap();
        assert_eq!(plan.prompt_tokens_per_lane, vec![2; DENSE_BATCH8_WIDTH]);
        assert_eq!(plan.min_prompt_tokens, 2);
        assert_eq!(plan.max_prompt_tokens, 2);
        assert_eq!(plan.requested_tokens, limits);
        assert_eq!(plan.shared_capacity, 38);

        let reversed: [&PreparedJsonlRequest; DENSE_BATCH8_WIDTH] =
            std::array::from_fn(|slot| &requests[DENSE_BATCH8_WIDTH - slot - 1]);
        let reversed_plan = cohort_generation_plan(&reversed, &args).unwrap();
        assert_eq!(reversed_plan.shared_capacity, plan.shared_capacity);
        assert_eq!(
            reversed_plan.requested_tokens,
            limits.into_iter().rev().collect::<Vec<_>>()
        );
    }

    #[test]
    fn prefix_fanout_requires_and_aligns_a_shared_minimum_prefix() {
        assert!(parse_prefix_fanout_enabled(None));
        assert!(parse_prefix_fanout_enabled(Some(OsStr::new("1"))));
        assert!(!parse_prefix_fanout_enabled(Some(OsStr::new("off"))));
        assert_eq!(DENSE_PREFIX_FANOUT_ENV, "QWEN_DENSE_BATCH8_PREFIX_FANOUT");
        assert_eq!(MOE_PREFIX_FANOUT_ENV, "QWEN_MOE_BATCH16_PREFIX_FANOUT");
        let shared = (0..PREFIX_FANOUT_MIN_TOKENS as i32).collect::<Vec<_>>();
        let requests = (0..DENSE_BATCH8_WIDTH)
            .map(|slot| {
                let mut tokens = shared.clone();
                tokens.push(10_000 + slot as i32);
                prepared(&format!("slot-{slot}"), &tokens)
            })
            .collect::<Vec<_>>();
        let request_refs = requests.iter().collect::<Vec<_>>();
        assert_eq!(
            common_prefix_tokens(&request_refs),
            PREFIX_FANOUT_MIN_TOKENS
        );
        assert_eq!(
            plan_prefix_fanout(&request_refs, true),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                selected_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                reason: "selected",
            }
        );
        assert_eq!(
            plan_prefix_fanout(&request_refs, false),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                selected_prefix_tokens: 0,
                reason: "disabled",
            }
        );
        assert_eq!(
            align_prefix_fanout(
                plan_prefix_fanout(&request_refs, true),
                257,
                128,
                QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            ),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                selected_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                reason: "selected",
            }
        );

        let wider = requests
            .iter()
            .enumerate()
            .map(|(slot, request)| {
                let mut tokens = request.prompt_ids.clone();
                tokens.splice(
                    PREFIX_FANOUT_MIN_TOKENS..PREFIX_FANOUT_MIN_TOKENS,
                    [8, 9, 10, 11],
                );
                prepared(&format!("wider-{slot}"), &tokens)
            })
            .collect::<Vec<_>>();
        let wider_refs = wider.iter().collect::<Vec<_>>();
        assert_eq!(
            align_prefix_fanout(
                plan_prefix_fanout(&wider_refs, true),
                261,
                256,
                QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            ),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS + 4,
                selected_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                reason: "selected_chunk_aligned",
            }
        );
        assert_eq!(
            align_prefix_fanout(
                plan_prefix_fanout(&wider_refs, true),
                261,
                256,
                QwenPrefixFanoutBoundaryPolicy::ExactLcp,
            ),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS + 4,
                selected_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS + 4,
                reason: "selected_exact_lcp",
            }
        );
        assert_eq!(
            align_prefix_fanout(
                plan_prefix_fanout(&wider_refs, true),
                261,
                256,
                QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp,
            ),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS + 4,
                selected_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS + 4,
                reason: "selected_tiny_suffix_exact_lcp",
            }
        );
        assert_eq!(
            align_prefix_fanout(
                plan_prefix_fanout(&wider_refs, true),
                261,
                200,
                QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            ),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS + 4,
                selected_prefix_tokens: 0,
                reason: "alignment_below_minimum",
            }
        );

        let short = requests
            .iter()
            .enumerate()
            .map(|(slot, request)| prepared(&format!("short-{slot}"), &request.prompt_ids[1..]))
            .collect::<Vec<_>>();
        let short_refs = short.iter().collect::<Vec<_>>();
        assert_eq!(
            plan_prefix_fanout(&short_refs, true),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS - 1,
                selected_prefix_tokens: 0,
                reason: "below_minimum",
            }
        );
    }

    #[test]
    fn fixed_fanout_tiny_suffix_policy_freezes_six_token_boundary() {
        fn plan(
            common_prefix_tokens: usize,
            prompt_tokens: usize,
            boundary_policy: QwenPrefixFanoutBoundaryPolicy,
        ) -> PrefixFanoutPlan {
            align_prefix_fanout(
                PrefixFanoutPlan {
                    common_prefix_tokens,
                    selected_prefix_tokens: common_prefix_tokens,
                    reason: "selected",
                },
                prompt_tokens,
                512,
                boundary_policy,
            )
        }

        assert_eq!(
            plan(513, 519, QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp),
            PrefixFanoutPlan {
                common_prefix_tokens: 513,
                selected_prefix_tokens: 513,
                reason: "selected_tiny_suffix_exact_lcp",
            }
        );
        assert_eq!(
            plan(513, 520, QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp),
            PrefixFanoutPlan {
                common_prefix_tokens: 513,
                selected_prefix_tokens: 512,
                reason: "selected_chunk_aligned",
            }
        );
        assert_eq!(
            plan(513, 519, QwenPrefixFanoutBoundaryPolicy::ChunkAligned),
            PrefixFanoutPlan {
                common_prefix_tokens: 513,
                selected_prefix_tokens: 512,
                reason: "selected_chunk_aligned",
            }
        );
        assert_eq!(
            plan(300, 305, QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp),
            PrefixFanoutPlan {
                common_prefix_tokens: 300,
                selected_prefix_tokens: 0,
                reason: "alignment_below_minimum",
            }
        );
        assert_eq!(
            plan(300, 305, QwenPrefixFanoutBoundaryPolicy::ExactLcp),
            PrefixFanoutPlan {
                common_prefix_tokens: 300,
                selected_prefix_tokens: 300,
                reason: "selected_exact_lcp",
            }
        );
    }

    #[test]
    fn reorder_buffer_emits_only_contiguous_input_order() {
        let mut pending = std::iter::repeat_with(|| None)
            .take(3)
            .collect::<Vec<Option<RequestOutput>>>();
        let mut next = 0usize;
        let mut stdout = Vec::new();
        pending[2] = Some(output("two"));
        assert_eq!(
            flush_ready_outputs(&mut stdout, &mut pending, &mut next).unwrap(),
            0
        );
        pending[0] = Some(output("zero"));
        assert_eq!(
            flush_ready_outputs(&mut stdout, &mut pending, &mut next).unwrap(),
            1
        );
        pending[1] = Some(output("one"));
        assert_eq!(
            flush_ready_outputs(&mut stdout, &mut pending, &mut next).unwrap(),
            2
        );
        let rows = String::from_utf8(stdout).unwrap();
        let zero = rows.find("zero").unwrap();
        let one = rows.find("one").unwrap();
        let two = rows.find("two").unwrap();
        assert!(zero < one && one < two);
        assert_eq!(next, 3);
    }

    #[test]
    fn lane_progress_separates_logical_work_from_padding() {
        let mut one = LaneProgress::new(1);
        assert!(one.record_selection(10, &[99]).unwrap());
        assert_eq!(one.stop_reason, Some(StopReason::TokenLimit));
        one.validate_complete(0).unwrap();

        let mut transition_eos = LaneProgress::new(3);
        assert!(transition_eos.record_selection(10, &[99]).unwrap());
        assert!(!transition_eos.record_batch_step(true, 99, &[99]).unwrap());
        assert_eq!(transition_eos.stop_reason, Some(StopReason::Eos));
        transition_eos.validate_complete(1).unwrap();

        let mut full = LaneProgress::new(3);
        assert!(full.record_selection(10, &[99]).unwrap());
        assert!(full.record_batch_step(true, 11, &[99]).unwrap());
        assert!(full.record_batch_step(true, 12, &[99]).unwrap());
        assert_eq!(full.stop_reason, Some(StopReason::TokenLimit));
        assert_eq!(full.generated, [10, 11, 12]);
        full.validate_complete(2).unwrap();

        let mut short_limit = LaneProgress::new(2);
        assert!(short_limit.record_selection(10, &[99]).unwrap());
        assert!(short_limit.record_batch_step(true, 11, &[99]).unwrap());
        assert!(!short_limit.record_batch_step(false, 7, &[99]).unwrap());
        assert!(!short_limit.record_batch_step(false, 8, &[99]).unwrap());
        assert_eq!(short_limit.generated, [10, 11]);
        assert_eq!(short_limit.logical_transitions, 1);
        assert_eq!(short_limit.padding_transitions, 2);
        short_limit.validate_complete(3).unwrap();

        let mut early = LaneProgress::new(3);
        assert!(!early.record_selection(99, &[99]).unwrap());
        assert_eq!(early.stop_reason, Some(StopReason::Eos));
        assert_eq!(early.transition_token(), FINISHED_LANE_FILL_TOKEN);
        assert!(!early.record_batch_step(false, 7, &[99]).unwrap());
        assert!(!early.record_batch_step(false, 8, &[99]).unwrap());
        assert_eq!(early.generated, [99]);
        assert_eq!(early.logical_transitions, 0);
        assert_eq!(early.padding_transitions, 2);
        early.validate_complete(2).unwrap();
    }
}
