//! `qwen` — interactive CLI for the qwen-llm engine.

mod messages;

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::{CommandFactory, FromArgMatches, Parser, parser::ValueSource};
use messages::{
    load_deepseek_v4_0731_messages_prompt, load_messages_prompt_with_policy, messages_thinking_mode,
};
use qwen_llm::checkpoint_identity::IdentityCacheOutcome;
use qwen_llm::checkpoint_store::{DurableCheckpointStore, PublishOutcome, StagedIntegrityMode};
use qwen_llm::deepseek_v4::{AttentionLane, DeepSeekV4Model, RouterWeights};
use qwen_llm::deepseek_v4_census::DeepSeekV4CensusV1;
use qwen_llm::deepseek_v4_metal::{
    DEEPSEEK_V4_PREFILL_MAX_TOKENS, DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY, DeepSeekV4MemorySamples,
    DeepSeekV4MetalResidency, DeepSeekV4Session,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::{
    MetalBufferSizeAndAlign, MetalContext, MetalMemoryAdmission, MetalMemorySignals,
    MetalPipelineCacheMetrics, evaluate_metal_memory_admission,
};
use qwen_llm::metal_dflash::{
    MetalDFlashLayerMajorScratch, MetalDFlashVerifyScratch, PrefillScratchConfig,
    PrefillScratchOverlayStats, PrefillScratchPlan, ensure_prompt_lookup_n8_supported,
    plan_prefill_scratch_with_matrix_max_pos_configured, prefill_tokens_with_multi_hidden,
};
use qwen_llm::metal_forward::{
    LogitsReadbackProfile, MetalForward, MfError, SnapshotValidationError, StructuralRowEvidence,
    TokenProfile,
};
use qwen_llm::model::{Arch, ArchKind};
use qwen_llm::model_family::ModelFamily;
use qwen_llm::prompt_lookup::{DRAFT_TOKENS, PromptLookupProposer, terminal_draft_window};
use qwen_llm::runtime::{
    LoadedModel, LoadedModelConfig, PreparedCheckpoint, Runtime, RuntimeError, Sequence,
    SequenceConfig,
};
use qwen_llm::sampling::{
    BoundedTopKEvidence, GreedySelection, SAMPLER_ALGORITHM_VERSION, SampledToken, Sampler,
    SamplingConfig, SamplingError, SamplingPhaseProfile,
};
use qwen_llm::tensor::GgmlType;
use qwen_llm::tokenizer::{Tokenizer, token_ids_sha256_i32le};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ffi::OsStr;
use std::fs::OpenOptions;
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const CHECKPOINT_STAGED_INTEGRITY_ENV: &str = "QWEN_CHECKPOINT_STAGED_INTEGRITY";
const GREEDY_GPU_ARGMAX_ENV: &str = "QWEN_GREEDY_GPU_ARGMAX";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GreedyGpuArgmaxMode {
    DefaultOff,
    ForceEnabled,
    ExplicitRollback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GreedyGpuDecision {
    enabled: bool,
    reason: &'static str,
}

fn parse_greedy_gpu_argmax_mode(value: Option<&OsStr>) -> GreedyGpuArgmaxMode {
    match value {
        None => GreedyGpuArgmaxMode::DefaultOff,
        Some(value)
            if value
                .to_str()
                .is_some_and(qwen_llm::env_flag::env_value_truthy) =>
        {
            GreedyGpuArgmaxMode::ForceEnabled
        }
        Some(value)
            if value
                .to_str()
                .is_some_and(qwen_llm::env_flag::env_value_falsy) =>
        {
            GreedyGpuArgmaxMode::ExplicitRollback
        }
        Some(_) => GreedyGpuArgmaxMode::ExplicitRollback,
    }
}

fn configured_greedy_gpu_argmax_mode() -> GreedyGpuArgmaxMode {
    static MODE: std::sync::OnceLock<GreedyGpuArgmaxMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        parse_greedy_gpu_argmax_mode(std::env::var_os(GREEDY_GPU_ARGMAX_ENV).as_deref())
    })
}

fn resolve_greedy_gpu_decision(
    mode: GreedyGpuArgmaxMode,
    sampling: SamplingConfig,
    prompt_lookup: bool,
) -> GreedyGpuDecision {
    if mode == GreedyGpuArgmaxMode::ExplicitRollback {
        return GreedyGpuDecision {
            enabled: false,
            reason: "disabled_by_explicit_rollback",
        };
    }
    if sampling.temperature > 0.0 || prompt_lookup {
        return GreedyGpuDecision {
            enabled: false,
            reason: "ineligible_request",
        };
    }
    match mode {
        GreedyGpuArgmaxMode::ExplicitRollback => unreachable!("handled above"),
        GreedyGpuArgmaxMode::ForceEnabled => GreedyGpuDecision {
            enabled: true,
            reason: "force_enabled",
        },
        GreedyGpuArgmaxMode::DefaultOff => GreedyGpuDecision {
            enabled: false,
            reason: "default_off",
        },
    }
}

#[derive(Parser, Debug)]
#[command(name = "qwen", version, about = "qwen-llm inference CLI")]
struct Args {
    /// Path to a Qwen or DeepSeek V4 GGUF file.
    #[arg(short = 'm', long)]
    model: Option<std::path::PathBuf>,

    /// Print device info and exit.
    #[arg(long)]
    info: bool,

    /// Print the deterministic DeepSeek V4 schema/quant census as JSON.
    #[arg(
        long,
        requires = "model",
        conflicts_with_all = ["info", "prompt", "prompt_file", "messages", "requests_jsonl"]
    )]
    deepseek_census_json: bool,

    /// Raw prompt text for single-turn generation.
    #[arg(short = 'p', long, conflicts_with_all = ["prompt_file", "messages"])]
    prompt: Option<String>,

    /// Read raw prompt text from a file.
    #[arg(long, conflicts_with_all = ["prompt", "messages"])]
    prompt_file: Option<PathBuf>,

    /// Render a bare or wrapped JSON messages file with the model-family encoder.
    #[arg(long, conflicts_with_all = ["prompt", "prompt_file", "requests_jsonl"])]
    messages: Option<PathBuf>,

    /// Render only the first N messages.
    #[arg(long, requires = "messages")]
    messages_max: Option<usize>,

    /// Preserve assistant `<think>...</think>` history.
    #[arg(
        long,
        requires = "messages",
        conflicts_with = "messages_strip_thinking"
    )]
    messages_preserve_thinking: bool,

    /// Strip a leading assistant `<think>...</think>` block from history.
    #[arg(long, requires = "messages")]
    messages_strip_thinking: bool,

    /// Do not append the assistant generation prompt after messages.
    #[arg(long, requires = "messages")]
    messages_no_generation_prompt: bool,

    /// Read JSONL request objects from a file or '-' while keeping one model loaded.
    #[arg(long, conflicts_with_all = ["prompt", "prompt_file", "messages"])]
    requests_jsonl: Option<PathBuf>,

    /// Maximum number of tokens to generate.
    #[arg(short = 'n', long, default_value_t = 64)]
    tokens: usize,

    /// Sampling temperature; zero preserves greedy decoding.
    #[arg(long = "temp", visible_alias = "temperature", default_value_t = 0.0)]
    temperature: f32,

    /// Top-k sampling cutoff; zero disables it.
    #[arg(long, default_value_t = 200)]
    top_k: usize,

    /// Nucleus sampling cutoff; one disables it.
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,

    /// Min-p sampling cutoff; zero disables it.
    #[arg(long, default_value_t = 0.05)]
    min_p: f32,

    /// Effective deterministic seed; identical requests reuse the same stream.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Enable experimental dense-27B Q4_K_M prompt-lookup decode.
    #[arg(long)]
    prompt_lookup: bool,

    /// Prompt prefill chunk size, or `auto` for the bounded MoE allowlist.
    #[arg(long, default_value = "1024")]
    prefill_chunk: PrefillChunkArg,

    /// Override sequence capacity. Defaults to prompt + generated tokens + slack.
    #[arg(long)]
    max_context_tokens: Option<usize>,

    /// Prefix-cache byte budget in MiB; oversized snapshots are retained alone.
    #[arg(long, default_value_t = 16 * 1024)]
    prefix_cache_max_mib: u64,

    /// Cache this many exact prompt tokens as the reusable prefix for requests.
    #[arg(long)]
    cache_prefix_tokens: Option<usize>,

    /// Auto-cache repeated JSONL prompt prefixes at or above this token length.
    #[arg(long, default_value_t = 1024)]
    cache_prefix_auto_min_tokens: usize,

    /// Persist anonymous prefix checkpoints under this private directory.
    #[arg(long)]
    durable_prefix_cache: Option<PathBuf>,

    /// Aggregate durable checkpoint budget in MiB.
    #[arg(long, default_value_t = 32 * 1024)]
    durable_prefix_cache_max_mib: u64,

    /// Maximum size of one encoded durable checkpoint record in MiB.
    #[arg(long, default_value_t = 16 * 1024)]
    durable_prefix_cache_max_entry_mib: u64,

    /// Auto-persist one-shot prompt boundaries at or above this token length.
    #[arg(long, default_value_t = 1024)]
    durable_prefix_cache_min_tokens: usize,

    /// Append per-request JSON stats for multi-request runs.
    ///
    /// Timing fields are model-internal; this JSONL mode writes each completion
    /// after full decode rather than streaming the first token to stdout.
    #[arg(long)]
    request_stats: Option<PathBuf>,

    /// Append single-turn first-post-model-load timing rows as JSONL.
    #[arg(long)]
    request_timings: Option<PathBuf>,

    /// Run one identical warm follow-up for paired request timing.
    #[arg(long)]
    request_timing_warm_followup: bool,

    /// Attribute sampler-v1 and full-logit host work on the frozen A3B request.
    #[arg(
        long,
        requires = "request_timings",
        conflicts_with_all = [
            "requests_jsonl",
            "request_timing_warm_followup",
            "prompt_lookup",
            "durable_prefix_cache"
        ]
    )]
    sampling_attribution: bool,

    /// Use bounded top-k over synchronized resident logits for sampled decode.
    #[arg(
        long,
        hide = true,
        conflicts_with_all = [
            "requests_jsonl",
            "request_timing_warm_followup",
            "prompt_lookup",
            "sampling_attribution",
            "durable_prefix_cache"
        ]
    )]
    sampled_structural: bool,

    /// Do not ask the tokenizer to add model-defined special tokens.
    #[arg(long)]
    no_special_tokens: bool,

    /// Append a FIFO request-trace row after single-turn generation completes.
    ///
    /// Format is compatible with `scripts/profile/replay_economics.py
    /// --request-trace`: `arrival_ms tokens id ...`.
    #[arg(long)]
    trace_request: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ExplicitCliOptions {
    prefill_chunk: bool,
    prefix_cache_max_mib: bool,
    cache_prefix_auto_min_tokens: bool,
    durable_prefix_cache_max_mib: bool,
    durable_prefix_cache_max_entry_mib: bool,
    durable_prefix_cache_min_tokens: bool,
}

impl ExplicitCliOptions {
    fn from_matches(matches: &clap::ArgMatches) -> Self {
        let command_line = |id| matches.value_source(id) == Some(ValueSource::CommandLine);
        Self {
            prefill_chunk: command_line("prefill_chunk"),
            prefix_cache_max_mib: command_line("prefix_cache_max_mib"),
            cache_prefix_auto_min_tokens: command_line("cache_prefix_auto_min_tokens"),
            durable_prefix_cache_max_mib: command_line("durable_prefix_cache_max_mib"),
            durable_prefix_cache_max_entry_mib: command_line("durable_prefix_cache_max_entry_mib"),
            durable_prefix_cache_min_tokens: command_line("durable_prefix_cache_min_tokens"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrefillChunkArg {
    Fixed(usize),
    Auto,
}

impl PrefillChunkArg {
    fn is_auto(self) -> bool {
        self == Self::Auto
    }

    fn validate(self) -> Result<()> {
        ensure!(
            !matches!(self, Self::Fixed(0)),
            "--prefill-chunk must be >= 1 or auto"
        );
        Ok(())
    }
}

impl FromStr for PrefillChunkArg {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value == "auto" {
            return Ok(Self::Auto);
        }
        value
            .parse::<usize>()
            .map(Self::Fixed)
            .map_err(|_| format!("expected a positive integer or auto, got {value:?}"))
    }
}

impl Serialize for PrefillChunkArg {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Fixed(value) => serializer.serialize_u64(*value as u64),
            Self::Auto => serializer.serialize_str("auto"),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct PrefillChunkDecision {
    policy: &'static str,
    profile: Option<&'static str>,
    classification: &'static str,
    reason: &'static str,
    candidate: Option<usize>,
    selected: usize,
    validated_prompt_range: Option<[usize; 2]>,
    evidence_baseline_chunk: Option<usize>,
    baseline: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    plan: Option<PrefillPlanDecision>,
    #[serde(skip_serializing_if = "Option::is_none")]
    admission: Option<PrefillAdmissionDecision>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AutoPrefillProfile {
    name: &'static str,
    outer_chunk: usize,
    query_heads: u64,
    gdn_overlay_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
struct PrefillPlanAllocationDecision {
    name: &'static str,
    deferred: bool,
    logical_bytes: u64,
    priced_bytes: u64,
    alignment: u64,
}

#[derive(Clone, Debug, Serialize)]
struct PrefillPlanDecision {
    block_size: u32,
    matrix_max_pos: u64,
    matrix_query_rows: u32,
    eager_allocation_count: usize,
    deferred_allocation_count: usize,
    eager_logical_bytes: u64,
    deferred_logical_bytes: u64,
    maximum_logical_bytes: u64,
    priced_upper_bytes: u64,
    overlay: PrefillScratchOverlayTimingStats,
    allocations: Vec<PrefillPlanAllocationDecision>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct PrefillMemorySignalsDecision {
    recommended_max_bytes: u64,
    current_allocated_bytes: u64,
    process_limit_remaining_bytes: Option<u64>,
    working_set_headroom_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
struct PrefillAdmissionDecision {
    current_allocated_before_sequence: u64,
    current_allocated_after_sequence: u64,
    sequence_allocation_delta_bytes: u64,
    transient_reserve_bytes: u64,
    reserve_bytes: u64,
    required_bytes: Option<u64>,
    signals: PrefillMemorySignalsDecision,
    evaluator_reason: &'static str,
    admitted: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonlRequest {
    id: Option<String>,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    tokens: Option<usize>,
    cache_prefix_tokens: Option<usize>,
    sampling: Option<JsonlSampling>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonlSampling {
    #[serde(alias = "temp")]
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    seed: Option<u64>,
}

#[derive(Debug)]
struct PreparedJsonlRequest {
    request: JsonlRequest,
    id: String,
    line: usize,
    prompt_ids: Vec<i32>,
    sampling: SamplingConfig,
    auto_cache_prefix_tokens: Option<usize>,
    auto_cache_future_hits: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CachePrefixSource {
    None,
    Request,
    RequestDisabled,
    Cli,
    CliDisabled,
    Auto,
}

impl CachePrefixSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Request => "request",
            Self::RequestDisabled => "request_disabled",
            Self::Cli => "cli",
            Self::CliDisabled => "cli_disabled",
            Self::Auto => "auto",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum PromptSource {
    Inline,
    File,
    Messages,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum StopReason {
    Eos,
    TokenLimit,
}

impl StopReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Eos => "eos",
            Self::TokenLimit => "token_limit",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct SamplingTelemetry {
    algorithm_version: u32,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    effective_seed: u64,
    draws: usize,
}

#[derive(Clone, Debug, Serialize)]
struct SampledStructuralTelemetry {
    version: u32,
    algorithm_version: u32,
    path: &'static str,
    prompt_owned_bounded_calls: u64,
    borrowed_transition_calls: u64,
    resident_head_wait_calls: u64,
    validated_shared_row_calls: u64,
    fallback_calls: u64,
    input_logits_total: u64,
    input_logits_min: u64,
    input_logits_max: u64,
    retained_top_k_total: u64,
    retained_top_k_min: u64,
    retained_top_k_max: u64,
    max_heap_len: u64,
    max_heap_capacity: u64,
    full_candidate_vector_allocations: u64,
    transition_logits_copy_bytes: u64,
    extra_command_buffers: u64,
    gpu_sampling_dispatches: u64,
}

impl Default for SampledStructuralTelemetry {
    fn default() -> Self {
        Self {
            version: 1,
            algorithm_version: SAMPLER_ALGORITHM_VERSION,
            path: "bounded_topk_borrowed_transitions",
            prompt_owned_bounded_calls: 0,
            borrowed_transition_calls: 0,
            resident_head_wait_calls: 0,
            validated_shared_row_calls: 0,
            fallback_calls: 0,
            input_logits_total: 0,
            input_logits_min: 0,
            input_logits_max: 0,
            retained_top_k_total: 0,
            retained_top_k_min: 0,
            retained_top_k_max: 0,
            max_heap_len: 0,
            max_heap_capacity: 0,
            full_candidate_vector_allocations: 0,
            transition_logits_copy_bytes: 0,
            extra_command_buffers: 0,
            gpu_sampling_dispatches: 0,
        }
    }
}

impl SampledStructuralTelemetry {
    fn record_bounded(&mut self, evidence: BoundedTopKEvidence, prompt: bool) -> Result<()> {
        let calls = self
            .prompt_owned_bounded_calls
            .checked_add(self.borrowed_transition_calls)
            .context("sampled structural call count overflow")?;
        if prompt {
            self.prompt_owned_bounded_calls = self
                .prompt_owned_bounded_calls
                .checked_add(1)
                .context("prompt bounded-call count overflow")?;
        } else {
            self.borrowed_transition_calls = self
                .borrowed_transition_calls
                .checked_add(1)
                .context("borrowed transition count overflow")?;
        }
        if !evidence.used_bounded_path {
            self.fallback_calls = self
                .fallback_calls
                .checked_add(1)
                .context("sampled structural fallback count overflow")?;
        }
        let input = u64::try_from(evidence.input_logits).context("input logits do not fit u64")?;
        let retained =
            u64::try_from(evidence.retained_top_k).context("retained top-k does not fit u64")?;
        let heap_len =
            u64::try_from(evidence.max_heap_len).context("heap length does not fit u64")?;
        let heap_capacity =
            u64::try_from(evidence.heap_capacity).context("heap capacity does not fit u64")?;
        self.input_logits_total = self
            .input_logits_total
            .checked_add(input)
            .context("input logits total overflow")?;
        self.retained_top_k_total = self
            .retained_top_k_total
            .checked_add(retained)
            .context("retained top-k total overflow")?;
        if calls == 0 {
            self.input_logits_min = input;
            self.input_logits_max = input;
            self.retained_top_k_min = retained;
            self.retained_top_k_max = retained;
        } else {
            self.input_logits_min = self.input_logits_min.min(input);
            self.input_logits_max = self.input_logits_max.max(input);
            self.retained_top_k_min = self.retained_top_k_min.min(retained);
            self.retained_top_k_max = self.retained_top_k_max.max(retained);
        }
        self.max_heap_len = self.max_heap_len.max(heap_len);
        self.max_heap_capacity = self.max_heap_capacity.max(heap_capacity);
        Ok(())
    }

    fn record_prompt(&mut self, evidence: BoundedTopKEvidence) -> Result<()> {
        self.record_bounded(evidence, true)
    }

    fn record_transition(
        &mut self,
        bounded: BoundedTopKEvidence,
        row: StructuralRowEvidence,
    ) -> Result<()> {
        self.record_bounded(bounded, false)?;
        self.resident_head_wait_calls = self
            .resident_head_wait_calls
            .checked_add(row.resident_head_wait_calls)
            .context("resident-head wait count overflow")?;
        self.validated_shared_row_calls = self
            .validated_shared_row_calls
            .checked_add(row.validated_shared_row_calls)
            .context("validated Shared-row count overflow")?;
        self.transition_logits_copy_bytes = self
            .transition_logits_copy_bytes
            .checked_add(row.transition_logits_copy_bytes)
            .context("transition logits-copy byte count overflow")?;
        self.extra_command_buffers = self
            .extra_command_buffers
            .checked_add(row.extra_command_buffers)
            .context("extra command-buffer count overflow")?;
        self.gpu_sampling_dispatches = self
            .gpu_sampling_dispatches
            .checked_add(row.gpu_sampling_dispatches)
            .context("GPU sampling-dispatch count overflow")?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct CountSummary {
    total: u64,
    min: u64,
    max: u64,
}

#[derive(Clone, Debug, Serialize)]
struct SamplingClockProbe {
    batches: u32,
    iterations_per_batch: u64,
    pair_ns: [f64; 7],
    upper_pair_ns: f64,
    new_timer_spans: u64,
    observer_upper_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
struct SamplerAttribution {
    calls: u64,
    timer_spans: u64,
    input_logits_total: u64,
    input_logits_min: u64,
    input_logits_max: u64,
    wall_ms: f64,
    shape_validation_ms: f64,
    candidate_alloc_ms: f64,
    candidate_fill_ms: f64,
    top_k_order_ms: f64,
    min_p_ms: f64,
    positive_infinity_ms: f64,
    temperature_scale_ms: f64,
    probability_weights_ms: f64,
    top_p_ms: f64,
    categorical_ms: f64,
    residual_ms: f64,
    candidate_capacity_bytes_total: u64,
    candidate_capacity_bytes_peak: u64,
    probability_capacity_bytes_total: u64,
    probability_capacity_bytes_peak: u64,
    after_top_k: CountSummary,
    after_min_p: CountSummary,
    after_positive_infinity: CountSummary,
    after_top_p: CountSummary,
    candidate_index: CountSummary,
}

#[derive(Clone, Debug, Serialize)]
struct TransitionAttribution {
    calls: u64,
    new_timer_spans: u64,
    logits_bytes_per_call: u64,
    logits_bytes_total: u64,
    outer_wall_ms: f64,
    inner_wall_ms: f64,
    cpu_encode_ms: f64,
    completion_wait_ms: f64,
    gpu_ms_nested: f64,
    logits_alloc_zero_ms: f64,
    logits_copy_ms: f64,
    inner_residual_ms: f64,
    outer_wrapper_advance_ms: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct SamplingAttributionBounds {
    observer_upper_ms: f64,
    workspace_raw_ms: f64,
    workspace_adjusted_ms: f64,
    workspace_adjusted_fraction: f64,
    borrowed_raw_ms: f64,
    borrowed_adjusted_ms: f64,
    borrowed_adjusted_fraction: f64,
    combined_raw_ms: f64,
    combined_adjusted_ms: f64,
    combined_adjusted_fraction: f64,
    structural_raw_ms: f64,
    structural_adjusted_ms: f64,
    structural_adjusted_fraction: f64,
}

#[derive(Clone, Debug, Serialize)]
struct SamplingAttribution {
    version: u32,
    prompt_token_ids_sha256: String,
    clock_probe: SamplingClockProbe,
    sampler: SamplerAttribution,
    transitions: TransitionAttribution,
    bounds: SamplingAttributionBounds,
}

#[derive(Debug, Default)]
struct CountAccumulator {
    total: u64,
    min: Option<u64>,
    max: u64,
}

impl CountAccumulator {
    fn record(&mut self, value: usize, label: &str) -> Result<()> {
        let value = u64::try_from(value).with_context(|| format!("{label} does not fit u64"))?;
        self.total = self
            .total
            .checked_add(value)
            .with_context(|| format!("{label} total overflow"))?;
        self.min = Some(self.min.map_or(value, |current| current.min(value)));
        self.max = self.max.max(value);
        Ok(())
    }

    fn finish(self) -> CountSummary {
        CountSummary {
            total: self.total,
            min: self.min.unwrap_or(0),
            max: self.max,
        }
    }
}

#[derive(Debug, Default)]
struct SamplerAttributionAccumulator {
    calls: u64,
    timer_spans: u64,
    input_logits: CountAccumulator,
    wall_ms: f64,
    shape_validation_ms: f64,
    candidate_alloc_ms: f64,
    candidate_fill_ms: f64,
    top_k_order_ms: f64,
    min_p_ms: f64,
    positive_infinity_ms: f64,
    temperature_scale_ms: f64,
    probability_weights_ms: f64,
    top_p_ms: f64,
    categorical_ms: f64,
    residual_ms: f64,
    candidate_capacity_bytes_total: u64,
    candidate_capacity_bytes_peak: u64,
    probability_capacity_bytes_total: u64,
    probability_capacity_bytes_peak: u64,
    after_top_k: CountAccumulator,
    after_min_p: CountAccumulator,
    after_positive_infinity: CountAccumulator,
    after_top_p: CountAccumulator,
    candidate_index: CountAccumulator,
}

impl SamplerAttributionAccumulator {
    fn record(&mut self, profile: SamplingPhaseProfile) -> Result<()> {
        ensure!(
            profile.input_logits > 0
                && profile.after_top_k > 0
                && profile.after_top_k <= profile.input_logits
                && profile.after_min_p > 0
                && profile.after_min_p <= profile.after_top_k
                && profile.after_positive_infinity > 0
                && profile.after_positive_infinity <= profile.after_min_p
                && profile.after_top_p > 0
                && profile.after_top_p <= profile.after_positive_infinity
                && profile.candidate_index < profile.after_top_p,
            "profiled sampler support accounting is invalid"
        );
        self.calls = self.calls.checked_add(1).context("sampler call overflow")?;
        self.timer_spans = self
            .timer_spans
            .checked_add(u64::from(profile.timer_spans))
            .context("sampler timer-span overflow")?;
        self.input_logits
            .record(profile.input_logits, "input logits")?;
        self.wall_ms += profile.total_ms;
        self.shape_validation_ms += profile.shape_validation_ms;
        self.candidate_alloc_ms += profile.candidate_alloc_ms;
        self.candidate_fill_ms += profile.candidate_fill_ms;
        self.top_k_order_ms += profile.top_k_order_ms;
        self.min_p_ms += profile.min_p_ms;
        self.positive_infinity_ms += profile.positive_infinity_ms;
        self.temperature_scale_ms += profile.temperature_scale_ms;
        self.probability_weights_ms += profile.probability_weights_ms;
        self.top_p_ms += profile.top_p_ms;
        self.categorical_ms += profile.categorical_ms;
        self.residual_ms += profile.residual_ms;

        let candidate_bytes = u64::try_from(profile.candidate_capacity_bytes)
            .context("candidate capacity bytes do not fit u64")?;
        self.candidate_capacity_bytes_total = self
            .candidate_capacity_bytes_total
            .checked_add(candidate_bytes)
            .context("candidate capacity-byte total overflow")?;
        self.candidate_capacity_bytes_peak =
            self.candidate_capacity_bytes_peak.max(candidate_bytes);
        let probability_bytes = u64::try_from(profile.probability_capacity_bytes)
            .context("probability capacity bytes do not fit u64")?;
        self.probability_capacity_bytes_total = self
            .probability_capacity_bytes_total
            .checked_add(probability_bytes)
            .context("probability capacity-byte total overflow")?;
        self.probability_capacity_bytes_peak =
            self.probability_capacity_bytes_peak.max(probability_bytes);

        self.after_top_k
            .record(profile.after_top_k, "after top-k")?;
        self.after_min_p
            .record(profile.after_min_p, "after min-p")?;
        self.after_positive_infinity.record(
            profile.after_positive_infinity,
            "after positive-infinity filter",
        )?;
        self.after_top_p
            .record(profile.after_top_p, "after top-p")?;
        self.candidate_index
            .record(profile.candidate_index, "candidate index")?;
        Ok(())
    }

    fn finish(self) -> SamplerAttribution {
        SamplerAttribution {
            calls: self.calls,
            timer_spans: self.timer_spans,
            input_logits_total: self.input_logits.total,
            input_logits_min: self.input_logits.min.unwrap_or(0),
            input_logits_max: self.input_logits.max,
            wall_ms: self.wall_ms,
            shape_validation_ms: self.shape_validation_ms,
            candidate_alloc_ms: self.candidate_alloc_ms,
            candidate_fill_ms: self.candidate_fill_ms,
            top_k_order_ms: self.top_k_order_ms,
            min_p_ms: self.min_p_ms,
            positive_infinity_ms: self.positive_infinity_ms,
            temperature_scale_ms: self.temperature_scale_ms,
            probability_weights_ms: self.probability_weights_ms,
            top_p_ms: self.top_p_ms,
            categorical_ms: self.categorical_ms,
            residual_ms: self.residual_ms,
            candidate_capacity_bytes_total: self.candidate_capacity_bytes_total,
            candidate_capacity_bytes_peak: self.candidate_capacity_bytes_peak,
            probability_capacity_bytes_total: self.probability_capacity_bytes_total,
            probability_capacity_bytes_peak: self.probability_capacity_bytes_peak,
            after_top_k: self.after_top_k.finish(),
            after_min_p: self.after_min_p.finish(),
            after_positive_infinity: self.after_positive_infinity.finish(),
            after_top_p: self.after_top_p.finish(),
            candidate_index: self.candidate_index.finish(),
        }
    }
}

#[derive(Debug, Default)]
struct TransitionAttributionAccumulator {
    calls: u64,
    new_timer_spans: u64,
    logits_bytes_per_call: Option<u64>,
    logits_bytes_total: u64,
    inner_wall_ms: f64,
    cpu_encode_ms: f64,
    completion_wait_ms: f64,
    gpu_ms_nested: f64,
    logits_alloc_zero_ms: f64,
    logits_copy_ms: f64,
    inner_residual_ms: f64,
}

impl TransitionAttributionAccumulator {
    fn record(&mut self, token: TokenProfile, readback: LogitsReadbackProfile) -> Result<()> {
        self.calls = self
            .calls
            .checked_add(1)
            .context("transition attribution call overflow")?;
        self.new_timer_spans = self
            .new_timer_spans
            .checked_add(u64::from(readback.timer_spans))
            .context("transition timer-span overflow")?;
        let bytes = u64::try_from(readback.bytes).context("logits bytes do not fit u64")?;
        if let Some(expected) = self.logits_bytes_per_call {
            ensure!(
                bytes == expected,
                "profiled transition logits bytes changed: {bytes} != {expected}"
            );
        } else {
            self.logits_bytes_per_call = Some(bytes);
        }
        self.logits_bytes_total = self
            .logits_bytes_total
            .checked_add(bytes)
            .context("transition logits-byte total overflow")?;
        self.inner_wall_ms += token.total_ms;
        self.cpu_encode_ms += token.cpu_encode_ms;
        self.completion_wait_ms += token.cpu_to_gpu_complete_ms;
        self.gpu_ms_nested += token.gpu_kernel_ms;
        self.logits_alloc_zero_ms += readback.allocation_zero_fill_ms;
        self.logits_copy_ms += readback.copy_ms;
        self.inner_residual_ms += token.total_ms
            - token.cpu_encode_ms
            - token.cpu_to_gpu_complete_ms
            - readback.allocation_zero_fill_ms
            - readback.copy_ms;
        Ok(())
    }

    fn finish(self, outer_wall_ms: f64) -> TransitionAttribution {
        TransitionAttribution {
            calls: self.calls,
            new_timer_spans: self.new_timer_spans,
            logits_bytes_per_call: self.logits_bytes_per_call.unwrap_or(0),
            logits_bytes_total: self.logits_bytes_total,
            outer_wall_ms,
            inner_wall_ms: self.inner_wall_ms,
            cpu_encode_ms: self.cpu_encode_ms,
            completion_wait_ms: self.completion_wait_ms,
            gpu_ms_nested: self.gpu_ms_nested,
            logits_alloc_zero_ms: self.logits_alloc_zero_ms,
            logits_copy_ms: self.logits_copy_ms,
            inner_residual_ms: self.inner_residual_ms,
            outer_wrapper_advance_ms: outer_wall_ms - self.inner_wall_ms,
        }
    }
}

fn measure_sampling_clock_probe() -> SamplingClockProbe {
    const BATCHES: usize = 7;
    const ITERATIONS: usize = 100_000;
    const NEW_TIMER_SPANS: u64 = 1_662;
    let mut pair_ns = [0.0; BATCHES];
    for value in &mut pair_ns {
        let batch_t0 = Instant::now();
        for _ in 0..ITERATIONS {
            let pair_t0 = Instant::now();
            std::hint::black_box(pair_t0.elapsed());
        }
        *value = batch_t0.elapsed().as_secs_f64() * 1e9 / ITERATIONS as f64;
    }
    let upper_pair_ns = pair_ns.iter().copied().fold(0.0f64, f64::max).ceil();
    SamplingClockProbe {
        batches: BATCHES as u32,
        iterations_per_batch: ITERATIONS as u64,
        pair_ns,
        upper_pair_ns,
        new_timer_spans: NEW_TIMER_SPANS,
        observer_upper_ms: upper_pair_ns * NEW_TIMER_SPANS as f64 / 1e6,
    }
}

fn adjusted_bound(raw_ms: f64, observer_upper_ms: f64) -> f64 {
    (raw_ms - observer_upper_ms).max(0.0)
}

fn bound_fraction(adjusted_ms: f64, generation_ms: f64) -> f64 {
    if generation_ms > 0.0 {
        adjusted_ms / generation_ms
    } else {
        0.0
    }
}

fn finalize_sampling_attribution(
    prompt_ids: &[i32],
    clock_probe: SamplingClockProbe,
    sampler: SamplerAttributionAccumulator,
    transitions: TransitionAttributionAccumulator,
    outer_transition_ms: f64,
    generation_ms: f64,
) -> SamplingAttribution {
    let sampler = sampler.finish();
    let transitions = transitions.finish(outer_transition_ms);
    let observer_upper_ms = clock_probe.observer_upper_ms;
    let workspace_raw_ms = transitions.logits_alloc_zero_ms + sampler.candidate_alloc_ms;
    let borrowed_raw_ms = transitions.logits_alloc_zero_ms + transitions.logits_copy_ms;
    let combined_raw_ms = borrowed_raw_ms + sampler.candidate_alloc_ms;
    let structural_raw_ms = combined_raw_ms + sampler.candidate_fill_ms + sampler.top_k_order_ms;
    let workspace_adjusted_ms = adjusted_bound(workspace_raw_ms, observer_upper_ms);
    let borrowed_adjusted_ms = adjusted_bound(borrowed_raw_ms, observer_upper_ms);
    let combined_adjusted_ms = adjusted_bound(combined_raw_ms, observer_upper_ms);
    let structural_adjusted_ms = adjusted_bound(structural_raw_ms, observer_upper_ms);
    SamplingAttribution {
        version: 1,
        prompt_token_ids_sha256: token_ids_sha256_i32le(prompt_ids),
        clock_probe,
        sampler,
        transitions,
        bounds: SamplingAttributionBounds {
            observer_upper_ms,
            workspace_raw_ms,
            workspace_adjusted_ms,
            workspace_adjusted_fraction: bound_fraction(workspace_adjusted_ms, generation_ms),
            borrowed_raw_ms,
            borrowed_adjusted_ms,
            borrowed_adjusted_fraction: bound_fraction(borrowed_adjusted_ms, generation_ms),
            combined_raw_ms,
            combined_adjusted_ms,
            combined_adjusted_fraction: bound_fraction(combined_adjusted_ms, generation_ms),
            structural_raw_ms,
            structural_adjusted_ms,
            structural_adjusted_fraction: bound_fraction(structural_adjusted_ms, generation_ms),
        },
    }
}

impl SamplingTelemetry {
    fn sampled(config: SamplingConfig, draws: usize) -> Option<Self> {
        (config.temperature > 0.0).then_some(Self {
            algorithm_version: SAMPLER_ALGORITHM_VERSION,
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
            effective_seed: config.seed,
            draws,
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct MetalAllocationSample {
    current_bytes: u64,
    delta_from_model_ready_bytes: i64,
    delta_from_request_start_bytes: i64,
}

#[derive(Debug, Serialize)]
struct MetalAllocationSamples {
    process_model_ready: MetalAllocationSample,
    request_start: MetalAllocationSample,
    after_scratch: MetalAllocationSample,
    after_sequence: MetalAllocationSample,
    after_prefill: MetalAllocationSample,
    after_first_stdout_flush: MetalAllocationSample,
    request_end_before_state_drop: MetalAllocationSample,
    after_request_state_drop: MetalAllocationSample,
    current_allocated_sampled_max_bytes: u64,
}

#[derive(Debug, Serialize)]
struct PrefillAttentionQueryStats {
    outer_chunk_rows: usize,
    query_rows: usize,
    tiled_layer_calls: u64,
    query_tile_calls: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct PrefillScratchOverlayTimingStats {
    backing_bytes: u64,
    attention_bytes: u64,
    gdn_bytes: u64,
    saved_bytes: u64,
}

#[derive(Debug, Serialize)]
struct RequestTimingRow {
    schema_version: u32,
    request_epoch: &'static str,
    request_index: usize,
    tokenizer_reused: bool,
    pair_requested: bool,
    pair_id: Option<String>,
    pair_request_equal: Option<bool>,
    pair_generated_tokens_equal: Option<bool>,
    prefix_cache_used: bool,
    build_commit: &'static str,
    build_dirty: &'static str,
    build_source_state: &'static str,
    model: String,
    runtime_identity_kind: &'static str,
    runtime_model_id: String,
    runtime_tokenizer_id: String,
    greedy_gpu_selection_reason: &'static str,
    request_start_unix_ms: u64,
    runtime_and_model_load_ms: f64,
    stdout_sink: &'static str,
    ttft_endpoint: &'static str,
    prompt_source: PromptSource,
    prompt_bytes: usize,
    prompt_tokens: usize,
    requested_tokens: usize,
    generated_tokens: usize,
    generated_token_sha256: String,
    stop_reason: StopReason,
    decode_policy: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    sampling: Option<SamplingTelemetry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sampling_attribution: Option<SamplingAttribution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sampled_structural: Option<SampledStructuralTelemetry>,
    terminal_token_target_transition_consumed: bool,
    no_special_tokens: bool,
    prefill_chunk_requested: PrefillChunkArg,
    prefill_chunk_effective: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_chunk_decision: Option<PrefillChunkDecision>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_attention_query: Option<PrefillAttentionQueryStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_scratch_overlay: Option<PrefillScratchOverlayTimingStats>,
    max_context_tokens: usize,
    prompt_acquisition_ms: f64,
    tokenizer_init_ms: f64,
    tokenization_ms: f64,
    capacity_validation_ms: f64,
    scratch_allocation_ms: f64,
    sequence_allocation_ms: f64,
    prefill_ms: f64,
    first_token_selection_ms: f64,
    first_token_callback_duration_ms: f64,
    first_token_ready_ms: f64,
    ttft_ms: f64,
    generation_ms: f64,
    transition_count: usize,
    transition_ms: f64,
    transition_tps: f64,
    inference_complete_ms: f64,
    total_request_ms: f64,
    pso_cache: PipelineCachePhaseMetrics,
    metal_allocated: MetalAllocationSamples,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_lookup: Option<PromptLookupDecodeStats>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct PipelineCacheMetricDelta {
    misses: u64,
    miss_wall_ns: u64,
    compiler_wall_ns: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct PipelineCachePhaseMetrics {
    prefill: PipelineCacheMetricDelta,
    generation: PipelineCacheMetricDelta,
    total: PipelineCacheMetricDelta,
}

#[derive(Debug)]
struct PreparedRequest {
    request_start_unix_ms: u64,
    request_t0: Instant,
    request_start_allocated: Option<u64>,
    pipeline_cache_start: Option<MetalPipelineCacheMetrics>,
    prompt: String,
    prompt_source: PromptSource,
    completed_checkpoint_eligible: bool,
    prompt_ids: Vec<i32>,
    prompt_acquisition_ms: f64,
    tokenizer_init_ms: f64,
    tokenization_ms: f64,
    tokenizer_reused: bool,
}

fn pipeline_cache_delta(
    after: MetalPipelineCacheMetrics,
    before: MetalPipelineCacheMetrics,
) -> PipelineCacheMetricDelta {
    let delta = after.saturating_delta_since(before);
    PipelineCacheMetricDelta {
        misses: delta.misses,
        miss_wall_ns: delta.miss_wall_ns,
        compiler_wall_ns: delta.compiler_wall_ns,
    }
}

fn pipeline_cache_phase_metrics(
    request_start: MetalPipelineCacheMetrics,
    prefill_entry: MetalPipelineCacheMetrics,
    prefill_exit: MetalPipelineCacheMetrics,
    generation_exit: MetalPipelineCacheMetrics,
) -> PipelineCachePhaseMetrics {
    PipelineCachePhaseMetrics {
        prefill: pipeline_cache_delta(prefill_exit, prefill_entry),
        generation: pipeline_cache_delta(generation_exit, prefill_exit),
        total: pipeline_cache_delta(generation_exit, request_start),
    }
}

#[derive(Debug)]
struct SingleTurnResult {
    row: Option<RequestTimingRow>,
    prompt: String,
    prompt_source: PromptSource,
    prompt_ids: Vec<i32>,
    generated: Vec<i32>,
    transitions: usize,
    stop_reason: StopReason,
    prefill_ms: f64,
    ttft_ms: f64,
    decode_tps: f64,
    transition_tps: f64,
    tokenizer_init_ms: f64,
}

fn allocation_delta(current: u64, model_ready: u64) -> i64 {
    if current >= model_ready {
        i64::try_from(current - model_ready).unwrap_or(i64::MAX)
    } else {
        -i64::try_from(model_ready - current).unwrap_or(i64::MAX)
    }
}

fn allocation_sample(current: u64, model_ready: u64, request_start: u64) -> MetalAllocationSample {
    MetalAllocationSample {
        current_bytes: current,
        delta_from_model_ready_bytes: allocation_delta(current, model_ready),
        delta_from_request_start_bytes: allocation_delta(current, request_start),
    }
}

fn metal_allocation_samples(
    model_ready: u64,
    request_start: u64,
    after_scratch: u64,
    after_sequence: u64,
    after_prefill: u64,
    after_first_stdout_flush: u64,
    request_end_before_state_drop: u64,
    after_request_state_drop: u64,
) -> MetalAllocationSamples {
    let sampled_max = [
        model_ready,
        request_start,
        after_scratch,
        after_sequence,
        after_prefill,
        after_first_stdout_flush,
        request_end_before_state_drop,
        after_request_state_drop,
    ]
    .into_iter()
    .max()
    .unwrap_or(model_ready);
    MetalAllocationSamples {
        process_model_ready: allocation_sample(model_ready, model_ready, request_start),
        request_start: allocation_sample(request_start, model_ready, request_start),
        after_scratch: allocation_sample(after_scratch, model_ready, request_start),
        after_sequence: allocation_sample(after_sequence, model_ready, request_start),
        after_prefill: allocation_sample(after_prefill, model_ready, request_start),
        after_first_stdout_flush: allocation_sample(
            after_first_stdout_flush,
            model_ready,
            request_start,
        ),
        request_end_before_state_drop: allocation_sample(
            request_end_before_state_drop,
            model_ready,
            request_start,
        ),
        after_request_state_drop: allocation_sample(
            after_request_state_drop,
            model_ready,
            request_start,
        ),
        current_allocated_sampled_max_bytes: sampled_max,
    }
}

fn validate_request_timing_invariants(
    first_token_ready_ms: f64,
    ttft_ms: f64,
    inference_complete_ms: f64,
    total_request_ms: f64,
    generated_tokens: usize,
    transition_count: usize,
) -> Result<()> {
    for (name, value) in [
        ("first_token_ready_ms", first_token_ready_ms),
        ("ttft_ms", ttft_ms),
        ("inference_complete_ms", inference_complete_ms),
        ("total_request_ms", total_request_ms),
    ] {
        ensure!(value.is_finite() && value >= 0.0, "invalid {name}: {value}");
    }
    ensure!(
        first_token_ready_ms <= ttft_ms
            && ttft_ms <= inference_complete_ms
            && inference_complete_ms <= total_request_ms,
        "request timing milestones are out of order"
    );
    ensure!(
        transition_count.checked_add(1) == Some(generated_tokens),
        "expected N-1 transitions for N generated tokens"
    );
    Ok(())
}

fn ensure_close_ms(actual: f64, expected: f64, label: &str) -> Result<()> {
    ensure!(
        (actual - expected).abs() <= 0.001,
        "{label} does not reconcile: actual={actual:.9} expected={expected:.9}"
    );
    Ok(())
}

fn validate_sampling_attribution_row(row: &RequestTimingRow) -> Result<()> {
    let Some(attribution) = row.sampling_attribution.as_ref() else {
        return Ok(());
    };
    let sampling = row
        .sampling
        .as_ref()
        .context("sampling attribution requires sampling telemetry")?;
    ensure!(
        row.schema_version == 11,
        "sampling attribution requires schema 11"
    );
    ensure!(
        row.decode_policy == "sampled_cpu"
            && row.stop_reason == StopReason::TokenLimit
            && row.generated_tokens == 128
            && row.transition_count == 127
            && sampling.draws == 128
            && row.runtime_model_id == "e6024ce53109fdf7"
            && row.runtime_tokenizer_id == "a4b0b26f8a8c9917"
            && row.greedy_gpu_selection_reason == "ineligible_request",
        "sampling attribution request shape or terminal semantics changed"
    );
    ensure!(
        sampling.algorithm_version == 1,
        "sampling attribution requires sampler algorithm version 1"
    );
    ensure!(
        attribution.version == 1
            && attribution.prompt_token_ids_sha256
                == "fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f",
        "sampling attribution version or prompt-token identity changed"
    );
    let clock = &attribution.clock_probe;
    ensure!(
        clock.batches == 7
            && clock.iterations_per_batch == 100_000
            && clock.new_timer_spans == 1_662,
        "sampling clock-probe shape changed"
    );
    for value in clock.pair_ns {
        ensure!(
            value.is_finite() && value >= 0.0,
            "invalid clock-pair observation {value}"
        );
    }
    let expected_upper = clock.pair_ns.iter().copied().fold(0.0f64, f64::max).ceil();
    ensure_close_ms(
        clock.upper_pair_ns / 1e6,
        expected_upper / 1e6,
        "clock upper bound",
    )?;
    ensure_close_ms(
        clock.observer_upper_ms,
        clock.upper_pair_ns * clock.new_timer_spans as f64 / 1e6,
        "clock observer bound",
    )?;

    let sampler = &attribution.sampler;
    ensure!(
        sampler.calls == 128
            && sampler.timer_spans == 1_408
            && sampler.input_logits_total == 128 * 248_320
            && sampler.input_logits_min == 248_320
            && sampler.input_logits_max == 248_320,
        "sampling attribution call, timer, or logits counts changed"
    );
    for (label, summary) in [
        ("after_top_k", sampler.after_top_k),
        ("after_min_p", sampler.after_min_p),
        ("after_positive_infinity", sampler.after_positive_infinity),
        ("after_top_p", sampler.after_top_p),
        ("candidate_index", sampler.candidate_index),
    ] {
        let calls = sampler.calls;
        ensure!(
            summary.min <= summary.max
                && summary.total >= calls.saturating_mul(summary.min)
                && summary.total <= calls.saturating_mul(summary.max),
            "invalid {label} count summary"
        );
    }
    ensure!(
        sampler.candidate_capacity_bytes_peak > 0
            && sampler.candidate_capacity_bytes_total >= sampler.candidate_capacity_bytes_peak
            && sampler.candidate_capacity_bytes_total
                <= sampler
                    .calls
                    .saturating_mul(sampler.candidate_capacity_bytes_peak)
            && sampler.probability_capacity_bytes_peak > 0
            && sampler.probability_capacity_bytes_total >= sampler.probability_capacity_bytes_peak
            && sampler.probability_capacity_bytes_total
                <= sampler
                    .calls
                    .saturating_mul(sampler.probability_capacity_bytes_peak),
        "invalid sampler capacity-byte accounting"
    );
    let sampler_non_residual = [
        sampler.wall_ms,
        sampler.shape_validation_ms,
        sampler.candidate_alloc_ms,
        sampler.candidate_fill_ms,
        sampler.top_k_order_ms,
        sampler.min_p_ms,
        sampler.positive_infinity_ms,
        sampler.temperature_scale_ms,
        sampler.probability_weights_ms,
        sampler.top_p_ms,
        sampler.categorical_ms,
    ];
    ensure!(
        sampler_non_residual
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            && sampler.residual_ms.is_finite(),
        "invalid sampler attribution duration"
    );
    let sampler_phase_sum = sampler.shape_validation_ms
        + sampler.candidate_alloc_ms
        + sampler.candidate_fill_ms
        + sampler.top_k_order_ms
        + sampler.min_p_ms
        + sampler.positive_infinity_ms
        + sampler.temperature_scale_ms
        + sampler.probability_weights_ms
        + sampler.top_p_ms
        + sampler.categorical_ms
        + sampler.residual_ms;
    ensure_close_ms(sampler.wall_ms, sampler_phase_sum, "sampler phase sum")?;

    let transitions = &attribution.transitions;
    ensure!(
        transitions.calls == 127
            && transitions.new_timer_spans == 254
            && transitions.logits_bytes_per_call == 993_280
            && transitions.logits_bytes_total == 126_146_560,
        "sampling attribution transition or logits-byte counts changed"
    );
    let transition_non_residual = [
        transitions.outer_wall_ms,
        transitions.inner_wall_ms,
        transitions.cpu_encode_ms,
        transitions.completion_wait_ms,
        transitions.gpu_ms_nested,
        transitions.logits_alloc_zero_ms,
        transitions.logits_copy_ms,
    ];
    ensure!(
        transition_non_residual
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            && transitions.inner_residual_ms.is_finite()
            && transitions.outer_wrapper_advance_ms.is_finite(),
        "invalid transition attribution duration"
    );
    ensure_close_ms(
        transitions.inner_wall_ms,
        transitions.cpu_encode_ms
            + transitions.completion_wait_ms
            + transitions.logits_alloc_zero_ms
            + transitions.logits_copy_ms
            + transitions.inner_residual_ms,
        "inner transition phase sum",
    )?;
    ensure_close_ms(
        transitions.outer_wall_ms,
        transitions.inner_wall_ms + transitions.outer_wrapper_advance_ms,
        "outer transition phase sum",
    )?;
    ensure_close_ms(
        transitions.outer_wall_ms,
        row.transition_ms,
        "request and attribution transition wall",
    )?;
    ensure!(
        transitions.gpu_ms_nested <= transitions.completion_wait_ms + 0.001,
        "nested GPU wall exceeds completion wait"
    );

    let bounds = attribution.bounds;
    let expected_workspace = transitions.logits_alloc_zero_ms + sampler.candidate_alloc_ms;
    let expected_borrowed = transitions.logits_alloc_zero_ms + transitions.logits_copy_ms;
    let expected_combined = expected_borrowed + sampler.candidate_alloc_ms;
    let expected_structural =
        expected_combined + sampler.candidate_fill_ms + sampler.top_k_order_ms;
    for (label, actual, expected) in [
        ("workspace raw", bounds.workspace_raw_ms, expected_workspace),
        ("borrowed raw", bounds.borrowed_raw_ms, expected_borrowed),
        ("combined raw", bounds.combined_raw_ms, expected_combined),
        (
            "structural raw",
            bounds.structural_raw_ms,
            expected_structural,
        ),
    ] {
        ensure_close_ms(actual, expected, label)?;
    }
    for (label, raw, adjusted, fraction) in [
        (
            "workspace",
            bounds.workspace_raw_ms,
            bounds.workspace_adjusted_ms,
            bounds.workspace_adjusted_fraction,
        ),
        (
            "borrowed",
            bounds.borrowed_raw_ms,
            bounds.borrowed_adjusted_ms,
            bounds.borrowed_adjusted_fraction,
        ),
        (
            "combined",
            bounds.combined_raw_ms,
            bounds.combined_adjusted_ms,
            bounds.combined_adjusted_fraction,
        ),
        (
            "structural",
            bounds.structural_raw_ms,
            bounds.structural_adjusted_ms,
            bounds.structural_adjusted_fraction,
        ),
    ] {
        let expected_adjusted = adjusted_bound(raw, bounds.observer_upper_ms);
        ensure_close_ms(adjusted, expected_adjusted, &format!("{label} adjusted"))?;
        let expected_fraction = bound_fraction(expected_adjusted, row.generation_ms);
        ensure!(
            (fraction - expected_fraction).abs() <= 1e-9,
            "{label} adjusted fraction does not reconcile"
        );
    }
    for value in [
        bounds.observer_upper_ms,
        bounds.workspace_raw_ms,
        bounds.workspace_adjusted_ms,
        bounds.workspace_adjusted_fraction,
        bounds.borrowed_raw_ms,
        bounds.borrowed_adjusted_ms,
        bounds.borrowed_adjusted_fraction,
        bounds.combined_raw_ms,
        bounds.combined_adjusted_ms,
        bounds.combined_adjusted_fraction,
        bounds.structural_raw_ms,
        bounds.structural_adjusted_ms,
        bounds.structural_adjusted_fraction,
    ] {
        ensure!(
            value.is_finite() && value >= 0.0,
            "invalid sampling attribution bound"
        );
    }
    ensure_close_ms(
        bounds.observer_upper_ms,
        clock.observer_upper_ms,
        "bound observer overhead",
    )?;
    Ok(())
}

fn validate_sampled_structural_row(row: &RequestTimingRow) -> Result<()> {
    let Some(structural) = row.sampled_structural.as_ref() else {
        ensure!(
            row.schema_version != 12,
            "schema 12 requires sampled structural telemetry"
        );
        return Ok(());
    };
    let sampling = row
        .sampling
        .as_ref()
        .context("sampled structural telemetry requires sampling telemetry")?;
    ensure!(
        SAMPLER_ALGORITHM_VERSION == 1
            && structural.algorithm_version == 1
            && sampling.algorithm_version == 1,
        "sampled structural requires sampler algorithm version 1"
    );
    ensure!(
        row.schema_version == 12
            && row.sampling_attribution.is_none()
            && row.decode_policy == "sampled_cpu"
            && sampling.draws == row.generated_tokens,
        "sampled structural schema or sampling contract changed"
    );
    ensure!(
        structural.version == 1 && structural.path == "bounded_topk_borrowed_transitions",
        "sampled structural version or path changed"
    );
    ensure!(
        structural.prompt_owned_bounded_calls == 1
            && structural.borrowed_transition_calls
                == u64::try_from(row.transition_count)
                    .context("transition count does not fit u64")?
            && structural.resident_head_wait_calls == structural.borrowed_transition_calls
            && structural.validated_shared_row_calls == structural.borrowed_transition_calls
            && structural.fallback_calls == 0,
        "sampled structural call accounting changed"
    );
    let calls = structural
        .prompt_owned_bounded_calls
        .checked_add(structural.borrowed_transition_calls)
        .context("sampled structural call count overflow")?;
    let generated_tokens =
        u64::try_from(row.generated_tokens).context("generated token count does not fit u64")?;
    let sampling_top_k =
        u64::try_from(sampling.top_k).context("sampling top-k does not fit u64")?;
    ensure!(
        calls == generated_tokens
            && structural.input_logits_min > 0
            && structural.input_logits_min == structural.input_logits_max
            && structural.retained_top_k_min > 0
            && structural.retained_top_k_min == structural.retained_top_k_max
            && structural.retained_top_k_min == sampling_top_k,
        "sampled structural support summaries changed"
    );
    ensure!(
        structural.input_logits_total
            == calls
                .checked_mul(structural.input_logits_min)
                .context("sampled structural input-logits product overflow")?
            && structural.retained_top_k_total
                == calls
                    .checked_mul(structural.retained_top_k_min)
                    .context("sampled structural retained-top-k product overflow")?
            && structural.max_heap_len == structural.retained_top_k_max
            && structural.max_heap_capacity >= structural.max_heap_len
            && structural.max_heap_capacity < structural.input_logits_min,
        "sampled structural heap or total accounting changed"
    );
    ensure!(
        structural.full_candidate_vector_allocations == 0
            && structural.transition_logits_copy_bytes == 0
            && structural.extra_command_buffers == 0
            && structural.gpu_sampling_dispatches == 0,
        "sampled structural path added excluded work"
    );
    Ok(())
}

#[derive(Debug, Serialize)]
struct RequestOutput {
    id: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    generated_token_sha256: String,
    generated_text: String,
    stop_reason: StopReason,
    terminal_token_target_transition_consumed: bool,
}

#[derive(Debug, Serialize)]
struct RequestStatsRow {
    schema_version: u32,
    id: String,
    line: usize,
    model: String,
    greedy_gpu_selection_reason: &'static str,
    arrival_ms: u64,
    finish_ms: u64,
    prompt_tokens: usize,
    prompt_hash: String,
    requested_tokens: usize,
    generated_tokens: usize,
    generated_token_sha256: String,
    decode_policy: &'static str,
    stop_reason: StopReason,
    terminal_token_target_transition_consumed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    sampling: Option<SamplingTelemetry>,
    cache_prefix_tokens: Option<usize>,
    cache_prefix_source: String,
    cache_prefix_hash: Option<String>,
    auto_cache_prefix_tokens: Option<usize>,
    auto_cache_future_hits: usize,
    cache_hit: bool,
    matched_prefix_tokens: usize,
    matched_prefix_hash: Option<String>,
    exact_cache_hit: bool,
    prefill_chunk: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_chunk_effective: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_chunk_decision: Option<PrefillChunkDecision>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_attention_query: Option<PrefillAttentionQueryStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_scratch_overlay: Option<PrefillScratchOverlayTimingStats>,
    max_context_tokens: usize,
    no_special_tokens: bool,
    restore_ms: f64,
    prefix_inserted_bytes: u64,
    prefix_insert_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    model_ttft_ms: f64,
    first_token_ms: f64,
    first_token_callback_ms: f64,
    first_decode_ms: f64,
    decode_tps: f64,
    decode_transitions: usize,
    transition_ms: f64,
    transition_tps: f64,
    total_ms: f64,
    cache_entries: usize,
    cache_bytes: u64,
    cache_max_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_lookup: Option<PromptLookupDecodeStats>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct PromptLookupDecodeStats {
    index_build_ms: f64,
    index_update_ms: f64,
    lookup_ms: f64,
    scratch_allocation_ms: f64,
    scratch_allocated_bytes: u64,
    scratch_peak_allocated_bytes: u64,
    serial_ms: f64,
    verify_ms: f64,
    restore_ms: f64,
    attempts: usize,
    abstentions: usize,
    verify_calls: usize,
    restore_calls: usize,
    accepted_drafts: usize,
    drafts_scored: usize,
    physical_target_positions: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let matches = Args::command().get_matches();
    let explicit_options = ExplicitCliOptions::from_matches(&matches);
    let args = Args::from_arg_matches(&matches).expect("validated clap arguments");
    validate_request_timing_mode(&args)?;
    validate_sampling_attribution_mode(&args)?;
    validate_sampled_structural_mode(&args)?;
    validate_durable_prefix_cache_mode(&args)?;
    if args.info {
        let runtime = Runtime::metal()?;
        println!("device: {}", runtime.describe());
        return Ok(());
    }

    let Some(model_path) = args.model.as_ref() else {
        eprintln!(
            "usage: qwen -m <path-to-gguf> (-p <prompt> | --messages <file>)  (or `qwen --info`)"
        );
        std::process::exit(2);
    };

    if args.deepseek_census_json {
        return print_deepseek_v4_census(model_path);
    }

    if args.prompt.is_none()
        && args.prompt_file.is_none()
        && args.messages.is_none()
        && args.requests_jsonl.is_none()
    {
        return print_model_info(model_path);
    }

    let staged_integrity = configured_checkpoint_staged_integrity()?;
    ensure!(
        staged_integrity.is_none() || args.durable_prefix_cache.is_some(),
        "{CHECKPOINT_STAGED_INTEGRITY_ENV} requires --durable-prefix-cache"
    );
    validate_request_before_model_open(&args)?;
    let gguf = GgufFile::open(model_path)
        .with_context(|| format!("open model {}", model_path.display()))?;
    if ModelFamily::detect(&gguf) == Some(ModelFamily::DeepSeek4) {
        return run_deepseek_v4_single_turn(model_path, gguf, &args, explicit_options);
    }

    if args.prompt.is_some() || args.prompt_file.is_some() || args.messages.is_some() {
        return run_single_turn(model_path, gguf, &args, staged_integrity);
    }

    if let Some(path) = args.requests_jsonl.as_ref() {
        return run_requests_jsonl(model_path, path, gguf, &args);
    }

    unreachable!("request mode was validated above")
}

fn validate_request_timing_mode(args: &Args) -> Result<()> {
    ensure!(
        !args.request_timing_warm_followup || args.request_timings.is_some(),
        "--request-timing-warm-followup requires --request-timings"
    );
    let Some(path) = args.request_timings.as_ref() else {
        return Ok(());
    };
    ensure!(!args.info, "--request-timings cannot be used with --info");
    ensure!(args.model.is_some(), "--request-timings requires --model");
    ensure!(
        args.requests_jsonl.is_none(),
        "--request-timings currently supports single-turn prompts only"
    );
    ensure!(
        args.prompt.is_some() || args.prompt_file.is_some() || args.messages.is_some(),
        "--request-timings requires --prompt, --prompt-file, or --messages"
    );
    ensure!(
        path != Path::new("-"),
        "--request-timings requires a file path, not stdout"
    );
    Ok(())
}

fn validate_sampling_attribution_mode(args: &Args) -> Result<()> {
    if !args.sampling_attribution {
        return Ok(());
    }
    ensure!(
        args.request_timings.is_some(),
        "--sampling-attribution requires --request-timings"
    );
    ensure!(
        args.prompt_file.is_some() && args.prompt.is_none() && args.messages.is_none(),
        "--sampling-attribution requires one --prompt-file request"
    );
    ensure!(
        args.requests_jsonl.is_none()
            && !args.request_timing_warm_followup
            && !args.prompt_lookup
            && args.durable_prefix_cache.is_none(),
        "--sampling-attribution is incompatible with JSONL, warm follow-up, prompt lookup, and durable cache"
    );
    ensure!(
        args.tokens == 128,
        "--sampling-attribution requires --tokens 128"
    );
    ensure!(
        args.temperature.to_bits() == 0.7f32.to_bits()
            && args.top_k == 200
            && args.top_p.to_bits() == 1.0f32.to_bits()
            && args.min_p.to_bits() == 0.05f32.to_bits()
            && args.seed == 42,
        "--sampling-attribution requires sampler-v1 qwen-chat parameters"
    );
    ensure!(
        args.prefill_chunk == PrefillChunkArg::Fixed(1024) && args.max_context_tokens == Some(1024),
        "--sampling-attribution requires chunk and context 1024"
    );
    ensure!(
        args.prefix_cache_max_mib == 0
            && args.cache_prefix_tokens.is_none()
            && args.cache_prefix_auto_min_tokens == 0,
        "--sampling-attribution requires zero prefix-cache admission"
    );
    ensure!(
        !args.no_special_tokens,
        "--sampling-attribution requires add_special_tokens=true"
    );
    ensure!(
        !std::io::stdout().is_terminal(),
        "--sampling-attribution requires redirected stdout"
    );
    let qwen_environment: Vec<_> = std::env::vars_os()
        .filter_map(|(key, value)| {
            let key_text = key.to_string_lossy();
            (key_text.starts_with("QWEN_") && !key_text.starts_with("QWEN_BUILD_"))
                .then_some((key, value))
        })
        .collect();
    ensure!(
        qwen_environment.is_empty(),
        "--sampling-attribution rejects inherited non-build QWEN_* variables"
    );
    Ok(())
}

fn validate_sampled_structural_mode(args: &Args) -> Result<()> {
    if !args.sampled_structural {
        return Ok(());
    }
    ensure!(
        args.prompt.is_some() || args.prompt_file.is_some() || args.messages.is_some(),
        "--sampled-structural requires one single-turn prompt"
    );
    ensure!(
        args.requests_jsonl.is_none()
            && !args.request_timing_warm_followup
            && !args.prompt_lookup
            && !args.sampling_attribution
            && args.durable_prefix_cache.is_none(),
        concat!(
            "--sampled-structural is incompatible with JSONL, warm follow-up, ",
            "prompt lookup, sampling attribution, and durable cache"
        )
    );
    ensure!(
        args.temperature > 0.0 && args.top_k > 0,
        "--sampled-structural requires positive temperature and top-k"
    );
    ensure!(
        args.prefix_cache_max_mib == 0
            && args.cache_prefix_tokens.is_none()
            && args.cache_prefix_auto_min_tokens == 0,
        "--sampled-structural requires zero RAM prefix-cache admission"
    );
    Ok(())
}

fn validate_durable_prefix_cache_mode(args: &Args) -> Result<()> {
    let Some(_) = args.durable_prefix_cache.as_ref() else {
        return Ok(());
    };
    ensure!(
        !args.info,
        "--durable-prefix-cache cannot be used with --info"
    );
    ensure!(
        args.prompt.is_some() || args.prompt_file.is_some() || args.messages.is_some(),
        "--durable-prefix-cache currently supports single-turn prompts only"
    );
    ensure!(
        args.requests_jsonl.is_none(),
        "--durable-prefix-cache does not yet support --requests-jsonl"
    );
    ensure!(
        args.request_timings.is_none() && !args.request_timing_warm_followup,
        "--durable-prefix-cache does not yet support --request-timings"
    );
    ensure!(
        args.durable_prefix_cache_max_mib > 0,
        "--durable-prefix-cache-max-mib must be >= 1"
    );
    ensure!(
        args.durable_prefix_cache_max_entry_mib > 0,
        "--durable-prefix-cache-max-entry-mib must be >= 1"
    );
    ensure!(
        args.durable_prefix_cache_max_entry_mib <= args.durable_prefix_cache_max_mib,
        "durable prefix per-entry budget cannot exceed aggregate budget"
    );
    Ok(())
}

fn validate_request_before_model_open(args: &Args) -> Result<()> {
    args.prefill_chunk.validate()?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    let sampling = cli_sampling_config(args)?;
    if args.requests_jsonl.is_none() {
        validate_sampling_decode_policy(sampling, args.prompt_lookup)?;
    }
    Ok(())
}

fn prompt_text(args: &Args) -> Result<(String, PromptSource, bool)> {
    if let Some(prompt) = args.prompt.as_ref() {
        return Ok((prompt.clone(), PromptSource::Inline, false));
    }
    if let Some(path) = args.prompt_file.as_ref() {
        return Ok((
            std::fs::read_to_string(path)
                .with_context(|| format!("read prompt file {}", path.display()))?,
            PromptSource::File,
            false,
        ));
    }
    if let Some(path) = args.messages.as_ref() {
        let (prompt, preserves_assistant_content) = load_messages_prompt_with_policy(
            path,
            args.messages_max,
            messages_thinking_mode(
                args.messages_preserve_thinking,
                args.messages_strip_thinking,
            ),
            !args.messages_no_generation_prompt,
        )?;
        return Ok((
            prompt,
            PromptSource::Messages,
            preserves_assistant_content && !args.messages_no_generation_prompt,
        ));
    }
    bail!("single-turn generation requires --prompt, --prompt-file, or --messages")
}

fn prompt_add_special_tokens(args: &Args, source: PromptSource) -> bool {
    source != PromptSource::Messages && !args.no_special_tokens
}

fn validate_deepseek_v4_generation_mode(args: &Args, explicit: ExplicitCliOptions) -> Result<()> {
    let mut unsupported = Vec::new();
    if args.requests_jsonl.is_some() {
        unsupported.push("--requests-jsonl");
    }
    if args.prompt_lookup {
        unsupported.push("--prompt-lookup");
    }
    if explicit.prefill_chunk || args.prefill_chunk != PrefillChunkArg::Fixed(1024) {
        unsupported.push("--prefill-chunk");
    }
    if args.max_context_tokens.is_some() {
        unsupported.push("--max-context-tokens");
    }
    if explicit.prefix_cache_max_mib || args.prefix_cache_max_mib != 16 * 1024 {
        unsupported.push("--prefix-cache-max-mib");
    }
    if args.cache_prefix_tokens.is_some() {
        unsupported.push("--cache-prefix-tokens");
    }
    if explicit.cache_prefix_auto_min_tokens || args.cache_prefix_auto_min_tokens != 1024 {
        unsupported.push("--cache-prefix-auto-min-tokens");
    }
    if args.durable_prefix_cache.is_some() {
        unsupported.push("--durable-prefix-cache");
    }
    if explicit.durable_prefix_cache_max_mib || args.durable_prefix_cache_max_mib != 32 * 1024 {
        unsupported.push("--durable-prefix-cache-max-mib");
    }
    if explicit.durable_prefix_cache_max_entry_mib
        || args.durable_prefix_cache_max_entry_mib != 16 * 1024
    {
        unsupported.push("--durable-prefix-cache-max-entry-mib");
    }
    if explicit.durable_prefix_cache_min_tokens || args.durable_prefix_cache_min_tokens != 1024 {
        unsupported.push("--durable-prefix-cache-min-tokens");
    }
    if args.request_stats.is_some() {
        unsupported.push("--request-stats");
    }
    if args.request_timings.is_some() {
        unsupported.push("--request-timings");
    }
    if args.request_timing_warm_followup {
        unsupported.push("--request-timing-warm-followup");
    }
    if args.messages_preserve_thinking {
        unsupported.push("--messages-preserve-thinking");
    }
    if args.messages_strip_thinking {
        unsupported.push("--messages-strip-thinking");
    }
    if args.messages_no_generation_prompt {
        unsupported.push("--messages-no-generation-prompt");
    }
    ensure!(
        unsupported.is_empty(),
        "DeepSeek V4 currently supports bounded raw or ordinary-message single-turn generation only; unsupported options: {}",
        unsupported.join(", ")
    );
    ensure!(
        args.prompt.is_some() || args.prompt_file.is_some() || args.messages.is_some(),
        "DeepSeek V4 generation requires --prompt, --prompt-file, or --messages"
    );
    Ok(())
}

fn deepseek_v4_required_forwards(prompt_tokens: usize, max_tokens: usize) -> Result<usize> {
    ensure!(prompt_tokens > 0, "prompt tokenized to zero tokens");
    ensure!(max_tokens > 0, "--tokens must be >= 1");
    let decode_transitions = max_tokens
        .checked_sub(1)
        .context("DeepSeek V4 decode transition count underflow")?;
    let required = prompt_tokens
        .checked_add(decode_transitions)
        .context("DeepSeek V4 forward budget overflow")?;
    ensure!(
        required <= DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
        "DeepSeek V4 request requires {required} token forwards ({prompt_tokens} prompt + {decode_transitions} maximum decode transitions), but the native session is promoted for {} forwards; shorten the prompt or reduce --tokens",
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
    );
    Ok(required)
}

fn deepseek_v4_packed_chunk_count(prompt_tokens: usize) -> usize {
    if prompt_tokens < 2 {
        0
    } else {
        prompt_tokens.div_ceil(DEEPSEEK_V4_PREFILL_MAX_TOKENS)
    }
}

fn checked_deepseek_v4_token_id(token: i32, vocab_size: u32, purpose: &str) -> Result<u32> {
    let token =
        u32::try_from(token).with_context(|| format!("{purpose} token ID {token} is negative"))?;
    ensure!(
        token < vocab_size,
        "{purpose} token ID {token} is outside vocabulary {vocab_size}"
    );
    Ok(token)
}

fn copy_deepseek_v4_logits(
    session: &DeepSeekV4Session,
    vocab_size: u32,
    purpose: &str,
) -> Result<Vec<f32>> {
    let logits = session
        .copy_logits_f32()
        .with_context(|| format!("copy {purpose} DeepSeek V4 logits"))?;
    ensure!(
        logits.len() == vocab_size as usize,
        "{purpose} DeepSeek V4 logits length {} differs from vocabulary {vocab_size}",
        logits.len(),
    );
    Ok(logits)
}

fn run_deepseek_v4_single_turn(
    model_path: &Path,
    gguf: GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<()> {
    validate_deepseek_v4_generation_mode(args, explicit)?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    let sampling = cli_sampling_config(args)?;
    let arrival_ms = unix_epoch_ms()?;
    let (prompt, prompt_source) = if let Some(path) = args.messages.as_ref() {
        (
            load_deepseek_v4_0731_messages_prompt(path, args.messages_max)
                .context("render DeepSeek V4 0731 messages")?,
            PromptSource::Messages,
        )
    } else {
        let (prompt, source, _) = prompt_text(args)?;
        (prompt, source)
    };
    let prompt_kind = match prompt_source {
        PromptSource::Inline | PromptSource::File => "raw",
        PromptSource::Messages => "messages_0731_chat",
    };

    let tokenizer_t0 = Instant::now();
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load DeepSeek V4 tokenizer")?;
    let tokenizer_ms = tokenizer_t0.elapsed().as_secs_f64() * 1e3;
    let prompt_ids = tokenizer
        .encode(&prompt, false)
        .context("tokenize raw DeepSeek V4 prompt")?;
    let required_forwards = deepseek_v4_required_forwards(prompt_ids.len(), args.tokens)?;
    let vocab_size = tokenizer.n_vocab();
    let prompt_token_ids = prompt_ids
        .iter()
        .enumerate()
        .map(|(index, &token)| {
            checked_deepseek_v4_token_id(token, vocab_size, &format!("prompt[{index}]"))
        })
        .collect::<Result<Vec<_>>>()?;
    let stop_tokens = gguf
        .stop_token_ids()
        .context("load producer-declared DeepSeek V4 stop tokens")?;
    for &token in &stop_tokens {
        checked_deepseek_v4_token_id(token, vocab_size, "stop")?;
    }

    eprintln!(
        "deepseek_v4: loading {} for generation; prompt_kind={} prompt_tokens={} max_generated_tokens={} reserved_forwards={}/{}",
        model_path.display(),
        prompt_kind,
        prompt_ids.len(),
        args.tokens,
        required_forwards,
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
    );
    let load_t0 = Instant::now();
    let ctx = MetalContext::new().context("init Metal context for DeepSeek V4")?;
    let load_plan = DeepSeekV4MetalResidency::plan(&ctx, &gguf)
        .context("plan strict DeepSeek V4 Metal residency and session")?;
    let memory_plan = load_plan.memory_plan().clone();
    let initial_memory_signals = ctx.memory_signals();
    eprintln!("deepseek_v4: memory plan; {memory_plan}");
    let admitted_load_plan = load_plan
        .admit(initial_memory_signals)
        .context("admit strict DeepSeek V4 Metal residency and session")?;
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted_load_plan)
        .context("load admitted strict DeepSeek V4 Metal residency")?;
    let (residency, memory_admission, after_residency_bytes) = realized.into_parts();
    let memory_signals = memory_admission.signals;
    eprintln!(
        "deepseek_v4: memory admission admitted={} reason={} recommended={} current={} process_remaining={:?} working_set_headroom={:?} required={:?}",
        memory_admission.admitted,
        memory_admission.reason.as_str(),
        memory_signals.recommended_max_bytes,
        memory_signals.current_allocated_bytes,
        memory_signals.process_limit_remaining_bytes,
        memory_admission.working_set_headroom_bytes,
        memory_admission.required_bytes,
    );
    let before_residency_bytes = memory_signals.current_allocated_bytes;
    memory_plan
        .reconcile_residency(before_residency_bytes, after_residency_bytes)
        .context("reconcile DeepSeek V4 residency allocation")?;
    ensure!(
        residency.config().vocab_size == vocab_size,
        "DeepSeek V4 tokenizer vocabulary {} differs from resident model vocabulary {}",
        vocab_size,
        residency.config().vocab_size,
    );
    let residency_report = residency.report().clone();
    let mut session =
        DeepSeekV4Session::new(&ctx, residency).context("create DeepSeek V4 session")?;
    let after_session_bytes = ctx.current_allocated_size();
    memory_plan
        .reconcile_session(
            before_residency_bytes,
            after_residency_bytes,
            after_session_bytes,
        )
        .context("reconcile DeepSeek V4 session allocation")?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "deepseek_v4: resident on {} in {:.1} ms; {}",
        ctx.describe(),
        load_ms,
        residency_report,
    );

    let prefill_t0 = Instant::now();
    let packed_chunk_count = deepseek_v4_packed_chunk_count(prompt_token_ids.len());
    let prefill_mode = if packed_chunk_count > 0 {
        for (chunk_index, chunk) in prompt_token_ids
            .chunks(DEEPSEEK_V4_PREFILL_MAX_TOKENS)
            .enumerate()
        {
            if chunk_index + 1 == packed_chunk_count {
                session.prefill_tokens(&ctx, chunk).with_context(|| {
                    format!("prefill final DeepSeek V4 prompt chunk {chunk_index}")
                })?;
            } else {
                session.advance_tokens(&ctx, chunk).with_context(|| {
                    format!("advance DeepSeek V4 prompt chunk {chunk_index} without logits")
                })?;
            }
        }
        if packed_chunk_count == 1 {
            "layer_major_128"
        } else {
            "layer_major_128_chunks"
        }
    } else {
        for (index, &token) in prompt_token_ids.iter().enumerate() {
            session
                .forward_token(&ctx, token)
                .with_context(|| format!("forward DeepSeek V4 prompt token {index}"))?;
        }
        "singleton"
    };
    let reconciliation = memory_plan
        .reconcile(DeepSeekV4MemorySamples {
            before_residency_bytes,
            after_residency_bytes,
            after_session_bytes,
            after_first_forward_bytes: ctx.current_allocated_size(),
        })
        .context("reconcile admitted DeepSeek V4 Metal memory")?;
    eprintln!("deepseek_v4: memory reconciliation; {reconciliation}");
    let logits = copy_deepseek_v4_logits(&session, vocab_size, "prompt")?;
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

    let mut sampler = Sampler::new(sampling).context("initialize DeepSeek V4 sampler")?;
    let stdout_handle = std::io::stdout();
    let mut stdout = stdout_handle.lock();
    let generation = generate_serial(
        logits,
        args.tokens,
        &stop_tokens,
        &mut sampler,
        |token| {
            let piece = tokenizer
                .try_decode_piece_bytes_exact(token)
                .with_context(|| format!("decode DeepSeek V4 token {token}"))?;
            stdout
                .write_all(piece)
                .with_context(|| format!("write DeepSeek V4 token {token}"))?;
            stdout.flush().context("flush DeepSeek V4 token")?;
            Ok(())
        },
        |token| {
            let token = checked_deepseek_v4_token_id(token, vocab_size, "generated")?;
            session
                .forward_token(&ctx, token)
                .context("forward generated DeepSeek V4 token")?;
            copy_deepseek_v4_logits(&session, vocab_size, "continuing")
        },
    )?;
    drop(stdout);

    let decode_tps = if generation.wall_ms > 0.0 {
        generation.tokens.len() as f64 / (generation.wall_ms / 1e3)
    } else {
        0.0
    };
    let transition_tps = if generation.transition_ms > 0.0 {
        generation.transitions as f64 / (generation.transition_ms / 1e3)
    } else {
        0.0
    };
    eprintln!(
        concat!(
            "deepseek_v4 stats: prompt_kind={} prefill_mode={} prompt_tokens={} generated_tokens={} transitions={} ",
            "stop_reason={} tokenizer_ms={:.1} load_ms={:.1} prefill_ms={:.1} ",
            "generation_ms={:.1} decode_tps={:.2} transition_tps={:.2} generated_ids={:?}"
        ),
        prompt_kind,
        prefill_mode,
        prompt_ids.len(),
        generation.tokens.len(),
        generation.transitions,
        generation.stop_reason.as_str(),
        tokenizer_ms,
        load_ms,
        prefill_ms,
        generation.wall_ms,
        decode_tps,
        transition_tps,
        generation.tokens,
    );
    if let Some(path) = args.trace_request.as_ref() {
        append_request_trace(path, arrival_ms, prompt_ids.len(), generation.tokens.len())?;
    }
    Ok(())
}

fn cli_sampling_config(args: &Args) -> Result<SamplingConfig> {
    SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
    }
    .validate()
    .map_err(anyhow::Error::new)
    .context("validate CLI sampling configuration")
}

fn request_sampling_config(request: &JsonlRequest, args: &Args) -> Result<SamplingConfig> {
    let request = request.sampling.unwrap_or_default();
    SamplingConfig {
        temperature: request.temperature.unwrap_or(args.temperature),
        top_k: request.top_k.unwrap_or(args.top_k),
        top_p: request.top_p.unwrap_or(args.top_p),
        min_p: request.min_p.unwrap_or(args.min_p),
        seed: request.seed.unwrap_or(args.seed),
    }
    .validate()
    .map_err(anyhow::Error::new)
    .context("validate request sampling configuration")
}

fn validate_sampling_decode_policy(config: SamplingConfig, prompt_lookup: bool) -> Result<()> {
    ensure!(
        !prompt_lookup || config.temperature == 0.0,
        "--prompt-lookup currently requires greedy decoding (--temp 0)"
    );
    Ok(())
}

fn decode_policy_label(
    config: SamplingConfig,
    prompt_lookup: bool,
    gpu_greedy: bool,
) -> &'static str {
    if prompt_lookup {
        "prompt_lookup_l8_d7_target_n8"
    } else if config.temperature > 0.0 {
        "sampled_cpu"
    } else if gpu_greedy {
        "greedy_gpu_argmax"
    } else {
        "greedy_argmax"
    }
}

fn jsonl_decode_policy_label(
    config: SamplingConfig,
    prompt_lookup: bool,
    gpu_greedy: bool,
) -> &'static str {
    decode_policy_label(config, prompt_lookup, gpu_greedy)
}

const AUTO_CHUNK_PROMPT_MIN: usize = 8192;
const AUTO_CHUNK_PROMPT_MAX: usize = 16384;
const AUTO_CHUNK_QUERY_ROWS: usize = 1024;
const AUTO_CHUNK_TRANSIENT_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
const AUTO_CHUNK_BASELINE: &str = "legacy_outer_1024_matrix_max_pos_v1";

fn request_schema_version(
    prefill_chunk: PrefillChunkArg,
    prompt_lookup: bool,
    has_query_topology: bool,
    has_scratch_overlay: bool,
    sampled: bool,
    sampling_attribution: bool,
    sampled_structural: bool,
) -> u32 {
    if sampled_structural {
        12
    } else if sampling_attribution {
        11
    } else if sampled {
        10
    } else if prefill_chunk.is_auto() {
        9
    } else if prompt_lookup || has_query_topology || has_scratch_overlay {
        8
    } else {
        7
    }
}

fn auto_prefill_cache_safe(cache_entries: usize, cache_prefix_tokens: Option<usize>) -> bool {
    cache_entries == 0 && cache_prefix_tokens.is_none()
}

fn cache_prefix_needs_extension(configured_prefix: usize, restored_prefix: usize) -> bool {
    configured_prefix > restored_prefix
}

fn auto_prefill_profile(
    arch: Arch,
    base_model_name: Option<&str>,
    file_type: Option<u64>,
) -> Option<AutoPrefillProfile> {
    if arch.kind != ArchKind::Moe
        || arch.expert_count != 256
        || arch.expert_used_count != 8
        || arch.full_attention_interval != 4
        || arch.attn_head_dim != 256
        || arch.partial_rotary_factor != 0.25
        || arch.gdn_n_k_heads != 16
        || arch.gdn_head_dim != 128
        || arch.gdn_conv_kernel != 4
        || arch.mtp_n_hidden_layers != 0
        || file_type != Some(15)
    {
        return None;
    }
    match base_model_name {
        Some("Qwen3.6 35B A3B")
            if arch.n_layer == 40
                && arch.hidden_size == 2048
                && arch.n_q_heads == 16
                && arch.n_kv_heads == 2
                && arch.gdn_n_v_heads == 32
                && arch.expert_feed_forward_length == 512
                && arch.expert_shared_feed_forward_length == 512 =>
        {
            Some(AutoPrefillProfile {
                name: "qwen3.6-35b-a3b-filetype15",
                outer_chunk: 2048,
                query_heads: 16,
                gdn_overlay_bytes: 235_405_312,
            })
        }
        Some("Qwen3.5 122B A10B")
            if arch.n_layer == 48
                && arch.hidden_size == 3072
                && arch.n_q_heads == 32
                && arch.n_kv_heads == 2
                && arch.gdn_n_v_heads == 64
                && arch.expert_feed_forward_length == 1024
                && arch.expert_shared_feed_forward_length == 1024 =>
        {
            Some(AutoPrefillProfile {
                name: "qwen3.5-122b-a10b-filetype15",
                outer_chunk: 4096,
                query_heads: 32,
                gdn_overlay_bytes: 807_403_520,
            })
        }
        _ => None,
    }
}

fn baseline_prefill_chunk(prompt_tokens: usize) -> usize {
    1024.min(prompt_tokens.max(1))
}

fn auto_prefill_chunk_decision(
    profile: Option<AutoPrefillProfile>,
    prompt_tokens: usize,
    environment_override: bool,
    cache_safe: bool,
) -> PrefillChunkDecision {
    let baseline = baseline_prefill_chunk(prompt_tokens);
    let (classification, reason, selected, profile_name, candidate) = match profile {
        None => ("baseline", "profile_not_allowlisted", baseline, None, None),
        Some(profile) if prompt_tokens < AUTO_CHUNK_PROMPT_MIN => (
            "baseline",
            "prompt_below_validated_range",
            baseline,
            Some(profile.name),
            Some(profile.outer_chunk),
        ),
        Some(profile) if prompt_tokens > AUTO_CHUNK_PROMPT_MAX => (
            "baseline",
            "prompt_above_memory_bounded_range",
            baseline,
            Some(profile.name),
            Some(profile.outer_chunk),
        ),
        Some(profile) if environment_override => (
            "baseline",
            "prefill_environment_override_present",
            baseline,
            Some(profile.name),
            Some(profile.outer_chunk),
        ),
        Some(profile) if !cache_safe => (
            "baseline",
            "prefix_cache_interaction_unvalidated",
            baseline,
            Some(profile.name),
            Some(profile.outer_chunk),
        ),
        Some(profile) => (
            "candidate",
            "matched_validated_profile",
            profile.outer_chunk.min(prompt_tokens.max(1)),
            Some(profile.name),
            Some(profile.outer_chunk),
        ),
    };
    PrefillChunkDecision {
        policy: "moe_allowlist_v2",
        profile: profile_name,
        classification,
        reason,
        candidate,
        selected,
        validated_prompt_range: profile_name
            .map(|_| [AUTO_CHUNK_PROMPT_MIN, AUTO_CHUNK_PROMPT_MAX]),
        evidence_baseline_chunk: profile_name.map(|_| 1024),
        baseline: AUTO_CHUNK_BASELINE,
        detail: None,
        plan: None,
        admission: None,
    }
}

fn prefill_environment_override_present() -> bool {
    prefill_environment_override_present_in(std::env::vars_os().map(|(key, _)| key))
}

fn prefill_environment_override_present_in<I, K>(keys: I) -> bool
where
    I: IntoIterator<Item = K>,
    K: AsRef<std::ffi::OsStr>,
{
    keys.into_iter()
        .any(|key| is_prefill_environment_key(key.as_ref()))
}

fn is_prefill_environment_key(key: &std::ffi::OsStr) -> bool {
    key.as_encoded_bytes().starts_with(b"QWEN_PREFILL_")
}

fn checked_overlay_alignment(value: u64) -> Result<u64> {
    value
        .checked_add(255)
        .map(|aligned| aligned & !255)
        .context("prefill overlay alignment overflow")
}

fn expected_auto_prefill_overlay(
    profile: AutoPrefillProfile,
    matrix_max_pos: usize,
) -> Result<PrefillScratchOverlayStats> {
    let matrix_max_pos =
        u64::try_from(matrix_max_pos).context("matrix max pos does not fit u64")?;
    let query_rows = AUTO_CHUNK_QUERY_ROWS as u64;
    let score_bytes = 2u64
        .checked_mul(query_rows)
        .and_then(|value| value.checked_mul(profile.query_heads))
        .and_then(|value| value.checked_mul(matrix_max_pos))
        .context("prefill score overlay byte overflow")?;
    let ml_bytes = 8u64
        .checked_mul(query_rows)
        .and_then(|value| value.checked_mul(profile.query_heads))
        .and_then(|value| value.checked_mul(matrix_max_pos.div_ceil(64)))
        .context("prefill sidecar overlay byte overflow")?;
    let attention_bytes = checked_overlay_alignment(
        checked_overlay_alignment(score_bytes)?
            .checked_add(ml_bytes)
            .context("prefill attention overlay byte overflow")?,
    )?;
    let gdn_bytes = profile.gdn_overlay_bytes;
    Ok(PrefillScratchOverlayStats {
        backing_bytes: attention_bytes.max(gdn_bytes),
        attention_bytes,
        gdn_bytes,
        saved_bytes: attention_bytes.min(gdn_bytes),
    })
}

fn overlay_timing_stats(value: PrefillScratchOverlayStats) -> PrefillScratchOverlayTimingStats {
    PrefillScratchOverlayTimingStats {
        backing_bytes: value.backing_bytes,
        attention_bytes: value.attention_bytes,
        gdn_bytes: value.gdn_bytes,
        saved_bytes: value.saved_bytes,
    }
}

fn price_prefill_allocations(
    allocations: impl IntoIterator<Item = (&'static str, bool, u64)>,
    mut price: impl FnMut(u64) -> Result<MetalBufferSizeAndAlign>,
) -> Result<(Vec<PrefillPlanAllocationDecision>, u64, u64, u64)> {
    let mut rows = Vec::new();
    let mut eager_logical_bytes = 0u64;
    let mut deferred_logical_bytes = 0u64;
    let mut priced_upper_bytes = 0u64;
    for (name, deferred, logical_bytes) in allocations {
        let priced = price(logical_bytes)?;
        ensure!(
            priced.size > 0 && priced.size >= logical_bytes && priced.alignment.is_power_of_two(),
            "auto-prefill allocation pricing is invalid"
        );
        if deferred {
            deferred_logical_bytes = deferred_logical_bytes
                .checked_add(logical_bytes)
                .context("deferred prefill logical byte overflow")?;
        } else {
            eager_logical_bytes = eager_logical_bytes
                .checked_add(logical_bytes)
                .context("eager prefill logical byte overflow")?;
        }
        priced_upper_bytes = priced_upper_bytes
            .checked_add(priced.size)
            .context("priced prefill byte overflow")?;
        rows.push(PrefillPlanAllocationDecision {
            name,
            deferred,
            logical_bytes,
            priced_bytes: priced.size,
            alignment: priced.alignment,
        });
    }
    Ok((
        rows,
        eager_logical_bytes,
        deferred_logical_bytes,
        priced_upper_bytes,
    ))
}

fn validate_auto_prefill_plan_topology(
    profile: AutoPrefillProfile,
    prompt_tokens: usize,
    block_size: u32,
    matrix_max_pos: u64,
    matrix_query_rows: u32,
    overlay: Option<PrefillScratchOverlayStats>,
) -> Result<PrefillScratchOverlayStats> {
    let expected_matrix_max_pos = prompt_tokens.max(profile.outer_chunk);
    ensure!(
        block_size == u32::try_from(profile.outer_chunk)?
            && matrix_max_pos == u64::try_from(expected_matrix_max_pos)?
            && matrix_query_rows == u32::try_from(AUTO_CHUNK_QUERY_ROWS)?,
        "auto-prefill candidate plan geometry drifted"
    );
    let expected_overlay = expected_auto_prefill_overlay(profile, expected_matrix_max_pos)?;
    let overlay = overlay.context("auto-prefill candidate plan has no scratch overlay")?;
    ensure!(
        overlay == expected_overlay,
        "auto-prefill candidate overlay geometry drifted"
    );
    Ok(overlay)
}

fn price_prefill_plan(
    ctx: &MetalContext,
    profile: AutoPrefillProfile,
    prompt_tokens: usize,
    plan: &PrefillScratchPlan,
) -> Result<PrefillPlanDecision> {
    let overlay = validate_auto_prefill_plan_topology(
        profile,
        prompt_tokens,
        plan.block_size(),
        plan.matrix_max_pos(),
        plan.matrix_query_rows(),
        plan.overlay(),
    )?;

    plan.allocation_count()
        .checked_add(plan.deferred_allocations().len())
        .context("auto-prefill allocation count overflow")?;
    let plan_allocations = plan
        .allocations()
        .iter()
        .map(|allocation| (allocation.name(), false, allocation.logical_bytes()))
        .chain(
            plan.deferred_allocations()
                .iter()
                .map(|allocation| (allocation.name(), true, allocation.logical_bytes())),
        );
    let (allocations, eager_logical_bytes, deferred_logical_bytes, priced_upper_bytes) =
        price_prefill_allocations(plan_allocations, |logical_bytes| {
            Ok(ctx.shared_buffer_size_and_align(logical_bytes)?)
        })?;
    let maximum_logical_bytes = plan.maximum_logical_bytes()?;
    ensure!(
        eager_logical_bytes.checked_add(deferred_logical_bytes) == Some(maximum_logical_bytes),
        "auto-prefill logical byte totals do not reconcile"
    );
    let independently_priced =
        plan.priced_upper_bound(|bytes| Ok(ctx.shared_buffer_size_and_align(bytes)?.size))?;
    ensure!(
        independently_priced == priced_upper_bytes,
        "auto-prefill priced byte totals do not reconcile"
    );
    Ok(PrefillPlanDecision {
        block_size: plan.block_size(),
        matrix_max_pos: plan.matrix_max_pos(),
        matrix_query_rows: plan.matrix_query_rows(),
        eager_allocation_count: plan.allocation_count(),
        deferred_allocation_count: plan.deferred_allocations().len(),
        eager_logical_bytes,
        deferred_logical_bytes,
        maximum_logical_bytes,
        priced_upper_bytes,
        overlay: overlay_timing_stats(overlay),
        allocations,
    })
}

fn prefill_memory_signals_decision(
    signals: MetalMemorySignals,
    admission: MetalMemoryAdmission,
) -> PrefillMemorySignalsDecision {
    PrefillMemorySignalsDecision {
        recommended_max_bytes: signals.recommended_max_bytes,
        current_allocated_bytes: signals.current_allocated_bytes,
        process_limit_remaining_bytes: signals.process_limit_remaining_bytes,
        working_set_headroom_bytes: admission.working_set_headroom_bytes,
    }
}

fn auto_prefill_reserve(
    before_sequence: u64,
    after_sequence: u64,
) -> std::result::Result<(u64, u64), &'static str> {
    let delta = after_sequence
        .checked_sub(before_sequence)
        .filter(|&delta| delta > 0)
        .ok_or("sequence_allocation_signal_invalid")?;
    let reserve = delta
        .checked_add(AUTO_CHUNK_TRANSIENT_RESERVE_BYTES)
        .ok_or("candidate_reserve_overflow")?;
    Ok((delta, reserve))
}

struct AllocatedPrefillRequestState<Scratch = MetalDFlashLayerMajorScratch, State = Sequence> {
    chunk: usize,
    decision: Option<PrefillChunkDecision>,
    scratch: Scratch,
    sequence: State,
    scratch_allocation_ms: f64,
    sequence_allocation_ms: f64,
    after_scratch_allocated: u64,
    after_sequence_allocated: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CandidatePlanFailure {
    Unavailable(String),
    InvalidOrUnpriceable(String),
}

trait PrefillRequestAllocator {
    type Scratch;
    type Sequence;
    type Plan;

    fn current_allocated_size(&mut self) -> u64;
    fn allocate_legacy_scratch(
        &mut self,
        chunk: usize,
        prompt_tokens: usize,
    ) -> Result<Self::Scratch>;
    fn allocate_sequence(&mut self, capacity: usize) -> Result<Self::Sequence>;
    fn build_candidate_plan(
        &mut self,
        profile: AutoPrefillProfile,
        prompt_tokens: usize,
    ) -> std::result::Result<(Self::Plan, PrefillPlanDecision), CandidatePlanFailure>;
    fn memory_signals(&mut self) -> MetalMemorySignals;
    fn allocate_candidate_scratch(&mut self, plan: Self::Plan) -> Result<Self::Scratch>;
}

struct MetalPrefillRequestAllocator<'a> {
    loaded: &'a LoadedModel,
}

fn allocate_legacy_prefill_scratch(
    loaded: &LoadedModel,
    chunk: usize,
    prompt_tokens: usize,
) -> Result<MetalDFlashLayerMajorScratch> {
    MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        loaded.context(),
        loaded.metal_model(),
        u32::try_from(chunk).context("prefill chunk does not fit u32")?,
        prompt_tokens.max(chunk),
    )
    .context("allocate legacy prefill scratch")
}

impl PrefillRequestAllocator for MetalPrefillRequestAllocator<'_> {
    type Scratch = MetalDFlashLayerMajorScratch;
    type Sequence = Sequence;
    type Plan = PrefillScratchPlan;

    fn current_allocated_size(&mut self) -> u64 {
        self.loaded.context().current_allocated_size()
    }

    fn allocate_legacy_scratch(
        &mut self,
        chunk: usize,
        prompt_tokens: usize,
    ) -> Result<Self::Scratch> {
        allocate_legacy_prefill_scratch(self.loaded, chunk, prompt_tokens)
    }

    fn allocate_sequence(&mut self, capacity: usize) -> Result<Self::Sequence> {
        self.loaded
            .create_sequence(SequenceConfig::new(capacity))
            .map_err(anyhow::Error::from)
    }

    fn build_candidate_plan(
        &mut self,
        profile: AutoPrefillProfile,
        prompt_tokens: usize,
    ) -> std::result::Result<(Self::Plan, PrefillPlanDecision), CandidatePlanFailure> {
        let block_size = u32::try_from(profile.outer_chunk)
            .map_err(|error| CandidatePlanFailure::Unavailable(error.to_string()))?;
        let matrix_max_pos = prompt_tokens.max(profile.outer_chunk);
        let plan = plan_prefill_scratch_with_matrix_max_pos_configured(
            self.loaded.metal_model(),
            block_size,
            matrix_max_pos,
            PrefillScratchConfig {
                matrix_query_cap: Some(AUTO_CHUNK_QUERY_ROWS),
            },
        )
        .map_err(|error| CandidatePlanFailure::Unavailable(error.to_string()))?;
        let decision = price_prefill_plan(self.loaded.context(), profile, prompt_tokens, &plan)
            .map_err(|error| CandidatePlanFailure::InvalidOrUnpriceable(error.to_string()))?;
        Ok((plan, decision))
    }

    fn memory_signals(&mut self) -> MetalMemorySignals {
        self.loaded.context().memory_signals()
    }

    fn allocate_candidate_scratch(&mut self, plan: Self::Plan) -> Result<Self::Scratch> {
        MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(
            self.loaded.context(),
            self.loaded.metal_model(),
            plan,
        )
        .context("allocate admitted auto-prefill scratch")
    }
}

fn allocate_scratch_then_sequence<A: PrefillRequestAllocator>(
    allocator: &mut A,
    capacity: usize,
    chunk: usize,
    prompt_tokens: usize,
    decision: Option<PrefillChunkDecision>,
) -> Result<AllocatedPrefillRequestState<A::Scratch, A::Sequence>> {
    let scratch_t0 = Instant::now();
    let scratch = allocator.allocate_legacy_scratch(chunk, prompt_tokens)?;
    let scratch_allocation_ms = scratch_t0.elapsed().as_secs_f64() * 1e3;
    let after_scratch_allocated = allocator.current_allocated_size();
    let sequence_t0 = Instant::now();
    let sequence = allocator.allocate_sequence(capacity)?;
    let sequence_allocation_ms = sequence_t0.elapsed().as_secs_f64() * 1e3;
    let after_sequence_allocated = allocator.current_allocated_size();
    Ok(AllocatedPrefillRequestState {
        chunk,
        decision,
        scratch,
        sequence,
        scratch_allocation_ms,
        sequence_allocation_ms,
        after_scratch_allocated,
        after_sequence_allocated,
    })
}

#[allow(clippy::too_many_arguments)]
fn allocate_legacy_after_sequence<A: PrefillRequestAllocator>(
    allocator: &mut A,
    prompt_tokens: usize,
    mut decision: PrefillChunkDecision,
    reason: &'static str,
    detail: Option<String>,
    sequence: A::Sequence,
    sequence_allocation_ms: f64,
    after_sequence_allocated: u64,
) -> Result<AllocatedPrefillRequestState<A::Scratch, A::Sequence>> {
    let chunk = baseline_prefill_chunk(prompt_tokens);
    decision.classification = "baseline";
    decision.reason = reason;
    decision.selected = chunk;
    decision.detail = detail;
    let scratch_t0 = Instant::now();
    let scratch = allocator.allocate_legacy_scratch(chunk, prompt_tokens)?;
    let scratch_allocation_ms = scratch_t0.elapsed().as_secs_f64() * 1e3;
    let after_scratch_allocated = allocator.current_allocated_size();
    Ok(AllocatedPrefillRequestState {
        chunk,
        decision: Some(decision),
        scratch,
        sequence,
        scratch_allocation_ms,
        sequence_allocation_ms,
        after_scratch_allocated,
        after_sequence_allocated,
    })
}

fn allocate_prefill_request_state_with<A: PrefillRequestAllocator>(
    allocator: &mut A,
    requested: PrefillChunkArg,
    prompt_tokens: usize,
    capacity: usize,
    cache_safe: bool,
    profile: Option<AutoPrefillProfile>,
    environment_override: bool,
) -> Result<AllocatedPrefillRequestState<A::Scratch, A::Sequence>> {
    if let PrefillChunkArg::Fixed(requested) = requested {
        let chunk = requested.min(prompt_tokens.max(1));
        return allocate_scratch_then_sequence(allocator, capacity, chunk, prompt_tokens, None);
    }

    let mut decision =
        auto_prefill_chunk_decision(profile, prompt_tokens, environment_override, cache_safe);
    if decision.classification != "candidate" {
        return allocate_scratch_then_sequence(
            allocator,
            capacity,
            decision.selected,
            prompt_tokens,
            Some(decision),
        );
    }
    let profile = profile.context("auto-prefill candidate is missing its profile")?;

    let current_allocated_before_sequence = allocator.current_allocated_size();
    let sequence_t0 = Instant::now();
    let sequence = allocator.allocate_sequence(capacity)?;
    let sequence_allocation_ms = sequence_t0.elapsed().as_secs_f64() * 1e3;
    let current_allocated_after_sequence = allocator.current_allocated_size();
    let (sequence_allocation_delta_bytes, reserve_bytes) = match auto_prefill_reserve(
        current_allocated_before_sequence,
        current_allocated_after_sequence,
    ) {
        Ok(value) => value,
        Err(reason) => {
            return allocate_legacy_after_sequence(
                allocator,
                prompt_tokens,
                decision,
                reason,
                None,
                sequence,
                sequence_allocation_ms,
                current_allocated_after_sequence,
            );
        }
    };

    let (plan, plan_decision) = match allocator.build_candidate_plan(profile, prompt_tokens) {
        Ok(value) => value,
        Err(CandidatePlanFailure::Unavailable(detail)) => {
            return allocate_legacy_after_sequence(
                allocator,
                prompt_tokens,
                decision,
                "candidate_plan_unavailable",
                Some(detail),
                sequence,
                sequence_allocation_ms,
                current_allocated_after_sequence,
            );
        }
        Err(CandidatePlanFailure::InvalidOrUnpriceable(detail)) => {
            return allocate_legacy_after_sequence(
                allocator,
                prompt_tokens,
                decision,
                "candidate_plan_invalid_or_unpriceable",
                Some(detail),
                sequence,
                sequence_allocation_ms,
                current_allocated_after_sequence,
            );
        }
    };
    decision.plan = Some(plan_decision.clone());
    let signals = allocator.memory_signals();
    let admission = evaluate_metal_memory_admission(
        plan_decision.priced_upper_bytes,
        reserve_bytes,
        signals,
        true,
    );
    decision.admission = Some(PrefillAdmissionDecision {
        current_allocated_before_sequence,
        current_allocated_after_sequence,
        sequence_allocation_delta_bytes,
        transient_reserve_bytes: AUTO_CHUNK_TRANSIENT_RESERVE_BYTES,
        reserve_bytes,
        required_bytes: admission.required_bytes,
        signals: prefill_memory_signals_decision(signals, admission),
        evaluator_reason: admission.reason.as_str(),
        admitted: admission.admitted,
    });
    if !admission.admitted {
        let reason = if admission.required_bytes.is_none() {
            "candidate_required_bytes_overflow"
        } else {
            "memory_admission_denied"
        };
        return allocate_legacy_after_sequence(
            allocator,
            prompt_tokens,
            decision,
            reason,
            None,
            sequence,
            sequence_allocation_ms,
            current_allocated_after_sequence,
        );
    }

    decision.reason = "admitted";
    let scratch_t0 = Instant::now();
    let scratch = allocator.allocate_candidate_scratch(plan)?;
    let scratch_allocation_ms = scratch_t0.elapsed().as_secs_f64() * 1e3;
    let after_scratch_allocated = allocator.current_allocated_size();
    Ok(AllocatedPrefillRequestState {
        chunk: profile.outer_chunk,
        decision: Some(decision),
        scratch,
        sequence,
        scratch_allocation_ms,
        sequence_allocation_ms,
        after_scratch_allocated,
        after_sequence_allocated: current_allocated_after_sequence,
    })
}

fn allocate_prefill_request_state(
    loaded: &LoadedModel,
    requested: PrefillChunkArg,
    prompt_tokens: usize,
    capacity: usize,
    cache_safe: bool,
) -> Result<AllocatedPrefillRequestState> {
    let profile = auto_prefill_profile(
        loaded.arch(),
        loaded.gguf().get_str("general.base_model.0.name"),
        loaded.gguf().get_u64("general.file_type"),
    );
    let mut allocator = MetalPrefillRequestAllocator { loaded };
    allocate_prefill_request_state_with(
        &mut allocator,
        requested,
        prompt_tokens,
        capacity,
        cache_safe,
        profile,
        prefill_environment_override_present(),
    )
}

fn report_prefill_chunk_decision(decision: Option<&PrefillChunkDecision>, prompt_tokens: usize) {
    let Some(decision) = decision else {
        return;
    };
    eprintln!(
        "prefill_chunk: policy={} profile={} prompt_tokens={} selected={} reason={}",
        decision.policy,
        decision.profile.unwrap_or("none"),
        prompt_tokens,
        decision.selected,
        decision.reason,
    );
}

fn run_single_turn(
    model_path: &Path,
    gguf: GgufFile,
    args: &Args,
    staged_integrity: Option<StagedIntegrityMode>,
) -> Result<()> {
    args.prefill_chunk.validate()?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    let sampling = cli_sampling_config(args)?;
    validate_sampling_decode_policy(sampling, args.prompt_lookup)?;
    let durable_store = durable_checkpoint_store(args, staged_integrity)?;
    let durable_max_record_bytes = if durable_store.is_some() {
        durable_prefix_cache_max_entry_bytes(args)?
    } else {
        0
    };

    let mut timing_file = args
        .request_timings
        .as_ref()
        .map(|path| open_append_file(path, "request timings"))
        .transpose()?;
    let timing_enabled = timing_file.is_some();
    let arrival_ms = unix_epoch_ms()?;
    let load_t0 = Instant::now();
    let runtime = Runtime::metal().context("init Metal runtime")?;
    let loaded = runtime
        .load_open_model_for_disposable_single_turn_with_config(
            gguf,
            LoadedModelConfig {
                prefix_cache_max_bytes: prefix_cache_max_bytes(args)?,
                ..LoadedModelConfig::default()
            },
        )
        .with_context(|| format!("load model {}", model_path.display()))?;
    if args.prompt_lookup {
        ensure_prompt_lookup_n8_supported(loaded.metal_model()).map_err(anyhow::Error::msg)?;
    }
    if args.sampling_attribution {
        let arch = loaded.arch();
        let lm_head = &loaded.metal_model().lm_head;
        ensure!(
            arch.kind == ArchKind::Moe
                && arch.n_layer == 40
                && arch.hidden_size == 2048
                && arch.vocab_size == 248_320
                && loaded.gguf().get_str("general.base_model.0.name") == Some("Qwen3.6 35B A3B")
                && loaded.gguf().get_u64("general.file_type") == Some(15)
                && lm_head.dtype == GgmlType::Q6_K
                && lm_head.shape.as_slice() == [2048, 248_320]
                && std::fs::metadata(model_path)
                    .is_ok_and(|metadata| metadata.len() == 22_134_528_992),
            "--sampling-attribution requires the frozen Qwen3.6 35B A3B profile"
        );
    }
    if args.sampled_structural {
        let vocab = usize::try_from(loaded.arch().vocab_size)
            .context("sampled structural vocabulary does not fit usize")?;
        ensure!(
            sampling.top_k < vocab,
            "--sampled-structural requires top-k smaller than vocabulary"
        );
        ensure!(
            loaded.gguf().get_str("general.base_model.0.name") == Some("Qwen3.6 35B A3B")
                && loaded.gguf().get_u64("general.file_type") == Some(15)
                && std::fs::metadata(model_path)
                    .is_ok_and(|metadata| metadata.len() == 22_134_528_992),
            "--sampled-structural requires the frozen Qwen3.6 35B A3B Q4_K_M profile"
        );
        loaded
            .forward()
            .ensure_sampled_structural_supported()
            .context("validate sampled structural decode organization")?;
    }
    let greedy_gpu_mode = configured_greedy_gpu_argmax_mode();
    let runtime_and_model_load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
    if timing_enabled {
        loaded.context().set_pipeline_cache_metrics_enabled(true);
    }
    let process_model_ready_allocated =
        timing_enabled.then(|| loaded.context().current_allocated_size());
    let stdout_sink = if std::io::stdout().is_terminal() {
        "terminal"
    } else {
        "redirected"
    };
    let pair_id = if args.request_timing_warm_followup {
        Some(format!("{}-{}", std::process::id(), unix_epoch_ms_u64()?))
    } else {
        None
    };
    let sampling_clock_probe = args.sampling_attribution.then(measure_sampling_clock_probe);

    ensure!(
        loaded.prefix_cache_stats().entries == 0,
        "single-turn timing requires an empty prefix cache"
    );
    let first_pipeline_cache_start =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let first_request_t0 = Instant::now();
    let first_request_start_unix_ms = unix_epoch_ms_u64()?;
    let first_request_start_allocated =
        timing_enabled.then(|| loaded.context().current_allocated_size());
    let prompt_t0 = Instant::now();
    let (first_prompt, first_prompt_source, first_completed_checkpoint_eligible) =
        prompt_text(args)?;
    let first_prompt_acquisition_ms = prompt_t0.elapsed().as_secs_f64() * 1e3;
    if args.sampling_attribution {
        ensure!(
            first_prompt.len() == 1_891
                && format!("{:x}", Sha256::digest(first_prompt.as_bytes()))
                    == "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474",
            "--sampling-attribution prompt byte identity changed"
        );
    }
    let tokenizer_t0 = Instant::now();
    let tokenizer = loaded.tokenizer().context("load tokenizer")?;
    let first_tokenizer_init_ms = tokenizer_t0.elapsed().as_secs_f64() * 1e3;
    let tokenization_t0 = Instant::now();
    let first_prompt_ids = tokenizer
        .encode(
            &first_prompt,
            prompt_add_special_tokens(args, first_prompt_source),
        )
        .context("tokenize prompt")?;
    let first_tokenization_ms = tokenization_t0.elapsed().as_secs_f64() * 1e3;
    if args.sampling_attribution {
        ensure!(
            first_prompt_ids.len() == 419
                && token_ids_sha256_i32le(&first_prompt_ids)
                    == "fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f",
            "--sampling-attribution prompt token identity changed"
        );
    }
    let first_prepared = PreparedRequest {
        request_start_unix_ms: first_request_start_unix_ms,
        request_t0: first_request_t0,
        request_start_allocated: first_request_start_allocated,
        pipeline_cache_start: first_pipeline_cache_start,
        prompt: first_prompt,
        prompt_source: first_prompt_source,
        completed_checkpoint_eligible: first_completed_checkpoint_eligible,
        prompt_ids: first_prompt_ids,
        prompt_acquisition_ms: first_prompt_acquisition_ms,
        tokenizer_init_ms: first_tokenizer_init_ms,
        tokenization_ms: first_tokenization_ms,
        tokenizer_reused: false,
    };
    let first = execute_single_turn_request(
        &loaded,
        &tokenizer,
        model_path,
        args,
        greedy_gpu_mode,
        runtime_and_model_load_ms,
        process_model_ready_allocated,
        pair_id.as_deref(),
        0,
        "first_post_model_load",
        first_prepared,
        stdout_sink,
        durable_store.as_ref(),
        durable_max_record_bytes,
        sampling_clock_probe.as_ref(),
    )?;
    let mut results = vec![first];

    if args.request_timing_warm_followup {
        ensure!(
            loaded.prefix_cache_stats().entries == 0,
            "warm follow-up requires an unused prefix cache"
        );
        let warm_pipeline_cache_start =
            timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
        let warm_request_t0 = Instant::now();
        let warm_request_start_unix_ms = unix_epoch_ms_u64()?;
        let warm_request_start_allocated =
            timing_enabled.then(|| loaded.context().current_allocated_size());
        let prompt_t0 = Instant::now();
        let (warm_prompt, warm_prompt_source, warm_completed_checkpoint_eligible) =
            prompt_text(args)?;
        let warm_prompt_acquisition_ms = prompt_t0.elapsed().as_secs_f64() * 1e3;
        let tokenization_t0 = Instant::now();
        let warm_prompt_ids = tokenizer
            .encode(
                &warm_prompt,
                prompt_add_special_tokens(args, warm_prompt_source),
            )
            .context("retokenize warm follow-up prompt")?;
        let warm_tokenization_ms = tokenization_t0.elapsed().as_secs_f64() * 1e3;
        ensure!(
            warm_prompt_source == results[0].prompt_source
                && warm_prompt == results[0].prompt
                && warm_prompt_ids == results[0].prompt_ids
                && warm_completed_checkpoint_eligible == first_completed_checkpoint_eligible,
            "warm follow-up prompt bytes or token IDs differ from request 0"
        );
        let warm_prepared = PreparedRequest {
            request_start_unix_ms: warm_request_start_unix_ms,
            request_t0: warm_request_t0,
            request_start_allocated: warm_request_start_allocated,
            pipeline_cache_start: warm_pipeline_cache_start,
            prompt: warm_prompt,
            prompt_source: warm_prompt_source,
            completed_checkpoint_eligible: warm_completed_checkpoint_eligible,
            prompt_ids: warm_prompt_ids,
            prompt_acquisition_ms: warm_prompt_acquisition_ms,
            tokenizer_init_ms: 0.0,
            tokenization_ms: warm_tokenization_ms,
            tokenizer_reused: true,
        };
        let warm = execute_single_turn_request(
            &loaded,
            &tokenizer,
            model_path,
            args,
            greedy_gpu_mode,
            runtime_and_model_load_ms,
            process_model_ready_allocated,
            pair_id.as_deref(),
            1,
            "warm_followup",
            warm_prepared,
            stdout_sink,
            durable_store.as_ref(),
            durable_max_record_bytes,
            sampling_clock_probe.as_ref(),
        )?;
        let first_stop = results[0].row.as_ref().map(|row| row.stop_reason);
        let warm_stop = warm.row.as_ref().map(|row| row.stop_reason);
        ensure!(
            warm.generated == results[0].generated && warm_stop == first_stop,
            "warm follow-up generated tokens or stop reason differ from request 0"
        );
        results.push(warm);
        for result in &mut results {
            let row = result.row.as_mut().expect("paired timing row");
            row.pair_request_equal = Some(true);
            row.pair_generated_tokens_equal = Some(true);
        }
    }

    if let Some(file) = timing_file.as_mut() {
        let mut payload = Vec::new();
        for result in &results {
            serde_json::to_writer(&mut payload, result.row.as_ref().expect("timing row"))
                .context("serialize request timings")?;
            payload.push(b'\n');
        }
        file.write_all(&payload).context("write request timings")?;
        file.flush().context("flush request timings")?;
    }

    let cache_stats = loaded.prefix_cache_stats();
    for (index, result) in results.iter().enumerate() {
        let load_ms = if index == 0 {
            runtime_and_model_load_ms + result.tokenizer_init_ms
        } else {
            0.0
        };
        let stats_prefix = if args.request_timing_warm_followup {
            format!("stats[{index}]")
        } else {
            "stats".to_owned()
        };
        eprintln!(
            concat!(
                "{}: prompt_tokens={} generated_tokens={} transitions={} stop_reason={} ",
                "load_ms={:.1} prefill_ms={:.1} ttft_ms={:.1} ",
                "decode_tps={:.2} transition_tps={:.2} cache_entries={} ",
                "cache_mib={:.1}/{:.1}"
            ),
            stats_prefix,
            result.prompt_ids.len(),
            result.generated.len(),
            result.transitions,
            result.stop_reason.as_str(),
            load_ms,
            result.prefill_ms,
            result.ttft_ms,
            result.decode_tps,
            result.transition_tps,
            cache_stats.entries,
            cache_stats.indexed_bytes as f64 / 1024.0 / 1024.0,
            cache_stats.max_indexed_bytes as f64 / 1024.0 / 1024.0,
        );
    }

    if let Some(path) = args.trace_request.as_ref() {
        append_request_trace(
            path,
            arrival_ms,
            results[0].prompt_ids.len(),
            results[0].generated.len(),
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_single_turn_request(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    model_path: &Path,
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
    runtime_and_model_load_ms: f64,
    process_model_ready_allocated: Option<u64>,
    pair_id: Option<&str>,
    request_index: usize,
    request_epoch: &'static str,
    prepared: PreparedRequest,
    stdout_sink: &'static str,
    durable_store: Option<&DurableCheckpointStore>,
    durable_max_record_bytes: u64,
    sampling_clock_probe: Option<&SamplingClockProbe>,
) -> Result<SingleTurnResult> {
    let PreparedRequest {
        request_start_unix_ms,
        request_t0,
        request_start_allocated,
        pipeline_cache_start,
        prompt,
        prompt_source,
        completed_checkpoint_eligible,
        prompt_ids,
        prompt_acquisition_ms,
        tokenizer_init_ms,
        tokenization_ms,
        tokenizer_reused,
    } = prepared;
    let timing_enabled = process_model_ready_allocated.is_some();
    let validation_t0 = Instant::now();
    if prompt_ids.is_empty() {
        bail!("prompt tokenized to zero tokens");
    }
    let min_capacity = prompt_ids
        .len()
        .checked_add(args.tokens)
        .and_then(|v| v.checked_add(16))
        .context("sequence capacity overflow")?;
    let capacity = args.max_context_tokens.unwrap_or(min_capacity);
    ensure!(
        capacity >= prompt_ids.len() + args.tokens,
        "max context {} is smaller than prompt {} + generation {}",
        capacity,
        prompt_ids.len(),
        args.tokens
    );
    let capacity_validation_ms = validation_t0.elapsed().as_secs_f64() * 1e3;
    let allocated = allocate_prefill_request_state(
        loaded,
        args.prefill_chunk,
        prompt_ids.len(),
        capacity,
        durable_store.is_none(),
    )?;
    let chunk = allocated.chunk;
    let prefill_chunk_decision = allocated.decision;
    let mut scratch = allocated.scratch;
    let mut sequence = allocated.sequence;
    let scratch_allocation_ms = allocated.scratch_allocation_ms;
    let sequence_allocation_ms = allocated.sequence_allocation_ms;
    let after_scratch_allocated = timing_enabled.then_some(allocated.after_scratch_allocated);
    let after_sequence_allocated = timing_enabled.then_some(allocated.after_sequence_allocated);
    let forward = loaded.forward();
    if args.sampled_structural {
        forward
            .ensure_sampled_structural_session_supported(sequence.metal_session())
            .context("validate sampled structural session row before prefill")?;
    }

    let pipeline_cache_prefill_entry =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let durable_capture_policy = durable_store.map_or(DurableCapturePolicy::Disabled, |_| {
        selected_single_turn_durable_policy(args, prompt_ids.len(), completed_checkpoint_eligible)
    });
    let durable_prefix_len = durable_capture_policy.prompt_prefix_len();
    let mut durable_prepared: Option<PreparedCheckpoint> = None;
    let mut durable_capture_kind = None;
    let mut durable_capture_stop_reason = None;
    let mut durable_restore_ms = 0.0;
    let mut durable_capture_ms = 0.0;
    let mut prompt_logits = None;
    if let Some(store) = durable_store {
        let restore_t0 = Instant::now();
        let has_blobs = match store.has_managed_blobs() {
            Ok(has_blobs) => Some(has_blobs),
            Err(error) => {
                durable_restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
                eprintln!(
                    "warning: durable prefix inventory failed after {:.1} ms; cold-prefilling: {error}",
                    durable_restore_ms,
                );
                None
            }
        };
        if has_blobs == Some(true) {
            let lookup_len = selected_single_turn_durable_lookup_len(args, prompt_ids.len());
            match loaded.restore_durable_prefix(
                store,
                &mut sequence,
                &prompt_ids[..lookup_len],
                durable_max_record_bytes,
            ) {
                Ok(attempt) => {
                    durable_restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
                    eprintln!(
                        concat!(
                            "durable_prefix_cache: identity_cache={} hashed_bytes={} ",
                            "checkpoint_hit={} matched={} restored={} exact={} candidates={} ",
                            "corrupt_removed={} restore_total_ms={:.1}"
                        ),
                        identity_cache_outcome_label(attempt.compatibility.outcome),
                        attempt.compatibility.bytes_hashed,
                        attempt.hit.is_some(),
                        attempt.lookup.matched_prefix_len,
                        attempt.lookup.restored_prefix_len,
                        attempt.lookup.exact,
                        attempt.lookup.candidates_examined,
                        attempt.lookup.corrupt_entries_removed,
                        durable_restore_ms,
                    );
                    if let Some(hit) = attempt.hit {
                        prompt_logits = hit.exact_final_logits;
                    }
                }
                Err(RuntimeError::CheckpointStore(error)) => {
                    durable_restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
                    eprintln!(
                        "warning: durable prefix lookup failed after {:.1} ms; cold-prefilling: {error}",
                        durable_restore_ms,
                    );
                }
                Err(error) => return Err(error.into()),
            }
        } else if has_blobs == Some(false) {
            durable_restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "durable_prefix_cache: store_empty=true restore_total_ms={:.1}",
                durable_restore_ms,
            );
        }
    }

    let mut prefill_ms = 0.0;
    if let Some(prefix_len) = durable_prefix_len
        && prefix_len > sequence.position()
    {
        let position = sequence.position();
        let (logits, ms) = prefill_span(
            &forward,
            &mut sequence,
            &mut scratch,
            &prompt_ids[position..prefix_len],
            position,
        )?;
        prefill_ms += ms;
        let capture_t0 = Instant::now();
        let estimated =
            loaded.estimate_checkpoint_boundary_sizes(&sequence, prefix_len, false, true)?;
        if estimated.record_bytes > durable_max_record_bytes {
            eprintln!(
                concat!(
                    "warning: durable prefix capture skipped: estimated_record_bytes={} ",
                    "estimated_snapshot_bytes={} max_entry_bytes={}"
                ),
                estimated.record_bytes, estimated.snapshot_bytes, durable_max_record_bytes,
            );
        } else {
            match loaded.prepare_checkpoint_boundary(
                &sequence,
                prompt_ids[..prefix_len].to_vec(),
                None,
                Some(logits.clone()),
            ) {
                Ok(prepared) => {
                    durable_prepared = Some(prepared);
                    durable_capture_kind = Some("prompt");
                }
                Err(RuntimeError::MetalModel(MfError::Snapshot(
                    SnapshotValidationError::AllocationFailed { .. },
                ))) => eprintln!(
                    "warning: durable prefix capture allocation failed; continuing without publication"
                ),
                Err(error) => return Err(error.into()),
            }
        }
        durable_capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3;
        prompt_logits = Some(logits);
    }
    if sequence.position() < prompt_ids.len() {
        let position = sequence.position();
        let (logits, ms) = prefill_span(
            &forward,
            &mut sequence,
            &mut scratch,
            &prompt_ids[position..],
            position,
        )?;
        prefill_ms += ms;
        prompt_logits = Some(logits);
    }
    let logits = prompt_logits.context("durable prefix restore did not produce prompt logits")?;
    let prefill_attention_query =
        (scratch.attn_matrix_tiled_layer_calls() > 0).then(|| PrefillAttentionQueryStats {
            outer_chunk_rows: chunk,
            query_rows: scratch.attn_matrix_query_rows(),
            tiled_layer_calls: scratch.attn_matrix_tiled_layer_calls(),
            query_tile_calls: scratch.attn_matrix_query_tile_calls(),
        });
    let prefill_scratch_overlay =
        scratch
            .prefill_scratch_overlay_stats()
            .map(|stats| PrefillScratchOverlayTimingStats {
                backing_bytes: stats.backing_bytes,
                attention_bytes: stats.attention_bytes,
                gdn_bytes: stats.gdn_bytes,
                saved_bytes: stats.saved_bytes,
            });
    let pipeline_cache_prefill_exit =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let after_prefill_allocated = timing_enabled.then(|| loaded.context().current_allocated_size());

    let stdout_handle = std::io::stdout();
    let mut stdout = stdout_handle.lock();
    let generation_start_ms = request_t0.elapsed().as_secs_f64() * 1e3;
    let mut first_delivery_ms = None;
    let mut first_callback_duration_ms = None;
    let mut first_delivery_allocated = None;
    let stop_tokens = loaded
        .gguf()
        .stop_token_ids()
        .context("load producer-declared stop tokens")?;
    let sampling_config = cli_sampling_config(args)?;
    let mut sampler = Sampler::new(sampling_config).context("initialize request sampler")?;
    let greedy_gpu_decision =
        resolve_greedy_gpu_decision(greedy_gpu_mode, sampling_config, args.prompt_lookup);
    let use_gpu_greedy = greedy_gpu_decision.enabled;
    let (generation, prompt_lookup_stats, sampling_attribution, sampled_structural) = if args
        .prompt_lookup
    {
        let result = generate_prompt_lookup(
            loaded,
            &forward,
            sequence,
            &prompt_ids,
            logits,
            args.tokens,
            &stop_tokens,
            |token| {
                let callback_t0 = Instant::now();
                write!(stdout, "{}", tokenizer.decode_piece(token))?;
                stdout.flush().context("flush generated token")?;
                if first_delivery_ms.is_none() {
                    first_delivery_ms = Some(request_t0.elapsed().as_secs_f64() * 1e3);
                    first_callback_duration_ms = Some(callback_t0.elapsed().as_secs_f64() * 1e3);
                    first_delivery_allocated =
                        timing_enabled.then(|| loaded.context().current_allocated_size());
                }
                Ok(())
            },
        )?;
        sequence = result.sequence;
        (result.generation, Some(result.stats), None, None)
    } else {
        let mut on_token = |token| {
            let callback_t0 = Instant::now();
            write!(stdout, "{}", tokenizer.decode_piece(token))?;
            stdout.flush().context("flush generated token")?;
            if first_delivery_ms.is_none() {
                first_delivery_ms = Some(request_t0.elapsed().as_secs_f64() * 1e3);
                first_callback_duration_ms = Some(callback_t0.elapsed().as_secs_f64() * 1e3);
                first_delivery_allocated =
                    timing_enabled.then(|| loaded.context().current_allocated_size());
            }
            Ok(())
        };
        let (generation, sampling_attribution, sampled_structural) = if args.sampling_attribution {
            let (generation, sampler_attribution, transition_attribution) =
                generate_serial_attributed(
                    logits,
                    args.tokens,
                    &stop_tokens,
                    &mut sampler,
                    &mut on_token,
                    |token| {
                        let position = sequence.position();
                        let next = forward
                            .single_token_sampled_attribution(
                                token,
                                u32::try_from(position).context("position does not fit u32")?,
                                unsafe { sequence.metal_session_mut() },
                            )
                            .context("decode token with sampling attribution")?;
                        sequence.advance_by(1)?;
                        Ok(next)
                    },
                )?;
            let clock_probe = sampling_clock_probe
                .context("sampling attribution clock probe was not prepared")?
                .clone();
            let attribution = finalize_sampling_attribution(
                &prompt_ids,
                clock_probe,
                sampler_attribution,
                transition_attribution,
                generation.transition_ms,
                generation.wall_ms,
            );
            (generation, Some(attribution), None)
        } else if args.sampled_structural {
            let (generation, telemetry) = generate_sampled_structural(
                logits,
                args.tokens,
                &stop_tokens,
                &mut sampler,
                &mut on_token,
                |token, trial, trial_telemetry| {
                    let position = sequence.position();
                    let (sampled, _profile, row) = forward
                        .single_token_sampled_structural(
                            token,
                            u32::try_from(position).context("position does not fit u32")?,
                            unsafe { sequence.metal_session_mut() },
                            trial,
                        )
                        .context("decode token with sampled structural path")?;
                    let state = match sampled {
                        Ok((sampled, evidence)) => {
                            ensure!(
                                evidence.used_bounded_path,
                                "sampled structural transition selection fell back"
                            );
                            trial_telemetry.record_transition(evidence, row)?;
                            SampledStructuralDecodeState::Selected(Ok(sampled))
                        }
                        Err(error) => SampledStructuralDecodeState::Selected(Err(error)),
                    };
                    sequence.advance_by(1)?;
                    Ok(state)
                },
            )?;
            (generation, None, Some(telemetry))
        } else if use_gpu_greedy {
            (
                generate_gpu_greedy(
                    logits,
                    args.tokens,
                    &stop_tokens,
                    &mut sampler,
                    &mut on_token,
                    |token| {
                        let position = sequence.position();
                        let next = forward
                            .single_token_greedy(
                                token,
                                u32::try_from(position).context("position does not fit u32")?,
                                unsafe { sequence.metal_session_mut() },
                            )
                            .context("decode token with GPU greedy selection")?;
                        sequence.advance_by(1)?;
                        Ok(next)
                    },
                )?,
                None,
                None,
            )
        } else {
            (
                generate_serial(
                    logits,
                    args.tokens,
                    &stop_tokens,
                    &mut sampler,
                    &mut on_token,
                    |token| {
                        let position = sequence.position();
                        let next = forward
                            .single_token(
                                token,
                                u32::try_from(position).context("position does not fit u32")?,
                                unsafe { sequence.metal_session_mut() },
                            )
                            .context("decode token")?;
                        sequence.advance_by(1)?;
                        Ok(next)
                    },
                )?,
                None,
                None,
            )
        };
        (generation, None, sampling_attribution, sampled_structural)
    };
    let mut inference_complete_ms = request_t0.elapsed().as_secs_f64() * 1e3;
    let pipeline_cache_generation_exit =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let stop_reason = generation.stop_reason;
    let generated = generation.tokens;
    let completed_boundary = if durable_capture_policy == DurableCapturePolicy::AutomaticCompleted {
        Some(derive_completed_checkpoint_boundary(
            prompt_ids.len(),
            &generated,
            generation.transitions,
            sequence.position(),
        )?)
    } else {
        None
    };
    if !generated.is_empty() {
        writeln!(stdout)?;
        stdout.flush().context("flush final newline")?;
    }
    if first_delivery_ms.is_none() {
        let delivery_ms = request_t0.elapsed().as_secs_f64() * 1e3;
        first_delivery_ms = Some(delivery_ms);
        first_callback_duration_ms = Some(0.0);
        first_delivery_allocated =
            timing_enabled.then(|| loaded.context().current_allocated_size());
        inference_complete_ms = inference_complete_ms.max(delivery_ms);
    }
    let response_flushed_t0 = Instant::now();
    let total_request_ms = request_t0.elapsed().as_secs_f64() * 1e3;
    report_prefill_chunk_decision(prefill_chunk_decision.as_ref(), prompt_ids.len());
    let request_end_allocated = timing_enabled.then(|| loaded.context().current_allocated_size());
    drop(stdout);

    if let Some(boundary) = completed_boundary {
        let capture_t0 = Instant::now();
        let estimated = loaded.estimate_checkpoint_boundary_sizes(
            &sequence,
            boundary.consumed_prefix_len,
            true,
            false,
        )?;
        if estimated.record_bytes > durable_max_record_bytes {
            eprintln!(
                concat!(
                    "warning: completed durable checkpoint skipped: estimated_record_bytes={} ",
                    "estimated_snapshot_bytes={} max_entry_bytes={}"
                ),
                estimated.record_bytes, estimated.snapshot_bytes, durable_max_record_bytes,
            );
        } else {
            let consumed = boundary.consumed_tokens(&prompt_ids, &generated);
            match loaded.prepare_checkpoint_boundary(
                &sequence,
                consumed,
                Some(boundary.pending_token),
                None,
            ) {
                Ok(prepared) => {
                    durable_prepared = Some(prepared);
                    durable_capture_kind = Some("completed");
                    durable_capture_stop_reason = Some(stop_reason);
                }
                Err(RuntimeError::MetalModel(MfError::Snapshot(
                    SnapshotValidationError::AllocationFailed { .. },
                ))) => eprintln!(
                    "warning: completed durable checkpoint allocation failed; continuing without publication"
                ),
                Err(error) => return Err(error.into()),
            }
        }
        durable_capture_ms += capture_t0.elapsed().as_secs_f64() * 1e3;
    }

    let ttft_ms = first_delivery_ms.context("generation produced no first-token delivery")?;
    let first_token_ready_ms = generation_start_ms
        + generation
            .first_token_ready_ms
            .context("generation produced no first-token selection")?;
    let decode_tps = if generation.wall_ms > 0.0 {
        generated.len() as f64 / (generation.wall_ms / 1e3)
    } else {
        0.0
    };
    let transition_tps = if generation.transition_ms > 0.0 {
        generation.transitions as f64 / (generation.transition_ms / 1e3)
    } else {
        0.0
    };
    let model_identity = if timing_enabled {
        Some(loaded.snapshot_identity(&sequence)?)
    } else {
        None
    };
    let timing_values = timing_enabled.then(|| {
        (
            process_model_ready_allocated.expect("timing sample"),
            request_start_allocated.expect("timing sample"),
            after_scratch_allocated.expect("timing sample"),
            after_sequence_allocated.expect("timing sample"),
            after_prefill_allocated.expect("timing sample"),
            first_delivery_allocated.expect("timing sample"),
            request_end_allocated.expect("timing sample"),
        )
    });
    drop(sequence);
    drop(scratch);
    let after_state_drop_allocated =
        timing_enabled.then(|| loaded.context().current_allocated_size());
    if let (Some(store), Some(prepared)) = (durable_store, durable_prepared.as_ref()) {
        let publish_t0 = Instant::now();
        match loaded.publish_prepared_checkpoint(store, prepared, durable_max_record_bytes) {
            Ok(report) => {
                let publish_elapsed = publish_t0.elapsed();
                if store.staged_integrity_is_explicit() {
                    eprintln!(
                        concat!(
                            "durable_prefix_cache: publish={} capture={} matched_tokens={} ",
                            "restored_tokens={} pending={} stop_reason={} blob_bytes={} ",
                            "evicted={} identity={} staged_integrity={} ",
                            "staged_integrity_us={} capture_ms={:.1} publish_us={} ",
                            "post_response_us={}"
                        ),
                        publish_outcome_label(report.store.outcome),
                        durable_capture_kind.unwrap_or("unknown"),
                        prepared.matched_prefix_len(),
                        prepared.restored_prefix_len(),
                        prepared.has_pending_token(),
                        durable_capture_stop_reason.map_or("none", StopReason::as_str),
                        report.store.blob_bytes,
                        report.store.evicted_entries,
                        identity_cache_outcome_label(report.compatibility.outcome),
                        report.store.staged_integrity.mode.as_str(),
                        report.store.staged_integrity.elapsed.as_micros(),
                        durable_capture_ms,
                        publish_elapsed.as_micros(),
                        response_flushed_t0.elapsed().as_micros(),
                    );
                } else {
                    eprintln!(
                        concat!(
                            "durable_prefix_cache: publish={} capture={} matched_tokens={} ",
                            "restored_tokens={} pending={} stop_reason={} blob_bytes={} ",
                            "evicted={} identity={} capture_ms={:.1} publish_ms={:.1}"
                        ),
                        publish_outcome_label(report.store.outcome),
                        durable_capture_kind.unwrap_or("unknown"),
                        prepared.matched_prefix_len(),
                        prepared.restored_prefix_len(),
                        prepared.has_pending_token(),
                        durable_capture_stop_reason.map_or("none", StopReason::as_str),
                        report.store.blob_bytes,
                        report.store.evicted_entries,
                        identity_cache_outcome_label(report.compatibility.outcome),
                        durable_capture_ms,
                        publish_elapsed.as_secs_f64() * 1e3,
                    );
                }
            }
            Err(error) => {
                let publish_elapsed = publish_t0.elapsed();
                if store.staged_integrity_is_explicit() {
                    eprintln!(
                        concat!(
                            "durable_prefix_cache: publish=failed staged_integrity={} ",
                            "staged_integrity_us=none capture_ms={:.1} publish_us={} ",
                            "post_response_us={} error={}"
                        ),
                        store.staged_integrity_mode().as_str(),
                        durable_capture_ms,
                        publish_elapsed.as_micros(),
                        response_flushed_t0.elapsed().as_micros(),
                        error,
                    );
                } else {
                    eprintln!(
                        concat!(
                            "warning: durable prefix publication failed after response ",
                            "(restore_ms={:.1} capture_ms={:.1}): {}"
                        ),
                        durable_restore_ms, durable_capture_ms, error,
                    );
                }
            }
        }
    }

    let row = timing_values.map(|samples| {
        let model_identity = model_identity.expect("timing identity");
        let mut metal_allocated = metal_allocation_samples(
            samples.0,
            samples.1,
            samples.2,
            samples.3,
            samples.4,
            samples.5,
            samples.6,
            after_state_drop_allocated.expect("timing sample"),
        );
        if let Some(stats) = prompt_lookup_stats.as_ref() {
            metal_allocated.current_allocated_sampled_max_bytes = metal_allocated
                .current_allocated_sampled_max_bytes
                .max(stats.scratch_peak_allocated_bytes);
        }
        RequestTimingRow {
            schema_version: request_schema_version(
                args.prefill_chunk,
                args.prompt_lookup,
                prefill_attention_query.is_some(),
                prefill_scratch_overlay.is_some(),
                sampling_config.temperature > 0.0,
                args.sampling_attribution,
                args.sampled_structural,
            ),
            request_epoch,
            request_index,
            tokenizer_reused,
            pair_requested: pair_id.is_some(),
            pair_id: pair_id.map(str::to_owned),
            pair_request_equal: None,
            pair_generated_tokens_equal: None,
            prefix_cache_used: false,
            build_commit: env!("QWEN_BUILD_COMMIT"),
            build_dirty: env!("QWEN_BUILD_DIRTY"),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE"),
            model: model_path.display().to_string(),
            runtime_identity_kind: "metadata_compatibility_v1",
            runtime_model_id: format!("{:016x}", model_identity.model_id),
            runtime_tokenizer_id: format!("{:016x}", model_identity.tokenizer_id),
            greedy_gpu_selection_reason: greedy_gpu_decision.reason,
            request_start_unix_ms,
            runtime_and_model_load_ms,
            stdout_sink,
            ttft_endpoint: "stdout_flush_complete",
            prompt_source,
            prompt_bytes: prompt.len(),
            prompt_tokens: prompt_ids.len(),
            requested_tokens: args.tokens,
            generated_tokens: generated.len(),
            generated_token_sha256: generated_token_sha256(&generated),
            stop_reason,
            decode_policy: decode_policy_label(sampling_config, args.prompt_lookup, use_gpu_greedy),
            sampling: SamplingTelemetry::sampled(sampling_config, sampler.draws()),
            sampling_attribution,
            sampled_structural,
            terminal_token_target_transition_consumed: false,
            no_special_tokens: !prompt_add_special_tokens(args, prompt_source),
            prefill_chunk_requested: args.prefill_chunk,
            prefill_chunk_effective: chunk,
            prefill_chunk_decision,
            prefill_attention_query,
            prefill_scratch_overlay,
            max_context_tokens: capacity,
            prompt_acquisition_ms,
            tokenizer_init_ms,
            tokenization_ms,
            capacity_validation_ms,
            scratch_allocation_ms,
            sequence_allocation_ms,
            prefill_ms,
            first_token_selection_ms: generation.first_token_selection_ms,
            first_token_callback_duration_ms: first_callback_duration_ms
                .expect("first callback duration"),
            first_token_ready_ms,
            ttft_ms,
            generation_ms: generation.wall_ms,
            transition_count: generation.transitions,
            transition_ms: generation.transition_ms,
            transition_tps,
            inference_complete_ms,
            total_request_ms,
            pso_cache: pipeline_cache_phase_metrics(
                pipeline_cache_start.expect("timing PSO snapshot"),
                pipeline_cache_prefill_entry.expect("timing PSO snapshot"),
                pipeline_cache_prefill_exit.expect("timing PSO snapshot"),
                pipeline_cache_generation_exit.expect("timing PSO snapshot"),
            ),
            metal_allocated,
            prompt_lookup: prompt_lookup_stats,
        }
    });
    if let Some(row) = row.as_ref() {
        validate_request_timing_invariants(
            row.first_token_ready_ms,
            row.ttft_ms,
            row.inference_complete_ms,
            row.total_request_ms,
            row.generated_tokens,
            row.transition_count,
        )?;
        validate_sampling_attribution_row(row)?;
        validate_sampled_structural_row(row)?;
    }
    Ok(SingleTurnResult {
        row,
        prompt,
        prompt_source,
        prompt_ids,
        generated,
        transitions: generation.transitions,
        stop_reason,
        prefill_ms,
        ttft_ms,
        decode_tps,
        transition_tps,
        tokenizer_init_ms,
    })
}

fn run_requests_jsonl(
    model_path: &Path,
    requests_path: &Path,
    gguf: GgufFile,
    args: &Args,
) -> Result<()> {
    args.prefill_chunk.validate()?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    cli_sampling_config(args)?;

    let load_t0 = Instant::now();
    let runtime = Runtime::metal().context("init Metal runtime")?;
    let loaded = runtime
        .load_open_model_with_config(
            gguf,
            LoadedModelConfig {
                prefix_cache_max_bytes: prefix_cache_max_bytes(args)?,
                ..LoadedModelConfig::default()
            },
        )
        .with_context(|| format!("load model {}", model_path.display()))?;
    if args.prompt_lookup {
        ensure_prompt_lookup_n8_supported(loaded.metal_model()).map_err(anyhow::Error::msg)?;
    }
    let greedy_gpu_mode = configured_greedy_gpu_argmax_mode();
    let tokenizer = loaded.tokenizer().context("load tokenizer")?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;

    let mut stats_file = args
        .request_stats
        .as_ref()
        .map(|path| open_append_file(path, "request stats"))
        .transpose()?;
    let mut stdout = std::io::stdout().lock();
    let mut n_requests = 0usize;

    eprintln!(
        "loaded {} in {:.1} ms; prefix_cache_max_mib={}",
        model_path.display(),
        load_ms,
        args.prefix_cache_max_mib,
    );

    if requests_path == Path::new("-") {
        let stdin = std::io::stdin();
        let reader = stdin.lock();
        if args.cache_prefix_auto_min_tokens > 0 {
            eprintln!(
                "prefix-cache auto admission needs request lookahead; disabled for stdin JSONL"
            );
        }
        for (line_idx, line) in reader.lines().enumerate() {
            let line_no = line_idx + 1;
            let line = line.with_context(|| format!("read requests line {line_no}"))?;
            let Some(prepared_request) =
                prepare_jsonl_request_line(line_no, &line, &tokenizer, args)?
            else {
                continue;
            };
            let (output, stats) = run_jsonl_request(
                &loaded,
                &tokenizer,
                &prepared_request,
                args,
                greedy_gpu_mode,
            )
            .with_context(|| format!("run request {}", prepared_request.id))?;

            serde_json::to_writer(&mut stdout, &output).context("write request output")?;
            writeln!(stdout)?;
            stdout.flush()?;

            if let Some(file) = stats_file.as_mut() {
                serde_json::to_writer(&mut *file, &stats).context("write request stats")?;
                writeln!(file)?;
                file.flush()?;
            }
            if let Some(path) = args.trace_request.as_ref() {
                append_request_trace(
                    path,
                    unix_epoch_ms()?,
                    stats.prompt_tokens,
                    stats.generated_tokens,
                )?;
            }
            n_requests += 1;
        }
    } else {
        let mut prepared = prepare_jsonl_requests(requests_path, &tokenizer, args)?;
        discover_auto_cache_prefixes(&mut prepared, args.cache_prefix_auto_min_tokens);

        for prepared_request in &prepared {
            let (output, stats) =
                run_jsonl_request(&loaded, &tokenizer, prepared_request, args, greedy_gpu_mode)
                    .with_context(|| format!("run request {}", prepared_request.id))?;

            serde_json::to_writer(&mut stdout, &output).context("write request output")?;
            writeln!(stdout)?;
            stdout.flush()?;

            if let Some(file) = stats_file.as_mut() {
                serde_json::to_writer(&mut *file, &stats).context("write request stats")?;
                writeln!(file)?;
                file.flush()?;
            }
            if let Some(path) = args.trace_request.as_ref() {
                append_request_trace(
                    path,
                    unix_epoch_ms()?,
                    stats.prompt_tokens,
                    stats.generated_tokens,
                )?;
            }
            n_requests += 1;
        }
    }

    ensure!(
        n_requests > 0,
        "requests JSONL {} contained no requests",
        requests_path.display()
    );

    let stats = loaded.prefix_cache_stats();
    eprintln!(
        "stats: requests={} cache_entries={} cache_mib={:.1}/{:.1}",
        n_requests,
        stats.entries,
        stats.indexed_bytes as f64 / 1024.0 / 1024.0,
        stats.max_indexed_bytes as f64 / 1024.0 / 1024.0,
    );
    Ok(())
}

fn prepare_jsonl_requests(
    requests_path: &Path,
    tokenizer: &Tokenizer,
    args: &Args,
) -> Result<Vec<PreparedJsonlRequest>> {
    let stdin;
    let reader: Box<dyn BufRead> = if requests_path == Path::new("-") {
        stdin = std::io::stdin();
        Box::new(stdin.lock())
    } else {
        let requests = std::fs::File::open(requests_path)
            .with_context(|| format!("open requests JSONL {}", requests_path.display()))?;
        Box::new(std::io::BufReader::new(requests))
    };

    let mut prepared = Vec::new();
    for (line_idx, line) in reader.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = line.with_context(|| format!("read requests line {line_no}"))?;
        if let Some(request) = prepare_jsonl_request_line(line_no, &line, tokenizer, args)? {
            prepared.push(request);
        }
    }
    ensure!(
        !prepared.is_empty(),
        "requests JSONL {} contained no requests",
        requests_path.display()
    );
    Ok(prepared)
}

fn prepare_jsonl_request_line(
    line_no: usize,
    line: &str,
    tokenizer: &Tokenizer,
    args: &Args,
) -> Result<Option<PreparedJsonlRequest>> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return Ok(None);
    }
    let request: JsonlRequest =
        serde_json::from_str(trimmed).with_context(|| format!("parse requests line {line_no}"))?;
    let id = request
        .id
        .clone()
        .unwrap_or_else(|| format!("line-{line_no}"));
    let prompt = request_prompt(&request, line_no)?;
    let prompt_ids = tokenizer
        .encode(&prompt, !args.no_special_tokens)
        .context("tokenize prompt")?;
    if prompt_ids.is_empty() {
        bail!("request {id} tokenized to zero tokens");
    }
    let sampling = request_sampling_config(&request, args)?;
    validate_sampling_decode_policy(sampling, args.prompt_lookup)
        .with_context(|| format!("validate decode policy for request {id}"))?;
    Ok(Some(PreparedJsonlRequest {
        request,
        id,
        line: line_no,
        prompt_ids,
        sampling,
        auto_cache_prefix_tokens: None,
        auto_cache_future_hits: 0,
    }))
}

fn discover_auto_cache_prefixes(requests: &mut [PreparedJsonlRequest], min_tokens: usize) {
    if min_tokens == 0 || requests.len() < 2 {
        return;
    }

    for idx in 0..requests.len() {
        let mut future_lcps = Vec::new();
        for future in &requests[idx + 1..] {
            let lcp = longest_common_prefix_len(&requests[idx].prompt_ids, &future.prompt_ids);
            if lcp >= min_tokens {
                future_lcps.push(lcp);
            }
        }
        if future_lcps.is_empty() {
            continue;
        }

        future_lcps.sort_unstable();
        let mut best_len = 0usize;
        let mut best_hits = 0usize;
        let mut best_score = 0usize;
        for (pos, &len) in future_lcps.iter().enumerate() {
            let hits = future_lcps.len() - pos;
            let score = len.saturating_mul(hits);
            if score > best_score || (score == best_score && len > best_len) {
                best_len = len;
                best_hits = hits;
                best_score = score;
            }
        }

        requests[idx].auto_cache_prefix_tokens = Some(best_len.min(requests[idx].prompt_ids.len()));
        requests[idx].auto_cache_future_hits = best_hits;
    }
}

fn longest_common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn selected_cache_prefix(
    request: &JsonlRequest,
    args: &Args,
    auto_cache_prefix_tokens: Option<usize>,
    prompt_len: usize,
) -> (Option<usize>, CachePrefixSource) {
    if let Some(n) = request.cache_prefix_tokens {
        return selected_cache_prefix_from_value(n, prompt_len, CachePrefixSource::Request);
    }
    if let Some(n) = args.cache_prefix_tokens {
        return selected_cache_prefix_from_value(n, prompt_len, CachePrefixSource::Cli);
    }
    if let Some(n) = auto_cache_prefix_tokens {
        return selected_cache_prefix_from_value(n, prompt_len, CachePrefixSource::Auto);
    }
    (None, CachePrefixSource::None)
}

fn selected_cache_prefix_from_value(
    n: usize,
    prompt_len: usize,
    source: CachePrefixSource,
) -> (Option<usize>, CachePrefixSource) {
    if n == 0 {
        let disabled = match source {
            CachePrefixSource::Request => CachePrefixSource::RequestDisabled,
            CachePrefixSource::Cli => CachePrefixSource::CliDisabled,
            _ => CachePrefixSource::None,
        };
        return (None, disabled);
    }
    (Some(n.min(prompt_len)).filter(|&n| n > 0), source)
}

fn run_jsonl_request(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    prepared: &PreparedJsonlRequest,
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
) -> Result<(RequestOutput, RequestStatsRow)> {
    let request = &prepared.request;
    let id = &prepared.id;
    let arrival_ms = unix_epoch_ms_u64()?;
    let total_t0 = Instant::now();
    let prompt_ids = &prepared.prompt_ids;

    let n_generate = request.tokens.unwrap_or(args.tokens);
    ensure!(n_generate > 0, "tokens must be >= 1 for request {id}");
    let min_capacity = prompt_ids
        .len()
        .checked_add(n_generate)
        .and_then(|v| v.checked_add(16))
        .context("sequence capacity overflow")?;
    let capacity = args.max_context_tokens.unwrap_or(min_capacity);
    ensure!(
        capacity >= prompt_ids.len() + n_generate,
        "max context {} is smaller than prompt {} + generation {} for request {}",
        capacity,
        prompt_ids.len(),
        n_generate,
        id,
    );

    let (cache_prefix_tokens, cache_prefix_source) = selected_cache_prefix(
        request,
        args,
        prepared.auto_cache_prefix_tokens,
        prompt_ids.len(),
    );
    let auto_prefill_cache_safe =
        auto_prefill_cache_safe(loaded.prefix_cache_stats().entries, cache_prefix_tokens);
    let allocated = allocate_prefill_request_state(
        loaded,
        args.prefill_chunk,
        prompt_ids.len(),
        capacity,
        auto_prefill_cache_safe,
    )?;
    let chunk = allocated.chunk;
    let prefill_chunk_decision = allocated.decision;
    let mut scratch = allocated.scratch;
    let mut sequence = allocated.sequence;
    let forward = loaded.forward();

    let prompt_hash = token_hash_hex(&prompt_ids);
    let cache_prefix_hash = cache_prefix_tokens.map(|n| token_hash_hex(&prompt_ids[..n]));

    let mut cache_hit = false;
    let mut matched_prefix_tokens = 0usize;
    let mut exact_cache_hit = false;
    let restore_ms;
    let mut prefix_inserted_bytes = 0u64;
    let mut prefix_insert_ms = 0.0;
    let mut prefill_ms = 0.0;

    let logits = {
        let restore_t0 = Instant::now();
        let hit = loaded
            .restore_cached_prefix(&mut sequence, &prompt_ids)
            .context("restore prefix cache")?;
        restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
        if let Some(hit) = hit {
            cache_hit = true;
            matched_prefix_tokens = hit.matched_prefix_len;
            exact_cache_hit = hit.exact;
            let restored_prefix_tokens = hit.restored_prefix_len;
            if restored_prefix_tokens == prompt_ids.len() {
                hit.exact_final_logits.with_context(|| {
                    format!("exact prefix-cache hit for request {id} did not store logits")
                })?
            } else if let Some(prefix_len) = cache_prefix_tokens
                && cache_prefix_needs_extension(prefix_len, restored_prefix_tokens)
            {
                let prefix_suffix = &prompt_ids[restored_prefix_tokens..prefix_len];
                let (prefix_logits, ms) = prefill_span(
                    &forward,
                    &mut sequence,
                    &mut scratch,
                    prefix_suffix,
                    restored_prefix_tokens,
                )?;
                prefill_ms += ms;

                let insert_t0 = Instant::now();
                let insert = loaded
                    .cache_sequence_prefix(
                        &sequence,
                        prompt_ids[..prefix_len].to_vec(),
                        Some(prefix_logits.clone()),
                    )
                    .context("insert prefix cache snapshot")?;
                prefix_insert_ms = insert_t0.elapsed().as_secs_f64() * 1e3;
                prefix_inserted_bytes = insert.snapshot_bytes;

                if prefix_len == prompt_ids.len() {
                    prefix_logits
                } else {
                    let suffix = &prompt_ids[prefix_len..];
                    let (logits, ms) =
                        prefill_span(&forward, &mut sequence, &mut scratch, suffix, prefix_len)?;
                    prefill_ms += ms;
                    logits
                }
            } else {
                let suffix = &prompt_ids[restored_prefix_tokens..];
                let (logits, ms) = prefill_span(
                    &forward,
                    &mut sequence,
                    &mut scratch,
                    suffix,
                    restored_prefix_tokens,
                )?;
                prefill_ms += ms;
                logits
            }
        } else if let Some(prefix_len) = cache_prefix_tokens {
            let prefix = &prompt_ids[..prefix_len];
            let (prefix_logits, ms) =
                prefill_span(&forward, &mut sequence, &mut scratch, prefix, 0)?;
            prefill_ms += ms;

            let insert_t0 = Instant::now();
            let insert = loaded
                .cache_sequence_prefix(&sequence, prefix.to_vec(), Some(prefix_logits.clone()))
                .context("insert prefix cache snapshot")?;
            prefix_insert_ms = insert_t0.elapsed().as_secs_f64() * 1e3;
            prefix_inserted_bytes = insert.snapshot_bytes;

            if prefix_len == prompt_ids.len() {
                prefix_logits
            } else {
                let suffix = &prompt_ids[prefix_len..];
                let (logits, ms) =
                    prefill_span(&forward, &mut sequence, &mut scratch, suffix, prefix_len)?;
                prefill_ms += ms;
                logits
            }
        } else {
            let (logits, ms) = prefill_span(&forward, &mut sequence, &mut scratch, &prompt_ids, 0)?;
            prefill_ms += ms;
            logits
        }
    };
    let prefill_attention_query =
        (scratch.attn_matrix_tiled_layer_calls() > 0).then(|| PrefillAttentionQueryStats {
            outer_chunk_rows: chunk,
            query_rows: scratch.attn_matrix_query_rows(),
            tiled_layer_calls: scratch.attn_matrix_tiled_layer_calls(),
            query_tile_calls: scratch.attn_matrix_query_tile_calls(),
        });
    let prefill_scratch_overlay =
        scratch
            .prefill_scratch_overlay_stats()
            .map(|stats| PrefillScratchOverlayTimingStats {
                backing_bytes: stats.backing_bytes,
                attention_bytes: stats.attention_bytes,
                gdn_bytes: stats.gdn_bytes,
                saved_bytes: stats.saved_bytes,
            });

    let stop_tokens = loaded
        .gguf()
        .stop_token_ids()
        .context("load producer-declared stop tokens")?;
    let sampling_config = prepared.sampling;
    let mut sampler = Sampler::new(sampling_config).context("initialize request sampler")?;
    let greedy_gpu_decision =
        resolve_greedy_gpu_decision(greedy_gpu_mode, sampling_config, args.prompt_lookup);
    let (generation, generated_text, prompt_lookup_stats) = if args.prompt_lookup {
        let (result, generated_text) = decode_prompt_lookup(
            loaded,
            &forward,
            tokenizer,
            sequence,
            prompt_ids,
            logits,
            prompt_ids.len(),
            n_generate,
            &stop_tokens,
        )?;
        sequence = result.sequence;
        (result.generation, generated_text, Some(result.stats))
    } else {
        let (generation, generated_text) = decode_serial(
            &forward,
            tokenizer,
            &mut sequence,
            logits,
            prompt_ids.len(),
            n_generate,
            &stop_tokens,
            &mut sampler,
            greedy_gpu_decision.enabled,
        )?;
        (generation, generated_text, None)
    };
    let stop_reason = generation.stop_reason;
    let generated = generation.tokens;
    let generated_token_sha256 = generated_token_sha256(&generated);
    drop(sequence);
    let decode_ms = generation.wall_ms;
    let decode_tps = if decode_ms > 0.0 {
        generated.len() as f64 / (decode_ms / 1e3)
    } else {
        0.0
    };
    let transition_tps = if generation.transition_ms > 0.0 {
        generation.transitions as f64 / (generation.transition_ms / 1e3)
    } else {
        0.0
    };
    let stats_now = loaded.prefix_cache_stats();
    let finish_ms = unix_epoch_ms_u64()?;
    let total_ms = total_t0.elapsed().as_secs_f64() * 1e3;
    report_prefill_chunk_decision(prefill_chunk_decision.as_ref(), prompt_ids.len());
    let stats = RequestStatsRow {
        schema_version: request_schema_version(
            args.prefill_chunk,
            args.prompt_lookup,
            prefill_attention_query.is_some(),
            prefill_scratch_overlay.is_some(),
            sampling_config.temperature > 0.0,
            false,
            false,
        ),
        id: id.to_string(),
        line: prepared.line,
        model: loaded.path().display().to_string(),
        greedy_gpu_selection_reason: greedy_gpu_decision.reason,
        arrival_ms,
        finish_ms,
        prompt_tokens: prompt_ids.len(),
        prompt_hash,
        requested_tokens: n_generate,
        generated_tokens: generated.len(),
        generated_token_sha256: generated_token_sha256.clone(),
        decode_policy: jsonl_decode_policy_label(
            sampling_config,
            args.prompt_lookup,
            greedy_gpu_decision.enabled,
        ),
        stop_reason,
        terminal_token_target_transition_consumed: false,
        sampling: SamplingTelemetry::sampled(sampling_config, sampler.draws()),
        cache_prefix_tokens,
        cache_prefix_source: cache_prefix_source.as_str().to_string(),
        cache_prefix_hash,
        auto_cache_prefix_tokens: prepared.auto_cache_prefix_tokens,
        auto_cache_future_hits: prepared.auto_cache_future_hits,
        cache_hit,
        matched_prefix_tokens,
        matched_prefix_hash: if matched_prefix_tokens > 0 {
            Some(token_hash_hex(&prompt_ids[..matched_prefix_tokens]))
        } else {
            None
        },
        exact_cache_hit,
        prefill_chunk: match args.prefill_chunk {
            PrefillChunkArg::Fixed(requested) => requested,
            PrefillChunkArg::Auto => chunk,
        },
        prefill_chunk_effective: args.prefill_chunk.is_auto().then_some(chunk),
        prefill_chunk_decision,
        prefill_attention_query,
        prefill_scratch_overlay,
        max_context_tokens: capacity,
        no_special_tokens: args.no_special_tokens,
        restore_ms,
        prefix_inserted_bytes,
        prefix_insert_ms,
        prefill_ms,
        decode_ms,
        model_ttft_ms: restore_ms
            + prefix_insert_ms
            + prefill_ms
            + generation.first_token_ready_ms.unwrap_or(0.0),
        first_token_ms: generation.first_token_ready_ms.unwrap_or(0.0),
        first_token_callback_ms: generation.first_token_callback_ms.unwrap_or(0.0),
        first_decode_ms: generation.first_transition_ms.unwrap_or(0.0),
        decode_tps,
        decode_transitions: generation.transitions,
        transition_ms: generation.transition_ms,
        transition_tps,
        total_ms,
        cache_entries: stats_now.entries,
        cache_bytes: stats_now.indexed_bytes,
        cache_max_bytes: stats_now.max_indexed_bytes,
        prompt_lookup: prompt_lookup_stats,
    };
    let output = RequestOutput {
        id: id.to_string(),
        prompt_tokens: prompt_ids.len(),
        generated_tokens: generated.len(),
        generated_token_sha256,
        generated_text,
        stop_reason,
        terminal_token_target_transition_consumed: false,
    };
    Ok((output, stats))
}

fn request_prompt(request: &JsonlRequest, line: usize) -> Result<String> {
    match (request.prompt.as_ref(), request.prompt_file.as_ref()) {
        (Some(_), Some(_)) => bail!("request line {line} has both prompt and prompt_file"),
        (Some(prompt), None) => Ok(prompt.clone()),
        (None, Some(path)) => std::fs::read_to_string(path)
            .with_context(|| format!("read prompt_file {} on line {line}", path.display())),
        (None, None) => bail!("request line {line} has neither prompt nor prompt_file"),
    }
}

fn prefill_span(
    forward: &MetalForward<'_>,
    sequence: &mut Sequence,
    scratch: &mut MetalDFlashLayerMajorScratch,
    token_ids: &[i32],
    start_position: usize,
) -> Result<(Vec<f32>, f64)> {
    ensure!(!token_ids.is_empty(), "cannot prefill an empty token span");
    sequence.check_position(start_position)?;
    let t0 = Instant::now();
    let logits = prefill_tokens_with_multi_hidden(
        forward,
        token_ids,
        u32::try_from(start_position).context("position does not fit u32")?,
        unsafe { sequence.metal_session_mut() },
        scratch,
        &[],
        None,
    )
    .context("prefill prompt span")?;
    sequence.advance_by(token_ids.len())?;
    Ok((logits, t0.elapsed().as_secs_f64() * 1e3))
}

#[derive(Debug)]
struct GenerationResult {
    tokens: Vec<i32>,
    wall_ms: f64,
    first_token_selection_ms: f64,
    first_token_ready_ms: Option<f64>,
    first_token_callback_ms: Option<f64>,
    transitions: usize,
    transition_ms: f64,
    first_transition_ms: Option<f64>,
    stop_reason: StopReason,
}

fn generate_serial<OnToken, Transition>(
    logits: Vec<f32>,
    max_tokens: usize,
    stop_tokens: &[i32],
    sampler: &mut Sampler,
    on_token: OnToken,
    transition: Transition,
) -> Result<GenerationResult>
where
    OnToken: FnMut(i32) -> Result<()>,
    Transition: FnMut(i32) -> Result<Vec<f32>>,
{
    generate_serial_state(
        logits,
        max_tokens,
        stop_tokens,
        |logits| Ok(sampler.sample(logits)?.token),
        on_token,
        transition,
    )
}

fn generate_serial_attributed<OnToken, Transition>(
    logits: Vec<f32>,
    max_tokens: usize,
    stop_tokens: &[i32],
    sampler: &mut Sampler,
    on_token: OnToken,
    mut transition: Transition,
) -> Result<(
    GenerationResult,
    SamplerAttributionAccumulator,
    TransitionAttributionAccumulator,
)>
where
    OnToken: FnMut(i32) -> Result<()>,
    Transition: FnMut(i32) -> Result<(Vec<f32>, TokenProfile, LogitsReadbackProfile)>,
{
    ensure!(
        sampler.config().temperature > 0.0,
        "sampling attribution requires positive temperature"
    );
    let mut sampler_attribution = SamplerAttributionAccumulator::default();
    let mut transition_attribution = TransitionAttributionAccumulator::default();
    let generation = generate_serial_state(
        logits,
        max_tokens,
        stop_tokens,
        |logits| {
            let (sampled, profile) = sampler.sample_profiled(logits)?;
            sampler_attribution.record(profile)?;
            Ok(sampled.token)
        },
        on_token,
        |token| {
            let (logits, token_profile, readback_profile) = transition(token)?;
            transition_attribution.record(token_profile, readback_profile)?;
            Ok(logits)
        },
    )?;
    Ok((generation, sampler_attribution, transition_attribution))
}

#[derive(Debug)]
enum GreedyDecodeState {
    PromptLogits(Vec<f32>),
    Device(GreedySelection),
}

fn generate_gpu_greedy<OnToken, Transition>(
    logits: Vec<f32>,
    max_tokens: usize,
    stop_tokens: &[i32],
    sampler: &mut Sampler,
    on_token: OnToken,
    mut transition: Transition,
) -> Result<GenerationResult>
where
    OnToken: FnMut(i32) -> Result<()>,
    Transition: FnMut(i32) -> Result<GreedySelection>,
{
    ensure!(
        sampler.config().temperature == 0.0,
        "GPU greedy generation requires temperature zero"
    );
    generate_serial_state(
        GreedyDecodeState::PromptLogits(logits),
        max_tokens,
        stop_tokens,
        |state| match state {
            GreedyDecodeState::PromptLogits(logits) => Ok(sampler.sample(logits)?.token),
            GreedyDecodeState::Device(selection) => {
                selection.into_token().map_err(anyhow::Error::new)
            }
        },
        on_token,
        |token| transition(token).map(GreedyDecodeState::Device),
    )
}

#[derive(Debug)]
enum SampledStructuralDecodeState {
    PromptLogits(Vec<f32>),
    Selected(std::result::Result<SampledToken, SamplingError>),
}

struct SampledStructuralContext<'a> {
    sampler: &'a mut Sampler,
    telemetry: SampledStructuralTelemetry,
}

fn with_transactional_sampled_structural_context<R, F>(
    context: &mut SampledStructuralContext<'_>,
    operation: F,
) -> Result<R>
where
    F: FnOnce(&mut Sampler, &mut SampledStructuralTelemetry) -> Result<R>,
{
    let mut trial_sampler = context.sampler.clone();
    let mut trial_telemetry = context.telemetry.clone();
    let result = operation(&mut trial_sampler, &mut trial_telemetry)?;
    *context.sampler = trial_sampler;
    context.telemetry = trial_telemetry;
    Ok(result)
}

fn generate_sampled_structural<OnToken, Transition>(
    logits: Vec<f32>,
    max_tokens: usize,
    stop_tokens: &[i32],
    sampler: &mut Sampler,
    on_token: OnToken,
    mut transition: Transition,
) -> Result<(GenerationResult, SampledStructuralTelemetry)>
where
    OnToken: FnMut(i32) -> Result<()>,
    Transition: FnMut(
        i32,
        &mut Sampler,
        &mut SampledStructuralTelemetry,
    ) -> Result<SampledStructuralDecodeState>,
{
    ensure!(
        sampler.config().temperature > 0.0 && sampler.config().top_k > 0,
        "sampled structural generation requires positive temperature and top-k"
    );
    let mut context = SampledStructuralContext {
        sampler,
        telemetry: SampledStructuralTelemetry::default(),
    };
    let generation = generate_serial_state_with_context(
        SampledStructuralDecodeState::PromptLogits(logits),
        max_tokens,
        stop_tokens,
        &mut context,
        |context, state| match state {
            SampledStructuralDecodeState::PromptLogits(logits) => {
                with_transactional_sampled_structural_context(context, |sampler, telemetry| {
                    let (sampled, evidence) = sampler.sample_bounded_top_k(logits)?;
                    ensure!(
                        evidence.used_bounded_path,
                        "sampled structural prompt selection fell back"
                    );
                    telemetry.record_prompt(evidence)?;
                    Ok(sampled.token)
                })
            }
            SampledStructuralDecodeState::Selected(result) => match result {
                Ok(sampled) => Ok(sampled.token),
                Err(error) => Err(anyhow::Error::new(error.clone())),
            },
        },
        on_token,
        |context, token| {
            with_transactional_sampled_structural_context(context, |sampler, telemetry| {
                transition(token, sampler, telemetry)
            })
        },
    )?;
    Ok((generation, context.telemetry))
}

fn generate_serial_state<State, Select, OnToken, Transition>(
    state: State,
    max_tokens: usize,
    stop_tokens: &[i32],
    mut select: Select,
    on_token: OnToken,
    mut transition: Transition,
) -> Result<GenerationResult>
where
    Select: FnMut(&State) -> Result<i32>,
    OnToken: FnMut(i32) -> Result<()>,
    Transition: FnMut(i32) -> Result<State>,
{
    let mut context = ();
    generate_serial_state_with_context(
        state,
        max_tokens,
        stop_tokens,
        &mut context,
        |_, state| select(state),
        on_token,
        |_, token| transition(token),
    )
}

fn generate_serial_state_with_context<Context, State, Select, OnToken, Transition>(
    mut state: State,
    max_tokens: usize,
    stop_tokens: &[i32],
    context: &mut Context,
    mut select: Select,
    mut on_token: OnToken,
    mut transition: Transition,
) -> Result<GenerationResult>
where
    Select: FnMut(&mut Context, &State) -> Result<i32>,
    OnToken: FnMut(i32) -> Result<()>,
    Transition: FnMut(&mut Context, i32) -> Result<State>,
{
    ensure!(max_tokens > 0, "max_tokens must be >= 1");
    let wall_t0 = Instant::now();
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut first_token_selection_ms = None;
    let mut first_token_ready_ms = None;
    let mut first_token_callback_ms = None;
    let mut transitions = 0usize;
    let mut transition_ms = 0.0;
    let mut first_transition_ms = None;
    let mut stop_reason = None;

    while tokens.len() < max_tokens {
        let selection_t0 = Instant::now();
        let token = select(context, &state)?;
        first_token_selection_ms.get_or_insert_with(|| selection_t0.elapsed().as_secs_f64() * 1e3);
        first_token_ready_ms.get_or_insert_with(|| wall_t0.elapsed().as_secs_f64() * 1e3);
        tokens.push(token);

        if stop_tokens.contains(&token) {
            stop_reason = Some(StopReason::Eos);
            break;
        }
        on_token(token)?;
        first_token_callback_ms.get_or_insert_with(|| wall_t0.elapsed().as_secs_f64() * 1e3);
        if tokens.len() == max_tokens {
            stop_reason = Some(StopReason::TokenLimit);
            break;
        }

        let transition_t0 = Instant::now();
        state = transition(context, token)?;
        let elapsed_ms = transition_t0.elapsed().as_secs_f64() * 1e3;
        transition_ms += elapsed_ms;
        first_transition_ms.get_or_insert(elapsed_ms);
        transitions += 1;
    }

    Ok(GenerationResult {
        tokens,
        wall_ms: wall_t0.elapsed().as_secs_f64() * 1e3,
        first_token_selection_ms: first_token_selection_ms.unwrap_or(0.0),
        first_token_ready_ms,
        first_token_callback_ms,
        transitions,
        transition_ms,
        first_transition_ms,
        stop_reason: stop_reason.expect("positive max_tokens must select a terminal token"),
    })
}

#[cfg(test)]
fn generate_greedy<OnToken, Transition>(
    logits: Vec<f32>,
    max_tokens: usize,
    stop_tokens: &[i32],
    on_token: OnToken,
    transition: Transition,
) -> Result<GenerationResult>
where
    OnToken: FnMut(i32) -> Result<()>,
    Transition: FnMut(i32) -> Result<Vec<f32>>,
{
    let mut sampler = Sampler::new(SamplingConfig::default()).expect("valid greedy sampler");
    generate_serial(
        logits,
        max_tokens,
        stop_tokens,
        &mut sampler,
        on_token,
        transition,
    )
}

struct PromptLookupGeneration {
    generation: GenerationResult,
    stats: PromptLookupDecodeStats,
    sequence: Sequence,
}

struct PromptLookupScratch {
    verify: MetalDFlashVerifyScratch,
    layer: MetalDFlashLayerMajorScratch,
}

fn generate_prompt_lookup<OnToken>(
    loaded: &LoadedModel,
    forward: &MetalForward<'_>,
    mut sequence: Sequence,
    prompt_ids: &[i32],
    logits: Vec<f32>,
    max_tokens: usize,
    stop_tokens: &[i32],
    mut on_token: OnToken,
) -> Result<PromptLookupGeneration>
where
    OnToken: FnMut(i32) -> Result<()>,
{
    ensure!(max_tokens > 0, "max_tokens must be >= 1");
    let wall_t0 = Instant::now();
    let selection_t0 = Instant::now();
    let mut carry = argmax_i32(&logits);
    let first_token_selection_ms = selection_t0.elapsed().as_secs_f64() * 1e3;
    let first_token_ready_ms = Some(wall_t0.elapsed().as_secs_f64() * 1e3);
    let mut first_token_callback_ms = None;
    let mut first_transition_ms = None;
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut transitions = 0usize;
    let mut proposer = None;
    let mut scratch = None;
    let mut stats = PromptLookupDecodeStats::default();
    let mut transition_wall_ms = 0.0;
    let mut post_callback_policy_ms = 0.0;

    let stop_reason = 'outer: loop {
        tokens.push(carry);

        if stop_tokens.contains(&carry) {
            break StopReason::Eos;
        }
        on_token(carry)?;
        first_token_callback_ms.get_or_insert_with(|| wall_t0.elapsed().as_secs_f64() * 1e3);
        if tokens.len() == max_tokens {
            break StopReason::TokenLimit;
        }

        let transition_t0 = Instant::now();
        if proposer.is_none() {
            let index_t0 = Instant::now();
            proposer = Some(PromptLookupProposer::new(prompt_ids));
            stats.index_build_ms += index_t0.elapsed().as_secs_f64() * 1e3;
        }
        let proposer = proposer
            .as_mut()
            .expect("prompt lookup proposer initialized");
        let update_t0 = Instant::now();
        proposer.commit_verified(&[carry]);
        stats.index_update_ms += update_t0.elapsed().as_secs_f64() * 1e3;

        let lookup_t0 = Instant::now();
        let candidate = proposer.propose();
        stats.lookup_ms += lookup_t0.elapsed().as_secs_f64() * 1e3;
        let Some(candidate) = candidate else {
            stats.abstentions += 1;
            sequence.ensure_can_append(1)?;
            let position =
                u32::try_from(sequence.position()).context("position does not fit u32")?;
            let serial_t0 = Instant::now();
            let next = forward
                .single_token(carry, position, unsafe { sequence.metal_session_mut() })
                .context("prompt-lookup serial decode")?;
            stats.serial_ms += serial_t0.elapsed().as_secs_f64() * 1e3;
            stats.physical_target_positions += 1;
            sequence.advance_by(1)?;
            transitions += 1;
            carry = argmax_i32(&next);
            let elapsed_ms = transition_t0.elapsed().as_secs_f64() * 1e3;
            transition_wall_ms += elapsed_ms;
            first_transition_ms.get_or_insert(elapsed_ms);
            continue;
        };

        stats.attempts += 1;
        let terminal_window =
            terminal_draft_window(&candidate.proposal, tokens.len(), max_tokens, stop_tokens);
        let mut verify_input = Vec::with_capacity(DRAFT_TOKENS + 1);
        verify_input.push(carry);
        let (n_eff, n_drafts_scored) = if let Some(window) = terminal_window {
            verify_input.extend_from_slice(&candidate.proposal[..window.count.saturating_sub(1)]);
            (window.count, window.count)
        } else {
            verify_input.extend_from_slice(&candidate.proposal);
            (DRAFT_TOKENS + 1, DRAFT_TOKENS)
        };
        ensure!(n_eff > 0, "prompt-lookup verifier planned an empty chain");
        sequence.ensure_can_append(n_eff)?;
        let start_position =
            u32::try_from(sequence.position()).context("position does not fit u32")?;

        if scratch.is_none() {
            let before = loaded.context().current_allocated_size();
            let allocation_t0 = Instant::now();
            let verify = MetalDFlashVerifyScratch::fresh(
                loaded.context(),
                loaded.metal_model(),
                (DRAFT_TOKENS + 1) as u32,
                0,
            )
            .context("allocate prompt-lookup verify scratch")?;
            let layer = MetalDFlashLayerMajorScratch::fresh(
                loaded.context(),
                loaded.metal_model(),
                (DRAFT_TOKENS + 1) as u32,
            )
            .context("allocate prompt-lookup layer scratch")?;
            stats.scratch_allocation_ms += allocation_t0.elapsed().as_secs_f64() * 1e3;
            let after = loaded.context().current_allocated_size();
            stats.scratch_allocated_bytes = after.saturating_sub(before);
            stats.scratch_peak_allocated_bytes = after;
            scratch = Some(PromptLookupScratch { verify, layer });
        }
        let scratch = scratch.as_mut().expect("prompt lookup scratch initialized");
        let verify_t0 = Instant::now();
        let verify_argmax = qwen_llm::metal_dflash::encode_packed_verify_layer_major_inner(
            forward,
            &[],
            &verify_input,
            start_position,
            &mut scratch.verify,
            &mut scratch.layer,
            unsafe { sequence.metal_session_mut() },
            None,
            Some(n_eff as u32),
        )
        .context("prompt-lookup packed verify")?;
        stats.verify_ms += verify_t0.elapsed().as_secs_f64() * 1e3;
        stats.verify_calls += 1;
        stats.drafts_scored += n_drafts_scored;
        stats.physical_target_positions += n_eff;

        let mut accepted = Vec::with_capacity(n_drafts_scored);
        let mut terminal = None;
        for (&draft, &target) in candidate.proposal[..n_drafts_scored]
            .iter()
            .zip(&verify_argmax)
        {
            if draft != target {
                break;
            }
            accepted.push(draft);
            if stop_tokens.contains(&draft) {
                terminal = Some(StopReason::Eos);
                break;
            }
            if tokens.len() + accepted.len() == max_tokens {
                terminal = Some(StopReason::TokenLimit);
                break;
            }
        }
        let n_accepted = accepted.len();
        let n_keep = if terminal.is_some() {
            n_accepted
        } else {
            1 + n_accepted
        };
        ensure!(
            n_keep > 0,
            "prompt-lookup terminal plan retained no target state"
        );
        if n_keep < n_eff {
            let restore_t0 = Instant::now();
            qwen_llm::metal_dflash::encode_restore_after_partial_accept_inner(
                forward,
                &scratch.verify,
                n_keep as u32,
                start_position,
                unsafe { sequence.metal_session_mut() },
                Some(n_eff as u32),
            )
            .context("prompt-lookup restore after partial accept")?;
            stats.restore_ms += restore_t0.elapsed().as_secs_f64() * 1e3;
            stats.restore_calls += 1;
        }
        sequence.advance_by(n_keep)?;
        transitions += n_keep;
        stats.accepted_drafts += n_accepted;
        let elapsed_ms = transition_t0.elapsed().as_secs_f64() * 1e3;
        transition_wall_ms += elapsed_ms;
        first_transition_ms.get_or_insert(elapsed_ms);

        for token in accepted {
            tokens.push(token);
            if !stop_tokens.contains(&token) {
                on_token(token)?;
            }
            let update_t0 = Instant::now();
            proposer.commit_verified(&[token]);
            let update_ms = update_t0.elapsed().as_secs_f64() * 1e3;
            stats.index_update_ms += update_ms;
            post_callback_policy_ms += update_ms;
        }
        if let Some(reason) = terminal {
            break 'outer reason;
        }
        carry = verify_argmax[n_accepted];
    };

    ensure!(
        transitions.checked_add(1) == Some(tokens.len()),
        "prompt-lookup generation violated N-1 transition semantics"
    );
    let transition_ms = transition_wall_ms + post_callback_policy_ms;
    Ok(PromptLookupGeneration {
        generation: GenerationResult {
            tokens,
            wall_ms: wall_t0.elapsed().as_secs_f64() * 1e3,
            first_token_selection_ms,
            first_token_ready_ms,
            first_token_callback_ms,
            transitions,
            transition_ms,
            first_transition_ms,
            stop_reason,
        },
        stats,
        sequence,
    })
}

fn decode_serial(
    forward: &MetalForward<'_>,
    tokenizer: &Tokenizer,
    sequence: &mut Sequence,
    logits: Vec<f32>,
    start_position: usize,
    max_tokens: usize,
    stop_tokens: &[i32],
    sampler: &mut Sampler,
    use_gpu_greedy: bool,
) -> Result<(GenerationResult, String)> {
    sequence.check_position(start_position)?;
    let mut generated_text = String::new();
    let mut on_token = |token| {
        generated_text.push_str(&tokenizer.decode_piece(token));
        Ok(())
    };
    let generation = if use_gpu_greedy {
        generate_gpu_greedy(
            logits,
            max_tokens,
            stop_tokens,
            sampler,
            &mut on_token,
            |token| {
                let position = sequence.position();
                let next = forward
                    .single_token_greedy(
                        token,
                        u32::try_from(position).context("position does not fit u32")?,
                        unsafe { sequence.metal_session_mut() },
                    )
                    .context("decode token with GPU greedy selection")?;
                sequence.advance_by(1)?;
                Ok(next)
            },
        )?
    } else {
        generate_serial(
            logits,
            max_tokens,
            stop_tokens,
            sampler,
            &mut on_token,
            |token| {
                let position = sequence.position();
                let next = forward
                    .single_token(
                        token,
                        u32::try_from(position).context("position does not fit u32")?,
                        unsafe { sequence.metal_session_mut() },
                    )
                    .context("decode token")?;
                sequence.advance_by(1)?;
                Ok(next)
            },
        )?
    };
    Ok((generation, generated_text))
}

fn decode_prompt_lookup(
    loaded: &LoadedModel,
    forward: &MetalForward<'_>,
    tokenizer: &Tokenizer,
    sequence: Sequence,
    prompt_ids: &[i32],
    logits: Vec<f32>,
    start_position: usize,
    max_tokens: usize,
    stop_tokens: &[i32],
) -> Result<(PromptLookupGeneration, String)> {
    sequence.check_position(start_position)?;
    let mut generated_text = String::new();
    let result = generate_prompt_lookup(
        loaded,
        forward,
        sequence,
        prompt_ids,
        logits,
        max_tokens,
        stop_tokens,
        |token| {
            generated_text.push_str(&tokenizer.decode_piece(token));
            Ok(())
        },
    )?;
    Ok((result, generated_text))
}

fn prefix_cache_max_bytes(args: &Args) -> Result<u64> {
    mib_to_bytes(args.prefix_cache_max_mib, "prefix cache byte budget")
}

fn durable_checkpoint_store(
    args: &Args,
    staged_integrity: Option<StagedIntegrityMode>,
) -> Result<Option<DurableCheckpointStore>> {
    let Some(root) = args.durable_prefix_cache.as_ref() else {
        return Ok(None);
    };
    let budget = mib_to_bytes(
        args.durable_prefix_cache_max_mib,
        "durable prefix cache byte budget",
    )?;
    Ok(Some(match staged_integrity {
        Some(mode) => DurableCheckpointStore::with_staged_integrity(root, budget, mode),
        None => DurableCheckpointStore::new(root, budget),
    }))
}

fn configured_checkpoint_staged_integrity() -> Result<Option<StagedIntegrityMode>> {
    match std::env::var(CHECKPOINT_STAGED_INTEGRITY_ENV) {
        Ok(value) => parse_checkpoint_staged_integrity(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_checkpoint_staged_integrity(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            bail!("invalid {CHECKPOINT_STAGED_INTEGRITY_ENV}; expected decode or deferred-restore")
        }
    }
}

fn parse_checkpoint_staged_integrity(value: Option<&str>) -> Result<Option<StagedIntegrityMode>> {
    match value {
        None => Ok(None),
        Some(value) => StagedIntegrityMode::parse(value).map(Some).ok_or_else(|| {
            anyhow!(
                "invalid {CHECKPOINT_STAGED_INTEGRITY_ENV}; expected decode or deferred-restore"
            )
        }),
    }
}

fn durable_prefix_cache_max_entry_bytes(args: &Args) -> Result<u64> {
    mib_to_bytes(
        args.durable_prefix_cache_max_entry_mib,
        "durable prefix cache per-entry budget",
    )
}

fn mib_to_bytes(mib: u64, label: &str) -> Result<u64> {
    mib.checked_mul(1024 * 1024)
        .with_context(|| format!("{label} overflow"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DurableCapturePolicy {
    Disabled,
    ExplicitPrompt(usize),
    AutomaticPrompt(usize),
    AutomaticCompleted,
}

impl DurableCapturePolicy {
    fn prompt_prefix_len(self) -> Option<usize> {
        match self {
            Self::ExplicitPrompt(len) | Self::AutomaticPrompt(len) => Some(len),
            Self::Disabled | Self::AutomaticCompleted => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompletedCheckpointBoundary {
    consumed_generated_tokens: usize,
    consumed_prefix_len: usize,
    pending_token: i32,
}

impl CompletedCheckpointBoundary {
    fn consumed_tokens(self, prompt_ids: &[i32], generated: &[i32]) -> Vec<i32> {
        let mut consumed = Vec::with_capacity(self.consumed_prefix_len);
        consumed.extend_from_slice(prompt_ids);
        consumed.extend_from_slice(&generated[..self.consumed_generated_tokens]);
        consumed
    }
}

fn derive_completed_checkpoint_boundary(
    prompt_len: usize,
    generated: &[i32],
    transitions: usize,
    sequence_position: usize,
) -> Result<CompletedCheckpointBoundary> {
    ensure!(!generated.is_empty(), "completed generation has no tokens");
    ensure!(
        transitions.checked_add(1) == Some(generated.len()),
        "completed generation transitions {} do not match token count {}",
        transitions,
        generated.len()
    );
    let consumed_prefix_len = prompt_len
        .checked_add(transitions)
        .context("completed checkpoint prefix length overflow")?;
    ensure!(
        sequence_position == consumed_prefix_len,
        "completed sequence position {} does not match consumed prefix length {}",
        sequence_position,
        consumed_prefix_len
    );
    Ok(CompletedCheckpointBoundary {
        consumed_generated_tokens: transitions,
        consumed_prefix_len,
        pending_token: generated[transitions],
    })
}

fn selected_single_turn_durable_policy(
    args: &Args,
    prompt_len: usize,
    completed_checkpoint_eligible: bool,
) -> DurableCapturePolicy {
    if let Some(configured) = args.cache_prefix_tokens {
        return if configured > 0 && prompt_len > 0 {
            DurableCapturePolicy::ExplicitPrompt(configured.min(prompt_len))
        } else {
            DurableCapturePolicy::Disabled
        };
    }
    if args.durable_prefix_cache_min_tokens == 0
        || prompt_len < args.durable_prefix_cache_min_tokens
    {
        return DurableCapturePolicy::Disabled;
    }
    if completed_checkpoint_eligible {
        DurableCapturePolicy::AutomaticCompleted
    } else {
        DurableCapturePolicy::AutomaticPrompt(prompt_len)
    }
}

fn selected_single_turn_durable_lookup_len(args: &Args, prompt_len: usize) -> usize {
    args.cache_prefix_tokens
        .filter(|&configured| configured > 0)
        .map_or(prompt_len, |configured| configured.min(prompt_len))
}

fn identity_cache_outcome_label(outcome: IdentityCacheOutcome) -> &'static str {
    match outcome {
        IdentityCacheOutcome::Hit => "hit",
        IdentityCacheOutcome::ComputedAndStored => "computed_stored",
        IdentityCacheOutcome::ComputedAndRepaired => "computed_repaired",
        IdentityCacheOutcome::ComputedUncached => "computed_uncached",
    }
}

fn publish_outcome_label(outcome: PublishOutcome) -> &'static str {
    match outcome {
        PublishOutcome::Published => "published",
        PublishOutcome::ExistingValid => "existing_valid",
        PublishOutcome::RepairedCorrupt => "repaired_corrupt",
    }
}

fn unix_epoch_ms_u64() -> Result<u64> {
    let ms = unix_epoch_ms()?;
    u64::try_from(ms).context("Unix epoch milliseconds do not fit u64")
}

const TOKEN_HASH_SEED: u64 = 0xcbf29ce484222325;
const TOKEN_HASH_PRIME: u64 = 0x100000001b3;

fn token_hash_hex(tokens: &[i32]) -> String {
    let mut hash = TOKEN_HASH_SEED;
    for &token in tokens {
        hash ^= (token as u32 as u64).wrapping_add(0x9e3779b97f4a7c15);
        hash = hash.wrapping_mul(TOKEN_HASH_PRIME);
    }
    format!("{hash:016x}")
}

fn generated_token_sha256(tokens: &[i32]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"qwen-generated-token-ids-v1\0");
    digest.update((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn open_append_file(path: &Path, label: &str) -> Result<std::fs::File> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {label} directory {}", parent.display()))?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {label} {}", path.display()))
}

fn unix_epoch_ms() -> Result<u128> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_millis())
}

fn append_request_trace(
    path: &Path,
    arrival_ms: u128,
    prompt_tokens: usize,
    generated_tokens: usize,
) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create trace directory {}", parent.display()))?;
    }
    let write_header = std::fs::metadata(path).map_or(true, |m| m.len() == 0);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open request trace {}", path.display()))?;
    if write_header {
        writeln!(file, "arrival_ms\ttokens\tid\tprompt_tokens")?;
    }
    let id = format!("{}-{arrival_ms}", std::process::id());
    writeln!(
        file,
        "{arrival_ms}\t{generated_tokens}\t{id}\t{prompt_tokens}"
    )?;
    Ok(())
}

fn argmax_i32(xs: &[f32]) -> i32 {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

fn print_model_info(model_path: &Path) -> Result<()> {
    let gguf = qwen_llm::gguf::GgufFile::open(&model_path)?;
    println!(
        "loaded {}: arch={} {} tensors, {} shard(s), mmap={} MiB, primary tensor-data starts at {}",
        model_path.display(),
        gguf.architecture().unwrap_or_else(|| "?".into()),
        gguf.tensors.len(),
        gguf.shard_count(),
        gguf.total_mapped_len() / (1024 * 1024),
        gguf.primary_shard().tensor_data_start,
    );

    if ModelFamily::detect(&gguf) == Some(ModelFamily::DeepSeek4) {
        return print_deepseek_v4_info(&gguf);
    }

    // Group tensors by layer index. The GDN-layer test is "has ssm_* tensor",
    // the full-attn-layer test is "has attn_q/k/v/o.weight" (NOT attn_qkv,
    // which is GDN's combined input projection in this naming scheme).
    use std::collections::BTreeMap;
    let mut by_layer: BTreeMap<u32, Vec<&str>> = BTreeMap::new();
    for t in &gguf.tensors {
        if let Some(rest) = t.name.strip_prefix("blk.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(idx) = rest[..dot].parse::<u32>() {
                    by_layer.entry(idx).or_default().push(&rest[dot + 1..]);
                }
            }
        }
    }
    let n_layers = by_layer.len();
    let mut gdn = 0usize;
    let mut attn = 0usize;
    for tensors in by_layer.values() {
        let has_ssm = tensors.iter().any(|s| s.starts_with("ssm_"));
        let has_attn_q = tensors
            .iter()
            .any(|s| *s == "attn_q.weight" || *s == "attn_qkv_real.weight");
        if has_ssm {
            gdn += 1;
        } else if has_attn_q {
            attn += 1;
        }
    }
    println!("blocks: {n_layers} total — {gdn} GDN, {attn} full-attn");

    // Show layer-0 and layer-3 tensor inventories: 0 should be GDN, 3 full-attn.
    for sample in [0u32, 3] {
        if let Some(t) = by_layer.get(&sample) {
            println!("blk.{sample} tensors ({}):", t.len());
            for name in t {
                println!("  blk.{sample}.{name}");
            }
        }
    }

    // Show metadata keys related to the architecture.
    let interesting_keys = [
        "qwen35.block_count",
        "qwen35.attention.head_count",
        "qwen35.attention.head_count_kv",
        "qwen35.attention.key_length",
        "qwen35.attention.value_length",
        "qwen35.embedding_length",
        "qwen35.feed_forward_length",
        "qwen35.context_length",
        "qwen35.ssm.conv_kernel",
        "qwen35.ssm.inner_size",
        "qwen35.ssm.state_size",
        "qwen35.ssm.time_step_rank",
        "qwen35.ssm.group_count",
    ];
    println!("relevant metadata:");
    for k in interesting_keys {
        if let Some(v) = gguf.get_u64(k) {
            println!("  {k} = {v}");
        }
    }

    Ok(())
}

fn print_deepseek_v4_info(gguf: &qwen_llm::gguf::GgufFile) -> Result<()> {
    use std::collections::BTreeSet;

    let model = DeepSeekV4Model::from_gguf_flash_0731(gguf)
        .context("bind strict DeepSeek V4 Flash-0731 schema")?;
    let config = &model.config;
    let (local, csa, hca) = config.attention_counts();
    let hash_layers = model
        .blocks
        .iter()
        .filter(|block| matches!(block.moe.router, RouterWeights::TokenHash { .. }))
        .count();
    let csa_layers = model
        .blocks
        .iter()
        .filter(|block| matches!(block.attention.lane, AttentionLane::CompressedSparse { .. }))
        .count();
    let gate_up_types = model
        .blocks
        .iter()
        .flat_map(|block| [block.moe.gate_experts.dtype, block.moe.up_experts.dtype])
        .map(|dtype| dtype.to_string())
        .collect::<BTreeSet<_>>();
    let down_types = model
        .blocks
        .iter()
        .map(|block| block.moe.down_experts.dtype.to_string())
        .collect::<BTreeSet<_>>();

    println!(
        "deepseek4 target: {} layers, hidden={}, vocab={}, context={}",
        config.layer_count, config.hidden_size, config.vocab_size, config.context_length
    );
    println!(
        "attention: {local} local, {csa} CSA ratio-4, {hca} HCA ratio-128; heads={} shared-KV={}x{} local-window={} index-topk={}",
        config.attention_head_count,
        config.kv_head_count,
        config.key_length,
        config.sliding_window,
        config.indexer_top_k,
    );
    println!(
        "mHC: streams={} sinkhorn-iters={} epsilon={}; MoE: experts={} topk={} hash-layers={hash_layers}",
        config.hyper_connection_count,
        config.sinkhorn_iterations,
        config.hyper_connection_epsilon,
        config.expert_count,
        config.expert_used_count,
    );
    println!(
        "tokenizer: {}/{} bos={:?} eos={:?} pad={:?}",
        config.tokenizer_model,
        config.tokenizer_pre,
        config.bos_token_id,
        config.eos_token_id,
        config.padding_token_id,
    );
    println!(
        "trailing compression-ratio entries={} (target-only GGUF); routed gate/up types={gate_up_types:?}, down types={down_types:?}",
        config.compress_ratio_tail.len(),
    );
    println!(
        "strict tensor schema: validated all {} tensors; CSA indexers={csa_layers}",
        model.source_tensor_count
    );
    Ok(())
}

fn print_deepseek_v4_census(model_path: &Path) -> Result<()> {
    let gguf = qwen_llm::gguf::GgufFile::open(model_path)
        .with_context(|| format!("open DeepSeek V4 model {}", model_path.display()))?;
    ensure!(
        ModelFamily::detect(&gguf) == Some(ModelFamily::DeepSeek4),
        "--deepseek-census-json requires general.architecture=deepseek4"
    );
    let census = DeepSeekV4CensusV1::from_gguf_flash_0731(&gguf)
        .context("construct DeepSeek V4 schema/quant census")?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &census)?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum AllocationEvent {
        CurrentAllocated,
        LegacyScratch(usize),
        Sequence(usize),
        CandidatePlan,
        MemorySignals,
        CandidateScratch,
    }

    struct FakePrefillAllocator {
        events: Vec<AllocationEvent>,
        current_allocated: VecDeque<u64>,
        plan_result: std::result::Result<PrefillPlanDecision, CandidatePlanFailure>,
        signals: MetalMemorySignals,
        candidate_error: bool,
    }

    impl PrefillRequestAllocator for FakePrefillAllocator {
        type Scratch = &'static str;
        type Sequence = &'static str;
        type Plan = ();

        fn current_allocated_size(&mut self) -> u64 {
            self.events.push(AllocationEvent::CurrentAllocated);
            self.current_allocated
                .pop_front()
                .expect("fake current allocation sample")
        }

        fn allocate_legacy_scratch(
            &mut self,
            chunk: usize,
            _prompt_tokens: usize,
        ) -> Result<Self::Scratch> {
            self.events.push(AllocationEvent::LegacyScratch(chunk));
            Ok("legacy")
        }

        fn allocate_sequence(&mut self, capacity: usize) -> Result<Self::Sequence> {
            self.events.push(AllocationEvent::Sequence(capacity));
            Ok("sequence")
        }

        fn build_candidate_plan(
            &mut self,
            _profile: AutoPrefillProfile,
            _prompt_tokens: usize,
        ) -> std::result::Result<(Self::Plan, PrefillPlanDecision), CandidatePlanFailure> {
            self.events.push(AllocationEvent::CandidatePlan);
            self.plan_result.clone().map(|decision| ((), decision))
        }

        fn memory_signals(&mut self) -> MetalMemorySignals {
            self.events.push(AllocationEvent::MemorySignals);
            self.signals
        }

        fn allocate_candidate_scratch(&mut self, _plan: Self::Plan) -> Result<Self::Scratch> {
            self.events.push(AllocationEvent::CandidateScratch);
            if self.candidate_error {
                bail!("candidate allocation failed")
            }
            Ok("candidate")
        }
    }

    fn test_auto_profile() -> AutoPrefillProfile {
        AutoPrefillProfile {
            name: "test-profile",
            outer_chunk: 2048,
            query_heads: 16,
            gdn_overlay_bytes: 235_405_312,
        }
    }

    fn test_a3b_arch() -> Arch {
        let mut arch = qwen_llm::model::QWEN3_27B;
        arch.kind = ArchKind::Moe;
        arch.n_layer = 40;
        arch.hidden_size = 2048;
        arch.n_q_heads = 16;
        arch.n_kv_heads = 2;
        arch.gdn_n_v_heads = 32;
        arch.expert_count = 256;
        arch.expert_used_count = 8;
        arch.expert_feed_forward_length = 512;
        arch.expert_shared_feed_forward_length = 512;
        arch.mtp_n_hidden_layers = 0;
        arch
    }

    fn test_a10b_arch() -> Arch {
        let mut arch = test_a3b_arch();
        arch.n_layer = 48;
        arch.hidden_size = 3072;
        arch.n_q_heads = 32;
        arch.gdn_n_v_heads = 64;
        arch.expert_feed_forward_length = 1024;
        arch.expert_shared_feed_forward_length = 1024;
        arch
    }

    fn fake_plan_decision(priced_upper_bytes: u64) -> PrefillPlanDecision {
        PrefillPlanDecision {
            block_size: 2048,
            matrix_max_pos: 10_000,
            matrix_query_rows: 1024,
            eager_allocation_count: 1,
            deferred_allocation_count: 1,
            eager_logical_bytes: 10,
            deferred_logical_bytes: 20,
            maximum_logical_bytes: 30,
            priced_upper_bytes,
            overlay: PrefillScratchOverlayTimingStats {
                backing_bytes: 1,
                attention_bytes: 1,
                gdn_bytes: 1,
                saved_bytes: 1,
            },
            allocations: vec![
                PrefillPlanAllocationDecision {
                    name: "eager",
                    deferred: false,
                    logical_bytes: 10,
                    priced_bytes: 10,
                    alignment: 256,
                },
                PrefillPlanAllocationDecision {
                    name: "deferred",
                    deferred: true,
                    logical_bytes: 20,
                    priced_bytes: priced_upper_bytes.saturating_sub(10),
                    alignment: 256,
                },
            ],
        }
    }

    fn fake_allocator(
        current_allocated: &[u64],
        plan_result: std::result::Result<PrefillPlanDecision, CandidatePlanFailure>,
        recommended_max_bytes: u64,
        candidate_error: bool,
    ) -> FakePrefillAllocator {
        fake_allocator_with_signals(
            current_allocated,
            plan_result,
            MetalMemorySignals {
                recommended_max_bytes,
                current_allocated_bytes: current_allocated.get(1).copied().unwrap_or(0),
                process_limit_remaining_bytes: Some(0),
            },
            candidate_error,
        )
    }

    fn fake_allocator_with_signals(
        current_allocated: &[u64],
        plan_result: std::result::Result<PrefillPlanDecision, CandidatePlanFailure>,
        signals: MetalMemorySignals,
        candidate_error: bool,
    ) -> FakePrefillAllocator {
        FakePrefillAllocator {
            events: Vec::new(),
            current_allocated: current_allocated.iter().copied().collect(),
            plan_result,
            signals,
            candidate_error,
        }
    }

    fn logits_with_argmax(token: usize) -> Vec<f32> {
        let mut logits = vec![0.0; 4];
        logits[token] = 1.0;
        logits
    }

    fn fake_structural_row_evidence() -> StructuralRowEvidence {
        StructuralRowEvidence {
            resident_head_wait_calls: 1,
            validated_shared_row_calls: 1,
            transition_logits_copy_bytes: 0,
            extra_command_buffers: 0,
            gpu_sampling_dispatches: 0,
        }
    }

    fn fake_structural_transition<Advance>(
        trial: &mut Sampler,
        telemetry: &mut SampledStructuralTelemetry,
        logits: &[f32],
        advance: Advance,
    ) -> Result<SampledStructuralDecodeState>
    where
        Advance: FnOnce() -> Result<()>,
    {
        let sampled = trial.sample_bounded_top_k(logits);
        let state = match sampled {
            Ok((sampled, evidence)) => {
                telemetry.record_transition(evidence, fake_structural_row_evidence())?;
                SampledStructuralDecodeState::Selected(Ok(sampled))
            }
            Err(error) => SampledStructuralDecodeState::Selected(Err(error)),
        };
        advance()?;
        Ok(state)
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

    #[test]
    fn deepseek_v4_forward_budget_accounts_for_unconsumed_final_token() {
        assert_eq!(
            deepseek_v4_required_forwards(1, DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY).unwrap(),
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY
        );
        assert_eq!(
            deepseek_v4_required_forwards(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY, 1).unwrap(),
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY
        );
        assert!(
            deepseek_v4_required_forwards(1, DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY + 1).is_err()
        );
        assert!(deepseek_v4_required_forwards(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY, 2).is_err());
        assert!(deepseek_v4_required_forwards(0, 1).is_err());
        assert!(deepseek_v4_required_forwards(1, 0).is_err());
        assert!(deepseek_v4_required_forwards(usize::MAX, 2).is_err());
    }

    #[test]
    fn deepseek_v4_prefill_chunks_every_retained_prompt_interval() {
        assert_eq!(deepseek_v4_packed_chunk_count(0), 0);
        assert_eq!(deepseek_v4_packed_chunk_count(1), 0);
        assert_eq!(deepseek_v4_packed_chunk_count(2), 1);
        assert_eq!(
            deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PREFILL_MAX_TOKENS),
            1
        );
        assert_eq!(
            deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PREFILL_MAX_TOKENS + 1),
            2
        );
        assert_eq!(
            deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY),
            9
        );
    }

    #[test]
    fn deepseek_v4_cli_accepts_only_bounded_single_turn_surfaces() {
        let raw = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--temp",
            "0.7",
            "--trace-request",
            "trace.txt",
            "--no-special-tokens",
        ])
        .unwrap();
        validate_deepseek_v4_generation_mode(&raw, ExplicitCliOptions::default()).unwrap();

        let unsupported = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--prompt-lookup",
            "--prefill-chunk",
            "auto",
            "--max-context-tokens",
            "255",
            "--cache-prefix-tokens",
            "4",
            "--request-stats",
            "stats.jsonl",
        ])
        .unwrap();
        let error =
            validate_deepseek_v4_generation_mode(&unsupported, ExplicitCliOptions::default())
                .unwrap_err()
                .to_string();
        for option in [
            "--prompt-lookup",
            "--prefill-chunk",
            "--max-context-tokens",
            "--cache-prefix-tokens",
            "--request-stats",
        ] {
            assert!(error.contains(option), "missing {option:?} from {error:?}");
        }

        let messages = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "messages.json",
        ])
        .unwrap();
        validate_deepseek_v4_generation_mode(&messages, ExplicitCliOptions::default()).unwrap();

        let messages_with_qwen_policy = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "messages.json",
            "--messages-preserve-thinking",
            "--messages-no-generation-prompt",
        ])
        .unwrap();
        let error = validate_deepseek_v4_generation_mode(
            &messages_with_qwen_policy,
            ExplicitCliOptions::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("--messages-preserve-thinking"));
        assert!(error.contains("--messages-no-generation-prompt"));

        let messages_with_strip = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "messages.json",
            "--messages-strip-thinking",
        ])
        .unwrap();
        assert!(
            validate_deepseek_v4_generation_mode(
                &messages_with_strip,
                ExplicitCliOptions::default(),
            )
            .unwrap_err()
            .to_string()
            .contains("--messages-strip-thinking")
        );

        let jsonl = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "requests.jsonl",
        ])
        .unwrap();
        assert!(
            validate_deepseek_v4_generation_mode(&jsonl, ExplicitCliOptions::default())
                .unwrap_err()
                .to_string()
                .contains("--requests-jsonl")
        );

        let matches = Args::command()
            .try_get_matches_from([
                "qwen",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--prefill-chunk",
                "1024",
                "--prefix-cache-max-mib",
                "16384",
                "--cache-prefix-auto-min-tokens",
                "1024",
                "--durable-prefix-cache-max-mib",
                "32768",
                "--durable-prefix-cache-max-entry-mib",
                "16384",
                "--durable-prefix-cache-min-tokens",
                "1024",
            ])
            .unwrap();
        let explicit = ExplicitCliOptions::from_matches(&matches);
        let explicit_defaults = Args::from_arg_matches(&matches).unwrap();
        let error = validate_deepseek_v4_generation_mode(&explicit_defaults, explicit)
            .unwrap_err()
            .to_string();
        for option in [
            "--prefill-chunk",
            "--prefix-cache-max-mib",
            "--cache-prefix-auto-min-tokens",
            "--durable-prefix-cache-max-mib",
            "--durable-prefix-cache-max-entry-mib",
            "--durable-prefix-cache-min-tokens",
        ] {
            assert!(error.contains(option), "missing {option:?} from {error:?}");
        }
    }

    #[test]
    #[ignore = "requires the local DeepSeek V4 Flash-0731 IQ3 fixture"]
    fn deepseek_v4_0731_message_prompts_match_flash_vocab() {
        const MODEL: &str = concat!(
            "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/",
            "DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
        );
        fn message(role: &str, content: &str) -> messages::ChatMessage {
            messages::ChatMessage {
                role: role.into(),
                content: content.into(),
                extra: Default::default(),
            }
        }
        fn render_ids(tokenizer: &Tokenizer, messages: &[messages::ChatMessage]) -> Vec<i32> {
            let prompt = messages::render_deepseek_v4_0731_messages_prompt(messages).unwrap();
            tokenizer.encode(&prompt, false).unwrap()
        }

        assert!(Path::new(MODEL).exists(), "missing DS4 fixture");
        let tokenizer = Tokenizer::open(MODEL).expect("open DS4 tokenizer");
        assert_eq!(
            render_ids(&tokenizer, &[message("user", "Hello")]),
            [0, 128_803, 19_923, 128_804, 128_822]
        );
        assert_eq!(
            render_ids(
                &tokenizer,
                &[message("system", "Be exact."), message("user", "Hello")]
            ),
            [0, 7_153, 6_319, 16, 128_803, 19_923, 128_804, 128_822]
        );
        assert_eq!(
            render_ids(
                &tokenizer,
                &[
                    message("system", "Be exact."),
                    message("user", "Hello"),
                    message("assistant", "Hi!"),
                    message("user", "上海 🙂"),
                ]
            ),
            [
                0, 7_153, 6_319, 16, 128_803, 19_923, 128_804, 128_822, 23_166, 3, 1, 128_803,
                7_241, 68_139, 128_804, 128_822,
            ]
        );
    }

    #[test]
    fn prefill_chunk_argument_preserves_numeric_json_and_accepts_auto() {
        let fixed = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--prefill-chunk",
            "2048",
        ])
        .unwrap();
        let auto = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--prefill-chunk",
            "auto",
        ])
        .unwrap();

        assert_eq!(fixed.prefill_chunk, PrefillChunkArg::Fixed(2048));
        assert_eq!(auto.prefill_chunk, PrefillChunkArg::Auto);
        assert_eq!(serde_json::to_string(&fixed.prefill_chunk).unwrap(), "2048");
        assert_eq!(
            serde_json::to_string(&auto.prefill_chunk).unwrap(),
            "\"auto\""
        );
    }

    #[test]
    fn sampling_request_contract_supports_cli_defaults_and_jsonl_overrides() {
        let args = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--temp",
            "0.7",
            "--seed",
            "99",
        ])
        .unwrap();
        assert_eq!(
            cli_sampling_config(&args).unwrap(),
            SamplingConfig::qwen_chat(99)
        );

        let request: JsonlRequest = serde_json::from_str(
            r#"{"prompt":"hello","sampling":{"temp":1.0,"top_k":8,"top_p":0.9,"min_p":0.1,"seed":7}}"#,
        )
        .unwrap();
        assert_eq!(
            request_sampling_config(&request, &args).unwrap(),
            SamplingConfig {
                temperature: 1.0,
                top_k: 8,
                top_p: 0.9,
                min_p: 0.1,
                seed: 7,
            }
        );
        assert!(validate_sampling_decode_policy(SamplingConfig::qwen_chat(7), true).is_err());
        assert!(validate_sampling_decode_policy(SamplingConfig::default(), true).is_ok());

        let prompt_lookup_args = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "-",
            "--prompt-lookup",
            "--temp",
            "0.7",
        ])
        .unwrap();
        cli_sampling_config(&prompt_lookup_args).unwrap();
        let greedy_override: JsonlRequest =
            serde_json::from_str(r#"{"prompt":"hello","sampling":{"temperature":0.0}}"#).unwrap();
        let effective = request_sampling_config(&greedy_override, &prompt_lookup_args).unwrap();
        assert!(validate_sampling_decode_policy(effective, true).is_ok());

        let typo = serde_json::from_str::<JsonlRequest>(
            r#"{"prompt":"hello","sampling":{"temprature":0.7}}"#,
        );
        assert!(typo.is_err(), "sampling field typos must fail closed");
    }

    #[test]
    fn request_schema_versions_include_exact_generation_telemetry() {
        assert_eq!(
            request_schema_version(
                PrefillChunkArg::Fixed(1024),
                false,
                false,
                false,
                false,
                false,
                false,
            ),
            7
        );
        assert_eq!(
            request_schema_version(
                PrefillChunkArg::Fixed(1024),
                true,
                false,
                false,
                false,
                false,
                false,
            ),
            8
        );
        assert_eq!(
            request_schema_version(
                PrefillChunkArg::Fixed(2048),
                false,
                true,
                true,
                false,
                false,
                false,
            ),
            8
        );
        assert_eq!(
            request_schema_version(
                PrefillChunkArg::Auto,
                false,
                false,
                false,
                false,
                false,
                false,
            ),
            9
        );
        assert_eq!(
            request_schema_version(PrefillChunkArg::Auto, true, true, true, false, false, false,),
            9
        );
        assert_eq!(
            request_schema_version(
                PrefillChunkArg::Fixed(1024),
                false,
                false,
                false,
                true,
                false,
                false,
            ),
            10
        );
        assert_eq!(
            request_schema_version(
                PrefillChunkArg::Fixed(1024),
                false,
                false,
                false,
                true,
                true,
                false,
            ),
            11
        );
        assert_eq!(
            request_schema_version(
                PrefillChunkArg::Fixed(1024),
                false,
                false,
                false,
                true,
                false,
                true,
            ),
            12
        );
    }

    #[test]
    fn sampling_attribution_cli_is_narrow_and_fail_closed() {
        let exact = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt-file",
            "prompt.txt",
            "--tokens",
            "128",
            "--temp",
            "0.7",
            "--top-k",
            "200",
            "--top-p",
            "1.0",
            "--min-p",
            "0.05",
            "--seed",
            "42",
            "--prefill-chunk",
            "1024",
            "--max-context-tokens",
            "1024",
            "--prefix-cache-max-mib",
            "0",
            "--cache-prefix-auto-min-tokens",
            "0",
            "--request-timings",
            "timing.jsonl",
            "--sampling-attribution",
        ])
        .unwrap();
        assert!(validate_sampling_attribution_mode(&exact).is_ok());

        let wrong_tokens = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt-file",
            "prompt.txt",
            "--tokens",
            "127",
            "--temp",
            "0.7",
            "--max-context-tokens",
            "1024",
            "--prefix-cache-max-mib",
            "0",
            "--cache-prefix-auto-min-tokens",
            "0",
            "--request-timings",
            "timing.jsonl",
            "--sampling-attribution",
        ])
        .unwrap();
        assert!(
            validate_sampling_attribution_mode(&wrong_tokens)
                .unwrap_err()
                .to_string()
                .contains("--tokens 128")
        );

        assert!(
            Args::try_parse_from([
                "qwen",
                "--model",
                "model.gguf",
                "--prompt-file",
                "prompt.txt",
                "--request-timings",
                "timing.jsonl",
                "--request-timing-warm-followup",
                "--sampling-attribution",
            ])
            .is_err(),
            "warm follow-up must conflict at clap parsing"
        );
    }

    #[test]
    fn sampled_structural_cli_is_hidden_bounded_and_fail_closed() {
        let mut help = Vec::new();
        Args::command().write_long_help(&mut help).unwrap();
        assert!(
            !String::from_utf8(help)
                .unwrap()
                .contains("sampled-structural")
        );

        let exact = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt-file",
            "prompt.txt",
            "--temp",
            "0.7",
            "--top-k",
            "200",
            "--prefix-cache-max-mib",
            "0",
            "--cache-prefix-auto-min-tokens",
            "0",
            "--request-timings",
            "timing.jsonl",
            "--sampled-structural",
        ])
        .unwrap();
        assert!(validate_sampled_structural_mode(&exact).is_ok());

        let without_timings = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--temp",
            "0.7",
            "--top-k",
            "200",
            "--prefix-cache-max-mib",
            "0",
            "--cache-prefix-auto-min-tokens",
            "0",
            "--sampled-structural",
        ])
        .unwrap();
        assert!(validate_sampled_structural_mode(&without_timings).is_ok());

        let default_cache = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--temp",
            "0.7",
            "--top-k",
            "200",
            "--request-timings",
            "timing.jsonl",
            "--sampled-structural",
        ])
        .unwrap();
        assert!(
            validate_sampled_structural_mode(&default_cache)
                .unwrap_err()
                .to_string()
                .contains("zero RAM prefix-cache admission")
        );

        let unbounded = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--temp",
            "0.7",
            "--top-k",
            "0",
            "--prefix-cache-max-mib",
            "0",
            "--cache-prefix-auto-min-tokens",
            "0",
            "--request-timings",
            "timing.jsonl",
            "--sampled-structural",
        ])
        .unwrap();
        assert!(
            validate_sampled_structural_mode(&unbounded)
                .unwrap_err()
                .to_string()
                .contains("positive temperature and top-k")
        );

        assert!(
            Args::try_parse_from([
                "qwen",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--request-timings",
                "timing.jsonl",
                "--sampling-attribution",
                "--sampled-structural",
            ])
            .is_err(),
            "sampling attribution must conflict at clap parsing"
        );
    }

    #[test]
    fn sampled_structural_telemetry_has_the_frozen_key_set() {
        let value = serde_json::to_value(SampledStructuralTelemetry::default()).unwrap();
        let object = value.as_object().expect("structural telemetry object");
        let actual: std::collections::BTreeSet<_> = object.keys().map(String::as_str).collect();
        let expected = std::collections::BTreeSet::from([
            "version",
            "algorithm_version",
            "path",
            "prompt_owned_bounded_calls",
            "borrowed_transition_calls",
            "resident_head_wait_calls",
            "validated_shared_row_calls",
            "fallback_calls",
            "input_logits_total",
            "input_logits_min",
            "input_logits_max",
            "retained_top_k_total",
            "retained_top_k_min",
            "retained_top_k_max",
            "max_heap_len",
            "max_heap_capacity",
            "full_candidate_vector_allocations",
            "transition_logits_copy_bytes",
            "extra_command_buffers",
            "gpu_sampling_dispatches",
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn auto_prefill_jsonl_cache_gate_requires_no_entries_or_snapshot() {
        assert!(auto_prefill_cache_safe(0, None));
        assert!(!auto_prefill_cache_safe(1, None));
        assert!(!auto_prefill_cache_safe(0, Some(1)));
        assert!(!auto_prefill_cache_safe(1, Some(1)));

        for source in [
            CachePrefixSource::Request,
            CachePrefixSource::Cli,
            CachePrefixSource::Auto,
        ] {
            let (selected, selected_source) =
                selected_cache_prefix_from_value(4_000, 10_000, source);
            assert_eq!(selected, Some(4_000));
            assert_eq!(selected_source, source);
            assert!(!auto_prefill_cache_safe(0, selected));
        }
    }

    #[test]
    fn cache_promotion_uses_restored_not_logical_prefix_depth() {
        assert!(cache_prefix_needs_extension(3, 2));
        assert!(!cache_prefix_needs_extension(3, 3));
        assert!(!cache_prefix_needs_extension(2, 3));
    }

    #[test]
    fn one_shot_durable_admission_respects_threshold_and_explicit_override() {
        let automatic = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--durable-prefix-cache",
            "cache",
        ])
        .unwrap();
        assert_eq!(
            selected_single_turn_durable_policy(&automatic, 1023, false),
            DurableCapturePolicy::Disabled
        );
        assert_eq!(
            selected_single_turn_durable_policy(&automatic, 1024, false),
            DurableCapturePolicy::AutomaticPrompt(1024)
        );
        assert_eq!(
            selected_single_turn_durable_lookup_len(&automatic, 2048),
            2048
        );

        let explicit = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--durable-prefix-cache",
            "cache",
            "--cache-prefix-tokens",
            "64",
        ])
        .unwrap();
        assert_eq!(
            selected_single_turn_durable_policy(&explicit, 32, true),
            DurableCapturePolicy::ExplicitPrompt(32)
        );
        assert_eq!(selected_single_turn_durable_lookup_len(&explicit, 2048), 64);

        let completed = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "messages.json",
            "--messages-preserve-thinking",
            "--durable-prefix-cache",
            "cache",
        ])
        .unwrap();
        assert_eq!(
            selected_single_turn_durable_policy(&completed, 1024, true),
            DurableCapturePolicy::AutomaticCompleted
        );

        let disabled = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--durable-prefix-cache",
            "cache",
            "--cache-prefix-tokens",
            "0",
        ])
        .unwrap();
        assert_eq!(
            selected_single_turn_durable_policy(&disabled, 4096, true),
            DurableCapturePolicy::Disabled
        );
        assert_eq!(
            selected_single_turn_durable_lookup_len(&disabled, 4096),
            4096
        );
    }

    #[test]
    fn completed_checkpoint_boundary_tracks_consumed_and_pending_tokens() {
        for generated in [[7].as_slice(), [7, 8, 9].as_slice()] {
            let transitions = generated.len() - 1;
            let boundary =
                derive_completed_checkpoint_boundary(3, generated, transitions, 3 + transitions)
                    .unwrap();
            assert_eq!(boundary.consumed_generated_tokens, transitions);
            assert_eq!(boundary.consumed_prefix_len, 3 + transitions);
            assert_eq!(boundary.pending_token, generated[transitions]);
            assert_eq!(
                boundary.consumed_tokens(&[1, 2, 3], generated),
                [1, 2, 3]
                    .into_iter()
                    .chain(generated[..transitions].iter().copied())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn completed_checkpoint_boundary_rejects_invalid_generation_state() {
        assert!(derive_completed_checkpoint_boundary(3, &[], 0, 3).is_err());
        assert!(derive_completed_checkpoint_boundary(3, &[7, 8], 0, 3).is_err());
        assert!(derive_completed_checkpoint_boundary(3, &[7, 8], 1, 3).is_err());
    }

    #[test]
    fn prompt_loading_gates_completed_capture_on_canonical_history() {
        let root =
            std::env::temp_dir().join(format!("qwen-completed-policy-{}", std::process::id()));
        let messages = root.with_extension("json");
        let prompt_file = root.with_extension("txt");
        std::fs::write(
            &messages,
            r#"{
                "meta": {"preserve_thinking": true},
                "messages": [{"role": "user", "content": "hi"}]
            }"#,
        )
        .unwrap();
        std::fs::write(&prompt_file, "hi").unwrap();

        let messages_path = messages.to_str().unwrap();
        let prompt_path = prompt_file.to_str().unwrap();
        let cases = [
            (
                vec!["qwen", "--model", "model.gguf", "--messages", messages_path],
                true,
            ),
            (
                vec![
                    "qwen",
                    "--model",
                    "model.gguf",
                    "--messages",
                    messages_path,
                    "--messages-strip-thinking",
                ],
                false,
            ),
            (
                vec![
                    "qwen",
                    "--model",
                    "model.gguf",
                    "--messages",
                    messages_path,
                    "--messages-no-generation-prompt",
                ],
                false,
            ),
            (
                vec!["qwen", "--model", "model.gguf", "--prompt", "hi"],
                false,
            ),
            (
                vec![
                    "qwen",
                    "--model",
                    "model.gguf",
                    "--prompt-file",
                    prompt_path,
                ],
                false,
            ),
        ];
        for (argv, expected) in cases {
            let args = Args::try_parse_from(argv).unwrap();
            let (_, _, eligible) = prompt_text(&args).unwrap();
            assert_eq!(eligible, expected);
        }

        std::fs::remove_file(messages).unwrap();
        std::fs::remove_file(prompt_file).unwrap();
    }

    #[test]
    fn durable_mode_rejects_unsupported_surfaces_and_invalid_budgets() {
        let jsonl = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "requests.jsonl",
            "--durable-prefix-cache",
            "cache",
        ])
        .unwrap();
        assert!(validate_durable_prefix_cache_mode(&jsonl).is_err());

        let invalid_budget = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--durable-prefix-cache",
            "cache",
            "--durable-prefix-cache-max-mib",
            "1024",
            "--durable-prefix-cache-max-entry-mib",
            "2048",
        ])
        .unwrap();
        assert!(validate_durable_prefix_cache_mode(&invalid_budget).is_err());

        let valid = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--durable-prefix-cache",
            "cache",
        ])
        .unwrap();
        validate_durable_prefix_cache_mode(&valid).unwrap();
    }

    #[test]
    fn checkpoint_staged_integrity_parser_is_strict() {
        assert_eq!(parse_checkpoint_staged_integrity(None).unwrap(), None);
        assert_eq!(
            parse_checkpoint_staged_integrity(Some("decode")).unwrap(),
            Some(StagedIntegrityMode::Decode)
        );
        assert_eq!(
            parse_checkpoint_staged_integrity(Some("deferred-restore")).unwrap(),
            Some(StagedIntegrityMode::DeferredRestore)
        );
        for invalid in ["", "Decode", "deferred_restore", "deferred", "true"] {
            assert!(parse_checkpoint_staged_integrity(Some(invalid)).is_err());
        }
    }

    #[test]
    fn messages_input_is_single_turn_and_durable_cache_eligible() {
        let args = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "game.json",
            "--messages-strip-thinking",
            "--durable-prefix-cache",
            "cache",
        ])
        .unwrap();
        validate_durable_prefix_cache_mode(&args).unwrap();
        assert!(!prompt_add_special_tokens(&args, PromptSource::Messages));

        assert!(
            Args::try_parse_from([
                "qwen",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--messages",
                "game.json",
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from([
                "qwen",
                "--model",
                "model.gguf",
                "--messages",
                "game.json",
                "--messages-preserve-thinking",
                "--messages-strip-thinking",
            ])
            .is_err()
        );
    }

    #[test]
    fn auto_prefill_profiles_require_complete_allowlisted_topology() {
        let a3b = test_a3b_arch();
        let a10b = test_a10b_arch();

        assert_eq!(
            auto_prefill_profile(a3b, Some("Qwen3.6 35B A3B"), Some(15)),
            Some(AutoPrefillProfile {
                name: "qwen3.6-35b-a3b-filetype15",
                outer_chunk: 2048,
                query_heads: 16,
                gdn_overlay_bytes: 235_405_312,
            })
        );
        assert_eq!(
            auto_prefill_profile(a10b, Some("Qwen3.5 122B A10B"), Some(15)),
            Some(AutoPrefillProfile {
                name: "qwen3.5-122b-a10b-filetype15",
                outer_chunk: 4096,
                query_heads: 32,
                gdn_overlay_bytes: 807_403_520,
            })
        );

        macro_rules! reject_field {
            ($arch:expr, $name:expr, $field:ident, $value:expr) => {{
                let mut invalid = $arch;
                invalid.$field = $value;
                assert_eq!(
                    auto_prefill_profile(invalid, Some($name), Some(15)),
                    None,
                    "{}",
                    stringify!($field)
                );
            }};
        }

        for (arch, name) in [(a3b, "Qwen3.6 35B A3B"), (a10b, "Qwen3.5 122B A10B")] {
            reject_field!(arch, name, kind, ArchKind::Dense);
            reject_field!(arch, name, expert_count, 255);
            reject_field!(arch, name, expert_used_count, 7);
            reject_field!(arch, name, full_attention_interval, 3);
            reject_field!(arch, name, attn_head_dim, 128);
            reject_field!(arch, name, partial_rotary_factor, 0.5);
            reject_field!(arch, name, gdn_n_k_heads, 8);
            reject_field!(arch, name, gdn_head_dim, 64);
            reject_field!(arch, name, gdn_conv_kernel, 3);
            reject_field!(arch, name, mtp_n_hidden_layers, 1);
            assert_eq!(auto_prefill_profile(arch, Some(name), Some(14)), None);
            assert_eq!(
                auto_prefill_profile(arch, Some("wrong model"), Some(15)),
                None
            );
            assert_eq!(auto_prefill_profile(arch, None, Some(15)), None);
        }

        reject_field!(a3b, "Qwen3.6 35B A3B", n_layer, 39);
        reject_field!(a3b, "Qwen3.6 35B A3B", hidden_size, 2049);
        reject_field!(a3b, "Qwen3.6 35B A3B", n_q_heads, 15);
        reject_field!(a3b, "Qwen3.6 35B A3B", n_kv_heads, 1);
        reject_field!(a3b, "Qwen3.6 35B A3B", gdn_n_v_heads, 31);
        reject_field!(a3b, "Qwen3.6 35B A3B", expert_feed_forward_length, 513);
        reject_field!(
            a3b,
            "Qwen3.6 35B A3B",
            expert_shared_feed_forward_length,
            513
        );
        reject_field!(a10b, "Qwen3.5 122B A10B", n_layer, 47);
        reject_field!(a10b, "Qwen3.5 122B A10B", hidden_size, 3071);
        reject_field!(a10b, "Qwen3.5 122B A10B", n_q_heads, 31);
        reject_field!(a10b, "Qwen3.5 122B A10B", n_kv_heads, 1);
        reject_field!(a10b, "Qwen3.5 122B A10B", gdn_n_v_heads, 63);
        reject_field!(a10b, "Qwen3.5 122B A10B", expert_feed_forward_length, 1023);
        reject_field!(
            a10b,
            "Qwen3.5 122B A10B",
            expert_shared_feed_forward_length,
            1023
        );
    }

    #[test]
    fn auto_prefill_chunk_decision_bounds_the_measured_prompt_range() {
        let profile = Some(AutoPrefillProfile {
            name: "test-profile",
            outer_chunk: 2048,
            query_heads: 16,
            gdn_overlay_bytes: 235_405_312,
        });
        for (prompt_tokens, selected, reason) in [
            (8191, 1024, "prompt_below_validated_range"),
            (8192, 2048, "matched_validated_profile"),
            (16384, 2048, "matched_validated_profile"),
            (16385, 1024, "prompt_above_memory_bounded_range"),
        ] {
            let decision = auto_prefill_chunk_decision(profile, prompt_tokens, false, true);
            assert_eq!(decision.selected, selected);
            assert_eq!(decision.reason, reason);
            assert_eq!(decision.validated_prompt_range, Some([8192, 16384]));
            assert_eq!(decision.evidence_baseline_chunk, Some(1024));
        }

        let unsupported = auto_prefill_chunk_decision(None, 10_000, false, true);
        assert_eq!(unsupported.selected, 1024);
        assert_eq!(unsupported.validated_prompt_range, None);
        assert_eq!(unsupported.evidence_baseline_chunk, None);

        let environment = auto_prefill_chunk_decision(profile, 10_000, true, true);
        assert_eq!(environment.reason, "prefill_environment_override_present");
        assert_eq!(environment.selected, 1024);

        let cache = auto_prefill_chunk_decision(profile, 10_000, false, false);
        assert_eq!(cache.reason, "prefix_cache_interaction_unvalidated");
        assert_eq!(cache.selected, 1024);
    }

    #[test]
    fn auto_prefill_overlay_equations_cover_range_boundaries() {
        let a3b = AutoPrefillProfile {
            name: "a3b",
            outer_chunk: 2048,
            query_heads: 16,
            gdn_overlay_bytes: 235_405_312,
        };
        let a10b = AutoPrefillProfile {
            name: "a10b",
            outer_chunk: 4096,
            query_heads: 32,
            gdn_overlay_bytes: 807_403_520,
        };
        assert_eq!(
            expected_auto_prefill_overlay(a3b, 8192).unwrap(),
            PrefillScratchOverlayStats {
                backing_bytes: 285_212_672,
                attention_bytes: 285_212_672,
                gdn_bytes: 235_405_312,
                saved_bytes: 235_405_312,
            }
        );
        assert_eq!(
            expected_auto_prefill_overlay(a3b, 16_384).unwrap(),
            PrefillScratchOverlayStats {
                backing_bytes: 570_425_344,
                attention_bytes: 570_425_344,
                gdn_bytes: 235_405_312,
                saved_bytes: 235_405_312,
            }
        );
        assert_eq!(
            expected_auto_prefill_overlay(a3b, 11_287).unwrap(),
            PrefillScratchOverlayStats {
                backing_bytes: 393_052_160,
                attention_bytes: 393_052_160,
                gdn_bytes: 235_405_312,
                saved_bytes: 235_405_312,
            }
        );
        assert_eq!(
            expected_auto_prefill_overlay(a10b, 11_287).unwrap(),
            PrefillScratchOverlayStats {
                backing_bytes: 807_403_520,
                attention_bytes: 786_104_320,
                gdn_bytes: 807_403_520,
                saved_bytes: 786_104_320,
            }
        );
    }

    #[test]
    fn auto_prefill_plan_topology_rejects_every_geometry_drift() {
        let profile = test_auto_profile();
        let prompt_tokens = 10_000;
        let overlay = expected_auto_prefill_overlay(profile, prompt_tokens).unwrap();
        assert_eq!(
            validate_auto_prefill_plan_topology(
                profile,
                prompt_tokens,
                2048,
                10_000,
                1024,
                Some(overlay),
            )
            .unwrap(),
            overlay
        );

        for (block_size, matrix_max_pos, query_rows) in [
            (2047, 10_000, 1024),
            (2048, 9_999, 1024),
            (2048, 10_000, 512),
        ] {
            let error = validate_auto_prefill_plan_topology(
                profile,
                prompt_tokens,
                block_size,
                matrix_max_pos,
                query_rows,
                Some(overlay),
            )
            .unwrap_err();
            assert!(error.to_string().contains("plan geometry drifted"));
        }

        let missing =
            validate_auto_prefill_plan_topology(profile, prompt_tokens, 2048, 10_000, 1024, None)
                .unwrap_err();
        assert!(missing.to_string().contains("has no scratch overlay"));

        let mut wrong_overlay = overlay;
        wrong_overlay.saved_bytes -= 1;
        let mismatch = validate_auto_prefill_plan_topology(
            profile,
            prompt_tokens,
            2048,
            10_000,
            1024,
            Some(wrong_overlay),
        )
        .unwrap_err();
        assert!(mismatch.to_string().contains("overlay geometry drifted"));
    }

    #[test]
    fn auto_prefill_allocation_pricing_reconciles_and_rejects_invalid_prices() {
        let allocations = [
            ("zero", false, 0),
            ("eager", false, 10),
            ("deferred", true, 20),
        ];
        let (rows, eager, deferred, priced) = price_prefill_allocations(allocations, |logical| {
            Ok(MetalBufferSizeAndAlign {
                size: match logical {
                    0 => 1,
                    10 => 16,
                    20 => 32,
                    _ => unreachable!(),
                },
                alignment: 256,
            })
        })
        .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!((eager, deferred, priced), (10, 20, 49));
        assert_eq!(rows[2].name, "deferred");
        assert!(rows[2].deferred);

        for priced in [
            MetalBufferSizeAndAlign {
                size: 0,
                alignment: 256,
            },
            MetalBufferSizeAndAlign {
                size: 9,
                alignment: 256,
            },
            MetalBufferSizeAndAlign {
                size: 10,
                alignment: 0,
            },
            MetalBufferSizeAndAlign {
                size: 10,
                alignment: 3,
            },
        ] {
            let error =
                price_prefill_allocations([("bad", false, 10)], |_| Ok(priced)).unwrap_err();
            assert!(error.to_string().contains("allocation pricing is invalid"));
        }

        let priced_overflow =
            price_prefill_allocations([("first", false, 1), ("second", true, 2)], |logical| {
                Ok(MetalBufferSizeAndAlign {
                    size: if logical == 1 { u64::MAX } else { 2 },
                    alignment: 256,
                })
            })
            .unwrap_err();
        assert!(
            priced_overflow
                .to_string()
                .contains("priced prefill byte overflow")
        );

        let logical_overflow = price_prefill_allocations(
            [("first", false, u64::MAX), ("second", false, 1)],
            |logical| {
                Ok(MetalBufferSizeAndAlign {
                    size: logical,
                    alignment: 256,
                })
            },
        )
        .unwrap_err();
        assert!(
            logical_overflow
                .to_string()
                .contains("eager prefill logical byte overflow")
        );
    }

    #[test]
    fn auto_prefill_environment_gate_is_prefix_scoped() {
        assert!(is_prefill_environment_key(std::ffi::OsStr::new(
            "QWEN_PREFILL_ATTN_MATRIX_ONLINE"
        )));
        assert!(!is_prefill_environment_key(std::ffi::OsStr::new(
            "QWEN_GGUF_NO_COPY"
        )));

        let keys = vec![
            std::ffi::OsString::from("PATH"),
            std::ffi::OsString::from("QWEN_PREFILL_ATTN_MATRIX_ONLINE"),
            std::ffi::OsString::from("QWEN_GGUF_NO_COPY"),
        ];
        let unchanged = keys.clone();
        assert!(prefill_environment_override_present_in(&keys));
        assert_eq!(keys, unchanged);
        assert!(!prefill_environment_override_present_in([
            std::ffi::OsString::from("PATH"),
            std::ffi::OsString::from("QWEN_GGUF_NO_COPY"),
        ]));
    }

    #[test]
    fn auto_prefill_reserve_requires_positive_checked_sequence_growth() {
        assert_eq!(
            auto_prefill_reserve(100, 101),
            Ok((1, AUTO_CHUNK_TRANSIENT_RESERVE_BYTES + 1))
        );
        assert_eq!(
            auto_prefill_reserve(100, 100),
            Err("sequence_allocation_signal_invalid")
        );
        assert_eq!(
            auto_prefill_reserve(101, 100),
            Err("sequence_allocation_signal_invalid")
        );
        assert_eq!(
            auto_prefill_reserve(0, u64::MAX),
            Err("candidate_reserve_overflow")
        );
    }

    #[test]
    fn ineligible_auto_prefill_preserves_legacy_constructor_order_and_reasons() {
        let profile = Some(test_auto_profile());
        for (candidate, prompt, environment, cache_safe, reason) in [
            (None, 10_000, false, true, "profile_not_allowlisted"),
            (
                profile,
                AUTO_CHUNK_PROMPT_MIN - 1,
                false,
                true,
                "prompt_below_validated_range",
            ),
            (
                profile,
                AUTO_CHUNK_PROMPT_MAX + 1,
                false,
                true,
                "prompt_above_memory_bounded_range",
            ),
            (
                profile,
                10_000,
                true,
                true,
                "prefill_environment_override_present",
            ),
            (
                profile,
                10_000,
                false,
                false,
                "prefix_cache_interaction_unvalidated",
            ),
        ] {
            let mut allocator = fake_allocator(&[10, 20], Ok(fake_plan_decision(1_000)), 0, false);
            let state = allocate_prefill_request_state_with(
                &mut allocator,
                PrefillChunkArg::Auto,
                prompt,
                10_017,
                cache_safe,
                candidate,
                environment,
            )
            .unwrap();
            let decision = state.decision.as_ref().unwrap();

            assert_eq!(state.chunk, 1024);
            assert_eq!(state.scratch, "legacy");
            assert_eq!(decision.reason, reason);
            assert_eq!(decision.classification, "baseline");
            assert!(decision.plan.is_none());
            assert!(decision.admission.is_none());
            assert_eq!(
                allocator.events,
                [
                    AllocationEvent::LegacyScratch(1024),
                    AllocationEvent::CurrentAllocated,
                    AllocationEvent::Sequence(10_017),
                    AllocationEvent::CurrentAllocated,
                ],
                "{reason}"
            );
        }
    }

    #[test]
    fn candidate_plan_unavailable_has_a_distinct_fail_closed_reason() {
        let mut allocator = fake_allocator(
            &[100, 200, 300],
            Err(CandidatePlanFailure::Unavailable(
                "planner failed".to_string(),
            )),
            10_000_000_000,
            false,
        );
        let state = allocate_prefill_request_state_with(
            &mut allocator,
            PrefillChunkArg::Auto,
            10_000,
            10_017,
            true,
            Some(test_auto_profile()),
            false,
        )
        .unwrap();
        let decision = state.decision.as_ref().unwrap();

        assert_eq!(decision.reason, "candidate_plan_unavailable");
        assert_eq!(decision.detail.as_deref(), Some("planner failed"));
        assert!(decision.plan.is_none());
        assert!(decision.admission.is_none());
        assert!(
            !allocator
                .events
                .contains(&AllocationEvent::CandidateScratch)
        );
    }

    #[test]
    fn sequence_accounting_failures_fall_back_before_candidate_planning() {
        for (samples, reason) in [
            ([100, 100, 300], "sequence_allocation_signal_invalid"),
            ([101, 100, 300], "sequence_allocation_signal_invalid"),
            ([0, u64::MAX, 300], "candidate_reserve_overflow"),
        ] {
            let mut allocator =
                fake_allocator(&samples, Ok(fake_plan_decision(1_000)), u64::MAX, false);
            let state = allocate_prefill_request_state_with(
                &mut allocator,
                PrefillChunkArg::Auto,
                10_000,
                10_017,
                true,
                Some(test_auto_profile()),
                false,
            )
            .unwrap();
            let decision = state.decision.as_ref().unwrap();

            assert_eq!(decision.reason, reason);
            assert_eq!(state.scratch, "legacy");
            assert!(!allocator.events.contains(&AllocationEvent::CandidatePlan));
            assert!(
                !allocator
                    .events
                    .contains(&AllocationEvent::CandidateScratch)
            );
        }
    }

    #[test]
    fn auto_prefill_memory_boundaries_reach_the_cli_telemetry() {
        let reserve = AUTO_CHUNK_TRANSIENT_RESERVE_BYTES + 100;
        let required = reserve + 1_000;
        for (signals, expected_reason, admitted) in [
            (
                MetalMemorySignals {
                    recommended_max_bytes: 200 + required,
                    current_allocated_bytes: 200,
                    process_limit_remaining_bytes: Some(required),
                },
                "admitted_with_process_budget",
                true,
            ),
            (
                MetalMemorySignals {
                    recommended_max_bytes: 200 + required,
                    current_allocated_bytes: 200,
                    process_limit_remaining_bytes: Some(required - 1),
                },
                "process_insufficient",
                false,
            ),
            (
                MetalMemorySignals {
                    recommended_max_bytes: 200 + required,
                    current_allocated_bytes: 200,
                    process_limit_remaining_bytes: None,
                },
                "process_signal_unavailable",
                false,
            ),
            (
                MetalMemorySignals {
                    recommended_max_bytes: 0,
                    current_allocated_bytes: 200,
                    process_limit_remaining_bytes: Some(required),
                },
                "invalid_working_set_signal",
                false,
            ),
            (
                MetalMemorySignals {
                    recommended_max_bytes: 200 + required - 1,
                    current_allocated_bytes: 200,
                    process_limit_remaining_bytes: Some(0),
                },
                "working_set_insufficient",
                false,
            ),
            (
                MetalMemorySignals {
                    recommended_max_bytes: 200 + required,
                    current_allocated_bytes: 200,
                    process_limit_remaining_bytes: Some(0),
                },
                "admitted_process_budget_omitted",
                true,
            ),
        ] {
            let mut allocator = fake_allocator_with_signals(
                &[100, 200, 300],
                Ok(fake_plan_decision(1_000)),
                signals,
                false,
            );
            let state = allocate_prefill_request_state_with(
                &mut allocator,
                PrefillChunkArg::Auto,
                10_000,
                10_017,
                true,
                Some(test_auto_profile()),
                false,
            )
            .unwrap();
            let decision = state.decision.as_ref().unwrap();
            let admission = decision.admission.as_ref().unwrap();

            assert_eq!(admission.required_bytes, Some(required));
            assert_eq!(admission.evaluator_reason, expected_reason);
            assert_eq!(admission.admitted, admitted);
            assert_eq!(state.scratch, if admitted { "candidate" } else { "legacy" });
            assert_eq!(
                allocator
                    .events
                    .contains(&AllocationEvent::CandidateScratch),
                admitted
            );
        }
    }

    #[test]
    fn numeric_prefill_preserves_scratch_before_sequence_order() {
        let mut allocator = fake_allocator(&[10, 20], Ok(fake_plan_decision(1_000)), 0, false);
        let state = allocate_prefill_request_state_with(
            &mut allocator,
            PrefillChunkArg::Fixed(2048),
            10_000,
            10_017,
            true,
            Some(test_auto_profile()),
            false,
        )
        .unwrap();

        assert_eq!(state.chunk, 2048);
        assert_eq!(state.scratch, "legacy");
        assert!(state.decision.is_none());
        assert_eq!(
            allocator.events,
            [
                AllocationEvent::LegacyScratch(2048),
                AllocationEvent::CurrentAllocated,
                AllocationEvent::Sequence(10_017),
                AllocationEvent::CurrentAllocated,
            ]
        );
    }

    #[test]
    fn admitted_auto_prefill_allocates_only_the_candidate_scratch() {
        let mut allocator = fake_allocator(
            &[100, 200, 300],
            Ok(fake_plan_decision(1_000)),
            10_000_000_000,
            false,
        );
        let state = allocate_prefill_request_state_with(
            &mut allocator,
            PrefillChunkArg::Auto,
            10_000,
            10_017,
            true,
            Some(test_auto_profile()),
            false,
        )
        .unwrap();
        let decision = state.decision.as_ref().unwrap();

        assert_eq!(state.chunk, 2048);
        assert_eq!(state.scratch, "candidate");
        assert_eq!(decision.reason, "admitted");
        assert!(decision.plan.is_some());
        assert!(decision.admission.as_ref().unwrap().admitted);
        assert_eq!(
            allocator.events,
            [
                AllocationEvent::CurrentAllocated,
                AllocationEvent::Sequence(10_017),
                AllocationEvent::CurrentAllocated,
                AllocationEvent::CandidatePlan,
                AllocationEvent::MemorySignals,
                AllocationEvent::CandidateScratch,
                AllocationEvent::CurrentAllocated,
            ]
        );
    }

    #[test]
    fn denied_auto_prefill_reuses_sequence_and_allocates_legacy_scratch() {
        let mut allocator =
            fake_allocator(&[100, 200, 300], Ok(fake_plan_decision(1_000)), 200, false);
        let state = allocate_prefill_request_state_with(
            &mut allocator,
            PrefillChunkArg::Auto,
            10_000,
            10_017,
            true,
            Some(test_auto_profile()),
            false,
        )
        .unwrap();
        let decision = state.decision.as_ref().unwrap();

        assert_eq!(state.chunk, 1024);
        assert_eq!(state.scratch, "legacy");
        assert_eq!(decision.reason, "memory_admission_denied");
        assert!(decision.plan.is_some());
        assert!(!decision.admission.as_ref().unwrap().admitted);
        assert_eq!(
            allocator.events,
            [
                AllocationEvent::CurrentAllocated,
                AllocationEvent::Sequence(10_017),
                AllocationEvent::CurrentAllocated,
                AllocationEvent::CandidatePlan,
                AllocationEvent::MemorySignals,
                AllocationEvent::LegacyScratch(1024),
                AllocationEvent::CurrentAllocated,
            ]
        );
    }

    #[test]
    fn unpriceable_auto_prefill_plan_falls_back_before_candidate_allocation() {
        let mut allocator = fake_allocator(
            &[100, 200, 300],
            Err(CandidatePlanFailure::InvalidOrUnpriceable(
                "bad price".to_string(),
            )),
            10_000_000_000,
            false,
        );
        let state = allocate_prefill_request_state_with(
            &mut allocator,
            PrefillChunkArg::Auto,
            10_000,
            10_017,
            true,
            Some(test_auto_profile()),
            false,
        )
        .unwrap();
        let decision = state.decision.as_ref().unwrap();

        assert_eq!(state.scratch, "legacy");
        assert_eq!(decision.reason, "candidate_plan_invalid_or_unpriceable");
        assert_eq!(decision.detail.as_deref(), Some("bad price"));
        assert_eq!(
            allocator.events,
            [
                AllocationEvent::CurrentAllocated,
                AllocationEvent::Sequence(10_017),
                AllocationEvent::CurrentAllocated,
                AllocationEvent::CandidatePlan,
                AllocationEvent::LegacyScratch(1024),
                AllocationEvent::CurrentAllocated,
            ]
        );
    }

    #[test]
    fn required_byte_overflow_preserves_plan_telemetry() {
        let mut allocator = fake_allocator(
            &[100, 200, 300],
            Ok(fake_plan_decision(u64::MAX)),
            u64::MAX,
            false,
        );
        let state = allocate_prefill_request_state_with(
            &mut allocator,
            PrefillChunkArg::Auto,
            10_000,
            10_017,
            true,
            Some(test_auto_profile()),
            false,
        )
        .unwrap();
        let decision = state.decision.as_ref().unwrap();
        let json = serde_json::to_value(decision).unwrap();

        assert_eq!(decision.reason, "candidate_required_bytes_overflow");
        assert!(decision.plan.is_some());
        let admission = decision.admission.as_ref().unwrap();
        assert_eq!(admission.required_bytes, None);
        assert_eq!(admission.evaluator_reason, "required_bytes_overflow");
        assert!(!admission.admitted);
        assert!(json.get("plan").is_some());
        assert!(json.get("admission").is_some());
        assert_eq!(
            allocator.events,
            [
                AllocationEvent::CurrentAllocated,
                AllocationEvent::Sequence(10_017),
                AllocationEvent::CurrentAllocated,
                AllocationEvent::CandidatePlan,
                AllocationEvent::MemorySignals,
                AllocationEvent::LegacyScratch(1024),
                AllocationEvent::CurrentAllocated,
            ]
        );
    }

    #[test]
    fn admitted_candidate_allocation_failure_is_not_retried_as_legacy() {
        let mut allocator = fake_allocator(
            &[100, 200],
            Ok(fake_plan_decision(1_000)),
            10_000_000_000,
            true,
        );
        let result = allocate_prefill_request_state_with(
            &mut allocator,
            PrefillChunkArg::Auto,
            10_000,
            10_017,
            true,
            Some(test_auto_profile()),
            false,
        );

        let error = match result {
            Ok(_) => panic!("candidate allocation unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("candidate allocation failed"));
        assert_eq!(
            allocator.events,
            [
                AllocationEvent::CurrentAllocated,
                AllocationEvent::Sequence(10_017),
                AllocationEvent::CurrentAllocated,
                AllocationEvent::CandidatePlan,
                AllocationEvent::MemorySignals,
                AllocationEvent::CandidateScratch,
            ]
        );
    }

    #[test]
    fn auto_cache_prefix_discovery_scores_reuse() {
        let mut requests = vec![
            prepared("a", &[1, 2, 3, 4, 10]),
            prepared("b", &[1, 2, 3, 4, 20]),
            prepared("c", &[1, 2, 3, 30]),
            prepared("d", &[9, 9, 9]),
        ];

        discover_auto_cache_prefixes(&mut requests, 3);

        assert_eq!(requests[0].auto_cache_prefix_tokens, Some(3));
        assert_eq!(requests[0].auto_cache_future_hits, 2);
        assert_eq!(requests[1].auto_cache_prefix_tokens, Some(3));
        assert_eq!(requests[1].auto_cache_future_hits, 1);
        assert_eq!(requests[2].auto_cache_prefix_tokens, None);
        assert_eq!(requests[3].auto_cache_prefix_tokens, None);
    }

    #[test]
    fn greedy_generation_delivers_before_transition_and_skips_terminal_step() {
        let events = RefCell::new(Vec::new());
        let generation = generate_greedy(
            logits_with_argmax(1),
            3,
            &[],
            |token| {
                events.borrow_mut().push(format!("token:{token}"));
                Ok(())
            },
            |token| {
                events.borrow_mut().push(format!("transition:{token}"));
                Ok(logits_with_argmax(match token {
                    1 => 2,
                    2 => 0,
                    _ => panic!("unexpected transition token {token}"),
                }))
            },
        )
        .unwrap();

        assert_eq!(generation.tokens, [1, 2, 0]);
        assert_eq!(generation.transitions, 2);
        assert_eq!(generation.stop_reason, StopReason::TokenLimit);
        assert_eq!(
            events.into_inner(),
            [
                "token:1",
                "transition:1",
                "token:2",
                "transition:2",
                "token:0",
            ]
        );
    }

    #[test]
    fn gpu_greedy_generation_shares_terminal_and_callback_ordering() {
        let events = RefCell::new(Vec::new());
        let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
        let generation = generate_gpu_greedy(
            logits_with_argmax(1),
            3,
            &[],
            &mut sampler,
            |token| {
                events.borrow_mut().push(format!("token:{token}"));
                Ok(())
            },
            |token| {
                events.borrow_mut().push(format!("transition:{token}"));
                Ok(GreedySelection::Token(match token {
                    1 => 2,
                    2 => 0,
                    _ => panic!("unexpected transition token {token}"),
                }))
            },
        )
        .unwrap();

        assert_eq!(generation.tokens, [1, 2, 0]);
        assert_eq!(generation.transitions, 2);
        assert_eq!(generation.stop_reason, StopReason::TokenLimit);
        assert_eq!(sampler.draws(), 0);
        assert_eq!(
            events.into_inner(),
            [
                "token:1",
                "transition:1",
                "token:2",
                "transition:2",
                "token:0",
            ]
        );
    }

    #[test]
    fn gpu_greedy_nan_is_reported_after_the_committed_transition() {
        let events = RefCell::new(Vec::new());
        let position = Cell::new(10usize);
        let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
        let error = generate_gpu_greedy(
            logits_with_argmax(1),
            3,
            &[],
            &mut sampler,
            |token| {
                events.borrow_mut().push(format!("token:{token}"));
                Ok(())
            },
            |token| {
                events.borrow_mut().push(format!("transition:{token}"));
                position.set(position.get() + 1);
                Ok(GreedySelection::NanLogit { token: 7 })
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("logit at token 7 is NaN"));
        assert_eq!(events.into_inner(), ["token:1", "transition:1"]);
        assert_eq!(position.get(), 11, "the completed forward advances state");
        assert_eq!(sampler.draws(), 0);
    }

    #[test]
    fn gpu_greedy_preserves_terminal_and_failure_boundaries() {
        for (max_tokens, stop_tokens, expected_reason) in [
            (1, Vec::new(), StopReason::TokenLimit),
            (3, vec![1], StopReason::Eos),
        ] {
            let callbacks = Cell::new(0usize);
            let transitions = Cell::new(0usize);
            let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
            let generation = generate_gpu_greedy(
                logits_with_argmax(1),
                max_tokens,
                &stop_tokens,
                &mut sampler,
                |_| {
                    callbacks.set(callbacks.get() + 1);
                    Ok(())
                },
                |_| {
                    transitions.set(transitions.get() + 1);
                    Ok(GreedySelection::Token(2))
                },
            )
            .unwrap();
            assert_eq!(generation.tokens, [1]);
            assert_eq!(generation.stop_reason, expected_reason);
            assert_eq!(generation.transitions, 0);
            assert_eq!(transitions.get(), 0);
            assert_eq!(
                callbacks.get(),
                usize::from(expected_reason == StopReason::TokenLimit)
            );
        }

        let transitions = Cell::new(0usize);
        let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
        let callback_error = generate_gpu_greedy(
            logits_with_argmax(1),
            2,
            &[],
            &mut sampler,
            |_| Err(anyhow!("callback failed")),
            |_| {
                transitions.set(transitions.get() + 1);
                Ok(GreedySelection::Token(2))
            },
        )
        .unwrap_err();
        assert!(callback_error.to_string().contains("callback failed"));
        assert_eq!(transitions.get(), 0);

        let callbacks = Cell::new(0usize);
        let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
        let transition_error = generate_gpu_greedy(
            logits_with_argmax(1),
            2,
            &[],
            &mut sampler,
            |_| {
                callbacks.set(callbacks.get() + 1);
                Ok(())
            },
            |_| Err(anyhow!("transition failed")),
        )
        .unwrap_err();
        assert!(transition_error.to_string().contains("transition failed"));
        assert_eq!(callbacks.get(), 1);
    }

    #[test]
    fn decode_policy_labels_name_selection_residency() {
        let greedy = SamplingConfig::default();
        assert_eq!(decode_policy_label(greedy, false, false), "greedy_argmax");
        assert_eq!(
            decode_policy_label(greedy, false, true),
            "greedy_gpu_argmax"
        );
        assert_eq!(
            decode_policy_label(greedy, true, true),
            "prompt_lookup_l8_d7_target_n8"
        );
        assert_eq!(
            decode_policy_label(SamplingConfig::qwen_chat(7), false, false),
            "sampled_cpu"
        );
        assert_eq!(
            jsonl_decode_policy_label(greedy, false, false),
            "greedy_argmax"
        );
        assert_eq!(
            jsonl_decode_policy_label(greedy, false, true),
            "greedy_gpu_argmax"
        );
    }

    #[test]
    fn gpu_greedy_policy_is_default_off_forceable_and_rollbackable() {
        let modes = [
            GreedyGpuArgmaxMode::DefaultOff,
            GreedyGpuArgmaxMode::ForceEnabled,
            GreedyGpuArgmaxMode::ExplicitRollback,
        ];
        let requests = [
            (SamplingConfig::default(), false),
            (SamplingConfig::qwen_chat(7), false),
            (SamplingConfig::default(), true),
        ];
        for mode in modes {
            for (sampling, prompt_lookup) in requests {
                let request_eligible = sampling.temperature == 0.0 && !prompt_lookup;
                let expected = if mode == GreedyGpuArgmaxMode::ExplicitRollback {
                    GreedyGpuDecision {
                        enabled: false,
                        reason: "disabled_by_explicit_rollback",
                    }
                } else if !request_eligible {
                    GreedyGpuDecision {
                        enabled: false,
                        reason: "ineligible_request",
                    }
                } else if mode == GreedyGpuArgmaxMode::ForceEnabled {
                    GreedyGpuDecision {
                        enabled: true,
                        reason: "force_enabled",
                    }
                } else {
                    GreedyGpuDecision {
                        enabled: false,
                        reason: "default_off",
                    }
                };
                assert_eq!(
                    resolve_greedy_gpu_decision(mode, sampling, prompt_lookup),
                    expected,
                    "mode={mode:?} sampled={} prompt_lookup={prompt_lookup}",
                    sampling.temperature > 0.0,
                );
            }
        }

        for value in ["1", "true", "TRUE", "yes", "YES"] {
            assert_eq!(
                parse_greedy_gpu_argmax_mode(Some(OsStr::new(value))),
                GreedyGpuArgmaxMode::ForceEnabled
            );
        }
        for value in ["0", "false", "FALSE", "no", "NO"] {
            assert_eq!(
                parse_greedy_gpu_argmax_mode(Some(OsStr::new(value))),
                GreedyGpuArgmaxMode::ExplicitRollback
            );
        }
        assert_eq!(
            parse_greedy_gpu_argmax_mode(None),
            GreedyGpuArgmaxMode::DefaultOff
        );
        for value in ["", "invalid", "True", "2"] {
            assert_eq!(
                parse_greedy_gpu_argmax_mode(Some(OsStr::new(value))),
                GreedyGpuArgmaxMode::ExplicitRollback
            );
        }
        use std::os::unix::ffi::OsStringExt;
        let non_unicode = std::ffi::OsString::from_vec(vec![0xff]);
        assert_eq!(
            parse_greedy_gpu_argmax_mode(Some(&non_unicode)),
            GreedyGpuArgmaxMode::ExplicitRollback
        );
    }

    #[test]
    fn generated_token_sha256_has_a_canonical_integer_encoding() {
        assert_eq!(
            generated_token_sha256(&[1, -2, 248_319]),
            "2d6affd554663e51e2db10dd3cf00fe650f606317fbbf953353c86680fe48f0e"
        );
    }

    #[test]
    fn serial_generation_uses_the_seeded_request_sampler() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0x1234_5678_9abc_def0,
        };
        let mut sampler = Sampler::new(config).unwrap();
        let logits = vec![2.0, 1.5, 1.0, 0.5];
        let generation = generate_serial(
            logits.clone(),
            4,
            &[],
            &mut sampler,
            |_| Ok(()),
            |_| Ok(logits.clone()),
        )
        .unwrap();

        assert_eq!(generation.tokens, [0, 1, 1, 1]);
        assert_eq!(generation.transitions, 3);
        assert_eq!(sampler.draws(), 4);
        let telemetry = SamplingTelemetry::sampled(config, sampler.draws()).unwrap();
        assert_eq!(telemetry.algorithm_version, SAMPLER_ALGORITHM_VERSION);
        assert_eq!(telemetry.effective_seed, config.seed);
        assert_eq!(telemetry.draws, 4);
    }

    #[test]
    fn sampled_structural_generation_preserves_transaction_boundaries() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            seed: 42,
        };

        let mut ordinary_one_sampler = Sampler::new(config).unwrap();
        let ordinary_one = generate_serial(
            logits_with_argmax(1),
            1,
            &[],
            &mut ordinary_one_sampler,
            |_| Ok(()),
            |_| -> Result<Vec<f32>> { panic!("one-token output must not transition") },
        )
        .unwrap();
        let mut one_sampler = Sampler::new(config).unwrap();
        let (one, one_telemetry) = generate_sampled_structural(
            logits_with_argmax(1),
            1,
            &[],
            &mut one_sampler,
            |_| Ok(()),
            |_, _, _| -> Result<_> { panic!("one-token output must not transition") },
        )
        .unwrap();
        assert_eq!(one.tokens, [1]);
        assert_eq!(one.tokens, ordinary_one.tokens);
        assert_eq!(one.stop_reason, StopReason::TokenLimit);
        assert_eq!(one.stop_reason, ordinary_one.stop_reason);
        assert_eq!(one.transitions, 0);
        assert_eq!(one.transitions, ordinary_one.transitions);
        assert_eq!(one_sampler.draws(), 1);
        assert_eq!(one_sampler.draws(), ordinary_one_sampler.draws());
        assert_eq!(one_telemetry.prompt_owned_bounded_calls, 1);
        assert_eq!(one_telemetry.borrowed_transition_calls, 0);

        let ordinary_callbacks = RefCell::new(Vec::new());
        let mut ordinary_sampler = Sampler::new(config).unwrap();
        let ordinary = generate_serial(
            logits_with_argmax(1),
            3,
            &[],
            &mut ordinary_sampler,
            |token| {
                ordinary_callbacks.borrow_mut().push(token);
                Ok(())
            },
            |token| {
                Ok(logits_with_argmax(match token {
                    1 => 2,
                    2 => 0,
                    _ => panic!("unexpected ordinary token {token}"),
                }))
            },
        )
        .unwrap();
        let structural_callbacks = RefCell::new(Vec::new());
        let structural_advances = Cell::new(0usize);
        let mut structural_sampler = Sampler::new(config).unwrap();
        let (structural, structural_telemetry) = generate_sampled_structural(
            logits_with_argmax(1),
            3,
            &[],
            &mut structural_sampler,
            |token| {
                structural_callbacks.borrow_mut().push(token);
                Ok(())
            },
            |token, trial, telemetry| {
                let logits = logits_with_argmax(match token {
                    1 => 2,
                    2 => 0,
                    _ => panic!("unexpected structural token {token}"),
                });
                fake_structural_transition(trial, telemetry, &logits, || {
                    structural_advances.set(structural_advances.get() + 1);
                    Ok(())
                })
            },
        )
        .unwrap();
        assert_eq!(structural.tokens, ordinary.tokens);
        assert_eq!(structural.stop_reason, ordinary.stop_reason);
        assert_eq!(structural.transitions, ordinary.transitions);
        assert_eq!(structural_sampler.draws(), ordinary_sampler.draws());
        assert_eq!(structural_advances.get(), ordinary.transitions);
        assert_eq!(
            structural_callbacks.into_inner(),
            ordinary_callbacks.into_inner()
        );
        assert_eq!(structural_telemetry.prompt_owned_bounded_calls, 1);
        assert_eq!(structural_telemetry.borrowed_transition_calls, 2);
        assert_eq!(structural_telemetry.resident_head_wait_calls, 2);
        assert_eq!(structural_telemetry.validated_shared_row_calls, 2);
        assert_eq!(structural_telemetry.input_logits_total, 12);
        assert_eq!(structural_telemetry.retained_top_k_total, 3);
        assert_eq!(structural_telemetry.max_heap_len, 1);
        assert!(structural_telemetry.max_heap_capacity >= 1);
        assert!(structural_telemetry.max_heap_capacity < 4);
        let ordinary_boundary = derive_completed_checkpoint_boundary(
            10,
            &ordinary.tokens,
            ordinary.transitions,
            10 + ordinary.transitions,
        )
        .unwrap();
        let structural_boundary = derive_completed_checkpoint_boundary(
            10,
            &structural.tokens,
            structural.transitions,
            10 + structural.transitions,
        )
        .unwrap();
        assert_eq!(
            structural_boundary.consumed_prefix_len,
            ordinary_boundary.consumed_prefix_len
        );
        assert_eq!(
            structural_boundary.pending_token,
            ordinary_boundary.pending_token
        );

        for (stop_tokens, expected_tokens, expected_callbacks, expected_transitions) in [
            (vec![1], vec![1], vec![], 0),
            (vec![2], vec![1, 2], vec![1], 1),
        ] {
            let ordinary_callbacks = RefCell::new(Vec::new());
            let mut ordinary_sampler = Sampler::new(config).unwrap();
            let ordinary = generate_serial(
                logits_with_argmax(1),
                4,
                &stop_tokens,
                &mut ordinary_sampler,
                |token| {
                    ordinary_callbacks.borrow_mut().push(token);
                    Ok(())
                },
                |_| Ok(logits_with_argmax(2)),
            )
            .unwrap();
            let callbacks = RefCell::new(Vec::new());
            let mut sampler = Sampler::new(config).unwrap();
            let (generation, telemetry) = generate_sampled_structural(
                logits_with_argmax(1),
                4,
                &stop_tokens,
                &mut sampler,
                |token| {
                    callbacks.borrow_mut().push(token);
                    Ok(())
                },
                |_, trial, telemetry| {
                    fake_structural_transition(trial, telemetry, &logits_with_argmax(2), || Ok(()))
                },
            )
            .unwrap();
            assert_eq!(generation.tokens, ordinary.tokens);
            assert_eq!(generation.tokens, expected_tokens);
            assert_eq!(generation.stop_reason, ordinary.stop_reason);
            assert_eq!(generation.stop_reason, StopReason::Eos);
            assert_eq!(generation.transitions, ordinary.transitions);
            assert_eq!(generation.transitions, expected_transitions);
            assert_eq!(sampler.draws(), ordinary_sampler.draws());
            assert_eq!(
                callbacks.borrow().as_slice(),
                ordinary_callbacks.borrow().as_slice()
            );
            assert_eq!(callbacks.into_inner(), expected_callbacks);
            assert_eq!(
                telemetry.borrowed_transition_calls,
                expected_transitions as u64
            );
        }

        let callback_transitions = Cell::new(0usize);
        let mut ordinary_callback_sampler = Sampler::new(config).unwrap();
        let ordinary_callback_error = generate_serial(
            logits_with_argmax(1),
            2,
            &[],
            &mut ordinary_callback_sampler,
            |_| bail!("callback failed"),
            |_| unreachable!("callback failure must prevent transition"),
        )
        .unwrap_err();
        let mut callback_sampler = Sampler::new(config).unwrap();
        let callback_error = generate_sampled_structural(
            logits_with_argmax(1),
            2,
            &[],
            &mut callback_sampler,
            |_| bail!("callback failed"),
            |_, _, _| {
                callback_transitions.set(callback_transitions.get() + 1);
                unreachable!("callback failure must prevent transition")
            },
        )
        .unwrap_err();
        assert_eq!(
            callback_error.to_string(),
            ordinary_callback_error.to_string()
        );
        assert!(callback_error.to_string().contains("callback failed"));
        assert_eq!(callback_transitions.get(), 0);
        assert_eq!(callback_sampler.draws(), 1);
        assert_eq!(callback_sampler.draws(), ordinary_callback_sampler.draws());

        let mut ordinary_transition_sampler = Sampler::new(config).unwrap();
        let ordinary_transition_error = generate_serial(
            logits_with_argmax(1),
            2,
            &[],
            &mut ordinary_transition_sampler,
            |_| Ok(()),
            |_| bail!("transition failed"),
        )
        .unwrap_err();
        let mut transition_sampler = Sampler::new(config).unwrap();
        let transition_error = generate_sampled_structural(
            logits_with_argmax(1),
            2,
            &[],
            &mut transition_sampler,
            |_| Ok(()),
            |_, _, _| bail!("transition failed"),
        )
        .unwrap_err();
        assert_eq!(
            transition_error.to_string(),
            ordinary_transition_error.to_string()
        );
        assert!(transition_error.to_string().contains("transition failed"));
        assert_eq!(transition_sampler.draws(), 1);
        assert_eq!(
            transition_sampler.draws(),
            ordinary_transition_sampler.draws()
        );

        let nan_logits = vec![0.0, f32::NAN, 2.0, 1.0];
        let ordinary_position = Cell::new(0usize);
        let mut ordinary_sampler = Sampler::new(config).unwrap();
        let ordinary_error = generate_serial(
            logits_with_argmax(1),
            2,
            &[],
            &mut ordinary_sampler,
            |_| Ok(()),
            |_| {
                ordinary_position.set(ordinary_position.get() + 1);
                Ok(nan_logits.clone())
            },
        )
        .unwrap_err();
        let structural_position = Cell::new(0usize);
        let mut structural_sampler = Sampler::new(config).unwrap();
        let structural_error = generate_sampled_structural(
            logits_with_argmax(1),
            2,
            &[],
            &mut structural_sampler,
            |_| Ok(()),
            |_, trial, telemetry| {
                fake_structural_transition(trial, telemetry, &nan_logits, || {
                    structural_position.set(structural_position.get() + 1);
                    Ok(())
                })
            },
        )
        .unwrap_err();
        assert_eq!(structural_error.to_string(), ordinary_error.to_string());
        assert_eq!(structural_position.get(), ordinary_position.get());
        assert_eq!(structural_position.get(), 1);
        assert_eq!(structural_sampler.draws(), ordinary_sampler.draws());
        assert_eq!(structural_sampler.draws(), 1);

        let advance_attempts = Cell::new(0usize);
        let mut advance_sampler = Sampler::new(config).unwrap();
        let advance_error = generate_sampled_structural(
            logits_with_argmax(1),
            2,
            &[],
            &mut advance_sampler,
            |_| Ok(()),
            |_, trial, telemetry| {
                fake_structural_transition(trial, telemetry, &logits_with_argmax(2), || {
                    advance_attempts.set(advance_attempts.get() + 1);
                    bail!("advance failed")
                })
            },
        )
        .unwrap_err();
        assert!(advance_error.to_string().contains("advance failed"));
        assert_eq!(advance_attempts.get(), 1);
        assert_eq!(
            advance_sampler.draws(),
            1,
            "failed advance must not commit the trial RNG draw"
        );

        let mut accounting_sampler = Sampler::new(config).unwrap();
        let mut context = SampledStructuralContext {
            sampler: &mut accounting_sampler,
            telemetry: SampledStructuralTelemetry {
                prompt_owned_bounded_calls: u64::MAX,
                ..SampledStructuralTelemetry::default()
            },
        };
        let accounting_error =
            with_transactional_sampled_structural_context(&mut context, |trial, telemetry| {
                let (_, evidence) = trial.sample_bounded_top_k(&logits_with_argmax(1))?;
                telemetry.record_prompt(evidence)
            })
            .unwrap_err();
        assert!(accounting_error.to_string().contains("count overflow"));
        assert_eq!(context.sampler.draws(), 0);
        assert_eq!(context.telemetry.prompt_owned_bounded_calls, u64::MAX);
    }

    fn fake_attributed_transition(
        logits: Vec<f32>,
    ) -> (Vec<f32>, TokenProfile, LogitsReadbackProfile) {
        let bytes = logits.len() * std::mem::size_of::<f32>();
        (
            logits,
            TokenProfile {
                cpu_encode_ms: 0.1,
                cpu_to_gpu_complete_ms: 0.7,
                gpu_kernel_ms: 0.6,
                total_ms: 1.0,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
            LogitsReadbackProfile {
                timer_spans: 2,
                bytes,
                allocation_zero_fill_ms: 0.05,
                copy_ms: 0.1,
            },
        )
    }

    #[test]
    fn attributed_generation_preserves_sampling_and_terminal_boundaries() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            seed: 42,
        };
        let first = vec![3.0, 2.0, 1.0, 0.0];
        let next = vec![0.0, 3.0, 1.0, 2.0];

        let mut ordinary_sampler = Sampler::new(config).unwrap();
        let ordinary_callbacks = RefCell::new(Vec::new());
        let ordinary = generate_serial(
            first.clone(),
            2,
            &[],
            &mut ordinary_sampler,
            |token| {
                ordinary_callbacks.borrow_mut().push(token);
                Ok(())
            },
            |_| Ok(next.clone()),
        )
        .unwrap();

        let mut profiled_sampler = Sampler::new(config).unwrap();
        let profiled_callbacks = RefCell::new(Vec::new());
        let (profiled, sampler_profile, transition_profile) = generate_serial_attributed(
            first,
            2,
            &[],
            &mut profiled_sampler,
            |token| {
                profiled_callbacks.borrow_mut().push(token);
                Ok(())
            },
            |_| Ok(fake_attributed_transition(next.clone())),
        )
        .unwrap();
        assert_eq!(profiled.tokens, ordinary.tokens);
        assert_eq!(profiled.stop_reason, ordinary.stop_reason);
        assert_eq!(profiled.transitions, ordinary.transitions);
        assert_eq!(
            profiled_callbacks.into_inner(),
            ordinary_callbacks.into_inner()
        );
        assert_eq!(profiled_sampler.draws(), ordinary_sampler.draws());
        assert_eq!(sampler_profile.calls, 2);
        assert_eq!(sampler_profile.timer_spans, 22);
        assert_eq!(transition_profile.calls, 1);
        assert_eq!(transition_profile.new_timer_spans, 2);

        let mut eos_sampler = Sampler::new(config).unwrap();
        let (eos, sampler_profile, transition_profile) = generate_serial_attributed(
            vec![3.0, 2.0],
            4,
            &[0],
            &mut eos_sampler,
            |_| -> Result<()> { panic!("EOS must not reach the callback") },
            |_| -> Result<_> { panic!("EOS must not be transitioned") },
        )
        .unwrap();
        assert_eq!(eos.tokens, [0]);
        assert_eq!(eos.stop_reason, StopReason::Eos);
        assert_eq!(eos.transitions, 0);
        assert_eq!(sampler_profile.calls, 1);
        assert_eq!(transition_profile.calls, 0);

        let mut middle_eos_sampler = Sampler::new(config).unwrap();
        let delivered = RefCell::new(Vec::new());
        let (middle_eos, _, transition_profile) = generate_serial_attributed(
            vec![3.0, 2.0],
            4,
            &[1],
            &mut middle_eos_sampler,
            |token| {
                delivered.borrow_mut().push(token);
                Ok(())
            },
            |_| Ok(fake_attributed_transition(vec![0.0, 3.0])),
        )
        .unwrap();
        assert_eq!(middle_eos.tokens, [0, 1]);
        assert_eq!(middle_eos.stop_reason, StopReason::Eos);
        assert_eq!(middle_eos.transitions, 1);
        assert_eq!(delivered.into_inner(), [0]);
        assert_eq!(transition_profile.calls, 1);
    }

    #[test]
    fn attributed_one_token_generation_has_no_transition() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            seed: 42,
        };
        let mut sampler = Sampler::new(config).unwrap();
        let delivered = RefCell::new(Vec::new());
        let (generation, sampler_profile, transition_profile) = generate_serial_attributed(
            vec![0.0, 3.0],
            1,
            &[],
            &mut sampler,
            |token| {
                delivered.borrow_mut().push(token);
                Ok(())
            },
            |_| -> Result<_> { panic!("one-token output must not transition") },
        )
        .unwrap();
        assert_eq!(generation.tokens, [1]);
        assert_eq!(generation.stop_reason, StopReason::TokenLimit);
        assert_eq!(generation.transitions, 0);
        assert_eq!(delivered.into_inner(), [1]);
        assert_eq!(sampler.draws(), 1);
        assert_eq!(sampler_profile.calls, 1);
        assert_eq!(transition_profile.calls, 0);
    }

    #[test]
    fn attributed_generation_preserves_callback_and_transition_failures() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            seed: 42,
        };
        let mut callback_sampler = Sampler::new(config).unwrap();
        let transitions = Cell::new(0usize);
        let error = generate_serial_attributed(
            vec![3.0, 2.0],
            2,
            &[],
            &mut callback_sampler,
            |_| bail!("callback failed"),
            |_| {
                transitions.set(transitions.get() + 1);
                Ok(fake_attributed_transition(vec![3.0, 2.0]))
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("callback failed"));
        assert_eq!(transitions.get(), 0);

        let mut transition_sampler = Sampler::new(config).unwrap();
        let callbacks = Cell::new(0usize);
        let error = generate_serial_attributed(
            vec![3.0, 2.0],
            2,
            &[],
            &mut transition_sampler,
            |_| {
                callbacks.set(callbacks.get() + 1);
                Ok(())
            },
            |_| bail!("transition failed"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("transition failed"));
        assert_eq!(callbacks.get(), 1);
    }

    #[test]
    fn sampling_clock_probe_and_prompt_digest_match_frozen_contract() {
        let probe = measure_sampling_clock_probe();
        assert_eq!(probe.batches, 7);
        assert_eq!(probe.iterations_per_batch, 100_000);
        assert_eq!(probe.new_timer_spans, 1_662);
        assert!(probe.pair_ns.iter().all(|value| *value >= 0.0));
        assert!(probe.upper_pair_ns >= probe.pair_ns.iter().copied().fold(0.0, f64::max));
        assert_eq!(
            token_ids_sha256_i32le(&[1, -2, 248_319]),
            "3f37364bc87f9ff835c64d4bdb3d993fe35e530097697da8cbacb5e6f92119d5"
        );
    }

    #[test]
    fn greedy_generation_does_not_transition_eos() {
        let delivered = RefCell::new(Vec::new());
        let generation = generate_greedy(
            logits_with_argmax(1),
            4,
            &[1],
            |token| {
                delivered.borrow_mut().push(token);
                Ok(())
            },
            |_| -> Result<Vec<f32>> { panic!("EOS must not be consumed") },
        )
        .unwrap();

        assert!(delivered.into_inner().is_empty());
        assert_eq!(generation.tokens, [1]);
        assert_eq!(generation.transitions, 0);
        assert_eq!(generation.transition_ms, 0.0);
        assert!(generation.first_transition_ms.is_none());
        assert_eq!(generation.stop_reason, StopReason::Eos);
    }

    #[test]
    fn greedy_generation_honors_every_producer_stop_token() {
        for terminal in [1_i32, 3] {
            let generation = generate_greedy(
                logits_with_argmax(terminal as usize),
                4,
                &[1, 3],
                |_| Ok(()),
                |_| -> Result<Vec<f32>> { panic!("stop token must not be consumed") },
            )
            .unwrap();

            assert_eq!(generation.tokens, [terminal]);
            assert_eq!(generation.transitions, 0);
            assert_eq!(generation.stop_reason, StopReason::Eos);
        }
    }

    #[test]
    fn greedy_generation_one_token_needs_no_transition() {
        let generation = generate_greedy(
            logits_with_argmax(2),
            1,
            &[],
            |_| Ok(()),
            |_| -> Result<Vec<f32>> { panic!("terminal token must not be consumed") },
        )
        .unwrap();

        assert_eq!(generation.tokens, [2]);
        assert_eq!(generation.transitions, 0);
        assert_eq!(generation.stop_reason, StopReason::TokenLimit);
    }

    #[test]
    fn greedy_generation_rejects_zero_token_limit() {
        let error = generate_greedy(
            logits_with_argmax(2),
            0,
            &[],
            |_| Ok(()),
            |_| Ok(logits_with_argmax(0)),
        )
        .unwrap_err();

        assert!(error.to_string().contains("max_tokens must be >= 1"));
    }

    #[test]
    fn greedy_generation_does_not_transition_middle_eos() {
        let delivered = RefCell::new(Vec::new());
        let generation = generate_greedy(
            logits_with_argmax(1),
            4,
            &[2],
            |token| {
                delivered.borrow_mut().push(token);
                Ok(())
            },
            |token| {
                assert_eq!(token, 1);
                Ok(logits_with_argmax(2))
            },
        )
        .unwrap();

        assert_eq!(delivered.into_inner(), [1]);
        assert_eq!(generation.tokens, [1, 2]);
        assert_eq!(generation.transitions, 1);
        assert_eq!(generation.stop_reason, StopReason::Eos);
    }

    #[test]
    fn greedy_generation_delivers_token_before_transition_error() {
        let events = RefCell::new(Vec::new());
        let error = generate_greedy(
            logits_with_argmax(1),
            3,
            &[],
            |token| {
                events.borrow_mut().push(format!("token:{token}"));
                Ok(())
            },
            |token| {
                events.borrow_mut().push(format!("transition:{token}"));
                bail!("transition failed")
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("transition failed"));
        assert_eq!(events.into_inner(), ["token:1", "transition:1"]);
    }

    #[test]
    fn metal_allocation_samples_keep_signed_deltas_and_sampled_max() {
        let samples = metal_allocation_samples(100, 110, 120, 90, 150, 140, 130, 80);

        assert_eq!(
            samples.process_model_ready.delta_from_request_start_bytes,
            -10
        );
        assert_eq!(samples.request_start.delta_from_request_start_bytes, 0);
        assert_eq!(samples.after_scratch.delta_from_model_ready_bytes, 20);
        assert_eq!(samples.after_scratch.delta_from_request_start_bytes, 10);
        assert_eq!(samples.after_sequence.delta_from_model_ready_bytes, -10);
        assert_eq!(samples.after_sequence.delta_from_request_start_bytes, -20);
        assert_eq!(samples.after_request_state_drop.current_bytes, 80);
        assert_eq!(samples.current_allocated_sampled_max_bytes, 150);
    }

    #[test]
    fn request_timing_mode_accepts_only_single_turn_file_sidecars() {
        let valid = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--request-timings",
            "timings.jsonl",
        ])
        .unwrap();
        validate_request_timing_mode(&valid).unwrap();

        let valid_pair = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--request-timings",
            "timings.jsonl",
            "--request-timing-warm-followup",
        ])
        .unwrap();
        validate_request_timing_mode(&valid_pair).unwrap();

        for argv in [
            vec!["qwen", "--info", "--request-timings", "timings.jsonl"],
            vec![
                "qwen",
                "--model",
                "model.gguf",
                "--requests-jsonl",
                "requests.jsonl",
                "--request-timings",
                "timings.jsonl",
            ],
            vec![
                "qwen",
                "--model",
                "model.gguf",
                "--request-timings",
                "timings.jsonl",
            ],
            vec![
                "qwen",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--request-timings",
                "-",
            ],
            vec![
                "qwen",
                "--prompt",
                "hello",
                "--request-timings",
                "timings.jsonl",
            ],
            vec![
                "qwen",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--request-timing-warm-followup",
            ],
        ] {
            let args = Args::try_parse_from(argv).unwrap();
            assert!(validate_request_timing_mode(&args).is_err());
        }
    }

    #[test]
    fn request_timing_invariants_require_ordered_milestones_and_n_minus_one() {
        validate_request_timing_invariants(10.0, 11.0, 20.0, 21.0, 4, 3).unwrap();
        validate_request_timing_invariants(10.0, 20.0, 20.0, 21.0, 1, 0).unwrap();
        assert!(validate_request_timing_invariants(12.0, 11.0, 20.0, 21.0, 4, 3).is_err());
        assert!(validate_request_timing_invariants(10.0, 11.0, 20.0, 21.0, 4, 4).is_err());
        assert!(validate_request_timing_invariants(10.0, f64::NAN, 20.0, 21.0, 4, 3).is_err());
    }

    #[test]
    fn pipeline_cache_phase_metrics_align_prefill_and_generation() {
        let snapshot = |misses, miss_wall_ns, compiler_wall_ns| MetalPipelineCacheMetrics {
            misses,
            miss_wall_ns,
            compiler_wall_ns,
        };
        let metrics = pipeline_cache_phase_metrics(
            snapshot(0, 0, 0),
            snapshot(1, 10, 8),
            snapshot(3, 30, 25),
            snapshot(4, 40, 33),
        );

        assert_eq!(metrics.prefill.misses, 2);
        assert_eq!(metrics.prefill.miss_wall_ns, 20);
        assert_eq!(metrics.prefill.compiler_wall_ns, 17);
        assert_eq!(metrics.generation.misses, 1);
        assert_eq!(metrics.generation.miss_wall_ns, 10);
        assert_eq!(metrics.generation.compiler_wall_ns, 8);
        assert_eq!(metrics.total.misses, 4);
        assert_eq!(metrics.total.miss_wall_ns, 40);
        assert_eq!(metrics.total.compiler_wall_ns, 33);
    }
}
