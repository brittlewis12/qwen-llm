//! `qwen` — interactive CLI for the qwen-llm engine.

mod messages;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use messages::{load_messages_prompt, messages_thinking_mode};
use qwen_llm::checkpoint_identity::IdentityCacheOutcome;
use qwen_llm::checkpoint_store::{DurableCheckpointStore, PublishOutcome};
use qwen_llm::metal::{
    MetalBufferSizeAndAlign, MetalContext, MetalMemoryAdmission, MetalMemorySignals,
    MetalPipelineCacheMetrics, evaluate_metal_memory_admission,
};
use qwen_llm::metal_dflash::{
    MetalDFlashLayerMajorScratch, MetalDFlashVerifyScratch, PrefillScratchConfig,
    PrefillScratchOverlayStats, PrefillScratchPlan, ensure_prompt_lookup_n8_supported,
    plan_prefill_scratch_with_matrix_max_pos_configured, prefill_tokens_with_multi_hidden,
};
use qwen_llm::metal_forward::{MetalForward, MfError, SnapshotValidationError};
use qwen_llm::model::{Arch, ArchKind};
use qwen_llm::prompt_lookup::{DRAFT_TOKENS, PromptLookupProposer, terminal_draft_window};
use qwen_llm::runtime::{
    LoadedModel, LoadedModelConfig, PreparedCheckpoint, Runtime, RuntimeError, Sequence,
    SequenceConfig,
};
use qwen_llm::sampling::{SAMPLER_ALGORITHM_VERSION, Sampler, SamplingConfig};
use qwen_llm::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Parser, Debug)]
#[command(name = "qwen", version, about = "qwen-llm inference CLI")]
struct Args {
    /// Path to a GGUF file (Qwen 3.5 / 3.6 family).
    #[arg(short = 'm', long)]
    model: Option<std::path::PathBuf>,

    /// Print device info and exit.
    #[arg(long)]
    info: bool,

    /// Raw prompt text for a single-turn greedy generation.
    #[arg(short = 'p', long, conflicts_with_all = ["prompt_file", "messages"])]
    prompt: Option<String>,

    /// Read raw prompt text from a file.
    #[arg(long, conflicts_with_all = ["prompt", "messages"])]
    prompt_file: Option<PathBuf>,

    /// Render a bare or wrapped JSON messages file with the Qwen chat template.
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
    request_start_unix_ms: u64,
    runtime_and_model_load_ms: f64,
    stdout_sink: &'static str,
    ttft_endpoint: &'static str,
    prompt_source: PromptSource,
    prompt_bytes: usize,
    prompt_tokens: usize,
    requested_tokens: usize,
    generated_tokens: usize,
    stop_reason: StopReason,
    decode_policy: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    sampling: Option<SamplingTelemetry>,
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

#[derive(Debug, Serialize)]
struct RequestOutput {
    id: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    generated_text: String,
}

#[derive(Debug, Serialize)]
struct RequestStatsRow {
    schema_version: u32,
    id: String,
    line: usize,
    model: String,
    arrival_ms: u64,
    finish_ms: u64,
    prompt_tokens: usize,
    prompt_hash: String,
    requested_tokens: usize,
    generated_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_policy: Option<&'static str>,
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
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    validate_request_timing_mode(&args)?;
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

    if args.prompt.is_some() || args.prompt_file.is_some() || args.messages.is_some() {
        return run_single_turn(model_path, &args);
    }

    if let Some(path) = args.requests_jsonl.as_ref() {
        return run_requests_jsonl(model_path, path, &args);
    }

    print_model_info(model_path)
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

fn prompt_text(args: &Args) -> Result<(String, PromptSource)> {
    if let Some(prompt) = args.prompt.as_ref() {
        return Ok((prompt.clone(), PromptSource::Inline));
    }
    if let Some(path) = args.prompt_file.as_ref() {
        return Ok((
            std::fs::read_to_string(path)
                .with_context(|| format!("read prompt file {}", path.display()))?,
            PromptSource::File,
        ));
    }
    if let Some(path) = args.messages.as_ref() {
        return Ok((
            load_messages_prompt(
                path,
                args.messages_max,
                messages_thinking_mode(
                    args.messages_preserve_thinking,
                    args.messages_strip_thinking,
                ),
                !args.messages_no_generation_prompt,
            )?,
            PromptSource::Messages,
        ));
    }
    bail!("single-turn generation requires --prompt, --prompt-file, or --messages")
}

fn prompt_add_special_tokens(args: &Args, source: PromptSource) -> bool {
    source != PromptSource::Messages && !args.no_special_tokens
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
) -> u32 {
    if sampled {
        6
    } else if prefill_chunk.is_auto() {
        5
    } else if prompt_lookup || has_query_topology || has_scratch_overlay {
        4
    } else {
        3
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

fn run_single_turn(model_path: &Path, args: &Args) -> Result<()> {
    args.prefill_chunk.validate()?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    let sampling = cli_sampling_config(args)?;
    validate_sampling_decode_policy(sampling, args.prompt_lookup)?;
    let durable_store = durable_checkpoint_store(args)?;
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
        .load_model_for_disposable_single_turn_with_config(
            model_path,
            LoadedModelConfig {
                prefix_cache_max_bytes: prefix_cache_max_bytes(args)?,
                ..LoadedModelConfig::default()
            },
        )
        .with_context(|| format!("load model {}", model_path.display()))?;
    if args.prompt_lookup {
        ensure_prompt_lookup_n8_supported(loaded.metal_model()).map_err(anyhow::Error::msg)?;
    }
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
    let (first_prompt, first_prompt_source) = prompt_text(args)?;
    let first_prompt_acquisition_ms = prompt_t0.elapsed().as_secs_f64() * 1e3;
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
    let first_prepared = PreparedRequest {
        request_start_unix_ms: first_request_start_unix_ms,
        request_t0: first_request_t0,
        request_start_allocated: first_request_start_allocated,
        pipeline_cache_start: first_pipeline_cache_start,
        prompt: first_prompt,
        prompt_source: first_prompt_source,
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
        runtime_and_model_load_ms,
        process_model_ready_allocated,
        pair_id.as_deref(),
        0,
        "first_post_model_load",
        first_prepared,
        stdout_sink,
        durable_store.as_ref(),
        durable_max_record_bytes,
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
        let (warm_prompt, warm_prompt_source) = prompt_text(args)?;
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
                && warm_prompt_ids == results[0].prompt_ids,
            "warm follow-up prompt bytes or token IDs differ from request 0"
        );
        let warm_prepared = PreparedRequest {
            request_start_unix_ms: warm_request_start_unix_ms,
            request_t0: warm_request_t0,
            request_start_allocated: warm_request_start_allocated,
            pipeline_cache_start: warm_pipeline_cache_start,
            prompt: warm_prompt,
            prompt_source: warm_prompt_source,
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
            runtime_and_model_load_ms,
            process_model_ready_allocated,
            pair_id.as_deref(),
            1,
            "warm_followup",
            warm_prepared,
            stdout_sink,
            durable_store.as_ref(),
            durable_max_record_bytes,
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
                "{}: prompt_tokens={} generated_tokens={} transitions={} ",
                "load_ms={:.1} prefill_ms={:.1} ttft_ms={:.1} ",
                "decode_tps={:.2} transition_tps={:.2} cache_entries={} ",
                "cache_mib={:.1}/{:.1}"
            ),
            stats_prefix,
            result.prompt_ids.len(),
            result.generated.len(),
            result.transitions,
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
    runtime_and_model_load_ms: f64,
    process_model_ready_allocated: Option<u64>,
    pair_id: Option<&str>,
    request_index: usize,
    request_epoch: &'static str,
    prepared: PreparedRequest,
    stdout_sink: &'static str,
    durable_store: Option<&DurableCheckpointStore>,
    durable_max_record_bytes: u64,
) -> Result<SingleTurnResult> {
    let PreparedRequest {
        request_start_unix_ms,
        request_t0,
        request_start_allocated,
        pipeline_cache_start,
        prompt,
        prompt_source,
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

    let pipeline_cache_prefill_entry =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let durable_prefix_len =
        durable_store.and_then(|_| selected_single_turn_durable_prefix(args, prompt_ids.len()));
    let mut durable_prepared: Option<PreparedCheckpoint> = None;
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
                Ok(prepared) => durable_prepared = Some(prepared),
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
    let (generation, prompt_lookup_stats) = if args.prompt_lookup {
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
        (result.generation, Some(result.stats))
    } else {
        let generation = generate_serial(
            logits,
            args.tokens,
            &stop_tokens,
            &mut sampler,
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
        )?;
        (generation, None)
    };
    let mut inference_complete_ms = request_t0.elapsed().as_secs_f64() * 1e3;
    let pipeline_cache_generation_exit =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let generated = generation.tokens;
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
    let total_request_ms = request_t0.elapsed().as_secs_f64() * 1e3;
    report_prefill_chunk_decision(prefill_chunk_decision.as_ref(), prompt_ids.len());
    let request_end_allocated = timing_enabled.then(|| loaded.context().current_allocated_size());
    drop(stdout);

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
            Ok(report) => eprintln!(
                concat!(
                    "durable_prefix_cache: publish={} prefix_tokens={} blob_bytes={} ",
                    "evicted={} identity={} capture_ms={:.1} publish_ms={:.1}"
                ),
                publish_outcome_label(report.store.outcome),
                prepared.matched_prefix_len(),
                report.store.blob_bytes,
                report.store.evicted_entries,
                identity_cache_outcome_label(report.compatibility.outcome),
                durable_capture_ms,
                publish_t0.elapsed().as_secs_f64() * 1e3,
            ),
            Err(error) => eprintln!(
                concat!(
                    "warning: durable prefix publication failed after response ",
                    "(restore_ms={:.1} capture_ms={:.1}): {}"
                ),
                durable_restore_ms, durable_capture_ms, error,
            ),
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
            request_start_unix_ms,
            runtime_and_model_load_ms,
            stdout_sink,
            ttft_endpoint: "stdout_flush_complete",
            prompt_source,
            prompt_bytes: prompt.len(),
            prompt_tokens: prompt_ids.len(),
            requested_tokens: args.tokens,
            generated_tokens: generated.len(),
            stop_reason: generation.stop_reason,
            decode_policy: if args.prompt_lookup {
                "prompt_lookup_l8_d7_target_n8"
            } else if sampling_config.temperature > 0.0 {
                "sampled_cpu"
            } else {
                "greedy_argmax"
            },
            sampling: SamplingTelemetry::sampled(sampling_config, sampler.draws()),
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
    }
    Ok(SingleTurnResult {
        row,
        prompt,
        prompt_source,
        prompt_ids,
        generated,
        transitions: generation.transitions,
        prefill_ms,
        ttft_ms,
        decode_tps,
        transition_tps,
        tokenizer_init_ms,
    })
}

fn run_requests_jsonl(model_path: &Path, requests_path: &Path, args: &Args) -> Result<()> {
    args.prefill_chunk.validate()?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    cli_sampling_config(args)?;

    let load_t0 = Instant::now();
    let runtime = Runtime::metal().context("init Metal runtime")?;
    let loaded = runtime
        .load_model_with_config(
            model_path,
            LoadedModelConfig {
                prefix_cache_max_bytes: prefix_cache_max_bytes(args)?,
                ..LoadedModelConfig::default()
            },
        )
        .with_context(|| format!("load model {}", model_path.display()))?;
    if args.prompt_lookup {
        ensure_prompt_lookup_n8_supported(loaded.metal_model()).map_err(anyhow::Error::msg)?;
    }
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
            let (output, stats) =
                run_jsonl_request(&loaded, &tokenizer, &prepared_request, args)
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
            let (output, stats) = run_jsonl_request(&loaded, &tokenizer, prepared_request, args)
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
        )?;
        (generation, generated_text, None)
    };
    let generated = generation.tokens;
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
        ),
        id: id.to_string(),
        line: prepared.line,
        model: loaded.path().display().to_string(),
        arrival_ms,
        finish_ms,
        prompt_tokens: prompt_ids.len(),
        prompt_hash,
        requested_tokens: n_generate,
        generated_tokens: generated.len(),
        decode_policy: if args.prompt_lookup {
            Some("prompt_lookup_l8_d7_target_n8")
        } else if sampling_config.temperature > 0.0 {
            Some("sampled_cpu")
        } else {
            None
        },
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
        generated_text,
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
    mut logits: Vec<f32>,
    max_tokens: usize,
    stop_tokens: &[i32],
    sampler: &mut Sampler,
    mut on_token: OnToken,
    mut transition: Transition,
) -> Result<GenerationResult>
where
    OnToken: FnMut(i32) -> Result<()>,
    Transition: FnMut(i32) -> Result<Vec<f32>>,
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
        let token = sampler.sample(&logits)?.token;
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
        logits = transition(token)?;
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
) -> Result<(GenerationResult, String)> {
    sequence.check_position(start_position)?;
    let mut generated_text = String::new();
    let generation = generate_serial(
        logits,
        max_tokens,
        stop_tokens,
        sampler,
        |token| {
            generated_text.push_str(&tokenizer.decode_piece(token));
            Ok(())
        },
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
    )?;
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

fn durable_checkpoint_store(args: &Args) -> Result<Option<DurableCheckpointStore>> {
    let Some(root) = args.durable_prefix_cache.as_ref() else {
        return Ok(None);
    };
    Ok(Some(DurableCheckpointStore::new(
        root,
        mib_to_bytes(
            args.durable_prefix_cache_max_mib,
            "durable prefix cache byte budget",
        )?,
    )))
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

fn selected_single_turn_durable_prefix(args: &Args, prompt_len: usize) -> Option<usize> {
    if let Some(configured) = args.cache_prefix_tokens {
        return (configured > 0 && prompt_len > 0).then_some(configured.min(prompt_len));
    }
    (args.durable_prefix_cache_min_tokens > 0 && prompt_len >= args.durable_prefix_cache_min_tokens)
        .then_some(prompt_len)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
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
    fn request_schema_versions_preserve_greedy_rows_and_reserve_sampling_v6() {
        assert_eq!(
            request_schema_version(PrefillChunkArg::Fixed(1024), false, false, false, false),
            3
        );
        assert_eq!(
            request_schema_version(PrefillChunkArg::Fixed(1024), true, false, false, false),
            4
        );
        assert_eq!(
            request_schema_version(PrefillChunkArg::Fixed(2048), false, true, true, false),
            4
        );
        assert_eq!(
            request_schema_version(PrefillChunkArg::Auto, false, false, false, false),
            5
        );
        assert_eq!(
            request_schema_version(PrefillChunkArg::Auto, true, true, true, false),
            5
        );
        assert_eq!(
            request_schema_version(PrefillChunkArg::Fixed(1024), false, false, false, true),
            6
        );
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
        assert_eq!(selected_single_turn_durable_prefix(&automatic, 1023), None);
        assert_eq!(
            selected_single_turn_durable_prefix(&automatic, 1024),
            Some(1024)
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
        assert_eq!(selected_single_turn_durable_prefix(&explicit, 32), Some(32));
        assert_eq!(selected_single_turn_durable_lookup_len(&explicit, 2048), 64);

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
        assert_eq!(selected_single_turn_durable_prefix(&disabled, 4096), None);
        assert_eq!(
            selected_single_turn_durable_lookup_len(&disabled, 4096),
            4096
        );
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
