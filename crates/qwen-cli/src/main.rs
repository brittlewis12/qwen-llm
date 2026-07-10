//! `qwen` — interactive CLI for the qwen-llm engine.

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use qwen_llm::metal::MetalPipelineCacheMetrics;
use qwen_llm::metal_dflash::{MetalDFlashLayerMajorScratch, prefill_tokens_with_multi_hidden};
use qwen_llm::metal_forward::MetalForward;
use qwen_llm::runtime::{LoadedModel, LoadedModelConfig, Runtime, Sequence, SequenceConfig};
use qwen_llm::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
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
    #[arg(short = 'p', long, conflicts_with = "prompt_file")]
    prompt: Option<String>,

    /// Read raw prompt text from a file.
    #[arg(long, conflicts_with = "prompt")]
    prompt_file: Option<PathBuf>,

    /// Read JSONL request objects from a file or '-' while keeping one model loaded.
    #[arg(long, conflicts_with_all = ["prompt", "prompt_file"])]
    requests_jsonl: Option<PathBuf>,

    /// Number of greedy tokens to generate.
    #[arg(short = 'n', long, default_value_t = 64)]
    tokens: usize,

    /// Prompt prefill chunk size. The safe default mirrors qwen-bench.
    #[arg(long, default_value_t = 1024)]
    prefill_chunk: usize,

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

#[derive(Debug, Deserialize)]
struct JsonlRequest {
    id: Option<String>,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    tokens: Option<usize>,
    cache_prefix_tokens: Option<usize>,
}

#[derive(Debug)]
struct PreparedJsonlRequest {
    request: JsonlRequest,
    id: String,
    line: usize,
    prompt_ids: Vec<i32>,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum StopReason {
    Eos,
    TokenLimit,
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
    terminal_token_target_transition_consumed: bool,
    no_special_tokens: bool,
    prefill_chunk_requested: usize,
    prefill_chunk_effective: usize,
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

    if args.info {
        let runtime = Runtime::metal()?;
        println!("device: {}", runtime.describe());
        return Ok(());
    }

    let Some(model_path) = args.model.as_ref() else {
        eprintln!("usage: qwen -m <path-to-gguf> -p <prompt>  (or `qwen --info`)");
        std::process::exit(2);
    };

    if args.prompt.is_some() || args.prompt_file.is_some() {
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
        args.prompt.is_some() || args.prompt_file.is_some(),
        "--request-timings requires --prompt or --prompt-file"
    );
    ensure!(
        path != Path::new("-"),
        "--request-timings requires a file path, not stdout"
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
    bail!("single-turn generation requires --prompt or --prompt-file")
}

fn run_single_turn(model_path: &Path, args: &Args) -> Result<()> {
    ensure!(args.prefill_chunk > 0, "--prefill-chunk must be >= 1");
    ensure!(args.tokens > 0, "--tokens must be >= 1");

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
        .load_model_with_config(
            model_path,
            LoadedModelConfig {
                prefix_cache_max_bytes: prefix_cache_max_bytes(args)?,
            },
        )
        .with_context(|| format!("load model {}", model_path.display()))?;
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
        .encode(&first_prompt, !args.no_special_tokens)
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
            .encode(&warm_prompt, !args.no_special_tokens)
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
            cache_stats.total_bytes as f64 / 1024.0 / 1024.0,
            cache_stats.max_bytes as f64 / 1024.0 / 1024.0,
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
    let chunk = args.prefill_chunk.min(prompt_ids.len().max(1));
    let block_size = u32::try_from(chunk).context("prefill chunk does not fit u32")?;
    let matrix_max_pos = prompt_ids.len().max(chunk);
    let capacity_validation_ms = validation_t0.elapsed().as_secs_f64() * 1e3;

    let scratch_t0 = Instant::now();
    let mut scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        loaded.context(),
        loaded.metal_model(),
        block_size,
        matrix_max_pos,
    )
    .context("allocate prefill scratch")?;
    let scratch_allocation_ms = scratch_t0.elapsed().as_secs_f64() * 1e3;
    let after_scratch_allocated = timing_enabled.then(|| loaded.context().current_allocated_size());

    let sequence_t0 = Instant::now();
    let mut sequence = loaded.create_sequence(SequenceConfig::new(capacity))?;
    let sequence_allocation_ms = sequence_t0.elapsed().as_secs_f64() * 1e3;
    let after_sequence_allocated =
        timing_enabled.then(|| loaded.context().current_allocated_size());
    let model_identity = timing_enabled.then(|| loaded.snapshot_identity(&sequence));
    let forward = loaded.forward();

    let pipeline_cache_prefill_entry =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let prefill_t0 = Instant::now();
    let logits = prefill_tokens_with_multi_hidden(
        &forward,
        &prompt_ids,
        0,
        sequence.metal_session_mut(),
        &mut scratch,
        &[],
        None,
    )
    .context("prefill prompt")?;
    sequence.advance_by(prompt_ids.len())?;
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;
    let pipeline_cache_prefill_exit =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let after_prefill_allocated = timing_enabled.then(|| loaded.context().current_allocated_size());

    let stdout_handle = std::io::stdout();
    let mut stdout = stdout_handle.lock();
    let generation_start_ms = request_t0.elapsed().as_secs_f64() * 1e3;
    let mut first_delivery_ms = None;
    let mut first_callback_duration_ms = None;
    let mut first_delivery_allocated = None;
    let generation = generate_greedy(
        logits,
        args.tokens,
        tokenizer.eos(),
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
                    sequence.metal_session_mut(),
                )
                .context("decode token")?;
            sequence.advance_by(1)?;
            Ok(next)
        },
    )?;
    let inference_complete_ms = request_t0.elapsed().as_secs_f64() * 1e3;
    let pipeline_cache_generation_exit =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let generated = generation.tokens;
    if !generated.is_empty() {
        writeln!(stdout)?;
        stdout.flush().context("flush final newline")?;
    }
    let total_request_ms = request_t0.elapsed().as_secs_f64() * 1e3;
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

    let row = timing_values.map(|samples| {
        let model_identity = model_identity.expect("timing identity");
        RequestTimingRow {
            schema_version: 3,
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
            decode_policy: "greedy_argmax",
            terminal_token_target_transition_consumed: false,
            no_special_tokens: args.no_special_tokens,
            prefill_chunk_requested: args.prefill_chunk,
            prefill_chunk_effective: chunk,
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
            metal_allocated: metal_allocation_samples(
                samples.0,
                samples.1,
                samples.2,
                samples.3,
                samples.4,
                samples.5,
                samples.6,
                after_state_drop_allocated.expect("timing sample"),
            ),
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
    ensure!(args.prefill_chunk > 0, "--prefill-chunk must be >= 1");
    ensure!(args.tokens > 0, "--tokens must be >= 1");

    let load_t0 = Instant::now();
    let runtime = Runtime::metal().context("init Metal runtime")?;
    let loaded = runtime
        .load_model_with_config(
            model_path,
            LoadedModelConfig {
                prefix_cache_max_bytes: prefix_cache_max_bytes(args)?,
            },
        )
        .with_context(|| format!("load model {}", model_path.display()))?;
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
        stats.total_bytes as f64 / 1024.0 / 1024.0,
        stats.max_bytes as f64 / 1024.0 / 1024.0,
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
    Ok(Some(PreparedJsonlRequest {
        request,
        id,
        line: line_no,
        prompt_ids,
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

    let chunk = args.prefill_chunk.min(prompt_ids.len().max(1));
    let block_size = u32::try_from(chunk).context("prefill chunk does not fit u32")?;
    let matrix_max_pos = prompt_ids.len().max(chunk);
    let mut scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        loaded.context(),
        loaded.metal_model(),
        block_size,
        matrix_max_pos,
    )
    .context("allocate prefill scratch")?;
    let mut sequence = loaded.create_sequence(SequenceConfig::new(capacity))?;
    let forward = loaded.forward();

    let (cache_prefix_tokens, cache_prefix_source) = selected_cache_prefix(
        request,
        args,
        prepared.auto_cache_prefix_tokens,
        prompt_ids.len(),
    );
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
            if matched_prefix_tokens == prompt_ids.len() {
                hit.exact_final_logits.with_context(|| {
                    format!("exact prefix-cache hit for request {id} did not store logits")
                })?
            } else if let Some(prefix_len) = cache_prefix_tokens
                && prefix_len > matched_prefix_tokens
            {
                let prefix_suffix = &prompt_ids[matched_prefix_tokens..prefix_len];
                let (prefix_logits, ms) = prefill_span(
                    &forward,
                    &mut sequence,
                    &mut scratch,
                    prefix_suffix,
                    matched_prefix_tokens,
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
                let suffix = &prompt_ids[matched_prefix_tokens..];
                let (logits, ms) = prefill_span(
                    &forward,
                    &mut sequence,
                    &mut scratch,
                    suffix,
                    matched_prefix_tokens,
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

    let (generation, generated_text) = decode_greedy(
        &forward,
        tokenizer,
        &mut sequence,
        logits,
        prompt_ids.len(),
        n_generate,
    )?;
    let generated = generation.tokens;
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
    let stats = RequestStatsRow {
        schema_version: 3,
        id: id.to_string(),
        line: prepared.line,
        model: loaded.path().display().to_string(),
        arrival_ms,
        finish_ms,
        prompt_tokens: prompt_ids.len(),
        prompt_hash,
        requested_tokens: n_generate,
        generated_tokens: generated.len(),
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
        prefill_chunk: args.prefill_chunk,
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
        total_ms: total_t0.elapsed().as_secs_f64() * 1e3,
        cache_entries: stats_now.entries,
        cache_bytes: stats_now.total_bytes,
        cache_max_bytes: stats_now.max_bytes,
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
        sequence.metal_session_mut(),
        scratch,
        &[],
        None,
    )
    .context("prefill prompt span")?;
    sequence.advance_by(token_ids.len())?;
    Ok((logits, t0.elapsed().as_secs_f64() * 1e3))
}

#[derive(Debug)]
struct GreedyGeneration {
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

fn generate_greedy<OnToken, Transition>(
    mut logits: Vec<f32>,
    max_tokens: usize,
    eos: Option<i32>,
    mut on_token: OnToken,
    mut transition: Transition,
) -> Result<GreedyGeneration>
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
        let token = argmax_i32(&logits);
        first_token_selection_ms.get_or_insert_with(|| selection_t0.elapsed().as_secs_f64() * 1e3);
        first_token_ready_ms.get_or_insert_with(|| wall_t0.elapsed().as_secs_f64() * 1e3);
        tokens.push(token);
        on_token(token)?;
        first_token_callback_ms.get_or_insert_with(|| wall_t0.elapsed().as_secs_f64() * 1e3);

        if Some(token) == eos {
            stop_reason = Some(StopReason::Eos);
            break;
        }
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

    Ok(GreedyGeneration {
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

fn decode_greedy(
    forward: &MetalForward<'_>,
    tokenizer: &Tokenizer,
    sequence: &mut Sequence,
    logits: Vec<f32>,
    start_position: usize,
    max_tokens: usize,
) -> Result<(GreedyGeneration, String)> {
    sequence.check_position(start_position)?;
    let mut generated_text = String::new();
    let generation = generate_greedy(
        logits,
        max_tokens,
        tokenizer.eos(),
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
                    sequence.metal_session_mut(),
                )
                .context("decode token")?;
            sequence.advance_by(1)?;
            Ok(next)
        },
    )?;
    Ok((generation, generated_text))
}

fn prefix_cache_max_bytes(args: &Args) -> Result<u64> {
    args.prefix_cache_max_mib
        .checked_mul(1024 * 1024)
        .context("prefix cache byte budget overflow")
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
            },
            id: id.to_string(),
            line: 1,
            prompt_ids: tokens.to_vec(),
            auto_cache_prefix_tokens: None,
            auto_cache_future_hits: 0,
        }
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
            None,
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
    fn greedy_generation_does_not_transition_eos() {
        let generation = generate_greedy(
            logits_with_argmax(1),
            4,
            Some(1),
            |_| Ok(()),
            |_| -> Result<Vec<f32>> { panic!("EOS must not be consumed") },
        )
        .unwrap();

        assert_eq!(generation.tokens, [1]);
        assert_eq!(generation.transitions, 0);
        assert_eq!(generation.transition_ms, 0.0);
        assert!(generation.first_transition_ms.is_none());
        assert_eq!(generation.stop_reason, StopReason::Eos);
    }

    #[test]
    fn greedy_generation_one_token_needs_no_transition() {
        let generation = generate_greedy(
            logits_with_argmax(2),
            1,
            None,
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
            None,
            |_| Ok(()),
            |_| Ok(logits_with_argmax(0)),
        )
        .unwrap_err();

        assert!(error.to_string().contains("max_tokens must be >= 1"));
    }

    #[test]
    fn greedy_generation_does_not_transition_middle_eos() {
        let generation = generate_greedy(
            logits_with_argmax(1),
            4,
            Some(2),
            |_| Ok(()),
            |token| {
                assert_eq!(token, 1);
                Ok(logits_with_argmax(2))
            },
        )
        .unwrap();

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
            None,
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
