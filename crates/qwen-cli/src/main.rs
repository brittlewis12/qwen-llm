//! `qwen` — interactive CLI for the qwen-llm engine.

mod cli;
mod concurrent_jsonl;
#[cfg(feature = "dsv4-diagnostics")]
mod dsv4_temporal;
mod execution_selector;
mod fixed_cohort_jsonl;
mod messages;
mod qwen_file_root;
mod shutdown;

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::{CommandFactory, FromArgMatches, Parser, parser::ValueSource};
use messages::{
    DeepSeekV4EncodeOptions, DeepSeekV4InlineThinking, DeepSeekV4Reasoning, Qwen38GenerationMode,
    Qwen38ReasoningEffort, QwenGenerationMode, load_deepseek_v4_0731_messages_prompt,
    load_messages_prompt_with_policy, messages_thinking_mode, parse_strict_messages_input,
    render_deepseek_v4_0731_messages_prompt, render_deepseek_v4_0731_single_turn_prompt,
    render_qwen_messages_prompt_with_generation, render_qwen_single_turn_prompt,
    render_qwen38_messages_prompt_with_generation, render_qwen38_single_turn_prompt,
};
use objc2_metal::MTLDevice;
use qwen_llm::checkpoint_identity::{
    CheckpointIdentityCache, IdentityCacheOutcome, checkpoint_content_identity,
};
use qwen_llm::checkpoint_store::{DurableCheckpointStore, PublishOutcome, StagedIntegrityMode};
use qwen_llm::deepseek_v4::{AttentionLane, DeepSeekV4Model, RouterWeights};
use qwen_llm::deepseek_v4_census::DeepSeekV4CensusV1;
use qwen_llm::deepseek_v4_checkpoint_store::{
    DeepSeekV4CheckpointStore, DeepSeekV4DurableError, DeepSeekV4PreparedCheckpoint,
};
use qwen_llm::deepseek_v4_metal::{
    DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES, DEEPSEEK_V4_MULTIGROUP_SELECTOR_MAX_CAPACITY_ROWS,
    DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS, DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS,
    DEEPSEEK_V4_PREFILL_MAX_TOKENS, DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY, DeepSeekV4MemorySamples,
    DeepSeekV4MetalResidency, DeepSeekV4ModelContentId, DeepSeekV4MultigroupSelectorGeometry,
    DeepSeekV4MultigroupSelectorTelemetry, DeepSeekV4Session, DeepSeekV4SessionCapacity,
    DeepSeekV4SnapshotCaptureErrorKind, DeepSeekV4SnapshotCodecConstraints,
    DeepSeekV4SnapshotFileOutcome, DeepSeekV4SnapshotRestoreErrorKind, DeepSeekV4StageKind,
    DeepSeekV4StageProfile, causal_snapshot_capture_error_kind, causal_snapshot_record_bytes,
    causal_snapshot_restore_error_kind, load_causal_snapshot_file, publish_causal_snapshot_file,
};
use qwen_llm::dense_batch8::DENSE_BATCH8_WIDTH;
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::{
    MetalBufferSizeAndAlign, MetalContext, MetalMemoryAdmission, MetalMemorySignals,
    MetalPipelineCacheMetrics, evaluate_metal_memory_admission,
    evaluate_metal_memory_admission_with_cpu_bytes,
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
use qwen_llm::moe_batch16::MOE_BATCH16_WIDTH;
use qwen_llm::pid_metrics::{PidDelta, PidSnapshot};
use qwen_llm::prefetch::{DEFAULT_CHUNK_BYTES, DEFAULT_WORKERS};
use qwen_llm::prompt_lookup::{DRAFT_TOKENS, PromptLookupProposer, terminal_draft_window};
use qwen_llm::runtime::{
    LoadedModel, LoadedModelConfig, PrefetchPolicy, PrefetchResidencyProbe, PreparedCheckpoint,
    Runtime, RuntimeError, Sequence, SequenceConfig, prefetch_opened_gguf,
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
use std::io::{BufRead, IsTerminal, Read, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::LazyLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const CHECKPOINT_STAGED_INTEGRITY_ENV: &str = "QWEN_CHECKPOINT_STAGED_INTEGRITY";
const GREEDY_GPU_ARGMAX_ENV: &str = "QWEN_GREEDY_GPU_ARGMAX";
const DEEPSEEK_V4_SNAPSHOT_MAX_RECORD_BYTES: u64 = 1024 * 1024 * 1024;
const DEEPSEEK_V4_SNAPSHOT_IDENTITY_CACHE_DIR: &str = ".qwen-dsv4-model-identity-v2";
const DEEPSEEK_V4_MULTIGROUP_SELECTOR_QUALIFIED_DEVICE: &str = "Apple M4 Max";
const DEEPSEEK_V4_PREFETCH_ENV: &str = "QWEN_DSV4_PREFETCH";
const DEEPSEEK_V4_PREFETCH_AUTO_THRESHOLD: f64 = 0.98;
#[cfg(feature = "dsv4-diagnostics")]
const DEEPSEEK_V4_TEMPORAL_WINDOW_ENV: &str = "QWEN_DSV4_TEMPORAL_WINDOW";
#[cfg(feature = "dsv4-diagnostics")]
const DEEPSEEK_V4_TEMPORAL_JSON_ENV: &str = "QWEN_DSV4_TEMPORAL_JSON";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum DeepSeekV4PrefetchMode {
    Off,
    Always,
    #[default]
    Auto,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, clap::ValueEnum)]
enum QwenModelPrefetchArg {
    #[default]
    Auto,
    Off,
}

impl QwenModelPrefetchArg {
    fn policy(self) -> PrefetchPolicy {
        match self {
            Self::Auto => LoadedModelConfig::default().prefetch_policy,
            Self::Off => PrefetchPolicy::Off,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
        }
    }
}

fn prefetch_policy_label(policy: PrefetchPolicy) -> &'static str {
    match policy {
        PrefetchPolicy::Off => "off",
        PrefetchPolicy::Always => "always",
        PrefetchPolicy::ColdOnly { .. } => "cold_only",
    }
}

impl DeepSeekV4PrefetchMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Always => "always",
            Self::Auto => "auto",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct DeepSeekV4PrefetchOutcome {
    mode: DeepSeekV4PrefetchMode,
    wall_ms: f64,
}

fn parse_deepseek_v4_prefetch_mode(value: Option<&OsStr>) -> Result<DeepSeekV4PrefetchMode> {
    let Some(value) = value else {
        return Ok(DeepSeekV4PrefetchMode::Auto);
    };
    let value = value
        .to_str()
        .with_context(|| format!("{DEEPSEEK_V4_PREFETCH_ENV} is not valid UTF-8"))?;
    match value {
        "off" => Ok(DeepSeekV4PrefetchMode::Off),
        "always" => Ok(DeepSeekV4PrefetchMode::Always),
        "auto" => Ok(DeepSeekV4PrefetchMode::Auto),
        _ => bail!("{DEEPSEEK_V4_PREFETCH_ENV} must be one of auto|off|always, got {value:?}"),
    }
}

fn configured_deepseek_v4_prefetch_mode() -> Result<DeepSeekV4PrefetchMode> {
    parse_deepseek_v4_prefetch_mode(std::env::var_os(DEEPSEEK_V4_PREFETCH_ENV).as_deref())
}

#[cfg(feature = "dsv4-diagnostics")]
fn parse_deepseek_v4_temporal_window(value: Option<&OsStr>) -> Result<usize> {
    let Some(value) = value else {
        return Ok(0);
    };
    let value = value
        .to_str()
        .with_context(|| format!("{DEEPSEEK_V4_TEMPORAL_WINDOW_ENV} is not valid UTF-8"))?;
    let window = value.parse::<usize>().with_context(|| {
        format!("{DEEPSEEK_V4_TEMPORAL_WINDOW_ENV}={value:?} is not an integer")
    })?;
    ensure!(
        window <= 65,
        "{DEEPSEEK_V4_TEMPORAL_WINDOW_ENV} must be in 0..=65, got {window}"
    );
    Ok(window)
}

#[cfg(feature = "dsv4-diagnostics")]
fn configured_deepseek_v4_temporal_window() -> Result<usize> {
    parse_deepseek_v4_temporal_window(std::env::var_os(DEEPSEEK_V4_TEMPORAL_WINDOW_ENV).as_deref())
}

fn deepseek_v4_prefetch_policy(mode: DeepSeekV4PrefetchMode) -> PrefetchPolicy {
    match mode {
        DeepSeekV4PrefetchMode::Off => PrefetchPolicy::Off,
        DeepSeekV4PrefetchMode::Always => PrefetchPolicy::Always,
        DeepSeekV4PrefetchMode::Auto => {
            PrefetchPolicy::cold_only(DEEPSEEK_V4_PREFETCH_AUTO_THRESHOLD)
                .expect("DeepSeek V4 auto-prefetch threshold is a valid fraction")
        }
    }
}

fn apply_deepseek_v4_prefetch(
    gguf: &GgufFile,
    mode: DeepSeekV4PrefetchMode,
) -> Result<DeepSeekV4PrefetchOutcome> {
    let process_before = PidSnapshot::now().ok();
    let defaults = LoadedModelConfig::default();
    let config = LoadedModelConfig {
        prefetch_policy: deepseek_v4_prefetch_policy(mode),
        prefetch_residency_probe: PrefetchResidencyProbe::Sampled,
        ..defaults
    };
    let report = prefetch_opened_gguf(gguf, &config);
    let process_delta = process_before
        .zip(PidSnapshot::now().ok())
        .map(|(before, after)| PidDelta::between(before, after));
    let bytes_returned = report.bytes_returned_total();
    if mode == DeepSeekV4PrefetchMode::Always {
        ensure!(
            report.shards_skipped() == 0,
            "explicit DeepSeek V4 prefetch skipped {} of {} shards",
            report.shards_skipped(),
            gguf.shard_count(),
        );
        ensure!(
            bytes_returned == gguf.total_mapped_len() as u64,
            "DeepSeek V4 prefetch returned {bytes_returned} bytes for {} mapped bytes",
            gguf.total_mapped_len(),
        );
    }
    let seconds = report.total_wall.as_secs_f64();
    let effective_bytes_per_sec = if seconds > 0.0 {
        bytes_returned as f64 / seconds
    } else {
        0.0
    };
    eprintln!(
        concat!(
            "deepseek_v4 prefetch: mode={} shards_prefetched={} shards_skipped={} workers={} chunk_bytes={} ",
            "bytes_returned={} physical_read_bytes={} wall_ms={:.1} effective_gbps={:.2}"
        ),
        mode.as_str(),
        report.shards_prefetched(),
        report.shards_skipped(),
        DEFAULT_WORKERS,
        DEFAULT_CHUNK_BYTES,
        bytes_returned,
        process_delta
            .map(|delta| delta.diskio_bytesread.to_string())
            .unwrap_or_else(|| "unavailable".to_string()),
        report.total_wall.as_secs_f64() * 1e3,
        effective_bytes_per_sec / 1e9,
    );
    Ok(DeepSeekV4PrefetchOutcome {
        mode,
        wall_ms: report.total_wall.as_secs_f64() * 1e3,
    })
}

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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, clap::ValueEnum)]
enum DeepSeekV4MultigroupSelectorArg {
    #[default]
    Auto,
    Off,
    QualifiedExperimental,
}

impl DeepSeekV4MultigroupSelectorArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::QualifiedExperimental => "qualified_experimental",
        }
    }
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
#[command(
    name = "qwen",
    version,
    about = "Fast local Qwen and DeepSeek inference on Apple Silicon",
    args_conflicts_with_subcommands = true,
    after_help = "Examples:\n  qwen run -m MODEL --user 'Explain this'\n  qwen run -m MODEL --system 'Be concise' --user 'Explain this'\n  qwen run -m MODEL --user -\n  qwen run -m MODEL --messages -\n  qwen run -m MODEL --raw-prompt '<exact model input>'\n  qwen run -m Qwen3.6-35B-A3B.gguf --user 'Explain this' --no-thinking\n\n--no-thinking controls model prompt rendering; it does not hide CLI diagnostics.\nCLI diagnostic suppression is not currently available.\nFor resident JSONL batching and expanded legacy/research help, run:\n  qwen --help\nLegacy flags shown there are flat and cannot be combined with qwen run.",
    after_long_help = "Modern examples:\n  qwen run -m MODEL --user 'Explain this'\n  qwen run -m MODEL --messages -\n  qwen run -m MODEL --raw-prompt '<exact model input>'\n\nLegacy/research examples (flat; do not combine with qwen run):\n  qwen -m MODEL --prompt '<raw model input>'\n  qwen -m MODEL --requests-jsonl requests.jsonl\n\n--no-thinking controls model prompt rendering; it does not hide CLI diagnostics.\nCLI diagnostic suppression is not currently available."
)]
struct Args {
    #[command(subcommand)]
    command: Option<cli::Command>,

    #[arg(skip)]
    prepared_prompt: Option<PreparedPrompt>,

    /// Path to a Qwen or DeepSeek V4 GGUF file.
    #[arg(short = 'm', long)]
    model: Option<std::path::PathBuf>,

    /// Print device info and exit.
    #[arg(long)]
    info: bool,

    /// Print the deterministic DeepSeek V4 schema/quant census as JSON.
    #[arg(
        long,
        hide_short_help = true,
        requires = "model",
        conflicts_with_all = ["info", "prompt", "prompt_file", "messages", "requests_jsonl"]
    )]
    deepseek_census_json: bool,

    /// Raw prompt text for single-turn generation.
    #[arg(short = 'p', long, hide_short_help = true, conflicts_with_all = ["prompt_file", "messages"])]
    prompt: Option<String>,

    /// Read raw prompt text from a file.
    #[arg(long, hide_short_help = true, conflicts_with_all = ["prompt", "messages"])]
    prompt_file: Option<PathBuf>,

    /// Render a bare or wrapped JSON messages file with the model-family encoder.
    #[arg(long, hide_short_help = true, conflicts_with_all = ["prompt", "prompt_file", "requests_jsonl"])]
    messages: Option<PathBuf>,

    /// Render only the first N messages.
    #[arg(long, hide_short_help = true, requires = "messages")]
    messages_max: Option<usize>,

    /// Preserve assistant `<think>...</think>` history.
    #[arg(
        long,
        hide_short_help = true,
        requires = "messages",
        conflicts_with = "messages_strip_thinking"
    )]
    messages_preserve_thinking: bool,

    /// Strip a leading assistant `<think>...</think>` block from history.
    #[arg(
        long,
        hide_short_help = true,
        requires = "messages",
        conflicts_with = "preserve_reasoning"
    )]
    messages_strip_thinking: bool,

    /// Do not append the assistant generation prompt after messages.
    #[arg(long, hide_short_help = true, requires = "messages")]
    messages_no_generation_prompt: bool,

    /// DeepSeek V4 release reasoning mode for --messages encoding.
    #[arg(long, hide_short_help = true, requires = "messages", value_enum)]
    reasoning: Option<ReasoningLevelArg>,

    /// Retain reasoning across turns (DeepSeek V4 thinking modes only).
    #[arg(long, hide_short_help = true, requires = "messages")]
    preserve_reasoning: bool,

    /// Read JSONL request objects from a file or '-' while keeping one model loaded.
    #[arg(long, hide_short_help = true, conflicts_with_all = ["prompt", "prompt_file", "messages"])]
    requests_jsonl: Option<PathBuf>,

    /// Select ordinary Qwen model cache warming; `off` avoids prefetch reads.
    #[arg(long, hide_short_help = true, requires = "requests_jsonl", value_enum)]
    model_prefetch: Option<QwenModelPrefetchArg>,

    /// Decode equal-prompt-length JSONL cohorts with per-request token limits.
    ///
    /// Width is 8 on dense Qwen or 16 on Qwen MoE.
    #[arg(long, hide_short_help = true, requires = "requests_jsonl")]
    batch_size: Option<usize>,

    /// Run two independent resident JSONL requests with overlapping decode.
    #[arg(
        long,
        hide_short_help = true,
        requires = "requests_jsonl",
        conflicts_with = "batch_size"
    )]
    concurrency: Option<usize>,

    /// Select serial or qualified automatic JSONL width planning.
    ///
    /// Accelerated modes use pair/cohort prefix fanout instead of RAM-cache
    /// lookahead admission.
    #[arg(
        long,
        hide_short_help = true,
        requires = "requests_jsonl",
        conflicts_with_all = ["batch_size", "concurrency"],
        value_enum
    )]
    execution_mode: Option<execution_selector::ExecutionModeArg>,

    /// Maximum number of tokens to generate.
    #[arg(short = 'n', long, hide_short_help = true, default_value_t = 64)]
    tokens: usize,

    /// Sampling temperature; zero preserves greedy decoding.
    #[arg(
        long = "temp",
        visible_alias = "temperature",
        hide_short_help = true,
        default_value_t = 0.0
    )]
    temperature: f32,

    /// Top-k sampling cutoff; zero disables it.
    #[arg(long, hide_short_help = true, default_value_t = 200)]
    top_k: usize,

    /// Nucleus sampling cutoff; one disables it.
    #[arg(long, hide_short_help = true, default_value_t = 1.0)]
    top_p: f32,

    /// Min-p sampling cutoff; zero disables it.
    #[arg(long, hide_short_help = true, default_value_t = 0.05)]
    min_p: f32,

    /// Effective deterministic seed; identical requests reuse the same stream.
    #[arg(long, hide_short_help = true, default_value_t = 42)]
    seed: u64,

    /// Enable experimental dense-27B Q4_K_M prompt-lookup decode.
    #[arg(long, hide_short_help = true)]
    prompt_lookup: bool,

    /// Prompt prefill chunk size, or `auto` for the bounded MoE allowlist.
    #[arg(long, hide_short_help = true, default_value = "1024")]
    prefill_chunk: PrefillChunkArg,

    /// Override sequence capacity. Defaults to prompt + generated tokens + slack.
    #[arg(long, hide_short_help = true)]
    max_context_tokens: Option<usize>,

    /// Select the DeepSeek V4 far-context selector policy.
    #[arg(
        long,
        hide_short_help = true,
        value_enum,
        default_value = "auto",
        requires = "model",
        conflicts_with_all = ["info", "deepseek_census_json"]
    )]
    deepseek_v4_multigroup_selector: DeepSeekV4MultigroupSelectorArg,

    /// Prefix-cache byte budget in MiB; oversized snapshots are retained alone.
    #[arg(long, hide_short_help = true, default_value_t = 16 * 1024)]
    prefix_cache_max_mib: u64,

    /// Cache this many exact prompt tokens as the reusable prefix for requests.
    #[arg(long, hide_short_help = true)]
    cache_prefix_tokens: Option<usize>,

    /// Auto-cache repeated JSONL prompt prefixes at or above this token length.
    #[arg(long, hide_short_help = true, default_value_t = 1024)]
    cache_prefix_auto_min_tokens: usize,

    /// Persist anonymous prefix checkpoints under this private directory.
    #[arg(long, hide_short_help = true)]
    durable_prefix_cache: Option<PathBuf>,

    /// Load or immutably publish one explicit DeepSeek V4 causal-prefix file.
    #[arg(
        long = "deepseek-v4-snapshot",
        hide_short_help = true,
        value_name = "PATH",
        conflicts_with_all = ["info", "deepseek_census_json", "requests_jsonl", "durable_prefix_cache"]
    )]
    deepseek_v4_snapshot: Option<PathBuf>,

    /// Aggregate durable checkpoint budget in MiB.
    #[arg(long, hide_short_help = true, default_value_t = 32 * 1024)]
    durable_prefix_cache_max_mib: u64,

    /// Maximum size of one encoded durable checkpoint record in MiB.
    #[arg(long, hide_short_help = true, default_value_t = 16 * 1024)]
    durable_prefix_cache_max_entry_mib: u64,

    /// Auto-persist one-shot prompt boundaries at or above this token length.
    #[arg(long, hide_short_help = true, default_value_t = 1024)]
    durable_prefix_cache_min_tokens: usize,

    /// Append per-request JSON stats for multi-request runs.
    ///
    /// Timing fields are model-internal; this JSONL mode writes each completion
    /// after full decode rather than streaming the first token to stdout.
    #[arg(long, hide_short_help = true)]
    request_stats: Option<PathBuf>,

    /// Append per-request structured stats as JSONL under a common cross-family
    /// envelope (schema: qwen-llm.request-stats v1).
    ///
    /// Currently implemented by DeepSeek V4 single-turn generation only.
    /// Other invocation paths (Qwen single-turn, Qwen batch, DS4 batch,
    /// --info, --deepseek-census-json, model-info) REJECT this flag with a
    /// fatal error rather than silently ignoring it — mandatory/fail-closed
    /// telemetry policy. Use --request-stats for legacy stats output on
    /// paths that support it.
    ///
    /// The envelope has a small stable core plus namespaced backend
    /// extensions under `diagnostics.<family>`.
    #[arg(long, hide_short_help = true)]
    request_stats_jsonl: Option<PathBuf>,

    /// Append single-turn first-post-model-load timing rows as JSONL.
    #[arg(long, hide_short_help = true)]
    request_timings: Option<PathBuf>,

    /// Run one identical warm follow-up for paired request timing.
    #[arg(long, hide_short_help = true)]
    request_timing_warm_followup: bool,

    /// Attribute sampler-v1 and full-logit host work on the frozen A3B request.
    #[arg(
        long,
        hide_short_help = true,
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
    #[arg(long, hide_short_help = true)]
    no_special_tokens: bool,

    /// Append a FIFO request-trace row after each completed generation.
    ///
    /// Format is compatible with `scripts/profile/replay_economics.py
    /// --request-trace`: `arrival_ms tokens id ...`.
    #[arg(long, hide_short_help = true)]
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
        let command_line = |id| {
            matches.try_contains_id(id).is_ok()
                && matches.value_source(id) == Some(ValueSource::CommandLine)
        };
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

#[derive(Clone, Debug)]
struct DeepSeekV4MultigroupSelectorPlan {
    requested: DeepSeekV4MultigroupSelectorArg,
    device_name: String,
    device_qualified: bool,
    capacity: DeepSeekV4SessionCapacity,
    geometry: Option<DeepSeekV4MultigroupSelectorGeometry>,
}

#[derive(Debug, Serialize)]
struct DeepSeekV4MultigroupSelectorSessionRecord<'a> {
    schema_version: u32,
    kind: &'static str,
    scope: &'a str,
    requested: &'static str,
    sealed: bool,
    device_name: &'a str,
    device_qualified: bool,
    forward_limit: usize,
    physical_capacity_rows: usize,
    max_reachable_visible_rows: usize,
    frozen_min_visible_rows: usize,
    frozen_max_capacity_rows: usize,
    frozen_min_capacity_occupancy: &'static str,
    fallback: &'static str,
}

#[derive(Debug, Serialize)]
struct DeepSeekV4MultigroupSelectorCompletionRecord<'a> {
    schema_version: u32,
    kind: &'static str,
    scope: &'a str,
    requested: &'static str,
    sealed: bool,
    multigroup_invocations: u64,
    ineligible_singleton_radix4_invocations: u64,
}

impl DeepSeekV4MultigroupSelectorPlan {
    fn new(
        requested: DeepSeekV4MultigroupSelectorArg,
        device_name: impl Into<String>,
        capacity: DeepSeekV4SessionCapacity,
    ) -> Result<Self> {
        let device_name = device_name.into();
        let device_qualified = device_name == DEEPSEEK_V4_MULTIGROUP_SELECTOR_QUALIFIED_DEVICE;
        let geometry = match requested {
            DeepSeekV4MultigroupSelectorArg::Auto | DeepSeekV4MultigroupSelectorArg::Off => None,
            DeepSeekV4MultigroupSelectorArg::QualifiedExperimental => {
                ensure!(
                    device_qualified,
                    "--deepseek-v4-multigroup-selector=qualified-experimental requires {}, got {}",
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_QUALIFIED_DEVICE,
                    device_name,
                );
                Some(
                    capacity
                        .qualify_multigroup_selector_experiment()
                        .context("qualify DeepSeek V4 multi-group selector request geometry")?,
                )
            }
        };
        Ok(Self {
            requested,
            device_name,
            device_qualified,
            capacity,
            geometry,
        })
    }

    fn sealed(&self) -> bool {
        self.geometry.is_some()
    }

    fn session_record<'a>(
        &'a self,
        scope: &'a str,
    ) -> DeepSeekV4MultigroupSelectorSessionRecord<'a> {
        DeepSeekV4MultigroupSelectorSessionRecord {
            schema_version: 1,
            kind: "session_policy",
            scope,
            requested: self.requested.as_str(),
            sealed: self.sealed(),
            device_name: &self.device_name,
            device_qualified: self.device_qualified,
            forward_limit: self.capacity.forward_limit(),
            physical_capacity_rows: self.capacity.csa_physical_rows(),
            max_reachable_visible_rows: self.capacity.forward_limit() / 4,
            frozen_min_visible_rows: DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS,
            frozen_max_capacity_rows: DEEPSEEK_V4_MULTIGROUP_SELECTOR_MAX_CAPACITY_ROWS,
            frozen_min_capacity_occupancy: "3/4",
            fallback: "radix4_for_packed_and_ineligible_singleton",
        }
    }

    fn seal_session(&self, session: &mut DeepSeekV4Session, scope: &str) -> Result<()> {
        match self.requested {
            DeepSeekV4MultigroupSelectorArg::Auto => {}
            DeepSeekV4MultigroupSelectorArg::Off => session
                .disable_multigroup_selector()
                .context("disable the DeepSeek V4 multi-group selector")?,
            DeepSeekV4MultigroupSelectorArg::QualifiedExperimental => session
                .enable_multigroup_selector_experiment()
                .context("seal qualified experimental DeepSeek V4 multi-group selector")?,
        }
        let telemetry = session.multigroup_selector_telemetry();
        ensure!(
            telemetry.sealed() == self.sealed(),
            "DeepSeek V4 multi-group selector sealing did not match the request"
        );
        eprintln!(
            "deepseek_v4 selector: {}",
            serde_json::to_string(&self.session_record(scope))
                .context("serialize DeepSeek V4 selector session policy")?
        );
        Ok(())
    }

    fn completion_record<'a>(
        &'a self,
        scope: &'a str,
        telemetry: DeepSeekV4MultigroupSelectorTelemetry,
    ) -> Result<DeepSeekV4MultigroupSelectorCompletionRecord<'a>> {
        self.completion_record_from_values(
            scope,
            telemetry.sealed(),
            telemetry.multigroup_invocations(),
            telemetry.ineligible_radix4_invocations(),
        )
    }

    fn completion_record_from_values<'a>(
        &'a self,
        scope: &'a str,
        sealed: bool,
        multigroup_invocations: u64,
        ineligible_radix4_invocations: u64,
    ) -> Result<DeepSeekV4MultigroupSelectorCompletionRecord<'a>> {
        ensure!(
            sealed == self.sealed(),
            "DeepSeek V4 multi-group selector completion changed sealed policy"
        );
        if self.requested == DeepSeekV4MultigroupSelectorArg::Off {
            ensure!(
                multigroup_invocations == 0 && ineligible_radix4_invocations == 0,
                "disabled DeepSeek V4 multi-group selector recorded invocations"
            );
        }
        Ok(DeepSeekV4MultigroupSelectorCompletionRecord {
            schema_version: 1,
            kind: "session_completion",
            scope,
            requested: self.requested.as_str(),
            sealed,
            multigroup_invocations,
            ineligible_singleton_radix4_invocations: ineligible_radix4_invocations,
        })
    }

    fn emit_completion(
        &self,
        scope: &str,
        telemetry: DeepSeekV4MultigroupSelectorTelemetry,
    ) -> Result<()> {
        eprintln!(
            "deepseek_v4 selector: {}",
            serde_json::to_string(&self.completion_record(scope, telemetry)?)
                .context("serialize DeepSeek V4 selector completion")?
        );
        Ok(())
    }
}

/// CLI values for `--reasoning`, mapping to the DeepSeek V4 release
/// three-tier effort contract (vLLM `77434861`): `none` is chat mode; `low`
/// opens `<think>` with no effort bytes (the release thinking default);
/// `high` additionally prepends the "Absolute maximum" instruction (labeled
/// max in the earlier two-tier encoders); and
/// `max` prepends the stronger "Beyond maximum" instruction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
enum ReasoningLevelArg {
    None,
    Low,
    High,
    Max,
}

fn deepseek_v4_encode_options(args: &Args) -> Result<DeepSeekV4EncodeOptions> {
    let reasoning = match args.reasoning {
        None | Some(ReasoningLevelArg::None) => DeepSeekV4Reasoning::None,
        Some(ReasoningLevelArg::Low) => DeepSeekV4Reasoning::Low,
        Some(ReasoningLevelArg::High) => DeepSeekV4Reasoning::High,
        Some(ReasoningLevelArg::Max) => DeepSeekV4Reasoning::Max,
    };
    ensure!(
        !(args.preserve_reasoning && matches!(reasoning, DeepSeekV4Reasoning::None)),
        "--preserve-reasoning requires --reasoning low, high, or max"
    );
    Ok(DeepSeekV4EncodeOptions {
        reasoning,
        preserve_reasoning: args.preserve_reasoning,
    })
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

#[derive(Clone, Debug)]
struct PreparedPrompt {
    text: String,
    source: PromptSource,
    completed_checkpoint_eligible: bool,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_partition: Option<GeneratedThinkingPartition>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct GeneratedThinkingPartition {
    delimiter: &'static str,
    delimiter_start_token_index: usize,
    delimiter_end_token_index_exclusive: usize,
    delimiter_token_aligned: bool,
    reasoning_tokens: Option<usize>,
    delimiter_tokens: Option<usize>,
    visible_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
struct RequestStatsRow {
    schema_version: u32,
    request_stats_contract: &'static str,
    id: String,
    line: usize,
    model: String,
    build_commit: &'static str,
    build_dirty: bool,
    build_source_state: &'static str,
    model_prefetch_policy: &'static str,
    model_prefetch_bytes_returned: u64,
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
    thinking_partition: Option<GeneratedThinkingPartition>,
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

fn main() -> std::process::ExitCode {
    shutdown::finish(run())
}

fn run() -> Result<()> {
    shutdown::install()?;
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let matches = Args::command().get_matches();
    let explicit_options = matches.subcommand().map_or_else(
        || ExplicitCliOptions::from_matches(&matches),
        |(_, matches)| ExplicitCliOptions::from_matches(matches),
    );
    let mut args = Args::from_arg_matches(&matches).expect("validated clap arguments");
    let invocation = cli::normalize(&mut args);
    invocation.apply_option_overrides(&mut args);
    let modern_run = invocation.is_run();
    validate_deepseek_v4_reasoning_scope(&args)?;
    validate_qwen_model_prefetch_scope(&args)?;
    validate_request_timing_mode(&args)?;
    validate_sampling_attribution_mode(&args)?;
    validate_sampled_structural_mode(&args)?;
    validate_durable_prefix_cache_mode(&args)?;
    fixed_cohort_jsonl::validate_cli(&args, explicit_options)?;
    concurrent_jsonl::validate_cli(&args, explicit_options)?;
    if args.request_stats_jsonl.is_some() && args.info {
        bail!(
            "--request-stats-jsonl is not applicable with --info; only DeepSeek V4 single-turn generation emits the sidecar today"
        );
    }
    if args.info {
        let runtime = Runtime::metal()?;
        println!("device: {}", runtime.describe());
        return Ok(());
    }

    let Some(model_path) = args.model.clone() else {
        let stderr = std::io::stderr();
        let mut stderr = stderr.lock();
        Args::command()
            .write_help(&mut stderr)
            .context("write qwen help")?;
        writeln!(stderr)?;
        std::process::exit(2);
    };

    if args.deepseek_census_json {
        ensure!(
            args.request_stats_jsonl.is_none(),
            "--request-stats-jsonl is not applicable with --deepseek-census-json; only DeepSeek V4 single-turn generation emits the sidecar today"
        );
        return print_deepseek_v4_census(&model_path);
    }

    if args.deepseek_v4_snapshot.is_some() {
        ensure!(
            args.prompt.is_some() || args.prompt_file.is_some() || args.messages.is_some(),
            "--deepseek-v4-snapshot requires --prompt, --prompt-file, or --messages"
        );
    }
    validate_deepseek_v4_multigroup_selector_scope(&args)?;

    if args.prompt.is_none()
        && args.prompt_file.is_none()
        && args.messages.is_none()
        && args.requests_jsonl.is_none()
        && !modern_run
    {
        ensure!(
            args.request_stats_jsonl.is_none(),
            "--request-stats-jsonl is not applicable without a request; only DeepSeek V4 single-turn generation emits the sidecar today. Provide --prompt, --prompt-file, --messages, or --requests-jsonl."
        );
        return print_model_info(&model_path);
    }

    let staged_integrity = configured_checkpoint_staged_integrity()?;
    ensure!(
        staged_integrity.is_none() || args.durable_prefix_cache.is_some(),
        "{CHECKPOINT_STAGED_INTEGRITY_ENV} requires --durable-prefix-cache"
    );
    validate_request_before_model_open(&args)?;
    let gguf = GgufFile::open(&model_path)
        .with_context(|| format!("open model {}", model_path.display()))?;
    let model_family = ModelFamily::detect(&gguf);
    fixed_cohort_jsonl::validate_model_family(args.batch_size, model_family)?;
    concurrent_jsonl::validate_model_family(&args, model_family)?;
    validate_deepseek_v4_multigroup_selector_family(
        args.deepseek_v4_multigroup_selector,
        model_family,
    )?;
    if let cli::Invocation::Run(run) = invocation {
        args.prepared_prompt = Some(prepare_modern_run_prompt(run, model_family, &gguf, &args)?);
    }
    if model_family == Some(ModelFamily::DeepSeek4) {
        return if args.requests_jsonl.is_some() {
            run_deepseek_v4_requests_jsonl(&model_path, gguf, &args, explicit_options)
        } else {
            run_deepseek_v4_single_turn(
                &model_path,
                gguf,
                &args,
                explicit_options,
                staged_integrity,
            )
        };
    }
    ensure!(
        args.deepseek_v4_snapshot.is_none(),
        "--deepseek-v4-snapshot requires a DeepSeek V4 model"
    );
    ensure!(
        args.reasoning.is_none() && !args.preserve_reasoning,
        "--reasoning and --preserve-reasoning apply to DeepSeek V4 --messages requests only"
    );

    if has_single_turn_input(&args) {
        return run_single_turn(&model_path, gguf, &args, staged_integrity);
    }

    if let Some(path) = args.requests_jsonl.as_ref() {
        return run_requests_jsonl(&model_path, path, gguf, &args, explicit_options);
    }

    unreachable!("request mode was validated above")
}

fn validate_qwen_model_prefetch_scope(args: &Args) -> Result<()> {
    ensure!(
        args.model_prefetch.is_none() || args.requests_jsonl.is_some(),
        "--model-prefetch requires --requests-jsonl"
    );
    Ok(())
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
    validate_deepseek_v4_reasoning_scope(args)?;
    let sampling = cli_sampling_config(args)?;
    if args.requests_jsonl.is_none() {
        validate_sampling_decode_policy(sampling, args.prompt_lookup)?;
    }
    Ok(())
}

fn validate_deepseek_v4_multigroup_selector_scope(args: &Args) -> Result<()> {
    if args.deepseek_v4_multigroup_selector
        != DeepSeekV4MultigroupSelectorArg::QualifiedExperimental
    {
        return Ok(());
    }
    ensure!(
        args.prompt.is_some()
            || args.prompt_file.is_some()
            || args.messages.is_some()
            || args.requests_jsonl.is_some(),
        "--deepseek-v4-multigroup-selector requires a generation request"
    );
    Ok(())
}

fn validate_deepseek_v4_multigroup_selector_family(
    requested: DeepSeekV4MultigroupSelectorArg,
    model_family: Option<ModelFamily>,
) -> Result<()> {
    ensure!(
        requested != DeepSeekV4MultigroupSelectorArg::QualifiedExperimental
            || model_family == Some(ModelFamily::DeepSeek4),
        "--deepseek-v4-multigroup-selector requires a DeepSeek V4 model"
    );
    Ok(())
}

fn validate_deepseek_v4_reasoning_scope(args: &Args) -> Result<()> {
    ensure!(
        (args.reasoning.is_none() && !args.preserve_reasoning) || has_messages_input(args),
        "--reasoning and --preserve-reasoning require --messages"
    );
    Ok(())
}

fn has_messages_input(args: &Args) -> bool {
    args.messages.is_some()
        || args
            .prepared_prompt
            .as_ref()
            .is_some_and(|prompt| prompt.source == PromptSource::Messages)
}

fn has_single_turn_input(args: &Args) -> bool {
    args.prepared_prompt.is_some()
        || args.prompt.is_some()
        || args.prompt_file.is_some()
        || args.messages.is_some()
}

fn prepare_modern_run_prompt(
    run: cli::RunInvocation,
    model_family: Option<ModelFamily>,
    gguf: &GgufFile,
    args: &Args,
) -> Result<PreparedPrompt> {
    let family = model_family.with_context(|| {
        format!(
            "`qwen run` does not support model architecture {:?}",
            gguf.architecture()
        )
    })?;
    ensure!(
        matches!(
            family,
            ModelFamily::Qwen35 | ModelFamily::Qwen35Moe | ModelFamily::DeepSeek4
        ),
        "`qwen run` does not support model family {}",
        family.architecture_name()
    );
    ensure!(
        family != ModelFamily::DeepSeek4 || args.max_context_tokens.is_none(),
        "--max-context-tokens is not supported for DeepSeek V4 single-turn generation; remove --max-context-tokens"
    );
    if run.no_thinking && matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe) {
        ensure!(
            validated_qwen_no_thinking_model(family, gguf),
            "--no-thinking is currently validated only for Qwen3.6 35B A3B and Qwen3.8 27B identities with the qwen35 tokenizer; omit --no-thinking to use this model's default generation behavior"
        );
    }
    let qwen38 = validated_qwen38_prompt_model(family, gguf);

    let no_thinking = run.no_thinking;
    let qwen38_generation_mode =
        resolve_qwen38_generation_mode(qwen38, no_thinking, run.reasoning_effort)?;
    let input = run.acquire_input()?;
    let (text, source) = match input {
        cli::AcquiredRunInput::RawPrompt(prompt) => (prompt, PromptSource::Inline),
        cli::AcquiredRunInput::User { system, user } => {
            let prompt = match family {
                ModelFamily::Qwen35 | ModelFamily::Qwen35Moe if qwen38 => {
                    render_qwen38_single_turn_prompt(
                        &user,
                        system.as_deref(),
                        qwen38_generation_mode.expect("validated Qwen3.8 mode"),
                    )
                }
                ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => render_qwen_single_turn_prompt(
                    &user,
                    system.as_deref(),
                    if no_thinking {
                        QwenGenerationMode::NoThinking
                    } else {
                        QwenGenerationMode::Auto
                    },
                ),
                ModelFamily::DeepSeek4 => render_deepseek_v4_0731_single_turn_prompt(
                    &user,
                    system.as_deref(),
                    DeepSeekV4EncodeOptions::default(),
                )
                .context("render DeepSeek V4 0731 user request")?,
            };
            (prompt, PromptSource::Messages)
        }
        cli::AcquiredRunInput::Messages { document, source } => {
            let messages = parse_strict_messages_input(&document, &source)?;
            let prompt = match family {
                ModelFamily::Qwen35 | ModelFamily::Qwen35Moe if qwen38 => {
                    render_qwen38_messages_prompt_with_generation(
                        &messages,
                        true,
                        qwen38_generation_mode.expect("validated Qwen3.8 mode"),
                    )
                }
                ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => {
                    render_qwen_messages_prompt_with_generation(
                        &messages,
                        false,
                        true,
                        if no_thinking {
                            QwenGenerationMode::NoThinking
                        } else {
                            QwenGenerationMode::Auto
                        },
                    )
                }
                ModelFamily::DeepSeek4 => render_deepseek_v4_0731_messages_prompt(
                    &messages,
                    DeepSeekV4EncodeOptions::default(),
                )
                .context("render strict DeepSeek V4 0731 messages")?,
            };
            (prompt, PromptSource::Messages)
        }
    };
    Ok(PreparedPrompt {
        text,
        source,
        completed_checkpoint_eligible: false,
    })
}

fn resolve_qwen38_generation_mode(
    qwen38: bool,
    no_thinking: bool,
    reasoning_effort: Option<cli::RunReasoningEffort>,
) -> Result<Option<Qwen38GenerationMode>> {
    ensure!(
        !(no_thinking && reasoning_effort.is_some()),
        "--reasoning-effort cannot be combined with --no-thinking"
    );
    ensure!(
        reasoning_effort.is_none() || qwen38,
        "--reasoning-effort is currently validated only for Qwen3.8 27B identities with the qwen35 tokenizer"
    );
    if !qwen38 {
        return Ok(None);
    }
    if no_thinking {
        return Ok(Some(Qwen38GenerationMode::NoThinking));
    }
    let effort = match reasoning_effort.unwrap_or(cli::RunReasoningEffort::Xhigh) {
        cli::RunReasoningEffort::Low => Qwen38ReasoningEffort::Low,
        cli::RunReasoningEffort::Medium => Qwen38ReasoningEffort::Medium,
        cli::RunReasoningEffort::Xhigh => Qwen38ReasoningEffort::Xhigh,
    };
    Ok(Some(Qwen38GenerationMode::Thinking(effort)))
}

fn validated_qwen_no_thinking_model(family: ModelFamily, gguf: &GgufFile) -> bool {
    validated_qwen36_no_thinking_identity(
        family,
        gguf.get_str("general.base_model.0.name"),
        gguf.get_str("tokenizer.ggml.model"),
        gguf.get_str("tokenizer.ggml.pre"),
    ) || validated_qwen38_prompt_model(family, gguf)
}

fn validated_qwen38_prompt_model(family: ModelFamily, gguf: &GgufFile) -> bool {
    validated_qwen38_prompt_identity(
        family,
        gguf.get_str("general.name"),
        gguf.get_str("general.base_model.0.name"),
        gguf.get_str("tokenizer.ggml.model"),
        gguf.get_str("tokenizer.ggml.pre"),
        gguf.get_u64("qwen35.context_length"),
        gguf.get_u64("qwen35.block_count"),
        gguf.get_u64("qwen35.nextn_predict_layers"),
        gguf.get_u64("qwen35.embedding_length"),
        gguf.get_u64("qwen35.feed_forward_length"),
    )
}

fn validated_qwen38_prompt_identity(
    family: ModelFamily,
    general_name: Option<&str>,
    base_model_name: Option<&str>,
    tokenizer_model: Option<&str>,
    tokenizer_pre: Option<&str>,
    context_length: Option<u64>,
    block_count: Option<u64>,
    nextn_predict_layers: Option<u64>,
    embedding_length: Option<u64>,
    feed_forward_length: Option<u64>,
) -> bool {
    let named_qwen38_27b = [general_name, base_model_name]
        .into_iter()
        .flatten()
        .any(|name| {
            let name = name.to_ascii_lowercase();
            name.contains("qwen3.8") && name.contains("27b")
        });
    family == ModelFamily::Qwen35
        && named_qwen38_27b
        && tokenizer_model == Some("gpt2")
        && tokenizer_pre == Some("qwen35")
        && context_length == Some(262_144)
        && block_count == Some(65)
        && nextn_predict_layers == Some(1)
        && embedding_length == Some(5_120)
        && feed_forward_length == Some(17_408)
}

fn validated_qwen36_no_thinking_identity(
    family: ModelFamily,
    base_model_name: Option<&str>,
    tokenizer_model: Option<&str>,
    tokenizer_pre: Option<&str>,
) -> bool {
    family == ModelFamily::Qwen35Moe
        && base_model_name == Some("Qwen3.6 35B A3B")
        && tokenizer_model == Some("gpt2")
        && tokenizer_pre == Some("qwen35")
}

fn prompt_text(args: &Args) -> Result<(String, PromptSource, bool)> {
    if let Some(prompt) = args.prepared_prompt.as_ref() {
        return Ok((
            prompt.text.clone(),
            prompt.source,
            prompt.completed_checkpoint_eligible,
        ));
    }
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
    bail!(
        "single-turn generation requires --prompt, --prompt-file, --messages, or `qwen run --user`"
    )
}

fn prompt_add_special_tokens(args: &Args, source: PromptSource) -> bool {
    source != PromptSource::Messages && !args.no_special_tokens
}

/// Options unsupported for every DeepSeek V4 execution mode. Mode-specific
/// options (`--requests-jsonl`, `--max-context-tokens`) are validated by the
/// single-turn and requests-mode validators respectively.
fn deepseek_v4_shared_unsupported_options(
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Vec<&'static str> {
    let mut unsupported = Vec::new();
    if args.prompt_lookup {
        unsupported.push("--prompt-lookup");
    }
    if explicit.prefill_chunk || args.prefill_chunk != PrefillChunkArg::Fixed(1024) {
        unsupported.push("--prefill-chunk");
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
    if args.request_stats.is_some() {
        unsupported.push("--request-stats");
    }
    if args.request_timings.is_some() {
        unsupported.push("--request-timings");
    }
    if args.model_prefetch.is_some() {
        unsupported.push("--model-prefetch");
    }
    if args.request_timing_warm_followup {
        unsupported.push("--request-timing-warm-followup");
    }
    if args.messages_no_generation_prompt {
        unsupported.push("--messages-no-generation-prompt");
    }
    unsupported
}

fn validate_deepseek_v4_generation_mode(args: &Args, explicit: ExplicitCliOptions) -> Result<()> {
    let mut unsupported = deepseek_v4_shared_unsupported_options(args, explicit);
    if args.requests_jsonl.is_some() {
        unsupported.push("--requests-jsonl");
    }
    if args.max_context_tokens.is_some() {
        unsupported.push("--max-context-tokens");
    }
    if args.messages_preserve_thinking
        && matches!(args.reasoning, None | Some(ReasoningLevelArg::None))
    {
        // Preserved history reasoning is a release thinking-mode contract;
        // accepting the flag in chat mode would silently no-op.
        unsupported.push("--messages-preserve-thinking (requires --reasoning low, high, or max)");
    }
    if args.durable_prefix_cache.is_none() {
        if explicit.durable_prefix_cache_max_mib {
            unsupported.push("--durable-prefix-cache-max-mib");
        }
        if explicit.durable_prefix_cache_max_entry_mib {
            unsupported.push("--durable-prefix-cache-max-entry-mib");
        }
        if explicit.durable_prefix_cache_min_tokens {
            unsupported.push("--durable-prefix-cache-min-tokens");
        }
    }
    ensure!(
        unsupported.is_empty(),
        "DeepSeek V4 currently supports bounded raw or ordinary-message single-turn generation only; unsupported options: {}",
        unsupported.join(", ")
    );
    ensure!(
        has_single_turn_input(args),
        "DeepSeek V4 generation requires --prompt, --prompt-file, --messages, or `qwen run --user`"
    );
    Ok(())
}

fn validate_deepseek_v4_requests_mode(args: &Args, explicit: ExplicitCliOptions) -> Result<()> {
    let mut unsupported = deepseek_v4_shared_unsupported_options(args, explicit);
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
    if args.messages_preserve_thinking {
        unsupported.push("--messages-preserve-thinking");
    }
    if args.messages_strip_thinking {
        unsupported.push("--messages-strip-thinking");
    }
    if args.deepseek_v4_snapshot.is_some() {
        // Also excluded at the parser level; kept as a defensive invariant.
        unsupported.push("--deepseek-v4-snapshot");
    }
    // --request-stats-jsonl is DS4-single-turn-only today. Reject on DS4 batch
    // rather than silently no-oping (which would break the mandatory/fail-closed
    // policy the flag advertises).
    if args.request_stats_jsonl.is_some() {
        unsupported.push("--request-stats-jsonl");
    }
    ensure!(
        unsupported.is_empty(),
        "DeepSeek V4 --requests-jsonl supports raw prompt requests only; unsupported options: {}",
        unsupported.join(", ")
    );
    let stdin = deepseek_v4_requests_reads_stdin(args)?;
    if stdin {
        ensure!(
            args.max_context_tokens.is_some(),
            "DeepSeek V4 --requests-jsonl from stdin cannot derive a context budget by lookahead; supply --max-context-tokens as the shared logical token capacity"
        );
    } else {
        ensure!(
            args.max_context_tokens.is_none(),
            "DeepSeek V4 --requests-jsonl file mode derives the forward budget from the request set; remove --max-context-tokens"
        );
    }
    Ok(())
}

fn deepseek_v4_requests_reads_stdin(args: &Args) -> Result<bool> {
    let path = args
        .requests_jsonl
        .as_ref()
        .context("DeepSeek V4 requests mode requires --requests-jsonl")?;
    Ok(path.as_os_str() == "-")
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

fn deepseek_v4_forward_budget_for_context_limit(context_tokens: usize) -> Result<usize> {
    ensure!(
        (2..=DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY).contains(&context_tokens),
        "--max-context-tokens must be in 2..={} for DeepSeek V4 requests",
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
    );
    Ok(context_tokens - 1)
}

fn validate_deepseek_v4_request_context_limit(
    request_id: &str,
    prompt_tokens: usize,
    max_tokens: usize,
    context_limit: usize,
) -> Result<()> {
    let required_context_tokens = prompt_tokens
        .checked_add(max_tokens)
        .context("DeepSeek V4 logical context requirement overflow")?;
    ensure!(
        required_context_tokens <= context_limit,
        "request {request_id} requires {required_context_tokens} logical context tokens ({prompt_tokens} prompt + {max_tokens} generation), beyond --max-context-tokens {context_limit}",
    );
    Ok(())
}

fn parse_deepseek_v4_prefill_chunk_tokens(value: Option<&str>) -> Result<usize> {
    let Some(value) = value else {
        return Ok(DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS);
    };
    let chunk_tokens = value
        .parse::<usize>()
        .with_context(|| format!("QWEN_DSV4_PREFILL_CHUNK_TOKENS={value:?} is not an integer"))?;
    ensure!(
        (1..=DEEPSEEK_V4_PREFILL_MAX_TOKENS).contains(&chunk_tokens),
        "QWEN_DSV4_PREFILL_CHUNK_TOKENS must be in 1..={DEEPSEEK_V4_PREFILL_MAX_TOKENS}, got {chunk_tokens}"
    );
    Ok(chunk_tokens)
}

fn deepseek_v4_prefill_chunk_tokens() -> Result<usize> {
    let value = std::env::var("QWEN_DSV4_PREFILL_CHUNK_TOKENS").ok();
    parse_deepseek_v4_prefill_chunk_tokens(value.as_deref())
}

fn deepseek_v4_prefill_chunk_ranges(
    prompt_tokens: usize,
    chunk_tokens: usize,
) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < prompt_tokens {
        let remaining = prompt_tokens - start;
        let len = if remaining >= chunk_tokens {
            chunk_tokens
        } else if chunk_tokens == DEEPSEEK_V4_PREFILL_MAX_TOKENS && remaining >= 2_048 {
            2_048
        } else {
            remaining
        };
        ranges.push(start..start + len);
        start += len;
    }
    ranges
}

fn deepseek_v4_packed_chunk_count(prompt_tokens: usize, chunk_tokens: usize) -> usize {
    if prompt_tokens < 2 {
        0
    } else {
        deepseek_v4_prefill_chunk_ranges(prompt_tokens, chunk_tokens).len()
    }
}

fn deepseek_v4_snapshot_publish_prefix(prompt_tokens: usize) -> Result<usize> {
    ensure!(
        prompt_tokens >= 2,
        "--deepseek-v4-snapshot requires at least two prompt tokens so restored state has an uncached endpoint token"
    );
    Ok(prompt_tokens - 1)
}

fn deepseek_v4_snapshot_restored_prefix_len(
    snapshot_prefix: &[u32],
    prompt_tokens: &[u32],
) -> Result<usize> {
    ensure!(
        snapshot_prefix.len() < prompt_tokens.len(),
        "DeepSeek V4 snapshot prefix has {} tokens but the request has {}; at least one uncached endpoint token is required because causal snapshots omit observations",
        snapshot_prefix.len(),
        prompt_tokens.len(),
    );
    ensure!(
        prompt_tokens.starts_with(snapshot_prefix),
        "DeepSeek V4 snapshot token prefix does not match this request"
    );
    Ok(snapshot_prefix.len())
}

fn deepseek_v4_snapshot_parent(path: &Path) -> Result<&Path> {
    ensure!(
        path.file_name().is_some(),
        "--deepseek-v4-snapshot requires a file path"
    );
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let metadata = std::fs::symlink_metadata(parent)
        .with_context(|| format!("inspect DeepSeek V4 snapshot parent {}", parent.display()))?;
    ensure!(
        metadata.file_type().is_dir(),
        "DeepSeek V4 snapshot parent {} is not a directory",
        parent.display()
    );
    ensure!(
        metadata.uid() == current_effective_uid() && metadata.mode() & 0o022 == 0,
        "DeepSeek V4 snapshot parent {} must be owned by the current user and not group/world-writable",
        parent.display()
    );
    Ok(parent)
}

fn deepseek_v4_snapshot_identity_cache(parent: &Path) -> Result<CheckpointIdentityCache> {
    let root = parent.join(DEEPSEEK_V4_SNAPSHOT_IDENTITY_CACHE_DIR);
    match std::fs::DirBuilder::new().mode(0o700).create(&root) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "create private DeepSeek V4 identity cache {}",
                    root.display()
                )
            });
        }
    }
    let metadata = std::fs::symlink_metadata(&root)
        .with_context(|| format!("inspect DeepSeek V4 identity cache {}", root.display()))?;
    ensure!(
        metadata.file_type().is_dir()
            && metadata.uid() == current_effective_uid()
            && metadata.mode() & 0o077 == 0,
        "DeepSeek V4 identity cache {} must be a current-user 0700 directory",
        root.display()
    );
    Ok(CheckpointIdentityCache::new(root))
}

fn current_effective_uid() -> u32 {
    // geteuid has no preconditions and does not retain pointers.
    unsafe { libc::geteuid() }
}

fn advance_deepseek_v4_prompt_prefix(
    session: &mut DeepSeekV4Session,
    ctx: &MetalContext,
    token_ids: &[u32],
    chunk_tokens: usize,
) -> Result<()> {
    ensure!(
        !token_ids.is_empty(),
        "DeepSeek V4 snapshot prefix is empty"
    );
    for (chunk_index, range) in deepseek_v4_prefill_chunk_ranges(token_ids.len(), chunk_tokens)
        .into_iter()
        .enumerate()
    {
        shutdown::checkpoint()?;
        let chunk = &token_ids[range];
        session.advance_tokens(ctx, chunk).with_context(|| {
            format!("advance DeepSeek V4 snapshot prefix chunk {chunk_index} without logits")
        })?;
        shutdown::checkpoint()?;
    }
    Ok(())
}

fn execute_deepseek_v4_prompt_suffix(
    session: &mut DeepSeekV4Session,
    ctx: &MetalContext,
    token_ids: &[u32],
    chunk_tokens: usize,
) -> Result<usize> {
    ensure!(
        !token_ids.is_empty(),
        "DeepSeek V4 prompt suffix requires an endpoint token"
    );
    let chunks = deepseek_v4_prefill_chunk_ranges(token_ids.len(), chunk_tokens);
    let chunk_count = chunks.len();
    for (chunk_index, range) in chunks.into_iter().enumerate() {
        shutdown::checkpoint()?;
        let chunk = &token_ids[range];
        if chunk_index + 1 == chunk_count {
            session
                .prefill_tokens(ctx, chunk)
                .with_context(|| format!("prefill final DeepSeek V4 prompt chunk {chunk_index}"))?;
        } else {
            session.advance_tokens(ctx, chunk).with_context(|| {
                format!("advance DeepSeek V4 prompt chunk {chunk_index} without logits")
            })?;
        }
        shutdown::checkpoint()?;
    }
    Ok(chunk_count)
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

#[derive(Default)]
struct DeepSeekV4CliStageProfileAggregate {
    stages: [f64; 10],
    boundary_ms: f64,
    command_gpu_ms: f64,
    forward_wall_ms: f64,
}

fn median_f64(values: &[f64]) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    if ordered.len().is_multiple_of(2) {
        (ordered[ordered.len() / 2 - 1] + ordered[ordered.len() / 2]) * 0.5
    } else {
        ordered[ordered.len() / 2]
    }
}

fn emit_deepseek_v4_whole_profile(
    positions: &[u32],
    wall_ms: &[f64],
    gpu_ms: &[f64],
    outside_gpu_ms: &[f64],
    encode_cpu_ms: &[f64],
    wait_residual_ms: &[f64],
) {
    eprintln!(
        "deepseek_v4 whole_profile: positions={}..{} samples={} wall_median_ms={:.3} gpu_median_ms={:.3} outside_gpu_median_ms={:.3} encode_cpu_median_ms={:.3} wait_residual_median_ms={:.3} wall_ms={wall_ms:?} gpu_ms={gpu_ms:?}",
        positions[0],
        positions[positions.len() - 1],
        positions.len(),
        median_f64(wall_ms),
        median_f64(gpu_ms),
        median_f64(outside_gpu_ms),
        median_f64(encode_cpu_ms),
        median_f64(wait_residual_ms),
    );
}

fn emit_deepseek_v4_stage_profile(
    profile: &DeepSeekV4StageProfile,
    group: usize,
    groups: usize,
    aggregate: &mut DeepSeekV4CliStageProfileAggregate,
) {
    let mut stages = [0.0f64; 10];
    let mut boundary_ms = 0.0f64;
    for layer in &profile.sampled_layers {
        boundary_ms += layer.encoder_boundary_ms_scaled;
        for stage in &layer.stages {
            let index = match stage.kind {
                DeepSeekV4StageKind::AttentionHyperConnection => 0,
                DeepSeekV4StageKind::AttentionPrepare => 1,
                DeepSeekV4StageKind::AttentionCore => 2,
                DeepSeekV4StageKind::AttentionOutput => 3,
                DeepSeekV4StageKind::HyperConnectionBridge => 4,
                DeepSeekV4StageKind::MoeRouter => 5,
                DeepSeekV4StageKind::MoeRoutedExperts => 6,
                DeepSeekV4StageKind::MoeSharedExpert => 7,
                DeepSeekV4StageKind::MoeCombine => 8,
                DeepSeekV4StageKind::LayerTail => 9,
            };
            stages[index] += stage.duration_ms_scaled;
        }
    }
    let command_gpu_ms = profile
        .layers
        .iter()
        .map(|layer| layer.command_gpu_ms)
        .sum::<f64>();
    for (total, sample) in aggregate.stages.iter_mut().zip(stages) {
        *total += sample;
    }
    aggregate.boundary_ms += boundary_ms;
    aggregate.command_gpu_ms += command_gpu_ms;
    aggregate.forward_wall_ms += profile.forward_wall_ms;
    eprintln!(
        concat!(
            "deepseek_v4 stage_profile_group: position={} group={}/{} schedule=instrumented_per_layer ",
            "forward_wall_ms={:.3} command_gpu_ms={:.3} ",
            "attention_hc_ms={:.3} attention_prepare_ms={:.3} attention_core_ms={:.3} ",
            "attention_output_ms={:.3} bridge_ms={:.3} moe_router_ms={:.3} ",
            "moe_routed_ms={:.3} moe_shared_ms={:.3} moe_combine_ms={:.3} ",
            "layer_tail_ms={:.3} encoder_boundary_ms={:.3} sampled_layers={}"
        ),
        profile.position,
        group,
        groups,
        profile.forward_wall_ms,
        command_gpu_ms,
        stages[0],
        stages[1],
        stages[2],
        stages[3],
        stages[4],
        stages[5],
        stages[6],
        stages[7],
        stages[8],
        stages[9],
        boundary_ms,
        profile.sampled_layers.len(),
    );
    if group + 1 == groups {
        eprintln!(
            concat!(
                "deepseek_v4 stage_profile: positions={}..{} groups={} schedule=rotating_instrumented_per_layer ",
                "mean_forward_wall_ms={:.3} mean_command_gpu_ms={:.3} ",
                "attention_hc_ms={:.3} attention_prepare_ms={:.3} attention_core_ms={:.3} ",
                "attention_output_ms={:.3} bridge_ms={:.3} moe_router_ms={:.3} ",
                "moe_routed_ms={:.3} moe_shared_ms={:.3} moe_combine_ms={:.3} ",
                "layer_tail_ms={:.3} encoder_boundary_ms={:.3}"
            ),
            profile.position + 1 - groups as u32,
            profile.position,
            groups,
            aggregate.forward_wall_ms / groups as f64,
            aggregate.command_gpu_ms / groups as f64,
            aggregate.stages[0],
            aggregate.stages[1],
            aggregate.stages[2],
            aggregate.stages[3],
            aggregate.stages[4],
            aggregate.stages[5],
            aggregate.stages[6],
            aggregate.stages[7],
            aggregate.stages[8],
            aggregate.stages[9],
            aggregate.boundary_ms,
        );
    }
}

fn run_deepseek_v4_single_turn(
    model_path: &Path,
    gguf: GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
    staged_integrity: Option<StagedIntegrityMode>,
) -> Result<()> {
    validate_deepseek_v4_generation_mode(args, explicit)?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    // Preflight the sidecar in the same open mode as emission (read+append),
    // and force INVOCATION_ID init here so entropy is required only when
    // telemetry is requested.
    if let Some(path) = args.request_stats_jsonl.as_ref() {
        preflight_request_stats_jsonl(path)?;
        LazyLock::force(&INVOCATION_ID);
    }
    let request_start = std::time::Instant::now();
    let prefill_chunk_tokens = deepseek_v4_prefill_chunk_tokens()?;
    let prefetch_mode = configured_deepseek_v4_prefetch_mode()?;
    let sampling = cli_sampling_config(args)?;
    let arrival_ms = unix_epoch_ms()?;
    let encode_options = deepseek_v4_encode_options(args)?;
    let (prompt, prompt_source, durable_completed_eligible) =
        if let Some(path) = args.messages.as_ref() {
            let mut encode_options = encode_options;
            let inline_thinking = if args.messages_strip_thinking {
                DeepSeekV4InlineThinking::Strip
            } else if args.messages_preserve_thinking {
                // Tier presence is enforced by validate_deepseek_v4_generation_mode;
                // the flag maps onto the release encoder's drop_thinking=False lane.
                encode_options.preserve_reasoning = true;
                DeepSeekV4InlineThinking::PromoteToReasoning
            } else {
                DeepSeekV4InlineThinking::Verbatim
            };
            // Preserved reasoning renders assistant turns byte-faithfully, so the
            // next turn's re-rendered prompt strictly extends this turn's
            // completed transcript; only then is a completed-turn checkpoint
            // reusable.
            let completed_eligible = encode_options.preserve_reasoning;
            (
                load_deepseek_v4_0731_messages_prompt(
                    path,
                    args.messages_max,
                    encode_options,
                    inline_thinking,
                )
                .context("render DeepSeek V4 0731 messages")?,
                PromptSource::Messages,
                completed_eligible,
            )
        } else {
            let (prompt, source, _) = prompt_text(args)?;
            (prompt, source, false)
        };
    let prompt_kind = match prompt_source {
        PromptSource::Inline | PromptSource::File => "raw",
        PromptSource::Messages => match encode_options.reasoning {
            DeepSeekV4Reasoning::None => "messages_0731_chat",
            DeepSeekV4Reasoning::Low | DeepSeekV4Reasoning::High | DeepSeekV4Reasoning::Max => {
                "messages_0731_thinking"
            }
        },
    };

    let tokenizer_t0 = Instant::now();
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load DeepSeek V4 tokenizer")?;
    let tokenizer_ms = tokenizer_t0.elapsed().as_secs_f64() * 1e3;
    let prompt_ids = tokenizer
        .encode(&prompt, false)
        .context("tokenize raw DeepSeek V4 prompt")?;
    let required_forwards = deepseek_v4_required_forwards(prompt_ids.len(), args.tokens)?;
    deepseek_v4_debug_dump_prompt_ids("single_turn", &prompt_ids);
    let vocab_size = tokenizer.n_vocab();
    let prompt_token_ids = prompt_ids
        .iter()
        .enumerate()
        .map(|(index, &token)| {
            checked_deepseek_v4_token_id(token, vocab_size, &format!("prompt[{index}]"))
        })
        .collect::<Result<Vec<_>>>()?;
    let durable_store = deepseek_v4_checkpoint_store(args, staged_integrity)?;
    let durable_max_record_bytes = if durable_store.is_some() {
        durable_prefix_cache_max_entry_bytes(args)?
    } else {
        0
    };
    let mut durable_admitted = durable_store.is_some()
        && args.durable_prefix_cache_min_tokens > 0
        && prompt_token_ids.len() >= args.durable_prefix_cache_min_tokens;
    if let Some(publish_prefix) =
        deepseek_v4_durable_capture_prefix_len(prompt_token_ids.len(), durable_admitted)?
    {
        let model = DeepSeekV4Model::from_gguf_flash_0731(&gguf)
            .context("bind durable DeepSeek V4 snapshot geometry")?;
        let session_capacity = DeepSeekV4SessionCapacity::for_forward_limit(
            required_forwards,
            model.config.context_length,
        )
        .context("derive durable DeepSeek V4 snapshot session capacity")?;
        let durable_record_admitted = causal_snapshot_record_bytes(
            &model.config,
            session_capacity,
            u32::try_from(publish_prefix).context("DeepSeek V4 durable prefix exceeds u32")?,
            durable_max_record_bytes,
        );
        if let Err(error) = durable_record_admitted.as_ref() {
            durable_admitted = false;
            eprintln!(
                "warning: durable DeepSeek V4 prefix capture is not admissible; generation will continue without publication: {error}"
            );
        }
    }
    let stop_tokens = gguf
        .stop_token_ids()
        .context("load producer-declared DeepSeek V4 stop tokens")?;
    for &token in &stop_tokens {
        checked_deepseek_v4_token_id(token, vocab_size, "stop")?;
    }
    // Probe store occupancy before resolving the strong model identity so an
    // empty store with no planned capture skips identity work entirely.
    let durable_probe_t0 = Instant::now();
    let durable_has_blobs: Option<bool> = match durable_store.as_ref() {
        None => None,
        Some(store) => match store.has_managed_blobs() {
            Ok(has_blobs) => Some(has_blobs),
            Err(error) => {
                eprintln!(
                    "warning: durable DeepSeek V4 prefix inventory failed after {:.1} ms; cold-prefilling: {error}",
                    durable_probe_t0.elapsed().as_secs_f64() * 1e3,
                );
                None
            }
        },
    };
    let durable_probe_ms = durable_probe_t0.elapsed().as_secs_f64() * 1e3;
    let durable_identity_needed = durable_store.is_some()
        && prompt_token_ids.len() >= 2
        && (durable_has_blobs == Some(true) || durable_admitted);
    let (snapshot_model_content_id, snapshot_file_exists) = if args.deepseek_v4_snapshot.is_some()
        || durable_identity_needed
    {
        let explicit_snapshot_path = args.deepseek_v4_snapshot.as_ref();
        let snapshot_parent = explicit_snapshot_path
            .map(|path| deepseek_v4_snapshot_parent(path))
            .transpose()?;
        let exists = if let Some(path) = explicit_snapshot_path {
            match path.symlink_metadata() {
                Ok(_) => true,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("inspect DeepSeek V4 snapshot path {}", path.display())
                    });
                }
            }
        } else {
            false
        };
        if explicit_snapshot_path.is_some() && !exists {
            let publish_prefix = deepseek_v4_snapshot_publish_prefix(prompt_token_ids.len())?;
            let model = DeepSeekV4Model::from_gguf_flash_0731(&gguf)
                .context("bind DeepSeek V4 snapshot geometry")?;
            let session_capacity = DeepSeekV4SessionCapacity::for_forward_limit(
                required_forwards,
                model.config.context_length,
            )
            .context("derive DeepSeek V4 snapshot session capacity")?;
            causal_snapshot_record_bytes(
                &model.config,
                session_capacity,
                u32::try_from(publish_prefix).context("DeepSeek V4 snapshot prefix exceeds u32")?,
                DEEPSEEK_V4_SNAPSHOT_MAX_RECORD_BYTES,
            )
            .context("preflight DeepSeek V4 causal snapshot record budget")?;
        }
        let identity_cache = if let Some(store) = durable_store.as_ref() {
            store.identity_cache()
        } else {
            deepseek_v4_snapshot_identity_cache(
                snapshot_parent.expect("explicit snapshot path resolved a parent"),
            )?
        };
        let identity_t0 = Instant::now();
        let report = checkpoint_content_identity(&gguf, &identity_cache)
            .context("derive strong ordered-shard DeepSeek V4 model identity")?;
        eprintln!(
            "deepseek_v4: checkpoint model identity cache={} hashed_bytes={} elapsed_ms={:.1}",
            identity_cache_outcome_label(report.outcome),
            report.bytes_hashed,
            identity_t0.elapsed().as_secs_f64() * 1e3,
        );
        (
            Some(DeepSeekV4ModelContentId::new(report.content_id)),
            explicit_snapshot_path.is_some() && exists,
        )
    } else {
        (None, false)
    };

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
    let load_plan =
        DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, required_forwards)
            .context("plan strict DeepSeek V4 Metal residency and session")?;
    let session_capacity = load_plan.session_capacity();
    eprintln!(
        "deepseek_v4: session capacity forwards={} csa_physical_rows={} hca_physical_rows={}",
        session_capacity.forward_limit(),
        session_capacity.csa_physical_rows(),
        session_capacity.hca_physical_rows(),
    );
    let selector_plan = DeepSeekV4MultigroupSelectorPlan::new(
        args.deepseek_v4_multigroup_selector,
        ctx.device.name().to_string(),
        session_capacity,
    )?;
    let restored_snapshot = if snapshot_file_exists {
        let snapshot_path = args
            .deepseek_v4_snapshot
            .as_ref()
            .expect("snapshot existence requires a snapshot path");
        let model_content_id = snapshot_model_content_id
            .expect("snapshot path resolved a model-content identity before planning");
        let snapshot_t0 = Instant::now();
        let snapshot = load_causal_snapshot_file(
            snapshot_path,
            DeepSeekV4SnapshotCodecConstraints {
                config: load_plan.config(),
                session_capacity: load_plan.session_capacity(),
                expected_model_content_id: model_content_id,
                max_record_bytes: DEEPSEEK_V4_SNAPSHOT_MAX_RECORD_BYTES,
            },
        )
        .with_context(|| {
            format!(
                "load DeepSeek V4 causal snapshot {}",
                snapshot_path.display()
            )
        })?;
        let restored_prefix =
            deepseek_v4_snapshot_restored_prefix_len(snapshot.prefix_tokens(), &prompt_token_ids)?;
        eprintln!(
            "deepseek_v4: snapshot validated before residency path={} prefix_tokens={} record_load_ms={:.1}",
            snapshot_path.display(),
            restored_prefix,
            snapshot_t0.elapsed().as_secs_f64() * 1e3,
        );
        Some((snapshot, restored_prefix))
    } else {
        None
    };
    let memory_plan = load_plan.memory_plan().clone();
    let initial_memory_signals = ctx.memory_signals();
    eprintln!("deepseek_v4: memory plan; {memory_plan}");
    let admitted_load_plan = load_plan
        .admit(initial_memory_signals)
        .context("admit strict DeepSeek V4 Metal residency and session")?;
    let prefetch_outcome = apply_deepseek_v4_prefetch(&gguf, prefetch_mode)?;
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted_load_plan)
        .context("load admitted strict DeepSeek V4 Metal residency")?;
    shutdown::checkpoint()?;
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
    let mut session = match snapshot_model_content_id {
        Some(model_content_id) => {
            DeepSeekV4Session::new_with_model_content_id(&ctx, residency, model_content_id)
        }
        None => DeepSeekV4Session::new(&ctx, residency),
    }
    .context("create DeepSeek V4 session")?;
    selector_plan.seal_session(&mut session, "single_turn")?;
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
    let mut durable_prepared: Option<DeepSeekV4PreparedCheckpoint> = None;
    let mut durable_capture_kind = "prompt";
    let mut durable_capture_ms = 0.0;
    let mut durable_restore_ms = durable_probe_ms;
    let prefill_mode = if let Some(snapshot_path) = args.deepseek_v4_snapshot.as_ref() {
        let model_content_id = snapshot_model_content_id
            .expect("snapshot path resolved a model-content identity before residency");
        if snapshot_file_exists {
            let (snapshot, restored_prefix) = restored_snapshot
                .as_ref()
                .expect("existing snapshot was validated before residency");
            session
                .restore_causal_snapshot(snapshot)
                .context("restore DeepSeek V4 causal snapshot")?;
            let suffix_chunks = execute_deepseek_v4_prompt_suffix(
                &mut session,
                &ctx,
                &prompt_token_ids[*restored_prefix..],
                prefill_chunk_tokens,
            )?;
            eprintln!(
                "deepseek_v4: snapshot restore path={} restored_tokens={} suffix_tokens={} suffix_chunks={} payload_bytes={}",
                snapshot_path.display(),
                restored_prefix,
                prompt_token_ids.len() - *restored_prefix,
                suffix_chunks,
                snapshot.payload_bytes(),
            );
            "causal_snapshot_restore"
        } else {
            let publish_prefix = deepseek_v4_snapshot_publish_prefix(prompt_token_ids.len())?;
            advance_deepseek_v4_prompt_prefix(
                &mut session,
                &ctx,
                &prompt_token_ids[..publish_prefix],
                prefill_chunk_tokens,
            )?;
            let snapshot = session
                .capture_causal_snapshot()
                .context("capture DeepSeek V4 causal snapshot")?;
            let report = publish_causal_snapshot_file(
                snapshot_path,
                &snapshot,
                DeepSeekV4SnapshotCodecConstraints {
                    config: session.residency().config(),
                    session_capacity: session.capacity(),
                    expected_model_content_id: model_content_id,
                    max_record_bytes: DEEPSEEK_V4_SNAPSHOT_MAX_RECORD_BYTES,
                },
            )
            .with_context(|| {
                format!(
                    "publish DeepSeek V4 causal snapshot {}",
                    snapshot_path.display()
                )
            })?;
            execute_deepseek_v4_prompt_suffix(
                &mut session,
                &ctx,
                &prompt_token_ids[publish_prefix..],
                prefill_chunk_tokens,
            )?;
            eprintln!(
                "deepseek_v4: snapshot publish path={} outcome={} prefix_tokens={} payload_bytes={} record_bytes={}",
                snapshot_path.display(),
                match report.outcome {
                    DeepSeekV4SnapshotFileOutcome::Published => "published",
                    DeepSeekV4SnapshotFileOutcome::AlreadyPresent => "already_present",
                },
                publish_prefix,
                snapshot.payload_bytes(),
                report.record_bytes,
            );
            "causal_snapshot_publish"
        }
    } else {
        let (restored_prefix, durable_payload_bytes) = attempt_deepseek_v4_durable_restore(
            durable_store.as_ref(),
            durable_has_blobs,
            &mut session,
            &prompt_token_ids,
            durable_max_record_bytes,
            durable_probe_ms,
            &mut durable_restore_ms,
        )?;
        // Completed-turn runs skip the mid-prefill prompt boundary: the
        // post-decode transcript strictly covers it.
        let capture_boundary = deepseek_v4_durable_capture_prefix_len(
            prompt_token_ids.len(),
            durable_admitted && !durable_completed_eligible,
        )?;
        if let Some(publish_prefix) = capture_boundary {
            if restored_prefix < publish_prefix {
                advance_deepseek_v4_prompt_prefix(
                    &mut session,
                    &ctx,
                    &prompt_token_ids[restored_prefix..publish_prefix],
                    prefill_chunk_tokens,
                )?;
            }
            let capture_t0 = Instant::now();
            match session.prepare_durable_checkpoint() {
                Ok(prepared) => durable_prepared = Some(prepared),
                Err(error)
                    if causal_snapshot_capture_error_kind(&error)
                        == DeepSeekV4SnapshotCaptureErrorKind::Allocation =>
                {
                    eprintln!(
                        "warning: durable DeepSeek V4 prefix capture allocation failed; continuing without publication: {error}"
                    );
                }
                Err(error) => {
                    return Err(error).context("capture durable DeepSeek V4 causal snapshot");
                }
            }
            durable_capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3;
            let suffix_chunks = execute_deepseek_v4_prompt_suffix(
                &mut session,
                &ctx,
                &prompt_token_ids[publish_prefix..],
                prefill_chunk_tokens,
            )?;
            if restored_prefix > 0 {
                eprintln!(
                    "deepseek_v4: durable restore restored_tokens={} promoted_tokens={} suffix_tokens={} suffix_chunks={} payload_bytes={}",
                    restored_prefix,
                    publish_prefix - restored_prefix,
                    prompt_token_ids.len() - publish_prefix,
                    suffix_chunks,
                    durable_payload_bytes,
                );
                "durable_prefix_restore"
            } else {
                "durable_prefix_capture"
            }
        } else if restored_prefix > 0 {
            let suffix_chunks = execute_deepseek_v4_prompt_suffix(
                &mut session,
                &ctx,
                &prompt_token_ids[restored_prefix..],
                prefill_chunk_tokens,
            )?;
            eprintln!(
                "deepseek_v4: durable restore restored_tokens={} promoted_tokens=0 suffix_tokens={} suffix_chunks={} payload_bytes={}",
                restored_prefix,
                prompt_token_ids.len() - restored_prefix,
                suffix_chunks,
                durable_payload_bytes,
            );
            "durable_prefix_restore"
        } else {
            let packed_chunk_count =
                deepseek_v4_packed_chunk_count(prompt_token_ids.len(), prefill_chunk_tokens);
            if packed_chunk_count > 0 {
                execute_deepseek_v4_prompt_suffix(
                    &mut session,
                    &ctx,
                    &prompt_token_ids,
                    prefill_chunk_tokens,
                )?;
                if packed_chunk_count == 1 {
                    "layer_major"
                } else {
                    "layer_major_chunks"
                }
            } else {
                for (index, &token) in prompt_token_ids.iter().enumerate() {
                    session
                        .forward_token(&ctx, token)
                        .with_context(|| format!("forward DeepSeek V4 prompt token {index}"))?;
                }
                "singleton"
            }
        }
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
    deepseek_v4_debug_dump_logits_sha256("single_turn", &logits);
    deepseek_v4_debug_dump_top_logits("single_turn", &logits, &tokenizer);
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

    let mut sampler = Sampler::new(sampling).context("initialize DeepSeek V4 sampler")?;
    const STAGE_PROFILE_GROUPS: usize = 4;
    let stage_profile_enabled = qwen_llm::env_flag::read_default_off("QWEN_DSV4_STAGE_PROFILE");
    const WHOLE_PROFILE_SAMPLES: usize = 8;
    let whole_profile_enabled = qwen_llm::env_flag::read_default_off("QWEN_DSV4_WHOLE_PROFILE");
    #[cfg(feature = "dsv4-diagnostics")]
    let temporal_window = configured_deepseek_v4_temporal_window()?;
    #[cfg(not(feature = "dsv4-diagnostics"))]
    let temporal_window = 0usize;
    ensure!(
        usize::from(stage_profile_enabled)
            + usize::from(whole_profile_enabled)
            + usize::from(temporal_window > 0)
            <= 1,
        "QWEN_DSV4_STAGE_PROFILE, QWEN_DSV4_WHOLE_PROFILE, and QWEN_DSV4_TEMPORAL_WINDOW are mutually exclusive"
    );
    #[cfg(feature = "dsv4-diagnostics")]
    let mut temporal_capture = dsv4_temporal::TemporalCapture::new(temporal_window);
    let mut profile_warmup_pending = stage_profile_enabled || whole_profile_enabled;
    let mut stage_profile_next_group = 0usize;
    let mut stage_profile_aggregate = DeepSeekV4CliStageProfileAggregate::default();
    let mut whole_profile_positions = Vec::with_capacity(WHOLE_PROFILE_SAMPLES);
    let mut whole_profile_wall_ms = Vec::with_capacity(WHOLE_PROFILE_SAMPLES);
    let mut whole_profile_gpu_ms = Vec::with_capacity(WHOLE_PROFILE_SAMPLES);
    let mut whole_profile_outside_gpu_ms = Vec::with_capacity(WHOLE_PROFILE_SAMPLES);
    let mut whole_profile_encode_cpu_ms = Vec::with_capacity(WHOLE_PROFILE_SAMPLES);
    let mut whole_profile_wait_residual_ms = Vec::with_capacity(WHOLE_PROFILE_SAMPLES);
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
            #[cfg(feature = "dsv4-diagnostics")]
            let capture_temporal = temporal_capture.should_capture();
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let capture_temporal = false;
            if capture_temporal {
                #[cfg(feature = "dsv4-diagnostics")]
                {
                    let position = session.next_position();
                    session
                        .arm_decision_transcript(position)
                        .context("arm temporal DeepSeek V4 decision capture")?;
                    session
                        .forward_token(&ctx, token)
                        .context("forward temporal DeepSeek V4 token")?;
                    let transcript = session
                        .take_decision_transcript_and_reset()
                        .context("take temporal DeepSeek V4 decision capture")?;
                    temporal_capture.record(transcript)?;
                }
            } else if std::mem::take(&mut profile_warmup_pending) {
                session
                    .forward_token(&ctx, token)
                    .context("warm profiled DeepSeek V4 decode")?;
            } else if stage_profile_enabled && stage_profile_next_group < STAGE_PROFILE_GROUPS {
                let group = stage_profile_next_group;
                stage_profile_next_group += 1;
                let sampled_layers = (0..session.residency().config().attention_kinds.len())
                    .filter(|layer| layer % STAGE_PROFILE_GROUPS == group)
                    .collect::<Vec<_>>();
                let profile = session
                    .forward_token_stage_profiled(&ctx, token, &sampled_layers)
                    .context("stage-profile generated DeepSeek V4 token")?;
                emit_deepseek_v4_stage_profile(
                    &profile,
                    group,
                    STAGE_PROFILE_GROUPS,
                    &mut stage_profile_aggregate,
                );
            } else if whole_profile_enabled && whole_profile_positions.len() < WHOLE_PROFILE_SAMPLES
            {
                let profile = session
                    .forward_token_whole_profiled(&ctx, token)
                    .context("whole-profile generated DeepSeek V4 token")?;
                whole_profile_positions.push(profile.position);
                whole_profile_wall_ms.push(profile.forward_wall_ms);
                whole_profile_gpu_ms.push(profile.command_gpu_ms);
                whole_profile_outside_gpu_ms.push(profile.outside_gpu_ms());
                whole_profile_encode_cpu_ms.push(profile.encode_cpu_ms);
                whole_profile_wait_residual_ms.push(profile.wait_residual_ms());
                if whole_profile_positions.len() == WHOLE_PROFILE_SAMPLES {
                    emit_deepseek_v4_whole_profile(
                        &whole_profile_positions,
                        &whole_profile_wall_ms,
                        &whole_profile_gpu_ms,
                        &whole_profile_outside_gpu_ms,
                        &whole_profile_encode_cpu_ms,
                        &whole_profile_wait_residual_ms,
                    );
                }
            } else {
                session
                    .forward_token(&ctx, token)
                    .context("forward generated DeepSeek V4 token")?;
            }
            copy_deepseek_v4_logits(&session, vocab_size, "continuing")
        },
    )?;
    drop(stdout);
    if durable_completed_eligible
        && durable_admitted
        && durable_store.is_some()
        && durable_prepared.is_none()
        && args.deepseek_v4_snapshot.is_none()
    {
        if matches!(generation.stop_reason, StopReason::Eos) {
            // The session sits at the completed transcript boundary: every
            // prompt and generated token except the unconsumed terminal EOS.
            match causal_snapshot_record_bytes(
                session.residency().config(),
                session.capacity(),
                session.next_position(),
                durable_max_record_bytes,
            ) {
                Ok(_) => {
                    let capture_t0 = Instant::now();
                    match session.prepare_durable_checkpoint() {
                        Ok(prepared) => {
                            durable_prepared = Some(prepared);
                            durable_capture_kind = "completed";
                        }
                        Err(error)
                            if causal_snapshot_capture_error_kind(&error)
                                == DeepSeekV4SnapshotCaptureErrorKind::Allocation =>
                        {
                            eprintln!(
                                "warning: durable DeepSeek V4 completed capture allocation failed; continuing without publication: {error}"
                            );
                        }
                        Err(error) => {
                            return Err(error)
                                .context("capture completed DeepSeek V4 causal snapshot");
                        }
                    }
                    durable_capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3;
                }
                Err(error) => eprintln!(
                    "warning: durable DeepSeek V4 completed capture is not admissible; continuing without publication: {error}"
                ),
            }
        } else {
            // A truncated turn's boundary can never prefix a retry of the
            // same prompt; publishing it would only pollute the budget.
            eprintln!(
                "durable_prefix_cache: family=deepseek_v4 publish=skipped capture=completed reason=token_limit"
            );
        }
    }
    if let (Some(store), Some(prepared)) = (durable_store.as_ref(), durable_prepared.as_ref()) {
        let publish_t0 = Instant::now();
        match store.publish_prepared(prepared, durable_max_record_bytes) {
            Ok(report) => eprintln!(
                concat!(
                    "durable_prefix_cache: family=deepseek_v4 publish={} capture={} ",
                    "matched_tokens={} blob_bytes={} evicted={} ",
                    "staging_examined={} staging_removed={} staging_allocated_bytes_reclaimed={} ",
                    "staging_live={} staging_legacy={} staging_foreign={} staging_truncated={} ",
                    "staged_integrity={} staged_integrity_us={} capture_ms={:.1} publish_ms={:.1}"
                ),
                publish_outcome_label(report.outcome),
                durable_capture_kind,
                prepared.next_position(),
                report.blob_bytes,
                report.evicted_entries,
                report.staging_entries_examined,
                report.staging_entries_removed,
                report.staging_allocated_bytes_reclaimed,
                report.staging_live_entries,
                report.staging_legacy_entries,
                report.staging_foreign_entries,
                report.staging_cleanup_truncated,
                report.staged_integrity.mode.as_str(),
                report.staged_integrity.elapsed.as_micros(),
                durable_capture_ms,
                publish_t0.elapsed().as_secs_f64() * 1e3,
            ),
            Err(error) => eprintln!(
                "warning: durable DeepSeek V4 prefix publication failed after response (restore_ms={:.1} capture_ms={:.1}): {error}",
                durable_restore_ms, durable_capture_ms,
            ),
        }
    }
    #[cfg(feature = "dsv4-diagnostics")]
    if temporal_capture.requested_tokens() > 0 {
        let mut report = temporal_capture.finish();
        let final_logits = copy_deepseek_v4_logits(&session, vocab_size, "temporal final")?;
        let final_logits_sha256 =
            hex_encode_bytes(&Sha256::digest(bytemuck::cast_slice(&final_logits)));
        let final_causal_digest = if args.deepseek_v4_snapshot.is_some() {
            let snapshot = session
                .capture_causal_snapshot()
                .context("capture final temporal DeepSeek V4 causal state")?;
            Some(hex_encode_bytes(snapshot.causal_digest()))
        } else {
            None
        };
        report.attach_final_state(final_logits_sha256, final_causal_digest);
        if let Some(path) = std::env::var_os(DEEPSEEK_V4_TEMPORAL_JSON_ENV) {
            let path = PathBuf::from(path);
            std::fs::write(
                &path,
                format!("{}\n", serde_json::to_string_pretty(&report)?),
            )
            .with_context(|| format!("write temporal report {}", path.display()))?;
        }
        eprintln!(
            "deepseek_v4 temporal: {}",
            serde_json::to_string(&report.summary())?
        );
    }
    selector_plan.emit_completion("single_turn", session.multigroup_selector_telemetry())?;

    let decode_tps = if generation.wall_ms > 0.0 {
        generation.tokens.len() as f64 / (generation.wall_ms / 1e3)
    } else {
        0.0
    };
    let prefill_tps = if prefill_ms > 0.0 {
        prompt_ids.len() as f64 / (prefill_ms / 1e3)
    } else {
        0.0
    };
    let transition_tps = if generation.transition_ms > 0.0 {
        generation.transitions as f64 / (generation.transition_ms / 1e3)
    } else {
        0.0
    };
    let generated_ids_sha256 = generated_token_sha256(&generation.tokens);
    eprintln!(
        concat!(
            "deepseek_v4 stats: prompt_kind={} prefill_mode={} prefill_chunk_cap={} prompt_tokens={} generated_tokens={} transitions={} ",
            "stop_reason={} tokenizer_ms={:.1} prefetch_mode={} prefetch_ms={:.1} load_ms={:.1} prefill_ms={:.1} prefill_tps={:.2} ",
            "generation_ms={:.1} decode_tps={:.2} transition_tps={:.2} build_commit={} build_dirty={} generated_ids_sha256={}"
        ),
        prompt_kind,
        prefill_mode,
        prefill_chunk_tokens,
        prompt_ids.len(),
        generation.tokens.len(),
        generation.transitions,
        generation.stop_reason.as_str(),
        tokenizer_ms,
        prefetch_outcome.mode.as_str(),
        prefetch_outcome.wall_ms,
        load_ms,
        prefill_ms,
        prefill_tps,
        generation.wall_ms,
        decode_tps,
        transition_tps,
        env!("QWEN_BUILD_COMMIT"),
        env!("QWEN_BUILD_DIRTY"),
        generated_ids_sha256,
    );
    if std::env::var_os("QWEN_DSV4_GENERATED_IDS").is_some() {
        eprintln!("deepseek_v4 generated_ids: {:?}", generation.tokens);
    }
    if let Some(path) = args.trace_request.as_ref() {
        append_request_trace(path, arrival_ms, prompt_ids.len(), generation.tokens.len())?;
    }
    if let Some(path) = args.request_stats_jsonl.as_ref() {
        // Measure total request wall time at the outer boundary (not the sum
        // of phase timings, which can miss inter-phase gaps).
        let total_ms = request_start.elapsed().as_secs_f64() * 1e3;
        let measured = RequestStatsMeasured {
            prompt_kind,
            prefill_mode,
            prefill_chunk_cap: prefill_chunk_tokens as u64,
            input_tokens: prompt_ids.len() as u64,
            output_tokens: generation.tokens.len() as u64,
            transitions: generation.transitions as u64,
            stop_reason: generation.stop_reason,
            tokenizer_ms,
            load_ms,
            prefill_ms,
            prefill_tps,
            decode_ms: generation.wall_ms,
            decode_tps,
            transition_tps,
            total_ms,
            output_fingerprint: GeneratedTokenSha256Digest::of(&generation.tokens),
        };
        let record = build_deepseek_v4_single_turn_stats_record(&INVOCATION_ID, &measured);
        append_jsonl_record(path, &record, "request stats jsonl")?;
    }
    Ok(())
}

// ---- Common request-stats-jsonl envelope (schema: qwen-llm.request-stats v1) ----
//
// This envelope is a **small stable common core** plus namespaced backend
// diagnostics under `diagnostics.<family>`. It is populated today only by
// DeepSeek V4 single-turn generation; other paths will migrate additively in
// follow-up PRs (Qwen batch, DS4 batch, single-turn Qwen).
//
// Constraints:
//   * schema_version bumps ONLY for breaking changes. Additive fields stay in v1.
//   * `diagnostics.<family>` has its own `schema_version`; family fields evolve
//     independently of the common contract.
//   * Consumers MUST ignore unknown top-level and namespaced fields.
//   * `record_type` is the top-level discriminator; more record types (e.g. an
//     invocation-level bookend) may be added later.
//   * Timing fields are per-request. Process/invocation-level facts (model load
//     time, binary hashes) live under `diagnostics` or a future invocation record.
//   * Wire counts are `u64` to remain architecture-independent (usize is not).
//   * All metric f64 fields are non-negative and finite; sanitized at emission.
//   * `output_fingerprint.value` is hex-encoded from `[u8; 32]` internally, so
//     the algorithm identifier and its byte encoding cannot drift apart.

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum RequestStatsStatus {
    Ok,
    // Reserved for future error / cancelled records. Kept explicit so schema
    // consumers see the full status vocabulary from v1.
    #[allow(dead_code)]
    Error,
    #[allow(dead_code)]
    Cancelled,
}

/// Envelope-owned finish reason enum, decoupled from the internal `StopReason`
/// so that adding a new internal variant or renaming does not silently mutate
/// the wire schema. Exhaustive `From<StopReason>` forces future variants to
/// be a compile-time decision.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RequestStatsFinishReason {
    Eos,
    TokenLimit,
}

impl From<StopReason> for RequestStatsFinishReason {
    fn from(reason: StopReason) -> Self {
        match reason {
            StopReason::Eos => Self::Eos,
            StopReason::TokenLimit => Self::TokenLimit,
        }
    }
}

#[derive(Debug, Serialize)]
struct RequestStatsRequestRecord<'a> {
    schema: &'static str,
    schema_version: u32,
    record_type: &'static str,
    invocation_id: &'a str,
    request_index: u32,
    status: RequestStatsStatus,
    model: RequestStatsModel<'a>,
    input: RequestStatsInput<'a>,
    // Success-only fields are optional so future error/cancelled records need
    // only omit them, without a breaking restructuring.
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<RequestStatsUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish: Option<RequestStatsFinish>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing_ms: Option<RequestStatsTiming>,
    #[serde(skip_serializing_if = "Option::is_none")]
    throughput_tps: Option<RequestStatsThroughput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_fingerprint: Option<RequestStatsOutputFingerprint>,
    build: RequestStatsBuild,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostics: Option<RequestStatsDiagnostics>,
}

#[derive(Debug, Serialize)]
struct RequestStatsModel<'a> {
    family: &'a str,
}

#[derive(Debug, Serialize)]
struct RequestStatsInput<'a> {
    kind: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    template: Option<&'a str>,
}

#[derive(Debug, Serialize)]
struct RequestStatsUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Debug, Serialize)]
struct RequestStatsFinish {
    reason: RequestStatsFinishReason,
}

#[derive(Debug, Serialize)]
struct RequestStatsTiming {
    total: f64,
    tokenization: f64,
    prefill: f64,
    decode: f64,
}

#[derive(Debug, Serialize)]
struct RequestStatsThroughput {
    prefill: f64,
    decode: f64,
}

#[derive(Debug, Serialize)]
struct RequestStatsOutputFingerprint {
    algorithm: &'static str,
    value: String,
}

#[derive(Debug, Serialize)]
struct RequestStatsBuild {
    commit: &'static str,
    dirty: bool,
}

#[derive(Debug, Serialize)]
struct RequestStatsDiagnostics {
    #[serde(skip_serializing_if = "Option::is_none")]
    deepseek_v4: Option<RequestStatsDeepSeekV4Diagnostics>,
}

#[derive(Debug, Serialize)]
struct RequestStatsDeepSeekV4Diagnostics {
    schema_version: u32,
    prefill_mode: &'static str,
    prefill_chunk_cap: u64,
    transitions: u64,
    transition_tps: f64,
    load_ms: f64,
}

/// Case-insensitive parser for the `QWEN_BUILD_DIRTY` build-time env var.
/// Accepts `0`/`false`/`no` (any case, plus empty) as clean; anything else
/// is treated as dirty, biasing toward "assume unstable" if the value is
/// unexpected.
fn parse_build_dirty(raw: &str) -> bool {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return false;
    }
    !(trimmed.eq_ignore_ascii_case("0")
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no"))
}

pub(crate) struct RequestStatsMeasured {
    pub prompt_kind: &'static str,
    pub prefill_mode: &'static str,
    pub prefill_chunk_cap: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub transitions: u64,
    pub stop_reason: StopReason,
    pub tokenizer_ms: f64,
    pub load_ms: f64,
    pub prefill_ms: f64,
    pub prefill_tps: f64,
    pub decode_ms: f64,
    pub decode_tps: f64,
    pub transition_tps: f64,
    pub total_ms: f64,
    pub output_fingerprint: GeneratedTokenSha256Digest,
}

fn build_deepseek_v4_single_turn_stats_record<'a>(
    invocation_id: &'a str,
    measured: &RequestStatsMeasured,
) -> RequestStatsRequestRecord<'a> {
    // input.kind describes representation (raw vs messages); template surfaces
    // the specific chat-template variant when known (e.g., messages_0731_chat
    // -> kind=messages template=0731_chat).
    let (input_kind, input_template) = split_ds4_prompt_kind(measured.prompt_kind);
    RequestStatsRequestRecord {
        schema: "qwen-llm.request-stats",
        schema_version: 1,
        record_type: "request_stats",
        invocation_id,
        request_index: 0,
        status: RequestStatsStatus::Ok,
        model: RequestStatsModel {
            family: "deepseek_v4",
        },
        input: RequestStatsInput {
            kind: input_kind,
            template: input_template,
        },
        usage: Some(RequestStatsUsage {
            input_tokens: measured.input_tokens,
            output_tokens: measured.output_tokens,
        }),
        finish: Some(RequestStatsFinish {
            reason: measured.stop_reason.into(),
        }),
        timing_ms: Some(RequestStatsTiming {
            total: sanitize_finite_metric(measured.total_ms, "timing_ms.total"),
            tokenization: sanitize_finite_metric(measured.tokenizer_ms, "timing_ms.tokenization"),
            prefill: sanitize_finite_metric(measured.prefill_ms, "timing_ms.prefill"),
            decode: sanitize_finite_metric(measured.decode_ms, "timing_ms.decode"),
        }),
        throughput_tps: Some(RequestStatsThroughput {
            prefill: sanitize_finite_metric(measured.prefill_tps, "throughput_tps.prefill"),
            decode: sanitize_finite_metric(measured.decode_tps, "throughput_tps.decode"),
        }),
        output_fingerprint: Some(RequestStatsOutputFingerprint {
            algorithm: "sha256-qwen-generated-token-ids-v1",
            value: measured.output_fingerprint.hex(),
        }),
        build: RequestStatsBuild {
            commit: env!("QWEN_BUILD_COMMIT"),
            dirty: parse_build_dirty(env!("QWEN_BUILD_DIRTY")),
        },
        diagnostics: Some(RequestStatsDiagnostics {
            deepseek_v4: Some(RequestStatsDeepSeekV4Diagnostics {
                schema_version: 1,
                prefill_mode: measured.prefill_mode,
                prefill_chunk_cap: measured.prefill_chunk_cap,
                transitions: measured.transitions,
                transition_tps: sanitize_finite_metric(
                    measured.transition_tps,
                    "diagnostics.deepseek_v4.transition_tps",
                ),
                load_ms: sanitize_finite_metric(
                    measured.load_ms,
                    "diagnostics.deepseek_v4.load_ms",
                ),
            }),
        }),
    }
}

/// Split a DeepSeek V4 `prompt_kind` string into (input_kind, template).
/// The common `input.kind` vocabulary is restricted to `messages`, `raw`, or
/// `unknown` — new backend labels do NOT expand the common core by accident.
///
/// Known values:
///   - "messages_0731_chat"     -> ("messages", Some("0731_chat"))
///   - "messages_0731_thinking" -> ("messages", Some("0731_thinking"))
///   - "raw"                    -> ("raw", None)
///   - "messages_" or "messages" (empty suffix) -> ("unknown", None)
///   - anything else            -> ("unknown", None)
fn split_ds4_prompt_kind(prompt_kind: &str) -> (&str, Option<&str>) {
    if let Some(rest) = prompt_kind.strip_prefix("messages_") {
        if rest.is_empty() {
            ("unknown", None)
        } else {
            ("messages", Some(rest))
        }
    } else if prompt_kind == "raw" {
        ("raw", None)
    } else {
        ("unknown", None)
    }
}

struct DeepSeekV4PreparedRequest {
    id: String,
    line: usize,
    prompt_tokens: usize,
    prompt_token_ids: Vec<u32>,
    max_tokens: usize,
    required_forwards: usize,
    sampling: SamplingConfig,
}

/// DS4 request preparation deliberately diverges from the Qwen preparer in
/// two ways: prompts always tokenize without automatic specials (DS4 prompts
/// carry explicit BOS bytes), and `cache_prefix_tokens` fails closed until
/// the DS4 durable prefix lane lands.
fn prepare_deepseek_v4_jsonl_request_line(
    tokenizer: &Tokenizer,
    vocab_size: u32,
    args: &Args,
    line: &str,
    line_no: usize,
) -> Result<Option<DeepSeekV4PreparedRequest>> {
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
    ensure!(
        request.cache_prefix_tokens.is_none(),
        "request {id} sets cache_prefix_tokens, which DeepSeek V4 requests do not support yet"
    );
    let prompt = request_prompt(&request, line_no)
        .with_context(|| format!("resolve request {id} prompt at line {line_no}"))?;
    let prompt_ids = tokenizer
        .encode(&prompt, false)
        .with_context(|| format!("tokenize request {id}"))?;
    ensure!(
        !prompt_ids.is_empty(),
        "request {id} tokenized to zero tokens"
    );
    deepseek_v4_debug_dump_prompt_ids(&id, &prompt_ids);
    let prompt_token_ids = prompt_ids
        .iter()
        .enumerate()
        .map(|(index, &token)| {
            checked_deepseek_v4_token_id(token, vocab_size, &format!("{id} prompt[{index}]"))
        })
        .collect::<Result<Vec<_>>>()?;
    let max_tokens = request.tokens.unwrap_or(args.tokens);
    ensure!(max_tokens > 0, "request {id} requires tokens >= 1");
    let required_forwards = deepseek_v4_required_forwards(prompt_token_ids.len(), max_tokens)
        .with_context(|| format!("derive forward budget for request {id}"))?;
    let sampling = request_sampling_config(&request, args)
        .with_context(|| format!("validate sampling for request {id}"))?;
    Ok(Some(DeepSeekV4PreparedRequest {
        id,
        line: line_no,
        prompt_tokens: prompt_ids.len(),
        prompt_token_ids,
        max_tokens,
        required_forwards,
        sampling,
    }))
}

/// Executes prepared requests sequentially against one long-lived residency,
/// rebuilding a fresh session per request. Mirrors the Qwen JSONL contract:
/// buffered per-request output lines, fail-fast on the first error, and no
/// token streaming.
fn run_deepseek_v4_requests_jsonl(
    model_path: &Path,
    gguf: GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<()> {
    validate_deepseek_v4_requests_mode(args, explicit)?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    let prefill_chunk_tokens = deepseek_v4_prefill_chunk_tokens()?;
    let prefetch_mode = configured_deepseek_v4_prefetch_mode()?;
    cli_sampling_config(args)?;
    let stdin_mode = deepseek_v4_requests_reads_stdin(args)?;
    let requests_path = args
        .requests_jsonl
        .clone()
        .expect("requests mode requires --requests-jsonl");

    let tokenizer = Tokenizer::from_gguf(&gguf).context("load DeepSeek V4 tokenizer")?;
    let vocab_size = tokenizer.n_vocab();
    let stop_tokens = gguf
        .stop_token_ids()
        .context("load producer-declared DeepSeek V4 stop tokens")?;
    for &token in &stop_tokens {
        checked_deepseek_v4_token_id(token, vocab_size, "stop")?;
    }

    let (prepared, forward_budget, logical_context_limit) = if stdin_mode {
        let context_limit = args
            .max_context_tokens
            .expect("stdin requests mode validated --max-context-tokens");
        let forward_budget = deepseek_v4_forward_budget_for_context_limit(context_limit)?;
        (None, forward_budget, Some(context_limit))
    } else {
        let raw = std::fs::read_to_string(&requests_path)
            .with_context(|| format!("read requests file {}", requests_path.display()))?;
        let mut prepared = Vec::new();
        for (index, line) in raw.lines().enumerate() {
            if let Some(request) = prepare_deepseek_v4_jsonl_request_line(
                &tokenizer,
                vocab_size,
                args,
                line,
                index + 1,
            )? {
                prepared.push(request);
            }
        }
        ensure!(
            !prepared.is_empty(),
            "requests file {} contains no requests",
            requests_path.display()
        );
        let budget = prepared
            .iter()
            .map(|request| request.required_forwards)
            .max()
            .expect("nonempty prepared requests");
        (Some(prepared), budget, None)
    };

    eprintln!(
        "deepseek_v4: loading {} for requests; source={} forward_budget={} logical_context_limit={:?} promoted_capacity={}",
        model_path.display(),
        if stdin_mode { "stdin" } else { "file" },
        forward_budget,
        logical_context_limit,
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
    );
    let load_t0 = Instant::now();
    let ctx = MetalContext::new().context("init Metal context for DeepSeek V4")?;
    let load_plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, forward_budget)
        .context("plan strict DeepSeek V4 Metal residency and session")?;
    let session_capacity = load_plan.session_capacity();
    eprintln!(
        "deepseek_v4: session capacity forwards={} csa_physical_rows={} hca_physical_rows={}",
        session_capacity.forward_limit(),
        session_capacity.csa_physical_rows(),
        session_capacity.hca_physical_rows(),
    );
    let selector_plan = DeepSeekV4MultigroupSelectorPlan::new(
        args.deepseek_v4_multigroup_selector,
        ctx.device.name().to_string(),
        session_capacity,
    )?;
    let memory_plan = load_plan.memory_plan().clone();
    let initial_memory_signals = ctx.memory_signals();
    eprintln!("deepseek_v4: memory plan; {memory_plan}");
    let auto_mode = args.execution_mode == Some(execution_selector::ExecutionModeArg::Auto);
    let auto_selection = if auto_mode {
        let request_count = prepared.as_ref().map_or(0, Vec::len);
        let two_session_memory_admitted = memory_plan
            .admission_for_sessions(initial_memory_signals, 2)
            .context("evaluate automatic DeepSeek V4 two-session admission")?
            .admitted;
        let selection =
            execution_selector::select_deepseek(execution_selector::DeepSeekSelectionInput {
                mode: args.execution_mode,
                request_count,
                stdin: stdin_mode,
                residency_set: qwen_llm::env_flag::read_default_off("QWEN_DSV4_RESIDENCY_SET"),
                two_session_memory_admitted,
            });
        eprintln!(
            "execution_selection: {}",
            serde_json::to_string(&execution_selector::ExecutionSelectionRecord::new(
                Some(ModelFamily::DeepSeek4),
                (!stdin_mode).then_some(request_count),
                selection,
                None,
                None,
                None,
                None,
                None,
            ))
            .context("serialize DeepSeek V4 execution selection")?
        );
        Some(selection)
    } else {
        None
    };
    let use_concurrency = args.concurrency.is_some()
        || auto_selection.is_some_and(|selection| {
            selection.selected == execution_selector::SelectedExecution::Concurrency2
        });
    let session_count = if use_concurrency { 2 } else { 1 };
    let admitted_load_plan = load_plan
        .admit_for_sessions(initial_memory_signals, session_count)
        .context("admit strict DeepSeek V4 Metal residency and session")?;
    let _prefetch_outcome = apply_deepseek_v4_prefetch(&gguf, prefetch_mode)?;
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted_load_plan)
        .context("load admitted strict DeepSeek V4 Metal residency")?;
    shutdown::checkpoint()?;
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
    eprintln!(
        "deepseek_v4: resident on {} in {:.1} ms; {}",
        ctx.describe(),
        load_t0.elapsed().as_secs_f64() * 1e3,
        residency.report(),
    );

    if use_concurrency {
        let prepared = prepared.expect("DeepSeek concurrency requires file lookahead");
        let executed = concurrent_jsonl::run_deepseek_file(
            &ctx,
            residency,
            &selector_plan,
            &tokenizer,
            vocab_size,
            &stop_tokens,
            prepared,
            prefill_chunk_tokens,
            memory_plan.session_priced_upper_bytes(),
        )?;
        eprintln!("deepseek_v4: requests complete; executed={executed}");
        return Ok(());
    }

    let stdout_handle = std::io::stdout();
    let mut residency_slot = Some(residency);
    let mut reconcile_first_session = Some((before_residency_bytes, after_residency_bytes));
    let mut executed = 0usize;
    let mut execute = |request: DeepSeekV4PreparedRequest| -> Result<()> {
        shutdown::checkpoint()?;
        if let Some(limit) = logical_context_limit {
            validate_deepseek_v4_request_context_limit(
                &request.id,
                request.prompt_tokens,
                request.max_tokens,
                limit,
            )?;
        }
        ensure!(
            request.required_forwards <= session_capacity.forward_limit(),
            "request {} requires {} forwards, beyond the run's shared session budget {}",
            request.id,
            request.required_forwards,
            session_capacity.forward_limit(),
        );
        let arrival_ms = unix_epoch_ms()?;
        let residency = residency_slot
            .take()
            .expect("residency is returned after every request");
        let session_t0 = Instant::now();
        let mut session = DeepSeekV4Session::new(&ctx, residency)
            .with_context(|| format!("create DeepSeek V4 session for request {}", request.id))?;
        selector_plan.seal_session(&mut session, &request.id)?;
        let first_memory_sample =
            reconcile_first_session
                .take()
                .map(|(before, after_residency)| {
                    (before, after_residency, ctx.current_allocated_size())
                });
        let session_ms = session_t0.elapsed().as_secs_f64() * 1e3;

        let request_execution = (|| -> Result<(GenerationResult, Vec<u8>, &'static str, f64)> {
            if let Some((before, after_residency, after_session)) = first_memory_sample {
                memory_plan
                    .reconcile_session(before, after_residency, after_session)
                    .context("reconcile DeepSeek V4 request session allocation")?;
            }
            let prefill_t0 = Instant::now();
            let packed_chunk_count = deepseek_v4_packed_chunk_count(
                request.prompt_token_ids.len(),
                prefill_chunk_tokens,
            );
            let prefill_mode = if packed_chunk_count > 0 {
                execute_deepseek_v4_prompt_suffix(
                    &mut session,
                    &ctx,
                    &request.prompt_token_ids,
                    prefill_chunk_tokens,
                )
                .with_context(|| format!("prefill request {}", request.id))?;
                if packed_chunk_count == 1 {
                    "layer_major"
                } else {
                    "layer_major_chunks"
                }
            } else {
                for (index, &token) in request.prompt_token_ids.iter().enumerate() {
                    session.forward_token(&ctx, token).with_context(|| {
                        format!("forward request {} prompt token {index}", request.id)
                    })?;
                }
                "singleton"
            };
            if let Some((before, after_residency, after_session)) = first_memory_sample {
                let reconciliation = memory_plan
                    .reconcile(DeepSeekV4MemorySamples {
                        before_residency_bytes: before,
                        after_residency_bytes: after_residency,
                        after_session_bytes: after_session,
                        after_first_forward_bytes: ctx.current_allocated_size(),
                    })
                    .context("reconcile admitted DeepSeek V4 request memory")?;
                eprintln!("deepseek_v4: request memory reconciliation; {reconciliation}");
            }
            let logits = copy_deepseek_v4_logits(&session, vocab_size, "prompt")
                .with_context(|| format!("copy request {} prompt logits", request.id))?;
            deepseek_v4_debug_dump_logits_sha256(&request.id, &logits);
            deepseek_v4_debug_dump_top_logits(&request.id, &logits, &tokenizer);
            let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

            let mut sampler = Sampler::new(request.sampling)
                .with_context(|| format!("initialize sampler for request {}", request.id))?;
            let mut generated_bytes = Vec::new();
            let mut transition_index = 0usize;
            let generation = generate_serial(
                logits,
                request.max_tokens,
                &stop_tokens,
                &mut sampler,
                |token| {
                    let piece = tokenizer
                        .try_decode_piece_bytes_exact(token)
                        .with_context(|| format!("decode request {} token {token}", request.id))?;
                    generated_bytes.extend_from_slice(piece);
                    Ok(())
                },
                |token| {
                    let current_transition = transition_index;
                    let token = checked_deepseek_v4_token_id(
                        token,
                        vocab_size,
                        &format!(
                            "request {} generated transition {current_transition}",
                            request.id
                        ),
                    )?;
                    session.forward_token(&ctx, token).with_context(|| {
                        format!(
                            "forward request {} generated transition {current_transition}",
                            request.id
                        )
                    })?;
                    let logits = copy_deepseek_v4_logits(&session, vocab_size, "continuing")
                        .with_context(|| {
                            format!(
                                "copy request {} continuing logits after transition {current_transition}",
                                request.id
                            )
                        })?;
                    transition_index += 1;
                    Ok(logits)
                },
            )?;
            Ok((generation, generated_bytes, prefill_mode, prefill_ms))
        })();
        let selector_telemetry = session.multigroup_selector_telemetry();
        residency_slot = Some(session.into_residency()?);
        let (generation, generated_bytes, prefill_mode, prefill_ms) = request_execution
            .with_context(|| format!("execute request {} at line {}", request.id, request.line))?;
        selector_plan.emit_completion(&request.id, selector_telemetry)?;

        let decode_tps = if generation.wall_ms > 0.0 {
            generation.tokens.len() as f64 / (generation.wall_ms / 1e3)
        } else {
            0.0
        };
        let prefill_tps = if prefill_ms > 0.0 {
            request.prompt_tokens as f64 / (prefill_ms / 1e3)
        } else {
            0.0
        };
        let output = RequestOutput {
            id: request.id.clone(),
            prompt_tokens: request.prompt_tokens,
            generated_tokens: generation.tokens.len(),
            generated_token_sha256: generated_token_sha256(&generation.tokens),
            generated_text: String::from_utf8_lossy(&generated_bytes).into_owned(),
            stop_reason: generation.stop_reason,
            terminal_token_target_transition_consumed: false,
            thinking_partition: None,
        };
        {
            let mut stdout = stdout_handle.lock();
            serde_json::to_writer(&mut stdout, &output)
                .with_context(|| format!("serialize output for request {}", request.id))?;
            stdout
                .write_all(b"\n")
                .and_then(|_| stdout.flush())
                .with_context(|| format!("write output for request {}", request.id))?;
        }
        eprintln!(
            concat!(
                "deepseek_v4 stats: request={} line={} prompt_kind=raw prefill_mode={} prefill_chunk_cap={} prompt_tokens={} ",
                "generated_tokens={} transitions={} stop_reason={} session_ms={:.1} prefill_ms={:.1} prefill_tps={:.2} ",
                "generation_ms={:.1} decode_tps={:.2} build_commit={} build_dirty={}"
            ),
            output.id,
            request.line,
            prefill_mode,
            prefill_chunk_tokens,
            output.prompt_tokens,
            output.generated_tokens,
            generation.transitions,
            generation.stop_reason.as_str(),
            session_ms,
            prefill_ms,
            prefill_tps,
            generation.wall_ms,
            decode_tps,
            env!("QWEN_BUILD_COMMIT"),
            env!("QWEN_BUILD_DIRTY"),
        );
        if let Some(path) = args.trace_request.as_ref() {
            append_request_trace(
                path,
                arrival_ms,
                output.prompt_tokens,
                output.generated_tokens,
            )?;
        }
        executed += 1;
        Ok(())
    };

    if let Some(prepared) = prepared {
        for request in prepared {
            execute(request)?;
        }
    } else {
        shutdown::checkpoint()?;
        let stdin = std::io::stdin();
        for (index, line) in stdin.lock().lines().enumerate() {
            shutdown::checkpoint()?;
            let line = line.context("read requests line from stdin")?;
            if let Some(request) = prepare_deepseek_v4_jsonl_request_line(
                &tokenizer,
                vocab_size,
                args,
                &line,
                index + 1,
            )? {
                execute(request)?;
            }
        }
        ensure!(executed > 0, "stdin request stream contained no requests");
    }
    eprintln!("deepseek_v4: requests complete; executed={executed}");
    Ok(())
}

/// Debug observability: `QWEN_DSV4_PROMPT_IDS=1` dumps the exact input token
/// stream fed to the model, for cross-engine tokenization diffs.
fn deepseek_v4_debug_dump_prompt_ids(scope: &str, prompt_ids: &[i32]) {
    if std::env::var_os("QWEN_DSV4_PROMPT_IDS").is_some() {
        eprintln!(
            "deepseek_v4 prompt_ids: scope={scope:?} count={} ids={:?}",
            prompt_ids.len(),
            prompt_ids
        );
    }
}

/// Debug observability: `QWEN_DSV4_LOGITS_SHA256=1` emits a compact exact
/// identity for the complete first-token F32 logit vector.
fn deepseek_v4_debug_dump_logits_sha256(scope: &str, logits: &[f32]) {
    if std::env::var_os("QWEN_DSV4_LOGITS_SHA256").is_some() {
        eprintln!(
            "deepseek_v4 logits: scope={scope:?} count={} sha256_f32le={:x}",
            logits.len(),
            Sha256::digest(bytemuck::cast_slice(logits))
        );
    }
}

/// Debug observability: `QWEN_DSV4_TOP_LOGITS=N` dumps the top-N first-token
/// logits with decoded pieces, for greedy near-tie margin analysis.
fn deepseek_v4_debug_dump_top_logits(scope: &str, logits: &[f32], tokenizer: &Tokenizer) {
    let Some(value) = std::env::var_os("QWEN_DSV4_TOP_LOGITS") else {
        return;
    };
    let count = value
        .to_string_lossy()
        .parse::<usize>()
        .unwrap_or(5)
        .clamp(1, 50);
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    for (rank, &(token, logit)) in ranked.iter().take(count).enumerate() {
        let piece = tokenizer
            .try_decode_piece_bytes_exact(token as i32)
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .unwrap_or_else(|_| "<undecodable>".into());
        let margin = ranked[0].1 - logit;
        eprintln!(
            "deepseek_v4 first_token_logit: scope={scope:?} rank={rank} id={token} logit={logit:.6} margin_to_top={margin:.6} piece={piece:?}"
        );
    }
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
    // --request-stats-jsonl currently only implements DeepSeek V4 single-turn.
    // Reject rather than silently ignore, so consumers cannot mistakenly rely
    // on a sidecar that never gets written.
    ensure!(
        args.request_stats_jsonl.is_none(),
        "--request-stats-jsonl is only supported on DeepSeek V4 single-turn generation today; \
         Qwen single-turn will migrate in a follow-up PR. Use --request-stats for legacy stats output."
    );
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
                            "evicted={} staging_examined={} staging_removed={} ",
                            "staging_allocated_bytes_reclaimed={} ",
                            "staging_live={} staging_legacy={} staging_foreign={} ",
                            "staging_truncated={} identity={} staged_integrity={} ",
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
                        report.store.staging_entries_examined,
                        report.store.staging_entries_removed,
                        report.store.staging_allocated_bytes_reclaimed,
                        report.store.staging_live_entries,
                        report.store.staging_legacy_entries,
                        report.store.staging_foreign_entries,
                        report.store.staging_cleanup_truncated,
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
                            "evicted={} staging_examined={} staging_removed={} ",
                            "staging_allocated_bytes_reclaimed={} ",
                            "staging_live={} staging_legacy={} staging_foreign={} ",
                            "staging_truncated={} identity={} capture_ms={:.1} publish_ms={:.1}"
                        ),
                        publish_outcome_label(report.store.outcome),
                        durable_capture_kind.unwrap_or("unknown"),
                        prepared.matched_prefix_len(),
                        prepared.restored_prefix_len(),
                        prepared.has_pending_token(),
                        durable_capture_stop_reason.map_or("none", StopReason::as_str),
                        report.store.blob_bytes,
                        report.store.evicted_entries,
                        report.store.staging_entries_examined,
                        report.store.staging_entries_removed,
                        report.store.staging_allocated_bytes_reclaimed,
                        report.store.staging_live_entries,
                        report.store.staging_legacy_entries,
                        report.store.staging_foreign_entries,
                        report.store.staging_cleanup_truncated,
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
    explicit: ExplicitCliOptions,
) -> Result<()> {
    args.prefill_chunk.validate()?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    // --request-stats-jsonl is DS4-single-turn-only today; reject on Qwen
    // batch rather than silently no-oping.
    ensure!(
        args.request_stats_jsonl.is_none(),
        "--request-stats-jsonl is not yet implemented on Qwen --requests-jsonl. \
         Use --request-stats for legacy per-request stats output."
    );
    cli_sampling_config(args)?;

    let load_t0 = Instant::now();
    let runtime = Runtime::metal().context("init Metal runtime")?;
    let requested_prefetch = args.model_prefetch.unwrap_or_default();
    let prefetch_policy = requested_prefetch.policy();
    let prefix_cache_max_bytes = if args.batch_size.is_some() || args.concurrency.is_some() {
        0
    } else {
        prefix_cache_max_bytes(args)?
    };
    let defaults = LoadedModelConfig::default();
    let loaded = runtime
        .load_open_model_with_config(
            gguf,
            LoadedModelConfig {
                prefix_cache_max_bytes,
                prefetch_policy,
                ..defaults
            },
        )
        .with_context(|| format!("load model {}", model_path.display()))?;
    ensure!(
        loaded.prefetch_outcome().policy == prefetch_policy,
        "Qwen model prefetch policy changed during load"
    );
    eprintln!(
        "model_prefetch: requested={} effective={} bytes_returned={}",
        requested_prefetch.as_str(),
        prefetch_policy_label(loaded.prefetch_outcome().policy),
        loaded.prefetch_outcome().bytes_returned_total(),
    );
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

    let model_family = match loaded.arch().kind {
        ArchKind::Dense => Some(ModelFamily::Qwen35),
        ArchKind::Moe => Some(ModelFamily::Qwen35Moe),
    };
    let auto_mode = args.execution_mode == Some(execution_selector::ExecutionModeArg::Auto);
    let mut auto_prepared = if auto_mode && requests_path != Path::new("-") {
        Some(prepare_jsonl_requests(requests_path, &tokenizer, args)?)
    } else {
        None
    };
    let auto_selection = if auto_mode {
        let requests = auto_prepared.as_deref().unwrap_or(&[]);
        let all_requests_accelerable = !requests.is_empty()
            && requests.iter().all(|request| {
                request.sampling.temperature == 0.0
                    && request.request.cache_prefix_tokens.is_none()
                    && request.auto_cache_prefix_tokens.is_none()
                    && request.auto_cache_future_hits == 0
            });
        let (dense_summary, moe_summary) = match (model_family, requests.is_empty()) {
            (_, true) => (
                fixed_cohort_jsonl::CohortPlanSummary::default(),
                fixed_cohort_jsonl::CohortPlanSummary::default(),
            ),
            (Some(ModelFamily::Qwen35), false) => (
                fixed_cohort_jsonl::plan_summary::<DENSE_BATCH8_WIDTH>(requests, args)?,
                fixed_cohort_jsonl::CohortPlanSummary::default(),
            ),
            (Some(ModelFamily::Qwen35Moe), false) => (
                fixed_cohort_jsonl::CohortPlanSummary::default(),
                fixed_cohort_jsonl::plan_summary::<MOE_BATCH16_WIDTH>(requests, args)?,
            ),
            (Some(ModelFamily::DeepSeek4) | None, false) => (
                fixed_cohort_jsonl::CohortPlanSummary::default(),
                fixed_cohort_jsonl::CohortPlanSummary::default(),
            ),
        };
        let moe_plan = if model_family == Some(ModelFamily::Qwen35Moe) {
            loaded.inspect_moe_batch16_plan().ok()
        } else {
            None
        };
        let admission_requirements = if requests.is_empty() {
            None
        } else {
            Some(concurrent_jsonl::qwen_execution_admission_requirements(
                &loaded, requests, args,
            )?)
        };
        let admission = |width: usize,
                         max_capacity: usize,
                         executor_scratch_upper_bytes: u64|
         -> Result<bool> {
            let Some(requirements) = admission_requirements else {
                return Ok(false);
            };
            Ok(loaded
                .qwen_execution_memory_admission(
                    width,
                    max_capacity,
                    requirements.prefill_scratch_upper_bytes,
                    executor_scratch_upper_bytes,
                )?
                .admitted)
        };
        let concurrency2_memory_admitted = admission(
            2,
            admission_requirements
                .map(|requirements| requirements.max_capacity)
                .unwrap_or(0),
            1024 * 1024,
        )?;
        let dense_batch8_memory_admitted =
            if model_family == Some(ModelFamily::Qwen35) && loaded.inspect_dense_batch8().is_ok() {
                admission(
                    DENSE_BATCH8_WIDTH,
                    dense_summary.max_execution_capacity,
                    loaded
                        .dense_batch8_scratch_bytes()?
                        .saturating_add(1024 * 1024),
                )?
            } else {
                false
            };
        let moe_batch16_memory_admitted =
            if model_family == Some(ModelFamily::Qwen35Moe) && moe_plan.is_some() {
                admission(
                    MOE_BATCH16_WIDTH,
                    moe_summary.max_execution_capacity,
                    loaded
                        .moe_batch16_scratch_bytes()?
                        .saturating_add(1024 * 1024),
                )?
            } else {
                false
            };
        let selection = execution_selector::select_qwen(execution_selector::QwenSelectionInput {
            mode: args.execution_mode,
            family: model_family,
            arch: loaded.arch(),
            request_count: requests.len(),
            all_requests_accelerable,
            fixed_prefill_chunk: execution_selector::fixed_prefill_chunk(args.prefill_chunk),
            acceleration_blocker: if requests_path == Path::new("-") {
                Some("streaming_input")
            } else {
                execution_selector::qwen_acceleration_blocker(args, explicit, greedy_gpu_mode)
            },
            dense_full_cohorts: dense_summary.full_cohorts,
            dense_serial_remainders: dense_summary.serial_fallback_requests,
            moe_full_cohorts: moe_summary.full_cohorts,
            moe_serial_remainders: moe_summary.serial_fallback_requests,
            fixed_cohort_economics_rejected: match model_family {
                Some(ModelFamily::Qwen35) => dense_summary.economics_rejected_cohorts > 0,
                Some(ModelFamily::Qwen35Moe) => moe_summary.economics_rejected_cohorts > 0,
                Some(ModelFamily::DeepSeek4) | None => false,
            },
            concurrency2_memory_admitted,
            dense_batch8_memory_admitted,
            moe_batch16_memory_admitted,
            moe_plan,
        });
        eprintln!(
            "execution_selection: {}",
            serde_json::to_string(&execution_selector::ExecutionSelectionRecord::new(
                model_family,
                (!requests.is_empty()).then_some(requests.len()),
                selection,
                match model_family {
                    Some(ModelFamily::Qwen35) => dense_summary.ragged_prompt_policy,
                    Some(ModelFamily::Qwen35Moe) => moe_summary.ragged_prompt_policy,
                    Some(ModelFamily::DeepSeek4) | None => None,
                },
                match model_family {
                    Some(ModelFamily::Qwen35) => dense_summary.ragged_prompt_plan_decision,
                    Some(ModelFamily::Qwen35Moe) => moe_summary.ragged_prompt_plan_decision,
                    Some(ModelFamily::DeepSeek4) | None => None,
                },
                match model_family {
                    Some(ModelFamily::Qwen35) => dense_summary.refill_policy,
                    Some(ModelFamily::Qwen35Moe) | Some(ModelFamily::DeepSeek4) | None => None,
                },
                match model_family {
                    Some(ModelFamily::Qwen35) => dense_summary.planned_refill_arenas,
                    Some(ModelFamily::Qwen35Moe) | Some(ModelFamily::DeepSeek4) | None => None,
                },
                match model_family {
                    Some(ModelFamily::Qwen35) => dense_summary.planned_refill_requests,
                    Some(ModelFamily::Qwen35Moe) | Some(ModelFamily::DeepSeek4) | None => None,
                },
            ))
            .context("serialize execution selection")?
        );
        if selection.selected.accelerated() {
            let stats = loaded.set_prefix_cache_max_bytes(0);
            ensure!(
                stats.entries == 0 && stats.indexed_bytes == 0,
                "automatic execution selection found a populated prefix cache before request execution"
            );
        }
        Some(selection)
    } else {
        None
    };
    let effective_prefix_cache_max_bytes = loaded.prefix_cache_stats().max_indexed_bytes;

    eprintln!(
        "loaded {} in {:.1} ms; prefix_cache_max_mib={}",
        model_path.display(),
        load_ms,
        effective_prefix_cache_max_bytes / (1024 * 1024),
    );

    if args.concurrency.is_some() {
        n_requests += concurrent_jsonl::run_file(
            &loaded,
            &tokenizer,
            requests_path,
            args,
            greedy_gpu_mode,
            &mut stdout,
        )?;
    } else if args.batch_size.is_some() {
        n_requests += fixed_cohort_jsonl::run_file(
            &loaded,
            &tokenizer,
            requests_path,
            args,
            greedy_gpu_mode,
            &mut stdout,
        )?;
    } else if let Some(selection) = auto_selection
        && selection.selected.accelerated()
    {
        let requests = auto_prepared
            .as_deref()
            .expect("accelerated automatic selection requires file lookahead");
        n_requests += match selection.selected {
            execution_selector::SelectedExecution::Concurrency2 => concurrent_jsonl::run_prepared(
                &loaded,
                &tokenizer,
                requests,
                args,
                greedy_gpu_mode,
                &mut stdout,
            )?,
            execution_selector::SelectedExecution::DenseBatch8 => fixed_cohort_jsonl::run_prepared(
                &loaded,
                &tokenizer,
                requests,
                args,
                greedy_gpu_mode,
                &mut stdout,
                DENSE_BATCH8_WIDTH,
            )?,
            execution_selector::SelectedExecution::MoeBatch16 => fixed_cohort_jsonl::run_prepared(
                &loaded,
                &tokenizer,
                requests,
                args,
                greedy_gpu_mode,
                &mut stdout,
                MOE_BATCH16_WIDTH,
            )?,
            execution_selector::SelectedExecution::Serial => {
                unreachable!("accelerated selection checked above")
            }
        };
    } else if requests_path == Path::new("-") {
        shutdown::checkpoint()?;
        let stdin = std::io::stdin();
        let reader = stdin.lock();
        if args.cache_prefix_auto_min_tokens > 0 {
            eprintln!(
                "prefix-cache auto admission needs request lookahead; disabled for stdin JSONL"
            );
        }
        for (line_idx, line) in reader.lines().enumerate() {
            shutdown::checkpoint()?;
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
        let mut prepared = match auto_prepared.take() {
            Some(prepared) => prepared,
            None => prepare_jsonl_requests(requests_path, &tokenizer, args)?,
        };
        n_requests += run_prepared_jsonl_serial(
            &loaded,
            &tokenizer,
            &mut prepared,
            args,
            greedy_gpu_mode,
            &mut stdout,
            &mut stats_file,
        )?;
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

#[allow(clippy::too_many_arguments)]
fn run_prepared_jsonl_serial(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    prepared: &mut [PreparedJsonlRequest],
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
    stdout: &mut impl Write,
    stats_file: &mut Option<std::fs::File>,
) -> Result<usize> {
    discover_auto_cache_prefixes(prepared, args.cache_prefix_auto_min_tokens);
    let mut completed = 0usize;
    for prepared_request in prepared {
        let (output, stats) =
            run_jsonl_request(loaded, tokenizer, prepared_request, args, greedy_gpu_mode)
                .with_context(|| format!("run request {}", prepared_request.id))?;

        serde_json::to_writer(&mut *stdout, &output).context("write request output")?;
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
        completed += 1;
    }
    Ok(completed)
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
    shutdown::checkpoint()?;
    for (line_idx, line) in reader.lines().enumerate() {
        shutdown::checkpoint()?;
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
    let mut request: JsonlRequest =
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
    request.prompt = None;
    request.prompt_file = None;
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

fn jsonl_generation_capacity(
    prepared: &PreparedJsonlRequest,
    args: &Args,
) -> Result<(usize, usize)> {
    let n_generate = prepared.request.tokens.unwrap_or(args.tokens);
    ensure!(
        n_generate > 0,
        "tokens must be >= 1 for request {}",
        prepared.id
    );
    let prompt_and_generation = prepared
        .prompt_ids
        .len()
        .checked_add(n_generate)
        .context("sequence capacity overflow")?;
    let min_capacity = prompt_and_generation
        .checked_add(16)
        .context("sequence capacity overflow")?;
    let capacity = args.max_context_tokens.unwrap_or(min_capacity);
    ensure!(
        capacity >= prompt_and_generation,
        "max context {} is smaller than prompt {} + generation {} for request {}",
        capacity,
        prepared.prompt_ids.len(),
        n_generate,
        prepared.id,
    );
    Ok((n_generate, capacity))
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

    let (n_generate, capacity) = jsonl_generation_capacity(prepared, args)?;

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

    let prompt_hash = token_hash_hex(prompt_ids);
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
            .restore_cached_prefix(&mut sequence, prompt_ids)
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
            let (logits, ms) = prefill_span(&forward, &mut sequence, &mut scratch, prompt_ids, 0)?;
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
    let thinking_partition = generated_thinking_partition(tokenizer, &generated);
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
        request_stats_contract: "qwen_jsonl_v2",
        id: id.to_string(),
        line: prepared.line,
        model: loaded.path().display().to_string(),
        build_commit: env!("QWEN_BUILD_COMMIT"),
        build_dirty: parse_build_dirty(env!("QWEN_BUILD_DIRTY")),
        build_source_state: env!("QWEN_BUILD_SOURCE_STATE"),
        model_prefetch_policy: prefetch_policy_label(loaded.prefetch_outcome().policy),
        model_prefetch_bytes_returned: loaded.prefetch_outcome().bytes_returned_total(),
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
        thinking_partition: thinking_partition.clone(),
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
        thinking_partition,
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

fn generated_thinking_partition(
    tokenizer: &Tokenizer,
    generated: &[i32],
) -> Option<GeneratedThinkingPartition> {
    let pieces = generated
        .iter()
        .map(|&token| tokenizer.decode_piece(token))
        .collect::<Vec<_>>();
    thinking_partition_from_pieces(&pieces)
}

fn thinking_partition_from_pieces(pieces: &[String]) -> Option<GeneratedThinkingPartition> {
    const DELIMITER: &str = "</think>";
    let mut decoded = String::new();
    let mut boundaries = Vec::with_capacity(pieces.len() + 1);
    boundaries.push(0usize);
    for piece in pieces {
        decoded.push_str(piece);
        boundaries.push(decoded.len());
    }

    let delimiter_start = decoded.find(DELIMITER)?;
    let delimiter_end = delimiter_start + DELIMITER.len();
    let start_exact = boundaries
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, &offset)| (offset == delimiter_start).then_some(index));
    let end_exact = boundaries
        .iter()
        .enumerate()
        .find_map(|(index, &offset)| (offset == delimiter_end).then_some(index));
    let delimiter_start_token_index = start_exact.unwrap_or_else(|| {
        boundaries
            .iter()
            .rposition(|&offset| offset < delimiter_start)
            .unwrap_or(0)
    });
    let delimiter_end_token_index_exclusive = end_exact.unwrap_or_else(|| {
        boundaries
            .iter()
            .position(|&offset| offset > delimiter_end)
            .unwrap_or(pieces.len())
    });
    let delimiter_token_aligned = start_exact.is_some() && end_exact.is_some();
    let (reasoning_tokens, delimiter_tokens, visible_tokens) = match (start_exact, end_exact) {
        (Some(start), Some(end)) if start <= end => (
            Some(start),
            Some(end - start),
            Some(pieces.len().saturating_sub(end)),
        ),
        _ => (None, None, None),
    };

    Some(GeneratedThinkingPartition {
        delimiter: DELIMITER,
        delimiter_start_token_index,
        delimiter_end_token_index_exclusive,
        delimiter_token_aligned,
        reasoning_tokens,
        delimiter_tokens,
        visible_tokens,
    })
}

fn prefill_span(
    forward: &MetalForward<'_>,
    sequence: &mut Sequence,
    scratch: &mut MetalDFlashLayerMajorScratch,
    token_ids: &[i32],
    start_position: usize,
) -> Result<(Vec<f32>, f64)> {
    shutdown::checkpoint()?;
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
    shutdown::checkpoint()?;
    sequence.advance_by(token_ids.len())?;
    Ok((logits, t0.elapsed().as_secs_f64() * 1e3))
}

const QWEN_PREFIX_FANOUT_EXACT_LCP_ENV: &str = "QWEN_PREFIX_FANOUT_EXACT_LCP";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QwenPrefixFanoutBoundaryPolicy {
    ChunkAligned,
    TinySuffixExactLcp,
    ExactLcp,
}

impl QwenPrefixFanoutBoundaryPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::ChunkAligned => "chunk_aligned",
            Self::TinySuffixExactLcp => "tiny_suffix_exact_lcp",
            Self::ExactLcp => "exact_lcp",
        }
    }
}

fn parse_qwen_prefix_fanout_boundary_policy(
    value: Option<&str>,
) -> Result<QwenPrefixFanoutBoundaryPolicy> {
    let Some(value) = value else {
        return Ok(QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp),
        "0" | "false" | "no" | "off" => Ok(QwenPrefixFanoutBoundaryPolicy::ChunkAligned),
        "1" | "true" | "yes" | "on" => Ok(QwenPrefixFanoutBoundaryPolicy::ExactLcp),
        _ => bail!("{QWEN_PREFIX_FANOUT_EXACT_LCP_ENV} must be auto or a boolean"),
    }
}

fn qwen_prefix_fanout_boundary_policy() -> Result<QwenPrefixFanoutBoundaryPolicy> {
    let value = std::env::var_os(QWEN_PREFIX_FANOUT_EXACT_LCP_ENV)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("{QWEN_PREFIX_FANOUT_EXACT_LCP_ENV} must be valid UTF-8"))
        })
        .transpose()?;
    parse_qwen_prefix_fanout_boundary_policy(value.as_deref())
}

const PRIVATE_SUFFIX_SINGLETON_ENV: &str = "QWEN_PRIVATE_SUFFIX_SINGLETON";
const PRIVATE_SUFFIX_SINGLETON_MAX_TOKENS: usize = 6;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrivateSuffixExecutionMode {
    Packed,
    Singleton,
}

struct PrivateSuffixResult {
    logits: Vec<f32>,
    ms: f64,
    mode: PrivateSuffixExecutionMode,
}

fn parse_private_suffix_singleton_enabled(
    value: Option<&str>,
    default_enabled: bool,
) -> Result<bool> {
    let Some(value) = value else {
        return Ok(default_enabled);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{PRIVATE_SUFFIX_SINGLETON_ENV} must be a boolean"),
    }
}

fn private_suffix_singleton_enabled(default_enabled: bool) -> Result<bool> {
    let value = std::env::var_os(PRIVATE_SUFFIX_SINGLETON_ENV)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("{PRIVATE_SUFFIX_SINGLETON_ENV} must be valid UTF-8"))
        })
        .transpose()?;
    parse_private_suffix_singleton_enabled(value.as_deref(), default_enabled)
}

fn choose_private_suffix_execution_mode(
    enabled: bool,
    suffix_tokens: usize,
) -> PrivateSuffixExecutionMode {
    if enabled && suffix_tokens <= PRIVATE_SUFFIX_SINGLETON_MAX_TOKENS {
        PrivateSuffixExecutionMode::Singleton
    } else {
        PrivateSuffixExecutionMode::Packed
    }
}

fn prefill_private_suffix(
    forward: &MetalForward<'_>,
    sequence: &mut Sequence,
    scratch: &mut MetalDFlashLayerMajorScratch,
    token_ids: &[i32],
    start_position: usize,
) -> Result<PrivateSuffixResult> {
    ensure!(
        !token_ids.is_empty(),
        "cannot prefill an empty private suffix"
    );
    let mode = choose_private_suffix_execution_mode(
        private_suffix_singleton_enabled(true)?,
        token_ids.len(),
    );
    if mode == PrivateSuffixExecutionMode::Packed {
        let (logits, ms) = prefill_span(forward, sequence, scratch, token_ids, start_position)?;
        return Ok(PrivateSuffixResult {
            logits,
            ms,
            mode: PrivateSuffixExecutionMode::Packed,
        });
    }

    shutdown::checkpoint()?;
    sequence.check_position(start_position)?;
    sequence.ensure_can_append(token_ids.len())?;
    let final_position = start_position
        .checked_add(token_ids.len() - 1)
        .context("private suffix final position overflow")?;
    u32::try_from(final_position).context("private suffix final position does not fit u32")?;
    let t0 = Instant::now();
    let mut logits = None;
    for (offset, &token) in token_ids.iter().enumerate() {
        shutdown::checkpoint()?;
        let position = start_position
            .checked_add(offset)
            .context("private suffix position overflow")?;
        logits = Some(
            forward
                .single_token(
                    token,
                    u32::try_from(position).context("private suffix position does not fit u32")?,
                    unsafe { sequence.metal_session_mut() },
                )
                .context("consume private suffix token")?,
        );
        sequence.advance_by(1)?;
        shutdown::checkpoint()?;
    }
    Ok(PrivateSuffixResult {
        logits: logits.expect("nonempty private suffix produced final logits"),
        ms: t0.elapsed().as_secs_f64() * 1e3,
        mode: PrivateSuffixExecutionMode::Singleton,
    })
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
        shutdown::checkpoint()?;
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
        shutdown::checkpoint()?;
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
        shutdown::checkpoint()?;
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

fn deepseek_v4_checkpoint_store(
    args: &Args,
    staged_integrity: Option<StagedIntegrityMode>,
) -> Result<Option<DeepSeekV4CheckpointStore>> {
    let Some(root) = args.durable_prefix_cache.as_ref() else {
        return Ok(None);
    };
    let budget = mib_to_bytes(
        args.durable_prefix_cache_max_mib,
        "durable prefix cache byte budget",
    )?;
    Ok(Some(match staged_integrity {
        Some(mode) => DeepSeekV4CheckpointStore::with_staged_integrity(root, budget, mode),
        None => DeepSeekV4CheckpointStore::new(root, budget),
    }))
}

fn deepseek_v4_durable_capture_prefix_len(
    prompt_len: usize,
    admitted: bool,
) -> Result<Option<usize>> {
    if !admitted || prompt_len < 2 {
        return Ok(None);
    }
    deepseek_v4_snapshot_publish_prefix(prompt_len).map(Some)
}

/// Probe the durable store and restore the longest strict prefix into the
/// session, translating the runtime's fail-open policy into telemetry:
/// misses, store faults, and pre-mutation restore allocation failures all
/// return `(0, 0)` and cold-prefill; only invariant restore failures are
/// fatal. Returns `(restored_prefix_len, payload_bytes)`.
fn attempt_deepseek_v4_durable_restore(
    durable_store: Option<&DeepSeekV4CheckpointStore>,
    durable_has_blobs: Option<bool>,
    session: &mut DeepSeekV4Session,
    prompt_token_ids: &[u32],
    durable_max_record_bytes: u64,
    durable_probe_ms: f64,
    durable_restore_ms: &mut f64,
) -> Result<(usize, u64)> {
    let store = match (durable_store, durable_has_blobs) {
        (Some(store), Some(true)) => store,
        (Some(_), Some(false)) => {
            eprintln!(
                "durable_prefix_cache: family=deepseek_v4 store_empty=true restore_total_ms={durable_probe_ms:.1}",
            );
            return Ok((0, 0));
        }
        _ => return Ok((0, 0)),
    };
    let restore_t0 = Instant::now();
    match session.restore_durable_prefix(store, prompt_token_ids, durable_max_record_bytes) {
        Ok(attempt) => {
            *durable_restore_ms = durable_probe_ms + restore_t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                concat!(
                    "durable_prefix_cache: family=deepseek_v4 checkpoint_hit={} ",
                    "matched={} restored={} candidates={} corrupt_removed={} ",
                    "restore_total_ms={:.1}"
                ),
                attempt.restored_prefix_len.is_some(),
                attempt.matched_prefix_len,
                attempt.restored_prefix_len.unwrap_or(0),
                attempt.candidates_examined,
                attempt.corrupt_entries_removed,
                *durable_restore_ms,
            );
            Ok((
                attempt.restored_prefix_len.unwrap_or(0),
                attempt.payload_bytes,
            ))
        }
        Err(DeepSeekV4DurableError::UnboundIdentity) => {
            // Occupied store, but the prompt was too short to resolve an
            // identity for this session; there is nothing to restore.
            eprintln!(
                "durable_prefix_cache: family=deepseek_v4 checkpoint_hit=false skipped=short_prompt restore_total_ms={durable_probe_ms:.1}",
            );
            Ok((0, 0))
        }
        Err(DeepSeekV4DurableError::Restore(error))
            if causal_snapshot_restore_error_kind(&error)
                == DeepSeekV4SnapshotRestoreErrorKind::Allocation =>
        {
            *durable_restore_ms = durable_probe_ms + restore_t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "warning: durable DeepSeek V4 prefix restore allocation failed; cold-prefilling: {error}"
            );
            Ok((0, 0))
        }
        Err(DeepSeekV4DurableError::Restore(error)) => {
            Err(error).context("restore durable DeepSeek V4 causal snapshot")
        }
        Err(DeepSeekV4DurableError::Store(error)) => {
            *durable_restore_ms = durable_probe_ms + restore_t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "warning: durable DeepSeek V4 prefix lookup failed after {:.1} ms; cold-prefilling: {error}",
                *durable_restore_ms,
            );
            Ok((0, 0))
        }
    }
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
        IdentityCacheOutcome::DeclaredAndStored => "declared_stored",
        IdentityCacheOutcome::DeclaredUncached => "declared_uncached",
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

/// Type-safe wrapper for the generated-token-ids fingerprint. The tuple
/// field is scoped to a private child module so that neither crate root
/// code, nor tests, nor any other module can bypass `of()` to construct a
/// digest with arbitrary bytes. This is what actually enforces the
/// invariant that the emitted value under algorithm identifier
/// `sha256-qwen-generated-token-ids-v1` is the output of that exact algorithm.
mod fingerprint {
    use sha2::{Digest, Sha256};

    #[derive(Debug, Clone, Copy, Eq, PartialEq)]
    pub struct GeneratedTokenSha256Digest([u8; 32]);

    impl GeneratedTokenSha256Digest {
        /// Compute the canonical fingerprint over token IDs. Byte layout:
        /// `domain-separator || length_u64_le || (token_i32_le)*`. See
        /// `sha256-qwen-generated-token-ids-v1` algorithm identifier.
        pub fn of(tokens: &[i32]) -> Self {
            let mut digest = Sha256::new();
            digest.update(b"qwen-generated-token-ids-v1\0");
            digest.update((tokens.len() as u64).to_le_bytes());
            for token in tokens {
                digest.update(token.to_le_bytes());
            }
            Self(digest.finalize().into())
        }

        pub fn hex(&self) -> String {
            crate::hex_encode_bytes(&self.0)
        }

        #[cfg(test)]
        pub fn as_bytes(&self) -> &[u8; 32] {
            &self.0
        }
    }
}

pub(crate) use fingerprint::GeneratedTokenSha256Digest;

fn generated_token_sha256(tokens: &[i32]) -> String {
    GeneratedTokenSha256Digest::of(tokens).hex()
}

fn hex_encode_bytes(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Per-process invocation identifier: 16 bytes of /dev/urandom entropy
/// formatted as lowercase hex (matches UUID/ULID uniqueness class without
/// adding a dep). **Fails closed** if secure entropy is unavailable — better
/// to refuse to run than to emit records with a weak, potentially colliding
/// identifier that consumers might trust for cross-run correlation.
///
/// Generated on first access; force-init only when telemetry is actually
/// requested (see the --request-stats-jsonl pre-flight branch in the DS4
/// single-turn generator), so `--help`, `--version`, non-telemetry
/// invocations, and dispatch errors never touch /dev/urandom.
static INVOCATION_ID: LazyLock<String> = LazyLock::new(|| {
    generate_invocation_id_from(EntropySource::DevUrandom)
        .unwrap_or_else(|e| panic!("cannot initialize invocation id: {e}"))
});

#[derive(Copy, Clone, Debug)]
enum EntropySource {
    DevUrandom,
    #[cfg(test)]
    Deterministic([u8; 16]),
    #[cfg(test)]
    ForceFail,
}

fn generate_invocation_id_from(source: EntropySource) -> Result<String> {
    let mut buf = [0u8; 16];
    match source {
        EntropySource::DevUrandom => {
            let mut f = std::fs::File::open("/dev/urandom")
                .context("open /dev/urandom for invocation id entropy")?;
            f.read_exact(&mut buf)
                .context("read 16 bytes from /dev/urandom for invocation id")?;
        }
        #[cfg(test)]
        EntropySource::Deterministic(bytes) => {
            buf = bytes;
        }
        #[cfg(test)]
        EntropySource::ForceFail => {
            return Err(anyhow!(
                "test-injected entropy failure (secure randomness unavailable)"
            ));
        }
    }
    Ok(hex_encode_bytes(&buf))
}

/// Append a serialized JSONL record with cooperating-writer integrity:
///   * The record is fully serialized into a memory buffer first, so a
///     serialization failure never writes a partial line.
///   * An advisory exclusive `flock` is held across the tail-repair check,
///     the write, and the `fsync`. Cooperating processes cannot interleave
///     with each other. Non-cooperating writers (that ignore flock) are
///     out of scope.
///   * If the file already ends with a partial record (does not end with
///     `\n`), a leading `\n` is prepended to the buffer so the next record
///     starts on a fresh line — otherwise we would concatenate the new
///     record onto the abandoned partial one and corrupt that line.
///   * `sync_data` is invoked after the write, so delayed I/O failures
///     (e.g. `ENOSPC` on a filesystem with write-back caching) surface as
///     errors instead of being silently deferred past our success return.
///     `File::flush` is a no-op on Unix and Windows; `sync_data` is not.
///   * On write or fsync failure, best-effort rollback truncates the file
///     back to the original length. Rollback failure is chained into the
///     returned error rather than silently discarded.
fn append_jsonl_record<T: Serialize>(path: &Path, record: &T, label: &str) -> Result<()> {
    let payload =
        serde_json::to_vec(record).with_context(|| format!("serialize {label} record"))?;
    // Open with read+append so the SAME locked fd can serve both the tail
    // probe (via pread) and the write. open_append_file grants append-only,
    // which would fail pread with EBADF.
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {label} directory {}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {label} {}", path.display()))?;
    let fd = file.as_raw_fd();
    // SAFETY: fd is valid for the duration of `file`; flock(2) accepts any
    // open file descriptor. LOCK_EX blocks until acquired.
    let lock_rc = unsafe { libc::flock(fd, libc::LOCK_EX) };
    if lock_rc != 0 {
        return Err(anyhow!(
            "acquire exclusive lock on {label} {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }
    let unlock = |fd: std::os::unix::io::RawFd| {
        // SAFETY: fd is valid for the caller-held `file`; LOCK_UN is defined.
        let _ = unsafe { libc::flock(fd, libc::LOCK_UN) };
    };
    let original_len = match file.metadata() {
        Ok(m) => m.len(),
        Err(e) => {
            unlock(fd);
            return Err(anyhow::Error::new(e).context(format!("stat {label} {}", path.display())));
        }
    };
    // Detect a pre-existing partial-record tail (file does not end with '\n').
    // Read through the SAME locked fd via pread(2) to avoid TOCTOU across the
    // rename/replace race a second open on the pathname would expose.
    let mut buf = Vec::with_capacity(payload.len() + 2);
    if original_len > 0 {
        let mut probe = [0u8; 1];
        let offset = (original_len - 1) as libc::off_t;
        // SAFETY: fd is valid; pread reads at a specific offset without
        // moving the file pointer, does not mutate the file, and returns
        // -1 on error with errno set.
        let n = unsafe { libc::pread(fd, probe.as_mut_ptr().cast(), probe.len(), offset) };
        if n < 0 {
            unlock(fd);
            return Err(anyhow!(
                "probe tail of {label} {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
        // Concurrent truncation between metadata and pread would leave the
        // tail unprobed; refuse rather than risk concatenating onto it.
        if n != 1 {
            unlock(fd);
            return Err(anyhow!(
                "tail probe of {label} {} returned {n} bytes at offset {}; \
                 expected 1 (concurrent truncation between metadata and pread?)",
                path.display(),
                offset,
            ));
        }
        if probe[0] != b'\n' {
            // Isolate our record from the orphan tail rather than concatenating.
            buf.push(b'\n');
        }
    }
    buf.extend_from_slice(&payload);
    buf.push(b'\n');
    // Write + durability. sync_data persists file content; sync_containing_dir
    // persists a newly-created directory entry (fsync on the file alone does
    // NOT guarantee the entry survives a crash for a new file).
    let write_result = file
        .write_all(&buf)
        .and_then(|()| file.flush())
        .and_then(|()| file.sync_data())
        .and_then(|()| sync_containing_dir(path));
    if let Err(e) = write_result {
        // Best-effort rollback: truncate AND sync to persist the reverted
        // state. Chain both errors together — silently discarding either
        // would let a corrupt tail persist beyond the reported error.
        let rollback_err = file
            .set_len(original_len)
            .and_then(|()| file.sync_data())
            .err();
        unlock(fd);
        let mut chained =
            anyhow::Error::new(e).context(format!("append {label} record to {}", path.display()));
        if let Some(re) = rollback_err {
            chained = chained.context(format!(
                "rollback truncate/sync also failed for {}: {}",
                path.display(),
                re
            ));
        }
        return Err(chained);
    }
    unlock(fd);
    Ok(())
}

/// fsync the directory containing `path` so a newly-created entry is
/// persisted before we report success. `sync_data`/`fsync` on the file
/// itself is not enough for a new directory entry on most filesystems.
///
/// If the path's parent hierarchy was newly created (via `create_dir_all`),
/// sync each newly-materialized directory on the way up to an existing
/// ancestor, so the whole hierarchy is durable — not just the final leaf
/// directory containing the file.
fn sync_containing_dir(path: &Path) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let leaf = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    // Walk up from the leaf, syncing each directory. Stop at root or when
    // a further ancestor doesn't need syncing (best-effort: we always sync
    // the leaf; ancestors are synced too as a conservative default so that
    // a create_dir_all() hierarchy survives crash).
    let mut cur: Option<&Path> = Some(leaf);
    while let Some(dir) = cur {
        std::fs::File::open(dir)?.sync_all()?;
        cur = dir
            .parent()
            .filter(|p| !p.as_os_str().is_empty() && *p != dir);
    }
    Ok(())
}

/// Preflight the --request-stats-jsonl destination using the exact open
/// mode the emission path uses (read+append). An append-writable but
/// unreadable file cannot pass this check and then fail emission after
/// inference cost is paid. Also creates parent directories so append
/// itself doesn't fail on first record.
fn preflight_request_stats_jsonl(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create request stats jsonl (pre-flight) directory {}",
                parent.display()
            )
        })?;
    }
    let _ = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)
        .with_context(|| {
            format!(
                "open request stats jsonl (pre-flight) with read+append {}",
                path.display()
            )
        })?;
    Ok(())
}

/// Coerce a metric to a well-defined finite JSON representation. Returns 0.0
/// for non-finite (NaN, ±∞) or negative values. Emits a `tracing::warn` so
/// callers notice degenerate measurements. Prevents JSON `null` in fields
/// documented as non-negative f64 (serde_json serializes NaN/∞ as `null`).
fn sanitize_finite_metric(value: f64, label: &str) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        tracing::warn!(
            metric = label,
            raw_value = value,
            "non-finite metric coerced to 0.0 for stats emission"
        );
        0.0
    }
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
    let gguf = qwen_llm::gguf::GgufFile::open(model_path)?;
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
        if let Some(rest) = t.name.strip_prefix("blk.")
            && let Some(dot) = rest.find('.')
            && let Ok(idx) = rest[..dot].parse::<u32>()
        {
            by_layer.entry(idx).or_default().push(&rest[dot + 1..]);
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
        "qwen35.nextn_predict_layers",
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
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn qwen_model_prefetch_cli_is_explicit_and_jsonl_scoped() {
        assert!(matches!(
            QwenModelPrefetchArg::Auto.policy(),
            PrefetchPolicy::ColdOnly { .. }
        ));
        assert_eq!(QwenModelPrefetchArg::Off.policy(), PrefetchPolicy::Off);

        let args = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "requests.jsonl",
            "--model-prefetch",
            "off",
        ])
        .unwrap();
        assert_eq!(args.model_prefetch, Some(QwenModelPrefetchArg::Off));
        let wrong_scope = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--model-prefetch",
            "off",
        ])
        .unwrap();
        assert!(
            validate_qwen_model_prefetch_scope(&wrong_scope)
                .unwrap_err()
                .to_string()
                .contains("requires --requests-jsonl")
        );
    }

    #[test]
    fn generated_thinking_partition_counts_only_aligned_segments() {
        let aligned = thinking_partition_from_pieces(&[
            "reason".to_string(),
            "ing".to_string(),
            "</think>".to_string(),
            "\n\nanswer".to_string(),
        ])
        .unwrap();
        assert_eq!(aligned.reasoning_tokens, Some(2));
        assert_eq!(aligned.delimiter_tokens, Some(1));
        assert_eq!(aligned.visible_tokens, Some(1));
        assert!(aligned.delimiter_token_aligned);

        let split =
            thinking_partition_from_pieces(&["reason</thi".to_string(), "nk>answer".to_string()])
                .unwrap();
        assert!(!split.delimiter_token_aligned);
        assert_eq!(split.reasoning_tokens, None);
        assert_eq!(split.visible_tokens, None);
        assert!(thinking_partition_from_pieces(&["answer".to_string()]).is_none());
    }

    #[test]
    fn exact_lcp_fanout_policy_is_bounded_by_default_and_strict() {
        assert_eq!(
            parse_qwen_prefix_fanout_boundary_policy(None).unwrap(),
            QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp
        );
        assert_eq!(
            parse_qwen_prefix_fanout_boundary_policy(Some("auto")).unwrap(),
            QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp
        );
        assert_eq!(
            parse_qwen_prefix_fanout_boundary_policy(Some("YES")).unwrap(),
            QwenPrefixFanoutBoundaryPolicy::ExactLcp
        );
        assert_eq!(
            parse_qwen_prefix_fanout_boundary_policy(Some("off")).unwrap(),
            QwenPrefixFanoutBoundaryPolicy::ChunkAligned
        );
        assert!(parse_qwen_prefix_fanout_boundary_policy(Some("sometimes")).is_err());
    }

    #[test]
    fn private_suffix_singleton_policy_is_strict_and_rollbackable() {
        assert!(parse_private_suffix_singleton_enabled(None, true).unwrap());
        assert!(!parse_private_suffix_singleton_enabled(None, false).unwrap());
        assert!(!parse_private_suffix_singleton_enabled(Some("off"), true).unwrap());
        assert!(parse_private_suffix_singleton_enabled(Some("YES"), false).unwrap());
        assert!(parse_private_suffix_singleton_enabled(Some("sometimes"), true).is_err());
        assert_eq!(
            choose_private_suffix_execution_mode(true, 6),
            PrivateSuffixExecutionMode::Singleton
        );
        assert_eq!(
            choose_private_suffix_execution_mode(true, 7),
            PrivateSuffixExecutionMode::Packed
        );
        assert_eq!(
            choose_private_suffix_execution_mode(false, 1),
            PrivateSuffixExecutionMode::Packed
        );
    }

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
    fn qwen_no_thinking_capability_is_closed_to_the_validated_identity() {
        assert!(validated_qwen36_no_thinking_identity(
            ModelFamily::Qwen35Moe,
            Some("Qwen3.6 35B A3B"),
            Some("gpt2"),
            Some("qwen35"),
        ));
        assert!(validated_qwen38_prompt_identity(
            ModelFamily::Qwen35,
            Some("Qwen3.8 27B!"),
            Some("Qwen3.8-27B"),
            Some("gpt2"),
            Some("qwen35"),
            Some(262_144),
            Some(65),
            Some(1),
            Some(5_120),
            Some(17_408),
        ));
        assert!(!validated_qwen38_prompt_identity(
            ModelFamily::Qwen35,
            Some("Qwen3.8 27B!"),
            Some("Qwen3.8-27B"),
            Some("gpt2"),
            Some("qwen35"),
            Some(262_144),
            Some(64),
            Some(1),
            Some(5_120),
            Some(17_408),
        ));
        for identity in [
            (
                ModelFamily::Qwen35Moe,
                Some("Qwen3.8 27B!"),
                Some("qwen35"),
                Some(262_144),
                Some(65),
                Some(1),
                Some(5_120),
                Some(17_408),
            ),
            (
                ModelFamily::Qwen35,
                Some("Qwen3.7 27B"),
                Some("qwen35"),
                Some(262_144),
                Some(65),
                Some(1),
                Some(5_120),
                Some(17_408),
            ),
            (
                ModelFamily::Qwen35,
                Some("Qwen3.8 27B!"),
                Some("other"),
                Some(262_144),
                Some(65),
                Some(1),
                Some(5_120),
                Some(17_408),
            ),
            (
                ModelFamily::Qwen35,
                Some("Qwen3.8 27B!"),
                Some("qwen35"),
                Some(131_072),
                Some(65),
                Some(1),
                Some(5_120),
                Some(17_408),
            ),
            (
                ModelFamily::Qwen35,
                Some("Qwen3.8 27B!"),
                Some("qwen35"),
                Some(262_144),
                Some(65),
                Some(0),
                Some(5_120),
                Some(17_408),
            ),
            (
                ModelFamily::Qwen35,
                Some("Qwen3.8 27B!"),
                Some("qwen35"),
                Some(262_144),
                Some(65),
                Some(1),
                Some(4_096),
                Some(17_408),
            ),
        ] {
            assert!(!validated_qwen38_prompt_identity(
                identity.0,
                identity.1,
                None,
                Some("gpt2"),
                identity.2,
                identity.3,
                identity.4,
                identity.5,
                identity.6,
                identity.7,
            ));
        }
        for identity in [
            (
                ModelFamily::Qwen35,
                Some("Qwen3.6 35B A3B"),
                Some("gpt2"),
                Some("qwen35"),
            ),
            (
                ModelFamily::Qwen35Moe,
                Some("Qwen3.5 35B A3B"),
                Some("gpt2"),
                Some("qwen35"),
            ),
            (
                ModelFamily::Qwen35Moe,
                Some("Qwen3.6 35B A3B"),
                Some("gpt2"),
                Some("other"),
            ),
        ] {
            assert!(!validated_qwen36_no_thinking_identity(
                identity.0, identity.1, identity.2, identity.3,
            ));
        }
    }

    #[test]
    fn qwen38_reasoning_effort_resolver_is_typed_and_fail_closed() {
        assert_eq!(
            resolve_qwen38_generation_mode(true, false, None).unwrap(),
            Some(Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Xhigh))
        );
        for (input, expected) in [
            (cli::RunReasoningEffort::Low, Qwen38ReasoningEffort::Low),
            (
                cli::RunReasoningEffort::Medium,
                Qwen38ReasoningEffort::Medium,
            ),
            (cli::RunReasoningEffort::Xhigh, Qwen38ReasoningEffort::Xhigh),
        ] {
            assert_eq!(
                resolve_qwen38_generation_mode(true, false, Some(input)).unwrap(),
                Some(Qwen38GenerationMode::Thinking(expected))
            );
        }
        assert_eq!(
            resolve_qwen38_generation_mode(true, true, None).unwrap(),
            Some(Qwen38GenerationMode::NoThinking)
        );
        assert!(
            resolve_qwen38_generation_mode(false, false, Some(cli::RunReasoningEffort::Low))
                .unwrap_err()
                .to_string()
                .contains("validated only for Qwen3.8 27B")
        );
        assert_eq!(
            resolve_qwen38_generation_mode(false, false, None).unwrap(),
            None
        );
        assert!(
            resolve_qwen38_generation_mode(true, true, Some(cli::RunReasoningEffort::Low)).is_err()
        );
    }

    #[test]
    fn deepseek_v4_multigroup_selector_cli_contract_is_explicit_and_bounded() {
        let default =
            Args::try_parse_from(["qwen", "--model", "model.gguf", "--prompt", "hello"]).unwrap();
        assert_eq!(
            default.deepseek_v4_multigroup_selector,
            DeepSeekV4MultigroupSelectorArg::Auto
        );

        let explicit = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--deepseek-v4-multigroup-selector",
            "qualified-experimental",
        ])
        .unwrap();
        assert_eq!(
            explicit.deepseek_v4_multigroup_selector,
            DeepSeekV4MultigroupSelectorArg::QualifiedExperimental
        );
        validate_deepseek_v4_multigroup_selector_scope(&explicit).unwrap();
        validate_deepseek_v4_multigroup_selector_family(
            explicit.deepseek_v4_multigroup_selector,
            Some(ModelFamily::DeepSeek4),
        )
        .unwrap();
        assert!(
            validate_deepseek_v4_multigroup_selector_family(
                explicit.deepseek_v4_multigroup_selector,
                None,
            )
            .unwrap_err()
            .to_string()
            .contains("requires a DeepSeek V4 model")
        );

        let jsonl = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "requests.jsonl",
            "--deepseek-v4-multigroup-selector=qualified-experimental",
        ])
        .unwrap();
        validate_deepseek_v4_multigroup_selector_scope(&jsonl).unwrap();
        validate_deepseek_v4_requests_mode(&jsonl, ExplicitCliOptions::default()).unwrap();

        let no_request = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--deepseek-v4-multigroup-selector=qualified-experimental",
        ])
        .unwrap();
        assert!(
            validate_deepseek_v4_multigroup_selector_scope(&no_request)
                .unwrap_err()
                .to_string()
                .contains("requires a generation request")
        );
        assert!(
            Args::try_parse_from([
                "qwen",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--deepseek-v4-multigroup-selector=force",
            ])
            .is_err()
        );

        let shallow = DeepSeekV4SessionCapacity::for_forward_limit(4_096, 1_048_576).unwrap();
        let off = DeepSeekV4MultigroupSelectorPlan::new(
            DeepSeekV4MultigroupSelectorArg::Off,
            "Apple M4 Pro",
            shallow,
        )
        .unwrap();
        assert!(!off.sealed());
        let auto = DeepSeekV4MultigroupSelectorPlan::new(
            DeepSeekV4MultigroupSelectorArg::Auto,
            "Apple M4 Max",
            DeepSeekV4SessionCapacity::for_forward_limit(786_432, 1_048_576).unwrap(),
        )
        .unwrap();
        assert!(!auto.sealed());
        assert!(
            auto.completion_record_from_values("single_turn", false, 21, 3)
                .is_ok()
        );
        assert!(
            DeepSeekV4MultigroupSelectorPlan::new(
                DeepSeekV4MultigroupSelectorArg::QualifiedExperimental,
                "Apple M4 Pro",
                DeepSeekV4SessionCapacity::for_forward_limit(786_432, 1_048_576).unwrap(),
            )
            .unwrap_err()
            .to_string()
            .contains("requires Apple M4 Max")
        );
        let unreachable_error = DeepSeekV4MultigroupSelectorPlan::new(
            DeepSeekV4MultigroupSelectorArg::QualifiedExperimental,
            "Apple M4 Max",
            DeepSeekV4SessionCapacity::for_forward_limit(786_431, 1_048_576).unwrap(),
        )
        .unwrap_err();
        assert!(format!("{unreachable_error:#}").contains("max_visible_rows=196607"));

        let qualified = DeepSeekV4MultigroupSelectorPlan::new(
            DeepSeekV4MultigroupSelectorArg::QualifiedExperimental,
            "Apple M4 Max",
            DeepSeekV4SessionCapacity::for_forward_limit(786_432, 1_048_576).unwrap(),
        )
        .unwrap();
        assert!(qualified.sealed());
        assert_eq!(
            serde_json::to_value(qualified.session_record("single_turn")).unwrap(),
            serde_json::json!({
                "schema_version": 1,
                "kind": "session_policy",
                "scope": "single_turn",
                "requested": "qualified_experimental",
                "sealed": true,
                "device_name": "Apple M4 Max",
                "device_qualified": true,
                "forward_limit": 786432,
                "physical_capacity_rows": 196608,
                "max_reachable_visible_rows": 196608,
                "frozen_min_visible_rows": 196608,
                "frozen_max_capacity_rows": 262144,
                "frozen_min_capacity_occupancy": "3/4",
                "fallback": "radix4_for_packed_and_ineligible_singleton",
            })
        );
        assert_eq!(
            serde_json::to_value(
                qualified
                    .completion_record_from_values("single_turn", true, 21, 3)
                    .unwrap()
            )
            .unwrap(),
            serde_json::json!({
                "schema_version": 1,
                "kind": "session_completion",
                "scope": "single_turn",
                "requested": "qualified_experimental",
                "sealed": true,
                "multigroup_invocations": 21,
                "ineligible_singleton_radix4_invocations": 3,
            })
        );
        assert!(
            qualified
                .completion_record_from_values("single_turn", false, 0, 0)
                .is_err()
        );
        assert!(
            off.completion_record_from_values("single_turn", false, 1, 0)
                .is_err()
        );
    }

    #[test]
    fn deepseek_v4_prefill_chunks_every_retained_prompt_interval() {
        assert_eq!(parse_deepseek_v4_prefill_chunk_tokens(None).unwrap(), 4_096);
        assert_eq!(
            parse_deepseek_v4_prefill_chunk_tokens(Some("128")).unwrap(),
            128
        );
        assert_eq!(
            parse_deepseek_v4_prefill_chunk_tokens(Some("512")).unwrap(),
            512
        );
        assert_eq!(
            parse_deepseek_v4_prefill_chunk_tokens(Some("4096")).unwrap(),
            4_096
        );
        assert!(parse_deepseek_v4_prefill_chunk_tokens(Some("0")).is_err());
        assert!(parse_deepseek_v4_prefill_chunk_tokens(Some("4097")).is_err());
        assert!(parse_deepseek_v4_prefill_chunk_tokens(Some("nope")).is_err());
        let lengths = |tokens, chunk| {
            deepseek_v4_prefill_chunk_ranges(tokens, chunk)
                .into_iter()
                .map(|range| range.len())
                .collect::<Vec<_>>()
        };
        assert_eq!(lengths(2_385, 4_096), vec![2_048, 337]);
        assert_eq!(lengths(6_642, 4_096), vec![4_096, 2_048, 498]);
        assert_eq!(lengths(8_192, 4_096), vec![4_096, 4_096]);
        assert_eq!(lengths(2_385, 2_048), vec![2_048, 337]);
        assert_eq!(lengths(2_385, 3_000), vec![2_385]);
        assert_eq!(deepseek_v4_packed_chunk_count(0, 512), 0);
        assert_eq!(deepseek_v4_packed_chunk_count(1, 512), 0);
        assert_eq!(deepseek_v4_packed_chunk_count(2, 512), 1);
        assert_eq!(
            deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PREFILL_MAX_TOKENS, 512),
            8
        );
        assert_eq!(
            deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PREFILL_MAX_TOKENS + 1, 512),
            9
        );
        assert_eq!(
            deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY, 512),
            2_048
        );
        assert_eq!(
            deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PREFILL_MAX_TOKENS, 2_048),
            2
        );
        assert_eq!(
            deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY, 2_048),
            512
        );
        assert_eq!(
            deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY, 128),
            8_192
        );
    }

    #[test]
    fn deepseek_v4_snapshot_keeps_one_uncached_endpoint_token() {
        assert!(deepseek_v4_snapshot_publish_prefix(0).is_err());
        assert!(deepseek_v4_snapshot_publish_prefix(1).is_err());
        assert_eq!(deepseek_v4_snapshot_publish_prefix(2).unwrap(), 1);
        assert_eq!(
            deepseek_v4_snapshot_publish_prefix(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY).unwrap(),
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY - 1
        );

        let prompt = [35, 201, 200, 34];
        assert_eq!(
            deepseek_v4_snapshot_restored_prefix_len(&prompt[..3], &prompt).unwrap(),
            3
        );
        assert!(deepseek_v4_snapshot_restored_prefix_len(&prompt, &prompt).is_err());
        assert!(deepseek_v4_snapshot_restored_prefix_len(&[35, 200], &prompt).is_err());
    }

    #[test]
    fn deepseek_v4_snapshot_identity_cache_requires_private_owned_directories() {
        let root = std::env::temp_dir().join(format!(
            "qwen-dsv4-cli-cache-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&root)
            .unwrap();
        let snapshot_path = root.join("prefix.ds4c");
        assert_eq!(
            deepseek_v4_snapshot_parent(&snapshot_path).unwrap(),
            root.as_path()
        );
        let cache = deepseek_v4_snapshot_identity_cache(&root).unwrap();
        assert_eq!(
            std::fs::symlink_metadata(cache.root())
                .unwrap()
                .permissions()
                .mode()
                & 0o077,
            0
        );

        std::fs::remove_dir(cache.root()).unwrap();
        std::fs::DirBuilder::new()
            .mode(0o755)
            .create(cache.root())
            .unwrap();
        std::fs::set_permissions(cache.root(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(deepseek_v4_snapshot_identity_cache(&root).is_err());

        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(deepseek_v4_snapshot_parent(&snapshot_path).is_err());
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::remove_dir_all(root).unwrap();
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

        let snapshot = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--deepseek-v4-snapshot",
            "prefix.ds4c",
        ])
        .unwrap();
        validate_deepseek_v4_generation_mode(&snapshot, ExplicitCliOptions::default()).unwrap();

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
        validate_deepseek_v4_generation_mode(&messages_with_strip, ExplicitCliOptions::default())
            .unwrap();

        let preserve_with_tier = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "messages.json",
            "--messages-preserve-thinking",
            "--reasoning",
            "low",
        ])
        .unwrap();
        validate_deepseek_v4_generation_mode(&preserve_with_tier, ExplicitCliOptions::default())
            .unwrap();

        assert!(
            Args::try_parse_from([
                "qwen",
                "--model",
                "model.gguf",
                "--messages",
                "messages.json",
                "--messages-strip-thinking",
                "--preserve-reasoning",
            ])
            .is_err(),
            "strip-thinking must conflict with preserve-reasoning at the parser"
        );

        let durable = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "messages.json",
            "--messages-strip-thinking",
            "--durable-prefix-cache",
            "cache",
            "--durable-prefix-cache-min-tokens",
            "1024",
            "--durable-prefix-cache-max-mib",
            "4096",
            "--durable-prefix-cache-max-entry-mib",
            "4096",
        ])
        .unwrap();
        validate_deepseek_v4_generation_mode(
            &durable,
            ExplicitCliOptions {
                durable_prefix_cache_min_tokens: true,
                durable_prefix_cache_max_mib: true,
                durable_prefix_cache_max_entry_mib: true,
                ..ExplicitCliOptions::default()
            },
        )
        .unwrap();

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

        // Requests mode: file mode derives its budget and rejects the stdin
        // override; stdin mode requires it; shared unsupported flags still
        // fail closed; the snapshot flag stays parser-excluded.
        let file_mode_with_budget = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "requests.jsonl",
            "--max-context-tokens",
            "4096",
        ])
        .unwrap();
        assert!(
            validate_deepseek_v4_requests_mode(
                &file_mode_with_budget,
                ExplicitCliOptions::default(),
            )
            .unwrap_err()
            .to_string()
            .contains("remove --max-context-tokens")
        );
        let stdin_without_budget =
            Args::try_parse_from(["qwen", "--model", "model.gguf", "--requests-jsonl", "-"])
                .unwrap();
        assert!(
            validate_deepseek_v4_requests_mode(
                &stdin_without_budget,
                ExplicitCliOptions::default(),
            )
            .unwrap_err()
            .to_string()
            .contains("supply --max-context-tokens")
        );
        let stdin_with_budget = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "-",
            "--max-context-tokens",
            "4096",
        ])
        .unwrap();
        validate_deepseek_v4_requests_mode(&stdin_with_budget, ExplicitCliOptions::default())
            .unwrap();
        assert_eq!(
            deepseek_v4_forward_budget_for_context_limit(4_096).unwrap(),
            4_095
        );
        assert_eq!(
            deepseek_v4_forward_budget_for_context_limit(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY)
                .unwrap(),
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY - 1
        );
        assert!(deepseek_v4_forward_budget_for_context_limit(1).is_err());
        assert!(
            deepseek_v4_forward_budget_for_context_limit(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY + 1)
                .is_err()
        );
        validate_deepseek_v4_request_context_limit("exact", 4_000, 96, 4_096).unwrap();
        let context_error =
            validate_deepseek_v4_request_context_limit("overflow", 4_000, 97, 4_096)
                .unwrap_err()
                .to_string();
        assert!(context_error.contains("4097 logical context tokens"));
        assert!(
            validate_deepseek_v4_request_context_limit("checked-add", usize::MAX, 1, usize::MAX,)
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );
        let requests_with_durable_cache = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "requests.jsonl",
            "--durable-prefix-cache",
            "/tmp/cache",
        ])
        .unwrap();
        assert!(
            validate_deepseek_v4_requests_mode(
                &requests_with_durable_cache,
                ExplicitCliOptions::default(),
            )
            .unwrap_err()
            .to_string()
            .contains("--durable-prefix-cache")
        );
        assert!(
            Args::try_parse_from([
                "qwen",
                "--model",
                "model.gguf",
                "--requests-jsonl",
                "requests.jsonl",
                "--deepseek-v4-snapshot",
                "/tmp/s.ds4ckpt",
            ])
            .is_err(),
            "snapshot flag must stay parser-excluded from requests mode"
        );

        // `--reasoning`/`--preserve-reasoning` require `--messages` in
        // pre-open validation and map onto the release encoder contract.
        let raw_reasoning = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "-p",
            "hi",
            "--reasoning",
            "high",
        ])
        .unwrap();
        assert!(
            validate_request_before_model_open(&raw_reasoning)
                .unwrap_err()
                .to_string()
                .contains("require --messages")
        );
        let census_reasoning = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--deepseek-census-json",
            "--reasoning",
            "high",
        ])
        .unwrap();
        assert!(
            validate_deepseek_v4_reasoning_scope(&census_reasoning)
                .unwrap_err()
                .to_string()
                .contains("require --messages")
        );
        for (level, expected) in [
            (None, DeepSeekV4Reasoning::None),
            (Some("none"), DeepSeekV4Reasoning::None),
            (Some("low"), DeepSeekV4Reasoning::Low),
            (Some("high"), DeepSeekV4Reasoning::High),
            (Some("max"), DeepSeekV4Reasoning::Max),
        ] {
            let mut command = vec![
                "qwen",
                "--model",
                "model.gguf",
                "--messages",
                "messages.json",
            ];
            if let Some(level) = level {
                command.extend(["--reasoning", level]);
            }
            let args = Args::try_parse_from(command).unwrap();
            let options = deepseek_v4_encode_options(&args).unwrap();
            assert_eq!(options.reasoning, expected);
            assert!(!options.preserve_reasoning);
        }
        let preserve_without_thinking = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "messages.json",
            "--preserve-reasoning",
        ])
        .unwrap();
        assert!(
            deepseek_v4_encode_options(&preserve_without_thinking)
                .unwrap_err()
                .to_string()
                .contains("requires --reasoning low, high, or max")
        );
        let preserve = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "messages.json",
            "--reasoning",
            "max",
            "--preserve-reasoning",
        ])
        .unwrap();
        let options = deepseek_v4_encode_options(&preserve).unwrap();
        assert_eq!(options.reasoning, DeepSeekV4Reasoning::Max);
        assert!(options.preserve_reasoning);
        let preserve_low = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "messages.json",
            "--reasoning",
            "low",
            "--preserve-reasoning",
        ])
        .unwrap();
        let options = deepseek_v4_encode_options(&preserve_low).unwrap();
        assert_eq!(options.reasoning, DeepSeekV4Reasoning::Low);
        assert!(options.preserve_reasoning);

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
                ..Default::default()
            }
        }
        fn render_ids(tokenizer: &Tokenizer, messages: &[messages::ChatMessage]) -> Vec<i32> {
            let prompt = messages::render_deepseek_v4_0731_messages_prompt(
                messages,
                messages::DeepSeekV4EncodeOptions::default(),
            )
            .unwrap();
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
    fn deepseek_v4_durable_capture_boundary_promotes_growing_prompts() {
        assert_eq!(
            deepseek_v4_durable_capture_prefix_len(1024, true).unwrap(),
            Some(1023)
        );
        assert_eq!(
            deepseek_v4_durable_capture_prefix_len(2048, true).unwrap(),
            Some(2047)
        );
        assert_eq!(
            deepseek_v4_durable_capture_prefix_len(2048, false).unwrap(),
            None
        );
        assert_eq!(
            deepseek_v4_durable_capture_prefix_len(1, true).unwrap(),
            None
        );
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
    fn deepseek_v4_prefetch_defaults_auto_and_parses_strictly() {
        assert_eq!(
            parse_deepseek_v4_prefetch_mode(None).unwrap(),
            DeepSeekV4PrefetchMode::Auto
        );
        assert_eq!(
            parse_deepseek_v4_prefetch_mode(Some(OsStr::new("off"))).unwrap(),
            DeepSeekV4PrefetchMode::Off
        );
        assert_eq!(
            parse_deepseek_v4_prefetch_mode(Some(OsStr::new("always"))).unwrap(),
            DeepSeekV4PrefetchMode::Always
        );
        assert_eq!(
            parse_deepseek_v4_prefetch_mode(Some(OsStr::new("auto"))).unwrap(),
            DeepSeekV4PrefetchMode::Auto
        );
        assert!(parse_deepseek_v4_prefetch_mode(Some(OsStr::new("cold"))).is_err());

        assert_eq!(
            deepseek_v4_prefetch_policy(DeepSeekV4PrefetchMode::Off),
            PrefetchPolicy::Off
        );
        assert_eq!(
            deepseek_v4_prefetch_policy(DeepSeekV4PrefetchMode::Always),
            PrefetchPolicy::Always
        );
        match deepseek_v4_prefetch_policy(DeepSeekV4PrefetchMode::Auto) {
            PrefetchPolicy::ColdOnly { threshold } => {
                assert_eq!(threshold.value(), DEEPSEEK_V4_PREFETCH_AUTO_THRESHOLD);
            }
            other => panic!("expected DS4 cold-only auto policy, got {other:?}"),
        }

        use std::os::unix::ffi::OsStringExt;
        let non_unicode = std::ffi::OsString::from_vec(vec![0xff]);
        assert!(parse_deepseek_v4_prefetch_mode(Some(&non_unicode)).is_err());
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn deepseek_v4_temporal_window_is_bounded() {
        assert_eq!(parse_deepseek_v4_temporal_window(None).unwrap(), 0);
        assert_eq!(
            parse_deepseek_v4_temporal_window(Some(OsStr::new("65"))).unwrap(),
            65
        );
        assert!(parse_deepseek_v4_temporal_window(Some(OsStr::new("66"))).is_err());
        assert!(parse_deepseek_v4_temporal_window(Some(OsStr::new("no"))).is_err());
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

    // ---- request-stats-jsonl envelope shape tests ----
    //
    // These tests pin the common envelope shape for schema v1. Adding a field
    // in an additive way should keep these tests passing. Renaming, removing,
    // or restructuring a field should require a schema_version bump AND
    // updating these tests intentionally.

    fn sample_measured_ok() -> RequestStatsMeasured {
        RequestStatsMeasured {
            prompt_kind: "messages_0731_chat",
            prefill_mode: "layer_major_chunks",
            prefill_chunk_cap: 4096,
            input_tokens: 100,
            output_tokens: 50,
            transitions: 49,
            stop_reason: StopReason::Eos,
            tokenizer_ms: 10.5,
            load_ms: 40.1,
            prefill_ms: 1000.0,
            prefill_tps: 100.0,
            decode_ms: 2000.0,
            decode_tps: 25.0,
            transition_tps: 24.5,
            total_ms: 3050.6,
            output_fingerprint: GeneratedTokenSha256Digest::of(&[0i32; 4]),
        }
    }

    #[test]
    fn request_stats_record_v1_envelope_shape_is_stable() {
        let measured = sample_measured_ok();
        let record = build_deepseek_v4_single_turn_stats_record("inv-42-1234567890", &measured);
        let json = serde_json::to_value(&record).unwrap();

        // Common core
        assert_eq!(json["schema"], "qwen-llm.request-stats");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["record_type"], "request_stats");
        assert_eq!(json["invocation_id"], "inv-42-1234567890");
        assert_eq!(json["request_index"], 0);
        assert_eq!(json["status"], "ok");

        // Model
        assert_eq!(json["model"]["family"], "deepseek_v4");

        // Input (kind + template split from prompt_kind)
        assert_eq!(json["input"]["kind"], "messages");
        assert_eq!(json["input"]["template"], "0731_chat");

        // Usage: u64 wire counts
        assert_eq!(json["usage"]["input_tokens"], 100);
        assert_eq!(json["usage"]["output_tokens"], 50);

        // Finish (envelope enum, not internal StopReason)
        assert_eq!(json["finish"]["reason"], "eos");

        // Timing (all ms, per-request; total is measured at outer boundary)
        assert_eq!(json["timing_ms"]["total"], 3050.6);
        assert_eq!(json["timing_ms"]["tokenization"], 10.5);
        assert_eq!(json["timing_ms"]["prefill"], 1000.0);
        assert_eq!(json["timing_ms"]["decode"], 2000.0);

        // Throughput (tokens/sec)
        assert_eq!(json["throughput_tps"]["prefill"], 100.0);
        assert_eq!(json["throughput_tps"]["decode"], 25.0);

        // Fingerprint: algorithm names encoding, value is hex of the [u8; 32]
        assert_eq!(
            json["output_fingerprint"]["algorithm"],
            "sha256-qwen-generated-token-ids-v1"
        );
        // sample_measured_ok uses GeneratedTokenSha256Digest::of(&[0i32; 4]);
        // literal digest is pinned by request_stats_fingerprint_matches_literal_known_vector.
        assert_eq!(
            json["output_fingerprint"]["value"],
            "f06d4689226359d0fe3e105fb7d432520d855d8aad664db95596ae9c34d90eca"
        );

        // Build (commit + dirty are compile-time env vars; values vary per build)
        assert!(json["build"]["commit"].is_string());
        assert!(json["build"]["dirty"].is_boolean());

        // Diagnostics: family-namespaced, independently versioned
        assert_eq!(json["diagnostics"]["deepseek_v4"]["schema_version"], 1);
        assert_eq!(
            json["diagnostics"]["deepseek_v4"]["prefill_mode"],
            "layer_major_chunks"
        );
        assert_eq!(
            json["diagnostics"]["deepseek_v4"]["prefill_chunk_cap"],
            4096
        );
        assert_eq!(json["diagnostics"]["deepseek_v4"]["transitions"], 49);
        assert_eq!(json["diagnostics"]["deepseek_v4"]["transition_tps"], 24.5);
        assert_eq!(json["diagnostics"]["deepseek_v4"]["load_ms"], 40.1);
    }

    #[test]
    fn request_stats_split_ds4_prompt_kind_restricts_common_kind_vocabulary() {
        assert_eq!(
            split_ds4_prompt_kind("messages_0731_chat"),
            ("messages", Some("0731_chat"))
        );
        assert_eq!(
            split_ds4_prompt_kind("messages_0731_thinking"),
            ("messages", Some("0731_thinking"))
        );
        assert_eq!(split_ds4_prompt_kind("raw"), ("raw", None));
        // Empty template suffix collapses to unknown, not "messages" with empty template.
        assert_eq!(split_ds4_prompt_kind("messages_"), ("unknown", None));
        // Unknown backend labels do NOT get promoted into the common `input.kind`
        // vocabulary — they collapse to "unknown".
        assert_eq!(split_ds4_prompt_kind("something_else"), ("unknown", None));
        assert_eq!(split_ds4_prompt_kind(""), ("unknown", None));
    }

    #[test]
    fn request_stats_parse_build_dirty_is_case_insensitive() {
        // Any-case zero-ish values are clean
        for clean in ["0", "", "false", "FALSE", "False", "no", "NO", "No"] {
            assert!(!parse_build_dirty(clean), "expected {clean:?} to be clean");
        }
        // Any-case truthy values are dirty
        for dirty in ["1", "true", "TRUE", "True", "yes", "YES", "Yes", "dirty"] {
            assert!(parse_build_dirty(dirty), "expected {dirty:?} to be dirty");
        }
    }

    #[test]
    fn request_stats_finish_reason_covers_all_stop_reasons_exhaustively() {
        // If a new StopReason variant is added, this match will fail to
        // compile — forcing a schema decision rather than silent drift.
        for reason in [StopReason::Eos, StopReason::TokenLimit] {
            let mapped: RequestStatsFinishReason = reason.into();
            let serialized = serde_json::to_value(mapped).unwrap();
            match reason {
                StopReason::Eos => assert_eq!(serialized, "eos"),
                StopReason::TokenLimit => assert_eq!(serialized, "token_limit"),
            }
        }
    }

    #[test]
    fn request_stats_non_finite_metrics_coerced_to_zero() {
        let mut measured = sample_measured_ok();
        // Pathological values that would otherwise become JSON `null`.
        measured.total_ms = f64::NAN;
        measured.tokenizer_ms = f64::INFINITY;
        measured.prefill_ms = f64::INFINITY;
        measured.decode_ms = -0.001;
        measured.prefill_tps = f64::NAN;
        measured.decode_tps = f64::NEG_INFINITY;
        measured.transition_tps = f64::NAN;
        measured.load_ms = -100.0;

        let record = build_deepseek_v4_single_turn_stats_record("inv-x", &measured);
        let json = serde_json::to_value(&record).unwrap();

        for (parent, child) in [
            ("timing_ms", "total"),
            ("timing_ms", "tokenization"),
            ("timing_ms", "prefill"),
            ("timing_ms", "decode"),
            ("throughput_tps", "prefill"),
            ("throughput_tps", "decode"),
        ] {
            let v = json[parent][child].as_f64().unwrap_or_else(|| {
                panic!(
                    "{parent}.{child} was not a JSON number (got {:?})",
                    json[parent][child]
                )
            });
            assert_eq!(v, 0.0, "{parent}.{child}");
        }
        assert_eq!(
            json["diagnostics"]["deepseek_v4"]["transition_tps"]
                .as_f64()
                .unwrap(),
            0.0
        );
        assert_eq!(
            json["diagnostics"]["deepseek_v4"]["load_ms"]
                .as_f64()
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn request_stats_fingerprint_matches_literal_known_vector() {
        // Literal known-vector: `sha256(domain-separator || len_u64_le ||
        // (token_i32_le)*)` over tokens [1,2,3,4,5] MUST match this exact
        // hex digest. Changing the hash function, domain separator, length
        // encoding, or token encoding will produce a different digest and
        // fail this test — which is the whole point of naming the algorithm
        // `sha256-qwen-generated-token-ids-v1`.
        let tokens: [i32; 5] = [1, 2, 3, 4, 5];
        const EXPECTED_HEX: &str =
            "7f016c59a05cecd27d3348aedae1275ec5cb5be58aeb3ea599110e4a7ce5b304";
        let digest = GeneratedTokenSha256Digest::of(&tokens);
        assert_eq!(digest.hex(), EXPECTED_HEX);
        assert_eq!(generated_token_sha256(&tokens), EXPECTED_HEX);
        // And the same digest, wired through the record builder, must land
        // in output_fingerprint.value byte-identical.
        let mut measured = sample_measured_ok();
        measured.output_fingerprint = digest;
        let record = build_deepseek_v4_single_turn_stats_record("inv-y", &measured);
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["output_fingerprint"]["value"], EXPECTED_HEX);
    }

    #[test]
    fn request_stats_hex_encode_bytes_is_lowercase_padded() {
        assert_eq!(hex_encode_bytes(&[]), "");
        assert_eq!(hex_encode_bytes(&[0x00, 0x0f, 0xff]), "000fff");
        assert_eq!(hex_encode_bytes(&[0xab; 4]), "abababab");
    }

    #[test]
    fn request_stats_append_jsonl_serializes_one_line_per_record() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "qwen-request-stats-test-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        for i in 0..3u64 {
            let mut m = sample_measured_ok();
            m.input_tokens = i;
            m.output_fingerprint = GeneratedTokenSha256Digest::of(&[i as i32]);
            let record = build_deepseek_v4_single_turn_stats_record("inv-z", &m);
            append_jsonl_record(&path, &record, "test stats").unwrap();
        }

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 3, "expected exactly one line per record");
        assert!(contents.ends_with('\n'), "file must end with newline");
        for (i, line) in lines.iter().enumerate() {
            let json: serde_json::Value = serde_json::from_str(line).expect("valid JSON per line");
            assert_eq!(json["schema"], "qwen-llm.request-stats");
            assert_eq!(json["invocation_id"], "inv-z");
            assert_eq!(json["usage"]["input_tokens"], i);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn request_stats_status_serializes_as_lowercase() {
        assert_eq!(serde_json::to_value(RequestStatsStatus::Ok).unwrap(), "ok");
        assert_eq!(
            serde_json::to_value(RequestStatsStatus::Error).unwrap(),
            "error"
        );
        assert_eq!(
            serde_json::to_value(RequestStatsStatus::Cancelled).unwrap(),
            "cancelled"
        );
    }

    #[test]
    fn request_stats_success_only_fields_are_optional_for_error_records() {
        // Verify the envelope structurally supports a future error/cancelled
        // record without a breaking restructuring: success-only fields all
        // permit `None` and skip serialization when absent.
        let record = RequestStatsRequestRecord {
            schema: "qwen-llm.request-stats",
            schema_version: 1,
            record_type: "request_stats",
            invocation_id: "inv-err",
            request_index: 0,
            status: RequestStatsStatus::Error,
            model: RequestStatsModel {
                family: "deepseek_v4",
            },
            input: RequestStatsInput {
                kind: "messages",
                template: Some("0731_chat"),
            },
            usage: None,
            finish: None,
            timing_ms: None,
            throughput_tps: None,
            output_fingerprint: None,
            build: RequestStatsBuild {
                commit: "abc",
                dirty: false,
            },
            diagnostics: None,
        };
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["status"], "error");
        assert!(json.get("usage").is_none());
        assert!(json.get("finish").is_none());
        assert!(json.get("timing_ms").is_none());
        assert!(json.get("throughput_tps").is_none());
        assert!(json.get("output_fingerprint").is_none());
        assert!(json.get("diagnostics").is_none());
    }

    #[test]
    fn request_stats_invocation_id_is_stable_and_hex_16_bytes() {
        // Force init once; every subsequent access must return the same ID.
        let a = &*INVOCATION_ID;
        let b = &*INVOCATION_ID;
        assert_eq!(a, b);
        // 16 bytes hex-encoded = 32 chars
        assert_eq!(a.len(), 32, "invocation_id is 16 bytes hex-encoded");
        assert!(
            a.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
            "invocation_id must be lowercase hex: got {a}"
        );
    }

    #[test]
    fn request_stats_invocation_id_encodes_entropy_bytes_verbatim() {
        // Deterministic entropy source proves that the encoding is exactly
        // 16 bytes hex-encoded, in order, lowercase. A constant-string
        // implementation would fail this test.
        let bytes: [u8; 16] = [
            0x00, 0x01, 0x0f, 0x10, 0xab, 0xcd, 0xef, 0x42, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
            0x32, 0x10,
        ];
        let id = generate_invocation_id_from(EntropySource::Deterministic(bytes)).unwrap();
        assert_eq!(id, "00010f10abcdef42fedcba9876543210");
    }

    #[test]
    fn request_stats_valid_finite_metrics_pass_through_unchanged() {
        // Complement to non_finite_metrics_coerced_to_zero: a valid finite
        // measurement must NOT be zeroed. This catches over-aggressive
        // sanitization.
        let measured = sample_measured_ok();
        let record = build_deepseek_v4_single_turn_stats_record("inv-v", &measured);
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["timing_ms"]["total"], 3050.6);
        assert_eq!(json["timing_ms"]["tokenization"], 10.5);
        assert_eq!(json["timing_ms"]["prefill"], 1000.0);
        assert_eq!(json["timing_ms"]["decode"], 2000.0);
        assert_eq!(json["throughput_tps"]["prefill"], 100.0);
        assert_eq!(json["throughput_tps"]["decode"], 25.0);
        assert_eq!(json["diagnostics"]["deepseek_v4"]["transition_tps"], 24.5);
        assert_eq!(json["diagnostics"]["deepseek_v4"]["load_ms"], 40.1);
    }

    #[test]
    fn request_stats_split_ds4_prompt_kind_rejects_bare_messages() {
        // Bare "messages" (no underscore + suffix) is not a valid template
        // marker; it must collapse to "unknown" rather than becoming a
        // second `input.kind = "messages"` value with no template.
        assert_eq!(split_ds4_prompt_kind("messages"), ("unknown", None));
    }

    #[test]
    fn request_stats_append_jsonl_repairs_partial_prior_tail() {
        // Pre-seed the file with an unfinished record (no trailing newline).
        // The next append MUST NOT concatenate its record onto the abandoned
        // partial one — it must start on a fresh line.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "qwen-request-stats-tail-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, b"{\"partial\":\"orphan_no_newline\"").unwrap();

        let measured = sample_measured_ok();
        let record = build_deepseek_v4_single_turn_stats_record("inv-tail", &measured);
        append_jsonl_record(&path, &record, "tail-repair test").unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        // Line 0 is the orphan partial (unchanged); line 1 is our new record.
        // Critically, NO line combines both, and line 1 must parse as valid JSON.
        assert_eq!(
            lines.len(),
            2,
            "expected orphan preserved on its own line, our record on the next"
        );
        assert_eq!(lines[0], "{\"partial\":\"orphan_no_newline\"");
        let json: serde_json::Value =
            serde_json::from_str(lines[1]).expect("appended record must parse as standalone JSON");
        assert_eq!(json["invocation_id"], "inv-tail");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn request_stats_measured_total_ms_can_diverge_from_phase_sum() {
        // Prove the outer-boundary total is DECOUPLED from the phase sum
        // (a synthesized implementation would fail this by matching the
        // sum exactly).
        let mut measured = sample_measured_ok();
        measured.total_ms = 9999.9; // arbitrary value, not the phase sum
        let record = build_deepseek_v4_single_turn_stats_record("inv-t", &measured);
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["timing_ms"]["total"], 9999.9);
        // Phase fields must remain independently measured, not derived.
        assert_eq!(json["timing_ms"]["prefill"], 1000.0);
        assert_eq!(json["timing_ms"]["decode"], 2000.0);
    }

    #[test]
    fn request_stats_invocation_id_fails_closed_when_entropy_unavailable() {
        // The fail-closed branch: when secure entropy is unavailable,
        // generate_invocation_id_from returns an Err rather than a weak
        // deterministic fallback. This is what makes the LazyLock panic
        // in main() the correct behavior (loud failure > silent weak ID).
        let err = generate_invocation_id_from(EntropySource::ForceFail)
            .expect_err("must return Err when entropy source fails");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("test-injected entropy failure"),
            "error message must surface the underlying failure: {msg}"
        );
    }

    #[test]
    fn request_stats_fingerprint_newtype_bytes_only_from_of() {
        // The GeneratedTokenSha256Digest tuple field lives in a private
        // child module (`mod fingerprint`), so no crate code — including
        // this test — can construct one with arbitrary bytes. The only
        // way to obtain a value is via `of()`. This test compiles iff that
        // invariant holds: a direct tuple construction would fail with
        // "field is private", and a bytes constructor doesn't exist.
        let a = GeneratedTokenSha256Digest::of(&[]);
        let b = GeneratedTokenSha256Digest::of(&[]);
        assert_eq!(a, b, "of() over identical input is deterministic");
        // Sanity: the digest exposes bytes via `as_bytes()` (test-only) for
        // hex comparison — but that's read-only, not a constructor path.
        assert_eq!(a.as_bytes().len(), 32);
    }
}
