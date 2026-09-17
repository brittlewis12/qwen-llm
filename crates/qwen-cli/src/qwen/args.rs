//! Flat `qwen` argument surface and explicit-option tracking.

use super::*;

#[derive(Parser, Debug)]
#[command(
    name = "qwen",
    version,
    about = "Fast local Qwen, DeepSeek, and Muse inference on Apple Silicon",
    args_conflicts_with_subcommands = true,
    after_help = "Examples:\n  qwen run -m MODEL --user 'Explain this'\n  qwen run -m MODEL --system 'Be concise' --user 'Explain this'\n  qwen run -m MODEL --user -\n  qwen run -m MODEL --messages -\n  qwen run -m MODEL --raw-prompt '<exact model input>'\n  qwen run -m Qwen3.6-35B-A3B.gguf --user 'Explain this' --no-thinking\n\n--no-thinking controls model prompt rendering; it does not hide CLI diagnostics.\nCLI diagnostic suppression is not currently available.\nFor resident JSONL batching and expanded legacy/research help, run:\n  qwen --help\nLegacy flags shown there are flat and cannot be combined with qwen run.",
    after_long_help = "Modern examples:\n  qwen run -m MODEL --user 'Explain this'\n  qwen run -m MODEL --messages -\n  qwen run -m MODEL --raw-prompt '<exact model input>'\n\nLegacy/research examples (flat; do not combine with qwen run):\n  qwen -m MODEL --prompt '<raw model input>'\n  qwen -m MODEL --requests-jsonl requests.jsonl\n\n--no-thinking controls model prompt rendering; it does not hide CLI diagnostics.\nCLI diagnostic suppression is not currently available."
)]
pub(crate) struct Args {
    #[command(subcommand)]
    pub(crate) command: Option<cli::Command>,

    #[arg(skip)]
    pub(crate) prepared_prompt: Option<PreparedPrompt>,

    /// Path to a Qwen, DeepSeek V4, or Muse Glimmer GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: Option<std::path::PathBuf>,

    /// Print device info and exit.
    #[arg(long)]
    pub(crate) info: bool,

    /// Print the deterministic DeepSeek V4 schema/quant census as JSON.
    #[arg(
        long,
        hide_short_help = true,
        requires = "model",
        conflicts_with_all = ["info", "prompt", "prompt_file", "messages", "requests_jsonl"]
    )]
    pub(crate) deepseek_census_json: bool,

    /// Raw prompt text for single-turn generation.
    #[arg(short = 'p', long, hide_short_help = true, conflicts_with_all = ["prompt_file", "messages"])]
    pub(crate) prompt: Option<String>,

    /// Read raw prompt text from a file.
    #[arg(long, hide_short_help = true, conflicts_with_all = ["prompt", "messages"])]
    pub(crate) prompt_file: Option<PathBuf>,

    /// Render a bare or wrapped JSON messages file with the model-family encoder.
    #[arg(long, hide_short_help = true, conflicts_with_all = ["prompt", "prompt_file", "requests_jsonl"])]
    pub(crate) messages: Option<PathBuf>,

    /// Render only the first N messages.
    #[arg(long, hide_short_help = true, requires = "messages")]
    pub(crate) messages_max: Option<usize>,

    /// Preserve assistant `<think>...</think>` history.
    #[arg(
        long,
        hide_short_help = true,
        requires = "messages",
        conflicts_with = "messages_strip_thinking"
    )]
    pub(crate) messages_preserve_thinking: bool,

    /// Strip a leading assistant `<think>...</think>` block from history.
    #[arg(
        long,
        hide_short_help = true,
        requires = "messages",
        conflicts_with = "preserve_reasoning"
    )]
    pub(crate) messages_strip_thinking: bool,

    /// Do not append the assistant generation prompt after messages.
    #[arg(long, hide_short_help = true, requires = "messages")]
    pub(crate) messages_no_generation_prompt: bool,

    /// DeepSeek V4 release reasoning mode for --messages encoding
    /// (none/low/high/max; none is ordinary chat).
    #[arg(
        long,
        hide_short_help = true,
        requires = "messages",
        value_name = "LEVEL"
    )]
    pub(crate) reasoning: Option<String>,

    /// Retain reasoning across turns (DeepSeek V4 thinking modes only).
    #[arg(long, hide_short_help = true, requires = "messages")]
    pub(crate) preserve_reasoning: bool,

    /// Read JSONL request objects from a file or '-' while keeping one model loaded.
    #[arg(long, hide_short_help = true, conflicts_with_all = ["prompt", "prompt_file", "messages"])]
    pub(crate) requests_jsonl: Option<PathBuf>,

    /// Select ordinary Qwen model cache warming; `off` avoids prefetch reads.
    #[arg(long, hide_short_help = true, requires = "requests_jsonl", value_enum)]
    pub(crate) model_prefetch: Option<QwenModelPrefetchArg>,

    /// Decode equal-prompt-length JSONL cohorts with per-request token limits.
    ///
    /// Width is 8 on dense Qwen or 16 on Qwen MoE.
    #[arg(long, hide_short_help = true, requires = "requests_jsonl")]
    pub(crate) batch_size: Option<usize>,

    /// Run two independent resident JSONL requests with overlapping decode.
    #[arg(
        long,
        hide_short_help = true,
        requires = "requests_jsonl",
        conflicts_with = "batch_size"
    )]
    pub(crate) concurrency: Option<usize>,

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
    pub(crate) execution_mode: Option<execution_selector::ExecutionModeArg>,

    /// What a request that cannot be prepared (parse, shape, render,
    /// tokenize, sampling, capacity, executor constraint) does to the batch.
    ///
    /// Every attempted request produces exactly one stdout row carrying its
    /// source `line`: a success row (`status: ok`) or a failure row
    /// (`status: error`, stable `code`). `stop` emits the failure row and
    /// ends the batch; `continue` keeps going and exits nonzero at the end.
    /// File requests are prepared before the model loads, so under `stop` a
    /// bad row costs no load. `continue` is serial-lane only for now.
    #[arg(
        long,
        hide_short_help = true,
        requires = "requests_jsonl",
        value_enum,
        default_value_t = RequestErrorPolicy::Stop
    )]
    pub(crate) on_request_error: RequestErrorPolicy,

    /// Maximum number of tokens to generate.
    #[arg(short = 'n', long, hide_short_help = true, default_value_t = 64)]
    pub(crate) tokens: usize,

    /// Sampling temperature; zero is greedy. Muse omission uses its released value 1.
    #[arg(
        long = "temp",
        visible_alias = "temperature",
        hide_short_help = true,
        default_value_t = 0.0
    )]
    pub(crate) temperature: f32,

    /// Top-k cutoff; zero disables it. Muse omission uses its released value 64.
    #[arg(long, hide_short_help = true, default_value_t = 200)]
    pub(crate) top_k: usize,

    /// Nucleus cutoff; one disables it. Muse omission uses its released value .95.
    #[arg(long, hide_short_help = true, default_value_t = 1.0)]
    pub(crate) top_p: f32,

    /// Min-p cutoff; zero disables it. Muse omission uses its released value 0.
    #[arg(long, hide_short_help = true, default_value_t = 0.05)]
    pub(crate) min_p: f32,

    /// Effective deterministic seed; identical requests reuse the same stream.
    #[arg(long, hide_short_help = true, default_value_t = 42)]
    pub(crate) seed: u64,

    /// Enable experimental dense-27B Q4_K_M prompt-lookup decode.
    #[arg(long, hide_short_help = true)]
    pub(crate) prompt_lookup: bool,

    /// DFlash drafter GGUF for speculative decode. Sampled DFlash2 proposals
    /// use sparse rejection sampling against the packed target verifier.
    #[arg(long, value_name = "GGUF")]
    pub(crate) drafter: Option<PathBuf>,

    /// Prompt prefill chunk size, or `auto` for the bounded MoE allowlist.
    #[arg(long, hide_short_help = true, default_value = "1024")]
    pub(crate) prefill_chunk: PrefillChunkArg,

    /// Override sequence capacity. Defaults to prompt + generated tokens + slack.
    #[arg(long, hide_short_help = true)]
    pub(crate) max_context_tokens: Option<usize>,

    /// Select the DeepSeek V4 far-context selector policy.
    #[arg(
        long,
        hide_short_help = true,
        value_enum,
        default_value = "auto",
        requires = "model",
        conflicts_with_all = ["info", "deepseek_census_json"]
    )]
    pub(crate) deepseek_v4_multigroup_selector: DeepSeekV4MultigroupSelectorArg,

    /// Prefix-cache byte budget in MiB; oversized snapshots are retained alone.
    #[arg(long, hide_short_help = true, default_value_t = 16 * 1024)]
    pub(crate) prefix_cache_max_mib: u64,

    /// Cache this many exact prompt tokens as the reusable prefix for requests.
    #[arg(long, hide_short_help = true)]
    pub(crate) cache_prefix_tokens: Option<usize>,

    /// Auto-cache repeated JSONL prompt prefixes at or above this token length.
    #[arg(long, hide_short_help = true, default_value_t = 1024)]
    pub(crate) cache_prefix_auto_min_tokens: usize,

    /// Persist anonymous prefix checkpoints under this private directory.
    #[arg(long, hide_short_help = true)]
    pub(crate) durable_prefix_cache: Option<PathBuf>,

    /// Load or immutably publish one explicit DeepSeek V4 causal-prefix file.
    #[arg(
        long = "deepseek-v4-snapshot",
        hide_short_help = true,
        value_name = "PATH",
        conflicts_with_all = ["info", "deepseek_census_json", "requests_jsonl", "durable_prefix_cache"]
    )]
    pub(crate) deepseek_v4_snapshot: Option<PathBuf>,

    /// Aggregate durable checkpoint budget in MiB.
    #[arg(long, hide_short_help = true, default_value_t = 32 * 1024)]
    pub(crate) durable_prefix_cache_max_mib: u64,

    /// Maximum size of one encoded durable checkpoint record in MiB.
    #[arg(long, hide_short_help = true, default_value_t = 16 * 1024)]
    pub(crate) durable_prefix_cache_max_entry_mib: u64,

    /// Auto-persist one-shot prompt boundaries at or above this token length.
    #[arg(long, hide_short_help = true, default_value_t = 1024)]
    pub(crate) durable_prefix_cache_min_tokens: usize,

    /// Append per-request JSON stats for multi-request runs.
    ///
    /// Timing fields are model-internal; this JSONL mode writes each completion
    /// after full decode rather than streaming the first token to stdout.
    #[arg(long, hide_short_help = true)]
    pub(crate) request_stats: Option<PathBuf>,

    /// Append one structured record per completed single-turn request as
    /// JSONL under the common cross-family envelope
    /// (schema: qwen-llm.request-stats v1): status, usage, finish reason,
    /// timing, throughput, output fingerprint, build identity.
    ///
    /// Supported on every family's single-turn lane (`run`, `-p`,
    /// `--prompt-file`, `--messages`). Batch lanes (`--requests-jsonl`) and
    /// non-generating invocations REJECT this flag rather than silently
    /// ignoring it — fail-closed telemetry policy.
    ///
    /// The envelope has a small stable core plus namespaced backend
    /// extensions under `diagnostics.<family>`.
    #[arg(long, hide_short_help = true)]
    pub(crate) request_stats_jsonl: Option<PathBuf>,

    /// Append single-turn first-post-model-load timing rows as JSONL.
    #[arg(long, hide_short_help = true)]
    pub(crate) request_timings: Option<PathBuf>,

    /// Run one identical warm follow-up for paired request timing.
    #[arg(long, hide_short_help = true)]
    pub(crate) request_timing_warm_followup: bool,

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
    pub(crate) sampling_attribution: bool,

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
    pub(crate) sampled_structural: bool,

    /// Do not ask the tokenizer to add model-defined special tokens.
    #[arg(long, hide_short_help = true)]
    pub(crate) no_special_tokens: bool,

    /// Append a FIFO request-trace row after each completed generation.
    ///
    /// Format is compatible with `scripts/profile/replay_economics.py
    /// --request-trace`: `arrival_ms tokens id ...`.
    #[arg(long, hide_short_help = true)]
    pub(crate) trace_request: Option<PathBuf>,
}

/// Which defaulted options were given on the command line. Constructible
/// only from clap matches so admission cannot be handed a value that
/// hides a supplied option.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExplicitCliOptions {
    pub(crate) temperature: bool,
    pub(crate) top_k: bool,
    pub(crate) top_p: bool,
    pub(crate) min_p: bool,
    pub(crate) prefill_chunk: bool,
    pub(crate) prefix_cache_max_mib: bool,
    pub(crate) cache_prefix_auto_min_tokens: bool,
    pub(crate) durable_prefix_cache_max_mib: bool,
    pub(crate) durable_prefix_cache_max_entry_mib: bool,
    pub(crate) durable_prefix_cache_min_tokens: bool,
    pub(crate) deepseek_v4_multigroup_selector: bool,
}

impl ExplicitCliOptions {
    pub(crate) fn from_matches(matches: &clap::ArgMatches) -> Self {
        let command_line = |id| {
            matches.try_contains_id(id).is_ok()
                && matches.value_source(id) == Some(ValueSource::CommandLine)
        };
        Self {
            temperature: command_line("temperature"),
            top_k: command_line("top_k"),
            top_p: command_line("top_p"),
            min_p: command_line("min_p"),
            prefill_chunk: command_line("prefill_chunk"),
            prefix_cache_max_mib: command_line("prefix_cache_max_mib"),
            cache_prefix_auto_min_tokens: command_line("cache_prefix_auto_min_tokens"),
            durable_prefix_cache_max_mib: command_line("durable_prefix_cache_max_mib"),
            durable_prefix_cache_max_entry_mib: command_line("durable_prefix_cache_max_entry_mib"),
            durable_prefix_cache_min_tokens: command_line("durable_prefix_cache_min_tokens"),
            deepseek_v4_multigroup_selector: command_line("deepseek_v4_multigroup_selector"),
        }
    }
}

#[cfg(test)]
impl Args {
    /// Parse as `main` does: `Args` plus the explicit-option record from the
    /// top-level or `run` subcommand matches.
    pub(crate) fn parse_with_explicit<I, T>(argv: I) -> (Self, ExplicitCliOptions)
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        use clap::{CommandFactory, FromArgMatches};
        let matches = Self::command()
            .try_get_matches_from(argv)
            .expect("test invocation parses");
        let explicit = matches.subcommand().map_or_else(
            || ExplicitCliOptions::from_matches(&matches),
            |(_, matches)| ExplicitCliOptions::from_matches(matches),
        );
        (
            Self::from_arg_matches(&matches).expect("test invocation binds"),
            explicit,
        )
    }
}
