use super::Args;
use anyhow::{Context, Result, ensure};
use clap::{ArgGroup, Args as ClapArgs, Subcommand};
use qwen_llm::snapshot_policy::SnapshotPolicyConfig;
use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Run one model-templated or explicitly raw request.
    #[command(
        after_help = "Examples:\n  qwen run -m MODEL --user 'Explain this'\n  qwen run -m MODEL --system 'Be concise' --user 'Explain this'\n  qwen run -m Qwen3.8-27B.gguf --reasoning-effort low --user 'Explain this'\n  qwen run -m MODEL --user -\n  qwen run -m MODEL --messages -\n  qwen run -m MODEL --raw-prompt '<exact model input>'\n  qwen run -m Qwen3.6-35B-A3B.gguf --user 'Explain this' --no-thinking"
    )]
    Run(RunArgs),
    /// Serve the Open Responses subset over loopback HTTP (docs/SERVE.md).
    #[command(
        after_help = "Examples:\n  qwen serve -m MODEL\n  qwen serve -m MODEL --addr 127.0.0.1:8737 --max-tokens 65536\n  qwen serve -m Muse-Glimmer.gguf --max-context-tokens 7168 --max-tokens 2048\n  qwen serve -m MODEL --trace-sse /tmp/qwen.sse.jsonl\n\nEndpoints: POST /v1/responses (stream and non-stream), GET /v1/models.\nSerial: one request in flight; stateless (store:false only)."
    )]
    Serve(ServeArgs),
    /// Inspect model metadata and capabilities without using the GPU.
    #[command(
        after_help = "Examples:\n  qwen info -m MODEL\n  qwen info -m MODEL --json\n\n--json reports the detected family and whether --drafter would be admitted per lane (run, serve), with a stable reason code when refused. Eligible K2 JSON inspection also hashes retained checkpoint bytes on the CPU to verify chat identity; it checks cancellation between bounded reads. Text inspection does not perform that chat verification."
    )]
    Info(InfoArgs),
}

#[derive(Debug, ClapArgs)]
pub(crate) struct InfoArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Emit a machine-readable projection instead of the text summary.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, ClapArgs)]
pub(crate) struct ServeArgs {
    /// Supported Qwen, DeepSeek V4, Muse Glimmer, or dense K2 GGUF (K2: raw only).
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Loopback listen address (required; there is no auth).
    #[arg(long, default_value = "127.0.0.1:8737")]
    addr: String,

    /// Default max_output_tokens; required for fixed-capacity families including K2.
    #[arg(long = "max-tokens", value_parser = parse_positive_usize)]
    max_tokens: Option<usize>,

    /// Fixed capacity for DS4/Muse/K2 within model context and device memory; Qwen is request-shaped.
    #[arg(long, value_parser = parse_positive_usize)]
    max_context_tokens: Option<usize>,

    /// Qwen/Flash-Next/DS4 RAM snapshot-cache MiB, or `auto`: min(25% of RAM, 50% of the
    /// Metal working set left after load), at least 1 GiB. K2 ignores it (live-session prefix reuse).
    #[arg(long, value_name = "MIB|auto", default_value = "auto", value_parser = parse_snapshot_cache_mib)]
    snapshot_cache_mib: SnapshotCacheMib,

    /// Expire snapshots unused for this long; 0 disables.
    #[arg(long, value_name = "SECS", default_value_t = 3600)]
    snapshot_idle_ttl_secs: u64,

    /// Expire snapshots this long after capture regardless of use; 0 disables.
    #[arg(long, value_name = "SECS", default_value_t = 86_400)]
    snapshot_max_age_secs: u64,

    /// Frecency half-life for eviction ranking; 0 evicts pure LRU.
    #[arg(long, value_name = "SECS", default_value_t = 600)]
    snapshot_half_life_secs: u64,

    /// Qwen/DS4 durable snapshot directory for warm prefixes across restarts,
    /// or `off`. Default ~/.cache/qwen-llm/serve-checkpoints; each family
    /// uses its own subdirectory and budget.
    #[arg(long, value_name = "PATH|off", value_parser = crate::serve::durable::parse_durable_dir)]
    durable_snapshot_dir: Option<crate::serve::durable::DurableDir>,

    /// Durable snapshot disk budget in MiB, or `auto`: min(64 GiB, 10% of
    /// the volume's free space). Least recently used records are evicted.
    #[arg(long, value_name = "MIB|auto", default_value = "auto", value_parser = parse_durable_max_mib)]
    durable_snapshot_max_mib: DurableMaxMib,

    /// Never persist (or look up on disk) prefixes shorter than this.
    #[arg(long, value_name = "TOKENS", default_value_t = crate::serve::durable::DEFAULT_MIN_TOKENS)]
    durable_snapshot_min_tokens: usize,

    /// DFlash drafter GGUF for speculative decode.
    ///
    /// Greedy and sampled proposals are verified by the target model. Sampled
    /// verification uses the packed target forward, whose floating-point
    /// arithmetic can differ slightly from serial token-major decoding.
    /// Speculation needs the relevant target-hidden window. Restored requests
    /// speculate when the cache entry carries a compatible capture tail;
    /// otherwise they decode serially while refreshing that tail. The
    /// per-request `serve phases:` line reports `decode_path=dflash|serial`.
    #[arg(long, value_name = "GGUF")]
    drafter: Option<PathBuf>,

    /// Append request and streamed SSE events as JSONL for wire debugging.
    #[arg(long, value_name = "PATH")]
    trace_sse: Option<PathBuf>,

    /// Whose conventions prompts follow where serve deliberately departs from
    /// a release chat template (Qwen and DeepSeek V4). `house`: past turns
    /// render as they were generated whatever the current thinking mode, and
    /// reasoning history is kept. `upstream`: the release template's own
    /// rules. Requests may override with `x_qwen.template_style`.
    #[arg(long, value_name = "house|upstream", default_value = "house", value_parser = parse_template_style)]
    template_style: crate::open_responses::items::TemplateStyle,
}

fn parse_template_style(
    value: &str,
) -> std::result::Result<crate::open_responses::items::TemplateStyle, String> {
    crate::open_responses::items::TemplateStyle::parse(value)
        .ok_or_else(|| format!("expected `house` or `upstream`, got {value:?}"))
}

/// `None` is `auto`.
#[derive(Clone, Copy, Debug)]
struct SnapshotCacheMib(Option<u64>);

/// `None` is `auto`.
#[derive(Clone, Copy, Debug)]
struct DurableMaxMib(Option<u64>);

fn parse_durable_max_mib(value: &str) -> std::result::Result<DurableMaxMib, String> {
    crate::serve::durable::parse_durable_max_mib(value).map(DurableMaxMib)
}

fn parse_snapshot_cache_mib(value: &str) -> std::result::Result<SnapshotCacheMib, String> {
    if value == "auto" {
        return Ok(SnapshotCacheMib(None));
    }
    value
        .parse::<u64>()
        .map(|mib| SnapshotCacheMib(Some(mib)))
        .map_err(|error| format!("expected `auto` or a MiB count, got {value:?}: {error}"))
}

fn parse_positive_usize(value: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid positive integer {value:?}: {error}"))?;
    if parsed == 0 {
        return Err("value must be at least 1".into());
    }
    Ok(parsed)
}

#[derive(Debug)]
pub(crate) enum Invocation {
    Legacy,
    Run(RunInvocation),
    Serve(ServeInvocation),
    Info(InfoInvocation),
}

#[derive(Debug)]
pub(crate) struct InfoInvocation {
    pub(crate) model: PathBuf,
    pub(crate) json: bool,
}

#[derive(Debug)]
pub(crate) struct ServeInvocation {
    pub(crate) model: PathBuf,
    pub(crate) addr: String,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) max_context_tokens: Option<usize>,
    /// `None` sizes the cache automatically after load.
    pub(crate) snapshot_cache_mib: Option<u64>,
    pub(crate) snapshot_policy: SnapshotPolicyConfig,
    pub(crate) durable: crate::serve::durable::DurableSnapshotConfig,
    pub(crate) drafter: Option<PathBuf>,
    pub(crate) trace_sse: Option<PathBuf>,
    pub(crate) template_style: crate::open_responses::items::TemplateStyle,
}

#[derive(Debug)]
pub(crate) struct RunInvocation {
    pub(crate) model: PathBuf,
    pub(crate) input: RunInput,
    pub(crate) no_thinking: bool,
    /// Effort spelling as supplied; each family binds it against its own
    /// levels after the model is detected (`qwen info --json` lists them).
    pub(crate) reasoning_effort: Option<String>,
    generation: GenerationOverrides,
}

#[derive(Debug)]
pub(crate) enum RunInput {
    User {
        system: Option<String>,
        user: TextSource,
    },
    Messages(DocumentSource),
    RawPrompt(String),
}

#[derive(Debug)]
pub(crate) enum AcquiredRunInput {
    User {
        system: Option<String>,
        user: String,
    },
    Messages {
        document: String,
        source: String,
    },
    RawPrompt(String),
}

#[derive(Debug)]
pub(crate) enum TextSource {
    Inline(String),
    Stdin,
}

#[derive(Debug)]
pub(crate) enum DocumentSource {
    File(PathBuf),
    Stdin,
}

#[derive(Debug, ClapArgs)]
#[command(group(
    ArgGroup::new("run_input")
        .required(true)
        .multiple(false)
        .args(["user", "messages", "raw_prompt"])
))]
pub(crate) struct RunArgs {
    /// Path to a Qwen, DeepSeek V4, Muse Glimmer, or K2 Horizon GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// One user message rendered with the model-family chat template; '-' reads stdin.
    #[arg(long, value_name = "TEXT|-")]
    user: Option<String>,

    /// Add one system message before --user.
    #[arg(
        long,
        requires = "user",
        conflicts_with_all = ["messages", "raw_prompt"]
    )]
    system: Option<String>,

    /// Render family-validated messages JSON (strict chat or Muse ATEM); '-' reads stdin.
    #[arg(long, value_name = "FILE|-")]
    messages: Option<PathBuf>,

    /// Exact untemplated input; supported model families retain legacy -p semantics.
    #[arg(long, value_name = "TEXT")]
    raw_prompt: Option<String>,

    /// Render the model's released non-thinking prompt transition; does not suppress CLI diagnostics.
    ///
    /// Supported on any GGUF identified as a Qwen3.5, Qwen3.6, Qwen3.8, or
    /// Flash-Next release; unidentified releases fail closed. DeepSeek
    /// ordinary chat is already non-thinking, so this is an idempotent
    /// guarantee there. This is not an output filter.
    #[arg(long, conflicts_with = "raw_prompt")]
    no_thinking: bool,

    /// Reasoning depth, bound against the detected model's own levels (see
    /// `qwen info --json` capabilities.reasoning). Qwen3.8: none/low/medium/xhigh,
    /// default xhigh; DeepSeek V4: none/low/high/max, default none (ordinary
    /// chat); Muse: low/medium/high/xhigh, default high; verified K2 chat:
    /// high/medium/low, default high (no non-thinking transition).
    #[arg(
        long,
        value_name = "LEVEL",
        conflicts_with_all = ["raw_prompt", "no_thinking"]
    )]
    reasoning_effort: Option<String>,

    #[command(flatten)]
    generation: GenerationOverrides,
}

#[derive(Debug, Default, ClapArgs)]
struct GenerationOverrides {
    /// Do not insert tokenizer BOS for raw input; retain any explicitly supplied special tokens.
    #[arg(long, requires = "raw_prompt")]
    no_special_tokens: bool,
    /// Maximum generated tokens (default: 64); prompt plus generation must fit sequence capacity.
    #[arg(short = 'n', long = "max-tokens", visible_alias = "tokens")]
    tokens: Option<usize>,

    /// Sampling temperature; omitted uses the detected family preset (Muse: 1, others: 0).
    #[arg(long = "temp", visible_alias = "temperature")]
    temperature: Option<f32>,

    /// Top-k cutoff; omitted uses the detected family preset (Muse: 64, others: 200).
    #[arg(long)]
    top_k: Option<usize>,

    /// Nucleus cutoff; omitted uses the detected family preset (Muse: .95, others: 1).
    #[arg(long)]
    top_p: Option<f32>,

    /// Min-p cutoff; omitted uses the detected family preset (Muse: 0, others: .05).
    #[arg(long)]
    min_p: Option<f32>,

    /// Effective deterministic seed (default: 42).
    #[arg(long)]
    seed: Option<u64>,

    /// Override family-specific sequence capacity (bounded by model context and memory).
    #[arg(long)]
    max_context_tokens: Option<usize>,

    /// DFlash drafter GGUF for speculative decode. Sampled DFlash2 proposals
    /// use sparse rejection sampling against the packed target verifier.
    #[arg(long, value_name = "GGUF")]
    drafter: Option<PathBuf>,

    /// Append one `qwen-llm.request-stats` v1 record for this request as
    /// JSONL: status, usage, finish reason, timing, throughput, output
    /// fingerprint. Raw stdout is unchanged.
    #[arg(long, value_name = "PATH")]
    request_stats_jsonl: Option<PathBuf>,
}

impl Invocation {
    pub(crate) fn is_run(&self) -> bool {
        matches!(self, Self::Run(_))
    }

    pub(crate) fn apply_option_overrides(&self, args: &mut Args) {
        let Self::Run(run) = self else {
            return;
        };
        args.model = Some(run.model.clone());
        run.generation.apply(args);
    }
}

impl RunInvocation {
    pub(crate) fn acquire_input(self) -> Result<AcquiredRunInput> {
        match self.input {
            RunInput::User { system, user } => {
                let user = user.read("--user -")?;
                ensure!(!user.is_empty(), "--user input is empty");
                Ok(AcquiredRunInput::User { system, user })
            }
            RunInput::Messages(document) => {
                let (document, source) = document.read()?;
                Ok(AcquiredRunInput::Messages { document, source })
            }
            RunInput::RawPrompt(prompt) => Ok(AcquiredRunInput::RawPrompt(prompt)),
        }
    }
}

impl GenerationOverrides {
    fn apply(&self, args: &mut Args) {
        if self.no_special_tokens {
            args.no_special_tokens = true;
        }
        if let Some(value) = self.tokens {
            args.tokens = value;
        }
        if let Some(value) = self.temperature {
            args.temperature = value;
        }
        if let Some(value) = self.top_k {
            args.top_k = value;
        }
        if let Some(value) = self.top_p {
            args.top_p = value;
        }
        if let Some(value) = self.min_p {
            args.min_p = value;
        }
        if let Some(value) = self.seed {
            args.seed = value;
        }
        if let Some(value) = self.max_context_tokens {
            args.max_context_tokens = Some(value);
        }
        if let Some(value) = self.drafter.as_ref() {
            args.drafter = Some(value.clone());
        }
        if let Some(value) = self.request_stats_jsonl.as_ref() {
            args.request_stats_jsonl = Some(value.clone());
        }
    }
}

impl TextSource {
    fn read(self, stdin_label: &str) -> Result<String> {
        match self {
            Self::Inline(value) => Ok(value),
            Self::Stdin => read_stdin_string(stdin_label),
        }
    }
}

impl DocumentSource {
    fn read(self) -> Result<(String, String)> {
        match self {
            Self::File(path) => {
                let document = std::fs::read_to_string(&path)
                    .with_context(|| format!("read messages input {}", path.display()))?;
                Ok((document, path.display().to_string()))
            }
            Self::Stdin => Ok((
                read_stdin_string("--messages -")?,
                "stdin (--messages -)".into(),
            )),
        }
    }
}

pub(crate) fn normalize(args: &mut Args) -> Invocation {
    let Some(command) = args.command.take() else {
        return Invocation::Legacy;
    };

    match command {
        Command::Info(info) => Invocation::Info(InfoInvocation {
            model: info.model,
            json: info.json,
        }),
        Command::Serve(serve) => Invocation::Serve(ServeInvocation {
            model: serve.model,
            addr: serve.addr,
            max_tokens: serve.max_tokens,
            max_context_tokens: serve.max_context_tokens,
            snapshot_cache_mib: serve.snapshot_cache_mib.0,
            snapshot_policy: SnapshotPolicyConfig {
                half_life: Duration::from_secs(serve.snapshot_half_life_secs),
                idle_ttl: Duration::from_secs(serve.snapshot_idle_ttl_secs),
                max_age: Duration::from_secs(serve.snapshot_max_age_secs),
            },
            durable: crate::serve::durable::DurableSnapshotConfig {
                dir: serve
                    .durable_snapshot_dir
                    .unwrap_or(crate::serve::durable::DurableDir::Default),
                max_mib: serve.durable_snapshot_max_mib.0,
                min_tokens: serve.durable_snapshot_min_tokens,
            },
            drafter: serve.drafter,
            trace_sse: serve.trace_sse,
            template_style: serve.template_style,
        }),
        Command::Run(run) => {
            let input = match (run.user, run.messages, run.raw_prompt) {
                (Some(user), None, None) => RunInput::User {
                    system: run.system,
                    user: if user == "-" {
                        TextSource::Stdin
                    } else {
                        TextSource::Inline(user)
                    },
                },
                (None, Some(messages), None) => {
                    RunInput::Messages(if messages.as_os_str() == "-" {
                        DocumentSource::Stdin
                    } else {
                        DocumentSource::File(messages)
                    })
                }
                (None, None, Some(raw_prompt)) => RunInput::RawPrompt(raw_prompt),
                _ => unreachable!("clap run_input group enforces exactly one input"),
            };
            Invocation::Run(RunInvocation {
                model: run.model,
                input,
                no_thinking: run.no_thinking,
                reasoning_effort: run.reasoning_effort,
                generation: run.generation,
            })
        }
    }
}

fn read_stdin_string(label: &str) -> Result<String> {
    crate::shutdown::checkpoint()?;
    let mut input = String::new();
    std::io::stdin()
        .lock()
        .read_to_string(&mut input)
        .with_context(|| format!("read {label}"))?;
    crate::shutdown::checkpoint()?;
    ensure!(!input.is_empty(), "{label} read empty stdin");
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};
    use std::path::Path;

    fn parse(argv: &[&str]) -> (Args, Invocation) {
        let mut args = Args::try_parse_from(argv).expect("parse test invocation");
        let invocation = normalize(&mut args);
        invocation.apply_option_overrides(&mut args);
        (args, invocation)
    }

    #[test]
    fn serve_rejects_zero_generation_and_context_limits() {
        for option in ["--max-tokens", "--max-context-tokens"] {
            assert!(
                Args::try_parse_from(["qwen", "serve", "-m", "model.gguf", option, "0"]).is_err()
            );
        }
    }

    #[test]
    fn serve_preserves_whether_max_tokens_was_explicit() {
        let (_, invocation) = parse(&["qwen", "serve", "-m", "model.gguf"]);
        let Invocation::Serve(defaults) = invocation else {
            panic!("expected serve invocation")
        };
        assert_eq!(defaults.max_tokens, None);

        let (_, invocation) = parse(&["qwen", "serve", "-m", "model.gguf", "--max-tokens", "2048"]);
        let Invocation::Serve(explicit) = invocation else {
            panic!("expected serve invocation")
        };
        assert_eq!(explicit.max_tokens, Some(2048));
    }

    #[test]
    fn serve_snapshot_cache_defaults_to_auto_with_service_policy() {
        let serve = |extra: &[&str]| {
            let mut argv = vec!["qwen", "serve", "-m", "model.gguf"];
            argv.extend_from_slice(extra);
            let Invocation::Serve(serve) = parse(&argv).1 else {
                panic!("expected serve invocation")
            };
            serve
        };
        let defaults = serve(&[]);
        assert_eq!(defaults.snapshot_cache_mib, None);
        assert_eq!(defaults.snapshot_policy, SnapshotPolicyConfig::default());
        use crate::open_responses::items::TemplateStyle;
        assert_eq!(defaults.template_style, TemplateStyle::House);
        assert_eq!(
            serve(&["--template-style", "upstream"]).template_style,
            TemplateStyle::Upstream
        );
        assert!(parse_template_style("vendor").is_err());
        assert_eq!(
            serve(&["--snapshot-cache-mib", "auto"]).snapshot_cache_mib,
            None
        );
        assert_eq!(
            serve(&["--snapshot-cache-mib", "0"]).snapshot_cache_mib,
            Some(0)
        );
        let tuned = serve(&[
            "--snapshot-cache-mib",
            "2048",
            "--snapshot-idle-ttl-secs",
            "0",
            "--snapshot-max-age-secs",
            "60",
            "--snapshot-half-life-secs",
            "0",
        ]);
        assert_eq!(tuned.snapshot_cache_mib, Some(2048));
        assert_eq!(
            tuned.snapshot_policy,
            SnapshotPolicyConfig {
                half_life: Duration::ZERO,
                idle_ttl: Duration::ZERO,
                max_age: Duration::from_secs(60),
            }
        );
        assert!(
            Args::try_parse_from([
                "qwen",
                "serve",
                "-m",
                "m.gguf",
                "--snapshot-cache-mib",
                "big"
            ])
            .is_err()
        );
        assert!(parse_snapshot_cache_mib("Auto").is_err());
        assert!(parse_snapshot_cache_mib("-1").is_err());
    }

    #[test]
    fn serve_durable_snapshots_default_on_with_auto_budget() {
        use crate::serve::durable::{DEFAULT_MIN_TOKENS, DurableDir, DurableSnapshotConfig};
        let serve = |extra: &[&str]| {
            let mut argv = vec!["qwen", "serve", "-m", "model.gguf"];
            argv.extend_from_slice(extra);
            let Invocation::Serve(serve) = parse(&argv).1 else {
                panic!("expected serve invocation")
            };
            serve.durable
        };
        assert_eq!(
            serve(&[]),
            DurableSnapshotConfig {
                dir: DurableDir::Default,
                max_mib: None,
                min_tokens: DEFAULT_MIN_TOKENS,
            }
        );
        assert_eq!(DEFAULT_MIN_TOKENS, 1024);
        assert_eq!(
            serve(&["--durable-snapshot-dir", "off"]).dir,
            DurableDir::Off
        );
        let tuned = serve(&[
            "--durable-snapshot-dir",
            "/tmp/warm",
            "--durable-snapshot-max-mib",
            "4096",
            "--durable-snapshot-min-tokens",
            "0",
        ]);
        assert_eq!(tuned.dir, DurableDir::Path("/tmp/warm".into()));
        assert_eq!(tuned.max_mib, Some(4096));
        assert_eq!(tuned.min_tokens, 0);
        assert_eq!(serve(&["--durable-snapshot-max-mib", "auto"]).max_mib, None);
        for bad in [
            &["--durable-snapshot-max-mib", "lots"][..],
            &["--durable-snapshot-dir", ""],
            &["--durable-snapshot-min-tokens", "-1"],
        ] {
            let mut argv = vec!["qwen", "serve", "-m", "m.gguf"];
            argv.extend_from_slice(bad);
            assert!(Args::try_parse_from(argv).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn run_user_is_closed_and_does_not_mutate_legacy_inputs() {
        let (args, invocation) = parse(&[
            "qwen",
            "run",
            "-m",
            "model.gguf",
            "--system",
            "Be exact",
            "--user",
            "Hello",
        ]);
        assert_eq!(args.model.as_deref(), Some(Path::new("model.gguf")));
        assert!(args.prompt.is_none());
        assert!(args.prompt_file.is_none());
        assert!(args.messages.is_none());
        assert!(args.requests_jsonl.is_none());
        let Invocation::Run(run) = invocation else {
            panic!("expected run invocation");
        };
        assert!(matches!(
            run.input,
            RunInput::User {
                system: Some(ref system),
                user: TextSource::Inline(ref user),
            } if system == "Be exact" && user == "Hello"
        ));
        assert_eq!(run.reasoning_effort, None);
    }

    #[test]
    fn run_passes_reasoning_levels_through_for_family_binding() {
        for value in ["low", "medium", "high", "xhigh", "max", "none"] {
            let (_, invocation) = parse(&[
                "qwen",
                "run",
                "-m",
                "model.gguf",
                "--user",
                "hi",
                "--reasoning-effort",
                value,
            ]);
            let Invocation::Run(run) = invocation else {
                panic!("expected run");
            };
            assert_eq!(run.reasoning_effort.as_deref(), Some(value));
        }
    }

    #[test]
    fn run_overlays_only_explicit_generation_values() {
        let (defaults, _) = parse(&["qwen", "run", "-m", "model.gguf", "--user", "Hello"]);
        assert_eq!(defaults.tokens, 64);
        assert_eq!(defaults.temperature, 0.0);
        assert_eq!(defaults.seed, 42);

        let (overridden, _) = parse(&[
            "qwen",
            "run",
            "-m",
            "model.gguf",
            "--user",
            "Hello",
            "--max-tokens",
            "512",
            "--temp",
            "0.7",
            "--seed",
            "7",
        ]);
        assert_eq!(overridden.tokens, 512);
        assert_eq!(overridden.temperature, 0.7);
        assert_eq!(overridden.seed, 7);
    }

    #[test]
    fn run_nested_matches_do_not_invent_legacy_explicit_options() {
        let matches = Args::command()
            .try_get_matches_from([
                "qwen",
                "run",
                "-m",
                "model.gguf",
                "--user",
                "Hello",
                "--max-tokens",
                "512",
            ])
            .unwrap();
        let (_, run_matches) = matches.subcommand().expect("run subcommand matches");
        let explicit = super::super::ExplicitCliOptions::from_matches(run_matches);
        assert_eq!(
            explicit,
            super::super::Args::parse_with_explicit([
                "qwen",
                "-m",
                "model.gguf",
                "--prompt",
                "x",
                "-n",
                "512"
            ])
            .1
        );
    }

    #[test]
    fn run_rejects_ambiguous_or_unsupported_input_shapes() {
        for argv in [
            vec!["qwen", "run", "-m", "model.gguf"],
            vec!["qwen", "run", "-m", "model.gguf", "Hello"],
            vec![
                "qwen",
                "run",
                "-m",
                "model.gguf",
                "--user",
                "Hello",
                "--messages",
                "messages.json",
            ],
            vec![
                "qwen",
                "run",
                "-m",
                "model.gguf",
                "--raw-prompt",
                "raw",
                "--no-thinking",
            ],
            vec![
                "qwen",
                "run",
                "-m",
                "model.gguf",
                "--raw-prompt",
                "raw",
                "--reasoning-effort",
                "low",
            ],
            vec![
                "qwen",
                "run",
                "-m",
                "model.gguf",
                "--user",
                "Hello",
                "--no-thinking",
                "--reasoning-effort",
                "medium",
            ],
            vec![
                "qwen",
                "run",
                "-m",
                "model.gguf",
                "--messages",
                "messages.json",
                "--system",
                "system",
            ],
        ] {
            assert!(
                Args::try_parse_from(&argv).is_err(),
                "unexpectedly accepted {argv:?}"
            );
        }
    }

    #[test]
    fn legacy_messages_dash_remains_legacy_and_unacquired() {
        let (args, invocation) = parse(&["qwen", "-m", "model.gguf", "--messages", "-"]);
        assert!(matches!(invocation, Invocation::Legacy));
        assert_eq!(args.messages.as_deref(), Some(Path::new("-")));
        assert!(args.prepared_prompt.is_none());
    }

    #[test]
    fn modern_raw_prompt_remains_distinct_from_legacy_state() {
        let (args, invocation) =
            parse(&["qwen", "run", "-m", "model.gguf", "--raw-prompt", "<raw>"]);
        assert!(args.prompt.is_none());
        assert!(matches!(
            invocation,
            Invocation::Run(RunInvocation {
                input: RunInput::RawPrompt(ref prompt),
                ..
            }) if prompt == "<raw>"
        ));
    }

    #[test]
    fn short_help_is_progressive_and_long_help_retains_legacy_flags() {
        let mut short = Vec::new();
        Args::command().write_help(&mut short).unwrap();
        let short = String::from_utf8(short).unwrap();
        assert!(short.contains("run"));
        assert!(short.contains("qwen run -m MODEL --user"));
        assert!(short.contains("--no-thinking controls model prompt rendering"));
        assert!(short.contains("CLI diagnostic suppression is not currently available"));
        assert!(short.contains("For resident JSONL batching and expanded legacy/research help"));
        assert!(short.contains("cannot be combined with qwen run"));
        assert!(!short.contains("--requests-jsonl"));
        assert!(!short.contains("--sampling-attribution"));

        let mut long = Vec::new();
        Args::command().write_long_help(&mut long).unwrap();
        let long = String::from_utf8(long).unwrap();
        assert!(long.contains("--requests-jsonl"));
        assert!(long.contains("--sampling-attribution"));
        assert!(long.contains("Legacy/research examples (flat"));
        assert!(long.contains("qwen -m MODEL --prompt '<raw model input>'"));
        assert!(long.contains("qwen -m MODEL --requests-jsonl requests.jsonl"));
        assert!(long.contains("CLI diagnostic suppression is not currently available"));
    }
}
