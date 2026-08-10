use super::Args;
use anyhow::{Context, Result, ensure};
use clap::{ArgGroup, Args as ClapArgs, Subcommand};
use std::io::Read;
use std::path::PathBuf;

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Run one model-templated or explicitly raw request.
    #[command(
        after_help = "Examples:\n  qwen run -m MODEL --user 'Explain this'\n  qwen run -m MODEL --system 'Be concise' --user 'Explain this'\n  qwen run -m MODEL --user -\n  qwen run -m MODEL --messages -\n  qwen run -m MODEL --raw-prompt '<exact model input>'\n  qwen run -m Qwen3.6-35B-A3B.gguf --user 'Explain this' --no-thinking"
    )]
    Run(RunArgs),
}

#[derive(Debug)]
pub(crate) enum Invocation {
    Legacy,
    Run(RunInvocation),
}

#[derive(Debug)]
pub(crate) struct RunInvocation {
    pub(crate) model: PathBuf,
    pub(crate) input: RunInput,
    pub(crate) no_thinking: bool,
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
    /// Path to a Qwen or DeepSeek V4 GGUF file.
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

    /// Render a strict bare array or {"messages":[...]} JSON document; '-' reads stdin.
    #[arg(long, value_name = "FILE|-")]
    messages: Option<PathBuf>,

    /// Exact untemplated input; supported model families retain legacy -p semantics.
    #[arg(long, value_name = "TEXT")]
    raw_prompt: Option<String>,

    /// Use validated non-thinking prompt rendering; does not suppress CLI diagnostics.
    ///
    /// The currently validated Qwen3.6 35B A3B identity uses a template
    /// transition; other Qwen identities fail closed. DeepSeek ordinary chat
    /// is already non-thinking, so this is an idempotent guarantee there. This
    /// is not an output filter.
    #[arg(long, conflicts_with = "raw_prompt")]
    no_thinking: bool,

    #[command(flatten)]
    generation: GenerationOverrides,
}

#[derive(Debug, Default, ClapArgs)]
struct GenerationOverrides {
    /// Maximum number of tokens to generate (default: 64).
    #[arg(short = 'n', long = "max-tokens", visible_alias = "tokens")]
    tokens: Option<usize>,

    /// Sampling temperature; zero preserves greedy decoding (default: 0).
    #[arg(long = "temp", visible_alias = "temperature")]
    temperature: Option<f32>,

    /// Top-k sampling cutoff; zero disables it (default: 200).
    #[arg(long)]
    top_k: Option<usize>,

    /// Nucleus sampling cutoff; one disables it (default: 1).
    #[arg(long)]
    top_p: Option<f32>,

    /// Min-p sampling cutoff; zero disables it (default: 0.05).
    #[arg(long)]
    min_p: Option<f32>,

    /// Effective deterministic seed (default: 42).
    #[arg(long)]
    seed: Option<u64>,

    /// Override Qwen sequence capacity; DeepSeek single-turn rejects this option.
    #[arg(long)]
    max_context_tokens: Option<usize>,
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
                generation: run.generation,
            })
        }
    }
}

fn read_stdin_string(label: &str) -> Result<String> {
    let mut input = String::new();
    std::io::stdin()
        .lock()
        .read_to_string(&mut input)
        .with_context(|| format!("read {label}"))?;
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
        assert_eq!(
            super::super::ExplicitCliOptions::from_matches(run_matches),
            super::super::ExplicitCliOptions::default()
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
