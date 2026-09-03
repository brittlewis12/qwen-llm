//! `qwen` — interactive CLI for the qwen-llm engine.

#[path = "qwen/args.rs"]
mod args;
mod cli;
mod concurrent_jsonl;
#[path = "qwen/decode.rs"]
mod decode;
#[path = "qwen/deepseek_v4.rs"]
mod deepseek_v4;
#[path = "qwen/dflash.rs"]
mod dflash;
#[cfg(feature = "dsv4-diagnostics")]
mod dsv4_temporal;
#[path = "qwen/durable_cache.rs"]
mod durable_cache;
mod execution_selector;
mod fixed_cohort_jsonl;
#[path = "qwen/jsonl.rs"]
mod jsonl;
mod messages;
mod model_request;
#[path = "qwen/muse_glimmer.rs"]
mod muse_glimmer;
mod open_responses;
#[path = "qwen/prefill_plan.rs"]
mod prefill_plan;
#[path = "qwen/prompt_lookup.rs"]
mod prompt_lookup;
#[path = "qwen/qwen4exp.rs"]
mod qwen4exp;
mod qwen_file_root;
#[path = "qwen/run_options.rs"]
mod run_options;
mod serve;
mod shutdown;
#[path = "qwen/single_turn.rs"]
mod single_turn;
#[path = "qwen/telemetry.rs"]
mod telemetry;
#[cfg(test)]
#[path = "qwen/tests.rs"]
mod tests;
mod tracing_init;

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
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDevice};
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
use qwen_llm::loader::{Model, open_dflash_drafter};
use qwen_llm::metal::{
    KernelEncoder, MetalBufferSizeAndAlign, MetalContext, MetalMemoryAdmission, MetalMemorySignals,
    MetalPipelineCacheMetrics, MetalTensor, evaluate_metal_memory_admission,
    evaluate_metal_memory_admission_with_cpu_bytes,
};
use qwen_llm::metal_dflash::{
    DFlashDecoder, MetalDFlashDebugScratch, MetalDFlashHead, MetalDFlashLayerMajorScratch,
    MetalDFlashSession, MetalDFlashVerifyScratch, PrefillScratchConfig, PrefillScratchOverlayStats,
    PrefillScratchPlan, ensure_prompt_lookup_n8_supported,
    plan_prefill_scratch_with_matrix_max_pos_configured, prefill_tokens_with_multi_hidden,
};
use qwen_llm::metal_forward::{
    LogitsReadbackProfile, MetalForward, MfError, SnapshotValidationError, StructuralRowEvidence,
    TokenProfile, encode_scatter_offset_f32,
};
use qwen_llm::model::{Arch, ArchKind};
use qwen_llm::model_family::ModelFamily;
use qwen_llm::moe_batch16::MOE_BATCH16_WIDTH;
use qwen_llm::muse_glimmer::{ARCHITECTURE_NAME as MUSE_GLIMMER_ARCHITECTURE, MuseGlimmerConfig};
use qwen_llm::muse_glimmer_prompt::MuseGlimmerReasoningStrength;
use qwen_llm::muse_glimmer_request::MuseGlimmerRequest;
use qwen_llm::muse_glimmer_runtime::MuseGlimmerLoadedModel;
use qwen_llm::muse_glimmer_text_session::{
    MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS, MUSE_GLIMMER_PACKED_PREFILL_QUANTUM,
};
use qwen_llm::pid_metrics::{PidDelta, PidSnapshot};
use qwen_llm::prefetch::{DEFAULT_CHUNK_BYTES, DEFAULT_WORKERS};
use qwen_llm::prompt_lookup::{DRAFT_TOKENS, PromptLookupProposer, terminal_draft_window};
use qwen_llm::qwen4exp::Qwen4ExpConfig;
use qwen_llm::qwen4exp_runtime::{
    Qwen4ExpLayerProfile, Qwen4ExpLayerStage, Qwen4ExpLoadedModel, Qwen4ExpPackedPrefillProfile,
    Qwen4ExpPackedProfileOutcome, Qwen4ExpPackedProfileSampling, Qwen4ExpPackedProfileScope,
    Qwen4ExpPackedProfileStageTiming, Qwen4ExpPrefillTiming, Qwen4ExpRuntimeError,
    Qwen4ExpSessionCapacity, Qwen4ExpTokenTiming,
};
use qwen_llm::runtime::{
    LoadedModel, LoadedModelConfig, PrefetchPolicy, PrefetchResidencyProbe, PreparedCheckpoint,
    Runtime, RuntimeError, Sequence, SequenceConfig, prefetch_opened_gguf,
};
use qwen_llm::sampling::{
    BoundedTopKEvidence, GreedySelection, SAMPLER_ALGORITHM_VERSION, SampledToken, Sampler,
    SamplingConfig, SamplingError, SamplingPhaseProfile, SpeculativeSamplingDecision,
};
use qwen_llm::tensor::GgmlType;
use qwen_llm::tokenizer::{LlamaCppTokenizer, Tokenizer, token_ids_sha256_i32le};
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

#[derive(Clone, Debug)]
struct MuseGlimmerPreparedPrompt {
    text: String,
    source: PromptSource,
    add_special_tokens: bool,
}

fn main() -> std::process::ExitCode {
    shutdown::finish(run())
}

fn run() -> Result<()> {
    shutdown::install()?;
    tracing_init::install_default_subscriber();

    let matches = Args::command().get_matches();
    let explicit_options = matches.subcommand().map_or_else(
        || ExplicitCliOptions::from_matches(&matches),
        |(_, matches)| ExplicitCliOptions::from_matches(matches),
    );
    let mut args = Args::from_arg_matches(&matches).expect("validated clap arguments");
    let invocation = cli::normalize(&mut args);
    let invocation = match invocation {
        cli::Invocation::Serve(serve_invocation) => return serve::run_serve(serve_invocation),
        other => other,
    };
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
    if gguf.architecture().as_deref() == Some(MUSE_GLIMMER_ARCHITECTURE) {
        return run_muse_glimmer_single_turn(
            &model_path,
            &gguf,
            &args,
            explicit_options,
            invocation,
        );
    }
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
    if model_family == Some(ModelFamily::Qwen4Exp) {
        return run_qwen4exp_single_turn(&model_path, &gguf, &args, explicit_options);
    }

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
    validate_drafter_decode_policy(args)?;
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
            ModelFamily::Qwen35
                | ModelFamily::Qwen35Moe
                | ModelFamily::Qwen4Exp
                | ModelFamily::DeepSeek4
        ),
        "`qwen run` does not support model family {}",
        family.architecture_name()
    );
    ensure!(
        family != ModelFamily::DeepSeek4 || args.max_context_tokens.is_none(),
        "--max-context-tokens is not supported for DeepSeek V4 single-turn generation; remove --max-context-tokens"
    );
    if family == ModelFamily::Qwen4Exp
        && (run.no_thinking || run.reasoning_effort.is_some())
        && let Some(failure) = qwen4exp_prompt_capability_failure(family, gguf)
    {
        bail!(
            "Qwen3.8-Flash-Next option requires the supported qwen35 prompt protocol; incompatible {}",
            failure.as_str(),
        );
    }
    if run.no_thinking
        && matches!(
            family,
            ModelFamily::Qwen35 | ModelFamily::Qwen35Moe | ModelFamily::Qwen4Exp
        )
    {
        ensure!(
            supports_qwen_no_thinking_prompt(family, gguf),
            "--no-thinking is currently supported only for Qwen3.6 35B A3B, Qwen3.8 27B, and Qwen3.8-Flash-Next models with a compatible qwen35 prompt protocol; omit --no-thinking to use this model's default generation behavior"
        );
    }
    let qwen38 = supports_qwen38_prompt_protocol(family, gguf);

    let no_thinking = run.no_thinking;
    let qwen38_generation_mode =
        resolve_qwen38_generation_mode(qwen38, no_thinking, run.reasoning_effort)?;
    let input = run.acquire_input()?;
    let (text, source) = match input {
        cli::AcquiredRunInput::RawPrompt(prompt) => (prompt, PromptSource::Inline),
        cli::AcquiredRunInput::User { system, user } => {
            let prompt = match family {
                ModelFamily::Qwen35 | ModelFamily::Qwen35Moe | ModelFamily::Qwen4Exp if qwen38 => {
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
                ModelFamily::Qwen4Exp => {
                    let failure = qwen4exp_prompt_capability_failure(family, gguf)
                        .expect("unsupported Flash-Next prompt has a capability failure");
                    bail!(
                        "Qwen3.8-Flash-Next chat rendering does not support the declared {}; use --raw-prompt for untemplated input",
                        failure.as_str(),
                    )
                }
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
                ModelFamily::Qwen35 | ModelFamily::Qwen35Moe | ModelFamily::Qwen4Exp if qwen38 => {
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
                ModelFamily::Qwen4Exp => {
                    let failure = qwen4exp_prompt_capability_failure(family, gguf)
                        .expect("unsupported Flash-Next prompt has a capability failure");
                    bail!(
                        "Qwen3.8-Flash-Next chat rendering does not support the declared {}; use --raw-prompt for untemplated input",
                        failure.as_str(),
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
        "--reasoning-effort is currently supported only for Qwen3.8 27B and Qwen3.8-Flash-Next models with a compatible qwen35 prompt protocol"
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
        cli::RunReasoningEffort::High => {
            bail!("--reasoning-effort high is supported by Muse Glimmer, not Qwen3.8")
        }
        cli::RunReasoningEffort::Xhigh => Qwen38ReasoningEffort::Xhigh,
    };
    Ok(Some(Qwen38GenerationMode::Thinking(effort)))
}

pub(crate) fn supports_qwen_no_thinking_prompt(family: ModelFamily, gguf: &GgufFile) -> bool {
    messages::supports_qwen36_no_thinking_prompt_protocol(family, gguf)
        || supports_qwen38_prompt_protocol(family, gguf)
}

pub(crate) fn supports_qwen38_prompt_protocol(family: ModelFamily, gguf: &GgufFile) -> bool {
    messages::supports_qwen38_release_prompt_protocol(family, gguf)
}

#[cfg(test)]
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
    messages::validated_qwen38_prompt_identity(
        family,
        general_name,
        base_model_name,
        tokenizer_model,
        tokenizer_pre,
        context_length,
        block_count,
        nextn_predict_layers,
        embedding_length,
        feed_forward_length,
    )
}

#[cfg(test)]
fn validated_qwen36_no_thinking_identity(
    family: ModelFamily,
    base_model_name: Option<&str>,
    tokenizer_model: Option<&str>,
    tokenizer_pre: Option<&str>,
) -> bool {
    messages::validated_qwen36_no_thinking_identity(
        family,
        base_model_name,
        tokenizer_model,
        tokenizer_pre,
    )
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

fn current_effective_uid() -> u32 {
    // geteuid has no preconditions and does not retain pointers.
    unsafe { libc::geteuid() }
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

use args::*;
use decode::*;
use deepseek_v4::*;
use dflash::*;
use durable_cache::*;
pub(crate) use fingerprint::GeneratedTokenSha256Digest;
use jsonl::*;
use muse_glimmer::*;
use prefill_plan::*;
use prompt_lookup::*;
use qwen4exp::*;
use run_options::*;
use single_turn::*;
use telemetry::*;

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
