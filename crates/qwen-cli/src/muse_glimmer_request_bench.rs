use anyhow::{Context, Result, ensure};
use clap::{ArgGroup, Parser, ValueEnum};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::muse_glimmer::{
    MuseGlimmerArtifactProfile, MuseGlimmerChatTemplateProfile, MuseGlimmerConfig,
};
use qwen_llm::muse_glimmer_prompt::MuseGlimmerReasoningStrength;
use qwen_llm::muse_glimmer_request::MuseGlimmerRequest;
use qwen_llm::muse_glimmer_runtime::{MuseGlimmerLoadedModel, MuseGlimmerTextRunner};
use qwen_llm::muse_glimmer_text_session::MUSE_GLIMMER_PACKED_PREFILL_QUANTUM;
use qwen_llm::sampling::{SAMPLER_ALGORITHM_VERSION, Sampler, SamplingConfig};
use qwen_llm::tokenizer::{LlamaCppTokenizer, token_ids_sha256_i32le};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_PROMPT: &str = "The quick brown fox jumps over the lazy dog";
const SCHEMA: &str = "qwen.muse_glimmer.request_benchmark";
const SCHEMA_VERSION: u32 = 2;

#[derive(Parser, Debug)]
#[command(group(
    ArgGroup::new("input")
        .multiple(false)
        .args(["prompt", "messages"])
))]
pub struct MuseRequestArgs {
    /// Path to the first Muse Glimmer GGUF shard.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// User text rendered through the Muse ATEM template.
    #[arg(short = 'p', long)]
    prompt: Option<String>,
    /// Optional system text for --prompt; the fixed default prompt is used when
    /// neither --prompt nor --messages is supplied.
    #[arg(long, conflicts_with = "messages")]
    system: Option<String>,
    /// Strict Muse request JSON: a bare message array or wrapped document with
    /// messages, tools, reasoning strength, namespaces, and optional date.
    #[arg(long)]
    messages: Option<PathBuf>,
    /// Override the Muse ATEM reasoning strength. A conflicting value authored
    /// in --messages fails closed.
    #[arg(long, value_enum)]
    reasoning_strength: Option<MuseReasoningStrengthArg>,
    /// Maximum sampled tokens, including a terminal EOS/EOT token.
    #[arg(long, default_value_t = 64)]
    tokens: usize,
    /// Resident forward capacity. Omitted resolves to prompt + tokens - 1.
    #[arg(long, alias = "max-context-tokens")]
    capacity: Option<usize>,
    /// Timed full-request repetitions after warmup.
    #[arg(long, default_value_t = 1)]
    runs: usize,
    /// Skip the full-request warmup. Such output is marked non-steady-state.
    #[arg(long)]
    no_warmup: bool,
    /// Override the released Muse temperature (1.0).
    #[arg(long, alias = "temp")]
    temperature: Option<f32>,
    /// Override the released Muse top-k (64); zero disables the filter.
    #[arg(long)]
    top_k: Option<usize>,
    /// Override the released Muse top-p (0.95).
    #[arg(long)]
    top_p: Option<f32>,
    /// Override the released Muse min-p (0.0).
    #[arg(long)]
    min_p: Option<f32>,
    /// Request-local sampler seed, reconstructed for warmup and every run.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Text summary or the dedicated versioned JSON object.
    #[arg(short = 'o', long, value_enum, default_value = "text")]
    output: super::OutputFormat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MuseReasoningStrengthArg {
    Low,
    Medium,
    High,
    Xhigh,
}

impl MuseReasoningStrengthArg {
    fn resolve(self) -> MuseGlimmerReasoningStrength {
        match self {
            Self::Low => MuseGlimmerReasoningStrength::Low,
            Self::Medium => MuseGlimmerReasoningStrength::Medium,
            Self::High => MuseGlimmerReasoningStrength::High,
            Self::Xhigh => MuseGlimmerReasoningStrength::Xhigh,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct Termination {
    kind: &'static str,
    token_id: Option<i32>,
}

#[derive(Debug)]
struct GenerationMeasurement {
    sampled_token_ids: Vec<i32>,
    emitted_token_ids: Vec<i32>,
    wall_ns: u64,
    transition_forwards: usize,
    transition_forward_ns: u64,
    first_sample_ready_ns: u64,
    termination: Termination,
}

#[allow(clippy::too_many_arguments)]
fn drive_generation<State, Select, Transition, Emit, Checkpoint>(
    mut state: State,
    max_tokens: usize,
    eos_token_id: i32,
    eot_token_id: i32,
    mut select: Select,
    mut transition: Transition,
    mut emit: Emit,
    mut checkpoint: Checkpoint,
) -> Result<GenerationMeasurement>
where
    Select: FnMut(&State) -> Result<i32>,
    Transition: FnMut(i32) -> Result<State>,
    Emit: FnMut(i32) -> Result<()>,
    Checkpoint: FnMut() -> Result<()>,
{
    ensure!(max_tokens > 0, "--tokens must be >= 1");
    let started = Instant::now();
    let mut sampled_token_ids = Vec::with_capacity(max_tokens);
    let mut emitted_token_ids = Vec::with_capacity(max_tokens);
    let mut transition_forwards = 0usize;
    let mut transition_forward_ns = 0u64;
    let mut first_sample_ready_ns = None;

    let termination = loop {
        checkpoint()?;
        let token = select(&state)?;
        first_sample_ready_ns.get_or_insert_with(|| elapsed_ns(started));
        sampled_token_ids.push(token);
        if token == eos_token_id {
            break Termination {
                kind: "eos_token",
                token_id: Some(token),
            };
        }
        if token == eot_token_id {
            break Termination {
                kind: "eot_token",
                token_id: Some(token),
            };
        }
        emit(token)?;
        emitted_token_ids.push(token);
        if sampled_token_ids.len() == max_tokens {
            break Termination {
                kind: "token_limit",
                token_id: None,
            };
        }
        let transition_started = Instant::now();
        state = transition(token)?;
        transition_forwards += 1;
        transition_forward_ns =
            transition_forward_ns.saturating_add(elapsed_ns(transition_started));
        checkpoint()?;
    };

    Ok(GenerationMeasurement {
        sampled_token_ids,
        emitted_token_ids,
        wall_ns: elapsed_ns(started),
        transition_forwards,
        transition_forward_ns,
        first_sample_ready_ns: first_sample_ready_ns.expect("positive token limit samples once"),
        termination,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct RunShape {
    sampled_tokens: usize,
    emitted_tokens: usize,
    transition_forwards: usize,
    termination: Termination,
    sampled_token_ids_sha256_i32le: String,
}

#[derive(Clone, Debug, Serialize)]
struct RunSample {
    repetition: usize,
    resident_position_reset_ns: u64,
    prefill_wall_ns: u64,
    generation_wall_ns: u64,
    transition_forward_ns: u64,
    request_wall_ns: u64,
    phase_sum_first_sample_ready_ns: u64,
    prompt_forwards: usize,
    sampled_tokens: usize,
    emitted_tokens: usize,
    transition_forwards: usize,
    termination: Termination,
    sampled_token_ids_sha256_i32le: String,
}

#[derive(Debug)]
struct Iteration {
    sample: RunSample,
    shape: RunShape,
    sampled_token_ids: Vec<i32>,
    emitted_token_ids: Vec<i32>,
    output_bytes: Vec<u8>,
}

fn run_iteration(
    repetition: usize,
    runner: &mut MuseGlimmerTextRunner<'_, '_>,
    tokenizer: &LlamaCppTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    eos_token_id: i32,
    eot_token_id: i32,
    vocab_size: u32,
    sampling: SamplingConfig,
) -> Result<Iteration> {
    let request_started = Instant::now();
    let reset_started = Instant::now();
    runner
        .reset()
        .context("reset resident Muse request position")?;
    let reset_ns = elapsed_ns(reset_started);

    let prefill_started = Instant::now();
    let logits = runner
        .prefill_with_command_checkpoint(prompt_tokens, || {
            super::shutdown::checkpoint().map_err(|error| error.to_string())
        })
        .context("prefill Muse benchmark request")?;
    let prefill_ns = elapsed_ns(prefill_started);

    let mut sampler = Sampler::new(sampling).context("initialize Muse benchmark sampler")?;
    let mut output_bytes = Vec::new();
    let generation = drive_generation(
        logits,
        max_tokens,
        eos_token_id,
        eot_token_id,
        |logits| Ok(sampler.sample(logits)?.token),
        |token| {
            let token = checked_token_id(token, vocab_size, "generated")?;
            runner
                .forward_token(token)
                .context("forward generated Muse benchmark token")
        },
        |token| {
            output_bytes.extend_from_slice(
                &tokenizer
                    .try_decode_piece_bytes_exact(token)
                    .with_context(|| format!("decode Muse benchmark token {token}"))?,
            );
            Ok(())
        },
        super::shutdown::checkpoint,
    )?;
    let request_ns = elapsed_ns(request_started);
    let transition_forwards = generation.transition_forwards;
    ensure!(
        transition_forwards == generation.sampled_token_ids.len().saturating_sub(1),
        "Muse generation transition accounting diverged from sampled/emitted shape"
    );
    ensure!(
        generation.emitted_token_ids.len()
            == generation
                .sampled_token_ids
                .len()
                .saturating_sub(usize::from(generation.termination.token_id.is_some())),
        "Muse generation emitted-token accounting diverged from termination shape"
    );
    let token_digest = token_ids_sha256_i32le(&generation.sampled_token_ids);
    let shape = RunShape {
        sampled_tokens: generation.sampled_token_ids.len(),
        emitted_tokens: generation.emitted_token_ids.len(),
        transition_forwards,
        termination: generation.termination.clone(),
        sampled_token_ids_sha256_i32le: token_digest.clone(),
    };
    let sample = RunSample {
        repetition,
        resident_position_reset_ns: reset_ns,
        prefill_wall_ns: prefill_ns,
        generation_wall_ns: generation.wall_ns,
        transition_forward_ns: generation.transition_forward_ns,
        request_wall_ns: request_ns,
        phase_sum_first_sample_ready_ns: reset_ns
            .saturating_add(prefill_ns)
            .saturating_add(generation.first_sample_ready_ns),
        prompt_forwards: prompt_tokens.len(),
        sampled_tokens: shape.sampled_tokens,
        emitted_tokens: shape.emitted_tokens,
        transition_forwards,
        termination: shape.termination.clone(),
        sampled_token_ids_sha256_i32le: token_digest,
    };
    Ok(Iteration {
        sample,
        shape,
        sampled_token_ids: generation.sampled_token_ids,
        emitted_token_ids: generation.emitted_token_ids,
        output_bytes,
    })
}

#[derive(Debug, Serialize)]
struct PhaseSummary {
    mean_ns: f64,
    sample_stdev_ns: f64,
    samples_ns: Vec<u64>,
}

impl PhaseSummary {
    fn from_samples(values: impl Iterator<Item = u64>) -> Self {
        let samples_ns = values.collect::<Vec<_>>();
        let values = samples_ns
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>();
        Self {
            mean_ns: super::sample_mean(&values),
            sample_stdev_ns: super::sample_stdev(&values),
            samples_ns,
        }
    }
}

#[derive(Debug, Serialize)]
struct Aggregate {
    resident_position_reset: PhaseSummary,
    prefill_wall: PhaseSummary,
    generation_wall: PhaseSummary,
    transition_forward: PhaseSummary,
    request_wall: PhaseSummary,
    phase_sum_first_sample_ready: PhaseSummary,
    prefill_forwards_per_second: f64,
    transition_forwards_per_second: Option<f64>,
    sampled_tokens_per_second: f64,
    emitted_tokens_per_second: f64,
}

fn aggregate(samples: &[RunSample]) -> Aggregate {
    let sum =
        |value: fn(&RunSample) -> u64| samples.iter().map(value).fold(0u64, u64::saturating_add);
    let count = |value: fn(&RunSample) -> usize| {
        samples
            .iter()
            .map(value)
            .fold(0usize, usize::saturating_add)
    };
    let prefill_ns = sum(|sample| sample.prefill_wall_ns);
    let generation_ns = sum(|sample| sample.generation_wall_ns);
    let transition_ns = sum(|sample| sample.transition_forward_ns);
    let prompt_forwards = count(|sample| sample.prompt_forwards);
    let transition_forwards = count(|sample| sample.transition_forwards);
    let sampled_tokens = count(|sample| sample.sampled_tokens);
    let emitted_tokens = count(|sample| sample.emitted_tokens);
    Aggregate {
        resident_position_reset: PhaseSummary::from_samples(
            samples
                .iter()
                .map(|sample| sample.resident_position_reset_ns),
        ),
        prefill_wall: PhaseSummary::from_samples(
            samples.iter().map(|sample| sample.prefill_wall_ns),
        ),
        generation_wall: PhaseSummary::from_samples(
            samples.iter().map(|sample| sample.generation_wall_ns),
        ),
        transition_forward: PhaseSummary::from_samples(
            samples.iter().map(|sample| sample.transition_forward_ns),
        ),
        request_wall: PhaseSummary::from_samples(
            samples.iter().map(|sample| sample.request_wall_ns),
        ),
        phase_sum_first_sample_ready: PhaseSummary::from_samples(
            samples
                .iter()
                .map(|sample| sample.phase_sum_first_sample_ready_ns),
        ),
        prefill_forwards_per_second: rate(prompt_forwards, prefill_ns).unwrap_or(0.0),
        transition_forwards_per_second: rate(transition_forwards, transition_ns),
        sampled_tokens_per_second: rate(sampled_tokens, generation_ns).unwrap_or(0.0),
        emitted_tokens_per_second: rate(emitted_tokens, generation_ns).unwrap_or(0.0),
    }
}

fn rate(count: usize, elapsed_ns: u64) -> Option<f64> {
    (count > 0 && elapsed_ns > 0).then(|| count as f64 * 1e9 / elapsed_ns as f64)
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    path: String,
    architecture: &'static str,
    artifact_profile: &'static str,
    weight_bytes: u64,
    parameter_count: u64,
    tokenizer_identity_sha256: String,
    chat_template_profile: &'static str,
    chat_template_sha256: String,
    stop_token_ids: [i32; 2],
    observed_weight_allocation_bytes: u64,
    observed_session_allocation_bytes: u64,
    aggregate_required_bytes: Option<u64>,
    weight_required_bytes: Option<u64>,
    session_required_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct RequestInfo {
    input_kind: &'static str,
    rendered_prompt_sha256: String,
    prompt_token_ids_sha256_i32le: String,
    prompt_tokens: usize,
    cli_reasoning_strength: Option<&'static str>,
    document_reasoning_strength: Option<&'static str>,
    effective_reasoning_strength: &'static str,
    maximum_sampled_tokens: usize,
    required_forwards: usize,
    resident_capacity: usize,
}

#[derive(Debug, Serialize)]
struct SamplingInfo {
    algorithm_version: u32,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    seed: u64,
    overridden_fields: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct SetupInfo {
    gguf_open_ns: u64,
    release_contract_bind_ns: u64,
    request_render_ns: u64,
    tokenizer_open_ns: u64,
    tokenization_ns: u64,
    metal_context_ns: u64,
    resident_model_load_ns: u64,
}

#[derive(Debug, Serialize)]
struct MethodInfo {
    warmup_repetitions: usize,
    timed_repetitions: usize,
    warmup_scope: &'static str,
    sampler_scope: &'static str,
    session_scope: &'static str,
    prefill_mode: &'static str,
    decode_mode: &'static str,
    output_sink: &'static str,
}

#[derive(Debug, Serialize)]
struct Qualification {
    released_model_profile: bool,
    tokenizer_contract: bool,
    ordered_stop_contract: bool,
    input_atem_ready: bool,
    output_atem_validation: &'static str,
    shape_consistent: bool,
    steady_state_qualified: bool,
    canonical_build: bool,
    workload_qualified: bool,
    llama_bench_comparable: bool,
}

#[derive(Debug, Serialize)]
struct LastOutput {
    sampled_token_ids: Vec<i32>,
    emitted_token_ids: Vec<i32>,
    emitted_bytes_sha256: String,
    emitted_utf8_valid: bool,
    emitted_text_lossy: String,
}

#[derive(Debug, Serialize)]
struct Report {
    schema: &'static str,
    schema_version: u32,
    benchmark_kind: &'static str,
    test_time: String,
    build_identity: super::BuildIdentity,
    device: String,
    qwen_env: BTreeMap<String, String>,
    power: Option<super::PowerSnapshot>,
    model: ModelInfo,
    request: RequestInfo,
    sampling: SamplingInfo,
    setup: SetupInfo,
    method: MethodInfo,
    warmup_shape: Option<RunShape>,
    samples: Vec<RunSample>,
    aggregate: Aggregate,
    qualification: Qualification,
    last_output: LastOutput,
}

pub fn run(args: MuseRequestArgs) -> Result<()> {
    ensure!(args.runs > 0, "--runs must be >= 1");
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    let json_mode = matches!(args.output, super::OutputFormat::Json);
    let power = super::capture_power_snapshot();

    let gguf_started = Instant::now();
    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open Muse model {}", args.model.display()))?;
    let gguf_open_ns = elapsed_ns(gguf_started);

    let contract_started = Instant::now();
    let config = MuseGlimmerConfig::from_gguf(&gguf)
        .context("bind Muse Glimmer release contract for benchmark")?;
    let release_contract_bind_ns = elapsed_ns(contract_started);

    let render_started = Instant::now();
    let (request, input_kind) = load_request(&args)?;
    let requested_strength = args
        .reasoning_strength
        .map(MuseReasoningStrengthArg::resolve);
    let effective_strength = request
        .resolved_reasoning_strength(requested_strength)
        .context("resolve Muse benchmark reasoning strength")?;
    let rendered_prompt = request
        .render_annotated(config.chat_template_profile, requested_strength)
        .context("render annotated Muse benchmark ATEM request")?
        .text;
    let request_render_ns = elapsed_ns(render_started);

    let tokenizer_started = Instant::now();
    let tokenizer =
        LlamaCppTokenizer::open(&args.model).context("load Muse benchmark tokenizer")?;
    let tokenizer_open_ns = elapsed_ns(tokenizer_started);
    config
        .validate_tokenizer(&tokenizer)
        .context("Muse Glimmer tokenizer contract")?;

    let tokenization_started = Instant::now();
    let prompt_token_ids = tokenizer
        .encode(&rendered_prompt, false)
        .context("tokenize rendered Muse benchmark request")?;
    let tokenization_ns = elapsed_ns(tokenization_started);
    let prompt_tokens = prompt_token_ids
        .iter()
        .enumerate()
        .map(|(index, &token)| {
            checked_token_id(token, config.vocab_size, &format!("prompt[{index}]"))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !prompt_tokens.is_empty(),
        "Muse prompt tokenized to zero tokens"
    );

    let required_forwards = required_forwards(prompt_tokens.len(), args.tokens)?;
    let capacity = args.capacity.unwrap_or(required_forwards);
    ensure!(
        capacity >= required_forwards,
        "--capacity {capacity} is below required forward count {required_forwards}"
    );
    ensure!(
        capacity <= config.context_length as usize,
        "--capacity {capacity} exceeds Muse model context {}",
        config.context_length
    );
    let stop_tokens = gguf
        .stop_token_ids()
        .context("load Muse producer stop tokens")?;
    let eos_token_id = config.eos_token_id as i32;
    let eot_token_id = config.eot_token_id as i32;
    ensure!(
        stop_tokens == [eos_token_id, eot_token_id],
        "Muse benchmark requires ordered stop tokens [{eos_token_id}, {eot_token_id}], got {stop_tokens:?}"
    );
    let (sampling, overridden_fields) = effective_sampling(&args)?;

    if !json_mode {
        eprintln!(
            "[muse-request] model={} prompt_tokens={} max_sampled_tokens={} capacity={} reasoning={} sampling=temp:{:.3}/top_k:{}/top_p:{:.3}/min_p:{:.3}/seed:{}",
            args.model.display(),
            prompt_tokens.len(),
            args.tokens,
            capacity,
            effective_strength.as_str(),
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            sampling.min_p,
            sampling.seed,
        );
    }

    let metal_started = Instant::now();
    let ctx = MetalContext::new().context("initialize Metal for Muse benchmark")?;
    let metal_context_ns = elapsed_ns(metal_started);
    if !json_mode {
        eprintln!("[muse-request] device: {}", ctx.describe());
    }
    let load_started = Instant::now();
    let mut loaded = MuseGlimmerLoadedModel::load(&ctx, &gguf, capacity)
        .context("load resident Muse benchmark model")?;
    let resident_model_load_ns = elapsed_ns(load_started);
    let artifact_profile = loaded.artifact_profile();
    let admission = loaded.admission();
    let observed_weight_bytes = loaded.observed_weight_bytes();
    let observed_session_bytes = loaded.observed_session_bytes();
    let mut runner = loaded
        .create_runner(&ctx)
        .context("bind Muse benchmark execution graph")?;

    let warmup = if args.no_warmup {
        None
    } else {
        let iteration = run_iteration(
            0,
            &mut runner,
            &tokenizer,
            &prompt_tokens,
            args.tokens,
            eos_token_id,
            eot_token_id,
            config.vocab_size,
            sampling,
        )?;
        if !json_mode {
            eprintln!(
                "[muse-request] warmup: sampled={} emitted={} transitions={} termination={}",
                iteration.shape.sampled_tokens,
                iteration.shape.emitted_tokens,
                iteration.shape.transition_forwards,
                iteration.shape.termination.kind,
            );
        }
        Some(iteration.shape)
    };

    let mut iterations = Vec::with_capacity(args.runs);
    for repetition in 1..=args.runs {
        let iteration = run_iteration(
            repetition,
            &mut runner,
            &tokenizer,
            &prompt_tokens,
            args.tokens,
            eos_token_id,
            eot_token_id,
            config.vocab_size,
            sampling,
        )?;
        if !json_mode {
            eprintln!(
                "[muse-request] rep {repetition}: reset={:.3} ms prefill={:.3} ms generation={:.3} ms transitions={:.3} ms sampled={} emitted={} forwards={} termination={}",
                iteration.sample.resident_position_reset_ns as f64 / 1e6,
                iteration.sample.prefill_wall_ns as f64 / 1e6,
                iteration.sample.generation_wall_ns as f64 / 1e6,
                iteration.sample.transition_forward_ns as f64 / 1e6,
                iteration.sample.sampled_tokens,
                iteration.sample.emitted_tokens,
                iteration.sample.transition_forwards,
                iteration.sample.termination.kind,
            );
        }
        iterations.push(iteration);
    }

    let baseline_shape = warmup
        .as_ref()
        .cloned()
        .unwrap_or_else(|| iterations[0].shape.clone());
    let shape_consistent = iterations
        .iter()
        .all(|iteration| iteration.shape == baseline_shape);
    let samples = iterations
        .iter()
        .map(|iteration| iteration.sample.clone())
        .collect::<Vec<_>>();
    let aggregate = aggregate(&samples);
    let last = iterations.last().expect("positive run count");
    let last_output = LastOutput {
        sampled_token_ids: last.sampled_token_ids.clone(),
        emitted_token_ids: last.emitted_token_ids.clone(),
        emitted_bytes_sha256: sha256_hex(&last.output_bytes),
        emitted_utf8_valid: std::str::from_utf8(&last.output_bytes).is_ok(),
        emitted_text_lossy: String::from_utf8_lossy(&last.output_bytes).into_owned(),
    };
    let build_identity = super::recorded_build_identity();
    let canonical_build = build_identity.status == "match" && build_identity.overrides.is_empty();
    let steady_state_qualified = warmup.is_some() && shape_consistent;
    let qualification = Qualification {
        released_model_profile: true,
        tokenizer_contract: true,
        ordered_stop_contract: true,
        input_atem_ready: true,
        output_atem_validation: "not_performed",
        shape_consistent,
        steady_state_qualified,
        canonical_build,
        workload_qualified: canonical_build && steady_state_qualified,
        llama_bench_comparable: false,
    };
    let report = Report {
        schema: SCHEMA,
        schema_version: SCHEMA_VERSION,
        benchmark_kind: "resident_real_request",
        test_time: super::utc_iso8601_now(),
        build_identity,
        device: ctx.describe(),
        qwen_env: super::capture_qwen_env(),
        power,
        model: ModelInfo {
            path: args.model.display().to_string(),
            architecture: qwen_llm::muse_glimmer::ARCHITECTURE_NAME,
            artifact_profile: artifact_profile_label(artifact_profile),
            weight_bytes: super::model_weight_bytes(&gguf),
            parameter_count: gguf
                .get_u64("general.parameter_count")
                .unwrap_or_else(|| gguf.tensors.iter().map(|tensor| tensor.n_elements()).sum()),
            tokenizer_identity_sha256: fixed_sha256_hex(config.tokenizer_identity_sha256),
            chat_template_profile: chat_template_profile_label(config.chat_template_profile),
            chat_template_sha256: fixed_sha256_hex(config.chat_template_sha256),
            stop_token_ids: [eos_token_id, eot_token_id],
            observed_weight_allocation_bytes: observed_weight_bytes,
            observed_session_allocation_bytes: observed_session_bytes,
            aggregate_required_bytes: admission.aggregate.required_bytes,
            weight_required_bytes: admission.weights.required_bytes,
            session_required_bytes: admission.session.required_bytes,
        },
        request: RequestInfo {
            input_kind,
            rendered_prompt_sha256: sha256_hex(rendered_prompt.as_bytes()),
            prompt_token_ids_sha256_i32le: token_ids_sha256_i32le(&prompt_token_ids),
            prompt_tokens: prompt_tokens.len(),
            cli_reasoning_strength: requested_strength.map(|strength| strength.as_str()),
            document_reasoning_strength: request
                .reasoning_strength
                .map(|strength| strength.as_str()),
            effective_reasoning_strength: effective_strength.as_str(),
            maximum_sampled_tokens: args.tokens,
            required_forwards,
            resident_capacity: capacity,
        },
        sampling: SamplingInfo {
            algorithm_version: SAMPLER_ALGORITHM_VERSION,
            temperature: sampling.temperature,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
            min_p: sampling.min_p,
            seed: sampling.seed,
            overridden_fields,
        },
        setup: SetupInfo {
            gguf_open_ns,
            release_contract_bind_ns,
            request_render_ns,
            tokenizer_open_ns,
            tokenization_ns,
            metal_context_ns,
            resident_model_load_ns,
        },
        method: MethodInfo {
            warmup_repetitions: usize::from(!args.no_warmup),
            timed_repetitions: args.runs,
            warmup_scope: "full_request",
            sampler_scope: "fresh_from_effective_config_per_iteration",
            session_scope: "one_resident_session_position_reset_per_iteration",
            prefill_mode: muse_prefill_mode(prompt_tokens.len()),
            decode_mode: "serial_full_logits",
            output_sink: "in_memory_exact_token_bytes",
        },
        warmup_shape: warmup,
        samples,
        aggregate,
        qualification,
        last_output,
    };

    if json_mode {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).context("serialize Muse benchmark report")?
        );
    } else {
        eprintln!(
            "[muse-request] aggregate: prefill={:.2} forwards/s transitions={} sampled={:.2} tokens/s emitted={:.2} tokens/s shape_consistent={} workload_qualified={}",
            report.aggregate.prefill_forwards_per_second,
            report
                .aggregate
                .transition_forwards_per_second
                .map(|value| format!("{value:.2} forwards/s"))
                .unwrap_or_else(|| "n/a".into()),
            report.aggregate.sampled_tokens_per_second,
            report.aggregate.emitted_tokens_per_second,
            report.qualification.shape_consistent,
            report.qualification.workload_qualified,
        );
        eprintln!(
            "[muse-request] last emitted text: {:?}",
            report.last_output.emitted_text_lossy
        );
    }
    Ok(())
}

fn load_request(args: &MuseRequestArgs) -> Result<(MuseGlimmerRequest, &'static str)> {
    match &args.messages {
        Some(path) => {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("read Muse request {}", path.display()))?;
            Ok((
                MuseGlimmerRequest::from_json(&raw)
                    .with_context(|| format!("parse Muse request {}", path.display()))?,
                "messages",
            ))
        }
        None => Ok((
            MuseGlimmerRequest::single_turn(
                args.prompt.clone().unwrap_or_else(|| DEFAULT_PROMPT.into()),
                args.system.clone(),
            ),
            "single_turn",
        )),
    }
}

fn effective_sampling(args: &MuseRequestArgs) -> Result<(SamplingConfig, Vec<&'static str>)> {
    let mut config = SamplingConfig::muse_glimmer(args.seed);
    let mut overrides = Vec::new();
    if let Some(value) = args.temperature {
        config.temperature = value;
        overrides.push("temperature");
    }
    if let Some(value) = args.top_k {
        config.top_k = value;
        overrides.push("top_k");
    }
    if let Some(value) = args.top_p {
        config.top_p = value;
        overrides.push("top_p");
    }
    if let Some(value) = args.min_p {
        config.min_p = value;
        overrides.push("min_p");
    }
    Ok((
        config
            .validate()
            .map_err(anyhow::Error::new)
            .context("validate Muse benchmark sampling")?,
        overrides,
    ))
}

fn required_forwards(prompt_tokens: usize, max_tokens: usize) -> Result<usize> {
    ensure!(prompt_tokens > 0, "Muse prompt tokenized to zero tokens");
    ensure!(max_tokens > 0, "--tokens must be >= 1");
    prompt_tokens
        .checked_add(max_tokens - 1)
        .context("Muse benchmark forward count overflow")
}

fn muse_prefill_mode(prompt_tokens: usize) -> &'static str {
    let packed_tokens =
        prompt_tokens / MUSE_GLIMMER_PACKED_PREFILL_QUANTUM * MUSE_GLIMMER_PACKED_PREFILL_QUANTUM;
    match (packed_tokens, prompt_tokens - packed_tokens) {
        (0, _) => "scalar_full_logits",
        (_, 0) => "packed_exact_full_logits",
        _ => "packed_exact_plus_scalar_tail_full_logits",
    }
}

fn checked_token_id(token: i32, vocab_size: u32, context: &str) -> Result<u32> {
    let token = u32::try_from(token)
        .with_context(|| format!("Muse {context} token {token} is negative"))?;
    ensure!(
        token < vocab_size,
        "Muse {context} token {token} exceeds vocabulary {vocab_size}"
    );
    Ok(token)
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn sha256_hex(bytes: &[u8]) -> String {
    fixed_sha256_hex(Sha256::digest(bytes).into())
}

fn fixed_sha256_hex(digest: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(output, "{byte:02x}").expect("write to String");
    }
    output
}

fn artifact_profile_label(profile: MuseGlimmerArtifactProfile) -> &'static str {
    match profile {
        MuseGlimmerArtifactProfile::UnslothQ8_0 => "unsloth_q8_0",
        MuseGlimmerArtifactProfile::UnslothBf16 => "unsloth_bf16",
    }
}

fn chat_template_profile_label(profile: MuseGlimmerChatTemplateProfile) -> &'static str {
    match profile {
        MuseGlimmerChatTemplateProfile::MetaFixed => "meta_fixed",
        MuseGlimmerChatTemplateProfile::UnslothLaunch => "unsloth_launch",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn parse(extra: &[&str]) -> MuseRequestArgs {
        let mut args = vec!["muse-request", "--model", "model.gguf"];
        args.extend_from_slice(extra);
        MuseRequestArgs::try_parse_from(args).unwrap()
    }

    #[test]
    fn command_defaults_to_released_sampling_and_high_reasoning() {
        let args = parse(&[]);
        let (sampling, overrides) = effective_sampling(&args).unwrap();
        assert_eq!(sampling, SamplingConfig::muse_glimmer(42));
        assert!(overrides.is_empty());
        let request = MuseGlimmerRequest::single_turn(DEFAULT_PROMPT, None);
        assert_eq!(
            request.resolved_reasoning_strength(None).unwrap(),
            MuseGlimmerReasoningStrength::High
        );
        assert_eq!(args.tokens, 64);
        assert_eq!(args.runs, 1);
    }

    #[test]
    fn qualification_names_workload_shape_without_claiming_performance() {
        let qualification = Qualification {
            released_model_profile: true,
            tokenizer_contract: true,
            ordered_stop_contract: true,
            input_atem_ready: true,
            output_atem_validation: "not_performed",
            shape_consistent: true,
            steady_state_qualified: true,
            canonical_build: true,
            workload_qualified: true,
            llama_bench_comparable: false,
        };
        let value = serde_json::to_value(qualification).unwrap();
        assert_eq!(SCHEMA_VERSION, 2);
        assert_eq!(value["workload_qualified"], true);
        assert!(value.get("performance_qualified").is_none());
    }

    #[test]
    fn sampling_overrides_are_independent_and_recorded() {
        let args = parse(&[
            "--temperature",
            "0",
            "--top-k",
            "0",
            "--top-p",
            "1",
            "--min-p",
            "0.1",
            "--seed",
            "7",
        ]);
        let (sampling, overrides) = effective_sampling(&args).unwrap();
        assert_eq!(sampling.temperature, 0.0);
        assert_eq!(sampling.top_k, 0);
        assert_eq!(sampling.top_p, 1.0);
        assert_eq!(sampling.min_p, 0.1);
        assert_eq!(sampling.seed, 7);
        assert_eq!(overrides, ["temperature", "top_k", "top_p", "min_p"]);
    }

    #[test]
    fn generation_accounts_for_stops_and_unforwarded_final_tokens() {
        let run = |tokens: &[i32], limit: usize| {
            let mut tokens = VecDeque::from(tokens.to_vec());
            let mut forwarded = Vec::new();
            let mut emitted = Vec::new();
            let result = drive_generation(
                (),
                limit,
                1,
                2,
                |_| Ok(tokens.pop_front().unwrap()),
                |token| {
                    forwarded.push(token);
                    Ok(())
                },
                |token| {
                    emitted.push(token);
                    Ok(())
                },
                || Ok(()),
            )
            .unwrap();
            (result, forwarded, emitted)
        };

        let (eot, forwarded, emitted) = run(&[10, 2], 4);
        assert_eq!(eot.sampled_token_ids, [10, 2]);
        assert_eq!(emitted, [10]);
        assert_eq!(forwarded, [10]);
        assert_eq!(eot.transition_forwards, 1);
        assert_eq!(eot.termination.kind, "eot_token");

        let (limit, forwarded, emitted) = run(&[10, 11], 2);
        assert_eq!(limit.sampled_token_ids, [10, 11]);
        assert_eq!(emitted, [10, 11]);
        assert_eq!(forwarded, [10]);
        assert_eq!(limit.transition_forwards, 1);
        assert_eq!(limit.termination.kind, "token_limit");
        assert_eq!(limit.termination.token_id, None);

        let (eos, forwarded, emitted) = run(&[1], 4);
        assert_eq!(eos.sampled_token_ids, [1]);
        assert!(emitted.is_empty());
        assert!(forwarded.is_empty());
        assert_eq!(eos.transition_forwards, 0);
        assert_eq!(eos.termination.kind, "eos_token");
    }

    #[test]
    fn command_preserves_all_released_reasoning_strengths() {
        for (label, expected) in [
            ("low", MuseGlimmerReasoningStrength::Low),
            ("medium", MuseGlimmerReasoningStrength::Medium),
            ("high", MuseGlimmerReasoningStrength::High),
            ("xhigh", MuseGlimmerReasoningStrength::Xhigh),
        ] {
            let args = parse(&["--reasoning-strength", label]);
            assert_eq!(
                args.reasoning_strength
                    .map(MuseReasoningStrengthArg::resolve),
                Some(expected)
            );
        }
    }

    #[test]
    fn forward_capacity_counts_prompt_and_only_possible_transitions() {
        assert_eq!(required_forwards(10, 1).unwrap(), 10);
        assert_eq!(required_forwards(10, 4).unwrap(), 13);
        assert!(required_forwards(0, 1).is_err());
        assert!(required_forwards(1, 0).is_err());
    }

    #[test]
    fn prefill_metadata_names_the_executed_prompt_shape() {
        assert_eq!(muse_prefill_mode(15), "scalar_full_logits");
        assert_eq!(muse_prefill_mode(16), "packed_exact_full_logits");
        assert_eq!(
            muse_prefill_mode(17),
            "packed_exact_plus_scalar_tail_full_logits"
        );
    }

    #[test]
    fn root_parser_exposes_isolated_muse_request_command() {
        let parsed = super::super::Args::try_parse_from([
            "qwen-bench",
            "muse-request",
            "--model",
            "model.gguf",
            "--reasoning-strength",
            "xhigh",
        ])
        .unwrap();
        assert!(matches!(
            parsed.cmd,
            super::super::Cmd::MuseRequest(MuseRequestArgs {
                reasoning_strength: Some(MuseReasoningStrengthArg::Xhigh),
                ..
            })
        ));
    }
}
