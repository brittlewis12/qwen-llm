//! Serial/greedy/sampled decode loops, prefill spans, prefix fan-out, and sampling policy.

use super::*;

pub(crate) fn cli_sampling_config(args: &Args) -> Result<SamplingConfig> {
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

pub(crate) fn muse_glimmer_sampling_config(
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<SamplingConfig> {
    let mut config = SamplingConfig::muse_glimmer(args.seed);
    if explicit.temperature {
        config.temperature = args.temperature;
    }
    if explicit.top_k {
        config.top_k = args.top_k;
    }
    if explicit.top_p {
        config.top_p = args.top_p;
    }
    if explicit.min_p {
        config.min_p = args.min_p;
    }
    config
        .validate()
        .map_err(anyhow::Error::new)
        .context("validate Muse Glimmer sampling configuration")
}

pub(crate) fn request_sampling_config(
    request: &JsonlRequest,
    args: &Args,
) -> Result<SamplingConfig> {
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

pub(crate) fn validate_sampling_decode_policy(
    config: SamplingConfig,
    prompt_lookup: bool,
) -> Result<()> {
    ensure!(
        !prompt_lookup || config.temperature == 0.0,
        "--prompt-lookup currently requires greedy decoding (--temp 0)"
    );
    Ok(())
}

/// **v0.77** admission gate for `--drafter` (DFlash speculative decode).
///
/// The remaining exclusions are all about the drafter's append-only
/// cross-context: it conditions on captured target hidden states for
/// every committed position, so any path that advances the target KV
/// without hidden capture (durable-prefix restore, JSONL prefix-cache
/// reuse) would leave an unfillable hole.
pub(crate) fn validate_drafter_decode_policy(args: &Args) -> Result<()> {
    if args.drafter.is_none() {
        return Ok(());
    }
    ensure!(
        !args.prompt_lookup,
        "--drafter and --prompt-lookup are mutually exclusive draft sources"
    );
    ensure!(
        args.durable_prefix_cache.is_none(),
        "--drafter is incompatible with --durable-prefix-cache (restored positions \
         carry no captured target hidden states for the drafter's cross-context)"
    );
    ensure!(
        args.requests_jsonl.is_none(),
        "--drafter is single-turn only; JSONL request mode is not supported yet"
    );
    ensure!(
        !args.sampling_attribution && !args.sampled_structural,
        "--drafter is incompatible with sampling attribution and structural sampling"
    );
    Ok(())
}

pub(crate) fn decode_policy_label(
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

pub(crate) fn jsonl_decode_policy_label(
    config: SamplingConfig,
    prompt_lookup: bool,
    gpu_greedy: bool,
) -> &'static str {
    decode_policy_label(config, prompt_lookup, gpu_greedy)
}

/// **v0.77** capture-aware prefill for DFlash speculative decode.
///
/// Identical to [`prefill_span`] except it asks the target to snapshot the
/// K drafter-conditioning layers for every prompt position into `hidden_dst`
/// (`[n_tokens, K*H]` contiguous). The drafter's cross-context is
/// append-only over committed positions, so the prompt must be captured
/// here or the drafter starts blind.
pub(crate) fn prefill_span_with_capture(
    forward: &MetalForward<'_>,
    sequence: &mut Sequence,
    scratch: &mut MetalDFlashLayerMajorScratch,
    token_ids: &[i32],
    start_position: usize,
    target_layer_ids: &[u32],
    hidden_dst: &MetalTensor,
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
        target_layer_ids,
        Some(hidden_dst),
    )
    .context("prefill prompt span with drafter hidden capture")?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    sequence.advance_by(token_ids.len())?;
    Ok((logits, ms))
}

pub(crate) fn prefill_span(
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

/// `prefill_span` through the runtime's owned facade: the model checks
/// provenance, takes the position from the sequence, and poisons the state
/// on failure. `start_position` remains an explicit assertion of what the
/// caller believes the sequence position is.
pub(crate) fn prefill_owned(
    loaded: &LoadedModel,
    sequence: &mut Sequence,
    scratch: &mut PackedPrefillScratch,
    token_ids: &[i32],
    start_position: usize,
) -> Result<(Vec<f32>, f64)> {
    shutdown::checkpoint()?;
    ensure!(!token_ids.is_empty(), "cannot prefill an empty token span");
    sequence.check_position(start_position)?;
    let t0 = Instant::now();
    let logits = loaded
        .prefill(sequence, scratch, token_ids)
        .context("prefill prompt span")?;
    shutdown::checkpoint()?;
    Ok((logits, t0.elapsed().as_secs_f64() * 1e3))
}

pub(crate) const QWEN_PREFIX_FANOUT_EXACT_LCP_ENV: &str = "QWEN_PREFIX_FANOUT_EXACT_LCP";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QwenPrefixFanoutBoundaryPolicy {
    ChunkAligned,
    TinySuffixExactLcp,
    ExactLcp,
}

impl QwenPrefixFanoutBoundaryPolicy {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ChunkAligned => "chunk_aligned",
            Self::TinySuffixExactLcp => "tiny_suffix_exact_lcp",
            Self::ExactLcp => "exact_lcp",
        }
    }
}

pub(crate) fn parse_qwen_prefix_fanout_boundary_policy(
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

pub(crate) fn qwen_prefix_fanout_boundary_policy() -> Result<QwenPrefixFanoutBoundaryPolicy> {
    let value = std::env::var_os(QWEN_PREFIX_FANOUT_EXACT_LCP_ENV)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("{QWEN_PREFIX_FANOUT_EXACT_LCP_ENV} must be valid UTF-8"))
        })
        .transpose()?;
    parse_qwen_prefix_fanout_boundary_policy(value.as_deref())
}

pub(crate) const PRIVATE_SUFFIX_SINGLETON_ENV: &str = "QWEN_PRIVATE_SUFFIX_SINGLETON";

pub(crate) const PRIVATE_SUFFIX_SINGLETON_MAX_TOKENS: usize = 6;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrivateSuffixExecutionMode {
    Packed,
    Singleton,
}

pub(crate) struct PrivateSuffixResult {
    pub(crate) logits: Vec<f32>,
    pub(crate) ms: f64,
    pub(crate) mode: PrivateSuffixExecutionMode,
}

pub(crate) fn parse_private_suffix_singleton_enabled(
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

pub(crate) fn private_suffix_singleton_enabled(default_enabled: bool) -> Result<bool> {
    let value = std::env::var_os(PRIVATE_SUFFIX_SINGLETON_ENV)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("{PRIVATE_SUFFIX_SINGLETON_ENV} must be valid UTF-8"))
        })
        .transpose()?;
    parse_private_suffix_singleton_enabled(value.as_deref(), default_enabled)
}

pub(crate) fn choose_private_suffix_execution_mode(
    enabled: bool,
    suffix_tokens: usize,
) -> PrivateSuffixExecutionMode {
    if enabled && suffix_tokens <= PRIVATE_SUFFIX_SINGLETON_MAX_TOKENS {
        PrivateSuffixExecutionMode::Singleton
    } else {
        PrivateSuffixExecutionMode::Packed
    }
}

pub(crate) fn prefill_private_suffix(
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
pub(crate) struct GenerationResult {
    pub(crate) tokens: Vec<i32>,
    pub(crate) wall_ms: f64,
    pub(crate) first_token_selection_ms: f64,
    pub(crate) first_token_ready_ms: Option<f64>,
    pub(crate) first_token_callback_ms: Option<f64>,
    pub(crate) transitions: usize,
    pub(crate) transition_ms: f64,
    pub(crate) first_transition_ms: Option<f64>,
    pub(crate) stop_reason: StopReason,
}

pub(crate) fn generate_serial<OnToken, Transition>(
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

pub(crate) fn generate_serial_attributed<OnToken, Transition>(
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
pub(crate) enum GreedyDecodeState {
    PromptLogits(Vec<f32>),
    Device(GreedySelection),
}

pub(crate) fn generate_gpu_greedy<OnToken, Transition>(
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
pub(crate) enum SampledStructuralDecodeState {
    PromptLogits(Vec<f32>),
    Selected(std::result::Result<SampledToken, SamplingError>),
}

pub(crate) struct SampledStructuralContext<'a> {
    pub(crate) sampler: &'a mut Sampler,
    pub(crate) telemetry: SampledStructuralTelemetry,
}

pub(crate) fn with_transactional_sampled_structural_context<R, F>(
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

pub(crate) fn generate_sampled_structural<OnToken, Transition>(
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

pub(crate) fn generate_serial_state<State, Select, OnToken, Transition>(
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

pub(crate) fn generate_serial_state_with_context<Context, State, Select, OnToken, Transition>(
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
pub(crate) fn generate_greedy<OnToken, Transition>(
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

pub(crate) fn decode_serial(
    loaded: &LoadedModel,
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
                loaded
                    .decode_token_greedy(sequence, token)
                    .context("decode token with GPU greedy selection")
            },
        )?
    } else {
        generate_serial(
            logits,
            max_tokens,
            stop_tokens,
            sampler,
            &mut on_token,
            |token| loaded.decode_token(sequence, token).context("decode token"),
        )?
    };
    Ok((generation, generated_text))
}

pub(crate) fn decode_prompt_lookup(
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

/// Convert a sampler-produced `i32` token into a vocabulary-bounded ID for
/// runtimes that consume `u32` tokens.
pub(crate) fn checked_token_id(token: i32, vocab_size: u32, purpose: &str) -> Result<u32> {
    let token =
        u32::try_from(token).with_context(|| format!("{purpose} token ID {token} is negative"))?;
    ensure!(
        token < vocab_size,
        "{purpose} token ID {token} is outside vocabulary {vocab_size}"
    );
    Ok(token)
}

/// Token forwards a request needs: every prompt token plus one transition per
/// generated token after the first. `capacity` bounds the total when the
/// session is promoted for a fixed forward budget.
pub(crate) fn required_forwards(
    family: &str,
    prompt_tokens: usize,
    max_tokens: usize,
    capacity: Option<usize>,
) -> Result<usize> {
    ensure!(
        prompt_tokens > 0,
        "{family} prompt tokenized to zero tokens"
    );
    ensure!(max_tokens > 0, "--tokens must be >= 1");
    let decode_transitions = max_tokens - 1;
    let required = prompt_tokens
        .checked_add(decode_transitions)
        .with_context(|| format!("{family} forward budget overflow"))?;
    if let Some(capacity) = capacity {
        ensure!(
            required <= capacity,
            "{family} request requires {required} token forwards ({prompt_tokens} prompt + {decode_transitions} maximum decode transitions), but the native session is promoted for {capacity} forwards; shorten the prompt or reduce --tokens",
        );
    }
    Ok(required)
}
