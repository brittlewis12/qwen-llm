//! Muse Glimmer single-turn path.

use super::*;

pub(crate) fn prepare_muse_glimmer_prompt(
    invocation: cli::Invocation,
    config: &MuseGlimmerConfig,
    args: &Args,
) -> Result<MuseGlimmerPreparedPrompt> {
    match invocation {
        cli::Invocation::Run(run) => {
            ensure!(
                !run.no_thinking,
                "Muse Glimmer does not declare a no-thinking ATEM profile; use --reasoning-effort low for the lightest supported reasoning mode"
            );
            let reasoning_strength = run
                .reasoning_effort
                .map(resolve_muse_glimmer_reasoning_strength);
            match run.acquire_input()? {
                cli::AcquiredRunInput::RawPrompt(text) => Ok(MuseGlimmerPreparedPrompt {
                    text,
                    source: PromptSource::Inline,
                    add_special_tokens: !args.no_special_tokens,
                }),
                cli::AcquiredRunInput::User { system, user } => {
                    let text = MuseGlimmerRequest::single_turn(user, system)
                        .render(config.chat_template_profile, reasoning_strength)
                        .context("render Muse Glimmer ATEM user request")?;
                    Ok(MuseGlimmerPreparedPrompt {
                        text,
                        source: PromptSource::Messages,
                        add_special_tokens: false,
                    })
                }
                cli::AcquiredRunInput::Messages { document, source } => {
                    let text = MuseGlimmerRequest::from_json(&document)
                        .with_context(|| format!("parse Muse Glimmer messages from {source}"))?
                        .render(config.chat_template_profile, reasoning_strength)
                        .context("render Muse Glimmer ATEM messages")?;
                    Ok(MuseGlimmerPreparedPrompt {
                        text,
                        source: PromptSource::Messages,
                        add_special_tokens: false,
                    })
                }
            }
        }
        cli::Invocation::Legacy => {
            ensure!(
                args.messages.is_none(),
                "Muse Glimmer legacy --messages rendering is not supported; use `qwen run --messages`"
            );
            let (text, source, _) = prompt_text(args)?;
            Ok(MuseGlimmerPreparedPrompt {
                text,
                source,
                add_special_tokens: prompt_add_special_tokens(args, source),
            })
        }
        cli::Invocation::Serve(_) => bail!("Muse Glimmer serve routing is not implemented"),
    }
}

pub(crate) fn resolve_muse_glimmer_reasoning_strength(
    requested: cli::RunReasoningEffort,
) -> MuseGlimmerReasoningStrength {
    match requested {
        cli::RunReasoningEffort::Low => MuseGlimmerReasoningStrength::Low,
        cli::RunReasoningEffort::Medium => MuseGlimmerReasoningStrength::Medium,
        cli::RunReasoningEffort::High => MuseGlimmerReasoningStrength::High,
        cli::RunReasoningEffort::Xhigh => MuseGlimmerReasoningStrength::Xhigh,
    }
}

pub(crate) fn validate_muse_glimmer_generation_mode(
    args: &Args,
    explicit: ExplicitCliOptions,
    invocation: &cli::Invocation,
) -> Result<()> {
    let mut unsupported = serial_lane_unsupported_options(
        args,
        explicit,
        "--drafter (Muse DFlash2 integration is not active yet)",
    );
    ensure_no_deepseek_v4_only_options(args, explicit, &mut unsupported)?;
    ensure!(
        unsupported.is_empty(),
        "Muse Glimmer currently supports request-shaped serial text generation only; unsupported options: {}",
        unsupported.join(", ")
    );
    ensure!(
        matches!(invocation, cli::Invocation::Run(_)) || has_single_turn_input(args),
        "Muse Glimmer generation requires --prompt, --prompt-file, or `qwen run --user|--messages|--raw-prompt`"
    );
    Ok(())
}

pub(crate) fn run_muse_glimmer_single_turn(
    model_path: &Path,
    gguf: &GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
    invocation: cli::Invocation,
) -> Result<()> {
    validate_muse_glimmer_generation_mode(args, explicit, &invocation)?;
    let request_t0 = Instant::now();
    let sampling = muse_glimmer_sampling_config(args, explicit)?;
    let config =
        MuseGlimmerConfig::from_gguf(gguf).context("bind Muse Glimmer release contract")?;
    let prepared = prepare_muse_glimmer_prompt(invocation, &config, args)?;

    let tokenizer_t0 = Instant::now();
    let tokenizer = LlamaCppTokenizer::open(model_path).context("load Muse Glimmer tokenizer")?;
    ensure!(
        tokenizer.n_vocab() == config.vocab_size,
        "Muse Glimmer tokenizer vocabulary {} differs from model vocabulary {}",
        tokenizer.n_vocab(),
        config.vocab_size
    );
    ensure!(
        tokenizer.bos() == Some(config.bos_token_id as i32),
        "Muse Glimmer tokenizer BOS {:?} differs from model BOS {}",
        tokenizer.bos(),
        config.bos_token_id
    );
    ensure!(
        tokenizer.eos() == Some(config.eos_token_id as i32),
        "Muse Glimmer tokenizer EOS {:?} differs from model EOS {}",
        tokenizer.eos(),
        config.eos_token_id
    );
    let prompt_ids = tokenizer
        .encode(&prepared.text, prepared.add_special_tokens)
        .context("tokenize Muse Glimmer prompt")?;
    let tokenizer_ms = tokenizer_t0.elapsed().as_secs_f64() * 1e3;
    let prompt_tokens = prompt_ids
        .into_iter()
        .enumerate()
        .map(|(index, token)| {
            checked_token_id(token, config.vocab_size, &format!("prompt[{index}]"))
        })
        .collect::<Result<Vec<_>>>()?;
    let required_forwards =
        required_forwards("Muse Glimmer", prompt_tokens.len(), args.tokens, None)?;
    let capacity = args.max_context_tokens.unwrap_or(required_forwards);
    ensure!(
        capacity >= required_forwards,
        "Muse Glimmer request requires {required_forwards} forwards, beyond --max-context-tokens {capacity}"
    );
    ensure!(
        capacity <= config.context_length as usize,
        "Muse Glimmer --max-context-tokens {capacity} exceeds model context {}",
        config.context_length
    );
    let stop_tokens = gguf
        .stop_token_ids()
        .context("load producer-declared Muse Glimmer stop tokens")?;
    let expected_stop_tokens = vec![config.eos_token_id as i32, config.eot_token_id as i32];
    ensure!(
        stop_tokens == expected_stop_tokens,
        "Muse Glimmer release stop tokens must be EOS/EOT {expected_stop_tokens:?}, got {stop_tokens:?}"
    );
    let prefill_packed_tokens = prompt_tokens.len() / MUSE_GLIMMER_PACKED_PREFILL_QUANTUM
        * MUSE_GLIMMER_PACKED_PREFILL_QUANTUM;
    let prefill_scalar_tail_commands = prompt_tokens.len() - prefill_packed_tokens;
    let prefill_commands = prefill_packed_tokens / MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS
        + usize::from(
            !prefill_packed_tokens.is_multiple_of(MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS),
        )
        + prefill_scalar_tail_commands;
    let prefill_mode = match (prefill_packed_tokens, prefill_scalar_tail_commands) {
        (0, _) => "scalar_tail",
        (_, 0) => "packed_exact",
        _ => "packed_exact+scalar_tail",
    };

    eprintln!(
        "muse_glimmer: loading {} for text generation; prompt_source={:?} prompt_tokens={} max_generated_tokens={} forward_capacity={} prefill_mode={} prefill_commands={} prefill_packed_tokens={} prefill_scalar_tail_commands={}",
        model_path.display(),
        prepared.source,
        prompt_tokens.len(),
        args.tokens,
        capacity,
        prefill_mode,
        prefill_commands,
        prefill_packed_tokens,
        prefill_scalar_tail_commands,
    );
    let load_t0 = Instant::now();
    let ctx = MetalContext::new().context("initialize Metal for Muse Glimmer")?;
    let mut loaded = MuseGlimmerLoadedModel::load(&ctx, gguf, capacity)
        .context("load admitted Muse Glimmer weights and text session")?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
    let admission = loaded.admission();
    eprintln!(
        "muse_glimmer: resident on {} in {:.1} ms; aggregate_required={:?} weight_required={:?} weight_observed={} session_required={:?} session_observed={}",
        ctx.describe(),
        load_ms,
        admission.aggregate.required_bytes,
        admission.weights.required_bytes,
        loaded.observed_weight_bytes(),
        admission.session.required_bytes,
        loaded.observed_session_bytes(),
    );
    let mut runner = loaded
        .create_runner(&ctx)
        .context("bind Muse Glimmer execution graph")?;

    let prefill_t0 = Instant::now();
    let logits = runner
        .prefill_with_command_checkpoint(&prompt_tokens, || {
            shutdown::checkpoint().map_err(|error| error.to_string())
        })
        .context("prefill Muse Glimmer prompt")?;
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

    let mut sampler = Sampler::new(sampling).context("initialize Muse Glimmer sampler")?;
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
                .with_context(|| format!("decode Muse Glimmer token {token}"))?;
            stdout
                .write_all(&piece)
                .with_context(|| format!("write Muse Glimmer token {token}"))?;
            stdout.flush().context("flush Muse Glimmer token")?;
            Ok(())
        },
        |token| {
            let token = checked_token_id(token, config.vocab_size, "generated")?;
            runner
                .forward_token(token)
                .context("forward generated Muse Glimmer token")
        },
    )?;
    if !generation.tokens.is_empty() {
        writeln!(stdout)?;
        stdout.flush().context("flush Muse Glimmer final newline")?;
    }
    let prefill_tps = if prefill_ms > 0.0 {
        prompt_tokens.len() as f64 / (prefill_ms / 1e3)
    } else {
        0.0
    };
    let decode_tps = if generation.wall_ms > 0.0 {
        generation.tokens.len() as f64 / (generation.wall_ms / 1e3)
    } else {
        0.0
    };
    eprintln!(
        "muse_glimmer stats: prompt_tokens={} generated_tokens={} transitions={} stop_reason={} tokenizer_ms={:.1} load_ms={:.1} prefill_mode={} prefill_ms={:.1} prefill_tps={:.2} prefill_commands={} prefill_packed_tokens={} prefill_scalar_tail_commands={} generation_ms={:.1} decode_tps={:.2} total_ms={:.1}",
        prompt_tokens.len(),
        generation.tokens.len(),
        generation.transitions,
        generation.stop_reason.as_str(),
        tokenizer_ms,
        load_ms,
        prefill_mode,
        prefill_ms,
        prefill_tps,
        prefill_commands,
        prefill_packed_tokens,
        prefill_scalar_tail_commands,
        generation.wall_ms,
        decode_tps,
        request_t0.elapsed().as_secs_f64() * 1e3,
    );
    Ok(())
}
