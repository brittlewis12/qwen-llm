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
            // Omission is preserved: the messages document may carry its own
            // strength, and the library merges request, document, then the
            // `high` fallback.
            let reasoning_strength = run
                .reasoning_effort
                .as_deref()
                .map(resolve_muse_glimmer_reasoning_strength)
                .transpose()?;
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
        cli::Invocation::Serve(_) | cli::Invocation::Info(_) => {
            bail!("Muse Glimmer prompt preparation is a run/legacy path")
        }
    }
}

pub(crate) fn resolve_muse_glimmer_reasoning_strength(
    requested: &str,
) -> Result<MuseGlimmerReasoningStrength> {
    MuseGlimmerReasoningStrength::parse(requested).ok_or_else(|| {
        anyhow!(
            "{}",
            crate::messages::CapabilityError::invalid_level(
                "Muse Glimmer",
                &MuseGlimmerReasoningStrength::level_names(),
                requested
            )
        )
    })
}

/// What Muse accepts as reasoning controls, from the library's own table.
pub(crate) fn muse_glimmer_reasoning_capability() -> prompt_template::ReasoningCapability {
    prompt_template::ReasoningCapability {
        levels: MuseGlimmerReasoningStrength::level_names(),
        fallback: Some(MuseGlimmerReasoningStrength::High.as_str()),
        no_thinking: prompt_template::Support::Unsupported {
            code: "no_thinking_unsupported",
            message: "Muse Glimmer declares no non-thinking ATEM profile; use reasoning effort low for the lightest supported mode".into(),
        },
        thinking: prompt_template::Support::Unsupported {
            code: "thinking_unsupported",
            message: "Muse Glimmer always reasons; there is no explicit thinking toggle".into(),
        },
    }
}

pub(crate) fn validate_muse_glimmer_generation_mode(
    args: &Args,
    explicit: ExplicitCliOptions,
    invocation: &cli::Invocation,
) -> Result<()> {
    admission::MUSE_GLIMMER_SINGLE_TURN.admit(&admission::supplied(args, explicit))?;
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
    config
        .validate_tokenizer(&tokenizer)
        .context("Muse Glimmer tokenizer contract")?;
    let encode_t0 = Instant::now();
    let prompt_ids = tokenizer
        .encode(&prepared.text, prepared.add_special_tokens)
        .context("tokenize Muse Glimmer prompt")?;
    let encode_ms = encode_t0.elapsed().as_secs_f64() * 1e3;
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
    config
        .validate_stop_tokens(&stop_tokens)
        .context("Muse Glimmer stop-token contract")?;
    let prefill_packed_tokens = prompt_tokens.len() / MUSE_GLIMMER_PACKED_PREFILL_QUANTUM
        * MUSE_GLIMMER_PACKED_PREFILL_QUANTUM;
    let prefill_scalar_tail_commands = prompt_tokens.len() - prefill_packed_tokens;
    let prefill_commands = prefill_packed_tokens / MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS
        + usize::from(
            !prefill_packed_tokens.is_multiple_of(MUSE_GLIMMER_PACKED_PREFILL_MAX_TOKENS),
        )
        + prefill_scalar_tail_commands;
    let read_math_opt_in = |name: &str| -> Result<bool> {
        match std::env::var(name) {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            _ => bail!("{name} must be 0 or 1"),
        }
    };
    let split_decode = read_math_opt_in("QWEN_MUSE_SPLIT_DECODE")?;
    let matrix_prefill = read_math_opt_in("QWEN_MUSE_MATRIX_PREFILL")?;
    let optimized_packed_tokens = if matrix_prefill {
        prefill_packed_tokens
    } else {
        0
    };
    let prefill_mode = match (
        prefill_packed_tokens,
        prefill_scalar_tail_commands,
        matrix_prefill,
    ) {
        (0, _, _) => "scalar_tail",
        (_, 0, false) => "packed_exact",
        (_, _, false) => "packed_exact+scalar_tail",
        (_, 0, true) => "packed_matrix_online",
        (_, _, true) => "packed_matrix_online+scalar_tail",
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
    let mut loaded = MuseGlimmerLoadedModel::load_with_options(
        &ctx,
        gguf,
        capacity,
        qwen_llm::muse_glimmer_runtime::MuseGlimmerRuntimeOptions {
            split_decode,
            matrix_prefill,
        },
    )
    .context("load admitted Muse Glimmer weights and text session")?;
    eprintln!(
        "muse_glimmer: split_decode={} split_min_visible_positions=1024 model_context={} matrix_prefill={} optimized_packed_tokens={} packed_attention={} tiled_packed_tokens={}",
        split_decode,
        config.context_length,
        matrix_prefill,
        optimized_packed_tokens,
        if matrix_prefill {
            "tiled_n128+online_remainder"
        } else {
            "exact"
        },
        if matrix_prefill {
            prompt_tokens.len() / 128 * 128
        } else {
            0
        }
    );
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
    if let Some(path) = args.request_stats_jsonl.as_ref() {
        let transition_tps = if generation.transition_ms > 0.0 {
            generation.transitions as f64 / (generation.transition_ms / 1e3)
        } else {
            0.0
        };
        let measured = RequestStatsMeasured {
            input_tokens: prompt_tokens.len() as u64,
            output_tokens: generation.tokens.len() as u64,
            transitions: generation.transitions as u64,
            stop_reason: generation.stop_reason,
            // Record semantics: encode-only tokenization; total without load.
            tokenizer_ms: encode_ms,
            load_ms,
            prefill_ms,
            prefill_tps,
            decode_ms: generation.wall_ms,
            decode_tps,
            transition_tps,
            total_ms: request_t0.elapsed().as_secs_f64() * 1e3 - load_ms,
            output_fingerprint: GeneratedTokenSha256Digest::of(&generation.tokens),
        };
        append_single_turn_stats_record(
            path,
            0,
            ModelFamily::MuseGlimmer.record_label(),
            request_stats_input(prepared.source, Some("atem")),
            &measured,
            None,
        )?;
    }
    Ok(())
}
