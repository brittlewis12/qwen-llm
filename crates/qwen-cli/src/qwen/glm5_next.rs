//! GLM-5.3-Flash: bounded raw serial generation in the dense attention range
//! (visible length below 2052). Chat templating, serve, lens and sparse
//! selection are not implemented; every other surface refuses rather than
//! falling through to Qwen protocols.

use super::*;
use qwen_llm::glm5_next::{self, ExecutionMode, Glm5NextModel};
use qwen_llm::glm5_next_metal::{Glm5NextSession, Glm5NextWeights};
use serde_json::{Value, json};

const FAMILY: &str = "GLM-5.3-Flash";

/// Header-only artifact admission: strict binding, serial-decode weight
/// coverage and the stop set. Not device or numerical qualification.
fn artifact_admission(gguf: &GgufFile) -> std::result::Result<u32, glm5_next::Glm5NextError> {
    let model = Glm5NextModel::from_gguf(gguf)?;
    model.validate_execution(ExecutionMode::SerialDecode)?;
    glm5_next::generation_stops(gguf, model.config.vocab_size)?;
    Ok(model.config.sparse_frontier())
}

pub(crate) fn capability_projection(gguf: &GgufFile) -> Result<Value> {
    let admission = artifact_admission(gguf);
    let run = match &admission {
        Ok(frontier) => json!({
            "status": "conditional", "implementation_status": "partial", "scope": "raw_only",
            "artifact_admission": {"status": "passed"},
            "capacity_policy": format!("dense_attention_range_below_{frontier}_and_device_memory"),
            "prefill": "packed_fast", "native_tokenizer": true, "latent_cache": "f16",
        }),
        Err(error) => json!({
            "status": "unsupported", "implementation_status": "partial",
            "artifact_admission": {"status": "rejected", "code": "glm5_next_artifact_rejected", "message": error.to_string()},
        }),
    };
    let bench = match &admission {
        Ok(_) => json!({"status": "conditional", "implementation_status": "partial",
            "command": "qwen-bench suite", "scope": "packed_prefill_and_serial_decode_rows"}),
        Err(_) => json!({"status": "unsupported", "implementation_status": "partial"}),
    };
    let unsupported = |lane: &str| {
        json!({"status": "unsupported", "implementation_status": "unsupported",
            "code": "glm5_next_lane_unimplemented",
            "message": format!("{FAMILY} has no {lane} implementation yet")})
    };
    let raw = match &admission {
        Ok(_) => json!({"status": "supported"}),
        Err(error) => {
            json!({"status": "unsupported", "code": "glm5_next_artifact_rejected", "message": error.to_string()})
        }
    };
    let templated = json!({"status": "unsupported", "code": "glm5_next_chat_unimplemented",
        "message": format!("{FAMILY} chat rendering is not implemented; use --raw-prompt")});
    Ok(json!({
        "execution": {"run": run, "bench": bench, "serve": unsupported("serve"), "lens": unsupported("lens"),
            "request_device": {"status": "not_evaluated",
                "requires": ["request_options_and_token_budget", "live_memory_admission"]}},
        "input": {"raw": raw, "user": templated, "messages": templated, "tools": templated},
        "reasoning": {"levels": [], "fallback": null,
            "no_thinking": {"status": "unsupported", "code": "glm5_next_chat_unimplemented", "message": "raw input has no reasoning controls"},
            "thinking": {"status": "unsupported", "code": "glm5_next_chat_unimplemented", "message": "raw input has no reasoning controls"}},
        "template": {"status": "unsupported", "rendered_as": null,
            "message": format!("{FAMILY} raw input has no template renderer yet")},
    }))
}

fn prepare_raw(
    invocation: cli::Invocation,
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<(String, PromptSource)> {
    admission::GLM5_NEXT_RAW_SINGLE_TURN.admit(&admission::supplied(args, explicit))?;
    ensure!(
        args.drafter.is_none(),
        "{FAMILY} does not support --drafter"
    );
    ensure!(
        args.messages.is_none(),
        "{FAMILY} currently supports raw input only"
    );
    match invocation {
        cli::Invocation::Run(run) => {
            ensure!(
                !run.no_thinking && run.reasoning_effort.is_none(),
                "{FAMILY} raw input does not accept reasoning controls"
            );
            match run.input {
                cli::RunInput::RawPrompt(text) => Ok((text, PromptSource::Inline)),
                _ => bail!(
                    "{FAMILY} chat rendering is not implemented; use --raw-prompt (include [gMASK]<sop> for the release prefix)"
                ),
            }
        }
        cli::Invocation::Legacy => {
            let (text, source, _) = prompt_text(args)?;
            Ok((text, source))
        }
        _ => bail!("{FAMILY} raw preparation is a single-turn generation path"),
    }
}

/// Forwards for the request, bounded by the dense attention range.
fn capacity(args: &Args, prompt_tokens: usize, frontier: u32) -> Result<usize> {
    let required = required_forwards(FAMILY, prompt_tokens, args.tokens, None)?;
    let capacity = args.max_context_tokens.unwrap_or(required);
    ensure!(
        capacity >= required,
        "{FAMILY} requires {required} forwards; requested capacity {capacity} is smaller"
    );
    ensure!(
        capacity < frontier as usize,
        "{FAMILY} currently supports visible lengths below {frontier} (sparse attention is not implemented); this request needs {capacity}"
    );
    Ok(capacity)
}

pub(crate) fn run_raw(
    gguf: &GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
    invocation: cli::Invocation,
) -> Result<()> {
    let request_t0 = Instant::now();
    let (text, source) = prepare_raw(invocation, args, explicit)?;
    let frontier = artifact_admission(gguf).with_context(|| format!("admit {FAMILY} artifact"))?;
    let vocab_size = glm5_next::RELEASE_VOCAB_SIZE;
    let tokenizer_t0 = Instant::now();
    let tokenizer = Tokenizer::from_gguf(gguf).context("build GLM-5.3 native tokenizer")?;
    // The glm4 tokenizer never inserts BOS; [gMASK]<sop> belongs in the text.
    let ids = tokenizer.encode(&text, !args.no_special_tokens)?;
    let tokenizer_ms = tokenizer_t0.elapsed().as_secs_f64() * 1e3;
    let tokens = ids
        .into_iter()
        .enumerate()
        .map(|(i, id)| checked_token_id(id, vocab_size, &format!("prompt[{i}]")))
        .collect::<Result<Vec<_>>>()?;
    let capacity = capacity(args, tokens.len(), frontier)?;
    let stops = glm5_next::generation_stops(gguf, vocab_size)?;
    let mut sampler = Sampler::new(cli_sampling_config(args)?)?;
    shutdown::checkpoint()?;

    let ctx = MetalContext::new().context("initialize Metal for GLM-5.3")?;
    // Warm cold retained windows with parallel reads before the zero-copy
    // weights are first touched; demand paging would otherwise land in the
    // first prefill at a fraction of the storage bandwidth.
    let prefetch = qwen_llm::glm5_next_metal::prefetch_retained(&ctx, gguf, 0.98)
        .context("prefetch GLM-5.3 retained windows")?;
    let prefetch_ms = prefetch.wall.as_secs_f64() * 1e3;
    eprintln!(
        "glm5_next prefetch: windows={} cold_windows={} bytes_read={} wall_ms={prefetch_ms:.1}",
        prefetch.windows, prefetch.cold_windows, prefetch.bytes_read
    );
    let load_t0 = Instant::now();
    let weights = Glm5NextWeights::load(&ctx, gguf).context("load GLM-5.3 weights")?;
    let prefill_rows = tokens
        .len()
        .min(qwen_llm::glm5_next_metal::DEFAULT_PREFILL_ROWS);
    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, capacity, prefill_rows)
        .context("create GLM-5.3 session")?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;

    let prefill_t0 = Instant::now();
    shutdown::checkpoint()?;
    let logits = session.prefill_packed(&ctx, &tokens)?;
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    let generation = generate_serial(
        logits,
        args.tokens,
        &stops,
        &mut sampler,
        |token| {
            stdout.write_all(tokenizer.try_decode_piece_bytes_exact(token)?)?;
            stdout.flush()?;
            Ok(())
        },
        |token| {
            shutdown::checkpoint()?;
            let token = checked_token_id(token, vocab_size, "generated")?;
            session.forward(&ctx, token).map_err(anyhow::Error::from)
        },
    )?;
    if !generation.tokens.is_empty() {
        writeln!(stdout)?;
        stdout.flush()?;
    }
    let total_ms = request_t0.elapsed().as_secs_f64() * 1e3;
    let prefill_tps = tokens.len() as f64 / (prefill_ms / 1e3).max(f64::MIN_POSITIVE);
    let decode_tps =
        generation.tokens.len() as f64 / (generation.wall_ms / 1e3).max(f64::MIN_POSITIVE);
    eprintln!(
        "glm5_next: prompt_tokens={} generated_tokens={} transitions={} stop={} capacity={capacity} prefill=packed_fast prefill_rows={prefill_rows} prefetch_ms={prefetch_ms:.1} load_ms={load_ms:.1} prefill_ms={prefill_ms:.1} prefill_tps={prefill_tps:.2} decode_tps={decode_tps:.2}",
        tokens.len(),
        generation.tokens.len(),
        generation.transitions,
        generation.stop_reason.as_str()
    );
    if let Some(path) = args.request_stats_jsonl.as_ref() {
        let measured = RequestStatsMeasured {
            input_tokens: tokens.len() as u64,
            output_tokens: generation.tokens.len() as u64,
            transitions: generation.transitions as u64,
            stop_reason: generation.stop_reason,
            tokenizer_ms,
            load_ms,
            prefill_ms,
            prefill_tps,
            decode_ms: generation.wall_ms,
            decode_tps,
            transition_tps: generation.transitions as f64
                / (generation.transition_ms / 1e3).max(f64::MIN_POSITIVE),
            total_ms,
            output_fingerprint: GeneratedTokenSha256Digest::of(&generation.tokens),
        };
        append_single_turn_stats_record(
            path,
            0,
            ModelFamily::Glm5Next.record_label(),
            request_stats_input(source, None),
            &measured,
            None,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> (Args, ExplicitCliOptions, cli::Invocation) {
        let (mut args, explicit) = Args::parse_with_explicit(argv);
        let invocation = cli::normalize(&mut args);
        invocation.apply_option_overrides(&mut args);
        (args, explicit, invocation)
    }

    #[test]
    fn raw_lane_refuses_templated_input() {
        let (args, explicit, invocation) =
            parse(&["qwen", "run", "-m", "m.gguf", "--user", "hello"]);
        let error = prepare_raw(invocation, &args, explicit)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("chat rendering is not implemented"),
            "{error}"
        );
        let (args, explicit, invocation) = parse(&[
            "qwen",
            "run",
            "-m",
            "m.gguf",
            "--raw-prompt",
            "[gMASK]<sop>hi",
        ]);
        assert_eq!(
            prepare_raw(invocation, &args, explicit).unwrap().0,
            "[gMASK]<sop>hi"
        );
    }

    #[test]
    fn capacity_stays_in_the_dense_range() {
        let (args, ..) = parse(&[
            "qwen",
            "run",
            "-m",
            "m.gguf",
            "--raw-prompt",
            "x",
            "-n",
            "16",
        ]);
        assert_eq!(capacity(&args, 10, 2052).unwrap(), 25);
        let error = capacity(&args, 2040, 2052).unwrap_err().to_string();
        assert!(error.contains("below 2052"), "{error}");
        let (args, ..) = parse(&[
            "qwen",
            "run",
            "-m",
            "m.gguf",
            "--raw-prompt",
            "x",
            "-n",
            "16",
            "--max-context-tokens",
            "20",
        ]);
        assert!(capacity(&args, 10, 2052).is_err());
    }
}
