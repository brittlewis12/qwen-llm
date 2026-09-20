//! Raw-text K2 lane; never falls through to Qwen prompt/output protocols.

use super::*;
use qwen_llm::k2_horizon::K2HorizonConfig;
use qwen_llm::k2_horizon_runtime::K2LoadedModel;
use qwen_llm::tokenizer::NativeTokenizer;

pub(crate) fn execution_capabilities() -> serde_json::Value {
    serde_json::json!({
        "run": {"status": "supported", "scope": "raw_single_turn", "requires_profile": "dense_7b",
            "capacity_policy": "checkpoint_context_and_device_memory", "native_tokenizer": true, "kv_storage": "f16"},
        "serve": {"status": "partial", "endpoint": "/v1/responses", "input": "raw_string_only",
            "capacity_policy": "checkpoint_context_and_device_memory", "snapshot_cache": false, "tools": false,
            "chat": false, "special_token_control": "x_k2.add_special_tokens"},
        "bench": {"status": "partial", "command": "qwen-bench k2-request",
            "scope": "raw_greedy_request_wall", "capacity_policy": "checkpoint_context_and_device_memory",
            "llama_bench_comparable": false},
        "lens": {"status": "partial", "command": "qwen-lens read-full --logit-lens",
            "transport_command": "qwen-lens read-full --full-lens",
            "scope": "raw_plain_or_data_only_linear_readout", "capacity_policy": "checkpoint_context_and_device_memory",
            "imported_assets": "llm.lens.linear_transport_v1_target_layer_35", "cli_interventions": false},
        "local_fitting": {"status": "unsupported"},
    })
}

fn prepare_raw(
    invocation: cli::Invocation,
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<(String, PromptSource)> {
    admission::K2_RAW_SINGLE_TURN.admit(&admission::supplied(args, explicit))?;
    ensure!(
        args.drafter.is_none(),
        "K2 Horizon does not support --drafter"
    );
    ensure!(
        args.messages.is_none(),
        "K2 Horizon currently supports raw input only; messages are not implemented"
    );
    match invocation {
        cli::Invocation::Run(run) => {
            ensure!(
                !run.no_thinking && run.reasoning_effort.is_none(),
                "K2 Horizon raw input does not accept reasoning controls"
            );
            // Refuse templated stdin/files before attempting to read them.
            match run.input {
                cli::RunInput::RawPrompt(text) => Ok((text, PromptSource::Inline)),
                _ => bail!(
                    "K2 Horizon currently requires --raw-prompt; user/messages rendering is not implemented"
                ),
            }
        }
        cli::Invocation::Legacy => {
            let (text, source, _) = prompt_text(args)?;
            Ok((text, source))
        }
        _ => bail!("K2 raw preparation is a single-turn generation path"),
    }
}

fn capacity(
    args: &Args,
    _explicit: ExplicitCliOptions,
    prompt_tokens: usize,
    declared: u32,
) -> Result<usize> {
    let required = required_forwards(
        "K2 Horizon",
        prompt_tokens,
        args.tokens,
        Some(declared as usize),
    )?;
    let capacity = args.max_context_tokens.unwrap_or(required);
    ensure!(
        capacity >= required && capacity <= declared as usize,
        "K2 Horizon requires {required} forwards; requested capacity {capacity} must fit checkpoint context {declared}"
    );
    Ok(capacity)
}

fn validate_stops(stops: &[i32]) -> Result<()> {
    ensure!(
        stops == [1],
        "K2 Horizon raw generation requires EOS 1 only; extra stop metadata is unsupported"
    );
    Ok(())
}

pub(crate) fn run_raw(
    gguf: &GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
    invocation: cli::Invocation,
) -> Result<()> {
    let request_t0 = Instant::now();
    let (text, source) = prepare_raw(invocation, args, explicit)?;
    let sampling = cli_sampling_config(args)?;
    let config = K2HorizonConfig::from_gguf(gguf).context("bind K2 dense 7B profile")?;
    let tokenizer = NativeTokenizer::from_gguf(gguf).context("bind native K2 tokenizer")?;
    let encode_t0 = Instant::now();
    // Native single-sequence policy inserts BOS once per encoding call. An
    // already serialized BOS requires explicit --no-special-tokens, not guessing.
    let ids = tokenizer.encode(&text, !args.no_special_tokens)?;
    let encode_ms = encode_t0.elapsed().as_secs_f64() * 1e3;
    let tokens = ids
        .into_iter()
        .enumerate()
        .map(|(i, id)| checked_token_id(id, config.vocab_size, &format!("prompt[{i}]")))
        .collect::<Result<Vec<_>>>()?;
    let capacity = capacity(args, explicit, tokens.len(), config.context_length)?;
    let stops = gguf.stop_token_ids()?;
    validate_stops(&stops)?;
    for &stop in &stops {
        checked_token_id(stop, config.vocab_size, "stop")?;
    }
    let mut sampler = Sampler::new(sampling)?;
    shutdown::checkpoint()?;
    let load_t0 = Instant::now();
    let ctx = MetalContext::new().context("initialize Metal for K2")?;
    let model = K2LoadedModel::load(&ctx, gguf, u32::try_from(capacity)?)?;
    let prefill = model.prefill_info(tokens.len());
    eprintln!(
        "k2_horizon: prefill={} chunk_tokens={} commands={} temporary_activation_bytes={}",
        prefill.mode, prefill.chunk_tokens, prefill.commands, prefill.temporary_activation_bytes
    );
    let mut session = model.create_session(0)?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
    let prefill_t0 = Instant::now();
    let mut logits = Vec::new();
    for chunk in tokens.chunks(prefill.chunk_tokens) {
        shutdown::checkpoint()?;
        logits = session.append(chunk)?;
    }
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
            let token = checked_token_id(token, config.vocab_size, "generated")?;
            session.append(&[token]).map_err(anyhow::Error::from)
        },
    )?;
    if !generation.tokens.is_empty() {
        writeln!(stdout)?;
        stdout.flush()?;
    }
    let prefill_tps = tokens.len() as f64 / (prefill_ms / 1e3).max(f64::MIN_POSITIVE);
    let decode_tps =
        generation.tokens.len() as f64 / (generation.wall_ms / 1e3).max(f64::MIN_POSITIVE);
    eprintln!(
        "k2_horizon: prompt_tokens={} generated_tokens={} transitions={} stop={} load_ms={load_ms:.1} prefill_ms={prefill_ms:.1}",
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
            tokenizer_ms: encode_ms,
            load_ms,
            prefill_ms,
            prefill_tps,
            decode_ms: generation.wall_ms,
            decode_tps,
            transition_tps: generation.transitions as f64
                / (generation.transition_ms / 1e3).max(f64::MIN_POSITIVE),
            total_ms: request_t0.elapsed().as_secs_f64() * 1e3 - load_ms,
            output_fingerprint: GeneratedTokenSha256Digest::of(&generation.tokens),
        };
        append_single_turn_stats_record(
            path,
            0,
            ModelFamily::K2Horizon.record_label(),
            request_stats_input(source, None),
            &measured,
            Some(RequestStatsDiagnostics {
                deepseek_v4: None,
                k2_horizon: Some(RequestStatsK2Diagnostics { prefill }),
            }),
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
    fn k2_stats_record_selected_prefill_without_other_family_diagnostics() {
        let diagnostics = RequestStatsDiagnostics {
            deepseek_v4: None,
            k2_horizon: Some(RequestStatsK2Diagnostics {
                prefill: qwen_llm::k2_horizon_runtime::K2PrefillInfo {
                    mode: "q8_lcpp_token_batch",
                    chunk_tokens: 32,
                    commands: 2,
                    temporary_activation_bytes: 7602304,
                },
            }),
        };
        let record = serde_json::to_value(diagnostics).unwrap();
        assert!(record.get("deepseek_v4").is_none());
        assert_eq!(record["k2_horizon"]["prefill"]["commands"], 2);
        assert_eq!(record["k2_horizon"]["prefill"]["chunk_tokens"], 32);
    }

    #[test]
    fn k2_raw_rejects_templates_and_unsupported_options_before_input_io() {
        for option in ["--user", "--messages"] {
            let (args, explicit, invocation) =
                parse(&["qwen", "run", "-m", "unused", option, "-", "-n", "8"]);
            assert!(prepare_raw(invocation, &args, explicit).is_err());
        }
        for option in ["--drafter", "--durable-prefix-cache", "--requests-jsonl"] {
            let mut argv = vec!["qwen", "-m", "unused", "-n", "8", option, "unused"];
            if option != "--requests-jsonl" {
                argv.extend(["--prompt", "raw"]);
            }
            let (args, explicit, invocation) = parse(&argv);
            assert!(prepare_raw(invocation, &args, explicit).is_err());
        }
    }

    #[test]
    fn k2_budget_uses_declared_context_and_never_truncates() {
        let (args, explicit, _) = parse(&["qwen", "run", "-m", "unused", "--raw-prompt", "raw"]);
        assert!(capacity(&args, explicit, 1, 8192).is_ok());
        let (args, explicit, invocation) = parse(&[
            "qwen",
            "run",
            "-m",
            "unused",
            "--raw-prompt",
            "raw",
            "-n",
            "8",
        ]);
        assert_eq!(prepare_raw(invocation, &args, explicit).unwrap().0, "raw");
        assert_eq!(capacity(&args, explicit, 25, 8192).unwrap(), 32);
        assert_eq!(capacity(&args, explicit, 249, 8192).unwrap(), 256);
        assert_eq!(capacity(&args, explicit, 250, 8192).unwrap(), 257);
        assert_eq!(capacity(&args, explicit, 8185, 8192).unwrap(), 8192);
        assert!(capacity(&args, explicit, 8186, 8192).is_err());
        assert!(capacity(&args, explicit, usize::MAX, 8192).is_err());
        assert!(capacity(&args, explicit, 0, 8192).is_err());
        assert!(capacity(&args, explicit, 25, 31).is_err());
    }

    #[test]
    fn k2_modern_raw_can_explicitly_disable_special_insertion() {
        let (args, explicit, invocation) = parse(&[
            "qwen",
            "run",
            "-m",
            "unused",
            "--raw-prompt",
            "<|ifm|begin_of_text|>raw",
            "--no-special-tokens",
            "-n",
            "8",
        ]);
        assert!(args.no_special_tokens);
        assert_eq!(
            prepare_raw(invocation, &args, explicit).unwrap().0,
            "<|ifm|begin_of_text|>raw"
        );
    }

    #[test]
    fn k2_capabilities_do_not_advertise_other_lanes_or_fitting() {
        let capabilities = execution_capabilities();
        for lane in ["run", "serve", "bench", "lens"] {
            assert_eq!(
                capabilities[lane]["capacity_policy"],
                "checkpoint_context_and_device_memory"
            );
            assert!(capabilities[lane].get("max_forward_tokens").is_none());
        }
        assert_eq!(capabilities["local_fitting"]["status"], "unsupported");
        assert_eq!(capabilities["serve"]["status"], "partial");
        assert_eq!(capabilities["serve"]["input"], "raw_string_only");
        assert_eq!(capabilities["serve"]["snapshot_cache"], false);
        assert_eq!(capabilities["serve"]["tools"], false);
        assert_eq!(capabilities["bench"]["status"], "partial");
        assert_eq!(capabilities["bench"]["command"], "qwen-bench k2-request");
        assert_eq!(capabilities["bench"]["llama_bench_comparable"], false);
        assert_eq!(capabilities["lens"]["status"], "partial");
        assert_eq!(
            capabilities["lens"]["command"],
            "qwen-lens read-full --logit-lens"
        );
        assert_eq!(
            capabilities["lens"]["imported_assets"],
            "llm.lens.linear_transport_v1_target_layer_35"
        );
        assert_eq!(capabilities["lens"]["cli_interventions"], false);
        assert_eq!(ModelFamily::K2Horizon.record_label(), "k2_horizon");
    }

    #[test]
    fn k2_raw_stop_policy_cannot_inherit_extra_eot_tokens() {
        assert!(validate_stops(&[1]).is_ok());
        for stops in [vec![], vec![0], vec![1, 2], vec![-1], vec![2, 1]] {
            assert!(validate_stops(&stops).is_err());
        }
    }
}
