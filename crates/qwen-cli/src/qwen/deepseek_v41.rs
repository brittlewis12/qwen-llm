//! GGUF-only V4.1 request lane; geometry is independent of local capacity.
use super::*;
use qwen_llm::deepseek_v41::{forward::Forward, gguf as weights};
use qwen_llm::tokenizer::NativeTokenizer;
use std::sync::Arc;

pub(crate) fn default_sampling(seed: u64) -> SamplingConfig {
    SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed,
    }
}

pub(crate) fn load(ctx: &MetalContext, gguf: Arc<GgufFile>) -> Result<weights::ResidentWeights> {
    shutdown::checkpoint()?;
    Ok(weights::ResidentWeights::load(ctx, gguf)?)
}

pub(crate) fn capabilities(gguf: &GgufFile) -> Result<serde_json::Value> {
    // Header validation deliberately has no Metal context or host-fit gate.
    let plan = match weights::plan(gguf) {
        Ok(plan) => plan,
        Err(error) => {
            return Ok(serde_json::json!({
                "input": prompt_template::InputCapability::none("invalid_v41_gguf", error.to_string()),
                "execution":{"status":"rejected","code":"invalid_v41_gguf","message":error.to_string()},
                "template":{"status":"unknown"},
                "local_fit":{"status":"not_evaluated","stage":"runtime_memory_admission"},
            }));
        }
    };
    Ok(serde_json::json!({
        "input": prompt_template::InputCapability::all_supported(),
        "reasoning": {"levels":["none","low","high","xhigh","max"],"fallback":"none",
            "no_thinking":{"status":"unsupported"},"thinking":{"status":"unsupported"}},
        "template": {"status":"identified","rendered_as":"deepseek_recipe_dsv41"},
        "execution":{"status":"supported","capacity_policy":"runtime_memory_admission",
            "prefill_mode":crate::family_options::V41_PREFILL_MODE,
            "prefill_rows_max":crate::family_options::V41_PREFILL_ROWS},
        "geometry":{"experts":plan.config.experts,"context_tokens":plan.engram_hash.max_position},
        "local_fit":{"status":"not_evaluated","stage":"runtime_memory_admission"},
        "bench":{"status":"supported","harness":"family_pp_tg"},
        "cache":{"policy":"exact_prefix_ram","prefix_reuse":true,"snapshots":true},
        "lens":{"status":"unsupported"},"speculation":{"status":"unsupported"}
    }))
}

fn only_message_fields(value: &serde_json::Value, scope: &str, fields: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .with_context(|| format!("V4.1 {scope} must be an object"))?;
    for key in object.keys() {
        ensure!(
            fields.contains(&key.as_str()),
            "V4.1 {scope} field {key:?} is unsupported; --messages accepts conversation/render fields only, not generation, transport, or output controls"
        );
    }
    Ok(())
}

fn messages_body(document: &str) -> Result<serde_json::Value> {
    let mut body: serde_json::Value = serde_json::from_str(document)?;
    if body.is_array() {
        body = serde_json::json!({"input":body});
    }
    // StrictChat cannot represent Responses history. Keep the family parser and
    // renderer, but do not admit serving controls that this text-only run drops.
    only_message_fields(
        &body,
        "messages",
        &["input", "messages", "instructions", "tools", "reasoning"],
    )?;
    ensure!(
        body.get("input").is_none() || body.get("messages").is_none(),
        "V4.1 messages cannot contain both input and messages"
    );
    if let Some(reasoning) = body.get("reasoning").filter(|value| !value.is_null()) {
        only_message_fields(reasoning, "reasoning", &["effort"])?;
    }
    if let Some(tools) = body.get("tools").and_then(serde_json::Value::as_array) {
        for (index, tool) in tools.iter().enumerate() {
            only_message_fields(
                tool,
                &format!("tools[{index}]"),
                &["type", "name", "description", "parameters", "strict"],
            )?;
        }
    }
    if let Some(messages) = body.as_object_mut().and_then(|map| map.remove("messages")) {
        body["input"] = messages;
    }
    body["model"] = "v41".into();
    Ok(body)
}

fn prepare_request_front(
    invocation: cli::Invocation,
    args: &Args,
) -> Result<(String, PromptSource)> {
    let cli::Invocation::Run(run) = invocation else {
        ensure!(
            args.messages.is_none(),
            "V4.1 messages require qwen run --messages"
        );
        let (text, source, _) = prompt_text(args)?;
        return Ok((text, source));
    };
    ensure!(
        run.format == chat_output::RunFormat::Text,
        "V4.1 run supports --format text only"
    );
    ensure!(
        !run.no_thinking,
        "V4.1 selects thinking with --reasoning-effort none/low/high/xhigh/max"
    );
    let effort = run.reasoning_effort.clone();
    prepare_request_input(run.acquire_input()?, effort, args)
}

fn prepare_request_input(
    input: cli::AcquiredRunInput,
    effort: Option<String>,
    args: &Args,
) -> Result<(String, PromptSource)> {
    let mut body = match input {
        cli::AcquiredRunInput::RawPrompt(text) => {
            ensure!(
                effort.is_none(),
                "raw input does not accept reasoning controls"
            );
            return Ok((text, PromptSource::Inline));
        }
        cli::AcquiredRunInput::User { system, user } => {
            let mut body =
                serde_json::json!({"model":"v41","input":[{"role":"user","content":user}]});
            if let Some(system) = system {
                body["instructions"] = system.into();
            }
            body
        }
        cli::AcquiredRunInput::Messages { document, source } => messages_body(&document)
            .with_context(|| format!("parse V4.1 messages input {source}"))?,
    };
    if let Some(effort) = effort {
        ensure!(
            body.get("reasoning").is_none(),
            "V4.1 --reasoning-effort conflicts with messages document reasoning; supply only one"
        );
        body["reasoning"] = serde_json::json!({"effort":effort});
    }
    let profile = serve::request_profile::RequestProfile::DeepSeekV41 {
        limits: serve::request_profile::OutputLimits {
            default_max_tokens: args.tokens,
            max_piece_bytes: 1,
        },
        sampling: None,
    };
    let mut request = profile.parse(&body).map_err(|e| anyhow!(e.message))?;
    profile
        .normalize(&mut request)
        .map_err(|e| anyhow!(e.message))?;
    let prompt = profile.render(&request).map_err(|e| anyhow!(e.message))?;
    Ok((prompt, PromptSource::Messages))
}

fn request_forwards(plan: &weights::Plan, args: &Args, prompt_tokens: usize) -> Result<usize> {
    let context = usize::try_from(plan.engram_hash.max_position)?;
    let capacity = args.max_context_tokens.unwrap_or(context);
    ensure!(
        capacity > 0 && capacity <= context,
        "V4.1 request context exceeds model context"
    );
    serve::decode_loop::required_forwards("V4.1", prompt_tokens, args.tokens, capacity)
        .map_err(|e| anyhow!(e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v41_run_and_info_use_validated_context_alias_without_fit_gate() {
        for experts in [272, 320, 384] {
            for key in [
                "deepseek41.context_length",
                "deepseek41.max_position_embeddings",
            ] {
                let gguf = crate::deepseek_v41_fixture::gguf(experts, key);
                let plan = weights::plan(&gguf).unwrap();
                let (mut args, _) = invocation(&[
                    "qwen",
                    "run",
                    "-m",
                    "model.gguf",
                    "--raw-prompt",
                    "hi",
                    "--max-tokens",
                    "2",
                ]);
                assert_eq!(request_forwards(&plan, &args, 3).unwrap(), 4);
                assert!(request_forwards(&plan, &args, 64).is_err());
                args.max_context_tokens = Some(65);
                assert!(request_forwards(&plan, &args, 3).is_err());
                let info = crate::project_info(std::path::Path::new("model.gguf"), &gguf).unwrap();
                assert_eq!(info["family"], "deepseek_v41");
                assert_eq!(info["capabilities"]["geometry"]["experts"], experts);
                assert_eq!(info["capabilities"]["geometry"]["context_tokens"], 64);
                assert_eq!(info["capabilities"]["execution"]["status"], "supported");
                assert_eq!(info["capabilities"]["local_fit"]["status"], "not_evaluated");
                assert_eq!(info["capabilities"]["bench"]["harness"], "family_pp_tg");
            }
        }
    }

    #[test]
    fn v41_info_reports_bad_metadata_separately_from_machine_fit() {
        let mut gguf = crate::deepseek_v41_fixture::gguf(384, "deepseek41.max_position_embeddings");
        gguf.tensors
            .retain(|t| t.name != "blk.0.ffn_up_exps.weight");
        let info = crate::project_info(std::path::Path::new("bad.gguf"), &gguf).unwrap();
        assert_eq!(info["family"], "deepseek_v41");
        assert_eq!(
            info["capabilities"]["execution"]["code"],
            "invalid_v41_gguf"
        );
        assert_eq!(info["capabilities"]["local_fit"]["status"], "not_evaluated");
    }

    fn invocation(argv: &[&str]) -> (Args, cli::Invocation) {
        let (mut args, _) = Args::parse_with_explicit(argv);
        let invocation = cli::normalize(&mut args);
        invocation.apply_option_overrides(&mut args);
        (args, invocation)
    }

    #[test]
    fn v41_uses_normal_run_and_serve_commands_only() {
        assert!(
            Args::command()
                .try_get_matches_from(["qwen", "deepseek-v41", "--model-dir", "unused"])
                .is_err()
        );
        let (_, serve) = invocation(&[
            "qwen",
            "serve",
            "-m",
            "model.gguf",
            "--max-context-tokens",
            "64",
            "--max-tokens",
            "8",
        ]);
        assert!(matches!(serve, cli::Invocation::Serve(_)));
        let (args, run) = invocation(&["qwen", "run", "-m", "model.gguf", "--raw-prompt", "hello"]);
        assert_eq!(
            prepare_request_front(run, &args).unwrap(),
            ("hello".into(), PromptSource::Inline)
        );
    }

    #[test]
    fn v41_run_uses_recipe_without_artifact_identity() {
        let (args, run) = invocation(&[
            "qwen",
            "run",
            "-m",
            "model.gguf",
            "--user",
            "hello",
            "--reasoning-effort",
            "high",
        ]);
        let (prompt, source) = prepare_request_front(run, &args).unwrap();
        assert_eq!(source, PromptSource::Messages);
        assert!(prompt.contains("hello"));
        assert!(prompt.contains("Reasoning Effort: 75"));
        assert!(prompt.ends_with("<think>"));
    }

    fn prepare_document(document: serde_json::Value, flags: &[&str]) -> Result<String> {
        let mut argv = vec![
            "qwen",
            "run",
            "-m",
            "model.gguf",
            "--messages",
            "unused.json",
        ];
        argv.extend_from_slice(flags);
        let (args, cli::Invocation::Run(run)) = invocation(&argv) else {
            unreachable!()
        };
        prepare_request_input(
            cli::AcquiredRunInput::Messages {
                document: document.to_string(),
                source: "test".into(),
            },
            run.reasoning_effort,
            &args,
        )
        .map(|(prompt, source)| {
            assert_eq!(source, PromptSource::Messages);
            prompt
        })
    }

    #[test]
    fn v41_messages_accept_arrays_and_both_document_aliases() {
        use serde_json::json;
        let history = json!([{"role":"user", "content":"hello"}]);
        let expected = prepare_document(history.clone(), &[]).unwrap();
        for document in [
            json!({"messages":history}),
            json!({"input":history}),
            json!({"input":"hello"}),
        ] {
            assert_eq!(prepare_document(document, &[]).unwrap(), expected);
        }
        assert!(expected.contains("hello"));
        assert!(expected.ends_with("</think>"));
        let prompt = prepare_document(history, &["--reasoning-effort", "high"]).unwrap();
        assert!(prompt.contains("Reasoning Effort: 75"));
        assert!(prompt.ends_with("<think>"));
    }

    #[test]
    fn v41_messages_preserve_responses_history_tools_and_reasoning() {
        use serde_json::json;
        for alias in ["input", "messages"] {
            for effort in ["none", "low", "high", "xhigh", "max"] {
                let document = json!({
                    alias: [
                        {"role":"user", "content":[{"type":"input_text", "text":"question"}]},
                        {"type":"reasoning", "content":"plan"},
                        {"role":"assistant", "content":"answer"},
                        {"type":"function_call", "call_id":"c", "name":"lookup", "arguments":"{}"},
                        {"type":"function_call_output", "call_id":"c", "output":{"result":"found"}},
                        {"role":"user", "content":"continue"}
                    ],
                    "instructions":"be helpful",
                    "tools":[{"type":"function", "name":"lookup", "description":"lookup facts",
                        "parameters":{"type":"object", "properties":{"temperature":{"type":"number"}}}}],
                    "reasoning":{"effort":effort}
                });
                let prompt = prepare_document(document, &[]).unwrap();
                for text in [
                    "be helpful",
                    "question",
                    "answer",
                    "lookup",
                    "found",
                    "continue",
                    "temperature",
                ] {
                    assert!(prompt.contains(text), "missing {text:?} in {prompt}");
                }
                assert_eq!(prompt.ends_with("<think>"), effort != "none");
                assert_eq!(prompt.contains("Reasoning Effort:"), effort != "none");
            }
        }
    }

    #[test]
    fn v41_messages_refuse_lost_controls_and_unknown_fields_even_when_null() {
        use serde_json::json;
        for (key, value) in [
            ("temperature", json!(0.2)),
            ("temp", json!(0.2)),
            ("top_p", json!(0.9)),
            ("top_k", json!(4)),
            ("min_p", json!(0.1)),
            ("seed", json!(42)),
            ("max_output_tokens", json!(12)),
            ("max_tokens", json!(12)),
            ("tokens", json!(12)),
            ("stream", json!(true)),
            ("store", json!(false)),
            ("truncation", json!("disabled")),
            ("parallel_tool_calls", json!(false)),
            ("response_format", json!({"type":"json_object"})),
            ("text", json!({"format":{"type":"text"}})),
            ("tool_choice", json!("auto")),
            (
                "tool_choice",
                json!({"type":"allowed_tools", "mode":"auto", "tools":[]}),
            ),
            ("x_qwen", json!({"seed":42})),
            ("x_qwen", json!({"stats":true})),
            ("x_qwen", json!({"template_style":"upstream"})),
            ("x_qwen", json!({"no_thinking":false})),
            ("reasoning_effort", json!("high")),
            ("no_thinking", json!(false)),
            ("thinking", json!(true)),
            ("model", json!("ignored-model")),
            ("meta", json!({"temperature":0.2})),
            ("unknown", json!(true)),
        ] {
            for value in [value, serde_json::Value::Null] {
                let document = json!({"input":"hello", key:value});
                let error = format!("{:#}", prepare_document(document, &[]).unwrap_err());
                assert!(
                    error.contains(key) && error.contains("unsupported"),
                    "{error}"
                );
            }
        }
    }

    #[test]
    fn v41_messages_refuse_nested_controls_without_restricting_schema_data() {
        use serde_json::json;
        for key in ["summary", "budget_tokens", "enabled", "unknown"] {
            for value in [json!(true), serde_json::Value::Null] {
                let document = json!({"input":"hello", "reasoning":{"effort":"high", key:value}});
                let error = format!("{:#}", prepare_document(document, &[]).unwrap_err());
                assert!(
                    error.contains("reasoning") && error.contains(key),
                    "{error}"
                );
            }
        }
        let document = json!({"input":"hello", "tools":[{"name":"lookup", "allowed":false}]});
        let error = format!("{:#}", prepare_document(document, &[]).unwrap_err());
        assert!(
            error.contains("tools[0]") && error.contains("allowed"),
            "{error}"
        );
    }

    #[test]
    fn v41_messages_aliases_and_reasoning_sources_are_mutually_exclusive() {
        use serde_json::json;
        for input in [json!("hello"), serde_json::Value::Null] {
            let error = format!(
                "{:#}",
                prepare_document(json!({"input":input, "messages":[]}), &[]).unwrap_err()
            );
            assert!(error.contains("both input and messages"), "{error}");
        }
        for reasoning in [
            json!({"effort":"low"}),
            json!({"effort":"high"}),
            json!({}),
            serde_json::Value::Null,
        ] {
            let error = prepare_document(
                json!({"input":"hello", "reasoning":reasoning}),
                &["--reasoning-effort", "high"],
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("--reasoning-effort conflicts"), "{error}");
        }
        for effort in [json!("medium"), json!(false)] {
            assert!(
                prepare_document(json!({"input":"hello", "reasoning":{"effort":effort}}), &[])
                    .is_err()
            );
        }
        for flags in [
            vec!["--messages", "unused.json", "--user", "hello"],
            vec!["--messages", "unused.json", "--raw-prompt", "hello"],
            vec!["--messages", "unused.json", "--system", "hello"],
            vec![
                "--messages",
                "unused.json",
                "--reasoning-effort",
                "high",
                "--no-thinking",
            ],
        ] {
            let mut argv = vec!["qwen", "run", "-m", "model.gguf"];
            argv.extend(flags);
            assert!(Args::command().try_get_matches_from(argv).is_err());
        }
    }

    #[test]
    fn v41_run_refuses_non_text_output_and_no_thinking_before_reading_messages() {
        for flags in [vec!["--format", "responses"], vec!["--no-thinking"]] {
            let mut argv = vec![
                "qwen",
                "run",
                "-m",
                "model.gguf",
                "--messages",
                "unused.json",
            ];
            argv.extend(flags);
            let (args, run) = invocation(&argv);
            let error = prepare_request_front(run, &args).unwrap_err().to_string();
            assert!(
                error.contains("--format text") || error.contains("--reasoning-effort"),
                "{error}"
            );
        }
    }

    #[test]
    fn v41_profile_is_ram_prefix_reuse_without_speculation() {
        let profile = family_profile::profile(ModelFamily::DeepSeekV41);
        assert!(profile.serve_backend);
        assert_eq!(
            profile.serve_warmth,
            family_profile::ServeWarmth::SnapshotsRam
        );
        assert_eq!(profile.fixed_cohort, family_profile::FixedCohort::None);
        assert!(matches!(
            profile.drafter,
            family_profile::DrafterSupport::Unsupported(_)
        ));
        assert!(!profile.idle_residency_eligible);
    }
}

pub(crate) fn run(
    gguf: GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
    invocation: cli::Invocation,
) -> Result<()> {
    admission::LaneAdmission {
        envelope: "DeepSeek V4.1 supports serial single-turn generation without prefix caching",
        supported: &[
            admission::LegacyOption::MaxContextTokens,
            admission::LegacyOption::RequestStatsJsonl,
        ],
    }
    .admit(&admission::supplied(args, explicit))?;
    let plan = weights::plan(&gguf)?;
    let tokenizer = NativeTokenizer::from_gguf(&gguf)?;
    ensure!(
        plan.config.vocab == tokenizer.n_vocab() as usize,
        "V4.1 tokenizer/model vocabulary mismatch"
    );
    let preparation = Instant::now();
    let (text, source) = prepare_request_front(invocation, args)?;
    let preparation_ms = preparation.elapsed().as_secs_f64() * 1e3;
    let encoding = Instant::now();
    let ids = tokenizer.encode(
        &text,
        source != PromptSource::Messages && !args.no_special_tokens,
    )?;
    let tokenizer_ms = encoding.elapsed().as_secs_f64() * 1e3;
    ensure!(!ids.is_empty(), "V4.1 prompt is empty");
    let required = request_forwards(&plan, args, ids.len())?;
    let eos = tokenizer.eos().context("V4.1 GGUF has no EOS")?;
    let mut sampler = Sampler::new(cli_sampling_config(args)?)?;
    shutdown::checkpoint()?;
    let loading = Instant::now();
    let ctx = MetalContext::new()?;
    let weights = load(&ctx, Arc::new(gguf))?;
    ensure!(
        weights.config.vocab == tokenizer.n_vocab() as usize,
        "V4.1 tokenizer/model vocabulary mismatch"
    );
    let forward =
        Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
            .map_err(anyhow::Error::msg)?;
    let load_ms = loading.elapsed().as_secs_f64() * 1e3;
    let mut state = forward.new_state_with_gpu_step(required)?;
    let execution = Instant::now();
    let prefill = Instant::now();
    let logits = forward
        .prefill(
            &mut state,
            &ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
            Some(shutdown::cancellation_flag()),
        )
        .context("V4.1 prefill")?;
    let prefill_ms = prefill.elapsed().as_secs_f64() * 1e3;
    tracing::info!(target: "qwen_diag", family="deepseek_v41", prefill_ms,
        prefill_mode=crate::family_options::V41_PREFILL_MODE,
        prefill_rows=ids.len().min(crate::family_options::V41_PREFILL_ROWS),
        prefill_chunks=ids.len().div_ceil(crate::family_options::V41_PREFILL_ROWS),
        "run prefill");
    let mut stdout = std::io::stdout().lock();
    let generation = decode::generate_serial(
        logits,
        args.tokens,
        &[eos],
        &mut sampler,
        |token| {
            stdout.write_all(tokenizer.try_decode_piece_bytes_exact(token)?)?;
            stdout.flush()?;
            Ok(())
        },
        |token| {
            shutdown::checkpoint()?;
            forward
                .step(
                    &mut state,
                    token as u32,
                    Some(shutdown::cancellation_flag()),
                )
                .map_err(anyhow::Error::msg)
        },
    )?;
    let total_ms = preparation_ms + tokenizer_ms + execution.elapsed().as_secs_f64() * 1e3;
    writeln!(stdout)?;
    if let Some(path) = &args.request_stats_jsonl {
        let measured = RequestStatsMeasured {
            input_tokens: ids.len() as u64,
            output_tokens: generation.tokens.len() as u64,
            transitions: generation.transitions as u64,
            stop_reason: generation.stop_reason,
            tokenizer_ms,
            load_ms,
            prefill_ms,
            prefill_tps: ids.len() as f64 / (prefill_ms / 1e3).max(f64::MIN_POSITIVE),
            decode_ms: generation.wall_ms,
            decode_tps: generation.tokens.len() as f64
                / (generation.wall_ms / 1e3).max(f64::MIN_POSITIVE),
            transition_tps: generation.transitions as f64
                / (generation.transition_ms / 1e3).max(f64::MIN_POSITIVE),
            total_ms,
            output_fingerprint: GeneratedTokenSha256Digest::of(&generation.tokens),
        };
        append_single_turn_stats_record(
            path,
            0,
            "deepseek_v41",
            request_stats_input(
                source,
                (source == PromptSource::Messages).then_some("deepseek_recipe_dsv41"),
            ),
            &measured,
            None,
        )?;
    }
    Ok(())
}
