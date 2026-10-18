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
    Ok(weights::ResidentWeights::load(
        ctx,
        gguf,
        weights::LoadLimits {
            // No CLI-specific budget; the loader performs typed runtime admission.
            resident_bytes: u64::MAX,
        },
    )?)
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
        "execution":{"status":"supported","capacity_policy":"runtime_memory_admission"},
        "geometry":{"experts":plan.config.experts,"context_tokens":plan.engram_hash.max_position},
        "local_fit":{"status":"not_evaluated","stage":"runtime_memory_admission"},
        "bench":{"status":"supported","harness":"family_pp_tg"},
        "cache":{"policy":"fresh_request","prefix_reuse":false,"snapshots":false},
        "lens":{"status":"unsupported"},"speculation":{"status":"unsupported"}
    }))
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
    let mut body = match run.acquire_input()? {
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
        cli::AcquiredRunInput::Messages { document, .. } => {
            let mut value: serde_json::Value = serde_json::from_str(&document)?;
            if value.is_array() {
                value = serde_json::json!({"input":value});
            }
            ensure!(
                value.is_object(),
                "V4.1 messages must be an array or an object"
            );
            ensure!(
                value.get("input").is_none() || value.get("messages").is_none(),
                "V4.1 messages cannot contain both input and messages"
            );
            if let Some(messages) = value.as_object_mut().and_then(|m| m.remove("messages")) {
                value["input"] = messages;
            }
            value["model"] = "v41".into();
            value
        }
    };
    if let Some(effort) = effort {
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

    #[test]
    fn v41_profile_is_fresh_request_without_speculation() {
        let profile = family_profile::profile(ModelFamily::DeepSeekV41);
        assert!(profile.serve_backend);
        assert_eq!(
            profile.serve_warmth,
            family_profile::ServeWarmth::FreshRequest
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
    let mut logits = Vec::new();
    for &id in &ids {
        shutdown::checkpoint()?;
        logits = forward
            .step(&mut state, id as u32, Some(shutdown::cancellation_flag()))
            .map_err(anyhow::Error::msg)?;
    }
    let prefill_ms = prefill.elapsed().as_secs_f64() * 1e3;
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
