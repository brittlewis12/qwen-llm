//! Minimal safetensors loading lane, separate from GGUF family registration.
use anyhow::{Context, Result, anyhow, ensure};
use clap::Args;
use qwen_llm::deepseek_v41::loader::{ByteCounters, LoadLimits, ResidentWeights, SourceStore};
use qwen_llm::metal::MetalContext;
use qwen_llm::sampling::SamplingConfig;
use serde_json::{Value, json};
use std::io::Write;
use std::{path::PathBuf, sync::Arc, time::Instant};

use qwen_cli::deepseek_v41_tokenizer as engram;

#[path = "deepseek_v41_cli/generate.rs"]
mod generate;
#[path = "deepseek_v41_cli/trace.rs"]
mod trace;

#[derive(Debug, Args)]
#[command(group(clap::ArgGroup::new("v41_generation").args(["prompt", "token_ids"])))]
pub(crate) struct Options {
    /// Old REAP-272E root containing config.json and safetensors index.
    #[arg(long)]
    model_dir: PathBuf,
    /// Original-base Engram shards 47/48; defaults to model-dir.
    #[arg(long)]
    engram_dir: Option<PathBuf>,
    /// Header-only inspection (the default); never read weight payloads.
    #[arg(long, conflicts_with_all = ["load_only", "prompt", "token_ids", "serve"])]
    inspect: bool,
    /// Materialize weights once, report residency, then release and exit.
    #[arg(long, conflicts_with_all = ["prompt", "token_ids", "serve"])]
    load_only: bool,
    /// Serve /v1/responses and /v1/models with fresh per-request state; no CED/caches.
    #[arg(long, conflicts_with_all = ["prompt", "token_ids"])]
    serve: bool,
    /// Loopback HTTP listen address for --serve.
    #[arg(long, requires = "serve", default_value = "127.0.0.1:8737")]
    addr: String,
    /// Served model name returned by /v1/models.
    #[arg(long, requires = "serve", default_value = "deepseek-v41")]
    model_id: String,
    /// User text rendered with the official V4.1 chat encoding.
    #[arg(long, conflicts_with = "token_ids")]
    prompt: Option<String>,
    /// Exact raw prompt token IDs, comma-delimited; bypasses chat rendering.
    #[arg(long, value_delimiter = ',', num_args = 1.., conflicts_with = "prompt")]
    token_ids: Option<Vec<u32>>,
    /// Dump every forwarded position's 40 layers and logits as little-endian F32.
    /// Requires generation and a fresh or empty directory; never overwrites traces.
    #[arg(long, requires = "v41_generation", conflicts_with_all = ["serve", "inspect", "load_only"])]
    trace_dir: Option<PathBuf>,
    /// System text for --prompt.
    #[arg(long, requires = "prompt")]
    system: Option<String>,
    /// Official V4.1 thinking tier for --prompt (default: chat).
    #[arg(long, requires = "prompt", value_parser = ["none", "low", "high", "xhigh", "max"])]
    reasoning_effort: Option<String>,
    /// Maximum sampled tokens, or server default max_output_tokens; includes EOS.
    #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..))]
    max_tokens: u32,
    /// Zero uses greedy decoding; positive values use the existing sampler.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Session capacity; inspect/load-only do not allocate KV.
    #[arg(long, default_value_t = 4096, value_parser = clap::value_parser!(u32).range(1..=1048576))]
    context: u32,
    /// Explicit resident weight ceiling in bytes; default recommended minus reserve and existing allocations.
    #[arg(long)]
    weight_budget_bytes: Option<u64>,
    /// Headroom left for KV/scratch; not a claim that a context fits.
    #[arg(long, default_value_t = 4096)]
    reserve_mib: u64,
    /// Include current Metal allocation and OS process footprint snapshots.
    #[arg(long)]
    report_memory: bool,
    /// Build and validate the tokenizer compression map (no large map output).
    #[arg(long)]
    check_engram_map: bool,
}

fn budget(
    recommended: u64,
    allocated: u64,
    reserve_mib: u64,
    explicit: Option<u64>,
) -> Result<u64> {
    let reserve = reserve_mib
        .checked_mul(1024 * 1024)
        .context("reserve overflow")?;
    let available = recommended
        .checked_sub(allocated)
        .and_then(|n| n.checked_sub(reserve))
        .context("recommended Metal budget has no room after reserve/current allocations")?;
    let budget = explicit.unwrap_or(available);
    ensure!(
        budget > 0 && budget <= available,
        "weight budget must be positive and <= {available} bytes after reserve"
    );
    Ok(budget)
}

fn counters(bytes: ByteCounters) -> Value {
    json!({"packed_experts_bytes":bytes.packed_experts, "dense_f32_bytes":bytes.dense_f32,
        "resident_payload_bytes":bytes.resident_bytes(), "disk_engram_bytes":bytes.disk_engram,
        "skipped_source_bytes":bytes.skipped_source, "peak_payload_staging_bound_bytes":bytes.peak_staging,
        "source_payload_bytes_read":bytes.source_bytes_read})
}

fn memory(ctx: Option<&MetalContext>) -> Result<Value> {
    let process = qwen_llm::pid_metrics::PidSnapshot::now()?;
    Ok(
        json!({"metal_allocated_bytes":ctx.map(MetalContext::current_allocated_size),
        "metal_recommended_bytes":ctx.map(MetalContext::recommended_max_working_set_size),
        "os_resident_bytes":process.resident_size, "os_physical_footprint_bytes":process.phys_footprint,
        "os_disk_read_bytes":process.diskio_bytesread}),
    )
}

fn chat_prompt(user: &str, system: Option<&str>, effort: Option<&str>) -> Result<String> {
    use deepseek_recipe_core::{conversation::Conversation, messages::InputMessage};
    use deepseek_recipe_encoding::{PromptEncoding, dsv41::DeepseekV41Encoding};
    let reasoning_effort = match effort {
        None | Some("none") => None,
        Some(value) => Some(serde_json::from_value(json!(value))?),
    };
    let mut conversation = Conversation {
        thinking_mode: reasoning_effort.is_some(),
        reasoning_effort,
        ..Conversation::default()
    };
    if let Some(system) = system {
        conversation.messages.push(InputMessage::System {
            content: system.into(),
        });
    }
    conversation.messages.push(InputMessage::User {
        content: user.into(),
        image_sources: vec![],
    });
    Ok(DeepseekV41Encoding::new()
        .render_conversation(&conversation)
        .prompt)
}

fn prompt_ids(options: &Options, tokenizer: &tokenizers::Tokenizer) -> Result<Vec<u32>> {
    if let Some(ids) = &options.token_ids {
        return Ok(ids.clone());
    }
    let prompt = chat_prompt(
        options.prompt.as_deref().context("missing prompt")?,
        options.system.as_deref(),
        options.reasoning_effort.as_deref(),
    )?;
    Ok(tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow!(e.to_string()))?
        .get_ids()
        .to_vec())
}

pub(crate) fn run(options: Options) -> Result<()> {
    crate::shutdown::checkpoint()?;
    let generating = options.prompt.is_some() || options.token_ids.is_some();
    let report = |value: Value| {
        if generating || options.serve {
            eprintln!("{value}");
        } else {
            println!("{value}");
        }
    };
    let sampling = SamplingConfig {
        temperature: options.temperature,
        top_k: options.top_k,
        top_p: options.top_p,
        min_p: options.min_p,
        seed: options.seed,
    }
    .validate()?;
    let started = Instant::now();
    let config: Value = serde_json::from_slice(&crate::bounded_file::read_regular_file_bounded(
        &options.model_dir.join("config.json"),
        1024 * 1024,
    )?)?;
    let max_context = config["text_config"]["max_position_embeddings"]
        .as_u64()
        .context("missing max_position_embeddings")?;
    ensure!(
        u64::from(options.context) <= max_context,
        "context exceeds root config limit {max_context}"
    );
    ensure!(
        options
            .model_dir
            .join("model.safetensors.index.json")
            .is_file(),
        "an indexed checkpoint is required to establish tensor completeness"
    );
    let source = SourceStore::open(&options.model_dir).map_err(|e| anyhow!(e))?;
    let engram_source = Arc::new(
        SourceStore::open_engram(options.engram_dir.as_ref().unwrap_or(&options.model_dir))
            .map_err(|e| anyhow!(e))?,
    );
    report(
        json!({"stage":"headers", "main_missing_shards":source.missing_shards,
        "engram_missing_shards":engram_source.missing_shards,
        "available_main_tensors":source.descriptors().count(), "requested_context":options.context,
        "kv_allocated":false, "source_revision_expected":qwen_llm::deepseek_v41::loader::OLD_REVISION,
        "provenance_verified":false}),
    );
    let plan = ResidentWeights::plan(&source, &engram_source, &config).map_err(|e| anyhow!(e))?;
    report(
        json!({"stage":"plan", "bytes":counters(plan), "wall_seconds":started.elapsed().as_secs_f64()}),
    );
    let mut prepared = if options.check_engram_map || generating || options.serve {
        let mut tokenizer =
            tokenizers::Tokenizer::from_file(options.model_dir.join("tokenizer.json"))
                .map_err(|e| anyhow!(e.to_string()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow!(e.to_string()))?;
        tokenizer.with_padding(None);
        let map = engram::CompressedTokenMap::for_config(&tokenizer, &config)?;
        if generating || options.serve {
            let eos = u32::try_from(config["eos_token_id"].as_u64().context("missing EOS ID")?)?;
            ensure!(
                tokenizer.token_to_id(deepseek_recipe_encoding::v4::EOS_TOKEN) == Some(eos),
                "tokenizer/config EOS mismatch"
            );
        }
        report(json!({"stage":"engram_map", "raw_vocab":map.lookup.len(),
            "compressed_vocab":map.vocab_size, "compressed_pad_id":map.pad_id}));
        let tokens = if generating {
            prompt_ids(&options, &tokenizer)?
        } else {
            vec![]
        };
        if generating {
            generate::required_context(
                &tokens,
                options.max_tokens as usize,
                options.context as usize,
                map.lookup.len(),
            )?;
        }
        Some((tokenizer, map, tokens))
    } else {
        None
    };
    let mut trace = options
        .trace_dir
        .as_deref()
        .map(|path| {
            let (_, _, tokens) = prepared
                .as_ref()
                .context("trace requires generation preparation")?;
            trace::Trace::create(
                path,
                &config,
                tokens,
                options.context,
                options.max_tokens,
                sampling,
            )
        })
        .transpose()?;
    if !options.load_only && !generating && !options.serve {
        if options.report_memory {
            report(json!({"stage":"inspect_memory", "memory":memory(None)?}));
        }
        return Ok(());
    }
    let server = if options.serve {
        ensure!(
            !options.model_id.is_empty(),
            "served model ID must not be empty"
        );
        let (tokenizer, map, _) = prepared.take().context("missing server tokenizer")?;
        let eos = u32::try_from(config["eos_token_id"].as_u64().context("missing EOS ID")?)?;
        let settings = crate::serve::backend_v41::Settings::new(
            tokenizer,
            options.context as usize,
            options.max_tokens as usize,
            sampling,
            eos,
            options.model_id.clone(),
        )?;
        let listener = crate::serve::bind_loopback(&options.addr)?;
        Some((settings, map, listener))
    } else {
        None
    };
    crate::shutdown::checkpoint()?;
    let ctx = MetalContext::new()?;
    let limit = budget(
        ctx.recommended_max_working_set_size(),
        ctx.current_allocated_size(),
        options.reserve_mib,
        options.weight_budget_bytes,
    )?;
    report(
        json!({"stage":"load_start", "weight_budget_bytes":limit, "reserve_mib":options.reserve_mib,
        "memory":if options.report_memory { Some(memory(Some(&ctx))?) } else { None }}),
    );
    ensure!(
        plan.resident_bytes() <= limit,
        "planned resident weights {} exceed budget {limit}",
        plan.resident_bytes()
    );
    let allocated_before = ctx.current_allocated_size();
    let start = Instant::now();
    let weights = ResidentWeights::load(
        &ctx,
        &source,
        engram_source,
        &config,
        LoadLimits {
            resident_bytes: limit,
        },
    )
    .map_err(|e| anyhow!(e))?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    drop(source);
    report(
        json!({"stage":"loaded", "load_wall_seconds":start.elapsed().as_secs_f64(),
        "bytes":counters(weights.bytes), "resident_banks":weights.banks.len(), "resident_dense_tensors":weights.dense.len(),
        "metal_allocated_bytes":ctx.current_allocated_size(),
        "metal_allocation_delta_bytes":ctx.current_allocated_size().saturating_sub(allocated_before),
        "memory":if options.report_memory { Some(memory(Some(&ctx))?) } else { None }}),
    );
    if let Some((settings, map, listener)) = server {
        let raw_pad = u32::try_from(
            config["text_config"]["engram_pad_token_id"]
                .as_u64()
                .context("missing raw Engram pad ID")?,
        )?;
        let forward = qwen_llm::deepseek_v41::forward::Forward::from_resident(
            &ctx, &weights, map.lookup, raw_pad,
        )
        .map_err(|e| anyhow!(e))?;
        let mut backend = crate::serve::backend_v41::Backend::new(&ctx, forward, settings);
        report(
            json!({"stage":"serve_ready", "model":options.model_id, "context":options.context,
            "default_max_output_tokens":options.max_tokens, "prefix_reuse":false, "ced":false,
            "memory":if options.report_memory {Some(memory(Some(&ctx))?)} else {None}}),
        );
        crate::serve::serve_v41(listener, &mut backend, load_ms)?;
    } else if generating {
        let (tokenizer, map, tokens) = prepared.context("missing generation preparation")?;
        let raw_pad = u32::try_from(
            config["text_config"]["engram_pad_token_id"]
                .as_u64()
                .context("missing raw Engram pad ID")?,
        )?;
        let eos = u32::try_from(config["eos_token_id"].as_u64().context("missing EOS ID")?)?;
        ensure!(
            tokenizer.token_to_id(deepseek_recipe_encoding::v4::EOS_TOKEN) == Some(eos),
            "tokenizer/config EOS mismatch"
        );
        let forward = qwen_llm::deepseek_v41::forward::Forward::from_resident(
            &ctx, &weights, map.lookup, raw_pad,
        )
        .map_err(|e| anyhow!(e))?;
        let required = generate::required_context(
            &tokens,
            options.max_tokens as usize,
            options.context as usize,
            tokenizer.get_vocab_size(true),
        )?;
        let request_memory = forward
            .estimate_request_memory(required)
            .map_err(|e| anyhow!(e))?;
        ensure!(
            serde_json::to_value(tokenizer.get_decoder())?["type"] == "ByteLevel",
            "V4.1 generation requires ByteLevel decoding for output budgeting"
        );
        let piece_bound = tokenizer
            .get_vocab(true)
            .keys()
            .map(String::len)
            .max()
            .context("empty vocabulary")?;
        // Decoded strings (including the final decode), decoder IDs, and growth
        // slack. Tracing retains only per-position JSON metadata, not activations.
        let output_peak = u64::from(options.max_tokens)
            .checked_mul(u64::try_from(piece_bound)?)
            .and_then(|n| n.checked_mul(3 * 8))
            .context("CLI output memory overflow")?;
        let trace_peak = if trace.is_some() {
            u64::try_from(required)?
                .checked_mul(4096)
                .context("trace metadata memory overflow")?
        } else {
            0
        };
        let caller_reserve = crate::serve::backend_v41::caller_cpu_reserve(
            output_peak,
            trace_peak,
            tokenizer.get_vocab_size(true),
        )
        .context("CLI request memory overflow")?;
        let admit = || {
            crate::serve::backend_v41::admit_request_memory(
                request_memory,
                caller_reserve,
                ctx.memory_signals(),
            )
        };
        report(
            json!({"stage":"request_memory", "forward_positions":required,
            "cpu_request_bytes":request_memory.cpu_bytes, "metal_request_bytes":request_memory.metal_bytes,
            "caller_cpu_reserve_bytes":caller_reserve, "recheck":"full_estimate_before_each_forward"}),
        );
        admit()?;
        let mut state = forward
            .new_state(options.context as usize)
            .map_err(|e| anyhow!(e))?;
        let mut decoder = tokenizer.decode_stream(false);
        let mut emitted = String::new();
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        let started = Instant::now();
        let mut first_token_seconds = None;
        let cancel = crate::shutdown::cancellation_flag();
        let outcome = generate::run(
            &tokens,
            options.max_tokens as usize,
            tokenizer.get_vocab_size(true),
            &[eos],
            sampling,
            cancel,
            |token| {
                // Same conservative current-headroom check as HTTP, before any
                // prefill/decode growth, not just before the first state.
                admit()?;
                if let Some(trace) = trace.as_mut() {
                    trace.begin_step(state.position(), token)?;
                    let logits = forward
                        .step_with_trace(
                            &mut state,
                            token,
                            Some(cancel),
                            Some(&mut |layer| trace.layer(layer).map_err(|e| format!("{e:#}"))),
                        )
                        .map_err(|e| anyhow!(e))?;
                    trace.finish_step(&logits)?;
                    Ok(logits)
                } else {
                    forward
                        .step(&mut state, token, Some(cancel))
                        .map_err(|e| anyhow!(e))
                }
            },
            |token| {
                first_token_seconds.get_or_insert_with(|| started.elapsed().as_secs_f64());
                if let Some(piece) = decoder.step(token).map_err(|e| anyhow!(e.to_string()))? {
                    stdout.write_all(piece.as_bytes())?;
                    stdout.flush()?;
                    emitted.push_str(&piece);
                }
                Ok(())
            },
        )?;
        let visible_ids =
            &outcome.tokens[..outcome.tokens.len() - usize::from(outcome.stop.is_some())];
        stdout.write_all(generate::finish_text(&tokenizer, visible_ids, &emitted)?.as_bytes())?;
        stdout.flush()?;
        if let Some(trace) = trace {
            trace.finish(&outcome.tokens, outcome.stop)?;
        }
        report(
            json!({"stage":"generated", "input_tokens":tokens.len(), "output_tokens":outcome.tokens.len(),
            "generated_token_ids":outcome.tokens, "stop_token_id":outcome.stop,
            "stop_reason":if outcome.stop.is_some() {"stop_token"} else {"token_limit"},
            "first_token_seconds":first_token_seconds, "generation_wall_seconds":started.elapsed().as_secs_f64(),
            "seed":options.seed, "memory":if options.report_memory {Some(memory(Some(&ctx))?)} else {None}}),
        );
    }
    drop(weights);
    if options.report_memory {
        report(json!({"stage":"released", "memory":memory(Some(&ctx))?}));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        options: Options,
    }

    #[test]
    fn v41_cli_is_inspect_by_default_and_load_is_explicit() {
        let parsed = TestCli::try_parse_from(["qwen", "--model-dir", "/models/reap"]).unwrap();
        assert!(!parsed.options.load_only);
        assert_eq!(parsed.options.reserve_mib, 4096);
        assert_eq!(parsed.options.context, 4096);
        for flags in [
            vec!["--inspect", "--load-only"],
            vec!["--context", "0"],
            vec!["--context", "1048577"],
            vec!["--prompt", "hi", "--load-only"],
            vec!["--prompt", "hi", "--token-ids", "0,1"],
            vec!["--max-tokens", "0"],
            vec!["--serve", "--load-only"],
            vec!["--serve", "--inspect"],
            vec!["--serve", "--prompt", "hi"],
            vec!["--serve", "--token-ids", "0,1"],
            vec!["--serve", "--prefix-reuse"],
            vec!["--trace-dir", "/tmp/trace"],
            vec!["--serve", "--trace-dir", "/tmp/trace"],
            vec!["--inspect", "--trace-dir", "/tmp/trace"],
            vec!["--load-only", "--trace-dir", "/tmp/trace"],
        ] {
            let mut args = vec!["qwen", "--model-dir", "/models/reap"];
            args.extend(flags);
            assert!(TestCli::try_parse_from(args).is_err());
        }
        let parsed = TestCli::try_parse_from([
            "qwen",
            "--model-dir",
            "/models/reap",
            "--load-only",
            "--report-memory",
            "--context",
            "8192",
        ])
        .unwrap();
        assert!(parsed.options.load_only && parsed.options.report_memory);
        let parsed = TestCli::try_parse_from([
            "qwen",
            "--model-dir",
            "/models/reap",
            "--token-ids",
            "0,42,128",
            "--max-tokens",
            "1",
        ])
        .unwrap();
        assert_eq!(parsed.options.token_ids, Some(vec![0, 42, 128]));
        let parsed = TestCli::try_parse_from([
            "qwen",
            "--model-dir",
            "/models/reap",
            "--prompt",
            "Hello",
            "--reasoning-effort",
            "high",
        ])
        .unwrap();
        assert_eq!(parsed.options.prompt.as_deref(), Some("Hello"));
        let parsed = TestCli::try_parse_from([
            "qwen",
            "--model-dir",
            "/models/reap",
            "--serve",
            "--reserve-mib",
            "1024",
            "--context",
            "128",
        ])
        .unwrap();
        assert!(parsed.options.serve);
        assert_eq!(parsed.options.addr, "127.0.0.1:8737");
        assert_eq!(parsed.options.model_id, "deepseek-v41");
        assert_eq!(parsed.options.reserve_mib, 1024);
        for input in [vec!["--prompt", "hi"], vec!["--token-ids", "0,42"]] {
            let mut args = vec![
                "qwen",
                "--model-dir",
                "/models/reap",
                "--trace-dir",
                "/tmp/trace",
            ];
            args.extend(input);
            let parsed = TestCli::try_parse_from(args).unwrap();
            assert_eq!(
                parsed.options.trace_dir.as_deref(),
                Some(std::path::Path::new("/tmp/trace"))
            );
        }
    }

    #[test]
    fn v41_cli_chat_render_matches_server_official_profile() {
        let profile =
            crate::serve::protocol_v41::profile(crate::serve::request_profile::OutputLimits::TEST);
        for effort in ["none", "low", "high", "xhigh", "max"] {
            let request = profile.parse(&json!({"model":"m", "input":"Hello", "instructions":"Be concise", "reasoning":{"effort":effort}})).unwrap();
            assert_eq!(
                chat_prompt("Hello", Some("Be concise"), Some(effort)).unwrap(),
                profile.render(&request).unwrap()
            );
        }
    }
    #[test]
    fn v41_weight_budget_preserves_reserve_and_existing_allocations() {
        let gib = 1024 * 1024 * 1024;
        assert_eq!(budget(256 * gib, gib, 4096, None).unwrap(), 251 * gib);
        assert!(budget(4 * gib, 1, 4096, None).is_err());
        assert!(budget(256 * gib, 0, u64::MAX, None).is_err());
        assert!(budget(256 * gib, 0, 4096, Some(253 * gib)).is_err());
        assert!(budget(256 * gib, 0, 4096, Some(0)).is_err());
        // Reported checkpoint and device sizes, rounded only for this policy test.
        let recommended = (222.7 * gib as f64) as u64;
        let resident = (221.33 * gib as f64) as u64;
        assert!(budget(recommended, 0, 4096, None).unwrap() < resident);
        assert!(budget(recommended, 0, 1024, None).unwrap() > resident);
    }
}
