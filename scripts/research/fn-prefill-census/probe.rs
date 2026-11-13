//! Flash-Next prefill stage census. Scratch uses the in-tree packed profiler.
//! MODEL CORPUS. Env: FN_CENSUS_DEPTHS=49152,81920 FN_CENSUS_SIZES=256,1024
//! FN_CENSUS_ROOT_ROWS=2048 FN_CENSUS_REPS=3 FN_CENSUS_SKIP_ROOT=0
use qwen_llm::{
    append_research as research,
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
    tokenizer::Tokenizer,
};
use serde_json::json;
use std::io::Write;
use std::time::Instant;

fn hash(values: &[f32]) -> String {
    assert!(values.iter().all(|value| value.is_finite()));
    blake3::hash(bytemuck::cast_slice(values))
        .to_hex()
        .to_string()
}

fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}

fn pressure() -> i32 {
    std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .ok()
        .and_then(|x| String::from_utf8(x.stdout).ok())
        .and_then(|x| x.split_whitespace().last()?.parse().ok())
        .unwrap_or(-1)
}

fn timing(v: &qwen_llm::qwen4exp_runtime::Qwen4ExpPrefillTiming) -> serde_json::Value {
    json!({
        "command_count": v.command_count,
        "encode_cpu_ms": v.encode_cpu_ms,
        "completion_wait_ms": v.completion_wait_ms,
        "gpu_ms": v.gpu_ms,
        "gpu_samples": v.gpu_samples,
        "total_wall_ms": v.total_wall_ms,
    })
}

fn profile_json(
    cell: &str,
    depth: usize,
    rows: usize,
    outcome: &qwen_llm::qwen4exp_runtime::Qwen4ExpPackedProfileOutcome,
) -> serde_json::Value {
    let p = outcome.profile.as_ref().ok();
    let stages = p
        .map(|p| {
            p.stages
                .iter()
                .map(|s| {
                    json!({
                        "name": s.label.name,
                        "scope": s.label.scope.as_str(),
                        "layer": s.label.layer,
                        "mixer": s.label.mixer.map(|m| format!("{m:?}")),
                        "depth": s.depth,
                        "start": s.start_sample,
                        "end": s.end_sample,
                        "gpu_ms": s.gpu_ms,
                        "fraction": s.fraction_of_gpu,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    json!({
        "kind": "profile",
        "cell": cell,
        "depth": depth,
        "rows": rows,
        "pressure": pressure(),
        "token": {
            "encode_cpu_ms": outcome.token.encode_cpu_ms,
            "completion_wait_ms": outcome.token.completion_wait_ms,
            "gpu_ms": outcome.token.gpu_ms,
            "total_wall_ms": outcome.token.total_wall_ms,
        },
        "encode": {
            "preflight_ms": outcome.encode.preflight_ms,
            "stage_inputs_ms": outcome.encode.stage_inputs_ms,
            "graph_encode_ms": outcome.encode.graph_encode_ms,
            "unattributed_ms": outcome.encode.unattributed_ms,
        },
        "command": {
            "commit_return_ms": outcome.command.commit_return_ms,
            "root_wait_ms": outcome.command.root_wait_ms,
            "child_publication_ms": outcome.command.child_publication_ms,
            "root_publish_ms": outcome.command.root_publish_ms,
            "release_total_ms": outcome.command.release_total_ms,
        },
        "sampling": outcome.sampling.as_str(),
        "sampling_fallback": outcome.sampling_fallback,
        "profile_error": outcome.profile.as_ref().err().map(|e| e.to_string()),
        "stages": stages,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let depths: Vec<usize> = std::env::var("FN_CENSUS_DEPTHS")
        .unwrap_or_else(|_| "49152,81920".into())
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let sizes: Vec<usize> = std::env::var("FN_CENSUS_SIZES")
        .unwrap_or_else(|_| "256,1024".into())
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let root_rows: usize = std::env::var("FN_CENSUS_ROOT_ROWS")
        .unwrap_or_else(|_| "2048".into())
        .parse()?;
    let reps: usize = std::env::var("FN_CENSUS_REPS")
        .unwrap_or_else(|_| "3".into())
        .parse()?;
    let skip_root = std::env::var("FN_CENSUS_SKIP_ROOT").is_ok_and(|v| v != "0");
    let dispatch_census = std::env::var("FN_CENSUS_DISPATCH").is_ok_and(|v| v != "0");

    let ctx = MetalContext::new()?;
    let g = GgufFile::open(&args[1])?;
    let config = Qwen4ExpConfig::from_gguf(&g)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let mut text = std::fs::read_to_string(&args[2])?;
    let max = depths.iter().copied().max().unwrap() + sizes.iter().copied().max().unwrap() + 16;
    while tok.encode(&text, false)?.len() < max {
        let extra = text.clone();
        text.push_str(&extra);
    }
    let ids: Vec<u32> = tok
        .encode(&text, false)?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let limit = max + 4096;
    let cap = Qwen4ExpSessionCapacity::for_forward_limit(&config, limit)?;
    let mut model = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &g, cap, limit)?;
    emit(json!({
        "kind": "identity",
        "depths": depths,
        "sizes": sizes,
        "root_rows": root_rows,
        "max_tokens": max,
        "pressure": pressure(),
        "config": {
            "layers": config.layer_count,
            "hidden": config.hidden_size,
            "vocab": config.vocab_size,
            "full_attention_interval": config.full_attention_interval,
            "hc_count": config.hyper_connection.count,
            "hc_low_rank": config.hyper_connection.low_rank,
            "attn_heads": config.attention.query_heads,
            "attn_kv_heads": config.attention.kv_heads,
            "attn_key_dim": config.attention.key_head_dim,
            "attn_value_dim": config.attention.value_head_dim,
            "gdn_key_heads": config.gated_delta_net.key_heads,
            "gdn_value_heads": config.gated_delta_net.value_heads,
            "gdn_key_dim": config.gated_delta_net.key_head_dim,
            "gdn_value_dim": config.gated_delta_net.value_head_dim,
            "gdn_inner": config.gated_delta_net.inner_size,
            "experts": config.moe.expert_count,
            "topk": config.moe.experts_per_token,
            "expert_intermediate": config.moe.expert_intermediate_size,
            "shared_intermediate": config.moe.shared_expert_intermediate_size,
            "qsa_heads": config.qsa.query_heads,
            "qsa_kv_heads": config.qsa.key_heads,
            "qsa_head_dim": config.qsa.head_dim,
            "qsa_token_budget": config.qsa.token_budget,
        },
        "inventory": model.append_research_inventory(),
    }));

    let mut runner = model.create_runner(&ctx)?;
    runner.configure_causal_executor(true);
    runner.configure_gpu_token_loop(true);

    if !skip_root {
        research::routes_begin();
        let out = runner.prefill_continuation_profiled(&ids[..root_rows])?;
        let routes = research::routes_finish(root_rows, config.moe.experts_per_token as usize);
        emit(profile_json("root_first_chunk", 0, root_rows, &out));
        emit(
            json!({"kind":"routes","cell":"root_first_chunk","depth":0,"rows":root_rows,"layers":routes}),
        );
        runner.reset()?;
    }

    let mut snapshot = None;
    let mut snapshot_logits = Vec::new();
    let mut previous = 0usize;
    for (ordinal, &depth) in depths.iter().enumerate() {
        if previous == 0 {
            runner.reset()?;
            let start = Instant::now();
            runner.prefill(&ids[..depth])?;
            let wall = start.elapsed().as_secs_f64() * 1e3;
            let t = runner.last_prefill_timing().ok_or("missing timing")?;
            emit(
                json!({"kind":"root","cell":"root","depth":depth,"rows":depth,"wall_ms":wall,"timing":timing(&t),"pressure":pressure(),"logits_hash":hash(&runner.logits()?.to_vec())}),
            );
        } else {
            runner.reset()?;
            runner.restore_snapshot_with_logits(
                snapshot.as_ref().ok_or("missing snapshot")?,
                &snapshot_logits,
            )?;
            let start = Instant::now();
            runner
                .prefill_continuation_with_command_checkpoint(&ids[previous..depth], || Ok(()))?;
            let wall = start.elapsed().as_secs_f64() * 1e3;
            let t = runner.last_prefill_timing().ok_or("missing timing")?;
            emit(
                json!({"kind":"extend","cell":"root","from":previous,"depth":depth,"rows":depth-previous,"wall_ms":wall,"timing":timing(&t),"pressure":pressure(),"logits_hash":hash(&runner.logits()?.to_vec())}),
            );
        }
        snapshot = Some(runner.capture_snapshot()?);
        snapshot_logits = runner.logits()?.to_vec();
        previous = depth;

        for &n in &sizes {
            let snap = snapshot.as_ref().unwrap();
            for rep in 0..reps {
                runner.restore_snapshot_with_logits(snap, &snapshot_logits)?;
                let start = Instant::now();
                runner
                    .prefill_continuation_with_command_checkpoint(&ids[depth..depth + n], || {
                        Ok(())
                    })?;
                let wall = start.elapsed().as_secs_f64() * 1e3;
                let t = runner.last_prefill_timing().ok_or("missing timing")?;
                emit(
                    json!({"kind":"append","cell":"cell","depth":depth,"rows":n,"rep":rep,"wall_ms":wall,"encode_cpu_ms":t.encode_cpu_ms,"gpu_ms":t.gpu_ms,"wait_ms":t.completion_wait_ms,"commands":t.command_count,"pressure":pressure(),"logits_hash":hash(&runner.logits()?.to_vec())}),
                );
            }
            if dispatch_census {
                runner.restore_snapshot_with_logits(snap, &snapshot_logits)?;
                let guard = qwen_llm::metal::dispatch_profile_begin(&ctx, 60_000)?;
                let start = Instant::now();
                runner
                    .prefill_continuation_with_command_checkpoint(&ids[depth..depth + n], || {
                        Ok(())
                    })?;
                let wall = start.elapsed().as_secs_f64() * 1e3;
                let report = qwen_llm::metal::dispatch_profile_finish(&ctx, guard)?;
                emit(
                    json!({"kind":"dispatch","cell":"append","depth":depth,"rows":n,"wall_ms":wall,"overflow":report.overflow,"dispatches":report.rows,"logits_hash":hash(&runner.logits()?.to_vec())}),
                );
                continue;
            }
            runner.restore_snapshot_with_logits(snap, &snapshot_logits)?;
            research::routes_begin();
            let out = runner.prefill_continuation_profiled(&ids[depth..depth + n])?;
            let routes = research::routes_finish(n, config.moe.experts_per_token as usize);
            let cell = format!("append");
            emit(profile_json(&cell, depth, n, &out));
            emit(json!({"kind":"routes","cell":cell,"depth":depth,"rows":n,"layers":routes}));
        }

        // A 2048-row chunk after this depth is the late-root geometry.
        if root_rows > 0 && depth + root_rows <= max {
            runner.reset()?;
            runner.restore_snapshot_with_logits(
                snapshot.as_ref().ok_or("missing snapshot")?,
                &snapshot_logits,
            )?;
            if dispatch_census {
                let guard = qwen_llm::metal::dispatch_profile_begin(&ctx, 60_000)?;
                let start = Instant::now();
                runner.prefill_continuation_with_command_checkpoint(
                    &ids[depth..depth + root_rows],
                    || Ok(()),
                )?;
                let wall = start.elapsed().as_secs_f64() * 1e3;
                let report = qwen_llm::metal::dispatch_profile_finish(&ctx, guard)?;
                emit(
                    json!({"kind":"dispatch","cell":"chunk","depth":depth,"rows":root_rows,"wall_ms":wall,"overflow":report.overflow,"dispatches":report.rows,"logits_hash":hash(&runner.logits()?.to_vec())}),
                );
            } else {
                research::routes_begin();
                let out = runner.prefill_continuation_profiled(&ids[depth..depth + root_rows])?;
                let routes =
                    research::routes_finish(root_rows, config.moe.experts_per_token as usize);
                emit(profile_json("chunk", depth, root_rows, &out));
                emit(
                    json!({"kind":"routes","cell":"chunk","depth":depth,"rows":root_rows,"layers":routes}),
                );
            }
        }
        if ordinal + 1 == depths.len() {
            break;
        }
    }
    Ok(())
}
