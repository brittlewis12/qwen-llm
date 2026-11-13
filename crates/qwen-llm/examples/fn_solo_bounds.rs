#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Flash-Next B=1 solo-path family holds and causal wave graph.
//! MODEL DEPTH [STEPS=32] [bounds|graph|nodes|serial]
//! Requires `--features perf-ablations`; run under the canonical Metal lease.
use qwen_llm::{
    gguf::GgufFile,
    metal::{self, MetalContext},
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
    sampling::{Sampler, SamplingConfig},
};
use serde_json::json;
use std::time::Instant;

fn ids(n: usize, v: u32, mut s: u64) -> Vec<u32> {
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % u64::from(v)) as u32
        })
        .collect()
}

fn thermal() -> u32 {
    std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .last()?
                .parse()
                .ok()
        })
        .unwrap_or(u32::MAX)
}

fn greedy() -> Sampler {
    Sampler::new(SamplingConfig {
        temperature: 0.,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.,
        seed: 1,
    })
    .unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let depth: usize = args[2].parse()?;
    let steps: usize = args.get(3).map(|x| x.parse()).transpose()?.unwrap_or(32);
    let mode = args.get(4).map(String::as_str).unwrap_or("bounds");
    let ctx = MetalContext::new()?;
    let gguf = GgufFile::open(&args[1])?;
    let config = Qwen4ExpConfig::from_gguf(&gguf)?;
    let prefill_nodes = mode == "prefill-nodes";
    let append_nodes = mode == "append-nodes";
    let capture_at = 49152usize;
    let capture_rows = if prefill_nodes { 2048 } else { 1024 };
    let limit = if prefill_nodes || append_nodes {
        capture_at + capture_rows + 256
    } else {
        depth + steps + 256
    };
    let cap = Qwen4ExpSessionCapacity::for_forward_limit(&config, limit)?;
    let mut model = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, cap, limit)?;
    let mut runner = model.create_runner(&ctx)?;
    runner.configure_causal_executor(true);
    runner.configure_gpu_token_loop(true);
    if prefill_nodes || append_nodes {
        let prefix = if prefill_nodes {
            capture_at - capture_rows
        } else {
            capture_at
        };
        let all = ids(
            capture_at + capture_rows,
            config.vocab_size,
            1 ^ 0x9e3779b97f4a7c15,
        );
        runner.prefill(&all[..prefix])?;
        let entry = runner.capture_snapshot()?;
        let entry_logits = runner.logits()?.to_vec();
        let chunk = &all[prefix..];
        let expected = runner
            .prefill_continuation_with_command_checkpoint(chunk, || Ok(()))?
            .to_vec();
        runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
        let guard = metal::causal_node_graph_begin(&ctx, 16384)?;
        let actual = runner
            .prefill_continuation_with_command_checkpoint(chunk, || Ok(()))?
            .to_vec();
        assert!(
            expected
                .iter()
                .zip(&actual)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "prefill node capture changed Flash-Next logits"
        );
        let mut value = metal::causal_graph_finish(&ctx, guard)?;
        value["phase"] = json!(if prefill_nodes { "prefill" } else { "append" });
        value["capture_rows"] = json!(capture_rows);
        value["depth"] = json!(capture_at);
        value["width"] = json!(1);
        value["rounds"] = json!(1);
        value["bitwise"] = json!(true);
        value["production_path"] = json!("packed prefill");
        println!("{value}");
        return Ok(());
    }
    runner.prefill(&ids(64, config.vocab_size, 1))?;
    runner.reset()?;
    runner.prefill(&ids(depth, config.vocab_size, 1 ^ 0x9e3779b97f4a7c15))?;
    for t in ids(64, config.vocab_size, 7) {
        runner.forward_token(t)?;
    }
    let entry = runner.capture_snapshot()?;
    let entry_logits = runner.logits()?.to_vec();
    // Import PLE banks and compile the fixed pipelines before the holds.
    runner.generate_gpu_tokens(&mut greedy(), 2, &[], |_| true)?;
    println!(
        "{}",
        json!({"kind":"header","depth":depth,"steps":steps,"mode":mode,"thermal":thermal(),"vocab":config.vocab_size})
    );

    if mode == "graph" || mode == "nodes" {
        let mut reference_ms = Vec::new();
        for _ in 0..3 {
            runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
            let started = Instant::now();
            let stats = runner.generate_gpu_tokens(&mut greedy(), 32, &[], |_| true)?;
            reference_ms
                .push(started.elapsed().as_secs_f64() * 1e3 / stats.forwarded_tokens as f64);
        }
        runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
        runner.generate_gpu_tokens(&mut greedy(), steps, &[], |_| true)?;
        let expected = runner.logits()?.to_vec();
        let guard = if mode == "nodes" {
            metal::causal_node_graph_begin(&ctx, 8192)?
        } else {
            metal::causal_graph_begin(&ctx, 1000)?
        };
        runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
        let mut sampler = greedy();
        let started = Instant::now();
        let stats = runner.generate_gpu_tokens(&mut sampler, steps, &[], |_| true)?;
        let wall = started.elapsed().as_secs_f64() * 1e3;
        assert!(
            expected
                .iter()
                .zip(runner.logits()?.as_slice().iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "node capture changed FN logits"
        );
        let mut value = metal::causal_graph_finish(&ctx, guard)?;
        value["width"] = json!(1);
        value["rounds"] = json!(stats.forwarded_tokens);
        value["reference_wall_ms_token"] = json!(reference_ms);
        value["bitwise"] = json!(true);
        value["wall_ms_round"] = json!(wall / steps as f64);
        value["stats"] = json!({"wall_ms":stats.wall_ms,"gpu_ms":stats.gpu_ms,"encode_ms":stats.encode_cpu_ms,"wait_ms":stats.completion_wait_ms});
        println!("{value}");
        return Ok(());
    }

    if mode == "serial" {
        runner.configure_gpu_token_loop(false);
        runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
        let mut sampler = greedy();
        let mut token = sampler.sample(runner.logits()?.as_slice())?.token;
        let started = Instant::now();
        for _ in 1..steps {
            token = sampler
                .sample(runner.forward_token(token as u32)?.as_slice())?
                .token;
        }
        let wall = started.elapsed().as_secs_f64() * 1e3;
        println!(
            "{}",
            json!({"kind":"arm","group":"serial","ms_token":wall / (steps - 1) as f64,"thermal":thermal()})
        );
        return Ok(());
    }

    let groups = [
        "",
        "dense",
        "q6",
        "routed",
        "qsa_core",
        "hc",
        "gdn",
        "routing",
        "norms",
        "transport",
        "copies",
        "head",
        "experts",
    ];
    // Forward then reverse so drift does not favor an end of the order.
    let mut order: Vec<_> = groups.into_iter().collect();
    let reverse: Vec<_> = order.iter().rev().copied().collect();
    order.extend(reverse);
    for group in order {
        thermal_start()?;
        runner.configure_gpu_token_loop(true);
        runner.restore_snapshot_with_logits(&entry, &entry_logits)?;
        let _scope = qwen_llm::metal::fn_batch_probe::scope(None, group, false);
        let mut sampler = greedy();
        let started = Instant::now();
        let stats = runner.generate_gpu_tokens(&mut sampler, steps, &[], |_| true)?;
        let wall = started.elapsed().as_secs_f64() * 1e3;
        println!(
            "{}",
            json!({
                "kind":"arm","group":if group.is_empty(){"reference"}else{group},
                "ms_token":wall / (steps - 1) as f64,
                "wall_ms":wall,"gpu_ms":stats.gpu_ms,"encode_ms":stats.encode_cpu_ms,
                "wait_ms":stats.completion_wait_ms,"skipped":qwen_llm::metal::fn_batch_probe::skipped(),
                "thermal":thermal()
            })
        );
    }
    Ok(())
}

fn thermal_start() -> Result<(), Box<dyn std::error::Error>> {
    if thermal() >= 2 {
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    Ok(())
}
