#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! GLM B=1 solo-path family holds and per-dispatch critical path.
//! MODEL DEPTH [STEPS=32] [bounds|profile|serial]
//! Requires `--features perf-ablations`; run under the canonical Metal lease.
use qwen_llm::{
    gguf::GgufFile,
    glm5_next_metal::*,
    metal::*,
    sampling::{Sampler, SamplingConfig},
};
use serde_json::{Value, json};
use std::{io::Write, time::Instant};

fn emit(v: Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
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

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let depth: usize = args[2].parse()?;
    let steps: usize = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(32);
    let mode = args.get(4).map(String::as_str).unwrap_or("bounds");
    let ctx = MetalContext::new()?;
    let g = GgufFile::open(&args[1])?;
    let w = Glm5NextWeights::load(&ctx, &g)?;
    let f: Value = serde_json::from_slice(include_bytes!(
        "../../../scripts/reference/glm53/quality-v1.json"
    ))?;
    let ids: Vec<u32> = f["documents"][0]["tokens"]
        .as_str()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let prompt: Vec<_> = ids.iter().copied().cycle().take(depth).collect();
    let cap = depth + steps + 1024;
    let mut s = Glm5NextSession::with_prefill_rows(&ctx, &w, cap, 2048)?;
    let mut head = s.prefill_packed(&ctx, &prompt)?;
    for &t in ids.iter().take(64) {
        head = s.forward(&ctx, t)?;
    }
    let entry = s.capture_snapshot()?;
    // Compile the fixed pipelines and import any lazy scratch before the arms.
    s.generate_gpu_tokens(&ctx, &head, &mut greedy(), 2, &[], |_| true)?;
    emit(json!({"kind":"header","depth":depth,"steps":steps,"mode":mode,"thermal":thermal()}));

    if mode == "profile" {
        s.restore_snapshot(&entry)?;
        let guard = dispatch_profile_begin(&ctx, 16384)?;
        let token = ids[64 % ids.len()];
        let _ = s.forward(&ctx, token)?;
        let report = dispatch_profile_finish(&ctx, guard)?;
        emit(json!({"kind":"profile","overflow":report.overflow,"rows":report.rows.len()}));
        for (i, row) in report.rows.iter().enumerate() {
            emit(json!({
                "kind":"dispatch","index":i,"kernel":row.kernel,
                "label":row.label,"layer":row.layer,
                "gpu_ns":row.gpu_ns,"gpu_start":row.gpu_start,"gpu_end":row.gpu_end,
                "grid":row.grid,"threads":row.threads,
                "concurrent":row.concurrent,"bindings":row.bindings()
            }));
        }
        return Ok(());
    }

    if mode == "serial" {
        s.restore_snapshot(&entry)?;
        let mut sampler = greedy();
        let mut token = sampler.sample(&head)?.token;
        let started = Instant::now();
        for _ in 1..steps {
            token = sampler.sample(&s.forward(&ctx, token as u32)?)?.token;
        }
        let wall = started.elapsed().as_secs_f64() * 1e3;
        emit(
            json!({"kind":"arm","group":"serial","ms_token":wall / (steps - 1) as f64,"thermal":thermal()}),
        );
        return Ok(());
    }

    if mode == "research" {
        // Compile both arms before the paired packet.
        for arm in [1u8, 2] {
            s.restore_snapshot(&entry)?;
            let _override = diagnostic_q6_research(arm);
            s.generate_gpu_tokens(&ctx, &head, &mut greedy(), 2, &[], |_| true)?;
        }
        for rep in 0..2 {
            for arm in [2u8, 1, 1, 2] {
                s.restore_snapshot(&entry)?;
                let _override = diagnostic_q6_research(arm);
                let mut sampler = greedy();
                let started = Instant::now();
                let stats =
                    s.generate_gpu_tokens(&ctx, &head, &mut sampler, steps, &[], |_| true)?;
                let wall = started.elapsed().as_secs_f64() * 1e3;
                emit(json!({
                    "kind":"arm","rep":rep,
                    "group":if arm == 1 {"research"} else {"reference"},
                    "ms_token":wall / (steps - 1) as f64,
                    "wall_ms":wall,"gpu_ms":stats.gpu_ms,"encode_ms":stats.encode_cpu_ms,
                    "wait_ms":stats.completion_wait_ms,"thermal":thermal()
                }));
            }
        }
        return Ok(());
    }

    if mode == "gridholds" {
        let holds: [(&str, &str, usize, u64); 10] = [
            ("", "q6_r1", 0, 0),
            ("q6_shared_gate_up", "q6_shared_swiglu", 512, 6_881_280),
            ("q6_mla_q_a", "q6_r1", 768, 5_160_960),
            ("q6_shared_down", "q6_r1", 2048, 6_881_280),
            ("q6_kda_out", "q6_r1", 2048, 27_525_120),
            ("q6_dense_down", "q6_r1", 2048, 41_287_680),
            ("q6_dense_gate_up", "q6_r1", 6144, 41_287_680),
            ("q6_kda_qkv_x3", "q6_x3", 6144, 27_525_120),
            ("q6_head", "q6_two_row", 38_720, 520_396_800),
            ("", "q6_r1", 0, 0),
        ];
        for (label, group, grid, bytes) in holds {
            for _ in 0..2 {
                if thermal() >= 2 {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                }
                s.restore_snapshot(&entry)?;
                let _scope = if label.is_empty() {
                    None
                } else {
                    Some(qwen_llm::metal::fn_batch_probe::scope_q6_shape(
                        group, grid, bytes,
                    ))
                };
                let mut sampler = greedy();
                let started = Instant::now();
                let stats =
                    s.generate_gpu_tokens(&ctx, &head, &mut sampler, steps, &[], |_| true)?;
                let wall = started.elapsed().as_secs_f64() * 1e3;
                emit(json!({
                    "kind":"arm","group":if label.is_empty(){"reference"}else{label},
                    "ms_token":wall / (steps - 1) as f64,
                    "wall_ms":wall,"gpu_ms":stats.gpu_ms,"encode_ms":stats.encode_cpu_ms,
                    "wait_ms":stats.completion_wait_ms,
                    "skipped":qwen_llm::metal::fn_batch_probe::skipped(),
                    "thermal":thermal()
                }));
            }
        }
        return Ok(());
    }

    let groups = [
        "",
        "q6",
        "q8",
        "routed",
        "attention",
        "index",
        "kda",
        "hc",
        "router",
        "pointwise",
    ];
    let mut order: Vec<_> = groups.into_iter().collect();
    let reverse: Vec<_> = order.iter().rev().copied().collect();
    order.extend(reverse);
    for group in order {
        if thermal() >= 2 {
            std::thread::sleep(std::time::Duration::from_secs(5));
        }
        s.restore_snapshot(&entry)?;
        let _scope = qwen_llm::metal::fn_batch_probe::scope(None, group, false);
        let mut sampler = greedy();
        let started = Instant::now();
        let stats = s.generate_gpu_tokens(&ctx, &head, &mut sampler, steps, &[], |_| true)?;
        let wall = started.elapsed().as_secs_f64() * 1e3;
        emit(json!({
            "kind":"arm","group":if group.is_empty(){"reference"}else{group},
            "ms_token":wall / (steps - 1) as f64,
            "wall_ms":wall,"gpu_ms":stats.gpu_ms,"encode_ms":stats.encode_cpu_ms,
            "wait_ms":stats.completion_wait_ms,"skipped":qwen_llm::metal::fn_batch_probe::skipped(),
            "thermal":thermal()
        }));
    }
    Ok(())
}
