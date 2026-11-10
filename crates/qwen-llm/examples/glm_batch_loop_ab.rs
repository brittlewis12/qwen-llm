#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Paired GLM batched loop A/B: the shipped serial `step` + host sampling versus
//! the one-ahead device-feedback loop. Greedy is token-exact; the released
//! preset reports mismatches. Timing uses restored snapshots and optional host
//! load, with the GPU-thread QoS guard on both arms.
//!
//! MODEL DEPTH [STEPS=32] [WIDTHS=2,3,4] [WARM=8] [LOAD=0]
//! Requires `--features perf-ablations`; build with `--profile iter`.
use qwen_llm::{
    gguf::GgufFile,
    glm5_next_metal::*,
    metal::{GpuThreadQos, MetalContext},
    sampling::{Sampler, SamplingConfig},
};
use serde_json::{Value, json};
use std::{
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires --features perf-ablations");
}

fn emit(v: Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}

struct Load {
    stop: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<()>>,
}
impl Load {
    fn start(n: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let threads = (0..n)
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        std::hint::spin_loop();
                    }
                })
            })
            .collect();
        Self { stop, threads }
    }
}
impl Drop for Load {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        while let Some(thread) = self.threads.pop() {
            let _ = thread.join();
        }
    }
}

#[cfg(feature = "perf-ablations")]
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let depth: usize = args[2].parse()?;
    let steps: usize = args
        .get(3)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(32usize);
    let widths: Vec<usize> = args
        .get(4)
        .map(|v| v.split(',').map(str::parse).collect())
        .transpose()?
        .unwrap_or(vec![2, 3, 4]);
    let warm: usize = args
        .get(5)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(8usize);
    let load: usize = args
        .get(6)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(0usize);
    assert!(std::env::var_os("QWEN_METAL_LEASE_FD").is_some());
    let ctx = MetalContext::new()?;
    let g = GgufFile::open(&args[1])?;
    let w = Glm5NextWeights::load(&ctx, &g)?;
    let f: Value = serde_json::from_slice(include_bytes!(
        "../../../scripts/reference/glm53/quality-v1.json"
    ))?;
    let docs: Vec<Vec<u32>> = f["documents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| {
            d["tokens"]
                .as_str()
                .unwrap()
                .split_whitespace()
                .map(|t| t.parse().unwrap())
                .collect()
        })
        .collect();
    let prefix = [0, 8, 16, 9];
    let suffix = [1, 10, 17, 11];
    let token = |r: usize, p: usize| docs[suffix[r]][p % docs[suffix[r]].len()];
    let mut ss: Vec<_> = (0..4)
        .map(|i| {
            Glm5NextSession::with_prefill_rows(&ctx, &w, depth + 512, if i == 0 { 2048 } else { 4 })
                .unwrap()
        })
        .collect();
    let mut batch = Glm5NextBatch::new(&ctx, &w, 4)?;
    let empty = ss[0].capture_snapshot()?;
    let mut banks = Vec::new();
    let mut prefill_logits: Vec<Vec<f32>> = Vec::new();
    for r in 0..4 {
        if r > 0 {
            ss[0].restore_snapshot(&banks[0])?;
            prefill_logits.push(
                ss[0].prefill_packed(
                    &ctx,
                    &docs[prefix[r]]
                        .iter()
                        .copied()
                        .cycle()
                        .take(r * 17)
                        .collect::<Vec<_>>(),
                )?,
            );
        } else {
            ss[0].restore_snapshot(&empty)?;
            prefill_logits.push(
                ss[0].prefill_packed(
                    &ctx,
                    &docs[prefix[r]]
                        .iter()
                        .copied()
                        .cycle()
                        .take(depth)
                        .collect::<Vec<_>>(),
                )?,
            );
        }
        banks.push(ss[0].capture_snapshot()?);
    }
    drop(empty);
    for r in 0..4 {
        ss[r].restore_snapshot(&banks[r])?;
    }
    for p in 0..64 {
        batch.step(
            &ctx,
            &mut ss.iter_mut().collect::<Vec<_>>(),
            &(0..4).map(|r| token(r, p)).collect::<Vec<_>>(),
        )?;
    }
    let entries: Vec<_> = ss.iter().map(|s| s.capture_snapshot().unwrap()).collect();
    let restore = |ss: &mut [Glm5NextSession<'_>]| -> anyhow::Result<()> {
        for (s, e) in ss.iter_mut().zip(&entries) {
            s.restore_snapshot(e)?;
        }
        Ok(())
    };
    let greedy = SamplingConfig {
        temperature: 0.,
        top_k: 0,
        top_p: 1.,
        min_p: 0.,
        seed: 1,
    };
    let released = SamplingConfig::glm5_next(1);
    let mut cells = Vec::new();
    for &width in &widths {
        let mut exact = Vec::new();
        for (name, config) in [("greedy", greedy), ("released", released)] {
            restore(&mut ss[..width])?;
            let mut host: Vec<_> = (0..width).map(|_| Sampler::new(config).unwrap()).collect();
            let mut gpu: Vec<_> = (0..width).map(|_| Sampler::new(config).unwrap()).collect();
            let mut host_pending: Vec<u32> = (0..width)
                .map(|i| host[i].sample(&prefill_logits[i]).unwrap().token as u32)
                .collect();
            let initial = host_pending.clone();
            let mut host_tokens: Vec<Vec<u32>> = vec![Vec::new(); width];
            for _ in 0..steps {
                let logits = batch.step(
                    &ctx,
                    &mut ss[..width].iter_mut().collect::<Vec<_>>(),
                    &host_pending,
                )?;
                for i in 0..width {
                    let token = host[i].sample(&logits[i]).unwrap().token as u32;
                    host_tokens[i].push(token);
                }
                host_pending = (0..width)
                    .map(|i| *host_tokens[i].last().unwrap())
                    .collect();
            }
            restore(&mut ss[..width])?;
            let mut gpu_pending: Vec<u32> = (0..width)
                .map(|i| gpu[i].sample(&prefill_logits[i]).unwrap().token as u32)
                .collect();
            assert_eq!(initial, gpu_pending);
            let mut gpu_tokens: Vec<Vec<u32>> = vec![Vec::new(); width];
            let mut gpu_sampler_refs: Vec<_> = gpu.iter_mut().collect();
            let stats = batch.generate_gpu_tokens(
                &ctx,
                &mut ss[..width].iter_mut().collect::<Vec<_>>(),
                &mut gpu_sampler_refs,
                &gpu_pending,
                steps,
                &vec![&[][..]; width],
                |member, token| {
                    gpu_tokens[member].push(token);
                    true
                },
            )?;
            let mismatch: usize = (0..width)
                .map(|i| {
                    host_tokens[i]
                        .iter()
                        .zip(&gpu_tokens[i])
                        .filter(|(a, b)| a != b)
                        .count()
                })
                .sum();
            exact.push(json!({
                "preset": name,
                "token_mismatches": mismatch,
                "token_vectors": width * steps,
                "sampled": stats.sampled_tokens,
                "forwarded": stats.forwarded_tokens,
            }));
        }
        // Released-preset timing, restored, under optional host load.
        let _load = (load > 0).then(|| Load::start(load));
        let _qos = GpuThreadQos::enter();
        restore(&mut ss[..width])?;
        let mut host_samplers: Vec<_> = (0..width)
            .map(|_| Sampler::new(released).unwrap())
            .collect();
        let mut host_pending: Vec<u32> = (0..width)
            .map(|i| host_samplers[i].sample(&prefill_logits[i]).unwrap().token as u32)
            .collect();
        let mut host_tokens: Vec<Vec<u32>> = vec![Vec::new(); width];
        let mut host_rounds = Vec::new();
        for _ in 0..steps {
            let t = Instant::now();
            let logits = batch.step(
                &ctx,
                &mut ss[..width].iter_mut().collect::<Vec<_>>(),
                &host_pending,
            )?;
            for i in 0..width {
                let token = host_samplers[i].sample(&logits[i]).unwrap().token as u32;
                host_tokens[i].push(token);
            }
            host_pending = (0..width)
                .map(|i| *host_tokens[i].last().unwrap())
                .collect();
            host_rounds.push(t.elapsed().as_secs_f64() * 1e3);
        }
        restore(&mut ss[..width])?;
        let mut gpu_samplers: Vec<_> = (0..width)
            .map(|_| Sampler::new(released).unwrap())
            .collect();
        let gpu_pending: Vec<u32> = (0..width)
            .map(|i| gpu_samplers[i].sample(&prefill_logits[i]).unwrap().token as u32)
            .collect();
        let mut gpu_sampler_refs: Vec<_> = gpu_samplers.iter_mut().collect();
        let mut loop_tokens: Vec<Vec<u32>> = vec![Vec::new(); width];
        let loop_start = Instant::now();
        let loop_stats = batch.generate_gpu_tokens(
            &ctx,
            &mut ss[..width].iter_mut().collect::<Vec<_>>(),
            &mut gpu_sampler_refs,
            &gpu_pending,
            steps,
            &vec![&[][..]; width],
            |member, token| {
                loop_tokens[member].push(token);
                true
            },
        )?;
        let loop_wall = loop_start.elapsed().as_secs_f64() * 1e3;
        let mismatch: usize = (0..width)
            .map(|i| {
                host_tokens[i]
                    .iter()
                    .zip(&loop_tokens[i])
                    .filter(|(a, b)| a != b)
                    .count()
            })
            .sum();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        cells.push(json!({
            "width": width,
            "exact": exact,
            "timing": {
                "host_round_ms": mean(&host_rounds),
                "loop_round_ms": loop_wall / steps as f64,
                "loop_encode_ms": loop_stats.encode_cpu_ms / steps as f64,
                "loop_wait_ms": loop_stats.completion_wait_ms / steps as f64,
                "loop_gpu_ms": loop_stats.gpu_ms / steps as f64,
                "gain": mean(&host_rounds) / (loop_wall / steps as f64) - 1.,
                "released_token_mismatches": mismatch,
            }
        }));
        drop(_qos);
        drop(_load);
    }
    emit(json!({
        "kind": "glm_batch_loop_ab",
        "depth": depth,
        "steps": steps,
        "widths": widths,
        "warm": warm,
        "load": load,
        "cells": cells,
    }));
    Ok(())
}
