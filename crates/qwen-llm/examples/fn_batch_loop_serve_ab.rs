#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Paired native A/B for the shipped Flash-Next serve path: serial GPU-sampled
//! `step_tokens` rounds versus the one-ahead device-feedback loop.
//!
//! Exactness is checked as before: greedy loop logits must be bitwise equal to
//! the host-staged forward, and the released preset must select the same tokens.
//! Timing restores the same prefill for each arm; optional host load saturates
//! the CPU outside the QoS-raised loop thread, and the QoS toggle wraps each
//! measured arm in `GpuThreadQos`.
//!
//! MODEL CODE PROSE DEPTH [STEPS=48] [WIDTHS=2,3,4] [WARM=8] [LOAD=0] [QOS=1]
//! Requires `--features perf-ablations`; build with `--profile iter`.
use qwen_llm::qwen4exp_text_session::Qwen4ExpTextSnapshot;
use qwen_llm::{
    gguf::GgufFile,
    metal::{GpuThreadQos, MetalContext},
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{
        Qwen4ExpDecodeBatch, Qwen4ExpDecodeOptions, Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity,
        Qwen4ExpTextRunner,
    },
    sampling::{Sampler, SamplingConfig},
    tokenizer::Tokenizer,
};
use std::{
    fs,
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

fn restore(
    runners: &mut [Qwen4ExpTextRunner<'_, '_, '_>],
    snapshots: &[Qwen4ExpTextSnapshot],
    logits: &[Vec<f32>],
) {
    for (r, (snap, logits)) in runners.iter_mut().zip(snapshots.iter().zip(logits)) {
        r.restore_snapshot_with_logits(snap, logits).unwrap();
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
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

struct LoopArm {
    tokens: Vec<Vec<u32>>,
    logits: Vec<Vec<Vec<f32>>>,
    wall_ms: f64,
    gpu_ms: f64,
    encode_ms: f64,
    wait_ms: f64,
}

struct SerialArm {
    tokens: Vec<Vec<u32>>,
    round_ms: Vec<f64>,
    gpu_ms: f64,
    encode_ms: f64,
    readback_ms: f64,
    wall_ms: f64,
}

#[cfg(feature = "perf-ablations")]
fn run_serial(
    native: &mut Qwen4ExpDecodeBatch,
    runners: &mut [Qwen4ExpTextRunner<'_, '_, '_>],
    prefill_logits: &[Vec<f32>],
    config: SamplingConfig,
    steps: usize,
    qos: bool,
) -> SerialArm {
    let width = runners.len();
    let mut samplers: Vec<_> = (0..width).map(|_| Sampler::new(config).unwrap()).collect();
    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); width];
    let mut next: Vec<u32> = prefill_logits
        .iter()
        .zip(samplers.iter_mut())
        .map(|(l, s)| s.sample(l).unwrap().token as u32)
        .collect();
    for (i, token) in next.iter().enumerate() {
        tokens[i].push(*token);
    }
    let _qos = GpuThreadQos::enter_if(qos);
    let started = Instant::now();
    let (mut gpu, mut encode, mut wall) = (0., 0., 0.);
    let mut round_ms = Vec::with_capacity(steps);
    for _ in 0..steps {
        let t = Instant::now();
        let mut members: Vec<_> = runners.iter_mut().collect();
        let mut refs: Vec<_> = samplers.iter_mut().collect();
        next = native.step_tokens(&mut members, &mut refs, &next).unwrap();
        round_ms.push(t.elapsed().as_secs_f64() * 1e3);
        gpu += native.diagnostic_ms[1];
        encode += native.diagnostic_ms[0];
        wall += native.diagnostic_ms[2];
        for (i, token) in next.iter().enumerate() {
            tokens[i].push(*token);
        }
    }
    SerialArm {
        tokens,
        round_ms,
        gpu_ms: gpu / steps as f64,
        encode_ms: encode / steps as f64,
        readback_ms: (wall - gpu - encode) / steps as f64,
        wall_ms: started.elapsed().as_secs_f64() * 1e3,
    }
}

#[cfg(feature = "perf-ablations")]
fn run_loop(
    native: &mut Qwen4ExpDecodeBatch,
    runners: &mut [Qwen4ExpTextRunner<'_, '_, '_>],
    config: SamplingConfig,
    steps: usize,
    qos: bool,
) -> LoopArm {
    let width = runners.len();
    let mut samplers: Vec<_> = (0..width).map(|_| Sampler::new(config).unwrap()).collect();
    let mut members: Vec<_> = runners.iter_mut().collect();
    let mut refs: Vec<_> = samplers.iter_mut().collect();
    let _qos = GpuThreadQos::enter_if(qos);
    let started = Instant::now();
    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); width];
    let mut per_member: Vec<Vec<Vec<f32>>> = vec![Vec::new(); width];
    let stats = native
        .generate_gpu_tokens(
            &mut members,
            &mut refs,
            &vec![steps; width],
            &vec![&[][..]; width],
            |member, token, logits| {
                tokens[member].push(token as u32);
                per_member[member].push(logits.to_vec());
                true
            },
        )
        .unwrap();
    let mut by_step = vec![vec![Vec::new(); width]; steps];
    for (member, rows) in per_member.into_iter().enumerate() {
        for (step, row) in rows.into_iter().enumerate() {
            by_step[step][member] = row;
        }
    }
    LoopArm {
        tokens,
        logits: by_step,
        wall_ms: started.elapsed().as_secs_f64() * 1e3,
        gpu_ms: stats.gpu_ms,
        encode_ms: stats.encode_cpu_ms,
        wait_ms: stats.completion_wait_ms,
    }
}

#[cfg(feature = "perf-ablations")]
fn run_host(
    native: &mut Qwen4ExpDecodeBatch,
    runners: &mut [Qwen4ExpTextRunner<'_, '_, '_>],
    prefill_logits: &[Vec<f32>],
    config: SamplingConfig,
    steps: usize,
) -> LoopArm {
    let width = runners.len();
    let mut samplers: Vec<_> = (0..width).map(|_| Sampler::new(config).unwrap()).collect();
    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); width];
    let mut history: Vec<Vec<Vec<f32>>> = Vec::with_capacity(steps);
    let mut logits = prefill_logits.to_vec();
    let started = Instant::now();
    for _ in 0..steps {
        history.push(logits.clone());
        let next: Vec<u32> = logits
            .iter()
            .zip(samplers.iter_mut())
            .map(|(l, s)| s.sample(l).unwrap().token as u32)
            .collect();
        for (i, token) in next.iter().enumerate() {
            tokens[i].push(*token);
        }
        let mut members: Vec<_> = runners.iter_mut().collect();
        logits = native.step(&mut members, &next).unwrap();
    }
    LoopArm {
        tokens,
        logits: history,
        wall_ms: started.elapsed().as_secs_f64() * 1e3,
        gpu_ms: 0.,
        encode_ms: 0.,
        wait_ms: 0.,
    }
}

#[cfg(feature = "perf-ablations")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let depth: usize = a[4].parse()?;
    let steps: usize = a.get(5).map_or(Ok(48), |s| s.parse())?;
    let widths: Vec<usize> = a
        .get(6)
        .map(|s| s.split(',').map(str::parse).collect())
        .transpose()?
        .unwrap_or(vec![2, 3, 4]);
    let warm: usize = a.get(7).map_or(Ok(8), |s| s.parse())?;
    let load: usize = a.get(8).map_or(Ok(0), |s| s.parse())?;
    let qos = a.get(9).map_or(true, |s| s != "0");
    assert!(steps > 1 && !widths.is_empty() && widths.iter().all(|w| (2..=4).contains(w)));
    let max_width = *widths.iter().max().unwrap();
    let ctx = MetalContext::new()?;
    let gg = GgufFile::open(&a[1])?;
    let c = Qwen4ExpConfig::from_gguf(&gg)?;
    let tok = Tokenizer::from_gguf(&gg)?;
    let texts = [fs::read_to_string(&a[2])?, fs::read_to_string(&a[3])?];
    let mut docs = Vec::new();
    for i in 0..max_width {
        let ids = tok.encode(&texts[i % 2], false)?;
        assert!(!ids.is_empty());
        docs.push(
            (0..depth + 512 + steps + warm)
                .map(|j| ids[(j + i * 157) % ids.len()] as u32)
                .collect::<Vec<_>>(),
        );
    }
    let extent = depth + 4096 + steps + warm;
    let mut loaded = Qwen4ExpLoadedModel::load_with_adaptive_prefill(
        &ctx,
        &gg,
        Qwen4ExpSessionCapacity::for_forward_limit(&c, extent)?,
        extent,
        Qwen4ExpDecodeOptions::default(),
    )?;
    let (factory, first) = loaded.session_factory(&ctx)?;
    let mut runners = vec![first];
    for _ in 1..max_width {
        runners.push(factory.create_runner(&ctx)?);
    }
    let mut snapshots = Vec::new();
    let mut prefill_logits = Vec::new();
    for (i, r) in runners.iter_mut().enumerate() {
        let logits = r.prefill(&docs[i][..depth + i * 17])?;
        assert_eq!(logits.as_slice().len(), c.vocab_size as usize);
        prefill_logits.push(logits.as_slice().to_vec());
        snapshots.push(r.capture_snapshot()?);
    }
    let mut native = Qwen4ExpDecodeBatch::new(&runners[0], max_width)?;
    let released = SamplingConfig::qwen38_flash_next(1);
    let greedy = SamplingConfig {
        temperature: 0.,
        top_k: 0,
        top_p: 1.,
        min_p: 0.,
        seed: 1,
    };
    for r in runners.iter_mut() {
        r.configure_gpu_token_loop(true);
    }

    // Warm both paths on restored state.
    restore(&mut runners, &snapshots, &prefill_logits);
    let _ = run_serial(
        &mut native,
        &mut runners,
        &prefill_logits,
        released,
        warm,
        qos,
    );
    restore(&mut runners, &snapshots, &prefill_logits);
    let _ = run_loop(&mut native, &mut runners, released, warm, qos);

    let mut cells = Vec::new();
    for &width in &widths {
        // Greedy exactness: host-staged forward versus the device-feedback loop.
        restore(
            &mut runners[..width],
            &snapshots[..width],
            &prefill_logits[..width],
        );
        let host = run_host(
            &mut native,
            &mut runners[..width],
            &prefill_logits[..width],
            greedy,
            steps,
        );
        restore(
            &mut runners[..width],
            &snapshots[..width],
            &prefill_logits[..width],
        );
        let loop_greedy = run_loop(&mut native, &mut runners[..width], greedy, steps, qos);
        let mut token_mismatch = Vec::new();
        let mut nonbitwise = 0usize;
        let mut max_abs = 0f64;
        for i in 0..width {
            if host.tokens[i] != loop_greedy.tokens[i] {
                token_mismatch.push(i);
            }
            for step in 0..steps {
                let (h, g) = (&host.logits[step][i], &loop_greedy.logits[step][i]);
                if bytemuck::cast_slice::<f32, u8>(h) != bytemuck::cast_slice::<f32, u8>(g) {
                    nonbitwise += 1;
                }
                for (&x, &y) in h.iter().zip(g) {
                    max_abs = max_abs.max((x as f64 - y as f64).abs());
                }
            }
        }

        // Released exactness between the host arm and the loop.
        restore(
            &mut runners[..width],
            &snapshots[..width],
            &prefill_logits[..width],
        );
        let host_released = run_host(
            &mut native,
            &mut runners[..width],
            &prefill_logits[..width],
            released,
            steps,
        );
        restore(
            &mut runners[..width],
            &snapshots[..width],
            &prefill_logits[..width],
        );
        let loop_released = run_loop(&mut native, &mut runners[..width], released, steps, qos);
        let released_mismatch: usize = (0..width)
            .map(|i| {
                host_released.tokens[i]
                    .iter()
                    .zip(&loop_released.tokens[i])
                    .filter(|(a, b)| a != b)
                    .count()
            })
            .sum();

        // Released throughput under the requested host load.
        let serial = {
            restore(
                &mut runners[..width],
                &snapshots[..width],
                &prefill_logits[..width],
            );
            let _load = (load > 0).then(|| Load::start(load));
            run_serial(
                &mut native,
                &mut runners[..width],
                &prefill_logits[..width],
                released,
                steps,
                qos,
            )
        };
        let looping = {
            restore(
                &mut runners[..width],
                &snapshots[..width],
                &prefill_logits[..width],
            );
            let _load = (load > 0).then(|| Load::start(load));
            run_loop(&mut native, &mut runners[..width], released, steps, qos)
        };
        let serial_round = mean(&serial.round_ms);
        let loop_round = looping.wall_ms / steps as f64;
        let emitted = width * steps;
        let rate = |round: f64| emitted as f64 / (round * steps as f64 / 1e3);
        cells.push(serde_json::json!({
            "width": width,
            "greedy_exact": {
                "token_mismatch_members": token_mismatch,
                "nonbitwise_logit_vectors": nonbitwise,
                "max_abs_logit_diff": max_abs,
            },
            "released_exact": { "token_mismatches": released_mismatch },
            "serial": {
                "round_ms": serial_round,
                "median_round_ms": median(serial.round_ms.clone()),
                "gpu_ms": serial.gpu_ms,
                "encode_ms": serial.encode_ms,
                "readback_ms": serial.readback_ms,
                "tokens_per_s": rate(serial_round),
            },
            "loop": {
                "round_ms": loop_round,
                "gpu_ms": looping.gpu_ms / steps as f64,
                "encode_ms": looping.encode_ms / steps as f64,
                "wait_ms": looping.wait_ms / steps as f64,
                "tokens_per_s": rate(loop_round),
            },
            "gain": rate(loop_round) / rate(serial_round) - 1.,
        }));
    }
    println!(
        "{}",
        serde_json::json!({
            "kind": "fn_batch_loop_serve_ab",
            "depth": depth,
            "steps": steps,
            "widths": widths,
            "warm": warm,
            "load": load,
            "qos": qos,
            "cells": cells,
        })
    );
    Ok(())
}
