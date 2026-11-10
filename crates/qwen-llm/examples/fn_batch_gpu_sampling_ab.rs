#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Paired A/B for non-pipelined Flash-Next GPU sampling. Two independent runner
//! sets are restored from the same prefills and fed identical teacher-forced
//! tokens. The host arm forwards, reads full logits and samples on the host; the
//! GPU arm forwards, samples every member in the same command and reads back
//! only tokens. Greedy is checked bitwise; the released preset reports token
//! mismatches (the B=1 token-feedback equivalence is pinned separately).
//!
//! MODEL CODE PROSE DEPTH [STEPS=48] [WIDTHS=2,3,4] [WARM=8]
//! Requires `--features perf-ablations`; build with `--profile iter` for A/B.
use qwen_llm::qwen4exp_text_session::Qwen4ExpTextSnapshot;
use qwen_llm::{
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{
        Qwen4ExpDecodeBatch, Qwen4ExpDecodeOptions, Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity,
        Qwen4ExpTextRunner,
    },
    sampling::{Sampler, SamplingConfig},
    tokenizer::Tokenizer,
};
use std::{fs, time::Instant};

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

fn fit(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    let n = points.len() as f64;
    if points.len() < 2 {
        return None;
    }
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let denom = n * sxx - sx * sx;
    if denom == 0. {
        return None;
    }
    let m = (n * sxy - sx * sy) / denom;
    Some(((sy - m * sx) / n, m))
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.
    } else {
        v.iter().sum::<f64>() / v.len() as f64
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
    assert!(steps > 1 && !widths.is_empty());
    let max_width = *widths.iter().max().unwrap();
    assert!(widths.iter().all(|w| (2..=4).contains(w)));
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
    let mut runners_a = vec![first];
    for _ in 1..max_width {
        runners_a.push(factory.create_runner(&ctx)?);
    }
    let mut runners_b = Vec::with_capacity(max_width);
    for _ in 0..max_width {
        runners_b.push(factory.create_runner(&ctx)?);
    }
    let mut snapshots = Vec::new();
    let mut prefill_logits = Vec::new();
    for (i, r) in runners_a.iter_mut().enumerate() {
        let logits = r.prefill(&docs[i][..depth + i * 17])?;
        assert_eq!(logits.as_slice().len(), c.vocab_size as usize);
        prefill_logits.push(logits.as_slice().to_vec());
        snapshots.push(r.capture_snapshot()?);
    }
    let mut native = Qwen4ExpDecodeBatch::new(&runners_a[0], max_width)?;
    let configs = [
        (
            "greedy",
            SamplingConfig {
                temperature: 0.,
                top_k: 0,
                top_p: 1.,
                min_p: 0.,
                seed: 1,
            },
        ),
        ("released", SamplingConfig::qwen38_flash_next(1)),
    ];
    let mut phases = Vec::new();
    for (name, config) in configs {
        let mut cells = Vec::new();
        let (mut host_fit, mut gpu_fit) = (Vec::new(), Vec::new());
        for &width in &widths {
            restore(
                &mut runners_a[..width],
                &snapshots[..width],
                &prefill_logits[..width],
            );
            restore(
                &mut runners_b[..width],
                &snapshots[..width],
                &prefill_logits[..width],
            );
            let mut samplers_a: Vec<_> =
                (0..width).map(|_| Sampler::new(config).unwrap()).collect();
            let mut samplers_b: Vec<_> =
                (0..width).map(|_| Sampler::new(config).unwrap()).collect();
            for step in 0..warm {
                let tokens: Vec<u32> = (0..width).map(|i| docs[i][depth + i * 17 + step]).collect();
                let mut members: Vec<_> = runners_a[..width].iter_mut().collect();
                let logits = native.step(&mut members, &tokens)?;
                for (i, sampler) in samplers_a.iter_mut().enumerate() {
                    let _ = sampler.sample(&logits[i])?;
                }
                let mut members: Vec<_> = runners_b[..width].iter_mut().collect();
                let mut refs: Vec<_> = samplers_b.iter_mut().collect();
                native.step_tokens(&mut members, &mut refs, &tokens)?;
            }
            restore(
                &mut runners_a[..width],
                &snapshots[..width],
                &prefill_logits[..width],
            );
            restore(
                &mut runners_b[..width],
                &snapshots[..width],
                &prefill_logits[..width],
            );
            for sampler in samplers_a.iter_mut().chain(samplers_b.iter_mut()) {
                *sampler = Sampler::new(config).unwrap();
            }
            let (mut host_ms, mut gpu_ms, mut sample_ms) = (Vec::new(), Vec::new(), Vec::new());
            let (mut host_diag, mut gpu_diag) = ([0f64; 3], [0f64; 3]);
            let mut host_tokens: Vec<Vec<u32>> = vec![Vec::new(); width];
            let mut gpu_tokens: Vec<Vec<u32>> = vec![Vec::new(); width];
            for step in 0..steps {
                let tokens: Vec<u32> = (0..width).map(|i| docs[i][depth + i * 17 + step]).collect();
                let t = Instant::now();
                let mut members: Vec<_> = runners_a[..width].iter_mut().collect();
                let logits = native.step(&mut members, &tokens)?;
                host_ms.push(t.elapsed().as_secs_f64() * 1e3);
                for (i, v) in native.diagnostic_ms.iter().enumerate() {
                    host_diag[i] += v;
                }
                let t = Instant::now();
                for (i, sampler) in samplers_a.iter_mut().enumerate() {
                    host_tokens[i].push(sampler.sample(&logits[i])?.token as u32);
                }
                sample_ms.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let mut members: Vec<_> = runners_b[..width].iter_mut().collect();
                let mut refs: Vec<_> = samplers_b.iter_mut().collect();
                let sampled = native.step_tokens(&mut members, &mut refs, &tokens)?;
                gpu_ms.push(t.elapsed().as_secs_f64() * 1e3);
                for (i, v) in native.diagnostic_ms.iter().enumerate() {
                    gpu_diag[i] += v;
                }
                for (i, token) in sampled.into_iter().enumerate() {
                    gpu_tokens[i].push(token);
                }
            }
            let mismatches: usize = (0..width)
                .map(|i| {
                    host_tokens[i]
                        .iter()
                        .zip(&gpu_tokens[i])
                        .filter(|(a, b)| a != b)
                        .count()
                })
                .sum();
            let host_round = mean(&host_ms) + mean(&sample_ms);
            let gpu_round = mean(&gpu_ms);
            host_fit.push((width as f64, host_round));
            gpu_fit.push((width as f64, gpu_round));
            cells.push(serde_json::json!({
                "width": width,
                "host_round_ms": host_round,
                "gpu_round_ms": gpu_round,
                "gain": host_round / gpu_round - 1.,
                "step_ms": mean(&host_ms),
                "sample_ms": mean(&sample_ms),
                "host_diag_ms": [host_diag[0] / steps as f64, host_diag[1] / steps as f64, host_diag[2] / steps as f64],
                "gpu_diag_ms": [gpu_diag[0] / steps as f64, gpu_diag[1] / steps as f64, gpu_diag[2] / steps as f64],
                "gpu_commit_readback_ms": (gpu_diag[2] - gpu_diag[1] - gpu_diag[0]) / steps as f64,
                "token_mismatches": mismatches,
                "token_vectors": width * steps,
            }));
        }
        phases.push(serde_json::json!({
            "preset": name,
            "cells": cells,
            "fit_host": fit(&host_fit).map(|(f, m)| serde_json::json!({"F_ms": f, "m_ms_per_row": m})),
            "fit_gpu": fit(&gpu_fit).map(|(f, m)| serde_json::json!({"F_ms": f, "m_ms_per_row": m})),
        }));
    }
    println!(
        "{}",
        serde_json::json!({
            "kind": "fn_batch_gpu_sampling_ab",
            "depth": depth,
            "steps": steps,
            "warm": warm,
            "phases": phases,
        })
    );
    Ok(())
}
