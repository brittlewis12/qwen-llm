#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Paired A/B for the non-pipelined Flash-Next batch step: CPU PLE staging
//! versus device-owned inputs. Two independent runner sets are restored from the
//! same prefills and fed identical teacher-forced tokens; the timing order
//! alternates each round and every logit vector is compared bitwise.
//!
//! MODEL CODE PROSE DEPTH [STEPS=64] [WIDTHS=2,3,4] [WARM=8]
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
    let steps: usize = a.get(5).map_or(Ok(64), |s| s.parse())?;
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
    let released = SamplingConfig::qwen38_flash_next(1);
    let mut cells = Vec::new();
    let (mut cpu_fit, mut gpu_fit) = (Vec::new(), Vec::new());
    for &width in &widths {
        // Warm both paths, then restore so the measured arms start together.
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
        for step in 0..warm {
            let tokens: Vec<u32> = (0..width).map(|i| docs[i][depth + i * 17 + step]).collect();
            let mut a_members: Vec<_> = runners_a[..width].iter_mut().collect();
            native.step_cpu_inputs(&mut a_members, &tokens)?;
            let mut b_members: Vec<_> = runners_b[..width].iter_mut().collect();
            native.step_gpu_inputs(&mut b_members, &tokens)?;
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
        let mut samplers: Vec<_> = (0..width)
            .map(|_| Sampler::new(released).unwrap())
            .collect();
        let (mut cpu_ms, mut gpu_ms) = (Vec::new(), Vec::new());
        let (mut cpu_diag, mut gpu_diag) = ([0f64; 3], [0f64; 3]);
        let (mut sample_ms, mut max_abs, mut nonbitwise) = (Vec::new(), 0f64, 0usize);
        for step in 0..steps {
            let tokens: Vec<u32> = (0..width).map(|i| docs[i][depth + i * 17 + step]).collect();
            let (cpu_values, gpu_values);
            if step % 2 == 0 {
                let t = Instant::now();
                let mut members: Vec<_> = runners_a[..width].iter_mut().collect();
                cpu_values = native.step_cpu_inputs(&mut members, &tokens)?;
                cpu_ms.push(t.elapsed().as_secs_f64() * 1e3);
                for (i, v) in native.diagnostic_ms.iter().enumerate() {
                    cpu_diag[i] += v;
                }
                let t = Instant::now();
                let mut members: Vec<_> = runners_b[..width].iter_mut().collect();
                gpu_values = native.step_gpu_inputs(&mut members, &tokens)?;
                gpu_ms.push(t.elapsed().as_secs_f64() * 1e3);
                for (i, v) in native.diagnostic_ms.iter().enumerate() {
                    gpu_diag[i] += v;
                }
            } else {
                let t = Instant::now();
                let mut members: Vec<_> = runners_b[..width].iter_mut().collect();
                gpu_values = native.step_gpu_inputs(&mut members, &tokens)?;
                gpu_ms.push(t.elapsed().as_secs_f64() * 1e3);
                for (i, v) in native.diagnostic_ms.iter().enumerate() {
                    gpu_diag[i] += v;
                }
                let t = Instant::now();
                let mut members: Vec<_> = runners_a[..width].iter_mut().collect();
                cpu_values = native.step_cpu_inputs(&mut members, &tokens)?;
                cpu_ms.push(t.elapsed().as_secs_f64() * 1e3);
                for (i, v) in native.diagnostic_ms.iter().enumerate() {
                    cpu_diag[i] += v;
                }
            }
            for i in 0..width {
                assert_eq!(cpu_values[i].len(), gpu_values[i].len());
                if bytemuck::cast_slice::<f32, u8>(&cpu_values[i])
                    != bytemuck::cast_slice::<f32, u8>(&gpu_values[i])
                {
                    nonbitwise += 1;
                }
                for (&x, &y) in cpu_values[i].iter().zip(&gpu_values[i]) {
                    max_abs = max_abs.max((x as f64 - y as f64).abs());
                }
            }
            let t = Instant::now();
            for (i, sampler) in samplers.iter_mut().enumerate() {
                let _ = sampler.sample(&cpu_values[i])?;
            }
            sample_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let cpu_round = mean(&cpu_ms);
        let gpu_round = mean(&gpu_ms);
        cpu_fit.push((width as f64, cpu_round));
        gpu_fit.push((width as f64, gpu_round));
        cells.push(serde_json::json!({
            "width": width,
            "cpu_round_ms": cpu_round,
            "gpu_round_ms": gpu_round,
            "gain": cpu_round / gpu_round - 1.,
            "sample_ms": mean(&sample_ms),
            "cpu_diag_ms": [cpu_diag[0] / steps as f64, cpu_diag[1] / steps as f64, cpu_diag[2] / steps as f64],
            "gpu_diag_ms": [gpu_diag[0] / steps as f64, gpu_diag[1] / steps as f64, gpu_diag[2] / steps as f64],
            "commit_readback_ms": (gpu_diag[2] - gpu_diag[1] - gpu_diag[0]) / steps as f64,
            "max_abs_logit_diff": max_abs,
            "nonbitwise_logit_vectors": nonbitwise,
        }));
    }
    let cpu = fit(&cpu_fit);
    let gpu = fit(&gpu_fit);
    println!(
        "{}",
        serde_json::json!({
            "kind": "fn_batch_gpu_inputs_ab",
            "depth": depth,
            "steps": steps,
            "warm": warm,
            "cells": cells,
            "fit_cpu": cpu.map(|(f, m)| serde_json::json!({"F_ms": f, "m_ms_per_row": m})),
            "fit_gpu": gpu.map(|(f, m)| serde_json::json!({"F_ms": f, "m_ms_per_row": m})),
        })
    );
    Ok(())
}
