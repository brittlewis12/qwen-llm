#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Paired A/B for V4.1 non-pipelined batched GPU sampling. Two independent
//! state sets are restored from the same snapshot and fed identical
//! teacher-forced tokens. Three arms share one forward:
//! - host: `step_batch` reads full logits and samples on the host;
//! - fast: `step_batch_sampled` with `Fast` members samples every member
//!   in-command with the fast Gumbel-max sampler and reads only tokens;
//! - exact: `step_batch_sampled` with `Exact` members runs the exact GPU
//!   sampler, which must match the host's tokens at the same seed.
//! The released preset reports fast mismatches (both arms are valid draws) and
//! exact mismatches (must be zero). Greedy is checked bitwise for the exact arm.
//!
//! MODEL DEPTH WIDTH [STEPS=32] [WARM=16]
//! Requires `--features perf-ablations`; build with `--profile iter` for A/B.
use qwen_llm::{
    deepseek_v41::{
        forward::{BatchMemberOutput, BatchSampleMode, Forward, Snapshot, State},
        gguf::{ResidentWeights, resident_storage_plan_with_target},
    },
    gguf::GgufFile,
    metal::{self, GpuTokenSampler, MetalContext},
    sampling::{Sampler, SamplingConfig},
    tokenizer::NativeTokenizer,
};
use std::{sync::Arc, time::Instant};

#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires --features perf-ablations");
}

fn corpus() -> String {
    let mut text = String::new();
    for piece in [
        include_str!("../src/deepseek_v41/forward/gpu_step.rs"),
        include_str!("../src/deepseek_v41/forward/attention_gpu.rs"),
        include_str!("../src/deepseek_v41/forward/ffn_gpu.rs"),
    ] {
        text.push_str(piece);
        text.push_str("\n\n");
    }
    text
}

fn anyhow_msg(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::Error::msg(e.to_string())
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

#[cfg(feature = "perf-ablations")]
fn main() -> anyhow::Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let depth: usize = a[2].parse()?;
    let width: usize = a[3].parse()?;
    let steps: usize = a.get(4).map_or(Ok(32), |s| s.parse())?;
    let warm: usize = a.get(5).map_or(Ok(16), |s| s.parse())?;
    assert!([32768, 49152, 65536].contains(&depth));
    assert!((2..=4).contains(&width) && steps > 0);
    let g = Arc::new(GgufFile::open(&a[1])?);
    let ctx = MetalContext::new()?;
    let storage = resident_storage_plan_with_target(
        &g,
        metal::host_page_size_bytes()?,
        ctx.max_buffer_length(),
        Some(4_294_967_296),
    )?;
    let w = ResidentWeights::load_with_storage_plan(&ctx, g.clone(), &storage)?;
    let f = Forward::from_resident(&ctx, &w, w.token_map.clone(), w.raw_pad).map_err(anyhow_msg)?;
    let tok = NativeTokenizer::from_gguf(&g)?;
    let mut text = corpus();
    let mut ids = tok.encode(&text, false)?;
    while ids.len() < depth + 4096 + steps + warm + 5000 * width {
        text.push_str(&text.clone());
        ids = tok.encode(&text, false)?;
    }
    let common: Vec<u32> = ids.into_iter().map(|v| v as u32).collect();
    let streams: Vec<Vec<u32>> = (0..width)
        .map(|i| common[1000 + i * 5000..].to_vec())
        .collect();
    let mut base = f.new_state(98304).map_err(anyhow_msg)?;
    let vocab = f
        .prefill(&mut base, &common[..depth - 4096], None)
        .map_err(anyhow_msg)?
        .len();
    let snap = f.capture_snapshot(&base).map_err(anyhow_msg)?;
    let make_states = |f: &Forward<'_>| -> anyhow::Result<Vec<State>> {
        let mut out = Vec::new();
        for i in 0..width {
            let mut state = f.new_state(98304).map_err(anyhow_msg)?;
            f.restore_snapshot(&mut state, &snap).map_err(anyhow_msg)?;
            f.prefill(&mut state, &streams[i][..4096 + i * 17], None)
                .map_err(anyhow_msg)?;
            out.push(state);
        }
        Ok(out)
    };
    let mut host_states = make_states(&f)?;
    let mut gpu_states = make_states(&f)?;
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
            false,
        ),
        ("released", SamplingConfig::deepseek_v4_0731(1), true),
    ];
    let mut phases = Vec::new();
    for (name, config, fast_enabled) in configs {
        let mut host_samplers: Vec<_> = (0..width)
            .map(|_| Sampler::new(config))
            .collect::<Result<_, _>>()?;
        let mut gpu_samplers: Vec<_> = (0..width)
            .map(|_| Sampler::new(config))
            .collect::<Result<_, _>>()?;
        let mut token_samplers = Vec::new();
        for _ in 0..width {
            token_samplers.push(GpuTokenSampler::new(&ctx, vocab, config)?);
        }
        let entries: Vec<Snapshot> = host_states
            .iter()
            .map(|s| f.capture_snapshot(s).map_err(anyhow_msg))
            .collect::<Result<_, _>>()?;
        // Warm both arms so the timed loops measure steady state.
        for step in 0..warm {
            let tokens: Vec<u32> = (0..width)
                .map(|i| streams[i][4096 + i * 17 + step])
                .collect();
            let mut members: Vec<_> = host_states.iter_mut().collect();
            let logits = f
                .step_batch(&mut members, &tokens, None)
                .map_err(anyhow_msg)?;
            for (i, sampler) in host_samplers.iter_mut().enumerate() {
                let _ = sampler.sample(&logits[i])?;
            }
            let modes: Vec<_> = token_samplers
                .iter()
                .map(|sampler| BatchSampleMode::Exact { sampler })
                .collect();
            let mut members: Vec<_> = gpu_states.iter_mut().collect();
            let mut refs: Vec<&mut Sampler> = gpu_samplers.iter_mut().collect();
            f.step_batch_sampled(&mut members, &tokens, &mut refs, &modes, None)
                .map_err(anyhow_msg)?;
        }
        for (state, entry) in host_states.iter_mut().zip(&entries) {
            f.restore_snapshot(state, entry).map_err(anyhow_msg)?;
        }
        for (state, entry) in gpu_states.iter_mut().zip(&entries) {
            f.restore_snapshot(state, entry).map_err(anyhow_msg)?;
        }
        for sampler in host_samplers.iter_mut().chain(gpu_samplers.iter_mut()) {
            *sampler = Sampler::new(config)?;
        }
        let inputs: Vec<Vec<u32>> = (0..steps)
            .map(|step| {
                (0..width)
                    .map(|i| streams[i][4096 + i * 17 + warm + step])
                    .collect()
            })
            .collect();

        // Host arm: full logits plus host sampling.
        let mut host_ms = Vec::new();
        let mut sample_ms = Vec::new();
        let mut host_tokens: Vec<Vec<u32>> = vec![Vec::new(); width];
        for tokens in &inputs {
            let t = Instant::now();
            let mut members: Vec<_> = host_states.iter_mut().collect();
            let logits = f
                .step_batch(&mut members, tokens, None)
                .map_err(anyhow_msg)?;
            host_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            for (i, sampler) in host_samplers.iter_mut().enumerate() {
                host_tokens[i].push(sampler.sample(&logits[i])?.token as u32);
            }
            sample_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let host_round = mean(&host_ms) + mean(&sample_ms);
        for (state, entry) in gpu_states.iter_mut().zip(&entries) {
            f.restore_snapshot(state, entry).map_err(anyhow_msg)?;
        }
        for sampler in gpu_samplers.iter_mut() {
            *sampler = Sampler::new(config)?;
        }

        // Exact GPU arm: must match the host tokens.
        let mut exact_ms = Vec::new();
        let mut exact_tokens: Vec<Vec<u32>> = vec![Vec::new(); width];
        let exact_modes: Vec<_> = token_samplers
            .iter()
            .map(|sampler| BatchSampleMode::Exact { sampler })
            .collect();
        for tokens in &inputs {
            let t = Instant::now();
            let mut members: Vec<_> = gpu_states.iter_mut().collect();
            let mut refs: Vec<&mut Sampler> = gpu_samplers.iter_mut().collect();
            let out = f
                .step_batch_sampled(&mut members, tokens, &mut refs, &exact_modes, None)
                .map_err(anyhow_msg)?;
            exact_ms.push(t.elapsed().as_secs_f64() * 1e3);
            for (i, out) in out.into_iter().enumerate() {
                match out {
                    BatchMemberOutput::Token(t) => exact_tokens[i].push(t),
                    BatchMemberOutput::Logits(_) => unreachable!("exact members sample in-command"),
                }
            }
        }
        for (state, entry) in gpu_states.iter_mut().zip(&entries) {
            f.restore_snapshot(state, entry).map_err(anyhow_msg)?;
        }
        for sampler in gpu_samplers.iter_mut() {
            *sampler = Sampler::new(config)?;
        }

        // Fast GPU arm: a valid draw, not token-identical with the host. It
        // only exists for a positive temperature, so greedy skips it.
        let count_mismatches = |a: &[Vec<u32>], b: &[Vec<u32>]| -> usize {
            (0..width)
                .map(|i| a[i].iter().zip(&b[i]).filter(|(x, y)| x != y).count())
                .sum()
        };
        let (fast_round, fast_mismatches) = if fast_enabled {
            let mut fast_ms = Vec::new();
            let mut fast_tokens: Vec<Vec<u32>> = vec![Vec::new(); width];
            let fast_modes: Vec<_> = (0..width)
                .map(|_| BatchSampleMode::Fast {
                    temperature: config.temperature,
                })
                .collect();
            for tokens in &inputs {
                let t = Instant::now();
                let mut members: Vec<_> = gpu_states.iter_mut().collect();
                let mut refs: Vec<&mut Sampler> = gpu_samplers.iter_mut().collect();
                let out = f
                    .step_batch_sampled(&mut members, tokens, &mut refs, &fast_modes, None)
                    .map_err(anyhow_msg)?;
                fast_ms.push(t.elapsed().as_secs_f64() * 1e3);
                for (i, out) in out.into_iter().enumerate() {
                    match out {
                        BatchMemberOutput::Token(t) => fast_tokens[i].push(t),
                        BatchMemberOutput::Logits(_) => {
                            unreachable!("fast members sample in-command")
                        }
                    }
                }
            }
            (
                Some(mean(&fast_ms)),
                Some(count_mismatches(&host_tokens, &fast_tokens)),
            )
        } else {
            (None, None)
        };
        let exact_mismatches = count_mismatches(&host_tokens, &exact_tokens);
        phases.push(serde_json::json!({
            "preset": name,
            "width": width,
            "host_round_ms": host_round,
            "exact_round_ms": mean(&exact_ms),
            "exact_gain": host_round / mean(&exact_ms) - 1.,
            "fast_round_ms": fast_round,
            "fast_gain": fast_round.map(|fast| host_round / fast - 1.),
            "step_ms": mean(&host_ms),
            "sample_ms": mean(&sample_ms),
            "exact_mismatches": exact_mismatches,
            "fast_mismatches": fast_mismatches,
            "token_vectors": width * steps,
        }));
    }
    println!(
        "{}",
        serde_json::json!({
            "kind": "v41_batch_gpu_sampling_ab",
            "depth": depth,
            "width": width,
            "steps": steps,
            "warm": warm,
            "phases": phases,
        })
    );
    Ok(())
}
