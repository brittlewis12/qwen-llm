#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Price the host bubble of the batched independent-decode round.
//!
//! This mirrors the serve steady state: tokens selected on the host from the
//! previous round's logits, one native batch forward, full-vocabulary readback.
//! It reports the forward's encode/GPU/wall split (`Qwen4ExpDecodeBatch`
//! diagnostics), the host sampling cost, and the round time minus GPU busy, which
//! is the removable critical path for a pre-encoded, GPU-sampled token loop.
//!
//! MODEL CODE PROSE DEPTH [STEPS=64] [WIDTH=3]
//! Requires `--features perf-ablations`.
use qwen_llm::{
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{
        Qwen4ExpDecodeBatch, Qwen4ExpDecodeOptions, Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity,
    },
    sampling::{Sampler, SamplingConfig},
    tokenizer::Tokenizer,
};
use std::{fs, time::Instant};

#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires --features perf-ablations");
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n == 0 {
        return 0.;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
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
    let width: usize = a.get(6).map_or(Ok(3), |s| s.parse())?;
    assert!(steps > 8, "need eight warm rounds plus retained rounds");
    assert!((2..=4).contains(&width), "price the batched path, B=2..4");
    let ctx = MetalContext::new()?;
    let gg = GgufFile::open(&a[1])?;
    let c = Qwen4ExpConfig::from_gguf(&gg)?;
    let tok = Tokenizer::from_gguf(&gg)?;
    let texts = [fs::read_to_string(&a[2])?, fs::read_to_string(&a[3])?];
    let mut docs = Vec::new();
    for i in 0..width {
        let ids = tok.encode(&texts[i % 2], false)?;
        assert!(!ids.is_empty());
        docs.push(
            (0..depth + 512 + steps)
                .map(|j| ids[(j + i * 157) % ids.len()] as u32)
                .collect::<Vec<_>>(),
        );
    }
    let extent = depth + 4096 + steps;
    let mut loaded = Qwen4ExpLoadedModel::load_with_adaptive_prefill(
        &ctx,
        &gg,
        Qwen4ExpSessionCapacity::for_forward_limit(&c, extent)?,
        extent,
        Qwen4ExpDecodeOptions::default(),
    )?;
    let (factory, first) = loaded.session_factory(&ctx)?;
    let mut runners = vec![first];
    for _ in 1..width {
        runners.push(factory.create_runner(&ctx)?);
    }
    for (i, r) in runners.iter_mut().enumerate() {
        r.prefill(&docs[i][..depth + i * 17])?;
    }
    let mut native = Qwen4ExpDecodeBatch::new(&runners[0], width)?;
    let config = SamplingConfig::qwen38_flash_next(1);
    let mut samplers: Vec<_> = (0..width).map(|_| Sampler::new(config).unwrap()).collect();

    // The model-default low-reasoning path is the agent case; sampling uses the
    // released Flash-Next preset (T=1, top_k=20, top_p=0.95).
    let mut tokens: Vec<u32> = (0..width).map(|i| docs[i][depth + i * 17]).collect();
    let mut logits = {
        let mut members: Vec<_> = runners.iter_mut().collect();
        native.step(&mut members, &tokens)?
    };
    // Eight excluded warm rounds, including host sampling.
    for _ in 0..8 {
        let next: Vec<u32> = logits
            .iter()
            .zip(samplers.iter_mut())
            .map(|(l, s)| Ok(s.sample(l)?.token as u32))
            .collect::<Result<_, qwen_llm::sampling::SamplingError>>()?;
        let mut members: Vec<_> = runners.iter_mut().collect();
        logits = native.step(&mut members, &next)?;
        tokens = next;
    }

    let mut sample_ms = Vec::new();
    let mut step_ms = Vec::new();
    let mut encode_ms = Vec::new();
    let mut gpu_ms = Vec::new();
    let mut round_minus_gpu_ms = Vec::new();
    for _ in 0..steps {
        let t = Instant::now();
        let next: Vec<u32> = logits
            .iter()
            .zip(samplers.iter_mut())
            .map(|(l, s)| Ok(s.sample(l)?.token as u32))
            .collect::<Result<_, qwen_llm::sampling::SamplingError>>()?;
        let sampled = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let mut members: Vec<_> = runners.iter_mut().collect();
        let values = native.step(&mut members, &next)?;
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let [encode, gpu, _total] = native.diagnostic_ms;

        sample_ms.push(sampled);
        step_ms.push(wall);
        encode_ms.push(encode);
        gpu_ms.push(gpu);
        round_minus_gpu_ms.push(sampled + wall - gpu);
        logits = values;
        tokens = next;
    }
    let out = serde_json::json!({
        "kind": "fn_batch_loop_price",
        "depth": depth,
        "width": width,
        "steps": steps,
        "tokens": tokens,
        "median_ms": {
            "sample": median(&sample_ms),
            "step": median(&step_ms),
            "encode": median(&encode_ms),
            "gpu": median(&gpu_ms),
            "round_minus_gpu": median(&round_minus_gpu_ms),
        },
        "mean_ms": {
            "sample": mean(&sample_ms),
            "step": mean(&step_ms),
            "encode": mean(&encode_ms),
            "gpu": mean(&gpu_ms),
            "round_minus_gpu": mean(&round_minus_gpu_ms),
        },
    });
    println!("{}", out);
    Ok(())
}
