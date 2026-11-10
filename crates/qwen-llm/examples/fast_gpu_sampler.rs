#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Price `FastGpuSampler` (Gumbel-max, unforced agent preset) at VOCAB.
//! Requires `--features perf-ablations`; build with `--profile iter`.
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use qwen_llm::{
    metal::{FastGpuSampler, KernelEncoder, MetalContext, MetalTensor, wait_completed},
    tensor::GgmlType,
};
use std::time::Instant;

#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires --features perf-ablations");
}

#[cfg(feature = "perf-ablations")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let vocab: usize = a.get(1).map_or(Ok(129_280), |s| s.parse())?;
    let reps: usize = a.get(2).map_or(Ok(300), |s| s.parse())?;
    let ctx = MetalContext::new()?;
    let row: Vec<f32> = (0..vocab)
        .map(|i| (((i.wrapping_mul(2_654_435_761) % 1000) as f32) - 500.) / 37.)
        .collect();
    for members in [1usize, 3, 4] {
        let logits: Vec<f32> = (0..members).flat_map(|_| row.iter().copied()).collect();
        let x = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&logits),
            vec![(members * vocab) as u64],
            GgmlType::F32,
        )?;
        let tok = MetalTensor::zeros_i32(&ctx, vec![members as u64])?;
        let sampler = FastGpuSampler::new(&ctx, vocab, members)?;
        for _ in 0..8 {
            let c = ctx.queue.commandBuffer().unwrap();
            let e = KernelEncoder::begin(&c);
            sampler.encode(&ctx, &e, &x, &tok, 1, 1.0)?;
            e.end();
            c.commit();
            wait_completed(&c)?;
        }
        let mut wall = Vec::with_capacity(reps);
        let mut gpu = Vec::with_capacity(reps);
        for k in 0..reps {
            let t = Instant::now();
            let c = ctx.queue.commandBuffer().unwrap();
            let e = KernelEncoder::begin(&c);
            sampler.encode(&ctx, &e, &x, &tok, 1 + k as u64, 1.0)?;
            e.end();
            c.commit();
            wait_completed(&c)?;
            wall.push(t.elapsed().as_secs_f64() * 1e3);
            gpu.push((c.GPUEndTime() - c.GPUStartTime()) * 1e3);
        }
        wall.sort_by(f64::total_cmp);
        gpu.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({
                "kind": "fast_sampler_price",
                "vocab": vocab,
                "members": members,
                "reps": reps,
                "wall_ms_median": wall[wall.len() / 2],
                "gpu_ms_median": gpu[gpu.len() / 2],
                "gpu_ms_mean": gpu.iter().sum::<f64>() / reps as f64,
            })
        );
    }
    Ok(())
}
