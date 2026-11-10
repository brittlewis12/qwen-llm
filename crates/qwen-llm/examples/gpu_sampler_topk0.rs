#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Price `GpuTokenSampler` alone for the agent presets. VOCAB [REPS=200].
//! Requires `--features perf-ablations`; build with `--profile iter`.
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLSize};
use qwen_llm::{
    metal::{GpuTokenSampler, KernelEncoder, MetalContext, MetalTensor, wait_completed},
    sampling::SamplingConfig,
    tensor::GgmlType,
};
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SampleArgs {
    n: u32,
    temperature: f32,
    min_p: f32,
    top_p: f32,
    uniform: f32,
}

#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires --features perf-ablations");
}

#[cfg(feature = "perf-ablations")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let vocab: usize = a.get(1).map_or(Ok(129_280), |s| s.parse())?;
    let reps: usize = a.get(2).map_or(Ok(200), |s| s.parse())?;
    let ctx = MetalContext::new()?;
    let logits: Vec<f32> = (0..vocab)
        .map(|i| (((i.wrapping_mul(2_654_435_761) % 1000) as f32) - 500.) / 37.)
        .collect();
    let x = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&logits),
        vec![vocab as u64],
        GgmlType::F32,
    )?;
    let presets = [
        (
            "v41",
            SamplingConfig {
                temperature: 1.,
                top_k: 0,
                top_p: 1.,
                min_p: 0.,
                seed: 1,
            },
        ),
        (
            "glm",
            SamplingConfig {
                temperature: 1.,
                top_k: 0,
                top_p: 0.95,
                min_p: 0.,
                seed: 1,
            },
        ),
        ("fn", SamplingConfig::qwen38_flash_next(1)),
    ];
    for (name, config) in presets {
        let sampler = GpuTokenSampler::new(&ctx, vocab, config)?;
        let token = MetalTensor::zeros_i32(&ctx, vec![1])?;
        // warm
        for _ in 0..8 {
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            sampler.encode(&ctx, &enc, &x, &token, 0.371)?;
            enc.end();
            command.commit();
            wait_completed(&command)?;
        }
        let mut wall = Vec::new();
        let mut gpu = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            sampler.encode(&ctx, &enc, &x, &token, 0.371)?;
            enc.end();
            command.commit();
            wait_completed(&command)?;
            wall.push(t.elapsed().as_secs_f64() * 1e3);
            gpu.push((command.GPUEndTime() - command.GPUStartTime()) * 1e3);
        }
        wall.sort_by(f64::total_cmp);
        gpu.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({
                "kind": "gpu_sampler_price",
                "preset": name,
                "vocab": vocab,
                "reps": reps,
                "wall_ms_median": wall[wall.len()/2],
                "wall_ms_mean": wall.iter().sum::<f64>()/wall.len() as f64,
                "gpu_ms_median": gpu[gpu.len()/2],
                "gpu_ms_mean": gpu.iter().sum::<f64>()/gpu.len() as f64,
            })
        );
    }
    // Final sample kernel alone over a sorted pair array.
    let pairs: Vec<u32> = (0..vocab)
        .flat_map(|i| {
            let value = ((vocab - i) as f32) / 100.;
            [value.to_bits(), i as u32]
        })
        .collect();
    let sorted = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&pairs),
        vec![(vocab * 2) as u64],
        GgmlType::I32,
    )?;
    let token = MetalTensor::zeros_i32(&ctx, vec![1])?;
    for (name, top_p) in [("v41", 1.0f32), ("glm", 0.95f32)] {
        let args = SampleArgs {
            n: vocab as u32,
            temperature: 1.,
            min_p: 0.,
            top_p,
            uniform: 0.371,
        };
        for _ in 0..8 {
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            enc.set_pipeline(&ctx.pipeline("kernel_token_sample_sorted")?);
            enc.set_bytes(0, &args);
            enc.set_tensor(1, &sorted);
            enc.set_tensor(2, &token);
            enc.dispatch(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end();
            command.commit();
            wait_completed(&command)?;
        }
        let mut wall = Vec::new();
        let mut gpu = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            enc.set_pipeline(&ctx.pipeline("kernel_token_sample_sorted")?);
            enc.set_bytes(0, &args);
            enc.set_tensor(1, &sorted);
            enc.set_tensor(2, &token);
            enc.dispatch(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end();
            command.commit();
            wait_completed(&command)?;
            wall.push(t.elapsed().as_secs_f64() * 1e3);
            gpu.push((command.GPUEndTime() - command.GPUStartTime()) * 1e3);
        }
        wall.sort_by(f64::total_cmp);
        gpu.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({
                "kind": "gpu_sample_kernel_price",
                "preset": name,
                "vocab": vocab,
                "wall_ms_median": wall[wall.len()/2],
                "gpu_ms_median": gpu[gpu.len()/2],
            })
        );
    }
    let weights = MetalTensor::zeros_f32(&ctx, vec![vocab as u64])?;
    for (name, top_p) in [("v41", 1.0f32), ("glm", 0.95f32)] {
        let args = SampleArgs {
            n: vocab as u32,
            temperature: 1.,
            min_p: 0.,
            top_p,
            uniform: 0.371,
        };
        for _ in 0..8 {
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            enc.set_pipeline(&ctx.pipeline("kernel_token_sample_weighted")?);
            enc.set_bytes(0, &args);
            enc.set_tensor(1, &sorted);
            enc.set_tensor(2, &weights);
            enc.set_tensor(3, &token);
            enc.dispatch(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end();
            command.commit();
            wait_completed(&command)?;
        }
        let mut wall = Vec::new();
        let mut gpu = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            enc.set_pipeline(&ctx.pipeline("kernel_token_sample_weighted")?);
            enc.set_bytes(0, &args);
            enc.set_tensor(1, &sorted);
            enc.set_tensor(2, &weights);
            enc.set_tensor(3, &token);
            enc.dispatch(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end();
            command.commit();
            wait_completed(&command)?;
            wall.push(t.elapsed().as_secs_f64() * 1e3);
            gpu.push((command.GPUEndTime() - command.GPUStartTime()) * 1e3);
        }
        wall.sort_by(f64::total_cmp);
        gpu.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({
                "kind": "gpu_weighted_sample_price",
                "preset": name,
                "wall_ms_median": wall[wall.len()/2],
                "gpu_ms_median": gpu[gpu.len()/2],
            })
        );
    }
    Ok(())
}
