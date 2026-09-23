//! GPU/command timing helpers, percentiles, synthetic prompts.

use super::*;

pub(crate) fn synthetic_prompt_ids(n: usize, vocab_size: u32, seed: u64) -> Vec<i32> {
    let mut state = if seed == 0 { 1 } else { seed };
    let vocab = vocab_size.max(1) as u64;
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ids.push((state % vocab) as i32);
    }
    ids
}

pub(crate) fn sample_mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

pub(crate) fn sample_stdev(xs: &[f64]) -> f64 {
    if xs.len() <= 1 {
        return 0.0;
    }
    let mean = sample_mean(xs);
    let variance = xs
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (xs.len() - 1) as f64;
    variance.sqrt()
}

pub(crate) fn time_gpu_reps<F>(
    ctx: &MetalContext,
    warmup: usize,
    iters: usize,
    mut encode: F,
) -> Result<(f64, f64)>
where
    F: FnMut(&KernelEncoder) -> Result<()>,
{
    for _ in 0..warmup {
        let cmd = ctx.queue.commandBuffer().context("warmup cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode(&enc)?;
        enc.end();
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;
    }

    let mut wall_ms = 0.0f64;
    let mut gpu_ms = 0.0f64;
    for _ in 0..iters {
        let cmd = ctx.queue.commandBuffer().context("timed cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode(&enc)?;
        enc.end();
        let t = Instant::now();
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;
        wall_ms += t.elapsed().as_secs_f64() * 1e3;
        gpu_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
    }
    Ok((wall_ms / iters as f64, gpu_ms / iters as f64))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TimedGpuStats {
    pub(crate) avg_wall_ms: f64,
    pub(crate) avg_gpu_ms: f64,
    pub(crate) p50_wall_ms: f64,
    pub(crate) p50_gpu_ms: f64,
    pub(crate) p90_gpu_ms: f64,
    pub(crate) max_gpu_ms: f64,
}

pub(crate) fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub(crate) fn time_gpu_reps_stats<F>(
    ctx: &MetalContext,
    warmup: usize,
    iters: usize,
    mut encode: F,
) -> Result<TimedGpuStats>
where
    F: FnMut(&KernelEncoder) -> Result<()>,
{
    for _ in 0..warmup {
        let cmd = ctx.queue.commandBuffer().context("warmup cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode(&enc)?;
        enc.end();
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;
    }

    let mut wall_samples = Vec::with_capacity(iters);
    let mut gpu_samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let cmd = ctx.queue.commandBuffer().context("timed cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode(&enc)?;
        enc.end();
        let t = Instant::now();
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;
        wall_samples.push(t.elapsed().as_secs_f64() * 1e3);
        gpu_samples.push((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
    }

    let avg_wall_ms = wall_samples.iter().sum::<f64>() / iters as f64;
    let avg_gpu_ms = gpu_samples.iter().sum::<f64>() / iters as f64;
    wall_samples.sort_by(|a, b| a.total_cmp(b));
    gpu_samples.sort_by(|a, b| a.total_cmp(b));
    Ok(TimedGpuStats {
        avg_wall_ms,
        avg_gpu_ms,
        p50_wall_ms: percentile(&wall_samples, 0.50),
        p50_gpu_ms: percentile(&gpu_samples, 0.50),
        p90_gpu_ms: percentile(&gpu_samples, 0.90),
        max_gpu_ms: *gpu_samples.last().unwrap_or(&0.0),
    })
}

pub(crate) fn time_cmd_reps_stats<F>(
    ctx: &MetalContext,
    warmup: usize,
    iters: usize,
    mut encode: F,
) -> Result<TimedGpuStats>
where
    F: FnMut(&Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Result<()>,
{
    for _ in 0..warmup {
        let cmd = ctx.queue.commandBuffer().context("warmup cmd")?;
        encode(&cmd)?;
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;
    }

    let mut wall_samples = Vec::with_capacity(iters);
    let mut gpu_samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let cmd = ctx.queue.commandBuffer().context("timed cmd")?;
        encode(&cmd)?;
        let t = Instant::now();
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;
        wall_samples.push(t.elapsed().as_secs_f64() * 1e3);
        gpu_samples.push((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
    }

    let avg_wall_ms = wall_samples.iter().sum::<f64>() / iters as f64;
    let avg_gpu_ms = gpu_samples.iter().sum::<f64>() / iters as f64;
    wall_samples.sort_by(|a, b| a.total_cmp(b));
    gpu_samples.sort_by(|a, b| a.total_cmp(b));
    Ok(TimedGpuStats {
        avg_wall_ms,
        avg_gpu_ms,
        p50_wall_ms: percentile(&wall_samples, 0.50),
        p50_gpu_ms: percentile(&gpu_samples, 0.50),
        p90_gpu_ms: percentile(&gpu_samples, 0.90),
        max_gpu_ms: *gpu_samples.last().unwrap_or(&0.0),
    })
}
