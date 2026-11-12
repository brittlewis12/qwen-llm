//! Device-scope phase handoff microbenchmark, launched by dispatch_deadtime.
//!
//! Each threadgroup dequeues exactly one task from a global atomic ticket. The
//! ticket order is a topological ordering: task N consumes task N-1, which has
//! already been dequeued and is therefore resident or complete. There is no
//! grid-wide barrier and no threadgroup waits for an unstarted predecessor.
//! Payload publication uses an explicit sequentially consistent device-scope
//! fence and relaxed atomic store; the consumer polls with a relaxed load then
//! executes the matching fence before reading the payload. The owner trace verifies every
//! edge crossed threadgroups. A second queue can run a wide arithmetic load
//! concurrently to stress shared-device progress and visibility.
use super::Args;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLDevice, MTLLanguageVersion, MTLMathMode, MTLSize,
};
use qwen_llm::metal::{Buffer, MetalContext, Pipeline};
use std::{ptr::NonNull, time::Instant};

type Error = Box<dyn std::error::Error>;
const DEFAULT_TASKS: usize = 1024;
const SEED: u32 = 0x9e37_79b9;
const GROUP_THREADS: usize = 32;
const LOAD_GROUPS: usize = 512;
const LOAD_THREADS: usize = 128;
const LOAD_ITERS: u32 = 8192;

fn mix(mut x: u32) -> u32 {
    x ^= x >> 13;
    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    x ^= x >> 13;
    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    x
}

fn expected(tasks: usize) -> u32 {
    (0..tasks).fold(SEED, |x, _| mix(x))
}

fn load_expected(gid: usize) -> u32 {
    let mut value = gid as u32 ^ 0xa511_e9b3;
    for _ in 0..LOAD_ITERS {
        value ^= value >> 13;
        value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    }
    value
}

fn check_load_output(buffer: &Buffer) -> Result<(), Error> {
    let words = unsafe { read_u32(buffer, LOAD_GROUPS * LOAD_THREADS) };
    for gid in [0, 1, words.len() / 2, words.len() - 1] {
        if words[gid] != load_expected(gid) {
            return Err(format!("contention output mismatch at {gid}").into());
        }
    }
    Ok(())
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn queue_overlap_us(
    first: &ProtocolObject<dyn MTLCommandBuffer>,
    second: &ProtocolObject<dyn MTLCommandBuffer>,
) -> f64 {
    let start = first.GPUStartTime().max(second.GPUStartTime());
    let end = first.GPUEndTime().min(second.GPUEndTime());
    (end - start).max(0.0) * 1e6
}

unsafe fn fill_u32(buffer: &Buffer, words: usize, value: u32) {
    // SAFETY: this probe allocates shared buffers with the exact requested
    // number of words and initializes them only after the prior command ends.
    let dst =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u32>(), words) };
    dst.fill(value);
}

unsafe fn read_u32(buffer: &Buffer, words: usize) -> Vec<u32> {
    // SAFETY: caller waits for the producing command before reading shared memory.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u32>(), words).to_vec() }
}

fn set_buffer(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, buffer: &Buffer, index: usize) {
    // SAFETY: buffer remains live through command completion and is bound at
    // the shader's matching buffer index.
    unsafe { enc.setBuffer_offset_atIndex(Some(buffer), 0, index) };
}

fn set_bytes<T>(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    // SAFETY: Metal copies this small POD value during encoding.
    unsafe {
        enc.setBytes_length_atIndex(NonNull::from(value).cast(), std::mem::size_of::<T>(), index)
    };
}

fn dispatch_groups(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    grid: usize,
    threads: usize,
) {
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: grid,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

struct Probe {
    ctx: MetalContext,
    load_ctx: MetalContext,
    phase: Pipeline,
    load: Pipeline,
    dependent: Pipeline,
    next: Buffer,
    ready: Buffer,
    payload: Buffer,
    owners: Buffer,
    touch: Buffer,
    load_out: Buffer,
    tasks: usize,
}

impl Probe {
    fn reset(&self) {
        // Atomic storage is four-byte aligned and initialized only with all
        // prior command buffers complete.
        unsafe {
            fill_u32(&self.next, 1, 0);
            fill_u32(&self.ready, self.tasks, 0);
            fill_u32(&self.payload, self.tasks, 0);
            fill_u32(&self.owners, self.tasks, u32::MAX);
        }
    }

    fn encode_phase(
        &self,
    ) -> Result<objc2::rc::Retained<ProtocolObject<dyn MTLCommandBuffer>>, Error> {
        let command = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or("no phase command buffer")?;
        let enc = command.computeCommandEncoder().ok_or("no phase encoder")?;
        enc.setComputePipelineState(&self.phase);
        set_buffer(&enc, &self.next, 0);
        set_buffer(&enc, &self.ready, 1);
        set_buffer(&enc, &self.payload, 2);
        set_buffer(&enc, &self.owners, 3);
        let tasks = self.tasks as u32;
        let seed = SEED;
        set_bytes(&enc, &tasks, 4);
        set_bytes(&enc, &seed, 5);
        dispatch_groups(&enc, self.tasks, GROUP_THREADS);
        enc.endEncoding();
        Ok(command)
    }

    fn encode_load(
        &self,
    ) -> Result<objc2::rc::Retained<ProtocolObject<dyn MTLCommandBuffer>>, Error> {
        let command = self
            .load_ctx
            .queue
            .commandBuffer()
            .ok_or("no load command buffer")?;
        let enc = command.computeCommandEncoder().ok_or("no load encoder")?;
        enc.setComputePipelineState(&self.load);
        set_buffer(&enc, &self.load_out, 0);
        let iterations = LOAD_ITERS;
        set_bytes(&enc, &iterations, 1);
        dispatch_groups(&enc, LOAD_GROUPS, LOAD_THREADS);
        enc.endEncoding();
        Ok(command)
    }

    fn warm_gpu(&self) -> Result<(), Error> {
        let start = Instant::now();
        while start.elapsed().as_millis() < 500 {
            let command = self.encode_load()?;
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return Err(error.to_string().into());
            }
            check_load_output(&self.load_out)?;
        }
        Ok(())
    }

    fn phase_sample(&self, contention: bool) -> Result<(f64, u64, f64), Error> {
        self.reset();
        let command = self.encode_phase()?;
        let load = if contention {
            Some(self.encode_load()?)
        } else {
            None
        };
        let start = Instant::now();
        command.commit();
        if let Some(load) = &load {
            load.commit();
        }
        command.waitUntilCompleted();
        if let Some(load) = &load {
            load.waitUntilCompleted();
        }
        if let Some(error) = command.error() {
            return Err(error.to_string().into());
        }
        if let Some(error) = load.as_ref().and_then(|c| c.error()) {
            return Err(error.to_string().into());
        }
        if contention {
            check_load_output(&self.load_out)?;
        }
        let elapsed = (command.GPUEndTime() - command.GPUStartTime()) * 1e6;
        let overlap = load
            .as_ref()
            .map(|load| queue_overlap_us(&command, load))
            .unwrap_or(0.0);
        let payload = unsafe { read_u32(&self.payload, self.tasks) };
        let owners = unsafe { read_u32(&self.owners, self.tasks) };
        let want = expected(self.tasks);
        if payload.last().copied() != Some(want) {
            return Err(format!(
                "phase chain mismatch: got {:?}, expected {want:#010x}",
                payload.last()
            )
            .into());
        }
        if payload
            .iter()
            .enumerate()
            .any(|(i, &v)| v != expected(i + 1))
        {
            return Err("phase chain intermediate payload mismatch".into());
        }
        let cross_group = owners.windows(2).filter(|p| p[0] != p[1]).count();
        if cross_group != self.tasks.saturating_sub(1) {
            return Err(format!(
                "task ownership was not one-to-one: {cross_group}/{} handoffs crossed groups",
                self.tasks.saturating_sub(1)
            )
            .into());
        }
        if start.elapsed().as_secs_f64() <= 0.0 || !elapsed.is_finite() || elapsed <= 0.0 {
            return Err("invalid phase timing".into());
        }
        Ok((elapsed, cross_group as u64, overlap))
    }

    fn encode_boundary(
        &self,
    ) -> Result<objc2::rc::Retained<ProtocolObject<dyn MTLCommandBuffer>>, Error> {
        let command = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or("no boundary command buffer")?;
        let enc = command
            .computeCommandEncoder()
            .ok_or("no boundary encoder")?;
        for _ in 0..self.tasks {
            enc.setComputePipelineState(&self.dependent);
            set_buffer(&enc, &self.touch, 0);
            let args = Args {
                total_threads: 32 * 128,
                words: 0,
                dispatch_index: 0,
            };
            set_bytes(&enc, &args, 1);
            dispatch_groups(&enc, 32, 128);
        }
        enc.endEncoding();
        Ok(command)
    }

    fn boundary_sample(&self, contention: bool) -> Result<(f64, f64), Error> {
        unsafe { fill_u32(&self.touch, self.touch.length() / 4, 0) };
        let command = self.encode_boundary()?;
        let load = if contention {
            Some(self.encode_load()?)
        } else {
            None
        };
        let start = Instant::now();
        command.commit();
        if let Some(load) = &load {
            load.commit();
        }
        command.waitUntilCompleted();
        if let Some(load) = &load {
            load.waitUntilCompleted();
            if let Some(error) = load.error() {
                return Err(error.to_string().into());
            }
        }
        if let Some(error) = command.error() {
            return Err(error.to_string().into());
        }
        if contention {
            check_load_output(&self.load_out)?;
        }
        let dispatches = unsafe { read_u32(&self.touch, 1) }[0];
        if dispatches != self.tasks as u32 {
            return Err(format!("boundary payload mismatch: {dispatches} dispatches").into());
        }
        if start.elapsed().as_secs_f64() <= 0.0 {
            return Err("invalid boundary timing".into());
        }
        let elapsed = (command.GPUEndTime() - command.GPUStartTime()) * 1e6;
        let overlap = load
            .as_ref()
            .map(|load| queue_overlap_us(&command, load))
            .unwrap_or(0.0);
        Ok((elapsed, overlap))
    }
}

fn phase_pipelines(ctx: &mut MetalContext) -> Result<(Pipeline, Pipeline), Error> {
    let old = ctx.library.clone();
    let result = (|| {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        options.setMathMode(MTLMathMode::Fast);
        ctx.library = ctx
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("hc_phase_sync.metal")),
                Some(&options),
            )
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        let phase = ctx.pipeline("kernel_hc_phase_sync")?;
        let load = ctx.pipeline("kernel_hc_phase_contention")?;
        Ok::<_, Error>((phase, load))
    })();
    ctx.library = old;
    result
}

pub(super) fn run() -> Result<(), Error> {
    let mut args = std::env::args().skip(2);
    let reps = args
        .next()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(31usize);
    let tasks = args
        .next()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(DEFAULT_TASKS);
    if reps < 3 || tasks < 2 || tasks > 16_384 {
        return Err("phase-sync requires reps >= 3 and tasks in 2..=16384".into());
    }

    let mut ctx = MetalContext::new()?;
    let (phase, load) = phase_pipelines(&mut ctx)?;
    let probe = Probe {
        load_ctx: ctx.with_new_command_queue()?,
        phase,
        load,
        dependent: ctx.pipeline("kernel_dispatch_deadtime_dependent")?,
        next: ctx.buffer_uninit(4)?,
        ready: ctx.buffer_uninit(tasks * 4)?,
        payload: ctx.buffer_uninit(tasks * 4)?,
        owners: ctx.buffer_uninit(tasks * 4)?,
        touch: ctx.buffer_uninit(4)?,
        load_out: ctx.buffer_uninit(LOAD_GROUPS * LOAD_THREADS * 4)?,
        tasks,
        ctx,
    };

    // Warm both GPU slices before timing. All correctness checks still run on
    // every timed repetition, including contention repetitions.
    probe.warm_gpu()?;
    for _ in 0..4 {
        probe.phase_sample(false)?;
        probe.boundary_sample(false)?;
    }

    println!(
        "{}",
        serde_json::json!({
            "kind": "hc_phase_sync_header",
            "device": probe.ctx.describe(),
            "tasks": tasks,
            "threadgroups": tasks,
            "threads_per_threadgroup": GROUP_THREADS,
            "dependency": "ticket i consumes payload i-1; one dequeue per threadgroup",
            "dispatch_control_shape": "32 threadgroups x 128 threads, one shared counter word creates each RAW edge",
            "fences": "coherent(device) buffers; relaxed atomic ticket/signal; device-scope seq_cst fence before publish and after observing signal",
            "contention": {"groups": LOAD_GROUPS, "threads_per_group": LOAD_THREADS, "iterations": LOAD_ITERS},
            "reps": reps,
            "order": "500ms GPU warmup; four arms interleaved in rotating order",
            "metal_language": "metal3.2 runtime compile succeeded",
        })
    );

    let mut free_phase = Vec::with_capacity(reps);
    let mut free_boundary = Vec::with_capacity(reps);
    let mut loaded_phase = Vec::with_capacity(reps);
    let mut loaded_boundary = Vec::with_capacity(reps);
    let mut loaded_phase_overlap = Vec::with_capacity(reps);
    let mut loaded_boundary_overlap = Vec::with_capacity(reps);
    let mut cross_group_edges = 0u64;
    for round in 0..reps {
        match round % 4 {
            0 => {
                let (value, edges, _) = probe.phase_sample(false)?;
                free_phase.push(value);
                cross_group_edges += edges;
                free_boundary.push(probe.boundary_sample(false)?.0);
                let (value, edges, overlap) = probe.phase_sample(true)?;
                loaded_phase.push(value);
                cross_group_edges += edges;
                loaded_phase_overlap.push(overlap);
                let (value, overlap) = probe.boundary_sample(true)?;
                loaded_boundary.push(value);
                loaded_boundary_overlap.push(overlap);
            }
            1 => {
                free_boundary.push(probe.boundary_sample(false)?.0);
                let (value, edges, _) = probe.phase_sample(false)?;
                free_phase.push(value);
                cross_group_edges += edges;
                let (value, overlap) = probe.boundary_sample(true)?;
                loaded_boundary.push(value);
                loaded_boundary_overlap.push(overlap);
                let (value, edges, overlap) = probe.phase_sample(true)?;
                loaded_phase.push(value);
                cross_group_edges += edges;
                loaded_phase_overlap.push(overlap);
            }
            2 => {
                let (value, edges, overlap) = probe.phase_sample(true)?;
                loaded_phase.push(value);
                cross_group_edges += edges;
                loaded_phase_overlap.push(overlap);
                let (value, overlap) = probe.boundary_sample(true)?;
                loaded_boundary.push(value);
                loaded_boundary_overlap.push(overlap);
                let (value, edges, _) = probe.phase_sample(false)?;
                free_phase.push(value);
                cross_group_edges += edges;
                free_boundary.push(probe.boundary_sample(false)?.0);
            }
            _ => {
                let (value, overlap) = probe.boundary_sample(true)?;
                loaded_boundary.push(value);
                loaded_boundary_overlap.push(overlap);
                let (value, edges, overlap) = probe.phase_sample(true)?;
                loaded_phase.push(value);
                cross_group_edges += edges;
                loaded_phase_overlap.push(overlap);
                free_boundary.push(probe.boundary_sample(false)?.0);
                let (value, edges, _) = probe.phase_sample(false)?;
                free_phase.push(value);
                cross_group_edges += edges;
            }
        }
    }
    let phase_us = median(&free_phase);
    let boundary_us = median(&free_boundary);
    let loaded_phase_us = median(&loaded_phase);
    let loaded_boundary_us = median(&loaded_boundary);
    println!(
        "{}",
        serde_json::json!({
            "kind": "hc_phase_sync_result",
            "tasks": tasks,
            "reps": reps,
            "phase_chain_us": phase_us,
            "phase_us_per_dependency": phase_us / (tasks - 1) as f64,
            "boundary_chain_us": boundary_us,
            "boundary_us_per_dependency": boundary_us / (tasks - 1) as f64,
            "phase_speedup_vs_boundary": boundary_us / phase_us,
            "loaded_phase_chain_us": loaded_phase_us,
            "loaded_phase_us_per_dependency": loaded_phase_us / (tasks - 1) as f64,
            "loaded_boundary_chain_us": loaded_boundary_us,
            "loaded_boundary_us_per_dependency": loaded_boundary_us / (tasks - 1) as f64,
            "loaded_phase_speedup_vs_boundary": loaded_boundary_us / loaded_phase_us,
            "cross_threadgroup_edges_checked": cross_group_edges,
            "all_payloads_bitwise": true,
            "load_output_words": LOAD_GROUPS * LOAD_THREADS,
            "per_rep_samples_us": {
                "phase": free_phase,
                "boundary": free_boundary,
                "loaded_phase": loaded_phase,
                "loaded_boundary": loaded_boundary,
                "loaded_phase_overlap": loaded_phase_overlap,
                "loaded_boundary_overlap": loaded_boundary_overlap,
            },
        })
    );
    Ok(())
}
