//! Single concurrent encoder: one independent long dispatch, then a 20-link
//! ping-pong chain. GPUStartTime/GPUEndTime measure the entire command buffer;
//! no dispatch counter samples or profiler encoder splits perturb overlap.
//! The no-barrier arm deliberately races and is only an overlap ceiling.
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDispatchType, MTLResource, MTLSize,
};
use qwen_llm::metal::{Buffer, MetalContext, Pipeline, commit_and_wait};
use std::ptr::NonNull;

type Error = Box<dyn std::error::Error>;
const LINKS: usize = 20;
const THREADS: usize = 128;

#[derive(Clone, Copy)]
enum Mode {
    Long,
    Chain,
    Full,
    Resources,
    None,
    TwoQueues,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Long => "long-only",
            Self::Chain => "chain-only",
            Self::Full => "full-buffers",
            Self::Resources => "chain-resources",
            Self::None => "no-barrier-unsafe-control",
            Self::TwoQueues => "two-queues-serial-chain",
        }
    }
}
const MODES: [Mode; 6] = [
    Mode::Long,
    Mode::Chain,
    Mode::Full,
    Mode::Resources,
    Mode::None,
    Mode::TwoQueues,
];

struct Probe {
    ctx: MetalContext,
    other: MetalContext,
    long: Pipeline,
    short: Pipeline,
    long_in: Buffer,
    long_out: Buffer,
    chain: [Buffer; 2],
    long_iterations: u32,
    short_iterations: u32,
    long_expected: [u32; THREADS],
    chain_expected: [u32; THREADS],
}

struct Sample {
    gpu_us: f64,
    long_bitwise: bool,
    chain_bitwise: bool,
    queue_start_skew_us: f64,
    main_command_us: f64,
    chain_command_us: Option<f64>,
    queue_overlap_us: Option<f64>,
}

fn mix(mut x: u32, iterations: u32) -> u32 {
    for _ in 0..iterations {
        x ^= x >> 13;
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
    }
    x
}

fn dispatch(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso: &Pipeline,
    input: &Buffer,
    output: &Buffer,
    iterations: u32,
) {
    enc.setComputePipelineState(pso);
    // SAFETY: all bindings and the scalar are live for encoding; Metal copies
    // the scalar. Buffers hold 128 uints and the grid contains exactly 128 lanes.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 0);
        enc.setBuffer_offset_atIndex(Some(output), 0, 1);
        enc.setBytes_length_atIndex(NonNull::from(&iterations).cast(), 4, 2);
    }
    let size = MTLSize {
        width: THREADS,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        size,
    );
}

impl Probe {
    fn update_oracles(&mut self) {
        self.long_expected = std::array::from_fn(|i| mix(i as u32 + 1, self.long_iterations));
        self.chain_expected =
            std::array::from_fn(|i| mix(i as u32 + 1, self.short_iterations * LINKS as u32));
    }

    fn warm_gpu(&self) -> Result<(), Error> {
        // Sustained work brings the device to its active clock before loop
        // calibration. Per-sample CPU oracle computation otherwise lets clocks
        // drift on M5; the measured repetitions only compare cached oracle bits.
        let start = std::time::Instant::now();
        while start.elapsed().as_millis() < 500 {
            let command = self
                .ctx
                .queue
                .commandBuffer()
                .ok_or("no warmup command buffer")?;
            let enc = command.computeCommandEncoder().ok_or("no warmup encoder")?;
            for _ in 0..64 {
                dispatch(
                    &enc,
                    &self.long,
                    &self.long_in,
                    &self.long_out,
                    self.long_iterations,
                );
            }
            enc.endEncoding();
            commit_and_wait(&command)?;
        }
        Ok(())
    }

    fn sample(&self, mode: Mode) -> Result<Sample, Error> {
        // Every previous command was waited for; CPU initialization is outside
        // the GPU timestamp window and cannot race a previous repetition.
        unsafe {
            let seed = std::slice::from_raw_parts_mut(
                self.chain[0].contents().as_ptr().cast::<u32>(),
                THREADS,
            );
            for (i, x) in seed.iter_mut().enumerate() {
                *x = i as u32 + 1;
            }
            self.chain[1]
                .contents()
                .as_ptr()
                .write_bytes(0, THREADS * 4);
        }
        let command = self.ctx.queue.commandBuffer().ok_or("no command buffer")?;
        let enc = command
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or("no concurrent encoder")?;
        let has_long = !matches!(mode, Mode::Chain);
        let has_chain = !matches!(mode, Mode::Long);
        if has_long {
            dispatch(
                &enc,
                &self.long,
                &self.long_in,
                &self.long_out,
                self.long_iterations,
            );
        }
        let separate = if matches!(mode, Mode::TwoQueues) {
            enc.endEncoding();
            let other = self
                .other
                .queue
                .commandBuffer()
                .ok_or("no second command buffer")?;
            let serial = other.computeCommandEncoder().ok_or("no serial encoder")?;
            for link in 0..LINKS {
                dispatch(
                    &serial,
                    &self.short,
                    &self.chain[link % 2],
                    &self.chain[(link + 1) % 2],
                    self.short_iterations,
                );
            }
            serial.endEncoding();
            // Commit both before waiting. No cross-queue shared writable data.
            command.commit();
            other.commit();
            command.waitUntilCompleted();
            commit_status(&command)?;
            other.waitUntilCompleted();
            commit_status(&other)?;
            Some(other)
        } else {
            if has_chain {
                for link in 0..LINKS {
                    if link > 0 {
                        match mode {
                            Mode::Full | Mode::Chain => {
                                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers)
                            }
                            Mode::Resources => {
                                let mut resources = self.chain.each_ref().map(|b| {
                                    NonNull::from(ProtocolObject::<dyn MTLResource>::from_ref(&**b))
                                });
                                // SAFETY: the nonempty array and both resources
                                // remain live for the call and command lifetime.
                                unsafe {
                                    enc.memoryBarrierWithResources_count(
                                        NonNull::new(resources.as_mut_ptr()).unwrap(),
                                        resources.len(),
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                    dispatch(
                        &enc,
                        &self.short,
                        &self.chain[link % 2],
                        &self.chain[(link + 1) % 2],
                        self.short_iterations,
                    );
                }
            }
            enc.endEncoding();
            commit_and_wait(&command)?;
            None
        };
        let start = command.GPUStartTime();
        let end = command.GPUEndTime();
        let main_command_us = (end - start) * 1e6;
        let (start, end, skew_us, chain_command_us, queue_overlap_us) = if let Some(other) =
            separate
        {
            (
                start.min(other.GPUStartTime()),
                end.max(other.GPUEndTime()),
                (start - other.GPUStartTime()).abs() * 1e6,
                Some((other.GPUEndTime() - other.GPUStartTime()) * 1e6),
                Some(
                    (end.min(other.GPUEndTime()) - start.max(other.GPUStartTime())).max(0.0) * 1e6,
                ),
            )
        } else {
            (start, end, 0.0, None, None)
        };
        if !start.is_finite() || end <= start {
            return Err("invalid GPU timestamps".into());
        }
        // Check every output bit against a CPU integer oracle, outside timing.
        let long_ok = !has_long
            || unsafe {
                std::slice::from_raw_parts(self.long_out.contents().as_ptr().cast::<u32>(), THREADS)
                    == self.long_expected
            };
        let chain_ok = !has_chain
            || unsafe {
                std::slice::from_raw_parts(
                    self.chain[LINKS % 2].contents().as_ptr().cast::<u32>(),
                    THREADS,
                ) == self.chain_expected
            };
        if !long_ok || (!chain_ok && !matches!(mode, Mode::None)) {
            return Err(format!(
                "{} failed bitwise oracle: long={long_ok} chain={chain_ok}",
                mode.name()
            )
            .into());
        }
        Ok(Sample {
            gpu_us: (end - start) * 1e6,
            long_bitwise: long_ok,
            chain_bitwise: chain_ok,
            queue_start_skew_us: skew_us,
            main_command_us,
            chain_command_us,
            queue_overlap_us,
        })
    }

    fn calibrate(&mut self, long_target: f64) -> Result<(), Error> {
        for (mode, target) in [(Mode::Chain, 180.0), (Mode::Long, long_target)] {
            for _ in 0..6 {
                let mut times = Vec::new();
                self.sample(mode)?; // warm the current loop count
                for _ in 0..3 {
                    times.push(self.sample(mode)?.gpu_us);
                }
                let us = super::median(&mut times);
                let iterations = if matches!(mode, Mode::Long) {
                    &mut self.long_iterations
                } else {
                    &mut self.short_iterations
                };
                println!(
                    "{}",
                    serde_json::json!({"kind":"barrier_calibration", "mode":mode.name(), "iterations":iterations, "gpu_us":us, "target_us":target})
                );
                if (0.9 * target..1.1 * target).contains(&us) {
                    break;
                }
                *iterations =
                    ((*iterations as f64 * target / us).round() as u32).clamp(1, 1_000_000);
                self.update_oracles();
            }
        }
        Ok(())
    }
}

fn commit_status(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), Error> {
    if let Some(error) = command.error() {
        return Err(error.to_string().into());
    }
    Ok(())
}

pub(super) fn run() -> Result<(), Error> {
    let reps = std::env::args()
        .nth(2)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(61);
    let target = std::env::args()
        .nth(3)
        .map(|s| s.parse::<f64>())
        .transpose()?
        .unwrap_or(400.0);
    if reps == 0 || !(300.0..=500.0).contains(&target) {
        return Err(
            "usage: dispatch_deadtime --resource-barriers [positive reps] [long_us: 300..500]"
                .into(),
        );
    }
    let ctx = MetalContext::new()?;
    println!(
        "{}",
        serde_json::json!({"kind":"resource_barrier_device", "describe":ctx.describe(), "reps":reps, "links":LINKS, "grid":1, "threads":THREADS, "target_long_us":target, "clock":"GPUStartTime/GPUEndTime; two-queue envelope uses min(start)..max(end)", "no_barrier_is_unsafe":true})
    );
    let mut probe = Probe {
        other: ctx.with_new_command_queue()?,
        long: ctx.pipeline("kernel_dispatch_barrier_long")?,
        short: ctx.pipeline("kernel_dispatch_barrier_short")?,
        long_in: ctx.buffer_from(&(1..=THREADS as u32).collect::<Vec<_>>())?,
        long_out: ctx.buffer_uninit(THREADS * 4)?,
        chain: [
            ctx.buffer_uninit(THREADS * 4)?,
            ctx.buffer_uninit(THREADS * 4)?,
        ],
        ctx,
        long_iterations: 10000,
        short_iterations: 128,
        long_expected: [0; THREADS],
        chain_expected: [0; THREADS],
    };
    probe.update_oracles();
    probe.warm_gpu()?;
    probe.calibrate(target)?;
    // Warm every mode. Rotate a forward/reverse sweep so no arm consistently
    // pays clock ramp or comes directly after the same control.
    for mode in MODES {
        probe.sample(mode)?;
    }
    let mut samples: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(reps));
    let mut correct = [0usize; 6];
    for rep in 0..reps {
        for step in 0..MODES.len() {
            let index = if rep % 2 == 0 {
                (step + rep) % MODES.len()
            } else {
                (rep + MODES.len() - step) % MODES.len()
            };
            let mode = MODES[index];
            let sample = probe.sample(mode)?;
            samples[index].push(sample.gpu_us);
            correct[index] += usize::from(sample.chain_bitwise && sample.long_bitwise);
            println!(
                "{}",
                serde_json::json!({"kind":"resource_barrier_sample", "rep":rep, "mode":mode.name(), "gpu_us":sample.gpu_us, "long_bitwise":sample.long_bitwise, "chain_bitwise":sample.chain_bitwise, "queue_start_skew_us":sample.queue_start_skew_us, "main_command_us":sample.main_command_us, "chain_command_us":sample.chain_command_us, "queue_overlap_us":sample.queue_overlap_us})
            );
        }
    }
    let medians: [f64; 6] = std::array::from_fn(|i| super::median(&mut samples[i]));
    let serial = medians[0] + medians[1];
    let ceiling = medians[0].max(medians[1]);
    for (i, mode) in MODES.iter().enumerate() {
        println!(
            "{}",
            serde_json::json!({"kind":"resource_barrier_summary", "mode":mode.name(), "gpu_us_p50":medians[i], "gpu_us_p10":samples[i][reps / 10], "gpu_us_p90":samples[i][reps * 9 / 10], "correct_reps":correct[i], "reps":reps, "long_iterations":probe.long_iterations, "short_iterations":probe.short_iterations, "isolated_sum_us":serial, "ideal_overlap_us":ceiling, "hidden_chain_fraction":(i >= 2).then_some((serial - medians[i]) / (serial - ceiling)), "ratio_to_full":medians[i]/medians[2]})
        );
    }
    if !(300.0..=500.0).contains(&medians[0]) {
        return Err(format!(
            "long-kernel median {} us is outside 300..500; rerun after checking clocks",
            medians[0]
        )
        .into());
    }
    Ok(())
}
