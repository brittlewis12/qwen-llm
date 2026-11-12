#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Dependent-dispatch dead-time probe (bench-only, requires perf-ablations).
//!
//! Encodes chains of N dispatches in one command buffer at caller-selected
//! grid/threadgroup/bytes shapes. Serial-dependent dispatches read back the
//! previous dispatch's payload; serial-independent dispatches write disjoint
//! regions; concurrent-independent dispatches share one concurrent encoder.
//! Reports the measured GPU command time for each N, a least-squares
//! `gpu_ms = intercept + slope*N` fit, and the per-dispatch profiler window.
//! The profiler splits each dispatch into its own encoder, so its window
//! includes encoder transitions; the chain slope is the per-dispatch cost
//! used for comparisons. Comparing the three modes across grid shapes answers
//! whether the missed time is constant per dispatch or tied to launch
//! geometry, payload size, or dispatch dependency.
//! `--resource-barriers [reps] [long_us]` instead runs the long-kernel/short-chain
//! overlap probe. See the sibling module for controls and timestamp semantics.
//! `--event-sync [reps] [event|shared|both]` prices a cross-queue handoff and join.
mod event_sync;
mod resource_barriers;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLSize};
use qwen_llm::metal::{
    Buffer, KernelEncoder, MetalContext, commit_and_wait, dispatch_profile_begin,
    dispatch_profile_finish,
};

const BUFFER_BYTES: usize = 128 * 1024 * 1024;
const NO_PAYLOAD_NS: [usize; 6] = [1, 4, 16, 64, 256, 1024];
const PAYLOAD_NS: [usize; 4] = [1, 4, 16, 64];
const REPS: usize = 5;
const PROFILE_DISPATCHES: usize = 8;
const PROFILE_REPS: usize = 3;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    grid: usize,
    threads: usize,
    words: usize,
}

impl Shape {
    fn total_threads(self) -> usize {
        self.grid * self.threads
    }

    fn payload_bytes(self) -> usize {
        self.total_threads() * self.words * std::mem::size_of::<u32>()
    }
}

const SHAPES: &[Shape] = &[
    // Threadgroup-count sweep at a fixed 128-thread group, no payload.
    Shape {
        name: "tg1x128",
        grid: 1,
        threads: 128,
        words: 0,
    },
    Shape {
        name: "tg8x128",
        grid: 8,
        threads: 128,
        words: 0,
    },
    Shape {
        name: "tg64x128",
        grid: 64,
        threads: 128,
        words: 0,
    },
    Shape {
        name: "tg384x128",
        grid: 384,
        threads: 128,
        words: 0,
    },
    Shape {
        name: "tg1536x128",
        grid: 1536,
        threads: 128,
        words: 0,
    },
    // Threads-per-threadgroup sweep at a fixed 64 groups, no payload.
    Shape {
        name: "tg64x32",
        grid: 64,
        threads: 32,
        words: 0,
    },
    Shape {
        name: "tg64x64",
        grid: 64,
        threads: 64,
        words: 0,
    },
    Shape {
        name: "tg64x256",
        grid: 64,
        threads: 256,
        words: 0,
    },
    Shape {
        name: "tg64x512",
        grid: 64,
        threads: 512,
        words: 0,
    },
    Shape {
        name: "tg64x1024",
        grid: 64,
        threads: 1024,
        words: 0,
    },
    // Payload sweep at a fixed 16x128 launch.
    Shape {
        name: "bytes0",
        grid: 16,
        threads: 128,
        words: 0,
    },
    Shape {
        name: "bytes1",
        grid: 16,
        threads: 128,
        words: 1,
    },
    Shape {
        name: "bytes4",
        grid: 16,
        threads: 128,
        words: 4,
    },
    Shape {
        name: "bytes16",
        grid: 16,
        threads: 128,
        words: 16,
    },
    Shape {
        name: "bytes64",
        grid: 16,
        threads: 128,
        words: 64,
    },
    // Shapes modelled on real decode kernels.
    Shape {
        name: "small-matvec",
        grid: 4,
        threads: 128,
        words: 1,
    },
    Shape {
        name: "norm",
        grid: 32,
        threads: 128,
        words: 1,
    },
    Shape {
        name: "reduction",
        grid: 1,
        threads: 256,
        words: 1,
    },
    Shape {
        name: "expert-wide",
        grid: 384,
        threads: 64,
        words: 4,
    },
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    total_threads: u32,
    words: u32,
    dispatch_index: u32,
}

fn encode_chain(
    ctx: &MetalContext,
    touch: &Buffer,
    pso: &qwen_llm::metal::Pipeline,
    shape: Shape,
    dispatches: usize,
    dependent: bool,
    concurrent: bool,
) -> Result<f64, Box<dyn std::error::Error>> {
    let command = ctx.queue.commandBuffer().ok_or("no command buffer")?;
    let encoder = if concurrent {
        KernelEncoder::begin_concurrent(&command)
    } else {
        KernelEncoder::begin(&command)
    };
    for dispatch_index in 0..dispatches {
        encoder.set_pipeline(pso);
        encoder.set_buffer(0, touch, 0);
        encoder.set_bytes(
            1,
            &Args {
                total_threads: shape.total_threads() as u32,
                words: shape.words as u32,
                dispatch_index: if dependent { 0 } else { dispatch_index as u32 },
            },
        );
        encoder.dispatch(
            MTLSize {
                width: shape.grid,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: shape.threads,
                height: 1,
                depth: 1,
            },
        );
    }
    encoder.end();
    commit_and_wait(&command)?;
    Ok((command.GPUEndTime() - command.GPUStartTime()) * 1000.0)
}

fn fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let covariance = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| (x - mx) * (y - my))
        .sum::<f64>();
    let variance = xs.iter().map(|x| (x - mx) * (x - mx)).sum::<f64>();
    let slope = covariance / variance;
    (slope, my - slope * mx)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn profile_window_us(
    ctx: &MetalContext,
    touch: &Buffer,
    pso: &qwen_llm::metal::Pipeline,
    shape: Shape,
    dependent: bool,
    concurrent: bool,
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut samples = Vec::new();
    for _ in 0..PROFILE_REPS {
        let guard = dispatch_profile_begin(ctx, PROFILE_DISPATCHES)?;
        encode_chain(
            ctx,
            touch,
            pso,
            shape,
            PROFILE_DISPATCHES,
            dependent,
            concurrent,
        )?;
        let report = dispatch_profile_finish(ctx, guard)?;
        if report.overflow != 0 {
            return Err(format!("dispatch profile overflow for {}", shape.name).into());
        }
        samples.extend(
            report
                .rows
                .iter()
                .filter_map(|row| row.gpu_ns)
                .map(|ns| ns as f64 / 1000.0),
        );
    }
    if samples.is_empty() {
        return Err(format!("no isolated samples for {}", shape.name).into());
    }
    Ok(median(&mut samples))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("--event-sync") {
        return event_sync::run();
    }
    if std::env::args().nth(1).as_deref() == Some("--resource-barriers") {
        return resource_barriers::run();
    }
    let ctx = MetalContext::new()?;
    let touch = ctx.buffer_uninit(BUFFER_BYTES)?;
    let dependent = ctx.pipeline("kernel_dispatch_deadtime_dependent")?;
    let independent = ctx.pipeline("kernel_dispatch_deadtime_independent")?;
    println!(
        "{}",
        serde_json::json!({"kind":"device","describe":ctx.describe()})
    );

    for shape in SHAPES {
        let ns: &[usize] = if shape.words == 0 {
            &NO_PAYLOAD_NS
        } else {
            &PAYLOAD_NS
        };
        for (mode, dependent_kernel, concurrent) in [
            ("serial-dependent", true, false),
            ("serial-independent", false, false),
            ("concurrent-independent", false, true),
        ] {
            let pso = if dependent_kernel {
                &dependent
            } else {
                &independent
            };
            let profile =
                profile_window_us(&ctx, &touch, pso, *shape, dependent_kernel, concurrent)?;
            let xs: Vec<f64> = ns.iter().map(|&n| n as f64).collect();
            let mut gpu_ms = Vec::new();
            for &dispatches in ns {
                let mut samples = Vec::with_capacity(REPS);
                for _ in 0..REPS {
                    samples.push(encode_chain(
                        &ctx,
                        &touch,
                        pso,
                        *shape,
                        dispatches,
                        dependent_kernel,
                        concurrent,
                    )?);
                }
                gpu_ms.push(median(&mut samples));
            }
            let (slope_ms, intercept_ms) = fit(&xs, &gpu_ms);
            let slope_us = slope_ms * 1000.0;
            let at64 = ns.iter().position(|&n| n == 64).unwrap();
            let at64_us = gpu_ms[at64] * 1000.0 / 64.0;
            println!(
                "{}",
                serde_json::json!({
                    "kind": "dispatch_deadtime",
                    "mode": mode,
                    "shape": shape.name,
                    "grid": shape.grid,
                    "threads": shape.threads,
                    "words": shape.words,
                    "payload_bytes": shape.payload_bytes(),
                    "dispatches": ns,
                    "gpu_ms": gpu_ms,
                    "slope_us": slope_us,
                    "intercept_us": intercept_ms * 1000.0,
                    "profile_window_us": profile,
                    "slope_minus_profile_us": slope_us - profile,
                    "at64_us": at64_us,
                    "reps": REPS,
                    "profile_reps": PROFILE_REPS,
                })
            );
        }
    }
    Ok(())
}
