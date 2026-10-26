//! Opt-in probe of the GPU cost of a chain of small dispatches, the shape of
//! a decode step (~1,300 dispatches per token). Prints per-dispatch GPU time;
//! asserts nothing. `QWEN_DISPATCH_LATENCY=1`.
//!
//! Modes, each `N` copies of `len` floats inside one command buffer:
//! - serial: one serial encoder, each copy reading the previous one's output
//!   (the decode step's structure today);
//! - concurrent+barrier: one concurrent encoder with a buffer memory barrier
//!   after every dispatch (same dependencies, different ordering mechanism);
//! - concurrent independent: one concurrent encoder, every copy into its own
//!   slice, no barriers (the overlap ceiling for independent work);
//! - serial independent: the same independent copies in a serial encoder.

use super::*;
use objc2_metal::{MTLBarrierScope, MTLComputeCommandEncoder};

fn timed(
    ctx: &MetalContext,
    samples: usize,
    encode: &dyn Fn(
        &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    ),
) -> f64 {
    let mut us = Vec::with_capacity(samples);
    for _ in 0..samples {
        let command = ctx.queue.commandBuffer().unwrap();
        encode(&command);
        commit_and_wait(&command).unwrap();
        us.push((command.GPUEndTime() - command.GPUStartTime()) * 1e6);
    }
    us.sort_by(f64::total_cmp);
    us[samples / 2]
}

#[test]
fn dispatch_chain_latency_timing() {
    if std::env::var_os("QWEN_DISPATCH_LATENCY").is_none() {
        return;
    }
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = MetalContext::new().expect("Metal");
    const N: usize = 1300;
    for len in [256usize, 5120, 65536] {
        let a = MetalTensor::zeros_f32(&ctx, vec![len as u64]).unwrap();
        let b = MetalTensor::zeros_f32(&ctx, vec![len as u64]).unwrap();
        let slab = MetalTensor::zeros_f32(&ctx, vec![(len * N) as u64]).unwrap();
        let serial = |command: &_| {
            let enc = KernelEncoder::begin(command);
            for i in 0..N {
                let (src, dst) = if i % 2 == 0 { (&a, &b) } else { (&b, &a) };
                encode_copy_offset_f32(&ctx, &enc, src, 0, dst, len).unwrap();
            }
            enc.end();
        };
        let concurrent_barrier = |command: &_| {
            let enc = KernelEncoder::begin_concurrent(command);
            for i in 0..N {
                let (src, dst) = if i % 2 == 0 { (&a, &b) } else { (&b, &a) };
                encode_copy_offset_f32(&ctx, &enc, src, 0, dst, len).unwrap();
                enc.raw
                    .borrow()
                    .memoryBarrierWithScope(MTLBarrierScope::Buffers);
            }
            enc.end();
        };
        let concurrent_independent = |command: &_| {
            let enc = KernelEncoder::begin_concurrent(command);
            for i in 0..N {
                let dst = slab.view_subrange((i * len) as u64, vec![len as u64]);
                encode_copy_offset_f32(&ctx, &enc, &a, 0, &dst, len).unwrap();
            }
            enc.end();
        };
        let serial_independent = |command: &_| {
            let enc = KernelEncoder::begin(command);
            for i in 0..N {
                let dst = slab.view_subrange((i * len) as u64, vec![len as u64]);
                encode_copy_offset_f32(&ctx, &enc, &a, 0, &dst, len).unwrap();
            }
            enc.end();
        };
        // Heat: the first command buffers after CPU work read slow.
        for _ in 0..4 {
            timed(&ctx, 1, &serial);
            timed(&ctx, 1, &concurrent_barrier);
        }
        for round in 0..2 {
            for (name, f) in [
                ("serial dependent", &serial as &dyn Fn(&_)),
                ("concurrent+barrier dependent", &concurrent_barrier),
                ("concurrent independent", &concurrent_independent),
                ("serial independent", &serial_independent),
            ] {
                let us = timed(&ctx, 7, f);
                eprintln!(
                    "dispatch_latency len={len:<6} {name:<30} r{round} total {us:9.1} us  per-dispatch {:6.2} us",
                    us / N as f64
                );
            }
        }
    }
}

/// Marginal cost of a tiny dependent dispatch inside a chain that also does
/// real work (decode interleaves big GEMVs with tiny norm/RoPE/HC kernels):
/// a chain of heavy copies, the same chain with a tiny copy after each heavy
/// one, and the difference per tiny dispatch. `QWEN_DISPATCH_LATENCY=1`.
#[test]
fn dispatch_chain_tiny_marginal_timing() {
    if std::env::var_os("QWEN_DISPATCH_LATENCY").is_none() {
        return;
    }
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = MetalContext::new().expect("Metal");
    const PAIRS: usize = 650;
    const HEAVY: usize = 1 << 20; // 4 MiB per copy, ~8 us at full bandwidth
    const TINY: usize = 256;
    let a = MetalTensor::zeros_f32(&ctx, vec![HEAVY as u64]).unwrap();
    let b = MetalTensor::zeros_f32(&ctx, vec![HEAVY as u64]).unwrap();
    let t = MetalTensor::zeros_f32(&ctx, vec![TINY as u64]).unwrap();
    let heavy_only = |command: &_| {
        let enc = KernelEncoder::begin(command);
        for i in 0..PAIRS {
            let (src, dst) = if i % 2 == 0 { (&a, &b) } else { (&b, &a) };
            encode_copy_offset_f32(&ctx, &enc, src, 0, dst, HEAVY).unwrap();
        }
        enc.end();
    };
    let heavy_tiny = |command: &_| {
        let enc = KernelEncoder::begin(command);
        for i in 0..PAIRS {
            let (src, dst) = if i % 2 == 0 { (&a, &b) } else { (&b, &a) };
            encode_copy_offset_f32(&ctx, &enc, src, 0, dst, HEAVY).unwrap();
            encode_copy_offset_f32(&ctx, &enc, dst, 0, &t, TINY).unwrap();
        }
        enc.end();
    };
    for _ in 0..4 {
        timed(&ctx, 1, &heavy_only);
        timed(&ctx, 1, &heavy_tiny);
    }
    for round in 0..3 {
        let h = timed(&ctx, 7, &heavy_only);
        let ht = timed(&ctx, 7, &heavy_tiny);
        eprintln!(
            "dispatch_tiny_marginal r{round} heavy-only {h:9.1} us ({:5.2} us/heavy)  heavy+tiny {ht:9.1} us  marginal per tiny {:5.2} us",
            h / PAIRS as f64,
            (ht - h) / PAIRS as f64
        );
    }
}
