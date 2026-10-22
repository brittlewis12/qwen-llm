//! Bounded test-only HC projection probe, not a production routing decision.

use super::*;
use crate::metal::checks::{bad_shape, check_disjoint};
use crate::metal::test_support::{assert_offset_guards, offset_tensor, tensor_f32_at_offset};

const K: usize = 10_240;
const M: usize = 4;
const MAX_N: usize = 2048;
const KERNEL: &str = "mat_mat_f32_hc4p32_strict_test";
const SHADER: &str = "kernel_mat_mat_f32_hc4p32_strict_test";
const GENERIC: &str = "kernel_mat_mat_f32_f32";

fn validate_dimensions(k: usize, m: usize, n: usize) -> Result<(), MetalError> {
    if k != K || m != M || !(1..=MAX_N).contains(&n) {
        return Err(bad_shape(
            KERNEL,
            format!("test probe requires K={K}, M={M}, N=1..={MAX_N}; got {k}/{m}/{n}"),
        ));
    }
    Ok(())
}

fn validate_bindings(
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
) -> Result<(), MetalError> {
    validate_dimensions(k, m, n)?;
    for (name, tensor, dims, writable) in [
        ("weight", weight, [k, m], false),
        ("input", x, [k, n], false),
        ("output", y, [m, n], true),
    ] {
        let count = checked_moe_product(KERNEL, name, &dims)?;
        validate_moe_decode_tensor(KERNEL, name, tensor, count, &[GgmlType::F32], writable, 4)?;
    }
    check_disjoint(KERNEL, y, &[(weight, "weight"), (x, "input")])
}

fn capacity_supported(width: usize, threads: usize, static_bytes: usize, limit: usize) -> bool {
    width == 32 && threads >= 32 && static_bytes <= limit
}

/// Load the embedded research shader into this context's library/cache and
/// validate capacity without encoding or submitting work. Real-model witnesses
/// can call this before measurement; the encoder also calls it on cache hits.
pub(crate) fn prepare_hc4_strict_test(ctx: &MetalContext) -> Result<Pipeline, MetalError> {
    let pso = ctx.pipeline(SHADER)?;
    if !capacity_supported(
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        pso.staticThreadgroupMemoryLength(),
        ctx.device.maxThreadgroupMemoryLength(),
    ) {
        return Err(bad_shape(
            KERNEL,
            "requires SIMD32, 32 threads and sufficient static TGM",
        ));
    }
    Ok(pso)
}

fn require_gpu_workflow() {
    for name in ["MTL_DEBUG_LAYER", "QWEN_METAL_LEASE_WAIT"] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok("1"),
            "HC4 GPU tests require {name}=1 before process startup"
        );
    }
}

/// Flat contiguous F32 views are accepted; output must not overlap either input.
/// All refusals precede encoder mutation. Bounds also keep the generic comparator's
/// uint indexing and 32-wide K loop well below overflow.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mat_mat_f32_hc4p32_strict_test(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
) -> Result<(), MetalError> {
    validate_bindings(weight, x, y, k, m, n)?;
    let device = ctx.device.registryID();
    if enc.parent_command_buffer().device().registryID() != device
        || [weight, x, y]
            .into_iter()
            .any(|t| t.buffer.device().registryID() != device)
    {
        return Err(bad_shape(
            KERNEL,
            "encoder/bindings must use the context device",
        ));
    }
    let pso = prepare_hc4_strict_test(ctx)?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        k: u32,
        m: u32,
        n: u32,
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            k: k as u32,
            m: m as u32,
            n: n as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: n.div_ceil(32),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

struct Fixture {
    weight: MetalTensor,
    x: MetalTensor,
    generic: MetalTensor,
    candidate: MetalTensor,
    n: usize,
    prefixes: [usize; 4],
}

impl Fixture {
    fn new(ctx: &MetalContext, n: usize, cancellation: bool, offsets: bool) -> Self {
        validate_dimensions(K, M, n).unwrap();
        // Scalar-aligned but deliberately not float4-aligned views.
        let prefixes = if offsets { [4, 12, 20, 28] } else { [0; 4] };
        let counts = [K * M, K * n, M * n, M * n];
        let bytes = std::array::from_fn::<_, 4, _>(|i| prefixes[i] + counts[i] * 4 + 20);
        let priced = bytes
            .iter()
            .try_fold(0u64, |total, &b| {
                total.checked_add(
                    ctx.price_shared_buffer_upper(b as u64)
                        .unwrap()
                        .priced_upper_bytes,
                )
            })
            .unwrap();
        // Compile before admission/measurement. No tensor allocation bypasses
        // the normal transaction, device pricing or dynamic reserve.
        prepare_hc4_strict_test(ctx).unwrap();
        let generic = ctx.pipeline(GENERIC).unwrap();
        assert!(capacity_supported(
            generic.threadExecutionWidth(),
            generic.maxTotalThreadsPerThreadgroup(),
            generic
                .staticThreadgroupMemoryLength()
                .checked_add(128)
                .unwrap(),
            ctx.device.maxThreadgroupMemoryLength(),
        ));
        let _allocation = ctx.begin_allocation_transaction();
        let before = ctx.memory_signals();
        let reserve = crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES;
        // Bound host vectors, offset_tensor's staging/reallocation peak, and
        // readback copies; CPU-only bytes do not consume the Metal headroom.
        let cpu_bytes = bytes
            .iter()
            .map(|&b| b as u64)
            .sum::<u64>()
            .checked_mul(4)
            .unwrap();
        let admission = evaluate_metal_memory_admission_with_cpu_bytes(
            priced, cpu_bytes, reserve, before, true,
        );
        assert!(admission.is_admitted(), "HC4 admission: {admission:?}");
        let mut state = 0x8b8b_8b8bu32;
        let mut sample = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 * (1.0 / 16_777_216.0) - 0.5) * 0.25
        };
        let weight: Vec<f32> = (0..counts[0])
            .map(|i| {
                if cancellation {
                    let row = i / K;
                    let scale = [1.0, -1.0, 0.5, -2.0][row];
                    scale
                        * [
                            16_777_216.0,
                            1.0,
                            -16_777_216.0,
                            1.0,
                            1.000_000_1,
                            -1.0,
                            0.000_000_119_209_29,
                            -0.5,
                        ][i % 8]
                } else {
                    sample()
                }
            })
            .collect();
        let x: Vec<f32> = (0..counts[1])
            .map(|i| {
                if cancellation {
                    let q = i / K;
                    let scale = [1.0, -1.0, 0.5, 2.0, 1.000_000_1][q % 5];
                    scale * [1.0, 1.0, 1.0, 1.0, 0.999_999_94, 1.0, 1.0, 1.0][i % 8]
                } else {
                    sample()
                }
            })
            .collect();
        let make = |i: usize, values: &[f32]| {
            offset_tensor(
                ctx,
                prefixes[i],
                bytemuck::cast_slice(values),
                20,
                vec![counts[i] as u64],
                GgmlType::F32,
            )
        };
        let f = Self {
            weight: make(0, &weight),
            x: make(1, &x),
            generic: make(2, &vec![f32::NAN; counts[2]]),
            candidate: make(3, &vec![f32::NAN; counts[3]]),
            n,
            prefixes,
        };
        let after = ctx.memory_signals();
        assert!(
            after
                .current_allocated_bytes
                .saturating_sub(before.current_allocated_bytes)
                <= priced
        );
        assert!(evaluate_metal_memory_admission(0, reserve, after, true).is_admitted());
        f
    }

    fn encode(&self, ctx: &MetalContext, enc: &KernelEncoder, candidate: bool, same_output: bool) {
        let y = if candidate && !same_output {
            &self.candidate
        } else {
            &self.generic
        };
        if candidate {
            encode_mat_mat_f32_hc4p32_strict_test(ctx, enc, &self.weight, &self.x, y, K, M, self.n)
                .unwrap();
        } else {
            validate_bindings(&self.weight, &self.x, y, K, M, self.n).unwrap();
            encode_mat_mat_f32(ctx, enc, &self.weight, &self.x, y, K, M, self.n).unwrap();
        }
    }

    fn check(&self, ctx: &MetalContext) {
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        self.encode(ctx, &enc, false, false);
        self.encode(ctx, &enc, true, false);
        enc.end();
        commit_and_wait(&cmd).unwrap();
        let generic = tensor_f32_at_offset(&self.generic);
        let candidate = tensor_f32_at_offset(&self.candidate);
        for (i, (a, b)) in generic.iter().zip(&candidate).enumerate() {
            assert!(
                a.is_finite() && b.is_finite(),
                "N={} output {i} nonfinite",
                self.n
            );
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "N={} output {i}: generic={a:?} candidate={b:?}",
                self.n
            );
        }
        for (i, t) in [&self.weight, &self.x, &self.generic, &self.candidate]
            .into_iter()
            .enumerate()
        {
            assert_offset_guards(t, self.prefixes[i], 20);
        }
    }
}

#[test]
fn hc4_strict_cpu_contracts() {
    for n in [1, 2, 7, 31, 32, 33, 128, 512, 2048] {
        validate_dimensions(K, M, n).unwrap();
    }
    for (k, m, n) in [
        (0, M, 1),
        (K - 1, M, 1),
        (K, 0, 1),
        (K, 8, 1),
        (K, M, 0),
        (K, M, MAX_N + 1),
        (usize::MAX, M, 1),
        (K, usize::MAX, 1),
        (K, M, usize::MAX),
        (K, M, u32::MAX as usize),
    ] {
        assert!(validate_dimensions(k, m, n).is_err());
    }
    assert!(checked_moe_product(KERNEL, "overflow", &[usize::MAX, 4]).is_err());
    assert!(checked_ggml_shape_bytes(&[u64::MAX, 2], GgmlType::F32).is_err());
    assert!(checked_ggml_shape_bytes(&[u64::MAX / 4 + 1], GgmlType::F32).is_err());
    assert!(capacity_supported(32, 32, 0, 0));
    assert!(!capacity_supported(16, 32, 0, 0));
    assert!(!capacity_supported(32, 31, 0, 0));
    assert!(!capacity_supported(32, 32, 1, 0));
}

#[test]
#[ignore = "research GPU witness; requires validation and production lease"]
fn hc4_strict_rejects_invalid_bindings() {
    require_gpu_workflow();
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    objc2::rc::autoreleasepool(|_| {
        let f = Fixture::new(&ctx, 1, false, true);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        for slot in 0..3 {
            for fault in 0..7 {
                let mut bindings = [f.weight.clone(), f.x.clone(), f.candidate.clone()];
                let t = &mut bindings[slot];
                match fault {
                    0 => t.dtype = GgmlType::F16,
                    1 => t.shape = vec![t.n_elements() - 1],
                    2 => t.shape = vec![u64::MAX, 2],
                    3 => t.offset += 1,
                    4 => t.offset = t.buffer.length() as u64,
                    5 => t.offset = u64::MAX - 3,
                    6 => t.shape = vec![u64::MAX / 4 + 1],
                    _ => unreachable!(),
                }
                assert!(
                    encode_mat_mat_f32_hc4p32_strict_test(
                        &ctx,
                        &enc,
                        &bindings[0],
                        &bindings[1],
                        &bindings[2],
                        K,
                        M,
                        1,
                    )
                    .is_err(),
                    "slot={slot} fault={fault}"
                );
            }
        }
        for source in [&f.weight, &f.x] {
            let aliased = source.view_subrange(0, vec![M as u64]);
            assert!(
                encode_mat_mat_f32_hc4p32_strict_test(
                    &ctx, &enc, &f.weight, &f.x, &aliased, K, M, 1,
                )
                .is_err()
            );
        }
        let readonly = MetalTensor {
            provenance: MetalTensorProvenance::RetainedGgufReadOnly,
            ..f.candidate.clone()
        };
        assert!(
            encode_mat_mat_f32_hc4p32_strict_test(&ctx, &enc, &f.weight, &f.x, &readonly, K, M, 1,)
                .is_err()
        );
        assert!(enc.bound_pipeline_name.borrow().is_none());
        enc.end(); // Invalid calls are never submitted.
    });
}

#[test]
#[ignore = "research GPU witness; requires validation and production lease"]
fn hc4_strict_matches_generic_bits() {
    require_gpu_workflow();
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for cancellation in [false, true] {
        for offsets in [false, true] {
            for n in [1, 2, 7, 31, 32, 33, 128, 512, 2048] {
                objc2::rc::autoreleasepool(|_| {
                    let f = Fixture::new(&ctx, n, cancellation, offsets);
                    f.check(&ctx);
                    eprintln!(
                        "HC4 bits passed K={K} M={M} N={n} cancellation={cancellation} offsets={offsets}"
                    );
                });
            }
        }
    }
}

#[test]
#[ignore = "validation-on resident GPU timing; requires MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1"]
fn hc4_strict_gpu_timing() {
    require_gpu_workflow();
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = crate::test_fixtures::metal_context_or_skip().expect("timing requires Metal");
    const LAUNCHES: usize = 4;
    eprintln!(
        "HC4 validation-on timing: MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1. Resident synthetic projection only; GPU command us/launch, not whole-model gain or validation-off performance. Same bindings in both arms; compilation/allocation/readback excluded; all samples retained."
    );
    for n in [128, 512, 2048] {
        objc2::rc::autoreleasepool(|_| {
            let f = Fixture::new(&ctx, n, false, false);
            f.check(&ctx);
            let timed = |candidate| {
                objc2::rc::autoreleasepool(|_| {
                    assert!(!dispatch_census_is_active());
                    let cmd = ctx.queue.commandBuffer().unwrap();
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..LAUNCHES {
                        f.encode(&ctx, &enc, candidate, true);
                    }
                    enc.end();
                    commit_and_wait(&cmd).unwrap();
                    let (start, end) = (cmd.GPUStartTime(), cmd.GPUEndTime());
                    assert!(
                        start.is_finite() && end.is_finite() && start > 0.0 && end > start,
                        "GPU timestamps unavailable"
                    );
                    (end - start) * 1e6 / LAUNCHES as f64
                })
            };
            for _ in 0..2 {
                for arm in [false, true, true, false] {
                    timed(arm);
                }
            }
            let mut samples = [Vec::new(), Vec::new()];
            for repeat in 0..6 {
                let order = if repeat % 2 == 0 {
                    [false, true, true, false]
                } else {
                    [true, false, false, true]
                };
                for (slot, arm) in order.into_iter().enumerate() {
                    let us = timed(arm);
                    samples[usize::from(arm)].push(us);
                    eprintln!(
                        "HC4 validation=on N={n} repeat={repeat} slot={slot} candidate={arm} gpu_us={us:.3}"
                    );
                }
            }
            let medians = samples.map(|mut times| {
                times.sort_by(f64::total_cmp);
                (times[times.len() / 2 - 1] + times[times.len() / 2]) * 0.5
            });
            eprintln!(
                "HC4 validation=on K={K} M={M} N={n} launches={LAUNCHES} median_generic_us={:.3} median_candidate_us={:.3} speedup={:.3}",
                medians[0],
                medians[1],
                medians[0] / medians[1]
            );
            f.check(&ctx);
        });
    }
}
