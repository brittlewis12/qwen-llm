//! CPU policy/contract tests and opt-in primitive witnesses, not model gates.
//! GPU tests require MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1, an explicit
//! production lease, and a release test binary. No timing here.

use super::*;
use crate::glm5_next_metal::packed;
use crate::metal::{
    MetalTensorProvenance, dispatch_census_begin, dispatch_census_take, wait_completed,
};
use crate::tensor::TensorDesc;
use objc2_metal::{MTLCommandQueue, MTLGPUFamily};

const HALF: Arith = Arith {
    lineage: PackedLineage::Fast,
    precision: FastPrecision::Half,
};

#[test]
fn cpu_policy_requires_ordinary_half_fast_resolved_mode_and_native_types() {
    let _scope = Scope::new(true);
    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        for precision in FastPrecision::ALL {
            for mode in [StageMode::Fast, StageMode::FastF32, StageMode::Exact] {
                for dtype in [
                    GgmlType::Q6_K,
                    GgmlType::Q8_0,
                    GgmlType::F32,
                    GgmlType::F16,
                    GgmlType::BF16,
                    GgmlType::Q4_K,
                    GgmlType::IQ3_XXS,
                    GgmlType::Unknown,
                ] {
                    let arith = Arith { lineage, precision };
                    assert_eq!(
                        eligible(arith, mode, dtype, 4096, 1536, 1024).unwrap(),
                        arith == HALF
                            && mode == StageMode::Fast
                            && matches!(dtype, GgmlType::Q6_K | GgmlType::Q8_0),
                        "{arith:?}/{mode:?}/{dtype:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn cpu_policy_row_boundary_matches_production_not_q6_test_n64_threshold() {
    let _scope = Scope::new(true);
    for dtype in [GgmlType::Q6_K, GgmlType::Q8_0] {
        for n in [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 32, 64, 512, 1023, 1024, 1025, 2048,
        ] {
            assert_eq!(
                eligible(HALF, StageMode::Fast, dtype, 4096, 65, n).unwrap(),
                n >= 1024
            );
        }
    }
}

#[test]
fn cpu_geometry_rejects_zero_partial_blocks_and_address_overflow() {
    for dtype in [GgmlType::Q6_K, GgmlType::Q8_0] {
        for (k, m, n) in [
            (0, 65, 1024),
            (256, 0, 1024),
            (256, 65, 0),
            (255, 65, 1024),
            (256, usize::MAX, 1024),
            (usize::MAX, 65, 1024),
            (256, 65, usize::MAX),
            (256, i32::MAX as usize + 1, 1024),
            (256, 65, i32::MAX as usize + 1),
        ] {
            assert!(geometry(dtype, k, m, n).is_err(), "{dtype:?}/{k}/{m}/{n}");
        }
    }
    assert!(geometry(GgmlType::Q6_K, 32, 65, 1024).is_err());
    assert!(geometry(GgmlType::Q8_0, 32, 65, 1024).is_ok());
    // K itself fits u32; its Q8 row-byte stride does not.
    assert!(geometry(GgmlType::Q8_0, u32::MAX as usize / 32 * 32, 1, 1024).is_err());
    // Individually valid shader arguments can still overflow physical bytes.
    assert!(
        geometry(
            GgmlType::Q6_K,
            u32::MAX as usize / 256 * 256,
            i32::MAX as usize,
            i32::MAX as usize
        )
        .is_err()
    );
    let _scope = Scope::new(true);
    assert!(eligible(HALF, StageMode::Fast, GgmlType::Q6_K, usize::MAX, 65, 1024).is_err());
    // Ineligible old paths do not gain new validation or change behavior.
    assert!(!eligible(HALF, StageMode::Fast, GgmlType::Q6_K, usize::MAX, 0, 8).unwrap());
}

#[test]
fn cpu_geometry_uses_storage_layout_at_actual_shapes_and_genuine_m_tail() {
    for dtype in [GgmlType::Q6_K, GgmlType::Q8_0] {
        let (block, bytes) = dtype.storage_layout().unwrap();
        for (k, m) in [
            (4096, 8192),
            (8192, 4096),
            (1536, 16384),
            (4096, 1536),
            (256, 65),
        ] {
            for n in [1024, 2048] {
                assert_eq!(
                    geometry(dtype, k, m, n).unwrap(),
                    Geometry {
                        weight_bytes: k / block as usize * bytes as usize * m,
                        input_elements: k * n,
                        output_elements: m * n,
                    }
                );
            }
        }
    }
}

#[test]
fn cpu_scope_is_default_on_nested_unwind_safe_and_thread_local() {
    let selected = || eligible(HALF, StageMode::Fast, GgmlType::Q6_K, 4096, 1536, 1024).unwrap();
    assert!(selected());
    with_override(false, || {
        assert!(!selected());
        with_override(true, || assert!(selected()));
        assert!(!selected());
        let unwind = std::panic::catch_unwind(|| {
            let _scope = Scope::new(true);
            assert!(selected());
            panic!("scope restoration");
        });
        assert!(unwind.is_err());
        assert!(!selected());
        std::thread::spawn(|| {
            assert!(enabled());
            with_override(false, || assert!(!enabled()));
            assert!(enabled());
        })
        .join()
        .unwrap();
        assert!(!selected());
    });
    assert!(selected());
}

#[test]
fn cpu_pipeline_capacity_includes_static_plus_eight_kib_dynamic_memory() {
    assert!(capacity_supported(32, 128, 0, 8192));
    assert!(capacity_supported(32, 256, 512, 8704));
    for (width, threads, static_bytes, limit) in [
        (16, 128, 0, 8192),
        (32, 127, 0, 8192),
        (32, 128, 0, 8191),
        (32, 128, 1, 8192),
        (32, 128, usize::MAX, usize::MAX),
    ] {
        assert!(!capacity_supported(width, threads, static_bytes, limit));
    }
}

fn validation_lease() -> impl Sized {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    crate::metal::acquire_metal_benchmark_lease().expect("explicit production GPU lease")
}

fn require_m5(ctx: &MetalContext) {
    assert!(
        ctx.device.name().to_string().contains("M5"),
        "requires actual M5 hardware"
    );
    assert!(ctx.device.supportsFamily(MTLGPUFamily::Apple10));
    assert!(
        ctx.tensor_ops_status().is_available(),
        "{:?}",
        ctx.tensor_ops_status()
    );
    assert!(
        metal::moe_grouped_tensor_selected(ctx),
        "global tensor rollback must be ON"
    );
}

const GUARD: usize = 256;
const WEIGHT_ROWS: usize = 7;
const INPUT_ROWS: usize = 13;

struct Fixture {
    w: MetalTensor,
    x: MetalTensor,
    y: MetalTensor,
    full: Vec<f64>,
    half: Vec<f64>,
    k: usize,
    m: usize,
    n: usize,
}

fn guarded(ctx: &MetalContext, payload: &[u8], shape: Vec<u64>, dtype: GgmlType) -> MetalTensor {
    let mut bytes: Vec<u8> = vec![0xa5u8; GUARD];
    bytes.extend_from_slice(payload);
    bytes.resize(bytes.len() + GUARD, 0x5au8);
    MetalTensor {
        buffer: ctx.buffer_from(&bytes).unwrap(),
        offset: GUARD as u64,
        shape,
        dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn assert_guards(t: &MetalTensor) {
    let bytes = unsafe {
        std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<u8>(), t.buffer.length())
    };
    assert!(bytes[..GUARD].iter().all(|&b| b == 0xa5u8));
    assert!(
        bytes[GUARD + t.n_bytes() as usize..]
            .iter()
            .all(|&b| b == 0x5au8)
    );
}

fn read(t: &MetalTensor) -> Vec<f32> {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<f32>(),
            t.n_elements() as usize,
        )
        .to_vec()
    }
}

impl Fixture {
    fn new(ctx: &MetalContext, dtype: GgmlType, k: usize, m: usize, n: usize) -> Self {
        let _allocation = ctx.begin_allocation_transaction();
        let g = geometry(dtype, k, m, n).unwrap();
        let logical = [g.weight_bytes, g.input_elements * 4, g.output_elements * 4];
        let priced: u64 = logical
            .iter()
            .map(|&b| {
                ctx.price_shared_buffer_upper((b + 2 * GUARD) as u64)
                    .unwrap()
                    .priced_upper_bytes
            })
            .sum();
        // Native upload copies, readbacks, compact CPU dequant/oracles and
        // retained A output are priced separately from the device buffers.
        let cpu = 4 * logical.iter().sum::<usize>() as u64 + (16 << 20);
        // Charge CPU payload against both budgets, including when macOS omits
        // the process limit (the production gate represents that as Some(0)).
        let admission = metal::evaluate_metal_memory_admission(
            priced.checked_add(cpu).unwrap(),
            crate::glm5_next::memory::DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        assert!(admission.is_admitted(), "{admission:?}");

        // Native finite GGML block recipes as in the generic tile fixtures.
        // Only seven distinct weight rows are dequantized, never a full F32 W.
        let (block, block_bytes) = dtype.storage_layout().unwrap();
        let row_bytes = k / block as usize * block_bytes as usize;
        let mut compact = vec![0u8; WEIGHT_ROWS * row_bytes];
        let mut rng = 0x1234_5678u32;
        for (ordinal, b) in compact.chunks_exact_mut(block_bytes as usize).enumerate() {
            for byte in b.iter_mut() {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                *byte = (rng >> 24) as u8;
            }
            let scale = half::f16::from_f32((1 + ordinal % 7) as f32 * 0.000113);
            let offset = if dtype == GgmlType::Q6_K { 208 } else { 0 };
            b[offset..offset + 2].copy_from_slice(&scale.to_le_bytes());
        }
        let desc = TensorDesc {
            name: "glm_tensor_compact_native".into(),
            shape: vec![k as u64, WEIGHT_ROWS as u64],
            dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: compact.len() as u64,
        };
        let decoded = crate::codec::dequant_to_f32(&desc, &compact).unwrap();
        let mut payload = Vec::with_capacity(g.weight_bytes);
        for row in 0..m {
            payload.extend_from_slice(
                &compact[(row % WEIGHT_ROWS) * row_bytes..(row % WEIGHT_ROWS + 1) * row_bytes],
            );
        }
        let mut w = guarded(ctx, &payload, vec![k as u64, m as u64], dtype);
        w.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        drop(payload);
        let inputs: Vec<f32> = (0..INPUT_ROWS * k)
            .map(|i| {
                let row = i / k;
                (((i % k * 37 + row * 17) % 211) as f32 - 105.0) * 0.003713
            })
            .collect();
        let mut payload: Vec<u8> = Vec::with_capacity(g.input_elements * 4);
        for row in 0..n {
            payload.extend_from_slice(bytemuck::cast_slice(
                &inputs[(row % INPUT_ROWS) * k..(row % INPUT_ROWS + 1) * k],
            ));
        }
        let x = guarded(ctx, &payload, vec![k as u64, n as u64], GgmlType::F32);
        drop(payload);
        let y = guarded(
            ctx,
            &vec![0xffu8; g.output_elements * 4],
            vec![m as u64, n as u64],
            GgmlType::F32,
        );
        let half = |v: f32| f64::from(half::f16::from_f32(v).to_f32());
        let mut full_ref = Vec::new();
        let mut half_ref = Vec::new();
        for row in 0..INPUT_ROWS {
            for out in 0..WEIGHT_ROWS {
                full_ref.push(
                    (0..k)
                        .map(|i| f64::from(decoded[out * k + i]) * f64::from(inputs[row * k + i]))
                        .sum(),
                );
                half_ref.push(
                    (0..k)
                        .map(|i| half(decoded[out * k + i]) * half(inputs[row * k + i]))
                        .sum(),
                );
            }
        }
        Self {
            w,
            x,
            y,
            full: full_ref,
            half: half_ref,
            k,
            m,
            n,
        }
    }

    fn encode(&self, ctx: &MetalContext, enc: &KernelEncoder, candidate: bool) {
        with_override(candidate, || {
            packed::matmat(
                ctx,
                enc,
                HALF,
                packed::Stage::DenseFfn,
                &self.w,
                &self.x,
                &self.y,
                self.k,
                self.m,
                self.n,
            )
            .unwrap()
        });
    }

    fn poison_output(&self) {
        assert!(self.y.is_writable());
        unsafe {
            self.y
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(self.y.offset as usize)
                .write_bytes(0xffu8, self.y.n_bytes() as usize);
        }
    }

    fn validate(&self, actual: &[f32], reference: &[f64], relative: f64) {
        assert_eq!(actual.len(), self.m * self.n);
        assert!(reference.iter().all(|v| v.is_finite()));
        let scale = reference.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(scale > 1e-5, "degenerate fixture");
        let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
        for (i, &a) in actual.iter().enumerate() {
            let b = reference[(i / self.m % INPUT_ROWS) * WEIGHT_ROWS + i % self.m % WEIGHT_ROWS];
            let a = f64::from(a);
            assert!(
                a.is_finite() && (a - b).abs() <= relative * scale + 1e-4,
                "{:?} K{} M{} N{} [{i}]: {a} vs {b}, scale {scale}",
                self.w.dtype,
                self.k,
                self.m,
                self.n
            );
            dot += a * b;
            aa += a * a;
            bb += b * b;
        }
        assert!(dot / (aa * bb).sqrt() > 0.999);
        for t in [&self.w, &self.x, &self.y] {
            assert_guards(t);
        }
    }
}

fn run(ctx: &MetalContext, body: impl FnOnce(&KernelEncoder)) -> Vec<metal::DispatchCensusRow> {
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    dispatch_census_begin();
    body(&enc);
    let census = dispatch_census_take();
    enc.end();
    command.commit();
    wait_completed(&command).unwrap();
    census
}

#[test]
#[ignore = "actual M5 + tensor tier, validation ON, production lease"]
fn m5_glm_shapes_baseline_vs_tensor_and_f64_oracles() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().expect("Metal is required, not skipped");
    require_m5(&ctx);
    for dtype in [GgmlType::Q6_K, GgmlType::Q8_0] {
        for (k, m) in [
            (4096, 8192),
            (8192, 4096),
            (1536, 16384),
            (4096, 1536),
            (256, 65),
        ] {
            let counts: &[usize] = if m == 65 {
                &[1024, 1025, 2048]
            } else {
                &[1024, 2048]
            };
            for &n in counts {
                let f = Fixture::new(&ctx, dtype, k, m, n);
                assert!(prepare(&ctx, dtype).unwrap());
                let baseline = run(&ctx, |enc| f.encode(&ctx, enc, false));
                assert_eq!(baseline.len(), 1);
                let expected = match dtype {
                    GgmlType::Q6_K if m.is_multiple_of(64) => "kernel_mat_mat_q6_K_f32_n64",
                    GgmlType::Q6_K => "kernel_mat_mat_q6_K_f32",
                    _ => "kernel_mat_mat_q8_0_f32",
                };
                assert_eq!(
                    baseline[0].kernel, expected,
                    "must use actual production-selected baseline"
                );
                let a = read(&f.y);
                f.validate(&a, &f.full, 1e-2);
                f.poison_output();
                let candidate = run(&ctx, |enc| f.encode(&ctx, enc, true));
                assert_eq!(candidate.len(), 1);
                assert_eq!(
                    candidate[0].kernel,
                    metal::mat_mat_tensor_pipeline_name(dtype, metal::TensorActivation::F32).unwrap()
                );
                assert_eq!(
                    [
                        candidate[0].grid_width,
                        candidate[0].grid_height,
                        candidate[0].grid_depth
                    ],
                    [n.div_ceil(32) as u64, m.div_ceil(64) as u64, 1]
                );
                assert_eq!(candidate[0].threads_width, 128);
                let b = read(&f.y);
                f.validate(&b, &f.full, 1e-2);
                f.validate(&b, &f.half, 2e-3);
                let scale = a.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
                assert!(
                    a.iter()
                        .zip(&b)
                        .all(|(&a, &b)| (a - b).abs() <= 2e-3 * scale + 1e-5)
                );
            }
        }
    }
}

#[test]
#[ignore = "actual M5, validation ON, production lease; binding failures must precede encoder mutation"]
fn m5_invalid_bindings_leave_encoder_and_output_untouched() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().unwrap();
    require_m5(&ctx);
    let f = Fixture::new(&ctx, GgmlType::Q6_K, 256, 65, 1024);
    let mut readonly = f.y.clone();
    readonly.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    let mut short = f.y.clone();
    short.shape[1] -= 1;
    let mut outside = f.y.clone();
    outside.offset = outside.buffer.length() as u64;
    let mut overflow = f.y.clone();
    overflow.offset = u64::MAX - 3;
    let mut shape_overflow = f.y.clone();
    shape_overflow.shape = vec![u64::MAX, u64::MAX];
    let mut misaligned = f.x.clone();
    misaligned.offset += 4;
    let mut eight_aligned = f.x.clone();
    eight_aligned.offset += 8;
    let mut weight_unaligned = f.w.clone();
    weight_unaligned.offset += 2;
    let mut weight_alias = f.w.clone();
    weight_alias.buffer = f.y.buffer.clone();
    let mut wrong_dtype = f.x.clone();
    wrong_dtype.dtype = GgmlType::F16;
    wrong_dtype.shape[1] *= 2;
    let mut output_unaligned = f.y.clone();
    output_unaligned.offset += 1;
    let mut alias = f.y.clone();
    alias.buffer = f.x.buffer.clone();
    let cases = [
        (&f.w, &f.x, &readonly),
        (&f.w, &f.x, &short),
        (&f.w, &f.x, &outside),
        (&f.w, &f.x, &overflow),
        (&f.w, &f.x, &shape_overflow),
        (&f.w, &misaligned, &f.y),
        (&f.w, &eight_aligned, &f.y),
        (&weight_unaligned, &f.x, &f.y),
        (&weight_alias, &f.x, &f.y),
        (&f.w, &wrong_dtype, &f.y),
        (&f.w, &f.x, &output_unaligned),
        (&f.w, &f.x, &alias),
    ];
    let _scope = Scope::new(true);
    for (w, x, y) in cases {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        dispatch_census_begin();
        assert!(try_encode(&ctx, &enc, HALF, StageMode::Fast, w, x, y, f.k, f.m, f.n).is_err());
        assert!(
            packed::matmat(
                &ctx,
                &enc,
                HALF,
                packed::Stage::DenseFfn,
                w,
                x,
                y,
                f.k,
                f.m,
                f.n
            )
            .is_err()
        );
        assert!(dispatch_census_take().is_empty());
        assert!(enc.bound_pipeline_name.borrow().is_none());
        enc.end();
        // Deliberately do not commit failed encodes.
    }
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin_concurrent(&command);
    assert!(
        try_encode(
            &ctx,
            &enc,
            HALF,
            StageMode::Fast,
            &f.w,
            &f.x,
            &f.y,
            f.k,
            f.m,
            f.n
        )
        .is_err()
    );
    assert!(enc.bound_pipeline_name.borrow().is_none());
    enc.end();
    assert!(read(&f.y).iter().all(|v| v.to_bits() == u32::MAX));
    for t in [&f.w, &f.x, &f.y] {
        assert_guards(t);
    }
}

#[test]
#[ignore = "actual M4 separate fallback witness, validation ON, production lease"]
fn m4_declines_without_mutation_then_runs_baseline() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().unwrap();
    assert!(
        ctx.device.name().to_string().contains("M4"),
        "requires actual M4 hardware"
    );
    assert!(
        matches!(
            ctx.tensor_ops_status(),
            metal::TensorOpsStatus::Unavailable {
                reason: metal::TensorOpsUnavailableReason::GpuFamilyAbsent { .. }
            }
        ),
        "{:?}",
        ctx.tensor_ops_status()
    );
    fallback_witness(&ctx);
}

#[test]
#[ignore = "actual M5 in fresh process with QWEN_EXPERT_TENSOR_TILE=0, validation ON, production lease"]
fn m5_global_rollback_disables_candidate_despite_override() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().unwrap();
    assert!(ctx.device.name().to_string().contains("M5"));
    assert!(ctx.tensor_ops_status().is_available());
    assert_eq!(std::env::var("QWEN_EXPERT_TENSOR_TILE").as_deref(), Ok("0"));
    assert!(!metal::moe_grouped_tensor_selected(&ctx));
    fallback_witness(&ctx);
}

fn fallback_witness(ctx: &MetalContext) {
    for dtype in [GgmlType::Q6_K, GgmlType::Q8_0] {
        let f = Fixture::new(ctx, dtype, 256, 65, 1024);
        let _scope = Scope::new(true);
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        dispatch_census_begin();
        assert!(
            !try_encode(
                ctx,
                &enc,
                HALF,
                StageMode::Fast,
                &f.w,
                &f.x,
                &f.y,
                f.k,
                f.m,
                f.n
            )
            .unwrap()
        );
        assert!(dispatch_census_take().is_empty());
        assert!(enc.bound_pipeline_name.borrow().is_none());
        enc.end();
        assert!(read(&f.y).iter().all(|v| v.to_bits() == u32::MAX));
        let census = run(ctx, |enc| f.encode(ctx, enc, true));
        assert_eq!(census.len(), 1);
        assert!(!census[0].kernel.contains("tensor"));
        f.validate(&read(&f.y), &f.full, 1e-2);
    }
}
