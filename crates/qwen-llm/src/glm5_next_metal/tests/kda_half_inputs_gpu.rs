//! Small opt-in witnesses. Shared outputs deliberately bound fixture memory;
//! streaming checks verify each operation before complete-fanout timing.
use super::*;
use crate::glm5_next::memory::{DYNAMIC_RESERVE_BYTES, KdaHalfInputsScope};
use crate::glm5_next_metal::kda_half_inputs as stage;
use crate::glm5_next_metal::tensor_q5_down::Scope as Q5Scope;

#[test]
fn cpu_kda_half_pipeline_twins_preserve_q8_wide_and_q6_narrow() {
    let _wide = Q8WideScope::new(true);
    for (dtype, name) in [
        (GgmlType::Q8_0, "kernel_mat_mat_q8_0_f16act_tensor_wide"),
        (GgmlType::Q6_K, "kernel_mat_mat_q6_K_f16act_tensor"),
    ] {
        assert_eq!(
            Tile::selected(dtype)
                .pipeline_for(dtype, metal::TensorActivation::F16)
                .as_deref(),
            Some(name)
        );
    }
    let _narrow = Q8WideScope::new(false);
    assert_eq!(
        Tile::selected(GgmlType::Q8_0)
            .pipeline_for(GgmlType::Q8_0, metal::TensorActivation::F16)
            .as_deref(),
        Some("kernel_mat_mat_q8_0_f16act_tensor")
    );
    assert!(
        Tile::Q8Wide
            .pipeline_for(GgmlType::Q6_K, metal::TensorActivation::F16)
            .is_none()
    );
}

fn bytes(t: &MetalTensor) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize),
            t.n_bytes() as usize,
        )
    }
}

fn hash(t: &MetalTensor) -> blake3::Hash {
    blake3::hash(bytes(t))
}

fn assert_same(a: &MetalTensor, b: &MetalTensor) {
    assert_eq!(a.n_bytes(), b.n_bytes());
    if let Some(i) = bytes(a).iter().zip(bytes(b)).position(|(a, b)| a != b) {
        panic!(
            "bitwise mismatch at byte {i}: {} vs {}",
            bytes(a)[i],
            bytes(b)[i]
        );
    }
}

fn empty_guarded(ctx: &MetalContext, width: usize, rows: usize, dtype: GgmlType) -> MetalTensor {
    let size = width * rows * if dtype == GgmlType::F16 { 2 } else { 4 };
    let buffer = ctx.buffer_uninit(size + 2 * GUARD).unwrap();
    unsafe {
        let p = buffer.contents().as_ptr().cast::<u8>();
        p.write_bytes(0xa5, GUARD);
        p.add(GUARD).write_bytes(0xff, size);
        p.add(GUARD + size).write_bytes(0x5a, GUARD);
    }
    MetalTensor {
        buffer,
        offset: GUARD as u64,
        shape: vec![width as u64, rows as u64],
        dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn admit(ctx: &MetalContext, logical: &[usize], cpu: u64) {
    let physical: u64 = logical
        .iter()
        .map(|&n| {
            ctx.price_shared_buffer_upper((n + 2 * GUARD) as u64)
                .unwrap()
                .priced_upper_bytes
        })
        .sum();
    let peak = physical
        .checked_add(cpu)
        .unwrap()
        .checked_add(DYNAMIC_RESERVE_BYTES)
        .unwrap();
    assert!(peak <= 1 << 30, "bounded witness exceeds 1 GiB: {peak}");
    let admission = metal::evaluate_metal_memory_admission(
        physical + cpu,
        DYNAMIC_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    assert!(admission.is_admitted(), "{admission:?}");
}

#[test]
#[ignore = "actual M5, release, validation ON, explicit production lease"]
fn m5_kda_half_converter_offsets_guards_and_edge_rounding() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().unwrap();
    require_m5(&ctx);
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.00048828125,
        1.00146484375,
        65504.0,
        -65504.0,
        2f32.powi(-24),
        -2f32.powi(-24),
        2f32.powi(-25),
        3.0 * 2f32.powi(-25),
        f32::from_bits(0x38800000),
        f32::from_bits(0x387fffff),
    ];
    let _allocation = ctx.begin_allocation_transaction();
    admit(&ctx, &[values.len() * 4, (values.len() + 16) * 2], 1 << 20);
    let src = guarded(
        &ctx,
        bytemuck::cast_slice(&values),
        vec![values.len() as u64],
        GgmlType::F32,
    );
    let dst = empty_guarded(&ctx, values.len() + 16, 1, GgmlType::F16);
    let before = hash(&src);
    let census = run(&ctx, |enc| {
        metal::encode_scatter_offset_f32_to_f16(&ctx, enc, &src, &dst, 8, values.len()).unwrap()
    });
    assert_eq!(census.len(), 1);
    assert_eq!(census[0].kernel, "kernel_scatter_offset_f32_to_f16");
    let actual: &[u16] = bytemuck::cast_slice(bytes(&dst));
    assert!(actual[..8].iter().all(|&v| v == u16::MAX));
    assert!(actual[8 + values.len()..].iter().all(|&v| v == u16::MAX));
    for (&value, &bits) in values.iter().zip(&actual[8..]) {
        assert_eq!(bits, half::f16::from_f32(value).to_bits(), "{value:?}");
    }
    assert_eq!(before, hash(&src));
    assert_guards(&src);
    assert_guards(&dst);
    small_preflight_witness(&ctx);
}

fn small_preflight_witness(ctx: &MetalContext) {
    let _on = KdaHalfInputsScope::new(true);
    let _wide = Q8WideScope::new(true);
    let _allocation = ctx.begin_allocation_transaction();
    admit(
        ctx,
        &[
            256 * 1024 * 4,
            256 * 1024 * 2,
            65 * 1024 * 4,
            256 * 65,
            65 * 1024 * 4,
        ],
        16 << 20,
    );
    let f = Fixture::new(ctx, GgmlType::Q6_K, 256, 65, 1024);
    let half = empty_guarded(ctx, 256, 1024, GgmlType::F16);
    let y = empty_guarded(ctx, 65, 1024, GgmlType::F32);
    let inputs = [(&f.w, &y, 65); 6];
    let initial = hash(&half);
    let attempt = |arith, rows, backing: &MetalTensor, inputs: &stage::Consumers<'_>, error| {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        dispatch_census_begin();
        let result = stage::prepare(ctx, &enc, arith, &f.x, Some(backing), inputs, 256, rows);
        if error {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_none());
        }
        assert!(dispatch_census_take().is_empty());
        assert!(enc.bound_pipeline_name.borrow().is_none());
        enc.end();
        assert_eq!(initial, hash(&half));
    };
    for arith in [
        Arith {
            lineage: PackedLineage::Exact,
            ..HALF
        },
        Arith {
            precision: FastPrecision::F32,
            ..HALF
        },
    ] {
        attempt(arith, 1024, &half, &inputs, false);
    }
    attempt(HALF, 1023, &half, &inputs, false);
    {
        let _off = KdaHalfInputsScope::new(false);
        attempt(HALF, 1024, &half, &inputs, false);
    }
    let mut readonly = half.clone();
    readonly.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    attempt(HALF, 1024, &readonly, &inputs, true);
    let mut outside = half.clone();
    outside.offset = 16;
    outside.shape[1] += 1;
    attempt(HALF, 1024, &outside, &inputs, true);
    let mut alias = y.clone();
    alias.buffer = half.buffer.clone();
    let mut bad = inputs;
    bad[5].1 = &alias;
    attempt(HALF, 1024, &half, &bad, true);
    let x_hash = hash(&f.x);
    run(ctx, |enc| f.encode(ctx, enc, true));
    let census = run(ctx, |enc| {
        let plan = stage::prepare(ctx, enc, HALF, &f.x, Some(&half), &inputs, 256, 1024)
            .unwrap()
            .unwrap();
        plan.copy(ctx, enc).unwrap();
        assert!(plan.encode(0, ctx, enc).unwrap());
    });
    assert_eq!(census.len(), 2);
    assert_eq!(census[1].kernel, "kernel_mat_mat_q6_K_f16act_tensor");
    assert_same(&f.y, &y);
    f.validate(&read(&y), &f.half, 2e-3);
    assert_eq!(x_hash, hash(&f.x));
    for t in [&f.w, &f.x, &f.y, &half, &y] {
        assert_guards(t);
    }
}

fn consumers<'a>(
    weight: &'a MetalTensor,
    small: &'a MetalTensor,
    beta: &'a MetalTensor,
    wide: &'a MetalTensor,
    rank: &'a MetalTensor,
    beta_out: &'a MetalTensor,
) -> stage::Consumers<'a> {
    [
        (weight, wide, 8192),
        (weight, wide, 8192),
        (weight, wide, 8192),
        (small, rank, 128),
        (beta, beta_out, 64),
        (small, rank, 128),
    ]
}

fn fanout(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    enabled: bool,
    x: &MetalTensor,
    half: &MetalTensor,
    inputs: &stage::Consumers<'_>,
    expansion: &MetalTensor,
    wide: &MetalTensor,
    rows: usize,
) {
    let _stage = KdaHalfInputsScope::new(enabled);
    packed::encode_kda_fanout(
        ctx,
        enc,
        HALF,
        x,
        Some(half),
        inputs,
        [(expansion, wide), (expansion, wide)],
        4096,
        8192,
        128,
        rows,
    )
    .unwrap();
}

fn timed(ctx: &MetalContext, body: impl Fn(&KernelEncoder)) -> (f64, f64) {
    let start = std::time::Instant::now();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    for _ in 0..4 {
        body(&enc);
    }
    enc.end();
    command.commit();
    wait_completed(&command).unwrap();
    let gpu = (command.GPUEndTime() - command.GPUStartTime()) * 250.0;
    assert!(gpu.is_finite() && gpu > 0.0);
    (gpu, start.elapsed().as_secs_f64() * 250.0)
}

#[test]
#[ignore = "actual M5, release, explicit production lease; validation ON or diagnostic timing OFF"]
fn m5_kda_half_complete_eight_op_fanout_bitwise_and_full_copy_cost() {
    let debug = std::env::var("MTL_DEBUG_LAYER").unwrap();
    assert!(matches!(debug.as_str(), "0" | "1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    require_m5(&ctx);
    let _tensor = Scope::new(true);
    let _wide = Q8WideScope::new(true);
    let _q5 = Q5Scope::new(true);
    assert!(
        crate::env_flag::read_default_off("QWEN_GLM53_KDA_HALF_INPUTS"),
        "launch with feature storage reserved in both arms"
    );
    let dtype = GgmlType::Q8_0;
    for rows in [1024, 1025, 2048] {
        let _allocation = ctx.begin_allocation_transaction();
        let weight_bytes = geometry(dtype, 4096, 8192, 1).unwrap().weight_bytes;
        let expansion_bytes = geometry(GgmlType::Q8_0, 128, 8192, 1).unwrap().weight_bytes;
        // Two wide outputs, one rank/beta set per arm, a full 2048-row
        // half backing even for a ragged packet. CPU includes construction
        // uploads and one streamed F64 comparison readback, never F32 W.
        admit(
            &ctx,
            &[
                weight_bytes,
                expansion_bytes,
                // Constructor scratch, including overlap during replacement.
                4096 * 4,
                8192 * 4,
                128 * 4,
                8192 * 4,
                4096 * rows * 4,
                4096 * 2048 * 2,
                8192 * rows * 4,
                8192 * rows * 4,
                128 * rows * 4,
                128 * rows * 4,
                64 * rows * 4,
                64 * rows * 4,
            ],
            160 << 20,
        );
        // Reuse the existing compact native recipes and F64 sample oracles
        // with a one-row constructor; large activations are filled in place.
        let mut f = Fixture::new(&ctx, dtype, 4096, 8192, 1);
        let expand = Fixture::new(&ctx, GgmlType::Q8_0, 128, 8192, 1);
        f.n = rows;
        f.x = empty_guarded(&ctx, 4096, rows, GgmlType::F32);
        f.y = empty_guarded(&ctx, 8192, rows, GgmlType::F32);
        unsafe {
            let data: &mut [f32] = std::slice::from_raw_parts_mut(
                f.x.buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(f.x.offset as usize)
                    .cast::<f32>(),
                4096 * rows,
            );
            for (i, value) in data.iter_mut().enumerate() {
                *value = (((i % 4096 * 37 + (i / 4096 % INPUT_ROWS) * 17) % 211) as f32 - 105.0)
                    * 0.003713;
            }
        }
        let baseline_y = empty_guarded(&ctx, 8192, rows, GgmlType::F32);
        let half = empty_guarded(&ctx, 4096, 2048, GgmlType::F16);
        let rank_a = empty_guarded(&ctx, 128, rows, GgmlType::F32);
        let rank_b = empty_guarded(&ctx, 128, rows, GgmlType::F32);
        let beta_a = empty_guarded(&ctx, 64, rows, GgmlType::F32);
        let beta_b = empty_guarded(&ctx, 64, rows, GgmlType::F32);
        let small = f.w.view_subrange(0, vec![4096, 128]);
        let beta = f.w.view_subrange(0, vec![4096, 64]);
        let a = consumers(&f.w, &small, &beta, &baseline_y, &rank_a, &beta_a);
        let b = consumers(&f.w, &small, &beta, &f.y, &rank_b, &beta_b);
        let input_hash = hash(&f.x);
        let weight_hash = hash(&f.w);
        // Failures in even the last eligible consumer must precede copy.
        let mut bad = rank_b.clone();
        bad.offset = u64::MAX - 3;
        let mut invalid = b;
        invalid[5].1 = &bad;
        let half_hash = hash(&half);
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let _on = KdaHalfInputsScope::new(true);
        dispatch_census_begin();
        assert!(stage::prepare(&ctx, &enc, HALF, &f.x, Some(&half), &invalid, 4096, rows).is_err());
        assert!(dispatch_census_take().is_empty());
        assert!(enc.bound_pipeline_name.borrow().is_none());
        enc.end();
        assert_eq!(half_hash, hash(&half));
        // Compare all six outputs directly, one operation at a time with
        // reused GPU storage; no full-size CPU baseline is retained.
        for index in 0..6 {
            let (w, y, m) = a[index];
            run(&ctx, |enc| {
                packed::matmat(
                    &ctx,
                    enc,
                    HALF,
                    packed::Stage::KdaProjection,
                    w,
                    &f.x,
                    y,
                    4096,
                    m,
                    rows,
                )
                .unwrap()
            });
            let census = run(&ctx, |enc| {
                let plan = stage::prepare(&ctx, enc, HALF, &f.x, Some(&half), &b, 4096, rows)
                    .unwrap()
                    .unwrap();
                // Selection is frozen: changing wide policy after prepare
                // cannot silently select another dispatch pipeline.
                let _changed_wide = Q8WideScope::new(false);
                plan.copy(&ctx, enc).unwrap();
                assert!(plan.encode(index, &ctx, enc).unwrap());
            });
            assert_eq!(census.len(), 2);
            assert_eq!(
                census[1].kernel,
                Tile::selected(dtype)
                    .pipeline_for(dtype, metal::TensorActivation::F16)
                    .unwrap()
            );
            assert_same(y, b[index].1);
            if index == 0 {
                f.validate(&read(&f.y), &f.half, 2e-3);
            }
        }
        for enabled in [false, true] {
            let inputs = if enabled { &b } else { &a };
            let out = if enabled { &f.y } else { &baseline_y };
            let census = run(&ctx, |enc| {
                fanout(
                    &ctx, enc, enabled, &f.x, &half, inputs, &expand.w, out, rows,
                )
            });
            assert_eq!(census.len(), 8 + usize::from(enabled));
            assert_eq!(
                census
                    .iter()
                    .filter(|r| r.kernel == "kernel_scatter_offset_f32_to_f16")
                    .count(),
                usize::from(enabled)
            );
            let half_consumers = census
                .iter()
                .filter(|r| r.kernel.contains("_f16act_tensor"))
                .count();
            assert_eq!(half_consumers, if enabled { 6 } else { 0 });
            // Both expansions still use the F32-input Q8 wide twin.
            let expansion_calls: Vec<_> = census
                .iter()
                .filter(|r| r.kernel == "kernel_mat_mat_q8_0_f32_tensor_wide")
                .collect();
            assert_eq!(expansion_calls.len(), if enabled { 2 } else { 8 });
        }
        assert_same(&baseline_y, &f.y);
        assert_same(&rank_a, &rank_b);
        assert_same(&beta_a, &beta_b);
        // The overwritten decay expansion is separately compared using the
        // already bit-exact rank outputs, still through the unchanged path.
        for (rank, out) in [(&rank_a, &baseline_y), (&rank_b, &f.y)] {
            run(&ctx, |enc| {
                packed::matmat(
                    &ctx,
                    enc,
                    HALF,
                    packed::Stage::KdaExpand,
                    &expand.w,
                    rank,
                    out,
                    128,
                    8192,
                    rows,
                )
                .unwrap()
            });
        }
        assert_same(&baseline_y, &f.y);
        assert_eq!(input_hash, hash(&f.x));
        assert_eq!(weight_hash, hash(&f.w));
        for t in [
            &f.w,
            &f.x,
            &f.y,
            &baseline_y,
            &half,
            &rank_a,
            &rank_b,
            &beta_a,
            &beta_b,
            &expand.w,
        ] {
            assert_guards(t);
        }
        assert!(bytes(&half)[4096 * rows * 2..].iter().all(|&b| b == 0xff));
        drop(_allocation);
        // No census, readback or logging in conditioning/measured ABBA.
        let measure = |enabled| {
            timed(&ctx, |enc| {
                fanout(&ctx, enc, enabled, &f.x, &half, &b, &expand.w, &f.y, rows)
            })
        };
        for enabled in [false, true, true, false] {
            measure(enabled);
        }
        let samples: Vec<_> = (0..3)
            .flat_map(|round| {
                [false, true, true, false].map(|enabled| (round, enabled, measure(enabled)))
            })
            .collect();
        eprintln!(
            "GLM KDA half {dtype:?} R{rows} debug={debug} full_copy_and_eight_ops=true both_arms_half_backing_mib=16 qualified_wide_q8_and_q5_down=ON repeats=4 (round,stage,(gpu_ms,wall_ms))={samples:?}; diagnostic only, no GLM gain assumed"
        );
    }
}
