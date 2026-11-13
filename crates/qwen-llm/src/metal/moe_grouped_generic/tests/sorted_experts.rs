use super::*;

const H: usize = 256;
const F: usize = 256;
const E: usize = 8;
const N: usize = 129;
const TOPK: usize = 2;
const S: usize = N * TOPK;
const GUARD: usize = 16;

fn guarded(ctx: &MetalContext, elements: usize, dtype: GgmlType) -> (MetalTensor, MetalTensor) {
    let unit = if dtype == GgmlType::F16 { 2 } else { 4 };
    let bytes = vec![0x5au8; (elements + 2 * GUARD) * unit];
    let backing =
        MetalTensor::from_bytes(ctx, &bytes, vec![(elements + 2 * GUARD) as u64], dtype).unwrap();
    let view = backing.view_subrange(GUARD as u64, vec![elements as u64]);
    (backing, view)
}

fn data(t: &MetalTensor) -> &[u8] {
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

fn guards(t: &MetalTensor) {
    let unit = if t.dtype == GgmlType::F16 { 2 } else { 4 };
    let bytes = data(t);
    assert!(
        bytes[..GUARD * unit]
            .iter()
            .chain(&bytes[bytes.len() - GUARD * unit..])
            .all(|&b| b == 0x5a)
    );
}

fn run(ctx: &MetalContext, encode: impl FnOnce(&KernelEncoder)) {
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    encode(&enc);
    enc.end();
    commit_and_wait(&command).unwrap();
}

// Both F32 storage boundaries are observable before the half conversion.
// Compare against the actual GPU weight/half chain, not a CPU approximation.
fn assert_bits(label: &str, actual: &[f32], reference: &[f32]) {
    let delta = crate::compare::assert_max_abs_diff_f32(label, actual.iter(), reference);
    eprintln!("{label}: max_abs={delta:e}, components={}", actual.len());
    for (i, (&a, &b)) in actual.iter().zip(reference).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label} component {i}: {a} != {b}"
        );
    }
}

#[test]
#[ignore = "self-leased M5 sorted IQ differential; MTL_DEBUG_LAYER=1; five formats, hot/tail/empty runs, aliased dead scratch, guards"]
fn sorted_iq_experts_match_rectangular_and_preserve_scratch_guards() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    assert!(moe_grouped_tensor_available(&ctx));
    let input: Vec<f32> = (0..H * N)
        .map(|i| ((i * 17 % 61) as f32 - 30.) * if i / H % 7 == 0 { 2. } else { 0.03125 })
        .collect();
    let x = f16_tensor(&ctx, &input);
    let weights: Vec<f32> = (0..S).map(|i| (i % 7) as f32 / 8.).collect();
    let weights_t = f32_tensor(&ctx, &weights);
    let raw_gu = MetalTensor::zeros_f32(&ctx, vec![(F * S) as u64]).unwrap();
    let reference = MetalTensor::zeros_f32(&ctx, vec![(H * S) as u64]).unwrap();
    let weighted = MetalTensor::zeros_f32(&ctx, vec![(F * S) as u64]).unwrap();
    let reference_inner = MetalTensor::zeros_f16(&ctx, vec![(F * S) as u64]).unwrap();
    let statuses = i32_typed_tensor(&ctx, &vec![1; N]);
    let types = [
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_XS,
        GgmlType::IQ2_S,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_S,
    ];
    for (format, dtype) in types.into_iter().enumerate() {
        let bank = |dt, seed| {
            MetalTensor::from_bytes(
                &ctx,
                &synthetic_bank(dt, H, F, E, seed),
                vec![H as u64, F as u64, E as u64],
                dt,
            )
            .unwrap()
        };
        let gate = bank(dtype, 11);
        let up = bank(dtype, 29);
        // Different down type as well as a second run with the same format.
        for down_type in [dtype, types[(format + 1) % types.len()]] {
            let down = bank(down_type, 47);
            for counts in [[0, 1, 15, 16, 31, 32, 64, 99], [129, 0, 0, 0, 129, 0, 0, 0]] {
                let mut ids = vec![-1; E * N];
                let mut at = 0;
                for (e, &count) in counts.iter().enumerate() {
                    for j in 0..count as usize {
                        ids[e * N + j] = ((at * 17) % S) as i32;
                        at += 1;
                    }
                }
                assert_eq!(at, S);
                let counts_t = i32_typed_tensor(&ctx, &counts);
                let ids_t = i32_typed_tensor(&ctx, &ids);
                run(&ctx, |enc| {
                    encode_moe_swiglu_f32_grouped_slots_tensor(
                        &ctx,
                        enc,
                        &gate,
                        &up,
                        &x,
                        &counts_t,
                        &ids_t,
                        &raw_gu,
                        H,
                        F,
                        E,
                        TOPK,
                        N,
                        Some(10.),
                    )
                    .unwrap()
                });
                // Use the real GPU weight/half chain, including its rounding,
                // rather than a CPU multiplication at a half rounding boundary.
                run(&ctx, |enc| {
                    enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_weight").unwrap());
                    enc.set_bytes(0, &((F * S) as u32));
                    enc.set_bytes(1, &(F as u32));
                    for (i, t) in [&raw_gu, &weights_t, &weighted, &statuses]
                        .iter()
                        .enumerate()
                    {
                        enc.set_tensor(i + 2, t);
                    }
                    enc.set_bytes(6, &(TOPK as u32));
                    enc.dispatch(
                        MTLSize {
                            width: (F * S).div_ceil(256),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                    crate::metal::encode_scatter_offset_f32_to_f16(
                        &ctx,
                        enc,
                        &weighted,
                        &reference_inner,
                        0,
                        F * S,
                    )
                    .unwrap();
                });
                run(&ctx, |enc| {
                    encode_moe_down_f32_grouped_slots_tensor(
                        &ctx,
                        enc,
                        &down,
                        &reference_inner,
                        &counts_t,
                        &ids_t,
                        &reference,
                        F,
                        H,
                        E,
                        N,
                    )
                    .unwrap()
                });
                let plan = guarded(&ctx, 2 * (E + 1), GgmlType::I32);
                let output = guarded(&ctx, H * S, GgmlType::F32);
                // The production lifetime aliases sorted input with down output.
                let mut packed = output.1.view_subrange(0, vec![(H * S / 2) as u64]);
                packed.dtype = GgmlType::F16;
                packed.shape = vec![(H * S) as u64];
                let inner = guarded(&ctx, F * S, GgmlType::F16);
                run(&ctx, |enc| {
                    let op = SortedExperts::prepare(
                        &ctx,
                        enc,
                        [&gate, &up, &down],
                        &x,
                        &counts_t,
                        &ids_t,
                        &weights_t,
                        &plan.1,
                        &packed,
                        &inner.1,
                        &output.1,
                        SortedExpertGeometry {
                            hidden: H,
                            inner: F,
                            experts: E,
                            topk: TOPK,
                            rows: N,
                            clamp: 10.,
                        },
                    )
                    .unwrap();
                    op.encode_gate_up(enc);
                });
                let plan_values: Vec<i32> = data(&plan.1)
                    .chunks_exact(4)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let mut off = 0;
                let mut tiles = 0;
                for e in 0..E {
                    assert_eq!(plan_values[e], off);
                    assert_eq!(plan_values[E + 1 + e], tiles);
                    for j in 0..counts[e] as usize {
                        let slot = ids[e * N + j] as usize;
                        let row = off as usize + j;
                        assert_eq!(
                            &data(&packed)[row * H * 2..(row + 1) * H * 2],
                            &data(&x)[slot / TOPK * H * 2..(slot / TOPK + 1) * H * 2]
                        );
                    }
                    off += counts[e];
                    tiles += (counts[e] + 63) / 64;
                }
                let mut original_order = vec![0f32; F * S];
                for e in 0..E {
                    for j in 0..counts[e] as usize {
                        let slot = ids[e * N + j] as usize;
                        let row = plan_values[e] as usize + j;
                        for k in 0..F {
                            let offset = (row * F + k) * 2;
                            original_order[slot * F + k] =
                                half::f16::from_bits(u16::from_le_bytes(
                                    data(&inner.1)[offset..offset + 2].try_into().unwrap(),
                                ))
                                .to_f32();
                        }
                    }
                }
                let reference_half: Vec<f32> = data(&reference_inner)
                    .chunks_exact(2)
                    .map(|b| {
                        half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()
                    })
                    .collect();
                assert_bits(
                    &format!("{dtype:?} inner"),
                    &original_order,
                    &reference_half,
                );
                assert_eq!(plan_values[E], S as i32);
                assert_eq!(plan_values[2 * E + 1], tiles);
                run(&ctx, |enc| {
                    let op = SortedExperts::prepare(
                        &ctx,
                        enc,
                        [&gate, &up, &down],
                        &x,
                        &counts_t,
                        &ids_t,
                        &weights_t,
                        &plan.1,
                        &packed,
                        &inner.1,
                        &output.1,
                        SortedExpertGeometry {
                            hidden: H,
                            inner: F,
                            experts: E,
                            topk: TOPK,
                            rows: N,
                            clamp: 10.,
                        },
                    )
                    .unwrap();
                    op.encode_down(enc);
                });
                assert_bits(
                    &format!("{dtype:?} / {down_type:?} down"),
                    &tensor_f32_at_offset(&output.1),
                    &tensor_f32_at_offset(&reference),
                );
                for t in [&plan.0, &inner.0, &output.0] {
                    guards(t);
                }
            }
        }
    }
}

#[test]
#[ignore = "self-leased M5 sorted IQ binding refusals; MTL_DEBUG_LAYER=1; refusal before any dispatch"]
fn sorted_iq_experts_refuse_bad_bindings_before_dispatch() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    assert!(moe_grouped_tensor_available(&ctx));
    let gate = MetalTensor::from_bytes(
        &ctx,
        &synthetic_bank(GgmlType::IQ2_XS, H, F, E, 11),
        vec![H as u64, F as u64, E as u64],
        GgmlType::IQ2_XS,
    )
    .unwrap();
    let x = MetalTensor::zeros_f16(&ctx, vec![(H * N) as u64]).unwrap();
    let counts = MetalTensor::zeros_i32(&ctx, vec![E as u64]).unwrap();
    let ids = MetalTensor::zeros_i32(&ctx, vec![(E * N) as u64]).unwrap();
    let weights = MetalTensor::zeros_f32(&ctx, vec![S as u64]).unwrap();
    let plan = guarded(&ctx, 2 * (E + 1), GgmlType::I32);
    let packed = guarded(&ctx, H * S, GgmlType::F16);
    let inner = guarded(&ctx, F * S, GgmlType::F16);
    let output = guarded(&ctx, H * S, GgmlType::F32);
    let g = SortedExpertGeometry {
        hidden: H,
        inner: F,
        experts: E,
        topk: TOPK,
        rows: N,
        clamp: 10.,
    };
    let original = [&plan.0, &packed.0, &inner.0, &output.0].map(|t| data(t).to_vec());
    for case in 0..10 {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let mut p = plan.1.clone();
        let mut i = inner.1.clone();
        let mut o = output.1.clone();
        let mut b = gate.clone();
        let mut geometry = g;
        match case {
            0 => p.dtype = GgmlType::F32,
            1 => p.shape = vec![1],
            2 => p.offset += 2,
            3 => i.dtype = GgmlType::F32,
            4 => i = packed.1.clone(),
            5 => o.offset = o.buffer.length() as u64,
            6 => b.dtype = GgmlType::F32,
            7 => geometry.rows = usize::MAX,
            8 => geometry.clamp = f32::NAN,
            9 => geometry.topk = 0,
            _ => unreachable!(),
        }
        crate::metal::dispatch_census_begin();
        assert!(
            SortedExperts::prepare(
                &ctx,
                &enc,
                [&b, &gate, &gate],
                &x,
                &counts,
                &ids,
                &weights,
                &p,
                &packed.1,
                &i,
                &o,
                geometry
            )
            .is_err(),
            "case {case}"
        );
        assert!(
            crate::metal::dispatch_census_take().is_empty(),
            "case {case} dispatched"
        );
        enc.end();
        commit_and_wait(&command).unwrap();
    }
    for (t, bytes) in [&plan.0, &packed.0, &inner.0, &output.0]
        .into_iter()
        .zip(original)
    {
        assert_eq!(data(t), bytes);
    }
}
