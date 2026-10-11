use super::*;
use crate::metal::test_support::{
    assert_offset_guards, dequant_expert, metal_test_context, offset_tensor, tensor_backing_bytes,
    tensor_f32_at_offset,
};

const SOURCE: &str = include_str!("../../../../../kernels/expert_native.metal");
const EXPERTS: usize = 3;
const IDS: [i32; 7] = [2, 0, 2, -1, EXPERTS as i32, i32::MIN, i32::MAX];

fn formats() -> Vec<GgmlType> {
    GROUPED_PROJECTION_DTYPES
        .iter()
        .copied()
        .chain([GgmlType::IQ2_XS])
        .collect()
}

#[test]
fn native_capabilities_match_actual_dispatch_and_abi() {
    assert_eq!(std::mem::size_of::<NativeArgs>(), 32);
    assert_eq!(std::mem::offset_of!(NativeArgs, row_bytes), 0);
    assert_eq!(std::mem::offset_of!(NativeArgs, mode), 24);
    assert_eq!(std::mem::offset_of!(NativeArgs, clamp), 28);
    assert_eq!(ROUTE_STATUS_READY, 1);
    assert_eq!(
        SOURCE.matches("template [[host_name(").count(),
        formats().len()
    );
    for raw in 0..=255 {
        let dtype = GgmlType::from_raw(raw);
        assert_eq!(
            all_slots_down_native_supported(dtype),
            formats().contains(&dtype)
        );
        if let Some(name) = native_pipeline_name(dtype) {
            assert!(
                SOURCE.contains(&format!("host_name(\"{name}\")")),
                "{dtype:?}"
            );
        }
        for other in formats()
            .into_iter()
            .chain([GgmlType::Unknown, GgmlType::I32])
        {
            assert_eq!(
                all_slots_gate_up_native_supported(dtype, other),
                formats().contains(&dtype) && formats().contains(&other)
            );
        }
    }
    // Route rejection must precede all bank/activation/gate-result reads.
    let status = SOURCE.find("if (status[0] != 1)").unwrap();
    let ids = SOURCE.find("const int expert = ids[slot]").unwrap();
    let invalid = SOURCE.find("if (expert < 0").unwrap();
    let bank = SOURCE.find("device const uchar * row = bank").unwrap();
    assert!(status < ids && ids < invalid && invalid < bank);
    assert!(SOURCE.contains("(ulong(expert) * args.n_out + channel) * args.row_bytes"));
    // Rust argument construction must preserve a row stride beyond 4 GiB.
    let args = dimensions("test", u32::MAX as usize, 7, 3, 7).unwrap();
    assert_eq!(u64::from(args.n_in) * 4, 17_179_869_180);
}

// Independent linked GGML decoder supplies the oracle; no shader arithmetic
// or lookup tables are reimplemented in the CPU reference.
fn bank_bytes(dtype: GgmlType, k: usize, m: usize, seed: usize) -> Vec<u8> {
    if matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        return (0..k * m * EXPERTS)
            .flat_map(|i| {
                let v = (((i * 23 + seed * 11) % 97) as f32 - 48.0) * 0.013007;
                match dtype {
                    GgmlType::F32 => v.to_le_bytes().to_vec(),
                    GgmlType::F16 => half::f16::from_f32(v).to_le_bytes().to_vec(),
                    GgmlType::BF16 => half::bf16::from_f32(v).to_le_bytes().to_vec(),
                    _ => unreachable!(),
                }
            })
            .collect();
    }
    let (block, bytes) = dtype.storage_layout().unwrap();
    let mut result = Vec::new();
    for ordinal in 0..k * m * EXPERTS / block as usize {
        let mut b: Vec<u8> = (0..bytes as usize)
            .map(|i| (ordinal * 73 + i * 29 + seed * 11 + ordinal / 7) as u8)
            .collect();
        let scales: &[usize] = match dtype {
            GgmlType::Q2_K => &[80, 82],
            GgmlType::Q3_K => &[108],
            GgmlType::Q6_K => &[208],
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q4_1 => &[0, 2],
            _ => &[0],
        };
        for (j, &offset) in scales.iter().enumerate() {
            let d = (1 + (ordinal + seed + j) % 7) as f32 * 0.0031;
            b[offset..offset + 2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        }
        result.extend(b);
    }
    result
}

fn bank(ctx: &MetalContext, bytes: &[u8], dtype: GgmlType, k: usize, m: usize) -> MetalTensor {
    offset_tensor(
        ctx,
        32,
        bytes,
        0,
        vec![k as u64, m as u64, EXPERTS as u64],
        dtype,
    )
}

fn f32_view(ctx: &MetalContext, values: &[f32], shape: &[usize], suffix: usize) -> MetalTensor {
    offset_tensor(
        ctx,
        20,
        bytemuck::cast_slice(values),
        suffix,
        shape.iter().map(|&v| v as u64).collect(),
        GgmlType::F32,
    )
}

fn i32_view(ctx: &MetalContext, values: &[i32]) -> MetalTensor {
    offset_tensor(
        ctx,
        12,
        bytemuck::cast_slice(values),
        0,
        vec![values.len() as u64],
        GgmlType::I32,
    )
}

fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 17 + seed * 29) % 89) as f32 - 44.0) * 0.030013)
        .collect()
}

fn dot(w: &[f32], x: &[f32]) -> f32 {
    w.iter()
        .zip(x)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum::<f64>() as f32
}

fn close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= 8e-5 * e.abs().max(1.0),
            "{label} index {i}: {a} vs {e}"
        );
        if e == 0.0 {
            assert_eq!(a, 0.0, "{label} index {i}: invalid slot");
        }
    }
}

fn gate_up_case(
    ctx: &MetalContext,
    gate_type: GgmlType,
    up_type: GgmlType,
    k: usize,
    m: usize,
    route_status: i32,
) {
    let gb = bank_bytes(gate_type, k, m, 3);
    let ub = bank_bytes(up_type, k, m, 7);
    let gate = bank(ctx, &gb, gate_type, k, m);
    let up = bank(ctx, &ub, up_type, k, m);
    let x = values(k, 9);
    let input = f32_view(ctx, &x, &[k], 0);
    let ids = i32_view(ctx, &IDS);
    let status = i32_view(ctx, &[route_status]);
    let output = f32_view(ctx, &vec![f32::NAN; m * IDS.len()], &[m, IDS.len()], 28);
    let inputs = [&gate, &up, &input, &ids, &status];
    let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
    let c = 0.37f32; // Exercises upper-only gate clamp and both up clamp sides.
    dispatch_census_begin();
    one_shot(ctx, |enc| {
        encode_all_slots_gate_up_swiglu_native(
            ctx,
            enc,
            &gate,
            &up,
            &input,
            &ids,
            &status,
            &output,
            k,
            m,
            EXPERTS,
            IDS.len(),
            c,
        )
    })
    .unwrap();
    let census = dispatch_census_take();
    // Census preserves dispatch order, including mixed gate/up storage types.
    // Preflighting both PSOs must not change attribution of the gate binding.
    assert_eq!(
        census
            .iter()
            .map(|row| row.kernel.as_str())
            .collect::<Vec<_>>(),
        [
            native_pipeline_name(gate_type).unwrap(),
            native_pipeline_name(up_type).unwrap(),
        ]
    );
    let mut expected = vec![0.0; m * IDS.len()];
    for (slot, &id) in IDS.iter().enumerate() {
        if route_status != ROUTE_STATUS_READY || id < 0 || id as usize >= EXPERTS {
            continue;
        }
        let g = dequant_expert(&gb, gate_type, k, m, id as usize);
        let u = dequant_expert(&ub, up_type, k, m, id as usize);
        for channel in 0..m {
            let g = dot(&g[channel * k..][..k], &x).min(c);
            let u = dot(&u[channel * k..][..k], &x).clamp(-c, c);
            expected[slot * m + channel] = g / (1.0 + (-g).exp()) * u;
        }
    }
    close(
        &format!("{gate_type:?}/{up_type:?} K={k} M={m}"),
        &tensor_f32_at_offset(&output),
        &expected,
    );
    assert_offset_guards(&output, 20, 28);
    for (t, before) in inputs.iter().zip(before) {
        assert_eq!(tensor_backing_bytes(t), before);
    }
}

#[test]
fn native_mixed_gate_up_cpu_oracles_and_route_rejection() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for (g, u) in [
        (GgmlType::Q2_K, GgmlType::Q3_K),
        (GgmlType::IQ2_S, GgmlType::IQ3_S),
        (GgmlType::F32, GgmlType::BF16),
        (GgmlType::IQ2_XS, GgmlType::Q4_K),
    ] {
        for k in [256, 768, 1280] {
            for (g, u) in [(g, u), (u, g)] {
                gate_up_case(&ctx, g, u, k, 7, ROUTE_STATUS_READY);
            }
        }
        for status in [ROUTE_STATUS_PENDING, -7, 2] {
            gate_up_case(&ctx, g, u, 256, 5, status);
        }
    }
    // Every capability is used in both roles; tiny/tail dense K is also legal.
    for dtype in formats() {
        gate_up_case(&ctx, dtype, dtype, 256, 3, ROUTE_STATUS_READY);
    }
    for k in [1, 15, 17, 33, 513] {
        gate_up_case(
            &ctx,
            GgmlType::F32,
            GgmlType::BF16,
            k,
            5,
            ROUTE_STATUS_READY,
        );
    }
}

#[test]
fn native_all_down_formats_cpu_oracle_offsets_and_slot_isolation() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in formats() {
        let (block, _) = dtype.storage_layout().unwrap();
        let widths = if block == 1 {
            vec![1, 17, 513]
        } else {
            vec![block as usize, block as usize * 3, block as usize * 5]
        };
        for k in widths {
            let m = 7;
            let bytes = bank_bytes(dtype, k, m, 11);
            let bank = bank(&ctx, &bytes, dtype, k, m);
            let mut x = values(k * IDS.len(), 5);
            x[k..2 * k].fill(0.0); // valid zero slot must not leak other slot inputs
            let input = f32_view(&ctx, &x, &[k, IDS.len()], 0);
            let ids = i32_view(&ctx, &IDS);
            for route_status in [ROUTE_STATUS_READY, ROUTE_STATUS_PENDING, -1, 2] {
                let status = i32_view(&ctx, &[route_status]);
                let output = f32_view(&ctx, &vec![f32::NAN; m * IDS.len()], &[m, IDS.len()], 28);
                let before = [
                    tensor_backing_bytes(&bank),
                    tensor_backing_bytes(&input),
                    tensor_backing_bytes(&ids),
                    tensor_backing_bytes(&status),
                ];
                dispatch_census_begin();
                one_shot(&ctx, |enc| {
                    encode_all_slots_down_native(
                        &ctx,
                        enc,
                        &bank,
                        &input,
                        &ids,
                        &status,
                        &output,
                        k,
                        m,
                        EXPERTS,
                        IDS.len(),
                    )
                })
                .unwrap();
                let census = dispatch_census_take();
                assert_eq!(census.len(), 1);
                assert_eq!(census[0].kernel, native_pipeline_name(dtype).unwrap());
                let mut expected = vec![0.0; m * IDS.len()];
                for (slot, &id) in IDS.iter().enumerate() {
                    if route_status != ROUTE_STATUS_READY || id < 0 || id as usize >= EXPERTS {
                        continue;
                    }
                    let w = dequant_expert(&bytes, dtype, k, m, id as usize);
                    for channel in 0..m {
                        expected[slot * m + channel] =
                            dot(&w[channel * k..][..k], &x[slot * k..][..k]);
                    }
                }
                close(
                    &format!("{dtype:?} down K={k}"),
                    &tensor_f32_at_offset(&output),
                    &expected,
                );
                assert_offset_guards(&output, 20, 28);
                for (t, b) in [&bank, &input, &ids, &status].iter().zip(before) {
                    assert_eq!(tensor_backing_bytes(t), b);
                }
            }
        }
    }
}

#[test]
fn native_validation_rejects_before_any_dispatch() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (k, m, top_k) = (256, 7, IDS.len());
    let gate = bank(
        &ctx,
        &bank_bytes(GgmlType::Q2_K, k, m, 3),
        GgmlType::Q2_K,
        k,
        m,
    );
    let up = bank(
        &ctx,
        &bank_bytes(GgmlType::Q3_K, k, m, 7),
        GgmlType::Q3_K,
        k,
        m,
    );
    let input = f32_view(&ctx, &values(k, 9), &[k], 0);
    let ids = i32_view(&ctx, &IDS);
    let status = i32_view(&ctx, &[ROUTE_STATUS_READY]);
    let output = f32_view(&ctx, &vec![0.0; m * top_k], &[m, top_k], 28);
    let original = [gate, up, input, ids, status, output];
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let encode = |t: &[MetalTensor; 6], k, m, e, top_k, clamp| {
        encode_all_slots_gate_up_swiglu_native(
            &ctx, &enc, &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], k, m, e, top_k, clamp,
        )
    };
    dispatch_census_begin();
    for c in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, -1.0] {
        let err = encode(&original, k, m, EXPERTS, top_k, c).unwrap_err();
        assert!(
            err.to_string()
                .contains("clamp must be finite and positive")
        );
    }
    // Invalid up must not leave an already-encoded gate dispatch behind.
    for which in 0..6 {
        for fault in 0..6 {
            let mut t = original.clone();
            match fault {
                0 => t[which].offset += 1,
                1 => t[which].offset = t[which].buffer.length() as u64,
                2 => t[which].offset = u64::MAX - 3,
                3 => t[which].dtype = GgmlType::Unknown,
                4 => t[which].shape = vec![2],
                _ => t[which].shape = vec![u64::MAX, u64::MAX, 2],
            }
            assert!(
                encode(&t, k, m, EXPERTS, top_k, 1.0).is_err(),
                "binding {which}, fault {fault}"
            );
        }
    }
    let mut t = original.clone();
    t[5].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(encode(&t, k, m, EXPERTS, top_k, 1.0).is_err());
    // Enlarge short route bindings so alias rejection, rather than a buffer
    // extent error, proves that every read binding is protected.
    for which in 0..5 {
        let mut t = original.clone();
        let shared = ctx.buffer_from(&vec![0u8; 65536]).unwrap();
        t[which].buffer = shared.clone();
        t[which].offset = 32;
        t[5].buffer = shared;
        t[5].offset = 32;
        let err = encode(&t, k, m, EXPERTS, top_k, 1.0).unwrap_err();
        assert!(err.to_string().contains("aliases"), "{err}");
    }
    for (k, m, e, t) in [
        (0, m, EXPERTS, top_k),
        (k, 0, EXPERTS, top_k),
        (k, m, 0, top_k),
        (k, m, EXPERTS, 0),
        (k, m, EXPERTS, 17),
        (u32::MAX as usize + 1, m, EXPERTS, top_k),
        (k, usize::MAX, EXPERTS, top_k),
        (k, m, usize::MAX, top_k),
    ] {
        assert!(encode(&original, k, m, e, t, 1.0).is_err());
    }
    // Bank rows must contain full quant blocks even if total bank size is valid.
    let mut t = original.clone();
    t[0].shape = vec![128, m as u64, EXPERTS as u64];
    t[1].shape = t[0].shape.clone();
    t[2].shape = vec![128];
    assert!(encode(&t, 128, m, EXPERTS, top_k, 1.0).is_err());
    enc.end();
    assert!(dispatch_census_take().is_empty());

    let mut down = [
        original[0].clone(),
        f32_view(&ctx, &values(k * top_k, 2), &[k, top_k], 0),
        original[3].clone(),
        original[4].clone(),
        original[5].clone(),
    ];
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    dispatch_census_begin();
    for which in 0..down.len() {
        let saved = down[which].clone();
        down[which].offset += 1;
        assert!(
            encode_all_slots_down_native(
                &ctx, &enc, &down[0], &down[1], &down[2], &down[3], &down[4], k, m, EXPERTS, top_k
            )
            .is_err()
        );
        down[which] = saved;
    }
    let mut aliased = down[4].clone();
    aliased.buffer = down[1].buffer.clone();
    aliased.offset = down[1].offset;
    assert!(
        encode_all_slots_down_native(
            &ctx, &enc, &down[0], &down[1], &down[2], &down[3], &aliased, k, m, EXPERTS, top_k
        )
        .is_err()
    );
    enc.end();
    assert!(dispatch_census_take().is_empty());
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin_concurrent(&command);
    assert!(
        encode_all_slots_down_native(
            &ctx, &enc, &down[0], &down[1], &down[2], &down[3], &down[4], k, m, EXPERTS, top_k
        )
        .is_err()
    );
    assert!(
        encode_all_slots_gate_up_swiglu_native(
            &ctx,
            &enc,
            &original[0],
            &original[1],
            &original[2],
            &original[3],
            &original[4],
            &original[5],
            k,
            m,
            EXPERTS,
            top_k,
            1.0
        )
        .is_err()
    );
    enc.end();
}

#[test]
fn native_dense_preserves_f32_operands_and_range() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (k, m) = (17, 5);
    for dtype in [GgmlType::F32, GgmlType::BF16, GgmlType::F16] {
        let mut bytes = Vec::new();
        for row in 0..m * EXPERTS {
            for i in 0..k {
                let v = if i == 0 {
                    if dtype == GgmlType::F16 {
                        1.125
                    } else {
                        70000.25 + row as f32
                    }
                } else {
                    0.0
                };
                match dtype {
                    GgmlType::F32 => bytes.extend(v.to_le_bytes()),
                    GgmlType::BF16 => bytes.extend(half::bf16::from_f32(v).to_le_bytes()),
                    GgmlType::F16 => bytes.extend(half::f16::from_f32(v).to_le_bytes()),
                    _ => unreachable!(),
                }
            }
        }
        let bank = bank(&ctx, &bytes, dtype, k, m);
        let mut x = vec![0.0; k * IDS.len()];
        for slot in 0..IDS.len() {
            x[slot * k] = 70000.125 + slot as f32;
        }
        let input = f32_view(&ctx, &x, &[k, IDS.len()], 0);
        let output = f32_view(&ctx, &vec![f32::NAN; m * IDS.len()], &[m, IDS.len()], 0);
        let ids = i32_view(&ctx, &IDS);
        let status = i32_view(&ctx, &[ROUTE_STATUS_READY]);
        one_shot(&ctx, |enc| {
            encode_all_slots_down_native(
                &ctx,
                enc,
                &bank,
                &input,
                &ids,
                &status,
                &output,
                k,
                m,
                EXPERTS,
                IDS.len(),
            )
        })
        .unwrap();
        let actual = tensor_f32_at_offset(&output);
        for (slot, &id) in IDS.iter().enumerate() {
            if id < 0 || id as usize >= EXPERTS {
                assert!(actual[slot * m..][..m].iter().all(|&v| v == 0.0));
                continue;
            }
            let decoded = dequant_expert(&bytes, dtype, k, m, id as usize);
            for channel in 0..m {
                assert_eq!(
                    actual[slot * m + channel],
                    decoded[channel * k] * x[slot * k]
                );
            }
        }
        assert_offset_guards(&output, 20, 0);
    }
}
