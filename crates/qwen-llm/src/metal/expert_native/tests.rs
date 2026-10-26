use super::*;
use crate::metal::test_support::{
    assert_offset_guards, dequant_expert, metal_test_context, offset_tensor, tensor_backing_bytes,
    tensor_f32_at_offset,
};

const SOURCE: &str = include_str!("../../../../../kernels/expert_native.metal");
const EXPERTS: usize = 3;
const IDS: [i32; 7] = [2, 0, 2, -1, EXPERTS as i32, i32::MIN, i32::MAX];

mod decode_probe;
mod half_rows;
mod iq1;
mod rows;

fn production_lease() -> impl Sized {
    crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease")
}

#[test]
fn native_mxfp4_top6_large_banks_real_widths_and_dispatch() {
    large_banks_case(GgmlType::MXFP4, &[32, 2304, 5120]);
}

#[test]
fn native_iq2_xxs_high_ids_generic_geometry_and_dispatch() {
    large_banks_case(GgmlType::IQ2_XXS, &[256, 768, 1280]);
}

fn large_banks_case(dtype: GgmlType, widths: &[usize]) {
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    crate::metal::admit_iq1_expert_fixture(&ctx);
    for experts in [272usize, 320, 384] {
        for &k in widths {
            let m = 5;
            let bytes = bank_bytes_for_experts(dtype, k, m, 3, experts);
            let weights = offset_tensor(
                &ctx,
                32,
                &bytes,
                16,
                vec![k as u64, m as u64, experts as u64],
                dtype,
            );
            let routes = [
                experts as i32 - 1,
                0,
                experts as i32 - 1,
                271,
                -1,
                experts as i32,
            ];
            let ids = i32_view(&ctx, &routes);
            let mut x = values(k * 6, 7);
            x[k..2 * k].fill(0.0);
            let down_input = f32_view(&ctx, &x, &[k, 6], 16);
            let gate_input = f32_view(&ctx, &x[..k], &[k], 16);
            for ready in [ROUTE_STATUS_READY, ROUTE_STATUS_PENDING, -7, 2] {
                let status = i32_view(&ctx, &[ready]);
                for down in [false, true] {
                    let output = f32_view(&ctx, &[f32::NAN; 30], &[m, 6], 28);
                    let input = if down { &down_input } else { &gate_input };
                    let inputs = [&weights, input, &ids, &status];
                    let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
                    dispatch_census_begin();
                    one_shot(&ctx, |enc| {
                        if down {
                            encode_all_slots_down_native(
                                &ctx, enc, &weights, input, &ids, &status, &output, k, m, experts,
                                6,
                            )
                        } else {
                            encode_all_slots_gate_up_swiglu_native(
                                &ctx, enc, &weights, &weights, input, &ids, &status, &output, k, m,
                                experts, 6, 0.37,
                            )
                        }
                    })
                    .unwrap();
                    let census = dispatch_census_take();
                    assert_eq!(census.len(), if down { 1 } else { 2 });
                    assert!(
                        census
                            .iter()
                            .all(|r| r.kernel == native_pipeline_name(dtype).unwrap())
                    );
                    let mut expected = vec![0.0; 30];
                    for (slot, &id) in routes.iter().enumerate() {
                        if ready != ROUTE_STATUS_READY || id < 0 || id as usize >= experts {
                            continue;
                        }
                        let w = dequant_expert(&bytes, dtype, k, m, id as usize);
                        for row in 0..m {
                            let v = dot(
                                &w[row * k..][..k],
                                if down { &x[slot * k..][..k] } else { &x[..k] },
                            );
                            let g = v.min(0.37);
                            expected[slot * m + row] = if down {
                                v
                            } else {
                                g / (1.0 + (-g).exp()) * v.clamp(-0.37, 0.37)
                            };
                        }
                    }
                    close(
                        &format!("{dtype:?} top6"),
                        &tensor_f32_at_offset(&output),
                        &expected,
                    );
                    assert_offset_guards(&output, 20, 28);
                    for (t, b) in inputs.iter().zip(before) {
                        assert_eq!(tensor_backing_bytes(t), b);
                    }
                }
            }
        }
    }
}

#[test]
fn native_mxfp4_ggml_scale_edges_f32_precision_and_clamp_asymmetry() {
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    crate::metal::admit_iq1_expert_fixture(&ctx);
    let k = 32;
    // Basis probes cover both nibble planes and every code. High exponents
    // use only finite GGML codes; 255 is not the source converter's contract.
    for scale in [0u8, 1, 127, 145, 252, 254, 255] {
        let codes = if scale >= 254 { 2 } else { 16 };
        let m = codes;
        let mut bytes = vec![0u8; 17 * m * EXPERTS];
        for row in 0..m * EXPERTS {
            let b = &mut bytes[row * 17..][..17];
            b[0] = scale;
            b[1] = (row % codes) as u8;
            b[16] = ((row % codes) as u8) << 4;
        }
        let weights = bank(&ctx, &bytes, GgmlType::MXFP4, k, m);
        let ids = i32_view(&ctx, &[0, 0]);
        let status = i32_view(&ctx, &[ROUTE_STATUS_READY]);
        let mut x = vec![0.0; 64];
        let activation = if scale < 2 {
            2.0f32.powi(126)
        } else if scale >= 252 {
            0.125
        } else {
            70000.125
        };
        x[0] = activation;
        x[63] = -activation;
        let input = f32_view(&ctx, &x, &[k, 2], 0);
        let output = f32_view(&ctx, &vec![f32::NAN; m * 2], &[m, 2], 28);
        dispatch_census_begin();
        one_shot(&ctx, |enc| {
            encode_all_slots_down_native(
                &ctx, enc, &weights, &input, &ids, &status, &output, k, m, EXPERTS, 2,
            )
        })
        .unwrap();
        let census = dispatch_census_take();
        assert_eq!(census.len(), 1);
        assert_eq!(census[0].kernel, "kernel_expert_native_mxfp4_f32");
        let w = dequant_expert(&bytes, GgmlType::MXFP4, k, m, 0);
        let expected: Vec<_> = (0..2)
            .flat_map(|slot| {
                let w = &w;
                let x = &x;
                (0..m).map(move |row| dot(&w[row * k..][..k], &x[slot * k..][..k]))
            })
            .collect();
        close(
            "GGML scale edges",
            &tensor_f32_at_offset(&output),
            &expected,
        );
        assert_offset_guards(&output, 20, 28);
    }

    // g=-4 must remain -4 (not -1), while u=-4 must become -1.
    let mut bytes = vec![0u8; 17 * EXPERTS];
    for b in bytes.chunks_exact_mut(17) {
        b[0] = 127;
        b[1] = 14;
    }
    let weights = bank(&ctx, &bytes, GgmlType::MXFP4, 32, 1);
    let mut x = vec![0.0; 32];
    x[0] = 1.0;
    let input = f32_view(&ctx, &x, &[32], 0);
    let ids = i32_view(&ctx, &[0]);
    let status = i32_view(&ctx, &[ROUTE_STATUS_READY]);
    let output = f32_view(&ctx, &[f32::NAN], &[1, 1], 28);
    one_shot(&ctx, |enc| {
        encode_all_slots_gate_up_swiglu_native(
            &ctx, enc, &weights, &weights, &input, &ids, &status, &output, 32, 1, EXPERTS, 1, 1.0,
        )
    })
    .unwrap();
    let w = dequant_expert(&bytes, GgmlType::MXFP4, 32, 1, 0);
    let g = dot(&w, &x);
    assert_eq!(g, -4.0);
    close(
        "upper-only gate clamp",
        &tensor_f32_at_offset(&output),
        &[-g / (1.0 + (-g).exp())],
    );
    assert_offset_guards(&output, 20, 28);
}

fn formats() -> Vec<GgmlType> {
    GROUPED_PROJECTION_DTYPES
        .iter()
        .copied()
        .chain([
            GgmlType::IQ2_XS,
            GgmlType::IQ2_XXS,
            GgmlType::MXFP4,
            GgmlType::IQ1_S,
            GgmlType::IQ1_M,
        ])
        .collect()
}

#[test]
fn native_capabilities_match_actual_dispatch_and_abi() {
    assert_eq!(std::mem::size_of::<NativeArgs>(), 32);
    assert_eq!(std::mem::offset_of!(NativeArgs, row_bytes), 0);
    assert_eq!(std::mem::offset_of!(NativeArgs, mode), 24);
    assert_eq!(std::mem::offset_of!(NativeArgs, clamp), 28);
    assert_eq!(ROUTE_STATUS_READY, 1);
    // One-row instantiations plus the fused gate/up macro's one line.
    assert_eq!(
        SOURCE.matches("template [[host_name(").count(),
        formats().len() + 1
    );
    for dtype in formats() {
        if let Some(name) = fused_gate_up_name(dtype) {
            if name.ends_with("_split") {
                assert!(
                    SOURCE.contains(&format!("kernel void {name}(")),
                    "{dtype:?}"
                );
                continue;
            }
            let suffix = name
                .strip_prefix("kernel_expert_native_gate_up_")
                .and_then(|s| s.strip_suffix("_f32"))
                .unwrap();
            assert!(
                SOURCE.contains(&format!("EXPERT_NATIVE_GATE_UP(\"{suffix}\"")),
                "{dtype:?}"
            );
        }
    }
    for raw in 0..=255 {
        let dtype = GgmlType::from_raw(raw);
        assert_eq!(
            all_slots_down_native_supported(dtype),
            formats().contains(&dtype)
        );
        if let Some(name) = native_pipeline_name(dtype) {
            // Template instantiation, or a plain kernel (the Q5_K split).
            assert!(
                SOURCE.contains(&format!("host_name(\"{name}\")"))
                    || SOURCE.contains(&format!("kernel void {name}(")),
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
    let status = SOURCE.find("if (status[token] != 1)").unwrap();
    let ids = SOURCE.find("const int expert = ids[selected]").unwrap();
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
    bank_bytes_for_experts(dtype, k, m, seed, EXPERTS)
}

fn bank_bytes_for_experts(
    dtype: GgmlType,
    k: usize,
    m: usize,
    seed: usize,
    experts: usize,
) -> Vec<u8> {
    if matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        return (0..k * m * experts)
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
    for ordinal in 0..k * m * experts / block as usize {
        let mut b: Vec<u8> = (0..bytes as usize)
            .map(|i| (ordinal * 73 + i * 29 + seed * 11 + ordinal / 7) as u8)
            .collect();
        if dtype == GgmlType::MXFP4 {
            b[0] = 120 + ((ordinal + seed) % 5) as u8;
            result.extend(b);
            continue;
        }
        if matches!(dtype, GgmlType::IQ1_S | GgmlType::IQ1_M) {
            let scale = (1 + (ordinal + seed) % 7) as f32 * 0.0031;
            crate::metal::iq1_test_scale(&mut b, dtype, scale);
            result.extend(b);
            continue;
        }
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
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    crate::metal::admit_iq1_expert_fixture(&ctx);
    for (g, u) in [
        (GgmlType::Q2_K, GgmlType::Q3_K),
        (GgmlType::IQ2_S, GgmlType::IQ3_S),
        (GgmlType::F32, GgmlType::BF16),
        (GgmlType::IQ2_XS, GgmlType::Q4_K),
        (GgmlType::IQ2_XXS, GgmlType::Q4_K),
        (GgmlType::IQ2_XXS, GgmlType::F32),
        (GgmlType::MXFP4, GgmlType::F32),
        (GgmlType::MXFP4, GgmlType::Q2_K),
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
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    crate::metal::admit_iq1_expert_fixture(&ctx);
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
    native_validation_case(GgmlType::Q2_K, GgmlType::Q3_K);
}

fn native_validation_case(gate_type: GgmlType, up_type: GgmlType) {
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    crate::metal::admit_iq1_expert_fixture(&ctx);
    let (k, m, top_k) = (256, 7, IDS.len());
    let gate = bank(&ctx, &bank_bytes(gate_type, k, m, 3), gate_type, k, m);
    let up = bank(&ctx, &bank_bytes(up_type, k, m, 7), up_type, k, m);
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
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    crate::metal::admit_iq1_expert_fixture(&ctx);
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

/// The fused gate/up dispatch equals the two dispatches bitwise for every
/// fused storage type: batched and single-token, ready and rejected routes,
/// invalid IDs, and an odd channel count (half-filled last threadgroup).
#[test]
fn native_fused_gate_up_equals_two_passes_bitwise() {
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in formats() {
        let Some(fused_name) = fused_gate_up_name(dtype) else {
            continue;
        };
        let (block, _) = dtype.storage_layout().unwrap();
        let k = (block as usize).max(256) * 3;
        for m in [7usize, 16] {
            let gate = bank(&ctx, &bank_bytes(dtype, k, m, 3), dtype, k, m);
            let up = bank(&ctx, &bank_bytes(dtype, k, m, 4), dtype, k, m);
            for rows in [None, Some(3usize)] {
                let n = rows.unwrap_or(1);
                let input = if rows.is_some() {
                    f32_view(&ctx, &values(k * n, 7), &[k, n], 0)
                } else {
                    f32_view(&ctx, &values(k, 7), &[k], 0)
                };
                let id_values: Vec<i32> = (0..n).flat_map(|_| IDS).collect();
                let ids = if rows.is_some() {
                    offset_tensor(
                        &ctx,
                        12,
                        bytemuck::cast_slice(&id_values),
                        0,
                        vec![IDS.len() as u64, n as u64],
                        GgmlType::I32,
                    )
                } else {
                    i32_view(&ctx, &IDS)
                };
                let statuses: Vec<i32> =
                    [ROUTE_STATUS_READY, ROUTE_STATUS_PENDING, ROUTE_STATUS_READY][..n].to_vec();
                let status = i32_view(&ctx, &statuses);
                let slots = IDS.len() * n;
                let mut outputs = Vec::new();
                for two_pass in [true, false] {
                    let out = f32_view(&ctx, &vec![f32::NAN; m * slots], &[m, slots], 0);
                    dispatch_census_begin();
                    one_shot(&ctx, |enc| {
                        encode_gate_up(
                            &ctx,
                            enc,
                            &gate,
                            &up,
                            &input,
                            &ids,
                            &status,
                            &out,
                            k,
                            m,
                            EXPERTS,
                            IDS.len(),
                            0.37,
                            rows,
                            NativeOperands::F32,
                            !two_pass,
                        )
                    })
                    .unwrap();
                    let census = dispatch_census_take();
                    let kernels: Vec<_> = census.iter().map(|r| r.kernel.as_str()).collect();
                    if two_pass {
                        assert_eq!(kernels.len(), 2, "{dtype:?}");
                    } else {
                        assert_eq!(kernels, [fused_name], "{dtype:?}");
                    }
                    outputs.push(
                        tensor_f32_at_offset(&out)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                    );
                }
                assert!(
                    outputs[0].iter().any(|&b| f32::from_bits(b) != 0.0),
                    "{dtype:?}"
                );
                assert_eq!(outputs[0], outputs[1], "{dtype:?} m={m} rows={rows:?}");
            }
        }
    }
}
