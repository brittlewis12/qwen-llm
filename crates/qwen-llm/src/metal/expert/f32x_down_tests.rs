//! Direct DOWN coverage, independent of the routed composition's gate/up policy.
use super::*;
use crate::metal::moe_grouped_generic::{
    MOE_GROUPED_GENERIC_LAYOUTS, check_moe_down_f32x_grouped_slots,
    moe_down_f32x_grouped_slots_pipeline_name, moe_down_f32x_grouped_slots_supported,
};

const SOURCE: &str = include_str!("../../../../../kernels/moe_grouped_f32.metal");
const DECODERS: &str = include_str!("../../../../../kernels/native_dequant_f32.h");

#[test]
fn down_f32x_capability_matches_native_instantiations() {
    assert_eq!(MOE_GROUPED_GENERIC_LAYOUTS.len(), 17);
    assert_eq!(
        SOURCE
            .matches("template [[host_name(\"kernel_moe_down_")
            .count(),
        17
    );
    assert!(SOURCE.contains("#include \"native_dequant_f32.h\""));
    for layout in MOE_GROUPED_GENERIC_LAYOUTS {
        let dtype = layout.dtype;
        assert!(moe_down_f32x_grouped_slots_supported(dtype));
        let name = moe_down_f32x_grouped_slots_pipeline_name(dtype).unwrap();
        let entry = SOURCE
            .lines()
            .find(|line| line.contains(&format!("host_name(\"{name}\")")))
            .unwrap();
        // Keep the established IQ3_S/IQ4_XS arithmetic; all new types use the
        // shared native decoder rather than another local decoder copy.
        let prefix = if matches!(dtype, GgmlType::IQ3_S | GgmlType::IQ4_XS) {
            "f32x"
        } else {
            "latent"
        };
        let constant = layout.suffix.to_uppercase();
        assert!(entry.contains(&format!(
            "<QT_{constant}_BYTES, QT_{constant}_NL, {prefix}_dequantize_{}>",
            layout.suffix
        )));
        assert!(DECODERS.contains(&format!("inline void latent_dequantize_{}(", layout.suffix)));
    }
    for dtype in [
        GgmlType::I32,
        GgmlType::Q5_0,
        GgmlType::IQ2_XXS,
        GgmlType::Unknown,
    ] {
        assert!(!moe_down_f32x_grouped_slots_supported(dtype));
        assert!(moe_down_f32x_grouped_slots_pipeline_name(dtype).is_none());
    }
}

/// Arbitrary native payloads with finite storage scales. Expected weights
/// come from the independent CPU codec, never from a GPU dequant/matvec.
fn native_bank(dtype: GgmlType, k: usize, m: usize, experts: usize) -> Vec<u8> {
    let mut state = 0x1234_5678_9abc_def1u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    if matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        return (0..k * m * experts)
            .flat_map(|_| {
                let v = ((next() % 1009) as f32 - 504.0) * 0.000317;
                match dtype {
                    GgmlType::F32 => v.to_le_bytes().to_vec(),
                    GgmlType::F16 => half::f16::from_f32(v).to_le_bytes().to_vec(),
                    GgmlType::BF16 => half::bf16::from_f32(v).to_le_bytes().to_vec(),
                    _ => unreachable!(),
                }
            })
            .collect();
    }
    let (block, block_bytes) = dtype.storage_layout().unwrap();
    let mut bytes = vec![0; k * m * experts / block as usize * block_bytes as usize];
    for byte in &mut bytes {
        *byte = next() as u8;
    }
    let offsets: &[usize] = match dtype {
        GgmlType::Q2_K => &[80, 82],
        GgmlType::Q3_K => &[108],
        GgmlType::Q6_K => &[208],
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q4_1 => &[0, 2],
        _ => &[0],
    };
    for (ordinal, block) in bytes.chunks_exact_mut(block_bytes as usize).enumerate() {
        for (j, &offset) in offsets.iter().enumerate() {
            let sign = if (ordinal + j) % 2 == 0 { 1.0 } else { -1.0 };
            let d = sign * (1 + (ordinal + j) % 7) as f32 * 0.00013;
            block[offset..offset + 2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        }
    }
    bytes
}

/// Bucket stride differs from output slot count. Expert 1 has `n` valid slots;
/// expert 4 has invalid ids around one valid slot. Experts 0/2/3 have zero,
/// negative and oversized counts, respectively. Nonzero offsets exercise
/// scalar-aligned weights and ids, vector-aligned activations, and output tails.
fn exercise_down(ctx: &MetalContext, dtype: GgmlType, k: usize, m: usize, n: usize) -> Vec<u32> {
    const E: usize = 5;
    const SENTINEL: f32 = -777.0;
    let stride = n.max(3);
    let slots = n + 3;
    let bytes = native_bank(dtype, k, m, E);
    let prefix = if dtype == GgmlType::F32 { 20 } else { 18 };
    let weight = offset_tensor(
        ctx,
        prefix,
        &bytes,
        22,
        vec![k as u64, m as u64, E as u64],
        dtype,
    );
    let x: Vec<f32> = (0..k * slots)
        .map(|i| {
            let scale = [131_073.0, 1e20, 1e-20][(i / k) % 3];
            (((i * 17 + (i / k) * 53) % 89) as f32 - 44.0) * 0.030013 * scale
        })
        .collect();
    let inner = f32_tensor(ctx, &x, vec![(k * slots) as u64]);
    let counts = offset_tensor(
        ctx,
        12,
        bytemuck::cast_slice(&[0i32, n as i32, -1, stride as i32 + 1, 3]),
        20,
        vec![E as u64],
        GgmlType::I32,
    );
    let mut ids = vec![i32::MAX; E * stride];
    for i in 0..n {
        ids[stride + i] = (n - 1 - i) as i32;
    }
    // These would overwrite the guards' sentinel rows if invalid counts ran.
    ids[2 * stride] = (n + 1) as i32;
    ids[3 * stride] = (n + 2) as i32;
    ids[4 * stride..4 * stride + 3].copy_from_slice(&[-1, n as i32, slots as i32]);
    let ids_t = offset_tensor(
        ctx,
        12,
        bytemuck::cast_slice(&ids),
        20,
        vec![(E * stride) as u64],
        GgmlType::I32,
    );
    let out = offset_tensor(
        ctx,
        20,
        bytemuck::cast_slice(&vec![SENTINEL; m * slots]),
        28,
        vec![(m * slots) as u64],
        GgmlType::F32,
    );
    let inputs = [&weight, &inner, &counts, &ids_t];
    let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
    run(ctx, |enc| {
        encode_moe_down_f32x_grouped_slots(
            ctx, enc, &weight, &inner, &counts, &ids_t, &out, k, m, E, stride,
        )
        .unwrap();
    });
    for (tensor, bytes) in inputs.iter().zip(before) {
        assert_eq!(
            tensor_backing_bytes(tensor),
            bytes,
            "{dtype:?}: input mutated"
        );
    }
    assert_offset_guards(&out, 20, 28);
    let got = read_f32(&out);
    for (expert, range) in [(1, 0..n), (4, n..n + 1)] {
        let decoded = dequant_expert(&bytes, dtype, k, m, expert);
        assert!(decoded.iter().all(|w| w.is_finite()));
        // Per-slot error prevents large activations from hiding mistakes in
        // the small-activation rows. Check both RMS and worst output error.
        for slot in range {
            let exact: Vec<f64> = decoded
                .chunks_exact(k)
                .map(|row| {
                    row.iter()
                        .zip(&x[slot * k..(slot + 1) * k])
                        .map(|(&w, &x)| f64::from(w) * f64::from(x))
                        .sum()
                })
                .collect();
            let actual = &got[slot * m..(slot + 1) * m];
            let rms = relative_rms(actual, &exact);
            let peak = exact.iter().map(|x| x.abs()).fold(0.0, f64::max);
            let max_error = crate::compare::assert_max_abs_diff_f64(
                concat!(file!(), ":", line!()),
                actual.iter().map(|&value| f64::from(value)),
                exact.iter(),
            );
            assert!(
                rms <= 1e-5 && max_error <= peak * 2e-5,
                "{dtype:?} K={k} M={m} n={n} expert={expert} slot={slot}: RMS {rms:e}, peak error {max_error:e}/{peak:e}"
            );
        }
    }
    assert!(
        got[(n + 1) * m..].iter().all(|&v| v == SENTINEL),
        "{dtype:?}: untouched slots"
    );

    // Move each valid slot within its bucket; every result stays bitwise equal.
    ids[stride..stride + n].reverse();
    let reordered = offset_tensor(
        ctx,
        12,
        bytemuck::cast_slice(&ids),
        20,
        vec![(E * stride) as u64],
        GgmlType::I32,
    );
    let again = f32_tensor(ctx, &vec![SENTINEL; m * slots], vec![(m * slots) as u64]);
    run(ctx, |enc| {
        encode_moe_down_f32x_grouped_slots(
            ctx, enc, &weight, &inner, &counts, &reordered, &again, k, m, E, stride,
        )
        .unwrap();
    });
    assert_offset_guards(&again, 16, 20);
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        read_f32(&again)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "{dtype:?}: bucket order"
    );
    got[..m].iter().map(|v| v.to_bits()).collect()
}

#[test]
fn down_f32x_all_native_decoders_match_cpu_and_preserve_slot_independence() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for layout in MOE_GROUPED_GENERIC_LAYOUTS {
        // Two storage blocks catch stepping errors; M=65 covers a row tail.
        let k = 2 * layout.block_elems.max(32);
        let mut first = None;
        for n in [1, 3, 17, 32, 33, 129] {
            let bits = exercise_down(&ctx, layout.dtype, k, 65, n);
            if let Some(ref first) = first {
                assert_eq!(&bits, first, "{:?}: bucket size {n}", layout.dtype);
            } else {
                first = Some(bits);
            }
        }
    }
}

#[test]
fn down_f32x_iq3_xxs_glm_dimensions_match_cpu() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    exercise_down(&ctx, GgmlType::IQ3_XXS, 2048, 4096, 3);
}

/// Dense F32/BF16 weights also retain values outside half's finite range.
#[test]
fn down_f32x_dense_weights_preserve_large_and_small_f32_range() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for dtype in [GgmlType::F32, GgmlType::BF16] {
        let source: Vec<f32> = (0..32)
            .map(|i| if i % 2 == 0 { 1e20 } else { -1e-20 })
            .collect();
        let bytes: Vec<u8> = source
            .iter()
            .flat_map(|&v| match dtype {
                GgmlType::F32 => v.to_le_bytes().to_vec(),
                GgmlType::BF16 => half::bf16::from_f32(v).to_le_bytes().to_vec(),
                _ => unreachable!(),
            })
            .collect();
        let weight = offset_tensor(&ctx, 20, &bytes, 20, vec![32, 1, 1], dtype);
        let decoded = dequant_expert(&bytes, dtype, 32, 1, 0);
        // Opposite scales yield finite products near one, including the tiny
        // weights that would disappear in half storage.
        let x: Vec<f32> = (0..32)
            .map(|i| if i % 2 == 0 { 1e-20 } else { -1e20 })
            .collect();
        let inner = f32_tensor(&ctx, &x, vec![32]);
        let counts = offset_tensor(
            &ctx,
            12,
            bytemuck::cast_slice(&[1i32]),
            20,
            vec![1],
            GgmlType::I32,
        );
        let ids = offset_tensor(
            &ctx,
            12,
            bytemuck::cast_slice(&[0i32]),
            20,
            vec![1],
            GgmlType::I32,
        );
        let out = f32_tensor(&ctx, &[0.0], vec![1]);
        run(&ctx, |enc| {
            encode_moe_down_f32x_grouped_slots(
                &ctx, enc, &weight, &inner, &counts, &ids, &out, 32, 1, 1, 1,
            )
            .unwrap();
        });
        let exact: f64 = decoded
            .iter()
            .zip(&x)
            .map(|(&w, &x)| f64::from(w) * f64::from(x))
            .sum();
        assert!(relative_rms(&read_f32(&out), &[exact]) < 1e-6, "{dtype:?}");
        assert_offset_guards(&out, 16, 20);
    }
}

#[test]
fn down_f32x_native_preflight_rejects_bad_shapes_types_and_alignment() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let counts = i32_zeros(&ctx, vec![1]);
    let ids = i32_zeros(&ctx, vec![1]);
    let out = f32_tensor(&ctx, &[0.0; 3], vec![3]);
    for layout in MOE_GROUPED_GENERIC_LAYOUTS {
        let k = layout.block_elems.max(32);
        let bytes = native_bank(layout.dtype, k, 3, 1);
        let prefix = if layout.dtype == GgmlType::F32 {
            20
        } else {
            18
        };
        let weight = offset_tensor(&ctx, prefix, &bytes, 20, vec![k as u64, 3, 1], layout.dtype);
        let inner = f32_tensor(&ctx, &vec![0.0; k], vec![k as u64]);
        let check = |w: &MetalTensor, x: &MetalTensor, width: usize| {
            check_moe_down_f32x_grouped_slots(&ctx, w, x, &counts, &ids, &out, width, 3, 1, 1)
        };
        assert!(
            check(&weight, &inner, k).is_ok(),
            "{:?}: minimum aligned K",
            layout.dtype
        );
        let unaligned = MetalTensor {
            offset: weight.offset + 1,
            ..weight.clone()
        };
        let truncated = MetalTensor {
            offset: weight.buffer.length() as u64,
            ..weight.clone()
        };
        let wrong_x = MetalTensor {
            dtype: GgmlType::I32,
            ..inner.clone()
        };
        for (label, result) in [
            ("odd bank", check(&unaligned, &inner, k)),
            ("bank past buffer", check(&truncated, &inner, k)),
            ("non-F32 input", check(&weight, &wrong_x, k)),
            ("zero K", check(&weight, &inner, 0)),
            ("partial block", check(&weight, &inner, k - 1)),
            (
                "i32 overflow",
                check(&weight, &inner, i32::MAX as usize + 1),
            ),
        ] {
            assert!(
                matches!(result, Err(MetalError::BadShape { .. })),
                "{:?} {label}: {result:?}",
                layout.dtype
            );
        }
        if layout.dtype == GgmlType::F32 {
            let half_aligned = MetalTensor {
                offset: weight.offset + 2,
                ..weight.clone()
            };
            assert!(matches!(
                check(&half_aligned, &inner, k),
                Err(MetalError::BadShape { .. })
            ));
        }
        if k > 32 {
            let short_inner = f32_tensor(&ctx, &[0.0; 32], vec![32]);
            assert!(matches!(
                check(&weight, &short_inner, 32),
                Err(MetalError::BadShape { .. })
            ));
        }
    }
    for dtype in [GgmlType::I32, GgmlType::Q5_0, GgmlType::Unknown] {
        let weight = MetalTensor {
            dtype,
            ..out.clone()
        };
        assert!(matches!(
            check_moe_down_f32x_grouped_slots(
                &ctx, &weight, &out, &counts, &ids, &out, 32, 3, 1, 1
            ),
            Err(MetalError::BadShape { .. })
        ));
    }
}
