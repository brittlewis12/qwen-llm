//! Independent CPU codec/F64 projections keep fused clamp saturation from
//! hiding decoder errors; the candidate's bucket arithmetic stays bitwise fixed.
use super::*;
use crate::metal::moe_grouped_generic::{
    check_moe_swiglu_clamped_f32x_grouped_slots, moe_swiglu_clamped_f32x_grouped_slots_supported,
};

fn assert_f64(label: &str, actual: &[f32], exact: &[f64]) {
    let rms = relative_rms(actual, exact);
    let peak = exact.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let error = crate::compare::assert_max_abs_diff_f64(
        label,
        actual.iter().map(|&v| f64::from(v)),
        exact.iter(),
    );
    assert!(
        rms <= 1e-5 && error <= 2e-5 * peak,
        "{label}: RMS={rms:e}, error={error:e}/{peak:e}"
    );
}

fn ints(ctx: &MetalContext, values: &[i32]) -> MetalTensor {
    offset_tensor(
        ctx,
        12,
        bytemuck::cast_slice(values),
        20,
        vec![values.len() as u64],
        GgmlType::I32,
    )
}

#[test]
fn same_type_native_f32_gate_up_capability_matches_instantiations() {
    let source = include_str!("../../../../../kernels/moe_grouped_f32.metal");
    let supported = [
        GgmlType::MXFP4,
        GgmlType::IQ2_S,
        GgmlType::IQ3_S,
        GgmlType::IQ3_XXS,
        GgmlType::Q3_K,
    ];
    assert_eq!(
        source
            .matches("template [[host_name(\"kernel_moe_swiglu_clamped_")
            .count(),
        supported.len()
    );
    for raw in 0..=255 {
        let dtype = GgmlType::from_raw(raw);
        assert_eq!(
            moe_swiglu_clamped_f32x_grouped_slots_supported(dtype),
            supported.contains(&dtype)
        );
    }
    for suffix in ["iq3_xxs", "q3_K"] {
        let line = source
            .lines()
            .find(|l| {
                l.contains(&format!(
                    "host_name(\"kernel_moe_swiglu_clamped_{suffix}_f32x_grouped_slots\")"
                ))
            })
            .unwrap();
        assert!(line.contains(&format!("latent_dequantize_{suffix}>")));
    }
    // Shared primitive expansion is not a GLM composition precision policy change.
    for dtype in [
        GgmlType::IQ3_XXS,
        GgmlType::Q3_K,
        GgmlType::IQ2_XS,
        GgmlType::IQ2_XXS,
    ] {
        assert!(!grouped_routed_experts_f32x_supported(
            dtype,
            dtype,
            GgmlType::IQ3_S
        ));
    }
}

fn exercise(ctx: &MetalContext, dtype: GgmlType, k: usize, m: usize, n: usize) {
    const E: usize = 5;
    const TOP: usize = 2;
    const SENTINEL: f32 = -777.0;
    let gate_bytes = down::native_bank(dtype, k, m, E);
    let mut up_bytes = gate_bytes.clone();
    let block_bytes = dtype.storage_layout().unwrap().1 as usize;
    up_bytes.rotate_left(3 * block_bytes);
    assert_ne!(gate_bytes, up_bytes);
    let make_bank = |bytes: &[u8]| {
        offset_tensor(
            ctx,
            18,
            bytes,
            22,
            vec![k as u64, m as u64, E as u64],
            dtype,
        )
    };
    let gate = make_bank(&gate_bytes);
    let up = make_bank(&up_bytes);
    let x: Vec<f32> = (0..k * n)
        .map(|i| {
            let scale = if (i / k) % 3 == 0 { 70000.125 } else { 0.00317 };
            ((i * 17 + i / k * 53) % 89) as f32 * scale - 44.0 * scale
        })
        .collect();
    let input = f32_tensor(ctx, &x, vec![k as u64, n as u64]);
    let counts = ints(ctx, &[0, n as i32, -1, n as i32 + 1, 3]);
    let mut ids = vec![i32::MAX; E * n];
    for token in 0..n {
        ids[n + token] = (token * TOP + token % TOP) as i32;
    }
    ids[2 * n] = 2;
    ids[3 * n] = 5;
    ids[4 * n..4 * n + 3].copy_from_slice(&[-1, 1, (n * TOP) as i32]);
    let slots = ints(ctx, &ids);
    let output = f32_tensor(
        ctx,
        &vec![SENTINEL; m * n * TOP],
        vec![(m * n * TOP) as u64],
    );
    let inputs = [&gate, &up, &input, &counts, &slots];
    let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
    run(ctx, |enc| {
        encode_moe_swiglu_clamped_f32x_grouped_slots(
            ctx, enc, &gate, &up, &input, &counts, &slots, &output, k, m, E, TOP, n, CLAMP,
        )
        .unwrap();
    });
    let actual = read_f32(&output);
    let expanded: Vec<_> = x
        .chunks_exact(k)
        .flat_map(|x| x.iter().copied().chain(x.iter().copied()))
        .collect();
    let expanded = f32_tensor(ctx, &expanded, vec![(k * n * TOP) as u64]);
    let projections: Vec<_> = [&gate, &up]
        .iter()
        .map(|bank| {
            let out = f32_tensor(
                ctx,
                &vec![SENTINEL; m * n * TOP],
                vec![(m * n * TOP) as u64],
            );
            run(ctx, |enc| {
                encode_moe_down_f32x_grouped_slots(
                    ctx, enc, bank, &expanded, &counts, &slots, &out, k, m, E, n,
                )
                .unwrap()
            });
            assert_offset_guards(&out, 16, 20);
            read_f32(&out)
        })
        .collect();
    let mut written = vec![false; n * TOP];
    let mut unsaturated = 0;
    for (expert, selected) in [(1, ids[n..2 * n].to_vec()), (4, vec![1])] {
        let g = dequant_expert(&gate_bytes, dtype, k, m, expert);
        let u = dequant_expert(&up_bytes, dtype, k, m, expert);
        for slot in selected {
            let slot = slot as usize;
            written[slot] = true;
            let x = &x[(slot / TOP) * k..(slot / TOP + 1) * k];
            let dot = |w: &[f32]| -> Vec<f64> {
                w.chunks_exact(k)
                    .map(|w| {
                        w.iter()
                            .zip(x)
                            .map(|(&w, &x)| f64::from(w) * f64::from(x))
                            .sum()
                    })
                    .collect()
            };
            let (g, u) = (dot(&g), dot(&u));
            for (projection, exact) in projections.iter().zip([&g, &u]) {
                assert_f64(
                    "independent projection",
                    &projection[slot * m..(slot + 1) * m],
                    exact,
                );
            }
            let exact: Vec<_> = g
                .iter()
                .zip(&u)
                .map(|(&g, &u)| {
                    unsaturated +=
                        usize::from(g.abs() < f64::from(CLAMP) && u.abs() < f64::from(CLAMP));
                    let g = g.min(f64::from(CLAMP));
                    g / (1.0 + (-g).exp()) * u.clamp(-f64::from(CLAMP), f64::from(CLAMP))
                })
                .collect();
            assert_f64("fused SwiGLU", &actual[slot * m..(slot + 1) * m], &exact);
        }
    }
    assert!(unsaturated >= m);
    for (slot, written) in written.iter().enumerate() {
        if !written {
            assert!(
                actual[slot * m..(slot + 1) * m]
                    .iter()
                    .all(|&v| v == SENTINEL)
            );
        }
    }
    for (tensor, before) in inputs.iter().zip(before) {
        assert_eq!(tensor_backing_bytes(tensor), before);
    }
    assert_offset_guards(&output, 16, 20);
    // Reorder and partition buckets without changing candidate arithmetic or slot addresses.
    ids[n..2 * n].reverse();
    let again = f32_tensor(
        ctx,
        &vec![SENTINEL; m * n * TOP],
        vec![(m * n * TOP) as u64],
    );
    for range in [0..1, 1..n] {
        let mut part = vec![-1; E * n];
        let count = range.len();
        part[n..n + count].copy_from_slice(&ids[n + range.start..n + range.end]);
        part[4 * n] = 1;
        let counts = ints(ctx, &[0, count as i32, 0, 0, i32::from(range.start == 0)]);
        let part = ints(ctx, &part);
        run(ctx, |enc| {
            encode_moe_swiglu_clamped_f32x_grouped_slots(
                ctx, enc, &gate, &up, &input, &counts, &part, &again, k, m, E, TOP, n, CLAMP,
            )
            .unwrap()
        });
    }
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        read_f32(&again)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_offset_guards(&again, 16, 20);
}

#[test]
fn native_f32_gate_up_cpu_f64_slots_buckets_and_partition() {
    let _lease = production_lease();
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for dtype in [GgmlType::IQ3_XXS, GgmlType::Q3_K] {
        for n in [3, 15, 16, 17, 31, 32, 33] {
            exercise(&ctx, dtype, 512, 65, n);
        }
    }
}

#[test]
fn native_f32_gate_up_v41_widths_reduced_banks() {
    let _lease = production_lease();
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for dtype in [GgmlType::IQ3_XXS, GgmlType::Q3_K] {
        exercise(&ctx, dtype, 5120, 2304, 3);
    }
}

#[test]
fn native_f32_gate_up_preflight_before_write() {
    let _lease = production_lease();
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for dtype in [GgmlType::IQ3_XXS, GgmlType::Q3_K] {
        let bytes = down::native_bank(dtype, 256, 65, 1);
        let gate = offset_tensor(&ctx, 18, &bytes, 22, vec![256, 65, 1], dtype);
        let x = f32_tensor(&ctx, &[0.1; 256], vec![256]);
        let counts = ints(&ctx, &[1]);
        let ids = ints(&ctx, &[0]);
        let out = f32_tensor(&ctx, &[777.0; 65], vec![65]);
        let name = check_moe_swiglu_clamped_f32x_grouped_slots(
            &ctx, &gate, &gate, &x, &counts, &ids, &out, 256, 65, 1, 1, 1, CLAMP,
        )
        .unwrap();
        assert!(name.contains(if dtype == GgmlType::Q3_K {
            "q3_K"
        } else {
            "iq3_xxs"
        }));
        for case in 0..10 {
            let (mut up, mut input, mut slots, mut output) =
                (gate.clone(), x.clone(), ids.clone(), out.clone());
            let mut k = 256;
            let mut clamp = CLAMP;
            match case {
                0 => {
                    up.dtype = if dtype == GgmlType::Q3_K {
                        GgmlType::IQ3_XXS
                    } else {
                        GgmlType::Q3_K
                    }
                }
                1 => up.offset += 1,
                2 => up.offset = up.buffer.length() as u64,
                3 => input.offset += 4,
                4 => input.dtype = GgmlType::F16,
                5 => slots.dtype = GgmlType::F32,
                6 => output.provenance = MetalTensorProvenance::OwnedWeightReadOnly,
                7 => output = x.view_subrange(0, vec![65]),
                8 => k = 255,
                9 => clamp = f32::NAN,
                _ => unreachable!(),
            }
            let tensors = [&gate, &up, &input, &counts, &slots, &output];
            let before: Vec<_> = tensors.iter().map(|t| tensor_backing_bytes(t)).collect();
            run(&ctx, |enc| {
                assert!(
                    matches!(
                        encode_moe_swiglu_clamped_f32x_grouped_slots(
                            &ctx, enc, &gate, &up, &input, &counts, &slots, &output, k, 65, 1, 1,
                            1, clamp
                        ),
                        Err(MetalError::BadShape { .. })
                    ),
                    "{dtype:?} case {case}"
                );
            });
            for (t, before) in tensors.iter().zip(before) {
                assert_eq!(tensor_backing_bytes(t), before);
            }
        }
    }
}

#[test]
fn native_f32_fused_controlled_clamp_boundaries_and_cancellation() {
    let _lease = production_lease();
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    const K: usize = 6 * 256;
    const LIMIT: f32 = 1.0;
    for dtype in [GgmlType::IQ3_XXS, GgmlType::Q3_K] {
        let scale_offset = if dtype == GgmlType::Q3_K { 108 } else { 0 };
        let gate_block = down::native_bank(dtype, 256, 1, 1);
        let mut up_block = gate_block.clone();
        for (i, byte) in up_block.iter_mut().enumerate() {
            if !(scale_offset..scale_offset + 2).contains(&i) {
                *byte ^= 0x5a;
            }
        }
        up_block[scale_offset..scale_offset + 2]
            .copy_from_slice(&half::f16::from_f32(-0.000317).to_le_bytes());
        let size = gate_block.len();
        let mut gate_bytes = vec![0; 6 * size];
        let mut up_bytes = vec![0; 6 * size];
        for b in 0..3 {
            gate_bytes[b * size..(b + 1) * size].copy_from_slice(&gate_block);
            up_bytes[(b + 3) * size..(b + 4) * size].copy_from_slice(&up_block);
        }
        assert_ne!(gate_bytes, up_bytes);
        let g = dequant_expert(&gate_bytes, dtype, K, 1, 0);
        let u = dequant_expert(&up_bytes, dtype, K, 1, 0);
        assert!(g[768..].iter().all(|&v| v == 0.0));
        assert!(u[..768].iter().all(|&v| v == 0.0));
        let gi = g[..256].iter().position(|v| v.abs() >= 0.0001).unwrap();
        let ui = u[768..1024].iter().position(|v| v.abs() >= 0.0001).unwrap();
        let mut cases = Vec::new();
        let delta = 1.0 / 1024.0;
        for target_g in [1.0 - delta, 1.0, 1.0 + delta, -1.0, -2.0, -8.0] {
            for target_u in [
                1.0 - delta,
                1.0,
                1.0 + delta,
                -1.0 + delta,
                -1.0,
                -1.0 - delta,
                0.25,
                -0.25,
            ] {
                let mut x = vec![0.0; K];
                x[512 + gi] = target_g / g[gi];
                x[1280 + ui] = target_u / u[768 + ui];
                cases.push((
                    format!("{dtype:?} g={target_g} u={target_u}"),
                    x,
                    target_g,
                    target_u,
                    false,
                ));
            }
        }
        for sign in [1.0, -1.0] {
            for residual_first in [false, true] {
                let mut x = vec![0.0; K];
                let (a, b, r) = if residual_first { (1, 2, 0) } else { (0, 1, 2) };
                x[a * 256 + gi] = sign * 1048576.0;
                x[b * 256 + gi] = -sign * 1048576.0;
                x[r * 256 + gi] = sign * 0.25 / g[gi];
                x[(a + 3) * 256 + ui] = -sign * 1048576.0;
                x[(b + 3) * 256 + ui] = sign * 1048576.0;
                x[(r + 3) * 256 + ui] = -sign * 0.5 / u[768 + ui];
                cases.push((
                    format!("{dtype:?} cancellation sign={sign} residual_first={residual_first}"),
                    x,
                    sign * 0.25,
                    -sign * 0.5,
                    true,
                ));
            }
        }
        let n = cases.len();
        let bank = |bytes: &[u8]| offset_tensor(&ctx, 18, bytes, 22, vec![K as u64, 1, 1], dtype);
        let gate = bank(&gate_bytes);
        let up = bank(&up_bytes);
        let x: Vec<_> = cases
            .iter()
            .flat_map(|(_, x, _, _, _)| x.iter().copied())
            .collect();
        let input = f32_tensor(&ctx, &x, vec![K as u64, n as u64]);
        let counts = ints(&ctx, &[n as i32]);
        let ids = ints(&ctx, &(0..n as i32).rev().collect::<Vec<_>>());
        let out = f32_tensor(&ctx, &vec![f32::NAN; n], vec![n as u64]);
        run(&ctx, |enc| {
            encode_moe_swiglu_clamped_f32x_grouped_slots(
                &ctx, enc, &gate, &up, &input, &counts, &ids, &out, K, 1, 1, 1, n, LIMIT,
            )
            .unwrap()
        });
        let actual = read_f32(&out);
        for (row, (label, x, target_g, target_u, cancellation)) in cases.iter().enumerate() {
            let dot = |w: &[f32]| -> f64 {
                w.iter()
                    .zip(x)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum()
            };
            let (g, u) = (dot(&g), dot(&u));
            assert!(
                (g - f64::from(*target_g)).abs() < 1e-6 && (u - f64::from(*target_u)).abs() < 1e-6,
                "{label}: controlled projections {g}/{u}"
            );
            if *cancellation {
                assert!(x.iter().any(|v| v.abs() > 65504.0));
                assert!(g.abs() < f64::from(LIMIT) && u.abs() < f64::from(LIMIT));
            }
            let clamped_g = g.min(f64::from(LIMIT));
            let exact = clamped_g / (1.0 + (-clamped_g).exp())
                * u.clamp(-f64::from(LIMIT), f64::from(LIMIT));
            assert_f64(label, &[actual[row]], &[exact]);
            if g < -1.5 {
                let wrong = -1.0 / (1.0 + 1.0f64.exp()) * u.clamp(-1.0, 1.0);
                assert!(
                    (f64::from(actual[row]) - wrong).abs() > exact.abs() * 0.1,
                    "{label}: negative gate must not be lower-clamped"
                );
            }
        }
        assert_offset_guards(&out, 16, 20);
    }
}
