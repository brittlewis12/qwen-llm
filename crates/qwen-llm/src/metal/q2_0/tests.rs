use super::*;
use crate::metal::test_support::{
    assert_offset_guards, metal_test_context, offset_tensor, tensor_f32_at_offset,
};

fn bank(k: usize, rows: usize, experts: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut bytes = Vec::new();
    for b in 0..k / 64 * rows * experts {
        let scale = if b % 11 == 0 {
            0.0
        } else {
            (b % 13 + 1) as f32 / 256.0
        };
        let scale = if b % 3 == 0 { -scale } else { scale };
        bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        for _ in 0..16 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            bytes.push((state >> 24) as u8);
        }
    }
    bytes
}

// Decode storage directly, independently of the quantizer and Metal tile code.
fn decode(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(18)
        .flat_map(|b| {
            let scale = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f64();
            b[2..].iter().flat_map(move |&q| {
                [q & 3, (q >> 2) & 3, (q >> 4) & 3, q >> 6]
                    .map(|code| (f64::from(code) - 1.0) * scale)
            })
        })
        .collect()
}

fn inputs(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 17 + i / 19) % 61) as f32 / 32.0 - 0.9375)
        .collect()
}

fn dot(w: &[f64], x: &[f32]) -> f64 {
    assert_eq!(w.len(), x.len());
    w.iter().zip(x).map(|(&w, &x)| w * f64::from(x)).sum()
}

fn close(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite(),
            "non-finite at {i}: {a} vs {e}"
        );
        assert!(
            (f64::from(a) - e).abs() <= tolerance * (1.0 + e.abs()),
            "element {i}: {a} vs {e}"
        );
    }
}

fn f32_tensor(ctx: &MetalContext, data: &[f32], shape: Vec<u64>) -> MetalTensor {
    offset_tensor(
        ctx,
        16,
        bytemuck::cast_slice(data),
        20,
        shape,
        GgmlType::F32,
    )
}

#[test]
fn q2_0_dense_and_all_slots_match_f64_offsets_tails() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for (k, rows) in [(64, 1), (128, 67), (640, 129)] {
        let experts = 7;
        let routes = [6i32, 1, 6, 0];
        let bytes = bank(k, rows, experts, 37);
        let decoded = decode(&bytes);
        let w = offset_tensor(
            &ctx,
            18,
            &bytes,
            22,
            vec![k as u64, rows as u64, experts as u64],
            GgmlType::Q2_0,
        );
        let x_data = inputs(k * routes.len());
        let x = f32_tensor(&ctx, &x_data, vec![k as u64, routes.len() as u64]);
        let y = f32_tensor(
            &ctx,
            &vec![f32::NAN; rows * routes.len()],
            vec![rows as u64, routes.len() as u64],
        );
        let mut expected = Vec::new();
        for (slot, &expert) in routes.iter().enumerate() {
            for row in 0..rows {
                let base = (expert as usize * rows + row) * k;
                expected.push(dot(
                    &decoded[base..base + k],
                    &x_data[slot * k..(slot + 1) * k],
                ));
            }
        }
        for dtype in [GgmlType::I32, GgmlType::F32] {
            let ids = offset_tensor(
                &ctx,
                12,
                bytemuck::cast_slice(&routes),
                20,
                vec![routes.len() as u64],
                dtype,
            );
            one_shot(&ctx, |enc| {
                encode_moe_down_q2_0_f32(
                    &ctx,
                    enc,
                    &w,
                    &x,
                    &ids,
                    &y,
                    k,
                    rows,
                    experts,
                    routes.len(),
                )
            })
            .unwrap();
            close(&tensor_f32_at_offset(&y), &expected, 2e-5);
            assert_offset_guards(&ids, 12, 20);
        }
        // A bank view starting at a nonzero expert also exercises dense weight offsets.
        let dense = w.view_bytes((6 * rows * k / 64 * 18) as u64, vec![k as u64, rows as u64]);
        let xv = x.view_subrange(0, vec![k as u64]);
        let yd = f32_tensor(&ctx, &vec![f32::NAN; rows], vec![rows as u64]);
        one_shot(&ctx, |enc| {
            encode_mat_vec_q2_0_f32(&ctx, enc, &dense, &xv, &yd, k, rows)
        })
        .unwrap();
        close(&tensor_f32_at_offset(&yd), &expected[..rows], 2e-5);
        assert_offset_guards(&w, 18, 22);
        for t in [&x, &y, &yd] {
            assert_offset_guards(t, 16, 20);
        }
    }
}

#[test]
fn q2_0_invalid_routes_zero_all_rows() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (k, rows, experts) = (64, 5, 4);
    let bytes = bank(k, rows, experts, 53);
    let w = offset_tensor(&ctx, 18, &bytes, 20, vec![64, 5, 4], GgmlType::Q2_0);
    let x = f32_tensor(&ctx, &inputs(k * 4), vec![64, 4]);
    let y = f32_tensor(&ctx, &vec![f32::NAN; rows * 4], vec![5, 4]);
    let ids = offset_tensor(
        &ctx,
        4,
        bytemuck::cast_slice(&[-1i32, 4, i32::MIN, i32::MAX]),
        20,
        vec![4],
        GgmlType::I32,
    );
    one_shot(&ctx, |enc| {
        encode_moe_down_q2_0_f32(&ctx, enc, &w, &x, &ids, &y, k, rows, experts, 4)
    })
    .unwrap();
    close(&tensor_f32_at_offset(&y), &vec![0.0; rows * 4], 0.0);
    assert_offset_guards(&y, 16, 20);
}

#[test]
fn q2_0_checked_contracts_reject_before_encoding() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let w =
        MetalTensor::from_bytes(&ctx, &bank(64, 5, 2, 1), vec![64, 5, 2], GgmlType::Q2_0).unwrap();
    let x = MetalTensor::zeros_f32(&ctx, vec![64, 2]).unwrap();
    let y = MetalTensor::zeros_f32(&ctx, vec![5, 2]).unwrap();
    let ids = MetalTensor::zeros_i32(&ctx, vec![2]).unwrap();
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    let reject = |w: &MetalTensor,
                  x: &MetalTensor,
                  ids: &MetalTensor,
                  y: &MetalTensor,
                  k,
                  rows,
                  experts,
                  topk| {
        assert!(
            encode_moe_down_q2_0_f32(&ctx, &enc, w, x, ids, y, k, rows, experts, topk).is_err()
        );
    };
    for dims in [
        (0, 5, 2, 2),
        (32, 5, 2, 2),
        (64, 0, 2, 2),
        (64, 5, 0, 2),
        (64, 5, 2, 0),
        (64, 5, 2, 3),
        (1usize << 32, 5, 2, 2),
        (64, 1usize << 32, 2, 2),
        (64, 5, 1usize << 32, 2),
        (
            u32::MAX as usize - 63,
            u32::MAX as usize,
            u32::MAX as usize,
            2,
        ),
    ] {
        reject(&w, &x, &ids, &y, dims.0, dims.1, dims.2, dims.3);
    }
    for which in 0..4 {
        for fault in 0..5 {
            let mut tensors = [w.clone(), x.clone(), ids.clone(), y.clone()];
            let t = &mut tensors[which];
            match fault {
                0 => t.dtype = GgmlType::F16,
                1 => t.offset = 1,
                2 => t.offset = t.buffer.length() as u64,
                3 => t.offset = u64::MAX - 3,
                _ => t.shape = vec![1],
            }
            reject(
                &tensors[0],
                &tensors[1],
                &tensors[2],
                &tensors[3],
                64,
                5,
                2,
                2,
            );
        }
    }
    let mut readonly = y.clone();
    readonly.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    reject(&w, &x, &ids, &readonly, 64, 5, 2, 2);
    let wd = w.view_bytes(0, vec![64, 5]);
    let xd = x.view_subrange(0, vec![64]);
    let yd = y.view_subrange(0, vec![5]);
    for (k, rows) in [(0, 5), (32, 5), (64, 0), (64, 6), (1usize << 32, 5)] {
        assert!(encode_mat_vec_q2_0_f32(&ctx, &enc, &wd, &xd, &yd, k, rows).is_err());
    }
    let mut readonly = yd.clone();
    readonly.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(encode_mat_vec_q2_0_f32(&ctx, &enc, &wd, &xd, &readonly, 64, 5).is_err());
    enc.end();
}

#[test]
fn q2_0_grouped_all_roles_match_f64_repeated_sparse_buckets() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const TOKENS: usize = 37;
    const TOPK: usize = 2;
    const EXPERTS: usize = 7;
    let routes: Vec<usize> = (0..TOKENS)
        .flat_map(|t| [5, if t % 3 == 0 { 3 } else { 1 }])
        .collect();
    let mut buckets = vec![Vec::new(); EXPERTS];
    for (slot, &expert) in routes.iter().enumerate() {
        buckets[expert].push(slot as i32);
    }
    let counts: Vec<i32> = buckets.iter().map(|b| b.len() as i32).collect();
    let mut ids = vec![-1i32; EXPERTS * TOKENS];
    for (e, slots) in buckets.iter_mut().enumerate() {
        slots.reverse();
        ids[e * TOKENS..e * TOKENS + slots.len()].copy_from_slice(slots);
    }
    let counts_t = offset_tensor(
        &ctx,
        12,
        bytemuck::cast_slice(&counts),
        20,
        vec![EXPERTS as u64],
        GgmlType::I32,
    );
    let ids_t = offset_tensor(
        &ctx,
        12,
        bytemuck::cast_slice(&ids),
        20,
        vec![TOKENS as u64, EXPERTS as u64],
        GgmlType::I32,
    );
    for (k, rows) in [(64, 5), (128, 67), (640, 129)] {
        let gate_bytes = bank(k, rows, EXPERTS, 127);
        let up_bytes = bank(k, rows, EXPERTS, 953);
        let gate = decode(&gate_bytes);
        let up = decode(&up_bytes);
        let shape = vec![k as u64, rows as u64, EXPERTS as u64];
        let gate_t = offset_tensor(&ctx, 18, &gate_bytes, 22, shape.clone(), GgmlType::Q2_0);
        let up_t = offset_tensor(&ctx, 18, &up_bytes, 22, shape.clone(), GgmlType::Q2_0);
        let dense_gate: Vec<f32> = gate.iter().map(|&v| v as f32).collect();
        let dense_gate_t = f32_tensor(&ctx, &dense_gate, shape);
        let slot_input = inputs(k * TOKENS * TOPK);
        let token_input = inputs(k * TOKENS);
        let slot_t = f32_tensor(&ctx, &slot_input, vec![k as u64, (TOKENS * TOPK) as u64]);
        let token_t = f32_tensor(&ctx, &token_input, vec![k as u64, TOKENS as u64]);
        let mut down_ref = Vec::new();
        let mut swiglu_ref = Vec::new();
        for (slot, &e) in routes.iter().enumerate() {
            for row in 0..rows {
                let base = (e * rows + row) * k;
                down_ref.push(dot(
                    &up[base..base + k],
                    &slot_input[slot * k..(slot + 1) * k],
                ));
                let x = &token_input[(slot / TOPK) * k..(slot / TOPK + 1) * k];
                let g = dot(&gate[base..base + k], x);
                let u = dot(&up[base..base + k], x);
                swiglu_ref.push(g / (1.0 + (-g).exp()) * u);
            }
        }
        for restricted in [false, true] {
            let (min, max) = if restricted {
                (14, 32)
            } else {
                (0, i32::MAX as u32)
            };
            for role in 0..3 {
                let out = f32_tensor(
                    &ctx,
                    &vec![-123.0; rows * TOKENS * TOPK],
                    vec![rows as u64, (TOKENS * TOPK) as u64],
                );
                one_shot(&ctx, |enc| {
                    if role == 0 {
                        encode_moe_down_f32_grouped_slots_generic_range(
                            &ctx, enc, &up_t, &slot_t, &counts_t, &ids_t, &out, k, rows, EXPERTS,
                            TOKENS, min, max,
                        )
                    } else {
                        // Mixed F32 gate / Q2_0 up selects the Q2_0 up_silu_mul instantiation.
                        let gate = if role == 1 { &gate_t } else { &dense_gate_t };
                        encode_moe_swiglu_f32_grouped_slots_generic_range(
                            &ctx, enc, gate, &up_t, &token_t, &counts_t, &ids_t, &out, k, rows,
                            EXPERTS, TOPK, TOKENS, min, max,
                        )
                    }
                })
                .unwrap();
                let mut expected = if role == 0 {
                    down_ref.clone()
                } else {
                    swiglu_ref.clone()
                };
                if restricted {
                    for (slot, &e) in routes.iter().enumerate() {
                        if counts[e] < min as i32 || counts[e] > max as i32 {
                            expected[slot * rows..(slot + 1) * rows].fill(-123.0);
                        }
                    }
                }
                close(&tensor_f32_at_offset(&out), &expected, 3e-4);
                assert_offset_guards(&out, 16, 20);
            }
        }
        for t in [&gate_t, &up_t] {
            assert_offset_guards(t, 18, 22);
        }
        for t in [&dense_gate_t, &slot_t, &token_t] {
            assert_offset_guards(t, 16, 20);
        }
    }
    for t in [&counts_t, &ids_t] {
        assert_offset_guards(t, 12, 20);
    }
}

#[test]
fn q2_0_nl4_tile_addressing_and_instantiations_cpu() {
    let l = moe_grouped_generic_layout(GgmlType::Q2_0).unwrap();
    assert_eq!((l.block_elems, l.block_bytes, l.suffix), (64, 18, "q2_0"));
    let tiles = include_str!("../../../../../kernels/quant_tiles.h");
    assert!(tiles.contains("#define QT_Q2_0_BYTES     18"));
    assert!(tiles.contains("#define QT_Q2_0_NL        4"));
    let source = include_str!("../../../../../kernels/moe.metal");
    for role in MoeGroupedGenericRole::ALL {
        let name = moe_grouped_generic_pipeline_name(role, GgmlType::Q2_0).unwrap();
        let line = source
            .lines()
            .find(|line| line.contains(&format!("host_name(\"{name}\")")))
            .unwrap();
        assert!(line.contains("QT_Q2_0_BYTES, QT_Q2_0_NL, qt_dequantize_q2_0"));
    }
    assert!(clamped_swiglu_pipeline_name(GgmlType::Q2_0).is_none());
    // Model the existing 32-element K-step for both loader lanes, across row/expert boundaries.
    for k in [64, 128, 640] {
        for expert in 0..3 {
            for row in 0..5 {
                let base = (expert * 5 + row) * (k / 64) * 18;
                for il0 in 0..2 {
                    let mut il = il0;
                    let mut ptr = base;
                    for step in (0..k).step_by(32) {
                        for i in 0..16 {
                            let logical = step + 16 * il0 + i;
                            assert_eq!(
                                ptr + 2 + (16 * il + i) / 4,
                                base + (logical / 64) * 18 + 2 + (logical % 64) / 4
                            );
                            assert_eq!((16 * il + i) % 4, logical % 4);
                        }
                        il = if il + 2 < 4 { il + 2 } else { il % 2 };
                        if il < 2 {
                            ptr += 18;
                        }
                    }
                    assert_eq!(ptr, base + k / 64 * 18);
                }
            }
        }
    }
}
