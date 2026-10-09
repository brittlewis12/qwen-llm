//! `encode_grouped_routed_experts_f32x`, the F32-operand grouped routed
//! experts (map #12 accuracy lane), at GLM-5.3 widths over reduced banks:
//! agreement with the per-row decode composition and with an F64 product
//! beside the half-staged grouped path's error, independence of each token's
//! outputs from the rows sharing its dispatch, and refusals.

use super::super::test_support::{dequant_expert, offset_tensor, tensor_backing_bytes};
use super::tests::{
    EXPERTS, FFN, HIDDEN, TOP_K, bank, f32_tensor, read_f32, run, synthetic_bank, values,
};
use super::*;

const CLAMP: f32 = 10.0;

struct Banks {
    gate_dtype: GgmlType,
    down_dtype: GgmlType,
    gate_bytes: Vec<u8>,
    down_bytes: Vec<u8>,
    gate: MetalTensor,
    up: MetalTensor,
    down: MetalTensor,
    up_bytes: Vec<u8>,
}

fn banks(ctx: &MetalContext, gate_dtype: GgmlType, down_dtype: GgmlType) -> Banks {
    let gate_bytes = synthetic_bank(gate_dtype, HIDDEN, FFN, 3);
    let up_bytes = synthetic_bank(gate_dtype, HIDDEN, FFN, 5);
    let down_bytes = synthetic_bank(down_dtype, FFN, HIDDEN, 7);
    Banks {
        gate: bank(ctx, &gate_bytes, gate_dtype, HIDDEN, FFN),
        up: bank(ctx, &up_bytes, gate_dtype, HIDDEN, FFN),
        down: bank(ctx, &down_bytes, down_dtype, FFN, HIDDEN),
        gate_dtype,
        down_dtype,
        gate_bytes,
        up_bytes,
        down_bytes,
    }
}

fn i32_zeros(ctx: &MetalContext, shape: Vec<u64>) -> MetalTensor {
    let n = shape.iter().product::<u64>() as usize;
    offset_tensor(
        ctx,
        16,
        bytemuck::cast_slice(&vec![0i32; n]),
        16,
        shape,
        GgmlType::I32,
    )
}

fn read_i32(tensor: &MetalTensor) -> Vec<i32> {
    let bytes = tensor_backing_bytes(tensor);
    let start = tensor.offset as usize;
    bytemuck::cast_slice(&bytes[start..start + 4 * tensor.n_elements() as usize]).to_vec()
}

/// One grouped run over `rows` rows.
struct Routed {
    output: Vec<f32>,
    slot_out: Vec<f32>,
    ids: Vec<i32>,
    weights: Vec<f32>,
}

/// Routes `rows` rows from `logits` (`[experts, rows]`) and runs the
/// grouped routed experts, F32-operand (`f32x`) or half-staged.
fn grouped(
    ctx: &MetalContext,
    b: &Banks,
    x: &[f32],
    logits: &[f32],
    rows: usize,
    f32x: bool,
) -> Routed {
    let route = LearnedRoute {
        experts: EXPERTS,
        top_k: TOP_K,
        score: RouteScore::Sigmoid,
        routed_scale: 2.5,
    };
    let (h, f, k, e, r) = (
        HIDDEN as u64,
        FFN as u64,
        TOP_K as u64,
        EXPERTS as u64,
        rows as u64,
    );
    let input = f32_tensor(ctx, x, vec![h, r]);
    let logits = f32_tensor(ctx, logits, vec![e, r]);
    let bias = f32_tensor(ctx, &[0.0; EXPERTS], vec![e]);
    let ids = i32_zeros(ctx, vec![k, r]);
    let weights = f32_tensor(ctx, &vec![0.0; TOP_K * rows], vec![k, r]);
    let status = i32_zeros(ctx, vec![r]);
    let counts = i32_zeros(ctx, vec![e]);
    let slots = i32_zeros(ctx, vec![e * r]);
    let inner = f32_tensor(ctx, &vec![0.0; FFN * TOP_K * rows], vec![f, k * r]);
    let slot_out = f32_tensor(ctx, &vec![0.0; HIDDEN * TOP_K * rows], vec![h, k * r]);
    let output = f32_tensor(ctx, &vec![7.0; HIDDEN * rows], vec![h, r]);
    run(ctx, |enc| {
        encode_route_learned_rows(
            ctx, enc, &route, rows, &logits, &bias, &ids, &weights, &status,
        )
        .unwrap();
        let g = GroupedExperts {
            gate_bank: &b.gate,
            up_bank: &b.up,
            down_bank: &b.down,
            input: &input,
            ids: &ids,
            weights: &weights,
            counts: &counts,
            slots: &slots,
            inner: &inner,
            slot_out: &slot_out,
            output: &output,
        };
        if f32x {
            encode_grouped_routed_experts_f32x(
                ctx, enc, &g, HIDDEN, FFN, EXPERTS, TOP_K, rows, CLAMP,
            )
            .unwrap();
        } else {
            encode_grouped_routed_experts(ctx, enc, &g, HIDDEN, FFN, EXPERTS, TOP_K, rows, CLAMP)
                .unwrap();
        }
    });
    Routed {
        output: read_f32(&output),
        slot_out: read_f32(&slot_out),
        ids: read_i32(&ids),
        weights: read_f32(&weights),
    }
}

/// The per-row decode composition (all-slot gate/up SwiGLU and down, then
/// the weighted sum) on the given routes: F32 dequantization and operands.
fn decode_reference(ctx: &MetalContext, b: &Banks, x: &[f32], routed: &Routed) -> Vec<f32> {
    let rows = x.len() / HIDDEN;
    let (h, f, k) = (HIDDEN as u64, FFN as u64, TOP_K as u64);
    let mut reference = Vec::with_capacity(HIDDEN * rows);
    for row in 0..rows {
        let ids = offset_tensor(
            ctx,
            16,
            bytemuck::cast_slice(&routed.ids[row * TOP_K..(row + 1) * TOP_K]),
            16,
            vec![k],
            GgmlType::I32,
        );
        let ready = offset_tensor(
            ctx,
            16,
            bytemuck::cast_slice(&[ROUTE_STATUS_READY]),
            16,
            vec![1],
            GgmlType::I32,
        );
        let row_x = f32_tensor(ctx, &x[row * HIDDEN..(row + 1) * HIDDEN], vec![h]);
        let inner = f32_tensor(ctx, &vec![0.0; FFN * TOP_K], vec![f, k]);
        let out = f32_tensor(ctx, &vec![0.0; HIDDEN * TOP_K], vec![h, k]);
        run(ctx, |enc| {
            encode_all_slots_gate_up_swiglu(
                ctx, enc, &b.gate, &b.up, &row_x, &ids, &ready, &inner, HIDDEN, FFN, EXPERTS,
                TOP_K, CLAMP,
            )
            .unwrap();
            encode_all_slots_down(
                ctx, enc, &b.down, &inner, &ids, &ready, &out, FFN, HIDDEN, EXPERTS, TOP_K,
            )
            .unwrap();
        });
        let slots = read_f32(&out);
        for d in 0..HIDDEN {
            reference.push(
                (0..TOP_K)
                    .map(|s| routed.weights[row * TOP_K + s] * slots[s * HIDDEN + d])
                    .sum::<f32>(),
            );
        }
    }
    reference
}

fn relative_rms(actual: &[f32], reference: &[f64]) -> f64 {
    assert_eq!(actual.len(), reference.len());
    assert!(actual.iter().all(|v| v.is_finite()), "non-finite output");
    let (mut diff, mut norm) = (0.0f64, 0.0f64);
    for (&a, &r) in actual.iter().zip(reference) {
        diff += (f64::from(a) - r).powi(2);
        norm += r * r;
    }
    (diff / norm).sqrt()
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// Against the per-row decode composition on the same routes, the F32 path
/// agrees within 1e-5 relative RMS, at least 20x closer than the
/// half-staged grouped path; two routed slots of row 0 agree with an F64
/// product of llama.cpp-dequantized weights (gate/up, clamped SwiGLU, down)
/// within 1e-5. Gate/up IQ2_S with IQ3_S and IQ4_XS down, and IQ3_S gate/up
/// with IQ4_XS down (GLM block 11).
#[test]
fn grouped_f32x_experts_track_decode_and_f64_beyond_half() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    const ROWS: usize = 24;
    let x = values(HIDDEN * ROWS, 3);
    let logits: Vec<f32> = (0..EXPERTS * ROWS)
        .map(|i| ((i * 29 + i / 7) % 53) as f32 * 0.09 - 2.0)
        .collect();
    for (gate_dtype, down_dtype) in [
        (GgmlType::IQ2_S, GgmlType::IQ3_S),
        (GgmlType::IQ2_S, GgmlType::IQ4_XS),
        (GgmlType::IQ3_S, GgmlType::IQ4_XS),
    ] {
        let label = format!("{gate_dtype:?}/{down_dtype:?}");
        let b = banks(&ctx, gate_dtype, down_dtype);
        let f32x = grouped(&ctx, &b, &x, &logits, ROWS, true);
        let half = grouped(&ctx, &b, &x, &logits, ROWS, false);
        assert_eq!(f32x.ids, half.ids, "{label}: routes");
        let decode = widen(&decode_reference(&ctx, &b, &x, &f32x));
        let (f32x_rms, half_rms) = (
            relative_rms(&f32x.output, &decode),
            relative_rms(&half.output, &decode),
        );
        // Two slots of row 0 against F64 products of the dequantized banks.
        let mut exact = Vec::new();
        let mut slot_values = Vec::new();
        for s in 0..2 {
            let expert = f32x.ids[s] as usize;
            let wg = dequant_expert(&b.gate_bytes, b.gate_dtype, HIDDEN, FFN, expert);
            let wu = dequant_expert(&b.up_bytes, b.gate_dtype, HIDDEN, FFN, expert);
            let wd = dequant_expert(&b.down_bytes, b.down_dtype, FFN, HIDDEN, expert);
            let dot = |w: &[f32], v: &[f64]| -> f64 {
                w.iter().zip(v).map(|(&w, &v)| f64::from(w) * v).sum()
            };
            let x0 = widen(&x[..HIDDEN]);
            let clamp = f64::from(CLAMP);
            let inner: Vec<f64> = (0..FFN)
                .map(|r| {
                    let g = dot(&wg[r * HIDDEN..(r + 1) * HIDDEN], &x0).min(clamp);
                    let u = dot(&wu[r * HIDDEN..(r + 1) * HIDDEN], &x0).clamp(-clamp, clamp);
                    g / (1.0 + (-g).exp()) * u
                })
                .collect();
            exact.extend((0..HIDDEN).map(|d| dot(&wd[d * FFN..(d + 1) * FFN], &inner)));
            slot_values.extend_from_slice(&f32x.slot_out[s * HIDDEN..(s + 1) * HIDDEN]);
        }
        let f64_rms = relative_rms(&slot_values, &exact);
        eprintln!(
            "{label}: F32 vs decode {f32x_rms:.3e}, half vs decode {half_rms:.3e}; F32 slots vs F64 {f64_rms:.3e}"
        );
        assert!(f32x_rms <= 1e-5, "{label}: F32 vs decode {f32x_rms:e}");
        assert!(
            f32x_rms * 20.0 < half_rms,
            "{label}: F32 {f32x_rms:e} vs half {half_rms:e}"
        );
        assert!(f64_rms <= 1e-5, "{label}: F32 slots vs F64 {f64_rms:e}");
    }
}

/// Each token's routed output is bitwise the same whether its rows run in one
/// 64-row dispatch (expert buckets past 32 slots: partial 16- and 32-slot
/// tiles) or in sub-dispatches of 1, 17, 33 and 13 rows (buckets of 0 to a
/// few slots), so neither a bucket's size nor a token's place in it changes
/// the token's outputs.
#[test]
fn grouped_f32x_experts_token_outputs_do_not_depend_on_the_dispatch() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    const ROWS: usize = 64;
    let x = values(HIDDEN * ROWS, 5);
    // Logits laid out [experts, rows] row-major by row: row r is
    // logits[r * EXPERTS..(r + 1) * EXPERTS].
    let logits: Vec<f32> = (0..EXPERTS * ROWS)
        .map(|i| ((i * 37 + i / 5) % 61) as f32 * 0.08 - 2.4)
        .collect();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for down_dtype in [GgmlType::IQ3_S, GgmlType::IQ4_XS] {
        let b = banks(&ctx, GgmlType::IQ2_S, down_dtype);
        let full = grouped(&ctx, &b, &x, &logits, ROWS, true);
        let mut counts = [0usize; EXPERTS];
        for &id in &full.ids {
            counts[id as usize] += 1;
        }
        assert!(
            counts.iter().any(|&c| c > 32),
            "a bucket spans more than one 32-slot tile: {counts:?}"
        );
        let mut start = 0;
        for rows in [1usize, 17, 33, 13] {
            let part = grouped(
                &ctx,
                &b,
                &x[start * HIDDEN..(start + rows) * HIDDEN],
                &logits[start * EXPERTS..(start + rows) * EXPERTS],
                rows,
                true,
            );
            assert_eq!(
                part.ids,
                full.ids[start * TOP_K..(start + rows) * TOP_K],
                "{down_dtype:?} rows {start}..{}: routes",
                start + rows
            );
            assert_eq!(
                bits(&part.output),
                bits(&full.output[start * HIDDEN..(start + rows) * HIDDEN]),
                "{down_dtype:?} rows {start}..{}",
                start + rows
            );
            start += rows;
        }
        assert_eq!(start, ROWS);
    }
}

/// Expert types without an F32-operand tile are refused before encoding.
#[test]
fn grouped_f32x_experts_refuse_types_without_a_tile() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    const ROWS: usize = 4;
    for (gate_dtype, down_dtype) in [
        (GgmlType::IQ4_XS, GgmlType::IQ3_S),
        (GgmlType::IQ2_S, GgmlType::IQ2_S),
    ] {
        let b = banks(&ctx, gate_dtype, down_dtype);
        let (h, f, k, e, r) = (
            HIDDEN as u64,
            FFN as u64,
            TOP_K as u64,
            EXPERTS as u64,
            ROWS as u64,
        );
        let input = f32_tensor(&ctx, &values(HIDDEN * ROWS, 1), vec![h, r]);
        let ids = i32_zeros(&ctx, vec![k, r]);
        let weights = f32_tensor(&ctx, &[0.0; TOP_K * ROWS], vec![k, r]);
        let counts = i32_zeros(&ctx, vec![e]);
        let slots = i32_zeros(&ctx, vec![e * r]);
        let inner = f32_tensor(&ctx, &vec![0.0; FFN * TOP_K * ROWS], vec![f, k * r]);
        let slot_out = f32_tensor(&ctx, &vec![0.0; HIDDEN * TOP_K * ROWS], vec![h, k * r]);
        let output = f32_tensor(&ctx, &vec![0.0; HIDDEN * ROWS], vec![h, r]);
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let result = encode_grouped_routed_experts_f32x(
            &ctx,
            &enc,
            &GroupedExperts {
                gate_bank: &b.gate,
                up_bank: &b.up,
                down_bank: &b.down,
                input: &input,
                ids: &ids,
                weights: &weights,
                counts: &counts,
                slots: &slots,
                inner: &inner,
                slot_out: &slot_out,
                output: &output,
            },
            HIDDEN,
            FFN,
            EXPERTS,
            TOP_K,
            ROWS,
            CLAMP,
        );
        enc.end();
        assert!(
            matches!(result, Err(MetalError::BadShape { .. })),
            "{gate_dtype:?}/{down_dtype:?}: {result:?}"
        );
    }
}
