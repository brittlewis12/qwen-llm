//! Exact dyadic CPU-codec/F64 witnesses; scalar Metal is not the oracle.
use super::*;

fn code(bytes: &mut [u8], k: usize, value: u8) {
    let byte = &mut bytes[k / 32 * 17 + 1 + k % 16];
    let shift = (k % 32 / 16) * 4;
    *byte = (*byte & !(15 << shift)) | (value << shift);
}

fn check(ctx: &MetalContext, label: &str, bytes: &[u8], x: &[f32], expected: f64) -> bool {
    let k = x.len();
    let decoded = dequant_expert(bytes, GgmlType::MXFP4, k, 1, 0);
    let mut terms: Vec<f64> = decoded
        .iter()
        .zip(x)
        .map(|(&w, &x)| f64::from(w) * f64::from(x))
        .collect();
    // Sum small magnitudes first; the closed-form dyadic result below must
    // agree exactly, including witnesses spanning ordinary/edge scale ranges.
    terms.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
    let exact: f64 = terms.iter().sum();
    assert_eq!(exact, expected, "{label}: independent codec/F64 oracle");
    assert!((exact as f32).is_normal(), "{label}: normal final result");
    assert!(decoded.iter().chain(x).all(|v| v.is_finite()));
    let weight = offset_tensor(ctx, 18, bytes, 22, vec![k as u64, 1, 1], GgmlType::MXFP4);
    let input = f32_tensor(ctx, x, vec![k as u64]);
    let ints = |values: &[i32]| {
        offset_tensor(
            ctx,
            12,
            bytemuck::cast_slice(values),
            20,
            vec![values.len() as u64],
            GgmlType::I32,
        )
    };
    let counts = ints(&[1]);
    let ids = ints(&[0]);
    let out = f32_tensor(ctx, &[f32::NAN], vec![1]);
    run(ctx, |enc| {
        encode_moe_down_f32x_grouped_slots(
            ctx, enc, &weight, &input, &counts, &ids, &out, k, 1, 1, 1,
        )
        .unwrap()
    });
    let actual = read_f32(&out)[0];
    let error = (f64::from(actual) - exact).abs();
    let relative = error / exact.abs();
    eprintln!("{label}: actual={actual:e}, exact={exact:e}, relative={relative:e}");
    assert_offset_guards(&out, 16, 20);
    actual.is_finite() && relative <= 1e-5 && error <= exact.abs() * 2e-5
}

#[test]
fn mxfp4_reduction_small_terms_ordinary_cancellation_and_bounded_inputs() {
    let _lease = production_lease();
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    const K: usize = 2304;
    let mut passed = true;
    for bounded in [false, true] {
        for cancel in [false, true] {
            // The bounded positive witness has all activations <= 8. The
            // cancellation witness is separately finite, but not bounded by 8.
            let exponent = if bounded { 28 } else { 128 };
            let ordinary = 2.0f64.powi(i32::from(exponent) - 128);
            let tiny = ordinary * 2.0f64.powi(-25);
            let mut bytes = vec![0u8; K / 32 * 17];
            bytes[0] = exponent;
            code(&mut bytes, 0, 1);
            let mut x = vec![0.0; K];
            x[0] = 1.0;
            for i in 32..K {
                code(&mut bytes, i, 1);
                x[i] = if bounded { 8.0 } else { 2.0f32.powi(103) };
            }
            let exact = if cancel {
                code(&mut bytes, K - 1, 4);
                x[K - 1] = -(ordinary * 2.0f64.powi(126)) as f32;
                (K - 33) as f64 * tiny
            } else {
                ordinary + (K - 32) as f64 * tiny
            };
            for sign in [1.0, -1.0] {
                let x: Vec<_> = x.iter().map(|v| v * sign).collect();
                passed &= check(
                    &ctx,
                    &format!("small terms bounded={bounded} cancel={cancel} sign={sign}"),
                    &bytes,
                    &x,
                    exact * f64::from(sign),
                );
            }
        }
    }
    assert!(
        passed,
        "small-term reduction must retain the existing F64 bounds"
    );
}

#[test]
fn mxfp4_reduction_ftz_products_and_cancellation_have_normal_final_values() {
    let _lease = production_lease();
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let mut passed = true;
    for (label, exponent, codes, exact) in [
        ("subnormal products", 1, vec![1; 32], 2.0f64.powi(-122)),
        (
            "subnormal partial cancellation",
            0,
            vec![5, 12, 4],
            6.0 * 2.0f64.powi(-128),
        ),
        ("smallest normal sum", 0, vec![1; 4], 2.0f64.powi(-126)),
    ] {
        let mut bytes = vec![0; 17];
        bytes[0] = exponent;
        let mut x = vec![0.0; 32];
        for (i, code_value) in codes.iter().enumerate() {
            code(&mut bytes, i, *code_value);
            x[i] = 1.0;
        }
        for sign in [1.0, -1.0] {
            let x: Vec<_> = x.iter().map(|v| v * sign).collect();
            passed &= check(&ctx, label, &bytes, &x, exact * f64::from(sign));
        }
    }
    assert!(
        passed,
        "FTZ intermediates must not erase these normal final results"
    );
}

#[test]
fn mxfp4_reduction_extreme_finite_and_normal_scale_boundaries() {
    let _lease = production_lease();
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let mut passed = true;
    for (exponent, c, activation) in [
        (0, 7, f32::MAX),
        (1, 7, f32::MAX),
        (2, 1, 1.0),
        (254, 3, 1.0 / 64.0),
        (255, 1, 1.0 / 64.0),
    ] {
        let mut bytes = vec![0; 17];
        bytes[0] = exponent;
        for i in 0..32 {
            code(&mut bytes, i, c);
        }
        let integer = match c {
            1 => 1.0,
            3 => 3.0,
            7 => 12.0,
            _ => unreachable!(),
        };
        let exact = 32.0 * integer * 2.0f64.powi(i32::from(exponent) - 128) * f64::from(activation);
        for sign in [1.0, -1.0] {
            passed &= check(
                &ctx,
                &format!("finite exponent={exponent} sign={sign}"),
                &bytes,
                &[activation * sign; 32],
                exact * f64::from(sign),
            );
        }
    }
    assert!(
        passed,
        "finite edge reductions and ordinary scale boundaries"
    );
}

#[test]
fn mxfp4_reduction_product_roundoff_and_large_ordinary_merge() {
    let _lease = production_lease();
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let mut passed = true;
    // 3*a - 2*a - a is exactly zero, but the first F32 product rounds.
    // Keeping only sum errors (not the FMA product residual) fails this case.
    for residual_first in [false, true] {
        let mut bytes = vec![0; 17];
        let mut x = vec![0.0; 32];
        let start = usize::from(residual_first);
        for (i, c) in [3, 10, 9].iter().enumerate() {
            code(&mut bytes, start + i, *c);
            x[start + i] = f32::MAX;
        }
        let residual = if residual_first { 0 } else { 3 };
        code(&mut bytes, residual, 1);
        x[residual] = 2.0f32.powi(103);
        for sign in [1.0, -1.0] {
            let x: Vec<_> = x.iter().map(|v| v * sign).collect();
            passed &= check(
                &ctx,
                "product roundoff cancellation",
                &bytes,
                &x,
                f64::from(sign) * 2.0f64.powi(-25),
            );
        }
    }
    for (exponent, ordinary_x, cancel_edges) in [
        (192, f32::from_bits(1.0f32.to_bits() - 1), false),
        (192, 1.0, false),
        (255, f32::from_bits(2.0f32.to_bits() - 1), true),
    ] {
        const K: usize = 1056;
        let mut bytes = vec![0; K / 32 * 17];
        bytes[0] = exponent;
        code(&mut bytes, 0, 1);
        let mut x = vec![0.0; K];
        x[0] = ordinary_x;
        for i in 32..K {
            bytes[i / 32 * 17] = 1;
            code(
                &mut bytes,
                i,
                if cancel_edges && i % 2 == 0 { 12 } else { 4 },
            );
            x[i] = 2.0f32.powi(127);
        }
        let exact = f64::from(ordinary_x) * 2.0f64.powi(i32::from(exponent) - 128)
            + if cancel_edges { 0.0 } else { 4096.0 };
        for sign in [1.0, -1.0] {
            let x: Vec<_> = x.iter().map(|v| v * sign).collect();
            passed &= check(
                &ctx,
                "large ordinary merge stays finite",
                &bytes,
                &x,
                f64::from(sign) * exact,
            );
        }
    }
    assert!(
        passed,
        "compensated products and finite ordinary/edge merge"
    );
}
