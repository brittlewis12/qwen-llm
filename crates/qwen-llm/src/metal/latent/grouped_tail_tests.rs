use super::*;
use crate::metal::test_support::{
    assert_offset_guards, metal_test_context, offset_tensor, synthetic_q8_0_bank,
    tensor_backing_bytes, tensor_f32_at_offset,
};

fn f32_tensor(
    ctx: &MetalContext,
    values: &[f32],
    width: usize,
    rows: usize,
    suffix: usize,
) -> MetalTensor {
    offset_tensor(
        ctx,
        20,
        bytemuck::cast_slice(values),
        suffix,
        vec![width as u64, rows as u64],
        GgmlType::F32,
    )
}

// Same peak-scaled tolerance as the established grouped-vs-GEMV regression,
// plus an L2 norm check. No requirement for identical reduction bits.
fn assert_norms(label: &str, got: &[f32], expected: &[impl Copy + Into<f64>]) {
    assert_eq!(got.len(), expected.len());
    let (mut diff2, mut ref2, mut peak, mut worst) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(expected) {
        let b = b.into();
        assert!(a.is_finite() && b.is_finite(), "{label}: nonfinite value");
        let delta = f64::from(a) - b;
        diff2 += delta * delta;
        ref2 += b * b;
        peak = peak.max(b.abs());
        worst = worst.max(delta.abs());
    }
    let relative_l2 = (diff2 / ref2.max(1e-30)).sqrt();
    assert!(
        worst <= 1e-5 * peak.max(1.0),
        "{label}: max={worst}, peak={peak}"
    );
    assert!(relative_l2 <= 1e-5, "{label}: relative L2={relative_l2}");
}

fn reference_samples(
    bytes: &[u8],
    x: &[f32],
    k: usize,
    m: usize,
    groups: usize,
    rows: usize,
) -> (Vec<usize>, Vec<f64>) {
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in [0, rows / 2, rows - 1] {
        for group in 0..groups {
            for channel in [0, 1, 7, 8, 15, 16, m / 2, m - 1] {
                let start = (group * m + channel) * (k / 32) * 34;
                let mut sum = 0.0f64;
                for block in 0..k / 32 {
                    let b = &bytes[start + block * 34..][..34];
                    let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f64();
                    for lane in 0..32 {
                        sum += d
                            * f64::from(b[2 + lane] as i8)
                            * f64::from(x[(row * groups + group) * k + block * 32 + lane]);
                    }
                }
                indices.push((row * groups + group) * m + channel);
                values.push(sum);
            }
        }
    }
    (indices, values)
}

#[test]
fn grouped_q8_tail_glm_shapes_match_cpu_and_gemv_with_physical_tails() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const G: usize = 64;
    for (k, m) in [(256, 512), (512, 256)] {
        let (bytes, _) = synthetic_q8_0_bank(k, m * G);
        let weight = offset_tensor(
            &ctx,
            18,
            &bytes,
            0,
            vec![k as u64, m as u64, G as u64],
            GgmlType::Q8_0,
        );
        let weight_before = tensor_backing_bytes(&weight);
        let full_x: Vec<f32> = (0..k * G * 128)
            .map(|i| (((i * 17 + (i / k) * 53) % 89) as f32 - 44.0) * 0.03)
            .collect();
        // Retain only the F32 readback; release the full-size GPU buffers before
        // allocating each short tail and its independent GEMV output.
        let full_reference = {
            let input = f32_tensor(&ctx, &full_x, k * G, 128, 0);
            let output = f32_tensor(&ctx, &vec![-777.0; m * G * 128], m * G, 128, 28);
            dispatch_census_begin();
            let result = one_shot(&ctx, |enc| {
                encode_mat_mat_q8_0_grouped_f32(&ctx, enc, &weight, &input, &output, k, m, G, 128)
            });
            let census = dispatch_census_take();
            result.unwrap();
            assert_eq!(census.len(), 1);
            assert_eq!(census[0].kernel, "kernel_mat_mat_q8_0_f32_r2c16k64_grouped");
            assert_offset_guards(&output, 20, 28);
            let reference = tensor_f32_at_offset(&output);
            assert!(reference.iter().all(|v| v.is_finite()));
            reference
        };
        for rows in (8..=128).step_by(8) {
            let x = &full_x[..k * G * rows];
            let (indices, reference) = reference_samples(&bytes, x, k, m, G, rows);
            // Zero suffix puts the final group/row exactly at the buffer boundary.
            for suffix in [0, 28] {
                let input = f32_tensor(&ctx, x, k * G, rows, suffix);
                let before = tensor_backing_bytes(&input);
                let output = f32_tensor(&ctx, &vec![-777.0; m * G * rows], m * G, rows, suffix);
                dispatch_census_begin();
                let result = one_shot(&ctx, |enc| {
                    encode_mat_mat_q8_0_grouped_tail_f32(
                        &ctx, enc, &weight, &input, &output, k, m, G, rows,
                    )
                });
                let census = dispatch_census_take();
                result.unwrap();
                assert_eq!(census.len(), 1);
                assert_eq!(census[0].kernel, GROUPED_TAIL_KERNEL);
                assert_eq!(
                    (
                        census[0].grid_width,
                        census[0].grid_height,
                        census[0].grid_depth
                    ),
                    (1, (m / 16) as u64, G as u64)
                );
                assert_eq!(census[0].threads_width, 128);
                let actual = tensor_f32_at_offset(&output);
                assert!(actual.iter().all(|v| v.is_finite()));
                let samples: Vec<_> = indices.iter().map(|&i| actual[i]).collect();
                assert_norms(&format!("CPU {k}->{m} N={rows}"), &samples, &reference);
                let gemv_out = f32_tensor(&ctx, &vec![-777.0; m * G * rows], m * G, rows, suffix);
                one_shot(&ctx, |enc| {
                    for row in 0..rows {
                        encode_mat_vec_q8_0_grouped_f32(
                            &ctx,
                            enc,
                            &weight,
                            &input.view_subrange((row * k * G) as u64, vec![(k * G) as u64]),
                            &gemv_out.view_subrange((row * m * G) as u64, vec![(m * G) as u64]),
                            k,
                            m,
                            G,
                        )?;
                    }
                    Ok(())
                })
                .unwrap();
                let gemv = tensor_f32_at_offset(&gemv_out);
                assert_norms(&format!("GEMV {k}->{m} N={rows}"), &actual, &gemv);
                assert_norms(
                    &format!("full128 prefix {k}->{m} N={rows}"),
                    &actual,
                    &full_reference[..m * G * rows],
                );
                assert_eq!(tensor_backing_bytes(&input), before);
                assert_eq!(tensor_backing_bytes(&weight), weight_before);
                assert_offset_guards(&output, 20, suffix);
                assert_offset_guards(&gemv_out, 20, suffix);
            }
        }
    }
}

#[test]
fn grouped_q8_tail_subviews_ignore_surrounding_rows_and_preserve_output_guards() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const G: usize = 64;
    const ROWS: usize = 24;
    for (k, m) in [(256, 512), (512, 256)] {
        let (bytes, _) = synthetic_q8_0_bank(k, m * G);
        // Exercise the supported flat weight view too.
        let weight = offset_tensor(
            &ctx,
            18,
            &bytes,
            22,
            vec![k as u64, (m * G) as u64],
            GgmlType::Q8_0,
        );
        let x: Vec<f32> = (0..k * G * ROWS)
            .map(|i| (i % 83) as f32 * 0.021 - 0.7)
            .collect();
        let (indices, reference) = reference_samples(&bytes, &x, k, m, G, ROWS);
        let mut baseline: Option<Vec<f64>> = None;
        for poison in [-12345.0, 98765.0] {
            let mut surrounding = vec![poison; k * G * (ROWS + 9)];
            surrounding[k * G * 3..k * G * (ROWS + 3)].copy_from_slice(&x);
            let input = f32_tensor(&ctx, &surrounding, k * G, ROWS + 9, 28);
            let before = tensor_backing_bytes(&input);
            let output = f32_tensor(&ctx, &vec![-777.0; m * G * (ROWS + 9)], m * G, ROWS + 9, 28);
            let x_view = input.view_subrange((3 * k * G) as u64, vec![(k * G) as u64, ROWS as u64]);
            let y_view =
                output.view_subrange((3 * m * G) as u64, vec![(m * G) as u64, ROWS as u64]);
            one_shot(&ctx, |enc| {
                encode_mat_mat_q8_0_grouped_tail_f32(
                    &ctx, enc, &weight, &x_view, &y_view, k, m, G, ROWS,
                )
            })
            .unwrap();
            let got = tensor_f32_at_offset(&y_view);
            assert_norms(
                "subview CPU",
                &indices.iter().map(|&i| got[i]).collect::<Vec<_>>(),
                &reference,
            );
            if let Some(ref reference) = baseline {
                assert_norms("surrounding-row invariance", &got, reference);
            } else {
                baseline = Some(got.iter().copied().map(f64::from).collect::<Vec<_>>());
            }
            let all = tensor_f32_at_offset(&output);
            assert!(
                all[..3 * m * G]
                    .iter()
                    .chain(&all[(ROWS + 3) * m * G..])
                    .all(|&v| v == -777.0)
            );
            assert_eq!(tensor_backing_bytes(&input), before);
            assert_offset_guards(&output, 20, 28);
        }
    }
}

#[test]
fn grouped_q8_tail_rejects_invalid_bindings_before_dispatch() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (k, m, g, rows) = (64, 16, 2, 8);
    let (bytes, _) = synthetic_q8_0_bank(k, m * g);
    let weight = offset_tensor(
        &ctx,
        16,
        &bytes,
        0,
        vec![k as u64, m as u64, g as u64],
        GgmlType::Q8_0,
    );
    let input = f32_tensor(&ctx, &vec![1.0; k * g * rows], k * g, rows, 0);
    let output = f32_tensor(&ctx, &vec![0.0; m * g * rows], m * g, rows, 0);
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let encode = |t: &[MetalTensor; 3], k, m, g, rows| {
        encode_mat_mat_q8_0_grouped_tail_f32(&ctx, &enc, &t[0], &t[1], &t[2], k, m, g, rows)
    };
    let original = [weight.clone(), input.clone(), output.clone()];
    dispatch_census_begin();
    for which in 0..3 {
        for fault in 0..5 {
            let mut t = original.clone();
            match fault {
                0 => t[which].offset += 1,
                1 => t[which].offset = t[which].buffer.length() as u64,
                2 => t[which].offset = u64::MAX - 3,
                3 => t[which].dtype = GgmlType::F16,
                _ => t[which].shape = vec![1],
            }
            assert!(encode(&t, k, m, g, rows).is_err());
        }
    }
    let mut t = original.clone();
    t[2].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(encode(&t, k, m, g, rows).is_err());
    for aliased in [&weight, &input] {
        let mut t = original.clone();
        t[2].buffer = aliased.buffer.clone();
        t[2].offset = aliased.offset;
        assert!(encode(&t, k, m, g, rows).is_err());
    }
    for (k, m, g, rows) in [
        (0, 16, 2, 8),
        (32, 16, 2, 8),
        (64, 0, 2, 8),
        (64, 17, 2, 8),
        (64, 16, 0, 8),
        (64, 16, 2, 0),
        (64, 16, 2, 7),
        (64, 16, 2, 9),
        (64, 16, 2, 127),
        (64, 16, 2, 136),
        (64, 16, usize::MAX, 8),
        (u32::MAX as usize + 1, 16, 1, 8),
        (64, u32::MAX as usize + 1, 1, 8),
        (64, 16, u32::MAX as usize, 8),
    ] {
        assert!(encode(&original, k, m, g, rows).is_err());
    }
    enc.end();
    assert!(dispatch_census_take().is_empty());
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin_concurrent(&command);
    assert!(
        encode_mat_mat_q8_0_grouped_tail_f32(&ctx, &enc, &weight, &input, &output, k, m, g, rows)
            .is_err()
    );
    enc.end();
}

#[test]
fn grouped_q8_tail_capacity_checks() {
    grouped_tail_capacity(32, 128, 0, 4096).unwrap();
    grouped_tail_capacity(32, 128, 32, 4128).unwrap();
    for (width, threads, static_bytes, device_bytes) in [
        (16, 128, 0, 32768),
        (32, 127, 0, 32768),
        (32, 128, 1, 4096),
        (32, 128, usize::MAX, usize::MAX),
    ] {
        assert!(grouped_tail_capacity(width, threads, static_bytes, device_bytes).is_err());
    }
}
