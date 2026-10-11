//! Small native-weight composition checks. GPU execution is left to the owner.
use super::*;
use crate::deepseek_v4_oracle::{hyper_connection_pre, rms_norm};
use crate::metal::test_support::{
    dequant_expert, encode_iq4_nl_block, encode_mxfp4_block, encode_q6_k_block, offset_tensor,
    synthetic_dense_linear_bank, synthetic_q8_0_bank, tensor_f32_at_offset,
};

const H: usize = 64;
const WIDTH: usize = H * MHC_STREAMS;
const EPS: Mhc4PreEps = Mhc4PreEps {
    hc_rms: 1e-5,
    hc: 1e-6,
    norm: 3e-5,
};
const SCALE: [f32; 3] = [0.8, -0.45, 1.1];

fn tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    offset_tensor(
        ctx,
        16,
        bytemuck::cast_slice(values),
        16,
        shape,
        GgmlType::F32,
    )
}

fn empty(ctx: &MetalContext, shape: Vec<u64>) -> MetalTensor {
    tensor(
        ctx,
        &vec![f32::NAN; shape.iter().product::<u64>() as usize],
        shape,
    )
}

struct Outputs {
    mixes: MetalTensor,
    pre: MetalTensor,
    post: MetalTensor,
    comb: MetalTensor,
    collapsed: MetalTensor,
    normed: MetalTensor,
}

impl Outputs {
    fn new(ctx: &MetalContext, rows: usize) -> Self {
        let r = rows as u64;
        Self {
            mixes: empty(ctx, vec![24, r]),
            pre: empty(ctx, vec![4, r]),
            post: empty(ctx, vec![4, r]),
            comb: empty(ctx, vec![4, 4, r]),
            collapsed: empty(ctx, vec![H as u64, r]),
            normed: empty(ctx, vec![H as u64, r]),
        }
    }

    fn bindings(&self) -> Mhc4PreOutputs<'_> {
        Mhc4PreOutputs {
            mixes: &self.mixes,
            pre: &self.pre,
            post: &self.post,
            comb: &self.comb,
            collapsed: &self.collapsed,
            normed: &self.normed,
        }
    }
}

fn weights(dtype: GgmlType) -> (Vec<u8>, Vec<f32>) {
    match dtype {
        GgmlType::Q8_0 => synthetic_q8_0_bank(WIDTH, MHC_MIXES),
        GgmlType::IQ2_XS => {
            // A native GEMV-only fallback here, even though generic dense GEMM
            // supports this format. Every index/sign bit pattern is valid.
            let (block, block_bytes) = dtype.storage_layout().unwrap();
            let mut bytes = Vec::new();
            for ordinal in 0..WIDTH * MHC_MIXES / block as usize {
                let mut b: Vec<u8> = (0..block_bytes as usize)
                    .map(|i| (ordinal * 73 + i * 29) as u8)
                    .collect();
                b[..2].copy_from_slice(&half::f16::from_f32(0.0001).to_le_bytes());
                bytes.extend_from_slice(&b);
            }
            let decoded = dequant_expert(&bytes, dtype, WIDTH, MHC_MIXES, 0);
            (bytes, decoded)
        }
        GgmlType::IQ4_NL => {
            let mut bytes = Vec::new();
            let mut decoded = Vec::new();
            for block in 0..WIDTH * MHC_MIXES / 32 {
                let (b, d) = encode_iq4_nl_block(0.001, block);
                bytes.extend_from_slice(&b);
                decoded.extend_from_slice(&d);
            }
            (bytes, decoded)
        }
        GgmlType::Q6_K => {
            let mut bytes = Vec::new();
            let mut decoded = Vec::new();
            for block in 0..WIDTH * MHC_MIXES / 256 {
                let (b, d) = encode_q6_k_block(0.001, block);
                bytes.extend_from_slice(&b);
                decoded.extend_from_slice(&d);
            }
            (bytes, decoded)
        }
        GgmlType::MXFP4 => {
            const VALUES: [f32; 16] = [
                0., 1., 2., 3., 4., 6., 8., 12., 0., -1., -2., -3., -4., -6., -8., -12.,
            ];
            let mut bytes = Vec::new();
            let mut decoded = Vec::new();
            for block in 0..WIDTH * MHC_MIXES / 32 {
                let indices = std::array::from_fn(|i| ((block * 7 + i * 3) % 16) as u8);
                bytes.extend_from_slice(&encode_mxfp4_block(122, &indices));
                decoded.extend(indices.map(|i| VALUES[i as usize] / 64.0));
            }
            (bytes, decoded)
        }
        _ => synthetic_dense_linear_bank(dtype, WIDTH, MHC_MIXES),
    }
}

fn close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "{label}[{i}]: {a} vs {e}"
        );
    }
}

#[test]
fn mhc4_pre_generic_native_weights_match_oracle_decode_and_packed() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let base: Vec<f32> = (0..24).map(|i| (i as f32 - 11.5) * 0.03).collect();
    let norm_weight: Vec<f32> = (0..H).map(|i| 0.5 + (i % 11) as f32 * 0.09).collect();
    let scale_t = tensor(&ctx, &SCALE, vec![3]);
    let base_t = tensor(&ctx, &base, vec![24]);
    let norm_t = tensor(&ctx, &norm_weight, vec![H as u64]);
    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q8_0,
        GgmlType::Q6_K,
        GgmlType::IQ4_NL,
        GgmlType::IQ2_XS,
        GgmlType::MXFP4,
    ] {
        let (bytes, decoded) = weights(dtype);
        let mix = offset_tensor(&ctx, 32, &bytes, 32, vec![WIDTH as u64, 24], dtype);
        for rows in [1, 3, 129] {
            // One zero row plus distinct, asymmetric streams. This catches
            // normalization across rows or independently across the streams.
            let residual: Vec<f32> = (0..WIDTH * rows)
                .map(|i| {
                    let row = i / WIDTH;
                    if rows > 1 && row == 0 {
                        return 0.0;
                    }
                    let stream = (i % WIDTH) / H;
                    (stream as f32 - 1.2) * (0.3 + row as f32 * 0.07)
                        + ((i * 7 + row * 3) % 29) as f32 * 0.021
                        - 0.1
                })
                .collect();
            let residual_t = tensor(&ctx, &residual, vec![H as u64, 4, rows as u64]);
            let normalized = empty(&ctx, vec![WIDTH as u64, rows as u64]);
            let ones = empty(&ctx, vec![WIDTH as u64]);
            let out = Outputs::new(&ctx, rows);
            let command = ctx.queue.commandBuffer().expect("command buffer");
            dispatch_census_begin();
            let enc = KernelEncoder::begin(&command);
            let result = encode_mhc4_pre_generic(
                &ctx,
                &enc,
                H,
                rows,
                EPS,
                &Mhc4PreInputs {
                    residual: &residual_t,
                    mix: &mix,
                    scale: &scale_t,
                    base: &base_t,
                    norm_weight: &norm_t,
                },
                &Mhc4PreGenericScratch {
                    normalized: &normalized,
                    ones: &ones,
                },
                &out.bindings(),
            );
            let census = dispatch_census_take();
            result.unwrap();
            enc.end();
            command.commit();
            wait_completed(&command).expect("command buffer failed");

            if rows > 1 && grouped_projection_supported(dtype) {
                // At M=24 even Q8 uses the generic grouped kernel. A single
                // projection covers every row, including the 129th-row tail.
                assert_eq!(census.len(), 6, "{dtype:?}/{rows}: packed launches");
                assert_eq!(
                    census[2].kernel,
                    grouped_projection_pipeline_name(dtype).unwrap()
                );
            } else {
                assert_eq!(census.len(), rows + 5, "{dtype:?}/{rows}: GEMV launches");
                for projection in &census[2..2 + rows] {
                    assert!(projection.kernel.contains("mat_vec"), "{projection:?}");
                }
            }
            // F32 operands throughout, allowing only reduction-order and
            // fast-math differences from the CPU oracle, including BF16 weights.
            let tolerance = 5e-5;
            for row in 0..rows {
                let x = &residual[row * WIDTH..(row + 1) * WIDTH];
                let cpu = hyper_connection_pre(
                    x,
                    H,
                    4,
                    &decoded,
                    &SCALE,
                    &base,
                    EPS.hc_rms,
                    MHC_SINKHORN_ITERATIONS,
                    EPS.hc,
                )
                .unwrap();
                let normed = rms_norm(&cpu.input, Some(&norm_weight), EPS.norm).unwrap();
                let normalized_cpu = rms_norm(x, None, EPS.hc_rms).unwrap();
                for (label, tensor, expected) in [
                    ("normalized", &normalized, &normalized_cpu),
                    ("mixes", &out.mixes, &cpu.mixes),
                    ("pre", &out.pre, &cpu.controls.pre),
                    ("post", &out.post, &cpu.controls.post),
                    ("comb", &out.comb, &cpu.controls.combination),
                    ("collapsed", &out.collapsed, &cpu.input),
                    ("normed", &out.normed, &normed),
                ] {
                    let actual = tensor_f32_at_offset(tensor);
                    let width = expected.len();
                    close(
                        &format!("{dtype:?}/{rows}/row {row}/{label}"),
                        &actual[row * width..(row + 1) * width],
                        expected,
                        tolerance,
                    );
                }
            }
            assert!(tensor_f32_at_offset(&ones).iter().all(|&v| v == 1.0));
        }
    }
}

#[test]
fn mhc4_pre_generic_packed_preserves_large_f32_activations_and_weights() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    // Residuals, mix weights, collapsed activations and block outputs exceed
    // half's finite range. Flattened RMS bounds projection inputs; fractional
    // normalized values still exercise F32 precision at the projection seam.
    // Keep controls away from saturation so a narrowed mix cannot be hidden
    // behind sigmoid saturation, then compare every stage to the CPU oracle.
    let scale = [1e-6, -0.7e-6, 0.5e-6];
    let base: Vec<f32> = (0..24).map(|i| (i as f32 - 11.5) * 0.03).collect();
    let norm_weight: Vec<f32> = (0..H).map(|i| 90000.25 + i as f32 * 127.125).collect();
    let scale_t = tensor(&ctx, &scale, vec![3]);
    let base_t = tensor(&ctx, &base, vec![24]);
    let norm_t = tensor(&ctx, &norm_weight, vec![H as u64]);
    for dtype in [GgmlType::F32, GgmlType::BF16] {
        let mut decoded = vec![0.0f32; WIDTH * MHC_MIXES];
        let mut bytes = Vec::new();
        for channel in 0..MHC_MIXES {
            // One nonzero per output avoids cancellation as a source of error.
            decoded[channel * WIDTH + (channel * 11) % WIDTH] = 70000.25 + channel as f32 * 123.125;
        }
        for value in &mut decoded {
            if dtype == GgmlType::BF16 {
                let stored = half::bf16::from_f32(*value);
                bytes.extend_from_slice(&stored.to_le_bytes());
                *value = stored.to_f32();
            } else {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let mix = offset_tensor(&ctx, 32, &bytes, 32, vec![WIDTH as u64, 24], dtype);
        for rows in [1, 3, 129] {
            let residual: Vec<f32> = (0..WIDTH * rows)
                .map(|i| 70000.125 + ((i * 17 + i / WIDTH * 13) % 197) as f32 * 137.375)
                .collect();
            let residual_t = tensor(&ctx, &residual, vec![H as u64, 4, rows as u64]);
            let normalized = empty(&ctx, vec![WIDTH as u64, rows as u64]);
            let ones = empty(&ctx, vec![WIDTH as u64]);
            let out = Outputs::new(&ctx, rows);
            let command = ctx.queue.commandBuffer().expect("command buffer");
            let enc = KernelEncoder::begin(&command);
            encode_mhc4_pre_generic(
                &ctx,
                &enc,
                H,
                rows,
                EPS,
                &Mhc4PreInputs {
                    residual: &residual_t,
                    mix: &mix,
                    scale: &scale_t,
                    base: &base_t,
                    norm_weight: &norm_t,
                },
                &Mhc4PreGenericScratch {
                    normalized: &normalized,
                    ones: &ones,
                },
                &out.bindings(),
            )
            .unwrap();
            enc.end();
            command.commit();
            wait_completed(&command).expect("command buffer failed");
            let actual: Vec<Vec<f32>> = [
                &normalized,
                &out.mixes,
                &out.pre,
                &out.post,
                &out.comb,
                &out.collapsed,
                &out.normed,
            ]
            .into_iter()
            .map(tensor_f32_at_offset)
            .collect();
            for row in 0..rows {
                let x = &residual[row * WIDTH..(row + 1) * WIDTH];
                let cpu = hyper_connection_pre(
                    x,
                    H,
                    4,
                    &decoded,
                    &scale,
                    &base,
                    EPS.hc_rms,
                    MHC_SINKHORN_ITERATIONS,
                    EPS.hc,
                )
                .unwrap();
                let normed = rms_norm(&cpu.input, Some(&norm_weight), EPS.norm).unwrap();
                let normalized_cpu = rms_norm(x, None, EPS.hc_rms).unwrap();
                for (values, (label, expected)) in actual.iter().zip([
                    ("normalized", &normalized_cpu),
                    ("mixes", &cpu.mixes),
                    ("pre", &cpu.controls.pre),
                    ("post", &cpu.controls.post),
                    ("comb", &cpu.controls.combination),
                    ("collapsed", &cpu.input),
                    ("normed", &normed),
                ]) {
                    let width = expected.len();
                    close(
                        &format!("full range {dtype:?}/{rows}/row {row}/{label}"),
                        &values[row * width..(row + 1) * width],
                        expected,
                        5e-5,
                    );
                }
            }
            for values in [&actual[1], &actual[5], &actual[6]] {
                assert!(values.iter().all(|v| v.is_finite()));
                assert!(values.iter().any(|&v| v > 65504.0));
            }
        }
    }
}

#[test]
fn mhc4_pre_generic_q8_matches_existing_reference_sequence() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (bytes, _) = weights(GgmlType::Q8_0);
    let mix = offset_tensor(&ctx, 32, &bytes, 32, vec![WIDTH as u64, 24], GgmlType::Q8_0);
    let residual = tensor(
        &ctx,
        &(0..WIDTH)
            .map(|i| (i % 23) as f32 * 0.07 - 0.8)
            .collect::<Vec<_>>(),
        vec![H as u64, 4, 1],
    );
    let scale = tensor(&ctx, &SCALE, vec![3]);
    let base = tensor(&ctx, &[0.1; 24], vec![24]);
    let norm = tensor(&ctx, &[0.7; H], vec![H as u64]);
    let normalized = empty(&ctx, vec![WIDTH as u64, 1]);
    let ones = empty(&ctx, vec![WIDTH as u64]);
    let out = Outputs::new(&ctx, 1);
    let reference = Outputs::new(&ctx, 1);
    let flat = |t: &MetalTensor| t.view_subrange(0, vec![t.n_elements()]);
    let command = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&command);
    encode_mhc4_pre_generic(
        &ctx,
        &enc,
        H,
        1,
        EPS,
        &Mhc4PreInputs {
            residual: &residual,
            mix: &mix,
            scale: &scale,
            base: &base,
            norm_weight: &norm,
        },
        &Mhc4PreGenericScratch {
            normalized: &normalized,
            ones: &ones,
        },
        &out.bindings(),
    )
    .unwrap();
    encode_rms_norm_mul_f32(
        &ctx,
        &enc,
        &flat(&residual),
        &ones,
        &flat(&normalized),
        EPS.hc_rms,
    )
    .unwrap();
    encode_mat_vec_q8_0_f32(
        &ctx,
        &enc,
        &mix,
        &flat(&normalized),
        &flat(&reference.mixes),
        WIDTH,
        24,
    )
    .unwrap();
    encode_mhc4_controls(
        &ctx,
        &enc,
        EPS.hc,
        &flat(&reference.mixes),
        &scale,
        &base,
        &flat(&reference.pre),
        &flat(&reference.post),
        &reference.comb.view_subrange(0, vec![4, 4]),
    )
    .unwrap();
    encode_mhc4_collapse(
        &ctx,
        &enc,
        H,
        &residual.view_subrange(0, vec![H as u64, 4]),
        &flat(&reference.pre),
        &flat(&reference.collapsed),
    )
    .unwrap();
    encode_rms_norm_mul_f32(
        &ctx,
        &enc,
        &flat(&reference.collapsed),
        &norm,
        &flat(&reference.normed),
        EPS.norm,
    )
    .unwrap();
    enc.end();
    command.commit();
    wait_completed(&command).expect("command buffer failed");
    for (a, b) in [
        (&out.mixes, &reference.mixes),
        (&out.pre, &reference.pre),
        (&out.post, &reference.post),
        (&out.comb, &reference.comb),
        (&out.collapsed, &reference.collapsed),
        (&out.normed, &reference.normed),
    ] {
        close(
            "reference sequence",
            &tensor_f32_at_offset(a),
            &tensor_f32_at_offset(b),
            2e-6,
        );
    }
}

#[test]
fn mhc4_pre_generic_rejects_aliases_and_invalid_geometry() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let residual = empty(&ctx, vec![H as u64, 4, 1]);
    let mix = empty(&ctx, vec![WIDTH as u64, 24]);
    let scale = empty(&ctx, vec![3]);
    let base = empty(&ctx, vec![24]);
    let norm = empty(&ctx, vec![H as u64]);
    let normalized = empty(&ctx, vec![WIDTH as u64, 1]);
    let ones = empty(&ctx, vec![WIDTH as u64]);
    let out = Outputs::new(&ctx, 1);
    let inputs = Mhc4PreInputs {
        residual: &residual,
        mix: &mix,
        scale: &scale,
        base: &base,
        norm_weight: &norm,
    };
    let scratch = Mhc4PreGenericScratch {
        normalized: &normalized,
        ones: &ones,
    };
    let command = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&command);
    let alias = residual.view_subrange(0, vec![WIDTH as u64, 1]);
    let err = encode_mhc4_pre_generic(
        &ctx,
        &enc,
        H,
        1,
        EPS,
        &inputs,
        &Mhc4PreGenericScratch {
            normalized: &alias,
            ones: &ones,
        },
        &out.bindings(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("aliases residual"), "{err}");
    let mut aliased_outputs = out.bindings();
    aliased_outputs.normed = &out.collapsed;
    let err = encode_mhc4_pre_generic(&ctx, &enc, H, 1, EPS, &inputs, &scratch, &aliased_outputs)
        .unwrap_err();
    assert!(err.to_string().contains("aliases normed"), "{err}");
    for (hidden, rows, eps) in [
        (0, 1, EPS),
        (H, 0, EPS),
        (H, usize::MAX, EPS),
        (
            H,
            1,
            Mhc4PreEps {
                hc_rms: f32::NAN,
                ..EPS
            },
        ),
        (H, 1, Mhc4PreEps { hc: 0.0, ..EPS }),
        (H, 1, Mhc4PreEps { norm: -1.0, ..EPS }),
    ] {
        assert!(
            encode_mhc4_pre_generic(
                &ctx,
                &enc,
                hidden,
                rows,
                eps,
                &inputs,
                &scratch,
                &out.bindings()
            )
            .is_err()
        );
    }
    enc.end();
    // These checks encode no work. Deliberately do not commit the buffer.
}
