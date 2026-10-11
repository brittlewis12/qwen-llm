use super::*;
use crate::deepseek_v4_oracle::{hyper_connection_pre, rms_norm};
use crate::metal::test_support::{
    assert_offset_guards, dequant_expert, offset_tensor, synthetic_dense_linear_bank,
    tensor_backing_bytes, tensor_f32_at_offset,
};

const SOURCE: &str = include_str!("../../../../../kernels/mhc_pre_native.metal");
const Q8_SOURCE: &str = include_str!("../../../../../kernels/deepseek_v4.metal");

#[test]
fn native_pre_decoder_selection_and_preserved_finish_expressions() {
    assert_eq!(GROUPED_PROJECTION_DTYPES.len(), 17);
    assert_eq!(SOURCE.matches("template [[host_name(").count(), 17);
    for &dtype in GROUPED_PROJECTION_DTYPES {
        let name = mhc4_pre_native_pipeline_name(dtype, 16384, 1).unwrap();
        assert!(SOURCE.contains(&format!("host_name(\"{name}\")")));
        assert!(mhc4_pre_native_pipeline_name(dtype, 16384, 3).is_none());
    }
    for dtype in [
        GgmlType::IQ2_XS,
        GgmlType::IQ1_S,
        GgmlType::MXFP4,
        GgmlType::I32,
    ] {
        assert!(mhc4_pre_native_pipeline_name(dtype, 16384, 1).is_none());
    }
    for width in [0, 16, 36] {
        assert!(mhc4_pre_native_pipeline_name(GgmlType::F32, width, 1).is_none());
    }
    for width in [32, 256, 288, 16384] {
        assert!(mhc4_pre_native_pipeline_name(GgmlType::F32, width, 1).is_some());
        assert!(24 * width.div_ceil(256) <= width);
    }
    assert!(SOURCE.contains("#include \"native_dequant_f32.h\""));
    assert!(!SOURCE.contains("hc_rms_eps"));
    assert!(!SOURCE.contains("partial_sumsq"));
    // Keep duplicated controls and collapse/block RMS expressions in sync
    // without changing the established Q8 function or its specialization.
    let controls = |s: &str, name: &str| {
        let start = s.find(name).unwrap();
        let end = s[start..].find("\n}\n").unwrap() + start;
        s[start..end].replace(name, "controls")
    };
    assert_eq!(
        controls(SOURCE, "mhc4_native_controls_registers("),
        controls(Q8_SOURCE, "mhc4_controls_registers(")
    );
    let tail = |s: &str| {
        let start = s
            .find("    device const float * r = residual + row * width;")
            .unwrap();
        let end = s[start..].find("\n}\n").unwrap() + start;
        s[start..end].to_owned()
    };
    assert_eq!(
        tail(SOURCE),
        tail(&Q8_SOURCE[Q8_SOURCE.find("kernel_mhc4_pre_finish(").unwrap()..])
    );
}

#[test]
fn native_pre_baseline_override_restores_nested_unwind_and_thread() {
    let selected = || mhc4_pre_native_pipeline_name(GgmlType::F32, 256, 1).is_some();
    assert!(selected());
    with_mhc4_pre_generic_baseline_override(true, || {
        assert!(!selected());
        with_mhc4_pre_generic_baseline_override(false, || assert!(selected()));
        assert!(!selected());
        assert!(
            std::panic::catch_unwind(|| {
                with_mhc4_pre_generic_baseline_override(false, || panic!("nested unwind"));
            })
            .is_err()
        );
        assert!(!selected());
        assert!(
            std::thread::spawn(|| mhc4_pre_native_pipeline_name(GgmlType::F32, 256, 1).is_some())
                .join()
                .unwrap()
        );
    });
    assert!(selected());
    assert!(
        std::panic::catch_unwind(|| {
            with_mhc4_pre_generic_baseline_override(true, || panic!("outer unwind"));
        })
        .is_err()
    );
    assert!(selected());
}

fn bank(dtype: GgmlType, width: usize) -> (Vec<u8>, Vec<f32>) {
    if matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        return synthetic_dense_linear_bank(dtype, width, 24);
    }
    let (block, bytes) = dtype.storage_layout().unwrap();
    let mut result = Vec::new();
    for ordinal in 0..width * 24 / block as usize {
        let mut b: Vec<u8> = (0..bytes as usize)
            .map(|i| (ordinal * 73 + i * 29 + ordinal / 7 * 11) as u8)
            .collect();
        let offsets: &[usize] = match dtype {
            GgmlType::Q2_K => &[80, 82],
            GgmlType::Q3_K => &[108],
            GgmlType::Q6_K => &[208],
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q4_1 => &[0, 2],
            _ => &[0],
        };
        for (j, &offset) in offsets.iter().enumerate() {
            let scale = (1 + (ordinal + j) % 7) as f32 * 0.00013;
            b[offset..offset + 2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        }
        result.extend_from_slice(&b);
    }
    let decoded = dequant_expert(&result, dtype, width, 24, 0);
    (result, decoded)
}

fn tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    offset_tensor(
        ctx,
        32,
        bytemuck::cast_slice(values),
        28,
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

fn outputs(ctx: &MetalContext, hidden: usize) -> [MetalTensor; 6] {
    [
        vec![24, 1],
        vec![4, 1],
        vec![4, 1],
        vec![4, 4, 1],
        vec![hidden as u64, 1],
        vec![hidden as u64, 1],
    ]
    .map(|shape| empty(ctx, shape))
}

fn bindings(out: &[MetalTensor; 6]) -> Mhc4PreOutputs<'_> {
    Mhc4PreOutputs {
        mixes: &out[0],
        pre: &out[1],
        post: &out[2],
        comb: &out[3],
        collapsed: &out[4],
        normed: &out[5],
    }
}

fn close(label: &str, got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance * b.abs().max(1.0),
            "{label}[{i}]: {a} vs {b}"
        );
    }
}

// range: 0 = ordinary dense/quant; 1 = raw dots overflow, normalized dots
// finite; 2 = tiny activations times huge weights (RMS epsilon dominates).
fn exercise(ctx: &MetalContext, hidden: usize, dtype: GgmlType, range: u32) {
    let width = hidden * 4;
    let (mut bytes, mut decoded) = bank(dtype, width);
    let residual: Vec<f32> = (0..width)
        .map(|i| {
            let v = 0.7 + (i % 17) as f32 * 0.017 + (i / hidden) as f32 * 0.11;
            v * match range {
                1 => 1e8,
                2 => 1e-20,
                _ => 1.0,
            }
        })
        .collect();
    let scale = if range == 1 {
        [1e-32, -0.7e-32, 0.5e-32]
    } else if range == 2 {
        [1e-14, -0.7e-14, 0.5e-14]
    } else {
        [0.8, -0.45, 1.1]
    };
    if range != 0 {
        assert!(matches!(dtype, GgmlType::F32 | GgmlType::BF16));
        decoded.fill(0.0);
        for m in 0..24 {
            decoded[m * width + (m * 11) % width] = 1e32 * (1.0 + m as f32 * 0.01);
        }
        bytes.clear();
        for value in &mut decoded {
            if dtype == GgmlType::BF16 {
                let stored = half::bf16::from_f32(*value);
                bytes.extend_from_slice(&stored.to_le_bytes());
                *value = stored.to_f32();
            } else {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        if range == 1 {
            let raw: f64 = decoded[..width]
                .iter()
                .zip(&residual)
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum();
            assert!(
                raw > f64::from(f32::MAX),
                "fixture must overflow raw projection"
            );
        }
    }
    let base: Vec<f32> = (0..24).map(|i| (i as f32 - 11.5) * 0.03).collect();
    let norm_weight: Vec<f32> = (0..hidden)
        .map(|i| (0.5 + (i % 11) as f32 * 0.09) * if range == 0 { 1.0 } else { 1e20 })
        .collect();
    let eps = Mhc4PreEps {
        hc_rms: 1e-5,
        hc: 1e-6,
        norm: 3e-5,
    };
    let mix = offset_tensor(ctx, 32, &bytes, 28, vec![width as u64, 24], dtype);
    let residual_t = tensor(ctx, &residual, vec![hidden as u64, 4, 1]);
    let scale_t = tensor(ctx, &scale, vec![3]);
    let base_t = tensor(ctx, &base, vec![24]);
    let norm_t = tensor(ctx, &norm_weight, vec![hidden as u64]);
    let inputs = Mhc4PreInputs {
        residual: &residual_t,
        mix: &mix,
        scale: &scale_t,
        base: &base_t,
        norm_weight: &norm_t,
    };
    let normalized = empty(ctx, vec![width as u64, 1]);
    let ones = empty(ctx, vec![width as u64]);
    let scratch = Mhc4PreGenericScratch {
        normalized: &normalized,
        ones: &ones,
    };
    let input_tensors = [&mix, &residual_t, &scale_t, &base_t, &norm_t];
    let before: Vec<_> = input_tensors
        .iter()
        .map(|t| tensor_backing_bytes(t))
        .collect();
    let candidate = outputs(ctx, hidden);
    let baseline = outputs(ctx, hidden);
    let has_baseline = crate::metal_forward::mat_vec_dispatch_supported(dtype);
    let selected = mhc4_pre_native_pipeline_name(dtype, width, 1);
    let command = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&command);
    allocation_census_begin();
    if has_baseline {
        dispatch_census_begin();
        let result = with_mhc4_pre_generic_baseline_override(true, || {
            encode_mhc4_pre_generic(
                ctx,
                &enc,
                hidden,
                1,
                eps,
                &inputs,
                &scratch,
                &bindings(&baseline),
            )
        });
        let census = dispatch_census_take();
        result.unwrap();
        assert_eq!(census.len(), 6, "scoped baseline");
    }
    dispatch_census_begin();
    let result = encode_mhc4_pre_generic(
        ctx,
        &enc,
        hidden,
        1,
        eps,
        &inputs,
        &scratch,
        &bindings(&candidate),
    );
    let census = dispatch_census_take();
    let allocations = allocation_census_take();
    result.unwrap();
    assert!(
        allocations.is_empty(),
        "pre must reuse caller scratch: {allocations:?}"
    );
    if let Some(name) = &selected {
        assert_eq!(census.len(), 4, "candidate must use four dispatches");
        assert_eq!(census[0].kernel, "kernel_fill_f32");
        assert_eq!(census[1].kernel, "kernel_rms_norm_mul_rows_f32");
        assert_eq!(&census[2].kernel, name);
        assert_eq!(census[3].kernel, "kernel_mhc4_pre_native_finish");
    } else {
        assert_eq!(census.len(), 6, "geometry fallback");
    }
    enc.end();
    command.commit();
    wait_completed(&command).expect("command buffer failed");
    let cpu = hyper_connection_pre(
        &residual,
        hidden,
        4,
        &decoded,
        &scale,
        &base,
        eps.hc_rms,
        MHC_SINKHORN_ITERATIONS,
        eps.hc,
    )
    .unwrap();
    let cpu_norm = rms_norm(&cpu.input, Some(&norm_weight), eps.norm).unwrap();
    for ((out, old), expected) in candidate.iter().zip(&baseline).zip([
        &cpu.mixes,
        &cpu.controls.pre,
        &cpu.controls.post,
        &cpu.controls.combination,
        &cpu.input,
        &cpu_norm,
    ]) {
        let got = tensor_f32_at_offset(out);
        close(
            &format!("CPU {dtype:?} H={hidden} range={range}"),
            &got,
            expected,
            3e-4,
        );
        if has_baseline {
            close("old baseline", &got, &tensor_f32_at_offset(old), 3e-4);
        }
        assert_offset_guards(out, 32, 28);
        assert_offset_guards(old, 32, 28);
    }
    // Independent f64 RMS and dot oracle, without F32 raw-dot overflow or
    // sequential accumulation. This checks projection order, not just gates.
    let mean = residual.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / width as f64;
    let inverse = (mean + f64::from(eps.hc_rms)).sqrt().recip();
    let expected: Vec<f32> = decoded
        .chunks_exact(width)
        .map(|w| {
            w.iter()
                .zip(&residual)
                .map(|(&w, &x)| f64::from(w) * (f64::from(x) * inverse))
                .sum::<f64>() as f32
        })
        .collect();
    close(
        "f64 mixes",
        &tensor_f32_at_offset(&candidate[0]),
        &expected,
        5e-5,
    );
    close(
        "normalized",
        &tensor_f32_at_offset(&normalized),
        &rms_norm(&residual, None, eps.hc_rms).unwrap(),
        5e-6,
    );
    let used = if selected.is_some() {
        24 * width.div_ceil(256)
    } else {
        0
    };
    assert!(
        tensor_f32_at_offset(&ones)[used..]
            .iter()
            .all(|&v| v == 1.0)
    );
    for t in [&normalized, &ones] {
        assert_offset_guards(t, 32, 28);
    }
    for (t, old) in input_tensors.iter().zip(before) {
        assert_eq!(tensor_backing_bytes(t), old, "input must remain unchanged");
    }
}

#[test]
fn native_pre_all_decoders_tiny_and_full_glm_oracles() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for &dtype in GROUPED_PROJECTION_DTYPES {
        exercise(&ctx, 64, dtype, 0);
    }
    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q6_K,
        GgmlType::Q8_0,
    ] {
        exercise(&ctx, 4096, dtype, 0);
    }
    // Minimum candidate, final partial chunk, and baseline-only geometries.
    for h in [8, 72, 4, 9] {
        exercise(&ctx, h, GgmlType::F32, 0);
    }
}

#[test]
fn native_pre_normalization_first_preserves_pathological_finite_range() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for dtype in [GgmlType::F32, GgmlType::BF16] {
        for h in [64, 4096] {
            for range in [1, 2] {
                exercise(&ctx, h, dtype, range);
            }
        }
    }
}
