use super::*;
use crate::metal::test_support::{
    assert_offset_guards, dequant_expert, metal_test_context, offset_tensor,
    synthetic_dense_linear_bank, tensor_backing_bytes, tensor_f32_at_offset,
};

const SOURCE: &str = include_str!("../../../../../kernels/latent_grouped.metal");
const TILES: &str = include_str!("../../../../../kernels/quant_tiles.h");

fn function_source<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("inline void {name}(")).unwrap();
    let open = start + source[start..].find('{').unwrap();
    let mut depth = 0;
    for (offset, ch) in source[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated decoder {name}");
}

/// Guard against adding a format to dispatch without its native kernel, or
/// changing quant_tiles.h's verified decoder logic without updating Exact.
#[test]
fn grouped_native_decoder_and_storage_mapping_matches_shared_tiles() {
    let normalize = |s: &str| s.split_whitespace().collect::<String>();
    assert_eq!(
        SOURCE.matches("template [[host_name(").count(),
        GROUPED_PROJECTION_DTYPES.len()
    );
    for &dtype in GROUPED_PROJECTION_DTYPES {
        let pipeline = grouped_projection_pipeline_name(dtype).unwrap();
        assert!(SOURCE.contains(&format!("host_name(\"{pipeline}\")")));
        let suffix = pipeline
            .strip_prefix("kernel_latent_grouped_")
            .unwrap()
            .strip_suffix("_f32")
            .unwrap();
        let canonical = function_source(TILES, &format!("qt_dequantize_{suffix}"))
            .replace("qt_dequantize_", "latent_dequantize_")
            .replace("half4x4", "float4x4")
            .replace("(half)", "(float)")
            .replace("half(", "float(");
        let exact = function_source(SOURCE, &format!("latent_dequantize_{suffix}"));
        assert_eq!(
            normalize(&canonical),
            normalize(exact),
            "{dtype:?}: F32 decoder drift"
        );
        let constant = |field: &str| -> u64 {
            let name = format!("QT_{}_{}", suffix.to_uppercase(), field);
            TILES
                .lines()
                .find_map(|line| {
                    let parts: Vec<_> = line.split_whitespace().collect();
                    (parts.len() == 3 && parts[0] == "#define" && parts[1] == name)
                        .then(|| parts[2].parse().unwrap())
                })
                .unwrap()
        };
        let (block, bytes) = dtype.storage_layout().unwrap();
        assert_eq!(constant("BYTES") * block, bytes * constant("NL") * 16);
    }
    assert!(!grouped_projection_supported(GgmlType::I32));
    assert!(grouped_projection_pipeline_name(GgmlType::Unknown).is_none());
}

/// Deterministic, nonuniform native bytes, with every floating storage scale
/// made finite. The independent linked GGML codec supplies the CPU reference.
fn bank(dtype: GgmlType, k: usize, m: usize, groups: usize) -> (Vec<u8>, Vec<f32>) {
    if matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        return synthetic_dense_linear_bank(dtype, k, m * groups);
    }
    let (block, block_bytes) = dtype.storage_layout().unwrap();
    let count = k * m * groups / block as usize;
    let mut bytes = Vec::with_capacity(count * block_bytes as usize);
    for ordinal in 0..count {
        let mut b: Vec<u8> = (0..block_bytes as usize)
            .map(|i| (ordinal * 73 + i * 29 + (ordinal / 7) * 11) as u8)
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
        bytes.extend_from_slice(&b);
    }
    let decoded = (0..groups)
        .flat_map(|g| dequant_expert(&bytes, dtype, k, m, g))
        .collect();
    (bytes, decoded)
}

fn f32_view(
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

fn expected_pipeline(dtype: GgmlType, rows: usize) -> String {
    match (dtype, rows) {
        (GgmlType::Q8_0, 1) => "kernel_mat_vec_q8_0_f32_lcpp_grouped".into(),
        (GgmlType::Q8_0, 128) => "kernel_mat_mat_q8_0_f32_r2c16k64_grouped".into(),
        (GgmlType::Q8_0, 8) => "kernel_mat_mat_q8_0_f32_r2c16k64_grouped_tail".into(),
        _ => grouped_projection_pipeline_name(dtype).unwrap(),
    }
}

fn exercise(
    ctx: &MetalContext,
    dtype: GgmlType,
    k: usize,
    m: usize,
    groups: usize,
    rows: usize,
    suffix: usize,
) {
    let (bytes, decoded) = bank(dtype, k, m, groups);
    let prefix = if dtype == GgmlType::F32 { 20 } else { 18 };
    let weight = offset_tensor(
        ctx,
        prefix,
        &bytes,
        suffix,
        vec![k as u64, m as u64, groups as u64],
        dtype,
    );
    let x: Vec<f32> = (0..k * groups * rows)
        .map(|i| {
            let group = (i / k) % groups;
            if groups > 1 && group == groups / 2 {
                0.0
            } else {
                (((i * 17 + (i / k) * 53) % 89) as f32 - 44.0) * 0.030013
            }
        })
        .collect();
    let input = f32_view(ctx, &x, k * groups, rows, suffix);
    let output = f32_view(
        ctx,
        &vec![-777.0; m * groups * rows],
        m * groups,
        rows,
        suffix,
    );
    let weight_before = tensor_backing_bytes(&weight);
    let input_before = tensor_backing_bytes(&input);
    dispatch_census_begin();
    one_shot(ctx, |enc| {
        encode_mat_mat_grouped_f32(ctx, enc, &weight, &input, &output, k, m, groups, rows)
    })
    .unwrap();
    let census = dispatch_census_take();
    assert_eq!(census.len(), 1);
    assert_eq!(census[0].kernel, expected_pipeline(dtype, rows));
    let actual = tensor_f32_at_offset(&output);
    assert!(actual.iter().all(|v| v.is_finite() && *v != -777.0));
    for token in [0, rows / 2, rows - 1] {
        for group in 0..groups {
            for channel in [0, 1, 7, 15, 16, m / 2, m - 1] {
                let w = &decoded[(group * m + channel) * k..][..k];
                let x = &x[(token * groups + group) * k..][..k];
                let expected: f64 = w
                    .iter()
                    .zip(x)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let got = f64::from(actual[(token * groups + group) * m + channel]);
                assert!(
                    (got - expected).abs() <= 2e-5 * expected.abs().max(1.0),
                    "{dtype:?} K={k} M={m} groups={groups} rows={rows} token={token} group={group} channel={channel}: {got} vs {expected}"
                );
            }
        }
    }
    if groups > 1 {
        for token in 0..rows {
            assert!(
                actual[(token * groups + groups / 2) * m..][..m]
                    .iter()
                    .all(|&v| v == 0.0),
                "cross-head leakage"
            );
        }
    }
    // Last packed row, including the 129th token, agrees with decode. Exercise
    // the decode wrapper's [K,groups]/[M,groups] views and a flat weight view.
    if rows > 1 {
        let mut x_last = input.view_subrange(
            ((rows - 1) * k * groups) as u64,
            vec![k as u64, groups as u64],
        );
        let mut y_last = f32_view(ctx, &vec![-777.0; m * groups], m * groups, 1, 28);
        y_last.shape = vec![m as u64, groups as u64];
        let mut flat_weight = weight.clone();
        flat_weight.shape = vec![k as u64, (m * groups) as u64];
        // Also cover the flat activation form for the small packed case.
        if rows == 3 {
            x_last.shape = vec![(k * groups) as u64];
        }
        one_shot(ctx, |enc| {
            encode_mat_vec_grouped_f32(ctx, enc, &flat_weight, &x_last, &y_last, k, m, groups)
        })
        .unwrap();
        for (&a, &b) in tensor_f32_at_offset(&y_last)
            .iter()
            .zip(&actual[(rows - 1) * m * groups..])
        {
            assert!(
                (a - b).abs() <= 2e-5 * b.abs().max(1.0),
                "{dtype:?}: packed/decode mismatch"
            );
        }
        assert_offset_guards(&y_last, 20, 28);
    }
    assert_offset_guards(&output, 20, suffix);
    assert_eq!(tensor_backing_bytes(&weight), weight_before);
    assert_eq!(tensor_backing_bytes(&input), input_before);
}

#[test]
fn grouped_native_required_shapes_offsets_and_head_isolation() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in [GgmlType::Q6_K, GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
        for (k, groups) in [(256, 1), (512, 3), (256, 64), (512, 64)] {
            for rows in [1, 3, 128, 129] {
                // Both a physical buffer end and suffix canaries, across shapes.
                exercise(
                    &ctx,
                    dtype,
                    k,
                    256,
                    groups,
                    rows,
                    if rows % 2 == 0 { 28 } else { 0 },
                );
            }
        }
    }
}

#[test]
fn grouped_native_all_shared_formats_and_q8_dispatch_preservation() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for &dtype in GROUPED_PROJECTION_DTYPES {
        exercise(&ctx, dtype, 256, 256, 3, 3, 28);
    }
    for rows in [1, 8, 128, 129] {
        exercise(&ctx, GgmlType::Q8_0, 512, 256, 3, rows, 0);
    }
}

#[test]
fn grouped_native_exact_preserves_f32_operands_and_dense_range_with_channel_tails() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (k, m, groups, rows) = (32, 5, 3, 129);
    for dtype in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
        let mut bytes = Vec::new();
        let mut w = Vec::new();
        for i in 0..k * m * groups {
            let value = if i % k == 0 {
                if dtype == GgmlType::F16 {
                    1.125
                } else {
                    70000.25 + (i / k) as f32
                }
            } else {
                0.0
            };
            match dtype {
                GgmlType::F32 => {
                    bytes.extend_from_slice(&value.to_le_bytes());
                    w.push(value);
                }
                GgmlType::F16 => {
                    let v = half::f16::from_f32(value);
                    bytes.extend_from_slice(&v.to_le_bytes());
                    w.push(v.to_f32());
                }
                GgmlType::BF16 => {
                    let v = half::bf16::from_f32(value);
                    bytes.extend_from_slice(&v.to_le_bytes());
                    w.push(v.to_f32());
                }
                _ => unreachable!(),
            }
        }
        let x: Vec<f32> = (0..k * groups * rows)
            .map(|i| {
                if i % k == 0 {
                    70000.125 + (i / k) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let weight = offset_tensor(
            &ctx,
            20,
            &bytes,
            0,
            vec![k as u64, m as u64, groups as u64],
            dtype,
        );
        let input = f32_view(&ctx, &x, k * groups, rows, 0);
        let output = f32_view(&ctx, &vec![-777.0; m * groups * rows], m * groups, rows, 28);
        one_shot(&ctx, |enc| {
            encode_mat_mat_grouped_f32(&ctx, enc, &weight, &input, &output, k, m, groups, rows)
        })
        .unwrap();
        for (i, got) in tensor_f32_at_offset(&output).into_iter().enumerate() {
            let group = (i / m) % groups;
            let channel = i % m;
            let token = i / (m * groups);
            assert_eq!(
                got,
                w[(group * m + channel) * k] * x[(token * groups + group) * k],
                "{dtype:?} index {i}"
            );
        }
        assert_offset_guards(&output, 20, 28);
    }
}

#[test]
fn grouped_native_rejects_invalid_bindings_without_dispatch() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (k, m, groups, rows) = (256, 256, 3, 3);
    let (bytes, _) = bank(GgmlType::Q6_K, k, m, groups);
    let weight = offset_tensor(
        &ctx,
        18,
        &bytes,
        0,
        vec![k as u64, m as u64, groups as u64],
        GgmlType::Q6_K,
    );
    let input = f32_view(&ctx, &vec![1.0; k * groups * rows], k * groups, rows, 0);
    let output = f32_view(&ctx, &vec![0.0; m * groups * rows], m * groups, rows, 0);
    let original = [weight, input, output];
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let encode = |t: &[MetalTensor; 3], k, m, groups, rows| {
        encode_mat_mat_grouped_f32(&ctx, &enc, &t[0], &t[1], &t[2], k, m, groups, rows)
    };
    dispatch_census_begin();
    for which in 0..3 {
        for fault in 0..6 {
            let mut t = original.clone();
            match fault {
                0 => t[which].offset += 1,
                1 => t[which].offset = t[which].buffer.length() as u64,
                2 => t[which].offset = u64::MAX - 3,
                3 => t[which].dtype = GgmlType::I32,
                4 => t[which].shape = vec![1],
                _ => t[which].shape = vec![u64::MAX, u64::MAX, 2],
            }
            assert!(encode(&t, k, m, groups, rows).is_err());
        }
    }
    let mut t = original.clone();
    t[2].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(encode(&t, k, m, groups, rows).is_err());
    // F32 bank is large enough to test output/weight alias independently of
    // range validation, as well as output/input alias.
    let (dense, _) = bank(GgmlType::F32, k, m, groups);
    let dense = offset_tensor(
        &ctx,
        20,
        &dense,
        0,
        original[0].shape.clone(),
        GgmlType::F32,
    );
    for aliased in [&dense, &original[1]] {
        let mut t = original.clone();
        t[0] = dense.clone();
        t[2].buffer = aliased.buffer.clone();
        t[2].offset = aliased.offset;
        assert!(encode(&t, k, m, groups, rows).is_err());
    }
    for (k, m, g, r) in [
        (0, m, groups, rows),
        (32, m, groups, rows),
        (k, 0, groups, rows),
        (k, m, 0, rows),
        (k, m, groups, 0),
        (k, m, usize::MAX, rows),
        (k, m, groups, usize::MAX),
        (u32::MAX as usize + 1, m, groups, rows),
    ] {
        assert!(encode(&original, k, m, g, r).is_err());
    }
    let mut transposed = original.clone();
    transposed[1].shape = vec![rows as u64, (k * groups) as u64];
    assert!(encode(&transposed, k, m, groups, rows).is_err());
    enc.end();
    assert!(dispatch_census_take().is_empty());
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin_concurrent(&command);
    assert!(
        encode_mat_mat_grouped_f32(
            &ctx,
            &enc,
            &original[0],
            &original[1],
            &original[2],
            k,
            m,
            groups,
            rows
        )
        .is_err()
    );
    enc.end();
}
