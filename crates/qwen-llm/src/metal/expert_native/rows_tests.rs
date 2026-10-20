use super::*;
use std::collections::BTreeMap;

fn bits(t: &MetalTensor) -> Vec<u32> {
    tensor_f32_at_offset(t)
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn fixture(
    ctx: &MetalContext,
    dtype: GgmlType,
    down: bool,
    edge: bool,
    n: usize,
    k: usize,
    m: usize,
    e: usize,
    batch: Option<usize>,
) -> Vec<u32> {
    const TOP: usize = 6;
    let mut bytes = bank_bytes_for_experts(dtype, k, m, 3, e);
    if edge {
        for (i, block) in bytes.chunks_exact_mut(17).enumerate() {
            block[0] = (i % 2) as u8;
        }
    }
    let bank = offset_tensor(
        ctx,
        32,
        &bytes,
        16,
        vec![k as u64, m as u64, e as u64],
        dtype,
    );
    let up = offset_tensor(
        ctx,
        32,
        &bank_bytes_for_experts(dtype, k, m, 7, e),
        16,
        bank.shape.clone(),
        dtype,
    );
    let width = if down { k * TOP } else { k };
    let x: Vec<_> = (0..width * n)
        .map(|i| {
            let scale = if edge {
                if i / width % 2 == 0 {
                    2.0f32.powi(120)
                } else {
                    1.0
                }
            } else {
                [1.0, 131073.0, 1e-20, 1e20][i / width % 4]
            };
            (((i * 17 + 29) % 89) as f32 - 44.0) * 0.030013 * scale
        })
        .collect();
    let input = f32_view(ctx, &x, &[k, if down { TOP * n } else { n }], 28);
    let routes: Vec<_> = (0..n)
        .flat_map(|r| {
            if r % 5 == 4 {
                vec![-1, e as i32, 0, 0, i32::MIN, i32::MAX]
            } else {
                vec![e as i32 - 1, (e / 2) as i32, e as i32 - 1, 0, 1, 0]
            }
        })
        .collect();
    let ids = i32_view(ctx, &routes).view_subrange(0, vec![TOP as u64, n as u64]);
    let statuses: Vec<_> = (0..n)
        .map(|r| {
            [
                ROUTE_STATUS_READY,
                0,
                -7,
                ROUTE_STATUS_READY,
                ROUTE_STATUS_READY,
            ][r % 5]
        })
        .collect();
    let status = i32_view(ctx, &statuses);
    let output = f32_view(ctx, &vec![f32::NAN; m * TOP * n], &[m, TOP * n], 28);
    let inputs = [&bank, &up, &input, &ids, &status];
    let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
    dispatch_census_begin();
    one_shot(ctx, |enc| {
        for r in (0..n).step_by(batch.unwrap_or(1)) {
            let rows = batch.unwrap_or(1).min(n - r);
            if batch.is_some() {
                let x = input.view_subrange(
                    (r * width) as u64,
                    vec![
                        k as u64,
                        if down {
                            (rows * TOP) as u64
                        } else {
                            rows as u64
                        },
                    ],
                );
                let ids = ids.view_subrange((r * TOP) as u64, vec![TOP as u64, rows as u64]);
                let status = status.view_subrange(r as u64, vec![rows as u64]);
                let out =
                    output.view_subrange((r * m * TOP) as u64, vec![m as u64, (rows * TOP) as u64]);
                if down {
                    encode_all_slots_down_native_rows(
                        ctx, enc, &bank, &x, &ids, &status, &out, k, m, e, TOP, rows,
                    )?;
                } else {
                    encode_all_slots_gate_up_swiglu_native_rows(
                        ctx, enc, &bank, &up, &x, &ids, &status, &out, k, m, e, TOP, rows, 0.37,
                    )?;
                }
                continue;
            }
            let x = input.view_subrange(
                (r * width) as u64,
                if down {
                    vec![k as u64, TOP as u64]
                } else {
                    vec![k as u64]
                },
            );
            let ids = ids.view_subrange((r * TOP) as u64, vec![TOP as u64]);
            let status = status.view_subrange(r as u64, vec![1]);
            let out = output.view_subrange((r * m * TOP) as u64, vec![m as u64, TOP as u64]);
            if down {
                encode_all_slots_down_native(
                    ctx, enc, &bank, &x, &ids, &status, &out, k, m, e, TOP,
                )?;
            } else {
                encode_all_slots_gate_up_swiglu_native(
                    ctx, enc, &bank, &up, &x, &ids, &status, &out, k, m, e, TOP, 0.37,
                )?;
            }
        }
        Ok(())
    })
    .unwrap();
    let census = dispatch_census_take();
    let chunks: Vec<_> = (0..n)
        .step_by(batch.unwrap_or(1))
        .map(|r| batch.unwrap_or(1).min(n - r))
        .collect();
    assert_eq!(census.len(), chunks.len() * if down { 1 } else { 2 });
    for (chunk, records) in chunks
        .iter()
        .zip(census.chunks_exact(if down { 1 } else { 2 }))
    {
        for record in records {
            assert_eq!(record.kernel, native_pipeline_name(dtype).unwrap());
            assert_eq!(
                (record.grid_width, record.grid_height, record.grid_depth),
                (m.div_ceil(4) as u64, TOP as u64, *chunk as u64)
            );
            assert_eq!(
                (
                    record.threads_width,
                    record.threads_height,
                    record.threads_depth
                ),
                (128, 1, 1)
            );
        }
    }
    assert!(tensor_f32_at_offset(&output).iter().all(|v| v.is_finite()));
    let actual = bits(&output);
    for r in 0..n {
        for s in 0..TOP {
            if statuses[r] != ROUTE_STATUS_READY
                || routes[r * TOP + s] < 0
                || routes[r * TOP + s] as usize >= e
            {
                assert!(
                    actual[(r * TOP + s) * m..(r * TOP + s + 1) * m]
                        .iter()
                        .all(|&v| v == 0)
                );
            }
        }
    }
    for (t, b) in inputs.iter().zip(before) {
        assert_eq!(tensor_backing_bytes(t), b);
    }
    assert_offset_guards(&output, 20, 28);
    bits(&output)
}

fn snapshot(ctx: &MetalContext, batched: bool) -> BTreeMap<String, Vec<u32>> {
    let mut result = BTreeMap::new();
    for dtype in [GgmlType::IQ3_XXS, GgmlType::Q3_K] {
        result.insert(
            format!("gate_up_{dtype:?}"),
            fixture(
                ctx,
                dtype,
                false,
                false,
                5,
                5120,
                7,
                384,
                batched.then_some(5),
            ),
        );
    }
    for dtype in [
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_S,
        GgmlType::Q3_K,
        GgmlType::MXFP4,
    ] {
        result.insert(
            format!("down_{dtype:?}"),
            fixture(
                ctx,
                dtype,
                true,
                false,
                5,
                2304,
                7,
                384,
                batched.then_some(5),
            ),
        );
    }
    result.insert(
        "down_MXFP4_edges".into(),
        fixture(
            ctx,
            GgmlType::MXFP4,
            true,
            true,
            5,
            2304,
            7,
            384,
            batched.then_some(5),
        ),
    );
    result
}

#[test]
#[ignore = "capture before shader edits; set QWEN_NATIVE_ROWS_FIXTURE to a new local file"]
fn capture_native_row_baseline() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let path = std::env::var("QWEN_NATIVE_ROWS_FIXTURE").expect("fixture output path");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer(file, &snapshot(&ctx, false)).unwrap();
}

#[test]
#[ignore = "requires a pre-change GPU capture in QWEN_NATIVE_ROWS_FIXTURE"]
fn native_rows_match_saved_prechange_gpu_outputs() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let path = std::env::var("QWEN_NATIVE_ROWS_FIXTURE").expect("pre-change fixture path");
    let expected: BTreeMap<String, Vec<u32>> =
        serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
    assert_eq!(
        snapshot(&ctx, false),
        expected,
        "single-token ABI still matches pre-change capture"
    );
    assert_eq!(
        snapshot(&ctx, true),
        expected,
        "batched outputs match pre-change capture"
    );
}

#[test]
fn native_rows_arithmetic_source_and_capability_contract() {
    use sha2::{Digest, Sha256};
    let body = &SOURCE
        [SOURCE.find("    float sum = 0.0f;").unwrap()..SOURCE.find("\ntypedef decltype").unwrap()];
    assert_eq!(
        format!("{:x}", Sha256::digest(body.as_bytes())),
        "1bb4298433e54ba446cc5da2cc7d78d3b3431c7d54463763f9c7c55bef500b49",
        "baseline arithmetic body must remain unchanged"
    );
    assert_eq!(std::mem::size_of::<NativeArgs>(), 32);
    for raw in 0..=255 {
        let dtype = GgmlType::from_raw(raw);
        assert_eq!(
            all_slots_down_native_rows_supported(dtype),
            matches!(
                dtype,
                GgmlType::IQ3_XXS | GgmlType::IQ3_S | GgmlType::Q3_K | GgmlType::MXFP4
            )
        );
        for other in [
            GgmlType::IQ3_XXS,
            GgmlType::Q3_K,
            GgmlType::IQ2_XS,
            GgmlType::IQ2_XXS,
        ] {
            assert_eq!(
                all_slots_gate_up_native_rows_supported(dtype, other),
                matches!(dtype, GgmlType::IQ3_XXS | GgmlType::Q3_K)
                    && matches!(other, GgmlType::IQ3_XXS | GgmlType::Q3_K)
            );
        }
    }
}

#[test]
fn native_rows_bitwise_partitions_high_ids_status_views_and_extremes() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    for (dtype, down, edge) in [
        (GgmlType::IQ3_XXS, false, false),
        (GgmlType::Q3_K, false, false),
        (GgmlType::IQ3_XXS, true, false),
        (GgmlType::IQ3_S, true, false),
        (GgmlType::Q3_K, true, false),
        (GgmlType::MXFP4, true, false),
        (GgmlType::MXFP4, true, true),
    ] {
        let expected = fixture(&ctx, dtype, down, edge, 257, 512, 7, 384, None);
        for rows in [31, 32, 33, 127, 128, 129, 255, 256, 257] {
            assert_eq!(
                fixture(&ctx, dtype, down, edge, 257, 512, 7, 384, Some(rows)),
                expected,
                "{dtype:?} down={down} edge={edge} partition={rows}"
            );
        }
    }
}

#[test]
fn native_rows_bitwise_real_widths_top6() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    assert_eq!(snapshot(&ctx, true), snapshot(&ctx, false));
    for dtype in [GgmlType::IQ3_XXS, GgmlType::Q3_K] {
        assert_eq!(
            fixture(&ctx, dtype, false, false, 3, 5120, 2304, 6, Some(3)),
            fixture(&ctx, dtype, false, false, 3, 5120, 2304, 6, None)
        );
    }
    for dtype in [
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_S,
        GgmlType::Q3_K,
        GgmlType::MXFP4,
    ] {
        assert_eq!(
            fixture(&ctx, dtype, true, false, 3, 2304, 5120, 6, Some(3)),
            fixture(&ctx, dtype, true, false, 3, 2304, 5120, 6, None)
        );
    }
}

#[test]
fn native_rows_preflight_is_atomic_for_both_passes() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let (k, m, e, top, n) = (256, 7, 3, 6, 3);
    let bank = |seed| {
        offset_tensor(
            &ctx,
            32,
            &bank_bytes_for_experts(GgmlType::IQ3_XXS, k, m, seed, e),
            16,
            vec![k as u64, m as u64, e as u64],
            GgmlType::IQ3_XXS,
        )
    };
    let original = [
        bank(3),
        bank(7),
        f32_view(&ctx, &values(k * n, 9), &[k, n], 28),
        i32_view(&ctx, &vec![0; top * n]).view_subrange(0, vec![top as u64, n as u64]),
        i32_view(&ctx, &vec![1; n]),
        f32_view(&ctx, &vec![777.0; m * top * n], &[m, top * n], 28),
    ];
    let before: Vec<_> = original.iter().map(tensor_backing_bytes).collect();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let encode = |t: &[MetalTensor; 6], k, m, e, top, n, clamp| {
        encode_all_slots_gate_up_swiglu_native_rows(
            &ctx, &enc, &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], k, m, e, top, n, clamp,
        )
    };
    dispatch_census_begin();
    for which in 0..6 {
        for fault in 0..6 {
            let mut t = original.clone();
            match fault {
                0 => t[which].offset += 1,
                1 => t[which].offset = t[which].buffer.length() as u64,
                2 => t[which].offset = u64::MAX - 3,
                3 => t[which].dtype = GgmlType::F16,
                4 => t[which].shape = vec![u64::MAX, u64::MAX, 2],
                _ => t[which].shape = vec![1],
            }
            assert!(
                encode(&t, k, m, e, top, n, 0.37).is_err(),
                "binding {which} fault {fault}"
            );
        }
    }
    let mut t = original.clone();
    t[5].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(encode(&t, k, m, e, top, n, 0.37).is_err());
    for which in 0..5 {
        let mut t = original.clone();
        let shared = ctx.buffer_from(&vec![0; 65536]).unwrap();
        t[which].buffer = shared.clone();
        t[which].offset = 32;
        t[5].buffer = shared;
        t[5].offset = 32;
        assert!(
            encode(&t, k, m, e, top, n, 0.37)
                .unwrap_err()
                .to_string()
                .contains("aliases")
        );
    }
    for clamp in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
        assert!(encode(&original, k, m, e, top, n, clamp).is_err());
    }
    for (k, m, e, top, n) in [
        (k, m, e, top, 0),
        (k, m, e, top, usize::MAX),
        (k, m, e, 0, n),
        (k, m, e, 17, n),
        (k, 0, e, top, n),
        (0, m, e, top, n),
        (k, m, 0, top, n),
        (k - 1, m, e, top, n),
        (usize::MAX, m, e, top, n),
        (
            u32::MAX as usize,
            u32::MAX as usize,
            e,
            16,
            u32::MAX as usize,
        ),
    ] {
        assert!(encode(&original, k, m, e, top, n, 0.37).is_err());
    }
    for dtype in [
        GgmlType::IQ2_XS,
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_S,
        GgmlType::MXFP4,
    ] {
        let mut t = original.clone();
        t[0].dtype = dtype;
        t[1].dtype = dtype;
        assert!(encode(&t, k, m, e, top, n, 0.37).is_err());
    }
    let input = f32_view(&ctx, &values(k * top * n, 5), &[k, top * n], 28);
    let original_down = [
        original[0].clone(),
        input,
        original[3].clone(),
        original[4].clone(),
        original[5].clone(),
    ];
    let down = |t: &[MetalTensor; 5], n| {
        encode_all_slots_down_native_rows(
            &ctx, &enc, &t[0], &t[1], &t[2], &t[3], &t[4], k, m, e, top, n,
        )
    };
    for which in 0..5 {
        let mut t = original_down.clone();
        t[which].offset += 1;
        assert!(down(&t, n).is_err());
        let mut t = original_down.clone();
        t[which].shape = vec![u64::MAX, u64::MAX];
        assert!(down(&t, n).is_err());
    }
    for rows in [0, usize::MAX] {
        assert!(down(&original_down, rows).is_err());
    }
    for which in 0..4 {
        let mut t = original_down.clone();
        let shared = ctx.buffer_from(&vec![0; 65536]).unwrap();
        t[which].buffer = shared.clone();
        t[which].offset = 32;
        t[4].buffer = shared;
        t[4].offset = 32;
        assert!(down(&t, n).unwrap_err().to_string().contains("aliases"));
    }
    let mut t = original_down.clone();
    t[4].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(down(&t, n).is_err());
    let devices = objc2_metal::MTLCopyAllDevices();
    if let Some(device) = devices
        .iter()
        .find(|d| d.registryID() != ctx.device.registryID())
    {
        for which in 0..6 {
            let mut t = original.clone();
            t[which].buffer = device
                .newBufferWithLength_options(
                    t[which].buffer.length(),
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap();
            assert!(encode(&t, k, m, e, top, n, 0.37).is_err());
        }
    }
    enc.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(dispatch_census_take().is_empty());
    for (t, b) in original.iter().zip(before) {
        assert_eq!(tensor_backing_bytes(t), b);
    }
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin_concurrent(&command);
    assert!(
        encode_all_slots_gate_up_swiglu_native_rows(
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
            e,
            top,
            n,
            0.37
        )
        .is_err()
    );
    assert!(
        encode_all_slots_down_native_rows(
            &ctx,
            &enc,
            &original_down[0],
            &original_down[1],
            &original_down[2],
            &original_down[3],
            &original_down[4],
            k,
            m,
            e,
            top,
            n
        )
        .is_err()
    );
    enc.end();
}

#[test]
fn native_rows_gate_and_up_projection_stores_are_bitwise() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let (k, m, e, top, n) = (5120, 7, 384, 6, 33);
    let x: Vec<_> = values(k * n, 9).iter().map(|v| v * 131073.0).collect();
    let input = f32_view(&ctx, &x, &[k, n], 28);
    let ids = i32_view(&ctx, &[383, 271, 383, 0, 1, 0].repeat(n))
        .view_subrange(0, vec![top as u64, n as u64]);
    let status = i32_view(&ctx, &vec![ROUTE_STATUS_READY; n]);
    for dtype in [GgmlType::IQ3_XXS, GgmlType::Q3_K] {
        for seed in [3, 7] {
            let bank = offset_tensor(
                &ctx,
                32,
                &bank_bytes_for_experts(dtype, k, m, seed, e),
                16,
                vec![k as u64, m as u64, e as u64],
                dtype,
            );
            let mut args = dimensions("raw projection witness", k, m, e, top).unwrap();
            let (pso, row_bytes) =
                bank_pipeline(&ctx, "raw projection witness", &bank, &args, "bank").unwrap();
            args.row_bytes = row_bytes;
            let batched = f32_view(&ctx, &vec![f32::NAN; m * top * n], &[m, top * n], 28);
            let scalar = f32_view(&ctx, &vec![f32::NAN; m * top * n], &[m, top * n], 28);
            one_shot(&ctx, |enc| {
                dispatch(enc, &pso, &args, &bank, &input, &ids, &status, &batched, n);
                for row in 0..n {
                    dispatch(
                        enc,
                        &pso,
                        &args,
                        &bank,
                        &input.view_subrange((row * k) as u64, vec![k as u64]),
                        &ids.view_subrange((row * top) as u64, vec![top as u64]),
                        &status.view_subrange(row as u64, vec![1]),
                        &scalar.view_subrange((row * m * top) as u64, vec![m as u64, top as u64]),
                        1,
                    );
                }
                Ok(())
            })
            .unwrap();
            assert_eq!(
                bits(&batched),
                bits(&scalar),
                "{dtype:?} seed={seed} raw projection"
            );
            assert_offset_guards(&batched, 20, 28);
            assert_offset_guards(&scalar, 20, 28);
        }
    }
}
