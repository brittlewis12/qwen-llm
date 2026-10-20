use super::*;

fn bank(ctx: &MetalContext, dtype: GgmlType, h: usize, e: usize, seed: usize) -> MetalTensor {
    let (block, size) = dtype.storage_layout().unwrap();
    let mut bytes = Vec::new();
    for ordinal in 0..h * h * e / block as usize {
        let mut b: Vec<_> = (0..size as usize)
            .map(|i| (ordinal * 73 + i * 29 + seed * 11 + ordinal / 7) as u8)
            .collect();
        if dtype == GgmlType::MXFP4 {
            b[0] = 120 + ((ordinal + seed) % 5) as u8;
        } else {
            let offset = if dtype == GgmlType::Q3_K { 108 } else { 0 };
            b[offset..offset + 2].copy_from_slice(
                &half::f16::from_f32((1 + (ordinal + seed) % 7) as f32 / 65536.0).to_le_bytes(),
            );
        }
        bytes.extend(b);
    }
    MetalTensor::from_bytes(ctx, &bytes, vec![h as u64, h as u64, e as u64], dtype).unwrap()
}

fn bits(t: &MetalTensor) -> Vec<u32> {
    f32_slice(t).unwrap().iter().map(|v| v.to_bits()).collect()
}
fn ints(t: &MetalTensor) -> Vec<i32> {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<i32>(),
            t.n_elements() as usize,
        )
        .to_vec()
    }
}

#[test]
fn native_rows_v41_live_route_stages_dispatch_and_partition_are_bitwise() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    const H: usize = 256;
    const E: usize = 384;
    const N: usize = 257;
    for (gate_type, down_type) in [
        (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS),
        (GgmlType::IQ3_XXS, GgmlType::IQ3_S),
        (GgmlType::IQ3_XXS, GgmlType::MXFP4),
        (GgmlType::Q3_K, GgmlType::Q3_K),
        (GgmlType::Q3_K, GgmlType::MXFP4),
    ] {
        let banks = [
            bank(&ctx, gate_type, H, E, 3),
            bank(&ctx, gate_type, H, E, 7),
            bank(&ctx, down_type, H, E, 11),
        ];
        let weights = toy::Weights::new(&ctx);
        let mut model = weights.model(&ctx);
        model.layers.truncate(2);
        model.config.hidden = H;
        model.config.intermediate = H;
        model.config.experts = E;
        let layer = &mut model.layers[0];
        layer.banks = [&banks[0], &banks[1], &banks[2]];
        layer.bias = (0..E).map(|i| if i >= E - 6 { 3.0 } else { 0.0 }).collect();
        layer.ffn_hc.norm = vec![1.0; H];
        let reset = |linear: &mut Linear<'_>, input, output, seed| {
            linear.input = input;
            linear.output = output;
            let x: Vec<_> = (0..input * output)
                .map(|i| ((i * 17 + seed * 13) % 31) as f32 / 8192.0 - 0.001)
                .collect();
            linear.tensor = upload(&ctx, &x, vec![input as u64, output as u64]).unwrap();
        };
        reset(&mut layer.ffn_hc.projection, 4 * H, 24, 1);
        reset(&mut layer.router, H, E, 3);
        for (i, linear) in layer.shared.iter_mut().enumerate() {
            reset(linear, H, H, i + 5);
        }
        model.layers[1].ffn_hc.norm = vec![1.0; H];
        model.layers[1].bias = model.layers[0].bias.clone();
        let c = &model.config;
        let layer = &model.layers[0];
        assert!(!grouped_supported(c, layer));
        assert!(native_rows_supported(c, layer));
        let ws = Workspace::new(&model).unwrap();
        let x: Vec<_> = (0..4 * H * N)
            .map(|i| ((i * 19 + i / (4 * H) * 31) % 101) as f32 / 17.0 - 3.0)
            .collect();
        let input = upload(&ctx, &x, vec![H as u64, 4, N as u64]).unwrap();
        let incoming = upload(&ctx, &[0.7, 0.1, 0.3, 0.9].repeat(N), vec![4, N as u64]).unwrap();
        let output = MetalTensor::zeros_f32(&ctx, vec![H as u64, 4, N as u64]).unwrap();
        let scratch = Rows::new(&ctx, c, N, 2).unwrap();
        let b = &scratch.buffers;
        // Layer-major status offset must not overwrite the other layer's row span.
        unsafe {
            std::slice::from_raw_parts_mut(b[6].buffer.contents().as_ptr().cast::<i32>(), N)
                .fill(777);
        }
        metal::dispatch_census_begin();
        submit(&ctx, "native_rows_full_ffn", [0; 4], |enc| {
            ws.encode_rows(&ctx, enc, c, layer, 1, &input, &incoming, &output, &scratch)
        })
        .unwrap();
        let census = metal::dispatch_census_take();
        assert!(ints(&b[6])[..N].iter().all(|&s| s == 777));
        assert!(
            ints(&b[6])[N..]
                .iter()
                .all(|&s| s == metal::ROUTE_STATUS_READY)
        );
        let at = census
            .iter()
            .position(|r| r.kernel == "kernel_deepseek_v41_route")
            .unwrap();
        let routed = &census[at..at + 6];
        assert_eq!(
            (routed[0].grid_width, routed[0].threads_width),
            (N as u64, 1)
        );
        assert_eq!(routed[3].kernel, "kernel_deepseek_v41_weight");
        assert_eq!(routed[5].kernel, "kernel_deepseek_v41_sum");
        for i in [1, 2, 4] {
            assert!(routed[i].kernel.starts_with("kernel_expert_native_"));
            assert_eq!(
                (
                    routed[i].grid_width,
                    routed[i].grid_height,
                    routed[i].grid_depth
                ),
                (H.div_ceil(4) as u64, c.topk as u64, N as u64)
            );
            assert_eq!(
                (
                    routed[i].threads_width,
                    routed[i].threads_height,
                    routed[i].threads_depth
                ),
                (128, 1, 1)
            );
        }
        assert_eq!(
            (routed[3].grid_width, routed[3].threads_width),
            ((H * c.topk * N).div_ceil(256) as u64, 256)
        );
        assert_eq!(
            (routed[5].grid_width, routed[5].threads_width),
            ((H * N).div_ceil(256) as u64, 256)
        );
        assert!(
            !census
                .iter()
                .any(|r| r.kernel.contains("moe_") || r.kernel.contains("bucket"))
        );
        assert!(ints(&b[10]).iter().chain(&ints(&b[11])).all(|&v| v == 0));
        let expected_sum = MetalTensor::zeros_f32(&ctx, vec![H as u64, N as u64]).unwrap();
        let mut expected = [Vec::new(), Vec::new(), Vec::new()];
        for row in 0..N {
            submit(&ctx, "native_rows_scalar_reference", [0; 4], |enc| {
                ws.encode_routed(
                    &ctx,
                    enc,
                    c,
                    layer,
                    &ws.constants[1][3],
                    &prefill::row(&scratch.hc.normalized, H, row),
                    &prefill::row(&b[0], E, row),
                    &prefill::row(&b[4], H, row),
                    &prefill::row(&expected_sum, H, row),
                    &ws.statuses[1],
                )
            })
            .unwrap();
            assert_eq!(
                ints(&ws.buffers[10]),
                ints(&prefill::row(&b[8], c.topk, row))
            );
            assert!(ints(&ws.buffers[10]).iter().all(|&id| id >= 378));
            assert_eq!(
                bits(&ws.buffers[11]),
                bits(&prefill::row(&b[9], c.topk, row))
            );
            assert_eq!(ws.status(1), ints(&b[6])[N + row]);
            for (index, src) in [13, 14, 15].into_iter().enumerate() {
                expected[index].extend(bits(&ws.buffers[src]));
            }
        }
        for (index, actual) in [12, 13, 14].into_iter().enumerate() {
            assert_eq!(
                bits(&b[actual]),
                expected[index],
                "{gate_type:?}/{down_type:?} stage {actual}"
            );
        }
        assert_eq!(bits(&b[5]), bits(&expected_sum));
        let expected_hc = MetalTensor::zeros_f32(&ctx, output.shape.clone()).unwrap();
        submit(&ctx, "native_rows_hc_reference", [0; 4], |enc| {
            metal::encode_mhc4_post_rows(
                &ctx,
                enc,
                H,
                N,
                &expected_sum,
                &input,
                &scratch.hc.post,
                &scratch.hc.comb,
                &expected_hc,
            )
        })
        .unwrap();
        assert_eq!(bits(&output), bits(&expected_hc));
        for chunk in [31, 32, 33, 127, 128, 129, 255, 256, 257] {
            for start in (0..N).step_by(chunk) {
                let rows = chunk.min(N - start);
                let mut part = Rows::new(&ctx, c, rows, 2).unwrap();
                part.hc.normalized = scratch
                    .hc
                    .normalized
                    .view_subrange((start * H) as u64, vec![H as u64, rows as u64]);
                for (index, width) in [(0, E), (4, H)] {
                    part.buffers[index] = b[index]
                        .view_subrange((start * width) as u64, vec![width as u64, rows as u64]);
                }
                submit(&ctx, "native_rows_partition", [0; 4], |enc| {
                    ws.encode_routed_rows(&ctx, enc, c, layer, 1, &part)
                })
                .unwrap();
                assert_eq!(
                    ints(&part.buffers[8]),
                    ints(&b[8])[start * c.topk..(start + rows) * c.topk]
                );
                for (index, width) in [
                    (7, E),
                    (9, c.topk),
                    (12, H * c.topk),
                    (13, H * c.topk),
                    (14, H * c.topk),
                    (5, H),
                ] {
                    assert_eq!(
                        bits(&part.buffers[index]),
                        bits(&b[index])[start * width..(start + rows) * width],
                        "{gate_type:?}/{down_type:?} chunk {chunk} buffer {index}"
                    );
                }
            }
        }
    }
}

#[test]
fn native_rows_v41_eligibility_keeps_grouped_and_deferred_paths() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let base = weights.model(&ctx);
    for (g, d, grouped, native) in [
        (GgmlType::IQ2_S, GgmlType::MXFP4, true, false),
        (GgmlType::IQ3_S, GgmlType::IQ3_S, true, false),
        (GgmlType::IQ3_XXS, GgmlType::IQ3_S, false, true),
        (GgmlType::Q3_K, GgmlType::MXFP4, false, true),
        (GgmlType::IQ2_XS, GgmlType::MXFP4, false, false),
        (GgmlType::IQ2_XXS, GgmlType::IQ3_XXS, false, false),
        (GgmlType::Q3_K, GgmlType::IQ2_XXS, false, false),
    ] {
        let banks = [
            MetalTensor {
                dtype: g,
                ..base.layers[0].banks[0].clone()
            },
            MetalTensor {
                dtype: g,
                ..base.layers[0].banks[1].clone()
            },
            MetalTensor {
                dtype: d,
                ..base.layers[0].banks[2].clone()
            },
        ];
        let mut model = weights.model(&ctx);
        model.config.hidden = 256;
        model.config.intermediate = 256;
        let layer = &mut model.layers[0];
        layer.banks = [&banks[0], &banks[1], &banks[2]];
        assert_eq!(grouped_supported(&model.config, layer), grouped);
        assert_eq!(native_rows_supported(&model.config, layer), native);
    }
}
