#[test]
#[ignore = "serial Metal, F32 tiled prefill correctness and short current-online screen"]
fn tiled_prefill_current_online_screen() {
    use objc2_metal::MTLBuffer;
    let ctx = MetalContext::new().unwrap();
    let mut k = vec![0_u16; 256 + 32768 * 256];
    let mut v = k.clone();
    for i in 0..32768 * 256 {
        k[256 + i] = half::f16::from_f32(((i * 13 % 103) as f32 - 51.0) * 0.03125).to_bits();
        v[256 + i] = half::f16::from_f32(
            (if i / 256 % 2 == 0 { 0.75 } else { -0.75 })
                + (i % 256 / 128) as f32 * 0.25
                + (i % 128 % 7) as f32 * 0.03125,
        )
        .to_bits();
    }
    let values = |rows: usize, uniform: bool| -> Vec<f32> {
        (0..rows * 4096)
            .map(|i| {
                if uniform {
                    0.0
                } else {
                    ((i / 4096 * 11 + i / 128 % 32 * 17 + i % 128 * 7) % 127) as f32 * 0.0379
                        - 2.3877
                }
            })
            .collect()
    };
    let peaked = values(128, false);
    for dim in 0..128 {
        k[256 + 32767 * 256 + 128 + dim] =
            half::f16::from_f32(peaked[127 * 4096 + 31 * 128 + dim] * 8.0).to_bits();
    }
    let key_storage = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&k),
        vec![k.len() as u64],
        GgmlType::F16,
    )
    .unwrap();
    let value_storage = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&v),
        vec![v.len() as u64],
        GgmlType::F16,
    )
    .unwrap();
    for (base, rows, window, uniform) in [
        (0, 1, None, false),
        (0, 1, Some(1), true),
        (31, 3, None, false),
        (31, 3, Some(1), false),
        (31, 3, Some(32), false),
        (0, 16, None, false),
        (0, 128, None, false),
        (512, 16, None, false),
        (512, 128, None, false),
        (2048, 16, None, false),
        (2048, 128, None, false),
        (6208, 16, None, false),
        (6208, 16, Some(2048), false),
        (6096, 128, None, false),
        (32640, 16, None, false),
        (32640, 128, None, false),
        (32640, 128, Some(2048), false),
        (32640, 128, None, true),
    ] {
        let values = values(rows, uniform);
        let mut padded = vec![-77.0; 8];
        padded.extend(&values);
        let query = tensor_from_f32(&ctx, &padded).view_subrange(8, vec![values.len() as u64]);
        let key = key_storage.view_subrange(256, vec![((base + rows) * 256) as u64]);
        let value = value_storage.view_subrange(256, vec![((base + rows) * 256) as u64]);
        let storages = [
            tensor_from_f32(&ctx, &vec![-77.0; values.len() + 16]),
            tensor_from_f32(&ctx, &vec![-77.0; values.len() + 16]),
        ];
        let outputs = storages
            .each_ref()
            .map(|storage| storage.view_subrange(8, vec![values.len() as u64]));
        let run = |tiled: bool| {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let before = TILED_PREFILL_DISPATCHES.get();
            with_tiled_prefill(tiled, || {
                encode_muse_glimmer_attn_prefill_with_online(
                    &ctx,
                    &encoder,
                    &query,
                    &key,
                    &value,
                    &outputs[usize::from(tiled)],
                    rows,
                    base,
                    32,
                    2,
                    128,
                    window,
                    true,
                )
            })
            .unwrap();
            assert_eq!(TILED_PREFILL_DISPATCHES.get() - before, u64::from(tiled));
            encoder.end();
            let start = std::time::Instant::now();
            command.commit();
            command.waitUntilCompleted();
            let wall = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(
                command.status(),
                objc2_metal::MTLCommandBufferStatus::Completed
            );
            assert!(command.error().is_none());
            (
                wall,
                (command.GPUEndTime() - command.GPUStartTime()) * 1000.0,
            )
        };
        for output in &outputs {
            unsafe {
                (output.buffer.contents().as_ptr() as *mut u8)
                    .add(output.offset as usize)
                    .write_bytes(0xff, values.len() * 4);
            }
        }
        run(false);
        run(true);
        let actual = read_f32(&outputs[1]);
        let gpu_delta = context_numerical_check(&actual, &read_f32(&outputs[0]));
        let mut f64_delta = 0.0_f64;
        for row in [0, (rows - 1).min(1), rows - 1] {
            let end = base + row + 1;
            let start = window.map(|w| end.saturating_sub(w)).unwrap_or(0);
            for head in [0, 31] {
                let expected = context_oracle_f64(
                    &values[row * 4096..(row + 1) * 4096],
                    &k,
                    &v,
                    head,
                    start,
                    end,
                );
                for dim in 0..128 {
                    f64_delta = f64_delta
                        .max((actual[row * 4096 + head * 128 + dim] as f64 - expected[dim]).abs());
                }
            }
        }
        assert!(f64_delta <= 5e-4, "independent F64 error {f64_delta}");
        for storage in &storages {
            let data = read_f32(storage);
            assert!(
                data[..8]
                    .iter()
                    .chain(&data[data.len() - 8..])
                    .all(|&v| v == -77.0)
            );
        }
        eprintln!(
            "MUSE_TILED_JSON {}",
            serde_json::json!({"kind":"correctness", "base":base,"rows":rows,"window":window,"uniform":uniform,"gpu_max_abs":gpu_delta,"f64_max_abs":f64_delta,"guards":true,"dispatch_witness":true})
        );
        if rows >= 16 && !uniform {
            run(false);
            run(true);
            for (pair, order) in [[false, true], [true, false]].into_iter().enumerate() {
                for tiled in order {
                    let (wall_ms, gpu_ms) = run(tiled);
                    eprintln!(
                        "MUSE_TILED_JSON {}",
                        serde_json::json!({"kind":"timing_screen","base":base,"rows":rows,"window":window,"pair":pair,"tiled":tiled,"wall_ms":wall_ms,"gpu_ms":gpu_ms})
                    );
                }
            }
            context_numerical_check(&read_f32(&outputs[1]), &read_f32(&outputs[0]));
            for storage in &storages {
                let data = read_f32(storage);
                assert!(
                    data[..8]
                        .iter()
                        .chain(&data[data.len() - 8..])
                        .all(|&v| v == -77.0)
                );
            }
        }
    }
}

#[test]
#[ignore = "serial Metal, tiled prefill independent model-context oracle"]
fn tiled_prefill_model_context_crosschecks() {
    with_tiled_prefill(true, || attention_model_context_oracle(false));
}

#[test]
fn tiled_prefill_policy_uses_work_not_context() {
    for rows in [0, 1, 16, 32, 64, 112, 127, 128, 129] {
        for offset in [0, 16, 32, 64] {
            assert_eq!(
                tiled_prefill_work_eligible(rows, offset),
                rows == 128 && offset % 32 == 0
            );
        }
    }
}
