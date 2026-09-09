#[test]
#[ignore = "serial Metal, equal512-row Q8 FFN batch-size falsifier; not model throughput"]
fn muse_prefill_equal_work_batch_screen() {
    let gguf = GgufFile::open(crate::test_fixtures::MUSE_GLIMMER_Q8_0.path()).unwrap();
    MuseGlimmerConfig::from_gguf(&gguf).unwrap();
    let ctx = MetalContext::new().unwrap();
    for (name, expected_shape) in [
        ("blk.0.ffn_gate.weight", [6656, 19968]),
        ("blk.0.ffn_down.weight", [19968, 6656]),
    ] {
        let desc = gguf.find(name).unwrap();
        assert_eq!(desc.dtype, GgmlType::Q8_0);
        assert_eq!(desc.shape, expected_shape);
        let n_in = desc.shape[0] as usize;
        let n_out = desc.shape[1] as usize;
        let transaction = ctx.begin_allocation_transaction();
        let sizes = [
            desc.n_bytes,
            (512 * n_in * 4) as u64,
            ((512 * n_out + 16) * 4) as u64,
        ];
        let price = |bytes| {
            ctx.price_shared_buffer_upper(bytes)
                .unwrap()
                .priced_upper_bytes
        };
        let admission = evaluate_metal_memory_admission_with_cpu_bytes(
            price(sizes[0]) + price(sizes[1]) + 3 * price(sizes[2]),
            256 * 1024 * 1024,
            MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        assert!(admission.admitted, "batch screen admission {admission:?}");
        let weight = MetalTensor::from_gguf_tensor(&ctx, desc, gguf.slice(desc)).unwrap();
        let values: Vec<f32> = (0..512 * n_in)
            .map(|i| ((i * 17 + i / n_in * 13) % 251) as f32 * 0.0079 - 0.9871)
            .collect();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&values),
            vec![(512 * n_in) as u64],
            GgmlType::F32,
        )
        .unwrap();
        drop(values);
        let storages: Vec<_> = (0..3)
            .map(|_| {
                MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&vec![-77.0_f32; 512 * n_out + 16]),
                    vec![(512 * n_out + 16) as u64],
                    GgmlType::F32,
                )
                .unwrap()
            })
            .collect();
        let outputs: Vec<_> = storages
            .iter()
            .map(|storage| storage.view_subrange(8, vec![(512 * n_out) as u64]))
            .collect();
        drop(transaction);
        let run = |index: usize| {
            let batch = [128, 256, 512][index];
            let started = std::time::Instant::now();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            for row in (0..512).step_by(batch) {
                let x = input.view_subrange((row * n_in) as u64, vec![(batch * n_in) as u64]);
                let y = outputs[index]
                    .view_subrange((row * n_out) as u64, vec![(batch * n_out) as u64]);
                encode_mat_mat_q8_0_f32(&ctx, &encoder, &weight, &x, &y, n_in, n_out, batch)
                    .unwrap();
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert!(command.error().is_none());
            let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
            assert!(gpu_ms.is_finite() && gpu_ms > 0.0 && wall_ms.is_finite() && wall_ms > 0.0);
            (wall_ms, gpu_ms)
        };
        for (i, output) in outputs.iter().enumerate() {
            unsafe {
                (output.buffer.contents().as_ptr() as *mut u8)
                    .add(output.offset as usize)
                    .write_bytes(0xff, 512 * n_out * 4);
            }
            run(i);
        }
        let reference = read_f32(&outputs[0]);
        assert!(reference.iter().all(|v| v.is_finite()));
        for output in &outputs[1..] {
            assert_logits_bitwise_equal("batch-size output", &read_f32(output), &reference);
        }
        for storage in &storages {
            let data = read_f32(storage);
            assert!(
                data[..8]
                    .iter()
                    .chain(&data[data.len() - 8..])
                    .all(|&v| v == -77.0)
            );
        }
        for i in 0..3 {
            run(i);
        }
        for (round, order) in [[0, 1, 2], [2, 1, 0]].into_iter().enumerate() {
            for index in order {
                let (wall_ms, gpu_ms) = run(index);
                eprintln!(
                    "MUSE_BATCH_JSON {}",
                    serde_json::json!({"kind":"equal_work","weight":name,"rows":512,"batch":([128,256,512][index]),"round":round,"wall_ms":wall_ms,"gpu_ms":gpu_ms,"same_encoder":true})
                );
            }
        }
        for storage in &storages {
            let data = read_f32(storage);
            assert!(
                data[..8]
                    .iter()
                    .chain(&data[data.len() - 8..])
                    .all(|&v| v == -77.0)
            );
        }
        for output in &outputs {
            assert_logits_bitwise_equal("post-timing batch output", &read_f32(output), &reference);
        }
    }
}
