#[test]
#[ignore = "serial Metal, Muse packed online causal/window and paired kernel screen"]
fn packed_online_attention_screen() {
    packed_online_screen(false);
}

#[test]
#[ignore = "serial Metal, Muse 8K/32K packed online attention screen"]
fn packed_online_long_attention_screen() {
    packed_online_screen(true);
}

fn packed_online_screen(long: bool) {
    let ctx = MetalContext::new().unwrap();
    let cases: &[(usize, usize, Option<usize>, bool, bool)] = if long {
        &[
            (7160, 16, None, false, false),
            (8064, 128, None, false, true),
            (8064, 128, Some(2048), false, false),
            (32640, 128, None, false, true),
            (32640, 128, Some(2048), false, true),
            (32767, 1, None, false, false),
            (32760, 16, Some(2048), true, false),
        ]
    } else {
        &[
            (0, 1, None, false, false),
            (0, 16, None, false, false),
            (31, 16, None, false, false),
            (896, 128, None, false, true),
            (1920, 128, Some(2048), false, false),
            (2046, 16, Some(2048), true, false),
            (6144, 80, None, false, true),
            (6144, 80, Some(2048), false, true),
            (7040, 128, None, false, false),
            (7167, 1, None, false, false),
        ]
    };
    for &(base, rows, window, analytic, timed) in cases {
        let end = base + rows;
        let q: Vec<f32> = (0..rows * 4096)
            .map(|i| {
                if analytic {
                    0.0
                } else {
                    ((i * 17 % 101) as f32 - 50.0) * 0.02
                }
            })
            .collect();
        let k: Vec<f32> = (0..end * 256)
            .map(|i| ((i * 13 % 89) as f32 - 44.0) * 0.024)
            .collect();
        let v: Vec<f32> = (0..end * 256)
            .map(|i| {
                if analytic {
                    ((i / 256 % 31) as f32 - 15.0) * 0.01
                        + (i % 256 / 128) as f32 * 0.5
                        + (i % 128 % 7) as f32 * 0.02
                } else {
                    ((i * 19 % 97) as f32 - 48.0) * 0.02
                }
            })
            .collect();
        let mut q_storage = vec![-77.0; 4];
        q_storage.extend(&q);
        let q_tensor = tensor_from_f32(&ctx, &q_storage).view_subrange(4, vec![q.len() as u64]);
        let cache = |values: &[f32]| {
            let mut storage = vec![-77.0; 256];
            storage.extend_from_slice(values);
            tensor_from_f16(&ctx, &storage).view_subrange(256, vec![values.len() as u64])
        };
        let key = cache(&k);
        let value = cache(&v);
        let output_storage = tensor_from_f32(&ctx, &vec![-77.0; q.len() + 8]);
        let output = output_storage.view_subrange(4, vec![q.len() as u64]);
        let run = |online: bool, chains: usize| {
            let started = std::time::Instant::now();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            with_packed_online(online, || {
                for _ in 0..chains {
                    if !online
                        && window.map(|window| end.min(window)).unwrap_or(end)
                            > MUSE_GLIMMER_MATERIALIZED_ATTENTION_MAX_POSITIONS
                    {
                        for row in 0..rows {
                            let visible_end = base + row + 1;
                            let start = window
                                .map(|window| visible_end.saturating_sub(window))
                                .unwrap_or(0);
                            let q_row = q_tensor.view_subrange((row * 4096) as u64, vec![4096]);
                            let out_row = output.view_subrange((row * 4096) as u64, vec![4096]);
                            let k_row = key.view_subrange(
                                (start * 256) as u64,
                                vec![((visible_end - start) * 256) as u64],
                            );
                            let v_row = value.view_subrange(
                                (start * 256) as u64,
                                vec![((visible_end - start) * 256) as u64],
                            );
                            encode_muse_glimmer_attn_decode_f16kv_f32(
                                &ctx,
                                &encoder,
                                &q_row,
                                &k_row,
                                &v_row,
                                &out_row,
                                32,
                                2,
                                128,
                                visible_end - start,
                            )
                            .unwrap();
                        }
                    } else {
                        encode_muse_glimmer_attn_prefill_f16kv_f32(
                            &ctx, &encoder, &q_tensor, &key, &value, &output, rows, base, 32, 2,
                            128, window,
                        )
                        .unwrap();
                    }
                }
            });
            encoder.end();
            command.commit();
            crate::metal::wait_unchecked(&command);
            let wall_ms = started.elapsed().as_secs_f64() * 1e3 / chains as f64;
            assert_eq!(
                command.status(),
                objc2_metal::MTLCommandBufferStatus::Completed
            );
            assert!(command.error().is_none());
            (
                wall_ms,
                (command.GPUEndTime() - command.GPUStartTime()) * 1e3 / chains as f64,
            )
        };
        run(false, 1);
        let reference = read_f32(&output);
        unsafe {
            (output.buffer.contents().as_ptr() as *mut u8)
                .add(output.offset as usize)
                .write_bytes(0xff, q.len() * 4);
        }
        run(true, 1);
        let actual = read_f32(&output);
        let mut dot = 0.0_f64;
        let mut aa = 0.0_f64;
        let mut bb = 0.0_f64;
        let mut max_abs = 0.0_f32;
        for (&a, &b) in actual.iter().zip(&reference) {
            assert!(a.is_finite() && b.is_finite());
            dot += a as f64 * b as f64;
            aa += (a as f64).powi(2);
            bb += (b as f64).powi(2);
            max_abs = max_abs.max((a - b).abs());
        }
        let cosine = dot / (aa * bb).sqrt();
        assert!(
            cosine >= 0.999_999 && max_abs <= 5e-4,
            "base={base} rows={rows} window={window:?} cosine={cosine} max_abs={max_abs}"
        );
        let guarded = read_f32(&output_storage);
        assert!(
            guarded[..4]
                .iter()
                .chain(&guarded[q.len() + 4..])
                .all(|&x| x == -77.0)
        );
        if analytic {
            for row in 0..rows {
                let end = base + row + 1;
                let start = end.saturating_sub(window.unwrap());
                for kv_head in 0..2 {
                    for dim in 0..128 {
                        let expected = (start..end)
                            .map(|pos| {
                                half::f16::from_f32(v[pos * 256 + kv_head * 128 + dim]).to_f64()
                            })
                            .sum::<f64>()
                            / (end - start) as f64;
                        for group in 0..16 {
                            let got = actual[row * 4096 + (kv_head * 16 + group) * 128 + dim];
                            assert!(
                                (got as f64 - expected).abs() <= 5e-5,
                                "analytic causal/window row={row} head={kv_head} dim={dim}"
                            );
                        }
                    }
                }
            }
        }
        eprintln!(
            "MUSE_PACKED_ONLINE_JSON {}",
            serde_json::json!({"kind":"oracle","base":base,"rows":rows,"window":window,"cosine":cosine,"max_abs":max_abs,"analytic":analytic,"offset_views_and_guards":true,"new_scratch_bytes":0})
        );
        if timed {
            for kind in ["warmup", "sample"] {
                for online in [false, true, true, false] {
                    let (wall_ms, gpu_ms) = run(online, 8);
                    eprintln!(
                        "MUSE_PACKED_ONLINE_JSON {}",
                        serde_json::json!({"kind":kind,"base":base,"rows":rows,"window":window,"arm":if online {"B"} else {"A"},"wall_ms":wall_ms,"gpu_ms":gpu_ms,"chains":8})
                    );
                }
            }
        }
    }
}
