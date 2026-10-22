use super::*;

mod real;

#[test]
fn rowwise_linear_override_is_scoped_and_changes_dispatch() {
    let ctx = MetalContext::new().unwrap();
    let linear = Linear {
        ctx: &ctx,
        tensor: upload(&ctx, &[0.25; 256], vec![32, 8]).unwrap(),
        native_control: None,
        input: 32,
        output: 8,
    };
    let x = upload(&ctx, &[0.125; 96], vec![32, 3]).unwrap();
    let y = allocate(&ctx, 8, 3, GgmlType::F32).unwrap();
    let dispatch = || {
        metal::dispatch_census_begin();
        let result = submit(&ctx, "linear_rows_override", [3, 0, 0, 0], |enc| {
            linear.encode_rows(enc, &x, &y, 3)
        });
        let rows = metal::dispatch_census_take();
        result.unwrap();
        rows
    };
    let grouped = metal::grouped_projection_pipeline_name(GgmlType::F32).unwrap();
    let normal = dispatch();
    assert_eq!(normal.len(), 1);
    assert_eq!(normal[0].kernel, grouped);
    let expected = f32_slice(&y).unwrap().to_vec();
    with_rowwise_linears(|| {
        let rows = dispatch();
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|r| r.kernel != grouped));
        assert_eq!(f32_slice(&y).unwrap(), expected);
        with_rowwise_linears(|| assert!(ROWWISE_LINEAR_ROWS.with(|flag| flag.get())));
        assert!(ROWWISE_LINEAR_ROWS.with(|flag| flag.get()));
        assert!(
            !std::thread::spawn(|| ROWWISE_LINEAR_ROWS.with(|flag| flag.get()))
                .join()
                .unwrap()
        );
    });
    assert!(!ROWWISE_LINEAR_ROWS.with(|flag| flag.get()));
    assert!(std::panic::catch_unwind(|| with_rowwise_linears(|| panic!("scope test"))).is_err());
    assert!(!ROWWISE_LINEAR_ROWS.with(|flag| flag.get()));
    assert_eq!(dispatch()[0].kernel, grouped);
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() <= 3e-5 + 3e-5 * b.abs(), "{i}: {a} != {b}");
    }
}

#[test]
fn chunk_dispatch_and_submission_census() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    for rows in [2, 7, 32, 33] {
        let tokens = (0..rows).map(|i| (i % 5) as u32).collect::<Vec<_>>();
        let mut dispatch_counts = Vec::new();
        for with_logits in [false, true] {
            let mut state = model.new_state(rows).unwrap();
            let scope = profile::Scope::begin().unwrap();
            metal::dispatch_census_begin();
            let result = run(
                &model,
                &mut state,
                &tokens,
                None,
                DEFAULT_CHUNK_ROWS,
                with_logits,
            );
            let dispatches = metal::dispatch_census_take();
            let report = scope.finish();
            let logits = result.unwrap();
            assert_eq!(
                logits.len(),
                if with_logits { model.config.vocab } else { 0 }
            );
            let chunks = rows.div_ceil(DEFAULT_CHUNK_ROWS);
            assert_eq!(
                report
                    .operations
                    .iter()
                    .map(|o| o.completed_submits)
                    .sum::<usize>(),
                chunks
            );
            assert!(
                report
                    .operations
                    .iter()
                    .all(|o| matches!(o.operation, "prefill_chunk_rows" | "gpu_step"))
            );
            let encoders = dispatches
                .iter()
                .map(|d| d.encoder_ordinal)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(encoders.len(), chunks);
            assert!(dispatches.iter().all(|d| !d.encoder_concurrent));
            assert_eq!(
                dispatches
                    .iter()
                    .filter(|d| d.kernel == "kernel_deepseek_v41_route")
                    .count(),
                chunks * model.layers.len()
            );
            let grouped_chunks = tokens
                .chunks(DEFAULT_CHUNK_ROWS)
                .filter(|c| c.len() > 1)
                .count();
            // Half-staged MXFP4 route: down tile for the gate pass and the
            // down pass, clamped up tile in between; the tensor tile where
            // the tier exists, the generic tile elsewhere.
            let (down_name, up_name) = if metal::moe_grouped_tensor_selected(&ctx) {
                (
                    "kernel_moe_down_mxfp4_f32_grouped_slots_tensor",
                    "kernel_moe_up_clamped_mxfp4_f32_grouped_slots_tensor",
                )
            } else {
                (
                    "kernel_moe_down_mxfp4_f32_grouped_slots_generic",
                    "kernel_moe_up_clamped_mxfp4_f32_grouped_slots_clamped",
                )
            };
            for (name, per_layer) in [(down_name, 2), (up_name, 1)] {
                assert_eq!(
                    dispatches.iter().filter(|d| d.kernel == name).count(),
                    per_layer * grouped_chunks * model.layers.len()
                );
            }
            assert_eq!(
                dispatches
                    .iter()
                    .filter(|d| d.kernel == "kernel_deepseek_v4_hc_controls_batch")
                    .count(),
                2 * model.layers.len()
                    * tokens
                        .chunks(DEFAULT_CHUNK_ROWS)
                        .filter(|c| c.len() > 1)
                        .count()
            );
            assert!(dispatches.iter().any(|d| Some(&d.kernel)
                == metal::grouped_projection_pipeline_name(GgmlType::F32).as_ref()));
            dispatch_counts.push(dispatches.len());
        }
        assert_eq!(
            dispatch_counts[1] - dispatch_counts[0],
            3,
            "only final collapse/norm/head added"
        );
    }
}

#[test]
fn early_layer_route_failure_survives_later_healthy_layer() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let mut model = weights.model(&ctx);
    model.layers[0].bias[0] = f32::NAN;
    let c = &model.config;
    let ws = ffn_gpu::Workspace::new(&model).unwrap();
    let scratch = ffn_gpu::Rows::new(&ctx, c, 3, 2).unwrap();
    let x = upload(
        &ctx,
        &vec![0.25; 3 * 4 * c.hidden],
        vec![c.hidden as u64, 4, 3],
    )
    .unwrap();
    let pre = upload(&ctx, &[1.0, 0.0, 0.0, 0.0].repeat(3), vec![4, 3]).unwrap();
    let output = allocate(&ctx, 4 * c.hidden, 3, GgmlType::F32)
        .unwrap()
        .view_bytes(0, vec![c.hidden as u64, 4, 3]);
    // Both layers read healthy independent inputs, so layer 1 succeeds even
    // though layer 0 fails. Shared activation scratch is overwritten normally.
    submit(&ctx, "two_layer_route_status", [3, 2, 0, 0], |enc| {
        for id in 0..2 {
            ws.encode_rows(
                &ctx,
                enc,
                c,
                &model.layers[id],
                id,
                &x,
                &pre,
                &output,
                &scratch,
            )?;
        }
        Ok(())
    })
    .unwrap();
    assert!(
        scratch
            .validate_status()
            .unwrap_err()
            .contains("layer 0 row 0 route status")
    );
    let mut state = model.new_state(3).unwrap();
    let scope = profile::Scope::begin().unwrap();
    let error = model.prefill(&mut state, &[0, 3, 1], None).unwrap_err();
    let report = scope.finish();
    assert!(error.to_string().contains("layer 0 row 0 route status"));
    assert_eq!(
        report
            .operations
            .iter()
            .map(|o| o.completed_submits)
            .sum::<usize>(),
        1
    );
    assert!(state.is_poisoned());
    assert_eq!(state.position(), 0);
}

#[test]
fn public_single_row_prefill_uses_one_scalar_command() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut serial = model.new_state(2).unwrap();
    let mut batch = model.new_state(2).unwrap();
    let expected = model.step(&mut serial, 3, None).unwrap();
    let scope = profile::Scope::begin().unwrap();
    let result: std::result::Result<Vec<f32>, StateError> = model.prefill(&mut batch, &[3], None);
    assert_eq!(result.unwrap(), expected);
    let report = scope.finish();
    assert_eq!(
        report
            .operations
            .iter()
            .map(|o| o.completed_submits)
            .sum::<usize>(),
        1
    );
    assert!(report.operations.iter().all(|o| o.operation == "gpu_step"));
    assert_eq!(
        model.step(&mut batch, 1, None).unwrap(),
        model.step(&mut serial, 1, None).unwrap()
    );
}

#[test]
fn headless_prefill_skips_head_including_single_row_and_tail() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let mut model = weights.model(&ctx);
    for count in [1, 7, 33] {
        let tokens = (0..count).map(|i| (i % 5) as u32).collect::<Vec<_>>();
        let mut serial = model.new_state(count + 1).unwrap();
        let mut batch = model.new_state(count + 1).unwrap();
        for &token in &tokens {
            model.advance(&mut serial, token, None).unwrap();
        }
        let head = model.head.clone();
        // An accidental vocabulary projection fails its input-shape validation.
        model.head.input += 1;
        let scope = profile::Scope::begin().unwrap();
        let result: std::result::Result<(), StateError> =
            model.advance_prefill(&mut batch, &tokens, None);
        result.unwrap();
        let report = scope.finish();
        model.head = head;
        let scalar_rows = tokens
            .chunks(DEFAULT_CHUNK_ROWS)
            .filter(|rows| rows.len() == 1)
            .count();
        assert_eq!(
            report
                .operations
                .iter()
                .filter(|o| o.operation == "gpu_step")
                .map(|o| o.completed_submits)
                .sum::<usize>(),
            scalar_rows
        );
        assert_eq!(batch.position(), count);
        assert_eq!(batch.addresses, serial.addresses);
        close(
            &model.step(&mut batch, 2, None).unwrap(),
            &model.step(&mut serial, 2, None).unwrap(),
        );
    }
}

#[test]
fn caller_split_headless_chunks_match_final_prefill() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let tokens = [0, 3, 1, 4, 2, 0, 4];
    let mut full = model.new_state(9).unwrap();
    let mut split = model.new_state(9).unwrap();
    let expected = model.prefill(&mut full, &tokens, None).unwrap();
    model
        .advance_prefill(&mut split, &tokens[..3], None)
        .unwrap();
    model
        .advance_prefill(&mut split, &tokens[3..6], None)
        .unwrap();
    close(
        &model.prefill(&mut split, &tokens[6..], None).unwrap(),
        &expected,
    );
    for token in [2, 1] {
        close(
            &model.step(&mut split, token, None).unwrap(),
            &model.step(&mut full, token, None).unwrap(),
        );
    }
}

#[test]
fn layer_major_chunks_match_serial_and_snapshot_continuation() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let mut model = weights.model(&ctx);
    model.config.index_topk = 1;
    model.config.candidate_topk = 1;
    for chunk in [1, 2, 3, 7] {
        for prefix in [0, 1] {
            let mut serial = model.new_state(16).unwrap();
            let mut batch = model.new_state(16).unwrap();
            if prefix != 0 {
                model.advance(&mut serial, 2, None).unwrap();
                model.advance(&mut batch, 2, None).unwrap();
            }
            let tokens = [0, 3, 1, 4, 2, 0, 4];
            let mut expected = Vec::new();
            for token in tokens {
                expected = model.step(&mut serial, token, None).unwrap();
            }
            let scope = profile::Scope::begin().unwrap();
            let actual = run(&model, &mut batch, &tokens, None, chunk, true).unwrap();
            let report = scope.finish();
            assert_eq!(
                report
                    .operations
                    .iter()
                    .filter(|o| o.operation == "prefill_chunk_rows")
                    .map(|o| o.completed_submits)
                    .sum::<usize>(),
                tokens.chunks(chunk).filter(|rows| rows.len() > 1).count()
            );
            assert_eq!(
                report
                    .operations
                    .iter()
                    .filter(|o| o.operation == "gpu_step")
                    .map(|o| o.completed_submits)
                    .sum::<usize>(),
                tokens.chunks(chunk).filter(|rows| rows.len() == 1).count()
            );
            close(&actual, &expected);
            assert_eq!(batch.addresses, serial.addresses);
            let snapshot = model.capture_snapshot(&batch).unwrap();
            let mut restored = model.new_state(16).unwrap();
            for _ in 0..10 {
                model.advance(&mut restored, 4, None).unwrap();
            }
            model.restore_snapshot(&mut restored, &snapshot).unwrap();
            for token in [2, 1, 3] {
                let expected = model.step(&mut serial, token, None).unwrap();
                let actual = model.step(&mut batch, token, None).unwrap();
                close(&actual, &expected);
                assert_eq!(actual, model.step(&mut restored, token, None).unwrap());
            }
        }
    }
}

#[test]
fn hybrid_reference_matches_layer_major() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut hybrid = model.new_state(8).unwrap();
    let mut batch = model.new_state(8).unwrap();
    let tokens = [0, 3, 1, 4, 2, 0, 4];
    close(
        &run_hybrid(&model, &mut hybrid, &tokens, None, 3).unwrap(),
        &run(&model, &mut batch, &tokens, None, 3, true).unwrap(),
    );
}

#[test]
fn linear_rows_native_f32_q8_and_bf16_with_offsets() {
    // The F32-operand contract (3e-5) is pinned on the grouped path.
    with_f32_linears(|| linear_rows_with_offsets(close));
}

#[test]
fn linear_rows_mma_q8_and_bf16_with_offsets() {
    // Production multi-row route: half-staged MMA tiles, judged at a
    // half-activation tolerance; offsets and untouched rows still hold.
    linear_rows_with_offsets(|a, b| {
        assert_eq!(a.len(), b.len());
        for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
            assert!((a - b).abs() <= 2e-3 + 4e-3 * b.abs(), "{i}: {a} != {b}");
        }
    });
}

fn linear_rows_with_offsets(check: impl Fn(&[f32], &[f32])) {
    let ctx = MetalContext::new().unwrap();
    let (input, output) = (32, 8);
    for dtype in [GgmlType::F32, GgmlType::Q8_0, GgmlType::BF16] {
        let mut bytes = Vec::new();
        for row in 0..16 {
            if dtype == GgmlType::Q8_0 {
                bytes.extend_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
                bytes.extend((0..32).map(|j| ((row + j) as i8 - 17) as u8));
            } else {
                for j in 0..32 {
                    let v = ((row + j) as f32 - 17.0) / 32.0;
                    if dtype == GgmlType::BF16 {
                        bytes.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
                    } else {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        let tensor = MetalTensor::from_bytes(&ctx, &bytes, vec![32, 16], dtype).unwrap();
        let stride = row_bytes(&tensor).unwrap();
        let linear = Linear {
            ctx: &ctx,
            tensor: tensor.view_bytes(8 * stride, vec![32, 8]),
            native_control: native_control(&ctx, dtype).unwrap(),
            input,
            output,
        };
        for rows in [1, 2, 3, 7] {
            let values = (0..input * rows)
                .map(|i| (i as f32 - 11.0) / 31.001)
                .collect::<Vec<_>>();
            let mut padded = vec![123.0; input];
            padded.extend_from_slice(&values);
            let x = upload(&ctx, &padded, vec![padded.len() as u64])
                .unwrap()
                .view_subrange(input as u64, vec![input as u64, rows as u64]);
            let y = upload(
                &ctx,
                &vec![123.0; output * (rows + 2)],
                vec![(output * (rows + 2)) as u64],
            )
            .unwrap();
            let view = y.view_subrange(output as u64, vec![output as u64, rows as u64]);
            submit(&ctx, "linear_rows_test", [rows, 0, 0, 0], |enc| {
                linear.encode_rows(enc, &x, &view, rows)
            })
            .unwrap();
            let expected = values
                .chunks(input)
                .flat_map(|row| linear.forward(row).unwrap())
                .collect::<Vec<_>>();
            check(f32_slice(&view).unwrap(), &expected);
            let all = f32_slice(&y).unwrap();
            assert!(
                all[..output]
                    .iter()
                    .chain(&all[output * (rows + 1)..])
                    .all(|&v| v == 123.0)
            );
        }
    }
}

#[test]
fn invalid_prefill_request_preserves_state() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut state = model.new_state(3).unwrap();
    for (tokens, chunk) in [
        (vec![], 1),
        (vec![0], 0),
        (vec![0], DEFAULT_CHUNK_ROWS + 1),
        (vec![0, u32::MAX], 2),
        (vec![0; 4], 2),
    ] {
        assert!(run(&model, &mut state, &tokens, None, chunk, true).is_err());
        assert_eq!(state.position(), 0);
        assert!(!state.is_poisoned());
    }
    assert!(
        run(
            &model,
            &mut state,
            &[0],
            Some(&AtomicBool::new(true)),
            1,
            true
        )
        .is_err()
    );
    assert_eq!(state.position(), 0);
    assert!(!state.is_poisoned());
}

fn vary(linear: &mut Linear<'_>, seed: usize) {
    let values = (0..linear.input * linear.output)
        .map(|i| {
            (((i * 17 + seed * 13 + i / linear.input * 7) % 61) as f32 - 30.0)
                / (20.0 * (linear.input as f32).sqrt())
        })
        .collect::<Vec<_>>();
    linear.tensor = upload(
        linear.ctx,
        &values,
        vec![linear.input as u64, linear.output as u64],
    )
    .unwrap();
}

#[test]
fn nonuniform_dynamic_schedule_preserves_all_causal_regions() {
    // Serial steps versus chunked prefill at the F32-operand contract; the
    // routed experts are pinned to the F32 tiles for that comparison.
    super::super::ffn_gpu::tests::with_grouped_rows(
        nonuniform_dynamic_schedule_preserves_all_causal_regions_body,
    );
}

fn nonuniform_dynamic_schedule_preserves_all_causal_regions_body() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let mut model = weights.model(&ctx);
    let engram = model.layers[1].engram.take();
    model.layers = model
        .layers
        .into_iter()
        .enumerate()
        .filter_map(|(id, layer)| [0, 2, 3, 20, 24, 25].contains(&id).then_some(layer))
        .collect();
    model.schedule = super::super::super::gguf::Schedule {
        layers: 6,
        kv_source_layers: vec![1, 3],
        index_source_layers: vec![1, 3, 4],
        compression_ratios: vec![0, 2, 2, 1, 1, 1],
        candidate_source_layer: Some(3),
    };
    model.layers[0].engram = engram;
    model.engram_hash.layers.truncate(1);
    model.engram_hash.layers[0].layer = 0;
    model.config.index_topk = 2;
    let embeddings = upload(
        &ctx,
        &(0..256)
            .map(|i| ((i * 7 % 37) as f32 - 18.0) / 19.001)
            .collect::<Vec<_>>(),
        vec![32, 8],
    )
    .unwrap();
    model.embedding = &embeddings;
    vary(&mut model.head, 91);
    for (id, layer) in model.layers.iter_mut().enumerate() {
        vary(&mut layer.attn_hc.projection, id + 1);
        vary(&mut layer.ffn_hc.projection, id + 3);
        vary(&mut layer.router, id + 7);
        for (j, p) in layer.shared.iter_mut().enumerate() {
            vary(p, id + j + 11);
        }
        let a = &mut layer.attention;
        vary(&mut a.query_a, id + 19);
        vary(&mut a.query_b, id + 23);
        vary(&mut a.kv, id + 29);
        for (j, p) in a.output_a.iter_mut().enumerate() {
            vary(p, id + j + 31);
        }
        vary(&mut a.output_b, id + 37);
        if let Some(compressor) = a.compressor.as_mut() {
            vary(&mut compressor.kv, id + 41);
            if let Some(gate) = compressor.gate.as_mut() {
                vary(gate, id + 43);
            }
        }
        if let Some(index) = a.index.as_mut() {
            index.topk = model.config.index_topk;
            vary(&mut index.query, id + 47);
            vary(&mut index.weights, id + 53);
            if let Some(key) = index.key.as_mut() {
                vary(key, id + 59);
            }
        }
    }
    for chunk in [1, 2, 3, 7, 32] {
        for prefix in [0, 1, 4] {
            let tokens = [0, 3, 1, 4, 2, 0, 4, 7, 5, 6, 3, 2, 1];
            let mut serial = model.new_state(32).unwrap();
            let mut batch = model.new_state(prefix + tokens.len()).unwrap();
            for i in 0..prefix {
                model.advance(&mut serial, i as u32, None).unwrap();
            }
            if prefix != 0 {
                let prefix_tokens = (0..prefix).map(|i| i as u32).collect::<Vec<_>>();
                run(&model, &mut batch, &prefix_tokens, None, chunk, true).unwrap();
            }
            let mut expected = Vec::new();
            for token in tokens {
                expected = model.step(&mut serial, token, None).unwrap();
            }
            close(
                &run(&model, &mut batch, &tokens, None, chunk, true).unwrap(),
                &expected,
            );
            let position = batch.position();
            let a = serial
                .gpu_step
                .as_ref()
                .unwrap()
                .attention()
                .causal_regions(position)
                .unwrap();
            let b = batch
                .gpu_step
                .as_ref()
                .unwrap()
                .attention()
                .causal_regions(position)
                .unwrap();
            for ((a, na), (b, nb)) in a.iter().zip(&b) {
                assert_eq!(na, nb);
                close(&f32_slice(a).unwrap()[..*na], &f32_slice(b).unwrap()[..*nb]);
            }
            let snapshot = model.capture_snapshot(&batch).unwrap();
            let mut restored = model.new_state(position).unwrap();
            model.restore_snapshot(&mut restored, &snapshot).unwrap();
            model.ensure_capacity(&mut batch, position + 3).unwrap();
            model.ensure_capacity(&mut restored, position + 3).unwrap();
            let suffix = [5, 2, 7];
            for token in suffix {
                expected = model.step(&mut serial, token, None).unwrap();
            }
            let actual = run(&model, &mut batch, &suffix, None, chunk, true).unwrap();
            close(&actual, &expected);
            assert_eq!(
                actual,
                run(&model, &mut restored, &suffix, None, chunk, true).unwrap()
            );
        }
    }
}

#[test]
fn unaligned_f32_geometry_uses_native_row_fallback() {
    let ctx = MetalContext::new().unwrap();
    let linear = Linear {
        ctx: &ctx,
        tensor: upload(&ctx, &[0.25; 24], vec![8, 3]).unwrap(),
        native_control: None,
        input: 8,
        output: 3,
    };
    let values = (0..24).map(|i| i as f32 / 7.001).collect::<Vec<_>>();
    let x = upload(&ctx, &values, vec![8, 3]).unwrap();
    let y = upload(&ctx, &[0.0; 9], vec![3, 3]).unwrap();
    submit(&ctx, "unaligned_rows_test", [3, 0, 0, 0], |enc| {
        linear.encode_rows(enc, &x, &y, 3)
    })
    .unwrap();
    let expected = values
        .chunks(8)
        .flat_map(|row| linear.forward(row).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(f32_slice(&y).unwrap(), expected);
}

#[test]
fn batch_failure_poisoning_and_all_row_route_statuses() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let mut model = weights.model(&ctx);
    let c = &model.config;
    let ws = ffn_gpu::Workspace::new(&model).unwrap();
    let scratch = ffn_gpu::Rows::new(&ctx, c, 3, 1).unwrap();
    let mut values = vec![0.25; 3 * 4 * c.hidden];
    values[0] = f32::NAN;
    let x = upload(&ctx, &values, vec![c.hidden as u64, 4, 3]).unwrap();
    let pre = upload(&ctx, &[1.0, 0.0, 0.0, 0.0].repeat(3), vec![4, 3]).unwrap();
    let output = allocate(&ctx, 4 * c.hidden, 3, GgmlType::F32)
        .unwrap()
        .view_bytes(0, vec![c.hidden as u64, 4, 3]);
    submit(&ctx, "ffn_bad_first_row", [3, 0, 0, 0], |enc| {
        ws.encode_rows(
            &ctx,
            enc,
            c,
            &model.layers[0],
            0,
            &x,
            &pre,
            &output,
            &scratch,
        )
    })
    .unwrap();
    assert!(scratch.validate_status().is_err());
    model.layers[2].bias[0] = f32::NAN;
    let mut state = model.new_state(8).unwrap();
    assert!(
        run(&model, &mut state, &[0, 3, 1], None, 3, true)
            .unwrap_err()
            .to_string()
            .contains("route status")
    );
    assert!(state.is_poisoned());
    assert_eq!(state.position(), 0);
    assert!(model.capture_snapshot(&state).is_err());
    assert!(model.step(&mut state, 2, None).is_err());
}
