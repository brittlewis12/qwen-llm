use super::*;

#[test]
fn metadata_schedule_drives_full_gpu_graph_and_engram_at_zero() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    for use_engram in [false, true] {
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
        if use_engram {
            model.layers[0].engram = engram;
            model.engram_hash.layers.truncate(1);
            model.engram_hash.layers[0].layer = 0;
        } else {
            model.engram_hash.layers.clear();
        }
        let mut state = model.new_state(3).unwrap();
        let mut plain = model.new_state_with_gpu_step(3).unwrap();
        for (position, token) in [0, 3, 1].into_iter().enumerate() {
            let mut count = 0;
            let logits = model
                .step_with_trace(
                    &mut state,
                    token,
                    None,
                    Some(&mut |trace| {
                        assert_eq!((trace.position, trace.layer), (position, count));
                        count += 1;
                        assert!(trace.hidden.iter().all(|v| v.is_finite()));
                        Ok(())
                    }),
                )
                .unwrap();
            assert_eq!(count, 6);
            assert_eq!(logits, model.step(&mut plain, token, None).unwrap());
            assert_eq!(state.position(), position + 1);
            let workspace = state.gpu_step.as_ref().unwrap();
            assert_eq!(workspace.commands, if use_engram { 2 } else { 1 });
            assert_eq!(workspace.constants.len(), 6);
            assert_eq!(workspace.trace_copies, 12);
        }
    }
}

// F32 reduction trees differ from the scalar CPU/Python reference.
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() <= 3e-5 + 3e-5 * b.abs(), "{i}: {a} != {b}");
    }
}

#[test]
fn forty_layers_python_legacy_trace_and_three_commands() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("../toy_expected.json")).unwrap();
    let mut legacy = model.new_cpu_state(5).unwrap();
    let mut traced = model.new_state_with_gpu_step(5).unwrap();
    let mut plain = model.new_state_with_gpu_step(5).unwrap();
    for (position, token) in [0, 3, 1, 4, 2].into_iter().enumerate() {
        let mut reference = Vec::new();
        let legacy_logits = model
            .step_with_trace(
                &mut legacy,
                token,
                None,
                Some(&mut |t: LayerTrace<'_>| {
                    reference.push((t.hidden.to_vec(), t.next_pre_mix.to_vec()));
                    Ok(())
                }),
            )
            .unwrap();
        let mut count = 0;
        let logits = model
            .step_with_trace(
                &mut traced,
                token,
                None,
                Some(&mut |t: LayerTrace<'_>| {
                    assert_eq!((t.position, t.layer), (position, count));
                    close(t.hidden, &reference[count].0);
                    close(t.next_pre_mix, &reference[count].1);
                    if position < 3 {
                        let row = &expected["traces"][position * 40 + count];
                        for stream in 0..4 {
                            close(
                                &t.hidden[stream * 32..(stream + 1) * 32],
                                &vec![row[stream].as_f64().unwrap() as f32; 32],
                            );
                            close(
                                &t.next_pre_mix[stream..stream + 1],
                                &[row[4 + stream].as_f64().unwrap() as f32],
                            );
                        }
                    }
                    count += 1;
                    Ok(())
                }),
            )
            .unwrap();
        assert_eq!(count, 40);
        close(&logits, &legacy_logits);
        if position < 3 {
            close(
                &logits,
                &expected["logits"][position]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>(),
            );
        }
        let scope = profile::Scope::begin().unwrap();
        let untraced = model.step(&mut plain, token, None).unwrap();
        let report = scope.finish();
        assert_eq!(
            report
                .operations
                .iter()
                .map(|o| o.completed_submits)
                .sum::<usize>(),
            3
        );
        assert!(report.operations.iter().all(|o| o.operation == "gpu_step"));
        assert_eq!(logits, untraced);
        let ws = plain.gpu_step.as_ref().unwrap();
        assert_eq!((ws.commands, ws.trace_copies), (3, 0));
        let ws = traced.gpu_step.as_ref().unwrap();
        assert_eq!((ws.commands, ws.trace_copies), (3, 80));
        assert_eq!(plain.position(), position + 1);
        assert_eq!(plain.addresses, legacy.addresses);
        assert_eq!(
            format!("{:?}", plain.attention),
            format!("{:?}", attention::Session::default())
        );
    }
    assert!(
        model
            .step(&mut plain, 0, None)
            .unwrap_err()
            .contains("maximum context")
    );
    assert!(!plain.is_poisoned());
}

#[test]
fn earlier_route_failure_is_not_overwritten_by_later_layers() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let mut model = weights.model(&ctx);
    model.layers[3].bias[0] = f32::NAN;
    let mut state = model.new_state_with_gpu_step(1).unwrap();
    let error = model.step(&mut state, 0, None).unwrap_err();
    assert!(error.contains("GPU FFN layer 3 route status"), "{error}");
    let ws = state.gpu_step.as_ref().unwrap();
    assert_ne!(ws.ffn.status(3), metal::ROUTE_STATUS_READY);
    assert_eq!(ws.ffn.status(13), metal::ROUTE_STATUS_READY);
    assert_eq!(ws.commands, 2);
    assert!(state.is_poisoned());
    assert_eq!(state.position(), 0);
    assert!(
        model
            .step(&mut state, 0, None)
            .unwrap_err()
            .contains("poisoned")
    );
}

#[test]
fn cancellation_callback_errors_and_encode_failure_poison() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let mut model = weights.model(&ctx);
    let cancel = AtomicBool::new(true);
    let mut state = model.new_state_with_gpu_step(2).unwrap();
    assert!(
        model
            .step(&mut state, 0, Some(&cancel))
            .unwrap_err()
            .contains("cancelled")
    );
    assert!(!state.is_poisoned());
    assert!(model.step(&mut state, 8, None).is_err());
    assert!(!state.is_poisoned());
    for at in [0, 4, 14, 39] {
        cancel.store(false, Ordering::Relaxed);
        let mut state = model.new_state_with_gpu_step(2).unwrap();
        let error = model
            .step_with_trace(
                &mut state,
                0,
                Some(&cancel),
                Some(&mut |t: LayerTrace<'_>| {
                    if t.layer == at {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    Ok(())
                }),
            )
            .unwrap_err();
        assert!(error.contains("cancelled"));
        assert!(state.is_poisoned());
        assert_eq!(state.position(), 0);
    }
    let mut state = model.new_state_with_gpu_step(2).unwrap();
    assert_eq!(
        model
            .step_with_trace(
                &mut state,
                0,
                None,
                Some(&mut |_| Err("trace failure".into()))
            )
            .unwrap_err(),
        "trace failure"
    );
    assert!(state.is_poisoned());
    let mut state = model.new_state_with_gpu_step(2).unwrap();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = model.step_with_trace(&mut state, 0, None, Some(&mut |_| panic!("trace panic")));
    }));
    assert!(panicked.is_err());
    assert!(state.is_poisoned());
    let mut state = model.new_state_with_gpu_step(2).unwrap();
    model.layers[3].attention.head_dim += 1;
    assert!(
        model
            .step(&mut state, 0, None)
            .unwrap_err()
            .contains("GPU attention weights")
    );
    assert!(state.is_poisoned());
    assert_eq!(state.gpu_step.as_ref().unwrap().commands, 1);
}

#[test]
fn sticky_attention_status_is_checked_after_completion() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut state = model.new_state_with_gpu_step(1).unwrap();
    let status = state.gpu_step.as_ref().unwrap().attention.status();
    unsafe {
        *status.buffer.contents().as_ptr().cast::<i32>() = 4;
    }
    let error = model.step(&mut state, 0, None).unwrap_err();
    assert!(error.contains("GPU attention status 4"), "{error}");
    assert!(state.is_poisoned());
    assert_eq!(state.gpu_step.as_ref().unwrap().commands, 1);
}

#[test]
fn capacity_memory_and_modes_are_explicit() {
    let c = Config::default();
    let small = c.estimate_request_memory_with_gpu_step(1).unwrap();
    let large = c.estimate_request_memory_with_gpu_step(32768).unwrap();
    eprintln!("GPU step 32768-position request bound: {large:?}");
    assert!(large.total_bytes < 1024 * 1024 * 1024, "{large:?}");
    assert!(large.metal_bytes > small.metal_bytes);
    assert_eq!(large.cpu_bytes, small.cpu_bytes);
    assert_eq!(large.total_bytes, large.cpu_bytes + large.metal_bytes);
    assert!(c.estimate_request_memory_with_gpu_step(0).is_err());
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let state = model.new_state(1).unwrap();
    assert!(state.gpu_step.is_some());
}
