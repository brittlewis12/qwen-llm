use super::*;

#[test]
fn private_coarse_decode_restored_observer_is_bitwise_and_one_command() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    for use_engram in [false, true] {
        let mut model = weights.model(&ctx);
        if !use_engram {
            model.engram_hash.layers.clear();
            for layer in &mut model.layers {
                layer.engram = None;
            }
        }
        let engrams: Vec<_> = model.engram_hash.layers.iter().map(|l| l.layer).collect();
        let layout = profile::decode::layout(40, &engrams, false, true).unwrap();
        for depth in [0, 3, 6] {
            let mut state = model.new_state_with_gpu_step(40).unwrap();
            if depth > 0 {
                model
                    .prefill(
                        &mut state,
                        &(0..depth).map(|i| (i % 5) as u32).collect::<Vec<_>>(),
                        None,
                    )
                    .unwrap();
            }
            let snapshot = model.capture_snapshot(&state).unwrap();
            let mut reference = Vec::new();
            let mut reference_dispatches = Vec::new();
            let mut cached = None;
            for profiled in [false, true, false, true] {
                model.restore_snapshot(&mut state, &snapshot).unwrap();
                let scope = if profiled {
                    profile::Scope::begin_stages()
                } else {
                    profile::Scope::begin()
                }
                .unwrap();
                let mut rows = Vec::new();
                for step in 0..32 {
                    metal::dispatch_census_begin();
                    let logits = model
                        .step(&mut state, ((step * 3) % 5) as u32, None)
                        .unwrap();
                    let census = metal::dispatch_census_take();
                    assert!(!census.is_empty());
                    assert!(census.iter().all(|r| !r.encoder_concurrent));
                    let encoders: std::collections::BTreeSet<_> =
                        census.iter().map(|r| r.encoder_ordinal).collect();
                    assert_eq!(encoders.len(), if profiled { layout.len() } else { 1 });
                    let dispatches: Vec<_> = census
                        .into_iter()
                        .map(|r| {
                            (
                                r.kernel,
                                [r.grid_width, r.grid_height, r.grid_depth],
                                [r.threads_width, r.threads_height, r.threads_depth],
                                r.grid_tgs,
                                r.tg_threads,
                            )
                        })
                        .collect();
                    if reference_dispatches.len() == step {
                        reference_dispatches.push(dispatches);
                    } else {
                        assert_eq!(reference_dispatches[step], dispatches);
                    }
                    assert!(logits.iter().all(|v| v.is_finite()));
                    rows.push(logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    assert_eq!(state.diagnostic_gpu_counts(), Some((1, 0)));
                    assert_eq!(state.position(), depth + step + 1);
                }
                if reference.is_empty() {
                    reference = rows;
                } else {
                    assert_eq!(reference, rows);
                }
                let report = scope.finish();
                assert_eq!(
                    report.decode_intervals.len(),
                    if profiled { 32 * layout.len() } else { 0 }
                );
                let mut overlaps = 0;
                for intervals in report.decode_intervals.chunks_exact(layout.len()) {
                    let spans: Vec<_> = intervals.iter().map(|i| (i.start_ns, i.end_ns)).collect();
                    let summary = profile::decode::summarize_envelopes(&spans).unwrap();
                    overlaps += usize::from(!summary.exclusive);
                    assert_eq!(
                        summary.interval_union_ns + summary.overlap_multiplicity_ns,
                        summary.sum_inclusive_durations_ns
                    );
                    assert_eq!(
                        summary.interval_union_ns + summary.uncovered_sampled_envelope_ns,
                        summary.sampled_envelope_ns
                    );
                    assert!(summary.covered_by_two_or_more_ns <= summary.overlap_multiplicity_ns);
                }
                eprintln!(
                    "private fixture: engram={use_engram} depth={depth} profiled={profiled} nonexclusive_positions={overlaps}; valid inclusive envelopes, nonadditive and not proof of concurrent execution"
                );
                assert_eq!(
                    report.operations.len(),
                    32 * (1 + if profiled { layout.len() } else { 0 })
                );
                assert_eq!(
                    report
                        .operations
                        .iter()
                        .map(|o| o.completed_submits)
                        .sum::<usize>(),
                    32
                );
                assert!(
                    report
                        .operations
                        .iter()
                        .all(|o| o.invalid_gpu_timestamps == 0
                            && o.gpu_ms.is_some_and(|ms| ms > 0.0 && ms.is_finite()))
                );
                if profiled {
                    let samples = state
                        .gpu_step
                        .as_ref()
                        .unwrap()
                        .decode_samples
                        .as_ref()
                        .unwrap();
                    assert_eq!(samples.sample_count(), profile::decode::MAX_SAMPLES);
                    let address = std::rc::Rc::as_ptr(samples);
                    if let Some(previous) = cached {
                        assert_eq!(previous, address);
                    }
                    cached = Some(address);
                    for position in depth..depth + 32 {
                        for &(name, layer) in &layout {
                            let op = report
                                .operations
                                .iter()
                                .find(|o| {
                                    o.operation == name && o.dimensions == [position, layer, 1, 0]
                                })
                                .unwrap();
                            assert_eq!((op.count, op.completed_submits), (1, 0));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn private_coarse_decode_rejects_trace_and_clears_scope_on_failure() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut state = model.new_state_with_gpu_step(1).unwrap();
    let scope = profile::Scope::begin_stages().unwrap();
    let error = model
        .step_with_trace(&mut state, 0, None, Some(&mut |_| Ok(())))
        .unwrap_err();
    assert!(error.contains("forbids trace/capture"));
    assert_eq!(state.diagnostic_gpu_counts(), Some((0, 0)));
    assert!(scope.finish().operations.is_empty());
    profile::Scope::begin().unwrap().finish();
}

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
            assert_eq!(workspace.commands, 1);
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
fn forty_layers_python_legacy_trace_and_one_command() {
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
            1
        );
        assert!(report.operations.iter().all(|o| o.operation == "gpu_step"));
        assert_eq!(logits, untraced);
        let ws = plain.gpu_step.as_ref().unwrap();
        assert_eq!((ws.commands, ws.trace_copies), (1, 0));
        let ws = traced.gpu_step.as_ref().unwrap();
        assert_eq!((ws.commands, ws.trace_copies), (1, 80));
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
    assert_eq!(ws.commands, 1);
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
    assert_eq!(state.gpu_step.as_ref().unwrap().commands, 0);
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
