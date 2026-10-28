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
        // Host Engram row reads are a wall span, not a command.
        assert!(
            report
                .operations
                .iter()
                .all(|o| ["gpu_step", "decode_engram_rows"].contains(&o.operation))
        );
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
fn dispatch_profile_times_every_decode_dispatch_without_changing_logits() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut plain = model.new_state_with_gpu_step(4).unwrap();
    let mut profiled = model.new_state_with_gpu_step(4).unwrap();
    for token in [0, 3, 1] {
        let expected = model.step(&mut plain, token, None).unwrap();
        metal::dispatch_census_begin();
        let scope = profile::Scope::begin_dispatches().unwrap();
        let logits = model.step(&mut profiled, token, None).unwrap();
        let report = scope.finish();
        let census = metal::dispatch_census_take();
        // Same kernels in the same order; only encoder boundaries differ.
        assert_eq!(logits, expected);
        assert_eq!(report.dispatch_overflow, 0);
        assert_eq!(
            report
                .dispatches
                .iter()
                .map(|r| r.kernel.as_str())
                .collect::<Vec<_>>(),
            census.iter().map(|r| r.kernel.as_str()).collect::<Vec<_>>()
        );
        assert!(report.dispatches.iter().all(|r| r.gpu_ns.is_some()));
        let labels: std::collections::BTreeSet<_> =
            report.dispatches.iter().map(|r| r.label).collect();
        for label in [
            "hc_pre",
            "attn_qkv",
            "attn_core",
            "attn_output",
            "hc_post",
            "ffn_hc_pre",
            "ffn_router",
            "ffn_shared",
            "ffn_routed",
            "ffn_post",
            "head",
        ] {
            assert!(labels.contains(label), "{label} missing from {labels:?}");
        }
    }
    // The profile is scoped to the step: later steps run unsplit.
    let scope = profile::Scope::begin().unwrap();
    model.step(&mut profiled, 2, None).unwrap();
    assert!(scope.finish().dispatches.is_empty());
}

#[test]
fn wide_hc_pre_matches_two_dispatch_pre_within_reduction_order() {
    use super::super::hc_gpu::HC_WIDE_OVERRIDE;
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let layer = &model.layers[0];
    let ws = model.new_state_with_gpu_step(1).unwrap();
    let gpu = ws.gpu_step.as_ref().unwrap();
    HC_WIDE_OVERRIDE.with(|o| o.set(Some(true)));
    let eligible = super::super::hc_gpu::wide(
        &layer.attn_hc,
        model.config.hidden,
        &gpu.x,
        &gpu.collapsed,
        &gpu.normalized,
        &gpu.constants[0][2],
    );
    HC_WIDE_OVERRIDE.with(|o| o.set(None));
    assert!(eligible, "the toy model must exercise the wide pre-block");
    use super::super::ffn_gpu::FOLDED_COMBINE_OVERRIDE;
    let set = |wide: Option<bool>, folded: Option<bool>| {
        HC_WIDE_OVERRIDE.with(|o| o.set(wide));
        FOLDED_COMBINE_OVERRIDE.with(|o| o.set(folded));
    };
    let mut two = model.new_state_with_gpu_step(4).unwrap();
    let mut wide = model.new_state_with_gpu_step(4).unwrap();
    let mut folded = model.new_state_with_gpu_step(4).unwrap();
    for token in [0, 3, 1, 2] {
        set(Some(false), Some(false));
        let expected = model.step(&mut two, token, None);
        set(Some(true), Some(false));
        let wide_logits = model.step(&mut wide, token, None);
        set(Some(true), Some(true));
        let folded_logits = model.step(&mut folded, token, None);
        set(None, None);
        let expected = expected.unwrap();
        let scale = expected.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for (name, logits) in [("wide", wide_logits), ("wide+folded", folded_logits)] {
            let logits = logits.unwrap();
            assert_eq!(logits.len(), expected.len());
            for (index, (got, want)) in logits.iter().zip(&expected).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-4 * scale,
                    "token {token} logit {index}: {name} {got} vs two-dispatch {want}"
                );
            }
        }
    }
}

#[test]
fn fused_rank_norms_match_split_qkv_chain_bitwise() {
    use super::super::attention_gpu::FUSED_RANK_NORMS_OVERRIDE;
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    assert!(
        model.config.rope_dim > 0,
        "the toy model must rotate the KV tail"
    );
    let mut split = model.new_state_with_gpu_step(4).unwrap();
    let mut fused = model.new_state_with_gpu_step(4).unwrap();
    for token in [0, 3, 1, 2] {
        FUSED_RANK_NORMS_OVERRIDE.with(|o| o.set(Some(false)));
        let expected = model.step(&mut split, token, None);
        FUSED_RANK_NORMS_OVERRIDE.with(|o| o.set(Some(true)));
        let logits = model.step(&mut fused, token, None);
        FUSED_RANK_NORMS_OVERRIDE.with(|o| o.set(None));
        assert_eq!(logits.unwrap(), expected.unwrap(), "token {token}");
    }
}

#[test]
fn split_submission_matches_one_command_bitwise() {
    // Lead (layers before the first Engram layer), middle (Engram
    // projections + the next layers), rest + head: three commands.
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut one = model.new_state_with_gpu_step(4).unwrap();
    let mut split = model.new_state_with_gpu_step(4).unwrap();
    for token in [0, 3, 1, 2] {
        SPLIT_SUBMIT_OVERRIDE.with(|o| o.set(Some(false)));
        let expected = model.step(&mut one, token, None);
        SPLIT_SUBMIT_OVERRIDE.with(|o| o.set(Some(true)));
        let logits = model.step(&mut split, token, None);
        SPLIT_SUBMIT_OVERRIDE.with(|o| o.set(None));
        assert_eq!(logits.unwrap(), expected.unwrap(), "token {token}");
        assert_eq!(one.gpu_step.as_ref().unwrap().commands, 1);
        assert_eq!(split.gpu_step.as_ref().unwrap().commands, 3);
    }
}

#[test]
fn split_ffn_encoders_match_one_encoder_bitwise() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut one = model.new_state_with_gpu_step(4).unwrap();
    let mut split = model.new_state_with_gpu_step(4).unwrap();
    for token in [0, 3, 1, 2] {
        FFN_SPLIT_OVERRIDE.with(|o| o.set(Some(false)));
        let expected = model.step(&mut one, token, None).unwrap();
        FFN_SPLIT_OVERRIDE.with(|o| o.set(Some(true)));
        let logits = model.step(&mut split, token, None);
        FFN_SPLIT_OVERRIDE.with(|o| o.set(None));
        assert_eq!(logits.unwrap(), expected);
    }
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

/// Batch-level levers of one `batched_steps_case`.
#[derive(Clone, Copy, Debug)]
struct BatchLevers {
    shared: bool,
    attention: bool,
    qkv: bool,
    output_a: bool,
    head: bool,
    ffn_split: bool,
    per_column_copy: bool,
    fused_rank_norms: bool,
}

impl BatchLevers {
    /// Set (or, with `None`, clear) every lever's thread-local override.
    fn apply(levers: Option<Self>) {
        use super::super::attention_gpu::FUSED_RANK_NORMS_OVERRIDE;
        use super::super::columns::PER_COLUMN_COPY_OVERRIDE;
        use batch::{
            BATCH_ATTN_OVERRIDE, BATCH_HEAD_OVERRIDE, BATCH_OUTPUT_A_OVERRIDE, BATCH_QKV_OVERRIDE,
            BATCH_SHARED_OVERRIDE,
        };
        let l = levers;
        BATCH_SHARED_OVERRIDE.with(|o| o.set(l.map(|l| l.shared)));
        BATCH_ATTN_OVERRIDE.with(|o| o.set(l.map(|l| l.attention)));
        BATCH_QKV_OVERRIDE.with(|o| o.set(l.map(|l| l.qkv)));
        BATCH_OUTPUT_A_OVERRIDE.with(|o| o.set(l.map(|l| l.output_a)));
        BATCH_HEAD_OVERRIDE.with(|o| o.set(l.map(|l| l.head)));
        FFN_SPLIT_OVERRIDE.with(|o| o.set(l.map(|l| l.ffn_split)));
        PER_COLUMN_COPY_OVERRIDE.with(|o| o.set(l.map(|l| l.per_column_copy)));
        FUSED_RANK_NORMS_OVERRIDE.with(|o| o.set(l.map(|l| l.fused_rank_norms)));
    }

    /// Column gathers and scatters of one step over `layers` layers: per
    /// layer one each for the shared expert, two each for q_b and output_b
    /// (or output_a and output_b), three for the rank pair (one gather, two
    /// scatters); one each for the head.
    fn transfers(self, layers: usize) -> usize {
        let staged = self.attention && self.fused_rank_norms;
        let per_layer =
            2 * self.shared as usize + 4 * staged as usize + 3 * (staged && self.qkv) as usize;
        per_layer * layers + 2 * self.head as usize
    }
}

/// The toy's F32 weights take every batched projection's bitwise column
/// fallback, so batched steps equal separate steps bitwise with the batched
/// shared expert, the staged attention (q_b, output_b), the batched rank
/// pair, output_a and head each on or off, one or split FFN encoders,
/// one-dispatch or per-column column copies, and the unfused query/KV chain
/// (which keeps per-sequence attention). The column copies are counted, so
/// the batched paths ran.
#[test]
fn batched_steps_match_separate_steps_bitwise() {
    let on = BatchLevers {
        shared: true,
        attention: true,
        qkv: true,
        output_a: true,
        head: true,
        ffn_split: true,
        per_column_copy: false,
        fused_rank_norms: true,
    };
    for levers in [
        on,
        BatchLevers {
            ffn_split: false,
            ..on
        },
        BatchLevers {
            per_column_copy: true,
            ..on
        },
        BatchLevers {
            shared: false,
            ..on
        },
        BatchLevers {
            attention: false,
            ..on
        },
        BatchLevers {
            shared: false,
            attention: false,
            ..on
        },
        BatchLevers {
            fused_rank_norms: false,
            ..on
        },
        BatchLevers { qkv: false, ..on },
        BatchLevers {
            output_a: false,
            ..on
        },
        BatchLevers { head: false, ..on },
        BatchLevers {
            shared: false,
            attention: false,
            qkv: false,
            output_a: false,
            head: false,
            ..on
        },
    ] {
        BatchLevers::apply(Some(levers));
        batched_steps_case(levers);
        BatchLevers::apply(None);
    }
}

fn batched_steps_case(levers: BatchLevers) {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let count = |census: &[metal::DispatchCensusRow], kernels: &[&str]| {
        census
            .iter()
            .filter(|r| kernels.contains(&r.kernel.as_str()))
            .count()
    };
    let copy = ["kernel_copy_offset_f32"];
    let columns = [
        "kernel_deepseek_v41_gather_columns",
        "kernel_deepseek_v41_scatter_columns",
    ];
    for split in [false, true] {
        SPLIT_SUBMIT_OVERRIDE.with(|o| o.set(Some(split)));
        let new = || {
            (0..3)
                .map(|_| model.new_state_with_gpu_step(6).unwrap())
                .collect::<Vec<_>>()
        };
        let (mut alone, mut batched) = (new(), new());
        // Different depths: the last sequence starts two tokens ahead.
        for token in [4, 2] {
            model.step(&mut alone[2], token, None).unwrap();
            model.step(&mut batched[2], token, None).unwrap();
        }
        for tokens in [[0, 3, 1], [3, 3, 2], [1, 0, 4]] {
            metal::dispatch_census_begin();
            let expected: Vec<_> = alone
                .iter_mut()
                .zip(tokens)
                .map(|(state, token)| model.step(state, token, None).unwrap())
                .collect();
            let serial = metal::dispatch_census_take();
            let mut states: Vec<&mut State> = batched.iter_mut().collect();
            metal::dispatch_census_begin();
            let logits = model.step_batch(&mut states, &tokens, None).unwrap();
            let batch = metal::dispatch_census_take();
            let transfers = levers.transfers(model.layers.len());
            let (copies, column_copies) = if levers.per_column_copy {
                (3 * transfers, 0)
            } else {
                (0, transfers)
            };
            assert_eq!(
                count(&batch, &copy),
                count(&serial, &copy) + copies,
                "{levers:?}"
            );
            assert_eq!(count(&batch, &columns), column_copies, "{levers:?}");
            assert_eq!(
                logits, expected,
                "{levers:?} split {split} tokens {tokens:?}"
            );
            for (alone, batched) in alone.iter().zip(&batched) {
                assert!(batched.is_ready());
                assert_eq!(batched.position(), alone.position());
                assert_eq!(batched.addresses, alone.addresses);
                let commands = batched.gpu_step.as_ref().unwrap().commands;
                assert_eq!(commands, if split { 3 } else { 1 });
            }
        }
        SPLIT_SUBMIT_OVERRIDE.with(|o| o.set(None));
    }
}

/// The staged decode API refuses out-of-order stages before encoding, and a
/// pending staged step blocks `encode`; the cursors stay put until finish.
#[test]
fn staged_attention_decode_refuses_out_of_order_stages() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut state = model.new_state_with_gpu_step(2).unwrap();
    let ws = state.gpu_step.as_mut().unwrap();
    let w = &model.layers[0].attention;
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let input = ws.normalized.clone();
    let attention = &mut ws.attention;
    assert!(attention.staged_decode_supported());
    let order = |e: String| assert!(e.contains("staged decode order"), "{e}");
    order(
        attention
            .encode_middle(&ctx, &enc, 0, 0, w, &input)
            .unwrap_err(),
    );
    order(
        attention
            .encode_finish(&ctx, &enc, 0, 0)
            .map(|_| ())
            .unwrap_err(),
    );
    // Out of schedule order: refused by the front's checks.
    let error = attention
        .encode_front(&ctx, &enc, 1, 0, w, &input)
        .unwrap_err();
    assert!(error.contains("order/capacity"), "{error}");
    attention.encode_front(&ctx, &enc, 0, 0, w, &input).unwrap();
    assert!(!attention.staged_decode_supported());
    let error = attention
        .encode(&ctx, &enc, 0, 0, w, &input)
        .map(|_| ())
        .unwrap_err();
    assert!(error.contains("staged"), "{error}");
    order(
        attention
            .encode_finish(&ctx, &enc, 0, 0)
            .map(|_| ())
            .unwrap_err(),
    );
    order(
        attention
            .encode_middle(&ctx, &enc, 0, 1, w, &input)
            .unwrap_err(),
    );
    attention
        .encode_middle(&ctx, &enc, 0, 0, w, &input)
        .unwrap();
    order(
        attention
            .encode_middle(&ctx, &enc, 0, 0, w, &input)
            .unwrap_err(),
    );
    // The front for the next layer is refused until this step finishes.
    let error = attention
        .encode_front(&ctx, &enc, 1, 0, w, &input)
        .unwrap_err();
    assert!(error.contains("staged"), "{error}");
    // output_a belongs after the middle; the finish still follows.
    attention.encode_output_a(&ctx, &enc, 0, 0, w).unwrap();
    attention.encode_finish(&ctx, &enc, 0, 0).unwrap();
    assert!(attention.staged_decode_supported());
    order(attention.encode_output_a(&ctx, &enc, 1, 0, w).unwrap_err());
    // The checks-only front: rank norms are due before the middle, and only
    // once; nothing else runs until then.
    order(
        attention
            .encode_rank_norms(&ctx, &enc, 1, 0, w)
            .unwrap_err(),
    );
    attention.encode_checks(&enc, 1, 0, w, &input).unwrap();
    order(
        attention
            .encode_middle(&ctx, &enc, 1, 0, w, &input)
            .unwrap_err(),
    );
    order(attention.encode_output_a(&ctx, &enc, 1, 0, w).unwrap_err());
    let error = attention.encode_checks(&enc, 1, 0, w, &input).unwrap_err();
    assert!(error.contains("staged"), "{error}");
    attention.encode_rank_norms(&ctx, &enc, 1, 0, w).unwrap();
    order(
        attention
            .encode_rank_norms(&ctx, &enc, 1, 0, w)
            .unwrap_err(),
    );
    attention
        .encode_middle(&ctx, &enc, 1, 0, w, &input)
        .unwrap();
    attention.encode_finish(&ctx, &enc, 1, 0).unwrap();
    // Out of schedule order: the checks refuse layer 1 again.
    let error = attention.encode_checks(&enc, 1, 0, w, &input).unwrap_err();
    assert!(error.contains("order/capacity"), "{error}");
    enc.end();
}

#[test]
fn refused_batches_keep_states_and_failed_batches_poison_all() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let mut model = weights.model(&ctx);
    let mut a = model.new_state_with_gpu_step(1).unwrap();
    let mut b = model.new_state_with_gpu_step(3).unwrap();
    assert!(model.step_batch(&mut [], &[], None).is_err());
    let error = model
        .step_batch(&mut [&mut a, &mut b], &[0], None)
        .unwrap_err();
    assert!(error.contains("one token per state"), "{error}");
    let error = model
        .step_batch(&mut [&mut a, &mut b], &[0, 8], None)
        .unwrap_err();
    assert!(error.contains("vocabulary"), "{error}");
    let cancel = AtomicBool::new(true);
    let error = model
        .step_batch(&mut [&mut a, &mut b], &[0, 1], Some(&cancel))
        .unwrap_err();
    assert!(error.contains("cancelled"), "{error}");
    assert!(a.is_ready() && b.is_ready());
    assert_eq!((a.position(), b.position()), (0, 0));
    model
        .step_batch(&mut [&mut a, &mut b], &[0, 1], None)
        .unwrap();
    // `a` is full: the batch is refused before `b` is touched.
    let error = model
        .step_batch(&mut [&mut b, &mut a], &[0, 1], None)
        .unwrap_err();
    assert!(error.contains("maximum context"), "{error}");
    assert!(a.is_ready() && b.is_ready());
    assert_eq!((a.position(), b.position()), (1, 1));
    // A device failure in one sequence poisons the whole batch.
    let mut c = model.new_state_with_gpu_step(2).unwrap();
    let status = b.gpu_step.as_ref().unwrap().attention.status();
    unsafe {
        *status.buffer.contents().as_ptr().cast::<i32>() = 4;
    }
    let error = model
        .step_batch(&mut [&mut c, &mut b], &[0, 1], None)
        .unwrap_err();
    assert!(
        error.contains("sequence 1: GPU attention status 4"),
        "{error}"
    );
    assert!(b.is_poisoned() && c.is_poisoned());
    assert_eq!((b.position(), c.position()), (1, 0));
    // A poisoned member is refused without touching the others.
    let mut d = model.new_state_with_gpu_step(2).unwrap();
    let error = model
        .step_batch(&mut [&mut d, &mut b], &[0, 0], None)
        .unwrap_err();
    assert!(error.contains("poisoned"), "{error}");
    assert!(d.is_ready());
    // An encode failure poisons every member and commits nothing.
    let mut e = model.new_state_with_gpu_step(2).unwrap();
    model.layers[3].attention.head_dim += 1;
    let error = model
        .step_batch(&mut [&mut d, &mut e], &[0, 1], None)
        .unwrap_err();
    assert!(error.contains("GPU attention weights"), "{error}");
    assert!(d.is_poisoned() && e.is_poisoned());
    assert_eq!(d.gpu_step.as_ref().unwrap().commands, 0);
}

/// One sequence never takes the batched shared expert or the staged
/// attention: `step_batch` is `step`, dispatch for dispatch.
#[test]
fn single_sequence_batch_is_step_with_batched_shared_on() {
    use batch::{BATCH_ATTN_OVERRIDE, BATCH_SHARED_OVERRIDE};
    BATCH_SHARED_OVERRIDE.with(|o| o.set(Some(true)));
    BATCH_ATTN_OVERRIDE.with(|o| o.set(Some(true)));
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut alone = model.new_state_with_gpu_step(3).unwrap();
    let mut batched = model.new_state_with_gpu_step(3).unwrap();
    for token in [0, 3, 1] {
        metal::dispatch_census_begin();
        let expected = model.step(&mut alone, token, None).unwrap();
        let step: Vec<_> = metal::dispatch_census_take()
            .into_iter()
            .map(|r| r.kernel)
            .collect();
        metal::dispatch_census_begin();
        let logits = model
            .step_batch(&mut [&mut batched], &[token], None)
            .unwrap();
        let batch: Vec<_> = metal::dispatch_census_take()
            .into_iter()
            .map(|r| r.kernel)
            .collect();
        assert_eq!(logits, vec![expected]);
        assert_eq!(batch, step);
    }
    BATCH_SHARED_OVERRIDE.with(|o| o.set(None));
    BATCH_ATTN_OVERRIDE.with(|o| o.set(None));
}
