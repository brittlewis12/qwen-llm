use super::*;
use objc2_metal::MTLCommandBufferStatus;

fn force<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OVERRIDE.with(|x| x.set(self.0));
        }
    }
    let _restore = Restore(OVERRIDE.with(|x| x.replace(Some(enabled))));
    f()
}

#[derive(Default)]
struct ReadControl {
    fail: Option<usize>,
    panic: Option<usize>,
    cancel: Option<(usize, Arc<AtomicBool>)>,
    seen: Vec<usize>,
}
thread_local! {
    static READ: std::cell::RefCell<ReadControl> = Default::default();
}
struct ResetRead;
impl Drop for ResetRead {
    fn drop(&mut self) {
        READ.with(|r| *r.borrow_mut() = ReadControl::default());
    }
}
fn rows(layer: usize, ids: &[u64]) -> Result<Vec<f32>> {
    // A submitted GPU prefix must still be unjoined when its following
    // Engram read begins; this is an ordering contract, not a timing test.
    assert_eq!(SUBMITS.with(|s| s.get()), DRAINS.with(|d| d.get()) + 1);
    READ.with(|r| {
        let mut r = r.borrow_mut();
        r.seen.push(layer);
        assert_ne!(r.panic, Some(layer), "injected Engram panic");
        if r.fail == Some(layer) {
            return Err("injected Engram read failure".into());
        }
        if let Some((id, cancel)) = &r.cancel
            && *id == layer
        {
            cancel.store(true, Ordering::Relaxed);
        }
        Ok(ids
            .iter()
            .flat_map(|row| vec![((row + layer as u64) % 17) as f32 / 128.0 - 8.0 / 128.0; 256])
            .collect())
    })
}

#[test]
fn segmented_prefill_matches_one_command_and_decode_bitwise() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    for at_zero in [false, true] {
        let mut model = weights.model(&ctx);
        if at_zero {
            model.layers[0].engram = model.layers[1].engram.take();
            model.engram_hash.layers[0].layer = 0;
        }
        for (prefix, n) in [(0usize, 3usize), (5, 7), (0, 33)] {
            let mut old = model.new_state(prefix + n + 5).unwrap();
            let mut new = model.new_state(prefix + n + 5).unwrap();
            let tokens = (0..prefix + n + 5)
                .map(|i| (i % 8) as u32)
                .collect::<Vec<_>>();
            if prefix > 0 {
                force(false, || model.prefill(&mut old, &tokens[..prefix], None)).unwrap();
                force(false, || model.prefill(&mut new, &tokens[..prefix], None)).unwrap();
            }
            let a = force(false, || {
                model.prefill(&mut old, &tokens[prefix..prefix + n], None)
            })
            .unwrap();
            let b = force(true, || {
                model.prefill(&mut new, &tokens[prefix..prefix + n], None)
            })
            .unwrap();
            assert_eq!(a, b);
            assert_eq!(
                new.gpu_step.as_ref().unwrap().commands,
                if at_zero { 2 } else { 3 }
            );
            assert_eq!(old.gpu_step.as_ref().unwrap().commands, 1);
            assert_eq!(old.addresses, new.addresses);
            for &t in &tokens[prefix + n..] {
                assert_eq!(
                    model.step(&mut old, t, None).unwrap(),
                    model.step(&mut new, t, None).unwrap()
                );
            }
        }
    }
}

#[test]
fn read_failure_after_each_prefix_drains_and_poisons() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    for (layer, commands) in [(1usize, 1usize), (14, 2)] {
        let _reset = ResetRead;
        let mut model = weights.model(&ctx);
        model.engram_rows = EngramRows::Toy(rows);
        let mut state = model.new_state(4).unwrap();
        let before = state.addresses;
        READ.with(|r| {
            *r.borrow_mut() = ReadControl {
                fail: Some(layer),
                ..Default::default()
            }
        });
        DRAINS.with(|d| d.set(0));
        SUBMITS.with(|s| s.set(0));
        let error = force(true, || model.prefill(&mut state, &[0, 3, 1], None)).unwrap_err();
        assert!(error.to_string().contains("injected Engram"), "{error}");
        assert!(state.is_poisoned());
        assert_eq!(state.addresses, before);
        assert_eq!(state.position(), 0);
        assert_eq!(state.gpu_step.as_ref().unwrap().commands, commands);
        assert_eq!(DRAINS.with(|d| d.get()), commands);
        assert!(model.step(&mut state, 0, None).is_err());
    }
}

#[test]
fn cancellation_during_each_read_drains_and_preserves_position() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    for (layer, commands) in [(1usize, 1usize), (14, 2)] {
        let _reset = ResetRead;
        let mut model = weights.model(&ctx);
        model.engram_rows = EngramRows::Toy(rows);
        let mut state = model.new_state(4).unwrap();
        let before = state.addresses;
        let cancelled = Arc::new(AtomicBool::new(false));
        READ.with(|r| {
            *r.borrow_mut() = ReadControl {
                cancel: Some((layer, cancelled.clone())),
                ..Default::default()
            }
        });
        DRAINS.with(|d| d.set(0));
        SUBMITS.with(|s| s.set(0));
        let error = force(true, || {
            model.prefill(&mut state, &[0, 3, 1], Some(&cancelled))
        })
        .unwrap_err();
        assert!(error.to_string().contains("cancel"), "{error}");
        assert!(state.is_poisoned());
        assert_eq!(state.addresses, before);
        assert_eq!(state.position(), 0);
        assert_eq!(state.gpu_step.as_ref().unwrap().commands, commands);
        assert_eq!(DRAINS.with(|d| d.get()), commands);
    }
}

#[test]
fn failed_prefix_completion_is_drained_and_never_commits_position() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut state = model.new_state(4).unwrap();
    let before = state.addresses;
    DRAINS.with(|d| d.set(0));
    SUBMITS.with(|s| s.set(0));
    let _fault = metal::fail_next_completion_check();
    let error = force(true, || model.prefill(&mut state, &[0, 3, 1], None)).unwrap_err();
    assert!(error.to_string().contains("injected discarded"), "{error}");
    assert_eq!(DRAINS.with(|d| d.get()), 1);
    assert_eq!(state.gpu_step.as_ref().unwrap().commands, 1);
    assert!(state.is_poisoned());
    assert_eq!(state.addresses, before);
}

#[test]
fn submitted_drop_waits_for_the_copy_before_returning() {
    let ctx = MetalContext::new().unwrap();
    let source = upload(&ctx, &[1.25, 2.5, 3.75, 5.0], vec![4]).unwrap();
    let target = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    metal::encode_copy_offset_f32(&ctx, &enc, &source, 0, &target, 4).unwrap();
    enc.end();
    DRAINS.with(|d| d.set(0));
    SUBMITS.with(|s| s.set(0));
    {
        let _pending = Submitted::commit(command.clone(), [0, 1, 4, 0], None);
        // Simulate the caller abandoning an IO/encode path before joining.
    }
    assert_eq!(DRAINS.with(|d| d.get()), 1);
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert_eq!(f32_slice(&target).unwrap(), [1.25, 2.5, 3.75, 5.0]);
}

#[test]
fn panic_during_read_drains_before_unwinding_to_caller() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let mut model = weights.model(&ctx);
    model.engram_rows = EngramRows::Toy(rows);
    let mut state = model.new_state(4).unwrap();
    let before = state.addresses;
    let _reset = ResetRead;
    READ.with(|r| {
        *r.borrow_mut() = ReadControl {
            panic: Some(14),
            ..Default::default()
        }
    });
    DRAINS.with(|d| d.set(0));
    SUBMITS.with(|s| s.set(0));
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        force(true, || model.prefill(&mut state, &[0, 3, 1], None))
    }));
    assert!(error.is_err());
    assert_eq!(DRAINS.with(|d| d.get()), 2);
    assert_eq!(state.gpu_step.as_ref().unwrap().commands, 2);
    assert!(state.is_poisoned());
    assert_eq!(state.addresses, before);
}

#[test]
fn stage_sampling_across_segments_keeps_results_and_valid_timestamps() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut old = model.new_state(4).unwrap();
    let mut new = model.new_state(4).unwrap();
    let baseline = profile::Scope::begin_stages().unwrap();
    let expected = force(false, || model.prefill(&mut old, &[0, 3, 1], None)).unwrap();
    let baseline = baseline.finish();
    let scope = profile::Scope::begin_stages().unwrap();
    let actual = force(true, || model.prefill(&mut new, &[0, 3, 1], None)).unwrap();
    let report = scope.finish();
    assert_eq!(actual, expected);
    let submits: Vec<_> = report
        .operations
        .iter()
        .filter(|x| x.operation == "prefill_chunk_rows")
        .collect();
    assert_eq!(
        submits.iter().map(|x| x.completed_submits).sum::<usize>(),
        3
    );
    // Toy schedules have empty sample bins (unneeded index/mid passes).
    // Require valid, nonempty HC samples at each command boundary and valid
    // whole-command clocks, rather than treating empty bins as GPU work.
    for measured in [&baseline, &report] {
        for layer in [0, 1, 14] {
            let op = measured
                .operations
                .iter()
                .find(|x| x.operation == "prefill_attention_hc" && x.dimensions[1] == layer)
                .unwrap();
            assert_eq!(op.invalid_gpu_timestamps, 0, "{layer}: {op:?}");
            assert!(op.gpu_ms.is_some_and(|x| x > 0.0), "{layer}: {op:?}");
        }
        assert!(
            measured
                .operations
                .iter()
                .filter(|x| x.operation == "prefill_chunk_rows")
                .all(|x| x.invalid_gpu_timestamps == 0 && x.gpu_ms.is_some_and(|v| v > 0.0))
        );
    }
    assert_eq!(
        report
            .operations
            .iter()
            .filter(|x| x.operation == "prefill_attention_serial")
            .map(|x| x.count)
            .sum::<usize>(),
        40
    );
}
