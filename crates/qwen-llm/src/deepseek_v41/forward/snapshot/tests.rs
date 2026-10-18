use super::*;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn wrapped_odd_even_restore_and_growth_are_bitwise() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    for position in [0, 3, 4, 5, 6] {
        let mut source = model.new_state(12).unwrap();
        let mut grown = model.new_state(position.max(1)).unwrap();
        for i in 0..position {
            let token = (i % 5) as u32;
            assert_eq!(
                bits(&model.step(&mut source, token, None).unwrap()),
                bits(&model.step(&mut grown, token, None).unwrap())
            );
        }
        let snapshot = model.capture_snapshot(&source).unwrap();
        assert_eq!(
            model.snapshot_payload_bytes(&source).unwrap(),
            snapshot.payload_bytes()
        );
        assert_eq!(snapshot.position(), position);
        assert_eq!(snapshot.payload_bytes(), snapshot.payload.len() * 4);
        let mut restored = model.new_state(12).unwrap();
        // Recycle a destination whose physical rings and completed rows are dirty.
        for _ in 0..8 {
            model.step(&mut restored, 4, None).unwrap();
        }
        let status = restored.gpu_step.as_ref().unwrap().attention().status();
        unsafe {
            status
                .buffer
                .contents()
                .as_ptr()
                .cast::<i32>()
                .add((status.offset / 4) as usize)
                .write(1);
        }
        model.restore_snapshot(&mut restored, &snapshot).unwrap();
        let status = restored.gpu_step.as_ref().unwrap().attention().status();
        assert_eq!(
            unsafe {
                status
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<i32>()
                    .add((status.offset / 4) as usize)
                    .read()
            },
            0
        );
        let old_capacity = grown.max_context;
        model.ensure_capacity(&mut grown, old_capacity + 1).unwrap();
        assert_eq!(grown.max_context, old_capacity * 2);
        model.ensure_capacity(&mut grown, 12).unwrap();
        assert_eq!(
            bits(&snapshot.payload),
            bits(&model.capture_snapshot(&restored).unwrap().payload)
        );
        assert_eq!(
            bits(&snapshot.payload),
            bits(&model.capture_snapshot(&grown).unwrap().payload)
        );
        for token in [2, 0, 4] {
            let expected = bits(&model.step(&mut source, token, None).unwrap());
            assert_eq!(
                expected,
                bits(&model.step(&mut restored, token, None).unwrap())
            );
            assert_eq!(
                expected,
                bits(&model.step(&mut grown, token, None).unwrap())
            );
        }
    }
}

#[test]
fn rejection_does_not_mutate_healthy_destination() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let other = weights.model(&ctx);
    let mut state = model.new_state(2).unwrap();
    model.step(&mut state, 1, None).unwrap();
    let snapshot = model.capture_snapshot(&state).unwrap();
    let foreign = other
        .capture_snapshot(&other.new_state(2).unwrap())
        .unwrap();
    assert!(model.restore_snapshot(&mut state, &foreign).is_err());
    assert!(other.capture_snapshot(&state).is_err());
    assert!(other.snapshot_payload_bytes(&state).is_err());
    assert!(model.ensure_capacity(&mut state, usize::MAX).is_err());
    assert!(!state.is_poisoned());
    assert_eq!(
        bits(&snapshot.payload),
        bits(&model.capture_snapshot(&state).unwrap().payload)
    );
    let mut malformed = model.capture_snapshot(&state).unwrap();
    malformed.payload.pop();
    assert!(model.restore_snapshot(&mut state, &malformed).is_err());
    assert!(!state.is_poisoned());
    let mut too_long = model.new_state(3).unwrap();
    for _ in 0..3 {
        model.step(&mut too_long, 2, None).unwrap();
    }
    let too_long = model.capture_snapshot(&too_long).unwrap();
    assert!(model.restore_snapshot(&mut state, &too_long).is_err());
    assert!(!state.is_poisoned());
    assert_eq!(
        bits(&snapshot.payload),
        bits(&model.capture_snapshot(&state).unwrap().payload)
    );
    state.poisoned = true;
    assert!(model.snapshot_payload_bytes(&state).is_err());
    assert!(model.capture_snapshot(&state).is_err());
    assert!(model.restore_snapshot(&mut state, &snapshot).is_err());
    assert!(model.ensure_capacity(&mut state, 4).is_err());
}

fn admission_denial() -> StateError {
    let signals = metal::MetalMemorySignals {
        recommended_max_bytes: 100,
        current_allocated_bytes: 90,
        process_limit_remaining_bytes: Some(5),
    };
    StateError::MemoryAdmissionDenied(
        metal::evaluate_metal_memory_admission_with_cpu_bytes(20, 10, 0, signals, true)
            .refusal()
            .unwrap(),
    )
}

#[test]
fn capacity_fallback_retries_only_admission_denial() {
    let mut attempts = Vec::new();
    let capacity = allocate_with_capacity_fallback(16, 9, |n| {
        attempts.push(n);
        if n == 16 {
            Err(admission_denial())
        } else {
            Ok(n)
        }
    })
    .unwrap();
    assert_eq!(attempts, [16, 9]);
    assert_eq!(capacity, 9);

    attempts.clear();
    assert_eq!(
        allocate_with_capacity_fallback(16, 9, |n| {
            attempts.push(n);
            Ok(n)
        })
        .unwrap(),
        16
    );
    assert_eq!(attempts, [16]);

    attempts.clear();
    let error = allocate_with_capacity_fallback::<()>(16, 9, |n| {
        attempts.push(n);
        Err(StateError::Invalid("allocation error".into()))
    })
    .unwrap_err();
    assert!(matches!(error, StateError::Invalid(_)));
    assert_eq!(attempts, [16]);

    for preferred in [9, 16] {
        attempts.clear();
        let error = allocate_with_capacity_fallback::<()>(preferred, 9, |n| {
            attempts.push(n);
            Err(admission_denial())
        })
        .unwrap_err();
        assert!(error.memory_admission().is_some());
        assert_eq!(attempts, if preferred == 9 { vec![9] } else { vec![16, 9] });
    }
}
