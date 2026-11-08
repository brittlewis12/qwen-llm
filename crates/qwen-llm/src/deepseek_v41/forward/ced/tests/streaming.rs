use super::*;

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn prefix_episodes_yield_provisional_then_match_full_and_restore_by_bits() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().unwrap();
    prefill::with_batched_f32(|| {
        let ctx = MetalContext::new().unwrap();
        let weights = toy::Weights::new(&ctx);
        let embedding = embeddings(&ctx);
        let mut model = nonuniform_model(&weights, &ctx, &embedding);
        model.prefill_rows = 4096;
        for (prefix, suffix, mixed) in [
            (0, 4097, false),
            (343, 4096, false),
            (343, 4097, false),
            (343, 5121, true),
        ] {
            let window = if mixed { 1025 } else { 3 };
            model.config.window = window;
            for layer in &mut model.layers {
                layer.attention.window = window;
            }
            let n = prefix + suffix;
            let tokens = (0..n).map(|i| (i % 8) as u32).collect::<Vec<_>>();
            let mut full = model.new_state(n + 32).unwrap();
            let mut ced = model.new_state(n + 32).unwrap();
            if prefix > 0 {
                model
                    .advance_prefill(&mut full, &tokens[..prefix], None)
                    .unwrap();
                let snapshot = model.capture_snapshot(&full).unwrap();
                model.restore_snapshot(&mut ced, &snapshot).unwrap();
            }
            let widths = if mixed {
                vec![4096, 512, 512, 512]
            } else {
                vec![4096; suffix.div_ceil(4096)]
            };
            let mut position = prefix;
            let mut expected = Vec::new();
            for &width in &widths {
                let end = (position + width).min(n);
                expected = model
                    .prefill(&mut full, &tokens[position..end], None)
                    .unwrap();
                position = end;
            }
            let mut width_index = 0;
            assert!(model.begin_cold_prefill_episode(&mut ced, suffix).unwrap());
            let actual = loop {
                let slice = model
                    .cold_prefill_episode_slice(
                        &mut ced,
                        &tokens[prefix..],
                        None,
                        widths[width_index],
                    )
                    .unwrap();
                width_index += 1;
                if slice.complete {
                    break slice.logits;
                }
                assert_eq!(ced.position(), prefix);
                assert!(!ced.is_ready());
                assert!(model.capture_snapshot(&ced).is_err());
                assert!(model.step(&mut ced, 1, None).is_err());
            };
            assert_eq!(
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(full.addresses, ced.addresses);
            for (a, b) in causal_bits(&full).iter().zip(causal_bits(&ced)) {
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            let snapshot = model.capture_snapshot(&ced).unwrap();
            let mut restored = model.new_state(n + 64).unwrap();
            model.step(&mut restored, 3, None).unwrap();
            model.restore_snapshot(&mut restored, &snapshot).unwrap();
            for token in (0..24).map(|i| i % 8) {
                let a = model.step(&mut ced, token, None).unwrap();
                let b = model.step(&mut restored, token, None).unwrap();
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    });
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn episode_cancellation_at_yield_poisons_without_publishing_partial_prefix() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().unwrap();
    prefill::with_batched_f32(|| {
        let ctx = MetalContext::new().unwrap();
        let weights = toy::Weights::new(&ctx);
        let embedding = embeddings(&ctx);
        let mut model = nonuniform_model(&weights, &ctx, &embedding);
        model.prefill_rows = 4096;
        let mut state = model.new_state(8193).unwrap();
        let tokens = vec![1; 8192];
        let cancel = AtomicBool::new(false);
        assert!(
            model
                .begin_cold_prefill_episode(&mut state, tokens.len())
                .unwrap()
        );
        assert!(
            !model
                .cold_prefill_episode_slice(&mut state, &tokens, Some(&cancel), 4096)
                .unwrap()
                .complete
        );
        cancel.store(true, Ordering::Release);
        assert!(
            model
                .cold_prefill_episode_slice(&mut state, &tokens, Some(&cancel), 4096)
                .is_err()
        );
        assert!(state.is_poisoned());
        assert_eq!(state.position(), 0);
        assert!(model.capture_snapshot(&state).is_err());
        assert!(model.step(&mut state, 1, None).is_err());
    });
}
