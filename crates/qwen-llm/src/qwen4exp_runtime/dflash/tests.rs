use super::*;

#[test]
fn admission_bounds_cover_t_minus_one_and_ragged_verification() {
    assert_eq!(request_extent(7, 1, 8, 7, None).unwrap(), 1);
    assert_eq!(request_extent(7, 2, 8, 8, None).unwrap(), 1);
    assert_eq!(request_extent(7, 4, 8, 10, Some(3)).unwrap(), 3);
    assert!(request_extent(7, 4, 8, 9, Some(8)).is_err());
    assert!(request_extent(7, 4, 8, 10, Some(2)).is_err());
    assert!(request_extent(0, 4, 8, 32, Some(8)).is_err());
    assert!(request_extent(7, 0, 8, 32, Some(8)).is_err());
    assert!(request_extent(7, 4, 0, 32, Some(8)).is_err());
    assert!(request_extent(usize::MAX, 2, 8, usize::MAX, Some(8)).is_err());
    assert!(request_extent(i32::MAX as usize, 1, 8, usize::MAX, Some(8)).is_err());
}

#[test]
fn initial_and_bonus_emission_stop_before_the_next_forward() {
    assert_eq!(stop_after(9, 1, 1, &[9]), Some(Qwen4ExpDFlashFinish::Eos));
    assert_eq!(
        stop_after(8, 1, 1, &[9]),
        Some(Qwen4ExpDFlashFinish::TokenLimit)
    );
    assert_eq!(stop_after(8, 1, 2, &[9]), None);
}

fn visible_state(runner: &Qwen4ExpTextRunner<'_, '_, '_>) -> Vec<Vec<u8>> {
    runner
        .workspace
        .checkpoint_visible_regions_for_tests()
        .unwrap()
        .iter()
        .map(|tensor| {
            // SAFETY: only completed test commands, with checked logical visibility.
            unsafe {
                std::slice::from_raw_parts(
                    tensor
                        .buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(tensor.offset as usize),
                    tensor.n_bytes() as usize,
                )
                .to_vec()
            }
        })
        .collect()
}

fn assert_bits_equal(expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len());
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(expected.to_bits(), actual.to_bits(), "logit {index}");
    }
}

fn real_fixture<'gguf>(
    ctx: &MetalContext,
    gguf: &'gguf GgufFile,
    draft_gguf: &GgufFile,
) -> (Qwen4ExpLoadedModel<'gguf>, MetalDFlashHead, Vec<u32>) {
    use crate::loader::{DFlashTargetShape, open_dflash_drafter_for_target};
    use crate::tokenizer::Tokenizer;

    let tokenizer = Tokenizer::from_gguf(gguf).unwrap();
    let seed: Vec<u32> = tokenizer.encode(
        "<|im_start|>user\nCount one two three four five six seven.<|im_end|>\n<|im_start|>assistant\n", false,
    ).unwrap().into_iter().map(|id| u32::try_from(id).unwrap()).collect();
    let capacity =
        Qwen4ExpSessionCapacity::for_forward_limit(&Qwen4ExpConfig::flash_next_reference(), 32)
            .unwrap();
    let loaded = Qwen4ExpLoadedModel::load_with_packed_prefill(ctx, gguf, capacity, 32).unwrap();
    let cfg = loaded.config();
    let bound = open_dflash_drafter_for_target(
        draft_gguf,
        DFlashTargetShape {
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            n_layer: cfg.layer_count,
        },
    )
    .unwrap();
    let head = {
        let price = MetalDFlashHead::priced_metal_weight_bytes(ctx, &bound).unwrap();
        let _allocation = ctx.begin_allocation_transaction();
        assert!(
            evaluate_metal_memory_admission(
                price,
                QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
                ctx.memory_signals(),
                true
            )
            .refusal()
            .is_none()
        );
        MetalDFlashHead::load(ctx, draft_gguf, &bound).unwrap()
    };
    (loaded, head, seed)
}

/// Real resident target + drafter; no downloads. The all-row readout test separately
/// checks intermediate distributions against native endpoints. Here every retained
/// packet is checked against its native execution schedule (same-width capture-OFF
/// packed on full retention, serial on restore/replay), not a serial-all-the-time
/// equality claim. Host snapshots are test oracles only, never adapter scratch.
#[test]
#[ignore = "exclusive GPU lease; QWEN4EXP_Q3_K_XL_RUNTIME_GGUF and QWEN4EXP_DFLASH_GGUF local paths"]
fn real_packets_greedy_sampled_native_retention_and_cancel_cleanup() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let draft_path = std::env::var("QWEN4EXP_DFLASH_GGUF").expect("local drafter GGUF path");
    let draft_gguf = GgufFile::open(draft_path).unwrap();
    let ctx = MetalContext::new().unwrap();
    let (mut loaded, head, seed) = real_fixture(&ctx, &gguf, &draft_gguf);
    let mut runner = loaded.create_runner(&ctx).unwrap();
    assert!(head.config.block_size > 1);
    for (temperature, prefix, budget, cancel) in [
        (0.0, 7, 1, false),
        (0.0, 7, 26, false),
        (1.0, 3, 30, false),
        (1.0, 7, 25, true),
    ] {
        let prompt: Vec<u32> = seed.iter().copied().cycle().take(prefix).collect();
        let config = SamplingConfig {
            temperature,
            seed: 57,
            ..SamplingConfig::qwen38_flash_next(57)
        };
        runner.reset().unwrap();
        runner.prefill(&prompt).unwrap();
        let expected_initial = Sampler::new(config)
            .unwrap()
            .sample(runner.logits().unwrap().as_slice())
            .unwrap()
            .token;
        runner.reset().unwrap();
        let (mut adapter, initial) = Qwen4ExpDFlashAdapter::start(
            &mut runner,
            &head,
            &prompt,
            Sampler::new(config).unwrap(),
            budget,
            &[],
        )
        .unwrap();
        assert_eq!(initial.tokens, [expected_initial]);
        let mut emitted = initial.tokens;
        let initial_context = adapter.committed_context_rows();
        while adapter.state() == Qwen4ExpDFlashState::Running {
            let entry = adapter.runner.capture_snapshot().unwrap();
            let position = adapter.target_position();
            let carry = *emitted.last().unwrap();
            let previous_full = adapter.stats().full_retention_packets;
            let packet = adapter.step().unwrap();
            assert!(!packet.tokens.is_empty());
            let retained = adapter.target_position() - position;
            let reference_inputs: Vec<u32> = std::iter::once(carry)
                .chain(packet.tokens.iter().copied())
                .take(retained)
                .map(|id| id as u32)
                .collect();
            let expected_state = visible_state(adapter.runner);
            let expected_history = adapter.runner.workspace.ple_prior_tokens().to_vec();
            let expected_logits = adapter.runner.logits().unwrap().to_vec();
            // Replace the test target with an independent native-schedule replay.
            // The adapter context is not changed and its next checkpoint overwrites
            // the old entry; snapshots here are deliberately outside production.
            adapter.runner.restore_snapshot(&entry).unwrap();
            if adapter.stats().full_retention_packets > previous_full {
                execute_qwen4exp_text_packed_sync(
                    adapter.runner.ctx,
                    &reference_inputs,
                    adapter.runner.ple_table,
                    &adapter.runner.weights,
                    &mut adapter.runner.workspace,
                )
                .unwrap();
            } else {
                for &token in &reference_inputs {
                    adapter.runner.forward_token(token).unwrap();
                }
            }
            assert_eq!(visible_state(adapter.runner), expected_state);
            assert_eq!(
                adapter.runner.workspace.ple_prior_tokens(),
                expected_history
            );
            assert_bits_equal(
                &expected_logits,
                adapter.runner.logits().unwrap().as_slice(),
            );
            emitted.extend(packet.tokens);
            assert_eq!(adapter.target_position(), prefix + emitted.len() - 1);
            assert_eq!(
                adapter.committed_context_rows(),
                initial_context + adapter.stats().retained_inputs
            );
            assert!(adapter.draft_session.ctx_h_ready_n <= adapter.draft_session.target_ctx_n);
            assert!(adapter.draft_session.kv_ctx_ready_n <= adapter.draft_session.target_ctx_n);
            if cancel && adapter.stats().packets == 2 {
                let position = adapter.target_position();
                let context = adapter.committed_context_rows();
                let packet = adapter.cancel().unwrap();
                assert_eq!(packet.finish, Some(Qwen4ExpDFlashFinish::Cancelled));
                assert!(packet.tokens.is_empty());
                assert!(adapter.step().is_err());
                assert_eq!(adapter.target_position(), position);
                assert_eq!(adapter.committed_context_rows(), context);
            }
        }
        assert!(!adapter.target_is_poisoned());
        if !cancel {
            assert_eq!(emitted.len(), budget);
            assert_eq!(
                adapter.state(),
                Qwen4ExpDFlashState::Finished(Qwen4ExpDFlashFinish::TokenLimit)
            );
        }
        if budget > 1 {
            assert!(adapter.stats().packets >= 2);
        }
        let stats = adapter.finish();
        assert_eq!(
            stats.proposed_tokens,
            stats.draft_calls * (head.config.block_size as usize - 1)
        );
        if temperature > 0.0 {
            assert_eq!(stats.proposal_draws, stats.proposed_tokens);
        }
        assert_eq!(runner.next_position(), prefix + emitted.len() - 1);
    }

    runner.reset().unwrap();
    let prompt: Vec<u32> = seed.iter().copied().cycle().take(7).collect();
    assert!(
        Qwen4ExpDFlashAdapter::start(
            &mut runner,
            &head,
            &prompt,
            Sampler::new(SamplingConfig::default()).unwrap(),
            0,
            &[],
        )
        .is_err()
    );
    assert_eq!(runner.next_position(), 0);
    runner.prefill(&prompt).unwrap();
    let before = visible_state(&runner);
    let stop = Sampler::new(SamplingConfig::default())
        .unwrap()
        .sample(runner.logits().unwrap().as_slice())
        .unwrap()
        .token;
    assert!(
        Qwen4ExpDFlashAdapter::start(
            &mut runner,
            &head,
            &prompt,
            Sampler::new(SamplingConfig::default()).unwrap(),
            25,
            &[],
        )
        .is_err()
    );
    assert_eq!(visible_state(&runner), before);
    runner.reset().unwrap();
    let (mut adapter, packet) = Qwen4ExpDFlashAdapter::start(
        &mut runner,
        &head,
        &prompt,
        Sampler::new(SamplingConfig::default()).unwrap(),
        25,
        &[stop],
    )
    .unwrap();
    assert_eq!(packet.tokens, [stop]);
    assert_eq!(packet.finish, Some(Qwen4ExpDFlashFinish::Eos));
    assert!(adapter.step().is_err());
    assert_eq!(adapter.target_position(), prompt.len());
    assert_eq!(adapter.finish().draft_calls, 0);

    let workspace = runner.into_workspace();
    loaded.restore_workspace(workspace).unwrap();
    let runner = loaded.create_runner(&ctx).unwrap();
    assert_eq!(runner.next_position(), 0);
    let workspace = runner.into_workspace();
    loaded.restore_workspace(workspace).unwrap();
}

#[test]
#[ignore = "exclusive GPU lease; QWEN4EXP_Q3_K_XL_RUNTIME_GGUF and QWEN4EXP_DFLASH_GGUF local paths"]
fn real_packet_failure_publication_and_checkpoint_poison() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let draft_path = std::env::var("QWEN4EXP_DFLASH_GGUF").expect("local drafter GGUF path");
    let draft_gguf = GgufFile::open(draft_path).unwrap();
    let ctx = MetalContext::new().unwrap();
    let (mut loaded, head, seed) = real_fixture(&ctx, &gguf, &draft_gguf);
    assert!(head.config.block_size > 1);
    assert!(
        head.selector.is_some(),
        "sampled proposals require a selector"
    );
    let prompt: Vec<u32> = seed.iter().copied().cycle().take(7).collect();
    let mut runner = loaded.create_runner(&ctx).unwrap();
    for (budget, poison_checkpoint) in [(2, false), (3, false), (3, true)] {
        runner.reset().unwrap();
        let (mut adapter, initial) = Qwen4ExpDFlashAdapter::start(
            &mut runner,
            &head,
            &prompt,
            Sampler::new(SamplingConfig {
                temperature: 1.0,
                ..SamplingConfig::qwen38_flash_next(57)
            })
            .unwrap(),
            budget,
            &[],
        )
        .unwrap();
        assert_eq!(initial.tokens.len(), 1);
        assert_eq!(initial.finish, None);
        let position = adapter.target_position();
        let context = adapter.committed_context_rows();
        let carry = adapter.carry;
        let stats = adapter.stats().clone();
        let target_rng = adapter.sampler.clone();
        let proposal_rng = adapter.proposal_sampler.as_ref().unwrap().clone();
        let entry = visible_state(adapter.runner);
        let history = adapter.runner.workspace.ple_prior_tokens().to_vec();

        if poison_checkpoint {
            // Verification checks Metal status directly, bypassing the existing
            // completion seam. Keep a live entry, then poison via a separate
            // checkpoint capture to test that step must not restore that entry.
            adapter
                .runner
                .capture_device_checkpoint(&mut adapter.checkpoint)
                .unwrap();
            adapter.checkpoint_live = true;
            let mut failed_capture = adapter.runner.allocate_device_checkpoint_bank().unwrap();
            let failure = {
                let _failure = crate::metal::fail_next_completion_check();
                adapter
                    .runner
                    .capture_device_checkpoint(&mut failed_capture)
                    .unwrap_err()
            };
            assert_eq!(failed_capture.captured_position(), None);
            drop(failed_capture);
            assert!(
                failure
                    .to_string()
                    .contains("injected discarded command buffer")
            );
            assert!(adapter.target_is_poisoned());
            assert_eq!(adapter.target_position(), position);
            assert!(adapter.runner.logits().is_err());
        } else {
            // Restrict only logical append capacity; allocated buffers and the
            // committed context remain valid for drafting and verification.
            adapter.draft_session.target_ctx_capacity = context;
        }

        let error = adapter.step().unwrap_err();
        let Qwen4ExpDFlashError::Packet {
            source,
            target_position,
            poisoned,
            restored_entry,
            rollback_error,
        } = error
        else {
            panic!("expected packet failure, got {error:?}");
        };
        assert_eq!(target_position, adapter.target_position());
        assert_eq!(poisoned, poison_checkpoint);
        assert_eq!(poisoned, adapter.target_is_poisoned());
        assert_eq!(rollback_error, None);
        assert!(!adapter.checkpoint_live);
        assert_eq!(adapter.stats().draft_calls, 1);
        assert_eq!(
            adapter.stats().proposed_tokens,
            head.config.block_size as usize - 1
        );

        if poison_checkpoint {
            assert!(matches!(*source, Qwen4ExpDFlashError::Target(_)));
            assert!(!restored_entry);
            assert_eq!(target_position, position);
            assert_eq!(adapter.stats().restores, 0);
            assert_eq!(adapter.stats().verified_inputs, 0);
            assert_eq!(adapter.checkpoint.captured_position(), Some(position));
            assert!(
                adapter
                    .runner
                    .restore_device_checkpoint(&mut adapter.checkpoint)
                    .is_err()
            );
            assert!(adapter.target_is_poisoned());
        } else {
            assert!(matches!(
                *source,
                Qwen4ExpDFlashError::Draft(DFlashError::CtxOverflow(n, capacity))
                    if n > context && capacity == context
            ));
            assert_eq!(adapter.stats().verified_inputs, budget - 1);
            // Width one must retain its anchor. Width two can instead consume
            // the checkpoint for replay; late append failure must not reuse it.
            let replayed = adapter.stats().replay_inputs;
            assert!(replayed <= 1);
            if budget == 2 {
                assert_eq!(replayed, 0);
            }
            assert_eq!(restored_entry, replayed == 0);
            assert_eq!(adapter.stats().restores, 1);
            assert_eq!(adapter.checkpoint.captured_position(), None);
            assert_eq!(target_position, position + replayed);
            if restored_entry {
                assert_eq!(visible_state(adapter.runner), entry);
                assert_eq!(adapter.runner.workspace.ple_prior_tokens(), history);
                assert!(
                    adapter.runner.logits().is_err(),
                    "restore invalidates logits"
                );
            } else {
                assert!(adapter.runner.logits().is_ok());
            }
        }

        assert_eq!(adapter.state(), Qwen4ExpDFlashState::Failed);
        assert!(adapter.step().is_err());
        assert!(adapter.cancel().is_err());
        assert_eq!(adapter.state(), Qwen4ExpDFlashState::Failed);
        assert_eq!(adapter.target_position(), target_position);
        assert_eq!(adapter.target_is_poisoned(), poisoned);
        assert_eq!(adapter.committed_context_rows(), context);
        assert_eq!(adapter.carry, carry);
        assert_eq!(adapter.stats().emitted_tokens, stats.emitted_tokens);
        assert_eq!(adapter.stats().packets, stats.packets);
        assert_eq!(adapter.stats().retained_inputs, stats.retained_inputs);
        assert_eq!(adapter.stats().target_draws, stats.target_draws);
        assert_eq!(adapter.stats().proposal_draws, stats.proposal_draws);
        for (actual, expected) in [
            (&adapter.sampler, &target_rng),
            (adapter.proposal_sampler.as_ref().unwrap(), &proposal_rng),
        ] {
            assert_eq!(actual.draws(), expected.draws());
            assert_eq!(
                actual.diagnose_next(&[0.0, 1.0, 2.0]).unwrap(),
                expected.diagnose_next(&[0.0, 1.0, 2.0]).unwrap()
            );
        }
        adapter.finish();
        assert_eq!(runner.is_poisoned(), poisoned);
        assert_eq!(runner.next_position(), target_position);
    }
}
