use super::*;
use crate::sampling::SamplingConfig;

fn sampler(temperature: f32, seed: u64) -> Sampler {
    Sampler::new(SamplingConfig {
        temperature,
        seed,
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn cpu_aggregate_admission_includes_cpu_and_one_safety_reserve() {
    use crate::metal::{MetalMemoryAdmissionReason, MetalMemorySignals};
    let safety = memory::DYNAMIC_RESERVE_BYTES;
    let signals = MetalMemorySignals {
        host_budget: Some(crate::metal::HostMemoryBudget {
            physical_bytes: 1024 << 30,
            conservative_available_bytes: 1024 << 30,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
        recommended_max_bytes: safety + 6,
        current_allocated_bytes: 0,
        process_limit_remaining_bytes: Some(safety + 11),
    };
    assert_eq!(admit_request(1, 2, 3, 5, signals).unwrap(), 6);
    assert_eq!(admit_request(1, 2, 3, 0, signals).unwrap(), 6);
    let resident = MetalMemorySignals {
        current_allocated_bytes: 6,
        process_limit_remaining_bytes: Some(safety + 5),
        ..signals
    };
    assert_eq!(admit_request(0, 0, 0, 5, resident).unwrap(), 0);
    assert!(
        admit_request(
            0,
            0,
            0,
            5,
            MetalMemorySignals {
                current_allocated_bytes: 7,
                ..resident
            },
        )
        .unwrap_err()
        .is_memory_pressure()
    );
    for (signals, reason) in [
        (
            MetalMemorySignals {
                process_limit_remaining_bytes: Some(safety + 10),
                ..signals
            },
            MetalMemoryAdmissionReason::ProcessInsufficient,
        ),
        (
            MetalMemorySignals {
                host_budget: Some(crate::metal::HostMemoryBudget {
                    physical_bytes: 1024 << 30,
                    conservative_available_bytes: 1024 << 30,
                    external_gap_bytes: 0,
                    private_shadow_bytes: 0,
                }),
                recommended_max_bytes: safety + 5,
                ..signals
            },
            MetalMemoryAdmissionReason::WorkingSetInsufficient,
        ),
    ] {
        let error = admit_request(1, 2, 3, 5, signals).unwrap_err();
        assert!(error.is_memory_pressure());
        let Glm5NextDFlashError::Target(Glm5NextMetalError::MemoryAdmission { denied, .. }) = error
        else {
            panic!("typed native admission refusal required");
        };
        assert_eq!(denied.reason(), reason);
    }
    let overflow = admit_request(1, 2, 3, u64::MAX, signals).unwrap_err();
    assert!(!overflow.is_memory_pressure());
    let Glm5NextDFlashError::Target(Glm5NextMetalError::MemoryAdmission { denied, .. }) = overflow
    else {
        panic!("typed reserve overflow refusal required");
    };
    assert_eq!(
        denied.reason(),
        MetalMemoryAdmissionReason::RequiredBytesOverflow
    );
    assert!(admit_request(u64::MAX, 1, 1, 0, signals).is_err());
}

#[test]
fn cpu_packet_bounds_preserve_ragged_greedy_and_no_draw_sampled_tail() {
    for block in [1, 2, 8] {
        for free in 0..=10 {
            for remaining in 0..=10 {
                for sampled in [false, true] {
                    let plan = packet_plan(block, 8, free, free, remaining, sampled).unwrap();
                    assert!(plan.rows <= block && plan.rows <= free && plan.rows <= 8);
                    assert!(plan.rows <= remaining + 1);
                    assert_eq!(plan.rows == 0, free == 0 || remaining == 0);
                    if sampled && plan.rows > 1 {
                        assert_eq!(plan.rows, block);
                        assert!(plan.sampled);
                    }
                    if plan.rows <= 1 {
                        assert!(!plan.sampled);
                    }
                }
            }
        }
    }
    assert_eq!(packet_plan(8, 8, 20, 20, 2, false).unwrap().rows, 3);
    assert_eq!(packet_plan(8, 8, 20, 20, 2, true).unwrap().rows, 1);
    assert_eq!(packet_plan(8, 2, 20, 20, 20, true).unwrap().rows, 1);
    assert!(packet_plan(8, 8, 8, 0, 8, false).is_err());
    assert_eq!(packet_plan(8, 8, 8, 7, 7, false).unwrap().rows, 8);
}

fn logits(winners: &[usize], vocab: usize) -> Vec<f32> {
    let mut logits = vec![-10.0; winners.len() * vocab];
    for (row, &winner) in winners.iter().enumerate() {
        logits[row * vocab + winner] = 10.0;
    }
    logits
}

#[test]
fn cpu_packet_emission_has_no_duplicate_anchor_or_stop_callback() {
    let input = [0, 1, 2, 3];
    for (winners, budget, stops, tokens, retained, stop) in [
        ([1, 2, 3, 4], 8, vec![], vec![1, 2, 3, 4], 4, None),
        ([1, 4, 3, 4], 8, vec![], vec![1, 4], 2, None),
        (
            [1, 2, 3, 4],
            8,
            vec![2],
            vec![1],
            2,
            Some(Glm5NextDFlashStop::Eos),
        ),
        (
            [1, 2, 3, 4],
            2,
            vec![],
            vec![1, 2],
            2,
            Some(Glm5NextDFlashStop::TokenLimit),
        ),
        (
            [1, 2, 3, 4],
            8,
            vec![4],
            vec![1, 2, 3],
            4,
            Some(Glm5NextDFlashStop::Eos),
        ),
        (
            [1, 4, 3, 4],
            8,
            vec![4],
            vec![1],
            2,
            Some(Glm5NextDFlashStop::Eos),
        ),
        (
            [1, 2, 3, 4],
            4,
            vec![],
            vec![1, 2, 3, 4],
            4,
            Some(Glm5NextDFlashStop::TokenLimit),
        ),
        (
            [1, 2, 3, 4],
            0,
            vec![],
            vec![],
            0,
            Some(Glm5NextDFlashStop::TokenLimit),
        ),
    ] {
        let mut rng = sampler(0.0, 1);
        let decision = decide_dflash_accept_prefix(
            &input,
            &logits(&winners, 5),
            5,
            None,
            &mut rng,
            budget,
            &stops,
        )
        .unwrap();
        let (packet, outputs) = packet_from_decision(decision, budget, &stops);
        assert_eq!(packet.tokens, tokens);
        assert_eq!(packet.retain_input_count, retained);
        assert_eq!(packet.stop_reason, stop);
        assert_eq!(packet.costs.selection, Duration::ZERO);
        assert_eq!(packet.emitted_tokens.len(), outputs);
        assert_eq!(
            packet.tokens,
            packet
                .emitted_tokens
                .iter()
                .copied()
                .filter(|t| !stops.contains(t))
                .collect::<Vec<_>>()
        );
        assert!(outputs <= budget);
        assert!(!packet.tokens.contains(&input[0]));
        assert!(packet.tokens.iter().all(|t| !stops.contains(t)));
        assert_eq!(rng.draws(), 0);
    }
}

#[test]
fn cpu_inclusive_history_retains_the_selected_stop_id_not_the_first_stop_id() {
    for (winners, stops, emitted, callbacks) in [
        ([1, 2, 3, 4], [5, 2], vec![1, 2], vec![1]),
        ([1, 4, 3, 4], [5, 4], vec![1, 4], vec![1]),
        ([1, 2, 3, 4], [5, 4], vec![1, 2, 3, 4], vec![1, 2, 3]),
    ] {
        let mut rng = sampler(0.0, 1);
        let decision = decide_dflash_accept_prefix(
            &[0, 1, 2, 3],
            &logits(&winners, 6),
            6,
            None,
            &mut rng,
            8,
            &stops,
        )
        .unwrap();
        let (packet, outputs) = packet_from_decision(decision, 8, &stops);
        assert_eq!(packet.emitted_tokens, emitted);
        assert_eq!(packet.tokens, callbacks);
        assert_eq!(packet.stop_reason, Some(Glm5NextDFlashStop::Eos));
        assert_eq!(outputs, emitted.len());
        assert_eq!(packet.emitted_tokens.last(), Some(&stops[1]));
        assert!(!packet.tokens.contains(&stops[1]));
        assert!(!packet.emitted_tokens.contains(&stops[0]));
    }
    for stop in [Glm5NextDFlashStop::Capacity, Glm5NextDFlashStop::TokenLimit] {
        assert!(empty_packet(stop).emitted_tokens.is_empty());
        assert_eq!(empty_packet(stop).costs.selection, Duration::ZERO);
    }
}

#[test]
fn cpu_sampled_anchor_only_uses_one_target_draw_and_no_proposal_draws() {
    let mut target = sampler(0.8, 7);
    let proposal = sampler(0.8, 13);
    let mut oracle = target.clone();
    let rows = [0.0, 1.0, 2.0, 3.0];
    let expected = oracle.sample(&rows).unwrap().token;
    let decision = decide_dflash_accept_prefix(&[0], &rows, 4, None, &mut target, 1, &[]).unwrap();
    let (packet, outputs) = packet_from_decision(decision, 1, &[]);
    assert_eq!(packet.tokens, [expected]);
    assert_eq!(packet.emitted_tokens, [expected]);
    assert_eq!(outputs, 1);
    assert_eq!(packet.retain_input_count, 1);
    assert_eq!(packet.stop_reason, Some(Glm5NextDFlashStop::TokenLimit));
    assert_eq!(target.draws(), 1);
    assert_eq!(proposal.draws(), 0);
    assert_eq!(target.diagnose_next(&rows), oracle.diagnose_next(&rows));
}

// Explicit opt-in only. Neither environment path is opened by nonignored tests.
fn load(ctx: &MetalContext) -> (Glm5NextWeights, MetalDFlashHead) {
    let target_path = std::env::var("QWEN_GLM_DFLASH_TARGET")
        .expect("set QWEN_GLM_DFLASH_TARGET to the archived GLM UD-IQ3_XXS GGUF");
    let drafter_path = std::env::var("QWEN_GLM_DFLASH_DRAFTER").unwrap_or_else(|_| {
        "/Users/tito/models/incoai-dflash2/GLM-5.3-Flash-DFlash2-Q8_0.gguf".into()
    });
    let target_gguf = GgufFile::open(target_path).unwrap();
    let weights = Glm5NextWeights::load(ctx, &target_gguf).unwrap();
    let drafter_gguf = GgufFile::open(drafter_path).unwrap();
    let description = crate::loader::open_dflash_drafter_for_target(
        &drafter_gguf,
        crate::loader::DFlashTargetShape {
            hidden_size: weights.config.hidden_size,
            vocab_size: weights.config.vocab_size,
            n_layer: weights.config.blocks.len() as u32,
        },
    )
    .unwrap();
    let head = MetalDFlashHead::load(ctx, &drafter_gguf, &description).unwrap();
    (weights, head)
}

fn options(prompt: usize, cap: usize) -> Glm5NextDFlashOptions {
    Glm5NextDFlashOptions {
        prompt_tokens: prompt,
        max_new_tokens: cap,
        stop_ids: vec![],
        verification: Glm5NextDFlashVerification::ConfiguredPacked,
    }
}

#[test]
#[ignore = "real GLM postcommit append failure: explicit model env, idle GPU lease, MTL_DEBUG_LAYER=1"]
fn metal_postcommit_context_append_failure_aborts_without_publication() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let (weights, head) = load(&ctx);
    let width = head.config.block_size as usize;
    let prompt = [1, 2, 3];
    let mut target = Glm5NextSession::with_prefill_rows(&ctx, &weights, 64, width).unwrap();

    // A refused aggregate must allocate none of its request buffers or advance
    // the target. Overflow is deterministic, independent of live memory pressure.
    let allocated = ctx.current_allocated_size();
    assert!(matches!(
        Glm5NextDFlashRunner::new_with_cpu_reserve(
            &ctx,
            &mut target,
            &head,
            options(prompt.len(), width + 2),
            sampler(0.8, 17),
            Some(sampler(0.8, 29)),
            u64::MAX,
        ),
        Err(Glm5NextDFlashError::Target(
            Glm5NextMetalError::MemoryAdmission { .. }
        ))
    ));
    assert_eq!(ctx.current_allocated_size(), allocated);
    assert_eq!(target.position(), 0);
    assert!(!target.is_poisoned());

    let mut committed_position = None;
    for fail in [false, true] {
        target.reset().unwrap();
        let mut runner = Glm5NextDFlashRunner::new_with_cpu_reserve(
            &ctx,
            &mut target,
            &head,
            options(prompt.len(), width + 2),
            sampler(0.8, 17),
            Some(sampler(0.8, 29)),
            4096,
        )
        .unwrap();
        let initial = runner.prefill(&prompt, &mut || false).unwrap();
        let carry = initial.next_carry.unwrap().token();
        let target_rng = runner.sampler().clone();
        let proposal_rng = runner.proposal_sampler().unwrap().clone();
        runner.fail_context_append = fail;
        let result = runner.step(carry, &mut || false);
        if fail {
            assert!(
                matches!(
                    result,
                    Err(Glm5NextDFlashError::Draft(DFlashError::Metal(
                        MetalError::CommandBufferFailed { .. }
                    )))
                ),
                "postcommit append failure must not return a packet"
            );
            assert_eq!(Some(runner.target_position()), committed_position);
            assert!(runner.target_position() > prompt.len());
            assert!(runner.target_is_poisoned());
            assert_eq!(runner.sampler().draws(), target_rng.draws());
            assert_eq!(
                runner.sampler().diagnose_next(&[0.0, 1.0, 2.0]),
                target_rng.diagnose_next(&[0.0, 1.0, 2.0])
            );
            assert_eq!(
                runner.proposal_sampler().unwrap().draws(),
                proposal_rng.draws()
            );
            assert_eq!(
                runner
                    .proposal_sampler()
                    .unwrap()
                    .diagnose_next(&[0.0, 1.0, 2.0]),
                proposal_rng.diagnose_next(&[0.0, 1.0, 2.0])
            );
            assert_eq!(
                (
                    runner.draft.target_ctx_n,
                    runner.draft.ctx_h_ready_n,
                    runner.draft.kv_ctx_ready_n
                ),
                (0, 0, 0)
            );
            assert_eq!(runner.counters().packets, 0);
            assert_eq!(
                runner.counters().output_tokens,
                initial.emitted_tokens.len()
            );
            assert_eq!(runner.counters().failures, 1);
            assert!(matches!(
                runner.step(carry, &mut || false),
                Err(Glm5NextDFlashError::Closed)
            ));
            assert!(matches!(
                runner.prefill(&prompt, &mut || false),
                Err(Glm5NextDFlashError::Closed)
            ));
            drop(runner);
            assert!(target.is_poisoned());
            assert!(matches!(target.reset(), Err(Glm5NextMetalError::Poisoned)));
        } else {
            let packet = result.unwrap();
            assert_eq!(
                runner.target_position(),
                prompt.len() + packet.retain_input_count
            );
            committed_position = Some(runner.target_position());
            assert!(runner.draft.ctx_h_ready_n > 0);
            assert!(runner.draft.kv_ctx_ready_n > 0);
        }
    }
}

#[test]
#[ignore = "real GLM+drafter packet test: explicit model env, idle GPU lease, MTL_DEBUG_LAYER=1"]
fn metal_packets_prefill_greedy_sampled_seeded_and_counters() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let (weights, head) = load(&ctx);
    let width = head.config.block_size as usize;
    let prompt = [1, 2, 3, 4, 5];
    let cap = width * 2 + 3;
    let mut target = Glm5NextSession::with_prefill_rows(&ctx, &weights, 128, width).unwrap();
    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        target.reset().unwrap();
        target.set_packed_lineage(lineage).unwrap();
        for sampled in [false, true] {
            let mut previous = None;
            for _ in 0..2 {
                target.reset().unwrap();
                let mut runner = Glm5NextDFlashRunner::new(
                    &ctx,
                    &mut target,
                    &head,
                    options(prompt.len(), cap),
                    sampler(if sampled { 0.8 } else { 0.0 }, 17),
                    sampled.then(|| sampler(0.8, 29)),
                )
                .unwrap();
                let mut packet = runner.prefill(&prompt, &mut || false).unwrap();
                assert_eq!(runner.target_position(), prompt.len());
                assert_eq!(runner.draft.target_ctx_n, prompt.len());
                let mut emitted = packet.tokens.clone();
                while packet.stop_reason.is_none() {
                    let carry = packet.next_carry.unwrap().token();
                    packet = runner.step(carry, &mut || false).unwrap();
                    emitted.extend_from_slice(&packet.tokens);
                    assert!(runner.draft.kv_ctx_ready_n <= runner.draft.ctx_h_ready_n);
                    assert!(runner.draft.ctx_h_ready_n <= runner.draft.target_ctx_n);
                    assert_eq!(runner.draft.target_ctx_n, runner.target_position());
                    assert_eq!(runner.target.packed_lineage(), Some(lineage));
                    assert_eq!(
                        runner.target.effective_fast_precision(),
                        (lineage == PackedLineage::Fast).then_some(FastPrecision::Half)
                    );
                }
                assert_eq!(packet.stop_reason, Some(Glm5NextDFlashStop::TokenLimit));
                let counts = runner.counters();
                eprintln!(
                    "{lineage:?} sampled={sampled} counts={counts:?} last_costs={:?}",
                    packet.costs
                );
                assert_eq!(counts.output_tokens, cap);
                assert_eq!(counts.callback_tokens, emitted.len());
                assert_eq!(counts.retained_inputs, cap - 1);
                assert_eq!(runner.target_position(), prompt.len() + cap - 1);
                assert!(counts.packets > 1);
                assert!(counts.draft_calls > 0);
                assert_eq!(runner.remaining_output_tokens(), 0);
                if !sampled {
                    assert_eq!(counts.target_draws, 0);
                }
                let result = (emitted, counts);
                if let Some(expected) = &previous {
                    assert_eq!(&result, expected);
                }
                previous = Some(result);
            }
        }
    }
}

#[test]
#[ignore = "real GLM+drafter boundaries/cancellation: explicit model env, idle GPU lease, MTL_DEBUG_LAYER=1"]
fn metal_packets_limits_eos_and_sampled_cancellation_cleanup() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let (weights, head) = load(&ctx);
    let width = head.config.block_size as usize;
    let prompt = [1, 2, 3];
    let mut target = Glm5NextSession::with_prefill_rows(&ctx, &weights, 64, width).unwrap();
    let mut first = None;
    for cap in [0, 1, 2] {
        target.reset().unwrap();
        let mut runner = Glm5NextDFlashRunner::new(
            &ctx,
            &mut target,
            &head,
            options(prompt.len(), cap),
            sampler(0.8, 17),
            Some(sampler(0.8, 29)),
        )
        .unwrap();
        let mut packet = runner.prefill(&prompt, &mut || false).unwrap();
        if cap > 0 {
            first = packet.next_carry.map(Glm5NextDFlashCarry::token);
        }
        if packet.stop_reason.is_none() {
            packet = runner
                .step(packet.next_carry.unwrap().token(), &mut || false)
                .unwrap();
            assert!(packet.anchor_only);
        }
        assert_eq!(packet.stop_reason, Some(Glm5NextDFlashStop::TokenLimit));
        assert_eq!(
            runner.target_position(),
            if cap == 0 { 0 } else { prompt.len() + cap - 1 }
        );
        assert_eq!(runner.sampler().draws(), cap);
        assert_eq!(runner.proposal_sampler().unwrap().draws(), 0);
    }
    target.reset().unwrap();
    {
        let mut config = options(prompt.len(), 20);
        config.stop_ids = vec![
            (first.unwrap() + 1) % weights.config.vocab_size as i32,
            first.unwrap(),
        ];
        let mut runner = Glm5NextDFlashRunner::new(
            &ctx,
            &mut target,
            &head,
            config,
            sampler(0.8, 17),
            Some(sampler(0.8, 29)),
        )
        .unwrap();
        let packet = runner.prefill(&prompt, &mut || false).unwrap();
        assert!(packet.tokens.is_empty());
        assert_eq!(packet.emitted_tokens, [first.unwrap()]);
        assert_eq!(packet.costs.selection, packet.costs.accept);
        assert!(packet.costs.selection <= packet.costs.total);
        assert_eq!(runner.counters().output_tokens, 1);
        assert_eq!(runner.counters().callback_tokens, 0);
        assert_eq!(packet.stop_reason, Some(Glm5NextDFlashStop::Eos));
        assert_eq!(runner.target_position(), prompt.len());
        assert!(matches!(
            runner.step(first.unwrap(), &mut || false),
            Err(Glm5NextDFlashError::Closed)
        ));
    }
    for boundary in 1..=5 {
        target.reset().unwrap();
        let mut runner = Glm5NextDFlashRunner::new(
            &ctx,
            &mut target,
            &head,
            options(prompt.len(), 32),
            sampler(0.8, 17),
            Some(sampler(0.8, 29)),
        )
        .unwrap();
        let packet = runner.prefill(&prompt, &mut || false).unwrap();
        let carry = packet.next_carry.unwrap().token();
        let baseline = runner.target.capture_snapshot().unwrap();
        let target_rng = runner.sampler().clone();
        let proposal_rng = runner.proposal_sampler().unwrap().clone();
        let mut calls = 0;
        assert!(matches!(
            runner.step(carry, &mut || {
                calls += 1;
                calls == boundary
            }),
            Err(Glm5NextDFlashError::Cancelled)
        ));
        assert!(!runner.target_is_poisoned());
        assert!(
            runner
                .target
                .capture_snapshot()
                .unwrap()
                .same_state(&baseline)
        );
        assert_eq!(runner.sampler().draws(), target_rng.draws());
        assert_eq!(
            runner.sampler().diagnose_next(&[0.0, 1.0, 2.0]),
            target_rng.diagnose_next(&[0.0, 1.0, 2.0])
        );
        assert_eq!(
            runner.proposal_sampler().unwrap().draws(),
            proposal_rng.draws()
        );
        assert_eq!(
            (
                runner.draft.target_ctx_n,
                runner.draft.ctx_h_ready_n,
                runner.draft.kv_ctx_ready_n
            ),
            (0, 0, 0)
        );
        assert_eq!(runner.counters().cancellations, 1);
        assert!(matches!(
            runner.step(carry, &mut || false),
            Err(Glm5NextDFlashError::Closed)
        ));
    }
    target.reset().unwrap();
    {
        let mut config = options(prompt.len(), 32);
        config.verification = Glm5NextDFlashVerification::PackedThenSerial;
        let mut runner = Glm5NextDFlashRunner::new(
            &ctx,
            &mut target,
            &head,
            config,
            sampler(0.8, 17),
            Some(sampler(0.8, 29)),
        )
        .unwrap();
        let packet = runner.prefill(&prompt, &mut || false).unwrap();
        let carry = packet.next_carry.unwrap().token();
        let before_draws = runner.sampler().draws();
        let mut calls = 0;
        let mut injection = None;
        let error = runner
            .step(carry, &mut || {
                calls += 1;
                if calls == 3 {
                    // Checkpoint capture completed; the next wait is verification.
                    injection = Some(crate::metal::fail_next_completion_check());
                }
                false
            })
            .err()
            .expect("fatal verification command failure");
        drop(injection);
        assert!(matches!(
            error,
            Glm5NextDFlashError::Target(Glm5NextMetalError::Metal(
                MetalError::CommandBufferFailed { .. }
            ))
        ));
        assert!(runner.target_is_poisoned());
        assert_eq!(
            runner.counters().serial_verifications,
            0,
            "no fallback after a command failure"
        );
        assert_eq!(runner.sampler().draws(), before_draws);
        assert_eq!(runner.proposal_sampler().unwrap().draws(), 0);
        assert_eq!(
            (
                runner.draft.target_ctx_n,
                runner.draft.ctx_h_ready_n,
                runner.draft.kv_ctx_ready_n
            ),
            (0, 0, 0)
        );
        assert!(matches!(
            runner.step(carry, &mut || false),
            Err(Glm5NextDFlashError::Closed)
        ));
    }
    assert!(target.is_poisoned());
    assert!(matches!(target.reset(), Err(Glm5NextMetalError::Poisoned)));
    drop(target);
    let mut target = Glm5NextSession::with_prefill_rows(&ctx, &weights, 64, width).unwrap();
    let mut serial_result = None;
    for verification in [
        Glm5NextDFlashVerification::Serial,
        Glm5NextDFlashVerification::PackedThenSerial,
    ] {
        target.reset().unwrap();
        let mut config = options(prompt.len(), 3);
        config.verification = verification;
        let mut runner = Glm5NextDFlashRunner::new(
            &ctx,
            &mut target,
            &head,
            config,
            sampler(0.8, 17),
            Some(sampler(0.8, 29)),
        )
        .unwrap();
        let mut packet = runner.prefill(&prompt, &mut || false).unwrap();
        let mut tokens = packet.tokens.clone();
        while packet.stop_reason.is_none() {
            packet = runner
                .step(packet.next_carry.unwrap().token(), &mut || false)
                .unwrap();
            assert_eq!(
                packet.verification_arithmetic,
                Some(Glm5NextVerificationArithmetic::Serial)
            );
            tokens.extend_from_slice(&packet.tokens);
        }
        let result = (
            tokens,
            runner.sampler().draws(),
            runner.proposal_sampler().unwrap().draws(),
        );
        let state = runner.target.capture_snapshot().unwrap();
        if let Some((expected, expected_state)) = &serial_result {
            assert_eq!(&result, expected);
            assert!(state.same_state(expected_state));
        }
        serial_result = Some((result, state));
    }
    drop(target);
    for boundary in 1..=4 {
        let mut target = Glm5NextSession::with_prefill_rows(&ctx, &weights, 64, width).unwrap();
        let mut runner = Glm5NextDFlashRunner::new(
            &ctx,
            &mut target,
            &head,
            options(prompt.len(), 10),
            sampler(0.8, 17),
            Some(sampler(0.8, 29)),
        )
        .unwrap();
        let mut calls = 0;
        assert!(matches!(
            runner.prefill(&prompt, &mut || {
                calls += 1;
                calls == boundary
            }),
            Err(Glm5NextDFlashError::Cancelled)
        ));
        assert_eq!(runner.target_is_poisoned(), boundary != 1);
        assert_eq!(
            runner.target_position(),
            if boundary <= 2 { 0 } else { prompt.len() }
        );
        assert_eq!(runner.counters().prefill_tokens, runner.target_position());
        assert_eq!(runner.sampler().draws(), 0);
        assert_eq!(runner.draft.target_ctx_n, 0);
        assert!(matches!(
            runner.prefill(&prompt, &mut || false),
            Err(Glm5NextDFlashError::Closed)
        ));
        drop(runner);
        if boundary == 1 {
            target.reset().unwrap();
        } else {
            assert!(matches!(target.reset(), Err(Glm5NextMetalError::Poisoned)));
        }
        drop(target);
    }
    let mut target = Glm5NextSession::with_prefill_rows(&ctx, &weights, 64, width).unwrap();
    {
        let long_prompt: Vec<u32> = (1..=(2 * width + 1) as u32).collect();
        let mut runner = Glm5NextDFlashRunner::new(
            &ctx,
            &mut target,
            &head,
            options(long_prompt.len(), 10),
            sampler(0.8, 17),
            Some(sampler(0.8, 29)),
        )
        .unwrap();
        let mut calls = 0;
        let result = runner.prefill(&long_prompt, &mut || {
            calls += 1;
            calls == 3 // Request boundary, first chunk, then refuse second chunk.
        });
        assert!(matches!(result, Err(Glm5NextDFlashError::Cancelled)));
        assert_eq!(calls, 3);
        assert_eq!(runner.target_position(), width);
        assert_eq!(runner.counters().prefill_tokens, width);
        assert_eq!(runner.counters().cancellations, 1);
        assert_eq!(runner.sampler().draws(), 0);
        assert_eq!(runner.proposal_sampler().unwrap().draws(), 0);
        assert_eq!(runner.draft.target_ctx_n, 0);
        assert!(
            runner.target_is_poisoned(),
            "cancelled request cannot resume a partial captured prompt"
        );
        assert!(matches!(
            runner.prefill(&long_prompt, &mut || false),
            Err(Glm5NextDFlashError::Closed)
        ));
    }
    assert!(matches!(target.reset(), Err(Glm5NextMetalError::Poisoned)));
    drop(target);
    let mut target = Glm5NextSession::with_prefill_rows(&ctx, &weights, 64, width).unwrap();
    {
        let mut runner = Glm5NextDFlashRunner::new(
            &ctx,
            &mut target,
            &head,
            options(prompt.len(), 10),
            sampler(0.0, 17),
            None,
        )
        .unwrap();
        runner.prefill(&prompt, &mut || false).unwrap();
        runner.abort_output();
        assert!(runner.target_is_poisoned());
        assert_eq!(runner.draft.target_ctx_n, 0);
    }
    assert!(matches!(target.reset(), Err(Glm5NextMetalError::Poisoned)));
    drop(target);
    let mut target =
        Glm5NextSession::with_prefill_rows(&ctx, &weights, prompt.len(), width.min(prompt.len()))
            .unwrap();
    let mut runner = Glm5NextDFlashRunner::new(
        &ctx,
        &mut target,
        &head,
        options(prompt.len(), 10),
        sampler(0.0, 17),
        None,
    )
    .unwrap();
    let initial = runner.prefill(&prompt, &mut || false).unwrap();
    let packet = runner
        .step(initial.next_carry.unwrap().token(), &mut || false)
        .unwrap();
    assert_eq!(packet.stop_reason, Some(Glm5NextDFlashStop::Capacity));
    assert!(packet.tokens.is_empty());
    assert_eq!(runner.target_position(), prompt.len());
}
