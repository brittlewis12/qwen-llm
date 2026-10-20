use super::*;
use crate::glm5_next::BlockKind;

fn config() -> Glm5NextConfig {
    Glm5NextConfig {
        architecture: crate::glm5_next::ARCHITECTURE_NAME.into(),
        stored_block_count: 46,
        nextn_layer_count: 1,
        context_length: 8192,
        hidden_size: 4096,
        dense_ffn_size: 12288,
        head_count: 64,
        vocab_size: 154880,
        rms_epsilon: 1e-5,
        layer_norm_epsilon: 1e-6,
        expert_count: 288,
        expert_used_count: 8,
        expert_ffn_size: 2048,
        shared_expert_count: 1,
        shared_expert_ffn_size: 2048,
        leading_dense_block_count: 3,
        expert_weights_scale: 2.5,
        swiglu_clamp: 10.0,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        mla_key_head_dim: 256,
        mla_value_head_dim: 256,
        kda_head_dim: 128,
        kda_conv_kernel: 4,
        kda_gate_lower_bound: -5.0,
        indexer_head_count: 32,
        indexer_head_dim: 128,
        indexer_top_k: 2048,
        indexer_pool: 4,
        hc_streams: 4,
        hc_sinkhorn_iterations: 20,
        hc_epsilon: 1e-6,
        blocks: (0..45)
            .map(|i| BlockKind {
                mixer: if i % 4 == 3 {
                    MixerKind::Mla
                } else {
                    MixerKind::Kda
                },
                ffn: if i < 3 { FfnKind::Dense } else { FfnKind::Moe },
            })
            .collect(),
    }
}

fn stamp() -> Stamp {
    Stamp {
        owner: Owner {
            session: 10,
            weights: 20,
            device: 30,
        },
        generation: 1,
        position: 2050,
        end: 2058,
        lineage: Some(PackedLineage::Exact),
        precision: None,
        pool: 4,
    }
}

#[test]
fn cpu_checkpoint_inventory_is_bounded_and_preserves_ordinary_ledger() {
    let mut c = config();
    let logical = Glm5NextAppendCheckpoint::priced_bytes_for(&c, &Some).unwrap();
    assert_eq!(logical, 152_655_872);
    let specs: Vec<_> = c
        .blocks
        .iter()
        .flat_map(|b| region_specs(&c, b.mixer))
        .collect();
    assert_eq!(specs.len(), 34 * 2 + 11);
    assert_eq!(
        region_specs(&c, MixerKind::Mla),
        vec![memory::mla_pending_spec(&c)]
    );
    assert_eq!(memory::mla_pending_spec(&c).bytes(), Some(2048));
    let ledger = Glm5NextMemoryLedger::new(&c, 1234, 4096, 8, &memory::apple_16k_price).unwrap();
    let priced = Glm5NextAppendCheckpoint::priced_bytes_for(&c, &memory::apple_16k_price).unwrap();
    assert_eq!(
        priced,
        memory::priced(&specs, &memory::apple_16k_price).unwrap()
    );
    assert!(priced >= logical);
    for context in [1, 2052, 1_048_576] {
        c.context_length = context;
        assert_eq!(
            Glm5NextAppendCheckpoint::priced_bytes_for(&c, &Some).unwrap(),
            logical
        );
    }
    assert_eq!(
        ledger,
        Glm5NextMemoryLedger::new(&c, 1234, 4096, 8, &memory::apple_16k_price).unwrap()
    );
    assert!(Glm5NextAppendCheckpoint::priced_bytes_for(&c, &|_| None).is_err());
    assert!(Glm5NextAppendCheckpoint::priced_bytes_for(&c, &|_| Some(u64::MAX)).is_err());
}

#[test]
fn cpu_checkpoint_stamp_refuses_stale_foreign_and_changed_visibility() {
    let s = stamp();
    let check = |active, owner, position, lineage, precision, pool| {
        s.validate(active, owner, position, lineage, precision, pool)
    };
    for position in s.position..=s.end {
        check(Some(s), s.owner, position, s.lineage, s.precision, 4).unwrap();
    }
    assert!(check(None, s.owner, s.position, s.lineage, s.precision, 4).is_err());
    let mut next = s;
    next.generation += 1;
    assert!(check(Some(next), s.owner, s.position, s.lineage, s.precision, 4).is_err());
    for owner in [
        Owner {
            session: 11,
            ..s.owner
        },
        Owner {
            weights: 21,
            ..s.owner
        },
        Owner {
            device: 31,
            ..s.owner
        },
    ] {
        assert!(check(Some(s), owner, s.position, s.lineage, s.precision, 4).is_err());
    }
    for position in [0, s.position - 1, s.end + 1, usize::MAX] {
        assert!(check(Some(s), s.owner, position, s.lineage, s.precision, 4).is_err());
    }
    assert!(
        check(
            Some(s),
            s.owner,
            s.position,
            Some(PackedLineage::Fast),
            None,
            4
        )
        .is_err()
    );
    assert!(
        check(
            Some(s),
            s.owner,
            s.position,
            s.lineage,
            Some(FastPrecision::Half),
            4
        )
        .is_err()
    );
    assert!(check(Some(s), s.owner, s.position, s.lineage, None, 8).is_err());
    let mut other_history = s;
    other_history.generation += 1;
    other_history.position = s.position;
    assert!(
        s.validate(Some(other_history), s.owner, s.position, s.lineage, None, 4)
            .is_err()
    );
}

#[test]
fn cpu_append_lifecycle_and_explicit_retention() {
    for phase in [
        Phase::Ready,
        Phase::Running,
        Phase::Failed,
        Phase::Fatal,
        Phase::Finished,
    ] {
        assert!(phase.retained_inputs(0, 8).is_err());
    }
    for arithmetic in [
        Glm5NextVerificationArithmetic::Serial,
        Glm5NextVerificationArithmetic::Packed {
            lineage: PackedLineage::Exact,
            precision: None,
        },
        Glm5NextVerificationArithmetic::Packed {
            lineage: PackedLineage::Fast,
            precision: Some(FastPrecision::Half),
        },
    ] {
        let phase = Phase::Verified(arithmetic);
        assert!(phase.can_restore());
        assert!(phase.poisons_on_drop());
        for count in 0..=8 {
            assert_eq!(phase.retained_inputs(count, 8).unwrap(), arithmetic);
        }
        assert!(phase.retained_inputs(9, 8).is_err());
        assert!(phase.retained_inputs(usize::MAX, 8).is_err());
        assert_eq!(phase.retained_inputs(0, 1).unwrap(), arithmetic);
        assert_eq!(phase.retained_inputs(1, 1).unwrap(), arithmetic);
    }
    assert!(Phase::Ready.can_restore());
    assert!(Phase::Failed.can_restore());
    assert!(!Phase::Running.can_restore());
    assert!(!Phase::Fatal.can_restore());
    assert!(!Phase::Finished.can_restore());
    assert!(!Phase::Ready.poisons_on_drop());
    assert!(!Phase::Finished.poisons_on_drop());
    assert!(Phase::Running.poisons_on_drop());
    assert!(Phase::Failed.poisons_on_drop());
    assert!(Phase::Fatal.poisons_on_drop());
}

#[test]
fn cpu_execution_failures_only_recover_completed_kernel_validation() {
    let validation = Glm5NextMetalError::KernelValidation {
        stage: "packed sparse selection",
        block: 3,
        row: Some(4),
        status: -1,
    };
    assert_eq!(Phase::after_execution_error(&validation), Phase::Failed);
    let completion = Glm5NextMetalError::Metal(MetalError::CommandBufferFailed {
        status: "Error".into(),
        error: "injected discarded command buffer".into(),
    });
    assert_eq!(Phase::after_execution_error(&completion), Phase::Fatal);
    assert!(!Phase::after_execution_error(&completion).can_restore());
    assert_eq!(
        Phase::after_execution_error(&Glm5NextMetalError::Invalid(
            "unknown execution failure".into()
        )),
        Phase::Fatal
    );
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn cpu_shared_accept_count_leaves_terminal_and_next_carry_unconsumed() {
    use crate::metal_dflash::{DFlashNextCarry, decide_dflash_accept_prefix};
    use crate::sampling::{Sampler, SamplingConfig};

    let inputs = [0, 1, 2, 3];
    for (winners, budget, stop, retained, next) in [
        ([1, 2, 3, 4], 8, vec![], 4, Some(DFlashNextCarry::Bonus(4))),
        (
            [1, 4, 3, 4],
            8,
            vec![],
            2,
            Some(DFlashNextCarry::Correction(4)),
        ),
        ([1, 2, 3, 4], 8, vec![2], 2, None),
        ([1, 2, 3, 4], 2, vec![], 2, None),
        ([1, 2, 3, 4], 8, vec![0], 0, None),
        ([1, 2, 3, 4], 0, vec![], 0, None),
    ] {
        let mut logits = vec![-10.0; inputs.len() * 5];
        for (row, winner) in winners.into_iter().enumerate() {
            logits[row * 5 + winner] = 10.0;
        }
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        })
        .unwrap();
        let decision =
            decide_dflash_accept_prefix(&inputs, &logits, 5, None, &mut sampler, budget, &stop)
                .unwrap();
        assert_eq!(decision.retain_input_count, retained);
        assert_eq!(decision.next_carry, next);
        Phase::Verified(Glm5NextVerificationArithmetic::Serial)
            .retained_inputs(decision.retain_input_count, inputs.len())
            .unwrap();
        if let Some(terminal) = decision
            .accepted_drafts
            .last()
            .filter(|_| decision.stop_reason.is_some())
        {
            assert_eq!(inputs[retained], *terminal);
        }
    }
}

fn pointers(checkpoint: &Glm5NextAppendCheckpoint) -> Vec<(usize, u64)> {
    checkpoint
        .regions
        .iter()
        .flat_map(|r| match r {
            Regions::Kda { conv, state } => vec![conv, state],
            Regions::Mla { pending } => vec![pending],
        })
        .map(|t| (t.buffer.contents().as_ptr() as usize, t.n_bytes()))
        .collect()
}

#[test]
#[ignore = "GLM compact checkpoint oracle: model weights, production lease, MTL_DEBUG_LAYER=1; no automatic run"]
fn metal_checkpoint_all_ring_phases_pool_completion_and_sparse_frontier() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required()).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, 2080, 128).unwrap();
    let taps = [42, 5, 24];
    let mut bank = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &taps, 8, 8).unwrap(),
    )
    .unwrap();
    let mut checkpoint = Glm5NextAppendCheckpoint::new(&ctx, &session).unwrap();
    let storage = pointers(&checkpoint);
    let priced = checkpoint.priced_bytes();
    let ledger = *session.ledger();
    let prompt: Vec<u32> = (0..2052).map(|i| 1 + (i % 127) as u32).collect();
    let inputs = [8, 9, 10, 11, 12, 13, 14, 15];
    let continuation = [41, 42, 43, 44];
    let v = weights.config.vocab_size as usize;
    let features = taps.len() * weights.config.hidden_size as usize;

    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        session.reset().unwrap();
        session.set_packed_lineage(lineage).unwrap();
        let arithmetic = Glm5NextVerificationArithmetic::Packed {
            lineage,
            precision: session.effective_fast_precision(),
        };
        for position in [100, 101, 102, 103, 2048, 2049, 2050, 2051] {
            session.reset().unwrap();
            session.prefill_packed(&ctx, &prompt[..position]).unwrap();
            let baseline = session.capture_snapshot().unwrap();
            let mut serial_logits = Vec::new();
            let mut serial_features = Vec::new();
            for &token in &inputs {
                let row = session
                    .forward_with_captures(&ctx, token, &mut bank)
                    .unwrap();
                serial_logits.extend(read_f32(row.logits().unwrap()).unwrap());
                serial_features.extend(read_f32(row.features().unwrap()).unwrap());
            }
            session.restore_snapshot(&baseline).unwrap();
            let packed_last = session.prefill_packed(&ctx, &inputs).unwrap();
            let packed_state = session.capture_snapshot().unwrap();
            session.restore_snapshot(&baseline).unwrap();
            let packed_rows = session
                .verify_packed_with_captures(&ctx, &inputs, &mut bank)
                .unwrap();
            let packed_logits = read_f32(packed_rows.logits().unwrap()).unwrap();
            let packed_features = read_f32(packed_rows.features().unwrap()).unwrap();
            assert_eq!(bits(&packed_logits[7 * v..]), bits(&packed_last));
            assert!(
                session
                    .capture_snapshot()
                    .unwrap()
                    .same_state(&packed_state)
            );
            let counts: Vec<usize> = if position == 100 {
                (0..=8).collect()
            } else {
                vec![0, 1, 4, 8]
            };
            for count in counts {
                session.restore_snapshot(&baseline).unwrap();
                if count == inputs.len() {
                    session.prefill_packed(&ctx, &inputs).unwrap();
                } else if count > 0 {
                    session.prefill(&ctx, &inputs[..count]).unwrap();
                }
                let accepted_state = session.capture_snapshot().unwrap();
                let mut future = Vec::new();
                for &token in &continuation {
                    future.extend(session.forward(&ctx, token).unwrap());
                }
                let future_state = session.capture_snapshot().unwrap();
                session.restore_snapshot(&baseline).unwrap();
                {
                    let mut tx = session
                        .begin_append(&ctx, &mut checkpoint, &mut bank, &inputs)
                        .unwrap();
                    assert_eq!(tx.base_position(), position);
                    let packed = tx.verify_packed().unwrap();
                    assert_eq!(packed.arithmetic(), arithmetic);
                    assert_eq!(packed.input_positions(), position..position + 8);
                    assert_eq!(
                        bits(&read_f32(packed.logits()).unwrap()),
                        bits(&packed_logits)
                    );
                    // A full commit must not submit restore/replay commands or wait.
                    let no_completion =
                        (count == inputs.len()).then(crate::metal::fail_next_completion_check);
                    let committed = tx.commit_retained_inputs(count).unwrap();
                    drop(no_completion);
                    assert_eq!(committed.verification_arithmetic(), arithmetic);
                    let retained_arithmetic = if count == 0 {
                        None
                    } else if count == inputs.len() {
                        Some(arithmetic)
                    } else {
                        Some(Glm5NextVerificationArithmetic::Serial)
                    };
                    assert_eq!(committed.retained_arithmetic(), retained_arithmetic);
                    if count == 0 {
                        assert!(committed.rows().is_none());
                    } else {
                        let rows = committed.rows().unwrap();
                        assert_eq!(rows.input_positions(), position..position + count);
                        assert_eq!(rows.feature_positions(), position..position + count);
                        let (expected_features, expected_logits) = if count == inputs.len() {
                            assert_eq!(rows.logit_positions(), position..position + count);
                            (&packed_features[..], &packed_logits[..])
                        } else {
                            assert_eq!(
                                rows.logit_positions(),
                                position + count - 1..position + count
                            );
                            (
                                &serial_features[..count * features],
                                &serial_logits[(count - 1) * v..count * v],
                            )
                        };
                        assert_eq!(
                            bits(&read_f32(rows.features().unwrap()).unwrap()),
                            bits(expected_features)
                        );
                        assert_eq!(
                            bits(&read_f32(rows.logits().unwrap()).unwrap()),
                            bits(expected_logits)
                        );
                    }
                    assert!(
                        tx.rollback().is_err(),
                        "committed checkpoint must be invalid"
                    );
                    assert!(tx.commit_retained_inputs(0).is_err());
                }
                assert_eq!(
                    session.position(),
                    position + count,
                    "bonus/replacement must remain unconsumed"
                );
                assert!(!session.is_poisoned());
                assert!(
                    session
                        .capture_snapshot()
                        .unwrap()
                        .same_state(&accepted_state)
                );
                let mut actual_future = Vec::new();
                for &token in &continuation {
                    actual_future.extend(session.forward(&ctx, token).unwrap());
                }
                assert_eq!(bits(&actual_future), bits(&future));
                assert!(
                    session
                        .capture_snapshot()
                        .unwrap()
                        .same_state(&future_state)
                );
                assert_eq!(
                    pointers(&checkpoint),
                    storage,
                    "checkpoint must reuse device storage"
                );
                assert_eq!(checkpoint.priced_bytes(), priced);
                assert_eq!(*session.ledger(), ledger);
                assert!(checkpoint.active.is_none());
            }
            // Explicit serial fallback is checked separately, never inferred from a
            // loose packed/scalar tolerance or performed after drawing acceptance.
            session.restore_snapshot(&baseline).unwrap();
            {
                let mut tx = session
                    .begin_append(&ctx, &mut checkpoint, &mut bank, &inputs)
                    .unwrap();
                tx.verify_packed().unwrap();
                let serial = tx.verify_serial().unwrap();
                assert_eq!(
                    bits(&read_f32(serial.logits()).unwrap()),
                    bits(&serial_logits)
                );
                let committed = tx.commit_retained_inputs(8).unwrap();
                assert_eq!(
                    committed.retained_arithmetic(),
                    Some(Glm5NextVerificationArithmetic::Serial)
                );
                assert_eq!(
                    bits(&read_f32(committed.rows().unwrap().features().unwrap()).unwrap()),
                    bits(&serial_features)
                );
            }
        }
    }
}

#[test]
#[ignore = "GLM append failure/lifecycle gate: model weights, production lease, MTL_DEBUG_LAYER=1; no automatic run"]
fn metal_checkpoint_failure_recovery_and_stale_foreign_refusals() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required()).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, 2080, 128).unwrap();
    session.set_packed_lineage(PackedLineage::Exact).unwrap();
    session.prefill_packed(&ctx, &[1; 2048]).unwrap();
    let baseline = session.capture_snapshot().unwrap();
    let mut bank = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &[5], 8, 8).unwrap(),
    )
    .unwrap();
    let mut checkpoint = Glm5NextAppendCheckpoint::new(&ctx, &session).unwrap();
    let inputs = [2; 8];

    {
        let _failure = crate::metal::fail_next_completion_check();
        assert!(
            session
                .begin_append(&ctx, &mut checkpoint, &mut bank, &inputs)
                .is_err()
        );
    }
    assert!(
        checkpoint.active.is_none(),
        "partial capture must never become restorable"
    );
    assert!(!session.is_poisoned());
    assert!(session.capture_snapshot().unwrap().same_state(&baseline));

    session.corrupt_sparse_row = Some(4);
    {
        let mut tx = session
            .begin_append(&ctx, &mut checkpoint, &mut bank, &inputs)
            .unwrap();
        assert!(matches!(
            tx.verify_packed(),
            Err(Glm5NextMetalError::KernelValidation { row: Some(4), .. })
        ));
        assert!(tx.session.is_poisoned());
        assert!(
            !tx.is_fatal(),
            "completed selector validation remains recoverable"
        );
        assert_eq!(tx.session.position(), 2048);
        assert!(tx.commit_retained_inputs(0).is_err());
        let scalar = tx.verify_serial().unwrap();
        assert_eq!(scalar.input_positions(), 2048..2056);
        assert!(
            read_f32(scalar.logits())
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
        tx.rollback().unwrap();
        assert!(tx.session.capture_snapshot().unwrap().same_state(&baseline));
    }
    assert!(!session.is_poisoned());

    let old_stamp;
    {
        let mut tx = session
            .begin_append(&ctx, &mut checkpoint, &mut bank, &inputs)
            .unwrap();
        old_stamp = tx.stamp;
        tx.verify_serial().unwrap();
    }
    assert!(
        session.is_poisoned(),
        "dropping tentative state must not commit it"
    );
    assert!(checkpoint.active.is_none());
    assert!(
        session
            .begin_append(&ctx, &mut checkpoint, &mut bank, &inputs)
            .is_err()
    );
    session.restore_snapshot(&baseline).unwrap();
    {
        let mut tx = session
            .begin_append(&ctx, &mut checkpoint, &mut bank, &inputs)
            .unwrap();
        let current = tx.stamp;
        tx.stamp = old_stamp;
        assert!(matches!(
            tx.rollback(),
            Err(Glm5NextMetalError::AppendCheckpointMismatch(_))
        ));
        assert!(!tx.session.is_poisoned());
        assert!(tx.session.capture_snapshot().unwrap().same_state(&baseline));
        tx.stamp = current;
    }

    let mut other = Glm5NextSession::new(&ctx, &weights, 16).unwrap();
    let other_before = other.capture_snapshot().unwrap();
    assert!(matches!(
        other.begin_append(&ctx, &mut checkpoint, &mut bank, &inputs),
        Err(Glm5NextMetalError::AppendCheckpointMismatch(_))
    ));
    assert!(other.capture_snapshot().unwrap().same_state(&other_before));
    drop(other);

    session.reset().unwrap();
    session.set_packed_lineage(PackedLineage::Fast).unwrap();
    let reset_state = session.capture_snapshot().unwrap();
    {
        let mut tx = session
            .begin_append(&ctx, &mut checkpoint, &mut bank, &inputs)
            .unwrap();
        assert_eq!(tx.base_position(), 0);
        let current = tx.stamp;
        tx.stamp = old_stamp;
        assert!(tx.rollback().is_err());
        assert!(
            tx.session
                .capture_snapshot()
                .unwrap()
                .same_state(&reset_state)
        );
        tx.stamp = current;
        assert_eq!(
            tx.verify_packed().unwrap().arithmetic(),
            Glm5NextVerificationArithmetic::Packed {
                lineage: PackedLineage::Fast,
                precision: Some(FastPrecision::Half),
            }
        );
        tx.verify_serial().unwrap();
        let committed = tx.commit_retained_inputs(1).unwrap();
        assert_eq!(committed.rows().unwrap().feature_positions(), 0..1);
    }
    assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
    assert_eq!(
        session.effective_fast_precision(),
        Some(FastPrecision::Half)
    );
    assert_eq!(session.position(), 1);
    assert!(!session.is_poisoned());
}

#[test]
#[ignore = "GLM fatal completion failure gate: model weights, production lease, MTL_DEBUG_LAYER=1; no automatic run"]
fn metal_checkpoint_command_failures_verify_restore_replay_are_terminal() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required()).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, 64, 8).unwrap();
    let mut bank = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &[5], 8, 8).unwrap(),
    )
    .unwrap();
    let mut checkpoint = Glm5NextAppendCheckpoint::new(&ctx, &session).unwrap();
    let inputs = [2; 8];
    for failure in ["packed verify", "serial verify", "restore", "replay"] {
        session.reset().unwrap();
        session.prefill_packed(&ctx, &[1; 3]).unwrap();
        {
            let mut tx = session
                .begin_append(&ctx, &mut checkpoint, &mut bank, &inputs)
                .unwrap();
            let error = match failure {
                "packed verify" => {
                    let _failure = crate::metal::fail_next_completion_check();
                    tx.verify_packed()
                        .err()
                        .expect("verification completion must fail")
                }
                "serial verify" => {
                    let _failure = crate::metal::fail_next_completion_check();
                    tx.verify_serial()
                        .err()
                        .expect("verification completion must fail")
                }
                "restore" => {
                    tx.verify_packed().unwrap();
                    let _failure = crate::metal::fail_next_completion_check();
                    tx.commit_retained_inputs(4)
                        .err()
                        .expect("restore completion must fail")
                }
                "replay" => {
                    tx.verify_packed().unwrap();
                    // Restore succeeds; fail the first replay command, not the blit.
                    tx.fail_replay_completion = true;
                    tx.commit_retained_inputs(4)
                        .err()
                        .expect("replay completion must fail")
                }
                _ => unreachable!(),
            };
            assert!(matches!(
                error,
                Glm5NextMetalError::Metal(MetalError::CommandBufferFailed { .. })
            ));
            assert!(tx.is_fatal(), "{failure}");
            assert!(tx.session.is_poisoned());
            let position = tx.session.position();
            assert_eq!(position, if failure == "restore" { 11 } else { 3 });
            // With injection disarmed, another successful copy MUST NOT repair
            // the transaction. Every public operation refuses before any writes.
            assert!(matches!(tx.rollback(), Err(Glm5NextMetalError::Poisoned)));
            assert!(matches!(
                tx.verify_serial(),
                Err(Glm5NextMetalError::Poisoned)
            ));
            assert!(matches!(
                tx.verify_packed(),
                Err(Glm5NextMetalError::Poisoned)
            ));
            assert!(matches!(
                tx.commit_retained_inputs(0),
                Err(Glm5NextMetalError::Poisoned)
            ));
            assert!(matches!(
                tx.commit_retained_inputs(8),
                Err(Glm5NextMetalError::Poisoned)
            ));
            assert_eq!(tx.session.position(), position);
            assert!(tx.session.is_poisoned());
        }
        assert!(session.is_poisoned());
        assert!(checkpoint.active.is_none());
        assert!(
            session
                .begin_append(&ctx, &mut checkpoint, &mut bank, &inputs)
                .is_err()
        );
    }
}
