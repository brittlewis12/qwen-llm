use super::*;
use qwen_llm::glm5_next_metal::{Glm5NextDFlashCarry, Glm5NextDFlashCosts};
use std::cell::RefCell;

fn packet(
    tokens: &[i32],
    carry: Option<Glm5NextDFlashCarry>,
    stop_reason: Option<Glm5NextDFlashStop>,
    retain_input_count: usize,
) -> Glm5NextDFlashPacket {
    Glm5NextDFlashPacket {
        tokens: tokens.to_vec(),
        emitted_tokens: tokens.to_vec(),
        next_carry: carry,
        stop_reason,
        accepted_drafts: 0,
        retain_input_count,
        verification_arithmetic: None,
        retained_arithmetic: None,
        anchor_only: false,
        costs: Glm5NextDFlashCosts::default(),
    }
}

fn eos_packet(
    tokens: &[i32],
    carry: Option<Glm5NextDFlashCarry>,
    terminal_token: i32,
    retain_input_count: usize,
) -> Glm5NextDFlashPacket {
    let mut packet = packet(
        tokens,
        carry,
        Some(Glm5NextDFlashStop::Eos),
        retain_input_count,
    );
    packet.emitted_tokens.push(terminal_token);
    packet
}

#[test]
fn short_prompt_keeps_full_block_width_unless_request_itself_is_shorter() {
    assert_eq!(packed_rows(1, 100, 16).unwrap(), 16);
    assert_eq!(packed_rows(512, 1000, 16).unwrap(), 512);
    assert_eq!(packed_rows(1, 1, 16).unwrap(), 1);
    assert_eq!(packed_rows(1, 8, 16).unwrap(), 8);
    assert!(packed_rows(1, 0, 16).is_err());
    assert!(packed_rows(1, 10, 0).is_err());
}

#[test]
fn scratch_geometry_covers_seed_verify_logits_and_t_minus_one_appends() {
    assert_eq!(
        request_geometry(1, 100, 100, 16, 16, 64).unwrap(),
        RequestGeometry {
            draft_capacity: 100,
            feature_rows: 16,
            verify_rows: 16
        }
    );
    assert_eq!(
        request_geometry(1000, 10, 1009, 16, 512, 64).unwrap(),
        RequestGeometry {
            draft_capacity: 73,
            feature_rows: 64,
            verify_rows: 16
        }
    );
    assert_eq!(
        request_geometry(1, 1, 1, 16, 1, 64).unwrap(),
        RequestGeometry {
            draft_capacity: 1,
            feature_rows: 1,
            verify_rows: 1
        }
    );
    assert!(request_geometry(0, 10, 100, 16, 16, 64).is_err());
    assert!(request_geometry(101, 10, 100, 16, 16, 64).is_err());
    assert!(request_geometry(1, 10, 100, 16, 0, 64).is_err());
}

#[test]
fn head_residency_and_request_scratch_share_ordinary_working_set_budget() {
    use qwen_llm::metal::MetalMemorySignals;
    const GIB: u64 = 1 << 30;
    let signals = MetalMemorySignals {
        recommended_max_bytes: 118 * GIB,
        current_allocated_bytes: 2 * GIB,
        process_limit_remaining_bytes: Some(200 * GIB),
    };
    let target_peak = 113 * GIB;
    assert!(
        evaluate_metal_memory_admission(target_peak, 0, signals, true)
            .refusal()
            .is_none()
    );
    let aggregate = aggregate_bytes(&[target_peak, 4 * GIB]).unwrap();
    assert!(
        evaluate_metal_memory_admission(aggregate, 0, signals, true)
            .refusal()
            .is_some()
    );
    assert_eq!(aggregate_bytes(&[1, 2, 3]).unwrap(), 6);
    assert!(aggregate_bytes(&[u64::MAX, 1]).is_err());
}

#[test]
fn target_sampling_is_unchanged_and_proposal_stream_is_temperature_only() {
    let target = SamplingConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.95,
        min_p: 0.05,
        seed: 7,
    };
    let proposal = proposal_config(target, true).unwrap().unwrap();
    assert_eq!(proposal.temperature, target.temperature);
    assert_eq!(
        (proposal.top_k, proposal.top_p, proposal.min_p),
        (0, 1.0, 0.0)
    );
    assert_eq!(
        proposal.seed,
        target.seed ^ super::super::super::dflash::DFLASH_PROPOSAL_SEED_XOR
    );
    assert!(
        proposal_config(target, false)
            .unwrap_err()
            .to_string()
            .contains("DFlash2 selector")
    );
    assert_eq!(
        proposal_config(
            SamplingConfig {
                temperature: 0.0,
                ..target
            },
            false
        )
        .unwrap(),
        None
    );
    assert_eq!(
        (target.top_k, target.top_p, target.min_p, target.seed),
        (40, 0.95, 0.05, 7)
    );
}

#[test]
fn initial_eos_zero_budget_and_one_token_never_step() {
    for (first, budget, expected, expected_callbacks, stop) in [
        (
            eos_packet(&[], Some(Glm5NextDFlashCarry::Initial(99)), 99, 0),
            10,
            vec![99],
            vec![],
            StopReason::Eos,
        ),
        (
            eos_packet(&[], Some(Glm5NextDFlashCarry::Initial(97)), 97, 0),
            1,
            vec![97],
            vec![],
            StopReason::Eos,
        ),
        (
            packet(&[], None, Some(Glm5NextDFlashStop::TokenLimit), 0),
            0,
            vec![],
            vec![],
            StopReason::TokenLimit,
        ),
        (
            packet(
                &[7],
                Some(Glm5NextDFlashCarry::Initial(7)),
                Some(Glm5NextDFlashStop::TokenLimit),
                0,
            ),
            1,
            vec![7],
            vec![7],
            StopReason::TokenLimit,
        ),
    ] {
        let mut emitted = Vec::new();
        let result = collect_packets(
            first,
            budget,
            |t| {
                emitted.push(t);
                Ok(())
            },
            |_| panic!("terminal carry was forwarded"),
        )
        .unwrap();
        assert_eq!(result.tokens, expected);
        assert_eq!(emitted, expected_callbacks);
        assert_eq!(result.stop_reason, stop);
        assert_eq!(result.transitions, 0);
        assert_eq!(result.transitions, result.tokens.len().saturating_sub(1));
        assert_eq!(result.first_transition_ms, None);
        assert_eq!(
            result.first_token_callback_ms.is_some(),
            !expected_callbacks.is_empty()
        );
        assert_eq!(result.first_token_ready_ms.is_some(), budget > 0);
    }
}

#[test]
fn correction_and_bonus_publish_once_before_next_step_and_leave_final_token_unconsumed() {
    let emitted = RefCell::new(Vec::new());
    let mut calls = 0;
    let result = collect_packets(
        packet(&[1], Some(Glm5NextDFlashCarry::Initial(1)), None, 0),
        6,
        |t| {
            emitted.borrow_mut().push(t);
            Ok(())
        },
        |carry| {
            calls += 1;
            match calls {
                1 => {
                    assert_eq!(carry, 1);
                    assert_eq!(*emitted.borrow(), [1]);
                    Ok(packet(
                        &[2, 3],
                        Some(Glm5NextDFlashCarry::Correction(3)),
                        None,
                        2,
                    ))
                }
                2 => {
                    assert_eq!(carry, 3);
                    assert_eq!(*emitted.borrow(), [1, 2, 3]);
                    Ok(packet(
                        &[4, 5, 6],
                        Some(Glm5NextDFlashCarry::Bonus(6)),
                        Some(Glm5NextDFlashStop::TokenLimit),
                        3,
                    ))
                }
                _ => panic!("terminal bonus was forwarded"),
            }
        },
    )
    .unwrap();
    assert_eq!(result.tokens, [1, 2, 3, 4, 5, 6]);
    assert_eq!(*emitted.borrow(), result.tokens);
    assert_eq!(result.transitions, result.tokens.len() - 1);
    assert_eq!(result.stop_reason, StopReason::TokenLimit);
    assert_eq!(calls, 2);
}

#[test]
fn eos_after_accepted_prefix_is_hidden_and_does_not_trigger_another_step() {
    for (terminal, next, budget) in [
        (97, None, 4),
        (99, None, 20),
        (97, Some(Glm5NextDFlashCarry::Correction(97)), 4),
        (99, Some(Glm5NextDFlashCarry::Bonus(99)), 20),
    ] {
        let mut calls = 0;
        let mut emitted = Vec::new();
        let result = collect_packets(
            packet(&[1], Some(Glm5NextDFlashCarry::Initial(1)), None, 0),
            budget,
            |t| {
                emitted.push(t);
                Ok(())
            },
            |_| {
                calls += 1;
                assert_eq!(calls, 1);
                Ok(eos_packet(&[2, 3], next, terminal, 3))
            },
        )
        .unwrap();
        assert_eq!(emitted, [1, 2, 3]);
        assert_eq!(result.tokens, [1, 2, 3, terminal]);
        assert_eq!(result.transitions, 3);
        assert_eq!(result.transitions, result.tokens.len() - 1);
        assert_eq!(result.stop_reason, StopReason::Eos);
    }
}

#[test]
fn callback_failure_stops_publication_without_stepping_again() {
    for fail_at in [1, 3] {
        let mut calls = 0;
        let mut emitted = Vec::new();
        let error = collect_packets(
            packet(&[1], Some(Glm5NextDFlashCarry::Initial(1)), None, 0),
            10,
            |t| {
                ensure!(t != fail_at, "writer failed");
                emitted.push(t);
                Ok(())
            },
            |_| {
                calls += 1;
                assert_eq!(calls, 1);
                Ok(packet(
                    &[2, 3, 4],
                    Some(Glm5NextDFlashCarry::Bonus(4)),
                    None,
                    3,
                ))
            },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "writer failed");
        assert_eq!(calls, usize::from(fail_at != 1));
        assert_eq!(emitted, if fail_at == 1 { vec![] } else { vec![1, 2] });
    }
}

#[test]
fn capacity_and_adapter_errors_are_not_reported_as_successful_token_limits() {
    let error = collect_packets(
        packet(&[], None, Some(Glm5NextDFlashStop::Capacity), 0),
        10,
        |_| Ok(()),
        |_| panic!("capacity stop was stepped"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("exhausted target capacity"));
    let error = collect_packets(
        packet(&[1], Some(Glm5NextDFlashCarry::Initial(1)), None, 0),
        10,
        |_| Ok(()),
        |_| bail!("adapter failed"),
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "adapter failed");
}

#[test]
fn malformed_packet_cannot_overrun_budget_or_step_without_a_carry() {
    for first in [
        packet(&[1, 2], None, Some(Glm5NextDFlashStop::TokenLimit), 0),
        packet(&[], None, None, 0),
        packet(&[1], Some(Glm5NextDFlashCarry::Initial(1)), None, 0),
    ] {
        assert!(
            collect_packets(
                first,
                1,
                |_| Ok(()),
                |_| panic!("invalid packet was stepped")
            )
            .is_err()
        );
    }
}

#[test]
fn eos_consumes_budget_even_though_it_has_no_callback() {
    for budget in [0, 2] {
        let first = if budget == 0 {
            eos_packet(&[], None, 99, 0)
        } else {
            eos_packet(&[1, 2], None, 97, 2)
        };
        let error = collect_packets(
            first,
            budget,
            |_| panic!("over-budget packet must not emit"),
            |_| panic!("over-budget packet must not step"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("exceeds the output budget"));
    }
}

#[test]
fn eos_requires_actual_terminal_token_and_never_duplicates_it() {
    let mut nonterminal = packet(&[], Some(Glm5NextDFlashCarry::Initial(99)), None, 0);
    nonterminal.emitted_tokens.push(99);
    let mut capped = packet(&[1], None, Some(Glm5NextDFlashStop::TokenLimit), 0);
    capped.emitted_tokens.push(99);
    let mut wrong_prefix = eos_packet(&[1, 2], None, 99, 2);
    wrong_prefix.tokens.swap(0, 1);
    let mut extra_history = eos_packet(&[1], None, 99, 2);
    extra_history.emitted_tokens.insert(1, 2);
    let mut extra_callback = eos_packet(&[1], None, 99, 0);
    extra_callback.tokens.push(99);
    for first in [
        // Even a terminal carry is not a substitute for the adapter's explicit ID.
        packet(
            &[],
            Some(Glm5NextDFlashCarry::Initial(99)),
            Some(Glm5NextDFlashStop::Eos),
            0,
        ),
        packet(&[], None, Some(Glm5NextDFlashStop::Eos), 0),
        eos_packet(&[1, 99], None, 99, 2),
        nonterminal,
        capped,
        wrong_prefix,
        extra_history,
        extra_callback,
    ] {
        assert!(
            collect_packets(
                first,
                10,
                |_| panic!("invalid terminal packet must not emit"),
                |_| panic!("invalid terminal packet must not step"),
            )
            .is_err()
        );
    }
}

#[test]
fn eos_and_cap_both_require_t_minus_one_committed_inputs() {
    for first in [
        eos_packet(&[], None, 99, 1),
        packet(&[1], None, Some(Glm5NextDFlashStop::TokenLimit), 1),
    ] {
        let error = collect_packets(
            first,
            1,
            |_| Ok(()),
            |_| panic!("terminal packet was stepped"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("generated tokens minus one"));
    }
    let error = collect_packets(
        packet(&[1], None, Some(Glm5NextDFlashStop::TokenLimit), 0),
        2,
        |_| Ok(()),
        |_| panic!("terminal packet was stepped"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("must exhaust the output budget"));
}

#[test]
fn initial_selection_uses_adapter_measurement_not_prefill_residual() {
    use std::time::Duration;
    let mut first = packet(&[1], None, Some(Glm5NextDFlashStop::TokenLimit), 0);
    first.costs.selection = Duration::from_millis(5);
    first.costs.prefill = Duration::from_millis(100);
    first.costs.append = Duration::from_millis(20);
    first.costs.total = Duration::from_millis(140);
    let result = collect_packets(
        first,
        1,
        |_| Ok(()),
        |_| panic!("terminal packet was stepped"),
    )
    .unwrap();
    assert_eq!(result.first_token_selection_ms, 5.0);
    assert_eq!(result.first_token_ready_ms, Some(5.0));
    assert!(result.first_token_callback_ms.unwrap() >= 5.0);
    assert!(result.wall_ms >= 5.0);

    let zero = collect_packets(
        packet(&[], None, Some(Glm5NextDFlashStop::TokenLimit), 0),
        0,
        |_| panic!("zero budget must not emit"),
        |_| panic!("zero budget must not step"),
    )
    .unwrap();
    assert_eq!(zero.first_token_selection_ms, 0.0);
    assert_eq!(zero.first_token_ready_ms, None);
}
