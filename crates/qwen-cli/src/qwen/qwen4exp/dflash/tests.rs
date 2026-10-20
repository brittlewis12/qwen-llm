use super::*;
use std::cell::Cell;

fn packet(tokens: &[i32], finish: Option<Qwen4ExpDFlashFinish>) -> Qwen4ExpDFlashPacket {
    Qwen4ExpDFlashPacket {
        tokens: tokens.to_vec(),
        finish,
    }
}

#[test]
fn initial_selection_moves_from_prefill_to_generation_without_double_counting() {
    for (stop_reason, callback) in [(StopReason::TokenLimit, Some(2.0)), (StopReason::Eos, None)] {
        let mut generation = GenerationResult {
            tokens: vec![if stop_reason == StopReason::Eos { 9 } else { 1 }],
            wall_ms: 13.0,
            first_token_selection_ms: 0.0,
            first_token_ready_ms: Some(0.0),
            first_token_callback_ms: callback,
            transitions: 0,
            transition_ms: 0.0,
            first_transition_ms: None,
            stop_reason,
        };
        let prefill_ms = attribute_initial_selection(
            Duration::from_millis(100),
            Duration::from_millis(7),
            &mut generation,
        );
        assert_eq!(prefill_ms, 93.0);
        assert_eq!(generation.wall_ms, 20.0);
        assert_eq!(prefill_ms + generation.wall_ms, 113.0);
        assert_eq!(generation.first_token_selection_ms, 7.0);
        assert_eq!(generation.first_token_ready_ms, Some(7.0));
        assert_eq!(
            generation.first_token_callback_ms,
            callback.map(|ms| ms + 7.0)
        );
        assert_eq!(generation.transition_ms, 0.0);
        assert_eq!(generation.first_transition_ms, None);
        assert_eq!(generation.stop_reason, stop_reason);
    }
}

#[test]
fn tiny_prompts_reserve_effective_verification_width() {
    for (budget, width, extent) in [
        (1, 1, None),
        (2, 1, None),
        (3, 2, Some(2)),
        (17, 16, Some(16)),
        (64, 16, Some(16)),
    ] {
        assert_eq!(verification_width(16, 1, budget, budget).unwrap(), width);
        assert_eq!(packed_extent(1, width, budget, 2051), extent);
        let plan = RequestPlan {
            packed_extent: extent,
            verification_width: width,
        };
        assert!(plan.validate_packed_capacity(extent).is_ok());
        assert_eq!(
            plan.minimum_packed_rows(),
            extent.map_or(0, |_| width.max(2))
        );
        if width > 1 {
            assert!(plan.validate_packed_capacity(Some(width - 1)).is_err());
            assert!(plan.validate_packed_capacity(None).is_err());
        }
    }
}

#[test]
fn long_prefill_keeps_a_packed_floor_even_for_singleton_verification() {
    for (width, floor) in [(1, 2), (8, 8), (10, 10)] {
        let plan = RequestPlan {
            packed_extent: Some(4096),
            verification_width: width,
        };
        assert_eq!(plan.minimum_packed_rows(), floor);
    }
}

#[test]
fn future_selected_verification_reserves_selected_scratch() {
    assert_eq!(packed_extent(2040, 16, 2051, 2051), Some(2040));
    assert_eq!(packed_extent(2040, 16, 2052, 2051), Some(2052));
    assert_eq!(packed_extent(4099, 16, 4100, 2051), Some(4100));
    assert_eq!(packed_extent(2040, 1, 2052, 2051), Some(2040));
}

#[test]
fn request_extent_keeps_t_minus_one_and_nominal_draft_headroom() {
    assert_eq!(verification_width(16, 8, 4, 11).unwrap(), 3);
    assert!(verification_width(16, 8, 4, 10).is_err());
    for (block, prompt, budget) in [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (16, usize::MAX, 2),
        (16, i32::MAX as usize, 1),
    ] {
        assert!(verification_width(block, prompt, budget, usize::MAX).is_err());
    }
}

#[test]
fn profiles_are_rejected_only_for_drafting() {
    for (layer, packed) in [(true, false), (false, true), (true, true)] {
        assert!(validate_profiles(true, layer, packed).is_err());
        assert!(validate_profiles(false, layer, packed).is_ok());
    }
    assert!(validate_profiles(true, false, false).is_ok());
}

#[test]
fn initial_stop_is_retained_without_piece_or_transition() {
    let result = drive_packets(
        packet(&[9], Some(Qwen4ExpDFlashFinish::Eos)),
        1,
        &[8, 9],
        || panic!("terminal initial packet must not step"),
        || Ok(()),
        |_| panic!("stop token must not be rendered"),
    )
    .unwrap();
    assert_eq!(result.tokens, [9]);
    assert_eq!(result.transitions, 0);
    assert_eq!(result.stop_reason, StopReason::Eos);
    assert_eq!(result.first_token_callback_ms, None);
    assert_eq!(result.first_transition_ms, None);
}

#[test]
fn one_token_budget_publishes_without_consuming_carry() {
    let mut pieces = Vec::new();
    let result = drive_packets(
        packet(&[1], Some(Qwen4ExpDFlashFinish::TokenLimit)),
        1,
        &[9],
        || panic!("budget is exhausted"),
        || Ok(()),
        |token| {
            pieces.push(token);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(pieces, [1]);
    assert_eq!(result.tokens, [1]);
    assert_eq!(result.transitions, 0);
    assert_eq!(result.stop_reason, StopReason::TokenLimit);
}

#[test]
fn packets_publish_each_piece_but_retain_stop_only_in_result() {
    let mut steps = [
        packet(&[2, 3], None),
        packet(&[4, 8], Some(Qwen4ExpDFlashFinish::Eos)),
    ]
    .into_iter();
    let mut pieces = Vec::new();
    let ticks = Cell::new(0);
    let result = drive_packets(
        packet(&[1], None),
        8,
        &[8, 9],
        || Ok(steps.next().expect("no step after stop")),
        || {
            ticks.set(ticks.get() + 1);
            Ok(())
        },
        |token| {
            pieces.push(token);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(pieces, [1, 2, 3, 4]);
    assert_eq!(result.tokens, [1, 2, 3, 4, 8]);
    assert_eq!(result.transitions, 4);
    assert_eq!(result.stop_reason, StopReason::Eos);
    assert_eq!(ticks.get(), 9); // Each token, plus before and after each step.
    assert!(result.first_token_callback_ms.is_some());
    assert!(result.first_transition_ms.is_some());
}

#[test]
fn ragged_final_packet_stops_at_budget_without_extra_step() {
    let mut steps = [
        packet(&[2, 3, 4], None),
        packet(&[5], Some(Qwen4ExpDFlashFinish::TokenLimit)),
    ]
    .into_iter();
    let mut pieces = Vec::new();
    let result = drive_packets(
        packet(&[1], None),
        5,
        &[9],
        || Ok(steps.next().expect("no extra step")),
        || Ok(()),
        |token| {
            pieces.push(token);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(pieces, [1, 2, 3, 4, 5]);
    assert_eq!(result.tokens, pieces);
    assert_eq!(result.transitions, 4);
    assert_eq!(result.stop_reason, StopReason::TokenLimit);
}

#[test]
fn cancellation_after_piece_prevents_next_step() {
    let cancelled = Cell::new(false);
    let result = drive_packets(
        packet(&[1], None),
        5,
        &[9],
        || panic!("cancelled publication must not step"),
        || {
            ensure!(!cancelled.get(), "cancelled");
            Ok(())
        },
        |_| {
            cancelled.set(true);
            Ok(())
        },
    );
    assert_eq!(result.unwrap_err().to_string(), "cancelled");
}

#[test]
fn cancellation_between_committed_packet_pieces_stops_publication() {
    let cancelled = Cell::new(false);
    let mut pieces = Vec::new();
    let result = drive_packets(
        packet(&[1], None),
        5,
        &[9],
        || Ok(packet(&[2, 3], None)),
        || {
            ensure!(!cancelled.get(), "cancelled");
            Ok(())
        },
        |token| {
            pieces.push(token);
            cancelled.set(token == 2);
            Ok(())
        },
    );
    assert!(result.is_err());
    assert_eq!(pieces, [1, 2]);
}

#[test]
fn adapter_error_does_not_publish_uncommitted_tokens() {
    let mut pieces = Vec::new();
    let result = drive_packets(
        packet(&[1], None),
        4,
        &[9],
        || bail!("verify failed"),
        || Ok(()),
        |token| {
            pieces.push(token);
            Ok(())
        },
    );
    assert_eq!(result.unwrap_err().to_string(), "verify failed");
    assert_eq!(pieces, [1]);
}

#[test]
fn output_failure_does_not_run_more_target_work() {
    let result = drive_packets(
        packet(&[1], None),
        4,
        &[9],
        || panic!("broken output must not step"),
        || Ok(()),
        |_| bail!("broken pipe"),
    );
    assert_eq!(result.unwrap_err().to_string(), "broken pipe");
}

#[test]
fn malformed_packets_fail_before_publishing_any_of_their_tokens() {
    for bad in [
        packet(&[], None),
        packet(&[2, 3, 4, 5], Some(Qwen4ExpDFlashFinish::TokenLimit)),
        packet(&[9, 3], Some(Qwen4ExpDFlashFinish::Eos)),
        packet(&[2], Some(Qwen4ExpDFlashFinish::Eos)),
        packet(&[9], None),
        packet(&[2], Some(Qwen4ExpDFlashFinish::Cancelled)),
    ] {
        let mut bad = Some(bad);
        let mut pieces = Vec::new();
        assert!(
            drive_packets(
                packet(&[1], None),
                4,
                &[9],
                || Ok(bad.take().expect("must fail immediately")),
                || Ok(()),
                |token| {
                    pieces.push(token);
                    Ok(())
                },
            )
            .is_err()
        );
        assert_eq!(pieces, [1]);
    }
}
