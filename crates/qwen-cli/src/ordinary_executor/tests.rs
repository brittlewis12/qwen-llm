use super::*;
use qwen_llm::sampling::{Sampler, SamplingConfig};
use std::cell::{Cell, RefCell};

#[test]
fn prefill_preserves_restored_prefix_and_last_step_output_without_extra_work() {
    let mut spans = Vec::new();
    let result = prefill(
        2,
        8,
        |position| {
            let (next, output) = match position {
                2 => (5, None),
                5 => (7, Some("intermediate")),
                7 => (8, Some("final")),
                _ => panic!("unexpected position"),
            };
            spans.push((position, next));
            Ok((next, output))
        },
        || "invalid",
    )
    .unwrap();
    assert_eq!(spans, [(2, 5), (5, 7), (7, 8)]);
    assert_eq!(result, Some("final"));
    for end in [0, 8] {
        assert_eq!(
            prefill::<(), _>(end, end, |_| panic!("already restored"), || "invalid").unwrap(),
            None
        );
    }
    assert_eq!(
        prefill(
            0,
            2,
            |p| Ok((p + 1, (p == 0).then_some("stale"))),
            || "invalid"
        )
        .unwrap(),
        None
    );
}

#[test]
fn prefill_rejects_invalid_progress_and_does_not_retry_failed_steps() {
    assert_eq!(
        prefill::<(), _>(3, 2, |_| panic!("invalid initial bound"), || "invalid"),
        Err("invalid")
    );
    for next in [0, 1, 4] {
        let mut calls = 0;
        assert_eq!(
            prefill::<(), _>(
                1,
                3,
                |_| {
                    calls += 1;
                    Ok((next, None))
                },
                || "invalid"
            ),
            Err("invalid")
        );
        assert_eq!(calls, 1);
    }
    let mut consumed = Vec::new();
    let result = prefill::<(), _>(
        0,
        3,
        |position| {
            if position == 1 {
                return Err("cancelled");
            }
            consumed.push(position);
            Ok((position + 1, None))
        },
        || "invalid",
    );
    assert_eq!(result, Err("cancelled"));
    assert_eq!(consumed, [0]);
}

#[test]
fn prefill_releases_obsolete_output_before_next_step() {
    struct Output<'a>(&'a Cell<usize>);
    impl Drop for Output<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let dropped = Cell::new(0);
    let output = prefill(
        0,
        3,
        |position| {
            assert_eq!(dropped.get(), position);
            Ok((position + 1, Some(Output(&dropped))))
        },
        || "invalid",
    )
    .unwrap();
    assert_eq!(dropped.get(), 2);
    drop(output);
    assert_eq!(dropped.get(), 3);
}

fn options(max_tokens: usize, stop_tokens: &[i32]) -> DecodeOptions<'_> {
    DecodeOptions {
        max_tokens,
        stop_tokens,
        allocation: TokenAllocation::Incremental,
    }
}

#[test]
fn selection_publication_consumption_and_terminal_order_are_explicit() {
    for allocation in [TokenAllocation::Upfront, TokenAllocation::Incremental] {
        let events = RefCell::new(Vec::new());
        let result = decode(
            1,
            DecodeOptions {
                allocation,
                ..options(3, &[3])
            },
            &mut (),
            || {
                events.borrow_mut().push("check".into());
                Ok(())
            },
            |_, state| {
                events.borrow_mut().push(format!("select:{state}"));
                Ok(*state)
            },
            |token| {
                events.borrow_mut().push(format!("publish:{token}"));
                Ok(())
            },
            |_, token| {
                events.borrow_mut().push(format!("forward:{token}"));
                Ok(token + 1)
            },
        )
        .unwrap();
        assert_eq!(result.tokens, [1, 2, 3]);
        assert_eq!(result.transitions, 2);
        assert_eq!(result.stop_reason, TerminalReason::StopToken);
        assert_eq!(
            events.into_inner(),
            [
                "check",
                "select:1",
                "publish:1",
                "check",
                "forward:1",
                "check",
                "check",
                "select:2",
                "publish:2",
                "check",
                "forward:2",
                "check",
                "check",
                "select:3",
            ]
        );
        assert!(result.wall_ms >= result.first_token_ready_ms.unwrap());
        assert!(result.first_token_callback_ms.unwrap() >= result.first_token_ready_ms.unwrap());
        assert!(result.first_token_selection_ms >= 0.0);
        assert!(result.transition_ms >= result.first_transition_ms.unwrap());
    }
}

#[test]
fn final_nonstop_is_published_but_neither_terminal_kind_is_forwarded() {
    for stops in [vec![], vec![7]] {
        let emitted = RefCell::new(Vec::new());
        let result = decode(
            7,
            options(1, &stops),
            &mut (),
            || Ok(()),
            |_, state| Ok(*state),
            |token| {
                emitted.borrow_mut().push(token);
                Ok(())
            },
            |_, _| -> Result<i32> { panic!("terminal sample must not be consumed") },
        )
        .unwrap();
        assert_eq!(result.tokens, [7]);
        assert_eq!(result.transitions, 0);
        assert_eq!(result.transition_ms, 0.0);
        assert!(result.first_transition_ms.is_none());
        if stops.is_empty() {
            assert_eq!(emitted.into_inner(), [7]);
            assert_eq!(result.stop_reason, TerminalReason::TokenLimit);
            assert!(result.first_token_callback_ms.is_some());
        } else {
            assert!(emitted.into_inner().is_empty());
            assert_eq!(result.stop_reason, TerminalReason::StopToken);
            assert!(result.first_token_callback_ms.is_none());
        }
    }
}

#[test]
fn zero_limit_fails_before_any_callback_or_allocation() {
    let error = decode(
        (),
        options(0, &[]),
        &mut (),
        || panic!("checkpoint must not run"),
        |_, _| panic!("selection must not run"),
        |_| panic!("publication must not run"),
        |_, _| -> Result<()> { panic!("forward must not run") },
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "max_tokens must be >= 1");
}

#[test]
fn caller_allocation_policy_preserves_early_stop_memory_behavior() {
    for allocation in [TokenAllocation::Upfront, TokenAllocation::Incremental] {
        let result = decode(
            (),
            DecodeOptions {
                allocation,
                ..options(128, &[7])
            },
            &mut (),
            || Ok(()),
            |_, _| Ok(7),
            |_| panic!("stop token is not emitted"),
            |_, _| -> Result<()> { panic!("stop token is not consumed") },
        )
        .unwrap();
        match allocation {
            TokenAllocation::Upfront => assert!(result.tokens.capacity() >= 128),
            TokenAllocation::Incremental => assert!(result.tokens.capacity() < 128),
        }
    }
}

#[test]
fn request_cancellation_is_shared_idempotent_and_not_cancel_on_drop() {
    let first = ExecutionControl::default();
    let second = first.clone();
    drop(first.clone());
    assert!(second.checkpoint().is_ok());
    std::thread::spawn(move || {
        first.cancel();
        first.cancel();
    })
    .join()
    .unwrap();
    assert_eq!(second.checkpoint(), Err(ExecutionCancelled));
    assert!(second.is_cancelled());
    assert!(ExecutionControl::default().checkpoint().is_ok());
}

#[test]
fn cancellation_after_publication_prevents_the_next_forward() {
    let control = ExecutionControl::default();
    let emitted = RefCell::new(Vec::new());
    let error = decode(
        1,
        options(3, &[]),
        &mut (),
        || {
            control.checkpoint()?;
            Ok(())
        },
        |_, token| Ok(*token),
        |token| {
            emitted.borrow_mut().push(token);
            control.cancel();
            Ok(())
        },
        |_, _| -> Result<i32> { panic!("cancelled publication must not be consumed") },
    )
    .unwrap_err();
    assert!(error.downcast_ref::<ExecutionCancelled>().is_some());
    assert_eq!(emitted.into_inner(), [1]);
}

#[test]
fn callback_failure_precedes_cancellation_and_prevents_forwarding() {
    let control = ExecutionControl::default();
    let error = decode(
        1,
        options(3, &[]),
        &mut (),
        || {
            control.checkpoint()?;
            Ok(())
        },
        |_, token| Ok(*token),
        |_| {
            control.cancel();
            anyhow::bail!("callback failed")
        },
        |_, _| -> Result<i32> { panic!("failed publication must not be consumed") },
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "callback failed");
}

#[test]
fn post_transition_cancellation_does_not_roll_back_advanced_context() {
    let control = ExecutionControl::default();
    let selected = Cell::new(0);
    let mut consumed = Vec::new();
    let error = decode(
        1,
        options(3, &[]),
        &mut consumed,
        || {
            control.checkpoint()?;
            Ok(())
        },
        |_, token| {
            selected.set(selected.get() + 1);
            Ok(*token)
        },
        |_| Ok(()),
        |consumed, token| {
            consumed.push(token);
            control.cancel();
            Ok(token + 1)
        },
    )
    .unwrap_err();
    assert!(error.downcast_ref::<ExecutionCancelled>().is_some());
    assert_eq!(consumed, [1]);
    assert_eq!(selected.get(), 1);
}

#[test]
fn terminal_decisions_do_not_add_a_late_cancellation_checkpoint() {
    for stop in [false, true] {
        let control = ExecutionControl::default();
        let checks = Cell::new(0);
        let result = decode(
            7,
            options(1, if stop { &[7] } else { &[] }),
            &mut (),
            || {
                checks.set(checks.get() + 1);
                control.checkpoint()?;
                Ok(())
            },
            |_, token| {
                if stop {
                    control.cancel();
                }
                Ok(*token)
            },
            |_| {
                assert!(!stop);
                control.cancel();
                Ok(())
            },
            |_, _| -> Result<i32> { panic!("terminal token") },
        )
        .unwrap();
        assert_eq!(result.tokens, [7]);
        assert!(control.is_cancelled());
        assert_eq!(checks.get(), 1);
    }
}

#[test]
fn callback_errors_and_process_diagnostics_are_not_rewritten() {
    for failing in ["checkpoint", "select", "forward"] {
        let error = decode(
            1,
            options(3, &[]),
            &mut (),
            || {
                if failing == "checkpoint" {
                    anyhow::bail!("termination signal 15 received")
                }
                Ok(())
            },
            |_, token| {
                if failing == "select" {
                    anyhow::bail!("select failed")
                }
                Ok(*token)
            },
            |_| Ok(()),
            |_, _| -> Result<i32> { anyhow::bail!("forward failed") },
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            match failing {
                "checkpoint" => "termination signal 15 received",
                "select" => "select failed",
                _ => "forward failed",
            }
        );
    }
}

#[test]
fn seeded_sampler_tokens_and_rng_progress_match_existing_goldens() {
    for temperature in [0.0, 1.0] {
        let config = SamplingConfig {
            temperature,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0x1234_5678_9abc_def0,
        };
        let mut sampler = Sampler::new(config).unwrap();
        let logits = vec![2.0, 1.5, 1.0, 0.5];
        let mut consumed = Vec::new();
        let result = decode(
            logits.clone(),
            options(4, &[]),
            &mut sampler,
            || Ok(()),
            |sampler, logits| Ok(sampler.sample(logits)?.token),
            |_| Ok(()),
            |_, token| {
                consumed.push(token);
                Ok(logits.clone())
            },
        )
        .unwrap();
        assert_eq!(
            result.tokens,
            if temperature == 0.0 {
                vec![0, 0, 0, 0]
            } else {
                vec![0, 1, 1, 1]
            }
        );
        assert_eq!(consumed, result.tokens[..3]);
        assert_eq!(result.transitions, 3);
        assert_eq!(sampler.draws(), if temperature == 0.0 { 0 } else { 4 });
        let mut reference = Sampler::new(config).unwrap();
        for expected in &result.tokens {
            assert_eq!(reference.sample(&logits).unwrap().token, *expected);
        }
        assert_eq!(
            sampler.sample(&logits).unwrap().token,
            reference.sample(&logits).unwrap().token
        );
        assert_eq!(sampler.draws(), reference.draws());
    }
}
