use super::*;
use crate::metal::{MemoryAdmissionDenied, MetalMemoryAdmissionReason, MetalMemorySignals};
use crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES;

thread_local! {
    static FAIL_ALLOCATION_ONCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static ALLOCATION_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn allocation_checkpoint() -> Result<(), Qwen4ExpTextSessionError> {
    ALLOCATION_ATTEMPTS.set(ALLOCATION_ATTEMPTS.get() + 1);
    if FAIL_ALLOCATION_ONCE.replace(false) {
        Err(MetalError::NoBuffer(4096).into())
    } else {
        Ok(())
    }
}

#[test]
fn narrower_prefill_candidates_are_bounded_and_end_at_scalar() {
    for (start, expected) in [
        (Some(2048), Some(1024)),
        (Some(154), Some(128)),
        (Some(3), Some(2)),
        (Some(2), None),
        (None, None),
    ] {
        assert_eq!(smaller_prefill_rows(start), expected);
    }
}

#[test]
fn prefill_retry_requires_a_resource_failure() {
    for reason in [
        MetalMemoryAdmissionReason::ProcessInsufficient,
        MetalMemoryAdmissionReason::WorkingSetInsufficient,
        MetalMemoryAdmissionReason::BothInsufficient,
        MetalMemoryAdmissionReason::InvalidWorkingSetSignal,
        MetalMemoryAdmissionReason::ProcessSignalUnavailable,
        MetalMemoryAdmissionReason::RequiredBytesOverflow,
    ] {
        let denied = MemoryAdmissionDenied {
            reason,
            required_bytes: Some(4096),
            signals: MetalMemorySignals {
                recommended_max_bytes: 1024,
                current_allocated_bytes: 0,
                process_limit_remaining_bytes: Some(1024),
            },
            working_set_headroom_bytes: Some(1024),
        };
        assert_eq!(
            session_allocation_can_narrow(&Qwen4ExpTextSessionError::MemoryAdmission(denied)),
            reason.is_pressure()
        );
    }
    assert!(session_allocation_can_narrow(
        &Qwen4ExpTextSessionError::Metal(MetalError::NoBuffer(4096))
    ));
    assert!(!session_allocation_can_narrow(
        &Qwen4ExpTextSessionError::Metal(MetalError::NoFunction("missing".into()))
    ));
    assert!(!session_allocation_can_narrow(
        &Qwen4ExpTextSessionError::Invalid("bad geometry".into())
    ));
}

fn measured_plan(
    ctx: &MetalContext,
    gguf: &GgufFile,
    capacity: Qwen4ExpSessionCapacity,
    extent: usize,
) -> Qwen4ExpTextSessionPlan {
    let weights = Qwen4ExpMetalWeightPlan::for_native(ctx, gguf).unwrap();
    Qwen4ExpTextSessionPlan::for_config_with_packed_prefill_tokens(
        ctx,
        weights.config(),
        capacity.qsa_physical_capacity(),
        weights.memory_plan(),
        extent,
    )
    .unwrap()
}

fn task_answer(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tok: &crate::tokenizer::Tokenizer,
    stops: &[i32],
    mut logits: Vec<f32>,
) -> String {
    let mut output = String::new();
    for _ in 0..48 {
        assert!(logits.iter().all(|x| x.is_finite()));
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0 as i32;
        if stops.contains(&token) {
            assert!(output.to_lowercase().contains("rename"));
            return output;
        }
        output.push_str(&tok.decode_piece(token));
        logits = runner.forward_token(token as u32).unwrap().to_vec();
    }
    panic!("answer did not stop within 48 tokens: {output:?}");
}

#[test]
#[ignore = "QWEN4EXP_Q3_K_XL_RUNTIME_GGUF; metadata and Metal pricing only"]
fn flash_prefill_pressure_plan_preserves_context_and_selection() {
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let ctx = MetalContext::new().unwrap();
    let capacity =
        Qwen4ExpSessionCapacity::for_forward_limit(&Qwen4ExpConfig::flash_next_reference(), 32768)
            .unwrap();
    let full = measured_plan(&ctx, &gguf, capacity, 32768);
    let split = full.clone().with_split_decode(&ctx, true).unwrap();
    assert!(
        split
            .with_prefill_rows(&ctx, Some(512))
            .unwrap()
            .memory_plan()
            .split_decode_enabled()
    );
    assert!(full.with_prefill_rows(&ctx, Some(1)).is_err());
    assert!(full.with_prefill_rows(&ctx, Some(4096)).is_err());
    for rows in [Some(2048), Some(1024), Some(512), Some(256), Some(2), None] {
        let plan = full.with_prefill_rows(&ctx, rows).unwrap();
        assert_eq!(plan.geometry(), full.geometry());
        assert_eq!(plan.memory_plan().packed_selected_capable(), rows.is_some());
        eprintln!(
            "Flash prefill rows={rows:?} session_priced_bytes={} residency_bytes={}",
            plan.memory_plan().session_priced_upper_bytes(),
            plan.memory_plan().residency_priced_upper_bytes()
        );
        for before in [true, false] {
            let budget = if before {
                plan.memory_plan()
                    .priced_upper_bytes_for_sessions(1)
                    .unwrap()
            } else {
                plan.memory_plan().session_priced_upper_bytes()
            } + QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES;
            let signals = MetalMemorySignals {
                recommended_max_bytes: u64::MAX / 2,
                current_allocated_bytes: 0,
                process_limit_remaining_bytes: Some(budget),
            };
            let (selected, _) =
                select_prefill_plan(&ctx, full.clone(), signals, before, true).unwrap();
            assert_eq!(selected.memory_plan().packed_prefill_capacity(), rows);
            assert_eq!(
                selected.memory_plan().packed_selected_capable(),
                rows.is_some()
            );
            if rows != Some(2048) {
                assert!(matches!(
                    select_prefill_plan(&ctx, full.clone(), signals, before, false),
                    Err(Qwen4ExpRuntimeError::MemoryAdmission(_))
                ));
            }
        }
    }
    let unavailable = MetalMemorySignals {
        recommended_max_bytes: u64::MAX / 2,
        current_allocated_bytes: 0,
        process_limit_remaining_bytes: None,
    };
    assert!(
        matches!(select_prefill_plan(&ctx, full.clone(), unavailable, true, true),
        Err(Qwen4ExpRuntimeError::MemoryAdmission(denied)) if denied.reason == MetalMemoryAdmissionReason::ProcessSignalUnavailable)
    );
    let exhausted = MetalMemorySignals {
        process_limit_remaining_bytes: Some(1),
        ..unavailable
    };
    assert!(
        matches!(select_prefill_plan(&ctx, full, exhausted, true, true),
        Err(Qwen4ExpRuntimeError::MemoryAdmission(denied)) if denied.reason.is_pressure())
    );
}

#[test]
#[ignore = "QWEN4EXP_Q3_K_XL_RUNTIME_GGUF; long prefill, snapshot and continuation under Metal validation"]
fn flash_narrowed_prefill_retains_selected_execution_and_snapshot_continuation() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let stops = gguf.stop_token_ids().unwrap();
    let mut body = String::new();
    for i in 0..100 {
        body.push_str(&format!("Record {i}: the storage service validates each key, writes a temporary file, flushes it, and atomically renames it into place. Failed writes leave the previous complete record unchanged. Unrelated records remain untouched.\n"));
    }
    let text = format!(
        "<|im_start|>user\n{body}\nWhich operation ensures readers see a complete record? Answer briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let tokens: Vec<u32> = tok
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .map(|id| id as u32)
        .collect();
    assert!(tokens.len() > 4096);
    let ctx = MetalContext::new().unwrap();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(
        &Qwen4ExpConfig::flash_next_reference(),
        tokens.len() + 128,
    )
    .unwrap();
    struct ClearFault;
    impl Drop for ClearFault {
        fn drop(&mut self) {
            FAIL_ALLOCATION_ONCE.set(false);
        }
    }
    let _clear_fault = ClearFault;
    FAIL_ALLOCATION_ONCE.set(true);
    ALLOCATION_ATTEMPTS.set(0);
    let mut loaded = Qwen4ExpLoadedModel::load_with_adaptive_prefill(
        &ctx,
        &gguf,
        capacity,
        tokens.len(),
        Qwen4ExpDecodeOptions::default(),
    )
    .unwrap();
    assert_eq!(ALLOCATION_ATTEMPTS.get(), 2);
    assert_eq!(loaded.packed_prefill_capacity(), Some(1024));
    assert!(loaded.packed_selected_capable());
    let guarded_topk = loaded.guarded_topk_enabled();
    let hc_up_mix = loaded.hc_up_mix_enabled();
    assert!(
        guarded_topk && hc_up_mix,
        "probe requires production defaults"
    );
    let full = measured_plan(&ctx, &gguf, capacity, tokens.len());
    let mut results = Vec::new();
    let suffix_start = tokens.len() - 128;
    let mut common_prefix = None;
    for rows in [Some(2048), Some(512), None] {
        drop(loaded.workspace.take());
        let plan = full.with_prefill_rows(&ctx, rows).unwrap();
        let admitted = plan
            .admit_after_residency(&loaded.weights, ctx.memory_signals())
            .unwrap();
        let mut workspace =
            Qwen4ExpTextSessionMetalWorkspace::from_admitted(&ctx, admitted).unwrap();
        workspace
            .configure_guarded_topk(&ctx, guarded_topk)
            .unwrap();
        workspace.configure_hc_up_mix(&ctx, hc_up_mix).unwrap();
        loaded.workspace = Some(workspace);
        assert_eq!(loaded.guarded_topk_enabled(), guarded_topk);
        assert_eq!(loaded.hc_up_mix_enabled(), hc_up_mix);
        let allocated = loaded.observed_session_bytes();
        let mut runner = loaded.create_runner(&ctx).unwrap();
        if rows.is_some() {
            runner.prefill(&tokens).unwrap();
            let cold = runner.last_prefill_timing().unwrap().total_wall_ms;
            runner.reset().unwrap();
            let full_logits = runner.prefill(&tokens).unwrap().to_vec();
            let warm = runner.last_prefill_timing().unwrap().total_wall_ms;
            eprintln!(
                "Flash rows={rows:?} full_tokens={} first_ms={cold:.1} warm_ms={warm:.1} session_bytes={allocated}",
                tokens.len()
            );
            let answer = task_answer(&mut runner, &tok, &stops, full_logits);
            eprintln!("Flash rows={rows:?} full-prefill response={answer:?}");
            runner.reset().unwrap();
        }
        if common_prefix.is_none() {
            runner.prefill(&tokens[..suffix_start]).unwrap();
            common_prefix = Some(runner.capture_snapshot().unwrap());
            runner.reset().unwrap();
        }
        runner
            .restore_snapshot(common_prefix.as_ref().unwrap())
            .unwrap();
        let logits = runner
            .prefill_continuation_with_command_checkpoint(&tokens[suffix_start..], || Ok(()))
            .unwrap()
            .to_vec();
        assert!(logits.iter().all(|x| x.is_finite()));
        let expected_plan = plan_qwen4exp_prefill_execution_from(
            suffix_start,
            tokens.len() - suffix_start,
            rows,
            rows.is_some(),
            runner.packed_qsa_dense_end().unwrap(),
        )
        .unwrap();
        let timing = runner.last_prefill_timing().unwrap();
        assert_eq!(timing.packed_token_count, expected_plan.packed_token_count);
        assert_eq!(timing.contains_selection, expected_plan.contains_selection);
        assert_eq!(runner.next_position(), tokens.len());
        let ms = timing.total_wall_ms;
        eprintln!(
            "Flash rows={rows:?} suffix_tokens={} prefix_tokens={suffix_start} prefill_ms={ms:.1} session_bytes={allocated} commands={}",
            tokens.len() - suffix_start,
            timing.command_count
        );
        let snapshot = runner.capture_snapshot().unwrap();
        let suffix: Vec<u32> = tok
            .encode("The atomic rename", false)
            .unwrap()
            .into_iter()
            .map(|id| id as u32)
            .collect();
        let expected = runner
            .prefill_continuation_with_command_checkpoint(&suffix, || Ok(()))
            .unwrap()
            .to_vec();
        runner.reset().unwrap();
        runner.restore_snapshot(&snapshot).unwrap();
        let actual = runner
            .prefill_continuation_with_command_checkpoint(&suffix, || Ok(()))
            .unwrap()
            .to_vec();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "same-schedule restored continuation"
        );
        runner.reset().unwrap();
        runner.restore_snapshot(&snapshot).unwrap();
        let output = task_answer(&mut runner, &tok, &stops, logits);
        eprintln!("Flash rows={rows:?} response={output:?}");
        results.push((rows, ms, allocated));
        let workspace = runner.into_workspace();
        loaded.restore_workspace(workspace).unwrap();
    }
    eprintln!(
        "Flash common-prefix suffix narrowed/scalar speedup={:.3}x; 2048/512 time ratio={:.3}",
        results[2].1 / results[1].1,
        results[0].1 / results[1].1
    );
}
