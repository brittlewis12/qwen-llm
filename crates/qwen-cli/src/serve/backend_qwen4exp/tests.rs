use super::*;
use qwen_llm::qwen4exp::Qwen4ExpConfig;

#[test]
fn startup_limits_use_flash_next_profile_and_require_bounded_capacity() {
    let config = Qwen4ExpConfig::flash_next_reference();
    let profile = crate::serve::profile(qwen_llm::model_family::ModelFamily::Qwen4Exp);
    for (context, output, expected) in [
        (None, Some(8), "requires --max-context-tokens"),
        (Some(64), None, "requires explicit --max-tokens"),
        (Some(0), Some(1), "must be greater than 0"),
        (Some(64), Some(65), "exceeds --max-context-tokens"),
    ] {
        let error = crate::serve::resolve_serve_limits(
            profile,
            Some(config.context_length as usize),
            context,
            output,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains(expected), "{error}");
    }

    let limits = crate::serve::resolve_serve_limits(
        profile,
        Some(config.context_length as usize),
        Some(256),
        Some(64),
    )
    .unwrap()
    .unwrap();
    assert_eq!((limits.context_tokens, limits.max_tokens), (256, 64));
}

#[test]
fn session_capacity_bounds_forward_limit() {
    let config = Qwen4ExpConfig::flash_next_reference();
    for forward_limit in [1, 64, 255, 256, 4096] {
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, forward_limit).unwrap();
        assert_eq!(capacity.forward_limit(), forward_limit);
        assert!(capacity.qsa_physical_capacity() >= forward_limit);
        assert!(capacity.qsa_physical_capacity() <= config.context_length as usize);
    }
    assert!(Qwen4ExpSessionCapacity::for_forward_limit(&config, 0).is_err());
    assert!(
        Qwen4ExpSessionCapacity::for_forward_limit(&config, config.context_length as usize + 1)
            .is_err()
    );
}

#[test]
fn invalid_sampling_is_refused_before_model_work() {
    let request = crate::serve::items::parse_request(&serde_json::json!({
        "model":"flash-next", "input":"hello", "temperature":-1.0
    }));
    assert!(request.is_err());
    assert!(
        request
            .unwrap_err()
            .message
            .contains("temperature must be >= 0")
    );
}

#[test]
fn transcript_plan_splits_only_when_boundary_is_ahead_of_cache() {
    assert_eq!(transcript_split(None, 4), None);
    assert_eq!(transcript_split(Some(8), 4), Some(8));
    assert_eq!(transcript_split(Some(4), 4), None);
    assert_eq!(transcript_split(Some(2), 4), None);
}

#[test]
fn prefill_failure_maps_checkpoint_abort_separately_from_runtime_error() {
    let abort = map_prefill_result::<()>(
        Err(Qwen4ExpRuntimeError::Invalid("prefill failed".into())),
        Some(io::Error::new(io::ErrorKind::BrokenPipe, "client gone")),
    );
    assert!(matches!(abort, Err(BackendFailure::Aborted(_))));

    let failure = map_prefill_result::<()>(
        Err(Qwen4ExpRuntimeError::Invalid("prefill failed".into())),
        None,
    );
    assert!(matches!(failure, Err(BackendFailure::Serve(error)) if error.status == 500));
}

#[test]
fn failed_snapshot_restore_evicts_the_entry_and_resets_the_runner() {
    let mut cache = SnapshotCache::new(1024, Default::default());
    assert!(cache.insert_strict(vec![1, 2], "snapshot", 16));
    let mut runner = 99;
    let mut reset = false;
    let matched = restore_best_prefix(
        &mut cache,
        &[1, 2, 3],
        &mut runner,
        |_, _| Err("corrupt snapshot"),
        |runner| {
            *runner = 0;
            reset = true;
            Ok::<_, &str>(())
        },
    )
    .unwrap();
    assert_eq!(matched, 0);
    assert!(reset);
    assert_eq!(runner, 0);
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.indexed_bytes(), 0);
}

#[test]
#[ignore = "GPU-only QWEN4EXP_Q3_K_XL resident/cold parity and request-abort coverage"]
fn gpu_qwen4exp_resident_abort_and_transcript_cache_parity() {
    let _fixture = qwen_llm::test_fixtures::QWEN4EXP_Q3_K_XL;
}

#[test]
#[ignore = "GPU-only QWEN4EXP_Q3_K_XL transcript snapshot restore parity"]
fn gpu_qwen4exp_transcript_restore_matches_cold_run() {
    let _fixture = qwen_llm::test_fixtures::QWEN4EXP_Q3_K_XL;
}
