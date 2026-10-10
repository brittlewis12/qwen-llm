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
#[ignore = "GPU-only QWEN4EXP_Q3_K_XL resident/cold parity and request-abort coverage"]
fn gpu_qwen4exp_resident_abort_and_transcript_cache_parity() {
    let _fixture = qwen_llm::test_fixtures::QWEN4EXP_Q3_K_XL;
}

#[test]
#[ignore = "GPU-only QWEN4EXP_Q3_K_XL transcript snapshot restore parity"]
fn gpu_qwen4exp_transcript_restore_matches_cold_run() {
    let _fixture = qwen_llm::test_fixtures::QWEN4EXP_Q3_K_XL;
}
