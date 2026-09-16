use super::*;
use clap::CommandFactory;
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::os::unix::fs::PermissionsExt;

#[test]
fn muse_glimmer_forward_budget_counts_only_required_transitions() {
    assert_eq!(required_forwards("Muse Glimmer", 2, 1, None).unwrap(), 2);
    assert_eq!(required_forwards("Muse Glimmer", 2, 3, None).unwrap(), 4);
    assert_eq!(
        required_forwards("Muse Glimmer", 7_168, 2, None).unwrap(),
        7_169
    );
    assert!(required_forwards("Muse Glimmer", 0, 1, None).is_err());
    assert!(required_forwards("Muse Glimmer", 1, 0, None).is_err());
    assert!(required_forwards("Muse Glimmer", usize::MAX, 2, None).is_err());
}

#[test]
fn muse_glimmer_modern_user_input_renders_atem_without_tokenizer_bos() {
    let mut args = Args::try_parse_from([
        "qwen",
        "run",
        "-m",
        "model.gguf",
        "--system",
        "Be exact.",
        "--user",
        "Hello",
        "--reasoning-effort",
        "low",
    ])
    .unwrap();
    let invocation = cli::normalize(&mut args);
    invocation.apply_option_overrides(&mut args);
    let prepared = prepare_muse_glimmer_prompt(
        invocation,
        &MuseGlimmerConfig::unsloth_release_reference(),
        &args,
    )
    .unwrap();
    assert_eq!(prepared.source, PromptSource::Messages);
    assert!(!prepared.add_special_tokens);
    assert!(
        prepared
            .text
            .starts_with("<|begin_of_text|><|start|>system")
    );
    assert!(prepared.text.contains("Reasoning strength: low."));
    assert!(prepared.text.ends_with("<|start|>assistant"));
}

#[test]
fn muse_glimmer_messages_use_the_shared_structured_atem_contract() {
    let path = std::env::temp_dir().join(format!(
        "qwen-muse-messages-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(
        &path,
        r#"{
            "messages":[
                {"role":"user","content":"Weather?"},
                {"role":"assistant","content":"","tool_calls":[{"name":"weather.lookup","arguments":{"city":"Paris"}}]},
                {"role":"tool","name":"weather.lookup","content":"Sunny"}
            ],
            "tools":[{"name":"weather.lookup","description":"Weather","parameters":{"type":"object"}}]
        }"#,
    )
    .unwrap();
    let mut args = Args::try_parse_from([
        "qwen",
        "run",
        "-m",
        "model.gguf",
        "--messages",
        path.to_str().unwrap(),
        "--reasoning-effort",
        "medium",
    ])
    .unwrap();
    let invocation = cli::normalize(&mut args);
    invocation.apply_option_overrides(&mut args);
    let prepared = prepare_muse_glimmer_prompt(
        invocation,
        &MuseGlimmerConfig::unsloth_release_reference(),
        &args,
    )
    .unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(prepared.source, PromptSource::Messages);
    assert!(!prepared.add_special_tokens);
    assert!(prepared.text.contains("Reasoning strength: medium."));
    assert!(
        prepared
            .text
            .contains("<atem:invoke name=\"weather.lookup\">")
    );
    assert!(
        prepared
            .text
            .contains("<tool_output name=\"weather.lookup\">\nSunny")
    );
    assert!(prepared.text.ends_with("<|start|>assistant"));
}

#[test]
fn muse_reasoning_levels_bind_through_the_library_table() {
    for (requested, expected) in [
        ("low", MuseGlimmerReasoningStrength::Low),
        ("medium", MuseGlimmerReasoningStrength::Medium),
        ("high", MuseGlimmerReasoningStrength::High),
        ("xhigh", MuseGlimmerReasoningStrength::Xhigh),
    ] {
        assert_eq!(
            resolve_muse_glimmer_reasoning_strength(requested).unwrap(),
            expected
        );
    }
    let err = resolve_muse_glimmer_reasoning_strength("max").unwrap_err();
    assert!(err.to_string().contains("low|medium|high|xhigh"), "{err}");
    // Advertised levels come from the same table the parser uses.
    assert_eq!(
        muse_glimmer_reasoning_capability().levels,
        MuseGlimmerReasoningStrength::level_names()
    );
}

#[test]
fn qwen4exp_full_shard_prefetch_scope_is_default_off_and_release_scoped() {
    assert!(validate_qwen4exp_full_shard_prefetch_scope(false, 0).is_ok());
    assert!(validate_qwen4exp_full_shard_prefetch_scope(true, 3).is_ok());
    assert!(validate_qwen4exp_full_shard_prefetch_scope(true, 2).is_err());
}

#[test]
fn qwen4exp_hc_up_mix_default_on_has_strict_rollback() {
    use std::ffi::OsStr;
    let parse = |value| parse_qwen4exp_decode_flag(value, QWEN4EXP_HC_UP_MIX_ENV);
    assert!(parse(None).unwrap());
    assert!(!parse(Some(OsStr::new("0"))).unwrap());
    assert!(parse(Some(OsStr::new("1"))).unwrap());
    for value in ["", "true", "false", " 1", "1 ", "2"] {
        assert!(
            parse(Some(OsStr::new(value)))
                .unwrap_err()
                .to_string()
                .contains(QWEN4EXP_HC_UP_MIX_ENV)
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(parse(Some(OsStr::from_bytes(&[0xff]))).is_err());
    }
}

#[test]
fn qwen4exp_guarded_topk_default_on_has_strict_rollback() {
    use std::ffi::OsStr;
    let parse = |value| parse_qwen4exp_decode_flag(value, QWEN4EXP_GUARDED_TOPK_ENV);
    assert!(parse(None).unwrap());
    assert!(!parse(Some(OsStr::new("0"))).unwrap());
    assert!(parse(Some(OsStr::new("1"))).unwrap());
    for value in ["", "true", "false", " 1", "1 ", "2"] {
        assert!(
            parse(Some(OsStr::new(value)))
                .unwrap_err()
                .to_string()
                .contains(QWEN4EXP_GUARDED_TOPK_ENV)
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(parse(Some(OsStr::from_bytes(&[0xff]))).is_err());
    }
    assert!(qwen_llm::qwen4exp_runtime::Qwen4ExpDecodeOptions::default().guarded_topk);
}

#[test]
fn qwen4exp_full_shard_prefetch_report_requires_complete_exact_read() {
    assert!(validate_qwen4exp_full_shard_prefetch_report(3, 3, 3, 0, 90_000, 90_000).is_ok());
    for invalid in [(2, 3, 0, 90_000), (3, 2, 1, 90_000), (3, 3, 0, 89_999)] {
        assert!(
            validate_qwen4exp_full_shard_prefetch_report(
                3, invalid.0, invalid.1, invalid.2, 90_000, invalid.3,
            )
            .is_err()
        );
    }
}

#[test]
fn qwen4exp_prefill_mode_reports_execution_not_selected_authority() {
    assert_eq!(
        qwen4exp_prefill_execution_mode(false, false, false, 2_051, 1, false),
        "packed_then_scalar"
    );
    assert_eq!(
        qwen4exp_prefill_execution_mode(false, false, false, 2_053, 0, true),
        "packed_contains_selection"
    );
    assert_eq!(
        qwen4exp_prefill_execution_mode(false, false, false, 2_048, 0, false),
        "packed_dense"
    );
}

#[test]
fn dflash_uses_serial_tail_when_a_full_verify_block_will_not_fit() {
    const BLOCK: usize = 8;
    const OFF_CTX: usize = 16_384;
    for remaining in 0..BLOCK {
        assert!(!dflash_speculation_enabled(
            false, 100, OFF_CTX, remaining, BLOCK, false, false
        ));
    }
    assert!(dflash_speculation_enabled(
        false, 100, OFF_CTX, BLOCK, BLOCK, false, false
    ));
    assert!(!dflash_speculation_enabled(
        false, 100, OFF_CTX, BLOCK, BLOCK, true, false
    ));
    assert!(dflash_speculation_enabled(
        false, 100, OFF_CTX, BLOCK, BLOCK, true, true
    ));
    assert!(!dflash_speculation_enabled(
        false, OFF_CTX, OFF_CTX, BLOCK, BLOCK, false, false
    ));
    assert_eq!(parse_dflash_off_ctx(None).unwrap(), DFLASH_OFF_CTX_DEFAULT);
    assert_eq!(
        parse_dflash_off_ctx(Some(OsStr::new("65536"))).unwrap(),
        65_536
    );
    assert!(parse_dflash_off_ctx(Some(OsStr::new("invalid"))).is_err());
}

#[test]
fn dflash_prefix_replay_requires_an_exact_emitted_prefix() {
    let history = [10, 11, 12, 13, 14, 15, 16, 17, 18];
    assert_eq!(
        dflash_prefix_replay_drafts(&history, &[10, 11], 7),
        Some(&history[2..9])
    );
    assert_eq!(dflash_prefix_replay_drafts(&history, &[10, 99], 7), None);
    assert_eq!(
        dflash_prefix_replay_drafts(&history, &[10, 11, 12], 7),
        None
    );
    assert!(dflash_prefix_replay_allowed(
        false,
        100,
        DFLASH_OFF_CTX_DEFAULT,
        8,
        8,
        false,
        false,
        false
    ));
    assert!(!dflash_prefix_replay_allowed(
        true,
        100,
        DFLASH_OFF_CTX_DEFAULT,
        8,
        8,
        false,
        false,
        false
    ));
    assert!(!dflash_prefix_replay_allowed(
        false,
        DFLASH_OFF_CTX_DEFAULT,
        DFLASH_OFF_CTX_DEFAULT,
        8,
        8,
        false,
        false,
        false
    ));
    assert!(!dflash_prefix_replay_allowed(
        false,
        100,
        DFLASH_OFF_CTX_DEFAULT,
        7,
        8,
        false,
        false,
        false
    ));
    assert!(dflash_prefix_replay_allowed(
        false,
        100,
        DFLASH_OFF_CTX_DEFAULT,
        8,
        8,
        false,
        true,
        false
    ));
    assert!(dflash_prefix_replay_allowed(
        false, 20_000, 65_536, 8, 8, true, true, true
    ));
    assert!(!dflash_prefix_replay_allowed(
        false, 20_000, 65_536, 8, 8, true, true, false
    ));
}

fn record_long_policy_step(
    state: &mut DflashAdaptiveState,
    n_accepted: usize,
    fallback_ran: bool,
    was_probe: bool,
) {
    state.prepare_mode(true);
    state.record_spec_step(
        n_accepted,
        fallback_ran,
        was_probe,
        true,
        false,
        true,
        64_000,
    );
}

#[test]
fn dflash_long_policy_separates_measured_acceptance_bands() {
    let mut losing = DflashAdaptiveState::default();
    for accepted in [2; 9].into_iter().chain([1; 7]) {
        record_long_policy_step(&mut losing, accepted, false, false);
    }
    assert_eq!(losing.reason, Some(DflashBackoffReason::Acceptance));

    let mut winning = DflashAdaptiveState::default();
    for accepted in [3; 7].into_iter().chain([2; 9]) {
        record_long_policy_step(&mut winning, accepted, false, false);
    }
    assert_eq!(winning.reason, None);

    let mut strong = DflashAdaptiveState::default();
    for accepted in [5; 15].into_iter().chain([4]) {
        record_long_policy_step(&mut strong, accepted, false, false);
    }
    assert_eq!(strong.reason, None);
}

#[test]
fn dflash_long_policy_backs_off_on_dense_exact_fallback() {
    let mut early = DflashAdaptiveState::default();
    for fallback in [true, false, true, false] {
        record_long_policy_step(&mut early, 1, fallback, false);
    }
    assert_eq!(early.reason, Some(DflashBackoffReason::Fallback));

    let mut early_high_acceptance = DflashAdaptiveState::default();
    for fallback in [true, false, true, false] {
        record_long_policy_step(&mut early_high_acceptance, 4, fallback, false);
    }
    assert_eq!(early_high_acceptance.reason, None);

    let mut dense = DflashAdaptiveState::default();
    for fallback in [true, false, true, false, true, false, true, false] {
        record_long_policy_step(&mut dense, 4, fallback, false);
    }
    assert_eq!(dense.reason, Some(DflashBackoffReason::Fallback));

    let mut sparse = DflashAdaptiveState::default();
    for fallback in [true, false, true, false, true, false, false, false] {
        record_long_policy_step(&mut sparse, 4, fallback, false);
    }
    assert_eq!(sparse.reason, None);
}

#[test]
fn dflash_long_policy_prices_probes_and_requires_sustained_reentry() {
    let mut acceptance = DflashAdaptiveState {
        reason: Some(DflashBackoffReason::Acceptance),
        long_mode: true,
        ..DflashAdaptiveState::default()
    };
    for _ in 0..DFLASH_LONG_REPROBE_INTERVAL - 1 {
        acceptance.record_off_step();
    }
    assert!(!acceptance.probe_due(false, true));
    acceptance.record_off_step();
    assert!(acceptance.probe_due(false, true));

    let mut fallback = DflashAdaptiveState {
        reason: Some(DflashBackoffReason::Fallback),
        long_mode: true,
        ..DflashAdaptiveState::default()
    };
    for _ in 0..DFLASH_LONG_FALLBACK_REPROBE_INTERVAL {
        fallback.record_off_step();
    }
    assert!(fallback.probe_due(false, true));
    record_long_policy_step(&mut fallback, 2, false, true);
    assert_eq!(fallback.reason, Some(DflashBackoffReason::Fallback));
    record_long_policy_step(&mut fallback, 3, false, true);
    assert_eq!(fallback.reason, None);

    let mut interrupted = DflashAdaptiveState {
        reason: Some(DflashBackoffReason::Acceptance),
        long_mode: true,
        ..DflashAdaptiveState::default()
    };
    record_long_policy_step(&mut interrupted, 4, false, true);
    record_long_policy_step(&mut interrupted, 4, true, true);
    record_long_policy_step(&mut interrupted, 4, false, true);
    assert_eq!(interrupted.reason, Some(DflashBackoffReason::Fallback));

    let mut replay_only = DflashAdaptiveState {
        reason: Some(DflashBackoffReason::Acceptance),
        long_mode: true,
        ..DflashAdaptiveState::default()
    };
    for _ in 0..DFLASH_LONG_REENTRY_PROBES {
        replay_only.record_spec_step(7, false, true, false, false, true, 64_000);
    }
    assert_eq!(replay_only.reason, Some(DflashBackoffReason::Acceptance));
    assert!(replay_only.reentry_accepts.is_empty());
}

#[test]
fn dflash_fallback_density_does_not_change_short_context_policy() {
    let mut state = DflashAdaptiveState::default();
    for _ in 0..DFLASH_LONG_FALLBACK_WINDOW {
        state.record_spec_step(4, true, false, true, false, false, 8_000);
    }
    assert_eq!(state.reason, None);
}

#[test]
fn qwen_model_prefetch_cli_is_explicit_and_jsonl_scoped() {
    assert!(matches!(
        QwenModelPrefetchArg::Auto.policy(),
        PrefetchPolicy::ColdOnly { .. }
    ));
    assert_eq!(QwenModelPrefetchArg::Off.policy(), PrefetchPolicy::Off);

    let args = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--requests-jsonl",
        "requests.jsonl",
        "--model-prefetch",
        "off",
    ])
    .unwrap();
    assert_eq!(args.model_prefetch, Some(QwenModelPrefetchArg::Off));
    let wrong_scope = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--model-prefetch",
        "off",
    ])
    .unwrap();
    assert!(
        validate_qwen_model_prefetch_scope(&wrong_scope)
            .unwrap_err()
            .to_string()
            .contains("requires --requests-jsonl")
    );
}

#[test]
fn generated_thinking_partition_counts_only_aligned_segments() {
    let aligned = thinking_partition_from_pieces(&[
        "reason".to_string(),
        "ing".to_string(),
        "</think>".to_string(),
        "\n\nanswer".to_string(),
    ])
    .unwrap();
    assert_eq!(aligned.reasoning_tokens, Some(2));
    assert_eq!(aligned.delimiter_tokens, Some(1));
    assert_eq!(aligned.visible_tokens, Some(1));
    assert!(aligned.delimiter_token_aligned);

    let split =
        thinking_partition_from_pieces(&["reason</thi".to_string(), "nk>answer".to_string()])
            .unwrap();
    assert!(!split.delimiter_token_aligned);
    assert_eq!(split.reasoning_tokens, None);
    assert_eq!(split.visible_tokens, None);
    assert!(thinking_partition_from_pieces(&["answer".to_string()]).is_none());
}

#[test]
fn exact_lcp_fanout_policy_is_bounded_by_default_and_strict() {
    assert_eq!(
        parse_qwen_prefix_fanout_boundary_policy(None).unwrap(),
        QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp
    );
    assert_eq!(
        parse_qwen_prefix_fanout_boundary_policy(Some("auto")).unwrap(),
        QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp
    );
    assert_eq!(
        parse_qwen_prefix_fanout_boundary_policy(Some("YES")).unwrap(),
        QwenPrefixFanoutBoundaryPolicy::ExactLcp
    );
    assert_eq!(
        parse_qwen_prefix_fanout_boundary_policy(Some("off")).unwrap(),
        QwenPrefixFanoutBoundaryPolicy::ChunkAligned
    );
    assert!(parse_qwen_prefix_fanout_boundary_policy(Some("sometimes")).is_err());
}

#[test]
fn private_suffix_singleton_policy_is_strict_and_rollbackable() {
    assert!(parse_private_suffix_singleton_enabled(None, true).unwrap());
    assert!(!parse_private_suffix_singleton_enabled(None, false).unwrap());
    assert!(!parse_private_suffix_singleton_enabled(Some("off"), true).unwrap());
    assert!(parse_private_suffix_singleton_enabled(Some("YES"), false).unwrap());
    assert!(parse_private_suffix_singleton_enabled(Some("sometimes"), true).is_err());
    assert_eq!(
        choose_private_suffix_execution_mode(true, 6),
        PrivateSuffixExecutionMode::Singleton
    );
    assert_eq!(
        choose_private_suffix_execution_mode(true, 7),
        PrivateSuffixExecutionMode::Packed
    );
    assert_eq!(
        choose_private_suffix_execution_mode(false, 1),
        PrivateSuffixExecutionMode::Packed
    );
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum AllocationEvent {
    CurrentAllocated,
    LegacyScratch(usize),
    Sequence(usize),
    CandidatePlan,
    MemorySignals,
    CandidateScratch,
}

struct FakePrefillAllocator {
    events: Vec<AllocationEvent>,
    current_allocated: VecDeque<u64>,
    plan_result: std::result::Result<PrefillPlanDecision, CandidatePlanFailure>,
    signals: MetalMemorySignals,
    candidate_error: bool,
}

impl PrefillRequestAllocator for FakePrefillAllocator {
    type Scratch = &'static str;
    type Sequence = &'static str;
    type Plan = ();

    fn current_allocated_size(&mut self) -> u64 {
        self.events.push(AllocationEvent::CurrentAllocated);
        self.current_allocated
            .pop_front()
            .expect("fake current allocation sample")
    }

    fn allocate_legacy_scratch(
        &mut self,
        chunk: usize,
        _prompt_tokens: usize,
    ) -> Result<Self::Scratch> {
        self.events.push(AllocationEvent::LegacyScratch(chunk));
        Ok("legacy")
    }

    fn allocate_sequence(&mut self, capacity: usize) -> Result<Self::Sequence> {
        self.events.push(AllocationEvent::Sequence(capacity));
        Ok("sequence")
    }

    fn build_candidate_plan(
        &mut self,
        _profile: AutoPrefillProfile,
        _prompt_tokens: usize,
    ) -> std::result::Result<(Self::Plan, PrefillPlanDecision), CandidatePlanFailure> {
        self.events.push(AllocationEvent::CandidatePlan);
        self.plan_result.clone().map(|decision| ((), decision))
    }

    fn memory_signals(&mut self) -> MetalMemorySignals {
        self.events.push(AllocationEvent::MemorySignals);
        self.signals
    }

    fn allocate_candidate_scratch(&mut self, _plan: Self::Plan) -> Result<Self::Scratch> {
        self.events.push(AllocationEvent::CandidateScratch);
        if self.candidate_error {
            bail!("candidate allocation failed")
        }
        Ok("candidate")
    }
}

fn test_auto_profile() -> AutoPrefillProfile {
    AutoPrefillProfile {
        name: "test-profile",
        outer_chunk: 2048,
        query_heads: 16,
        gdn_overlay_bytes: 235_405_312,
    }
}

fn test_a3b_arch() -> Arch {
    let mut arch = qwen_llm::model::QWEN3_27B;
    arch.kind = ArchKind::Moe;
    arch.n_layer = 40;
    arch.hidden_size = 2048;
    arch.n_q_heads = 16;
    arch.n_kv_heads = 2;
    arch.gdn_n_v_heads = 32;
    arch.expert_count = 256;
    arch.expert_used_count = 8;
    arch.expert_feed_forward_length = 512;
    arch.expert_shared_feed_forward_length = 512;
    arch.mtp_n_hidden_layers = 0;
    arch
}

fn test_a10b_arch() -> Arch {
    let mut arch = test_a3b_arch();
    arch.n_layer = 48;
    arch.hidden_size = 3072;
    arch.n_q_heads = 32;
    arch.gdn_n_v_heads = 64;
    arch.expert_feed_forward_length = 1024;
    arch.expert_shared_feed_forward_length = 1024;
    arch
}

fn fake_plan_decision(priced_upper_bytes: u64) -> PrefillPlanDecision {
    PrefillPlanDecision {
        block_size: 2048,
        matrix_max_pos: 10_000,
        matrix_query_rows: 1024,
        eager_allocation_count: 1,
        deferred_allocation_count: 1,
        eager_logical_bytes: 10,
        deferred_logical_bytes: 20,
        maximum_logical_bytes: 30,
        priced_upper_bytes,
        overlay: PrefillScratchOverlayTimingStats {
            backing_bytes: 1,
            attention_bytes: 1,
            gdn_bytes: 1,
            saved_bytes: 1,
        },
        allocations: vec![
            PrefillPlanAllocationDecision {
                name: "eager",
                deferred: false,
                logical_bytes: 10,
                priced_bytes: 10,
                alignment: 256,
            },
            PrefillPlanAllocationDecision {
                name: "deferred",
                deferred: true,
                logical_bytes: 20,
                priced_bytes: priced_upper_bytes.saturating_sub(10),
                alignment: 256,
            },
        ],
    }
}

fn fake_allocator(
    current_allocated: &[u64],
    plan_result: std::result::Result<PrefillPlanDecision, CandidatePlanFailure>,
    recommended_max_bytes: u64,
    candidate_error: bool,
) -> FakePrefillAllocator {
    fake_allocator_with_signals(
        current_allocated,
        plan_result,
        MetalMemorySignals {
            recommended_max_bytes,
            current_allocated_bytes: current_allocated.get(1).copied().unwrap_or(0),
            process_limit_remaining_bytes: Some(0),
        },
        candidate_error,
    )
}

fn fake_allocator_with_signals(
    current_allocated: &[u64],
    plan_result: std::result::Result<PrefillPlanDecision, CandidatePlanFailure>,
    signals: MetalMemorySignals,
    candidate_error: bool,
) -> FakePrefillAllocator {
    FakePrefillAllocator {
        events: Vec::new(),
        current_allocated: current_allocated.iter().copied().collect(),
        plan_result,
        signals,
        candidate_error,
    }
}

fn logits_with_argmax(token: usize) -> Vec<f32> {
    let mut logits = vec![0.0; 4];
    logits[token] = 1.0;
    logits
}

fn fake_structural_row_evidence() -> StructuralRowEvidence {
    StructuralRowEvidence {
        resident_head_wait_calls: 1,
        validated_shared_row_calls: 1,
        transition_logits_copy_bytes: 0,
        extra_command_buffers: 0,
        gpu_sampling_dispatches: 0,
    }
}

fn fake_structural_transition<Advance>(
    trial: &mut Sampler,
    telemetry: &mut SampledStructuralTelemetry,
    logits: &[f32],
    advance: Advance,
) -> Result<SampledStructuralDecodeState>
where
    Advance: FnOnce() -> Result<()>,
{
    let sampled = trial.sample_bounded_top_k(logits);
    let state = match sampled {
        Ok((sampled, evidence)) => {
            telemetry.record_transition(evidence, fake_structural_row_evidence())?;
            SampledStructuralDecodeState::Selected(Ok(sampled))
        }
        Err(error) => SampledStructuralDecodeState::Selected(Err(error)),
    };
    advance()?;
    Ok(state)
}

fn prepared(id: &str, tokens: &[i32]) -> PreparedJsonlRequest {
    PreparedJsonlRequest {
        request: JsonlRequest {
            id: Some(id.to_string()),
            prompt: None,
            prompt_file: None,
            user: None,
            system: None,
            no_thinking: None,
            reasoning_effort: None,
            tokens: None,
            cache_prefix_tokens: None,
            sampling: None,
        },
        id: id.to_string(),
        line: 1,
        input: JsonlInputLabel::RAW,
        prompt_ids: tokens.to_vec(),
        sampling: SamplingConfig::default(),
        auto_cache_prefix_tokens: None,
        auto_cache_future_hits: 0,
    }
}

#[test]
fn deepseek_v4_forward_budget_accounts_for_unconsumed_final_token() {
    assert_eq!(
        required_forwards(
            "DeepSeek V4",
            1,
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
            Some(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY)
        )
        .unwrap(),
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY
    );
    assert_eq!(
        required_forwards(
            "DeepSeek V4",
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
            1,
            Some(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY)
        )
        .unwrap(),
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY
    );
    assert!(
        required_forwards(
            "DeepSeek V4",
            1,
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY + 1,
            Some(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY)
        )
        .is_err()
    );
    assert!(
        required_forwards(
            "DeepSeek V4",
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
            2,
            Some(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY)
        )
        .is_err()
    );
    assert!(
        required_forwards(
            "DeepSeek V4",
            0,
            1,
            Some(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY)
        )
        .is_err()
    );
    assert!(
        required_forwards(
            "DeepSeek V4",
            1,
            0,
            Some(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY)
        )
        .is_err()
    );
    assert!(
        required_forwards(
            "DeepSeek V4",
            usize::MAX,
            2,
            Some(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY)
        )
        .is_err()
    );
}

#[test]
fn qwen4exp_forward_budget_accounts_for_unconsumed_final_token() {
    assert_eq!(
        required_forwards("Qwen3.8-Flash-Next", 18, 8, None).unwrap(),
        25
    );
    assert_eq!(
        required_forwards("Qwen3.8-Flash-Next", 1, 1, None).unwrap(),
        1
    );
    assert!(required_forwards("Qwen3.8-Flash-Next", 0, 1, None).is_err());
    assert!(required_forwards("Qwen3.8-Flash-Next", 1, 0, None).is_err());
    assert!(required_forwards("Qwen3.8-Flash-Next", usize::MAX, 2, None).is_err());
}

#[test]
fn qwen4exp_timing_totals_expose_only_complete_gpu_coverage() {
    let mut totals = Qwen4ExpTimingTotals::default();
    totals.record(Qwen4ExpTokenTiming {
        position: 0,
        encode_cpu_ms: 1.0,
        completion_wait_ms: 4.0,
        gpu_ms: Some(3.0),
        total_wall_ms: 5.0,
    });
    assert_eq!(totals.complete_gpu_ms(), Some(3.0));
    assert_eq!(totals.outside_gpu_ms(), Some(2.0));
    totals.record(Qwen4ExpTokenTiming {
        position: 1,
        encode_cpu_ms: 1.0,
        completion_wait_ms: 4.0,
        gpu_ms: None,
        total_wall_ms: 5.0,
    });
    assert_eq!(totals.complete_gpu_ms(), None);
    assert_eq!(totals.outside_gpu_ms(), None);
    assert_eq!((totals.gpu_samples, totals.forwards), (1, 2));

    let mut packed = Qwen4ExpTimingTotals::default();
    packed.record_prefill(Qwen4ExpPrefillTiming {
        token_count: 2_050,
        packed_token_count: 2_048,
        contains_selection: false,
        command_count: 3,
        encode_cpu_ms: 7.0,
        completion_wait_ms: 18.0,
        gpu_ms: 16.0,
        gpu_samples: 3,
        total_wall_ms: 25.0,
    });
    assert_eq!((packed.gpu_samples, packed.forwards), (3, 3));
    assert_eq!(packed.complete_gpu_ms(), Some(16.0));
    assert_eq!(packed.outside_gpu_ms(), Some(9.0));

    let mut partial = Qwen4ExpTimingTotals::default();
    partial.record_prefill(Qwen4ExpPrefillTiming {
        token_count: 2_050,
        packed_token_count: 2_048,
        contains_selection: false,
        command_count: 3,
        encode_cpu_ms: 7.0,
        completion_wait_ms: 18.0,
        gpu_ms: 11.0,
        gpu_samples: 2,
        total_wall_ms: 25.0,
    });
    assert_eq!((partial.gpu_samples, partial.forwards), (2, 3));
    assert_eq!(partial.complete_gpu_ms(), None);
    assert_eq!(partial.outside_gpu_ms(), None);
}

#[test]
fn qwen4exp_profile_logit_replay_compares_float_bits() {
    let nan = f32::from_bits(0x7fc0_0001);
    assert!(qwen4exp_logits_bitwise_equal(&[nan, -0.0], &[nan, -0.0]));
    assert!(!qwen4exp_logits_bitwise_equal(&[0.0], &[-0.0]));
    assert!(!qwen4exp_logits_bitwise_equal(
        &[nan],
        &[f32::from_bits(0x7fc0_0002)]
    ));
}

#[test]
fn qwen4exp_prompt_capability_is_scoped_to_the_declared_protocol() {
    assert_eq!(
        classify_qwen4exp_prompt_capability(
            ModelFamily::Qwen4Exp,
            Some("gpt2"),
            Some("qwen35"),
            true,
        ),
        None
    );
    for (protocol, expected) in [
        (
            (ModelFamily::Qwen35, Some("gpt2"), Some("qwen35"), true),
            Qwen4ExpPromptCapabilityFailure::Architecture,
        ),
        (
            (ModelFamily::Qwen4Exp, Some("other"), Some("qwen35"), true),
            Qwen4ExpPromptCapabilityFailure::TokenizerModel,
        ),
        (
            (ModelFamily::Qwen4Exp, Some("gpt2"), Some("other"), true),
            Qwen4ExpPromptCapabilityFailure::Pretokenizer,
        ),
        (
            (ModelFamily::Qwen4Exp, Some("gpt2"), Some("qwen35"), false),
            Qwen4ExpPromptCapabilityFailure::ChatTemplate,
        ),
    ] {
        assert_eq!(
            classify_qwen4exp_prompt_capability(protocol.0, protocol.1, protocol.2, protocol.3,),
            Some(expected)
        );
    }
    assert!(validated_qwen36_no_thinking_identity(
        ModelFamily::Qwen35Moe,
        Some("Qwen3.6 35B A3B"),
        Some("gpt2"),
        Some("qwen35"),
    ));
    assert!(validated_qwen38_prompt_identity(
        ModelFamily::Qwen35,
        Some("Qwen3.8 27B!"),
        Some("Qwen3.8-27B"),
        Some("gpt2"),
        Some("qwen35"),
        Some(262_144),
        Some(65),
        Some(1),
        Some(5_120),
        Some(17_408),
    ));
    assert!(!validated_qwen38_prompt_identity(
        ModelFamily::Qwen35,
        Some("Qwen3.8 27B!"),
        Some("Qwen3.8-27B"),
        Some("gpt2"),
        Some("qwen35"),
        Some(262_144),
        Some(64),
        Some(1),
        Some(5_120),
        Some(17_408),
    ));
    for identity in [
        (
            ModelFamily::Qwen35Moe,
            Some("Qwen3.8 27B!"),
            Some("qwen35"),
            Some(262_144),
            Some(65),
            Some(1),
            Some(5_120),
            Some(17_408),
        ),
        (
            ModelFamily::Qwen35,
            Some("Qwen3.7 27B"),
            Some("qwen35"),
            Some(262_144),
            Some(65),
            Some(1),
            Some(5_120),
            Some(17_408),
        ),
        (
            ModelFamily::Qwen35,
            Some("Qwen3.8 27B!"),
            Some("other"),
            Some(262_144),
            Some(65),
            Some(1),
            Some(5_120),
            Some(17_408),
        ),
        (
            ModelFamily::Qwen35,
            Some("Qwen3.8 27B!"),
            Some("qwen35"),
            Some(131_072),
            Some(65),
            Some(1),
            Some(5_120),
            Some(17_408),
        ),
        (
            ModelFamily::Qwen35,
            Some("Qwen3.8 27B!"),
            Some("qwen35"),
            Some(262_144),
            Some(65),
            Some(0),
            Some(5_120),
            Some(17_408),
        ),
        (
            ModelFamily::Qwen35,
            Some("Qwen3.8 27B!"),
            Some("qwen35"),
            Some(262_144),
            Some(65),
            Some(1),
            Some(4_096),
            Some(17_408),
        ),
    ] {
        assert!(!validated_qwen38_prompt_identity(
            identity.0,
            identity.1,
            None,
            Some("gpt2"),
            identity.2,
            identity.3,
            identity.4,
            identity.5,
            identity.6,
            identity.7,
        ));
    }
    for identity in [
        (
            ModelFamily::Qwen35,
            Some("Qwen3.6 35B A3B"),
            Some("gpt2"),
            Some("qwen35"),
        ),
        (
            ModelFamily::Qwen35Moe,
            Some("Qwen3.5 35B A3B"),
            Some("gpt2"),
            Some("qwen35"),
        ),
        (
            ModelFamily::Qwen35Moe,
            Some("Qwen3.6 35B A3B"),
            Some("gpt2"),
            Some("other"),
        ),
    ] {
        assert!(!validated_qwen36_no_thinking_identity(
            identity.0, identity.1, identity.2, identity.3,
        ));
    }
}

#[test]
#[ignore = "set QWEN4EXP_TOKENIZER_GGUF to the pinned Flash-Next release"]
fn released_qwen4exp_chat_template_matches_supported_protocol() {
    let path = std::env::var_os("QWEN4EXP_TOKENIZER_GGUF")
        .expect("QWEN4EXP_TOKENIZER_GGUF must point to the first Q3 shard");
    let gguf = GgufFile::open(path).expect("open released Flash-Next GGUF");
    assert_eq!(gguf.architecture().as_deref(), Some("qwen4exp"));
    let template = gguf
        .get_str("tokenizer.chat_template")
        .expect("released Flash-Next GGUF must declare a chat template");
    assert!(qwen4exp_chat_template_matches(template));

    let mut changed = template.to_owned();
    changed.push(' ');
    assert!(!qwen4exp_chat_template_matches(&changed));
}

#[test]
fn qwen4exp_stop_validation_honors_valid_producer_vectors() {
    validate_qwen4exp_stop_tokens(&[248_046], 248_320).unwrap();
    validate_qwen4exp_stop_tokens(&[42], 248_320).unwrap();
    validate_qwen4exp_stop_tokens(&[42, 43], 248_320).unwrap();
    assert!(validate_qwen4exp_stop_tokens(&[], 248_320).is_err());
    assert!(validate_qwen4exp_stop_tokens(&[-1], 248_320).is_err());
    assert!(validate_qwen4exp_stop_tokens(&[248_320], 248_320).is_err());
    assert!(
        validate_qwen4exp_stop_tokens(&vec![42; QWEN4EXP_MAX_STOP_TOKENS + 1], 248_320).is_err()
    );
}

#[test]
fn qwen4exp_serial_mode_rejects_inert_advanced_options() {
    let make_args =
        || Args::try_parse_from(["qwen", "--model", "model.gguf", "--prompt", "hello"]).unwrap();
    validate_qwen4exp_generation_mode(&make_args(), ExplicitCliOptions::default()).unwrap();

    for explicit in [
        ExplicitCliOptions {
            durable_prefix_cache_max_mib: true,
            ..ExplicitCliOptions::default()
        },
        ExplicitCliOptions {
            durable_prefix_cache_max_entry_mib: true,
            ..ExplicitCliOptions::default()
        },
        ExplicitCliOptions {
            durable_prefix_cache_min_tokens: true,
            ..ExplicitCliOptions::default()
        },
        ExplicitCliOptions {
            deepseek_v4_multigroup_selector: true,
            ..ExplicitCliOptions::default()
        },
    ] {
        assert!(validate_qwen4exp_generation_mode(&make_args(), explicit).is_err());
    }

    let mut requests = make_args();
    requests.requests_jsonl = Some(PathBuf::from("requests.jsonl"));
    assert!(validate_qwen4exp_generation_mode(&requests, ExplicitCliOptions::default()).is_err());
    let mut concurrency = make_args();
    concurrency.concurrency = Some(2);
    assert!(
        validate_qwen4exp_generation_mode(&concurrency, ExplicitCliOptions::default()).is_err()
    );
}

#[test]
fn reasoning_controls_bind_through_each_family_table() {
    use crate::messages::Qwen38ReasoningEffort;
    use crate::open_responses::items::QwenTemplate;
    use crate::prompt_template::{
        QwenBoundGeneration, QwenReasoningControls, QwenUserPromptProtocol,
    };

    // Qwen3.8: every advertised level binds; `none` is the no-thinking mode;
    // omission is upstream xhigh; a foreign level fails with the level list.
    assert_eq!(
        Qwen38GenerationMode::parse(None, false).unwrap(),
        Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Xhigh)
    );
    for (name, expected) in Qwen38GenerationMode::LEVELS {
        assert_eq!(
            Qwen38GenerationMode::parse(Some(name), false).unwrap(),
            *expected
        );
    }
    assert_eq!(
        Qwen38GenerationMode::parse(Some("none"), false).unwrap(),
        Qwen38GenerationMode::NoThinking
    );
    assert_eq!(
        Qwen38GenerationMode::parse(None, true).unwrap(),
        Qwen38GenerationMode::NoThinking
    );
    let err = Qwen38GenerationMode::parse(Some("high"), false).unwrap_err();
    assert_eq!(err.code, "reasoning_effort_invalid");
    assert!(err.message.contains("none|low|medium|xhigh"), "{err}");
    assert_eq!(
        Qwen38GenerationMode::parse(Some("low"), true)
            .unwrap_err()
            .code,
        "reasoning_conflict"
    );

    // DeepSeek: none/low/high/max; omission is chat; `xhigh` is foreign.
    assert_eq!(
        DeepSeekV4Reasoning::parse(None).unwrap(),
        DeepSeekV4Reasoning::None
    );
    assert_eq!(
        DeepSeekV4Reasoning::parse(Some("max")).unwrap(),
        DeepSeekV4Reasoning::Max
    );
    assert!(
        DeepSeekV4Reasoning::parse(Some("xhigh"))
            .unwrap_err()
            .message
            .contains("none|low|high|max")
    );

    // Ordinary non-3.8 Qwen has no effort control and binds only the
    // no-thinking transition, which needs a pinned template.
    let pinned = QwenUserPromptProtocol::for_test(false, QwenTemplate::Qwen36);
    assert_eq!(
        pinned
            .bind(QwenReasoningControls {
                effort: None,
                no_thinking: true
            })
            .unwrap(),
        QwenBoundGeneration::Template(QwenGenerationMode::NoThinking)
    );
    assert_eq!(
        pinned
            .bind(QwenReasoningControls {
                effort: Some("low"),
                no_thinking: false
            })
            .unwrap_err()
            .code,
        "reasoning_effort_unsupported"
    );
    let generic = QwenUserPromptProtocol::for_test(false, QwenTemplate::Generic);
    assert_eq!(
        generic
            .bind(QwenReasoningControls {
                effort: None,
                no_thinking: true
            })
            .unwrap_err()
            .code,
        "no_thinking_unsupported"
    );
    // The advertised capability is derived from the same tables.
    let q38 = QwenUserPromptProtocol::for_test(true, QwenTemplate::Qwen38);
    assert_eq!(
        q38.reasoning_capability().levels,
        Qwen38GenerationMode::level_names()
    );
    assert!(pinned.reasoning_capability().levels.is_empty());
    assert!(matches!(
        generic.reasoning_capability().no_thinking,
        crate::prompt_template::Support::Unsupported { .. }
    ));
}

#[test]
fn input_forms_bind_through_one_family_table() {
    use crate::open_responses::items::QwenTemplate;
    use crate::prompt_template::{
        InputCapability, QwenUserPromptProtocol, Support, qwen_tools_support,
    };

    // Ordinary Qwen: plain chat is ChatML everywhere; tools need the
    // released tool block, so only a pinned template advertises them, and
    // the refusal a lane raises is the advertised one.
    for template in [
        QwenTemplate::Qwen35,
        QwenTemplate::Qwen36,
        QwenTemplate::Qwen38,
    ] {
        let protocol = QwenUserPromptProtocol::for_test(template == QwenTemplate::Qwen38, template);
        assert_eq!(
            protocol.input_capability(),
            InputCapability::all_supported(),
            "{template:?}"
        );
        assert!(qwen_tools_support(template).require().is_ok());
    }
    let generic = QwenUserPromptProtocol::for_test(false, QwenTemplate::Generic).input_capability();
    assert_eq!(generic.raw, Support::Supported);
    assert_eq!(generic.user, Support::Supported);
    assert_eq!(generic.messages, Support::Supported);
    let refusal = generic.tools.require().unwrap_err();
    assert_eq!(refusal.code, "tools_require_pinned_template");
    assert_eq!(
        qwen_tools_support(QwenTemplate::Generic)
            .require()
            .unwrap_err(),
        refusal
    );

    // Every templated form shares one refusal when the protocol is absent;
    // raw text still renders.
    let raw_only = InputCapability::raw_only("prompt_protocol_unsupported", "x".into());
    assert_eq!(raw_only.raw, Support::Supported);
    for form in [&raw_only.user, &raw_only.messages, &raw_only.tools] {
        assert_eq!(
            form.require().unwrap_err().code,
            "prompt_protocol_unsupported"
        );
    }
    assert_eq!(
        InputCapability::none("unknown_family", "x".into())
            .raw
            .require()
            .unwrap_err()
            .code,
        "unknown_family"
    );

    // The projection carries the same status/code vocabulary as reasoning.
    let json = serde_json::to_value(&generic).unwrap();
    assert_eq!(json["user"], serde_json::json!({"status": "supported"}));
    assert_eq!(json["tools"]["status"], "unsupported");
    assert_eq!(json["tools"]["code"], "tools_require_pinned_template");
}

#[test]
fn deepseek_v4_multigroup_selector_cli_contract_is_explicit_and_bounded() {
    let default =
        Args::try_parse_from(["qwen", "--model", "model.gguf", "--prompt", "hello"]).unwrap();
    assert_eq!(
        default.deepseek_v4_multigroup_selector,
        DeepSeekV4MultigroupSelectorArg::Auto
    );

    let explicit = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--deepseek-v4-multigroup-selector",
        "qualified-experimental",
    ])
    .unwrap();
    assert_eq!(
        explicit.deepseek_v4_multigroup_selector,
        DeepSeekV4MultigroupSelectorArg::QualifiedExperimental
    );
    validate_deepseek_v4_multigroup_selector_scope(&explicit).unwrap();
    validate_deepseek_v4_multigroup_selector_family(
        explicit.deepseek_v4_multigroup_selector,
        Some(ModelFamily::DeepSeek4),
    )
    .unwrap();
    assert!(
        validate_deepseek_v4_multigroup_selector_family(
            explicit.deepseek_v4_multigroup_selector,
            None,
        )
        .unwrap_err()
        .to_string()
        .contains("requires a DeepSeek V4 model")
    );

    let jsonl = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--requests-jsonl",
        "requests.jsonl",
        "--deepseek-v4-multigroup-selector=qualified-experimental",
    ])
    .unwrap();
    validate_deepseek_v4_multigroup_selector_scope(&jsonl).unwrap();
    validate_deepseek_v4_requests_mode(&jsonl, ExplicitCliOptions::default()).unwrap();

    let no_request = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--deepseek-v4-multigroup-selector=qualified-experimental",
    ])
    .unwrap();
    assert!(
        validate_deepseek_v4_multigroup_selector_scope(&no_request)
            .unwrap_err()
            .to_string()
            .contains("requires a generation request")
    );
    assert!(
        Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--deepseek-v4-multigroup-selector=force",
        ])
        .is_err()
    );

    let shallow = DeepSeekV4SessionCapacity::for_forward_limit(4_096, 1_048_576).unwrap();
    let off = DeepSeekV4MultigroupSelectorPlan::new(
        DeepSeekV4MultigroupSelectorArg::Off,
        "Apple M4 Pro",
        shallow,
    )
    .unwrap();
    assert!(!off.sealed());
    let auto = DeepSeekV4MultigroupSelectorPlan::new(
        DeepSeekV4MultigroupSelectorArg::Auto,
        "Apple M4 Max",
        DeepSeekV4SessionCapacity::for_forward_limit(786_432, 1_048_576).unwrap(),
    )
    .unwrap();
    assert!(!auto.sealed());
    assert!(
        auto.completion_record_from_values("single_turn", false, 21, 3)
            .is_ok()
    );
    assert!(
        DeepSeekV4MultigroupSelectorPlan::new(
            DeepSeekV4MultigroupSelectorArg::QualifiedExperimental,
            "Apple M4 Pro",
            DeepSeekV4SessionCapacity::for_forward_limit(786_432, 1_048_576).unwrap(),
        )
        .unwrap_err()
        .to_string()
        .contains("requires Apple M4 Max")
    );
    let unreachable_error = DeepSeekV4MultigroupSelectorPlan::new(
        DeepSeekV4MultigroupSelectorArg::QualifiedExperimental,
        "Apple M4 Max",
        DeepSeekV4SessionCapacity::for_forward_limit(786_431, 1_048_576).unwrap(),
    )
    .unwrap_err();
    assert!(format!("{unreachable_error:#}").contains("max_visible_rows=196607"));

    let qualified = DeepSeekV4MultigroupSelectorPlan::new(
        DeepSeekV4MultigroupSelectorArg::QualifiedExperimental,
        "Apple M4 Max",
        DeepSeekV4SessionCapacity::for_forward_limit(786_432, 1_048_576).unwrap(),
    )
    .unwrap();
    assert!(qualified.sealed());
    assert_eq!(
        serde_json::to_value(qualified.session_record("single_turn")).unwrap(),
        serde_json::json!({
            "schema_version": 1,
            "kind": "session_policy",
            "scope": "single_turn",
            "requested": "qualified_experimental",
            "sealed": true,
            "device_name": "Apple M4 Max",
            "device_qualified": true,
            "forward_limit": 786432,
            "physical_capacity_rows": 196608,
            "max_reachable_visible_rows": 196608,
            "frozen_min_visible_rows": 196608,
            "frozen_max_capacity_rows": 262144,
            "frozen_min_capacity_occupancy": "3/4",
            "fallback": "radix4_for_packed_and_ineligible_singleton",
        })
    );
    assert_eq!(
        serde_json::to_value(
            qualified
                .completion_record_from_values("single_turn", true, 21, 3)
                .unwrap()
        )
        .unwrap(),
        serde_json::json!({
            "schema_version": 1,
            "kind": "session_completion",
            "scope": "single_turn",
            "requested": "qualified_experimental",
            "sealed": true,
            "multigroup_invocations": 21,
            "ineligible_singleton_radix4_invocations": 3,
        })
    );
    assert!(
        qualified
            .completion_record_from_values("single_turn", false, 0, 0)
            .is_err()
    );
    assert!(
        off.completion_record_from_values("single_turn", false, 1, 0)
            .is_err()
    );
}

#[test]
fn deepseek_v4_prefill_chunks_every_retained_prompt_interval() {
    assert_eq!(parse_deepseek_v4_prefill_chunk_tokens(None).unwrap(), 4_096);
    assert_eq!(
        parse_deepseek_v4_prefill_chunk_tokens(Some("128")).unwrap(),
        128
    );
    assert_eq!(
        parse_deepseek_v4_prefill_chunk_tokens(Some("512")).unwrap(),
        512
    );
    assert_eq!(
        parse_deepseek_v4_prefill_chunk_tokens(Some("4096")).unwrap(),
        4_096
    );
    assert!(parse_deepseek_v4_prefill_chunk_tokens(Some("0")).is_err());
    assert!(parse_deepseek_v4_prefill_chunk_tokens(Some("4097")).is_err());
    assert!(parse_deepseek_v4_prefill_chunk_tokens(Some("nope")).is_err());
    let lengths = |tokens, chunk| {
        deepseek_v4_prefill_chunk_ranges(tokens, chunk)
            .into_iter()
            .map(|range| range.len())
            .collect::<Vec<_>>()
    };
    assert_eq!(lengths(2_385, 4_096), vec![2_048, 337]);
    assert_eq!(lengths(6_642, 4_096), vec![4_096, 2_048, 498]);
    assert_eq!(lengths(8_192, 4_096), vec![4_096, 4_096]);
    assert_eq!(lengths(2_385, 2_048), vec![2_048, 337]);
    assert_eq!(lengths(2_385, 3_000), vec![2_385]);
    assert_eq!(deepseek_v4_packed_chunk_count(0, 512), 0);
    assert_eq!(deepseek_v4_packed_chunk_count(1, 512), 0);
    assert_eq!(deepseek_v4_packed_chunk_count(2, 512), 1);
    assert_eq!(
        deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PREFILL_MAX_TOKENS, 512),
        8
    );
    assert_eq!(
        deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PREFILL_MAX_TOKENS + 1, 512),
        9
    );
    assert_eq!(
        deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY, 512),
        2_048
    );
    assert_eq!(
        deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PREFILL_MAX_TOKENS, 2_048),
        2
    );
    assert_eq!(
        deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY, 2_048),
        512
    );
    assert_eq!(
        deepseek_v4_packed_chunk_count(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY, 128),
        8_192
    );
}

#[test]
fn deepseek_v4_snapshot_keeps_one_uncached_endpoint_token() {
    assert!(deepseek_v4_snapshot_publish_prefix(0).is_err());
    assert!(deepseek_v4_snapshot_publish_prefix(1).is_err());
    assert_eq!(deepseek_v4_snapshot_publish_prefix(2).unwrap(), 1);
    assert_eq!(
        deepseek_v4_snapshot_publish_prefix(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY).unwrap(),
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY - 1
    );

    let prompt = [35, 201, 200, 34];
    assert_eq!(
        deepseek_v4_snapshot_restored_prefix_len(&prompt[..3], &prompt).unwrap(),
        3
    );
    assert!(deepseek_v4_snapshot_restored_prefix_len(&prompt, &prompt).is_err());
    assert!(deepseek_v4_snapshot_restored_prefix_len(&[35, 200], &prompt).is_err());
}

#[test]
fn deepseek_v4_snapshot_identity_cache_requires_private_owned_directories() {
    let root = std::env::temp_dir().join(format!(
        "qwen-dsv4-cli-cache-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&root)
        .unwrap();
    let snapshot_path = root.join("prefix.ds4c");
    assert_eq!(
        deepseek_v4_snapshot_parent(&snapshot_path).unwrap(),
        root.as_path()
    );
    let cache = deepseek_v4_snapshot_identity_cache(&root).unwrap();
    assert_eq!(
        std::fs::symlink_metadata(cache.root())
            .unwrap()
            .permissions()
            .mode()
            & 0o077,
        0
    );

    std::fs::remove_dir(cache.root()).unwrap();
    std::fs::DirBuilder::new()
        .mode(0o755)
        .create(cache.root())
        .unwrap();
    std::fs::set_permissions(cache.root(), std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(deepseek_v4_snapshot_identity_cache(&root).is_err());

    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o777)).unwrap();
    assert!(deepseek_v4_snapshot_parent(&snapshot_path).is_err());
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn deepseek_v4_cli_accepts_only_bounded_single_turn_surfaces() {
    let raw = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--temp",
        "0.7",
        "--trace-request",
        "trace.txt",
        "--no-special-tokens",
    ])
    .unwrap();
    validate_deepseek_v4_generation_mode(&raw, ExplicitCliOptions::default()).unwrap();

    let snapshot = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--deepseek-v4-snapshot",
        "prefix.ds4c",
    ])
    .unwrap();
    validate_deepseek_v4_generation_mode(&snapshot, ExplicitCliOptions::default()).unwrap();

    let unsupported = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--prompt-lookup",
        "--prefill-chunk",
        "auto",
        "--max-context-tokens",
        "255",
        "--cache-prefix-tokens",
        "4",
        "--request-stats",
        "stats.jsonl",
    ])
    .unwrap();
    let error = validate_deepseek_v4_generation_mode(&unsupported, ExplicitCliOptions::default())
        .unwrap_err()
        .to_string();
    for option in [
        "--prompt-lookup",
        "--prefill-chunk",
        "--max-context-tokens",
        "--cache-prefix-tokens",
        "--request-stats",
    ] {
        assert!(error.contains(option), "missing {option:?} from {error:?}");
    }

    let messages = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--messages",
        "messages.json",
    ])
    .unwrap();
    validate_deepseek_v4_generation_mode(&messages, ExplicitCliOptions::default()).unwrap();

    let messages_with_qwen_policy = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--messages",
        "messages.json",
        "--messages-preserve-thinking",
        "--messages-no-generation-prompt",
    ])
    .unwrap();
    let error = validate_deepseek_v4_generation_mode(
        &messages_with_qwen_policy,
        ExplicitCliOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("--messages-preserve-thinking"));
    assert!(error.contains("--messages-no-generation-prompt"));

    let messages_with_strip = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--messages",
        "messages.json",
        "--messages-strip-thinking",
    ])
    .unwrap();
    validate_deepseek_v4_generation_mode(&messages_with_strip, ExplicitCliOptions::default())
        .unwrap();

    let preserve_with_tier = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--messages",
        "messages.json",
        "--messages-preserve-thinking",
        "--reasoning",
        "low",
    ])
    .unwrap();
    validate_deepseek_v4_generation_mode(&preserve_with_tier, ExplicitCliOptions::default())
        .unwrap();

    assert!(
        Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "messages.json",
            "--messages-strip-thinking",
            "--preserve-reasoning",
        ])
        .is_err(),
        "strip-thinking must conflict with preserve-reasoning at the parser"
    );

    let durable = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--messages",
        "messages.json",
        "--messages-strip-thinking",
        "--durable-prefix-cache",
        "cache",
        "--durable-prefix-cache-min-tokens",
        "1024",
        "--durable-prefix-cache-max-mib",
        "4096",
        "--durable-prefix-cache-max-entry-mib",
        "4096",
    ])
    .unwrap();
    validate_deepseek_v4_generation_mode(
        &durable,
        ExplicitCliOptions {
            durable_prefix_cache_min_tokens: true,
            durable_prefix_cache_max_mib: true,
            durable_prefix_cache_max_entry_mib: true,
            ..ExplicitCliOptions::default()
        },
    )
    .unwrap();

    let jsonl = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--requests-jsonl",
        "requests.jsonl",
    ])
    .unwrap();
    assert!(
        validate_deepseek_v4_generation_mode(&jsonl, ExplicitCliOptions::default())
            .unwrap_err()
            .to_string()
            .contains("--requests-jsonl")
    );

    // Requests mode: file mode derives its budget and rejects the stdin
    // override; stdin mode requires it; shared unsupported flags still
    // fail closed; the snapshot flag stays parser-excluded.
    let file_mode_with_budget = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--requests-jsonl",
        "requests.jsonl",
        "--max-context-tokens",
        "4096",
    ])
    .unwrap();
    assert!(
        validate_deepseek_v4_requests_mode(&file_mode_with_budget, ExplicitCliOptions::default(),)
            .unwrap_err()
            .to_string()
            .contains("remove --max-context-tokens")
    );
    let stdin_without_budget =
        Args::try_parse_from(["qwen", "--model", "model.gguf", "--requests-jsonl", "-"]).unwrap();
    assert!(
        validate_deepseek_v4_requests_mode(&stdin_without_budget, ExplicitCliOptions::default(),)
            .unwrap_err()
            .to_string()
            .contains("supply --max-context-tokens")
    );
    let stdin_with_budget = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--requests-jsonl",
        "-",
        "--max-context-tokens",
        "4096",
    ])
    .unwrap();
    validate_deepseek_v4_requests_mode(&stdin_with_budget, ExplicitCliOptions::default()).unwrap();
    assert_eq!(
        deepseek_v4_forward_budget_for_context_limit(4_096).unwrap(),
        4_095
    );
    assert_eq!(
        deepseek_v4_forward_budget_for_context_limit(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY)
            .unwrap(),
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY - 1
    );
    assert!(deepseek_v4_forward_budget_for_context_limit(1).is_err());
    assert!(
        deepseek_v4_forward_budget_for_context_limit(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY + 1)
            .is_err()
    );
    validate_deepseek_v4_request_context_limit("exact", 4_000, 96, 4_096).unwrap();
    let context_error = validate_deepseek_v4_request_context_limit("overflow", 4_000, 97, 4_096)
        .unwrap_err()
        .to_string();
    assert!(context_error.contains("4097 logical context tokens"));
    assert!(
        validate_deepseek_v4_request_context_limit("checked-add", usize::MAX, 1, usize::MAX,)
            .unwrap_err()
            .to_string()
            .contains("overflow")
    );
    let requests_with_durable_cache = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--requests-jsonl",
        "requests.jsonl",
        "--durable-prefix-cache",
        "/tmp/cache",
    ])
    .unwrap();
    assert!(
        validate_deepseek_v4_requests_mode(
            &requests_with_durable_cache,
            ExplicitCliOptions::default(),
        )
        .unwrap_err()
        .to_string()
        .contains("--durable-prefix-cache")
    );
    assert!(
        Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "requests.jsonl",
            "--deepseek-v4-snapshot",
            "/tmp/s.ds4ckpt",
        ])
        .is_err(),
        "snapshot flag must stay parser-excluded from requests mode"
    );

    // `--reasoning`/`--preserve-reasoning` require `--messages` in
    // pre-open validation and map onto the release encoder contract.
    let raw_reasoning = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "-p",
        "hi",
        "--reasoning",
        "high",
    ])
    .unwrap();
    assert!(
        validate_request_before_model_open(&raw_reasoning)
            .unwrap_err()
            .to_string()
            .contains("require --messages")
    );
    let census_reasoning = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--deepseek-census-json",
        "--reasoning",
        "high",
    ])
    .unwrap();
    assert!(
        validate_deepseek_v4_reasoning_scope(&census_reasoning)
            .unwrap_err()
            .to_string()
            .contains("require --messages")
    );
    for (level, expected) in [
        (None, DeepSeekV4Reasoning::None),
        (Some("none"), DeepSeekV4Reasoning::None),
        (Some("low"), DeepSeekV4Reasoning::Low),
        (Some("high"), DeepSeekV4Reasoning::High),
        (Some("max"), DeepSeekV4Reasoning::Max),
    ] {
        let mut command = vec![
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "messages.json",
        ];
        if let Some(level) = level {
            command.extend(["--reasoning", level]);
        }
        let args = Args::try_parse_from(command).unwrap();
        let options = deepseek_v4_encode_options(&args).unwrap();
        assert_eq!(options.reasoning, expected);
        assert!(!options.preserve_reasoning);
    }
    let preserve_without_thinking = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--messages",
        "messages.json",
        "--preserve-reasoning",
    ])
    .unwrap();
    assert!(
        deepseek_v4_encode_options(&preserve_without_thinking)
            .unwrap_err()
            .to_string()
            .contains("requires --reasoning low, high, or max")
    );
    let preserve = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--messages",
        "messages.json",
        "--reasoning",
        "max",
        "--preserve-reasoning",
    ])
    .unwrap();
    let options = deepseek_v4_encode_options(&preserve).unwrap();
    assert_eq!(options.reasoning, DeepSeekV4Reasoning::Max);
    assert!(options.preserve_reasoning);
    let preserve_low = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--messages",
        "messages.json",
        "--reasoning",
        "low",
        "--preserve-reasoning",
    ])
    .unwrap();
    let options = deepseek_v4_encode_options(&preserve_low).unwrap();
    assert_eq!(options.reasoning, DeepSeekV4Reasoning::Low);
    assert!(options.preserve_reasoning);

    let matches = Args::command()
        .try_get_matches_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--prefill-chunk",
            "1024",
            "--prefix-cache-max-mib",
            "16384",
            "--cache-prefix-auto-min-tokens",
            "1024",
            "--durable-prefix-cache-max-mib",
            "32768",
            "--durable-prefix-cache-max-entry-mib",
            "16384",
            "--durable-prefix-cache-min-tokens",
            "1024",
        ])
        .unwrap();
    let explicit = ExplicitCliOptions::from_matches(&matches);
    let explicit_defaults = Args::from_arg_matches(&matches).unwrap();
    let error = validate_deepseek_v4_generation_mode(&explicit_defaults, explicit)
        .unwrap_err()
        .to_string();
    for option in [
        "--prefill-chunk",
        "--prefix-cache-max-mib",
        "--cache-prefix-auto-min-tokens",
        "--durable-prefix-cache-max-mib",
        "--durable-prefix-cache-max-entry-mib",
        "--durable-prefix-cache-min-tokens",
    ] {
        assert!(error.contains(option), "missing {option:?} from {error:?}");
    }
}

#[test]
#[ignore = "requires the local DeepSeek V4 Flash-0731 IQ3 fixture"]
fn deepseek_v4_0731_message_prompts_match_flash_vocab() {
    const MODEL: &str = concat!(
        "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/",
        "DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
    );
    fn message(role: &str, content: &str) -> messages::ChatMessage {
        messages::ChatMessage {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }
    fn render_ids(tokenizer: &Tokenizer, messages: &[messages::ChatMessage]) -> Vec<i32> {
        let prompt = messages::render_deepseek_v4_0731_messages_prompt(
            messages,
            &[],
            messages::DeepSeekV4EncodeOptions::default(),
        )
        .unwrap();
        tokenizer.encode(&prompt, false).unwrap()
    }

    assert!(Path::new(MODEL).exists(), "missing DS4 fixture");
    let tokenizer = Tokenizer::open(MODEL).expect("open DS4 tokenizer");
    assert_eq!(
        render_ids(&tokenizer, &[message("user", "Hello")]),
        [0, 128_803, 19_923, 128_804, 128_822]
    );
    assert_eq!(
        render_ids(
            &tokenizer,
            &[message("system", "Be exact."), message("user", "Hello")]
        ),
        [0, 7_153, 6_319, 16, 128_803, 19_923, 128_804, 128_822]
    );
    assert_eq!(
        render_ids(
            &tokenizer,
            &[
                message("system", "Be exact."),
                message("user", "Hello"),
                message("assistant", "Hi!"),
                message("user", "上海 🙂"),
            ]
        ),
        [
            0, 7_153, 6_319, 16, 128_803, 19_923, 128_804, 128_822, 23_166, 3, 1, 128_803, 7_241,
            68_139, 128_804, 128_822,
        ]
    );
}

#[test]
fn prefill_chunk_argument_preserves_numeric_json_and_accepts_auto() {
    let fixed = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--prefill-chunk",
        "2048",
    ])
    .unwrap();
    let auto = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--prefill-chunk",
        "auto",
    ])
    .unwrap();

    assert_eq!(fixed.prefill_chunk, PrefillChunkArg::Fixed(2048));
    assert_eq!(auto.prefill_chunk, PrefillChunkArg::Auto);
    assert_eq!(serde_json::to_string(&fixed.prefill_chunk).unwrap(), "2048");
    assert_eq!(
        serde_json::to_string(&auto.prefill_chunk).unwrap(),
        "\"auto\""
    );
}

#[test]
fn sampling_request_contract_supports_cli_defaults_and_jsonl_overrides() {
    let args = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--temp",
        "0.7",
        "--seed",
        "99",
    ])
    .unwrap();
    assert_eq!(
        cli_sampling_config(&args).unwrap(),
        SamplingConfig::qwen_chat(99)
    );

    let request: JsonlRequest = serde_json::from_str(
        r#"{"prompt":"hello","sampling":{"temp":1.0,"top_k":8,"top_p":0.9,"min_p":0.1,"seed":7}}"#,
    )
    .unwrap();
    assert_eq!(
        request_sampling_config(&request, &args).unwrap(),
        SamplingConfig {
            temperature: 1.0,
            top_k: 8,
            top_p: 0.9,
            min_p: 0.1,
            seed: 7,
        }
    );
    assert!(validate_sampling_decode_policy(SamplingConfig::qwen_chat(7), true).is_err());
    assert!(validate_sampling_decode_policy(SamplingConfig::default(), true).is_ok());

    let prompt_lookup_args = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--requests-jsonl",
        "-",
        "--prompt-lookup",
        "--temp",
        "0.7",
    ])
    .unwrap();
    cli_sampling_config(&prompt_lookup_args).unwrap();
    let greedy_override: JsonlRequest =
        serde_json::from_str(r#"{"prompt":"hello","sampling":{"temperature":0.0}}"#).unwrap();
    let effective = request_sampling_config(&greedy_override, &prompt_lookup_args).unwrap();
    assert!(validate_sampling_decode_policy(effective, true).is_ok());

    let typo =
        serde_json::from_str::<JsonlRequest>(r#"{"prompt":"hello","sampling":{"temprature":0.7}}"#);
    assert!(typo.is_err(), "sampling field typos must fail closed");
}

#[test]
fn muse_glimmer_sampling_uses_released_defaults_and_explicit_overrides() {
    let matches = Args::command()
        .try_get_matches_from(["qwen", "run", "-m", "model.gguf", "--user", "hello"])
        .unwrap();
    let (_, run_matches) = matches.subcommand().unwrap();
    let explicit = ExplicitCliOptions::from_matches(run_matches);
    let mut args = Args::from_arg_matches(&matches).unwrap();
    let invocation = cli::normalize(&mut args);
    invocation.apply_option_overrides(&mut args);
    assert_eq!(
        muse_glimmer_sampling_config(&args, explicit).unwrap(),
        SamplingConfig::muse_glimmer(42)
    );

    let matches = Args::command()
        .try_get_matches_from([
            "qwen",
            "-m",
            "model.gguf",
            "--prompt",
            "hello",
            "--top-k",
            "0",
        ])
        .unwrap();
    let explicit = ExplicitCliOptions::from_matches(&matches);
    let args = Args::from_arg_matches(&matches).unwrap();
    assert_eq!(
        muse_glimmer_sampling_config(&args, explicit).unwrap(),
        SamplingConfig {
            top_k: 0,
            ..SamplingConfig::muse_glimmer(42)
        }
    );

    let matches = Args::command()
        .try_get_matches_from([
            "qwen",
            "run",
            "-m",
            "model.gguf",
            "--user",
            "hello",
            "--temp",
            "0",
            "--top-k",
            "0",
            "--top-p",
            "1",
            "--min-p",
            "0.1",
            "--seed",
            "7",
        ])
        .unwrap();
    let (_, run_matches) = matches.subcommand().unwrap();
    let explicit = ExplicitCliOptions::from_matches(run_matches);
    let mut args = Args::from_arg_matches(&matches).unwrap();
    let invocation = cli::normalize(&mut args);
    invocation.apply_option_overrides(&mut args);
    assert_eq!(
        muse_glimmer_sampling_config(&args, explicit).unwrap(),
        SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.1,
            seed: 7,
        }
    );
}

#[test]
fn request_schema_versions_include_exact_generation_telemetry() {
    assert_eq!(
        request_schema_version(
            PrefillChunkArg::Fixed(1024),
            false,
            false,
            false,
            false,
            false,
            false,
        ),
        7
    );
    assert_eq!(
        request_schema_version(
            PrefillChunkArg::Fixed(1024),
            true,
            false,
            false,
            false,
            false,
            false,
        ),
        8
    );
    assert_eq!(
        request_schema_version(
            PrefillChunkArg::Fixed(2048),
            false,
            true,
            true,
            false,
            false,
            false,
        ),
        8
    );
    assert_eq!(
        request_schema_version(
            PrefillChunkArg::Auto,
            false,
            false,
            false,
            false,
            false,
            false,
        ),
        9
    );
    assert_eq!(
        request_schema_version(PrefillChunkArg::Auto, true, true, true, false, false, false,),
        9
    );
    assert_eq!(
        request_schema_version(
            PrefillChunkArg::Fixed(1024),
            false,
            false,
            false,
            true,
            false,
            false,
        ),
        10
    );
    assert_eq!(
        request_schema_version(
            PrefillChunkArg::Fixed(1024),
            false,
            false,
            false,
            true,
            true,
            false,
        ),
        11
    );
    assert_eq!(
        request_schema_version(
            PrefillChunkArg::Fixed(1024),
            false,
            false,
            false,
            true,
            false,
            true,
        ),
        12
    );
}

#[test]
fn sampling_attribution_cli_is_narrow_and_fail_closed() {
    let exact = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt-file",
        "prompt.txt",
        "--tokens",
        "128",
        "--temp",
        "0.7",
        "--top-k",
        "200",
        "--top-p",
        "1.0",
        "--min-p",
        "0.05",
        "--seed",
        "42",
        "--prefill-chunk",
        "1024",
        "--max-context-tokens",
        "1024",
        "--prefix-cache-max-mib",
        "0",
        "--cache-prefix-auto-min-tokens",
        "0",
        "--request-timings",
        "timing.jsonl",
        "--sampling-attribution",
    ])
    .unwrap();
    assert!(validate_sampling_attribution_mode(&exact).is_ok());

    let wrong_tokens = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt-file",
        "prompt.txt",
        "--tokens",
        "127",
        "--temp",
        "0.7",
        "--max-context-tokens",
        "1024",
        "--prefix-cache-max-mib",
        "0",
        "--cache-prefix-auto-min-tokens",
        "0",
        "--request-timings",
        "timing.jsonl",
        "--sampling-attribution",
    ])
    .unwrap();
    assert!(
        validate_sampling_attribution_mode(&wrong_tokens)
            .unwrap_err()
            .to_string()
            .contains("--tokens 128")
    );

    assert!(
        Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt-file",
            "prompt.txt",
            "--request-timings",
            "timing.jsonl",
            "--request-timing-warm-followup",
            "--sampling-attribution",
        ])
        .is_err(),
        "warm follow-up must conflict at clap parsing"
    );
}

#[test]
fn sampled_structural_cli_is_hidden_bounded_and_fail_closed() {
    let mut help = Vec::new();
    Args::command().write_long_help(&mut help).unwrap();
    assert!(
        !String::from_utf8(help)
            .unwrap()
            .contains("sampled-structural")
    );

    let exact = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt-file",
        "prompt.txt",
        "--temp",
        "0.7",
        "--top-k",
        "200",
        "--prefix-cache-max-mib",
        "0",
        "--cache-prefix-auto-min-tokens",
        "0",
        "--request-timings",
        "timing.jsonl",
        "--sampled-structural",
    ])
    .unwrap();
    assert!(validate_sampled_structural_mode(&exact).is_ok());

    let without_timings = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--temp",
        "0.7",
        "--top-k",
        "200",
        "--prefix-cache-max-mib",
        "0",
        "--cache-prefix-auto-min-tokens",
        "0",
        "--sampled-structural",
    ])
    .unwrap();
    assert!(validate_sampled_structural_mode(&without_timings).is_ok());

    let default_cache = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--temp",
        "0.7",
        "--top-k",
        "200",
        "--request-timings",
        "timing.jsonl",
        "--sampled-structural",
    ])
    .unwrap();
    assert!(
        validate_sampled_structural_mode(&default_cache)
            .unwrap_err()
            .to_string()
            .contains("zero RAM prefix-cache admission")
    );

    let unbounded = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--temp",
        "0.7",
        "--top-k",
        "0",
        "--prefix-cache-max-mib",
        "0",
        "--cache-prefix-auto-min-tokens",
        "0",
        "--request-timings",
        "timing.jsonl",
        "--sampled-structural",
    ])
    .unwrap();
    assert!(
        validate_sampled_structural_mode(&unbounded)
            .unwrap_err()
            .to_string()
            .contains("positive temperature and top-k")
    );

    assert!(
        Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--request-timings",
            "timing.jsonl",
            "--sampling-attribution",
            "--sampled-structural",
        ])
        .is_err(),
        "sampling attribution must conflict at clap parsing"
    );
}

#[test]
fn sampled_structural_telemetry_has_the_frozen_key_set() {
    let value = serde_json::to_value(SampledStructuralTelemetry::default()).unwrap();
    let object = value.as_object().expect("structural telemetry object");
    let actual: std::collections::BTreeSet<_> = object.keys().map(String::as_str).collect();
    let expected = std::collections::BTreeSet::from([
        "version",
        "algorithm_version",
        "path",
        "prompt_owned_bounded_calls",
        "borrowed_transition_calls",
        "resident_head_wait_calls",
        "validated_shared_row_calls",
        "fallback_calls",
        "input_logits_total",
        "input_logits_min",
        "input_logits_max",
        "retained_top_k_total",
        "retained_top_k_min",
        "retained_top_k_max",
        "max_heap_len",
        "max_heap_capacity",
        "full_candidate_vector_allocations",
        "transition_logits_copy_bytes",
        "extra_command_buffers",
        "gpu_sampling_dispatches",
    ]);
    assert_eq!(actual, expected);
}

#[test]
fn auto_prefill_jsonl_cache_gate_requires_no_entries_or_snapshot() {
    assert!(auto_prefill_cache_safe(0, None));
    assert!(!auto_prefill_cache_safe(1, None));
    assert!(!auto_prefill_cache_safe(0, Some(1)));
    assert!(!auto_prefill_cache_safe(1, Some(1)));

    for source in [
        CachePrefixSource::Request,
        CachePrefixSource::Cli,
        CachePrefixSource::Auto,
    ] {
        let (selected, selected_source) = selected_cache_prefix_from_value(4_000, 10_000, source);
        assert_eq!(selected, Some(4_000));
        assert_eq!(selected_source, source);
        assert!(!auto_prefill_cache_safe(0, selected));
    }
}

#[test]
fn cache_promotion_uses_restored_not_logical_prefix_depth() {
    assert!(cache_prefix_needs_extension(3, 2));
    assert!(!cache_prefix_needs_extension(3, 3));
    assert!(!cache_prefix_needs_extension(2, 3));
}

#[test]
fn one_shot_durable_admission_respects_threshold_and_explicit_override() {
    let automatic = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--durable-prefix-cache",
        "cache",
    ])
    .unwrap();
    assert_eq!(
        selected_single_turn_durable_policy(&automatic, 1023, false),
        DurableCapturePolicy::Disabled
    );
    assert_eq!(
        selected_single_turn_durable_policy(&automatic, 1024, false),
        DurableCapturePolicy::AutomaticPrompt(1024)
    );
    assert_eq!(
        selected_single_turn_durable_lookup_len(&automatic, 2048),
        2048
    );

    let explicit = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--durable-prefix-cache",
        "cache",
        "--cache-prefix-tokens",
        "64",
    ])
    .unwrap();
    assert_eq!(
        selected_single_turn_durable_policy(&explicit, 32, true),
        DurableCapturePolicy::ExplicitPrompt(32)
    );
    assert_eq!(selected_single_turn_durable_lookup_len(&explicit, 2048), 64);

    let completed = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--messages",
        "messages.json",
        "--messages-preserve-thinking",
        "--durable-prefix-cache",
        "cache",
    ])
    .unwrap();
    assert_eq!(
        selected_single_turn_durable_policy(&completed, 1024, true),
        DurableCapturePolicy::AutomaticCompleted
    );

    let disabled = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--durable-prefix-cache",
        "cache",
        "--cache-prefix-tokens",
        "0",
    ])
    .unwrap();
    assert_eq!(
        selected_single_turn_durable_policy(&disabled, 4096, true),
        DurableCapturePolicy::Disabled
    );
    assert_eq!(
        selected_single_turn_durable_lookup_len(&disabled, 4096),
        4096
    );
}

#[test]
fn completed_checkpoint_boundary_tracks_consumed_and_pending_tokens() {
    for generated in [[7].as_slice(), [7, 8, 9].as_slice()] {
        let transitions = generated.len() - 1;
        let boundary =
            derive_completed_checkpoint_boundary(3, generated, transitions, 3 + transitions)
                .unwrap();
        assert_eq!(boundary.consumed_generated_tokens, transitions);
        assert_eq!(boundary.consumed_prefix_len, 3 + transitions);
        assert_eq!(boundary.pending_token, generated[transitions]);
        assert_eq!(
            boundary.consumed_tokens(&[1, 2, 3], generated),
            [1, 2, 3]
                .into_iter()
                .chain(generated[..transitions].iter().copied())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn deepseek_v4_durable_capture_boundary_promotes_growing_prompts() {
    assert_eq!(
        deepseek_v4_durable_capture_prefix_len(1024, true).unwrap(),
        Some(1023)
    );
    assert_eq!(
        deepseek_v4_durable_capture_prefix_len(2048, true).unwrap(),
        Some(2047)
    );
    assert_eq!(
        deepseek_v4_durable_capture_prefix_len(2048, false).unwrap(),
        None
    );
    assert_eq!(
        deepseek_v4_durable_capture_prefix_len(1, true).unwrap(),
        None
    );
}

#[test]
fn completed_checkpoint_boundary_rejects_invalid_generation_state() {
    assert!(derive_completed_checkpoint_boundary(3, &[], 0, 3).is_err());
    assert!(derive_completed_checkpoint_boundary(3, &[7, 8], 0, 3).is_err());
    assert!(derive_completed_checkpoint_boundary(3, &[7, 8], 1, 3).is_err());
}

#[test]
fn prompt_loading_gates_completed_capture_on_canonical_history() {
    let root = std::env::temp_dir().join(format!("qwen-completed-policy-{}", std::process::id()));
    let messages = root.with_extension("json");
    let prompt_file = root.with_extension("txt");
    std::fs::write(
        &messages,
        r#"{
            "meta": {"preserve_thinking": true},
            "messages": [{"role": "user", "content": "hi"}]
        }"#,
    )
    .unwrap();
    std::fs::write(&prompt_file, "hi").unwrap();

    let messages_path = messages.to_str().unwrap();
    let prompt_path = prompt_file.to_str().unwrap();
    let cases = [
        (
            vec!["qwen", "--model", "model.gguf", "--messages", messages_path],
            true,
        ),
        (
            vec![
                "qwen",
                "--model",
                "model.gguf",
                "--messages",
                messages_path,
                "--messages-strip-thinking",
            ],
            false,
        ),
        (
            vec![
                "qwen",
                "--model",
                "model.gguf",
                "--messages",
                messages_path,
                "--messages-no-generation-prompt",
            ],
            false,
        ),
        (
            vec!["qwen", "--model", "model.gguf", "--prompt", "hi"],
            false,
        ),
        (
            vec![
                "qwen",
                "--model",
                "model.gguf",
                "--prompt-file",
                prompt_path,
            ],
            false,
        ),
    ];
    for (argv, expected) in cases {
        let args = Args::try_parse_from(argv).unwrap();
        let (_, _, eligible) = prompt_text(&args).unwrap();
        assert_eq!(eligible, expected);
    }

    std::fs::remove_file(messages).unwrap();
    std::fs::remove_file(prompt_file).unwrap();
}

#[test]
fn durable_mode_rejects_unsupported_surfaces_and_invalid_budgets() {
    let jsonl = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--requests-jsonl",
        "requests.jsonl",
        "--durable-prefix-cache",
        "cache",
    ])
    .unwrap();
    assert!(validate_durable_prefix_cache_mode(&jsonl).is_err());

    let invalid_budget = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--durable-prefix-cache",
        "cache",
        "--durable-prefix-cache-max-mib",
        "1024",
        "--durable-prefix-cache-max-entry-mib",
        "2048",
    ])
    .unwrap();
    assert!(validate_durable_prefix_cache_mode(&invalid_budget).is_err());

    let valid = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--durable-prefix-cache",
        "cache",
    ])
    .unwrap();
    validate_durable_prefix_cache_mode(&valid).unwrap();
}

#[test]
fn checkpoint_staged_integrity_parser_is_strict() {
    assert_eq!(parse_checkpoint_staged_integrity(None).unwrap(), None);
    assert_eq!(
        parse_checkpoint_staged_integrity(Some("decode")).unwrap(),
        Some(StagedIntegrityMode::Decode)
    );
    assert_eq!(
        parse_checkpoint_staged_integrity(Some("deferred-restore")).unwrap(),
        Some(StagedIntegrityMode::DeferredRestore)
    );
    for invalid in ["", "Decode", "deferred_restore", "deferred", "true"] {
        assert!(parse_checkpoint_staged_integrity(Some(invalid)).is_err());
    }
}

#[test]
fn messages_input_is_single_turn_and_durable_cache_eligible() {
    let args = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--messages",
        "game.json",
        "--messages-strip-thinking",
        "--durable-prefix-cache",
        "cache",
    ])
    .unwrap();
    validate_durable_prefix_cache_mode(&args).unwrap();
    assert!(!prompt_add_special_tokens(&args, PromptSource::Messages));

    assert!(
        Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--messages",
            "game.json",
        ])
        .is_err()
    );
    assert!(
        Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--messages",
            "game.json",
            "--messages-preserve-thinking",
            "--messages-strip-thinking",
        ])
        .is_err()
    );
}

#[test]
fn auto_prefill_profiles_require_complete_allowlisted_topology() {
    let a3b = test_a3b_arch();
    let a10b = test_a10b_arch();

    assert_eq!(
        auto_prefill_profile(a3b, Some("Qwen3.6 35B A3B"), Some(15)),
        Some(AutoPrefillProfile {
            name: "qwen3.6-35b-a3b-filetype15",
            outer_chunk: 2048,
            query_heads: 16,
            gdn_overlay_bytes: 235_405_312,
        })
    );
    assert_eq!(
        auto_prefill_profile(a10b, Some("Qwen3.5 122B A10B"), Some(15)),
        Some(AutoPrefillProfile {
            name: "qwen3.5-122b-a10b-filetype15",
            outer_chunk: 4096,
            query_heads: 32,
            gdn_overlay_bytes: 807_403_520,
        })
    );

    macro_rules! reject_field {
        ($arch:expr, $name:expr, $field:ident, $value:expr) => {{
            let mut invalid = $arch;
            invalid.$field = $value;
            assert_eq!(
                auto_prefill_profile(invalid, Some($name), Some(15)),
                None,
                "{}",
                stringify!($field)
            );
        }};
    }

    for (arch, name) in [(a3b, "Qwen3.6 35B A3B"), (a10b, "Qwen3.5 122B A10B")] {
        reject_field!(arch, name, kind, ArchKind::Dense);
        reject_field!(arch, name, expert_count, 255);
        reject_field!(arch, name, expert_used_count, 7);
        reject_field!(arch, name, full_attention_interval, 3);
        reject_field!(arch, name, attn_head_dim, 128);
        reject_field!(arch, name, partial_rotary_factor, 0.5);
        reject_field!(arch, name, gdn_n_k_heads, 8);
        reject_field!(arch, name, gdn_head_dim, 64);
        reject_field!(arch, name, gdn_conv_kernel, 3);
        reject_field!(arch, name, mtp_n_hidden_layers, 1);
        assert_eq!(auto_prefill_profile(arch, Some(name), Some(14)), None);
        assert_eq!(
            auto_prefill_profile(arch, Some("wrong model"), Some(15)),
            None
        );
        assert_eq!(auto_prefill_profile(arch, None, Some(15)), None);
    }

    reject_field!(a3b, "Qwen3.6 35B A3B", n_layer, 39);
    reject_field!(a3b, "Qwen3.6 35B A3B", hidden_size, 2049);
    reject_field!(a3b, "Qwen3.6 35B A3B", n_q_heads, 15);
    reject_field!(a3b, "Qwen3.6 35B A3B", n_kv_heads, 1);
    reject_field!(a3b, "Qwen3.6 35B A3B", gdn_n_v_heads, 31);
    reject_field!(a3b, "Qwen3.6 35B A3B", expert_feed_forward_length, 513);
    reject_field!(
        a3b,
        "Qwen3.6 35B A3B",
        expert_shared_feed_forward_length,
        513
    );
    reject_field!(a10b, "Qwen3.5 122B A10B", n_layer, 47);
    reject_field!(a10b, "Qwen3.5 122B A10B", hidden_size, 3071);
    reject_field!(a10b, "Qwen3.5 122B A10B", n_q_heads, 31);
    reject_field!(a10b, "Qwen3.5 122B A10B", n_kv_heads, 1);
    reject_field!(a10b, "Qwen3.5 122B A10B", gdn_n_v_heads, 63);
    reject_field!(a10b, "Qwen3.5 122B A10B", expert_feed_forward_length, 1023);
    reject_field!(
        a10b,
        "Qwen3.5 122B A10B",
        expert_shared_feed_forward_length,
        1023
    );
}

#[test]
fn auto_prefill_chunk_decision_bounds_the_measured_prompt_range() {
    let profile = Some(AutoPrefillProfile {
        name: "test-profile",
        outer_chunk: 2048,
        query_heads: 16,
        gdn_overlay_bytes: 235_405_312,
    });
    for (prompt_tokens, selected, reason) in [
        (8191, 1024, "prompt_below_validated_range"),
        (8192, 2048, "matched_validated_profile"),
        (16384, 2048, "matched_validated_profile"),
        (16385, 1024, "prompt_above_memory_bounded_range"),
    ] {
        let decision = auto_prefill_chunk_decision(profile, prompt_tokens, false, true);
        assert_eq!(decision.selected, selected);
        assert_eq!(decision.reason, reason);
        assert_eq!(decision.validated_prompt_range, Some([8192, 16384]));
        assert_eq!(decision.evidence_baseline_chunk, Some(1024));
    }

    let unsupported = auto_prefill_chunk_decision(None, 10_000, false, true);
    assert_eq!(unsupported.selected, 1024);
    assert_eq!(unsupported.validated_prompt_range, None);
    assert_eq!(unsupported.evidence_baseline_chunk, None);

    let environment = auto_prefill_chunk_decision(profile, 10_000, true, true);
    assert_eq!(environment.reason, "prefill_environment_override_present");
    assert_eq!(environment.selected, 1024);

    let cache = auto_prefill_chunk_decision(profile, 10_000, false, false);
    assert_eq!(cache.reason, "prefix_cache_interaction_unvalidated");
    assert_eq!(cache.selected, 1024);
}

#[test]
fn auto_prefill_overlay_equations_cover_range_boundaries() {
    let a3b = AutoPrefillProfile {
        name: "a3b",
        outer_chunk: 2048,
        query_heads: 16,
        gdn_overlay_bytes: 235_405_312,
    };
    let a10b = AutoPrefillProfile {
        name: "a10b",
        outer_chunk: 4096,
        query_heads: 32,
        gdn_overlay_bytes: 807_403_520,
    };
    assert_eq!(
        expected_auto_prefill_overlay(a3b, 8192).unwrap(),
        PrefillScratchOverlayStats {
            backing_bytes: 285_212_672,
            attention_bytes: 285_212_672,
            gdn_bytes: 235_405_312,
            saved_bytes: 235_405_312,
        }
    );
    assert_eq!(
        expected_auto_prefill_overlay(a3b, 16_384).unwrap(),
        PrefillScratchOverlayStats {
            backing_bytes: 570_425_344,
            attention_bytes: 570_425_344,
            gdn_bytes: 235_405_312,
            saved_bytes: 235_405_312,
        }
    );
    assert_eq!(
        expected_auto_prefill_overlay(a3b, 11_287).unwrap(),
        PrefillScratchOverlayStats {
            backing_bytes: 393_052_160,
            attention_bytes: 393_052_160,
            gdn_bytes: 235_405_312,
            saved_bytes: 235_405_312,
        }
    );
    assert_eq!(
        expected_auto_prefill_overlay(a10b, 11_287).unwrap(),
        PrefillScratchOverlayStats {
            backing_bytes: 807_403_520,
            attention_bytes: 786_104_320,
            gdn_bytes: 807_403_520,
            saved_bytes: 786_104_320,
        }
    );
}

#[test]
fn auto_prefill_plan_topology_rejects_every_geometry_drift() {
    let profile = test_auto_profile();
    let prompt_tokens = 10_000;
    let overlay = expected_auto_prefill_overlay(profile, prompt_tokens).unwrap();
    assert_eq!(
        validate_auto_prefill_plan_topology(
            profile,
            prompt_tokens,
            2048,
            10_000,
            1024,
            Some(overlay),
        )
        .unwrap(),
        overlay
    );

    for (block_size, matrix_max_pos, query_rows) in [
        (2047, 10_000, 1024),
        (2048, 9_999, 1024),
        (2048, 10_000, 512),
    ] {
        let error = validate_auto_prefill_plan_topology(
            profile,
            prompt_tokens,
            block_size,
            matrix_max_pos,
            query_rows,
            Some(overlay),
        )
        .unwrap_err();
        assert!(error.to_string().contains("plan geometry drifted"));
    }

    let missing =
        validate_auto_prefill_plan_topology(profile, prompt_tokens, 2048, 10_000, 1024, None)
            .unwrap_err();
    assert!(missing.to_string().contains("has no scratch overlay"));

    let mut wrong_overlay = overlay;
    wrong_overlay.saved_bytes -= 1;
    let mismatch = validate_auto_prefill_plan_topology(
        profile,
        prompt_tokens,
        2048,
        10_000,
        1024,
        Some(wrong_overlay),
    )
    .unwrap_err();
    assert!(mismatch.to_string().contains("overlay geometry drifted"));
}

#[test]
fn auto_prefill_allocation_pricing_reconciles_and_rejects_invalid_prices() {
    let allocations = [
        ("zero", false, 0),
        ("eager", false, 10),
        ("deferred", true, 20),
    ];
    let (rows, eager, deferred, priced) = price_prefill_allocations(allocations, |logical| {
        Ok(MetalBufferSizeAndAlign {
            size: match logical {
                0 => 1,
                10 => 16,
                20 => 32,
                _ => unreachable!(),
            },
            alignment: 256,
        })
    })
    .unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!((eager, deferred, priced), (10, 20, 49));
    assert_eq!(rows[2].name, "deferred");
    assert!(rows[2].deferred);

    for priced in [
        MetalBufferSizeAndAlign {
            size: 0,
            alignment: 256,
        },
        MetalBufferSizeAndAlign {
            size: 9,
            alignment: 256,
        },
        MetalBufferSizeAndAlign {
            size: 10,
            alignment: 0,
        },
        MetalBufferSizeAndAlign {
            size: 10,
            alignment: 3,
        },
    ] {
        let error = price_prefill_allocations([("bad", false, 10)], |_| Ok(priced)).unwrap_err();
        assert!(error.to_string().contains("allocation pricing is invalid"));
    }

    let priced_overflow =
        price_prefill_allocations([("first", false, 1), ("second", true, 2)], |logical| {
            Ok(MetalBufferSizeAndAlign {
                size: if logical == 1 { u64::MAX } else { 2 },
                alignment: 256,
            })
        })
        .unwrap_err();
    assert!(
        priced_overflow
            .to_string()
            .contains("priced prefill byte overflow")
    );

    let logical_overflow = price_prefill_allocations(
        [("first", false, u64::MAX), ("second", false, 1)],
        |logical| {
            Ok(MetalBufferSizeAndAlign {
                size: logical,
                alignment: 256,
            })
        },
    )
    .unwrap_err();
    assert!(
        logical_overflow
            .to_string()
            .contains("eager prefill logical byte overflow")
    );
}

#[test]
fn auto_prefill_environment_gate_is_prefix_scoped() {
    assert!(is_prefill_environment_key(std::ffi::OsStr::new(
        "QWEN_PREFILL_ATTN_MATRIX_ONLINE"
    )));
    assert!(!is_prefill_environment_key(std::ffi::OsStr::new(
        "QWEN_GGUF_NO_COPY"
    )));

    let keys = vec![
        std::ffi::OsString::from("PATH"),
        std::ffi::OsString::from("QWEN_PREFILL_ATTN_MATRIX_ONLINE"),
        std::ffi::OsString::from("QWEN_GGUF_NO_COPY"),
    ];
    let unchanged = keys.clone();
    assert!(prefill_environment_override_present_in(&keys));
    assert_eq!(keys, unchanged);
    assert!(!prefill_environment_override_present_in([
        std::ffi::OsString::from("PATH"),
        std::ffi::OsString::from("QWEN_GGUF_NO_COPY"),
    ]));
}

#[test]
fn auto_prefill_reserve_requires_positive_checked_sequence_growth() {
    assert_eq!(
        auto_prefill_reserve(100, 101),
        Ok((1, AUTO_CHUNK_TRANSIENT_RESERVE_BYTES + 1))
    );
    assert_eq!(
        auto_prefill_reserve(100, 100),
        Err("sequence_allocation_signal_invalid")
    );
    assert_eq!(
        auto_prefill_reserve(101, 100),
        Err("sequence_allocation_signal_invalid")
    );
    assert_eq!(
        auto_prefill_reserve(0, u64::MAX),
        Err("candidate_reserve_overflow")
    );
}

#[test]
fn ineligible_auto_prefill_preserves_legacy_constructor_order_and_reasons() {
    let profile = Some(test_auto_profile());
    for (candidate, prompt, environment, cache_safe, reason) in [
        (None, 10_000, false, true, "profile_not_allowlisted"),
        (
            profile,
            AUTO_CHUNK_PROMPT_MIN - 1,
            false,
            true,
            "prompt_below_validated_range",
        ),
        (
            profile,
            AUTO_CHUNK_PROMPT_MAX + 1,
            false,
            true,
            "prompt_above_memory_bounded_range",
        ),
        (
            profile,
            10_000,
            true,
            true,
            "prefill_environment_override_present",
        ),
        (
            profile,
            10_000,
            false,
            false,
            "prefix_cache_interaction_unvalidated",
        ),
    ] {
        let mut allocator = fake_allocator(&[10, 20], Ok(fake_plan_decision(1_000)), 0, false);
        let state = allocate_prefill_request_state_with(
            &mut allocator,
            PrefillChunkArg::Auto,
            prompt,
            10_017,
            cache_safe,
            candidate,
            environment,
        )
        .unwrap();
        let decision = state.decision.as_ref().unwrap();

        assert_eq!(state.chunk, 1024);
        assert_eq!(state.scratch, "legacy");
        assert_eq!(decision.reason, reason);
        assert_eq!(decision.classification, "baseline");
        assert!(decision.plan.is_none());
        assert!(decision.admission.is_none());
        assert_eq!(
            allocator.events,
            [
                AllocationEvent::LegacyScratch(1024),
                AllocationEvent::CurrentAllocated,
                AllocationEvent::Sequence(10_017),
                AllocationEvent::CurrentAllocated,
            ],
            "{reason}"
        );
    }
}

#[test]
fn candidate_plan_unavailable_has_a_distinct_fail_closed_reason() {
    let mut allocator = fake_allocator(
        &[100, 200, 300],
        Err(CandidatePlanFailure::Unavailable(
            "planner failed".to_string(),
        )),
        10_000_000_000,
        false,
    );
    let state = allocate_prefill_request_state_with(
        &mut allocator,
        PrefillChunkArg::Auto,
        10_000,
        10_017,
        true,
        Some(test_auto_profile()),
        false,
    )
    .unwrap();
    let decision = state.decision.as_ref().unwrap();

    assert_eq!(decision.reason, "candidate_plan_unavailable");
    assert_eq!(decision.detail.as_deref(), Some("planner failed"));
    assert!(decision.plan.is_none());
    assert!(decision.admission.is_none());
    assert!(
        !allocator
            .events
            .contains(&AllocationEvent::CandidateScratch)
    );
}

#[test]
fn sequence_accounting_failures_fall_back_before_candidate_planning() {
    for (samples, reason) in [
        ([100, 100, 300], "sequence_allocation_signal_invalid"),
        ([101, 100, 300], "sequence_allocation_signal_invalid"),
        ([0, u64::MAX, 300], "candidate_reserve_overflow"),
    ] {
        let mut allocator =
            fake_allocator(&samples, Ok(fake_plan_decision(1_000)), u64::MAX, false);
        let state = allocate_prefill_request_state_with(
            &mut allocator,
            PrefillChunkArg::Auto,
            10_000,
            10_017,
            true,
            Some(test_auto_profile()),
            false,
        )
        .unwrap();
        let decision = state.decision.as_ref().unwrap();

        assert_eq!(decision.reason, reason);
        assert_eq!(state.scratch, "legacy");
        assert!(!allocator.events.contains(&AllocationEvent::CandidatePlan));
        assert!(
            !allocator
                .events
                .contains(&AllocationEvent::CandidateScratch)
        );
    }
}

#[test]
fn auto_prefill_memory_boundaries_reach_the_cli_telemetry() {
    let reserve = AUTO_CHUNK_TRANSIENT_RESERVE_BYTES + 100;
    let required = reserve + 1_000;
    for (signals, expected_reason, admitted) in [
        (
            MetalMemorySignals {
                recommended_max_bytes: 200 + required,
                current_allocated_bytes: 200,
                process_limit_remaining_bytes: Some(required),
            },
            "admitted_with_process_budget",
            true,
        ),
        (
            MetalMemorySignals {
                recommended_max_bytes: 200 + required,
                current_allocated_bytes: 200,
                process_limit_remaining_bytes: Some(required - 1),
            },
            "process_insufficient",
            false,
        ),
        (
            MetalMemorySignals {
                recommended_max_bytes: 200 + required,
                current_allocated_bytes: 200,
                process_limit_remaining_bytes: None,
            },
            "process_signal_unavailable",
            false,
        ),
        (
            MetalMemorySignals {
                recommended_max_bytes: 0,
                current_allocated_bytes: 200,
                process_limit_remaining_bytes: Some(required),
            },
            "invalid_working_set_signal",
            false,
        ),
        (
            MetalMemorySignals {
                recommended_max_bytes: 200 + required - 1,
                current_allocated_bytes: 200,
                process_limit_remaining_bytes: Some(0),
            },
            "working_set_insufficient",
            false,
        ),
        (
            MetalMemorySignals {
                recommended_max_bytes: 200 + required,
                current_allocated_bytes: 200,
                process_limit_remaining_bytes: Some(0),
            },
            "admitted_process_budget_omitted",
            true,
        ),
    ] {
        let mut allocator = fake_allocator_with_signals(
            &[100, 200, 300],
            Ok(fake_plan_decision(1_000)),
            signals,
            false,
        );
        let state = allocate_prefill_request_state_with(
            &mut allocator,
            PrefillChunkArg::Auto,
            10_000,
            10_017,
            true,
            Some(test_auto_profile()),
            false,
        )
        .unwrap();
        let decision = state.decision.as_ref().unwrap();
        let admission = decision.admission.as_ref().unwrap();

        assert_eq!(admission.required_bytes, Some(required));
        assert_eq!(admission.evaluator_reason, expected_reason);
        assert_eq!(admission.admitted, admitted);
        assert_eq!(state.scratch, if admitted { "candidate" } else { "legacy" });
        assert_eq!(
            allocator
                .events
                .contains(&AllocationEvent::CandidateScratch),
            admitted
        );
    }
}

#[test]
fn numeric_prefill_preserves_scratch_before_sequence_order() {
    let mut allocator = fake_allocator(&[10, 20], Ok(fake_plan_decision(1_000)), 0, false);
    let state = allocate_prefill_request_state_with(
        &mut allocator,
        PrefillChunkArg::Fixed(2048),
        10_000,
        10_017,
        true,
        Some(test_auto_profile()),
        false,
    )
    .unwrap();

    assert_eq!(state.chunk, 2048);
    assert_eq!(state.scratch, "legacy");
    assert!(state.decision.is_none());
    assert_eq!(
        allocator.events,
        [
            AllocationEvent::LegacyScratch(2048),
            AllocationEvent::CurrentAllocated,
            AllocationEvent::Sequence(10_017),
            AllocationEvent::CurrentAllocated,
        ]
    );
}

#[test]
fn admitted_auto_prefill_allocates_only_the_candidate_scratch() {
    let mut allocator = fake_allocator(
        &[100, 200, 300],
        Ok(fake_plan_decision(1_000)),
        10_000_000_000,
        false,
    );
    let state = allocate_prefill_request_state_with(
        &mut allocator,
        PrefillChunkArg::Auto,
        10_000,
        10_017,
        true,
        Some(test_auto_profile()),
        false,
    )
    .unwrap();
    let decision = state.decision.as_ref().unwrap();

    assert_eq!(state.chunk, 2048);
    assert_eq!(state.scratch, "candidate");
    assert_eq!(decision.reason, "admitted");
    assert!(decision.plan.is_some());
    assert!(decision.admission.as_ref().unwrap().admitted);
    assert_eq!(
        allocator.events,
        [
            AllocationEvent::CurrentAllocated,
            AllocationEvent::Sequence(10_017),
            AllocationEvent::CurrentAllocated,
            AllocationEvent::CandidatePlan,
            AllocationEvent::MemorySignals,
            AllocationEvent::CandidateScratch,
            AllocationEvent::CurrentAllocated,
        ]
    );
}

#[test]
fn denied_auto_prefill_reuses_sequence_and_allocates_legacy_scratch() {
    let mut allocator = fake_allocator(&[100, 200, 300], Ok(fake_plan_decision(1_000)), 200, false);
    let state = allocate_prefill_request_state_with(
        &mut allocator,
        PrefillChunkArg::Auto,
        10_000,
        10_017,
        true,
        Some(test_auto_profile()),
        false,
    )
    .unwrap();
    let decision = state.decision.as_ref().unwrap();

    assert_eq!(state.chunk, 1024);
    assert_eq!(state.scratch, "legacy");
    assert_eq!(decision.reason, "memory_admission_denied");
    assert!(decision.plan.is_some());
    assert!(!decision.admission.as_ref().unwrap().admitted);
    assert_eq!(
        allocator.events,
        [
            AllocationEvent::CurrentAllocated,
            AllocationEvent::Sequence(10_017),
            AllocationEvent::CurrentAllocated,
            AllocationEvent::CandidatePlan,
            AllocationEvent::MemorySignals,
            AllocationEvent::LegacyScratch(1024),
            AllocationEvent::CurrentAllocated,
        ]
    );
}

#[test]
fn unpriceable_auto_prefill_plan_falls_back_before_candidate_allocation() {
    let mut allocator = fake_allocator(
        &[100, 200, 300],
        Err(CandidatePlanFailure::InvalidOrUnpriceable(
            "bad price".to_string(),
        )),
        10_000_000_000,
        false,
    );
    let state = allocate_prefill_request_state_with(
        &mut allocator,
        PrefillChunkArg::Auto,
        10_000,
        10_017,
        true,
        Some(test_auto_profile()),
        false,
    )
    .unwrap();
    let decision = state.decision.as_ref().unwrap();

    assert_eq!(state.scratch, "legacy");
    assert_eq!(decision.reason, "candidate_plan_invalid_or_unpriceable");
    assert_eq!(decision.detail.as_deref(), Some("bad price"));
    assert_eq!(
        allocator.events,
        [
            AllocationEvent::CurrentAllocated,
            AllocationEvent::Sequence(10_017),
            AllocationEvent::CurrentAllocated,
            AllocationEvent::CandidatePlan,
            AllocationEvent::LegacyScratch(1024),
            AllocationEvent::CurrentAllocated,
        ]
    );
}

#[test]
fn required_byte_overflow_preserves_plan_telemetry() {
    let mut allocator = fake_allocator(
        &[100, 200, 300],
        Ok(fake_plan_decision(u64::MAX)),
        u64::MAX,
        false,
    );
    let state = allocate_prefill_request_state_with(
        &mut allocator,
        PrefillChunkArg::Auto,
        10_000,
        10_017,
        true,
        Some(test_auto_profile()),
        false,
    )
    .unwrap();
    let decision = state.decision.as_ref().unwrap();
    let json = serde_json::to_value(decision).unwrap();

    assert_eq!(decision.reason, "candidate_required_bytes_overflow");
    assert!(decision.plan.is_some());
    let admission = decision.admission.as_ref().unwrap();
    assert_eq!(admission.required_bytes, None);
    assert_eq!(admission.evaluator_reason, "required_bytes_overflow");
    assert!(!admission.admitted);
    assert!(json.get("plan").is_some());
    assert!(json.get("admission").is_some());
    assert_eq!(
        allocator.events,
        [
            AllocationEvent::CurrentAllocated,
            AllocationEvent::Sequence(10_017),
            AllocationEvent::CurrentAllocated,
            AllocationEvent::CandidatePlan,
            AllocationEvent::MemorySignals,
            AllocationEvent::LegacyScratch(1024),
            AllocationEvent::CurrentAllocated,
        ]
    );
}

#[test]
fn admitted_candidate_allocation_failure_is_not_retried_as_legacy() {
    let mut allocator = fake_allocator(
        &[100, 200],
        Ok(fake_plan_decision(1_000)),
        10_000_000_000,
        true,
    );
    let result = allocate_prefill_request_state_with(
        &mut allocator,
        PrefillChunkArg::Auto,
        10_000,
        10_017,
        true,
        Some(test_auto_profile()),
        false,
    );

    let error = match result {
        Ok(_) => panic!("candidate allocation unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("candidate allocation failed"));
    assert_eq!(
        allocator.events,
        [
            AllocationEvent::CurrentAllocated,
            AllocationEvent::Sequence(10_017),
            AllocationEvent::CurrentAllocated,
            AllocationEvent::CandidatePlan,
            AllocationEvent::MemorySignals,
            AllocationEvent::CandidateScratch,
        ]
    );
}

#[test]
fn auto_cache_prefix_discovery_scores_reuse() {
    let mut requests = vec![
        prepared("a", &[1, 2, 3, 4, 10]),
        prepared("b", &[1, 2, 3, 4, 20]),
        prepared("c", &[1, 2, 3, 30]),
        prepared("d", &[9, 9, 9]),
    ];

    discover_auto_cache_prefixes(&mut requests, 3);

    assert_eq!(requests[0].auto_cache_prefix_tokens, Some(3));
    assert_eq!(requests[0].auto_cache_future_hits, 2);
    assert_eq!(requests[1].auto_cache_prefix_tokens, Some(3));
    assert_eq!(requests[1].auto_cache_future_hits, 1);
    assert_eq!(requests[2].auto_cache_prefix_tokens, None);
    assert_eq!(requests[3].auto_cache_prefix_tokens, None);
}

#[test]
fn greedy_generation_delivers_before_transition_and_skips_terminal_step() {
    let events = RefCell::new(Vec::new());
    let generation = generate_greedy(
        logits_with_argmax(1),
        3,
        &[],
        |token| {
            events.borrow_mut().push(format!("token:{token}"));
            Ok(())
        },
        |token| {
            events.borrow_mut().push(format!("transition:{token}"));
            Ok(logits_with_argmax(match token {
                1 => 2,
                2 => 0,
                _ => panic!("unexpected transition token {token}"),
            }))
        },
    )
    .unwrap();

    assert_eq!(generation.tokens, [1, 2, 0]);
    assert_eq!(generation.transitions, 2);
    assert_eq!(generation.stop_reason, StopReason::TokenLimit);
    assert_eq!(
        events.into_inner(),
        [
            "token:1",
            "transition:1",
            "token:2",
            "transition:2",
            "token:0",
        ]
    );
}

#[test]
fn gpu_greedy_generation_shares_terminal_and_callback_ordering() {
    let events = RefCell::new(Vec::new());
    let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
    let generation = generate_gpu_greedy(
        logits_with_argmax(1),
        3,
        &[],
        &mut sampler,
        |token| {
            events.borrow_mut().push(format!("token:{token}"));
            Ok(())
        },
        |token| {
            events.borrow_mut().push(format!("transition:{token}"));
            Ok(GreedySelection::Token(match token {
                1 => 2,
                2 => 0,
                _ => panic!("unexpected transition token {token}"),
            }))
        },
    )
    .unwrap();

    assert_eq!(generation.tokens, [1, 2, 0]);
    assert_eq!(generation.transitions, 2);
    assert_eq!(generation.stop_reason, StopReason::TokenLimit);
    assert_eq!(sampler.draws(), 0);
    assert_eq!(
        events.into_inner(),
        [
            "token:1",
            "transition:1",
            "token:2",
            "transition:2",
            "token:0",
        ]
    );
}

#[test]
fn gpu_greedy_nan_is_reported_after_the_committed_transition() {
    let events = RefCell::new(Vec::new());
    let position = Cell::new(10usize);
    let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
    let error = generate_gpu_greedy(
        logits_with_argmax(1),
        3,
        &[],
        &mut sampler,
        |token| {
            events.borrow_mut().push(format!("token:{token}"));
            Ok(())
        },
        |token| {
            events.borrow_mut().push(format!("transition:{token}"));
            position.set(position.get() + 1);
            Ok(GreedySelection::NanLogit { token: 7 })
        },
    )
    .unwrap_err();

    assert!(error.to_string().contains("logit at token 7 is NaN"));
    assert_eq!(events.into_inner(), ["token:1", "transition:1"]);
    assert_eq!(position.get(), 11, "the completed forward advances state");
    assert_eq!(sampler.draws(), 0);
}

#[test]
fn gpu_greedy_preserves_terminal_and_failure_boundaries() {
    for (max_tokens, stop_tokens, expected_reason) in [
        (1, Vec::new(), StopReason::TokenLimit),
        (3, vec![1], StopReason::Eos),
    ] {
        let callbacks = Cell::new(0usize);
        let transitions = Cell::new(0usize);
        let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
        let generation = generate_gpu_greedy(
            logits_with_argmax(1),
            max_tokens,
            &stop_tokens,
            &mut sampler,
            |_| {
                callbacks.set(callbacks.get() + 1);
                Ok(())
            },
            |_| {
                transitions.set(transitions.get() + 1);
                Ok(GreedySelection::Token(2))
            },
        )
        .unwrap();
        assert_eq!(generation.tokens, [1]);
        assert_eq!(generation.stop_reason, expected_reason);
        assert_eq!(generation.transitions, 0);
        assert_eq!(transitions.get(), 0);
        assert_eq!(
            callbacks.get(),
            usize::from(expected_reason == StopReason::TokenLimit)
        );
    }

    let transitions = Cell::new(0usize);
    let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
    let callback_error = generate_gpu_greedy(
        logits_with_argmax(1),
        2,
        &[],
        &mut sampler,
        |_| Err(anyhow!("callback failed")),
        |_| {
            transitions.set(transitions.get() + 1);
            Ok(GreedySelection::Token(2))
        },
    )
    .unwrap_err();
    assert!(callback_error.to_string().contains("callback failed"));
    assert_eq!(transitions.get(), 0);

    let callbacks = Cell::new(0usize);
    let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
    let transition_error = generate_gpu_greedy(
        logits_with_argmax(1),
        2,
        &[],
        &mut sampler,
        |_| {
            callbacks.set(callbacks.get() + 1);
            Ok(())
        },
        |_| Err(anyhow!("transition failed")),
    )
    .unwrap_err();
    assert!(transition_error.to_string().contains("transition failed"));
    assert_eq!(callbacks.get(), 1);
}

#[test]
fn decode_policy_labels_name_selection_residency() {
    let greedy = SamplingConfig::default();
    assert_eq!(decode_policy_label(greedy, false, false), "greedy_argmax");
    assert_eq!(
        decode_policy_label(greedy, false, true),
        "greedy_gpu_argmax"
    );
    assert_eq!(
        decode_policy_label(greedy, true, true),
        "prompt_lookup_l8_d7_target_n8"
    );
    assert_eq!(
        decode_policy_label(SamplingConfig::qwen_chat(7), false, false),
        "sampled_cpu"
    );
    assert_eq!(
        jsonl_decode_policy_label(greedy, false, false),
        "greedy_argmax"
    );
    assert_eq!(
        jsonl_decode_policy_label(greedy, false, true),
        "greedy_gpu_argmax"
    );
}

#[test]
fn gpu_greedy_policy_is_default_off_forceable_and_rollbackable() {
    let modes = [
        GreedyGpuArgmaxMode::DefaultOff,
        GreedyGpuArgmaxMode::ForceEnabled,
        GreedyGpuArgmaxMode::ExplicitRollback,
    ];
    let requests = [
        (SamplingConfig::default(), false),
        (SamplingConfig::qwen_chat(7), false),
        (SamplingConfig::default(), true),
    ];
    for mode in modes {
        for (sampling, prompt_lookup) in requests {
            let request_eligible = sampling.temperature == 0.0 && !prompt_lookup;
            let expected = if mode == GreedyGpuArgmaxMode::ExplicitRollback {
                GreedyGpuDecision {
                    enabled: false,
                    reason: "disabled_by_explicit_rollback",
                }
            } else if !request_eligible {
                GreedyGpuDecision {
                    enabled: false,
                    reason: "ineligible_request",
                }
            } else if mode == GreedyGpuArgmaxMode::ForceEnabled {
                GreedyGpuDecision {
                    enabled: true,
                    reason: "force_enabled",
                }
            } else {
                GreedyGpuDecision {
                    enabled: false,
                    reason: "default_off",
                }
            };
            assert_eq!(
                resolve_greedy_gpu_decision(mode, sampling, prompt_lookup),
                expected,
                "mode={mode:?} sampled={} prompt_lookup={prompt_lookup}",
                sampling.temperature > 0.0,
            );
        }
    }

    for value in ["1", "true", "TRUE", "yes", "YES"] {
        assert_eq!(
            parse_greedy_gpu_argmax_mode(Some(OsStr::new(value))),
            GreedyGpuArgmaxMode::ForceEnabled
        );
    }
    for value in ["0", "false", "FALSE", "no", "NO"] {
        assert_eq!(
            parse_greedy_gpu_argmax_mode(Some(OsStr::new(value))),
            GreedyGpuArgmaxMode::ExplicitRollback
        );
    }
    assert_eq!(
        parse_greedy_gpu_argmax_mode(None),
        GreedyGpuArgmaxMode::DefaultOff
    );
    for value in ["", "invalid", "True", "2"] {
        assert_eq!(
            parse_greedy_gpu_argmax_mode(Some(OsStr::new(value))),
            GreedyGpuArgmaxMode::ExplicitRollback
        );
    }
    use std::os::unix::ffi::OsStringExt;
    let non_unicode = std::ffi::OsString::from_vec(vec![0xff]);
    assert_eq!(
        parse_greedy_gpu_argmax_mode(Some(&non_unicode)),
        GreedyGpuArgmaxMode::ExplicitRollback
    );
}

#[test]
fn deepseek_v4_prefetch_defaults_auto_and_parses_strictly() {
    assert_eq!(
        parse_deepseek_v4_prefetch_mode(None).unwrap(),
        DeepSeekV4PrefetchMode::Auto
    );
    assert_eq!(
        parse_deepseek_v4_prefetch_mode(Some(OsStr::new("off"))).unwrap(),
        DeepSeekV4PrefetchMode::Off
    );
    assert_eq!(
        parse_deepseek_v4_prefetch_mode(Some(OsStr::new("always"))).unwrap(),
        DeepSeekV4PrefetchMode::Always
    );
    assert_eq!(
        parse_deepseek_v4_prefetch_mode(Some(OsStr::new("auto"))).unwrap(),
        DeepSeekV4PrefetchMode::Auto
    );
    assert!(parse_deepseek_v4_prefetch_mode(Some(OsStr::new("cold"))).is_err());

    assert_eq!(
        deepseek_v4_prefetch_policy(DeepSeekV4PrefetchMode::Off),
        PrefetchPolicy::Off
    );
    assert_eq!(
        deepseek_v4_prefetch_policy(DeepSeekV4PrefetchMode::Always),
        PrefetchPolicy::Always
    );
    match deepseek_v4_prefetch_policy(DeepSeekV4PrefetchMode::Auto) {
        PrefetchPolicy::ColdOnly { threshold } => {
            assert_eq!(threshold.value(), DEEPSEEK_V4_PREFETCH_AUTO_THRESHOLD);
        }
        other => panic!("expected DS4 cold-only auto policy, got {other:?}"),
    }

    use std::os::unix::ffi::OsStringExt;
    let non_unicode = std::ffi::OsString::from_vec(vec![0xff]);
    assert!(parse_deepseek_v4_prefetch_mode(Some(&non_unicode)).is_err());
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn deepseek_v4_temporal_window_is_bounded() {
    assert_eq!(parse_deepseek_v4_temporal_window(None).unwrap(), 0);
    assert_eq!(
        parse_deepseek_v4_temporal_window(Some(OsStr::new("65"))).unwrap(),
        65
    );
    assert!(parse_deepseek_v4_temporal_window(Some(OsStr::new("66"))).is_err());
    assert!(parse_deepseek_v4_temporal_window(Some(OsStr::new("no"))).is_err());
}

#[test]
fn generated_token_sha256_has_a_canonical_integer_encoding() {
    assert_eq!(
        generated_token_sha256(&[1, -2, 248_319]),
        "2d6affd554663e51e2db10dd3cf00fe650f606317fbbf953353c86680fe48f0e"
    );
}

#[test]
fn serial_generation_uses_the_seeded_request_sampler() {
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 0x1234_5678_9abc_def0,
    };
    let mut sampler = Sampler::new(config).unwrap();
    let logits = vec![2.0, 1.5, 1.0, 0.5];
    let generation = generate_serial(
        logits.clone(),
        4,
        &[],
        &mut sampler,
        |_| Ok(()),
        |_| Ok(logits.clone()),
    )
    .unwrap();

    assert_eq!(generation.tokens, [0, 1, 1, 1]);
    assert_eq!(generation.transitions, 3);
    assert_eq!(sampler.draws(), 4);
    let telemetry = SamplingTelemetry::sampled(config, sampler.draws()).unwrap();
    assert_eq!(telemetry.algorithm_version, SAMPLER_ALGORITHM_VERSION);
    assert_eq!(telemetry.effective_seed, config.seed);
    assert_eq!(telemetry.draws, 4);
}

#[test]
fn sampled_structural_generation_preserves_transaction_boundaries() {
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        seed: 42,
    };

    let mut ordinary_one_sampler = Sampler::new(config).unwrap();
    let ordinary_one = generate_serial(
        logits_with_argmax(1),
        1,
        &[],
        &mut ordinary_one_sampler,
        |_| Ok(()),
        |_| -> Result<Vec<f32>> { panic!("one-token output must not transition") },
    )
    .unwrap();
    let mut one_sampler = Sampler::new(config).unwrap();
    let (one, one_telemetry) = generate_sampled_structural(
        logits_with_argmax(1),
        1,
        &[],
        &mut one_sampler,
        |_| Ok(()),
        |_, _, _| -> Result<_> { panic!("one-token output must not transition") },
    )
    .unwrap();
    assert_eq!(one.tokens, [1]);
    assert_eq!(one.tokens, ordinary_one.tokens);
    assert_eq!(one.stop_reason, StopReason::TokenLimit);
    assert_eq!(one.stop_reason, ordinary_one.stop_reason);
    assert_eq!(one.transitions, 0);
    assert_eq!(one.transitions, ordinary_one.transitions);
    assert_eq!(one_sampler.draws(), 1);
    assert_eq!(one_sampler.draws(), ordinary_one_sampler.draws());
    assert_eq!(one_telemetry.prompt_owned_bounded_calls, 1);
    assert_eq!(one_telemetry.borrowed_transition_calls, 0);

    let ordinary_callbacks = RefCell::new(Vec::new());
    let mut ordinary_sampler = Sampler::new(config).unwrap();
    let ordinary = generate_serial(
        logits_with_argmax(1),
        3,
        &[],
        &mut ordinary_sampler,
        |token| {
            ordinary_callbacks.borrow_mut().push(token);
            Ok(())
        },
        |token| {
            Ok(logits_with_argmax(match token {
                1 => 2,
                2 => 0,
                _ => panic!("unexpected ordinary token {token}"),
            }))
        },
    )
    .unwrap();
    let structural_callbacks = RefCell::new(Vec::new());
    let structural_advances = Cell::new(0usize);
    let mut structural_sampler = Sampler::new(config).unwrap();
    let (structural, structural_telemetry) = generate_sampled_structural(
        logits_with_argmax(1),
        3,
        &[],
        &mut structural_sampler,
        |token| {
            structural_callbacks.borrow_mut().push(token);
            Ok(())
        },
        |token, trial, telemetry| {
            let logits = logits_with_argmax(match token {
                1 => 2,
                2 => 0,
                _ => panic!("unexpected structural token {token}"),
            });
            fake_structural_transition(trial, telemetry, &logits, || {
                structural_advances.set(structural_advances.get() + 1);
                Ok(())
            })
        },
    )
    .unwrap();
    assert_eq!(structural.tokens, ordinary.tokens);
    assert_eq!(structural.stop_reason, ordinary.stop_reason);
    assert_eq!(structural.transitions, ordinary.transitions);
    assert_eq!(structural_sampler.draws(), ordinary_sampler.draws());
    assert_eq!(structural_advances.get(), ordinary.transitions);
    assert_eq!(
        structural_callbacks.into_inner(),
        ordinary_callbacks.into_inner()
    );
    assert_eq!(structural_telemetry.prompt_owned_bounded_calls, 1);
    assert_eq!(structural_telemetry.borrowed_transition_calls, 2);
    assert_eq!(structural_telemetry.resident_head_wait_calls, 2);
    assert_eq!(structural_telemetry.validated_shared_row_calls, 2);
    assert_eq!(structural_telemetry.input_logits_total, 12);
    assert_eq!(structural_telemetry.retained_top_k_total, 3);
    assert_eq!(structural_telemetry.max_heap_len, 1);
    assert!(structural_telemetry.max_heap_capacity >= 1);
    assert!(structural_telemetry.max_heap_capacity < 4);
    let ordinary_boundary = derive_completed_checkpoint_boundary(
        10,
        &ordinary.tokens,
        ordinary.transitions,
        10 + ordinary.transitions,
    )
    .unwrap();
    let structural_boundary = derive_completed_checkpoint_boundary(
        10,
        &structural.tokens,
        structural.transitions,
        10 + structural.transitions,
    )
    .unwrap();
    assert_eq!(
        structural_boundary.consumed_prefix_len,
        ordinary_boundary.consumed_prefix_len
    );
    assert_eq!(
        structural_boundary.pending_token,
        ordinary_boundary.pending_token
    );

    for (stop_tokens, expected_tokens, expected_callbacks, expected_transitions) in [
        (vec![1], vec![1], vec![], 0),
        (vec![2], vec![1, 2], vec![1], 1),
    ] {
        let ordinary_callbacks = RefCell::new(Vec::new());
        let mut ordinary_sampler = Sampler::new(config).unwrap();
        let ordinary = generate_serial(
            logits_with_argmax(1),
            4,
            &stop_tokens,
            &mut ordinary_sampler,
            |token| {
                ordinary_callbacks.borrow_mut().push(token);
                Ok(())
            },
            |_| Ok(logits_with_argmax(2)),
        )
        .unwrap();
        let callbacks = RefCell::new(Vec::new());
        let mut sampler = Sampler::new(config).unwrap();
        let (generation, telemetry) = generate_sampled_structural(
            logits_with_argmax(1),
            4,
            &stop_tokens,
            &mut sampler,
            |token| {
                callbacks.borrow_mut().push(token);
                Ok(())
            },
            |_, trial, telemetry| {
                fake_structural_transition(trial, telemetry, &logits_with_argmax(2), || Ok(()))
            },
        )
        .unwrap();
        assert_eq!(generation.tokens, ordinary.tokens);
        assert_eq!(generation.tokens, expected_tokens);
        assert_eq!(generation.stop_reason, ordinary.stop_reason);
        assert_eq!(generation.stop_reason, StopReason::Eos);
        assert_eq!(generation.transitions, ordinary.transitions);
        assert_eq!(generation.transitions, expected_transitions);
        assert_eq!(sampler.draws(), ordinary_sampler.draws());
        assert_eq!(
            callbacks.borrow().as_slice(),
            ordinary_callbacks.borrow().as_slice()
        );
        assert_eq!(callbacks.into_inner(), expected_callbacks);
        assert_eq!(
            telemetry.borrowed_transition_calls,
            expected_transitions as u64
        );
    }

    let callback_transitions = Cell::new(0usize);
    let mut ordinary_callback_sampler = Sampler::new(config).unwrap();
    let ordinary_callback_error = generate_serial(
        logits_with_argmax(1),
        2,
        &[],
        &mut ordinary_callback_sampler,
        |_| bail!("callback failed"),
        |_| unreachable!("callback failure must prevent transition"),
    )
    .unwrap_err();
    let mut callback_sampler = Sampler::new(config).unwrap();
    let callback_error = generate_sampled_structural(
        logits_with_argmax(1),
        2,
        &[],
        &mut callback_sampler,
        |_| bail!("callback failed"),
        |_, _, _| {
            callback_transitions.set(callback_transitions.get() + 1);
            unreachable!("callback failure must prevent transition")
        },
    )
    .unwrap_err();
    assert_eq!(
        callback_error.to_string(),
        ordinary_callback_error.to_string()
    );
    assert!(callback_error.to_string().contains("callback failed"));
    assert_eq!(callback_transitions.get(), 0);
    assert_eq!(callback_sampler.draws(), 1);
    assert_eq!(callback_sampler.draws(), ordinary_callback_sampler.draws());

    let mut ordinary_transition_sampler = Sampler::new(config).unwrap();
    let ordinary_transition_error = generate_serial(
        logits_with_argmax(1),
        2,
        &[],
        &mut ordinary_transition_sampler,
        |_| Ok(()),
        |_| bail!("transition failed"),
    )
    .unwrap_err();
    let mut transition_sampler = Sampler::new(config).unwrap();
    let transition_error = generate_sampled_structural(
        logits_with_argmax(1),
        2,
        &[],
        &mut transition_sampler,
        |_| Ok(()),
        |_, _, _| bail!("transition failed"),
    )
    .unwrap_err();
    assert_eq!(
        transition_error.to_string(),
        ordinary_transition_error.to_string()
    );
    assert!(transition_error.to_string().contains("transition failed"));
    assert_eq!(transition_sampler.draws(), 1);
    assert_eq!(
        transition_sampler.draws(),
        ordinary_transition_sampler.draws()
    );

    let nan_logits = vec![0.0, f32::NAN, 2.0, 1.0];
    let ordinary_position = Cell::new(0usize);
    let mut ordinary_sampler = Sampler::new(config).unwrap();
    let ordinary_error = generate_serial(
        logits_with_argmax(1),
        2,
        &[],
        &mut ordinary_sampler,
        |_| Ok(()),
        |_| {
            ordinary_position.set(ordinary_position.get() + 1);
            Ok(nan_logits.clone())
        },
    )
    .unwrap_err();
    let structural_position = Cell::new(0usize);
    let mut structural_sampler = Sampler::new(config).unwrap();
    let structural_error = generate_sampled_structural(
        logits_with_argmax(1),
        2,
        &[],
        &mut structural_sampler,
        |_| Ok(()),
        |_, trial, telemetry| {
            fake_structural_transition(trial, telemetry, &nan_logits, || {
                structural_position.set(structural_position.get() + 1);
                Ok(())
            })
        },
    )
    .unwrap_err();
    assert_eq!(structural_error.to_string(), ordinary_error.to_string());
    assert_eq!(structural_position.get(), ordinary_position.get());
    assert_eq!(structural_position.get(), 1);
    assert_eq!(structural_sampler.draws(), ordinary_sampler.draws());
    assert_eq!(structural_sampler.draws(), 1);

    let advance_attempts = Cell::new(0usize);
    let mut advance_sampler = Sampler::new(config).unwrap();
    let advance_error = generate_sampled_structural(
        logits_with_argmax(1),
        2,
        &[],
        &mut advance_sampler,
        |_| Ok(()),
        |_, trial, telemetry| {
            fake_structural_transition(trial, telemetry, &logits_with_argmax(2), || {
                advance_attempts.set(advance_attempts.get() + 1);
                bail!("advance failed")
            })
        },
    )
    .unwrap_err();
    assert!(advance_error.to_string().contains("advance failed"));
    assert_eq!(advance_attempts.get(), 1);
    assert_eq!(
        advance_sampler.draws(),
        1,
        "failed advance must not commit the trial RNG draw"
    );

    let mut accounting_sampler = Sampler::new(config).unwrap();
    let mut context = SampledStructuralContext {
        sampler: &mut accounting_sampler,
        telemetry: SampledStructuralTelemetry {
            prompt_owned_bounded_calls: u64::MAX,
            ..SampledStructuralTelemetry::default()
        },
    };
    let accounting_error =
        with_transactional_sampled_structural_context(&mut context, |trial, telemetry| {
            let (_, evidence) = trial.sample_bounded_top_k(&logits_with_argmax(1))?;
            telemetry.record_prompt(evidence)
        })
        .unwrap_err();
    assert!(accounting_error.to_string().contains("count overflow"));
    assert_eq!(context.sampler.draws(), 0);
    assert_eq!(context.telemetry.prompt_owned_bounded_calls, u64::MAX);
}

fn fake_attributed_transition(logits: Vec<f32>) -> (Vec<f32>, TokenProfile, LogitsReadbackProfile) {
    let bytes = logits.len() * std::mem::size_of::<f32>();
    (
        logits,
        TokenProfile {
            cpu_encode_ms: 0.1,
            cpu_to_gpu_complete_ms: 0.7,
            gpu_kernel_ms: 0.6,
            total_ms: 1.0,
            moe_cpu_route_ms: 0.0,
            moe_cmd_count: 1,
        },
        LogitsReadbackProfile {
            timer_spans: 2,
            bytes,
            allocation_zero_fill_ms: 0.05,
            copy_ms: 0.1,
        },
    )
}

#[test]
fn attributed_generation_preserves_sampling_and_terminal_boundaries() {
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        seed: 42,
    };
    let first = vec![3.0, 2.0, 1.0, 0.0];
    let next = vec![0.0, 3.0, 1.0, 2.0];

    let mut ordinary_sampler = Sampler::new(config).unwrap();
    let ordinary_callbacks = RefCell::new(Vec::new());
    let ordinary = generate_serial(
        first.clone(),
        2,
        &[],
        &mut ordinary_sampler,
        |token| {
            ordinary_callbacks.borrow_mut().push(token);
            Ok(())
        },
        |_| Ok(next.clone()),
    )
    .unwrap();

    let mut profiled_sampler = Sampler::new(config).unwrap();
    let profiled_callbacks = RefCell::new(Vec::new());
    let (profiled, sampler_profile, transition_profile) = generate_serial_attributed(
        first,
        2,
        &[],
        &mut profiled_sampler,
        |token| {
            profiled_callbacks.borrow_mut().push(token);
            Ok(())
        },
        |_| Ok(fake_attributed_transition(next.clone())),
    )
    .unwrap();
    assert_eq!(profiled.tokens, ordinary.tokens);
    assert_eq!(profiled.stop_reason, ordinary.stop_reason);
    assert_eq!(profiled.transitions, ordinary.transitions);
    assert_eq!(
        profiled_callbacks.into_inner(),
        ordinary_callbacks.into_inner()
    );
    assert_eq!(profiled_sampler.draws(), ordinary_sampler.draws());
    assert_eq!(sampler_profile.calls, 2);
    assert_eq!(sampler_profile.timer_spans, 22);
    assert_eq!(transition_profile.calls, 1);
    assert_eq!(transition_profile.new_timer_spans, 2);

    let mut eos_sampler = Sampler::new(config).unwrap();
    let (eos, sampler_profile, transition_profile) = generate_serial_attributed(
        vec![3.0, 2.0],
        4,
        &[0],
        &mut eos_sampler,
        |_| -> Result<()> { panic!("EOS must not reach the callback") },
        |_| -> Result<_> { panic!("EOS must not be transitioned") },
    )
    .unwrap();
    assert_eq!(eos.tokens, [0]);
    assert_eq!(eos.stop_reason, StopReason::Eos);
    assert_eq!(eos.transitions, 0);
    assert_eq!(sampler_profile.calls, 1);
    assert_eq!(transition_profile.calls, 0);

    let mut middle_eos_sampler = Sampler::new(config).unwrap();
    let delivered = RefCell::new(Vec::new());
    let (middle_eos, _, transition_profile) = generate_serial_attributed(
        vec![3.0, 2.0],
        4,
        &[1],
        &mut middle_eos_sampler,
        |token| {
            delivered.borrow_mut().push(token);
            Ok(())
        },
        |_| Ok(fake_attributed_transition(vec![0.0, 3.0])),
    )
    .unwrap();
    assert_eq!(middle_eos.tokens, [0, 1]);
    assert_eq!(middle_eos.stop_reason, StopReason::Eos);
    assert_eq!(middle_eos.transitions, 1);
    assert_eq!(delivered.into_inner(), [0]);
    assert_eq!(transition_profile.calls, 1);
}

#[test]
fn attributed_one_token_generation_has_no_transition() {
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        seed: 42,
    };
    let mut sampler = Sampler::new(config).unwrap();
    let delivered = RefCell::new(Vec::new());
    let (generation, sampler_profile, transition_profile) = generate_serial_attributed(
        vec![0.0, 3.0],
        1,
        &[],
        &mut sampler,
        |token| {
            delivered.borrow_mut().push(token);
            Ok(())
        },
        |_| -> Result<_> { panic!("one-token output must not transition") },
    )
    .unwrap();
    assert_eq!(generation.tokens, [1]);
    assert_eq!(generation.stop_reason, StopReason::TokenLimit);
    assert_eq!(generation.transitions, 0);
    assert_eq!(delivered.into_inner(), [1]);
    assert_eq!(sampler.draws(), 1);
    assert_eq!(sampler_profile.calls, 1);
    assert_eq!(transition_profile.calls, 0);
}

#[test]
fn attributed_generation_preserves_callback_and_transition_failures() {
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        seed: 42,
    };
    let mut callback_sampler = Sampler::new(config).unwrap();
    let transitions = Cell::new(0usize);
    let error = generate_serial_attributed(
        vec![3.0, 2.0],
        2,
        &[],
        &mut callback_sampler,
        |_| bail!("callback failed"),
        |_| {
            transitions.set(transitions.get() + 1);
            Ok(fake_attributed_transition(vec![3.0, 2.0]))
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("callback failed"));
    assert_eq!(transitions.get(), 0);

    let mut transition_sampler = Sampler::new(config).unwrap();
    let callbacks = Cell::new(0usize);
    let error = generate_serial_attributed(
        vec![3.0, 2.0],
        2,
        &[],
        &mut transition_sampler,
        |_| {
            callbacks.set(callbacks.get() + 1);
            Ok(())
        },
        |_| bail!("transition failed"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("transition failed"));
    assert_eq!(callbacks.get(), 1);
}

#[test]
fn sampling_clock_probe_and_prompt_digest_match_frozen_contract() {
    let probe = measure_sampling_clock_probe();
    assert_eq!(probe.batches, 7);
    assert_eq!(probe.iterations_per_batch, 100_000);
    assert_eq!(probe.new_timer_spans, 1_662);
    assert!(probe.pair_ns.iter().all(|value| *value >= 0.0));
    assert!(probe.upper_pair_ns >= probe.pair_ns.iter().copied().fold(0.0, f64::max));
    assert_eq!(
        token_ids_sha256_i32le(&[1, -2, 248_319]),
        "3f37364bc87f9ff835c64d4bdb3d993fe35e530097697da8cbacb5e6f92119d5"
    );
}

#[test]
fn greedy_generation_does_not_transition_eos() {
    let delivered = RefCell::new(Vec::new());
    let generation = generate_greedy(
        logits_with_argmax(1),
        4,
        &[1],
        |token| {
            delivered.borrow_mut().push(token);
            Ok(())
        },
        |_| -> Result<Vec<f32>> { panic!("EOS must not be consumed") },
    )
    .unwrap();

    assert!(delivered.into_inner().is_empty());
    assert_eq!(generation.tokens, [1]);
    assert_eq!(generation.transitions, 0);
    assert_eq!(generation.transition_ms, 0.0);
    assert!(generation.first_transition_ms.is_none());
    assert_eq!(generation.stop_reason, StopReason::Eos);
}

#[test]
fn greedy_generation_honors_every_producer_stop_token() {
    for terminal in [1_i32, 3] {
        let generation = generate_greedy(
            logits_with_argmax(terminal as usize),
            4,
            &[1, 3],
            |_| Ok(()),
            |_| -> Result<Vec<f32>> { panic!("stop token must not be consumed") },
        )
        .unwrap();

        assert_eq!(generation.tokens, [terminal]);
        assert_eq!(generation.transitions, 0);
        assert_eq!(generation.stop_reason, StopReason::Eos);
    }
}

#[test]
fn greedy_generation_one_token_needs_no_transition() {
    let generation = generate_greedy(
        logits_with_argmax(2),
        1,
        &[],
        |_| Ok(()),
        |_| -> Result<Vec<f32>> { panic!("terminal token must not be consumed") },
    )
    .unwrap();

    assert_eq!(generation.tokens, [2]);
    assert_eq!(generation.transitions, 0);
    assert_eq!(generation.stop_reason, StopReason::TokenLimit);
}

#[test]
fn greedy_generation_rejects_zero_token_limit() {
    let error = generate_greedy(
        logits_with_argmax(2),
        0,
        &[],
        |_| Ok(()),
        |_| Ok(logits_with_argmax(0)),
    )
    .unwrap_err();

    assert!(error.to_string().contains("max_tokens must be >= 1"));
}

#[test]
fn greedy_generation_does_not_transition_middle_eos() {
    let delivered = RefCell::new(Vec::new());
    let generation = generate_greedy(
        logits_with_argmax(1),
        4,
        &[2],
        |token| {
            delivered.borrow_mut().push(token);
            Ok(())
        },
        |token| {
            assert_eq!(token, 1);
            Ok(logits_with_argmax(2))
        },
    )
    .unwrap();

    assert_eq!(delivered.into_inner(), [1]);
    assert_eq!(generation.tokens, [1, 2]);
    assert_eq!(generation.transitions, 1);
    assert_eq!(generation.stop_reason, StopReason::Eos);
}

#[test]
fn greedy_generation_delivers_token_before_transition_error() {
    let events = RefCell::new(Vec::new());
    let error = generate_greedy(
        logits_with_argmax(1),
        3,
        &[],
        |token| {
            events.borrow_mut().push(format!("token:{token}"));
            Ok(())
        },
        |token| {
            events.borrow_mut().push(format!("transition:{token}"));
            bail!("transition failed")
        },
    )
    .unwrap_err();

    assert!(error.to_string().contains("transition failed"));
    assert_eq!(events.into_inner(), ["token:1", "transition:1"]);
}

#[test]
fn metal_allocation_samples_keep_signed_deltas_and_sampled_max() {
    let samples = metal_allocation_samples(100, 110, 120, 90, 150, 140, 130, 80);

    assert_eq!(
        samples.process_model_ready.delta_from_request_start_bytes,
        -10
    );
    assert_eq!(samples.request_start.delta_from_request_start_bytes, 0);
    assert_eq!(samples.after_scratch.delta_from_model_ready_bytes, 20);
    assert_eq!(samples.after_scratch.delta_from_request_start_bytes, 10);
    assert_eq!(samples.after_sequence.delta_from_model_ready_bytes, -10);
    assert_eq!(samples.after_sequence.delta_from_request_start_bytes, -20);
    assert_eq!(samples.after_request_state_drop.current_bytes, 80);
    assert_eq!(samples.current_allocated_sampled_max_bytes, 150);
}

#[test]
fn request_timing_mode_accepts_only_single_turn_file_sidecars() {
    let valid = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--request-timings",
        "timings.jsonl",
    ])
    .unwrap();
    validate_request_timing_mode(&valid).unwrap();

    let valid_pair = Args::try_parse_from([
        "qwen",
        "--model",
        "model.gguf",
        "--prompt",
        "hello",
        "--request-timings",
        "timings.jsonl",
        "--request-timing-warm-followup",
    ])
    .unwrap();
    validate_request_timing_mode(&valid_pair).unwrap();

    for argv in [
        vec!["qwen", "--info", "--request-timings", "timings.jsonl"],
        vec![
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "requests.jsonl",
            "--request-timings",
            "timings.jsonl",
        ],
        vec![
            "qwen",
            "--model",
            "model.gguf",
            "--request-timings",
            "timings.jsonl",
        ],
        vec![
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--request-timings",
            "-",
        ],
        vec![
            "qwen",
            "--prompt",
            "hello",
            "--request-timings",
            "timings.jsonl",
        ],
        vec![
            "qwen",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--request-timing-warm-followup",
        ],
    ] {
        let args = Args::try_parse_from(argv).unwrap();
        assert!(validate_request_timing_mode(&args).is_err());
    }
}

#[test]
fn request_timing_invariants_require_ordered_milestones_and_n_minus_one() {
    validate_request_timing_invariants(10.0, 11.0, 20.0, 21.0, 4, 3).unwrap();
    validate_request_timing_invariants(10.0, 20.0, 20.0, 21.0, 1, 0).unwrap();
    assert!(validate_request_timing_invariants(12.0, 11.0, 20.0, 21.0, 4, 3).is_err());
    assert!(validate_request_timing_invariants(10.0, 11.0, 20.0, 21.0, 4, 4).is_err());
    assert!(validate_request_timing_invariants(10.0, f64::NAN, 20.0, 21.0, 4, 3).is_err());
}

#[test]
fn pipeline_cache_phase_metrics_align_prefill_and_generation() {
    let snapshot = |misses, miss_wall_ns, compiler_wall_ns| MetalPipelineCacheMetrics {
        misses,
        miss_wall_ns,
        compiler_wall_ns,
    };
    let metrics = pipeline_cache_phase_metrics(
        snapshot(0, 0, 0),
        snapshot(1, 10, 8),
        snapshot(3, 30, 25),
        snapshot(4, 40, 33),
    );

    assert_eq!(metrics.prefill.misses, 2);
    assert_eq!(metrics.prefill.miss_wall_ns, 20);
    assert_eq!(metrics.prefill.compiler_wall_ns, 17);
    assert_eq!(metrics.generation.misses, 1);
    assert_eq!(metrics.generation.miss_wall_ns, 10);
    assert_eq!(metrics.generation.compiler_wall_ns, 8);
    assert_eq!(metrics.total.misses, 4);
    assert_eq!(metrics.total.miss_wall_ns, 40);
    assert_eq!(metrics.total.compiler_wall_ns, 33);
}

// ---- request-stats-jsonl envelope shape tests ----
//
// These tests pin the common envelope shape for schema v1. Adding a field
// in an additive way should keep these tests passing. Renaming, removing,
// or restructuring a field should require a schema_version bump AND
// updating these tests intentionally.

fn sample_measured_ok() -> RequestStatsMeasured {
    RequestStatsMeasured {
        input_tokens: 100,
        output_tokens: 50,
        transitions: 49,
        stop_reason: StopReason::Eos,
        tokenizer_ms: 10.5,
        load_ms: 40.1,
        prefill_ms: 1000.0,
        prefill_tps: 100.0,
        decode_ms: 2000.0,
        decode_tps: 25.0,
        transition_tps: 24.5,
        total_ms: 3050.6,
        output_fingerprint: GeneratedTokenSha256Digest::of(&[0i32; 4]),
    }
}

#[test]
fn request_stats_record_v1_envelope_shape_is_stable() {
    let measured = sample_measured_ok();
    let record = build_deepseek_v4_single_turn_stats_record(
        "inv-42-1234567890",
        "messages_0731_chat",
        "layer_major_chunks",
        4096,
        &measured,
    );
    let json = serde_json::to_value(&record).unwrap();

    // Common core
    assert_eq!(json["schema"], "qwen-llm.request-stats");
    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["record_type"], "request_stats");
    assert_eq!(json["invocation_id"], "inv-42-1234567890");
    assert_eq!(json["request_index"], 0);
    assert_eq!(json["status"], "ok");

    // Model
    assert_eq!(json["model"]["family"], "deepseek_v4");

    // Input (kind + template split from prompt_kind)
    assert_eq!(json["input"]["kind"], "messages");
    assert_eq!(json["input"]["template"], "0731_chat");

    // Usage: u64 wire counts
    assert_eq!(json["usage"]["input_tokens"], 100);
    assert_eq!(json["usage"]["output_tokens"], 50);

    // Finish (envelope enum, not internal StopReason)
    assert_eq!(json["finish"]["reason"], "eos");

    // Timing (all ms, per-request; total is measured at outer boundary)
    assert_eq!(json["timing_ms"]["total"], 3050.6);
    assert_eq!(json["timing_ms"]["tokenization"], 10.5);
    assert_eq!(json["timing_ms"]["prefill"], 1000.0);
    assert_eq!(json["timing_ms"]["decode"], 2000.0);

    // Throughput (tokens/sec)
    assert_eq!(json["throughput_tps"]["prefill"], 100.0);
    assert_eq!(json["throughput_tps"]["decode"], 25.0);

    // Fingerprint: algorithm names encoding, value is hex of the [u8; 32]
    assert_eq!(
        json["output_fingerprint"]["algorithm"],
        "sha256-qwen-generated-token-ids-v1"
    );
    // sample_measured_ok uses GeneratedTokenSha256Digest::of(&[0i32; 4]);
    // literal digest is pinned by request_stats_fingerprint_matches_literal_known_vector.
    assert_eq!(
        json["output_fingerprint"]["value"],
        "f06d4689226359d0fe3e105fb7d432520d855d8aad664db95596ae9c34d90eca"
    );

    // Build (commit + dirty are compile-time env vars; values vary per build)
    assert!(json["build"]["commit"].is_string());
    assert!(json["build"]["dirty"].is_boolean());

    // Diagnostics: family-namespaced, independently versioned
    assert_eq!(json["diagnostics"]["deepseek_v4"]["schema_version"], 1);
    assert_eq!(
        json["diagnostics"]["deepseek_v4"]["prefill_mode"],
        "layer_major_chunks"
    );
    assert_eq!(
        json["diagnostics"]["deepseek_v4"]["prefill_chunk_cap"],
        4096
    );
    assert_eq!(json["diagnostics"]["deepseek_v4"]["transitions"], 49);
    assert_eq!(json["diagnostics"]["deepseek_v4"]["transition_tps"], 24.5);
    assert_eq!(json["diagnostics"]["deepseek_v4"]["load_ms"], 40.1);
}

#[test]
fn request_stats_split_ds4_prompt_kind_restricts_common_kind_vocabulary() {
    assert_eq!(
        split_ds4_prompt_kind("messages_0731_chat"),
        ("messages", Some("0731_chat"))
    );
    assert_eq!(
        split_ds4_prompt_kind("messages_0731_thinking"),
        ("messages", Some("0731_thinking"))
    );
    assert_eq!(split_ds4_prompt_kind("raw"), ("raw", None));
    // Empty template suffix collapses to unknown, not "messages" with empty template.
    assert_eq!(split_ds4_prompt_kind("messages_"), ("unknown", None));
    // Unknown backend labels do NOT get promoted into the common `input.kind`
    // vocabulary — they collapse to "unknown".
    assert_eq!(split_ds4_prompt_kind("something_else"), ("unknown", None));
    assert_eq!(split_ds4_prompt_kind(""), ("unknown", None));
}

#[test]
fn request_stats_parse_build_dirty_is_case_insensitive() {
    // Any-case zero-ish values are clean
    for clean in ["0", "", "false", "FALSE", "False", "no", "NO", "No"] {
        assert!(!parse_build_dirty(clean), "expected {clean:?} to be clean");
    }
    // Any-case truthy values are dirty
    for dirty in ["1", "true", "TRUE", "True", "yes", "YES", "Yes", "dirty"] {
        assert!(parse_build_dirty(dirty), "expected {dirty:?} to be dirty");
    }
}

#[test]
fn request_stats_finish_reason_covers_all_stop_reasons_exhaustively() {
    // If a new StopReason variant is added, this match will fail to
    // compile — forcing a schema decision rather than silent drift.
    for reason in [StopReason::Eos, StopReason::TokenLimit] {
        let mapped: RequestStatsFinishReason = reason.into();
        let serialized = serde_json::to_value(mapped).unwrap();
        match reason {
            StopReason::Eos => assert_eq!(serialized, "eos"),
            StopReason::TokenLimit => assert_eq!(serialized, "token_limit"),
        }
    }
}

#[test]
fn request_stats_non_finite_metrics_coerced_to_zero() {
    let mut measured = sample_measured_ok();
    // Pathological values that would otherwise become JSON `null`.
    measured.total_ms = f64::NAN;
    measured.tokenizer_ms = f64::INFINITY;
    measured.prefill_ms = f64::INFINITY;
    measured.decode_ms = -0.001;
    measured.prefill_tps = f64::NAN;
    measured.decode_tps = f64::NEG_INFINITY;
    measured.transition_tps = f64::NAN;
    measured.load_ms = -100.0;

    let record = build_deepseek_v4_single_turn_stats_record(
        "inv-x",
        "messages_0731_chat",
        "layer_major_chunks",
        4096,
        &measured,
    );
    let json = serde_json::to_value(&record).unwrap();

    for (parent, child) in [
        ("timing_ms", "total"),
        ("timing_ms", "tokenization"),
        ("timing_ms", "prefill"),
        ("timing_ms", "decode"),
        ("throughput_tps", "prefill"),
        ("throughput_tps", "decode"),
    ] {
        let v = json[parent][child].as_f64().unwrap_or_else(|| {
            panic!(
                "{parent}.{child} was not a JSON number (got {:?})",
                json[parent][child]
            )
        });
        assert_eq!(v, 0.0, "{parent}.{child}");
    }
    assert_eq!(
        json["diagnostics"]["deepseek_v4"]["transition_tps"]
            .as_f64()
            .unwrap(),
        0.0
    );
    assert_eq!(
        json["diagnostics"]["deepseek_v4"]["load_ms"]
            .as_f64()
            .unwrap(),
        0.0
    );
}

#[test]
fn request_stats_fingerprint_matches_literal_known_vector() {
    // Literal known-vector: `sha256(domain-separator || len_u64_le ||
    // (token_i32_le)*)` over tokens [1,2,3,4,5] MUST match this exact
    // hex digest. Changing the hash function, domain separator, length
    // encoding, or token encoding will produce a different digest and
    // fail this test — which is the whole point of naming the algorithm
    // `sha256-qwen-generated-token-ids-v1`.
    let tokens: [i32; 5] = [1, 2, 3, 4, 5];
    const EXPECTED_HEX: &str = "7f016c59a05cecd27d3348aedae1275ec5cb5be58aeb3ea599110e4a7ce5b304";
    let digest = GeneratedTokenSha256Digest::of(&tokens);
    assert_eq!(digest.hex(), EXPECTED_HEX);
    assert_eq!(generated_token_sha256(&tokens), EXPECTED_HEX);
    // And the same digest, wired through the record builder, must land
    // in output_fingerprint.value byte-identical.
    let mut measured = sample_measured_ok();
    measured.output_fingerprint = digest;
    let record = build_deepseek_v4_single_turn_stats_record(
        "inv-y",
        "messages_0731_chat",
        "layer_major_chunks",
        4096,
        &measured,
    );
    let json = serde_json::to_value(&record).unwrap();
    assert_eq!(json["output_fingerprint"]["value"], EXPECTED_HEX);
}

#[test]
fn request_stats_hex_encode_bytes_is_lowercase_padded() {
    assert_eq!(hex_encode_bytes(&[]), "");
    assert_eq!(hex_encode_bytes(&[0x00, 0x0f, 0xff]), "000fff");
    assert_eq!(hex_encode_bytes(&[0xab; 4]), "abababab");
}

#[test]
fn request_stats_append_jsonl_serializes_one_line_per_record() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "qwen-request-stats-test-{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);

    for i in 0..3u64 {
        let mut m = sample_measured_ok();
        m.input_tokens = i;
        m.output_fingerprint = GeneratedTokenSha256Digest::of(&[i as i32]);
        let record = build_deepseek_v4_single_turn_stats_record(
            "inv-z",
            "messages_0731_chat",
            "layer_major_chunks",
            4096,
            &m,
        );
        append_jsonl_record(&path, &record, "test stats").unwrap();
    }

    let contents = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = contents.lines().collect();
    assert_eq!(lines.len(), 3, "expected exactly one line per record");
    assert!(contents.ends_with('\n'), "file must end with newline");
    for (i, line) in lines.iter().enumerate() {
        let json: serde_json::Value = serde_json::from_str(line).expect("valid JSON per line");
        assert_eq!(json["schema"], "qwen-llm.request-stats");
        assert_eq!(json["invocation_id"], "inv-z");
        assert_eq!(json["usage"]["input_tokens"], i);
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn request_stats_status_serializes_as_lowercase() {
    assert_eq!(serde_json::to_value(RequestStatsStatus::Ok).unwrap(), "ok");
    assert_eq!(
        serde_json::to_value(RequestStatsStatus::Error).unwrap(),
        "error"
    );
    assert_eq!(
        serde_json::to_value(RequestStatsStatus::Cancelled).unwrap(),
        "cancelled"
    );
}

#[test]
fn request_stats_success_only_fields_are_optional_for_error_records() {
    // Verify the envelope structurally supports a future error/cancelled
    // record without a breaking restructuring: success-only fields all
    // permit `None` and skip serialization when absent.
    let record = RequestStatsRequestRecord {
        schema: "qwen-llm.request-stats",
        schema_version: 1,
        record_type: "request_stats",
        invocation_id: "inv-err",
        request_index: 0,
        status: RequestStatsStatus::Error,
        model: RequestStatsModel {
            family: "deepseek_v4",
        },
        input: RequestStatsInput {
            kind: "messages",
            template: Some("0731_chat"),
        },
        usage: None,
        finish: None,
        timing_ms: None,
        throughput_tps: None,
        output_fingerprint: None,
        build: RequestStatsBuild {
            commit: "abc",
            dirty: false,
        },
        diagnostics: None,
    };
    let json = serde_json::to_value(&record).unwrap();
    assert_eq!(json["status"], "error");
    assert!(json.get("usage").is_none());
    assert!(json.get("finish").is_none());
    assert!(json.get("timing_ms").is_none());
    assert!(json.get("throughput_tps").is_none());
    assert!(json.get("output_fingerprint").is_none());
    assert!(json.get("diagnostics").is_none());
}

#[test]
fn request_stats_invocation_id_is_stable_and_hex_16_bytes() {
    // Force init once; every subsequent access must return the same ID.
    let a = &*INVOCATION_ID;
    let b = &*INVOCATION_ID;
    assert_eq!(a, b);
    // 16 bytes hex-encoded = 32 chars
    assert_eq!(a.len(), 32, "invocation_id is 16 bytes hex-encoded");
    assert!(
        a.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
        "invocation_id must be lowercase hex: got {a}"
    );
}

#[test]
fn request_stats_invocation_id_encodes_entropy_bytes_verbatim() {
    // Deterministic entropy source proves that the encoding is exactly
    // 16 bytes hex-encoded, in order, lowercase. A constant-string
    // implementation would fail this test.
    let bytes: [u8; 16] = [
        0x00, 0x01, 0x0f, 0x10, 0xab, 0xcd, 0xef, 0x42, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32,
        0x10,
    ];
    let id = generate_invocation_id_from(EntropySource::Deterministic(bytes)).unwrap();
    assert_eq!(id, "00010f10abcdef42fedcba9876543210");
}

#[test]
fn request_stats_valid_finite_metrics_pass_through_unchanged() {
    // Complement to non_finite_metrics_coerced_to_zero: a valid finite
    // measurement must NOT be zeroed. This catches over-aggressive
    // sanitization.
    let measured = sample_measured_ok();
    let record = build_deepseek_v4_single_turn_stats_record(
        "inv-v",
        "messages_0731_chat",
        "layer_major_chunks",
        4096,
        &measured,
    );
    let json = serde_json::to_value(&record).unwrap();
    assert_eq!(json["timing_ms"]["total"], 3050.6);
    assert_eq!(json["timing_ms"]["tokenization"], 10.5);
    assert_eq!(json["timing_ms"]["prefill"], 1000.0);
    assert_eq!(json["timing_ms"]["decode"], 2000.0);
    assert_eq!(json["throughput_tps"]["prefill"], 100.0);
    assert_eq!(json["throughput_tps"]["decode"], 25.0);
    assert_eq!(json["diagnostics"]["deepseek_v4"]["transition_tps"], 24.5);
    assert_eq!(json["diagnostics"]["deepseek_v4"]["load_ms"], 40.1);
}

#[test]
fn request_stats_split_ds4_prompt_kind_rejects_bare_messages() {
    // Bare "messages" (no underscore + suffix) is not a valid template
    // marker; it must collapse to "unknown" rather than becoming a
    // second `input.kind = "messages"` value with no template.
    assert_eq!(split_ds4_prompt_kind("messages"), ("unknown", None));
}

#[test]
fn request_stats_append_jsonl_repairs_partial_prior_tail() {
    // Pre-seed the file with an unfinished record (no trailing newline).
    // The next append MUST NOT concatenate its record onto the abandoned
    // partial one — it must start on a fresh line.
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "qwen-request-stats-tail-{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    std::fs::write(&path, b"{\"partial\":\"orphan_no_newline\"").unwrap();

    let measured = sample_measured_ok();
    let record = build_deepseek_v4_single_turn_stats_record(
        "inv-tail",
        "messages_0731_chat",
        "layer_major_chunks",
        4096,
        &measured,
    );
    append_jsonl_record(&path, &record, "tail-repair test").unwrap();

    let contents = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = contents.lines().collect();
    // Line 0 is the orphan partial (unchanged); line 1 is our new record.
    // Critically, NO line combines both, and line 1 must parse as valid JSON.
    assert_eq!(
        lines.len(),
        2,
        "expected orphan preserved on its own line, our record on the next"
    );
    assert_eq!(lines[0], "{\"partial\":\"orphan_no_newline\"");
    let json: serde_json::Value =
        serde_json::from_str(lines[1]).expect("appended record must parse as standalone JSON");
    assert_eq!(json["invocation_id"], "inv-tail");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn request_stats_measured_total_ms_can_diverge_from_phase_sum() {
    // Prove the outer-boundary total is DECOUPLED from the phase sum
    // (a synthesized implementation would fail this by matching the
    // sum exactly).
    let mut measured = sample_measured_ok();
    measured.total_ms = 9999.9; // arbitrary value, not the phase sum
    let record = build_deepseek_v4_single_turn_stats_record(
        "inv-t",
        "messages_0731_chat",
        "layer_major_chunks",
        4096,
        &measured,
    );
    let json = serde_json::to_value(&record).unwrap();
    assert_eq!(json["timing_ms"]["total"], 9999.9);
    // Phase fields must remain independently measured, not derived.
    assert_eq!(json["timing_ms"]["prefill"], 1000.0);
    assert_eq!(json["timing_ms"]["decode"], 2000.0);
}

#[test]
fn request_stats_invocation_id_fails_closed_when_entropy_unavailable() {
    // The fail-closed branch: when secure entropy is unavailable,
    // generate_invocation_id_from returns an Err rather than a weak
    // deterministic fallback. This is what makes the LazyLock panic
    // in main() the correct behavior (loud failure > silent weak ID).
    let err = generate_invocation_id_from(EntropySource::ForceFail)
        .expect_err("must return Err when entropy source fails");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("test-injected entropy failure"),
        "error message must surface the underlying failure: {msg}"
    );
}

#[test]
fn request_stats_fingerprint_newtype_bytes_only_from_of() {
    // The GeneratedTokenSha256Digest tuple field lives in a private
    // child module (`mod fingerprint`), so no crate code — including
    // this test — can construct one with arbitrary bytes. The only
    // way to obtain a value is via `of()`. This test compiles iff that
    // invariant holds: a direct tuple construction would fail with
    // "field is private", and a bytes constructor doesn't exist.
    let a = GeneratedTokenSha256Digest::of(&[]);
    let b = GeneratedTokenSha256Digest::of(&[]);
    assert_eq!(a, b, "of() over identical input is deterministic");
    // Sanity: the digest exposes bytes via `as_bytes()` (test-only) for
    // hex comparison — but that's read-only, not a constructor path.
    assert_eq!(a.as_bytes().len(), 32);
}

mod jsonl_templated_rows {
    use crate::open_responses::items::QwenTemplate;
    use crate::{
        JsonlInputLabel, JsonlRequest, JsonlRowProtocol, QwenUserPromptProtocol,
        resolve_jsonl_request_input,
    };

    fn row(json: &str) -> JsonlRequest {
        serde_json::from_str(json).unwrap()
    }

    fn pinned() -> JsonlRowProtocol {
        JsonlRowProtocol::Resolved(QwenUserPromptProtocol::for_test(
            false,
            QwenTemplate::Qwen36,
        ))
    }

    #[test]
    fn raw_rows_keep_their_bytes_and_tokenizer_specials() {
        let (prompt, specials, label) =
            resolve_jsonl_request_input(&row(r#"{"prompt":"  hi  "}"#), 1, &pinned()).unwrap();
        assert_eq!(prompt, "  hi  ");
        assert!(specials);
        assert_eq!(label, JsonlInputLabel::RAW);
    }

    #[test]
    fn exactly_one_input_form_is_required() {
        for json in [
            r#"{}"#,
            r#"{"prompt":"a","user":"b"}"#,
            r#"{"prompt":"a","prompt_file":"/x"}"#,
        ] {
            let err = resolve_jsonl_request_input(&row(json), 7, &pinned()).unwrap_err();
            assert!(err.to_string().contains("exactly one of"), "{json}: {err}");
        }
    }

    #[test]
    fn rendering_controls_are_rejected_on_raw_rows() {
        for json in [
            r#"{"prompt":"a","system":"s"}"#,
            r#"{"prompt":"a","no_thinking":true}"#,
            r#"{"prompt":"a","reasoning_effort":"low"}"#,
        ] {
            let err = resolve_jsonl_request_input(&row(json), 3, &pinned()).unwrap_err();
            assert!(
                err.to_string().contains("apply to user rows only"),
                "{json}: {err}"
            );
        }
    }

    #[test]
    fn unknown_reasoning_effort_values_fail_at_binding_with_the_family_levels() {
        // The row parses (effort is a spelling bound per family), and the
        // binding names the model's accepted levels.
        let row = row(r#"{"user":"hi","reasoning_effort":"turbo"}"#);
        let q38 = JsonlRowProtocol::Resolved(QwenUserPromptProtocol::for_test(
            true,
            QwenTemplate::Qwen38,
        ));
        let err = resolve_jsonl_request_input(&row, 1, &q38).unwrap_err();
        assert!(
            format!("{err:#}").contains("none|low|medium|xhigh"),
            "{err:#}"
        );
        assert!(serde_json::from_str::<JsonlRequest>(r#"{"user":"hi","unknown":1}"#).is_err());
    }

    #[test]
    fn effort_requires_qwen38_and_conflicts_with_no_thinking() {
        let err = resolve_jsonl_request_input(
            &row(r#"{"user":"hi","reasoning_effort":"low"}"#),
            4,
            &pinned(),
        )
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("no reasoning-effort control"),
            "{err:#}"
        );
        let q38 = QwenUserPromptProtocol::for_test(true, QwenTemplate::Qwen38);
        let err = resolve_jsonl_request_input(
            &row(r#"{"user":"hi","reasoning_effort":"low","no_thinking":true}"#),
            4,
            &JsonlRowProtocol::Resolved(q38),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("cannot be combined"), "{err:#}");
    }

    /// An unpinned template follows the same family rule as `run --user`:
    /// plain chat renders the legacy ChatML contract (visible in the row's
    /// `template` label); reasoning controls refuse with their capability
    /// code because the transition bytes are unproven there.
    #[test]
    fn user_rows_on_unpinned_templates_follow_the_run_rule() {
        let generic = JsonlRowProtocol::Resolved(QwenUserPromptProtocol::for_test(
            false,
            QwenTemplate::Generic,
        ));
        let (rendered, specials, label) =
            resolve_jsonl_request_input(&row(r#"{"user":"hi","system":"be terse"}"#), 2, &generic)
                .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(!specials);
        assert_eq!(label.template, Some("generic"));
        let err =
            resolve_jsonl_request_input(&row(r#"{"user":"hi","no_thinking":true}"#), 3, &generic)
                .unwrap_err();
        assert!(
            format!("{err:#}")
                .contains("no-thinking requires a model whose chat template is pinned"),
            "{err:#}"
        );
    }

    /// Every single-turn `user`/`system` case in the Qwen3.6 oracle, driven
    /// as a batch row, renders the released bytes and is tokenized without
    /// added specials.
    #[test]
    fn user_rows_render_the_qwen36_oracle_bytes() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/qwen36_chat_template_oracle_v1.json"
        ))
        .unwrap();
        let mut checked = 0;
        for case in fixture["cases"].as_array().unwrap() {
            let input = &case["input"];
            let messages = input["messages"].as_array().unwrap();
            let roles: Vec<&str> = messages
                .iter()
                .map(|m| m["role"].as_str().unwrap())
                .collect();
            let shape_ok = matches!(roles.as_slice(), ["user"] | ["system", "user"])
                && input["tools"].as_array().is_none_or(|t| t.is_empty())
                && input["preserve_thinking"] != serde_json::json!(true);
            if !shape_ok {
                continue;
            }
            let user = messages.last().unwrap()["content"].as_str().unwrap();
            let system = (roles.len() == 2).then(|| messages[0]["content"].as_str().unwrap());
            let mut row = serde_json::json!({ "user": user });
            if let Some(system) = system {
                row["system"] = serde_json::json!(system);
            }
            if input["enable_thinking"] == serde_json::json!(false) {
                row["no_thinking"] = serde_json::json!(true);
            }
            let request: JsonlRequest = serde_json::from_value(row).unwrap();
            let (rendered, specials, label) =
                resolve_jsonl_request_input(&request, 1, &pinned()).unwrap();
            assert_eq!(
                rendered,
                case["rendered"].as_str().unwrap(),
                "case {}",
                case["id"]
            );
            assert!(!specials);
            assert_eq!(label.kind, "messages");
            assert_eq!(label.template, Some("qwen36"));
            checked += 1;
        }
        assert!(
            checked >= 5,
            "expected the single-turn oracle cases, checked {checked}"
        );
    }
}

mod jsonl_typed_preparation {
    use super::*;
    use crate::{JsonlRowOutcome, JsonlRowProtocol, prepare_jsonl_row};
    use qwen_llm::test_fixtures::QWEN35_0_8B_F32;

    /// A real tokenizer from the smallest local fixture; header-only, no
    /// Metal. Absent fixture skips; a present but unreadable fixture fails.
    fn tokenizer() -> Option<(Tokenizer, GgufFile)> {
        let path = QWEN35_0_8B_F32.path_or_skip()?;
        let gguf = GgufFile::open(&path)
            .unwrap_or_else(|error| panic!("open fixture {}: {error}", path.display()));
        let tokenizer = Tokenizer::from_gguf(&gguf)
            .unwrap_or_else(|error| panic!("tokenizer from {}: {error}", path.display()));
        Some((tokenizer, gguf))
    }

    fn args(extra: &[&str]) -> Args {
        let mut argv = vec!["qwen", "--model", "m.gguf", "--requests-jsonl", "r.jsonl"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).unwrap()
    }

    fn code(outcome: &JsonlRowOutcome) -> Option<(&'static str, String, usize)> {
        match outcome {
            JsonlRowOutcome::Failed(failure) => {
                assert_eq!(failure.status, "error");
                Some((failure.code, failure.id.clone(), failure.line))
            }
            _ => None,
        }
    }

    #[test]
    fn malformed_json_is_a_typed_failure_with_a_line_derived_id() {
        let Some((tokenizer, _)) = tokenizer() else {
            return;
        };
        let outcome = prepare_jsonl_row(
            7,
            "{not json",
            &tokenizer,
            &args(&[]),
            &JsonlRowProtocol::NotOrdinaryQwen,
        );
        assert_eq!(code(&outcome), Some(("parse", "line-7".into(), 7)));
    }

    #[test]
    fn blank_and_comment_lines_are_skipped() {
        let Some((tokenizer, _)) = tokenizer() else {
            return;
        };
        for line in ["", "   ", "# note"] {
            assert!(matches!(
                prepare_jsonl_row(
                    1,
                    line,
                    &tokenizer,
                    &args(&[]),
                    &JsonlRowProtocol::NotOrdinaryQwen
                ),
                JsonlRowOutcome::Skipped
            ));
        }
    }

    #[test]
    fn shape_capacity_and_executor_constraints_fail_at_preparation() {
        let Some((tokenizer, _)) = tokenizer() else {
            return;
        };
        let a = args(&[]);
        let shape = prepare_jsonl_row(
            2,
            r#"{"id":"s","prompt":"a","user":"b"}"#,
            &tokenizer,
            &a,
            &JsonlRowProtocol::NotOrdinaryQwen,
        );
        assert_eq!(code(&shape), Some(("input", "s".into(), 2)));
        let capacity = prepare_jsonl_row(
            3,
            r#"{"id":"c","prompt":"hi","tokens":0}"#,
            &tokenizer,
            &a,
            &JsonlRowProtocol::NotOrdinaryQwen,
        );
        assert_eq!(code(&capacity), Some(("capacity", "c".into(), 3)));
        let too_big = prepare_jsonl_row(
            4,
            r#"{"id":"b","prompt":"hi","tokens":10}"#,
            &tokenizer,
            &args(&["--max-context-tokens", "4"]),
            &JsonlRowProtocol::NotOrdinaryQwen,
        );
        assert_eq!(code(&too_big), Some(("capacity", "b".into(), 4)));
        let forced = args(&["--batch-size", "8"]);
        let sampled = prepare_jsonl_row(
            5,
            r#"{"id":"t","prompt":"hi","sampling":{"temperature":0.7}}"#,
            &tokenizer,
            &forced,
            &JsonlRowProtocol::NotOrdinaryQwen,
        );
        assert_eq!(code(&sampled), Some(("executor_constraint", "t".into(), 5)));
        let cached = prepare_jsonl_row(
            6,
            r#"{"id":"k","prompt":"hi","cache_prefix_tokens":4}"#,
            &tokenizer,
            &forced,
            &JsonlRowProtocol::NotOrdinaryQwen,
        );
        assert_eq!(code(&cached), Some(("executor_constraint", "k".into(), 6)));
    }

    #[test]
    fn a_runnable_row_is_prepared_with_its_source_line() {
        let Some((tokenizer, _)) = tokenizer() else {
            return;
        };
        match prepare_jsonl_row(
            9,
            r#"{"id":"ok","prompt":"hi","tokens":4}"#,
            &tokenizer,
            &args(&[]),
            &JsonlRowProtocol::NotOrdinaryQwen,
        ) {
            JsonlRowOutcome::Prepared(prepared) => {
                assert_eq!((prepared.id.as_str(), prepared.line), ("ok", 9));
                assert!(!prepared.prompt_ids.is_empty());
            }
            other => panic!("expected prepared row, got {other:?}"),
        }
    }
}
