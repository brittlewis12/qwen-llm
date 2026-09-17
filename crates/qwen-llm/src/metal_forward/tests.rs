use super::*;
use crate::forward::Forward;
use crate::gguf::GgufFile;
use crate::loader::Model;
use crate::sampling::{Sampler, SamplingConfig};

fn metal_test_context() -> Option<MetalContext> {
    crate::test_fixtures::metal_context_or_skip()
}

fn snapshot_validation_fixture() -> SessionSnapshot {
    SessionSnapshot {
        identity: SnapshotIdentity {
            model_id: 11,
            tokenizer_id: 12,
            layout_version: SNAPSHOT_LAYOUT_VERSION,
            n_attn_layers: 2,
            n_gdn_layers: 3,
            kv_dim_elements: 4,
            kv_bytes_per_token: 8,
            kv_storage_kind: SnapshotKvStorageKind::F16,
            gdn_state_elements_per_layer: 5,
            gdn_conv_elements_per_layer: 6,
        },
        prefix_tokens: vec![1, 2],
        pending_token: None,
        kv_n_pos: vec![2, 2],
        kv_k_arena: vec![0; 32],
        kv_v_arena: vec![0; 32],
        gdn_conv_arena: vec![0; 72],
        gdn_state_arena: vec![0; 60],
        final_logits: Some(vec![0.0; 4]),
        capture_tail: None,
    }
}

#[test]
fn f32_q8_mat_mat_addressing_rejects_shader_index_overflow() {
    validate_f32_q8_mat_mat_addressing(GgmlType::F32, 10_240, 320, 2_048)
        .expect("released HC down geometry");
    validate_f32_q8_mat_mat_addressing(GgmlType::Q8_0, 10_240, 320, 2_048)
        .expect("released Q8 HC down geometry");
    assert!(validate_f32_q8_mat_mat_addressing(GgmlType::F32, 65_536, 65_536, 1).is_err());
    assert!(validate_f32_q8_mat_mat_addressing(GgmlType::F32, 65_536, 1, 65_536).is_err());
    let q8_stride_overflow = (u32::MAX as usize / 32) * 32;
    assert!(validate_f32_q8_mat_mat_addressing(GgmlType::Q8_0, q8_stride_overflow, 1, 1,).is_err());
}

#[test]
fn hidden_capture_destinations_require_safe_independent_f32_ranges() {
    let Some(context) = metal_test_context() else {
        return;
    };
    const HIDDEN: usize = 8;
    const LAYERS: usize = 3;
    const ELEMENTS: usize = HIDDEN * LAYERS;
    let shape = vec![HIDDEN as u64, LAYERS as u64];
    let bytes = ELEMENTS * std::mem::size_of::<f32>();
    let valid = MetalTensor::zeros_f32(&context, shape.clone()).unwrap();
    let valid_range = validate_hidden_capture_destination("valid", &valid, &shape, bytes)
        .expect("valid capture destination");
    assert_eq!(valid_range, (0, bytes as u64));

    let wrong_shape = MetalTensor::zeros_f32(&context, vec![ELEMENTS as u64]).unwrap();
    validate_hidden_capture_destination("shape", &wrong_shape, &shape, bytes)
        .expect_err("flattened shape must be rejected");
    let wrong_dtype = MetalTensor::zeros_f16(&context, shape.clone()).unwrap();
    validate_hidden_capture_destination("dtype", &wrong_dtype, &shape, bytes)
        .expect_err("F16 capture storage must be rejected");
    let mut read_only = valid.clone();
    read_only.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    validate_hidden_capture_destination("readonly", &read_only, &shape, bytes)
        .expect_err("read-only capture storage must be rejected");
    let mut short = valid.clone();
    short.offset = std::mem::size_of::<f32>() as u64;
    validate_hidden_capture_destination("short", &short, &shape, bytes)
        .expect_err("out-of-range capture view must be rejected");

    let arena = MetalTensor::zeros_f32(&context, vec![(2 * ELEMENTS) as u64]).unwrap();
    let first = arena.view_subrange(0, shape.clone());
    let overlapping = arena.view_subrange(1, shape.clone());
    let second = arena.view_subrange(ELEMENTS as u64, shape);
    let first_range = (first.offset, first.offset + bytes as u64);
    let overlapping_range = (overlapping.offset, overlapping.offset + bytes as u64);
    let second_range = (second.offset, second.offset + bytes as u64);
    assert!(capture_ranges_overlap(
        &first,
        first_range,
        &overlapping,
        overlapping_range
    ));
    assert!(!capture_ranges_overlap(
        &first,
        first_range,
        &second,
        second_range
    ));
}

#[test]
fn snapshot_validation_accepts_only_complete_consistent_sections() {
    let valid = snapshot_validation_fixture();
    valid
        .validate_for_restore(&valid.identity, 8, Some(4))
        .expect("valid fixture");

    let mut without_logits = valid.clone();
    without_logits.final_logits = None;
    without_logits
        .validate_for_restore(&without_logits.identity, 8, Some(4))
        .expect("logits are an optional snapshot capability");

    let mut with_pending = valid.clone();
    with_pending.pending_token = Some(3);
    assert_eq!(with_pending.matched_prefix_len(), 3);
    with_pending
        .validate_for_restore(&with_pending.identity, 8, Some(4))
        .expect("valid pending token");
    with_pending.pending_token = Some(-1);
    assert!(matches!(
        with_pending.validate_for_restore(&with_pending.identity, 8, Some(4)),
        Err(SnapshotValidationError::TokenOutOfRange { index: 2, .. })
    ));
    with_pending.pending_token = Some(4);
    assert!(matches!(
        with_pending.validate_for_restore(&with_pending.identity, 8, Some(4)),
        Err(SnapshotValidationError::TokenOutOfRange { index: 2, .. })
    ));

    let mut bad = valid.clone();
    bad.kv_n_pos.pop();
    assert!(matches!(
        bad.validate_for_restore(&bad.identity, 8, Some(4)),
        Err(SnapshotValidationError::SectionLength {
            section: "kv_n_pos",
            ..
        })
    ));

    let mut bad = valid.clone();
    bad.kv_n_pos[1] = 1;
    assert!(matches!(
        bad.validate_for_restore(&bad.identity, 8, Some(4)),
        Err(SnapshotValidationError::KvPosition { layer: 1, .. })
    ));

    for section in [
        "kv_k_arena",
        "kv_v_arena",
        "gdn_conv_arena",
        "gdn_state_arena",
        "final_logits",
    ] {
        let mut bad = valid.clone();
        match section {
            "kv_k_arena" => bad.kv_k_arena.pop(),
            "kv_v_arena" => bad.kv_v_arena.pop(),
            "gdn_conv_arena" => bad.gdn_conv_arena.pop(),
            "gdn_state_arena" => bad.gdn_state_arena.pop(),
            "final_logits" => {
                bad.final_logits.as_mut().expect("logits").pop();
                Some(0)
            }
            _ => unreachable!(),
        };
        assert!(matches!(
            bad.validate_for_restore(&bad.identity, 8, Some(4)),
            Err(SnapshotValidationError::SectionLength {
                section: actual,
                ..
            }) if actual == section
        ));
    }

    let mut bad = valid.clone();
    bad.prefix_tokens[1] = 4;
    assert!(matches!(
        bad.validate_for_restore(&bad.identity, 8, Some(4)),
        Err(SnapshotValidationError::TokenOutOfRange { index: 1, .. })
    ));
    let mut bad = valid.clone();
    bad.prefix_tokens[0] = -1;
    assert!(matches!(
        bad.validate_for_restore(&bad.identity, 8, Some(4)),
        Err(SnapshotValidationError::TokenOutOfRange { index: 0, .. })
    ));
    assert!(matches!(
        valid.validate_for_restore(&valid.identity, 1, Some(4)),
        Err(SnapshotValidationError::PrefixCapacity { .. })
    ));

    let mut wrong_identity = valid.identity.clone();
    wrong_identity.layout_version += 1;
    assert!(matches!(
        valid.validate_for_restore(&wrong_identity, 8, Some(4)),
        Err(SnapshotValidationError::IdentityMismatch { .. })
    ));
}

#[test]
fn snapshot_validation_accepts_empty_and_single_state_families() {
    let mut empty = snapshot_validation_fixture();
    empty.prefix_tokens.clear();
    empty.kv_n_pos.fill(0);
    empty.kv_k_arena.clear();
    empty.kv_v_arena.clear();
    empty
        .validate_for_restore(&empty.identity, 8, Some(4))
        .expect("empty prefix");

    let mut pure_gdn = empty.clone();
    pure_gdn.identity.n_attn_layers = 0;
    pure_gdn.kv_n_pos.clear();
    pure_gdn
        .validate_for_restore(&pure_gdn.identity, 8, Some(4))
        .expect("pure GDN");

    let mut pure_attention = empty;
    pure_attention.identity.n_gdn_layers = 0;
    pure_attention.gdn_conv_arena.clear();
    pure_attention.gdn_state_arena.clear();
    pure_attention
        .validate_for_restore(&pure_attention.identity, 8, Some(4))
        .expect("pure attention");
}

#[test]
fn snapshot_validation_rejects_oversized_sections() {
    let valid = snapshot_validation_fixture();
    for section in [
        "kv_k_arena",
        "kv_v_arena",
        "gdn_conv_arena",
        "gdn_state_arena",
        "final_logits",
    ] {
        let mut bad = valid.clone();
        match section {
            "kv_k_arena" => bad.kv_k_arena.push(0),
            "kv_v_arena" => bad.kv_v_arena.push(0),
            "gdn_conv_arena" => bad.gdn_conv_arena.push(0),
            "gdn_state_arena" => bad.gdn_state_arena.push(0),
            "final_logits" => bad.final_logits.as_mut().expect("logits").push(0.0),
            _ => unreachable!(),
        }
        assert!(matches!(
            bad.validate_for_restore(&bad.identity, 8, Some(4)),
            Err(SnapshotValidationError::SectionLength {
                section: actual,
                ..
            }) if actual == section
        ));
    }
}

#[test]
fn snapshot_validation_rejects_section_length_overflow() {
    let mut snapshot = snapshot_validation_fixture();
    snapshot.identity.n_attn_layers = 0;
    snapshot.identity.n_gdn_layers = u32::MAX;
    snapshot.identity.gdn_conv_elements_per_layer = u32::MAX;
    snapshot.kv_n_pos.clear();
    snapshot.kv_k_arena.clear();
    snapshot.kv_v_arena.clear();
    assert!(matches!(
        snapshot.validate_for_restore(&snapshot.identity, 8, Some(4)),
        Err(SnapshotValidationError::LengthOverflow {
            section: "gdn_conv_arena"
        })
    ));

    assert!(matches!(
        checked_snapshot_product("kv_arena", &[usize::MAX, 2]),
        Err(SnapshotValidationError::LengthOverflow {
            section: "kv_arena"
        })
    ));
}

#[test]
fn gguf_no_copy_mode_is_strict_and_default_off() {
    assert_eq!(
        parse_gguf_owned_arena_mode(None).unwrap(),
        GgufOwnedArenaMode::Disabled
    );
    for value in ["1", "true", "TRUE", "yes", "YES"] {
        assert_eq!(
            parse_gguf_owned_arena_mode(Some(value)).unwrap(),
            GgufOwnedArenaMode::Forced
        );
    }
    for value in ["0", "false", "FALSE", "no", "NO"] {
        assert_eq!(
            parse_gguf_owned_arena_mode(Some(value)).unwrap(),
            GgufOwnedArenaMode::Disabled
        );
    }
    assert!(parse_gguf_owned_arena_mode(Some("enabled")).is_err());
    assert!(parse_gguf_owned_arena_mode(Some("")).is_err());

    assert_eq!(
        parse_gguf_no_copy_mode(None).unwrap(),
        GgufNoCopyMode::Disabled
    );
    for value in ["1", "true", "TRUE", "yes", "YES"] {
        assert_eq!(
            parse_gguf_no_copy_mode(Some(value)).unwrap(),
            GgufNoCopyMode::Forced
        );
    }
    for value in ["0", "false", "FALSE", "no", "NO"] {
        assert_eq!(
            parse_gguf_no_copy_mode(Some(value)).unwrap(),
            GgufNoCopyMode::Disabled
        );
    }
    assert!(parse_gguf_no_copy_mode(Some("enabled")).is_err());
    assert!(parse_gguf_no_copy_mode(Some("")).is_err());

    assert_eq!(
        parse_gguf_no_copy_prefault(None).unwrap(),
        GgufNoCopyPrefaultMode::Default
    );
    for value in ["1", "true", "TRUE", "yes", "YES"] {
        assert_eq!(
            parse_gguf_no_copy_prefault(Some(value)).unwrap(),
            GgufNoCopyPrefaultMode::Enabled
        );
    }
    for value in ["0", "false", "FALSE", "no", "NO"] {
        assert_eq!(
            parse_gguf_no_copy_prefault(Some(value)).unwrap(),
            GgufNoCopyPrefaultMode::Disabled
        );
    }
    assert!(parse_gguf_no_copy_prefault(Some("enabled")).is_err());
    assert!(parse_gguf_no_copy_prefault(Some("")).is_err());
}

#[test]
fn parallel_copy_policy_is_strict_and_conflict_complete() {
    assert_eq!(
        parse_gguf_parallel_copy_mode(None).unwrap(),
        GgufParallelCopyMode::Auto
    );
    for value in ["1", "true", "TRUE", "yes", "YES"] {
        assert_eq!(
            parse_gguf_parallel_copy_mode(Some(value)).unwrap(),
            GgufParallelCopyMode::ForcedCopy
        );
    }
    assert_eq!(
        parse_gguf_parallel_copy_mode(Some("pread")).unwrap(),
        GgufParallelCopyMode::ForcedPread
    );
    assert_eq!(
        parse_gguf_parallel_copy_mode(Some("PREAD")).unwrap(),
        GgufParallelCopyMode::ForcedPread
    );
    assert_eq!(
        parse_gguf_parallel_copy_mode(Some("page-rounded-copy")).unwrap(),
        GgufParallelCopyMode::ForcedPageRoundedCopy
    );
    assert_eq!(
        parse_gguf_parallel_copy_mode(Some("PAGE-ROUNDED-COPY")).unwrap(),
        GgufParallelCopyMode::ForcedPageRoundedCopy
    );
    for value in ["0", "false", "FALSE", "no", "NO"] {
        assert_eq!(
            parse_gguf_parallel_copy_mode(Some(value)).unwrap(),
            GgufParallelCopyMode::Disabled
        );
    }
    assert!(parse_gguf_parallel_copy_mode(Some("enabled")).is_err());
    assert!(parse_gguf_parallel_copy_mode(Some("")).is_err());

    let valid = |router_f16| {
        validate_parallel_copy_policy(
            GgufParallelCopyMode::ForcedCopy,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            false,
            false,
            router_f16,
        )
    };
    assert!(valid(None).is_ok());
    assert!(valid(Some("0")).is_ok());
    assert!(valid(Some("false")).is_ok());
    assert!(valid(Some("1")).is_err());
    assert!(valid(Some("invalid")).is_err());
    assert!(
        validate_parallel_copy_policy(
            GgufParallelCopyMode::ForcedPageRoundedCopy,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            false,
            false,
            None,
        )
        .is_ok()
    );
    assert!(
        validate_parallel_copy_policy(
            GgufParallelCopyMode::ForcedPread,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            false,
            false,
            None,
        )
        .is_ok()
    );
    assert!(
        validate_parallel_copy_policy(
            GgufParallelCopyMode::ForcedPread,
            GgufNoCopyMode::Forced,
            GgufOwnedArenaMode::Disabled,
            false,
            false,
            None,
        )
        .is_err()
    );
    assert!(
        validate_parallel_copy_policy(
            GgufParallelCopyMode::ForcedCopy,
            GgufNoCopyMode::Forced,
            GgufOwnedArenaMode::Disabled,
            false,
            false,
            None,
        )
        .is_err()
    );
    assert!(
        validate_parallel_copy_policy(
            GgufParallelCopyMode::ForcedCopy,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Forced,
            false,
            false,
            None,
        )
        .is_err()
    );
    assert!(
        validate_parallel_copy_policy(
            GgufParallelCopyMode::ForcedCopy,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            true,
            false,
            None,
        )
        .is_err()
    );
    assert!(
        validate_parallel_copy_policy(
            GgufParallelCopyMode::ForcedCopy,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            false,
            true,
            None,
        )
        .is_err()
    );
    assert!(
        validate_parallel_copy_policy(
            GgufParallelCopyMode::Auto,
            GgufNoCopyMode::Forced,
            GgufOwnedArenaMode::Forced,
            true,
            true,
            Some("invalid"),
        )
        .is_ok()
    );
}

#[test]
fn parallel_copy_auto_requires_scope_and_yields_to_overrides() {
    assert!(auto_parallel_copy_a3b_enabled(
        true,
        GgufParallelCopyMode::Auto,
        false,
    ));
    assert!(!auto_parallel_copy_a3b_enabled(
        false,
        GgufParallelCopyMode::Auto,
        false,
    ));
    assert!(!auto_parallel_copy_a3b_enabled(
        true,
        GgufParallelCopyMode::Disabled,
        false,
    ));
    assert!(!auto_parallel_copy_a3b_enabled(
        true,
        GgufParallelCopyMode::ForcedCopy,
        false,
    ));
    assert!(!auto_parallel_copy_a3b_enabled(
        true,
        GgufParallelCopyMode::ForcedPageRoundedCopy,
        false,
    ));
    assert!(!auto_parallel_copy_a3b_enabled(
        true,
        GgufParallelCopyMode::Auto,
        true,
    ));
    assert!(!auto_parallel_copy_a3b_override_present(|_| false));
    for expected in A3B_PARALLEL_COPY_AUTO_OVERRIDE_ENVS {
        assert!(auto_parallel_copy_a3b_override_present(
            |name| name == expected
        ));
    }
    assert!(!MetalModelLoadOptions::default().auto_parallel_copy_a3b);
    assert!(!MetalModelLoadOptions::default().auto_retained_single_pass);
}

#[test]
fn automatic_retained_single_pass_is_general_and_override_safe() {
    assert!(auto_retained_single_pass_enabled(
        true,
        true,
        GgufNoCopyMode::Disabled,
        GgufOwnedArenaMode::Disabled,
        GgufParallelCopyMode::Auto,
        GgufNoCopyPrefaultMode::Default,
        false,
    ));
    let rejected = [
        auto_retained_single_pass_enabled(
            false,
            true,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            GgufParallelCopyMode::Auto,
            GgufNoCopyPrefaultMode::Default,
            false,
        ),
        auto_retained_single_pass_enabled(
            true,
            false,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            GgufParallelCopyMode::Auto,
            GgufNoCopyPrefaultMode::Default,
            false,
        ),
        auto_retained_single_pass_enabled(
            true,
            true,
            GgufNoCopyMode::Forced,
            GgufOwnedArenaMode::Disabled,
            GgufParallelCopyMode::Auto,
            GgufNoCopyPrefaultMode::Default,
            false,
        ),
        auto_retained_single_pass_enabled(
            true,
            true,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Forced,
            GgufParallelCopyMode::Auto,
            GgufNoCopyPrefaultMode::Default,
            false,
        ),
        auto_retained_single_pass_enabled(
            true,
            true,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            GgufParallelCopyMode::Disabled,
            GgufNoCopyPrefaultMode::Default,
            false,
        ),
        auto_retained_single_pass_enabled(
            true,
            true,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            GgufParallelCopyMode::Auto,
            GgufNoCopyPrefaultMode::Enabled,
            false,
        ),
        auto_retained_single_pass_enabled(
            true,
            true,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            GgufParallelCopyMode::Auto,
            GgufNoCopyPrefaultMode::Default,
            true,
        ),
    ];
    assert!(rejected.into_iter().all(|selected| !selected));

    let planner_called = std::cell::Cell::new(false);
    let skipped = prepare_auto_retained_selection(true, true, || {
        planner_called.set(true);
        unreachable!()
    });
    assert!(matches!(
        skipped,
        PreparedAutoRetainedSelection::NotEligible
    ));
    assert!(!planner_called.get());

    let fallback = prepare_auto_retained_selection(false, true, || {
        Err(MfError::LoadPolicy("unsupported fixture".into()))
    });
    assert!(matches!(
        fallback,
        PreparedAutoRetainedSelection::NoMatch(reason)
            if reason.contains("unsupported fixture")
    ));

    let selected = prepare_auto_retained_selection(false, true, || {
        Ok(RetainedStoragePlan {
            page_size: 16_384,
            max_buffer_length: 1 << 30,
            usable_window_length: 1 << 30,
            required_alignment: GGUF_NO_COPY_ALIGNMENT,
            windows: Vec::new(),
            entries: Vec::new(),
            unique_view_bytes: 0,
            logical_view_bytes: 0,
            unique_fallback_bytes: 0,
            alias_bytes: 0,
        })
    });
    assert!(matches!(
        selected,
        PreparedAutoRetainedSelection::Selected(_)
    ));
}

#[test]
#[ignore = "requires local Qwen3.6-27B BF16 fixture"]
fn automatic_retained_single_pass_plans_qwen36_bf16_without_realizing_weights() {
    let path = "/Volumes/wdblack/weights-archive/qwen3.6-27b-bf16/Qwen3.6-27B-BF16.gguf";
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(path).expect("open Qwen3.6 BF16 fixture");
    let model = Model::from_gguf(&gguf).expect("bind Qwen3.6 BF16 fixture");
    let prepared = MetalModel::prepare_load_with_options(
        &ctx,
        &gguf,
        &model,
        MetalModelLoadOptions {
            auto_parallel_copy_a3b: false,
            auto_retained_single_pass: true,
        },
    )
    .expect("prepare BF16 single-pass load");
    let PreparedAutoRetainedSelection::Selected(plan) = &prepared.auto_retained else {
        panic!("Qwen3.6 BF16 did not select a retained single-pass plan")
    };
    assert!(!plan.windows.is_empty());
    assert!(plan.unique_view_bytes > 0);
    assert!(plan.entries.iter().all(|entry| !matches!(
        entry.disposition,
        RetainedStorageDisposition::CopyFallback { reason }
            if reason != RetainedStorageFallback::FinalPartialPage
    )));
    assert_eq!(
        prepared.prefetch_advice(),
        MetalLoadPrefetchAdvice::PreserveConfiguredPolicy
    );
}

#[test]
fn parallel_copy_auto_population_is_exact_and_narrow() {
    assert_eq!(
        auto_parallel_copy_population(ParallelCopyProfileId::A3bQ4kmV1),
        Some(ParallelPopulationMethod::Pread)
    );
    assert_eq!(
        auto_parallel_copy_population(ParallelCopyProfileId::A10bQ4xlV1),
        None
    );
    assert_eq!(
        auto_parallel_copy_population(ParallelCopyProfileId::Dense27bQ4kmV1),
        None
    );
}

#[test]
fn gguf_parallel_pread_capability_table_is_exact() {
    let observed = PARALLEL_COPY_PROFILES
        .iter()
        .map(|profile| (profile.id, profile.supports_direct_pread))
        .collect::<Vec<_>>();
    assert_eq!(
        observed,
        vec![
            (ParallelCopyProfileId::A3bQ4kmV1, true),
            (ParallelCopyProfileId::A10bQ4xlV1, true),
            (ParallelCopyProfileId::Dense27bQ4kmV1, true),
        ]
    );
    for profile in PARALLEL_COPY_PROFILES {
        if profile.id == ParallelCopyProfileId::A10bQ4xlV1 {
            assert!(
                validate_parallel_population(profile, ParallelPopulationMethod::MmapCopy).is_err()
            );
        } else {
            validate_parallel_population(profile, ParallelPopulationMethod::MmapCopy)
                .expect("mmap population capability");
        }
        validate_parallel_population(profile, ParallelPopulationMethod::Pread)
            .expect("pread population capability");
    }
}

#[test]
fn gguf_parallel_pread_dense_marker_table_is_exact() {
    assert_eq!(
        parallel_copy_marker_label(
            &DENSE27B_PARALLEL_COPY_PROFILE,
            ParallelPopulationMethod::MmapCopy,
            ParallelDestinationLength::LogicalExact,
        )
        .unwrap(),
        "[metal-gguf-parallel-copied]"
    );
    assert_eq!(
        parallel_copy_marker_label(
            &DENSE27B_PARALLEL_COPY_PROFILE,
            ParallelPopulationMethod::Pread,
            ParallelDestinationLength::LogicalExact,
        )
        .unwrap(),
        "[metal-gguf-parallel-pread]"
    );
}

#[test]
fn page_rounded_parallel_copy_contract_is_narrow_and_checked() {
    assert_eq!(
        GgufParallelCopyMode::ForcedCopy.forced_configuration(),
        Some((
            ParallelPopulationMethod::MmapCopy,
            ParallelDestinationLength::LogicalExact,
        ))
    );
    assert_eq!(
        GgufParallelCopyMode::ForcedPread.forced_configuration(),
        Some((
            ParallelPopulationMethod::Pread,
            ParallelDestinationLength::LogicalExact,
        ))
    );
    assert_eq!(
        GgufParallelCopyMode::ForcedPageRoundedCopy.forced_configuration(),
        Some((
            ParallelPopulationMethod::MmapCopy,
            ParallelDestinationLength::PageRounded16K,
        ))
    );
    assert!(GgufParallelCopyMode::Auto.forced_configuration().is_none());
    assert!(
        GgufParallelCopyMode::Disabled
            .forced_configuration()
            .is_none()
    );

    assert!(
        validate_parallel_destination_length(
            &A3B_PARALLEL_COPY_PROFILE,
            ParallelPopulationMethod::MmapCopy,
            ParallelDestinationLength::PageRounded16K,
        )
        .is_ok()
    );
    assert!(
        validate_parallel_destination_length(
            &A3B_PARALLEL_COPY_PROFILE,
            ParallelPopulationMethod::Pread,
            ParallelDestinationLength::PageRounded16K,
        )
        .is_err()
    );
    assert!(
        validate_parallel_destination_length(
            &DENSE27B_PARALLEL_COPY_PROFILE,
            ParallelPopulationMethod::MmapCopy,
            ParallelDestinationLength::PageRounded16K,
        )
        .is_err()
    );
    assert_eq!(
        parallel_copy_marker_label(
            &A3B_PARALLEL_COPY_PROFILE,
            ParallelPopulationMethod::MmapCopy,
            ParallelDestinationLength::PageRounded16K,
        )
        .unwrap(),
        "[metal-gguf-parallel-page-rounded]"
    );

    assert_eq!(
        parallel_destination_resource_length(4, ParallelDestinationLength::PageRounded16K, 16_384,)
            .unwrap(),
        16_384
    );
    assert_eq!(
        parallel_destination_resource_length(
            16_384,
            ParallelDestinationLength::PageRounded16K,
            16_384,
        )
        .unwrap(),
        16_384
    );
    assert_eq!(
        parallel_destination_resource_length(
            16_385,
            ParallelDestinationLength::PageRounded16K,
            32_768,
        )
        .unwrap(),
        32_768
    );
    assert!(
        parallel_destination_resource_length(
            16_385,
            ParallelDestinationLength::PageRounded16K,
            32_767,
        )
        .is_err()
    );
    assert!(
        parallel_destination_resource_length(
            usize::MAX as u64,
            ParallelDestinationLength::PageRounded16K,
            usize::MAX,
        )
        .is_err()
    );
    assert!(
        parallel_destination_resource_length(
            0,
            ParallelDestinationLength::LogicalExact,
            usize::MAX,
        )
        .is_err()
    );

    let identity = |index: usize, source_bytes: u64| ModelWeightStorageIdentity {
        name: format!("weight.{index}"),
        shard_idx: 0,
        data_offset: index as u64 * 65_536,
        source_bytes,
        dtype: GgmlType::F32,
        shape: vec![source_bytes / 4],
        kind: ModelWeightStorageKind::Direct,
        resident_bytes: source_bytes,
    };
    let identities = [identity(0, 4), identity(1, 16_384), identity(2, 16_388)];
    assert_eq!(
        parallel_destination_accounting(
            &identities,
            ParallelDestinationLength::LogicalExact,
            usize::MAX,
        )
        .unwrap(),
        (32_776, 0)
    );
    assert_eq!(
        parallel_destination_accounting(
            &identities,
            ParallelDestinationLength::PageRounded16K,
            usize::MAX,
        )
        .unwrap(),
        (65_536, 2)
    );
}

#[test]
fn gguf_parallel_pread_non_a3b_auto_remains_none() {
    assert_eq!(
        auto_parallel_copy_population(ParallelCopyProfileId::Dense27bQ4kmV1),
        None
    );
    assert_eq!(
        auto_parallel_copy_population(ParallelCopyProfileId::A10bQ4xlV1),
        None
    );
    assert_eq!(
        auto_parallel_copy_population(ParallelCopyProfileId::A3bQ4kmV1),
        Some(ParallelPopulationMethod::Pread)
    );
}

#[test]
fn gguf_parallel_pread_a10b_profile_is_exact_and_force_only() {
    let profile = &A10B_PARALLEL_PREAD_PROFILE;
    assert_eq!(profile.id.label(), "a10b-q4xl-v1");
    assert_eq!(profile.architecture_label, Some("qwen35moe"));
    assert_eq!(profile.arch, A10B_PARALLEL_COPY_ARCH);
    assert_eq!(
        profile.shard_mapped_lengths,
        [10_943_552, 49_640_779_424, 27_378_273_056]
    );
    assert_eq!(profile.descriptor_layout_digest, 0x3eb2_9091_5bec_2041);
    assert_eq!(
        profile.inventory_digest,
        "b331c475123dbee3bc862a495266dee3996c5f3adabcd6fbeaff9bbabd71a4f8"
    );
    assert_eq!(profile.embedding_dtype, GgmlType::Q8_0);
    assert_eq!(profile.embedding_shape, [3072, 248_320]);
    assert_eq!(profile.request_count, 879);
    assert_eq!(profile.source_bytes, 77_018_996_736);
    assert_eq!(profile.cuts, [214, 435, 658]);
    assert_eq!(profile.task_counts, [214, 221, 223, 221]);
    assert_eq!(
        profile.worker_bytes,
        [
            19_474_295_808,
            19_228_744_704,
            19_231_902_720,
            19_084_053_504
        ]
    );
    assert!(validate_parallel_population(profile, ParallelPopulationMethod::Pread).is_ok());
    assert!(validate_parallel_population(profile, ParallelPopulationMethod::MmapCopy).is_err());
    assert_eq!(
        parallel_copy_marker_label(
            profile,
            ParallelPopulationMethod::Pread,
            ParallelDestinationLength::LogicalExact,
        )
        .unwrap(),
        "[metal-gguf-parallel-pread]"
    );
}

#[test]
fn gguf_parallel_pread_dense_advice_preserves_configured_policy() {
    let selected = PreparedAutoSelection::Selected(PreparedParallelCopiedProfile {
        profile: &DENSE27B_PARALLEL_COPY_PROFILE,
        population: ParallelPopulationMethod::Pread,
        destination_length: ParallelDestinationLength::LogicalExact,
        expected_identities: Vec::new(),
        sorted_request_indices: Vec::new(),
        _proof: PreparedParallelCopyProof::DensePlannerFree,
    });
    assert_eq!(
        selected.prefetch_advice(),
        MetalLoadPrefetchAdvice::PreserveConfiguredPolicy
    );
    assert_eq!(
        PreparedAutoSelection::NotEligible.prefetch_advice(),
        MetalLoadPrefetchAdvice::PreserveConfiguredPolicy
    );
}

#[test]
fn prepared_auto_prefetch_advice_selector_table_is_fail_closed() {
    #[derive(Clone, Copy)]
    enum Selection {
        A3bPread,
        A3bCopy,
        A3bPageRoundedCopy,
        A3bPreadWrongProof,
        DenseCopy,
        NoMatch,
    }

    let selected = |profile: &'static ParallelCopyProfile,
                    population: ParallelPopulationMethod,
                    destination_length: ParallelDestinationLength,
                    proof: PreparedParallelCopyProof| {
        PreparedAutoSelection::Selected(PreparedParallelCopiedProfile {
            profile,
            population,
            destination_length,
            expected_identities: Vec::new(),
            sorted_request_indices: Vec::new(),
            _proof: proof,
        })
    };
    let selection = |case| match case {
        Selection::A3bPread => selected(
            &A3B_PARALLEL_COPY_PROFILE,
            ParallelPopulationMethod::Pread,
            ParallelDestinationLength::LogicalExact,
            PreparedParallelCopyProof::A3bRetainedPlan,
        ),
        Selection::A3bCopy => selected(
            &A3B_PARALLEL_COPY_PROFILE,
            ParallelPopulationMethod::MmapCopy,
            ParallelDestinationLength::LogicalExact,
            PreparedParallelCopyProof::A3bRetainedPlan,
        ),
        Selection::A3bPageRoundedCopy => selected(
            &A3B_PARALLEL_COPY_PROFILE,
            ParallelPopulationMethod::MmapCopy,
            ParallelDestinationLength::PageRounded16K,
            PreparedParallelCopyProof::A3bRetainedPlan,
        ),
        Selection::A3bPreadWrongProof => selected(
            &A3B_PARALLEL_COPY_PROFILE,
            ParallelPopulationMethod::Pread,
            ParallelDestinationLength::LogicalExact,
            PreparedParallelCopyProof::DensePlannerFree,
        ),
        Selection::DenseCopy => selected(
            &DENSE27B_PARALLEL_COPY_PROFILE,
            ParallelPopulationMethod::MmapCopy,
            ParallelDestinationLength::LogicalExact,
            PreparedParallelCopyProof::DensePlannerFree,
        ),
        Selection::NoMatch => PreparedAutoSelection::NoMatch,
    };
    let cases = [
        (
            "disposable-auto-authenticated-a3b-pread",
            true,
            GgufParallelCopyMode::Auto,
            false,
            Selection::A3bPread,
            MetalLoadPrefetchAdvice::SuppressColdOnlyAuthenticatedA3bDirectPread,
        ),
        (
            "force-only",
            false,
            GgufParallelCopyMode::Auto,
            false,
            Selection::A3bPread,
            MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
        ),
        (
            "forced-copy",
            true,
            GgufParallelCopyMode::ForcedCopy,
            false,
            Selection::A3bPread,
            MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
        ),
        (
            "forced-pread",
            true,
            GgufParallelCopyMode::ForcedPread,
            false,
            Selection::A3bPread,
            MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
        ),
        (
            "parallel-copy-disabled",
            true,
            GgufParallelCopyMode::Disabled,
            false,
            Selection::A3bPread,
            MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
        ),
        (
            "explicit-override",
            true,
            GgufParallelCopyMode::Auto,
            true,
            Selection::A3bPread,
            MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
        ),
        (
            "no-profile-match",
            true,
            GgufParallelCopyMode::Auto,
            false,
            Selection::NoMatch,
            MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
        ),
        (
            "mmap-copy-population",
            true,
            GgufParallelCopyMode::Auto,
            false,
            Selection::A3bCopy,
            MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
        ),
        (
            "page-rounded-mmap-copy-population",
            true,
            GgufParallelCopyMode::Auto,
            false,
            Selection::A3bPageRoundedCopy,
            MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
        ),
        (
            "authentication-proof-mismatch",
            true,
            GgufParallelCopyMode::Auto,
            false,
            Selection::A3bPreadWrongProof,
            MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
        ),
        (
            "other-profile",
            true,
            GgufParallelCopyMode::Auto,
            false,
            Selection::DenseCopy,
            MetalLoadPrefetchAdvice::PreserveConfiguredPolicy,
        ),
    ];

    for (name, admission, mode, override_present, selected, expected) in cases {
        let auto = if auto_parallel_copy_a3b_enabled(admission, mode, override_present) {
            selection(selected)
        } else {
            PreparedAutoSelection::NotEligible
        };
        assert_eq!(auto.prefetch_advice(), expected, "{name}");
    }
}

#[test]
fn parallel_copy_auto_host_gate_is_exact() {
    assert!(a3b_parallel_copy_auto_host_supported(
        true,
        "Apple M4 Max",
        Some(128 * 1024 * 1024 * 1024),
    ));
    assert!(a3b_parallel_copy_auto_host_supported(
        true,
        "Apple M4 Max",
        Some(192 * 1024 * 1024 * 1024),
    ));
    assert!(!a3b_parallel_copy_auto_host_supported(
        false,
        "Apple M4 Max",
        Some(128 * 1024 * 1024 * 1024),
    ));
    assert!(!a3b_parallel_copy_auto_host_supported(
        true,
        "Apple M4 Pro",
        Some(128 * 1024 * 1024 * 1024),
    ));
    assert!(!a3b_parallel_copy_auto_host_supported(
        true,
        "Apple M4 Max",
        Some(128 * 1024 * 1024 * 1024 - 1),
    ));
    assert!(!a3b_parallel_copy_auto_host_supported(
        true,
        "Apple M4 Max",
        None,
    ));
}

#[test]
fn parallel_copy_profile_selection_is_exactly_one() {
    let selected = select_unique_parallel_copy_profile(&PARALLEL_COPY_PROFILES, |profile| {
        Ok(profile.id == ParallelCopyProfileId::Dense27bQ4kmV1)
    })
    .unwrap();
    assert_eq!(selected.id, ParallelCopyProfileId::Dense27bQ4kmV1);

    assert!(select_unique_parallel_copy_profile(&PARALLEL_COPY_PROFILES, |_| Ok(false)).is_err());
    assert!(select_unique_parallel_copy_profile(&PARALLEL_COPY_PROFILES, |_| Ok(true)).is_err());
    assert!(
        select_unique_parallel_copy_profile(
            &[&A3B_PARALLEL_COPY_PROFILE, &A3B_PARALLEL_COPY_PROFILE],
            |_| Ok(true),
        )
        .is_err()
    );
}

#[test]
fn owned_arena_worker_boundaries_cover_nondivisible_page_counts() {
    let page = 16_384;
    let boundaries = owned_arena_four_worker_boundaries(11 * page, page).expect("valid boundaries");
    assert_eq!(boundaries, [0, 2 * page, 5 * page, 8 * page, 11 * page]);
    assert!(owned_arena_four_worker_boundaries(3 * page, page).is_err());
    assert!(owned_arena_four_worker_boundaries(4 * page + 1, page).is_err());
}

#[test]
fn model_weight_request_sequence_rejects_equal_aggregate_drift() {
    let direct_desc = TensorDesc {
        name: "direct".to_string(),
        shape: vec![8],
        dtype: GgmlType::F32,
        shard_idx: 0,
        data_offset: 32,
        n_bytes: 32,
    };
    let converted_desc = TensorDesc {
        name: "converted".to_string(),
        shape: vec![8],
        dtype: GgmlType::F32,
        shard_idx: 0,
        data_offset: 64,
        n_bytes: 32,
    };
    let expected = vec![
        ModelWeightStorageRequest {
            desc: &direct_desc,
            kind: ModelWeightStorageKind::Direct,
            resident_bytes: 32,
        },
        ModelWeightStorageRequest {
            desc: &converted_desc,
            kind: ModelWeightStorageKind::ConvertedF32,
            resident_bytes: 32,
        },
    ];
    let actual = expected
        .iter()
        .map(expected_model_weight_identity)
        .collect::<Vec<_>>();
    validate_model_weight_request_sequence(&actual, &expected).unwrap();

    let mut reordered = actual.clone();
    reordered.swap(0, 1);
    assert!(validate_model_weight_request_sequence(&reordered, &expected).is_err());

    let mut swapped_kinds = actual.clone();
    swapped_kinds[0].kind = ModelWeightStorageKind::ConvertedF32;
    swapped_kinds[1].kind = ModelWeightStorageKind::Direct;
    assert!(validate_model_weight_request_sequence(&swapped_kinds, &expected).is_err());

    let mut duplicated = actual;
    duplicated[0] = duplicated[1].clone();
    assert!(validate_model_weight_request_sequence(&duplicated, &expected).is_err());
}

struct GenericRetainedArmResult {
    prefill_logits: Vec<f32>,
    prefill_snapshot: SessionSnapshot,
    next_token: i32,
    decode_logits: Vec<f32>,
    decode_snapshot: SessionSnapshot,
}

#[derive(Clone, Copy)]
struct GenericRetainedContract {
    windows: usize,
    window_bytes: u64,
    direct: usize,
    views: usize,
    view_bytes: u64,
    aliases: usize,
    alias_bytes: u64,
    fallbacks: usize,
    fallback_bytes: u64,
}

fn assert_generic_retained_contract(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    native_embedding: bool,
    contract: GenericRetainedContract,
) {
    let expected = model_weight_storage_requests(model, native_embedding, false)
        .expect("generic fixture storage requests");
    let direct = expected
        .iter()
        .filter(|request| request.kind == ModelWeightStorageKind::Direct)
        .map(|request| request.desc)
        .collect::<Vec<_>>();
    let mut storage = planned_retained_storage_for_load(ctx, gguf, &expected, false)
        .expect("generic fixture retained storage");
    let window_bytes = storage
        .plan
        .windows
        .iter()
        .map(|window| window.length as u64)
        .sum::<u64>();
    let views = storage
        .plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
        .count();
    let aliases = storage
        .plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::Alias { .. }))
        .count();
    let fallbacks = storage
        .plan
        .entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.disposition,
                RetainedStorageDisposition::CopyFallback { .. }
            )
        })
        .count();
    assert_eq!(storage.plan.windows.len(), contract.windows);
    assert_eq!(window_bytes, contract.window_bytes);
    assert_eq!(storage.plan.entries.len(), contract.direct);
    assert_eq!(views, contract.views);
    assert_eq!(storage.plan.unique_view_bytes, contract.view_bytes);
    assert_eq!(aliases, contract.aliases);
    assert_eq!(storage.plan.alias_bytes, contract.alias_bytes);
    assert_eq!(fallbacks, contract.fallbacks);
    assert_eq!(storage.plan.unique_fallback_bytes, contract.fallback_bytes);

    let mut ledger = WeightLoadLedger::default();
    for desc in &direct {
        let (tensor, materialization) = storage
            .load_direct(ctx, gguf, desc)
            .expect("materialize generic fixture tensor");
        ledger
            .record_source(desc, materialization, tensor.n_bytes())
            .expect("record generic fixture tensor");
    }
    storage
        .validate_complete(&ledger)
        .expect("complete generic fixture realization");
    for (index, entry) in storage.plan.entries.iter().enumerate() {
        if let RetainedStorageDisposition::Alias {
            source_request_index,
        } = entry.disposition
        {
            let source = storage.realized[source_request_index]
                .as_ref()
                .expect("realized alias source");
            let alias = storage.realized[index]
                .as_ref()
                .expect("realized alias tensor");
            assert_eq!(source.offset, alias.offset);
            assert_eq!(
                Retained::as_ptr(&source.buffer),
                Retained::as_ptr(&alias.buffer)
            );
        }
    }
}

fn run_generic_retained_arm(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    tokens: &[i32],
    mode: GgufNoCopyMode,
    owned_mode: GgufOwnedArenaMode,
    forced_next: Option<i32>,
) -> GenericRetainedArmResult {
    let metal_model = MetalModel::load_with_storage_policy(
        ctx,
        gguf,
        model,
        mode,
        false,
        owned_mode,
        GgufParallelCopyMode::Disabled,
        false,
    )
    .expect("load generic storage arm");
    run_loaded_model_arm(ctx, metal_model, tokens, forced_next)
}

fn run_loaded_model_arm(
    ctx: &MetalContext,
    metal_model: MetalModel,
    tokens: &[i32],
    forced_next: Option<i32>,
) -> GenericRetainedArmResult {
    use crate::metal_dflash::{
        MetalDFlashLayerMajorScratch, PrefillScratchConfig,
        plan_prefill_scratch_with_matrix_max_pos_configured, prefill_tokens_with_multi_hidden,
    };

    let forward = MetalForward::new(ctx, &metal_model);
    let capacity = 64;
    let mut session =
        MetalSession::fresh(ctx, &metal_model, capacity).expect("fresh generic session");
    let plan = plan_prefill_scratch_with_matrix_max_pos_configured(
        &metal_model,
        tokens.len() as u32,
        capacity,
        PrefillScratchConfig::default(),
    )
    .expect("generic prefill scratch plan");
    let mut scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(ctx, &metal_model, plan)
            .expect("generic prefill scratch");
    let prefill_logits = prefill_tokens_with_multi_hidden(
        &forward,
        tokens,
        0,
        &mut session,
        &mut scratch,
        &[],
        None,
    )
    .expect("generic packed prefill");
    let identity = session.snapshot_identity(0x594, 0x594);
    let prefill_snapshot = session
        .snapshot(
            identity.clone(),
            tokens.to_vec(),
            Some(prefill_logits.clone()),
        )
        .expect("generic prefill snapshot");
    let next_token = forced_next.unwrap_or_else(|| {
        prefill_logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                if value > best.1 { (index, value) } else { best }
            })
            .0 as i32
    });
    let decode_logits = forward
        .single_token(next_token, tokens.len() as u32, &mut session)
        .expect("generic forced decode transition");
    let mut consumed = tokens.to_vec();
    consumed.push(next_token);
    let decode_snapshot = session
        .snapshot(identity, consumed, Some(decode_logits.clone()))
        .expect("generic decode snapshot");
    GenericRetainedArmResult {
        prefill_logits,
        prefill_snapshot,
        next_token,
        decode_logits,
        decode_snapshot,
    }
}

fn assert_generic_retained_f32_bits(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label} length");
    for (index, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{label} bit mismatch at {index}");
    }
}

fn assert_generic_retained_snapshot(label: &str, a: &SessionSnapshot, b: &SessionSnapshot) {
    assert_eq!(a.identity, b.identity, "{label} identity");
    assert_eq!(a.prefix_tokens, b.prefix_tokens, "{label} tokens");
    assert_eq!(a.kv_n_pos, b.kv_n_pos, "{label} KV positions");
    assert_eq!(a.kv_k_arena, b.kv_k_arena, "{label} K arena");
    assert_eq!(a.kv_v_arena, b.kv_v_arena, "{label} V arena");
    assert_eq!(a.gdn_conv_arena, b.gdn_conv_arena, "{label} conv arena");
    assert_eq!(a.gdn_state_arena, b.gdn_state_arena, "{label} state arena");
    match (&a.final_logits, &b.final_logits) {
        (Some(a), Some(b)) => assert_generic_retained_f32_bits(&format!("{label} logits"), a, b),
        (None, None) => {}
        _ => panic!("{label} final-logits presence mismatch"),
    }
}

fn assert_generic_retained_model_exact(
    model_path: &str,
    expect_tied: bool,
    embedding_mode: NativeQuantEmbeddingMode,
    native_embedding: bool,
    contract: GenericRetainedContract,
) {
    assert_eq!(
        native_quant_embedding_mode(),
        embedding_mode,
        "native embedding environment does not match the fixture contract"
    );
    assert!(
        std::path::Path::new(model_path).is_file(),
        "missing generic retained fixture {model_path}"
    );
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(model_path).expect("open generic retained fixture");
    let model = Model::from_gguf(&gguf).expect("bind generic retained fixture");
    assert_eq!(model.tied_embeddings, expect_tied);
    assert!(!matches_no_copy_27b_sentinel(&gguf, &model));
    assert_generic_retained_contract(&ctx, &gguf, &model, native_embedding, contract);
    let tokenizer = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
    let tokens = tokenizer
        .encode("Retained storage must preserve this state.", true)
        .expect("tokenize generic retained prompt");
    assert!(tokens.len() < 64);

    let copied = run_generic_retained_arm(
        &ctx,
        &gguf,
        &model,
        &tokens,
        GgufNoCopyMode::Disabled,
        GgufOwnedArenaMode::Disabled,
        None,
    );
    let retained = run_generic_retained_arm(
        &ctx,
        &gguf,
        &model,
        &tokens,
        GgufNoCopyMode::Forced,
        GgufOwnedArenaMode::Disabled,
        Some(copied.next_token),
    );
    let retained_argmax = retained
        .prefill_logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 { (index, value) } else { best }
        })
        .0 as i32;
    assert_eq!(copied.next_token, retained_argmax);
    assert_eq!(retained.next_token, retained_argmax);
    assert_generic_retained_f32_bits(
        "generic prefill logits",
        &copied.prefill_logits,
        &retained.prefill_logits,
    );
    assert_generic_retained_snapshot(
        "generic prefill snapshot",
        &copied.prefill_snapshot,
        &retained.prefill_snapshot,
    );
    assert_generic_retained_f32_bits(
        "generic decode logits",
        &copied.decode_logits,
        &retained.decode_logits,
    );
    assert_generic_retained_snapshot(
        "generic decode snapshot",
        &copied.decode_snapshot,
        &retained.decode_snapshot,
    );
}

#[test]
#[ignore = "requires local tied 0.8B Q8_0 fixture and explicit native embedding"]
fn gguf_no_copy_generic_tied_q8_is_bit_exact() {
    assert_generic_retained_model_exact(
        "/Users/tito/models/Qwen3.5-0.8B-Q8_0.gguf",
        true,
        NativeQuantEmbeddingMode::Forced,
        true,
        GenericRetainedContract {
            windows: 1,
            window_bytes: 800_882_688,
            direct: 321,
            views: 319,
            view_bytes: 800_877_824,
            aliases: 1,
            alias_bytes: 270_172_160,
            fallbacks: 1,
            fallback_bytes: 4_096,
        },
    );
}

#[test]
#[ignore = "requires local tied 0.8B Q8_0 fixture and no native embedding override"]
fn gguf_no_copy_generic_tied_q8_converted_embedding_is_bit_exact() {
    assert_generic_retained_model_exact(
        "/Users/tito/models/Qwen3.5-0.8B-Q8_0.gguf",
        true,
        NativeQuantEmbeddingMode::Auto,
        false,
        GenericRetainedContract {
            windows: 1,
            window_bytes: 800_882_688,
            direct: 320,
            views: 319,
            view_bytes: 800_877_824,
            aliases: 0,
            alias_bytes: 0,
            fallbacks: 1,
            fallback_bytes: 4_096,
        },
    );
}

#[test]
#[ignore = "requires local Qwen3.6 A3B Q4 fixture and explicit native embedding"]
fn gguf_no_copy_generic_a3b_q4_is_bit_exact() {
    assert_generic_retained_model_exact(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        false,
        NativeQuantEmbeddingMode::Forced,
        true,
        GenericRetainedContract {
            windows: 1,
            window_bytes: 22_123_544_576,
            direct: 733,
            views: 732,
            view_bytes: 22_123_530_752,
            aliases: 0,
            alias_bytes: 0,
            fallbacks: 1,
            fallback_bytes: 8_192,
        },
    );
}

#[test]
#[ignore = "requires local Qwen3.6 A3B Q4 fixture"]
fn gguf_owned_arena_a3b_q4_is_bit_exact() {
    let model_path = crate::test_fixtures::A3B_Q4_K_M.path();
    assert_eq!(
        native_quant_embedding_mode(),
        NativeQuantEmbeddingMode::Auto,
        "owned fixture requires production-auto embedding selection"
    );
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(model_path).expect("open owned A3B fixture");
    let model = Model::from_gguf(&gguf).expect("bind owned A3B fixture");
    let expected =
        model_weight_storage_requests(&model, true, false).expect("owned A3B storage requests");
    let direct = expected
        .iter()
        .map(|request| request.desc)
        .collect::<Vec<_>>();
    let mut storage = planned_owned_storage_for_load(
        &ctx,
        &gguf,
        &model,
        &expected,
        NativeQuantEmbeddingSelection::AutoPromoted,
    )
    .expect("realize owned A3B storage");
    assert_eq!(storage.resources.len(), 2);
    assert_ne!(
        Retained::as_ptr(&storage.resources[0]),
        Retained::as_ptr(&storage.resources[1])
    );
    let mut ledger = WeightLoadLedger::default();
    for desc in &direct {
        let (tensor, materialization) = storage
            .load_direct(desc)
            .expect("materialize owned A3B tensor");
        assert_eq!(
            tensor.provenance(),
            MetalTensorProvenance::OwnedWeightReadOnly
        );
        assert!(!tensor.is_writable());
        ledger
            .record_source(desc, materialization, tensor.n_bytes())
            .expect("record owned A3B tensor");
    }
    storage
        .validate_complete(&ledger)
        .expect("complete owned A3B realization");
    for (index, entry) in storage.plan.entries.iter().enumerate() {
        let tensor = storage.realized[index]
            .as_ref()
            .expect("realized owned tensor");
        let resource_index = match entry.disposition {
            RetainedStorageDisposition::View { window_index, .. } => window_index,
            RetainedStorageDisposition::CopyFallback { .. } => 1,
            RetainedStorageDisposition::Alias { .. } => {
                panic!("owned A3B sentinel must not contain aliases")
            }
        };
        assert_eq!(
            Retained::as_ptr(&tensor.buffer),
            Retained::as_ptr(&storage.resources[resource_index])
        );
    }
    drop(storage);

    let tokenizer = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
    let tokens = tokenizer
        .encode("Owned storage must preserve this state.", true)
        .expect("tokenize owned prompt");
    assert!(tokens.len() < 64);
    let copied = run_generic_retained_arm(
        &ctx,
        &gguf,
        &model,
        &tokens,
        GgufNoCopyMode::Disabled,
        GgufOwnedArenaMode::Disabled,
        None,
    );
    let owned = run_generic_retained_arm(
        &ctx,
        &gguf,
        &model,
        &tokens,
        GgufNoCopyMode::Disabled,
        GgufOwnedArenaMode::Forced,
        Some(copied.next_token),
    );
    let owned_argmax = owned
        .prefill_logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 { (index, value) } else { best }
        })
        .0 as i32;
    assert_eq!(copied.next_token, owned_argmax);
    assert_eq!(owned.next_token, owned_argmax);
    assert_generic_retained_f32_bits(
        "owned prefill logits",
        &copied.prefill_logits,
        &owned.prefill_logits,
    );
    assert_generic_retained_snapshot(
        "owned prefill snapshot",
        &copied.prefill_snapshot,
        &owned.prefill_snapshot,
    );
    assert_generic_retained_f32_bits(
        "owned decode logits",
        &copied.decode_logits,
        &owned.decode_logits,
    );
    assert_generic_retained_snapshot(
        "owned decode snapshot",
        &copied.decode_snapshot,
        &owned.decode_snapshot,
    );
}

fn assert_gguf_parallel_a3b_q4_is_bit_exact(
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
    marker: &str,
) {
    let model_path = crate::test_fixtures::A3B_Q4_K_M.path();
    assert_eq!(
        native_quant_embedding_mode(),
        NativeQuantEmbeddingMode::Auto,
        "parallel-copy fixture requires production-auto embedding selection"
    );
    assert!(!moe_router_f16_enabled());
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(model_path).expect("open parallel-copy A3B fixture");
    let model = Model::from_gguf(&gguf).expect("bind parallel-copy A3B fixture");
    let tokens = [
        7734, 264, 12654, 709, 310, 12204, 279, 76938, 8240, 5199, 7638, 13,
    ];

    let ((reference, parallel), load_lines) = capture_metal_load_lines(|| {
        let run_parallel = |population, destination_length, expected_next_token| {
            let embedding_selection = NativeQuantEmbeddingSelection::AutoPromoted;
            emit_native_quant_embedding_policy(&model, embedding_selection);
            let expected = model_weight_storage_requests(&model, true, false)
                .expect("parallel-copy A3B storage requests");
            let storage = planned_parallel_copied_storage_for_load(
                &ctx,
                &gguf,
                &model,
                &expected,
                embedding_selection,
                population,
                destination_length,
            )
            .expect("realize parallel A3B storage");
            storage
                .validate_source_bytes(&gguf, &expected)
                .expect("audit every parallel logical source byte");
            validate_parallel_copied_topology(
                storage.profile,
                storage.destination_length,
                &storage.expected,
                &storage.resources,
                &storage.tensors,
            )
            .expect("validate parallel topology before model construction");
            assert_eq!(
                frozen_parallel_copy_order(storage.profile, &storage.expected)
                    .expect("frozen schedule"),
                storage.sorted_request_indices
            );
            assert_eq!(storage.destination_length, destination_length);
            let (allocated_bytes, padded_resources) = parallel_destination_accounting(
                &storage.expected,
                destination_length,
                ctx.max_buffer_length(),
            )
            .expect("parallel destination accounting");
            match destination_length {
                ParallelDestinationLength::LogicalExact => {
                    assert_eq!(allocated_bytes, GGUF_OWNED_A3B_SOURCE_BYTES);
                    assert_eq!(padded_resources, 0);
                }
                ParallelDestinationLength::PageRounded16K => {
                    assert_eq!(allocated_bytes, GGUF_PAGE_ROUNDED_A3B_ALLOCATED_BYTES);
                    assert_eq!(padded_resources, GGUF_PAGE_ROUNDED_A3B_PADDED_RESOURCES);
                    assert_eq!(
                        allocated_bytes - GGUF_OWNED_A3B_SOURCE_BYTES,
                        GGUF_PAGE_ROUNDED_A3B_PADDING_BYTES
                    );
                    assert!(
                        storage
                            .resources
                            .iter()
                            .all(|resource| resource.length() % 16_384 == 0)
                    );
                }
            }

            let command = ctx.queue.commandBuffer().expect("write guard command");
            let encoder = crate::metal::KernelEncoder::begin(&command);
            let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                encoder.note_write(&storage.tensors[0]);
            }));
            assert!(
                write_result.is_err(),
                "parallel weights must reject compute writes"
            );
            encoder.end();

            let command = ctx.queue.commandBuffer().expect("blit guard command");
            let blit = crate::metal::BlitEncoder::begin(&command);
            let blit_destination = storage
                .tensors
                .iter()
                .find(|tensor| tensor.dtype == GgmlType::F32 && tensor.n_bytes() <= 1_048_576)
                .expect("small direct F32 candidate weight");
            let writable_source = MetalTensor::zeros_f32(&ctx, blit_destination.shape.clone())
                .expect("equal-sized blit source");
            assert_eq!(writable_source.n_bytes(), blit_destination.n_bytes());
            let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                blit.copy_tensor(&writable_source, blit_destination);
            }));
            assert!(
                write_result.is_err(),
                "parallel weights must reject blit writes"
            );
            blit.end();

            let parallel_model = MetalModel::load_with_direct_storage(
                &ctx,
                &gguf,
                &model,
                ResolvedWeightLoadChoices {
                    embedding_selection,
                    router_f16: false,
                    fused_qkv_g8: false,
                },
                &expected,
                DirectStorage::ForcedParallelCopied(storage),
                false,
            )
            .expect("construct model from audited parallel storage");
            run_loaded_model_arm(&ctx, parallel_model, &tokens, expected_next_token)
        };

        let reference = match destination_length {
            ParallelDestinationLength::LogicalExact => run_generic_retained_arm(
                &ctx,
                &gguf,
                &model,
                &tokens,
                GgufNoCopyMode::Disabled,
                GgufOwnedArenaMode::Disabled,
                None,
            ),
            ParallelDestinationLength::PageRounded16K => run_parallel(
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::LogicalExact,
                None,
            ),
        };
        let parallel = run_parallel(population, destination_length, Some(reference.next_token));
        (reference, parallel)
    });
    let parallel_argmax = parallel
        .prefill_logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 { (index, value) } else { best }
        })
        .0 as i32;
    assert_eq!(reference.next_token, parallel_argmax);
    assert_eq!(parallel.next_token, parallel_argmax);
    assert_generic_retained_f32_bits(
        "parallel prefill logits",
        &reference.prefill_logits,
        &parallel.prefill_logits,
    );
    assert_generic_retained_snapshot(
        "parallel prefill snapshot",
        &reference.prefill_snapshot,
        &parallel.prefill_snapshot,
    );
    assert_generic_retained_f32_bits(
        "parallel decode logits",
        &reference.decode_logits,
        &parallel.decode_logits,
    );
    assert_generic_retained_snapshot(
        "parallel decode snapshot",
        &reference.decode_snapshot,
        &parallel.decode_snapshot,
    );

    let native_line = format!(
        "[metal-load] native quantized token embedding policy: auto-promoted ({:?} {:?})",
        model.token_embd.dtype, model.token_embd.shape
    );
    let ledger_line = concat!(
        "[metal-load-ledger] source=733/22123538944 ",
        "direct_copy=733/22123538944 direct_view=0/0 direct_alias=0/0 ",
        "tail_fallback=0/0 converted=0/0/0 derived=0/0"
    );
    let candidate_marker_index = match destination_length {
        ParallelDestinationLength::LogicalExact => {
            assert_eq!(load_lines.len(), 5, "recognized load-line count");
            assert_eq!(load_lines[0], native_line, "A native policy line");
            assert_eq!(load_lines[1], ledger_line, "A copied ledger line");
            assert_eq!(load_lines[2], native_line, "B native policy line");
            assert_eq!(load_lines[4], ledger_line, "B copied ledger line");
            3
        }
        ParallelDestinationLength::PageRounded16K => {
            assert_eq!(load_lines.len(), 6, "recognized load-line count");
            assert_eq!(load_lines[0], native_line, "A native policy line");
            assert!(
                load_lines[1].starts_with("[metal-gguf-parallel-copied] schema=1 "),
                "A exact-parallel marker"
            );
            assert_eq!(load_lines[2], ledger_line, "A copied ledger line");
            assert_eq!(load_lines[3], native_line, "B native policy line");
            assert_eq!(load_lines[5], ledger_line, "B copied ledger line");
            4
        }
    };
    assert_eq!(
        load_lines
            .iter()
            .filter(|line| line.starts_with(marker))
            .count(),
        1,
        "candidate marker count"
    );
    assert_eq!(
        load_lines
            .iter()
            .filter(|line| *line == ledger_line)
            .count(),
        2,
        "copied ledger count"
    );

    let marker_contract = match destination_length {
        ParallelDestinationLength::LogicalExact => concat!(
            " schema=1 resources=733 bytes=22123538944 ",
            "workers=4 cuts=155,359,539 tasks=155,204,180,194 ",
            "worker_bytes=5532746240,5462315776,5595522304,5532954624 ",
            "first_offsets=10990048,5543736288,11006052064,16601574368 ",
            "last_offsets=5392741344,11004937952,16450579424,22134520800 ",
            "create=shared,default_cache,default observed=shared,default_cache,tracked ",
            "page=16384 alignment=32 max_buffer=77309411328 mapped=22134528992 ",
            "layout=0x5ae645df5cf7d568 ",
            "inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5 ",
            "plan=fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af "
        ),
        ParallelDestinationLength::PageRounded16K => concat!(
            " schema=2 resources=733 logical_bytes=22123538944 ",
            "allocated_bytes=22126297088 padding_bytes=2758144 ",
            "padded_resources=232 workers=4 cuts=155,359,539 ",
            "tasks=155,204,180,194 ",
            "worker_bytes=5532746240,5462315776,5595522304,5532954624 ",
            "first_offsets=10990048,5543736288,11006052064,16601574368 ",
            "last_offsets=5392741344,11004937952,16450579424,22134520800 ",
            "create=shared,default_cache,default observed=shared,default_cache,tracked ",
            "page=16384 alignment=32 max_buffer=77309411328 mapped=22134528992 ",
            "layout=0x5ae645df5cf7d568 ",
            "inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5 ",
            "plan=fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af "
        ),
    };
    let marker_prefix = format!("{marker}{marker_contract}");
    let timing_suffix = load_lines[candidate_marker_index]
        .strip_prefix(&marker_prefix)
        .expect("exact candidate marker prefix and field order");
    let timing_fields = timing_suffix.split(' ').collect::<Vec<_>>();
    assert_eq!(timing_fields.len(), 5, "candidate timing field count");
    let parse_timing = |index: usize, name: &str| {
        let value = timing_fields[index]
            .strip_prefix(name)
            .expect("candidate timing field name");
        assert!(
            !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
            "candidate timing must be unsigned decimal"
        );
        let parsed = value.parse::<u64>().expect("candidate timing value");
        assert_eq!(parsed.to_string(), value, "candidate timing canonical form");
        parsed
    };
    let allocation_us = parse_timing(0, "allocation_us=");
    let source_us = parse_timing(1, "source_us=");
    let copy_us = parse_timing(2, "copy_us=");
    let binding_us = parse_timing(3, "binding_us=");
    let ready_us = parse_timing(4, "ready_us=");
    let phase_us = allocation_us + source_us + copy_us + binding_us;
    assert!(
        ready_us.abs_diff(phase_us) <= 4,
        "candidate timing reconciliation"
    );
}

#[test]
#[ignore = "requires local Qwen3.6 A3B Q4 fixture"]
fn gguf_parallel_copied_a3b_q4_is_bit_exact() {
    assert_gguf_parallel_a3b_q4_is_bit_exact(
        ParallelPopulationMethod::MmapCopy,
        ParallelDestinationLength::LogicalExact,
        "[metal-gguf-parallel-copied]",
    );
}

#[test]
#[ignore = "requires local Qwen3.6 A3B Q4 fixture"]
fn gguf_parallel_pread_a3b_q4_is_bit_exact() {
    assert_gguf_parallel_a3b_q4_is_bit_exact(
        ParallelPopulationMethod::Pread,
        ParallelDestinationLength::LogicalExact,
        "[metal-gguf-parallel-pread]",
    );
}

#[test]
#[ignore = "requires local Qwen3.6 A3B Q4 fixture"]
fn gguf_parallel_page_rounded_a3b_q4_is_bit_exact() {
    assert_gguf_parallel_a3b_q4_is_bit_exact(
        ParallelPopulationMethod::MmapCopy,
        ParallelDestinationLength::PageRounded16K,
        "[metal-gguf-parallel-page-rounded]",
    );
}

fn assert_gguf_parallel_dense27b_q4_is_bit_exact(
    population: ParallelPopulationMethod,
    marker: &str,
) {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    assert_eq!(
        native_quant_embedding_mode(),
        NativeQuantEmbeddingMode::Auto,
        "parallel-copy fixture requires production-auto embedding selection"
    );
    assert!(!moe_router_f16_enabled());
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(model_path).expect("open parallel-copy dense fixture");
    let model = Model::from_gguf(&gguf).expect("bind parallel-copy dense fixture");
    let profile = &DENSE27B_PARALLEL_COPY_PROFILE;
    assert_eq!(gguf.architecture().as_deref(), profile.architecture_label);
    assert_eq!(gguf.shard_mapped_lengths(), profile.shard_mapped_lengths);
    assert_eq!(
        gguf_descriptor_layout_digest(&gguf),
        profile.descriptor_layout_digest
    );
    assert_eq!(model.arch, profile.arch);
    assert_eq!(model.tied_embeddings, profile.tied_embeddings);
    assert_eq!(model.mtp.is_some(), profile.mtp_present);
    assert_eq!(model.token_embd.dtype, profile.embedding_dtype);
    assert_eq!(model.token_embd.shape, profile.embedding_shape);
    assert_eq!(ctx.device.name().to_string(), "Apple M4 Max");
    assert_eq!(host_page_size_bytes().unwrap(), 16_384);
    assert_eq!(ctx.max_buffer_length(), 77_309_411_328);
    let expected_probe =
        model_weight_storage_requests(&model, true, false).expect("dense profile requests");
    assert_eq!(expected_probe.len(), profile.request_count);
    assert_eq!(
        model_weight_storage_inventory_digest(&expected_probe),
        profile.inventory_digest
    );
    assert!(
        parallel_copy_profile_matches(
            &ctx,
            &gguf,
            &model,
            &expected_probe,
            NativeQuantEmbeddingSelection::AutoPromoted,
            profile,
        )
        .expect("dense profile match")
    );
    drop(expected_probe);
    let tokens = [
        7734, 264, 12654, 709, 310, 12204, 279, 76938, 8240, 5199, 7638, 13,
    ];

    let ((copied, parallel), load_lines) = capture_metal_load_lines(|| {
        let copied = run_generic_retained_arm(
            &ctx,
            &gguf,
            &model,
            &tokens,
            GgufNoCopyMode::Disabled,
            GgufOwnedArenaMode::Disabled,
            None,
        );

        let embedding_selection = NativeQuantEmbeddingSelection::AutoPromoted;
        emit_native_quant_embedding_policy(&model, embedding_selection);
        let expected = model_weight_storage_requests(&model, true, false)
            .expect("parallel-copy dense storage requests");
        let storage = planned_parallel_copied_storage_for_load(
            &ctx,
            &gguf,
            &model,
            &expected,
            embedding_selection,
            population,
            ParallelDestinationLength::LogicalExact,
        )
        .expect("realize parallel dense storage");
        assert_eq!(storage.profile.id, ParallelCopyProfileId::Dense27bQ4kmV1);
        storage
            .validate_source_bytes(&gguf, &expected)
            .expect("audit every parallel resource byte");
        validate_parallel_copied_topology(
            storage.profile,
            storage.destination_length,
            &storage.expected,
            &storage.resources,
            &storage.tensors,
        )
        .expect("validate parallel-copy topology before model construction");
        assert_eq!(
            frozen_parallel_copy_order(storage.profile, &storage.expected)
                .expect("frozen schedule"),
            storage.sorted_request_indices
        );

        let command = ctx.queue.commandBuffer().expect("write guard command");
        let encoder = crate::metal::KernelEncoder::begin(&command);
        let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            encoder.note_write(&storage.tensors[0]);
        }));
        assert!(
            write_result.is_err(),
            "parallel weights must reject compute writes"
        );
        encoder.end();

        let command = ctx.queue.commandBuffer().expect("blit guard command");
        let blit = crate::metal::BlitEncoder::begin(&command);
        let blit_destination = storage
            .tensors
            .iter()
            .find(|tensor| tensor.dtype == GgmlType::F32 && tensor.n_bytes() <= 1_048_576)
            .expect("small direct F32 candidate weight");
        let writable_source = MetalTensor::zeros_f32(&ctx, blit_destination.shape.clone())
            .expect("equal-sized blit source");
        assert_eq!(writable_source.n_bytes(), blit_destination.n_bytes());
        let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            blit.copy_tensor(&writable_source, blit_destination);
        }));
        assert!(
            write_result.is_err(),
            "parallel weights must reject blit writes"
        );
        blit.end();

        let parallel_model = MetalModel::load_with_direct_storage(
            &ctx,
            &gguf,
            &model,
            ResolvedWeightLoadChoices {
                embedding_selection,
                router_f16: false,
                fused_qkv_g8: false,
            },
            &expected,
            DirectStorage::ForcedParallelCopied(storage),
            false,
        )
        .expect("construct model from audited parallel-copy storage");
        let parallel = run_loaded_model_arm(&ctx, parallel_model, &tokens, Some(copied.next_token));
        (copied, parallel)
    });
    let parallel_argmax = parallel
        .prefill_logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 { (index, value) } else { best }
        })
        .0 as i32;
    assert_eq!(copied.next_token, parallel_argmax);
    assert_eq!(parallel.next_token, parallel_argmax);
    assert_generic_retained_f32_bits(
        "parallel prefill logits",
        &copied.prefill_logits,
        &parallel.prefill_logits,
    );
    assert_generic_retained_snapshot(
        "parallel prefill snapshot",
        &copied.prefill_snapshot,
        &parallel.prefill_snapshot,
    );
    assert_generic_retained_f32_bits(
        "parallel decode logits",
        &copied.decode_logits,
        &parallel.decode_logits,
    );
    assert_generic_retained_snapshot(
        "parallel decode snapshot",
        &copied.decode_snapshot,
        &parallel.decode_snapshot,
    );

    let native_line = format!(
        "[metal-load] native quantized token embedding policy: auto-promoted ({:?} {:?})",
        model.token_embd.dtype, model.token_embd.shape
    );
    let ledger_line = concat!(
        "[metal-load-ledger] source=851/16806250496 ",
        "direct_copy=851/16806250496 direct_view=0/0 direct_alias=0/0 ",
        "tail_fallback=0/0 converted=0/0/0 derived=0/0"
    );
    assert_eq!(load_lines.len(), 5, "recognized load-line count");
    assert_eq!(load_lines[0], native_line, "A native policy line");
    assert_eq!(load_lines[1], ledger_line, "A copied ledger line");
    assert_eq!(load_lines[2], native_line, "B native policy line");
    assert_eq!(load_lines[4], ledger_line, "B copied ledger line");
    assert_eq!(
        load_lines
            .iter()
            .filter(|line| line.starts_with(marker))
            .count(),
        1,
        "candidate marker count"
    );
    assert_eq!(
        load_lines
            .iter()
            .filter(|line| *line == ledger_line)
            .count(),
        2,
        "copied ledger count"
    );

    let marker_prefix = format!(
        "{}{}",
        marker,
        concat!(
            " schema=2 profile=dense27b-q4km-v1 ",
            "resources=851 bytes=16806250496 workers=4 cuts=136,377,618 ",
            "tasks=136,241,241,233 ",
            "worker_bytes=4194110464,4214375808,4204933376,4192830848 ",
            "w0_first=2,output.weight,0,10993888,1042944000 ",
            "w0_last=135,blk.9.ssm_norm.weight,0,4205103840,512 ",
            "w1_first=136,blk.9.ssm_out.weight,0,4205104352,21626880 ",
            "w1_last=379,blk.28.attn_qkv.weight,0,8376472160,43008000 ",
            "w2_first=378,blk.28.ffn_down.weight,0,8419480160,73113600 ",
            "w2_last=618,blk.46.ffn_down.weight,0,12551299936,73113600 ",
            "w3_first=616,blk.46.ffn_gate.weight,0,12624413536,50135040 ",
            "w3_last=844,blk.63.post_attention_norm.weight,0,16817223904,20480 ",
            "create=shared,default_cache,default ",
            "observed=shared,default_cache,tracked ",
            "page=16384 alignment=32 max_buffer=77309411328 ",
            "mapped=16817244384 layout=0xd116405fd99f54d9 ",
            "inventory=50e9af4e4f590fc85687a71f5602ce035e7fdf0e2a31e928b2c7a2be10458a07 "
        )
    );
    assert!(!load_lines[3].contains(" plan="));
    let timing_suffix = load_lines[3]
        .strip_prefix(&marker_prefix)
        .expect("exact dense candidate marker prefix and field order");
    let timing_fields = timing_suffix.split(' ').collect::<Vec<_>>();
    let timing_names = [
        "allocation_us",
        "source_us",
        "copy_us",
        "binding_us",
        "ready_us",
        "user_cpu_us",
        "system_cpu_us",
        "total_cpu_us",
        "timer_minor_faults",
        "timer_major_faults",
        "instructions_delta_raw",
        "cycles_delta_raw",
    ];
    assert_eq!(
        timing_fields.len(),
        timing_names.len(),
        "candidate dynamic field count"
    );
    let mut timing_values = Vec::with_capacity(timing_names.len());
    for (field, name) in timing_fields.iter().zip(timing_names) {
        let value = field
            .strip_prefix(&format!("{name}="))
            .expect("candidate dynamic field name");
        assert!(
            !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
            "candidate dynamic field must be unsigned decimal"
        );
        let parsed = value.parse::<u64>().expect("candidate dynamic value");
        assert_eq!(
            parsed.to_string(),
            value,
            "candidate dynamic canonical form"
        );
        timing_values.push(parsed);
    }
    assert!(
        timing_values[4] > 0,
        "candidate ready time must be positive"
    );
    assert!(
        timing_values[4].abs_diff(timing_values[..4].iter().sum()) <= 4,
        "candidate timing reconciliation"
    );
    assert_eq!(
        timing_values[7],
        timing_values[5] + timing_values[6],
        "candidate CPU reconciliation"
    );
}

#[test]
#[ignore = "requires local Qwen3.6 dense 27B Q4 fixture"]
fn gguf_parallel_copied_dense27b_q4_is_bit_exact() {
    assert_gguf_parallel_dense27b_q4_is_bit_exact(
        ParallelPopulationMethod::MmapCopy,
        "[metal-gguf-parallel-copied]",
    );
}

#[test]
#[ignore = "requires local Qwen3.6 dense 27B Q4 fixture"]
fn gguf_parallel_pread_dense27b_q4_is_bit_exact() {
    assert_gguf_parallel_dense27b_q4_is_bit_exact(
        ParallelPopulationMethod::Pread,
        "[metal-gguf-parallel-pread]",
    );
}

#[test]
#[ignore = "requires local split A10B fixture and substantial virtual Metal residency"]
fn gguf_no_copy_split_a10b_resources_outlive_loader() {
    let model_path = concat!(
        "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/",
        "Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf"
    );
    assert!(std::path::Path::new(model_path).is_file());
    let ctx = MetalContext::new().expect("Metal context");
    let weak_mmaps = objc2::rc::autoreleasepool(|_| {
        let gguf = GgufFile::open(model_path).expect("open split A10B fixture");
        let model = Model::from_gguf(&gguf).expect("bind split A10B fixture");
        assert_eq!(gguf.shard_count(), 3);
        assert!(native_quant_embedding_storage_supported(&model));
        let expected = model_weight_storage_requests(&model, true, false)
            .expect("split A10B storage requests");
        let direct = expected
            .iter()
            .filter(|request| request.kind == ModelWeightStorageKind::Direct)
            .map(|request| request.desc)
            .collect::<Vec<_>>();
        let mut storage = planned_retained_storage_for_load(&ctx, &gguf, &expected, false)
            .expect("realize split A10B retained storage");
        assert_eq!(storage.plan.windows.len(), 2);
        assert_eq!(
            storage
                .plan
                .windows
                .iter()
                .map(|window| window.length as u64)
                .sum::<u64>(),
            77_015_662_592
        );
        assert_eq!(storage.plan.entries.len(), 879);
        assert_eq!(storage.plan.unique_view_bytes, 77_015_642_112);
        assert_eq!(storage.plan.alias_bytes, 0);
        assert_eq!(storage.plan.unique_fallback_bytes, 3_354_624);
        assert_eq!(
            storage
                .plan
                .entries
                .iter()
                .filter(|entry| {
                    matches!(entry.disposition, RetainedStorageDisposition::View { .. })
                })
                .count(),
            877
        );
        assert_eq!(
            storage
                .plan
                .entries
                .iter()
                .filter(|entry| {
                    matches!(
                        entry.disposition,
                        RetainedStorageDisposition::CopyFallback { .. }
                    )
                })
                .count(),
            2
        );
        let active_shards = storage
            .plan
            .windows
            .iter()
            .map(|window| window.shard_idx)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(active_shards.len(), 2);

        let mut ledger = WeightLoadLedger::default();
        for desc in &direct {
            let (tensor, materialization) = storage
                .load_direct(&ctx, &gguf, desc)
                .expect("materialize split A10B tensor");
            ledger
                .record_source(desc, materialization, tensor.n_bytes())
                .expect("record split A10B tensor");
        }
        storage
            .validate_complete(&ledger)
            .expect("complete split A10B realization");

        let mut sample_indices = std::collections::BTreeSet::new();
        for window_index in 0..storage.plan.windows.len() {
            let indices = storage
                .plan
                .entries
                .iter()
                .enumerate()
                .filter_map(|(index, entry)| match entry.disposition {
                    RetainedStorageDisposition::View {
                        window_index: entry_window,
                        ..
                    } if entry_window == window_index => Some(index),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(!indices.is_empty());
            sample_indices.insert(indices[0]);
            sample_indices.insert(indices[indices.len() / 2]);
            sample_indices.insert(indices[indices.len() - 1]);
        }
        let mut samples = Vec::new();
        for index in sample_indices {
            let desc = direct[index];
            let tensor = storage.realized[index]
                .as_ref()
                .expect("realized sample tensor")
                .clone();
            let sample_len = usize::try_from(desc.n_bytes.min(256)).unwrap();
            let tail_offset = desc.n_bytes - sample_len as u64;
            for byte_offset in [0, tail_offset] {
                let start = byte_offset as usize;
                let expected_bytes = gguf.slice(desc)[start..start + sample_len].to_vec();
                samples.push((tensor.clone(), byte_offset, expected_bytes));
            }
        }
        let weak_mmaps = active_shards
            .iter()
            .map(|&shard_idx| {
                let mmap = gguf
                    .retained_shard_mmap(shard_idx)
                    .expect("active split A10B shard mmap");
                std::sync::Arc::downgrade(&mmap)
            })
            .collect::<Vec<_>>();
        drop(direct);
        drop(expected);
        drop(model);
        drop(gguf);
        assert!(weak_mmaps.iter().all(|weak| weak.upgrade().is_some()));
        drop(storage);
        assert!(weak_mmaps.iter().all(|weak| weak.upgrade().is_some()));

        let outputs = samples
            .iter()
            .map(|(_, _, expected)| ctx.buffer_uninit(expected.len()))
            .collect::<Result<Vec<_>, _>>()
            .expect("sample output buffers");
        let command = ctx
            .queue
            .commandBuffer()
            .expect("split A10B sample command");
        let blit = crate::metal::BlitEncoder::begin(&command);
        for ((tensor, byte_offset, expected_bytes), output) in samples.iter().zip(&outputs) {
            blit.copy_buffer(
                &tensor.buffer,
                tensor.offset + byte_offset,
                output,
                0,
                expected_bytes.len() as u64,
            );
        }
        blit.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "split A10B sample blit failed");
        for ((_, _, expected), output) in samples.iter().zip(&outputs) {
            let actual = unsafe {
                std::slice::from_raw_parts(output.contents().as_ptr().cast::<u8>(), expected.len())
            };
            assert_eq!(actual, expected);
        }
        weak_mmaps
    });
    assert!(weak_mmaps.iter().all(|weak| weak.upgrade().is_none()));
}

#[test]
#[ignore = "requires QWEN_GGUF_NO_COPY_MODEL local exact 27B fixture"]
fn gguf_no_copy_layout_digest_probe() {
    let path = std::env::var("QWEN_GGUF_NO_COPY_MODEL")
        .expect("set QWEN_GGUF_NO_COPY_MODEL to the exact 27B GGUF");
    let gguf = GgufFile::open(&path).expect("open exact 27B GGUF");
    eprintln!(
        "[gguf-no-copy-layout] model={path} digest={:#018x}",
        gguf_descriptor_layout_digest(&gguf)
    );
}

#[test]
#[ignore = "requires local exact 27B model and frozen Reva prompt"]
fn gguf_no_copy_27b_prefill_and_continuation_are_bit_exact() {
    use crate::metal_dflash::{
        MetalDFlashLayerMajorScratch, PrefillScratchConfig,
        plan_prefill_scratch_with_matrix_max_pos_configured, prefill_tokens_with_multi_hidden,
    };

    struct ArmResult {
        prefill_logits: Vec<f32>,
        prefill_snapshot: SessionSnapshot,
        next_token: i32,
        decode_logits: Vec<f32>,
        decode_snapshot: SessionSnapshot,
    }

    fn run_arm(
        ctx: &MetalContext,
        gguf: &GgufFile,
        model: &Model<'_>,
        tokens: &[i32],
        mode: GgufNoCopyMode,
        prefault_enabled: bool,
        forced_next: Option<i32>,
    ) -> ArmResult {
        let metal_model =
            MetalModel::load_with_no_copy_policy(ctx, gguf, model, mode, prefault_enabled)
                .expect("load exactness arm");
        let forward = MetalForward::new(ctx, &metal_model);
        let capacity = 512;
        let mut session = MetalSession::fresh(ctx, &metal_model, capacity).expect("fresh session");
        let chunk = tokens.len() as u32;
        let plan = plan_prefill_scratch_with_matrix_max_pos_configured(
            &metal_model,
            chunk,
            capacity,
            PrefillScratchConfig::default(),
        )
        .expect("prefill scratch plan");
        let mut scratch =
            MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(ctx, &metal_model, plan)
                .expect("prefill scratch");
        let prefill_logits = prefill_tokens_with_multi_hidden(
            &forward,
            tokens,
            0,
            &mut session,
            &mut scratch,
            &[],
            None,
        )
        .expect("packed prefill");
        let identity = session.snapshot_identity(0x591, 0x27b);
        let prefill_snapshot = session
            .snapshot(
                identity.clone(),
                tokens.to_vec(),
                Some(prefill_logits.clone()),
            )
            .expect("prefill snapshot");
        let next_token = forced_next.unwrap_or_else(|| {
            prefill_logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                    if value > best.1 { (index, value) } else { best }
                })
                .0 as i32
        });
        let decode_logits = forward
            .single_token(next_token, tokens.len() as u32, &mut session)
            .expect("forced decode transition");
        let mut consumed = tokens.to_vec();
        consumed.push(next_token);
        let decode_snapshot = session
            .snapshot(identity, consumed, Some(decode_logits.clone()))
            .expect("decode snapshot");
        ArmResult {
            prefill_logits,
            prefill_snapshot,
            next_token,
            decode_logits,
            decode_snapshot,
        }
    }

    fn assert_f32_bits(label: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "{label} length");
        for (index, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{label} bit mismatch at {index}");
        }
    }

    fn assert_snapshot(label: &str, a: &SessionSnapshot, b: &SessionSnapshot) {
        assert_eq!(a.identity, b.identity, "{label} identity");
        assert_eq!(a.prefix_tokens, b.prefix_tokens, "{label} tokens");
        assert_eq!(a.kv_n_pos, b.kv_n_pos, "{label} KV positions");
        assert_eq!(a.kv_k_arena, b.kv_k_arena, "{label} K arena");
        assert_eq!(a.kv_v_arena, b.kv_v_arena, "{label} V arena");
        assert_eq!(a.gdn_conv_arena, b.gdn_conv_arena, "{label} conv arena");
        assert_eq!(a.gdn_state_arena, b.gdn_state_arena, "{label} state arena");
        match (&a.final_logits, &b.final_logits) {
            (Some(a), Some(b)) => assert_f32_bits(&format!("{label} logits"), a, b),
            (None, None) => {}
            _ => panic!("{label} final-logits presence mismatch"),
        }
    }

    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    let prompt_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt");
    assert!(
        std::path::Path::new(model_path).is_file(),
        "missing exact 27B fixture {model_path}"
    );
    assert!(
        prompt_path.is_file(),
        "missing frozen prompt {}",
        prompt_path.display()
    );
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(model_path).expect("open exact 27B GGUF");
    let model = Model::from_gguf(&gguf).expect("bind exact 27B model");
    assert!(matches_no_copy_27b_sentinel(&gguf, &model));
    let converted_embedding_requests =
        model_weight_storage_requests(&model, false, false).expect("converted request plan");
    let rollback_error = match direct_storage_for_load(
        &ctx,
        &gguf,
        &model,
        &converted_embedding_requests,
        GgufNoCopyMode::Forced,
        false,
        GgufOwnedArenaMode::Disabled,
        GgufParallelCopyMode::Disabled,
        PreparedAutoSelection::NotEligible,
        PreparedAutoRetainedSelection::NotEligible,
        NativeQuantEmbeddingSelection::AutoUnpromoted,
    ) {
        Ok(_) => panic!("exact 27B rollback must fail before resource realization"),
        Err(error) => error,
    };
    assert!(
        rollback_error
            .to_string()
            .contains("requires native token embedding residency")
    );
    let tokenizer = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
    let prompt = std::fs::read_to_string(prompt_path).expect("read frozen prompt");
    let tokens = tokenizer
        .encode(&prompt, true)
        .expect("tokenize frozen prompt");
    assert_eq!(tokens.len(), 419, "frozen Reva token count drifted");

    let copied = run_arm(
        &ctx,
        &gguf,
        &model,
        &tokens,
        GgufNoCopyMode::Disabled,
        true,
        None,
    );
    let retained = run_arm(
        &ctx,
        &gguf,
        &model,
        &tokens,
        GgufNoCopyMode::Forced,
        false,
        Some(copied.next_token),
    );
    assert_eq!(copied.next_token, retained.next_token);
    assert_f32_bits(
        "prefill logits",
        &copied.prefill_logits,
        &retained.prefill_logits,
    );
    assert_snapshot(
        "prefill snapshot",
        &copied.prefill_snapshot,
        &retained.prefill_snapshot,
    );
    assert_f32_bits(
        "decode logits",
        &copied.decode_logits,
        &retained.decode_logits,
    );
    assert_snapshot(
        "decode snapshot",
        &copied.decode_snapshot,
        &retained.decode_snapshot,
    );
}

#[test]
fn native_quant_embedding_support_requires_an_aligned_matrix() {
    assert!(native_quant_embedding_supported(
        GgmlType::Q4_K,
        &[5120, 248_320]
    ));
    assert!(native_quant_embedding_supported(
        GgmlType::Q8_0,
        &[2048, 248_320]
    ));
    assert!(native_quant_embedding_supported(
        GgmlType::Q8_0,
        &[3072, 248_320]
    ));
    assert!(native_quant_embedding_supported(
        GgmlType::Q6_K,
        &[5120, 248_320]
    ));
    assert!(native_quant_embedding_supported(
        GgmlType::IQ4_NL,
        &[160, 320_001_536]
    ));
    assert!(!native_quant_embedding_supported(
        GgmlType::Q4_K,
        &[5119, 248_320]
    ));
    assert!(!native_quant_embedding_supported(
        GgmlType::Q6_K,
        &[5119, 248_320]
    ));
    assert!(!native_quant_embedding_supported(
        GgmlType::Q8_0,
        &[2047, 248_320]
    ));
    assert!(!native_quant_embedding_supported(
        GgmlType::IQ4_NL,
        &[159, 320_001_536]
    ));
    assert!(!native_quant_embedding_supported(GgmlType::Q8_0, &[2048]));
    assert!(!native_quant_embedding_supported(
        GgmlType::Q8_0,
        &[0, 248_320]
    ));
}

#[test]
fn routed_moe_gate_up_support_requires_matching_native_formats() {
    assert!(moe_routed_gate_up_decode_supported(
        GgmlType::IQ4_XS,
        GgmlType::IQ4_XS
    ));
    assert!(moe_routed_gate_up_decode_supported(
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_XXS
    ));
    assert!(!moe_routed_gate_up_decode_supported(
        GgmlType::IQ4_XS,
        GgmlType::IQ3_XXS
    ));
    assert!(!moe_routed_gate_up_decode_supported(
        GgmlType::IQ4_NL,
        GgmlType::IQ4_NL
    ));
}

#[test]
fn native_quant_embedding_mode_is_strict_and_tri_state() {
    assert_eq!(
        parse_native_quant_embedding_mode(None),
        NativeQuantEmbeddingMode::Auto
    );
    for value in ["1", "true", "TRUE", "yes", "YES"] {
        assert_eq!(
            parse_native_quant_embedding_mode(Some(value)),
            NativeQuantEmbeddingMode::Forced
        );
    }
    for value in ["0", "false", "FALSE", "no", "NO"] {
        assert_eq!(
            parse_native_quant_embedding_mode(Some(value)),
            NativeQuantEmbeddingMode::Disabled
        );
    }
    for value in ["", "on", "off", "ture", "2"] {
        assert_eq!(
            parse_native_quant_embedding_mode(Some(value)),
            NativeQuantEmbeddingMode::Invalid
        );
    }
}

#[test]
fn native_quant_embedding_resolution_preserves_force_and_rollback() {
    use NativeQuantEmbeddingMode::{Auto, Disabled, Forced, Invalid};
    use NativeQuantEmbeddingSelection::Unsupported;
    use NativeQuantEmbeddingSelection::{
        AutoPromoted, AutoUnpromoted, Forced as On, InvalidDisabled, RollbackDisabled,
    };

    assert_eq!(
        resolve_native_quant_embedding(Auto, true, true),
        AutoPromoted
    );
    assert_eq!(
        resolve_native_quant_embedding(Auto, true, false),
        AutoUnpromoted
    );
    assert_eq!(resolve_native_quant_embedding(Forced, true, false), On);
    assert_eq!(
        resolve_native_quant_embedding(Disabled, true, true),
        RollbackDisabled
    );
    assert_eq!(
        resolve_native_quant_embedding(Invalid, true, true),
        InvalidDisabled
    );
    assert_eq!(
        resolve_native_quant_embedding(Forced, false, true),
        Unsupported
    );
}

#[test]
fn native_quant_embedding_defaults_only_on_promoted_fingerprints() {
    let dense = crate::model::QWEN3_27B;
    assert!(native_quant_embedding_default_promoted(
        &dense,
        false,
        GgmlType::Q4_K,
        &[5120, 248_320],
    ));
    let mut dense_without_mtp = dense;
    dense_without_mtp.mtp_n_hidden_layers = 0;
    assert!(native_quant_embedding_default_promoted(
        &dense_without_mtp,
        false,
        GgmlType::Q4_K,
        &[5120, 248_320],
    ));
    assert!(!native_quant_embedding_default_promoted(
        &dense,
        true,
        GgmlType::Q4_K,
        &[5120, 248_320],
    ));
    assert!(!native_quant_embedding_default_promoted(
        &dense,
        false,
        GgmlType::Q8_0,
        &[5120, 248_320],
    ));
    assert!(native_quant_embedding_default_promoted(
        &dense,
        false,
        GgmlType::Q6_K,
        &[5120, 248_320],
    ));

    let mut a3b = crate::model::Arch {
        kind: ArchKind::Moe,
        n_layer: 40,
        hidden_size: 2048,
        intermediate_size: 0,
        vocab_size: 248_320,
        full_attention_interval: 4,
        n_q_heads: 16,
        n_kv_heads: 2,
        attn_head_dim: 256,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        gdn_n_v_heads: 32,
        gdn_n_k_heads: 16,
        gdn_head_dim: 128,
        gdn_conv_kernel: 4,
        expert_count: 256,
        expert_used_count: 8,
        expert_feed_forward_length: 512,
        expert_shared_feed_forward_length: 512,
        mtp_n_hidden_layers: 0,
    };
    assert!(native_quant_embedding_default_promoted(
        &a3b,
        false,
        GgmlType::Q8_0,
        &[2048, 248_320],
    ));
    a3b.mtp_n_hidden_layers = 1;
    assert!(native_quant_embedding_default_promoted(
        &a3b,
        false,
        GgmlType::Q8_0,
        &[2048, 248_320],
    ));
    assert!(!native_quant_embedding_default_promoted(
        &a3b,
        true,
        GgmlType::Q8_0,
        &[2048, 248_320],
    ));
    assert!(!native_quant_embedding_default_promoted(
        &a3b,
        false,
        GgmlType::Q4_K,
        &[2048, 248_320],
    ));
    assert!(!native_quant_embedding_default_promoted(
        &a3b,
        false,
        GgmlType::Q8_0,
        &[3072, 248_320],
    ));
}

#[test]
#[ignore = "requires QWEN_EMBED_RESIDENCY_MODEL local fixture"]
fn quantized_embedding_residency_load_probe() {
    let path = std::env::var("QWEN_EMBED_RESIDENCY_MODEL")
        .expect("set QWEN_EMBED_RESIDENCY_MODEL to a local GGUF");
    let ctx = MetalContext::new().expect("metal context");
    let g = GgufFile::open(&path).expect("open fixture");
    let m = Model::from_gguf(&g).expect("bind model");
    let source = m.token_embd;
    assert!(
        !matches!(source.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16),
        "probe requires a quantized token embedding"
    );
    let f32_bytes = source.n_elements().checked_mul(4).expect("F32 byte size");
    let allocated_before = ctx.current_allocated_size();
    let started = std::time::Instant::now();
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let load_ms = started.elapsed().as_secs_f64() * 1e3;
    let allocated_after = ctx.current_allocated_size();
    let allocated_delta = allocated_after.saturating_sub(allocated_before);
    let resident_bytes = mm.token_embd.n_bytes();
    let backing_bytes = mm.token_embd.buffer.length() as u64;
    let selection = resolve_native_quant_embedding(
        native_quant_embedding_mode(),
        native_quant_embedding_supported(source.dtype, &source.shape),
        native_quant_embedding_default_promoted(
            &m.arch,
            m.tied_embeddings,
            source.dtype,
            &source.shape,
        ),
    );
    let expect_native = selection.uses_native();

    assert_eq!(mm.token_embd.dtype == source.dtype, expect_native);
    assert!(
        mm.token_embd.offset + resident_bytes <= backing_bytes,
        "logical embedding range must fit its backing buffer"
    );
    if expect_native {
        assert_eq!(resident_bytes, source.n_bytes);
        let theoretical_savings = f32_bytes - source.n_bytes;
        let observed_savings = f32_bytes - resident_bytes;
        assert!(observed_savings * 100 >= theoretical_savings * 95);
    } else if !matches!(source.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        assert_eq!(mm.token_embd.dtype, GgmlType::F32);
        assert_eq!(resident_bytes, f32_bytes);
    }
    eprintln!(
        concat!(
            "[embed-residency] model={} source={:?} resident={:?} ",
            "source_bytes={} resident_bytes={} backing_bytes={} f32_bytes={} device_delta={} ",
            "load_ms={:.3}"
        ),
        path,
        source.dtype,
        mm.token_embd.dtype,
        source.n_bytes,
        resident_bytes,
        backing_bytes,
        f32_bytes,
        allocated_delta,
        load_ms,
    );
}

#[test]
fn checked_u64_div_exact_rejects_zero_or_remainder() {
    assert!(checked_u64_div_exact(12, 3, "ok").is_ok());
    assert!(checked_u64_div_exact(12, 0, "zero").is_err());
    assert!(checked_u64_div_exact(13, 3, "remainder").is_err());
}

fn argmax_i32_local(xs: &[f32]) -> i32 {
    let mut best = (0usize, xs[0]);
    for (i, &v) in xs.iter().enumerate().skip(1) {
        if v.total_cmp(&best.1) == std::cmp::Ordering::Greater {
            best = (i, v);
        }
    }
    best.0 as i32
}

fn run_argmax_chain_equivalence(model_path: &str, label: &str, cos_floor: f64) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[argmax-chain-{label}] skipped — fixture missing");
        return;
    }
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let mut ids = tok
        .encode("The quick brown fox jumps over the lazy dog", false)
        .expect("tokenize");
    ids.truncate(ids.len().min(4));
    assert!(!ids.is_empty(), "tokenizer returned empty prompt");

    let mf = MetalForward::new(&ctx, &mm);
    let cap = ids.len() + 8;
    let mut s_full = MetalSession::fresh(&ctx, &mm, cap).expect("full session");
    let mut s_arg = MetalSession::fresh(&ctx, &mm, cap).expect("arg session");
    let mut last_full = Vec::new();

    for (i, &tid) in ids.iter().enumerate() {
        last_full = mf
            .single_token(tid, i as u32, &mut s_full)
            .expect("full forward");
        let arg = mf
            .single_token_argmax(tid, i as u32, &mut s_arg)
            .expect("argmax forward");
        assert_eq!(
            arg,
            argmax_i32_local(&last_full),
            "[argmax-chain-{label}] prompt-step argmax mismatch at position {i}"
        );
    }

    let next_tok = argmax_i32_local(&last_full);
    let logits_full = mf
        .single_token(next_tok, ids.len() as u32, &mut s_full)
        .expect("full follow-up");
    let logits_arg = mf
        .single_token(next_tok, ids.len() as u32, &mut s_arg)
        .expect("argmax follow-up");

    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (a, b) in logits_full.iter().zip(logits_arg.iter()) {
        max_abs = max_abs.max((a - b).abs());
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    let arg_full = argmax_i32_local(&logits_full);
    let arg_arg = argmax_i32_local(&logits_arg);
    eprintln!(
        "[argmax-chain-{label}] cos={cos:.6} max|Δ|={max_abs:.4} argmax full={arg_full} argmax-path={arg_arg}"
    );
    assert_eq!(
        arg_full, arg_arg,
        "[argmax-chain-{label}] follow-up argmax mismatch"
    );
    assert!(
        cos > cos_floor,
        "[argmax-chain-{label}] cos={cos} below floor {cos_floor}"
    );
}

fn run_exact_greedy_chain_equivalence(model_path: &str, label: &str) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[greedy-chain-{label}] skipped - fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
    let mut prompt = tok
        .encode("The quick brown fox jumps over the lazy dog", false)
        .expect("tokenize");
    prompt.truncate(prompt.len().min(4));
    assert!(!prompt.is_empty(), "tokenizer returned empty prompt");

    let select = |logits: &[f32]| {
        let mut sampler = crate::sampling::Sampler::new(crate::sampling::SamplingConfig::default())
            .expect("greedy sampler");
        sampler.sample(logits).expect("valid logits").token
    };
    let mf = MetalForward::new(&ctx, &mm);
    let capacity = prompt.len() + 8;
    let mut full = MetalSession::fresh(&ctx, &mm, capacity).expect("full session");
    let mut greedy = MetalSession::fresh(&ctx, &mm, capacity).expect("greedy session");
    let mut consumed = Vec::new();
    let mut full_logits = Vec::new();

    for (position, &token) in prompt.iter().enumerate() {
        full_logits = mf
            .single_token(token, position as u32, &mut full)
            .expect("full prompt step");
        let selected = mf
            .single_token_greedy(token, position as u32, &mut greedy)
            .expect("greedy prompt step")
            .into_token()
            .expect("finite greedy prompt logits");
        assert_eq!(
            selected,
            select(&full_logits),
            "[greedy-chain-{label}] prompt selection at {position}"
        );
        consumed.push(token);
    }

    for step in 0..4 {
        let token = select(&full_logits);
        let position = consumed.len() as u32;
        full_logits = mf
            .single_token(token, position, &mut full)
            .expect("full generation step");
        let selected = mf
            .single_token_greedy(token, position, &mut greedy)
            .expect("greedy generation step")
            .into_token()
            .expect("finite greedy generation logits");
        assert_eq!(
            selected,
            select(&full_logits),
            "[greedy-chain-{label}] generation selection at {step}"
        );
        consumed.push(token);
    }

    let token = select(&full_logits);
    let position = consumed.len() as u32;
    let full_continuation = mf
        .single_token(token, position, &mut full)
        .expect("full continuation");
    let greedy_continuation = mf
        .single_token(token, position, &mut greedy)
        .expect("greedy continuation");
    consumed.push(token);
    assert!(
        full_continuation
            .iter()
            .zip(&greedy_continuation)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "[greedy-chain-{label}] continuation logits differ"
    );

    let identity = full.snapshot_identity(1, 2);
    let full_snapshot = full
        .snapshot(identity.clone(), consumed.clone(), None)
        .expect("full snapshot");
    let greedy_snapshot = greedy
        .snapshot(identity, consumed, None)
        .expect("greedy snapshot");
    assert_eq!(full_snapshot.identity, greedy_snapshot.identity);
    assert_eq!(full_snapshot.prefix_tokens, greedy_snapshot.prefix_tokens);
    assert_eq!(full_snapshot.pending_token, greedy_snapshot.pending_token);
    assert_eq!(full_snapshot.kv_n_pos, greedy_snapshot.kv_n_pos);
    assert_eq!(full_snapshot.kv_k_arena, greedy_snapshot.kv_k_arena);
    assert_eq!(full_snapshot.kv_v_arena, greedy_snapshot.kv_v_arena);
    assert_eq!(full_snapshot.gdn_conv_arena, greedy_snapshot.gdn_conv_arena);
    assert_eq!(
        full_snapshot.gdn_state_arena,
        greedy_snapshot.gdn_state_arena
    );
    eprintln!("[greedy-chain-{label}] exact-state PASS");
}

fn run_concurrent_gdn_moe_equivalence(
    model_path: &str,
    label: &str,
    max_prompt_tokens: usize,
    cos_floor: f64,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[moe-concurrent-gdn-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    assert_eq!(m.arch.kind, ArchKind::Moe);
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let mut ids = tok
        .encode("The quick brown fox jumps over the lazy dog", false)
        .expect("tokenize");
    ids.truncate(ids.len().min(max_prompt_tokens));
    assert!(!ids.is_empty(), "tokenizer returned empty prompt");

    let mf = MetalForward::new(&ctx, &mm);
    let cap = ids.len() + 8;
    let mut s_serial = MetalSession::fresh(&ctx, &mm, cap).expect("serial session");
    let mut s_conc = MetalSession::fresh(&ctx, &mm, cap).expect("concurrent session");
    let mut last_serial = Vec::new();

    let compare = |lhs: &[f32], rhs: &[f32], where_label: &str| {
        let mut max_abs = 0.0f32;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..lhs.len() {
            max_abs = max_abs.max((lhs[i] - rhs[i]).abs());
            dot += lhs[i] as f64 * rhs[i] as f64;
            na += (lhs[i] as f64).powi(2);
            nb += (rhs[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        let arg_l = argmax_i32_local(lhs);
        let arg_r = argmax_i32_local(rhs);
        eprintln!(
            "[moe-concurrent-gdn-{label}] {where_label}: argmax serial={arg_l} conc={arg_r} max|Δ|={max_abs:.4} cos={cos:.6}"
        );
        assert_eq!(arg_l, arg_r, "[{where_label}] argmax disagreement");
        assert!(
            cos >= cos_floor,
            "[{where_label}] cos={cos} below floor {cos_floor}"
        );
    };

    for (i, &tid) in ids.iter().enumerate() {
        let (serial, _) = mf
            .single_token_profiled_moe(tid, i as u32, &mut s_serial)
            .expect("serial prompt step");
        let (concurrent, _) = mf
            .single_token_profiled_concurrent_gdn_moe(tid, i as u32, &mut s_conc)
            .expect("concurrent prompt step");
        compare(&serial, &concurrent, &format!("prompt-step-{i}"));
        last_serial = serial;
    }

    let next_tok = argmax_i32_local(&last_serial);
    let (serial, _) = mf
        .single_token_profiled_moe(next_tok, ids.len() as u32, &mut s_serial)
        .expect("serial follow-up");
    let (concurrent, _) = mf
        .single_token_profiled_concurrent_gdn_moe(next_tok, ids.len() as u32, &mut s_conc)
        .expect("concurrent follow-up");
    compare(&serial, &concurrent, "follow-up");
}

/// Validate a single GDN block end-to-end on Metal vs the CPU oracle.
/// Uses block 0 of Qwen3.5-0.8B-F32 (the first GDN block, n_v=n_k=16).
/// Compares the residual stream (s.x) after the block against what
/// the CPU forward produces after running just block 0.
#[test]
fn metal_gdn_block_matches_cpu() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        eprintln!("[metal-gdn] skipped — model missing");
        return;
    }
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    // Build inputs for block 0:
    //   * residual stream = the embedding row for token "Hello" (id 9419)
    // The CPU oracle and Metal driver both run from this same starting state.
    let token_id = 9419usize;
    let h = m.arch.hidden_size as usize;
    let embed = crate::codec::dequant_to_f32(m.token_embd, g.slice(m.token_embd)).expect("embed");
    let initial_x: Vec<f32> = embed[token_id * h..(token_id + 1) * h].to_vec();

    // CPU oracle: run Forward through one block manually.
    // We reuse Forward::single_token but only after the embedding
    // step; easier path is to just replicate the block in CPU code,
    // matching what forward.rs does. But that's a duplication risk.
    // Instead: call the public `Forward::single_token` and intercept
    // by limiting blocks. Forward doesn't expose that, so we carve
    // out a CPU-block helper here that mirrors forward.rs:single_token's
    // inner block loop for block 0 only.
    //
    // For the validation we rely on Forward::single_token computing
    // the same x state (post-block-0) — but it doesn't expose that.
    // Simplest: inline the block 0 computation here using the same
    // primitives. Given block 0 of 0.8B is a GDN block, this reads
    // exactly like the GDN inner of forward.rs.
    let cpu_x_after_block0 = run_cpu_block0_for_test(&g, &m, &initial_x);

    // Metal: build a fresh session, plant initial_x in the residual
    // stream, run block 0.
    let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
    let mf = MetalForward::new(&ctx, &mm);
    mf.set_residual_for_test(&mut s, &initial_x);
    let metal_x = mf
        .run_one_gdn_block_for_test(0, 0, &mut s)
        .expect("metal block 0");

    let max_abs = metal_x
        .iter()
        .zip(cpu_x_after_block0.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = metal_x
        .iter()
        .zip(cpu_x_after_block0.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let na: f64 = metal_x.iter().map(|v| (*v as f64).powi(2)).sum();
    let nb: f64 = cpu_x_after_block0.iter().map(|v| (*v as f64).powi(2)).sum();
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    eprintln!("[metal-gdn-block0] hidden={h} max|Δ|={max_abs:.2e} cos={cos:.6}");
    assert!(max_abs < 1e-3, "block-0 drift {max_abs}");
    assert!(cos > 0.9999, "block-0 cos {cos}");
}

/// **End-to-end Metal forward** validated against `llm`/`llama_core`'s
/// snapshot dump (which uses llama.cpp under the hood and is what
/// the CPU oracle is also validated against). One token, one
/// command buffer, all 24 blocks of Qwen3.5-0.8B-F32 chained.
#[test]
fn metal_single_token_matches_cpu_oracle() {
    let model_path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    let oracle_path = "/tmp/qwen-oracle/hello_t0.f32";
    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists() {
        eprintln!("[metal-e2e] skipped — fixtures missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };

    let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
    let n = oracle_bytes.len() / 4;
    let oracle: Vec<f32> = (0..n)
        .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();

    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    assert_eq!(oracle.len(), m.arch.vocab_size as usize);

    // Tokenize "Hello" → 9419.
    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let ids = tok.encode("Hello", false).expect("tokenize");
    eprintln!("[metal-e2e] 'Hello' -> {ids:?}");
    assert_eq!(ids.len(), 1);

    let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
    let mf = MetalForward::new(&ctx, &mm);
    let t = std::time::Instant::now();
    let logits = mf.single_token(ids[0], 0, &mut s).expect("forward");
    let ms = t.elapsed().as_secs_f64() * 1e3;

    let mut max_abs = 0.0f32;
    let mut argmax_ours = 0usize;
    let mut argmax_oracle = 0usize;
    let mut max_ours = f32::NEG_INFINITY;
    let mut max_oracle = f32::NEG_INFINITY;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let d = (logits[i] - oracle[i]).abs();
        max_abs = max_abs.max(d);
        if logits[i] > max_ours {
            max_ours = logits[i];
            argmax_ours = i;
        }
        if oracle[i] > max_oracle {
            max_oracle = oracle[i];
            argmax_oracle = i;
        }
        dot += logits[i] as f64 * oracle[i] as f64;
        na += (logits[i] as f64).powi(2);
        nb += (oracle[i] as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    eprintln!(
        "[metal-e2e] {ms:.1}ms — argmax: ours={argmax_ours} ({:.4}) | oracle={argmax_oracle} ({:.4}) | max|Δ|={max_abs:.4} cos={cos:.6}",
        max_ours, max_oracle
    );
    assert_eq!(argmax_ours, argmax_oracle, "argmax disagreement");
    assert!(cos > 0.9999, "cos={cos} below threshold");
    assert!(max_abs < 0.05, "max|Δ|={max_abs} above noise floor");
}

#[test]
fn metal_single_token_concurrent_gdn_matches_serial() {
    let model_path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[metal-concurrent-gdn] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };

    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let ids = tok.encode("Hello", false).expect("tokenize");
    assert_eq!(ids.len(), 1);

    let mf = MetalForward::new(&ctx, &mm);
    let mut s_serial = MetalSession::fresh(&ctx, &mm, 256).expect("session-serial");
    let mut s_conc = MetalSession::fresh(&ctx, &mm, 256).expect("session-concurrent");

    let (serial, _) = mf
        .single_token_profiled_dense_serial(ids[0], 0, &mut s_serial)
        .expect("serial");
    let (concurrent, _) = mf
        .single_token_profiled_concurrent_gdn_dense(ids[0], 0, &mut s_conc)
        .expect("concurrent");

    let mut max_abs = 0.0f32;
    let mut argmax_serial = 0usize;
    let mut argmax_conc = 0usize;
    let mut max_serial = f32::NEG_INFINITY;
    let mut max_conc = f32::NEG_INFINITY;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..serial.len() {
        let d = (serial[i] - concurrent[i]).abs();
        max_abs = max_abs.max(d);
        if serial[i] > max_serial {
            max_serial = serial[i];
            argmax_serial = i;
        }
        if concurrent[i] > max_conc {
            max_conc = concurrent[i];
            argmax_conc = i;
        }
        dot += serial[i] as f64 * concurrent[i] as f64;
        na += (serial[i] as f64).powi(2);
        nb += (concurrent[i] as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    eprintln!(
        "[metal-concurrent-gdn] argmax serial={argmax_serial} conc={argmax_conc} max|Δ|={max_abs:.4} cos={cos:.6}"
    );
    assert_eq!(argmax_serial, argmax_conc, "argmax disagreement");
    assert!(cos > 0.9999, "cos={cos} below threshold");
    assert!(max_abs < 0.05, "max|Δ|={max_abs} above noise floor");

    let mut s_serial_argmax = MetalSession::fresh(&ctx, &mm, 256).expect("session-serial-argmax");
    let mut s_conc_argmax = MetalSession::fresh(&ctx, &mm, 256).expect("session-concurrent-argmax");
    let (serial_argmax, _) = mf
        .single_token_argmax_profiled_dense_serial(
            ids[0],
            0,
            &mut s_serial_argmax,
            ArgmaxReduction::SpeculativeLowest,
        )
        .expect("serial argmax");
    let (concurrent_argmax, _) = mf
        .single_token_argmax_profiled_concurrent_gdn_dense(ids[0], 0, &mut s_conc_argmax)
        .expect("concurrent argmax");
    assert_eq!(serial_argmax, concurrent_argmax, "GPU argmax disagreement");
}

#[test]
fn metal_single_token_concurrent_gdn_attn_matches_serial() {
    let model_path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[metal-concurrent-gdn-attn] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };

    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let ids = tok.encode("Hello", false).expect("tokenize");
    assert_eq!(ids.len(), 1);

    let mf = MetalForward::new(&ctx, &mm);
    let mut s_serial = MetalSession::fresh(&ctx, &mm, 256).expect("session-serial");
    let mut s_conc = MetalSession::fresh(&ctx, &mm, 256).expect("session-concurrent");

    let (serial, _) = mf
        .single_token_profiled_dense_serial(ids[0], 0, &mut s_serial)
        .expect("serial");
    let (concurrent, _) = mf
        .single_token_profiled_concurrent_gdn_attn_dense(ids[0], 0, &mut s_conc)
        .expect("concurrent");

    let mut max_abs = 0.0f32;
    let mut argmax_serial = 0usize;
    let mut argmax_conc = 0usize;
    let mut max_serial = f32::NEG_INFINITY;
    let mut max_conc = f32::NEG_INFINITY;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..serial.len() {
        let d = (serial[i] - concurrent[i]).abs();
        max_abs = max_abs.max(d);
        if serial[i] > max_serial {
            max_serial = serial[i];
            argmax_serial = i;
        }
        if concurrent[i] > max_conc {
            max_conc = concurrent[i];
            argmax_conc = i;
        }
        dot += serial[i] as f64 * concurrent[i] as f64;
        na += (serial[i] as f64).powi(2);
        nb += (concurrent[i] as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    eprintln!(
        "[metal-concurrent-gdn-attn] argmax serial={argmax_serial} conc={argmax_conc} max|Δ|={max_abs:.4} cos={cos:.6}"
    );
    assert_eq!(argmax_serial, argmax_conc, "argmax disagreement");
    assert!(cos > 0.9999, "cos={cos} below threshold");
    assert!(max_abs < 0.05, "max|Δ|={max_abs} above noise floor");
}

#[test]
fn metal_single_token_concurrent_gdn_moe_matches_serial_a3b() {
    run_concurrent_gdn_moe_equivalence(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b", 4, 0.995);
}

#[test]
#[ignore = "requires the 22 GB A3B fixture and Metal GPU"]
fn metal_sampled_attribution_matches_production_a3b() {
    let model_path = crate::test_fixtures::A3B_Q4_K_M.path();
    let metadata = std::fs::metadata(model_path).expect("required A3B fixture is missing");
    assert_eq!(metadata.len(), 22_134_528_992, "A3B fixture size changed");
    let mut file = std::fs::File::open(model_path).expect("open A3B for authentication");
    let mut digest = Sha256::new();
    let mut bytes = vec![0u8; 16 * 1024 * 1024];
    loop {
        let count =
            std::io::Read::read(&mut file, &mut bytes).expect("hash authenticated A3B fixture");
        if count == 0 {
            break;
        }
        digest.update(&bytes[..count]);
    }
    assert_eq!(
        format!("{:x}", digest.finalize()),
        "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
        "A3B fixture SHA-256 changed"
    );
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(model_path).expect("open A3B");
    let model = Model::from_gguf(&gguf).expect("parse A3B");
    let metal = MetalModel::load(&ctx, &gguf, &model).expect("load A3B");
    let tokenizer = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
    let ids = tokenizer.encode("Hello", false).expect("tokenize");
    assert_eq!(ids.len(), 1);

    let forward = MetalForward::new(&ctx, &metal);
    let mut ordinary = MetalSession::fresh(&ctx, &metal, 4).expect("ordinary session");
    let mut profiled = MetalSession::fresh(&ctx, &metal, 4).expect("profiled session");
    let (ordinary_logits, _) = forward
        .single_token_profiled_concurrent_gdn_moe(ids[0], 0, &mut ordinary)
        .expect("ordinary transition");
    let (profiled_logits, _, readback) = forward
        .single_token_sampled_attribution(ids[0], 0, &mut profiled)
        .expect("profiled transition");
    assert_eq!(readback.timer_spans, 2);
    assert_eq!(
        readback.bytes,
        ordinary_logits.len() * std::mem::size_of::<f32>()
    );
    assert!(readback.allocation_zero_fill_ms >= 0.0 && readback.copy_ms >= 0.0);
    assert!(
        ordinary_logits
            .iter()
            .zip(&profiled_logits)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "attributed transition logits differ"
    );

    let identity = ordinary.snapshot_identity(1, 2);
    let ordinary_snapshot = ordinary
        .snapshot(identity.clone(), ids.clone(), None)
        .expect("ordinary snapshot");
    let profiled_snapshot = profiled
        .snapshot(identity.clone(), ids.clone(), None)
        .expect("profiled snapshot");
    assert_eq!(ordinary_snapshot.kv_n_pos, profiled_snapshot.kv_n_pos);
    assert_eq!(ordinary_snapshot.kv_k_arena, profiled_snapshot.kv_k_arena);
    assert_eq!(ordinary_snapshot.kv_v_arena, profiled_snapshot.kv_v_arena);
    assert_eq!(
        ordinary_snapshot.gdn_conv_arena,
        profiled_snapshot.gdn_conv_arena
    );
    assert_eq!(
        ordinary_snapshot.gdn_state_arena,
        profiled_snapshot.gdn_state_arena
    );

    let next = argmax_i32_local(&ordinary_logits);
    let ordinary_continuation = forward
        .single_token_profiled_concurrent_gdn_moe(next, 1, &mut ordinary)
        .expect("ordinary continuation")
        .0;
    let profiled_continuation = forward
        .single_token_profiled_concurrent_gdn_moe(next, 1, &mut profiled)
        .expect("profiled continuation")
        .0;
    assert!(
        ordinary_continuation
            .iter()
            .zip(&profiled_continuation)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "continuation logits differ"
    );
    let prefix = vec![ids[0], next];
    let ordinary_snapshot = ordinary
        .snapshot(identity.clone(), prefix.clone(), None)
        .expect("ordinary continuation snapshot");
    let profiled_snapshot = profiled
        .snapshot(identity, prefix, None)
        .expect("profiled continuation snapshot");
    assert_eq!(ordinary_snapshot.kv_n_pos, profiled_snapshot.kv_n_pos);
    assert_eq!(ordinary_snapshot.kv_k_arena, profiled_snapshot.kv_k_arena);
    assert_eq!(ordinary_snapshot.kv_v_arena, profiled_snapshot.kv_v_arena);
    assert_eq!(
        ordinary_snapshot.gdn_conv_arena,
        profiled_snapshot.gdn_conv_arena
    );
    assert_eq!(
        ordinary_snapshot.gdn_state_arena,
        profiled_snapshot.gdn_state_arena
    );
    eprintln!("[sampling-attribution-a3b] exact-state PASS");
}

#[test]
#[ignore = "requires the 22 GB A3B fixture, frozen prompt, and Metal GPU"]
fn metal_sampled_structural_matches_copied_a3b() {
    use crate::metal_dflash::{
        MetalDFlashLayerMajorScratch, PrefillScratchConfig,
        plan_prefill_scratch_with_matrix_max_pos_configured, prefill_tokens_with_multi_hidden,
    };

    let model_path = crate::test_fixtures::A3B_Q4_K_M.path();
    let metadata = std::fs::metadata(model_path).expect("required A3B fixture is missing");
    assert_eq!(metadata.len(), 22_134_528_992, "A3B fixture size changed");
    let mut file = std::fs::File::open(model_path).expect("open A3B for authentication");
    let mut digest = Sha256::new();
    let mut bytes = vec![0u8; 16 * 1024 * 1024];
    loop {
        let count =
            std::io::Read::read(&mut file, &mut bytes).expect("hash authenticated A3B fixture");
        if count == 0 {
            break;
        }
        digest.update(&bytes[..count]);
    }
    assert_eq!(
        format!("{:x}", digest.finalize()),
        "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
        "A3B fixture SHA-256 changed"
    );

    let prompt_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt");
    let prompt = std::fs::read_to_string(&prompt_path).expect("read frozen Reva prompt");
    assert_eq!(prompt.len(), 1_891, "frozen prompt length changed");
    assert_eq!(
        format!("{:x}", Sha256::digest(prompt.as_bytes())),
        "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474",
        "frozen prompt SHA-256 changed"
    );

    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(model_path).expect("open A3B");
    let model = Model::from_gguf(&gguf).expect("parse A3B");
    let metal = MetalModel::load(&ctx, &gguf, &model).expect("load A3B");
    let tokenizer = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
    let ids = tokenizer
        .encode(&prompt, true)
        .expect("tokenize frozen prompt");
    assert_eq!(ids.len(), 419, "frozen prompt token count changed");
    assert_eq!(
        crate::tokenizer::token_ids_sha256_i32le(&ids),
        "fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f",
        "frozen prompt token identity changed"
    );

    let forward = MetalForward::new(&ctx, &metal);
    forward
        .ensure_sampled_structural_supported()
        .expect("sampled structural support");
    let capacity = 1_024;
    let mut ordinary = MetalSession::fresh(&ctx, &metal, capacity).expect("ordinary session");
    let mut structural = MetalSession::fresh(&ctx, &metal, capacity).expect("structural session");
    forward
        .ensure_sampled_structural_session_supported(&ordinary)
        .expect("ordinary session row support");
    forward
        .ensure_sampled_structural_session_supported(&structural)
        .expect("structural session row support");
    let mut invalid = MetalSession::fresh(&ctx, &metal, capacity).expect("validation session");
    let valid_logits = invalid.logits.clone();
    invalid.logits.dtype = GgmlType::F16;
    assert!(
        forward
            .ensure_sampled_structural_session_supported(&invalid)
            .is_err(),
        "wrong logits dtype must fail preflight"
    );
    invalid.logits = valid_logits.clone();
    invalid.logits.shape = vec![248_319];
    assert!(
        forward
            .ensure_sampled_structural_session_supported(&invalid)
            .is_err(),
        "wrong logits shape must fail preflight"
    );
    invalid.logits = valid_logits.clone();
    invalid.logits.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(
        forward
            .ensure_sampled_structural_session_supported(&invalid)
            .is_err(),
        "read-only logits provenance must fail preflight"
    );
    invalid.logits = valid_logits.clone();
    invalid.logits.offset = std::mem::size_of::<f32>() as u64;
    assert!(
        forward
            .ensure_sampled_structural_session_supported(&invalid)
            .is_err(),
        "out-of-bounds logits range must fail preflight"
    );
    let mut misaligned =
        MetalTensor::zeros_f32(&ctx, vec![248_321]).expect("oversized logits alignment fixture");
    misaligned.shape = vec![248_320];
    misaligned.offset = 1;
    invalid.logits = misaligned;
    assert!(
        forward
            .ensure_sampled_structural_session_supported(&invalid)
            .is_err(),
        "misaligned logits address must fail preflight"
    );
    let plan = plan_prefill_scratch_with_matrix_max_pos_configured(
        &metal,
        1_024,
        capacity,
        PrefillScratchConfig::default(),
    )
    .expect("prefill scratch plan");
    let mut ordinary_scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(&ctx, &metal, plan.clone())
            .expect("ordinary prefill scratch");
    let mut structural_scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(&ctx, &metal, plan)
            .expect("structural prefill scratch");
    let ordinary_prompt_logits = prefill_tokens_with_multi_hidden(
        &forward,
        &ids,
        0,
        &mut ordinary,
        &mut ordinary_scratch,
        &[],
        None,
    )
    .expect("ordinary prompt prefill");
    let structural_prompt_logits = prefill_tokens_with_multi_hidden(
        &forward,
        &ids,
        0,
        &mut structural,
        &mut structural_scratch,
        &[],
        None,
    )
    .expect("structural prompt prefill");
    assert_generic_retained_f32_bits(
        "sampled structural prompt logits",
        &ordinary_prompt_logits,
        &structural_prompt_logits,
    );

    let config = SamplingConfig::qwen_chat(42);
    assert_eq!(config.temperature.to_bits(), 0.7f32.to_bits());
    assert_eq!(config.top_k, 200);
    assert_eq!(config.top_p.to_bits(), 1.0f32.to_bits());
    assert_eq!(config.min_p.to_bits(), 0.05f32.to_bits());
    assert_eq!(config.seed, 42);
    let mut ordinary_sampler = Sampler::new(config).expect("ordinary sampler");
    let mut structural_sampler = ordinary_sampler.clone();
    let ordinary_initial = ordinary_sampler
        .sample(&ordinary_prompt_logits)
        .expect("ordinary prompt sample");
    let (structural_initial, prompt_evidence) = structural_sampler
        .sample_bounded_top_k(&structural_prompt_logits)
        .expect("structural prompt sample");
    assert!(prompt_evidence.used_bounded_path);
    assert_eq!(ordinary_initial, structural_initial);
    assert_eq!(ordinary_sampler.draws(), structural_sampler.draws());

    let mut current = ordinary_initial;
    let mut consumed = ids.clone();
    let mut resident_head_wait_calls = 0u64;
    let mut validated_shared_row_calls = 0u64;
    for step in 0..127usize {
        let position = ids.len() + step;
        let (ordinary_logits, _) = forward
            .single_token_profiled_concurrent_gdn_moe(current.token, position as u32, &mut ordinary)
            .expect("ordinary sampled transition");
        let ordinary_next = ordinary_sampler
            .sample(&ordinary_logits)
            .expect("ordinary transition sample");
        let (structural_next, _, evidence) = forward
            .single_token_sampled_structural_scoped(
                current.token,
                position as u32,
                &mut structural,
                |row| {
                    assert_generic_retained_f32_bits(
                        "sampled structural transition logits",
                        &ordinary_logits,
                        row,
                    );
                    structural_sampler.sample_bounded_top_k(row)
                },
            )
            .expect("structural sampled transition");
        let (structural_next, bounded) = structural_next.expect("structural transition sample");
        assert!(bounded.used_bounded_path);
        assert_eq!(evidence.resident_head_wait_calls, 1);
        assert_eq!(evidence.validated_shared_row_calls, 1);
        assert_eq!(evidence.transition_logits_copy_bytes, 0);
        assert_eq!(evidence.extra_command_buffers, 0);
        assert_eq!(evidence.gpu_sampling_dispatches, 0);
        assert_eq!(ordinary_next, structural_next, "sample mismatch at {step}");
        assert_eq!(
            ordinary_sampler.draws(),
            structural_sampler.draws(),
            "draw mismatch at {step}"
        );
        resident_head_wait_calls += evidence.resident_head_wait_calls;
        validated_shared_row_calls += evidence.validated_shared_row_calls;
        consumed.push(current.token);
        current = ordinary_next;
    }
    assert_eq!(resident_head_wait_calls, 127);
    assert_eq!(validated_shared_row_calls, 127);
    assert_eq!(consumed.len(), ids.len() + 127);

    let identity = ordinary.snapshot_identity(0x660, 0x660);
    let ordinary_snapshot = ordinary
        .snapshot(identity.clone(), consumed.clone(), None)
        .expect("ordinary sampled snapshot");
    let structural_snapshot = structural
        .snapshot(identity.clone(), consumed.clone(), None)
        .expect("structural sampled snapshot");
    assert_generic_retained_snapshot(
        "sampled structural state",
        &ordinary_snapshot,
        &structural_snapshot,
    );

    let continuation_position = consumed.len() as u32;
    let ordinary_continuation = forward
        .single_token_profiled_concurrent_gdn_moe(
            current.token,
            continuation_position,
            &mut ordinary,
        )
        .expect("ordinary continuation")
        .0;
    let structural_continuation = forward
        .single_token_profiled_concurrent_gdn_moe(
            current.token,
            continuation_position,
            &mut structural,
        )
        .expect("structural continuation")
        .0;
    assert_generic_retained_f32_bits(
        "sampled structural continuation logits",
        &ordinary_continuation,
        &structural_continuation,
    );
    consumed.push(current.token);
    let ordinary_snapshot = ordinary
        .snapshot(identity.clone(), consumed.clone(), None)
        .expect("ordinary continuation snapshot");
    let structural_snapshot = structural
        .snapshot(identity, consumed, None)
        .expect("structural continuation snapshot");
    assert_generic_retained_snapshot(
        "sampled structural continuation state",
        &ordinary_snapshot,
        &structural_snapshot,
    );
    eprintln!(
        "[sampled-structural-a3b] exact rows/state PASS \
         prompt_token_sha256=fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f"
    );
}

#[test]
fn metal_single_token_concurrent_gdn_moe_matches_serial_a10b_smoke() {
    run_concurrent_gdn_moe_equivalence(
        "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",
        "a10b",
        2,
        0.995,
    );
}

#[test]
fn metal_35b_a3b_moe_matches_cpu_smoke() {
    let model_path = crate::test_fixtures::A3B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[metal-moe-a3b] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    assert_eq!(m.arch.kind, ArchKind::Moe);
    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let ids = tok.encode("Hello", false).expect("tokenize");
    assert_eq!(ids.len(), 1);

    let f = Forward::new(&g, &m);
    let mut cpu_state = crate::forward::GdnState::fresh(&m);
    let mut cpu_kv = crate::forward::KvCache::with_capacity(&m, 8);
    let cpu = f
        .single_token(ids[0], 0, &mut cpu_state, &mut cpu_kv)
        .expect("cpu forward");

    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let mut s = MetalSession::fresh(&ctx, &mm, 8).expect("session");
    let (metal, prof) = mf
        .single_token_profiled(ids[0], 0, &mut s)
        .expect("metal forward");

    let mut argmax_cpu = 0usize;
    let mut argmax_metal = 0usize;
    let mut max_cpu = f32::NEG_INFINITY;
    let mut max_metal = f32::NEG_INFINITY;
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..cpu.len() {
        let d = (metal[i] - cpu[i]).abs();
        max_abs = max_abs.max(d);
        if cpu[i] > max_cpu {
            max_cpu = cpu[i];
            argmax_cpu = i;
        }
        if metal[i] > max_metal {
            max_metal = metal[i];
            argmax_metal = i;
        }
        dot += metal[i] as f64 * cpu[i] as f64;
        na += (metal[i] as f64).powi(2);
        nb += (cpu[i] as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    eprintln!(
        "[metal-moe-a3b] total={:.2}ms gpu={:.2}ms cmd_bufs={} argmax metal={argmax_metal}({max_metal:.4}) cpu={argmax_cpu}({max_cpu:.4}) max|Δ|={max_abs:.4} cos={cos:.6}",
        prof.total_ms, prof.gpu_kernel_ms, prof.moe_cmd_count
    );
    assert_eq!(argmax_metal, argmax_cpu, "argmax disagreement");
    assert!(cos > 0.995, "cos={cos} below threshold");
}

#[test]
fn metal_argmax_chain_matches_full_logits_dense() {
    run_argmax_chain_equivalence(
        crate::test_fixtures::QWEN35_0_8B_F32.path(),
        "dense-0p8b",
        0.9999,
    );
}

#[test]
fn metal_argmax_chain_matches_full_logits_moe() {
    run_argmax_chain_equivalence(crate::test_fixtures::A3B_Q4_K_M.path(), "moe-a3b", 0.995);
}

#[test]
fn metal_exact_greedy_chain_matches_full_logits_dense() {
    run_exact_greedy_chain_equivalence(crate::test_fixtures::QWEN36_27B_Q4_K_M.path(), "dense-27b");
}

#[test]
fn metal_exact_greedy_chain_matches_full_logits_moe() {
    run_exact_greedy_chain_equivalence(crate::test_fixtures::A3B_Q4_K_M.path(), "moe-a3b");
}

/// **H5.2** — Metal multi-layer hidden capture matches CPU oracle.
/// Reads hiddens at K layer indices via Metal in one command buffer,
/// then captures the same hiddens via the CPU
/// `single_token_capture_layers` reference. Cosine ≥ 0.9999 per
/// captured layer (Q4_K_M + Q6_K weights through F32 norms +
/// elementwise residual chain — same noise floor as the single-token
/// oracle test above).
///
/// Skipped on the F32 0.8B model since its layer count (24) does
/// fit `target_layer_ids = [1, 16, 31, 46, 61]` which is a 27B-style
/// list. We use [1, 5, 10, 15, 23] for the 0.8B variant — different
/// indices, same shape K=5.
#[test]
fn metal_multi_hidden_matches_cpu() {
    let model_path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[metal-multi-hidden] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };

    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    let h = m.arch.hidden_size as usize;
    let target_layer_ids: Vec<u32> = vec![1, 5, 10, 15, 23];
    let k = target_layer_ids.len();
    let token_id = 9419i32; // "Hello"
    let position = 0u32;

    // Metal capture.
    let mf = MetalForward::new(&ctx, &mm);
    let mut sm = MetalSession::fresh(&ctx, &mm, 256).expect("session-m");
    let hidden_dst = MetalTensor::zeros_f32(&ctx, vec![(k * h) as u64]).expect("hidden_dst");
    let _logits_metal = mf
        .single_token_with_multi_hidden(token_id, position, &mut sm, &target_layer_ids, &hidden_dst)
        .expect("metal multi-hidden");
    // Read back hidden_dst.
    let mut metal_hidden = vec![0.0f32; k * h];
    unsafe {
        let src = hidden_dst.buffer.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, metal_hidden.as_mut_ptr(), metal_hidden.len());
    }

    // CPU capture via the existing oracle.
    let cpu = crate::forward::Forward::new(&g, &m);
    let mut cpu_state = crate::forward::GdnState::fresh(&m);
    let mut cpu_kv = crate::forward::KvCache::new(&m);
    let cpu_hidden = cpu
        .single_token_capture_layers(
            token_id,
            position,
            &mut cpu_state,
            &mut cpu_kv,
            &target_layer_ids,
        )
        .expect("cpu multi-hidden");
    assert_eq!(metal_hidden.len(), cpu_hidden.len());

    // Compare per-layer cosine + max|Δ|.
    for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
        let off = k_idx * h;
        let mh = &metal_hidden[off..off + h];
        let ch = &cpu_hidden[off..off + h];
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut max_abs = 0.0f32;
        for i in 0..h {
            dot += mh[i] as f64 * ch[i] as f64;
            na += (mh[i] as f64).powi(2);
            nb += (ch[i] as f64).powi(2);
            max_abs = max_abs.max((mh[i] - ch[i]).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("[metal-multi-hidden] k={k_idx} (layer {lid}): cos={cos:.6} max|Δ|={max_abs:.4}");
        assert!(
            cos > 0.9999,
            "k={k_idx} layer {lid}: cos {cos} below threshold"
        );
        assert!(
            mh.iter().all(|x| x.is_finite()),
            "k={k_idx} layer {lid}: NaN/Inf in metal hidden"
        );
        // 0.8B-F32 has effectively zero quant noise; bound tight.
        assert!(
            max_abs < 1e-2,
            "k={k_idx} layer {lid}: max|Δ| {max_abs} above F32 noise floor"
        );
    }

    // Sanity: hiddens at different layers must NOT be identical
    // (catches a layout bug where we'd accidentally write the same
    // layer's residual into all K slots).
    for k_idx in 0..k - 1 {
        let a = &metal_hidden[k_idx * h..(k_idx + 1) * h];
        let b = &metal_hidden[(k_idx + 1) * h..(k_idx + 2) * h];
        let identical = a.iter().zip(b.iter()).all(|(x, y)| x == y);
        assert!(
            !identical,
            "captured hiddens at k={k_idx} and k={} are identical — multi-hidden layout bug",
            k_idx + 1
        );
    }
}

#[test]
fn dense_ffn_capture_brackets_the_residual_update() {
    let model_path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[dense-ffn-capture] skipped - fixture missing");
        return;
    }
    let Some(context) = metal_test_context() else {
        return;
    };
    let gguf = GgufFile::open(model_path).expect("open");
    let model = Model::from_gguf(&gguf).expect("load");
    let metal_model = MetalModel::load(&context, &gguf, &model).expect("metal load");
    let forward = MetalForward::new(&context, &metal_model);
    let mut session = MetalSession::fresh(&context, &metal_model, 16).expect("session");
    let hidden_size = model.arch.hidden_size as usize;
    let intermediate_size = model.arch.intermediate_size as usize;
    let layers = [5u32, 1, 5];
    let capture_shape = vec![hidden_size as u64, layers.len() as u64];
    let pre_ffn = MetalTensor::zeros_f32(&context, capture_shape.clone()).unwrap();
    let post_block = MetalTensor::zeros_f32(&context, capture_shape).unwrap();
    forward
        .single_token_with_dense_ffn_capture(9419, 0, &mut session, &layers, &pre_ffn, &post_block)
        .expect("dense FFN capture");

    let read_f32 = |tensor: &MetalTensor, len: usize| {
        let mut values = vec![0.0f32; len];
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>();
            std::ptr::copy_nonoverlapping(source, values.as_mut_ptr(), values.len());
        }
        values
    };
    let pre_values = read_f32(&pre_ffn, layers.len() * hidden_size);
    let post_values = read_f32(&post_block, layers.len() * hidden_size);
    assert_eq!(
        &pre_values[..hidden_size],
        &pre_values[2 * hidden_size..3 * hidden_size],
        "duplicate pre-FFN layer capture differs"
    );
    assert_eq!(
        &post_values[..hidden_size],
        &post_values[2 * hidden_size..3 * hidden_size],
        "duplicate post-block layer capture differs"
    );

    for (slot, &layer) in layers.iter().enumerate() {
        let pre = &pre_values[slot * hidden_size..(slot + 1) * hidden_size];
        let post = &post_values[slot * hidden_size..(slot + 1) * hidden_size];
        let (norm, gate_weight, up_weight, down_weight) = match &metal_model.blocks[layer as usize]
        {
            MetalBlock::Gdn(block) => (
                &block.post_attn_norm,
                &block.ffn_gate,
                &block.ffn_up,
                &block.ffn_down,
            ),
            MetalBlock::Attn(block) => (
                &block.post_attn_norm,
                &block.ffn_gate,
                &block.ffn_up,
                &block.ffn_down,
            ),
        };
        let pre_tensor = MetalTensor::from_bytes(
            &context,
            bytemuck::cast_slice(pre),
            vec![hidden_size as u64],
            GgmlType::F32,
        )
        .unwrap();
        let normalized = MetalTensor::zeros_f32(&context, vec![hidden_size as u64]).unwrap();
        let gate = MetalTensor::zeros_f32(&context, vec![intermediate_size as u64]).unwrap();
        let up = MetalTensor::zeros_f32(&context, vec![intermediate_size as u64]).unwrap();
        let inner = MetalTensor::zeros_f32(&context, vec![intermediate_size as u64]).unwrap();
        let output = MetalTensor::zeros_f32(&context, vec![hidden_size as u64]).unwrap();
        let command = context.queue.commandBuffer().expect("command buffer");
        let encoder = KernelEncoder::begin(&command);
        encode_rms_norm_mul_f32(&context, &encoder, &pre_tensor, norm, &normalized, RMS_EPS)
            .unwrap();
        encode_mat_vec_dispatch(
            &context,
            &encoder,
            gate_weight,
            &normalized,
            &gate,
            hidden_size,
            intermediate_size,
        )
        .unwrap();
        encode_mat_vec_dispatch(
            &context,
            &encoder,
            up_weight,
            &normalized,
            &up,
            hidden_size,
            intermediate_size,
        )
        .unwrap();
        encode_silu_mul_f32(&context, &encoder, &gate, &up, &inner).unwrap();
        encode_mat_vec_dispatch(
            &context,
            &encoder,
            down_weight,
            &inner,
            &output,
            intermediate_size,
            hidden_size,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());
        let ffn_output = read_f32(&output, hidden_size);
        let max_abs = pre
            .iter()
            .zip(&ffn_output[..hidden_size])
            .zip(post)
            .map(|((pre, ffn), post)| (pre + ffn - post).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 2e-4,
            "layer {layer} slot {slot}: captured FFN boundary error {max_abs}"
        );
    }
}

#[test]
fn dense_fixed_add_seam_is_bounded_and_ordered() {
    let model_path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[dense-fixed-add] skipped - fixture missing");
        return;
    }
    let Some(context) = metal_test_context() else {
        return;
    };
    let gguf = GgufFile::open(model_path).expect("open");
    let model = Model::from_gguf(&gguf).expect("load");
    let metal_model = MetalModel::load(&context, &gguf, &model).expect("metal load");
    let forward = MetalForward::new(&context, &metal_model);
    let hidden_size = model.arch.hidden_size as usize;
    let target_layer = 5u32;
    let coefficient = 0.25f32;
    let direction_values: Vec<f32> = (0..hidden_size)
        .map(|i| ((i * 13 + 7) % 29) as f32 / 14.0 - 1.0)
        .collect();
    let direction = MetalTensor::from_bytes(
        &context,
        bytemuck::cast_slice(&direction_values),
        vec![hidden_size as u64],
        GgmlType::F32,
    )
    .expect("direction");

    let read_capture = |tensor: &MetalTensor| {
        let mut values = vec![0.0f32; hidden_size];
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>();
            std::ptr::copy_nonoverlapping(source, values.as_mut_ptr(), values.len());
        }
        values
    };
    let capture = || {
        (
            MetalTensor::zeros_f32(&context, vec![hidden_size as u64, 1]).unwrap(),
            MetalTensor::zeros_f32(&context, vec![hidden_size as u64, 1]).unwrap(),
        )
    };
    let assert_close = |label: &str, actual: &[f32], expected: &[f32], tolerance: f32| {
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= tolerance,
            "{label}: max|delta|={max_abs} > tolerance={tolerance}"
        );
    };

    let mut baseline_session = MetalSession::fresh(&context, &metal_model, 16).unwrap();
    let (baseline_pre_dst, baseline_post_dst) = capture();
    let baseline_logits = forward
        .single_token_with_dense_ffn_capture(
            9419,
            0,
            &mut baseline_session,
            &[target_layer],
            &baseline_pre_dst,
            &baseline_post_dst,
        )
        .expect("baseline forward");
    let baseline_pre = read_capture(&baseline_pre_dst);
    let baseline_post = read_capture(&baseline_post_dst);

    let mut empty_session = MetalSession::fresh(&context, &metal_model, 16).unwrap();
    let (empty_pre_dst, empty_post_dst) = capture();
    let empty_logits = forward
        .single_token_with_dense_ffn_capture_and_interventions(
            9419,
            0,
            &mut empty_session,
            &[target_layer],
            &empty_pre_dst,
            &empty_post_dst,
            &[],
        )
        .expect("empty fixed-add forward");
    let empty_pre = read_capture(&empty_pre_dst);
    let empty_post = read_capture(&empty_post_dst);
    assert_close("empty logits", &empty_logits, &baseline_logits, 1e-6);
    assert_close("empty pre-target capture", &empty_pre, &baseline_pre, 1e-6);
    assert_close(
        "empty post-target capture",
        &empty_post,
        &baseline_post,
        1e-6,
    );

    let fixed_then_projection = [
        PostBlockIntervention::Fixed {
            layer: target_layer,
            direction: &direction,
            coefficient,
        },
        PostBlockIntervention::Projection {
            layer: target_layer,
            direction: &direction,
            coefficient: 0.5,
        },
    ];
    let projection_then_fixed = [
        PostBlockIntervention::Projection {
            layer: target_layer,
            direction: &direction,
            coefficient: 0.5,
        },
        PostBlockIntervention::Fixed {
            layer: target_layer,
            direction: &direction,
            coefficient,
        },
    ];
    let mut ordered_session = MetalSession::fresh(&context, &metal_model, 16).unwrap();
    let (ordered_pre_dst, ordered_post_dst) = capture();
    let ordered_logits = forward
        .single_token_with_dense_ffn_capture_and_interventions(
            9419,
            0,
            &mut ordered_session,
            &[target_layer],
            &ordered_pre_dst,
            &ordered_post_dst,
            &fixed_then_projection,
        )
        .expect("fixed-then-projection forward");
    let ordered_pre = read_capture(&ordered_pre_dst);
    let ordered_post = read_capture(&ordered_post_dst);

    let mut reversed_session = MetalSession::fresh(&context, &metal_model, 16).unwrap();
    let (reversed_pre_dst, reversed_post_dst) = capture();
    let reversed_logits = forward
        .single_token_with_dense_ffn_capture_and_interventions(
            9419,
            0,
            &mut reversed_session,
            &[target_layer],
            &reversed_pre_dst,
            &reversed_post_dst,
            &projection_then_fixed,
        )
        .expect("projection-then-fixed forward");
    let reversed_pre = read_capture(&reversed_pre_dst);
    let reversed_post = read_capture(&reversed_post_dst);
    let baseline_dot: f32 = baseline_post
        .iter()
        .zip(&direction_values)
        .map(|(x, direction)| x * direction)
        .sum();
    let projected_baseline: Vec<f32> = baseline_post
        .iter()
        .zip(&direction_values)
        .map(|(x, direction)| *x - 0.5 * baseline_dot * direction)
        .collect();
    let fixed_baseline: Vec<f32> = baseline_post
        .iter()
        .zip(&direction_values)
        .map(|(x, direction)| *x + coefficient * direction)
        .collect();
    let fixed_dot: f32 = fixed_baseline
        .iter()
        .zip(&direction_values)
        .map(|(x, direction)| x * direction)
        .sum();
    let expected_ordered: Vec<f32> = fixed_baseline
        .iter()
        .zip(&direction_values)
        .map(|(x, direction)| *x - 0.5 * fixed_dot * direction)
        .collect();
    let expected_reversed: Vec<f32> = projected_baseline
        .iter()
        .zip(&direction_values)
        .map(|(x, direction)| *x + coefficient * direction)
        .collect();

    assert_close(
        "ordered pre-target capture",
        &ordered_pre,
        &baseline_pre,
        2e-6,
    );
    assert_close(
        "reversed pre-target capture",
        &reversed_pre,
        &baseline_pre,
        2e-6,
    );
    assert_close(
        "fixed-then-projection target capture",
        &ordered_post,
        &expected_ordered,
        2e-5,
    );
    assert_close(
        "projection-then-fixed target capture",
        &reversed_post,
        &expected_reversed,
        2e-5,
    );
    let max_order_delta = ordered_post
        .iter()
        .zip(&reversed_post)
        .map(|(ordered, reversed)| (ordered - reversed).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_order_delta > 1e-5,
        "intervention caller order was not observable: max|delta|={max_order_delta}"
    );
    assert!(
        ordered_logits
            .iter()
            .chain(reversed_logits.iter())
            .all(|value| value.is_finite()),
        "ordered intervention logits contain NaN/Inf"
    );
    let max_logit_delta = ordered_logits
        .iter()
        .zip(&reversed_logits)
        .map(|(ordered, reversed)| (ordered - reversed).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_logit_delta > 1e-4,
        "intervention caller order did not change downstream logits: max|delta|={max_logit_delta}"
    );
}

#[test]
fn dense_intervention_no_tail_matches_full_tail() {
    let model_path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[dense-intervention-no-tail] skipped - fixture missing");
        return;
    }
    let Some(context) = metal_test_context() else {
        return;
    };
    let gguf = GgufFile::open(model_path).expect("open");
    let model = Model::from_gguf(&gguf).expect("load");
    let metal_model = MetalModel::load(&context, &gguf, &model).expect("metal load");
    let forward = MetalForward::new(&context, &metal_model);
    let tokenizer = crate::tokenizer::Tokenizer::open(model_path).expect("tokenizer");
    let mut token_ids = tokenizer
        .encode("Hello from the lens", false)
        .expect("tokenize");
    token_ids.truncate(3);
    assert!(
        token_ids.len() >= 2,
        "test prompt needs at least two tokens"
    );

    let hidden_size = model.arch.hidden_size as usize;
    let capture_layers = [4u32, 5, 5];
    let capture_elements = hidden_size * capture_layers.len();
    let direction_values: Vec<f32> = (0..hidden_size)
        .map(|index| ((index * 11 + 5) % 23) as f32 / 23.0 - 0.5)
        .collect();
    let direction = MetalTensor::from_bytes(
        &context,
        bytemuck::cast_slice(&direction_values),
        vec![hidden_size as u64],
        GgmlType::F32,
    )
    .expect("direction");
    let interventions = [
        PostBlockIntervention::Fixed {
            layer: 5,
            direction: &direction,
            coefficient: 0.125,
        },
        PostBlockIntervention::Projection {
            layer: 5,
            direction: &direction,
            coefficient: 0.25,
        },
    ];
    let read_capture = |tensor: &MetalTensor| {
        let mut values = vec![0.0f32; tensor.n_elements() as usize];
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>();
            std::ptr::copy_nonoverlapping(source, values.as_mut_ptr(), values.len());
        }
        values
    };
    let assert_bits_equal = |label: &str, left: &[f32], right: &[f32]| {
        assert_eq!(left.len(), right.len(), "{label} length");
        if let Some((index, (left, right))) = left
            .iter()
            .zip(right)
            .enumerate()
            .find(|(_, (left, right))| left.to_bits() != right.to_bits())
        {
            panic!(
                "{label} differs at {index}: {left:?} ({:#010x}) != {right:?} ({:#010x})",
                left.to_bits(),
                right.to_bits()
            );
        }
    };
    let assert_logits_untouched = |label: &str, session: &MetalSession| {
        let logits = unsafe {
            let source = session
                .logits
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(session.logits.offset as usize)
                .cast::<f32>();
            std::slice::from_raw_parts(source, model.arch.vocab_size as usize)
        };
        assert!(
            logits.iter().all(|value| value.to_bits() == 0),
            "{label} wrote the zero-initialized logits buffer"
        );
    };

    let capture_full = MetalTensor::zeros_f32(&context, vec![capture_elements as u64]).unwrap();
    let mut session_full =
        MetalSession::fresh(&context, &metal_model, token_ids.len() + 2).unwrap();
    let mut captures_full = Vec::new();
    let mut logits_full = Vec::new();
    for (position, &token_id) in token_ids.iter().enumerate() {
        logits_full = forward
            .single_token_with_post_block_interventions(
                token_id,
                position as u32,
                &mut session_full,
                &capture_layers,
                &capture_full,
                &interventions,
            )
            .expect("full-tail capture forward");
        captures_full.extend(read_capture(&capture_full));
    }

    let capture_skip = MetalTensor::zeros_f32(&context, vec![capture_elements as u64]).unwrap();
    let mut session_skip =
        MetalSession::fresh(&context, &metal_model, token_ids.len() + 2).unwrap();
    let mut captures_skip = Vec::new();
    let mut logits_skip = Vec::new();
    for (position, &token_id) in token_ids.iter().enumerate() {
        if position + 1 == token_ids.len() {
            logits_skip = forward
                .single_token_with_post_block_interventions(
                    token_id,
                    position as u32,
                    &mut session_skip,
                    &capture_layers,
                    &capture_skip,
                    &interventions,
                )
                .expect("final capture forward");
        } else {
            forward
                .single_token_with_post_block_interventions_no_tail(
                    token_id,
                    position as u32,
                    &mut session_skip,
                    &capture_layers,
                    &capture_skip,
                    &interventions,
                )
                .expect("no-tail capture forward");
            if position == 0 {
                assert_logits_untouched("capture no-tail", &session_skip);
            }
        }
        captures_skip.extend(read_capture(&capture_skip));
    }
    assert_bits_equal("post-block captures", &captures_full, &captures_skip);
    assert_bits_equal("capture final logits", &logits_full, &logits_skip);

    let narrow_storage = MetalTensor::zeros_f32(&context, vec![capture_elements as u64]).unwrap();
    let narrow_capture = narrow_storage.view_subrange(0, vec![hidden_size as u64]);
    let mut narrow_session =
        MetalSession::fresh(&context, &metal_model, token_ids.len() + 2).unwrap();
    let mut narrow_captures = Vec::new();
    let mut narrow_logits = Vec::new();
    for (position, &token_id) in token_ids.iter().enumerate() {
        narrow_logits = forward
            .single_token_with_post_block_interventions(
                token_id,
                position as u32,
                &mut narrow_session,
                &[5],
                &narrow_capture,
                &interventions,
            )
            .expect("event-local narrow capture forward");
        narrow_captures.extend(read_capture(&narrow_capture));
    }
    let expected_narrow = captures_full
        .chunks_exact(capture_elements)
        .flat_map(|capture| capture[hidden_size..2 * hidden_size].iter().copied())
        .collect::<Vec<_>>();
    assert_bits_equal(
        "event-local narrowed captures",
        &expected_narrow,
        &narrow_captures,
    );
    assert_bits_equal(
        "event-local narrowed final logits",
        &logits_full,
        &narrow_logits,
    );

    let mut no_capture_full =
        MetalSession::fresh(&context, &metal_model, token_ids.len() + 2).unwrap();
    let mut no_capture_full_logits = Vec::new();
    for (position, &token_id) in token_ids.iter().enumerate() {
        no_capture_full_logits = forward
            .single_token_with_post_block_interventions_no_capture(
                token_id,
                position as u32,
                &mut no_capture_full,
                &interventions,
            )
            .expect("full-tail no-capture forward");
    }
    assert_bits_equal(
        "capture elision final logits",
        &logits_full,
        &no_capture_full_logits,
    );

    let mut no_capture_skip =
        MetalSession::fresh(&context, &metal_model, token_ids.len() + 2).unwrap();
    let mut no_capture_skip_logits = Vec::new();
    for (position, &token_id) in token_ids.iter().enumerate() {
        if position + 1 == token_ids.len() {
            no_capture_skip_logits = forward
                .single_token_with_post_block_interventions_no_capture(
                    token_id,
                    position as u32,
                    &mut no_capture_skip,
                    &interventions,
                )
                .expect("final no-capture forward");
        } else {
            forward
                .single_token_with_post_block_interventions_no_capture_no_tail(
                    token_id,
                    position as u32,
                    &mut no_capture_skip,
                    &interventions,
                )
                .expect("no-tail no-capture forward");
            if position == 0 {
                assert_logits_untouched("no-capture no-tail", &no_capture_skip);
            }
        }
    }
    assert_bits_equal(
        "no-capture final logits",
        &no_capture_full_logits,
        &no_capture_skip_logits,
    );
}

#[test]
fn ordinary_moe_serial_post_block_fixed_add_seam() {
    let model_path = crate::test_fixtures::A3B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[moe-fixed-add] skipped - fixture missing");
        return;
    }
    let Some(context) = metal_test_context() else {
        return;
    };
    let gguf = GgufFile::open(model_path).expect("open");
    let model = Model::from_gguf(&gguf).expect("load");
    assert_eq!(model.arch.kind, ArchKind::Moe);
    let metal_model = MetalModel::load(&context, &gguf, &model).expect("metal load");
    let forward = MetalForward::new(&context, &metal_model);
    let hidden_size = model.arch.hidden_size as usize;
    let target_layer = 5u32;
    let target_layers = [target_layer - 1, target_layer];
    let coefficient = 0.25f32;
    let direction_values: Vec<f32> = (0..hidden_size)
        .map(|i| ((i * 17 + 3) % 31) as f32 / 15.0 - 1.0)
        .collect();
    let direction = MetalTensor::from_bytes(
        &context,
        bytemuck::cast_slice(&direction_values),
        vec![hidden_size as u64],
        GgmlType::F32,
    )
    .expect("direction");
    let read_capture = |tensor: &MetalTensor| {
        let mut values = vec![0.0f32; hidden_size * target_layers.len()];
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>();
            std::ptr::copy_nonoverlapping(source, values.as_mut_ptr(), values.len());
        }
        values
    };
    let assert_close = |label: &str, actual: &[f32], expected: &[f32], tolerance: f32| {
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= tolerance,
            "{label}: max|delta|={max_abs} > tolerance={tolerance}"
        );
    };

    let mut baseline_session = MetalSession::fresh(&context, &metal_model, 4).unwrap();
    let baseline_dst =
        MetalTensor::zeros_f32(&context, vec![(hidden_size * target_layers.len()) as u64]).unwrap();
    let baseline_logits = forward
        .single_token_with_post_block_interventions(
            9419,
            0,
            &mut baseline_session,
            &target_layers,
            &baseline_dst,
            &[],
        )
        .expect("ordinary MoE baseline forward");
    let baseline_hidden = read_capture(&baseline_dst);

    let mut intervention_session = MetalSession::fresh(&context, &metal_model, 4).unwrap();
    let intervention_dst =
        MetalTensor::zeros_f32(&context, vec![(hidden_size * target_layers.len()) as u64]).unwrap();
    let interventions = [PostBlockIntervention::Fixed {
        layer: target_layer,
        direction: &direction,
        coefficient,
    }];
    let intervention_logits = forward
        .single_token_with_post_block_interventions(
            9419,
            0,
            &mut intervention_session,
            &target_layers,
            &intervention_dst,
            &interventions,
        )
        .expect("ordinary MoE fixed-add forward");
    let intervention_hidden = read_capture(&intervention_dst);
    let expected_target: Vec<f32> = baseline_hidden[hidden_size..2 * hidden_size]
        .iter()
        .zip(&direction_values)
        .map(|(baseline, direction)| *baseline + coefficient * direction)
        .collect();

    assert_close(
        "MoE pre-target row",
        &intervention_hidden[..hidden_size],
        &baseline_hidden[..hidden_size],
        2e-5,
    );
    assert_close(
        "MoE fixed-add target row",
        &intervention_hidden[hidden_size..2 * hidden_size],
        &expected_target,
        2e-5,
    );
    assert!(
        baseline_logits
            .iter()
            .chain(intervention_logits.iter())
            .all(|value| value.is_finite()),
        "ordinary MoE logits contain NaN/Inf"
    );
    let max_logit_delta = intervention_logits
        .iter()
        .zip(&baseline_logits)
        .map(|(intervention, baseline)| (intervention - baseline).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_logit_delta > 1e-4,
        "MoE fixed addition did not change downstream logits: max|delta|={max_logit_delta}"
    );

    let assert_bits_equal = |label: &str, left: &[f32], right: &[f32]| {
        assert_eq!(left.len(), right.len(), "{label} length");
        if let Some((index, (left, right))) = left
            .iter()
            .zip(right)
            .enumerate()
            .find(|(_, (left, right))| left.to_bits() != right.to_bits())
        {
            panic!(
                "{label} differs at {index}: {left:?} ({:#010x}) != {right:?} ({:#010x})",
                left.to_bits(),
                right.to_bits()
            );
        }
    };
    let token_ids = [9419, 198];
    let mut full_session =
        MetalSession::fresh(&context, &metal_model, token_ids.len() + 2).unwrap();
    let full_dst =
        MetalTensor::zeros_f32(&context, vec![(hidden_size * target_layers.len()) as u64]).unwrap();
    let mut full_captures = Vec::new();
    let mut full_logits = Vec::new();
    for (position, &token_id) in token_ids.iter().enumerate() {
        full_logits = forward
            .single_token_with_post_block_interventions(
                token_id,
                position as u32,
                &mut full_session,
                &target_layers,
                &full_dst,
                &interventions,
            )
            .expect("MoE full-tail capture forward");
        full_captures.extend(read_capture(&full_dst));
    }

    let mut skip_session =
        MetalSession::fresh(&context, &metal_model, token_ids.len() + 2).unwrap();
    let skip_dst =
        MetalTensor::zeros_f32(&context, vec![(hidden_size * target_layers.len()) as u64]).unwrap();
    let mut skip_captures = Vec::new();
    let mut skip_logits = Vec::new();
    for (position, &token_id) in token_ids.iter().enumerate() {
        if position + 1 == token_ids.len() {
            skip_logits = forward
                .single_token_with_post_block_interventions(
                    token_id,
                    position as u32,
                    &mut skip_session,
                    &target_layers,
                    &skip_dst,
                    &interventions,
                )
                .expect("MoE final capture forward");
        } else {
            forward
                .single_token_with_post_block_interventions_no_tail(
                    token_id,
                    position as u32,
                    &mut skip_session,
                    &target_layers,
                    &skip_dst,
                    &interventions,
                )
                .expect("MoE no-tail capture forward");
        }
        skip_captures.extend(read_capture(&skip_dst));
    }
    assert_bits_equal("MoE no-tail captures", &full_captures, &skip_captures);
    assert_bits_equal("MoE no-tail final logits", &full_logits, &skip_logits);

    let mut no_capture_full =
        MetalSession::fresh(&context, &metal_model, token_ids.len() + 2).unwrap();
    let mut no_capture_full_logits = Vec::new();
    for (position, &token_id) in token_ids.iter().enumerate() {
        no_capture_full_logits = forward
            .single_token_with_post_block_interventions_no_capture(
                token_id,
                position as u32,
                &mut no_capture_full,
                &interventions,
            )
            .expect("MoE full-tail no-capture forward");
    }
    assert_bits_equal(
        "MoE capture elision final logits",
        &full_logits,
        &no_capture_full_logits,
    );

    let mut no_capture_skip =
        MetalSession::fresh(&context, &metal_model, token_ids.len() + 2).unwrap();
    let mut no_capture_skip_logits = Vec::new();
    for (position, &token_id) in token_ids.iter().enumerate() {
        if position + 1 == token_ids.len() {
            no_capture_skip_logits = forward
                .single_token_with_post_block_interventions_no_capture(
                    token_id,
                    position as u32,
                    &mut no_capture_skip,
                    &interventions,
                )
                .expect("MoE final no-capture forward");
        } else {
            forward
                .single_token_with_post_block_interventions_no_capture_no_tail(
                    token_id,
                    position as u32,
                    &mut no_capture_skip,
                    &interventions,
                )
                .expect("MoE no-tail no-capture forward");
        }
    }
    assert_bits_equal(
        "MoE no-capture final logits",
        &no_capture_full_logits,
        &no_capture_skip_logits,
    );
}

/// Same as `metal_gdn_block_matches_cpu` but for 27B-Q4_K_M block 0.
/// Tests the dispatch-by-dtype path on real Q4_K + Q6_K weights.
#[test]
#[ignore]
fn metal_27b_gdn_block0_matches_cpu() {
    let path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    if !std::path::Path::new(path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    // Same harness as metal_gdn_block_matches_cpu, but with token id
    // and arch dims pulled from 27B.
    let token_id = 9419usize; // "Hello"
    let h = m.arch.hidden_size as usize;
    let embed = crate::codec::dequant_to_f32(m.token_embd, g.slice(m.token_embd)).expect("embed");
    let initial_x: Vec<f32> = embed[token_id * h..(token_id + 1) * h].to_vec();

    let cpu_x = run_cpu_block0_for_test(&g, &m, &initial_x);

    let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
    let mf = MetalForward::new(&ctx, &mm);
    mf.set_residual_for_test(&mut s, &initial_x);
    let metal_x = mf
        .run_one_gdn_block_for_test(0, 0, &mut s)
        .expect("metal block 0");

    let max_abs = metal_x
        .iter()
        .zip(cpu_x.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = metal_x
        .iter()
        .zip(cpu_x.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let na: f64 = metal_x.iter().map(|v| (*v as f64).powi(2)).sum();
    let nb: f64 = cpu_x.iter().map(|v| (*v as f64).powi(2)).sum();
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    // Inspect ssm_a values for the first GDN block.
    let block0 = &m.blocks[0];
    let ssm_a_desc = match block0 {
        crate::loader::Block::Gdn(g) => g.a_log,
        _ => panic!("not gdn"),
    };
    let ssm_a = crate::codec::dequant_to_f32(ssm_a_desc, g.slice(ssm_a_desc)).unwrap();
    eprintln!(
        "[metal-27b-gdn0] ssm_a[0..8]={:?}",
        &ssm_a[..8.min(ssm_a.len())]
    );
    eprintln!(
        "[metal-27b-gdn0] ssm_a min={:.4} max={:.4}",
        ssm_a.iter().cloned().fold(f32::INFINITY, f32::min),
        ssm_a.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );

    let nm: f32 = metal_x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nc: f32 = cpu_x.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!(
        "[metal-27b-gdn0] hidden={h} ||metal||={nm:.4e} ||cpu||={nc:.4e} max|Δ|={max_abs:.4} cos={cos:.6}"
    );
    eprintln!(
        "[metal-27b-gdn0] metal[0..4]={:?}\n              cpu[0..4]={:?}",
        &metal_x[..4.min(metal_x.len())],
        &cpu_x[..4.min(cpu_x.len())]
    );
    // Q4_K + Q6_K: relax noise floor a bit.
    assert!(cos > 0.999, "27B block 0 cos={cos}");
}

/// **End-to-end Metal forward on the 27B Q4_K_M target.** Validates
/// the quantized weight path: native Q4_K and Q6_K mat-vec kernels
/// dispatched based on tensor dtype, no per-call dequant.
///
/// This is the test that proves we can run the full production
/// 27B target on Metal with bit-tight correctness vs llm/llama_core.
/// Once this passes, we benchmark vs llama-bench.
///
/// Marked #[ignore] because (1) loading 16.8 GB of weights through
/// the loader takes a few seconds and (2) the codec-fallback path
/// for unsupported quants (Q5_K, etc.) might dequant some tensors,
/// and we want to flag that explicitly when run.
#[test]
#[ignore]
fn metal_27b_q4_k_m_matches_oracle() {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    let oracle_path = "/tmp/qwen-oracle/hello_27b_q4km.f32";
    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists() {
        eprintln!("[metal-27b] skipped — fixtures missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };

    let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
    let n = oracle_bytes.len() / 4;
    let oracle: Vec<f32> = (0..n)
        .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();

    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let ids = tok.encode("Hello", false).expect("tokenize");
    eprintln!("[metal-27b] 'Hello' -> {ids:?}");

    let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
    let mf = MetalForward::new(&ctx, &mm);

    // Warmup: first call compiles all 20+ kernel pipeline state objects.
    let _ = mf.single_token(ids[0], 0, &mut s).expect("warmup");
    // Reset session for the timed run.
    let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session2");

    let t = std::time::Instant::now();
    let logits = mf.single_token(ids[0], 0, &mut s).expect("forward");
    let ms = t.elapsed().as_secs_f64() * 1e3;

    let mut max_abs = 0.0f32;
    let mut argmax_ours = 0usize;
    let mut argmax_oracle = 0usize;
    let mut max_ours = f32::NEG_INFINITY;
    let mut max_oracle = f32::NEG_INFINITY;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let d = (logits[i] - oracle[i]).abs();
        max_abs = max_abs.max(d);
        if logits[i] > max_ours {
            max_ours = logits[i];
            argmax_ours = i;
        }
        if oracle[i] > max_oracle {
            max_oracle = oracle[i];
            argmax_oracle = i;
        }
        dot += logits[i] as f64 * oracle[i] as f64;
        na += (logits[i] as f64).powi(2);
        nb += (oracle[i] as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    eprintln!(
        "[metal-27b] {ms:.1}ms — argmax: ours={argmax_ours} ({:.4}) | oracle={argmax_oracle} ({:.4}) | max|Δ|={max_abs:.4} cos={cos:.6}",
        max_ours, max_oracle
    );
    eprintln!(
        "[metal-27b] effective decode tok/s (single-token, single-shot): {:.2}",
        1000.0 / ms
    );
    assert_eq!(argmax_ours, argmax_oracle, "argmax disagreement");
    assert!(cos > 0.999, "cos={cos} below threshold");
}

/// **Bench the Q5_K → F32 fallback cost.** Replay all 48 GDN layers'
/// `ssm_out.weight` mat-vecs in F32 (current state), measure GPU
/// kernel time. Then estimate the native-Q5_K time as
/// `f32_time × (q5_bytes / f32_bytes)` and report the delta.
#[test]
#[ignore]
fn metal_27b_q5_fallback_bench() {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");

    // Find all blk.*.ssm_out.weight tensors with Q5_K dtype.
    let q5_tensors: Vec<&TensorDesc> = g
        .tensors
        .iter()
        .filter(|t| {
            t.name.ends_with(".ssm_out.weight") && t.dtype == GgmlType::Q5_K && t.shape.len() == 2
        })
        .collect();
    eprintln!("[q5-bench] {} Q5_K ssm_out tensors", q5_tensors.len());
    if q5_tensors.is_empty() {
        return;
    }

    // For each one: current path is dequant-to-F32 then F32 mat_vec.
    // Build the F32 weight buffers, plus a constant input vector and
    // an output buffer. Replay all 48 mat_vecs in one command buffer
    // and time it.
    let n_in = q5_tensors[0].shape[0] as usize; // 6144 for 27B GDN
    let n_out = q5_tensors[0].shape[1] as usize; // 5120
    let total_q5_bytes: u64 = q5_tensors.iter().map(|t| t.n_bytes).sum();
    let total_f32_bytes: u64 = q5_tensors
        .iter()
        .map(|t| (t.shape.iter().product::<u64>()) * 4)
        .sum();

    eprintln!("[q5-bench] shape [{n_in}, {n_out}], 48 layers");
    eprintln!(
        "[q5-bench] total Q5_K bytes: {:.2} MiB",
        total_q5_bytes as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "[q5-bench] total F32 bytes:  {:.2} MiB (current resident)",
        total_f32_bytes as f64 / (1024.0 * 1024.0)
    );

    // Dequant all to F32 + upload as MetalTensor.
    let f32_weights: Vec<MetalTensor> = q5_tensors
        .iter()
        .map(|t| {
            let f = crate::codec::dequant_to_f32(t, g.slice(t)).unwrap();
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&f),
                t.shape.clone(),
                GgmlType::F32,
            )
            .unwrap()
        })
        .collect();

    // Input + output buffers.
    let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 1e-3).sin()).collect();
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .unwrap();
    let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();

    // Warmup.
    for _ in 0..3 {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        for w in &f32_weights {
            crate::metal::encode_mat_vec_f32(&ctx, &enc, w, &x_t, &y_t, n_in, n_out).unwrap();
        }
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    // Timed replay.
    const ITERS: usize = 30;
    let t = std::time::Instant::now();
    let mut gpu_sum_ms = 0.0f64;
    for _ in 0..ITERS {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        for w in &f32_weights {
            crate::metal::encode_mat_vec_f32(&ctx, &enc, w, &x_t, &y_t, n_in, n_out).unwrap();
        }
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        gpu_sum_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
    }
    let total_ms = t.elapsed().as_secs_f64() * 1e3;
    let per_iter_total = total_ms / ITERS as f64;
    let per_iter_gpu = gpu_sum_ms / ITERS as f64;

    let bytes_per_iter_f32 = total_f32_bytes as f64;
    let bytes_per_iter_q5 = total_q5_bytes as f64;
    let bw_f32 = bytes_per_iter_f32 / (per_iter_gpu / 1000.0) / 1e9;
    let predicted_q5_ms = per_iter_gpu * (bytes_per_iter_q5 / bytes_per_iter_f32);

    eprintln!(
        "[q5-bench] {ITERS} iters: total {per_iter_total:.2} ms/iter, gpu {per_iter_gpu:.2} ms/iter"
    );
    eprintln!("[q5-bench]   F32 BW achieved:    {bw_f32:.0} GB/s");
    eprintln!("[q5-bench]   F32 mat_vec cost (current):  {per_iter_gpu:.2} ms/token");
    eprintln!(
        "[q5-bench]   estimated native Q5_K cost:  {predicted_q5_ms:.2} ms/token  (BW-scaled)"
    );
    eprintln!(
        "[q5-bench]   POTENTIAL SAVINGS:           {:.2} ms/token",
        per_iter_gpu - predicted_q5_ms
    );
    eprintln!(
        "[q5-bench]   we're at 51.26 ms total; saving this would put us at {:.2} ms = {:.2} t/s",
        51.26 - (per_iter_gpu - predicted_q5_ms),
        1000.0 / (51.26 - (per_iter_gpu - predicted_q5_ms))
    );
}

/// **Per-tensor byte ledger.** Audit what's actually loaded into Metal
/// memory vs what came out of the GGUF. Specifically: which tensors
/// got native dtype, which got dequant-fallback to F32, and how many
/// bytes per category. Run before optimization to ground decisions.
#[test]
#[ignore]
fn metal_27b_byte_ledger() {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        return;
    }
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");

    // Walk loader::Model and classify each tensor.
    // Loader emits: token_embd, output_norm, lm_head, then per-block.
    // For each tensor, we ask: would MetalModel::load() preserve native
    // or fallback to F32?
    // Match the policy in MetalModel::load:
    //   * load_f32 (ALWAYS dequant to F32): norms, ssm_a, ssm_dt, conv1d,
    //     ssm_norm, q_norm, k_norm, output_norm; token_embd is F32 by
    //     default and Q4_K/Q6_K/Q8_0-native under QWEN_NATIVE_QUANT_EMBED.
    //   * load_weight (preserves F32/Q4_K/Q6_K, falls back to F32 for
    //     others): all the mat_vec weights — lm_head, ffn_*, attn_q/k/v/o,
    //     attn_qkv, attn_gate, in_proj_qkv, in_proj_z, beta_proj,
    //     alpha_proj, out_proj
    let mut stats: std::collections::BTreeMap<String, (u64, u64, u64)> =
        std::collections::BTreeMap::new(); // role -> (gguf_bytes, metal_bytes, count)
    let bump = |stats: &mut std::collections::BTreeMap<String, (u64, u64, u64)>,
                role: &str,
                gguf_b: u64,
                metal_b: u64| {
        let e = stats.entry(role.into()).or_insert((0, 0, 0));
        e.0 += gguf_b;
        e.1 += metal_b;
        e.2 += 1;
    };
    let f32_size = |shape: &[u64]| -> u64 { shape.iter().product::<u64>() * 4 };

    // Top-level tensors.
    bump(
        &mut stats,
        "token_embd (F32 default; Q4_K/Q6_K/Q8_0 native opt-in)",
        m.token_embd.n_bytes,
        f32_size(&m.token_embd.shape),
    );
    bump(
        &mut stats,
        "output_norm (load_f32)",
        m.output_norm.n_bytes,
        f32_size(&m.output_norm.shape),
    );
    let lm_head_kept = weight_dtype_kept_native(m.lm_head.dtype);
    bump(
        &mut stats,
        if lm_head_kept {
            "lm_head (native)"
        } else {
            "lm_head (FALLBACK F32)"
        },
        m.lm_head.n_bytes,
        if lm_head_kept {
            m.lm_head.n_bytes
        } else {
            f32_size(&m.lm_head.shape)
        },
    );

    for b in &m.blocks {
        match b {
            crate::loader::Block::Gdn(g) => {
                let f32_descs: &[&TensorDesc] = &[
                    g.attn_norm,
                    g.post_attention_norm,
                    g.a_log,
                    g.dt_bias,
                    g.conv1d,
                    g.norm,
                ];
                for d in f32_descs {
                    bump(
                        &mut stats,
                        "gdn f32-required",
                        d.n_bytes,
                        f32_size(&d.shape),
                    );
                }
                let weight_descs: &[(&TensorDesc, &str)] = &[
                    (g.in_proj_qkv, "gdn in_proj_qkv"),
                    (g.in_proj_z, "gdn in_proj_z"),
                    (g.beta_proj, "gdn beta_proj"),
                    (g.alpha_proj, "gdn alpha_proj"),
                    (g.out_proj, "gdn out_proj"),
                    (g.ffn_gate, "gdn ffn_gate"),
                    (g.ffn_up, "gdn ffn_up"),
                    (g.ffn_down, "gdn ffn_down"),
                ];
                for (d, role) in weight_descs {
                    let kept = weight_dtype_kept_native(d.dtype);
                    let key = format!(
                        "{role} ({:?}{})",
                        d.dtype,
                        if kept { "" } else { " FALLBACK→F32" }
                    );
                    bump(
                        &mut stats,
                        &key,
                        d.n_bytes,
                        if kept { d.n_bytes } else { f32_size(&d.shape) },
                    );
                }
            }
            crate::loader::Block::Attn(a) => {
                let f32_descs: &[&TensorDesc] =
                    &[a.attn_norm, a.post_attention_norm, a.q_norm, a.k_norm];
                for d in f32_descs {
                    bump(
                        &mut stats,
                        "attn f32-required",
                        d.n_bytes,
                        f32_size(&d.shape),
                    );
                }
                let weight_descs: &[(&TensorDesc, &str)] = &[
                    (a.q, "attn q"),
                    (a.k, "attn k"),
                    (a.v, "attn v"),
                    (a.o, "attn o"),
                    (a.ffn_gate, "attn ffn_gate"),
                    (a.ffn_up, "attn ffn_up"),
                    (a.ffn_down, "attn ffn_down"),
                ];
                for (d, role) in weight_descs {
                    let kept = weight_dtype_kept_native(d.dtype);
                    let key = format!(
                        "{role} ({:?}{})",
                        d.dtype,
                        if kept { "" } else { " FALLBACK→F32" }
                    );
                    bump(
                        &mut stats,
                        &key,
                        d.n_bytes,
                        if kept { d.n_bytes } else { f32_size(&d.shape) },
                    );
                }
            }
        }
    }

    eprintln!("[ledger] role  count  gguf_MB  metal_MB  delta_MB");
    let mut total_gguf = 0u64;
    let mut total_metal = 0u64;
    for (role, (gguf_b, metal_b, count)) in &stats {
        let dg = *gguf_b as f64 / (1024.0 * 1024.0);
        let dm = *metal_b as f64 / (1024.0 * 1024.0);
        let delta = dm - dg;
        eprintln!(
            "[ledger]   {role:60} {count:4}  {dg:8.2}  {dm:8.2}  {:+.2}",
            delta
        );
        total_gguf += gguf_b;
        total_metal += metal_b;
    }
    let total_gguf_gb = total_gguf as f64 / (1024.0 * 1024.0 * 1024.0);
    let total_metal_gb = total_metal as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!("[ledger] === TOTALS ===");
    eprintln!("[ledger]   gguf  bytes: {total_gguf_gb:.2} GiB");
    eprintln!("[ledger]   metal bytes: {total_metal_gb:.2} GiB");
    eprintln!(
        "[ledger]   inflation:    {:+.2} GiB ({:+.1}% from quant fallbacks)",
        total_metal_gb - total_gguf_gb,
        (total_metal_gb / total_gguf_gb - 1.0) * 100.0
    );
    let bw_floor_native = total_gguf_gb * 1024.0 / 546.0; // ms at peak BW (note: GiB->GB unit fudge but consistent)
    let bw_floor_metal = total_metal_gb * 1024.0 / 546.0;
    eprintln!("[ledger]   bandwidth floor at GGUF native bytes: {bw_floor_native:.2} ms");
    eprintln!("[ledger]   bandwidth floor at Metal bytes:       {bw_floor_metal:.2} ms");
    eprintln!(
        "[ledger]   estimated cost of fallbacks: {:+.2} ms",
        bw_floor_metal - bw_floor_native
    );
}

/// **MTP tensor inventory**: scan the 27B GGUF for `mtp.*` tensors
/// to see what speculative-decoding state is shipped in the file.
/// Per the Qwen3.5/3.6 spec, the MTP head is a single decoder layer
/// with shared `embed_tokens` + `lm_head`. The released checkpoint
/// includes the trained MTP weights even though HF transformers
/// ignores them.
#[test]
#[ignore]
fn mtp_tensor_inventory() {
    let path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    if !std::path::Path::new(path).exists() {
        return;
    }
    let g = GgufFile::open(path).expect("open");
    let mtp_tensors: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("mtp") || t.name.contains(".mtp"))
        .collect();
    eprintln!("[mtp] found {} MTP-prefixed tensors:", mtp_tensors.len());
    let mut total_bytes = 0u64;
    for t in &mtp_tensors {
        eprintln!(
            "[mtp]   {:40} {:?}  shape={:?}  ({} bytes)",
            t.name, t.dtype, t.shape, t.n_bytes
        );
        total_bytes += t.n_bytes;
    }
    eprintln!(
        "[mtp] total MTP weight bytes: {:.2} MiB",
        total_bytes as f64 / (1024.0 * 1024.0)
    );
    // For comparison: also list the canonical "next" architecture key.
    for k in g
        .model
        .metadata()
        .keys()
        .filter(|k| k.contains("mtp") || k.contains("next") || k.contains("speculative"))
    {
        eprintln!("[mtp] metadata key: {k}");
    }
}

/// **Attn intra-layer profile**: same idea as the GDN intra
/// profiler but for full-attn blocks. Critical for the long-context
/// regression — tells us whether the cost lives in the score loop,
/// softmax, or V-aggregate inside attn_decode.
fn attn_intra_profile_single_block(
    mf: &MetalForward,
    attn_block_idx: usize,
    attn_idx_in_session: usize,
    position: u32,
    s: &mut MetalSession,
) -> Result<Vec<(String, f64)>, MfError> {
    let ab = match &mf.model.blocks[attn_block_idx] {
        MetalBlock::Attn(a) => a,
        _ => {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "attn_intra",
                detail: format!("block {attn_block_idx} is not attn"),
            }));
        }
    };
    let arch = &mf.model.arch;
    let h = arch.hidden_size as usize;
    let head_dim = arch.attn_head_dim as usize;
    let n_q = arch.n_q_heads as usize;
    let n_kv = arch.n_kv_heads as usize;
    let q_dim = n_q * head_dim;
    let kv_dim = n_kv * head_dim;
    let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
    let sigmoid_mul = decode_attn_sigmoid_mul_enabled();
    let rope_pair = decode_rope_pair_enabled();
    let fused_qk_norm_rope = decode_qk_norm_rope_fused_enabled() && rope_pair && sigmoid_mul;

    let mut phases: Vec<(String, f64)> = Vec::new();
    let timed = |label: &str,
                 cb: &dyn Fn(&KernelEncoder) -> Result<(), MfError>,
                 phases: &mut Vec<(String, f64)>|
     -> Result<(), MfError> {
        let cmd = mf.ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        cb(&enc)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        phases.push((label.into(), ms));
        Ok(())
    };

    // Pre-mixer norm.
    timed(
        "pre_norm (rms_norm)",
        &|enc| {
            encode_rms_norm_mul_f32(mf.ctx, enc, &s.x, &ab.attn_norm, &s.h, RMS_EPS)
                .map_err(MfError::from)
        },
        &mut phases,
    )?;
    // Q projection.
    timed(
        "q_proj_2x (mat_vec)",
        &|enc| encode_mat_vec_dispatch(mf.ctx, enc, &ab.q, &s.h, &s.attn_q_full, h, 2 * q_dim),
        &mut phases,
    )?;
    // Split Q + gate.
    // v0.432: default path skips split_q_gate (strided q-norm +
    // strided gate sigmoid_mul read the interleave directly); the
    // phase name is kept only for the rollback branch.
    if !sigmoid_mul {
        timed(
            "split_q_gate",
            &|enc| {
                encode_split_q_gate_f32(
                    mf.ctx,
                    enc,
                    &s.attn_q_full,
                    &s.attn_q,
                    &s.attn_gate,
                    n_q,
                    head_dim,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
    }
    // K, V projections.
    timed(
        "k_proj (mat_vec)",
        &|enc| encode_mat_vec_dispatch(mf.ctx, enc, &ab.k, &s.h, &s.attn_k_now, h, kv_dim),
        &mut phases,
    )?;
    timed(
        "v_proj (mat_vec)",
        &|enc| encode_mat_vec_dispatch(mf.ctx, enc, &ab.v, &s.h, &s.attn_v_now, h, kv_dim),
        &mut phases,
    )?;
    if fused_qk_norm_rope {
        timed(
            "qk_norm_rope (fused)",
            &|enc| {
                encode_qk_rms_norm_rope_f32_packed_consecutive(
                    mf.ctx,
                    enc,
                    &s.attn_q_full,
                    &ab.q_norm,
                    &s.attn_q_normed,
                    &s.attn_k_now,
                    &ab.k_norm,
                    &s.attn_k_normed,
                    1,
                    n_q,
                    n_kv,
                    head_dim,
                    n_rot,
                    position,
                    RMS_EPS,
                    arch.rope_theta,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
    } else {
        timed(
            "q_norm (batched rms)",
            &|enc| {
                if sigmoid_mul {
                    encode_rms_norm_batched_src_strided_f32(
                        mf.ctx,
                        enc,
                        &s.attn_q_full,
                        &ab.q_norm,
                        &s.attn_q_normed,
                        n_q,
                        head_dim,
                        2 * head_dim,
                        0,
                        RMS_EPS,
                    )
                    .map_err(MfError::from)
                } else {
                    encode_rms_norm_batched_f32(
                        mf.ctx,
                        enc,
                        &s.attn_q,
                        &ab.q_norm,
                        &s.attn_q_normed,
                        n_q,
                        head_dim,
                        RMS_EPS,
                    )
                    .map_err(MfError::from)
                }
            },
            &mut phases,
        )?;
        timed(
            "k_norm (batched rms)",
            &|enc| {
                encode_rms_norm_batched_f32(
                    mf.ctx,
                    enc,
                    &s.attn_k_now,
                    &ab.k_norm,
                    &s.attn_k_normed,
                    n_kv,
                    head_dim,
                    RMS_EPS,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        if rope_pair {
            timed(
                "rope Q+K (paired)",
                &|enc| {
                    encode_rope_neox_pair_f32(
                        mf.ctx,
                        enc,
                        &s.attn_q_normed,
                        &s.attn_k_normed,
                        n_q,
                        n_kv,
                        head_dim,
                        n_rot,
                        position,
                        arch.rope_theta,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        } else {
            timed(
                "rope Q",
                &|enc| {
                    encode_rope_neox_f32(
                        mf.ctx,
                        enc,
                        &s.attn_q_normed,
                        n_q,
                        head_dim,
                        n_rot,
                        position,
                        arch.rope_theta,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
            timed(
                "rope K",
                &|enc| {
                    encode_rope_neox_f32(
                        mf.ctx,
                        enc,
                        &s.attn_k_normed,
                        n_kv,
                        head_dim,
                        n_rot,
                        position,
                        arch.rope_theta,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
    }
    // KV scatter (fused K+V, 1 dispatch).
    timed(
        "kv scatter (fused)",
        &|enc| {
            encode_scatter_offset_f32_to_f16_kv(
                mf.ctx,
                enc,
                &s.attn_k_normed,
                &s.attn_v_now,
                &s.kv_k[attn_idx_in_session],
                &s.kv_v[attn_idx_in_session],
                (position as usize) * kv_dim,
                kv_dim,
            )
            .map_err(MfError::from)
        },
        &mut phases,
    )?;
    s.kv_n_pos[attn_idx_in_session] = position as usize + 1;
    // Attn decode: mirror the production dispatcher so the profiler
    // tracks the kernel path we actually ship.
    const V4_HEAD_DIM: usize = 256;
    let group = n_q / n_kv;
    let n_pos = s.kv_n_pos[attn_idx_in_session];
    let use_v4 = head_dim == V4_HEAD_DIM && matches!(group, 4 | 6 | 8 | 16);
    if use_v4 {
        let nwg = attn_v4_choose_nwg(n_pos, group);
        let tile_c = attn_v4_choose_tile_c(n_pos, group);
        timed(
            "attn_decode_v4_main",
            &|enc| {
                crate::metal::encode_attn_decode_v4_main_only_f32(
                    mf.ctx,
                    enc,
                    &s.attn_q_normed,
                    &s.kv_k[attn_idx_in_session],
                    &s.kv_v[attn_idx_in_session],
                    &s.attn_v4_o_partial,
                    &s.attn_v4_ml_partial,
                    n_q,
                    n_kv,
                    head_dim,
                    n_pos,
                    nwg,
                    tile_c,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
        timed(
            "attn_decode_v4_reduce",
            &|enc| {
                crate::metal::encode_attn_decode_v4_reduce_only_f32(
                    mf.ctx,
                    enc,
                    &s.attn_v4_o_partial,
                    &s.attn_v4_ml_partial,
                    &s.attn_o,
                    n_q,
                    n_kv,
                    head_dim,
                    nwg,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
    } else {
        timed(
            "attn_decode_f16kv",
            &|enc| {
                encode_attn_decode_f16kv_f32(
                    mf.ctx,
                    enc,
                    &s.attn_q_normed,
                    &s.kv_k[attn_idx_in_session],
                    &s.kv_v[attn_idx_in_session],
                    &s.attn_o,
                    n_q,
                    n_kv,
                    head_dim,
                    n_pos,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
    }
    // Sigmoid + mul (gated-attn).
    timed(
        "gate sigmoid + mul",
        &|enc| {
            if sigmoid_mul {
                encode_sigmoid_mul_gate_strided_f32(
                    mf.ctx,
                    enc,
                    &s.attn_q_full,
                    &s.attn_o,
                    &s.attn_o,
                    n_q,
                    head_dim,
                    2 * head_dim,
                    head_dim,
                )
                .map_err(MfError::from)
            } else {
                encode_sigmoid_f32(mf.ctx, enc, &s.attn_gate, &s.attn_q)?;
                encode_mul_f32(mf.ctx, enc, &s.attn_o, &s.attn_q, &s.attn_o).map_err(MfError::from)
            }
        },
        &mut phases,
    )?;
    // Output proj.
    timed(
        "o_proj (mat_vec)",
        &|enc| encode_mat_vec_dispatch(mf.ctx, enc, &ab.o, &s.attn_o, &s.mixer_out, q_dim, h),
        &mut phases,
    )?;
    // Residual #1.
    timed(
        "residual_add #1",
        &|enc| encode_add_inplace_f32(mf.ctx, enc, &s.x, &s.mixer_out).map_err(MfError::from),
        &mut phases,
    )?;
    // Pre-FFN norm.
    timed(
        "post_norm (rms_norm)",
        &|enc| {
            encode_rms_norm_mul_f32(mf.ctx, enc, &s.x, &ab.post_attn_norm, &s.h, RMS_EPS)
                .map_err(MfError::from)
        },
        &mut phases,
    )?;
    // FFN — mirror the production fused-or-fallback path from
    // encode_block, so the intra-profiler measurements track what
    // actually runs at decode. Q4_K + Q4_K weights take the fused
    // SwiGLU path (1 dispatch); other dtypes fall back to the
    // 3-dispatch sequence.
    let ffn_fused = ab.ffn_gate.dtype == GgmlType::Q4_K && ab.ffn_up.dtype == GgmlType::Q4_K;
    if ffn_fused {
        timed(
            "ffn_swiglu_q4_K (fused gate+up+silu_mul)",
            &|enc| {
                encode_ffn_swiglu_q4_K_f32(
                    mf.ctx,
                    enc,
                    &ab.ffn_gate,
                    &ab.ffn_up,
                    &s.h,
                    &s.ffn_inner,
                    h,
                    arch.intermediate_size as usize,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
    } else {
        timed(
            "ffn_gate (mat_vec) [unfused fallback]",
            &|enc| {
                encode_mat_vec_dispatch(
                    mf.ctx,
                    enc,
                    &ab.ffn_gate,
                    &s.h,
                    &s.ffn_gate,
                    h,
                    arch.intermediate_size as usize,
                )
            },
            &mut phases,
        )?;
        timed(
            "ffn_up (mat_vec) [unfused fallback]",
            &|enc| {
                encode_mat_vec_dispatch(
                    mf.ctx,
                    enc,
                    &ab.ffn_up,
                    &s.h,
                    &s.ffn_up,
                    h,
                    arch.intermediate_size as usize,
                )
            },
            &mut phases,
        )?;
        timed(
            "silu_mul [unfused fallback]",
            &|enc| {
                encode_silu_mul_f32(mf.ctx, enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)
                    .map_err(MfError::from)
            },
            &mut phases,
        )?;
    }
    timed(
        "ffn_down (mat_vec)",
        &|enc| {
            encode_mat_vec_dispatch(
                mf.ctx,
                enc,
                &ab.ffn_down,
                &s.ffn_inner,
                &s.ffn_out,
                arch.intermediate_size as usize,
                h,
            )
        },
        &mut phases,
    )?;
    timed(
        "residual_add #2",
        &|enc| encode_add_inplace_f32(mf.ctx, enc, &s.x, &s.ffn_out).map_err(MfError::from),
        &mut phases,
    )?;
    Ok(phases)
}

/// **GDN intra-layer profile**: split a single GDN block across its
/// 8+ logical sub-phases so we can attribute the ~0.64 ms/layer cost
/// to which sub-kernels. Free fn (not on MetalForward) because it
/// lives in the test module.
fn gdn_intra_profile_single_block(
    mf: &MetalForward,
    gdn_block_idx: usize,
    gdn_idx_in_session: usize,
    s: &mut MetalSession,
) -> Result<Vec<(String, f64)>, MfError> {
    let gb = match &mf.model.blocks[gdn_block_idx] {
        MetalBlock::Gdn(g) => g,
        _ => {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "gdn_intra_profile",
                detail: format!("block {gdn_block_idx} is not GDN"),
            }));
        }
    };
    let arch = &mf.model.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;

    let mut phases: Vec<(String, f64)> = Vec::new();
    let timed = |label: &str,
                 cb: &dyn Fn(&KernelEncoder) -> Result<(), MfError>,
                 phases: &mut Vec<(String, f64)>|
     -> Result<(), MfError> {
        let cmd = mf.ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        cb(&enc)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        phases.push((label.into(), ms));
        Ok(())
    };

    // Pre-mixer norm.
    timed(
        "pre_norm (rms_norm)",
        &|enc| {
            encode_rms_norm_mul_f32(mf.ctx, enc, &s.x, &gb.attn_norm, &s.h, RMS_EPS)
                .map_err(MfError::from)
        },
        &mut phases,
    )?;

    // QKV projection.
    timed(
        "in_proj_qkv (mat_vec)",
        &|enc| encode_mat_vec_dispatch(mf.ctx, enc, &gb.in_proj_qkv, &s.h, &s.gdn_qkv, h, conv_dim),
        &mut phases,
    )?;
    // z projection.
    timed(
        "in_proj_z (mat_vec)",
        &|enc| encode_mat_vec_dispatch(mf.ctx, enc, &gb.in_proj_z, &s.h, &s.gdn_z, h, v_dim),
        &mut phases,
    )?;
    // beta proj + sigmoid.
    timed(
        "beta_proj+sigmoid",
        &|enc| {
            encode_mat_vec_dispatch(mf.ctx, enc, &gb.beta_proj, &s.h, &s.gdn_b, h, n_v)?;
            encode_sigmoid_f32(mf.ctx, enc, &s.gdn_b, &s.gdn_beta).map_err(MfError::from)
        },
        &mut phases,
    )?;
    // alpha proj + decay-chain, matching production.
    timed(
        "alpha_proj+decay_chain",
        &|enc| {
            encode_mat_vec_dispatch(mf.ctx, enc, &gb.alpha_proj, &s.h, &s.gdn_a, h, n_v)?;
            encode_gdn_decay_chain_f32(mf.ctx, enc, &s.gdn_a, &gb.dt_bias, &gb.a_log, &s.gdn_alpha)
                .map_err(MfError::from)
        },
        &mut phases,
    )?;
    // ssm_conv (with internal silu).
    timed(
        "ssm_conv_silu",
        &|enc| {
            encode_ssm_conv_silu_f32(
                mf.ctx,
                enc,
                &s.gdn_qkv,
                &s.gdn_conv[gdn_idx_in_session],
                &gb.conv1d,
                &s.gdn_qkv_conv,
                conv_dim,
            )
            .map_err(MfError::from)
        },
        &mut phases,
    )?;
    // qkv split via zero-copy views (no dispatch). Per Jeff & Sanjay:
    // avoid copies / use indices instead of pointers.
    let q_view = s
        .gdn_qkv_conv
        .view_subrange(0, vec![(n_k * head_dim) as u64]);
    let k_view = s
        .gdn_qkv_conv
        .view_subrange((n_k * head_dim) as u64, vec![(n_k * head_dim) as u64]);
    let v_view = s
        .gdn_qkv_conv
        .view_subrange((2 * n_k * head_dim) as u64, vec![v_dim as u64]);
    timed(
        "l2_norm_qk (2 batched)",
        &|enc| {
            encode_l2_norm_batched_f32(
                mf.ctx,
                enc,
                &q_view,
                &s.gdn_q_norm,
                n_k,
                head_dim,
                RMS_EPS,
            )?;
            encode_l2_norm_batched_f32(mf.ctx, enc, &k_view, &s.gdn_k_norm, n_k, head_dim, RMS_EPS)
                .map_err(MfError::from)
        },
        &mut phases,
    )?;
    // gdn_step_decay: production recurrence with precomputed decay.
    timed(
        "gdn_step_decay (recurrence)",
        &|enc| {
            encode_gdn_step_decay_f32(
                mf.ctx,
                enc,
                &s.gdn_q_norm,
                &s.gdn_k_norm,
                &v_view,
                &s.gdn_alpha,
                &s.gdn_beta,
                &s.gdn_state[gdn_idx_in_session],
                &s.gdn_out,
                n_v,
                n_k,
                head_dim,
            )
            .map_err(MfError::from)
        },
        &mut phases,
    )?;
    // rmsnorm_gated.
    timed(
        "rmsnorm_gated",
        &|enc| {
            encode_rmsnorm_gated_f32(
                mf.ctx,
                enc,
                &s.gdn_out,
                &gb.norm,
                &s.gdn_z,
                &s.gdn_normed,
                n_v,
                head_dim,
                RMS_EPS * head_dim as f32,
            )
            .map_err(MfError::from)
        },
        &mut phases,
    )?;
    // out_proj.
    timed(
        "out_proj (mat_vec)",
        &|enc| {
            encode_mat_vec_dispatch(
                mf.ctx,
                enc,
                &gb.out_proj,
                &s.gdn_normed,
                &s.mixer_out,
                v_dim,
                h,
            )
        },
        &mut phases,
    )?;
    // residual #1.
    timed(
        "residual_add #1",
        &|enc| encode_add_inplace_f32(mf.ctx, enc, &s.x, &s.mixer_out).map_err(MfError::from),
        &mut phases,
    )?;
    // post-FFN norm.
    timed(
        "post_norm (rms_norm)",
        &|enc| {
            encode_rms_norm_mul_f32(mf.ctx, enc, &s.x, &gb.post_attn_norm, &s.h, RMS_EPS)
                .map_err(MfError::from)
        },
        &mut phases,
    )?;
    // FFN — mirror the production fused-or-fallback path.
    let ffn_fused = gb.ffn_gate.dtype == GgmlType::Q4_K && gb.ffn_up.dtype == GgmlType::Q4_K;
    if ffn_fused {
        timed(
            "ffn_swiglu_q4_K (fused gate+up+silu_mul)",
            &|enc| {
                encode_ffn_swiglu_q4_K_f32(
                    mf.ctx,
                    enc,
                    &gb.ffn_gate,
                    &gb.ffn_up,
                    &s.h,
                    &s.ffn_inner,
                    h,
                    arch.intermediate_size as usize,
                )
                .map_err(MfError::from)
            },
            &mut phases,
        )?;
    } else {
        timed(
            "ffn_gate (mat_vec) [unfused fallback]",
            &|enc| {
                encode_mat_vec_dispatch(
                    mf.ctx,
                    enc,
                    &gb.ffn_gate,
                    &s.h,
                    &s.ffn_gate,
                    h,
                    arch.intermediate_size as usize,
                )
            },
            &mut phases,
        )?;
        timed(
            "ffn_up (mat_vec) [unfused fallback]",
            &|enc| {
                encode_mat_vec_dispatch(
                    mf.ctx,
                    enc,
                    &gb.ffn_up,
                    &s.h,
                    &s.ffn_up,
                    h,
                    arch.intermediate_size as usize,
                )
            },
            &mut phases,
        )?;
        timed(
            "silu_mul [unfused fallback]",
            &|enc| {
                encode_silu_mul_f32(mf.ctx, enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)
                    .map_err(MfError::from)
            },
            &mut phases,
        )?;
    }
    timed(
        "ffn_down (mat_vec)",
        &|enc| {
            encode_mat_vec_dispatch(
                mf.ctx,
                enc,
                &gb.ffn_down,
                &s.ffn_inner,
                &s.ffn_out,
                arch.intermediate_size as usize,
                h,
            )
        },
        &mut phases,
    )?;
    // residual #2.
    timed(
        "residual_add #2",
        &|enc| encode_add_inplace_f32(mf.ctx, enc, &s.x, &s.ffn_out).map_err(MfError::from),
        &mut phases,
    )?;
    Ok(phases)
}

/// Split one MoE block into mixer, route, routed expert FFN, shared FFN,
/// and residual pieces. This is profiler-only; production keeps these in a
/// single command buffer for normal decode.
fn moe_intra_profile_single_block(
    mf: &MetalForward,
    block_idx: usize,
    mixer_slot: MixerSlot,
    position: u32,
    s: &mut MetalSession,
) -> Result<Vec<(String, f64)>, MfError> {
    let block = &mf.model.blocks[block_idx];
    let (ffn_gate, ffn_up, ffn_down, moe) = match block {
        MetalBlock::Gdn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
        MetalBlock::Attn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
    };
    let moe = moe.ok_or(MfError::UnsupportedMoe)?;
    let arch = &mf.model.arch;
    let h = arch.hidden_size as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let f_shared = arch.expert_shared_feed_forward_length as usize;
    let n_expert = arch.expert_count as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;

    let mut phases: Vec<(String, f64)> = Vec::new();
    let timed = |label: &str,
                 cb: &dyn Fn(&KernelEncoder) -> Result<(), MfError>,
                 phases: &mut Vec<(String, f64)>|
     -> Result<(), MfError> {
        let cmd = mf.ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        cb(&enc)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        phases.push((label.into(), ms));
        Ok(())
    };
    let timed_mut = |label: &str,
                     cb: &mut dyn FnMut(&KernelEncoder) -> Result<(), MfError>,
                     phases: &mut Vec<(String, f64)>|
     -> Result<(), MfError> {
        let cmd = mf.ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        cb(&enc)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        phases.push((label.into(), ms));
        Ok(())
    };

    timed_mut(
        "mixer_prep (norm+mixer+resid+postnorm)",
        &mut |enc| mf.encode_moe_mixer_prep(enc, block, mixer_slot, position, s),
        &mut phases,
    )?;
    timed_mut(
        "route_prepare (router+topk+shared_gate)",
        &mut |enc| mf.encode_moe_route_prepare(enc, s, moe),
        &mut phases,
    )?;

    let moe_inner = s.moe_inner.view_subrange(0, vec![(topk * f_exp) as u64]);
    let moe_expert_out = s.moe_expert_out.view_subrange(0, vec![(topk * h) as u64]);
    let topk_idx = s.moe_topk_idx.view_subrange(0, vec![topk as u64]);
    let topk_w = s.moe_topk_weight.view_subrange(0, vec![topk as u64]);

    match moe.gate_exps.dtype {
        GgmlType::Q4_K => {
            timed(
                "routed_gate_up_swiglu_q4_K",
                &|enc| {
                    encode_moe_swiglu_q4_K_f32(
                        mf.ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &s.h,
                        &topk_idx,
                        &moe_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
        GgmlType::Q5_K => {
            let gate_pack = s
                .moe_expert_out
                .view_subrange(0, vec![(topk * f_exp) as u64]);
            let up_pack = s
                .moe_expert_out
                .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
            timed(
                "routed_gate_q5_K",
                &|enc| {
                    encode_moe_mat_vec_q5_K_f32(
                        mf.ctx,
                        enc,
                        &moe.gate_exps,
                        &s.h,
                        &topk_idx,
                        &gate_pack,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
            timed(
                "routed_up_q5_K",
                &|enc| {
                    encode_moe_mat_vec_q5_K_f32(
                        mf.ctx,
                        enc,
                        &moe.up_exps,
                        &s.h,
                        &topk_idx,
                        &up_pack,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
            timed(
                "routed_silu_mul",
                &|enc| {
                    encode_silu_mul_f32(mf.ctx, enc, &gate_pack, &up_pack, &moe_inner)
                        .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
        GgmlType::IQ4_XS => {
            timed(
                "routed_gate_up_swiglu_iq4_xs",
                &|enc| {
                    encode_moe_swiglu_iq4_xs_f32(
                        mf.ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &s.h,
                        &topk_idx,
                        &moe_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
        GgmlType::F32 => {
            let gate_pack = s
                .moe_expert_out
                .view_subrange(0, vec![(topk * f_exp) as u64]);
            let up_pack = s
                .moe_expert_out
                .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
            timed(
                "routed_gate_f32",
                &|enc| {
                    encode_moe_mat_vec_f32(
                        mf.ctx,
                        enc,
                        &moe.gate_exps,
                        &s.h,
                        &topk_idx,
                        &gate_pack,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
            timed(
                "routed_up_f32",
                &|enc| {
                    encode_moe_mat_vec_f32(
                        mf.ctx,
                        enc,
                        &moe.up_exps,
                        &s.h,
                        &topk_idx,
                        &up_pack,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
            timed(
                "routed_silu_mul",
                &|enc| {
                    encode_silu_mul_f32(mf.ctx, enc, &gate_pack, &up_pack, &moe_inner)
                        .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
        dtype => {
            return Err(MfError::UnsupportedDtype {
                name: "MoE routed gate/up expert banks".into(),
                dtype,
            });
        }
    }

    match moe.down_exps.dtype {
        GgmlType::Q4_K => {
            timed(
                "routed_down_q4_K",
                &|enc| {
                    encode_moe_down_q4_K_f32(
                        mf.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
            timed(
                "routed_weighted_sum",
                &|enc| {
                    encode_moe_weighted_sum_f32(
                        mf.ctx,
                        enc,
                        &moe_expert_out,
                        &topk_w,
                        &s.mixer_out,
                        h,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
        GgmlType::Q5_K => {
            if decode_moe_q5_down_fused_enabled() {
                timed(
                    "routed_down_weighted_sum_q5_K",
                    &|enc| {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                            mf.ctx,
                            enc,
                            &moe.down_exps,
                            &moe_inner,
                            &topk_idx,
                            &topk_w,
                            &s.mixer_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            1,
                        )
                        .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
            } else {
                timed(
                    "routed_down_q5_K",
                    &|enc| {
                        encode_moe_down_q5_K_f32(
                            mf.ctx,
                            enc,
                            &moe.down_exps,
                            &moe_inner,
                            &topk_idx,
                            &moe_expert_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                        )
                        .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
                timed(
                    "routed_weighted_sum",
                    &|enc| {
                        encode_moe_weighted_sum_f32(
                            mf.ctx,
                            enc,
                            &moe_expert_out,
                            &topk_w,
                            &s.mixer_out,
                            h,
                            topk,
                        )
                        .map_err(MfError::from)
                    },
                    &mut phases,
                )?;
            }
        }
        GgmlType::Q6_K => {
            timed(
                "routed_down_weighted_sum_q6_K",
                &|enc| {
                    encode_moe_down_weighted_sum_q6_K_f32(
                        mf.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &topk_w,
                        &s.mixer_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
        GgmlType::IQ4_XS => {
            timed(
                "routed_down_iq4_xs",
                &|enc| {
                    encode_moe_down_iq4_xs_f32(
                        mf.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
            timed(
                "routed_weighted_sum",
                &|enc| {
                    encode_moe_weighted_sum_f32(
                        mf.ctx,
                        enc,
                        &moe_expert_out,
                        &topk_w,
                        &s.mixer_out,
                        h,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
        GgmlType::IQ4_NL => {
            timed(
                "routed_down_iq4_nl",
                &|enc| {
                    encode_moe_down_iq4_nl_f32(
                        mf.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
            timed(
                "routed_weighted_sum",
                &|enc| {
                    encode_moe_weighted_sum_f32(
                        mf.ctx,
                        enc,
                        &moe_expert_out,
                        &topk_w,
                        &s.mixer_out,
                        h,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
        GgmlType::F32 => {
            timed(
                "routed_down_f32",
                &|enc| {
                    encode_moe_down_f32_f32(
                        mf.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
            timed(
                "routed_weighted_sum",
                &|enc| {
                    encode_moe_weighted_sum_f32(
                        mf.ctx,
                        enc,
                        &moe_expert_out,
                        &topk_w,
                        &s.mixer_out,
                        h,
                        topk,
                    )
                    .map_err(MfError::from)
                },
                &mut phases,
            )?;
        }
        dtype => {
            return Err(MfError::UnsupportedDtype {
                name: "MoE routed down expert bank".into(),
                dtype,
            });
        }
    }

    let shared_gate_tmp = s.ffn_gate.view_subrange(0, vec![f_shared as u64]);
    let shared_up_tmp = s.ffn_up.view_subrange(0, vec![f_shared as u64]);
    let shared_inner_tmp = s.ffn_inner.view_subrange(0, vec![f_shared as u64]);
    let shared_out_tmp = s.ffn_out.view_subrange(0, vec![h as u64]);
    timed(
        "shared_gate (mat_vec)",
        &|enc| encode_mat_vec_dispatch(mf.ctx, enc, ffn_gate, &s.h, &shared_gate_tmp, h, f_shared),
        &mut phases,
    )?;
    timed(
        "shared_up (mat_vec)",
        &|enc| encode_mat_vec_dispatch(mf.ctx, enc, ffn_up, &s.h, &shared_up_tmp, h, f_shared),
        &mut phases,
    )?;
    timed(
        "shared_silu_mul",
        &|enc| {
            encode_silu_mul_f32(
                mf.ctx,
                enc,
                &shared_gate_tmp,
                &shared_up_tmp,
                &shared_inner_tmp,
            )
            .map_err(MfError::from)
        },
        &mut phases,
    )?;
    timed(
        "shared_down (mat_vec)",
        &|enc| {
            encode_mat_vec_dispatch(
                mf.ctx,
                enc,
                ffn_down,
                &shared_inner_tmp,
                &shared_out_tmp,
                f_shared,
                h,
            )
        },
        &mut phases,
    )?;
    timed(
        "shared_axpy_scalar",
        &|enc| {
            encode_axpy_scalar_f32(
                mf.ctx,
                enc,
                &shared_out_tmp,
                &s.moe_shared_gate,
                &s.mixer_out,
            )
            .map_err(MfError::from)
        },
        &mut phases,
    )?;
    timed(
        "residual_add #2",
        &|enc| encode_add_inplace_f32(mf.ctx, enc, &s.x, &s.mixer_out).map_err(MfError::from),
        &mut phases,
    )?;

    Ok(phases)
}

/// **Phase-resolved profile**: at each context length we care about,
/// run a phase-split forward to attribute GPU time to logical phases.
/// This is the experiment that should tell us whether the long-context
/// regression lives in attn layers, GDN layers, or somewhere else.
#[test]
#[ignore]
fn metal_27b_phase_profile() {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    let max_n = 4096usize;
    let mf = MetalForward::new(&ctx, &mm);
    // Single warmup session for pipeline cache.
    {
        let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("warmup");
        for i in 0..3 {
            let _ = mf.single_token(0, i as u32, &mut s).expect("warmup");
        }
    }

    for &target in &[1usize, 1024, 4096] {
        let mut s = MetalSession::fresh(&ctx, &mm, max_n + 16).expect("session");
        // Ramp to `target` positions (no timing).
        for p in 0..(target as u32) {
            let _ = mf.single_token(0, p, &mut s).expect("ramp");
        }
        // Now do a phase-profiled call at position `target`.
        let (_logits, wall_with_artifact_ms, phases) = mf
            .single_token_phase_profiled(0, target as u32, &mut s)
            .expect("phase");

        let phase_sum_ms: f64 = phases.iter().map(|p| p.1).sum();
        // ★ phase_sum is the production-realistic GPU time; the wall
        // includes ~12 ms of per-phase cmdbuf overhead (artifact).
        // For production ms/token use metal_27b_context_sweep instead.
        eprintln!(
            "[phase ctx={target:>5}] phase_sum {phase_sum_ms:.2} ms (production-realistic) | \
             wall_with_artifact {wall_with_artifact_ms:.2} ms (DO NOT use for prod ms/token)"
        );
        for (name, ms) in &phases {
            let pct = ms / phase_sum_ms * 100.0;
            eprintln!("[phase ctx={target:>5}]   {name:25} {ms:7.2} ms  ({pct:5.1}%)");
        }
    }
}

/// **Attn intra-layer breakdown across context lengths**. Tells us
/// which sub-phase of attention scales with n_pos. Critical: only
/// `attn_decode` should grow with context; everything else should
/// be flat. If something else grows, that's an unexpected scaling
/// problem.
#[test]
#[ignore]
fn metal_27b_attn_intra_profile() {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    // Warmup.
    {
        let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
        for i in 0..3 {
            let _ = mf.single_token(0, i as u32, &mut s).expect("warmup");
        }
    }

    // Find first attn block index (0.8B has it at block 3 — let me find it
    // for 27B). 27B has pattern [GDN×3, attn×1] × 16, so block 3 is the
    // first attn block; 27B has 16 attn blocks total.
    let attn_block_idx = 3usize;

    for &target in &[1usize, 1024, 4096] {
        let mut s = MetalSession::fresh(&ctx, &mm, target + 32).expect("session");
        // Ramp to `target`.
        for p in 0..(target as u32) {
            let _ = mf.single_token(0, p, &mut s).expect("ramp");
        }
        // Profile one attn block in isolation, averaging 5 runs.
        let n_runs = 5usize;
        let mut agg: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
        let mut order: Vec<String> = Vec::new();
        for run in 0..n_runs {
            // Position must increment with each call (for KV scatter).
            let pos = target as u32 + run as u32;
            let phases = attn_intra_profile_single_block(&mf, attn_block_idx, 0, pos, &mut s)
                .expect("attn-intra");
            for (name, ms) in phases {
                if run == 0 {
                    order.push(name.clone());
                }
                *agg.entry(name).or_default() += ms;
            }
        }
        let total: f64 = agg.values().sum::<f64>() / n_runs as f64;
        eprintln!(
            "[attn-intra ctx={target}] one ATTN layer total: {total:.3} ms (×16 = {:.2} ms)",
            total * 16.0
        );
        for name in &order {
            let avg_ms = agg[name] / n_runs as f64;
            let pct = avg_ms / total * 100.0;
            eprintln!("[attn-intra ctx={target}]   {name:35} {avg_ms:6.3} ms  ({pct:5.1}%)");
        }
        eprintln!();
    }
}

/// **GDN intra-layer breakdown**: profile a single GDN block by
/// sub-phase. Tells us where the ~0.64 ms/layer cost lives —
/// which sub-kernels are big, which are negligible, and which
/// fusion targets are worth pursuing.
#[test]
#[ignore]
fn metal_27b_gdn_intra_profile() {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    // Warmup pipeline cache.
    let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
    for i in 0..3 {
        let _ = mf.single_token(0, i as u32, &mut s).expect("warmup");
    }

    // Run a real forward to populate state, then profile block 0 (a
    // GDN block).
    let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
    let _ = mf.single_token(9419, 0, &mut s).expect("p0");

    // Profile just one GDN block in isolation. Aggregate across
    // 5 runs to get noise-floor stable numbers.
    let n_runs = 5usize;
    let mut agg: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for run in 0..n_runs {
        let phases = gdn_intra_profile_single_block(&mf, 0, 0, &mut s).expect("intra");
        for (name, ms) in phases {
            if run == 0 {
                order.push(name.clone());
            }
            *agg.entry(name).or_default() += ms;
        }
    }
    let total: f64 = agg.values().sum::<f64>() / n_runs as f64;
    eprintln!("[gdn-intra] === per-sub-phase breakdown (avg of {n_runs} runs) ===");
    eprintln!("[gdn-intra] one GDN layer total: {total:.3} ms");
    for name in &order {
        let avg_ms = agg[name] / n_runs as f64;
        let pct = avg_ms / total * 100.0;
        eprintln!("[gdn-intra]   {name:35} {avg_ms:6.3} ms  ({pct:5.1}%)");
    }
    eprintln!(
        "[gdn-intra] extrapolated to 48 layers: {:.2} ms",
        total * 48.0
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_gdn_intra_profile() {
    let model_path = crate::test_fixtures::A3B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
    for i in 0..3 {
        let _ = mf.single_token(0, i as u32, &mut s).expect("warmup");
    }

    let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
    let _ = mf.single_token(0, 0, &mut s).expect("p0");

    let n_runs = 8usize;
    let mut agg: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for run in 0..n_runs {
        let phases = gdn_intra_profile_single_block(&mf, 0, 0, &mut s).expect("intra");
        for (name, ms) in phases {
            if run == 0 {
                order.push(name.clone());
            }
            *agg.entry(name).or_default() += ms;
        }
    }
    let total: f64 = agg.values().sum::<f64>() / n_runs as f64;
    eprintln!("[gdn-intra-a3b] === per-sub-phase breakdown (avg of {n_runs} runs) ===");
    eprintln!("[gdn-intra-a3b] one GDN layer total: {total:.3} ms");
    for name in &order {
        let avg_ms = agg[name] / n_runs as f64;
        let pct = avg_ms / total * 100.0;
        eprintln!("[gdn-intra-a3b]   {name:35} {avg_ms:6.3} ms  ({pct:5.1}%)");
    }
    eprintln!(
        "[gdn-intra-a3b] extrapolated to 30 layers: {:.2} ms",
        total * 30.0
    );
}

fn run_moe_intra_profile(model_path: &str, label: &str, n_runs: usize) {
    if !std::path::Path::new(model_path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
    for i in 0..3 {
        let _ = mf.single_token(0, i as u32, &mut s).expect("warmup");
    }

    let mut s = MetalSession::fresh(&ctx, &mm, 32).expect("session");
    let _ = mf.single_token(0, 0, &mut s).expect("p0");
    let slot = match &mf.model.blocks[0] {
        MetalBlock::Gdn(_) => MixerSlot::Gdn(0),
        MetalBlock::Attn(_) => MixerSlot::Attn(0),
    };

    let mut agg: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for run in 0..n_runs {
        let phases =
            moe_intra_profile_single_block(&mf, 0, slot, run as u32, &mut s).expect("moe-intra");
        for (name, ms) in phases {
            if run == 0 {
                order.push(name.clone());
            }
            *agg.entry(name).or_default() += ms;
        }
    }
    let total: f64 = agg.values().sum::<f64>() / n_runs as f64;
    eprintln!("[moe-intra-{label}] === per-sub-phase breakdown (avg of {n_runs} runs) ===");
    eprintln!("[moe-intra-{label}] one MoE block total: {total:.3} ms");
    for name in &order {
        let avg_ms = agg[name] / n_runs as f64;
        let pct = avg_ms / total * 100.0;
        eprintln!("[moe-intra-{label}]   {name:42} {avg_ms:6.3} ms  ({pct:5.1}%)");
    }
    eprintln!(
        "[moe-intra-{label}] extrapolated to {} blocks: {:.2} ms",
        mf.model.blocks.len(),
        total * mf.model.blocks.len() as f64
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_moe_intra_profile() {
    run_moe_intra_profile(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b", 6);
}

#[test]
#[ignore]
fn metal_122b_a10b_moe_intra_profile() {
    run_moe_intra_profile(crate::test_fixtures::A10B_Q4_K_XL.path(), "122b", 4);
}

/// **Context-length sweep**: how does decode throughput scale as the
/// KV cache and GDN state grow? llama-bench's `tg128` is at fixed
/// position 0..127. We sweep further to see where the cliffs are.
///
/// Drives 1, 64, 256, 1024, 4096 tokens and reports per-token cost
/// at each prefix length. The KV cache grows linearly with context
/// (16 attn layers × 64 KB / token), so attn_decode kernel
/// time should grow linearly too. GDN state is fixed-size so GDN
/// layer cost is invariant.
#[test]
#[ignore]
fn metal_27b_context_sweep() {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    // Sweep targets. Naive attn_decode_f32 / f16kv had a hard cap at
    // ~7000 positions (28KB threadgroup memory ÷ 4 B/score). v4
    // (online softmax + split-K) unlocks arbitrary context.
    // 16384 ramp + window costs ~5 minutes; trim if quick iteration is
    // needed.
    let checkpoints = [1usize, 64, 256, 1024, 4096, 8192, 16384];

    let max_n = *checkpoints.iter().max().unwrap();
    let mut s = MetalSession::fresh(&ctx, &mm, max_n + 16).expect("session");
    let mf = MetalForward::new(&ctx, &mm);

    // Warmup pipeline state cache.
    for i in 0..3 {
        let _ = mf.single_token(0, i as u32, &mut s).expect("warmup");
    }
    let mut s = MetalSession::fresh(&ctx, &mm, max_n + 16).expect("session2");
    // One pre-warmed token at position 0 to populate everything.
    let _ = mf.single_token(0, 0, &mut s).expect("p0");

    eprintln!("[ctx-sweep] === per-token decode cost vs context ===");
    eprintln!("[ctx-sweep] context  total_ms  gpu_ms  cpu_enc_ms  t/s   GB/s   %peak");

    let mut prev_pos = 1u32;
    for &target in &checkpoints {
        // Ramp KV cache + GDN state to `target` positions.
        // For positions 1..target we don't need to time; just need them
        // populated. Use any token id (0).
        for p in prev_pos..(target as u32) {
            let _ = mf.single_token(0, p, &mut s).expect("ramp");
        }
        prev_pos = target as u32;

        // Time a window at this context length.
        const WINDOW: usize = 5;
        let mut samples = Vec::with_capacity(WINDOW);
        for i in 0..WINDOW {
            let pos = prev_pos + i as u32;
            let (_, p) = mf.single_token_profiled(0, pos, &mut s).expect("timed");
            samples.push(p);
        }
        prev_pos += WINDOW as u32;

        let avg_total = samples.iter().map(|p| p.total_ms).sum::<f64>() / WINDOW as f64;
        let avg_gpu = samples.iter().map(|p| p.gpu_kernel_ms).sum::<f64>() / WINDOW as f64;
        let avg_enc = samples.iter().map(|p| p.cpu_encode_ms).sum::<f64>() / WINDOW as f64;
        let bw = 16.8_f64 / (avg_gpu / 1000.0); // model-only bytes
        eprintln!(
            "[ctx-sweep] {target:>7}  {avg_total:>8.2}  {avg_gpu:>6.2}  {avg_enc:>10.2}  {:>4.1}  {bw:>5.0}   {:>4.0}%",
            1000.0 / avg_total,
            bw / 5.46
        );
    }
}

/// **Per-token profiling on 27B-Q4_K_M.** Runs N steady-state tokens,
/// reports the CPU-encode / GPU-kernel / total-wall split, and the
/// dispatch count. The data we feed to optimization decisions.
#[test]
#[ignore]
fn metal_27b_perf_profile() {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let ids = tok
        .encode(
            "The quick brown fox jumps over the lazy dog and runs into the field where",
            false,
        )
        .expect("tokenize");
    eprintln!("[perf-27b] {} prompt tokens", ids.len());

    let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 32).expect("session");
    let mf = MetalForward::new(&ctx, &mm);

    // Warmup: enough tokens to fully populate pipeline cache.
    for (i, &tid) in ids.iter().take(3).enumerate() {
        let _ = mf.single_token(tid, i as u32, &mut s).expect("warmup");
    }
    // Reset session for a clean steady-state run.
    let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 32).expect("session2");
    // Re-warmup PSO cache by running once.
    let _ = mf.single_token(ids[0], 0, &mut s).expect("warmup2");
    let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 32).expect("session3");

    let mut profiles: Vec<TokenProfile> = Vec::new();
    for (i, &tid) in ids.iter().enumerate() {
        let (_, p) = mf
            .single_token_profiled(tid, i as u32, &mut s)
            .expect("forward");
        profiles.push(p);
    }

    // Skip the first to avoid first-call jitter.
    let steady = &profiles[1..];
    let avg = |f: fn(&TokenProfile) -> f64| -> f64 {
        steady.iter().map(f).sum::<f64>() / steady.len() as f64
    };
    let total = avg(|p| p.total_ms);
    let cpu_enc = avg(|p| p.cpu_encode_ms);
    let gpu_kern = avg(|p| p.gpu_kernel_ms);
    let cpu_gpu = avg(|p| p.cpu_to_gpu_complete_ms);
    let queue_overhead = cpu_gpu - gpu_kern;
    let readback_etc = total - cpu_enc - cpu_gpu;

    eprintln!(
        "[perf-27b] === avg over {} steady-state tokens ===",
        steady.len()
    );
    eprintln!(
        "[perf-27b]   total wall:           {total:.2} ms = {:.2} t/s",
        1000.0 / total
    );
    eprintln!(
        "[perf-27b]   cpu encode:           {cpu_enc:.2} ms ({:.0}%)",
        cpu_enc / total * 100.0
    );
    eprintln!(
        "[perf-27b]   gpu kernels:          {gpu_kern:.2} ms ({:.0}%)",
        gpu_kern / total * 100.0
    );
    eprintln!(
        "[perf-27b]   queue/sched overhead: {queue_overhead:.2} ms ({:.0}%)",
        queue_overhead / total * 100.0
    );
    eprintln!(
        "[perf-27b]   readback + misc:      {readback_etc:.2} ms ({:.0}%)",
        readback_etc / total * 100.0
    );

    // Theoretical bandwidth-bound floor for this model: 16.8 GB / 546 GB/s
    // = 30.7 ms. So gpu_kernel_ms tells us how close we are to the BW wall.
    let gb = 16.8_f64;
    let peak = 546.0_f64;
    let bw_floor = gb / peak * 1000.0;
    eprintln!(
        "[perf-27b]   bandwidth floor:      {bw_floor:.2} ms ({:.0} GB/s peak; we're at {:.0} GB/s = {:.0}%)",
        peak,
        gb / (gpu_kern / 1000.0),
        gb / (gpu_kern / 1000.0) / peak * 100.0
    );
    // llama.cpp clean baseline: 21.21 t/s = 47.1 ms/token.
    eprintln!("[perf-27b]   llama.cpp baseline:   47.15 ms (21.21 t/s)");
    eprintln!(
        "[perf-27b]   our headroom to BW floor: {:.2} ms",
        gpu_kern - bw_floor
    );
    eprintln!(
        "[perf-27b]   our headroom to llama.cpp: {:.2} ms ({:+.1} t/s)",
        total - 47.15,
        1000.0 / total - 21.21
    );

    eprintln!("[perf-27b] per-token profiles:");
    for (i, p) in profiles.iter().enumerate() {
        eprintln!(
            "[perf-27b]   t{i}: total={:.2} cpu_enc={:.2} gpu={:.2} q={:.2}",
            p.total_ms,
            p.cpu_encode_ms,
            p.gpu_kernel_ms,
            p.cpu_to_gpu_complete_ms - p.gpu_kernel_ms
        );
    }
}

/// **27B Q4_K_M, multi-token**: validates position > 0 + steady-state
/// throughput. The headline number we've been working toward.
#[test]
#[ignore]
fn metal_27b_multi_token_perf() {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    let oracle_path = "/tmp/qwen-oracle/longprompt_27b.f32";
    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists() {
        eprintln!("[metal-27b-multi] skipped — fixtures missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };

    let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
    let n = oracle_bytes.len() / 4;
    let oracle: Vec<f32> = (0..n)
        .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();

    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let ids = tok
        .encode("The quick brown fox jumps over the lazy dog", false)
        .expect("tokenize");
    eprintln!("[metal-27b-multi] {} tokens: {ids:?}", ids.len());
    assert_eq!(ids.len(), 9);

    let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session");
    let mf = MetalForward::new(&ctx, &mm);

    // Warmup pass to compile pipeline state objects + warm caches.
    let _ = mf.single_token(ids[0], 0, &mut s).expect("warmup");
    // Reset session for the actual run.
    let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session2");

    let t = std::time::Instant::now();
    let mut last = vec![];
    let mut per_token_ms: Vec<f64> = Vec::new();
    for (i, &tid) in ids.iter().enumerate() {
        let tt = std::time::Instant::now();
        last = mf.single_token(tid, i as u32, &mut s).expect("forward");
        per_token_ms.push(tt.elapsed().as_secs_f64() * 1e3);
    }
    let total_ms = t.elapsed().as_secs_f64() * 1e3;

    let mut max_abs = 0.0f32;
    let mut argmax_ours = 0usize;
    let mut argmax_oracle = 0usize;
    let mut max_ours = f32::NEG_INFINITY;
    let mut max_oracle = f32::NEG_INFINITY;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let d = (last[i] - oracle[i]).abs();
        max_abs = max_abs.max(d);
        if last[i] > max_ours {
            max_ours = last[i];
            argmax_ours = i;
        }
        if oracle[i] > max_oracle {
            max_oracle = oracle[i];
            argmax_oracle = i;
        }
        dot += last[i] as f64 * oracle[i] as f64;
        na += (last[i] as f64).powi(2);
        nb += (oracle[i] as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    eprintln!(
        "[metal-27b-multi] {total_ms:.1}ms total, {:.1}ms/token (avg) — argmax: ours={argmax_ours} ({:.4}) | oracle={argmax_oracle} ({:.4}) | max|Δ|={max_abs:.4} cos={cos:.6}",
        total_ms / ids.len() as f64,
        max_ours,
        max_oracle
    );
    eprintln!("[metal-27b-multi] per-token (ms): {per_token_ms:?}");
    let avg_excl_first = per_token_ms[1..].iter().sum::<f64>() / (per_token_ms.len() - 1) as f64;
    eprintln!(
        "[metal-27b-multi] steady-state (excl. first): {avg_excl_first:.1} ms/token = {:.2} t/s",
        1000.0 / avg_excl_first
    );
    assert_eq!(argmax_ours, argmax_oracle, "argmax disagreement");
    assert!(cos > 0.999, "cos={cos} below threshold");
}

/// **End-to-end Metal forward, multi-token**. Exercises position > 0
/// in the attn block (RoPE, KV cache reads at multiple positions).
/// Oracle: llm/llama_core's snapshot dump for "The quick brown fox
/// jumps over the lazy dog" (9 tokens), Qwen3.5-0.8B-F32.
#[test]
fn metal_multi_token_matches_cpu_oracle() {
    let model_path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    let oracle_path = "/tmp/qwen-oracle/longprompt_t0.f32";
    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists() {
        eprintln!("[metal-e2e-multi] skipped — fixtures missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };

    let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
    let n = oracle_bytes.len() / 4;
    let oracle: Vec<f32> = (0..n)
        .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();

    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let ids = tok
        .encode("The quick brown fox jumps over the lazy dog", false)
        .expect("tokenize");
    eprintln!("[metal-e2e-multi] {} tokens: {ids:?}", ids.len());
    assert_eq!(ids.len(), 9);

    let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session");
    let mf = MetalForward::new(&ctx, &mm);
    let t = std::time::Instant::now();
    let mut last = vec![];
    for (i, &tid) in ids.iter().enumerate() {
        last = mf.single_token(tid, i as u32, &mut s).expect("forward");
    }
    let total_ms = t.elapsed().as_secs_f64() * 1e3;

    let mut max_abs = 0.0f32;
    let mut argmax_ours = 0usize;
    let mut argmax_oracle = 0usize;
    let mut max_ours = f32::NEG_INFINITY;
    let mut max_oracle = f32::NEG_INFINITY;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let d = (last[i] - oracle[i]).abs();
        max_abs = max_abs.max(d);
        if last[i] > max_ours {
            max_ours = last[i];
            argmax_ours = i;
        }
        if oracle[i] > max_oracle {
            max_oracle = oracle[i];
            argmax_oracle = i;
        }
        dot += last[i] as f64 * oracle[i] as f64;
        na += (last[i] as f64).powi(2);
        nb += (oracle[i] as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    eprintln!(
        "[metal-e2e-multi] {total_ms:.1}ms ({:.1}ms/token) — argmax: ours={argmax_ours} ({:.4}) | oracle={argmax_oracle} ({:.4}) | max|Δ|={max_abs:.4} cos={cos:.6}",
        total_ms / ids.len() as f64,
        max_ours,
        max_oracle
    );
    assert_eq!(argmax_ours, argmax_oracle, "argmax disagreement");
    assert!(cos > 0.9999, "cos={cos} below threshold");
}

/// **v0.75.0 correctness gate**: skip-tail prefill must produce
/// bit-identical session state to the full-tail path. Two sessions
/// run the same 9-token prompt: session A goes through `single_token`
/// for every token, session B goes through `single_token_no_tail`
/// for tokens [0..n-1) and `single_token` for the last token. Final
/// logits MUST match bit-exactly (same forward path through embed +
/// blocks + final norm + lm_head; the no_tail path just skips work
/// that doesn't feed back into the next iteration).
///
/// Equally critical: the `target_layer_ids` capture path
/// (`single_token_with_multi_hidden_no_tail`) must produce
/// bit-identical hidden_dst on every prefill step. We accumulate
/// the per-token captures across the prompt and compare.
#[test]
fn no_tail_prefill_matches_full_tail() {
    let override_path = std::env::var("QWEN_NO_TAIL_TEST_MODEL").ok();
    let model_path = override_path
        .as_deref()
        .unwrap_or(crate::test_fixtures::QWEN35_0_8B_F32.path());
    if !std::path::Path::new(model_path).exists() {
        assert!(
            override_path.is_none(),
            "explicit no-tail fixture is missing"
        );
        eprintln!("[no-tail-prefill] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };

    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let ids = tok
        .encode("The quick brown fox jumps over the lazy dog", false)
        .expect("tokenize");
    let n = ids.len();
    assert!(n >= 2, "need ≥2 tokens to test the skip-tail path");

    // Path A: full-tail every token.
    let mut s_a = MetalSession::fresh(&ctx, &mm, n + 4).expect("session A");
    let mut last_a = vec![];
    for (i, &tid) in ids.iter().enumerate() {
        last_a = mf.single_token(tid, i as u32, &mut s_a).expect("forward A");
    }

    // Path B: no_tail for [0..n-1), full tail for the last token.
    let mut s_b = MetalSession::fresh(&ctx, &mm, n + 4).expect("session B");
    let mut last_b = vec![];
    for (i, &tid) in ids.iter().enumerate() {
        if i + 1 < n {
            mf.single_token_no_tail(tid, i as u32, &mut s_b)
                .expect("forward B no_tail");
        } else {
            last_b = mf
                .single_token(tid, i as u32, &mut s_b)
                .expect("forward B tail");
        }
    }

    // Bit-exact match required: same kernels in the same order with
    // same inputs (no mat-mat half-staging on the no_tail path).
    assert_eq!(
        last_a.len(),
        last_b.len(),
        "logits length mismatch ({} vs {})",
        last_a.len(),
        last_b.len()
    );
    let mut max_abs = 0.0f32;
    for i in 0..last_a.len() {
        max_abs = max_abs.max((last_a[i] - last_b[i]).abs());
    }
    eprintln!("[no-tail-prefill] full-tail vs no-tail final logits max|Δ|={max_abs:.6e}");
    assert_eq!(
        max_abs, 0.0,
        "logits must match BIT-EXACTLY (max|Δ|={max_abs:e})"
    );
    assert!(
        last_a
            .iter()
            .zip(&last_b)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );

    // Multi-hidden capture must also be bit-exact across all prefill
    // positions. We accumulate the per-position hidden captures into
    // a host-side buffer (mirroring the bench's
    // `append_target_ctx_column_now` pattern) and compare path A vs
    // path B's accumulations.
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    // Pick a few capture layers spanning the network (the H5 drafter
    // captures K=5; the 0.8B-F32 oracle has 36 blocks so 5 evenly
    // spaced layers exercises a realistic K).
    let capture_layers: Vec<u32> = vec![
        0,
        (mm.blocks.len() / 4) as u32,
        (mm.blocks.len() / 2) as u32,
        (3 * mm.blocks.len() / 4) as u32,
        (mm.blocks.len() - 1) as u32,
    ];
    let k = capture_layers.len();

    let h_dst_a = MetalTensor::zeros_f32(&ctx, vec![(k * h) as u64]).expect("h_dst_a");
    let h_dst_b = MetalTensor::zeros_f32(&ctx, vec![(k * h) as u64]).expect("h_dst_b");

    // Per-position accumulator: [n, k * h]
    let mut accum_a = vec![0.0f32; n * k * h];
    let mut accum_b = vec![0.0f32; n * k * h];

    let mut s2_a = MetalSession::fresh(&ctx, &mm, n + 4).expect("session2 A");
    for (i, &tid) in ids.iter().enumerate() {
        mf.single_token_with_multi_hidden(tid, i as u32, &mut s2_a, &capture_layers, &h_dst_a)
            .expect("multi_hidden A");
        unsafe {
            let src = h_dst_a.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(
                src,
                accum_a[i * k * h..(i + 1) * k * h].as_mut_ptr(),
                k * h,
            );
        }
    }

    let mut s2_b = MetalSession::fresh(&ctx, &mm, n + 4).expect("session2 B");
    for (i, &tid) in ids.iter().enumerate() {
        if i + 1 < n {
            mf.single_token_with_multi_hidden_no_tail(
                tid,
                i as u32,
                &mut s2_b,
                &capture_layers,
                &h_dst_b,
            )
            .expect("multi_hidden B no_tail");
        } else {
            mf.single_token_with_multi_hidden(tid, i as u32, &mut s2_b, &capture_layers, &h_dst_b)
                .expect("multi_hidden B tail");
        }
        unsafe {
            let src = h_dst_b.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(
                src,
                accum_b[i * k * h..(i + 1) * k * h].as_mut_ptr(),
                k * h,
            );
        }
    }

    let mut max_abs_h = 0.0f32;
    let mut first_pos = usize::MAX;
    for i in 0..(n * k * h) {
        let d = (accum_a[i] - accum_b[i]).abs();
        if d > max_abs_h {
            max_abs_h = d;
            first_pos = i;
        }
    }
    eprintln!(
        "[no-tail-prefill] accumulated multi-hidden max|Δ|={max_abs_h:.6e} (first nonzero idx={first_pos})"
    );
    assert_eq!(
        max_abs_h, 0.0,
        "accumulated multi-hidden must match BIT-EXACTLY (max|Δ|={max_abs_h:e})"
    );
    assert!(
        accum_a
            .iter()
            .zip(&accum_b)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    for (a, b) in [(&s_a, &s_b), (&s2_a, &s2_b)] {
        let identity = a.snapshot_identity(1, 2);
        let a = a
            .snapshot(identity.clone(), ids.clone(), None)
            .expect("snapshot A");
        let b = b.snapshot(identity, ids.clone(), None).expect("snapshot B");
        assert_eq!(a.kv_n_pos, b.kv_n_pos);
        assert!(a.kv_k_arena == b.kv_k_arena, "K state differs");
        assert!(a.kv_v_arena == b.kv_v_arena, "V state differs");
        assert!(
            a.gdn_conv_arena == b.gdn_conv_arena,
            "convolution state differs"
        );
        assert!(
            a.gdn_state_arena == b.gdn_state_arena,
            "recurrent state differs"
        );
    }
    for (a, b) in [(&mut s_a, &mut s_b), (&mut s2_a, &mut s2_b)] {
        let a = mf
            .single_token(ids[0], n as u32, a)
            .expect("continuation A");
        let b = mf
            .single_token(ids[0], n as u32, b)
            .expect("continuation B");
        assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

/// Validate a single full-attention block end-to-end on Metal vs the
/// CPU oracle. Uses block 3 of Qwen3.5-0.8B-F32 (the first attn block,
/// n_q=8, n_kv=2, head_dim=256, 4:1 GQA).
#[test]
fn metal_attn_block_matches_cpu() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        eprintln!("[metal-attn] skipped — model missing");
        return;
    }
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    // Inputs: a synthetic but fixed residual stream (no need for a
    // real model state for block-level validation; we just need
    // identical inputs to CPU and GPU paths).
    let h = m.arch.hidden_size as usize;
    let initial_x: Vec<f32> = (0..h).map(|i| ((i % 31) as f32 - 15.0) * 0.02).collect();
    let position: u32 = 0;
    let attn_block_idx = 3usize; // first attn block in 0.8B
    // attn_idx_in_session is the 0-indexed count among ATTN blocks
    // before this one. block 3 is the first attn block, so 0.
    let attn_idx_in_session = 0usize;

    // CPU reference: replicate exactly what forward.rs:attn_step does.
    let cpu_x = run_cpu_attn_block_for_test(&g, &m, &initial_x, attn_block_idx, position);

    // Metal.
    let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
    let mf = MetalForward::new(&ctx, &mm);
    mf.set_residual_for_test(&mut s, &initial_x);
    let metal_x = mf
        .run_one_attn_block_for_test(attn_block_idx, attn_idx_in_session, position, &mut s)
        .expect("metal attn block");

    let max_abs = metal_x
        .iter()
        .zip(cpu_x.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = metal_x
        .iter()
        .zip(cpu_x.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let na: f64 = metal_x.iter().map(|v| (*v as f64).powi(2)).sum();
    let nb: f64 = cpu_x.iter().map(|v| (*v as f64).powi(2)).sum();
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    eprintln!("[metal-attn-block3] hidden={h} max|Δ|={max_abs:.2e} cos={cos:.6}");
    assert!(max_abs < 1e-3, "attn block-3 drift {max_abs}");
    assert!(cos > 0.9999, "attn block-3 cos {cos}");
}

/// CPU reference: full-attn block (norm → attn → residual → post_norm
/// → FFN → residual) replicated inline. Mirrors forward.rs's
/// single_token block flow for an attn block.
fn run_cpu_attn_block_for_test(
    gguf: &GgufFile,
    model: &Model<'_>,
    initial_x: &[f32],
    block_idx: usize,
    position: u32,
) -> Vec<f32> {
    let block = &model.blocks[block_idx];
    let ab = match block {
        crate::loader::Block::Attn(a) => a,
        _ => panic!("not an attn block"),
    };
    let arch = &model.arch;
    let h = arch.hidden_size as usize;
    let head_dim = arch.attn_head_dim as usize;
    let n_q = arch.n_q_heads as usize;
    let n_kv = arch.n_kv_heads as usize;
    let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
    let q_dim = n_q * head_dim;
    let kv_dim = n_kv * head_dim;
    let group = n_q / n_kv;
    let theta = arch.rope_theta;

    let mut x = initial_x.to_vec();

    // Pre-mixer norm.
    let attn_norm_w = crate::codec::dequant_to_f32(ab.attn_norm, gguf.slice(ab.attn_norm)).unwrap();
    let cur = crate::forward::rms_norm_pub(&x, &attn_norm_w, super::RMS_EPS);

    // Q projection (2 * q_dim) → split.
    let q_w = crate::codec::dequant_to_f32(ab.q, gguf.slice(ab.q)).unwrap();
    let q_full = crate::forward::mat_vec_pub(&q_w, h, 2 * q_dim, &cur);
    let mut qcur = vec![0.0f32; q_dim];
    let mut gate = vec![0.0f32; q_dim];
    for hi in 0..n_q {
        let src = &q_full[hi * 2 * head_dim..(hi + 1) * 2 * head_dim];
        qcur[hi * head_dim..(hi + 1) * head_dim].copy_from_slice(&src[..head_dim]);
        gate[hi * head_dim..(hi + 1) * head_dim].copy_from_slice(&src[head_dim..]);
    }

    // Q-norm.
    let qnorm_w = crate::codec::dequant_to_f32(ab.q_norm, gguf.slice(ab.q_norm)).unwrap();
    for hi in 0..n_q {
        let s = hi * head_dim;
        let n = crate::forward::rms_norm_pub(&qcur[s..s + head_dim], &qnorm_w, super::RMS_EPS);
        qcur[s..s + head_dim].copy_from_slice(&n);
    }

    // K, V.
    let k_w = crate::codec::dequant_to_f32(ab.k, gguf.slice(ab.k)).unwrap();
    let v_w = crate::codec::dequant_to_f32(ab.v, gguf.slice(ab.v)).unwrap();
    let mut kcur = crate::forward::mat_vec_pub(&k_w, h, kv_dim, &cur);
    let vcur = crate::forward::mat_vec_pub(&v_w, h, kv_dim, &cur);

    // K-norm.
    let knorm_w = crate::codec::dequant_to_f32(ab.k_norm, gguf.slice(ab.k_norm)).unwrap();
    for hi in 0..n_kv {
        let s = hi * head_dim;
        let n = crate::forward::rms_norm_pub(&kcur[s..s + head_dim], &knorm_w, super::RMS_EPS);
        kcur[s..s + head_dim].copy_from_slice(&n);
    }

    // RoPE on Q and K (NEOX/IMROPE pairing for text positions).
    rope_in_place_local(&mut qcur, n_q, head_dim, n_rot, position, theta);
    rope_in_place_local(&mut kcur, n_kv, head_dim, n_rot, position, theta);

    // Single-token cache: KV is just the current step.
    // Attention with one position (position itself).
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; q_dim];
    for qh in 0..n_q {
        let kvh = qh / group;
        let q_slice = &qcur[qh * head_dim..(qh + 1) * head_dim];
        let k_slice = &kcur[kvh * head_dim..(kvh + 1) * head_dim];
        let v_slice = &vcur[kvh * head_dim..(kvh + 1) * head_dim];

        let mut s = 0.0f32;
        for i in 0..head_dim {
            s += q_slice[i] * k_slice[i];
        }
        let _score = s * scale;
        // softmax over a single value = 1.0 → out = v
        for i in 0..head_dim {
            attn_out[qh * head_dim + i] = v_slice[i];
        }
    }

    // Apply gated-attention sigmoid gate.
    for i in 0..attn_out.len() {
        let sg = 1.0 / (1.0 + (-gate[i]).exp());
        attn_out[i] *= sg;
    }

    // Output projection.
    let o_w = crate::codec::dequant_to_f32(ab.o, gguf.slice(ab.o)).unwrap();
    let mixer_out = crate::forward::mat_vec_pub(&o_w, q_dim, h, &attn_out);

    // Residual #1.
    for (xi, mo) in x.iter_mut().zip(mixer_out.iter()) {
        *xi += *mo;
    }

    // Pre-FFN norm.
    let post_norm_w =
        crate::codec::dequant_to_f32(ab.post_attention_norm, gguf.slice(ab.post_attention_norm))
            .unwrap();
    let cur = crate::forward::rms_norm_pub(&x, &post_norm_w, super::RMS_EPS);

    // FFN.
    let f = arch.intermediate_size as usize;
    let g_w = crate::codec::dequant_to_f32(ab.ffn_gate, gguf.slice(ab.ffn_gate)).unwrap();
    let u_w = crate::codec::dequant_to_f32(ab.ffn_up, gguf.slice(ab.ffn_up)).unwrap();
    let d_w = crate::codec::dequant_to_f32(ab.ffn_down, gguf.slice(ab.ffn_down)).unwrap();
    let gated = crate::forward::mat_vec_pub(&g_w, h, f, &cur);
    let upped = crate::forward::mat_vec_pub(&u_w, h, f, &cur);
    let mut hidden = vec![0.0f32; f];
    for i in 0..f {
        let g = gated[i];
        let silu_g = g / (1.0 + (-g).exp());
        hidden[i] = silu_g * upped[i];
    }
    let ffn_out = crate::forward::mat_vec_pub(&d_w, f, h, &hidden);

    // Residual #2.
    for (xi, fo) in x.iter_mut().zip(ffn_out.iter()) {
        *xi += *fo;
    }
    x
}

fn rope_in_place_local(
    buf: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    position: u32,
    theta_base: f32,
) {
    let pos = position as f32;
    let half = n_rot / 2;
    for hi in 0..n_heads {
        let h_off = hi * head_dim;
        for i in 0..half {
            let exponent = (2 * i) as f32 / n_rot as f32;
            let freq = pos / theta_base.powf(exponent);
            let (s, c) = freq.sin_cos();
            let a = buf[h_off + i];
            let b = buf[h_off + i + half];
            buf[h_off + i] = a * c - b * s;
            buf[h_off + i + half] = a * s + b * c;
        }
    }
}

/// CPU reference: replicate exactly what forward.rs:single_token does
/// for block 0 of Qwen3.5-0.8B (a GDN block), starting from
/// `initial_x` (the token embedding).
fn run_cpu_block0_for_test(gguf: &GgufFile, model: &Model<'_>, initial_x: &[f32]) -> Vec<f32> {
    let f = Forward::new(gguf, model);
    // Use Forward's public single_token but at position 0 with token
    // id derived from initial_x: too indirect. Easier path: take the
    // raw GDN-block computation and replicate it inline.
    //
    // forward.rs's `single_token` already does exactly this. The
    // simplest validation is to run it in full and compare logits —
    // but that requires the full attn block, which Metal doesn't
    // have yet. So instead we replicate just block 0 here.
    //
    // Block 0 in 0.8B is a GDN block. The CPU computation is in
    // forward.rs lines 290-510-ish. We recreate the same flow with
    // the public helpers in `forward`.
    let mut x = initial_x.to_vec();
    let block = &model.blocks[0];
    let gb = match block {
        crate::loader::Block::Gdn(g) => g,
        _ => panic!("block 0 is not GDN"),
    };

    // Pre-mixer norm.
    let attn_norm_w = crate::codec::dequant_to_f32(gb.attn_norm, gguf.slice(gb.attn_norm)).unwrap();
    let cur = crate::forward::rms_norm_pub(&x, &attn_norm_w, super::RMS_EPS);

    // GDN inner. The cleanest way to replicate is to hand-call
    // Forward's gdn_step. It takes a GdnState; we'll build a fresh
    // one and ignore the state output.
    let mut state = crate::forward::GdnState::fresh(model);
    let mixer_out = call_gdn_step_directly(&f, gb, &cur, &mut state);

    // Residual #1.
    for (xi, mi) in x.iter_mut().zip(mixer_out.iter()) {
        *xi += *mi;
    }

    // Pre-FFN norm.
    let post_norm_w =
        crate::codec::dequant_to_f32(gb.post_attention_norm, gguf.slice(gb.post_attention_norm))
            .unwrap();
    let cur = crate::forward::rms_norm_pub(&x, &post_norm_w, super::RMS_EPS);

    // FFN.
    let arch = &model.arch;
    let h = arch.hidden_size as usize;
    let fdim = arch.intermediate_size as usize;
    let g_w = crate::codec::dequant_to_f32(gb.ffn_gate, gguf.slice(gb.ffn_gate)).unwrap();
    let u_w = crate::codec::dequant_to_f32(gb.ffn_up, gguf.slice(gb.ffn_up)).unwrap();
    let d_w = crate::codec::dequant_to_f32(gb.ffn_down, gguf.slice(gb.ffn_down)).unwrap();

    let gated = crate::forward::mat_vec_pub(&g_w, h, fdim, &cur);
    let upped = crate::forward::mat_vec_pub(&u_w, h, fdim, &cur);
    let mut hidden = vec![0.0f32; fdim];
    for i in 0..fdim {
        let g = gated[i];
        let silu_g = g / (1.0 + (-g).exp());
        hidden[i] = silu_g * upped[i];
    }
    let ffn_out = crate::forward::mat_vec_pub(&d_w, fdim, h, &hidden);

    // Residual #2.
    for (xi, fo) in x.iter_mut().zip(ffn_out.iter()) {
        *xi += *fo;
    }
    x
}

/// Use a private CPU-only path to run forward::Forward's gdn_step on
/// block 0. We can't call it directly because it's private to
/// Forward; the test re-implements the same math inline.
// CPU GDN inline re-impl: strided 2D state access
// `state.ssm[0][s_off + dv * head_dim + dk]`. Iterator rewrite
// hides the stride math (the whole point of this reference).
#[allow(clippy::needless_range_loop)]
fn call_gdn_step_directly(
    f: &Forward,
    gb: &crate::loader::GdnBlock,
    x: &[f32],
    state: &mut crate::forward::GdnState,
) -> Vec<f32> {
    // Public re-exports added below in forward.rs would let us avoid
    // this. For now, since Forward's gdn_step is private, we run the
    // *full* Forward::single_token and extract the post-block-0
    // residual. That requires a hook in Forward we don't have.
    //
    // Pragmatic shortcut: re-implement gdn_step here using the public
    // mat_vec_pub and matching the exact CPU forward path. This is
    // ~50 lines of duplication but isolates the test from Forward's
    // internals.
    let arch = &f.model.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = 2 * n_k * head_dim + n_v * head_dim;
    let conv_kernel = arch.gdn_conv_kernel as usize;
    let v_dim = n_v * head_dim;

    let qkv_w = crate::codec::dequant_to_f32(gb.in_proj_qkv, f.gguf.slice(gb.in_proj_qkv)).unwrap();
    let qkv = crate::forward::mat_vec_pub(&qkv_w, h, conv_dim, x);
    let z_w = crate::codec::dequant_to_f32(gb.in_proj_z, f.gguf.slice(gb.in_proj_z)).unwrap();
    let z = crate::forward::mat_vec_pub(&z_w, h, v_dim, x);

    let beta_w = crate::codec::dequant_to_f32(gb.beta_proj, f.gguf.slice(gb.beta_proj)).unwrap();
    let mut beta = crate::forward::mat_vec_pub(&beta_w, h, n_v, x);
    for v in beta.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
    let alpha_w = crate::codec::dequant_to_f32(gb.alpha_proj, f.gguf.slice(gb.alpha_proj)).unwrap();
    let mut alpha = crate::forward::mat_vec_pub(&alpha_w, h, n_v, x);
    let dt = crate::codec::dequant_to_f32(gb.dt_bias, f.gguf.slice(gb.dt_bias)).unwrap();
    for (a, &dti) in alpha.iter_mut().zip(dt.iter()) {
        *a += dti;
    }
    let a_log = crate::codec::dequant_to_f32(gb.a_log, f.gguf.slice(gb.a_log)).unwrap();
    let mut g = vec![0.0f32; n_v];
    for i in 0..n_v {
        let sp = if alpha[i] > 20.0 {
            alpha[i]
        } else if alpha[i] < -20.0 {
            alpha[i].exp()
        } else {
            (1.0 + alpha[i].exp()).ln()
        };
        g[i] = sp * a_log[i];
    }

    let conv_w = crate::codec::dequant_to_f32(gb.conv1d, f.gguf.slice(gb.conv1d)).unwrap();
    let kmin1 = conv_kernel - 1;
    let mut conv_input = vec![0.0f32; conv_kernel * conv_dim];
    for t in 0..kmin1 {
        conv_input[t * conv_dim..(t + 1) * conv_dim]
            .copy_from_slice(&state.conv[0][t * conv_dim..(t + 1) * conv_dim]);
    }
    conv_input[kmin1 * conv_dim..].copy_from_slice(&qkv);

    let mut conv_out = vec![0.0f32; conv_dim];
    for c in 0..conv_dim {
        let mut sm = 0.0f32;
        for k in 0..conv_kernel {
            sm += conv_w[c * conv_kernel + k] * conv_input[k * conv_dim + c];
        }
        conv_out[c] = sm / (1.0 + (-sm).exp());
    }
    // (slide buffer; not asked for in test, just for completeness it
    // would happen here, but state.conv is local to this fn here)
    for t in 0..kmin1 - 1 {
        for i in 0..conv_dim {
            state.conv[0][t * conv_dim + i] = state.conv[0][(t + 1) * conv_dim + i];
        }
    }
    let last = (kmin1 - 1) * conv_dim;
    state.conv[0][last..last + conv_dim].copy_from_slice(&qkv);

    // Split conv_out → q,k,v.
    let q_full = conv_out[0..n_k * head_dim].to_vec();
    let k_full = conv_out[n_k * head_dim..2 * n_k * head_dim].to_vec();
    let v_full = conv_out[2 * n_k * head_dim..].to_vec();

    // Per-head L2 norm of Q and K.
    let mut q_full = q_full.clone();
    let mut k_full = k_full.clone();
    for hi in 0..n_k {
        let off = hi * head_dim;
        let sq: f32 = q_full[off..off + head_dim].iter().map(|v| v * v).sum();
        let scale = 1.0 / sq.sqrt().max(super::RMS_EPS);
        for i in 0..head_dim {
            q_full[off + i] *= scale;
        }
        let sq: f32 = k_full[off..off + head_dim].iter().map(|v| v * v).sum();
        let scale = 1.0 / sq.sqrt().max(super::RMS_EPS);
        for i in 0..head_dim {
            k_full[off + i] *= scale;
        }
    }

    // Recurrence.
    let mut o = vec![0.0f32; v_dim];
    for hi in 0..n_v {
        let hk = hi % n_k;
        let s_off = hi * head_dim * head_dim;
        let q_h = &q_full[hk * head_dim..(hk + 1) * head_dim];
        let k_h = &k_full[hk * head_dim..(hk + 1) * head_dim];
        let v_h = &v_full[hi * head_dim..(hi + 1) * head_dim];
        let g_h = g[hi].exp();
        let b_h = beta[hi];
        for j in 0..head_dim * head_dim {
            state.ssm[0][s_off + j] *= g_h;
        }
        let mut sk = vec![0.0f32; head_dim];
        for dv in 0..head_dim {
            let mut sm = 0.0f32;
            for dk in 0..head_dim {
                sm += state.ssm[0][s_off + dv * head_dim + dk] * k_h[dk];
            }
            sk[dv] = sm;
        }
        for dv in 0..head_dim {
            let coeff = b_h * (v_h[dv] - sk[dv]);
            for dk in 0..head_dim {
                state.ssm[0][s_off + dv * head_dim + dk] += coeff * k_h[dk];
            }
        }
        for dv in 0..head_dim {
            let mut sm = 0.0f32;
            for dk in 0..head_dim {
                sm += state.ssm[0][s_off + dv * head_dim + dk] * q_h[dk];
            }
            o[hi * head_dim + dv] = sm;
        }
    }

    // RMSNormGated: norm(o) * silu(z), per-head.
    let norm_w = crate::codec::dequant_to_f32(gb.norm, f.gguf.slice(gb.norm)).unwrap();
    let mut gated = vec![0.0f32; v_dim];
    for hi in 0..n_v {
        let off = hi * head_dim;
        let normed = crate::forward::rms_norm_pub(
            &o[off..off + head_dim],
            &norm_w,
            super::RMS_EPS * head_dim as f32,
        );
        for i in 0..head_dim {
            let zi = z[off + i];
            let silu_z = zi / (1.0 + (-zi).exp());
            gated[off + i] = normed[i] * silu_z;
        }
    }

    // Output projection.
    let out_w = crate::codec::dequant_to_f32(gb.out_proj, f.gguf.slice(gb.out_proj)).unwrap();
    crate::forward::mat_vec_pub(&out_w, v_dim, h, &gated)
}

/// **H2.0 — prefix-cache correctness spike.** Validates that
/// snapshot+restore at a prefix boundary yields the same final
/// logits as cold prefill of the full sequence.
///
/// Mechanism (deliberately minimal — no public API yet, just to
/// prove the principle):
///   1. Prefill prompt into session A (cold path).
///   2. Capture last-position logits from A.
///   3. Make a fresh session B, prefill ONLY the prefix into B.
///   4. Snapshot B's state by raw-byte-cloning all six per-layer
///      MTLBuffers via `MTLBuffer.contents()` → `Vec<u8>`.
///   5. Make a fresh session C, restore the snapshot bytes into C's
///      buffers (also via raw `contents()` write).
///   6. Run the suffix tokens through C, capture last-position logits.
///   7. Assert cos(A_logits, C_logits) ≥ 0.99999 and same argmax.
///
/// If this passes, snapshot mechanism is sound and we can build the
/// real packed-arena LRU on top. If it fails, H2 is dead and we
/// pivot to backend sampler / attention-surround fusions.
///
/// Falsifiable kill criterion (per codex's H2 review): cos < 0.99999
/// or argmax mismatch at any tested prefix length.
#[test]
#[ignore]
fn h2_prefix_cache_correctness_spike() {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[h2-spike] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };

    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let mf = MetalForward::new(&ctx, &mm);

    // Use a longer prompt so we can split it into prefix+suffix.
    let prompt = "The quick brown fox jumps over the lazy dog and then runs into the deep forest";
    let ids = tok.encode(prompt, false).expect("tokenize");
    eprintln!("[h2-spike] {} prompt tokens: {ids:?}", ids.len());
    // Test multiple prefix split points (per codex's kill criteria).
    let prefix_lens = [3usize, 5, 8, ids.len() - 1];

    // Read all bytes from a MetalTensor's MTLBuffer (shared storage).
    let read_bytes = |t: &MetalTensor| -> Vec<u8> {
        let n = t.n_bytes() as usize;
        let mut out = vec![0u8; n];
        unsafe {
            let src = (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    };
    let write_bytes = |t: &MetalTensor, bytes: &[u8]| {
        assert_eq!(
            bytes.len() as u64,
            t.n_bytes(),
            "snapshot byte size mismatch"
        );
        unsafe {
            let dst = (t.buffer.contents().as_ptr() as *mut u8).add(t.offset as usize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
    };

    for &prefix_len in &prefix_lens {
        assert!(prefix_len < ids.len() && prefix_len > 0);
        let suffix_len = ids.len() - prefix_len;
        eprintln!("[h2-spike] === prefix_len={prefix_len} suffix_len={suffix_len} ===");

        // ---- 1+2: Cold prefill of full prompt. Capture last logits. ----
        let mut sess_cold = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session A");
        let mut logits_cold = vec![];
        for (i, &tid) in ids.iter().enumerate() {
            logits_cold = mf
                .single_token(tid, i as u32, &mut sess_cold)
                .expect("cold");
        }
        let n = logits_cold.len();

        // ---- 3: Fresh session, prefill only the prefix. ----
        let mut sess_pre = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session B");
        for (i, &tid) in ids.iter().take(prefix_len).enumerate() {
            let _ = mf.single_token(tid, i as u32, &mut sess_pre).expect("pre");
        }

        // ---- 4: Snapshot all six per-layer state bytes. ----
        let snap_kv_k: Vec<Vec<u8>> = sess_pre.kv_k.iter().map(read_bytes).collect();
        let snap_kv_v: Vec<Vec<u8>> = sess_pre.kv_v.iter().map(read_bytes).collect();
        let snap_kv_n_pos = sess_pre.kv_n_pos.clone();
        let snap_gdn_conv: Vec<Vec<u8>> = sess_pre.gdn_conv.iter().map(read_bytes).collect();
        let snap_gdn_state: Vec<Vec<u8>> = sess_pre.gdn_state.iter().map(read_bytes).collect();

        let total_snap_bytes: usize = snap_kv_k.iter().map(|v| v.len()).sum::<usize>()
            + snap_kv_v.iter().map(|v| v.len()).sum::<usize>()
            + snap_gdn_conv.iter().map(|v| v.len()).sum::<usize>()
            + snap_gdn_state.iter().map(|v| v.len()).sum::<usize>();
        eprintln!(
            "[h2-spike]   snapshot size: {:.1} MB ({} attn KV + {} GDN conv + {} GDN state buffers)",
            total_snap_bytes as f64 / 1e6,
            snap_kv_k.len() * 2,
            snap_gdn_conv.len(),
            snap_gdn_state.len()
        );

        // ---- 5: Fresh session, restore the snapshot bytes. ----
        let mut sess_restored = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session C");
        for (i, src) in snap_kv_k.iter().enumerate() {
            write_bytes(&sess_restored.kv_k[i], src);
        }
        for (i, src) in snap_kv_v.iter().enumerate() {
            write_bytes(&sess_restored.kv_v[i], src);
        }
        sess_restored.kv_n_pos.copy_from_slice(&snap_kv_n_pos);
        for (i, src) in snap_gdn_conv.iter().enumerate() {
            write_bytes(&sess_restored.gdn_conv[i], src);
        }
        for (i, src) in snap_gdn_state.iter().enumerate() {
            write_bytes(&sess_restored.gdn_state[i], src);
        }

        // ---- 6: Run suffix tokens through restored session. ----
        let mut logits_restored = vec![];
        for k in 0..suffix_len {
            let pos = (prefix_len + k) as u32;
            let tid = ids[prefix_len + k];
            logits_restored = mf
                .single_token(tid, pos, &mut sess_restored)
                .expect("restored forward");
        }

        // ---- 7: Compare last-position logits. ----
        assert_eq!(
            logits_cold.len(),
            logits_restored.len(),
            "logits len mismatch"
        );
        let mut max_abs = 0.0f32;
        let mut argmax_cold = 0usize;
        let mut argmax_restored = 0usize;
        let mut max_cold = f32::NEG_INFINITY;
        let mut max_restored = f32::NEG_INFINITY;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..n {
            let d = (logits_cold[i] - logits_restored[i]).abs();
            max_abs = max_abs.max(d);
            if logits_cold[i] > max_cold {
                max_cold = logits_cold[i];
                argmax_cold = i;
            }
            if logits_restored[i] > max_restored {
                max_restored = logits_restored[i];
                argmax_restored = i;
            }
            dot += logits_cold[i] as f64 * logits_restored[i] as f64;
            na += (logits_cold[i] as f64).powi(2);
            nb += (logits_restored[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[h2-spike]   cos={cos:.7} max|Δ|={max_abs:.4} argmax: cold={argmax_cold} restored={argmax_restored} {}",
            if argmax_cold == argmax_restored {
                "✓"
            } else {
                "✗ MISMATCH"
            }
        );
        assert_eq!(
            argmax_cold, argmax_restored,
            "argmax mismatch at prefix_len={prefix_len}: cold={argmax_cold} restored={argmax_restored}"
        );
        assert!(
            cos > 0.99999,
            "cos={cos} below 0.99999 at prefix_len={prefix_len}"
        );
    }
    eprintln!("[h2-spike] ALL prefix splits passed — H2 mechanism validated.");
}

/// **H2.1 — packed-arena snapshot via the production API.**
/// Same correctness guarantee as the H2.0 spike, but uses the new
/// `MetalSession::snapshot` / `restore_from` methods on top of
/// `SessionSnapshot` arenas. Validates the production API matches
/// the bit-exact spike.
///
/// Also reports snapshot/restore wall-clock so we can compare to
/// the bandwidth model (codex predicted ~0.9-2 ms; let's see).
#[test]
#[ignore]
fn h2_packed_arena_snapshot_matches_cold() {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[h2-arena] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => panic!("init failed: {e}"),
    };

    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let mf = MetalForward::new(&ctx, &mm);

    // Larger prompt for more meaningful prefix lengths.
    let prompt = "The quick brown fox jumps over the lazy dog and then runs into the deep dark forest where it meets a wise old owl who teaches it the meaning of life";
    let ids = tok.encode(prompt, false).expect("tokenize");
    eprintln!("[h2-arena] {} prompt tokens", ids.len());

    for &prefix_len in &[1usize, 5, 16, ids.len() - 1] {
        assert!(prefix_len < ids.len() && prefix_len > 0);
        let suffix_len = ids.len() - prefix_len;
        eprintln!("[h2-arena] === prefix_len={prefix_len} suffix_len={suffix_len} ===");

        // Cold reference.
        let mut sess_cold = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("A");
        let mut logits_cold = vec![];
        for (i, &tid) in ids.iter().enumerate() {
            logits_cold = mf
                .single_token(tid, i as u32, &mut sess_cold)
                .expect("cold");
        }

        // Build a snapshot via the production API.
        let mut sess_pre = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("B");
        let mut last_pre_logits = vec![];
        for (i, &tid) in ids.iter().take(prefix_len).enumerate() {
            last_pre_logits = mf.single_token(tid, i as u32, &mut sess_pre).expect("pre");
        }
        let identity = sess_pre.snapshot_identity(0xDEADBEEF, 0xCAFE);
        let prefix_tokens: Vec<i32> = ids[..prefix_len].to_vec();

        let t = std::time::Instant::now();
        let snap = sess_pre
            .snapshot(identity.clone(), prefix_tokens, Some(last_pre_logits))
            .expect("snapshot");
        let snap_ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "[h2-arena]   snapshot: {:.2} MB in {:.2} ms = {:.1} GB/s",
            snap.n_bytes() as f64 / 1e6,
            snap_ms,
            (snap.n_bytes() as f64 / 1e9) / (snap_ms / 1e3)
        );

        // Restore into a fresh session via the production API.
        let mut sess_restored = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("C");
        let t = std::time::Instant::now();
        sess_restored
            .restore_from(&snap, &identity)
            .expect("restore");
        let restore_ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "[h2-arena]   restore:  {:.2} ms = {:.1} GB/s",
            restore_ms,
            (snap.n_bytes() as f64 / 1e9) / (restore_ms / 1e3)
        );

        // Run suffix tokens through restored session.
        let mut logits_restored = vec![];
        for k in 0..suffix_len {
            let pos = (prefix_len + k) as u32;
            let tid = ids[prefix_len + k];
            logits_restored = mf
                .single_token(tid, pos, &mut sess_restored)
                .expect("restored forward");
        }

        // Compare last-position logits.
        let n = logits_cold.len();
        assert_eq!(n, logits_restored.len());
        let mut max_abs = 0.0f32;
        let mut argmax_cold = 0usize;
        let mut argmax_restored = 0usize;
        let mut max_cold = f32::NEG_INFINITY;
        let mut max_restored = f32::NEG_INFINITY;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..n {
            let d = (logits_cold[i] - logits_restored[i]).abs();
            max_abs = max_abs.max(d);
            if logits_cold[i] > max_cold {
                max_cold = logits_cold[i];
                argmax_cold = i;
            }
            if logits_restored[i] > max_restored {
                max_restored = logits_restored[i];
                argmax_restored = i;
            }
            dot += logits_cold[i] as f64 * logits_restored[i] as f64;
            na += (logits_cold[i] as f64).powi(2);
            nb += (logits_restored[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[h2-arena]   cos={cos:.7} max|Δ|={max_abs:.4} argmax: cold={argmax_cold} restored={argmax_restored} {}",
            if argmax_cold == argmax_restored {
                "✓"
            } else {
                "✗ MISMATCH"
            }
        );
        assert_eq!(argmax_cold, argmax_restored);
        assert!(cos > 0.99999, "cos={cos} below threshold");
    }

    // Identity-mismatch refusal.
    {
        let sess = MetalSession::fresh(&ctx, &mm, 64).expect("ident");
        let bogus_identity = SnapshotIdentity {
            model_id: 0,
            tokenizer_id: 0,
            layout_version: 999,
            n_attn_layers: 0,
            n_gdn_layers: 0,
            kv_dim_elements: 0,
            kv_bytes_per_token: 0,
            kv_storage_kind: SnapshotKvStorageKind::None,
            gdn_state_elements_per_layer: 0,
            gdn_conv_elements_per_layer: 0,
        };
        let bad_snap = SessionSnapshot {
            identity: bogus_identity,
            prefix_tokens: vec![],
            pending_token: None,
            kv_n_pos: vec![],
            kv_k_arena: vec![],
            kv_v_arena: vec![],
            gdn_conv_arena: vec![],
            gdn_state_arena: vec![],
            final_logits: None,
            capture_tail: None,
        };
        let mut s2 = sess;
        assert!(
            s2.restore_from(&bad_snap, &s2.snapshot_identity(1, 1))
                .is_err(),
            "identity mismatch must error"
        );
        eprintln!("[h2-arena]   identity-mismatch refusal: ✓");
    }

    eprintln!("[h2-arena] all prefix splits passed via packed-arena API.");
}
