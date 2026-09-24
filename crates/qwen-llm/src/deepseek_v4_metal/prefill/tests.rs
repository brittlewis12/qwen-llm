use super::*;

#[cfg(feature = "dsv4-diagnostics")]
use sha2::{Digest, Sha256};

const PACKED_ROUTE_AGGREGATE_WIDTH: usize = 4;
const PACKED_ROUTE_STALE_ROUTE: i32 = -101;
const PACKED_ROUTE_FAILED_ROUTE: i32 = -102;
const PACKED_ROUTE_INVALID_ID: i32 = -103;
const PACKED_ROUTE_DUPLICATE_ID: i32 = -104;
const PACKED_ROUTE_INVALID_WEIGHT: i32 = -105;
const PACKED_ROUTE_STALE_SCHEDULE: i32 = -106;
const PACKED_ROUTE_INVALID_COUNT: i32 = -107;
const PACKED_ROUTE_INVALID_SCHEDULE: i32 = -108;
const PACKED_ROUTE_INVALID_PADDING: i32 = -109;
const PACKED_ROUTE_INVALID_AGGREGATE: i32 = -200;
const PACKED_ROUTE_SLOT_GUARD_BYTES: usize = 64;
const PACKED_ROUTE_SLOT_PREFIX: u8 = 0xa5;
const PACKED_ROUTE_SLOT_SUFFIX: u8 = 0x5a;
#[cfg(feature = "dsv4-diagnostics")]
const PACKED_GROUPED_IQ2_COUNT_CENSUS_FIXTURE: &str =
    include_str!("../../../tests/fixtures/deepseek_v4_packed_grouped_iq2_count_census_v1.json");
#[cfg(feature = "dsv4-diagnostics")]
const PACKED_GROUPED_IQ2_ROUTE_CENSUS_FIXTURE: &str =
    include_str!("../../../tests/fixtures/deepseek_v4_packed_grouped_iq2_route_census_v1.json");
#[cfg(feature = "dsv4-diagnostics")]
const PACKED_GROUPED_IQ2_REPRESENTATIVE_ROUTE_CENSUS_FIXTURE: &str = include_str!(
    "../../../tests/fixtures/deepseek_v4_packed_grouped_iq2_route_census_representative_v1.json"
);
#[cfg(feature = "dsv4-diagnostics")]
const PACKED_ALL_IQ3_ROUTE_CENSUS_FIXTURE: &str =
    include_str!("../../../tests/fixtures/deepseek_v4_packed_all_iq3_route_census_v1.json");

#[cfg(feature = "dsv4-diagnostics")]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PackedGroupedIq2CountCensusFixture {
    schema_version: usize,
    count_payload_domain_hex: String,
    count_payload_sha256: String,
    model_content_id: String,
    prompt_token_ids_sha256: String,
    n_tokens: usize,
    top_k: usize,
    expert_count: usize,
    grouped_layer_ids: Vec<usize>,
    total_active_experts: usize,
    inactive_experts: usize,
    total_t32: usize,
    total_padding: usize,
    active_count_histogram_1_to_128: Vec<usize>,
    tile_histogram_1_to_4: Vec<usize>,
    layers: Vec<PackedGroupedIq2CountLayerFixture>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PackedGroupedIq2CountLayerFixture {
    layer: usize,
    active_experts: usize,
    t32: usize,
    padding: usize,
    counts: Vec<u16>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PackedGroupedIq2RouteCensusFixture {
    schema_version: usize,
    route_payload_domain_hex: String,
    route_payload_sha256: String,
    model_content_id: String,
    prompt_token_ids_sha256: String,
    prompt_token_ids: Option<Vec<i32>>,
    count_payload_sha256: String,
    n_tokens: usize,
    top_k: usize,
    expert_count: usize,
    layer_count: usize,
    route_count: usize,
    grouped_layer_ids: Vec<usize>,
    layers: Vec<PackedGroupedIq2RouteLayerFixture>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PackedGroupedIq2RouteLayerFixture {
    layer: usize,
    route_expert_ids: Vec<u16>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PackedAllIq3RouteCensusFixture {
    schema_version: usize,
    count_payload_domain_hex: String,
    count_payload_sha256: String,
    route_payload_domain_hex: String,
    route_payload_sha256: String,
    model_content_id: String,
    prompt_token_ids_sha256: String,
    n_tokens: usize,
    top_k: usize,
    expert_count: usize,
    layer_count: usize,
    route_count: usize,
    total_active_experts: usize,
    total_t32: usize,
    total_padding: usize,
    all_iq3_layer_ids: Vec<usize>,
    layers: Vec<PackedAllIq3RouteLayerFixture>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PackedAllIq3RouteLayerFixture {
    layer: usize,
    active_experts: usize,
    t32: usize,
    padding: usize,
    expert_counts: Vec<u16>,
    route_expert_ids: Vec<u16>,
}

#[cfg(feature = "dsv4-diagnostics")]
fn packed_grouped_iq2_count_census_fixture() -> PackedGroupedIq2CountCensusFixture {
    serde_json::from_str(PACKED_GROUPED_IQ2_COUNT_CENSUS_FIXTURE)
        .expect("valid grouped-IQ2 count census fixture")
}

#[cfg(feature = "dsv4-diagnostics")]
fn packed_grouped_iq2_route_census_fixture() -> PackedGroupedIq2RouteCensusFixture {
    serde_json::from_str(PACKED_GROUPED_IQ2_ROUTE_CENSUS_FIXTURE)
        .expect("valid grouped-IQ2 route census fixture")
}

#[cfg(feature = "dsv4-diagnostics")]
fn packed_grouped_iq2_representative_route_census_fixture() -> PackedGroupedIq2RouteCensusFixture {
    serde_json::from_str(PACKED_GROUPED_IQ2_REPRESENTATIVE_ROUTE_CENSUS_FIXTURE)
        .expect("valid representative grouped-IQ2 route census fixture")
}

#[cfg(feature = "dsv4-diagnostics")]
fn packed_all_iq3_route_census_fixture() -> PackedAllIq3RouteCensusFixture {
    serde_json::from_str(PACKED_ALL_IQ3_ROUTE_CENSUS_FIXTURE)
        .expect("valid packed all-IQ3 route census fixture")
}

#[cfg(feature = "dsv4-diagnostics")]
fn packed_grouped_schedule_from_route_ids(
    n_tokens: usize,
    route_ids: &[u16],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<ExpertBucket>) {
    assert_eq!(route_ids.len(), n_tokens * MOE_TOP_K);
    let expert_ids = route_ids
        .iter()
        .map(|&expert| {
            assert!(usize::from(expert) < MOE_EXPERT_COUNT);
            i32::from(expert)
        })
        .collect::<Vec<_>>();
    for token_ids in route_ids.chunks_exact(MOE_TOP_K) {
        let mut seen = [false; MOE_EXPERT_COUNT];
        for &expert in token_ids {
            assert!(!std::mem::replace(&mut seen[usize::from(expert)], true));
        }
    }
    let mut rows = Vec::with_capacity(route_ids.len());
    let mut slots = Vec::with_capacity(route_ids.len());
    let mut schedule = Vec::new();
    for expert in 0..MOE_EXPERT_COUNT {
        let start = slots.len();
        for (slot, &routed_expert) in route_ids.iter().enumerate() {
            if usize::from(routed_expert) == expert {
                rows.push((slot / MOE_TOP_K) as i32);
                slots.push(slot as i32);
            }
        }
        if slots.len() > start {
            schedule.push(ExpertBucket {
                expert,
                start,
                len: slots.len() - start,
            });
        }
    }
    validate_packed_expert_schedule(
        n_tokens,
        MOE_EXPERT_COUNT,
        &expert_ids,
        &rows,
        &slots,
        &schedule,
    )
    .unwrap();
    (expert_ids, rows, slots, schedule)
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn mhc_delete_profile_reports_raw_and_union_gpu_time() {
    let profile = DeepSeekV4MhcDeleteProfile {
        execution: DeepSeekV4MhcExecutionKind::Producer,
        queue_identity: 1,
        wall_ms: 9.0,
        sites: Vec::new(),
        command_intervals: vec![
            DeepSeekV4MhcCommandInterval {
                submission_ordinal: 0,
                layer: 0,
                kind: DeepSeekV4MhcCommandKind::PreExpert,
                gpu_start_seconds: 1.000,
                gpu_end_seconds: 1.004,
            },
            DeepSeekV4MhcCommandInterval {
                submission_ordinal: 1,
                layer: 0,
                kind: DeepSeekV4MhcCommandKind::SharedOverlap,
                gpu_start_seconds: 1.003,
                gpu_end_seconds: 1.006,
            },
            DeepSeekV4MhcCommandInterval {
                submission_ordinal: 2,
                layer: 0,
                kind: DeepSeekV4MhcCommandKind::Expert,
                gpu_start_seconds: 1.007,
                gpu_end_seconds: 1.009,
            },
        ],
    };

    assert!((profile.raw_gpu_ms() - 9.0).abs() < 1e-9);
    assert!((profile.union_gpu_ms() - 8.0).abs() < 1e-9);
    assert_eq!(MHC_DELETE_SITE_COUNT, 86);
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn mhc_delete_oracle_plans_all_sites_and_arm_ownership() {
    let ctx = MetalContext::new().expect("create Metal context");
    let buffer = PackedMhcOracleBuffer::new_capture(&ctx, 1).expect("create mHC oracle");
    let identity = DeepSeekV4MhcOracleIdentity {
        model_content_id: [1; 32],
        compatibility_id: [2; 32],
        token_sha256: [3; 32],
        policy_sha256: [4; 32],
        policy_manifest: "test-policy".to_string(),
        start_position: 0,
        n_tokens: 1,
        rms_epsilon_bits: 1.0e-6f32.to_bits(),
        hc_epsilon_bits: 1.0e-6f32.to_bits(),
        device_registry_id: ctx.device.registryID(),
        residency_tensor_count: 1_328,
        residency_source_bytes: 89_920_886_108,
        expert_count: 160,
    };
    let ordinary = MetalTensor::zeros_f32(&ctx, vec![DEEPSEEK_V4_HC_PARAMETER_COUNT as u64, 1])
        .expect("create ordinary mixes");
    let mut capture = PackedMhcExecution::capture(buffer.clone());
    capture
        .bind_identity(identity.clone())
        .expect("bind capture identity");
    let mut offsets = Vec::new();
    for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
        for site in [DeepSeekV4MhcSiteKind::Attention, DeepSeekV4MhcSiteKind::Ffn] {
            let plan = capture
                .site_plan(layer, site, &ordinary)
                .expect("plan capture site");
            assert!(plan.producer_runs);
            let producer = plan.producer_output.as_ref().expect("capture producer");
            let controls = plan.controls_input.as_ref().expect("capture controls");
            assert_eq!(producer.offset, controls.offset);
            offsets.push(producer.offset);
        }
    }
    let (capture_sites, _) = capture.finish().expect("finish capture");
    assert_eq!(capture_sites.len(), 86);
    let site_bytes = (DEEPSEEK_V4_HC_PARAMETER_COUNT * std::mem::size_of::<f32>()) as u64;
    assert_eq!(
        offsets,
        (0..MHC_DELETE_SITE_COUNT)
            .map(|ordinal| ordinal as u64 * site_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        offsets.last().unwrap() + site_bytes,
        buffer.storage.tensor.n_bytes()
    );

    let oracle = DeepSeekV4MhcOracle {
        buffer,
        identity: identity.clone(),
        payload_sha256: [5; 32],
        endpoint_sha256: [6; 32],
    };
    for (arm, producer_runs, producer_role, controls_role) in [
        (
            DeepSeekV4MhcDeleteArm::Current,
            true,
            DeepSeekV4MhcBufferRole::OrdinaryMixes,
            DeepSeekV4MhcBufferRole::OrdinaryMixes,
        ),
        (
            DeepSeekV4MhcDeleteArm::Producer,
            true,
            DeepSeekV4MhcBufferRole::OrdinaryMixes,
            DeepSeekV4MhcBufferRole::Oracle,
        ),
        (
            DeepSeekV4MhcDeleteArm::Zero,
            false,
            DeepSeekV4MhcBufferRole::None,
            DeepSeekV4MhcBufferRole::Oracle,
        ),
    ] {
        let mut execution = PackedMhcExecution::replay(arm, &oracle, true);
        for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
            for site in [DeepSeekV4MhcSiteKind::Attention, DeepSeekV4MhcSiteKind::Ffn] {
                let plan = execution
                    .site_plan(layer, site, &ordinary)
                    .expect("plan replay site");
                assert_eq!(plan.producer_runs, producer_runs);
                let producer_is_oracle = plan.producer_output.as_ref().is_some_and(|tensor| {
                    Retained::as_ptr(&tensor.buffer)
                        == Retained::as_ptr(&oracle.buffer.storage.tensor.buffer)
                });
                let controls_are_oracle = plan.controls_input.as_ref().is_some_and(|tensor| {
                    Retained::as_ptr(&tensor.buffer)
                        == Retained::as_ptr(&oracle.buffer.storage.tensor.buffer)
                });
                assert_eq!(
                    producer_is_oracle,
                    producer_role == DeepSeekV4MhcBufferRole::Oracle
                );
                assert_eq!(
                    controls_are_oracle,
                    controls_role == DeepSeekV4MhcBufferRole::Oracle
                );
            }
        }
        let (sites, replay_identity) = execution.finish().expect("finish replay");
        assert_eq!(replay_identity, identity);
        assert_eq!(sites.len(), 86);
        assert!(sites.iter().all(|site| {
            site.producer_runs == producer_runs
                && site.producer_output_role == producer_role
                && site.controls_input_role == controls_role
        }));
    }

    let mut wrong_order = PackedMhcExecution::capture(oracle.buffer.clone());
    wrong_order
        .bind_identity(identity.clone())
        .expect("bind wrong-order identity");
    assert!(
        wrong_order
            .site_plan(0, DeepSeekV4MhcSiteKind::Ffn, &ordinary)
            .is_err()
    );
    let mut incomplete = PackedMhcExecution::capture(oracle.buffer.clone());
    incomplete
        .bind_identity(identity)
        .expect("bind incomplete identity");
    incomplete
        .site_plan(0, DeepSeekV4MhcSiteKind::Attention, &ordinary)
        .expect("plan first incomplete site");
    assert!(incomplete.finish().is_err());
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn mhc_delete_oracle_requires_matching_capture_pair_and_endpoint() {
    let ctx = MetalContext::new().expect("create Metal context");
    let identity = DeepSeekV4MhcOracleIdentity {
        model_content_id: [1; 32],
        compatibility_id: [2; 32],
        token_sha256: [3; 32],
        policy_sha256: [4; 32],
        policy_manifest: "test-policy".to_string(),
        start_position: 0,
        n_tokens: 1,
        rms_epsilon_bits: 1.0e-6f32.to_bits(),
        hc_epsilon_bits: 1.0e-6f32.to_bits(),
        device_registry_id: ctx.device.registryID(),
        residency_tensor_count: 1_328,
        residency_source_bytes: 89_920_886_108,
        expert_count: 160,
    };
    let capture = || {
        let elements = DEEPSEEK_V4_HC_PARAMETER_COUNT * MHC_DELETE_SITE_COUNT;
        let tensor = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&vec![0.0f32; elements]),
            vec![
                DEEPSEEK_V4_HC_PARAMETER_COUNT as u64,
                1,
                MHC_DELETE_SITE_COUNT as u64,
            ],
            GgmlType::F32,
        )
        .expect("create deterministic capture buffer");
        let buffer = PackedMhcOracleBuffer::from_tensor(tensor, 1, ctx.device.registryID())
            .expect("plan deterministic capture views");
        DeepSeekV4MhcCapture {
            payload_sha256: buffer
                .current_payload_sha256()
                .expect("hash deterministic capture"),
            buffer,
            identity: identity.clone(),
        }
    };
    let evidence = |causal: u8| {
        let mut evidence = DeepSeekV4MhcEndpointEvidence {
            endpoint_logits_bits: vec![1],
            endpoint_hidden_bits: vec![2],
            endpoint_position: 1,
            endpoint_tokens: vec![35],
            endpoint_prefix_digest: [5; 32],
            endpoint_compatibility_id: [6; 32],
            endpoint_causal_digest: [causal; 32],
            endpoint_observation: DeepSeekV4SnapshotObservation::Available,
            continuation_logits_bits: vec![3],
            continuation_hidden_bits: vec![4],
            continuation_position: 2,
            continuation_tokens: vec![35, 35],
            continuation_prefix_digest: [7; 32],
            continuation_compatibility_id: [6; 32],
            continuation_causal_digest: [8; 32],
            continuation_observation: DeepSeekV4SnapshotObservation::Available,
            sha256: [0; 32],
        };
        evidence.refresh_sha256();
        evidence
    };
    let verified = |capture, evidence| DeepSeekV4MhcVerifiedCapture { capture, evidence };

    assert!(
        seal_mhc_delete_oracle_pair(
            verified(capture(), evidence(8)),
            verified(capture(), evidence(9)),
        )
        .is_err()
    );
    let expected_endpoint = evidence(8).sha256();
    let oracle = seal_mhc_delete_oracle_pair(
        verified(capture(), evidence(8)),
        verified(capture(), evidence(8)),
    )
    .expect("seal matching captures");
    assert_eq!(oracle.identity(), &identity);
    assert_eq!(oracle.endpoint_sha256(), expected_endpoint);
}

#[test]
fn packed_schedule_honors_runtime_expert_count_and_duplicate_slots() {
    const E: usize = 160;
    let n_tokens = 2;
    let expert_ids = vec![0, 0, 159, 5, 5, 7, 159, 0, 42, 42, 7, 7];
    let mut rows = Vec::with_capacity(expert_ids.len());
    let mut slots = Vec::with_capacity(expert_ids.len());
    let mut schedule = Vec::new();
    for expert in 0..E {
        let start = slots.len();
        for (slot, &routed_expert) in expert_ids.iter().enumerate() {
            if routed_expert == expert as i32 {
                rows.push((slot / MOE_TOP_K) as i32);
                slots.push(slot as i32);
            }
        }
        if slots.len() != start {
            schedule.push(ExpertBucket {
                expert,
                start,
                len: slots.len() - start,
            });
        }
    }

    validate_packed_expert_schedule(n_tokens, E, &expert_ids, &rows, &slots, &schedule).unwrap();
    let iq2_tiles = packed_grouped_iq2_mma16_tiles(n_tokens, &schedule).unwrap();
    assert!(
        iq2_tiles
            .iter()
            .any(|tile| tile.expert == 0 && tile.count == 3)
    );
    assert_eq!(schedule.last().unwrap().expert, E - 1);
    assert!(
        validate_packed_expert_schedule(n_tokens, E - 1, &expert_ids, &rows, &slots, &schedule,)
            .is_err()
    );
    assert!(
        validate_packed_expert_schedule(
            n_tokens,
            MOE_EXPERT_COUNT + 1,
            &expert_ids,
            &rows,
            &slots,
            &schedule,
        )
        .is_err()
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn packed_prefill_stage_resolver_closes_signed_overlaps() {
    let mut records = Vec::new();
    let mut timestamps = Vec::new();
    let mut cursor = 100u64;
    for (index, kind) in PACKED_PREFILL_STAGE_KINDS.into_iter().enumerate() {
        let start_sample = timestamps.len();
        let start_timestamp = if index == 1 { cursor - 5 } else { cursor };
        timestamps.push(start_timestamp);
        cursor = start_timestamp + 10 + index as u64;
        let end_sample = timestamps.len();
        timestamps.push(cursor);
        records.push(PackedPrefillPendingStageSample {
            layer: 0,
            kind,
            samples: Some((start_sample, end_sample)),
        });
        cursor += 3;
    }
    let profile =
        resolve_packed_prefill_layer_stage_samples(0, &records, &timestamps, 2.0, &[]).unwrap();
    assert_eq!(profile.layer, 0);
    assert_eq!(profile.command_gpu_ms, 2.0);
    assert_eq!(profile.stages.len(), PACKED_PREFILL_STAGE_KINDS.len());
    assert!(profile.sampled_span_ticks > 0);
    assert!(profile.raw_span_ms_assuming_ns > 0.0);
    assert!(profile.raw_coverage_assuming_ns > 0.0);
    assert!(profile.encoder_gap_ms_scaled > 0.0);
    assert!(profile.encoder_overlap_ms_scaled > 0.0);
    let stage_ms = profile
        .stages
        .iter()
        .zip(PACKED_PREFILL_STAGE_KINDS)
        .map(|(stage, expected)| {
            assert_eq!(stage.kind, expected);
            if let (Some(start), Some(end)) = (stage.start_timestamp, stage.end_timestamp) {
                assert_eq!(stage.duration_ticks, end - start);
            } else {
                panic!("physical test stage {expected:?} has no samples");
            }
            stage.duration_ms_scaled
        })
        .sum::<f64>();
    assert!(
        (stage_ms + profile.encoder_gap_ms_scaled - profile.encoder_overlap_ms_scaled - 2.0).abs()
            < 1e-12
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn packed_post_route_stage_resolver_closes_signed_overlaps() {
    let mut records = Vec::new();
    let mut timestamps = Vec::new();
    let mut cursor = 100u64;
    for (index, kind) in PACKED_POST_ROUTE_STAGE_KINDS.into_iter().enumerate() {
        let start_sample = timestamps.len();
        let start_timestamp = if index == 2 { cursor - 4 } else { cursor };
        timestamps.push(start_timestamp);
        cursor = start_timestamp + 20 + index as u64;
        let end_sample = timestamps.len();
        timestamps.push(cursor);
        records.push(PackedPostRoutePendingStageSample {
            layer: 0,
            kind,
            start_sample,
            end_sample,
        });
        cursor += 3;
    }
    let profile = resolve_packed_post_route_layer_stage_samples(
        0,
        &records,
        &PACKED_POST_ROUTE_STAGE_KINDS,
        &timestamps,
        2.0,
    )
    .unwrap();
    assert_eq!(profile.layer, 0);
    assert_eq!(profile.command_gpu_ms, 2.0);
    assert_eq!(profile.stages.len(), PACKED_POST_ROUTE_STAGE_KINDS.len());
    assert!(profile.sampled_span_ticks > 0);
    assert!(profile.raw_span_ms_assuming_ns > 0.0);
    assert!(profile.raw_coverage_assuming_ns > 0.0);
    assert!(profile.encoder_gap_ms_scaled > 0.0);
    assert!(profile.encoder_overlap_ms_scaled > 0.0);
    let stage_ms = profile
        .stages
        .iter()
        .zip(PACKED_POST_ROUTE_STAGE_KINDS)
        .map(|(stage, expected)| {
            assert_eq!(stage.kind, expected);
            assert_eq!(
                stage.duration_ticks,
                stage.end_timestamp - stage.start_timestamp
            );
            stage.duration_ms_scaled
        })
        .sum::<f64>();
    assert!(
        (stage_ms + profile.encoder_gap_ms_scaled
            - profile.encoder_overlap_ms_scaled
            - profile.command_gpu_ms)
            .abs()
            < 1e-12
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn packed_post_route_stage_resolver_accepts_bm16_split() {
    let mut records = Vec::new();
    let mut timestamps = Vec::new();
    let mut cursor = 500u64;
    for &kind in &PACKED_BM16_POST_ROUTE_STAGE_KINDS {
        let start_sample = timestamps.len();
        timestamps.push(cursor);
        cursor += 17;
        let end_sample = timestamps.len();
        timestamps.push(cursor);
        records.push(PackedPostRoutePendingStageSample {
            layer: 3,
            kind,
            start_sample,
            end_sample,
        });
        cursor += 2;
    }
    let profile = resolve_packed_post_route_layer_stage_samples(
        3,
        &records,
        &PACKED_BM16_POST_ROUTE_STAGE_KINDS,
        &timestamps,
        4.0,
    )
    .unwrap();
    assert_eq!(profile.layer, 3);
    assert_eq!(profile.stages.len(), 6);
    assert_eq!(
        profile.stages[0].kind,
        PackedPostRouteStageKind::RoutedGateUp
    );
    assert_eq!(
        profile.stages[1].kind,
        PackedPostRouteStageKind::RoutedSwiGlu
    );
    assert_eq!(profile.stages[2].kind, PackedPostRouteStageKind::RoutedDown);
    let accounted = profile
        .stages
        .iter()
        .map(|stage| stage.duration_ms_scaled)
        .sum::<f64>()
        + profile.encoder_gap_ms_scaled
        - profile.encoder_overlap_ms_scaled;
    assert!((accounted - profile.command_gpu_ms).abs() < 1e-12);
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn packed_prefill_stage_resolver_represents_empty_stage_explicitly() {
    let records = [
        PackedPrefillPendingStageSample {
            layer: 0,
            kind: PackedPrefillStageKind::BeforeAttentionBody,
            samples: Some((0, 1)),
        },
        PackedPrefillPendingStageSample {
            layer: 0,
            kind: PackedPrefillStageKind::SparseIndexerPrepare,
            samples: None,
        },
        PackedPrefillPendingStageSample {
            layer: 0,
            kind: PackedPrefillStageKind::SparseIndexerScore,
            samples: None,
        },
        PackedPrefillPendingStageSample {
            layer: 0,
            kind: PackedPrefillStageKind::SparseSelection,
            samples: None,
        },
        PackedPrefillPendingStageSample {
            layer: 0,
            kind: PackedPrefillStageKind::AttentionCore,
            samples: Some((2, 3)),
        },
        PackedPrefillPendingStageSample {
            layer: 0,
            kind: PackedPrefillStageKind::InverseRope,
            samples: Some((4, 5)),
        },
        PackedPrefillPendingStageSample {
            layer: 0,
            kind: PackedPrefillStageKind::AttentionOutputProjections,
            samples: Some((6, 7)),
        },
        PackedPrefillPendingStageSample {
            layer: 0,
            kind: PackedPrefillStageKind::AfterAttentionOutput,
            samples: Some((8, 9)),
        },
    ];
    let profile = resolve_packed_prefill_layer_stage_samples(
        0,
        &records,
        &[100, 110, 113, 120, 123, 130, 133, 140, 143, 150],
        1.0,
        &[
            PackedPrefillStageKind::SparseIndexerPrepare,
            PackedPrefillStageKind::SparseIndexerScore,
            PackedPrefillStageKind::SparseSelection,
        ],
    )
    .unwrap();
    for stage in &profile.stages[1..=3] {
        assert_eq!(stage.start_timestamp, None);
        assert_eq!(stage.end_timestamp, None);
        assert_eq!(stage.duration_ticks, 0);
        assert_eq!(stage.duration_ms_scaled, 0.0);
    }
    assert_eq!(profile.transitions.len(), 4);
    assert_eq!(
        profile.transitions[0].from,
        PackedPrefillStageKind::BeforeAttentionBody
    );
    assert_eq!(
        profile.transitions[0].to,
        PackedPrefillStageKind::AttentionCore
    );
}

fn grouped_test_bank(
    ctx: &MetalContext,
    dtype: GgmlType,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    seed: usize,
) -> MetalTensor {
    let (block_elements, block_bytes) = ggml_type_layout(dtype).unwrap();
    let block_elements = block_elements as usize;
    let block_bytes = block_bytes as usize;
    let blocks = n_in * n_out * expert_count / block_elements;
    let mut payload = vec![0u8; blocks * block_bytes];
    for block in 0..blocks {
        let start = block * block_bytes;
        for byte in 0..block_bytes {
            payload[start + byte] = (block * 31 + byte * 13 + seed * 19 + 5) as u8;
        }
        let scale = half::f16::from_f32(0.00390625 * (1 + (block + seed) % 7) as f32)
            .to_bits()
            .to_le_bytes();
        match dtype {
            GgmlType::Q3_K => {
                payload[start + block_bytes - 2..start + block_bytes].copy_from_slice(&scale);
            }
            GgmlType::Q4_K => {
                payload[start..start + 2].copy_from_slice(&scale);
                payload[start + 2..start + 4].copy_from_slice(
                    &half::f16::from_f32(0.001953125 * (1 + (block + seed) % 5) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
            _ => payload[start..start + 2].copy_from_slice(&scale),
        }
    }
    MetalTensor {
        buffer: ctx.buffer_from(&payload).unwrap(),
        offset: 0,
        shape: vec![n_in as u64, n_out as u64, expert_count as u64],
        dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn grouped_guarded_f32(ctx: &MetalContext, shape: Vec<u64>, poison: f32) -> MetalTensor {
    const GUARD: usize = 64;
    let elements = shape.iter().product::<u64>() as usize;
    let mut bytes = vec![0xa5u8; GUARD];
    let poison_values = vec![poison; elements];
    bytes.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&poison_values));
    bytes.extend_from_slice(&[0x5au8; GUARD]);
    MetalTensor {
        buffer: ctx.buffer_from(&bytes).unwrap(),
        offset: GUARD as u64,
        shape,
        dtype: GgmlType::F32,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn assert_grouped_guards(label: &str, tensor: &MetalTensor) {
    const GUARD: usize = 64;
    let base = tensor.buffer.contents().as_ptr().cast::<u8>();
    let prefix =
        unsafe { std::slice::from_raw_parts(base.add(tensor.offset as usize - GUARD), GUARD) };
    let suffix = unsafe {
        std::slice::from_raw_parts(
            base.add(tensor.offset as usize + tensor.n_bytes() as usize),
            GUARD,
        )
    };
    assert!(
        prefix.iter().all(|&byte| byte == 0xa5),
        "{label} prefix guard changed: {prefix:?}"
    );
    assert!(
        suffix.iter().all(|&byte| byte == 0x5a),
        "{label} suffix guard changed: {suffix:?}"
    );
}

fn q8_precision_test_weight(ctx: &MetalContext, n_in: usize, n_out: usize) -> MetalTensor {
    const SCALE_BITS: [u16; 6] = [0x0001, 0x03ff, 0x0400, 0x1a24, 0x2e66, 0x3800];
    const QUANTS: [i8; 12] = [-128, -127, -63, -1, 0, 1, 17, 63, 126, 127, -31, 47];
    let blocks = n_in * n_out / 32;
    let mut payload = vec![0u8; blocks * 34];
    for block in 0..blocks {
        let start = block * 34;
        payload[start..start + 2]
            .copy_from_slice(&SCALE_BITS[block % SCALE_BITS.len()].to_le_bytes());
        for index in 0..32 {
            payload[start + 2 + index] = QUANTS[(block * 7 + index * 5) % QUANTS.len()] as u8;
        }
    }
    MetalTensor::from_bytes(
        ctx,
        &payload,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q8_0,
    )
    .unwrap()
}

fn q8_precision_test_input(elements: usize) -> Vec<f32> {
    const SPECIAL: [f32; 12] = [
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        0.000_061_005_354,
        -0.000_061_005_354,
        0.333_251_95,
        -0.333_251_95,
        1.000_488_3,
        -1.000_488_3,
        0.125_030_52,
        -0.125_030_52,
    ];
    (0..elements)
        .map(|index| {
            if index % 5 == 0 {
                SPECIAL[(index / 5) % SPECIAL.len()]
            } else {
                ((index * 37 + index / 11 + 3) % 509) as f32 * 0.001 - 0.254
            }
        })
        .collect()
}

fn q8_output_test_weight(
    ctx: &MetalContext,
    n_in: usize,
    n_out: usize,
    seed: usize,
) -> MetalTensor {
    let blocks = n_in * n_out / 32;
    let mut payload = vec![0u8; blocks * 34];
    for block in 0..blocks {
        let start = block * 34;
        let scale = half::f16::from_f32(0.003 + ((block + seed) % 19) as f32 * 0.0002);
        payload[start..start + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
        for index in 0..32 {
            payload[start + 2 + index] =
                (((block * 13 + index * 17 + seed * 7) % 127) as i8 - 63) as u8;
        }
    }
    MetalTensor::from_bytes(
        ctx,
        &payload,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q8_0,
    )
    .unwrap()
}

fn q8_differential(actual: &[f32], expected: &[f32]) -> (f64, f64, f32) {
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut error = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        let actual_f64 = actual as f64;
        let expected_f64 = expected as f64;
        let delta = actual - expected;
        dot += actual_f64 * expected_f64;
        actual_norm += actual_f64 * actual_f64;
        expected_norm += expected_f64 * expected_f64;
        error += (delta as f64) * (delta as f64);
        max_abs = max_abs.max(delta.abs());
    }
    (
        dot / (actual_norm * expected_norm).sqrt(),
        (error / expected_norm).sqrt(),
        max_abs,
    )
}

fn grouped_test_schedule_for_experts(
    n_tokens: usize,
    expert_count: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<ExpertBucket>) {
    let mut expert_ids = Vec::with_capacity(n_tokens * MOE_TOP_K);
    for token in 0..n_tokens {
        expert_ids.push((expert_count - 1) as i32);
        for slot in 1..MOE_TOP_K {
            expert_ids.push(((token * (MOE_TOP_K - 1) + slot - 1) % (expert_count - 1)) as i32);
        }
    }
    let mut assignments = (0..expert_count)
        .map(|_| Vec::<(usize, usize)>::new())
        .collect::<Vec<_>>();
    for token in 0..n_tokens {
        for slot in 0..MOE_TOP_K {
            let route_slot = token * MOE_TOP_K + slot;
            assignments[expert_ids[route_slot] as usize].push((token, route_slot));
        }
    }
    let mut rows = Vec::with_capacity(n_tokens * MOE_TOP_K);
    let mut slots = Vec::with_capacity(n_tokens * MOE_TOP_K);
    let mut schedule = Vec::new();
    for (expert, assignments) in assignments.into_iter().enumerate() {
        if assignments.is_empty() {
            continue;
        }
        let start = rows.len();
        for (row, slot) in assignments {
            rows.push(row as i32);
            slots.push(slot as i32);
        }
        schedule.push(ExpertBucket {
            expert,
            start,
            len: rows.len() - start,
        });
    }
    validate_packed_expert_schedule(
        n_tokens,
        expert_count,
        &expert_ids,
        &rows,
        &slots,
        &schedule,
    )
    .unwrap();
    (expert_ids, rows, slots, schedule)
}

fn grouped_test_schedule(n_tokens: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<ExpertBucket>) {
    grouped_test_schedule_for_experts(n_tokens, MOE_EXPERT_COUNT)
}

#[test]
fn packed_grouped_expert_mode_has_an_isolated_fail_closed_rollback() {
    assert_eq!(
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
        PACKED_GROUPED_EXPERT_MAX_TOKENS
    );
    let mma16 = PackedExpertPolicy::GroupedIq2XsIq3XxsMma16QualifiedChunk;
    assert!(!mma16.uses_iq2_mma16(127));
    assert!(mma16.uses_iq2_mma16(PACKED_GROUPED_IQ2_MMA16_NARROW_TOKENS));
    assert!(!mma16.uses_iq2_mma16(129));
    assert!(!mma16.uses_iq2_mma16(255));
    assert!(mma16.uses_iq2_mma16(256));
    assert!(mma16.uses_iq2_mma16(337));
    assert!(mma16.uses_iq2_mma16(PACKED_GROUPED_IQ2_MMA16_MEDIUM_TOKENS));
    assert!(mma16.uses_iq2_mma16(PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS - 1));
    assert!(mma16.uses_iq2_mma16(PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS));
    assert!(!mma16.uses_iq2_mma16(PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS + 1));
    for tokens in [256, 337, 4_095] {
        for (bytes, experts) in [
            (
                PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
                MOE_EXPERT_COUNT,
            ),
            (PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES, 216),
        ] {
            assert!(packed_grouped_iq2_matrix_scope_qualified(
                PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE,
                1_328,
                bytes,
                experts,
                tokens,
            ));
        }
    }
    for (device, tensors, bytes, experts) in [
        (
            "Apple M3 Max",
            1_328,
            PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
            216,
        ),
        (
            PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE,
            1_327,
            PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
            216,
        ),
        (
            PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            216,
        ),
        (
            PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            160,
        ),
        (
            PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
            256,
        ),
        (
            PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
            216,
        ),
    ] {
        assert!(!packed_grouped_iq2_matrix_scope_qualified(
            device, tensors, bytes, experts, 337,
        ));
    }
    assert!(!packed_grouped_iq2_matrix_scope_qualified(
        PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
        216,
        255,
    ));
    assert!(!packed_grouped_iq2_matrix_scope_qualified(
        PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        MOE_EXPERT_COUNT,
        255,
    ));
    assert!(!mma16.uses_iq3_target());
    let gpu_compact = mma16.with_iq3_target();
    assert_eq!(
        gpu_compact,
        PackedExpertPolicy::GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk
    );
    assert!(gpu_compact.uses_iq2_mma16(PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS));
    assert!(gpu_compact.uses_iq3_target());
    assert!(packed_grouped_iq_expert_count_qualified(216));
    assert!(packed_grouped_iq_expert_count_qualified(MOE_EXPERT_COUNT));
    assert!(!packed_grouped_iq_expert_count_qualified(160));
    assert!(!packed_grouped_iq_expert_count_qualified(200));
    assert!(
        packed_grouped_expert_scope(
            PackedGroupedExpertMode::Auto,
            PACKED_GROUPED_EXPERT_MAX_TOKENS,
        )
        .unwrap()
    );
    assert!(
        packed_grouped_expert_scope(
            PackedGroupedExpertMode::Auto,
            DEEPSEEK_V4_PREFILL_MAX_TOKENS,
        )
        .unwrap()
    );
    assert!(
        packed_grouped_expert_scope(
            PackedGroupedExpertMode::ForceOn,
            DEEPSEEK_V4_PREFILL_MAX_TOKENS + 1,
        )
        .unwrap_err()
        .to_string()
        .contains("qualified through 4096 tokens")
    );
    assert_eq!(
        parse_packed_grouped_expert_mode(None),
        PackedGroupedExpertMode::Auto
    );
    for enabled in ["1", "true", "TRUE", "yes", "YES"] {
        assert_eq!(
            parse_packed_grouped_expert_mode(Some(enabled)),
            PackedGroupedExpertMode::ForceOn
        );
    }
    for disabled in ["0", "false", "FALSE", "no", "NO", "invalid"] {
        assert_eq!(
            parse_packed_grouped_expert_mode(Some(disabled)),
            PackedGroupedExpertMode::ForceOff
        );
    }
    for automatic in ["auto", "AUTO"] {
        assert_eq!(
            parse_packed_grouped_expert_mode(Some(automatic)),
            PackedGroupedExpertMode::Auto
        );
    }
}

#[test]
fn packed_q8_qb_matrix_policy_and_scope_are_explicit() {
    assert_eq!(
        parse_packed_q8_qb_policy(None).unwrap(),
        PackedQ8MatrixPolicy::Auto
    );
    assert_eq!(
        parse_packed_q8_qb_policy(Some("auto")).unwrap(),
        PackedQ8MatrixPolicy::Auto
    );
    assert_eq!(
        parse_packed_q8_qb_policy(Some("exact")).unwrap(),
        PackedQ8MatrixPolicy::Exact
    );
    assert_eq!(
        parse_packed_q8_qb_policy(Some("f32_matrix")).unwrap(),
        PackedQ8MatrixPolicy::F32Matrix
    );
    assert_eq!(
        parse_packed_q8_qb_policy(Some("wide_f32_matrix")).unwrap(),
        PackedQ8MatrixPolicy::WideF32Matrix
    );
    assert_eq!(
        resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::Auto, true, 512),
        Q8PrecisionProjection::WideF32Matrix
    );
    assert_eq!(
        resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::Auto, true, 337),
        Q8PrecisionProjection::F32Matrix
    );
    assert_eq!(
        resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::Auto, false, 337),
        Q8PrecisionProjection::Exact
    );
    assert_eq!(
        resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::Exact, true, 512),
        Q8PrecisionProjection::Exact
    );
    assert_eq!(
        resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::F32Matrix, false, 337),
        Q8PrecisionProjection::F32Matrix
    );
    assert_eq!(
        resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::WideF32Matrix, false, 512),
        Q8PrecisionProjection::WideF32Matrix
    );
    assert_eq!(
        resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::WideF32Matrix, false, 337),
        Q8PrecisionProjection::F32Matrix
    );
    let matrix = Q8PrecisionProjection::F32Matrix;
    assert!(matrix.uses_full_chunk_f32(337));
    assert!(matrix.uses_full_chunk_f32(512));
    assert!(matrix.uses_full_chunk_f32(PACKED_MATRIX_MIN_TOKENS));
    assert!(matrix.uses_full_chunk_f32(DEEPSEEK_V4_PREFILL_MAX_TOKENS));
    let wide = Q8PrecisionProjection::WideF32Matrix;
    assert!(wide.uses_full_chunk_f32(512));
    assert!(!wide.uses_full_chunk_f32(337));
    assert!(wide.uses_full_chunk_f32(PACKED_MATRIX_MIN_TOKENS));
    assert!(wide.uses_full_chunk_f32(DEEPSEEK_V4_PREFILL_MAX_TOKENS));
    assert!(parse_packed_q8_qb_policy(Some("half_matrix")).is_err());
}

#[test]
fn packed_q8_output_matrix_policy_is_explicit_and_qualified_width_only() {
    assert_eq!(
        parse_packed_q8_output_policy(None).unwrap(),
        PackedQ8MatrixPolicy::Auto
    );
    assert_eq!(
        parse_packed_q8_output_policy(Some("auto")).unwrap(),
        PackedQ8MatrixPolicy::Auto
    );
    assert_eq!(
        parse_packed_q8_output_policy(Some("exact")).unwrap(),
        PackedQ8MatrixPolicy::Exact
    );
    assert_eq!(
        parse_packed_q8_output_policy(Some("f32_matrix")).unwrap(),
        PackedQ8MatrixPolicy::F32Matrix
    );
    assert_eq!(
        parse_packed_q8_output_policy(Some("wide_f32_matrix")).unwrap(),
        PackedQ8MatrixPolicy::WideF32Matrix
    );
    assert_eq!(
        resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::Auto, true, 512),
        Q8PrecisionProjection::WideF32Matrix
    );
    assert_eq!(
        resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::Auto, true, 337),
        Q8PrecisionProjection::F32Matrix
    );
    assert_eq!(
        resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::Auto, false, 337),
        Q8PrecisionProjection::Exact
    );
    assert_eq!(
        resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::WideF32Matrix, false, 337),
        Q8PrecisionProjection::F32Matrix
    );
    let matrix = Q8PrecisionProjection::F32Matrix;
    assert!(matrix.uses_full_chunk_f32(512));
    assert!(matrix.uses_full_chunk_f32(PACKED_MATRIX_MIN_TOKENS));
    assert!(matrix.uses_full_chunk_f32(DEEPSEEK_V4_PREFILL_MAX_TOKENS));
    assert!(parse_packed_q8_output_policy(Some("half_matrix")).is_err());
}

#[test]
fn packed_q8_compressor_matrix_scope_is_exact() {
    let qualified = |device, tensors, bytes, experts, tokens| {
        packed_q8_compressor_matrix_scope_qualified(device, tensors, bytes, experts, tokens)
    };
    for tokens in [256, 337, 512, 2_048, 4_095, DEEPSEEK_V4_PREFILL_MAX_TOKENS] {
        for (bytes, experts) in [
            (
                PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
                MOE_EXPERT_COUNT,
            ),
            (PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES, 160),
            (PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES, 216),
        ] {
            assert!(qualified(
                PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
                1_328,
                bytes,
                experts,
                tokens,
            ));
        }
    }
    for (bytes, experts) in [
        (
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
            MOE_EXPERT_COUNT,
        ),
        (PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES, 160),
        (PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES, 216),
    ] {
        assert!(!qualified(
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            bytes,
            experts,
            255,
        ));
    }
    assert!(!qualified(
        "Apple M3 Max",
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        256,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_327,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        256,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES - 1,
        256,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES - 1,
        160,
        PACKED_MATRIX_MIN_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES - 1,
        216,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        216,
        337,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
        256,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ));
    assert!(!packed_gpu_route_compact_scope_qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        MOE_EXPERT_COUNT,
        337,
    ));
    assert!(packed_gpu_route_compact_scope_qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        MOE_EXPERT_COUNT,
        PACKED_MATRIX_MIN_TOKENS,
    ));
}

#[test]
fn packed_router_e8p32_scope_is_exactly_k160_m4() {
    assert_eq!(parse_packed_router_e8p32_strict_env(None), Ok(true));
    for value in ["1", "true", "TRUE", " yes ", "On"] {
        assert_eq!(parse_packed_router_e8p32_strict_env(Some(value)), Ok(true));
    }
    for value in ["0", "false", "FALSE", " no ", "Off"] {
        assert_eq!(parse_packed_router_e8p32_strict_env(Some(value)), Ok(false));
    }
    for value in ["", "ture", "2"] {
        assert!(parse_packed_router_e8p32_strict_env(Some(value)).is_err());
    }
    let qualified = |device, tensors, bytes, experts, tokens| {
        packed_router_e8p32_scope_qualified(device, tensors, bytes, experts, tokens)
    };
    for tokens in [
        256,
        337,
        512,
        PACKED_MATRIX_MIN_TOKENS,
        4_095,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ] {
        assert!(qualified(
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            160,
            tokens,
        ));
    }
    assert!(!qualified(
        "Apple M3 Max",
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        PACKED_MATRIX_MIN_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_327,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        PACKED_MATRIX_MIN_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES - 1,
        160,
        PACKED_MATRIX_MIN_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        216,
        PACKED_MATRIX_MIN_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        255,
    ));
}

#[test]
fn packed_q8_qa_kv_matrix_policy_is_scoped_and_fail_closed() {
    assert_eq!(
        parse_packed_q8_qa_kv_matrix_mode(None),
        Ok(PackedQaKvMatrixMode::Both)
    );
    for (value, expected) in [
        ("off", PackedQaKvMatrixMode::Off),
        ("qa", PackedQaKvMatrixMode::Qa),
        ("kv", PackedQaKvMatrixMode::Kv),
        ("both", PackedQaKvMatrixMode::Both),
        (" TRUE ", PackedQaKvMatrixMode::Both),
    ] {
        assert_eq!(parse_packed_q8_qa_kv_matrix_mode(Some(value)), Ok(expected));
    }
    assert!(parse_packed_q8_qa_kv_matrix_mode(Some("qk")).is_err());

    let qualified = |device, tensors, bytes, experts, tokens| {
        packed_q8_qa_kv_matrix_scope_qualified(device, tensors, bytes, experts, tokens)
    };
    for tokens in [
        256,
        337,
        512,
        PACKED_MATRIX_MIN_TOKENS,
        4_095,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ] {
        assert!(qualified(
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            160,
            tokens,
        ));
    }
    assert!(!qualified(
        "Apple M3 Max",
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        PACKED_MATRIX_MIN_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
        216,
        PACKED_MATRIX_MIN_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        255,
    ));
}

#[test]
fn packed_mxfp4_matrix_scope_covers_both_fresh_and_k216_full_widths() {
    let qualified = |device, tensors, bytes, experts, tokens| {
        packed_mxfp4_matrix_scope_qualified(device, tensors, bytes, experts, tokens)
    };
    for tokens in [PACKED_MATRIX_MIN_TOKENS, DEEPSEEK_V4_PREFILL_MAX_TOKENS] {
        assert!(qualified(
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
            MOE_EXPERT_COUNT,
            tokens,
        ));
        assert!(qualified(
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
            216,
            tokens,
        ));
    }
    assert!(!qualified(
        "Apple M3 Max",
        1_328,
        PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
        216,
        PACKED_MATRIX_MIN_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_327,
        PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
        216,
        PACKED_MATRIX_MIN_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES - 1,
        216,
        PACKED_MATRIX_MIN_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
        160,
        PACKED_MATRIX_MIN_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
        216,
        1_024,
    ));
    for tokens in [PACKED_MATRIX_MIN_TOKENS, DEEPSEEK_V4_PREFILL_MAX_TOKENS] {
        assert!(!qualified(
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            160,
            tokens,
        ));
    }
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        MOE_EXPERT_COUNT,
        1_024,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        216,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ));
}

#[test]
fn packed_grouped_q3q4_scope_is_exact() {
    let qualified = |device, tensors, bytes, experts, tokens, gate, up, down| {
        packed_grouped_q3q4_scope_qualified(device, tensors, bytes, experts, tokens, gate, up, down)
    };
    for tokens in [
        PACKED_GROUPED_Q3Q4_NARROW_TOKENS,
        256,
        337,
        512,
        PACKED_MATRIX_MIN_TOKENS,
        4_095,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ] {
        assert!(qualified(
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            160,
            tokens,
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
        ));
    }
    for (device, tensors, bytes, experts, gate, up, down) in [
        (
            "Apple M3 Max",
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            160,
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
        ),
        (
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_327,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            160,
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
        ),
        (
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES - 1,
            160,
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
        ),
        (
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES + 1,
            160,
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
        ),
        (
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            159,
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
        ),
        (
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            161,
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
        ),
        (
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            160,
            GgmlType::Q4_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
        ),
        (
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            160,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q4_K,
        ),
        (
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
            160,
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            GgmlType::Q3_K,
        ),
    ] {
        assert!(!qualified(
            device,
            tensors,
            bytes,
            experts,
            PACKED_GROUPED_Q3Q4_NARROW_TOKENS,
            gate,
            up,
            down,
        ));
    }
    assert!(!packed_shared_route_overlap_scope_qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        PACKED_GROUPED_Q3Q4_NARROW_TOKENS,
        GgmlType::Q3_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
    assert!(packed_shared_route_overlap_scope_qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        256,
        GgmlType::Q3_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        PACKED_GROUPED_Q3Q4_NARROW_TOKENS - 1,
        GgmlType::Q3_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        PACKED_GROUPED_Q3Q4_NARROW_TOKENS + 1,
        GgmlType::Q3_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        255,
        GgmlType::Q3_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        256,
        PACKED_MATRIX_MIN_TOKENS,
        GgmlType::Q3_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
        216,
        PACKED_MATRIX_MIN_TOKENS,
        GgmlType::Q3_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
    assert!(!qualified(
        "Apple M3 Max",
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        PACKED_MATRIX_MIN_TOKENS,
        GgmlType::Q3_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_327,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        PACKED_MATRIX_MIN_TOKENS,
        GgmlType::Q3_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        160,
        PACKED_MATRIX_MIN_TOKENS,
        GgmlType::Q3_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        PACKED_MATRIX_MIN_TOKENS,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q4_K,
    ));
}

#[test]
fn packed_indexer_q_matrix_scope_is_4096_only() {
    let qualified = |device, tensors, bytes, experts, tokens| {
        packed_indexer_q_matrix_scope_qualified(device, tensors, bytes, experts, tokens)
    };
    assert!(qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        256,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ));
    assert!(qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES,
        160,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ));
    assert!(qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES,
        216,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        256,
        PACKED_MATRIX_MIN_TOKENS,
    ));
    assert!(!qualified(
        "Apple M3 Max",
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        256,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ));
    assert!(!qualified(
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES + 1,
        256,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    ));
}

#[test]
fn packed_grouped_schedule_preflight_and_tile_bound_are_exact() {
    let (expert_ids, rows, slots, schedule) = grouped_test_schedule(128);
    let tiles = packed_grouped_expert_tiles(128, &schedule).unwrap();
    assert_eq!(tiles.len(), 259);
    assert_eq!(
        tiles.iter().map(|tile| tile.count as usize).sum::<usize>(),
        768
    );
    assert!(tiles.iter().all(|tile| (1..=32).contains(&tile.count)));
    #[cfg(feature = "dsv4-diagnostics")]
    assert_eq!(
        packed_post_route_expert_ids(128, MOE_EXPERT_COUNT, &expert_ids, &rows, &slots, &schedule,)
            .unwrap(),
        expert_ids
            .iter()
            .map(|&expert| expert as u16)
            .collect::<Vec<_>>()
    );

    let mut bad_rows = rows.clone();
    bad_rows[0] ^= 1;
    assert!(
        validate_packed_expert_schedule(
            128,
            MOE_EXPERT_COUNT,
            &expert_ids,
            &bad_rows,
            &slots,
            &schedule,
        )
        .is_err()
    );
    let mut bad_slots = slots.clone();
    bad_slots[1] = bad_slots[0];
    assert!(
        validate_packed_expert_schedule(
            128,
            MOE_EXPERT_COUNT,
            &expert_ids,
            &rows,
            &bad_slots,
            &schedule,
        )
        .is_err()
    );
    let mut bad_ids = expert_ids.clone();
    bad_ids[slots[0] as usize] ^= 1;
    assert!(
        validate_packed_expert_schedule(128, MOE_EXPERT_COUNT, &bad_ids, &rows, &slots, &schedule,)
            .is_err()
    );

    let mut cursor = 0usize;
    let maximum = (0..MOE_EXPERT_COUNT)
        .map(|expert| {
            let len = if expert < 16 { 33 } else { 1 };
            let bucket = ExpertBucket {
                expert,
                start: cursor,
                len,
            };
            cursor += len;
            bucket
        })
        .collect::<Vec<_>>();
    assert_eq!(cursor, 128 * MOE_TOP_K);
    assert_eq!(
        packed_grouped_expert_tiles(128, &maximum).unwrap().len(),
        272
    );
    assert_eq!(
        packed_grouped_iq2_mma16_tiles(128, &maximum).unwrap().len(),
        288
    );

    let mut cursor = 0usize;
    let maximum = (0..MOE_EXPERT_COUNT)
        .map(|expert| {
            let len = match expert {
                0..5 => 4_065,
                5 => 4_001,
                _ => 1,
            };
            let bucket = ExpertBucket {
                expert,
                start: cursor,
                len,
            };
            cursor += len;
            bucket
        })
        .collect::<Vec<_>>();
    assert_eq!(cursor, DEEPSEEK_V4_PREFILL_MAX_TOKENS * MOE_TOP_K);
    assert_eq!(
        packed_grouped_expert_tiles(DEEPSEEK_V4_PREFILL_MAX_TOKENS, &maximum)
            .unwrap()
            .len(),
        PACKED_GROUPED_EXPERT_MAX_TILES
    );

    let mut cursor = 0usize;
    let mma16_maximum = (0..MOE_EXPERT_COUNT)
        .map(|expert| {
            let len = match expert {
                0..5 => 4_081,
                5 => 3_921,
                _ => 1,
            };
            let bucket = ExpertBucket {
                expert,
                start: cursor,
                len,
            };
            cursor += len;
            bucket
        })
        .collect::<Vec<_>>();
    assert_eq!(cursor, PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS * MOE_TOP_K);
    assert_eq!(
        packed_grouped_iq2_mma16_tiles(PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS, &mma16_maximum,)
            .unwrap()
            .len(),
        PACKED_GROUPED_IQ2_MMA16_MAX_TILES
    );
    assert_eq!(PACKED_GROUPED_IQ2_MMA16_MAX_TILES, 1_776);
    assert_eq!(PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS, 5_328);
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn packed_post_route_expert_counts_pin_panel_boundaries() {
    let lengths = [31usize, 32, 33, 128, 128, 128, 128, 128, 32];
    let mut cursor = 0usize;
    let schedule = lengths
        .into_iter()
        .enumerate()
        .map(|(expert, len)| {
            let bucket = ExpertBucket {
                expert,
                start: cursor,
                len,
            };
            cursor += len;
            bucket
        })
        .collect::<Vec<_>>();
    assert_eq!(cursor, 128 * MOE_TOP_K);

    let counts = packed_post_route_expert_counts(128, &schedule, MOE_EXPERT_COUNT).unwrap();
    assert_eq!(&counts[..lengths.len()], &lengths.map(|len| len as u16));
    assert!(counts[lengths.len()..].iter().all(|&count| count == 0));
    assert_eq!(
        counts
            .iter()
            .map(|&count| usize::from(count).div_ceil(32))
            .sum::<usize>(),
        25
    );

    let copy_schedule = || {
        schedule
            .iter()
            .map(|bucket| ExpertBucket {
                expert: bucket.expert,
                start: bucket.start,
                len: bucket.len,
            })
            .collect::<Vec<_>>()
    };
    let mut duplicate = copy_schedule();
    duplicate[1].expert = duplicate[0].expert;
    assert!(packed_post_route_expert_counts(128, &duplicate, MOE_EXPERT_COUNT).is_err());

    let mut oversized = copy_schedule();
    oversized[0].len = 129;
    assert!(packed_post_route_expert_counts(128, &oversized, MOE_EXPERT_COUNT).is_err());
    assert!(packed_post_route_expert_counts(128, &schedule[..8], MOE_EXPERT_COUNT).is_err());
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn packed_grouped_iq2_count_census_fixture_is_canonical() {
    const DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-grouped-iq2-counts:v1\0";
    const GROUPED_LAYERS: [usize; 25] = [
        0, 2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 35, 36, 37, 38, 39,
        41,
    ];

    fn hex(digest: impl AsRef<[u8]>) -> String {
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    let fixture = packed_grouped_iq2_count_census_fixture();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.count_payload_domain_hex,
        "7177656e2d6c6c6d3a647376343a7061636b65642d67726f757065642d6971322d636f756e74733a763100"
    );
    assert_eq!(
        fixture.count_payload_sha256,
        "0ab9925350288116288794f3d7f5081595dfad4ffdb9671a9a146f27568358a6"
    );
    assert_eq!(
        fixture.model_content_id,
        "ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2"
    );
    assert_eq!(
        fixture.prompt_token_ids_sha256,
        "b57816bcb0d5fdf5a8e2ddc7a0afe9e57fb0ca6ffc2b849285e1635d04772843"
    );
    assert_eq!(fixture.n_tokens, 128);
    assert_eq!(fixture.top_k, MOE_TOP_K);
    assert_eq!(fixture.expert_count, MOE_EXPERT_COUNT);
    assert_eq!(fixture.grouped_layer_ids, GROUPED_LAYERS);
    assert_eq!(fixture.layers.len(), GROUPED_LAYERS.len());

    let mut payload = Sha256::new();
    payload.update(DOMAIN);
    let mut active_count_histogram = vec![0usize; fixture.n_tokens];
    let mut tile_histogram = vec![0usize; 4];
    let mut total_active_experts = 0usize;
    let mut total_t32 = 0usize;
    let mut total_padding = 0usize;
    for (&expected_layer, layer) in GROUPED_LAYERS.iter().zip(&fixture.layers) {
        assert_eq!(layer.layer, expected_layer);
        assert_eq!(layer.counts.len(), MOE_EXPERT_COUNT);
        assert!(
            layer
                .counts
                .iter()
                .all(|&count| usize::from(count) <= fixture.n_tokens)
        );
        assert_eq!(
            layer
                .counts
                .iter()
                .map(|&count| usize::from(count))
                .sum::<usize>(),
            fixture.n_tokens * fixture.top_k
        );
        let active_experts = layer.counts.iter().filter(|&&count| count > 0).count();
        let t32 = layer
            .counts
            .iter()
            .map(|&count| usize::from(count).div_ceil(32))
            .sum::<usize>();
        let padding = 32 * t32 - fixture.n_tokens * fixture.top_k;
        assert_eq!(layer.active_experts, active_experts);
        assert_eq!(layer.t32, t32);
        assert_eq!(layer.padding, padding);
        total_active_experts += active_experts;
        total_t32 += t32;
        total_padding += padding;
        payload.update((layer.layer as u32).to_le_bytes());
        for &count in &layer.counts {
            payload.update(count.to_le_bytes());
            if count > 0 {
                active_count_histogram[usize::from(count) - 1] += 1;
                tile_histogram[usize::from(count).div_ceil(32) - 1] += 1;
            }
        }
    }
    assert_eq!(fixture.total_active_experts, total_active_experts);
    assert_eq!(fixture.inactive_experts, 25 * 256 - total_active_experts);
    assert_eq!(fixture.total_t32, total_t32);
    assert_eq!(fixture.total_padding, total_padding);
    assert_eq!(
        fixture.active_count_histogram_1_to_128,
        active_count_histogram
    );
    assert_eq!(fixture.tile_histogram_1_to_4, tile_histogram);
    assert_eq!(total_active_experts, 1_542);
    assert_eq!(total_t32, 1_748);
    assert_eq!(total_padding, 36_736);
    assert_eq!(fixture.tile_histogram_1_to_4, [1_395, 100, 35, 12]);
    assert_eq!(hex(payload.finalize()), fixture.count_payload_sha256);
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn packed_grouped_iq2_route_census_fixture_is_canonical() {
    const COUNT_DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-grouped-iq2-counts:v1\0";
    const ROUTE_DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-grouped-iq2-route-ids:v1\0";

    fn hex(digest: impl AsRef<[u8]>) -> String {
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn decode_hex_32(value: &str) -> [u8; 32] {
        assert_eq!(value.len(), 64);
        std::array::from_fn(|index| {
            u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap()
        })
    }

    let counts = packed_grouped_iq2_count_census_fixture();
    let routes = packed_grouped_iq2_route_census_fixture();
    assert_eq!(routes.schema_version, 1);
    assert_eq!(
        routes.route_payload_domain_hex,
        "7177656e2d6c6c6d3a647376343a7061636b65642d67726f757065642d6971322d726f7574652d6964733a763100"
    );
    assert_eq!(
        routes.route_payload_sha256,
        "505cb93ff9c3e1557bbad8f27a773e08b0e8c3475fbb4d7096069667c1fbafdd"
    );
    assert_eq!(routes.model_content_id, counts.model_content_id);
    assert_eq!(
        routes.prompt_token_ids_sha256,
        counts.prompt_token_ids_sha256
    );
    assert!(routes.prompt_token_ids.is_none());
    assert_eq!(routes.count_payload_sha256, counts.count_payload_sha256);
    assert_eq!(routes.n_tokens, counts.n_tokens);
    assert_eq!(routes.top_k, counts.top_k);
    assert_eq!(routes.expert_count, counts.expert_count);
    assert_eq!(routes.layer_count, counts.layers.len());
    assert_eq!(routes.route_count, routes.n_tokens * routes.top_k);
    assert_eq!(routes.grouped_layer_ids, counts.grouped_layer_ids);
    assert_eq!(routes.layers.len(), routes.layer_count);

    let mut count_payload = Sha256::new();
    count_payload.update(COUNT_DOMAIN);
    for layer in &counts.layers {
        count_payload.update((layer.layer as u32).to_le_bytes());
        for &count in &layer.counts {
            count_payload.update(count.to_le_bytes());
        }
    }
    let count_payload_digest = count_payload.finalize();
    assert_eq!(hex(count_payload_digest), routes.count_payload_sha256);

    let mut route_payload = Sha256::new();
    route_payload.update(ROUTE_DOMAIN);
    route_payload.update(decode_hex_32(&routes.model_content_id));
    route_payload.update(decode_hex_32(&routes.prompt_token_ids_sha256));
    route_payload.update(count_payload_digest);
    for value in [
        routes.n_tokens,
        routes.top_k,
        routes.expert_count,
        routes.layer_count,
        routes.route_count,
    ] {
        route_payload.update((value as u32).to_le_bytes());
    }
    for (count_layer, route_layer) in counts.layers.iter().zip(&routes.layers) {
        assert_eq!(route_layer.layer, count_layer.layer);
        assert_eq!(route_layer.route_expert_ids.len(), routes.route_count);
        let (expert_ids, rows, slots, schedule) =
            packed_grouped_schedule_from_route_ids(routes.n_tokens, &route_layer.route_expert_ids);
        assert_eq!(
            packed_post_route_expert_ids(
                routes.n_tokens,
                routes.expert_count,
                &expert_ids,
                &rows,
                &slots,
                &schedule,
            )
            .unwrap(),
            route_layer.route_expert_ids
        );
        assert_eq!(
            packed_post_route_expert_counts(routes.n_tokens, &schedule, routes.expert_count,)
                .unwrap()
                .as_slice(),
            count_layer.counts.as_slice()
        );
        assert_eq!(
            packed_grouped_expert_tiles(routes.n_tokens, &schedule)
                .unwrap()
                .len(),
            count_layer.t32
        );
        route_payload.update((route_layer.layer as u32).to_le_bytes());
        for &expert in &route_layer.route_expert_ids {
            route_payload.update(expert.to_le_bytes());
        }
    }
    assert_eq!(hex(route_payload.finalize()), routes.route_payload_sha256);
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn packed_grouped_iq2_representative_route_census_fixture_is_canonical() {
    const COUNT_DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-grouped-iq2-counts:v1\0";
    const ROUTE_DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-grouped-iq2-route-ids:v1\0";

    fn hex(digest: impl AsRef<[u8]>) -> String {
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn decode_hex_32(value: &str) -> [u8; 32] {
        assert_eq!(value.len(), 64);
        std::array::from_fn(|index| {
            u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap()
        })
    }

    let routes = packed_grouped_iq2_representative_route_census_fixture();
    assert_eq!(routes.schema_version, 1);
    assert_eq!(routes.route_payload_domain_hex, hex(ROUTE_DOMAIN));
    assert_eq!(
        routes.route_payload_sha256,
        "7454b2692359464c0d932e1c2fffe53d345e0ea969db9de67458ed992ddd539c"
    );
    assert_eq!(
        routes.model_content_id,
        "ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2"
    );
    assert_eq!(
        routes.prompt_token_ids_sha256,
        "ee09a95c18d0231d195a88cc4d96c34ae27df5b3fd2fecb0c5e909807e4be8da"
    );
    let prompt_token_ids = routes
        .prompt_token_ids
        .as_ref()
        .expect("representative route fixture retains exact prompt IDs");
    assert_eq!(prompt_token_ids.len(), routes.n_tokens);
    assert_eq!(
        hex(Sha256::digest(bytemuck::cast_slice(prompt_token_ids))),
        routes.prompt_token_ids_sha256
    );
    assert_eq!(
        routes.count_payload_sha256,
        "0486f37c39a37cab0cbb1cfe41d8d4fca401059b7abbe872901e841fdd4d394a"
    );
    assert_eq!(routes.n_tokens, 128);
    assert_eq!(routes.top_k, MOE_TOP_K);
    assert_eq!(routes.expert_count, MOE_EXPERT_COUNT);
    assert_eq!(routes.layer_count, 25);
    assert_eq!(routes.route_count, routes.n_tokens * routes.top_k);
    assert_eq!(routes.layers.len(), routes.layer_count);

    let mut count_payload = Sha256::new();
    count_payload.update(COUNT_DOMAIN);
    let mut route_payload = Sha256::new();
    route_payload.update(ROUTE_DOMAIN);
    route_payload.update(decode_hex_32(&routes.model_content_id));
    route_payload.update(decode_hex_32(&routes.prompt_token_ids_sha256));
    let mut total_active_experts = 0usize;
    let mut total_tiles_16 = 0usize;
    let mut total_padding_16 = 0usize;
    let mut counts_by_layer = Vec::with_capacity(routes.layer_count);
    for layer in &routes.layers {
        assert_eq!(layer.route_expert_ids.len(), routes.route_count);
        let (_, rows, slots, schedule) =
            packed_grouped_schedule_from_route_ids(routes.n_tokens, &layer.route_expert_ids);
        let counts =
            packed_post_route_expert_counts(routes.n_tokens, &schedule, routes.expert_count)
                .unwrap();
        assert_eq!(counts.len(), MOE_EXPERT_COUNT);
        let tiles_16 = packed_grouped_iq2_mma16_tiles(routes.n_tokens, &schedule)
            .unwrap()
            .len();
        total_active_experts += schedule.len();
        total_tiles_16 += tiles_16;
        total_padding_16 += tiles_16 * 16 - routes.route_count;
        count_payload.update((layer.layer as u32).to_le_bytes());
        for &count in &counts {
            count_payload.update(count.to_le_bytes());
        }
        assert_eq!(rows.len(), routes.route_count);
        assert_eq!(slots.len(), routes.route_count);
        counts_by_layer.push(counts);
    }
    let count_payload_digest = count_payload.finalize();
    assert_eq!(hex(count_payload_digest), routes.count_payload_sha256);
    route_payload.update(count_payload_digest);
    for value in [
        routes.n_tokens,
        routes.top_k,
        routes.expert_count,
        routes.layer_count,
        routes.route_count,
    ] {
        route_payload.update((value as u32).to_le_bytes());
    }
    for layer in &routes.layers {
        route_payload.update((layer.layer as u32).to_le_bytes());
        for &expert in &layer.route_expert_ids {
            route_payload.update(expert.to_le_bytes());
        }
    }
    assert_eq!(hex(route_payload.finalize()), routes.route_payload_sha256);
    assert_eq!(total_active_experts, 3_154);
    assert_eq!(total_tiles_16, 3_558);
    assert_eq!(total_padding_16, 37_728);
    assert_eq!(counts_by_layer.len(), routes.layer_count);
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn packed_all_iq3_route_census_fixture_is_canonical() {
    const COUNT_DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-all-iq3-counts:v1\0";
    const ROUTE_DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-all-iq3-route-ids:v1\0";
    const LAYERS: [usize; 16] = [1, 4, 5, 8, 9, 20, 24, 27, 28, 29, 30, 31, 32, 33, 34, 40];
    const BUCKETS: [usize; 16] = [
        23, 38, 46, 68, 73, 72, 71, 79, 75, 79, 84, 77, 65, 62, 59, 65,
    ];

    fn hex(digest: impl AsRef<[u8]>) -> String {
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn decode_hex_32(value: &str) -> [u8; 32] {
        assert_eq!(value.len(), 64);
        std::array::from_fn(|index| {
            u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap()
        })
    }

    let fixture = packed_all_iq3_route_census_fixture();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.count_payload_domain_hex, hex(COUNT_DOMAIN));
    assert_eq!(fixture.route_payload_domain_hex, hex(ROUTE_DOMAIN));
    assert_eq!(
        fixture.count_payload_sha256,
        "72b5d2dba179d5d65334ec413bd82df784b9001d545b8c6e7b74aa29f309ff7c"
    );
    assert_eq!(
        fixture.route_payload_sha256,
        "ee106712f42aed80cc559414140327537dd9f256110b6acffee6a1b893319582"
    );
    assert_eq!(
        fixture.model_content_id,
        "ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2"
    );
    assert_eq!(
        fixture.prompt_token_ids_sha256,
        "b57816bcb0d5fdf5a8e2ddc7a0afe9e57fb0ca6ffc2b849285e1635d04772843"
    );
    assert_eq!(fixture.n_tokens, 128);
    assert_eq!(fixture.top_k, MOE_TOP_K);
    assert_eq!(fixture.expert_count, MOE_EXPERT_COUNT);
    assert_eq!(fixture.layer_count, LAYERS.len());
    assert_eq!(fixture.route_count, fixture.n_tokens * fixture.top_k);
    assert_eq!(fixture.all_iq3_layer_ids, LAYERS);
    assert_eq!(fixture.layers.len(), fixture.layer_count);

    let mut count_payload = Sha256::new();
    count_payload.update(COUNT_DOMAIN);
    count_payload.update(decode_hex_32(&fixture.model_content_id));
    count_payload.update(decode_hex_32(&fixture.prompt_token_ids_sha256));
    for value in [
        fixture.n_tokens,
        fixture.top_k,
        fixture.expert_count,
        fixture.layer_count,
        fixture.route_count,
    ] {
        count_payload.update((value as u32).to_le_bytes());
    }

    let mut route_payload = Sha256::new();
    route_payload.update(ROUTE_DOMAIN);
    route_payload.update(decode_hex_32(&fixture.model_content_id));
    route_payload.update(decode_hex_32(&fixture.prompt_token_ids_sha256));
    route_payload.update(decode_hex_32(&fixture.count_payload_sha256));
    for value in [
        fixture.n_tokens,
        fixture.top_k,
        fixture.expert_count,
        fixture.layer_count,
        fixture.route_count,
    ] {
        route_payload.update((value as u32).to_le_bytes());
    }

    let mut total_active_experts = 0usize;
    let mut total_t32 = 0usize;
    let mut total_padding = 0usize;
    for (index, layer) in fixture.layers.iter().enumerate() {
        assert_eq!(layer.layer, LAYERS[index]);
        assert_eq!(layer.active_experts, BUCKETS[index]);
        assert_eq!(layer.expert_counts.len(), MOE_EXPERT_COUNT);
        assert_eq!(layer.route_expert_ids.len(), fixture.route_count);
        assert_eq!(
            layer
                .expert_counts
                .iter()
                .map(|&count| usize::from(count))
                .sum::<usize>(),
            fixture.route_count
        );
        assert_eq!(
            layer
                .expert_counts
                .iter()
                .filter(|&&count| count > 0)
                .count(),
            layer.active_experts
        );
        let t32 = layer
            .expert_counts
            .iter()
            .map(|&count| usize::from(count).div_ceil(32))
            .sum::<usize>();
        assert_eq!(layer.t32, t32);
        assert_eq!(layer.padding, t32 * 32 - fixture.route_count);

        let (expert_ids, rows, slots, schedule) =
            packed_grouped_schedule_from_route_ids(fixture.n_tokens, &layer.route_expert_ids);
        assert_eq!(schedule.len(), layer.active_experts);
        assert_eq!(
            packed_post_route_expert_ids(
                fixture.n_tokens,
                fixture.expert_count,
                &expert_ids,
                &rows,
                &slots,
                &schedule,
            )
            .unwrap(),
            layer.route_expert_ids
        );
        assert_eq!(
            packed_post_route_expert_counts(fixture.n_tokens, &schedule, fixture.expert_count,)
                .unwrap()
                .as_slice(),
            layer.expert_counts.as_slice()
        );
        assert_eq!(
            packed_grouped_expert_tiles(fixture.n_tokens, &schedule)
                .unwrap()
                .len(),
            layer.t32
        );

        count_payload.update((layer.layer as u32).to_le_bytes());
        for &count in &layer.expert_counts {
            count_payload.update(count.to_le_bytes());
        }
        route_payload.update((layer.layer as u32).to_le_bytes());
        for &expert in &layer.route_expert_ids {
            route_payload.update(expert.to_le_bytes());
        }
        total_active_experts += layer.active_experts;
        total_t32 += layer.t32;
        total_padding += layer.padding;
    }
    assert_eq!(total_active_experts, 1_036);
    assert_eq!(total_t32, 1_187);
    assert_eq!(total_padding, 25_696);
    assert_eq!(fixture.total_active_experts, total_active_experts);
    assert_eq!(fixture.total_t32, total_t32);
    assert_eq!(fixture.total_padding, total_padding);
    assert_eq!(hex(count_payload.finalize()), fixture.count_payload_sha256);
    assert_eq!(hex(route_payload.finalize()), fixture.route_payload_sha256);
}

#[test]
fn packed_grouped_mapped_iq3_consumes_explicit_source_rows() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const H: usize = 256;
    const O: usize = 64;
    const N: usize = 2;
    const E: usize = MOE_EXPERT_COUNT;
    const K: usize = MOE_TOP_K;

    let bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, O, E, 23);
    let (_expert_ids, rows, slots, schedule) = grouped_test_schedule(N);
    let mapped_rows = rows
        .iter()
        .map(|&row| i32::try_from(N - 1).unwrap() - row)
        .collect::<Vec<_>>();
    assert!(
        mapped_rows
            .iter()
            .zip(&slots)
            .any(|(&row, &slot)| row as usize != slot as usize / K)
    );
    let input_values = (0..N * H)
        .map(|index| ((index * 41 + 7) % 257) as f32 * 0.002 - 0.25)
        .collect::<Vec<_>>();
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&input_values),
        vec![H as u64, N as u64],
        GgmlType::F32,
    )
    .unwrap();
    let source_rows = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&mapped_rows),
        vec![(N * K) as u64],
        GgmlType::I32,
    )
    .unwrap();
    let destination_slots = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&slots),
        vec![(N * K) as u64],
        GgmlType::I32,
    )
    .unwrap();
    let gathered = MetalTensor::zeros_f32(&ctx, vec![H as u64, N as u64]).unwrap();
    let projected = MetalTensor::zeros_f32(&ctx, vec![O as u64, N as u64]).unwrap();
    let control = grouped_guarded_f32(&ctx, vec![O as u64, (N * K) as u64], 7.0);
    let candidate = grouped_guarded_f32(&ctx, vec![O as u64, (N * K) as u64], 11.0);

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    for bucket in &schedule {
        let row_view = i32_slice(
            &source_rows,
            bucket.start,
            bucket.len,
            "mapped IQ3 control rows",
        )
        .unwrap();
        let slot_view = i32_slice(
            &destination_slots,
            bucket.start,
            bucket.len,
            "mapped IQ3 control slots",
        )
        .unwrap();
        let input_view = f32_prefix(
            &gathered,
            vec![H as u64, bucket.len as u64],
            "mapped IQ3 control input",
        )
        .unwrap();
        let output_view = f32_prefix(
            &projected,
            vec![O as u64, bucket.len as u64],
            "mapped IQ3 control output",
        )
        .unwrap();
        encode_get_rows_f32(
            &ctx,
            &encoder,
            &input,
            &row_view,
            &input_view,
            bucket.len,
            H,
        )
        .unwrap();
        let weight =
            expert_weight_view(&bank, H, O, bucket.expert, "mapped IQ3 control weight").unwrap();
        encode_batch_projection(
            &ctx,
            &encoder,
            &weight,
            &input_view,
            &output_view,
            H,
            O,
            bucket.len,
            "mapped IQ3 control projection",
        )
        .unwrap();
        crate::metal::encode_scatter_rows_f32_unique(
            &ctx,
            &encoder,
            &output_view,
            &slot_view,
            &control,
            O,
            bucket.len,
        )
        .unwrap();
    }
    encode_packed_grouped_mapped_iq3_xxs_f32(
        &ctx,
        &encoder,
        &bank,
        &input,
        &source_rows,
        &destination_slots,
        &schedule,
        &candidate,
        H,
        O,
        E,
        K,
        N,
        N,
        N * K,
    )
    .unwrap();
    encoder.end();
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert!(command.error().is_none(), "{:?}", command.error());

    assert_eq!(
        host_read_f32(&candidate, "mapped IQ3 candidate").unwrap(),
        host_read_f32(&control, "mapped IQ3 control").unwrap()
    );
    assert_grouped_guards("mapped IQ3 control", &control);
    assert_grouped_guards("mapped IQ3 candidate", &candidate);
}

#[test]
fn packed_grouped_mapped_q3_q4_match_static_expert_views() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const H: usize = 512;
    const O: usize = 128;
    const N: usize = 33;
    const E: usize = 160;
    const K: usize = MOE_TOP_K;
    const ROUTED: [usize; K] = [0, 1, 2, 3, E - 1, E - 1];

    let expert_ids = (0..N)
        .flat_map(|_| ROUTED.map(|expert| expert as i32))
        .collect::<Vec<_>>();
    let mut by_expert = (0..E)
        .map(|_| Vec::<(usize, usize)>::new())
        .collect::<Vec<_>>();
    for token in 0..N {
        for (slot, &expert) in ROUTED.iter().enumerate() {
            by_expert[expert].push((token, token * K + slot));
        }
    }
    let mut rows = Vec::with_capacity(N * K);
    let mut slots = Vec::with_capacity(N * K);
    let mut schedule = Vec::with_capacity(K);
    for (expert, assignments) in by_expert.into_iter().enumerate() {
        if assignments.is_empty() {
            continue;
        }
        let start = rows.len();
        for (token, slot) in assignments {
            rows.push(token as i32);
            slots.push(slot as i32);
        }
        schedule.push(ExpertBucket {
            expert,
            start,
            len: rows.len() - start,
        });
    }
    validate_packed_expert_schedule(N, E, &expert_ids, &rows, &slots, &schedule).unwrap();
    let mapped_rows = rows
        .iter()
        .map(|&row| i32::try_from(N - 1).unwrap() - row)
        .collect::<Vec<_>>();
    let input_values = (0..N * H)
        .map(|index| ((index * 41 + 7) % 257) as f32 * 0.002 - 0.25)
        .collect::<Vec<_>>();
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&input_values),
        vec![H as u64, N as u64],
        GgmlType::F32,
    )
    .unwrap();
    let source_rows = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&mapped_rows),
        vec![(N * K) as u64],
        GgmlType::I32,
    )
    .unwrap();
    let destination_slots = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&slots),
        vec![(N * K) as u64],
        GgmlType::I32,
    )
    .unwrap();

    for (dtype, seed) in [(GgmlType::Q3_K, 23usize), (GgmlType::Q4_K, 29)] {
        let bank = grouped_test_bank(&ctx, dtype, H, O, E, seed);
        let gathered = MetalTensor::zeros_f32(&ctx, vec![H as u64, (N * K) as u64]).unwrap();
        let projected = MetalTensor::zeros_f32(&ctx, vec![O as u64, (N * K) as u64]).unwrap();
        let control = grouped_guarded_f32(&ctx, vec![O as u64, (N * K) as u64], 7.0);
        let candidate = grouped_guarded_f32(&ctx, vec![O as u64, (N * K) as u64], 11.0);
        let plan = PackedGroupedExpertPlan::new(N, &schedule, None).unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for bucket in &schedule {
            let gathered_view = f32_prefix(
                &gathered,
                vec![H as u64, bucket.len as u64],
                "mapped K-block control gathered",
            )
            .unwrap();
            let projected_view = f32_prefix(
                &projected,
                vec![O as u64, bucket.len as u64],
                "mapped K-block control projected",
            )
            .unwrap();
            let row_view = i32_slice(
                &source_rows,
                bucket.start,
                bucket.len,
                "mapped K-block control rows",
            )
            .unwrap();
            let slot_view = i32_slice(
                &destination_slots,
                bucket.start,
                bucket.len,
                "mapped K-block control slots",
            )
            .unwrap();
            encode_get_rows_f32(
                &ctx,
                &encoder,
                &input,
                &row_view,
                &gathered_view,
                bucket.len,
                H,
            )
            .unwrap();
            let weight =
                expert_weight_view(&bank, H, O, bucket.expert, "mapped K-block control weight")
                    .unwrap();
            encode_batch_projection(
                &ctx,
                &encoder,
                &weight,
                &gathered_view,
                &projected_view,
                H,
                O,
                bucket.len,
                "mapped K-block control projection",
            )
            .unwrap();
            crate::metal::encode_scatter_rows_f32_unique(
                &ctx,
                &encoder,
                &projected_view,
                &slot_view,
                &control,
                O,
                bucket.len,
            )
            .unwrap();
        }
        encode_packed_grouped_mapped_k_block_f32_plan(
            &ctx,
            &encoder,
            &bank,
            &input,
            &source_rows,
            &destination_slots,
            &plan,
            &candidate,
            H,
            O,
            E,
            K,
            N,
            N,
            N * K,
        )
        .unwrap();
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(
            command.error().is_none(),
            "{dtype:?}: {:?}",
            command.error()
        );

        assert_eq!(
            host_read_f32(&candidate, "mapped K-block candidate").unwrap(),
            host_read_f32(&control, "mapped K-block control").unwrap(),
            "{dtype:?} mapped projection changed dense matrix lineage"
        );
        assert_grouped_guards("mapped K-block control", &control);
        assert_grouped_guards("mapped K-block candidate", &candidate);
    }
}

#[test]
fn q8_f32_mma_r2c4k64_reduces_operand_rounding_and_preserves_guards() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };

    fn submit(
        ctx: &MetalContext,
        encode: impl FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    ) {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let result = encode(&encoder);
        encoder.end();
        result.unwrap();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(command.error().is_none(), "{:?}", command.error());
    }

    for (n_in, n_out) in [(64usize, 16usize), (128, 32)] {
        let weight = q8_precision_test_weight(&ctx, n_in, n_out);
        for n_tokens in [1usize, 31, 32, 33, 128, 337] {
            let padded_tokens = n_tokens.div_ceil(32) * 32;
            let input_storage = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q8_precision_test_input(padded_tokens * n_in)),
                vec![n_in as u64, padded_tokens as u64],
                GgmlType::F32,
            )
            .unwrap();
            let input = input_storage.view_subrange(0, vec![n_in as u64, n_tokens as u64]);
            let exact = grouped_guarded_f32(&ctx, vec![n_out as u64, n_tokens as u64], 3.0);
            let half = grouped_guarded_f32(&ctx, vec![n_out as u64, n_tokens as u64], 5.0);
            let candidate = grouped_guarded_f32(&ctx, vec![n_out as u64, n_tokens as u64], 7.0);
            let repeat = grouped_guarded_f32(&ctx, vec![n_out as u64, n_tokens as u64], 11.0);
            submit(&ctx, |encoder| {
                crate::metal::encode_mat_vec_q8_0_batch_f32(
                    &ctx, encoder, &weight, &input, &exact, n_in, n_out, n_tokens,
                )
                .map_err(DeepSeekV4MetalError::Metal)
            });
            submit(&ctx, |encoder| {
                crate::metal::encode_mat_mat_q8_0_f32(
                    &ctx, encoder, &weight, &input, &half, n_in, n_out, n_tokens,
                )
                .map_err(DeepSeekV4MetalError::Metal)
            });
            for output in [&candidate, &repeat] {
                submit(&ctx, |encoder| {
                    encode_q8_f32_mma_r2c4k64(
                        &ctx, encoder, &weight, &input, output, n_in, n_out, n_tokens,
                    )
                });
            }

            let exact_values = host_read_f32(&exact, "Q8 F32 exact").unwrap();
            let half_values = host_read_f32(&half, "Q8 F32 half").unwrap();
            let candidate_values = host_read_f32(&candidate, "Q8 F32 candidate").unwrap();
            let repeat_values = host_read_f32(&repeat, "Q8 F32 repeat").unwrap();
            let half_diff = q8_differential(&half_values, &exact_values);
            let candidate_diff = q8_differential(&candidate_values, &exact_values);
            assert!(
                1.0 - candidate_diff.0 <= 1.0 - half_diff.0 + 1e-15
                    && candidate_diff.1 < half_diff.1
                    && candidate_diff.2 < half_diff.2,
                "K={n_in} M={n_out} N={n_tokens} half={half_diff:?} candidate={candidate_diff:?}"
            );
            assert_eq!(
                candidate_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                repeat_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "K={n_in} M={n_out} N={n_tokens} repeat"
            );
            for (label, tensor) in [
                ("Q8 F32 exact", &exact),
                ("Q8 F32 half", &half),
                ("Q8 F32 candidate", &candidate),
                ("Q8 F32 repeat", &repeat),
            ] {
                assert_grouped_guards(label, tensor);
            }
        }
    }
}

#[test]
fn q8_f32_mma_r2c16k64_matches_r2c4k64_bits() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const M: usize = 32;
    const N: usize = 256;

    fn submit(
        ctx: &MetalContext,
        encode: impl FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    ) {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let result = encode(&encoder);
        encoder.end();
        result.unwrap();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(command.error().is_none(), "{:?}", command.error());
    }

    for k in [128usize, 2_048, 4_096] {
        let weight = q8_precision_test_weight(&ctx, k, M);
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&q8_precision_test_input(k * N)),
            vec![k as u64, N as u64],
            GgmlType::F32,
        )
        .unwrap();
        let control = grouped_guarded_f32(&ctx, vec![M as u64, N as u64], 3.0);
        let candidate = grouped_guarded_f32(&ctx, vec![M as u64, N as u64], 5.0);
        let repeat = grouped_guarded_f32(&ctx, vec![M as u64, N as u64], 7.0);

        submit(&ctx, |encoder| {
            encode_q8_f32_mma_r2c4k64(&ctx, encoder, &weight, &input, &control, k, M, N)
        });
        for output in [&candidate, &repeat] {
            submit(&ctx, |encoder| {
                encode_q8_f32_mma_r2c16k64(&ctx, encoder, &weight, &input, output, k, M, N)
            });
        }
        let bits = |tensor: &MetalTensor, label| {
            host_read_f32(tensor, label)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        let control_bits = bits(&control, "Q8 R2C4K64 control");
        let candidate_bits = bits(&candidate, "Q8 R2C16K64 candidate");
        let repeat_bits = bits(&repeat, "Q8 R2C16K64 repeat");
        let first_mismatch = candidate_bits
            .iter()
            .zip(&control_bits)
            .position(|(candidate, control)| candidate != control);
        assert!(
            first_mismatch.is_none(),
            "K={k} first mismatch={first_mismatch:?} candidate={:?} control={:?}",
            &candidate_bits[..64],
            &control_bits[..64],
        );
        assert_eq!(repeat_bits, candidate_bits, "K={k} repeat");
        assert_grouped_guards("Q8 R2C4K64 control", &control);
        assert_grouped_guards("Q8 R2C16K64 candidate", &candidate);
        assert_grouped_guards("Q8 R2C16K64 repeat", &repeat);
    }
}

#[test]
fn q8_f32_grouped_r2c16k64_matches_per_group_controls_bits() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const GROUPS: usize = 2;
    const M: usize = 32;
    const N: usize = 128;

    fn submit(
        ctx: &MetalContext,
        encode: impl FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    ) {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let result = encode(&encoder);
        encoder.end();
        result.unwrap();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(command.error().is_none(), "{:?}", command.error());
    }

    for k in [128usize, 4_096] {
        let weight = q8_precision_test_weight(&ctx, k, M * GROUPS);
        let input_values = q8_precision_test_input(k * GROUPS * N);
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input_values),
            vec![(k * GROUPS) as u64, N as u64],
            GgmlType::F32,
        )
        .unwrap();
        let controls = (0..GROUPS)
            .map(|group| {
                let mut values = Vec::with_capacity(k * N);
                for token in 0..N {
                    let start = token * k * GROUPS + group * k;
                    values.extend_from_slice(&input_values[start..start + k]);
                }
                let group_input = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&values),
                    vec![k as u64, N as u64],
                    GgmlType::F32,
                )
                .unwrap();
                let group_weight = group_weight_view(&weight, k, M, group).unwrap();
                let control =
                    grouped_guarded_f32(&ctx, vec![M as u64, N as u64], 3.0 + group as f32);
                submit(&ctx, |encoder| {
                    encode_q8_f32_mma_r2c16k64(
                        &ctx,
                        encoder,
                        &group_weight,
                        &group_input,
                        &control,
                        k,
                        M,
                        N,
                    )
                });
                control
            })
            .collect::<Vec<_>>();
        let candidate = grouped_guarded_f32(&ctx, vec![(M * GROUPS) as u64, N as u64], 7.0);
        let repeat = grouped_guarded_f32(&ctx, vec![(M * GROUPS) as u64, N as u64], 11.0);
        for output in [&candidate, &repeat] {
            submit(&ctx, |encoder| {
                encode_q8_f32_mma_r2c16k64_grouped(
                    &ctx, encoder, &weight, &input, output, k, M, GROUPS, N,
                )
            });
        }

        let mut expected = vec![0.0f32; M * GROUPS * N];
        for (group, control) in controls.iter().enumerate() {
            let values = host_read_f32(control, "grouped Q8 control").unwrap();
            for token in 0..N {
                let source = token * M;
                let destination = token * M * GROUPS + group * M;
                expected[destination..destination + M].copy_from_slice(&values[source..source + M]);
            }
            assert_grouped_guards("grouped Q8 control", control);
        }
        let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let expected_bits = bits(expected);
        let candidate_bits = bits(host_read_f32(&candidate, "grouped Q8 candidate").unwrap());
        let repeat_bits = bits(host_read_f32(&repeat, "grouped Q8 repeat").unwrap());
        assert_eq!(candidate_bits, expected_bits, "K={k} candidate");
        assert_eq!(repeat_bits, candidate_bits, "K={k} repeat");
        assert_grouped_guards("grouped Q8 candidate", &candidate);
        assert_grouped_guards("grouped Q8 repeat", &repeat);
    }
}

#[test]
fn q8_f32_mma_r2c4k64_rejects_unqualified_storage_before_dispatch() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const K: usize = 64;
    const M: usize = 16;
    const N: usize = 33;

    fn reject(
        ctx: &MetalContext,
        encode: impl FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    ) -> String {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let result = encode(&encoder);
        encoder.end();
        result.unwrap_err().to_string()
    }

    let weight = q8_precision_test_weight(&ctx, K, M);
    let input = MetalTensor::zeros_f32(&ctx, vec![K as u64, N as u64]).unwrap();
    let output = MetalTensor::zeros_f32(&ctx, vec![M as u64, N as u64]).unwrap();
    let error = reject(&ctx, |encoder| {
        encode_q8_f32_mma_r2c4k64(&ctx, encoder, &weight, &input, &output, K, M, N)
    });
    assert!(error.contains("padded input backing"), "{error}");

    let padded = MetalTensor::zeros_f32(&ctx, vec![K as u64, 64]).unwrap();
    let padded_input = padded.view_subrange(0, vec![K as u64, N as u64]);
    let overlapping_output = padded.view_subrange(0, vec![M as u64, N as u64]);
    let error = reject(&ctx, |encoder| {
        encode_q8_f32_mma_r2c4k64(
            &ctx,
            encoder,
            &weight,
            &padded_input,
            &overlapping_output,
            K,
            M,
            N,
        )
    });
    assert!(error.contains("overlaps an input"), "{error}");

    let padding_output = padded.view_subrange((K * N) as u64, vec![M as u64, N as u64]);
    let error = reject(&ctx, |encoder| {
        encode_q8_f32_mma_r2c4k64(
            &ctx,
            encoder,
            &weight,
            &padded_input,
            &padding_output,
            K,
            M,
            N,
        )
    });
    assert!(error.contains("overlaps an input"), "{error}");

    let malformed_input = padded.view_subrange(0, vec![(K * N) as u64]);
    let error = reject(&ctx, |encoder| {
        encode_q8_f32_mma_r2c4k64(&ctx, encoder, &weight, &malformed_input, &output, K, M, N)
    });
    assert!(error.contains("input"), "{error}");

    let wrong_weight = MetalTensor::zeros_f32(&ctx, vec![K as u64, M as u64]).unwrap();
    let error = reject(&ctx, |encoder| {
        encode_q8_f32_mma_r2c4k64(
            &ctx,
            encoder,
            &wrong_weight,
            &padded_input,
            &output,
            K,
            M,
            N,
        )
    });
    assert!(error.contains("invalid geometry or storage"), "{error}");
}

#[test]
#[ignore = "sealed GO; do not rerun without material implementation or device drift"]
fn profile_q8_f32_mma_r2c4k64_attention_output_packet() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    assert_eq!(
        ctx.device.name().to_string(),
        PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE
    );
    const N: usize = 128;
    const SAMPLES: usize = 24;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Variant {
        Exact,
        HalfA,
        HalfB,
        HalfBoth,
        F32A,
        F32B,
        F32Both,
    }

    fn projections(variant: Variant) -> (Q8PrecisionProjection, Q8PrecisionProjection) {
        match variant {
            Variant::Exact => (Q8PrecisionProjection::Exact, Q8PrecisionProjection::Exact),
            Variant::HalfA => (
                Q8PrecisionProjection::HalfMatrix,
                Q8PrecisionProjection::Exact,
            ),
            Variant::HalfB => (
                Q8PrecisionProjection::Exact,
                Q8PrecisionProjection::HalfMatrix,
            ),
            Variant::HalfBoth => (
                Q8PrecisionProjection::HalfMatrix,
                Q8PrecisionProjection::HalfMatrix,
            ),
            Variant::F32A => (
                Q8PrecisionProjection::F32Matrix,
                Q8PrecisionProjection::Exact,
            ),
            Variant::F32B => (
                Q8PrecisionProjection::Exact,
                Q8PrecisionProjection::F32Matrix,
            ),
            Variant::F32Both => (
                Q8PrecisionProjection::F32Matrix,
                Q8PrecisionProjection::F32Matrix,
            ),
        }
    }

    let output_a = q8_output_test_weight(&ctx, GROUP_WIDTH, LOW_RANK_WIDTH, 3);
    let output_b = q8_output_test_weight(&ctx, LOW_RANK_WIDTH, DEEPSEEK_V4_HIDDEN_SIZE, 11);
    let attention_values = (0..QUERY_WIDTH * N)
        .map(|index| ((index * 31 + index / 11 + 5) % 257) as f32 * 0.004 - 0.51)
        .collect::<Vec<_>>();
    let attention = grouped_guarded_f32(&ctx, vec![QUERY_WIDTH as u64, N as u64], 13.0);
    write_raw_f32(&attention, &attention_values);
    let mut scratch = DeepSeekV4PrefillScratch::new(&ctx, DEEPSEEK_V4_CSA_TOP_K, MOE_EXPERT_COUNT)
        .unwrap()
        .attention;
    scratch.low_rank = grouped_guarded_f32(&ctx, vec![LOW_RANK_WIDTH as u64, N as u64], 17.0);
    scratch.output =
        grouped_guarded_f32(&ctx, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, N as u64], 19.0);
    scratch.group_input = grouped_guarded_f32(&ctx, vec![GROUP_WIDTH as u64, N as u64], 23.0);
    scratch.group_output = grouped_guarded_f32(&ctx, vec![1_024, N as u64], 29.0);
    let low_rank = f32_prefix(
        &scratch.low_rank,
        vec![LOW_RANK_WIDTH as u64, N as u64],
        "Q8 F32 packet low rank",
    )
    .unwrap();
    let output = f32_prefix(
        &scratch.output,
        vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, N as u64],
        "Q8 F32 packet output",
    )
    .unwrap();

    let execute = |variant: Variant| -> (f64, f64) {
        let started = std::time::Instant::now();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let result = if variant == Variant::Exact {
            scratch.encode_output(&ctx, &encoder, &attention, &output_a, &output_b, N)
        } else {
            let (output_a_projection, output_b_projection) = projections(variant);
            scratch.encode_output_q8_precision(
                &ctx,
                &encoder,
                &attention,
                &output_a,
                &output_b,
                N,
                output_a_projection,
                output_b_projection,
            )
        };
        encoder.end();
        result.unwrap();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        assert!(
            command.error().is_none(),
            "{variant:?}: {:?}",
            command.error()
        );
        (
            (command.GPUEndTime() - command.GPUStartTime()) * 1e3,
            wall_ms,
        )
    };
    let capture = || {
        (
            host_read_f32(&low_rank, "Q8 F32 packet low rank").unwrap(),
            host_read_f32(&output, "Q8 F32 packet output").unwrap(),
        )
    };
    let bits = |values: &[f32]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };

    execute(Variant::Exact);
    let (reference_low, reference_output) = capture();
    let evaluate = |variant| {
        execute(variant);
        let (low, output) = capture();
        let low_diff = q8_differential(&low, &reference_low);
        let output_diff = q8_differential(&output, &reference_output);
        (low, output, low_diff, output_diff)
    };
    let half_a = evaluate(Variant::HalfA);
    let half_b = evaluate(Variant::HalfB);
    let half_both = evaluate(Variant::HalfBoth);
    let f32_a = evaluate(Variant::F32A);
    let f32_b = evaluate(Variant::F32B);
    let f32_both = evaluate(Variant::F32Both);

    for (variant, expected_low, expected_output) in [
        (Variant::F32A, &f32_a.0, &f32_a.1),
        (Variant::F32B, &f32_b.0, &f32_b.1),
        (Variant::F32Both, &f32_both.0, &f32_both.1),
    ] {
        execute(variant);
        let (actual_low, actual_output) = capture();
        assert_eq!(
            bits(&actual_low),
            bits(expected_low),
            "{variant:?} low repeat"
        );
        assert_eq!(
            bits(&actual_output),
            bits(expected_output),
            "{variant:?} output repeat"
        );
    }

    eprintln!(
        "deepseek_v4 q8_f32_precision half_a_low={:?} half_a_output={:?} half_b_output={:?} half_both_low={:?} half_both_output={:?} f32_a_low={:?} f32_a_output={:?} f32_b_output={:?} f32_both_low={:?} f32_both_output={:?}",
        half_a.2,
        half_a.3,
        half_b.3,
        half_both.2,
        half_both.3,
        f32_a.2,
        f32_a.3,
        f32_b.3,
        f32_both.2,
        f32_both.3,
    );

    for index in 0usize..5 {
        if index.is_multiple_of(2) {
            execute(Variant::Exact);
            execute(Variant::F32Both);
        } else {
            execute(Variant::F32Both);
            execute(Variant::Exact);
        }
    }
    let collect = |variant| (0..SAMPLES).map(|_| execute(variant)).collect::<Vec<_>>();
    let control_before = collect(Variant::Exact);
    let candidate = collect(Variant::F32Both);
    let control_after = collect(Variant::Exact);
    let split = |samples: &[(f64, f64)]| {
        (
            samples.iter().map(|sample| sample.0).collect::<Vec<_>>(),
            samples.iter().map(|sample| sample.1).collect::<Vec<_>>(),
        )
    };
    let (control_before_gpu, control_before_wall) = split(&control_before);
    let (candidate_gpu, candidate_wall) = split(&candidate);
    let (control_after_gpu, control_after_wall) = split(&control_after);
    let gpu_control_drift = relative_drift(
        median_ms(&control_before_gpu),
        median_ms(&control_after_gpu),
    );
    let wall_control_drift = relative_drift(
        median_ms(&control_before_wall),
        median_ms(&control_after_wall),
    );
    let gpu_candidate_drift = relative_drift(
        median_ms(&candidate_gpu[..SAMPLES / 2]),
        median_ms(&candidate_gpu[SAMPLES / 2..]),
    );
    let wall_candidate_drift = relative_drift(
        median_ms(&candidate_wall[..SAMPLES / 2]),
        median_ms(&candidate_wall[SAMPLES / 2..]),
    );
    let gpu_control_median = median_ms(&control_before_gpu).min(median_ms(&control_after_gpu));
    let wall_control_median = median_ms(&control_before_wall).min(median_ms(&control_after_wall));
    let gpu_control_p95 =
        percentile_ms(&control_before_gpu, 0.95).min(percentile_ms(&control_after_gpu, 0.95));
    let wall_control_p95 =
        percentile_ms(&control_before_wall, 0.95).min(percentile_ms(&control_after_wall, 0.95));
    let candidate_gpu_median = median_ms(&candidate_gpu);
    let gpu_median_saving = 1.0 - candidate_gpu_median / gpu_control_median;
    let wall_median_saving = 1.0 - median_ms(&candidate_wall) / wall_control_median;
    let gpu_p95_saving = 1.0 - percentile_ms(&candidate_gpu, 0.95) / gpu_control_p95;
    let wall_p95_saving = 1.0 - percentile_ms(&candidate_wall, 0.95) / wall_control_p95;

    execute(Variant::Exact);
    let exact_after = capture();
    execute(Variant::F32Both);
    let candidate_after = capture();
    assert_eq!(
        bits(&exact_after.0),
        bits(&reference_low),
        "post-timing exact low"
    );
    assert_eq!(
        bits(&exact_after.1),
        bits(&reference_output),
        "post-timing exact output"
    );
    assert_eq!(
        bits(&candidate_after.0),
        bits(&f32_both.0),
        "post-timing candidate low"
    );
    assert_eq!(
        bits(&candidate_after.1),
        bits(&f32_both.1),
        "post-timing candidate output"
    );

    let _trace = crate::metal::kernel_trace_begin();
    execute(Variant::Exact);
    let control_trace = crate::metal::kernel_trace_take_delta();
    execute(Variant::F32Both);
    let candidate_trace = crate::metal::kernel_trace_take_delta();
    assert_eq!(control_trace.encoders, 1);
    assert_eq!(control_trace.concurrent_encoders, 0);
    assert_eq!(control_trace.dispatches, 25);
    assert_eq!(candidate_trace.encoders, 1);
    assert_eq!(candidate_trace.concurrent_encoders, 0);
    assert_eq!(candidate_trace.dispatches, 25);

    eprintln!(
        "deepseek_v4 q8_f32_packet control_before_gpu_ms={control_before_gpu:?} control_before_wall_ms={control_before_wall:?} candidate_gpu_ms={candidate_gpu:?} candidate_wall_ms={candidate_wall:?} control_after_gpu_ms={control_after_gpu:?} control_after_wall_ms={control_after_wall:?} gpu_control_drift={gpu_control_drift:.6} wall_control_drift={wall_control_drift:.6} gpu_candidate_drift={gpu_candidate_drift:.6} wall_candidate_drift={wall_candidate_drift:.6} candidate_gpu_median_ms={candidate_gpu_median:.6} gpu_median_saving={gpu_median_saving:.6} wall_median_saving={wall_median_saving:.6} gpu_p95_saving={gpu_p95_saving:.6} wall_p95_saving={wall_p95_saving:.6}"
    );

    for (label, tensor) in [
        ("Q8 F32 packet attention", &attention),
        ("Q8 F32 packet low rank", &scratch.low_rank),
        ("Q8 F32 packet output", &scratch.output),
        ("Q8 F32 packet group input", &scratch.group_input),
        ("Q8 F32 packet group output", &scratch.group_output),
    ] {
        assert_grouped_guards(label, tensor);
    }

    for (label, half, candidate) in [
        ("A-only output", half_a.3, f32_a.3),
        ("B-only output", half_b.3, f32_b.3),
        ("A+B output", half_both.3, f32_both.3),
        ("A-only low rank", half_a.2, f32_a.2),
        ("A+B low rank", half_both.2, f32_both.2),
    ] {
        let half_deficit = (1.0 - half.0).max(0.0);
        let candidate_deficit = (1.0 - candidate.0).max(0.0);
        assert!(
            candidate_deficit <= half_deficit * 0.25 + 1e-15
                && candidate.1 <= half.1 * 0.25
                && candidate.2 <= half.2 * 0.25,
            "{label} did not improve fourfold: half={half:?} candidate={candidate:?}"
        );
    }
    assert!(
        f32_a.2.1 <= 0.00020 && f32_a.2.2 <= 0.001,
        "F32 A low-rank gate failed: {:?}",
        f32_a.2
    );
    assert!(
        f32_both.3.0 >= 0.999_999_8 && f32_both.3.1 <= 0.00030 && f32_both.3.2 <= 0.008,
        "F32 A+B output gate failed: {:?}",
        f32_both.3
    );
    assert!(
        gpu_control_drift <= 0.05,
        "GPU control drift {gpu_control_drift}"
    );
    assert!(
        wall_control_drift <= 0.05,
        "wall control drift {wall_control_drift}"
    );
    assert!(
        gpu_candidate_drift <= 0.05,
        "GPU candidate drift {gpu_candidate_drift}"
    );
    assert!(
        wall_candidate_drift <= 0.05,
        "wall candidate drift {wall_candidate_drift}"
    );
    assert!(
        gpu_median_saving >= 0.58,
        "GPU median saving {gpu_median_saving}"
    );
    assert!(
        wall_median_saving >= 0.55,
        "wall median saving {wall_median_saving}"
    );
    assert!(gpu_p95_saving >= 0.55, "GPU p95 saving {gpu_p95_saving}");
    assert!(wall_p95_saving >= 0.50, "wall p95 saving {wall_p95_saving}");
    assert!(
        candidate_gpu_median <= 3.75,
        "candidate GPU {candidate_gpu_median} ms"
    );
}

#[test]
fn packed_grouped_iq2_xs_iq3_xxs_matches_bucket_path() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const H: usize = 256;
    const F: usize = 256;
    const E: usize = 216;
    const K: usize = MOE_TOP_K;
    const CLAMP: f32 = 0.25;

    let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 1);
    let up_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 3);
    let down_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, F, H, E, 5);
    let tile_buffer =
        MetalTensor::zeros_i32(&ctx, vec![PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64]).unwrap();

    for n_tokens in [1, 12, 31, 32, 33, 64, 128, 2_048] {
        let (_expert_ids, rows, slots, schedule) = grouped_test_schedule_for_experts(n_tokens, E);
        let grouped_plan =
            PackedGroupedExpertPlan::new(n_tokens, &schedule, Some(&tile_buffer)).unwrap();
        let input_values = (0..n_tokens * H)
            .map(|index| ((index * 37 + 5) % 251) as f32 * 0.001 - 0.125)
            .collect::<Vec<_>>();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input_values),
            vec![H as u64, n_tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let bucket_rows = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&rows),
            vec![(n_tokens * K) as u64],
            GgmlType::I32,
        )
        .unwrap();
        let bucket_slots = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&slots),
            vec![(n_tokens * K) as u64],
            GgmlType::I32,
        )
        .unwrap();

        let expert_input = MetalTensor::zeros_f32(&ctx, vec![H as u64, n_tokens as u64]).unwrap();
        let gate = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
        let up = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
        let bucket_inner = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
        let bucket_output = MetalTensor::zeros_f32(&ctx, vec![H as u64, n_tokens as u64]).unwrap();
        let control_inner =
            grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 7.0);
        let control_output =
            grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 9.0);
        let candidate_inner =
            grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 11.0);
        let candidate_output =
            grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 13.0);
        let repeat_inner =
            grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 15.0);
        let repeat_output =
            grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 17.0);

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for bucket in &schedule {
            let row_view = i32_slice(
                &bucket_rows,
                bucket.start,
                bucket.len,
                "grouped control rows",
            )
            .unwrap();
            let slot_view = i32_slice(
                &bucket_slots,
                bucket.start,
                bucket.len,
                "grouped control slots",
            )
            .unwrap();
            let input_view = f32_prefix(
                &expert_input,
                vec![H as u64, bucket.len as u64],
                "grouped control input",
            )
            .unwrap();
            let gate_view = f32_prefix(
                &gate,
                vec![F as u64, bucket.len as u64],
                "grouped control gate",
            )
            .unwrap();
            let up_view =
                f32_prefix(&up, vec![F as u64, bucket.len as u64], "grouped control up").unwrap();
            let inner_view = f32_prefix(
                &bucket_inner,
                vec![F as u64, bucket.len as u64],
                "grouped control inner",
            )
            .unwrap();
            let output_view = f32_prefix(
                &bucket_output,
                vec![H as u64, bucket.len as u64],
                "grouped control output",
            )
            .unwrap();
            encode_get_rows_f32(
                &ctx,
                &encoder,
                &input,
                &row_view,
                &input_view,
                bucket.len,
                H,
            )
            .unwrap();
            let gate_weight =
                expert_weight_view(&gate_bank, H, F, bucket.expert, "grouped control gate")
                    .unwrap();
            let up_weight =
                expert_weight_view(&up_bank, H, F, bucket.expert, "grouped control up").unwrap();
            let down_weight =
                expert_weight_view(&down_bank, F, H, bucket.expert, "grouped control down")
                    .unwrap();
            encode_batch_projection(
                &ctx,
                &encoder,
                &gate_weight,
                &input_view,
                &gate_view,
                H,
                F,
                bucket.len,
                "grouped control gate",
            )
            .unwrap();
            encode_batch_projection(
                &ctx,
                &encoder,
                &up_weight,
                &input_view,
                &up_view,
                H,
                F,
                bucket.len,
                "grouped control up",
            )
            .unwrap();
            encode_ds4_clamped_swiglu(
                &ctx,
                &encoder,
                &gate_view.view_subrange(0, vec![(F * bucket.len) as u64]),
                &up_view.view_subrange(0, vec![(F * bucket.len) as u64]),
                &inner_view.view_subrange(0, vec![(F * bucket.len) as u64]),
                CLAMP,
            )
            .unwrap();
            encode_batch_projection(
                &ctx,
                &encoder,
                &down_weight,
                &inner_view,
                &output_view,
                F,
                H,
                bucket.len,
                "grouped control down",
            )
            .unwrap();
            crate::metal::encode_scatter_rows_f32_unique(
                &ctx,
                &encoder,
                &inner_view,
                &slot_view,
                &control_inner,
                F,
                bucket.len,
            )
            .unwrap();
            crate::metal::encode_scatter_rows_f32_unique(
                &ctx,
                &encoder,
                &output_view,
                &slot_view,
                &control_output,
                H,
                bucket.len,
            )
            .unwrap();
        }
        for (inner, output) in [
            (&candidate_inner, &candidate_output),
            (&repeat_inner, &repeat_output),
        ] {
            encode_packed_grouped_swiglu_iq2_xs_f32(
                &ctx,
                &encoder,
                &gate_bank,
                &up_bank,
                &input,
                &bucket_slots,
                &grouped_plan,
                inner,
                H,
                F,
                E,
                K,
                n_tokens,
                CLAMP,
            )
            .unwrap();
            encode_packed_grouped_down_iq3_xxs_f32(
                &ctx,
                &encoder,
                &down_bank,
                inner,
                &bucket_slots,
                &grouped_plan,
                output,
                F,
                H,
                E,
                K,
                n_tokens,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(
            command.error().is_none(),
            "N={n_tokens}: {:?}",
            command.error()
        );

        let bits = |tensor: &MetalTensor| {
            host_read_f32(tensor, "packed grouped differential")
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(&candidate_inner),
            bits(&control_inner),
            "N={n_tokens} inner"
        );
        assert_eq!(
            bits(&candidate_output),
            bits(&control_output),
            "N={n_tokens} output"
        );
        assert_eq!(
            bits(&repeat_inner),
            bits(&candidate_inner),
            "N={n_tokens} repeat inner"
        );
        assert_eq!(
            bits(&repeat_output),
            bits(&candidate_output),
            "N={n_tokens} repeat output"
        );
        for (label, tensor) in [
            ("control inner", &control_inner),
            ("control output", &control_output),
            ("candidate inner", &candidate_inner),
            ("candidate output", &candidate_output),
            ("repeat inner", &repeat_inner),
            ("repeat output", &repeat_output),
        ] {
            assert_grouped_guards(label, tensor);
        }
    }
}

#[test]
fn packed_grouped_iq2_xs_f32_matrix_schedules_match_reduced_k_scalar() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const H: usize = 256;
    const F: usize = 256;
    const E: usize = 216;
    const K: usize = MOE_TOP_K;
    const CLAMP: f32 = 0.25;

    fn metrics(reference: &[f32], candidate: &[f32]) -> (f64, f64, f64) {
        assert_eq!(reference.len(), candidate.len());
        let mut dot = 0.0f64;
        let mut reference_sq = 0.0f64;
        let mut candidate_sq = 0.0f64;
        let mut error_sq = 0.0f64;
        let mut max_abs = 0.0f64;
        for (&reference, &candidate) in reference.iter().zip(candidate) {
            assert!(reference.is_finite() && candidate.is_finite());
            let reference = f64::from(reference);
            let candidate = f64::from(candidate);
            let error = candidate - reference;
            dot += reference * candidate;
            reference_sq += reference * reference;
            candidate_sq += candidate * candidate;
            error_sq += error * error;
            max_abs = max_abs.max(error.abs());
        }
        (
            dot / (reference_sq.sqrt() * candidate_sq.sqrt()),
            (error_sq / reference_sq).sqrt(),
            max_abs,
        )
    }

    let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 71);
    let up_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 73);
    let grouped_tile_buffer =
        MetalTensor::zeros_i32(&ctx, vec![PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64]).unwrap();
    let mma16_tile_buffer =
        MetalTensor::zeros_i32(&ctx, vec![PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS as u64])
            .unwrap();
    for n_tokens in [1, 12, 15, 16, 17, 31, 32, 33, 64, 128, 337, 2_048, 4_096] {
        let route_count = n_tokens * K;
        let (_expert_ids, rows, slots, schedule) = grouped_test_schedule_for_experts(n_tokens, E);
        let grouped_plan =
            PackedGroupedExpertPlan::new(n_tokens, &schedule, Some(&grouped_tile_buffer)).unwrap();
        let mma16_plan =
            PackedGroupedExpertPlan::new_iq2_mma16(n_tokens, &schedule, Some(&mma16_tile_buffer))
                .unwrap();
        let input_values = (0..n_tokens * H)
            .map(|index| ((index * 37 + index / 11 + 5) % 251) as f32 * 0.001 - 0.125)
            .collect::<Vec<_>>();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input_values),
            vec![H as u64, n_tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let source_rows = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&rows),
            vec![route_count as u64],
            GgmlType::I32,
        )
        .unwrap();
        let destination_slots = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&slots),
            vec![route_count as u64],
            GgmlType::I32,
        )
        .unwrap();

        let expert_input = MetalTensor::zeros_f32(&ctx, vec![H as u64, n_tokens as u64]).unwrap();
        let bucket_gate = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
        let bucket_up = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
        let control_gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 74.0);
        let control_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 76.0);
        let control_inner =
            grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 79.0);
        let candidate_gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 83.0);
        let candidate_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 89.0);
        let candidate_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 97.0);
        let repeat_gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 101.0);
        let repeat_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 103.0);
        let repeat_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 107.0);
        let half_gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 109.0);
        let half_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 113.0);
        let half_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 127.0);

        let _trace = crate::metal::kernel_trace_begin();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for bucket in &schedule {
            let row_view = i32_slice(
                &source_rows,
                bucket.start,
                bucket.len,
                "IQ2 MMA control rows",
            )
            .unwrap();
            let slot_view = i32_slice(
                &destination_slots,
                bucket.start,
                bucket.len,
                "IQ2 MMA control slots",
            )
            .unwrap();
            let input_view = f32_prefix(
                &expert_input,
                vec![H as u64, bucket.len as u64],
                "IQ2 MMA control input",
            )
            .unwrap();
            let gate_view = f32_prefix(
                &bucket_gate,
                vec![F as u64, bucket.len as u64],
                "IQ2 MMA control gate",
            )
            .unwrap();
            let up_view = f32_prefix(
                &bucket_up,
                vec![F as u64, bucket.len as u64],
                "IQ2 MMA control up",
            )
            .unwrap();
            encode_get_rows_f32(
                &ctx,
                &encoder,
                &input,
                &row_view,
                &input_view,
                bucket.len,
                H,
            )
            .unwrap();
            for (bank, temporary, destination, name) in [
                (
                    &gate_bank,
                    &gate_view,
                    &control_gate,
                    "IQ2 MMA control gate",
                ),
                (&up_bank, &up_view, &control_up, "IQ2 MMA control up"),
            ] {
                let weight = expert_weight_view(bank, H, F, bucket.expert, name).unwrap();
                encode_batch_projection(
                    &ctx,
                    &encoder,
                    &weight,
                    &input_view,
                    temporary,
                    H,
                    F,
                    bucket.len,
                    name,
                )
                .unwrap();
                crate::metal::encode_scatter_rows_f32_unique(
                    &ctx,
                    &encoder,
                    temporary,
                    &slot_view,
                    destination,
                    F,
                    bucket.len,
                )
                .unwrap();
            }
        }
        encode_packed_grouped_swiglu_iq2_xs_f32(
            &ctx,
            &encoder,
            &gate_bank,
            &up_bank,
            &input,
            &destination_slots,
            &grouped_plan,
            &control_inner,
            H,
            F,
            E,
            K,
            n_tokens,
            CLAMP,
        )
        .unwrap();
        for (plan, work_unit, gate, up, inner) in [
            (
                &mma16_plan,
                PackedIq2MatrixWorkUnit::Mma16,
                &candidate_gate,
                &candidate_up,
                &candidate_inner,
            ),
            (
                &grouped_plan,
                PackedIq2MatrixWorkUnit::Mm64x32,
                &repeat_gate,
                &repeat_up,
                &repeat_inner,
            ),
            (
                &grouped_plan,
                PackedIq2MatrixWorkUnit::Mm64x32F16,
                &half_gate,
                &half_up,
                &half_inner,
            ),
        ] {
            encode_packed_grouped_mapped_iq2_xs_swiglu_f32_matrix(
                &ctx,
                &encoder,
                &gate_bank,
                &up_bank,
                &input,
                &source_rows,
                &destination_slots,
                plan,
                gate,
                up,
                inner,
                H,
                F,
                E,
                K,
                n_tokens,
                n_tokens,
                route_count,
                CLAMP,
                work_unit,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(
            command.error().is_none(),
            "N={n_tokens}: {:?}",
            command.error()
        );
        let trace = crate::metal::kernel_trace_take_delta();
        assert_eq!(trace.encoders, 1);
        assert_eq!(trace.concurrent_encoders, 0);
        assert_eq!(trace.dispatches, (schedule.len() * 5 + 10) as u64);

        let read = |tensor: &MetalTensor, label| host_read_f32(tensor, label).unwrap();
        let control = read(&control_inner, "IQ2 MMA control inner");
        let control_gate_values = read(&control_gate, "IQ2 MMA control gate");
        let control_up_values = read(&control_up, "IQ2 MMA control up");
        let candidate_gate_values = read(&candidate_gate, "IQ2 MMA candidate gate");
        let candidate_up_values = read(&candidate_up, "IQ2 MMA candidate up");
        let candidate = read(&candidate_inner, "IQ2 MMA candidate inner");
        let wide_gate_values = read(&repeat_gate, "IQ2 MM64x32 gate");
        let wide_up_values = read(&repeat_up, "IQ2 MM64x32 up");
        let wide = read(&repeat_inner, "IQ2 MM64x32 inner");
        let half_gate_values = read(&half_gate, "IQ2 F16 MM64x32 gate");
        let half_up_values = read(&half_up, "IQ2 F16 MM64x32 up");
        let half = read(&half_inner, "IQ2 F16 MM64x32 inner");
        let gate_result = metrics(&control_gate_values, &candidate_gate_values);
        let up_result = metrics(&control_up_values, &candidate_up_values);
        let result = metrics(&control, &candidate);
        let half_gate_result = metrics(&wide_gate_values, &half_gate_values);
        let half_up_result = metrics(&wide_up_values, &half_up_values);
        let half_result = metrics(&wide, &half);
        eprintln!(
            "deepseek_v4 iq2_mma16_differential n={n_tokens} gate={gate_result:?} up={up_result:?} inner={result:?} half_gate={half_gate_result:?} half_up={half_up_result:?} half_inner={half_result:?}",
        );
        assert!(
            result.0 >= 0.999 && result.1 <= 0.05,
            "N={n_tokens} numerical gate failed: {result:?}"
        );
        assert!(
            [half_gate_result, half_up_result, half_result]
                .into_iter()
                .all(|result| result.0 >= 0.999 && result.1 <= 0.05),
            "N={n_tokens} half-staged numerical gate failed: gate={half_gate_result:?} up={half_up_result:?} inner={half_result:?}"
        );
        assert_eq!(
            control_gate_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            candidate_gate_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "N={n_tokens} reduced-K gate scalar lineage"
        );
        assert_eq!(
            control_up_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            candidate_up_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "N={n_tokens} reduced-K up scalar lineage"
        );
        assert_eq!(
            control
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            candidate
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "N={n_tokens} reduced-K SwiGLU scalar lineage"
        );
        assert_eq!(
            candidate
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            wide.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            "N={n_tokens} MM64x32 inner"
        );
        assert_eq!(
            candidate_gate_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            wide_gate_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "N={n_tokens} MM64x32 gate"
        );
        assert_eq!(
            candidate_up_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            wide_up_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "N={n_tokens} MM64x32 up"
        );
        for (label, tensor) in [
            ("IQ2 MMA control gate", &control_gate),
            ("IQ2 MMA control up", &control_up),
            ("IQ2 MMA control inner", &control_inner),
            ("IQ2 MMA candidate gate", &candidate_gate),
            ("IQ2 MMA candidate up", &candidate_up),
            ("IQ2 MMA candidate inner", &candidate_inner),
            ("IQ2 MM64x32 gate", &repeat_gate),
            ("IQ2 MM64x32 up", &repeat_up),
            ("IQ2 MM64x32 inner", &repeat_inner),
            ("IQ2 F16 MM64x32 gate", &half_gate),
            ("IQ2 F16 MM64x32 up", &half_up),
            ("IQ2 F16 MM64x32 inner", &half_inner),
        ] {
            assert_grouped_guards(label, tensor);
        }
    }
}

#[test]
fn packed_grouped_iq2_xs_f32_matrix_schedules_traverse_production_k() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const H: usize = 4_096;
    const F: usize = 64;
    const E: usize = MOE_EXPERT_COUNT;
    const K: usize = MOE_TOP_K;
    const CLAMP: f32 = 0.25;

    fn metrics(reference: &[f32], candidate: &[f32]) -> (f64, f64, f64) {
        assert_eq!(reference.len(), candidate.len());
        let mut dot = 0.0f64;
        let mut reference_sq = 0.0f64;
        let mut candidate_sq = 0.0f64;
        let mut error_sq = 0.0f64;
        let mut max_abs = 0.0f64;
        for (&reference, &candidate) in reference.iter().zip(candidate) {
            assert!(reference.is_finite() && candidate.is_finite());
            let reference = f64::from(reference);
            let candidate = f64::from(candidate);
            let error = candidate - reference;
            dot += reference * candidate;
            reference_sq += reference * reference;
            candidate_sq += candidate * candidate;
            error_sq += error * error;
            max_abs = max_abs.max(error.abs());
        }
        (
            dot / (reference_sq.sqrt() * candidate_sq.sqrt()),
            (error_sq / reference_sq).sqrt(),
            max_abs,
        )
    }

    let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 109);
    let up_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 113);
    let grouped_tile_buffer =
        MetalTensor::zeros_i32(&ctx, vec![PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64]).unwrap();
    let mma16_tile_buffer =
        MetalTensor::zeros_i32(&ctx, vec![PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS as u64])
            .unwrap();
    for n_tokens in [1, 15, 16, 17, 128, 337, 2_048, 4_096] {
        let route_count = n_tokens * K;
        let (_expert_ids, rows, slots, schedule) = grouped_test_schedule(n_tokens);
        let grouped_plan =
            PackedGroupedExpertPlan::new(n_tokens, &schedule, Some(&grouped_tile_buffer)).unwrap();
        let mma16_plan =
            PackedGroupedExpertPlan::new_iq2_mma16(n_tokens, &schedule, Some(&mma16_tile_buffer))
                .unwrap();
        let input_values = (0..n_tokens * H)
            .map(|index| {
                ((index * 41 + index / 17 + index / H * 13 + 7) % 509) as f32 * 0.0005 - 0.127
            })
            .collect::<Vec<_>>();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input_values),
            vec![H as u64, n_tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let source_rows = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&rows),
            vec![route_count as u64],
            GgmlType::I32,
        )
        .unwrap();
        let destination_slots = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&slots),
            vec![route_count as u64],
            GgmlType::I32,
        )
        .unwrap();
        let expert_input = MetalTensor::zeros_f32(&ctx, vec![H as u64, n_tokens as u64]).unwrap();
        let bucket_gate = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
        let bucket_up = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
        let control_gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 117.0);
        let control_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 121.0);
        let control_inner =
            grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 127.0);
        let candidate_gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 131.0);
        let candidate_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 137.0);
        let candidate_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 139.0);
        let repeat_gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 149.0);
        let repeat_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 151.0);
        let repeat_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 157.0);
        let half_gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 163.0);
        let half_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 167.0);
        let half_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 173.0);

        let _trace = crate::metal::kernel_trace_begin();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for bucket in &schedule {
            let row_view = i32_slice(
                &source_rows,
                bucket.start,
                bucket.len,
                "production-K control rows",
            )
            .unwrap();
            let slot_view = i32_slice(
                &destination_slots,
                bucket.start,
                bucket.len,
                "production-K control slots",
            )
            .unwrap();
            let input_view = f32_prefix(
                &expert_input,
                vec![H as u64, bucket.len as u64],
                "production-K control input",
            )
            .unwrap();
            let gate_view = f32_prefix(
                &bucket_gate,
                vec![F as u64, bucket.len as u64],
                "production-K control gate",
            )
            .unwrap();
            let up_view = f32_prefix(
                &bucket_up,
                vec![F as u64, bucket.len as u64],
                "production-K control up",
            )
            .unwrap();
            encode_get_rows_f32(
                &ctx,
                &encoder,
                &input,
                &row_view,
                &input_view,
                bucket.len,
                H,
            )
            .unwrap();
            for (bank, temporary, destination, name) in [
                (
                    &gate_bank,
                    &gate_view,
                    &control_gate,
                    "production-K control gate",
                ),
                (&up_bank, &up_view, &control_up, "production-K control up"),
            ] {
                let weight = expert_weight_view(bank, H, F, bucket.expert, name).unwrap();
                encode_batch_projection(
                    &ctx,
                    &encoder,
                    &weight,
                    &input_view,
                    temporary,
                    H,
                    F,
                    bucket.len,
                    name,
                )
                .unwrap();
                crate::metal::encode_scatter_rows_f32_unique(
                    &ctx,
                    &encoder,
                    temporary,
                    &slot_view,
                    destination,
                    F,
                    bucket.len,
                )
                .unwrap();
            }
        }
        encode_packed_grouped_swiglu_iq2_xs_f32(
            &ctx,
            &encoder,
            &gate_bank,
            &up_bank,
            &input,
            &destination_slots,
            &grouped_plan,
            &control_inner,
            H,
            F,
            E,
            K,
            n_tokens,
            CLAMP,
        )
        .unwrap();
        for (plan, work_unit, gate, up, inner) in [
            (
                &mma16_plan,
                PackedIq2MatrixWorkUnit::Mma16,
                &candidate_gate,
                &candidate_up,
                &candidate_inner,
            ),
            (
                &grouped_plan,
                PackedIq2MatrixWorkUnit::Mm64x32,
                &repeat_gate,
                &repeat_up,
                &repeat_inner,
            ),
            (
                &grouped_plan,
                PackedIq2MatrixWorkUnit::Mm64x32F16,
                &half_gate,
                &half_up,
                &half_inner,
            ),
        ] {
            encode_packed_grouped_mapped_iq2_xs_swiglu_f32_matrix(
                &ctx,
                &encoder,
                &gate_bank,
                &up_bank,
                &input,
                &source_rows,
                &destination_slots,
                plan,
                gate,
                up,
                inner,
                H,
                F,
                E,
                K,
                n_tokens,
                n_tokens,
                route_count,
                CLAMP,
                work_unit,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(
            command.error().is_none(),
            "N={n_tokens}: {:?}",
            command.error()
        );
        let trace = crate::metal::kernel_trace_take_delta();
        assert_eq!(trace.encoders, 1);
        assert_eq!(trace.concurrent_encoders, 0);
        assert_eq!(trace.dispatches, (schedule.len() * 5 + 10) as u64);

        let control_gate_values = host_read_f32(&control_gate, "production-K scalar gate").unwrap();
        let control_up_values = host_read_f32(&control_up, "production-K scalar up").unwrap();
        let candidate_gate_values =
            host_read_f32(&candidate_gate, "production-K BM16 gate").unwrap();
        let candidate_up_values = host_read_f32(&candidate_up, "production-K BM16 up").unwrap();
        let control = host_read_f32(&control_inner, "production-K scalar inner").unwrap();
        let candidate = host_read_f32(&candidate_inner, "production-K BM16 inner").unwrap();
        let wide_gate_values = host_read_f32(&repeat_gate, "production-K MM64x32 gate").unwrap();
        let wide_up_values = host_read_f32(&repeat_up, "production-K MM64x32 up").unwrap();
        let wide = host_read_f32(&repeat_inner, "production-K MM64x32 inner").unwrap();
        let half_gate_values = host_read_f32(&half_gate, "production-K F16 MM64x32 gate").unwrap();
        let half_up_values = host_read_f32(&half_up, "production-K F16 MM64x32 up").unwrap();
        let half = host_read_f32(&half_inner, "production-K F16 MM64x32 inner").unwrap();
        let result = metrics(&control, &candidate);
        let half_gate_result = metrics(&wide_gate_values, &half_gate_values);
        let half_up_result = metrics(&wide_up_values, &half_up_values);
        let half_result = metrics(&wide, &half);
        eprintln!(
            "deepseek_v4 iq2_mma16_production_k n={n_tokens} cosine={:.9} rel_rms={:.9} max_abs={:.9} half_gate={half_gate_result:?} half_up={half_up_result:?} half_inner={half_result:?}",
            result.0, result.1, result.2,
        );
        assert!(
            result.0 >= 0.999_999 && result.1 <= 0.001,
            "N={n_tokens} production-K numerical gate failed: {result:?}"
        );
        assert!(
            [half_gate_result, half_up_result, half_result]
                .into_iter()
                .all(|result| result.0 >= 0.999_99 && result.1 <= 0.005),
            "N={n_tokens} production-K half-staged numerical gate failed: gate={half_gate_result:?} up={half_up_result:?} inner={half_result:?}"
        );
        assert_eq!(
            control_gate_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            candidate_gate_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "N={n_tokens} production-K gate scalar lineage"
        );
        assert_eq!(
            control_up_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            candidate_up_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "N={n_tokens} production-K up scalar lineage"
        );
        assert_eq!(
            control
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            candidate
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "N={n_tokens} production-K SwiGLU scalar lineage"
        );
        assert_eq!(
            candidate
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            wide.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            "N={n_tokens} production-K MM64x32 inner"
        );
        assert_eq!(
            candidate_gate_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            wide_gate_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "N={n_tokens} production-K MM64x32 gate"
        );
        assert_eq!(
            candidate_up_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            wide_up_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "N={n_tokens} production-K MM64x32 up"
        );
        for (label, tensor) in [
            ("production-K control gate", &control_gate),
            ("production-K control up", &control_up),
            ("production-K control inner", &control_inner),
            ("production-K candidate gate", &candidate_gate),
            ("production-K candidate up", &candidate_up),
            ("production-K candidate inner", &candidate_inner),
            ("production-K MM64x32 gate", &repeat_gate),
            ("production-K MM64x32 up", &repeat_up),
            ("production-K MM64x32 inner", &repeat_inner),
            ("production-K F16 MM64x32 gate", &half_gate),
            ("production-K F16 MM64x32 up", &half_up),
            ("production-K F16 MM64x32 inner", &half_inner),
        ] {
            assert_grouped_guards(label, tensor);
        }
    }
}

#[test]
fn packed_gpu_route_compaction_feeds_all_iq3_at_n2048() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    assert!(packed_grouped_iq3_candidate_supported(&ctx));
    const H: usize = 512;
    const F: usize = 256;
    const E: usize = MOE_EXPERT_COUNT;
    const K: usize = MOE_TOP_K;
    const N: usize = PACKED_GPU_ROUTE_MAX_TOKENS;
    const CLAMP: f32 = 0.25;

    let fixture = PackedRouteFixture::new(&ctx);
    let policy = PackedExpertPolicy::GroupedIq2XsIq3XxsMma16QualifiedChunk.with_iq3_target();
    assert!(packed_gpu_compact_expert_layer_qualified(
        &ctx,
        policy,
        N,
        true,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_XXS,
    ));
    assert!(!packed_gpu_compact_expert_layer_qualified(
        &ctx,
        policy,
        N,
        false,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_XXS,
    ));
    assert!(packed_gpu_compact_expert_layer_qualified(
        &ctx,
        policy,
        N,
        false,
        GgmlType::IQ2_XS,
        GgmlType::IQ2_XS,
        GgmlType::IQ3_XXS,
    ));
    assert!(!packed_gpu_compact_expert_layer_qualified(
        &ctx,
        policy,
        N,
        true,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_XXS,
        GgmlType::MXFP4,
    ));

    let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, F, E, 163);
    let up_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, F, E, 167);
    let down_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, F, H, E, 173);
    let route_count = N * K;
    let input_values = (0..N * H)
        .map(|index| ((index * 43 + index / H * 17 + 11) % 509) as f32 * 0.0005 - 0.127)
        .collect::<Vec<_>>();
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&input_values),
        vec![H as u64, N as u64],
        GgmlType::F32,
    )
    .unwrap();
    let device_plan = PackedGroupedExpertPlan::from_device(
        &fixture.scratch.compact_tiles32,
        PACKED_GROUPED_EXPERT_MAX_TILES,
    )
    .unwrap();
    let device_output = grouped_guarded_f32(&ctx, vec![H as u64, K as u64, N as u64], 181.0);
    let device_output_flat = device_output.view_subrange(0, vec![H as u64, route_count as u64]);
    let device_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 183.0);
    let generation = fixture.generations.take().unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    fixture
        .scratch
        .encode_learned(&ctx, &encoder, &fixture.bias, N, N, generation)
        .unwrap();
    fixture
        .scratch
        .encode_compact(&ctx, &encoder, N, generation)
        .unwrap();
    encode_packed_grouped_all_iq3(
        &ctx,
        &encoder,
        &gate_bank,
        &up_bank,
        &down_bank,
        &input,
        &fixture.scratch.compact_rows,
        &fixture.scratch.compact_slots,
        &device_plan,
        &device_output_flat,
        &device_inner,
        H,
        F,
        E,
        K,
        N,
        CLAMP,
    )
    .unwrap();
    encoder.end();
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert!(command.error().is_none(), "{:?}", command.error());

    let capture = fixture.scratch.capture_compact(N);
    assert_packed_compact_capture(
        &fixture,
        PackedRouteMicroproofSource::Learned,
        N,
        generation.get(),
        &capture,
    );
    let (_, _, _, schedule) = expected_compact_schedule(&capture.expert_ids, N);
    let host_descriptors =
        MetalTensor::zeros_i32(&ctx, vec![PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64]).unwrap();
    let host_plan = PackedGroupedExpertPlan::new(N, &schedule, Some(&host_descriptors)).unwrap();
    let host_output = grouped_guarded_f32(&ctx, vec![H as u64, K as u64, N as u64], 191.0);
    let host_output_flat = host_output.view_subrange(0, vec![H as u64, route_count as u64]);
    let host_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 193.0);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_packed_grouped_all_iq3(
        &ctx,
        &encoder,
        &gate_bank,
        &up_bank,
        &down_bank,
        &input,
        &fixture.scratch.compact_rows,
        &fixture.scratch.compact_slots,
        &host_plan,
        &host_output_flat,
        &host_inner,
        H,
        F,
        E,
        K,
        N,
        CLAMP,
    )
    .unwrap();
    encoder.end();
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert!(command.error().is_none(), "{:?}", command.error());

    let bits = |tensor: &MetalTensor, label| {
        host_read_f32(tensor, label)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bits(&device_inner, "all-IQ3 device-plan inner"),
        bits(&host_inner, "all-IQ3 host-plan inner")
    );
    assert_eq!(
        bits(&device_output, "all-IQ3 device-plan output"),
        bits(&host_output, "all-IQ3 host-plan output")
    );
    for (label, tensor) in [
        ("all-IQ3 host-plan inner", &host_inner),
        ("all-IQ3 host-plan output", &host_output),
        ("all-IQ3 device-plan inner", &device_inner),
        ("all-IQ3 device-plan output", &device_output),
    ] {
        assert_grouped_guards(label, tensor);
    }
}

#[test]
fn packed_grouped_all_iq3_matches_bucket_path_and_arena_contract() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    assert!(
        packed_grouped_iq3_fused_candidate_supported(&ctx),
        "fused all-IQ3 pipeline is not qualified"
    );
    const H: usize = 512;
    const F: usize = 256;
    const E: usize = 216;
    const K: usize = MOE_TOP_K;
    const CLAMP: f32 = 0.25;

    fn submit(
        ctx: &MetalContext,
        label: &str,
        encode: impl FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    ) {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let result = encode(&encoder);
        encoder.end();
        result.unwrap();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(command.error().is_none(), "{label}: {:?}", command.error());
    }

    let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, F, E, 29);
    let up_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, F, E, 31);
    let down_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, F, H, E, 37);

    for n_tokens in [1, 12, 31, 32, 33, 64, 128] {
        let route_count = n_tokens * K;
        let (_expert_ids, rows, slots, schedule) = grouped_test_schedule_for_experts(n_tokens, E);
        let input_values = (0..n_tokens * H)
            .map(|index| ((index * 43 + 11) % 263) as f32 * 0.001 - 0.125)
            .collect::<Vec<_>>();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input_values),
            vec![H as u64, n_tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let bucket_rows = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&rows),
            vec![route_count as u64],
            GgmlType::I32,
        )
        .unwrap();
        let bucket_slots = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&slots),
            vec![route_count as u64],
            GgmlType::I32,
        )
        .unwrap();

        let expert_input = MetalTensor::zeros_f32(&ctx, vec![H as u64, n_tokens as u64]).unwrap();
        let gate = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
        let up = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
        let bucket_inner = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
        let bucket_output = MetalTensor::zeros_f32(&ctx, vec![H as u64, n_tokens as u64]).unwrap();
        let control_gate =
            grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 3.0);
        let control_up = grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 5.0);
        let control_inner =
            grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 7.0);
        let control_output =
            grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 9.0);

        submit(&ctx, "all-IQ3 bucket control", |encoder| {
            for bucket in &schedule {
                let row_view = i32_slice(
                    &bucket_rows,
                    bucket.start,
                    bucket.len,
                    "all-IQ3 control rows",
                )?;
                let slot_view = i32_slice(
                    &bucket_slots,
                    bucket.start,
                    bucket.len,
                    "all-IQ3 control slots",
                )?;
                let input_view = f32_prefix(
                    &expert_input,
                    vec![H as u64, bucket.len as u64],
                    "all-IQ3 control input",
                )?;
                let gate_view = f32_prefix(
                    &gate,
                    vec![F as u64, bucket.len as u64],
                    "all-IQ3 control gate",
                )?;
                let up_view =
                    f32_prefix(&up, vec![F as u64, bucket.len as u64], "all-IQ3 control up")?;
                let inner_view = f32_prefix(
                    &bucket_inner,
                    vec![F as u64, bucket.len as u64],
                    "all-IQ3 control inner",
                )?;
                let output_view = f32_prefix(
                    &bucket_output,
                    vec![H as u64, bucket.len as u64],
                    "all-IQ3 control output",
                )?;
                encode_get_rows_f32(&ctx, encoder, &input, &row_view, &input_view, bucket.len, H)?;
                let gate_weight =
                    expert_weight_view(&gate_bank, H, F, bucket.expert, "all-IQ3 control gate")?;
                let up_weight =
                    expert_weight_view(&up_bank, H, F, bucket.expert, "all-IQ3 control up")?;
                let down_weight =
                    expert_weight_view(&down_bank, F, H, bucket.expert, "all-IQ3 control down")?;
                encode_batch_projection(
                    &ctx,
                    encoder,
                    &gate_weight,
                    &input_view,
                    &gate_view,
                    H,
                    F,
                    bucket.len,
                    "all-IQ3 control gate",
                )?;
                encode_batch_projection(
                    &ctx,
                    encoder,
                    &up_weight,
                    &input_view,
                    &up_view,
                    H,
                    F,
                    bucket.len,
                    "all-IQ3 control up",
                )?;
                let inner_len = F * bucket.len;
                encode_ds4_clamped_swiglu(
                    &ctx,
                    encoder,
                    &gate_view.view_subrange(0, vec![inner_len as u64]),
                    &up_view.view_subrange(0, vec![inner_len as u64]),
                    &inner_view.view_subrange(0, vec![inner_len as u64]),
                    CLAMP,
                )?;
                encode_batch_projection(
                    &ctx,
                    encoder,
                    &down_weight,
                    &inner_view,
                    &output_view,
                    F,
                    H,
                    bucket.len,
                    "all-IQ3 control down",
                )?;
                for (source, destination, width) in [
                    (&gate_view, &control_gate, F),
                    (&up_view, &control_up, F),
                    (&inner_view, &control_inner, F),
                    (&output_view, &control_output, H),
                ] {
                    crate::metal::encode_scatter_rows_f32_unique(
                        &ctx,
                        encoder,
                        source,
                        &slot_view,
                        destination,
                        width,
                        bucket.len,
                    )?;
                }
            }
            Ok(())
        });

        if n_tokens == 128 {
            let gate_values = host_read_f32(&control_gate, "all-IQ3 clamp gate").unwrap();
            let up_values = host_read_f32(&control_up, "all-IQ3 clamp up").unwrap();
            let gate_max = gate_values
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let up_min = up_values.iter().copied().fold(f32::INFINITY, f32::min);
            let up_max = up_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                gate_max > CLAMP,
                "synthetic gate never crosses upper clamp: {gate_max}"
            );
            assert!(
                up_min < -CLAMP && up_max > CLAMP,
                "synthetic up misses clamp sides: [{up_min}, {up_max}]"
            );
        }

        let bits = |tensor: &MetalTensor| {
            host_read_f32(tensor, "all-IQ3 grouped differential")
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        let candidate_arena =
            grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 11.0);
        let candidate_output = candidate_arena.view_subrange(0, vec![H as u64, route_count as u64]);
        let (candidate_gate, candidate_up) =
            packed_grouped_gate_up_views(&candidate_output, H, F, route_count).unwrap();
        let candidate_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 13.0);
        let up_poison = bits(&candidate_up);

        submit(&ctx, "all-IQ3 mapped gate", |encoder| {
            encode_packed_grouped_mapped_iq3_xxs_f32(
                &ctx,
                encoder,
                &gate_bank,
                &input,
                &bucket_rows,
                &bucket_slots,
                &schedule,
                &candidate_gate,
                H,
                F,
                E,
                K,
                n_tokens,
                n_tokens,
                route_count,
            )
        });
        assert_eq!(
            bits(&candidate_gate),
            bits(&control_gate),
            "N={n_tokens} gate"
        );
        assert_eq!(bits(&candidate_up), up_poison, "N={n_tokens} up poison");
        let gate_after_gate = bits(&candidate_gate);

        submit(&ctx, "all-IQ3 mapped up", |encoder| {
            encode_packed_grouped_mapped_iq3_xxs_f32(
                &ctx,
                encoder,
                &up_bank,
                &input,
                &bucket_rows,
                &bucket_slots,
                &schedule,
                &candidate_up,
                H,
                F,
                E,
                K,
                n_tokens,
                n_tokens,
                route_count,
            )
        });
        assert_eq!(
            bits(&candidate_gate),
            gate_after_gate,
            "N={n_tokens} gate stable"
        );
        assert_eq!(bits(&candidate_up), bits(&control_up), "N={n_tokens} up");
        let up_after_up = bits(&candidate_up);

        submit(&ctx, "all-IQ3 mapped SwiGLU", |encoder| {
            encode_ds4_clamped_swiglu(
                &ctx,
                encoder,
                &candidate_gate.view_subrange(0, vec![(F * route_count) as u64]),
                &candidate_up.view_subrange(0, vec![(F * route_count) as u64]),
                &candidate_inner.view_subrange(0, vec![(F * route_count) as u64]),
                CLAMP,
            )
        });
        assert_eq!(
            bits(&candidate_gate),
            gate_after_gate,
            "N={n_tokens} gate after SwiGLU"
        );
        assert_eq!(
            bits(&candidate_up),
            up_after_up,
            "N={n_tokens} up after SwiGLU"
        );
        assert_eq!(
            bits(&candidate_inner),
            bits(&control_inner),
            "N={n_tokens} inner"
        );
        let inner_after_swiglu = bits(&candidate_inner);

        submit(&ctx, "all-IQ3 mapped down", |encoder| {
            encode_packed_grouped_mapped_iq3_xxs_f32(
                &ctx,
                encoder,
                &down_bank,
                &candidate_inner,
                &bucket_slots,
                &bucket_slots,
                &schedule,
                &candidate_output,
                F,
                H,
                E,
                K,
                n_tokens,
                route_count,
                route_count,
            )
        });
        assert_eq!(
            bits(&candidate_arena),
            bits(&control_output),
            "N={n_tokens} output"
        );
        assert_eq!(
            bits(&candidate_inner),
            inner_after_swiglu,
            "N={n_tokens} inner after down"
        );

        let fused_arena =
            grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 23.0);
        let fused_output = fused_arena.view_subrange(0, vec![H as u64, route_count as u64]);
        let (fused_gate, fused_up) =
            packed_grouped_gate_up_views(&fused_output, H, F, route_count).unwrap();
        let fused_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 29.0);
        submit(&ctx, "all-IQ3 fused gate/up/SwiGLU", |encoder| {
            encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                &ctx,
                encoder,
                &gate_bank,
                &up_bank,
                &input,
                &bucket_rows,
                &bucket_slots,
                &schedule,
                &fused_output,
                &fused_inner,
                H,
                F,
                E,
                K,
                n_tokens,
                n_tokens,
                route_count,
                CLAMP,
            )
        });
        assert_eq!(
            bits(&fused_gate),
            bits(&control_gate),
            "N={n_tokens} fused gate"
        );
        assert_eq!(bits(&fused_up), bits(&control_up), "N={n_tokens} fused up");
        assert_eq!(
            bits(&fused_inner),
            bits(&control_inner),
            "N={n_tokens} fused inner"
        );
        let fused_gate_after = bits(&fused_gate);
        let fused_up_after = bits(&fused_up);
        let fused_inner_after = bits(&fused_inner);
        submit(&ctx, "all-IQ3 fused mapped down", |encoder| {
            encode_packed_grouped_mapped_iq3_xxs_f32(
                &ctx,
                encoder,
                &down_bank,
                &fused_inner,
                &bucket_slots,
                &bucket_slots,
                &schedule,
                &fused_output,
                F,
                H,
                E,
                K,
                n_tokens,
                route_count,
                route_count,
            )
        });
        assert_eq!(
            bits(&fused_arena),
            bits(&control_output),
            "N={n_tokens} fused output"
        );
        assert_eq!(
            bits(&fused_inner),
            fused_inner_after,
            "N={n_tokens} fused inner stable"
        );

        let fused_repeat_arena =
            grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 31.0);
        let fused_repeat_output =
            fused_repeat_arena.view_subrange(0, vec![H as u64, route_count as u64]);
        let (fused_repeat_gate, fused_repeat_up) =
            packed_grouped_gate_up_views(&fused_repeat_output, H, F, route_count).unwrap();
        let fused_repeat_inner =
            grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 37.0);
        submit(&ctx, "all-IQ3 fused repeat gate/up/SwiGLU", |encoder| {
            encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                &ctx,
                encoder,
                &gate_bank,
                &up_bank,
                &input,
                &bucket_rows,
                &bucket_slots,
                &schedule,
                &fused_repeat_output,
                &fused_repeat_inner,
                H,
                F,
                E,
                K,
                n_tokens,
                n_tokens,
                route_count,
                CLAMP,
            )
        });
        assert_eq!(
            bits(&fused_repeat_gate),
            fused_gate_after,
            "N={n_tokens} fused repeat gate"
        );
        assert_eq!(
            bits(&fused_repeat_up),
            fused_up_after,
            "N={n_tokens} fused repeat up"
        );
        assert_eq!(
            bits(&fused_repeat_inner),
            fused_inner_after,
            "N={n_tokens} fused repeat inner"
        );
        submit(&ctx, "all-IQ3 fused repeat down", |encoder| {
            encode_packed_grouped_mapped_iq3_xxs_f32(
                &ctx,
                encoder,
                &down_bank,
                &fused_repeat_inner,
                &bucket_slots,
                &bucket_slots,
                &schedule,
                &fused_repeat_output,
                F,
                H,
                E,
                K,
                n_tokens,
                route_count,
                route_count,
            )
        });
        assert_eq!(
            bits(&fused_repeat_inner),
            fused_inner_after,
            "N={n_tokens} fused repeat inner"
        );
        assert_eq!(
            bits(&fused_repeat_arena),
            bits(&control_output),
            "N={n_tokens} fused repeat output"
        );

        let fused_chain_arena =
            grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 41.0);
        let fused_chain_output =
            fused_chain_arena.view_subrange(0, vec![H as u64, route_count as u64]);
        let fused_chain_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 43.0);
        submit(&ctx, "all-IQ3 fused one-command chain", |encoder| {
            encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                &ctx,
                encoder,
                &gate_bank,
                &up_bank,
                &input,
                &bucket_rows,
                &bucket_slots,
                &schedule,
                &fused_chain_output,
                &fused_chain_inner,
                H,
                F,
                E,
                K,
                n_tokens,
                n_tokens,
                route_count,
                CLAMP,
            )?;
            encode_packed_grouped_mapped_iq3_xxs_f32(
                &ctx,
                encoder,
                &down_bank,
                &fused_chain_inner,
                &bucket_slots,
                &bucket_slots,
                &schedule,
                &fused_chain_output,
                F,
                H,
                E,
                K,
                n_tokens,
                route_count,
                route_count,
            )
        });
        assert_eq!(
            bits(&fused_chain_arena),
            bits(&control_output),
            "N={n_tokens} fused one-command output"
        );
        assert_eq!(
            bits(&fused_chain_inner),
            bits(&control_inner),
            "N={n_tokens} fused one-command inner"
        );

        let repeat_arena =
            grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 17.0);
        let repeat_output = repeat_arena.view_subrange(0, vec![H as u64, route_count as u64]);
        let (repeat_gate, repeat_up) =
            packed_grouped_gate_up_views(&repeat_output, H, F, route_count).unwrap();
        let repeat_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 19.0);
        submit(&ctx, "all-IQ3 one-command chain", |encoder| {
            encode_packed_grouped_mapped_iq3_xxs_f32(
                &ctx,
                encoder,
                &gate_bank,
                &input,
                &bucket_rows,
                &bucket_slots,
                &schedule,
                &repeat_gate,
                H,
                F,
                E,
                K,
                n_tokens,
                n_tokens,
                route_count,
            )?;
            encode_packed_grouped_mapped_iq3_xxs_f32(
                &ctx,
                encoder,
                &up_bank,
                &input,
                &bucket_rows,
                &bucket_slots,
                &schedule,
                &repeat_up,
                H,
                F,
                E,
                K,
                n_tokens,
                n_tokens,
                route_count,
            )?;
            encode_ds4_clamped_swiglu(
                &ctx,
                encoder,
                &repeat_gate.view_subrange(0, vec![(F * route_count) as u64]),
                &repeat_up.view_subrange(0, vec![(F * route_count) as u64]),
                &repeat_inner.view_subrange(0, vec![(F * route_count) as u64]),
                CLAMP,
            )?;
            encode_packed_grouped_mapped_iq3_xxs_f32(
                &ctx,
                encoder,
                &down_bank,
                &repeat_inner,
                &bucket_slots,
                &bucket_slots,
                &schedule,
                &repeat_output,
                F,
                H,
                E,
                K,
                n_tokens,
                route_count,
                route_count,
            )
        });
        assert_eq!(
            bits(&repeat_arena),
            bits(&control_output),
            "N={n_tokens} repeat output"
        );
        assert_eq!(
            bits(&repeat_inner),
            bits(&control_inner),
            "N={n_tokens} repeat inner"
        );

        for (label, tensor) in [
            ("all-IQ3 control gate", &control_gate),
            ("all-IQ3 control up", &control_up),
            ("all-IQ3 control inner", &control_inner),
            ("all-IQ3 control output", &control_output),
            ("all-IQ3 candidate arena", &candidate_arena),
            ("all-IQ3 candidate inner", &candidate_inner),
            ("all-IQ3 fused arena", &fused_arena),
            ("all-IQ3 fused inner", &fused_inner),
            ("all-IQ3 fused repeat arena", &fused_repeat_arena),
            ("all-IQ3 fused repeat inner", &fused_repeat_inner),
            ("all-IQ3 fused chain arena", &fused_chain_arena),
            ("all-IQ3 fused chain inner", &fused_chain_inner),
            ("all-IQ3 repeat arena", &repeat_arena),
            ("all-IQ3 repeat inner", &repeat_inner),
        ] {
            assert_grouped_guards(label, tensor);
        }
    }
}

#[test]
fn packed_grouped_fused_all_iq3_rejects_invalid_contracts_before_dispatch() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const H: usize = 512;
    const F: usize = 256;
    const E: usize = MOE_EXPERT_COUNT;
    const K: usize = MOE_TOP_K;
    const N: usize = 1;

    fn reject(
        ctx: &MetalContext,
        encode: impl FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
    ) -> String {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let result = encode(&encoder);
        encoder.end();
        result.unwrap_err().to_string()
    }

    let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, F, E, 47);
    let up_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, F, E, 53);
    let wrong_dtype_bank = grouped_test_bank(&ctx, GgmlType::IQ3_S, H, F, E, 59);
    let input = MetalTensor::zeros_f32(&ctx, vec![H as u64, N as u64]).unwrap();
    let (_expert_ids, rows, slots, schedule) = grouped_test_schedule(N);
    let source_rows = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&rows),
        vec![K as u64],
        GgmlType::I32,
    )
    .unwrap();
    let destination_slots = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&slots),
        vec![K as u64],
        GgmlType::I32,
    )
    .unwrap();
    let malformed_rows = MetalTensor::zeros_f32(&ctx, vec![K as u64]).unwrap();
    let arena = MetalTensor::zeros_f32(&ctx, vec![H as u64, K as u64]).unwrap();
    let inner = MetalTensor::zeros_f32(&ctx, vec![F as u64, K as u64]).unwrap();

    let error = reject(&ctx, |encoder| {
        encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
            &ctx,
            encoder,
            &gate_bank,
            &wrong_dtype_bank,
            &input,
            &source_rows,
            &destination_slots,
            &schedule,
            &arena,
            &inner,
            H,
            F,
            E,
            K,
            N,
            N,
            K,
            0.25,
        )
    });
    assert!(error.contains("invalid geometry or storage"), "{error}");

    let error = reject(&ctx, |encoder| {
        encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
            &ctx,
            encoder,
            &gate_bank,
            &up_bank,
            &input,
            &malformed_rows,
            &destination_slots,
            &schedule,
            &arena,
            &inner,
            H,
            F,
            E,
            K,
            N,
            N,
            K,
            0.25,
        )
    });
    assert!(error.contains("source rows"), "{error}");

    let mut malformed_schedule = schedule
        .iter()
        .map(|bucket| ExpertBucket {
            expert: bucket.expert,
            start: bucket.start,
            len: bucket.len,
        })
        .collect::<Vec<_>>();
    malformed_schedule[0].start = 1;
    let error = reject(&ctx, |encoder| {
        encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
            &ctx,
            encoder,
            &gate_bank,
            &up_bank,
            &input,
            &source_rows,
            &destination_slots,
            &malformed_schedule,
            &arena,
            &inner,
            H,
            F,
            E,
            K,
            N,
            N,
            K,
            0.25,
        )
    });
    assert!(error.contains("invalid bucket geometry"), "{error}");

    let overlapping_inner = arena.view_subrange(0, vec![F as u64, K as u64]);
    let error = reject(&ctx, |encoder| {
        encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
            &ctx,
            encoder,
            &gate_bank,
            &up_bank,
            &input,
            &source_rows,
            &destination_slots,
            &schedule,
            &arena,
            &overlapping_inner,
            H,
            F,
            E,
            K,
            N,
            N,
            K,
            0.25,
        )
    });
    assert!(error.contains("inner overlaps gate/up arena"), "{error}");

    let error = reject(&ctx, |encoder| {
        encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
            &ctx,
            encoder,
            &gate_bank,
            &up_bank,
            &input,
            &source_rows,
            &destination_slots,
            &schedule,
            &arena,
            &inner,
            H,
            F,
            E,
            K,
            N,
            N,
            K,
            0.0,
        )
    });
    assert!(error.contains("invalid geometry or storage"), "{error}");
}

#[test]
#[ignore = "sealed KILL; do not rerun without material implementation or device drift"]
fn profile_packed_grouped_fused_all_iq3_production_shape() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    assert_eq!(
        ctx.device.name().to_string(),
        PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE
    );
    assert!(
        packed_grouped_iq3_fused_candidate_supported(&ctx),
        "fused all-IQ3 pipeline is not qualified"
    );

    const H: usize = 4_096;
    const F: usize = 2_048;
    const E: usize = MOE_EXPERT_COUNT;
    const K: usize = MOE_TOP_K;
    const N: usize = 128;
    const ROUTES: usize = N * K;
    const CLAMP: f32 = 7.0;
    const SAMPLES: usize = 24;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Arm {
        Control,
        Candidate,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum ScheduleShape {
        Hot,
        Sparse,
    }

    #[derive(Debug)]
    struct GateResult {
        schedule: &'static str,
        gpu_control_drift: f64,
        wall_control_drift: f64,
        gpu_candidate_drift: f64,
        wall_candidate_drift: f64,
        gpu_median_saving: f64,
        wall_median_saving: f64,
        gpu_p95_saving: f64,
        wall_p95_saving: f64,
    }

    fn timing_bank(ctx: &MetalContext, n_in: usize, n_out: usize, fill: u8) -> MetalTensor {
        let (block_elements, block_bytes) = ggml_type_layout(GgmlType::IQ3_XXS).unwrap();
        let elements = n_in * n_out * E;
        assert!(elements.is_multiple_of(block_elements as usize));
        let bytes = elements / block_elements as usize * block_bytes as usize;
        let buffer = ctx.buffer_uninit(bytes).unwrap();
        unsafe {
            std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), fill, bytes);
        }
        MetalTensor {
            buffer,
            offset: 0,
            shape: vec![n_in as u64, n_out as u64, E as u64],
            dtype: GgmlType::IQ3_XXS,
            provenance: MetalTensorProvenance::OwnedWritable,
        }
    }

    fn timing_schedule(shape: ScheduleShape) -> (Vec<i32>, Vec<i32>, Vec<ExpertBucket>) {
        let mut assignments = (0..E)
            .map(|_| Vec::<(usize, usize)>::new())
            .collect::<Vec<_>>();
        for token in 0..N {
            for slot in 0..K {
                let route_slot = token * K + slot;
                let expert = match shape {
                    ScheduleShape::Hot => slot,
                    ScheduleShape::Sparse => route_slot % E,
                };
                assignments[expert].push((token, route_slot));
            }
        }
        let mut rows = Vec::with_capacity(ROUTES);
        let mut slots = Vec::with_capacity(ROUTES);
        let mut schedule = Vec::new();
        for (expert, assignments) in assignments.into_iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let start = rows.len();
            for (row, slot) in assignments {
                rows.push(row as i32);
                slots.push(slot as i32);
            }
            schedule.push(ExpertBucket {
                expert,
                start,
                len: rows.len() - start,
            });
        }
        assert_eq!(rows.len(), ROUTES);
        assert_eq!(slots.len(), ROUTES);
        assert!(packed_grouped_expert_tiles(N, &schedule).is_ok());
        (rows, slots, schedule)
    }

    let gate_bank = timing_bank(&ctx, H, F, 0x20);
    let up_bank = timing_bank(&ctx, H, F, 0x24);
    let down_bank = timing_bank(&ctx, F, H, 0x28);
    let input_values = (0..H * N)
        .map(|index| ((index * 43 + index / 17 + 11) % 521) as f32 * 0.0005 - 0.13)
        .collect::<Vec<_>>();
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&input_values),
        vec![H as u64, N as u64],
        GgmlType::F32,
    )
    .unwrap();
    eprintln!(
        "deepseek_v4 fused_all_iq3_fixture allocated_bytes={} bank_bytes={}",
        ctx.current_allocated_size(),
        gate_bank.n_bytes() + up_bank.n_bytes() + down_bank.n_bytes(),
    );

    let mut results = Vec::new();
    for schedule_shape in [ScheduleShape::Hot, ScheduleShape::Sparse] {
        let schedule_name = match schedule_shape {
            ScheduleShape::Hot => "hot",
            ScheduleShape::Sparse => "sparse",
        };
        let (rows, slots, schedule) = timing_schedule(schedule_shape);
        let source_rows = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&rows),
            vec![ROUTES as u64],
            GgmlType::I32,
        )
        .unwrap();
        let destination_slots = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&slots),
            vec![ROUTES as u64],
            GgmlType::I32,
        )
        .unwrap();
        let control_arena = grouped_guarded_f32(&ctx, vec![H as u64, ROUTES as u64], 47.0);
        let control_output = control_arena.view_subrange(0, vec![H as u64, ROUTES as u64]);
        let (control_gate, control_up) =
            packed_grouped_gate_up_views(&control_output, H, F, ROUTES).unwrap();
        let control_inner = grouped_guarded_f32(&ctx, vec![F as u64, ROUTES as u64], 53.0);
        let candidate_arena = grouped_guarded_f32(&ctx, vec![H as u64, ROUTES as u64], 59.0);
        let candidate_output = candidate_arena.view_subrange(0, vec![H as u64, ROUTES as u64]);
        let candidate_inner = grouped_guarded_f32(&ctx, vec![F as u64, ROUTES as u64], 61.0);

        let sample = |arm: Arm| -> (f64, f64) {
            let started = std::time::Instant::now();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let result = match arm {
                Arm::Control => (|| {
                    encode_packed_grouped_mapped_iq3_xxs_f32(
                        &ctx,
                        &encoder,
                        &gate_bank,
                        &input,
                        &source_rows,
                        &destination_slots,
                        &schedule,
                        &control_gate,
                        H,
                        F,
                        E,
                        K,
                        N,
                        N,
                        ROUTES,
                    )?;
                    encode_packed_grouped_mapped_iq3_xxs_f32(
                        &ctx,
                        &encoder,
                        &up_bank,
                        &input,
                        &source_rows,
                        &destination_slots,
                        &schedule,
                        &control_up,
                        H,
                        F,
                        E,
                        K,
                        N,
                        N,
                        ROUTES,
                    )?;
                    encode_ds4_clamped_swiglu(
                        &ctx,
                        &encoder,
                        &control_gate.view_subrange(0, vec![(F * ROUTES) as u64]),
                        &control_up.view_subrange(0, vec![(F * ROUTES) as u64]),
                        &control_inner.view_subrange(0, vec![(F * ROUTES) as u64]),
                        CLAMP,
                    )?;
                    encode_packed_grouped_mapped_iq3_xxs_f32(
                        &ctx,
                        &encoder,
                        &down_bank,
                        &control_inner,
                        &destination_slots,
                        &destination_slots,
                        &schedule,
                        &control_output,
                        F,
                        H,
                        E,
                        K,
                        N,
                        ROUTES,
                        ROUTES,
                    )
                })(),
                Arm::Candidate => (|| {
                    encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                        &ctx,
                        &encoder,
                        &gate_bank,
                        &up_bank,
                        &input,
                        &source_rows,
                        &destination_slots,
                        &schedule,
                        &candidate_output,
                        &candidate_inner,
                        H,
                        F,
                        E,
                        K,
                        N,
                        N,
                        ROUTES,
                        CLAMP,
                    )?;
                    encode_packed_grouped_mapped_iq3_xxs_f32(
                        &ctx,
                        &encoder,
                        &down_bank,
                        &candidate_inner,
                        &destination_slots,
                        &destination_slots,
                        &schedule,
                        &candidate_output,
                        F,
                        H,
                        E,
                        K,
                        N,
                        ROUTES,
                        ROUTES,
                    )
                })(),
            };
            encoder.end();
            result.unwrap();
            command.commit();
            crate::metal::wait_completed(&command).expect("command buffer completed");
            let wall_ms = started.elapsed().as_secs_f64() * 1e3;
            assert!(
                command.error().is_none(),
                "{schedule_name} {arm:?}: {:?}",
                command.error()
            );
            let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
            (gpu_ms, wall_ms)
        };

        let bits = |tensor: &MetalTensor, label: &str| {
            host_read_f32(tensor, label)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        let assert_exact = || {
            assert_eq!(
                bits(&candidate_arena, "fused all-IQ3 candidate output"),
                bits(&control_arena, "fused all-IQ3 control output"),
                "{schedule_name} output"
            );
            assert_eq!(
                bits(&candidate_inner, "fused all-IQ3 candidate inner"),
                bits(&control_inner, "fused all-IQ3 control inner"),
                "{schedule_name} inner"
            );
        };

        sample(Arm::Control);
        sample(Arm::Candidate);
        assert_exact();
        for index in 0usize..5 {
            if index.is_multiple_of(2) {
                sample(Arm::Control);
                sample(Arm::Candidate);
            } else {
                sample(Arm::Candidate);
                sample(Arm::Control);
            }
        }

        let collect = |arm| (0..SAMPLES).map(|_| sample(arm)).collect::<Vec<_>>();
        let control_before = collect(Arm::Control);
        let candidate = collect(Arm::Candidate);
        let control_after = collect(Arm::Control);
        assert_exact();

        let split = |samples: &[(f64, f64)]| {
            (
                samples.iter().map(|sample| sample.0).collect::<Vec<_>>(),
                samples.iter().map(|sample| sample.1).collect::<Vec<_>>(),
            )
        };
        let (control_before_gpu, control_before_wall) = split(&control_before);
        let (candidate_gpu, candidate_wall) = split(&candidate);
        let (control_after_gpu, control_after_wall) = split(&control_after);
        let gpu_control_drift = relative_drift(
            median_ms(&control_before_gpu),
            median_ms(&control_after_gpu),
        );
        let wall_control_drift = relative_drift(
            median_ms(&control_before_wall),
            median_ms(&control_after_wall),
        );
        let gpu_candidate_drift = relative_drift(
            median_ms(&candidate_gpu[..SAMPLES / 2]),
            median_ms(&candidate_gpu[SAMPLES / 2..]),
        );
        let wall_candidate_drift = relative_drift(
            median_ms(&candidate_wall[..SAMPLES / 2]),
            median_ms(&candidate_wall[SAMPLES / 2..]),
        );
        let gpu_control_median = median_ms(&control_before_gpu).min(median_ms(&control_after_gpu));
        let wall_control_median =
            median_ms(&control_before_wall).min(median_ms(&control_after_wall));
        let gpu_control_p95 =
            percentile_ms(&control_before_gpu, 0.95).min(percentile_ms(&control_after_gpu, 0.95));
        let wall_control_p95 =
            percentile_ms(&control_before_wall, 0.95).min(percentile_ms(&control_after_wall, 0.95));
        let gpu_median_saving = 1.0 - median_ms(&candidate_gpu) / gpu_control_median;
        let wall_median_saving = 1.0 - median_ms(&candidate_wall) / wall_control_median;
        let gpu_p95_saving = 1.0 - percentile_ms(&candidate_gpu, 0.95) / gpu_control_p95;
        let wall_p95_saving = 1.0 - percentile_ms(&candidate_wall, 0.95) / wall_control_p95;

        eprintln!(
            "deepseek_v4 fused_all_iq3_packet schedule={schedule_name} control_before_gpu_ms={control_before_gpu:?} control_before_wall_ms={control_before_wall:?} candidate_gpu_ms={candidate_gpu:?} candidate_wall_ms={candidate_wall:?} control_after_gpu_ms={control_after_gpu:?} control_after_wall_ms={control_after_wall:?} gpu_control_drift={gpu_control_drift:.6} wall_control_drift={wall_control_drift:.6} gpu_candidate_drift={gpu_candidate_drift:.6} wall_candidate_drift={wall_candidate_drift:.6} gpu_median_saving={gpu_median_saving:.6} wall_median_saving={wall_median_saving:.6} gpu_p95_saving={gpu_p95_saving:.6} wall_p95_saving={wall_p95_saving:.6}"
        );

        if schedule_shape == ScheduleShape::Sparse {
            let _trace = crate::metal::kernel_trace_begin();
            sample(Arm::Control);
            let control_trace = crate::metal::kernel_trace_take_delta();
            sample(Arm::Candidate);
            let candidate_trace = crate::metal::kernel_trace_take_delta();
            assert_eq!(control_trace.encoders, 1);
            assert_eq!(control_trace.concurrent_encoders, 0);
            assert_eq!(control_trace.dispatches, 4);
            assert_eq!(candidate_trace.encoders, 1);
            assert_eq!(candidate_trace.concurrent_encoders, 0);
            assert_eq!(candidate_trace.dispatches, 2);
        }
        for (label, tensor) in [
            ("fused all-IQ3 control arena", &control_arena),
            ("fused all-IQ3 control inner", &control_inner),
            ("fused all-IQ3 candidate arena", &candidate_arena),
            ("fused all-IQ3 candidate inner", &candidate_inner),
        ] {
            assert_grouped_guards(label, tensor);
        }
        results.push(GateResult {
            schedule: schedule_name,
            gpu_control_drift,
            wall_control_drift,
            gpu_candidate_drift,
            wall_candidate_drift,
            gpu_median_saving,
            wall_median_saving,
            gpu_p95_saving,
            wall_p95_saving,
        });
    }

    for result in results {
        assert!(
            result.gpu_control_drift <= 0.05,
            "{} GPU control drift {:.3} exceeded 5%",
            result.schedule,
            result.gpu_control_drift
        );
        assert!(
            result.wall_control_drift <= 0.05,
            "{} wall control drift {:.3} exceeded 5%",
            result.schedule,
            result.wall_control_drift
        );
        assert!(
            result.gpu_candidate_drift <= 0.05,
            "{} GPU candidate drift {:.3} exceeded 5%",
            result.schedule,
            result.gpu_candidate_drift
        );
        assert!(
            result.wall_candidate_drift <= 0.05,
            "{} wall candidate drift {:.3} exceeded 5%",
            result.schedule,
            result.wall_candidate_drift
        );
        assert!(
            result.gpu_median_saving >= 0.15,
            "{} GPU median saving {:.3} missed 15%",
            result.schedule,
            result.gpu_median_saving
        );
        assert!(
            result.wall_median_saving >= 0.10,
            "{} wall median saving {:.3} missed 10%",
            result.schedule,
            result.wall_median_saving
        );
        assert!(
            result.gpu_p95_saving >= 0.10,
            "{} GPU p95 saving {:.3} missed 10%",
            result.schedule,
            result.gpu_p95_saving
        );
        assert!(
            result.wall_p95_saving >= 0.10,
            "{} wall p95 saving {:.3} missed 10%",
            result.schedule,
            result.wall_p95_saving
        );
    }
}

struct PackedRouteGenerationOwner {
    next: Cell<u32>,
}

impl PackedRouteGenerationOwner {
    fn new() -> Self {
        Self { next: Cell::new(1) }
    }

    fn with_next(next: u32) -> Self {
        Self {
            next: Cell::new(next),
        }
    }

    fn take(&self) -> Result<NonZeroU32, DeepSeekV4MetalError> {
        let generation = NonZeroU32::new(self.next.get()).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed route generation owner reached zero".into())
        })?;
        let next = generation.get().checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "packed route generation owner exhausted before wrap".into(),
            )
        })?;
        self.next.set(next);
        Ok(generation)
    }
}

#[derive(Clone, Copy, Debug)]
enum PackedRouteMicroproofSource {
    Learned,
    Hash,
}

struct PackedRouteMicroproofScratch {
    logits: MetalTensor,
    token_ids: MetalTensor,
    expert_ids: MetalTensor,
    weights: MetalTensor,
    route_generations: MetalTensor,
    route_status: MetalTensor,
    counts: MetalTensor,
    slot_ids: MetalTensor,
    schedule_generations: MetalTensor,
    aggregate: MetalTensor,
    signature: MetalTensor,
    compact_header: MetalTensor,
    compact_rows: MetalTensor,
    compact_slots: MetalTensor,
    compact_tiles32: MetalTensor,
    compact_tiles16: MetalTensor,
}

struct PackedRouteCapture {
    generation: u32,
    expert_ids: Vec<i32>,
    weights: Vec<f32>,
    route_generations: Vec<i32>,
    route_status: Vec<i32>,
    counts: Vec<i32>,
    slot_ids: Vec<i32>,
    schedule_generations: Vec<i32>,
    aggregate: Vec<i32>,
    signature: Vec<i32>,
}

struct PackedRouteCompactCapture {
    header: Vec<i32>,
    expert_ids: Vec<i32>,
    counts: Vec<i32>,
    rows: Vec<i32>,
    slots: Vec<i32>,
    tiles32: Vec<i32>,
    tiles16: Vec<i32>,
}

impl PackedRouteMicroproofScratch {
    fn new(ctx: &MetalContext) -> Result<Self, DeepSeekV4MetalError> {
        let n = PACKED_GPU_ROUTE_MAX_TOKENS as u64;
        let slot_elements = PACKED_GPU_ROUTE_MAX_TOKENS * MOE_EXPERT_COUNT;
        let mut guarded_slots = vec![
            PACKED_ROUTE_SLOT_PREFIX;
            PACKED_ROUTE_SLOT_GUARD_BYTES
                + slot_elements * std::mem::size_of::<i32>()
                + PACKED_ROUTE_SLOT_GUARD_BYTES
        ];
        guarded_slots[PACKED_ROUTE_SLOT_GUARD_BYTES + slot_elements * std::mem::size_of::<i32>()..]
            .fill(PACKED_ROUTE_SLOT_SUFFIX);
        let slot_ids = MetalTensor {
            buffer: ctx.buffer_from(&guarded_slots)?,
            offset: PACKED_ROUTE_SLOT_GUARD_BYTES as u64,
            shape: vec![n, MOE_EXPERT_COUNT as u64],
            dtype: GgmlType::I32,
            provenance: MetalTensorProvenance::OwnedWritable,
        };
        Ok(Self {
            logits: MetalTensor::zeros_f32(ctx, vec![MOE_EXPERT_COUNT as u64, n])?,
            token_ids: MetalTensor::zeros_i32(ctx, vec![n])?,
            expert_ids: MetalTensor::zeros_i32(ctx, vec![MOE_TOP_K as u64, n])?,
            weights: MetalTensor::zeros_f32(ctx, vec![MOE_TOP_K as u64, n])?,
            route_generations: MetalTensor::zeros_i32(ctx, vec![n])?,
            route_status: MetalTensor::zeros_i32(ctx, vec![n])?,
            counts: MetalTensor::zeros_i32(ctx, vec![MOE_EXPERT_COUNT as u64])?,
            slot_ids,
            schedule_generations: MetalTensor::zeros_i32(ctx, vec![MOE_EXPERT_COUNT as u64])?,
            aggregate: MetalTensor::zeros_i32(ctx, vec![PACKED_ROUTE_AGGREGATE_WIDTH as u64])?,
            signature: MetalTensor::zeros_i32(ctx, vec![PACKED_ROUTE_AGGREGATE_WIDTH as u64])?,
            compact_header: MetalTensor::zeros_i32(
                ctx,
                vec![PACKED_COMPACT_ROUTE_HEADER_WIDTH as u64],
            )?,
            compact_rows: MetalTensor::zeros_i32(ctx, vec![n * MOE_TOP_K as u64])?,
            compact_slots: MetalTensor::zeros_i32(ctx, vec![n * MOE_TOP_K as u64])?,
            compact_tiles32: MetalTensor::zeros_i32(
                ctx,
                vec![PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64],
            )?,
            compact_tiles16: MetalTensor::zeros_i32(
                ctx,
                vec![PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS as u64],
            )?,
        })
    }

    fn assert_slot_guards(&self) {
        let payload_bytes = self.slot_ids.n_bytes() as usize;
        let base = self.slot_ids.buffer.contents().as_ptr().cast::<u8>();
        let prefix = unsafe {
            std::slice::from_raw_parts(
                base.add(self.slot_ids.offset as usize - PACKED_ROUTE_SLOT_GUARD_BYTES),
                PACKED_ROUTE_SLOT_GUARD_BYTES,
            )
        };
        let suffix = unsafe {
            std::slice::from_raw_parts(
                base.add(self.slot_ids.offset as usize + payload_bytes),
                PACKED_ROUTE_SLOT_GUARD_BYTES,
            )
        };
        assert!(prefix.iter().all(|&byte| byte == PACKED_ROUTE_SLOT_PREFIX));
        assert!(suffix.iter().all(|&byte| byte == PACKED_ROUTE_SLOT_SUFFIX));
    }

    fn buffers(&self) -> PackedGpuRouteBuffers<'_> {
        PackedGpuRouteBuffers {
            logits: &self.logits,
            token_ids: &self.token_ids,
            expert_ids: &self.expert_ids,
            weights: &self.weights,
            route_generations: &self.route_generations,
            route_status: &self.route_status,
            counts: &self.counts,
            slot_ids: &self.slot_ids,
            schedule_generations: &self.schedule_generations,
            aggregate: &self.aggregate,
            signature: &self.signature,
            compact_header: &self.compact_header,
        }
    }

    fn encode_learned(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        bias: &MetalTensor,
        n_tokens: usize,
        produced_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        self.buffers()
            .encode_learned(ctx, enc, bias, n_tokens, produced_tokens, generation, 1.5)
    }

    fn encode_hash(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        token_to_expert: &MetalTensor,
        n_tokens: usize,
        produced_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        self.buffers().encode_hash(
            ctx,
            enc,
            token_to_expert,
            n_tokens,
            produced_tokens,
            generation,
            1.5,
        )
    }

    fn encode_schedule(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        n_tokens: usize,
        produced_experts: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        self.buffers()
            .encode_schedule(ctx, enc, n_tokens, produced_experts, generation)
    }

    fn encode_validate(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        self.buffers()
            .encode_validate(ctx, enc, n_tokens, generation)
    }

    fn encode_signature(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        self.buffers()
            .encode_signature(ctx, enc, n_tokens, generation)
    }

    fn encode_compact(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let route_count = n_tokens * MOE_TOP_K;
        self.buffers().encode_compact(
            ctx,
            enc,
            &i32_prefix(
                &self.compact_rows,
                vec![route_count as u64],
                "packed compact proof rows",
            )?,
            &i32_prefix(
                &self.compact_slots,
                vec![route_count as u64],
                "packed compact proof slots",
            )?,
            &self.compact_tiles32,
            &self.compact_tiles16,
            n_tokens,
            generation,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_pipeline(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        source: PackedRouteMicroproofSource,
        bias: &MetalTensor,
        token_to_expert: &MetalTensor,
        n_tokens: usize,
        produced_tokens: usize,
        produced_experts: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        match source {
            PackedRouteMicroproofSource::Learned => {
                self.encode_learned(ctx, enc, bias, n_tokens, produced_tokens, generation)?
            }
            PackedRouteMicroproofSource::Hash => self.encode_hash(
                ctx,
                enc,
                token_to_expert,
                n_tokens,
                produced_tokens,
                generation,
            )?,
        }
        self.encode_schedule(ctx, enc, n_tokens, produced_experts, generation)?;
        self.encode_validate(ctx, enc, n_tokens, generation)?;
        self.encode_signature(ctx, enc, n_tokens, generation)
    }

    fn capture(&self, n_tokens: usize, generation: u32) -> PackedRouteCapture {
        let routes = n_tokens * MOE_TOP_K;
        let schedule = n_tokens * MOE_EXPERT_COUNT;
        let mut expert_ids = host_read_i32(&self.expert_ids, "packed route IDs").unwrap();
        let mut weights = host_read_f32(&self.weights, "packed route weights").unwrap();
        let mut route_generations =
            host_read_i32(&self.route_generations, "packed route generations").unwrap();
        let mut route_status = host_read_i32(&self.route_status, "packed route statuses").unwrap();
        let mut slot_ids = host_read_i32(&self.slot_ids, "packed route slot IDs").unwrap();
        expert_ids.truncate(routes);
        weights.truncate(routes);
        route_generations.truncate(n_tokens);
        route_status.truncate(n_tokens);
        slot_ids.truncate(schedule);
        PackedRouteCapture {
            generation,
            expert_ids,
            weights,
            route_generations,
            route_status,
            counts: host_read_i32(&self.counts, "packed route counts").unwrap(),
            slot_ids,
            schedule_generations: host_read_i32(
                &self.schedule_generations,
                "packed schedule generations",
            )
            .unwrap(),
            aggregate: host_read_i32(&self.aggregate, "packed route aggregate").unwrap(),
            signature: host_read_i32(&self.signature, "packed route signature").unwrap(),
        }
    }

    fn capture_compact(&self, n_tokens: usize) -> PackedRouteCompactCapture {
        let route_count = n_tokens * MOE_TOP_K;
        let mut expert_ids =
            host_read_i32(&self.expert_ids, "packed compact proof expert IDs").unwrap();
        let mut rows = host_read_i32(&self.compact_rows, "packed compact proof rows").unwrap();
        let mut slots = host_read_i32(&self.compact_slots, "packed compact proof slots").unwrap();
        rows.truncate(route_count);
        slots.truncate(route_count);
        expert_ids.truncate(route_count);
        PackedRouteCompactCapture {
            header: host_read_i32(&self.compact_header, "packed compact proof header").unwrap(),
            expert_ids,
            counts: host_read_i32(&self.counts, "packed compact proof counts").unwrap(),
            rows,
            slots,
            tiles32: host_read_i32(&self.compact_tiles32, "packed compact proof 32-row tiles")
                .unwrap(),
            tiles16: host_read_i32(&self.compact_tiles16, "packed compact proof 16-row tiles")
                .unwrap(),
        }
    }
}

struct PackedRouteFixture {
    scratch: PackedRouteMicroproofScratch,
    bias: MetalTensor,
    token_to_expert: MetalTensor,
    logits: Vec<f32>,
    bias_values: Vec<f32>,
    token_ids: Vec<i32>,
    hash_map: Vec<i32>,
    generations: PackedRouteGenerationOwner,
}

impl PackedRouteFixture {
    fn new(ctx: &MetalContext) -> Self {
        const VOCAB_SIZE: usize = PACKED_GPU_ROUTE_MAX_TOKENS + 1;
        let scratch = PackedRouteMicroproofScratch::new(ctx).unwrap();
        let logits = (0..PACKED_GPU_ROUTE_MAX_TOKENS)
            .flat_map(|token| {
                (0..MOE_EXPERT_COUNT).map(move |expert| {
                    if token.is_multiple_of(29) {
                        (token % 5) as f32 * 0.125 - 0.25
                    } else {
                        let mixed =
                            expert * 1_103 + token * 7_919 + (expert ^ token) * 53 + token / 3;
                        (mixed % 8_191) as f32 * 0.0025 - 10.0
                    }
                })
            })
            .collect::<Vec<_>>();
        let bias_values = (0..MOE_EXPERT_COUNT)
            .map(|expert| ((expert * 193 + 7) % 257) as f32 * 0.0002 - 0.0256)
            .collect::<Vec<_>>();
        let token_ids = (0..PACKED_GPU_ROUTE_MAX_TOKENS as i32).collect::<Vec<_>>();
        let hash_map = (0..VOCAB_SIZE)
            .flat_map(|token| {
                (0..MOE_TOP_K).map(move |slot| ((token * 17 + slot * 37) % 256) as i32)
            })
            .collect::<Vec<_>>();
        for row in hash_map.chunks_exact(MOE_TOP_K) {
            let mut sorted = row.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), MOE_TOP_K);
        }
        host_write_f32(&scratch.logits, &logits, "packed route fixture logits").unwrap();
        host_write_i32(
            &scratch.token_ids,
            &token_ids,
            "packed route fixture token IDs",
        )
        .unwrap();
        let bias = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&bias_values),
            vec![MOE_EXPERT_COUNT as u64],
            GgmlType::F32,
        )
        .unwrap();
        let token_to_expert = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&hash_map),
            vec![MOE_TOP_K as u64, VOCAB_SIZE as u64],
            GgmlType::I32,
        )
        .unwrap();
        Self {
            scratch,
            bias,
            token_to_expert,
            logits,
            bias_values,
            token_ids,
            hash_map,
            generations: PackedRouteGenerationOwner::new(),
        }
    }

    fn run(
        &self,
        ctx: &MetalContext,
        source: PackedRouteMicroproofSource,
        n_tokens: usize,
        produced_tokens: usize,
        produced_experts: usize,
    ) -> PackedRouteCapture {
        let generation = self.generations.take().unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        self.scratch
            .encode_pipeline(
                ctx,
                &encoder,
                source,
                &self.bias,
                &self.token_to_expert,
                n_tokens,
                produced_tokens,
                produced_experts,
                generation,
            )
            .unwrap();
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(
            command.error().is_none(),
            "packed route command failed: {:?}",
            command.error()
        );
        self.scratch.capture(n_tokens, generation.get())
    }

    fn run_compact(
        &self,
        ctx: &MetalContext,
        source: PackedRouteMicroproofSource,
        n_tokens: usize,
    ) -> (u32, PackedRouteCompactCapture) {
        let generation = self.generations.take().unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        match source {
            PackedRouteMicroproofSource::Learned => self
                .scratch
                .encode_learned(ctx, &encoder, &self.bias, n_tokens, n_tokens, generation)
                .unwrap(),
            PackedRouteMicroproofSource::Hash => self
                .scratch
                .encode_hash(
                    ctx,
                    &encoder,
                    &self.token_to_expert,
                    n_tokens,
                    n_tokens,
                    generation,
                )
                .unwrap(),
        }
        self.scratch
            .encode_compact(ctx, &encoder, n_tokens, generation)
            .unwrap();
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(
            command.error().is_none(),
            "packed compact route command failed: {:?}",
            command.error()
        );
        (generation.get(), self.scratch.capture_compact(n_tokens))
    }

    fn expected_routes(
        &self,
        source: PackedRouteMicroproofSource,
        n_tokens: usize,
    ) -> (Vec<i32>, Vec<f32>) {
        let mut ids = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut weights = Vec::with_capacity(n_tokens * MOE_TOP_K);
        for token in 0..n_tokens {
            let logits = &self.logits[token * MOE_EXPERT_COUNT..(token + 1) * MOE_EXPERT_COUNT];
            let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(logits).unwrap();
            let decision = match source {
                PackedRouteMicroproofSource::Learned => crate::deepseek_v4_oracle::learned_route(
                    &scores,
                    &self.bias_values,
                    MOE_TOP_K,
                    1.5,
                ),
                PackedRouteMicroproofSource::Hash => {
                    let token_id = self.token_ids[token] as usize;
                    let selected = self.hash_map[token_id * MOE_TOP_K..(token_id + 1) * MOE_TOP_K]
                        .iter()
                        .map(|&expert| expert as usize)
                        .collect::<Vec<_>>();
                    crate::deepseek_v4_oracle::hash_route(&scores, &selected, 1.5)
                }
            }
            .unwrap();
            ids.extend(decision.expert_ids.iter().map(|&expert| expert as i32));
            weights.extend_from_slice(&decision.weights);
        }
        (ids, weights)
    }
}

fn expected_packed_schedule(expert_ids: &[i32], n_tokens: usize) -> (Vec<i32>, Vec<i32>) {
    let mut counts = vec![0; MOE_EXPERT_COUNT];
    let mut slots = vec![-1; MOE_EXPERT_COUNT * n_tokens];
    for expert in 0..MOE_EXPERT_COUNT {
        let mut count = 0;
        for token in 0..n_tokens {
            for slot in 0..MOE_TOP_K {
                let global_slot = token * MOE_TOP_K + slot;
                if expert_ids[global_slot] == expert as i32 {
                    slots[expert * n_tokens + count] = global_slot as i32;
                    count += 1;
                }
            }
        }
        counts[expert] = count as i32;
    }
    (counts, slots)
}

fn expected_compact_schedule(
    expert_ids: &[i32],
    n_tokens: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<ExpertBucket>) {
    assert_eq!(expert_ids.len(), n_tokens * MOE_TOP_K);
    let mut counts = vec![0; MOE_EXPERT_COUNT];
    let mut rows = Vec::with_capacity(n_tokens * MOE_TOP_K);
    let mut slots = Vec::with_capacity(n_tokens * MOE_TOP_K);
    let mut buckets = Vec::new();
    for (expert, expert_count) in counts.iter_mut().enumerate() {
        let start = slots.len();
        for (slot, &routed_expert) in expert_ids.iter().enumerate() {
            if routed_expert == expert as i32 {
                rows.push((slot / MOE_TOP_K) as i32);
                slots.push(slot as i32);
            }
        }
        let count = slots.len() - start;
        *expert_count = count as i32;
        if count != 0 {
            buckets.push(ExpertBucket {
                expert,
                start,
                len: count,
            });
        }
    }
    (counts, rows, slots, buckets)
}

fn padded_tile_words(tiles: &[PackedGroupedExpertTile], descriptor_words: usize) -> Vec<i32> {
    let mut words = bytemuck::cast_slice::<PackedGroupedExpertTile, i32>(tiles).to_vec();
    words.resize(descriptor_words, 0);
    words
}

fn assert_packed_compact_capture(
    fixture: &PackedRouteFixture,
    source: PackedRouteMicroproofSource,
    n_tokens: usize,
    generation: u32,
    capture: &PackedRouteCompactCapture,
) {
    let (expected_ids, _) = fixture.expected_routes(source, n_tokens);
    for token in 0..n_tokens {
        let start = token * MOE_TOP_K;
        let mut actual = capture.expert_ids[start..start + MOE_TOP_K].to_vec();
        let mut expected = expected_ids[start..start + MOE_TOP_K].to_vec();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(
            actual, expected,
            "packed {source:?} route set at token {token}"
        );
    }
    let (counts, rows, slots, buckets) = expected_compact_schedule(&capture.expert_ids, n_tokens);
    let tiles32 = packed_grouped_expert_tiles(n_tokens, &buckets).unwrap();
    let tiles16 = packed_grouped_iq2_mma16_tiles(n_tokens, &buckets).unwrap();
    assert_eq!(capture.counts, counts);
    assert_eq!(capture.rows, rows);
    assert_eq!(capture.slots, slots);
    assert_eq!(
        capture.tiles32,
        padded_tile_words(&tiles32, PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS)
    );
    assert_eq!(
        capture.tiles16,
        padded_tile_words(&tiles16, PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS)
    );
    assert_eq!(
        capture.header,
        vec![
            generation as i32,
            PACKED_COMPACT_ROUTE_STATUS_READY,
            (n_tokens * MOE_TOP_K) as i32,
            buckets.len() as i32,
            tiles16.len() as i32,
            tiles32.len() as i32,
            packed_route_compact_completion(generation, n_tokens) as i32,
            n_tokens as i32,
        ]
    );
}

fn write_raw_f32(tensor: &MetalTensor, values: &[f32]) {
    assert_eq!(tensor.dtype, GgmlType::F32);
    assert!(tensor.is_writable());
    assert_eq!(tensor.n_elements() as usize, values.len());
    let destination = unsafe {
        tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<f32>()
    };
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), destination, values.len());
    }
}

fn assert_packed_route_capture(
    fixture: &PackedRouteFixture,
    source: PackedRouteMicroproofSource,
    n_tokens: usize,
    capture: &PackedRouteCapture,
) {
    let (expected_ids, expected_weights) = fixture.expected_routes(source, n_tokens);
    if let Some(index) = capture
        .expert_ids
        .iter()
        .zip(&expected_ids)
        .position(|(actual, expected)| actual != expected)
    {
        let token = index / MOE_TOP_K;
        let start = token * MOE_TOP_K;
        panic!(
            "packed {source:?} route ID differs at N={n_tokens} token {token}: actual={:?} expected={:?}",
            &capture.expert_ids[start..start + MOE_TOP_K],
            &expected_ids[start..start + MOE_TOP_K],
        );
    }
    for (index, (&actual, &expected)) in capture.weights.iter().zip(&expected_weights).enumerate() {
        let allowed = 1e-4 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= allowed,
            "packed route weight {index}: {actual} vs {expected}, allowed {allowed}"
        );
    }
    let (expected_counts, expected_slots) = expected_packed_schedule(&expected_ids, n_tokens);
    assert_eq!(capture.counts, expected_counts);
    assert_eq!(capture.slot_ids, expected_slots);
    assert_eq!(
        capture.route_generations,
        vec![capture.generation as i32; n_tokens]
    );
    assert_eq!(
        capture.route_status,
        vec![DEEPSEEK_V4_ROUTE_STATUS_READY; n_tokens]
    );
    assert_eq!(
        capture.schedule_generations,
        vec![capture.generation as i32; MOE_EXPERT_COUNT]
    );
    assert_eq!(
        capture.aggregate,
        vec![
            capture.generation as i32,
            DEEPSEEK_V4_ROUTE_STATUS_READY,
            (n_tokens * MOE_TOP_K) as i32,
            packed_route_completion(capture.generation, n_tokens) as i32,
        ]
    );
    assert_eq!(capture.signature[0], capture.generation as i32);
    assert_eq!(capture.signature[1], DEEPSEEK_V4_ROUTE_STATUS_READY);
    assert_eq!(
        capture.signature[2] as u32,
        packed_route_signature_hash(
            &capture.expert_ids,
            &capture.weights,
            &capture.counts,
            &capture.slot_ids,
            n_tokens,
        )
        .unwrap()
    );
    assert_eq!(
        capture.signature[3],
        packed_route_signature_completion(capture.generation, n_tokens) as i32
    );
}

fn assert_packed_route_set_capture(
    fixture: &PackedRouteFixture,
    source: PackedRouteMicroproofSource,
    n_tokens: usize,
    capture: &PackedRouteCapture,
) {
    let (expected_ids, expected_weights) = fixture.expected_routes(source, n_tokens);
    for token in 0..n_tokens {
        let start = token * MOE_TOP_K;
        let actual_ids = &capture.expert_ids[start..start + MOE_TOP_K];
        let expected_ids = &expected_ids[start..start + MOE_TOP_K];
        let mut actual_set = actual_ids.to_vec();
        let mut expected_set = expected_ids.to_vec();
        actual_set.sort_unstable();
        expected_set.sort_unstable();
        assert_eq!(
            actual_set, expected_set,
            "packed {source:?} route set at N={n_tokens} token {token}"
        );
        for (slot, &expert) in actual_ids.iter().enumerate() {
            let expected_slot = expected_ids
                .iter()
                .position(|&expected| expected == expert)
                .unwrap();
            let actual = capture.weights[start + slot];
            let expected = expected_weights[start + expected_slot];
            let allowed = 1e-4 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= allowed,
                "packed route weight for expert {expert}: {actual} vs {expected}, allowed {allowed}"
            );
        }
    }
    let (expected_counts, expected_slots) = expected_packed_schedule(&capture.expert_ids, n_tokens);
    assert_eq!(capture.counts, expected_counts);
    assert_eq!(capture.slot_ids, expected_slots);
    assert_eq!(
        capture.route_generations,
        vec![capture.generation as i32; n_tokens]
    );
    assert_eq!(
        capture.route_status,
        vec![DEEPSEEK_V4_ROUTE_STATUS_READY; n_tokens]
    );
    assert_eq!(
        capture.schedule_generations,
        vec![capture.generation as i32; MOE_EXPERT_COUNT]
    );
    assert_eq!(
        capture.aggregate,
        vec![
            capture.generation as i32,
            DEEPSEEK_V4_ROUTE_STATUS_READY,
            (n_tokens * MOE_TOP_K) as i32,
            packed_route_completion(capture.generation, n_tokens) as i32,
        ]
    );
    assert_eq!(
        capture.signature,
        vec![
            capture.generation as i32,
            DEEPSEEK_V4_ROUTE_STATUS_READY,
            packed_route_signature_hash(
                &capture.expert_ids,
                &capture.weights,
                &capture.counts,
                &capture.slot_ids,
                n_tokens,
            )
            .unwrap() as i32,
            packed_route_signature_completion(capture.generation, n_tokens) as i32,
        ]
    );
}

fn assert_packed_route_failure(
    capture: &PackedRouteCapture,
    n_tokens: usize,
    status: i32,
    total: i32,
) {
    assert_eq!(
        capture.aggregate,
        vec![
            capture.generation as i32,
            status,
            total,
            packed_route_completion(capture.generation, n_tokens) as i32,
        ]
    );
    assert_eq!(
        capture.signature,
        vec![
            capture.generation as i32,
            PACKED_ROUTE_INVALID_AGGREGATE,
            0,
            packed_route_signature_completion(capture.generation, n_tokens) as i32,
        ]
    );
}

#[test]
fn packed_gpu_routes_and_schedules_match_cpu_at_representative_widths() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let fixture = PackedRouteFixture::new(&ctx);
    for n_tokens in [1, 2, 7, 16, 31, 32, 127, 128, 129, 511, 512, 2_047, 2_048] {
        for source in [
            PackedRouteMicroproofSource::Learned,
            PackedRouteMicroproofSource::Hash,
        ] {
            let capture = fixture.run(&ctx, source, n_tokens, n_tokens, MOE_EXPERT_COUNT);
            if n_tokens <= 128 || matches!(source, PackedRouteMicroproofSource::Hash) {
                assert_packed_route_capture(&fixture, source, n_tokens, &capture);
            } else {
                assert_packed_route_set_capture(&fixture, source, n_tokens, &capture);
            }
        }
    }
}

#[test]
fn packed_gpu_route_compaction_matches_cpu_at_capacity_boundaries() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let fixture = PackedRouteFixture::new(&ctx);
    for n_tokens in [1, 127, 128, 129, 2_047, 2_048] {
        for source in [
            PackedRouteMicroproofSource::Learned,
            PackedRouteMicroproofSource::Hash,
        ] {
            let (generation, capture) = fixture.run_compact(&ctx, source, n_tokens);
            assert_packed_compact_capture(&fixture, source, n_tokens, generation, &capture);
        }
    }
}

#[test]
fn packed_duplicate_hash_routes_flow_through_grouped_experts_and_sum() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const H: usize = 256;
    const F: usize = 256;
    const E: usize = MOE_EXPERT_COUNT;
    const N: usize = 1;
    const EXPERT: usize = 7;
    const CLAMP: f32 = 0.25;
    let fixture = PackedRouteFixture::new(&ctx);
    let duplicate_map_values = vec![EXPERT as i32; MOE_TOP_K];
    let duplicate_map = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&duplicate_map_values),
        vec![MOE_TOP_K as u64, 1],
        GgmlType::I32,
    )
    .unwrap();
    let generation = fixture.generations.take().unwrap();
    let route_command = ctx.queue.commandBuffer().unwrap();
    let route_encoder = KernelEncoder::begin(&route_command);
    fixture
        .scratch
        .encode_hash(&ctx, &route_encoder, &duplicate_map, N, N, generation)
        .unwrap();
    fixture
        .scratch
        .encode_compact(&ctx, &route_encoder, N, generation)
        .unwrap();
    route_encoder.end();
    route_command.commit();
    crate::metal::wait_completed(&route_command).expect("command buffer completed");
    assert!(route_command.error().is_none());

    let route_count = N * MOE_TOP_K;
    let compact = fixture.scratch.capture_compact(N);
    assert_eq!(
        compact.header,
        vec![
            generation.get() as i32,
            PACKED_COMPACT_ROUTE_STATUS_READY,
            route_count as i32,
            1,
            1,
            1,
            packed_route_compact_completion(generation.get(), N) as i32,
            N as i32,
        ]
    );
    assert_eq!(compact.expert_ids, duplicate_map_values);
    assert_eq!(compact.rows, vec![0; route_count]);
    assert_eq!(compact.slots, (0..route_count as i32).collect::<Vec<_>>());
    assert_eq!(compact.tiles16[..3], [EXPERT as i32, 0, route_count as i32]);
    assert_eq!(compact.tiles32[..3], [EXPERT as i32, 0, route_count as i32]);
    let rows = i32_prefix(
        &fixture.scratch.compact_rows,
        vec![route_count as u64],
        "duplicate grouped source rows",
    )
    .unwrap();
    let slots = i32_prefix(
        &fixture.scratch.compact_slots,
        vec![route_count as u64],
        "duplicate grouped destination slots",
    )
    .unwrap();
    let tiles16 = i32_prefix(
        &fixture.scratch.compact_tiles16,
        vec![3],
        "duplicate grouped 16-row tile",
    )
    .unwrap();
    let tiles32 = i32_prefix(
        &fixture.scratch.compact_tiles32,
        vec![3],
        "duplicate grouped 32-row tile",
    )
    .unwrap();
    let plan16 = PackedGroupedExpertPlan::from_device(&tiles16, 1).unwrap();
    let plan32 = PackedGroupedExpertPlan::from_device(&tiles32, 1).unwrap();
    let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 211);
    let up_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 223);
    let down_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, F, H, E, 227);
    let input_values = (0..H)
        .map(|index| ((index * 41 + 17) % 251) as f32 * 0.001 - 0.125)
        .collect::<Vec<_>>();
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&input_values),
        vec![H as u64, N as u64],
        GgmlType::F32,
    )
    .unwrap();
    let gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], f32::NAN);
    let up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], f32::NAN);
    let inner = grouped_guarded_f32(&ctx, vec![F as u64, MOE_TOP_K as u64, N as u64], f32::NAN);
    let expert_outputs =
        grouped_guarded_f32(&ctx, vec![H as u64, MOE_TOP_K as u64, N as u64], f32::NAN);
    let routed_output = grouped_guarded_f32(&ctx, vec![H as u64, N as u64], f32::NAN);
    let weights = f32_prefix(
        &fixture.scratch.weights,
        vec![MOE_TOP_K as u64, N as u64],
        "duplicate grouped route weights",
    )
    .unwrap();

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_packed_grouped_mapped_iq2_xs_swiglu_f32_matrix(
        &ctx,
        &encoder,
        &gate_bank,
        &up_bank,
        &input,
        &rows,
        &slots,
        &plan16,
        &gate,
        &up,
        &inner,
        H,
        F,
        E,
        MOE_TOP_K,
        N,
        N,
        route_count,
        CLAMP,
        PackedIq2MatrixWorkUnit::Mma16,
    )
    .unwrap();
    encode_packed_grouped_down_iq3_xxs_f32(
        &ctx,
        &encoder,
        &down_bank,
        &inner,
        &slots,
        &plan32,
        &expert_outputs,
        F,
        H,
        E,
        MOE_TOP_K,
        N,
    )
    .unwrap();
    crate::metal::encode_moe_weighted_sum_packed_f32(
        &ctx,
        &encoder,
        &expert_outputs,
        &weights,
        &routed_output,
        H,
        MOE_TOP_K,
        N,
    )
    .unwrap();
    encoder.end();
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert!(command.error().is_none(), "{:?}", command.error());

    let gate_values = host_read_f32(&gate, "duplicate grouped gate").unwrap();
    let up_values = host_read_f32(&up, "duplicate grouped up").unwrap();
    let inner_values = host_read_f32(&inner, "duplicate grouped inner").unwrap();
    let expert_values = host_read_f32(&expert_outputs, "duplicate grouped expert outputs").unwrap();
    let weight_values = host_read_f32(&weights, "duplicate grouped weights").unwrap();
    let routed_values = host_read_f32(&routed_output, "duplicate grouped routed output").unwrap();
    for (label, values) in [
        ("gate", &gate_values),
        ("up", &up_values),
        ("inner", &inner_values),
        ("expert output", &expert_values),
        ("routed output", &routed_values),
    ] {
        assert!(
            values.iter().all(|value| value.is_finite()),
            "duplicate grouped {label} retained an unwritten value"
        );
    }
    assert!(
        weight_values
            .iter()
            .all(|weight| weight.is_finite() && *weight > 0.0)
    );
    assert!((weight_values.iter().sum::<f32>() - 1.5).abs() <= 1e-5);
    for slot in 1..MOE_TOP_K {
        assert_eq!(
            expert_values[slot * H..(slot + 1) * H]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expert_values[..H]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
    for column in 0..H {
        let expected = (0..MOE_TOP_K).fold(0.0f32, |sum, slot| {
            sum + weight_values[slot] * expert_values[slot * H + column]
        });
        let allowed = 1e-5 * expected.abs().max(1.0);
        assert!((routed_values[column] - expected).abs() <= allowed);
    }
    for (label, tensor) in [
        ("duplicate grouped gate", &gate),
        ("duplicate grouped up", &up),
        ("duplicate grouped inner", &inner),
        ("duplicate grouped expert outputs", &expert_outputs),
        ("duplicate grouped routed output", &routed_output),
    ] {
        assert_grouped_guards(label, tensor);
    }
}

#[test]
fn packed_gpu_route_compaction_handles_maximum_concentrated_count() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const EXPERT: usize = 7;
    let fixture = PackedRouteFixture::new(&ctx);
    let n_tokens = PACKED_GPU_ROUTE_MAX_TOKENS;
    let route_count = n_tokens * MOE_TOP_K;
    let duplicate_map_values = vec![EXPERT as i32; route_count];
    let duplicate_map = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&duplicate_map_values),
        vec![MOE_TOP_K as u64, n_tokens as u64],
        GgmlType::I32,
    )
    .unwrap();
    let generation = fixture.generations.take().unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    fixture
        .scratch
        .encode_hash(
            &ctx,
            &encoder,
            &duplicate_map,
            n_tokens,
            n_tokens,
            generation,
        )
        .unwrap();
    fixture
        .scratch
        .encode_compact(&ctx, &encoder, n_tokens, generation)
        .unwrap();
    encoder.end();
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert!(command.error().is_none());
    let capture = fixture.scratch.capture_compact(n_tokens);
    let schedule = vec![ExpertBucket {
        expert: EXPERT,
        start: 0,
        len: route_count,
    }];
    let tiles32 = packed_grouped_expert_tiles(n_tokens, &schedule).unwrap();
    let tiles16 = packed_grouped_iq2_mma16_tiles(n_tokens, &schedule).unwrap();
    assert_eq!(capture.header[1], PACKED_COMPACT_ROUTE_STATUS_READY);
    assert_eq!(capture.header[2], route_count as i32);
    assert_eq!(capture.header[3], 1);
    assert_eq!(capture.header[4], tiles16.len() as i32);
    assert_eq!(capture.header[5], tiles32.len() as i32);
    assert_eq!(capture.counts[EXPERT], route_count as i32);
    assert!(
        capture
            .counts
            .iter()
            .enumerate()
            .all(|(expert, &count)| { expert == EXPERT || count == 0 })
    );
    assert_eq!(capture.rows[0], 0);
    assert_eq!(capture.rows[route_count - 1], (n_tokens - 1) as i32);
    assert_eq!(capture.slots, (0..route_count as i32).collect::<Vec<_>>());
    assert_eq!(
        capture.tiles32,
        padded_tile_words(&tiles32, PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS)
    );
    assert_eq!(
        capture.tiles16,
        padded_tile_words(&tiles16, PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS)
    );
}

#[test]
fn packed_gpu_route_compaction_rejects_invalid_late_tokens() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let fixture = PackedRouteFixture::new(&ctx);
    let n_tokens = 512;
    let (valid_generation, valid_capture) =
        fixture.run_compact(&ctx, PackedRouteMicroproofSource::Learned, n_tokens);
    assert_packed_compact_capture(
        &fixture,
        PackedRouteMicroproofSource::Learned,
        n_tokens,
        valid_generation,
        &valid_capture,
    );
    let (valid_ids, valid_weights) =
        fixture.expected_routes(PackedRouteMicroproofSource::Learned, n_tokens);
    let run = |ids: &[i32], weights: &[f32], generations: &[i32], statuses: &[i32]| {
        let generation = fixture.generations.take().unwrap();
        let mut full_ids = vec![-1; PACKED_GPU_ROUTE_MAX_TOKENS * MOE_TOP_K];
        full_ids[..ids.len()].copy_from_slice(ids);
        host_write_i32(
            &fixture.scratch.expert_ids,
            &full_ids,
            "compact invalid IDs",
        )
        .unwrap();
        let mut full_weights = vec![0.0; PACKED_GPU_ROUTE_MAX_TOKENS * MOE_TOP_K];
        full_weights[..weights.len()].copy_from_slice(weights);
        write_raw_f32(&fixture.scratch.weights, &full_weights);
        let mut full_generations = vec![0; PACKED_GPU_ROUTE_MAX_TOKENS];
        full_generations[..generations.len()].copy_from_slice(generations);
        host_write_i32(
            &fixture.scratch.route_generations,
            &full_generations,
            "compact invalid generations",
        )
        .unwrap();
        let mut full_statuses = vec![0; PACKED_GPU_ROUTE_MAX_TOKENS];
        full_statuses[..statuses.len()].copy_from_slice(statuses);
        host_write_i32(
            &fixture.scratch.route_status,
            &full_statuses,
            "compact invalid statuses",
        )
        .unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        fixture
            .scratch
            .encode_compact(&ctx, &encoder, n_tokens, generation)
            .unwrap();
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(command.error().is_none());
        let header =
            host_read_i32(&fixture.scratch.compact_header, "compact invalid header").unwrap();
        assert_eq!(header[0], generation.get() as i32);
        assert_eq!(header[1], PACKED_COMPACT_ROUTE_STATUS_INVALID_ROUTE);
        assert_eq!(
            header[6],
            packed_route_compact_completion(generation.get(), n_tokens) as i32
        );
        assert_eq!(header[7], n_tokens as i32);
    };

    let ready = vec![DEEPSEEK_V4_ROUTE_STATUS_READY; n_tokens];
    let next_generation = fixture.generations.next.get() as i32;
    let valid_generations = vec![next_generation; n_tokens];

    let mut stale_generations = valid_generations.clone();
    stale_generations[256] = 0;
    run(&valid_ids, &valid_weights, &stale_generations, &ready);

    let next_generation = fixture.generations.next.get() as i32;
    let valid_generations = vec![next_generation; n_tokens];
    let mut failed_status = ready.clone();
    failed_status[300] = DEEPSEEK_V4_ROUTE_STATUS_INVALID_TOKEN;
    run(
        &valid_ids,
        &valid_weights,
        &valid_generations,
        &failed_status,
    );

    let next_generation = fixture.generations.next.get() as i32;
    let valid_generations = vec![next_generation; n_tokens];
    let mut invalid_ids = valid_ids.clone();
    invalid_ids[400 * MOE_TOP_K + 1] = MOE_EXPERT_COUNT as i32;
    run(&invalid_ids, &valid_weights, &valid_generations, &ready);

    let next_generation = fixture.generations.next.get() as i32;
    let valid_generations = vec![next_generation; n_tokens];
    let mut nonfinite_weights = valid_weights;
    nonfinite_weights[511 * MOE_TOP_K + 5] = f32::NAN;
    run(&valid_ids, &nonfinite_weights, &valid_generations, &ready);
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn packed_gpu_route_policy_rejects_unqualified_widths() {
    validate_packed_route_policy_scope(
        PackedRoutePolicy::GpuExperimental,
        PACKED_GPU_ROUTE_MAX_TOKENS,
    )
    .unwrap();
    assert!(
        validate_packed_route_policy_scope(
            PackedRoutePolicy::GpuExperimental,
            PACKED_GPU_ROUTE_MAX_TOKENS + 1,
        )
        .is_err()
    );
    validate_packed_route_policy_scope(PackedRoutePolicy::Cpu, DEEPSEEK_V4_PREFILL_MAX_TOKENS)
        .unwrap();
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn packed_diagnostic_hash_route_detects_duplicate_slots_before_mutation() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let unique = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        vec![MOE_TOP_K as u64, 2],
        GgmlType::I32,
    )
    .unwrap();
    let duplicate = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[0i32, 1, 2, 3, 4, 5, 6, 6, 8, 9, 10, 11]),
        vec![MOE_TOP_K as u64, 2],
        GgmlType::I32,
    )
    .unwrap();
    assert!(!packed_hash_route_has_duplicate_slots(&[0, 1], &unique, MOE_EXPERT_COUNT,).unwrap());
    assert!(packed_hash_route_has_duplicate_slots(&[0, 1], &duplicate, MOE_EXPERT_COUNT,).unwrap());
    assert_eq!(
        packed_diagnostic_route_policy(true, false, true).unwrap(),
        PackedRoutePolicy::CpuNoCompactPromotion
    );
    assert_eq!(
        packed_diagnostic_route_policy(true, true, true).unwrap(),
        PackedRoutePolicy::CpuNoCompactPromotion
    );
    assert_eq!(
        packed_diagnostic_route_policy(true, false, false).unwrap(),
        PackedRoutePolicy::GpuExperimental
    );
    assert_eq!(
        packed_diagnostic_route_policy(true, true, false).unwrap(),
        PackedRoutePolicy::GpuExperimentalCpuWeights
    );
    assert!(packed_diagnostic_route_policy(false, true, false).is_err());
}

#[test]
fn packed_gpu_routes_are_bitwise_singleton_equivalent_on_adversarial_scores() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let packed = PackedRouteMicroproofScratch::new(&ctx).unwrap();
    let singleton = DeepSeekV4MoeScratch::new(
        &ctx,
        DeepSeekV4MoeConfig {
            hidden_size: 1,
            ffn_size: 1,
            expert_count: MOE_EXPERT_COUNT,
            top_k: MOE_TOP_K,
            routed_scale: 1.5,
        },
    )
    .unwrap();
    let generations = PackedRouteGenerationOwner::new();
    let branch_values = [
        f32::from_bits((-20.0f32).to_bits() + 1),
        -20.0,
        f32::from_bits((-20.0f32).to_bits() - 1),
        f32::from_bits(20.0f32.to_bits() - 1),
        20.0,
        f32::from_bits(20.0f32.to_bits() + 1),
        -f32::MAX,
        f32::MAX,
    ];
    let mut cutoff_bias = vec![-1.0; MOE_EXPERT_COUNT];
    cutoff_bias[..7].fill(0.25);
    cutoff_bias[6] = f32::from_bits(0.25f32.to_bits() - 1);
    let cases = [
        (vec![0.0; MOE_EXPERT_COUNT], vec![0.0; MOE_EXPERT_COUNT]),
        (vec![0.0; MOE_EXPERT_COUNT], cutoff_bias),
        (
            (0..MOE_EXPERT_COUNT)
                .map(|expert| branch_values[expert % branch_values.len()])
                .collect(),
            (0..MOE_EXPERT_COUNT)
                .map(|expert| (expert % 11) as f32 * 0.0001 - 0.0005)
                .collect(),
        ),
    ];
    let hash_values = (0..MOE_TOP_K as i32).collect::<Vec<_>>();
    let hash_map = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&hash_values),
        vec![MOE_TOP_K as u64, 1],
        GgmlType::I32,
    )
    .unwrap();
    host_write_i32(
        &packed.token_ids,
        &vec![0; PACKED_GPU_ROUTE_MAX_TOKENS],
        "singleton-equivalent packed token IDs",
    )
    .unwrap();

    for (case, (logits, bias_values)) in cases.into_iter().enumerate() {
        host_write_f32(
            &singleton.logits,
            &logits,
            "singleton-equivalent singleton logits",
        )
        .unwrap();
        let mut packed_logits = vec![0.0; MOE_EXPERT_COUNT * PACKED_GPU_ROUTE_MAX_TOKENS];
        packed_logits[..MOE_EXPERT_COUNT].copy_from_slice(&logits);
        host_write_f32(
            &packed.logits,
            &packed_logits,
            "singleton-equivalent packed logits",
        )
        .unwrap();
        let bias = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&bias_values),
            vec![MOE_EXPERT_COUNT as u64],
            GgmlType::F32,
        )
        .unwrap();

        for source in [
            PackedRouteMicroproofSource::Learned,
            PackedRouteMicroproofSource::Hash,
        ] {
            let generation = generations.take().unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            match source {
                PackedRouteMicroproofSource::Learned => {
                    singleton
                        .encode_route_learned_gpu(&ctx, &encoder, &bias)
                        .unwrap();
                    packed
                        .encode_learned(&ctx, &encoder, &bias, 1, 1, generation)
                        .unwrap();
                }
                PackedRouteMicroproofSource::Hash => {
                    singleton
                        .encode_route_hash_gpu(&ctx, &encoder, 0, &hash_map)
                        .unwrap();
                    packed
                        .encode_hash(&ctx, &encoder, &hash_map, 1, 1, generation)
                        .unwrap();
                }
            }
            encoder.end();
            command.commit();
            crate::metal::wait_completed(&command).expect("command buffer completed");
            assert!(command.error().is_none(), "route case {case} failed");
            let expected = singleton.capture_gpu_route_record().unwrap();
            let mut actual_ids =
                host_read_i32(&packed.expert_ids, "singleton-equivalent packed IDs").unwrap();
            let mut actual_weights =
                host_read_f32(&packed.weights, "singleton-equivalent packed weights").unwrap();
            let actual_status =
                host_read_i32(&packed.route_status, "singleton-equivalent packed status").unwrap();
            actual_ids.truncate(MOE_TOP_K);
            actual_weights.truncate(MOE_TOP_K);
            assert_eq!(actual_status[0], expected.status, "case {case}");
            assert_eq!(actual_ids, expected.expert_ids, "case {case}");
            assert_eq!(
                actual_weights
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .weights
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "case {case}"
            );
        }
    }
}

#[test]
fn packed_route_signature_binds_each_payload_class() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let fixture = PackedRouteFixture::new(&ctx);
    let n_tokens = 12;
    let capture = fixture.run(
        &ctx,
        PackedRouteMicroproofSource::Learned,
        n_tokens,
        n_tokens,
        MOE_EXPERT_COUNT,
    );
    assert_packed_route_capture(
        &fixture,
        PackedRouteMicroproofSource::Learned,
        n_tokens,
        &capture,
    );
    let hash = |ids: &[i32], weights: &[f32], counts: &[i32], slots: &[i32]| {
        packed_route_signature_hash(ids, weights, counts, slots, n_tokens).unwrap()
    };
    let baseline = hash(
        &capture.expert_ids,
        &capture.weights,
        &capture.counts,
        &capture.slot_ids,
    );

    let mut ids = capture.expert_ids.clone();
    ids[0] ^= 1;
    assert_ne!(
        hash(&ids, &capture.weights, &capture.counts, &capture.slot_ids),
        baseline
    );
    let mut weights = capture.weights.clone();
    weights[0] = f32::from_bits(weights[0].to_bits() ^ 1);
    assert_ne!(
        hash(
            &capture.expert_ids,
            &weights,
            &capture.counts,
            &capture.slot_ids,
        ),
        baseline
    );
    let mut counts = capture.counts.clone();
    counts[0] ^= 1;
    assert_ne!(
        hash(
            &capture.expert_ids,
            &capture.weights,
            &counts,
            &capture.slot_ids,
        ),
        baseline
    );
    let mut slots = capture.slot_ids.clone();
    let occupied = slots.iter().position(|&slot| slot >= 0).unwrap();
    slots[occupied] ^= 1;
    assert_ne!(
        hash(
            &capture.expert_ids,
            &capture.weights,
            &capture.counts,
            &slots,
        ),
        baseline
    );
    let mut padding = capture.slot_ids.clone();
    let sentinel = padding.iter().position(|&slot| slot == -1).unwrap();
    padding[sentinel] = -2;
    assert_ne!(
        hash(
            &capture.expert_ids,
            &capture.weights,
            &capture.counts,
            &padding,
        ),
        baseline
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn production_packed_gpu_route_owns_and_compacts_the_qualified_schedule() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let fixture = PackedRouteFixture::new(&ctx);
    let production = DeepSeekV4PrefillScratch::new(
        &ctx,
        DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
        MOE_EXPERT_COUNT,
    )
    .unwrap();
    let production_logits = f32_prefix(
        &production.moe.logits,
        vec![MOE_EXPERT_COUNT as u64, PACKED_GPU_ROUTE_MAX_TOKENS as u64],
        "production packed GPU route logits fixture",
    )
    .unwrap();
    host_write_f32(
        &production_logits,
        &fixture.logits,
        "production packed GPU route logits",
    )
    .unwrap();
    let production_token_ids = i32_prefix(
        &production.token_ids,
        vec![PACKED_GPU_ROUTE_MAX_TOKENS as u64],
        "production packed GPU route token fixture",
    )
    .unwrap();
    host_write_i32(
        &production_token_ids,
        &fixture.token_ids,
        "production packed GPU route token IDs",
    )
    .unwrap();

    for (source_kind, n_tokens) in [
        (PackedRouteMicroproofSource::Learned, 12),
        (PackedRouteMicroproofSource::Hash, 128),
    ] {
        let logits = f32_prefix(
            &production.moe.logits,
            vec![MOE_EXPERT_COUNT as u64, n_tokens as u64],
            "production packed GPU route logits view",
        )
        .unwrap();
        let normalized_input = f32_prefix(
            &production.moe.normalized_input,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "production packed GPU route normalized view",
        )
        .unwrap();
        let token_ids = i32_prefix(
            &production.token_ids,
            vec![n_tokens as u64],
            "production packed GPU route token view",
        )
        .unwrap();
        let views = PackedMoeViews {
            normalized_input,
            logits,
            hash_ids: None,
        };
        let generation = production.moe.take_gpu_route_generation().unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let source = match source_kind {
            PackedRouteMicroproofSource::Learned => PackedRouteSource::Learned(&fixture.bias),
            PackedRouteMicroproofSource::Hash => PackedRouteSource::Hash,
        };
        production
            .moe
            .encode_gpu_route_compact(
                &ctx,
                &encoder,
                &views,
                source,
                &token_ids,
                (matches!(source_kind, PackedRouteMicroproofSource::Hash))
                    .then_some(&fixture.token_to_expert),
                n_tokens,
                1.5,
                generation,
            )
            .unwrap();
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(command.error().is_none());
        let schedule = production
            .moe
            .capture_gpu_compact_schedule(n_tokens, generation)
            .unwrap();

        let (expected_ids, expected_weights) = fixture.expected_routes(source_kind, n_tokens);
        let (_, compact_rows, compact_slots, expected_buckets) =
            expected_compact_schedule(&expected_ids, n_tokens);
        let mut actual_ids =
            host_read_i32(&production.moe.expert_ids, "production packed route IDs").unwrap();
        let mut actual_weights =
            host_read_f32(&production.moe.weights, "production packed route weights").unwrap();
        actual_ids.truncate(n_tokens * MOE_TOP_K);
        actual_weights.truncate(n_tokens * MOE_TOP_K);
        assert_eq!(actual_ids, expected_ids);
        for (&actual, &expected) in actual_weights.iter().zip(&expected_weights) {
            assert!((actual - expected).abs() <= 1e-4 * expected.abs().max(1.0));
        }

        assert_eq!(schedule.len(), expected_buckets.len());
        for (actual, expected) in schedule.iter().zip(&expected_buckets) {
            assert_eq!(
                (actual.expert, actual.start, actual.len),
                (expected.expert, expected.start, expected.len)
            );
        }
        let mut actual_rows = host_read_i32(
            &production.moe.bucket_rows,
            "production packed compact rows",
        )
        .unwrap();
        let mut actual_slots = host_read_i32(
            &production.moe.bucket_slots,
            "production packed compact slots",
        )
        .unwrap();
        actual_rows.truncate(n_tokens * MOE_TOP_K);
        actual_slots.truncate(n_tokens * MOE_TOP_K);
        assert_eq!(actual_rows, compact_rows);
        assert_eq!(actual_slots, compact_slots);
        let expected_tiles32 = packed_grouped_expert_tiles(n_tokens, &expected_buckets).unwrap();
        let expected_tiles16 = packed_grouped_iq2_mma16_tiles(n_tokens, &expected_buckets).unwrap();
        assert_eq!(
            host_read_i32(
                &production.moe.grouped_tiles,
                "production packed compact 32-row tiles",
            )
            .unwrap(),
            padded_tile_words(&expected_tiles32, PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS,)
        );
        assert_eq!(
            host_read_i32(
                &production.moe.grouped_iq2_mma16_tiles,
                "production packed compact 16-row tiles",
            )
            .unwrap(),
            padded_tile_words(&expected_tiles16, PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS,)
        );
    }

    let n_tokens = 1;
    let duplicate_expert = 7usize;
    let duplicate_map_values = vec![duplicate_expert as i32; MOE_TOP_K];
    let duplicate_map = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&duplicate_map_values),
        vec![MOE_TOP_K as u64, 1],
        GgmlType::I32,
    )
    .unwrap();
    let logits = f32_prefix(
        &production.moe.logits,
        vec![MOE_EXPERT_COUNT as u64, n_tokens as u64],
        "duplicate packed GPU route logits",
    )
    .unwrap();
    let normalized_input = f32_prefix(
        &production.moe.normalized_input,
        vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
        "duplicate packed GPU route normalized input",
    )
    .unwrap();
    let token_ids = i32_prefix(
        &production.token_ids,
        vec![n_tokens as u64],
        "duplicate packed GPU route tokens",
    )
    .unwrap();
    let views = PackedMoeViews {
        normalized_input,
        logits,
        hash_ids: None,
    };
    let generation = production.moe.take_gpu_route_generation().unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    production
        .moe
        .encode_gpu_route_compact(
            &ctx,
            &encoder,
            &views,
            PackedRouteSource::Hash,
            &token_ids,
            Some(&duplicate_map),
            n_tokens,
            1.5,
            generation,
        )
        .unwrap();
    encoder.end();
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert!(command.error().is_none());

    let schedule = production
        .moe
        .capture_gpu_compact_schedule(n_tokens, generation)
        .unwrap();
    assert_eq!(schedule.len(), 1);
    assert_eq!(
        (schedule[0].expert, schedule[0].start, schedule[0].len),
        (duplicate_expert, 0, MOE_TOP_K)
    );
    let mut actual_ids =
        host_read_i32(&production.moe.expert_ids, "duplicate packed route IDs").unwrap();
    let mut actual_weights =
        host_read_f32(&production.moe.weights, "duplicate packed route weights").unwrap();
    actual_ids.truncate(MOE_TOP_K);
    actual_weights.truncate(MOE_TOP_K);
    assert_eq!(actual_ids, duplicate_map_values);
    let scores =
        crate::deepseek_v4_oracle::sqrt_softplus_scores(&fixture.logits[..MOE_EXPERT_COUNT])
            .unwrap();
    let expected =
        crate::deepseek_v4_oracle::hash_route(&scores, &[duplicate_expert; MOE_TOP_K], 1.5)
            .unwrap();
    for (&actual, &expected) in actual_weights.iter().zip(&expected.weights) {
        assert!((actual - expected).abs() <= 1e-4 * expected.abs().max(1.0));
    }
    let mut expected_counts = vec![0; MOE_EXPERT_COUNT];
    expected_counts[duplicate_expert] = MOE_TOP_K as i32;
    assert_eq!(
        host_read_i32(
            &production.moe.gpu_route.counts,
            "duplicate packed route counts"
        )
        .unwrap(),
        expected_counts
    );
    let mut actual_rows =
        host_read_i32(&production.moe.bucket_rows, "duplicate packed route rows").unwrap();
    let mut actual_slots =
        host_read_i32(&production.moe.bucket_slots, "duplicate packed route slots").unwrap();
    actual_rows.truncate(MOE_TOP_K);
    actual_slots.truncate(MOE_TOP_K);
    assert_eq!(actual_rows, vec![0; MOE_TOP_K]);
    assert_eq!(actual_slots, (0..MOE_TOP_K as i32).collect::<Vec<_>>());
    let expected_tiles32 = packed_grouped_expert_tiles(n_tokens, &schedule).unwrap();
    let expected_tiles16 = packed_grouped_iq2_mma16_tiles(n_tokens, &schedule).unwrap();
    assert_eq!(expected_tiles32.len(), 1);
    assert_eq!(expected_tiles16.len(), 1);
    assert_eq!(expected_tiles32[0].count, MOE_TOP_K as u32);
    assert_eq!(expected_tiles16[0].count, MOE_TOP_K as u32);
    assert_eq!(
        host_read_i32(
            &production.moe.grouped_tiles,
            "duplicate packed route 32-row tiles",
        )
        .unwrap(),
        padded_tile_words(&expected_tiles32, PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS)
    );
    assert_eq!(
        host_read_i32(
            &production.moe.grouped_iq2_mma16_tiles,
            "duplicate packed route 16-row tiles",
        )
        .unwrap(),
        padded_tile_words(&expected_tiles16, PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS,)
    );

    production.moe.gpu_route.next_generation.set(u32::MAX);
    assert!(
        production
            .moe
            .take_gpu_route_generation()
            .unwrap_err()
            .to_string()
            .contains("exhausted before wrap")
    );
}

#[test]
fn packed_gpu_route_records_repeat_and_reject_missing_producers() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let fixture = PackedRouteFixture::new(&ctx);
    for (source, n_tokens) in [
        (PackedRouteMicroproofSource::Learned, 12),
        (PackedRouteMicroproofSource::Hash, 128),
    ] {
        let first = fixture.run(&ctx, source, n_tokens, n_tokens, MOE_EXPERT_COUNT);
        let second = fixture.run(&ctx, source, n_tokens, n_tokens, MOE_EXPERT_COUNT);
        assert_packed_route_capture(&fixture, source, n_tokens, &first);
        assert_packed_route_capture(&fixture, source, n_tokens, &second);
        assert_eq!(second.expert_ids, first.expert_ids);
        assert_eq!(
            second
                .weights
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            first
                .weights
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(second.counts, first.counts);
        assert_eq!(second.slot_ids, first.slot_ids);
        assert_eq!(second.signature[2], first.signature[2]);
    }

    let n_tokens = 12;
    fixture.run(
        &ctx,
        PackedRouteMicroproofSource::Learned,
        n_tokens,
        n_tokens,
        MOE_EXPERT_COUNT,
    );
    let missing_route = fixture.run(
        &ctx,
        PackedRouteMicroproofSource::Learned,
        n_tokens,
        n_tokens - 1,
        MOE_EXPERT_COUNT,
    );
    assert_packed_route_failure(
        &missing_route,
        n_tokens,
        PACKED_ROUTE_STALE_ROUTE,
        (n_tokens * MOE_TOP_K) as i32,
    );

    let wide_tokens = 512;
    let valid_hash_route = fixture.run(
        &ctx,
        PackedRouteMicroproofSource::Hash,
        wide_tokens,
        wide_tokens,
        MOE_EXPERT_COUNT,
    );
    assert_packed_route_capture(
        &fixture,
        PackedRouteMicroproofSource::Hash,
        wide_tokens,
        &valid_hash_route,
    );
    let missing_hash_route = fixture.run(
        &ctx,
        PackedRouteMicroproofSource::Hash,
        wide_tokens,
        257,
        MOE_EXPERT_COUNT,
    );
    assert_packed_route_failure(
        &missing_hash_route,
        wide_tokens,
        PACKED_ROUTE_STALE_ROUTE,
        (wide_tokens * MOE_TOP_K) as i32,
    );

    let missing_schedule = fixture.run(
        &ctx,
        PackedRouteMicroproofSource::Learned,
        n_tokens,
        n_tokens,
        MOE_EXPERT_COUNT - 1,
    );
    assert_packed_route_failure(
        &missing_schedule,
        n_tokens,
        PACKED_ROUTE_STALE_SCHEDULE,
        missing_schedule.counts[..MOE_EXPERT_COUNT - 1].iter().sum(),
    );

    let valid_before_missing_validator = fixture.run(
        &ctx,
        PackedRouteMicroproofSource::Learned,
        n_tokens,
        n_tokens,
        MOE_EXPERT_COUNT,
    );
    let generation = fixture.generations.take().unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    fixture
        .scratch
        .encode_learned(
            &ctx,
            &encoder,
            &fixture.bias,
            n_tokens,
            n_tokens,
            generation,
        )
        .unwrap();
    fixture
        .scratch
        .encode_schedule(&ctx, &encoder, n_tokens, MOE_EXPERT_COUNT, generation)
        .unwrap();
    fixture
        .scratch
        .encode_signature(&ctx, &encoder, n_tokens, generation)
        .unwrap();
    encoder.end();
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert!(command.error().is_none());
    let missing_validator = fixture.scratch.capture(n_tokens, generation.get());
    assert_eq!(
        missing_validator.aggregate,
        valid_before_missing_validator.aggregate
    );
    assert_eq!(
        missing_validator.signature,
        vec![
            generation.get() as i32,
            PACKED_ROUTE_INVALID_AGGREGATE,
            0,
            packed_route_signature_completion(generation.get(), n_tokens) as i32,
        ]
    );

    fixture.scratch.assert_slot_guards();

    let concurrent_command = ctx.queue.commandBuffer().unwrap();
    let concurrent = KernelEncoder::begin_concurrent(&concurrent_command);
    let concurrent_generation = fixture.generations.take().unwrap();
    for error in [
        fixture.scratch.encode_learned(
            &ctx,
            &concurrent,
            &fixture.bias,
            n_tokens,
            n_tokens,
            concurrent_generation,
        ),
        fixture.scratch.encode_hash(
            &ctx,
            &concurrent,
            &fixture.token_to_expert,
            n_tokens,
            n_tokens,
            concurrent_generation,
        ),
        fixture.scratch.encode_schedule(
            &ctx,
            &concurrent,
            n_tokens,
            MOE_EXPERT_COUNT,
            concurrent_generation,
        ),
        fixture
            .scratch
            .encode_validate(&ctx, &concurrent, n_tokens, concurrent_generation),
        fixture
            .scratch
            .encode_signature(&ctx, &concurrent, n_tokens, concurrent_generation),
    ] {
        assert!(error.unwrap_err().to_string().contains("ordered serial"));
    }
    concurrent.end();

    let exhausted = PackedRouteGenerationOwner::with_next(u32::MAX);
    let error = exhausted.take().unwrap_err();
    assert!(error.to_string().contains("exhausted before wrap"));
    assert!(
        exhausted
            .take()
            .unwrap_err()
            .to_string()
            .contains("exhausted")
    );
}

#[test]
fn packed_route_authority_rejects_invalid_producers_and_private_state() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let fixture = PackedRouteFixture::new(&ctx);
    let run_pipeline = |source: PackedRouteMicroproofSource,
                        bias: &MetalTensor,
                        token_to_expert: &MetalTensor,
                        n_tokens: usize| {
        let generation = fixture.generations.take().unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        fixture
            .scratch
            .encode_pipeline(
                &ctx,
                &encoder,
                source,
                bias,
                token_to_expert,
                n_tokens,
                n_tokens,
                MOE_EXPERT_COUNT,
                generation,
            )
            .unwrap();
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(command.error().is_none());
        fixture.scratch.capture(n_tokens, generation.get())
    };

    let mut nonfinite_logits = fixture.logits.clone();
    nonfinite_logits[0] = f32::NAN;
    write_raw_f32(&fixture.scratch.logits, &nonfinite_logits);
    let failed = run_pipeline(
        PackedRouteMicroproofSource::Learned,
        &fixture.bias,
        &fixture.token_to_expert,
        1,
    );
    assert_eq!(
        failed.route_status[0],
        DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_LOGIT
    );
    assert_packed_route_failure(&failed, 1, PACKED_ROUTE_FAILED_ROUTE, 0);

    host_write_f32(
        &fixture.scratch.logits,
        &fixture.logits,
        "restore packed route logits",
    )
    .unwrap();
    let mut nonfinite_bias = fixture.bias_values.clone();
    nonfinite_bias[0] = f32::NAN;
    let nonfinite_bias = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&nonfinite_bias),
        vec![MOE_EXPERT_COUNT as u64],
        GgmlType::F32,
    )
    .unwrap();
    let failed = run_pipeline(
        PackedRouteMicroproofSource::Learned,
        &nonfinite_bias,
        &fixture.token_to_expert,
        1,
    );
    assert_eq!(
        failed.route_status[0],
        DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_BIAS
    );
    assert_packed_route_failure(&failed, 1, PACKED_ROUTE_FAILED_ROUTE, 0);

    let mut invalid_tokens = fixture.token_ids.clone();
    invalid_tokens[0] = -1;
    host_write_i32(
        &fixture.scratch.token_ids,
        &invalid_tokens,
        "invalid packed hash token",
    )
    .unwrap();
    let failed = run_pipeline(
        PackedRouteMicroproofSource::Hash,
        &fixture.bias,
        &fixture.token_to_expert,
        1,
    );
    assert_eq!(
        failed.route_status[0],
        DEEPSEEK_V4_ROUTE_STATUS_INVALID_TOKEN
    );
    assert_packed_route_failure(&failed, 1, PACKED_ROUTE_FAILED_ROUTE, 0);
    host_write_i32(
        &fixture.scratch.token_ids,
        &fixture.token_ids,
        "restore packed hash tokens",
    )
    .unwrap();

    let invalid_map = vec![-1, 1, 2, 3, 4, 5];
    let invalid_map = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&invalid_map),
        vec![MOE_TOP_K as u64, 1],
        GgmlType::I32,
    )
    .unwrap();
    let failed = run_pipeline(
        PackedRouteMicroproofSource::Hash,
        &fixture.bias,
        &invalid_map,
        1,
    );
    assert_eq!(
        failed.route_status[0],
        DEEPSEEK_V4_ROUTE_STATUS_INVALID_EXPERT
    );
    assert_packed_route_failure(&failed, 1, PACKED_ROUTE_FAILED_ROUTE, 0);

    write_raw_f32(&fixture.scratch.logits, &nonfinite_logits);
    let failed = run_pipeline(
        PackedRouteMicroproofSource::Hash,
        &fixture.bias,
        &fixture.token_to_expert,
        1,
    );
    assert_eq!(
        failed.route_status[0],
        DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_LOGIT
    );
    assert_packed_route_failure(&failed, 1, PACKED_ROUTE_FAILED_ROUTE, 0);
    host_write_f32(
        &fixture.scratch.logits,
        &fixture.logits,
        "restore packed route logits after hash fault",
    )
    .unwrap();

    let write_route_state =
        |generation: NonZeroU32, n_tokens: usize, expert_ids: &[i32], weights: &[f32]| {
            let mut full_ids = vec![-1; PACKED_GPU_ROUTE_MAX_TOKENS * MOE_TOP_K];
            full_ids[..expert_ids.len()].copy_from_slice(expert_ids);
            host_write_i32(&fixture.scratch.expert_ids, &full_ids, "prepared route IDs").unwrap();
            let mut full_weights = vec![0.0; PACKED_GPU_ROUTE_MAX_TOKENS * MOE_TOP_K];
            full_weights[..weights.len()].copy_from_slice(weights);
            write_raw_f32(&fixture.scratch.weights, &full_weights);
            let mut generations = vec![0; PACKED_GPU_ROUTE_MAX_TOKENS];
            generations[..n_tokens].fill(generation.get() as i32);
            host_write_i32(
                &fixture.scratch.route_generations,
                &generations,
                "prepared route generations",
            )
            .unwrap();
            let mut statuses = vec![0; PACKED_GPU_ROUTE_MAX_TOKENS];
            statuses[..n_tokens].fill(DEEPSEEK_V4_ROUTE_STATUS_READY);
            host_write_i32(
                &fixture.scratch.route_status,
                &statuses,
                "prepared route statuses",
            )
            .unwrap();
        };
    let run_route_state = |n_tokens: usize, expert_ids: &[i32], weights: &[f32]| {
        let generation = fixture.generations.take().unwrap();
        write_route_state(generation, n_tokens, expert_ids, weights);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        fixture
            .scratch
            .encode_schedule(&ctx, &encoder, n_tokens, MOE_EXPERT_COUNT, generation)
            .unwrap();
        fixture
            .scratch
            .encode_validate(&ctx, &encoder, n_tokens, generation)
            .unwrap();
        fixture
            .scratch
            .encode_signature(&ctx, &encoder, n_tokens, generation)
            .unwrap();
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(command.error().is_none());
        fixture.scratch.capture(n_tokens, generation.get())
    };

    let n_tokens = 12;
    let valid = fixture.run(
        &ctx,
        PackedRouteMicroproofSource::Learned,
        n_tokens,
        n_tokens,
        MOE_EXPERT_COUNT,
    );
    let mut invalid_ids = valid.expert_ids.clone();
    invalid_ids[0] = -1;
    let failed = run_route_state(n_tokens, &invalid_ids, &valid.weights);
    assert_packed_route_failure(
        &failed,
        n_tokens,
        PACKED_ROUTE_INVALID_ID,
        (n_tokens * MOE_TOP_K - 1) as i32,
    );

    let mut duplicate_ids = valid.expert_ids.clone();
    duplicate_ids[1] = duplicate_ids[0];
    let failed = run_route_state(n_tokens, &duplicate_ids, &valid.weights);
    assert_packed_route_failure(
        &failed,
        n_tokens,
        PACKED_ROUTE_DUPLICATE_ID,
        (n_tokens * MOE_TOP_K) as i32,
    );

    let mut invalid_weights = valid.weights.clone();
    invalid_weights[0] = f32::NAN;
    let failed = run_route_state(n_tokens, &valid.expert_ids, &invalid_weights);
    assert_packed_route_failure(
        &failed,
        n_tokens,
        PACKED_ROUTE_INVALID_WEIGHT,
        (n_tokens * MOE_TOP_K) as i32,
    );

    let run_schedule_state =
        |counts: &[i32], slot_ids: &[i32], expected_status: i32, expected_total: i32| {
            let generation = fixture.generations.take().unwrap();
            write_route_state(generation, n_tokens, &valid.expert_ids, &valid.weights);
            host_write_i32(&fixture.scratch.counts, counts, "corrupt schedule counts").unwrap();
            let mut full_slots = vec![-1; PACKED_GPU_ROUTE_MAX_TOKENS * MOE_EXPERT_COUNT];
            full_slots[..slot_ids.len()].copy_from_slice(slot_ids);
            host_write_i32(
                &fixture.scratch.slot_ids,
                &full_slots,
                "corrupt schedule slots",
            )
            .unwrap();
            host_write_i32(
                &fixture.scratch.schedule_generations,
                &vec![generation.get() as i32; MOE_EXPERT_COUNT],
                "corrupt schedule generations",
            )
            .unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            fixture
                .scratch
                .encode_validate(&ctx, &encoder, n_tokens, generation)
                .unwrap();
            fixture
                .scratch
                .encode_signature(&ctx, &encoder, n_tokens, generation)
                .unwrap();
            encoder.end();
            command.commit();
            crate::metal::wait_completed(&command).expect("command buffer completed");
            assert!(command.error().is_none());
            let capture = fixture.scratch.capture(n_tokens, generation.get());
            assert_packed_route_failure(&capture, n_tokens, expected_status, expected_total);
        };

    let mut invalid_counts = valid.counts.clone();
    let counted_expert = valid.expert_ids[0] as usize;
    let omitted_count = invalid_counts[counted_expert];
    invalid_counts[counted_expert] = n_tokens as i32 + 1;
    run_schedule_state(
        &invalid_counts,
        &valid.slot_ids,
        PACKED_ROUTE_INVALID_COUNT,
        (n_tokens * MOE_TOP_K) as i32 - omitted_count,
    );
    let mut invalid_slots = valid.slot_ids.clone();
    let occupied = invalid_slots.iter().position(|&slot| slot >= 0).unwrap();
    invalid_slots[occupied] ^= 1;
    run_schedule_state(
        &valid.counts,
        &invalid_slots,
        PACKED_ROUTE_INVALID_SCHEDULE,
        (n_tokens * MOE_TOP_K) as i32,
    );
    let mut invalid_padding = valid.slot_ids.clone();
    let padding = invalid_padding.iter().position(|&slot| slot == -1).unwrap();
    invalid_padding[padding] = -2;
    run_schedule_state(
        &valid.counts,
        &invalid_padding,
        PACKED_ROUTE_INVALID_PADDING,
        (n_tokens * MOE_TOP_K) as i32,
    );

    let terminal = fixture.run(
        &ctx,
        PackedRouteMicroproofSource::Learned,
        PACKED_GPU_ROUTE_MAX_TOKENS,
        PACKED_GPU_ROUTE_MAX_TOKENS,
        MOE_EXPERT_COUNT,
    );
    let terminal_ids = vec![255; PACKED_GPU_ROUTE_MAX_TOKENS * MOE_TOP_K];
    let failed = run_route_state(
        PACKED_GPU_ROUTE_MAX_TOKENS,
        &terminal_ids,
        &terminal.weights,
    );
    assert_packed_route_failure(
        &failed,
        PACKED_GPU_ROUTE_MAX_TOKENS,
        PACKED_ROUTE_INVALID_COUNT,
        0,
    );
    assert_eq!(
        failed.counts[255],
        (PACKED_GPU_ROUTE_MAX_TOKENS * MOE_TOP_K) as i32
    );
    fixture.scratch.assert_slot_guards();
}

fn percentile_ms(values: &[f64], percentile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (percentile * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn median_ms(values: &[f64]) -> f64 {
    percentile_ms(values, 0.5)
}

fn relative_drift(left: f64, right: f64) -> f64 {
    (left - right).abs() / left.min(right)
}

#[test]
#[ignore = "focused exact packed-route profiler; run release with --nocapture"]
fn profile_exact_packed_gpu_route_and_schedule_packet() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let fixture = PackedRouteFixture::new(&ctx);
    let sample = |n_tokens: usize, stages: usize| {
        assert!((1..=4).contains(&stages));
        let started = std::time::Instant::now();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let mut last_generation = None;
        for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
            let generation = fixture.generations.take().unwrap();
            let source = if layer < 3 {
                PackedRouteMicroproofSource::Hash
            } else {
                PackedRouteMicroproofSource::Learned
            };
            match source {
                PackedRouteMicroproofSource::Learned => fixture
                    .scratch
                    .encode_learned(
                        &ctx,
                        &encoder,
                        &fixture.bias,
                        n_tokens,
                        n_tokens,
                        generation,
                    )
                    .unwrap(),
                PackedRouteMicroproofSource::Hash => fixture
                    .scratch
                    .encode_hash(
                        &ctx,
                        &encoder,
                        &fixture.token_to_expert,
                        n_tokens,
                        n_tokens,
                        generation,
                    )
                    .unwrap(),
            }
            if stages >= 2 {
                fixture
                    .scratch
                    .encode_schedule(&ctx, &encoder, n_tokens, MOE_EXPERT_COUNT, generation)
                    .unwrap();
            }
            if stages >= 3 {
                fixture
                    .scratch
                    .encode_validate(&ctx, &encoder, n_tokens, generation)
                    .unwrap();
            }
            if stages >= 4 {
                fixture
                    .scratch
                    .encode_signature(&ctx, &encoder, n_tokens, generation)
                    .unwrap();
            }
            last_generation = Some(generation);
        }
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        assert!(command.error().is_none());
        let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        let generation = last_generation.unwrap();
        if stages == 4 {
            let signature = host_read_i32(&fixture.scratch.signature, "profile signature").unwrap();
            assert_eq!(signature[0], generation.get() as i32);
            assert_eq!(signature[1], DEEPSEEK_V4_ROUTE_STATUS_READY);
            assert_eq!(
                signature[3],
                packed_route_signature_completion(generation.get(), n_tokens) as i32
            );
        }
        (gpu_ms, wall_ms)
    };

    for n_tokens in [12, 128] {
        let warm_started = std::time::Instant::now();
        let mut warm_samples = 0;
        while warm_samples < 96 || warm_started.elapsed() < std::time::Duration::from_secs(1) {
            sample(n_tokens, 4);
            warm_samples += 1;
        }
        for stages in 1..=4 {
            for _ in 0..2 {
                sample(n_tokens, stages);
            }
            let phase_samples = (0..9).map(|_| sample(n_tokens, stages)).collect::<Vec<_>>();
            let phase_gpu = phase_samples
                .iter()
                .map(|sample| sample.0)
                .collect::<Vec<_>>();
            let phase_wall = phase_samples
                .iter()
                .map(|sample| sample.1)
                .collect::<Vec<_>>();
            eprintln!(
                "deepseek_v4 packed_route_stage n={n_tokens} stages={stages} gpu_median_ms={:.6} gpu_p95_ms={:.6} wall_median_ms={:.6} wall_p95_ms={:.6}",
                median_ms(&phase_gpu),
                percentile_ms(&phase_gpu, 0.95),
                median_ms(&phase_wall),
                percentile_ms(&phase_wall, 0.95),
            );
        }
        for _ in 0..32 {
            sample(n_tokens, 4);
        }
        let collect = |count: usize| (0..count).map(|_| sample(n_tokens, 4)).collect::<Vec<_>>();
        let control_a = collect(12);
        let candidate = collect(40);
        let control_b = collect(12);
        let split = |samples: &[(f64, f64)]| {
            (
                samples.iter().map(|sample| sample.0).collect::<Vec<_>>(),
                samples.iter().map(|sample| sample.1).collect::<Vec<_>>(),
            )
        };
        let (control_a_gpu, control_a_wall) = split(&control_a);
        let (candidate_gpu, candidate_wall) = split(&candidate);
        let (control_b_gpu, control_b_wall) = split(&control_b);
        let gpu_drift = relative_drift(median_ms(&control_a_gpu), median_ms(&control_b_gpu));
        let wall_drift = relative_drift(median_ms(&control_a_wall), median_ms(&control_b_wall));
        let gpu_p95 = percentile_ms(&candidate_gpu, 0.95);
        let wall_p95 = percentile_ms(&candidate_wall, 0.95);
        eprintln!(
            "deepseek_v4 packed_route_packet n={n_tokens} candidate_gpu_ms={candidate_gpu:?} candidate_wall_ms={candidate_wall:?} control_a_gpu_ms={control_a_gpu:?} control_a_wall_ms={control_a_wall:?} control_b_gpu_ms={control_b_gpu:?} control_b_wall_ms={control_b_wall:?} gpu_p95_ms={gpu_p95:.6} wall_p95_ms={wall_p95:.6} gpu_control_drift={gpu_drift:.6} wall_control_drift={wall_drift:.6}"
        );
        let ceiling_ms = if n_tokens == 12 { 4.3 } else { 8.6 };
        assert!(gpu_p95 <= ceiling_ms, "GPU p95 {gpu_p95} > {ceiling_ms}");
        assert!(wall_p95 <= ceiling_ms, "wall p95 {wall_p95} > {ceiling_ms}");
        assert!(gpu_drift <= 0.05, "GPU control drift {gpu_drift}");
        assert!(wall_drift <= 0.05, "wall control drift {wall_drift}");
    }
}

#[test]
fn packed_sparse_visibility_tracks_publication_cadence() {
    assert_eq!(
        packed_sparse_visible_counts(2_051, 0, 5, 514).unwrap(),
        [513, 513, 513, 513, 514]
    );
    assert_eq!(
        packed_sparse_visible_counts(2_047, 4, 9, 514).unwrap(),
        [513, 513, 513, 513, 514]
    );
    assert!(
        packed_sparse_visible_counts(2_051, 0, 5, 515)
            .unwrap_err()
            .to_string()
            .contains("differs from published row count 515")
    );
    assert!(
        packed_sparse_visible_counts(2_051, 5, 5, 514)
            .unwrap_err()
            .to_string()
            .contains("visibility geometry is invalid")
    );
}

#[cfg(feature = "dsv4-diagnostics")]
#[test]
fn fp4_selection_counterfactual_rejects_multi_query_chunks_before_encoding() {
    validate_fp4_selection_counterfactual_packed(0, 128).unwrap();
    validate_fp4_selection_counterfactual_packed(2_048, 4).unwrap();
    let error = validate_fp4_selection_counterfactual_packed(2_048, 5).unwrap_err();
    assert!(error.to_string().contains("got 2"), "{error}");
    let error = validate_fp4_selection_counterfactual_packed(2_052, 2).unwrap_err();
    assert!(error.to_string().contains("got 2"), "{error}");
}

#[test]
fn tiled_hca_offset_starts_at_the_513th_visible_row() {
    assert_eq!(tiled_hca_query_offset(65_535, 1), None);
    assert_eq!(tiled_hca_query_offset(65_536, 127), None);
    assert_eq!(tiled_hca_query_offset(65_536, 128), Some(127));
    assert_eq!(tiled_hca_query_offset(65_662, 2), Some(1));
    assert_eq!(tiled_hca_query_offset(65_663, 1), Some(0));
    assert_eq!(tiled_hca_query_offset(1_048_448, 128), Some(0));
}

#[test]
fn packed_raw_chunk_publication_matches_ordered_ring_updates() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const HEAD_DIM: usize = 16;

    fn read_f16_bits(tensor: &MetalTensor) -> Vec<u16> {
        assert_eq!(tensor.dtype, GgmlType::F16);
        unsafe {
            let pointer = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<u16>();
            std::slice::from_raw_parts(pointer, tensor.n_elements() as usize).to_vec()
        }
    }

    for (start_position, n_tokens) in [
        (0_u32, 1_usize),
        (123, 12),
        (257, 127),
        (511, 128),
        (511, DEEPSEEK_V4_PREFILL_MAX_TOKENS),
    ] {
        let source_values = (0..n_tokens * HEAD_DIM)
            .map(|index| ((index * 29 + index / 7 + 3) % 197) as f32 * 0.0031 - 0.29)
            .collect::<Vec<_>>();
        let initial_ring = (0..DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM)
            .map(|index| half::f16::from_f32((index % 113) as f32 * 0.001 - 0.04).to_bits())
            .collect::<Vec<_>>();
        let source = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&source_values),
            vec![HEAD_DIM as u64, n_tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let chunk = MetalTensor::zeros_f16(&ctx, vec![HEAD_DIM as u64, n_tokens as u64]).unwrap();
        let ring = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&initial_ring),
            vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            GgmlType::F16,
        )
        .unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_publish_raw_chunk_f16(
            &ctx,
            &encoder,
            &source,
            &chunk,
            &ring,
            start_position,
            n_tokens,
            HEAD_DIM,
        )
        .unwrap();
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(command.error().is_none());

        let expected_chunk = source_values
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        let mut expected_ring = initial_ring;
        for row in 0..n_tokens {
            let slot = (start_position as usize + row) % DEEPSEEK_V4_LOCAL_WINDOW;
            let source = &expected_chunk[row * HEAD_DIM..(row + 1) * HEAD_DIM];
            expected_ring[slot * HEAD_DIM..(slot + 1) * HEAD_DIM].copy_from_slice(source);
        }
        assert_eq!(read_f16_bits(&chunk), expected_chunk);
        assert_eq!(read_f16_bits(&ring), expected_ring);
    }
}

#[test]
fn packed_dense_attention_matches_ordered_singleton_rows_within_roundoff() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let n_tokens = 8;
    let config = deepseek_v4_session_attention_config();
    let dims = config.checked().unwrap();
    let queries = (0..n_tokens * dims.query_width)
        .map(|index| ((index * 17 + index / 11) % 257) as f32 * 0.0007 - 0.08)
        .collect::<Vec<_>>();
    let raw = (0..DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim)
        .map(|index| ((index * 13 + 5) % 193) as f32 * 0.0011 - 0.09)
        .collect::<Vec<_>>();
    let compressed = (0..DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS * config.head_dim)
        .map(|index| ((index * 19 + 3) % 211) as f32 * 0.0009 - 0.085)
        .collect::<Vec<_>>();
    let sinks = (0..config.head_count)
        .map(|head| head as f32 * 0.013 - 0.31)
        .collect::<Vec<_>>();
    let queries = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&queries),
        vec![dims.query_width as u64, n_tokens as u64],
        GgmlType::F32,
    )
    .unwrap();
    let raw_bits = raw
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let raw = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&raw_bits),
        vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        GgmlType::F16,
    )
    .unwrap();
    let raw_chunk = raw.view_subrange(0, vec![config.head_dim as u64, n_tokens as u64]);
    let raw_before = MetalTensor::zeros_f16(
        &ctx,
        vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
    )
    .unwrap();
    let compressed_bits = compressed
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let compressed = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&compressed_bits),
        vec![
            config.head_dim as u64,
            DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
        ],
        GgmlType::F16,
    )
    .unwrap();
    let sinks = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&sinks),
        vec![config.head_count as u64],
        GgmlType::F32,
    )
    .unwrap();
    let packed =
        MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, n_tokens as u64]).unwrap();
    let ordered =
        MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, n_tokens as u64]).unwrap();
    let overlapping_raw_storage = MetalTensor::zeros_f16(
        &ctx,
        vec![
            config.head_dim as u64,
            (DEEPSEEK_V4_LOCAL_WINDOW + 1) as u64,
        ],
    )
    .unwrap();
    let overlapping_raw =
        overlapping_raw_storage.view_subrange(0, vec![config.head_dim as u64, n_tokens as u64]);
    let overlapping_raw_before = overlapping_raw_storage.view_subrange(
        config.head_dim as u64,
        vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
    );
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let empty = encode_cooperative_dense_sink_attention_f16(
        &ctx,
        &encoder,
        &queries,
        &raw_chunk,
        &raw_before,
        DeepSeekV4RawCacheLayout::Chunk,
        None,
        &sinks,
        &packed,
        AttentionKind::SlidingWindow,
        0,
        0,
        config,
    )
    .unwrap_err();
    assert!(empty.to_string().contains("requires at least one token"));
    let oversized = encode_cooperative_dense_sink_attention_f16(
        &ctx,
        &encoder,
        &queries,
        &raw_chunk,
        &raw_before,
        DeepSeekV4RawCacheLayout::Chunk,
        None,
        &sinks,
        &packed,
        AttentionKind::SlidingWindow,
        0,
        DEEPSEEK_V4_PREFILL_MAX_TOKENS + 1,
        config,
    )
    .unwrap_err();
    assert!(
        oversized
            .to_string()
            .contains("exceeds retained chunk limit")
    );
    let aliased = encode_cooperative_dense_sink_attention_f16(
        &ctx,
        &encoder,
        &queries,
        &raw_chunk,
        &raw,
        DeepSeekV4RawCacheLayout::Chunk,
        Some(DeepSeekV4PublishedRows {
            cache: &compressed,
            count: n_tokens / 4,
            capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
        }),
        &sinks,
        &packed,
        AttentionKind::CompressedSparse,
        0,
        n_tokens,
        config,
    )
    .unwrap_err();
    assert!(
        aliased
            .to_string()
            .contains("requires disjoint current and preserved raw caches")
    );
    let partially_aliased = encode_cooperative_dense_sink_attention_f16(
        &ctx,
        &encoder,
        &queries,
        &overlapping_raw,
        &overlapping_raw_before,
        DeepSeekV4RawCacheLayout::Chunk,
        Some(DeepSeekV4PublishedRows {
            cache: &compressed,
            count: n_tokens / 4,
            capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
        }),
        &sinks,
        &packed,
        AttentionKind::CompressedSparse,
        0,
        n_tokens,
        config,
    )
    .unwrap_err();
    assert!(
        partially_aliased
            .to_string()
            .contains("requires disjoint current and preserved raw caches")
    );
    encode_packed_dense_sink_attention_f16(
        &ctx,
        &encoder,
        &queries,
        &raw_chunk,
        &raw_before,
        Some(DeepSeekV4PublishedRows {
            cache: &compressed,
            count: n_tokens / 4,
            capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
        }),
        &sinks,
        &packed,
        AttentionKind::CompressedSparse,
        0,
        n_tokens,
    )
    .unwrap();
    for row in 0..n_tokens {
        let query = f32_row(
            &queries,
            row,
            dims.query_width,
            vec![config.head_dim as u64, config.head_count as u64],
            "ordered query",
        )
        .unwrap();
        let output = f32_row(
            &ordered,
            row,
            dims.query_width,
            vec![config.head_dim as u64, config.head_count as u64],
            "ordered output",
        )
        .unwrap();
        let count = (row + 1) / 4;
        encode_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &query,
            &raw,
            (count > 0).then_some(DeepSeekV4PublishedRows {
                cache: &compressed,
                count,
                capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
            }),
            &sinks,
            &output,
            row as u32,
            config,
        )
        .unwrap();
    }
    encoder.end();
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert!(command.error().is_none());
    let packed = host_read_f32(&packed, "packed attention").unwrap();
    let ordered = host_read_f32(&ordered, "ordered attention").unwrap();
    assert_eq!(packed.len(), ordered.len());
    for (index, (&packed, &ordered)) in packed.iter().zip(&ordered).enumerate() {
        let allowed = 2.0 * f32::EPSILON * ordered.abs().max(1.0);
        assert!(
            (packed - ordered).abs() <= allowed,
            "packed attention differs at {index}: {packed} vs {ordered}, allowed {allowed}"
        );
    }
}

#[test]
fn packed_hca_splits_the_first_tiled_query_without_future_raw_leakage() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const START_POSITION: usize = 65_662;
    const N_TOKENS: usize = 2;
    const CAPACITY: usize = 768;
    let config = deepseek_v4_session_attention_config();
    let dims = config.checked().unwrap();
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let query_values = (0..N_TOKENS * dims.query_width)
        .map(|index| {
            let token = index / dims.query_width;
            let within = index % dims.query_width;
            let head = within / config.head_dim;
            let dimension = within % config.head_dim;
            let tag = (token * 31 + head * 17 + dimension * 7) % 149;
            (tag as f32 - 74.0) * 0.0011
        })
        .collect::<Vec<_>>();
    let future_row = query_values[..config.head_dim]
        .iter()
        .map(|value| round_f16(value * 512.0))
        .collect::<Vec<_>>();
    let raw_value = |position: usize, dimension: usize| {
        let tag = (position * 23 + dimension * 11 + position / 13) % 137;
        round_f16((tag as f32 - 68.0) * 0.0013)
    };
    let mut raw_before = vec![0.0f32; DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim];
    for position in START_POSITION - DEEPSEEK_V4_LOCAL_WINDOW..START_POSITION {
        let slot = position % DEEPSEEK_V4_LOCAL_WINDOW;
        for dimension in 0..config.head_dim {
            raw_before[slot * config.head_dim + dimension] = raw_value(position, dimension);
        }
    }
    let future_row_values = &future_row;
    let raw_current = (0..N_TOKENS)
        .flat_map(|token| {
            (0..config.head_dim).map(move |dimension| {
                if token == 1 {
                    future_row_values[dimension]
                } else {
                    raw_value(START_POSITION + token, dimension)
                }
            })
        })
        .collect::<Vec<_>>();
    let mut compressed_values = (0..CAPACITY * config.head_dim)
        .map(|index| {
            let row = index / config.head_dim;
            let dimension = index % config.head_dim;
            let tag = (row * 43 + dimension * 5 + row / 7) % 151;
            round_f16((tag as f32 - 75.0) * 0.0012)
        })
        .collect::<Vec<_>>();
    for dimension in 0..config.head_dim {
        compressed_values[512 * config.head_dim + dimension] =
            round_f16(query_values[dimension] * 640.0);
    }
    let sinks_values = (0..config.head_count)
        .map(|head| head as f32 * 0.003 - 0.27)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(N_TOKENS * dims.query_width);
    for token in 0..N_TOKENS {
        let position = START_POSITION + token;
        let raw_start = position + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
        let mut raw_rows = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim);
        for logical_position in raw_start..=position {
            if logical_position == START_POSITION + 1 {
                raw_rows.extend_from_slice(&future_row);
            } else {
                raw_rows.extend(
                    (0..config.head_dim).map(|dimension| raw_value(logical_position, dimension)),
                );
            }
        }
        let count = (position + 1) / 128;
        expected.extend(
            crate::deepseek_v4_oracle::shared_kv_attention(
                &query_values[token * dims.query_width..(token + 1) * dims.query_width],
                config.head_count,
                config.head_dim,
                &raw_rows,
                &compressed_values[..count * config.head_dim],
                None,
                &sinks_values,
            )
            .unwrap(),
        );
    }
    let first_raw_start = START_POSITION + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
    let mut leaked_raw = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim);
    for logical_position in first_raw_start..=START_POSITION {
        if logical_position == first_raw_start {
            leaked_raw.extend_from_slice(&future_row);
        } else {
            leaked_raw.extend(
                (0..config.head_dim).map(|dimension| raw_value(logical_position, dimension)),
            );
        }
    }
    let leaked = crate::deepseek_v4_oracle::shared_kv_attention(
        &query_values[..dims.query_width],
        config.head_count,
        config.head_dim,
        &leaked_raw,
        &compressed_values[..DEEPSEEK_V4_CSA_TOP_K * config.head_dim],
        None,
        &sinks_values,
    )
    .unwrap();
    assert!(
        expected[..dims.query_width]
            .iter()
            .zip(leaked)
            .any(|(correct, leaked)| (correct - leaked).abs() > 1e-3)
    );

    let queries = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&query_values),
        vec![dims.query_width as u64, N_TOKENS as u64],
        GgmlType::F32,
    )
    .unwrap();
    let make_raw = |values: &[f32], rows: usize| {
        let bits = values
            .iter()
            .map(|value| half::f16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&bits),
            vec![config.head_dim as u64, rows as u64],
            GgmlType::F16,
        )
        .unwrap()
    };
    let raw_before = make_raw(&raw_before, DEEPSEEK_V4_LOCAL_WINDOW);
    let raw_current = make_raw(&raw_current, N_TOKENS);
    let compressed_bits = compressed_values
        .iter()
        .map(|value| half::f16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let compressed = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&compressed_bits),
        vec![config.head_dim as u64, CAPACITY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let sinks = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&sinks_values),
        vec![config.head_count as u64],
        GgmlType::F32,
    )
    .unwrap();
    let output =
        MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, N_TOKENS as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_packed_dense_sink_attention_f16(
        &ctx,
        &encoder,
        &queries,
        &raw_current,
        &raw_before,
        Some(DeepSeekV4PublishedRows {
            cache: &compressed,
            count: 513,
            capacity_rows: CAPACITY,
        }),
        &sinks,
        &output,
        AttentionKind::HeavilyCompressed,
        START_POSITION as u32,
        N_TOKENS,
    )
    .unwrap();
    encoder.end();
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert!(
        command.error().is_none(),
        "packed HCA split command failed: {:?}",
        command.error()
    );
    let actual = host_read_f32(&output, "packed HCA split output").unwrap();
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let allowed = 8e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= allowed,
            "packed HCA split output[{index}]={actual}, expected {expected}, allowed {allowed}"
        );
    }
}

#[test]
fn packed_sparse_suffix_matches_cpu_with_original_chunk_ring_visibility() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let config = deepseek_v4_session_attention_config();
    let dims = config.checked().unwrap();
    let start_position = 1_540_u32;
    let n_tokens = 512;
    let query_offset = n_tokens - 1;
    let query_count = n_tokens - query_offset;
    let raw_value = |position: usize, dimension: usize| {
        let tag = (position * 31 + dimension * 17 + position / 11) % 181;
        (tag as f32 - 90.0) * 0.0017
    };
    let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
    let mut prior_ring = vec![0.0; DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim];
    for position in 0..start_position as usize {
        let slot = position % DEEPSEEK_V4_LOCAL_WINDOW;
        for dimension in 0..config.head_dim {
            prior_ring[slot * config.head_dim + dimension] =
                round_f16(raw_value(position, dimension));
        }
    }
    let prior_bits = prior_ring
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let raw_cache = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&prior_bits),
        vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        GgmlType::F16,
    )
    .unwrap();
    let raw_cache_before_chunk = MetalTensor::zeros_f16(
        &ctx,
        vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
    )
    .unwrap();
    let raw_chunk =
        MetalTensor::zeros_f16(&ctx, vec![config.head_dim as u64, n_tokens as u64]).unwrap();
    let new_raw = (start_position as usize..start_position as usize + n_tokens)
        .flat_map(|position| {
            (0..config.head_dim).map(move |dimension| raw_value(position, dimension))
        })
        .collect::<Vec<_>>();
    let new_raw_tensor = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&new_raw),
        vec![config.head_dim as u64, n_tokens as u64],
        GgmlType::F32,
    )
    .unwrap();
    let query_values = (0..n_tokens * dims.query_width)
        .map(|index| {
            let token = index / dims.query_width;
            let within = index % dims.query_width;
            let head = within / config.head_dim;
            let dimension = within % config.head_dim;
            let tag = (token * 23 + head * 13 + dimension * 7) % 173;
            (tag as f32 - 86.0) * 0.0013
        })
        .collect::<Vec<_>>();
    let queries = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&query_values),
        vec![dims.query_width as u64, n_tokens as u64],
        GgmlType::F32,
    )
    .unwrap();
    let compressed_values = (0..DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS * config.head_dim)
        .map(|index| {
            let row = index / config.head_dim;
            let dimension = index % config.head_dim;
            let tag = (row * 43 + dimension * 5 + row / 7) % 191;
            round_f16((tag as f32 - 95.0) * 0.0015)
        })
        .collect::<Vec<_>>();
    let compressed_bits = compressed_values
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let compressed = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&compressed_bits),
        vec![
            config.head_dim as u64,
            DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
        ],
        GgmlType::F16,
    )
    .unwrap();
    let indexer = MetalTensor::zeros_f16(
        &ctx,
        vec![
            INDEXER_HEAD_DIM as u64,
            DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
        ],
    )
    .unwrap();
    let selected_ids = (1..=DEEPSEEK_V4_CSA_TOP_K as i32).collect::<Vec<_>>();
    let selected_ids = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&selected_ids),
        vec![DEEPSEEK_V4_CSA_TOP_K as u64, query_count as u64],
        GgmlType::I32,
    )
    .unwrap();
    let selected_counts = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&vec![DEEPSEEK_V4_CSA_TOP_K as i32; query_count]),
        vec![query_count as u64],
        GgmlType::I32,
    )
    .unwrap();
    let visible_counts = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&vec![513_i32; query_count]),
        vec![query_count as u64],
        GgmlType::I32,
    )
    .unwrap();
    let sinks_values = (0..config.head_count)
        .map(|head| head as f32 * 0.007 - 0.23)
        .collect::<Vec<_>>();
    let sinks = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&sinks_values),
        vec![config.head_count as u64],
        GgmlType::F32,
    )
    .unwrap();
    let output =
        MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, n_tokens as u64]).unwrap();
    let rows = DeepSeekV4CsaRows {
        attention_cache: &compressed,
        indexer_cache: &indexer,
        #[cfg(feature = "dsv4-diagnostics")]
        indexer_fp4_sidecar: None,
        count: 513,
        capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
    };
    let sparse = PackedSparseCsaViews {
        query_offset,
        query_count,
        cache_order_ids: selected_ids,
        selected_counts,
        visible_counts,
        index_queries: MetalTensor::zeros_f32(&ctx, vec![128, 64, query_count as u64]).unwrap(),
        head_weights: MetalTensor::zeros_f32(&ctx, vec![64, query_count as u64]).unwrap(),
        scores: MetalTensor::zeros_f32(
            &ctx,
            vec![
                DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
                query_count as u64,
            ],
        )
        .unwrap(),
        selected_mask: MetalTensor::zeros_i32(
            &ctx,
            vec![
                DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
                query_count as u64,
            ],
        )
        .unwrap(),
        status: MetalTensor::zeros_i32(&ctx, vec![query_count as u64]).unwrap(),
    };

    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_copy_raw_ring_f16_bits(&ctx, &encoder, &raw_cache, &raw_cache_before_chunk).unwrap();
    encode_publish_raw_chunk_f16(
        &ctx,
        &encoder,
        &new_raw_tensor,
        &raw_chunk,
        &raw_cache,
        start_position,
        n_tokens,
        config.head_dim,
    )
    .unwrap();
    let dense_queries = f32_prefix(
        &queries,
        vec![dims.query_width as u64, query_offset as u64],
        "packed sparse dense-prefix queries",
    )
    .unwrap();
    let dense_output = f32_prefix(
        &output,
        vec![dims.query_width as u64, query_offset as u64],
        "packed sparse dense-prefix output",
    )
    .unwrap();
    encode_packed_dense_sink_attention_f16(
        &ctx,
        &encoder,
        &dense_queries,
        &raw_chunk,
        &raw_cache_before_chunk,
        Some(DeepSeekV4PublishedRows {
            cache: &compressed,
            count: DEEPSEEK_V4_CSA_TOP_K,
            capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
        }),
        &sinks,
        &dense_output,
        AttentionKind::CompressedSparse,
        start_position,
        query_offset,
    )
    .unwrap();
    encode_packed_selected_sink_attention_f16(
        &ctx,
        &encoder,
        &queries,
        &raw_chunk,
        &raw_cache_before_chunk,
        rows,
        sparse.selection_view(),
        &sinks,
        &output,
        start_position,
        n_tokens,
    )
    .unwrap();
    encoder.end();
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert!(
        command.error().is_none(),
        "packed sparse attention failed: {:?}",
        command.error()
    );

    let actual = host_read_f32(&output, "packed sparse attention").unwrap();
    for token in [0, 127, 128, 255, 384, query_offset] {
        let position = start_position as usize + token;
        let raw_start = position + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
        let raw_rows = (raw_start..=position)
            .flat_map(|logical_position| {
                (0..config.head_dim)
                    .map(move |dimension| round_f16(raw_value(logical_position, dimension)))
            })
            .collect::<Vec<_>>();
        let compressed_count = (position + 1) / 4;
        let mask = (token >= query_offset).then(|| {
            let local = token - query_offset;
            let mut mask = vec![false; compressed_count];
            if local == 0 {
                mask[1..=DEEPSEEK_V4_CSA_TOP_K].fill(true);
            } else {
                mask[..DEEPSEEK_V4_CSA_TOP_K].fill(true);
            }
            mask
        });
        let expected = crate::deepseek_v4_oracle::shared_kv_attention(
            &query_values[token * dims.query_width..(token + 1) * dims.query_width],
            config.head_count,
            config.head_dim,
            &raw_rows,
            &compressed_values[..compressed_count * config.head_dim],
            mask.as_deref(),
            &sinks_values,
        )
        .unwrap();
        let actual = &actual[token * dims.query_width..(token + 1) * dims.query_width];
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let allowed = 8e-5 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= allowed,
                "packed sparse token {token} differs at {index}: {actual} vs {expected}, allowed {allowed}"
            );
        }
    }
}

#[test]
fn retained_packed_attention_preserves_ring_and_absolute_visibility() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };

    fn run_case(
        ctx: &MetalContext,
        kind: AttentionKind,
        start_position: u32,
        n_tokens: usize,
        checked_tokens: Option<&[usize]>,
    ) {
        let config = deepseek_v4_session_attention_config();
        let dims = config.checked().unwrap();
        let ratio = match kind {
            AttentionKind::SlidingWindow => 0,
            AttentionKind::CompressedSparse => 4,
            AttentionKind::HeavilyCompressed => 128,
        };
        let end_position = start_position as usize + n_tokens;
        let raw_value = |position: usize, dimension: usize| {
            let tag = (position * 29 + dimension * 11 + position / 7) % 137;
            (tag as f32 - 68.0) * 0.0027
                + if (position + dimension).is_multiple_of(31) {
                    0.043
                } else {
                    -0.009
                }
        };
        let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
        let mut prior_ring = vec![0.0; DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim];
        for position in 0..start_position as usize {
            let slot = position % DEEPSEEK_V4_LOCAL_WINDOW;
            for dimension in 0..config.head_dim {
                prior_ring[slot * config.head_dim + dimension] =
                    round_f16(raw_value(position, dimension));
            }
        }
        let prior_bits = prior_ring
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        let raw_cache = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&prior_bits),
            vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            GgmlType::F16,
        )
        .unwrap();
        let raw_cache_before_chunk = MetalTensor::zeros_f16(
            ctx,
            vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        )
        .unwrap();
        let raw_chunk =
            MetalTensor::zeros_f16(ctx, vec![config.head_dim as u64, n_tokens as u64]).unwrap();
        let new_raw = (start_position as usize..end_position)
            .flat_map(|position| {
                (0..config.head_dim).map(move |dimension| raw_value(position, dimension))
            })
            .collect::<Vec<_>>();
        let new_raw = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&new_raw),
            vec![config.head_dim as u64, n_tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let queries_values = (0..n_tokens * dims.query_width)
            .map(|index| {
                let token = index / dims.query_width;
                let within = index % dims.query_width;
                let head = within / config.head_dim;
                let dimension = within % config.head_dim;
                let tag =
                    (start_position as usize * 13 + token * 17 + head * 19 + dimension * 5) % 149;
                (tag as f32 - 74.0) * 0.0019
            })
            .collect::<Vec<_>>();
        let queries = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&queries_values),
            vec![dims.query_width as u64, n_tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let compressed_values = (0..DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS * config.head_dim)
            .map(|index| {
                let row = index / config.head_dim;
                let dimension = index % config.head_dim;
                let tag = (row * 37 + dimension * 7 + row / 3) % 139;
                round_f16((tag as f32 - 69.0) * 0.0023 - 0.007)
            })
            .collect::<Vec<_>>();
        let compressed_bits = compressed_values
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        let compressed = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&compressed_bits),
            vec![
                config.head_dim as u64,
                DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
            ],
            GgmlType::F16,
        )
        .unwrap();
        let sinks_values = (0..config.head_count)
            .map(|head| head as f32 * 0.009 - 0.27)
            .collect::<Vec<_>>();
        let sinks = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&sinks_values),
            vec![config.head_count as u64],
            GgmlType::F32,
        )
        .unwrap();
        let output =
            MetalTensor::zeros_f32(ctx, vec![dims.query_width as u64, n_tokens as u64]).unwrap();
        let final_compressed_count = end_position.checked_div(ratio).unwrap_or(0);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_copy_raw_ring_f16_bits(ctx, &encoder, &raw_cache, &raw_cache_before_chunk).unwrap();
        encode_publish_raw_chunk_f16(
            ctx,
            &encoder,
            &new_raw,
            &raw_chunk,
            &raw_cache,
            start_position,
            n_tokens,
            config.head_dim,
        )
        .unwrap();
        encode_packed_dense_sink_attention_f16(
            ctx,
            &encoder,
            &queries,
            &raw_chunk,
            &raw_cache_before_chunk,
            (final_compressed_count > 0).then_some(DeepSeekV4PublishedRows {
                cache: &compressed,
                count: final_compressed_count,
                capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
            }),
            &sinks,
            &output,
            kind,
            start_position,
            n_tokens,
        )
        .unwrap();
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(
            command.error().is_none(),
            "retained {kind:?} command failed: {:?}",
            command.error()
        );

        let actual = host_read_f32(&output, "retained packed attention").unwrap();
        let all_tokens = (0..n_tokens).collect::<Vec<_>>();
        for &token in checked_tokens.unwrap_or(&all_tokens) {
            assert!(token < n_tokens);
            let position = start_position as usize + token;
            let raw_start = (position + 1).saturating_sub(DEEPSEEK_V4_LOCAL_WINDOW);
            let raw_rows = (raw_start..=position)
                .flat_map(|logical_position| {
                    (0..config.head_dim)
                        .map(move |dimension| round_f16(raw_value(logical_position, dimension)))
                })
                .collect::<Vec<_>>();
            let compressed_count = (position + 1).checked_div(ratio).unwrap_or(0);
            let expected = crate::deepseek_v4_oracle::shared_kv_attention(
                &queries_values[token * dims.query_width..(token + 1) * dims.query_width],
                config.head_count,
                config.head_dim,
                &raw_rows,
                &compressed_values[..compressed_count * config.head_dim],
                None,
                &sinks_values,
            )
            .unwrap();
            let actual = &actual[token * dims.query_width..(token + 1) * dims.query_width];
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let allowed = 8e-5 * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= allowed,
                    "retained {kind:?} token {token} attention differs at {index}: {actual} vs {expected}, allowed {allowed}"
                );
            }
        }
    }

    run_case(&ctx, AttentionKind::SlidingWindow, 127, 4, None);
    run_case(&ctx, AttentionKind::CompressedSparse, 125, 8, None);
    run_case(&ctx, AttentionKind::HeavilyCompressed, 125, 8, None);
    const WIDE_CHUNK_TOKENS: usize = 512;

    run_case(
        &ctx,
        AttentionKind::CompressedSparse,
        128,
        128,
        Some(&[0, 1, 63, 127]),
    );
    run_case(&ctx, AttentionKind::CompressedSparse, 1_020, 4, None);
    run_case(
        &ctx,
        AttentionKind::CompressedSparse,
        2_044,
        4,
        Some(&[0, 3]),
    );
    run_case(
        &ctx,
        AttentionKind::SlidingWindow,
        129,
        WIDE_CHUNK_TOKENS,
        Some(&[0, 127, 128, 255, 384, 511]),
    );
    run_case(
        &ctx,
        AttentionKind::HeavilyCompressed,
        65_152,
        WIDE_CHUNK_TOKENS,
        Some(&[0, 383, 384, 510, 511]),
    );
}

#[test]
fn q8_token_axis_gemv_is_bitwise_singleton_equivalent() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    const N_IN: usize = 64;
    const N_OUT: usize = 7;
    const N_TOKENS: usize = 4;
    let mut weight_bytes = Vec::with_capacity(N_OUT * (N_IN / 32) * 34);
    for row in 0..N_OUT {
        for block in 0..N_IN / 32 {
            let scale = half::f16::from_f32(0.0075 + row as f32 * 0.0003);
            weight_bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
            for index in 0..32 {
                let quant = ((row * 19 + block * 11 + index * 7) % 101) as i8 - 50;
                weight_bytes.push(quant as u8);
            }
        }
    }
    let inputs = (0..N_TOKENS * N_IN)
        .map(|index| ((index * 13 + index / 9) % 89) as f32 * 0.013 - 0.51)
        .collect::<Vec<_>>();
    let weight = MetalTensor::from_bytes(
        &ctx,
        &weight_bytes,
        vec![N_IN as u64, N_OUT as u64],
        GgmlType::Q8_0,
    )
    .unwrap();
    let inputs = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&inputs),
        vec![N_IN as u64, N_TOKENS as u64],
        GgmlType::F32,
    )
    .unwrap();
    let packed = MetalTensor::zeros_f32(&ctx, vec![N_OUT as u64, N_TOKENS as u64]).unwrap();
    let singleton = MetalTensor::zeros_f32(&ctx, vec![N_OUT as u64, N_TOKENS as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    crate::metal::encode_mat_vec_q8_0_batch_f32(
        &ctx, &encoder, &weight, &inputs, &packed, N_IN, N_OUT, N_TOKENS,
    )
    .unwrap();
    for token in 0..N_TOKENS {
        let input = f32_row(
            &inputs,
            token,
            N_IN,
            vec![N_IN as u64],
            "singleton Q8 input",
        )
        .unwrap();
        let output = f32_row(
            &singleton,
            token,
            N_OUT,
            vec![N_OUT as u64],
            "singleton Q8 output",
        )
        .unwrap();
        crate::metal::encode_mat_vec_q8_0_f32(
            &ctx, &encoder, &weight, &input, &output, N_IN, N_OUT,
        )
        .unwrap();
    }
    encoder.end();
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert!(command.error().is_none());
    let packed = host_read_f32(&packed, "packed Q8 output").unwrap();
    let singleton = host_read_f32(&singleton, "singleton Q8 output").unwrap();
    assert_eq!(
        packed
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        singleton
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}
