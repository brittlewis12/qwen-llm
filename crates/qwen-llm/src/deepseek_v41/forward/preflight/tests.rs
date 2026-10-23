use super::*;

fn signals() -> metal::MetalMemorySignals {
    metal::MetalMemorySignals {
        recommended_max_bytes: 1 << 50,
        current_allocated_bytes: 0,
        process_limit_remaining_bytes: Some(0),
        host_budget: Some(metal::HostMemoryBudget {
            physical_bytes: 1 << 50,
            conservative_available_bytes: 1 << 50,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
    }
}

fn storage() -> ResidentStoragePlan {
    ResidentStoragePlan {
        page_size: 16384,
        max_buffer_length: 1 << 40,
        bytes: gguf::ByteCounters::default(),
        windows: vec![],
        metal_view_bytes: 100,
        private_shadow_bytes: 200,
        shared_eof_window_count: 0,
        private_window_count: 0,
    }
}

fn schedule() -> gguf::Schedule {
    gguf::Schedule {
        layers: 4,
        kv_source_layers: vec![0, 2],
        index_source_layers: vec![0, 2, 3],
        compression_ratios: vec![2, 2, 1, 1],
        candidate_source_layer: Some(2),
    }
}

fn hash(count: usize) -> engram::HashConfig {
    let mut hash = engram::HashConfig::default();
    hash.layers.truncate(count);
    for (i, layer) in hash.layers.iter_mut().enumerate() {
        layer.layer = i;
    }
    hash
}

fn price(bytes: u64) -> Result<u64> {
    if bytes == 0 {
        return Ok(0);
    }
    bytes
        .checked_add(16383)
        .map(|n| (n / 16384) * 16384 + 1024)
        .ok_or_else(|| "test pricing overflow".into())
}

#[test]
fn custom_schedule_engram_variants_and_shared_ledgers() {
    let c = super::super::toy::config();
    let schedule = schedule();
    for count in 0..=2 {
        let hash = hash(count);
        let projections = vec![(6144, 5 * c.hidden); count];
        for half in [false, true] {
            let (request, full, ced) = runtime_from_parts(
                &c,
                &schedule,
                &hash,
                &projections,
                16,
                24,
                &[
                    PrefillExecution::FullForward,
                    PrefillExecution::ShrinkingCed,
                ],
                half,
                &price,
            )
            .unwrap();
            assert_eq!(
                request,
                memory::estimate_gpu_step(&c, 24, &schedule, count).unwrap()
            );
            let widths = gpu_step::chunk_widths_from_geometry(&c, 4, &projections, 24, 16).unwrap();
            let expected = widths
                .into_iter()
                .map(|width| price((width * 16 * 4) as u64).unwrap())
                .sum::<u64>()
                + prefill::half_scratch_budget_with(&c, 16, half, &price).unwrap();
            assert_eq!(full.0, expected);
            assert_eq!(
                full.1,
                (6144 * usize::from(count > 0) + 4 * c.hidden) as u64 * 16 * 8
                    + prefill::chunk_hash_cpu_bytes(count, 16).unwrap()
            );
            let plan = ced_plan::CedPlan::shrinking(
                &c,
                &schedule,
                &hash,
                &ced_plan::CedRequest::cold(16, 24),
            )
            .unwrap();
            let ced = ced.unwrap();
            assert_eq!(
                ced,
                ced::estimate_from_parts(&c, 4, &projections, count, 24, &plan, half, &price)
                    .unwrap()
            );
            assert_eq!(ced.retained_selected_bytes, plan.memory.selected_bytes);
            assert_eq!(ced.retained_candidate_bytes, plan.memory.candidate_bytes);
            assert_eq!(ced.process_ru_maxrss_before_bytes, None);
            assert_eq!(ced.process_ru_maxrss_after_bytes, None);
            assert!(ced.observation_method.contains("no execution"));
        }
    }
}

#[test]
fn policy_repetitions_are_peaks_not_sums() {
    let c = Config::default();
    let schedule = gguf::Schedule::default();
    let hash = engram::HashConfig::default();
    let projections = vec![(6144, 5 * c.hidden); 2];
    let policies = [
        PrefillExecution::FullForward,
        PrefillExecution::ShrinkingCed,
    ];
    let estimate = |policies: &[PrefillExecution]| {
        let (request, full, ced) = runtime_from_parts(
            &c,
            &schedule,
            &hash,
            &projections,
            8192,
            8448,
            policies,
            true,
            &price,
        )
        .unwrap();
        combine(storage(), request, full, ced, 400, 334_226_432, signals()).unwrap()
    };
    let both = estimate(&policies);
    let repeated = estimate(&policies.repeat(4));
    assert_eq!(
        both.request,
        c.estimate_request_memory_with_gpu_step(8448).unwrap()
    );
    assert_eq!(both.runtime_metal_bytes, repeated.runtime_metal_bytes);
    assert_eq!(both.runtime_cpu_bytes, repeated.runtime_cpu_bytes);
    assert_eq!(
        both.runtime_metal_bytes,
        estimate(&[PrefillExecution::ShrinkingCed]).runtime_metal_bytes
    );
    let ced = both.ced.as_ref().unwrap();
    assert_eq!(
        both.runtime_metal_bytes,
        both.request.metal_bytes
            + both
                .full_incremental_metal_bytes
                .max(ced.incremental_metal_bytes)
    );
    assert_eq!(
        both.runtime_cpu_bytes,
        both.request.cpu_bytes
            + both
                .full_incremental_cpu_bytes
                .max(ced.incremental_cpu_bytes)
    );
    assert_eq!(both.required_metal_bytes, 100 + both.runtime_metal_bytes);
    assert_eq!(
        both.required_cpu_bytes,
        200 + 400 + both.runtime_cpu_bytes + 334_226_432
    );
    assert_eq!(
        both.admission.reason,
        metal::MetalMemoryAdmissionReason::AdmittedWithHostBudget
    );
    assert!(estimate(&[PrefillExecution::FullForward]).ced.is_none());
}

#[test]
fn combined_budget_keeps_the_original_pair_sum_and_overflow_regression() {
    let request = RequestMemoryEstimate {
        cpu_persistent_bytes: 500,
        cpu_transient_bytes: 0,
        cpu_bytes: 500,
        metal_bytes: 600,
        total_bytes: 1100,
    };
    let capture = 132_899_840 + 192 * 1024 * 1024;
    let cold = combine(storage(), request, (0, 0), None, 400, capture, signals()).unwrap();
    assert_eq!(
        (cold.required_metal_bytes, cold.required_cpu_bytes),
        (700, 1100 + capture)
    );
    let mut resident = storage();
    resident.metal_view_bytes = u64::MAX;
    assert!(combine(resident, request, (0, 0), None, 400, capture, signals()).is_err());
    assert!(
        binding_vectors_bytes(
            &Config {
                hidden: usize::MAX,
                ..Config::default()
            },
            40
        )
        .is_err()
    );
    assert!(binding_vectors_bytes(&Config::default(), usize::MAX).is_err());
}

#[test]
fn checked_geometry_context_and_pricing_fail_closed() {
    let c = super::super::toy::config();
    let mut hash = hash(0);
    hash.max_position = 32;
    let full = [PrefillExecution::FullForward];
    let run = |c: &Config,
               s: &gguf::Schedule,
               h: &engram::HashConfig,
               n,
               capacity,
               policies: &[PrefillExecution]| {
        runtime_from_parts(c, s, h, &[], n, capacity, policies, true, &price)
    };
    assert!(run(&c, &schedule(), &hash, 0, 24, &full).is_err());
    assert!(run(&c, &schedule(), &hash, 25, 24, &full).is_err());
    assert!(run(&c, &schedule(), &hash, 16, 33, &full).is_err());
    assert!(run(&c, &schedule(), &hash, 16, 24, &[]).is_err());
    let mut invalid = schedule();
    invalid.compression_ratios.pop();
    assert!(run(&c, &invalid, &hash, 16, 24, &full).is_err());
    let mut invalid_hash = hash.clone();
    invalid_hash.layers = engram::HashConfig::default().layers;
    assert!(run(&c, &schedule(), &invalid_hash, 16, 24, &full).is_err());
    for c in [
        Config {
            hidden: usize::MAX,
            ..c.clone()
        },
        Config {
            head_dim: usize::MAX,
            ..c.clone()
        },
        Config {
            candidate_block: 0,
            ..c.clone()
        },
    ] {
        assert!(run(&c, &schedule(), &hash, 16, 24, &full).is_err());
        assert!(gpu_step::chunk_widths_from_geometry(&c, 4, &[], 24, 16).is_err());
        assert!(gpu_step::ced::decoder_chunk_widths_from_geometry(&c, 4, 24, 16).is_err());
    }
    assert!(
        runtime_from_parts(&c, &schedule(), &hash, &[], 16, 24, &full, true, &|_| Err(
            "pricing refused".into()
        ))
        .is_err()
    );
    let scalar = run(&c, &schedule(), &hash, 1, 24, &full).unwrap();
    assert_eq!(scalar.1, (0, 0));
    let overflow =
        combine(storage(), scalar.0, (0, 0), None, 400, u64::MAX, signals()).unwrap_err();
    assert_eq!(
        overflow.as_memory_admission().unwrap().reason(),
        metal::MetalMemoryAdmissionReason::RequiredBytesOverflow
    );
}

#[test]
fn unsupported_ced_geometry_uses_the_runtime_full_fallback() {
    let c = super::super::toy::config();
    let mut schedule = schedule();
    schedule.candidate_source_layer = None;
    let (_, full, ced) = runtime_from_parts(
        &c,
        &schedule,
        &hash(0),
        &[],
        16,
        24,
        &[PrefillExecution::ShrinkingCed],
        false,
        &price,
    )
    .unwrap();
    assert!(ced.is_none());
    assert!(full.0 > 0);
}

#[test]
fn default_half_scratch_is_202_mib_without_gpu_family_guesses() {
    assert_eq!(
        prefill::half_scratch_budget_with(&Config::default(), 2048, true, &Ok).unwrap(),
        202 * 1024 * 1024
    );
    assert_eq!(
        prefill::half_scratch_budget_with(&Config::default(), 2048, false, &Ok).unwrap(),
        0
    );
    assert!(
        prefill::half_scratch_budget_with(
            &Config {
                hidden: usize::MAX,
                ..Config::default()
            },
            2048,
            true,
            &Ok
        )
        .is_err()
    );
}
