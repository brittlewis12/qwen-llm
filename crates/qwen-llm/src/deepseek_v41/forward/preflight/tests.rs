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
        combine(
            storage(),
            request,
            full,
            ced,
            400,
            800,
            334_226_432,
            signals(),
        )
        .unwrap()
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
    assert_eq!(both.binding_metal_bytes, 800);
    assert_eq!(both.required_metal_bytes, 900 + both.runtime_metal_bytes);
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
    let cold = combine(storage(), request, (0, 0), None, 400, 0, capture, signals()).unwrap();
    assert_eq!(
        (cold.required_metal_bytes, cold.required_cpu_bytes),
        (700, 1100 + capture)
    );
    let mut resident = storage();
    resident.metal_view_bytes = u64::MAX;
    assert!(combine(resident, request, (0, 0), None, 400, 0, capture, signals()).is_err());
    assert!(
        combine(
            storage(),
            request,
            (0, 0),
            None,
            400,
            u64::MAX,
            0,
            signals()
        )
        .is_err()
    );
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
    let overflow = combine(
        storage(),
        scalar.0,
        (0, 0),
        None,
        400,
        0,
        u64::MAX,
        signals(),
    )
    .unwrap_err();
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

#[test]
fn binding_controls_price_each_quantized_resource_and_skip_direct_types() {
    let quantized = [
        GgmlType::Q2_0,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q8_0,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::IQ2_S,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_S,
        GgmlType::IQ4_NL,
        GgmlType::IQ4_XS,
        GgmlType::IQ2_XS,
        GgmlType::MXFP4,
    ];
    let direct = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::IQ2_XXS,
        GgmlType::IQ1_S,
        GgmlType::IQ1_M,
    ];
    let calls = std::cell::Cell::new(0);
    let price_control = |bytes| {
        assert_eq!(bytes, 8);
        calls.set(calls.get() + 1);
        Ok(16_384 + 1024)
    };
    assert_eq!(
        binding_metal_bytes(quantized.into_iter().chain(direct), &price_control).unwrap(),
        16 * (16_384 + 1024)
    );
    assert_eq!(calls.get(), 16);
    assert_eq!(
        binding_metal_bytes(direct, &|_| panic!("direct linear allocated a control")).unwrap(),
        0
    );
    for dtype in [
        GgmlType::Unknown,
        GgmlType::I32,
        GgmlType::Q8_1,
        GgmlType::Q5_0,
    ] {
        assert!(binding_metal_bytes([dtype], &Ok).is_err());
    }
    assert!(binding_metal_bytes([GgmlType::Q8_0], &|_| Err("pricing refused".into())).is_err());
    assert!(binding_metal_bytes([GgmlType::Q8_0; 2], &|_| Ok(u64::MAX)).is_err());
}

#[test]
fn ratio_two_256_layer_binding_controls_cost_64_mib_plus_one_page() {
    let schedule = gguf::Schedule {
        layers: 256,
        kv_source_layers: (0..256).collect(),
        index_source_layers: (0..256).collect(),
        compression_ratios: vec![2; 256],
        candidate_source_layer: None,
    };
    schedule.validate().unwrap();
    let specs: Vec<_> =
        binding::linear_inventory(&Config::default(), &schedule, &hash(0)).collect();
    // Per layer: five attention, two compressor, three indexer, two HC,
    // one router and three shared experts. The head is one more allocation.
    assert_eq!(specs.len(), 4097);
    let unique: std::collections::BTreeSet<_> = specs.iter().map(|s| &s.name).collect();
    assert_eq!(unique.len(), 4097);
    let calls = std::cell::Cell::new(0);
    let bytes = binding_metal_bytes(specs.iter().map(|_| GgmlType::Q8_0), &|n| {
        assert_eq!(n, 8);
        calls.set(calls.get() + 1);
        Ok(16_384)
    })
    .unwrap();
    assert_eq!(calls.get(), 4097);
    assert_eq!(bytes, 64 * 1024 * 1024 + 16_384);
    assert!(bytes > (4097u64 * 8).next_multiple_of(16_384));
}

#[test]
fn binding_inventory_has_exact_schedule_roles_and_one_grouped_output_control() {
    let c = Config::default();
    let specs: std::collections::BTreeMap<_, _> =
        binding::linear_inventory(&c, &schedule(), &hash(2))
            .map(|s| (s.name, (s.input, s.output)))
            .collect();
    let base = [
        "attn.wq_a.weight",
        "attn.wq_b.weight",
        "attn.wkv.weight",
        "attn.wo_a.weight",
        "attn.wo_b.weight",
        "hc_attn_fn",
        "hc_ffn_fn",
        "ffn.gate.weight",
        "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w3.weight",
        "ffn.shared_experts.w2.weight",
    ];
    let mut expected = std::collections::BTreeSet::from(["head.weight".to_string()]);
    for layer in 0..4 {
        expected.extend(base.iter().map(|tail| format!("layers.{layer}.{tail}")));
    }
    for (layer, tails) in [
        (
            0,
            vec![
                "attn.compressor.wkv.weight",
                "attn.compressor.wgate.weight",
                "attn.indexer.wk.weight",
                "attn.indexer.wq_b.weight",
                "attn.indexer.weights_proj.weight",
                "engram.wkv.weight",
            ],
        ),
        (1, vec!["engram.wkv.weight"]),
        (
            2,
            vec![
                "attn.compressor.wkv.weight",
                "attn.indexer.wk.weight",
                "attn.indexer.wq_b.weight",
                "attn.indexer.weights_proj.weight",
            ],
        ),
        (
            3,
            vec![
                "attn.indexer.wq_b.weight",
                "attn.indexer.weights_proj.weight",
            ],
        ),
    ] {
        expected.extend(
            tails
                .into_iter()
                .map(|tail| format!("layers.{layer}.{tail}")),
        );
    }
    assert_eq!(
        specs
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        expected
    );
    assert_eq!(specs.len(), 58);
    assert_eq!(specs["layers.0.attn.wo_a.weight"], (4096, 8192));
    assert_eq!(specs["layers.1.engram.wkv.weight"], (6144, 25_600));
    assert_eq!(specs["layers.0.hc_attn_fn"], (20_480, 24));
    assert_eq!(
        binding_metal_bytes(specs.values().map(|_| GgmlType::Q8_0), &|_| Ok(16_384)).unwrap(),
        58 * 16_384
    );
}
