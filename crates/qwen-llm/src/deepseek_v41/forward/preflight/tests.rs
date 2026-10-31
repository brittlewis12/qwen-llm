use super::*;

fn binding_metal_bytes(
    dtypes: impl IntoIterator<Item = GgmlType>,
    price: &impl Fn(u64) -> Result<u64>,
) -> Result<u64> {
    let dtypes: Vec<_> = dtypes.into_iter().collect();
    BindingMemoryPlan::new(
        (0..dtypes.len()).map(|i| binding::LinearSpec {
            name: i.to_string(),
            input: 256,
            output: 1,
        }),
        |spec| Ok(dtypes[spec.name.parse::<usize>().unwrap()]),
        false,
        price,
    )
    .map(|p| p.control_metal_bytes())
}

#[test]
fn loader_control_cpu_budget_is_explicit_checked_and_never_charges_shared_weights() {
    let controls = gguf::ControlAccounting {
        payload_bytes_read: 517_952,
        cold_cpu_bytes: 3_105_632,
        retained_cpu_bytes: 519_648,
        source_bookkeeping_bytes: 4578,
    };
    assert_eq!(loader_control_cpu_bytes(controls).unwrap(), 3_631_938);
    let mut normalization = controls;
    normalization.cold_cpu_bytes = 4_000_000;
    assert_eq!(loader_control_cpu_bytes(normalization).unwrap(), 4_524_226);
    for bad in [
        gguf::ControlAccounting {
            payload_bytes_read: u64::MAX,
            ..controls
        },
        gguf::ControlAccounting {
            retained_cpu_bytes: u64::MAX,
            ..controls
        },
        gguf::ControlAccounting {
            source_bookkeeping_bytes: u64::MAX,
            ..controls
        },
    ] {
        assert!(loader_control_cpu_bytes(bad).is_err());
    }
    assert_eq!(
        loader_control_cpu_bytes(gguf::ControlAccounting::default()).unwrap(),
        0
    );
}

fn combine(
    resident_storage: ResidentStoragePlan,
    request: RequestMemoryEstimate,
    full: (u64, u64),
    ced: Option<CedMemoryEstimate>,
    binding_cpu_bytes: u64,
    binding_metal_bytes: u64,
    caller_cpu_bytes: u64,
    signals: metal::MetalMemorySignals,
) -> std::result::Result<ColdPrefillPreflight, LoadError> {
    combine_plan(
        resident_storage,
        request,
        full,
        ced,
        binding_cpu_bytes,
        binding_metal_bytes,
        caller_cpu_bytes,
    )
    .map_err(|error| error.with_signals(signals))?
    .into_strict(signals)
}

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
        soft_coalescing_target: None,
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
fn cold_candidates_select_before_import_and_freeze_geometry_without_repricing_cpu_as_weights() {
    let request = RequestMemoryEstimate {
        cpu_persistent_bytes: 10,
        cpu_transient_bytes: 20,
        cpu_bytes: 30,
        metal_bytes: 40,
        total_bytes: 70,
    };
    let make = |rows, scratch| {
        let mut plan = combine_plan(storage(), request, (scratch, 16), None, 17, 50, 9).unwrap();
        plan.full_chunk_rows = rows;
        plan.geometry = PrefillGeometry {
            start: 0,
            tokens: 4097,
            selected_rows: rows,
        };
        plan.capacity = 8192;
        plan
    };
    let mut preferred = make(4096, 9000);
    preferred.alternatives.push(make(2048, 4000));
    let mut observed = signals();
    observed.recommended_max_bytes = 8000;
    observed
        .host_budget
        .as_mut()
        .unwrap()
        .conservative_available_bytes = 1000;
    let (selected, admission) = preferred.select_load(observed).unwrap();
    assert!(admission.load_admission.is_admitted());
    assert_eq!(selected.full_chunk_rows, 2048);
    assert_eq!(selected.capacity, 8192);
    assert!(selected.alternatives.is_empty());
    assert_eq!(selected.required_cpu_bytes, 200 + 17 + 30 + 16 + 9);
    assert!(selected.required_metal_bytes > 1000);
    observed.current_allocated_bytes = selected.load_metal_bytes;
    selected.assess_runtime(observed).unwrap();
    observed.recommended_max_bytes = 0;
    assert!(preferred.select_load(observed).is_err());
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
                prefill::DEFAULT_CHUNK_ROWS,
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
                + prefill::half_scratch_budget_with(&c, 16, half, &price).unwrap()
                + if half {
                    price(prefill::dense_preweight_bytes(&c).unwrap()).unwrap()
                } else {
                    0
                };
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
                ced::estimate_from_parts(
                    &c,
                    4,
                    &projections,
                    count,
                    24,
                    &plan,
                    half,
                    crate::deepseek_v41::forward::gpu_step::ced::encoder_pipeline_priced(),
                    &price
                )
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
            prefill::DEFAULT_CHUNK_ROWS,
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
    assert_eq!(both.load_metal_bytes, 900);
    assert_eq!(both.load_cpu_bytes, 200 + 400 + 334_226_432);
    assert_eq!(
        both.runtime_gate_cpu_bytes,
        both.runtime_cpu_bytes + 400 + 334_226_432
    );
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
        runtime_from_parts(
            c,
            s,
            h,
            &[],
            n,
            capacity,
            policies,
            prefill::DEFAULT_CHUNK_ROWS,
            true,
            &price,
        )
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
        runtime_from_parts(
            &c,
            &schedule(),
            &hash,
            &[],
            16,
            24,
            &full,
            prefill::DEFAULT_CHUNK_ROWS,
            true,
            &|_| Err("pricing refused".into())
        )
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
        prefill::DEFAULT_CHUNK_ROWS,
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

fn staged_signals() -> metal::MetalMemorySignals {
    metal::MetalMemorySignals {
        recommended_max_bytes: 1000,
        current_allocated_bytes: 0,
        process_limit_remaining_bytes: Some(0),
        host_budget: Some(metal::HostMemoryBudget {
            physical_bytes: 1000,
            // Shared load fits; the whole peak's 50-byte CPU payload does not.
            conservative_available_bytes: 25,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
    }
}

fn staged_plan() -> ColdPrefillPlan {
    let mut resident = storage();
    resident.metal_view_bytes = 600;
    resident.private_shadow_bytes = 0;
    combine_plan(
        resident,
        RequestMemoryEstimate {
            cpu_persistent_bytes: 0,
            cpu_transient_bytes: 50,
            cpu_bytes: 50,
            metal_bytes: 100,
            total_bytes: 150,
        },
        (0, 0),
        None,
        0,
        50,
        0,
    )
    .unwrap()
}

#[test]
fn staged_load_fits_while_strict_peak_remains_refused() {
    let plan = staged_plan();
    let observed = staged_signals();
    assert_eq!((plan.load_metal_bytes, plan.load_cpu_bytes), (650, 0));
    assert_eq!(
        (plan.runtime_metal_bytes, plan.runtime_gate_cpu_bytes),
        (100, 50)
    );
    assert_eq!(plan.required_metal_bytes + plan.required_cpu_bytes, 800);
    let load = plan.assess_load(observed).unwrap();
    assert_eq!(load.peak_capacity.required_bytes, 800);
    assert_eq!(load.peak_capacity.signals, observed);
    assert_eq!(load.load_admission.required_bytes, Some(650));
    assert!(load.load_admission.is_admitted());
    assert_eq!(
        load.whole_peak_now.reason,
        metal::MetalMemoryAdmissionReason::HostAvailabilityInsufficient
    );
    let strict = plan.clone().into_strict(observed).unwrap_err();
    assert_eq!(
        strict.as_memory_admission(),
        load.whole_peak_now.refusal().as_ref()
    );
    let strict = plan.into_strict(signals()).unwrap();
    assert_eq!(
        strict.required_metal_bytes,
        strict.plan.required_metal_bytes
    );
    assert!(strict.admission.is_admitted());
}

#[test]
fn staged_runtime_uses_explicit_postload_signals_and_refuses_before_state_creation() {
    let plan = staged_plan();
    let load = plan.assess_load(staged_signals()).unwrap();
    let mut after = staged_signals();
    after.current_allocated_bytes = 650;
    after
        .host_budget
        .as_mut()
        .unwrap()
        .conservative_available_bytes = 50;
    let runtime = plan.assess_runtime(after).unwrap();
    assert_eq!(runtime.required_bytes, Some(150));
    assert_eq!(runtime.signals, after);
    assert_eq!(runtime.signals.host_remaining_bytes(), Some(50));
    assert_ne!(runtime.signals, load.load_admission.signals);

    after
        .host_budget
        .as_mut()
        .unwrap()
        .conservative_available_bytes = 49;
    let states = std::cell::Cell::new(0);
    let commands = std::cell::Cell::new(0);
    let denied = plan
        .assess_runtime(after)
        .map(|_| {
            states.set(states.get() + 1);
            commands.set(commands.get() + 1);
        })
        .unwrap_err();
    assert_eq!(
        denied.as_memory_admission().unwrap().reason(),
        metal::MetalMemoryAdmissionReason::HostAvailabilityInsufficient
    );
    assert_eq!(states.get(), 0);
    assert_eq!(commands.get(), 0);
    // The earlier load assessment cannot authorize a subsequent arm either.
    assert!(load.load_admission.is_admitted());
    assert!(plan.assess_runtime(after).is_err());
}

#[test]
fn staged_capacity_checks_constraints_hidden_by_whole_peak_availability() {
    use metal::MetalMemoryAdmissionReason as R;
    let plan = staged_plan();
    let mut process = staged_signals();
    process.process_limit_remaining_bytes = Some(750);
    let mut device = staged_signals();
    device.recommended_max_bytes = 749;
    let mut ownership = staged_signals();
    // At exact physical capacity, one byte of existing gap or shadow blocks the peak.
    ownership.host_budget.as_mut().unwrap().physical_bytes =
        plan.required_metal_bytes + plan.required_cpu_bytes;
    ownership.host_budget.as_mut().unwrap().external_gap_bytes = 1;
    let mut shadow = staged_signals();
    shadow.host_budget.as_mut().unwrap().physical_bytes =
        plan.required_metal_bytes + plan.required_cpu_bytes;
    shadow.host_budget.as_mut().unwrap().private_shadow_bytes = 1;
    for (signals, reason) in [
        (process, R::ProcessInsufficient),
        (device, R::WorkingSetInsufficient),
        (ownership, R::HostInsufficient),
        (shadow, R::HostInsufficient),
    ] {
        assert!(
            metal::evaluate_metal_memory_admission_with_cpu_bytes(
                plan.load_metal_bytes,
                plan.load_cpu_bytes,
                0,
                signals,
                true,
            )
            .is_admitted()
        );
        let imports = std::cell::Cell::new(0);
        let error = plan
            .assess_load(signals)
            .map(|_| imports.set(1))
            .unwrap_err();
        assert_eq!(error.as_memory_admission().unwrap().reason(), reason);
        assert_eq!(imports.get(), 0);
    }
    process.process_limit_remaining_bytes = Some(800);
    assert!(plan.assess_load(process).is_ok());

    let mut short_load = staged_signals();
    short_load
        .host_budget
        .as_mut()
        .unwrap()
        .private_shadow_bytes = 26;
    assert_eq!(
        plan.assess_load(short_load)
            .unwrap_err()
            .as_memory_admission()
            .unwrap()
            .reason(),
        R::HostAvailabilityInsufficient
    );
}

#[test]
fn staged_runtime_recharges_binding_and_caller_cpu_but_not_loaded_metal() {
    let mut resident = storage();
    resident.metal_view_bytes = 600;
    resident.private_shadow_bytes = 0;
    let plan = combine_plan(
        resident,
        RequestMemoryEstimate {
            cpu_persistent_bytes: 0,
            cpu_transient_bytes: 0,
            cpu_bytes: 0,
            metal_bytes: 100,
            total_bytes: 100,
        },
        (0, 0),
        None,
        30,
        0,
        20,
    )
    .unwrap();
    assert_eq!((plan.load_metal_bytes, plan.load_cpu_bytes), (600, 50));
    assert_eq!(plan.runtime_cpu_bytes, 0);
    assert_eq!(plan.runtime_gate_cpu_bytes, 50);
    assert_eq!(plan.required_cpu_bytes, 50);
    let mut observed = staged_signals();
    observed
        .host_budget
        .as_mut()
        .unwrap()
        .conservative_available_bytes = 50;
    assert!(plan.assess_load(observed).is_ok());

    let mut after = staged_signals();
    after.current_allocated_bytes = 600;
    after
        .host_budget
        .as_mut()
        .unwrap()
        .conservative_available_bytes = 50;
    assert_eq!(
        plan.assess_runtime(after).unwrap().required_bytes,
        Some(150)
    );
    after.process_limit_remaining_bytes = Some(149);
    assert_eq!(
        plan.assess_runtime(after)
            .unwrap_err()
            .as_memory_admission()
            .unwrap()
            .reason(),
        metal::MetalMemoryAdmissionReason::ProcessInsufficient
    );
    after.process_limit_remaining_bytes = Some(0);
    after
        .host_budget
        .as_mut()
        .unwrap()
        .conservative_available_bytes = 49;
    assert!(plan.assess_runtime(after).is_err());
}

#[test]
fn staged_private_shadows_and_uncovered_imports_remain_charged_exactly_once() {
    let mut plan = staged_plan();
    // Shift ten bytes from shared views to new private-shadow exposure.
    plan.resident_storage.metal_view_bytes = 590;
    plan.resident_storage.private_shadow_bytes = 10;
    let plan = combine_plan(plan.resident_storage, plan.request, (0, 0), None, 0, 50, 0).unwrap();
    assert_eq!((plan.load_metal_bytes, plan.load_cpu_bytes), (640, 10));
    assert_eq!(plan.required_metal_bytes + plan.required_cpu_bytes, 800);
    assert!(plan.assess_load(staged_signals()).is_ok());

    let mut after = staged_signals();
    after.current_allocated_bytes = 540;
    let host = after.host_budget.as_mut().unwrap();
    host.external_gap_bytes = 100;
    host.private_shadow_bytes = 10;
    // Live device usage, shadow and runtime demand exactly fill physical capacity.
    host.physical_bytes = plan.required_metal_bytes + plan.required_cpu_bytes;
    host.conservative_available_bytes = 160;
    assert_eq!(after.effective_device_usage(), Some(640));
    assert_eq!(after.host_remaining_bytes(), Some(150));
    assert_eq!(
        plan.assess_runtime(after).unwrap().required_bytes,
        Some(150)
    );
    after.host_budget.as_mut().unwrap().external_gap_bytes += 1;
    assert_eq!(
        plan.assess_runtime(after)
            .unwrap_err()
            .as_memory_admission()
            .unwrap()
            .reason(),
        metal::MetalMemoryAdmissionReason::HostInsufficient
    );

    let mut original = staged_plan();
    original.resident_storage.private_shadow_bytes = 1;
    let private = combine_plan(
        original.resident_storage,
        original.request,
        (0, 0),
        None,
        0,
        50,
        0,
    )
    .unwrap();
    assert_eq!(private.load_cpu_bytes, 1);
    assert_eq!(
        private.required_metal_bytes + private.required_cpu_bytes,
        801
    );
    let mut exact_capacity = staged_signals();
    exact_capacity.host_budget.as_mut().unwrap().physical_bytes = 800;
    assert!(staged_plan().assess_load(exact_capacity).is_ok());
    assert_eq!(
        private
            .assess_load(exact_capacity)
            .unwrap_err()
            .as_memory_admission()
            .unwrap()
            .reason(),
        metal::MetalMemoryAdmissionReason::HostInsufficient
    );
}

#[test]
fn staged_assessments_refuse_invalid_telemetry_and_runtime_combined_overflow() {
    let plan = staged_plan();
    let mut missing_host = staged_signals();
    missing_host.host_budget = None;
    let mut missing_process = staged_signals();
    missing_process.process_limit_remaining_bytes = None;
    let mut invalid_host = staged_signals();
    invalid_host.host_budget.as_mut().unwrap().physical_bytes = 0;
    let mut invalid_available = staged_signals();
    invalid_available
        .host_budget
        .as_mut()
        .unwrap()
        .conservative_available_bytes = 1001;
    let mut invalid_device = staged_signals();
    invalid_device.recommended_max_bytes = 0;
    for signals in [
        missing_host,
        missing_process,
        invalid_host,
        invalid_available,
        invalid_device,
    ] {
        assert!(
            plan.assess_load(signals)
                .unwrap_err()
                .as_memory_admission()
                .is_some()
        );
        assert!(
            plan.assess_runtime(signals)
                .unwrap_err()
                .as_memory_admission()
                .is_some()
        );
    }
    let mut overflow = plan.clone();
    overflow.runtime_metal_bytes = u64::MAX;
    assert_eq!(
        overflow
            .assess_runtime(staged_signals())
            .unwrap_err()
            .as_memory_admission()
            .unwrap()
            .reason(),
        metal::MetalMemoryAdmissionReason::RequiredBytesOverflow
    );
    overflow.required_cpu_bytes = u64::MAX;
    assert_eq!(
        overflow
            .assess_load(staged_signals())
            .unwrap_err()
            .as_memory_admission()
            .unwrap()
            .reason(),
        metal::MetalMemoryAdmissionReason::RequiredBytesOverflow
    );
}

#[test]
fn staged_iq3_views_exceed_cpu_availability_but_still_require_device_capacity() {
    let mut original = staged_plan();
    let gib = 1u64 << 30;
    original.resident_storage.metal_view_bytes = 217_414 * gib / 1000;
    let plan = combine_plan(
        original.resident_storage,
        original.request,
        (0, 0),
        None,
        0,
        50,
        0,
    )
    .unwrap();
    let signals = metal::MetalMemorySignals {
        recommended_max_bytes: 256 * gib,
        current_allocated_bytes: 0,
        process_limit_remaining_bytes: Some(0),
        host_budget: Some(metal::HostMemoryBudget {
            physical_bytes: 256 * gib,
            conservative_available_bytes: 156 * gib,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
    };
    let imports = std::cell::Cell::new(0);
    let load = plan
        .assess_load(signals)
        .map(|load| {
            imports.set(imports.get() + 1);
            load
        })
        .unwrap();
    assert!(load.load_admission.is_admitted());
    assert!(load.whole_peak_now.is_admitted());
    assert!(
        plan.clone()
            .into_strict(signals)
            .unwrap()
            .admission
            .is_admitted()
    );
    assert_eq!(imports.get(), 1);

    let mut short_device = signals;
    short_device.recommended_max_bytes = plan.required_metal_bytes - 1;
    let error = plan
        .assess_load(short_device)
        .map(|_| imports.set(imports.get() + 1))
        .unwrap_err();
    assert_eq!(
        error.as_memory_admission().unwrap().reason(),
        metal::MetalMemoryAdmissionReason::WorkingSetInsufficient
    );
    assert_eq!(imports.get(), 1);
}
