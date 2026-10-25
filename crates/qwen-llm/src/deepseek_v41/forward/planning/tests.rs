use super::*;
use crate::metal::HostMemoryBudget;

mod metal;

fn signals() -> MetalMemorySignals {
    MetalMemorySignals {
        host_budget: Some(HostMemoryBudget {
            physical_bytes: 100_000,
            conservative_available_bytes: 100_000,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
        recommended_max_bytes: 10_000,
        current_allocated_bytes: 1_000,
        process_limit_remaining_bytes: Some(100_000),
    }
}

#[test]
fn candidate_selection_is_pure_and_smaller_geometry_precedes_any_allocation() {
    let mut prices = vec![];
    let state_position = 71;
    let allocations = std::cell::Cell::new(0);
    let (rows, cost) = select(
        4096,
        None,
        4096,
        Cost {
            metal: 1_000,
            cpu: 99,
        },
        signals(),
        |rows| {
            prices.push(rows);
            assert_eq!(allocations.get(), 0);
            Ok(Cost {
                metal: rows as u64 * 2,
                cpu: rows as u64,
            })
        },
    )
    .unwrap();
    assert_eq!(prices, [4096, 2048]);
    assert_eq!(rows, 2048);
    assert_eq!(
        cost,
        Cost {
            metal: 5096,
            cpu: 2147
        }
    );
    assert_eq!(state_position, 71);
    allocations.set(1); // Encoding/allocation is a separate caller-owned phase.
}

#[test]
fn cold_peak_reserve_and_live_capacity_cost_are_not_erased_by_smaller_selection() {
    for (base, expected) in [
        (Cost { metal: 0, cpu: 0 }, Some(4096)),
        (
            Cost {
                metal: 2000,
                cpu: 0,
            },
            Some(2048),
        ),
        (
            Cost {
                metal: 6000,
                cpu: 0,
            },
            None,
        ),
        (
            Cost {
                metal: 0,
                cpu: 100_000,
            },
            None,
        ),
    ] {
        let result = select(4096, None, 5000, base, signals(), |rows| {
            Ok(Cost {
                metal: rows as u64 * 2,
                cpu: rows as u64,
            })
        });
        assert_eq!(result.ok().map(|(rows, _)| rows), expected);
    }
}

#[test]
fn frozen_geometry_refuses_without_retry_or_mutation_and_invalid_signals_fail_closed() {
    for (fixed, observed) in [
        (Some(4096), signals()),
        (
            None,
            MetalMemorySignals {
                recommended_max_bytes: 0,
                ..signals()
            },
        ),
    ] {
        let mut attempts = vec![];
        assert!(
            select(
                4096,
                fixed,
                4096,
                Cost {
                    metal: 1000,
                    cpu: 0
                },
                observed,
                |rows| {
                    attempts.push(rows);
                    Ok(Cost {
                        metal: rows as u64 * 2,
                        cpu: 0,
                    })
                }
            )
            .is_err()
        );
        assert_eq!(attempts, [4096]);
    }
    let mut attempts = vec![];
    let (rows, _) = select(2048, None, 9, Cost::default(), signals(), |rows| {
        attempts.push(rows);
        Ok(Cost::default())
    })
    .unwrap();
    assert_eq!(rows, 2048);
    assert_eq!(attempts, [9], "normal row minimum, not a third geometry");
}

#[test]
fn new_live_request_can_narrow_before_mutation_but_keeps_an_admissible_retained_width() {
    let price = |rows: usize| {
        Ok(Cost {
            metal: rows as u64 * 2,
            cpu: 0,
        })
    };
    let base = Cost {
        metal: 1000,
        cpu: 0,
    };
    assert!(select(4096, Some(4096), 4096, base, signals(), price).is_err());
    let (rows, _) = select(4096, None, 4096, base, signals(), price).unwrap();
    assert_eq!(
        rows, 2048,
        "a new request need not discard compatible KV to narrow"
    );
    let mut attempts = vec![];
    let (rows, _) = select(2048, None, 4096, Cost::default(), signals(), |rows| {
        attempts.push(rows);
        price(rows)
    })
    .unwrap();
    assert_eq!(rows, 2048);
    assert_eq!(
        attempts,
        [2048],
        "retain cheap live geometry rather than widening it"
    );
}

#[test]
fn shared_gpu_growth_uses_mwss_and_physical_ceiling_not_cpu_availability() {
    let mut observed = signals();
    observed
        .host_budget
        .as_mut()
        .unwrap()
        .conservative_available_bytes = 110;
    assert!(
        Cost {
            metal: 8000,
            cpu: 100
        }
        .admit(observed)
        .is_ok()
    );
    assert_eq!(
        Cost {
            metal: 8000,
            cpu: 111
        }
        .admit(observed)
        .unwrap_err()
        .memory_admission()
        .unwrap()
        .reason(),
        Reason::HostAvailabilityInsufficient
    );
    observed.host_budget.as_mut().unwrap().physical_bytes = 8_999;
    assert_eq!(
        Cost {
            metal: 8000,
            cpu: 100
        }
        .admit(observed)
        .unwrap_err()
        .memory_admission()
        .unwrap()
        .reason(),
        Reason::HostInsufficient
    );
}

#[test]
fn current_device_and_process_debt_still_bind_candidate_admission() {
    for observed in [
        MetalMemorySignals {
            current_allocated_bytes: 9000,
            ..signals()
        },
        MetalMemorySignals {
            process_limit_remaining_bytes: Some(100),
            ..signals()
        },
    ] {
        assert!(
            select(4096, None, 4096, Cost::default(), observed, |rows| {
                Ok(Cost {
                    metal: rows as u64,
                    cpu: 1,
                })
            })
            .is_err()
        );
    }
}

#[test]
fn reports_actual_internal_ranges_for_logical_4096_partial_suffix_and_nonzero_start() {
    for (start, tokens, expected) in [
        (0, 4096, vec![0..2048, 2048..4096]),
        (73, 4096, vec![73..2121, 2121..4169]),
        (73, 4097, vec![73..2121, 2121..4169, 4169..4170]),
        (9000, 17, vec![9000..9017]),
    ] {
        let report = PrefillGeometry {
            start,
            tokens,
            selected_rows: 2048,
        };
        assert_eq!(report.chunk_count(), expected.len());
        assert_eq!(report.chunk_ranges().collect::<Vec<_>>(), expected);
        assert_eq!(report.max_chunk_rows(), tokens.min(2048));
    }
}

#[test]
fn ced_lineage_uses_admitted_geometry_and_distinguishes_genuine_scalar_tails() {
    let c = Config::default();
    let schedule = super::super::super::gguf::Schedule::default();
    let hash = engram::HashConfig::default();
    for tokens in [2049, 4097] {
        for selected_rows in [2048, 4096] {
            let geometry = PrefillGeometry {
                start: 0,
                tokens,
                selected_rows,
            };
            let plan =
                ced::plan_for_geometry(&c, &schedule, &hash, geometry, tokens + 256).unwrap();
            assert_eq!(plan.full_chunk_rows, selected_rows);
            assert_eq!(
                plan.scalar_tail(),
                (tokens % selected_rows == 1).then_some(tokens - 1)
            );
            let json = serde_json::to_value(PrefillReport {
                geometry,
                admission: None,
                execution: PrefillExecution::ShrinkingCed,
                fallback: None,
                counters: None,
                memory: None,
                observed_incremental_metal_bytes: None,
            })
            .unwrap();
            assert_eq!(json["input_geometry"]["selected_rows"], selected_rows);
            assert_eq!(geometry.chunk_count(), tokens.div_ceil(selected_rows));
            assert!(
                plan.decoder
                    .iter()
                    .all(|layer| layer.output_rows.end <= tokens)
            );
        }
    }
    assert!(
        ced::plan_for_geometry(
            &c,
            &schedule,
            &hash,
            PrefillGeometry {
                start: 73,
                tokens: 4096,
                selected_rows: 2048
            },
            8192
        )
        .is_err()
    );
}

#[test]
fn runtime_price_uses_sequential_full_ced_maxima_and_keeps_cpu_domains_separate() {
    let base = RequestMemoryEstimate {
        cpu_persistent_bytes: 10,
        cpu_transient_bytes: 20,
        cpu_bytes: 30,
        metal_bytes: 40,
        total_bytes: 70,
    };
    let ced = CedMemoryEstimate {
        incremental_metal_bytes: 250,
        incremental_cpu_bytes: 7,
        ..Default::default()
    };
    let runtime = runtime_memory(base, (100, 10), Some(&ced)).unwrap();
    assert_eq!(runtime.metal_bytes, 290);
    assert_eq!(runtime.cpu_bytes, 40);
    assert_eq!(runtime.cpu_transient_bytes, 30);
    assert_eq!(runtime.total_bytes, 330);
    assert!(runtime_memory(base, (u64::MAX, 1), None).is_err());
    assert!(runtime_memory(base, (1, u64::MAX), None).is_err());
}

#[test]
fn intrinsic_execution_refusal_precedes_candidate_scratch_pricing() {
    let mut c = Config::default();
    c.hc_iterations = 3;
    let prices = std::cell::Cell::new(0);
    let result = preflight::runtime_from_parts(
        &c,
        &super::super::super::gguf::Schedule::default(),
        &engram::HashConfig::default(),
        &[(6144, 5 * c.hidden); 2],
        4097,
        8192,
        &[PrefillExecution::ShrinkingCed],
        2048,
        false,
        &|bytes| {
            prices.set(prices.get() + 1);
            Ok(bytes)
        },
    );
    assert!(result.unwrap_err().contains("Sinkhorn"));
    assert_eq!(prices.get(), 0);
}

#[test]
fn request_identity_is_arc_bound_not_config_or_capacity_equality() {
    let identity = Arc::new(());
    let plan = RequestPlan {
        capacity: 8192,
        report: PrefillGeometry {
            start: 0,
            tokens: 4097,
            selected_rows: 2048,
        },
        memory: RequestMemoryEstimate {
            cpu_persistent_bytes: 0,
            cpu_transient_bytes: 0,
            cpu_bytes: 0,
            metal_bytes: 0,
            total_bytes: 0,
        },
        cpu_reserve: 99,
        binding_cpu_reserve: 0,
        admission: PrefillAdmission {
            intent: PrefillIntent::Full,
            fallback: None,
        },
        identity: identity.clone(),
    };
    plan.validate_identity(&identity).unwrap();
    assert!(plan.validate_identity(&Arc::new(())).is_err());
    assert_eq!(plan.prefill().selected_rows, 2048);
    assert_eq!(plan.caller_cpu_reserve(), 99);
}

fn runtime_fixture(with_ced: bool) -> RequestMemoryEstimate {
    runtime_memory(
        RequestMemoryEstimate {
            cpu_persistent_bytes: 10,
            cpu_transient_bytes: 20,
            cpu_bytes: 30,
            metal_bytes: 40,
            total_bytes: 70,
        },
        (100, 10),
        with_ced.then_some(&CedMemoryEstimate {
            incremental_metal_bytes: 250,
            incremental_cpu_bytes: 70,
            ..Default::default()
        }),
    )
    .unwrap()
}

#[test]
fn optional_ced_reprices_full_at_both_boundaries_with_fresh_signals_and_all_reserves() {
    // These are the disjoint caller/binding/source/private CPU envelopes.
    let reserve = 13 + 17 + 19 + 23;
    for stage in ["post_binding", "allocation"] {
        let prices = std::cell::RefCell::new(vec![]);
        let samples = std::cell::Cell::new(0);
        let (memory, admission) = admit_runtime(
            PrefillAdmission {
                intent: PrefillIntent::OptionalCed,
                fallback: None,
            },
            reserve,
            stage,
            |ced| {
                prices.borrow_mut().push((ced, 8192, 2048));
                Ok((runtime_fixture(ced), ced))
            },
            || {
                samples.set(samples.get() + 1);
                let mut s = signals();
                s.host_budget.as_mut().unwrap().conservative_available_bytes =
                    if samples.get() == 1 { 150 } else { 113 };
                s
            },
        )
        .unwrap();
        assert_eq!(*prices.borrow(), [(true, 8192, 2048), (false, 8192, 2048)]);
        assert_eq!(samples.get(), 2);
        assert_eq!(memory, runtime_fixture(false));
        assert_eq!(admission.intent, PrefillIntent::OptionalCed);
        assert!(admission.fallback.as_ref().unwrap().starts_with(stage));
        assert!(
            admission
                .fallback
                .as_ref()
                .unwrap()
                .contains("host_availability_insufficient")
        );

        let mut attempted = vec![];
        admit_runtime(
            admission.clone(),
            reserve,
            "allocation",
            |ced| {
                attempted.push(ced);
                Ok((runtime_fixture(ced), ced))
            },
            signals,
        )
        .unwrap();
        assert_eq!(
            attempted,
            [false],
            "allocation cannot re-enable declined CED"
        );

        let mut s = signals();
        s.host_budget.as_mut().unwrap().conservative_available_bytes = 111;
        assert!(
            admit_runtime(
                admission,
                reserve,
                stage,
                |ced| Ok((runtime_fixture(ced), ced)),
                || s
            )
            .is_err(),
            "Full must still admit every reserve; 40 + 72 exceeds 111"
        );
    }
}

#[test]
fn required_pair_refuses_pressure_at_both_boundaries_without_full_retry() {
    for stage in ["post_binding", "allocation"] {
        let mut attempted = vec![];
        let mut s = signals();
        s.host_budget.as_mut().unwrap().conservative_available_bytes = 150;
        let error = admit_runtime(
            PrefillAdmission {
                intent: PrefillIntent::RequiredFullCed,
                fallback: None,
            },
            72,
            stage,
            |ced| {
                attempted.push(ced);
                Ok((runtime_fixture(ced), ced))
            },
            || s,
        )
        .unwrap_err();
        assert_eq!(attempted, [true]);
        assert_eq!(
            error.memory_admission().unwrap().reason(),
            Reason::HostAvailabilityInsufficient
        );
    }
}

#[test]
fn optional_ced_invalid_signals_and_overflow_are_not_pressure_or_fallback() {
    let invalid = [
        MetalMemorySignals {
            recommended_max_bytes: 0,
            ..signals()
        },
        MetalMemorySignals {
            host_budget: None,
            ..signals()
        },
        MetalMemorySignals {
            host_budget: Some(HostMemoryBudget {
                physical_bytes: 0,
                conservative_available_bytes: 0,
                external_gap_bytes: 0,
                private_shadow_bytes: 0,
            }),
            ..signals()
        },
    ];
    for stage in ["post_binding", "allocation"] {
        for (observed, reserve) in invalid
            .into_iter()
            .map(|s| (s, 0))
            .chain([(signals(), u64::MAX)])
        {
            let mut attempted = vec![];
            let error = admit_runtime(
                PrefillAdmission {
                    intent: PrefillIntent::OptionalCed,
                    fallback: None,
                },
                reserve,
                stage,
                |ced| {
                    attempted.push(ced);
                    Ok((runtime_fixture(ced), ced))
                },
                || observed,
            )
            .unwrap_err();
            assert_eq!(attempted, [true]);
            assert!(
                !error
                    .memory_admission()
                    .is_some_and(|e| pressure_reason(e.reason()))
            );
        }
        let mut attempted = vec![];
        assert!(
            admit_runtime(
                PrefillAdmission {
                    intent: PrefillIntent::OptionalCed,
                    fallback: None
                },
                0,
                stage,
                |ced| {
                    attempted.push(ced);
                    Err("arithmetic overflow".into())
                },
                signals,
            )
            .is_err()
        );
        assert_eq!(attempted, [true]);
    }
}

#[test]
fn unsupported_ced_is_optional_only_and_intent_survives_serialization() {
    for intent in [PrefillIntent::OptionalCed, PrefillIntent::RequiredFullCed] {
        let mut attempted = vec![];
        let result = admit_runtime(
            PrefillAdmission {
                intent,
                fallback: None,
            },
            72,
            "post_binding",
            |ced| {
                attempted.push(ced);
                Ok((runtime_fixture(false), false))
            },
            signals,
        );
        if intent == PrefillIntent::RequiredFullCed {
            assert!(result.is_err());
            assert_eq!(attempted, [true]);
        } else {
            let (_, admission) = result.unwrap();
            assert_eq!(attempted, [true, false]);
            let json = serde_json::to_value(admission).unwrap();
            assert_eq!(json["intent"], "optional_ced");
            assert_eq!(json["fallback"], "post_binding: CED unsupported");
        }
    }
    assert_eq!(
        PrefillIntent::from_executions(&[PrefillExecution::ShrinkingCed]).unwrap(),
        PrefillIntent::OptionalCed
    );
    assert_eq!(
        PrefillIntent::from_executions(&[
            PrefillExecution::FullForward,
            PrefillExecution::ShrinkingCed
        ])
        .unwrap(),
        PrefillIntent::RequiredFullCed
    );
    assert!(PrefillIntent::from_executions(&[]).is_err());
}

#[test]
fn ced_executor_contract_uses_admitted_state_width_not_resident_preference() {
    let geometry = PrefillGeometry {
        start: 0,
        tokens: 2049,
        selected_rows: 2048,
    };
    let plan = ced::plan_for_geometry(
        &Config::default(),
        &super::super::super::gguf::Schedule::default(),
        &engram::HashConfig::default(),
        geometry,
        4096,
    )
    .unwrap();
    let resident_preference = 4096;
    assert_ne!(
        plan.full_chunk_rows, resident_preference,
        "old executor comparison rejects a valid request"
    );
    let ticket = ced::AdmittedPlan::validate(&plan, geometry, Some(2048), 2049, 4096).unwrap();
    assert_eq!(ticket.plan().full_chunk_rows, 2048);
    for (report, state_rows, tokens, capacity) in [
        (geometry, Some(4096), 2049, 4096),
        (geometry, None, 2049, 4096),
        (
            PrefillGeometry {
                selected_rows: 4096,
                ..geometry
            },
            Some(4096),
            2049,
            4096,
        ),
        (
            PrefillGeometry {
                start: 1,
                ..geometry
            },
            Some(2048),
            2049,
            4096,
        ),
        (geometry, Some(2048), 2048, 4096),
        (geometry, Some(2048), 2049, 2048),
    ] {
        assert!(ced::AdmittedPlan::validate(&plan, report, state_rows, tokens, capacity).is_err());
    }
}
