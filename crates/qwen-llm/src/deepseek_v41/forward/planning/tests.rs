use super::*;
use crate::metal::HostMemoryBudget;

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
        let report = PrefillReport {
            start,
            tokens,
            selected_rows: 2048,
        };
        assert_eq!(report.chunk_count(), expected.len());
        assert_eq!(report.chunk_ranges().collect::<Vec<_>>(), expected);
        assert_eq!(report.max_chunk_rows(), tokens.min(2048));
    }
}
