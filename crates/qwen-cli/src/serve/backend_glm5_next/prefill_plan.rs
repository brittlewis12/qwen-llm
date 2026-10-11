use super::super::SnapshotCaptureDenial;
use super::{ServeError, is_pressure_refusal};
use qwen_llm::glm5_next_metal::DEFAULT_PREFILL_ROWS;
use qwen_llm::metal::{MetalMemorySignals, evaluate_metal_memory_admission_with_cpu_bytes};

pub(super) fn candidates(capacity: usize, packed: bool) -> Vec<usize> {
    let mut rows: Vec<_> = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0]
        .into_iter()
        .map(|n| if packed { n.min(capacity) } else { 0 })
        .collect();
    rows.dedup();
    rows
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Price {
    pub buffers: u64,
    pub admission: u64,
}

impl Price {
    pub(super) fn from_ledger(ledger: &qwen_llm::glm5_next::Glm5NextMemoryLedger) -> Self {
        Self {
            buffers: ledger.session_buffer_bytes(),
            admission: ledger.peak_bytes() - ledger.phase_peaks().resident,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    pub rows: usize,
    pub price: Price,
    pub prefix_bytes: Option<u64>,
    pub preserved: bool,
}

pub(super) fn freeze_after_creation<T, E>(
    frozen: &mut Option<usize>,
    rows: usize,
    created: Result<T, E>,
) -> Result<T, E> {
    let session = created?;
    assert!(frozen.is_none_or(|old| old == rows));
    *frozen = Some(rows);
    Ok(session)
}

pub(super) fn create_with_pressure_retry<T>(
    mut plan: Plan,
    mut replan: impl FnMut(usize) -> Result<Plan, ServeError>,
    mut create: impl FnMut(usize) -> Result<T, ServeError>,
) -> Result<(T, Plan), ServeError> {
    loop {
        match create(plan.rows) {
            Ok(session) => return Ok((session, plan)),
            Err(error) if plan.rows > 0 && is_pressure_refusal(&error) => {
                let maximum = candidates(plan.rows, true)
                    .into_iter()
                    .find(|&rows| rows < plan.rows)
                    .expect("serial is the final candidate");
                plan = replan(maximum)?;
            }
            Err(error) => return Err(error),
        }
    }
}

pub(super) fn eligible_prefix(
    reuse: bool,
    scheduled: bool,
    cut: Option<usize>,
    cache_bytes: u64,
    entry: impl FnOnce(usize) -> Result<u64, ServeError>,
) -> Result<Option<u64>, ServeError> {
    if !reuse || !scheduled || cache_bytes == 0 {
        return Ok(None);
    }
    cut.filter(|&cut| cut >= DEFAULT_PREFILL_ROWS)
        .map(|cut| entry(cut).map(|bytes| (bytes <= cache_bytes).then_some(bytes)))
        .transpose()
        .map(Option::flatten)
}

fn invalid(message: impl Into<String>) -> ServeError {
    ServeError::server_error(message.into())
}

fn capture(
    mut signals: MetalMemorySignals,
    buffers: u64,
    required: u64,
) -> Result<(), SnapshotCaptureDenial> {
    signals.current_allocated_bytes = signals
        .current_allocated_bytes
        .checked_add(buffers)
        .ok_or(SnapshotCaptureDenial::EstimateOverflow)?;
    signals.process_limit_remaining_bytes = match signals.process_limit_remaining_bytes {
        Some(0) => Some(0),
        // A finite budget consumed by the session must not turn into Darwin's
        // zero-means-omitted sentinel in the capture guard.
        Some(n) if n <= buffers => {
            return Err(SnapshotCaptureDenial::ProcessHeadroom {
                deficit_bytes: buffers - n + 1,
            });
        }
        Some(n) => Some(n - buffers),
        None => None,
    };
    super::super::snapshot_capture_admission(required, signals)
}

/// Prefer reusable prefixes within the measured 512..=2048 throughput range.
/// Smaller widths remain execution fallbacks, not speculative cache sacrifices.
pub(super) fn select(
    rows: &[usize],
    signals: MetalMemorySignals,
    transport: u64,
    prefix_bytes: Option<u64>,
    mut price: impl FnMut(usize) -> Result<Price, ServeError>,
) -> Result<Plan, ServeError> {
    let required = prefix_bytes
        .map(|n| {
            n.checked_add(transport)
                .ok_or_else(|| invalid("GLM prefill prefix reserve overflow"))
        })
        .transpose()?;
    let mut executable = None;
    let mut refusal = None;
    for &rows in rows {
        if rows < DEFAULT_PREFILL_ROWS
            && let Some(plan) = executable
        {
            return Ok(plan);
        }
        let price = price(rows)?;
        let decision = evaluate_metal_memory_admission_with_cpu_bytes(
            price.admission,
            transport,
            0,
            signals,
            true,
        );
        if let Some(denied) = decision.refusal() {
            let error = super::super::transport_memory::memory_refusal("GLM prefill plan", &denied);
            if !is_pressure_refusal(&error) {
                return Err(error);
            }
            refusal = Some(error);
            continue;
        }
        let plan = Plan {
            rows,
            price,
            prefix_bytes,
            preserved: false,
        };
        executable.get_or_insert(plan);
        if rows < DEFAULT_PREFILL_ROWS {
            return Ok(plan);
        }
        let Some(required) = required else {
            return Ok(plan);
        };
        match capture(signals, price.buffers, required) {
            Ok(()) => {
                return Ok(Plan {
                    preserved: true,
                    ..plan
                });
            }
            Err(
                SnapshotCaptureDenial::MetalHeadroom
                | SnapshotCaptureDenial::ProcessHeadroom { .. },
            ) => {}
            Err(reason) => return Err(invalid(format!("GLM prefill capture plan: {reason:?}"))),
        }
    }
    executable.ok_or_else(|| refusal.unwrap_or_else(|| invalid("empty GLM prefill candidates")))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn signals(n: u64) -> MetalMemorySignals {
        MetalMemorySignals {
            recommended_max_bytes: n,
            current_allocated_bytes: 0,
            process_limit_remaining_bytes: Some(n),
        }
    }
    fn price(rows: usize) -> Result<Price, ServeError> {
        Ok(Price {
            buffers: rows as u64 * 1_000_000,
            admission: rows as u64 * 1_000_000,
        })
    }
    #[test]
    fn capped_unique_and_serial() {
        assert_eq!(candidates(3, true), [3, 2, 1, 0]);
        assert_eq!(candidates(3000, false), [0]);
    }
    #[test]
    fn abundant_and_pressure_keep_context() {
        let rows = candidates(4096, true);
        assert_eq!(
            select(&rows, signals(10_000_000_000), 0, None, price)
                .unwrap()
                .rows,
            2048
        );
        assert_eq!(
            select(&rows, signals(600_000_000), 0, None, price)
                .unwrap()
                .rows,
            512
        );
        assert_eq!(rows, candidates(4096, true));
    }
    #[test]
    fn prefix_preference_and_execution_fallback() {
        let rows = candidates(4096, true);
        let plan = select(&rows, signals(3_000_000_000), 0, Some(1_000_000_000), price).unwrap();
        assert!(plan.preserved);
        assert!(plan.rows < 2048);
        let plan = select(&rows, signals(3_000_000_000), 0, Some(4_000_000_000), price).unwrap();
        assert!(!plan.preserved);
        assert_eq!(plan.rows, 2048);
        let plan = select(
            &[2048, 0],
            signals(3_000_000_000),
            0,
            Some(2_000_000_000),
            price,
        )
        .unwrap();
        assert_eq!(plan.rows, 2048);
        assert!(!plan.preserved, "speculative reuse must not force serial");
        let plan = select(&rows, signals(400_000_000), 0, Some(100_000_000), price).unwrap();
        assert_eq!(plan.rows, 256, "execution pressure may go below 512");
        assert!(!plan.preserved);
    }
    #[test]
    fn telemetry_and_overflow_fail_closed() {
        let mut s = signals(10_000_000_000);
        s.process_limit_remaining_bytes = None;
        assert!(select(&[2048, 0], s, 0, None, price).is_err());
        assert!(select(&[2048, 0], signals(u64::MAX), 1, Some(u64::MAX), price).is_err());
        assert!(select(&[2048, 0], signals(u64::MAX), 0, Some(u64::MAX), price).is_err());
        assert!(select(&[2048, 0], signals(u64::MAX), u64::MAX, None, price).is_err());
        assert!(
            select(&[2048, 0], signals(u64::MAX), 0, None, |_| Err(invalid(
                "geometry"
            )))
            .is_err()
        );
    }
    #[test]
    fn omitted_is_not_exhausted() {
        let mut s = signals(10_000_000_000);
        s.process_limit_remaining_bytes = Some(0);
        assert!(capture(s, 100, 100).is_ok());
        s.process_limit_remaining_bytes = Some(100);
        assert!(matches!(
            capture(s, 100, 100),
            Err(SnapshotCaptureDenial::ProcessHeadroom { .. })
        ));
    }

    #[test]
    fn transport_and_dynamic_reserve_remain_charged() {
        let rows = [2, 1, 0];
        let price = |rows| {
            Ok(Price {
                buffers: rows as u64 * 100,
                admission: rows as u64 * 100 + 50,
            })
        };
        assert_eq!(select(&rows, signals(250), 1, None, price).unwrap().rows, 1);
        let mut s = signals(1_000_000_000);
        s.process_limit_remaining_bytes = Some(600_000_000);
        assert!(capture(s, 100_000_000, 100_000_000).is_err());
        s.process_limit_remaining_bytes = Some(0);
        assert!(capture(s, 100_000_000, 100_000_000).is_ok());
    }

    #[test]
    fn failed_creation_does_not_freeze() {
        let mut frozen = None;
        assert!(freeze_after_creation(&mut frozen, 2048, Err::<(), _>("allocation")).is_err());
        assert_eq!(frozen, None);
        freeze_after_creation(&mut frozen, 1024, Ok::<_, ()>(())).unwrap();
        assert_eq!(frozen, Some(1024));
        assert!(freeze_after_creation(&mut frozen, 1024, Err::<(), _>("pressure")).is_err());
        assert_eq!(frozen, Some(1024));
    }

    #[test]
    fn creation_pressure_replans_downward_without_freezing_failures() {
        let initial = select(
            &candidates(4096, true),
            signals(10_000_000_000),
            0,
            None,
            price,
        )
        .unwrap();
        let mut attempts = Vec::new();
        let mut frozen = None;
        let (session, plan) = create_with_pressure_retry(
            initial,
            |maximum| {
                select(
                    &candidates(maximum, true),
                    signals(10_000_000_000),
                    0,
                    None,
                    price,
                )
            },
            |rows| {
                attempts.push(rows);
                if rows > 512 {
                    let refusal =
                        evaluate_metal_memory_admission_with_cpu_bytes(2, 0, 0, signals(1), true)
                            .refusal()
                            .unwrap();
                    Err(super::super::super::transport_memory::memory_refusal(
                        "test", &refusal,
                    ))
                } else {
                    Ok(rows)
                }
            },
        )
        .unwrap();
        assert_eq!(attempts, [2048, 1024, 512]);
        assert_eq!(frozen, None);
        assert_eq!(
            freeze_after_creation(&mut frozen, plan.rows, Ok::<_, ServeError>(session)).unwrap(),
            512
        );
        assert_eq!(frozen, Some(512));

        let error = create_with_pressure_retry(
            initial,
            |_| panic!("not pressure"),
            |_| Err::<(), _>(invalid("geometry")),
        )
        .unwrap_err();
        assert!(error.message.contains("geometry"));
    }

    #[test]
    fn creation_retry_uses_new_budget_and_stops_after_serial_pressure() {
        let initial = select(
            &candidates(4096, true),
            signals(10_000_000_000),
            0,
            None,
            price,
        )
        .unwrap();
        let pressure = || {
            let denied = evaluate_metal_memory_admission_with_cpu_bytes(2, 0, 0, signals(1), true)
                .refusal()
                .unwrap();
            super::super::super::transport_memory::memory_refusal("test", &denied)
        };
        let mut attempts = Vec::new();
        let mut maxima = Vec::new();
        let result = create_with_pressure_retry(
            initial,
            |maximum| {
                maxima.push(maximum);
                // Fresh pressure can skip candidates, not just choose the next width.
                let available = if maximum == 1024 { 300_000_000 } else { 1 };
                select(
                    &candidates(maximum, true),
                    signals(available),
                    0,
                    None,
                    price,
                )
            },
            |rows| {
                attempts.push(rows);
                Err::<(), _>(pressure())
            },
        );
        assert!(is_pressure_refusal(&result.unwrap_err()));
        assert_eq!(attempts, [2048, 256, 0]);
        assert_eq!(maxima, [1024, 128]);

        let mut attempts = 0;
        let error = create_with_pressure_retry(
            initial,
            |maximum| {
                let mut unavailable = signals(10_000_000_000);
                unavailable.process_limit_remaining_bytes = None;
                select(&candidates(maximum, true), unavailable, 0, None, price)
            },
            |_| {
                attempts += 1;
                Err::<(), _>(pressure())
            },
        )
        .unwrap_err();
        assert!(!is_pressure_refusal(&error));
        assert_eq!(
            attempts, 1,
            "unknown fresh signals cannot trigger another allocation"
        );
    }

    #[test]
    fn ineligible_prefix_is_not_reserved() {
        for (reuse, scheduled, cut, budget) in [
            (false, true, Some(1024), 100),
            (true, false, Some(1024), 100),
            (true, true, None, 100),
            (true, true, Some(1024), 0),
            (true, true, Some(511), 100),
        ] {
            assert_eq!(
                eligible_prefix(reuse, scheduled, cut, budget, |_| panic!("not eligible")).unwrap(),
                None
            );
        }
        assert_eq!(
            eligible_prefix(true, true, Some(512), 99, |_| Ok(100)).unwrap(),
            None
        );
        assert_eq!(
            eligible_prefix(true, true, Some(512), 100, |_| Ok(100)).unwrap(),
            Some(100)
        );
        assert!(eligible_prefix(true, true, Some(512), 100, |_| Err(invalid("overflow"))).is_err());
    }
}
