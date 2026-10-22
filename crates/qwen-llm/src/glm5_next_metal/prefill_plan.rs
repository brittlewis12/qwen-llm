//! Bounded packed workspace selection; context and pricing belong to the caller.
//!
//! Throughput capture: `docs/bench/2026-10-09-glm53-prefill-attribution/rows-cost.json`.
//! Chunk boundaries change Fast's arithmetic schedule, not its precision;
//! different selected widths do not imply bit-identical results.

use super::Result;

/// Ranked widths capped to the useful prompt/endpoint (or an approved maximum).
/// Zero is serial execution, including when packed execution is unavailable.
pub fn candidates(max_rows: usize, packed: bool) -> Vec<usize> {
    let mut rows: Vec<_> = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0]
        .into_iter()
        .map(|n| if packed { n.min(max_rows) } else { 0 })
        .collect();
    rows.dedup();
    rows
}

/// Try ranked widths, returning the first successful width and value unchanged.
/// Only typed memory pressure permits a smaller workspace; the final refusal
/// retains its capacity advice. The closure must keep requested context fixed.
///
/// Before loading, use `preflight_session` to charge weights, session and reserve.
/// After loading, start at that approved width and use the normal session
/// constructor's resident-aware admission. Do not retry an executing session or
/// use this to change an already frozen benchmark width.
pub fn select<T>(
    max_rows: usize,
    mut attempt: impl FnMut(usize) -> Result<T>,
) -> Result<(usize, T)> {
    for rows in candidates(max_rows, true) {
        match attempt(rows) {
            Ok(value) => return Ok((rows, value)),
            Err(error) if rows > 0 && error.is_memory_pressure() => {}
            Err(error) => return Err(error),
        }
    }
    unreachable!("serial is the final candidate")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm5_next::Glm5NextError;
    use crate::glm5_next::memory::DYNAMIC_RESERVE_BYTES;
    use crate::glm5_next_metal::{CapacityAdvice, Glm5NextMetalError};
    use crate::metal::{
        MetalError, MetalMemoryAdmissionReason, MetalMemorySignals, evaluate_metal_memory_admission,
    };

    fn signals(budget: u64) -> MetalMemorySignals {
        MetalMemorySignals {
            recommended_max_bytes: budget,
            current_allocated_bytes: 0,
            process_limit_remaining_bytes: Some(budget),
        }
    }

    fn admit(bytes: u64, reserve: u64, signals: MetalMemorySignals) -> Result<()> {
        match evaluate_metal_memory_admission(bytes, reserve, signals, true).refusal() {
            None => Ok(()),
            Some(denied) => Err(Glm5NextMetalError::MemoryAdmission {
                denied,
                budget_bytes: signals.recommended_max_bytes - signals.current_allocated_bytes,
                advice: CapacityAdvice::NotEvaluated,
            }),
        }
    }

    fn pressure() -> Glm5NextMetalError {
        admit(2, 0, signals(1)).unwrap_err()
    }

    #[test]
    fn ranked_capped_unique_zero_one_and_unpacked() {
        assert_eq!(
            candidates(usize::MAX, true),
            [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0]
        );
        assert_eq!(
            candidates(300, true),
            [300, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0]
        );
        assert_eq!(candidates(3, true), [3, 2, 1, 0]);
        assert_eq!(candidates(1, true), [1, 0]);
        assert_eq!(candidates(0, true), [0]);
        assert_eq!(candidates(usize::MAX, false), [0]);
        for maximum in [0, 1, 3, 300, 2048, usize::MAX] {
            let mut calls = 0;
            let (rows, value) = select(maximum, |rows| {
                calls += 1;
                Ok(Box::new(rows))
            })
            .unwrap();
            assert_eq!(calls, 1, "success ends selection");
            assert_eq!(rows, maximum.min(2048));
            assert_eq!(*value, rows);
        }
    }

    #[test]
    fn huge_context_stays_fixed_across_full_model_and_resident_admission() {
        let capacity = 1usize << 20;
        let weights = 90u64 << 30;
        let session = capacity as u64 * 4096;
        let scratch = |rows: usize| rows as u64 * (1 << 20);
        let budget = weights + session + scratch(256) + DYNAMIC_RESERVE_BYTES;
        let mut attempts = Vec::new();
        let (approved, _) = select(capacity, |rows| {
            attempts.push((capacity, rows));
            admit(
                weights + session + scratch(rows),
                DYNAMIC_RESERVE_BYTES,
                signals(budget),
            )
        })
        .unwrap();
        assert_eq!(approved, 256);
        assert_eq!(
            attempts,
            [
                (capacity, 2048),
                (capacity, 1024),
                (capacity, 512),
                (capacity, 256)
            ]
        );

        // Loaded weights are already in the signals; only session plus reserve
        // is new. Fresh pressure after loading can reduce the approved width.
        let resident = MetalMemorySignals {
            current_allocated_bytes: weights,
            process_limit_remaining_bytes: Some(budget - weights),
            ..signals(budget)
        };
        assert_eq!(
            select(approved, |rows| admit(
                session + scratch(rows),
                DYNAMIC_RESERVE_BYTES,
                resident
            ))
            .unwrap()
            .0,
            256
        );
        let pressured = MetalMemorySignals {
            recommended_max_bytes: budget - scratch(128),
            process_limit_remaining_bytes: Some(budget - weights - scratch(128)),
            ..resident
        };
        let mut attempts = Vec::new();
        let (rows, value) = select(approved, |rows| {
            attempts.push((capacity, rows));
            admit(session + scratch(rows), DYNAMIC_RESERVE_BYTES, pressured)?;
            Ok((capacity, rows))
        })
        .unwrap();
        assert_eq!(attempts, [(capacity, 256), (capacity, 128)]);
        assert_eq!((rows, value), (128, (capacity, 128)));
    }

    #[test]
    fn serial_is_attempted_once_and_final_advice_is_preserved() {
        let mut attempts = Vec::new();
        let error = select(3, |rows| {
            attempts.push(rows);
            let mut error = pressure();
            if let Glm5NextMetalError::MemoryAdmission { advice, .. } = &mut error {
                *advice = CapacityAdvice::Fits(100 + rows as u64);
            }
            Err::<(), _>(error)
        })
        .unwrap_err();
        assert_eq!(attempts, [3, 2, 1, 0]);
        assert!(matches!(
            error,
            Glm5NextMetalError::MemoryAdmission {
                advice: CapacityAdvice::Fits(100),
                ..
            }
        ));
        assert_eq!(
            select(3, |rows| if rows > 0 {
                Err(pressure())
            } else {
                Ok("serial")
            })
            .unwrap(),
            (0, "serial")
        );
    }

    #[test]
    fn non_pressure_refusals_never_try_smaller() {
        let missing = MetalMemorySignals {
            process_limit_remaining_bytes: None,
            ..signals(10)
        };
        let invalid = MetalMemorySignals {
            recommended_max_bytes: 0,
            ..signals(10)
        };
        let errors = [
            admit(1, 0, missing).unwrap_err(),
            admit(1, 0, invalid).unwrap_err(),
            admit(u64::MAX, 1, signals(u64::MAX)).unwrap_err(),
            Glm5NextError::Overflow("test geometry").into(),
            Glm5NextMetalError::Invalid("test geometry".into()),
            MetalError::NoBuffer(4096).into(),
            MetalError::CommandBufferFailed {
                status: "failed".into(),
                error: "test GPU".into(),
            }
            .into(),
            Glm5NextMetalError::Cancelled("test".into()),
        ];
        for error in errors {
            let expected = error.to_string();
            let mut error = Some(error);
            let mut calls = 0;
            let result = select(2048, |_| {
                calls += 1;
                Err::<(), _>(error.take().expect("must not retry"))
            })
            .unwrap_err();
            assert_eq!(calls, 1);
            assert_eq!(result.to_string(), expected);
        }
        let mut attempts = Vec::new();
        let error = select(2048, |rows| {
            attempts.push(rows);
            if rows == 2048 {
                Err(pressure())
            } else {
                admit(1, 0, missing)
            }
        })
        .unwrap_err();
        assert_eq!(attempts, [2048, 1024]);
        assert!(
            matches!(error, Glm5NextMetalError::MemoryAdmission { denied, .. } if denied.reason() == MetalMemoryAdmissionReason::ProcessSignalUnavailable)
        );
    }
}
