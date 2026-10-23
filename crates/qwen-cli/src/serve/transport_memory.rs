//! Additional CPU-only transport memory. Resident-family model/session plans
//! remain responsible for Metal allocations; ordinary Qwen combines this with
//! its dynamic request admission and snapshot pressure relief instead.
//!
//! Memory refusals map to one status table across every lane
//! ([`refusal_kind`]): memory pressure is 503 `memory_admission_denied`
//! (retry once memory frees); missing or invalid memory telemetry and a
//! required size that overflows are 500, each with its own code. Geometry,
//! input and GPU failures never reach these mappers.

use super::items::ServeError;
use qwen_llm::metal::{HostMemoryHeadroom, MemoryAdmissionDenied, MetalMemoryAdmissionReason};

/// Status, error type and code of a refused admission's reason.
pub(crate) fn refusal_kind(
    reason: MetalMemoryAdmissionReason,
) -> (u16, &'static str, &'static str) {
    use MetalMemoryAdmissionReason as R;
    match reason {
        R::WorkingSetInsufficient
        | R::ProcessInsufficient
        | R::BothInsufficient
        | R::HostInsufficient
        | R::HostAvailabilityInsufficient => (503, "server_busy", "memory_admission_denied"),
        R::ProcessSignalUnavailable | R::HostSignalUnavailable => {
            (500, "server_error", "memory_signal_unavailable")
        }
        R::InvalidWorkingSetSignal | R::InvalidHostSignal => {
            (500, "server_error", "memory_signal_invalid")
        }
        R::RequiredBytesOverflow => (500, "server_error", "memory_size_overflow"),
        // A refusal is built only from a decision that did not admit.
        R::AdmittedWithProcessBudget | R::AdmittedWithHostBudget => {
            (500, "server_error", "memory_admission_inconsistent")
        }
    }
}

/// The serve error for a typed memory refusal of `what` (a session, a
/// request's state, a model).
pub(crate) fn memory_refusal(what: &str, denied: &MemoryAdmissionDenied) -> ServeError {
    let (status, error_type, code) = refusal_kind(denied.reason());
    ServeError {
        status,
        error_type,
        code: Some(code),
        param: None,
        message: format!("{what}: {denied}"),
    }
}

pub(super) fn combined_reserve(durable: u64, transport: u64) -> Result<u64, ServeError> {
    durable
        .checked_add(transport)
        .ok_or_else(|| ServeError::server_error("combined CPU reservation overflow"))
}

#[derive(Debug)]
pub(super) struct CpuMemoryRefusal {
    pub(super) reason: MetalMemoryAdmissionReason,
    pub(super) error: ServeError,
}

/// Keep the binding CPU limit typed until snapshot relief and fresh admission
/// finish. HTTP pressure alone does not say whether CPU eviction can help.
pub(super) fn admit_resident_transport_with_headroom(
    bytes: u64,
    headroom: Option<HostMemoryHeadroom>,
) -> Result<(), CpuMemoryRefusal> {
    if bytes == 0 {
        return Ok(());
    }
    let reason = headroom.map_or(
        MetalMemoryAdmissionReason::HostSignalUnavailable,
        |headroom| headroom.admission_reason(bytes),
    );
    if reason.is_admitted() {
        return Ok(());
    }
    let (status, error_type, code) = refusal_kind(reason);
    Err(CpuMemoryRefusal {
        reason,
        error: ServeError {
            status,
            error_type,
            code: Some(code),
            param: None,
            message: format!(
                "transport CPU memory admission denied ({}): required={bytes} headroom={headroom:?}",
                reason.as_str(),
            ),
        },
    })
}

pub(super) fn admit_resident_transport(
    bytes: u64,
    host_remaining: Option<u64>,
) -> Result<(), ServeError> {
    // This is effective headroom, not the raw Mach process-limit field.
    // A derived zero is exhaustion and cannot omit the host budget.
    if bytes == 0 || host_remaining.is_some_and(|remaining| remaining >= bytes) {
        return Ok(());
    }
    let reason = if host_remaining.is_none() {
        MetalMemoryAdmissionReason::HostSignalUnavailable
    } else {
        MetalMemoryAdmissionReason::HostInsufficient
    };
    let (status, error_type, code) = refusal_kind(reason);
    Err(ServeError {
        status,
        error_type,
        code: Some(code),
        param: None,
        message: format!(
            "transport CPU memory admission denied: required={bytes} host_remaining={host_remaining:?}"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_cpu_transport_keeps_binding_reason_and_zero_allowance() {
        use qwen_llm::metal::{HostMemoryBudget, MetalMemoryAdmissionReason as R};
        assert!(admit_resident_transport_with_headroom(0, None).is_ok());
        let missing = admit_resident_transport_with_headroom(100, None).unwrap_err();
        assert_eq!(missing.reason, R::HostSignalUnavailable);
        assert_eq!(missing.error.status, 500);
        for (available, current, process, reason) in [
            (1_000, 701, 0, R::HostInsufficient),
            (299, 100, 0, R::HostAvailabilityInsufficient),
            (1_000, 100, 99, R::ProcessInsufficient),
        ] {
            let headroom = HostMemoryHeadroom::from_budget(
                HostMemoryBudget {
                    physical_bytes: 1_000,
                    conservative_available_bytes: available,
                    external_gap_bytes: 0,
                    private_shadow_bytes: 0,
                },
                current,
                process,
            );
            let denied = admit_resident_transport_with_headroom(100, headroom).unwrap_err();
            assert_eq!(denied.reason, reason);
            assert_eq!(
                (denied.error.status, denied.error.code),
                (503, Some("memory_admission_denied"))
            );
            assert!(denied.error.message.contains(reason.as_str()));
            assert!(admit_resident_transport_with_headroom(99, headroom).is_ok());
        }
    }

    #[test]
    fn transport_and_durable_reservations_add_without_overflow() {
        assert_eq!(combined_reserve(100, 25).unwrap(), 125);
        assert_eq!(combined_reserve(100, 0).unwrap(), 100);
        assert!(combined_reserve(u64::MAX, 1).is_err());
    }

    #[test]
    fn resident_transport_uses_effective_host_budget_without_pricing_cpu_as_metal() {
        assert!(admit_resident_transport(0, None).is_ok());
        assert!(admit_resident_transport(100, None).is_err());
        assert!(admit_resident_transport(100, Some(99)).is_err());
        assert!(admit_resident_transport(100, Some(100)).is_ok());
        assert!(admit_resident_transport(100, Some(0)).is_err());
    }

    /// Pressure is 503; a missing process signal is unavailable telemetry
    /// (500 with its own code), not pressure.
    #[test]
    fn transport_refusals_tell_pressure_from_missing_telemetry() {
        let pressure = admit_resident_transport(100, Some(99)).unwrap_err();
        assert_eq!(
            (pressure.status, pressure.error_type, pressure.code),
            (503, "server_busy", Some("memory_admission_denied"))
        );
        let exhausted = admit_resident_transport(1, Some(0)).unwrap_err();
        assert_eq!(
            (exhausted.status, exhausted.code),
            (503, Some("memory_admission_denied"))
        );
        let missing = admit_resident_transport(100, None).unwrap_err();
        assert_eq!(
            (missing.status, missing.error_type, missing.code),
            (500, "server_error", Some("memory_signal_unavailable"))
        );
    }

    /// Every refusal reason has one status across lanes; only pressure is 503.
    #[test]
    fn memory_refusals_map_by_reason() {
        use qwen_llm::metal::{MetalMemoryAdmissionReason as R, MetalMemorySignals};
        let signals = MetalMemorySignals {
            host_budget: Some(qwen_llm::metal::HostMemoryBudget {
                physical_bytes: 1024 << 30,
                conservative_available_bytes: 1024 << 30,
                external_gap_bytes: 0,
                private_shadow_bytes: 0,
            }),
            recommended_max_bytes: 1 << 30,
            current_allocated_bytes: 0,
            process_limit_remaining_bytes: Some(1 << 20),
        };
        for (reason, status, code) in [
            (R::WorkingSetInsufficient, 503, "memory_admission_denied"),
            (R::ProcessInsufficient, 503, "memory_admission_denied"),
            (R::BothInsufficient, 503, "memory_admission_denied"),
            (R::HostInsufficient, 503, "memory_admission_denied"),
            (
                R::HostAvailabilityInsufficient,
                503,
                "memory_admission_denied",
            ),
            (
                R::ProcessSignalUnavailable,
                500,
                "memory_signal_unavailable",
            ),
            (R::InvalidWorkingSetSignal, 500, "memory_signal_invalid"),
            (R::HostSignalUnavailable, 500, "memory_signal_unavailable"),
            (R::InvalidHostSignal, 500, "memory_signal_invalid"),
            (R::RequiredBytesOverflow, 500, "memory_size_overflow"),
        ] {
            let denied = MemoryAdmissionDenied::new(
                reason,
                (reason != R::RequiredBytesOverflow).then_some(1 << 31),
                signals,
                Some(1 << 30),
            )
            .expect("test reason refuses admission");
            let error = memory_refusal("GLM-5.3-Flash session", &denied);
            assert_eq!(
                (error.status, error.code),
                (status, Some(code)),
                "{reason:?}"
            );
            assert_eq!(reason.is_pressure(), status == 503, "{reason:?}");
            assert!(error.message.contains(reason.as_str()), "{}", error.message);
        }
    }
    #[test]
    fn resident_control_and_transport_allowances_both_survive_pressure_checks() {
        let control = super::super::control::CPU_RESERVE_BYTES;
        let reserve = combined_reserve(control, 128).unwrap();
        assert!(admit_resident_transport(reserve, Some(control)).is_err());
        assert!(admit_resident_transport(reserve, Some(128)).is_err());
        assert!(admit_resident_transport(reserve, Some(reserve)).is_ok());
        assert!(admit_resident_transport(reserve, None).is_err());
        assert_eq!(combined_reserve(0, 128).unwrap(), 128);
    }

    #[test]
    fn pressure_release_retries_once_but_telemetry_refusal_does_not_release() {
        for missing in [false, true] {
            let attempts = std::cell::Cell::new(0);
            let releases = std::cell::Cell::new(0);
            let attempt = || {
                attempts.set(attempts.get() + 1);
                let headroom = (!missing).then(|| {
                    HostMemoryHeadroom::from_budget(
                        qwen_llm::metal::HostMemoryBudget {
                            physical_bytes: 1_000,
                            conservative_available_bytes: 1_000,
                            external_gap_bytes: 0,
                            private_shadow_bytes: 0,
                        },
                        100,
                        if releases.get() > 0 { 100 } else { 99 },
                    )
                    .unwrap()
                });
                admit_resident_transport_with_headroom(100, headroom)
            };
            let result = crate::serve::memory_relief::retry(
                "test",
                attempt(),
                |result| {
                    result
                        .as_ref()
                        .err()
                        .map(|denied| (denied.reason, u64::MAX))
                },
                |_| {
                    releases.set(releases.get() + 1);
                    crate::serve::memory_relief::test_evicted(10)
                },
                attempt,
            );
            assert_eq!(attempts.get(), if missing { 1 } else { 2 });
            assert_eq!(releases.get(), usize::from(!missing));
            if missing {
                let denied = result.unwrap_err();
                assert_eq!(
                    denied.reason,
                    MetalMemoryAdmissionReason::HostSignalUnavailable
                );
                assert_eq!(denied.error.status, 500);
            } else {
                assert!(result.is_ok());
            }
        }
    }
}
