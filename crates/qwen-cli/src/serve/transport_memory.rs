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
use qwen_llm::metal::{MemoryAdmissionDenied, MetalMemoryAdmissionReason};

/// Status, error type and code of a refused admission's reason.
pub(crate) fn refusal_kind(
    reason: MetalMemoryAdmissionReason,
) -> (u16, &'static str, &'static str) {
    use MetalMemoryAdmissionReason as R;
    match reason {
        R::WorkingSetInsufficient | R::ProcessInsufficient | R::BothInsufficient => {
            (503, "server_busy", "memory_admission_denied")
        }
        R::ProcessSignalUnavailable => (500, "server_error", "memory_signal_unavailable"),
        R::InvalidWorkingSetSignal => (500, "server_error", "memory_signal_invalid"),
        R::RequiredBytesOverflow => (500, "server_error", "memory_size_overflow"),
        // A refusal is built only from a decision that did not admit.
        R::AdmittedWithProcessBudget | R::AdmittedProcessBudgetOmitted => {
            (500, "server_error", "memory_admission_inconsistent")
        }
    }
}

/// The serve error for a typed memory refusal of `what` (a session, a
/// request's state, a model).
pub(crate) fn memory_refusal(what: &str, denied: &MemoryAdmissionDenied) -> ServeError {
    let (status, error_type, code) = refusal_kind(denied.reason);
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

pub(super) fn admit_resident_transport(
    bytes: u64,
    process_remaining: Option<u64>,
) -> Result<(), ServeError> {
    // Match the resident loaders' convention: a reported zero means this OS
    // omits the process budget, not zero usable RAM. A missing signal fails
    // closed as unavailable telemetry (500), not as memory pressure (503).
    if bytes == 0 || process_remaining.is_some_and(|remaining| remaining == 0 || remaining >= bytes)
    {
        return Ok(());
    }
    let reason = if process_remaining.is_none() {
        MetalMemoryAdmissionReason::ProcessSignalUnavailable
    } else {
        MetalMemoryAdmissionReason::ProcessInsufficient
    };
    let (status, error_type, code) = refusal_kind(reason);
    Err(ServeError {
        status,
        error_type,
        code: Some(code),
        param: None,
        message: format!(
            "transport CPU memory admission denied: required={bytes} process_remaining={process_remaining:?}"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_and_durable_reservations_add_without_overflow() {
        assert_eq!(combined_reserve(100, 25).unwrap(), 125);
        assert_eq!(combined_reserve(100, 0).unwrap(), 100);
        assert!(combined_reserve(u64::MAX, 1).is_err());
    }

    #[test]
    fn resident_transport_uses_process_budget_without_pricing_cpu_as_metal() {
        assert!(admit_resident_transport(0, None).is_ok());
        assert!(admit_resident_transport(100, None).is_err());
        assert!(admit_resident_transport(100, Some(99)).is_err());
        assert!(admit_resident_transport(100, Some(100)).is_ok());
        assert!(admit_resident_transport(100, Some(0)).is_ok());
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
            recommended_max_bytes: 1 << 30,
            current_allocated_bytes: 0,
            process_limit_remaining_bytes: Some(1 << 20),
        };
        for (reason, status, code) in [
            (R::WorkingSetInsufficient, 503, "memory_admission_denied"),
            (R::ProcessInsufficient, 503, "memory_admission_denied"),
            (R::BothInsufficient, 503, "memory_admission_denied"),
            (
                R::ProcessSignalUnavailable,
                500,
                "memory_signal_unavailable",
            ),
            (R::InvalidWorkingSetSignal, 500, "memory_signal_invalid"),
            (R::RequiredBytesOverflow, 500, "memory_size_overflow"),
            (
                R::AdmittedWithProcessBudget,
                500,
                "memory_admission_inconsistent",
            ),
            (
                R::AdmittedProcessBudgetOmitted,
                500,
                "memory_admission_inconsistent",
            ),
        ] {
            let denied = MemoryAdmissionDenied {
                reason,
                required_bytes: (reason != R::RequiredBytesOverflow).then_some(1 << 31),
                signals,
                working_set_headroom_bytes: Some(1 << 30),
            };
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
}
