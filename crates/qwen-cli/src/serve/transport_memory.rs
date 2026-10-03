//! Additional CPU-only transport memory. Resident-family model/session plans
//! remain responsible for Metal allocations; ordinary Qwen combines this with
//! its dynamic request admission and snapshot pressure relief instead.

use super::items::ServeError;

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
    // omits the process budget, not zero usable RAM. Missing signals fail closed.
    if bytes == 0 || process_remaining.is_some_and(|remaining| remaining == 0 || remaining >= bytes)
    {
        return Ok(());
    }
    Err(ServeError {
        status: 503,
        error_type: "server_busy",
        code: Some("memory_admission_denied"),
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
