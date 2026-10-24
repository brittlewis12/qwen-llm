//! One host policy for combined CPU/Metal demand. These are conservative
//! admission bounds, not physical-residency measurements or OS reservations.

use super::*;
use std::sync::atomic::AtomicU64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostMemoryBudget {
    pub physical_bytes: u64,
    /// Conservative CPU availability, not a shared-Metal working-set ceiling.
    pub conservative_available_bytes: u64,
    /// Live import lengths not covered by the observed Metal allocation API.
    pub external_gap_bytes: u64,
    /// Potential additional private shadows; never credit a small dirty set.
    pub private_shadow_bytes: u64,
}

impl HostMemoryBudget {
    /// Actual host capacity; device and observed availability bounds apply separately.
    pub fn ceiling_bytes(self) -> Option<u64> {
        (self.physical_bytes > 0).then_some(self.physical_bytes)
    }

    pub fn effective_device_usage(self, current_allocated_bytes: u64) -> Option<u64> {
        current_allocated_bytes.checked_add(self.external_gap_bytes)
    }

    /// Additional CPU bytes that fit both physical capacity and conservative
    /// CPU availability, retaining the full outstanding private-shadow charge.
    pub fn remaining_bytes(self, current_allocated_bytes: u64) -> Option<u64> {
        if self.conservative_available_bytes > self.physical_bytes {
            return None;
        }
        let ceiling = self.ceiling_bytes()?;
        let device = self.effective_device_usage(current_allocated_bytes)?;
        Some(
            ceiling
                .saturating_sub(device)
                .saturating_sub(self.private_shadow_bytes)
                .min(
                    self.conservative_available_bytes
                        .saturating_sub(self.private_shadow_bytes),
                ),
        )
    }

    pub fn ceiling_remaining_bytes(self, current_allocated_bytes: u64) -> Option<u64> {
        Some(
            self.ceiling_bytes()?
                .saturating_sub(self.effective_device_usage(current_allocated_bytes)?)
                .saturating_sub(self.private_shadow_bytes),
        )
    }

    pub(crate) fn within_ceiling(self, current_allocated_bytes: u64) -> Option<bool> {
        Some(
            self.effective_device_usage(current_allocated_bytes)?
                .checked_add(self.private_shadow_bytes)?
                <= self.ceiling_bytes()?,
        )
    }

    pub(crate) fn within_availability(self) -> Option<bool> {
        if self.physical_bytes == 0 || self.conservative_available_bytes > self.physical_bytes {
            return None;
        }
        Some(self.private_shadow_bytes <= self.conservative_available_bytes)
    }
}

pub fn host_physical_memory_bytes() -> Option<u64> {
    static PHYSICAL_BYTES: OnceLock<Option<u64>> = OnceLock::new();
    *PHYSICAL_BYTES.get_or_init(|| {
        let mut bytes = 0u64;
        let mut length = std::mem::size_of_val(&bytes);
        // SAFETY: the sysctl name is terminated and the output is correctly sized.
        let result = unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr(),
                std::ptr::addr_of_mut!(bytes).cast(),
                &mut length,
                std::ptr::null_mut(),
                0,
            )
        };
        (result == 0 && length == std::mem::size_of_val(&bytes) && bytes > 0).then_some(bytes)
    })
}

fn available_from_pages(free: u64, inactive: u64, internal: u64, page: u64) -> Option<u64> {
    if !page.is_power_of_two() {
        return None;
    }
    free.checked_add(inactive.saturating_sub(internal))?
        .checked_mul(page)
}

/// Only free pages plus a lower bound on inactive non-anonymous pages receive
/// credit. Speculative pages are already free; purgeable pages overlap other
/// categories. Active file cache receives no credit. This deliberately may
/// reject cache-heavy hosts rather than guessing at reclaimability.
fn conservative_available_memory_bytes() -> Option<u64> {
    let capacity =
        u32::try_from(std::mem::size_of::<libc::vm_statistics64>() / std::mem::size_of::<i32>())
            .ok()?;
    // SAFETY: mach_host_self returns a send right owned by this call.
    let host = unsafe { mach2::mach_init::mach_host_self() };
    if host == 0 {
        return None;
    }
    let mut stats = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count = capacity;
    // SAFETY: stats is writable for count natural_t words.
    let result = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            stats.as_mut_ptr().cast(),
            &mut count,
        )
    };
    // SAFETY: release only the send right obtained above, including on errors.
    let release = unsafe { mach2::mach_port::mach_port_deallocate(mach_task_self_, host) };
    let needed = u32::try_from(
        (std::mem::offset_of!(libc::vm_statistics64, internal_page_count)
            + std::mem::size_of::<libc::natural_t>())
            / std::mem::size_of::<i32>(),
    )
    .ok()?;
    if !valid_host_response(result, count, capacity, needed) || release != 0 {
        return None;
    }
    // SAFETY: the validated revision initializes each field read below. Do not
    // assume newer trailing libc fields were returned by this kernel.
    let (free, inactive, internal) = unsafe {
        (
            std::ptr::addr_of!((*stats.as_ptr()).free_count).read_unaligned(),
            std::ptr::addr_of!((*stats.as_ptr()).inactive_count).read_unaligned(),
            std::ptr::addr_of!((*stats.as_ptr()).internal_page_count).read_unaligned(),
        )
    };
    available_from_pages(
        u64::from(free),
        u64::from(inactive),
        u64::from(internal),
        u64::try_from(host_page_size_bytes().ok()?).ok()?,
    )
}

fn valid_host_response(result: i32, count: u32, capacity: u32, needed: u32) -> bool {
    result == libc::KERN_SUCCESS && count >= needed && count <= capacity
}

#[derive(Default)]
pub(crate) struct ExternalMemoryLedger {
    gap: AtomicU64,
    shadow: AtomicU64,
}

fn external_ledger() -> &'static Arc<ExternalMemoryLedger> {
    static LEDGER: OnceLock<Arc<ExternalMemoryLedger>> = OnceLock::new();
    LEDGER.get_or_init(|| Arc::new(ExternalMemoryLedger::default()))
}

impl ExternalMemoryLedger {
    fn reserve(self: &Arc<Self>, length: u64, shadow: u64) -> Option<ExternalMemoryCharge> {
        self.gap
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                old.checked_add(length)
            })
            .ok()?;
        if self
            .shadow
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                old.checked_add(shadow)
            })
            .is_err()
        {
            self.gap.fetch_sub(length, Ordering::AcqRel);
            return None;
        }
        Some(ExternalMemoryCharge {
            ledger: Arc::clone(self),
            gap: AtomicU64::new(length),
            shadow: AtomicU64::new(shadow),
        })
    }
}

/// Captured by the resource's deallocator, never by a Rust tensor wrapper.
/// Release and failure rollback are idempotent and acquire no allocation lock.
pub(crate) struct ExternalMemoryCharge {
    ledger: Arc<ExternalMemoryLedger>,
    gap: AtomicU64,
    shadow: AtomicU64,
}

impl ExternalMemoryCharge {
    pub(crate) fn reserve(length: u64, shadow: u64) -> Result<Arc<Self>, MetalError> {
        external_ledger()
            .reserve(length, shadow)
            .map(Arc::new)
            .ok_or_else(|| MetalError::GgufNoCopy("external memory accounting overflow".into()))
    }

    pub(crate) fn credit_allocated(&self, allocated: u64) {
        let old = self
            .gap
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |gap| {
                Some(gap.saturating_sub(allocated))
            })
            .expect("infallible gap update");
        self.ledger
            .gap
            .fetch_sub(old.min(allocated), Ordering::AcqRel);
    }

    pub(crate) fn release(&self) {
        self.ledger
            .gap
            .fetch_sub(self.gap.swap(0, Ordering::AcqRel), Ordering::AcqRel);
        self.ledger
            .shadow
            .fetch_sub(self.shadow.swap(0, Ordering::AcqRel), Ordering::AcqRel);
    }
}

impl Drop for ExternalMemoryCharge {
    fn drop(&mut self) {
        self.release();
    }
}

pub(crate) fn capture_host_memory_budget() -> Option<HostMemoryBudget> {
    let ledger = external_ledger();
    Some(HostMemoryBudget {
        physical_bytes: host_physical_memory_bytes()?,
        conservative_available_bytes: conservative_available_memory_bytes()?,
        external_gap_bytes: ledger.gap.load(Ordering::Acquire),
        private_shadow_bytes: ledger.shadow.load(Ordering::Acquire),
    })
}

pub(crate) fn admit_external_import(
    length: u64,
    shadow: u64,
    signals: MetalMemorySignals,
) -> Result<(), MetalError> {
    evaluate_metal_memory_admission_with_cpu_bytes(length, shadow, 0, signals, true)
        .refusal()
        .map_or(Ok(()), |denied| Err(denied.into()))
}

/// Validated CPU-only bounds, retaining which limit CPU cache relief can change.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostMemoryHeadroom {
    ceiling_remaining_bytes: u64,
    host_remaining_bytes: u64,
    raw_process_remaining_bytes: u64,
}

impl HostMemoryHeadroom {
    pub fn from_budget(
        host: HostMemoryBudget,
        current_allocated_bytes: u64,
        raw_process_remaining_bytes: u64,
    ) -> Option<Self> {
        host.within_ceiling(current_allocated_bytes)?;
        host.within_availability()?;
        Some(Self {
            ceiling_remaining_bytes: host.ceiling_remaining_bytes(current_allocated_bytes)?,
            host_remaining_bytes: host.remaining_bytes(current_allocated_bytes)?,
            raw_process_remaining_bytes,
        })
    }

    pub fn effective_remaining_bytes(self) -> u64 {
        match self.raw_process_remaining_bytes {
            0 => self.host_remaining_bytes,
            process => self.host_remaining_bytes.min(process),
        }
    }

    pub fn admission_reason(self, bytes: u64) -> MetalMemoryAdmissionReason {
        use MetalMemoryAdmissionReason as R;
        if bytes > self.ceiling_remaining_bytes {
            R::HostInsufficient
        } else if bytes > self.host_remaining_bytes {
            R::HostAvailabilityInsufficient
        } else if self.raw_process_remaining_bytes > 0 && bytes > self.raw_process_remaining_bytes {
            R::ProcessInsufficient
        } else if self.raw_process_remaining_bytes == 0 {
            R::AdmittedWithHostBudget
        } else {
            R::AdmittedWithProcessBudget
        }
    }
}

/// Fresh CPU-only headroom under the allocation transaction, without a context,
/// command queue, library, or inference lease. Missing telemetry stays unknown.
pub fn host_memory_headroom() -> Option<HostMemoryHeadroom> {
    let _transaction = METAL_ALLOCATION_TRANSACTION.lock();
    let device = MTLCreateSystemDefaultDevice()?;
    HostMemoryHeadroom::from_budget(
        capture_host_memory_budget()?,
        u64::try_from(device.currentAllocatedSize()).ok()?,
        MetalContext::process_limit_bytes_remaining()?,
    )
}

/// Scalar compatibility wrapper. Effective zero is exhausted, not omitted.
pub fn effective_host_memory_remaining_bytes() -> Option<u64> {
    host_memory_headroom().map(HostMemoryHeadroom::effective_remaining_bytes)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GgufMappingOwnership {
    /// Caller guarantees the immutable mapping is MAP_SHARED, not MAP_PRIVATE.
    Shared,
    /// The entire independently shadowable private span, not its CPU dirty set.
    PrivateCow { shadow_bytes: u64 },
}

impl GgufMappingOwnership {
    pub(crate) fn shadow_bytes(self, length: u64) -> Result<u64, MetalError> {
        match self {
            Self::Shared => Ok(0),
            Self::PrivateCow { shadow_bytes } if shadow_bytes >= length => Ok(shadow_bytes),
            Self::PrivateCow { .. } => Err(MetalError::GgufNoCopy(
                "private shadow span is smaller than imported window".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signals() -> MetalMemorySignals {
        MetalMemorySignals {
            recommended_max_bytes: 1_000,
            current_allocated_bytes: 100,
            process_limit_remaining_bytes: Some(0),
            host_budget: Some(HostMemoryBudget {
                physical_bytes: 1_000,
                conservative_available_bytes: 1_000,
                external_gap_bytes: 0,
                private_shadow_bytes: 0,
            }),
        }
    }

    #[test]
    fn cpu_headroom_preserves_ceiling_availability_and_raw_process_bounds() {
        use MetalMemoryAdmissionReason as R;
        let host = signals().host_budget.unwrap();
        for (available, current, process, bytes, expected) in [
            (1_000, 901, 0, 100, R::HostInsufficient),
            (99, 100, 0, 100, R::HostAvailabilityInsufficient),
            (1_000, 100, 99, 100, R::ProcessInsufficient),
            (1_000, 901, 1, 100, R::HostInsufficient),
            (99, 100, 1, 100, R::HostAvailabilityInsufficient),
            (1_000, 100, 0, 900, R::AdmittedWithHostBudget),
            (1_000, 100, 900, 900, R::AdmittedWithProcessBudget),
            (100, 100, 0, 100, R::AdmittedWithHostBudget),
            (0, 100, 0, 1, R::HostAvailabilityInsufficient),
            (0, 100, 0, 0, R::AdmittedWithHostBudget),
        ] {
            let headroom = HostMemoryHeadroom::from_budget(
                HostMemoryBudget {
                    conservative_available_bytes: available,
                    ..host
                },
                current,
                process,
            )
            .unwrap();
            assert_eq!(headroom.admission_reason(bytes), expected);
            assert_eq!(
                headroom.effective_remaining_bytes() >= bytes,
                expected.is_admitted()
            );
        }
    }

    #[test]
    fn cpu_headroom_validates_host_telemetry_and_ledger_overflow() {
        let host = signals().host_budget.unwrap();
        for invalid in [
            HostMemoryBudget {
                physical_bytes: 0,
                ..host
            },
            HostMemoryBudget {
                conservative_available_bytes: 1_001,
                ..host
            },
            HostMemoryBudget {
                external_gap_bytes: u64::MAX,
                ..host
            },
            HostMemoryBudget {
                private_shadow_bytes: u64::MAX,
                ..host
            },
        ] {
            assert!(HostMemoryHeadroom::from_budget(invalid, 100, 0).is_none());
        }
        let headroom = HostMemoryHeadroom::from_budget(
            HostMemoryBudget {
                external_gap_bytes: 20,
                private_shadow_bytes: 10,
                ..host
            },
            100,
            0,
        )
        .unwrap();
        assert_eq!(headroom.ceiling_remaining_bytes, 870);
        assert_eq!(headroom.effective_remaining_bytes(), 870);
    }

    #[test]
    fn combined_host_demand_survives_raw_zero_process_budget() {
        let exact = evaluate_metal_memory_admission_with_cpu_bytes(200, 650, 50, signals(), true);
        assert_eq!(exact.required_bytes, Some(900));
        assert_eq!(
            exact.reason,
            MetalMemoryAdmissionReason::AdmittedWithHostBudget
        );
        assert_eq!(exact.signals.host_remaining_bytes(), Some(900));
        for (metal, cpu, reserve) in [(200, 651, 50), (0, 901, 0), (900, 0, 1)] {
            let denied = evaluate_metal_memory_admission_with_cpu_bytes(
                metal,
                cpu,
                reserve,
                signals(),
                true,
            );
            assert_eq!(denied.reason, MetalMemoryAdmissionReason::HostInsufficient);
            assert!(denied.refusal().unwrap().reason().is_pressure());
        }
        assert!(
            evaluate_metal_memory_admission_with_cpu_bytes(0, 900, 0, signals(), true)
                .is_admitted()
        );
    }

    #[test]
    fn process_budget_is_additional_and_unknown_host_never_means_zero() {
        let mut observed = signals();
        observed.process_limit_remaining_bytes = Some(699);
        assert_eq!(
            evaluate_metal_memory_admission_with_cpu_bytes(200, 450, 50, observed, true).reason,
            MetalMemoryAdmissionReason::ProcessInsufficient
        );
        observed.process_limit_remaining_bytes = Some(u64::MAX);
        observed
            .host_budget
            .as_mut()
            .unwrap()
            .conservative_available_bytes = 650;
        assert_eq!(observed.host_remaining_bytes(), Some(650));
        assert_eq!(
            evaluate_metal_memory_admission_with_cpu_bytes(1, 651, 0, observed, true).reason,
            MetalMemoryAdmissionReason::HostAvailabilityInsufficient
        );
        observed.host_budget = None;
        assert_eq!(
            evaluate_metal_memory_admission(1, 0, observed, true).reason,
            MetalMemoryAdmissionReason::HostSignalUnavailable
        );
        observed.host_budget = signals().host_budget;
        observed.host_budget.as_mut().unwrap().physical_bytes = 0;
        assert_eq!(
            evaluate_metal_memory_admission(1, 0, observed, true).reason,
            MetalMemoryAdmissionReason::InvalidHostSignal
        );
        observed = signals();
        observed.process_limit_remaining_bytes = None;
        assert_eq!(
            evaluate_metal_memory_admission(1, 0, observed, true).reason,
            MetalMemoryAdmissionReason::ProcessSignalUnavailable
        );
        assert_eq!(
            evaluate_metal_memory_admission(1, 0, signals(), false).reason,
            MetalMemoryAdmissionReason::ProcessSignalUnavailable
        );
    }

    #[test]
    fn only_availability_limited_host_pressure_can_use_cpu_relief() {
        let mut observed = signals();
        observed
            .host_budget
            .as_mut()
            .unwrap()
            .conservative_available_bytes = 300;
        assert_eq!(
            evaluate_metal_memory_admission_with_cpu_bytes(301, 301, 0, observed, true).reason,
            MetalMemoryAdmissionReason::HostAvailabilityInsufficient
        );
        observed.recommended_max_bytes = 200;
        assert_eq!(
            evaluate_metal_memory_admission_with_cpu_bytes(301, 301, 0, observed, true).reason,
            MetalMemoryAdmissionReason::WorkingSetInsufficient
        );
        observed.recommended_max_bytes = 1_000;
        observed.current_allocated_bytes = 700;
        assert_eq!(
            evaluate_metal_memory_admission_with_cpu_bytes(301, 301, 0, observed, true).reason,
            MetalMemoryAdmissionReason::HostInsufficient
        );
    }

    #[test]
    fn zero_cpu_availability_allows_only_zero_cpu_demand_and_valid_telemetry() {
        let mut observed = signals();
        observed
            .host_budget
            .as_mut()
            .unwrap()
            .conservative_available_bytes = 0;
        assert_eq!(observed.host_remaining_bytes(), Some(0));
        for bytes in [0, 1] {
            assert!(evaluate_metal_memory_admission(bytes, 0, observed, true).is_admitted());
            assert_eq!(
                evaluate_metal_memory_admission_with_cpu_bytes(bytes, 1, 0, observed, true).reason,
                MetalMemoryAdmissionReason::HostAvailabilityInsufficient
            );
        }
        assert_eq!(
            evaluate_metal_memory_admission(1, 1, observed, true).reason,
            MetalMemoryAdmissionReason::HostAvailabilityInsufficient
        );
        observed.host_budget.as_mut().unwrap().private_shadow_bytes = 1;
        assert_eq!(
            evaluate_metal_memory_admission(1, 0, observed, true).reason,
            MetalMemoryAdmissionReason::HostAvailabilityInsufficient
        );
        observed
            .host_budget
            .as_mut()
            .unwrap()
            .conservative_available_bytes = 1_001;
        assert_eq!(observed.host_remaining_bytes(), None);
        observed = signals();
        observed.host_budget.as_mut().unwrap().external_gap_bytes = u64::MAX;
        assert_eq!(observed.host_remaining_bytes(), None);
        assert_eq!(
            evaluate_metal_memory_admission(u64::MAX, 1, observed, true).reason,
            MetalMemoryAdmissionReason::RequiredBytesOverflow
        );
        assert_eq!(
            HostMemoryBudget {
                physical_bytes: u64::MAX,
                ..signals().host_budget.unwrap()
            }
            .ceiling_bytes(),
            Some(u64::MAX)
        );
    }

    #[test]
    fn availability_counts_only_free_and_lower_bound_inactive_file_pages() {
        assert_eq!(available_from_pages(10, 30, 20, 4096), Some(20 * 4096));
        assert_eq!(available_from_pages(10, 30, 40, 4096), Some(10 * 4096));
        assert_eq!(available_from_pages(0, 0, 0, 4096), Some(0));
        assert_eq!(available_from_pages(1, 0, 0, 0), None);
        assert_eq!(available_from_pages(1, 0, 0, 3), None);
        assert_eq!(available_from_pages(u64::MAX, 1, 0, 4096), None);
        assert_eq!(available_from_pages(u64::MAX, 0, 0, 4096), None);
        assert!(valid_host_response(0, 20, 30, 20));
        assert!(!valid_host_response(1, 30, 30, 20));
        assert!(!valid_host_response(0, 19, 30, 20));
        assert!(!valid_host_response(0, 31, 30, 20));
    }

    #[test]
    fn gaps_and_private_shadows_consume_headroom_without_changing_raw_signals() {
        let mut observed = signals();
        observed.host_budget.as_mut().unwrap().external_gap_bytes = 20;
        observed.host_budget.as_mut().unwrap().private_shadow_bytes = 10;
        assert_eq!(observed.current_allocated_bytes, 100);
        assert_eq!(observed.effective_device_usage(), Some(120));
        assert_eq!(observed.host_remaining_bytes(), Some(870));
        let exact = evaluate_metal_memory_admission_with_cpu_bytes(200, 670, 0, observed, true);
        assert!(exact.is_admitted());
        assert_eq!(exact.working_set_headroom_bytes, Some(880));
        assert_eq!(
            evaluate_metal_memory_admission_with_cpu_bytes(200, 671, 0, observed, true).reason,
            MetalMemoryAdmissionReason::HostInsufficient
        );
        observed
            .host_budget
            .as_mut()
            .unwrap()
            .conservative_available_bytes = 500;
        assert_eq!(observed.host_remaining_bytes(), Some(490));
        assert!(
            evaluate_metal_memory_admission_with_cpu_bytes(200, 490, 0, observed, true)
                .is_admitted()
        );
        assert_eq!(
            evaluate_metal_memory_admission_with_cpu_bytes(200, 491, 0, observed, true).reason,
            MetalMemoryAdmissionReason::HostAvailabilityInsufficient
        );
    }

    #[test]
    fn shared_metal_demand_does_not_consume_cpu_availability() {
        use MetalMemoryAdmissionReason as R;
        let mut observed = signals();
        observed.process_limit_remaining_bytes = Some(740);
        let host = observed.host_budget.as_mut().unwrap();
        host.conservative_available_bytes = 150;
        host.external_gap_bytes = 20;
        host.private_shadow_bytes = 10;
        assert_eq!(observed.host_remaining_bytes(), Some(140));
        let exact = evaluate_metal_memory_admission_with_cpu_bytes(600, 100, 40, observed, true);
        assert_eq!(exact.reason, R::AdmittedWithProcessBudget);
        assert_eq!(exact.required_bytes, Some(740));
        assert_eq!(exact.working_set_headroom_bytes, Some(880));
        assert!(admit_external_import(140, 140, observed).is_ok());
        assert_eq!(
            evaluate_metal_memory_admission_with_cpu_bytes(600, 101, 40, observed, true).reason,
            R::HostAvailabilityInsufficient
        );
        assert!(admit_external_import(141, 141, observed).is_err());

        for (physical, recommended, process, reason) in [
            (869, 1000, 740, R::HostInsufficient),
            (1000, 759, 740, R::WorkingSetInsufficient),
            (1000, 1000, 739, R::ProcessInsufficient),
        ] {
            let mut short = observed;
            short.host_budget.as_mut().unwrap().physical_bytes = physical;
            short.recommended_max_bytes = recommended;
            short.process_limit_remaining_bytes = Some(process);
            assert_eq!(
                evaluate_metal_memory_admission_with_cpu_bytes(600, 100, 40, short, true).reason,
                reason
            );
        }
    }

    #[test]
    fn large_residency_fits_but_excess_capacity_and_private_exposure_refuse_before_import() {
        let gib = 1u64 << 30;
        let mut observed = signals();
        observed.recommended_max_bytes = 256 * gib;
        observed.current_allocated_bytes = 0;
        observed.host_budget = Some(HostMemoryBudget {
            physical_bytes: 256 * gib,
            conservative_available_bytes: 156 * gib,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        });
        // No additional physical-RAM margin above the device's working-set bound.
        assert!(admit_external_import(217_414 * gib / 1_000, 0, observed).is_ok());
        let called = Cell::new(false);
        let denied = admit_external_import(256 * gib + 1, 0, observed).map(|()| called.set(true));
        assert!(matches!(denied, Err(MetalError::MemoryAdmission(_))));
        assert!(!called.get());
        assert!(admit_external_import(180 * gib, 156 * gib, observed).is_err());
        assert!(
            GgufMappingOwnership::PrivateCow { shadow_bytes: 99 }
                .shadow_bytes(100)
                .is_err()
        );
    }

    #[test]
    fn import_refusal_keeps_its_typed_cause_through_metal_and_v41_load_errors() {
        use std::error::Error;
        let denied = evaluate_metal_memory_admission_with_cpu_bytes(1, 900, 0, signals(), true)
            .refusal()
            .unwrap();
        let metal = MetalError::from(denied);
        assert_eq!(
            metal
                .source()
                .unwrap()
                .downcast_ref::<MemoryAdmissionDenied>(),
            Some(&denied)
        );
        let load = crate::deepseek_v41::gguf::LoadError::from(metal);
        assert_eq!(load.as_memory_admission(), Some(&denied));
    }

    #[test]
    fn resource_owned_charge_survives_owner_clones_and_retires_exactly_once() {
        let ledger = Arc::new(ExternalMemoryLedger::default());
        let token = Arc::new(ledger.reserve(100, 100).unwrap());
        token.credit_allocated(40);
        assert_eq!(ledger.gap.load(Ordering::Acquire), 60);
        assert_eq!(ledger.shadow.load(Ordering::Acquire), 100);
        let resource = Arc::clone(&token);
        let retained = Arc::clone(&resource);
        drop(token);
        drop(resource);
        assert_eq!(ledger.gap.load(Ordering::Acquire), 60);
        assert_eq!(ledger.shadow.load(Ordering::Acquire), 100);
        retained.release();
        retained.release();
        drop(retained);
        assert_eq!(ledger.gap.load(Ordering::Acquire), 0);
        assert_eq!(ledger.shadow.load(Ordering::Acquire), 0);
    }

    #[test]
    fn creation_and_end_failure_rollback_and_accounting_overflow_do_not_leak() {
        let ledger = Arc::new(ExternalMemoryLedger::default());
        {
            let failed = ledger.reserve(100, 100).unwrap();
            failed.release();
        }
        assert_eq!(ledger.gap.load(Ordering::Acquire), 0);
        assert_eq!(ledger.shadow.load(Ordering::Acquire), 0);
        {
            let ended = ledger.reserve(100, 100).unwrap();
            ended.credit_allocated(100);
            assert_eq!(ledger.gap.load(Ordering::Acquire), 0);
            assert_eq!(ledger.shadow.load(Ordering::Acquire), 100);
        }
        assert_eq!(ledger.shadow.load(Ordering::Acquire), 0);
        let first = ledger.reserve(1, u64::MAX).unwrap();
        assert!(ledger.reserve(2, 1).is_none());
        assert_eq!(ledger.gap.load(Ordering::Acquire), 1);
        assert_eq!(ledger.shadow.load(Ordering::Acquire), u64::MAX);
        assert!(ledger.reserve(u64::MAX, 0).is_none());
        drop(first);
        assert_eq!(ledger.gap.load(Ordering::Acquire), 0);
        assert_eq!(ledger.shadow.load(Ordering::Acquire), 0);
    }

    #[test]
    fn a_second_import_cannot_spend_the_first_imports_outstanding_debt() {
        let ledger = Arc::new(ExternalMemoryLedger::default());
        let first = ledger.reserve(400, 200).unwrap();
        let mut observed = signals();
        observed.host_budget.as_mut().unwrap().external_gap_bytes =
            ledger.gap.load(Ordering::Acquire);
        observed.host_budget.as_mut().unwrap().private_shadow_bytes =
            ledger.shadow.load(Ordering::Acquire);
        assert_eq!(observed.host_remaining_bytes(), Some(300));
        assert!(admit_external_import(501, 0, observed).is_err());
        assert!(admit_external_import(500, 0, observed).is_ok());
        drop(first);
        observed.host_budget.as_mut().unwrap().external_gap_bytes =
            ledger.gap.load(Ordering::Acquire);
        observed.host_budget.as_mut().unwrap().private_shadow_bytes =
            ledger.shadow.load(Ordering::Acquire);
        assert!(admit_external_import(501, 0, observed).is_ok());
    }

    #[test]
    fn competing_import_transactions_reserve_before_the_next_snapshot() {
        let ledger = Arc::new(ExternalMemoryLedger::default());
        let charges = Mutex::new(Vec::new());
        let admitted = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    let _transaction = METAL_ALLOCATION_TRANSACTION.lock();
                    let mut observed = signals();
                    observed.host_budget.as_mut().unwrap().external_gap_bytes =
                        ledger.gap.load(Ordering::Acquire);
                    // Each import exceeds half the 900-byte remaining capacity.
                    if admit_external_import(500, 0, observed).is_ok() {
                        charges.lock().push(ledger.reserve(500, 0).unwrap());
                        admitted.fetch_add(1, Ordering::AcqRel);
                    }
                });
            }
        });
        assert_eq!(admitted.load(Ordering::Acquire), 1);
        assert_eq!(ledger.gap.load(Ordering::Acquire), 500);
        charges.lock().clear();
        assert_eq!(ledger.gap.load(Ordering::Acquire), 0);
    }

    #[test]
    #[ignore = "run alone under the production lease; two-page no-copy retain/drop witness, no model or dispatch"]
    fn tiny_private_import_debt_outlives_rust_backing_until_final_buffer_release() {
        let _lease = acquire_metal_benchmark_lease().expect("production memory/lease guard");
        let ctx = MetalContext::new().expect("context");
        let _transaction = ctx.begin_allocation_transaction();
        let page = host_page_size_bytes().expect("page");
        let length = page.checked_mul(2).expect("two pages");
        assert!(length <= 128 * 1024);
        struct Fixture(PathBuf);
        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let fixture = Fixture(
            std::env::temp_dir().join(format!("qwen-host-budget-{}-{nonce}", std::process::id())),
        );
        let writer = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&fixture.0)
            .unwrap();
        writer.set_len(length as u64).unwrap();
        drop(writer);
        let file = File::open(&fixture.0).unwrap();
        let ledger = external_ledger();
        let before_gap = ledger.gap.load(Ordering::Acquire);
        let before_shadow = ledger.shadow.load(Ordering::Acquire);
        let calls = Arc::new(AtomicUsize::new(0));
        let (retained, mapping_weak) = objc2::rc::autoreleasepool(|_| {
            // SAFETY: the fixture is private, immutable and retained until all
            // buffer references are released; its extent is exactly two pages.
            let mmap = Arc::new(
                unsafe {
                    memmap2::MmapOptions::new()
                        .len(length)
                        .map_copy(&file)
                        .unwrap()
                }
                .make_read_only()
                .unwrap(),
            );
            let mapping_weak = Arc::downgrade(&mmap);
            let observed = Arc::clone(&calls);
            let backing = ctx
                .gguf_no_copy_window_with_observer(
                    mmap,
                    0,
                    0,
                    length,
                    4,
                    GgufMappingOwnership::PrivateCow {
                        shadow_bytes: length as u64,
                    },
                    move |_, _| {
                        observed.fetch_add(1, Ordering::Release);
                    },
                )
                .expect("bounded private import");
            let retained = backing.buffer.clone();
            drop(backing);
            (retained, mapping_weak)
        });
        assert!(mapping_weak.upgrade().is_some());
        assert_eq!(calls.load(Ordering::Acquire), 0);
        assert_eq!(
            ledger.shadow.load(Ordering::Acquire),
            before_shadow + length as u64
        );
        objc2::rc::autoreleasepool(|_| drop(retained));
        assert_eq!(calls.load(Ordering::Acquire), 1);
        assert!(mapping_weak.upgrade().is_none());
        assert_eq!(ledger.gap.load(Ordering::Acquire), before_gap);
        assert_eq!(ledger.shadow.load(Ordering::Acquire), before_shadow);
    }
}
