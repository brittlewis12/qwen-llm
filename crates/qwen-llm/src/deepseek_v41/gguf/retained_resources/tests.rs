use super::*;
use std::cell::Cell;

#[test]
fn summary_bound_covers_retained_and_temporary_capacity_without_growth() {
    for (banks, dense, windows) in [
        (0, 0, 0),
        (1, 0, 1),
        (0, 8, 1),
        (3, 24, 27),
        (3, 24, 1),
        (0, 0, 32),
        (1024, 4096, 8192),
    ] {
        let b = for_counts(banks, dense, windows).unwrap();
        let mut inventory = ResourceInventory::new(b.resource_capacity).unwrap();
        let queries = Cell::new(0);
        for i in 0..b.resource_capacity {
            assert_eq!(
                inventory
                    .record(i, || {
                        queries.set(queries.get() + 1);
                        16384
                    })
                    .unwrap(),
                i
            );
            assert_eq!(
                inventory
                    .record(i, || panic!("duplicate length query"))
                    .unwrap(),
                i
            );
        }
        assert_eq!(queries.get(), b.resource_capacity);
        assert!(
            inventory
                .record(usize::MAX, || panic!("query beyond capacity"))
                .is_err()
        );
        let retained = size_of::<RetainedWeightResourceSummary>()
            + inventory.resources.capacity() * size_of::<RetainedWeightResource>()
            + HEX_DIGEST_BYTES;
        let temporary = size_of::<Vec<usize>>()
            + size_of::<String>()
            + inventory.identities.capacity() * size_of::<usize>()
            + DIGEST_FORMAT_CPU_BYTES as usize
            + if banks == 0 { 0 } else { BANK_NAME_BYTES };
        assert_eq!(retained as u64, b.retained_cpu_bytes);
        assert_eq!(temporary as u64, b.temporary_cpu_bytes);
        assert!(b.cpu_bytes >= (retained + temporary) as u64);
        assert!(b.cpu_bytes >= (banks + dense) * 1024 + windows * 4096);
        assert_eq!(b.resource_capacity as u64, (banks + dense).min(windows));
    }
}

#[test]
fn summary_zero_case_and_descriptor_mismatch_are_priced_and_checked() {
    let b = for_counts(0, 0, 0).unwrap();
    assert!(b.cpu_bytes > 0);
    assert_eq!(b.resource_capacity, 0);
    let summary = summarize(&[], &BTreeMap::new(), b).unwrap();
    assert!(summary.resources.is_empty());
    assert_eq!(summary.tensor_view_count, 0);
    assert_eq!(summary.total_unique_length, 0);
    assert_eq!(summary.max_unique_length, 0);
    assert_eq!(summary.binding_layout_sha256.len(), HEX_DIGEST_BYTES);
    assert_eq!(summary.binding_layout_sha256.capacity(), HEX_DIGEST_BYTES);
    assert_eq!(
        summary.binding_layout_sha256,
        format!(
            "{:x}",
            Sha256::digest(b"v41-live-retained-weight-bindings-v1\0")
        )
    );
    assert!(summarize(&[], &BTreeMap::new(), for_counts(1, 0, 1).unwrap()).is_err());
    assert!(for_counts(1, 0, 0).is_err());
    assert!(for_counts(0, 1, 0).is_err());
    eprintln!("empty_summary_bookkeeping={b:?}");
}

#[test]
fn summary_bound_rejects_checked_count_byte_and_allocation_overflow() {
    for (banks, dense, windows) in [
        (u64::MAX, 1, 1),
        (1, u64::MAX, 1),
        (1, 0, u64::MAX),
        (u64::MAX, 0, u64::MAX),
        (0, u64::MAX, u64::MAX),
        (
            0,
            isize::MAX as u64 / size_of::<RetainedWeightResource>() as u64 + 1,
            isize::MAX as u64 / size_of::<RetainedWeightResource>() as u64 + 1,
        ),
    ] {
        assert!(for_counts(banks, dense, windows).is_err());
    }
    let b = for_counts(3, 24, 2).unwrap();
    let controls = ControlAccounting {
        retained_cpu_bytes: u64::MAX,
        ..Default::default()
    };
    assert!(b.load_cpu_bytes(&controls, 0).is_err());
    assert!(
        b.load_cpu_bytes(&ControlAccounting::default(), u64::MAX)
            .is_err()
    );
}

#[test]
fn summary_bank_name_workspace_covers_max_layer_and_every_projection() {
    let mut name = String::with_capacity(BANK_NAME_BYTES);
    for layer in [0, 1, usize::MAX] {
        for projection in ["w1", "w2", "w3"] {
            name.clear();
            write!(&mut name, "bank/{layer}/{projection}").unwrap();
            assert!(name.len() <= BANK_NAME_BYTES);
            assert_eq!(name.capacity(), BANK_NAME_BYTES);
        }
    }
    assert_eq!(name.len(), BANK_NAME_BYTES);
}

#[test]
fn summary_shared_envelope_is_max_not_an_additional_caller_charge() {
    let b = for_counts(3, 24, 2).unwrap();
    let controls = ControlAccounting {
        payload_bytes_read: 428,
        cold_cpu_bytes: 1528,
        retained_cpu_bytes: 1276,
        source_bookkeeping_bytes: 1092,
    };
    let private = 16384;
    let disjoint = private + controls.retained_cpu_bytes + controls.source_bookkeeping_bytes;
    assert_eq!(
        b.load_cpu_bytes(&controls, private).unwrap(),
        disjoint + b.cpu_bytes
    );
    for caller in [0, b.cpu_bytes - 1, b.cpu_bytes, b.cpu_bytes + 8192] {
        let common = b.shared_cpu_envelope(caller);
        assert_eq!(common, caller.max(b.cpu_bytes));
        assert!(disjoint + common >= b.load_cpu_bytes(&controls, private).unwrap());
        if caller >= b.cpu_bytes {
            assert_eq!(disjoint + common, disjoint + caller);
        }
    }
}

#[test]
fn summary_local_cpu_admission_keeps_mwss_physical_and_pending_charges() {
    use metal::{HostMemoryBudget, MetalMemoryAdmissionReason as Reason, MetalMemorySignals};
    const GIB: u64 = 1 << 30;
    let b = for_counts(3, 24, 2).unwrap();
    let cpu = b.load_cpu_bytes(&ControlAccounting::default(), 0).unwrap();
    let metal_bytes = 10 * GIB;
    let signals = MetalMemorySignals {
        recommended_max_bytes: 16 * GIB,
        current_allocated_bytes: GIB,
        process_limit_remaining_bytes: Some(20 * GIB),
        host_budget: Some(HostMemoryBudget {
            physical_bytes: 20 * GIB,
            conservative_available_bytes: cpu + 32,
            external_gap_bytes: GIB,
            private_shadow_bytes: 32,
        }),
    };
    let decide =
        |s| metal::evaluate_metal_memory_admission_with_cpu_bytes(metal_bytes, cpu, 0, s, true);
    assert!(decide(signals).is_admitted());
    assert_eq!(decide(signals).cpu_required_bytes, Some(cpu));
    assert_eq!(decide(signals).required_bytes, Some(metal_bytes + cpu));
    let mut low = signals;
    low.host_budget
        .as_mut()
        .unwrap()
        .conservative_available_bytes -= 1;
    assert_eq!(decide(low).reason, Reason::HostAvailabilityInsufficient);
    let mut low = signals;
    low.recommended_max_bytes = 12 * GIB - 1;
    assert_eq!(decide(low).reason, Reason::WorkingSetInsufficient);
    let mut low = signals;
    low.host_budget.as_mut().unwrap().physical_bytes = 12 * GIB + cpu + 31;
    assert_eq!(decide(low).reason, Reason::HostInsufficient);
    let mut low = signals;
    low.process_limit_remaining_bytes = Some(metal_bytes + cpu - 1);
    assert_eq!(decide(low).reason, Reason::ProcessInsufficient);
    let mut low = signals;
    low.host_budget.as_mut().unwrap().external_gap_bytes = 7 * GIB;
    assert_eq!(decide(low).reason, Reason::WorkingSetInsufficient);
    let mut low = signals;
    low.host_budget.as_mut().unwrap().private_shadow_bytes += 1;
    assert_eq!(decide(low).reason, Reason::HostAvailabilityInsufficient);
}
