//! Device models on which specific optimized paths were qualified.
//!
//! Pipeline probes remain responsible for checking whether a path is legal;
//! these records only describe where its performance or numerical behavior
//! was checked.

use super::DeviceFacts;
use serde::Serialize;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum QualifiedMemory {
    Any,
    AtLeastGiB(u16),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qualification {
    pub path: &'static str,
    pub device_model: &'static str,
    pub memory: QualifiedMemory,
    pub host_page_size_bytes: Option<usize>,
    pub max_buffer_length_bytes: Option<usize>,
    pub required_headroom_bytes: Option<u64>,
    pub evidence: &'static str,
    pub witness: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "facet", rename_all = "snake_case")]
pub enum QualificationDeclineReason {
    DeviceModel {
        expected: &'static str,
        actual: String,
    },
    Memory {
        minimum_gib: u16,
        actual_bytes: Option<u64>,
    },
    HostPageSize {
        expected_bytes: usize,
        actual_bytes: Option<usize>,
    },
    MaxBufferLength {
        expected_bytes: usize,
        actual_bytes: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct QualificationStatus {
    pub path: &'static str,
    pub device_model: &'static str,
    pub memory: QualifiedMemory,
    pub host_page_size_bytes: Option<usize>,
    pub max_buffer_length_bytes: Option<usize>,
    pub required_headroom_bytes: Option<u64>,
    pub evidence: &'static str,
    pub witness: &'static str,
    pub holds_for_device: bool,
    pub decline_reason: Option<QualificationDeclineReason>,
}

#[derive(Serialize)]
pub struct DeviceFactsReport<'a> {
    #[serde(flatten)]
    pub facts: &'a DeviceFacts,
    pub qualifications: Vec<QualificationStatus>,
}

impl Qualification {
    pub fn holds_for(&self, facts: &DeviceFacts) -> bool {
        self.decline_reason(facts).is_none()
    }

    pub fn decline_reason(&self, facts: &DeviceFacts) -> Option<QualificationDeclineReason> {
        if facts.name != self.device_model {
            return Some(QualificationDeclineReason::DeviceModel {
                expected: self.device_model,
                actual: facts.name.clone(),
            });
        }
        if let QualifiedMemory::AtLeastGiB(gib) = self.memory
            && !facts
                .physical_memory_bytes
                .is_some_and(|bytes| bytes >= u64::from(gib) * 1024 * 1024 * 1024)
        {
            return Some(QualificationDeclineReason::Memory {
                minimum_gib: gib,
                actual_bytes: facts.physical_memory_bytes,
            });
        }
        if let Some(expected_bytes) = self.host_page_size_bytes
            && facts.host_page_size_bytes != Some(expected_bytes)
        {
            return Some(QualificationDeclineReason::HostPageSize {
                expected_bytes,
                actual_bytes: facts.host_page_size_bytes,
            });
        }
        if let Some(expected_bytes) = self.max_buffer_length_bytes
            && facts.max_buffer_length_bytes != expected_bytes
        {
            return Some(QualificationDeclineReason::MaxBufferLength {
                expected_bytes,
                actual_bytes: facts.max_buffer_length_bytes,
            });
        }
        None
    }
}

/// Admission floor measured for the force-only A10B destination-pread path.
pub const A10B_PARALLEL_PREAD_REQUIRED_HEADROOM_BYTES: u64 = 85_608_931_328;

const NO_RUNNABLE_WITNESS: &str =
    "none: cited evidence is a historical benchmark or code commit; no replay command is recorded";

/// Return the first decline for a qualification once per process.
pub fn declined_reason_once(
    qualification: &Qualification,
    facts: &DeviceFacts,
) -> Option<QualificationDeclineReason> {
    let reason = qualification.decline_reason(facts)?;
    static LOGGED: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut logged = LOGGED.get_or_init(Default::default).lock().unwrap();
    logged.insert(qualification.path).then_some(reason)
}

const DEVICE: &str = "Apple M4 Max";

pub static DEEPSEEK_V4_MULTIGROUP_SELECTOR: Qualification = Qualification {
    path: "deepseek_v4.multigroup_selector",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "d52815a3",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_F16_MATRIX_SCORER: Qualification = Qualification {
    path: "deepseek_v4.singleton_f16_matrix_scorer",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "PERF-LOG 2026-08-08: Far Lightning F16 Matrix Candidate HOLD; Real-History F16 Lightning KILL",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_GROUPED_LONG_HCA: Qualification = Qualification {
    path: "deepseek_v4.grouped_long_hca",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "PERF-LOG 2026-08-07: Grouped Long-HCA Default GO",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_SPLITK_HCA: Qualification = Qualification {
    path: "deepseek_v4.splitk_hca",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "PERF-LOG 2026-08-07: Eight-Way Split-K HCA Default GO",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_PACKED_Q8_MATRIX_FAMILY: Qualification = Qualification {
    path: "deepseek_v4.packed_q8_matrix_family",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "4d3ed878, c9e6b729, c74d791d, 673830e4, 85a0ca34",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_PACKED_GPU_ROUTE_COMPACTION: Qualification = Qualification {
    path: "deepseek_v4.packed_gpu_route_compaction",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "878d0d49",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_PACKED_MXFP4_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_mxfp4_matrix",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "56a10b85, b1a534e8, f75f1eb1",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_PACKED_E8P32_ROUTER: Qualification = Qualification {
    path: "deepseek_v4.packed_e8p32_router",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "6ea94010",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_PACKED_QA_RAW_KV_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_q_a_raw_kv_matrix",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "6ea94010",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_PACKED_INDEXER_Q_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_indexer_q_matrix",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "30c8ef59, 85a0ca34",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_PACKED_GROUPED_Q3Q4: Qualification = Qualification {
    path: "deepseek_v4.packed_grouped_q3q4",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "c035668e, 795ee2e7",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_PACKED_SHARED_ROUTE_OVERLAP: Qualification = Qualification {
    path: "deepseek_v4.packed_shared_route_overlap",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "PERF-LOG 2026-08-07: K160 Shared/Route Overlap Default GO (K216 overlap was killed in b788bbac)",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_PACKED_GROUPED_EXPERT: Qualification = Qualification {
    path: "deepseek_v4.packed_grouped_expert",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "da9e5d02",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_ALL_SLOTS_Q3Q4: Qualification = Qualification {
    path: "deepseek_v4.all_slots_q3q4_decode",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "PERF-LOG 2026-08-07: K160 All-Slot Decode Default GO",
    witness: NO_RUNNABLE_WITNESS,
};
pub static DEEPSEEK_V4_RESIDENCY_SET: Qualification = Qualification {
    path: "deepseek_v4.residency_set",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    evidence: "PERF-LOG 2026-08-09: FRESH, K216, K160 Residency-Set Default GO; 2026-08-10 safety rollback",
    witness: NO_RUNNABLE_WITNESS,
};
pub static A3B_PARALLEL_COPY_AUTO: Qualification = Qualification {
    path: "metal_forward.a3b_parallel_copy_auto",
    device_model: DEVICE,
    memory: QualifiedMemory::AtLeastGiB(128),
    host_page_size_bytes: Some(16_384),
    max_buffer_length_bytes: Some(77_309_411_328),
    required_headroom_bytes: None,
    evidence: "0a3cef59",
    witness: "none: cited evidence is a historical benchmark; 0a3cef59 records no replay command",
};
pub static PARALLEL_COPY_EXACT_UNIFIED: Qualification = Qualification {
    path: "metal_forward.exact_unified_parallel_copy_profiles",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: Some(16_384),
    max_buffer_length_bytes: Some(77_309_411_328),
    required_headroom_bytes: Some(A10B_PARALLEL_PREAD_REQUIRED_HEADROOM_BYTES),
    evidence: "8c223624, fba3b97a",
    witness: "none: cited evidence is a historical benchmark; commits 8c223624 and fba3b97a record no replay command",
};

pub static QUALIFICATIONS: &[&Qualification] = &[
    &DEEPSEEK_V4_MULTIGROUP_SELECTOR,
    &DEEPSEEK_V4_F16_MATRIX_SCORER,
    &DEEPSEEK_V4_GROUPED_LONG_HCA,
    &DEEPSEEK_V4_SPLITK_HCA,
    &DEEPSEEK_V4_PACKED_Q8_MATRIX_FAMILY,
    &DEEPSEEK_V4_PACKED_GPU_ROUTE_COMPACTION,
    &DEEPSEEK_V4_PACKED_MXFP4_MATRIX,
    &DEEPSEEK_V4_PACKED_E8P32_ROUTER,
    &DEEPSEEK_V4_PACKED_QA_RAW_KV_MATRIX,
    &DEEPSEEK_V4_PACKED_INDEXER_Q_MATRIX,
    &DEEPSEEK_V4_PACKED_GROUPED_Q3Q4,
    &DEEPSEEK_V4_PACKED_SHARED_ROUTE_OVERLAP,
    &DEEPSEEK_V4_PACKED_GROUPED_EXPERT,
    &DEEPSEEK_V4_ALL_SLOTS_Q3Q4,
    &DEEPSEEK_V4_RESIDENCY_SET,
    &A3B_PARALLEL_COPY_AUTO,
    &PARALLEL_COPY_EXACT_UNIFIED,
];

pub fn qualification_statuses(facts: &DeviceFacts) -> Vec<QualificationStatus> {
    QUALIFICATIONS
        .iter()
        .map(|qualification| QualificationStatus {
            path: qualification.path,
            device_model: qualification.device_model,
            memory: qualification.memory,
            host_page_size_bytes: qualification.host_page_size_bytes,
            max_buffer_length_bytes: qualification.max_buffer_length_bytes,
            required_headroom_bytes: qualification.required_headroom_bytes,
            evidence: qualification.evidence,
            witness: qualification.witness,
            holds_for_device: qualification.holds_for(facts),
            decline_reason: qualification.decline_reason(facts),
        })
        .collect()
}

impl DeviceFacts {
    pub fn report(&self) -> DeviceFactsReport<'_> {
        DeviceFactsReport {
            facts: self,
            qualifications: qualification_statuses(self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::device_facts::{DEVICE_FACTS_VERSION, GpuFamilySupport};

    fn facts(name: &str, memory: Option<u64>) -> DeviceFacts {
        DeviceFacts {
            version: DEVICE_FACTS_VERSION,
            name: name.into(),
            architecture: "test".into(),
            registry_id: 0,
            gpu_families: GpuFamilySupport {
                apple7: false,
                apple8: false,
                apple9: false,
                apple10: false,
                metal3: false,
                metal4: false,
            },
            max_threadgroup_memory_bytes: 0,
            max_buffer_length_bytes: 77_309_411_328,
            recommended_max_working_set_bytes: 0,
            unified_memory: true,
            host_page_size_bytes: Some(16_384),
            physical_memory_bytes: memory,
            os_version: None,
            product_metallib_deployment_target: "",
            research_metallib_deployment_target: "",
        }
    }

    #[test]
    fn qualification_matches_model_and_memory() {
        assert!(DEEPSEEK_V4_GROUPED_LONG_HCA.holds_for(&facts(DEVICE, None)));
        assert!(DEEPSEEK_V4_SPLITK_HCA.holds_for(&facts(DEVICE, None)));
        assert!(!DEEPSEEK_V4_GROUPED_LONG_HCA.holds_for(&facts("Apple M5 Ultra", None)));
        assert!(!DEEPSEEK_V4_SPLITK_HCA.holds_for(&facts("Apple M5 Ultra", None)));
        let large = 128_u64 * 1024 * 1024 * 1024;
        assert!(A3B_PARALLEL_COPY_AUTO.holds_for(&facts(DEVICE, Some(large))));
        assert!(!A3B_PARALLEL_COPY_AUTO.holds_for(&facts(DEVICE, Some(large - 1))));
        assert!(!A3B_PARALLEL_COPY_AUTO.holds_for(&facts(DEVICE, None)));
    }

    #[test]
    fn every_record_preserves_the_original_model_gate() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let expected = [
            ("deepseek_v4.multigroup_selector", QualifiedMemory::Any),
            (
                "deepseek_v4.singleton_f16_matrix_scorer",
                QualifiedMemory::Any,
            ),
            ("deepseek_v4.grouped_long_hca", QualifiedMemory::Any),
            ("deepseek_v4.splitk_hca", QualifiedMemory::Any),
            ("deepseek_v4.packed_q8_matrix_family", QualifiedMemory::Any),
            (
                "deepseek_v4.packed_gpu_route_compaction",
                QualifiedMemory::Any,
            ),
            ("deepseek_v4.packed_mxfp4_matrix", QualifiedMemory::Any),
            ("deepseek_v4.packed_e8p32_router", QualifiedMemory::Any),
            ("deepseek_v4.packed_q_a_raw_kv_matrix", QualifiedMemory::Any),
            ("deepseek_v4.packed_indexer_q_matrix", QualifiedMemory::Any),
            ("deepseek_v4.packed_grouped_q3q4", QualifiedMemory::Any),
            (
                "deepseek_v4.packed_shared_route_overlap",
                QualifiedMemory::Any,
            ),
            ("deepseek_v4.packed_grouped_expert", QualifiedMemory::Any),
            ("deepseek_v4.all_slots_q3q4_decode", QualifiedMemory::Any),
            ("deepseek_v4.residency_set", QualifiedMemory::Any),
            (
                "metal_forward.a3b_parallel_copy_auto",
                QualifiedMemory::AtLeastGiB(128),
            ),
            (
                "metal_forward.exact_unified_parallel_copy_profiles",
                QualifiedMemory::Any,
            ),
        ];
        assert_eq!(QUALIFICATIONS.len(), expected.len());
        for (qualification, (path, memory_requirement)) in QUALIFICATIONS.iter().zip(expected) {
            assert_eq!(qualification.path, path);
            assert_eq!(qualification.device_model, "Apple M4 Max");
            assert_eq!(qualification.memory, memory_requirement);
            for (memory, expected_hold) in [
                (None, memory_requirement == QualifiedMemory::Any),
                (
                    Some(128 * GIB - 1),
                    memory_requirement == QualifiedMemory::Any,
                ),
                (Some(128 * GIB), true),
                (Some(192 * GIB), true),
            ] {
                let matching = facts("Apple M4 Max", memory);
                assert_eq!(qualification.holds_for(&matching), expected_hold, "{path}");
            }
            assert!(!qualification.holds_for(&facts("Apple M4 Pro", Some(192 * GIB))));
        }
    }

    #[test]
    fn synthetic_m4_facts_hold_and_m5_declines_on_model_first() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let m4 = facts(DEVICE, Some(128 * GIB));
        assert!(QUALIFICATIONS.iter().all(|record| record.holds_for(&m4)));

        let m5 = facts("Apple M5 Ultra", Some(128 * GIB));
        for status in qualification_statuses(&m5) {
            assert!(!status.holds_for_device, "{}", status.path);
            assert!(matches!(
                status.decline_reason,
                Some(QualificationDeclineReason::DeviceModel { .. })
            ));
        }
    }

    #[test]
    fn changed_m4_max_buffer_declines_copy_qualifications_on_that_facet() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let mut changed = facts(DEVICE, Some(128 * GIB));
        changed.max_buffer_length_bytes -= 1;
        for qualification in [&A3B_PARALLEL_COPY_AUTO, &PARALLEL_COPY_EXACT_UNIFIED] {
            assert_eq!(
                qualification.decline_reason(&changed),
                Some(QualificationDeclineReason::MaxBufferLength {
                    expected_bytes: 77_309_411_328,
                    actual_bytes: 77_309_411_327,
                })
            );
        }
        for status in qualification_statuses(&changed)
            .into_iter()
            .filter(|status| {
                status.path == A3B_PARALLEL_COPY_AUTO.path
                    || status.path == PARALLEL_COPY_EXACT_UNIFIED.path
            })
        {
            assert_eq!(
                status.decline_reason,
                Some(QualificationDeclineReason::MaxBufferLength {
                    expected_bytes: 77_309_411_328,
                    actual_bytes: 77_309_411_327,
                })
            );
        }
        let mut changed_page = facts(DEVICE, Some(128 * GIB));
        changed_page.host_page_size_bytes = Some(16_000);
        assert_eq!(
            PARALLEL_COPY_EXACT_UNIFIED.decline_reason(&changed_page),
            Some(QualificationDeclineReason::HostPageSize {
                expected_bytes: 16_384,
                actual_bytes: Some(16_000),
            })
        );
        for qualification in QUALIFICATIONS.iter().filter(|record| {
            !std::ptr::eq(**record, &A3B_PARALLEL_COPY_AUTO)
                && !std::ptr::eq(**record, &PARALLEL_COPY_EXACT_UNIFIED)
        }) {
            assert!(qualification.holds_for(&changed), "{}", qualification.path);
        }
    }
}
