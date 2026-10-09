//! Device models on which specific optimized paths were qualified.
//!
//! Pipeline probes remain responsible for checking whether a path is legal;
//! these records only describe where its performance or numerical behavior
//! was checked.

use super::DeviceFacts;
use serde::Serialize;

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
    pub evidence: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct QualificationStatus {
    pub path: &'static str,
    pub device_model: &'static str,
    pub memory: QualifiedMemory,
    pub evidence: &'static str,
    pub holds_for_device: bool,
}

#[derive(Serialize)]
pub struct DeviceFactsReport<'a> {
    #[serde(flatten)]
    pub facts: &'a DeviceFacts,
    pub qualifications: Vec<QualificationStatus>,
}

impl Qualification {
    pub fn holds_for(&self, facts: &DeviceFacts) -> bool {
        facts.name == self.device_model
            && match self.memory {
                QualifiedMemory::Any => true,
                QualifiedMemory::AtLeastGiB(gib) => facts
                    .physical_memory_bytes
                    .is_some_and(|bytes| bytes >= u64::from(gib) * 1024 * 1024 * 1024),
            }
    }
}

const DEVICE: &str = "Apple M4 Max";

pub static DEEPSEEK_V4_MULTIGROUP_SELECTOR: Qualification = Qualification {
    path: "deepseek_v4.multigroup_selector",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "d52815a3",
};
pub static DEEPSEEK_V4_F16_MATRIX_SCORER: Qualification = Qualification {
    path: "deepseek_v4.singleton_f16_matrix_scorer",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "PERF-LOG 2026-08-08: Far Lightning F16 Matrix Candidate HOLD; Real-History F16 Lightning KILL",
};
pub static DEEPSEEK_V4_GROUPED_LONG_HCA: Qualification = Qualification {
    path: "deepseek_v4.grouped_long_hca",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "PERF-LOG 2026-08-07: Grouped Long-HCA Default GO",
};
pub static DEEPSEEK_V4_SPLITK_HCA: Qualification = Qualification {
    path: "deepseek_v4.splitk_hca",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "PERF-LOG 2026-08-07: Eight-Way Split-K HCA Default GO",
};
pub static DEEPSEEK_V4_PACKED_Q8_MATRIX_FAMILY: Qualification = Qualification {
    path: "deepseek_v4.packed_q8_matrix_family",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "4d3ed878, c9e6b729, c74d791d, 673830e4, 85a0ca34",
};
pub static DEEPSEEK_V4_PACKED_GPU_ROUTE_COMPACTION: Qualification = Qualification {
    path: "deepseek_v4.packed_gpu_route_compaction",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "878d0d49",
};
pub static DEEPSEEK_V4_PACKED_MXFP4_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_mxfp4_matrix",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "56a10b85, b1a534e8, f75f1eb1",
};
pub static DEEPSEEK_V4_PACKED_E8P32_ROUTER: Qualification = Qualification {
    path: "deepseek_v4.packed_e8p32_router",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "6ea94010",
};
pub static DEEPSEEK_V4_PACKED_QA_RAW_KV_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_q_a_raw_kv_matrix",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "6ea94010",
};
pub static DEEPSEEK_V4_PACKED_INDEXER_Q_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_indexer_q_matrix",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "30c8ef59, 85a0ca34",
};
pub static DEEPSEEK_V4_PACKED_GROUPED_Q3Q4: Qualification = Qualification {
    path: "deepseek_v4.packed_grouped_q3q4",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "c035668e, 795ee2e7",
};
pub static DEEPSEEK_V4_PACKED_SHARED_ROUTE_OVERLAP: Qualification = Qualification {
    path: "deepseek_v4.packed_shared_route_overlap",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "PERF-LOG 2026-08-07: K160 Shared/Route Overlap Default GO (K216 overlap was killed in b788bbac)",
};
pub static DEEPSEEK_V4_PACKED_GROUPED_EXPERT: Qualification = Qualification {
    path: "deepseek_v4.packed_grouped_expert",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "da9e5d02",
};
pub static DEEPSEEK_V4_ALL_SLOTS_Q3Q4: Qualification = Qualification {
    path: "deepseek_v4.all_slots_q3q4_decode",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "PERF-LOG 2026-08-07: K160 All-Slot Decode Default GO",
};
pub static DEEPSEEK_V4_RESIDENCY_SET: Qualification = Qualification {
    path: "deepseek_v4.residency_set",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "PERF-LOG 2026-08-09: FRESH, K216, K160 Residency-Set Default GO; 2026-08-10 safety rollback",
};
pub static A3B_PARALLEL_COPY_AUTO: Qualification = Qualification {
    path: "metal_forward.a3b_parallel_copy_auto",
    device_model: DEVICE,
    memory: QualifiedMemory::AtLeastGiB(128),
    evidence: "0a3cef59",
};
pub static PARALLEL_COPY_EXACT_UNIFIED: Qualification = Qualification {
    path: "metal_forward.exact_unified_parallel_copy_profiles",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "8c223624, fba3b97a",
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
            evidence: qualification.evidence,
            holds_for_device: qualification.holds_for(facts),
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
            max_buffer_length_bytes: 0,
            recommended_max_working_set_bytes: 0,
            unified_memory: true,
            host_page_size_bytes: None,
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
}
