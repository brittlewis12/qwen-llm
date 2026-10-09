//! Device models on which specific optimized paths were qualified.
//!
//! Pipeline probes remain responsible for checking whether a path is legal;
//! these records only describe where its performance or numerical behavior
//! was checked.

use super::DeviceFacts;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    evidence: "unrecorded",
};
pub static DEEPSEEK_V4_F16_MATRIX_SCORER: Qualification = Qualification {
    path: "deepseek_v4.f16_matrix_scorer",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "unrecorded",
};
pub static DEEPSEEK_V4_LONG_HCA: Qualification = Qualification {
    path: "deepseek_v4.long_hca",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "unrecorded",
};
pub static DEEPSEEK_V4_PACKED_Q8_COMPRESSOR_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_q8_compressor_matrix",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "unrecorded",
};
pub static DEEPSEEK_V4_PACKED_GROUPED_EXPERT: Qualification = Qualification {
    path: "deepseek_v4.packed_grouped_expert",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "unrecorded",
};
pub static DEEPSEEK_V4_ALL_SLOTS_Q3Q4: Qualification = Qualification {
    path: "deepseek_v4.all_slots_q3q4_decode",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "unrecorded",
};
pub static DEEPSEEK_V4_RESIDENCY_SET: Qualification = Qualification {
    path: "deepseek_v4.residency_set",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "unrecorded",
};
pub static A3B_PARALLEL_COPY_AUTO: Qualification = Qualification {
    path: "metal_forward.a3b_parallel_copy_auto",
    device_model: DEVICE,
    memory: QualifiedMemory::AtLeastGiB(128),
    evidence: "unrecorded",
};
pub static PARALLEL_COPY_EXACT_UNIFIED: Qualification = Qualification {
    path: "metal_forward.exact_unified_parallel_copy_profiles",
    device_model: DEVICE,
    memory: QualifiedMemory::Any,
    evidence: "unrecorded",
};

pub static QUALIFICATIONS: &[&Qualification] = &[
    &DEEPSEEK_V4_MULTIGROUP_SELECTOR,
    &DEEPSEEK_V4_F16_MATRIX_SCORER,
    &DEEPSEEK_V4_LONG_HCA,
    &DEEPSEEK_V4_PACKED_Q8_COMPRESSOR_MATRIX,
    &DEEPSEEK_V4_PACKED_GROUPED_EXPERT,
    &DEEPSEEK_V4_ALL_SLOTS_Q3Q4,
    &DEEPSEEK_V4_RESIDENCY_SET,
    &A3B_PARALLEL_COPY_AUTO,
    &PARALLEL_COPY_EXACT_UNIFIED,
];

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
        assert!(DEEPSEEK_V4_LONG_HCA.holds_for(&facts(DEVICE, None)));
        assert!(!DEEPSEEK_V4_LONG_HCA.holds_for(&facts("Apple M5 Ultra", None)));
        let large = 128_u64 * 1024 * 1024 * 1024;
        assert!(A3B_PARALLEL_COPY_AUTO.holds_for(&facts(DEVICE, Some(large))));
        assert!(!A3B_PARALLEL_COPY_AUTO.holds_for(&facts(DEVICE, Some(large - 1))));
        assert!(!A3B_PARALLEL_COPY_AUTO.holds_for(&facts(DEVICE, None)));
    }
}
