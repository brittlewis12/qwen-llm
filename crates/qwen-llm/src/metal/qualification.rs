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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationWitnessKind {
    Replay,
    /// The command characterizes M4 behavior and explicitly skips elsewhere.
    M4CharacterizationSkip,
    /// The command checks qualification scope but does not execute the path.
    ScopeOnly,
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadroomRequirementScope {
    /// Dynamic load-admission floor for A10B Q4_K_XL forced destination pread.
    A10bQ4xlForcedPread,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "scope", content = "details", rename_all = "snake_case")]
pub enum QualifiedDevices {
    Model(&'static str),
    AppleFamily7 {
        oldest_measured: &'static str,
        newest_measured: &'static str,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qualification {
    pub path: &'static str,
    pub qualified_devices: QualifiedDevices,
    pub memory: QualifiedMemory,
    pub host_page_size_bytes: Option<usize>,
    pub max_buffer_length_bytes: Option<usize>,
    pub required_headroom_bytes: Option<u64>,
    pub required_headroom_scope: Option<HeadroomRequirementScope>,
    pub evidence: &'static str,
    pub correctness: &'static str,
    pub correctness_kind: QualificationWitnessKind,
    pub performance: &'static str,
    pub performance_kind: QualificationWitnessKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "facet", rename_all = "snake_case")]
pub enum QualificationDeclineReason {
    DeviceModel {
        expected: &'static str,
        actual: String,
    },
    GpuFamily {
        family: &'static str,
        supported: bool,
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
    pub qualified_devices: QualifiedDevices,
    pub memory: QualifiedMemory,
    pub host_page_size_bytes: Option<usize>,
    pub max_buffer_length_bytes: Option<usize>,
    pub required_headroom_bytes: Option<u64>,
    pub required_headroom_scope: Option<HeadroomRequirementScope>,
    pub evidence: &'static str,
    pub correctness: &'static str,
    pub correctness_kind: QualificationWitnessKind,
    pub performance: &'static str,
    pub performance_kind: QualificationWitnessKind,
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
        self.first_failing_facet(facts).is_none()
    }

    pub fn decline_reason(&self, facts: &DeviceFacts) -> Option<QualificationDeclineReason> {
        match self.first_failing_facet(facts)? {
            QualificationFacet::DeviceModel => {
                let QualifiedDevices::Model(expected) = self.qualified_devices else {
                    unreachable!("model facet only applies to model-scoped record")
                };
                Some(QualificationDeclineReason::DeviceModel {
                    expected,
                    actual: facts.name.clone(),
                })
            }
            QualificationFacet::GpuFamily => Some(QualificationDeclineReason::GpuFamily {
                family: "Apple7",
                supported: facts.gpu_families.apple7,
            }),
            QualificationFacet::Memory => {
                let QualifiedMemory::AtLeastGiB(minimum_gib) = self.memory else {
                    unreachable!("memory facet only applies to a memory-gated record")
                };
                Some(QualificationDeclineReason::Memory {
                    minimum_gib,
                    actual_bytes: facts.physical_memory_bytes,
                })
            }
            QualificationFacet::HostPageSize => Some(QualificationDeclineReason::HostPageSize {
                expected_bytes: self.host_page_size_bytes.unwrap(),
                actual_bytes: facts.host_page_size_bytes,
            }),
            QualificationFacet::MaxBufferLength => {
                Some(QualificationDeclineReason::MaxBufferLength {
                    expected_bytes: self.max_buffer_length_bytes.unwrap(),
                    actual_bytes: facts.max_buffer_length_bytes,
                })
            }
        }
    }

    fn first_failing_facet(&self, facts: &DeviceFacts) -> Option<QualificationFacet> {
        // Runtime gates use this allocation-free check; owned detail is built
        // only when a caller requests a diagnostic or report.
        match self.qualified_devices {
            QualifiedDevices::Model(model) if facts.name != model => {
                return Some(QualificationFacet::DeviceModel);
            }
            QualifiedDevices::AppleFamily7 { .. } if !facts.gpu_families.apple7 => {
                return Some(QualificationFacet::GpuFamily);
            }
            _ => {}
        }
        if let QualifiedMemory::AtLeastGiB(gib) = self.memory
            && !facts
                .physical_memory_bytes
                .is_some_and(|bytes| bytes >= u64::from(gib) * 1024 * 1024 * 1024)
        {
            return Some(QualificationFacet::Memory);
        }
        if self
            .host_page_size_bytes
            .is_some_and(|expected| facts.host_page_size_bytes != Some(expected))
        {
            return Some(QualificationFacet::HostPageSize);
        }
        if self
            .max_buffer_length_bytes
            .is_some_and(|expected| facts.max_buffer_length_bytes != expected)
        {
            return Some(QualificationFacet::MaxBufferLength);
        }
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QualificationFacet {
    DeviceModel,
    GpuFamily,
    Memory,
    HostPageSize,
    MaxBufferLength,
}

/// Admission floor measured for the force-only A10B destination-pread path.
pub const A10B_PARALLEL_PREAD_REQUIRED_HEADROOM_BYTES: u64 = 85_608_931_328;

/// Return the first decline for a qualification once per process.
pub fn declined_reason_once(
    qualification: &Qualification,
    facts: &DeviceFacts,
) -> Option<QualificationDeclineReason> {
    qualification.first_failing_facet(facts)?;
    static LOGGED: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut logged = LOGGED.get_or_init(Default::default).lock().unwrap();
    if !logged.insert(qualification.path) {
        return None;
    }
    qualification.decline_reason(facts)
}

const DEVICE: &str = "Apple M4 Max";

pub static DEEPSEEK_V4_MULTIGROUP_SELECTOR: Qualification = Qualification {
    path: "deepseek_v4.multigroup_selector",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "d52815a3",
    correctness: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::multigroup_selector_full_matches_current_and_fails_closed -- --exact",
    correctness_kind: QualificationWitnessKind::Replay,
    performance: "cargo test --release -p qwen-llm --features dsv4-diagnostics --lib deepseek_v4_metal::tests::current_asset_multigroup_selector_whole_token_gate -- --ignored --exact --nocapture --test-threads=1",
    performance_kind: QualificationWitnessKind::M4CharacterizationSkip,
};
pub static DEEPSEEK_V4_F16_MATRIX_SCORER: Qualification = Qualification {
    path: "deepseek_v4.singleton_f16_matrix_scorer",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "PERF-LOG 2026-08-08: Far Lightning F16 Matrix Candidate HOLD; Real-History F16 Lightning KILL",
    correctness: "cargo test --release -p qwen-llm --features dsv4-diagnostics --lib deepseek_v4_metal::tests::current_weight_boundary_optimization_differentials -- --ignored --exact --nocapture --test-threads=1",
    correctness_kind: QualificationWitnessKind::M4CharacterizationSkip,
    performance: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::profile_lightning_matrix_ceiling_at_far_context -- --ignored --exact --nocapture",
    performance_kind: QualificationWitnessKind::Replay,
};
pub static DEEPSEEK_V4_GROUPED_LONG_HCA: Qualification = Qualification {
    path: "deepseek_v4.grouped_long_hca",
    qualified_devices: QualifiedDevices::AppleFamily7 {
        oldest_measured: "Apple M1 Pro",
        newest_measured: "Apple M4 Max",
    },
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "PERF-LOG 2026-08-07: Grouped Long-HCA Default GO; Apple M1 Pro (Apple7), 2026-10-12: 8192 rows 4.31 ms grouped / 1.28 ms split4 / 1.30 ms split8 / 1.14 ms split16 vs 5.17 ms online; 513 rows 0.341 ms grouped / 0.136 ms split8 vs 0.413 ms online",
    correctness: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::online_hca_matches_legacy_envelope_at_structural_boundaries -- --exact",
    correctness_kind: QualificationWitnessKind::Replay,
    performance: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::profile_tiled_hca_at_far_context -- --ignored --exact --nocapture",
    performance_kind: QualificationWitnessKind::Replay,
};
pub static DEEPSEEK_V4_SPLITK_HCA: Qualification = Qualification {
    path: "deepseek_v4.splitk_hca",
    qualified_devices: QualifiedDevices::AppleFamily7 {
        oldest_measured: "Apple M1 Pro",
        newest_measured: "Apple M4 Max",
    },
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "PERF-LOG 2026-08-07: Eight-Way Split-K HCA Default GO; M4-only characterization uses splitk_hca_profiles_current_synthetic_terminal_tokens and skips off-M4; Apple M1 Pro (Apple7), 2026-10-12: 8192 rows split8 1.30 ms vs 5.17 ms online; 513 rows split8 0.136 ms vs 0.413 ms online",
    correctness: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::online_hca_matches_legacy_envelope_at_structural_boundaries -- --exact",
    correctness_kind: QualificationWitnessKind::Replay,
    performance: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::profile_tiled_hca_at_far_context -- --ignored --exact --nocapture",
    performance_kind: QualificationWitnessKind::Replay,
};
pub static DEEPSEEK_V4_PACKED_Q8_MATRIX_FAMILY: Qualification = Qualification {
    path: "deepseek_v4.packed_q8_matrix_family",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "4d3ed878, c9e6b729, c74d791d, 673830e4, 85a0ca34",
    correctness: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::prefill::tests::q8_f32_mma_r2c4k64_reduces_operand_rounding_and_preserves_guards -- --exact",
    correctness_kind: QualificationWitnessKind::Replay,
    performance: "cargo test --release -p qwen-llm --features dsv4-diagnostics --lib deepseek_v4_metal::prefill::tests::profile_q8_f32_mma_r2c4k64_attention_output_packet -- --ignored --exact --nocapture",
    performance_kind: QualificationWitnessKind::M4CharacterizationSkip,
};
pub static DEEPSEEK_V4_PACKED_GPU_ROUTE_COMPACTION: Qualification = Qualification {
    path: "deepseek_v4.packed_gpu_route_compaction",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "878d0d49",
    correctness: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::prefill::tests::packed_gpu_route_compaction_matches_cpu_at_capacity_boundaries -- --exact",
    correctness_kind: QualificationWitnessKind::Replay,
    performance: "cargo test --release -p qwen-llm --features dsv4-diagnostics --lib deepseek_v4_metal::tests::current_asset_packed_gpu_route_kill_packet -- --ignored --exact --nocapture",
    performance_kind: QualificationWitnessKind::M4CharacterizationSkip,
};
pub static DEEPSEEK_V4_PACKED_MXFP4_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_mxfp4_matrix",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "56a10b85, b1a534e8, f75f1eb1",
    correctness: "cargo test --release -p qwen-llm --lib metal::tests::mxfp4_f32_matrix_tile_matches_scalar_envelope_and_guards -- --exact",
    correctness_kind: QualificationWitnessKind::Replay,
    performance: "cargo test --release -p qwen-llm --lib metal::tests::mxfp4_f32_matrix_tile_k216_n2048_bucket_floor -- --ignored --exact --nocapture",
    performance_kind: QualificationWitnessKind::Replay,
};
pub static DEEPSEEK_V4_PACKED_E8P32_ROUTER: Qualification = Qualification {
    path: "deepseek_v4.packed_e8p32_router",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "6ea94010",
    correctness: "cargo test --release -p qwen-llm --lib metal::moe::tests::mat_mat_f32_router_e8p32_strict_matches_generic_bits -- --exact",
    correctness_kind: QualificationWitnessKind::Replay,
    performance: "none: the cited PERF-LOG entry records sampled E8 router intervals and a noisy bracket but no executable replay command; the available post-route packet does not isolate this router",
    performance_kind: QualificationWitnessKind::None,
};
pub static DEEPSEEK_V4_PACKED_QA_RAW_KV_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_q_a_raw_kv_matrix",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "6ea94010",
    correctness: "scope-only: cargo test --release -p qwen-llm --lib deepseek_v4_metal::prefill::tests::packed_q8_qa_kv_matrix_policy_is_scoped_and_fail_closed -- --exact",
    correctness_kind: QualificationWitnessKind::ScopeOnly,
    performance: "none: the 2026-08-08 PERF-LOG entry gives Q-A/raw-KV A/B results but no exact command or packet runner for that measurement",
    performance_kind: QualificationWitnessKind::None,
};
pub static DEEPSEEK_V4_PACKED_INDEXER_Q_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_indexer_q_matrix",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "30c8ef59, 85a0ca34",
    correctness: "scope-only: cargo test --release -p qwen-llm --lib deepseek_v4_metal::prefill::tests::packed_indexer_q_matrix_scope_is_4096_only -- --exact",
    correctness_kind: QualificationWitnessKind::ScopeOnly,
    performance: "none: the cited PERF-LOG sections record the indexed-Q measurements and rollback but contain no replay command or packet path for that schedule",
    performance_kind: QualificationWitnessKind::None,
};
pub static DEEPSEEK_V4_PACKED_GROUPED_Q3Q4: Qualification = Qualification {
    path: "deepseek_v4.packed_grouped_q3q4",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "c035668e, 795ee2e7",
    correctness: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::prefill::tests::packed_grouped_mapped_q3_q4_match_static_expert_views -- --exact",
    correctness_kind: QualificationWitnessKind::Replay,
    performance: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::prefill::k160_n128_floor::k160_n128_grouped_q3q4_representative_layer_floor -- --ignored --exact --nocapture",
    performance_kind: QualificationWitnessKind::M4CharacterizationSkip,
};
pub static DEEPSEEK_V4_PACKED_SHARED_ROUTE_OVERLAP: Qualification = Qualification {
    path: "deepseek_v4.packed_shared_route_overlap",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "PERF-LOG 2026-08-07: K160 Shared/Route Overlap Default GO (K216 overlap was killed in b788bbac)",
    correctness: "scope-only: cargo test --release -p qwen-llm --lib deepseek_v4_metal::prefill::tests::packed_scope_helpers_use_their_named_device_qualifications -- --exact",
    correctness_kind: QualificationWitnessKind::ScopeOnly,
    performance: "none: the PERF-LOG records the serial/default wall pair, but no exact command or dedicated shared-route overlap runner is recorded",
    performance_kind: QualificationWitnessKind::None,
};
pub static DEEPSEEK_V4_PACKED_GROUPED_EXPERT: Qualification = Qualification {
    path: "deepseek_v4.packed_grouped_expert",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "da9e5d02",
    correctness: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::prefill::tests::packed_grouped_iq2_xs_iq3_xxs_matches_bucket_path -- --exact",
    correctness_kind: QualificationWitnessKind::Replay,
    performance: "QWEN_DSV4_PACKED_GROUPED_EXPERTS=1 QWEN_DSV4_PACKED_GROUPED_SAMPLES=5 QWEN_DSV4_PACKED_GROUPED_ONLY_N=128 QWEN_MATMAT_IQ3_XXS_MM=1 cargo test --release -p qwen-llm --features dsv4-diagnostics --lib deepseek_v4_metal::tests::current_asset_packed_grouped_expert_integration_packet -- --ignored --exact --nocapture",
    performance_kind: QualificationWitnessKind::M4CharacterizationSkip,
};
pub static DEEPSEEK_V4_ALL_SLOTS_Q3Q4: Qualification = Qualification {
    path: "deepseek_v4.all_slots_q3q4_decode",
    qualified_devices: QualifiedDevices::AppleFamily7 {
        oldest_measured: "Apple M1 Pro",
        newest_measured: "Apple M4 Max",
    },
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "PERF-LOG 2026-08-07: K160 All-Slot Decode Default GO; 2026-10-14 forced K160 fast vs generic all-slot (the off-M4 incumbent), dispatch-census verified, interleaved medians: M1 Pro 0.62 vs 2.58 ms, M4 Max 0.24 vs 0.67 ms; correctness witness passing on both",
    correctness: "QWEN_METAL_LEASE_WAIT=1 cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::all_slot_q3q4_fast_k160_matches_generic_and_serial_with_dispatch_evidence -- --exact --nocapture",
    correctness_kind: QualificationWitnessKind::Replay,
    performance: "QWEN_METAL_LEASE_WAIT=1 cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::profile_all_slot_routed_experts_against_serial_indexed -- --ignored --exact --nocapture",
    performance_kind: QualificationWitnessKind::Replay,
};
pub static DEEPSEEK_V4_RESIDENCY_SET: Qualification = Qualification {
    path: "deepseek_v4.residency_set",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "PERF-LOG 2026-08-09: FRESH, K216, K160 Residency-Set Default GO; 2026-08-10 safety rollback",
    correctness: "scope-only: cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::model_residency_set_scope_is_exact -- --exact",
    correctness_kind: QualificationWitnessKind::ScopeOnly,
    performance: "none: PERF-LOG records product load/prefill runs and the safety rollback, but no replay command or packet runner for those pairs is recorded",
    performance_kind: QualificationWitnessKind::None,
};
pub static A3B_PARALLEL_COPY_AUTO: Qualification = Qualification {
    path: "metal_forward.a3b_parallel_copy_auto",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::AtLeastGiB(128),
    host_page_size_bytes: Some(16_384),
    max_buffer_length_bytes: Some(77_309_411_328),
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "0a3cef59",
    correctness: "cargo test --release -p qwen-llm --lib metal_forward::tests::gguf_parallel_pread_a3b_q4_is_bit_exact -- --ignored --exact",
    correctness_kind: QualificationWitnessKind::M4CharacterizationSkip,
    performance: "none: the v0.608 PERF-LOG reports inherited v0.602 timing and Auto smoke results but records no executable benchmark command",
    performance_kind: QualificationWitnessKind::None,
};
pub static PARALLEL_COPY_EXACT_UNIFIED: Qualification = Qualification {
    path: "metal_forward.exact_unified_parallel_copy_profiles",
    qualified_devices: QualifiedDevices::Model(DEVICE),
    memory: QualifiedMemory::Any,
    host_page_size_bytes: Some(16_384),
    max_buffer_length_bytes: Some(77_309_411_328),
    required_headroom_bytes: Some(A10B_PARALLEL_PREAD_REQUIRED_HEADROOM_BYTES),
    required_headroom_scope: Some(HeadroomRequirementScope::A10bQ4xlForcedPread),
    evidence: "8c223624, fba3b97a",
    correctness: "scope-only: cargo test --release -p qwen-llm --lib metal_forward::tests::gguf_parallel_pread_a10b_profile_is_exact_and_force_only -- --exact",
    correctness_kind: QualificationWitnessKind::ScopeOnly,
    performance: "none: commits 8c223624 and fba3b97a plus their PERF-LOG entries report measurements but record no executable A/B replay command",
    performance_kind: QualificationWitnessKind::None,
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

pub static DEEPSEEK_V4_QUALIFICATIONS: &[&Qualification] = &[
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
];

pub fn qualification_statuses(facts: &DeviceFacts) -> Vec<QualificationStatus> {
    QUALIFICATIONS
        .iter()
        .map(|qualification| QualificationStatus {
            path: qualification.path,
            qualified_devices: qualification.qualified_devices,
            memory: qualification.memory,
            host_page_size_bytes: qualification.host_page_size_bytes,
            max_buffer_length_bytes: qualification.max_buffer_length_bytes,
            required_headroom_bytes: qualification.required_headroom_bytes,
            required_headroom_scope: qualification.required_headroom_scope,
            evidence: qualification.evidence,
            correctness: qualification.correctness,
            correctness_kind: qualification.correctness_kind,
            performance: qualification.performance,
            performance_kind: qualification.performance_kind,
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
                apple7: true,
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

    fn is_single_cargo_test_command(value: &str) -> bool {
        if value.contains(';') || value.contains('\n') || value.matches("cargo test").count() != 1 {
            return false;
        }
        let Some((environment, arguments)) = value.split_once("cargo test") else {
            return false;
        };
        let environment_is_valid = environment.is_empty()
            || environment
                .split_whitespace()
                .all(|assignment| assignment.contains('='));
        let arguments_are_command_tokens = arguments.split_whitespace().all(|token| {
            token
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || "_./:=+-".contains(character))
        });
        environment_is_valid && arguments.starts_with(' ') && arguments_are_command_tokens
    }

    fn has_valid_witness_shape(value: &str) -> bool {
        if let Some(explanation) = value.strip_prefix("none: ") {
            return !explanation.trim().is_empty();
        }
        let command = value.strip_prefix("scope-only: ").unwrap_or(value);
        is_single_cargo_test_command(command)
    }

    #[test]
    fn qualification_matches_model_and_memory() {
        assert!(DEEPSEEK_V4_GROUPED_LONG_HCA.holds_for(&facts(DEVICE, None)));
        assert!(DEEPSEEK_V4_SPLITK_HCA.holds_for(&facts(DEVICE, None)));
        assert!(DEEPSEEK_V4_GROUPED_LONG_HCA.holds_for(&facts("Apple M5 Ultra", None)));
        assert!(DEEPSEEK_V4_SPLITK_HCA.holds_for(&facts("Apple M5 Ultra", None)));
        let large = 128_u64 * 1024 * 1024 * 1024;
        assert!(A3B_PARALLEL_COPY_AUTO.holds_for(&facts(DEVICE, Some(large))));
        assert!(!A3B_PARALLEL_COPY_AUTO.holds_for(&facts(DEVICE, Some(large - 1))));
        assert!(!A3B_PARALLEL_COPY_AUTO.holds_for(&facts(DEVICE, None)));
    }

    #[test]
    fn every_record_keeps_its_scope_and_memory_gate() {
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
            if matches!(qualification.qualified_devices, QualifiedDevices::Model(_)) {
                assert_eq!(
                    qualification.qualified_devices,
                    QualifiedDevices::Model("Apple M4 Max")
                );
            }
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
            if matches!(qualification.qualified_devices, QualifiedDevices::Model(_)) {
                assert!(!qualification.holds_for(&facts("Apple M4 Pro", Some(192 * GIB))));
            }
        }
    }

    #[test]
    fn family_scopes_name_distinct_measurements_at_both_supported_ends() {
        let family_records = QUALIFICATIONS
            .iter()
            .filter_map(|record| match record.qualified_devices {
                QualifiedDevices::AppleFamily7 {
                    oldest_measured,
                    newest_measured,
                } => Some((record.path, oldest_measured, newest_measured)),
                QualifiedDevices::Model(_) => None,
            })
            .collect::<Vec<_>>();
        assert!(!family_records.is_empty());
        for (path, oldest, newest) in family_records {
            assert!(!oldest.trim().is_empty(), "{path}");
            assert!(!newest.trim().is_empty(), "{path}");
            assert_ne!(oldest, newest, "{path}");
        }
    }

    #[test]
    fn every_qualification_witness_is_one_command_or_an_explicit_non_replay() {
        for record in QUALIFICATIONS {
            assert!(
                has_valid_witness_shape(record.correctness),
                "{} correctness witness: {}",
                record.path,
                record.correctness
            );
            assert!(
                has_valid_witness_shape(record.performance),
                "{} performance witness: {}",
                record.path,
                record.performance
            );
        }
    }

    #[test]
    fn synthetic_m4_holds_all_and_m5_holds_family_scoped_records() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let m4 = facts(DEVICE, Some(128 * GIB));
        assert!(QUALIFICATIONS.iter().all(|record| record.holds_for(&m4)));

        let m5 = facts("Apple M5 Ultra", Some(128 * GIB));
        for (record, status) in QUALIFICATIONS.iter().zip(qualification_statuses(&m5)) {
            let family = matches!(
                record.qualified_devices,
                QualifiedDevices::AppleFamily7 { .. }
            );
            assert_eq!(status.holds_for_device, family, "{}", status.path);
            if !family {
                assert!(matches!(
                    status.decline_reason,
                    Some(QualificationDeclineReason::DeviceModel { .. })
                ));
            }
        }
    }

    #[test]
    fn apple7_scope_holds_only_for_family_scoped_records() {
        let m1 = facts("Apple M1 Pro", None);
        assert!(DEEPSEEK_V4_GROUPED_LONG_HCA.holds_for(&m1));
        assert!(DEEPSEEK_V4_SPLITK_HCA.holds_for(&m1));
        assert!(DEEPSEEK_V4_ALL_SLOTS_Q3Q4.holds_for(&m1));
        for (record, status) in QUALIFICATIONS.iter().zip(qualification_statuses(&m1)) {
            if matches!(
                record.qualified_devices,
                QualifiedDevices::AppleFamily7 { .. }
            ) {
                assert!(status.holds_for_device, "{}", status.path);
            } else {
                assert!(!status.holds_for_device, "{}", status.path);
                assert!(
                    matches!(
                        status.decline_reason,
                        Some(QualificationDeclineReason::DeviceModel { .. })
                    ),
                    "{}",
                    status.path
                );
            }
        }

        let mut no_apple7 = facts("Apple GPU without Apple7", None);
        no_apple7.gpu_families.apple7 = false;
        for qualification in [&DEEPSEEK_V4_GROUPED_LONG_HCA, &DEEPSEEK_V4_SPLITK_HCA] {
            assert_eq!(
                qualification.decline_reason(&no_apple7),
                Some(QualificationDeclineReason::GpuFamily {
                    family: "Apple7",
                    supported: false,
                })
            );
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
