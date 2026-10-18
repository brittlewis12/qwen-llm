//! Capability requirements and evidence for optimized device paths.
//!
//! Pipeline probes remain responsible for checking whether a path is legal;
//! the Apple GPU-family gate checks only the minimum capability. Measurements
//! describe tested devices and never enable or decline a path.

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
#[serde(rename_all = "snake_case")]
pub enum QualifiedDevices {
    AppleFamily7,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementResult {
    Passed,
    Failed,
    NotRun,
    NotRecorded,
    NotApplicable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct DeviceMeasurement {
    pub device: &'static str,
    pub correctness: MeasurementResult,
    pub performance: MeasurementResult,
    pub date: &'static str,
    pub evidence: &'static str,
    pub note: Option<&'static str>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct DeviceExclusion {
    pub device: &'static str,
    pub evidence: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qualification {
    pub path: &'static str,
    pub capability_gate: QualifiedDevices,
    pub requires_unified_memory: bool,
    pub minimum_os_version: Option<&'static str>,
    pub measurements: &'static [DeviceMeasurement],
    pub excluded: &'static [DeviceExclusion],
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
    pub correctness_characterized_device: Option<&'static str>,
    pub performance_characterized_device: Option<&'static str>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "facet", rename_all = "snake_case")]
pub enum QualificationDeclineReason {
    GpuFamily {
        family: &'static str,
        supported: bool,
    },
    UnifiedMemory {
        supported: bool,
    },
    OsVersion {
        minimum_version: &'static str,
        actual_version: Option<String>,
    },
    Excluded {
        device: &'static str,
        evidence: &'static str,
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
    pub capability_gate: QualifiedDevices,
    pub requires_unified_memory: bool,
    pub minimum_os_version: Option<&'static str>,
    pub measurements: &'static [DeviceMeasurement],
    pub excluded: &'static [DeviceExclusion],
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
    pub correctness_characterized_device: Option<&'static str>,
    pub performance_characterized_device: Option<&'static str>,
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
            QualificationFacet::Excluded(device, evidence) => {
                Some(QualificationDeclineReason::Excluded { device, evidence })
            }
            QualificationFacet::GpuFamily => Some(QualificationDeclineReason::GpuFamily {
                family: "Apple7",
                supported: facts.gpu_families.apple7,
            }),
            QualificationFacet::UnifiedMemory => Some(QualificationDeclineReason::UnifiedMemory {
                supported: facts.unified_memory,
            }),
            QualificationFacet::OsVersion(minimum_version) => {
                Some(QualificationDeclineReason::OsVersion {
                    minimum_version,
                    actual_version: facts.os_version.clone(),
                })
            }
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
        if matches!(self.capability_gate, QualifiedDevices::AppleFamily7)
            && !facts.gpu_families.apple7
        {
            return Some(QualificationFacet::GpuFamily);
        }
        if self.requires_unified_memory && !facts.unified_memory {
            return Some(QualificationFacet::UnifiedMemory);
        }
        if let Some(minimum) = self.minimum_os_version
            && facts.os_is_at_least(minimum) != Some(true)
        {
            return Some(QualificationFacet::OsVersion(minimum));
        }
        if let Some(exclusion) = self
            .excluded
            .iter()
            .find(|entry| entry.device == facts.name)
        {
            return Some(QualificationFacet::Excluded(
                exclusion.device,
                exclusion.evidence,
            ));
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
    GpuFamily,
    UnifiedMemory,
    OsVersion(&'static str),
    Excluded(&'static str, &'static str),
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

const fn device_measurement(
    device: &'static str,
    correctness: MeasurementResult,
    performance: MeasurementResult,
    date: &'static str,
    evidence: &'static str,
    note: Option<&'static str>,
) -> DeviceMeasurement {
    DeviceMeasurement {
        device,
        correctness,
        performance,
        date,
        evidence,
        note,
    }
}

const MEASUREMENTS_DEEPSEEK_V4_MULTIGROUP_SELECTOR: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::Passed,
        MeasurementResult::Failed,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::Passed,
        MeasurementResult::NotApplicable,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::Passed,
        MeasurementResult::NotApplicable,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_F16_MATRIX_SCORER: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::Passed,
        MeasurementResult::Passed,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::NotApplicable,
        MeasurementResult::Passed,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::NotApplicable,
        MeasurementResult::Passed,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_GROUPED_LONG_HCA: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::Passed,
        MeasurementResult::Passed,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::Passed,
        MeasurementResult::Passed,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::Passed,
        MeasurementResult::Failed,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        Some(
            "M5 speed failure is in the witness ungrouped-online comparison arm; this path governs the grouped/split arm. Record as failed witness, not device exclusion.",
        ),
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_SPLITK_HCA: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::Passed,
        MeasurementResult::Passed,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::Passed,
        MeasurementResult::Passed,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::Passed,
        MeasurementResult::Failed,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        Some(
            "M5 speed failure is in the witness ungrouped-online comparison arm; this path governs the grouped/split arm. Record as failed witness, not device exclusion.",
        ),
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_PACKED_Q8_MATRIX_FAMILY: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::Passed,
        MeasurementResult::Passed,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::Passed,
        MeasurementResult::NotApplicable,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::Passed,
        MeasurementResult::NotApplicable,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_PACKED_GPU_ROUTE_COMPACTION: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::Passed,
        MeasurementResult::Failed,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::Passed,
        MeasurementResult::NotApplicable,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::Passed,
        MeasurementResult::NotApplicable,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_PACKED_MXFP4_MATRIX: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::Passed,
        MeasurementResult::Passed,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::Passed,
        MeasurementResult::Failed,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::Passed,
        MeasurementResult::Failed,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_PACKED_E8P32_ROUTER: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::Passed,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::Passed,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::Passed,
        MeasurementResult::NotRecorded,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_PACKED_QA_RAW_KV_MATRIX: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_PACKED_INDEXER_Q_MATRIX: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_PACKED_GROUPED_Q3Q4: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::Passed,
        MeasurementResult::Passed,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::Passed,
        MeasurementResult::NotApplicable,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::Passed,
        MeasurementResult::NotApplicable,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_PACKED_SHARED_ROUTE_OVERLAP: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_PACKED_GROUPED_EXPERT: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::Passed,
        MeasurementResult::Failed,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::Passed,
        MeasurementResult::NotApplicable,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::Passed,
        MeasurementResult::NotApplicable,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_ALL_SLOTS_Q3Q4: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::Passed,
        MeasurementResult::Passed,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::Passed,
        MeasurementResult::Passed,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::Passed,
        MeasurementResult::Passed,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_DEEPSEEK_V4_RESIDENCY_SET: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_A3B_PARALLEL_COPY_AUTO: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::Passed,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::NotApplicable,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::NotApplicable,
        MeasurementResult::NotRecorded,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const MEASUREMENTS_PARALLEL_COPY_EXACT_UNIFIED: &[DeviceMeasurement] = &[
    device_measurement(
        "Apple M4 Max",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m4max.json sha256=aa186d57e040e263df0bec60531eead39beeafedd2d5705133db42ba8d83df8b",
        None,
    ),
    device_measurement(
        "Apple M1 Pro",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-14",
        "qual-m1pro-2.json sha256=fc18b4f7ef79b7fe5f0586d964049f52ed7987679d64f28522ecd0edd8c9dd2c",
        None,
    ),
    device_measurement(
        "Apple M5 Ultra",
        MeasurementResult::NotRun,
        MeasurementResult::NotRecorded,
        "2026-10-17",
        "docs/bench/2026-10-17-m5-toolchain-baseline/results.json sha256=1ba1ce14804b9ada2330f374d792bf7b9396072ba0961785e6973801639eb2c2",
        None,
    ),
];

const NO_EXCLUSIONS: &[DeviceExclusion] = &[];

pub static DEEPSEEK_V4_MULTIGROUP_SELECTOR: Qualification = Qualification {
    path: "deepseek_v4.multigroup_selector",
    measurements: MEASUREMENTS_DEEPSEEK_V4_MULTIGROUP_SELECTOR,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
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
    correctness_characterized_device: None,
    performance_characterized_device: Some("Apple M4 Max"),
};
pub static DEEPSEEK_V4_F16_MATRIX_SCORER: Qualification = Qualification {
    path: "deepseek_v4.singleton_f16_matrix_scorer",
    measurements: MEASUREMENTS_DEEPSEEK_V4_F16_MATRIX_SCORER,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
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
    correctness_characterized_device: Some("Apple M4 Max"),
    performance_characterized_device: None,
};
pub static DEEPSEEK_V4_GROUPED_LONG_HCA: Qualification = Qualification {
    path: "deepseek_v4.grouped_long_hca",
    measurements: MEASUREMENTS_DEEPSEEK_V4_GROUPED_LONG_HCA,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "PERF-LOG 2026-08-07: Grouped Long-HCA Default GO; Apple M1 Pro (Apple7), 2026-10-12: 8192 rows 4.31 ms grouped / 1.28 ms split4 / 1.30 ms split8 / 1.14 ms split16 vs 5.17 ms online; 513 rows 0.341 ms grouped / 0.136 ms split8 vs 0.413 ms online",
    correctness: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::online_hca_matches_legacy_envelope_at_structural_boundaries -- --exact",
    correctness_kind: QualificationWitnessKind::Replay,
    performance: "QWEN_METAL_LEASE_WAIT=1 cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::profile_grouped_long_hca_against_online_direct -- --ignored --exact --nocapture --test-threads=1",
    performance_kind: QualificationWitnessKind::Replay,
    correctness_characterized_device: None,
    performance_characterized_device: None,
};
pub static DEEPSEEK_V4_SPLITK_HCA: Qualification = Qualification {
    path: "deepseek_v4.splitk_hca",
    measurements: MEASUREMENTS_DEEPSEEK_V4_SPLITK_HCA,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "PERF-LOG 2026-08-07: Eight-Way Split-K HCA Default GO; M4-only characterization uses splitk_hca_profiles_current_synthetic_terminal_tokens and skips off-M4; Apple M1 Pro (Apple7), 2026-10-12: 8192 rows split8 1.30 ms vs 5.17 ms online; 513 rows split8 0.136 ms vs 0.413 ms online",
    correctness: "cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::online_hca_matches_legacy_envelope_at_structural_boundaries -- --exact",
    correctness_kind: QualificationWitnessKind::Replay,
    performance: "QWEN_METAL_LEASE_WAIT=1 cargo test --release -p qwen-llm --lib deepseek_v4_metal::tests::profile_splitk_hca_against_grouped_long -- --ignored --exact --nocapture --test-threads=1",
    performance_kind: QualificationWitnessKind::Replay,
    correctness_characterized_device: None,
    performance_characterized_device: None,
};
pub static DEEPSEEK_V4_PACKED_Q8_MATRIX_FAMILY: Qualification = Qualification {
    path: "deepseek_v4.packed_q8_matrix_family",
    measurements: MEASUREMENTS_DEEPSEEK_V4_PACKED_Q8_MATRIX_FAMILY,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
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
    correctness_characterized_device: None,
    performance_characterized_device: Some("Apple M4 Max"),
};
pub static DEEPSEEK_V4_PACKED_GPU_ROUTE_COMPACTION: Qualification = Qualification {
    path: "deepseek_v4.packed_gpu_route_compaction",
    measurements: MEASUREMENTS_DEEPSEEK_V4_PACKED_GPU_ROUTE_COMPACTION,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
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
    correctness_characterized_device: None,
    performance_characterized_device: Some("Apple M4 Max"),
};
pub static DEEPSEEK_V4_PACKED_MXFP4_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_mxfp4_matrix",
    measurements: MEASUREMENTS_DEEPSEEK_V4_PACKED_MXFP4_MATRIX,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
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
    correctness_characterized_device: None,
    performance_characterized_device: None,
};
pub static DEEPSEEK_V4_PACKED_E8P32_ROUTER: Qualification = Qualification {
    path: "deepseek_v4.packed_e8p32_router",
    measurements: MEASUREMENTS_DEEPSEEK_V4_PACKED_E8P32_ROUTER,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
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
    correctness_characterized_device: None,
    performance_characterized_device: None,
};
pub static DEEPSEEK_V4_PACKED_QA_RAW_KV_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_q_a_raw_kv_matrix",
    measurements: MEASUREMENTS_DEEPSEEK_V4_PACKED_QA_RAW_KV_MATRIX,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
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
    correctness_characterized_device: None,
    performance_characterized_device: None,
};
pub static DEEPSEEK_V4_PACKED_INDEXER_Q_MATRIX: Qualification = Qualification {
    path: "deepseek_v4.packed_indexer_q_matrix",
    measurements: MEASUREMENTS_DEEPSEEK_V4_PACKED_INDEXER_Q_MATRIX,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
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
    correctness_characterized_device: None,
    performance_characterized_device: None,
};
pub static DEEPSEEK_V4_PACKED_GROUPED_Q3Q4: Qualification = Qualification {
    path: "deepseek_v4.packed_grouped_q3q4",
    measurements: MEASUREMENTS_DEEPSEEK_V4_PACKED_GROUPED_Q3Q4,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
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
    correctness_characterized_device: None,
    performance_characterized_device: Some("Apple M4 Max"),
};
pub static DEEPSEEK_V4_PACKED_SHARED_ROUTE_OVERLAP: Qualification = Qualification {
    path: "deepseek_v4.packed_shared_route_overlap",
    measurements: MEASUREMENTS_DEEPSEEK_V4_PACKED_SHARED_ROUTE_OVERLAP,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
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
    correctness_characterized_device: None,
    performance_characterized_device: None,
};
pub static DEEPSEEK_V4_PACKED_GROUPED_EXPERT: Qualification = Qualification {
    path: "deepseek_v4.packed_grouped_expert",
    measurements: MEASUREMENTS_DEEPSEEK_V4_PACKED_GROUPED_EXPERT,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
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
    correctness_characterized_device: None,
    performance_characterized_device: Some("Apple M4 Max"),
};
pub static DEEPSEEK_V4_ALL_SLOTS_Q3Q4: Qualification = Qualification {
    path: "deepseek_v4.all_slots_q3q4_decode",
    measurements: MEASUREMENTS_DEEPSEEK_V4_ALL_SLOTS_Q3Q4,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: None,
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
    correctness_characterized_device: None,
    performance_characterized_device: None,
};
pub static DEEPSEEK_V4_RESIDENCY_SET: Qualification = Qualification {
    path: "deepseek_v4.residency_set",
    measurements: MEASUREMENTS_DEEPSEEK_V4_RESIDENCY_SET,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: false,
    minimum_os_version: Some("15.0"),
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
    correctness_characterized_device: None,
    performance_characterized_device: None,
};
pub static A3B_PARALLEL_COPY_AUTO: Qualification = Qualification {
    path: "metal_forward.a3b_parallel_copy_auto",
    measurements: MEASUREMENTS_A3B_PARALLEL_COPY_AUTO,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: true,
    minimum_os_version: None,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: None,
    required_headroom_scope: None,
    evidence: "0a3cef59",
    correctness: "cargo test --release -p qwen-llm --lib metal_forward::tests::gguf_parallel_pread_a3b_q4_is_bit_exact -- --ignored --exact",
    correctness_kind: QualificationWitnessKind::M4CharacterizationSkip,
    performance: "none: the v0.608 PERF-LOG reports inherited v0.602 timing and Auto smoke results but records no executable benchmark command",
    performance_kind: QualificationWitnessKind::None,
    correctness_characterized_device: Some("Apple M4 Max"),
    performance_characterized_device: None,
};
pub static PARALLEL_COPY_EXACT_UNIFIED: Qualification = Qualification {
    path: "metal_forward.exact_unified_parallel_copy_profiles",
    measurements: MEASUREMENTS_PARALLEL_COPY_EXACT_UNIFIED,
    excluded: NO_EXCLUSIONS,
    capability_gate: QualifiedDevices::AppleFamily7,
    requires_unified_memory: true,
    minimum_os_version: None,
    memory: QualifiedMemory::Any,
    host_page_size_bytes: None,
    max_buffer_length_bytes: None,
    required_headroom_bytes: Some(A10B_PARALLEL_PREAD_REQUIRED_HEADROOM_BYTES),
    required_headroom_scope: Some(HeadroomRequirementScope::A10bQ4xlForcedPread),
    evidence: "8c223624, fba3b97a",
    correctness: "scope-only: cargo test --release -p qwen-llm --lib metal_forward::tests::gguf_parallel_pread_a10b_profile_is_exact_and_force_only -- --exact",
    correctness_kind: QualificationWitnessKind::ScopeOnly,
    performance: "none: commits 8c223624 and fba3b97a plus their PERF-LOG entries report measurements but record no executable A/B replay command",
    performance_kind: QualificationWitnessKind::None,
    correctness_characterized_device: None,
    performance_characterized_device: None,
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
            capability_gate: qualification.capability_gate,
            requires_unified_memory: qualification.requires_unified_memory,
            minimum_os_version: qualification.minimum_os_version,
            measurements: qualification.measurements,
            excluded: qualification.excluded,
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
            correctness_characterized_device: qualification.correctness_characterized_device,
            performance_characterized_device: qualification.performance_characterized_device,
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
            os_version: Some("15.0".into()),
            product_metallib_deployment_target: "",
            research_metallib_deployment_target: "",
            metallib_compiler_identity: "",
            metallib_sdk_version: "",
        }
    }

    #[test]
    fn all_records_use_only_apple7_capability_and_hold_for_three_generations() {
        fn capability_gate(gate: QualifiedDevices) {
            match gate {
                QualifiedDevices::AppleFamily7 => {}
            }
        }
        let devices = [
            facts("Apple M4 Max", Some(128 * 1024 * 1024 * 1024)),
            facts("Apple M1 Pro", Some(32 * 1024 * 1024 * 1024)),
            facts("Apple M5 Ultra", Some(512 * 1024 * 1024 * 1024)),
        ];
        for record in QUALIFICATIONS {
            capability_gate(record.capability_gate);
            assert_eq!(
                record.capability_gate,
                QualifiedDevices::AppleFamily7,
                "{}",
                record.path
            );
            assert!(record.excluded.is_empty(), "{}", record.path);
            for device in &devices {
                assert!(
                    record.holds_for(device),
                    "{} on {}: {:?}",
                    record.path,
                    device.name,
                    record.decline_reason(device)
                );
            }
        }
    }

    #[test]
    fn measurement_reports_preserve_outcomes_dates_and_hca_witness_scope() {
        for record in QUALIFICATIONS {
            assert_eq!(record.measurements.len(), 3, "{}", record.path);
            assert_eq!(record.measurements[0].device, "Apple M4 Max");
            assert_eq!(record.measurements[1].device, "Apple M1 Pro");
            assert_eq!(record.measurements[2].device, "Apple M5 Ultra");
            assert!(
                record.measurements.iter().all(|measurement| {
                    !measurement.date.is_empty() && measurement.evidence.contains("sha256=")
                }),
                "{}",
                record.path
            );
            assert!(record.excluded.is_empty(), "{}", record.path);
        }
        assert_ne!(MeasurementResult::NotRun, MeasurementResult::NotRecorded);
        assert_ne!(MeasurementResult::NotApplicable, MeasurementResult::Failed);

        for record in [&DEEPSEEK_V4_GROUPED_LONG_HCA, &DEEPSEEK_V4_SPLITK_HCA] {
            let m5 = record
                .measurements
                .iter()
                .find(|measurement| measurement.device == "Apple M5 Ultra")
                .unwrap();
            assert_eq!(m5.performance, MeasurementResult::Failed, "{}", record.path);
            assert!(m5.note.unwrap().contains("ungrouped-online comparison arm"));
        }
    }

    #[test]
    fn capability_decline_and_evidence_backed_exclusion_are_distinct() {
        let mut unsupported = facts("Generic Mac", None);
        unsupported.gpu_families.apple7 = false;
        for record in QUALIFICATIONS {
            assert_eq!(
                record.decline_reason(&unsupported),
                Some(QualificationDeclineReason::GpuFamily {
                    family: "Apple7",
                    supported: false
                }),
                "{}",
                record.path
            );
        }

        let mut non_unified = facts("Apple M4 Max", None);
        non_unified.unified_memory = false;
        assert_eq!(
            A3B_PARALLEL_COPY_AUTO.decline_reason(&non_unified),
            Some(QualificationDeclineReason::UnifiedMemory { supported: false })
        );
        assert!(DEEPSEEK_V4_MULTIGROUP_SELECTOR.holds_for(&non_unified));

        let mut old_os = facts("Apple M4 Max", None);
        old_os.os_version = Some("14.7.1".into());
        assert_eq!(
            DEEPSEEK_V4_RESIDENCY_SET.decline_reason(&old_os),
            Some(QualificationDeclineReason::OsVersion {
                minimum_version: "15.0",
                actual_version: Some("14.7.1".into()),
            })
        );
        assert!(DEEPSEEK_V4_ALL_SLOTS_Q3Q4.holds_for(&old_os));

        static TEST_EXCLUSIONS: &[DeviceExclusion] = &[DeviceExclusion {
            device: "Regression Mac",
            evidence: "PERF-LOG test fixture: explicit regression reproduction",
        }];
        let excluded = Qualification {
            excluded: TEST_EXCLUSIONS,
            ..A3B_PARALLEL_COPY_AUTO
        };
        assert_eq!(
            excluded.decline_reason(&facts("Regression Mac", None)),
            Some(QualificationDeclineReason::Excluded {
                device: "Regression Mac",
                evidence: "PERF-LOG test fixture: explicit regression reproduction",
            })
        );
    }

    #[test]
    fn copy_plan_constraints_are_dynamic_and_residency_headroom_stays_explicit() {
        assert!(A3B_PARALLEL_COPY_AUTO.requires_unified_memory);
        assert!(PARALLEL_COPY_EXACT_UNIFIED.requires_unified_memory);
        assert_eq!(A3B_PARALLEL_COPY_AUTO.memory, QualifiedMemory::Any);
        assert_eq!(A3B_PARALLEL_COPY_AUTO.host_page_size_bytes, None);
        assert_eq!(A3B_PARALLEL_COPY_AUTO.max_buffer_length_bytes, None);
        assert_eq!(PARALLEL_COPY_EXACT_UNIFIED.host_page_size_bytes, None);
        assert_eq!(PARALLEL_COPY_EXACT_UNIFIED.max_buffer_length_bytes, None);
        assert_eq!(
            PARALLEL_COPY_EXACT_UNIFIED.required_headroom_bytes,
            Some(A10B_PARALLEL_PREAD_REQUIRED_HEADROOM_BYTES)
        );
    }
}
