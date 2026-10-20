//! Runtime capability and library loading for Metal 4 TensorOps.

use objc2_metal::{MTLDevice, MTLGPUFamily};
use serde::Serialize;

use crate::{KERNELS_TENSOR_METALLIB, TENSOR_METALLIB_TARGET};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TensorOpsStatus {
    Available,
    Unavailable { reason: TensorOpsUnavailableReason },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum TensorOpsUnavailableReason {
    NotBuiltByToolchain,
    OsBelowTarget {
        required: &'static str,
        actual: Option<String>,
    },
    GpuFamilyAbsent {
        metal4: bool,
        apple10: bool,
    },
    LoadFailed {
        message: String,
    },
}

impl TensorOpsStatus {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }
}

/// The sole runtime gate for the optional tensor library. Returning the
/// loaded library ensures a positive result always means it can be used.
pub(crate) fn load_tensor_library(
    device: &super::Device,
    facts: &super::DeviceFacts,
) -> Result<super::Library, TensorOpsStatus> {
    if KERNELS_TENSOR_METALLIB.is_empty() || TENSOR_METALLIB_TARGET.is_empty() {
        return Err(TensorOpsStatus::Unavailable {
            reason: TensorOpsUnavailableReason::NotBuiltByToolchain,
        });
    }
    let metal4 = device.supportsFamily(MTLGPUFamily::Metal4);
    let apple10 = device.supportsFamily(MTLGPUFamily::Apple10);
    if !metal4 || !apple10 {
        return Err(TensorOpsStatus::Unavailable {
            reason: TensorOpsUnavailableReason::GpuFamilyAbsent { metal4, apple10 },
        });
    }
    if facts.os_is_at_least(TENSOR_METALLIB_TARGET) != Some(true) {
        return Err(TensorOpsStatus::Unavailable {
            reason: TensorOpsUnavailableReason::OsBelowTarget {
                required: TENSOR_METALLIB_TARGET,
                actual: facts.os_version.clone(),
            },
        });
    }
    super::context::load_library(device, KERNELS_TENSOR_METALLIB).map_err(|error| {
        TensorOpsStatus::Unavailable {
            reason: TensorOpsUnavailableReason::LoadFailed {
                message: error.to_string(),
            },
        }
    })
}

/// Probe the default device and load the library when all static gates pass.
pub fn probe_tensor_ops_status() -> TensorOpsStatus {
    let Some(device) = objc2_metal::MTLCreateSystemDefaultDevice() else {
        return TensorOpsStatus::Unavailable {
            reason: TensorOpsUnavailableReason::GpuFamilyAbsent {
                metal4: false,
                apple10: false,
            },
        };
    };
    let facts = super::DeviceFacts::from_device(&device);
    match load_tensor_library(&device, &facts) {
        Ok(_) => TensorOpsStatus::Available,
        Err(status) => status,
    }
}
