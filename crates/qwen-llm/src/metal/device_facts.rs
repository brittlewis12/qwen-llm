//! Device and host facts the engine can read without creating a
//! `MetalContext`: no inference lease, Metal library or command queue.
//! These are capabilities (what the device reports), not qualification
//! (which paths were measured or checked on it).

use std::fmt;

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily};
use serde::Serialize;

use crate::{PRODUCT_METALLIB_TARGET, RESEARCH_METALLIB_TARGET};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct GpuFamilySupport {
    pub apple7: bool,
    pub apple8: bool,
    pub apple9: bool,
    pub apple10: bool,
    pub metal3: bool,
    pub metal4: bool,
}

/// Stable identifier of the serialized report; bump it when a field's
/// name or meaning changes.
pub const DEVICE_FACTS_VERSION: &str = "qwen_device_info_v1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DeviceFacts {
    pub version: &'static str,
    pub name: String,
    pub architecture: String,
    pub registry_id: u64,
    pub gpu_families: GpuFamilySupport,
    pub max_threadgroup_memory_bytes: usize,
    pub max_buffer_length_bytes: usize,
    /// Sampled during each probe; memory headroom can change while a process runs.
    pub recommended_max_working_set_bytes: u64,
    pub unified_memory: bool,
    pub host_page_size_bytes: Option<usize>,
    pub physical_memory_bytes: Option<u64>,
    pub os_version: Option<String>,
    /// The macOS deployment target the libraries were configured to build
    /// for; the research library may be empty when it has no sources.
    pub product_metallib_deployment_target: &'static str,
    pub research_metallib_deployment_target: &'static str,
}

impl DeviceFacts {
    /// Query static device and host properties without creating a context,
    /// acquiring the inference lease, loading a library, or creating a queue.
    pub fn probe() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let architecture = device.architecture().name().to_string();
        Some(Self {
            version: DEVICE_FACTS_VERSION,
            name: device.name().to_string(),
            architecture,
            registry_id: device.registryID(),
            gpu_families: GpuFamilySupport {
                apple7: device.supportsFamily(MTLGPUFamily::Apple7),
                apple8: device.supportsFamily(MTLGPUFamily::Apple8),
                apple9: device.supportsFamily(MTLGPUFamily::Apple9),
                apple10: device.supportsFamily(MTLGPUFamily::Apple10),
                metal3: device.supportsFamily(MTLGPUFamily::Metal3),
                metal4: device.supportsFamily(MTLGPUFamily::Metal4),
            },
            max_threadgroup_memory_bytes: device.maxThreadgroupMemoryLength(),
            max_buffer_length_bytes: device.maxBufferLength(),
            recommended_max_working_set_bytes: device.recommendedMaxWorkingSetSize(),
            unified_memory: device.hasUnifiedMemory(),
            host_page_size_bytes: super::host_page_size_bytes().ok(),
            physical_memory_bytes: query_sysctl_value("hw.memsize"),
            os_version: query_sysctl_string("kern.osproductversion"),
            product_metallib_deployment_target: PRODUCT_METALLIB_TARGET,
            research_metallib_deployment_target: RESEARCH_METALLIB_TARGET,
        })
    }

    pub fn format_report(&self) -> String {
        format!(
            "device: {}\narchitecture: {}\nregistry_id: {}\ngpu_families: Apple7={} Apple8={} Apple9={} Apple10={} Metal3={} Metal4={}\nmax_threadgroup_memory: {} bytes\nmax_buffer_length: {} bytes\nrecommended_max_working_set: {} bytes\nunified_memory: {}\nhost_page_size: {}\nphysical_memory: {}\nos_version: {}\nproduct_metallib_deployment_target: {}\nresearch_metallib_deployment_target: {}",
            self.name,
            self.architecture,
            self.registry_id,
            self.gpu_families.apple7,
            self.gpu_families.apple8,
            self.gpu_families.apple9,
            self.gpu_families.apple10,
            self.gpu_families.metal3,
            self.gpu_families.metal4,
            self.max_threadgroup_memory_bytes,
            self.max_buffer_length_bytes,
            self.recommended_max_working_set_bytes,
            self.unified_memory,
            optional_value(self.host_page_size_bytes.map(|n| n.to_string()).as_deref()),
            optional_value(
                self.physical_memory_bytes
                    .map(|n| format!("{n} bytes"))
                    .as_deref()
            ),
            optional_value(self.os_version.as_deref()),
            self.product_metallib_deployment_target,
            self.research_metallib_deployment_target,
        )
    }
}

impl fmt::Display for DeviceFacts {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.format_report())
    }
}

fn optional_value(value: Option<&str>) -> &str {
    value.unwrap_or("unavailable")
}

fn query_sysctl_value(name: &str) -> Option<u64> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut value = 0_u64;
    let mut length = std::mem::size_of_val(&value);
    // SAFETY: the output buffer is valid for `length` writable bytes and the
    // sysctl name is a nul-terminated static string at each call site.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut value as *mut u64).cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && length == std::mem::size_of_val(&value)).then_some(value)
}

fn query_sysctl_string(name: &str) -> Option<String> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut bytes = [0_u8; 128];
    let mut length = bytes.len();
    // SAFETY: `bytes` is writable for `length` bytes and `name` is
    // nul-terminated by CString.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            bytes.as_mut_ptr().cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || length == 0 || length > bytes.len() {
        return None;
    }
    let length = bytes[..length]
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(length);
    Some(String::from_utf8_lossy(&bytes[..length]).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> DeviceFacts {
        DeviceFacts {
            version: DEVICE_FACTS_VERSION,
            name: "Test GPU".into(),
            architecture: "test-arch".into(),
            registry_id: 42,
            gpu_families: GpuFamilySupport {
                apple7: true,
                apple8: false,
                apple9: false,
                apple10: false,
                metal3: true,
                metal4: false,
            },
            max_threadgroup_memory_bytes: 32,
            max_buffer_length_bytes: 4096,
            recommended_max_working_set_bytes: 8192,
            unified_memory: true,
            host_page_size_bytes: Some(16384),
            physical_memory_bytes: None,
            os_version: Some("15.7".into()),
            product_metallib_deployment_target: "15.0",
            research_metallib_deployment_target: "15.0",
        }
    }

    /// The JSON field names and order are the `qwen info --json` device
    /// contract; a change here needs a new DEVICE_FACTS_VERSION.
    #[test]
    fn device_facts_json_is_pinned() {
        assert_eq!(
            serde_json::to_string(&sample()).unwrap(),
            concat!(
                r#"{"version":"qwen_device_info_v1","name":"Test GPU","architecture":"test-arch","#,
                r#""registry_id":42,"gpu_families":{"apple7":true,"apple8":false,"apple9":false,"#,
                r#""apple10":false,"metal3":true,"metal4":false},"max_threadgroup_memory_bytes":32,"#,
                r#""max_buffer_length_bytes":4096,"recommended_max_working_set_bytes":8192,"#,
                r#""unified_memory":true,"host_page_size_bytes":16384,"physical_memory_bytes":null,"#,
                r#""os_version":"15.7","product_metallib_deployment_target":"15.0","#,
                r#""research_metallib_deployment_target":"15.0"}"#
            )
        );
    }

    #[test]
    fn device_facts_report_formats_values_and_unavailable_host_signals() {
        let report = sample().format_report();
        assert!(report.contains("device: Test GPU"));
        assert!(report.contains("gpu_families: Apple7=true Apple8=false"));
        assert!(report.contains("host_page_size: 16384"));
        assert!(report.contains("physical_memory: unavailable"));
        assert!(report.contains("product_metallib_deployment_target: 15.0"));
    }
}
