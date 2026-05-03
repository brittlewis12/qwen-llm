//! Host-side Metal lifecycle.
//!
//! v1 contract:
//!
//! * One [`MetalContext`] per process (or per session, if we ever want
//!   isolation). Holds the `MTLDevice`, a single `MTLCommandQueue`, and
//!   the embedded `kernels.metallib` loaded as a `MTLLibrary`.
//! * Pipeline state objects ([`MTLComputePipelineState`]) are cached by
//!   kernel name → pipeline.
//! * v2: `MTLBinaryArchive` shipped alongside the binary so kernel
//!   compilation never blocks startup.
//! * v2: `MTL4CommandBuffer` if available (objc2-metal 0.3.2 already
//!   exposes the bindings).
//!
//! v1 implementation status: device + queue + library + PSO cache scaffold.
//! Real kernel dispatch lives in higher-level modules.

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("no Metal device available")]
    NoDevice,
    #[error("could not create command queue")]
    NoQueue,
    #[error("kernels.metallib is empty (no .metal sources compiled yet)")]
    EmptyLibrary,
    #[error("metal: {0}")]
    Other(String),
}

pub struct MetalContext {
    pub device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>,
    pub queue:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    pub library:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLLibrary>>>,
    #[allow(dead_code)] // TODO(v1): real PSO cache once kernels land
    pso_cache: Arc<Mutex<HashMap<String, ()>>>,
}

impl MetalContext {
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device.newCommandQueue().ok_or(MetalError::NoQueue)?;

        // Load embedded library if any kernels have been compiled.
        let library = if crate::KERNELS_METALLIB.is_empty() {
            None
        } else {
            // TODO(v1): wire `MTLDevice::newLibraryWithData_error_` once
            // we have at least one .metal kernel landed. Until then,
            // KERNELS_METALLIB is empty and there's nothing to load.
            None
        };

        Ok(Self {
            device,
            queue,
            library,
            pso_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Print device characterization. Useful for `qwen --info`.
    pub fn describe(&self) -> String {
        let name = self.device.name().to_string();
        let max_tg = self.device.maxThreadgroupMemoryLength();
        let unified = self.device.hasUnifiedMemory();
        format!("{name} | unified_memory={unified} | max_threadgroup_memory={max_tg} bytes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_context_initializes() {
        // Skip on CI without a GPU.
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::NoDevice) => return,
            Err(e) => panic!("unexpected error: {e}"),
        };
        let desc = ctx.describe();
        assert!(!desc.is_empty());
    }
}
