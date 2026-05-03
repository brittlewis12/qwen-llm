//! Host-side Metal lifecycle: device, queue, library, pipeline cache,
//! and kernel-dispatch helpers.
//!
//! v1 design notes:
//! * One [`MetalContext`] per process (eventually per session). Holds
//!   `MTLDevice`, `MTLCommandQueue`, the embedded `kernels.metallib`
//!   loaded as a `MTLLibrary`, and a name → pipeline-state cache.
//! * Pipelines are created lazily and cached. We use `Mutex<HashMap>`
//!   because pipeline creation must be synchronized with cache lookup;
//!   contention is irrelevant since this only happens at first dispatch.
//! * v2: ship a precompiled `MTLBinaryArchive` next to the binary so
//!   first-dispatch isn't blocked on shader compilation.
//! * v2: `MTL4CommandBuffer` (already exposed by `objc2-metal` 0.3.2)
//!   for lower per-step encoding overhead.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString, NSURL};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
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
    #[error("could not load embedded library: {0}")]
    LoadLibrary(String),
    #[error("kernel function not found: {0}")]
    NoFunction(String),
    #[error("could not create pipeline for {0}: {1}")]
    Pipeline(String, String),
    #[error("could not create buffer of {0} bytes")]
    NoBuffer(usize),
}

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub struct MetalContext {
    pub device: Device,
    pub queue: Queue,
    pub library: Library,
    pso_cache: Arc<Mutex<HashMap<String, Pipeline>>>,
}

// SAFETY: All `Retained<ProtocolObject<dyn MTL*>>` are thread-safe per
// Apple's Metal docs (the protocol objects are themselves backed by
// thread-safe Objective-C classes; method dispatch is internally
// synchronized).
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

impl MetalContext {
    /// Initialize a Metal context backed by the embedded `kernels.metallib`.
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device.newCommandQueue().ok_or(MetalError::NoQueue)?;

        let metallib_bytes = crate::KERNELS_METALLIB;
        if metallib_bytes.is_empty() {
            return Err(MetalError::EmptyLibrary);
        }
        let library = load_library(&device, metallib_bytes)?;

        Ok(Self {
            device,
            queue,
            library,
            pso_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Look up a kernel function by name, compiling its pipeline state
    /// object on first request and caching it thereafter.
    pub fn pipeline(&self, name: &str) -> Result<Pipeline, MetalError> {
        if let Some(p) = self.pso_cache.lock().get(name) {
            return Ok(p.clone());
        }
        let func_name = NSString::from_str(name);
        let function = self
            .library
            .newFunctionWithName(&func_name)
            .ok_or_else(|| MetalError::NoFunction(name.to_string()))?;
        let pso = unsafe {
            self.device
                .newComputePipelineStateWithFunction_error(&function)
        }
        .map_err(|e: Retained<NSError>| {
            MetalError::Pipeline(name.to_string(), e.localizedDescription().to_string())
        })?;
        self.pso_cache.lock().insert(name.to_string(), pso.clone());
        Ok(pso)
    }

    /// Allocate an `MTLBuffer` and populate it from a slice of `T: bytemuck::Pod`.
    /// The buffer uses Apple Silicon shared (unified) storage so the GPU
    /// reads it directly with no copy.
    pub fn buffer_from<T: bytemuck::Pod>(
        &self,
        data: &[T],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        let n = bytes.len();
        if n == 0 {
            return self
                .device
                .newBufferWithLength_options(1, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::NoBuffer(1));
        }
        let ptr = std::ptr::NonNull::new(bytes.as_ptr() as *mut std::ffi::c_void)
            .ok_or(MetalError::NoBuffer(n))?;
        // SAFETY: `bytes` is a valid pointer to `n` bytes for the duration
        // of the call; Metal copies into the new buffer.
        let buf = unsafe {
            self.device.newBufferWithBytes_length_options(
                ptr,
                n,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::NoBuffer(n))?;
        Ok(buf)
    }

    /// Allocate an uninitialized output `MTLBuffer` of `n_bytes`.
    pub fn buffer_uninit(
        &self,
        n_bytes: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let n = n_bytes.max(1);
        self.device
            .newBufferWithLength_options(n, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::NoBuffer(n))
    }

    pub fn describe(&self) -> String {
        let name = self.device.name().to_string();
        let max_tg = self.device.maxThreadgroupMemoryLength();
        let unified = self.device.hasUnifiedMemory();
        format!("{name} | unified_memory={unified} | max_threadgroup_memory={max_tg} bytes")
    }
}

fn load_library(device: &Device, bytes: &[u8]) -> Result<Library, MetalError> {
    // Strategy: write the embedded `.metallib` to a temp file in this
    // process's tmp dir and load it via `newLibraryWithURL:error:`. The
    // library bytes are small (10s of KB even with all kernels) so the
    // I/O cost is negligible compared to pipeline-state compilation.
    //
    // The alternative — `newLibraryWithData:` — wants `dispatch_data_t`
    // which is a separate toll-free-bridged type that objc2-metal 0.3
    // doesn't expose ergonomically yet. URL-based loading is the cleanest
    // path that avoids a full FFI rewrite for a one-time lifecycle event.
    use std::io::Write;
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "qwen-kernels-{}-{}.metallib",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    {
        let mut f = std::fs::File::create(&tmp)
            .map_err(|e| MetalError::LoadLibrary(format!("temp file create: {e}")))?;
        f.write_all(bytes)
            .map_err(|e| MetalError::LoadLibrary(format!("temp file write: {e}")))?;
    }
    let path_str = tmp
        .to_str()
        .ok_or_else(|| MetalError::LoadLibrary("temp path not UTF-8".to_string()))?;
    let url = unsafe { NSURL::fileURLWithPath(&NSString::from_str(path_str)) };
    let lib = unsafe { device.newLibraryWithURL_error(&url) }.map_err(|e: Retained<NSError>| {
        MetalError::LoadLibrary(e.localizedDescription().to_string())
    })?;
    // The library has been parsed; the file is no longer needed.
    let _ = std::fs::remove_file(&tmp);
    Ok(lib)
}

// ===== Kernels =====

/// Dispatch the lifted RMSNorm-with-weight kernel.
///
/// `y[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`
///
/// CPU oracle: [`crate::forward::rms_norm`].
pub fn rms_norm_mul_f32(
    ctx: &MetalContext,
    x: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, MetalError> {
    debug_assert_eq!(x.len(), weight.len());
    let n_dim = x.len();
    let pso = ctx.pipeline("kernel_rms_norm_mul_f32")?;

    let buf_x = ctx.buffer_from(x)?;
    let buf_w = ctx.buffer_from(weight)?;
    let buf_y = ctx.buffer_uninit(n_dim * std::mem::size_of::<f32>())?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_dim: u32,
        eps: f32,
    }
    let args = Args {
        n_dim: n_dim as u32,
        eps,
    };
    let buf_args = ctx.buffer_from(&[args])?;

    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd_buf.computeCommandEncoder().expect("compute encoder");

    enc.setComputePipelineState(&pso);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*buf_args), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*buf_x), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*buf_w), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*buf_y), 0, 3);
    }

    // Threadgroup memory for partial sums: one f32 per simdgroup.
    // Apple Silicon has 32-thread simdgroups; up to 1024 threads/tg = 32 simdgroups.
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_simdgroups = (tg_threads + 31) / 32;
    unsafe {
        enc.setThreadgroupMemoryLength_atIndex(
            (n_simdgroups * std::mem::size_of::<f32>()).max(32),
            0,
        );
    }

    let grid = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_threads,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd_buf.commit();
    unsafe { cmd_buf.waitUntilCompleted() };

    // Read results back from the shared-memory buffer.
    let mut out = vec![0.0f32; n_dim];
    unsafe {
        let src = buf_y.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n_dim);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_context_initializes() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::NoDevice) => return,
            Err(e) => panic!("unexpected error: {e}"),
        };
        eprintln!("[metal] {}", ctx.describe());
    }

    /// Validate the lifted RMSNorm kernel against the CPU oracle.
    #[test]
    fn rms_norm_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) => {
                eprintln!("[metal] no kernels compiled yet — skipping");
                return;
            }
            Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        // Random-ish input + weight, n_dim covering single + multi-simdgroup.
        for &n in &[1024usize, 5120, 17408] {
            let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
            let w: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
            let eps = 1e-6;

            let cpu = crate::forward::rms_norm_pub(&x, &w, eps);
            let gpu = rms_norm_mul_f32(&ctx, &x, &w, eps).expect("metal rms_norm");

            assert_eq!(gpu.len(), cpu.len());
            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            for (a, b) in gpu.iter().zip(cpu.iter()) {
                let diff = (a - b).abs();
                max_abs = max_abs.max(diff);
                let rel = diff / b.abs().max(1e-8);
                max_rel = max_rel.max(rel);
            }
            eprintln!("[rms_norm n={n}] max|Δ|={max_abs:.2e}  max_rel={max_rel:.2e}");
            assert!(
                max_abs < 1e-4,
                "rms_norm n={n}: max|Δ|={max_abs} exceeds 1e-4"
            );
        }
    }
}
