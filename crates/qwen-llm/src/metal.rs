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

/// Single-dispatch Q4_K mat-vec on persistent buffers. Useful for testing
/// and as the building block for `mat_vec_q4_k_f32_chained`.
pub fn mat_vec_q4_k_f32_bufs(
    ctx: &MetalContext,
    buf_args: &Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_w: &Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_x: &Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_y: &Retained<ProtocolObject<dyn MTLBuffer>>,
    n_out: usize,
) -> Result<(), MetalError> {
    mat_vec_q4_k_f32_chained(ctx, buf_args, buf_w, buf_x, buf_y, n_out, 1)
}

/// Chain `n_dispatches` Q4_K mat-vec dispatches into a single command
/// buffer with one `waitUntilCompleted` at the end. Approximates the
/// per-step economics of a layered forward where many kernels run back
/// to back without serialized waits. Useful for benchmarking how
/// dispatch overhead amortizes; for production inference the layers
/// would each have different weights, but the per-dispatch cost is
/// approximately equivalent.
pub fn mat_vec_q4_k_f32_chained(
    ctx: &MetalContext,
    buf_args: &Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_w: &Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_x: &Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_y: &Retained<ProtocolObject<dyn MTLBuffer>>,
    n_out: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let pso = ctx.pipeline("kernel_mat_vec_q4_K_f32")?;
    const NR0: usize = 2;
    const NSG: usize = 2;
    let rows_per_tg = NR0 * NSG;
    let n_tg = n_out.div_ceil(rows_per_tg);

    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd_buf.computeCommandEncoder().expect("compute encoder");
    enc.setComputePipelineState(&pso);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_args.as_ref()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_w.as_ref()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(buf_x.as_ref()), 0, 2);
        enc.setBuffer_offset_atIndex(Some(buf_y.as_ref()), 0, 3);
    }
    let grid = MTLSize {
        width: n_tg,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: NSG * 32,
        height: 1,
        depth: 1,
    };
    for _ in 0..n_dispatches {
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }
    enc.endEncoding();
    cmd_buf.commit();
    unsafe { cmd_buf.waitUntilCompleted() };
    Ok(())
}

/// Q6_K mat-vec on persistent buffers. See `mat_vec_q4_k_f32_chained` for
/// rationale. Q6_K has the same NSG/NR0=2/2 launch geometry as Q4_K.
pub fn mat_vec_q6_k_f32_chained(
    ctx: &MetalContext,
    buf_args: &Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_w: &Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_x: &Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_y: &Retained<ProtocolObject<dyn MTLBuffer>>,
    n_out: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let pso = ctx.pipeline("kernel_mat_vec_q6_K_f32")?;
    const NR0: usize = 2;
    const NSG: usize = 2;
    let rows_per_tg = NR0 * NSG;
    let n_tg = n_out.div_ceil(rows_per_tg);
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd_buf.computeCommandEncoder().expect("compute encoder");
    enc.setComputePipelineState(&pso);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_args.as_ref()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_w.as_ref()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(buf_x.as_ref()), 0, 2);
        enc.setBuffer_offset_atIndex(Some(buf_y.as_ref()), 0, 3);
    }
    let grid = MTLSize {
        width: n_tg,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: NSG * 32,
        height: 1,
        depth: 1,
    };
    for _ in 0..n_dispatches {
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }
    enc.endEncoding();
    cmd_buf.commit();
    unsafe { cmd_buf.waitUntilCompleted() };
    Ok(())
}

pub fn mat_vec_q6_k_f32(
    ctx: &MetalContext,
    weight_bytes: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, MetalError> {
    debug_assert!(n_in % 256 == 0);
    debug_assert_eq!(weight_bytes.len(), n_out * (n_in / 256) * 210);
    debug_assert_eq!(x.len(), n_in);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }
    let buf_args = ctx.buffer_from(&[Args {
        n_in: n_in as u32,
        n_out: n_out as u32,
    }])?;
    let buf_w = ctx.buffer_from(weight_bytes)?;
    let buf_x = ctx.buffer_from(x)?;
    let buf_y = ctx.buffer_uninit(n_out * std::mem::size_of::<f32>())?;
    mat_vec_q6_k_f32_chained(ctx, &buf_args, &buf_w, &buf_x, &buf_y, n_out, 1)?;
    let mut out = vec![0.0f32; n_out];
    unsafe {
        let src = buf_y.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n_out);
    }
    Ok(out)
}

/// Dispatch the Q4_K mat-vec kernel directly on raw block_q4_K bytes.
///
/// `weight_bytes` is `n_out * (n_in/256) * 144` bytes; `x` is `[n_in]` f32.
/// Returns `[n_out]` f32 logits.
///
/// CPU oracle: dequant via `crate::codec::dequant_to_f32` then
/// `crate::forward::mat_vec_pub`.
pub fn mat_vec_q4_k_f32(
    ctx: &MetalContext,
    weight_bytes: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, MetalError> {
    debug_assert!(n_in % 256 == 0, "Q4_K requires n_in % 256 == 0");
    debug_assert_eq!(weight_bytes.len(), n_out * (n_in / 256) * 144);
    debug_assert_eq!(x.len(), n_in);

    let pso = ctx.pipeline("kernel_mat_vec_q4_K_f32")?;
    let buf_w = ctx.buffer_from(weight_bytes)?;
    let buf_x = ctx.buffer_from(x)?;
    let buf_y = ctx.buffer_uninit(n_out * std::mem::size_of::<f32>())?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }
    let args = Args {
        n_in: n_in as u32,
        n_out: n_out as u32,
    };
    let buf_args = ctx.buffer_from(&[args])?;

    // Lifted-from-llama.cpp kernel: 2 simdgroups per threadgroup, 2 rows
    // per simdgroup → 4 rows per threadgroup.
    const NR0: usize = 2;
    const NSG: usize = 2;
    let rows_per_tg = NR0 * NSG;
    let n_tg = n_out.div_ceil(rows_per_tg);

    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd_buf.computeCommandEncoder().expect("compute encoder");
    enc.setComputePipelineState(&pso);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*buf_args), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*buf_w), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*buf_x), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*buf_y), 0, 3);
    }
    let grid = MTLSize {
        width: n_tg,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: NSG * 32,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd_buf.commit();
    unsafe { cmd_buf.waitUntilCompleted() };

    let mut out = vec![0.0f32; n_out];
    unsafe {
        let src = buf_y.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n_out);
    }
    Ok(out)
}

/// Dispatch the F32 mat-vec kernel.
///
/// `y[o] = sum_i weight[o*n_in + i] * x[i]` for `o ∈ [0, n_out)`.
///
/// CPU oracle: [`crate::forward::mat_vec_pub`].
pub fn mat_vec_f32(
    ctx: &MetalContext,
    weight: &[f32],
    x: &[f32],
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, MetalError> {
    debug_assert_eq!(weight.len(), n_in * n_out);
    debug_assert_eq!(x.len(), n_in);

    let pso = ctx.pipeline("kernel_mat_vec_f32_f32")?;
    let buf_w = ctx.buffer_from(weight)?;
    let buf_x = ctx.buffer_from(x)?;
    let buf_y = ctx.buffer_uninit(n_out * std::mem::size_of::<f32>())?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }
    let args = Args {
        n_in: n_in as u32,
        n_out: n_out as u32,
    };
    let buf_args = ctx.buffer_from(&[args])?;

    const ROWS_PER_TG: usize = 4;
    let n_tg = n_out.div_ceil(ROWS_PER_TG);

    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd_buf.computeCommandEncoder().expect("compute encoder");
    enc.setComputePipelineState(&pso);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&*buf_args), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*buf_w), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*buf_x), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*buf_y), 0, 3);
    }
    let grid = MTLSize {
        width: n_tg,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: ROWS_PER_TG * 32,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd_buf.commit();
    unsafe { cmd_buf.waitUntilCompleted() };

    let mut out = vec![0.0f32; n_out];
    unsafe {
        let src = buf_y.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n_out);
    }
    Ok(out)
}

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

    /// Validate the F32 mat-vec kernel against the CPU oracle on the
    /// shapes that matter for the 0.8B and 27B forward pass:
    ///   - 1024 x 248320 (lm_head 0.8B)
    ///   - 1024 x 6144 (GDN attn_qkv 0.8B)
    ///   - 5120 x 17408 (FFN gate/up 27B)
    ///   - 5120 x 5120 (attn_q 27B half — output dim 2x)
    #[test]
    fn mat_vec_f32_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(n_in, n_out) in &[
            (1024usize, 248_320usize), // 0.8B lm_head
            (1024, 6144),              // 0.8B GDN attn_qkv
            (5120, 17408),             // 27B FFN gate/up
            (5120, 5120),              // 27B attn_q (half of 2x)
        ] {
            // Deterministic synthetic inputs.
            let w: Vec<f32> = (0..n_in * n_out)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-3)
                .collect();
            let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();

            let cpu_t = std::time::Instant::now();
            let cpu = crate::forward::mat_vec_pub(&w, n_in, n_out, &x);
            let cpu_ms = cpu_t.elapsed().as_secs_f64() * 1e3;

            let gpu_t = std::time::Instant::now();
            let gpu = mat_vec_f32(&ctx, &w, &x, n_in, n_out).expect("metal mat_vec");
            let gpu_ms = gpu_t.elapsed().as_secs_f64() * 1e3;

            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            for (a, b) in gpu.iter().zip(cpu.iter()) {
                let d = (a - b).abs();
                max_abs = max_abs.max(d);
                let r = d / b.abs().max(1e-6);
                max_rel = max_rel.max(r);
            }
            let bytes = (n_in * n_out + n_in + n_out) * 4;
            let bw = bytes as f64 / (gpu_ms * 1e-3) / 1e9;
            eprintln!(
                "[mat_vec n_in={n_in} n_out={n_out}] cpu={cpu_ms:.1}ms gpu={gpu_ms:.1}ms ({bw:.0} GB/s) max|Δ|={max_abs:.2e} rel={max_rel:.2e}"
            );
            assert!(max_abs < 1e-3, "mat_vec drift {max_abs} exceeds 1e-3");
        }
    }

    fn bench_q4k_one(
        ctx: &MetalContext,
        g: &crate::gguf::GgufFile,
        q4k: &crate::tensor::TensorDesc,
    ) {
        let n_in = q4k.shape[0] as usize;
        let n_out = q4k.shape[1] as usize;
        let buf_w = ctx.buffer_from(g.slice(q4k)).expect("w");
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 1e-3).sin()).collect();
        let buf_x = ctx.buffer_from(&x).expect("x");
        let buf_y = ctx
            .buffer_uninit(n_out * std::mem::size_of::<f32>())
            .expect("y");

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Args {
            n_in: u32,
            n_out: u32,
        }
        let buf_args = ctx
            .buffer_from(&[Args {
                n_in: n_in as u32,
                n_out: n_out as u32,
            }])
            .expect("args");

        for _ in 0..3 {
            mat_vec_q4_k_f32_bufs(ctx, &buf_args, &buf_w, &buf_x, &buf_y, n_out).expect("dispatch");
        }
        const ITERS: usize = 100;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            mat_vec_q4_k_f32_bufs(ctx, &buf_args, &buf_w, &buf_x, &buf_y, n_out).expect("dispatch");
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_iter_ms = (elapsed / ITERS as f64) * 1e3;
        let bw = q4k.n_bytes as f64 / (elapsed / ITERS as f64) / 1e9;
        eprintln!(
            "    {ITERS} iters: {per_iter_ms:.3} ms/iter → {bw:.0} GB/s ({:.0}% of 546 GB/s)",
            bw / 5.46
        );
    }

    /// Performance benchmark for Q4_K mat_vec on a real 27B tensor with
    /// persistent buffers. Reports kernel-only time after warmup, isolating
    /// kernel cost from buffer-setup overhead. This is the honest number
    /// to use for "are we beating llama.cpp."
    #[test]
    #[ignore]
    fn mat_vec_q4_k_perf() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");

        // Sweep across the Q4_K shapes that actually run during 27B decode.
        let candidates: Vec<&str> = vec![
            "blk.0.attn_gate.weight", //  5120 x 6144   — GDN gate
            "blk.0.ssm_out.weight",   //  6144 x 5120   — GDN out
            "blk.0.ffn_gate.weight",  //  5120 x 17408  — FFN gate (the heavy one)
            "blk.0.ffn_up.weight",    //  5120 x 17408  — FFN up
            "token_embd.weight",      //  5120 x 248320 — embedding
        ];
        for name in &candidates {
            if let Some(t) = g.find(name) {
                eprintln!(
                    "--- {name} ({:?} {:?}, {} MB) ---",
                    t.dtype,
                    t.shape,
                    t.n_bytes / (1024 * 1024)
                );
                if t.dtype == crate::tensor::GgmlType::Q4_K {
                    bench_q4k_one(&ctx, &g, t);
                }
            }
        }
        // Then keep going with the original biggest-tensor measurement.
        let q4k = g
            .tensors
            .iter()
            .filter(|t| {
                t.dtype == crate::tensor::GgmlType::Q4_K && t.shape.len() == 2 && t.shape[0] >= 5120
            })
            .max_by_key(|t| t.n_bytes)
            .expect("no Q4_K tensor");
        let n_in = q4k.shape[0] as usize;
        let n_out = q4k.shape[1] as usize;
        eprintln!(
            "[perf-q4_k] {} shape=[{n_in}, {n_out}] {} bytes",
            q4k.name, q4k.n_bytes
        );

        // Persistent buffers (mimics what real inference does).
        let buf_w = ctx.buffer_from(g.slice(q4k)).expect("w");
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 1e-3).sin()).collect();
        let buf_x = ctx.buffer_from(&x).expect("x");
        let buf_y = ctx
            .buffer_uninit(n_out * std::mem::size_of::<f32>())
            .expect("y");

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Args {
            n_in: u32,
            n_out: u32,
        }
        let args = Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
        };
        let buf_args = ctx.buffer_from(&[args]).expect("args");

        // Warmup.
        for _ in 0..3 {
            mat_vec_q4_k_f32_bufs(&ctx, &buf_args, &buf_w, &buf_x, &buf_y, n_out)
                .expect("dispatch");
        }

        // Timed loop.
        const ITERS: usize = 100;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            mat_vec_q4_k_f32_bufs(&ctx, &buf_args, &buf_w, &buf_x, &buf_y, n_out)
                .expect("dispatch");
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_iter_ms = (elapsed / ITERS as f64) * 1e3;
        let bw = q4k.n_bytes as f64 / (elapsed / ITERS as f64) / 1e9;
        eprintln!(
            "[perf-q4_k] {ITERS} iters: {:.2} ms each → {bw:.0} GB/s ({:.0}% of 546 GB/s peak)",
            per_iter_ms,
            bw / 5.46
        );
    }

    #[test]
    fn mat_vec_q6_k_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let q6k = g
            .tensors
            .iter()
            .find(|t| {
                t.name.starts_with("blk.0.")
                    && t.dtype == crate::tensor::GgmlType::Q6_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
            })
            .expect("no Q6_K tensor");
        let n_in = q6k.shape[0] as usize;
        let n_out = q6k.shape[1] as usize;
        eprintln!(
            "[q6_k-test] {} shape=[{n_in}, {n_out}] {} bytes",
            q6k.name, q6k.n_bytes
        );

        let weight_f32 = crate::codec::dequant_to_f32(q6k, g.slice(q6k)).expect("dequant");
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let gpu = mat_vec_q6_k_f32(&ctx, g.slice(q6k), &x, n_in, n_out).expect("metal q6k");

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (a, b) in gpu.iter().zip(cpu.iter()) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            let r = d / b.abs().max(1e-6);
            max_rel = max_rel.max(r);
        }
        eprintln!("[q6_k] max|Δ|={max_abs:.2e} rel={max_rel:.2e}");
        assert!(max_abs < 1e-2, "q6_k drift {max_abs}");
    }

    /// Validate the Q4_K mat-vec kernel against the CPU dequant + mat_vec
    /// path. Uses a real tensor from the 27B Q4_K_M GGUF if available;
    /// skips otherwise.
    #[test]
    fn mat_vec_q4_k_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");

        // Find a Q4_K tensor we can use (FFN gate or up are Q4_K in Q4_K_M).
        let q4k = g
            .tensors
            .iter()
            .find(|t| {
                t.name.starts_with("blk.0.")
                    && t.dtype == crate::tensor::GgmlType::Q4_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
            })
            .expect("no suitable Q4_K tensor in 27B");
        let n_in = q4k.shape[0] as usize;
        let n_out = q4k.shape[1] as usize;
        eprintln!(
            "[q4_k-test] tensor {} shape=[{n_in}, {n_out}] bytes={}",
            q4k.name, q4k.n_bytes
        );

        // CPU dequant via the codec seam.
        let cpu_dequant_t = std::time::Instant::now();
        let weight_f32 = crate::codec::dequant_to_f32(q4k, g.slice(q4k)).expect("dequant");
        let cpu_dequant_ms = cpu_dequant_t.elapsed().as_secs_f64() * 1e3;
        eprintln!("[q4_k-test] cpu dequant: {cpu_dequant_ms:.1}ms");

        // Synthetic input.
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();

        // CPU oracle.
        let cpu_t = std::time::Instant::now();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let cpu_ms = cpu_t.elapsed().as_secs_f64() * 1e3;

        // Metal Q4_K mat_vec.
        let weight_bytes = g.slice(q4k);
        let gpu_t = std::time::Instant::now();
        let gpu =
            mat_vec_q4_k_f32(&ctx, weight_bytes, &x, n_in, n_out).expect("metal mat_vec_q4_k");
        let gpu_ms = gpu_t.elapsed().as_secs_f64() * 1e3;

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (a, b) in gpu.iter().zip(cpu.iter()) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            let r = d / b.abs().max(1e-6);
            max_rel = max_rel.max(r);
        }
        let bw = q4k.n_bytes as f64 / (gpu_ms * 1e-3) / 1e9;
        eprintln!(
            "[q4_k] cpu={cpu_ms:.1}ms gpu={gpu_ms:.1}ms ({bw:.0} GB/s) max|Δ|={max_abs:.2e} rel={max_rel:.2e}"
        );
        assert!(max_abs < 1e-2, "mat_vec_q4_k drift {max_abs}");
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
