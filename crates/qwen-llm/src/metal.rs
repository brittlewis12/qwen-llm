//! Host-side Metal lifecycle and kernel-encoding API.
//!
//! ## v1 design (post-codex review fc6f43e)
//!
//! * One [`MetalContext`] per process: holds `MTLDevice`, `MTLCommandQueue`,
//!   the embedded `kernels.metallib` loaded as a `MTLLibrary`, and a
//!   name → pipeline-state cache (`Mutex<HashMap>`; contention is irrelevant
//!   since pipeline creation only happens at first dispatch).
//!
//! * Persistent buffers via [`MetalTensor`]. One `MTLBuffer` per weight tensor;
//!   loaded once at model setup, reused for every forward step. Allocation
//!   uses `StorageModeShared` so reads/writes go straight to unified memory
//!   with no host↔device copies.
//!
//! * **Encode-only kernel API**: every `encode_*` function takes a
//!   `&KernelEncoder` (a thin wrapper around `MTLComputeCommandEncoder`) plus
//!   typed `&MetalTensor` views. Kernels never commit the command buffer,
//!   never wait, never read back to host. The forward pass owns the command
//!   buffer lifetime: encode N kernels into one buffer, commit once, wait
//!   once at logits readback.
//!
//! * For tests we keep the convenient one-shot `*_readback_for_test` API
//!   that allocates buffers, dispatches, waits, and copies out — but those
//!   are clearly named so they don't accidentally end up on the inference
//!   hot path.
//!
//! * v2: ship a precompiled `MTLBinaryArchive` next to the binary; switch to
//!   `MTL4CommandBuffer` (already exposed in `objc2-metal` 0.3.2) for lower
//!   per-step encoding overhead; design ICB.

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

use crate::tensor::{GgmlType, TensorDesc};

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
    #[error("bad shape for kernel {kernel}: {detail}")]
    BadShape {
        kernel: &'static str,
        detail: String,
    },
}

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Owned Metal buffer wrapper, the public type for buffer-style values.
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

// ===========================================================================
// MetalContext
// ===========================================================================

pub struct MetalContext {
    pub device: Device,
    pub queue: Queue,
    pub library: Library,
    pso_cache: Arc<Mutex<HashMap<String, Pipeline>>>,
}

// SAFETY: `Retained<ProtocolObject<dyn MTL*>>` are thread-safe per Apple's
// Metal docs (the protocol objects are themselves backed by thread-safe
// Objective-C classes; method dispatch is internally synchronized).
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
        let pso = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e: Retained<NSError>| {
                MetalError::Pipeline(name.to_string(), e.localizedDescription().to_string())
            })?;
        self.pso_cache.lock().insert(name.to_string(), pso.clone());
        Ok(pso)
    }

    /// Allocate a buffer populated from a `bytemuck::Pod` slice.
    /// Uses `StorageModeShared` (unified memory).
    pub fn buffer_from<T: bytemuck::Pod>(&self, data: &[T]) -> Result<Buffer, MetalError> {
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
        // SAFETY: `bytes` is valid for `n` bytes for the duration of the call;
        // Metal copies the data into the new buffer.
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

    /// Allocate an uninitialized output buffer of `n_bytes`.
    pub fn buffer_uninit(&self, n_bytes: usize) -> Result<Buffer, MetalError> {
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
    // Strategy: write the embedded `.metallib` to a temp file and load via
    // `newLibraryWithURL:`. The library bytes are small (≲100 KB even with
    // all kernels) so I/O is negligible vs pipeline-state compilation.
    // Alternative `newLibraryWithData:` wants `dispatch_data_t`, which
    // objc2-metal 0.3 doesn't expose ergonomically yet.
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
    let lib = device
        .newLibraryWithURL_error(&url)
        .map_err(|e: Retained<NSError>| {
            MetalError::LoadLibrary(e.localizedDescription().to_string())
        })?;
    let _ = std::fs::remove_file(&tmp);
    Ok(lib)
}

// ===========================================================================
// MetalTensor — typed buffer view
// ===========================================================================

/// A typed, shape-aware view into an `MTLBuffer`. The buffer is owned via
/// `Retained` (cloned into multiple `MetalTensor`s if you want sub-views;
/// they share the same underlying storage).
///
/// `dtype` is the on-disk ggml type for weight tensors (Q4_K, Q6_K, F32,
/// etc.); for activation/scratch buffers it's typically `F32`.
///
/// `offset` is in bytes from the buffer base. v1 always sets `offset=0`
/// (one buffer per tensor); v2 may pack multiple tensors into one arena
/// buffer with non-zero offsets.
#[derive(Clone)]
pub struct MetalTensor {
    pub buffer: Buffer,
    pub offset: u64,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
}

impl MetalTensor {
    /// Total element count.
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Total byte length (consults the dtype's per-block layout via
    /// `llama_cpp_sys_2`).
    pub fn n_bytes(&self) -> u64 {
        let raw = self.dtype as i32 as u32;
        // SAFETY: ggml_row_size is a pure size calculator with no allocation.
        unsafe { llama_cpp_sys_2::ggml_row_size(raw, self.n_elements() as i64) as u64 }
    }

    /// Build a tensor from raw bytes (e.g. the GGUF mmap slice for a
    /// weight tensor). Copies into a fresh `MTLBuffer` with shared storage.
    pub fn from_bytes(
        ctx: &MetalContext,
        bytes: &[u8],
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> Result<Self, MetalError> {
        let buffer = ctx.buffer_from(bytes)?;
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype,
        })
    }

    /// Build from a `[TensorDesc]` + the GGUF mmap. This is the production
    /// path for loading model weights.
    pub fn from_gguf_tensor(
        ctx: &MetalContext,
        desc: &TensorDesc,
        bytes: &[u8],
    ) -> Result<Self, MetalError> {
        Self::from_bytes(ctx, bytes, desc.shape.clone(), desc.dtype)
    }

    /// Allocate an F32 activation/scratch tensor of the given shape,
    /// uninitialized.
    pub fn zeros_f32(ctx: &MetalContext, shape: Vec<u64>) -> Result<Self, MetalError> {
        let n: u64 = shape.iter().product();
        let buffer = ctx.buffer_uninit(n as usize * std::mem::size_of::<f32>())?;
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype: GgmlType::F32,
        })
    }
}

// ===========================================================================
// KernelEncoder — caller-owned compute encoder
// ===========================================================================

/// Caller-owned wrapper around `MTLComputeCommandEncoder`. Produced by
/// [`KernelEncoder::begin`] from a `MTLCommandBuffer`. All `encode_*`
/// kernels accept this and mutate it; the caller is responsible for
/// `end()` (which calls `endEncoding`), `commit()` on the parent command
/// buffer, and `wait()` if a result needs to be read back.
///
/// This is the *one* place in the engine where Metal lifecycle is exposed
/// to client code. The forward-pass driver owns the command buffer; every
/// kernel just appends dispatches.
pub struct KernelEncoder {
    pub raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl KernelEncoder {
    pub fn begin(cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        let raw = cmd.computeCommandEncoder().expect("compute encoder");
        Self { raw }
    }

    pub fn end(self) {
        self.raw.endEncoding();
    }

    /// Bind a buffer at slot `index`.
    pub fn set_buffer(&self, index: usize, buf: &Buffer, offset: u64) {
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(Some(buf.as_ref()), offset as usize, index);
        }
    }

    /// Bind a `MetalTensor` at slot `index`. Convenience wrapper over
    /// `set_buffer` that respects the tensor's `offset`.
    pub fn set_tensor(&self, index: usize, tensor: &MetalTensor) {
        self.set_buffer(index, &tensor.buffer, tensor.offset);
    }

    /// Bind a small Pod argument struct inline (≤4 KB). Uses Metal's
    /// `setBytes:length:atIndex:` which avoids creating a buffer object
    /// for tiny per-dispatch scalars.
    pub fn set_bytes<T: bytemuck::Pod>(&self, index: usize, value: &T) {
        let bytes = bytemuck::bytes_of(value);
        let ptr = std::ptr::NonNull::new(bytes.as_ptr() as *mut std::ffi::c_void)
            .expect("non-null bytemuck pointer");
        unsafe {
            self.raw.setBytes_length_atIndex(ptr, bytes.len(), index);
        }
    }

    /// Bind threadgroup memory of `n_bytes` at slot `index`.
    pub fn set_threadgroup_memory(&self, index: usize, n_bytes: usize) {
        unsafe {
            self.raw.setThreadgroupMemoryLength_atIndex(n_bytes, index);
        }
    }

    pub fn set_pipeline(&self, pso: &Pipeline) {
        self.raw.setComputePipelineState(pso);
    }

    pub fn dispatch(&self, grid: MTLSize, threads: MTLSize) {
        self.raw
            .dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
    }
}

// ===========================================================================
// Encode-only kernel API (production)
// ===========================================================================
//
// Each `encode_*` function appends a single kernel dispatch onto the
// caller-provided `KernelEncoder`. No buffers are created here; no
// command buffer is committed; no readback is performed. The caller
// orchestrates the command buffer lifetime.

/// RMSNorm-with-weight: `y[i] = (x[i] / sqrt(mean(x²) + eps)) * weight[i]`.
///
/// Operates on a single row of length `n_dim`. One threadgroup per dispatch;
/// up to 1024 threads per threadgroup, internally reduced via simdgroup_sum.
///
/// CPU oracle: [`crate::forward::rms_norm_pub`].
pub fn encode_rms_norm_mul_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    weight: &MetalTensor,
    y: &MetalTensor,
    eps: f32,
) -> Result<(), MetalError> {
    let n_dim = x.n_elements() as usize;
    if y.n_elements() as usize != n_dim {
        return Err(MetalError::BadShape {
            kernel: "rms_norm",
            detail: format!("y.n_elements={} != x.n_elements={n_dim}", y.n_elements()),
        });
    }
    if weight.n_elements() as usize != n_dim {
        return Err(MetalError::BadShape {
            kernel: "rms_norm",
            detail: format!(
                "weight.n_elements={} != x.n_elements={n_dim}",
                weight.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_rms_norm_mul_f32")?;
    enc.set_pipeline(&pso);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_dim: u32,
        eps: f32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_dim: n_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// F32 mat-vec: `y[o] = Σ_i W[o, i] * x[i]`, GGUF stride convention.
/// `W` has shape `[n_in, n_out]` (ne[0]=n_in fastest); `x` is `[n_in]`,
/// `y` is `[n_out]`.
///
/// CPU oracle: [`crate::forward::mat_vec_pub`].
pub fn encode_mat_vec_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if x.n_elements() as usize != n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_f32",
            detail: format!("x.n_elements={} != n_in={n_in}", x.n_elements()),
        });
    }
    if y.n_elements() as usize != n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_f32",
            detail: format!("y.n_elements={} != n_out={n_out}", y.n_elements()),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_f32_f32")?;
    enc.set_pipeline(&pso);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);

    const ROWS_PER_TG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(ROWS_PER_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: ROWS_PER_TG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Q4_K mat-vec on raw `block_q4_K` bytes. Same shape semantics as
/// [`encode_mat_vec_f32`]; `weight.dtype` must be `Q4_K`.
pub fn encode_mat_vec_q4_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k",
            detail: format!("n_in={n_in} not divisible by 256 (Q4_K super-block)"),
        });
    }
    if weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k",
            detail: format!("weight.dtype = {:?}, expected Q4_K", weight.dtype),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_q4_K_f32")?;
    enc.set_pipeline(&pso);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

// ----- elementwise + small ops -----

/// Per-element kernel arg used by silu/sigmoid/softplus/add/mul/silu_mul.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NArgs {
    n: u32,
}

/// Per-element kernel arg shape for L2 norm.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct L2NormArgs {
    n_dim: u32,
    eps: f32,
}

/// Generic 1-input → 1-output elementwise dispatcher (silu, sigmoid,
/// softplus). All share the (NArgs, x, y) bind pattern.
fn encode_elementwise_1in_1out(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kernel: &str,
    x: &MetalTensor,
    y: &MetalTensor,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if y.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "elementwise_1in_1out",
            detail: format!("y.n={} != x.n={n}", y.n_elements()),
        });
    }
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_silu_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_1in_1out(ctx, enc, "kernel_silu_f32", x, y)
}

pub fn encode_sigmoid_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_1in_1out(ctx, enc, "kernel_sigmoid_f32", x, y)
}

pub fn encode_softplus_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_1in_1out(ctx, enc, "kernel_softplus_f32", x, y)
}

/// Generic 2-input → 1-output elementwise (add, mul, silu_mul). All share
/// the (NArgs, a, b, out) bind pattern.
fn encode_elementwise_2in_1out(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kernel: &str,
    a: &MetalTensor,
    b: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    let n = a.n_elements() as usize;
    if b.n_elements() as usize != n || out.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "elementwise_2in_1out",
            detail: format!("lengths a={} b={} out={n}", a.n_elements(), b.n_elements()),
        });
    }
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, a);
    enc.set_tensor(2, b);
    enc.set_tensor(3, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_add_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,
    b: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_2in_1out(ctx, enc, "kernel_add_f32", a, b, out)
}

pub fn encode_mul_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,
    b: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_2in_1out(ctx, enc, "kernel_mul_f32", a, b, out)
}

/// SwiGLU FFN inner: out = silu(gate) * up. Fuses two ops + saves a
/// scratch buffer on the FFN path. Used as
/// `down(silu_mul(gate(x), up(x)))`.
pub fn encode_silu_mul_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_2in_1out(ctx, enc, "kernel_silu_mul_f32", gate, up, out)
}

/// In-place residual add: x += y. Used after each transformer block's
/// mixer and FFN to add the residual stream back. Saves a scratch buffer
/// vs out-of-place add.
pub fn encode_add_inplace_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if y.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "add_inplace",
            detail: format!("y.n={} != x.n={n}", y.n_elements()),
        });
    }
    let pso = ctx.pipeline("kernel_add_inplace_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// In-place softmax over the (only) dimension of `x`. Used for attention
/// scores. Input is mutated.
pub fn encode_softmax_inplace_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    let pso = ctx.pipeline("kernel_softmax_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, x);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// L2 norm: y = x / max(||x||, eps). Per-vector. ggml semantics (NOT
/// `1/sqrt(sum+eps)` — that's RMSNorm).
pub fn encode_l2_norm_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
    eps: f32,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if y.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "l2_norm",
            detail: format!("y.n={} != x.n={n}", y.n_elements()),
        });
    }
    let pso = ctx.pipeline("kernel_l2_norm_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &L2NormArgs {
            n_dim: n as u32,
            eps,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Embedding lookup: `y[r * n_cols + i] = embed[ids[r] * n_cols + i]`.
/// `embed` is `[vocab, n_cols]`-shaped F32; `ids` is `[n_rows]` i32.
/// Decode uses `n_rows = 1`; prefill uses `n_rows = batch`.
pub fn encode_get_rows_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    embed: &MetalTensor,
    ids: &MetalTensor,
    y: &MetalTensor,
    n_rows: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    if y.n_elements() as usize != n_rows * n_cols {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                "y.n={} != n_rows*n_cols={}",
                y.n_elements(),
                n_rows * n_cols
            ),
        });
    }
    if ids.n_elements() as usize != n_rows {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!("ids.n={} != n_rows={n_rows}", ids.n_elements()),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GetRowsArgs {
        n_rows: u32,
        n_cols: u32,
    }
    let pso = ctx.pipeline("kernel_get_rows_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &GetRowsArgs {
            n_rows: n_rows as u32,
            n_cols: n_cols as u32,
        },
    );
    enc.set_tensor(1, embed);
    enc.set_tensor(2, ids);
    enc.set_tensor(3, y);

    // 2D grid: (n_cols, n_rows).
    enc.dispatch(
        MTLSize {
            width: n_cols.div_ceil(32),
            height: n_rows,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Q6_K mat-vec, same API shape as [`encode_mat_vec_q4_k_f32`].
pub fn encode_mat_vec_q6_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k",
            detail: format!("weight.dtype = {:?}, expected Q6_K", weight.dtype),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_q6_K_f32")?;
    enc.set_pipeline(&pso);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

// ===========================================================================
// Test helpers (one-shot wrappers — NOT for the inference hot path)
// ===========================================================================
//
// These wrappers create buffers, encode one dispatch, commit, wait, and
// copy the result to a `Vec<f32>`. They exist so unit tests can compare
// kernel output against the CPU oracle without orchestrating the full
// command-buffer lifecycle. **Don't call these from forward-pass code.**

fn one_shot<F>(ctx: &MetalContext, encode: F) -> Result<(), MetalError>
where
    F: FnOnce(&KernelEncoder) -> Result<(), MetalError>,
{
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    encode(&enc)?;
    enc.end();
    cmd_buf.commit();
    unsafe { cmd_buf.waitUntilCompleted() };
    Ok(())
}

fn read_back_f32(buf: &Buffer, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    unsafe {
        let src = buf.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }
    out
}

/// One-shot RMSNorm for tests. Production code uses [`encode_rms_norm_mul_f32`].
pub fn rms_norm_mul_f32_readback_for_test(
    ctx: &MetalContext,
    x: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, MetalError> {
    let n = x.len();
    let x_t = MetalTensor::from_bytes(ctx, bytemuck::cast_slice(x), vec![n as u64], GgmlType::F32)?;
    let w_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(weight),
        vec![n as u64],
        GgmlType::F32,
    )?;
    let y_t = MetalTensor::zeros_f32(ctx, vec![n as u64])?;
    one_shot(ctx, |enc| {
        encode_rms_norm_mul_f32(ctx, enc, &x_t, &w_t, &y_t, eps)
    })?;
    Ok(read_back_f32(&y_t.buffer, n))
}

/// One-shot F32 mat-vec for tests.
pub fn mat_vec_f32_readback_for_test(
    ctx: &MetalContext,
    weight: &[f32],
    x: &[f32],
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, MetalError> {
    let w_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(weight),
        vec![n_in as u64, n_out as u64],
        GgmlType::F32,
    )?;
    let x_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(x),
        vec![n_in as u64],
        GgmlType::F32,
    )?;
    let y_t = MetalTensor::zeros_f32(ctx, vec![n_out as u64])?;
    one_shot(ctx, |enc| {
        encode_mat_vec_f32(ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
    })?;
    Ok(read_back_f32(&y_t.buffer, n_out))
}

/// One-shot Q4_K mat-vec for tests.
pub fn mat_vec_q4_k_f32_readback_for_test(
    ctx: &MetalContext,
    weight_bytes: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, MetalError> {
    let w_t = MetalTensor::from_bytes(
        ctx,
        weight_bytes,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q4_K,
    )?;
    let x_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(x),
        vec![n_in as u64],
        GgmlType::F32,
    )?;
    let y_t = MetalTensor::zeros_f32(ctx, vec![n_out as u64])?;
    one_shot(ctx, |enc| {
        encode_mat_vec_q4_k_f32(ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
    })?;
    Ok(read_back_f32(&y_t.buffer, n_out))
}

/// One-shot Q6_K mat-vec for tests.
pub fn mat_vec_q6_k_f32_readback_for_test(
    ctx: &MetalContext,
    weight_bytes: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, MetalError> {
    let w_t = MetalTensor::from_bytes(
        ctx,
        weight_bytes,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q6_K,
    )?;
    let x_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(x),
        vec![n_in as u64],
        GgmlType::F32,
    )?;
    let y_t = MetalTensor::zeros_f32(ctx, vec![n_out as u64])?;
    one_shot(ctx, |enc| {
        encode_mat_vec_q6_k_f32(ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
    })?;
    Ok(read_back_f32(&y_t.buffer, n_out))
}

// ===========================================================================
// Bench helpers — chain N dispatches into one command buffer
// ===========================================================================

/// Chain `n_dispatches` Q4_K mat-vecs into one command buffer, one wait at
/// the end. Used for kernel benchmarks; the production forward pass uses
/// `encode_*` directly with its own command-buffer orchestration.
pub fn bench_q4_k_chained(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    for _ in 0..n_dispatches {
        encode_mat_vec_q4_k_f32(ctx, &enc, weight, x, y, n_in, n_out)?;
    }
    enc.end();
    cmd_buf.commit();
    unsafe { cmd_buf.waitUntilCompleted() };
    Ok(())
}

/// Same but for Q6_K.
pub fn bench_q6_k_chained(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    for _ in 0..n_dispatches {
        encode_mat_vec_q6_k_f32(ctx, &enc, weight, x, y, n_in, n_out)?;
    }
    enc.end();
    cmd_buf.commit();
    unsafe { cmd_buf.waitUntilCompleted() };
    Ok(())
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

    #[test]
    fn rms_norm_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &n in &[1024usize, 5120, 17408] {
            let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
            let w: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
            let eps = 1e-6;

            let cpu = crate::forward::rms_norm_pub(&x, &w, eps);
            let gpu =
                rms_norm_mul_f32_readback_for_test(&ctx, &x, &w, eps).expect("metal rms_norm");

            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[rms_norm n={n}] max|Δ|={max_abs:.2e}");
            assert!(max_abs < 1e-4, "rms_norm n={n}: max|Δ|={max_abs}");
        }
    }

    #[test]
    fn mat_vec_f32_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(n_in, n_out) in &[
            (1024usize, 248_320usize),
            (1024, 6144),
            (5120, 17408),
            (5120, 5120),
        ] {
            let w: Vec<f32> = (0..n_in * n_out)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-3)
                .collect();
            let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
            let cpu = crate::forward::mat_vec_pub(&w, n_in, n_out, &x);
            let gpu = mat_vec_f32_readback_for_test(&ctx, &w, &x, n_in, n_out).expect("gpu");
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[mat_vec n_in={n_in} n_out={n_out}] max|Δ|={max_abs:.2e}");
            assert!(max_abs < 1e-3);
        }
    }

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
        let q4k = g
            .tensors
            .iter()
            .find(|t| {
                t.name.starts_with("blk.0.")
                    && t.dtype == GgmlType::Q4_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
            })
            .expect("no Q4_K tensor");
        let n_in = q4k.shape[0] as usize;
        let n_out = q4k.shape[1] as usize;
        eprintln!("[q4_k-test] {} shape=[{n_in}, {n_out}]", q4k.name);

        let weight_f32 = crate::codec::dequant_to_f32(q4k, g.slice(q4k)).expect("dequant");
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let gpu =
            mat_vec_q4_k_f32_readback_for_test(&ctx, g.slice(q4k), &x, n_in, n_out).expect("gpu");
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q4_k] max|Δ|={max_abs:.2e}");
        assert!(max_abs < 1e-2);
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
                    && t.dtype == GgmlType::Q6_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
            })
            .expect("no Q6_K tensor");
        let n_in = q6k.shape[0] as usize;
        let n_out = q6k.shape[1] as usize;
        eprintln!("[q6_k-test] {} shape=[{n_in}, {n_out}]", q6k.name);
        let weight_f32 = crate::codec::dequant_to_f32(q6k, g.slice(q6k)).expect("dequant");
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let gpu =
            mat_vec_q6_k_f32_readback_for_test(&ctx, g.slice(q6k), &x, n_in, n_out).expect("gpu");
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q6_k] max|Δ|={max_abs:.2e}");
        assert!(max_abs < 1e-2);
    }

    /// Helper: take an `encode_*` closure that produces a single F32
    /// output buffer of length `n_out`, run it one-shot, and return the
    /// readback. Common shape across the elementwise tests.
    fn one_shot_f32_out<F>(ctx: &MetalContext, n_out: usize, encode: F) -> Vec<f32>
    where
        F: FnOnce(&KernelEncoder, &MetalTensor) -> Result<(), MetalError>,
    {
        let y_t = MetalTensor::zeros_f32(ctx, vec![n_out as u64]).unwrap();
        one_shot(ctx, |enc| encode(enc, &y_t)).unwrap();
        read_back_f32(&y_t.buffer, n_out)
    }

    #[test]
    fn elementwise_silu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let x: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let n = x.len();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let gpu = one_shot_f32_out(&ctx, n, |enc, y| encode_silu_f32(&ctx, enc, &x_t, y));
        for (i, &v) in x.iter().enumerate() {
            let expected = v / (1.0 + (-v).exp());
            assert!(
                (gpu[i] - expected).abs() < 1e-5,
                "silu[{i}] {v} -> {} vs {}",
                gpu[i],
                expected
            );
        }
    }

    #[test]
    fn elementwise_sigmoid_softplus() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let x: Vec<f32> = (-30..30).map(|i| i as f32 * 0.5).collect();
        let n = x.len();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();

        let sig = one_shot_f32_out(&ctx, n, |enc, y| encode_sigmoid_f32(&ctx, enc, &x_t, y));
        let sp = one_shot_f32_out(&ctx, n, |enc, y| encode_softplus_f32(&ctx, enc, &x_t, y));

        for (i, &v) in x.iter().enumerate() {
            let exp_sig = 1.0 / (1.0 + (-v).exp());
            assert!((sig[i] - exp_sig).abs() < 1e-5);
            let exp_sp = if v > 20.0 {
                v
            } else if v < -20.0 {
                v.exp()
            } else {
                (1.0 + v.exp()).ln()
            };
            assert!((sp[i] - exp_sp).abs() < 1e-5);
        }
    }

    #[test]
    fn elementwise_add_mul_silu_mul() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let n = 17408usize; // FFN dim — exercise the realistic shape
        let a: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
        let a_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&a),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let b_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&b),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();

        let added = one_shot_f32_out(&ctx, n, |enc, y| encode_add_f32(&ctx, enc, &a_t, &b_t, y));
        for i in 0..n {
            assert!((added[i] - (a[i] + b[i])).abs() < 1e-5);
        }
        let muled = one_shot_f32_out(&ctx, n, |enc, y| encode_mul_f32(&ctx, enc, &a_t, &b_t, y));
        for i in 0..n {
            assert!((muled[i] - (a[i] * b[i])).abs() < 1e-5);
        }
        let silumul = one_shot_f32_out(&ctx, n, |enc, y| {
            encode_silu_mul_f32(&ctx, enc, &a_t, &b_t, y)
        });
        for i in 0..n {
            let silu_a = a[i] / (1.0 + (-a[i]).exp());
            assert!(
                (silumul[i] - silu_a * b[i]).abs() < 1e-5,
                "silu_mul[{i}] = {} vs {}",
                silumul[i],
                silu_a * b[i]
            );
        }
    }

    #[test]
    fn elementwise_add_inplace() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let n = 5120usize;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-3).collect();
        let b: Vec<f32> = (0..n).map(|i| -(i as f32) * 2e-3).collect();
        let a_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&a),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let b_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&b),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        one_shot(&ctx, |enc| encode_add_inplace_f32(&ctx, enc, &a_t, &b_t)).unwrap();
        let result = read_back_f32(&a_t.buffer, n);
        for i in 0..n {
            assert!((result[i] - (a[i] + b[i])).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        // Cover small (one warp) to large (multi-warp reduce) shapes.
        for &n in &[16usize, 256, 4096, 32768] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| encode_softmax_inplace_f32(&ctx, enc, &x_t)).unwrap();
            let gpu = read_back_f32(&x_t.buffer, n);

            // CPU reference.
            let mut cpu = x.clone();
            let m = cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0f32;
            for v in cpu.iter_mut() {
                *v = (*v - m).exp();
                s += *v;
            }
            for v in cpu.iter_mut() {
                *v /= s;
            }
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let total: f32 = gpu.iter().sum();
            assert!((total - 1.0).abs() < 1e-4, "softmax sum n={n}: {total}");
            assert!(max_abs < 1e-5, "softmax n={n} max|Δ|={max_abs}");
        }
    }

    #[test]
    fn l2_norm_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &n in &[128usize, 256, 1024] {
            // include a few that hit the eps clamp (very small magnitudes)
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 1e-2).sin()).collect();
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n as u64],
                GgmlType::F32,
            )
            .unwrap();
            let gpu = one_shot_f32_out(&ctx, n, |enc, y| {
                encode_l2_norm_f32(&ctx, enc, &x_t, y, 1e-6)
            });

            // CPU reference: y = x / max(||x||, eps).
            let sq: f32 = x.iter().map(|v| v * v).sum();
            let scale = 1.0 / sq.sqrt().max(1e-6);
            let cpu: Vec<f32> = x.iter().map(|v| v * scale).collect();
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max_abs < 1e-5, "l2_norm n={n}: max|Δ|={max_abs}");
        }
    }

    #[test]
    fn get_rows_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let vocab = 100usize;
        let n_cols = 64usize;
        let embed: Vec<f32> = (0..vocab * n_cols).map(|i| i as f32 * 0.001).collect();
        let ids: Vec<i32> = vec![3, 17, 42, 99];
        let n_rows = ids.len();

        let embed_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&embed),
            vec![n_cols as u64, vocab as u64],
            GgmlType::F32,
        )
        .unwrap();
        let ids_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![n_rows as u64],
            GgmlType::F32, // dtype tag is unused for the i32 buffer here
        )
        .unwrap();
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_rows as u64, n_cols as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_get_rows_f32(&ctx, enc, &embed_t, &ids_t, &y_t, n_rows, n_cols)
        })
        .unwrap();
        let gpu = read_back_f32(&y_t.buffer, n_rows * n_cols);
        for r in 0..n_rows {
            let row = ids[r] as usize;
            for i in 0..n_cols {
                let expected = embed[row * n_cols + i];
                let got = gpu[r * n_cols + i];
                assert!(
                    (got - expected).abs() < 1e-7,
                    "get_rows row={r} ids={} col={i}: {got} vs {expected}",
                    ids[r]
                );
            }
        }
    }

    /// Ensures the chained-encoding API is correctness-equivalent to one-shot.
    /// (The bench's only structural difference vs `encode_*` is that it loops
    /// `encode_*` inside the same encoder; if it diverges, the kernel is
    /// reading non-deterministic state — a bug.)
    #[test]
    fn chained_encoding_is_correct() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let n_in = 1024;
        let n_out = 4096;
        let w: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i % 17) as f32 - 8.0) * 1e-3)
            .collect();
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 7) as f32 - 3.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&w, n_in, n_out, &x);

        let w_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&w),
            vec![n_in as u64, n_out as u64],
            GgmlType::F32,
        )
        .unwrap();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .unwrap();
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();

        // Chain 8 dispatches; final result should equal one dispatch (each
        // overwrites the previous).
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        for _ in 0..8 {
            encode_mat_vec_f32(&ctx, &enc, &w_t, &x_t, &y_t, n_in, n_out).unwrap();
        }
        enc.end();
        cmd.commit();
        unsafe { cmd.waitUntilCompleted() };
        let gpu = read_back_f32(&y_t.buffer, n_out);

        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "chained encoding diverged: max|Δ|={max_abs}"
        );
    }
}
