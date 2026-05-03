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
    // Defense against the silent-NaN class of bug. If a quantized weight
    // tensor lands here by mistake (someone forgot to call
    // encode_mat_vec_dispatch), we'd reinterpret block bytes as floats
    // and produce garbage that propagates as NaNs through later layers.
    // Caught codex-review-style by adding this guard. v1.
    if weight.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_f32",
            detail: format!(
                "weight.dtype = {:?}, expected F32 — use encode_mat_vec_dispatch \
                 to handle quantized weights",
                weight.dtype
            ),
        });
    }
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

/// Copy `n_elements` floats starting at `src_off` (in elements) of `src`
/// into `dst[0..n_elements]`. Used to slice fused buffers (e.g. the GDN
/// post-conv qkv buffer) into per-role tensors. v2 will replace many of
/// these with kernels that take offsets directly.
pub fn encode_copy_offset_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    src_off: usize,
    dst: &MetalTensor,
    n_elements: usize,
) -> Result<(), MetalError> {
    if dst.n_elements() as usize != n_elements {
        return Err(MetalError::BadShape {
            kernel: "copy_offset",
            detail: format!("dst.n={} != n_elements={n_elements}", dst.n_elements()),
        });
    }
    if (src_off + n_elements) as u64 > src.n_elements() {
        return Err(MetalError::BadShape {
            kernel: "copy_offset",
            detail: format!(
                "src_off+n={} > src.n={}",
                src_off + n_elements,
                src.n_elements()
            ),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        src_off: u32,
    }
    let pso = ctx.pipeline("kernel_copy_offset_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n_elements as u32,
            src_off: src_off as u32,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, dst);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_tg = n_elements.div_ceil(tg_threads);
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

/// Per-head RMSNorm with shared per-channel weight. Used for Q-norm
/// and K-norm in the gated-attention block. One dispatch covers all
/// heads (n_heads threadgroups, simdgroup-reduce inside).
pub fn encode_rms_norm_batched_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    weight: &MetalTensor,
    y: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let want = (n_heads * head_dim) as u64;
    if x.n_elements() != want || y.n_elements() != want {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_batched",
            detail: format!("x/y expected {want} elements"),
        });
    }
    if weight.n_elements() as usize != head_dim {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_batched",
            detail: format!("weight expected {head_dim} elements"),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    }
    let pso = ctx.pipeline("kernel_rms_norm_batched_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
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
            width: n_heads,
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

/// Split the gated-attention Q-projection output into separate Q and
/// gate tensors. Input layout per head: `[head_dim Q, head_dim gate]`,
/// total length `n_heads * 2 * head_dim`. Outputs are
/// `[n_heads, head_dim]` each.
pub fn encode_split_q_gate_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_full: &MetalTensor,
    q: &MetalTensor,
    gate: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    let want_full = (n_heads * 2 * head_dim) as u64;
    let want_each = (n_heads * head_dim) as u64;
    if q_full.n_elements() != want_full {
        return Err(MetalError::BadShape {
            kernel: "split_q_gate",
            detail: format!("q_full expected {want_full} elements"),
        });
    }
    if q.n_elements() != want_each || gate.n_elements() != want_each {
        return Err(MetalError::BadShape {
            kernel: "split_q_gate",
            detail: format!("q/gate expected {want_each} elements"),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
    }
    let pso = ctx.pipeline("kernel_split_q_gate_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
        },
    );
    enc.set_tensor(1, q_full);
    enc.set_tensor(2, q);
    enc.set_tensor(3, gate);

    let total = n_heads * head_dim;
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_tg = total.div_ceil(tg_threads);
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

/// Fused attention decode (single-token Q step). For each Q head:
///   1. scores[p] = (q · k_cache[p, kvh, :]) * scale
///   2. softmax over scores
///   3. out[d] = Σ_p scores[p] · v_cache[p, kvh, d]
///
/// `kvh = qh / (n_q_heads / n_kv_heads)` — GQA group mapping.
///
/// Constraint: `n_pos * sizeof(f32)` must fit in threadgroup memory.
/// Apple Silicon's typical max is 32 KB, so `n_pos ≤ ~8000`. For longer
/// contexts in v2 we'll switch to streaming softmax.
pub fn encode_attn_decode_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    out: &MetalTensor,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_pos: usize,
) -> Result<(), MetalError> {
    if n_q_heads % n_kv_heads != 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode",
            detail: format!("n_q_heads={n_q_heads} not multiple of n_kv_heads={n_kv_heads}"),
        });
    }
    let want = (n_q_heads * head_dim) as u64;
    if q.n_elements() != want || out.n_elements() != want {
        return Err(MetalError::BadShape {
            kernel: "attn_decode",
            detail: format!("q/out expected {want} elements"),
        });
    }

    let kv_stride = n_kv_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        scale: f32,
    }
    let pso = ctx.pipeline("kernel_attn_decode_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            scale,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_simdgroups = tg_threads.div_ceil(32);
    let scores_bytes = n_pos * std::mem::size_of::<f32>();
    let shred_bytes = (n_simdgroups * std::mem::size_of::<f32>()).max(32);
    if scores_bytes > 28 * 1024 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode",
            detail: format!(
                "n_pos={n_pos} requires {scores_bytes} B threadgroup memory; \
                 v1 max ~28 KB. Use streaming softmax for longer contexts."
            ),
        });
    }
    enc.set_threadgroup_memory(0, scores_bytes);
    enc.set_threadgroup_memory(1, shred_bytes);

    enc.dispatch(
        MTLSize {
            width: n_q_heads,
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

/// Per-head L2-norm: `y[h, :] = x[h, :] / max(||x[h, :]||, eps)` for
/// `h ∈ [0, n_heads)`. One dispatch covers all heads. Used in the GDN
/// front-end where Q and K are l2-normed per K-head before the
/// recurrence; replaces n_heads separate calls.
pub fn encode_l2_norm_batched_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let want = (n_heads * head_dim) as u64;
    if x.n_elements() != want || y.n_elements() != want {
        return Err(MetalError::BadShape {
            kernel: "l2_norm_batched",
            detail: format!("x/y expected {want} elements"),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    }
    let pso = ctx.pipeline("kernel_l2_norm_batched_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
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
            width: n_heads,
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

/// In-place partial RoPE with NEOX pairing. Rotates the first `n_rot`
/// dims of each head; leaves `[n_rot, head_dim)` untouched. For text-only
/// positions, IMROPE/MROPE collapse to this plain form — sections only
/// differ for vision/video.
///
/// `n_rot` should be `head_dim * partial_rotary_factor` (= 64 for
/// Qwen3.5/3.6 with head_dim=256, factor=0.25).
///
/// CPU oracle: `forward::rope_in_place`.
pub fn encode_rope_neox_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    buf: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    if buf.n_elements() as usize != n_heads * head_dim {
        return Err(MetalError::BadShape {
            kernel: "rope_neox",
            detail: format!(
                "buf.n={} != n_heads*head_dim={}",
                buf.n_elements(),
                n_heads * head_dim
            ),
        });
    }
    if n_rot % 2 != 0 || n_rot > head_dim {
        return Err(MetalError::BadShape {
            kernel: "rope_neox",
            detail: format!("n_rot={n_rot} must be even and ≤ head_dim={head_dim}"),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        n_rot: u32,
        position: u32,
        theta_base: f32,
    }
    let pso = ctx.pipeline("kernel_rope_neox_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            n_rot: n_rot as u32,
            position,
            theta_base,
        },
    );
    enc.set_tensor(1, buf);

    let total_pairs = n_heads * (n_rot / 2);
    let tg_threads = 64usize;
    let n_tg = total_pairs.div_ceil(tg_threads);
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

/// SSM conv1d step + SiLU. Per-channel depthwise convolution of width K
/// (=4 for Qwen3.5/3.6), then SiLU. Mutates `conv_buf` (slides time
/// window). See `kernels/ssm_conv.metal` for layout details.
///
/// CPU oracle: the conv block in `forward::Forward::gdn_step` (lines
/// ~358-400 of forward.rs).
pub fn encode_ssm_conv_silu_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    qkv_now: &MetalTensor,
    conv_buf: &MetalTensor,
    conv_w: &MetalTensor,
    out: &MetalTensor,
    conv_dim: usize,
) -> Result<(), MetalError> {
    if qkv_now.n_elements() as usize != conv_dim {
        return Err(MetalError::BadShape {
            kernel: "ssm_conv",
            detail: format!("qkv_now.n={} != conv_dim={conv_dim}", qkv_now.n_elements()),
        });
    }
    if out.n_elements() as usize != conv_dim {
        return Err(MetalError::BadShape {
            kernel: "ssm_conv",
            detail: format!("out.n={} != conv_dim", out.n_elements()),
        });
    }
    // conv_buf must be (K-1) * conv_dim
    if conv_buf.n_elements() as usize != 3 * conv_dim {
        return Err(MetalError::BadShape {
            kernel: "ssm_conv",
            detail: format!(
                "conv_buf.n={} != (K-1)*conv_dim={}",
                conv_buf.n_elements(),
                3 * conv_dim
            ),
        });
    }
    // conv_w must be conv_dim * K
    if conv_w.n_elements() as usize != 4 * conv_dim {
        return Err(MetalError::BadShape {
            kernel: "ssm_conv",
            detail: format!(
                "conv_w.n={} != K*conv_dim={}",
                conv_w.n_elements(),
                4 * conv_dim
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        conv_dim: u32,
    }
    let pso = ctx.pipeline("kernel_ssm_conv_silu_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            conv_dim: conv_dim as u32,
        },
    );
    enc.set_tensor(1, qkv_now);
    enc.set_tensor(2, conv_buf);
    enc.set_tensor(3, conv_w);
    enc.set_tensor(4, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_tg = conv_dim.div_ceil(tg_threads);
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

/// RMSNormGated: per-head RMSNorm of `o` with weight, multiplied by
/// silu(z). Used immediately after the GDN recurrence, before out_proj.
///
/// CPU oracle: per-head loop in `forward::Forward::gdn_step` (the
/// "RMSNormGated" block).
pub fn encode_rmsnorm_gated_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    o: &MetalTensor,
    weight: &MetalTensor,
    z: &MetalTensor,
    y: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let want = (n_heads * head_dim) as u64;
    if o.n_elements() != want || z.n_elements() != want || y.n_elements() != want {
        return Err(MetalError::BadShape {
            kernel: "rmsnorm_gated",
            detail: format!(
                "o/z/y expected {want} elements (n_heads={n_heads} * head_dim={head_dim})"
            ),
        });
    }
    if weight.n_elements() as usize != head_dim {
        return Err(MetalError::BadShape {
            kernel: "rmsnorm_gated",
            detail: format!("weight expected {head_dim} elements"),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    }
    let pso = ctx.pipeline("kernel_rmsnorm_gated_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, o);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, z);
    enc.set_tensor(4, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: n_heads,
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

/// Single-step GDN recurrence: per-V-head delta-rule update + output.
///
/// Performs (per V-head) all of:
///   * decay:  S ← exp(g) · S
///   * inner:  s_k = S · k
///   * delta:  Δ = (v − s_k) · β
///   * update: S += Δ ⊗ k
///   * output: o = (S · q) / √head_dim
///
/// All in one kernel, with S held in registers across the (decay → inner
/// → update → output) sequence. State is read from / written back to
/// `state`; the rest are read-only (one timestep per call). For multi-
/// token prefill we'd loop the recurrence inside the kernel; v1 is
/// single-token decode, so T=1.
///
/// Hardcoded for `head_dim = 128` (Qwen3.5/3.6 GDN). When that changes,
/// the kernel needs templating on `dks_per_lane = head_dim / 32`.
///
/// CPU oracle: per-V-head loop in `crate::forward::Forward::gdn_step`.
pub fn encode_gdn_step_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    g: &MetalTensor,
    beta: &MetalTensor,
    state: &MetalTensor,
    out: &MetalTensor,
    n_v_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    if head_dim != 128 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("head_dim={head_dim} but kernel hardcodes 128"),
        });
    }
    if n_v_heads % n_k_heads != 0 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("n_v_heads={n_v_heads} not multiple of n_k_heads={n_k_heads}"),
        });
    }
    let want_qk = (n_k_heads * head_dim) as u64;
    let want_v = (n_v_heads * head_dim) as u64;
    if q.n_elements() != want_qk || k.n_elements() != want_qk {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("q/k expected {want_qk} elements (n_k_heads={n_k_heads})"),
        });
    }
    if v.n_elements() != want_v {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("v expected {want_v} elements (n_v_heads={n_v_heads})"),
        });
    }
    if g.n_elements() != n_v_heads as u64 || beta.n_elements() != n_v_heads as u64 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("g/beta expected {n_v_heads} elements"),
        });
    }
    let want_state = (n_v_heads * head_dim * head_dim) as u64;
    if state.n_elements() != want_state {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("state expected {want_state} elements"),
        });
    }
    if out.n_elements() != want_v {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("out expected {want_v} elements"),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_v_heads: u32,
        n_k_heads: u32,
    }
    let pso = ctx.pipeline("kernel_gdn_step_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_v_heads: n_v_heads as u32,
            n_k_heads: n_k_heads as u32,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k);
    enc.set_tensor(3, v);
    enc.set_tensor(4, g);
    enc.set_tensor(5, beta);
    enc.set_tensor(6, state);
    enc.set_tensor(7, out);

    // 2D grid: (head_dim, n_v_heads). One simdgroup (32 threads) per (dv, hi).
    enc.dispatch(
        MTLSize {
            width: head_dim,
            height: n_v_heads,
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

/// Q5_K mat-vec, same API shape as [`encode_mat_vec_q4_k_f32`].
/// Block size 176 bytes / 256 elements.
pub fn encode_mat_vec_q5_k_f32(
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
            kernel: "mat_vec_q5_k",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q5_k",
            detail: format!("weight.dtype = {:?}, expected Q5_K", weight.dtype),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_q5_K_f32")?;
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

    // NR0=1, NSG=2 → 2 output rows per threadgroup.
    const NR0: usize = 1;
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

/// One-shot Q5_K mat-vec for tests.
pub fn mat_vec_q5_k_f32_readback_for_test(
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
        GgmlType::Q5_K,
    )?;
    let x_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(x),
        vec![n_in as u64],
        GgmlType::F32,
    )?;
    let y_t = MetalTensor::zeros_f32(ctx, vec![n_out as u64])?;
    one_shot(ctx, |enc| {
        encode_mat_vec_q5_k_f32(ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
    })?;
    Ok(read_back_f32(&y_t.buffer, n_out))
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
    fn mat_vec_q5_k_matches_cpu() {
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
        let q5k = g
            .tensors
            .iter()
            .find(|t| {
                t.name.starts_with("blk.0.")
                    && t.dtype == GgmlType::Q5_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
            })
            .expect("no Q5_K tensor in 27B layer 0");
        let n_in = q5k.shape[0] as usize;
        let n_out = q5k.shape[1] as usize;
        eprintln!("[q5_k-test] {} shape=[{n_in}, {n_out}]", q5k.name);

        let weight_f32 = crate::codec::dequant_to_f32(q5k, g.slice(q5k)).expect("dequant");
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let gpu = mat_vec_q5_k_f32_readback_for_test(&ctx, g.slice(q5k), &x, n_in, n_out)
            .expect("metal q5k");
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q5_k] max|Δ|={max_abs:.2e}");
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
    fn l2_norm_batched_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(n_heads, head_dim) in &[(16usize, 128usize), (48, 128), (8, 256)] {
            let total = n_heads * head_dim;
            let x: Vec<f32> = (0..total)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
                .collect();
            let eps = 1e-6f32;

            // CPU reference: per head, y_h = x_h / max(||x_h||, eps).
            let mut cpu = vec![0.0f32; total];
            for h in 0..n_heads {
                let off = h * head_dim;
                let sq: f32 = (0..head_dim).map(|i| x[off + i].powi(2)).sum();
                let scale = 1.0 / sq.sqrt().max(eps);
                for i in 0..head_dim {
                    cpu[off + i] = x[off + i] * scale;
                }
            }

            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            let y_t = MetalTensor::zeros_f32(&ctx, vec![total as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_l2_norm_batched_f32(&ctx, enc, &x_t, &y_t, n_heads, head_dim, eps)
            })
            .unwrap();
            let gpu = read_back_f32(&y_t.buffer, total);

            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs < 1e-5,
                "l2_norm_batched n_heads={n_heads} head_dim={head_dim}: max|Δ|={max_abs}"
            );
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

    /// CPU reference: forward::rope_in_place — applies NEOX-pairing partial
    /// RoPE in place. We use it via the existing `forward::rope_in_place_pub`
    /// helper added below.
    fn rope_neox_cpu_ref(
        buf: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        n_rot: usize,
        position: u32,
        theta_base: f32,
    ) {
        let pos = position as f32;
        let half = n_rot / 2;
        for hi in 0..n_heads {
            let h_off = hi * head_dim;
            for i in 0..half {
                let exponent = (2 * i) as f32 / n_rot as f32;
                let freq = pos / theta_base.powf(exponent);
                let (s, c) = freq.sin_cos();
                let a = buf[h_off + i];
                let b = buf[h_off + i + half];
                buf[h_off + i] = a * c - b * s;
                buf[h_off + i + half] = a * s + b * c;
            }
        }
    }

    #[test]
    fn rope_neox_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        // Real shapes from Qwen3.5 family:
        //   0.8B: 8 Q heads, 256 head_dim, 64 rotated dims
        //   27B:  24 Q heads or 4 KV heads, 256 head_dim, 64 rotated dims
        let head_dim = 256;
        let n_rot = 64;
        let theta_base = 10_000_000.0f32;
        for &n_heads in &[8usize, 24, 4] {
            for &position in &[0u32, 1, 7, 100] {
                let total = n_heads * head_dim;
                let buf_init: Vec<f32> =
                    (0..total).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();

                let mut buf_cpu = buf_init.clone();
                rope_neox_cpu_ref(&mut buf_cpu, n_heads, head_dim, n_rot, position, theta_base);

                let buf_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&buf_init),
                    vec![total as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_rope_neox_f32(
                        &ctx, enc, &buf_t, n_heads, head_dim, n_rot, position, theta_base,
                    )
                })
                .unwrap();
                let gpu = read_back_f32(&buf_t.buffer, total);

                let max_abs = gpu
                    .iter()
                    .zip(buf_cpu.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    max_abs < 1e-5,
                    "rope_neox n_heads={n_heads} pos={position}: max|Δ|={max_abs}"
                );
            }
        }
    }

    /// CPU reference for ssm_conv_silu — mirrors the conv block in
    /// `forward::Forward::gdn_step`. Mutates `conv_buf` in place,
    /// returns conv output (post-SiLU).
    fn ssm_conv_silu_cpu_ref(
        qkv_now: &[f32],
        conv_buf: &mut [f32],
        conv_w: &[f32],
        conv_dim: usize,
    ) -> Vec<f32> {
        const K: usize = 4;
        let kmin1 = K - 1;
        let mut conv_input = vec![0.0f32; K * conv_dim];
        for t in 0..kmin1 {
            conv_input[t * conv_dim..(t + 1) * conv_dim]
                .copy_from_slice(&conv_buf[t * conv_dim..(t + 1) * conv_dim]);
        }
        conv_input[kmin1 * conv_dim..].copy_from_slice(qkv_now);

        let mut out = vec![0.0f32; conv_dim];
        for c in 0..conv_dim {
            let mut s = 0.0f32;
            for k in 0..K {
                s += conv_w[c * K + k] * conv_input[k * conv_dim + c];
            }
            out[c] = s / (1.0 + (-s).exp());
        }
        // Slide buffer (drop oldest, append current).
        for t in 0..kmin1 - 1 {
            for i in 0..conv_dim {
                conv_buf[t * conv_dim + i] = conv_buf[(t + 1) * conv_dim + i];
            }
        }
        conv_buf[(kmin1 - 1) * conv_dim..].copy_from_slice(qkv_now);
        out
    }

    #[test]
    fn ssm_conv_silu_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        // Real shapes: 0.8B has conv_dim = 2*16*128 + 16*128 = 6144;
        // 27B has conv_dim = 2*16*128 + 48*128 = 10240.
        for &conv_dim in &[6144usize, 10240] {
            const K: usize = 4;
            let qkv_now: Vec<f32> = (0..conv_dim)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let mut conv_buf: Vec<f32> = (0..(K - 1) * conv_dim)
                .map(|i| ((i % 13) as f32 - 6.0) * 5e-3)
                .collect();
            let conv_w: Vec<f32> = (0..conv_dim * K)
                .map(|i| ((i % 7) as f32 - 3.0) * 1e-2)
                .collect();

            // CPU oracle.
            let mut buf_cpu = conv_buf.clone();
            let out_cpu = ssm_conv_silu_cpu_ref(&qkv_now, &mut buf_cpu, &conv_w, conv_dim);

            // GPU.
            let qkv_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&qkv_now),
                vec![conv_dim as u64],
                GgmlType::F32,
            )
            .unwrap();
            let buf_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&conv_buf),
                vec![((K - 1) * conv_dim) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let w_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&conv_w),
                vec![(conv_dim * K) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let out_t = MetalTensor::zeros_f32(&ctx, vec![conv_dim as u64]).unwrap();

            one_shot(&ctx, |enc| {
                encode_ssm_conv_silu_f32(&ctx, enc, &qkv_t, &buf_t, &w_t, &out_t, conv_dim)
            })
            .unwrap();

            let out_gpu = read_back_f32(&out_t.buffer, conv_dim);
            let buf_gpu = read_back_f32(&buf_t.buffer, (K - 1) * conv_dim);

            let max_out = out_gpu
                .iter()
                .zip(out_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let max_buf = buf_gpu
                .iter()
                .zip(buf_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "[ssm_conv conv_dim={conv_dim}] max|out_Δ|={max_out:.2e} max|buf_Δ|={max_buf:.2e}"
            );
            assert!(max_out < 1e-5, "out drift {max_out}");
            assert!(max_buf < 1e-7, "buf drift {max_buf}");
        }
    }

    /// CPU reference for rmsnorm_gated — per-head RMSNorm of `o` * silu(z).
    fn rmsnorm_gated_cpu_ref(
        o: &[f32],
        weight: &[f32],
        z: &[f32],
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; n_heads * head_dim];
        for hi in 0..n_heads {
            let off = hi * head_dim;
            let sumsq: f32 = (0..head_dim).map(|i| o[off + i].powi(2)).sum();
            let scale = 1.0 / (sumsq / head_dim as f32 + eps).sqrt();
            for i in 0..head_dim {
                let zi = z[off + i];
                let silu_z = zi / (1.0 + (-zi).exp());
                y[off + i] = (o[off + i] * scale * weight[i]) * silu_z;
            }
        }
        y
    }

    #[test]
    fn rmsnorm_gated_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(n_heads, head_dim) in &[(16usize, 128usize), (48, 128)] {
            let total = n_heads * head_dim;
            let o: Vec<f32> = (0..total)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.05)
                .collect();
            let z: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
            let weight: Vec<f32> = (0..head_dim).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
            let eps = 1e-6;

            let cpu = rmsnorm_gated_cpu_ref(&o, &weight, &z, n_heads, head_dim, eps);

            let o_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&o),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            let w_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&weight),
                vec![head_dim as u64],
                GgmlType::F32,
            )
            .unwrap();
            let z_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&z),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            let y_t = MetalTensor::zeros_f32(&ctx, vec![total as u64]).unwrap();

            one_shot(&ctx, |enc| {
                encode_rmsnorm_gated_f32(&ctx, enc, &o_t, &w_t, &z_t, &y_t, n_heads, head_dim, eps)
            })
            .unwrap();

            let gpu = read_back_f32(&y_t.buffer, total);
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[rmsnorm_gated n_heads={n_heads} head_dim={head_dim}] max|Δ|={max_abs:.2e}");
            assert!(max_abs < 1e-4, "rmsnorm_gated drift {max_abs}");
        }
    }

    /// CPU reference for the GDN-step kernel — mirrors the per-V-head
    /// inner loop in `forward::Forward::gdn_step` exactly. Mutates
    /// `state` in place and returns `out`.
    fn gdn_step_cpu_ref(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state: &mut [f32],
        n_v: usize,
        n_k: usize,
        hd: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n_v * hd];
        let scale = 1.0 / (hd as f32).sqrt();
        for hi in 0..n_v {
            let hk = hi % n_k;
            let s_off = hi * hd * hd;
            let q_h = &q[hk * hd..(hk + 1) * hd];
            let k_h = &k[hk * hd..(hk + 1) * hd];
            let v_h = &v[hi * hd..(hi + 1) * hd];
            let g_h = g[hi].exp();
            let b_h = beta[hi];

            // Decay: S *= g_h
            for j in 0..hd * hd {
                state[s_off + j] *= g_h;
            }
            // s_k[dv] = sum_dk S[dv,dk] * k[dk]
            let mut sk = vec![0.0f32; hd];
            for dv in 0..hd {
                let mut s = 0.0f32;
                for dk in 0..hd {
                    s += state[s_off + dv * hd + dk] * k_h[dk];
                }
                sk[dv] = s;
            }
            // Update: S[dv,dk] += beta * (v[dv] - sk[dv]) * k[dk]
            for dv in 0..hd {
                let coeff = b_h * (v_h[dv] - sk[dv]);
                for dk in 0..hd {
                    state[s_off + dv * hd + dk] += coeff * k_h[dk];
                }
            }
            // Output: o[dv] = (sum_dk S[dv,dk] * q[dk]) * scale
            for dv in 0..hd {
                let mut s = 0.0f32;
                for dk in 0..hd {
                    s += state[s_off + dv * hd + dk] * q_h[dk];
                }
                out[hi * hd + dv] = s * scale;
            }
        }
        out
    }

    #[test]
    fn gdn_step_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        // Cover both Qwen3.5/3.6 sizes:
        //   0.8B: n_v_heads = 16, head_dim = 128
        //   27B:  n_v_heads = 48, head_dim = 128
        // 0.8B has n_v == n_k == 16; 27B has n_v=48, n_k=16 (3:1 repeat).
        for &(n_v, n_k) in &[(16usize, 16usize), (48, 16)] {
            let hd = 128usize;
            // Synthetic but realistic-magnitude inputs. Q/K are sized to
            // n_k heads; V is sized to n_v heads.
            let q: Vec<f32> = (0..n_k * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k: Vec<f32> = (0..n_k * hd)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v: Vec<f32> = (0..n_v * hd)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();
            let g: Vec<f32> = (0..n_v).map(|i| -((i % 7) as f32) * 1e-3).collect();
            let beta: Vec<f32> = (0..n_v).map(|i| 0.5 + ((i % 11) as f32) * 1e-2).collect();
            // Random-ish but deterministic state, including the
            // post-first-token regime (nonzero initial state) since
            // codex flagged "first-token only" coverage as inadequate.
            let state: Vec<f32> = (0..n_v * hd * hd)
                .map(|i| ((i % 13) as f32 - 6.0) * 1e-3)
                .collect();

            // CPU oracle.
            let mut state_cpu = state.clone();
            let out_cpu = gdn_step_cpu_ref(&q, &k, &v, &g, &beta, &mut state_cpu, n_v, n_k, hd);

            // GPU.
            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_k * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&k),
                vec![(n_k * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let v_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&v),
                vec![(n_v * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let g_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&g),
                vec![n_v as u64],
                GgmlType::F32,
            )
            .unwrap();
            let beta_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&beta),
                vec![n_v as u64],
                GgmlType::F32,
            )
            .unwrap();
            let state_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&state),
                vec![(n_v * hd * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let out_t = MetalTensor::zeros_f32(&ctx, vec![(n_v * hd) as u64]).unwrap();

            one_shot(&ctx, |enc| {
                encode_gdn_step_f32(
                    &ctx, enc, &q_t, &k_t, &v_t, &g_t, &beta_t, &state_t, &out_t, n_v, n_k, hd,
                )
            })
            .unwrap();

            let out_gpu = read_back_f32(&out_t.buffer, n_v * hd);
            let state_gpu = read_back_f32(&state_t.buffer, n_v * hd * hd);

            // Output comparison.
            let max_out = out_gpu
                .iter()
                .zip(out_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            // State comparison (this is the recurrent variable; correctness
            // here matters more than the output for multi-step decode).
            let max_state = state_gpu
                .iter()
                .zip(state_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);

            eprintln!(
                "[gdn_step n_v={n_v} n_k={n_k}] max|out_Δ|={max_out:.2e}  max|state_Δ|={max_state:.2e}"
            );
            // simd_sum reduction order can drift slightly from the
            // sequential CPU version; 1e-4 covers it for our magnitudes.
            assert!(max_out < 1e-4, "out drift {max_out}");
            assert!(max_state < 1e-4, "state drift {max_state}");
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
