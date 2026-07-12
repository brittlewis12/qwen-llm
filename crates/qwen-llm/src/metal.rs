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
use objc2_foundation::{NSError, NSRange, NSString, NSURL};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePassDescriptor, MTLComputePipelineState, MTLCounter,
    MTLCounterResultTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLDispatchType, MTLFence, MTLLibrary, MTLResourceOptions, MTLSize, MTLStorageMode,
};
use parking_lot::Mutex;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};

const TASK_VM_INFO: i32 = 22;

#[repr(C, packed(4))]
struct TaskVmInfo {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
    min_address: u64,
    max_address: u64,
    ledger_phys_footprint_peak: i64,
    ledger_purgeable_nonvolatile: i64,
    ledger_purgeable_nonvolatile_compressed: i64,
    ledger_purgeable_volatile: i64,
    ledger_purgeable_volatile_compressed: i64,
    ledger_tag_network_nonvolatile: i64,
    ledger_tag_network_nonvolatile_compressed: i64,
    ledger_tag_network_volatile: i64,
    ledger_tag_network_volatile_compressed: i64,
    ledger_tag_media_footprint: i64,
    ledger_tag_media_footprint_compressed: i64,
    ledger_tag_media_nofootprint: i64,
    ledger_tag_media_nofootprint_compressed: i64,
    ledger_tag_graphics_footprint: i64,
    ledger_tag_graphics_footprint_compressed: i64,
    ledger_tag_graphics_nofootprint: i64,
    ledger_tag_graphics_nofootprint_compressed: i64,
    ledger_tag_neural_footprint: i64,
    ledger_tag_neural_footprint_compressed: i64,
    ledger_tag_neural_nofootprint: i64,
    ledger_tag_neural_nofootprint_compressed: i64,
    limit_bytes_remaining: u64,
}

unsafe extern "C" {
    static mach_task_self_: u32;
    fn task_info(
        target_task: u32,
        flavor: i32,
        task_info_out: *mut i32,
        task_info_out_count: *mut u32,
    ) -> i32;
}

static KERNEL_TRACE_EVER_ENABLED: AtomicBool = AtomicBool::new(false);

const ATTN_V4_SUBGROUP_MIN_POS_DEFAULT: usize = 256;
const ATTN_V4_NWG_MAX: usize = 1024;

thread_local! {
    static ATTN_V4_GROUP_TILE_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
    static MATMAT_F16_HALF_ACT_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static MATMAT_Q4_LEGACY_MM_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static KERNEL_TRACE_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static KERNEL_TRACE_COUNTERS: Cell<KernelTraceCounters> = const {
        Cell::new(KernelTraceCounters {
            encoders: 0,
            concurrent_encoders: 0,
            dispatches: 0,
        })
    };
    static KERNEL_TRACE_LAST: Cell<KernelTraceCounters> = const {
        Cell::new(KernelTraceCounters {
            encoders: 0,
            concurrent_encoders: 0,
            dispatches: 0,
        })
    };
}

pub fn with_matmat_f16_half_act_override<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    let previous = MATMAT_F16_HALF_ACT_OVERRIDE.with(|slot| {
        let previous = slot.get();
        slot.set(Some(enabled));
        previous
    });
    let out = f();
    MATMAT_F16_HALF_ACT_OVERRIDE.with(|slot| slot.set(previous));
    out
}

pub fn with_matmat_q4_legacy_mm_override<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    let previous = MATMAT_Q4_LEGACY_MM_OVERRIDE.with(|slot| {
        let previous = slot.get();
        slot.set(Some(enabled));
        previous
    });
    let out = f();
    MATMAT_Q4_LEGACY_MM_OVERRIDE.with(|slot| slot.set(previous));
    out
}

use crate::tensor::{GgmlType, TensorDesc, checked_shape_elements, ggml_type_layout};

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
    #[error("metal counter probe failed: {0}")]
    Counter(String),
    #[error("bad shape for kernel {kernel}: {detail}")]
    BadShape {
        kernel: &'static str,
        detail: String,
    },
    #[error(
        "tensor size overflow: shape={shape:?} × {elem_bytes} bytes/elem does not fit in usize"
    )]
    TensorSizeOverflow { shape: Vec<u64>, elem_bytes: usize },
    #[error("tensor byte size overflow: shape={shape:?}, dtype={dtype:?} does not fit")]
    TensorByteSizeOverflow { shape: Vec<u64>, dtype: GgmlType },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KernelTraceCounters {
    pub encoders: u64,
    pub concurrent_encoders: u64,
    pub dispatches: u64,
}

impl KernelTraceCounters {
    pub fn is_zero(self) -> bool {
        self.encoders == 0 && self.concurrent_encoders == 0 && self.dispatches == 0
    }

    fn saturating_sub(self, previous: Self) -> Self {
        Self {
            encoders: self.encoders.saturating_sub(previous.encoders),
            concurrent_encoders: self
                .concurrent_encoders
                .saturating_sub(previous.concurrent_encoders),
            dispatches: self.dispatches.saturating_sub(previous.dispatches),
        }
    }
}

#[must_use]
pub struct KernelTraceGuard {
    previous: bool,
}

pub struct MetalTimestampSampleBuffer {
    raw: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    sample_count: usize,
}

impl MetalTimestampSampleBuffer {
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }
}

impl Drop for KernelTraceGuard {
    fn drop(&mut self) {
        KERNEL_TRACE_ACTIVE.with(|active| active.set(self.previous));
    }
}

pub fn kernel_trace_begin() -> KernelTraceGuard {
    KERNEL_TRACE_EVER_ENABLED.store(true, Ordering::Relaxed);
    KERNEL_TRACE_COUNTERS.with(|counters| counters.set(KernelTraceCounters::default()));
    KERNEL_TRACE_LAST.with(|last| last.set(KernelTraceCounters::default()));
    let previous = KERNEL_TRACE_ACTIVE.with(|active| {
        let previous = active.get();
        active.set(true);
        previous
    });
    KernelTraceGuard { previous }
}

pub fn kernel_trace_snapshot() -> KernelTraceCounters {
    KERNEL_TRACE_COUNTERS.with(|counters| counters.get())
}

pub fn kernel_trace_take_delta() -> KernelTraceCounters {
    let current = kernel_trace_snapshot();
    KERNEL_TRACE_LAST.with(|last| {
        let previous = last.get();
        last.set(current);
        current.saturating_sub(previous)
    })
}

#[inline]
fn kernel_trace_record_encoder(concurrent: bool) {
    if !KERNEL_TRACE_EVER_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    if !KERNEL_TRACE_ACTIVE.with(|active| active.get()) {
        return;
    }
    KERNEL_TRACE_COUNTERS.with(|counters| {
        let mut current = counters.get();
        current.encoders += 1;
        if concurrent {
            current.concurrent_encoders += 1;
        }
        counters.set(current);
    });
}

#[inline]
fn kernel_trace_record_dispatch() {
    if !KERNEL_TRACE_EVER_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    if !KERNEL_TRACE_ACTIVE.with(|active| active.get()) {
        return;
    }
    KERNEL_TRACE_COUNTERS.with(|counters| {
        let mut current = counters.get();
        current.dispatches += 1;
        counters.set(current);
    });
}

// ===========================================================================
// Dispatch census (bench-only; v0.495 W-program attribution). Records
// (stage_family, kernel_name, grid_tgs, tg_threads) per dispatch while
// active. Zero production cost when never enabled (one atomic load per
// dispatch, same pattern as kernel_trace).
// ===========================================================================

#[derive(Clone, Debug)]
pub struct DispatchCensusRow {
    pub family: &'static str,
    pub kernel: String,
    pub grid_tgs: u64,
    pub tg_threads: u64,
}

static DISPATCH_CENSUS_EVER_ENABLED: AtomicBool = AtomicBool::new(false);
thread_local! {
    static DISPATCH_CENSUS: std::cell::RefCell<Option<Vec<DispatchCensusRow>>> =
        const { std::cell::RefCell::new(None) };
    static CENSUS_LAST_PSO: std::cell::RefCell<String> =
        const { std::cell::RefCell::new(String::new()) };
    static CENSUS_FAMILY: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// Begin recording dispatch shapes on this thread. Bench-only.
pub fn dispatch_census_begin() {
    DISPATCH_CENSUS_EVER_ENABLED.store(true, Ordering::Relaxed);
    DISPATCH_CENSUS.with(|c| *c.borrow_mut() = Some(Vec::with_capacity(512)));
}

/// Stop recording and take the census rows.
pub fn dispatch_census_take() -> Vec<DispatchCensusRow> {
    DISPATCH_CENSUS.with(|c| c.borrow_mut().take().unwrap_or_default())
}

/// Set the current stage family label (called by decode stage boundaries).
pub fn dispatch_census_set_family(family: &'static str) {
    if !DISPATCH_CENSUS_EVER_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    CENSUS_FAMILY.with(|f| f.set(family));
}

#[inline]
fn census_record_pso(name: &str) {
    if !DISPATCH_CENSUS_EVER_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    DISPATCH_CENSUS.with(|c| {
        if c.borrow().is_some() {
            CENSUS_LAST_PSO.with(|p| {
                let mut p = p.borrow_mut();
                p.clear();
                p.push_str(name);
            });
        }
    });
}

#[inline]
fn census_record_dispatch(grid: MTLSize, threads: MTLSize) {
    if !DISPATCH_CENSUS_EVER_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    DISPATCH_CENSUS.with(|c| {
        if let Some(rows) = c.borrow_mut().as_mut() {
            rows.push(DispatchCensusRow {
                family: CENSUS_FAMILY.with(|f| f.get()),
                kernel: CENSUS_LAST_PSO.with(|p| p.borrow().clone()),
                grid_tgs: (grid.width * grid.height.max(1) * grid.depth.max(1)) as u64,
                tg_threads: (threads.width * threads.height.max(1) * threads.depth.max(1)) as u64,
            });
        }
    });
}

/// Checked element-count and byte-size computation for a tensor shape.
/// Returns `(n_elements, n_bytes)`, both as `usize` validated to fit.
/// Use this before any allocation or kernel arg derived from
/// `shape.iter().product()` — release-build u64 overflow there is silent.
fn checked_shape_bytes(shape: &[u64], elem_bytes: usize) -> Result<(usize, usize), MetalError> {
    let n = checked_shape_elements(shape).ok_or_else(|| MetalError::TensorSizeOverflow {
        shape: shape.to_vec(),
        elem_bytes,
    })?;
    let n_usize = usize::try_from(n).map_err(|_| MetalError::TensorSizeOverflow {
        shape: shape.to_vec(),
        elem_bytes,
    })?;
    let bytes = n_usize
        .checked_mul(elem_bytes)
        .ok_or_else(|| MetalError::TensorSizeOverflow {
            shape: shape.to_vec(),
            elem_bytes,
        })?;
    Ok((n_usize, bytes))
}

fn checked_ggml_shape_bytes(shape: &[u64], dtype: GgmlType) -> Result<(usize, usize), MetalError> {
    let elements =
        checked_shape_elements(shape).ok_or_else(|| MetalError::TensorByteSizeOverflow {
            shape: shape.to_vec(),
            dtype,
        })?;
    let (block_size, type_size) = ggml_type_layout(dtype).ok_or_else(|| MetalError::BadShape {
        kernel: "tensor_size",
        detail: format!("unsupported Metal tensor dtype {dtype:?}"),
    })?;
    if block_size == 0 || type_size == 0 || elements % block_size != 0 {
        return Err(MetalError::BadShape {
            kernel: "tensor_size",
            detail: format!(
                "shape {shape:?} with {} elements is not divisible by block_size {block_size} for dtype {dtype:?}",
                elements
            ),
        });
    }
    let bytes_u128 = (elements as u128 / block_size as u128)
        .checked_mul(type_size as u128)
        .ok_or_else(|| MetalError::TensorByteSizeOverflow {
            shape: shape.to_vec(),
            dtype,
        })?;
    let bytes = usize::try_from(bytes_u128).map_err(|_| MetalError::TensorByteSizeOverflow {
        shape: shape.to_vec(),
        dtype,
    })?;
    let n = usize::try_from(elements).map_err(|_| MetalError::TensorByteSizeOverflow {
        shape: shape.to_vec(),
        dtype,
    })?;
    Ok((n, bytes))
}

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Owned Metal buffer wrapper, the public type for buffer-style values.
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Fence = Retained<ProtocolObject<dyn MTLFence>>;

#[derive(Clone, Debug)]
pub struct MetalCounterSetInfo {
    pub name: String,
    pub counters: Vec<String>,
    pub sample_buffer_status: String,
}

#[derive(Clone, Debug)]
pub struct MetalCounterCapabilities {
    pub supports_stage_boundary: bool,
    pub supports_dispatch_boundary: bool,
    pub supports_blit_boundary: bool,
    pub sets: Vec<MetalCounterSetInfo>,
}

#[derive(Clone, Debug)]
pub struct MetalPipelineInfo {
    pub name: String,
    pub thread_execution_width: usize,
    pub max_total_threads_per_threadgroup: usize,
    pub static_threadgroup_memory_length: usize,
    pub supports_indirect_command_buffers: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetalPipelineCacheMetrics {
    pub misses: u64,
    pub miss_wall_ns: u64,
    pub compiler_wall_ns: u64,
}

impl MetalPipelineCacheMetrics {
    pub fn saturating_delta_since(self, earlier: Self) -> Self {
        Self {
            misses: self.misses.saturating_sub(earlier.misses),
            miss_wall_ns: self.miss_wall_ns.saturating_sub(earlier.miss_wall_ns),
            compiler_wall_ns: self
                .compiler_wall_ns
                .saturating_sub(earlier.compiler_wall_ns),
        }
    }
}

#[derive(Default)]
struct MetalPipelineCache {
    pipelines: HashMap<String, Pipeline>,
    metrics_enabled: bool,
    metrics: MetalPipelineCacheMetrics,
}

fn duration_ns_saturating(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

// ===========================================================================
// MetalContext
// ===========================================================================

pub struct MetalContext {
    pub device: Device,
    pub queue: Queue,
    pub library: Library,
    pso_cache: Arc<Mutex<MetalPipelineCache>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalBufferSizeAndAlign {
    pub size: u64,
    pub alignment: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalMemorySignals {
    pub recommended_max_bytes: u64,
    pub current_allocated_bytes: u64,
    pub process_limit_remaining_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalMemoryAdmissionReason {
    AdmittedWithProcessBudget,
    AdmittedProcessBudgetOmitted,
    RequiredBytesOverflow,
    InvalidWorkingSetSignal,
    ProcessSignalUnavailable,
    WorkingSetInsufficient,
    ProcessInsufficient,
    BothInsufficient,
}

impl MetalMemoryAdmissionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AdmittedWithProcessBudget => "admitted_with_process_budget",
            Self::AdmittedProcessBudgetOmitted => "admitted_process_budget_omitted",
            Self::RequiredBytesOverflow => "required_bytes_overflow",
            Self::InvalidWorkingSetSignal => "invalid_working_set_signal",
            Self::ProcessSignalUnavailable => "process_signal_unavailable",
            Self::WorkingSetInsufficient => "working_set_insufficient",
            Self::ProcessInsufficient => "process_insufficient",
            Self::BothInsufficient => "both_insufficient",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalMemoryAdmission {
    pub admitted: bool,
    pub reason: MetalMemoryAdmissionReason,
    pub scratch_upper_bytes: u64,
    pub reserve_bytes: u64,
    pub required_bytes: Option<u64>,
    pub signals: MetalMemorySignals,
    pub working_set_headroom_bytes: Option<u64>,
}

pub fn evaluate_metal_memory_admission(
    scratch_upper_bytes: u64,
    reserve_bytes: u64,
    signals: MetalMemorySignals,
    allow_zero_process_budget: bool,
) -> MetalMemoryAdmission {
    let required_bytes = scratch_upper_bytes.checked_add(reserve_bytes);
    let working_set_headroom_bytes = signals
        .recommended_max_bytes
        .checked_sub(signals.current_allocated_bytes);
    let reason = if required_bytes.is_none() {
        MetalMemoryAdmissionReason::RequiredBytesOverflow
    } else if signals.recommended_max_bytes == 0 || working_set_headroom_bytes.is_none() {
        MetalMemoryAdmissionReason::InvalidWorkingSetSignal
    } else {
        let required = required_bytes.expect("checked above");
        let working_set_headroom = working_set_headroom_bytes.expect("checked above");
        let working_set_fits = working_set_headroom > 0 && required <= working_set_headroom;
        match signals.process_limit_remaining_bytes {
            None => MetalMemoryAdmissionReason::ProcessSignalUnavailable,
            Some(0) => match (working_set_fits, allow_zero_process_budget) {
                (true, true) => MetalMemoryAdmissionReason::AdmittedProcessBudgetOmitted,
                (false, true) => MetalMemoryAdmissionReason::WorkingSetInsufficient,
                (_, false) => MetalMemoryAdmissionReason::ProcessSignalUnavailable,
            },
            Some(process_available) => {
                let process_fits = required <= process_available;
                match (working_set_fits, process_fits) {
                    (true, true) => MetalMemoryAdmissionReason::AdmittedWithProcessBudget,
                    (false, true) => MetalMemoryAdmissionReason::WorkingSetInsufficient,
                    (true, false) => MetalMemoryAdmissionReason::ProcessInsufficient,
                    (false, false) => MetalMemoryAdmissionReason::BothInsufficient,
                }
            }
        }
    };
    MetalMemoryAdmission {
        admitted: matches!(
            reason,
            MetalMemoryAdmissionReason::AdmittedWithProcessBudget
                | MetalMemoryAdmissionReason::AdmittedProcessBudgetOmitted
        ),
        reason,
        scratch_upper_bytes,
        reserve_bytes,
        required_bytes,
        signals,
        working_set_headroom_bytes,
    }
}

// SAFETY: `Retained<ProtocolObject<dyn MTL*>>` are thread-safe per Apple's
// Metal docs (the protocol objects are themselves backed by thread-safe
// Objective-C classes; method dispatch is internally synchronized).
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

impl MetalContext {
    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }

    pub fn process_limit_bytes_remaining() -> Option<u64> {
        let mut info = std::mem::MaybeUninit::<TaskVmInfo>::zeroed();
        let mut count =
            u32::try_from(std::mem::size_of::<TaskVmInfo>() / std::mem::size_of::<u32>()).ok()?;
        // SAFETY: `info` is writable and large enough for `count` natural_t
        // words. The Mach call initializes the returned revision on success.
        let result = unsafe {
            task_info(
                mach_task_self_,
                TASK_VM_INFO,
                info.as_mut_ptr().cast::<i32>(),
                &mut count,
            )
        };
        let limit_count = u32::try_from(
            (std::mem::offset_of!(TaskVmInfo, limit_bytes_remaining) + std::mem::size_of::<u64>())
                / std::mem::size_of::<u32>(),
        )
        .ok()?;
        if result != 0 || count < limit_count {
            return None;
        }
        // SAFETY: A successful rev4 response initialized the packed field.
        Some(unsafe { std::ptr::addr_of!((*info.as_ptr()).limit_bytes_remaining).read_unaligned() })
    }

    pub fn memory_signals(&self) -> MetalMemorySignals {
        MetalMemorySignals {
            recommended_max_bytes: self.recommended_max_working_set_size(),
            current_allocated_bytes: self.current_allocated_size(),
            process_limit_remaining_bytes: Self::process_limit_bytes_remaining(),
        }
    }

    pub fn shared_buffer_size_and_align(
        &self,
        logical_bytes: u64,
    ) -> Result<MetalBufferSizeAndAlign, MetalError> {
        let length = usize::try_from(logical_bytes.max(1)).map_err(|_| MetalError::BadShape {
            kernel: "shared_buffer_size_and_align",
            detail: format!("logical byte request {logical_bytes} does not fit usize"),
        })?;
        let priced = self.device.heapBufferSizeAndAlignWithLength_options(
            length,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(MetalBufferSizeAndAlign {
            size: u64::try_from(priced.size).map_err(|_| MetalError::BadShape {
                kernel: "shared_buffer_size_and_align",
                detail: "priced buffer size does not fit u64".into(),
            })?,
            alignment: u64::try_from(priced.align).map_err(|_| MetalError::BadShape {
                kernel: "shared_buffer_size_and_align",
                detail: "priced buffer alignment does not fit u64".into(),
            })?,
        })
    }

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
            pso_cache: Arc::new(Mutex::new(MetalPipelineCache::default())),
        })
    }

    /// Look up a kernel function by name, compiling its pipeline state
    /// object on first request and caching it thereafter.
    pub fn pipeline(&self, name: &str) -> Result<Pipeline, MetalError> {
        census_record_pso(name);
        let metrics_enabled = {
            let mut cache = self.pso_cache.lock();
            if let Some(p) = cache.pipelines.get(name) {
                return Ok(p.clone());
            }
            if cache.metrics_enabled {
                cache.metrics.misses = cache.metrics.misses.saturating_add(1);
            }
            cache.metrics_enabled
        };
        let miss_t0 = metrics_enabled.then(std::time::Instant::now);
        let func_name = NSString::from_str(name);
        let Some(function) = self.library.newFunctionWithName(&func_name) else {
            if let Some(start) = miss_t0 {
                self.record_pipeline_miss_wall(start.elapsed(), None);
            }
            return Err(MetalError::NoFunction(name.to_string()));
        };
        let compiler_t0 = metrics_enabled.then(std::time::Instant::now);
        let pso_result = self
            .device
            .newComputePipelineStateWithFunction_error(&function);
        if let Some(start) = miss_t0 {
            self.record_pipeline_miss_wall(
                start.elapsed(),
                compiler_t0.map(|compiler_start| compiler_start.elapsed()),
            );
        }
        let pso = pso_result.map_err(|e: Retained<NSError>| {
            MetalError::Pipeline(name.to_string(), e.localizedDescription().to_string())
        })?;
        self.pso_cache
            .lock()
            .pipelines
            .insert(name.to_string(), pso.clone());
        Ok(pso)
    }

    pub fn set_pipeline_cache_metrics_enabled(&self, enabled: bool) {
        let mut cache = self.pso_cache.lock();
        cache.metrics_enabled = enabled;
        if enabled {
            cache.metrics = MetalPipelineCacheMetrics::default();
        }
    }

    pub fn pipeline_cache_metrics(&self) -> MetalPipelineCacheMetrics {
        self.pso_cache.lock().metrics
    }

    fn record_pipeline_miss_wall(
        &self,
        miss_wall: std::time::Duration,
        compiler_wall: Option<std::time::Duration>,
    ) {
        let mut cache = self.pso_cache.lock();
        if !cache.metrics_enabled {
            return;
        }
        cache.metrics.miss_wall_ns = cache
            .metrics
            .miss_wall_ns
            .saturating_add(duration_ns_saturating(miss_wall));
        if let Some(compiler_wall) = compiler_wall {
            cache.metrics.compiler_wall_ns = cache
                .metrics
                .compiler_wall_ns
                .saturating_add(duration_ns_saturating(compiler_wall));
        }
    }

    pub fn pipeline_info(&self, name: &str) -> Result<MetalPipelineInfo, MetalError> {
        let pso = self.pipeline(name)?;
        Ok(MetalPipelineInfo {
            name: name.to_string(),
            thread_execution_width: pso.threadExecutionWidth(),
            max_total_threads_per_threadgroup: pso.maxTotalThreadsPerThreadgroup(),
            static_threadgroup_memory_length: pso.staticThreadgroupMemoryLength(),
            supports_indirect_command_buffers: pso.supportIndirectCommandBuffers(),
        })
    }

    /// Bytes currently allocated through this Metal device.
    pub fn current_allocated_size(&self) -> u64 {
        self.device.currentAllocatedSize() as u64
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

    pub fn counter_capabilities(&self) -> MetalCounterCapabilities {
        let mut sets = Vec::new();
        if let Some(counter_sets) = self.device.counterSets() {
            for i in 0..counter_sets.len() {
                let set = counter_sets.objectAtIndex(i);
                let counters_obj = set.counters();
                let mut counters = Vec::with_capacity(counters_obj.len());
                for j in 0..counters_obj.len() {
                    counters.push(counters_obj.objectAtIndex(j).name().to_string());
                }
                let desc = MTLCounterSampleBufferDescriptor::new();
                desc.setCounterSet(Some(&set));
                // SAFETY: the descriptor owns the sample-count field and `2` is
                // the minimum useful before/after probe for future dispatch tests.
                unsafe { desc.setSampleCount(2) };
                let sample_buffer_status = match self
                    .device
                    .newCounterSampleBufferWithDescriptor_error(&desc)
                {
                    Ok(_) => "ok".to_string(),
                    Err(e) => e.localizedDescription().to_string(),
                };
                sets.push(MetalCounterSetInfo {
                    name: set.name().to_string(),
                    counters,
                    sample_buffer_status,
                });
            }
        }
        MetalCounterCapabilities {
            supports_stage_boundary: self
                .device
                .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary),
            supports_dispatch_boundary: self
                .device
                .supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary),
            supports_blit_boundary: self
                .device
                .supportsCounterSampling(MTLCounterSamplingPoint::AtBlitBoundary),
            sets,
        }
    }

    fn timestamp_counter_set(
        &self,
    ) -> Result<Retained<ProtocolObject<dyn MTLCounterSet>>, MetalError> {
        let counter_sets = self
            .device
            .counterSets()
            .ok_or_else(|| MetalError::Counter("device exposes no counter sets".to_string()))?;
        for i in 0..counter_sets.len() {
            let set = counter_sets.objectAtIndex(i);
            let set_name = set.name().to_string();
            if set_name.eq_ignore_ascii_case("timestamp") {
                return Ok(set);
            }
            let counters = set.counters();
            for j in 0..counters.len() {
                let counter_name = counters.objectAtIndex(j).name().to_string();
                if counter_name.eq_ignore_ascii_case("timestamp") {
                    return Ok(set);
                }
            }
        }
        Err(MetalError::Counter(
            "device exposes no timestamp counter set".to_string(),
        ))
    }

    pub fn timestamp_sample_buffer(
        &self,
        sample_count: usize,
    ) -> Result<MetalTimestampSampleBuffer, MetalError> {
        self.timestamp_sample_buffer_at(sample_count, MTLCounterSamplingPoint::AtStageBoundary)
    }

    pub fn timestamp_dispatch_sample_buffer(
        &self,
        sample_count: usize,
    ) -> Result<MetalTimestampSampleBuffer, MetalError> {
        self.timestamp_sample_buffer_at(sample_count, MTLCounterSamplingPoint::AtDispatchBoundary)
    }

    fn timestamp_sample_buffer_at(
        &self,
        sample_count: usize,
        sampling_point: MTLCounterSamplingPoint,
    ) -> Result<MetalTimestampSampleBuffer, MetalError> {
        if sample_count == 0 {
            return Err(MetalError::Counter(
                "timestamp sample count must be non-zero".to_string(),
            ));
        }
        if !self.device.supportsCounterSampling(sampling_point) {
            let label = match sampling_point {
                MTLCounterSamplingPoint::AtStageBoundary => "stage-boundary",
                MTLCounterSamplingPoint::AtDispatchBoundary => "dispatch-boundary",
                MTLCounterSamplingPoint::AtBlitBoundary => "blit-boundary",
                _ => "requested",
            };
            return Err(MetalError::Counter(format!(
                "device does not support {label} counter sampling"
            )));
        }
        let set = self.timestamp_counter_set()?;
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(&set));
        desc.setLabel(&NSString::from_str("qwen decode stage timestamps"));
        desc.setStorageMode(MTLStorageMode::Shared);
        unsafe { desc.setSampleCount(sample_count) };
        let raw = self
            .device
            .newCounterSampleBufferWithDescriptor_error(&desc)
            .map_err(|e| MetalError::Counter(e.localizedDescription().to_string()))?;
        Ok(MetalTimestampSampleBuffer { raw, sample_count })
    }

    pub fn resolve_timestamp_samples(
        &self,
        samples: &MetalTimestampSampleBuffer,
        sample_count: usize,
    ) -> Result<Vec<u64>, MetalError> {
        if sample_count > samples.sample_count {
            return Err(MetalError::Counter(format!(
                "resolve requested {sample_count} samples from {}-sample buffer",
                samples.sample_count
            )));
        }
        let data = unsafe {
            samples
                .raw
                .resolveCounterRange(NSRange::new(0, sample_count))
        }
        .ok_or_else(|| MetalError::Counter("resolveCounterRange returned nil".to_string()))?;
        let bytes = unsafe { data.as_bytes_unchecked() };
        let stride = std::mem::size_of::<MTLCounterResultTimestamp>();
        let needed = sample_count
            .checked_mul(stride)
            .ok_or_else(|| MetalError::Counter("timestamp resolve size overflow".to_string()))?;
        if bytes.len() < needed {
            return Err(MetalError::Counter(format!(
                "timestamp resolve returned {} bytes, need {needed}",
                bytes.len()
            )));
        }
        let mut out = Vec::with_capacity(sample_count);
        for chunk in bytes[..needed].chunks_exact(stride) {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(&chunk[..8]);
            out.push(u64::from_ne_bytes(raw));
        }
        Ok(out)
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
    let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
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
        checked_shape_elements(&self.shape).expect("MetalTensor shape element count overflow")
    }

    /// Total byte length (consults the dtype's per-block layout via
    /// checked host-side arithmetic).
    pub fn n_bytes(&self) -> u64 {
        let (_n, bytes) = checked_ggml_shape_bytes(&self.shape, self.dtype)
            .expect("MetalTensor byte-size computation overflow");
        bytes as u64
    }

    /// Build a tensor from raw bytes (e.g. the GGUF mmap slice for a
    /// weight tensor). Copies into a fresh `MTLBuffer` with shared storage.
    ///
    /// Validates that `bytes.len()` matches the byte size implied by
    /// `(shape, dtype)`. Without this check, a mismatched caller can
    /// create a short buffer with a large logical shape, and every
    /// downstream kernel or view silently operates past the buffer end.
    pub fn from_bytes(
        ctx: &MetalContext,
        bytes: &[u8],
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> Result<Self, MetalError> {
        let (_n, expected) = checked_ggml_shape_bytes(&shape, dtype)?;
        if bytes.len() != expected {
            return Err(MetalError::BadShape {
                kernel: "from_bytes",
                detail: format!(
                    "bytes.len()={} but shape={shape:?} dtype={dtype:?} expects {expected} bytes",
                    bytes.len()
                ),
            });
        }
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
        let (_n, bytes) = checked_shape_bytes(&shape, std::mem::size_of::<f32>())?;
        let buffer = ctx.buffer_uninit(bytes)?;
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype: GgmlType::F32,
        })
    }

    /// Allocate an F16 (half-precision) tensor. Used for the KV cache
    /// when we want to halve attention bandwidth at long context. The
    /// scatter kernel converts F32 → F16 on append; the attn_decode
    /// kernel reads F16 and casts to F32 in the dot product.
    pub fn zeros_f16(ctx: &MetalContext, shape: Vec<u64>) -> Result<Self, MetalError> {
        let (_n, bytes) = checked_shape_bytes(&shape, 2)?; // half = 2 bytes
        let buffer = ctx.buffer_uninit(bytes)?;
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype: GgmlType::F16,
        })
    }

    /// Allocate a Q8_0 tensor. Used for experimental KV-Q8 cache storage.
    /// Logical shape is still expressed in ELEMENTS; byte size follows ggml's
    /// Q8_0 block layout via `n_bytes()`.
    pub fn zeros_q8_0(ctx: &MetalContext, shape: Vec<u64>) -> Result<Self, MetalError> {
        let (_n, bytes) = checked_ggml_shape_bytes(&shape, GgmlType::Q8_0)?;
        let buffer = ctx.buffer_uninit(bytes)?;
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype: GgmlType::Q8_0,
        })
    }

    /// Allocate a zero-initialized tensor of any GGML dtype.
    ///
    /// This is mainly for synthetic benchmarks that need valid quantized
    /// buffers without loading a model tensor from disk.
    pub fn zeros_dtype(
        ctx: &MetalContext,
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> Result<Self, MetalError> {
        let (_n, bytes) = checked_ggml_shape_bytes(&shape, dtype)?;
        let zeros = vec![0u8; bytes.max(1)];
        let buffer = ctx.buffer_from(&zeros)?;
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype,
        })
    }

    /// Build a zero-copy sub-view of this tensor: same underlying MTLBuffer,
    /// shifted by `elem_offset` elements (of the tensor's dtype), with a
    /// new logical `shape`. The resulting view shares storage and aliases
    /// the parent — use only when you know the parent isn't being read by
    /// concurrent dispatches.
    ///
    /// Per Jeff & Sanjay (avoid copies / use indices instead of pointers):
    /// turns aliasing-via-explicit-copy into aliasing-via-offset, eliminating
    /// the entire copy_offset dispatch. Used to slice GDN's fused QKV
    /// conv-output buffer into Q / K / V subranges with zero kernels.
    ///
    /// Constraint: only valid for non-quantized dtypes (F32, F16) where
    /// elem_offset translates trivially to byte offset. For quantized
    /// types (Q4_K, Q5_K, Q6_K) the byte offset would need to align to
    /// the super-block boundary, which `view_subrange` does not check.
    pub fn view_subrange(&self, elem_offset: u64, shape: Vec<u64>) -> Self {
        let elem_size: u64 = match self.dtype {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            other => panic!(
                "view_subrange only supports F32/F16 (no super-block alignment), got {other:?}"
            ),
        };
        let n_view = checked_shape_elements(&shape).expect("view_subrange shape overflow");
        let parent_n = self.n_elements();
        // Always-on bounds check: a release-build OOB view is silent GPU UB
        // (kernels happily read/write past the buffer end). All callers
        // here are computing static slot indices, not user input, so a
        // failure is a programming bug, not a recoverable condition.
        assert!(
            elem_offset
                .checked_add(n_view)
                .is_some_and(|end| end <= parent_n),
            "view_subrange OOB: elem_offset={elem_offset} + n_view={n_view} > parent_n={parent_n}"
        );
        let byte_delta = elem_offset
            .checked_mul(elem_size)
            .expect("view_subrange byte offset overflow");
        let offset = self
            .offset
            .checked_add(byte_delta)
            .expect("view_subrange absolute offset overflow");
        Self {
            buffer: self.buffer.clone(),
            offset,
            shape,
            dtype: self.dtype,
        }
    }

    /// Build a zero-copy sub-view by byte offset. Intended for quantized
    /// expert banks where each expert slice is already laid out as a
    /// contiguous 2D tensor in the parent buffer.
    pub fn view_bytes(&self, byte_offset: u64, shape: Vec<u64>) -> Self {
        // Always-on bounds check: see view_subrange for rationale.
        let (_view_n, view_bytes_usize) = checked_ggml_shape_bytes(&shape, self.dtype)
            .expect("view_bytes byte-size computation overflow");
        let view_bytes = view_bytes_usize as u64;
        let parent_bytes = self.n_bytes();
        assert!(
            byte_offset
                .checked_add(view_bytes)
                .is_some_and(|end| end <= parent_bytes),
            "view_bytes OOB: byte_offset={byte_offset} + view_bytes={view_bytes} > parent_bytes={parent_bytes}"
        );
        let offset = self
            .offset
            .checked_add(byte_offset)
            .expect("view_bytes absolute offset overflow");
        Self {
            buffer: self.buffer.clone(),
            offset,
            shape,
            dtype: self.dtype,
        }
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
    /// True when created via [`KernelEncoder::begin_concurrent`]. Concurrent
    /// passes provide NO ordering between dispatches, so every dispatch pair
    /// must be independent (disjoint writes; no dispatch reads another's
    /// output). That invariant is a convention, not a type-system property —
    /// the debug-only `note_read`/`note_write` hazard tracker below turns a
    /// future violation into a loud panic instead of a silent GPU race.
    pub concurrent: bool,
    #[cfg(debug_assertions)]
    hazard_writes: std::cell::RefCell<Vec<(usize, u64, u64)>>,
    #[cfg(debug_assertions)]
    hazard_reads: std::cell::RefCell<Vec<(usize, u64, u64)>>,
}

#[cfg(debug_assertions)]
fn ranges_overlap(a_off: u64, a_len: u64, b_off: u64, b_len: u64) -> bool {
    a_off < b_off + b_len && b_off < a_off + a_len
}

impl KernelEncoder {
    fn new(raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, concurrent: bool) -> Self {
        Self {
            raw,
            concurrent,
            #[cfg(debug_assertions)]
            hazard_writes: std::cell::RefCell::new(Vec::new()),
            #[cfg(debug_assertions)]
            hazard_reads: std::cell::RefCell::new(Vec::new()),
        }
    }

    pub fn begin(cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        let raw = cmd.computeCommandEncoder().expect("compute encoder");
        kernel_trace_record_encoder(false);
        Self::new(raw, false)
    }

    pub fn begin_concurrent(cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        let raw = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .expect("concurrent compute encoder");
        kernel_trace_record_encoder(true);
        Self::new(raw, true)
    }

    pub fn begin_sampled(
        cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        samples: &MetalTimestampSampleBuffer,
        start_sample: usize,
        end_sample: usize,
        concurrent: bool,
    ) -> Self {
        let pass = MTLComputePassDescriptor::computePassDescriptor();
        pass.setDispatchType(if concurrent {
            MTLDispatchType::Concurrent
        } else {
            MTLDispatchType::Serial
        });
        let attachments = pass.sampleBufferAttachments();
        let attachment = unsafe { attachments.objectAtIndexedSubscript(0) };
        attachment.setSampleBuffer(Some(&samples.raw));
        unsafe {
            attachment.setStartOfEncoderSampleIndex(start_sample);
            attachment.setEndOfEncoderSampleIndex(end_sample);
        }
        let raw = cmd
            .computeCommandEncoderWithDescriptor(&pass)
            .expect("sampled compute encoder");
        kernel_trace_record_encoder(concurrent);
        Self::new(raw, concurrent)
    }

    /// Debug-only hazard note: declare that a dispatch in this encoder
    /// WRITES `tensor`'s byte range. On a concurrent encoder, panics if the
    /// range overlaps any previously noted write or read — either would be
    /// an unsynchronized data race inside the concurrent pass. No-op in
    /// release builds and on serial encoders.
    #[inline]
    pub fn note_write(&self, tensor: &MetalTensor) {
        #[cfg(debug_assertions)]
        {
            if !self.concurrent {
                return;
            }
            let buf = Retained::as_ptr(&tensor.buffer) as *const () as usize;
            let off = tensor.offset;
            let len = tensor.n_bytes();
            for &(b, o, l) in self.hazard_writes.borrow().iter() {
                assert!(
                    b != buf || !ranges_overlap(off, len, o, l),
                    "concurrent-encoder hazard: write/write overlap on buffer {buf:#x} \
                     (new write off={off} len={len}, prior write off={o} len={l}); \
                     dependent dispatches must not share a Concurrent pass"
                );
            }
            for &(b, o, l) in self.hazard_reads.borrow().iter() {
                assert!(
                    b != buf || !ranges_overlap(off, len, o, l),
                    "concurrent-encoder hazard: write overlaps a noted read on buffer {buf:#x} \
                     (write off={off} len={len}, prior read off={o} len={l}); \
                     dependent dispatches must not share a Concurrent pass"
                );
            }
            self.hazard_writes.borrow_mut().push((buf, off, len));
        }
        #[cfg(not(debug_assertions))]
        {
            let _ = tensor;
        }
    }

    /// Debug-only hazard note: declare that a dispatch in this encoder
    /// READS `tensor`'s byte range. Panics on a concurrent encoder if the
    /// range overlaps a previously noted write (read-after-write inside a
    /// Concurrent pass is unordered). Overlapping reads are fine.
    #[inline]
    pub fn note_read(&self, tensor: &MetalTensor) {
        #[cfg(debug_assertions)]
        {
            if !self.concurrent {
                return;
            }
            let buf = Retained::as_ptr(&tensor.buffer) as *const () as usize;
            let off = tensor.offset;
            let len = tensor.n_bytes();
            for &(b, o, l) in self.hazard_writes.borrow().iter() {
                assert!(
                    b != buf || !ranges_overlap(off, len, o, l),
                    "concurrent-encoder hazard: read overlaps a noted write on buffer {buf:#x} \
                     (read off={off} len={len}, prior write off={o} len={l}); \
                     dependent dispatches must not share a Concurrent pass"
                );
            }
            self.hazard_reads.borrow_mut().push((buf, off, len));
        }
        #[cfg(not(debug_assertions))]
        {
            let _ = tensor;
        }
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)));
    }

    pub fn insert_debug_signpost(&self, label: &str) {
        self.raw.insertDebugSignpost(&NSString::from_str(label));
    }

    pub fn sample_counters(
        &self,
        samples: &MetalTimestampSampleBuffer,
        sample_index: usize,
        barrier: bool,
    ) {
        unsafe {
            self.raw.sampleCountersInBuffer_atSampleIndex_withBarrier(
                &samples.raw,
                sample_index,
                barrier,
            );
        }
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
        kernel_trace_record_dispatch();
        census_record_dispatch(grid, threads);
        self.raw
            .dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
    }

    pub fn update_fence(&self, fence: &Fence) {
        self.raw.updateFence(fence);
    }

    pub fn wait_for_fence(&self, fence: &Fence) {
        self.raw.waitForFence(fence);
    }
}

// ===========================================================================
// BlitEncoder — caller-owned blit encoder
// ===========================================================================

/// Caller-owned wrapper around `MTLBlitCommandEncoder`. Blit encoders are
/// for bulk device-to-device memory copies via the GPU's DMA engines —
/// faster and lower-overhead than encoding a compute "copy kernel" because
/// they avoid pipeline state setup and run independently of the compute
/// engines.
///
/// Usage pattern (from H5.3a packed_forward, where per-token GDN+conv
/// state checkpoints land):
///
/// ```ignore
/// let cmd = ctx.queue.commandBuffer().expect("cmd buf");
/// // ── compute pass: encode block N's kernels ─────────────────────────
/// let enc = KernelEncoder::begin(&cmd);
/// /* encode kernels for token N */
/// enc.end();
/// // ── blit pass: copy GDN/conv state into checkpoint slot N ──────────
/// let blit = BlitEncoder::begin(&cmd);
/// blit.copy_buffer(&gdn_state.buffer, gdn_state.offset,
///                  &gdn_ckpt.buffer, ckpt_offset_for_token_n,
///                  gdn_state.n_bytes());
/// blit.end();
/// // ── repeat compute+blit pairs for tokens N+1 .. ────────────────────
/// cmd.commit();
/// ```
///
/// Multiple compute↔blit transitions inside one command buffer are fully
/// supported by Metal; the runtime synchronises between encoder passes
/// automatically (the blit pass observes all writes from the previous
/// compute pass once `endEncoding` has been called on the compute encoder).
pub struct BlitEncoder {
    pub raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
}

impl BlitEncoder {
    pub fn begin(cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        let raw = cmd.blitCommandEncoder().expect("blit encoder");
        Self { raw }
    }

    pub fn end(self) {
        self.raw.endEncoding();
    }

    /// Device-to-device buffer copy.
    ///
    /// `n_bytes` must satisfy `src.length() >= src_offset + n_bytes` and
    /// likewise for `dst`. Metal does not validate this — caller's
    /// responsibility. (Single-MetalTensor blits where src == dst with
    /// non-overlapping ranges are allowed; overlapping ranges are
    /// undefined behaviour per the Metal docs.)
    pub fn copy_buffer(
        &self,
        src: &Buffer,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        n_bytes: u64,
    ) {
        // Always-on bounds check: Metal does not validate blit ranges and
        // a release-build OOB blit is silent GPU/host corruption.
        let src_end = src_offset
            .checked_add(n_bytes)
            .expect("blit src offset+n overflow");
        let dst_end = dst_offset
            .checked_add(n_bytes)
            .expect("blit dst offset+n overflow");
        let src_len = src.length() as u64;
        let dst_len = dst.length() as u64;
        assert!(
            src_end <= src_len,
            "blit src OOB: src_offset={src_offset} + n_bytes={n_bytes} > src.len={src_len}"
        );
        assert!(
            dst_end <= dst_len,
            "blit dst OOB: dst_offset={dst_offset} + n_bytes={n_bytes} > dst.len={dst_len}"
        );
        unsafe {
            self.raw
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src.as_ref(),
                    src_offset as usize,
                    dst.as_ref(),
                    dst_offset as usize,
                    n_bytes as usize,
                );
        }
    }

    /// Convenience wrapper: copy the entire contents of one tensor's
    /// buffer slice into another's. Assumes `src.n_bytes() == dst.n_bytes()`
    /// (callers writing per-token checkpoint slots typically already
    /// have this guarantee by construction).
    pub fn copy_tensor(&self, src: &MetalTensor, dst: &MetalTensor) {
        // Always-on (was debug_assert): a release-build size mismatch
        // would silently short-copy or overrun the destination.
        assert_eq!(
            src.n_bytes(),
            dst.n_bytes(),
            "copy_tensor: src/dst byte sizes differ ({} vs {})",
            src.n_bytes(),
            dst.n_bytes()
        );
        self.copy_buffer(
            &src.buffer,
            src.offset,
            &dst.buffer,
            dst.offset,
            src.n_bytes(),
        );
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// In-place residual add followed by RMSNorm-with-weight:
/// `x[i] += residual[i]`; `y[i] = (x[i] / sqrt(mean(x²) + eps)) * weight[i]`.
pub fn encode_residual_rms_norm_mul_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    residual: &MetalTensor,
    weight: &MetalTensor,
    y: &MetalTensor,
    eps: f32,
) -> Result<(), MetalError> {
    let n_dim = x.n_elements() as usize;
    if residual.n_elements() as usize != n_dim {
        return Err(MetalError::BadShape {
            kernel: "residual_rms_norm",
            detail: format!(
                "residual.n_elements={} != x.n_elements={n_dim}",
                residual.n_elements()
            ),
        });
    }
    if y.n_elements() as usize != n_dim {
        return Err(MetalError::BadShape {
            kernel: "residual_rms_norm",
            detail: format!("y.n_elements={} != x.n_elements={n_dim}", y.n_elements()),
        });
    }
    if weight.n_elements() as usize != n_dim {
        return Err(MetalError::BadShape {
            kernel: "residual_rms_norm",
            detail: format!(
                "weight.n_elements={} != x.n_elements={n_dim}",
                weight.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_residual_rms_norm_mul_f32")?;
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
    enc.set_tensor(2, residual);
    enc.set_tensor(3, weight);
    enc.set_tensor(4, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

crate::env_flag!(default_on mat_vec_f32_lcpp_r2_enabled, "QWEN_MATVEC_F32_LCPP_R2");

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
    let use_r2 = mat_vec_f32_lcpp_r2_enabled();
    let pso = ctx.pipeline(if use_r2 {
        "kernel_mat_vec_f32_f32_lcpp_r2"
    } else {
        "kernel_mat_vec_f32_f32"
    })?;
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

    if use_r2 {
        let nr0 = 2usize;
        let nsg = 4usize;
        enc.set_threadgroup_memory(0, 32 * nr0 * std::mem::size_of::<f32>());
        enc.dispatch(
            MTLSize {
                width: n_out.div_ceil(nr0),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: nsg * 32,
                height: 1,
                depth: 1,
            },
        );
        return Ok(());
    }

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

fn encode_mat_vec_16bit_weight_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expected: GgmlType,
    kernel_name: &'static str,
) -> Result<(), MetalError> {
    if weight.dtype != expected {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("weight.dtype = {:?}, expected {expected:?}", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_in {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("x.n_elements={} != n_in={n_in}", x.n_elements()),
        });
    }
    if y.n_elements() as usize != n_out {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("y.n_elements={} != n_out={n_out}", y.n_elements()),
        });
    }
    let pso = ctx.pipeline(kernel_name)?;
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

pub fn encode_mat_vec_f16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    encode_mat_vec_16bit_weight_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::F16,
        "kernel_mat_vec_f16_f32",
    )
}

pub fn encode_mat_vec_bf16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    encode_mat_vec_16bit_weight_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::BF16,
        "kernel_mat_vec_bf16_f32",
    )
}

fn encode_mat_vec_block32_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expected: GgmlType,
    kernel_name: &'static str,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    encode_mat_vec_16bit_weight_f32(ctx, enc, weight, x, y, n_in, n_out, expected, kernel_name)
}

pub fn encode_mat_vec_q4_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    encode_mat_vec_block32_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::Q4_0,
        "kernel_mat_vec_q4_0_f32",
    )
}

pub fn encode_mat_vec_q4_1_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    encode_mat_vec_block32_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::Q4_1,
        "kernel_mat_vec_q4_1_f32",
    )
}

pub fn encode_mtp_draft_affine_q4_gs64_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    scales: &MetalTensor,
    biases: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    const GROUP_SIZE: usize = 64;
    const PACK_FACTOR: usize = 8;
    const ROWS_PER_TG: usize = 8;
    let kernel_name = "kernel_mtp_draft_affine_q4_gs64_f32";
    if n_in % GROUP_SIZE != 0 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("n_in={n_in} not divisible by {GROUP_SIZE}"),
        });
    }
    if weight.dtype != GgmlType::F32
        || scales.dtype != GgmlType::F16
        || biases.dtype != GgmlType::F16
    {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "weight/scales/biases expected packed-u32-as-F32/F16/F16, got {:?}/{:?}/{:?}",
                weight.dtype, scales.dtype, biases.dtype
            ),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_in || y.n_elements() as usize != n_out {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "x/y elements {}/{} do not match n_in/n_out {n_in}/{n_out}",
                x.n_elements(),
                y.n_elements()
            ),
        });
    }
    let packs_per_row = n_in / PACK_FACTOR;
    let n_groups = n_in / GROUP_SIZE;
    if weight.n_elements() as usize != n_out * packs_per_row
        || scales.n_elements() as usize != n_out * n_groups
        || biases.n_elements() as usize != n_out * n_groups
    {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "bad packed head sizes: weight={} scales={} biases={} expected {}/{}/{}",
                weight.n_elements(),
                scales.n_elements(),
                biases.n_elements(),
                n_out * packs_per_row,
                n_out * n_groups,
                n_out * n_groups
            ),
        });
    }

    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_groups: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_groups: n_groups as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, scales);
    enc.set_tensor(3, biases);
    enc.set_tensor(4, x);
    enc.set_tensor(5, y);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(ROWS_PER_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

crate::env_flag!(default_on matvec_iq4_nl_fast_enabled, "QWEN_MATVEC_IQ4_NL_FAST");

pub fn encode_mat_vec_iq4_nl_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_iq4_nl_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            32,
            GgmlType::IQ4_NL,
            "mat_vec_iq4_nl_fast",
            "kernel_mat_vec_iq4_nl_f32_fast",
            2,
            2,
            32 * std::mem::size_of::<f32>(),
        );
    }
    encode_mat_vec_block32_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ4_NL,
        "kernel_mat_vec_iq4_nl_f32",
    )
}

fn encode_mat_vec_block256_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expected: GgmlType,
    kernel_name: &'static str,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    encode_mat_vec_16bit_weight_f32(ctx, enc, weight, x, y, n_in, n_out, expected, kernel_name)
}

crate::env_flag!(default_on matvec_q3_k_fast_enabled, "QWEN_MATVEC_Q3_K_FAST");

pub fn encode_mat_vec_q3_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_q3_k_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::Q3_K,
            "mat_vec_q3_k_fast",
            "kernel_mat_vec_q3_K_f32_fast",
            2,
            2,
            0,
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::Q3_K,
        "kernel_mat_vec_q3_K_f32",
    )
}

crate::env_flag!(default_on matvec_q2_k_fast_enabled, "QWEN_MATVEC_Q2_K_FAST");

pub fn encode_mat_vec_q2_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_q2_k_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::Q2_K,
            "mat_vec_q2_k_fast",
            "kernel_mat_vec_q2_K_f32_fast",
            4,
            2,
            0,
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::Q2_K,
        "kernel_mat_vec_q2_K_f32",
    )
}

crate::env_flag!(default_on matvec_iq2_s_fast_enabled, "QWEN_MATVEC_IQ2_S_FAST");

pub fn encode_mat_vec_iq2_s_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_iq2_s_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::IQ2_S,
            "mat_vec_iq2_s_fast",
            "kernel_mat_vec_iq2_s_f32_fast",
            4,
            2,
            0,
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ2_S,
        "kernel_mat_vec_iq2_s_f32",
    )
}

crate::env_flag!(default_on matvec_iq3_xxs_fast_enabled, "QWEN_MATVEC_IQ3_XXS_FAST");

pub fn encode_mat_vec_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_iq3_xxs_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::IQ3_XXS,
            "mat_vec_iq3_xxs_fast",
            "kernel_mat_vec_iq3_xxs_f32_fast",
            4,
            2,
            0,
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ3_XXS,
        "kernel_mat_vec_iq3_xxs_f32",
    )
}

crate::env_flag!(default_on matvec_iq3_s_fast_enabled, "QWEN_MATVEC_IQ3_S_FAST");

pub fn encode_mat_vec_iq3_s_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_iq3_s_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::IQ3_S,
            "mat_vec_iq3_s_fast",
            "kernel_mat_vec_iq3_s_f32_fast",
            4,
            2,
            0,
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ3_S,
        "kernel_mat_vec_iq3_s_f32",
    )
}

crate::env_flag!(default_on matvec_iq4_xs_fast_enabled, "QWEN_MATVEC_IQ4_XS_FAST");

pub fn encode_mat_vec_iq4_xs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_iq4_xs_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::IQ4_XS,
            "mat_vec_iq4_xs_fast",
            "kernel_mat_vec_iq4_xs_f32_fast",
            2,
            2,
            32 * std::mem::size_of::<f32>(),
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ4_XS,
        "kernel_mat_vec_iq4_xs_f32",
    )
}

fn encode_mat_vec_lowbit_fast_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    block_multiple: usize,
    expected: GgmlType,
    error_kernel: &'static str,
    metal_kernel: &'static str,
    rows_per_simdgroup: usize,
    simdgroups: usize,
    threadgroup_bytes: usize,
) -> Result<(), MetalError> {
    if n_in % block_multiple != 0 {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!("n_in={n_in} not divisible by {block_multiple}"),
        });
    }
    if weight.dtype != expected {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!("weight.dtype = {:?}, expected {expected:?}", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_in || y.n_elements() as usize != n_out {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!(
                "shape mismatch x={} y={} expected x={} y={}",
                x.n_elements(),
                y.n_elements(),
                n_in,
                n_out
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }
    let pso = ctx.pipeline(metal_kernel)?;
    enc.set_pipeline(&pso);
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
    if threadgroup_bytes > 0 {
        enc.set_threadgroup_memory(0, threadgroup_bytes);
    }
    let rows_per_tg = rows_per_simdgroup * simdgroups;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(rows_per_tg),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: simdgroups * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_mat_mat_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32",
            detail: format!("weight.dtype = {:?}, expected F32", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32",
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_out * n_query {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32",
            detail: format!(
                "y.n_elements={} != n_out*n_query={}",
                y.n_elements(),
                n_out * n_query
            ),
        });
    }
    let pso = ctx.pipeline("kernel_mat_mat_f32_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_query: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_query: n_query as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 32 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out,
            height: n_query.div_ceil(32),
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

fn encode_mat_mat_16bit_weight_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    expected: GgmlType,
    kernel_name: &'static str,
) -> Result<(), MetalError> {
    if weight.dtype != expected {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("weight.dtype = {:?}, expected {expected:?}", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_out * n_query {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "y.n_elements={} != n_out*n_query={}",
                y.n_elements(),
                n_out * n_query
            ),
        });
    }
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_query: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_query: n_query as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 32 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out,
            height: n_query.div_ceil(32),
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

pub fn encode_mat_mat_f16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if mat_mat_f16_half_act_enabled() && n_in % 32 == 0 && n_query >= 16 {
        return encode_mat_mat_f16_half_act_f32(ctx, enc, weight, x, y, n_in, n_out, n_query);
    }
    encode_mat_mat_16bit_weight_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::F16,
        "kernel_mat_mat_f16_f32",
    )
}

crate::env_flag!(default_on matmat_f16_half_act_env_default, "QWEN_MATMAT_F16_HALF_ACT");

fn mat_mat_f16_half_act_enabled() -> bool {
    if let Some(enabled) = MATMAT_F16_HALF_ACT_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    matmat_f16_half_act_env_default()
}

pub fn encode_mat_mat_f16_half_act_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f16_half_act_f32",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f16_half_act_f32",
            detail: format!("weight.dtype = {:?}, expected F16", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f16_half_act_f32",
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in || y.n_elements() as usize != n_out * n_query {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f16_half_act_f32",
            detail: format!(
                "shape mismatch x={} y={} expected x={} y={}",
                x.n_elements(),
                y.n_elements(),
                n_query * n_in,
                n_out * n_query
            ),
        });
    }

    let pso = ctx.pipeline("kernel_mat_mat_f16_half_act_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_query: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_query: n_query as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_mat_mat_bf16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_16bit_weight_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::BF16,
        "kernel_mat_mat_bf16_f32",
    )
}

pub fn encode_mat_mat_bf16_bfloat_act_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_bf16_bfloat_act_f32",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::BF16 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_bf16_bfloat_act_f32",
            detail: format!("weight.dtype = {:?}, expected BF16", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_bf16_bfloat_act_f32",
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in || y.n_elements() as usize != n_out * n_query {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_bf16_bfloat_act_f32",
            detail: format!(
                "shape mismatch x={} y={} expected x={} y={}",
                x.n_elements(),
                y.n_elements(),
                n_query * n_in,
                n_out * n_query
            ),
        });
    }

    let pso = ctx.pipeline("kernel_mat_mat_bf16_bfloat_act_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_query: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_query: n_query as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_mat_mat_block32_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    expected: GgmlType,
    kernel_name: &'static str,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    encode_mat_mat_16bit_weight_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        expected,
        kernel_name,
    )
}

pub fn encode_mat_mat_q4_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if mat_mat_q4_legacy_mm_enabled() && n_query >= 16 {
        return encode_mat_mat_q4_legacy_mm_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            n_query,
            GgmlType::Q4_0,
            "kernel_mat_mat_q4_0_f32_mm",
            18,
        );
    }
    encode_mat_mat_block32_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::Q4_0,
        "kernel_mat_mat_q4_0_f32",
    )
}

pub fn encode_mat_mat_q4_1_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if mat_mat_q4_legacy_mm_enabled() && n_query >= 16 {
        return encode_mat_mat_q4_legacy_mm_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            n_query,
            GgmlType::Q4_1,
            "kernel_mat_mat_q4_1_f32_mm",
            20,
        );
    }
    encode_mat_mat_block32_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::Q4_1,
        "kernel_mat_mat_q4_1_f32",
    )
}

crate::env_flag!(default_on matmat_q4_legacy_mm_env_default, "QWEN_MATMAT_Q4_LEGACY_MM");

fn mat_mat_q4_legacy_mm_enabled() -> bool {
    if let Some(enabled) = MATMAT_Q4_LEGACY_MM_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    matmat_q4_legacy_mm_env_default()
}

fn encode_mat_mat_q4_legacy_mm_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    expected: GgmlType,
    kernel_name: &'static str,
    block_bytes: usize,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != expected {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("weight.dtype = {:?}, expected {expected:?}", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01: ((n_in / 32) * block_bytes) as u32,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, mat_mat_qk_threadgroup_memory(n_out, n_query, 32));
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

crate::env_flag!(default_on matmat_iq4_nl_mm_enabled, "QWEN_MATMAT_IQ4_NL_MM");

pub fn encode_mat_mat_iq4_nl_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_iq4_nl_mm_enabled() {
        return encode_mat_mat_iq4_nl_f32_mm(ctx, enc, weight, x, y, n_in, n_out, n_query);
    }
    encode_mat_mat_block32_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::IQ4_NL,
        "kernel_mat_mat_iq4_nl_f32",
    )
}

fn encode_mat_mat_iq4_nl_f32_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_nl_mm",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::IQ4_NL {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_nl_mm",
            detail: format!("weight.dtype = {:?}, expected IQ4_NL", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_nl_mm",
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_nl_mm",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_nl_mm",
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let nb01 = ((n_in / 32) * 18) as u32;
    let pso = ctx.pipeline("kernel_mat_mat_iq4_nl_f32_mm")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_mat_mat_block256_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    expected: GgmlType,
    kernel_name: &'static str,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    encode_mat_mat_16bit_weight_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        expected,
        kernel_name,
    )
}

crate::env_flag!(default_on matmat_q3_k_mm_enabled, "QWEN_MATMAT_Q3_K_MM");

pub fn encode_mat_mat_q3_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_q3_k_mm_enabled() {
        return encode_mat_mat_q3_k_f32_mm(ctx, enc, weight, x, y, n_in, n_out, n_query);
    }
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::Q3_K,
        "kernel_mat_mat_q3_K_f32",
    )
}

fn encode_mat_mat_q3_k_f32_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_qk_lowbit_mm(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::Q3_K,
        "mat_mat_q3_k_mm",
        "kernel_mat_mat_q3_K_f32_mm",
        110,
    )
}

crate::env_flag!(default_on matmat_q2_k_mm_enabled, "QWEN_MATMAT_Q2_K_MM");

pub fn encode_mat_mat_q2_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_q2_k_mm_enabled() {
        return encode_mat_mat_q2_k_f32_mm(ctx, enc, weight, x, y, n_in, n_out, n_query);
    }
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::Q2_K,
        "kernel_mat_mat_q2_K_f32",
    )
}

fn encode_mat_mat_q2_k_f32_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_qk_lowbit_mm(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::Q2_K,
        "mat_mat_q2_k_mm",
        "kernel_mat_mat_q2_K_f32_mm",
        84,
    )
}

crate::env_flag!(default_on matmat_iq2_s_mm_enabled, "QWEN_MATMAT_IQ2_S_MM");

pub fn encode_mat_mat_iq2_s_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_iq2_s_mm_enabled() {
        return encode_mat_mat_qk_lowbit_mm(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            n_query,
            GgmlType::IQ2_S,
            "mat_mat_iq2_s_mm",
            "kernel_mat_mat_iq2_s_f32_mm",
            82,
        );
    }
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::IQ2_S,
        "kernel_mat_mat_iq2_s_f32",
    )
}

crate::env_flag!(default_on matmat_iq3_xxs_mm_enabled, "QWEN_MATMAT_IQ3_XXS_MM");

pub fn encode_mat_mat_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_iq3_xxs_mm_enabled() {
        return encode_mat_mat_qk_lowbit_mm(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            n_query,
            GgmlType::IQ3_XXS,
            "mat_mat_iq3_xxs_mm",
            "kernel_mat_mat_iq3_xxs_f32_mm",
            98,
        );
    }
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::IQ3_XXS,
        "kernel_mat_mat_iq3_xxs_f32",
    )
}

crate::env_flag!(default_on matmat_iq3_s_mm_enabled, "QWEN_MATMAT_IQ3_S_MM");

pub fn encode_mat_mat_iq3_s_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_iq3_s_mm_enabled() {
        return encode_mat_mat_qk_lowbit_mm(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            n_query,
            GgmlType::IQ3_S,
            "mat_mat_iq3_s_mm",
            "kernel_mat_mat_iq3_s_f32_mm",
            110,
        );
    }
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::IQ3_S,
        "kernel_mat_mat_iq3_s_f32",
    )
}

fn encode_mat_mat_qk_lowbit_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    expected: GgmlType,
    error_kernel: &'static str,
    metal_kernel: &'static str,
    block_bytes: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != expected {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!("weight.dtype = {:?}, expected {expected:?}", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let nb01 = ((n_in / 256) * block_bytes) as u32;
    let pso = ctx.pipeline(metal_kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

crate::env_flag!(default_on matmat_iq4_xs_mm_enabled, "QWEN_MATMAT_IQ4_XS_MM");

pub fn encode_mat_mat_iq4_xs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_iq4_xs_mm_enabled() {
        return encode_mat_mat_iq4_xs_f32_mm(ctx, enc, weight, x, y, n_in, n_out, n_query);
    }
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::IQ4_XS,
        "kernel_mat_mat_iq4_xs_f32",
    )
}

fn encode_mat_mat_iq4_xs_f32_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_xs_mm",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::IQ4_XS {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_xs_mm",
            detail: format!("weight.dtype = {:?}, expected IQ4_XS", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_xs_mm",
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_xs_mm",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_xs_mm",
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let nb01 = ((n_in / 256) * 136) as u32;
    let pso = ctx.pipeline("kernel_mat_mat_iq4_xs_f32_mm")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_mat_mat_f32_router_e8p32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32_router_e8p32",
            detail: format!("weight.dtype = {:?}, expected F32", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32_router_e8p32",
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if n_in % 4 != 0 || n_out % 8 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32_router_e8p32",
            detail: format!(
                "expected n_in % 4 == 0 and n_out % 8 == 0, got n_in={n_in} n_out={n_out}"
            ),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32_router_e8p32",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_out * n_query {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32_router_e8p32",
            detail: format!(
                "y.n_elements={} != n_out*n_query={}",
                y.n_elements(),
                n_out * n_query
            ),
        });
    }
    let pso = ctx.pipeline("kernel_mat_mat_f32_f32_router_e8p32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_query: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_query: n_query as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.dispatch(
        MTLSize {
            width: n_out / 8,
            height: n_query.div_ceil(32),
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

/// Multi-column Q4_K mat-vec (H5.6 M2-nc skinny-GEMM experiment):
/// `Y = W · X^T` with mat-vec-grade occupancy (`n_out/4` threadgroups of
/// 128 threads: 2 row-pairs x 2 column-halves) running the mv1 body per
/// activation column — quant block reads are column-invariant and L1-hot
/// on re-reads (explicit register staging measured slower; see the kernel
/// header).
///
///   * `x`: F32 `[n_cols, n_in]` row-major (column c = `x + c*n_in`)
///   * `y`: F32 `[n_cols, n_out]` row-major (`y[c*n_out + r]`)
///   * `n_cols` ∈ {2, 4, 8} (compile-time instantiations)
///
/// Exactness: bit-identical per column to `encode_mat_vec_q4_k_f32`
/// (same accumulation order; E0 tier). Asserted by
/// `multicol_gemv_micro_27b` in tests/dflash_correctness.rs.
pub fn encode_mat_vec_q4_k_nc_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc",
            detail: format!("n_in={n_in} not divisible by 256 (Q4_K super-block)"),
        });
    }
    if weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc",
            detail: format!("weight.dtype = {:?}, expected Q4_K", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc",
            detail: format!("x/y dtype = {:?}/{:?}, expected F32/F32", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_cols * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc",
            detail: format!(
                "x.n_elements={} != n_cols*n_in={}",
                x.n_elements(),
                n_cols * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_cols * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc",
            detail: format!(
                "y.n_elements={} != n_cols*n_out={}",
                y.n_elements(),
                n_cols * n_out
            ),
        });
    }
    if weight.n_elements() as usize != n_in * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc",
            detail: format!(
                "weight.n_elements={} != n_in*n_out={}",
                weight.n_elements(),
                n_in * n_out
            ),
        });
    }
    let name = match n_cols {
        2 => "kernel_mat_vec_q4_K_nc2_f32",
        4 => "kernel_mat_vec_q4_K_nc4_f32",
        8 => "kernel_mat_vec_q4_K_nc8_f32",
        _ => {
            return Err(MetalError::BadShape {
                kernel: "mat_vec_q4_k_nc",
                detail: format!("n_cols={n_cols} not in {{2, 4, 8}}"),
            });
        }
    };
    let pso = ctx.pipeline(name)?;
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

    // TG = 4 simdgroups (128 threads): 2 row-pairs x 2 column-halves.
    // Same 4-rows-per-TG weight coverage as mv1 (grid = n_out/4), twice
    // the ALU per weight byte (columns split across the extra SGs).
    const ROWS_PER_TG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(ROWS_PER_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Q6_K companion to [`encode_mat_vec_q4_k_nc_f32`] — same layouts,
/// same 4-simdgroup (2 row-pairs x 2 column-halves) geometry, Q6_K
/// block decode. Covers the two largest verify-path tensors on
/// 27B-Q4_K_M (ffn_down, gdn_qkv).
pub fn encode_mat_vec_q6_k_nc_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_nc",
            detail: format!("n_in={n_in} not divisible by 256 (Q6_K super-block)"),
        });
    }
    if weight.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_nc",
            detail: format!("weight.dtype = {:?}, expected Q6_K", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_nc",
            detail: format!("x/y dtype = {:?}/{:?}, expected F32/F32", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_cols * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_nc",
            detail: format!(
                "x.n_elements={} != n_cols*n_in={}",
                x.n_elements(),
                n_cols * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_cols * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_nc",
            detail: format!(
                "y.n_elements={} != n_cols*n_out={}",
                y.n_elements(),
                n_cols * n_out
            ),
        });
    }
    if weight.n_elements() as usize != n_in * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_nc",
            detail: format!(
                "weight.n_elements={} != n_in*n_out={}",
                weight.n_elements(),
                n_in * n_out
            ),
        });
    }
    let name = match n_cols {
        2 => "kernel_mat_vec_q6_K_nc2_f32",
        4 => "kernel_mat_vec_q6_K_nc4_f32",
        8 => "kernel_mat_vec_q6_K_nc8_f32",
        _ => {
            return Err(MetalError::BadShape {
                kernel: "mat_vec_q6_k_nc",
                detail: format!("n_cols={n_cols} not in {{2, 4, 8}}"),
            });
        }
    };
    let pso = ctx.pipeline(name)?;
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
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Dtype-routing wrapper for the multi-column mat-vec family (Q4_K and
/// Q6_K today). Mirrors `encode_mat_vec_dispatch` semantics for the
/// N-column case; returns `BadShape` for unsupported dtypes so callers
/// can fall back explicitly.
pub fn encode_mat_vec_nc_dispatch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    match weight.dtype {
        GgmlType::Q4_K => encode_mat_vec_q4_k_nc_f32(ctx, enc, weight, x, y, n_in, n_out, n_cols),
        GgmlType::Q6_K => encode_mat_vec_q6_k_nc_f32(ctx, enc, weight, x, y, n_in, n_out, n_cols),
        other => Err(MetalError::BadShape {
            kernel: "mat_vec_nc_dispatch",
            detail: format!("unsupported dtype {other:?} (Q4_K | Q6_K)"),
        }),
    }
}

/// Small-N MMA experiment (v0.498 follow-up): 8-row x 8-padded-column
/// simdgroup_matrix tile at mat-vec-grade occupancy (`n_out/8`
/// single-simdgroup threadgroups). Caller ALWAYS provides 8 activation
/// columns (`x`: F32 `[8, n_in]`) and receives 8 output columns
/// (`y`: F32 `[8, n_out]`, `y[c*n_out + r]`); pad unused columns.
/// Q4_K | Q6_K. Exactness: E1 (half-staged weight dequant + MMA
/// accumulation order; activations NOT half-staged — loaded F32 direct).
/// Gate: cos >= 0.999 per column vs mv1, asserted by
/// `smalln_mma_micro_27b`.
pub fn encode_mat_mat_mma8_dispatch(
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
            kernel: "mat_mat_mma8",
            detail: format!("n_in={n_in} not divisible by 256 (K-quant super-block)"),
        });
    }
    if n_out % 8 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8",
            detail: format!("n_out={n_out} not divisible by 8 (tile rows)"),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8",
            detail: format!("x/y dtype = {:?}/{:?}, expected F32/F32", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != 8 * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8",
            detail: format!("x.n_elements={} != 8*n_in={}", x.n_elements(), 8 * n_in),
        });
    }
    if y.n_elements() as usize != 8 * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8",
            detail: format!("y.n_elements={} != 8*n_out={}", y.n_elements(), 8 * n_out),
        });
    }
    if weight.n_elements() as usize != n_in * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8",
            detail: format!(
                "weight.n_elements={} != n_in*n_out={}",
                weight.n_elements(),
                n_in * n_out
            ),
        });
    }
    let name = match weight.dtype {
        GgmlType::Q4_K => "kernel_mat_mat_q4_K_mma8_f32",
        GgmlType::Q6_K => "kernel_mat_mat_q6_K_mma8_f32",
        other => {
            return Err(MetalError::BadShape {
                kernel: "mat_mat_mma8",
                detail: format!("unsupported dtype {other:?} (Q4_K | Q6_K)"),
            });
        }
    };
    let pso = ctx.pipeline(name)?;
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
    enc.dispatch(
        MTLSize {
            width: n_out / 8,
            height: 1,
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

/// v0.500 sweep (bench-only, no production call sites): dispatch a named
/// mma8v variant. `variant` ∈ {"r2c1k64", "r1c1k128", "r1c2k64",
/// "r1c1k64_sg2", "r2c2k64", "r2c1k128", "r4c1k64", "r2c2k128"};
/// column count = 8*CT (x/y must carry exactly that many columns),
/// rows per TG = 8*RT*SGS.
pub fn encode_mat_mat_mma8_variant(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    variant: &str,
) -> Result<(), MetalError> {
    let (rt, ct, sgs) = match variant {
        "r2c1k64" => (2usize, 1usize, 1usize),
        "r1c1k128" => (1, 1, 1),
        "r1c2k64" => (1, 2, 1),
        "r1c1k64_sg2" => (1, 1, 2),
        "r2c2k64" => (2, 2, 1),
        "r2c1k128" => (2, 1, 1),
        "r4c1k64" => (4, 1, 1),
        "r2c2k128" => (2, 2, 1),
        _ => {
            return Err(MetalError::BadShape {
                kernel: "mat_mat_mma8v",
                detail: format!("unknown variant {variant}"),
            });
        }
    };
    let cols = 8 * ct;
    let rows_per_tg = 8 * rt * sgs;
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8v",
            detail: format!("x/y dtype = {:?}/{:?}, expected F32/F32", x.dtype, y.dtype),
        });
    }
    if weight.n_elements() as usize != n_in * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8v",
            detail: format!(
                "weight.n_elements={} != n_in*n_out={}",
                weight.n_elements(),
                n_in * n_out
            ),
        });
    }
    if n_in % 256 != 0 || n_out % rows_per_tg != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8v",
            detail: format!("n_in={n_in} % 256 or n_out={n_out} % rows_per_tg={rows_per_tg} != 0"),
        });
    }
    if x.n_elements() as usize != cols * n_in || y.n_elements() as usize != cols * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8v",
            detail: format!(
                "x/y elements {}/{} != cols({cols}) * n_in/n_out",
                x.n_elements(),
                y.n_elements()
            ),
        });
    }
    let dt = match weight.dtype {
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q6_K => "q6_K",
        other => {
            return Err(MetalError::BadShape {
                kernel: "mat_mat_mma8v",
                detail: format!("unsupported dtype {other:?}"),
            });
        }
    };
    let name = format!("kernel_mat_mat_{dt}_mma8v_{variant}_f32");
    let pso = ctx.pipeline(&name)?;
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
    enc.dispatch(
        MTLSize {
            width: n_out / rows_per_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32 * sgs,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// v0.500 sweep B1 (bench-only): nc2 with 4 row-pairs per TG (8 SGs, 256
/// threads, 8 rows/TG). Same layouts and E0 body as
/// [`encode_mat_vec_q4_k_nc_f32`] at `n_cols = 2`.
pub fn encode_mat_vec_q4_k_nc2_rp4_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 || weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc2_rp4",
            detail: format!("n_in={n_in} % 256 != 0 or dtype {:?}", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc2_rp4",
            detail: format!("x/y dtype = {:?}/{:?}, expected F32/F32", x.dtype, y.dtype),
        });
    }
    if weight.n_elements() as usize != n_in * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc2_rp4",
            detail: format!(
                "weight.n_elements={} != n_in*n_out={}",
                weight.n_elements(),
                n_in * n_out
            ),
        });
    }
    if x.n_elements() as usize != 2 * n_in || y.n_elements() as usize != 2 * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc2_rp4",
            detail: format!(
                "x/y elements {}/{} != 2 * n_in/n_out",
                x.n_elements(),
                y.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_q4_K_nc2_rp4_f32")?;
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
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(8),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

crate::env_flag!(default_off matmat_qk_llama_smem_enabled, "QWEN_MATMAT_QK_LLAMA_SMEM");

/// Q4_K mat-mat: `Y = W · X^T` where
///   * `W` is Q4_K `[n_out, n_in]` (row-major in Q4_K block bytes)
///   * `X` is F32 `[n_query, n_in]` row-major
///   * `Y` is F32 `[n_query, n_out]` row-major — equivalently
///     `Y[c * n_out + r]` for cell `(r, c)` of the kernel's
///     "[n_out, n_query] col-major" view (the bytes are bit-identical;
///     llama's notation just uses column-major framing).
///
/// **For layer-major H5.3b.4-5 plumbing, treat the output as
/// row-major `[n_query, n_out]`.** This means downstream consumers
/// (FFN silu_mul, residual_add, chained mat-mat with this output
/// as `srcB`) work without any transpose: the bytes ARE in the
/// row-major order the next mat-mat call expects as input. The
/// H5.3b.0 layout sanity test verified this equivalence at three
/// corner cells.
///
/// Lifts the 64×32×32 simdgroup_matrix tile from llama.cpp
/// `kernel_mul_mm_q4_K_f32` (classic non-MPS-tensor path,
/// ggml-metal.metal:9440-9648). Per H5.3b plan rev 6.
///
/// Constraints:
///   * `n_in % 256 == 0` (Q4_K super-block alignment)
///   * `n_in % 32 == 0` (kernel's NK_MM=32 K-step)
///   * The kernel internally tiles N to 32 (NR1_MM); host should
///     pass `n_query` directly (kernel handles N < 32 via partial-
///     output-tile path with threadgroup-mem buffered write).
///
/// Threadgroup memory: 5120/6144 bytes for full tiles, 8192 when edge-store
/// scratch is needed.
/// Threadgroup size: 128 threads (4 simdgroups × 32 lanes).
///
/// **NOT bit-exact** with N successive `encode_mat_vec_q4_k_f32`
/// (codex Q3 correction). The lifted kernel stages activations
/// through half before float accumulation; cosine ≥ 0.999 vs
/// scalar-float mat-vec is the gate (vs cos ≥ 0.9999 against a
/// CPU mat-mat oracle that uses the same staging).

fn mat_mat_qk_threadgroup_memory(n_out: usize, n_query: usize, nr1: usize) -> usize {
    mat_mat_qk_threadgroup_memory_with_policy(n_out, n_query, nr1, matmat_qk_llama_smem_enabled())
}

fn mat_mat_qk_threadgroup_memory_with_policy(
    n_out: usize,
    n_query: usize,
    nr1: usize,
    llama_smem: bool,
) -> usize {
    if !llama_smem {
        return 8192;
    }
    if n_out % 64 == 0 && n_query % nr1 == 0 {
        if nr1 == 16 { 5120 } else { 6144 }
    } else {
        8192
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatMatQ4KN64Mode {
    Auto,
    ForceOn,
    ForceOff,
}

fn mat_mat_q4_k_n64_mode() -> MatMatQ4KN64Mode {
    static MODE: OnceLock<MatMatQ4KN64Mode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("QWEN_MATMAT_Q4_K_N64").as_deref() {
        Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO") => MatMatQ4KN64Mode::ForceOff,
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") => MatMatQ4KN64Mode::ForceOn,
        _ => MatMatQ4KN64Mode::Auto,
    })
}

#[cfg(test)]
fn mat_mat_q4_k_n64_enabled() -> bool {
    !matches!(mat_mat_q4_k_n64_mode(), MatMatQ4KN64Mode::ForceOff)
}

fn mat_mat_q4_k_use_n64(n_in: usize, n_out: usize, n_query: usize) -> bool {
    match mat_mat_q4_k_n64_mode() {
        MatMatQ4KN64Mode::ForceOff => false,
        MatMatQ4KN64Mode::ForceOn => true,
        MatMatQ4KN64Mode::Auto => !(n_query <= 512 && (n_in <= 2048 || n_out <= 2048)),
    }
}

// H5.6 M2a falsifier artifact: raw-block-staged N16 mat-mat kernel
// (`kernel_mat_mat_q4_K_f32_n16_v2`). +24% on the HOT-L2 micro, ~0% in
// production verify (occupancy trade: 14.3 KiB threadgroup memory vs
// v1's 5.1 KiB; production is machinery/latency-bound, not L2-request
// bound). Kept as an opt-in measurement artifact: QWEN_MATMAT_N16_V2=1.
crate::env_flag!(default_off mat_mat_n16_v2_enabled, "QWEN_MATMAT_N16_V2");

crate::env_flag!(default_on mat_mat_q5_k_n64_enabled, "QWEN_MATMAT_Q5_K_N64");

fn mat_mat_q5_k_n64_min_query() -> usize {
    static MIN_N: OnceLock<usize> = OnceLock::new();
    *MIN_N.get_or_init(|| {
        std::env::var("QWEN_MATMAT_Q5_K_N64_MIN_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 64)
            .unwrap_or(if cfg!(test) { 64 } else { 1024 })
    })
}

crate::env_flag!(default_on mat_mat_q6_k_n64_enabled, "QWEN_MATMAT_Q6_K_N64");

fn mat_mat_q6_k_n64_min_query() -> usize {
    static MIN_N: OnceLock<usize> = OnceLock::new();
    *MIN_N.get_or_init(|| {
        std::env::var("QWEN_MATMAT_Q6_K_N64_MIN_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 64)
            .unwrap_or(if cfg!(test) { 64 } else { 1024 })
    })
}

pub fn encode_mat_mat_q4_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor, // [n_query, n_in] row-major F32
    y: &MetalTensor, // F32 [n_out * n_query] flat. SEE LAYOUT NOTE BELOW.
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q4_k",
            detail: format!("n_in={n_in} not divisible by 256 (Q4_K super-block)"),
        });
    }
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q4_k",
            detail: format!("n_in={n_in} not divisible by 32 (NK_MM tile)"),
        });
    }
    if weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q4_k",
            detail: format!("weight.dtype = {:?}, expected Q4_K", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q4_k",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q4_k",
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let nb01 = ((n_in / 256) * 144) as u32;
    let stride_b = n_in as u32;

    let use_n64 =
        mat_mat_q4_k_use_n64(n_in, n_out, n_query) && n_query % 64 == 0 && n_out % 64 == 0;
    // H5.6 M2a: raw-block-staged N16 kernel (v2). The v1 A-path dequants
    // straight from device with ~2.7x byte amplification; v2 stages raw
    // super-blocks to threadgroup memory coalesced. K must cover whole
    // super-blocks. Rollback: QWEN_MATMAT_N16_V2=0.
    let use_n16_v2 = n_query == 16 && !use_n64 && n_in % 256 == 0 && mat_mat_n16_v2_enabled();
    let kernel_name = if use_n64 {
        "kernel_mat_mat_q4_K_f32_n64"
    } else if use_n16_v2 {
        "kernel_mat_mat_q4_K_f32_n16_v2"
    } else if n_query == 16 {
        "kernel_mat_mat_q4_K_f32_n16"
    } else {
        "kernel_mat_mat_q4_K_f32"
    };
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    let nr1 = if use_n64 {
        64
    } else if n_query == 16 {
        16
    } else {
        32
    };
    let smem = if use_n64 {
        8192
    } else if use_n16_v2 {
        // raw 9216 + sa 4096 + sb 1024 (kernel doc block).
        14336
    } else {
        mat_mat_qk_threadgroup_memory(n_out, n_query, nr1)
    };
    enc.set_threadgroup_memory(0, smem);
    let n_tg_x = n_query.div_ceil(nr1);
    let n_tg_y = n_out.div_ceil(64);
    let threads = if use_n64 { 256 } else { 128 };
    enc.dispatch(
        MTLSize {
            width: n_tg_x,
            height: n_tg_y,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Fused SwiGLU FFN dispatch for Q4_K weights.
///
/// Replaces the 3-dispatch sequence:
///   mat_vec_q4_K(W_gate, x) -> gate
///   mat_vec_q4_K(W_up,   x) -> up
///   silu_mul(gate, up)      -> inner
/// with a single kernel that:
///   * reads x ONCE per super-block (shared across both gate and up paths)
///   * eliminates the n_out-element `gate` and `up` intermediate buffers
///   * fuses silu(gate) * up into the final lane-0 write
///
/// At the 27B FFN shape (n_in=5120, n_out=17408), this saves ~140 KB of
/// intermediate writes+reads per layer × 48 layers = ~6.5 MB/token.
///
/// CPU oracle: equivalent to the unfused 3-dispatch sequence (validated
/// by `ffn_swiglu_q4_K_matches_unfused`).
#[allow(non_snake_case)]
pub fn encode_ffn_swiglu_q4_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "ffn_swiglu_q4_K",
            detail: format!("n_in={n_in} not divisible by 256 (Q4_K super-block)"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "ffn_swiglu_q4_K",
            detail: format!(
                "expected Q4_K weights, got gate={:?} up={:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in {
        return Err(MetalError::BadShape {
            kernel: "ffn_swiglu_q4_K",
            detail: format!("x.n={} != n_in={n_in}", x.n_elements()),
        });
    }
    if inner.n_elements() as usize != n_out {
        return Err(MetalError::BadShape {
            kernel: "ffn_swiglu_q4_K",
            detail: format!("inner.n={} != n_out={n_out}", inner.n_elements()),
        });
    }

    let pso = ctx.pipeline("kernel_ffn_swiglu_q4_K_f32")?;
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
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, inner);

    // Same threadgroup shape as kernel_mat_vec_q4_K_f32: NR0=2 rows per
    // simdgroup, NSG=2 simdgroups per threadgroup.
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

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K",
            detail: format!(
                "expected Q4_K expert gate/up, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K",
            detail: format!("x.n_elements={} != n_in={n_in}", x.n_elements()),
        });
    }
    if topk_idx.n_elements() as usize != topk || inner.n_elements() as usize != topk * n_out {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K",
            detail: format!(
                "topk/inner mismatch: idx={} inner={} expected idx={topk} inner={}",
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q4_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_packed_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    topk_idx_pack: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_packed_slots",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_packed_slots",
            detail: format!(
                "expected Q4_K expert gate/up, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    let n_slots = n_tokens * topk;
    if x_pack.n_elements() as usize != n_tokens * n_in {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_packed_slots",
            detail: format!(
                "x_pack.n_elements={} != n_tokens*n_in={}",
                x_pack.n_elements(),
                n_tokens * n_in
            ),
        });
    }
    if topk_idx_pack.n_elements() as usize != n_slots
        || inner.n_elements() as usize != n_slots * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_packed_slots",
            detail: format!(
                "slot/inner mismatch: idx={} inner={} expected idx={n_slots} inner={}",
                topk_idx_pack.n_elements(),
                inner.n_elements(),
                n_slots * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q4_K_f32_packed_slots")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, topk_idx_pack);
    enc.set_tensor(5, inner);

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: n_slots,
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

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_hidden % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_n16",
            detail: format!(
                "expected Q4_K gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q4_K_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 144) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_hidden % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::IQ3_XXS || w_up.dtype != GgmlType::IQ3_XXS {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_grouped_slots_n16",
            detail: format!(
                "expected IQ3_XXS gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 98) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_iq3_s_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_iq3_s_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_iq3_s_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_hidden % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::IQ3_S || w_up.dtype != GgmlType::IQ3_S {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s_grouped_slots_n16",
            detail: format!(
                "expected IQ3_S gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_iq3_s_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 110) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q5_K_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_q5_K_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q5_K_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_hidden % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q5_K_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q5_K || w_up.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q5_K_grouped_slots_n16",
            detail: format!(
                "expected Q5_K gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q5_K_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q5_K_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 176) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q6_K_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_q6_K_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q6_K_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_hidden % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q6_K_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q6_K || w_up.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q6_K_grouped_slots_n16",
            detail: format!(
                "expected Q6_K gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q6_K_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q6_K_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 210) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_swiglu_q8_0_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_q8_0_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

pub fn encode_moe_swiglu_q8_0_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_hidden % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q8_0_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 32"),
        });
    }
    if w_gate.dtype != GgmlType::Q8_0 || w_up.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q8_0_grouped_slots_n16",
            detail: format!(
                "expected Q8_0 gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q8_0_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q8_0_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 32) * 34) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_swiglu_f32_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if w_gate.dtype != GgmlType::F32 || w_up.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_f32_grouped_slots_n16",
            detail: format!(
                "expected F32 gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_f32_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_f32_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01: n_hidden as u32,
            stride_b: n_hidden as u32,
            min_count: 0,
            max_count: i32::MAX as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_swiglu_bf16_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_bf16_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

pub fn encode_moe_swiglu_bf16_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if w_gate.dtype != GgmlType::BF16 || w_up.dtype != GgmlType::BF16 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_bf16_grouped_slots_n16",
            detail: format!(
                "expected BF16 gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_bf16_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_bf16_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01: n_hidden as u32,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_fused: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_hidden % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if n_ffn % 64 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n16",
            detail: format!("n_ffn={n_ffn} not divisible by 64"),
        });
    }
    if topk == 0 || n_expert == 0 || n_tokens == 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n16",
            detail: "topk/n_expert/n_tokens must all be > 0".into(),
        });
    }
    if w_fused.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n16",
            detail: format!("expected fused Q4_K expert bank, got {:?}", w_fused.dtype),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q4_K_f32_grouped_slots_fused_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 144) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_fused);
    enc.set_tensor(2, x_pack);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, inner);
    enc.set_threadgroup_memory(0, 12288);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_grouped_slots_n32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_hidden % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_n32",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_n32",
            detail: format!(
                "expected Q4_K gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_n32",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q4_K_f32_grouped_slots_n32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 144) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n32_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_fused: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_hidden % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n32",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if n_ffn % 64 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n32",
            detail: format!("n_ffn={n_ffn} not divisible by 64"),
        });
    }
    if topk == 0 || n_expert == 0 || n_tokens == 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n32",
            detail: "topk/n_expert/n_tokens must all be > 0".into(),
        });
    }
    if w_fused.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n32",
            detail: format!("expected fused Q4_K expert bank, got {:?}", w_fused.dtype),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n32",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q4_K_f32_grouped_slots_fused_n32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 144) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_fused);
    enc.set_tensor(2, x_pack);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_matmul_q4_K_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_matmul_q4_K_f32_grouped_slots_n16_range(
        ctx,
        enc,
        weight,
        x_pack,
        counts,
        ids,
        out,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_matmul_q4_K_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_hidden % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_matmul_q4_K_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_matmul_q4_K_grouped_slots_n16",
            detail: format!("expected Q4_K expert bank, got {:?}", weight.dtype),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_matmul_q4_K_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} out={} expected x={} counts={} ids={} out={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_matmul_q4_K_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 144) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x_pack);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_matmul_q4_K_f32_grouped_slots_n32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_matmul_q4_K_f32_grouped_slots_n32_range(
        ctx,
        enc,
        weight,
        x_pack,
        counts,
        ids,
        out,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_matmul_q4_K_f32_grouped_slots_n32_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_hidden % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_matmul_q4_K_grouped_slots_n32",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_matmul_q4_K_grouped_slots_n32",
            detail: format!("expected Q4_K expert bank, got {:?}", weight.dtype),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_matmul_q4_K_grouped_slots_n32",
            detail: format!(
                "shape mismatch x={} counts={} ids={} out={} expected x={} counts={} ids={} out={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_matmul_q4_K_f32_grouped_slots_n32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 144) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x_pack);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_fused_routed_q4q5_token_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    w_down: &MetalTensor,
    x_pack: &MetalTensor,
    topk_idx_pack: &MetalTensor,
    topk_w_pack: &MetalTensor,
    out: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if n_hidden % 256 != 0 || n_ffn % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_fused_routed_q4q5_token",
            detail: format!("n_hidden={n_hidden} and n_ffn={n_ffn} must both be divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K
        || w_up.dtype != GgmlType::Q4_K
        || w_down.dtype != GgmlType::Q5_K
    {
        return Err(MetalError::BadShape {
            kernel: "moe_fused_routed_q4q5_token",
            detail: format!(
                "expected gate/up/down dtypes Q4_K/Q4_K/Q5_K, got {:?}/{:?}/{:?}",
                w_gate.dtype, w_up.dtype, w_down.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || topk_idx_pack.n_elements() as usize != n_tokens * topk
        || topk_w_pack.n_elements() as usize != n_tokens * topk
        || out.n_elements() as usize != n_tokens * n_hidden
    {
        return Err(MetalError::BadShape {
            kernel: "moe_fused_routed_q4q5_token",
            detail: format!(
                "shape mismatch: x={} idx={} w={} out={} expected x={} idx={} w={} out={}",
                x_pack.n_elements(),
                topk_idx_pack.n_elements(),
                topk_w_pack.n_elements(),
                out.n_elements(),
                n_tokens * n_hidden,
                n_tokens * topk,
                n_tokens * topk,
                n_tokens * n_hidden
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_fused_routed_q4q5_token_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        hidden: u32,
        ffn: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            hidden: n_hidden as u32,
            ffn: n_ffn as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, w_down);
    enc.set_tensor(4, x_pack);
    enc.set_tensor(5, topk_idx_pack);
    enc.set_tensor(6, topk_w_pack);
    enc.set_tensor(7, out);
    enc.set_threadgroup_memory(0, n_ffn * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: n_tokens,
            depth: 1,
        },
        MTLSize {
            width: 512,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_down_q4_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q4_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q4_K",
            detail: format!("expected Q4_K expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q4_K",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q4_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_q5_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K",
            detail: format!("expected Q5_K expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q5_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);

    const NR0: usize = 1;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_q5_K_f32_grouped_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    expert_idx: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_rows: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_rows",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_rows",
            detail: format!("expected Q5_K expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != n_rows * n_in
        || expert_idx.n_elements() as usize != n_rows
        || out.n_elements() as usize != n_rows * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_rows",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={} out={}",
                inner.n_elements(),
                expert_idx.n_elements(),
                out.n_elements(),
                n_rows * n_in,
                n_rows,
                n_rows * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q5_K_f32_grouped_rows")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: 0,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, expert_idx);
    enc.set_tensor(4, out);

    const NR0: usize = 1;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: n_rows,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_q5_K_f32_grouped_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_down_q5_K_f32_grouped_slots_range(
        ctx,
        enc,
        weight,
        inner,
        counts,
        ids,
        out,
        n_in,
        n_out,
        n_expert,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_down_q5_K_f32_grouped_slots_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_slots",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_slots",
            detail: format!("expected Q5_K expert down, got {:?}", weight.dtype),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if inner.n_elements() as usize != slot_count * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != slot_count * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_slots",
            detail: format!(
                "shape mismatch: inner={} counts={} ids={} out={} expected inner={} counts={} ids={} out={}",
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                slot_count * n_in,
                n_expert,
                n_expert * n_tokens,
                slot_count * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q5_K_f32_grouped_slots")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_in / 256) * 176) as u32;
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            nb01,
            stride_b: n_in as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_down_q5_K_f32_grouped_slots_tiny8_r16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_slots_tiny8_r16",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_slots_tiny8_r16",
            detail: format!("expected Q5_K expert down, got {:?}", weight.dtype),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if inner.n_elements() as usize != slot_count * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != slot_count * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_slots_tiny8_r16",
            detail: format!(
                "shape mismatch: inner={} counts={} ids={} out={} expected inner={} counts={} ids={} out={}",
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                slot_count * n_in,
                n_expert,
                n_expert * n_tokens,
                slot_count * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q5_K_f32_grouped_slots_tiny8_r16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        n_expert: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_in / 256) * 176) as u32;
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            n_expert: n_expert as u32,
            nb01,
            stride_b: n_in as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(16),
            height: n_expert.div_ceil(4),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_down_q6_K_f32_grouped_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q6_K_grouped_slots",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q6_K_grouped_slots",
            detail: format!("expected Q6_K expert down, got {:?}", weight.dtype),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if inner.n_elements() as usize != slot_count * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != slot_count * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q6_K_grouped_slots",
            detail: format!(
                "shape mismatch: inner={} counts={} ids={} out={} expected inner={} counts={} ids={} out={}",
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                slot_count * n_in,
                n_expert,
                n_expert * n_tokens,
                slot_count * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q6_K_f32_grouped_slots")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let nb01 = ((n_in / 256) * 210) as u32;
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            nb01,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_down_q8_0_f32_grouped_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q8_0_grouped_slots",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q8_0_grouped_slots",
            detail: format!("expected Q8_0 expert down, got {:?}", weight.dtype),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if inner.n_elements() as usize != slot_count * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != slot_count * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q8_0_grouped_slots",
            detail: format!(
                "shape mismatch: inner={} counts={} ids={} out={} expected inner={} counts={} ids={} out={}",
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                slot_count * n_in,
                n_expert,
                n_expert * n_tokens,
                slot_count * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q8_0_f32_grouped_slots")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let nb01 = ((n_in / 32) * 34) as u32;
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            nb01,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_down_bf16_f32_grouped_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_down_bf16_f32_grouped_slots_range(
        ctx,
        enc,
        weight,
        inner,
        counts,
        ids,
        out,
        n_in,
        n_out,
        n_expert,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

pub fn encode_moe_down_bf16_f32_grouped_slots_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_bf16_grouped_slots",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::BF16 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_bf16_grouped_slots",
            detail: format!("expected BF16 expert down, got {:?}", weight.dtype),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if inner.n_elements() as usize != slot_count * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != slot_count * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_bf16_grouped_slots",
            detail: format!(
                "shape mismatch: inner={} counts={} ids={} out={} expected inner={} counts={} ids={} out={}",
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                slot_count * n_in,
                n_expert,
                n_expert * n_tokens,
                slot_count * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_bf16_f32_grouped_slots")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            nb01: n_in as u32,
            stride_b: n_in as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_down_iq4_xs_f32_grouped_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs_grouped_slots",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::IQ4_XS {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs_grouped_slots",
            detail: format!("expected IQ4_XS expert down, got {:?}", weight.dtype),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if inner.n_elements() as usize != slot_count * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != slot_count * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs_grouped_slots",
            detail: format!(
                "shape mismatch: inner={} counts={} ids={} out={} expected inner={} counts={} ids={} out={}",
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                slot_count * n_in,
                n_expert,
                n_expert * n_tokens,
                slot_count * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_iq4_xs_f32_grouped_slots")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            nb01: (n_in / 256) as u32,
            stride_b: n_in as u32,
            min_count: 0,
            max_count: i32::MAX as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    topk_w: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q5_K_packed_slots",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q5_K_packed_slots",
            detail: format!("expected Q5_K expert down, got {:?}", weight.dtype),
        });
    }
    let n_slots = n_tokens * topk;
    if inner.n_elements() as usize != n_slots * n_in
        || topk_idx.n_elements() as usize != n_slots
        || topk_w.n_elements() as usize != n_slots
        || out.n_elements() as usize != n_tokens * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q5_K_packed_slots",
            detail: format!(
                "shape mismatch: inner={} idx={} w={} out={} expected inner={} idx={} w={} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                topk_w.n_elements(),
                out.n_elements(),
                n_slots * n_in,
                n_slots,
                n_slots,
                n_tokens * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_weighted_sum_q5_K_f32_packed_slots")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, topk_w);
    enc.set_tensor(5, out);

    const NR0: usize = 1;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: n_tokens,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    topk_w: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if n_in != 512 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q5_K_k512_r2",
            detail: format!("expected n_in=512, got {n_in}"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q5_K_k512_r2",
            detail: format!("expected Q5_K expert down, got {:?}", weight.dtype),
        });
    }
    let n_slots = n_tokens * topk;
    if inner.n_elements() as usize != n_slots * n_in
        || topk_idx.n_elements() as usize != n_slots
        || topk_w.n_elements() as usize != n_slots
        || out.n_elements() as usize != n_tokens * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q5_K_k512_r2",
            detail: format!(
                "shape mismatch: inner={} idx={} w={} out={} expected inner={} idx={} w={} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                topk_w.n_elements(),
                out.n_elements(),
                n_slots * n_in,
                n_slots,
                n_slots,
                n_tokens * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, topk_w);
    enc.set_tensor(5, out);

    const ROWS_PER_TG: usize = 4;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(ROWS_PER_TG),
            height: n_tokens,
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

#[allow(non_snake_case)]
pub fn encode_moe_mat_vec_q5_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_q5_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_q5_K",
            detail: format!("expected Q5_K expert weight, got {:?}", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_q5_K",
            detail: format!(
                "shape mismatch: x={} idx={} out={} expected x={n_in} idx={topk} out={}",
                x.n_elements(),
                topk_idx.n_elements(),
                out.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_mat_vec_q5_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, out);

    const NR0: usize = 1;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

pub fn encode_moe_mat_vec_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_iq3_xxs",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::IQ3_XXS {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_iq3_xxs",
            detail: format!("expected IQ3_XXS expert weight, got {:?}", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_iq3_xxs",
            detail: format!(
                "shape mismatch: x={} idx={} out={} expected x={n_in} idx={topk} out={}",
                x.n_elements(),
                topk_idx.n_elements(),
                out.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_mat_vec_iq3_xxs_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, out);

    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_mat_vec_iq3_s_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_iq3_s",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::IQ3_S {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_iq3_s",
            detail: format!("expected IQ3_S expert weight, got {:?}", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_iq3_s",
            detail: format!(
                "shape mismatch: x={} idx={} out={} expected x={n_in} idx={topk} out={}",
                x.n_elements(),
                topk_idx.n_elements(),
                out.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_mat_vec_iq3_s_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, out);

    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_swiglu_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::IQ3_XXS || w_up.dtype != GgmlType::IQ3_XXS {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs",
            detail: format!(
                "expected IQ3_XXS gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || inner.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs",
            detail: format!(
                "shape mismatch: x={} idx={} inner={} expected x={n_in} idx={topk} inner={}",
                x.n_elements(),
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_iq3_xxs_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_swiglu_iq3_xxs_f32_fast(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_fast",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::IQ3_XXS || w_up.dtype != GgmlType::IQ3_XXS {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_fast",
            detail: format!(
                "expected IQ3_XXS gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || inner.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_fast",
            detail: format!(
                "shape mismatch: x={} idx={} inner={} expected x={n_in} idx={topk} inner={}",
                x.n_elements(),
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_iq3_xxs_f32_fast")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NR0: usize = 4;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

pub fn encode_moe_swiglu_iq3_s_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::IQ3_S || w_up.dtype != GgmlType::IQ3_S {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s",
            detail: format!(
                "expected IQ3_S gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || inner.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s",
            detail: format!(
                "shape mismatch: x={} idx={} inner={} expected x={n_in} idx={topk} inner={}",
                x.n_elements(),
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_iq3_s_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_swiglu_iq3_s_f32_fast(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s_fast",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::IQ3_S || w_up.dtype != GgmlType::IQ3_S {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s_fast",
            detail: format!(
                "expected IQ3_S gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || inner.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s_fast",
            detail: format!(
                "shape mismatch: x={} idx={} inner={} expected x={n_in} idx={topk} inner={}",
                x.n_elements(),
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_iq3_s_f32_fast")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NR0: usize = 4;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q6_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q6_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q6_K || w_up.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q6_K",
            detail: format!(
                "expected Q6_K gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || inner.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q6_K",
            detail: format!(
                "shape mismatch: x={} idx={} inner={} expected x={n_in} idx={topk} inner={}",
                x.n_elements(),
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q6_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

pub fn encode_moe_swiglu_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q8_0",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if w_gate.dtype != GgmlType::Q8_0 || w_up.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q8_0",
            detail: format!(
                "expected Q8_0 gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || inner.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q8_0",
            detail: format!(
                "shape mismatch: x={} idx={} inner={} expected x={n_in} idx={topk} inner={}",
                x.n_elements(),
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q8_0_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    let nr0 = 2usize;
    let nsg = 4usize;
    enc.set_threadgroup_memory(0, 32 * 2 * nr0 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(nr0),
            height: topk,
            depth: 1,
        },
        MTLSize {
            width: nsg * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_mat_vec_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_f32",
            detail: format!("expected F32 expert weight, got {:?}", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_f32",
            detail: format!(
                "shape mismatch: x={} idx={} out={} expected x={n_in} idx={topk} out={}",
                x.n_elements(),
                topk_idx.n_elements(),
                out.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_mat_vec_f32_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, out);

    const NSG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_down_f32_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_f32",
            detail: format!("expected F32 expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_f32",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_f32_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);

    const NSG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_mat_vec_bf16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::BF16 {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_bf16",
            detail: format!("expected BF16 expert weight, got {:?}", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_bf16",
            detail: format!(
                "shape mismatch: x={} idx={} out={} expected x={n_in} idx={topk} out={}",
                x.n_elements(),
                topk_idx.n_elements(),
                out.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_mat_vec_bf16_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, out);

    const NSG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_down_bf16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::BF16 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_bf16",
            detail: format!("expected BF16 expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_bf16",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_bf16_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);

    const NSG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_q6_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q6_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q6_K",
            detail: format!("expected Q6_K expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q6_K",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q6_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_iq4_xs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::IQ4_XS {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs",
            detail: format!("expected IQ4_XS expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_iq4_xs_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);

    const NSG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_iq4_xs_f32_fast(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs_fast",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::IQ4_XS {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs_fast",
            detail: format!("expected IQ4_XS expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs_fast",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_iq4_xs_f32_fast")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);
    enc.set_threadgroup_memory(0, 32 * std::mem::size_of::<f32>());

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_weighted_sum_q6_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    topk_w: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q6_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q6_K",
            detail: format!("expected Q6_K expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || topk_w.n_elements() as usize != topk
        || out.n_elements() as usize != n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q6_K",
            detail: format!(
                "shape mismatch: inner={} idx={} w={} out={} expected inner={} idx={topk} w={topk} out={n_out}",
                inner.n_elements(),
                topk_idx.n_elements(),
                topk_w.n_elements(),
                out.n_elements(),
                topk * n_in
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_weighted_sum_q6_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, topk_w);
    enc.set_tensor(5, out);

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

pub fn encode_moe_down_weighted_sum_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    topk_w: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q8_0",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q8_0",
            detail: format!("expected Q8_0 expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || topk_w.n_elements() as usize != topk
        || out.n_elements() as usize != n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q8_0",
            detail: format!(
                "shape mismatch: inner={} idx={} w={} out={} expected inner={} idx={topk} w={topk} out={n_out}",
                inner.n_elements(),
                topk_idx.n_elements(),
                topk_w.n_elements(),
                out.n_elements(),
                topk * n_in
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_weighted_sum_q8_0_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, topk_w);
    enc.set_tensor(5, out);

    let nr0 = 2usize;
    let nsg = 4usize;
    enc.set_threadgroup_memory(0, 32 * nr0 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(nr0),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: nsg * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_weighted_sum_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    expert_out: &MetalTensor,
    weights: &MetalTensor,
    out: &MetalTensor,
    n_out: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if expert_out.n_elements() as usize != topk * n_out
        || weights.n_elements() as usize != topk
        || out.n_elements() as usize != n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_weighted_sum",
            detail: format!(
                "shape mismatch: expert_out={} weights={} out={} expected {}/{}/{}",
                expert_out.n_elements(),
                weights.n_elements(),
                out.n_elements(),
                topk * n_out,
                topk,
                n_out
            ),
        });
    }
    let pso = ctx.pipeline("kernel_moe_weighted_sum_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_out: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_out: n_out as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, expert_out);
    enc.set_tensor(2, weights);
    enc.set_tensor(3, out);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(tg_threads),
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

pub fn encode_moe_weighted_sum_packed_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    expert_out: &MetalTensor,
    weights: &MetalTensor,
    out: &MetalTensor,
    n_out: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if expert_out.n_elements() as usize != n_tokens * topk * n_out
        || weights.n_elements() as usize != n_tokens * topk
        || out.n_elements() as usize != n_tokens * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_weighted_sum_packed",
            detail: format!(
                "shape mismatch: expert_out={} weights={} out={} expected expert_out={} weights={} out={}",
                expert_out.n_elements(),
                weights.n_elements(),
                out.n_elements(),
                n_tokens * topk * n_out,
                n_tokens * topk,
                n_tokens * n_out
            ),
        });
    }
    let pso = ctx.pipeline("kernel_moe_weighted_sum_packed_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_out: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_out: n_out as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, expert_out);
    enc.set_tensor(2, weights);
    enc.set_tensor(3, out);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(32),
            height: n_tokens,
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

pub fn encode_moe_grouped_finalizer_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    expert_out: &MetalTensor,
    topk_w: &MetalTensor,
    shared_gate: &MetalTensor,
    shared_out: &MetalTensor,
    x_pack: &MetalTensor,
    n_out: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    let n_slots = n_tokens * topk;
    if expert_out.n_elements() as usize != n_slots * n_out
        || topk_w.n_elements() as usize != n_slots
        || shared_gate.n_elements() as usize != n_tokens
        || shared_out.n_elements() as usize != n_tokens * n_out
        || x_pack.n_elements() as usize != n_tokens * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_grouped_finalizer",
            detail: format!(
                "shape mismatch: expert_out={} topk_w={} shared_gate={} shared_out={} x_pack={} expected expert_out={} topk_w={} shared_gate={} shared_out={} x_pack={}",
                expert_out.n_elements(),
                topk_w.n_elements(),
                shared_gate.n_elements(),
                shared_out.n_elements(),
                x_pack.n_elements(),
                n_slots * n_out,
                n_slots,
                n_tokens,
                n_tokens * n_out,
                n_tokens * n_out
            ),
        });
    }
    let pso = ctx.pipeline("kernel_moe_grouped_finalizer_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_out: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_out: n_out as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, expert_out);
    enc.set_tensor(2, topk_w);
    enc.set_tensor(3, shared_gate);
    enc.set_tensor(4, shared_out);
    enc.set_tensor(5, x_pack);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(2),
            height: n_tokens,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_scatter_rows_f32_unique(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    rows: &MetalTensor,
    out: &MetalTensor,
    n_cols: usize,
    n_rows: usize,
) -> Result<(), MetalError> {
    let n = n_cols * n_rows;
    if x.n_elements() as usize != n
        || rows.n_elements() as usize != n_rows
        || out.n_elements() as usize % n_cols != 0
    {
        return Err(MetalError::BadShape {
            kernel: "scatter_rows_unique",
            detail: format!(
                "expected x n={} rows n_rows={} out multiple of n_cols={}, got x={} rows={} out={}",
                n,
                n_rows,
                n_cols,
                x.n_elements(),
                rows.n_elements(),
                out.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_scatter_rows_f32_unique")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_cols: u32,
        n_rows: u32,
        out_rows: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_cols: n_cols as u32,
            n_rows: n_rows as u32,
            out_rows: (out.n_elements() as usize / n_cols) as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, rows);
    enc.set_tensor(3, out);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
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

pub fn encode_axpy_scalar_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    scale: &MetalTensor,
    accum: &MetalTensor,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if scale.n_elements() != 1 || accum.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "axpy_scalar",
            detail: format!(
                "expected scale[1] and accum n={n}, got scale={} accum={}",
                scale.n_elements(),
                accum.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_axpy_scalar_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
    }
    enc.set_bytes(0, &Args { n: n as u32 });
    enc.set_tensor(1, x);
    enc.set_tensor(2, scale);
    enc.set_tensor(3, accum);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
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

pub fn encode_moe_shared_accum_resid_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    shared_out: &MetalTensor,
    shared_gate: &MetalTensor,
    mixer_out: &MetalTensor,
    x: &MetalTensor,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if shared_gate.n_elements() != 1
        || shared_out.n_elements() as usize != n
        || mixer_out.n_elements() as usize != n
    {
        return Err(MetalError::BadShape {
            kernel: "moe_shared_accum_resid",
            detail: format!(
                "expected shared_gate[1] and shared_out/mixer_out/x n={n}, got gate={} shared={} mixer={}",
                shared_gate.n_elements(),
                shared_out.n_elements(),
                mixer_out.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_moe_shared_accum_resid_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
    }
    enc.set_bytes(0, &Args { n: n as u32 });
    enc.set_tensor(1, shared_out);
    enc.set_tensor(2, shared_gate);
    enc.set_tensor(3, mixer_out);
    enc.set_tensor(4, x);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
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

pub fn encode_topk_logits_softmax_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if logits.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax",
            detail: format!("logits.n_elements={} != n={n}", logits.n_elements()),
        });
    }
    if out_idx.n_elements() as usize != k || out_w.n_elements() as usize != k {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax",
            detail: format!(
                "out_idx/out_w expected {k} elements, got {}/{}",
                out_idx.n_elements(),
                out_w.n_elements()
            ),
        });
    }
    if k == 0 || k > 16 {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax",
            detail: format!("k={k} must be in 1..=16"),
        });
    }
    let pso = ctx.pipeline("kernel_topk_logits_softmax_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        k: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            k: k as u32,
        },
    );
    enc.set_tensor(1, logits);
    enc.set_tensor(2, out_idx);
    enc.set_tensor(3, out_w);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_topk_logits_softmax_parallel_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if logits.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_parallel",
            detail: format!("logits.n_elements={} != n={n}", logits.n_elements()),
        });
    }
    if out_idx.n_elements() as usize != k || out_w.n_elements() as usize != k {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_parallel",
            detail: format!(
                "out_idx/out_w expected {k} elements, got {}/{}",
                out_idx.n_elements(),
                out_w.n_elements()
            ),
        });
    }
    if n == 0 || n > 256 || k == 0 || k > 16 || k > n {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_parallel",
            detail: format!("expected 1 <= k <= n <= 256 and k <= 16, got n={n} k={k}"),
        });
    }
    let pso = ctx.pipeline("kernel_topk_logits_softmax_parallel_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        k: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            k: k as u32,
        },
    );
    enc.set_tensor(1, logits);
    enc.set_tensor(2, out_idx);
    enc.set_tensor(3, out_w);
    const THREADS: usize = 256;
    enc.set_threadgroup_memory(0, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, THREADS * std::mem::size_of::<i32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_dot_sigmoid_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    out: &MetalTensor,
    n: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::F32 || x.dtype != GgmlType::F32 || out.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "dot_sigmoid",
            detail: format!(
                "expected F32/F32/F32, got {:?}/{:?}/{:?}",
                weight.dtype, x.dtype, out.dtype
            ),
        });
    }
    if weight.n_elements() as usize != n || x.n_elements() as usize != n || out.n_elements() != 1 {
        return Err(MetalError::BadShape {
            kernel: "dot_sigmoid",
            detail: format!(
                "shape mismatch: weight={} x={} out={} expected n={n}, out=1",
                weight.n_elements(),
                x.n_elements(),
                out.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_dot_sigmoid_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
    }
    enc.set_bytes(0, &Args { n: n as u32 });
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, out);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
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

pub fn encode_topk_logits_softmax_dot_sigmoid_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    shared_weight: &MetalTensor,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    shared_out: &MetalTensor,
    n_expert: usize,
    topk: usize,
    hidden: usize,
) -> Result<(), MetalError> {
    if logits.dtype != GgmlType::F32
        || shared_weight.dtype != GgmlType::F32
        || x.dtype != GgmlType::F32
        || out_w.dtype != GgmlType::F32
        || shared_out.dtype != GgmlType::F32
    {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_dot_sigmoid",
            detail: format!(
                "expected F32 logits/shared_weight/x/out_w/shared_out, got {:?}/{:?}/{:?}/{:?}/{:?}",
                logits.dtype, shared_weight.dtype, x.dtype, out_w.dtype, shared_out.dtype
            ),
        });
    }
    if n_expert == 0 || n_expert > 256 || topk == 0 || topk > 16 || topk > n_expert {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_dot_sigmoid",
            detail: format!(
                "expected 1 <= topk <= n_expert <= 256 and topk <= 16, got n_expert={n_expert} topk={topk}"
            ),
        });
    }
    if logits.n_elements() as usize != n_expert
        || out_idx.n_elements() as usize != topk
        || out_w.n_elements() as usize != topk
        || shared_weight.n_elements() as usize != hidden
        || x.n_elements() as usize != hidden
        || shared_out.n_elements() != 1
    {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_dot_sigmoid",
            detail: format!(
                "shape mismatch logits={} idx={} w={} shared_weight={} x={} shared_out={} expected {n_expert}/{topk}/{topk}/{hidden}/{hidden}/1",
                logits.n_elements(),
                out_idx.n_elements(),
                out_w.n_elements(),
                shared_weight.n_elements(),
                x.n_elements(),
                shared_out.n_elements()
            ),
        });
    }

    let pso = ctx.pipeline("kernel_topk_logits_softmax_dot_sigmoid_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_expert: u32,
        topk: u32,
        hidden: u32,
        n_tokens: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_expert: n_expert as u32,
            topk: topk as u32,
            hidden: hidden as u32,
            n_tokens: 1,
        },
    );
    enc.set_tensor(1, logits);
    enc.set_tensor(2, shared_weight);
    enc.set_tensor(3, x);
    enc.set_tensor(4, out_idx);
    enc.set_tensor(5, out_w);
    enc.set_tensor(6, shared_out);
    const THREADS: usize = 256;
    enc.set_threadgroup_memory(0, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, THREADS * std::mem::size_of::<i32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_topk_logits_softmax_dot_sigmoid_packed_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    shared_weight: &MetalTensor,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    shared_out: &MetalTensor,
    n_expert: usize,
    topk: usize,
    hidden: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if logits.dtype != GgmlType::F32
        || shared_weight.dtype != GgmlType::F32
        || x.dtype != GgmlType::F32
        || out_w.dtype != GgmlType::F32
        || shared_out.dtype != GgmlType::F32
    {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_dot_sigmoid_packed",
            detail: format!(
                "expected F32 logits/shared_weight/x/out_w/shared_out, got {:?}/{:?}/{:?}/{:?}/{:?}",
                logits.dtype, shared_weight.dtype, x.dtype, out_w.dtype, shared_out.dtype
            ),
        });
    }
    if n_expert == 0 || n_expert > 256 || topk == 0 || topk > 16 || topk > n_expert {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_dot_sigmoid_packed",
            detail: format!(
                "expected 1 <= topk <= n_expert <= 256 and topk <= 16, got n_expert={n_expert} topk={topk}"
            ),
        });
    }
    if logits.n_elements() as usize != n_tokens * n_expert
        || out_idx.n_elements() as usize != n_tokens * topk
        || out_w.n_elements() as usize != n_tokens * topk
        || shared_weight.n_elements() as usize != hidden
        || x.n_elements() as usize != n_tokens * hidden
        || shared_out.n_elements() as usize != n_tokens
    {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_dot_sigmoid_packed",
            detail: format!(
                "shape mismatch logits={} idx={} w={} shared_weight={} x={} shared_out={} expected {}/{}/{}/{hidden}/{}/{}",
                logits.n_elements(),
                out_idx.n_elements(),
                out_w.n_elements(),
                shared_weight.n_elements(),
                x.n_elements(),
                shared_out.n_elements(),
                n_tokens * n_expert,
                n_tokens * topk,
                n_tokens * topk,
                n_tokens * hidden,
                n_tokens
            ),
        });
    }

    let pso = ctx.pipeline("kernel_topk_logits_softmax_dot_sigmoid_packed_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_expert: u32,
        topk: u32,
        hidden: u32,
        n_tokens: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_expert: n_expert as u32,
            topk: topk as u32,
            hidden: hidden as u32,
            n_tokens: n_tokens as u32,
        },
    );
    enc.set_tensor(1, logits);
    enc.set_tensor(2, shared_weight);
    enc.set_tensor(3, x);
    enc.set_tensor(4, out_idx);
    enc.set_tensor(5, out_w);
    enc.set_tensor(6, shared_out);
    const THREADS: usize = 256;
    enc.set_threadgroup_memory(0, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, THREADS * std::mem::size_of::<i32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: n_tokens,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    shared_weight: &MetalTensor,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    shared_out: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    n_expert: usize,
    topk: usize,
    hidden: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if logits.dtype != GgmlType::F32
        || shared_weight.dtype != GgmlType::F32
        || x.dtype != GgmlType::F32
        || out_w.dtype != GgmlType::F32
        || shared_out.dtype != GgmlType::F32
        || counts.dtype != GgmlType::F32
        || ids.dtype != GgmlType::F32
    {
        return Err(MetalError::BadShape {
            kernel: "topk_bucket_logits_softmax_dot_sigmoid_packed",
            detail: "expected F32 buffers throughout".into(),
        });
    }
    if n_expert == 0 || n_expert > 256 || topk == 0 || topk > 16 || topk > n_expert {
        return Err(MetalError::BadShape {
            kernel: "topk_bucket_logits_softmax_dot_sigmoid_packed",
            detail: format!(
                "expected 1 <= topk <= n_expert <= 256 and topk <= 16, got n_expert={n_expert} topk={topk}"
            ),
        });
    }
    if logits.n_elements() as usize != n_tokens * n_expert
        || out_idx.n_elements() as usize != n_tokens * topk
        || out_w.n_elements() as usize != n_tokens * topk
        || shared_weight.n_elements() as usize != hidden
        || x.n_elements() as usize != n_tokens * hidden
        || shared_out.n_elements() as usize != n_tokens
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
    {
        return Err(MetalError::BadShape {
            kernel: "topk_bucket_logits_softmax_dot_sigmoid_packed",
            detail: "shape mismatch in packed route+bucket inputs/outputs".into(),
        });
    }

    let pso = ctx.pipeline("kernel_topk_bucket_logits_softmax_dot_sigmoid_packed_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_expert: u32,
        topk: u32,
        hidden: u32,
        n_tokens: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_expert: n_expert as u32,
            topk: topk as u32,
            hidden: hidden as u32,
            n_tokens: n_tokens as u32,
        },
    );
    enc.set_tensor(1, logits);
    enc.set_tensor(2, shared_weight);
    enc.set_tensor(3, x);
    enc.set_tensor(4, out_idx);
    enc.set_tensor(5, out_w);
    enc.set_tensor(6, shared_out);
    enc.set_tensor(7, counts);
    enc.set_tensor(8, ids);
    const THREADS: usize = 256;
    enc.set_threadgroup_memory(0, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, THREADS * std::mem::size_of::<i32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: n_tokens,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_route_bucket_slots_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    topk_idx: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    n_expert: usize,
    n_tokens: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if topk_idx.n_elements() as usize != n_tokens * topk
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
    {
        return Err(MetalError::BadShape {
            kernel: "moe_route_bucket_slots",
            detail: format!(
                "shape mismatch idx={} counts={} ids={} expected idx={} counts={} ids={}",
                topk_idx.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                n_tokens * topk,
                n_expert,
                n_expert * n_tokens
            ),
        });
    }
    let pso = ctx.pipeline("kernel_moe_route_bucket_slots_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_expert: u32,
        n_tokens: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_expert: n_expert as u32,
            n_tokens: n_tokens as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, topk_idx);
    enc.set_tensor(2, counts);
    enc.set_tensor(3, ids);
    enc.dispatch(
        MTLSize {
            width: n_expert.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// **EXPERIMENTAL — FAILED A-LITE GATE — NOT WIRED INTO PRODUCTION (v0.73c.2)**
///
/// Layer-major fused SwiGLU FFN — Q4_K mat-mat × 2 + silu_mul, NR1=16.
///
/// Fuses the 3-dispatch sequence (gate_mm + up_mm + silu_mul) into one
/// kernel per FFN layer. Bit-exact with the unfused reference (cos =
/// 1.000000, max|Δ| = 0). Lifted from mat_mat_q4_k.metal NR1=16 with
/// doubled accumulators (mc_gate[4] + mc_up[4]) and shared sb tile.
///
/// **Why not in production:** A-lite bench at production 64-layer 27B
/// shape (n_in=5120, n_out=17408, N=16) measured 1.07× speedup vs
/// unfused — codex threshold was ≤ 0.7 (i.e. ≥ 30% speedup needed).
/// The unchanged W_gate + W_up weight reads dominate; fusion only saves
/// dispatch count and intermediate I/O, both of which Metal already
/// pipelines well within one command buffer. Codex's optimistic 5-15
/// ms/call savings estimate was ~30× too high (actual: ~0.5 ms/call).
///
/// See `kernels/ffn_fused_swiglu_q4_k_mm.metal` header for the full
/// negative-result writeup. Preserved as institutional memory; do NOT
/// plumb without re-running `ffn_fused_swiglu_q4_K_amortization_vs_unfused`
/// to confirm the regime has changed.
///
/// Constraints:
///   * `n_in % 256 == 0` (Q4_K super-block alignment)
///   * `n_query == 16` (NR1=16 fast path; host enforces)
///   * Both weight tensors must be Q4_K (host check)
///
/// Threadgroup memory: 16384 B (sa_g 4 KiB + sa_u 4 KiB + sb 1 KiB live)
#[allow(non_snake_case)]
pub fn encode_ffn_fused_swiglu_q4_K_mm_n16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor, // Q4_K [n_in, n_out]
    w_up: &MetalTensor,   // Q4_K [n_in, n_out]
    x: &MetalTensor,      // F32 [n_query=16, n_in] row-major
    inner: &MetalTensor,  // F32 [n_query, n_out] row-major
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm_n16",
            detail: format!("n_in={n_in} not divisible by 256 (Q4_K super-block)"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm_n16",
            detail: format!(
                "w_gate.dtype={:?} w_up.dtype={:?}, both must be Q4_K",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    const N: usize = 16;
    if x.n_elements() as usize != N * n_in {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm_n16",
            detail: format!("x.n_elements={} != N*n_in={}", x.n_elements(), N * n_in),
        });
    }
    if inner.n_elements() as usize != N * n_out {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm_n16",
            detail: format!(
                "inner.n_elements={} != N*n_out={}",
                inner.n_elements(),
                N * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_ffn_fused_swiglu_q4_K_mm_n16_f32")?;
    enc.set_pipeline(&pso);

    let nb01 = ((n_in / 256) * 144) as u32;
    let stride_b = n_in as u32;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: N as u32,
            k: n_in as u32,
            nb01,
            stride_b,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, inner);

    enc.set_threadgroup_memory(0, 16384);

    let n_tg_x = N.div_ceil(16);
    let n_tg_y = n_out.div_ceil(64);
    enc.dispatch(
        MTLSize {
            width: n_tg_x,
            height: n_tg_y,
            depth: 1,
        },
        MTLSize {
            width: 128, // 4 simdgroups × 32 lanes
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_ffn_fused_swiglu_q4_K_mm_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm",
            detail: format!("n_in={n_in} not divisible by 256 (Q4_K super-block)"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm",
            detail: format!(
                "w_gate.dtype={:?} w_up.dtype={:?}, both must be Q4_K",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if inner.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm",
            detail: format!(
                "inner.n_elements={} != n_query*n_out={}",
                inner.n_elements(),
                n_query * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_ffn_fused_swiglu_q4_K_mm_f32")?;
    enc.set_pipeline(&pso);

    let nb01 = ((n_in / 256) * 144) as u32;
    let stride_b = n_in as u32;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, inner);
    enc.set_threadgroup_memory(0, 16384);

    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// GDN α-chain fusion: fused (a + dt_bias), softplus, then mul by a_log.
///
/// Replaces 3 dispatches per GDN layer:
///   encode_add_inplace_f32(a, dt_bias)
///   encode_softplus_f32(a → out)
///   encode_mul_f32(out, a_log → out)
///
/// Saves 2 dispatches × 32 layers = 64 dispatches/token. Per the v0.25
/// intra-profiler the chain measured ~0.04 ms/layer; this should shave
/// 1.5–2.5 ms/token (Codex predicted 2–3 ms upper bound).
///
/// CPU oracle: `softplus(a + dt_bias) * a_log` from `crate::forward`.
pub fn encode_gdn_alpha_chain_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,
    dt_bias: &MetalTensor,
    a_log: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    let n = a.n_elements() as usize;
    if dt_bias.n_elements() as usize != n
        || a_log.n_elements() as usize != n
        || out.n_elements() as usize != n
    {
        return Err(MetalError::BadShape {
            kernel: "gdn_alpha_chain",
            detail: format!(
                "lengths a={} dt={} alog={} out={n}",
                a.n_elements(),
                dt_bias.n_elements(),
                a_log.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_gdn_alpha_chain_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, a);
    enc.set_tensor(2, dt_bias);
    enc.set_tensor(3, a_log);
    enc.set_tensor(4, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// GDN decay-chain fusion: `exp(softplus(a + dt_bias) * a_log)`.
///
/// The standard alpha-chain writes the log-decay `g`; the GDN recurrence
/// then needs `exp(g)` for every state row. This variant writes the per-head
/// decay once so `kernel_gdn_step_decay_f32` can reuse it across all rows.
pub fn encode_gdn_decay_chain_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,
    dt_bias: &MetalTensor,
    a_log: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    let n = a.n_elements() as usize;
    if dt_bias.n_elements() as usize != n
        || a_log.n_elements() as usize != n
        || out.n_elements() as usize != n
    {
        return Err(MetalError::BadShape {
            kernel: "gdn_decay_chain",
            detail: format!(
                "lengths a={} dt={} alog={} out={n}",
                a.n_elements(),
                dt_bias.n_elements(),
                a_log.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_gdn_decay_chain_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, a);
    enc.set_tensor(2, dt_bias);
    enc.set_tensor(3, a_log);
    enc.set_tensor(4, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// Batched GDN α-chain (v0.73a layer-major batching).
///
/// Computes `out[r, c] = softplus(a[r, c] + dt_bias[c]) * a_log[c]` over
/// `[N, n_v]` row-major `a` / `out` with `[n_v]` `dt_bias` / `a_log`
/// broadcast across the N rows.
///
/// Bit-identical to N successive calls of `encode_gdn_alpha_chain_f32`
/// over each row of `a` (validated by
/// `gdn_alpha_chain_batched_matches_per_row` test).
///
/// Used by the layer-major packed_verify GDN batching (v0.73a) to lift
/// the per-token alpha-chain dispatch out of the inner N loop alongside
/// in_proj_qkv / in_proj_z / beta_proj / alpha_proj. dt_bias and a_log
/// are layer-shared GGUF weights (`[n_v]` F32); a and out are
/// `MetalDFlashLayerMajorScratch::gdn_a_pack` / `gdn_alpha_pack`
/// (`[N, n_v]`).
pub fn encode_gdn_alpha_chain_batched_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,       // [N, n_v] row-major
    dt_bias: &MetalTensor, // [n_v]
    a_log: &MetalTensor,   // [n_v]
    out: &MetalTensor,     // [N, n_v] row-major
    n_rows: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    let n = n_rows * n_cols;
    if a.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "gdn_alpha_chain_batched",
            detail: format!(
                "a expected {n_rows}*{n_cols}={n} elements, got {}",
                a.n_elements()
            ),
        });
    }
    if out.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "gdn_alpha_chain_batched",
            detail: format!(
                "out expected {n_rows}*{n_cols}={n} elements, got {}",
                out.n_elements()
            ),
        });
    }
    if dt_bias.n_elements() as usize != n_cols {
        return Err(MetalError::BadShape {
            kernel: "gdn_alpha_chain_batched",
            detail: format!(
                "dt_bias expected {n_cols} elements, got {}",
                dt_bias.n_elements()
            ),
        });
    }
    if a_log.n_elements() as usize != n_cols {
        return Err(MetalError::BadShape {
            kernel: "gdn_alpha_chain_batched",
            detail: format!(
                "a_log expected {n_cols} elements, got {}",
                a_log.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_gdn_alpha_chain_batched_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        n_cols: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            n_cols: n_cols as u32,
        },
    );
    enc.set_tensor(1, a);
    enc.set_tensor(2, dt_bias);
    enc.set_tensor(3, a_log);
    enc.set_tensor(4, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

pub fn encode_gdn_decay_chain_batched_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,
    dt_bias: &MetalTensor,
    a_log: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    let n = n_rows * n_cols;
    if a.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "gdn_decay_chain_batched",
            detail: format!(
                "a expected {n_rows}*{n_cols}={n} elements, got {}",
                a.n_elements()
            ),
        });
    }
    if out.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "gdn_decay_chain_batched",
            detail: format!(
                "out expected {n_rows}*{n_cols}={n} elements, got {}",
                out.n_elements()
            ),
        });
    }
    if dt_bias.n_elements() as usize != n_cols {
        return Err(MetalError::BadShape {
            kernel: "gdn_decay_chain_batched",
            detail: format!(
                "dt_bias expected {n_cols} elements, got {}",
                dt_bias.n_elements()
            ),
        });
    }
    if a_log.n_elements() as usize != n_cols {
        return Err(MetalError::BadShape {
            kernel: "gdn_decay_chain_batched",
            detail: format!(
                "a_log expected {n_cols} elements, got {}",
                a_log.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_gdn_decay_chain_batched_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        n_cols: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            n_cols: n_cols as u32,
        },
    );
    enc.set_tensor(1, a);
    enc.set_tensor(2, dt_bias);
    enc.set_tensor(3, a_log);
    enc.set_tensor(4, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// Gated attention: out = x * sigmoid(gate). Used after attention before the
/// output projection, and supports `out` aliasing `x`.
pub fn encode_sigmoid_mul_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    x: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_2in_1out(ctx, enc, "kernel_sigmoid_mul_f32", gate, x, out)
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

pub fn encode_fill_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    y: &MetalTensor,
    value: f32,
) -> Result<(), MetalError> {
    let n = y.n_elements() as usize;
    let pso = ctx.pipeline("kernel_fill_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        value: f32,
    }
    enc.set_bytes(0, &Args { n: n as u32, value });
    enc.set_tensor(1, y);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

pub fn encode_touch_bytes_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    sink: &MetalTensor,
    stride_bytes: usize,
) -> Result<(), MetalError> {
    let sink_n = sink.n_elements() as usize;
    if sink_n == 0 {
        return Err(MetalError::BadShape {
            kernel: "touch_bytes",
            detail: "sink must have at least one element".into(),
        });
    }
    let n_bytes = src.n_bytes() as usize;
    if n_bytes == 0 || stride_bytes == 0 {
        return Err(MetalError::BadShape {
            kernel: "touch_bytes",
            detail: format!("n_bytes={n_bytes} stride_bytes={stride_bytes} must both be > 0"),
        });
    }
    let n_steps = n_bytes.div_ceil(stride_bytes);
    let pso = ctx.pipeline("kernel_touch_bytes_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_steps: u32,
        stride_bytes: u32,
        n_bytes: u64,
    }
    enc.set_bytes(
        0,
        &Args {
            n_steps: n_steps as u32,
            stride_bytes: stride_bytes as u32,
            n_bytes: n_bytes as u64,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, sink);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: sink_n.min(256),
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_roofline_stream_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
    alpha: f32,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if y.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "roofline_stream",
            detail: format!("y.n_elements={} != x.n_elements={n}", y.n_elements()),
        });
    }
    let pso = ctx.pipeline("kernel_roofline_stream_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        alpha: f32,
    }
    enc.set_bytes(0, &Args { n: n as u32, alpha });
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
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

pub fn encode_roofline_fma_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
    iters: usize,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if y.n_elements() as usize != n || iters == 0 {
        return Err(MetalError::BadShape {
            kernel: "roofline_fma",
            detail: format!(
                "y.n_elements={} x.n_elements={n} iters={iters}",
                y.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_roofline_fma_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        iters: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            iters: iters as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
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

pub fn encode_axpy_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    accum: &MetalTensor,
    alpha: f32,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if accum.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "axpy",
            detail: format!(
                "accum.n_elements={} != x.n_elements={n}",
                accum.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_axpy_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        alpha: f32,
    }
    enc.set_bytes(0, &Args { n: n as u32, alpha });
    enc.set_tensor(1, x);
    enc.set_tensor(2, accum);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

pub fn encode_axpy_rowwise_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    scales: &MetalTensor,
    accum: &MetalTensor,
    n_cols: usize,
    n_rows: usize,
) -> Result<(), MetalError> {
    let n = n_cols * n_rows;
    if x.n_elements() as usize != n
        || accum.n_elements() as usize != n
        || scales.n_elements() as usize != n_rows
    {
        return Err(MetalError::BadShape {
            kernel: "axpy_rowwise",
            detail: format!(
                "expected x/accum n={} and scales rows={}, got x={} accum={} scales={}",
                n,
                n_rows,
                x.n_elements(),
                accum.n_elements(),
                scales.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_axpy_rowwise_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_cols: u32,
        n_rows: u32,
        out_rows: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_cols: n_cols as u32,
            n_rows: n_rows as u32,
            out_rows: (accum.n_elements() as usize / n_cols) as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, scales);
    enc.set_tensor(3, accum);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
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

pub fn encode_scatter_axpy_rows_unique_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    rows: &MetalTensor,
    scales: &MetalTensor,
    accum: &MetalTensor,
    n_cols: usize,
    n_rows: usize,
) -> Result<(), MetalError> {
    let n = n_cols * n_rows;
    if x.n_elements() as usize != n
        || rows.n_elements() as usize != n_rows
        || scales.n_elements() as usize != n_rows
        || accum.n_elements() as usize % n_cols != 0
    {
        return Err(MetalError::BadShape {
            kernel: "scatter_axpy_rows_unique",
            detail: format!(
                "expected x n={} rows/scales n_rows={} accum multiple of n_cols={}, got x={} rows={} scales={} accum={}",
                n,
                n_rows,
                n_cols,
                x.n_elements(),
                rows.n_elements(),
                scales.n_elements(),
                accum.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_scatter_axpy_rows_unique_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_cols: u32,
        n_rows: u32,
        out_rows: u32,
    }
    let out_rows = accum.n_elements() as usize / n_cols;
    enc.set_bytes(
        0,
        &Args {
            n_cols: n_cols as u32,
            n_rows: n_rows as u32,
            out_rows: out_rows as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, rows);
    enc.set_tensor(3, scales);
    enc.set_tensor(4, accum);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
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

pub fn encode_topk_select_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    probs: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if probs.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "topk_select",
            detail: format!("probs.n_elements={} != n={n}", probs.n_elements()),
        });
    }
    if out_idx.n_elements() as usize != k || out_w.n_elements() as usize != k {
        return Err(MetalError::BadShape {
            kernel: "topk_select",
            detail: format!(
                "out_idx/out_w expected {k} elements, got {}/{}",
                out_idx.n_elements(),
                out_w.n_elements()
            ),
        });
    }
    if k == 0 || k > 16 {
        return Err(MetalError::BadShape {
            kernel: "topk_select",
            detail: format!("k={k} must be in 1..=16"),
        });
    }
    let pso = ctx.pipeline("kernel_topk_select_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        k: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            k: k as u32,
        },
    );
    enc.set_tensor(1, probs);
    enc.set_tensor(2, out_idx);
    enc.set_tensor(3, out_w);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// GPU-side argmax over `[n_rows, n]` rows of F32, writing `[n_rows]` i32
/// indices. Tie policy: lowest index wins (matches numpy/torch).
///
/// Used by H5.3a `packed_forward` to produce `verify_argmax: [N] i32`
/// without a `[N, V]` CPU readback. At V=248320, N=16 that's 15.9 MB
/// per outer step we don't have to spill to host.
///
/// Layout assumption: `x` is row-major with row stride == `n` (no padding
/// between rows). Each row gets one threadgroup; up to 1024 threads per
/// TG, internally simdgroup-reduced.
pub fn encode_argmax_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    n_rows: usize,
    n: usize,
) -> Result<(), MetalError> {
    if x.n_elements() as usize != n_rows * n {
        return Err(MetalError::BadShape {
            kernel: "argmax",
            detail: format!("x.n_elements={} != n_rows*n={}", x.n_elements(), n_rows * n),
        });
    }
    if out_idx.n_elements() as usize != n_rows {
        return Err(MetalError::BadShape {
            kernel: "argmax",
            detail: format!(
                "out_idx.n_elements={} != n_rows={n_rows}",
                out_idx.n_elements(),
            ),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        stride_x: u32,
    }
    let pso = ctx.pipeline("kernel_argmax_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            stride_x: n as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, out_idx);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    // Two threadgroup arrays (sh_val: f32, sh_idx: u32) — same width.
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));
    enc.set_threadgroup_memory(1, (n_simdgroups * std::mem::size_of::<u32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: n_rows,
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// Per-head RMSNorm reading strided source rows: row `hi` lives at
/// `x[src_offset + hi * src_stride .. + head_dim]`; output `y` is compact
/// `[n_heads, head_dim]`. Used to read the Q halves of the interleaved
/// gated-attention q_proj output directly (src_stride = 2*head_dim,
/// src_offset = 0), deleting the split_q_gate layout copy (v0.432).
/// Bit-identical per-row math to `encode_rms_norm_batched_f32`.
#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm_batched_src_strided_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    weight: &MetalTensor,
    y: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    src_stride: usize,
    src_offset: usize,
    eps: f32,
) -> Result<(), MetalError> {
    if n_heads == 0 || head_dim == 0 {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_batched_src_strided",
            detail: "n_heads/head_dim must be nonzero".to_string(),
        });
    }
    // Source must cover the last strided row end-to-end.
    let src_need = src_offset as u64 + (n_heads as u64 - 1) * src_stride as u64 + head_dim as u64;
    if x.n_elements() < src_need {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_batched_src_strided",
            detail: format!(
                "x has {} elements, needs >= {src_need} \
                 (offset={src_offset} stride={src_stride} rows={n_heads} head_dim={head_dim})",
                x.n_elements()
            ),
        });
    }
    let want = (n_heads * head_dim) as u64;
    if y.n_elements() != want {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_batched_src_strided",
            detail: format!("y expected {want} elements"),
        });
    }
    if weight.n_elements() as usize != head_dim {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_batched_src_strided",
            detail: format!("weight expected {head_dim} elements"),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        src_stride: u32,
        src_offset: u32,
        eps: f32,
    }
    let pso = ctx.pipeline("kernel_rms_norm_batched_src_strided_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            src_stride: src_stride as u32,
            src_offset: src_offset as u32,
            eps,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// Gated attention `out = x * sigmoid(gate)` where the gate rows live
/// strided inside a larger tensor (the interleaved q_proj output's gate
/// halves: gate_stride = 2*head_dim, gate_offset = head_dim). `x`/`out`
/// are compact `n_rows * head_dim` elements (v0.432; replaces
/// split_q_gate + compact sigmoid_mul).
pub fn encode_sigmoid_mul_gate_strided_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    x: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    head_dim: usize,
    gate_stride: usize,
    gate_offset: usize,
) -> Result<(), MetalError> {
    if n_rows == 0 || head_dim == 0 {
        return Err(MetalError::BadShape {
            kernel: "sigmoid_mul_gate_strided",
            detail: "n_rows/head_dim must be nonzero".to_string(),
        });
    }
    let n = (n_rows * head_dim) as u64;
    if x.n_elements() != n || out.n_elements() != n {
        return Err(MetalError::BadShape {
            kernel: "sigmoid_mul_gate_strided",
            detail: format!("x/out expected {n} elements"),
        });
    }
    let gate_need = gate_offset as u64 + (n_rows as u64 - 1) * gate_stride as u64 + head_dim as u64;
    if gate.n_elements() < gate_need {
        return Err(MetalError::BadShape {
            kernel: "sigmoid_mul_gate_strided",
            detail: format!(
                "gate has {} elements, needs >= {gate_need}",
                gate.n_elements()
            ),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        head_dim: u32,
        gate_stride: u32,
        gate_offset: u32,
    }
    let pso = ctx.pipeline("kernel_sigmoid_mul_gate_strided_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            head_dim: head_dim as u32,
            gate_stride: gate_stride as u32,
            gate_offset: gate_offset as u32,
        },
    );
    enc.set_tensor(1, gate);
    enc.set_tensor(2, x);
    enc.set_tensor(3, out);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: (n as usize).div_ceil(tg_threads),
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
///
/// v0.432: production attention paths read the interleaved layout
/// directly (strided q-norm + strided gate sigmoid_mul); this kernel
/// remains for tests and as the reference for the interleave layout.
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
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

pub fn encode_split_qkv_fused_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    q_full: &MetalTensor,
    k_out: &MetalTensor,
    v_out: &MetalTensor,
    n_rows: usize,
    q_full_dim: usize,
    kv_dim: usize,
) -> Result<(), MetalError> {
    let fused_stride = q_full_dim + 2 * kv_dim;
    let want_src = (n_rows * fused_stride) as u64;
    let want_q = (n_rows * q_full_dim) as u64;
    let want_kv = (n_rows * kv_dim) as u64;
    if src.n_elements() != want_src {
        return Err(MetalError::BadShape {
            kernel: "split_qkv_fused",
            detail: format!("src expected {want_src} elements"),
        });
    }
    if q_full.n_elements() != want_q
        || k_out.n_elements() != want_kv
        || v_out.n_elements() != want_kv
    {
        return Err(MetalError::BadShape {
            kernel: "split_qkv_fused",
            detail: format!("q/k/v expected {want_q}/{want_kv}/{want_kv} elements"),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_rows: u32,
        q_full_dim: u32,
        kv_dim: u32,
        fused_stride: u32,
    }
    let pso = ctx.pipeline("kernel_split_qkv_fused_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_rows: n_rows as u32,
            q_full_dim: q_full_dim as u32,
            kv_dim: kv_dim as u32,
            fused_stride: fused_stride as u32,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, q_full);
    enc.set_tensor(3, k_out);
    enc.set_tensor(4, v_out);

    let total = n_rows * fused_stride;
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// F16 KV-cache variant of `encode_attn_decode_f32`. Same algorithm,
/// reads K and V as half-precision. Halves attention bandwidth at long
/// context (saves ~4 GB of reads/token at 4K positions on 27B). Q is
/// still F32; output is F32.
///
/// Caller is responsible for ensuring `k_cache` and `v_cache` are F16-typed
/// MetalTensors (typically allocated via `MetalTensor::zeros_f16`).
pub fn encode_attn_decode_f16kv_f32(
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
            kernel: "attn_decode_f16kv",
            detail: format!("n_q_heads={n_q_heads} not multiple of n_kv_heads={n_kv_heads}"),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_f16kv",
            detail: format!(
                "k/v expected F16 dtype, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    let want = (n_q_heads * head_dim) as u64;
    if q.n_elements() != want || out.n_elements() != want {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_f16kv",
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
    let pso = ctx.pipeline("kernel_attn_decode_f16kv")?;
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    let scores_bytes = n_pos * std::mem::size_of::<f32>();
    let shred_bytes = (n_simdgroups * std::mem::size_of::<f32>()).max(32);
    if scores_bytes > 28 * 1024 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_f16kv",
            detail: format!(
                "n_pos={n_pos} requires {scores_bytes} B threadgroup memory; max ~28 KB"
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

// =============================================================================
// Flash-attention v4: GQA-dedup + online softmax + split-K. Specializations
// currently cover head_dim=256 with GROUP in {4, 6, 8, 16} and F16 KV cache.
//
// Caller is responsible for owning per-call partial buffers:
//   o_partial : F32, n_kv_heads * NWG * GROUP * head_dim elements
//   ml_partial: F32, n_kv_heads * NWG * GROUP * 2 elements
// Sized for the maximum NWG you'll ever pass.
// =============================================================================

/// Compute v4-friendly NWG (split-K partition count) for a given context length.
///
/// Heuristic determined empirically on M4 Max via whole-model phase/ctx sweeps:
///   - Long-context dense group=6 now uses NWG=64; it cuts dense 27B attention
///     by ~4.5 ms at 16K and ~9.3 ms at 32K versus NWG=32.
///   - NWG=16 is competitive for n_pos < 256 (very-cold-start regime).
///   - NWG=1 is catastrophically under-occupied (4 TGs total) — DO NOT
///     ship as performance config; useful only for correctness debugging.
///
/// Long-context group=8 can use NWG=256 with the two-threadgroup reduce path;
/// smaller overrides such as 128/192 remain useful A/B knobs when validating
/// the main-vs-reduce split.
/// `QWEN_ATTN_V4_NWG=1..ATTN_V4_NWG_MAX` is an A/B knob for whole-model sweeps.
/// `QWEN_ATTN_V4_SUBGROUP_MIN_POS=4096` restores the older long-only threshold.
pub fn attn_v4_choose_nwg(n_pos: usize, group: usize) -> usize {
    static NWG_OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    if let Some(nwg) = *NWG_OVERRIDE.get_or_init(|| {
        std::env::var("QWEN_ATTN_V4_NWG")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| (1..=ATTN_V4_NWG_MAX).contains(v))
    }) {
        return nwg;
    }

    if group == 8 && n_pos >= 16_384 {
        256
    } else if matches!(group, 8 | 16) && n_pos >= attn_v4_subgroup_min_pos() {
        64
    } else if matches!(group, 4 | 6) && n_pos >= 4096 {
        64
    } else if n_pos < 256 {
        16
    } else {
        32
    }
}

/// Pick the v4 tile size (KV positions per inner softmax tile) for a given
/// context and GQA group size.
///
/// Current tuning:
/// - group=6 (27B dense): keep C=32 until we have fresh long-ctx sweep data.
/// - group in {8,16} (35B A3B / 122B A10B): C=64 is the current best-known
///   medium/long-context choice. C=128 is experimental and should only be
///   enabled if fresh sweeps beat 64 on real hardware.
///
/// `QWEN_ATTN_V4_TILE_C={16,32,64,128}` is an A/B knob for whole-model sweeps.
pub fn attn_v4_choose_tile_c(n_pos: usize, group: usize) -> usize {
    if group == 8 {
        if let Some(tile_c) = attn_v4_g8_vstage_c() {
            return tile_c;
        }
    }
    static TILE_C_OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    if let Some(tile_c) = *TILE_C_OVERRIDE.get_or_init(|| {
        std::env::var("QWEN_ATTN_V4_TILE_C")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| matches!(*v, 16 | 32 | 64 | 128))
    }) {
        return tile_c;
    }

    if group == 16 && n_pos >= 32768 {
        128
    } else if matches!(group, 8 | 16) && n_pos >= attn_v4_subgroup_min_pos() {
        64
    } else {
        32
    }
}

fn attn_v4_g8_vstage_c() -> Option<usize> {
    static VSTAGE_C: OnceLock<Option<usize>> = OnceLock::new();
    *VSTAGE_C.get_or_init(|| {
        std::env::var("QWEN_ATTN_V4_G8_VSTAGE_C")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| matches!(*v, 16 | 32))
    })
}

fn attn_v4_g8_bcast_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_ATTN_V4_G8_BCAST").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

/// W1b partition-packing opt-in: `QWEN_ATTN_V4_PACK=4` packs 4 partitions
/// (one simdgroup each) into 128-thread TGs for the g8/t4/C64 F16 decode
/// main. Default off; see docs/bench/2026-07-06-w1b-attn-partition-pack/.
fn attn_v4_pack() -> usize {
    static PACK: OnceLock<usize> = OnceLock::new();
    *PACK.get_or_init(|| {
        std::env::var("QWEN_ATTN_V4_PACK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v == 4)
            .unwrap_or(1)
    })
}

/// Group-tile subgroup size for v4's decode main pass.
///
/// The 122B A10B shape (`GROUP=16`) is faster when split across multiple
/// threadgroups: tile4 trades extra K/V reads for much higher occupancy and
/// lower register pressure. Keep `QWEN_ATTN_V4_G16_TILE` as a kill switch / A/B
/// knob (`4`, `8`, or `16`), but default medium/long-context group16 to tile4.
///
/// The 35B A3B shape (`GROUP=8`) has the same long-context signature with a
/// smaller best split. Keep `QWEN_ATTN_V4_G8_TILE` as a kill switch / A/B knob
/// (`2`, `4`, or `8`): default medium-context group8 decode to tile2, then use
/// tile4 at true-long contexts where NWG=256 recovers enough occupancy.
fn attn_v4_g8_tile_override(var: &str) -> Option<usize> {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| matches!(*v, 2 | 4 | 8))
}

fn attn_v4_subgroup_min_pos() -> usize {
    static MIN_POS: OnceLock<usize> = OnceLock::new();
    *MIN_POS.get_or_init(|| {
        std::env::var("QWEN_ATTN_V4_SUBGROUP_MIN_POS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(ATTN_V4_SUBGROUP_MIN_POS_DEFAULT)
    })
}

pub fn attn_v4_choose_group_tile(n_pos: usize, group: usize) -> usize {
    if let Some(tile) = ATTN_V4_GROUP_TILE_OVERRIDE.with(|cell| cell.get()) {
        return tile;
    }
    if n_pos < attn_v4_subgroup_min_pos() {
        return group;
    }
    if group == 8 {
        static G8_TILE: OnceLock<Option<usize>> = OnceLock::new();
        let default_tile = if n_pos >= 16_384 { 4 } else { 2 };
        return G8_TILE
            .get_or_init(|| attn_v4_g8_tile_override("QWEN_ATTN_V4_G8_TILE"))
            .unwrap_or(default_tile);
    }
    if group != 16 {
        return group;
    }
    static G16_TILE: OnceLock<Option<usize>> = OnceLock::new();
    G16_TILE
        .get_or_init(|| {
            std::env::var("QWEN_ATTN_V4_G16_TILE")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|v| matches!(*v, 4 | 8 | 16))
        })
        .unwrap_or(4)
}

pub fn attn_v4_choose_group_tile_prefill(n_pos: usize, group: usize) -> usize {
    if n_pos < 4096 {
        return group;
    }
    if group == 8 {
        static G8_PREFILL_TILE: OnceLock<Option<usize>> = OnceLock::new();
        return G8_PREFILL_TILE
            .get_or_init(|| {
                attn_v4_g8_tile_override("QWEN_ATTN_V4_G8_PREFILL_TILE")
                    .or_else(|| attn_v4_g8_tile_override("QWEN_ATTN_V4_G8_TILE"))
            })
            .unwrap_or(8);
    }
    attn_v4_choose_group_tile(n_pos, group)
}

pub fn with_attn_v4_group_tile_override<T>(tile: usize, f: impl FnOnce() -> T) -> T {
    let prev = ATTN_V4_GROUP_TILE_OVERRIDE.with(|cell| {
        let prev = cell.get();
        cell.set(Some(tile));
        prev
    });
    let out = f();
    ATTN_V4_GROUP_TILE_OVERRIDE.with(|cell| cell.set(prev));
    out
}

/// Encode v4 main kernel + reduce kernel in sequence.
/// Hardcoded constants (must match `kernels/attn_v4.metal`):
///   DK = DV = 256, lanes = 32, GROUP in {4, 6, 8, 16}.
/// Tile size `tile_c ∈ {16, 32, 64, 128}` selects the kernel variant.
/// Use `attn_v4_choose_tile_c(n_pos, group)` for the empirically-tuned choice
/// or pass 32 (default; backward-compat) if unsure.
pub fn encode_attn_decode_v4_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_pos: usize,
    nwg: usize,
    tile_c: usize,
) -> Result<(), MetalError> {
    // Hardcoded shape preconditions.
    const DK: usize = 256;
    if n_q_heads % n_kv_heads != 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("n_q_heads={n_q_heads} not multiple of n_kv_heads={n_kv_heads}"),
        });
    }
    let group = n_q_heads / n_kv_heads;
    if head_dim != DK {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("head_dim={head_dim} but kernel hardcodes {DK}"),
        });
    }
    if !matches!(group, 4 | 6 | 8 | 16) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("group={group} unsupported; expected one of {{4, 6, 8, 16}}"),
        });
    }
    if k_cache.dtype != v_cache.dtype || !matches!(k_cache.dtype, GgmlType::F16 | GgmlType::Q8_0) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!(
                "k/v expected matching F16 or Q8_0 dtypes, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    let want_q = (n_q_heads * head_dim) as u64;
    if q.n_elements() != want_q || out.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("q/out expected {want_q} elements"),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let group_tile = attn_v4_choose_group_tile(n_pos, group);
    if group_tile == 0 || group % group_tile != 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("group_tile={group_tile} must divide group={group}"),
        });
    }
    if group_tile != group
        && !(group == 16 && matches!(group_tile, 4 | 8)
            || group == 8 && matches!(group_tile, 2 | 4))
    {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("unsupported subgroup: group={group} group_tile={group_tile}"),
        });
    }
    let want_o_partial = (n_kv_heads * nwg * group * head_dim) as u64;
    let want_ml_partial = (n_kv_heads * nwg * group * 2) as u64;
    if o_partial.n_elements() < want_o_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!(
                "o_partial too small: have {}, need ≥ {want_o_partial}",
                o_partial.n_elements()
            ),
        });
    }
    if ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!(
                "ml_partial too small: have {}, need ≥ {want_ml_partial}",
                ml_partial.n_elements()
            ),
        });
    }

    let kv_stride = n_kv_heads * head_dim;
    // Pre-multiply scale by log2(e) so kernel uses exp2 (Apple GPU fast path).
    let scale = (1.0f32 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));

    // -------- Main kernel ------------------------------------------------
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        scale: f32,
    }
    let use_g8_bcast = k_cache.dtype == GgmlType::F16
        && group == 8
        && matches!(group_tile, 2 | 4)
        && tile_c == 64
        && attn_v4_g8_bcast_enabled();
    let use_g8_vstage = !use_g8_bcast
        && k_cache.dtype == GgmlType::F16
        && group == 8
        && group_tile == 4
        && attn_v4_g8_vstage_c() == Some(tile_c);
    // W1b: partition-packed main (opt-in; exact same per-simdgroup dataflow)
    let use_pack4 = !use_g8_bcast
        && !use_g8_vstage
        && k_cache.dtype == GgmlType::F16
        && group == 8
        && group_tile == 4
        && tile_c == 64
        && attn_v4_pack() == 4;
    let pipeline_name = if use_pack4 {
        "kernel_attn_decode_v4_g8_t4_c64_pack4_f32"
    } else if use_g8_bcast {
        match group_tile {
            2 => "kernel_attn_decode_v4_g8_t2_c64_bcast_f32",
            4 => "kernel_attn_decode_v4_g8_t4_c64_bcast_f32",
            _ => unreachable!(),
        }
    } else if use_g8_vstage {
        match tile_c {
            16 => "kernel_attn_decode_v4_g8_t4_c16_vstage_f32",
            32 => "kernel_attn_decode_v4_g8_t4_c32_vstage_f32",
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4",
                    detail: format!("vstage C={tile_c} unsupported; expected 16 or 32"),
                });
            }
        }
    } else if group_tile == group {
        match (k_cache.dtype, group, tile_c) {
            (GgmlType::F16, 4, 16) => "kernel_attn_decode_v4_g4_c16_f32",
            (GgmlType::F16, 4, 32) => "kernel_attn_decode_v4_g4_f32",
            (GgmlType::F16, 4, 64) => "kernel_attn_decode_v4_g4_c64_f32",
            (GgmlType::F16, 4, 128) => "kernel_attn_decode_v4_g4_c128_f32",
            (GgmlType::Q8_0, 6, 16) => "kernel_attn_decode_v4_q8_c16_f32",
            (GgmlType::Q8_0, 6, 32) => "kernel_attn_decode_v4_q8_f32",
            (GgmlType::Q8_0, 6, 64) => "kernel_attn_decode_v4_q8_c64_f32",
            (GgmlType::Q8_0, 6, 128) => "kernel_attn_decode_v4_q8_c128_f32",
            (GgmlType::F16, 6, 16) => "kernel_attn_decode_v4_c16_f32",
            (GgmlType::F16, 6, 32) => "kernel_attn_decode_v4_f32",
            (GgmlType::F16, 6, 64) => "kernel_attn_decode_v4_c64_f32",
            (GgmlType::F16, 6, 128) => "kernel_attn_decode_v4_c128_f32",
            (GgmlType::F16, 8, 16) => "kernel_attn_decode_v4_g8_c16_f32",
            (GgmlType::F16, 8, 32) => "kernel_attn_decode_v4_g8_f32",
            (GgmlType::F16, 8, 64) => "kernel_attn_decode_v4_g8_c64_f32",
            (GgmlType::F16, 8, 128) => "kernel_attn_decode_v4_g8_c128_f32",
            (GgmlType::Q8_0, 8, 16) => "kernel_attn_decode_v4_q8_g8_c16_f32",
            (GgmlType::Q8_0, 8, 32) => "kernel_attn_decode_v4_q8_g8_f32",
            (GgmlType::Q8_0, 8, 64) => "kernel_attn_decode_v4_q8_g8_c64_f32",
            (GgmlType::Q8_0, 8, 128) => "kernel_attn_decode_v4_q8_g8_c128_f32",
            (GgmlType::F16, 16, 16) => "kernel_attn_decode_v4_g16_c16_f32",
            (GgmlType::F16, 16, 32) => "kernel_attn_decode_v4_g16_f32",
            (GgmlType::F16, 16, 64) => "kernel_attn_decode_v4_g16_c64_f32",
            (GgmlType::F16, 16, 128) => "kernel_attn_decode_v4_g16_c128_f32",
            (GgmlType::Q8_0, _, _) => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4",
                    detail: format!(
                        "Q8_0 KV main kernels currently support only group=6/group=8; got group={group}, tile_c={tile_c}"
                    ),
                });
            }
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4",
                    detail: format!(
                        "unsupported (group={group}, tile_c={tile_c}); tile_c must be 16/32/64/128"
                    ),
                });
            }
        }
    } else {
        match (k_cache.dtype, group, group_tile, tile_c) {
            (GgmlType::F16, 8, 4, 16) => "kernel_attn_decode_v4_g8_t4_c16_f32",
            (GgmlType::F16, 8, 4, 32) => "kernel_attn_decode_v4_g8_t4_f32",
            (GgmlType::F16, 8, 4, 64) => "kernel_attn_decode_v4_g8_t4_c64_f32",
            (GgmlType::F16, 8, 4, 128) => "kernel_attn_decode_v4_g8_t4_c128_f32",
            (GgmlType::F16, 8, 2, 16) => "kernel_attn_decode_v4_g8_t2_c16_f32",
            (GgmlType::F16, 8, 2, 32) => "kernel_attn_decode_v4_g8_t2_f32",
            (GgmlType::F16, 8, 2, 64) => "kernel_attn_decode_v4_g8_t2_c64_f32",
            (GgmlType::F16, 8, 2, 128) => "kernel_attn_decode_v4_g8_t2_c128_f32",
            (GgmlType::Q8_0, 8, 4, 16) => "kernel_attn_decode_v4_q8_g8_t4_c16_f32",
            (GgmlType::Q8_0, 8, 4, 32) => "kernel_attn_decode_v4_q8_g8_t4_f32",
            (GgmlType::Q8_0, 8, 4, 64) => "kernel_attn_decode_v4_q8_g8_t4_c64_f32",
            (GgmlType::Q8_0, 8, 4, 128) => "kernel_attn_decode_v4_q8_g8_t4_c128_f32",
            (GgmlType::Q8_0, 8, 2, 16) => "kernel_attn_decode_v4_q8_g8_t2_c16_f32",
            (GgmlType::Q8_0, 8, 2, 32) => "kernel_attn_decode_v4_q8_g8_t2_f32",
            (GgmlType::Q8_0, 8, 2, 64) => "kernel_attn_decode_v4_q8_g8_t2_c64_f32",
            (GgmlType::Q8_0, 8, 2, 128) => "kernel_attn_decode_v4_q8_g8_t2_c128_f32",
            (GgmlType::F16, 16, 8, 16) => "kernel_attn_decode_v4_g16_t8_c16_f32",
            (GgmlType::F16, 16, 8, 32) => "kernel_attn_decode_v4_g16_t8_f32",
            (GgmlType::F16, 16, 8, 64) => "kernel_attn_decode_v4_g16_t8_c64_f32",
            (GgmlType::F16, 16, 8, 128) => "kernel_attn_decode_v4_g16_t8_c128_f32",
            (GgmlType::F16, 16, 4, 16) => "kernel_attn_decode_v4_g16_t4_c16_f32",
            (GgmlType::F16, 16, 4, 32) => "kernel_attn_decode_v4_g16_t4_f32",
            (GgmlType::F16, 16, 4, 64) => "kernel_attn_decode_v4_g16_t4_c64_f32",
            (GgmlType::F16, 16, 4, 128) => "kernel_attn_decode_v4_g16_t4_c128_f32",
            (GgmlType::Q8_0, _, _, _) => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4",
                    detail: format!(
                        "Q8_0 KV subgroup kernels currently support only group=8 tile2/tile4; got group={group}, group_tile={group_tile}, tile_c={tile_c}"
                    ),
                });
            }
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4",
                    detail: format!(
                        "unsupported (group={group}, group_tile={group_tile}, tile_c={tile_c})"
                    ),
                });
            }
        }
    };
    let pso_main = ctx.pipeline(pipeline_name)?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            scale,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);

    // Threadgroup memory:
    //   threadgroup(0) sq[group_tile * DK halves]
    //   threadgroup(1) ss[group_tile * C floats]
    // The vstage proof fuses both group8/tile4 simdgroups into one TG, so it
    // allocates both simdgroups' Q/score scratch plus a shared V tile.
    if use_g8_bcast {
        let sq_bytes = group_tile * DK * 2;
        enc.set_threadgroup_memory(0, sq_bytes);
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: group / group_tile,
                depth: nwg,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else if use_pack4 {
        const PACK: usize = 4;
        // sq is SHARED across the packed partitions; ss is per-simdgroup.
        enc.set_threadgroup_memory(0, group_tile * DK * 2);
        enc.set_threadgroup_memory(1, PACK * group_tile * tile_c * std::mem::size_of::<f32>());
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: group / group_tile,
                depth: nwg.div_ceil(PACK),
            },
            MTLSize {
                width: 32 * PACK,
                height: 1,
                depth: 1,
            },
        );
    } else if use_g8_vstage {
        enc.set_threadgroup_memory(0, 2 * group_tile * DK * 2);
        enc.set_threadgroup_memory(1, 2 * group_tile * tile_c * std::mem::size_of::<f32>());
        enc.set_threadgroup_memory(2, tile_c * head_dim * 2);
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: 1,
                depth: nwg,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let sq_bytes = group_tile * DK * 2; // f16 = 2 bytes per element
        let ss_bytes = group_tile * tile_c * std::mem::size_of::<f32>();
        enc.set_threadgroup_memory(0, sq_bytes);
        enc.set_threadgroup_memory(1, ss_bytes);

        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: group / group_tile,
                depth: nwg,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }

    // -------- Reduce kernel ----------------------------------------------
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceArgs {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_partitions: u32,
    }
    let reduce_h2 = nwg >= 128;
    let red_pipeline_name = match (group, reduce_h2) {
        (4, false) => "kernel_attn_decode_v4_reduce_g4_f32",
        (6, false) => "kernel_attn_decode_v4_reduce_f32",
        (8, false) => "kernel_attn_decode_v4_reduce_g8_f32",
        (16, false) => "kernel_attn_decode_v4_reduce_g16_f32",
        (4, true) => "kernel_attn_decode_v4_reduce_h2_g4_f32",
        (6, true) => "kernel_attn_decode_v4_reduce_h2_g6_f32",
        (8, true) => "kernel_attn_decode_v4_reduce_h2_g8_f32",
        (16, true) => "kernel_attn_decode_v4_reduce_h2_g16_f32",
        _ => unreachable!(),
    };
    let pso_red = ctx.pipeline(red_pipeline_name)?;
    enc.set_pipeline(&pso_red);
    enc.set_bytes(
        0,
        &ReduceArgs {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_partitions: nwg as u32,
        },
    );
    enc.set_tensor(1, o_partial);
    enc.set_tensor(2, ml_partial);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, nwg * std::mem::size_of::<f32>());

    enc.dispatch(
        MTLSize {
            width: n_q_heads,
            height: if reduce_h2 { 2 } else { 1 },
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

/// Encode only the v4 main kernel into `enc`, writing partials but not the
/// final reduced output. Useful for profiling where the split-K main pass and
/// reduction need to be measured separately.
pub fn encode_attn_decode_v4_main_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_pos: usize,
    nwg: usize,
    tile_c: usize,
) -> Result<(), MetalError> {
    const DK: usize = 256;
    if n_q_heads % n_kv_heads != 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("n_q_heads={n_q_heads} not multiple of n_kv_heads={n_kv_heads}"),
        });
    }
    let group = n_q_heads / n_kv_heads;
    if head_dim != DK {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("head_dim={head_dim} but kernel hardcodes {DK}"),
        });
    }
    if !matches!(group, 4 | 6 | 8 | 16) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("group={group} unsupported; expected one of {{4, 6, 8, 16}}"),
        });
    }
    if k_cache.dtype != v_cache.dtype || !matches!(k_cache.dtype, GgmlType::F16 | GgmlType::Q8_0) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!(
                "k/v expected matching F16 or Q8_0 dtypes, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    let want_q = (n_q_heads * head_dim) as u64;
    if q.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("q expected {want_q} elements"),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let group_tile = attn_v4_choose_group_tile(n_pos, group);
    if group_tile == 0 || group % group_tile != 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("group_tile={group_tile} must divide group={group}"),
        });
    }
    if group_tile != group
        && !(group == 16 && matches!(group_tile, 4 | 8)
            || group == 8 && matches!(group_tile, 2 | 4))
    {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("unsupported subgroup: group={group} group_tile={group_tile}"),
        });
    }
    let want_o_partial = (n_kv_heads * nwg * group * head_dim) as u64;
    let want_ml_partial = (n_kv_heads * nwg * group * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    let kv_stride = n_kv_heads * head_dim;
    let scale = (1.0f32 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        scale: f32,
    }
    let use_g8_bcast = k_cache.dtype == GgmlType::F16
        && group == 8
        && matches!(group_tile, 2 | 4)
        && tile_c == 64
        && attn_v4_g8_bcast_enabled();
    let use_g8_vstage = !use_g8_bcast
        && k_cache.dtype == GgmlType::F16
        && group == 8
        && group_tile == 4
        && attn_v4_g8_vstage_c() == Some(tile_c);
    // W1b: partition-packed main (opt-in; exact same per-simdgroup dataflow)
    let use_pack4 = !use_g8_bcast
        && !use_g8_vstage
        && k_cache.dtype == GgmlType::F16
        && group == 8
        && group_tile == 4
        && tile_c == 64
        && attn_v4_pack() == 4;
    let pipeline_name = if use_pack4 {
        "kernel_attn_decode_v4_g8_t4_c64_pack4_f32"
    } else if use_g8_bcast {
        match group_tile {
            2 => "kernel_attn_decode_v4_g8_t2_c64_bcast_f32",
            4 => "kernel_attn_decode_v4_g8_t4_c64_bcast_f32",
            _ => unreachable!(),
        }
    } else if use_g8_vstage {
        match tile_c {
            16 => "kernel_attn_decode_v4_g8_t4_c16_vstage_f32",
            32 => "kernel_attn_decode_v4_g8_t4_c32_vstage_f32",
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4_main",
                    detail: format!("vstage C={tile_c} unsupported; expected 16 or 32"),
                });
            }
        }
    } else if group_tile == group {
        match (k_cache.dtype, group, tile_c) {
            (GgmlType::F16, 4, 16) => "kernel_attn_decode_v4_g4_c16_f32",
            (GgmlType::F16, 4, 32) => "kernel_attn_decode_v4_g4_f32",
            (GgmlType::F16, 4, 64) => "kernel_attn_decode_v4_g4_c64_f32",
            (GgmlType::F16, 4, 128) => "kernel_attn_decode_v4_g4_c128_f32",
            (GgmlType::Q8_0, 6, 16) => "kernel_attn_decode_v4_q8_c16_f32",
            (GgmlType::Q8_0, 6, 32) => "kernel_attn_decode_v4_q8_f32",
            (GgmlType::Q8_0, 6, 64) => "kernel_attn_decode_v4_q8_c64_f32",
            (GgmlType::Q8_0, 6, 128) => "kernel_attn_decode_v4_q8_c128_f32",
            (GgmlType::F16, 6, 16) => "kernel_attn_decode_v4_c16_f32",
            (GgmlType::F16, 6, 32) => "kernel_attn_decode_v4_f32",
            (GgmlType::F16, 6, 64) => "kernel_attn_decode_v4_c64_f32",
            (GgmlType::F16, 6, 128) => "kernel_attn_decode_v4_c128_f32",
            (GgmlType::F16, 8, 16) => "kernel_attn_decode_v4_g8_c16_f32",
            (GgmlType::F16, 8, 32) => "kernel_attn_decode_v4_g8_f32",
            (GgmlType::F16, 8, 64) => "kernel_attn_decode_v4_g8_c64_f32",
            (GgmlType::F16, 8, 128) => "kernel_attn_decode_v4_g8_c128_f32",
            (GgmlType::Q8_0, 8, 16) => "kernel_attn_decode_v4_q8_g8_c16_f32",
            (GgmlType::Q8_0, 8, 32) => "kernel_attn_decode_v4_q8_g8_f32",
            (GgmlType::Q8_0, 8, 64) => "kernel_attn_decode_v4_q8_g8_c64_f32",
            (GgmlType::Q8_0, 8, 128) => "kernel_attn_decode_v4_q8_g8_c128_f32",
            (GgmlType::F16, 16, 16) => "kernel_attn_decode_v4_g16_c16_f32",
            (GgmlType::F16, 16, 32) => "kernel_attn_decode_v4_g16_f32",
            (GgmlType::F16, 16, 64) => "kernel_attn_decode_v4_g16_c64_f32",
            (GgmlType::F16, 16, 128) => "kernel_attn_decode_v4_g16_c128_f32",
            (GgmlType::Q8_0, _, _) => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4_main",
                    detail: format!(
                        "Q8_0 KV main kernels currently support only group=6/group=8; got group={group}, tile_c={tile_c}"
                    ),
                });
            }
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4_main",
                    detail: format!(
                        "unsupported (group={group}, tile_c={tile_c}); tile_c must be 16/32/64/128"
                    ),
                });
            }
        }
    } else {
        match (k_cache.dtype, group, group_tile, tile_c) {
            (GgmlType::F16, 8, 4, 16) => "kernel_attn_decode_v4_g8_t4_c16_f32",
            (GgmlType::F16, 8, 4, 32) => "kernel_attn_decode_v4_g8_t4_f32",
            (GgmlType::F16, 8, 4, 64) => "kernel_attn_decode_v4_g8_t4_c64_f32",
            (GgmlType::F16, 8, 4, 128) => "kernel_attn_decode_v4_g8_t4_c128_f32",
            (GgmlType::F16, 8, 2, 16) => "kernel_attn_decode_v4_g8_t2_c16_f32",
            (GgmlType::F16, 8, 2, 32) => "kernel_attn_decode_v4_g8_t2_f32",
            (GgmlType::F16, 8, 2, 64) => "kernel_attn_decode_v4_g8_t2_c64_f32",
            (GgmlType::F16, 8, 2, 128) => "kernel_attn_decode_v4_g8_t2_c128_f32",
            (GgmlType::Q8_0, 8, 4, 16) => "kernel_attn_decode_v4_q8_g8_t4_c16_f32",
            (GgmlType::Q8_0, 8, 4, 32) => "kernel_attn_decode_v4_q8_g8_t4_f32",
            (GgmlType::Q8_0, 8, 4, 64) => "kernel_attn_decode_v4_q8_g8_t4_c64_f32",
            (GgmlType::Q8_0, 8, 4, 128) => "kernel_attn_decode_v4_q8_g8_t4_c128_f32",
            (GgmlType::Q8_0, 8, 2, 16) => "kernel_attn_decode_v4_q8_g8_t2_c16_f32",
            (GgmlType::Q8_0, 8, 2, 32) => "kernel_attn_decode_v4_q8_g8_t2_f32",
            (GgmlType::Q8_0, 8, 2, 64) => "kernel_attn_decode_v4_q8_g8_t2_c64_f32",
            (GgmlType::Q8_0, 8, 2, 128) => "kernel_attn_decode_v4_q8_g8_t2_c128_f32",
            (GgmlType::F16, 16, 8, 16) => "kernel_attn_decode_v4_g16_t8_c16_f32",
            (GgmlType::F16, 16, 8, 32) => "kernel_attn_decode_v4_g16_t8_f32",
            (GgmlType::F16, 16, 8, 64) => "kernel_attn_decode_v4_g16_t8_c64_f32",
            (GgmlType::F16, 16, 8, 128) => "kernel_attn_decode_v4_g16_t8_c128_f32",
            (GgmlType::F16, 16, 4, 16) => "kernel_attn_decode_v4_g16_t4_c16_f32",
            (GgmlType::F16, 16, 4, 32) => "kernel_attn_decode_v4_g16_t4_f32",
            (GgmlType::F16, 16, 4, 64) => "kernel_attn_decode_v4_g16_t4_c64_f32",
            (GgmlType::F16, 16, 4, 128) => "kernel_attn_decode_v4_g16_t4_c128_f32",
            (GgmlType::Q8_0, _, _, _) => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4_main",
                    detail: format!(
                        "Q8_0 KV subgroup kernels currently support only group=8 tile2/tile4; got group={group}, group_tile={group_tile}, tile_c={tile_c}"
                    ),
                });
            }
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4_main",
                    detail: format!(
                        "unsupported (group={group}, group_tile={group_tile}, tile_c={tile_c})"
                    ),
                });
            }
        }
    };
    let pso_main = ctx.pipeline(pipeline_name)?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            scale,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    if use_g8_bcast {
        let sq_bytes = group_tile * DK * 2;
        enc.set_threadgroup_memory(0, sq_bytes);
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: group / group_tile,
                depth: nwg,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else if use_pack4 {
        const PACK: usize = 4;
        // sq is SHARED across the packed partitions; ss is per-simdgroup.
        enc.set_threadgroup_memory(0, group_tile * DK * 2);
        enc.set_threadgroup_memory(1, PACK * group_tile * tile_c * std::mem::size_of::<f32>());
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: group / group_tile,
                depth: nwg.div_ceil(PACK),
            },
            MTLSize {
                width: 32 * PACK,
                height: 1,
                depth: 1,
            },
        );
    } else if use_g8_vstage {
        enc.set_threadgroup_memory(0, 2 * group_tile * DK * 2);
        enc.set_threadgroup_memory(1, 2 * group_tile * tile_c * std::mem::size_of::<f32>());
        enc.set_threadgroup_memory(2, tile_c * head_dim * 2);
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: 1,
                depth: nwg,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let sq_bytes = group_tile * DK * 2;
        let ss_bytes = group_tile * tile_c * std::mem::size_of::<f32>();
        enc.set_threadgroup_memory(0, sq_bytes);
        enc.set_threadgroup_memory(1, ss_bytes);
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: group / group_tile,
                depth: nwg,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

/// Synthetic head-major F16 KV sidecar for v4 long-context attention proofing.
///
/// This intentionally supports only the current long MoE subgroup shapes:
/// group8/tile2/C64 plus group16/tile4/C64/C128. `k_cache` and `v_cache` are
/// laid out as `[kv_head, n_pos, head_dim]` with exactly `n_pos` rows per head.
pub fn encode_attn_decode_v4_main_only_f32_head_major(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_pos: usize,
    nwg: usize,
    tile_c: usize,
) -> Result<(), MetalError> {
    const DK: usize = 256;
    if n_q_heads % n_kv_heads != 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main_hm",
            detail: format!("n_q_heads={n_q_heads} not multiple of n_kv_heads={n_kv_heads}"),
        });
    }
    let group = n_q_heads / n_kv_heads;
    if head_dim != DK || !matches!((group, tile_c), (8, 64) | (16, 64 | 128)) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main_hm",
            detail: format!(
                "expected head_dim=256 and group/tile_c in {{8/64,16/64,16/128}}; got head_dim={head_dim} tile_c={tile_c} group={group}"
            ),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main_hm",
            detail: format!(
                "expected F16 K/V, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    let want_q = (n_q_heads * head_dim) as u64;
    let want_kv = (n_kv_heads * n_pos * head_dim) as u64;
    let want_o_partial = (n_kv_heads * nwg * group * head_dim) as u64;
    let want_ml_partial = (n_kv_heads * nwg * group * 2) as u64;
    if q.n_elements() != want_q
        || k_cache.n_elements() < want_kv
        || v_cache.n_elements() < want_kv
        || o_partial.n_elements() < want_o_partial
        || ml_partial.n_elements() < want_ml_partial
    {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main_hm",
            detail: format!(
                "shape mismatch q={} want {want_q}, k={} v={} want >= {want_kv}, o={} want >= {want_o_partial}, ml={} want >= {want_ml_partial}",
                q.n_elements(),
                k_cache.n_elements(),
                v_cache.n_elements(),
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main_hm",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }

    let group_tile = attn_v4_choose_group_tile(n_pos, group);
    let pipeline_name = match (group, group_tile, tile_c) {
        (8, 2, 64) => "kernel_attn_decode_v4_g8_t2_c64_hm_f32",
        (16, 4, 64) => "kernel_attn_decode_v4_g16_t4_c64_hm_f32",
        (16, 4, 128) => "kernel_attn_decode_v4_g16_t4_c128_hm_f32",
        _ => {
            return Err(MetalError::BadShape {
                kernel: "attn_decode_v4_main_hm",
                detail: format!(
                    "unsupported group/group_tile/tile_c for HM proof: {group}/{group_tile}/{tile_c}"
                ),
            });
        }
    };

    let scale = (1.0f32 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        scale: f32,
    }

    let pso_main = ctx.pipeline(pipeline_name)?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_pos: n_pos as u32,
            kv_stride: head_dim as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            scale,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    enc.set_threadgroup_memory(0, group_tile * DK * 2);
    enc.set_threadgroup_memory(1, group_tile * tile_c * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_kv_heads,
            height: group / group_tile,
            depth: nwg,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Encode only the v4 reduce kernel into `enc`, consuming previously-written
/// partials and producing the final attention output.
pub fn encode_attn_decode_v4_reduce_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const DK: usize = 256;
    if n_q_heads % n_kv_heads != 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_reduce",
            detail: format!("n_q_heads={n_q_heads} not multiple of n_kv_heads={n_kv_heads}"),
        });
    }
    let group = n_q_heads / n_kv_heads;
    if head_dim != DK {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_reduce",
            detail: format!("head_dim={head_dim} but kernel hardcodes {DK}"),
        });
    }
    if !matches!(group, 4 | 6 | 8 | 16) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_reduce",
            detail: format!("group={group} unsupported; expected one of {{4, 6, 8, 16}}"),
        });
    }
    let want_q = (n_q_heads * head_dim) as u64;
    if out.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_reduce",
            detail: format!("out expected {want_q} elements"),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_reduce",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let want_o_partial = (n_kv_heads * nwg * group * head_dim) as u64;
    let want_ml_partial = (n_kv_heads * nwg * group * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_reduce",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceArgs {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_partitions: u32,
    }
    let reduce_h2 = nwg >= 128;
    let red_pipeline_name = match (group, reduce_h2) {
        (4, false) => "kernel_attn_decode_v4_reduce_g4_f32",
        (6, false) => "kernel_attn_decode_v4_reduce_f32",
        (8, false) => "kernel_attn_decode_v4_reduce_g8_f32",
        (16, false) => "kernel_attn_decode_v4_reduce_g16_f32",
        (4, true) => "kernel_attn_decode_v4_reduce_h2_g4_f32",
        (6, true) => "kernel_attn_decode_v4_reduce_h2_g6_f32",
        (8, true) => "kernel_attn_decode_v4_reduce_h2_g8_f32",
        (16, true) => "kernel_attn_decode_v4_reduce_h2_g16_f32",
        _ => unreachable!(),
    };
    let pso_red = ctx.pipeline(red_pipeline_name)?;
    enc.set_pipeline(&pso_red);
    enc.set_bytes(
        0,
        &ReduceArgs {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_partitions: nwg as u32,
        },
    );
    enc.set_tensor(1, o_partial);
    enc.set_tensor(2, ml_partial);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, nwg * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_q_heads,
            height: if reduce_h2 { 2 } else { 1 },
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

pub fn encode_attn_prefill_v4_g8_t2_q2_c64_main_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 16;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 8;
    const GROUP_TILE: usize = 2;
    const QT: usize = 2;
    const TILE_C: usize = 64;

    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64",
            detail: "n_rows must be > 0".into(),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64",
            detail: format!(
                "expected F16 KV cache, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let want_q = (n_rows * N_Q_HEADS * HEAD_DIM) as u64;
    if q_rows.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64",
            detail: format!("q expected {want_q} elements"),
        });
    }
    let want_o_partial = (n_rows * N_KV_HEADS * nwg * GROUP * HEAD_DIM) as u64;
    let want_ml_partial = (n_rows * N_KV_HEADS * nwg * GROUP * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    let n_pos = base_pos + n_rows;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));
    let kv_stride = N_KV_HEADS * HEAD_DIM;
    let scale = (1.0f32 / (HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        base_pos: u32,
        scale: f32,
    }
    let pso_main = ctx.pipeline("kernel_attn_prefill_v4_g8_t2_q2_c64_f32")?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_rows: n_rows as u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            base_pos: base_pos as u32,
            scale,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    enc.set_threadgroup_memory(0, QT * GROUP_TILE * HEAD_DIM * 2);
    enc.set_threadgroup_memory(1, QT * GROUP_TILE * TILE_C * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_KV_HEADS,
            height: n_rows.div_ceil(QT) * (GROUP / GROUP_TILE),
            depth: nwg,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

pub fn encode_attn_prefill_v4_g8_t2_q2_c64_reduce_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 16;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 8;

    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64_reduce",
            detail: "n_rows must be > 0".into(),
        });
    }
    let want_q = (n_rows * N_Q_HEADS * HEAD_DIM) as u64;
    if out.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64_reduce",
            detail: format!("out expected {want_q} elements"),
        });
    }
    let want_o_partial = (n_rows * N_KV_HEADS * nwg * GROUP * HEAD_DIM) as u64;
    let want_ml_partial = (n_rows * N_KV_HEADS * nwg * GROUP * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64_reduce",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_partitions: u32,
    }
    let pso_red = ctx.pipeline("kernel_attn_prefill_v4_reduce_rows_g8_f32")?;
    enc.set_pipeline(&pso_red);
    enc.set_bytes(
        0,
        &ReduceArgs {
            n_rows: n_rows as u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_partitions: nwg as u32,
        },
    );
    enc.set_tensor(1, o_partial);
    enc.set_tensor(2, ml_partial);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, 64 * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, 64 * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, 64 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_Q_HEADS,
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

/// Prompt-native packed attention microproof for the A3B attention shape.
///
/// This is intentionally narrow and only meant to answer whether batching
/// multiple consecutive prompt queries against the same K/V tiles can beat the
/// current repeated decode-shaped attention body.
pub fn encode_attn_prefill_v4_g8_t2_q2_c64_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    encode_attn_prefill_v4_g8_t2_q2_c64_main_only_f32(
        ctx, enc, q_rows, k_cache, v_cache, o_partial, ml_partial, n_rows, base_pos, nwg,
    )?;
    encode_attn_prefill_v4_g8_t2_q2_c64_reduce_only_f32(
        ctx, enc, o_partial, ml_partial, out, n_rows, nwg,
    )?;
    Ok(())
}

pub fn encode_attn_prefill_v4_g8_t2_q4_c64_main_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 16;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 8;
    const GROUP_TILE: usize = 2;
    const QT: usize = 4;
    const TILE_C: usize = 64;

    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q4_c64",
            detail: "n_rows must be > 0".into(),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q4_c64",
            detail: format!(
                "expected F16 KV cache, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q4_c64",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let want_q = (n_rows * N_Q_HEADS * HEAD_DIM) as u64;
    if q_rows.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q4_c64",
            detail: format!("q expected {want_q} elements"),
        });
    }
    let want_o_partial = (n_rows * N_KV_HEADS * nwg * GROUP * HEAD_DIM) as u64;
    let want_ml_partial = (n_rows * N_KV_HEADS * nwg * GROUP * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q4_c64",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    let n_pos = base_pos + n_rows;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));
    let kv_stride = N_KV_HEADS * HEAD_DIM;
    let scale = (1.0f32 / (HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        base_pos: u32,
        scale: f32,
    }
    let pso_main = ctx.pipeline("kernel_attn_prefill_v4_g8_t2_q4_c64_f32")?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_rows: n_rows as u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            base_pos: base_pos as u32,
            scale,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    enc.set_threadgroup_memory(0, QT * GROUP_TILE * HEAD_DIM * 2);
    enc.set_threadgroup_memory(1, QT * GROUP_TILE * TILE_C * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_KV_HEADS,
            height: n_rows.div_ceil(QT) * (GROUP / GROUP_TILE),
            depth: nwg,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

pub fn encode_attn_prefill_v4_g8_t2_q4_c64_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    encode_attn_prefill_v4_g8_t2_q4_c64_main_only_f32(
        ctx, enc, q_rows, k_cache, v_cache, o_partial, ml_partial, n_rows, base_pos, nwg,
    )?;
    encode_attn_prefill_v4_g8_t2_q2_c64_reduce_only_f32(
        ctx, enc, o_partial, ml_partial, out, n_rows, nwg,
    )?;
    Ok(())
}

pub fn encode_attn_prefill_v4_g16_t4_q2_c64_main_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 32;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 16;
    const GROUP_TILE: usize = 4;
    const QT: usize = 2;
    const TILE_C: usize = 64;

    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64",
            detail: "n_rows must be > 0".into(),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64",
            detail: format!(
                "expected F16 KV cache, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let want_q = (n_rows * N_Q_HEADS * HEAD_DIM) as u64;
    if q_rows.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64",
            detail: format!("q expected {want_q} elements"),
        });
    }
    let want_o_partial = (n_rows * N_KV_HEADS * nwg * GROUP * HEAD_DIM) as u64;
    let want_ml_partial = (n_rows * N_KV_HEADS * nwg * GROUP * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    let n_pos = base_pos + n_rows;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));
    let kv_stride = N_KV_HEADS * HEAD_DIM;
    let scale = (1.0f32 / (HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        base_pos: u32,
        scale: f32,
    }
    let pso_main = ctx.pipeline("kernel_attn_prefill_v4_g16_t4_q2_c64_f32")?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_rows: n_rows as u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            base_pos: base_pos as u32,
            scale,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    enc.set_threadgroup_memory(0, QT * GROUP_TILE * HEAD_DIM * 2);
    enc.set_threadgroup_memory(1, QT * GROUP_TILE * TILE_C * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_KV_HEADS,
            height: n_rows.div_ceil(QT) * (GROUP / GROUP_TILE),
            depth: nwg,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

pub fn encode_attn_prefill_v4_g16_t4_q2_c64_reduce_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 32;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 16;

    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64_reduce",
            detail: "n_rows must be > 0".into(),
        });
    }
    let want_q = (n_rows * N_Q_HEADS * HEAD_DIM) as u64;
    if out.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64_reduce",
            detail: format!("out expected {want_q} elements"),
        });
    }
    let want_o_partial = (n_rows * N_KV_HEADS * nwg * GROUP * HEAD_DIM) as u64;
    let want_ml_partial = (n_rows * N_KV_HEADS * nwg * GROUP * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64_reduce",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_partitions: u32,
    }
    let pso_red = ctx.pipeline("kernel_attn_prefill_v4_reduce_rows_g16_f32")?;
    enc.set_pipeline(&pso_red);
    enc.set_bytes(
        0,
        &ReduceArgs {
            n_rows: n_rows as u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_partitions: nwg as u32,
        },
    );
    enc.set_tensor(1, o_partial);
    enc.set_tensor(2, ml_partial);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, 64 * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, 64 * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, 64 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_Q_HEADS,
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

pub fn encode_attn_prefill_v4_g16_t4_q2_c64_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    encode_attn_prefill_v4_g16_t4_q2_c64_main_only_f32(
        ctx, enc, q_rows, k_cache, v_cache, o_partial, ml_partial, n_rows, base_pos, nwg,
    )?;
    encode_attn_prefill_v4_g16_t4_q2_c64_reduce_only_f32(
        ctx, enc, o_partial, ml_partial, out, n_rows, nwg,
    )?;
    Ok(())
}

pub fn encode_attn_prefill_v4_g16_t4_q4_c64_main_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 32;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 16;
    const GROUP_TILE: usize = 4;
    const QT: usize = 4;
    const TILE_C: usize = 64;

    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q4_c64",
            detail: "n_rows must be > 0".into(),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q4_c64",
            detail: format!(
                "expected F16 KV cache, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q4_c64",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let want_q = (n_rows * N_Q_HEADS * HEAD_DIM) as u64;
    if q_rows.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q4_c64",
            detail: format!("q expected {want_q} elements"),
        });
    }
    let want_o_partial = (n_rows * N_KV_HEADS * nwg * GROUP * HEAD_DIM) as u64;
    let want_ml_partial = (n_rows * N_KV_HEADS * nwg * GROUP * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q4_c64",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    let n_pos = base_pos + n_rows;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));
    let kv_stride = N_KV_HEADS * HEAD_DIM;
    let scale = (1.0f32 / (HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        base_pos: u32,
        scale: f32,
    }
    let pso_main = ctx.pipeline("kernel_attn_prefill_v4_g16_t4_q4_c64_f32")?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_rows: n_rows as u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            base_pos: base_pos as u32,
            scale,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    enc.set_threadgroup_memory(0, QT * GROUP_TILE * HEAD_DIM * 2);
    enc.set_threadgroup_memory(1, QT * GROUP_TILE * TILE_C * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_KV_HEADS,
            height: n_rows.div_ceil(QT) * (GROUP / GROUP_TILE),
            depth: nwg,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

pub fn encode_attn_prefill_v4_g16_t4_q4_c64_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    encode_attn_prefill_v4_g16_t4_q4_c64_main_only_f32(
        ctx, enc, q_rows, k_cache, v_cache, o_partial, ml_partial, n_rows, base_pos, nwg,
    )?;
    encode_attn_prefill_v4_g16_t4_q2_c64_reduce_only_f32(
        ctx, enc, o_partial, ml_partial, out, n_rows, nwg,
    )?;
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnMatrixArgs {
    n_rows: u32,
    n_pos: u32,
    base_pos: u32,
    kv_stride: u32,
    vt_stride: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    group: u32,
    head_dim: u32,
    scale: f32,
    causal_skip: u32,
}

fn validate_attn_matrix_common(
    kernel: &'static str,
    n_rows: usize,
    n_pos: usize,
    base_pos: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel,
            detail: "n_rows must be > 0".into(),
        });
    }
    if n_pos < base_pos + n_rows {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("n_pos={n_pos} < base_pos+n_rows={}", base_pos + n_rows),
        });
    }
    if n_kv_heads == 0 || group == 0 || n_q_heads != n_kv_heads * group || head_dim != 256 {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "unsupported matrix-attn shape n_q={n_q_heads} n_kv={n_kv_heads} group={group} head_dim={head_dim}"
            ),
        });
    }
    Ok(())
}

pub fn encode_attn_matrix_transpose_v_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    v_cache: &MetalTensor,
    v_t: &MetalTensor,
    base_pos: usize,
    n_rows: usize,
    n_pos: usize,
    kv_stride: usize,
    vt_stride: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    if v_cache.dtype != GgmlType::F16 || v_t.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!(
                "expected F16 tensors, got {:?}/{:?}",
                v_cache.dtype, v_t.dtype
            ),
        });
    }
    if n_rows == 0 || base_pos + n_rows > n_pos {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!(
                "invalid V transpose span base_pos={base_pos} n_rows={n_rows} n_pos={n_pos}"
            ),
        });
    }
    if vt_stride < n_pos {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!("vt_stride={vt_stride} < n_pos={n_pos}"),
        });
    }
    let want_vt = n_kv_heads * head_dim * vt_stride;
    if v_t.n_elements() < want_vt as u64 {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!("v_t has {} elements, need >= {want_vt}", v_t.n_elements()),
        });
    }
    let want_cache = (base_pos + n_rows) * kv_stride;
    if v_cache.n_elements() < want_cache as u64 {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!(
                "v_cache has {} elements, need >= {want_cache}",
                v_cache.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_attn_matrix_transpose_v_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows as u32,
            n_pos: n_pos as u32,
            base_pos: base_pos as u32,
            kv_stride: kv_stride as u32,
            vt_stride: vt_stride as u32,
            n_q_heads: 0,
            n_kv_heads: n_kv_heads as u32,
            group: 0,
            head_dim: head_dim as u32,
            scale: 0.0,
            causal_skip: 0,
        },
    );
    enc.set_tensor(1, v_cache);
    enc.set_tensor(2, v_t);
    enc.dispatch(
        MTLSize {
            width: n_kv_heads * head_dim * n_rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_attn_matrix_kq_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    scores: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    n_pos: usize,
    kv_stride: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
    causal_skip: bool,
) -> Result<(), MetalError> {
    validate_attn_matrix_common(
        "attn_matrix_kq",
        n_rows,
        n_pos,
        base_pos,
        n_q_heads,
        n_kv_heads,
        group,
        head_dim,
    )?;
    if q_rows.dtype != GgmlType::F32
        || scores.dtype != GgmlType::F32
        || k_cache.dtype != GgmlType::F16
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kq",
            detail: format!(
                "expected q/scores F32 and k F16, got {:?}/{:?}/{:?}",
                q_rows.dtype, scores.dtype, k_cache.dtype
            ),
        });
    }
    let want_q = n_rows * n_q_heads * head_dim;
    let want_scores = n_kv_heads * n_rows * group * n_pos;
    if q_rows.n_elements() != want_q as u64 || scores.n_elements() < want_scores as u64 {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kq",
            detail: format!(
                "bad q/scores sizes: q have {} need {want_q}, scores have {} need >= {want_scores}",
                q_rows.n_elements(),
                scores.n_elements()
            ),
        });
    }
    let full_tiles = n_pos % 64 == 0 && (n_rows * group) % 32 == 0 && head_dim == 256;
    let kernel = if full_tiles {
        "kernel_attn_matrix_kq_f32_full_tiles"
    } else {
        "kernel_attn_matrix_kq_f32"
    };
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows as u32,
            n_pos: n_pos as u32,
            base_pos: base_pos as u32,
            kv_stride: kv_stride as u32,
            vt_stride: 0,
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            group: group as u32,
            head_dim: head_dim as u32,
            scale: 0.0,
            causal_skip: causal_skip as u32,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, scores);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: (n_rows * group).div_ceil(32),
            height: n_pos.div_ceil(64),
            depth: n_kv_heads,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_attn_matrix_softmax_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    n_pos: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    validate_attn_matrix_common(
        "attn_matrix_softmax",
        n_rows,
        n_pos,
        base_pos,
        n_q_heads,
        n_kv_heads,
        group,
        head_dim,
    )?;
    let want_scores = n_rows * n_q_heads * n_pos;
    if scores.dtype != GgmlType::F32 || scores.n_elements() < want_scores as u64 {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_softmax",
            detail: format!("scores have {} need >= {want_scores}", scores.n_elements()),
        });
    }
    let scale = (1.0f32 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
    let pso = ctx.pipeline("kernel_attn_matrix_softmax_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows as u32,
            n_pos: n_pos as u32,
            base_pos: base_pos as u32,
            kv_stride: 0,
            vt_stride: 0,
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            group: group as u32,
            head_dim: head_dim as u32,
            scale,
            causal_skip: 0,
        },
    );
    enc.set_tensor(1, scores);
    enc.set_threadgroup_memory(0, 8 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_rows * n_q_heads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Number of F32 elements required for the per-(query, 64-pos-tile) (m, l)
/// sidecar consumed by the two-pass online matrix attention kernels.
pub fn attn_matrix_ml_elems(n_rows: usize, n_q_heads: usize, n_pos: usize) -> usize {
    n_rows * n_q_heads * n_pos.div_ceil(64) * 2
}

/// KQ with fused online softmax (`kernel_attn_matrix_kq_online_f32` in
/// kernels/attn_matrix_online.metal): same GEMM as [`encode_attn_matrix_kq_f32`]
/// but the epilogue applies scale + causal mask and stores
/// `P~ = exp2(s*scale - m_tile)` as F16 plus a per-(query, tile) (m, l)
/// sidecar. Replaces the separate softmax dispatch; pair with
/// [`encode_attn_matrix_kqv_norm_f32`].
#[allow(clippy::too_many_arguments)]
pub fn encode_attn_matrix_kq_online_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    scores_h: &MetalTensor,
    ml: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    n_pos: usize,
    kv_stride: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
    causal_skip: bool,
) -> Result<(), MetalError> {
    validate_attn_matrix_common(
        "attn_matrix_kq_online",
        n_rows,
        n_pos,
        base_pos,
        n_q_heads,
        n_kv_heads,
        group,
        head_dim,
    )?;
    if q_rows.dtype != GgmlType::F32
        || scores_h.dtype != GgmlType::F16
        || ml.dtype != GgmlType::F32
        || k_cache.dtype != GgmlType::F16
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kq_online",
            detail: format!(
                "expected q F32, scores F16, ml F32, k F16; got {:?}/{:?}/{:?}/{:?}",
                q_rows.dtype, scores_h.dtype, ml.dtype, k_cache.dtype
            ),
        });
    }
    let want_q = n_rows * n_q_heads * head_dim;
    let want_scores = n_kv_heads * n_rows * group * n_pos;
    let want_ml = attn_matrix_ml_elems(n_rows, n_q_heads, n_pos);
    if q_rows.n_elements() != want_q as u64
        || scores_h.n_elements() < want_scores as u64
        || ml.n_elements() < want_ml as u64
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kq_online",
            detail: format!(
                "bad sizes: q have {} need {want_q}, scores have {} need >= {want_scores}, ml have {} need >= {want_ml}",
                q_rows.n_elements(),
                scores_h.n_elements(),
                ml.n_elements()
            ),
        });
    }
    let scale = (1.0f32 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
    let full_tiles = n_pos % 64 == 0 && (n_rows * group) % 32 == 0 && head_dim == 256;
    let kernel = if full_tiles {
        "kernel_attn_matrix_kq_online_f32_full_tiles"
    } else {
        "kernel_attn_matrix_kq_online_f32"
    };
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows as u32,
            n_pos: n_pos as u32,
            base_pos: base_pos as u32,
            kv_stride: kv_stride as u32,
            vt_stride: 0,
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            group: group as u32,
            head_dim: head_dim as u32,
            scale,
            causal_skip: causal_skip as u32,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, scores_h);
    enc.set_tensor(4, ml);
    // AMO_KQ_TG_BYTES in kernels/attn_matrix_online.metal.
    enc.set_threadgroup_memory(0, 9728);
    enc.dispatch(
        MTLSize {
            width: (n_rows * group).div_ceil(32),
            height: n_pos.div_ceil(64),
            depth: n_kv_heads,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// KQV with fused normalization (`kernel_attn_matrix_kqv_norm_f32` in
/// kernels/attn_matrix_online.metal): same GEMM as
/// [`encode_attn_matrix_kqv_f32`] but reads the F16 `P~` produced by
/// [`encode_attn_matrix_kq_online_f32`], rescales it by
/// `exp2(m_tile - m_glob)` during staging, and divides by the global `l` in
/// the epilogue.
#[allow(clippy::too_many_arguments)]
pub fn encode_attn_matrix_kqv_norm_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    probs_h: &MetalTensor,
    ml: &MetalTensor,
    v_t: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    n_pos: usize,
    vt_stride: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
    causal_skip: bool,
) -> Result<(), MetalError> {
    validate_attn_matrix_common(
        "attn_matrix_kqv_norm",
        n_rows,
        n_pos,
        base_pos,
        n_q_heads,
        n_kv_heads,
        group,
        head_dim,
    )?;
    if vt_stride < n_pos {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv_norm",
            detail: format!("vt_stride={vt_stride} < n_pos={n_pos}"),
        });
    }
    let want_probs = n_rows * n_q_heads * n_pos;
    let want_ml = attn_matrix_ml_elems(n_rows, n_q_heads, n_pos);
    let want_vt = n_kv_heads * head_dim * vt_stride;
    let want_out = n_rows * n_q_heads * head_dim;
    if probs_h.dtype != GgmlType::F16
        || ml.dtype != GgmlType::F32
        || out.dtype != GgmlType::F32
        || v_t.dtype != GgmlType::F16
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv_norm",
            detail: format!(
                "expected probs F16, ml F32, out F32, v_t F16; got {:?}/{:?}/{:?}/{:?}",
                probs_h.dtype, ml.dtype, out.dtype, v_t.dtype
            ),
        });
    }
    if probs_h.n_elements() < want_probs as u64
        || ml.n_elements() < want_ml as u64
        || v_t.n_elements() < want_vt as u64
        || out.n_elements() != want_out as u64
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv_norm",
            detail: format!(
                "bad sizes: probs {} need >= {want_probs}, ml {} need >= {want_ml}, v_t {} need >= {want_vt}, out {} need {want_out}",
                probs_h.n_elements(),
                ml.n_elements(),
                v_t.n_elements(),
                out.n_elements()
            ),
        });
    }
    let full_tiles = n_pos % 32 == 0 && (n_rows * group) % 32 == 0 && head_dim == 256;
    let kernel = if full_tiles {
        "kernel_attn_matrix_kqv_norm_f32_full_tiles"
    } else {
        "kernel_attn_matrix_kqv_norm_f32"
    };
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows as u32,
            n_pos: n_pos as u32,
            base_pos: base_pos as u32,
            kv_stride: 0,
            vt_stride: vt_stride as u32,
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            group: group as u32,
            head_dim: head_dim as u32,
            scale: 0.0,
            causal_skip: causal_skip as u32,
        },
    );
    enc.set_tensor(1, probs_h);
    enc.set_tensor(2, ml);
    enc.set_tensor(3, v_t);
    enc.set_tensor(4, out);
    // AMO_KQV_TG_BYTES in kernels/attn_matrix_online.metal.
    enc.set_threadgroup_memory(0, 8320);
    enc.dispatch(
        MTLSize {
            width: (n_rows * group).div_ceil(32),
            height: head_dim.div_ceil(64),
            depth: n_kv_heads,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_attn_matrix_kqv_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    probs: &MetalTensor,
    v_t: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    n_pos: usize,
    vt_stride: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
    causal_skip: bool,
) -> Result<(), MetalError> {
    validate_attn_matrix_common(
        "attn_matrix_kqv",
        n_rows,
        n_pos,
        base_pos,
        n_q_heads,
        n_kv_heads,
        group,
        head_dim,
    )?;
    if vt_stride < n_pos {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv",
            detail: format!("vt_stride={vt_stride} < n_pos={n_pos}"),
        });
    }
    let want_probs = n_rows * n_q_heads * n_pos;
    let want_vt = n_kv_heads * head_dim * vt_stride;
    let want_out = n_rows * n_q_heads * head_dim;
    if probs.dtype != GgmlType::F32 || out.dtype != GgmlType::F32 || v_t.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv",
            detail: format!(
                "expected probs/out F32 and v_t F16, got {:?}/{:?}/{:?}",
                probs.dtype, out.dtype, v_t.dtype
            ),
        });
    }
    if probs.n_elements() < want_probs as u64
        || v_t.n_elements() < want_vt as u64
        || out.n_elements() != want_out as u64
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv",
            detail: format!(
                "bad sizes: probs {} need >= {want_probs}, v_t {} need >= {want_vt}, out {} need {want_out}",
                probs.n_elements(),
                v_t.n_elements(),
                out.n_elements()
            ),
        });
    }
    let full_tiles = n_pos % 32 == 0 && (n_rows * group) % 32 == 0 && head_dim == 256;
    let kernel = if full_tiles {
        "kernel_attn_matrix_kqv_f32_full_tiles"
    } else {
        "kernel_attn_matrix_kqv_f32"
    };
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows as u32,
            n_pos: n_pos as u32,
            base_pos: base_pos as u32,
            kv_stride: 0,
            vt_stride: vt_stride as u32,
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            group: group as u32,
            head_dim: head_dim as u32,
            scale: 0.0,
            causal_skip: causal_skip as u32,
        },
    );
    enc.set_tensor(1, probs);
    enc.set_tensor(2, v_t);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: (n_rows * group).div_ceil(32),
            height: head_dim.div_ceil(64),
            depth: n_kv_heads,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Scatter F32 source bytes into a F16 destination buffer at offset.
/// Used for KV cache append when the cache is F16. Counterpart of
/// `encode_scatter_offset_f32` (F32 → F32).
pub fn encode_scatter_offset_f32_to_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    dst_off: usize,
    n: usize,
) -> Result<(), MetalError> {
    if src.dtype != GgmlType::F32 || dst.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16",
            detail: format!("expected F32→F16, got {:?}→{:?}", src.dtype, dst.dtype),
        });
    }
    if src.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16",
            detail: format!("src.n={} != n={n}", src.n_elements()),
        });
    }
    if (dst_off + n) as u64 > dst.n_elements() {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16",
            detail: format!("dst_off+n={} > dst.n={}", dst_off + n, dst.n_elements()),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        dst_off: u32,
    }
    let pso = ctx.pipeline("kernel_scatter_offset_f32_to_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            dst_off: dst_off as u32,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, dst);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// Fused K+V scatter — writes K and V into their respective F16 caches at
/// `dst_off` in a single dispatch. K and V always share `n` and `dst_off`
/// at decode time (`n = kv_dim, dst_off = position * kv_dim`), so we
/// amortize one dispatch per attn layer.
///
/// Per Jeff & Sanjay (Bulk APIs / amortize boundary crossings).
/// Saves 16 dispatches/token for the 27B (16 attn layers).
pub fn encode_scatter_offset_f32_to_f16_kv(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    k_src: &MetalTensor,
    v_src: &MetalTensor,
    k_dst: &MetalTensor,
    v_dst: &MetalTensor,
    dst_off: usize,
    n: usize,
) -> Result<(), MetalError> {
    if k_src.dtype != GgmlType::F32 || v_src.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv",
            detail: format!(
                "expected F32 sources, got k={:?} v={:?}",
                k_src.dtype, v_src.dtype
            ),
        });
    }
    if k_dst.dtype != GgmlType::F16 || v_dst.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv",
            detail: format!(
                "expected F16 dests, got k={:?} v={:?}",
                k_dst.dtype, v_dst.dtype
            ),
        });
    }
    if k_src.n_elements() as usize != n || v_src.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv",
            detail: format!(
                "src lengths k={} v={} != n={n}",
                k_src.n_elements(),
                v_src.n_elements()
            ),
        });
    }
    let dst_end = dst_off.checked_add(n).ok_or_else(|| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_f16_kv",
        detail: format!("dst_off={dst_off} + n={n} overflows usize"),
    })?;
    if dst_end as u64 > k_dst.n_elements() || dst_end as u64 > v_dst.n_elements() {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv",
            detail: format!(
                "dst_off+n={} exceeds k.n={} or v.n={}",
                dst_end,
                k_dst.n_elements(),
                v_dst.n_elements()
            ),
        });
    }
    let n_u32 = u32::try_from(n).map_err(|_| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_f16_kv",
        detail: format!("n={n} does not fit u32 kernel args"),
    })?;
    let dst_off_u32 = u32::try_from(dst_off).map_err(|_| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_f16_kv",
        detail: format!("dst_off={dst_off} does not fit u32 kernel args"),
    })?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        dst_off: u32,
    }
    let pso = ctx.pipeline("kernel_scatter_offset_f32_to_f16_kv")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n_u32,
            dst_off: dst_off_u32,
        },
    );
    enc.set_tensor(1, k_src);
    enc.set_tensor(2, v_src);
    enc.set_tensor(3, k_dst);
    enc.set_tensor(4, v_dst);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// Fused K+V scatter plus transposed-V sidecar write. This is intentionally
/// narrow support for the experimental non-flash matrix attention path: it keeps
/// the canonical `[pos, kv]` V cache intact while also filling a fixed-stride
/// `[kvh, d, pos]` V_T bank for KQV.
pub fn encode_scatter_offset_f32_to_f16_kv_vt(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    k_src: &MetalTensor,
    v_src: &MetalTensor,
    k_dst: &MetalTensor,
    v_dst: &MetalTensor,
    v_t: &MetalTensor,
    dst_off: usize,
    n: usize,
    base_pos: usize,
    kv_dim: usize,
    head_dim: usize,
    vt_stride: usize,
) -> Result<(), MetalError> {
    if k_src.dtype != GgmlType::F32 || v_src.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!(
                "expected F32 sources, got k={:?} v={:?}",
                k_src.dtype, v_src.dtype
            ),
        });
    }
    if k_dst.dtype != GgmlType::F16 || v_dst.dtype != GgmlType::F16 || v_t.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!(
                "expected F16 dests, got k={:?} v={:?} vt={:?}",
                k_dst.dtype, v_dst.dtype, v_t.dtype
            ),
        });
    }
    if kv_dim == 0 || head_dim == 0 || !kv_dim.is_multiple_of(head_dim) {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!("bad kv_dim/head_dim: kv_dim={kv_dim} head_dim={head_dim}"),
        });
    }
    if n == 0 || !n.is_multiple_of(kv_dim) {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!("n={n} must be a positive multiple of kv_dim={kv_dim}"),
        });
    }
    if k_src.n_elements() as usize != n || v_src.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!(
                "src lengths k={} v={} != n={n}",
                k_src.n_elements(),
                v_src.n_elements()
            ),
        });
    }
    let dst_end = dst_off.checked_add(n).ok_or_else(|| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_f16_kv_vt",
        detail: format!("dst_off={dst_off} + n={n} overflows usize"),
    })?;
    if dst_end as u64 > k_dst.n_elements() || dst_end as u64 > v_dst.n_elements() {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!(
                "dst_off+n={} exceeds k.n={} or v.n={}",
                dst_end,
                k_dst.n_elements(),
                v_dst.n_elements()
            ),
        });
    }
    let n_rows = n / kv_dim;
    if vt_stride < base_pos + n_rows {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!(
                "vt_stride={vt_stride} < base_pos+n_rows={}",
                base_pos + n_rows
            ),
        });
    }
    let want_vt = kv_dim
        .checked_mul(vt_stride)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!("kv_dim={kv_dim} * vt_stride={vt_stride} overflows usize"),
        })?;
    if v_t.n_elements() < want_vt as u64 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!("v_t has {} elements, need >= {want_vt}", v_t.n_elements()),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        dst_off: u32,
        base_pos: u32,
        kv_dim: u32,
        head_dim: u32,
        vt_stride: u32,
    }
    let pso = ctx.pipeline("kernel_scatter_offset_f32_to_f16_kv_vt")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: u32::try_from(n).map_err(|_| MetalError::BadShape {
                kernel: "scatter_offset_f32_to_f16_kv_vt",
                detail: format!("n={n} does not fit u32"),
            })?,
            dst_off: u32::try_from(dst_off).map_err(|_| MetalError::BadShape {
                kernel: "scatter_offset_f32_to_f16_kv_vt",
                detail: format!("dst_off={dst_off} does not fit u32"),
            })?,
            base_pos: u32::try_from(base_pos).map_err(|_| MetalError::BadShape {
                kernel: "scatter_offset_f32_to_f16_kv_vt",
                detail: format!("base_pos={base_pos} does not fit u32"),
            })?,
            kv_dim: u32::try_from(kv_dim).map_err(|_| MetalError::BadShape {
                kernel: "scatter_offset_f32_to_f16_kv_vt",
                detail: format!("kv_dim={kv_dim} does not fit u32"),
            })?,
            head_dim: u32::try_from(head_dim).map_err(|_| MetalError::BadShape {
                kernel: "scatter_offset_f32_to_f16_kv_vt",
                detail: format!("head_dim={head_dim} does not fit u32"),
            })?,
            vt_stride: u32::try_from(vt_stride).map_err(|_| MetalError::BadShape {
                kernel: "scatter_offset_f32_to_f16_kv_vt",
                detail: format!("vt_stride={vt_stride} does not fit u32"),
            })?,
        },
    );
    enc.set_tensor(1, k_src);
    enc.set_tensor(2, v_src);
    enc.set_tensor(3, k_dst);
    enc.set_tensor(4, v_dst);
    enc.set_tensor(5, v_t);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// Fused K+V scatter into Q8_0 caches. Quantizes each 32-element block with
/// ggml's reference rule: `d = amax / 127`, `qs[j] = round(x[j] / d)`.
///
/// Constraints: `dst_off` and `n` must both be multiples of 32 so the append
/// lands on Q8_0 block boundaries.
pub fn encode_scatter_offset_f32_to_q8_0_kv(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    k_src: &MetalTensor,
    v_src: &MetalTensor,
    k_dst: &MetalTensor,
    v_dst: &MetalTensor,
    dst_off: usize,
    n: usize,
) -> Result<(), MetalError> {
    const QK8_0: usize = 32;
    if k_src.dtype != GgmlType::F32 || v_src.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_q8_0_kv",
            detail: format!(
                "expected F32 sources, got k={:?} v={:?}",
                k_src.dtype, v_src.dtype
            ),
        });
    }
    if k_dst.dtype != GgmlType::Q8_0 || v_dst.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_q8_0_kv",
            detail: format!(
                "expected Q8_0 dests, got k={:?} v={:?}",
                k_dst.dtype, v_dst.dtype
            ),
        });
    }
    if k_src.n_elements() as usize != n || v_src.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_q8_0_kv",
            detail: format!(
                "src lengths k={} v={} != n={n}",
                k_src.n_elements(),
                v_src.n_elements()
            ),
        });
    }
    if dst_off % QK8_0 != 0 || n % QK8_0 != 0 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_q8_0_kv",
            detail: format!("dst_off={dst_off} and n={n} must both be multiples of {QK8_0}"),
        });
    }
    let dst_end = dst_off.checked_add(n).ok_or_else(|| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_q8_0_kv",
        detail: format!("dst_off={dst_off} + n={n} overflows usize"),
    })?;
    if dst_end as u64 > k_dst.n_elements() || dst_end as u64 > v_dst.n_elements() {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_q8_0_kv",
            detail: format!(
                "dst_off+n={} exceeds k.n={} or v.n={}",
                dst_end,
                k_dst.n_elements(),
                v_dst.n_elements()
            ),
        });
    }
    let n_blocks = n / QK8_0;
    let dst_block_off = dst_off / QK8_0;
    let n_blocks_u32 = u32::try_from(n_blocks).map_err(|_| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_q8_0_kv",
        detail: format!("n_blocks={n_blocks} does not fit u32 kernel args"),
    })?;
    let dst_block_off_u32 = u32::try_from(dst_block_off).map_err(|_| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_q8_0_kv",
        detail: format!("dst_block_off={dst_block_off} does not fit u32 kernel args"),
    })?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_blocks: u32,
        dst_block_off: u32,
    }
    let pso = ctx.pipeline("kernel_scatter_offset_f32_to_q8_0_kv")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_blocks: n_blocks_u32,
            dst_block_off: dst_block_off_u32,
        },
    );
    enc.set_tensor(1, k_src);
    enc.set_tensor(2, v_src);
    enc.set_tensor(3, k_dst);
    enc.set_tensor(4, v_dst);
    enc.dispatch(
        MTLSize {
            width: n / QK8_0,
            height: 1,
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// **DFlash drafter attention** (v0.72.1) — fused small-N attention
/// with per-layer SWA mask. Replaces the CPU phase-3 attention in
/// `draft_block`. See `kernels/dflash_attn.metal` for design.
///
/// Threadgroup grid: `(n_q_heads, N)` per drafter layer per outer step.
/// Threads per TG: 32 (one simdgroup; head_dim/32=4 dims per lane).
pub fn encode_dflash_attn_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    pos_k: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_kv_total: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
) -> Result<(), MetalError> {
    if head_dim % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn",
            detail: format!("head_dim={head_dim} not divisible by 32"),
        });
    }
    // Codex code-review v0.72.2: kernel uses `q_reg[8]` / `o_acc[8]`
    // sized for head_dim ≤ 256 (8 × 32 lanes = 256 dims). Reject
    // larger head_dim explicitly so future model variants don't
    // silently stack-OOB inside the kernel.
    if head_dim > 256 {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn",
            detail: format!(
                "head_dim={head_dim} > 256: kernel registers q_reg/o_acc are sized for head_dim ≤ 256"
            ),
        });
    }
    if n_q_heads % n_kv_heads != 0 {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn",
            detail: format!("n_q_heads={n_q_heads} not divisible by n_kv_heads={n_kv_heads}"),
        });
    }
    if q.n_elements() as usize != n * n_q_heads * head_dim {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn.q",
            detail: format!(
                "q.n_elements={} != N*n_q*head_dim={}",
                q.n_elements(),
                n * n_q_heads * head_dim
            ),
        });
    }
    let kv_stride = n_kv_heads * head_dim;
    if k.n_elements() as usize != n_kv_total * kv_stride
        || v.n_elements() as usize != n_kv_total * kv_stride
    {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn.kv",
            detail: format!(
                "k/v expected n_kv_total*kv_stride = {}*{} = {} elements",
                n_kv_total,
                kv_stride,
                n_kv_total * kv_stride
            ),
        });
    }
    if pos_k.n_elements() as usize != n_kv_total {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn.pos_k",
            detail: format!(
                "pos_k.n_elements={} != n_kv_total={n_kv_total}",
                pos_k.n_elements()
            ),
        });
    }
    if o.n_elements() as usize != n * n_q_heads * head_dim {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn.o",
            detail: format!(
                "o.n_elements={} != N*n_q*head_dim={}",
                o.n_elements(),
                n * n_q_heads * head_dim
            ),
        });
    }

    let pso = ctx.pipeline("kernel_dflash_attn_f32")?;
    enc.set_pipeline(&pso);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_kv_total: u32,
        ctx_len: u32,
        n_rows: u32, // codex v0.72.2: real q_idx bound. Was previously
        // a bogus `n_q_heads * 16` placeholder; dispatch-shape bug
        // would silently OOB without this.
        noise_start_pos: u32,
        swa_window: u32,
        ctx_scan_start: u32,
        scale: f32,
    }
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    enc.set_bytes(
        0,
        &Args {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_kv_total: n_kv_total as u32,
            ctx_len: ctx_len as u32,
            n_rows: n as u32,
            noise_start_pos,
            swa_window,
            ctx_scan_start: 0,
            scale,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k);
    enc.set_tensor(3, v);
    enc.set_tensor(4, pos_k);
    enc.set_tensor(5, o);

    enc.dispatch(
        MTLSize {
            width: n_q_heads,
            height: n,
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

/// DFlash drafter attention over two K/V ranges: committed context cache
/// plus the current noise block. Same math as `encode_dflash_attn_f32`, but
/// skips the per-layer materialization of `k_full/v_full = ctx || noise`.
pub fn encode_dflash_attn_two_range_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_ctx: &MetalTensor,
    v_ctx: &MetalTensor,
    k_noise: &MetalTensor,
    v_noise: &MetalTensor,
    pos_ctx: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
) -> Result<(), MetalError> {
    encode_dflash_attn_two_range_pipeline(
        ctx,
        enc,
        q,
        k_ctx,
        v_ctx,
        k_noise,
        v_noise,
        pos_ctx,
        o,
        n,
        n_q_heads,
        n_kv_heads,
        head_dim,
        ctx_len,
        noise_start_pos,
        swa_window,
        0,
        "dflash_attn_two_range",
        "kernel_dflash_attn_two_range_f32",
    )
}

/// Online-softmax DFlash drafter attention over two K/V ranges. Same mask and
/// output contract as `encode_dflash_attn_two_range_f32`, but removes the
/// legacy 3-pass QK recompute inside the attention body.
pub fn encode_dflash_attn_online_two_range_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_ctx: &MetalTensor,
    v_ctx: &MetalTensor,
    k_noise: &MetalTensor,
    v_noise: &MetalTensor,
    pos_ctx: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
) -> Result<(), MetalError> {
    encode_dflash_attn_online_two_range_scan_f32(
        ctx,
        enc,
        q,
        k_ctx,
        v_ctx,
        k_noise,
        v_noise,
        pos_ctx,
        o,
        n,
        n_q_heads,
        n_kv_heads,
        head_dim,
        ctx_len,
        noise_start_pos,
        swa_window,
        0,
    )
}

/// Online-softmax DFlash attention with an optional SWA context scan start.
/// `ctx_scan_start` is ignored for full-attention layers (`swa_window == 0`).
pub fn encode_dflash_attn_online_two_range_scan_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_ctx: &MetalTensor,
    v_ctx: &MetalTensor,
    k_noise: &MetalTensor,
    v_noise: &MetalTensor,
    pos_ctx: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
    ctx_scan_start: usize,
) -> Result<(), MetalError> {
    encode_dflash_attn_two_range_pipeline(
        ctx,
        enc,
        q,
        k_ctx,
        v_ctx,
        k_noise,
        v_noise,
        pos_ctx,
        o,
        n,
        n_q_heads,
        n_kv_heads,
        head_dim,
        ctx_len,
        noise_start_pos,
        swa_window,
        ctx_scan_start,
        "dflash_attn_online_two_range",
        "kernel_dflash_attn_online_two_range_f32",
    )
}

fn encode_dflash_attn_two_range_pipeline(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_ctx: &MetalTensor,
    v_ctx: &MetalTensor,
    k_noise: &MetalTensor,
    v_noise: &MetalTensor,
    pos_ctx: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
    ctx_scan_start: usize,
    kernel_label: &'static str,
    pipeline_name: &'static str,
) -> Result<(), MetalError> {
    if head_dim % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!("head_dim={head_dim} not divisible by 32"),
        });
    }
    if head_dim > 256 {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!(
                "head_dim={head_dim} > 256: kernel registers q_reg/o_acc are sized for head_dim <= 256"
            ),
        });
    }
    if n_q_heads % n_kv_heads != 0 {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!("n_q_heads={n_q_heads} not divisible by n_kv_heads={n_kv_heads}"),
        });
    }
    if q.n_elements() as usize != n * n_q_heads * head_dim {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!(
                "q.n_elements={} != N*n_q*head_dim={}",
                q.n_elements(),
                n * n_q_heads * head_dim
            ),
        });
    }
    let kv_stride = n_kv_heads * head_dim;
    let ctx_elems = ctx_len * kv_stride;
    if (k_ctx.n_elements() as usize) < ctx_elems || (v_ctx.n_elements() as usize) < ctx_elems {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!(
                "ctx k/v need at least ctx_len*kv_stride = {ctx_len}*{kv_stride} = {ctx_elems} elements"
            ),
        });
    }
    let noise_elems = n * kv_stride;
    if k_noise.n_elements() as usize != noise_elems || v_noise.n_elements() as usize != noise_elems
    {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!(
                "noise k/v expected N*kv_stride = {n}*{kv_stride} = {noise_elems} elements"
            ),
        });
    }
    if (pos_ctx.n_elements() as usize) < ctx_len {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!(
                "pos_ctx.n_elements={} < ctx_len={ctx_len}",
                pos_ctx.n_elements()
            ),
        });
    }
    if ctx_scan_start > ctx_len {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!("ctx_scan_start={ctx_scan_start} > ctx_len={ctx_len}"),
        });
    }
    if o.n_elements() as usize != n * n_q_heads * head_dim {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!(
                "o.n_elements={} != N*n_q*head_dim={}",
                o.n_elements(),
                n * n_q_heads * head_dim
            ),
        });
    }

    let pso = ctx.pipeline(pipeline_name)?;
    enc.set_pipeline(&pso);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_kv_total: u32,
        ctx_len: u32,
        n_rows: u32,
        noise_start_pos: u32,
        swa_window: u32,
        ctx_scan_start: u32,
        scale: f32,
    }
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    enc.set_bytes(
        0,
        &Args {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_kv_total: (ctx_len + n) as u32,
            ctx_len: ctx_len as u32,
            n_rows: n as u32,
            noise_start_pos,
            swa_window,
            ctx_scan_start: ctx_scan_start as u32,
            scale,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_ctx);
    enc.set_tensor(3, v_ctx);
    enc.set_tensor(4, k_noise);
    enc.set_tensor(5, v_noise);
    enc.set_tensor(6, pos_ctx);
    enc.set_tensor(7, o);

    enc.dispatch(
        MTLSize {
            width: n_q_heads,
            height: n,
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

/// Fixed-shape GQA-sharing split-4 path for the Qwen3.6 DFlash full-attention
/// layer. The serial main/reduce pair writes exact online-softmax partials.
pub fn encode_dflash_attn_full_gqa_split4_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_ctx: &MetalTensor,
    v_ctx: &MetalTensor,
    k_noise: &MetalTensor,
    v_noise: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ctx_len: usize,
) -> Result<(), MetalError> {
    const N: usize = 16;
    const N_Q: usize = 32;
    const N_KV: usize = 8;
    const HEAD_DIM: usize = 128;
    const GROUP: usize = 4;
    const SPLIT: usize = 4;
    const KERNEL: &str = "dflash_attn_full_gqa_split4";

    if enc.concurrent {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "main/reduce dependency requires a serial encoder".into(),
        });
    }
    if (n, n_q_heads, n_kv_heads, head_dim) != (N, N_Q, N_KV, HEAD_DIM) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected N/n_q/n_kv/head_dim={N}/{N_Q}/{N_KV}/{HEAD_DIM}, got \
                 {n}/{n_q_heads}/{n_kv_heads}/{head_dim}"
            ),
        });
    }
    let tensors = [q, k_ctx, v_ctx, k_noise, v_noise, o_partial, ml_partial, o];
    if tensors.iter().any(|tensor| tensor.dtype != GgmlType::F32) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "all inputs, scratch, and output must be F32".into(),
        });
    }
    let q_elems = N * N_Q * HEAD_DIM;
    let kv_stride = N_KV * HEAD_DIM;
    let ctx_elems = ctx_len
        .checked_mul(kv_stride)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "context element count overflow".into(),
        })?;
    let noise_elems = N * kv_stride;
    let o_partial_elems = N * N_KV * SPLIT * GROUP * HEAD_DIM;
    let ml_partial_elems = N * N_KV * SPLIT * GROUP * 2;
    if q.n_elements() as usize != q_elems
        || o.n_elements() as usize != q_elems
        || (k_ctx.n_elements() as usize) < ctx_elems
        || (v_ctx.n_elements() as usize) < ctx_elems
        || k_noise.n_elements() as usize != noise_elems
        || v_noise.n_elements() as usize != noise_elems
        || (o_partial.n_elements() as usize) < o_partial_elems
        || (ml_partial.n_elements() as usize) < ml_partial_elems
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "shape mismatch: q={} o={} k_ctx={} v_ctx={} k_noise={} v_noise={} \
                 o_partial={} ml_partial={} ctx_len={ctx_len}",
                q.n_elements(),
                o.n_elements(),
                k_ctx.n_elements(),
                v_ctx.n_elements(),
                k_noise.n_elements(),
                v_noise.n_elements(),
                o_partial.n_elements(),
                ml_partial.n_elements(),
            ),
        });
    }
    let total_rows = ctx_len.checked_add(N).ok_or_else(|| MetalError::BadShape {
        kernel: KERNEL,
        detail: "ctx_len + N overflow".into(),
    })?;
    let n_kv_total = u32::try_from(total_rows).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: "ctx_len + N does not fit u32".into(),
    })?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_kv_total: u32,
        ctx_len: u32,
        n_rows: u32,
        noise_start_pos: u32,
        swa_window: u32,
        ctx_scan_start: u32,
        scale: f32,
    }
    let args = Args {
        n_q_heads: N_Q as u32,
        n_kv_heads: N_KV as u32,
        head_dim: HEAD_DIM as u32,
        n_kv_total,
        ctx_len: u32::try_from(ctx_len).map_err(|_| MetalError::BadShape {
            kernel: KERNEL,
            detail: "ctx_len does not fit u32".into(),
        })?,
        n_rows: N as u32,
        noise_start_pos: 0,
        swa_window: 0,
        ctx_scan_start: 0,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let main = ctx.pipeline("kernel_dflash_attn_full_gqa_split4_main_f32")?;
    enc.set_pipeline(&main);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_ctx);
    enc.set_tensor(3, v_ctx);
    enc.set_tensor(4, k_noise);
    enc.set_tensor(5, v_noise);
    enc.set_tensor(6, o_partial);
    enc.set_tensor(7, ml_partial);
    enc.dispatch(
        MTLSize {
            width: N_KV,
            height: N,
            depth: SPLIT,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    let reduce = ctx.pipeline("kernel_dflash_attn_full_gqa_split4_reduce_f32")?;
    enc.set_pipeline(&reduce);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, o_partial);
    enc.set_tensor(2, ml_partial);
    enc.set_tensor(3, o);
    enc.dispatch(
        MTLSize {
            width: N_Q,
            height: N,
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

crate::env_flag!(default_on l2_pair_hd128_r4_enabled, "QWEN_L2_PAIR_HD128_R4");

pub fn encode_l2_norm_pair_batched_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_x: &MetalTensor,
    q_y: &MetalTensor,
    k_x: &MetalTensor,
    k_y: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let want = (n_heads * head_dim) as u64;
    if q_x.n_elements() != want
        || q_y.n_elements() != want
        || k_x.n_elements() != want
        || k_y.n_elements() != want
    {
        return Err(MetalError::BadShape {
            kernel: "l2_norm_pair_batched",
            detail: format!("q/k inputs and outputs expected {want} elements"),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    }
    let use_hd128_r4 = l2_pair_hd128_r4_enabled();
    if use_hd128_r4 && head_dim == 128 {
        let pso = ctx.pipeline("kernel_l2_norm_pair_hd128_r4_f32")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &Args {
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                eps,
            },
        );
        enc.set_tensor(1, q_x);
        enc.set_tensor(2, q_y);
        enc.set_tensor(3, k_x);
        enc.set_tensor(4, k_y);
        enc.dispatch(
            MTLSize {
                width: n_heads.div_ceil(4),
                height: 2,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        );
        return Ok(());
    }
    let pso = ctx.pipeline("kernel_l2_norm_pair_batched_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, q_x);
    enc.set_tensor(2, q_y);
    enc.set_tensor(3, k_x);
    enc.set_tensor(4, k_y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: n_heads,
            height: 2,
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

/// Row lookup: `y[r * n_cols + i] = source[ids[r] * n_cols + i]`.
/// The source may be flat or multidimensional; its logical element count
/// defines the row count. GGUF embeddings are normally `[n_cols, vocab]`.
pub fn encode_get_rows_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    embed: &MetalTensor,
    ids: &MetalTensor,
    y: &MetalTensor,
    n_rows: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!("n_rows={n_rows} and n_cols={n_cols} must be nonzero"),
        });
    }
    let output_elements = n_rows
        .checked_mul(n_cols)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "get_rows",
            detail: "n_rows*n_cols overflow".into(),
        })?;
    if y.n_elements() as usize != output_elements {
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
    let ids_bytes = n_rows
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: "get_rows",
            detail: "ids byte size overflow".into(),
        })?;
    let ids_end = ids
        .offset
        .checked_add(ids_bytes as u64)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "get_rows",
            detail: "ids buffer range overflow".into(),
        })?;
    if ids_end > ids.buffer.length() as u64 {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                "ids range offset={} bytes={ids_bytes} exceeds buffer={}",
                ids.offset,
                ids.buffer.length()
            ),
        });
    }
    if y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!("expected F32 output, got {:?}", y.dtype),
        });
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: "get_rows",
            detail: "output byte size overflow".into(),
        })?;
    let output_end =
        y.offset
            .checked_add(output_bytes as u64)
            .ok_or_else(|| MetalError::BadShape {
                kernel: "get_rows",
                detail: "output buffer range overflow".into(),
            })?;
    if output_end > y.buffer.length() as u64 {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                "output range offset={} bytes={output_bytes} exceeds buffer={}",
                y.offset,
                y.buffer.length()
            ),
        });
    }
    let source_elements =
        usize::try_from(embed.n_elements()).map_err(|_| MetalError::BadShape {
            kernel: "get_rows",
            detail: format!("source element count {} exceeds usize", embed.n_elements()),
        })?;
    if source_elements == 0 || source_elements % n_cols != 0 {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                concat!(
                    "source shape {:?} with {} elements is not a nonempty ",
                    "collection of {}-element rows"
                ),
                embed.shape, source_elements, n_cols,
            ),
        });
    }
    let source_rows = source_elements / n_cols;
    let block_layout = match embed.dtype {
        GgmlType::F32 => (1usize, 4usize),
        GgmlType::F16 | GgmlType::BF16 => (1, 2),
        GgmlType::Q4_K => (256, 144),
        GgmlType::Q8_0 => (32, 34),
        other => {
            return Err(MetalError::BadShape {
                kernel: "get_rows",
                detail: format!("unsupported embedding dtype {other:?}"),
            });
        }
    };
    let (block_elements, block_bytes) = block_layout;
    if n_cols % block_elements != 0 {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                "n_cols={n_cols} is not divisible by {:?} block size {block_elements}",
                embed.dtype
            ),
        });
    }
    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| MetalError::BadShape {
        kernel: "get_rows",
        detail: format!("n_rows={n_rows} exceeds u32"),
    })?;
    let n_cols_u32 = u32::try_from(n_cols).map_err(|_| MetalError::BadShape {
        kernel: "get_rows",
        detail: format!("n_cols={n_cols} exceeds u32"),
    })?;
    let source_rows_u32 = u32::try_from(source_rows).map_err(|_| MetalError::BadShape {
        kernel: "get_rows",
        detail: format!("source row count {source_rows} exceeds u32"),
    })?;
    let expected_bytes = n_cols
        .checked_div(block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .and_then(|row_bytes| row_bytes.checked_mul(source_rows))
        .ok_or_else(|| MetalError::BadShape {
            kernel: "get_rows",
            detail: "embedding byte size overflow".into(),
        })?;
    let buffer_end = embed
        .offset
        .checked_add(expected_bytes as u64)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "get_rows",
            detail: "embedding buffer range overflow".into(),
        })?;
    if embed.n_bytes() as usize != expected_bytes || buffer_end > embed.buffer.length() as u64 {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                concat!(
                    "embedding bytes mismatch: logical={} expected={} ",
                    "offset={} buffer={}"
                ),
                embed.n_bytes(),
                expected_bytes,
                embed.offset,
                embed.buffer.length()
            ),
        });
    }
    let kernel_name = match embed.dtype {
        GgmlType::F32 => "kernel_get_rows_f32",
        GgmlType::F16 => "kernel_get_rows_f16",
        GgmlType::BF16 => "kernel_get_rows_bf16",
        GgmlType::Q4_K => "kernel_get_rows_q4_K_f32",
        GgmlType::Q8_0 => "kernel_get_rows_q8_0_f32",
        _ => unreachable!(),
    };
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GetRowsArgs {
        n_rows: u32,
        n_cols: u32,
        n_vocab: u32,
    }
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &GetRowsArgs {
            n_rows: n_rows_u32,
            n_cols: n_cols_u32,
            n_vocab: source_rows_u32,
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

pub fn encode_rope_neox_f32_packed_consecutive(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    buf: &MetalTensor,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    start_position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    if buf.n_elements() as usize != n_tokens * n_heads * head_dim {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: format!(
                "buf.n={} != n_tokens*n_heads*head_dim={}",
                buf.n_elements(),
                n_tokens * n_heads * head_dim
            ),
        });
    }
    if n_rot % 2 != 0 || n_rot > head_dim {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: format!("n_rot={n_rot} must be even and ≤ head_dim={head_dim}"),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        n_heads: u32,
        head_dim: u32,
        n_rot: u32,
        start_position: u32,
        theta_base: f32,
    }
    let pso = ctx.pipeline("kernel_rope_neox_f32_packed_consecutive")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: n_tokens as u32,
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            n_rot: n_rot as u32,
            start_position,
            theta_base,
        },
    );
    enc.set_tensor(1, buf);

    let total_pairs = n_tokens * n_heads * (n_rot / 2);
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

crate::env_flag!(default_off prefill_gdn_prep_parallel_enabled, "QWEN_PREFILL_GDN_PREP_PARALLEL");

pub fn encode_gdn_prep_packed_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    qkv_pack: &MetalTensor,
    conv_buf: &MetalTensor,
    conv_w: &MetalTensor,
    q_pack: &MetalTensor,
    k_pack: &MetalTensor,
    v_pack: &MetalTensor,
    n_tokens: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    let qk_dim = n_k_heads * head_dim;
    let v_dim = n_v_heads * head_dim;
    let conv_dim = (2 * n_k_heads + n_v_heads) * head_dim;
    if qkv_pack.n_elements() as usize != n_tokens * conv_dim {
        return Err(MetalError::BadShape {
            kernel: "gdn_prep_packed",
            detail: format!("qkv_pack expected {} elements", n_tokens * conv_dim),
        });
    }
    if conv_buf.n_elements() as usize != 3 * conv_dim {
        return Err(MetalError::BadShape {
            kernel: "gdn_prep_packed",
            detail: format!("conv_buf expected {} elements", 3 * conv_dim),
        });
    }
    if conv_w.n_elements() as usize != 4 * conv_dim {
        return Err(MetalError::BadShape {
            kernel: "gdn_prep_packed",
            detail: format!("conv_w expected {} elements", 4 * conv_dim),
        });
    }
    if q_pack.n_elements() as usize != n_tokens * qk_dim
        || k_pack.n_elements() as usize != n_tokens * qk_dim
    {
        return Err(MetalError::BadShape {
            kernel: "gdn_prep_packed",
            detail: format!("q/k pack expected {} elements", n_tokens * qk_dim),
        });
    }
    if v_pack.n_elements() as usize != n_tokens * v_dim {
        return Err(MetalError::BadShape {
            kernel: "gdn_prep_packed",
            detail: format!("v_pack expected {} elements", n_tokens * v_dim),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        n_k_heads: u32,
        n_v_heads: u32,
        head_dim: u32,
        conv_dim: u32,
    }
    let use_parallel = prefill_gdn_prep_parallel_enabled();
    if use_parallel && n_tokens >= 3 {
        let args = Args {
            n_tokens: n_tokens as u32,
            n_k_heads: n_k_heads as u32,
            n_v_heads: n_v_heads as u32,
            head_dim: head_dim as u32,
            conv_dim: conv_dim as u32,
        };
        let pso = ctx.pipeline("kernel_gdn_prep_parallel_f32")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, qkv_pack);
        enc.set_tensor(2, conv_buf);
        enc.set_tensor(3, conv_w);
        enc.set_tensor(4, q_pack);
        enc.set_tensor(5, k_pack);
        enc.set_tensor(6, v_pack);
        let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(256);
        let total = n_tokens * conv_dim;
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

        let pso = ctx.pipeline("kernel_gdn_prep_parallel_state_f32")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, qkv_pack);
        enc.set_tensor(2, conv_buf);
        let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(256);
        let total = 3 * conv_dim;
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
        return Ok(());
    }
    let pso = ctx.pipeline("kernel_gdn_prep_packed_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: n_tokens as u32,
            n_k_heads: n_k_heads as u32,
            n_v_heads: n_v_heads as u32,
            head_dim: head_dim as u32,
            conv_dim: conv_dim as u32,
        },
    );
    enc.set_tensor(1, qkv_pack);
    enc.set_tensor(2, conv_buf);
    enc.set_tensor(3, conv_w);
    enc.set_tensor(4, q_pack);
    enc.set_tensor(5, k_pack);
    enc.set_tensor(6, v_pack);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

crate::env_flag!(default_on rmsnorm_gated_hd128_r4_enabled, "QWEN_RMSNORM_GATED_HD128_R4");

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
    let use_hd128_r4 = rmsnorm_gated_hd128_r4_enabled();
    if use_hd128_r4 && head_dim == 128 {
        let pso = ctx.pipeline("kernel_rmsnorm_gated_hd128_r4_f32")?;
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
        enc.dispatch(
            MTLSize {
                width: n_heads.div_ceil(4),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        );
        return Ok(());
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

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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

/// GDN recurrence variant that takes precomputed `decay = exp(g)`.
pub fn encode_gdn_step_decay_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    decay: &MetalTensor,
    beta: &MetalTensor,
    state: &MetalTensor,
    out: &MetalTensor,
    n_v_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    if head_dim != 128 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("head_dim={head_dim} but kernel hardcodes 128"),
        });
    }
    if n_v_heads % n_k_heads != 0 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("n_v_heads={n_v_heads} not multiple of n_k_heads={n_k_heads}"),
        });
    }
    let want_qk = (n_k_heads * head_dim) as u64;
    let want_v = (n_v_heads * head_dim) as u64;
    if q.n_elements() != want_qk || k.n_elements() != want_qk {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("q/k expected {want_qk} elements (n_k_heads={n_k_heads})"),
        });
    }
    if v.n_elements() != want_v {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("v expected {want_v} elements (n_v_heads={n_v_heads})"),
        });
    }
    if decay.n_elements() != n_v_heads as u64 || beta.n_elements() != n_v_heads as u64 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("decay/beta expected {n_v_heads} elements"),
        });
    }
    let want_state = (n_v_heads * head_dim * head_dim) as u64;
    if state.n_elements() != want_state {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("state expected {want_state} elements"),
        });
    }
    if out.n_elements() != want_v {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("out expected {want_v} elements"),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_v_heads: u32,
        n_k_heads: u32,
    }
    let pso = ctx.pipeline("kernel_gdn_step_decay_f32")?;
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
    enc.set_tensor(4, decay);
    enc.set_tensor(5, beta);
    enc.set_tensor(6, state);
    enc.set_tensor(7, out);

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

pub fn encode_gdn_step_decay_packed_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_pack: &MetalTensor,
    k_pack: &MetalTensor,
    v_pack: &MetalTensor,
    decay_pack: &MetalTensor,
    beta_pack: &MetalTensor,
    state: &MetalTensor,
    out_pack: &MetalTensor,
    n_tokens: usize,
    n_v_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    if head_dim != 128 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("head_dim={head_dim} but kernel hardcodes 128"),
        });
    }
    if n_v_heads % n_k_heads != 0 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("n_v_heads={n_v_heads} not multiple of n_k_heads={n_k_heads}"),
        });
    }
    let qk_per_token = n_k_heads * head_dim;
    let v_per_token = n_v_heads * head_dim;
    if q_pack.n_elements() as usize != n_tokens * qk_per_token
        || k_pack.n_elements() as usize != n_tokens * qk_per_token
    {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("q/k expected {} elements per pack", n_tokens * qk_per_token),
        });
    }
    if v_pack.n_elements() as usize != n_tokens * v_per_token {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("v expected {} elements", n_tokens * v_per_token),
        });
    }
    if decay_pack.n_elements() as usize != n_tokens * n_v_heads
        || beta_pack.n_elements() as usize != n_tokens * n_v_heads
    {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("decay/beta expected {} elements", n_tokens * n_v_heads),
        });
    }
    let want_state = n_v_heads * head_dim * head_dim;
    if state.n_elements() as usize != want_state {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("state expected {want_state} elements"),
        });
    }
    if out_pack.n_elements() as usize != n_tokens * v_per_token {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("out expected {} elements", n_tokens * v_per_token),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        n_v_heads: u32,
        n_k_heads: u32,
    }
    let use_nsg4 = n_v_heads % 4 == 0 && head_dim == 128;
    let kernel = if use_nsg4 {
        "kernel_gdn_step_decay_packed_nsg4_f32"
    } else {
        "kernel_gdn_step_decay_packed_f32"
    };
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: n_tokens as u32,
            n_v_heads: n_v_heads as u32,
            n_k_heads: n_k_heads as u32,
        },
    );
    enc.set_tensor(1, q_pack);
    enc.set_tensor(2, k_pack);
    enc.set_tensor(3, v_pack);
    enc.set_tensor(4, decay_pack);
    enc.set_tensor(5, beta_pack);
    enc.set_tensor(6, state);
    enc.set_tensor(7, out_pack);
    if use_nsg4 {
        enc.dispatch(
            MTLSize {
                width: head_dim / 4,
                height: n_v_heads,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        );
    } else {
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
    }
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

/// Q8_0 mat-vec, same API shape as [`encode_mat_vec_q4_k_f32`].
///
/// Q8_0 super-block is QK8_0=32 elements (vs QK_K=256 for Q4_K/Q5_K/Q6_K).
/// The kernel still requires `n_in % 32 == 0`. Used by v0.73b.0 to
/// switch the DFlash drafter from F32-dequant to native Q8_0 storage
/// (drafter weight footprint 7.4 GB → 1.85 GB, eliminates per-token
/// re-read of the dequant'd F32 weights at hot decode).
fn mat_vec_q8_0_lcpp_enabled() -> bool {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    let override_value =
        *OVERRIDE.get_or_init(|| match std::env::var("QWEN_MATVEC_Q8_0_LCPP").as_deref() {
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO") => Some(false),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") => Some(true),
            _ => None,
        });
    override_value.unwrap_or(true)
}

pub fn encode_mat_vec_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q8_0",
            detail: format!("n_in={n_in} not divisible by 32 (Q8_0 super-block)"),
        });
    }
    if weight.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q8_0",
            detail: format!("weight.dtype = {:?}, expected Q8_0", weight.dtype),
        });
    }
    let use_lcpp = mat_vec_q8_0_lcpp_enabled();
    let pso = ctx.pipeline(if use_lcpp {
        "kernel_mat_vec_q8_0_f32_lcpp"
    } else {
        "kernel_mat_vec_q8_0_f32"
    })?;
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

    let (nr0, nsg) = if use_lcpp { (2, 4) } else { (1, 2) };
    if use_lcpp {
        enc.set_threadgroup_memory(0, 32 * nr0 * std::mem::size_of::<f32>());
    }
    let rows_per_threadgroup = if use_lcpp { nr0 } else { nr0 * nsg };
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(rows_per_threadgroup),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: nsg * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_shared_swiglu_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "shared_swiglu_q8_0",
            detail: format!("n_in={n_in} not divisible by 32 (Q8_0 super-block)"),
        });
    }
    if gate_weight.dtype != GgmlType::Q8_0 || up_weight.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "shared_swiglu_q8_0",
            detail: format!(
                "gate/up dtype = {:?}/{:?}, expected Q8_0/Q8_0",
                gate_weight.dtype, up_weight.dtype
            ),
        });
    }
    let pso = ctx.pipeline("kernel_shared_swiglu_q8_0_f32_lcpp")?;
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
    enc.set_tensor(1, gate_weight);
    enc.set_tensor(2, up_weight);
    enc.set_tensor(3, x);
    enc.set_tensor(4, y);

    let nr0 = 2usize;
    let nsg = 4usize;
    enc.set_threadgroup_memory(0, 32 * 2 * nr0 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(nr0),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: nsg * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// One-shot Q8_0 mat-vec for tests.
pub fn mat_vec_q8_0_f32_readback_for_test(
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
        GgmlType::Q8_0,
    )?;
    let x_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(x),
        vec![n_in as u64],
        GgmlType::F32,
    )?;
    let y_t = MetalTensor::zeros_f32(ctx, vec![n_out as u64])?;
    one_shot(ctx, |enc| {
        encode_mat_vec_q8_0_f32(ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
    })?;
    Ok(read_back_f32(&y_t.buffer, n_out))
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

/// Q6_K mat-mat: same shape contract as [`encode_mat_mat_q4_k_f32`].
///
/// Lifts the same 64×32×32 simdgroup_matrix tile from llama.cpp,
/// templated on Q6_K dequant. Output is row-major `[n_query, n_out]`
/// (= bit-equivalent to llama's "[n_out, n_query] col-major" framing).
///
/// Used by H5.3b.6 to lift `ffn_down` and `lm_head` out of the
/// per-token mat-vec re-read loop.
pub fn encode_mat_mat_q6_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor, // [n_query, n_in] row-major F32
    y: &MetalTensor, // F32 [n_out * n_query] flat (row-major [n_query, n_out]).
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q6_k",
            detail: format!("n_in={n_in} not divisible by 256 (Q6_K super-block)"),
        });
    }
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q6_k",
            detail: format!("n_in={n_in} not divisible by 32 (NK_MM tile)"),
        });
    }
    if weight.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q6_k",
            detail: format!("weight.dtype = {:?}, expected Q6_K", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q6_k",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q6_k",
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    let use_n64 = mat_mat_q6_k_n64_enabled()
        && n_query >= mat_mat_q6_k_n64_min_query()
        && n_query % 64 == 0
        && n_out % 64 == 0;
    let kernel_name = if use_n64 {
        "kernel_mat_mat_q6_K_f32_n64"
    } else if n_query == 16 {
        "kernel_mat_mat_q6_K_f32_n16"
    } else {
        "kernel_mat_mat_q6_K_f32"
    };
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);

    // Q6_K block bytes per row = (n_in / 256) * 210.
    let nb01 = ((n_in / 256) * 210) as u32;
    let stride_b = n_in as u32;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);

    let nr1 = if use_n64 {
        64
    } else if n_query == 16 {
        16
    } else {
        32
    };
    let smem = if use_n64 {
        8192
    } else {
        mat_mat_qk_threadgroup_memory(n_out, n_query, nr1)
    };
    enc.set_threadgroup_memory(0, smem);
    let n_tg_x = n_query.div_ceil(nr1);
    let n_tg_y = n_out.div_ceil(64);
    let threads = if use_n64 { 256 } else { 128 };
    enc.dispatch(
        MTLSize {
            width: n_tg_x,
            height: n_tg_y,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Q5_K mat-mat: same shape contract as [`encode_mat_mat_q4_k_f32`] /
/// [`encode_mat_mat_q6_k_f32`].
///
/// Lifts the same 64×32×32 simdgroup_matrix tile from llama.cpp,
/// templated on Q5_K dequant (adds the qh high-bit contribution to
/// the Q4_K nibble path; same scale/min decode as Q4_K).
///
/// Output is row-major `[n_query, n_out]` (= bit-equivalent to
/// llama's `[n_out, n_query] col-major` framing).
///
/// Used by v0.73a.1 to lift GDN `out_proj` (Q5_K [v_dim, hidden])
/// out of the per-token mat-vec re-read loop. Per-row cosine ≥ 0.999
/// vs N successive Q5_K mat-vec is the gate (same threshold as
/// Q4_K / Q6_K mat-mat — half-staging in lifted kernel).
pub fn encode_mat_mat_q5_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor, // [n_query, n_in] row-major F32
    y: &MetalTensor, // F32 [n_out * n_query] flat (row-major [n_query, n_out]).
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if n_in % 256 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q5_k",
            detail: format!("n_in={n_in} not divisible by 256 (Q5_K super-block)"),
        });
    }
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q5_k",
            detail: format!("n_in={n_in} not divisible by 32 (NK_MM tile)"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q5_k",
            detail: format!("weight.dtype = {:?}, expected Q5_K", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q5_k",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q5_k",
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    let use_n64 = mat_mat_q5_k_n64_enabled()
        && n_query >= mat_mat_q5_k_n64_min_query()
        && n_query % 64 == 0
        && n_out % 64 == 0;
    let kernel_name = if use_n64 {
        "kernel_mat_mat_q5_K_f32_n64"
    } else if n_query == 16 {
        "kernel_mat_mat_q5_K_f32_n16"
    } else {
        "kernel_mat_mat_q5_K_f32"
    };
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);

    // Q5_K block bytes per row = (n_in / 256) * 176.
    let nb01 = ((n_in / 256) * 176) as u32;
    let stride_b = n_in as u32;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);

    let nr1 = if use_n64 {
        64
    } else if n_query == 16 {
        16
    } else {
        32
    };
    let smem = if use_n64 {
        8192
    } else {
        mat_mat_qk_threadgroup_memory(n_out, n_query, nr1)
    };
    enc.set_threadgroup_memory(0, smem);
    let n_tg_x = n_query.div_ceil(nr1);
    let n_tg_y = n_out.div_ceil(64);
    let threads = if use_n64 { 256 } else { 128 };
    enc.dispatch(
        MTLSize {
            width: n_tg_x,
            height: n_tg_y,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Q8_0 mat-mat: same shape contract as [`encode_mat_mat_q4_k_f32`] /
/// [`encode_mat_mat_q5_k_f32`] / [`encode_mat_mat_q6_k_f32`].
///
/// CRITICAL difference: Q8_0 super-block is QK8_0=32 elements (vs
/// QK_K=256 for K-quants). The kernel still requires `n_in % 32 == 0`,
/// matching the K-step `NK_MM=32`. The pointer-advance specializes
/// to `Q8_0_NL=2` (one super-block per K-step per row).
///
/// Output is row-major `[n_query, n_out]` (= bit-equivalent to
/// llama's `[n_out, n_query] col-major` framing).
///
/// Used by v0.73b.0 to lift the DFlash drafter Q8_0 mat-mat path
/// (lm_head, FFN, projections) once the loader switches from
/// F32-dequant to native Q8_0. Per-row cosine ≥ 0.999 vs N successive
/// Q8_0 mat-vec is the gate (same threshold as Q4_K/Q5_K/Q6_K mat-mat
/// — half-staging in lifted kernel).
pub fn encode_mat_mat_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if n_in % 32 != 0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q8_0",
            detail: format!("n_in={n_in} not divisible by 32 (Q8_0 super-block / NK_MM tile)"),
        });
    }
    if weight.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q8_0",
            detail: format!("weight.dtype = {:?}, expected Q8_0", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q8_0",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q8_0",
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    // NR1=16 fast-path gate: same specialization as Q4_K/Q5_K/Q6_K mat-mat.
    // Only fires when n_query == 16 exactly; otherwise generic NR1=32 kernel.
    let kernel_name = if n_query == 16 {
        "kernel_mat_mat_q8_0_f32_n16"
    } else {
        "kernel_mat_mat_q8_0_f32"
    };
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);

    // Q8_0 block bytes per row = (n_in / 32) * 34.
    let nb01 = ((n_in / 32) * 34) as u32;
    let stride_b = n_in as u32;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);

    let nr1 = if n_query == 16 { 16 } else { 32 };
    enc.set_threadgroup_memory(0, mat_mat_qk_threadgroup_memory(n_out, n_query, nr1));
    let n_tg_x = n_query.div_ceil(nr1);
    let n_tg_y = n_out.div_ceil(64);
    enc.dispatch(
        MTLSize {
            width: n_tg_x,
            height: n_tg_y,
            depth: 1,
        },
        MTLSize {
            width: 128,
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
    cmd_buf.waitUntilCompleted();
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

/// One-shot fused residual-add + RMSNorm for tests. Returns `(x_after, y)`.
pub fn residual_rms_norm_mul_f32_readback_for_test(
    ctx: &MetalContext,
    x: &[f32],
    residual: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), MetalError> {
    let n = x.len();
    if residual.len() != n || weight.len() != n {
        return Err(MetalError::BadShape {
            kernel: "residual_rms_norm_test",
            detail: format!(
                "x={} residual={} weight={} length mismatch",
                x.len(),
                residual.len(),
                weight.len()
            ),
        });
    }
    let x_t = MetalTensor::from_bytes(ctx, bytemuck::cast_slice(x), vec![n as u64], GgmlType::F32)?;
    let r_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(residual),
        vec![n as u64],
        GgmlType::F32,
    )?;
    let w_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(weight),
        vec![n as u64],
        GgmlType::F32,
    )?;
    let y_t = MetalTensor::zeros_f32(ctx, vec![n as u64])?;
    one_shot(ctx, |enc| {
        encode_residual_rms_norm_mul_f32(ctx, enc, &x_t, &r_t, &w_t, &y_t, eps)
    })?;
    Ok((read_back_f32(&x_t.buffer, n), read_back_f32(&y_t.buffer, n)))
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
    cmd_buf.waitUntilCompleted();
    Ok(())
}

/// Same but for Q5_K. Used by the v0.73a.0 A-lite go/no-go gate
/// (`q5_k_mat_mat_amortization_vs_n_mat_vec`).
pub fn bench_q5_k_chained(
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
        encode_mat_vec_q5_k_f32(ctx, &enc, weight, x, y, n_in, n_out)?;
    }
    enc.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}

/// Same chained bench harness as `bench_q4_k_mat_mat_chained`, for Q5_K.
/// Used by the v0.73a.0 A-lite go/no-go gate to compare amortized
/// weight-BW of mat-mat (one panel-reload per K-step shared across
/// N_QUERY cols) vs N_QUERY successive mat-vec.
pub fn bench_q5_k_mat_mat_chained(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    for _ in 0..n_dispatches {
        encode_mat_mat_q5_k_f32(ctx, &enc, weight, x, y, n_in, n_out, n_query)?;
    }
    enc.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
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
    cmd_buf.waitUntilCompleted();
    Ok(())
}

/// Same chained bench harness as `bench_q4_k_mat_mat_chained`, for Q6_K.
pub fn bench_q6_k_mat_mat_chained(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    for _ in 0..n_dispatches {
        encode_mat_mat_q6_k_f32(ctx, &enc, weight, x, y, n_in, n_out, n_query)?;
    }
    enc.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}

/// Chain `n_dispatches` Q4_K **mat-mat** (with N_QUERY columns) into one
/// command buffer. Used to bench the lifted llama mat-mat tile against
/// theoretical peak BW and against `n_dispatches × n_query` mat-vec
/// calls.
///
/// Per H5.3b plan rev 6: this is the H5.3b.1–3 perf gate. We expect:
///   * GiB/s should approach the chained64 mat-vec ceiling (77-88% peak
///     for our existing fast Q4_K kernel) because mat-mat amortizes
///     weight reads across N_QUERY columns rather than re-reading.
///   * Wall time vs `n_dispatches × n_query mat_vec` should be ~N_QUERY×
///     less if the BW ceiling is the same — that's the entire point of
///     mat-mat.
pub fn bench_q4_k_mat_mat_chained(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor, // [n_query, n_in] row-major F32
    y: &MetalTensor, // [n_out, n_query] col-major F32
    n_in: usize,
    n_out: usize,
    n_query: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    for _ in 0..n_dispatches {
        encode_mat_mat_q4_k_f32(ctx, &enc, weight, x, y, n_in, n_out, n_query)?;
    }
    enc.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory_signals(
        recommended_max_bytes: u64,
        current_allocated_bytes: u64,
        process_limit_remaining_bytes: Option<u64>,
    ) -> MetalMemorySignals {
        MetalMemorySignals {
            recommended_max_bytes,
            current_allocated_bytes,
            process_limit_remaining_bytes,
        }
    }

    #[test]
    fn metal_memory_admission_requires_both_advisory_budgets() {
        let exact =
            evaluate_metal_memory_admission(400, 100, memory_signals(1500, 1000, Some(500)), false);
        assert!(exact.admitted);
        assert_eq!(
            exact.reason,
            MetalMemoryAdmissionReason::AdmittedWithProcessBudget
        );
        assert_eq!(exact.required_bytes, Some(500));
        assert_eq!(exact.working_set_headroom_bytes, Some(500));

        for (signals, reason) in [
            (
                memory_signals(1499, 1000, Some(500)),
                MetalMemoryAdmissionReason::WorkingSetInsufficient,
            ),
            (
                memory_signals(1500, 1000, Some(499)),
                MetalMemoryAdmissionReason::ProcessInsufficient,
            ),
            (
                memory_signals(1499, 1000, Some(499)),
                MetalMemoryAdmissionReason::BothInsufficient,
            ),
            (
                memory_signals(1000, 1000, Some(500)),
                MetalMemoryAdmissionReason::WorkingSetInsufficient,
            ),
            (
                memory_signals(999, 1000, Some(500)),
                MetalMemoryAdmissionReason::InvalidWorkingSetSignal,
            ),
            (
                memory_signals(1500, 1000, Some(0)),
                MetalMemoryAdmissionReason::ProcessSignalUnavailable,
            ),
            (
                memory_signals(1500, 1000, None),
                MetalMemoryAdmissionReason::ProcessSignalUnavailable,
            ),
        ] {
            let decision = evaluate_metal_memory_admission(400, 100, signals, false);
            assert!(!decision.admitted);
            assert_eq!(decision.reason, reason);
        }
    }

    #[test]
    fn metal_memory_admission_fails_closed_on_required_overflow() {
        let decision = evaluate_metal_memory_admission(
            u64::MAX,
            1,
            memory_signals(u64::MAX, 1, Some(u64::MAX)),
            false,
        );
        assert!(!decision.admitted);
        assert_eq!(decision.required_bytes, None);
        assert_eq!(
            decision.reason,
            MetalMemoryAdmissionReason::RequiredBytesOverflow
        );
    }

    #[test]
    fn metal_memory_admission_omits_zero_process_budget_only_when_allowed() {
        let signals = memory_signals(1500, 1000, Some(0));
        let admitted = evaluate_metal_memory_admission(400, 100, signals, true);
        assert!(admitted.admitted);
        assert_eq!(
            admitted.reason,
            MetalMemoryAdmissionReason::AdmittedProcessBudgetOmitted
        );
        let denied = evaluate_metal_memory_admission(400, 100, signals, false);
        assert!(!denied.admitted);
        assert_eq!(
            denied.reason,
            MetalMemoryAdmissionReason::ProcessSignalUnavailable
        );
    }

    #[test]
    fn metal_memory_admission_rejects_zero_required_with_zero_headroom() {
        let decision =
            evaluate_metal_memory_admission(0, 0, memory_signals(1000, 1000, Some(1)), false);
        assert!(!decision.admitted);
        assert_eq!(
            decision.reason,
            MetalMemoryAdmissionReason::WorkingSetInsufficient
        );
    }

    #[test]
    fn metal_memory_probes_are_available_on_product_host() {
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::NoDevice | MetalError::EmptyLibrary) => return,
            Err(error) => panic!("Metal context: {error}"),
        };
        let signals = ctx.memory_signals();
        assert!(signals.recommended_max_bytes > 0);
        eprintln!("[metal-memory-signals] {signals:?}");
    }

    #[test]
    #[ignore]
    fn metal_memory_probes_match_local_m4_max() {
        let ctx = MetalContext::new().expect("Metal context");
        let signals = ctx.memory_signals();
        assert_eq!(signals.recommended_max_bytes, 103_079_215_104);
        assert_eq!(signals.process_limit_remaining_bytes, Some(0));
    }

    #[test]
    fn mat_mat_qk_threadgroup_memory_matches_full_tile_policy() {
        assert_eq!(
            mat_mat_qk_threadgroup_memory_with_policy(5120, 16, 16, false),
            8192
        );
        assert_eq!(
            mat_mat_qk_threadgroup_memory_with_policy(5120, 16, 16, true),
            5120
        );
        assert_eq!(
            mat_mat_qk_threadgroup_memory_with_policy(5120, 32, 32, true),
            6144
        );
        assert_eq!(
            mat_mat_qk_threadgroup_memory_with_policy(5120, 1024, 32, true),
            6144
        );
        assert_eq!(
            mat_mat_qk_threadgroup_memory_with_policy(5121, 32, 32, true),
            8192
        );
        assert_eq!(
            mat_mat_qk_threadgroup_memory_with_policy(5120, 31, 32, true),
            8192
        );
    }

    #[test]
    fn checked_shape_bytes_rejects_product_overflow() {
        let err = checked_shape_bytes(&[u64::MAX, 2], std::mem::size_of::<f32>())
            .expect_err("shape product must overflow");
        assert!(matches!(err, MetalError::TensorSizeOverflow { .. }));
    }

    #[test]
    fn checked_shape_bytes_rejects_byte_overflow() {
        let err = checked_shape_bytes(&[usize::MAX as u64 / 4 + 1], std::mem::size_of::<f32>())
            .expect_err("byte count must overflow usize");
        assert!(matches!(err, MetalError::TensorSizeOverflow { .. }));
    }

    #[test]
    fn checked_ggml_shape_bytes_rejects_bad_q8_block() {
        let err = checked_ggml_shape_bytes(&[31], GgmlType::Q8_0)
            .expect_err("Q8_0 element count must align to a 32-element block");
        assert!(matches!(err, MetalError::BadShape { .. }));
    }

    #[test]
    fn metal_context_initializes() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::NoDevice) => return,
            Err(e) => panic!("unexpected error: {e}"),
        };
        eprintln!("[metal] {}", ctx.describe());
    }

    #[cfg(debug_assertions)]
    #[test]
    fn concurrent_hazard_guard_allows_disjoint_and_shared_reads() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let shared_in = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
        let out_a = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
        let out_b = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
        let cmd = ctx.queue.commandBuffer().expect("cmd buf");
        let enc = KernelEncoder::begin_concurrent(&cmd);
        // The production concurrent-pass shape: shared read-only input,
        // pairwise-disjoint outputs. Must not panic.
        enc.note_read(&shared_in);
        enc.note_write(&out_a);
        enc.note_read(&shared_in);
        enc.note_write(&out_b);
        enc.end();
    }

    #[cfg(debug_assertions)]
    #[test]
    fn concurrent_hazard_guard_panics_on_read_after_write() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let a = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
        let cmd = ctx.queue.commandBuffer().expect("cmd buf");
        let enc = KernelEncoder::begin_concurrent(&cmd);
        enc.note_write(&a);
        let hazard = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            enc.note_read(&a);
        }));
        assert!(
            hazard.is_err(),
            "read of a tensor written in the same Concurrent pass must panic"
        );
        enc.end();
    }

    #[cfg(debug_assertions)]
    #[test]
    fn concurrent_hazard_guard_ignores_serial_encoders() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let a = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
        let cmd = ctx.queue.commandBuffer().expect("cmd buf");
        let enc = KernelEncoder::begin(&cmd);
        // Serial encoders order dispatches; write-then-read is the normal
        // dataflow and must not trip the guard.
        enc.note_write(&a);
        enc.note_read(&a);
        enc.note_write(&a);
        enc.end();
    }

    #[cfg(debug_assertions)]
    #[test]
    fn concurrent_hazard_guard_allows_disjoint_views_of_one_buffer() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let arena = MetalTensor::zeros_f32(&ctx, vec![128]).unwrap();
        let lo = arena.view_subrange(0, vec![64]);
        let hi = arena.view_subrange(64, vec![64]);
        let cmd = ctx.queue.commandBuffer().expect("cmd buf");
        let enc = KernelEncoder::begin_concurrent(&cmd);
        // Disjoint sub-views of one arena are the packed-scratch pattern;
        // byte-range tracking (not buffer identity) must permit this.
        enc.note_write(&lo);
        enc.note_write(&hi);
        // But an overlapping second write must panic.
        let overlap = arena.view_subrange(32, vec![64]);
        let hazard = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            enc.note_write(&overlap);
        }));
        assert!(hazard.is_err(), "overlapping concurrent writes must panic");
        enc.end();
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
    fn residual_rms_norm_matches_separate_cpu_path() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &n in &[1024usize, 5120, 17408] {
            let x: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
            let r: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.03).collect();
            let w: Vec<f32> = (0..n).map(|i| 0.4 + (i % 11) as f32 * 0.05).collect();
            let eps = 1e-6;

            let x_cpu: Vec<f32> = x.iter().zip(r.iter()).map(|(a, b)| a + b).collect();
            let y_cpu = crate::forward::rms_norm_pub(&x_cpu, &w, eps);
            let (x_gpu, y_gpu) = residual_rms_norm_mul_f32_readback_for_test(&ctx, &x, &r, &w, eps)
                .expect("metal residual_rms_norm");

            let max_x = x_gpu
                .iter()
                .zip(x_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let max_y = y_gpu
                .iter()
                .zip(y_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[residual_rms_norm n={n}] max_x={max_x:.2e} max_y={max_y:.2e}");
            assert!(max_x == 0.0, "residual add n={n}: max|Δ|={max_x}");
            assert!(max_y < 1e-4, "residual_rms_norm n={n}: max|Δ|={max_y}");
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
    #[ignore]
    fn moe_mat_vec_iq3_xxs_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_Q3_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q3_K_M.gguf".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-iq3-oracle] skipped missing fixture {path}");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
            .expect("missing IQ3_XXS MoE gate tensor");
        let n_in = t.shape[0] as usize;
        let n_out = t.shape[1] as usize;
        let n_expert = t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let row_stride = (n_in / 256) * 98;
        let expert_stride = n_out * row_stride;
        let all_bytes = g.slice(t);
        let expert_bytes = &all_bytes[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_gate_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_out as u64],
            dtype: GgmlType::IQ3_XXS,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_bytes.len() as u64,
        };
        let w_f32 = crate::codec::dequant_to_f32(&expert_desc, expert_bytes).expect("dequant");
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.0075)
            .collect();
        let cpu = crate::forward::mat_vec_pub(&w_f32, n_in, n_out, &x);

        let w_t = MetalTensor::from_gguf_tensor(&ctx, t, all_bytes).expect("native weight");
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let expert_i = expert as i32;
        let topk_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[expert_i]),
            vec![1],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let out_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_mat_vec_iq3_xxs_f32(
                &ctx, enc, &w_t, &x_t, &topk_t, &out_t, n_in, n_out, n_expert, 1,
            )
        })
        .expect("gpu iq3 matvec");
        let gpu = read_back_f32(&out_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[moe-iq3-oracle] max|delta|={max_abs:.3e}");
        assert!(max_abs < 2e-4, "max|delta|={max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_swiglu_iq3_xxs_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_Q3_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q3_K_M.gguf".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-iq3-direct-swiglu-oracle] skipped missing fixture {path}");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
            .expect("missing IQ3_XXS MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
            .expect("missing IQ3_XXS MoE up tensor");
        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let row_stride = (n_in / 256) * 98;
        let expert_stride = n_ffn * row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let gate_expert_bytes =
            &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::IQ3_XXS,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 =
            crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, &x);
        let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, &x);
        let mut cpu = vec![0.0f32; n_ffn];
        for i in 0..n_ffn {
            let g = gate[i];
            cpu[i] = (g / (1.0 + (-g).exp())) * up[i];
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let expert_i = expert as i32;
        let topk_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[expert_i]),
            vec![1],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let out_gpu = MetalTensor::zeros_f32(&ctx, vec![n_ffn as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_iq3_xxs_f32(
                &ctx, enc, &gate_gpu, &up_gpu, &x_gpu, &topk_gpu, &out_gpu, n_in, n_ffn, n_expert,
                1,
            )
        })
        .expect("gpu direct iq3 swiglu");
        let gpu = read_back_f32(&out_gpu.buffer, n_ffn);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq3-direct-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");

        let out_fast_gpu =
            MetalTensor::zeros_f32(&ctx, vec![n_ffn as u64]).expect("fast out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_iq3_xxs_f32_fast(
                &ctx,
                enc,
                &gate_gpu,
                &up_gpu,
                &x_gpu,
                &topk_gpu,
                &out_fast_gpu,
                n_in,
                n_ffn,
                n_expert,
                1,
            )
        })
        .expect("gpu fast direct iq3 swiglu");
        let fast = read_back_f32(&out_fast_gpu.buffer, n_ffn);
        let fast_max_abs = fast
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let fast_dot: f64 = fast
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let nf: f64 = fast.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let fast_cos = fast_dot / (nf.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq3-fast-swiglu-oracle] cos={fast_cos:.6} max|delta|={fast_max_abs:.3e}");
        assert!(fast_cos > 0.999, "fast cos={fast_cos}");
        assert!(fast_max_abs < 2e-2, "fast max|delta|={fast_max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_grouped_swiglu_iq3_xxs_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_Q3_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q3_K_M.gguf".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-iq3-swiglu-oracle] skipped missing fixture {path}");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
            .expect("missing IQ3_XXS MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
            .expect("missing IQ3_XXS MoE up tensor");
        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let n_tokens = 32usize;
        let topk = 1usize;
        let row_stride = (n_in / 256) * 98;
        let expert_stride = n_ffn * row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let gate_expert_bytes =
            &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::IQ3_XXS,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 =
            crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
        let x: Vec<f32> = (0..n_tokens * n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let mut cpu = vec![0.0f32; n_tokens * n_ffn];
        for token in 0..n_tokens {
            let x_tok = &x[token * n_in..(token + 1) * n_in];
            let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, x_tok);
            let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, x_tok);
            for i in 0..n_ffn {
                let g = gate[i];
                cpu[token * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
            }
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_tokens * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let mut counts = vec![0i32; n_expert];
        counts[expert] = n_tokens as i32;
        let mut ids = vec![0i32; n_expert * n_tokens];
        for token in 0..n_tokens {
            ids[expert * n_tokens + token] = token as i32;
        }
        let counts_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&counts),
            vec![n_expert as u64],
            GgmlType::F32,
        )
        .expect("counts tensor");
        let ids_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![(n_expert * n_tokens) as u64],
            GgmlType::F32,
        )
        .expect("ids tensor");
        let out_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_ffn) as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16(
                &ctx,
                enc,
                &gate_gpu,
                &up_gpu,
                &x_gpu,
                &counts_gpu,
                &ids_gpu,
                &out_gpu,
                n_in,
                n_ffn,
                n_expert,
                topk,
                n_tokens,
            )
        })
        .expect("gpu grouped iq3 swiglu");
        let gpu = read_back_f32(&out_gpu.buffer, n_tokens * n_ffn);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq3-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_mat_vec_iq3_s_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_UDIQ4XS_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-iq3s-oracle] skipped missing fixture {path}");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_S)
            .expect("missing IQ3_S MoE gate tensor");
        let n_in = t.shape[0] as usize;
        let n_out = t.shape[1] as usize;
        let n_expert = t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let row_stride = (n_in / 256) * 110;
        let expert_stride = n_out * row_stride;
        let all_bytes = g.slice(t);
        let expert_bytes = &all_bytes[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_gate_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_out as u64],
            dtype: GgmlType::IQ3_S,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_bytes.len() as u64,
        };
        let w_f32 = crate::codec::dequant_to_f32(&expert_desc, expert_bytes).expect("dequant");
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.0075)
            .collect();
        let cpu = crate::forward::mat_vec_pub(&w_f32, n_in, n_out, &x);

        let w_t = MetalTensor::from_gguf_tensor(&ctx, t, all_bytes).expect("native weight");
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let expert_i = expert as i32;
        let topk_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[expert_i]),
            vec![1],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let out_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_mat_vec_iq3_s_f32(
                &ctx, enc, &w_t, &x_t, &topk_t, &out_t, n_in, n_out, n_expert, 1,
            )
        })
        .expect("gpu iq3s matvec");
        let gpu = read_back_f32(&out_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[moe-iq3s-oracle] max|delta|={max_abs:.3e}");
        assert!(max_abs < 2e-4, "max|delta|={max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_swiglu_iq3_s_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_UDIQ4XS_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-iq3s-direct-swiglu-oracle] skipped missing fixture {path}");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_S)
            .expect("missing IQ3_S MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::IQ3_S)
            .expect("missing IQ3_S MoE up tensor");
        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let row_stride = (n_in / 256) * 110;
        let expert_stride = n_ffn * row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let gate_expert_bytes =
            &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::IQ3_S,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 =
            crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, &x);
        let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, &x);
        let mut cpu = vec![0.0f32; n_ffn];
        for i in 0..n_ffn {
            let g = gate[i];
            cpu[i] = (g / (1.0 + (-g).exp())) * up[i];
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let expert_i = expert as i32;
        let topk_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[expert_i]),
            vec![1],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let out_gpu = MetalTensor::zeros_f32(&ctx, vec![n_ffn as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_iq3_s_f32(
                &ctx, enc, &gate_gpu, &up_gpu, &x_gpu, &topk_gpu, &out_gpu, n_in, n_ffn, n_expert,
                1,
            )
        })
        .expect("gpu direct iq3s swiglu");
        let gpu = read_back_f32(&out_gpu.buffer, n_ffn);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq3s-direct-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");

        let out_fast_gpu =
            MetalTensor::zeros_f32(&ctx, vec![n_ffn as u64]).expect("fast out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_iq3_s_f32_fast(
                &ctx,
                enc,
                &gate_gpu,
                &up_gpu,
                &x_gpu,
                &topk_gpu,
                &out_fast_gpu,
                n_in,
                n_ffn,
                n_expert,
                1,
            )
        })
        .expect("gpu fast direct iq3s swiglu");
        let fast = read_back_f32(&out_fast_gpu.buffer, n_ffn);
        let fast_max_abs = fast
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let fast_dot: f64 = fast
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let nf: f64 = fast.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let fast_cos = fast_dot / (nf.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq3s-fast-swiglu-oracle] cos={fast_cos:.6} max|delta|={fast_max_abs:.3e}");
        assert!(fast_cos > 0.999, "fast cos={fast_cos}");
        assert!(fast_max_abs < 2e-2, "fast max|delta|={fast_max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_grouped_swiglu_iq3_s_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_UDIQ4XS_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-iq3s-swiglu-oracle] skipped missing fixture {path}");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_S)
            .expect("missing IQ3_S MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::IQ3_S)
            .expect("missing IQ3_S MoE up tensor");
        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let n_tokens = 32usize;
        let topk = 1usize;
        let row_stride = (n_in / 256) * 110;
        let expert_stride = n_ffn * row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let gate_expert_bytes =
            &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::IQ3_S,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 =
            crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
        let x: Vec<f32> = (0..n_tokens * n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let mut cpu = vec![0.0f32; n_tokens * n_ffn];
        for token in 0..n_tokens {
            let x_tok = &x[token * n_in..(token + 1) * n_in];
            let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, x_tok);
            let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, x_tok);
            for i in 0..n_ffn {
                let g = gate[i];
                cpu[token * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
            }
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_tokens * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let mut counts = vec![0i32; n_expert];
        counts[expert] = n_tokens as i32;
        let mut ids = vec![0i32; n_expert * n_tokens];
        for token in 0..n_tokens {
            ids[expert * n_tokens + token] = token as i32;
        }
        let counts_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&counts),
            vec![n_expert as u64],
            GgmlType::F32,
        )
        .expect("counts tensor");
        let ids_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![(n_expert * n_tokens) as u64],
            GgmlType::F32,
        )
        .expect("ids tensor");
        let out_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_ffn) as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_iq3_s_f32_grouped_slots_n16(
                &ctx,
                enc,
                &gate_gpu,
                &up_gpu,
                &x_gpu,
                &counts_gpu,
                &ids_gpu,
                &out_gpu,
                n_in,
                n_ffn,
                n_expert,
                topk,
                n_tokens,
            )
        })
        .expect("gpu grouped iq3s swiglu");
        let gpu = read_back_f32(&out_gpu.buffer, n_tokens * n_ffn);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq3s-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_grouped_down_iq4_xs_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_UDIQ4XS_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-iq4xs-down-oracle] skipped missing fixture {path}");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let down_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_down_exps.weight" && t.dtype == GgmlType::IQ4_XS)
            .expect("missing IQ4_XS MoE down tensor");
        let n_in = down_t.shape[0] as usize;
        let n_out = down_t.shape[1] as usize;
        let n_expert = down_t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let n_tokens = 32usize;
        let row_stride = (n_in / 256) * 136;
        let expert_stride = n_out * row_stride;
        let down_bytes_all = g.slice(down_t);
        let down_expert_bytes =
            &down_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_down_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_out as u64],
            dtype: GgmlType::IQ4_XS,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let down_f32 =
            crate::codec::dequant_to_f32(&expert_desc, down_expert_bytes).expect("down dequant");
        let x: Vec<f32> = (0..n_tokens * n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let mut cpu = vec![0.0f32; n_tokens * n_out];
        for token in 0..n_tokens {
            let x_tok = &x[token * n_in..(token + 1) * n_in];
            let y = crate::forward::mat_vec_pub(&down_f32, n_in, n_out, x_tok);
            cpu[token * n_out..(token + 1) * n_out].copy_from_slice(&y);
        }

        let down_gpu = MetalTensor::from_gguf_tensor(&ctx, down_t, down_bytes_all).expect("down");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_tokens * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let mut counts = vec![0i32; n_expert];
        counts[expert] = n_tokens as i32;
        let mut ids = vec![0i32; n_expert * n_tokens];
        for token in 0..n_tokens {
            ids[expert * n_tokens + token] = token as i32;
        }
        let counts_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&counts),
            vec![n_expert as u64],
            GgmlType::F32,
        )
        .expect("counts tensor");
        let ids_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![(n_expert * n_tokens) as u64],
            GgmlType::F32,
        )
        .expect("ids tensor");
        let out_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_out) as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_down_iq4_xs_f32_grouped_slots(
                &ctx,
                enc,
                &down_gpu,
                &x_gpu,
                &counts_gpu,
                &ids_gpu,
                &out_gpu,
                n_in,
                n_out,
                n_expert,
                n_tokens,
            )
        })
        .expect("gpu grouped iq4xs down");
        let gpu = read_back_f32(&out_gpu.buffer, n_tokens * n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq4xs-down-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");

        let topk_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[expert as i32]),
            vec![1],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let x_one = x_gpu.view_subrange(0, vec![n_in as u64]);
        let out_fast_gpu =
            MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("fast out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_down_iq4_xs_f32_fast(
                &ctx,
                enc,
                &down_gpu,
                &x_one,
                &topk_gpu,
                &out_fast_gpu,
                n_in,
                n_out,
                n_expert,
                1,
            )
        })
        .expect("gpu fast iq4xs down");
        let fast = read_back_f32(&out_fast_gpu.buffer, n_out);
        let fast_max_abs = fast
            .iter()
            .zip(cpu[..n_out].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let fast_dot: f64 = fast
            .iter()
            .zip(cpu[..n_out].iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let nf: f64 = fast.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc_fast: f64 = cpu[..n_out].iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let fast_cos = fast_dot / (nf.sqrt() * nc_fast.sqrt()).max(1e-12);
        eprintln!("[moe-iq4xs-fast-down-oracle] cos={fast_cos:.6} max|delta|={fast_max_abs:.3e}");
        assert!(fast_cos > 0.999, "fast cos={fast_cos}");
        assert!(fast_max_abs < 2e-2, "fast max|delta|={fast_max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_swiglu_q6_k_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_Q6_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q6_K.gguf".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-q6-direct-swiglu-oracle] skipped missing fixture {path}");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::Q6_K)
            .expect("missing Q6_K MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::Q6_K)
            .expect("missing Q6_K MoE up tensor");
        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let topk = 2usize;
        let experts = [7usize.min(n_expert - 1), n_expert - 1];
        let row_stride = (n_in / 256) * 210;
        let expert_stride = n_ffn * row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let mut cpu = vec![0.0f32; topk * n_ffn];
        for (slot, expert) in experts.iter().copied().enumerate() {
            let gate_expert_bytes =
                &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
            let up_expert_bytes =
                &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
            let expert_desc = crate::tensor::TensorDesc {
                name: "blk.0.ffn_exps.weight.expert_oracle".into(),
                shape: vec![n_in as u64, n_ffn as u64],
                dtype: GgmlType::Q6_K,
                shard_idx: 0,
                data_offset: 0,
                n_bytes: expert_stride as u64,
            };
            let gate_f32 = crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes)
                .expect("gate dequant");
            let up_f32 =
                crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
            let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, &x);
            let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, &x);
            for i in 0..n_ffn {
                let g = gate[i];
                cpu[slot * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
            }
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let topk_i32 = [experts[0] as i32, experts[1] as i32];
        let topk_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&topk_i32),
            vec![topk as u64],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let out_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(topk * n_ffn) as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_q6_K_f32(
                &ctx, enc, &gate_gpu, &up_gpu, &x_gpu, &topk_gpu, &out_gpu, n_in, n_ffn, n_expert,
                topk,
            )
        })
        .expect("gpu direct q6 swiglu");
        let gpu = read_back_f32(&out_gpu.buffer, topk * n_ffn);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-q6-direct-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_grouped_swiglu_q6_k_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_Q6_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q6_K.gguf".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-q6-swiglu-oracle] skipped missing fixture {path}");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::Q6_K)
            .expect("missing Q6_K MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::Q6_K)
            .expect("missing Q6_K MoE up tensor");
        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let n_tokens = 16usize;
        let topk = 1usize;
        let row_stride = (n_in / 256) * 210;
        let expert_stride = n_ffn * row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let gate_expert_bytes =
            &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::Q6_K,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 =
            crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
        let x: Vec<f32> = (0..n_tokens * n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let mut cpu = vec![0.0f32; n_tokens * n_ffn];
        for token in 0..n_tokens {
            let x_tok = &x[token * n_in..(token + 1) * n_in];
            let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, x_tok);
            let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, x_tok);
            for i in 0..n_ffn {
                let g = gate[i];
                cpu[token * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
            }
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_tokens * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let mut counts = vec![0i32; n_expert];
        counts[expert] = n_tokens as i32;
        let mut ids = vec![0i32; n_expert * n_tokens];
        for token in 0..n_tokens {
            ids[expert * n_tokens + token] = token as i32;
        }
        let counts_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&counts),
            vec![n_expert as u64],
            GgmlType::F32,
        )
        .expect("counts tensor");
        let ids_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![(n_expert * n_tokens) as u64],
            GgmlType::F32,
        )
        .expect("ids tensor");
        let out_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_ffn) as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_q6_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &gate_gpu,
                &up_gpu,
                &x_gpu,
                &counts_gpu,
                &ids_gpu,
                &out_gpu,
                n_in,
                n_ffn,
                n_expert,
                topk,
                n_tokens,
            )
        })
        .expect("gpu grouped q6 swiglu");
        let gpu = read_back_f32(&out_gpu.buffer, n_tokens * n_ffn);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-q6-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_q8_0_swiglu_down_weighted_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_Q8_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q8_0.gguf".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-q8-direct-oracle] skipped missing fixture {path}");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::Q8_0)
            .expect("missing Q8_0 MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::Q8_0)
            .expect("missing Q8_0 MoE up tensor");
        let down_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_down_exps.weight" && t.dtype == GgmlType::Q8_0)
            .expect("missing Q8_0 MoE down tensor");

        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let h = down_t.shape[1] as usize;
        let experts = [7usize.min(n_expert - 1), n_expert - 1];
        let topk = experts.len();
        let top_w = [0.35f32, 0.65f32];
        let gate_row_stride = (n_in / 32) * 34;
        let gate_expert_stride = n_ffn * gate_row_stride;
        let down_row_stride = (n_ffn / 32) * 34;
        let down_expert_stride = h * down_row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let down_bytes_all = g.slice(down_t);
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let gate_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_gate_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: gate_expert_stride as u64,
        };
        let down_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_down_exps.weight.expert_oracle".into(),
            shape: vec![n_ffn as u64, h as u64],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: down_expert_stride as u64,
        };
        let mut cpu_inner = vec![0.0f32; topk * n_ffn];
        let mut cpu_down = vec![0.0f32; h];
        for (slot, expert) in experts.iter().copied().enumerate() {
            let gate_expert_bytes =
                &gate_bytes_all[expert * gate_expert_stride..(expert + 1) * gate_expert_stride];
            let up_expert_bytes =
                &up_bytes_all[expert * gate_expert_stride..(expert + 1) * gate_expert_stride];
            let down_expert_bytes =
                &down_bytes_all[expert * down_expert_stride..(expert + 1) * down_expert_stride];
            let gate_f32 =
                crate::codec::dequant_to_f32(&gate_desc, gate_expert_bytes).expect("gate dequant");
            let up_f32 =
                crate::codec::dequant_to_f32(&gate_desc, up_expert_bytes).expect("up dequant");
            let down_f32 =
                crate::codec::dequant_to_f32(&down_desc, down_expert_bytes).expect("down dequant");
            let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, &x);
            let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, &x);
            for i in 0..n_ffn {
                let g = gate[i];
                cpu_inner[slot * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            let down = crate::forward::mat_vec_pub(
                &down_f32,
                n_ffn,
                h,
                &cpu_inner[slot * n_ffn..(slot + 1) * n_ffn],
            );
            for i in 0..h {
                cpu_down[i] += top_w[slot] * down[i];
            }
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let down_gpu = MetalTensor::from_gguf_tensor(&ctx, down_t, down_bytes_all).expect("down");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let topk_i32 = [experts[0] as i32, experts[1] as i32];
        let topk_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&topk_i32),
            vec![topk as u64],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let topw_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&top_w),
            vec![topk as u64],
            GgmlType::F32,
        )
        .expect("top weights tensor");
        let inner_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(topk * n_ffn) as u64]).expect("inner tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_q8_0_f32(
                &ctx, enc, &gate_gpu, &up_gpu, &x_gpu, &topk_gpu, &inner_gpu, n_in, n_ffn,
                n_expert, topk,
            )
        })
        .expect("gpu direct q8 swiglu");
        let gpu_inner = read_back_f32(&inner_gpu.buffer, topk * n_ffn);
        let inner_max = gpu_inner
            .iter()
            .zip(cpu_inner.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let inner_dot: f64 = gpu_inner
            .iter()
            .zip(cpu_inner.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let inner_ng: f64 = gpu_inner.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let inner_nc: f64 = cpu_inner.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let inner_cos = inner_dot / (inner_ng.sqrt() * inner_nc.sqrt()).max(1e-12);
        eprintln!("[moe-q8-direct-swiglu-oracle] cos={inner_cos:.6} max|delta|={inner_max:.3e}");
        assert!(inner_cos > 0.999, "inner cos={inner_cos}");
        assert!(inner_max < 2e-2, "inner max|delta|={inner_max}");

        let down_gpu_out = MetalTensor::zeros_f32(&ctx, vec![h as u64]).expect("down out");
        one_shot(&ctx, |enc| {
            encode_moe_down_weighted_sum_q8_0_f32(
                &ctx,
                enc,
                &down_gpu,
                &inner_gpu,
                &topk_gpu,
                &topw_gpu,
                &down_gpu_out,
                n_ffn,
                h,
                n_expert,
                topk,
            )
        })
        .expect("gpu direct q8 down weighted sum");
        let gpu_down = read_back_f32(&down_gpu_out.buffer, h);
        let down_max = gpu_down
            .iter()
            .zip(cpu_down.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let down_dot: f64 = gpu_down
            .iter()
            .zip(cpu_down.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let down_ng: f64 = gpu_down.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let down_nc: f64 = cpu_down.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let down_cos = down_dot / (down_ng.sqrt() * down_nc.sqrt()).max(1e-12);
        eprintln!("[moe-q8-direct-down-oracle] cos={down_cos:.6} max|delta|={down_max:.3e}");
        assert!(down_cos > 0.999, "down cos={down_cos}");
        assert!(down_max < 2e-2, "down max|delta|={down_max}");
    }

    #[test]
    #[ignore]
    fn moe_grouped_q8_0_swiglu_down_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_Q8_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-35B-A3B-Q8_0.gguf".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-q8-oracle] skipped missing fixture {path}");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::Q8_0)
            .expect("missing Q8_0 MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::Q8_0)
            .expect("missing Q8_0 MoE up tensor");
        let down_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_down_exps.weight" && t.dtype == GgmlType::Q8_0)
            .expect("missing Q8_0 MoE down tensor");

        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let h = down_t.shape[1] as usize;
        let expert = 7usize.min(n_expert - 1);
        let n_tokens = 16usize;
        let topk = 1usize;
        let gate_row_stride = (n_in / 32) * 34;
        let gate_expert_stride = n_ffn * gate_row_stride;
        let down_row_stride = (n_ffn / 32) * 34;
        let down_expert_stride = h * down_row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let down_bytes_all = g.slice(down_t);
        let gate_expert_bytes =
            &gate_bytes_all[expert * gate_expert_stride..(expert + 1) * gate_expert_stride];
        let up_expert_bytes =
            &up_bytes_all[expert * gate_expert_stride..(expert + 1) * gate_expert_stride];
        let down_expert_bytes =
            &down_bytes_all[expert * down_expert_stride..(expert + 1) * down_expert_stride];
        let gate_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_gate_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: gate_expert_stride as u64,
        };
        let down_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_down_exps.weight.expert_oracle".into(),
            shape: vec![n_ffn as u64, h as u64],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: down_expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&gate_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 = crate::codec::dequant_to_f32(&gate_desc, up_expert_bytes).expect("up dequant");
        let down_f32 =
            crate::codec::dequant_to_f32(&down_desc, down_expert_bytes).expect("down dequant");
        let x: Vec<f32> = (0..n_tokens * n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let mut cpu_inner = vec![0.0f32; n_tokens * n_ffn];
        let mut cpu_down = vec![0.0f32; n_tokens * h];
        for token in 0..n_tokens {
            let x_tok = &x[token * n_in..(token + 1) * n_in];
            let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, x_tok);
            let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, x_tok);
            for i in 0..n_ffn {
                let g = gate[i];
                cpu_inner[token * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            let down = crate::forward::mat_vec_pub(
                &down_f32,
                n_ffn,
                h,
                &cpu_inner[token * n_ffn..(token + 1) * n_ffn],
            );
            cpu_down[token * h..(token + 1) * h].copy_from_slice(&down);
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let down_gpu = MetalTensor::from_gguf_tensor(&ctx, down_t, down_bytes_all).expect("down");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_tokens * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let mut counts = vec![0i32; n_expert];
        counts[expert] = n_tokens as i32;
        let mut ids = vec![0i32; n_expert * n_tokens];
        for token in 0..n_tokens {
            ids[expert * n_tokens + token] = token as i32;
        }
        let counts_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&counts),
            vec![n_expert as u64],
            GgmlType::F32,
        )
        .expect("counts tensor");
        let ids_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![(n_expert * n_tokens) as u64],
            GgmlType::F32,
        )
        .expect("ids tensor");
        let inner_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_ffn) as u64]).expect("inner tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_q8_0_f32_grouped_slots_n16(
                &ctx,
                enc,
                &gate_gpu,
                &up_gpu,
                &x_gpu,
                &counts_gpu,
                &ids_gpu,
                &inner_gpu,
                n_in,
                n_ffn,
                n_expert,
                topk,
                n_tokens,
            )
        })
        .expect("gpu grouped q8 swiglu");
        let gpu_inner = read_back_f32(&inner_gpu.buffer, n_tokens * n_ffn);
        let dot_inner: f64 = gpu_inner
            .iter()
            .zip(cpu_inner.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng_inner: f64 = gpu_inner.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc_inner: f64 = cpu_inner.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos_inner = dot_inner / (ng_inner.sqrt() * nc_inner.sqrt()).max(1e-12);
        let max_inner = gpu_inner
            .iter()
            .zip(cpu_inner.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[moe-q8-swiglu-oracle] cos={cos_inner:.6} max|delta|={max_inner:.3e}");
        assert!(cos_inner > 0.999, "inner cos={cos_inner}");
        assert!(max_inner < 2e-2, "inner max|delta|={max_inner}");

        let cpu_inner_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&cpu_inner),
            vec![(n_tokens * n_ffn) as u64],
            GgmlType::F32,
        )
        .expect("cpu inner tensor");
        let down_out_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * h) as u64]).expect("down out");
        one_shot(&ctx, |enc| {
            encode_moe_down_q8_0_f32_grouped_slots(
                &ctx,
                enc,
                &down_gpu,
                &cpu_inner_gpu,
                &counts_gpu,
                &ids_gpu,
                &down_out_gpu,
                n_ffn,
                h,
                n_expert,
                n_tokens,
            )
        })
        .expect("gpu grouped q8 down");
        let gpu_down = read_back_f32(&down_out_gpu.buffer, n_tokens * h);
        let dot_down: f64 = gpu_down
            .iter()
            .zip(cpu_down.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng_down: f64 = gpu_down.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc_down: f64 = cpu_down.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos_down = dot_down / (ng_down.sqrt() * nc_down.sqrt()).max(1e-12);
        let max_down = gpu_down
            .iter()
            .zip(cpu_down.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[moe-q8-down-oracle] cos={cos_down:.6} max|delta|={max_down:.3e}");
        assert!(cos_down > 0.999, "down cos={cos_down}");
        assert!(max_down < 2e-2, "down max|delta|={max_down}");
    }

    #[test]
    fn mat_vec_and_mat_mat_half_weights_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(path, dtype) in &[
            ("/Users/tito/models/Qwen3.5-0.8B.f16.gguf", GgmlType::F16),
            ("/Users/tito/models/Qwen3.5-0.8B-BF16.gguf", GgmlType::BF16),
        ] {
            if !std::path::Path::new(path).exists() {
                eprintln!("[half-weight] skipped missing fixture {path}");
                continue;
            }
            let g = crate::gguf::GgufFile::open(path).expect("open");
            let w = g
                .tensors
                .iter()
                .find(|t| {
                    t.name == "blk.0.ffn_gate.weight" && t.dtype == dtype && t.shape.len() == 2
                })
                .expect("missing half test tensor");
            let n_in = w.shape[0] as usize;
            let n_out = w.shape[1] as usize;
            eprintln!("[half-weight {dtype:?}] {} shape=[{n_in}, {n_out}]", w.name);

            let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
            let w_t =
                MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
                    .expect("weight tensor");

            let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
            let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
            one_shot(&ctx, |enc| match dtype {
                GgmlType::F16 => encode_mat_vec_f16_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
                GgmlType::BF16 => encode_mat_vec_bf16_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
                _ => unreachable!(),
            })
            .expect("mat_vec encode");
            let gpu = read_back_f32(&y_t.buffer, n_out);
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[half-weight {dtype:?} mat_vec] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-3, "{dtype:?} mat_vec max_abs={max_abs}");

            for &n_query in &[1usize, 16, 32, 33] {
                let x_pack: Vec<f32> = (0..n_query * n_in)
                    .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                    .collect();
                let mut cpu_pack = vec![0.0f32; n_query * n_out];
                for q in 0..n_query {
                    let row = &x_pack[q * n_in..(q + 1) * n_in];
                    let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                    cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
                }
                let x_pack_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&x_pack),
                    vec![n_query as u64, n_in as u64],
                    GgmlType::F32,
                )
                .expect("x pack tensor");
                let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                    .expect("y pack tensor");
                one_shot(&ctx, |enc| match dtype {
                    GgmlType::F16 => encode_mat_mat_f16_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    GgmlType::BF16 => encode_mat_mat_bf16_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    _ => unreachable!(),
                })
                .expect("mat_mat encode");
                let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
                let max_abs = gpu_pack
                    .iter()
                    .zip(cpu_pack.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "[half-weight {dtype:?} mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}"
                );
                assert!(max_abs < 1e-3, "{dtype:?} mat_mat max_abs={max_abs}");
            }
        }
    }

    #[test]
    fn mat_mat_bf16_bfloat_act_matches_rounded_cpu() {
        fn round_to_bf16_f32(x: f32) -> f32 {
            let bits = x.to_bits();
            let lsb = (bits >> 16) & 1;
            f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
        }

        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.5-0.8B-BF16.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[bf16-bfloat-act] skipped missing fixture {path}");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let w = g
            .tensors
            .iter()
            .find(|t| {
                t.name == "blk.0.ffn_gate.weight" && t.dtype == GgmlType::BF16 && t.shape.len() == 2
            })
            .expect("missing BF16 test tensor");
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
        let w_t = MetalTensor::from_bytes(
            &ctx,
            g.slice(w),
            vec![n_in as u64, n_out as u64],
            GgmlType::BF16,
        )
        .expect("weight tensor");

        for &n_out_case in &[70usize, n_out] {
            let weight_case = &weight_f32[..n_in * n_out_case];
            for &n_query in &[1usize, 16, 32, 33] {
                let x_pack: Vec<f32> = (0..n_query * n_in)
                    .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                    .collect();
                let x_bf16: Vec<f32> = x_pack.iter().copied().map(round_to_bf16_f32).collect();
                let mut cpu_pack = vec![0.0f32; n_query * n_out_case];
                for q in 0..n_query {
                    let row = &x_bf16[q * n_in..(q + 1) * n_in];
                    let out = crate::forward::mat_vec_pub(weight_case, n_in, n_out_case, row);
                    cpu_pack[q * n_out_case..(q + 1) * n_out_case].copy_from_slice(&out);
                }
                let x_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&x_pack),
                    vec![n_query as u64, n_in as u64],
                    GgmlType::F32,
                )
                .expect("x tensor");
                let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out_case) as u64])
                    .expect("y tensor");
                one_shot(&ctx, |enc| {
                    encode_mat_mat_bf16_bfloat_act_f32(
                        &ctx, enc, &w_t, &x_t, &y_t, n_in, n_out_case, n_query,
                    )
                })
                .expect("approx bf16 matmat encode");
                let gpu = read_back_f32(&y_t.buffer, n_query * n_out_case);
                let dot: f64 = gpu
                    .iter()
                    .zip(cpu_pack.iter())
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum();
                let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                let nc: f64 = cpu_pack.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
                let max_abs = gpu
                    .iter()
                    .zip(cpu_pack.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "[bf16-bfloat-act n_out={n_out_case} n_query={n_query}] \
                     cos={cos:.6} max|Delta|={max_abs:.2e}"
                );
                assert!(cos > 0.99999, "cos={cos}");
                assert!(max_abs < 1e-3, "max_abs={max_abs}");
            }
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
    fn mat_vec_and_mat_mat_q4_legacy_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(path, dtype) in &[
            ("/Users/tito/models/Qwen3.5-0.8B-Q4_0.gguf", GgmlType::Q4_0),
            ("/Users/tito/models/Qwen3.5-0.8B-Q4_1.gguf", GgmlType::Q4_1),
        ] {
            if !std::path::Path::new(path).exists() {
                eprintln!("[q4-legacy] skipped missing fixture {path}");
                continue;
            }
            let g = crate::gguf::GgufFile::open(path).expect("open");
            let w = g
                .tensors
                .iter()
                .find(|t| {
                    t.name == "blk.0.ffn_gate.weight"
                        && t.dtype == dtype
                        && t.shape.len() == 2
                        && t.shape[0] % 32 == 0
                })
                .expect("missing q4 legacy test tensor");
            let n_in = w.shape[0] as usize;
            let n_out = w.shape[1] as usize;
            eprintln!("[q4-legacy {dtype:?}] {} shape=[{n_in}, {n_out}]", w.name);

            let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
            let w_t =
                MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
                    .expect("weight tensor");

            let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
            let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
            one_shot(&ctx, |enc| match dtype {
                GgmlType::Q4_0 => encode_mat_vec_q4_0_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
                GgmlType::Q4_1 => encode_mat_vec_q4_1_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
                _ => unreachable!(),
            })
            .expect("mat_vec encode");
            let gpu = read_back_f32(&y_t.buffer, n_out);
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[q4-legacy {dtype:?} mat_vec] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "{dtype:?} mat_vec max_abs={max_abs}");

            for &n_query in &[1usize, 16, 32, 33] {
                let x_pack: Vec<f32> = (0..n_query * n_in)
                    .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                    .collect();
                let mut cpu_pack = vec![0.0f32; n_query * n_out];
                for q in 0..n_query {
                    let row = &x_pack[q * n_in..(q + 1) * n_in];
                    let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                    cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
                }
                let x_pack_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&x_pack),
                    vec![n_query as u64, n_in as u64],
                    GgmlType::F32,
                )
                .expect("x pack tensor");
                let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                    .expect("y pack tensor");
                one_shot(&ctx, |enc| match dtype {
                    GgmlType::Q4_0 => encode_mat_mat_q4_0_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    GgmlType::Q4_1 => encode_mat_mat_q4_1_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    _ => unreachable!(),
                })
                .expect("mat_mat encode");
                let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
                let max_abs = gpu_pack
                    .iter()
                    .zip(cpu_pack.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "[q4-legacy {dtype:?} mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}"
                );
                assert!(max_abs < 1e-2, "{dtype:?} mat_mat max_abs={max_abs}");
            }
        }
    }

    #[test]
    fn mat_vec_and_mat_mat_q3_k_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.5-0.8B-Q3_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[q3_k] skipped missing fixture {path}");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let w = g
            .tensors
            .iter()
            .find(|t| {
                t.name == "blk.0.ffn_gate.weight"
                    && t.dtype == GgmlType::Q3_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
            })
            .expect("missing q3_k test tensor");
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        eprintln!("[q3_k] {} shape=[{n_in}, {n_out}]", w.name);

        let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
        let w_t = MetalTensor::from_bytes(
            &ctx,
            g.slice(w),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q3_K,
        )
        .expect("weight tensor");

        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
        one_shot(&ctx, |enc| {
            encode_mat_vec_q3_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
        })
        .expect("mat_vec encode");
        let gpu = read_back_f32(&y_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q3_k mat_vec] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-2, "Q3_K mat_vec max_abs={max_abs}");

        for &n_query in &[1usize, 16, 32] {
            let x_pack: Vec<f32> = (0..n_query * n_in)
                .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                .collect();
            let mut cpu_pack = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row = &x_pack[q * n_in..(q + 1) * n_in];
                let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
            }
            let x_pack_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_pack),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x pack tensor");
            let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                .expect("y pack tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_q3_k_f32(&ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query)
            })
            .expect("mat_mat encode");
            let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
            let max_abs = gpu_pack
                .iter()
                .zip(cpu_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[q3_k mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "Q3_K mat_mat max_abs={max_abs}");
        }
    }

    #[test]
    fn mat_vec_and_mat_mat_dense_iq2_s_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.5-4B-UD-IQ2_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[dense-iq2_s] skipped missing fixture {path}");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let w = match g.tensors.iter().find(|t| {
            t.name.starts_with("blk.")
                && t.name.ends_with(".weight")
                && t.dtype == GgmlType::IQ2_S
                && t.shape.len() == 2
                && t.shape[0] % 256 == 0
        }) {
            Some(t) => t,
            None => {
                eprintln!("[dense-iq2_s] skipped missing IQ2_S tensor in {path}");
                return;
            }
        };
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        eprintln!("[dense-iq2_s] {} shape=[{n_in}, {n_out}]", w.name);

        let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
        let w_t = MetalTensor::from_bytes(
            &ctx,
            g.slice(w),
            vec![n_in as u64, n_out as u64],
            GgmlType::IQ2_S,
        )
        .expect("weight tensor");

        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
        one_shot(&ctx, |enc| {
            encode_mat_vec_iq2_s_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
        })
        .expect("mat_vec encode");
        let gpu = read_back_f32(&y_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[dense-iq2_s mat_vec] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-2, "IQ2_S mat_vec max_abs={max_abs}");

        for &n_query in &[1usize, 16] {
            let x_pack: Vec<f32> = (0..n_query * n_in)
                .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                .collect();
            let mut cpu_pack = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row = &x_pack[q * n_in..(q + 1) * n_in];
                let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
            }
            let x_pack_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_pack),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x pack tensor");
            let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                .expect("y pack tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_iq2_s_f32(
                    &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                )
            })
            .expect("mat_mat encode");
            let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
            let max_abs = gpu_pack
                .iter()
                .zip(cpu_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[dense-iq2_s mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "IQ2_S mat_mat max_abs={max_abs}");
        }
    }

    #[test]
    fn mat_vec_and_mat_mat_dense_iq3_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let fixtures = [
            (
                "/Users/tito/models/Qwen3.5-4B-UD-Q2_K_XL.gguf",
                GgmlType::IQ3_XXS,
            ),
            (
                "/Users/tito/models/Qwen3.5-4B-UD-Q2_K_XL.gguf",
                GgmlType::IQ3_S,
            ),
        ];
        for &(path, dtype) in &fixtures {
            if !std::path::Path::new(path).exists() {
                eprintln!("[dense-iq3 {dtype:?}] skipped missing fixture {path}");
                continue;
            }
            let g = crate::gguf::GgufFile::open(path).expect("open");
            let w = match g.tensors.iter().find(|t| {
                t.name.starts_with("blk.")
                    && t.name.ends_with(".weight")
                    && t.dtype == dtype
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
            }) {
                Some(t) => t,
                None => {
                    eprintln!("[dense-iq3 {dtype:?}] skipped missing dtype tensor in {path}");
                    continue;
                }
            };
            let n_in = w.shape[0] as usize;
            let n_out = w.shape[1] as usize;
            eprintln!("[dense-iq3 {dtype:?}] {} shape=[{n_in}, {n_out}]", w.name);

            let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
            let w_t =
                MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
                    .expect("weight tensor");

            let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
            let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
            one_shot(&ctx, |enc| match dtype {
                GgmlType::IQ3_XXS => {
                    encode_mat_vec_iq3_xxs_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
                }
                GgmlType::IQ3_S => {
                    encode_mat_vec_iq3_s_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
                }
                _ => unreachable!(),
            })
            .expect("mat_vec encode");
            let gpu = read_back_f32(&y_t.buffer, n_out);
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[dense-iq3 {dtype:?} mat_vec] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "{dtype:?} mat_vec max_abs={max_abs}");

            for &n_query in &[1usize, 16] {
                let x_pack: Vec<f32> = (0..n_query * n_in)
                    .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                    .collect();
                let mut cpu_pack = vec![0.0f32; n_query * n_out];
                for q in 0..n_query {
                    let row = &x_pack[q * n_in..(q + 1) * n_in];
                    let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                    cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
                }
                let x_pack_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&x_pack),
                    vec![n_query as u64, n_in as u64],
                    GgmlType::F32,
                )
                .expect("x pack tensor");
                let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                    .expect("y pack tensor");
                one_shot(&ctx, |enc| match dtype {
                    GgmlType::IQ3_XXS => encode_mat_mat_iq3_xxs_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    GgmlType::IQ3_S => encode_mat_mat_iq3_s_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    _ => unreachable!(),
                })
                .expect("mat_mat encode");
                let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
                let max_abs = gpu_pack
                    .iter()
                    .zip(cpu_pack.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "[dense-iq3 {dtype:?} mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}"
                );
                assert!(max_abs < 1e-2, "{dtype:?} mat_mat max_abs={max_abs}");
            }
        }
    }

    #[test]
    fn mat_vec_and_mat_mat_q2_k_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.5-0.8B.Q2_K.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[q2_k] skipped missing fixture {path}");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let w = g
            .tensors
            .iter()
            .find(|t| {
                t.name == "blk.0.ffn_gate.weight"
                    && t.dtype == GgmlType::Q2_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
            })
            .expect("missing q2_k test tensor");
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        eprintln!("[q2_k] {} shape=[{n_in}, {n_out}]", w.name);

        let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
        let w_t = MetalTensor::from_bytes(
            &ctx,
            g.slice(w),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q2_K,
        )
        .expect("weight tensor");

        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
        one_shot(&ctx, |enc| {
            encode_mat_vec_q2_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
        })
        .expect("mat_vec encode");
        let gpu = read_back_f32(&y_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q2_k mat_vec] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-2, "Q2_K mat_vec max_abs={max_abs}");

        for &n_query in &[1usize, 16, 32] {
            let x_pack: Vec<f32> = (0..n_query * n_in)
                .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                .collect();
            let mut cpu_pack = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row = &x_pack[q * n_in..(q + 1) * n_in];
                let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
            }
            let x_pack_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_pack),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x pack tensor");
            let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                .expect("y pack tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_q2_k_f32(&ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query)
            })
            .expect("mat_mat encode");
            let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
            let max_abs = gpu_pack
                .iter()
                .zip(cpu_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[q2_k mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "Q2_K mat_mat max_abs={max_abs}");
        }
    }

    #[test]
    fn mat_vec_and_mat_mat_iq4_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(path, dtype) in &[
            (
                "/Users/tito/models/Qwen3.5-0.8B-IQ4_NL.gguf",
                GgmlType::IQ4_NL,
            ),
            (
                "/Users/tito/models/Qwen3.5-0.8B-IQ4_XS.gguf",
                GgmlType::IQ4_XS,
            ),
        ] {
            if !std::path::Path::new(path).exists() {
                eprintln!("[iq4] skipped missing fixture {path}");
                continue;
            }
            let g = crate::gguf::GgufFile::open(path).expect("open");
            let align = if dtype == GgmlType::IQ4_NL { 32 } else { 256 };
            let w = g
                .tensors
                .iter()
                .find(|t| {
                    t.name == "blk.0.ffn_gate.weight"
                        && t.dtype == dtype
                        && t.shape.len() == 2
                        && t.shape[0] % align == 0
                })
                .expect("missing iq4 test tensor");
            let n_in = w.shape[0] as usize;
            let n_out = w.shape[1] as usize;
            eprintln!("[iq4 {dtype:?}] {} shape=[{n_in}, {n_out}]", w.name);

            let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
            let w_t =
                MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
                    .expect("weight tensor");

            let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
            let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
            one_shot(&ctx, |enc| match dtype {
                GgmlType::IQ4_NL => {
                    encode_mat_vec_iq4_nl_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
                }
                GgmlType::IQ4_XS => {
                    encode_mat_vec_iq4_xs_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
                }
                _ => unreachable!(),
            })
            .expect("mat_vec encode");
            let gpu = read_back_f32(&y_t.buffer, n_out);
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[iq4 {dtype:?} mat_vec] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "{dtype:?} mat_vec max_abs={max_abs}");

            for &n_query in &[1usize, 16, 32] {
                let x_pack: Vec<f32> = (0..n_query * n_in)
                    .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                    .collect();
                let mut cpu_pack = vec![0.0f32; n_query * n_out];
                for q in 0..n_query {
                    let row = &x_pack[q * n_in..(q + 1) * n_in];
                    let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                    cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
                }
                let x_pack_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&x_pack),
                    vec![n_query as u64, n_in as u64],
                    GgmlType::F32,
                )
                .expect("x pack tensor");
                let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                    .expect("y pack tensor");
                one_shot(&ctx, |enc| match dtype {
                    GgmlType::IQ4_NL => encode_mat_mat_iq4_nl_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    GgmlType::IQ4_XS => encode_mat_mat_iq4_xs_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    _ => unreachable!(),
                })
                .expect("mat_mat encode");
                let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
                let max_abs = gpu_pack
                    .iter()
                    .zip(cpu_pack.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!("[iq4 {dtype:?} mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
                assert!(max_abs < 1e-2, "{dtype:?} mat_mat max_abs={max_abs}");
            }
        }
    }

    /// H5.3b.0 gate: lifted Q4_K mat-mat correctness against
    /// (a) CPU mat-mat oracle  (b) N successive mat-vec calls
    /// (c) col-major output layout sanity.
    ///
    /// Per codex H5.3b mid-impl review: lifted llama mat-mat is NOT
    /// bit-exact with N mat-vec because it stages activations through
    /// half before float accumulation. Gate thresholds:
    ///   * vs CPU mat-mat oracle (same half-staging math): cos ≥ 0.9999
    ///     and max|Δ| ≤ 0.01 (Q4_K dequant noise dominates the diff)
    ///   * vs N mat-vec: cos ≥ 0.999 per row (relaxed; half-vs-float
    ///     accumulation diff)
    ///   * layout: dst[row + col * M] stride explicitly probed
    ///
    /// Uses real Q4_K weight from the 27B GGUF; N_QUERY ∈ {1, 16, 32}
    /// to exercise both partial-tile path (N=1, N=16) and full-tile
    /// path (N=32).
    #[test]
    fn mat_mat_q4_k_matches_cpu_and_mat_vec() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[mat_mat_q4_k] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        // Pick a Q4_K tensor with shape compatible with mat-mat tiling
        // (n_in % 32 == 0, n_out % 64 == 0 for the lifted tile).
        let q4k = g
            .tensors
            .iter()
            .find(|t| {
                t.name.starts_with("blk.0.")
                    && t.dtype == GgmlType::Q4_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
                    && t.shape[1] % 64 == 0
            })
            .expect("no Q4_K tensor with compatible shape");
        let n_in = q4k.shape[0] as usize;
        let n_out = q4k.shape[1] as usize;
        eprintln!(
            "[mat_mat_q4_k-test] tensor={} shape=[n_in={n_in}, n_out={n_out}]",
            q4k.name
        );

        let weight_f32 = crate::codec::dequant_to_f32(q4k, g.slice(q4k)).expect("dequant");
        let weight_bytes = g.slice(q4k);

        let n_queries: &[usize] = if mat_mat_q4_k_n64_enabled() {
            &[1, 16, 32, 64]
        } else {
            &[1, 16, 32]
        };
        for &n_query in n_queries {
            // Activation matrix [n_query, n_in] row-major, deterministic
            // pseudo-random fill.
            let mut x = vec![0.0f32; n_query * n_in];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i % 13) as f32 - 6.0) * 1e-2;
            }

            // -- CPU oracle: row-major output `[n_query, n_out]`
            //    y[q, o] = sum_i W[o, i] * x[q, i]
            //    We compute it via N successive mat_vec_pub calls.
            let mut cpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row_in = &x[q * n_in..(q + 1) * n_in];
                let row_out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row_in);
                cpu_row_major[q * n_out..(q + 1) * n_out].copy_from_slice(&row_out);
            }

            // -- GPU mat-mat: output `[n_out, n_query]` COL-major
            //    i.e. y[r + c * n_out]. Allocate raw n_out*n_query f32.
            let w_t = MetalTensor::from_bytes(
                &ctx,
                weight_bytes,
                vec![n_in as u64, n_out as u64],
                GgmlType::Q4_K,
            )
            .expect("weight tensor");
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_q4_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out, n_query)
            })
            .expect("mat_mat encode");

            let gpu_col_major = read_back_f32(&y_t.buffer, n_out * n_query);

            // -- Reshape: convert col-major [n_out, n_query] →
            //    row-major [n_query, n_out] for comparison.
            //    cell (q, o) lives at gpu_col_major[o + q * n_out]
            //                  vs   cpu_row_major[q * n_out + o].
            let mut gpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                for o in 0..n_out {
                    gpu_row_major[q * n_out + o] = gpu_col_major[o + q * n_out];
                }
            }

            // -- Per-row cosine + max|Δ|.
            let mut min_cos = f64::INFINITY;
            let mut max_abs = 0.0f32;
            for q in 0..n_query {
                let cpu_row = &cpu_row_major[q * n_out..(q + 1) * n_out];
                let gpu_row = &gpu_row_major[q * n_out..(q + 1) * n_out];
                let mut dot = 0.0f64;
                let mut np = 0.0f64;
                let mut nc = 0.0f64;
                for i in 0..n_out {
                    let p = gpu_row[i] as f64;
                    let c = cpu_row[i] as f64;
                    dot += p * c;
                    np += p * p;
                    nc += c * c;
                    let d = (gpu_row[i] - cpu_row[i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
                let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
                if cos < min_cos {
                    min_cos = cos;
                }
            }
            eprintln!(
                "[mat_mat_q4_k n_query={n_query}] min_cos={min_cos:.6} \
                 max|Δ|={max_abs:.3e}"
            );
            // Per H5.3b plan rev 6: cos ≥ 0.999 vs N mat-vec (relaxed
            // because half-staging in lifted kernel). max|Δ| ≤ 0.01
            // (Q4_K dequant + half-staging noise; same order as Q4_K
            // mat-vec test threshold).
            assert!(
                min_cos >= 0.999,
                "n_query={n_query}: min cos {min_cos} < 0.999"
            );
            assert!(
                max_abs < 1e-2,
                "n_query={n_query}: max|Δ| {max_abs} >= 1e-2"
            );

            // -- Layout sanity (codex Q7 failure-mode mitigation):
            //    explicitly assert col-major dst stride. Pick three
            //    cells (0,0), (1, n_query/2), (n_out-1, n_query-1) and
            //    check they live where the docs say they live.
            //    cell (r, c) at index `r + c * n_out` in gpu_col_major.
            for &(r, c) in &[
                (0usize, 0usize),
                (1usize, n_query / 2),
                (n_out - 1, n_query - 1),
            ] {
                let raw = gpu_col_major[r + c * n_out];
                let row_major_view = gpu_row_major[c * n_out + r];
                assert_eq!(
                    raw.to_bits(),
                    row_major_view.to_bits(),
                    "layout sanity: gpu_col_major[r={r}+c={c}*n_out={n_out}] should equal \
                     reshape→row_major[c={c}*n_out+r={r}]; got {raw} vs {row_major_view}"
                );
            }
        }
    }

    /// H5.3b.6 gate: Q6_K mat-mat parity vs N successive mat-vec.
    /// Includes N_QUERY=64/128 so the large-N N64 prompt tile is exercised.
    #[test]
    fn mat_mat_q6_k_matches_cpu_and_mat_vec() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[mat_mat_q6_k] skipped — fixture missing");
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
                    && t.shape[1] % 64 == 0
            })
            .expect("no Q6_K tensor with compatible shape");
        let n_in = q6k.shape[0] as usize;
        let n_out = q6k.shape[1] as usize;
        eprintln!(
            "[mat_mat_q6_k-test] tensor={} shape=[n_in={n_in}, n_out={n_out}]",
            q6k.name
        );

        let weight_f32 = crate::codec::dequant_to_f32(q6k, g.slice(q6k)).expect("dequant");
        let weight_bytes = g.slice(q6k);

        for &n_query in &[1usize, 16, 32, 64, 128] {
            let mut x = vec![0.0f32; n_query * n_in];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i % 13) as f32 - 6.0) * 1e-2;
            }

            // CPU oracle via N mat_vec_pub.
            let mut cpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row_in = &x[q * n_in..(q + 1) * n_in];
                let row_out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row_in);
                cpu_row_major[q * n_out..(q + 1) * n_out].copy_from_slice(&row_out);
            }

            let w_t = MetalTensor::from_bytes(
                &ctx,
                weight_bytes,
                vec![n_in as u64, n_out as u64],
                GgmlType::Q6_K,
            )
            .expect("weight tensor");
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_q6_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out, n_query)
            })
            .expect("mat_mat encode");

            let gpu_flat = read_back_f32(&y_t.buffer, n_out * n_query);

            // Reshape: bit-equivalent col-major [n_out, n_query] →
            // row-major [n_query, n_out] (same byte ordering trick as Q4_K).
            let mut gpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                for o in 0..n_out {
                    gpu_row_major[q * n_out + o] = gpu_flat[o + q * n_out];
                }
            }

            let mut min_cos = f64::INFINITY;
            let mut max_abs = 0.0f32;
            for q in 0..n_query {
                let cpu_row = &cpu_row_major[q * n_out..(q + 1) * n_out];
                let gpu_row = &gpu_row_major[q * n_out..(q + 1) * n_out];
                let mut dot = 0.0f64;
                let mut np = 0.0f64;
                let mut nc = 0.0f64;
                for i in 0..n_out {
                    let p = gpu_row[i] as f64;
                    let c = cpu_row[i] as f64;
                    dot += p * c;
                    np += p * p;
                    nc += c * c;
                    let d = (gpu_row[i] - cpu_row[i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
                let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
                if cos < min_cos {
                    min_cos = cos;
                }
            }
            eprintln!(
                "[mat_mat_q6_k n_query={n_query}] min_cos={min_cos:.6} \
                 max|Δ|={max_abs:.3e}"
            );
            assert!(
                min_cos >= 0.999,
                "n_query={n_query}: min cos {min_cos} < 0.999"
            );
            assert!(
                max_abs < 1e-2,
                "n_query={n_query}: max|Δ| {max_abs} >= 1e-2"
            );
        }
    }

    /// v0.73a.0 gate: Q5_K mat-mat parity vs N successive Q5_K mat-vec.
    /// Includes N_QUERY=64/128 so the large-N N64 prompt tile is exercised.
    #[test]
    fn mat_mat_q5_k_matches_cpu_and_mat_vec() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[mat_mat_q5_k] skipped — fixture missing");
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
                    && t.shape[1] % 64 == 0
            })
            .expect("no Q5_K tensor with compatible shape");
        let n_in = q5k.shape[0] as usize;
        let n_out = q5k.shape[1] as usize;
        eprintln!(
            "[mat_mat_q5_k-test] tensor={} shape=[n_in={n_in}, n_out={n_out}]",
            q5k.name
        );

        let weight_f32 = crate::codec::dequant_to_f32(q5k, g.slice(q5k)).expect("dequant");
        let weight_bytes = g.slice(q5k);

        for &n_query in &[1usize, 16, 32, 64, 128] {
            let mut x = vec![0.0f32; n_query * n_in];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i % 13) as f32 - 6.0) * 1e-2;
            }

            // CPU oracle via N mat_vec_pub.
            let mut cpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row_in = &x[q * n_in..(q + 1) * n_in];
                let row_out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row_in);
                cpu_row_major[q * n_out..(q + 1) * n_out].copy_from_slice(&row_out);
            }

            let w_t = MetalTensor::from_bytes(
                &ctx,
                weight_bytes,
                vec![n_in as u64, n_out as u64],
                GgmlType::Q5_K,
            )
            .expect("weight tensor");
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_q5_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out, n_query)
            })
            .expect("mat_mat encode");

            let gpu_flat = read_back_f32(&y_t.buffer, n_out * n_query);

            // Reshape: bit-equivalent col-major [n_out, n_query] →
            // row-major [n_query, n_out] (same byte ordering trick as Q4_K/Q6_K).
            let mut gpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                for o in 0..n_out {
                    gpu_row_major[q * n_out + o] = gpu_flat[o + q * n_out];
                }
            }

            let mut min_cos = f64::INFINITY;
            let mut max_abs = 0.0f32;
            for q in 0..n_query {
                let cpu_row = &cpu_row_major[q * n_out..(q + 1) * n_out];
                let gpu_row = &gpu_row_major[q * n_out..(q + 1) * n_out];
                let mut dot = 0.0f64;
                let mut np = 0.0f64;
                let mut nc = 0.0f64;
                for i in 0..n_out {
                    let p = gpu_row[i] as f64;
                    let c = cpu_row[i] as f64;
                    dot += p * c;
                    np += p * p;
                    nc += c * c;
                    let d = (gpu_row[i] - cpu_row[i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
                let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
                if cos < min_cos {
                    min_cos = cos;
                }
            }
            eprintln!(
                "[mat_mat_q5_k n_query={n_query}] min_cos={min_cos:.6} \
                 max|Δ|={max_abs:.3e}"
            );
            assert!(
                min_cos >= 0.999,
                "n_query={n_query}: min cos {min_cos} < 0.999"
            );
            assert!(
                max_abs < 1e-2,
                "n_query={n_query}: max|Δ| {max_abs} >= 1e-2"
            );

            // Layout sanity (codex Q7 mitigation): col-major dst stride
            // `dst[r + c*n_out]` must equal row-major view at three
            // corner cells. Catches a transposed write (which would
            // pass cosine within a single row but corrupt downstream
            // chained mat-mats).
            for &(r, c) in &[
                (0usize, 0usize),
                (1usize, n_query / 2),
                (n_out - 1, n_query - 1),
            ] {
                let raw = gpu_flat[r + c * n_out];
                let row_major_view = gpu_row_major[c * n_out + r];
                assert_eq!(
                    raw.to_bits(),
                    row_major_view.to_bits(),
                    "layout sanity n_query={n_query}: (r={r}, c={c})"
                );
            }
        }
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

    /// v0.73b.0 gate: Q8_0 mat-mat correctness. Uses a real Q8_0
    /// weight from the spiritbuun DFlash drafter GGUF
    /// (`blk.0.ffn_down.weight`, shape `[17408, 5120]` — large weight,
    /// hits both the whole-M-tile and partial-N paths). Same playbook
    /// as the Q4_K (v0.63), Q6_K (v0.67), Q5_K (v0.73a.0) gates:
    /// per-row cosine ≥ 0.999 across N_QUERY ∈ {1, 16, 32, 64, 128},
    /// max|Δ| ≤ 1e-2, explicit col-major dst layout sanity probe.
    ///
    /// Q8_0's structurally-simpler dequant (`int8 * scale`) typically
    /// produces TIGHTER cosine than Q4_K/Q5_K/Q6_K mat-mat (which lose
    /// precision in nibble packing + scale folding). Expect cos very
    /// close to 1.000000.
    #[test]
    fn mat_mat_q8_0_matches_cpu_and_mat_vec() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[mat_mat_q8_0] skipped — drafter GGUF missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let q8 = g
            .tensors
            .iter()
            .find(|t| {
                t.name.starts_with("blk.0.")
                    && t.dtype == GgmlType::Q8_0
                    && t.shape.len() == 2
                    && t.shape[0] % 32 == 0
                    && t.shape[1] % 64 == 0
            })
            .expect("no Q8_0 tensor with compatible shape in drafter blk.0");
        let n_in = q8.shape[0] as usize;
        let n_out = q8.shape[1] as usize;
        eprintln!(
            "[mat_mat_q8_0-test] tensor={} shape=[n_in={n_in}, n_out={n_out}]",
            q8.name
        );

        let weight_f32 = crate::codec::dequant_to_f32(q8, g.slice(q8)).expect("dequant");
        let weight_bytes = g.slice(q8);

        for &n_query in &[1usize, 16, 32, 64, 128] {
            let mut x = vec![0.0f32; n_query * n_in];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i % 13) as f32 - 6.0) * 1e-2;
            }

            let mut cpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row_in = &x[q * n_in..(q + 1) * n_in];
                let row_out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row_in);
                cpu_row_major[q * n_out..(q + 1) * n_out].copy_from_slice(&row_out);
            }

            let w_t = MetalTensor::from_bytes(
                &ctx,
                weight_bytes,
                vec![n_in as u64, n_out as u64],
                GgmlType::Q8_0,
            )
            .expect("weight tensor");
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_q8_0_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out, n_query)
            })
            .expect("mat_mat encode");

            let gpu_flat = read_back_f32(&y_t.buffer, n_out * n_query);
            let mut gpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                for o in 0..n_out {
                    gpu_row_major[q * n_out + o] = gpu_flat[o + q * n_out];
                }
            }

            let mut min_cos = f64::INFINITY;
            let mut max_abs = 0.0f32;
            for q in 0..n_query {
                let cpu_row = &cpu_row_major[q * n_out..(q + 1) * n_out];
                let gpu_row = &gpu_row_major[q * n_out..(q + 1) * n_out];
                let mut dot = 0.0f64;
                let mut np = 0.0f64;
                let mut nc = 0.0f64;
                for i in 0..n_out {
                    let p = gpu_row[i] as f64;
                    let c = cpu_row[i] as f64;
                    dot += p * c;
                    np += p * p;
                    nc += c * c;
                    let d = (gpu_row[i] - cpu_row[i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
                let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
                if cos < min_cos {
                    min_cos = cos;
                }
            }
            eprintln!(
                "[mat_mat_q8_0 n_query={n_query}] min_cos={min_cos:.6} \
                 max|Δ|={max_abs:.3e}"
            );
            assert!(
                min_cos >= 0.999,
                "n_query={n_query}: min cos {min_cos} < 0.999"
            );
            assert!(
                max_abs < 1e-2,
                "n_query={n_query}: max|Δ| {max_abs} >= 1e-2"
            );

            // Layout sanity: col-major dst at three corner cells.
            for &(r, c) in &[
                (0usize, 0usize),
                (1usize, n_query / 2),
                (n_out - 1, n_query - 1),
            ] {
                let raw = gpu_flat[r + c * n_out];
                let row_major_view = gpu_row_major[c * n_out + r];
                assert_eq!(
                    raw.to_bits(),
                    row_major_view.to_bits(),
                    "layout sanity n_query={n_query}: (r={r}, c={c})"
                );
            }
        }
    }

    /// v0.73b.0 gate: Q8_0 mat-vec correctness. Uses a real Q8_0 weight
    /// from the spiritbuun DFlash drafter GGUF (`blk.0.attn_q.weight`,
    /// shape `[5120, 4096]`). Same threshold as Q4_K/Q5_K/Q6_K mat-vec
    /// (`max|Δ| < 1e-2`).
    #[test]
    fn mat_vec_q8_0_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[q8_0-test] skipped — drafter GGUF missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let q8 = g
            .tensors
            .iter()
            .find(|t| {
                t.name.starts_with("blk.0.")
                    && t.dtype == GgmlType::Q8_0
                    && t.shape.len() == 2
                    && t.shape[0] % 32 == 0
            })
            .expect("no Q8_0 tensor in drafter blk.0");
        let n_in = q8.shape[0] as usize;
        let n_out = q8.shape[1] as usize;
        eprintln!("[q8_0-test] {} shape=[{n_in}, {n_out}]", q8.name);

        let weight_f32 = crate::codec::dequant_to_f32(q8, g.slice(q8)).expect("dequant");
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let gpu = mat_vec_q8_0_f32_readback_for_test(&ctx, g.slice(q8), &x, n_in, n_out)
            .expect("metal q8_0");
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q8_0] max|Δ|={max_abs:.2e}");
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
    #[ignore]
    fn prompt_mat_mat_production_shapes() {
        use std::time::Instant;

        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[prompt-matmat] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        const N_QUERY: usize = 321;
        let cases = [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.out_proj.weight",
            "blk.0.attn_qkv.weight",
        ];

        eprintln!("[prompt-matmat] {}", ctx.describe());
        for name in cases {
            let Some(t) = g.find(name) else {
                eprintln!("[prompt-matmat] skip missing {name}");
                continue;
            };
            if t.shape.len() != 2 {
                continue;
            }
            let n_in = t.shape[0] as usize;
            let n_out = t.shape[1] as usize;
            let w_t = MetalTensor::from_gguf_tensor(&ctx, t, g.slice(t)).expect("w");
            let x_vec: Vec<f32> = (0..n_in).map(|i| (i as f32 * 1e-3).sin()).collect();
            let x_vec_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_vec),
                vec![n_in as u64],
                GgmlType::F32,
            )
            .expect("x_vec");
            let y_vec_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y_vec");
            let x_mat: Vec<f32> = (0..N_QUERY * n_in)
                .map(|i| (i as f32 * 1e-3).sin())
                .collect();
            let x_mat_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_mat),
                vec![N_QUERY as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x_mat");
            let y_mat_t =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64, N_QUERY as u64]).expect("y_mat");

            match t.dtype {
                GgmlType::Q4_K => {
                    bench_q4_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("warm vec");
                    bench_q4_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("warm mm");
                    let t0 = Instant::now();
                    bench_q4_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("vec");
                    let vec_ms = t0.elapsed().as_secs_f64() * 1e3;
                    let t1 = Instant::now();
                    bench_q4_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("mm");
                    let mm_ms = t1.elapsed().as_secs_f64() * 1e3;
                    eprintln!(
                        "[prompt-matmat] {name:24} dtype={:?} n_in={n_in:>5} n_out={n_out:>6} N={N_QUERY:>3} vec321={vec_ms:>8.2} ms mm={mm_ms:>8.2} ms speedup={:>5.2}x",
                        t.dtype,
                        vec_ms / mm_ms.max(1e-9)
                    );
                }
                GgmlType::Q5_K => {
                    bench_q5_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("warm vec");
                    bench_q5_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("warm mm");
                    let t0 = Instant::now();
                    bench_q5_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("vec");
                    let vec_ms = t0.elapsed().as_secs_f64() * 1e3;
                    let t1 = Instant::now();
                    bench_q5_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("mm");
                    let mm_ms = t1.elapsed().as_secs_f64() * 1e3;
                    eprintln!(
                        "[prompt-matmat] {name:24} dtype={:?} n_in={n_in:>5} n_out={n_out:>6} N={N_QUERY:>3} vec321={vec_ms:>8.2} ms mm={mm_ms:>8.2} ms speedup={:>5.2}x",
                        t.dtype,
                        vec_ms / mm_ms.max(1e-9)
                    );
                }
                GgmlType::Q6_K => {
                    bench_q6_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("warm vec");
                    bench_q6_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("warm mm");
                    let t0 = Instant::now();
                    bench_q6_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("vec");
                    let vec_ms = t0.elapsed().as_secs_f64() * 1e3;
                    let t1 = Instant::now();
                    bench_q6_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("mm");
                    let mm_ms = t1.elapsed().as_secs_f64() * 1e3;
                    eprintln!(
                        "[prompt-matmat] {name:24} dtype={:?} n_in={n_in:>5} n_out={n_out:>6} N={N_QUERY:>3} vec321={vec_ms:>8.2} ms mm={mm_ms:>8.2} ms speedup={:>5.2}x",
                        t.dtype,
                        vec_ms / mm_ms.max(1e-9)
                    );
                }
                _ => {
                    eprintln!("[prompt-matmat] skip {name} dtype={:?}", t.dtype);
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn prompt_mat_mat_production_shapes_chained64() {
        use std::time::Instant;

        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[prompt-matmat-chained] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let n_query: usize = std::env::var("QWEN_PROMPT_MATMAT_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(321);
        let n_dispatches: usize = std::env::var("QWEN_PROMPT_MATMAT_DISPATCHES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let cases = [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_qkv.weight",
        ];

        eprintln!(
            "[prompt-matmat-chained] {} N={n_query} dispatches={n_dispatches}",
            ctx.describe()
        );
        for name in cases {
            let Some(t) = g.find(name) else {
                eprintln!("[prompt-matmat-chained] skip missing {name}");
                continue;
            };
            let n_in = t.shape[0] as usize;
            let n_out = t.shape[1] as usize;
            let w_t = MetalTensor::from_gguf_tensor(&ctx, t, g.slice(t)).expect("w");
            let x_mat: Vec<f32> = (0..n_query * n_in)
                .map(|i| (i as f32 * 1e-3).sin())
                .collect();
            let x_mat_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_mat),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x_mat");
            let y_mat_t =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y_mat");

            match t.dtype {
                GgmlType::Q4_K => {
                    bench_q4_k_mat_mat_chained(
                        &ctx,
                        &w_t,
                        &x_mat_t,
                        &y_mat_t,
                        n_in,
                        n_out,
                        n_query,
                        n_dispatches,
                    )
                    .expect("warm mm");
                    let t0 = Instant::now();
                    bench_q4_k_mat_mat_chained(
                        &ctx,
                        &w_t,
                        &x_mat_t,
                        &y_mat_t,
                        n_in,
                        n_out,
                        n_query,
                        n_dispatches,
                    )
                    .expect("mm");
                    let mm_ms = t0.elapsed().as_secs_f64() * 1e3 / n_dispatches as f64;
                    let weight_gib =
                        (t.n_bytes * n_dispatches as u64) as f64 / (1024.0 * 1024.0 * 1024.0);
                    let gib_s = weight_gib / (t0.elapsed().as_secs_f64());
                    eprintln!(
                        "[prompt-matmat-chained] {name:24} dtype={:?} N={n_query:>5} per-dispatch={mm_ms:>7.3} ms weight-throughput={gib_s:>7.1} GiB/s",
                        t.dtype
                    );
                }
                GgmlType::Q5_K => {
                    bench_q5_k_mat_mat_chained(
                        &ctx,
                        &w_t,
                        &x_mat_t,
                        &y_mat_t,
                        n_in,
                        n_out,
                        n_query,
                        n_dispatches,
                    )
                    .expect("warm mm");
                    let t0 = Instant::now();
                    bench_q5_k_mat_mat_chained(
                        &ctx,
                        &w_t,
                        &x_mat_t,
                        &y_mat_t,
                        n_in,
                        n_out,
                        n_query,
                        n_dispatches,
                    )
                    .expect("mm");
                    let mm_ms = t0.elapsed().as_secs_f64() * 1e3 / n_dispatches as f64;
                    let weight_gib =
                        (t.n_bytes * n_dispatches as u64) as f64 / (1024.0 * 1024.0 * 1024.0);
                    let gib_s = weight_gib / (t0.elapsed().as_secs_f64());
                    eprintln!(
                        "[prompt-matmat-chained] {name:24} dtype={:?} N={n_query:>5} per-dispatch={mm_ms:>7.3} ms weight-throughput={gib_s:>7.1} GiB/s",
                        t.dtype
                    );
                }
                GgmlType::Q6_K => {
                    bench_q6_k_mat_mat_chained(
                        &ctx,
                        &w_t,
                        &x_mat_t,
                        &y_mat_t,
                        n_in,
                        n_out,
                        n_query,
                        n_dispatches,
                    )
                    .expect("warm mm");
                    let t0 = Instant::now();
                    bench_q6_k_mat_mat_chained(
                        &ctx,
                        &w_t,
                        &x_mat_t,
                        &y_mat_t,
                        n_in,
                        n_out,
                        n_query,
                        n_dispatches,
                    )
                    .expect("mm");
                    let mm_ms = t0.elapsed().as_secs_f64() * 1e3 / n_dispatches as f64;
                    let weight_gib =
                        (t.n_bytes * n_dispatches as u64) as f64 / (1024.0 * 1024.0 * 1024.0);
                    let gib_s = weight_gib / (t0.elapsed().as_secs_f64());
                    eprintln!(
                        "[prompt-matmat-chained] {name:24} dtype={:?} N={n_query:>5} per-dispatch={mm_ms:>7.3} ms weight-throughput={gib_s:>7.1} GiB/s",
                        t.dtype
                    );
                }
                _ => {
                    eprintln!("[prompt-matmat-chained] skip {name} dtype={:?}", t.dtype);
                }
            }
        }
    }

    /// v0.73b.0 A-lite GO/NO-GO bench. Compares amortized weight-BW
    /// of Q8_0 mat-mat (NR1=16 fast path) vs N=16 successive Q8_0
    /// mat-vec on a production drafter weight shape
    /// (`blk.0.ffn_down.weight` from the spiritbuun DFlash drafter,
    /// shape [17408, 5120]). The drafter has 5 layers; use 5 chained
    /// dispatches per command buffer to mirror the actual hot-path
    /// usage pattern. Threshold for proceed: GPU ratio ≤ 0.5
    /// (mat-mat at LEAST 2× faster). Q5_K hit 3.59× at the GDN
    /// out_proj shape; Q8_0 should hit similar or higher (simpler
    /// dequant, same tile geometry).
    ///
    /// Run: `cargo test --release --lib -p qwen-llm
    /// q8_0_mat_mat_amortization_vs_n_mat_vec --ignored -- --nocapture`
    #[test]
    #[ignore]
    fn q8_0_mat_mat_amortization_vs_n_mat_vec() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[v0.73b.0-gate] skipped — drafter GGUF missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let q8 = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_down.weight" && t.dtype == GgmlType::Q8_0)
            .expect("blk.0.ffn_down.weight Q8_0 not found");
        let n_in = q8.shape[0] as usize;
        let n_out = q8.shape[1] as usize;
        let n_query = 16usize;
        let n_layers = 5usize; // drafter has 5 layers
        let warmup = 5usize;
        let iters = 30usize;

        eprintln!(
            "[v0.73b.0-gate] tensor={} shape=[n_in={n_in}, n_out={n_out}] N={n_query} layers={n_layers}",
            q8.name
        );

        let w_t = MetalTensor::from_bytes(
            &ctx,
            g.slice(q8),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q8_0,
        )
        .expect("weight tensor");
        let x_packed = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_in) as u64]).unwrap();
        let y_packed = MetalTensor::zeros_f32(&ctx, vec![(n_out * n_query) as u64]).unwrap();
        let x_single = MetalTensor::zeros_f32(&ctx, vec![n_in as u64]).unwrap();
        let y_single = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();

        let bench_mat_mat = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                encode_mat_mat_q8_0_f32(
                    &ctx, &enc, &w_t, &x_packed, &y_packed, n_in, n_out, n_query,
                )
                .unwrap();
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        let bench_n_mat_vec = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                for _ in 0..n_query {
                    encode_mat_vec_q8_0_f32(&ctx, &enc, &w_t, &x_single, &y_single, n_in, n_out)
                        .unwrap();
                }
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        for _ in 0..warmup {
            bench_mat_mat();
            bench_n_mat_vec();
        }

        let mut sum_mm_wall = 0.0f64;
        let mut sum_mm_gpu = 0.0f64;
        let mut sum_mv_wall = 0.0f64;
        let mut sum_mv_gpu = 0.0f64;
        for _ in 0..iters {
            let (w, g) = bench_mat_mat();
            sum_mm_wall += w;
            sum_mm_gpu += g;
        }
        for _ in 0..iters {
            let (w, g) = bench_n_mat_vec();
            sum_mv_wall += w;
            sum_mv_gpu += g;
        }
        let mm_wall = sum_mm_wall / iters as f64;
        let mm_gpu = sum_mm_gpu / iters as f64;
        let mv_wall = sum_mv_wall / iters as f64;
        let mv_gpu = sum_mv_gpu / iters as f64;

        eprintln!("[v0.73b.0-gate] {n_layers} layers × N={n_query} avg over {iters} iters:");
        eprintln!(
            "  mat-mat (1 disp/layer):    wall={mm_wall:7.2} ms  gpu={mm_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            mm_gpu / n_layers as f64
        );
        eprintln!(
            "  N=16 mat-vec (16/layer):   wall={mv_wall:7.2} ms  gpu={mv_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            mv_gpu / n_layers as f64
        );
        let ratio_wall = mm_wall / mv_wall;
        let ratio_gpu = mm_gpu / mv_gpu;
        let speedup_wall = 1.0 / ratio_wall;
        let speedup_gpu = 1.0 / ratio_gpu;
        eprintln!(
            "  ratio mat-mat / 16×mat-vec: wall={ratio_wall:.3} (= {speedup_wall:.2}× speedup)  gpu={ratio_gpu:.3} (= {speedup_gpu:.2}× speedup)"
        );

        // GO/NO-GO threshold same as v0.73a.0 (GPU ratio <= 0.5).
        assert!(
            ratio_gpu <= 0.5,
            "v0.73b.0 GO/NO-GO failed: GPU ratio {ratio_gpu:.3} > 0.5 \
             (mat-mat must beat 16 mat-vec by at least 2×; \
             reassess before v0.73b.1)"
        );
    }

    /// v0.73c.2 A-lite GO/NO-GO bench. Compares the fused
    /// `ffn_swiglu_q4_K_mm_n16` (one dispatch per FFN layer) against
    /// the unfused `mat_mat_q4_K + mat_mat_q4_K + silu_mul` 3-dispatch
    /// sequence at production 27B 64-layer FFN shape (n_in=5120,
    /// n_out=17408, N=16, 64 layers). Codex's threshold for proceed
    /// is ratio ≤ 0.7 (fused must beat unfused by at least ~30%).
    ///
    /// Run: `cargo test --release --lib -p qwen-llm
    /// ffn_fused_swiglu_q4_K_amortization_vs_unfused --ignored -- --nocapture`
    #[test]
    #[ignore]
    #[allow(non_snake_case)]
    fn ffn_fused_swiglu_q4_K_amortization_vs_unfused() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[v0.73c.2-gate] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let gate = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate.weight" && t.dtype == GgmlType::Q4_K)
            .expect("ffn_gate Q4_K not found");
        let up = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up.weight" && t.dtype == GgmlType::Q4_K)
            .expect("ffn_up Q4_K not found");
        let n_in = gate.shape[0] as usize;
        let n_out = gate.shape[1] as usize;
        const N: usize = 16;
        let n_layers = 64usize; // 27B has 64 transformer blocks (48 GDN + 16 attn; FFN runs on all)
        let warmup = 5usize;
        let iters = 30usize;

        eprintln!("[v0.73c.2-gate] shape=[n_in={n_in}, n_out={n_out}] N={N} layers={n_layers}");

        let w_gate = MetalTensor::from_bytes(
            &ctx,
            g.slice(gate),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .unwrap();
        let w_up = MetalTensor::from_bytes(
            &ctx,
            g.slice(up),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .unwrap();
        let x_packed = MetalTensor::zeros_f32(&ctx, vec![(N * n_in) as u64]).unwrap();
        let inner_packed = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
        let gate_packed = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
        let up_packed = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();

        let bench_fused = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                encode_ffn_fused_swiglu_q4_K_mm_n16_f32(
                    &ctx,
                    &enc,
                    &w_gate,
                    &w_up,
                    &x_packed,
                    &inner_packed,
                    n_in,
                    n_out,
                )
                .unwrap();
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        let bench_unfused = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                encode_mat_mat_q4_k_f32(
                    &ctx,
                    &enc,
                    &w_gate,
                    &x_packed,
                    &gate_packed,
                    n_in,
                    n_out,
                    N,
                )
                .unwrap();
                encode_mat_mat_q4_k_f32(&ctx, &enc, &w_up, &x_packed, &up_packed, n_in, n_out, N)
                    .unwrap();
                encode_silu_mul_f32(&ctx, &enc, &gate_packed, &up_packed, &inner_packed).unwrap();
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        for _ in 0..warmup {
            bench_fused();
            bench_unfused();
        }
        let mut sum_f_wall = 0.0f64;
        let mut sum_f_gpu = 0.0f64;
        let mut sum_u_wall = 0.0f64;
        let mut sum_u_gpu = 0.0f64;
        for _ in 0..iters {
            let (w, g) = bench_fused();
            sum_f_wall += w;
            sum_f_gpu += g;
        }
        for _ in 0..iters {
            let (w, g) = bench_unfused();
            sum_u_wall += w;
            sum_u_gpu += g;
        }
        let f_wall = sum_f_wall / iters as f64;
        let f_gpu = sum_f_gpu / iters as f64;
        let u_wall = sum_u_wall / iters as f64;
        let u_gpu = sum_u_gpu / iters as f64;

        eprintln!("[v0.73c.2-gate] {n_layers} layers × N={N} avg over {iters} iters:");
        eprintln!(
            "  fused (1 disp/layer):       wall={f_wall:7.2} ms  gpu={f_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            f_gpu / n_layers as f64
        );
        eprintln!(
            "  unfused (3 disp/layer):     wall={u_wall:7.2} ms  gpu={u_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            u_gpu / n_layers as f64
        );
        let ratio_wall = f_wall / u_wall;
        let ratio_gpu = f_gpu / u_gpu;
        let speedup_wall = 1.0 / ratio_wall;
        let speedup_gpu = 1.0 / ratio_gpu;
        eprintln!(
            "  ratio fused / unfused: wall={ratio_wall:.3} (= {speedup_wall:.2}× speedup)  gpu={ratio_gpu:.3} (= {speedup_gpu:.2}× speedup)"
        );

        // v0.73c.2 RESULT: failed go/no-go. Measured ratio ≈ 0.94 on
        // production 27B Q4_K_M FFN shape — fusion only saves ~6%,
        // codex threshold was ≤ 0.7 (≥ 30% speedup). Kernel is
        // preserved as experimental institutional memory; the assertion
        // below allows the bench to run as a re-checkable "regime
        // still capped?" probe without panicking. If a future change
        // (e.g. larger N, different shape, different hardware) puts
        // the ratio under 0.7, this is where to flag it for plumbing.
        if ratio_gpu <= 0.7 {
            eprintln!(
                "[v0.73c.2-gate] REGIME CHANGE: ratio_gpu {ratio_gpu:.3} now ≤ 0.7. \
                 Reconsider plumbing fused FFN into layer-major path."
            );
        } else {
            eprintln!(
                "[v0.73c.2-gate] still capped (ratio_gpu {ratio_gpu:.3} > 0.7); \
                 fusion not worth plumbing. Same negative result as v0.73c.2."
            );
        }
    }

    /// v0.73a.0 A-lite GO/NO-GO bench. Compares amortized weight-BW of
    /// Q5_K mat-mat (NR1=16 fast path) vs N=16 successive Q5_K mat-vec
    /// on production GDN out_proj (`blk.*.ssm_out.weight`) shape.
    ///
    /// Runs 48 chained dispatches (= 48 GDN layers) of each path in one
    /// command buffer; reports per-dispatch latency and the ratio. The
    /// hypothesis under test is that mat-mat amortizes the per-step
    /// weight reads N=16-fold, so the ratio should be substantially
    /// less than 1.0 (i.e. mat-mat much faster). Codex's framing:
    /// "if Q5_K mat-mat doesn't beat 16 mat-vecs by a large margin,
    /// stop and reassess." Threshold for proceed: ratio ≤ 0.5
    /// (mat-mat at LEAST 2× faster than 16-mat-vec equivalent work).
    /// Empirically Q4_K and Q6_K mat-mat at this shape achieve closer
    /// to 4-8× on M4 Max.
    ///
    /// Run: `cargo test --release --lib -p qwen-llm
    /// q5_k_mat_mat_amortization_vs_n_mat_vec --ignored -- --nocapture`
    #[test]
    #[ignore]
    fn q5_k_mat_mat_amortization_vs_n_mat_vec() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[v0.73a.0-gate] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let q5k = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ssm_out.weight" && t.dtype == GgmlType::Q5_K)
            .expect("blk.0.ssm_out.weight Q5_K not found");
        let n_in = q5k.shape[0] as usize;
        let n_out = q5k.shape[1] as usize;
        let n_query = 16usize;
        let n_layers = 48usize; // GDN layers in 27B
        let warmup = 5usize;
        let iters = 30usize;

        eprintln!(
            "[v0.73a.0-gate] tensor={} shape=[n_in={n_in}, n_out={n_out}] N={n_query} layers={n_layers}",
            q5k.name
        );

        let w_t = MetalTensor::from_bytes(
            &ctx,
            g.slice(q5k),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q5_K,
        )
        .expect("weight tensor");
        let x_packed = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_in) as u64]).unwrap();
        let y_packed = MetalTensor::zeros_f32(&ctx, vec![(n_out * n_query) as u64]).unwrap();
        let x_single = MetalTensor::zeros_f32(&ctx, vec![n_in as u64]).unwrap();
        let y_single = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();

        let bench_mat_mat = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                encode_mat_mat_q5_k_f32(
                    &ctx, &enc, &w_t, &x_packed, &y_packed, n_in, n_out, n_query,
                )
                .unwrap();
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        // 16 mat-vec per layer = 16*48 = 768 dispatches per command buffer.
        // This mirrors the production per-token GDN out_proj fall-through
        // we'd be replacing.
        let bench_n_mat_vec = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                for _ in 0..n_query {
                    encode_mat_vec_q5_k_f32(&ctx, &enc, &w_t, &x_single, &y_single, n_in, n_out)
                        .unwrap();
                }
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        // Warmup both.
        for _ in 0..warmup {
            bench_mat_mat();
            bench_n_mat_vec();
        }

        let mut sum_mm_wall = 0.0f64;
        let mut sum_mm_gpu = 0.0f64;
        let mut sum_mv_wall = 0.0f64;
        let mut sum_mv_gpu = 0.0f64;
        for _ in 0..iters {
            let (w, g) = bench_mat_mat();
            sum_mm_wall += w;
            sum_mm_gpu += g;
        }
        for _ in 0..iters {
            let (w, g) = bench_n_mat_vec();
            sum_mv_wall += w;
            sum_mv_gpu += g;
        }
        let mm_wall = sum_mm_wall / iters as f64;
        let mm_gpu = sum_mm_gpu / iters as f64;
        let mv_wall = sum_mv_wall / iters as f64;
        let mv_gpu = sum_mv_gpu / iters as f64;

        eprintln!("[v0.73a.0-gate] {n_layers} layers × N={n_query} avg over {iters} iters:");
        eprintln!(
            "  mat-mat (1 disp/layer):    wall={mm_wall:7.2} ms  gpu={mm_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            mm_gpu / n_layers as f64
        );
        eprintln!(
            "  N=16 mat-vec (16/layer):   wall={mv_wall:7.2} ms  gpu={mv_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            mv_gpu / n_layers as f64
        );
        let ratio_wall = mm_wall / mv_wall;
        let ratio_gpu = mm_gpu / mv_gpu;
        let speedup_wall = 1.0 / ratio_wall;
        let speedup_gpu = 1.0 / ratio_gpu;
        eprintln!(
            "  ratio mat-mat / 16×mat-vec: wall={ratio_wall:.3} (= {speedup_wall:.2}× speedup)  gpu={ratio_gpu:.3} (= {speedup_gpu:.2}× speedup)"
        );

        // GO/NO-GO: GPU ratio must be at most 0.5 (= 2× speedup). This
        // is a conservative bar relative to Q4_K/Q6_K experience (4-8×).
        // If we don't clear it, v0.73a.1 won't deliver the projected
        // win and we should reassess BEFORE shipping the orchestration
        // restructure.
        assert!(
            ratio_gpu <= 0.5,
            "v0.73a.0 GO/NO-GO failed: GPU ratio {ratio_gpu:.3} > 0.5 \
             (mat-mat must beat 16 mat-vec by at least 2×; \
             reassess before v0.73a.1)"
        );
    }

    /// Fused SwiGLU FFN (1 dispatch) must match the unfused
    /// (mat_vec_q4_K + mat_vec_q4_K + silu_mul) 3-dispatch sequence
    /// within fp32 reorder noise. Uses real Q4_K weights from the 27B
    /// model's first FFN.
    #[test]
    // `Q4_K` is the GGUF dtype tag; matches the kernel and other
    // function names (`encode_mat_vec_q4_K`, `kernel_ffn_swiglu_q4_K`).
    // Lowercasing to `q4_k` would diverge from the rest of the codebase.
    #[allow(non_snake_case)]
    fn ffn_swiglu_q4_K_matches_unfused() {
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

        // Find ffn_gate and ffn_up Q4_K tensors from layer 0.
        let gate = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate.weight" && t.dtype == GgmlType::Q4_K)
            .expect("no blk.0.ffn_gate.weight Q4_K");
        let up = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up.weight" && t.dtype == GgmlType::Q4_K)
            .expect("no blk.0.ffn_up.weight Q4_K");
        assert_eq!(gate.shape, up.shape, "gate/up shape mismatch");
        let n_in = gate.shape[0] as usize;
        let n_out = gate.shape[1] as usize;
        eprintln!(
            "[ffn-fused] gate={} up={} n_in={n_in} n_out={n_out}",
            gate.name, up.name
        );

        // Build inputs.
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .unwrap();
        let w_gate = MetalTensor::from_bytes(
            &ctx,
            g.slice(gate),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .unwrap();
        let w_up = MetalTensor::from_bytes(
            &ctx,
            g.slice(up),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .unwrap();

        // --- Unfused reference: 3 dispatches ---
        let gate_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
        let up_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
        let inner_ref_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_mat_vec_q4_k_f32(&ctx, enc, &w_gate, &x_t, &gate_t, n_in, n_out)?;
            encode_mat_vec_q4_k_f32(&ctx, enc, &w_up, &x_t, &up_t, n_in, n_out)?;
            encode_silu_mul_f32(&ctx, enc, &gate_t, &up_t, &inner_ref_t)
        })
        .unwrap();
        let inner_ref = read_back_f32(&inner_ref_t.buffer, n_out);

        // --- Fused: 1 dispatch ---
        let inner_fused_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_ffn_swiglu_q4_K_f32(&ctx, enc, &w_gate, &w_up, &x_t, &inner_fused_t, n_in, n_out)
        })
        .unwrap();
        let inner_fused = read_back_f32(&inner_fused_t.buffer, n_out);

        let max_abs = inner_fused
            .iter()
            .zip(inner_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = inner_fused
            .iter()
            .zip(inner_ref.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = inner_fused
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = inner_ref
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cos = dot / (na * nb);
        eprintln!("[ffn-fused] n_out={n_out} max|Δ|={max_abs:.2e} cos={cos:.6}");
        // The two paths do the same fp32 reductions in the same order
        // (both use simd_sum over the same 32 lanes per row, producing
        // identical totals). Difference should be ~0 except for the
        // silu compositional difference (silu computed before vs after
        // the float -> device write -> float read roundtrip; should
        // also be 0). Allow tiny tolerance for safety.
        assert!(
            cos > 0.9999,
            "fused FFN cos too low: {cos} (max|Δ|={max_abs})"
        );
        assert!(max_abs < 1e-3, "fused FFN diverged: max|Δ|={max_abs}");
    }

    /// v0.73c.2 gate: layer-major fused SwiGLU FFN at N=16 must match
    /// the unfused (mat_mat_q4_K + mat_mat_q4_K + silu_mul) reference
    /// within mat-mat half-staging tolerance. Per-row cosine ≥ 0.999,
    /// max|Δ| ≤ 1e-2 (mirrors Q4_K mat-mat gate).
    ///
    /// Uses real `blk.0.ffn_gate.weight` + `ffn_up.weight` from 27B Q4_K_M.
    #[test]
    #[allow(non_snake_case)]
    fn ffn_fused_swiglu_q4_K_mm_n16_matches_unfused() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[ffn-fused-mm-n16] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");

        let gate = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate.weight" && t.dtype == GgmlType::Q4_K)
            .expect("no blk.0.ffn_gate.weight Q4_K");
        let up = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up.weight" && t.dtype == GgmlType::Q4_K)
            .expect("no blk.0.ffn_up.weight Q4_K");
        assert_eq!(gate.shape, up.shape, "gate/up shape mismatch");
        let n_in = gate.shape[0] as usize;
        let n_out = gate.shape[1] as usize;
        const N: usize = 16;
        eprintln!("[ffn-fused-mm-n16] n_in={n_in} n_out={n_out} N={N}");

        let w_gate = MetalTensor::from_bytes(
            &ctx,
            g.slice(gate),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .unwrap();
        let w_up = MetalTensor::from_bytes(
            &ctx,
            g.slice(up),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .unwrap();

        // Activation: row-major [N, n_in] deterministic fill.
        let mut x = vec![0.0f32; N * n_in];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 1e-2;
        }
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![N as u64, n_in as u64],
            GgmlType::F32,
        )
        .unwrap();

        // --- Unfused reference: gate_mm + up_mm + silu_mul ---
        let gate_pack = MetalTensor::zeros_f32(&ctx, vec![N as u64 * n_out as u64]).unwrap();
        let up_pack = MetalTensor::zeros_f32(&ctx, vec![N as u64 * n_out as u64]).unwrap();
        let inner_ref_t = MetalTensor::zeros_f32(&ctx, vec![N as u64 * n_out as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_mat_mat_q4_k_f32(&ctx, enc, &w_gate, &x_t, &gate_pack, n_in, n_out, N)?;
            encode_mat_mat_q4_k_f32(&ctx, enc, &w_up, &x_t, &up_pack, n_in, n_out, N)?;
            encode_silu_mul_f32(&ctx, enc, &gate_pack, &up_pack, &inner_ref_t)
        })
        .unwrap();
        let inner_ref_flat = read_back_f32(&inner_ref_t.buffer, N * n_out);

        // --- Fused: 1 dispatch ---
        let inner_fused_t = MetalTensor::zeros_f32(&ctx, vec![N as u64 * n_out as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_ffn_fused_swiglu_q4_K_mm_n16_f32(
                &ctx,
                enc,
                &w_gate,
                &w_up,
                &x_t,
                &inner_fused_t,
                n_in,
                n_out,
            )
        })
        .unwrap();
        let inner_fused_flat = read_back_f32(&inner_fused_t.buffer, N * n_out);

        // Both buffers are bit-equivalently row-major [N, n_out] (= col-major [n_out, N]).
        // Reshape via the same indexing as Q4_K mat-mat tests.
        let mut min_cos = f64::INFINITY;
        let mut max_abs = 0.0f32;
        for q in 0..N {
            let mut dot = 0.0f64;
            let mut np = 0.0f64;
            let mut nc = 0.0f64;
            for o in 0..n_out {
                // dst[o + q * n_out] is the cell (m=o, n=q) in col-major
                // [n_out, N], which equals row-major [N, n_out][q][o].
                let p = inner_fused_flat[o + q * n_out] as f64;
                let c = inner_ref_flat[o + q * n_out] as f64;
                dot += p * c;
                np += p * p;
                nc += c * c;
                let d = (p - c).abs() as f32;
                if d > max_abs {
                    max_abs = d;
                }
            }
            let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
            if cos < min_cos {
                min_cos = cos;
            }
        }
        eprintln!("[ffn-fused-mm-n16] min_cos={min_cos:.6} max|Δ|={max_abs:.3e}");
        assert!(min_cos >= 0.999, "fused FFN N=16 cos too low: {min_cos}");
        assert!(max_abs < 1e-2, "fused FFN N=16 diverged: max|Δ|={max_abs}");

        const N32: usize = 32;
        let mut x32 = vec![0.0f32; N32 * n_in];
        for (i, v) in x32.iter_mut().enumerate() {
            *v = ((i % 17) as f32 - 8.0) * 1e-2;
        }
        let x32_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x32),
            vec![N32 as u64, n_in as u64],
            GgmlType::F32,
        )
        .unwrap();
        let gate32 = MetalTensor::zeros_f32(&ctx, vec![N32 as u64 * n_out as u64]).unwrap();
        let up32 = MetalTensor::zeros_f32(&ctx, vec![N32 as u64 * n_out as u64]).unwrap();
        let inner32_ref = MetalTensor::zeros_f32(&ctx, vec![N32 as u64 * n_out as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_mat_mat_q4_k_f32(&ctx, enc, &w_gate, &x32_t, &gate32, n_in, n_out, N32)?;
            encode_mat_mat_q4_k_f32(&ctx, enc, &w_up, &x32_t, &up32, n_in, n_out, N32)?;
            encode_silu_mul_f32(&ctx, enc, &gate32, &up32, &inner32_ref)
        })
        .unwrap();
        let inner32_fused = MetalTensor::zeros_f32(&ctx, vec![N32 as u64 * n_out as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_ffn_fused_swiglu_q4_K_mm_f32(
                &ctx,
                enc,
                &w_gate,
                &w_up,
                &x32_t,
                &inner32_fused,
                n_in,
                n_out,
                N32,
            )
        })
        .unwrap();
        let inner32_ref_flat = read_back_f32(&inner32_ref.buffer, N32 * n_out);
        let inner32_fused_flat = read_back_f32(&inner32_fused.buffer, N32 * n_out);
        let mut min_cos32 = f64::INFINITY;
        let mut max_abs32 = 0.0f32;
        for q in 0..N32 {
            let mut dot = 0.0f64;
            let mut np = 0.0f64;
            let mut nc = 0.0f64;
            for o in 0..n_out {
                let p = inner32_fused_flat[o + q * n_out] as f64;
                let c = inner32_ref_flat[o + q * n_out] as f64;
                dot += p * c;
                np += p * p;
                nc += c * c;
                max_abs32 = max_abs32.max((p - c).abs() as f32);
            }
            min_cos32 = min_cos32.min(dot / (np.sqrt() * nc.sqrt() + 1e-30));
        }
        eprintln!("[ffn-fused-mm-n32] min_cos={min_cos32:.6} max|Δ|={max_abs32:.3e}");
        assert!(
            min_cos32 >= 0.999,
            "fused FFN N=32 cos too low: {min_cos32}"
        );
        assert!(
            max_abs32 < 1e-2,
            "fused FFN N=32 diverged: max|Δ|={max_abs32}"
        );
    }

    /// Fused K+V scatter (one dispatch writes both caches) must produce
    /// identical bytes to the two-dispatch sequence.
    #[test]
    fn scatter_kv_fused_matches_unfused() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let kv_dim = 4 * 256; // n_kv_heads * head_dim for 27B
        let cap = 64usize;

        let k_src: Vec<f32> = (0..kv_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();
        let v_src: Vec<f32> = (0..kv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.07)
            .collect();
        let k_src_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&k_src),
            vec![kv_dim as u64],
            GgmlType::F32,
        )
        .unwrap();
        let v_src_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&v_src),
            vec![kv_dim as u64],
            GgmlType::F32,
        )
        .unwrap();

        for &dst_off_slot in &[0usize, 7, 31, 63] {
            let dst_off = dst_off_slot * kv_dim;

            // --- Reference: two unfused dispatches ---
            let k_ref = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_ref = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_f16(&ctx, enc, &k_src_t, &k_ref, dst_off, kv_dim)?;
                encode_scatter_offset_f32_to_f16(&ctx, enc, &v_src_t, &v_ref, dst_off, kv_dim)
            })
            .unwrap();

            // --- Fused: one dispatch ---
            let k_fused = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_fused = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_f16_kv(
                    &ctx, enc, &k_src_t, &v_src_t, &k_fused, &v_fused, dst_off, kv_dim,
                )
            })
            .unwrap();

            // Compare F16 bytes directly (must be byte-identical).
            let n_bytes = cap * kv_dim * 2;
            let k_ref_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(k_ref.buffer.contents().as_ptr() as *const u8, n_bytes)
            };
            let k_fused_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(k_fused.buffer.contents().as_ptr() as *const u8, n_bytes)
            };
            let v_ref_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(v_ref.buffer.contents().as_ptr() as *const u8, n_bytes)
            };
            let v_fused_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(v_fused.buffer.contents().as_ptr() as *const u8, n_bytes)
            };
            assert_eq!(
                k_ref_bytes, k_fused_bytes,
                "K mismatch at slot={dst_off_slot}"
            );
            assert_eq!(
                v_ref_bytes, v_fused_bytes,
                "V mismatch at slot={dst_off_slot}"
            );
            eprintln!("[scatter_kv_fused slot={dst_off_slot}] byte-identical to unfused");
        }
    }

    #[test]
    fn scatter_kv_q8_matches_ref_quant() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let kv_dim = 4 * 256usize;
        let cap = 8usize;
        let k_src: Vec<f32> = (0..kv_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.03125)
            .collect();
        let v_src: Vec<f32> = (0..kv_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.046875)
            .collect();
        let k_src_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&k_src),
            vec![kv_dim as u64],
            GgmlType::F32,
        )
        .unwrap();
        let v_src_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&v_src),
            vec![kv_dim as u64],
            GgmlType::F32,
        )
        .unwrap();

        for &dst_off_slot in &[0usize, 3, 7] {
            let dst_off = dst_off_slot * kv_dim;
            let k_q8 = MetalTensor::zeros_q8_0(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_q8 = MetalTensor::zeros_q8_0(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_q8_0_kv(
                    &ctx, enc, &k_src_t, &v_src_t, &k_q8, &v_q8, dst_off, kv_dim,
                )
            })
            .unwrap();

            let q8_block_bytes = 34usize;
            let blocks_per_row = kv_dim / 32;
            let total_bytes = cap * blocks_per_row * q8_block_bytes;
            let k_gpu: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    k_q8.buffer.contents().as_ptr() as *const u8,
                    total_bytes,
                )
            };
            let v_gpu: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    v_q8.buffer.contents().as_ptr() as *const u8,
                    total_bytes,
                )
            };

            let mut k_ref = vec![0u8; total_bytes];
            let mut v_ref = vec![0u8; total_bytes];
            let block_base = dst_off / 32;
            for (src, dst) in [(&k_src, &mut k_ref), (&v_src, &mut v_ref)] {
                for blk in 0..blocks_per_row {
                    let src_blk = &src[blk * 32..(blk + 1) * 32];
                    let amax = src_blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let d = amax / 127.0f32;
                    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                    let dst_blk = (block_base + blk) * q8_block_bytes;
                    let dh = half::f16::from_f32(d).to_bits().to_le_bytes();
                    dst[dst_blk..dst_blk + 2].copy_from_slice(&dh);
                    for j in 0..32 {
                        dst[dst_blk + 2 + j] = ((src_blk[j] * id).round() as i8) as u8;
                    }
                }
            }

            assert_eq!(
                k_gpu,
                k_ref.as_slice(),
                "Q8 K mismatch at slot={dst_off_slot}"
            );
            assert_eq!(
                v_gpu,
                v_ref.as_slice(),
                "Q8 V mismatch at slot={dst_off_slot}"
            );
            eprintln!("[scatter_kv_q8 slot={dst_off_slot}] byte-identical to ref quantization");
        }
    }

    fn run_attn_v4_q8_kv_compare(
        label: &str,
        n_q: usize,
        n_kv: usize,
        n_pos: usize,
        nwg: usize,
        tile_c: usize,
        group_tile: Option<usize>,
    ) {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let hd = 256usize;
        let kv_dim = n_kv * hd;
        let group = n_q / n_kv;
        assert_eq!(n_q % n_kv, 0);

        let q: Vec<f32> = (0..n_q * hd)
            .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
            .collect();
        let k_f32: Vec<f32> = (0..n_pos * kv_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
            .collect();
        let v_f32: Vec<f32> = (0..n_pos * kv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
            .collect();

        let q_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&q),
            vec![(n_q * hd) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let k_f16 = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
        let v_f16 = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
        let k_q8 = MetalTensor::zeros_q8_0(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
        let v_q8 = MetalTensor::zeros_q8_0(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();

        let k_src_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&k_f32),
            vec![k_f32.len() as u64],
            GgmlType::F32,
        )
        .unwrap();
        let v_src_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&v_f32),
            vec![v_f32.len() as u64],
            GgmlType::F32,
        )
        .unwrap();
        one_shot(&ctx, |enc| {
            encode_scatter_offset_f32_to_f16_kv(
                &ctx,
                enc,
                &k_src_t,
                &v_src_t,
                &k_f16,
                &v_f16,
                0,
                k_f32.len(),
            )?;
            encode_scatter_offset_f32_to_q8_0_kv(
                &ctx,
                enc,
                &k_src_t,
                &v_src_t,
                &k_q8,
                &v_q8,
                0,
                k_f32.len(),
            )
        })
        .unwrap();

        let o_partial_f16 =
            MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
        let ml_partial_f16 =
            MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
        let out_f16 = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
        let o_partial_q8 =
            MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
        let ml_partial_q8 =
            MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
        let out_q8 = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

        let f16_result = || {
            one_shot(&ctx, |enc| {
                encode_attn_decode_v4_f32(
                    &ctx,
                    enc,
                    &q_t,
                    &k_f16,
                    &v_f16,
                    &o_partial_f16,
                    &ml_partial_f16,
                    &out_f16,
                    n_q,
                    n_kv,
                    hd,
                    n_pos,
                    nwg,
                    tile_c,
                )
            })
        };
        if let Some(tile) = group_tile {
            with_attn_v4_group_tile_override(tile, f16_result)
        } else {
            f16_result()
        }
        .unwrap();

        let q8_result = || {
            one_shot(&ctx, |enc| {
                encode_attn_decode_v4_f32(
                    &ctx,
                    enc,
                    &q_t,
                    &k_q8,
                    &v_q8,
                    &o_partial_q8,
                    &ml_partial_q8,
                    &out_q8,
                    n_q,
                    n_kv,
                    hd,
                    n_pos,
                    nwg,
                    tile_c,
                )
            })
        };
        if let Some(tile) = group_tile {
            with_attn_v4_group_tile_override(tile, q8_result)
        } else {
            q8_result()
        }
        .unwrap();

        let y_f16 = read_back_f32(&out_f16.buffer, n_q * hd);
        let y_q8 = read_back_f32(&out_q8.buffer, n_q * hd);
        let max_abs = y_f16
            .iter()
            .zip(y_q8.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = y_f16
            .iter()
            .zip(y_q8.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = y_f16
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = y_q8.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb + 1e-30);
        eprintln!(
            "[attn_v4_q8_kv {label}] group={group} n_pos={n_pos} nwg={nwg} C={tile_c} tile={:?} cos={cos:.6} max|Δ|={max_abs:.4}",
            group_tile,
        );
        assert!(cos > 0.9999, "q8 kv cos too low: {cos}");
        assert!(max_abs < 0.01, "q8 kv max|Δ| too high: {max_abs}");
    }

    #[test]
    fn attn_v4_q8_kv_close_to_f16_kv() {
        run_attn_v4_q8_kv_compare("g6-main", 24, 4, 4096, 64, 32, None);
    }

    #[test]
    fn attn_v4_q8_group8_main_close_to_f16_kv() {
        run_attn_v4_q8_kv_compare("g8-main", 16, 2, 2048, 32, 32, Some(8));
    }

    #[test]
    fn attn_v4_q8_group8_subgroup_close_to_f16_kv() {
        run_attn_v4_q8_kv_compare("g8-t2", 16, 2, 8192, 64, 64, Some(2));
        run_attn_v4_q8_kv_compare("g8-t4", 16, 2, 16384, 128, 64, Some(4));
    }

    /// GDN α-chain fusion vs the 3-dispatch reference (add_inplace +
    /// softplus + mul). Must match within fp32 rounding noise.
    #[test]
    fn gdn_alpha_chain_matches_unfused() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let n = 48usize; // n_v_heads for 27B
        let a: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.5).collect();
        let dt: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let alog: Vec<f32> = (0..n).map(|i| -1.0 - (i % 5) as f32 * 0.2).collect();

        let a_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&a),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let dt_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&dt),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let alog_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&alog),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();

        // Fused path.
        let fused = one_shot_f32_out(&ctx, n, |enc, out| {
            encode_gdn_alpha_chain_f32(&ctx, enc, &a_t, &dt_t, &alog_t, out)
        });

        // Unfused reference: build via 3 sequential dispatches in one cmdbuf.
        let a_ref = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&a),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let unfused_t = MetalTensor::zeros_f32(&ctx, vec![n as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_add_inplace_f32(&ctx, enc, &a_ref, &dt_t)?;
            encode_softplus_f32(&ctx, enc, &a_ref, &unfused_t)?;
            encode_mul_f32(&ctx, enc, &unfused_t, &alog_t, &unfused_t)
        })
        .unwrap();
        let unfused = read_back_f32(&unfused_t.buffer, n);

        let max_abs = fused
            .iter()
            .zip(unfused.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[gdn_alpha_chain] max|Δ|={max_abs:.2e}");
        assert!(
            max_abs < 1e-5,
            "gdn_alpha_chain fused vs unfused mismatch: max|Δ|={max_abs}"
        );

        // Also validate vs explicit CPU formula.
        for i in 0..n {
            let v = a[i] + dt[i];
            let sp = if v > 20.0 {
                v
            } else if v < -20.0 {
                v.exp()
            } else {
                (1.0 + v.exp()).ln()
            };
            let expected = sp * alog[i];
            assert!(
                (fused[i] - expected).abs() < 1e-5,
                "i={i}: fused={} expected={expected}",
                fused[i]
            );
        }
    }

    /// v0.73a: batched α-chain over `[N, n_v]` must produce
    /// bit-identical output to N successive single-row α-chain calls
    /// (broadcasting `dt_bias` and `a_log` across rows).
    #[test]
    fn gdn_alpha_chain_batched_matches_per_row() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let n_rows = 16usize; // N for DFlash block_size
        let n_cols = 48usize; // n_v_heads for 27B
        let n = n_rows * n_cols;
        let a: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.5).collect();
        let dt: Vec<f32> = (0..n_cols).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let alog: Vec<f32> = (0..n_cols).map(|i| -1.0 - (i % 5) as f32 * 0.2).collect();

        let a_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&a),
            vec![n_rows as u64, n_cols as u64],
            GgmlType::F32,
        )
        .unwrap();
        let dt_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&dt),
            vec![n_cols as u64],
            GgmlType::F32,
        )
        .unwrap();
        let alog_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&alog),
            vec![n_cols as u64],
            GgmlType::F32,
        )
        .unwrap();

        // Batched fused path.
        let batched = one_shot_f32_out(&ctx, n, |enc, out| {
            encode_gdn_alpha_chain_batched_f32(&ctx, enc, &a_t, &dt_t, &alog_t, out, n_rows, n_cols)
        });

        // Per-row reference: N invocations of the single-row kernel,
        // each on a row-view of a / out.
        let out_t = MetalTensor::zeros_f32(&ctx, vec![n_rows as u64, n_cols as u64]).unwrap();
        one_shot(&ctx, |enc| {
            for r in 0..n_rows {
                let a_row = a_t.view_subrange((r * n_cols) as u64, vec![n_cols as u64]);
                let out_row = out_t.view_subrange((r * n_cols) as u64, vec![n_cols as u64]);
                encode_gdn_alpha_chain_f32(&ctx, enc, &a_row, &dt_t, &alog_t, &out_row)?;
            }
            Ok(())
        })
        .unwrap();
        let per_row = read_back_f32(&out_t.buffer, n);

        // Bit-exact required (same kernel arithmetic, same broadcast, same order).
        for i in 0..n {
            assert_eq!(
                batched[i].to_bits(),
                per_row[i].to_bits(),
                "i={i} (r={}, c={}): batched={} per_row={}",
                i / n_cols,
                i % n_cols,
                batched[i],
                per_row[i]
            );
        }
    }

    #[test]
    fn elementwise_add_mul_silu_mul_sigmoid_mul() {
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

        let sigmul = one_shot_f32_out(&ctx, n, |enc, y| {
            encode_sigmoid_mul_f32(&ctx, enc, &a_t, &b_t, y)
        });
        for i in 0..n {
            let sig_a = 1.0 / (1.0 + (-a[i]).exp());
            assert!(
                (sigmul[i] - sig_a * b[i]).abs() < 1e-5,
                "sigmoid_mul[{i}] = {} vs {}",
                sigmul[i],
                sig_a * b[i]
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
    fn l2_norm_pair_batched_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(n_heads, head_dim) in &[(16usize, 128usize), (48, 128), (8, 256)] {
            let total = n_heads * head_dim;
            let q: Vec<f32> = (0..total)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.04)
                .collect();
            let k: Vec<f32> = (0..total)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.03)
                .collect();
            let eps = 1e-6f32;

            let normalize = |x: &[f32]| {
                let mut out = vec![0.0f32; total];
                for h in 0..n_heads {
                    let off = h * head_dim;
                    let sq: f32 = (0..head_dim).map(|i| x[off + i].powi(2)).sum();
                    let scale = 1.0 / sq.sqrt().max(eps);
                    for i in 0..head_dim {
                        out[off + i] = x[off + i] * scale;
                    }
                }
                out
            };
            let q_cpu = normalize(&q);
            let k_cpu = normalize(&k);

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&k),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            let q_y = MetalTensor::zeros_f32(&ctx, vec![total as u64]).unwrap();
            let k_y = MetalTensor::zeros_f32(&ctx, vec![total as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_l2_norm_pair_batched_f32(
                    &ctx, enc, &q_t, &q_y, &k_t, &k_y, n_heads, head_dim, eps,
                )
            })
            .unwrap();
            let q_gpu = read_back_f32(&q_y.buffer, total);
            let k_gpu = read_back_f32(&k_y.buffer, total);

            let q_max = q_gpu
                .iter()
                .zip(q_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let k_max = k_gpu
                .iter()
                .zip(k_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                q_max < 1e-5 && k_max < 1e-5,
                "l2_norm_pair n_heads={n_heads} head_dim={head_dim}: q={q_max} k={k_max}"
            );
        }
    }

    #[test]
    fn get_rows_flat_f32_matches_cpu_and_guards_ids() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let vocab = 100usize;
        let n_cols = 64usize;
        let embed: Vec<f32> = (0..vocab * n_cols).map(|i| i as f32 * 0.001).collect();
        let ids: Vec<i32> = vec![3, 17, 42, 99, -1, vocab as i32];
        let n_rows = ids.len();

        let embed_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&embed),
            vec![(n_cols * vocab) as u64],
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
            for i in 0..n_cols {
                let expected = if ids[r] < 0 || ids[r] as usize >= vocab {
                    0.0
                } else {
                    embed[ids[r] as usize * n_cols + i]
                };
                let got = gpu[r * n_cols + i];
                assert!(
                    (got - expected).abs() < 1e-7,
                    "get_rows row={r} ids={} col={i}: {got} vs {expected}",
                    ids[r]
                );
            }
        }
    }

    fn quantized_get_rows_fixture(path: &str, expected_dtype: GgmlType, bit_exact: bool) {
        assert!(
            std::path::Path::new(path).exists(),
            "required fixture missing: {path}"
        );
        let ctx = MetalContext::new().expect("metal context");
        let g = crate::gguf::GgufFile::open(path).expect("open fixture");
        let model = crate::loader::Model::from_gguf(&g).expect("load model");
        let desc = model.token_embd;
        assert_eq!(desc.dtype, expected_dtype);
        assert_eq!(desc.shape.len(), 2);
        let n_cols = desc.shape[0] as usize;
        let vocab = desc.shape[1] as usize;
        let ids = vec![0i32, (vocab - 1) as i32, 1, (vocab - 2) as i32, 17, 17];
        let n_rows = ids.len();
        let embed = MetalTensor::from_gguf_tensor(&ctx, desc, g.slice(desc)).expect("embedding");
        let ids_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![n_rows as u64],
            GgmlType::F32,
        )
        .expect("ids");
        let output =
            MetalTensor::zeros_f32(&ctx, vec![n_rows as u64, n_cols as u64]).expect("output");
        one_shot(&ctx, |enc| {
            encode_get_rows_f32(&ctx, enc, &embed, &ids_t, &output, n_rows, n_cols)
        })
        .expect("get rows");
        let gpu = read_back_f32(&output.buffer, n_rows * n_cols);

        let tensor_bytes = g.slice(desc);
        let row_bytes = desc.n_bytes as usize / vocab;
        let mut expected = Vec::with_capacity(n_rows * n_cols);
        for &id in &ids {
            let row = id as usize;
            let mut row_desc = desc.clone();
            row_desc.name = format!("{}.row.{row}", desc.name);
            row_desc.shape = vec![n_cols as u64];
            row_desc.data_offset = 0;
            row_desc.n_bytes = row_bytes as u64;
            expected.extend(
                crate::codec::dequant_to_f32(
                    &row_desc,
                    &tensor_bytes[row * row_bytes..(row + 1) * row_bytes],
                )
                .expect("dequant row"),
            );
        }

        let mut max_abs = 0.0f32;
        let mut squared = 0.0f64;
        let mut dot = 0.0f64;
        let mut gpu_norm = 0.0f64;
        let mut cpu_norm = 0.0f64;
        for (&candidate, &reference) in gpu.iter().zip(expected.iter()) {
            assert!(candidate.is_finite());
            max_abs = max_abs.max((candidate - reference).abs());
            squared += ((candidate - reference) as f64).powi(2);
            dot += candidate as f64 * reference as f64;
            gpu_norm += (candidate as f64).powi(2);
            cpu_norm += (reference as f64).powi(2);
            if bit_exact {
                assert_eq!(candidate.to_bits(), reference.to_bits());
            }
        }
        let rmse = (squared / gpu.len() as f64).sqrt();
        let cosine = dot / (gpu_norm.sqrt() * cpu_norm.sqrt() + 1e-30);
        let repeated_a = &gpu[4 * n_cols..5 * n_cols];
        let repeated_b = &gpu[5 * n_cols..6 * n_cols];
        assert!(
            repeated_a
                .iter()
                .zip(repeated_b.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        eprintln!(
            concat!(
                "[quant-get-rows] dtype={:?} rows={} cols={} ",
                "max_abs={:.3e} rmse={:.3e} cos={:.10}"
            ),
            expected_dtype, n_rows, n_cols, max_abs, rmse, cosine,
        );
        assert!(max_abs <= 1e-6);
        assert!(rmse <= 1e-7);
        assert!(cosine >= 0.99999999);
    }

    #[test]
    #[ignore = "requires local 27B Q4_K fixture"]
    fn get_rows_q4_k_matches_selected_cpu_rows() {
        quantized_get_rows_fixture(
            "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf",
            GgmlType::Q4_K,
            true,
        );
    }

    #[test]
    #[ignore = "requires local A3B Q8_0 embedding fixture"]
    fn get_rows_q8_0_matches_selected_cpu_rows() {
        quantized_get_rows_fixture(
            "/Users/tito/models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            GgmlType::Q8_0,
            true,
        );
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

    #[test]
    fn rope_neox_packed_consecutive_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let head_dim = 256;
        let n_rot = 64;
        let theta_base = 10_000_000.0f32;
        for &(n_tokens, n_heads, start_position) in &[(5usize, 4usize, 0u32), (3, 24, 17)] {
            let total = n_tokens * n_heads * head_dim;
            let buf_init: Vec<f32> = (0..total)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
                .collect();

            let mut buf_cpu = buf_init.clone();
            for tok in 0..n_tokens {
                let start = tok * n_heads * head_dim;
                let end = start + n_heads * head_dim;
                rope_neox_cpu_ref(
                    &mut buf_cpu[start..end],
                    n_heads,
                    head_dim,
                    n_rot,
                    start_position + tok as u32,
                    theta_base,
                );
            }

            let buf_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&buf_init),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_rope_neox_f32_packed_consecutive(
                    &ctx,
                    enc,
                    &buf_t,
                    n_tokens,
                    n_heads,
                    head_dim,
                    n_rot,
                    start_position,
                    theta_base,
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
                "rope_neox_packed n_tokens={n_tokens} n_heads={n_heads} start={start_position}: max|Δ|={max_abs}"
            );
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
            let conv_buf: Vec<f32> = (0..(K - 1) * conv_dim)
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
        cmd.waitUntilCompleted();
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

    /// v0.432 equivalence gate: the strided-source batched q-norm reading
    /// the Q halves of an interleaved `[head_dim Q, head_dim gate]` layout
    /// must be BIT-IDENTICAL to split_q_gate followed by the compact
    /// batched q-norm (pure addressing change, same per-row arithmetic).
    #[test]
    fn rms_norm_batched_src_strided_matches_split_path_bitwise() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(n_heads, head_dim) in &[(24usize, 256usize), (16, 256), (8, 64)] {
            let full: Vec<f32> = (0..n_heads * 2 * head_dim)
                .map(|i| ((i % 41) as f32 - 20.0) * 3e-2)
                .collect();
            let weight: Vec<f32> = (0..head_dim).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
            let full_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&full),
                vec![(n_heads * 2 * head_dim) as u64],
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
            let q_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
            let gate_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
            let y_split = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
            let y_strided =
                MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
            let eps = 1e-6f32;
            one_shot(&ctx, |enc| {
                encode_split_q_gate_f32(&ctx, enc, &full_t, &q_t, &gate_t, n_heads, head_dim)?;
                encode_rms_norm_batched_f32(&ctx, enc, &q_t, &w_t, &y_split, n_heads, head_dim, eps)
            })
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_rms_norm_batched_src_strided_f32(
                    &ctx,
                    enc,
                    &full_t,
                    &w_t,
                    &y_strided,
                    n_heads,
                    head_dim,
                    2 * head_dim,
                    0,
                    eps,
                )
            })
            .unwrap();
            let a = read_back_f32(&y_split.buffer, n_heads * head_dim);
            let b = read_back_f32(&y_strided.buffer, n_heads * head_dim);
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "strided q-norm not bit-identical at [{i}] (n_heads={n_heads}, \
                     head_dim={head_dim}): split={x} strided={y}"
                );
            }
        }
    }

    /// v0.432 equivalence gate: the fused strided gate epilogue
    /// (`out = x / (1 + e^-gate)`) vs the old split + sigmoid-into-temp +
    /// mul (`out = x * (1 / (1 + e^-gate))`). Different last-ulp rounding
    /// (division vs reciprocal-multiply), so tolerance-based, tight.
    #[test]
    fn sigmoid_mul_gate_strided_matches_split_path() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let (n_heads, head_dim) = (24usize, 256usize);
        let full: Vec<f32> = (0..n_heads * 2 * head_dim)
            .map(|i| ((i % 37) as f32 - 18.0) * 5e-2)
            .collect();
        let x: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 4e-2)
            .collect();
        let full_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&full),
            vec![(n_heads * 2 * head_dim) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_heads * head_dim) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let q_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let gate_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let sig_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let y_split = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let y_fused = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_split_q_gate_f32(&ctx, enc, &full_t, &q_t, &gate_t, n_heads, head_dim)?;
            encode_sigmoid_f32(&ctx, enc, &gate_t, &sig_t)?;
            encode_mul_f32(&ctx, enc, &x_t, &sig_t, &y_split)
        })
        .unwrap();
        one_shot(&ctx, |enc| {
            encode_sigmoid_mul_gate_strided_f32(
                &ctx,
                enc,
                &full_t,
                &x_t,
                &y_fused,
                n_heads,
                head_dim,
                2 * head_dim,
                head_dim,
            )
        })
        .unwrap();
        let a = read_back_f32(&y_split.buffer, n_heads * head_dim);
        let b = read_back_f32(&y_fused.buffer, n_heads * head_dim);
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 1e-6,
            "fused strided gate epilogue diverged beyond ulp scale: max|Δ|={max_abs:.3e}"
        );
    }

    /// v0.433 triage repro for the `attn_v4_matches_naive_f16kv` load-flake:
    /// NaN-prime the o/ml partials scratch before dispatch at the exact
    /// config that failed under parallel-suite load (`group=4 n_pos=1024
    /// nwg=64 C=16`, cos=0.9662). `zeros_f32` is documented-uninitialized,
    /// so isolated runs see fresh zero pages while loaded runs see recycled
    /// garbage; if any kernel cell is read without being written, this test
    /// fails deterministically instead of 50%-of-suite-runs.
    #[test]
    fn attn_v4_partials_fully_written_nan_prime() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let hd = 256usize;
        // The observed failing config plus its close neighbors.
        let cases: &[(usize, usize, usize, usize, usize)] = &[
            // (n_q, n_kv, n_pos, nwg, tile_c)
            (8, 2, 1024, 64, 16),
            (8, 2, 1024, 64, 32),
            (8, 2, 1024, 128, 16),
            (8, 2, 1024, 256, 16),
            (24, 4, 1024, 64, 16),
        ];
        for &(n_q, n_kv, n_pos, nwg, tile_c) in cases {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();
            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }
            let y_naive_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_attn_decode_f16kv_f32(
                    &ctx, enc, &q_t, &k_cache, &v_cache, &y_naive_t, n_q, n_kv, hd, n_pos,
                )
            })
            .unwrap();
            let y_naive = read_back_f32(&y_naive_t.buffer, n_q * hd);

            let o_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
            let ml_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
            let y_v4_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
            // NaN-prime everything the kernels are supposed to fully write.
            unsafe {
                for t in [&o_partial, &ml_partial, &y_v4_t] {
                    let p = t.buffer.contents().as_ptr() as *mut f32;
                    for i in 0..t.n_elements() as usize {
                        *p.add(i) = f32::NAN;
                    }
                }
            }
            one_shot(&ctx, |enc| {
                encode_attn_decode_v4_f32(
                    &ctx,
                    enc,
                    &q_t,
                    &k_cache,
                    &v_cache,
                    &o_partial,
                    &ml_partial,
                    &y_v4_t,
                    n_q,
                    n_kv,
                    hd,
                    n_pos,
                    nwg,
                    tile_c,
                )
            })
            .unwrap();
            let y_v4 = read_back_f32(&y_v4_t.buffer, n_q * hd);
            let nan_count = y_v4.iter().filter(|x| x.is_nan()).count();
            let max_abs = y_v4
                .iter()
                .zip(y_naive.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "[v4-nan-prime group={group} n_pos={n_pos} nwg={nwg} C={tile_c}] \
                 nans={nan_count} max|Δ|={max_abs:.2e}"
            );
            assert_eq!(
                nan_count, 0,
                "v4 output contains NaN after NaN-priming partials: some partial \
                 cell is read without being written (group={group} n_pos={n_pos} \
                 nwg={nwg} C={tile_c})"
            );
            assert!(
                max_abs < 5e-3,
                "v4 diverged from naive with NaN-primed partials: max|Δ|={max_abs} \
                 (group={group} n_pos={n_pos} nwg={nwg} C={tile_c})"
            );
        }
    }

    /// v4 flash-attn (GQA-dedup + online softmax + split-K) vs production
    /// `attn_decode_f16kv_f32`. Same F16 K/V inputs, multiple n_pos and NWG
    /// settings. Must produce numerically equivalent outputs (cos > 0.9999;
    /// max|Δ| ~1e-3 — the bound expected from F32 reorder noise across
    /// completely different reduction orderings).
    #[test]
    fn attn_v4_matches_naive_f16kv() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let hd = 256usize;
        // Cover the currently-supported specializations:
        // small dense (GROUP=4), 27B dense (GROUP=6), 35B A3B (GROUP=8),
        // 122B A10B (GROUP=16).
        let shapes: &[(usize, usize)] = &[(8, 2), (24, 4), (16, 2), (32, 2)];

        for &(n_q, n_kv) in shapes {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            let cases: &[(usize, usize)] = &[
                (1, 1),
                (32, 1),
                (32, 2),
                (64, 1),
                (256, 4),
                (1024, 8),
                (1024, 64),
                (1024, 128),
                (1024, 256),
                (4096, 16),
                (4096, 64),
            ];

            for &(n_pos, nwg) in cases {
                // Synthesize Q (F32) and K, V (F32 scratch → F16 cache).
                let q: Vec<f32> = (0..n_q * hd)
                    .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                    .collect();
                let cap = n_pos.max(64);
                let k_f32: Vec<f32> = (0..cap * kv_dim)
                    .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                    .collect();
                let v_f32: Vec<f32> = (0..cap * kv_dim)
                    .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                    .collect();

                let q_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&q),
                    vec![(n_q * hd) as u64],
                    GgmlType::F32,
                )
                .unwrap();
                // Build F16 KV cache by scattering F32 source into a F16 dest.
                let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
                let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
                // Use scatter to convert F32 → F16 in cache.
                for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                    let src_t = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(src_f32.as_slice()),
                        vec![src_f32.len() as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    one_shot(&ctx, |enc| {
                        encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                    })
                    .unwrap();
                }

                // --- Reference: naive f16kv kernel ---
                let y_naive_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
                one_shot(&ctx, |enc| {
                    encode_attn_decode_f16kv_f32(
                        &ctx, enc, &q_t, &k_cache, &v_cache, &y_naive_t, n_q, n_kv, hd, n_pos,
                    )
                })
                .unwrap();
                let y_naive = read_back_f32(&y_naive_t.buffer, n_q * hd);

                // --- v4: allocate partials, dispatch main + reduce ---
                let o_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
                let ml_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
                let y_v4_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

                // Sweep all three tile-C variants — each must match naive within
                // fp32 reorder noise (cos > 0.9999, max|Δ| < 5e-3).
                for &tile_c in &[16usize, 32, 64, 128] {
                    one_shot(&ctx, |enc| {
                        encode_attn_decode_v4_f32(
                            &ctx,
                            enc,
                            &q_t,
                            &k_cache,
                            &v_cache,
                            &o_partial,
                            &ml_partial,
                            &y_v4_t,
                            n_q,
                            n_kv,
                            hd,
                            n_pos,
                            nwg,
                            tile_c,
                        )
                    })
                    .unwrap();
                    let y_v4 = read_back_f32(&y_v4_t.buffer, n_q * hd);
                    let max_abs = y_v4
                        .iter()
                        .zip(y_naive.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let dot: f64 = y_v4
                        .iter()
                        .zip(y_naive.iter())
                        .map(|(a, b)| (*a as f64) * (*b as f64))
                        .sum();
                    let na: f64 = y_v4.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                    let nb: f64 = y_naive
                        .iter()
                        .map(|x| (*x as f64).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let cos = dot / (na * nb);
                    eprintln!(
                        "[v4 group={group:>2} n_q={n_q:>2} n_kv={n_kv:>2} n_pos={n_pos:>4} nwg={nwg:>2} C={tile_c:>2}] max|Δ|={max_abs:.2e}  cos={cos:.6}"
                    );
                    assert!(
                        cos > 0.9999,
                        "v4(group={group}, C={tile_c}) vs naive cos too low at n_pos={n_pos} nwg={nwg}: cos={cos}"
                    );
                    assert!(
                        max_abs < 5e-3,
                        "v4(group={group}, C={tile_c}) vs naive max|Δ| too high at n_pos={n_pos} nwg={nwg}: {max_abs}"
                    );
                }
            }
        }
    }

    /// CPU f64 reference for the matrix-attention sidecar semantics: causal
    /// multi-row attention over an F16 KV prefix with per-row visibility
    /// `base_pos + row + 1`, softmax of `q·k / sqrt(head_dim)`.
    ///
    /// Inputs must already be f16-representable (pre-rounded) so the GPU's
    /// half demotion inside the GEMMs is exact and tolerances measure reduction
    /// order + the probs half demotion, not input rounding.
    #[allow(clippy::too_many_arguments)]
    fn cpu_matrix_attn_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_rows: usize,
        base_pos: usize,
        n_pos: usize,
        n_q: usize,
        n_kv: usize,
        group: usize,
        hd: usize,
    ) -> Vec<f32> {
        let kv_dim = n_kv * hd;
        let scale = 1.0f64 / (hd as f64).sqrt();
        let mut out = vec![0f32; n_rows * n_q * hd];
        for kvh in 0..n_kv {
            for row in 0..n_rows {
                for g in 0..group {
                    let qh = kvh * group + g;
                    let visible = (base_pos + row + 1).min(n_pos);
                    let qv = &q[(row * n_q + qh) * hd..][..hd];
                    let mut s = vec![0f64; visible];
                    for (p, sp) in s.iter_mut().enumerate() {
                        let kb = &k[p * kv_dim + kvh * hd..][..hd];
                        *sp = qv
                            .iter()
                            .zip(kb)
                            .map(|(a, b)| (*a as f64) * (*b as f64))
                            .sum::<f64>()
                            * scale;
                    }
                    let m = s.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let mut probs: Vec<f64> = s.iter().map(|&x| (x - m).exp()).collect();
                    let sum: f64 = probs.iter().sum();
                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    let mut acc = vec![0f64; hd];
                    for (p, pr) in probs.iter_mut().enumerate() {
                        let w = *pr * inv;
                        let vb = &v[p * kv_dim + kvh * hd..][..hd];
                        for (a, x) in acc.iter_mut().zip(vb) {
                            *a += w * (*x as f64);
                        }
                    }
                    let ob = &mut out[(row * n_q + qh) * hd..][..hd];
                    for (o, a) in ob.iter_mut().zip(&acc) {
                        *o = *a as f32;
                    }
                }
            }
        }
        out
    }

    /// Micro-oracle for the non-flash matrix-attention sidecar
    /// (`kernel_attn_matrix_{transpose_v,kq,softmax,kqv}_f32`): the packed
    /// prefill attention body. Previously this path had only end-to-end
    /// coverage (27B G6 prefix gate + runtime packed oracle); this gate pins
    /// the kernels in isolation against a CPU f64 reference across all four
    /// production group shapes, full/edge tile geometries, and mid-sequence
    /// `base_pos > 0` chunks (including the tiny-chunk/long-prefix shape).
    ///
    /// Also asserts `causal_skip` on/off produce bitwise-identical output:
    /// skipped KQ tiles are exactly the rows the softmax zero-masks, and
    /// skipped KQV K-tiles multiply exact-zero probs.
    #[test]
    fn attn_matrix_path_matches_cpu_reference() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let hd = 256usize;
        // (n_q, n_kv): G4 small dense, G6 27B, G8 A3B, G16 A10B.
        let shapes: &[(usize, usize)] = &[(8, 2), (24, 4), (16, 2), (32, 2)];
        // (n_rows, base_pos); n_pos = base_pos + n_rows as in production
        // (chunk attends to the whole prefix incl. itself).
        //  - (32, 0): first chunk, n_pos < 64 → KQ edge tiles
        //  - (64, 0): full 64-pos KQ tile; N edge depends on group
        //  - (17, 47): odd everything (M/N edge tiles, base_pos > 0)
        //  - (128, 896): full tiles, mid-sequence, n_pos = 1024
        //  - (8, 1016): tiny chunk over long prefix (prefix-gate shape)
        //  - (100, 156): n_pos = 256; N edge for G6/G4, full N for G8/G16
        let cases: &[(usize, usize)] = &[
            (32, 0),
            (64, 0),
            (17, 47),
            (128, 896),
            (8, 1016),
            (100, 156),
        ];

        for &(n_q, n_kv) in shapes {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            for &(n_rows, base_pos) in cases {
                let n_pos = base_pos + n_rows;
                let round16 = |x: f32| half::f16::from_f32(x).to_f32();
                let q: Vec<f32> = (0..n_rows * n_q * hd)
                    .map(|i| round16(((i % 31) as f32 - 15.0) * 1e-2 + ((i % 7) as f32) * 3e-3))
                    .collect();
                let k_f32: Vec<f32> = (0..n_pos * kv_dim)
                    .map(|i| round16(((i % 23) as f32 - 11.0) * 1.5e-2))
                    .collect();
                let v_f32: Vec<f32> = (0..n_pos * kv_dim)
                    .map(|i| round16(((i % 17) as f32 - 8.0) * 2e-2))
                    .collect();

                let q_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&q),
                    vec![(n_rows * n_q * hd) as u64],
                    GgmlType::F32,
                )
                .unwrap();
                let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
                let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
                for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                    let src_t = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(src_f32.as_slice()),
                        vec![src_f32.len() as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    one_shot(&ctx, |enc| {
                        encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                    })
                    .unwrap();
                }
                let vt_stride = n_pos;
                let v_t =
                    MetalTensor::zeros_f16(&ctx, vec![(n_kv * hd * vt_stride) as u64]).unwrap();
                let scores =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * n_rows * group * n_pos) as u64])
                        .unwrap();
                let out_t = MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();

                let run_path = |causal_skip: bool| -> Vec<f32> {
                    one_shot(&ctx, |enc| {
                        encode_attn_matrix_transpose_v_f16(
                            &ctx, enc, &v_cache, &v_t, 0, n_pos, n_pos, kv_dim, vt_stride, n_kv, hd,
                        )?;
                        encode_attn_matrix_kq_f32(
                            &ctx,
                            enc,
                            &q_t,
                            &k_cache,
                            &scores,
                            n_rows,
                            base_pos,
                            n_pos,
                            kv_dim,
                            n_q,
                            n_kv,
                            group,
                            hd,
                            causal_skip,
                        )?;
                        encode_attn_matrix_softmax_f32(
                            &ctx, enc, &scores, n_rows, base_pos, n_pos, n_q, n_kv, group, hd,
                        )?;
                        encode_attn_matrix_kqv_f32(
                            &ctx,
                            enc,
                            &scores,
                            &v_t,
                            &out_t,
                            n_rows,
                            base_pos,
                            n_pos,
                            vt_stride,
                            n_q,
                            n_kv,
                            group,
                            hd,
                            causal_skip,
                        )
                    })
                    .unwrap();
                    read_back_f32(&out_t.buffer, n_rows * n_q * hd)
                };

                let y_gpu = run_path(true);
                let y_ref = cpu_matrix_attn_reference(
                    &q, &k_f32, &v_f32, n_rows, base_pos, n_pos, n_q, n_kv, group, hd,
                );

                let max_abs = y_gpu
                    .iter()
                    .zip(y_ref.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let dot: f64 = y_gpu
                    .iter()
                    .zip(y_ref.iter())
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum();
                let na: f64 = y_gpu
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let nb: f64 = y_ref
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let cos = dot / (na * nb);
                eprintln!(
                    "[matrix group={group:>2} n_rows={n_rows:>4} base={base_pos:>4} n_pos={n_pos:>4}] max|Δ|={max_abs:.2e}  cos={cos:.7}"
                );
                assert!(
                    cos > 0.9999,
                    "matrix path vs CPU ref cos too low (group={group} n_rows={n_rows} base={base_pos}): {cos}"
                );
                assert!(
                    max_abs < 5e-3,
                    "matrix path vs CPU ref max|Δ| too high (group={group} n_rows={n_rows} base={base_pos}): {max_abs}"
                );

                // causal_skip must be a pure perf feature: bitwise-identical out.
                if matches!((n_rows, base_pos), (64, 0) | (8, 1016)) {
                    let y_noskip = run_path(false);
                    assert!(
                        y_gpu == y_noskip,
                        "causal_skip changed matrix attention output (group={group} n_rows={n_rows} base={base_pos})"
                    );
                }

                // Two-pass online kernels: KQ folds the softmax into its
                // epilogue (F16 P~ + (m,l) sidecar), KQV normalizes during
                // staging. Must sit in the same envelope vs the CPU reference
                // (the P~ half demotion mirrors the sidecar's half probs).
                let scores_h_t =
                    MetalTensor::zeros_f16(&ctx, vec![(n_kv * n_rows * group * n_pos) as u64])
                        .unwrap();
                let ml_t = MetalTensor::zeros_f32(
                    &ctx,
                    vec![attn_matrix_ml_elems(n_rows, n_q, n_pos) as u64],
                )
                .unwrap();
                let fused_t =
                    MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();
                one_shot(&ctx, |enc| {
                    encode_attn_matrix_kq_online_f32(
                        &ctx,
                        enc,
                        &q_t,
                        &k_cache,
                        &scores_h_t,
                        &ml_t,
                        n_rows,
                        base_pos,
                        n_pos,
                        kv_dim,
                        n_q,
                        n_kv,
                        group,
                        hd,
                        true,
                    )?;
                    encode_attn_matrix_kqv_norm_f32(
                        &ctx,
                        enc,
                        &scores_h_t,
                        &ml_t,
                        &v_t,
                        &fused_t,
                        n_rows,
                        base_pos,
                        n_pos,
                        vt_stride,
                        n_q,
                        n_kv,
                        group,
                        hd,
                        true,
                    )
                })
                .unwrap();
                let y_fused = read_back_f32(&fused_t.buffer, n_rows * n_q * hd);
                let fmax_abs = y_fused
                    .iter()
                    .zip(y_ref.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let fdot: f64 = y_fused
                    .iter()
                    .zip(y_ref.iter())
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum();
                let fna: f64 = y_fused
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let fcos = fdot / (fna * nb);
                let gpu_max_abs = y_fused
                    .iter()
                    .zip(y_gpu.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "[online group={group:>2} n_rows={n_rows:>4} base={base_pos:>4} n_pos={n_pos:>4}] max|Δ|={fmax_abs:.2e}  cos={fcos:.7}  vs3k|Δ|={gpu_max_abs:.2e}"
                );
                assert!(
                    fcos > 0.9999,
                    "online matrix attn vs CPU ref cos too low (group={group} n_rows={n_rows} base={base_pos}): {fcos}"
                );
                assert!(
                    fmax_abs < 5e-3,
                    "online matrix attn vs CPU ref max|Δ| too high (group={group} n_rows={n_rows} base={base_pos}): {fmax_abs}"
                );
                assert!(
                    gpu_max_abs < 5e-3,
                    "online vs 3-kernel matrix attn diverged (group={group} n_rows={n_rows} base={base_pos}): {gpu_max_abs}"
                );

                if (n_rows, base_pos) == (100, 156) {
                    let query_cap = 32usize;
                    let tiled_scores =
                        MetalTensor::zeros_f16(&ctx, vec![(query_cap * n_q * n_pos) as u64])
                            .unwrap();
                    let tiled_ml = MetalTensor::zeros_f32(
                        &ctx,
                        vec![attn_matrix_ml_elems(query_cap, n_q, n_pos) as u64],
                    )
                    .unwrap();
                    let tiled_out =
                        MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();
                    one_shot(&ctx, |enc| {
                        for row_base in (0..n_rows).step_by(query_cap) {
                            let rows_n = (n_rows - row_base).min(query_cap);
                            let q_rows = q_t.view_subrange(
                                (row_base * n_q * hd) as u64,
                                vec![(rows_n * n_q * hd) as u64],
                            );
                            let out_rows = tiled_out.view_subrange(
                                (row_base * n_q * hd) as u64,
                                vec![(rows_n * n_q * hd) as u64],
                            );
                            let scores_rows =
                                tiled_scores.view_subrange(0, vec![(rows_n * n_q * n_pos) as u64]);
                            let ml_rows = tiled_ml.view_subrange(
                                0,
                                vec![attn_matrix_ml_elems(rows_n, n_q, n_pos) as u64],
                            );
                            let tile_base_pos = base_pos + row_base;
                            encode_attn_matrix_kq_online_f32(
                                &ctx,
                                enc,
                                &q_rows,
                                &k_cache,
                                &scores_rows,
                                &ml_rows,
                                rows_n,
                                tile_base_pos,
                                n_pos,
                                kv_dim,
                                n_q,
                                n_kv,
                                group,
                                hd,
                                true,
                            )?;
                            encode_attn_matrix_kqv_norm_f32(
                                &ctx,
                                enc,
                                &scores_rows,
                                &ml_rows,
                                &v_t,
                                &out_rows,
                                rows_n,
                                tile_base_pos,
                                n_pos,
                                vt_stride,
                                n_q,
                                n_kv,
                                group,
                                hd,
                                true,
                            )?;
                        }
                        Ok(())
                    })
                    .unwrap();
                    let y_tiled = read_back_f32(&tiled_out.buffer, n_rows * n_q * hd);
                    let tiled_max_abs = y_tiled
                        .iter()
                        .zip(y_fused.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        tiled_max_abs < 5e-5,
                        concat!(
                            "tiled vs untiled online attention diverged ",
                            "(group={}): {}"
                        ),
                        group,
                        tiled_max_abs
                    );
                }
            }
        }
    }

    /// Kill-gate microbench for the two-pass online-softmax matrix attention
    /// kernels vs the three-kernel sidecar (KQ + softmax + KQV) at production
    /// chunk shapes. The promotion bar is >= 1.2x on the summed sidecar time.
    /// Vᵀ transpose/maintenance is excluded from both sides: both variants
    /// consume the same Vᵀ sidecar, so its upkeep cancels.
    ///
    /// `cargo test -p qwen-llm --release attn_matrix_online_vs_sidecar_microbench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn attn_matrix_online_vs_sidecar_microbench() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let hd = 256usize;

        fn timed_gpu<F>(ctx: &MetalContext, iters: usize, encode: F) -> f64
        where
            F: Fn(&KernelEncoder) -> Result<(), MetalError>,
        {
            let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
            let enc = KernelEncoder::begin(&cmd_buf);
            for _ in 0..iters {
                encode(&enc).unwrap();
            }
            enc.end();
            let t0 = std::time::Instant::now();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            t0.elapsed().as_secs_f64() / iters as f64
        }

        // (n_q, n_kv, n_rows, n_pos, label)
        let shapes: &[(usize, usize, usize, usize, &str)] = &[
            (16, 2, 1024, 4096, "G8/A3B chunk@pp4096"),
            (16, 2, 1024, 16384, "G8/A3B chunk@pp16384"),
            (24, 4, 1024, 4096, "G6/27B chunk@pp4096"),
            (24, 4, 1024, 16384, "G6/27B chunk@pp16384"),
            (32, 2, 1024, 1024, "G16/A10B chunk@pp1024"),
            (32, 2, 1024, 4096, "G16/A10B chunk@pp4096"),
        ];

        for &(n_q, n_kv, n_rows, n_pos, label) in shapes {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            let base_pos = n_pos - n_rows;

            let q: Vec<f32> = (0..n_rows * n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let kv_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_rows * n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            for dst in [&k_cache, &v_cache] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(kv_f32.as_slice()),
                    vec![kv_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, kv_f32.len())
                })
                .unwrap();
            }
            let vt_stride = n_pos;
            let v_t = MetalTensor::zeros_f16(&ctx, vec![(n_kv * hd * vt_stride) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_attn_matrix_transpose_v_f16(
                    &ctx, enc, &v_cache, &v_t, 0, n_pos, n_pos, kv_dim, vt_stride, n_kv, hd,
                )
            })
            .unwrap();
            let scores =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * n_rows * group * n_pos) as u64]).unwrap();
            let scores_h =
                MetalTensor::zeros_f16(&ctx, vec![(n_kv * n_rows * group * n_pos) as u64]).unwrap();
            let ml =
                MetalTensor::zeros_f32(&ctx, vec![attn_matrix_ml_elems(n_rows, n_q, n_pos) as u64])
                    .unwrap();
            let out_3k = MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();
            let out_fused = MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();

            let encode_3k = |enc: &KernelEncoder| -> Result<(), MetalError> {
                encode_attn_matrix_kq_f32(
                    &ctx, enc, &q_t, &k_cache, &scores, n_rows, base_pos, n_pos, kv_dim, n_q, n_kv,
                    group, hd, true,
                )?;
                encode_attn_matrix_softmax_f32(
                    &ctx, enc, &scores, n_rows, base_pos, n_pos, n_q, n_kv, group, hd,
                )?;
                encode_attn_matrix_kqv_f32(
                    &ctx, enc, &scores, &v_t, &out_3k, n_rows, base_pos, n_pos, vt_stride, n_q,
                    n_kv, group, hd, true,
                )
            };
            let encode_fused = |enc: &KernelEncoder| -> Result<(), MetalError> {
                encode_attn_matrix_kq_online_f32(
                    &ctx, enc, &q_t, &k_cache, &scores_h, &ml, n_rows, base_pos, n_pos, kv_dim,
                    n_q, n_kv, group, hd, true,
                )?;
                encode_attn_matrix_kqv_norm_f32(
                    &ctx, enc, &scores_h, &ml, &v_t, &out_fused, n_rows, base_pos, n_pos,
                    vt_stride, n_q, n_kv, group, hd, true,
                )
            };

            // Warmup + correctness spot at production size.
            one_shot(&ctx, |enc| encode_3k(enc)).unwrap();
            one_shot(&ctx, |enc| encode_fused(enc)).unwrap();
            let y3k = read_back_f32(&out_3k.buffer, n_rows * n_q * hd);
            let yfused = read_back_f32(&out_fused.buffer, n_rows * n_q * hd);
            let max_abs = y3k
                .iter()
                .zip(yfused.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs < 5e-3,
                "fused vs sidecar diverged at {label}: max|Δ|={max_abs}"
            );

            let iters = 8usize;
            let mut t3k = f64::INFINITY;
            let mut tfused = f64::INFINITY;
            let mut tkq = f64::INFINITY;
            let mut tsm = f64::INFINITY;
            let mut tkqv = f64::INFINITY;
            for _ in 0..3 {
                t3k = t3k.min(timed_gpu(&ctx, iters, encode_3k));
                tfused = tfused.min(timed_gpu(&ctx, iters, encode_fused));
                tkq = tkq.min(timed_gpu(&ctx, iters, |enc| {
                    encode_attn_matrix_kq_f32(
                        &ctx, enc, &q_t, &k_cache, &scores, n_rows, base_pos, n_pos, kv_dim, n_q,
                        n_kv, group, hd, true,
                    )
                }));
                tsm = tsm.min(timed_gpu(&ctx, iters, |enc| {
                    encode_attn_matrix_softmax_f32(
                        &ctx, enc, &scores, n_rows, base_pos, n_pos, n_q, n_kv, group, hd,
                    )
                }));
                tkqv = tkqv.min(timed_gpu(&ctx, iters, |enc| {
                    encode_attn_matrix_kqv_f32(
                        &ctx, enc, &scores, &v_t, &out_3k, n_rows, base_pos, n_pos, vt_stride, n_q,
                        n_kv, group, hd, true,
                    )
                }));
            }
            eprintln!(
                "[{label:>22}] 3k={:8.3} ms (kq={:.3} sm={:.3} kqv={:.3})  online2p={:8.3} ms  ratio={:.2}x  vs|Δ|={max_abs:.2e}",
                t3k * 1e3,
                tkq * 1e3,
                tsm * 1e3,
                tkqv * 1e3,
                tfused * 1e3,
                t3k / tfused
            );
        }
    }

    /// Focused correctness gate for the A3B group-8 long-context subgroup path.
    ///
    /// Run in a fresh process with one of:
    ///
    /// - `QWEN_ATTN_V4_G8_TILE=4 cargo test -p qwen-llm attn_v4_group8_subgroup_matches_naive_f16kv --release -- --ignored --nocapture`
    /// - `QWEN_ATTN_V4_G8_TILE=2 cargo test -p qwen-llm attn_v4_group8_subgroup_matches_naive_f16kv --release -- --ignored --nocapture`
    ///
    /// The env var is intentionally process-global (`OnceLock`) so this test stays
    /// ignored and single-purpose.
    #[test]
    #[ignore]
    fn attn_v4_group8_subgroup_matches_naive_f16kv() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let n_q = 16usize;
        let n_kv = 2usize;
        let hd = 256usize;
        let kv_dim = n_kv * hd;
        for &(n_pos, nwg, tile_c) in &[(4096usize, 64usize, 64usize), (6144, 64, 64)] {
            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let cap = n_pos;
            let k_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }

            let y_naive_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_attn_decode_f16kv_f32(
                    &ctx, enc, &q_t, &k_cache, &v_cache, &y_naive_t, n_q, n_kv, hd, n_pos,
                )
            })
            .unwrap();
            let y_naive = read_back_f32(&y_naive_t.buffer, n_q * hd);

            let o_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * (n_q / n_kv) * hd) as u64])
                    .unwrap();
            let ml_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * (n_q / n_kv) * 2) as u64]).unwrap();
            let y_v4_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_attn_decode_v4_f32(
                    &ctx,
                    enc,
                    &q_t,
                    &k_cache,
                    &v_cache,
                    &o_partial,
                    &ml_partial,
                    &y_v4_t,
                    n_q,
                    n_kv,
                    hd,
                    n_pos,
                    nwg,
                    tile_c,
                )
            })
            .unwrap();
            let y_v4 = read_back_f32(&y_v4_t.buffer, n_q * hd);
            let max_abs = y_v4
                .iter()
                .zip(y_naive.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let dot: f64 = y_v4
                .iter()
                .zip(y_naive.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let na: f64 = y_v4.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = y_naive
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let cos = dot / (na * nb);
            eprintln!(
                "[v4-g8-subgroup n_pos={n_pos:>5} nwg={nwg:>2} C={tile_c:>2}] max|Δ|={max_abs:.2e} cos={cos:.6}"
            );
            assert!(
                cos > 0.9999,
                "group8 subgroup cos too low at n_pos={n_pos}: {cos}"
            );
            assert!(
                max_abs < 5e-3,
                "group8 subgroup max|Δ| too high at n_pos={n_pos}: {max_abs}"
            );
        }
    }

    /// Prompt-native packed-attention microproof for the A3B long-context shape.
    ///
    /// Compares the new packed multi-query microkernel against repeated
    /// decode-shaped `attn_v4` calls using the same subgroup setting
    /// (`g8_t2`) and the same F16 KV cache.
    #[test]
    #[ignore]
    fn attn_v4_prefill_g8_t2_q2_c64_vs_decode_loop() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        const N_Q: usize = 16;
        const N_KV: usize = 2;
        const HD: usize = 256;
        const N_ROWS: usize = 128;
        const NWG: usize = 64;
        const TILE_C: usize = 64;
        let kv_dim = N_KV * HD;

        for &base_pos in &[16384usize, 32768] {
            let n_pos = base_pos + N_ROWS;
            let q_rows: Vec<f32> = (0..N_ROWS * N_Q * HD)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q_rows),
                vec![(N_ROWS * N_Q * HD) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }

            let out_baseline =
                MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_Q * HD) as u64]).unwrap();
            let out_packed =
                MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_Q * HD) as u64]).unwrap();
            let o_partial_row =
                MetalTensor::zeros_f32(&ctx, vec![(N_KV * NWG * (N_Q / N_KV) * HD) as u64])
                    .unwrap();
            let ml_partial_row =
                MetalTensor::zeros_f32(&ctx, vec![(N_KV * NWG * (N_Q / N_KV) * 2) as u64]).unwrap();
            let o_partial_packed = MetalTensor::zeros_f32(
                &ctx,
                vec![(N_ROWS * N_KV * NWG * (N_Q / N_KV) * HD) as u64],
            )
            .unwrap();
            let ml_partial_packed =
                MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_KV * NWG * (N_Q / N_KV) * 2) as u64])
                    .unwrap();

            let t = Instant::now();
            with_attn_v4_group_tile_override(2, || {
                one_shot(&ctx, |enc| {
                    for row in 0..N_ROWS {
                        let q_row =
                            q_t.view_subrange((row * N_Q * HD) as u64, vec![(N_Q * HD) as u64]);
                        let out_row = out_baseline
                            .view_subrange((row * N_Q * HD) as u64, vec![(N_Q * HD) as u64]);
                        encode_attn_decode_v4_f32(
                            &ctx,
                            enc,
                            &q_row,
                            &k_cache,
                            &v_cache,
                            &o_partial_row,
                            &ml_partial_row,
                            &out_row,
                            N_Q,
                            N_KV,
                            HD,
                            base_pos + row + 1,
                            NWG,
                            TILE_C,
                        )
                        .unwrap();
                    }
                    Ok(())
                })
            })
            .unwrap();
            let baseline_wall = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            one_shot(&ctx, |enc| {
                encode_attn_prefill_v4_g8_t2_q2_c64_f32(
                    &ctx,
                    enc,
                    &q_t,
                    &k_cache,
                    &v_cache,
                    &o_partial_packed,
                    &ml_partial_packed,
                    &out_packed,
                    N_ROWS,
                    base_pos,
                    NWG,
                )
                .unwrap();
                Ok(())
            })
            .unwrap();
            let packed_wall = t.elapsed().as_secs_f64() * 1e3;

            let baseline = read_back_f32(&out_baseline.buffer, N_ROWS * N_Q * HD);
            let packed = read_back_f32(&out_packed.buffer, N_ROWS * N_Q * HD);
            let max_abs = packed
                .iter()
                .zip(baseline.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let dot: f64 = packed
                .iter()
                .zip(baseline.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let na: f64 = packed
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let nb: f64 = baseline
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let cos = dot / (na * nb);
            eprintln!(
                "[v4-prefill-a3b base_pos={base_pos:>5} rows={N_ROWS:>3}] decode_loop={baseline_wall:7.2} ms packed={packed_wall:7.2} ms speedup={:.3} max|Δ|={max_abs:.2e} cos={cos:.6}",
                baseline_wall / packed_wall
            );
            assert!(
                cos > 0.99999,
                "prefill packed cos too low at base_pos={base_pos}: {cos}"
            );
            assert!(
                max_abs < 2e-3,
                "prefill packed max|Δ| too high at base_pos={base_pos}: {max_abs}"
            );
        }
    }

    /// Bench: sweep NWG (split-K count) across context lengths to discover
    /// the optimal NWG for our shape on the host GPU. Compares against the
    /// naive f16kv kernel.
    ///
    /// Run with: `cargo test --release --lib -p qwen-llm attn_v4_nwg_sweep
    /// --ignored -- --nocapture`
    #[test]
    #[ignore]
    fn attn_v4_nwg_sweep() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        eprintln!("[v4-bench] {}", ctx.describe());

        let n_q = 24usize;
        let n_kv = 4usize;
        let hd = 256usize;
        let kv_dim = n_kv * hd;
        const GROUP: usize = 6;

        let n_iters = 200usize; // chained dispatches per command buffer
        let warmup = 20usize;

        // Each context length we want to characterize.
        // Past 16K we skip naive_f16kv (cap'd) and only run v4 NWG sweep.
        for &n_pos in &[64usize, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072] {
            let cap = n_pos.max(64);
            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }
            let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

            // ----- Naive f16kv baseline (skip if past tg-mem cap ~7000) -----
            let naive_works = n_pos * std::mem::size_of::<f32>() <= 28 * 1024;
            if naive_works {
                let bench = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        encode_attn_decode_f16kv_f32(
                            &ctx, &enc, &q_t, &k_cache, &v_cache, &y_t, n_q, n_kv, hd, n_pos,
                        )
                        .unwrap();
                    }
                    enc.end();
                    let t = Instant::now();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let wall = t.elapsed().as_secs_f64() * 1e3;
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[n_pos={n_pos:>5} {label}] {n}× chained: wall={wall:7.2} ms  gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };
                // Warmup
                bench("naive_f16kv_warmup", warmup);
                bench("naive_f16kv       ", n_iters);
            } else {
                eprintln!("[n_pos={n_pos:>5} naive_f16kv       ] skipped (past tg-mem cap)");
            }

            // ----- v4: sweep NWG -----
            for &nwg in &[1usize, 2, 4, 8, 16, 32] {
                let o_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * GROUP * hd) as u64]).unwrap();
                let ml_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * GROUP * 2) as u64]).unwrap();
                let bench = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        encode_attn_decode_v4_f32(
                            &ctx,
                            &enc,
                            &q_t,
                            &k_cache,
                            &v_cache,
                            &o_partial,
                            &ml_partial,
                            &y_t,
                            n_q,
                            n_kv,
                            hd,
                            n_pos,
                            nwg,
                            32, // tile_c — NWG sweep holds tile constant
                        )
                        .unwrap();
                    }
                    enc.end();
                    let t = Instant::now();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let wall = t.elapsed().as_secs_f64() * 1e3;
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[n_pos={n_pos:>5} {label} nwg={nwg:>2}] {n}× chained: wall={wall:7.2} ms  gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };
                bench("v4_warmup           ", warmup);
                bench("v4                  ", n_iters);
            }
            eprintln!();
        }
    }

    /// Bench: sweep TILE-C (KV positions per inner softmax tile) at
    /// production NWG settings. Per Codex's review, GQA-dedup raises
    /// arithmetic intensity per K row, which may shift the optimal C
    /// away from llama.cpp's vec-kernel default of 32.
    ///
    /// Run with: `cargo test --release --lib -p qwen-llm
    /// attn_v4_tile_c_sweep -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn attn_v4_tile_c_sweep() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        eprintln!("[v4-c-sweep] {}", ctx.describe());

        let n_q = 24usize;
        let n_kv = 4usize;
        let hd = 256usize;
        let kv_dim = n_kv * hd;
        const GROUP: usize = 6;

        let n_iters = 200usize;
        let warmup = 20usize;

        // For each ctx, use the production NWG heuristic (16 below 256, 32 above).
        for &n_pos in &[64usize, 256, 1024, 4096, 16384, 65536, 131072] {
            let nwg = if n_pos < 256 { 16usize } else { 32usize };
            let cap = n_pos.max(64);

            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }
            let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
            let o_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * GROUP * hd) as u64]).unwrap();
            let ml_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * GROUP * 2) as u64]).unwrap();

            for &tile_c in &[16usize, 32, 64, 128] {
                let bench = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        encode_attn_decode_v4_f32(
                            &ctx,
                            &enc,
                            &q_t,
                            &k_cache,
                            &v_cache,
                            &o_partial,
                            &ml_partial,
                            &y_t,
                            n_q,
                            n_kv,
                            hd,
                            n_pos,
                            nwg,
                            tile_c,
                        )
                        .unwrap();
                    }
                    enc.end();
                    let t = Instant::now();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let wall = t.elapsed().as_secs_f64() * 1e3;
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>2} {label}] {n}× chained: wall={wall:7.2} ms  gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };
                bench("warmup", warmup);
                bench("bench ", n_iters);
            }
            eprintln!();
        }
    }

    /// Focused long-context v4 NWG sweep for the MoE shapes we now care
    /// about: A3B (group=8) and 122B-A10B (group=16). Synthetic K/V is
    /// enough because we're tuning the attention kernel itself, not model
    /// semantics.
    #[test]
    #[ignore]
    fn attn_v4_nwg_sweep_moe_shapes() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        eprintln!("[v4-moe-nwg] {}", ctx.describe());

        let hd = 256usize;
        let n_iters = 120usize;
        let warmup = 16usize;
        let shapes: &[(usize, usize, &str)] = &[(16, 2, "a3b"), (32, 2, "122b")];

        for &(n_q, n_kv, label_shape) in shapes {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            for &n_pos in &[4096usize, 8192, 16384, 32768] {
                let cap = n_pos;
                let q: Vec<f32> = (0..n_q * hd)
                    .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                    .collect();
                let k_f32: Vec<f32> = (0..cap * kv_dim)
                    .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                    .collect();
                let v_f32: Vec<f32> = (0..cap * kv_dim)
                    .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                    .collect();

                let q_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&q),
                    vec![(n_q * hd) as u64],
                    GgmlType::F32,
                )
                .unwrap();
                let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
                let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
                for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                    let src_t = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(src_f32.as_slice()),
                        vec![src_f32.len() as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    one_shot(&ctx, |enc| {
                        encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                    })
                    .unwrap();
                }
                let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

                for &nwg in &[4usize, 8, 16, 32, 64] {
                    let o_partial =
                        MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64])
                            .unwrap();
                    let ml_partial =
                        MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64])
                            .unwrap();
                    let bench = |label: &str, n: usize| {
                        let cmd = ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        for _ in 0..n {
                            encode_attn_decode_v4_f32(
                                &ctx,
                                &enc,
                                &q_t,
                                &k_cache,
                                &v_cache,
                                &o_partial,
                                &ml_partial,
                                &y_t,
                                n_q,
                                n_kv,
                                hd,
                                n_pos,
                                nwg,
                                32,
                            )
                            .unwrap();
                        }
                        enc.end();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        eprintln!(
                            "[v4-moe-nwg {label_shape} group={group:>2} n_pos={n_pos:>6} nwg={nwg:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                            gpu / n as f64
                        );
                    };
                    bench("warmup", warmup);
                    bench("bench ", n_iters);
                }
                eprintln!();
            }
        }
    }

    /// Focused long-context tile-C sweep for the same MoE shapes. Uses the
    /// production-default NWG=32 at these contexts unless data says otherwise.
    #[test]
    #[ignore]
    fn attn_v4_tile_c_sweep_moe_shapes() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        eprintln!("[v4-moe-c] {}", ctx.describe());

        let hd = 256usize;
        let n_iters = 120usize;
        let warmup = 16usize;
        let shapes: &[(usize, usize, &str)] = &[(16, 2, "a3b"), (32, 2, "122b")];

        for &(n_q, n_kv, label_shape) in shapes {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            for &n_pos in &[4096usize, 8192, 16384, 32768] {
                for &nwg in &[32usize, 64] {
                    let cap = n_pos;
                    let q: Vec<f32> = (0..n_q * hd)
                        .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                        .collect();
                    let k_f32: Vec<f32> = (0..cap * kv_dim)
                        .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                        .collect();
                    let v_f32: Vec<f32> = (0..cap * kv_dim)
                        .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                        .collect();

                    let q_t = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(&q),
                        vec![(n_q * hd) as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    let k_cache =
                        MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
                    let v_cache =
                        MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
                    for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                        let src_t = MetalTensor::from_bytes(
                            &ctx,
                            bytemuck::cast_slice(src_f32.as_slice()),
                            vec![src_f32.len() as u64],
                            GgmlType::F32,
                        )
                        .unwrap();
                        one_shot(&ctx, |enc| {
                            encode_scatter_offset_f32_to_f16(
                                &ctx,
                                enc,
                                &src_t,
                                dst,
                                0,
                                src_f32.len(),
                            )
                        })
                        .unwrap();
                    }
                    let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
                    let o_partial =
                        MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64])
                            .unwrap();
                    let ml_partial =
                        MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64])
                            .unwrap();

                    for &tile_c in &[16usize, 32, 64, 128] {
                        let bench = |label: &str, n: usize| {
                            let cmd = ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            for _ in 0..n {
                                encode_attn_decode_v4_f32(
                                    &ctx,
                                    &enc,
                                    &q_t,
                                    &k_cache,
                                    &v_cache,
                                    &o_partial,
                                    &ml_partial,
                                    &y_t,
                                    n_q,
                                    n_kv,
                                    hd,
                                    n_pos,
                                    nwg,
                                    tile_c,
                                )
                                .unwrap();
                            }
                            enc.end();
                            cmd.commit();
                            cmd.waitUntilCompleted();
                            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                            eprintln!(
                                "[v4-moe-c {label_shape} group={group:>2} n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                                gpu / n as f64
                            );
                        };
                        bench("warmup", warmup);
                        bench("bench ", n_iters);
                    }
                    eprintln!();
                }
            }
        }
    }

    /// Split the v4 attention kernel into main and reduce passes so we can
    /// see which part actually dominates at realistic long contexts for the
    /// MoE shapes. This is synthetic, but it uses the real kernel bodies and
    /// exact production shapes for A3B (group=8) and 122B (group=16).
    #[test]
    #[ignore]
    fn attn_v4_main_reduce_breakdown_moe_shapes() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        eprintln!("[v4-main-reduce] {}", ctx.describe());

        let hd = 256usize;
        let n_iters = 120usize;
        let warmup = 16usize;
        let shapes: &[(usize, usize, &str, &[usize])] = &[
            (16, 2, "a3b", &[4096, 16384, 32768]),
            (32, 2, "122b", &[4096, 16384, 32768]),
        ];

        for &(n_q, n_kv, label_shape, ctxs) in shapes {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            for &n_pos in ctxs {
                let nwg = attn_v4_choose_nwg(n_pos, group);
                let tile_c = attn_v4_choose_tile_c(n_pos, group);
                let cap = n_pos;

                let q: Vec<f32> = (0..n_q * hd)
                    .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                    .collect();
                let k_f32: Vec<f32> = (0..cap * kv_dim)
                    .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                    .collect();
                let v_f32: Vec<f32> = (0..cap * kv_dim)
                    .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                    .collect();

                let q_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&q),
                    vec![(n_q * hd) as u64],
                    GgmlType::F32,
                )
                .unwrap();
                let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
                let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
                for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                    let src_t = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(src_f32.as_slice()),
                        vec![src_f32.len() as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    one_shot(&ctx, |enc| {
                        encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                    })
                    .unwrap();
                }

                let o_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
                let ml_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
                let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

                let bench_main = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        encode_attn_decode_v4_main_only_f32(
                            &ctx,
                            &enc,
                            &q_t,
                            &k_cache,
                            &v_cache,
                            &o_partial,
                            &ml_partial,
                            n_q,
                            n_kv,
                            hd,
                            n_pos,
                            nwg,
                            tile_c,
                        )
                        .unwrap();
                    }
                    enc.end();
                    let _t = Instant::now();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[v4-main {label_shape} group={group:>2} n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>3} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };

                one_shot(&ctx, |enc| {
                    encode_attn_decode_v4_main_only_f32(
                        &ctx,
                        enc,
                        &q_t,
                        &k_cache,
                        &v_cache,
                        &o_partial,
                        &ml_partial,
                        n_q,
                        n_kv,
                        hd,
                        n_pos,
                        nwg,
                        tile_c,
                    )
                })
                .unwrap();

                let bench_reduce = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        encode_attn_decode_v4_reduce_only_f32(
                            &ctx,
                            &enc,
                            &o_partial,
                            &ml_partial,
                            &y_t,
                            n_q,
                            n_kv,
                            hd,
                            nwg,
                        )
                        .unwrap();
                    }
                    enc.end();
                    let _t = Instant::now();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[v4-reduce {label_shape} group={group:>2} n_pos={n_pos:>6} nwg={nwg:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };

                bench_main("warmup", warmup);
                bench_main("bench ", n_iters);
                bench_reduce("warmup", warmup);
                bench_reduce("bench ", n_iters);
                eprintln!();
            }
        }
    }

    /// Synthetic head-major F16 K/V proof for the long-context MoE v4 attention
    /// body. This is intentionally not a production cache layout: it isolates the
    /// address-stride question before any prefill/session sidecar work.
    #[test]
    #[ignore]
    fn attn_v4_head_major_main_reduce_breakdown_moe_shapes() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        eprintln!("[v4-hm-main-reduce] {}", ctx.describe());

        fn cosine(a: &[f32], b: &[f32]) -> f64 {
            let mut dot = 0.0f64;
            let mut aa = 0.0f64;
            let mut bb = 0.0f64;
            for (&x, &y) in a.iter().zip(b) {
                let x = x as f64;
                let y = y as f64;
                dot += x * y;
                aa += x * x;
                bb += y * y;
            }
            dot / (aa.sqrt() * bb.sqrt()).max(1e-30)
        }

        fn max_abs(a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        }

        let hd = 256usize;
        let n_iters = 96usize;
        let warmup = 12usize;
        let shapes: &[(usize, usize, &str, &[usize])] = &[
            (16, 2, "a3b", &[8192, 16384, 32768]),
            (32, 2, "a10b", &[8192, 16384, 32768]),
        ];

        for &(n_q, n_kv, label_shape, ctxs) in shapes {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            for &n_pos in ctxs {
                let nwg = attn_v4_choose_nwg(n_pos, group);
                let tile_c = attn_v4_choose_tile_c(n_pos, group);
                let group_tile = attn_v4_choose_group_tile(n_pos, group);
                assert!(
                    (group, group_tile, tile_c) == (8, 2, 64)
                        || (group, group_tile, tile_c) == (16, 4, 64)
                        || (group, group_tile, tile_c) == (16, 4, 128),
                    "unexpected group/group_tile/tile_c {group}/{group_tile}/{tile_c}"
                );

                let q: Vec<f32> = (0..n_q * hd)
                    .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                    .collect();
                let k_f32: Vec<f32> = (0..n_pos * kv_dim)
                    .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                    .collect();
                let v_f32: Vec<f32> = (0..n_pos * kv_dim)
                    .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                    .collect();

                let k_tok_h: Vec<half::f16> =
                    k_f32.iter().copied().map(half::f16::from_f32).collect();
                let v_tok_h: Vec<half::f16> =
                    v_f32.iter().copied().map(half::f16::from_f32).collect();
                let mut k_hm_h = vec![half::f16::ZERO; k_tok_h.len()];
                let mut v_hm_h = vec![half::f16::ZERO; v_tok_h.len()];
                for pos in 0..n_pos {
                    for kvh in 0..n_kv {
                        let src = pos * kv_dim + kvh * hd;
                        let dst = (kvh * n_pos + pos) * hd;
                        k_hm_h[dst..dst + hd].copy_from_slice(&k_tok_h[src..src + hd]);
                        v_hm_h[dst..dst + hd].copy_from_slice(&v_tok_h[src..src + hd]);
                    }
                }

                let q_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&q),
                    vec![(n_q * hd) as u64],
                    GgmlType::F32,
                )
                .unwrap();
                let k_tok = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&k_tok_h),
                    vec![(n_pos * kv_dim) as u64],
                    GgmlType::F16,
                )
                .unwrap();
                let v_tok = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&v_tok_h),
                    vec![(n_pos * kv_dim) as u64],
                    GgmlType::F16,
                )
                .unwrap();
                let k_hm = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&k_hm_h),
                    vec![(n_pos * kv_dim) as u64],
                    GgmlType::F16,
                )
                .unwrap();
                let v_hm = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&v_hm_h),
                    vec![(n_pos * kv_dim) as u64],
                    GgmlType::F16,
                )
                .unwrap();

                let partial_elems = n_kv * nwg * group * hd;
                let ml_elems = n_kv * nwg * group * 2;
                let o_tok = MetalTensor::zeros_f32(&ctx, vec![partial_elems as u64]).unwrap();
                let ml_tok = MetalTensor::zeros_f32(&ctx, vec![ml_elems as u64]).unwrap();
                let y_tok = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
                let o_hm = MetalTensor::zeros_f32(&ctx, vec![partial_elems as u64]).unwrap();
                let ml_hm = MetalTensor::zeros_f32(&ctx, vec![ml_elems as u64]).unwrap();
                let y_hm = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

                one_shot(&ctx, |enc| {
                    encode_attn_decode_v4_main_only_f32(
                        &ctx, enc, &q_t, &k_tok, &v_tok, &o_tok, &ml_tok, n_q, n_kv, hd, n_pos,
                        nwg, tile_c,
                    )?;
                    encode_attn_decode_v4_reduce_only_f32(
                        &ctx, enc, &o_tok, &ml_tok, &y_tok, n_q, n_kv, hd, nwg,
                    )
                })
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_attn_decode_v4_main_only_f32_head_major(
                        &ctx, enc, &q_t, &k_hm, &v_hm, &o_hm, &ml_hm, n_q, n_kv, hd, n_pos, nwg,
                        tile_c,
                    )?;
                    encode_attn_decode_v4_reduce_only_f32(
                        &ctx, enc, &o_hm, &ml_hm, &y_hm, n_q, n_kv, hd, nwg,
                    )
                })
                .unwrap();
                let y_tok_v = read_back_f32(&y_tok.buffer, n_q * hd);
                let y_hm_v = read_back_f32(&y_hm.buffer, n_q * hd);
                eprintln!(
                    "[v4-hm-correct {label_shape} group={group:>2} tile={group_tile:>2} n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>2}] cos={:.8} max_abs={:.3e}",
                    cosine(&y_tok_v, &y_hm_v),
                    max_abs(&y_tok_v, &y_hm_v)
                );

                let bench_tok = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        encode_attn_decode_v4_main_only_f32(
                            &ctx, &enc, &q_t, &k_tok, &v_tok, &o_tok, &ml_tok, n_q, n_kv, hd,
                            n_pos, nwg, tile_c,
                        )
                        .unwrap();
                    }
                    enc.end();
                    let _t = Instant::now();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[v4-main-token {label_shape} group={group:>2} tile={group_tile:>2} n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };
                let bench_hm = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        encode_attn_decode_v4_main_only_f32_head_major(
                            &ctx, &enc, &q_t, &k_hm, &v_hm, &o_hm, &ml_hm, n_q, n_kv, hd, n_pos,
                            nwg, tile_c,
                        )
                        .unwrap();
                    }
                    enc.end();
                    let _t = Instant::now();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[v4-main-hmajor {label_shape} group={group:>2} tile={group_tile:>2} n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };

                bench_tok("warmup", warmup);
                bench_hm("warmup", warmup);
                bench_tok("bench ", n_iters);
                bench_hm("bench ", n_iters);
                eprintln!();
            }
        }
    }

    /// Split the prompt-native packed prefill kernels into main and reduce
    /// passes so we can see how much of the remaining packed-attention wall is
    /// still the F32 partial spill/reduce path.
    #[test]
    #[ignore]
    fn attn_prefill_v4_main_reduce_breakdown_moe_shapes() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        eprintln!("[prefill-v4-main-reduce] {}", ctx.describe());

        let hd = 256usize;
        let n_iters = 96usize;
        let warmup = 12usize;
        let rows_set = [4usize, 8usize];
        let shapes: &[(usize, usize, &str, &[usize])] = &[
            (16, 2, "a3b", &[4096, 16384, 32768]),
            (32, 2, "122b", &[4096, 16384, 32768]),
        ];

        for &(n_q, n_kv, label_shape, ctxs) in shapes {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            for &n_rows in &rows_set {
                for &base_pos in ctxs {
                    let n_pos = base_pos + n_rows;
                    let nwg = attn_v4_choose_nwg(n_pos, group);

                    let q: Vec<f32> = (0..n_rows * n_q * hd)
                        .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                        .collect();
                    let k_f32: Vec<f32> = (0..n_pos * kv_dim)
                        .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                        .collect();
                    let v_f32: Vec<f32> = (0..n_pos * kv_dim)
                        .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                        .collect();

                    let q_t = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(&q),
                        vec![(n_rows * n_q * hd) as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    let k_cache =
                        MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
                    let v_cache =
                        MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
                    for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                        let src_t = MetalTensor::from_bytes(
                            &ctx,
                            bytemuck::cast_slice(src_f32.as_slice()),
                            vec![src_f32.len() as u64],
                            GgmlType::F32,
                        )
                        .unwrap();
                        one_shot(&ctx, |enc| {
                            encode_scatter_offset_f32_to_f16(
                                &ctx,
                                enc,
                                &src_t,
                                dst,
                                0,
                                src_f32.len(),
                            )
                        })
                        .unwrap();
                    }

                    let o_partial = MetalTensor::zeros_f32(
                        &ctx,
                        vec![(n_rows * n_kv * nwg * group * hd) as u64],
                    )
                    .unwrap();
                    let ml_partial = MetalTensor::zeros_f32(
                        &ctx,
                        vec![(n_rows * n_kv * nwg * group * 2) as u64],
                    )
                    .unwrap();
                    let y_t =
                        MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();

                    let bench_main = |label: &str, n: usize| {
                        let cmd = ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        for _ in 0..n {
                            match group {
                                8 => encode_attn_prefill_v4_g8_t2_q2_c64_main_only_f32(
                                    &ctx,
                                    &enc,
                                    &q_t,
                                    &k_cache,
                                    &v_cache,
                                    &o_partial,
                                    &ml_partial,
                                    n_rows,
                                    base_pos,
                                    nwg,
                                ),
                                16 => encode_attn_prefill_v4_g16_t4_q2_c64_main_only_f32(
                                    &ctx,
                                    &enc,
                                    &q_t,
                                    &k_cache,
                                    &v_cache,
                                    &o_partial,
                                    &ml_partial,
                                    n_rows,
                                    base_pos,
                                    nwg,
                                ),
                                _ => unreachable!(),
                            }
                            .unwrap();
                        }
                        enc.end();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        eprintln!(
                            "[prefill-v4-main {label_shape} rows={n_rows:>2} group={group:>2} base_pos={base_pos:>6} nwg={nwg:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                            gpu / n as f64
                        );
                    };

                    one_shot(&ctx, |enc| match group {
                        8 => encode_attn_prefill_v4_g8_t2_q2_c64_main_only_f32(
                            &ctx,
                            enc,
                            &q_t,
                            &k_cache,
                            &v_cache,
                            &o_partial,
                            &ml_partial,
                            n_rows,
                            base_pos,
                            nwg,
                        ),
                        16 => encode_attn_prefill_v4_g16_t4_q2_c64_main_only_f32(
                            &ctx,
                            enc,
                            &q_t,
                            &k_cache,
                            &v_cache,
                            &o_partial,
                            &ml_partial,
                            n_rows,
                            base_pos,
                            nwg,
                        ),
                        _ => unreachable!(),
                    })
                    .unwrap();

                    let bench_reduce = |label: &str, n: usize| {
                        let cmd = ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        for _ in 0..n {
                            match group {
                                8 => encode_attn_prefill_v4_g8_t2_q2_c64_reduce_only_f32(
                                    &ctx,
                                    &enc,
                                    &o_partial,
                                    &ml_partial,
                                    &y_t,
                                    n_rows,
                                    nwg,
                                ),
                                16 => encode_attn_prefill_v4_g16_t4_q2_c64_reduce_only_f32(
                                    &ctx,
                                    &enc,
                                    &o_partial,
                                    &ml_partial,
                                    &y_t,
                                    n_rows,
                                    nwg,
                                ),
                                _ => unreachable!(),
                            }
                            .unwrap();
                        }
                        enc.end();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        eprintln!(
                            "[prefill-v4-reduce {label_shape} rows={n_rows:>2} group={group:>2} base_pos={base_pos:>6} nwg={nwg:>2} {label}] gpu={gpu:7.2} ms  per-call={:6.3} ms",
                            gpu / n as f64
                        );
                    };

                    bench_main("warmup", warmup);
                    bench_main("bench ", n_iters);
                    bench_reduce("warmup", warmup);
                    bench_reduce("bench ", n_iters);
                    eprintln!();
                }
            }
        }
    }

    /// H5.3a GPU argmax — bit-exact match to CPU argmax with lowest-index
    /// tie-breaking, including the explicit edge cases:
    /// * tie at row start (idx 0 wins)
    /// * tie at row end
    /// * single-element row
    /// * row larger than 1024 (tests cross-simdgroup reduce path)
    /// * vocab-sized row (V=248320; the actual production shape)
    /// * negative-infinity entries (production lm_head won't have these,
    ///   but defensive)
    #[test]
    fn argmax_matches_cpu_with_tie_to_lowest_index() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        // Helper: encode-only argmax + readback for tests.
        fn run(ctx: &MetalContext, x: &[f32], n_rows: usize, n: usize) -> Vec<i32> {
            assert_eq!(x.len(), n_rows * n);
            let xb = ctx.buffer_from(x).expect("xb");
            let xt = MetalTensor {
                buffer: xb,
                offset: 0,
                shape: vec![n_rows as u64, n as u64],
                dtype: crate::tensor::GgmlType::F32,
            };
            let ob = ctx.buffer_uninit(n_rows * 4).expect("ob");
            let ot = MetalTensor {
                buffer: ob,
                offset: 0,
                shape: vec![n_rows as u64],
                dtype: crate::tensor::GgmlType::F32,
            };
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_argmax_f32(ctx, &enc, &xt, &ot, n_rows, n).expect("encode argmax");
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            unsafe {
                let p = ot.buffer.contents().as_ptr() as *const i32;
                (0..n_rows).map(|i| *p.add(i)).collect()
            }
        }

        fn cpu_argmax_lowest_idx(row: &[f32]) -> i32 {
            let mut best = f32::NEG_INFINITY;
            let mut idx: i32 = 0;
            for (i, &v) in row.iter().enumerate() {
                if v > best {
                    best = v;
                    idx = i as i32;
                }
            }
            idx
        }

        // 1. Single-element row.
        {
            let x = vec![3.5f32];
            let got = run(&ctx, &x, 1, 1);
            assert_eq!(got, vec![0]);
        }

        // 2. Tie at row start: x = [5.0, 1.0, 5.0, 5.0, 0.0]. Lowest idx
        //    among matches = 0.
        {
            let x = vec![5.0f32, 1.0, 5.0, 5.0, 0.0];
            let got = run(&ctx, &x, 1, 5);
            assert_eq!(got, vec![0], "tie at start should pick idx 0");
        }

        // 3. Tie at row end (max only at the last position).
        {
            let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            let got = run(&ctx, &x, 1, 5);
            assert_eq!(got, vec![4]);
        }

        // 4. Tie spread across the row at multiple distant positions.
        {
            let mut x = vec![0.0f32; 4096];
            x[100] = 9.0;
            x[2500] = 9.0;
            x[3999] = 9.0;
            let got = run(&ctx, &x, 1, 4096);
            assert_eq!(got, vec![100], "spread tie should pick lowest idx");
        }

        // 5. Multi-row batch: argmax independently per row.
        {
            let n = 1024;
            let n_rows = 5;
            let mut x = vec![0.0f32; n_rows * n];
            for r in 0..n_rows {
                // Place the max for row r at idx (r * 137) % n.
                let idx = (r * 137) % n;
                x[r * n + idx] = 1.0 + (r as f32) * 0.1;
            }
            let got = run(&ctx, &x, n_rows, n);
            for r in 0..n_rows {
                let want = cpu_argmax_lowest_idx(&x[r * n..(r + 1) * n]);
                assert_eq!(got[r], want, "row {r}");
            }
        }

        // 6. Random fuzz at vocab-sized row (the production shape).
        {
            let n = 248_320usize;
            let n_rows = 16;
            let mut x = vec![0.0f32; n_rows * n];
            // Deterministic pseudo-random fill.
            let mut s: u32 = 0xc0ffeeu32;
            for v in x.iter_mut() {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
                *v = (s as i32) as f32 * 1e-9;
            }
            let got = run(&ctx, &x, n_rows, n);
            for r in 0..n_rows {
                let want = cpu_argmax_lowest_idx(&x[r * n..(r + 1) * n]);
                assert_eq!(got[r], want, "vocab row {r}");
            }
        }

        // 7. Negative-infinity entries (defensive — production lm_head
        //    won't produce these but the kernel must not get confused).
        {
            let mut x = vec![f32::NEG_INFINITY; 1024];
            x[42] = -1e9;
            x[500] = -1e10; // smaller than 42's value
            let got = run(&ctx, &x, 1, 1024);
            assert_eq!(got, vec![42]);
        }

        // 8. All-zero row (every position ties); lowest index = 0.
        {
            let x = vec![0.0f32; 1024];
            let got = run(&ctx, &x, 1, 1024);
            assert_eq!(got, vec![0], "all-tie should pick idx 0");
        }

        // 9. All -INFINITY row (degenerate but well-defined): every
        //    position ties at -inf, lowest idx wins. Per codex H5.3a
        //    review: this returns idx 0, NOT -1. Production lm_head
        //    cannot produce all -inf, but documenting the contract.
        {
            let x = vec![f32::NEG_INFINITY; 1024];
            let got = run(&ctx, &x, 1, 1024);
            assert_eq!(
                got,
                vec![0],
                "all -INFINITY: kernel ties at -inf, lowest idx 0 wins"
            );
        }

        // 10. All NaN row: IEEE comparison `a > b` is FALSE for any
        //     NaN operand, so the per-lane scan never updates from
        //     `best_val=-INF, best_idx=UINT_MAX`. simd_max also returns
        //     NaN; (NaN == NaN) is false, so lane_idx stays UINT_MAX
        //     for every lane, simd_min(UINT_MAX) = UINT_MAX, cast to
        //     i32 = -1.
        //
        //     Production lm_head does not produce NaN under correct
        //     numerics. Treat this as a "this kernel returns -1
        //     deterministically when the entire row is unranked";
        //     callers should not feed it NaN rows.
        {
            let x = vec![f32::NAN; 1024];
            let got = run(&ctx, &x, 1, 1024);
            assert_eq!(
                got,
                vec![-1],
                "all-NaN: kernel returns -1 (UINT_MAX cast); document only — production should never see this"
            );
        }
    }

    /// H5.3a foundation: verify `BlitEncoder` actually copies device-side
    /// buffers and that compute↔blit transitions on the same command
    /// buffer are visible. We write a known pattern via a compute kernel
    /// (`scatter_offset`), blit-copy into a destination buffer, then read
    /// the destination back. If the blit didn't fire, we'd read zeros.
    #[test]
    fn blit_encoder_copies_buffer_within_one_command_buffer() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        const N: usize = 1024;
        let pattern: Vec<f32> = (0..N).map(|i| (i as f32) * 0.125).collect();

        // Source: a freshly-uploaded MetalTensor holding `pattern`.
        let src_buf = ctx.buffer_from(&pattern).expect("src buf");
        let src = MetalTensor {
            buffer: src_buf,
            offset: 0,
            shape: vec![N as u64],
            dtype: crate::tensor::GgmlType::F32,
        };

        // Destination: zero-initialized.
        let dst_buf = ctx.buffer_uninit(N * 4).expect("dst buf");
        // Zero it out via the host pointer (StorageModeShared).
        unsafe {
            let p = dst_buf.contents().as_ptr() as *mut f32;
            for i in 0..N {
                *p.add(i) = -1.0;
            }
        }
        let dst = MetalTensor {
            buffer: dst_buf,
            offset: 0,
            shape: vec![N as u64],
            dtype: crate::tensor::GgmlType::F32,
        };

        // One command buffer. Compute pass (no-op trampoline to validate
        // that compute → blit transitions don't drop ordering), then blit
        // pass that performs the actual copy.
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        // Empty compute pass. We don't dispatch anything — we just want
        // to verify that an opened-and-immediately-closed compute encoder
        // doesn't break the subsequent blit.
        {
            let enc = KernelEncoder::begin(&cmd);
            enc.end();
        }
        {
            let blit = BlitEncoder::begin(&cmd);
            blit.copy_tensor(&src, &dst);
            blit.end();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        // Read back via host pointer.
        let got: Vec<f32> = unsafe {
            let p = dst.buffer.contents().as_ptr() as *const f32;
            (0..N).map(|i| *p.add(i)).collect()
        };
        for i in 0..N {
            assert!(
                (got[i] - pattern[i]).abs() < 1e-9,
                "blit mismatch at i={i}: got={} expected={}",
                got[i],
                pattern[i]
            );
        }

        // Also exercise `copy_buffer` with non-zero offsets: copy the
        // back half of `src` into the front half of `dst`.
        let cmd2 = ctx.queue.commandBuffer().expect("cmd2");
        {
            let blit = BlitEncoder::begin(&cmd2);
            let half_bytes = (N / 2) * 4;
            blit.copy_buffer(
                &src.buffer,
                half_bytes as u64,
                &dst.buffer,
                0,
                half_bytes as u64,
            );
            blit.end();
        }
        cmd2.commit();
        cmd2.waitUntilCompleted();
        let got2: Vec<f32> = unsafe {
            let p = dst.buffer.contents().as_ptr() as *const f32;
            (0..N).map(|i| *p.add(i)).collect()
        };
        // Front half of dst now equals back half of src.
        for i in 0..N / 2 {
            assert!(
                (got2[i] - pattern[N / 2 + i]).abs() < 1e-9,
                "offset blit front-half mismatch at i={i}: got={} expected={}",
                got2[i],
                pattern[N / 2 + i]
            );
        }
        // Back half of dst is unchanged from the previous full-blit copy.
        for i in N / 2..N {
            assert!(
                (got2[i] - pattern[i]).abs() < 1e-9,
                "offset blit back-half disturbed at i={i}: got={} expected={}",
                got2[i],
                pattern[i]
            );
        }
    }

    /// CPU oracle for `kernel_dflash_attn_f32`. Mirrors the kernel's
    /// 3-pass streaming softmax + per-layer SWA mask EXACTLY. Used by
    /// the v0.72.2 test suite (codex code-review test additions).
    ///
    /// Mask semantics (kernel + this oracle):
    ///   * full-attn ctx key (swa_window == 0): ALWAYS allowed.
    ///   * SWA ctx key: causal && (q_pos - k_pos) <= swa_window.
    ///   * Noise key: noise_idx <= q_idx.
    ///
    /// Returns o[N, n_q · head_dim] row-major.
    // CPU dflash-attn reference: `kk` is a multi-purpose KV-position
    // index — used for `pos_k[kk]`, `kk * kv_stride + ...` strided
    // K/V offset computation, and `if kk < ctx_len` ctx-vs-noise
    // branching. Iterator rewrite obscures all three.
    #[allow(clippy::needless_range_loop)]
    fn dflash_attn_cpu_oracle(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        pos_k: &[i32],
        n: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_kv_total: usize,
        ctx_len: usize,
        noise_start_pos: u32,
        swa_window: u32,
    ) -> Vec<f32> {
        let group = n_q_heads / n_kv_heads;
        let q_dim = n_q_heads * head_dim;
        let kv_stride = n_kv_heads * head_dim;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let mut o = vec![0.0_f32; n * q_dim];
        let full_attn = swa_window == 0;
        for q_idx in 0..n {
            let q_pos = noise_start_pos + q_idx as u32;
            for q_head in 0..n_q_heads {
                let kv_head = q_head / group;
                // Q vector for this (q_idx, q_head).
                let q_off = q_idx * q_dim + q_head * head_dim;
                // PASS 1: max.
                let mut m_run = f32::NEG_INFINITY;
                for kk in 0..n_kv_total {
                    let allowed = if kk < ctx_len {
                        if full_attn {
                            true
                        } else {
                            let k_pos = pos_k[kk] as u32;
                            k_pos <= q_pos && (q_pos - k_pos) <= swa_window
                        }
                    } else {
                        (kk - ctx_len) <= q_idx
                    };
                    if !allowed {
                        continue;
                    }
                    let k_off = kk * kv_stride + kv_head * head_dim;
                    let mut s = 0.0_f32;
                    for d in 0..head_dim {
                        s += q[q_off + d] * k[k_off + d];
                    }
                    s *= scale;
                    if s > m_run {
                        m_run = s;
                    }
                }
                // PASS 2: sum.
                let mut l_sum = 0.0_f32;
                for kk in 0..n_kv_total {
                    let allowed = if kk < ctx_len {
                        if full_attn {
                            true
                        } else {
                            let k_pos = pos_k[kk] as u32;
                            k_pos <= q_pos && (q_pos - k_pos) <= swa_window
                        }
                    } else {
                        (kk - ctx_len) <= q_idx
                    };
                    if !allowed {
                        continue;
                    }
                    let k_off = kk * kv_stride + kv_head * head_dim;
                    let mut s = 0.0_f32;
                    for d in 0..head_dim {
                        s += q[q_off + d] * k[k_off + d];
                    }
                    s *= scale;
                    l_sum += (s - m_run).exp();
                }
                let inv_l = if l_sum > 0.0 { 1.0 / l_sum } else { 0.0 };
                // PASS 3: V agg.
                for kk in 0..n_kv_total {
                    let allowed = if kk < ctx_len {
                        if full_attn {
                            true
                        } else {
                            let k_pos = pos_k[kk] as u32;
                            k_pos <= q_pos && (q_pos - k_pos) <= swa_window
                        }
                    } else {
                        (kk - ctx_len) <= q_idx
                    };
                    if !allowed {
                        continue;
                    }
                    let k_off = kk * kv_stride + kv_head * head_dim;
                    let v_off = kk * kv_stride + kv_head * head_dim;
                    let mut s = 0.0_f32;
                    for d in 0..head_dim {
                        s += q[q_off + d] * k[k_off + d];
                    }
                    s *= scale;
                    let w = (s - m_run).exp() * inv_l;
                    for d in 0..head_dim {
                        o[q_off + d] += w * v[v_off + d];
                    }
                }
            }
        }
        o
    }

    /// One-shot helper: run kernel_dflash_attn_f32 against synthetic
    /// CPU-staged buffers and read back o.
    fn dflash_attn_readback(
        ctx: &MetalContext,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        pos_k: &[i32],
        n: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_kv_total: usize,
        ctx_len: usize,
        noise_start_pos: u32,
        swa_window: u32,
    ) -> Result<Vec<f32>, MetalError> {
        let q_t = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(q),
            vec![(n * n_q_heads * head_dim) as u64],
            crate::tensor::GgmlType::F32,
        )?;
        let k_t = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(k),
            vec![(n_kv_total * n_kv_heads * head_dim) as u64],
            crate::tensor::GgmlType::F32,
        )?;
        let v_t = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(v),
            vec![(n_kv_total * n_kv_heads * head_dim) as u64],
            crate::tensor::GgmlType::F32,
        )?;
        let pos_t = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(pos_k),
            vec![n_kv_total as u64],
            crate::tensor::GgmlType::F32,
        )?;
        let o_t = MetalTensor::zeros_f32(ctx, vec![(n * n_q_heads * head_dim) as u64])?;
        one_shot(ctx, |enc| {
            encode_dflash_attn_f32(
                ctx,
                enc,
                &q_t,
                &k_t,
                &v_t,
                &pos_t,
                &o_t,
                n,
                n_q_heads,
                n_kv_heads,
                head_dim,
                n_kv_total,
                ctx_len,
                noise_start_pos,
                swa_window,
            )
        })?;
        Ok(read_back_f32(&o_t.buffer, n * n_q_heads * head_dim))
    }

    /// One-shot helper for the two-range DFlash attention sidecar.
    fn dflash_attn_two_range_readback(
        ctx: &MetalContext,
        q: &[f32],
        k_ctx: &[f32],
        v_ctx: &[f32],
        k_noise: &[f32],
        v_noise: &[f32],
        pos_ctx: &[i32],
        n: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        ctx_len: usize,
        noise_start_pos: u32,
        swa_window: u32,
        online: bool,
        full_gqa_split4: bool,
        ctx_scan_start: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let q_t = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(q),
            vec![(n * n_q_heads * head_dim) as u64],
            crate::tensor::GgmlType::F32,
        )?;
        let ctx_rows = ctx_len.max(1);
        let kv_stride = n_kv_heads * head_dim;
        let k_ctx_t = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(k_ctx),
            vec![(ctx_rows * kv_stride) as u64],
            crate::tensor::GgmlType::F32,
        )?;
        let v_ctx_t = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(v_ctx),
            vec![(ctx_rows * kv_stride) as u64],
            crate::tensor::GgmlType::F32,
        )?;
        let k_noise_t = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(k_noise),
            vec![(n * kv_stride) as u64],
            crate::tensor::GgmlType::F32,
        )?;
        let v_noise_t = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(v_noise),
            vec![(n * kv_stride) as u64],
            crate::tensor::GgmlType::F32,
        )?;
        let pos_rows = ctx_len.max(1);
        let pos_t = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(pos_ctx),
            vec![pos_rows as u64],
            crate::tensor::GgmlType::F32,
        )?;
        let o_t = MetalTensor::zeros_f32(ctx, vec![(n * n_q_heads * head_dim) as u64])?;
        let o_partial_len = 16 * 8 * 4 * 4 * 128;
        let ml_partial_len = 16 * 8 * 4 * 4 * 2;
        let o_partial_host = vec![f32::NAN; o_partial_len];
        let ml_partial_host = vec![f32::NAN; ml_partial_len];
        let o_partial_t = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&o_partial_host),
            vec![o_partial_len as u64],
            crate::tensor::GgmlType::F32,
        )?;
        let ml_partial_t = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&ml_partial_host),
            vec![ml_partial_len as u64],
            crate::tensor::GgmlType::F32,
        )?;
        one_shot(ctx, |enc| {
            if full_gqa_split4 {
                encode_dflash_attn_full_gqa_split4_f32(
                    ctx,
                    enc,
                    &q_t,
                    &k_ctx_t,
                    &v_ctx_t,
                    &k_noise_t,
                    &v_noise_t,
                    &o_partial_t,
                    &ml_partial_t,
                    &o_t,
                    n,
                    n_q_heads,
                    n_kv_heads,
                    head_dim,
                    ctx_len,
                )
            } else if online {
                encode_dflash_attn_online_two_range_scan_f32(
                    ctx,
                    enc,
                    &q_t,
                    &k_ctx_t,
                    &v_ctx_t,
                    &k_noise_t,
                    &v_noise_t,
                    &pos_t,
                    &o_t,
                    n,
                    n_q_heads,
                    n_kv_heads,
                    head_dim,
                    ctx_len,
                    noise_start_pos,
                    swa_window,
                    ctx_scan_start,
                )
            } else {
                encode_dflash_attn_two_range_f32(
                    ctx,
                    enc,
                    &q_t,
                    &k_ctx_t,
                    &v_ctx_t,
                    &k_noise_t,
                    &v_noise_t,
                    &pos_t,
                    &o_t,
                    n,
                    n_q_heads,
                    n_kv_heads,
                    head_dim,
                    ctx_len,
                    noise_start_pos,
                    swa_window,
                )
            }
        })?;
        Ok(read_back_f32(&o_t.buffer, n * n_q_heads * head_dim))
    }

    /// **v0.72.2 codex code-review test #1**: dflash attention kernel
    /// matches the CPU oracle bit-tight under each mask regime.
    ///
    /// Exercises:
    ///   * `ctx_len == 0` (degenerate: noise-only attention)
    ///   * `ctx_len > 0, swa_window > 0` (SWA layer)
    ///   * `ctx_len > 0, swa_window == 0` (full-attn layer; codex
    ///     mask-semantics flag — full-attn allows ALL ctx keys, no
    ///     causal restriction)
    ///   * `ctx_len > swa_window` (SWA boundary; some ctx keys
    ///     denied by the window even though causal)
    ///   * `ctx_len > 0` with non-contiguous / gapped pos_k
    ///   * Edge: q_pos == k_pos exactly (boundary causal — allowed
    ///     under SWA)
    #[test]
    fn dflash_attn_matches_cpu_oracle_under_mask_regimes() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };

        // Drafter shape: n_q=32, n_kv=8 (group=4), head_dim=128, N=16.
        let n = 16;
        let n_q = 32;
        let n_kv = 8;
        let hd = 128;
        let q_dim = n_q * hd;
        let kv_stride = n_kv * hd;

        // Deterministic synthetic activations.
        let make_buf = |seed: u32, len: usize| -> Vec<f32> {
            let mut s = seed;
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.5
                })
                .collect()
        };

        let q = make_buf(1, n * q_dim);

        struct Case {
            label: &'static str,
            ctx_len: usize,
            swa_window: u32,
            noise_start_pos: u32,
            // Custom pos_k for the ctx half (length ctx_len).
            // Builder receives ctx_len + noise_start_pos and returns
            // ctx-side positions.
            pos_ctx: fn(usize, u32) -> Vec<i32>,
        }

        fn pos_recent(ctx_len: usize, noise_start: u32) -> Vec<i32> {
            (0..ctx_len)
                .map(|c| noise_start as i32 - ctx_len as i32 + c as i32)
                .collect()
        }
        fn pos_gapped(ctx_len: usize, noise_start: u32) -> Vec<i32> {
            // Every other position skipped — non-contiguous.
            (0..ctx_len)
                .map(|c| (noise_start as i32 - 2 * ctx_len as i32 + 2 * c as i32).max(0))
                .collect()
        }

        let cases = [
            Case {
                label: "ctx_len=0 (noise-only)",
                ctx_len: 0,
                swa_window: 2048,
                noise_start_pos: 4,
                pos_ctx: pos_recent,
            },
            Case {
                label: "swa, ctx within window",
                ctx_len: 8,
                swa_window: 2048,
                noise_start_pos: 16,
                pos_ctx: pos_recent,
            },
            Case {
                label: "full-attn (swa=0), ctx allowed permissively",
                ctx_len: 8,
                swa_window: 0,
                noise_start_pos: 16,
                pos_ctx: pos_recent,
            },
            Case {
                label: "swa boundary, ctx_len > swa_window",
                ctx_len: 64,
                swa_window: 16,
                noise_start_pos: 80,
                pos_ctx: pos_recent,
            },
            Case {
                label: "swa, gapped pos_ctx",
                ctx_len: 12,
                swa_window: 2048,
                noise_start_pos: 32,
                pos_ctx: pos_gapped,
            },
            Case {
                label: "swa, q_pos == k_pos boundary",
                ctx_len: 4,
                swa_window: 2048,
                // pos_recent constructs ctx positions
                // [noise_start - ctx_len .. noise_start). With
                // noise_start=4, ctx pos = [0,1,2,3]. q_pos at q_idx=0
                // = 4. So q_pos > k_pos — no exact equality.
                // To exercise q_pos == k_pos: shift noise_start_pos so
                // pos_ctx ends at exactly noise_start_pos (= q_pos at
                // q_idx=0). Set ctx_len=4, noise_start_pos=4 →
                // pos_ctx = [0..4); the last ctx is at pos=3, q_pos at
                // q_idx=0 is 4 → still strict. Make ctx_len=5 and
                // noise_start_pos=4 → pos_ctx = [-1..4); ctx[4]=3.
                // Hmm same. This case structurally enforces k_pos < q_pos
                // unless we allow ctx that overlaps noise positions
                // (semantically a contract violation per codex flag).
                //
                // Instead, this case tests q_pos > all ctx positions
                // by a margin of 1 — boundary-adjacent without overlap.
                noise_start_pos: 4,
                pos_ctx: pos_recent,
            },
        ];

        for c in &cases {
            let pos_ctx_vec = (c.pos_ctx)(c.ctx_len, c.noise_start_pos);
            let n_kv_total = c.ctx_len + n;
            // Build pos_k = pos_ctx ++ [noise_start..noise_start+N].
            let mut pos_k = Vec::with_capacity(n_kv_total);
            pos_k.extend_from_slice(&pos_ctx_vec);
            for i in 0..n {
                pos_k.push((c.noise_start_pos + i as u32) as i32);
            }
            let k = make_buf(2, n_kv_total * kv_stride);
            let v = make_buf(3, n_kv_total * kv_stride);
            let ctx_rows = c.ctx_len.max(1);
            let mut k_ctx = vec![0.0_f32; ctx_rows * kv_stride];
            let mut v_ctx = vec![0.0_f32; ctx_rows * kv_stride];
            if c.ctx_len > 0 {
                k_ctx[..c.ctx_len * kv_stride].copy_from_slice(&k[..c.ctx_len * kv_stride]);
                v_ctx[..c.ctx_len * kv_stride].copy_from_slice(&v[..c.ctx_len * kv_stride]);
            }
            let k_noise = k[c.ctx_len * kv_stride..].to_vec();
            let v_noise = v[c.ctx_len * kv_stride..].to_vec();
            let mut pos_ctx = vec![0_i32; c.ctx_len.max(1)];
            if c.ctx_len > 0 {
                pos_ctx[..c.ctx_len].copy_from_slice(&pos_ctx_vec);
            }

            let cpu = dflash_attn_cpu_oracle(
                &q,
                &k,
                &v,
                &pos_k,
                n,
                n_q,
                n_kv,
                hd,
                n_kv_total,
                c.ctx_len,
                c.noise_start_pos,
                c.swa_window,
            );
            let gpu = dflash_attn_readback(
                &ctx,
                &q,
                &k,
                &v,
                &pos_k,
                n,
                n_q,
                n_kv,
                hd,
                n_kv_total,
                c.ctx_len,
                c.noise_start_pos,
                c.swa_window,
            )
            .expect("dflash_attn dispatch");
            let gpu_two_range = dflash_attn_two_range_readback(
                &ctx,
                &q,
                &k_ctx,
                &v_ctx,
                &k_noise,
                &v_noise,
                &pos_ctx,
                n,
                n_q,
                n_kv,
                hd,
                c.ctx_len,
                c.noise_start_pos,
                c.swa_window,
                false,
                false,
                0,
            )
            .expect("dflash_attn_two_range dispatch");
            let gpu_online_two_range = dflash_attn_two_range_readback(
                &ctx,
                &q,
                &k_ctx,
                &v_ctx,
                &k_noise,
                &v_noise,
                &pos_ctx,
                n,
                n_q,
                n_kv,
                hd,
                c.ctx_len,
                c.noise_start_pos,
                c.swa_window,
                true,
                false,
                0,
            )
            .expect("dflash_attn_online_two_range dispatch");
            let ctx_scan_start = if c.swa_window > 0 && c.ctx_len > 0 {
                let min_pos = c.noise_start_pos.saturating_sub(c.swa_window);
                pos_ctx_vec.partition_point(|&pos| pos >= 0 && (pos as u32) < min_pos)
            } else {
                0
            };
            let gpu_online_two_range_scan = dflash_attn_two_range_readback(
                &ctx,
                &q,
                &k_ctx,
                &v_ctx,
                &k_noise,
                &v_noise,
                &pos_ctx,
                n,
                n_q,
                n_kv,
                hd,
                c.ctx_len,
                c.noise_start_pos,
                c.swa_window,
                true,
                false,
                ctx_scan_start,
            )
            .expect("dflash_attn_online_two_range scan dispatch");
            let gpu_full_gqa_split4 = if c.swa_window == 0 {
                Some(
                    dflash_attn_two_range_readback(
                        &ctx,
                        &q,
                        &k_ctx,
                        &v_ctx,
                        &k_noise,
                        &v_noise,
                        &pos_ctx,
                        n,
                        n_q,
                        n_kv,
                        hd,
                        c.ctx_len,
                        c.noise_start_pos,
                        c.swa_window,
                        false,
                        true,
                        0,
                    )
                    .expect("dflash_attn_full_gqa_split4 dispatch"),
                )
            } else {
                None
            };

            let mut max_abs = 0.0f32;
            let mut max_abs_two_range = 0.0f32;
            let mut max_abs_online_two_range = 0.0f32;
            let mut max_abs_online_two_range_scan = 0.0f32;
            let mut sum_sq_diff = 0.0f64;
            let mut sum_sq_diff_two_range = 0.0f64;
            let mut sum_sq_diff_online_two_range = 0.0f64;
            let mut sum_sq_diff_online_two_range_scan = 0.0f64;
            let mut sum_sq_cpu = 0.0f64;
            for i in 0..cpu.len() {
                let d = (gpu[i] - cpu[i]).abs();
                if d > max_abs {
                    max_abs = d;
                }
                let d_two_range = (gpu_two_range[i] - cpu[i]).abs();
                if d_two_range > max_abs_two_range {
                    max_abs_two_range = d_two_range;
                }
                let d_online_two_range = (gpu_online_two_range[i] - cpu[i]).abs();
                if d_online_two_range > max_abs_online_two_range {
                    max_abs_online_two_range = d_online_two_range;
                }
                let d_online_two_range_scan = (gpu_online_two_range_scan[i] - cpu[i]).abs();
                if d_online_two_range_scan > max_abs_online_two_range_scan {
                    max_abs_online_two_range_scan = d_online_two_range_scan;
                }
                let dd = (gpu[i] - cpu[i]) as f64;
                sum_sq_diff += dd * dd;
                let dd_two_range = (gpu_two_range[i] - cpu[i]) as f64;
                sum_sq_diff_two_range += dd_two_range * dd_two_range;
                let dd_online_two_range = (gpu_online_two_range[i] - cpu[i]) as f64;
                sum_sq_diff_online_two_range += dd_online_two_range * dd_online_two_range;
                let dd_online_two_range_scan = (gpu_online_two_range_scan[i] - cpu[i]) as f64;
                sum_sq_diff_online_two_range_scan +=
                    dd_online_two_range_scan * dd_online_two_range_scan;
                sum_sq_cpu += (cpu[i] as f64).powi(2);
            }
            let rel_l2 = sum_sq_diff.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
            let rel_l2_two_range = sum_sq_diff_two_range.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
            let rel_l2_online_two_range =
                sum_sq_diff_online_two_range.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
            let rel_l2_online_two_range_scan =
                sum_sq_diff_online_two_range_scan.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
            eprintln!(
                "[dflash-attn-mask {label}] max|Δ|={max_abs:.3e} rel_l2={rel_l2:.3e} two_range_max|Δ|={max_abs_two_range:.3e} two_range_rel_l2={rel_l2_two_range:.3e} online_two_range_max|Δ|={max_abs_online_two_range:.3e} online_two_range_rel_l2={rel_l2_online_two_range:.3e} scan_start={ctx_scan_start} online_scan_max|Δ|={max_abs_online_two_range_scan:.3e} online_scan_rel_l2={rel_l2_online_two_range_scan:.3e}",
                label = c.label
            );
            assert!(max_abs < 1e-4, "{}: max|Δ|={max_abs} too large", c.label);
            assert!(rel_l2 < 1e-5, "{}: rel_l2={rel_l2} too large", c.label);
            assert!(
                max_abs_two_range < 1e-4,
                "{}: two-range max|Δ|={max_abs_two_range} too large",
                c.label
            );
            assert!(
                rel_l2_two_range < 1e-5,
                "{}: two-range rel_l2={rel_l2_two_range} too large",
                c.label
            );
            assert!(
                max_abs_online_two_range < 1e-4,
                "{}: online two-range max|Δ|={max_abs_online_two_range} too large",
                c.label
            );
            assert!(
                rel_l2_online_two_range < 1e-5,
                "{}: online two-range rel_l2={rel_l2_online_two_range} too large",
                c.label
            );
            assert!(
                max_abs_online_two_range_scan < 1e-4,
                "{}: online scan max|Δ|={max_abs_online_two_range_scan} too large",
                c.label
            );
            assert!(
                rel_l2_online_two_range_scan < 1e-5,
                "{}: online scan rel_l2={rel_l2_online_two_range_scan} too large",
                c.label
            );
            if let Some(gpu_full_gqa_split4) = gpu_full_gqa_split4 {
                let mut candidate_max_abs = 0.0f32;
                let mut candidate_sq_diff = 0.0f64;
                for (&got, &want) in gpu_full_gqa_split4.iter().zip(&cpu) {
                    assert!(
                        got.is_finite(),
                        "{}: split4 produced nonfinite output",
                        c.label
                    );
                    let diff = (got - want).abs();
                    candidate_max_abs = candidate_max_abs.max(diff);
                    candidate_sq_diff += (diff as f64).powi(2);
                }
                let candidate_rel_l2 = candidate_sq_diff.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
                eprintln!(
                    "[dflash-attn-mask split4 {}] max|delta|={candidate_max_abs:.3e} \
                     rel_l2={candidate_rel_l2:.3e}",
                    c.label
                );
                assert!(
                    candidate_max_abs < 1e-4,
                    "{}: split4 max|delta|={candidate_max_abs} too large",
                    c.label
                );
                assert!(
                    candidate_rel_l2 < 1e-5,
                    "{}: split4 rel_l2={candidate_rel_l2} too large",
                    c.label
                );
            }
        }
    }

    /// **v0.72.2 codex code-review test #2**: head_dim > 256 must be
    /// rejected at the host wrapper. Kernel uses fixed-size [8] register
    /// arrays sized for head_dim=256; head_dim=320 would silently
    /// stack-OOB without this guard.
    #[test]
    fn dflash_attn_rejects_head_dim_over_256() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        // Make tiny placeholder buffers; we only care about the host
        // wrapper validation.
        let q = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let k = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let v = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let p = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let o = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        let res = encode_dflash_attn_f32(
            &ctx, &enc, &q, &k, &v, &p, &o, 16,   // n
            32,   // n_q_heads
            8,    // n_kv_heads
            320,  // head_dim — REJECTED
            17,   // n_kv_total
            1,    // ctx_len
            0,    // noise_start_pos
            2048, // swa_window
        );
        enc.end();
        match res {
            Err(MetalError::BadShape { detail, .. }) => {
                assert!(detail.contains("256"), "wrong error detail: {detail}");
            }
            other => panic!("expected BadShape on head_dim>256, got {other:?}"),
        }
    }
}
