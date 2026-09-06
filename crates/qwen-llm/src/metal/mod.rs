//! Host-side Metal lifecycle and kernel-encoding API.
//!
//! ## v1 design (post-codex review fc6f43e)
//!
//! * One [`MetalContext`] per process: holds `MTLDevice`, `MTLCommandQueue`,
//!   the embedded `kernels.metallib` loaded as a `MTLLibrary`, and a
//!   name → pipeline-state cache (`Mutex<HashMap>`; contention is irrelevant
//!   since pipeline creation only happens at first dispatch).
//!
//! * Persistent buffers via [`MetalTensor`]. Weights use either one copied
//!   `MTLBuffer` per tensor or an explicitly retained GGUF-backed buffer view.
//!   Both use `StorageModeShared` so reads/writes go straight to unified memory
//!   with no host↔device copies after setup.
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

use block2::RcBlock;
use memmap2::Mmap;
use objc2::AnyThread;
use objc2::rc::{Retained, Weak};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSRange, NSString, NSURL};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePassDescriptor, MTLComputePipelineState, MTLCounter,
    MTLCounterResultTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLDispatchType, MTLFence, MTLLibrary, MTLResource, MTLResourceOptions, MTLSize,
    MTLStorageMode,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixFindTopK,
};
use parking_lot::{Mutex, ReentrantMutex, ReentrantMutexGuard, const_reentrant_mutex};
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicUsize, Ordering},
};

use crate::gguf::GgufFile;

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
    fn getpagesize() -> i32;
    fn task_info(
        target_task: u32,
        flavor: i32,
        task_info_out: *mut i32,
        task_info_out_count: *mut u32,
    ) -> i32;
}

static KERNEL_TRACE_ACTIVE_THREADS: AtomicUsize = AtomicUsize::new(0);

const ATTN_V4_SUBGROUP_MIN_POS_DEFAULT: usize = 256;
const ATTN_V4_NWG_MAX: usize = 1024;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AttnMatrixVtDispatchStats {
    pub calls: u64,
    pub row_sum: u64,
    pub element_sum: u64,
    pub threadgroup_sum: u64,
    pub compact_calls: u64,
    pub legacy_calls: u64,
    pub base_pos_sum: u64,
    pub n_pos_sum: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttnMatrixVtDispatchCapture {
    pub owner_thread: String,
    pub stats: AttnMatrixVtDispatchStats,
}

thread_local! {
    static ATTN_V4_GROUP_TILE_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
    static ATTN_MATRIX_VT_COMPACT_DISPATCH_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static ATTN_MATRIX_VT_CAPTURE_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static ATTN_MATRIX_VT_CAPTURE_STATS: Cell<AttnMatrixVtDispatchStats> = const {
        Cell::new(AttnMatrixVtDispatchStats {
            calls: 0,
            row_sum: 0,
            element_sum: 0,
            threadgroup_sum: 0,
            compact_calls: 0,
            legacy_calls: 0,
            base_pos_sum: 0,
            n_pos_sum: 0,
        })
    };
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

fn attn_matrix_vt_override_owner() -> &'static Mutex<Option<std::thread::ThreadId>> {
    static OWNER: OnceLock<Mutex<Option<std::thread::ThreadId>>> = OnceLock::new();
    OWNER.get_or_init(|| Mutex::new(None))
}

fn attn_matrix_vt_capture_owner() -> &'static Mutex<Option<std::thread::ThreadId>> {
    static OWNER: OnceLock<Mutex<Option<std::thread::ThreadId>>> = OnceLock::new();
    OWNER.get_or_init(|| Mutex::new(None))
}

fn claim_attn_matrix_vt_scope(
    owner_slot: &Mutex<Option<std::thread::ThreadId>>,
    kernel: &'static str,
) -> Result<std::thread::ThreadId, MetalError> {
    let owner = std::thread::current().id();
    let mut active = owner_slot.lock();
    if let Some(active_owner) = active.as_ref() {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("scope already owned by {active_owner:?}"),
        });
    }
    *active = Some(owner);
    Ok(owner)
}

fn release_attn_matrix_vt_scope(
    owner_slot: &Mutex<Option<std::thread::ThreadId>>,
    owner: &std::thread::ThreadId,
) {
    let mut active = owner_slot.lock();
    if active.as_ref() == Some(owner) {
        *active = None;
    }
}

struct AttnMatrixVtOverrideGuard {
    previous: Option<bool>,
    owner: std::thread::ThreadId,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for AttnMatrixVtOverrideGuard {
    fn drop(&mut self) {
        ATTN_MATRIX_VT_COMPACT_DISPATCH_OVERRIDE.with(|slot| slot.set(self.previous));
        release_attn_matrix_vt_scope(attn_matrix_vt_override_owner(), &self.owner);
    }
}

pub fn with_attn_matrix_vt_compact_dispatch_override<R>(
    enabled: bool,
    f: impl FnOnce() -> R,
) -> Result<R, MetalError> {
    let owner =
        claim_attn_matrix_vt_scope(attn_matrix_vt_override_owner(), "attn_matrix_vt_override")?;
    let previous = ATTN_MATRIX_VT_COMPACT_DISPATCH_OVERRIDE.with(|slot| {
        let previous = slot.get();
        slot.set(Some(enabled));
        previous
    });
    if previous.is_some() {
        ATTN_MATRIX_VT_COMPACT_DISPATCH_OVERRIDE.with(|slot| slot.set(previous));
        release_attn_matrix_vt_scope(attn_matrix_vt_override_owner(), &owner);
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_vt_override",
            detail: "nested compact-dispatch override is forbidden".into(),
        });
    }
    let guard = AttnMatrixVtOverrideGuard {
        previous,
        owner,
        _not_send: PhantomData,
    };
    let out = f();
    drop(guard);
    Ok(out)
}

struct AttnMatrixVtCaptureGuard {
    owner: std::thread::ThreadId,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for AttnMatrixVtCaptureGuard {
    fn drop(&mut self) {
        ATTN_MATRIX_VT_CAPTURE_ACTIVE.with(|slot| slot.set(false));
        release_attn_matrix_vt_scope(attn_matrix_vt_capture_owner(), &self.owner);
    }
}

pub fn capture_attn_matrix_vt_dispatches<R>(
    f: impl FnOnce() -> R,
) -> Result<(R, AttnMatrixVtDispatchCapture), MetalError> {
    let owner =
        claim_attn_matrix_vt_scope(attn_matrix_vt_capture_owner(), "attn_matrix_vt_capture")?;
    let already_active = ATTN_MATRIX_VT_CAPTURE_ACTIVE.with(|slot| slot.replace(true));
    if already_active {
        ATTN_MATRIX_VT_CAPTURE_ACTIVE.with(|slot| slot.set(true));
        release_attn_matrix_vt_scope(attn_matrix_vt_capture_owner(), &owner);
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_vt_capture",
            detail: "nested V_T dispatch capture is forbidden".into(),
        });
    }
    ATTN_MATRIX_VT_CAPTURE_STATS.with(|slot| slot.set(AttnMatrixVtDispatchStats::default()));
    let guard = AttnMatrixVtCaptureGuard {
        owner,
        _not_send: PhantomData,
    };
    let owner_thread = format!("{:?}", std::thread::current().id());
    let out = f();
    let stats = ATTN_MATRIX_VT_CAPTURE_STATS.with(Cell::get);
    drop(guard);
    Ok((
        out,
        AttnMatrixVtDispatchCapture {
            owner_thread,
            stats,
        },
    ))
}

use crate::tensor::{GgmlType, TensorDesc, checked_shape_elements, ggml_type_layout};

mod attn;
mod bench;
mod census;
mod context;
mod dflash;
mod elementwise;
mod encoder;
mod gdn;
mod mat_mat;
mod mat_vec;
mod moe;
mod norm;
mod research;
mod rope;
mod tensor;
#[cfg(test)]
mod test_support;
mod testing;
#[cfg(test)]
mod tests;
mod vjp;

pub use attn::*;
pub use bench::*;
pub use census::*;
pub use context::*;
pub use dflash::*;
pub use elementwise::*;
pub use encoder::*;
pub use gdn::*;
pub use mat_mat::*;
pub use mat_vec::*;
pub use moe::*;
pub use norm::*;
pub use research::*;
pub use rope::*;
pub use tensor::*;
pub use testing::*;
pub use vjp::*;

#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("no Metal device available")]
    NoDevice,
    #[error("could not create command queue")]
    NoQueue,
    #[error("could not create blit command encoder")]
    NoBlitEncoder,
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
    #[error("GGUF no-copy backing: {0}")]
    GgufNoCopy(String),
    #[error("Metal process lease unavailable: {0}")]
    ProcessLease(String),
    #[error("could not inspect host memory before Metal initialization: {0}")]
    HostMemoryTelemetry(String),
    #[error(
        "refusing Metal initialization after wired memory remained unsafe for 15 seconds: wired={wired_bytes} physical={physical_bytes}; reboot if no large Metal process remains"
    )]
    UnsafeHostWiredMemory {
        wired_bytes: u64,
        physical_bytes: u64,
    },
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
    activated: bool,
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
        if self.activated {
            let previous = KERNEL_TRACE_ACTIVE_THREADS.fetch_sub(1, Ordering::Relaxed);
            debug_assert!(previous > 0);
        }
    }
}

pub fn kernel_trace_begin() -> KernelTraceGuard {
    KERNEL_TRACE_COUNTERS.with(|counters| counters.set(KernelTraceCounters::default()));
    KERNEL_TRACE_LAST.with(|last| last.set(KernelTraceCounters::default()));
    let previous = KERNEL_TRACE_ACTIVE.with(|active| {
        let previous = active.get();
        active.set(true);
        previous
    });
    let activated = !previous;
    if activated {
        KERNEL_TRACE_ACTIVE_THREADS.fetch_add(1, Ordering::Relaxed);
    }
    KernelTraceGuard {
        previous,
        activated,
    }
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
    census_record_encoder(concurrent);
    if KERNEL_TRACE_ACTIVE_THREADS.load(Ordering::Relaxed) == 0 {
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
    if KERNEL_TRACE_ACTIVE_THREADS.load(Ordering::Relaxed) == 0 {
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
