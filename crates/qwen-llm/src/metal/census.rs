//! Dispatch census (bench-only attribution of kernel launches).

use super::*;

#[derive(Clone, Debug)]
pub struct DispatchCensusRow {
    pub family: &'static str,
    pub tag: Option<String>,
    pub encoder_ordinal: u64,
    pub encoder_concurrent: bool,
    pub kernel: String,
    pub grid_width: u64,
    pub grid_height: u64,
    pub grid_depth: u64,
    pub threads_width: u64,
    pub threads_height: u64,
    pub threads_depth: u64,
    pub grid_tgs: u64,
    pub tg_threads: u64,
}

pub(crate) static DISPATCH_CENSUS_ACTIVE_THREADS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static DISPATCH_CENSUS: std::cell::RefCell<Option<Vec<DispatchCensusRow>>> =
        const { std::cell::RefCell::new(None) };
    static CENSUS_LAST_PSO: std::cell::RefCell<String> =
        const { std::cell::RefCell::new(String::new()) };
    static CENSUS_FAMILY: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
    static CENSUS_TAG: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    static CENSUS_NEXT_ENCODER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static CENSUS_ENCODER: std::cell::Cell<(u64, bool)> = const { std::cell::Cell::new((u64::MAX, false)) };
}

#[must_use]
pub struct DispatchCensusTagGuard {
    pub(crate) previous: Option<String>,
}

impl Drop for DispatchCensusTagGuard {
    fn drop(&mut self) {
        CENSUS_TAG.with(|tag| *tag.borrow_mut() = self.previous.take());
    }
}

/// Begin recording dispatch shapes on this thread. Bench-only.
pub fn dispatch_census_begin() {
    let activated = DISPATCH_CENSUS.with(|c| {
        let activated = c.borrow().is_none();
        *c.borrow_mut() = Some(Vec::with_capacity(512));
        activated
    });
    if activated {
        DISPATCH_CENSUS_ACTIVE_THREADS.fetch_add(1, Ordering::Relaxed);
    }
    CENSUS_TAG.with(|tag| *tag.borrow_mut() = None);
    CENSUS_NEXT_ENCODER.with(|next| next.set(0));
    CENSUS_ENCODER.with(|encoder| encoder.set((u64::MAX, false)));
}

/// Stop recording and take the census rows.
pub fn dispatch_census_take() -> Vec<DispatchCensusRow> {
    let rows = DISPATCH_CENSUS.with(|c| c.borrow_mut().take());
    if rows.is_some() {
        let previous = DISPATCH_CENSUS_ACTIVE_THREADS.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(previous > 0);
    }
    rows.unwrap_or_default()
}

/// Set the current stage family label (called by decode stage boundaries).
pub fn dispatch_census_set_family(family: &'static str) {
    if DISPATCH_CENSUS_ACTIVE_THREADS.load(Ordering::Relaxed) == 0 {
        return;
    }
    CENSUS_FAMILY.with(|f| f.set(family));
}

pub fn dispatch_census_is_active() -> bool {
    DISPATCH_CENSUS_ACTIVE_THREADS.load(Ordering::Relaxed) != 0
        && DISPATCH_CENSUS.with(|census| census.borrow().is_some())
}

#[doc(hidden)]
pub fn diagnostics_observer_active_counts() -> [usize; 3] {
    [
        KERNEL_TRACE_ACTIVE_THREADS.load(Ordering::Relaxed),
        DISPATCH_CENSUS_ACTIVE_THREADS.load(Ordering::Relaxed),
        ALLOCATION_CENSUS_ACTIVE_THREADS.load(Ordering::Relaxed),
    ]
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalAllocationCensusRow {
    pub requested_bytes: u64,
    pub buffer_length: u64,
    pub storage_mode: &'static str,
}

pub(crate) static ALLOCATION_CENSUS_ACTIVE_THREADS: AtomicUsize = AtomicUsize::new(0);

pub(crate) static METAL_ALLOCATION_TRANSACTION: ReentrantMutex<()> = const_reentrant_mutex(());

pub(crate) struct MetalAllocationTransactionGuard {
    pub(crate) _guard: ReentrantMutexGuard<'static, ()>,
}

thread_local! {
    static ALLOCATION_CENSUS: std::cell::RefCell<Option<Vec<MetalAllocationCensusRow>>> =
        const { std::cell::RefCell::new(None) };
}

#[doc(hidden)]
pub fn allocation_census_begin() {
    let activated = ALLOCATION_CENSUS.with(|census| {
        let activated = census.borrow().is_none();
        *census.borrow_mut() = Some(Vec::with_capacity(512));
        activated
    });
    if activated {
        ALLOCATION_CENSUS_ACTIVE_THREADS.fetch_add(1, Ordering::Relaxed);
    }
}

#[doc(hidden)]
pub fn allocation_census_take() -> Vec<MetalAllocationCensusRow> {
    let rows = ALLOCATION_CENSUS.with(|census| census.borrow_mut().take());
    if rows.is_some() {
        let previous = ALLOCATION_CENSUS_ACTIVE_THREADS.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(previous > 0);
    }
    rows.unwrap_or_default()
}

pub(crate) fn record_buffer_allocation(requested_bytes: usize, buffer: &Buffer) {
    if ALLOCATION_CENSUS_ACTIVE_THREADS.load(Ordering::Relaxed) == 0 {
        return;
    }
    ALLOCATION_CENSUS.with(|census| {
        if let Some(rows) = census.borrow_mut().as_mut() {
            let storage_mode = match buffer.storageMode() {
                MTLStorageMode::Shared => "shared",
                MTLStorageMode::Managed => "managed",
                MTLStorageMode::Private => "private",
                MTLStorageMode::Memoryless => "memoryless",
                _ => "unknown",
            };
            rows.push(MetalAllocationCensusRow {
                requested_bytes: requested_bytes as u64,
                buffer_length: buffer.length() as u64,
                storage_mode,
            });
        }
    });
}

/// Attach an explicit diagnostics tag to dispatches recorded in this scope.
/// Returns `None` without allocating when no census is active on this thread.
pub fn dispatch_census_tag_scope(tag: impl FnOnce() -> String) -> Option<DispatchCensusTagGuard> {
    if !dispatch_census_is_active() {
        return None;
    }
    let tag = tag();
    let previous = CENSUS_TAG.with(|current| current.borrow_mut().replace(tag));
    Some(DispatchCensusTagGuard { previous })
}

#[inline]
pub(crate) fn census_record_pso(name: &str) {
    if DISPATCH_CENSUS_ACTIVE_THREADS.load(Ordering::Relaxed) == 0 {
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
pub(crate) fn census_record_encoder(concurrent: bool) {
    if DISPATCH_CENSUS_ACTIVE_THREADS.load(Ordering::Relaxed) == 0 {
        return;
    }
    DISPATCH_CENSUS.with(|census| {
        if census.borrow().is_some() {
            let ordinal = CENSUS_NEXT_ENCODER.with(|next| {
                let ordinal = next.get();
                next.set(ordinal + 1);
                ordinal
            });
            CENSUS_ENCODER.with(|encoder| encoder.set((ordinal, concurrent)));
        }
    });
}

#[inline]
pub(crate) fn census_record_dispatch(grid: MTLSize, threads: MTLSize) {
    if DISPATCH_CENSUS_ACTIVE_THREADS.load(Ordering::Relaxed) == 0 {
        return;
    }
    DISPATCH_CENSUS.with(|c| {
        if let Some(rows) = c.borrow_mut().as_mut() {
            #[cfg(test)]
            if std::env::var_os("QWEN_TEST_DISPATCH_TRACE").is_some() {
                eprintln!(
                    "validation_dispatch kernel={} grid={grid:?} threads={threads:?}",
                    CENSUS_LAST_PSO.with(|p| p.borrow().clone())
                );
            }
            rows.push(DispatchCensusRow {
                family: CENSUS_FAMILY.with(|f| f.get()),
                tag: CENSUS_TAG.with(|tag| tag.borrow().clone()),
                encoder_ordinal: CENSUS_ENCODER.with(|encoder| encoder.get().0),
                encoder_concurrent: CENSUS_ENCODER.with(|encoder| encoder.get().1),
                kernel: CENSUS_LAST_PSO.with(|p| p.borrow().clone()),
                grid_width: grid.width as u64,
                grid_height: grid.height as u64,
                grid_depth: grid.depth as u64,
                threads_width: threads.width as u64,
                threads_height: threads.height as u64,
                threads_depth: threads.depth as u64,
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
pub(crate) fn checked_shape_bytes(
    shape: &[u64],
    elem_bytes: usize,
) -> Result<(usize, usize), MetalError> {
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

pub(crate) fn checked_ggml_shape_bytes(
    shape: &[u64],
    dtype: GgmlType,
) -> Result<(usize, usize), MetalError> {
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

pub(crate) type Device = Retained<ProtocolObject<dyn MTLDevice>>;

pub(crate) type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

pub(crate) type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

pub(crate) type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

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
pub(crate) struct MetalPipelineCache {
    pub(crate) pipelines: HashMap<String, Pipeline>,
    pub(crate) metrics_enabled: bool,
    pub(crate) metrics: MetalPipelineCacheMetrics,
}

pub(crate) fn duration_ns_saturating(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
