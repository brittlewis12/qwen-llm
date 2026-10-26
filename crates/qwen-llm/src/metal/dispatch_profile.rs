//! Opt-in per-dispatch GPU timing (bench-only).
//!
//! Apple GPUs here sample timestamps only at encoder (stage) boundaries; M4 Max
//! and M5 reject dispatch-boundary sampling. While a profile is active on this
//! thread, every dispatch of a plain [`KernelEncoder`] (from `begin` or
//! `begin_concurrent`, created after the profile began) therefore runs in its
//! own sampled serial encoder, with the parent's bindings replayed. Each
//! sample is an isolated kernel duration: concurrency is removed, and encoder
//! transitions fall outside the samples. Compare the sum of kernel durations
//! with an unprofiled command's GPU time to separate dispatch gaps from kernel
//! time. Sampled encoders (`begin_sampled`) are never split.

use super::*;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

static ACTIVE_THREADS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static PROFILE: RefCell<Option<Active>> = const { RefCell::new(None) };
}

/// Metal caps one counter sample buffer at 32 KiB (4,096 timestamps), so
/// each buffer holds the start/end pairs of this many dispatches.
const DISPATCHES_PER_BUFFER: usize = 2048;

struct Active {
    samples: Vec<MetalTimestampSampleBuffer>,
    rows: Vec<DispatchProfileRow>,
    label: &'static str,
    layer: Option<usize>,
    overflow: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct DispatchProfileRow {
    pub label: &'static str,
    pub layer: Option<usize>,
    pub kernel: String,
    pub grid: [usize; 3],
    pub threads: [usize; 3],
    /// The dispatch came from a concurrent encoder (it ran serialized here).
    pub concurrent: bool,
    /// None when the timestamp pair was unavailable or not ordered.
    pub gpu_ns: Option<u64>,
    /// Raw encoder start/end timestamps (GPU ns), kept with `gpu_ns`. A start
    /// can precede the previous dispatch's end (the encoder begins before
    /// its dependency drains), so attribute by end-to-end increments when
    /// summing a timeline.
    pub gpu_start: Option<u64>,
    pub gpu_end: Option<u64>,
}

#[derive(Debug, serde::Serialize)]
pub struct DispatchProfileReport {
    pub rows: Vec<DispatchProfileRow>,
    /// Dispatches that ran unsplit because the sample buffer was full.
    pub overflow: usize,
}

/// Thread-bound; dropping it without [`dispatch_profile_finish`] discards the
/// profile. No nesting.
#[must_use]
pub struct DispatchProfileGuard {
    _thread: PhantomData<Rc<()>>,
}

impl Drop for DispatchProfileGuard {
    fn drop(&mut self) {
        if PROFILE.with(|p| p.borrow_mut().take()).is_some() {
            let previous = ACTIVE_THREADS.fetch_sub(1, Ordering::Relaxed);
            debug_assert!(previous > 0);
        }
    }
}

/// Start splitting dispatches on this thread, for up to `max_dispatches`.
pub fn dispatch_profile_begin(
    ctx: &MetalContext,
    max_dispatches: usize,
) -> Result<DispatchProfileGuard, MetalError> {
    if max_dispatches == 0 {
        return Err(MetalError::Counter("empty dispatch profile".into()));
    }
    let samples = (0..max_dispatches.div_ceil(DISPATCHES_PER_BUFFER))
        .map(|i| {
            let dispatches =
                (max_dispatches - i * DISPATCHES_PER_BUFFER).min(DISPATCHES_PER_BUFFER);
            ctx.timestamp_sample_buffer(2 * dispatches)
        })
        .collect::<Result<Vec<_>, _>>()?;
    PROFILE.with(|p| {
        let mut p = p.borrow_mut();
        if p.is_some() {
            return Err(MetalError::Counter(
                "nested dispatch profile on one thread".into(),
            ));
        }
        *p = Some(Active {
            samples,
            rows: Vec::with_capacity(max_dispatches),
            label: "",
            layer: None,
            overflow: 0,
        });
        ACTIVE_THREADS.fetch_add(1, Ordering::Relaxed);
        Ok(DispatchProfileGuard {
            _thread: PhantomData,
        })
    })
}

/// Attribute subsequent dispatches on this thread. No-op without a profile.
#[inline]
pub fn dispatch_profile_label(label: &'static str, layer: Option<usize>) {
    if ACTIVE_THREADS.load(Ordering::Relaxed) == 0 {
        return;
    }
    PROFILE.with(|p| {
        if let Some(active) = p.borrow_mut().as_mut() {
            active.label = label;
            active.layer = layer;
        }
    });
}

/// Resolve the recorded dispatches. Call only after every command that
/// carried them has completed.
pub fn dispatch_profile_finish(
    ctx: &MetalContext,
    guard: DispatchProfileGuard,
) -> Result<DispatchProfileReport, MetalError> {
    let active = PROFILE.with(|p| p.borrow_mut().take());
    // The slot is already empty, so the guard's drop does not decrement.
    drop(guard);
    let Some(mut active) = active else {
        return Err(MetalError::Counter("dispatch profile is not active".into()));
    };
    let previous = ACTIVE_THREADS.fetch_sub(1, Ordering::Relaxed);
    debug_assert!(previous > 0);
    if !active.rows.is_empty() {
        for (rows, samples) in active
            .rows
            .chunks_mut(DISPATCHES_PER_BUFFER)
            .zip(&active.samples)
        {
            let times = ctx.resolve_timestamp_samples(samples, 2 * rows.len())?;
            for (row, &[start, end]) in rows.iter_mut().zip(times.as_chunks::<2>().0) {
                if start > 0 && end > start && end != u64::MAX {
                    row.gpu_ns = Some(end - start);
                    (row.gpu_start, row.gpu_end) = (Some(start), Some(end));
                }
            }
        }
        if active.overflow > 0 {
            // Unsplit overflow dispatches ran inside the last sampled encoder.
            let last = active.rows.last_mut().unwrap();
            (last.gpu_ns, last.gpu_start, last.gpu_end) = (None, None, None);
        }
    }
    Ok(DispatchProfileReport {
        rows: active.rows,
        overflow: active.overflow,
    })
}

/// Whether a new plain encoder on this thread should record bindings.
#[inline]
pub(crate) fn dispatch_profile_records() -> bool {
    ACTIVE_THREADS.load(Ordering::Relaxed) != 0 && PROFILE.with(|p| p.borrow().is_some())
}

/// Bindings set on a plain encoder while a profile is active, replayed onto
/// each per-dispatch encoder.
#[derive(Default)]
pub(crate) struct Replay {
    pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    slots: Vec<(usize, Slot)>,
    threadgroup: Vec<(usize, usize)>,
    resources: Vec<Buffer>,
}

enum Slot {
    Buffer(Buffer, usize),
    Bytes(Vec<u8>),
}

impl Replay {
    pub(crate) fn pipeline(&mut self, pso: &Pipeline) {
        self.pipeline = Some(pso.raw.clone());
    }

    pub(crate) fn buffer(&mut self, index: usize, buf: &Buffer, offset: u64) {
        self.slot(index, Slot::Buffer(buf.clone(), offset as usize));
    }

    pub(crate) fn bytes(&mut self, index: usize, bytes: &[u8]) {
        self.slot(index, Slot::Bytes(bytes.to_vec()));
    }

    pub(crate) fn threadgroup(&mut self, index: usize, n_bytes: usize) {
        match self.threadgroup.iter_mut().find(|(i, _)| *i == index) {
            Some(entry) => entry.1 = n_bytes,
            None => self.threadgroup.push((index, n_bytes)),
        }
    }

    pub(crate) fn resource(&mut self, buf: &Buffer) {
        self.resources.push(buf.clone());
    }

    fn slot(&mut self, index: usize, slot: Slot) {
        match self.slots.iter_mut().find(|(i, _)| *i == index) {
            Some(entry) => entry.1 = slot,
            None => self.slots.push((index, slot)),
        }
    }

    fn apply(&self, raw: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        if let Some(pso) = &self.pipeline {
            raw.setComputePipelineState(pso);
        }
        for (index, slot) in &self.slots {
            match slot {
                Slot::Buffer(buf, offset) => unsafe {
                    raw.setBuffer_offset_atIndex(Some(buf.as_ref()), *offset, *index);
                },
                Slot::Bytes(bytes) => {
                    let ptr = std::ptr::NonNull::new(bytes.as_ptr() as *mut std::ffi::c_void)
                        .expect("non-null replay bytes");
                    unsafe { raw.setBytes_length_atIndex(ptr, bytes.len(), *index) };
                }
            }
        }
        for &(index, n_bytes) in &self.threadgroup {
            unsafe { raw.setThreadgroupMemoryLength_atIndex(n_bytes, index) };
        }
        for buf in &self.resources {
            let resource: &ProtocolObject<dyn objc2_metal::MTLResource> =
                ProtocolObject::from_ref(&**buf);
            raw.useResource_usage(resource, objc2_metal::MTLResourceUsage::Read);
        }
    }
}

/// Move `enc` onto a fresh sampled serial encoder for its next dispatch.
pub(crate) fn split_for_dispatch(enc: &KernelEncoder, grid: MTLSize, threads: MTLSize) {
    PROFILE.with(|p| {
        let mut p = p.borrow_mut();
        let Some(active) = p.as_mut() else { return };
        let replay = enc.replay.borrow();
        let Some(replay) = replay.as_ref() else {
            return;
        };
        let k = active.rows.len();
        let (buffer, slot) = (k / DISPATCHES_PER_BUFFER, k % DISPATCHES_PER_BUFFER);
        let Some(samples) = active
            .samples
            .get(buffer)
            .filter(|s| 2 * slot + 1 < s.sample_count)
        else {
            active.overflow += 1;
            return;
        };
        enc.raw.borrow().endEncoding();
        let raw = KernelEncoder::sampled_raw(&enc.parent, samples, 2 * slot, 2 * slot + 1, false)
            .expect("per-dispatch sampled compute encoder");
        replay.apply(&raw);
        *enc.raw.borrow_mut() = raw;
        active.rows.push(DispatchProfileRow {
            label: active.label,
            layer: active.layer,
            kernel: enc
                .bound_pipeline_name
                .borrow()
                .as_deref()
                .unwrap_or("")
                .to_string(),
            grid: [grid.width, grid.height, grid.depth],
            threads: [threads.width, threads.height, threads.depth],
            concurrent: enc.concurrent,
            gpu_ns: None,
            gpu_start: None,
            gpu_end: None,
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE: MTLSize = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    fn read(buf: &Buffer) -> f32 {
        // Called only after successful command completion.
        unsafe { *buf.contents().as_ptr().cast::<f32>() }
    }

    /// add(b0, b1) -> b2, then mul(b0, b1) -> b3 with only the output
    /// rebound, so the second dispatch depends on replayed bindings.
    fn run(ctx: &MetalContext, capacity: usize) -> (DispatchProfileReport, [f32; 2]) {
        let buffers =
            [2.0f32, 3.0, 0.0, 0.0].map(|v| ctx.buffer_from(bytemuck::bytes_of(&v)).unwrap());
        let add = ctx.pipeline("kernel_add_f32").unwrap();
        let mul = ctx.pipeline("kernel_mul_f32").unwrap();
        let guard = dispatch_profile_begin(ctx, capacity).unwrap();
        assert!(matches!(
            dispatch_profile_begin(ctx, capacity),
            Err(MetalError::Counter(_))
        ));
        dispatch_profile_label("first", Some(3));
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        enc.set_bytes(0, &1u32);
        enc.set_buffer(1, &buffers[0], 0);
        enc.set_buffer(2, &buffers[1], 0);
        enc.set_buffer(3, &buffers[2], 0);
        enc.set_pipeline(&add);
        enc.dispatch(ONE, ONE);
        dispatch_profile_label("second", None);
        enc.set_pipeline(&mul);
        enc.set_buffer(3, &buffers[3], 0);
        enc.dispatch(ONE, ONE);
        enc.end();
        cmd.commit();
        wait_completed(&cmd).unwrap();
        assert!(cmd.error().is_none(), "command failed: {:?}", cmd.error());
        let report = dispatch_profile_finish(ctx, guard).unwrap();
        assert!(!dispatch_profile_records());
        (report, [read(&buffers[2]), read(&buffers[3])])
    }

    #[test]
    fn profile_splits_each_dispatch_and_replays_bindings() {
        let Some(ctx) = crate::metal::test_support::metal_test_context() else {
            return;
        };
        let (report, outputs) = run(&ctx, 8);
        assert_eq!(outputs, [5.0, 6.0]);
        assert_eq!(report.overflow, 0);
        assert_eq!(
            report
                .rows
                .iter()
                .map(|r| (r.label, r.layer, r.kernel.as_str(), r.concurrent))
                .collect::<Vec<_>>(),
            [
                ("first", Some(3), "kernel_add_f32", false),
                ("second", None, "kernel_mul_f32", false)
            ]
        );
        assert!(report.rows.iter().all(|r| r.gpu_ns.is_some()));
    }

    #[test]
    fn profile_overflow_runs_unsplit_and_voids_the_shared_sample() {
        let Some(ctx) = crate::metal::test_support::metal_test_context() else {
            return;
        };
        let (report, outputs) = run(&ctx, 1);
        assert_eq!(outputs, [5.0, 6.0]);
        assert_eq!(report.overflow, 1);
        assert_eq!(report.rows.len(), 1);
        assert_eq!(report.rows[0].gpu_ns, None);
    }

    #[test]
    fn encoders_begun_outside_a_profile_never_split() {
        let Some(ctx) = crate::metal::test_support::metal_test_context() else {
            return;
        };
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        let guard = dispatch_profile_begin(&ctx, 4).unwrap();
        assert!(enc.replay.borrow().is_none());
        enc.end();
        cmd.commit();
        wait_completed(&cmd).unwrap();
        let report = dispatch_profile_finish(&ctx, guard).unwrap();
        assert!(report.rows.is_empty());
    }
}
