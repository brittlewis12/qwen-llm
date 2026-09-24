//! KernelEncoder and BlitEncoder: caller-owned command encoders.

use super::*;

/// Caller-owned wrapper around `MTLComputeCommandEncoder`. Produced by
/// [`KernelEncoder::begin`] from a `MTLCommandBuffer`. All `encode_*`
/// kernels accept this and mutate it. Callers should use `end()` as the explicit
/// pass boundary before committing the parent command buffer. Early returns and
/// panics are finalized by `Drop`; `Drop` never commits partial work.
///
/// This is the *one* place in the engine where Metal lifecycle is exposed
/// to client code. The forward-pass driver owns the command buffer; every
/// kernel just appends dispatches.
pub struct KernelEncoder {
    pub(crate) raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    pub(crate) parent: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub(crate) ended: bool,
    /// True when created via [`KernelEncoder::begin_concurrent`]. Concurrent
    /// passes provide NO ordering between dispatches, so every dispatch pair
    /// must be independent (disjoint writes; no dispatch reads another's
    /// output). That invariant is a convention, not a type-system property —
    /// the debug-only `note_read`/`note_write` hazard tracker below turns a
    /// future violation into a loud panic instead of a silent GPU race.
    pub(crate) concurrent: bool,
    #[cfg(debug_assertions)]
    pub(crate) hazard_writes: std::cell::RefCell<Vec<(usize, u64, u64)>>,
    #[cfg(debug_assertions)]
    pub(crate) hazard_reads: std::cell::RefCell<Vec<(usize, u64, u64)>>,
}

#[cfg(debug_assertions)]
pub(crate) fn ranges_overlap(a_off: u64, a_len: u64, b_off: u64, b_len: u64) -> bool {
    a_off < b_off + b_len && b_off < a_off + a_len
}

impl KernelEncoder {
    pub(crate) fn new(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        parent: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        concurrent: bool,
    ) -> Self {
        Self {
            raw,
            parent: parent.clone(),
            ended: false,
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
        Self::new(raw, cmd, false)
    }

    pub fn begin_concurrent(cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        let raw = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .expect("concurrent compute encoder");
        kernel_trace_record_encoder(true);
        Self::new(raw, cmd, true)
    }

    pub(crate) fn is_concurrent(&self) -> bool {
        self.concurrent
    }

    pub(crate) fn parent_command_buffer(&self) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        self.parent.clone()
    }

    pub fn begin_sampled(
        cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        samples: &MetalTimestampSampleBuffer,
        start_sample: usize,
        end_sample: usize,
        concurrent: bool,
    ) -> Self {
        Self::try_begin_sampled(cmd, samples, start_sample, end_sample, concurrent)
            .expect("sampled compute encoder")
    }

    pub fn try_begin_sampled(
        cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        samples: &MetalTimestampSampleBuffer,
        start_sample: usize,
        end_sample: usize,
        concurrent: bool,
    ) -> Result<Self, MetalError> {
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
            .ok_or_else(|| {
                MetalError::Counter("could not create sampled compute encoder".into())
            })?;
        kernel_trace_record_encoder(concurrent);
        Ok(Self::new(raw, cmd, concurrent))
    }

    /// Debug-only hazard note: declare that a dispatch in this encoder
    /// WRITES `tensor`'s byte range. On a concurrent encoder, panics if the
    /// range overlaps any previously noted write or read — either would be
    /// an unsynchronized data race inside the concurrent pass. The access-mode
    /// check is always active; range tracking is a no-op in release builds and
    /// on serial encoders.
    #[inline]
    pub fn note_write(&self, tensor: &MetalTensor) {
        tensor.assert_writable("KernelEncoder::note_write");
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

    pub(crate) fn finish(&mut self) {
        if !self.ended {
            self.ended = true;
            self.raw.endEncoding();
        }
    }

    pub fn end(mut self) {
        self.finish();
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

    /// Bind a nonempty Pod slice inline. Metal limits inline payloads to 4 KiB.
    pub(crate) fn set_bytes_slice<T: bytemuck::Pod>(&self, index: usize, values: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(values);
        assert!(!bytes.is_empty(), "inline Metal slice must not be empty");
        assert!(
            bytes.len() <= 4_096,
            "inline Metal slice exceeds the 4 KiB API limit"
        );
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

impl Drop for KernelEncoder {
    fn drop(&mut self) {
        self.finish();
    }
}

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
    pub(crate) ended: bool,
}

impl BlitEncoder {
    pub fn try_begin(
        cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) -> Result<Self, MetalError> {
        let raw = cmd.blitCommandEncoder().ok_or(MetalError::NoBlitEncoder)?;
        Ok(Self { raw, ended: false })
    }

    pub fn begin(cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self::try_begin(cmd).expect("blit encoder")
    }

    pub(crate) fn finish(&mut self) {
        if !self.ended {
            self.ended = true;
            self.raw.endEncoding();
        }
    }

    pub fn end(mut self) {
        self.finish();
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
        dst.assert_writable("BlitEncoder::copy_tensor");
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

impl Drop for BlitEncoder {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_encoder_drop_closes_validation_error_pass() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let embed = MetalTensor::zeros_f32(&ctx, vec![8]).unwrap();
        let wrong_ids = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let output = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        {
            let enc = KernelEncoder::begin(&cmd);
            let error = encode_get_rows_f32(&ctx, &enc, &embed, &wrong_ids, &output, 1, 4)
                .expect_err("F32 IDs must be rejected");
            assert!(matches!(error, MetalError::BadShape { .. }));
        }
        let enc = KernelEncoder::begin(&cmd);
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).expect("command buffer completed");
        assert!(cmd.error().is_none(), "command failed: {:?}", cmd.error());
    }

    #[test]
    fn kernel_encoder_drop_closes_panicking_pass() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _enc = KernelEncoder::begin(&cmd);
            panic!("synthetic encoder unwind");
        }));
        assert!(panic.is_err());
        let enc = KernelEncoder::begin(&cmd);
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).expect("command buffer completed");
        assert!(cmd.error().is_none(), "command failed: {:?}", cmd.error());
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
}
