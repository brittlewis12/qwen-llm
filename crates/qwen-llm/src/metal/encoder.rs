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
    pub(crate) census_identity: Cell<(usize, u64)>,
    // Always track the binding: census can start after this encoder was bound.
    // Arc cloning costs no allocation or Objective-C message send.
    pub(crate) bound_pipeline_name: std::cell::RefCell<Option<Arc<str>>>,
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
            census_identity: Cell::new(census_encoder_identity()),
            bound_pipeline_name: std::cell::RefCell::new(None),
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

    /// Mark `buf` used for reading by this pass without binding it, so the
    /// command keeps it resident (`MTLComputeCommandEncoder.useResource`).
    pub fn use_resource_read(&self, buf: &Buffer) {
        let resource: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLResource> =
            objc2::runtime::ProtocolObject::from_ref(&**buf);
        self.raw
            .useResource_usage(resource, objc2_metal::MTLResourceUsage::Read);
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

    /// Bind at least `n_bytes`, rounded to Metal's 16-byte allocation alignment.
    pub fn set_threadgroup_memory(&self, index: usize, n_bytes: usize) {
        let n_bytes = n_bytes
            .checked_next_multiple_of(16)
            .expect("threadgroup memory size overflow");
        unsafe {
            self.raw.setThreadgroupMemoryLength_atIndex(n_bytes, index);
        }
    }

    pub fn set_pipeline(&self, pso: &Pipeline) {
        self.raw.setComputePipelineState(&pso.raw);
        *self.bound_pipeline_name.borrow_mut() = Some(Arc::clone(&pso.name));
    }

    pub fn dispatch(&self, grid: MTLSize, threads: MTLSize) {
        kernel_trace_record_dispatch();
        census_record_dispatch(self, grid, threads);
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

    // These are real Metal dispatches, deliberately bypassing encode_* helpers
    // so pipeline lookup order and binding order can differ.
    const CENSUS_ADD: &str = "kernel_add_f32";
    const CENSUS_MUL: &str = "kernel_mul_f32";
    const CENSUS_SIZE: MTLSize = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    struct CensusCapture;

    impl CensusCapture {
        fn begin() -> Self {
            dispatch_census_begin();
            Self
        }
    }

    impl Drop for CensusCapture {
        fn drop(&mut self) {
            dispatch_census_take();
        }
    }

    fn census_buffers(ctx: &MetalContext) -> [Buffer; 4] {
        [2.0f32, 3.0, 0.0, 0.0].map(|v| ctx.buffer_from(bytemuck::bytes_of(&v)).unwrap())
    }

    fn census_bind_buffers(enc: &KernelEncoder, buffers: &[Buffer; 4], output: usize) {
        enc.set_bytes(0, &1u32);
        enc.set_buffer(1, &buffers[0], 0);
        enc.set_buffer(2, &buffers[1], 0);
        enc.set_buffer(3, &buffers[output], 0);
    }

    fn census_check_outputs(buffers: &[Buffer; 4]) {
        // Called only after successful command-buffer completion.
        let values = [2, 3].map(|i| unsafe { *buffers[i].contents().as_ptr().cast::<f32>() });
        assert_eq!(values, [5.0, 6.0]);
    }

    #[test]
    fn census_follows_bound_pso_after_preflight_and_cache_hits() {
        for prewarm in [false, true] {
            let Some(ctx) = crate::metal::test_support::metal_test_context() else {
                return;
            };
            if prewarm {
                ctx.pipeline(CENSUS_ADD).unwrap();
                ctx.pipeline(CENSUS_MUL).unwrap();
            }
            ctx.set_pipeline_cache_metrics_enabled(true);
            let buffers = census_buffers(&ctx);
            let _capture = CensusCapture::begin();
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            // A/B preflight must not label the first (A) dispatch as B.
            let add = ctx.pipeline(CENSUS_ADD).unwrap();
            let mul = ctx.pipeline(CENSUS_MUL).unwrap();
            enc.set_pipeline(&add);
            // Legacy explicit tracing calls must not override the actual PSO either.
            census_record_pso(CENSUS_MUL);
            // Even a failed lookup after binding cannot change attribution.
            assert!(matches!(
                ctx.pipeline("census_missing_kernel"),
                Err(MetalError::NoFunction(_))
            ));
            census_bind_buffers(&enc, &buffers, 2);
            enc.dispatch(CENSUS_SIZE, CENSUS_SIZE);
            enc.set_pipeline(&mul);
            census_bind_buffers(&enc, &buffers, 3);
            enc.dispatch(CENSUS_SIZE, CENSUS_SIZE);
            // An unbound cache hit also cannot relabel the current binding.
            let cached_add = ctx.pipeline(CENSUS_ADD).unwrap();
            assert!(Arc::ptr_eq(&add.name, &cached_add.name));
            enc.dispatch(CENSUS_SIZE, CENSUS_SIZE);
            enc.end();
            let rows = dispatch_census_take();
            cmd.commit();
            wait_completed(&cmd).unwrap();
            census_check_outputs(&buffers);
            assert_eq!(
                rows.iter().map(|r| r.kernel.as_str()).collect::<Vec<_>>(),
                [CENSUS_ADD, CENSUS_MUL, CENSUS_MUL]
            );
            assert!(
                rows.iter()
                    .all(|r| r.encoder_ordinal == 0 && !r.encoder_concurrent)
            );
            assert!(rows.iter().all(|r| r.grid_tgs == 1 && r.tg_threads == 1));
            // One additional miss is the deliberately missing function.
            assert_eq!(
                ctx.pipeline_cache_metrics().misses,
                if prewarm { 1 } else { 3 }
            );
        }
    }

    #[test]
    fn census_interleaved_encoders_keep_their_bindings_and_ordinals() {
        let Some(ctx) = crate::metal::test_support::metal_test_context() else {
            return;
        };
        let buffers = census_buffers(&ctx);
        let add = ctx.pipeline(CENSUS_ADD).unwrap();
        let mul = ctx.pipeline(CENSUS_MUL).unwrap();
        let _capture = CensusCapture::begin();
        // Separate command buffers permit both encoders to remain open.
        let cmd_a = ctx.queue.commandBuffer().unwrap();
        let cmd_b = ctx.queue.commandBuffer().unwrap();
        let a = KernelEncoder::begin(&cmd_a);
        a.set_pipeline(&add);
        census_bind_buffers(&a, &buffers, 2);
        let b = KernelEncoder::begin_concurrent(&cmd_b);
        b.set_pipeline(&mul);
        census_bind_buffers(&b, &buffers, 3);
        a.dispatch(CENSUS_SIZE, CENSUS_SIZE);
        b.dispatch(CENSUS_SIZE, CENSUS_SIZE);
        a.dispatch(CENSUS_SIZE, CENSUS_SIZE);
        a.end();
        b.end();
        let rows = dispatch_census_take();
        cmd_a.commit();
        cmd_b.commit();
        wait_completed(&cmd_a).unwrap();
        wait_completed(&cmd_b).unwrap();
        census_check_outputs(&buffers);
        assert_eq!(
            rows.iter()
                .map(|r| (r.kernel.as_str(), r.encoder_ordinal, r.encoder_concurrent))
                .collect::<Vec<_>>(),
            [
                (CENSUS_ADD, 0, false),
                (CENSUS_MUL, 1, true),
                (CENSUS_ADD, 0, false)
            ]
        );
    }

    #[test]
    fn census_tracks_bindings_before_capture_and_across_restart() {
        let Some(ctx) = crate::metal::test_support::metal_test_context() else {
            return;
        };
        let buffers = census_buffers(&ctx);
        let add = ctx.pipeline(CENSUS_ADD).unwrap();
        let mul = ctx.pipeline(CENSUS_MUL).unwrap();
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        enc.set_pipeline(&add);
        census_bind_buffers(&enc, &buffers, 2);
        enc.dispatch(CENSUS_SIZE, CENSUS_SIZE);
        assert!(!dispatch_census_is_active());
        assert!(dispatch_census_tag_scope(|| panic!("inactive tag evaluated")).is_none());
        assert!(dispatch_census_take().is_empty());
        let _capture = CensusCapture::begin();
        enc.dispatch(CENSUS_SIZE, CENSUS_SIZE);
        let rows = dispatch_census_take();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            (rows[0].kernel.as_str(), rows[0].encoder_ordinal),
            (CENSUS_ADD, 0)
        );

        // Change the binding while the observer is off, then resume capture.
        enc.set_pipeline(&mul);
        census_bind_buffers(&enc, &buffers, 3);
        enc.dispatch(CENSUS_SIZE, CENSUS_SIZE);
        assert!(dispatch_census_take().is_empty());
        dispatch_census_begin();
        let other_cmd = ctx.queue.commandBuffer().unwrap();
        let other = KernelEncoder::begin(&other_cmd);
        other.set_pipeline(&add);
        census_bind_buffers(&other, &buffers, 2);
        other.dispatch(CENSUS_SIZE, CENSUS_SIZE);
        enc.dispatch(CENSUS_SIZE, CENSUS_SIZE);
        let rows = dispatch_census_take();
        assert_eq!(
            rows.iter()
                .map(|r| (r.kernel.as_str(), r.encoder_ordinal))
                .collect::<Vec<_>>(),
            [(CENSUS_ADD, 0), (CENSUS_MUL, 1)]
        );
        enc.end();
        other.end();
        cmd.commit();
        other_cmd.commit();
        wait_completed(&cmd).unwrap();
        wait_completed(&other_cmd).unwrap();
        census_check_outputs(&buffers);
    }

    #[test]
    fn kernel_encoder_drop_closes_validation_error_pass() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
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
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
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
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
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
