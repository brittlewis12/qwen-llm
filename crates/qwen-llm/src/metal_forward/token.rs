//! Single-token forward, logits, argmax, and capture/intervention hooks.

use super::*;

/// Install capture slots: (ffn_call_index, h_dst, inner_dst) triples.
pub fn t9_ffn_capture_install(slots: Vec<(usize, MetalTensor, MetalTensor)>) {
    T9_FFN_CAPTURE.with(|c| *c.borrow_mut() = Some(slots));
    T9_FFN_CALL_IDX.with(|c| c.set(0));
}

/// Reset the per-token FFN call counter (call before every token).
pub fn t9_ffn_capture_reset_token() {
    T9_FFN_CALL_IDX.with(|c| c.set(0));
}

/// Uninstall capture.
pub fn t9_ffn_capture_uninstall() {
    T9_FFN_CAPTURE.with(|c| *c.borrow_mut() = None);
}

pub(super) fn t9_ffn_capture_slots_for_current_call() -> Option<(MetalTensor, MetalTensor)> {
    T9_FFN_CAPTURE.with(|c| {
        let borrow = c.borrow();
        let slots = borrow.as_ref()?;
        let idx = T9_FFN_CALL_IDX.with(|i| {
            let v = i.get();
            i.set(v + 1);
            v
        });
        slots
            .iter()
            .find(|(want, _, _)| *want == idx)
            .map(|(_, h, inner)| (h.clone(), inner.clone()))
    })
}

#[cfg(test)]
pub(super) fn capture_metal_load_lines<T>(run: impl FnOnce() -> T) -> (T, Vec<String>) {
    struct CaptureGuard;

    impl Drop for CaptureGuard {
        fn drop(&mut self) {
            METAL_LOAD_TEST_LINES.with(|lines| {
                lines.borrow_mut().take();
            });
        }
    }

    METAL_LOAD_TEST_LINES.with(|lines| {
        assert!(lines.borrow().is_none(), "nested Metal load-line capture");
        *lines.borrow_mut() = Some(Vec::new());
    });
    let guard = CaptureGuard;
    let value = run();
    let lines = METAL_LOAD_TEST_LINES.with(|lines| {
        lines
            .borrow_mut()
            .take()
            .expect("Metal load-line capture disappeared")
    });
    drop(guard);
    (value, lines)
}

pub(super) fn lm_head_tail_error(detail: impl Into<String>) -> MfError {
    MfError::Metal(MetalError::BadShape {
        kernel: "lm_head_tail",
        detail: detail.into(),
    })
}

pub(super) fn encode_argmax_reduction(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    output: &MetalTensor,
    n_rows: usize,
    vocab: usize,
    reduction: ArgmaxReduction,
) -> Result<(), MetalError> {
    match reduction {
        ArgmaxReduction::SpeculativeLowest => {
            encode_argmax_f32(ctx, enc, logits, output, n_rows, vocab)
        }
        ArgmaxReduction::GreedyTotal => {
            encode_argmax_f32_greedy(ctx, enc, logits, output, n_rows, vocab)
        }
    }
}

pub(super) fn capture_parallel_copy_usage() -> Result<ParallelCopyUsage, MfError> {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy getrusage failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    let usage = unsafe { usage.assume_init() };
    Ok(ParallelCopyUsage {
        minor_faults: usage.ru_minflt,
        major_faults: usage.ru_majflt,
        user_time_us: parallel_copy_timeval_us(usage.ru_utime)?,
        system_time_us: parallel_copy_timeval_us(usage.ru_stime)?,
    })
}

pub(super) fn capture_parallel_copy_proc_usage() -> Result<ParallelCopyProcUsage, MfError> {
    let mut usage = MaybeUninit::<libc::rusage_info_v4>::zeroed();
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            usage.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    if rc != 0 {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy proc_pid_rusage v4 failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    let usage = unsafe { usage.assume_init() };
    Ok(ParallelCopyProcUsage {
        instructions: usage.ri_instructions,
        cycles: usage.ri_cycles,
    })
}

pub(super) fn validate_post_block_intervention_tensor(
    session: &MetalSession,
    tensor: &MetalTensor,
    hidden_size: usize,
    op_index: usize,
    role: &str,
) -> Result<(), MfError> {
    let bytes = hidden_size
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            MfError::Metal(MetalError::BadShape {
                kernel: "single_token_with_post_block_interventions",
                detail: format!("intervention {op_index} {role} byte count overflow"),
            })
        })?;
    let end = tensor.offset.checked_add(bytes as u64).ok_or_else(|| {
        MfError::Metal(MetalError::BadShape {
            kernel: "single_token_with_post_block_interventions",
            detail: format!("intervention {op_index} {role} endpoint overflow"),
        })
    })?;
    if tensor.dtype != GgmlType::F32
        || tensor.shape != [hidden_size as u64]
        || tensor.n_elements() as usize != hidden_size
        || !tensor
            .offset
            .is_multiple_of(std::mem::align_of::<f32>() as u64)
        || end > tensor.buffer.length() as u64
    {
        return Err(MfError::Metal(MetalError::BadShape {
            kernel: "single_token_with_post_block_interventions",
            detail: format!(
                "intervention {op_index} {role} must be aligned F32 [{hidden_size}], got {:?} {:?} offset={} buffer_bytes={}",
                tensor.dtype,
                tensor.shape,
                tensor.offset,
                tensor.buffer.length()
            ),
        }));
    }
    if session.aliases_mutable_buffer(tensor) {
        return Err(MfError::Metal(MetalError::BadShape {
            kernel: "single_token_with_post_block_interventions",
            detail: format!("intervention {op_index} {role} aliases mutable session storage"),
        }));
    }
    Ok(())
}

pub(super) fn validate_hidden_capture_destination(
    site: &str,
    destination: &MetalTensor,
    expected_shape: &[u64],
    expected_bytes: usize,
) -> Result<(u64, u64), MfError> {
    if destination.dtype != GgmlType::F32
        || !destination.is_writable()
        || destination.shape != expected_shape
        || !destination
            .offset
            .is_multiple_of(std::mem::align_of::<f32>() as u64)
    {
        return Err(MfError::Metal(MetalError::BadShape {
            kernel: "single_token_with_hidden_sites",
            detail: format!(
                "{site} destination must be writable aligned F32 {expected_shape:?}, got {:?} {:?} writable={} offset={}",
                destination.dtype,
                destination.shape,
                destination.is_writable(),
                destination.offset
            ),
        }));
    }
    let expected_bytes = u64::try_from(expected_bytes).map_err(|_| {
        MfError::Metal(MetalError::BadShape {
            kernel: "single_token_with_hidden_sites",
            detail: format!("{site} byte count does not fit u64"),
        })
    })?;
    let end = destination
        .offset
        .checked_add(expected_bytes)
        .ok_or_else(|| {
            MfError::Metal(MetalError::BadShape {
                kernel: "single_token_with_hidden_sites",
                detail: format!("{site} destination endpoint overflow"),
            })
        })?;
    if end > destination.buffer.length() as u64 {
        return Err(MfError::Metal(MetalError::BadShape {
            kernel: "single_token_with_hidden_sites",
            detail: format!(
                "{site} destination range [{}..{end}) exceeds buffer length {}",
                destination.offset,
                destination.buffer.length()
            ),
        }));
    }
    Ok((destination.offset, end))
}

pub(super) fn capture_ranges_overlap(
    left: &MetalTensor,
    left_range: (u64, u64),
    right: &MetalTensor,
    right_range: (u64, u64),
) -> bool {
    Retained::as_ptr(&left.buffer) == Retained::as_ptr(&right.buffer)
        && left_range.0 < right_range.1
        && right_range.0 < left_range.1
}

/// Per-token timing profile.
///
/// * `cpu_encode_ms` — time spent in `KernelEncoder::begin` through
///   `enc.end()`. This is the CPU-side cost of encoding all dispatches
///   into the command buffer. ICB will collapse this to ~0.
/// * `gpu_kernel_ms` — `GPUEndTime - GPUStartTime`, the wall-clock the
///   GPU spent actually executing kernels. This is the floor a
///   correctness-preserving optimization can reach.
/// * `cpu_to_gpu_complete_ms` — `commit() + waitUntilCompleted()` wall
///   clock. Difference vs `gpu_kernel_ms` is mostly driver/queue
///   submission + completion handler overhead.
/// * `total_ms` — the user-visible per-token latency (incl. logits
///   readback).
#[derive(Debug, Clone, Copy)]
pub struct TokenProfile {
    pub cpu_encode_ms: f64,
    pub cpu_to_gpu_complete_ms: f64,
    pub gpu_kernel_ms: f64,
    pub total_ms: f64,
    pub moe_cpu_route_ms: f64,
    pub moe_cmd_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct LogitsReadbackProfile {
    pub timer_spans: u32,
    pub bytes: usize,
    pub allocation_zero_fill_ms: f64,
    pub copy_ms: f64,
}

pub type SampledStructuralOutcome = (
    Result<(SampledToken, BoundedTopKEvidence), SamplingError>,
    TokenProfile,
    StructuralRowEvidence,
);

pub(super) struct ValidatedSharedLogits {
    pub(super) source: std::ptr::NonNull<f32>,
    pub(super) len: usize,
}

impl<'a> MetalForward<'a> {
    pub fn new(ctx: &'a MetalContext, model: &'a MetalModel) -> Self {
        Self { ctx, model }
    }

    /// Run a single token through the model. Encodes all kernels into
    /// one command buffer, commits, waits, reads back logits.
    ///
    /// `position` is the 0-indexed sequence position (used by RoPE for
    /// the full-attn layers; ignored by GDN layers).
    pub fn single_token(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<Vec<f32>, MfError> {
        session.ensure_usable()?;
        if self.model.arch.kind == ArchKind::Moe {
            return self.single_token_moe(token_id, position, session);
        }
        let (logits, _) = self.single_token_profiled(token_id, position, session)?;
        Ok(logits)
    }

    pub(super) fn resident_lm_head_tail_evidence(
        &self,
        session: &MetalSession,
    ) -> LmHeadTailEvidence {
        LmHeadTailEvidence {
            kind: LmHeadTailKind::Resident,
            n_in: self.model.arch.hidden_size as usize,
            n_out: self.model.arch.vocab_size as usize,
            weight_offset: self.model.lm_head.offset,
            output_offset: session.logits.offset,
            tail_dispatches: 1,
            full_head_dispatches: 1,
            command_completed: false,
            command_error_none: false,
        }
    }

    pub(crate) fn validate_lm_head_tail(
        &self,
        session: &MetalSession,
        tail: LmHeadTail<'_>,
    ) -> Result<LmHeadTailEvidence, MfError> {
        let h = self.model.arch.hidden_size as usize;
        let vocab = self.model.arch.vocab_size as usize;
        match tail {
            LmHeadTail::Resident => {
                if self.model.lm_head.shape.as_slice() != [h as u64, vocab as u64] {
                    return Err(lm_head_tail_error("resident head shape mismatch"));
                }
                if session.logits.dtype != GgmlType::F32
                    || session.logits.shape.as_slice() != [vocab as u64]
                    || !session.logits.is_writable()
                {
                    return Err(lm_head_tail_error("resident logits shape mismatch"));
                }
                let output_bytes = vocab
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| lm_head_tail_error("resident logits bytes overflow"))?;
                checked_tail_range(&session.logits, output_bytes, "resident logits")?;
                Ok(self.resident_lm_head_tail_evidence(session))
            }
            LmHeadTail::CompactQ6K { weight, output } => {
                if weight.dtype != GgmlType::Q6_K
                    || weight.shape.len() != 2
                    || weight.shape[0] != h as u64
                    || weight.shape[1] == 0
                    || weight.shape[1] > 17
                    || weight.provenance() != MetalTensorProvenance::OwnedWeightReadOnly
                    || weight.is_writable()
                {
                    return Err(lm_head_tail_error("compact Q6_K weight contract mismatch"));
                }
                let n_out = usize::try_from(weight.shape[1])
                    .map_err(|_| lm_head_tail_error("compact width does not fit usize"))?;
                if !h.is_multiple_of(256) {
                    return Err(lm_head_tail_error(
                        "compact input width is not Q6_K block aligned",
                    ));
                }
                if output.dtype != GgmlType::F32
                    || output.shape.as_slice() != [n_out as u64]
                    || !output.is_writable()
                    || output.offset % std::mem::align_of::<f32>() as u64 != 0
                    || weight.offset % 32 != 0
                {
                    return Err(lm_head_tail_error("compact output contract mismatch"));
                }
                if session.h.dtype != GgmlType::F32
                    || session.h.shape.as_slice() != [h as u64]
                    || !session.h.is_writable()
                    || !session
                        .h
                        .offset
                        .is_multiple_of(std::mem::align_of::<f32>() as u64)
                    || session.x.dtype != GgmlType::F32
                    || session.x.shape.as_slice() != [h as u64]
                    || !session.x.is_writable()
                    || !session
                        .x
                        .offset
                        .is_multiple_of(std::mem::align_of::<f32>() as u64)
                {
                    return Err(lm_head_tail_error("session hidden contract mismatch"));
                }
                if session.aliases_mutable_buffer(weight) {
                    return Err(lm_head_tail_error(
                        "compact weight aliases mutable session storage",
                    ));
                }
                if session.aliases_mutable_buffer(output) {
                    return Err(lm_head_tail_error(
                        "compact output aliases mutable session storage",
                    ));
                }
                let weight_bytes = h
                    .checked_div(256)
                    .and_then(|blocks| blocks.checked_mul(210))
                    .and_then(|row| row.checked_mul(n_out))
                    .ok_or_else(|| lm_head_tail_error("compact weight bytes overflow"))?;
                let output_bytes = n_out
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| lm_head_tail_error("compact output bytes overflow"))?;
                let hidden_bytes = h
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| lm_head_tail_error("hidden bytes overflow"))?;
                let weight_range = checked_tail_range(weight, weight_bytes, "compact weight")?;
                let output_range = checked_tail_range(output, output_bytes, "compact output")?;
                let hidden_range = checked_tail_range(&session.h, hidden_bytes, "session.h")?;
                let residual_range = checked_tail_range(&session.x, hidden_bytes, "session.x")?;
                if tail_ranges_overlap(weight, weight_range, output, output_range)
                    || tail_ranges_overlap(&session.h, hidden_range, output, output_range)
                    || tail_ranges_overlap(&session.x, residual_range, output, output_range)
                {
                    return Err(lm_head_tail_error("compact tail buffers overlap"));
                }
                Ok(LmHeadTailEvidence {
                    kind: LmHeadTailKind::CompactQ6K,
                    n_in: h,
                    n_out,
                    weight_offset: weight.offset,
                    output_offset: output.offset,
                    tail_dispatches: 1,
                    full_head_dispatches: 0,
                    command_completed: false,
                    command_error_none: false,
                })
            }
        }
    }

    pub(crate) fn encode_lm_head_tail(
        &self,
        enc: &KernelEncoder,
        session: &MetalSession,
        tail: LmHeadTail<'_>,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        match tail {
            LmHeadTail::Resident => encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                h,
                self.model.arch.vocab_size as usize,
            )?,
            LmHeadTail::CompactQ6K { weight, output } => encode_mat_vec_dispatch(
                self.ctx,
                enc,
                weight,
                &session.h,
                output,
                h,
                usize::try_from(weight.shape[1])
                    .map_err(|_| lm_head_tail_error("compact width does not fit usize"))?,
            )?,
        }
        Ok(())
    }

    /// Run the production concurrent-MoE full-logit transition with opt-in
    /// attribution around only the existing host destination allocation and
    /// Shared-buffer copy.
    pub fn single_token_sampled_attribution(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile, LogitsReadbackProfile), MfError> {
        if !concurrent_gdn_moe_decode_enabled() {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "sampling_attribution",
                detail: "requires the production concurrent-GDN MoE decode path".into(),
            }));
        }
        let (mut profile, evidence, t_total) = self
            .single_token_profiled_concurrent_gdn_moe_tail_inner(
                token_id,
                position,
                session,
                LmHeadTail::Resident,
                false,
            )?;
        debug_assert_eq!(evidence.kind, LmHeadTailKind::Resident);

        let allocation_t0 = std::time::Instant::now();
        let mut out = vec![0.0f32; self.model.arch.vocab_size as usize];
        let allocation_zero_fill_ms = allocation_t0.elapsed().as_secs_f64() * 1e3;
        if session.logits.dtype != GgmlType::F32 || session.logits.n_elements() < out.len() as u64 {
            return Err(lm_head_tail_error(
                "sampled logits readback requires a complete F32 logits row",
            ));
        }
        let bytes = out
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| lm_head_tail_error("sampled logits byte size overflow"))?;
        let (source_start, _) =
            checked_tail_range(&session.logits, bytes, "sampled logits readback")?;
        if source_start % std::mem::align_of::<f32>() != 0 {
            return Err(lm_head_tail_error(
                "sampled logits source is not aligned for F32 readback",
            ));
        }

        let copy_t0 = std::time::Instant::now();
        unsafe {
            let src = (session.logits.buffer.contents().as_ptr() as *const u8).add(source_start)
                as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let copy_ms = copy_t0.elapsed().as_secs_f64() * 1e3;
        profile.total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            out,
            profile,
            LogitsReadbackProfile {
                timer_spans: 2,
                bytes,
                allocation_zero_fill_ms,
                copy_ms,
            },
        ))
    }

    /// Validate the architecture and decode organization required by scoped
    /// resident-logit sampling before a request starts prefill.
    pub fn ensure_sampled_structural_supported(&self) -> Result<(), MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Moe
            || arch.n_layer != 40
            || arch.hidden_size != 2_048
            || arch.vocab_size != 248_320
            || arch.n_q_heads != 16
            || arch.n_kv_heads != 2
            || arch.attn_head_dim != 256
            || arch.full_attention_interval != 4
            || arch.partial_rotary_factor.to_bits() != 0.25f32.to_bits()
            || arch.gdn_n_k_heads != 16
            || arch.gdn_n_v_heads != 32
            || arch.gdn_head_dim != 128
            || arch.gdn_conv_kernel != 4
            || arch.expert_count != 256
            || arch.expert_used_count != 8
            || arch.expert_feed_forward_length != 512
            || arch.expert_shared_feed_forward_length != 512
            || arch.mtp_n_hidden_layers != 0
            || self.model.blocks.len() != arch.n_layer as usize
            || self.model.lm_head.dtype != GgmlType::Q6_K
            || self.model.lm_head.shape.as_slice() != [2_048, 248_320]
        {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "sampled_structural",
                detail: "requires the frozen Qwen3.6 35B A3B resident-head geometry".into(),
            }));
        }
        if !concurrent_gdn_moe_decode_enabled() {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "sampled_structural",
                detail: "requires the production concurrent-GDN MoE decode path".into(),
            }));
        }
        Ok(())
    }

    /// Validate the exact session-row contract before a flagged request starts
    /// prompt prefill.
    pub fn ensure_sampled_structural_session_supported(
        &self,
        session: &MetalSession,
    ) -> Result<(), MfError> {
        self.ensure_sampled_structural_supported()?;
        self.validate_sampled_structural_logits(session)?;
        Ok(())
    }

    /// Execute the production concurrent-MoE transition and expose its
    /// synchronized Shared logits row only to the bounded CPU sampler.
    pub fn single_token_sampled_structural(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        sampler: &mut Sampler,
    ) -> Result<SampledStructuralOutcome, MfError> {
        self.single_token_sampled_structural_scoped(token_id, position, session, |row| {
            sampler.sample_bounded_top_k(row)
        })
    }

    pub(super) fn single_token_sampled_structural_scoped<R, F>(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        consume: F,
    ) -> Result<(R, TokenProfile, StructuralRowEvidence), MfError>
    where
        F: for<'row> FnOnce(&'row [f32]) -> R,
    {
        self.ensure_sampled_structural_supported()?;
        let (mut profile, evidence, t_total) = self
            .single_token_profiled_concurrent_gdn_moe_tail_inner(
                token_id,
                position,
                session,
                LmHeadTail::Resident,
                false,
            )?;
        debug_assert_eq!(evidence.kind, LmHeadTailKind::Resident);
        let mut structural_evidence = StructuralRowEvidence {
            resident_head_wait_calls: 1,
            ..StructuralRowEvidence::default()
        };
        let validated = self.validate_sampled_structural_logits(session)?;
        structural_evidence.validated_shared_row_calls = 1;
        let row = unsafe { std::slice::from_raw_parts(validated.source.as_ptr(), validated.len) };
        let result = consume(row);
        profile.total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((result, profile, structural_evidence))
    }

    pub(super) fn validate_sampled_structural_logits(
        &self,
        session: &MetalSession,
    ) -> Result<ValidatedSharedLogits, MfError> {
        let vocab = usize::try_from(self.model.arch.vocab_size)
            .map_err(|_| lm_head_tail_error("sampled logits vocabulary does not fit usize"))?;
        if session.logits.dtype != GgmlType::F32
            || session.logits.shape != [self.model.arch.vocab_size as u64]
            || session.logits.provenance() != MetalTensorProvenance::OwnedWritable
            || session.logits.buffer.storageMode() != MTLStorageMode::Shared
        {
            return Err(lm_head_tail_error(
                "sampled structural logits require exact writable Shared F32 [vocab] storage",
            ));
        }
        let bytes = vocab
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| lm_head_tail_error("sampled structural logits byte size overflow"))?;
        let (source_start, _) =
            checked_tail_range(&session.logits, bytes, "sampled structural logits")?;
        let base = std::ptr::NonNull::new(session.logits.buffer.contents().as_ptr() as *mut u8)
            .ok_or_else(|| {
                lm_head_tail_error("sampled structural logits have null host contents")
            })?;
        let source = unsafe { base.as_ptr().add(source_start) };
        if !(source as usize).is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(lm_head_tail_error(
                "sampled structural logits source is not aligned for F32 access",
            ));
        }
        let source = std::ptr::NonNull::new(source.cast::<f32>())
            .ok_or_else(|| lm_head_tail_error("sampled structural logits have null F32 source"))?;
        Ok(ValidatedSharedLogits { source, len: vocab })
    }

    pub fn single_token_argmax(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<i32, MfError> {
        session.ensure_usable()?;
        let (argmax, _) = self.single_token_argmax_profiled(token_id, position, session)?;
        Ok(argmax)
    }

    pub fn single_token_greedy(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<GreedySelection, MfError> {
        session.ensure_usable()?;
        let (raw, _) = self.single_token_reduced_profiled(
            token_id,
            position,
            session,
            ArgmaxReduction::GreedyTotal,
        )?;
        Ok(GreedySelection::from_encoded(raw))
    }

    pub(super) fn single_token_argmax_profiled_dense_serial(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        reduction: ArgmaxReduction,
    ) -> Result<(i32, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }

        let t_total = std::time::Instant::now();
        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }
        let ids_buf = session.ids_buf.clone();
        let argmax_tok = session.argmax_tok.clone();

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        self.encode_single_token_argmax_dense_with_reduction(
            &enc,
            position,
            session,
            &ids_buf,
            &argmax_tok,
            reduction,
        )?;

        enc.end();
        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;

        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        crate::metal::wait_completed(&cmd_buf)?;
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let argmax = unsafe {
            let src = argmax_tok.buffer.contents().as_ptr() as *const i32;
            *src
        };
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            argmax,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    pub fn single_token_argmax_profiled(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(i32, TokenProfile), MfError> {
        session.ensure_usable()?;
        self.single_token_reduced_profiled(
            token_id,
            position,
            session,
            ArgmaxReduction::SpeculativeLowest,
        )
    }

    pub(super) fn single_token_reduced_profiled(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        reduction: ArgmaxReduction,
    ) -> Result<(i32, TokenProfile), MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return if concurrent_gdn_moe_decode_enabled() {
                self.single_token_argmax_profiled_concurrent_gdn_moe_with_reduction(
                    token_id, position, session, reduction,
                )
            } else {
                self.single_token_argmax_profiled_moe(token_id, position, session, reduction)
            };
        }
        if concurrent_gdn_dense_decode_enabled() {
            return self.single_token_argmax_profiled_concurrent_gdn_dense_with_reduction(
                token_id, position, session, reduction,
            );
        }
        self.single_token_argmax_profiled_dense_serial(token_id, position, session, reduction)
    }

    pub fn encode_single_token_argmax(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
    ) -> Result<(), MfError> {
        self.encode_single_token_argmax_with_reduction(
            enc,
            position,
            session,
            ids_buf,
            argmax_tok,
            ArgmaxReduction::SpeculativeLowest,
        )
    }

    pub(crate) fn encode_single_token_greedy(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
    ) -> Result<(), MfError> {
        session.ensure_usable()?;
        self.encode_single_token_argmax_with_reduction(
            enc,
            position,
            session,
            ids_buf,
            argmax_tok,
            ArgmaxReduction::GreedyTotal,
        )
    }

    pub(super) fn encode_single_token_argmax_with_reduction(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
        reduction: ArgmaxReduction,
    ) -> Result<(), MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return self.encode_single_token_argmax_moe_with_reduction(
                enc, position, session, ids_buf, argmax_tok, reduction,
            );
        }
        self.encode_single_token_argmax_dense_with_reduction(
            enc, position, session, ids_buf, argmax_tok, reduction,
        )
    }

    pub(super) fn encode_single_token_argmax_dense_with_reduction(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
        reduction: ArgmaxReduction,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;

        encode_get_rows_f32(
            self.ctx,
            enc,
            &self.model.token_embd,
            ids_buf,
            &session.x,
            1,
            arch.hidden_size as usize,
        )?;

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
        }

        encode_rms_norm_mul_f32(
            self.ctx,
            enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            arch.hidden_size as usize,
            arch.vocab_size as usize,
        )?;
        encode_argmax_reduction(
            self.ctx,
            enc,
            &session.logits,
            argmax_tok,
            1,
            arch.vocab_size as usize,
            reduction,
        )?;
        Ok(())
    }

    /// Same as [`single_token`] but ALSO copies the hidden state for the MTP
    /// carry into `hidden_dst`. By default this is the pre-output_norm residual;
    /// `post_norm_hidden` instead captures the final RMSNorm output.
    ///
    /// `hidden_dst` must be a zero-copy F32 tensor of shape `[H]`. It's
    /// kept GPU-resident so the next MTP draft call can consume it
    /// without a CPU readback. The copy happens inside the same command
    /// buffer as the forward, so `hidden_dst` is up-to-date by the time
    /// this call returns (which commits + waits).
    ///
    /// Caller must ensure `hidden_dst` is not aliased with any tensor
    /// the next forward call writes (typically allocate it as part of
    /// the MTP session arena).
    /// Multi-layer hidden-state capture variant of [`single_token`].
    /// Captures the residual stream `s.x` (post-FFN, post-residual) at
    /// each layer index in `target_layer_ids`, writing into the
    /// caller-supplied `hidden_dst` of shape `[K · H]` where
    /// `K = target_layer_ids.len()`.
    ///
    /// Captured layout: `hidden_dst[k * H .. (k+1) * H]` holds the
    /// residual after `target_layer_ids[k]` runs (in the same K order
    /// the caller specified, NOT sorted by layer index).
    ///
    /// All scatters happen inside the same command buffer as the
    /// forward, so `hidden_dst` is up-to-date by the time this returns.
    /// `target_layer_ids` may be empty (degenerate case: produces no
    /// hidden capture; equivalent to `single_token`).
    ///
    /// Used by H5 to bootstrap `target_ctx` from prompt prefill: per
    /// docs/H5-DFLASH.md §1.1, the DFlash drafter consumes K=5 target
    /// layer hiddens fused via `dflash_fc`. Caller is responsible for
    /// stacking these K hiddens into the final
    /// `[K · H_target, ctx_len]` cross-context buffer.
    pub fn single_token_with_multi_hidden(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        target_layer_ids: &[u32],
        hidden_dst: &MetalTensor,
    ) -> Result<Vec<f32>, MfError> {
        self.single_token_with_hidden_sites(
            token_id,
            position,
            session,
            target_layer_ids,
            Some(hidden_dst),
            None,
            &[],
            false,
            true,
        )
    }

    /// Capture both sides of selected dense FFN residual updates in one
    /// ordinary-Qwen token forward.
    ///
    /// `pre_ffn_dst[k]` receives the post-mixer residual immediately before
    /// post-attention RMSNorm. `post_block_dst[k]` receives the residual after
    /// the FFN update. Both destinations use caller layer order and shape
    /// `[H, K]`.
    pub fn single_token_with_dense_ffn_capture(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        target_layer_ids: &[u32],
        pre_ffn_dst: &MetalTensor,
        post_block_dst: &MetalTensor,
    ) -> Result<Vec<f32>, MfError> {
        self.single_token_with_hidden_sites(
            token_id,
            position,
            session,
            target_layer_ids,
            Some(post_block_dst),
            Some(pre_ffn_dst),
            &[],
            false,
            true,
        )
    }

    /// Capture both sides of selected dense FFN residual updates while
    /// applying caller-ordered interventions to `session.x` after the
    /// corresponding block and before the post-block capture.
    ///
    /// Each direction/source/target tensor must be a separate, aligned F32
    /// tensor of shape `[H]`, and must not alias mutable session storage.
    /// Operations at one layer are encoded in the order supplied by
    /// `interventions`. An empty slice is exactly the ordinary capture path.
    pub fn single_token_with_dense_ffn_capture_and_interventions(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        target_layer_ids: &[u32],
        pre_ffn_dst: &MetalTensor,
        post_block_dst: &MetalTensor,
        interventions: &[PostBlockIntervention<'_>],
    ) -> Result<Vec<f32>, MfError> {
        self.single_token_with_hidden_sites(
            token_id,
            position,
            session,
            target_layer_ids,
            Some(post_block_dst),
            Some(pre_ffn_dst),
            interventions,
            false,
            true,
        )
    }

    /// Serial full-logit forward for ordinary dense and ordinary MoE models
    /// with caller-ordered interventions after selected blocks. Captured
    /// post-block residuals are written to `hidden_dst[k * H .. (k + 1) * H]`
    /// for `target_layer_ids[k]`, in caller order.
    ///
    /// This deliberately uses the serial block encoders, including for MoE;
    /// it does not route through concurrent, packed-prefill, or argmax-only
    /// paths. An empty `interventions` slice is the serial post-block capture
    /// path without intervention.
    pub fn single_token_with_post_block_interventions(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        target_layer_ids: &[u32],
        hidden_dst: &MetalTensor,
        interventions: &[PostBlockIntervention<'_>],
    ) -> Result<Vec<f32>, MfError> {
        self.single_token_with_hidden_sites(
            token_id,
            position,
            session,
            target_layer_ids,
            Some(hidden_dst),
            None,
            interventions,
            true,
            true,
        )
    }

    /// Serial full-logit forward for ordinary dense and ordinary MoE models
    /// with post-block interventions but no capture destination.
    pub fn single_token_with_post_block_interventions_no_capture(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        interventions: &[PostBlockIntervention<'_>],
    ) -> Result<Vec<f32>, MfError> {
        self.single_token_with_hidden_sites(
            token_id,
            position,
            session,
            &[],
            None,
            None,
            interventions,
            true,
            true,
        )
    }

    /// Skip-tail counterpart of
    /// [`single_token_with_post_block_interventions`]. Advances ordinary
    /// dense or MoE state, applies caller-ordered post-block interventions,
    /// and captures the requested post-block rows without running final
    /// RMSNorm, the LM head, or logits readback.
    pub fn single_token_with_post_block_interventions_no_tail(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        target_layer_ids: &[u32],
        hidden_dst: &MetalTensor,
        interventions: &[PostBlockIntervention<'_>],
    ) -> Result<(), MfError> {
        self.single_token_with_hidden_sites(
            token_id,
            position,
            session,
            target_layer_ids,
            Some(hidden_dst),
            None,
            interventions,
            true,
            false,
        )?;
        Ok(())
    }

    /// Skip-tail counterpart of
    /// [`single_token_with_post_block_interventions_no_capture`]. Advances
    /// ordinary dense or MoE state and applies caller-ordered post-block
    /// interventions without capture, final RMSNorm, LM-head work, or logits
    /// readback.
    pub fn single_token_with_post_block_interventions_no_capture_no_tail(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        interventions: &[PostBlockIntervention<'_>],
    ) -> Result<(), MfError> {
        self.single_token_with_hidden_sites(
            token_id,
            position,
            session,
            &[],
            None,
            None,
            interventions,
            true,
            false,
        )?;
        Ok(())
    }

    /// Skip-tail variant of [`single_token_with_dense_ffn_capture`]. Captures
    /// both residual sites but does not run final RMSNorm, the LM head, or a
    /// logits readback. The mutable sequence state advances exactly as in the
    /// ordinary token forward.
    pub fn single_token_with_dense_ffn_capture_no_tail(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        target_layer_ids: &[u32],
        pre_ffn_dst: &MetalTensor,
        post_block_dst: &MetalTensor,
    ) -> Result<(), MfError> {
        self.single_token_with_hidden_sites(
            token_id,
            position,
            session,
            target_layer_ids,
            Some(post_block_dst),
            Some(pre_ffn_dst),
            &[],
            false,
            false,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn single_token_with_hidden_sites(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        target_layer_ids: &[u32],
        post_block_dst: Option<&MetalTensor>,
        pre_ffn_dst: Option<&MetalTensor>,
        interventions: &[PostBlockIntervention<'_>],
        allow_moe: bool,
        run_tail: bool,
    ) -> Result<Vec<f32>, MfError> {
        session.ensure_usable()?;
        if self.model.arch.kind == ArchKind::Moe && !allow_moe {
            return Err(MfError::UnsupportedMoe);
        }
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let k = target_layer_ids.len();
        let expected_capture_elements = k.checked_mul(h).ok_or_else(|| {
            MfError::Metal(MetalError::BadShape {
                kernel: "single_token_with_hidden_sites",
                detail: "capture element count overflow".into(),
            })
        })?;
        let dense_capture_shape = vec![h as u64, k as u64];
        let flat_capture_shape = vec![expected_capture_elements as u64];
        let expected_capture_bytes = expected_capture_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                MfError::Metal(MetalError::BadShape {
                    kernel: "single_token_with_hidden_sites",
                    detail: "capture byte count overflow".into(),
                })
            })?;
        let mut post_block_range = None;
        let mut pre_ffn_range = None;
        for (site, destination, range) in [
            ("post_block", post_block_dst, &mut post_block_range),
            ("pre_ffn", pre_ffn_dst, &mut pre_ffn_range),
        ] {
            if let Some(destination) = destination {
                let expected_shape = if pre_ffn_dst.is_some() {
                    &dense_capture_shape
                } else {
                    &flat_capture_shape
                };
                *range = Some(validate_hidden_capture_destination(
                    site,
                    destination,
                    expected_shape,
                    expected_capture_bytes,
                )?);
                if session.aliases_mutable_buffer(destination) {
                    return Err(MfError::Metal(MetalError::BadShape {
                        kernel: "single_token_with_hidden_sites",
                        detail: format!("{site} destination aliases mutable session storage"),
                    }));
                }
            }
        }
        if let (Some(pre_ffn_dst), Some(pre_range), Some(post_block_dst), Some(post_range)) =
            (pre_ffn_dst, pre_ffn_range, post_block_dst, post_block_range)
            && capture_ranges_overlap(pre_ffn_dst, pre_range, post_block_dst, post_range)
        {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "single_token_with_hidden_sites",
                detail: "pre_ffn and post_block destinations overlap".into(),
            }));
        }
        if u32::try_from(expected_capture_elements).is_err() {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "single_token_with_hidden_sites",
                detail: format!(
                    "capture element count {expected_capture_elements} exceeds u32 scatter addressing"
                ),
            }));
        }
        for &lid in target_layer_ids {
            if (lid as usize) >= self.model.blocks.len() {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "single_token_with_multi_hidden.target_layer_ids",
                    detail: format!("layer id {lid} >= n_layer {}", self.model.blocks.len()),
                }));
            }
        }
        for (op_index, intervention) in interventions.iter().enumerate() {
            let (layer, coefficient) = match intervention {
                PostBlockIntervention::Fixed {
                    layer, coefficient, ..
                }
                | PostBlockIntervention::ResidualL2Relative {
                    layer, coefficient, ..
                }
                | PostBlockIntervention::Projection {
                    layer, coefficient, ..
                }
                | PostBlockIntervention::SourceToTarget {
                    layer, coefficient, ..
                } => (*layer, *coefficient),
            };
            if (layer as usize) >= self.model.blocks.len() {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "single_token_with_post_block_interventions",
                    detail: format!(
                        "intervention {op_index} layer {layer} >= n_layer {}",
                        self.model.blocks.len()
                    ),
                }));
            }
            if !coefficient.is_finite() || coefficient == 0.0 {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "single_token_with_post_block_interventions",
                    detail: format!(
                        "intervention {op_index} coefficient must be finite and nonzero, got {coefficient}"
                    ),
                }));
            }
            match intervention {
                PostBlockIntervention::Fixed { direction, .. }
                | PostBlockIntervention::ResidualL2Relative { direction, .. }
                | PostBlockIntervention::Projection { direction, .. } => {
                    validate_post_block_intervention_tensor(
                        session,
                        direction,
                        h,
                        op_index,
                        "direction",
                    )?;
                }
                PostBlockIntervention::SourceToTarget { source, target, .. } => {
                    validate_post_block_intervention_tensor(
                        session, source, h, op_index, "source",
                    )?;
                    validate_post_block_intervention_tensor(
                        session, target, h, op_index, "target",
                    )?;
                }
            }
        }

        // Stage token id.
        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let cmd_buf = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| MfError::CommandBuffer {
                status: "unavailable".into(),
                error: "Metal did not provide a command buffer".into(),
            })?;
        let enc = KernelEncoder::begin(&cmd_buf);

        // Embed → s.x.
        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            h,
        )?;

        // Per-block, capturing at requested layer indices AFTER each
        // block's residual #2 (s.x is the post-FFN residual, exactly
        // matching the CPU `single_token_capture_layers` capture point).
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            let pre_ffn_offsets: Vec<usize> = if pre_ffn_dst.is_some() {
                target_layer_ids
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, &layer)| (layer as usize == il).then_some(slot * h))
                    .collect()
            } else {
                Vec::new()
            };
            let pre_ffn_capture = pre_ffn_dst
                .filter(|_| !pre_ffn_offsets.is_empty())
                .map(|destination| (destination, pre_ffn_offsets.as_slice()));
            if self.model.arch.kind == ArchKind::Moe {
                let slot = match block {
                    MetalBlock::Gdn(_) => {
                        let slot = MixerSlot::Gdn(gdn_idx);
                        gdn_idx += 1;
                        slot
                    }
                    MetalBlock::Attn(_) => {
                        let slot = MixerSlot::Attn(attn_idx);
                        attn_idx += 1;
                        slot
                    }
                };
                self.encode_moe_block_gpu(&enc, block, slot, position, session)?;
            } else {
                self.encode_block_impl(
                    &enc,
                    il,
                    block,
                    &mut gdn_idx,
                    &mut attn_idx,
                    position,
                    session,
                    pre_ffn_capture,
                )?;
            }
            for intervention in interventions.iter().filter(|intervention| {
                let layer = match intervention {
                    PostBlockIntervention::Fixed { layer, .. }
                    | PostBlockIntervention::ResidualL2Relative { layer, .. }
                    | PostBlockIntervention::Projection { layer, .. }
                    | PostBlockIntervention::SourceToTarget { layer, .. } => *layer,
                };
                layer as usize == il
            }) {
                encode_post_block_intervention_f32(self.ctx, &enc, &session.x, intervention)?;
            }
            // Capture at any (possibly multiple) target_layer_ids slot
            // matching this block. Scatters run inline with the rest of
            // the command buffer; reads s.x BEFORE the next block writes
            // it, which is required since s.x is reused per layer.
            if let Some(post_block_dst) = post_block_dst {
                for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                    if lid as usize == il {
                        encode_scatter_offset_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            post_block_dst,
                            k_idx * h,
                            h,
                        )?;
                    }
                }
            }
        }

        if run_tail {
            // Final RMSNorm + lm_head — produces final logits as usual.
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                h,
                arch.vocab_size as usize,
            )?;
        }

        enc.end();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let status = cmd_buf.status();
        let error = cmd_buf.error();
        if status != MTLCommandBufferStatus::Completed || error.is_some() {
            session.poison("multi-hidden forward command failed");
            return Err(MfError::CommandBuffer {
                status: format!("{status:?}"),
                error: format!("{error:?}"),
            });
        }
        let mut logits = Vec::new();
        if run_tail {
            logits.resize(arch.vocab_size as usize, 0.0);
            unsafe {
                let src = session.logits.buffer.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, logits.as_mut_ptr(), logits.len());
            }
        }
        Ok(logits)
    }

    /// Skip-tail variant of [`single_token`] for prefill loops.
    ///
    /// Encodes embedding + per-block forward into one command buffer,
    /// commits, waits — but DOES NOT run final RMSNorm, lm_head, or
    /// readback logits. Returns `Ok(())` on success.
    ///
    /// The session state (KV cache, GDN state, position counters) is
    /// advanced exactly as if [`single_token`] had been called. Only
    /// `session.h` and `session.logits` are left in an unspecified
    /// state (downstream consumers must treat them as scratch). Callers
    /// MUST run a non-no-tail variant for the LAST prompt token to
    /// produce the bootstrap logits for the decode phase.
    ///
    /// v0.75.0: shipped to skip ~2 ms/token of lm_head Q6_K mat-vec +
    /// readback during prompt prefill. Estimated ~5% TTFT win at
    /// ctx ≥ 181. Establishes the API shape for v0.75.1's packed
    /// multi-token prefill (which replaces the body wholesale).
    pub fn single_token_no_tail(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        session.ensure_usable()?;
        if self.model.arch.kind == ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        if concurrent_gdn_dense_decode_enabled() {
            return self
                .single_token_profiled_concurrent_gdn_dense_with_tail(
                    token_id, position, session, false,
                )
                .map(|_| ());
        }
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let cmd_buf = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| MfError::CommandBuffer {
                status: "unavailable".into(),
                error: "Metal did not provide a command buffer".into(),
            })?;
        let enc = KernelEncoder::begin(&cmd_buf);

        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            h,
        )?;

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
        }

        // SKIP final RMSNorm + lm_head + readback (the "tail").
        enc.end();
        cmd_buf.commit();
        // Codex Q3: keep wait. Skipping the wait introduces a
        // session.ids_buf reuse hazard — the next prefill iteration
        // CPU-writes ids_buf for the next token, and Metal cmd-buffer
        // ordering does NOT order CPU writes to shared buffers after
        // commit. If get_rows for token i hasn't run yet, it would
        // read the overwritten id. Defer real async pipelining to
        // v0.75.1 where packed prefill restructures this.
        cmd_buf.waitUntilCompleted();
        let status = cmd_buf.status();
        let error = cmd_buf.error();
        if status != MTLCommandBufferStatus::Completed || error.is_some() {
            session.poison("no-tail forward command failed");
            return Err(MfError::CommandBuffer {
                status: format!("{status:?}"),
                error: format!("{error:?}"),
            });
        }
        Ok(())
    }

    /// Skip-tail variant of [`single_token_with_multi_hidden`] for
    /// DFlash prefill loops. Captures the K layer hiddens into
    /// `hidden_dst` exactly as [`single_token_with_multi_hidden`] does
    /// (those go on to feed `target_ctx_stacked` via the bench's
    /// `append_target_ctx_column_now`), but skips final RMSNorm,
    /// lm_head, and logits readback.
    ///
    /// See [`single_token_no_tail`] for the rationale and constraints.
    pub fn single_token_with_multi_hidden_no_tail(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        target_layer_ids: &[u32],
        hidden_dst: &MetalTensor,
    ) -> Result<(), MfError> {
        session.ensure_usable()?;
        if self.model.arch.kind == ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let k = target_layer_ids.len();
        if hidden_dst.n_elements() as usize != k * h {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "single_token_with_multi_hidden_no_tail.hidden_dst",
                detail: format!(
                    "expected {} elements (K={k} layers × H={h}), got {}",
                    k * h,
                    hidden_dst.n_elements()
                ),
            }));
        }
        for &lid in target_layer_ids {
            if (lid as usize) >= self.model.blocks.len() {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "single_token_with_multi_hidden_no_tail.target_layer_ids",
                    detail: format!("layer id {lid} >= n_layer {}", self.model.blocks.len()),
                }));
            }
        }

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let cmd_buf = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| MfError::CommandBuffer {
                status: "unavailable".into(),
                error: "Metal did not provide a command buffer".into(),
            })?;
        let enc = KernelEncoder::begin(&cmd_buf);

        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            h,
        )?;

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
            // Capture post-residual-#2 hidden at any matching layer
            // (matches v0.74.4 capture-point semantics). Runs inside
            // the same command buffer, before the next block writes
            // session.x.
            for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                if lid as usize == il {
                    encode_scatter_offset_f32(
                        self.ctx,
                        &enc,
                        &session.x,
                        hidden_dst,
                        k_idx * h,
                        h,
                    )?;
                }
            }
        }

        // SKIP final RMSNorm + lm_head + readback.
        enc.end();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let status = cmd_buf.status();
        let error = cmd_buf.error();
        if status != MTLCommandBufferStatus::Completed || error.is_some() {
            session.poison("multi-hidden no-tail forward command failed");
            return Err(MfError::CommandBuffer {
                status: format!("{status:?}"),
                error: format!("{error:?}"),
            });
        }
        Ok(())
    }

    pub fn single_token_with_hidden(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        hidden_dst: &MetalTensor,
    ) -> Result<Vec<f32>, MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        if hidden_dst.n_elements() as usize != h {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "single_token_with_hidden.hidden_dst",
                detail: format!("expected {h} elements, got {}", hidden_dst.n_elements()),
            }));
        }

        // Stage token id into the ids buffer.
        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        // (1) Embedding lookup → s.x.
        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            h,
        )?;

        // (2) Per-block.
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
        }

        // (3) Final RMSNorm over residual stream. session.x is read,
        // session.h is written. After this point s.x is still untouched
        // (output_norm doesn't write back to its input).
        encode_rms_norm_mul_f32(
            self.ctx,
            &enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;

        // (4) LM head → logits.
        encode_mat_vec_dispatch(
            self.ctx,
            &enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            h,
            arch.vocab_size as usize,
        )?;

        // (5) Capture pre-output_norm hidden into the caller-supplied dst.
        // Runs LAST in the command buffer to avoid any chance of
        // interleaving with the rms_norm + lm_head reads of session.x.
        // session.x has not been mutated since step (2) ended; the read
        // here pulls the same bytes RMSNorm read.
        encode_scatter_offset_f32(self.ctx, &enc, &session.x, hidden_dst, 0, h)?;

        enc.end();
        cmd_buf.commit();
        crate::metal::wait_completed(&cmd_buf)?;
        // Read back logits to CPU.
        let mut logits = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, logits.as_mut_ptr(), logits.len());
        }
        Ok(logits)
    }

    /// Same forward as [`single_token_with_hidden`] but reads back only the
    /// argmax token id instead of the full logits row. Used by the MTP
    /// speculative path, which needs the greedy next token and the hidden
    /// carry but never consumes full-vocab logits on CPU.
    pub fn single_token_argmax_with_hidden(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        hidden_dst: &MetalTensor,
        post_norm_hidden: bool,
    ) -> Result<i32, MfError> {
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        if hidden_dst.n_elements() as usize != h {
            return Err(MfError::Metal(MetalError::BadShape {
                kernel: "single_token_argmax_with_hidden.hidden_dst",
                detail: format!("expected {h} elements, got {}", hidden_dst.n_elements()),
            }));
        }

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");

        if arch.kind == ArchKind::Moe && concurrent_gdn_moe_decode_enabled() {
            // D1 fix (2026-07-20, packets w0b/w0c-econ): this path must use
            // the same encoder organization as production A3B decode. Keep
            // that invariant structurally by sharing the production body.
            self.encode_single_token_concurrent_gdn_moe_body(&cmd_buf, position, session)?;
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                h,
                arch.vocab_size as usize,
            )?;
            let hidden_src = if post_norm_hidden {
                &session.h
            } else {
                &session.x
            };
            encode_scatter_offset_f32(self.ctx, &enc, hidden_src, hidden_dst, 0, h)?;
            encode_argmax_f32(
                self.ctx,
                &enc,
                &session.logits,
                &session.argmax_tok,
                1,
                arch.vocab_size as usize,
            )?;
            enc.end();
            cmd_buf.commit();
            crate::metal::wait_completed(&cmd_buf)?;
            let argmax = unsafe {
                let src = session.argmax_tok.buffer.contents().as_ptr() as *const i32;
                *src
            };
            return Ok(argmax);
        }

        let enc = KernelEncoder::begin(&cmd_buf);

        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            h,
        )?;

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            if arch.kind == ArchKind::Moe {
                let slot = match block {
                    MetalBlock::Gdn(_) => {
                        let s = MixerSlot::Gdn(gdn_idx);
                        gdn_idx += 1;
                        s
                    }
                    MetalBlock::Attn(_) => {
                        let s = MixerSlot::Attn(attn_idx);
                        attn_idx += 1;
                        s
                    }
                };
                self.encode_moe_block_gpu(&enc, block, slot, position, session)?;
            } else {
                self.encode_block(
                    &enc,
                    il,
                    block,
                    &mut gdn_idx,
                    &mut attn_idx,
                    position,
                    session,
                )?;
            }
        }

        encode_rms_norm_mul_f32(
            self.ctx,
            &enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            &enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            h,
            arch.vocab_size as usize,
        )?;
        let hidden_src = if post_norm_hidden {
            &session.h
        } else {
            &session.x
        };
        encode_scatter_offset_f32(self.ctx, &enc, hidden_src, hidden_dst, 0, h)?;
        encode_argmax_f32(
            self.ctx,
            &enc,
            &session.logits,
            &session.argmax_tok,
            1,
            arch.vocab_size as usize,
        )?;

        enc.end();
        cmd_buf.commit();
        crate::metal::wait_completed(&cmd_buf)?;

        let argmax = unsafe {
            let src = session.argmax_tok.buffer.contents().as_ptr() as *const i32;
            *src
        };
        Ok(argmax)
    }

    /// Phase-resolved profiling: splits the per-token forward across
    /// MANY command buffers (one per block, plus embedding and lm_head)
    /// so we can attribute GPU time to logical phases. ★ ARTIFACT WARNING:
    /// the returned `wall_with_artifact_ms` is the WALL CLOCK of this
    /// split execution and includes ~10-15 ms of per-phase command-buffer
    /// overhead (commit + waitUntilCompleted + setup) NOT present in
    /// production single-token decode. Use the per-phase GPU times
    /// (sum-of-phases) for proportional reasoning, NOT the wall number,
    /// when comparing to production.
    ///
    /// For production-realistic ms/token, use [`single_token_profiled`]
    /// instead — that uses one command buffer per token (matching
    /// production) and returns true wall + true CPU encode + true GPU
    /// kernel times.
    ///
    /// Returns: (logits, wall_with_artifact_ms, per-phase GPU ms map)
    pub fn single_token_phase_profiled(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<PhaseProfileOutput, MfError> {
        if self.model.arch.kind == ArchKind::Moe {
            return self.single_token_phase_profiled_moe(token_id, position, session);
        }
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }

        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let mut phases: Vec<(String, f64)> = Vec::new();

        // Phase: embedding lookup.
        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_get_rows_f32(
                self.ctx,
                &enc,
                &self.model.token_embd,
                &session.ids_buf,
                &session.x,
                1,
                arch.hidden_size as usize,
            )?;
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd)?;
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            phases.push(("embedding".into(), ms));
        }

        // One command buffer per block. We aggregate by class (gdn vs attn)
        // so the report is digestible.
        let mut gdn_total_ms = 0.0f64;
        let mut attn_total_ms = 0.0f64;
        let mut gdn_count = 0usize;
        let mut attn_count = 0usize;
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd)?;
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            match block {
                MetalBlock::Gdn(_) => {
                    gdn_total_ms += ms;
                    gdn_count += 1;
                }
                MetalBlock::Attn(_) => {
                    attn_total_ms += ms;
                    attn_count += 1;
                }
            }
        }
        phases.push((format!("gdn layers (×{gdn_count})"), gdn_total_ms));
        phases.push((format!("attn layers (×{attn_count})"), attn_total_ms));

        // Phase: final norm.
        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd)?;
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            phases.push(("final norm".into(), ms));
        }

        // Phase: lm head.
        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                arch.hidden_size as usize,
                arch.vocab_size as usize,
            )?;
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd)?;
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            phases.push(("lm head".into(), ms));
        }

        if phase_lm_argmax_enabled() {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_argmax_f32(
                self.ctx,
                &enc,
                &session.logits,
                &session.argmax_tok,
                1,
                arch.vocab_size as usize,
            )?;
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd)?;
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            phases.push(("lm argmax".into(), ms));
        }

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((out, total_ms, phases))
    }

    pub(super) fn single_token_profiled_dense_serial(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }

        let t_total = std::time::Instant::now();

        // Stage the token id into the I32 ids buffer.
        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let t_encode = std::time::Instant::now();
        let cmd_buf = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| MfError::CommandBuffer {
                status: "unavailable".into(),
                error: "Metal did not provide a command buffer".into(),
            })?;
        let enc = KernelEncoder::begin(&cmd_buf);

        // (1) Embedding lookup → x.
        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            arch.hidden_size as usize,
        )?;

        // (2) Per-block.
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
        }

        // (3) Final RMSNorm over residual stream.
        encode_rms_norm_mul_f32(
            self.ctx,
            &enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;

        // (4) LM head → logits. Dispatch on dtype (Q4_K, Q6_K, F32).
        encode_mat_vec_dispatch(
            self.ctx,
            &enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            arch.hidden_size as usize,
            arch.vocab_size as usize,
        )?;

        enc.end();
        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;

        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let status = cmd_buf.status();
        let error = cmd_buf.error();
        if status != MTLCommandBufferStatus::Completed || error.is_some() {
            session.poison("dense serial single-token command failed");
            return Err(MfError::CommandBuffer {
                status: format!("{status:?}"),
                error: format!("{error:?}"),
            });
        }
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;

        // GPU-reported wall-clock execution time (CFTimeInterval seconds).
        let gpu_start = cmd_buf.GPUStartTime();
        let gpu_end = cmd_buf.GPUEndTime();
        let gpu_kernel_ms = (gpu_end - gpu_start) * 1e3;

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;

        Ok((
            out,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    pub fn single_token_profiled(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        session.ensure_usable()?;
        if self.model.arch.kind == ArchKind::Moe {
            return if concurrent_gdn_moe_decode_enabled() {
                self.single_token_profiled_concurrent_gdn_moe(token_id, position, session)
            } else {
                self.single_token_profiled_moe(token_id, position, session)
            };
        }
        if concurrent_gdn_dense_decode_enabled() {
            return self.single_token_profiled_concurrent_gdn_dense(token_id, position, session);
        }
        self.single_token_profiled_dense_serial(token_id, position, session)
    }

    pub fn encode_block(
        &self,
        enc: &KernelEncoder,
        il: usize,
        block: &MetalBlock,
        gdn_idx: &mut usize,
        attn_idx: &mut usize,
        position: u32,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        self.encode_block_impl(enc, il, block, gdn_idx, attn_idx, position, s, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_block_impl(
        &self,
        enc: &KernelEncoder,
        _il: usize,
        block: &MetalBlock,
        gdn_idx: &mut usize,
        attn_idx: &mut usize,
        position: u32,
        s: &mut MetalSession,
        pre_ffn_capture: Option<(&MetalTensor, &[usize])>,
    ) -> Result<(), MfError> {
        s.ensure_usable()?;
        // Pre-mixer norm.
        let attn_norm = match block {
            MetalBlock::Gdn(g) => &g.attn_norm,
            MetalBlock::Attn(a) => &a.attn_norm,
        };
        encode_rms_norm_mul_f32(self.ctx, enc, &s.x, attn_norm, &s.h, RMS_EPS)?;

        // Mixer (GDN or Attn) → mixer_out.
        match block {
            MetalBlock::Gdn(g) => {
                let i = *gdn_idx;
                *gdn_idx += 1;
                self.encode_gdn(enc, g, i, s)?;
            }
            MetalBlock::Attn(a) => {
                let i = *attn_idx;
                *attn_idx += 1;
                self.encode_attn(enc, a, i, position, s)?;
            }
        }

        self.encode_post_mixer_ffn_impl(enc, block, s, pre_ffn_capture)?;
        Ok(())
    }

    /// Apply the active production residual/post-norm organization to a
    /// caller-provided mixer row. Batch executors use this seam so rollback
    /// flags cannot silently diverge from singleton decode.
    #[doc(hidden)]
    pub fn encode_post_mixer_norm(
        &self,
        enc: &KernelEncoder,
        residual: &MetalTensor,
        mixer_output: &MetalTensor,
        norm: &MetalTensor,
        normalized: &MetalTensor,
    ) -> Result<(), MfError> {
        if decode_fused_residual_rmsnorm_enabled() {
            encode_residual_rms_norm_mul_f32(
                self.ctx,
                enc,
                residual,
                mixer_output,
                norm,
                normalized,
                RMS_EPS,
            )?;
        } else {
            encode_add_inplace_f32(self.ctx, enc, residual, mixer_output)?;
            encode_rms_norm_mul_f32(self.ctx, enc, residual, norm, normalized, RMS_EPS)?;
        }
        Ok(())
    }

    pub(super) fn encode_post_mixer_ffn(
        &self,
        enc: &KernelEncoder,
        block: &MetalBlock,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        self.encode_post_mixer_ffn_impl(enc, block, s, None)
    }

    pub(super) fn encode_post_mixer_ffn_impl(
        &self,
        enc: &KernelEncoder,
        block: &MetalBlock,
        s: &mut MetalSession,
        pre_ffn_capture: Option<(&MetalTensor, &[usize])>,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;

        // Pre-FFN norm.
        let post_norm = match block {
            MetalBlock::Gdn(g) => &g.post_attn_norm,
            MetalBlock::Attn(a) => &a.post_attn_norm,
        };
        self.encode_post_mixer_norm(enc, &s.x, &s.mixer_out, post_norm, &s.h)?;
        if let Some((destination, offsets)) = pre_ffn_capture {
            for &offset in offsets {
                encode_scatter_offset_f32(self.ctx, enc, &s.x, destination, offset, h)?;
            }
        }

        // SwiGLU FFN.
        let (g_w, u_w, d_w) = match block {
            MetalBlock::Gdn(g) => (&g.ffn_gate, &g.ffn_up, &g.ffn_down),
            MetalBlock::Attn(a) => (&a.ffn_gate, &a.ffn_up, &a.ffn_down),
        };
        let f = arch.intermediate_size as usize;
        // Fused SwiGLU when both gate and up are Q4_K (27B production).
        // Falls back to 3-dispatch path for F32 weights (0.8B) or other dtypes.
        // Per Jeff & Sanjay: amortize boundary crossings + eliminate
        // intermediate materialization (no separate gate/up writes).
        let ffn_fused = g_w.dtype == GgmlType::Q4_K && u_w.dtype == GgmlType::Q4_K;
        if ffn_fused {
            encode_ffn_swiglu_q4_K_f32(self.ctx, enc, g_w, u_w, &s.h, &s.ffn_inner, h, f)?;
        } else {
            encode_mat_vec_dispatch(self.ctx, enc, g_w, &s.h, &s.ffn_gate, h, f)?;
            encode_mat_vec_dispatch(self.ctx, enc, u_w, &s.h, &s.ffn_up, h, f)?;
            encode_silu_mul_f32(self.ctx, enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)?;
        }
        // T9 bench-only capture (no-op unless installed by the capture
        // harness; see t9_ffn_capture_install).
        if let Some((h_dst, inner_dst)) = t9_ffn_capture_slots_for_current_call() {
            encode_scatter_offset_f32(self.ctx, enc, &s.h, &h_dst, 0, h)?;
            encode_scatter_offset_f32(self.ctx, enc, &s.ffn_inner, &inner_dst, 0, f)?;
        }
        encode_mat_vec_dispatch(self.ctx, enc, d_w, &s.ffn_inner, &s.ffn_out, f, h)?;

        // Residual #2: x += ffn_out.
        encode_add_inplace_f32(self.ctx, enc, &s.x, &s.ffn_out)?;
        Ok(())
    }

    /// Test helper: write `data` into `s.x` (the residual stream).
    pub fn set_residual_for_test(&self, s: &mut MetalSession, data: &[f32]) {
        unsafe {
            let dst = s.x.buffer.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }
}
