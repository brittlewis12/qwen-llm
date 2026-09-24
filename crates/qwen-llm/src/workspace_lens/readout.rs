//! F16 transport application, packed capture, and top-k readouts.

use super::*;

pub(super) fn admit_scalar_observer_allocations(
    context: &crate::metal::MetalContext,
    gpu_buffers: &[usize],
    host_bytes: usize,
) -> Result<(), WorkspaceLensError> {
    let gpu_bytes = gpu_buffers.iter().try_fold(0u64, |total, &bytes| {
        let bytes = u64::try_from(bytes).map_err(|_| WorkspaceLensError::SizeOverflow)?;
        total
            .checked_add(context.shared_buffer_size_and_align(bytes)?.size)
            .ok_or(WorkspaceLensError::SizeOverflow)
    })?;
    let host_bytes = u64::try_from(host_bytes).map_err(|_| WorkspaceLensError::SizeOverflow)?;
    // Current signals already include resident model/sequence buffers. This is
    // an incremental admission check, not a second reservation of their bytes.
    let admission = crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
        gpu_bytes,
        host_bytes,
        0,
        context.memory_signals(),
        true,
    );
    if !admission.admitted {
        return Err(WorkspaceLensError::FullReadoutMemoryAdmissionDenied {
            reason: admission.reason,
            requested_bytes: gpu_bytes
                .checked_add(host_bytes)
                .ok_or(WorkspaceLensError::SizeOverflow)?,
            working_set_headroom_bytes: admission.working_set_headroom_bytes,
            process_remaining_bytes: admission.signals.process_limit_remaining_bytes,
        });
    }
    Ok(())
}

fn encode_scalar_lens_head(
    context: &crate::metal::MetalContext,
    encoder: &KernelEncoder,
    model: &crate::metal_forward::MetalModel,
    residual: &MetalTensor,
    normalized: &MetalTensor,
    logits: &MetalTensor,
) -> Result<(), WorkspaceLensError> {
    encode_rms_norm_mul_f32(
        context,
        encoder,
        residual,
        &model.output_norm,
        normalized,
        RMS_EPS,
    )?;
    encode_mat_vec_dispatch(
        context,
        encoder,
        &model.lm_head,
        normalized,
        logits,
        model.arch.hidden_size as usize,
        model.arch.vocab_size as usize,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn reduce_workspace_vjp_readouts(
    trajectories: &[f32],
    n_sources: usize,
    chunk_queries: usize,
    n_tokens: usize,
    hidden_size: usize,
    valid_positions: std::ops::Range<usize>,
    destination: &mut [f32],
    total_queries: usize,
    first_query: usize,
) -> Result<(), WorkspaceLensError> {
    let trajectory_elements = checked_product(n_tokens, hidden_size)?;
    let source_elements = checked_product(chunk_queries, trajectory_elements)?;
    let expected = checked_product(n_sources, source_elements)?;
    if trajectories.len() != expected {
        return Err(WorkspaceLensError::ActivationSize {
            name: "workspace readout source trajectory bank",
            got: trajectories.len(),
            expected,
        });
    }
    let destination_expected =
        checked_product(n_sources, checked_product(total_queries, hidden_size)?)?;
    if destination.len() != destination_expected {
        return Err(WorkspaceLensError::ActivationSize {
            name: "workspace readout destination bank",
            got: destination.len(),
            expected: destination_expected,
        });
    }
    let chunk_end = first_query
        .checked_add(chunk_queries)
        .ok_or(WorkspaceLensError::SizeOverflow)?;
    if chunk_end > total_queries {
        return Err(WorkspaceLensError::ActivationSize {
            name: "workspace readout destination query range",
            got: chunk_end,
            expected: total_queries,
        });
    }
    for source in 0..n_sources {
        for query in 0..chunk_queries {
            let source_start = checked_product(source, source_elements)?
                .checked_add(checked_product(query, trajectory_elements)?)
                .ok_or(WorkspaceLensError::SizeOverflow)?;
            let source_end = source_start
                .checked_add(trajectory_elements)
                .ok_or(WorkspaceLensError::SizeOverflow)?;
            let destination_row = checked_product(source, total_queries)?
                .checked_add(first_query)
                .and_then(|row| row.checked_add(query))
                .ok_or(WorkspaceLensError::SizeOverflow)?;
            let destination_start = checked_product(destination_row, hidden_size)?;
            let destination_end = destination_start
                .checked_add(hidden_size)
                .ok_or(WorkspaceLensError::SizeOverflow)?;
            reduce_workspace_source_positions(
                &trajectories[source_start..source_end],
                n_tokens,
                hidden_size,
                valid_positions.clone(),
                &mut destination[destination_start..destination_end],
            )?;
        }
    }
    Ok(())
}

pub(super) fn f16_transport_readout_peak_bytes(
    hidden_size: usize,
    transport_bytes: usize,
    query_count: usize,
) -> Result<usize, WorkspaceLensError> {
    let covector_bytes = checked_product(
        checked_product(query_count, hidden_size)?,
        std::mem::size_of::<f32>(),
    )?;
    transport_bytes
        .checked_mul(F16_TRANSPORT_READOUT_LIVE_TRANSPORT_BANKS)
        .and_then(|bytes| {
            bytes
                .checked_add(covector_bytes.checked_mul(F16_TRANSPORT_READOUT_LIVE_COVECTOR_BANKS)?)
        })
        .ok_or(WorkspaceLensError::SizeOverflow)
}

pub(super) fn f16_transport_readout_query_capacity(
    hidden_size: usize,
    transport_bytes: usize,
    additional_live_bytes: usize,
) -> Result<usize, WorkspaceLensError> {
    let fixed_bytes = transport_bytes
        .checked_mul(F16_TRANSPORT_READOUT_LIVE_TRANSPORT_BANKS)
        .and_then(|bytes| bytes.checked_add(additional_live_bytes))
        .ok_or(WorkspaceLensError::SizeOverflow)?;
    let bytes_per_query = checked_product(
        checked_product(hidden_size, std::mem::size_of::<f32>())?,
        F16_TRANSPORT_READOUT_LIVE_COVECTOR_BANKS,
    )?;
    let available = MAX_WORKSPACE_LENS_OWNED_RESULT_BYTES
        .checked_sub(fixed_bytes)
        .ok_or(WorkspaceLensError::WorkspaceLensResultByteBudgetExceeded {
            name: "F16 transport readout projection",
            requested_bytes: fixed_bytes,
            max_bytes: MAX_WORKSPACE_LENS_OWNED_RESULT_BYTES,
        })?;
    let capacity = available / bytes_per_query;
    if capacity == 0 {
        return Err(WorkspaceLensError::WorkspaceLensResultByteBudgetExceeded {
            name: "F16 transport readout projection",
            requested_bytes: fixed_bytes
                .checked_add(bytes_per_query)
                .ok_or(WorkspaceLensError::SizeOverflow)?,
            max_bytes: MAX_WORKSPACE_LENS_OWNED_RESULT_BYTES,
        });
    }
    Ok(capacity)
}

pub(super) fn multiply_token_readout_gamma_in_place(
    rows: &mut [f32],
    gamma: &[f32],
    n_rows: usize,
    hidden_size: usize,
) -> Result<(), WorkspaceLensError> {
    let expected_rows = checked_product(n_rows, hidden_size)?;
    if rows.len() != expected_rows {
        return Err(WorkspaceLensError::ActivationSize {
            name: "selected LM-head rows",
            got: rows.len(),
            expected: expected_rows,
        });
    }
    if gamma.len() != hidden_size {
        return Err(WorkspaceLensError::ActivationSize {
            name: "selected-token output norm gamma",
            got: gamma.len(),
            expected: hidden_size,
        });
    }
    if let Some(index) = rows.iter().position(|value| !value.is_finite()) {
        return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
            name: "LM-head rows",
            index,
        });
    }
    if let Some(index) = gamma.iter().position(|value| !value.is_finite()) {
        return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
            name: "output norm gamma",
            index,
        });
    }
    for (index, value) in rows.iter_mut().enumerate() {
        *value *= gamma[index % hidden_size];
        if !value.is_finite() {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "gamma-folded LM-head rows",
                index,
            });
        }
    }
    Ok(())
}

impl WorkspaceLensFullReadoutWorkspace<'_> {
    /// Bind one F16 hidden-to-hidden transport for repeated row or tile use.
    pub fn bind_f16_transport(&mut self, transport_bytes: &[u8]) -> Result<(), WorkspaceLensError> {
        self.transport_bound = false;
        let hidden_size = self.model.arch().hidden_size as usize;
        validate_full_readout_transport_size(transport_bytes, hidden_size)?;
        write_tensor_bytes(
            &self.transport,
            transport_bytes,
            transport_bytes.len(),
            "full readout transport upload",
        )?;
        self.transport_bound = true;
        Ok(())
    }

    /// Read one row with the historical matvec, full-logit validation, and
    /// exact CPU top-k path.
    pub fn apply_row_f16_transport_topk_with_vector(
        &mut self,
        transport_bytes: &[u8],
        source_residual: &[f32],
        top_k: usize,
    ) -> Result<WorkspaceLensFullVocabularyReadoutWithVector, WorkspaceLensError> {
        if top_k == 0 || top_k > MAX_FULL_READOUT_TOP_K {
            return Err(WorkspaceLensError::InvalidFullReadoutTopK {
                got: top_k,
                max: MAX_FULL_READOUT_TOP_K,
            });
        }
        self.apply_row_f16_transport_logits_with_vector(transport_bytes, source_residual)?
            .into_topk(top_k)
    }

    pub fn apply_row_f16_transport_logits_with_vector(
        &mut self,
        transport_bytes: &[u8],
        source_residual: &[f32],
    ) -> Result<WorkspaceLensFullVocabularyLogitsWithVector, WorkspaceLensError> {
        let arch = self.model.arch();
        let hidden_size = arch.hidden_size as usize;
        let vocab_size = arch.vocab_size as usize;
        if source_residual.len() != hidden_size {
            return Err(WorkspaceLensError::ActivationSize {
                name: "full readout source residual",
                got: source_residual.len(),
                expected: hidden_size,
            });
        }
        if let Some(index) = source_residual.iter().position(|value| !value.is_finite()) {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "full readout source residual",
                index,
            });
        }
        validate_full_readout_transport_size(transport_bytes, hidden_size)?;
        validate_scalar_readout_tail(self.model.metal_model(), hidden_size, vocab_size)?;
        let hidden_bytes = checked_product(hidden_size, std::mem::size_of::<f32>())?;
        let logits_bytes = checked_product(vocab_size, std::mem::size_of::<f32>())?;
        let peak_bytes = transport_bytes
            .len()
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(hidden_bytes.checked_mul(5)?))
            .and_then(|bytes| bytes.checked_add(logits_bytes.checked_mul(2)?))
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        enforce_workspace_lens_byte_budget("full-vocabulary F16 transport readout", peak_bytes)?;

        self.bind_f16_transport(transport_bytes)?;
        write_tensor_bytes(
            &self.source,
            bytemuck::cast_slice(source_residual),
            hidden_bytes,
            "full readout source upload",
        )?;
        let source = self.source.view_subrange(0, vec![hidden_size as u64]);
        let transported = self.transported.view_subrange(0, vec![hidden_size as u64]);
        let normalized = self.normalized.view_subrange(0, vec![hidden_size as u64]);
        let logits = self.logits.view_subrange(0, vec![vocab_size as u64]);
        let context = self.model.context();
        let model = self.model.metal_model();
        let command = context
            .queue
            .commandBuffer()
            .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
        let encoder = KernelEncoder::begin(&command);
        let encode_result = (|| -> Result<(), WorkspaceLensError> {
            encode_mat_vec_f16_f32(
                context,
                &encoder,
                &self.transport,
                &source,
                &transported,
                hidden_size,
                hidden_size,
            )?;
            encode_scalar_lens_head(context, &encoder, model, &transported, &normalized, &logits)?;
            Ok(())
        })();
        encoder.end();
        encode_result?;
        command.commit();
        crate::metal::wait_unchecked(&command);
        validate_completed_command(&command)?;

        let transported_values = read_f32_fallible(
            &transported,
            hidden_size,
            "full readout transported residual",
        )?;
        if let Some(index) = transported_values
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "full readout transported residual",
                index,
            });
        }
        let rms_denominator_f64_recomputed = (transported_values
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            / hidden_size as f64
            + f64::from(RMS_EPS))
        .sqrt() as f32;
        let full_logits = read_f32_fallible(&logits, vocab_size, "full readout logits")?;
        if let Some(index) = full_logits.iter().position(|value| !value.is_finite()) {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "full readout logits",
                index,
            });
        }
        Ok(WorkspaceLensFullVocabularyLogitsWithVector {
            logits: full_logits,
            rms_denominator_f64_recomputed,
            transported_values,
        })
    }

    pub fn apply_packed_capture_f16_transport_topk_with_vectors(
        &mut self,
        capture: &WorkspaceLensPackedPostBlockCapture<'_>,
        source_layer: u32,
        transport_bytes: &[u8],
        top_k: usize,
        transported_source_positions: &[usize],
    ) -> Result<WorkspaceLensPackedFullVocabularyReadout, WorkspaceLensError> {
        self.validate_packed_capture_readout(
            capture,
            source_layer,
            Some(transport_bytes),
            top_k,
            transported_source_positions,
            false,
        )?;
        self.bind_f16_transport(transport_bytes)?;
        self.apply_packed_capture_bound_f16_transport_topk_with_vectors(
            capture,
            source_layer,
            top_k,
            transported_source_positions,
        )
    }

    /// Apply the currently bound transport while preserving this capture's
    /// prompt-local row shape and arithmetic topology.
    pub fn apply_packed_capture_bound_f16_transport_topk_with_vectors(
        &mut self,
        capture: &WorkspaceLensPackedPostBlockCapture<'_>,
        source_layer: u32,
        top_k: usize,
        transported_source_positions: &[usize],
    ) -> Result<WorkspaceLensPackedFullVocabularyReadout, WorkspaceLensError> {
        self.apply_packed_capture_bound_f16_transport_topk_with_distribution_summaries(
            capture,
            source_layer,
            top_k,
            transported_source_positions,
            false,
        )
    }

    /// Opt-in full-vocabulary CPU reduction; retains only one host logit row.
    pub fn apply_packed_capture_bound_f16_transport_topk_with_distribution_summaries(
        &mut self,
        capture: &WorkspaceLensPackedPostBlockCapture<'_>,
        source_layer: u32,
        top_k: usize,
        transported_source_positions: &[usize],
        distribution_summaries: bool,
    ) -> Result<WorkspaceLensPackedFullVocabularyReadout, WorkspaceLensError> {
        if !self.transport_bound {
            return Err(WorkspaceLensError::FullReadoutTransportNotBound);
        }
        let validation = self.validate_packed_capture_readout(
            capture,
            source_layer,
            None,
            top_k,
            transported_source_positions,
            distribution_summaries,
        )?;
        let PackedFullReadoutValidation {
            layer_slot,
            position_count,
            hidden_size,
            vocab_size,
            transported_position_rows,
        } = validation;
        let vocab_size_u32 = self.model.arch().vocab_size;
        let hidden_shape = vec![position_count as u64, hidden_size as u64];
        let source = self.source.view_subrange(0, hidden_shape.clone());
        let transported = self.transported.view_subrange(0, hidden_shape.clone());
        let normalized = self.normalized.view_subrange(0, hidden_shape);
        let logits = self
            .logits
            .view_subrange(0, vec![position_count as u64, vocab_size as u64]);
        let compact_shape = vec![position_count as u64, MPS_FULL_READOUT_TOP_K as u64];
        let first_ids = self.first_ids.view_subrange(0, compact_shape.clone());
        let first_values = self.first_values.view_subrange(0, compact_shape.clone());
        let second_ids = self.second_ids.view_subrange(0, compact_shape.clone());
        let second_values = self.second_values.view_subrange(0, compact_shape);

        let context = self.model.context();
        let model = self.model.metal_model();
        let readout_started = Instant::now();
        let command = context
            .queue
            .commandBuffer()
            .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
        let encoder = KernelEncoder::begin(&command);
        let encode_result = (|| -> Result<(), WorkspaceLensError> {
            for position_row in 0..position_count {
                let source_offset = checked_product(
                    checked_product(position_row, capture.layer_ids.len())?
                        .checked_add(layer_slot)
                        .ok_or(WorkspaceLensError::SizeOverflow)?,
                    hidden_size,
                )?;
                let destination = source.view_subrange(
                    u64::try_from(checked_product(position_row, hidden_size)?)
                        .map_err(|_| WorkspaceLensError::SizeOverflow)?,
                    vec![hidden_size as u64],
                );
                encode_copy_offset_f32(
                    context,
                    &encoder,
                    &capture.values,
                    source_offset,
                    &destination,
                    hidden_size,
                )?;
            }
            encode_mat_mat_f16_f32(
                context,
                &encoder,
                &self.transport,
                &source,
                &transported,
                hidden_size,
                hidden_size,
                position_count,
            )?;
            encode_rms_norm_mul_rows_f32(
                context,
                &encoder,
                &transported,
                &model.output_norm,
                &normalized,
                position_count,
                hidden_size,
                RMS_EPS,
            )?;
            encode_mat_mat_dispatch(
                context,
                &encoder,
                &model.lm_head,
                &normalized,
                &logits,
                hidden_size,
                vocab_size,
                position_count,
            )?;
            Ok(())
        })();
        encoder.end();
        encode_result?;
        encode_mps_topk16_f32(
            context,
            &command,
            &logits,
            &first_ids,
            &first_values,
            position_count,
            vocab_size,
        )?;
        let mask_encoder = KernelEncoder::begin(&command);
        let mask_result = encode_mask_row_indices_f32(
            context,
            &mask_encoder,
            &logits,
            &first_ids,
            position_count,
            vocab_size,
            MPS_FULL_READOUT_TOP_K,
        );
        mask_encoder.end();
        mask_result?;
        encode_mps_topk16_f32(
            context,
            &command,
            &logits,
            &second_ids,
            &second_values,
            position_count,
            vocab_size,
        )?;
        command.commit();
        crate::metal::wait_unchecked(&command);
        validate_completed_command(&command)?;
        let readout_wall_ms = readout_started.elapsed().as_secs_f64() * 1e3;
        let readout_gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;

        let pass_elements = checked_product(position_count, MPS_FULL_READOUT_TOP_K)?;
        let first_ids =
            read_i32_fallible(&first_ids, pass_elements, "packed readout first-pass IDs")?;
        let first_values = read_f32_fallible(
            &first_values,
            pass_elements,
            "packed readout first-pass logits",
        )?;
        let second_ids =
            read_i32_fallible(&second_ids, pass_elements, "packed readout second-pass IDs")?;
        let second_values = read_f32_fallible(
            &second_values,
            pass_elements,
            "packed readout second-pass logits",
        )?;
        let mut positions = build_packed_vocabulary_positions(
            capture.token_ids(),
            capture.start_position(),
            top_k,
            vocab_size_u32,
            &first_ids,
            &first_values,
            &second_ids,
            &second_values,
        )?;
        if distribution_summaries {
            let row_bytes = checked_product(vocab_size, std::mem::size_of::<f32>())?;
            enforce_workspace_lens_byte_budget("distribution summary CPU row", row_bytes)?;
            for (row, position) in positions.iter_mut().enumerate() {
                let tensor = logits.view_subrange(
                    u64::try_from(checked_product(row, vocab_size)?)
                        .map_err(|_| WorkspaceLensError::SizeOverflow)?,
                    vec![vocab_size as u64],
                );
                let mut values =
                    read_f32_fallible(&tensor, vocab_size, "distribution summary CPU row")?;
                let start = checked_product(row, MPS_FULL_READOUT_TOP_K)?;
                let end = start + MPS_FULL_READOUT_TOP_K;
                super::distribution::restore_masked_distribution_row(
                    &mut values,
                    &first_ids[start..end],
                    &first_values[start..end],
                )?;
                position.distribution_summary = Some(
                    super::distribution::summarize_distribution_row(&values, &position.scores)?,
                );
            }
        }
        let transported_vectors = read_packed_transported_vectors(
            &transported,
            capture,
            transported_source_positions,
            &transported_position_rows,
            hidden_size,
        )?;
        Ok(WorkspaceLensPackedFullVocabularyReadout {
            source_layer,
            start_position: capture.start_position(),
            position_count,
            top_k,
            packed_prefill_gpu_ms: capture.packed_prefill_gpu_ms(),
            packed_prefill_wall_ms: capture.packed_prefill_wall_ms(),
            readout_gpu_ms,
            readout_wall_ms,
            positions,
            transported_vectors,
        })
    }

    pub(super) fn validate_packed_capture_readout(
        &self,
        capture: &WorkspaceLensPackedPostBlockCapture<'_>,
        source_layer: u32,
        transport_bytes: Option<&[u8]>,
        top_k: usize,
        transported_source_positions: &[usize],
        distribution_summaries: bool,
    ) -> Result<PackedFullReadoutValidation, WorkspaceLensError> {
        if !std::ptr::eq(self.model, capture.model) {
            return Err(WorkspaceLensError::PackedCaptureModelMismatch);
        }
        if top_k == 0 || top_k > MAX_FULL_READOUT_TOP_K {
            return Err(WorkspaceLensError::InvalidFullReadoutTopK {
                got: top_k,
                max: MAX_FULL_READOUT_TOP_K,
            });
        }
        let position_count = capture.position_count();
        if position_count > self.row_capacity {
            return Err(WorkspaceLensError::FullReadoutWorkspaceTooSmall {
                capacity: self.row_capacity,
                required: position_count,
            });
        }
        let layer_slot = capture.layer_slot(source_layer)?;
        let arch = self.model.arch();
        let hidden_size = arch.hidden_size as usize;
        let vocab_size = arch.vocab_size as usize;
        if let Some(transport_bytes) = transport_bytes {
            validate_full_readout_transport_size(transport_bytes, hidden_size)?;
        }
        validate_full_readout_tail(self.model.metal_model(), hidden_size, vocab_size)?;
        let transported_position_rows = validate_packed_transported_vector_positions(
            capture.start_position(),
            position_count,
            transported_source_positions,
        )?;
        let hidden_elements = checked_product(position_count, hidden_size)?;
        let logits_elements = checked_product(position_count, vocab_size)?;
        let compact_elements = checked_product(position_count, FULL_READOUT_CANDIDATE_COUNT)?;
        let hidden_bytes = checked_product(hidden_elements, std::mem::size_of::<f32>())?;
        let logits_bytes = checked_product(logits_elements, std::mem::size_of::<f32>())?;
        let compact_bytes = checked_product(compact_elements, 2 * std::mem::size_of::<u32>())?;
        let transported_vector_bytes = checked_product(
            checked_product(transported_position_rows.len(), hidden_size)?,
            std::mem::size_of::<f32>(),
        )?;
        let transport_bytes = checked_product(checked_product(hidden_size, hidden_size)?, 2)?;
        let peak_bytes = transport_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(hidden_bytes.checked_mul(3)?))
            .and_then(|bytes| bytes.checked_add(logits_bytes))
            .and_then(|bytes| bytes.checked_add(compact_bytes))
            .and_then(|bytes| bytes.checked_add(transported_vector_bytes))
            .and_then(|bytes| {
                bytes.checked_add(if distribution_summaries {
                    vocab_size.checked_mul(std::mem::size_of::<f32>())?
                } else {
                    0
                })
            })
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        enforce_workspace_lens_byte_budget(
            "packed full-vocabulary F16 transport readout",
            peak_bytes,
        )?;
        Ok(PackedFullReadoutValidation {
            layer_slot,
            position_count,
            hidden_size,
            vocab_size,
            transported_position_rows,
        })
    }
}

pub(super) fn validate_full_readout_transport_size(
    transport_bytes: &[u8],
    hidden_size: usize,
) -> Result<(), WorkspaceLensError> {
    let expected = checked_product(checked_product(hidden_size, hidden_size)?, 2)?;
    if transport_bytes.len() != expected {
        return Err(WorkspaceLensError::InvalidFullReadoutTransportSize {
            got: transport_bytes.len(),
            expected,
        });
    }
    Ok(())
}

pub(super) fn full_readout_workspace_allocation_bytes(
    row_capacity: usize,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<[usize; 9], WorkspaceLensError> {
    let transport = checked_product(checked_product(hidden_size, hidden_size)?, 2)?;
    let hidden = checked_product(
        checked_product(row_capacity, hidden_size)?,
        std::mem::size_of::<f32>(),
    )?;
    let logits = checked_product(
        checked_product(row_capacity, vocab_size)?,
        std::mem::size_of::<f32>(),
    )?;
    let compact = checked_product(
        checked_product(row_capacity, MPS_FULL_READOUT_TOP_K)?,
        std::mem::size_of::<u32>(),
    )?;
    Ok([
        transport, hidden, hidden, hidden, logits, compact, compact, compact, compact,
    ])
}

pub(super) fn validate_full_readout_tail(
    model: &crate::metal_forward::MetalModel,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<(), WorkspaceLensError> {
    let expected_lm_head_shape = [hidden_size, vocab_size];
    if linear_shape(WorkspaceLensLinear::LmHead, &model.lm_head).ok()
        != Some(expected_lm_head_shape)
    {
        return Err(WorkspaceLensError::InvalidTokenReadoutLmHeadShape {
            got: model.lm_head.shape.clone(),
            expected: expected_lm_head_shape,
        });
    }
    if !selected_readout_head_dtype_supported(model.lm_head.dtype) {
        return Err(WorkspaceLensError::UnsupportedTokenReadoutLmHeadDtype {
            dtype: model.lm_head.dtype,
        });
    }
    if model.output_norm.dtype != GgmlType::F32
        || model.output_norm.shape.as_slice() != [hidden_size as u64]
    {
        return Err(WorkspaceLensError::InvalidTokenReadoutOutputNorm {
            dtype: model.output_norm.dtype,
            shape: model.output_norm.shape.clone(),
            expected: hidden_size,
        });
    }
    Ok(())
}

pub(super) fn validate_scalar_readout_tail(
    model: &crate::metal_forward::MetalModel,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<(), WorkspaceLensError> {
    // Dtype support belongs to scalar matvec dispatch, not selected-row decoding.
    let expected_lm_head_shape = [hidden_size, vocab_size];
    if linear_shape(WorkspaceLensLinear::LmHead, &model.lm_head).ok()
        != Some(expected_lm_head_shape)
    {
        return Err(WorkspaceLensError::InvalidTokenReadoutLmHeadShape {
            got: model.lm_head.shape.clone(),
            expected: expected_lm_head_shape,
        });
    }
    if model.output_norm.dtype != GgmlType::F32
        || model.output_norm.shape.as_slice() != [hidden_size as u64]
    {
        return Err(WorkspaceLensError::InvalidTokenReadoutOutputNorm {
            dtype: model.output_norm.dtype,
            shape: model.output_norm.shape.clone(),
            expected: hidden_size,
        });
    }
    Ok(())
}

pub(super) fn validate_packed_transported_vector_positions(
    start_position: usize,
    position_count: usize,
    source_positions: &[usize],
) -> Result<Vec<usize>, WorkspaceLensError> {
    let end_position = start_position
        .checked_add(position_count)
        .ok_or(WorkspaceLensError::PositionOverflow(start_position))?;
    let mut position_rows = Vec::new();
    position_rows
        .try_reserve_exact(source_positions.len())
        .map_err(|_| WorkspaceLensError::WorkspaceLensHostAllocationFailed {
            name: "packed transported-vector position rows",
            elements: source_positions.len(),
        })?;
    for (index, &source_position) in source_positions.iter().enumerate() {
        if source_position < start_position || source_position >= end_position {
            return Err(
                WorkspaceLensError::PackedTransportedVectorPositionOutOfRange {
                    source_position,
                    start_position,
                    end_position,
                },
            );
        }
        if source_positions[..index].contains(&source_position) {
            return Err(
                WorkspaceLensError::DuplicatePackedTransportedVectorPosition { source_position },
            );
        }
        position_rows.push(source_position - start_position);
    }
    Ok(position_rows)
}

pub(super) fn read_packed_transported_vectors(
    transported: &MetalTensor,
    capture: &WorkspaceLensPackedPostBlockCapture<'_>,
    source_positions: &[usize],
    position_rows: &[usize],
    hidden_size: usize,
) -> Result<Vec<WorkspaceLensPackedTransportedVector>, WorkspaceLensError> {
    if source_positions.len() != position_rows.len() {
        return Err(WorkspaceLensError::ActivationSize {
            name: "packed transported-vector position rows",
            got: position_rows.len(),
            expected: source_positions.len(),
        });
    }
    let mut vectors = Vec::new();
    vectors
        .try_reserve_exact(source_positions.len())
        .map_err(|_| WorkspaceLensError::WorkspaceLensHostAllocationFailed {
            name: "packed transported vectors",
            elements: source_positions.len(),
        })?;
    for (vector_index, (&source_position, &position_row)) in
        source_positions.iter().zip(position_rows).enumerate()
    {
        let row_offset = checked_product(position_row, hidden_size)?;
        let row = transported.view_subrange(
            u64::try_from(row_offset).map_err(|_| WorkspaceLensError::SizeOverflow)?,
            vec![hidden_size as u64],
        );
        let values = read_f32_fallible(&row, hidden_size, "packed transported-vector values")?;
        if let Some(component) = values.iter().position(|value| !value.is_finite()) {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "packed transported-vector values",
                index: checked_product(vector_index, hidden_size)?
                    .checked_add(component)
                    .ok_or(WorkspaceLensError::SizeOverflow)?,
            });
        }
        let predicts_position = source_position
            .checked_add(1)
            .ok_or(WorkspaceLensError::PositionOverflow(source_position))?;
        vectors.push(WorkspaceLensPackedTransportedVector {
            source_position,
            source_token_id: capture.token_ids()[position_row],
            predicts_position,
            values,
        });
    }
    Ok(vectors)
}

pub(super) fn exact_vocabulary_top_k(
    logits: &[f32],
    top_k: usize,
) -> Result<Vec<WorkspaceLensVocabularyScore>, WorkspaceLensError> {
    let mut scores = Vec::new();
    scores.try_reserve_exact(top_k).map_err(|_| {
        WorkspaceLensError::WorkspaceLensHostAllocationFailed {
            name: "full-vocabulary top-k scores",
            elements: top_k,
        }
    })?;
    for (token_id, &logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "full readout logits",
                index: token_id,
            });
        }
        let token_id = token_id as u32;
        let insertion = scores.partition_point(|existing: &WorkspaceLensVocabularyScore| {
            existing.logit > logit || (existing.logit == logit && existing.token_id < token_id)
        });
        if insertion < top_k {
            scores.insert(insertion, WorkspaceLensVocabularyScore { token_id, logit });
            if scores.len() > top_k {
                scores.pop();
            }
        }
    }
    Ok(scores)
}

pub(super) fn build_packed_vocabulary_positions(
    token_ids: &[i32],
    start_position: usize,
    top_k: usize,
    vocab_size: u32,
    first_ids: &[i32],
    first_values: &[f32],
    second_ids: &[i32],
    second_values: &[f32],
) -> Result<Vec<WorkspaceLensPackedVocabularyPosition>, WorkspaceLensError> {
    let pass_elements = checked_product(token_ids.len(), MPS_FULL_READOUT_TOP_K)?;
    for (name, got) in [
        ("packed readout first-pass IDs", first_ids.len()),
        ("packed readout first-pass logits", first_values.len()),
        ("packed readout second-pass IDs", second_ids.len()),
        ("packed readout second-pass logits", second_values.len()),
    ] {
        if got != pass_elements {
            return Err(WorkspaceLensError::ActivationSize {
                name,
                got,
                expected: pass_elements,
            });
        }
    }
    let mut positions = Vec::new();
    positions.try_reserve_exact(token_ids.len()).map_err(|_| {
        WorkspaceLensError::WorkspaceLensHostAllocationFailed {
            name: "packed full-vocabulary positions",
            elements: token_ids.len(),
        }
    })?;
    for (row, &source_token_id) in token_ids.iter().enumerate() {
        let source_position = start_position
            .checked_add(row)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        let predicts_position = source_position
            .checked_add(1)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        let row_base = checked_product(row, MPS_FULL_READOUT_TOP_K)?;
        let mut candidates = Vec::new();
        candidates
            .try_reserve_exact(FULL_READOUT_CANDIDATE_COUNT)
            .map_err(|_| WorkspaceLensError::WorkspaceLensHostAllocationFailed {
                name: "packed full-vocabulary candidates",
                elements: FULL_READOUT_CANDIDATE_COUNT,
            })?;
        for (ids, values, name) in [
            (first_ids, first_values, "packed first-pass logits"),
            (second_ids, second_values, "packed second-pass logits"),
        ] {
            for offset in 0..MPS_FULL_READOUT_TOP_K {
                let index = row_base
                    .checked_add(offset)
                    .ok_or(WorkspaceLensError::SizeOverflow)?;
                let token_id = ids[index];
                let logit = values[index];
                if token_id < 0 || token_id as u32 >= vocab_size {
                    return Err(WorkspaceLensError::InvalidFullReadoutToken {
                        token_id,
                        vocab_size,
                    });
                }
                if !logit.is_finite() {
                    return Err(WorkspaceLensError::NonFiniteTokenReadoutData { name, index });
                }
                if candidates
                    .iter()
                    .any(|score: &WorkspaceLensVocabularyScore| score.token_id == token_id as u32)
                {
                    return Err(WorkspaceLensError::DuplicateTokenReadoutId {
                        token_id: token_id as u32,
                    });
                }
                candidates.push(WorkspaceLensVocabularyScore {
                    token_id: token_id as u32,
                    logit,
                });
            }
        }
        candidates.sort_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        let mut scores = Vec::new();
        scores.try_reserve_exact(top_k).map_err(|_| {
            WorkspaceLensError::WorkspaceLensHostAllocationFailed {
                name: "packed full-vocabulary top-k scores",
                elements: top_k,
            }
        })?;
        scores.extend(candidates.into_iter().take(top_k));
        positions.push(WorkspaceLensPackedVocabularyPosition {
            source_position,
            source_token_id,
            predicts_position,
            scores,
            distribution_summary: None,
        });
    }
    Ok(positions)
}

pub(super) fn validate_packed_capture_layers(
    n_layers: u32,
    capture_layers: &[u32],
) -> Result<(), WorkspaceLensError> {
    if capture_layers.is_empty() {
        return Err(WorkspaceLensError::EmptyWorkspaceSourceLayers);
    }
    validate_capture_layers(n_layers, capture_layers)?;
    for (index, &layer) in capture_layers.iter().enumerate() {
        if capture_layers[..index].contains(&layer) {
            return Err(WorkspaceLensError::DuplicatePackedCaptureLayer { layer });
        }
    }
    Ok(())
}

impl<'model, 'sequence> WorkspaceLensSession<'model, 'sequence> {
    /// Allocate reusable model-bound storage for full-vocabulary readout.
    /// All subsequent row calls reuse these transport, hidden, logit, and
    /// compact top-k buffers.
    pub fn full_readout_workspace(
        &self,
        row_capacity: usize,
    ) -> Result<WorkspaceLensFullReadoutWorkspace<'model>, WorkspaceLensError> {
        if row_capacity == 0 {
            return Err(WorkspaceLensError::EmptyFullReadoutWorkspace);
        }
        let arch = self.arch();
        let hidden_size = arch.hidden_size as usize;
        let vocab_size = arch.vocab_size as usize;
        let allocation_bytes =
            full_readout_workspace_allocation_bytes(row_capacity, hidden_size, vocab_size)?;
        let logical_bytes = allocation_bytes.iter().try_fold(0usize, |total, &bytes| {
            total
                .checked_add(bytes)
                .ok_or(WorkspaceLensError::SizeOverflow)
        })?;
        enforce_workspace_lens_byte_budget("full-vocabulary GPU workspace", logical_bytes)?;

        let context = self.model.context();
        let _allocation = context.begin_allocation_transaction();
        let priced_bytes = allocation_bytes.iter().try_fold(0u64, |total, &bytes| {
            let bytes = u64::try_from(bytes).map_err(|_| WorkspaceLensError::SizeOverflow)?;
            let priced = context.shared_buffer_size_and_align(bytes)?.size;
            total
                .checked_add(priced)
                .ok_or(WorkspaceLensError::SizeOverflow)
        })?;
        let admission =
            evaluate_metal_memory_admission(priced_bytes, 0, context.memory_signals(), true);
        if !admission.admitted {
            return Err(WorkspaceLensError::FullReadoutMemoryAdmissionDenied {
                reason: admission.reason,
                requested_bytes: priced_bytes,
                working_set_headroom_bytes: admission.working_set_headroom_bytes,
                process_remaining_bytes: admission.signals.process_limit_remaining_bytes,
            });
        }

        Ok(WorkspaceLensFullReadoutWorkspace {
            model: self.model,
            row_capacity,
            transport_bound: false,
            transport: MetalTensor::zeros_f16(
                context,
                vec![hidden_size as u64, hidden_size as u64],
            )?,
            source: MetalTensor::zeros_f32(context, vec![row_capacity as u64, hidden_size as u64])?,
            transported: MetalTensor::zeros_f32(
                context,
                vec![row_capacity as u64, hidden_size as u64],
            )?,
            normalized: MetalTensor::zeros_f32(
                context,
                vec![row_capacity as u64, hidden_size as u64],
            )?,
            logits: MetalTensor::zeros_f32(context, vec![row_capacity as u64, vocab_size as u64])?,
            first_ids: MetalTensor::zeros_i32(
                context,
                vec![row_capacity as u64, MPS_FULL_READOUT_TOP_K as u64],
            )?,
            first_values: MetalTensor::zeros_f32(
                context,
                vec![row_capacity as u64, MPS_FULL_READOUT_TOP_K as u64],
            )?,
            second_ids: MetalTensor::zeros_i32(
                context,
                vec![row_capacity as u64, MPS_FULL_READOUT_TOP_K as u64],
            )?,
            second_values: MetalTensor::zeros_f32(
                context,
                vec![row_capacity as u64, MPS_FULL_READOUT_TOP_K as u64],
            )?,
        })
    }

    /// Extract selected LM-head score covectors and fold output RMSNorm gamma
    /// into them. Native GGUF `[H,V]` rows are gathered by `encode_get_rows_f32`;
    /// its lower-level dtype/orientation tests are the row-lookup oracle, while
    /// the model-backed workspace smoke checks this integration against logit
    /// ranking. The result omits the shared RMS denominator and softmax.
    pub fn selected_token_readouts(
        &self,
        token_ids: &[u32],
    ) -> Result<WorkspaceLensTokenReadouts, WorkspaceLensError> {
        self.selected_token_covectors(
            token_ids,
            WorkspaceLensTokenCovectorKind::DeployedLogitNumerator,
        )
    }

    /// Extract raw selected LM-head rows without folding output RMSNorm gamma.
    /// This matches the target covector used by Neuronpedia JLENS interventions.
    pub fn selected_token_raw_lm_head_rows(
        &self,
        token_ids: &[u32],
    ) -> Result<WorkspaceLensTokenReadouts, WorkspaceLensError> {
        self.selected_token_covectors(token_ids, WorkspaceLensTokenCovectorKind::RawLmHead)
    }

    pub(super) fn selected_token_covectors(
        &self,
        token_ids: &[u32],
        covector_kind: WorkspaceLensTokenCovectorKind,
    ) -> Result<WorkspaceLensTokenReadouts, WorkspaceLensError> {
        if token_ids.is_empty() {
            return Err(WorkspaceLensError::EmptyTokenReadoutSelection);
        }
        let arch = self.arch();
        let hidden_size = arch.hidden_size as usize;
        let vocab_size = arch.vocab_size;
        let selected_elements =
            validate_selected_token_request_size(token_ids.len(), vocab_size, hidden_size)?;
        validate_selected_token_ids(token_ids, vocab_size)?;

        let model = self.model.metal_model();
        let lm_head = &model.lm_head;
        let expected_lm_head_shape = [hidden_size, vocab_size as usize];
        if linear_shape(WorkspaceLensLinear::LmHead, lm_head).ok() != Some(expected_lm_head_shape) {
            return Err(WorkspaceLensError::InvalidTokenReadoutLmHeadShape {
                got: lm_head.shape.clone(),
                expected: expected_lm_head_shape,
            });
        }
        if !selected_readout_head_dtype_supported(lm_head.dtype) {
            return Err(WorkspaceLensError::UnsupportedTokenReadoutLmHeadDtype {
                dtype: lm_head.dtype,
            });
        }
        let output_norm = &model.output_norm;
        if output_norm.dtype != GgmlType::F32
            || output_norm.shape.as_slice() != [hidden_size as u64]
        {
            return Err(WorkspaceLensError::InvalidTokenReadoutOutputNorm {
                dtype: output_norm.dtype,
                shape: output_norm.shape.clone(),
                expected: hidden_size,
            });
        }

        let mut ids = Vec::new();
        ids.try_reserve_exact(token_ids.len()).map_err(|_| {
            WorkspaceLensError::WorkspaceLensHostAllocationFailed {
                name: "selected-token Metal IDs",
                elements: token_ids.len(),
            }
        })?;
        ids.extend(token_ids.iter().map(|&token_id| token_id as i32));
        let ids_tensor = MetalTensor::from_bytes(
            self.model.context(),
            bytemuck::cast_slice(&ids),
            vec![u64::try_from(ids.len()).map_err(|_| WorkspaceLensError::SizeOverflow)?],
            GgmlType::I32,
        )?;
        let selected = MetalTensor::zeros_f32(
            self.model.context(),
            row_shape(hidden_size, token_ids.len())?,
        )?;
        let command = self
            .model
            .context()
            .queue
            .commandBuffer()
            .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
        let encoder = KernelEncoder::begin(&command);
        let encode_result = encode_get_rows_f32(
            self.model.context(),
            &encoder,
            lm_head,
            &ids_tensor,
            &selected,
            token_ids.len(),
            hidden_size,
        );
        encoder.end();
        encode_result?;
        command.commit();
        crate::metal::wait_unchecked(&command);
        validate_completed_command(&command)?;

        let mut values =
            read_f32_fallible(&selected, selected_elements, "selected-token LM-head rows")?;
        if covector_kind == WorkspaceLensTokenCovectorKind::DeployedLogitNumerator {
            let gamma = read_f32_fallible(output_norm, hidden_size, "output norm gamma")?;
            multiply_token_readout_gamma_in_place(
                &mut values,
                &gamma,
                token_ids.len(),
                hidden_size,
            )?;
        }
        Ok(WorkspaceLensTokenReadouts {
            covector_kind,
            token_ids: try_clone_slice(token_ids, "selected-token IDs")?,
            hidden_size,
            lm_head_dtype: lm_head.dtype,
            lm_head_shape: expected_lm_head_shape,
            output_norm_dtype: output_norm.dtype,
            output_norm_shape: try_clone_slice(
                &output_norm.shape,
                "selected-token output norm shape",
            )?,
            values,
        })
    }

    /// Maximum selected covectors one F16 transport projection can admit while
    /// retaining all live transport, Metal, host result, and caller-owned banks.
    pub fn f16_transport_readout_query_capacity(
        &self,
        additional_live_bytes: usize,
    ) -> Result<usize, WorkspaceLensError> {
        let hidden_size = self.arch().hidden_size as usize;
        let transport_bytes = checked_product(
            checked_product(hidden_size, hidden_size)?,
            std::mem::size_of::<half::f16>(),
        )?;
        f16_transport_readout_query_capacity(hidden_size, transport_bytes, additional_live_bytes)
    }

    pub fn prepare_f16_transport_readouts(
        &self,
        transport_bytes: &[u8],
    ) -> Result<WorkspaceLensPreparedF16Transport<'model>, WorkspaceLensError> {
        let hidden_size = self.arch().hidden_size as usize;
        let expected_bytes = checked_product(
            checked_product(hidden_size, hidden_size)?,
            std::mem::size_of::<half::f16>(),
        )?;
        if transport_bytes.len() != expected_bytes {
            return Err(WorkspaceLensError::InvalidFullReadoutTransportSize {
                got: transport_bytes.len(),
                expected: expected_bytes,
            });
        }
        Ok(WorkspaceLensPreparedF16Transport {
            model: self.model,
            hidden_size,
            tensor: MetalTensor::from_bytes(
                self.model.context(),
                transport_bytes,
                vec![hidden_size as u64, hidden_size as u64],
                GgmlType::F16,
            )?,
        })
    }

    /// Project selected target covectors through one row-major F16 transport
    /// matrix. The result is query-major `[Q,H]` and computes
    /// `transport^T * covector` for each selected token.
    pub fn project_f16_transport_readouts(
        &self,
        transport_bytes: &[u8],
        readouts: &WorkspaceLensTokenReadouts,
    ) -> Result<Vec<f32>, WorkspaceLensError> {
        let transport = self.prepare_f16_transport_readouts(transport_bytes)?;
        self.project_prepared_f16_transport_readouts(&transport, readouts)
    }

    pub fn project_prepared_f16_transport_readouts(
        &self,
        transport: &WorkspaceLensPreparedF16Transport<'_>,
        readouts: &WorkspaceLensTokenReadouts,
    ) -> Result<Vec<f32>, WorkspaceLensError> {
        if !std::ptr::eq(self.model, transport.model) {
            return Err(WorkspaceLensError::PreparedF16TransportModelMismatch);
        }
        let hidden_size = self.arch().hidden_size as usize;
        if transport.hidden_size != hidden_size {
            return Err(WorkspaceLensError::ActivationSize {
                name: "prepared transport hidden size",
                got: transport.hidden_size,
                expected: hidden_size,
            });
        }
        if readouts.hidden_size != hidden_size {
            return Err(WorkspaceLensError::ActivationSize {
                name: "transport readout hidden size",
                got: readouts.hidden_size,
                expected: hidden_size,
            });
        }
        let n_query = readouts.token_ids.len();
        if n_query == 0 {
            return Err(WorkspaceLensError::EmptyTokenReadoutSelection);
        }
        let expected_covectors = n_query
            .checked_mul(hidden_size)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        if readouts.values.len() != expected_covectors {
            return Err(WorkspaceLensError::CotangentSize {
                got: readouts.values.len(),
                expected: expected_covectors,
                n_query,
                n_out: hidden_size,
            });
        }
        if let Some(index) = readouts.values.iter().position(|value| !value.is_finite()) {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "transport target covectors",
                index,
            });
        }
        let peak_bytes = f16_transport_readout_peak_bytes(
            hidden_size,
            usize::try_from(transport.tensor.n_bytes())
                .map_err(|_| WorkspaceLensError::SizeOverflow)?,
            n_query,
        )?;
        enforce_workspace_lens_byte_budget("F16 transport readout projection", peak_bytes)?;

        let context = self.model.context();
        let grad_output = MetalTensor::from_bytes(
            context,
            bytemuck::cast_slice(&readouts.values),
            vec![hidden_size as u64, n_query as u64],
            GgmlType::F32,
        )?;
        let grad_input = MetalTensor::zeros_f32(context, vec![hidden_size as u64, n_query as u64])?;
        let command = context
            .queue
            .commandBuffer()
            .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
        let encoder = KernelEncoder::begin(&command);
        let encode_result = encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            &transport.tensor,
            &grad_output,
            &grad_input,
            hidden_size,
            hidden_size,
            n_query,
        );
        encoder.end();
        encode_result?;
        command.commit();
        crate::metal::wait_unchecked(&command);
        validate_completed_command(&command)?;
        read_f32_fallible(
            &grad_input,
            expected_covectors,
            "projected F16 transport readouts",
        )
    }

    /// Advance through a bounded contiguous prompt once and capture one
    /// or more unique, caller-ordered zero-based post-block residual layers.
    pub fn forward_packed_post_block_capture(
        &mut self,
        token_ids: &[i32],
        capture_layers: &[u32],
    ) -> Result<WorkspaceLensPackedPostBlockCapture<'model>, WorkspaceLensError> {
        let position_count = token_ids.len();
        if position_count == 0 {
            return Err(WorkspaceLensError::EmptyFullReadoutPrompt);
        }
        if position_count > MAX_WORKSPACE_LENS_PACKED_READOUT_POSITIONS {
            return Err(WorkspaceLensError::PackedFullReadoutTooLong {
                got: position_count,
                max: MAX_WORKSPACE_LENS_PACKED_READOUT_POSITIONS,
            });
        }
        let arch = self.arch();
        validate_packed_capture_layers(arch.n_layer, capture_layers)?;
        for &token_id in token_ids {
            if token_id < 0 || token_id as u32 >= arch.vocab_size {
                return Err(MfError::BadToken(token_id, arch.vocab_size).into());
            }
        }
        self.sequence.ensure_can_append(position_count)?;
        let start_position = self.sequence.position();
        let end_position = start_position
            .checked_add(position_count)
            .ok_or(WorkspaceLensError::PositionOverflow(start_position))?;
        let start_position_u32 = u32::try_from(start_position)
            .map_err(|_| WorkspaceLensError::PositionOverflow(start_position))?;
        u32::try_from(end_position)
            .map_err(|_| WorkspaceLensError::PositionOverflow(end_position))?;

        let hidden_size = arch.hidden_size as usize;
        let model = self.model.metal_model();
        let capture_elements = checked_product(
            checked_product(position_count, capture_layers.len())?,
            hidden_size,
        )?;
        enforce_workspace_lens_byte_budget(
            "packed post-block capture",
            checked_product(capture_elements, std::mem::size_of::<f32>())?,
        )?;
        let token_ids_owned = try_clone_slice(token_ids, "packed capture token IDs")?;
        let layer_ids_owned = try_clone_slice(capture_layers, "packed capture layer IDs")?;

        let context = self.model.context();
        let values = MetalTensor::zeros_f32(
            context,
            vec![
                position_count as u64,
                capture_layers.len() as u64,
                hidden_size as u64,
            ],
        )?;
        let mut prefill_scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
            context,
            model,
            PACKED_FULL_READOUT_CHUNK_SIZE as u32,
            end_position,
        )?;

        let forward = self.model.forward();
        let prefill_started = Instant::now();
        let prefill_result = {
            let state = unsafe { self.sequence.metal_session_mut() };
            state.ensure_usable()?;
            prefill_tokens_with_multi_hidden_prompt_only_profiled(
                &forward,
                token_ids,
                start_position_u32,
                state,
                &mut prefill_scratch,
                capture_layers,
                &values,
            )
        };
        let packed_prefill_gpu_ms = match prefill_result {
            Ok(gpu_ms) => gpu_ms,
            Err(error) => {
                let state = unsafe { self.sequence.metal_session_mut() };
                state.poison("packed full-vocabulary prompt capture failed");
                return Err(error.into());
            }
        };
        let packed_prefill_wall_ms = prefill_started.elapsed().as_secs_f64() * 1e3;

        // The Metal session has consumed every token. Advance the safe runtime
        // position immediately, before any diagnostic or readout work can fail.
        if let Err(error) = self.sequence.advance_by(position_count) {
            let state = unsafe { self.sequence.metal_session_mut() };
            state.poison("packed full-vocabulary sequence advancement failed");
            return Err(error.into());
        }

        Ok(WorkspaceLensPackedPostBlockCapture {
            model: self.model,
            start_position,
            token_ids: token_ids_owned,
            layer_ids: layer_ids_owned,
            hidden_size,
            packed_prefill_gpu_ms,
            packed_prefill_wall_ms,
            values,
        })
    }

    /// Apply one row-major F16 transport and full-vocabulary readout while
    /// returning only the caller-selected transported rows. Source positions
    /// are zero-based absolute positions and preserve caller request order.
    pub fn apply_packed_capture_f16_transport_topk_with_vectors(
        &self,
        capture: &WorkspaceLensPackedPostBlockCapture<'_>,
        source_layer: u32,
        transport_bytes: &[u8],
        top_k: usize,
        transported_source_positions: &[usize],
    ) -> Result<WorkspaceLensPackedFullVocabularyReadout, WorkspaceLensError> {
        if !std::ptr::eq(self.model, capture.model) {
            return Err(WorkspaceLensError::PackedCaptureModelMismatch);
        }
        if top_k == 0 || top_k > MAX_FULL_READOUT_TOP_K {
            return Err(WorkspaceLensError::InvalidFullReadoutTopK {
                got: top_k,
                max: MAX_FULL_READOUT_TOP_K,
            });
        }
        let layer_slot = capture.layer_slot(source_layer)?;
        let arch = self.arch();
        let hidden_size = arch.hidden_size as usize;
        let vocab_size = arch.vocab_size as usize;
        validate_full_readout_transport_size(transport_bytes, hidden_size)?;
        let model = self.model.metal_model();
        validate_full_readout_tail(model, hidden_size, vocab_size)?;

        let position_count = capture.position_count();
        let transported_position_rows = validate_packed_transported_vector_positions(
            capture.start_position(),
            position_count,
            transported_source_positions,
        )?;
        let hidden_elements = checked_product(position_count, hidden_size)?;
        let logits_elements = checked_product(position_count, vocab_size)?;
        let compact_elements = checked_product(position_count, FULL_READOUT_CANDIDATE_COUNT)?;
        let hidden_bytes = checked_product(hidden_elements, std::mem::size_of::<f32>())?;
        let logits_bytes = checked_product(logits_elements, std::mem::size_of::<f32>())?;
        let compact_bytes = checked_product(compact_elements, 2 * std::mem::size_of::<u32>())?;
        let transported_vector_bytes = checked_product(
            checked_product(transported_position_rows.len(), hidden_size)?,
            std::mem::size_of::<f32>(),
        )?;
        let peak_bytes = transport_bytes
            .len()
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(hidden_bytes.checked_mul(3)?))
            .and_then(|bytes| bytes.checked_add(logits_bytes))
            .and_then(|bytes| bytes.checked_add(compact_bytes))
            .and_then(|bytes| bytes.checked_add(transported_vector_bytes))
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        enforce_workspace_lens_byte_budget(
            "packed full-vocabulary F16 transport readout",
            peak_bytes,
        )?;

        let context = self.model.context();
        let selected =
            MetalTensor::zeros_f32(context, vec![position_count as u64, hidden_size as u64])?;
        let transport = MetalTensor::from_bytes(
            context,
            transport_bytes,
            vec![hidden_size as u64, hidden_size as u64],
            GgmlType::F16,
        )?;
        let transported =
            MetalTensor::zeros_f32(context, vec![position_count as u64, hidden_size as u64])?;
        let normalized =
            MetalTensor::zeros_f32(context, vec![position_count as u64, hidden_size as u64])?;
        let logits =
            MetalTensor::zeros_f32(context, vec![position_count as u64, vocab_size as u64])?;
        let first_ids = MetalTensor::zeros_i32(
            context,
            vec![position_count as u64, MPS_FULL_READOUT_TOP_K as u64],
        )?;
        let first_values = MetalTensor::zeros_f32(
            context,
            vec![position_count as u64, MPS_FULL_READOUT_TOP_K as u64],
        )?;
        let second_ids = MetalTensor::zeros_i32(
            context,
            vec![position_count as u64, MPS_FULL_READOUT_TOP_K as u64],
        )?;
        let second_values = MetalTensor::zeros_f32(
            context,
            vec![position_count as u64, MPS_FULL_READOUT_TOP_K as u64],
        )?;

        let readout_started = Instant::now();
        let command = context
            .queue
            .commandBuffer()
            .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
        let encoder = KernelEncoder::begin(&command);
        let encode_result = (|| -> Result<(), WorkspaceLensError> {
            for position_row in 0..position_count {
                let source_offset = checked_product(
                    checked_product(position_row, capture.layer_ids.len())?
                        .checked_add(layer_slot)
                        .ok_or(WorkspaceLensError::SizeOverflow)?,
                    hidden_size,
                )?;
                let destination = selected.view_subrange(
                    u64::try_from(checked_product(position_row, hidden_size)?)
                        .map_err(|_| WorkspaceLensError::SizeOverflow)?,
                    vec![hidden_size as u64],
                );
                encode_copy_offset_f32(
                    context,
                    &encoder,
                    &capture.values,
                    source_offset,
                    &destination,
                    hidden_size,
                )?;
            }
            encode_mat_mat_f16_f32(
                context,
                &encoder,
                &transport,
                &selected,
                &transported,
                hidden_size,
                hidden_size,
                position_count,
            )?;
            encode_rms_norm_mul_rows_f32(
                context,
                &encoder,
                &transported,
                &model.output_norm,
                &normalized,
                position_count,
                hidden_size,
                RMS_EPS,
            )?;
            encode_mat_mat_dispatch(
                context,
                &encoder,
                &model.lm_head,
                &normalized,
                &logits,
                hidden_size,
                vocab_size,
                position_count,
            )?;
            Ok(())
        })();
        encoder.end();
        encode_result?;
        encode_mps_topk16_f32(
            context,
            &command,
            &logits,
            &first_ids,
            &first_values,
            position_count,
            vocab_size,
        )?;
        let mask_encoder = KernelEncoder::begin(&command);
        let mask_result = encode_mask_row_indices_f32(
            context,
            &mask_encoder,
            &logits,
            &first_ids,
            position_count,
            vocab_size,
            MPS_FULL_READOUT_TOP_K,
        );
        mask_encoder.end();
        mask_result?;
        encode_mps_topk16_f32(
            context,
            &command,
            &logits,
            &second_ids,
            &second_values,
            position_count,
            vocab_size,
        )?;
        command.commit();
        crate::metal::wait_unchecked(&command);
        validate_completed_command(&command)?;
        let readout_wall_ms = readout_started.elapsed().as_secs_f64() * 1e3;
        let readout_gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;

        let pass_elements = checked_product(position_count, MPS_FULL_READOUT_TOP_K)?;
        let first_ids =
            read_i32_fallible(&first_ids, pass_elements, "packed readout first-pass IDs")?;
        let first_values = read_f32_fallible(
            &first_values,
            pass_elements,
            "packed readout first-pass logits",
        )?;
        let second_ids =
            read_i32_fallible(&second_ids, pass_elements, "packed readout second-pass IDs")?;
        let second_values = read_f32_fallible(
            &second_values,
            pass_elements,
            "packed readout second-pass logits",
        )?;
        let positions = build_packed_vocabulary_positions(
            capture.token_ids(),
            capture.start_position(),
            top_k,
            arch.vocab_size,
            &first_ids,
            &first_values,
            &second_ids,
            &second_values,
        )?;
        let transported_vectors = read_packed_transported_vectors(
            &transported,
            capture,
            transported_source_positions,
            &transported_position_rows,
            hidden_size,
        )?;

        Ok(WorkspaceLensPackedFullVocabularyReadout {
            source_layer,
            start_position: capture.start_position(),
            position_count,
            top_k,
            packed_prefill_gpu_ms: capture.packed_prefill_gpu_ms(),
            packed_prefill_wall_ms: capture.packed_prefill_wall_ms(),
            readout_gpu_ms,
            readout_wall_ms,
            positions,
            transported_vectors,
        })
    }

    /// Apply one row-major F16 transport and return both deployed top-k logits
    /// and the transported target-coordinate residual before output RMSNorm.
    pub fn apply_f16_transport_topk_with_vector(
        &self,
        transport_bytes: &[u8],
        source_residual: &[f32],
        top_k: usize,
    ) -> Result<WorkspaceLensFullVocabularyReadoutWithVector, WorkspaceLensError> {
        if top_k == 0 || top_k > MAX_FULL_READOUT_TOP_K {
            return Err(WorkspaceLensError::InvalidFullReadoutTopK {
                got: top_k,
                max: MAX_FULL_READOUT_TOP_K,
            });
        }
        self.apply_f16_transport_logits_with_vector(transport_bytes, source_residual)?
            .into_topk(top_k)
    }

    pub fn apply_f16_transport_logits_with_vector(
        &self,
        transport_bytes: &[u8],
        source_residual: &[f32],
    ) -> Result<WorkspaceLensFullVocabularyLogitsWithVector, WorkspaceLensError> {
        let arch = self.arch();
        let hidden_size = arch.hidden_size as usize;
        let vocab_size = arch.vocab_size as usize;
        if source_residual.len() != hidden_size {
            return Err(WorkspaceLensError::ActivationSize {
                name: "full readout source residual",
                got: source_residual.len(),
                expected: hidden_size,
            });
        }
        if let Some(index) = source_residual.iter().position(|value| !value.is_finite()) {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "full readout source residual",
                index,
            });
        }
        let model = self.model.metal_model();
        validate_full_readout_transport_size(transport_bytes, hidden_size)?;
        validate_scalar_readout_tail(model, hidden_size, vocab_size)?;
        let hidden_bytes = checked_product(hidden_size, std::mem::size_of::<f32>())?;
        let logits_bytes = checked_product(vocab_size, std::mem::size_of::<f32>())?;
        let peak_bytes = transport_bytes
            .len()
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(hidden_bytes.checked_mul(5)?))
            .and_then(|bytes| bytes.checked_add(logits_bytes.checked_mul(2)?))
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        enforce_workspace_lens_byte_budget("full-vocabulary F16 transport readout", peak_bytes)?;

        let context = self.model.context();
        let transport = MetalTensor::from_bytes(
            context,
            transport_bytes,
            vec![hidden_size as u64, hidden_size as u64],
            GgmlType::F16,
        )?;
        let source = MetalTensor::from_bytes(
            context,
            bytemuck::cast_slice(source_residual),
            vec![hidden_size as u64],
            GgmlType::F32,
        )?;
        let transported = MetalTensor::zeros_f32(context, vec![hidden_size as u64])?;
        let normalized = MetalTensor::zeros_f32(context, vec![hidden_size as u64])?;
        let logits = MetalTensor::zeros_f32(context, vec![vocab_size as u64])?;
        let command = context
            .queue
            .commandBuffer()
            .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
        let encoder = KernelEncoder::begin(&command);
        let encode_result = (|| -> Result<(), WorkspaceLensError> {
            encode_mat_vec_f16_f32(
                context,
                &encoder,
                &transport,
                &source,
                &transported,
                hidden_size,
                hidden_size,
            )?;
            encode_scalar_lens_head(context, &encoder, model, &transported, &normalized, &logits)?;
            Ok(())
        })();
        encoder.end();
        encode_result?;
        command.commit();
        crate::metal::wait_unchecked(&command);
        validate_completed_command(&command)?;

        let transported_values = read_f32_fallible(
            &transported,
            hidden_size,
            "full readout transported residual",
        )?;
        if let Some(index) = transported_values
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "full readout transported residual",
                index,
            });
        }
        let rms_denominator_f64_recomputed = (transported_values
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            / hidden_size as f64
            + f64::from(RMS_EPS))
        .sqrt() as f32;
        let full_logits = read_f32_fallible(&logits, vocab_size, "full readout logits")?;
        if let Some(index) = full_logits.iter().position(|value| !value.is_finite()) {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "full readout logits",
                index,
            });
        }
        Ok(WorkspaceLensFullVocabularyLogitsWithVector {
            logits: full_logits,
            rms_denominator_f64_recomputed,
            transported_values,
        })
    }

    /// Fit arbitrary query-major target covectors `[Q,H]`. Every covector is
    /// applied at every valid target position, and the corresponding source
    /// positions are mean-reduced into owned `[K,Q,H]` values. Peak accounting
    /// includes caller covectors, fitting output, batched target/current/next
    /// gradients, and the batched VJP bank. Reduce `dim_batch` when that
    /// conservative peak exceeds [`MAX_WORKSPACE_LENS_OWNED_RESULT_BYTES`].
    #[allow(clippy::too_many_arguments)]
    pub fn workspace_fit_readouts_batched(
        &self,
        forward: &WorkspaceLensPromptForward,
        target_layer: u32,
        source_layers: &[u32],
        target_covectors: &[f32],
        skip_first: usize,
        dim_batch: usize,
        rule: WorkspaceLensRule,
    ) -> Result<WorkspaceLensReadouts, WorkspaceLensError> {
        if dim_batch == 0 {
            return Err(WorkspaceLensError::EmptyQueryBatch);
        }
        if dim_batch > MAX_WORKSPACE_LENS_DIM_BATCH {
            return Err(WorkspaceLensError::WorkspaceQueryBatchTooLarge {
                got: dim_batch,
                max: MAX_WORKSPACE_LENS_DIM_BATCH,
            });
        }
        if target_covectors.is_empty() {
            return Err(WorkspaceLensError::EmptyWorkspaceTargetCovectors);
        }
        let hidden_size = forward.hidden_size();
        if hidden_size == 0 || !target_covectors.len().is_multiple_of(hidden_size) {
            return Err(WorkspaceLensError::WorkspaceTargetCovectorSize {
                got: target_covectors.len(),
                hidden_size,
            });
        }
        if let Some(index) = target_covectors.iter().position(|value| !value.is_finite()) {
            return Err(WorkspaceLensError::NonFiniteWorkspaceTargetCovector { index });
        }
        let (_arch, n_tokens, hidden_elements) =
            self.validate_workspace_vjp_forward(forward, target_layer)?;
        validate_workspace_source_layers(target_layer, source_layers)?;
        let n_query = target_covectors.len() / hidden_size;
        let valid_positions = workspace_valid_position_range(n_tokens, skip_first)?;
        let n_valid_positions = valid_positions.len();
        let source_query_elements = checked_product(n_query, hidden_size)?;
        let output_elements = checked_product(source_layers.len(), source_query_elements)?;
        let chunk_queries = dim_batch.min(n_query);
        let chunk_target_elements = checked_product(chunk_queries, hidden_elements)?;
        let chunk_vjp_elements = checked_product(source_layers.len(), chunk_target_elements)?;
        let caller_target_elements = target_covectors.len();
        let peak_elements = output_elements
            .checked_add(caller_target_elements)
            .and_then(|elements| elements.checked_add(chunk_target_elements))
            .and_then(|elements| elements.checked_add(chunk_target_elements))
            .and_then(|elements| elements.checked_add(chunk_target_elements))
            .and_then(|elements| elements.checked_add(chunk_vjp_elements))
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        enforce_workspace_lens_byte_budget(
            "workspace readout fit",
            checked_product(peak_elements, std::mem::size_of::<f32>())?
                .checked_add(checked_product(
                    source_layers.len(),
                    std::mem::size_of::<u32>(),
                )?)
                .ok_or(WorkspaceLensError::SizeOverflow)?,
        )?;
        let mut values = try_zeroed_f32(output_elements, "workspace readout result")?;
        let mut diagnostics = Vec::new();
        let mut first_query = 0usize;
        for covectors in target_covectors.chunks(checked_product(dim_batch, hidden_size)?) {
            let chunk_queries = covectors.len() / hidden_size;
            let target_cotangents = build_workspace_target_bank(
                covectors,
                chunk_queries,
                forward.n_tokens(),
                hidden_size,
                valid_positions.clone(),
            )?;
            let vjp = self.workspace_vjp_batch(
                forward,
                target_layer,
                source_layers,
                &target_cotangents,
                chunk_queries,
                rule,
            )?;
            merge_workspace_diagnostics(&mut diagnostics, &vjp.diagnostics)?;
            reduce_workspace_vjp_readouts(
                &vjp.values,
                source_layers.len(),
                chunk_queries,
                forward.n_tokens(),
                hidden_size,
                valid_positions.clone(),
                &mut values,
                n_query,
                first_query,
            )?;
            first_query = first_query
                .checked_add(chunk_queries)
                .ok_or(WorkspaceLensError::SizeOverflow)?;
        }
        Ok(WorkspaceLensReadouts {
            target_layer,
            source_layers: try_clone_slice(source_layers, "workspace readout source layers")?,
            n_query,
            n_tokens: forward.n_tokens(),
            n_valid_positions,
            hidden_size,
            values,
            diagnostics,
        })
    }
}

impl WorkspaceLensFullVocabularyLogitsWithVector {
    fn into_topk(
        self,
        top_k: usize,
    ) -> Result<WorkspaceLensFullVocabularyReadoutWithVector, WorkspaceLensError> {
        let scores = exact_vocabulary_top_k(&self.logits, top_k)?;
        Ok(WorkspaceLensFullVocabularyReadoutWithVector {
            readout: WorkspaceLensFullVocabularyReadout {
                scores,
                rms_denominator_f64_recomputed: self.rms_denominator_f64_recomputed,
            },
            transported_values: self.transported_values,
        })
    }
}

#[cfg(test)]
mod scalar_logits_tests {
    use super::*;

    #[test]
    fn scalar_topk_preserves_logit_bits_and_vector() {
        let full = WorkspaceLensFullVocabularyLogitsWithVector {
            logits: vec![-0.0, 3.25, -7.0, 1.5],
            transported_values: vec![0.125, -2.0],
            rms_denominator_f64_recomputed: 1.417,
        };
        let top = full.clone().into_topk(4).unwrap();
        for score in top.readout.scores {
            assert_eq!(
                score.logit.to_bits(),
                full.logits[score.token_id as usize].to_bits()
            );
        }
        assert_eq!(top.transported_values, full.transported_values);
        assert_eq!(
            top.readout.rms_denominator_f64_recomputed,
            full.rms_denominator_f64_recomputed
        );
    }

    #[test]
    fn scalar_topk_rejects_nonfinite_logits() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(exact_vocabulary_top_k(&[0.0, value], 1).is_err());
        }
    }
}

impl<'model> WorkspaceLensPassiveSession<'model, '_> {
    pub fn full_readout_workspace(
        &self,
        rows: usize,
    ) -> Result<WorkspaceLensFullReadoutWorkspace<'model>, WorkspaceLensError> {
        self.inner.full_readout_workspace(rows)
    }

    pub fn apply_f16_transport_logits_with_vector(
        &self,
        matrix: &[u8],
        residual: &[f32],
    ) -> Result<WorkspaceLensFullVocabularyLogitsWithVector, WorkspaceLensError> {
        self.inner
            .apply_f16_transport_logits_with_vector(matrix, residual)
    }

    pub fn selected_token_readouts(
        &self,
        tokens: &[u32],
    ) -> Result<WorkspaceLensTokenReadouts, WorkspaceLensError> {
        self.inner.selected_token_readouts(tokens)
    }

    pub fn selected_token_raw_lm_head_rows(
        &self,
        tokens: &[u32],
    ) -> Result<WorkspaceLensTokenReadouts, WorkspaceLensError> {
        self.inner.selected_token_raw_lm_head_rows(tokens)
    }

    pub fn f16_transport_readout_query_capacity(
        &self,
        additional_live_bytes: usize,
    ) -> Result<usize, WorkspaceLensError> {
        self.inner
            .f16_transport_readout_query_capacity(additional_live_bytes)
    }

    pub fn prepare_f16_transport_readouts(
        &self,
        matrix: &[u8],
    ) -> Result<WorkspaceLensPreparedF16Transport<'model>, WorkspaceLensError> {
        self.inner.prepare_f16_transport_readouts(matrix)
    }

    pub fn project_prepared_f16_transport_readouts(
        &self,
        transport: &WorkspaceLensPreparedF16Transport<'_>,
        readouts: &WorkspaceLensTokenReadouts,
    ) -> Result<Vec<f32>, WorkspaceLensError> {
        self.inner
            .project_prepared_f16_transport_readouts(transport, readouts)
    }

    pub fn deployed_logits_from_post_block_residual(
        &self,
        source_residual: &[f32],
    ) -> Result<WorkspaceLensFullVocabularyLogitsWithVector, WorkspaceLensError> {
        let arch = self.inner.arch();
        let hidden_size = arch.hidden_size as usize;
        let vocab_size = arch.vocab_size as usize;
        if source_residual.len() != hidden_size {
            return Err(WorkspaceLensError::ActivationSize {
                name: "full readout source residual",
                got: source_residual.len(),
                expected: hidden_size,
            });
        }
        if let Some(index) = source_residual.iter().position(|value| !value.is_finite()) {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "full readout source residual",
                index,
            });
        }
        let model = self.inner.model.metal_model();
        validate_scalar_readout_tail(model, hidden_size, vocab_size)?;
        let hidden_bytes = checked_product(hidden_size, std::mem::size_of::<f32>())?;
        let logits_bytes = checked_product(vocab_size, std::mem::size_of::<f32>())?;
        let peak_bytes = hidden_bytes
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(logits_bytes.checked_mul(2)?))
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        enforce_workspace_lens_byte_budget("full-vocabulary identity readout", peak_bytes)?;

        let context = self.inner.model.context();
        let allocation = context.begin_allocation_transaction();
        admit_scalar_observer_allocations(
            context,
            &[hidden_bytes, hidden_bytes, logits_bytes],
            hidden_bytes
                .checked_add(logits_bytes)
                .ok_or(WorkspaceLensError::SizeOverflow)?,
        )?;
        let transported = MetalTensor::from_bytes(
            context,
            bytemuck::cast_slice(source_residual),
            vec![hidden_size as u64],
            GgmlType::F32,
        )?;
        let normalized = MetalTensor::zeros_f32(context, vec![hidden_size as u64])?;
        let logits = MetalTensor::zeros_f32(context, vec![vocab_size as u64])?;
        drop(allocation);
        let command = context
            .queue
            .commandBuffer()
            .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
        let encoder = KernelEncoder::begin(&command);
        let encode_result = (|| -> Result<(), WorkspaceLensError> {
            encode_scalar_lens_head(context, &encoder, model, &transported, &normalized, &logits)?;
            Ok(())
        })();
        encoder.end();
        encode_result?;
        command.commit();
        crate::metal::wait_unchecked(&command);
        validate_completed_command(&command)?;

        let transported_values = read_f32_fallible(
            &transported,
            hidden_size,
            "full readout transported residual",
        )?;
        if let Some(index) = transported_values
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "full readout transported residual",
                index,
            });
        }
        let rms_denominator_f64_recomputed = (transported_values
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            / hidden_size as f64
            + f64::from(RMS_EPS))
        .sqrt() as f32;
        let full_logits = read_f32_fallible(&logits, vocab_size, "full readout logits")?;
        if let Some(index) = full_logits.iter().position(|value| !value.is_finite()) {
            return Err(WorkspaceLensError::NonFiniteTokenReadoutData {
                name: "full readout logits",
                index,
            });
        }
        Ok(WorkspaceLensFullVocabularyLogitsWithVector {
            logits: full_logits,
            rms_denominator_f64_recomputed,
            transported_values,
        })
    }
}
