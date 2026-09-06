//! Byte budgets, validation, and tensor helpers.

use super::*;

pub(super) fn rope_neox_rows_in_place(
    values: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    start_position: u32,
    rope_theta: f32,
    transpose: bool,
) -> Result<(), WorkspaceLensError> {
    let expected = checked_product(checked_product(n_tokens, n_heads)?, head_dim)?;
    if values.len() != expected
        || n_tokens == 0
        || n_heads == 0
        || head_dim == 0
        || n_rot == 0
        || n_rot > head_dim
        || !n_rot.is_multiple_of(2)
        || !rope_theta.is_finite()
        || rope_theta <= 0.0
    {
        return Err(WorkspaceLensError::ActivationSize {
            name: "RoPE row bank",
            got: values.len(),
            expected,
        });
    }
    let half = n_rot / 2;
    for token in 0..n_tokens {
        let position = start_position
            .checked_add(u32::try_from(token).map_err(|_| WorkspaceLensError::SizeOverflow)?)
            .ok_or(WorkspaceLensError::SizeOverflow)? as f32;
        for head in 0..n_heads {
            let base = (token * n_heads + head) * head_dim;
            for index in 0..half {
                let exponent = (2 * index) as f32 / n_rot as f32;
                let angle = position / rope_theta.powf(exponent);
                let (sin, cos) = angle.sin_cos();
                let left = values[base + index];
                let right = values[base + index + half];
                if transpose {
                    values[base + index] = left * cos + right * sin;
                    values[base + index + half] = -left * sin + right * cos;
                } else {
                    values[base + index] = left * cos - right * sin;
                    values[base + index + half] = left * sin + right * cos;
                }
            }
        }
    }
    Ok(())
}

pub(super) fn cpu_weighted_rms_vjp_rows(
    x: &[f32],
    weight: &[f32],
    grad_output: &[f32],
    n_rows: usize,
    width: usize,
    detach_scale: bool,
) -> Result<Vec<f32>, WorkspaceLensError> {
    let expected = checked_product(n_rows, width)?;
    if x.len() != expected || grad_output.len() != expected || weight.len() != width {
        return Err(WorkspaceLensError::ActivationSize {
            name: "CPU weighted RMSNorm VJP",
            got: x.len().min(grad_output.len()),
            expected,
        });
    }
    let mut result = vec![0.0f32; expected];
    for row in 0..n_rows {
        let base = row * width;
        let sum_squares = x[base..base + width]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let scale = (sum_squares / width as f32 + RMS_EPS).sqrt().recip();
        let dot = (0..width)
            .map(|index| x[base + index] * grad_output[base + index] * weight[index])
            .sum::<f32>();
        let correction = dot * scale * scale * scale / width as f32;
        for index in 0..width {
            let direct = grad_output[base + index] * weight[index] * scale;
            result[base + index] = if detach_scale {
                direct
            } else {
                direct - x[base + index] * correction
            };
        }
    }
    Ok(result)
}

pub(super) fn validate_workspace_source_layers(
    target_layer: u32,
    source_layers: &[u32],
) -> Result<(), WorkspaceLensError> {
    if source_layers.is_empty() {
        return Err(WorkspaceLensError::EmptyWorkspaceSourceLayers);
    }
    for &source in source_layers {
        if source >= target_layer {
            return Err(WorkspaceLensError::WorkspaceSourceNotBeforeTarget {
                source_layer: source,
                target_layer,
            });
        }
    }
    Ok(())
}

pub fn workspace_valid_position_range(
    n_tokens: usize,
    skip_first: usize,
) -> Result<std::ops::Range<usize>, WorkspaceLensError> {
    let minimum = skip_first
        .checked_add(2)
        .ok_or(WorkspaceLensError::SizeOverflow)?;
    if n_tokens < minimum {
        return Err(WorkspaceLensError::WorkspaceNoValidPositions {
            n_tokens,
            skip_first,
        });
    }
    Ok(skip_first..n_tokens - 1)
}

pub(super) fn merge_workspace_diagnostics(
    aggregate: &mut Vec<WorkspaceLensReplayDiagnostic>,
    current: &[WorkspaceLensReplayDiagnostic],
) -> Result<(), WorkspaceLensError> {
    validate_workspace_vjp_finite(&[], aggregate)?;
    validate_workspace_vjp_finite(&[], current)?;
    if aggregate.is_empty() {
        aggregate.extend_from_slice(current);
        return Ok(());
    }
    if aggregate.len() != current.len() {
        return Err(WorkspaceLensError::WorkspaceDiagnosticScheduleMismatch);
    }
    for (aggregate, current) in aggregate.iter_mut().zip(current) {
        if aggregate.layer != current.layer || aggregate.kind != current.kind {
            return Err(WorkspaceLensError::WorkspaceDiagnosticScheduleMismatch);
        }
        aggregate.residual_replay_max_abs_error = aggregate
            .residual_replay_max_abs_error
            .max(current.residual_replay_max_abs_error);
    }
    Ok(())
}

pub(super) fn checked_product(left: usize, right: usize) -> Result<usize, WorkspaceLensError> {
    left.checked_mul(right)
        .ok_or(WorkspaceLensError::SizeOverflow)
}

pub(super) fn enforce_workspace_lens_byte_budget(
    name: &'static str,
    requested_bytes: usize,
) -> Result<(), WorkspaceLensError> {
    if requested_bytes > MAX_WORKSPACE_LENS_OWNED_RESULT_BYTES {
        return Err(WorkspaceLensError::WorkspaceLensResultByteBudgetExceeded {
            name,
            requested_bytes,
            max_bytes: MAX_WORKSPACE_LENS_OWNED_RESULT_BYTES,
        });
    }
    Ok(())
}

pub(super) fn validate_selected_token_request_size(
    count: usize,
    vocab_size: u32,
    hidden_size: usize,
) -> Result<usize, WorkspaceLensError> {
    if count > vocab_size as usize {
        return Err(WorkspaceLensError::TokenReadoutCountExceedsVocabulary {
            got: count,
            vocab_size,
        });
    }
    let selected_elements = checked_product(count, hidden_size)?;
    enforce_workspace_lens_byte_budget(
        "selected-token readouts",
        selected_token_peak_bytes(count, hidden_size, selected_elements)?,
    )?;
    Ok(selected_elements)
}

pub(super) fn selected_token_peak_bytes(
    count: usize,
    hidden_size: usize,
    selected_elements: usize,
) -> Result<usize, WorkspaceLensError> {
    // Simultaneous peak: Metal gather + host readback/result, host gamma,
    // host/Metal/result ID copies, conservative HashSet buckets, and shape copy.
    const LIVE_SELECTED_BANKS: usize = 2;
    const LIVE_ID_COPIES: usize = 4;
    const HASHSET_BYTES_PER_TOKEN: usize = 32;

    let selected_bank_bytes = checked_product(selected_elements, std::mem::size_of::<f32>())?;
    let selected_banks = checked_product(selected_bank_bytes, LIVE_SELECTED_BANKS)?;
    let gamma_bytes = checked_product(hidden_size, std::mem::size_of::<f32>())?;
    let id_copy_bytes = checked_product(
        checked_product(count, std::mem::size_of::<u32>())?,
        LIVE_ID_COPIES,
    )?;
    let hashset_bytes = checked_product(count, HASHSET_BYTES_PER_TOKEN)?;
    selected_banks
        .checked_add(gamma_bytes)
        .and_then(|bytes| bytes.checked_add(id_copy_bytes))
        .and_then(|bytes| bytes.checked_add(hashset_bytes))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
        .ok_or(WorkspaceLensError::SizeOverflow)
}

pub(super) fn try_zeroed_f32(
    elements: usize,
    name: &'static str,
) -> Result<Vec<f32>, WorkspaceLensError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| WorkspaceLensError::WorkspaceLensHostAllocationFailed { name, elements })?;
    values.resize(elements, 0.0);
    Ok(values)
}

pub(super) fn try_clone_slice<T: Copy>(
    values: &[T],
    name: &'static str,
) -> Result<Vec<T>, WorkspaceLensError> {
    let mut output = Vec::new();
    output.try_reserve_exact(values.len()).map_err(|_| {
        WorkspaceLensError::WorkspaceLensHostAllocationFailed {
            name,
            elements: values.len(),
        }
    })?;
    output.extend_from_slice(values);
    Ok(output)
}

pub(super) fn validate_selected_token_ids(
    token_ids: &[u32],
    vocab_size: u32,
) -> Result<(), WorkspaceLensError> {
    if token_ids.is_empty() {
        return Err(WorkspaceLensError::EmptyTokenReadoutSelection);
    }
    let mut unique = std::collections::HashSet::new();
    unique.try_reserve(token_ids.len()).map_err(|_| {
        WorkspaceLensError::WorkspaceLensHostAllocationFailed {
            name: "selected-token uniqueness set",
            elements: token_ids.len(),
        }
    })?;
    for &token_id in token_ids {
        if token_id >= vocab_size || token_id > i32::MAX as u32 {
            return Err(WorkspaceLensError::TokenReadoutIdOutOfRange {
                token_id,
                vocab_size,
            });
        }
        if !unique.insert(token_id) {
            return Err(WorkspaceLensError::DuplicateTokenReadoutId { token_id });
        }
    }
    Ok(())
}

pub(super) fn write_tensor_bytes(
    tensor: &MetalTensor,
    bytes: &[u8],
    expected: usize,
    name: &'static str,
) -> Result<(), WorkspaceLensError> {
    if bytes.len() != expected {
        return Err(WorkspaceLensError::ActivationSize {
            name,
            got: bytes.len(),
            expected,
        });
    }
    debug_assert!(tensor.is_writable());
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
    }
    Ok(())
}

pub(super) fn read_f32_fallible(
    tensor: &MetalTensor,
    len: usize,
    name: &'static str,
) -> Result<Vec<f32>, WorkspaceLensError> {
    let mut output = try_zeroed_f32(len, name)?;
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<f32>();
        std::ptr::copy_nonoverlapping(source, output.as_mut_ptr(), len);
    }
    Ok(output)
}

pub(super) fn read_i32_fallible(
    tensor: &MetalTensor,
    len: usize,
    name: &'static str,
) -> Result<Vec<i32>, WorkspaceLensError> {
    let mut output = Vec::new();
    output.try_reserve_exact(len).map_err(|_| {
        WorkspaceLensError::WorkspaceLensHostAllocationFailed {
            name,
            elements: len,
        }
    })?;
    output.resize(len, 0);
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<i32>();
        std::ptr::copy_nonoverlapping(source, output.as_mut_ptr(), len);
    }
    Ok(output)
}

pub(super) fn row_shape(width: usize, rows: usize) -> Result<Vec<u64>, WorkspaceLensError> {
    Ok(vec![
        u64::try_from(width).map_err(|_| WorkspaceLensError::SizeOverflow)?,
        u64::try_from(rows).map_err(|_| WorkspaceLensError::SizeOverflow)?,
    ])
}

pub(super) fn flat_f32(
    context: &MetalContext,
    elements: usize,
) -> Result<MetalTensor, WorkspaceLensError> {
    Ok(MetalTensor::zeros_f32(
        context,
        vec![u64::try_from(elements).map_err(|_| WorkspaceLensError::SizeOverflow)?],
    )?)
}

pub(super) fn f32_from_slice(
    context: &MetalContext,
    values: &[f32],
) -> Result<MetalTensor, WorkspaceLensError> {
    Ok(MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(values),
        vec![u64::try_from(values.len()).map_err(|_| WorkspaceLensError::SizeOverflow)?],
        GgmlType::F32,
    )?)
}

pub(super) fn row_view(tensor: &MetalTensor, row: usize, width: usize) -> MetalTensor {
    tensor.view_subrange((row * width) as u64, vec![width as u64])
}

pub(super) fn flat_query_view(
    tensor: &MetalTensor,
    query: usize,
    elements: usize,
) -> Result<MetalTensor, WorkspaceLensError> {
    let offset = u64::try_from(checked_product(query, elements)?)
        .map_err(|_| WorkspaceLensError::SizeOverflow)?;
    let elements = u64::try_from(elements).map_err(|_| WorkspaceLensError::SizeOverflow)?;
    Ok(tensor.view_subrange(offset, vec![elements]))
}

pub(super) fn validate_completed_command(
    command: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>>,
) -> Result<(), WorkspaceLensError> {
    let status = command.status();
    let error = command.error();
    if status != MTLCommandBufferStatus::Completed || error.is_some() {
        return Err(WorkspaceLensError::CommandBuffer {
            status: format!("{status:?}"),
            error: format!("{error:?}"),
        });
    }
    Ok(())
}

pub(super) fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return f32::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(&left, &right)| finite_abs_difference(left, right))
        .fold(0.0f32, f32::max)
}

pub(super) fn finite_abs_difference(left: f32, right: f32) -> f32 {
    if !left.is_finite() || !right.is_finite() {
        return f32::INFINITY;
    }
    let difference = (left - right).abs();
    if difference.is_finite() {
        difference
    } else {
        f32::INFINITY
    }
}

pub(super) fn validate_vjp_dtype(
    id: WorkspaceLensLinear,
    weight: &MetalTensor,
) -> Result<(), WorkspaceLensError> {
    if !matches!(
        weight.dtype,
        GgmlType::Q8_0 | GgmlType::BF16 | GgmlType::F16 | GgmlType::F32
    ) {
        return Err(WorkspaceLensError::UnsupportedLinearDtype {
            id,
            dtype: weight.dtype,
        });
    }
    Ok(())
}

pub(super) fn linear_shape(
    id: WorkspaceLensLinear,
    tensor: &MetalTensor,
) -> Result<[usize; 2], WorkspaceLensError> {
    let [n_in, n_out] = tensor.shape.as_slice() else {
        return Err(WorkspaceLensError::InvalidLinearShape {
            id,
            shape: tensor.shape.clone(),
        });
    };
    Ok([
        usize::try_from(*n_in).map_err(|_| WorkspaceLensError::SizeOverflow)?,
        usize::try_from(*n_out).map_err(|_| WorkspaceLensError::SizeOverflow)?,
    ])
}

pub(super) fn read_f32(tensor: &MetalTensor, len: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; len];
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<f32>();
        std::ptr::copy_nonoverlapping(source, output.as_mut_ptr(), len);
    }
    output
}

impl<'model, 'sequence> WorkspaceLensSession<'model, 'sequence> {
    pub(super) fn validate_workspace_weights(&self) -> Result<(), WorkspaceLensError> {
        let arch = self.arch();
        if self.model.metal_model().blocks.len() != arch.n_layer as usize {
            return Err(WorkspaceLensError::ActivationSize {
                name: "resident workspace block schedule",
                got: self.model.metal_model().blocks.len(),
                expected: arch.n_layer as usize,
            });
        }
        for (layer, block) in self.model.metal_model().blocks.iter().enumerate() {
            let layer = u32::try_from(layer).map_err(|_| WorkspaceLensError::SizeOverflow)?;
            let (post_norm, gate, up, down) = match block {
                MetalBlock::Gdn(block) => (
                    &block.post_attn_norm,
                    &block.ffn_gate,
                    &block.ffn_up,
                    &block.ffn_down,
                ),
                MetalBlock::Attn(block) => (
                    &block.post_attn_norm,
                    &block.ffn_gate,
                    &block.ffn_up,
                    &block.ffn_down,
                ),
            };
            validate_dense_ffn_weights(
                layer,
                arch.hidden_size as usize,
                arch.intermediate_size as usize,
                post_norm,
                gate,
                up,
                down,
            )?;
            match block {
                MetalBlock::Gdn(block) => {
                    let geometry = GdnGeometry::new(layer, arch)?;
                    validate_gdn_weights(layer, block, geometry)?;
                }
                MetalBlock::Attn(block) => {
                    let geometry = AttnGeometry::new(arch)?;
                    validate_attn_weights(layer, block, geometry)?;
                }
            }
        }
        Ok(())
    }
}
