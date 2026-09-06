//! Dense-FFN capture and VJP readbacks.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn dense_ffn_vjp_readback(
    context: &MetalContext,
    layer: u32,
    hidden_size: usize,
    intermediate_size: usize,
    pre_ffn_residual: &[f32],
    post_norm: &MetalTensor,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    down_weight: &MetalTensor,
    grad_output: &[f32],
    n_query: usize,
    rule: DenseFfnVjpRule,
) -> Result<Vec<f32>, WorkspaceLensError> {
    if n_query == 0 {
        return Err(WorkspaceLensError::EmptyQueryBatch);
    }
    let gate_id = WorkspaceLensLinear::Layer {
        index: layer,
        role: LinearRole::FfnGate,
    };
    let up_id = WorkspaceLensLinear::Layer {
        index: layer,
        role: LinearRole::FfnUp,
    };
    let down_id = WorkspaceLensLinear::Layer {
        index: layer,
        role: LinearRole::FfnDown,
    };
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnGate,
        linear_shape(gate_id, gate_weight)?,
        [hidden_size, intermediate_size],
    )?;
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnUp,
        linear_shape(up_id, up_weight)?,
        [hidden_size, intermediate_size],
    )?;
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnDown,
        linear_shape(down_id, down_weight)?,
        [intermediate_size, hidden_size],
    )?;
    if post_norm.dtype != GgmlType::F32 || post_norm.shape != [hidden_size as u64] {
        return Err(WorkspaceLensError::InvalidDenseFfnNorm {
            layer,
            dtype: post_norm.dtype,
            shape: post_norm.shape.clone(),
            expected: hidden_size,
        });
    }
    for (id, weight) in [
        (gate_id, gate_weight),
        (up_id, up_weight),
        (down_id, down_weight),
    ] {
        validate_vjp_dtype(id, weight)?;
    }
    if pre_ffn_residual.len() != hidden_size {
        return Err(WorkspaceLensError::ActivationSize {
            name: "pre-FFN residual",
            got: pre_ffn_residual.len(),
            expected: hidden_size,
        });
    }
    let hidden_query_elements = n_query
        .checked_mul(hidden_size)
        .ok_or(WorkspaceLensError::SizeOverflow)?;
    if grad_output.len() != hidden_query_elements {
        return Err(WorkspaceLensError::CotangentSize {
            got: grad_output.len(),
            expected: hidden_query_elements,
            n_query,
            n_out: hidden_size,
        });
    }
    let n_query_u64 = u64::try_from(n_query).map_err(|_| WorkspaceLensError::SizeOverflow)?;
    let hidden_u64 = u64::try_from(hidden_size).map_err(|_| WorkspaceLensError::SizeOverflow)?;
    let intermediate_u64 =
        u64::try_from(intermediate_size).map_err(|_| WorkspaceLensError::SizeOverflow)?;

    let pre_ffn_residual = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(pre_ffn_residual),
        vec![hidden_u64],
        GgmlType::F32,
    )?;
    let normalized = MetalTensor::zeros_f32(context, vec![hidden_u64])?;
    let gate = MetalTensor::zeros_f32(context, vec![intermediate_u64])?;
    let up = MetalTensor::zeros_f32(context, vec![intermediate_u64])?;
    let grad_output = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(grad_output),
        vec![hidden_u64, n_query_u64],
        GgmlType::F32,
    )?;
    let intermediate_query_shape = vec![intermediate_u64, n_query_u64];
    let hidden_query_shape = vec![hidden_u64, n_query_u64];
    let grad_inner = MetalTensor::zeros_f32(context, intermediate_query_shape.clone())?;
    let grad_gate = MetalTensor::zeros_f32(context, intermediate_query_shape.clone())?;
    let grad_up = MetalTensor::zeros_f32(context, intermediate_query_shape)?;
    let grad_norm_gate = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_norm_up = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_norm = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_ffn_input = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(context, hidden_query_shape)?;

    let command = context
        .queue
        .commandBuffer()
        .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), WorkspaceLensError> {
        encode_rms_norm_mul_f32(
            context,
            &encoder,
            &pre_ffn_residual,
            post_norm,
            &normalized,
            RMS_EPS,
        )?;
        encode_mat_vec_dispatch(
            context,
            &encoder,
            gate_weight,
            &normalized,
            &gate,
            hidden_size,
            intermediate_size,
        )?;
        encode_mat_vec_dispatch(
            context,
            &encoder,
            up_weight,
            &normalized,
            &up,
            hidden_size,
            intermediate_size,
        )?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            down_weight,
            &grad_output,
            &grad_inner,
            intermediate_size,
            hidden_size,
            n_query,
        )?;
        let (rms_rule, swiglu_rule) = match rule {
            DenseFfnVjpRule::Jacobian => (RmsNormVjpRule::Jacobian, SwiGluVjpRule::Jacobian),
            DenseFfnVjpRule::Relp => (
                RmsNormVjpRule::RelpDetachedScale,
                SwiGluVjpRule::RelpIdentityHalf,
            ),
        };
        encode_silu_mul_vjp_broadcast_f32(
            context,
            &encoder,
            &gate,
            &up,
            &grad_inner,
            &grad_gate,
            &grad_up,
            n_query,
            intermediate_size,
            swiglu_rule,
        )?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            gate_weight,
            &grad_gate,
            &grad_norm_gate,
            hidden_size,
            intermediate_size,
            n_query,
        )?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            up_weight,
            &grad_up,
            &grad_norm_up,
            hidden_size,
            intermediate_size,
            n_query,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_norm_gate,
            &grad_norm_up,
            &grad_norm,
        )?;
        encode_rms_norm_mul_vjp_broadcast_f32(
            context,
            &encoder,
            &pre_ffn_residual,
            post_norm,
            &grad_norm,
            &grad_ffn_input,
            n_query,
            hidden_size,
            RMS_EPS,
            rms_rule,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_output,
            &grad_ffn_input,
            &grad_input,
        )?;
        Ok(())
    })();
    encoder.end();
    encode_result?;
    command.commit();
    command.waitUntilCompleted();
    let status = command.status();
    let error = command.error();
    if status != MTLCommandBufferStatus::Completed || error.is_some() {
        return Err(WorkspaceLensError::CommandBuffer {
            status: format!("{status:?}"),
            error: format!("{error:?}"),
        });
    }
    Ok(read_f32(&grad_input, hidden_query_elements))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dense_ffn_vjp_rows_readback(
    context: &MetalContext,
    layer: u32,
    hidden_size: usize,
    intermediate_size: usize,
    pre_ffn_residuals: &[f32],
    post_norm: &MetalTensor,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    down_weight: &MetalTensor,
    grad_outputs: &[f32],
    n_rows: usize,
    rule: DenseFfnVjpRule,
) -> Result<Vec<f32>, WorkspaceLensError> {
    if n_rows == 0 {
        return Err(WorkspaceLensError::EmptyQueryBatch);
    }
    let gate_id = WorkspaceLensLinear::Layer {
        index: layer,
        role: LinearRole::FfnGate,
    };
    let up_id = WorkspaceLensLinear::Layer {
        index: layer,
        role: LinearRole::FfnUp,
    };
    let down_id = WorkspaceLensLinear::Layer {
        index: layer,
        role: LinearRole::FfnDown,
    };
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnGate,
        linear_shape(gate_id, gate_weight)?,
        [hidden_size, intermediate_size],
    )?;
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnUp,
        linear_shape(up_id, up_weight)?,
        [hidden_size, intermediate_size],
    )?;
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnDown,
        linear_shape(down_id, down_weight)?,
        [intermediate_size, hidden_size],
    )?;
    if post_norm.dtype != GgmlType::F32 || post_norm.shape != [hidden_size as u64] {
        return Err(WorkspaceLensError::InvalidDenseFfnNorm {
            layer,
            dtype: post_norm.dtype,
            shape: post_norm.shape.clone(),
            expected: hidden_size,
        });
    }
    for (id, weight) in [
        (gate_id, gate_weight),
        (up_id, up_weight),
        (down_id, down_weight),
    ] {
        validate_vjp_dtype(id, weight)?;
    }
    let hidden_elements = checked_product(n_rows, hidden_size)?;
    checked_product(n_rows, intermediate_size)?;
    for (name, values) in [
        ("pre-FFN residual rows", pre_ffn_residuals),
        ("post-block cotangent rows", grad_outputs),
    ] {
        if values.len() != hidden_elements {
            return Err(WorkspaceLensError::ActivationSize {
                name,
                got: values.len(),
                expected: hidden_elements,
            });
        }
    }
    let hidden_shape = row_shape(hidden_size, n_rows)?;
    let intermediate_shape = row_shape(intermediate_size, n_rows)?;
    let residuals = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(pre_ffn_residuals),
        hidden_shape.clone(),
        GgmlType::F32,
    )?;
    let normalized = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let gate = MetalTensor::zeros_f32(context, intermediate_shape.clone())?;
    let up = MetalTensor::zeros_f32(context, intermediate_shape.clone())?;
    let grad_output = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(grad_outputs),
        hidden_shape.clone(),
        GgmlType::F32,
    )?;
    let grad_inner = MetalTensor::zeros_f32(context, intermediate_shape.clone())?;
    let grad_gate = MetalTensor::zeros_f32(context, intermediate_shape.clone())?;
    let grad_up = MetalTensor::zeros_f32(context, intermediate_shape)?;
    let grad_norm_gate = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_norm_up = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_norm = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_ffn_input = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(context, hidden_shape)?;

    let command = context
        .queue
        .commandBuffer()
        .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), WorkspaceLensError> {
        encode_rms_norm_mul_rows_f32(
            context,
            &encoder,
            &residuals,
            post_norm,
            &normalized,
            n_rows,
            hidden_size,
            RMS_EPS,
        )?;
        for row in 0..n_rows {
            let normalized_row = row_view(&normalized, row, hidden_size);
            let gate_row = row_view(&gate, row, intermediate_size);
            let up_row = row_view(&up, row, intermediate_size);
            encode_mat_vec_dispatch(
                context,
                &encoder,
                gate_weight,
                &normalized_row,
                &gate_row,
                hidden_size,
                intermediate_size,
            )?;
            encode_mat_vec_dispatch(
                context,
                &encoder,
                up_weight,
                &normalized_row,
                &up_row,
                hidden_size,
                intermediate_size,
            )?;
        }
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            down_weight,
            &grad_output,
            &grad_inner,
            intermediate_size,
            hidden_size,
            n_rows,
        )?;
        let (rms_rule, swiglu_rule) = match rule {
            DenseFfnVjpRule::Jacobian => (RmsNormVjpRule::Jacobian, SwiGluVjpRule::Jacobian),
            DenseFfnVjpRule::Relp => (
                RmsNormVjpRule::RelpDetachedScale,
                SwiGluVjpRule::RelpIdentityHalf,
            ),
        };
        encode_silu_mul_vjp_f32(
            context,
            &encoder,
            &gate,
            &up,
            &grad_inner,
            &grad_gate,
            &grad_up,
            n_rows,
            intermediate_size,
            swiglu_rule,
        )?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            gate_weight,
            &grad_gate,
            &grad_norm_gate,
            hidden_size,
            intermediate_size,
            n_rows,
        )?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            up_weight,
            &grad_up,
            &grad_norm_up,
            hidden_size,
            intermediate_size,
            n_rows,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_norm_gate,
            &grad_norm_up,
            &grad_norm,
        )?;
        encode_rms_norm_mul_vjp_rows_f32(
            context,
            &encoder,
            &residuals,
            post_norm,
            &grad_norm,
            &grad_ffn_input,
            n_rows,
            hidden_size,
            RMS_EPS,
            rms_rule,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_output,
            &grad_ffn_input,
            &grad_input,
        )?;
        Ok(())
    })();
    encoder.end();
    encode_result?;
    command.commit();
    command.waitUntilCompleted();
    validate_completed_command(&command)?;
    Ok(read_f32(&grad_input, hidden_elements))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dense_ffn_vjp_query_rows_readback(
    context: &MetalContext,
    layer: u32,
    hidden_size: usize,
    intermediate_size: usize,
    pre_ffn_residuals: &[f32],
    post_norm: &MetalTensor,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    down_weight: &MetalTensor,
    grad_outputs: &[f32],
    n_rows: usize,
    n_query_batches: usize,
    rule: DenseFfnVjpRule,
) -> Result<Vec<f32>, WorkspaceLensError> {
    if n_rows == 0 || n_query_batches == 0 {
        return Err(WorkspaceLensError::EmptyQueryBatch);
    }
    let gate_id = WorkspaceLensLinear::Layer {
        index: layer,
        role: LinearRole::FfnGate,
    };
    let up_id = WorkspaceLensLinear::Layer {
        index: layer,
        role: LinearRole::FfnUp,
    };
    let down_id = WorkspaceLensLinear::Layer {
        index: layer,
        role: LinearRole::FfnDown,
    };
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnGate,
        linear_shape(gate_id, gate_weight)?,
        [hidden_size, intermediate_size],
    )?;
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnUp,
        linear_shape(up_id, up_weight)?,
        [hidden_size, intermediate_size],
    )?;
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnDown,
        linear_shape(down_id, down_weight)?,
        [intermediate_size, hidden_size],
    )?;
    if post_norm.dtype != GgmlType::F32 || post_norm.shape != [hidden_size as u64] {
        return Err(WorkspaceLensError::InvalidDenseFfnNorm {
            layer,
            dtype: post_norm.dtype,
            shape: post_norm.shape.clone(),
            expected: hidden_size,
        });
    }
    for (id, weight) in [
        (gate_id, gate_weight),
        (up_id, up_weight),
        (down_id, down_weight),
    ] {
        validate_vjp_dtype(id, weight)?;
    }

    let hidden_elements = checked_product(n_rows, hidden_size)?;
    checked_product(n_rows, intermediate_size)?;
    let query_rows = checked_product(n_query_batches, n_rows)?;
    let hidden_query_elements = checked_product(query_rows, hidden_size)?;
    checked_product(query_rows, intermediate_size)?;
    if pre_ffn_residuals.len() != hidden_elements {
        return Err(WorkspaceLensError::ActivationSize {
            name: "pre-FFN residual rows",
            got: pre_ffn_residuals.len(),
            expected: hidden_elements,
        });
    }
    if grad_outputs.len() != hidden_query_elements {
        return Err(WorkspaceLensError::ActivationSize {
            name: "post-block cotangent query rows",
            got: grad_outputs.len(),
            expected: hidden_query_elements,
        });
    }

    let hidden_shape = row_shape(hidden_size, n_rows)?;
    let intermediate_shape = row_shape(intermediate_size, n_rows)?;
    let hidden_query_shape = row_shape(hidden_size, query_rows)?;
    let intermediate_query_shape = row_shape(intermediate_size, query_rows)?;
    let residuals = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(pre_ffn_residuals),
        hidden_shape.clone(),
        GgmlType::F32,
    )?;
    let normalized = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let gate = MetalTensor::zeros_f32(context, intermediate_shape.clone())?;
    let up = MetalTensor::zeros_f32(context, intermediate_shape.clone())?;
    let grad_output = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(grad_outputs),
        hidden_query_shape.clone(),
        GgmlType::F32,
    )?;
    let grad_inner = MetalTensor::zeros_f32(context, intermediate_query_shape.clone())?;
    let grad_gate = MetalTensor::zeros_f32(context, intermediate_query_shape.clone())?;
    let grad_up = MetalTensor::zeros_f32(context, intermediate_query_shape)?;
    let grad_norm_gate = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_norm_up = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_norm = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_ffn_input = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(context, hidden_query_shape)?;

    let command = context
        .queue
        .commandBuffer()
        .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), WorkspaceLensError> {
        encode_rms_norm_mul_rows_f32(
            context,
            &encoder,
            &residuals,
            post_norm,
            &normalized,
            n_rows,
            hidden_size,
            RMS_EPS,
        )?;
        for row in 0..n_rows {
            let normalized_row = row_view(&normalized, row, hidden_size);
            let gate_row = row_view(&gate, row, intermediate_size);
            let up_row = row_view(&up, row, intermediate_size);
            encode_mat_vec_dispatch(
                context,
                &encoder,
                gate_weight,
                &normalized_row,
                &gate_row,
                hidden_size,
                intermediate_size,
            )?;
            encode_mat_vec_dispatch(
                context,
                &encoder,
                up_weight,
                &normalized_row,
                &up_row,
                hidden_size,
                intermediate_size,
            )?;
        }
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            down_weight,
            &grad_output,
            &grad_inner,
            intermediate_size,
            hidden_size,
            query_rows,
        )?;
        let (rms_rule, swiglu_rule) = match rule {
            DenseFfnVjpRule::Jacobian => (RmsNormVjpRule::Jacobian, SwiGluVjpRule::Jacobian),
            DenseFfnVjpRule::Relp => (
                RmsNormVjpRule::RelpDetachedScale,
                SwiGluVjpRule::RelpIdentityHalf,
            ),
        };
        let intermediate_query_elements = checked_product(n_rows, intermediate_size)?;
        for query in 0..n_query_batches {
            let offset = u64::try_from(checked_product(query, intermediate_query_elements)?)
                .map_err(|_| WorkspaceLensError::SizeOverflow)?;
            let grad_inner_query = grad_inner.view_subrange(offset, intermediate_shape.clone());
            let grad_gate_query = grad_gate.view_subrange(offset, intermediate_shape.clone());
            let grad_up_query = grad_up.view_subrange(offset, intermediate_shape.clone());
            encode_silu_mul_vjp_f32(
                context,
                &encoder,
                &gate,
                &up,
                &grad_inner_query,
                &grad_gate_query,
                &grad_up_query,
                n_rows,
                intermediate_size,
                swiglu_rule,
            )?;
        }
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            gate_weight,
            &grad_gate,
            &grad_norm_gate,
            hidden_size,
            intermediate_size,
            query_rows,
        )?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            up_weight,
            &grad_up,
            &grad_norm_up,
            hidden_size,
            intermediate_size,
            query_rows,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_norm_gate,
            &grad_norm_up,
            &grad_norm,
        )?;
        for query in 0..n_query_batches {
            let offset = u64::try_from(checked_product(query, hidden_elements)?)
                .map_err(|_| WorkspaceLensError::SizeOverflow)?;
            let grad_norm_query = grad_norm.view_subrange(offset, hidden_shape.clone());
            let grad_ffn_input_query = grad_ffn_input.view_subrange(offset, hidden_shape.clone());
            encode_rms_norm_mul_vjp_rows_f32(
                context,
                &encoder,
                &residuals,
                post_norm,
                &grad_norm_query,
                &grad_ffn_input_query,
                n_rows,
                hidden_size,
                RMS_EPS,
                rms_rule,
            )?;
        }
        encode_add_f32(
            context,
            &encoder,
            &grad_output,
            &grad_ffn_input,
            &grad_input,
        )?;
        Ok(())
    })();
    encoder.end();
    encode_result?;
    command.commit();
    command.waitUntilCompleted();
    validate_completed_command(&command)?;
    Ok(read_f32(&grad_input, hidden_query_elements))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_dense_ffn_weights(
    layer: u32,
    hidden_size: usize,
    intermediate_size: usize,
    post_norm: &MetalTensor,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    down_weight: &MetalTensor,
) -> Result<(), WorkspaceLensError> {
    let gate_id = WorkspaceLensLinear::Layer {
        index: layer,
        role: LinearRole::FfnGate,
    };
    let up_id = WorkspaceLensLinear::Layer {
        index: layer,
        role: LinearRole::FfnUp,
    };
    let down_id = WorkspaceLensLinear::Layer {
        index: layer,
        role: LinearRole::FfnDown,
    };
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnGate,
        linear_shape(gate_id, gate_weight)?,
        [hidden_size, intermediate_size],
    )?;
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnUp,
        linear_shape(up_id, up_weight)?,
        [hidden_size, intermediate_size],
    )?;
    validate_dense_ffn_shape(
        layer,
        LinearRole::FfnDown,
        linear_shape(down_id, down_weight)?,
        [intermediate_size, hidden_size],
    )?;
    if post_norm.dtype != GgmlType::F32 || post_norm.shape != [hidden_size as u64] {
        return Err(WorkspaceLensError::InvalidDenseFfnNorm {
            layer,
            dtype: post_norm.dtype,
            shape: post_norm.shape.clone(),
            expected: hidden_size,
        });
    }
    for (id, weight) in [
        (gate_id, gate_weight),
        (up_id, up_weight),
        (down_id, down_weight),
    ] {
        validate_vjp_dtype(id, weight)?;
    }
    Ok(())
}

pub(super) fn validate_dense_ffn_shape(
    layer: u32,
    role: LinearRole,
    got: [usize; 2],
    expected: [usize; 2],
) -> Result<(), WorkspaceLensError> {
    if got != expected {
        return Err(WorkspaceLensError::InvalidDenseFfnShape {
            layer,
            role,
            got,
            expected,
        });
    }
    Ok(())
}

impl<'model, 'sequence> WorkspaceLensSession<'model, 'sequence> {
    /// Advance one token while capturing both sides of selected dense FFNs.
    pub fn forward_token_with_dense_ffn_capture(
        &mut self,
        token_id: i32,
        capture_layers: &[u32],
    ) -> Result<WorkspaceLensDenseFfnForward, WorkspaceLensError> {
        self.sequence.ensure_can_append(1)?;
        validate_capture_layers(self.arch().n_layer, capture_layers)?;
        let position = self.sequence.position();
        let position_u32 =
            u32::try_from(position).map_err(|_| WorkspaceLensError::PositionOverflow(position))?;
        let hidden_size = self.arch().hidden_size as usize;
        let capture_len = capture_layers
            .len()
            .checked_mul(hidden_size)
            .ok_or(WorkspaceLensError::SizeOverflow)?;

        let forward = self.model.forward();
        let state = unsafe { self.sequence.metal_session_mut() };
        state.ensure_usable()?;
        let (logits, pre_ffn_residuals, post_block_residuals) = if capture_layers.is_empty() {
            (
                forward.single_token(token_id, position_u32, state)?,
                Vec::new(),
                Vec::new(),
            )
        } else {
            let pre_ffn = MetalTensor::zeros_f32(
                self.model.context(),
                vec![hidden_size as u64, capture_layers.len() as u64],
            )?;
            let post_block = MetalTensor::zeros_f32(
                self.model.context(),
                vec![hidden_size as u64, capture_layers.len() as u64],
            )?;
            let logits = forward.single_token_with_dense_ffn_capture(
                token_id,
                position_u32,
                state,
                capture_layers,
                &pre_ffn,
                &post_block,
            )?;
            (
                logits,
                read_f32(&pre_ffn, capture_len),
                read_f32(&post_block, capture_len),
            )
        };
        self.sequence.advance_by(1)?;

        Ok(WorkspaceLensDenseFfnForward {
            position,
            token_id,
            logits,
            capture: DenseFfnActivationCapture {
                layer_ids: capture_layers.to_vec(),
                hidden_size,
                pre_ffn_residuals,
                post_block_residuals,
            },
        })
    }

    pub(super) fn forward_token_with_dense_ffn_capture_no_tail(
        &mut self,
        token_id: i32,
        capture_layers: &[u32],
    ) -> Result<DenseFfnActivationCapture, WorkspaceLensError> {
        self.sequence.ensure_can_append(1)?;
        validate_capture_layers(self.arch().n_layer, capture_layers)?;
        let position = self.sequence.position();
        let position_u32 =
            u32::try_from(position).map_err(|_| WorkspaceLensError::PositionOverflow(position))?;
        let hidden_size = self.arch().hidden_size as usize;
        let capture_len = checked_product(capture_layers.len(), hidden_size)?;
        let forward = self.model.forward();
        let state = unsafe { self.sequence.metal_session_mut() };
        state.ensure_usable()?;
        let (pre_ffn_residuals, post_block_residuals) = if capture_layers.is_empty() {
            forward.single_token_no_tail(token_id, position_u32, state)?;
            (Vec::new(), Vec::new())
        } else {
            let shape = vec![hidden_size as u64, capture_layers.len() as u64];
            let pre_ffn = MetalTensor::zeros_f32(self.model.context(), shape.clone())?;
            let post_block = MetalTensor::zeros_f32(self.model.context(), shape)?;
            forward.single_token_with_dense_ffn_capture_no_tail(
                token_id,
                position_u32,
                state,
                capture_layers,
                &pre_ffn,
                &post_block,
            )?;
            (
                read_f32(&pre_ffn, capture_len),
                read_f32(&post_block, capture_len),
            )
        };
        self.sequence.advance_by(1)?;
        Ok(DenseFfnActivationCapture {
            layer_ids: capture_layers.to_vec(),
            hidden_size,
            pre_ffn_residuals,
            post_block_residuals,
        })
    }

    /// Reverse one dense FFN residual update from post-block cotangents to the
    /// post-mixer, pre-FFN residual captured during the matching forward.
    ///
    /// The method recomputes RMSNorm, gate, and up primals from `pre_ffn_residual`
    /// and keeps all resident weights opaque. `grad_output` and the result are
    /// query-major `[n_query, H]`. The unchanged residual branch is included.
    /// All three FFN linears must be resident as Q8_0, BF16, F16, or F32.
    pub fn dense_ffn_vjp(
        &self,
        layer: u32,
        pre_ffn_residual: &[f32],
        grad_output: &[f32],
        n_query: usize,
        rule: DenseFfnVjpRule,
    ) -> Result<DenseFfnVjp, WorkspaceLensError> {
        let (post_norm, gate, up, down) = self.resolve_dense_ffn(layer)?;
        let hidden_size = self.arch().hidden_size as usize;
        let intermediate_size = self.arch().intermediate_size as usize;
        let values = dense_ffn_vjp_readback(
            self.model.context(),
            layer,
            hidden_size,
            intermediate_size,
            pre_ffn_residual,
            post_norm,
            gate,
            up,
            down,
            grad_output,
            n_query,
            rule,
        )?;
        Ok(DenseFfnVjp {
            layer,
            n_query,
            hidden_size,
            values,
        })
    }

    pub(super) fn resolve_dense_ffn(
        &self,
        layer: u32,
    ) -> Result<(&MetalTensor, &MetalTensor, &MetalTensor, &MetalTensor), WorkspaceLensError> {
        let block = self.model.metal_model().blocks.get(layer as usize).ok_or(
            WorkspaceLensError::InvalidLayer {
                layer,
                n_layers: self.arch().n_layer,
            },
        )?;
        Ok(match block {
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
        })
    }
}
