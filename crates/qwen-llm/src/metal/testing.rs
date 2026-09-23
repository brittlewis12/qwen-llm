//! One-shot test wrappers; not for the inference hot path.

use super::*;

pub(crate) fn one_shot<F>(ctx: &MetalContext, encode: F) -> Result<(), MetalError>
where
    F: FnOnce(&KernelEncoder) -> Result<(), MetalError>,
{
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    let encode_result = encode(&enc);
    enc.end();
    encode_result?;
    commit_and_wait(&cmd_buf)
}

pub(crate) fn read_back_f32(buf: &Buffer, n: usize) -> Vec<f32> {
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

/// One-shot weighted RMSNorm activation VJP for tests.
pub fn rms_norm_mul_vjp_rows_f32_readback_for_test(
    ctx: &MetalContext,
    x: &[f32],
    weight: &[f32],
    grad_output: &[f32],
    row_count: usize,
    n_dim: usize,
    eps: f32,
    rule: RmsNormVjpRule,
) -> Result<Vec<f32>, MetalError> {
    let x = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(x),
        vec![n_dim as u64, row_count as u64],
        GgmlType::F32,
    )?;
    let weight = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(weight),
        vec![n_dim as u64],
        GgmlType::F32,
    )?;
    let grad_output = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(grad_output),
        vec![n_dim as u64, row_count as u64],
        GgmlType::F32,
    )?;
    let grad_input = MetalTensor::zeros_f32(ctx, vec![n_dim as u64, row_count as u64])?;
    one_shot(ctx, |encoder| {
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &x,
            &weight,
            &grad_output,
            &grad_input,
            row_count,
            n_dim,
            eps,
            rule,
        )
    })?;
    Ok(read_back_f32(&grad_input.buffer, row_count * n_dim))
}

/// One-shot SwiGLU activation VJP for tests.
pub fn silu_mul_vjp_f32_readback_for_test(
    ctx: &MetalContext,
    gate: &[f32],
    up: &[f32],
    grad_output: &[f32],
    row_count: usize,
    n_dim: usize,
    rule: SwiGluVjpRule,
) -> Result<(Vec<f32>, Vec<f32>), MetalError> {
    let shape = vec![n_dim as u64, row_count as u64];
    let gate = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(gate),
        shape.clone(),
        GgmlType::F32,
    )?;
    let up = MetalTensor::from_bytes(ctx, bytemuck::cast_slice(up), shape.clone(), GgmlType::F32)?;
    let grad_output = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(grad_output),
        shape.clone(),
        GgmlType::F32,
    )?;
    let grad_gate = MetalTensor::zeros_f32(ctx, shape.clone())?;
    let grad_up = MetalTensor::zeros_f32(ctx, shape)?;
    one_shot(ctx, |encoder| {
        encode_silu_mul_vjp_f32(
            ctx,
            encoder,
            &gate,
            &up,
            &grad_output,
            &grad_gate,
            &grad_up,
            row_count,
            n_dim,
            rule,
        )
    })?;
    let len = row_count * n_dim;
    Ok((
        read_back_f32(&grad_gate.buffer, len),
        read_back_f32(&grad_up.buffer, len),
    ))
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
