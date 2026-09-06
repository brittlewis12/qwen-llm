//! Frozen-linear VJP kernels for workspace-lens research.

use super::*;

/// Ordinary sigmoid VJP using the saved forward output.
pub fn encode_sigmoid_output_vjp_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    sigmoid_output: &MetalTensor,
    grad_output: &MetalTensor,
    grad_input: &MetalTensor,
) -> Result<(), MetalError> {
    const KERNEL: &str = "sigmoid_output_vjp";
    let shape = sigmoid_output.shape.clone();
    let (n, _) = checked_shape_bytes(&shape, std::mem::size_of::<f32>())?;
    if n == 0 || u32::try_from(n).is_err() {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("element count {n} must fit nonzero u32"),
        });
    }
    validate_compact_f32_tensor(KERNEL, "sigmoid_output", sigmoid_output, &shape, false)?;
    validate_compact_f32_tensor(KERNEL, "grad_output", grad_output, &shape, false)?;
    validate_compact_f32_tensor(KERNEL, "grad_input", grad_input, &shape, true)?;
    if tensor_ranges_overlap(grad_input, sigmoid_output)
        || tensor_ranges_overlap(grad_input, grad_output)
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "grad_input must not overlap the sigmoid output or incoming gradient".into(),
        });
    }
    let pipeline = ctx.pipeline("kernel_sigmoid_output_vjp_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, sigmoid_output);
    enc.set_tensor(2, grad_output);
    enc.set_tensor(3, grad_input);
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SwiGluVjpRule {
    Jacobian,
    RelpIdentityHalf,
}

/// Activation VJP for `silu(gate) * up` over compact rows.
pub fn encode_silu_mul_vjp_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    grad_output: &MetalTensor,
    grad_gate: &MetalTensor,
    grad_up: &MetalTensor,
    row_count: usize,
    n_dim: usize,
    rule: SwiGluVjpRule,
) -> Result<(), MetalError> {
    encode_silu_mul_vjp_impl_f32(
        ctx,
        enc,
        gate,
        up,
        grad_output,
        grad_gate,
        grad_up,
        row_count,
        n_dim,
        rule,
        row_count,
        false,
    )
}

/// Activation VJP for a query bank sharing one SwiGLU primal row.
///
/// `gate` and `up` have shape `[n_dim]`; cotangents and gradient outputs have
/// compact-row shape `[n_dim, row_count]`.
pub fn encode_silu_mul_vjp_broadcast_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    grad_output: &MetalTensor,
    grad_gate: &MetalTensor,
    grad_up: &MetalTensor,
    row_count: usize,
    n_dim: usize,
    rule: SwiGluVjpRule,
) -> Result<(), MetalError> {
    encode_silu_mul_vjp_impl_f32(
        ctx,
        enc,
        gate,
        up,
        grad_output,
        grad_gate,
        grad_up,
        row_count,
        n_dim,
        rule,
        1,
        true,
    )
}

/// Activation VJP for a query bank periodically reusing SwiGLU primal rows.
///
/// `gate` and `up` have compact-row shape `[n_dim, primal_row_count]`;
/// cotangent row `r` uses primal row `r % primal_row_count`.
#[allow(clippy::too_many_arguments)]
pub fn encode_silu_mul_vjp_periodic_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    grad_output: &MetalTensor,
    grad_gate: &MetalTensor,
    grad_up: &MetalTensor,
    row_count: usize,
    primal_row_count: usize,
    n_dim: usize,
    rule: SwiGluVjpRule,
) -> Result<(), MetalError> {
    encode_silu_mul_vjp_impl_f32(
        ctx,
        enc,
        gate,
        up,
        grad_output,
        grad_gate,
        grad_up,
        row_count,
        n_dim,
        rule,
        primal_row_count,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_silu_mul_vjp_impl_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    grad_output: &MetalTensor,
    grad_gate: &MetalTensor,
    grad_up: &MetalTensor,
    row_count: usize,
    n_dim: usize,
    rule: SwiGluVjpRule,
    primal_row_count: usize,
    rank_one_primal: bool,
) -> Result<(), MetalError> {
    const KERNEL: &str = "silu_mul_vjp";
    if row_count == 0
        || primal_row_count == 0
        || primal_row_count > row_count
        || !row_count.is_multiple_of(primal_row_count)
        || (rank_one_primal && primal_row_count != 1)
        || n_dim == 0
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected periodic nonzero primal rows and width, got rows={row_count} primal_rows={primal_row_count} width={n_dim}"
            ),
        });
    }
    let row_count_u32 = u32::try_from(row_count).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: "row count exceeds u32".into(),
    })?;
    let n_dim_u32 = u32::try_from(n_dim).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: "row width exceeds u32".into(),
    })?;
    let primal_row_count_u32 =
        u32::try_from(primal_row_count).map_err(|_| MetalError::BadShape {
            kernel: KERNEL,
            detail: "primal row count exceeds u32".into(),
        })?;
    let element_count = row_count
        .checked_mul(n_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "element count overflow".into(),
        })?;
    let element_count_u32 = u32::try_from(element_count).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: "element count exceeds u32".into(),
    })?;
    let shape = vec![u64::from(n_dim_u32), u64::from(row_count_u32)];
    let primal_shape = if rank_one_primal {
        vec![u64::from(n_dim_u32)]
    } else {
        vec![u64::from(n_dim_u32), u64::from(primal_row_count_u32)]
    };
    let primals = [gate, up];
    let outputs = [grad_gate, grad_up];
    if primals
        .iter()
        .any(|tensor| tensor.dtype != GgmlType::F32 || tensor.shape != primal_shape)
        || grad_output.dtype != GgmlType::F32
        || grad_output.shape != shape
        || outputs
            .iter()
            .any(|tensor| tensor.dtype != GgmlType::F32 || tensor.shape != shape)
        || outputs.iter().any(|tensor| !tensor.is_writable())
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected F32 primals {primal_shape:?} and cotangent/outputs {shape:?} with writable outputs"
            ),
        });
    }
    let (_, primal_bytes) = checked_shape_bytes(&primal_shape, std::mem::size_of::<f32>())?;
    let (_, bytes) = checked_shape_bytes(&shape, std::mem::size_of::<f32>())?;
    if primals
        .iter()
        .any(|tensor| !tensor_physical_range_valid(tensor, primal_bytes, 4))
        || !tensor_physical_range_valid(grad_output, bytes, 4)
        || outputs
            .iter()
            .any(|tensor| !tensor_physical_range_valid(tensor, bytes, 4))
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "tensor has an unaligned or out-of-buffer byte range".into(),
        });
    }
    if tensor_ranges_overlap(grad_gate, grad_up)
        || outputs.iter().any(|output| {
            primals
                .iter()
                .chain(std::iter::once(&grad_output))
                .any(|input| tensor_ranges_overlap(output, input))
        })
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "gradient outputs must not overlap primals, incoming gradient, or each other"
                .into(),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        n_dim: u32,
        relp_identity_half: u32,
        primal_row_count: u32,
    }
    let pso = ctx.pipeline("kernel_silu_mul_vjp_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: element_count_u32,
            n_dim: n_dim_u32,
            relp_identity_half: u32::from(rule == SwiGluVjpRule::RelpIdentityHalf),
            primal_row_count: primal_row_count_u32,
        },
    );
    enc.set_tensor(1, gate);
    enc.set_tensor(2, up);
    enc.set_tensor(3, grad_output);
    enc.set_tensor(4, grad_gate);
    enc.set_tensor(5, grad_up);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: element_count.div_ceil(tg_threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(crate) fn validate_frozen_linear_q8_0_vjp_f32(
    weight: &MetalTensor,
    grad_output: &MetalTensor,
    grad_input: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(u32, u32), MetalError> {
    const KERNEL: &str = "frozen_linear_q8_0_vjp";
    if n_in == 0 || n_out == 0 || n_query == 0 || !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected nonzero dimensions with n_in divisible by 32, got n_in={n_in} n_out={n_out} n_query={n_query}"
            ),
        });
    }

    let n_in_u32 = u32::try_from(n_in).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("n_in={n_in} exceeds u32 indexing"),
    })?;
    let n_out_u32 = u32::try_from(n_out).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("n_out={n_out} exceeds u32 indexing"),
    })?;
    u32::try_from(n_query).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("n_query={n_query} exceeds Metal uint grid indexing"),
    })?;

    let weight_shape = vec![u64::from(n_in_u32), u64::from(n_out_u32)];
    let grad_output_shape = vec![u64::from(n_out_u32), n_query as u64];
    let grad_input_shape = vec![u64::from(n_in_u32), n_query as u64];
    if weight.dtype != GgmlType::Q8_0
        || grad_output.dtype != GgmlType::F32
        || grad_input.dtype != GgmlType::F32
        || !grad_input.is_writable()
        || weight.shape != weight_shape
        || grad_output.shape != grad_output_shape
        || grad_input.shape != grad_input_shape
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected Q8_0 {:?}, F32 {:?} -> writable F32 {:?}; got {:?} {:?}, {:?} {:?}, {:?} {:?} writable={}",
                weight_shape,
                grad_output_shape,
                grad_input_shape,
                weight.dtype,
                weight.shape,
                grad_output.dtype,
                grad_output.shape,
                grad_input.dtype,
                grad_input.shape,
                grad_input.is_writable(),
            ),
        });
    }

    let (_, weight_bytes) = checked_ggml_shape_bytes(&weight_shape, GgmlType::Q8_0)?;
    let (_, grad_output_bytes) =
        checked_shape_bytes(&grad_output_shape, std::mem::size_of::<f32>())?;
    let (_, grad_input_bytes) = checked_shape_bytes(&grad_input_shape, std::mem::size_of::<f32>())?;
    if !tensor_physical_range_valid(weight, weight_bytes, 2)
        || !tensor_physical_range_valid(
            grad_output,
            grad_output_bytes,
            std::mem::align_of::<f32>() as u64,
        )
        || !tensor_physical_range_valid(
            grad_input,
            grad_input_bytes,
            std::mem::align_of::<f32>() as u64,
        )
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "weight or cotangent has an unaligned or out-of-buffer byte range".into(),
        });
    }
    if tensor_ranges_overlap(grad_input, weight) || tensor_ranges_overlap(grad_input, grad_output) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "grad_input must not overlap weight or grad_output storage".into(),
        });
    }

    Ok((n_in_u32, n_out_u32))
}

/// Activation VJP for a frozen Q8_0 linear map.
///
/// `weight` has GGUF shape `[n_in, n_out]`. Cotangents and results are
/// contiguous row banks with tensor shapes `[n_out, n_query]` and
/// `[n_in, n_query]`, respectively. The operation computes
/// `grad_input = grad_output * weight` without differentiating the stored
/// quantized weights.
pub fn encode_frozen_linear_q8_0_vjp_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    grad_output: &MetalTensor,
    grad_input: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    let (n_in_u32, n_out_u32) =
        validate_frozen_linear_q8_0_vjp_f32(weight, grad_output, grad_input, n_in, n_out, n_query)?;

    let pso = ctx.pipeline("kernel_frozen_linear_q8_0_vjp_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in_u32,
            n_out: n_out_u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, grad_output);
    enc.set_tensor(3, grad_input);

    const NSG: usize = 8;
    enc.dispatch(
        MTLSize {
            width: (n_in / 32).div_ceil(NSG),
            height: 1,
            depth: n_query,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(crate) fn encode_frozen_linear_q8_0_vjp_r2c16k64_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    grad_output: &MetalTensor,
    grad_input: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    let (n_in_u32, n_out_u32) =
        validate_frozen_linear_q8_0_vjp_f32(weight, grad_output, grad_input, n_in, n_out, n_query)?;
    let fast_queries = n_query / 128 * 128;
    if fast_queries == 0 || !n_out.is_multiple_of(64) {
        return encode_frozen_linear_q8_0_vjp_f32(
            ctx,
            enc,
            weight,
            grad_output,
            grad_input,
            n_in,
            n_out,
            n_query,
        );
    }

    const KERNEL: &str = "frozen_linear_q8_0_vjp_r2c16k64";
    let pso = ctx.pipeline("kernel_frozen_linear_q8_0_vjp_r2c16k64_f32")?;
    if pso.threadExecutionWidth() != 32
        || pso.maxTotalThreadsPerThreadgroup() < 128
        || ctx.device.maxThreadgroupMemoryLength() < 4_096
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "requires four SIMDgroups and 4 KiB threadgroup memory".into(),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in_u32,
            n_out: n_out_u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, grad_output);
    enc.set_tensor(3, grad_input);
    enc.set_threadgroup_memory(0, 4_096);
    enc.dispatch(
        MTLSize {
            width: fast_queries / 128,
            height: n_in / 16,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );

    let tail_queries = n_query - fast_queries;
    if tail_queries != 0 {
        let grad_output_tail = grad_output.view_subrange(
            (fast_queries * n_out) as u64,
            vec![n_out as u64, tail_queries as u64],
        );
        let grad_input_tail = grad_input.view_subrange(
            (fast_queries * n_in) as u64,
            vec![n_in as u64, tail_queries as u64],
        );
        encode_frozen_linear_q8_0_vjp_f32(
            ctx,
            enc,
            weight,
            &grad_output_tail,
            &grad_input_tail,
            n_in,
            n_out,
            tail_queries,
        )?;
    }
    Ok(())
}

/// Activation VJP dispatcher for frozen resident linear weights.
///
/// This currently covers the exact dtypes needed by the Q8 target and BF16
/// engineering controls: Q8_0, BF16, F16, and F32.
pub fn encode_frozen_linear_vjp_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    grad_output: &MetalTensor,
    grad_input: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    match weight.dtype {
        GgmlType::Q8_0 => encode_frozen_linear_q8_0_vjp_f32(
            ctx,
            enc,
            weight,
            grad_output,
            grad_input,
            n_in,
            n_out,
            n_query,
        ),
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => encode_frozen_linear_dense_vjp_f32(
            ctx,
            enc,
            weight,
            grad_output,
            grad_input,
            n_in,
            n_out,
            n_query,
        ),
        dtype => Err(MetalError::BadShape {
            kernel: "frozen_linear_vjp",
            detail: format!("unsupported frozen linear dtype {dtype:?}"),
        }),
    }
}

/// Activation VJP dispatcher for a bank of frozen linear cotangents.
///
/// The scalar dispatcher remains the exact oracle path. This bank-specific
/// entry point selects the qualified native-Q8 matrix transpose when possible
/// and preserves the existing implementations for dense stored weights.
pub fn encode_frozen_linear_vjp_bank_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    grad_output: &MetalTensor,
    grad_input: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    match weight.dtype {
        GgmlType::Q8_0 => encode_frozen_linear_q8_0_vjp_r2c16k64_f32(
            ctx,
            enc,
            weight,
            grad_output,
            grad_input,
            n_in,
            n_out,
            n_query,
        ),
        _ => encode_frozen_linear_vjp_f32(
            ctx,
            enc,
            weight,
            grad_output,
            grad_input,
            n_in,
            n_out,
            n_query,
        ),
    }
}

pub(crate) fn encode_frozen_linear_dense_vjp_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    grad_output: &MetalTensor,
    grad_input: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "frozen_linear_dense_vjp";
    if n_in == 0 || n_out == 0 || n_query == 0 {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected nonzero dimensions, got n_in={n_in} n_out={n_out} n_query={n_query}"
            ),
        });
    }
    let n_in_u32 = u32::try_from(n_in).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: "n_in exceeds u32".into(),
    })?;
    let n_out_u32 = u32::try_from(n_out).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: "n_out exceeds u32".into(),
    })?;
    let n_query_u32 = u32::try_from(n_query).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: "n_query exceeds u32".into(),
    })?;
    let weight_shape = vec![u64::from(n_in_u32), u64::from(n_out_u32)];
    let grad_output_shape = vec![u64::from(n_out_u32), u64::from(n_query_u32)];
    let grad_input_shape = vec![u64::from(n_in_u32), u64::from(n_query_u32)];
    if !matches!(weight.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
        || grad_output.dtype != GgmlType::F32
        || grad_input.dtype != GgmlType::F32
        || !grad_input.is_writable()
        || weight.shape != weight_shape
        || grad_output.shape != grad_output_shape
        || grad_input.shape != grad_input_shape
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected F32/F16/BF16 {:?}, F32 {:?} -> writable F32 {:?}",
                weight_shape, grad_output_shape, grad_input_shape
            ),
        });
    }
    let (_, weight_bytes) = checked_ggml_shape_bytes(&weight_shape, weight.dtype)?;
    let (_, grad_output_bytes) =
        checked_shape_bytes(&grad_output_shape, std::mem::size_of::<f32>())?;
    let (_, grad_input_bytes) = checked_shape_bytes(&grad_input_shape, std::mem::size_of::<f32>())?;
    let weight_alignment = if weight.dtype == GgmlType::F32 { 4 } else { 2 };
    if !tensor_physical_range_valid(weight, weight_bytes, weight_alignment)
        || !tensor_physical_range_valid(grad_output, grad_output_bytes, 4)
        || !tensor_physical_range_valid(grad_input, grad_input_bytes, 4)
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "tensor has an unaligned or out-of-buffer byte range".into(),
        });
    }
    if tensor_ranges_overlap(grad_input, weight) || tensor_ranges_overlap(grad_input, grad_output) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "grad_input must not overlap weight or grad_output storage".into(),
        });
    }

    let kernel_name = match weight.dtype {
        GgmlType::F32 => "kernel_frozen_linear_f32_vjp_f32",
        GgmlType::F16 => "kernel_frozen_linear_f16_vjp_f32",
        GgmlType::BF16 => "kernel_frozen_linear_bf16_vjp_f32",
        _ => unreachable!("dense VJP dtype validated above"),
    };
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in_u32,
            n_out: n_out_u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, grad_output);
    enc.set_tensor(3, grad_input);
    const NSG: usize = 8;
    enc.dispatch(
        MTLSize {
            width: n_in.div_ceil(32).div_ceil(NSG),
            height: 1,
            depth: n_query,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// One-shot frozen Q8_0 linear activation VJP for tests.
pub fn frozen_linear_q8_0_vjp_f32_readback_for_test(
    ctx: &MetalContext,
    weight_bytes: &[u8],
    grad_output: &[f32],
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<Vec<f32>, MetalError> {
    let weight = MetalTensor::from_bytes(
        ctx,
        weight_bytes,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q8_0,
    )?;
    let grad_output = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(grad_output),
        vec![n_out as u64, n_query as u64],
        GgmlType::F32,
    )?;
    let grad_input = MetalTensor::zeros_f32(ctx, vec![n_in as u64, n_query as u64])?;
    one_shot(ctx, |enc| {
        encode_frozen_linear_q8_0_vjp_f32(
            ctx,
            enc,
            &weight,
            &grad_output,
            &grad_input,
            n_in,
            n_out,
            n_query,
        )
    })?;
    Ok(read_back_f32(&grad_input.buffer, n_query * n_in))
}

/// One-shot frozen linear activation VJP for supported dense dtypes.
pub fn frozen_linear_vjp_f32_readback_for_test(
    ctx: &MetalContext,
    weight_bytes: &[u8],
    dtype: GgmlType,
    grad_output: &[f32],
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<Vec<f32>, MetalError> {
    let weight =
        MetalTensor::from_bytes(ctx, weight_bytes, vec![n_in as u64, n_out as u64], dtype)?;
    let grad_output = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(grad_output),
        vec![n_out as u64, n_query as u64],
        GgmlType::F32,
    )?;
    let grad_input = MetalTensor::zeros_f32(ctx, vec![n_in as u64, n_query as u64])?;
    one_shot(ctx, |encoder| {
        encode_frozen_linear_vjp_f32(
            ctx,
            encoder,
            &weight,
            &grad_output,
            &grad_input,
            n_in,
            n_out,
            n_query,
        )
    })?;
    Ok(read_back_f32(&grad_input.buffer, n_query * n_in))
}

pub(crate) fn validate_vjp_storage_disjoint(
    kernel: &'static str,
    inputs: &[&MetalTensor],
    outputs: &[&MetalTensor],
) -> Result<(), MetalError> {
    for (index, output) in outputs.iter().enumerate() {
        if inputs
            .iter()
            .any(|input| tensor_ranges_overlap(output, input))
            || outputs[index + 1..]
                .iter()
                .any(|other| tensor_ranges_overlap(output, other))
        {
            return Err(MetalError::BadShape {
                kernel,
                detail: "gradient and scratch outputs must not overlap inputs or each other".into(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;

    #[test]
    fn swiglu_vjp_matches_jacobian_and_relp_rules() {
        let ctx = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const N_DIM: usize = 79;
        for row_count in [1usize, 2, 8] {
            let len = row_count * N_DIM;
            let gate: Vec<f32> = (0..len)
                .map(|index| ((index * 19 + 5) % 101) as f32 * 0.11 - 5.5)
                .collect();
            let up: Vec<f32> = (0..len)
                .map(|index| ((index * 13 + 2) % 47) as f32 * 0.031 - 0.67)
                .collect();
            let grad_output: Vec<f32> = (0..len)
                .map(|index| ((index * 7 + 3) % 37) as f32 * 0.017 - 0.29)
                .collect();
            let mut expected_j_gate = vec![0.0f32; len];
            let mut expected_j_up = vec![0.0f32; len];
            let mut expected_r_gate = vec![0.0f32; len];
            let mut expected_r_up = vec![0.0f32; len];
            for index in 0..len {
                let sigmoid = 1.0 / (1.0 + (-gate[index]).exp());
                let silu = gate[index] * sigmoid;
                let silu_derivative = sigmoid * (1.0 + gate[index] * (1.0 - sigmoid));
                expected_j_gate[index] = grad_output[index] * up[index] * silu_derivative;
                expected_j_up[index] = grad_output[index] * silu;
                expected_r_gate[index] = 0.5 * grad_output[index] * up[index] * sigmoid;
                expected_r_up[index] = 0.5 * grad_output[index] * silu;
            }

            let (actual_j_gate, actual_j_up) = silu_mul_vjp_f32_readback_for_test(
                &ctx,
                &gate,
                &up,
                &grad_output,
                row_count,
                N_DIM,
                SwiGluVjpRule::Jacobian,
            )
            .unwrap();
            let (actual_r_gate, actual_r_up) = silu_mul_vjp_f32_readback_for_test(
                &ctx,
                &gate,
                &up,
                &grad_output,
                row_count,
                N_DIM,
                SwiGluVjpRule::RelpIdentityHalf,
            )
            .unwrap();
            for (label, actual, expected) in [
                ("j_gate", &actual_j_gate, &expected_j_gate),
                ("j_up", &actual_j_up, &expected_j_up),
                ("r_gate", &actual_r_gate, &expected_r_gate),
                ("r_up", &actual_r_up, &expected_r_up),
            ] {
                let max_abs = actual
                    .iter()
                    .zip(expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                assert!(max_abs < 2e-6, "rows={row_count} {label} error {max_abs}");
            }

            let index = len - 3;
            let epsilon = 1e-4f64;
            let objective = |gate_delta: f64, up_delta: f64| {
                let gate_value = f64::from(gate[index]) + gate_delta;
                let up_value = f64::from(up[index]) + up_delta;
                f64::from(grad_output[index]) * gate_value / (1.0 + (-gate_value).exp()) * up_value
            };
            let gate_fd = (objective(epsilon, 0.0) - objective(-epsilon, 0.0)) / (2.0 * epsilon);
            let up_fd = (objective(0.0, epsilon) - objective(0.0, -epsilon)) / (2.0 * epsilon);
            assert!((gate_fd - f64::from(actual_j_gate[index])).abs() < 1e-5);
            assert!((up_fd - f64::from(actual_j_up[index])).abs() < 1e-5);
        }
    }

    #[test]
    fn periodic_vjps_match_per_row_controls_bits() {
        let ctx = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const PRIMAL_ROWS: usize = 3;
        const ROWS: usize = 9;
        const N_DIM: usize = 67;
        const EPS: f32 = 1.0e-6;
        let primal_len = PRIMAL_ROWS * N_DIM;
        let bank_len = ROWS * N_DIM;
        let x_values = (0..primal_len)
            .map(|index| ((index * 17 + 3) % 43) as f32 * 0.021 - 0.39)
            .collect::<Vec<_>>();
        let gate_values = (0..primal_len)
            .map(|index| ((index * 19 + 5) % 101) as f32 * 0.11 - 5.5)
            .collect::<Vec<_>>();
        let up_values = (0..primal_len)
            .map(|index| ((index * 13 + 2) % 47) as f32 * 0.031 - 0.67)
            .collect::<Vec<_>>();
        let weight_values = (0..N_DIM)
            .map(|index| 0.45 + (index % 11) as f32 * 0.07)
            .collect::<Vec<_>>();
        let grad_values = (0..bank_len)
            .map(|index| ((index * 7 + 1) % 31) as f32 * 0.013 - 0.18)
            .collect::<Vec<_>>();
        let primal_shape = vec![N_DIM as u64, PRIMAL_ROWS as u64];
        let bank_shape = vec![N_DIM as u64, ROWS as u64];
        let x = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x_values),
            primal_shape.clone(),
            GgmlType::F32,
        )
        .unwrap();
        let gate = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&gate_values),
            primal_shape.clone(),
            GgmlType::F32,
        )
        .unwrap();
        let up = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&up_values),
            primal_shape,
            GgmlType::F32,
        )
        .unwrap();
        let weight = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&weight_values),
            vec![N_DIM as u64],
            GgmlType::F32,
        )
        .unwrap();
        let grad_output = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&grad_values),
            bank_shape.clone(),
            GgmlType::F32,
        )
        .unwrap();

        for (rms_rule, swiglu_rule) in [
            (RmsNormVjpRule::Jacobian, SwiGluVjpRule::Jacobian),
            (
                RmsNormVjpRule::RelpDetachedScale,
                SwiGluVjpRule::RelpIdentityHalf,
            ),
        ] {
            let rms_candidate = MetalTensor::zeros_f32(&ctx, bank_shape.clone()).unwrap();
            let rms_control = MetalTensor::zeros_f32(&ctx, bank_shape.clone()).unwrap();
            let gate_candidate = MetalTensor::zeros_f32(&ctx, bank_shape.clone()).unwrap();
            let gate_control = MetalTensor::zeros_f32(&ctx, bank_shape.clone()).unwrap();
            let up_candidate = MetalTensor::zeros_f32(&ctx, bank_shape.clone()).unwrap();
            let up_control = MetalTensor::zeros_f32(&ctx, bank_shape.clone()).unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_rms_norm_mul_vjp_periodic_f32(
                &ctx,
                &encoder,
                &x,
                &weight,
                &grad_output,
                &rms_candidate,
                ROWS,
                PRIMAL_ROWS,
                N_DIM,
                EPS,
                rms_rule,
            )
            .unwrap();
            encode_silu_mul_vjp_periodic_f32(
                &ctx,
                &encoder,
                &gate,
                &up,
                &grad_output,
                &gate_candidate,
                &up_candidate,
                ROWS,
                PRIMAL_ROWS,
                N_DIM,
                swiglu_rule,
            )
            .unwrap();
            for row in 0..ROWS {
                let primal_row = row % PRIMAL_ROWS;
                let one_row_shape = vec![N_DIM as u64, 1];
                let x_row = x.view_subrange((primal_row * N_DIM) as u64, one_row_shape.clone());
                let gate_row =
                    gate.view_subrange((primal_row * N_DIM) as u64, one_row_shape.clone());
                let up_row = up.view_subrange((primal_row * N_DIM) as u64, one_row_shape.clone());
                let grad_row =
                    grad_output.view_subrange((row * N_DIM) as u64, one_row_shape.clone());
                let rms_row =
                    rms_control.view_subrange((row * N_DIM) as u64, one_row_shape.clone());
                let gate_out_row =
                    gate_control.view_subrange((row * N_DIM) as u64, one_row_shape.clone());
                let up_out_row = up_control.view_subrange((row * N_DIM) as u64, one_row_shape);
                encode_rms_norm_mul_vjp_rows_f32(
                    &ctx, &encoder, &x_row, &weight, &grad_row, &rms_row, 1, N_DIM, EPS, rms_rule,
                )
                .unwrap();
                encode_silu_mul_vjp_f32(
                    &ctx,
                    &encoder,
                    &gate_row,
                    &up_row,
                    &grad_row,
                    &gate_out_row,
                    &up_out_row,
                    1,
                    N_DIM,
                    swiglu_rule,
                )
                .unwrap();
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{:?}", command.error());
            for (name, candidate, control) in [
                ("RMSNorm", &rms_candidate, &rms_control),
                ("SwiGLU gate", &gate_candidate, &gate_control),
                ("SwiGLU up", &up_candidate, &up_control),
            ] {
                assert_eq!(
                    read_back_f32(&candidate.buffer, bank_len)
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    read_back_f32(&control.buffer, bank_len)
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "{name} {rms_rule:?}/{swiglu_rule:?}"
                );
            }
        }
    }

    #[test]
    fn frozen_linear_dense_vjp_matches_stored_precision() {
        let ctx = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const N_IN: usize = 73;
        const N_OUT: usize = 37;
        for dtype in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
            let (weight_bytes, weight_f32) = synthetic_dense_linear_bank(dtype, N_IN, N_OUT);
            for n_query in [1usize, 2, 8] {
                let grad_output: Vec<f32> = (0..n_query * N_OUT)
                    .map(|index| ((index * 17 + 3) % 29) as f32 * 0.007 - 0.091)
                    .collect();
                let mut expected = vec![0.0f32; n_query * N_IN];
                for query in 0..n_query {
                    for input in 0..N_IN {
                        expected[query * N_IN + input] = (0..N_OUT)
                            .map(|output| {
                                weight_f32[output * N_IN + input]
                                    * grad_output[query * N_OUT + output]
                            })
                            .sum();
                    }
                }
                let actual = frozen_linear_vjp_f32_readback_for_test(
                    &ctx,
                    &weight_bytes,
                    dtype,
                    &grad_output,
                    N_IN,
                    N_OUT,
                    n_query,
                )
                .unwrap();
                let max_abs = actual
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs < 1e-5,
                    "dtype={dtype:?} n_query={n_query}: max absolute error {max_abs}"
                );
            }
        }
    }

    #[test]
    fn frozen_linear_q8_0_vjp_matches_dequantized_adjoint() {
        let ctx = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const N_IN: usize = 96;
        const N_OUT: usize = 37;
        let (weight_bytes, weight_f32) = synthetic_q8_0_bank(N_IN, N_OUT);

        for n_query in [1usize, 2, 8] {
            let grad_output: Vec<f32> = (0..n_query * N_OUT)
                .map(|index| ((index * 17 + 3) % 29) as f32 * 0.007 - 0.091)
                .collect();
            let mut expected = vec![0.0f32; n_query * N_IN];
            for query in 0..n_query {
                for input in 0..N_IN {
                    let mut sum = 0.0f32;
                    for output in 0..N_OUT {
                        sum +=
                            weight_f32[output * N_IN + input] * grad_output[query * N_OUT + output];
                    }
                    expected[query * N_IN + input] = sum;
                }
            }

            let gpu = frozen_linear_q8_0_vjp_f32_readback_for_test(
                &ctx,
                &weight_bytes,
                &grad_output,
                N_IN,
                N_OUT,
                n_query,
            )
            .expect("Q8_0 activation VJP");
            let max_abs = gpu
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs < 1e-4,
                "n_query={n_query}: max absolute error {max_abs}"
            );

            let primal: Vec<f32> = (0..n_query * N_IN)
                .map(|index| ((index * 11 + 1) % 41) as f32 * 0.003 - 0.057)
                .collect();
            let mut forward_inner_product = 0.0f64;
            for query in 0..n_query {
                for output in 0..N_OUT {
                    let mut value = 0.0f64;
                    for input in 0..N_IN {
                        value += weight_f32[output * N_IN + input] as f64
                            * primal[query * N_IN + input] as f64;
                    }
                    forward_inner_product += value * grad_output[query * N_OUT + output] as f64;
                }
            }
            let reverse_inner_product: f64 = gpu
                .iter()
                .zip(&primal)
                .map(|(gradient, input)| *gradient as f64 * *input as f64)
                .sum();
            assert!(
                (forward_inner_product - reverse_inner_product).abs() < 2e-5,
                "n_query={n_query}: adjoint mismatch forward={forward_inner_product} reverse={reverse_inner_product}"
            );

            let query = n_query - 1;
            for input in [0usize, 47, N_IN - 1] {
                let epsilon = 1e-3f64;
                let objective = |delta: f64| {
                    let mut value = 0.0f64;
                    for output in 0..N_OUT {
                        let mut projected = 0.0f64;
                        for column in 0..N_IN {
                            let primal_value = primal[query * N_IN + column] as f64
                                + if column == input { delta } else { 0.0 };
                            projected += weight_f32[output * N_IN + column] as f64 * primal_value;
                        }
                        value += projected * grad_output[query * N_OUT + output] as f64;
                    }
                    value
                };
                let finite_difference =
                    (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon);
                let reverse = gpu[query * N_IN + input] as f64;
                assert!(
                    (finite_difference - reverse).abs() < 1e-4,
                    "n_query={n_query} input={input}: finite difference {finite_difference} != reverse {reverse}"
                );
            }
        }
    }

    #[test]
    fn frozen_linear_q8_0_vjp_supports_offset_tensor_views() {
        let ctx = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const N_IN: usize = 64;
        const N_OUT: usize = 5;
        const N_QUERY: usize = 2;
        let (weight_bytes, weight_f32) = synthetic_q8_0_bank(N_IN, N_OUT);
        let grad_output: Vec<f32> = (0..N_QUERY * N_OUT)
            .map(|index| index as f32 * 0.03 - 0.11)
            .collect();
        let weight = offset_tensor(
            &ctx,
            10,
            &weight_bytes,
            7,
            vec![N_IN as u64, N_OUT as u64],
            GgmlType::Q8_0,
        );
        let grad_output_tensor = offset_tensor(
            &ctx,
            8,
            bytemuck::cast_slice(&grad_output),
            12,
            vec![N_OUT as u64, N_QUERY as u64],
            GgmlType::F32,
        );
        let grad_input = offset_tensor(
            &ctx,
            12,
            &vec![0u8; N_QUERY * N_IN * std::mem::size_of::<f32>()],
            16,
            vec![N_IN as u64, N_QUERY as u64],
            GgmlType::F32,
        );
        one_shot(&ctx, |encoder| {
            encode_frozen_linear_q8_0_vjp_f32(
                &ctx,
                encoder,
                &weight,
                &grad_output_tensor,
                &grad_input,
                N_IN,
                N_OUT,
                N_QUERY,
            )
        })
        .unwrap();

        let actual = tensor_f32_at_offset(&grad_input);
        for query in 0..N_QUERY {
            for input in 0..N_IN {
                let expected: f32 = (0..N_OUT)
                    .map(|output| {
                        weight_f32[output * N_IN + input] * grad_output[query * N_OUT + output]
                    })
                    .sum();
                assert!((actual[query * N_IN + input] - expected).abs() < 1e-5);
            }
        }
        assert_offset_guards(&weight, 10, 7);
        assert_offset_guards(&grad_output_tensor, 8, 12);
        assert_offset_guards(&grad_input, 12, 16);
    }

    #[test]
    fn frozen_linear_q8_0_vjp_rejects_invalid_contracts() {
        let ctx = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const N: usize = 32;
        const N_QUERY: usize = 2;
        let (weight_bytes, _) = synthetic_q8_0_bank(N, N);
        let weight = MetalTensor::from_bytes(
            &ctx,
            &weight_bytes,
            vec![N as u64, N as u64],
            GgmlType::Q8_0,
        )
        .unwrap();
        let grad_output = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&vec![0.25f32; N * N_QUERY]),
            vec![N as u64, N_QUERY as u64],
            GgmlType::F32,
        )
        .unwrap();
        let grad_input = MetalTensor::zeros_f32(&ctx, vec![N as u64, N_QUERY as u64]).unwrap();

        let error = one_shot(&ctx, |enc| {
            encode_frozen_linear_q8_0_vjp_f32(
                &ctx,
                enc,
                &weight,
                &grad_output,
                &grad_input,
                N - 1,
                N,
                N_QUERY,
            )
        })
        .expect_err("non-block-aligned input must fail");
        assert!(format!("{error}").contains("frozen_linear_q8_0_vjp"));

        let mut read_only_output = grad_input.clone();
        read_only_output.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        one_shot(&ctx, |enc| {
            encode_frozen_linear_q8_0_vjp_f32(
                &ctx,
                enc,
                &weight,
                &grad_output,
                &read_only_output,
                N,
                N,
                N_QUERY,
            )
        })
        .expect_err("read-only output must fail");

        let overlapping_output = grad_output.clone();
        one_shot(&ctx, |enc| {
            encode_frozen_linear_q8_0_vjp_f32(
                &ctx,
                enc,
                &weight,
                &grad_output,
                &overlapping_output,
                N,
                N,
                N_QUERY,
            )
        })
        .expect_err("overlapping output must fail");

        let mut misaligned_output = grad_input.clone();
        misaligned_output.offset = 2;
        one_shot(&ctx, |enc| {
            encode_frozen_linear_q8_0_vjp_f32(
                &ctx,
                enc,
                &weight,
                &grad_output,
                &misaligned_output,
                N,
                N,
                N_QUERY,
            )
        })
        .expect_err("misaligned output must fail");
    }

    #[test]
    fn frozen_linear_q8_0_vjp_r2c16k64_matches_adjoint_and_tails() {
        let ctx = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };

        for (n_in, n_out, n_query) in [(32usize, 64usize, 128usize), (96, 128, 257), (96, 65, 129)]
        {
            let (weight_bytes, weight_f32) = synthetic_q8_0_bank(n_in, n_out);
            let grad_output = (0..n_query * n_out)
                .map(|index| ((index * 17 + 3) % 29) as f32 * 0.007 - 0.091)
                .collect::<Vec<_>>();
            let weight = offset_tensor(
                &ctx,
                10,
                &weight_bytes,
                7,
                vec![n_in as u64, n_out as u64],
                GgmlType::Q8_0,
            );
            let grad_output_tensor = offset_tensor(
                &ctx,
                8,
                bytemuck::cast_slice(&grad_output),
                12,
                vec![n_out as u64, n_query as u64],
                GgmlType::F32,
            );
            let candidate = offset_tensor(
                &ctx,
                12,
                &vec![0u8; n_query * n_in * std::mem::size_of::<f32>()],
                16,
                vec![n_in as u64, n_query as u64],
                GgmlType::F32,
            );
            let control = offset_tensor(
                &ctx,
                20,
                &vec![0u8; n_query * n_in * std::mem::size_of::<f32>()],
                24,
                vec![n_in as u64, n_query as u64],
                GgmlType::F32,
            );

            let command = ctx.queue.commandBuffer().expect("VJP matrix command");
            let encoder = KernelEncoder::begin(&command);
            encode_frozen_linear_vjp_bank_f32(
                &ctx,
                &encoder,
                &weight,
                &grad_output_tensor,
                &candidate,
                n_in,
                n_out,
                n_query,
            )
            .unwrap();
            encode_frozen_linear_q8_0_vjp_f32(
                &ctx,
                &encoder,
                &weight,
                &grad_output_tensor,
                &control,
                n_in,
                n_out,
                n_query,
            )
            .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{:?}", command.error());

            let actual = tensor_f32_at_offset(&candidate);
            let incumbent = tensor_f32_at_offset(&control);
            let mut expected = vec![0.0f64; n_query * n_in];
            for query in 0..n_query {
                for input in 0..n_in {
                    expected[query * n_in + input] = (0..n_out)
                        .map(|output| {
                            f64::from(weight_f32[output * n_in + input])
                                * f64::from(grad_output[query * n_out + output])
                        })
                        .sum();
                }
            }
            let cpu = f32_f64_differential(&actual, &expected);
            let incumbent_f64 = incumbent
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>();
            let differential = f32_f64_differential(&actual, &incumbent_f64);
            eprintln!(
                "[q8-vjp-r2c16] shape=({n_in},{n_out},{n_query}) cpu={cpu:?} incumbent={differential:?}"
            );
            assert!(actual.iter().all(|value| value.is_finite()));
            assert!(cpu.0 <= 2.0e-5, "relative L2 {cpu:?}");
            assert!(cpu.1 <= 5.0e-5, "normalized max {cpu:?}");
            assert!(cpu.2 >= 0.999_999_99, "cosine {cpu:?}");
            assert!(
                differential.0 <= 3.0e-4,
                "incumbent relative L2 {differential:?}"
            );
            assert!(
                differential.1 <= 1.0e-3,
                "incumbent normalized max {differential:?}"
            );
            assert!(
                differential.2 >= 0.999_999_9,
                "incumbent cosine {differential:?}"
            );

            let primal = (0..n_query * n_in)
                .map(|index| ((index * 11 + 1) % 41) as f32 * 0.003 - 0.057)
                .collect::<Vec<_>>();
            let mut forward_inner_product = 0.0f64;
            for query in 0..n_query {
                for output in 0..n_out {
                    let projected = (0..n_in)
                        .map(|input| {
                            f64::from(weight_f32[output * n_in + input])
                                * f64::from(primal[query * n_in + input])
                        })
                        .sum::<f64>();
                    forward_inner_product +=
                        projected * f64::from(grad_output[query * n_out + output]);
                }
            }
            let reverse_inner_product = actual
                .iter()
                .zip(&primal)
                .map(|(&gradient, &input)| f64::from(gradient) * f64::from(input))
                .sum::<f64>();
            let adjoint_error = (forward_inner_product - reverse_inner_product).abs()
                / forward_inner_product
                    .abs()
                    .max(reverse_inner_product.abs())
                    .max(1.0);
            assert!(adjoint_error <= 2.0e-5, "adjoint error {adjoint_error}");

            let query = n_query - 1;
            for input in [0usize, n_in / 2, n_in - 1] {
                let epsilon = 1.0e-3f64;
                let objective = |delta: f64| {
                    (0..n_out)
                        .map(|output| {
                            let projected = (0..n_in)
                                .map(|column| {
                                    let value = f64::from(primal[query * n_in + column])
                                        + if column == input { delta } else { 0.0 };
                                    f64::from(weight_f32[output * n_in + column]) * value
                                })
                                .sum::<f64>();
                            projected * f64::from(grad_output[query * n_out + output])
                        })
                        .sum::<f64>()
                };
                let finite_difference =
                    (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon);
                let reverse = f64::from(actual[query * n_in + input]);
                let tolerance = 1.0e-4 + 2.0e-5 * finite_difference.abs();
                assert!(
                    (finite_difference - reverse).abs() <= tolerance,
                    "shape=({n_in},{n_out},{n_query}) input={input} finite difference {finite_difference} != {reverse}"
                );
            }

            if !n_out.is_multiple_of(64) {
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    incumbent
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            assert_offset_guards(&weight, 10, 7);
            assert_offset_guards(&grad_output_tensor, 8, 12);
            assert_offset_guards(&candidate, 12, 16);
            assert_offset_guards(&control, 20, 24);
        }

        let weight_bytes = synthetic_q8_0_bytes(64, 64);
        let weight =
            MetalTensor::from_bytes(&ctx, &weight_bytes, vec![64, 64], GgmlType::Q8_0).unwrap();
        let grad_output = MetalTensor::zeros_f32(&ctx, vec![64, 128]).unwrap();
        let grad_input = MetalTensor::zeros_f32(&ctx, vec![64, 128]).unwrap();
        let error = one_shot(&ctx, |encoder| {
            encode_frozen_linear_vjp_bank_f32(
                &ctx,
                encoder,
                &weight,
                &grad_output,
                &grad_input,
                48,
                64,
                128,
            )
        })
        .expect_err("non-Q8 input width must fail");
        assert!(format!("{error}").contains("n_in=48"));
    }

    #[test]
    #[ignore = "bounded model-free Muse Q8 VJP qualification packet"]
    fn profile_frozen_linear_q8_0_vjp_r2c16k64_muse_shapes() {
        let ctx = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const FIXED_KILL_MS: f64 = 23.579;
        let cases = [
            ("gate-up-q128", 6_656usize, 19_968usize, 128usize),
            ("gate-up-q512", 6_656, 19_968, 512),
            ("down-q128", 19_968, 6_656, 128),
            ("down-q512", 19_968, 6_656, 512),
        ];

        let weight_bytes = synthetic_q8_0_bytes(6_656, 19_968);
        let base_weight =
            MetalTensor::from_bytes(&ctx, &weight_bytes, vec![6_656, 19_968], GgmlType::Q8_0)
                .unwrap();
        drop(weight_bytes);
        let mut speedups = Vec::with_capacity(cases.len());

        for (case_index, (label, n_in, n_out, n_query)) in cases.into_iter().enumerate() {
            let mut weight = base_weight.clone();
            weight.shape = vec![n_in as u64, n_out as u64];
            let grad_output_values = (0..n_query * n_out)
                .map(|index| ((index * 17 + 3) % 29) as f32 * 0.007 - 0.091)
                .collect::<Vec<_>>();
            let grad_output = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&grad_output_values),
                vec![n_out as u64, n_query as u64],
                GgmlType::F32,
            )
            .unwrap();
            drop(grad_output_values);
            let control = MetalTensor::zeros_f32(&ctx, vec![n_in as u64, n_query as u64]).unwrap();
            let candidate =
                MetalTensor::zeros_f32(&ctx, vec![n_in as u64, n_query as u64]).unwrap();

            let run = |matrix: bool| {
                let command = ctx.queue.commandBuffer().expect("Q8 VJP timing command");
                let encoder = KernelEncoder::begin(&command);
                if matrix {
                    encode_frozen_linear_vjp_bank_f32(
                        &ctx,
                        &encoder,
                        &weight,
                        &grad_output,
                        &candidate,
                        n_in,
                        n_out,
                        n_query,
                    )
                    .unwrap();
                } else {
                    encode_frozen_linear_q8_0_vjp_f32(
                        &ctx,
                        &encoder,
                        &weight,
                        &grad_output,
                        &control,
                        n_in,
                        n_out,
                        n_query,
                    )
                    .unwrap();
                }
                encoder.end();
                command.commit();
                command.waitUntilCompleted();
                assert!(command.error().is_none(), "{:?}", command.error());
                let start = command.GPUStartTime();
                let end = command.GPUEndTime();
                assert!(start.is_finite() && end.is_finite() && start > 0.0 && end > start);
                (end - start) * 1.0e3
            };

            run(false);
            run(true);
            let b1 = run(false);
            let c1 = run(true);
            let c2 = run(true);
            let b2 = run(false);
            let control_mean = (b1 + b2) * 0.5;
            let candidate_mean = (c1 + c2) * 0.5;
            let speedup = control_mean / candidate_mean;
            let numerical = f32_differential(
                &tensor_f32_at_offset(&candidate),
                &tensor_f32_at_offset(&control),
            );
            eprintln!(
                "[q8-vjp-r2c16] {label} B-C-C-B gpu_ms={b1:.6}/{c1:.6}/{c2:.6}/{b2:.6} mean={control_mean:.6}->{candidate_mean:.6} speedup={speedup:.3}x numerical={numerical:?}"
            );
            assert!(
                c1 < b1 && c2 < b2,
                "{label}: both balanced comparisons must improve"
            );
            assert!(speedup >= 4.0, "{label}: speedup={speedup:.3}x");
            assert!(numerical.0 <= 3.0e-4, "{label}: relative L2 {numerical:?}");
            assert!(
                numerical.1 <= 1.0e-3,
                "{label}: normalized max {numerical:?}"
            );
            assert!(numerical.2 >= 0.999_999_9, "{label}: cosine {numerical:?}");
            if case_index == 0 {
                assert!(
                    candidate_mean <= FIXED_KILL_MS,
                    "first stop failed: candidate={candidate_mean:.6} ms"
                );
            }
            speedups.push(speedup);
        }
        let geometric_mean = (speedups.iter().map(|speedup| speedup.ln()).sum::<f64>()
            / speedups.len() as f64)
            .exp();
        eprintln!("[q8-vjp-r2c16] geometric_mean={geometric_mean:.3}x");
        assert!(geometric_mean >= 5.5);
    }
}
