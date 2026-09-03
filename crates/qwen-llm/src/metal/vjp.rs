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
