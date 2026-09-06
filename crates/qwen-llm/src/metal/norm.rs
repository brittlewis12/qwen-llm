//! RMSNorm, L2-norm, and QK-norm kernels.

use super::*;

/// RMSNorm-with-weight: `y[i] = (x[i] / sqrt(mean(x²) + eps)) * weight[i]`.
///
/// Operates on a single row of length `n_dim`. One threadgroup per dispatch;
/// up to 1024 threads per threadgroup, internally reduced via simdgroup_sum.
///
/// CPU oracle: [`crate::forward::rms_norm_pub`].
pub fn encode_rms_norm_mul_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    weight: &MetalTensor,
    y: &MetalTensor,
    eps: f32,
) -> Result<(), MetalError> {
    let n_dim = x.n_elements() as usize;
    if y.n_elements() as usize != n_dim {
        return Err(MetalError::BadShape {
            kernel: "rms_norm",
            detail: format!("y.n_elements={} != x.n_elements={n_dim}", y.n_elements()),
        });
    }
    if weight.n_elements() as usize != n_dim {
        return Err(MetalError::BadShape {
            kernel: "rms_norm",
            detail: format!(
                "weight.n_elements={} != x.n_elements={n_dim}",
                weight.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_rms_norm_mul_f32")?;
    enc.set_pipeline(&pso);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_dim: u32,
        eps: f32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_dim: n_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: 1,
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

/// RMSNorm-with-weight over compact rows. Each row uses the same reduction
/// and arithmetic lineage as [`encode_rms_norm_mul_f32`].
pub fn encode_rms_norm_mul_rows_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    weight: &MetalTensor,
    y: &MetalTensor,
    row_count: usize,
    n_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let want = row_count
        .checked_mul(n_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rms_norm_rows",
            detail: "row element count overflow".to_string(),
        })? as u64;
    if row_count == 0 || n_dim == 0 || x.n_elements() != want || y.n_elements() != want {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_rows",
            detail: format!(
                "x/y expected {row_count} rows of {n_dim} elements, got {}/{}",
                x.n_elements(),
                y.n_elements()
            ),
        });
    }
    if weight.n_elements() as usize != n_dim {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_rows",
            detail: format!("weight expected {n_dim} elements"),
        });
    }
    let row_count = u32::try_from(row_count).map_err(|_| MetalError::BadShape {
        kernel: "rms_norm_rows",
        detail: "row count exceeds u32".to_string(),
    })?;
    let n_dim = u32::try_from(n_dim).map_err(|_| MetalError::BadShape {
        kernel: "rms_norm_rows",
        detail: "row width exceeds u32".to_string(),
    })?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_dim: u32,
        row_count: u32,
        eps: f32,
    }
    let pso = ctx.pipeline("kernel_rms_norm_mul_rows_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_dim,
            row_count,
            eps,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));
    enc.dispatch(
        MTLSize {
            width: row_count as usize,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmsNormVjpRule {
    Jacobian,
    RelpDetachedScale,
}

/// Activation VJP for weighted RMSNorm over compact rows.
///
/// [`RmsNormVjpRule::Jacobian`] computes the ordinary derivative.
/// [`RmsNormVjpRule::RelpDetachedScale`] implements the R-lens LN-rule by
/// treating the reciprocal RMS denominator as constant during propagation.
pub fn encode_rms_norm_mul_vjp_rows_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    weight: &MetalTensor,
    grad_output: &MetalTensor,
    grad_input: &MetalTensor,
    row_count: usize,
    n_dim: usize,
    eps: f32,
    rule: RmsNormVjpRule,
) -> Result<(), MetalError> {
    encode_rms_norm_mul_vjp_impl_f32(
        ctx,
        enc,
        x,
        weight,
        grad_output,
        grad_input,
        row_count,
        n_dim,
        eps,
        rule,
        row_count,
        false,
    )
}

/// Activation VJP for a query bank sharing one weighted RMSNorm primal row.
///
/// `x` has shape `[n_dim]`; cotangents and results have compact-row shape
/// `[n_dim, row_count]`. This avoids materializing one copy of the primal per
/// lens query while preserving the arithmetic of
/// [`encode_rms_norm_mul_vjp_rows_f32`].
pub fn encode_rms_norm_mul_vjp_broadcast_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    weight: &MetalTensor,
    grad_output: &MetalTensor,
    grad_input: &MetalTensor,
    row_count: usize,
    n_dim: usize,
    eps: f32,
    rule: RmsNormVjpRule,
) -> Result<(), MetalError> {
    encode_rms_norm_mul_vjp_impl_f32(
        ctx,
        enc,
        x,
        weight,
        grad_output,
        grad_input,
        row_count,
        n_dim,
        eps,
        rule,
        1,
        true,
    )
}

/// Activation VJP for a query bank periodically reusing weighted RMSNorm rows.
///
/// `x` has compact-row shape `[n_dim, primal_row_count]`; cotangent row `r`
/// uses primal row `r % primal_row_count`.
#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm_mul_vjp_periodic_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    weight: &MetalTensor,
    grad_output: &MetalTensor,
    grad_input: &MetalTensor,
    row_count: usize,
    primal_row_count: usize,
    n_dim: usize,
    eps: f32,
    rule: RmsNormVjpRule,
) -> Result<(), MetalError> {
    encode_rms_norm_mul_vjp_impl_f32(
        ctx,
        enc,
        x,
        weight,
        grad_output,
        grad_input,
        row_count,
        n_dim,
        eps,
        rule,
        primal_row_count,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rms_norm_mul_vjp_impl_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    weight: &MetalTensor,
    grad_output: &MetalTensor,
    grad_input: &MetalTensor,
    row_count: usize,
    n_dim: usize,
    eps: f32,
    rule: RmsNormVjpRule,
    primal_row_count: usize,
    rank_one_primal: bool,
) -> Result<(), MetalError> {
    const KERNEL: &str = "rms_norm_mul_vjp_rows";
    if row_count == 0
        || primal_row_count == 0
        || primal_row_count > row_count
        || !row_count.is_multiple_of(primal_row_count)
        || (rank_one_primal && primal_row_count != 1)
        || n_dim == 0
        || !eps.is_finite()
        || eps < 0.0
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected periodic nonzero primal rows, nonzero width, and finite nonnegative eps; got rows={row_count} primal_rows={primal_row_count} width={n_dim} eps={eps}"
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
    let row_shape = vec![u64::from(n_dim_u32), u64::from(row_count_u32)];
    let primal_shape = if rank_one_primal {
        vec![u64::from(n_dim_u32)]
    } else {
        vec![u64::from(n_dim_u32), u64::from(primal_row_count_u32)]
    };
    let weight_shape = vec![u64::from(n_dim_u32)];
    if x.dtype != GgmlType::F32
        || weight.dtype != GgmlType::F32
        || grad_output.dtype != GgmlType::F32
        || grad_input.dtype != GgmlType::F32
        || !grad_input.is_writable()
        || x.shape != primal_shape
        || grad_output.shape != row_shape
        || grad_input.shape != row_shape
        || weight.shape != weight_shape
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected F32 primal {:?}, weight {:?}, cotangent {:?} -> writable {:?}",
                primal_shape, weight_shape, row_shape, row_shape
            ),
        });
    }
    let (_, primal_bytes) = checked_shape_bytes(&primal_shape, std::mem::size_of::<f32>())?;
    let (_, row_bytes) = checked_shape_bytes(&row_shape, std::mem::size_of::<f32>())?;
    let (_, weight_bytes) = checked_shape_bytes(&weight_shape, std::mem::size_of::<f32>())?;
    if !tensor_physical_range_valid(x, primal_bytes, 4)
        || !tensor_physical_range_valid(weight, weight_bytes, 4)
        || !tensor_physical_range_valid(grad_output, row_bytes, 4)
        || !tensor_physical_range_valid(grad_input, row_bytes, 4)
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "tensor has an unaligned or out-of-buffer byte range".into(),
        });
    }
    if tensor_ranges_overlap(grad_input, x)
        || tensor_ranges_overlap(grad_input, weight)
        || tensor_ranges_overlap(grad_input, grad_output)
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "grad_input must not overlap primal, weight, or grad_output storage".into(),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_dim: u32,
        row_count: u32,
        eps: f32,
        detach_scale: u32,
        primal_row_count: u32,
    }
    let pso = ctx.pipeline("kernel_rms_norm_mul_vjp_rows_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_dim: n_dim_u32,
            row_count: row_count_u32,
            eps,
            detach_scale: u32::from(rule == RmsNormVjpRule::RelpDetachedScale),
            primal_row_count: primal_row_count_u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, grad_output);
    enc.set_tensor(4, grad_input);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (2 * n_simdgroups * std::mem::size_of::<f32>()).max(32));
    enc.dispatch(
        MTLSize {
            width: row_count,
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

/// In-place residual add followed by RMSNorm-with-weight:
/// `x[i] += residual[i]`; `y[i] = (x[i] / sqrt(mean(x²) + eps)) * weight[i]`.
pub fn encode_residual_rms_norm_mul_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    residual: &MetalTensor,
    weight: &MetalTensor,
    y: &MetalTensor,
    eps: f32,
) -> Result<(), MetalError> {
    let n_dim = x.n_elements() as usize;
    if residual.n_elements() as usize != n_dim {
        return Err(MetalError::BadShape {
            kernel: "residual_rms_norm",
            detail: format!(
                "residual.n_elements={} != x.n_elements={n_dim}",
                residual.n_elements()
            ),
        });
    }
    if y.n_elements() as usize != n_dim {
        return Err(MetalError::BadShape {
            kernel: "residual_rms_norm",
            detail: format!("y.n_elements={} != x.n_elements={n_dim}", y.n_elements()),
        });
    }
    if weight.n_elements() as usize != n_dim {
        return Err(MetalError::BadShape {
            kernel: "residual_rms_norm",
            detail: format!(
                "weight.n_elements={} != x.n_elements={n_dim}",
                weight.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_residual_rms_norm_mul_f32")?;
    enc.set_pipeline(&pso);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_dim: u32,
        eps: f32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_dim: n_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, residual);
    enc.set_tensor(3, weight);
    enc.set_tensor(4, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: 1,
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

/// Per-element kernel arg shape for L2 norm.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct L2NormArgs {
    pub(crate) n_dim: u32,
    pub(crate) eps: f32,
}

/// L2 norm: y = x / max(||x||, eps). Per-vector. ggml semantics (NOT
/// `1/sqrt(sum+eps)` — that's RMSNorm).
pub fn encode_l2_norm_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
    eps: f32,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if y.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "l2_norm",
            detail: format!("y.n={} != x.n={n}", y.n_elements()),
        });
    }
    let pso = ctx.pipeline("kernel_l2_norm_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &L2NormArgs {
            n_dim: n as u32,
            eps,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: 1,
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

/// Ordinary VJP for compact rows normalized as `x / max(||x||, eps)`.
/// The nondifferentiable equality case follows the clamped branch.
pub fn encode_l2_norm_vjp_batched_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    grad_output: &MetalTensor,
    grad_input: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    const KERNEL: &str = "l2_norm_vjp_batched";
    if n_heads == 0
        || head_dim == 0
        || !eps.is_finite()
        || eps <= 0.0
        || u32::try_from(n_heads).is_err()
        || u32::try_from(head_dim).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected nonzero u32 geometry and finite positive eps, got heads={n_heads} dim={head_dim} eps={eps}"
            ),
        });
    }
    let elements = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "element count overflow".into(),
        })?;
    let shape = vec![elements as u64];
    validate_compact_f32_tensor(KERNEL, "x", x, &shape, false)?;
    validate_compact_f32_tensor(KERNEL, "grad_output", grad_output, &shape, false)?;
    validate_compact_f32_tensor(KERNEL, "grad_input", grad_input, &shape, true)?;
    if tensor_ranges_overlap(grad_input, x) || tensor_ranges_overlap(grad_input, grad_output) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "grad_input must not overlap primal or incoming gradient".into(),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    }
    let pipeline = ctx.pipeline("kernel_l2_norm_vjp_batched_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, grad_output);
    enc.set_tensor(3, grad_input);
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024);
    let simdgroups = threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (2 * simdgroups * std::mem::size_of::<f32>()).max(32));
    enc.dispatch(
        MTLSize {
            width: n_heads,
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

/// Per-head RMSNorm with shared per-channel weight. Used for Q-norm
/// and K-norm in the gated-attention block. One dispatch covers all
/// heads (n_heads threadgroups, simdgroup-reduce inside).
pub fn encode_rms_norm_batched_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    weight: &MetalTensor,
    y: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let want = (n_heads * head_dim) as u64;
    if x.n_elements() != want || y.n_elements() != want {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_batched",
            detail: format!("x/y expected {want} elements"),
        });
    }
    if weight.n_elements() as usize != head_dim {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_batched",
            detail: format!("weight expected {head_dim} elements"),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    }
    let pso = ctx.pipeline("kernel_rms_norm_batched_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: n_heads,
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

/// Per-head RMSNorm reading strided source rows: row `hi` lives at
/// `x[src_offset + hi * src_stride .. + head_dim]`; output `y` is compact
/// `[n_heads, head_dim]`. Used to read the Q halves of the interleaved
/// gated-attention q_proj output directly (src_stride = 2*head_dim,
/// src_offset = 0), deleting the split_q_gate layout copy (v0.432).
/// Bit-identical per-row math to `encode_rms_norm_batched_f32`.
#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm_batched_src_strided_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    weight: &MetalTensor,
    y: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    src_stride: usize,
    src_offset: usize,
    eps: f32,
) -> Result<(), MetalError> {
    if n_heads == 0 || head_dim == 0 {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_batched_src_strided",
            detail: "n_heads/head_dim must be nonzero".to_string(),
        });
    }
    // Source must cover the last strided row end-to-end.
    let src_need = src_offset as u64 + (n_heads as u64 - 1) * src_stride as u64 + head_dim as u64;
    if x.n_elements() < src_need {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_batched_src_strided",
            detail: format!(
                "x has {} elements, needs >= {src_need} \
                 (offset={src_offset} stride={src_stride} rows={n_heads} head_dim={head_dim})",
                x.n_elements()
            ),
        });
    }
    let want = (n_heads * head_dim) as u64;
    if y.n_elements() != want {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_batched_src_strided",
            detail: format!("y expected {want} elements"),
        });
    }
    if weight.n_elements() as usize != head_dim {
        return Err(MetalError::BadShape {
            kernel: "rms_norm_batched_src_strided",
            detail: format!("weight expected {head_dim} elements"),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        src_stride: u32,
        src_offset: u32,
        eps: f32,
    }
    let pso = ctx.pipeline("kernel_rms_norm_batched_src_strided_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            src_stride: src_stride as u32,
            src_offset: src_offset as u32,
            eps,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: n_heads,
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

/// Fuses packed Q/K per-head RMSNorm with consecutive-position partial RoPE.
/// Q source rows are the interleaved `[Q, gate]` projection layout; K source
/// and both outputs are compact.
pub fn encode_qk_rms_norm_rope_f32_packed_consecutive(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_src: &MetalTensor,
    q_weight: &MetalTensor,
    q_out: &MetalTensor,
    k_src: &MetalTensor,
    k_weight: &MetalTensor,
    k_out: &MetalTensor,
    n_tokens: usize,
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    n_rot: usize,
    start_position: u32,
    eps: f32,
    theta_base: f32,
) -> Result<(), MetalError> {
    if q_src.dtype != GgmlType::F32
        || q_weight.dtype != GgmlType::F32
        || q_out.dtype != GgmlType::F32
        || k_src.dtype != GgmlType::F32
        || k_weight.dtype != GgmlType::F32
        || k_out.dtype != GgmlType::F32
        || !q_out.is_writable()
        || !k_out.is_writable()
    {
        return Err(MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: format!(
                "expected F32 inputs/weights and writable F32 outputs, got dtypes={:?}/{:?}/{:?}/{:?}/{:?}/{:?} writable={}/{}",
                q_src.dtype,
                q_weight.dtype,
                q_out.dtype,
                k_src.dtype,
                k_weight.dtype,
                k_out.dtype,
                q_out.is_writable(),
                k_out.is_writable(),
            ),
        });
    }
    if n_tokens == 0 || n_q_heads == 0 || n_k_heads == 0 || head_dim == 0 {
        return Err(MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: "token, head, and dimension counts must be nonzero".into(),
        });
    }
    if n_rot == 0 || !n_rot.is_multiple_of(2) || n_rot > head_dim {
        return Err(MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: format!("n_rot={n_rot} must be even and <= head_dim={head_dim}"),
        });
    }
    if !eps.is_finite() || eps <= 0.0 || !theta_base.is_finite() || theta_base <= 1.0 {
        return Err(MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail:
                "eps must be finite and positive; theta_base must be finite and greater than one"
                    .into(),
        });
    }
    let q_rows = n_tokens
        .checked_mul(n_q_heads)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: "Q row count overflow".into(),
        })?;
    let k_rows = n_tokens
        .checked_mul(n_k_heads)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: "K row count overflow".into(),
        })?;
    let q_source_elements = q_rows
        .checked_mul(2)
        .and_then(|count| count.checked_mul(head_dim))
        .ok_or_else(|| MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: "Q source shape overflow".into(),
        })?;
    let q_output_elements = q_rows
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: "Q output shape overflow".into(),
        })?;
    let k_elements = k_rows
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: "K shape overflow".into(),
        })?;
    if q_src.n_elements() as usize != q_source_elements
        || q_out.n_elements() as usize != q_output_elements
        || k_src.n_elements() as usize != k_elements
        || k_out.n_elements() as usize != k_elements
        || q_weight.n_elements() as usize != head_dim
        || k_weight.n_elements() as usize != head_dim
    {
        return Err(MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: format!(
                "expected q_src/q_out/k_src/k_out/weights={q_source_elements}/{q_output_elements}/{k_elements}/{k_elements}/{head_dim}, got {}/{}/{}/{}/{}/{}",
                q_src.n_elements(),
                q_out.n_elements(),
                k_src.n_elements(),
                k_out.n_elements(),
                q_weight.n_elements(),
                k_weight.n_elements()
            ),
        });
    }
    let f32_bytes = |elements: usize, detail: &'static str| {
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalError::BadShape {
                kernel: "qk_rms_norm_rope",
                detail: detail.into(),
            })
    };
    let q_source_bytes = f32_bytes(q_source_elements, "Q source byte size overflow")?;
    let q_output_bytes = f32_bytes(q_output_elements, "Q output byte size overflow")?;
    let k_bytes = f32_bytes(k_elements, "K byte size overflow")?;
    let weight_bytes = f32_bytes(head_dim, "weight byte size overflow")?;
    let f32_alignment = std::mem::align_of::<f32>() as u64;
    if !tensor_physical_range_valid(q_src, q_source_bytes, f32_alignment)
        || !tensor_physical_range_valid(q_weight, weight_bytes, f32_alignment)
        || !tensor_physical_range_valid(q_out, q_output_bytes, f32_alignment)
        || !tensor_physical_range_valid(k_src, k_bytes, f32_alignment)
        || !tensor_physical_range_valid(k_weight, weight_bytes, f32_alignment)
        || !tensor_physical_range_valid(k_out, k_bytes, f32_alignment)
    {
        return Err(MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: "tensor byte range is unaligned or outside its Metal buffer".into(),
        });
    }
    u32::try_from(q_source_elements).map_err(|_| MetalError::BadShape {
        kernel: "qk_rms_norm_rope",
        detail: "Q source index range exceeds u32".into(),
    })?;
    u32::try_from(q_output_elements).map_err(|_| MetalError::BadShape {
        kernel: "qk_rms_norm_rope",
        detail: "Q output index range exceeds u32".into(),
    })?;
    u32::try_from(k_elements).map_err(|_| MetalError::BadShape {
        kernel: "qk_rms_norm_rope",
        detail: "K index range exceeds u32".into(),
    })?;
    if tensor_ranges_overlap(q_out, q_src)
        || tensor_ranges_overlap(q_out, q_weight)
        || tensor_ranges_overlap(q_out, k_src)
        || tensor_ranges_overlap(q_out, k_weight)
        || tensor_ranges_overlap(q_out, k_out)
        || tensor_ranges_overlap(k_out, q_src)
        || tensor_ranges_overlap(k_out, q_weight)
        || tensor_ranges_overlap(k_out, k_src)
        || tensor_ranges_overlap(k_out, k_weight)
    {
        return Err(MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: "writable Q/K outputs must not overlap inputs, weights, or each other".into(),
        });
    }
    let n_tokens_u32 = u32::try_from(n_tokens).map_err(|_| MetalError::BadShape {
        kernel: "qk_rms_norm_rope",
        detail: "n_tokens exceeds u32".into(),
    })?;
    let n_q_heads_u32 = u32::try_from(n_q_heads).map_err(|_| MetalError::BadShape {
        kernel: "qk_rms_norm_rope",
        detail: "n_q_heads exceeds u32".into(),
    })?;
    let n_k_heads_u32 = u32::try_from(n_k_heads).map_err(|_| MetalError::BadShape {
        kernel: "qk_rms_norm_rope",
        detail: "n_k_heads exceeds u32".into(),
    })?;
    let head_dim_u32 = u32::try_from(head_dim).map_err(|_| MetalError::BadShape {
        kernel: "qk_rms_norm_rope",
        detail: "head_dim exceeds u32".into(),
    })?;
    let n_rot_u32 = u32::try_from(n_rot).map_err(|_| MetalError::BadShape {
        kernel: "qk_rms_norm_rope",
        detail: "n_rot exceeds u32".into(),
    })?;
    u32::try_from(q_rows).map_err(|_| MetalError::BadShape {
        kernel: "qk_rms_norm_rope",
        detail: "Q row count exceeds u32".into(),
    })?;
    u32::try_from(k_rows).map_err(|_| MetalError::BadShape {
        kernel: "qk_rms_norm_rope",
        detail: "K row count exceeds u32".into(),
    })?;
    let last_token = u32::try_from(n_tokens - 1).map_err(|_| MetalError::BadShape {
        kernel: "qk_rms_norm_rope",
        detail: "n_tokens exceeds u32".into(),
    })?;
    start_position
        .checked_add(last_token)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: "position span exceeds u32".into(),
        })?;
    let total_rows = q_rows
        .checked_add(k_rows)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "qk_rms_norm_rope",
            detail: "dispatch row count overflow".into(),
        })?;
    u32::try_from(total_rows).map_err(|_| MetalError::BadShape {
        kernel: "qk_rms_norm_rope",
        detail: "dispatch row count exceeds u32".into(),
    })?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        n_q_heads: u32,
        n_k_heads: u32,
        head_dim: u32,
        n_rot: u32,
        start_position: u32,
        eps: f32,
        theta_base: f32,
    }
    let pso = ctx.pipeline("kernel_qk_rms_norm_rope_f32_packed_consecutive")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: n_tokens_u32,
            n_q_heads: n_q_heads_u32,
            n_k_heads: n_k_heads_u32,
            head_dim: head_dim_u32,
            n_rot: n_rot_u32,
            start_position,
            eps,
            theta_base,
        },
    );
    enc.set_tensor(1, q_src);
    enc.set_tensor(2, q_weight);
    enc.set_tensor(3, q_out);
    enc.set_tensor(4, k_src);
    enc.set_tensor(5, k_weight);
    enc.set_tensor(6, k_out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));
    enc.dispatch(
        MTLSize {
            width: total_rows,
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

/// Per-head L2-norm: `y[h, :] = x[h, :] / max(||x[h, :]||, eps)` for
/// `h ∈ [0, n_heads)`. One dispatch covers all heads. Used in the GDN
/// front-end where Q and K are l2-normed per K-head before the
/// recurrence; replaces n_heads separate calls.
pub fn encode_l2_norm_batched_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let want = (n_heads * head_dim) as u64;
    if x.n_elements() != want || y.n_elements() != want {
        return Err(MetalError::BadShape {
            kernel: "l2_norm_batched",
            detail: format!("x/y expected {want} elements"),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    }
    let pso = ctx.pipeline("kernel_l2_norm_batched_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: n_heads,
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

crate::env_flag!(default_on l2_pair_hd128_r4_enabled, "QWEN_L2_PAIR_HD128_R4");

#[cfg(test)]
pub(crate) fn l2_pair_hd128_r4_enabled_for_test() -> bool {
    l2_pair_hd128_r4_enabled()
}

pub fn encode_l2_norm_pair_batched_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_x: &MetalTensor,
    q_y: &MetalTensor,
    k_x: &MetalTensor,
    k_y: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let want = (n_heads * head_dim) as u64;
    if q_x.n_elements() != want
        || q_y.n_elements() != want
        || k_x.n_elements() != want
        || k_y.n_elements() != want
    {
        return Err(MetalError::BadShape {
            kernel: "l2_norm_pair_batched",
            detail: format!("q/k inputs and outputs expected {want} elements"),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    }
    let use_hd128_r4 = l2_pair_hd128_r4_enabled();
    if use_hd128_r4 && head_dim == 128 {
        let pso = ctx.pipeline("kernel_l2_norm_pair_hd128_r4_f32")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &Args {
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                eps,
            },
        );
        enc.set_tensor(1, q_x);
        enc.set_tensor(2, q_y);
        enc.set_tensor(3, k_x);
        enc.set_tensor(4, k_y);
        enc.dispatch(
            MTLSize {
                width: n_heads.div_ceil(4),
                height: 2,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        );
        return Ok(());
    }
    let pso = ctx.pipeline("kernel_l2_norm_pair_batched_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, q_x);
    enc.set_tensor(2, q_y);
    enc.set_tensor(3, k_x);
    enc.set_tensor(4, k_y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: n_heads,
            height: 2,
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

crate::env_flag!(default_on rmsnorm_gated_hd128_r4_enabled, "QWEN_RMSNORM_GATED_HD128_R4");

/// RMSNormGated: per-head RMSNorm of `o` with weight, multiplied by
/// silu(z). Used immediately after the GDN recurrence, before out_proj.
///
/// CPU oracle: per-head loop in `forward::Forward::gdn_step` (the
/// "RMSNormGated" block).
pub fn encode_rmsnorm_gated_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    o: &MetalTensor,
    weight: &MetalTensor,
    z: &MetalTensor,
    y: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let want = (n_heads * head_dim) as u64;
    if o.n_elements() != want || z.n_elements() != want || y.n_elements() != want {
        return Err(MetalError::BadShape {
            kernel: "rmsnorm_gated",
            detail: format!(
                "o/z/y expected {want} elements (n_heads={n_heads} * head_dim={head_dim})"
            ),
        });
    }
    if weight.n_elements() as usize != head_dim {
        return Err(MetalError::BadShape {
            kernel: "rmsnorm_gated",
            detail: format!("weight expected {head_dim} elements"),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    }
    let use_hd128_r4 = rmsnorm_gated_hd128_r4_enabled();
    if use_hd128_r4 && head_dim == 128 {
        let pso = ctx.pipeline("kernel_rmsnorm_gated_hd128_r4_f32")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &Args {
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                eps,
            },
        );
        enc.set_tensor(1, o);
        enc.set_tensor(2, weight);
        enc.set_tensor(3, z);
        enc.set_tensor(4, y);
        enc.dispatch(
            MTLSize {
                width: n_heads.div_ceil(4),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        );
        return Ok(());
    }
    let pso = ctx.pipeline("kernel_rmsnorm_gated_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, o);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, z);
    enc.set_tensor(4, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: n_heads,
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

/// Ordinary activation VJP for per-head `RMSNorm(o) * SiLU(z)`.
///
/// The released Qwen R-lens leaves this GDN-internal gated norm on the
/// ordinary Jacobian; RelP rules apply only to residual-stream norms and FFNs.
#[allow(clippy::too_many_arguments)]
pub fn encode_rmsnorm_gated_vjp_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    o: &MetalTensor,
    weight: &MetalTensor,
    z: &MetalTensor,
    grad_y: &MetalTensor,
    grad_o: &MetalTensor,
    grad_z: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    const KERNEL: &str = "rmsnorm_gated_vjp";
    if n_heads == 0
        || head_dim == 0
        || !eps.is_finite()
        || eps < 0.0
        || u32::try_from(n_heads).is_err()
        || u32::try_from(head_dim).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected nonzero u32 geometry and finite nonnegative eps, got heads={n_heads} dim={head_dim} eps={eps}"
            ),
        });
    }
    let elements = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "element count overflow".into(),
        })?;
    let vector_shape = vec![elements as u64];
    let weight_shape = vec![head_dim as u64];
    for (name, tensor) in [("o", o), ("z", z), ("grad_y", grad_y)] {
        validate_compact_f32_tensor(KERNEL, name, tensor, &vector_shape, false)?;
    }
    validate_compact_f32_tensor(KERNEL, "weight", weight, &weight_shape, false)?;
    validate_compact_f32_tensor(KERNEL, "grad_o", grad_o, &vector_shape, true)?;
    validate_compact_f32_tensor(KERNEL, "grad_z", grad_z, &vector_shape, true)?;
    let inputs = [o, weight, z, grad_y];
    if inputs
        .iter()
        .any(|input| tensor_ranges_overlap(grad_o, input))
        || inputs
            .iter()
            .any(|input| tensor_ranges_overlap(grad_z, input))
        || tensor_ranges_overlap(grad_o, grad_z)
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "gradient outputs must not overlap inputs or each other".into(),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    }
    let pipeline = ctx.pipeline("kernel_rmsnorm_gated_vjp_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, o);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, z);
    enc.set_tensor(4, grad_y);
    enc.set_tensor(5, grad_o);
    enc.set_tensor(6, grad_z);
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024);
    let simdgroups = threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (2 * simdgroups * std::mem::size_of::<f32>()).max(32));
    enc.dispatch(
        MTLSize {
            width: n_heads,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;

    #[test]
    fn rms_norm_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &n in &[1024usize, 5120, 17408] {
            let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
            let w: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
            let eps = 1e-6;

            let cpu = crate::forward::rms_norm_pub(&x, &w, eps);
            let gpu =
                rms_norm_mul_f32_readback_for_test(&ctx, &x, &w, eps).expect("metal rms_norm");

            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[rms_norm n={n}] max|Δ|={max_abs:.2e}");
            assert!(max_abs < 1e-4, "rms_norm n={n}: max|Δ|={max_abs}");
        }
    }

    #[test]
    fn rms_norm_vjp_matches_jacobian_and_relp_rules() {
        let ctx = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        const N_DIM: usize = 67;
        const EPS: f32 = 1e-6;
        for row_count in [1usize, 2, 8] {
            let x: Vec<f32> = (0..row_count * N_DIM)
                .map(|index| ((index * 17 + 3) % 43) as f32 * 0.021 - 0.39)
                .collect();
            let weight: Vec<f32> = (0..N_DIM)
                .map(|index| 0.45 + (index % 11) as f32 * 0.07)
                .collect();
            let grad_output: Vec<f32> = (0..row_count * N_DIM)
                .map(|index| ((index * 7 + 1) % 31) as f32 * 0.013 - 0.18)
                .collect();
            let mut expected_j = vec![0.0f32; x.len()];
            let mut expected_r = vec![0.0f32; x.len()];
            for row in 0..row_count {
                let base = row * N_DIM;
                let sumsq: f32 = x[base..base + N_DIM]
                    .iter()
                    .map(|value| value * value)
                    .sum();
                let scale = (sumsq / N_DIM as f32 + EPS).sqrt().recip();
                let dot: f32 = (0..N_DIM)
                    .map(|index| x[base + index] * grad_output[base + index] * weight[index])
                    .sum();
                let correction = dot * scale * scale * scale / N_DIM as f32;
                for index in 0..N_DIM {
                    let direct = grad_output[base + index] * weight[index] * scale;
                    expected_j[base + index] = direct - x[base + index] * correction;
                    expected_r[base + index] = direct;
                }
            }

            let actual_j = rms_norm_mul_vjp_rows_f32_readback_for_test(
                &ctx,
                &x,
                &weight,
                &grad_output,
                row_count,
                N_DIM,
                EPS,
                RmsNormVjpRule::Jacobian,
            )
            .unwrap();
            let actual_r = rms_norm_mul_vjp_rows_f32_readback_for_test(
                &ctx,
                &x,
                &weight,
                &grad_output,
                row_count,
                N_DIM,
                EPS,
                RmsNormVjpRule::RelpDetachedScale,
            )
            .unwrap();
            let max_j = actual_j
                .iter()
                .zip(&expected_j)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            let max_r = actual_r
                .iter()
                .zip(&expected_r)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(max_j < 2e-5, "rows={row_count}: Jacobian error {max_j}");
            assert!(max_r < 2e-5, "rows={row_count}: RelP error {max_r}");
            assert!(
                actual_j
                    .iter()
                    .zip(&actual_r)
                    .any(|(jacobian, relp)| (jacobian - relp).abs() > 1e-3),
                "ordinary and RelP rules unexpectedly coincide"
            );

            let row = row_count - 1;
            for index in [0usize, 31, N_DIM - 1] {
                let epsilon = 1e-4f64;
                let objective = |delta: f64| {
                    let base = row * N_DIM;
                    let sumsq: f64 = (0..N_DIM)
                        .map(|column| {
                            let value = f64::from(x[base + column])
                                + if column == index { delta } else { 0.0 };
                            value * value
                        })
                        .sum();
                    let scale = (sumsq / N_DIM as f64 + f64::from(EPS)).sqrt().recip();
                    (0..N_DIM)
                        .map(|column| {
                            let value = f64::from(x[base + column])
                                + if column == index { delta } else { 0.0 };
                            f64::from(grad_output[base + column])
                                * value
                                * scale
                                * f64::from(weight[column])
                        })
                        .sum::<f64>()
                };
                let finite_difference =
                    (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon);
                let reverse = f64::from(actual_j[row * N_DIM + index]);
                assert!(
                    (finite_difference - reverse).abs() < 1e-4,
                    "rows={row_count} index={index}: finite difference {finite_difference} != {reverse}"
                );
            }
        }
    }

    #[test]
    fn residual_rms_norm_matches_separate_cpu_path() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &n in &[1024usize, 5120, 17408] {
            let x: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
            let r: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.03).collect();
            let w: Vec<f32> = (0..n).map(|i| 0.4 + (i % 11) as f32 * 0.05).collect();
            let eps = 1e-6;

            let x_cpu: Vec<f32> = x.iter().zip(r.iter()).map(|(a, b)| a + b).collect();
            let y_cpu = crate::forward::rms_norm_pub(&x_cpu, &w, eps);
            let (x_gpu, y_gpu) = residual_rms_norm_mul_f32_readback_for_test(&ctx, &x, &r, &w, eps)
                .expect("metal residual_rms_norm");

            let max_x = x_gpu
                .iter()
                .zip(x_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let max_y = y_gpu
                .iter()
                .zip(y_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[residual_rms_norm n={n}] max_x={max_x:.2e} max_y={max_y:.2e}");
            assert!(max_x == 0.0, "residual add n={n}: max|Δ|={max_x}");
            assert!(max_y < 1e-4, "residual_rms_norm n={n}: max|Δ|={max_y}");
        }
    }

    #[test]
    fn l2_norm_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &n in &[128usize, 256, 1024] {
            // include a few that hit the eps clamp (very small magnitudes)
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 1e-2).sin()).collect();
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n as u64],
                GgmlType::F32,
            )
            .unwrap();
            let gpu = one_shot_f32_out(&ctx, n, |enc, y| {
                encode_l2_norm_f32(&ctx, enc, &x_t, y, 1e-6)
            });

            // CPU reference: y = x / max(||x||, eps).
            let sq: f32 = x.iter().map(|v| v * v).sum();
            let scale = 1.0 / sq.sqrt().max(1e-6);
            let cpu: Vec<f32> = x.iter().map(|v| v * scale).collect();
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max_abs < 1e-5, "l2_norm n={n}: max|Δ|={max_abs}");
        }
    }

    #[test]
    fn l2_norm_batched_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(n_heads, head_dim) in &[(16usize, 128usize), (48, 128), (8, 256)] {
            let total = n_heads * head_dim;
            let x: Vec<f32> = (0..total)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
                .collect();
            let eps = 1e-6f32;

            // CPU reference: per head, y_h = x_h / max(||x_h||, eps).
            let mut cpu = vec![0.0f32; total];
            for h in 0..n_heads {
                let off = h * head_dim;
                let sq: f32 = (0..head_dim).map(|i| x[off + i].powi(2)).sum();
                let scale = 1.0 / sq.sqrt().max(eps);
                for i in 0..head_dim {
                    cpu[off + i] = x[off + i] * scale;
                }
            }

            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            let y_t = MetalTensor::zeros_f32(&ctx, vec![total as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_l2_norm_batched_f32(&ctx, enc, &x_t, &y_t, n_heads, head_dim, eps)
            })
            .unwrap();
            let gpu = read_back_f32(&y_t.buffer, total);

            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs < 1e-5,
                "l2_norm_batched n_heads={n_heads} head_dim={head_dim}: max|Δ|={max_abs}"
            );
        }
    }

    #[test]
    fn l2_norm_pair_batched_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(n_heads, head_dim) in &[(16usize, 128usize), (48, 128), (8, 256)] {
            let total = n_heads * head_dim;
            let q: Vec<f32> = (0..total)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.04)
                .collect();
            let k: Vec<f32> = (0..total)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.03)
                .collect();
            let eps = 1e-6f32;

            let normalize = |x: &[f32]| {
                let mut out = vec![0.0f32; total];
                for h in 0..n_heads {
                    let off = h * head_dim;
                    let sq: f32 = (0..head_dim).map(|i| x[off + i].powi(2)).sum();
                    let scale = 1.0 / sq.sqrt().max(eps);
                    for i in 0..head_dim {
                        out[off + i] = x[off + i] * scale;
                    }
                }
                out
            };
            let q_cpu = normalize(&q);
            let k_cpu = normalize(&k);

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&k),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            let q_y = MetalTensor::zeros_f32(&ctx, vec![total as u64]).unwrap();
            let k_y = MetalTensor::zeros_f32(&ctx, vec![total as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_l2_norm_pair_batched_f32(
                    &ctx, enc, &q_t, &q_y, &k_t, &k_y, n_heads, head_dim, eps,
                )
            })
            .unwrap();
            let q_gpu = read_back_f32(&q_y.buffer, total);
            let k_gpu = read_back_f32(&k_y.buffer, total);

            let q_max = q_gpu
                .iter()
                .zip(q_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let k_max = k_gpu
                .iter()
                .zip(k_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                q_max < 1e-5 && k_max < 1e-5,
                "l2_norm_pair n_heads={n_heads} head_dim={head_dim}: q={q_max} k={k_max}"
            );
        }
    }

    #[test]
    fn rmsnorm_gated_matches_cpu() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        for &(n_heads, head_dim) in &[(16usize, 128usize), (48, 128)] {
            let total = n_heads * head_dim;
            let o: Vec<f32> = (0..total)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.05)
                .collect();
            let z: Vec<f32> = (0..total).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
            let weight: Vec<f32> = (0..head_dim).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
            let eps = 1e-6;

            let cpu = rmsnorm_gated_cpu_ref(&o, &weight, &z, n_heads, head_dim, eps);

            let o_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&o),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            let w_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&weight),
                vec![head_dim as u64],
                GgmlType::F32,
            )
            .unwrap();
            let z_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&z),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            let y_t = MetalTensor::zeros_f32(&ctx, vec![total as u64]).unwrap();

            one_shot(&ctx, |enc| {
                encode_rmsnorm_gated_f32(&ctx, enc, &o_t, &w_t, &z_t, &y_t, n_heads, head_dim, eps)
            })
            .unwrap();

            let gpu = read_back_f32(&y_t.buffer, total);
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[rmsnorm_gated n_heads={n_heads} head_dim={head_dim}] max|Δ|={max_abs:.2e}");
            assert!(max_abs < 1e-4, "rmsnorm_gated drift {max_abs}");
        }
    }

    #[test]
    fn l2_norm_vjp_matches_clamp_and_finite_differences() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N_HEADS: usize = 4;
        const HEAD_DIM: usize = 128;
        const EPS: f32 = 0.5;
        let mut x = vec![0.0f32; N_HEADS * HEAD_DIM];
        for (index, value) in x[..HEAD_DIM].iter_mut().enumerate() {
            *value = ((index * 7 + 3) % 23) as f32 * 0.009 - 0.099;
        }
        x[2 * HEAD_DIM] = EPS;
        x[3 * HEAD_DIM] = EPS * 0.5;
        let grad_output: Vec<f32> = (0..x.len())
            .map(|index| ((index * 11 + 1) % 31) as f32 * 0.013 - 0.19)
            .collect();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![x.len() as u64],
            GgmlType::F32,
        )
        .unwrap();
        let grad_output_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&grad_output),
            vec![grad_output.len() as u64],
            GgmlType::F32,
        )
        .unwrap();
        let grad_input_t = MetalTensor::zeros_f32(&ctx, vec![x.len() as u64]).unwrap();
        one_shot(&ctx, |encoder| {
            encode_l2_norm_vjp_batched_f32(
                &ctx,
                encoder,
                &x_t,
                &grad_output_t,
                &grad_input_t,
                N_HEADS,
                HEAD_DIM,
                EPS,
            )
        })
        .unwrap();
        let actual = read_back_f32(&grad_input_t.buffer, x.len());
        let mut expected = vec![0.0f64; x.len()];
        for head in 0..N_HEADS {
            let base = head * HEAD_DIM;
            let radius = x[base..base + HEAD_DIM]
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                .sqrt();
            if radius > f64::from(EPS) {
                let dot = (0..HEAD_DIM)
                    .map(|index| f64::from(x[base + index]) * f64::from(grad_output[base + index]))
                    .sum::<f64>();
                for index in 0..HEAD_DIM {
                    expected[base + index] = f64::from(grad_output[base + index]) / radius
                        - f64::from(x[base + index]) * dot / radius.powi(3);
                }
            } else {
                for index in 0..HEAD_DIM {
                    expected[base + index] = f64::from(grad_output[base + index]) / f64::from(EPS);
                }
            }
        }
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(max_abs < 2e-5, "L2 VJP error {max_abs}");
        for head in [1usize, 2, 3] {
            let base = head * HEAD_DIM;
            for index in 0..HEAD_DIM {
                assert_eq!(
                    actual[base + index].to_bits(),
                    (grad_output[base + index] / EPS).to_bits(),
                    "clamped row {head} index {index}"
                );
            }
        }

        let epsilon = 1e-5f64;
        for index in [0usize, 31, HEAD_DIM - 1] {
            let objective = |delta: f64| {
                let mut row = x[..HEAD_DIM]
                    .iter()
                    .copied()
                    .map(f64::from)
                    .collect::<Vec<_>>();
                row[index] += delta;
                let radius = row.iter().map(|value| value * value).sum::<f64>().sqrt();
                row.iter()
                    .zip(&grad_output[..HEAD_DIM])
                    .map(|(value, grad)| value / radius * f64::from(*grad))
                    .sum::<f64>()
            };
            let finite_difference = (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon);
            assert!((finite_difference - f64::from(actual[index])).abs() < 2e-5);
        }
    }

    #[test]
    fn rmsnorm_gated_vjp_matches_finite_differences() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N_HEADS: usize = 3;
        const HEAD_DIM: usize = 128;
        const EPS: f32 = HEAD_DIM as f32 * 1e-6;
        let elements = N_HEADS * HEAD_DIM;
        let o: Vec<f32> = (0..elements)
            .map(|index| ((index * 7 + 1) % 37) as f32 * 0.011 - 0.19)
            .collect();
        let weight: Vec<f32> = (0..HEAD_DIM)
            .map(|index| 0.55 + (index % 13) as f32 * 0.037)
            .collect();
        let z: Vec<f32> = (0..elements)
            .map(|index| ((index * 11 + 5) % 43) as f32 * 0.09 - 1.8)
            .collect();
        let grad_y: Vec<f32> = (0..elements)
            .map(|index| ((index * 13 + 3) % 47) as f32 * 0.007 - 0.15)
            .collect();
        let tensor = |values: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(values),
                vec![values.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let o_t = tensor(&o);
        let weight_t = tensor(&weight);
        let z_t = tensor(&z);
        let grad_y_t = tensor(&grad_y);
        let grad_o_t = MetalTensor::zeros_f32(&ctx, vec![elements as u64]).unwrap();
        let grad_z_t = MetalTensor::zeros_f32(&ctx, vec![elements as u64]).unwrap();
        one_shot(&ctx, |encoder| {
            encode_rmsnorm_gated_vjp_f32(
                &ctx, encoder, &o_t, &weight_t, &z_t, &grad_y_t, &grad_o_t, &grad_z_t, N_HEADS,
                HEAD_DIM, EPS,
            )
        })
        .unwrap();
        let actual_o = read_back_f32(&grad_o_t.buffer, elements);
        let actual_z = read_back_f32(&grad_z_t.buffer, elements);
        let mut expected_o = vec![0.0f64; elements];
        let mut expected_z = vec![0.0f64; elements];
        for head in 0..N_HEADS {
            let base = head * HEAD_DIM;
            let sumsq = o[base..base + HEAD_DIM]
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>();
            let scale = (sumsq / HEAD_DIM as f64 + f64::from(EPS)).sqrt().recip();
            let mut dot = 0.0f64;
            for index in 0..HEAD_DIM {
                let offset = base + index;
                let z_value = f64::from(z[offset]);
                let sigmoid = 1.0 / (1.0 + (-z_value).exp());
                let silu = z_value * sigmoid;
                let grad_normed = f64::from(grad_y[offset]) * silu;
                dot += f64::from(o[offset]) * grad_normed * f64::from(weight[index]);
                let normed = f64::from(o[offset]) * scale * f64::from(weight[index]);
                let silu_derivative = sigmoid * (1.0 + z_value * (1.0 - sigmoid));
                expected_z[offset] = f64::from(grad_y[offset]) * normed * silu_derivative;
            }
            let correction = dot * scale.powi(3) / HEAD_DIM as f64;
            for index in 0..HEAD_DIM {
                let offset = base + index;
                let z_value = f64::from(z[offset]);
                let silu = z_value / (1.0 + (-z_value).exp());
                let weighted_grad = f64::from(grad_y[offset]) * silu * f64::from(weight[index]);
                expected_o[offset] = weighted_grad * scale - f64::from(o[offset]) * correction;
            }
        }
        let max_o = actual_o
            .iter()
            .zip(&expected_o)
            .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
            .fold(0.0f64, f64::max);
        let max_z = actual_z
            .iter()
            .zip(&expected_z)
            .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(max_o < 3e-5, "gated RMS grad_o error {max_o}");
        assert!(max_z < 2e-5, "gated RMS grad_z error {max_z}");

        let objective = |o: &[f64], z: &[f64]| {
            let mut value = 0.0f64;
            for head in 0..N_HEADS {
                let base = head * HEAD_DIM;
                let sumsq = o[base..base + HEAD_DIM]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>();
                let scale = (sumsq / HEAD_DIM as f64 + f64::from(EPS)).sqrt().recip();
                for index in 0..HEAD_DIM {
                    let offset = base + index;
                    let silu = z[offset] / (1.0 + (-z[offset]).exp());
                    value += f64::from(grad_y[offset])
                        * o[offset]
                        * scale
                        * f64::from(weight[index])
                        * silu;
                }
            }
            value
        };
        let o64 = o.iter().copied().map(f64::from).collect::<Vec<_>>();
        let z64 = z.iter().copied().map(f64::from).collect::<Vec<_>>();
        let epsilon = 1e-5;
        for &index in &[0usize, 127, 128, elements - 1] {
            let mut plus = o64.clone();
            let mut minus = o64.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let finite_difference =
                (objective(&plus, &z64) - objective(&minus, &z64)) / (2.0 * epsilon);
            assert!((finite_difference - f64::from(actual_o[index])).abs() < 3e-5);

            let mut plus = z64.clone();
            let mut minus = z64.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let finite_difference =
                (objective(&o64, &plus) - objective(&o64, &minus)) / (2.0 * epsilon);
            assert!((finite_difference - f64::from(actual_z[index])).abs() < 2e-5);
        }
    }

    /// v0.432 equivalence gate: the strided-source batched q-norm reading
    /// the Q halves of an interleaved `[head_dim Q, head_dim gate]` layout
    /// must be BIT-IDENTICAL to split_q_gate followed by the compact
    /// batched q-norm (pure addressing change, same per-row arithmetic).
    #[test]
    fn rms_norm_batched_src_strided_matches_split_path_bitwise() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(n_heads, head_dim) in &[(24usize, 256usize), (16, 256), (8, 64)] {
            let full: Vec<f32> = (0..n_heads * 2 * head_dim)
                .map(|i| ((i % 41) as f32 - 20.0) * 3e-2)
                .collect();
            let weight: Vec<f32> = (0..head_dim).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
            let full_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&full),
                vec![(n_heads * 2 * head_dim) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let w_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&weight),
                vec![head_dim as u64],
                GgmlType::F32,
            )
            .unwrap();
            let q_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
            let gate_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
            let y_split = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
            let y_strided =
                MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
            let eps = 1e-6f32;
            one_shot(&ctx, |enc| {
                encode_split_q_gate_f32(&ctx, enc, &full_t, &q_t, &gate_t, n_heads, head_dim)?;
                encode_rms_norm_batched_f32(&ctx, enc, &q_t, &w_t, &y_split, n_heads, head_dim, eps)
            })
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_rms_norm_batched_src_strided_f32(
                    &ctx,
                    enc,
                    &full_t,
                    &w_t,
                    &y_strided,
                    n_heads,
                    head_dim,
                    2 * head_dim,
                    0,
                    eps,
                )
            })
            .unwrap();
            let a = read_back_f32(&y_split.buffer, n_heads * head_dim);
            let b = read_back_f32(&y_strided.buffer, n_heads * head_dim);
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "strided q-norm not bit-identical at [{i}] (n_heads={n_heads}, \
                     head_dim={head_dim}): split={x} strided={y}"
                );
            }
        }
    }

    #[test]
    fn qk_rms_norm_rope_fused_matches_composed_path() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let n_q = 24;
        let n_k = 4;
        let head_dim = 256;
        let n_rot = 64;
        let eps = 1e-6f32;
        let theta = 10_000_000.0f32;
        for &(n_tokens, start_position) in
            &[(1usize, 0u32), (8, 65_531), (128, 65_536), (8, 1_048_568)]
        {
            let q_source: Vec<f32> = (0..n_tokens * n_q * 2 * head_dim)
                .map(|i| ((i % 41) as f32 - 20.0) * 0.03125)
                .collect();
            let k_source: Vec<f32> = (0..n_tokens * n_k * head_dim)
                .map(|i| ((i % 37) as f32 - 18.0) * 0.046875)
                .collect();
            let q_weight: Vec<f32> = (0..head_dim)
                .map(|i| 0.5 + (i % 11) as f32 * 0.0625)
                .collect();
            let k_weight: Vec<f32> = (0..head_dim)
                .map(|i| 0.625 + (i % 7) as f32 * 0.078125)
                .collect();
            let tensor = |values: &[f32]| {
                MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(values),
                    vec![values.len() as u64],
                    GgmlType::F32,
                )
                .unwrap()
            };
            let q_src = tensor(&q_source);
            let k_src = tensor(&k_source);
            let q_w = tensor(&q_weight);
            let k_w = tensor(&k_weight);
            let q_len = n_tokens * n_q * head_dim;
            let k_len = n_tokens * n_k * head_dim;
            let q_composed = MetalTensor::zeros_f32(&ctx, vec![q_len as u64]).unwrap();
            let k_composed = MetalTensor::zeros_f32(&ctx, vec![k_len as u64]).unwrap();
            let q_fused = MetalTensor::zeros_f32(&ctx, vec![q_len as u64]).unwrap();
            let k_fused = MetalTensor::zeros_f32(&ctx, vec![k_len as u64]).unwrap();

            one_shot(&ctx, |enc| {
                encode_rms_norm_batched_src_strided_f32(
                    &ctx,
                    enc,
                    &q_src,
                    &q_w,
                    &q_composed,
                    n_tokens * n_q,
                    head_dim,
                    2 * head_dim,
                    0,
                    eps,
                )?;
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &k_src,
                    &k_w,
                    &k_composed,
                    n_tokens * n_k,
                    head_dim,
                    eps,
                )?;
                encode_rope_neox_f32_packed_consecutive(
                    &ctx,
                    enc,
                    &q_composed,
                    n_tokens,
                    n_q,
                    head_dim,
                    n_rot,
                    start_position,
                    theta,
                )?;
                encode_rope_neox_f32_packed_consecutive(
                    &ctx,
                    enc,
                    &k_composed,
                    n_tokens,
                    n_k,
                    head_dim,
                    n_rot,
                    start_position,
                    theta,
                )
            })
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_qk_rms_norm_rope_f32_packed_consecutive(
                    &ctx,
                    enc,
                    &q_src,
                    &q_w,
                    &q_fused,
                    &k_src,
                    &k_w,
                    &k_fused,
                    n_tokens,
                    n_q,
                    n_k,
                    head_dim,
                    n_rot,
                    start_position,
                    eps,
                    theta,
                )
            })
            .unwrap();

            let q_baseline = read_back_f32(&q_composed.buffer, q_len);
            let k_baseline = read_back_f32(&k_composed.buffer, k_len);
            let q_candidate = read_back_f32(&q_fused.buffer, q_len);
            let k_candidate = read_back_f32(&k_fused.buffer, k_len);
            let q_max = max_abs_diff(&q_baseline, &q_candidate);
            let k_max = max_abs_diff(&k_baseline, &k_candidate);
            eprintln!(
                "[qk-norm-rope] N={n_tokens} start={start_position} q_max={q_max:.3e} k_max={k_max:.3e}"
            );
            assert_finite(
                &q_candidate,
                &format!("fused norm+RoPE Q for N={n_tokens} start={start_position}"),
            );
            assert_finite(
                &k_candidate,
                &format!("fused norm+RoPE K for N={n_tokens} start={start_position}"),
            );
            assert_bitwise_equal(
                &q_baseline,
                &q_candidate,
                &format!("fused norm+RoPE Q for N={n_tokens} start={start_position}"),
            );
            assert_bitwise_equal(
                &k_baseline,
                &k_candidate,
                &format!("fused norm+RoPE K for N={n_tokens} start={start_position}"),
            );
        }
    }
}
