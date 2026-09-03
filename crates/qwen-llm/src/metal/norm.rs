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
