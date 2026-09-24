//! Elementwise, copy/scatter, argmax, top-k, and small helper kernels.

use super::*;

/// Per-element kernel arg used by silu/sigmoid/softplus/add/mul/silu_mul.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NArgs {
    pub(crate) n: u32,
}

pub fn encode_scatter_rows_f32_unique(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    rows: &MetalTensor,
    out: &MetalTensor,
    n_cols: usize,
    n_rows: usize,
) -> Result<(), MetalError> {
    let n = n_cols * n_rows;
    if x.n_elements() as usize != n
        || rows.n_elements() as usize != n_rows
        || !(out.n_elements() as usize).is_multiple_of(n_cols)
    {
        return Err(MetalError::BadShape {
            kernel: "scatter_rows_unique",
            detail: format!(
                "expected x n={} rows n_rows={} out multiple of n_cols={}, got x={} rows={} out={}",
                n,
                n_rows,
                n_cols,
                x.n_elements(),
                rows.n_elements(),
                out.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_scatter_rows_f32_unique")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_cols: u32,
        n_rows: u32,
        out_rows: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_cols: n_cols as u32,
            n_rows: n_rows as u32,
            out_rows: (out.n_elements() as usize / n_cols) as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, rows);
    enc.set_tensor(3, out);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
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

pub fn encode_axpy_scalar_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    scale: &MetalTensor,
    accum: &MetalTensor,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if scale.n_elements() != 1 || accum.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "axpy_scalar",
            detail: format!(
                "expected scale[1] and accum n={n}, got scale={} accum={}",
                scale.n_elements(),
                accum.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_axpy_scalar_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
    }
    enc.set_bytes(0, &Args { n: n as u32 });
    enc.set_tensor(1, x);
    enc.set_tensor(2, scale);
    enc.set_tensor(3, accum);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
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

pub fn encode_dot_sigmoid_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    out: &MetalTensor,
    n: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::F32 || x.dtype != GgmlType::F32 || out.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "dot_sigmoid",
            detail: format!(
                "expected F32/F32/F32, got {:?}/{:?}/{:?}",
                weight.dtype, x.dtype, out.dtype
            ),
        });
    }
    if weight.n_elements() as usize != n || x.n_elements() as usize != n || out.n_elements() != 1 {
        return Err(MetalError::BadShape {
            kernel: "dot_sigmoid",
            detail: format!(
                "shape mismatch: weight={} x={} out={} expected n={n}, out=1",
                weight.n_elements(),
                x.n_elements(),
                out.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_dot_sigmoid_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
    }
    enc.set_bytes(0, &Args { n: n as u32 });
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, out);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    shared_weight: &MetalTensor,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    shared_out: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    n_expert: usize,
    topk: usize,
    hidden: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if logits.dtype != GgmlType::F32
        || shared_weight.dtype != GgmlType::F32
        || x.dtype != GgmlType::F32
        || out_w.dtype != GgmlType::F32
        || shared_out.dtype != GgmlType::F32
        || counts.dtype != GgmlType::F32
        || ids.dtype != GgmlType::F32
    {
        return Err(MetalError::BadShape {
            kernel: "topk_bucket_logits_softmax_dot_sigmoid_packed",
            detail: "expected F32 logits, shared inputs, weights, counts, and bucket IDs".into(),
        });
    }
    if n_expert == 0 || n_expert > 256 || topk == 0 || topk > 16 || topk > n_expert {
        return Err(MetalError::BadShape {
            kernel: "topk_bucket_logits_softmax_dot_sigmoid_packed",
            detail: format!(
                "expected 1 <= topk <= n_expert <= 256 and topk <= 16, got n_expert={n_expert} topk={topk}"
            ),
        });
    }
    if logits.n_elements() as usize != n_tokens * n_expert
        || out_idx.n_elements() as usize != n_tokens * topk
        || out_w.n_elements() as usize != n_tokens * topk
        || shared_weight.n_elements() as usize != hidden
        || x.n_elements() as usize != n_tokens * hidden
        || shared_out.n_elements() as usize != n_tokens
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
    {
        return Err(MetalError::BadShape {
            kernel: "topk_bucket_logits_softmax_dot_sigmoid_packed",
            detail: "shape mismatch in packed route+bucket inputs/outputs".into(),
        });
    }

    let pso = ctx.pipeline("kernel_topk_bucket_logits_softmax_dot_sigmoid_packed_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_expert: u32,
        topk: u32,
        hidden: u32,
        n_tokens: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_expert: n_expert as u32,
            topk: topk as u32,
            hidden: hidden as u32,
            n_tokens: n_tokens as u32,
        },
    );
    enc.set_tensor(1, logits);
    enc.set_tensor(2, shared_weight);
    enc.set_tensor(3, x);
    enc.set_tensor(4, out_idx);
    enc.set_tensor(5, out_w);
    enc.set_tensor(6, shared_out);
    enc.set_tensor(7, counts);
    enc.set_tensor(8, ids);
    const THREADS: usize = 256;
    enc.set_threadgroup_memory(0, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, THREADS * std::mem::size_of::<i32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: n_tokens,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Generic 1-input → 1-output elementwise dispatcher (silu, sigmoid,
/// softplus). All share the (NArgs, x, y) bind pattern.
pub(crate) fn encode_elementwise_1in_1out(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kernel: &str,
    x: &MetalTensor,
    y: &MetalTensor,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if y.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "elementwise_1in_1out",
            detail: format!("y.n={} != x.n={n}", y.n_elements()),
        });
    }
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
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

pub fn encode_silu_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_1in_1out(ctx, enc, "kernel_silu_f32", x, y)
}

pub fn encode_sigmoid_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_1in_1out(ctx, enc, "kernel_sigmoid_f32", x, y)
}

pub fn encode_softplus_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_1in_1out(ctx, enc, "kernel_softplus_f32", x, y)
}

/// Generic 2-input → 1-output elementwise (add, mul, silu_mul). All share
/// the (NArgs, a, b, out) bind pattern.
pub(crate) fn encode_elementwise_2in_1out(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kernel: &str,
    a: &MetalTensor,
    b: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    let n = a.n_elements() as usize;
    if b.n_elements() as usize != n || out.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "elementwise_2in_1out",
            detail: format!("lengths a={} b={} out={n}", a.n_elements(), b.n_elements()),
        });
    }
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, a);
    enc.set_tensor(2, b);
    enc.set_tensor(3, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
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

pub fn encode_add_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,
    b: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_2in_1out(ctx, enc, "kernel_add_f32", a, b, out)
}

pub fn encode_mul_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,
    b: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_2in_1out(ctx, enc, "kernel_mul_f32", a, b, out)
}

/// SwiGLU FFN inner: out = silu(gate) * up. Fuses two ops + saves a
/// scratch buffer on the FFN path. Used as
/// `down(silu_mul(gate(x), up(x)))`.
pub fn encode_silu_mul_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_2in_1out(ctx, enc, "kernel_silu_mul_f32", gate, up, out)
}

/// Gated attention: out = x * sigmoid(gate). Used after attention before the
/// output projection, and supports `out` aliasing `x`.
pub fn encode_sigmoid_mul_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    x: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    encode_elementwise_2in_1out(ctx, enc, "kernel_sigmoid_mul_f32", gate, x, out)
}

/// In-place residual add: x += y. Used after each transformer block's
/// mixer and FFN to add the residual stream back. Saves a scratch buffer
/// vs out-of-place add.
pub fn encode_add_inplace_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if y.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "add_inplace",
            detail: format!("y.n={} != x.n={n}", y.n_elements()),
        });
    }
    let pso = ctx.pipeline("kernel_add_inplace_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
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

pub fn encode_fill_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    y: &MetalTensor,
    value: f32,
) -> Result<(), MetalError> {
    let n = y.n_elements() as usize;
    let pso = ctx.pipeline("kernel_fill_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        value: f32,
    }
    enc.set_bytes(0, &Args { n: n as u32, value });
    enc.set_tensor(1, y);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
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

pub fn encode_axpy_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    accum: &MetalTensor,
    alpha: f32,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if accum.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "axpy",
            detail: format!(
                "accum.n_elements={} != x.n_elements={n}",
                accum.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_axpy_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        alpha: f32,
    }
    enc.set_bytes(0, &Args { n: n as u32, alpha });
    enc.set_tensor(1, x);
    enc.set_tensor(2, accum);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
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

/// One post-block F32 residual intervention. The fixed variant uses the
/// existing AXPY kernel; the remaining variants use one reduction/update
/// dispatch and keep the residual entirely GPU-resident.
#[derive(Clone, Copy)]
pub enum PostBlockIntervention<'a> {
    Fixed {
        layer: u32,
        direction: &'a MetalTensor,
        coefficient: f32,
    },
    ResidualL2Relative {
        layer: u32,
        direction: &'a MetalTensor,
        coefficient: f32,
    },
    Projection {
        layer: u32,
        direction: &'a MetalTensor,
        coefficient: f32,
    },
    SourceToTarget {
        layer: u32,
        source: &'a MetalTensor,
        target: &'a MetalTensor,
        coefficient: f32,
    },
}

/// Encode one post-block residual intervention without a CPU reduction or
/// readback. The caller validates session aliasing and operation bounds;
/// this low-level seam checks the F32 vector geometry needed by the kernel.
pub fn encode_post_block_intervention_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    intervention: &PostBlockIntervention<'_>,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if x.dtype != GgmlType::F32 || !x.is_writable() || n == 0 {
        return Err(MetalError::BadShape {
            kernel: "post_block_intervention",
            detail: format!(
                "x must be nonempty writable F32, got {:?}/{}",
                x.dtype,
                x.n_elements()
            ),
        });
    }
    let vector_bytes =
        n.checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalError::BadShape {
                kernel: "post_block_intervention",
                detail: "vector byte count overflow".into(),
            })?;
    let check_vector = |name: &str, tensor: &MetalTensor| -> Result<(), MetalError> {
        let end = tensor
            .offset
            .checked_add(vector_bytes as u64)
            .ok_or_else(|| MetalError::BadShape {
                kernel: "post_block_intervention",
                detail: format!("{name} endpoint overflow"),
            })?;
        if tensor.dtype != GgmlType::F32
            || tensor.n_elements() as usize != n
            || !tensor
                .offset
                .is_multiple_of(std::mem::align_of::<f32>() as u64)
            || end > tensor.buffer.length() as u64
        {
            return Err(MetalError::BadShape {
                kernel: "post_block_intervention",
                detail: format!("{name} must be aligned F32 with {n} elements"),
            });
        }
        Ok(())
    };

    let (kind, direction, source, target, coefficient) = match intervention {
        PostBlockIntervention::Fixed {
            direction,
            coefficient,
            ..
        } => {
            if !coefficient.is_finite() || *coefficient == 0.0 {
                return Err(MetalError::BadShape {
                    kernel: "post_block_intervention",
                    detail: format!("coefficient must be finite and nonzero, got {coefficient}"),
                });
            }
            check_vector("direction", direction)?;
            return encode_axpy_f32(ctx, enc, direction, x, *coefficient);
        }
        PostBlockIntervention::ResidualL2Relative {
            direction,
            coefficient,
            ..
        } => (0u32, *direction, *direction, *direction, *coefficient),
        PostBlockIntervention::Projection {
            direction,
            coefficient,
            ..
        } => (1u32, *direction, *direction, *direction, *coefficient),
        PostBlockIntervention::SourceToTarget {
            source,
            target,
            coefficient,
            ..
        } => (2u32, *source, *source, *target, *coefficient),
    };
    if !coefficient.is_finite() || coefficient == 0.0 {
        return Err(MetalError::BadShape {
            kernel: "post_block_intervention",
            detail: format!("coefficient must be finite and nonzero, got {coefficient}"),
        });
    }
    check_vector("direction/source", direction)?;
    check_vector("source", source)?;
    check_vector("target", target)?;
    let n_u32 = u32::try_from(n).map_err(|_| MetalError::BadShape {
        kernel: "post_block_intervention",
        detail: format!("n={n} does not fit kernel arguments"),
    })?;

    let pso = ctx.pipeline("kernel_post_block_intervention_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        kind: u32,
        coefficient: f32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n_u32,
            kind,
            coefficient,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, direction);
    enc.set_tensor(3, source);
    enc.set_tensor(4, target);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.set_threadgroup_memory(0, tg_threads * std::mem::size_of::<f32>());
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

pub fn encode_axpy_rowwise_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    scales: &MetalTensor,
    accum: &MetalTensor,
    n_cols: usize,
    n_rows: usize,
) -> Result<(), MetalError> {
    let n = n_cols * n_rows;
    if x.n_elements() as usize != n
        || accum.n_elements() as usize != n
        || scales.n_elements() as usize != n_rows
    {
        return Err(MetalError::BadShape {
            kernel: "axpy_rowwise",
            detail: format!(
                "expected x/accum n={} and scales rows={}, got x={} accum={} scales={}",
                n,
                n_rows,
                x.n_elements(),
                accum.n_elements(),
                scales.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_axpy_rowwise_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_cols: u32,
        n_rows: u32,
        out_rows: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_cols: n_cols as u32,
            n_rows: n_rows as u32,
            out_rows: (accum.n_elements() as usize / n_cols) as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, scales);
    enc.set_tensor(3, accum);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
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

pub fn encode_scatter_axpy_rows_unique_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    rows: &MetalTensor,
    scales: &MetalTensor,
    accum: &MetalTensor,
    n_cols: usize,
    n_rows: usize,
) -> Result<(), MetalError> {
    let n = n_cols * n_rows;
    if x.n_elements() as usize != n
        || rows.n_elements() as usize != n_rows
        || scales.n_elements() as usize != n_rows
        || !(accum.n_elements() as usize).is_multiple_of(n_cols)
    {
        return Err(MetalError::BadShape {
            kernel: "scatter_axpy_rows_unique",
            detail: format!(
                "expected x n={} rows/scales n_rows={} accum multiple of n_cols={}, got x={} rows={} scales={} accum={}",
                n,
                n_rows,
                n_cols,
                x.n_elements(),
                rows.n_elements(),
                scales.n_elements(),
                accum.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_scatter_axpy_rows_unique_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_cols: u32,
        n_rows: u32,
        out_rows: u32,
    }
    let out_rows = accum.n_elements() as usize / n_cols;
    enc.set_bytes(
        0,
        &Args {
            n_cols: n_cols as u32,
            n_rows: n_rows as u32,
            out_rows: out_rows as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, rows);
    enc.set_tensor(3, scales);
    enc.set_tensor(4, accum);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
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

/// In-place softmax over the (only) dimension of `x`. Used for attention
/// scores. Input is mutated.
pub fn encode_softmax_inplace_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    let pso = ctx.pipeline("kernel_softmax_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, x);

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

pub(crate) fn validate_i32_output(
    kernel: &'static str,
    output: &MetalTensor,
    expected_elements: usize,
) -> Result<(), MetalError> {
    if output.n_elements() as usize != expected_elements {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "output.n_elements={} != expected={expected_elements}",
                output.n_elements()
            ),
        });
    }
    if output.dtype != GgmlType::I32
        || !output.is_writable()
        || !output
            .offset
            .is_multiple_of(std::mem::align_of::<i32>() as u64)
    {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "output must be aligned writable I32, got dtype={:?} offset={} provenance={:?}",
                output.dtype,
                output.offset,
                output.provenance()
            ),
        });
    }
    let output_bytes = expected_elements
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: "output byte size overflow".into(),
        })?;
    let output_end = output
        .offset
        .checked_add(output_bytes as u64)
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: "output buffer range overflow".into(),
        })?;
    if output_end > output.buffer.length() as u64 {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "output range offset={} bytes={output_bytes} exceeds buffer={}",
                output.offset,
                output.buffer.length()
            ),
        });
    }
    Ok(())
}

/// GPU-side argmax over `[n_rows, n]` rows of F32, writing `[n_rows]` I32
/// indices. Tie policy: lowest index wins (matches numpy/torch).
///
/// Used by H5.3a `packed_forward` to produce `verify_argmax: [N] i32`
/// without a `[N, V]` CPU readback. At V=248320, N=16 that's 15.9 MB
/// per outer step we don't have to spill to host.
///
/// Layout assumption: `x` is row-major with row stride == `n` (no padding
/// between rows). Each row gets one threadgroup; up to 1024 threads per
/// TG, internally simdgroup-reduced.
pub fn encode_argmax_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    n_rows: usize,
    n: usize,
) -> Result<(), MetalError> {
    if x.n_elements() as usize != n_rows * n {
        return Err(MetalError::BadShape {
            kernel: "argmax",
            detail: format!("x.n_elements={} != n_rows*n={}", x.n_elements(), n_rows * n),
        });
    }
    validate_i32_output("argmax", out_idx, n_rows)?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        stride_x: u32,
    }
    let pso = ctx.pipeline("kernel_argmax_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            stride_x: n as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, out_idx);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    // Two threadgroup arrays (sh_val: f32, sh_idx: u32) — same width.
    enc.set_threadgroup_memory(0, (n_simdgroups * std::mem::size_of::<f32>()).max(32));
    enc.set_threadgroup_memory(1, (n_simdgroups * std::mem::size_of::<u32>()).max(32));

    enc.dispatch(
        MTLSize {
            width: n_rows,
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

/// Argmax index plus (top1 - top2) gap per row via `kernel_argmax_top2_f32`.
/// Same lowest-index tie contract as [`encode_argmax_f32`]. The gap is 0 when
/// the maximum appears twice, +inf for single-element rows, and ignores NaNs
/// (production logits contain none; the greedy NaN variant stays separate).
pub fn encode_argmax_top2_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    out_gap: &MetalTensor,
    n_rows: usize,
    n: usize,
) -> Result<(), MetalError> {
    if x.n_elements() as usize != n_rows * n {
        return Err(MetalError::BadShape {
            kernel: "argmax_top2",
            detail: format!("x.n_elements={} != n_rows*n={}", x.n_elements(), n_rows * n),
        });
    }
    validate_i32_output("argmax_top2", out_idx, n_rows)?;
    if out_gap.n_elements() as usize != n_rows {
        return Err(MetalError::BadShape {
            kernel: "argmax_top2",
            detail: format!(
                "out_gap.n_elements={} != n_rows={}",
                out_gap.n_elements(),
                n_rows
            ),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        stride_x: u32,
    }
    let pso = ctx.pipeline("kernel_argmax_top2_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            stride_x: n as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, out_idx);
    enc.set_tensor(3, out_gap);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let shmem = (tg_threads * std::mem::size_of::<f32>()).max(32);
    enc.set_threadgroup_memory(0, shmem);
    enc.set_threadgroup_memory(1, (tg_threads * std::mem::size_of::<u32>()).max(32));
    enc.set_threadgroup_memory(2, shmem);

    enc.dispatch(
        MTLSize {
            width: n_rows,
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

/// GPU-side top-16 over `[n_rows, n]` rows of F32 (kernels/dflash2.metal):
/// writes `[n_rows, 16]` I32 indices + `[n_rows, 16]` F32 values, sorted
/// descending, ties toward the lower index. DFlash 2 selector candidates:
/// replaces the `[N, V]` logits readback with an `[N, 16]` pair readback.
pub fn encode_topk16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    out_val: &MetalTensor,
    n_rows: usize,
    n: usize,
) -> Result<(), MetalError> {
    const TOP_K: usize = 16;
    if x.n_elements() as usize != n_rows * n {
        return Err(MetalError::BadShape {
            kernel: "topk16",
            detail: format!("x.n_elements={} != n_rows*n={}", x.n_elements(), n_rows * n),
        });
    }
    validate_i32_output("topk16", out_idx, n_rows * TOP_K)?;
    if (out_val.n_elements() as usize) < n_rows * TOP_K {
        return Err(MetalError::BadShape {
            kernel: "topk16",
            detail: format!(
                "out_val elements {} < n_rows*16={}",
                out_val.n_elements(),
                n_rows * TOP_K
            ),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        stride_x: u32,
    }
    let pso = ctx.pipeline("kernel_topk16_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            stride_x: n as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, out_idx);
    enc.set_tensor(3, out_val);

    // 128 threads → 128 · 16 · 4 B = 8 KB per threadgroup array (16 KB
    // total), safely under the 32 KB Apple GPU threadgroup limit.
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(128);
    enc.set_threadgroup_memory(0, tg_threads * TOP_K * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, tg_threads * TOP_K * std::mem::size_of::<u32>());
    enc.dispatch(
        MTLSize {
            width: n_rows,
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

/// Encode Apple's optimized per-row top-16 matrix selection into an existing
/// command buffer. Index outputs are UInt32 bit patterns stored in an I32
/// tensor.
pub fn encode_mps_topk16_f32(
    ctx: &MetalContext,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    out_val: &MetalTensor,
    n_rows: usize,
    n: usize,
) -> Result<(), MetalError> {
    const TOP_K: usize = 16;
    if x.dtype != GgmlType::F32 || x.n_elements() as usize != n_rows * n {
        return Err(MetalError::BadShape {
            kernel: "mps_topk16",
            detail: format!(
                "input dtype/size {:?}/{} != F32/{}",
                x.dtype,
                x.n_elements(),
                n_rows * n
            ),
        });
    }
    validate_i32_output("mps_topk16", out_idx, n_rows * TOP_K)?;
    if out_val.dtype != GgmlType::F32 || out_val.n_elements() as usize != n_rows * TOP_K {
        return Err(MetalError::BadShape {
            kernel: "mps_topk16",
            detail: format!(
                "value output dtype/size {:?}/{} != F32/{}",
                out_val.dtype,
                out_val.n_elements(),
                n_rows * TOP_K
            ),
        });
    }

    let input_descriptor = unsafe {
        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            n_rows,
            n,
            n * std::mem::size_of::<f32>(),
            MPSDataType::Float32,
        )
    };
    let index_descriptor = unsafe {
        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            n_rows,
            TOP_K,
            TOP_K * std::mem::size_of::<u32>(),
            MPSDataType::UInt32,
        )
    };
    let value_descriptor = unsafe {
        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            n_rows,
            TOP_K,
            TOP_K * std::mem::size_of::<f32>(),
            MPSDataType::Float32,
        )
    };
    let input = unsafe {
        MPSMatrix::initWithBuffer_offset_descriptor(
            MPSMatrix::alloc(),
            &x.buffer,
            x.offset as usize,
            &input_descriptor,
        )
    };
    let indices = unsafe {
        MPSMatrix::initWithBuffer_offset_descriptor(
            MPSMatrix::alloc(),
            &out_idx.buffer,
            out_idx.offset as usize,
            &index_descriptor,
        )
    };
    let values = unsafe {
        MPSMatrix::initWithBuffer_offset_descriptor(
            MPSMatrix::alloc(),
            &out_val.buffer,
            out_val.offset as usize,
            &value_descriptor,
        )
    };
    let topk = unsafe {
        MPSMatrixFindTopK::initWithDevice_numberOfTopKValues(
            MPSMatrixFindTopK::alloc(),
            &ctx.device,
            TOP_K,
        )
    };
    unsafe {
        topk.setSourceRows(n_rows);
        topk.setSourceColumns(n);
        topk.encodeToCommandBuffer_inputMatrix_resultIndexMatrix_resultValueMatrix(
            command, &input, &indices, &values,
        );
    }
    Ok(())
}

/// Replace one fixed-width set of selected indices per row with negative
/// infinity so a second selection pass returns the next disjoint set.
pub fn encode_mask_row_indices_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    values: &MetalTensor,
    indices: &MetalTensor,
    n_rows: usize,
    row_width: usize,
    index_count: usize,
) -> Result<(), MetalError> {
    values.assert_writable("mask row indices");
    if values.dtype != GgmlType::F32 || values.n_elements() as usize != n_rows * row_width {
        return Err(MetalError::BadShape {
            kernel: "mask_row_indices",
            detail: "value matrix shape or dtype mismatch".into(),
        });
    }
    validate_i32_output("mask_row_indices", indices, n_rows * index_count)?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_count: u32,
        row_width: u32,
        index_count: u32,
    }
    let pso = ctx.pipeline("kernel_mask_row_indices_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_count: n_rows as u32,
            row_width: row_width as u32,
            index_count: index_count as u32,
        },
    );
    enc.set_tensor(1, values);
    enc.set_tensor(2, indices);
    let total = n_rows * index_count;
    let threads = pso.maxTotalThreadsPerThreadgroup().min(256);
    enc.dispatch(
        MTLSize {
            width: total.div_ceil(threads),
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

/// GPU-side greedy selection matching the sampler's `f32::total_cmp` order.
/// Equal bit patterns choose the LOWEST token id (2026-08-22 tie-inversion
/// unification). Any NaN is encoded as the negative value `~token_id`, with
/// the lowest NaN token taking precedence.
pub fn encode_argmax_f32_greedy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    n_rows: usize,
    n: usize,
) -> Result<(), MetalError> {
    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "argmax_greedy",
            detail: "n_rows must be >= 1".to_string(),
        });
    }
    if n == 0 || n > i32::MAX as usize {
        return Err(MetalError::BadShape {
            kernel: "argmax_greedy",
            detail: format!("row length {n} must be in 1..=i32::MAX"),
        });
    }
    let expected = n_rows.checked_mul(n).ok_or_else(|| MetalError::BadShape {
        kernel: "argmax_greedy",
        detail: format!("n_rows*n overflows usize: {n_rows}*{n}"),
    })?;
    if x.n_elements() as usize != expected {
        return Err(MetalError::BadShape {
            kernel: "argmax_greedy",
            detail: format!("x.n_elements={} != n_rows*n={expected}", x.n_elements()),
        });
    }
    validate_i32_output("argmax_greedy", out_idx, n_rows)?;

    let pso = ctx.pipeline("kernel_argmax_f32_greedy")?;
    let simd_width = pso.threadExecutionWidth();
    if simd_width == 0 {
        return Err(MetalError::BadShape {
            kernel: "argmax_greedy",
            detail: "pipeline reported zero thread execution width".to_string(),
        });
    }
    let max_tg_threads = pso
        .maxTotalThreadsPerThreadgroup()
        .min(simd_width.saturating_mul(simd_width));
    let tg_threads = (max_tg_threads / simd_width) * simd_width;
    if tg_threads == 0 {
        return Err(MetalError::BadShape {
            kernel: "argmax_greedy",
            detail: format!(
                "pipeline max threadgroup width {max_tg_threads} is below SIMD width {simd_width}"
            ),
        });
    }
    let n_simdgroups = tg_threads / simd_width;
    if n_simdgroups > simd_width {
        return Err(MetalError::BadShape {
            kernel: "argmax_greedy",
            detail: format!(
                "{n_simdgroups} simdgroups exceed one {simd_width}-lane reduction group"
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        stride_x: u32,
        n_simdgroups: u32,
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            stride_x: n as u32,
            n_simdgroups: n_simdgroups as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, out_idx);

    let partial_bytes = (n_simdgroups * std::mem::size_of::<u32>()).max(32);
    enc.set_threadgroup_memory(0, partial_bytes);
    enc.set_threadgroup_memory(1, partial_bytes);
    enc.set_threadgroup_memory(2, partial_bytes);
    enc.dispatch(
        MTLSize {
            width: n_rows,
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

pub(crate) fn validate_copy_offset_32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    src_off: usize,
    dst: &MetalTensor,
    n_elements: usize,
    dtype: GgmlType,
    kernel: &'static str,
) -> Result<(u32, u32), MetalError> {
    if n_elements == 0 {
        return Err(MetalError::BadShape {
            kernel,
            detail: "n_elements must be nonzero".into(),
        });
    }
    if src.dtype != dtype || dst.dtype != dtype {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "src/dst expected {dtype:?}, got {:?}/{:?}",
                src.dtype, dst.dtype
            ),
        });
    }
    if !dst.is_writable() {
        return Err(MetalError::BadShape {
            kernel,
            detail: "destination must be writable".into(),
        });
    }
    if dst.n_elements() as usize != n_elements {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("dst.n={} != n_elements={n_elements}", dst.n_elements()),
        });
    }
    let source_end = src_off
        .checked_add(n_elements)
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: "source element range overflows".into(),
        })?;
    if source_end as u64 > src.n_elements() {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("src_off+n={source_end} > src.n={}", src.n_elements()),
        });
    }
    if !src.offset.is_multiple_of(4) || !dst.offset.is_multiple_of(4) {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "src/dst byte offsets must be 4-byte aligned, got {}/{}",
                src.offset, dst.offset
            ),
        });
    }
    let copy_bytes = (n_elements as u64)
        .checked_mul(4)
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: "copy byte count overflows".into(),
        })?;
    let source_start = (src_off as u64)
        .checked_mul(4)
        .and_then(|offset| src.offset.checked_add(offset))
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: "source byte offset overflows".into(),
        })?;
    let source_byte_end =
        source_start
            .checked_add(copy_bytes)
            .ok_or_else(|| MetalError::BadShape {
                kernel,
                detail: "source byte endpoint overflows".into(),
            })?;
    let destination_end =
        dst.offset
            .checked_add(copy_bytes)
            .ok_or_else(|| MetalError::BadShape {
                kernel,
                detail: "destination byte endpoint overflows".into(),
            })?;
    if source_byte_end > src.buffer.length() as u64 || destination_end > dst.buffer.length() as u64
    {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "copy range src={source_start}..{source_byte_end}/{} dst={}..{destination_end}/{} exceeds backing storage",
                src.buffer.length(),
                dst.offset,
                dst.buffer.length(),
            ),
        });
    }
    let expected_device = ctx.device.registryID();
    let encoder_device = enc.parent_command_buffer().device().registryID();
    let source_device = src.buffer.device().registryID();
    let destination_device = dst.buffer.device().registryID();
    if encoder_device != expected_device
        || source_device != expected_device
        || destination_device != expected_device
    {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "device mismatch context={expected_device} encoder={encoder_device} src={source_device} dst={destination_device}"
            ),
        });
    }
    if Retained::as_ptr(&src.buffer) == Retained::as_ptr(&dst.buffer)
        && source_start < destination_end
        && dst.offset < source_byte_end
    {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "overlapping copy ranges src={source_start}..{source_byte_end} dst={}..{destination_end}",
                dst.offset
            ),
        });
    }
    let n = u32::try_from(n_elements).map_err(|_| MetalError::BadShape {
        kernel,
        detail: format!("n_elements={n_elements} exceeds u32"),
    })?;
    let src_off = u32::try_from(src_off).map_err(|_| MetalError::BadShape {
        kernel,
        detail: format!("src_off={src_off} exceeds u32"),
    })?;
    Ok((n, src_off))
}

/// Copy `n_elements` floats starting at `src_off` (in elements) of `src`
/// into `dst[0..n_elements]`. Used to slice fused buffers (e.g. the GDN
/// post-conv qkv buffer) into per-role tensors. v2 will replace many of
/// these with kernels that take offsets directly.
pub fn encode_copy_offset_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    src_off: usize,
    dst: &MetalTensor,
    n_elements: usize,
) -> Result<(), MetalError> {
    let (n, src_off) = validate_copy_offset_32(
        ctx,
        enc,
        src,
        src_off,
        dst,
        n_elements,
        GgmlType::F32,
        "copy_offset",
    )?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        src_off: u32,
    }
    let pso = ctx.pipeline("kernel_copy_offset_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &Args { n, src_off });
    enc.set_tensor(1, src);
    enc.set_tensor(2, dst);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n_elements.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
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

pub fn encode_copy_offset_i32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    src_off: usize,
    dst: &MetalTensor,
    n_elements: usize,
) -> Result<(), MetalError> {
    let (n, src_off) = validate_copy_offset_32(
        ctx,
        enc,
        src,
        src_off,
        dst,
        n_elements,
        GgmlType::I32,
        "copy_offset_i32",
    )?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        src_off: u32,
    }
    let pso = ctx.pipeline("kernel_copy_offset_i32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &Args { n, src_off });
    enc.set_tensor(1, src);
    enc.set_tensor(2, dst);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n_elements.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
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

/// Gated attention `out = x * sigmoid(gate)` where the gate rows live
/// strided inside a larger tensor (the interleaved q_proj output's gate
/// halves: gate_stride = 2*head_dim, gate_offset = head_dim). `x`/`out`
/// are compact `n_rows * head_dim` elements (v0.432; replaces
/// split_q_gate + compact sigmoid_mul).
pub fn encode_sigmoid_mul_gate_strided_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    x: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    head_dim: usize,
    gate_stride: usize,
    gate_offset: usize,
) -> Result<(), MetalError> {
    if n_rows == 0 || head_dim == 0 {
        return Err(MetalError::BadShape {
            kernel: "sigmoid_mul_gate_strided",
            detail: "n_rows/head_dim must be nonzero".to_string(),
        });
    }
    let n = (n_rows * head_dim) as u64;
    if x.n_elements() != n || out.n_elements() != n {
        return Err(MetalError::BadShape {
            kernel: "sigmoid_mul_gate_strided",
            detail: format!("x/out expected {n} elements"),
        });
    }
    let gate_need = gate_offset as u64 + (n_rows as u64 - 1) * gate_stride as u64 + head_dim as u64;
    if gate.n_elements() < gate_need {
        return Err(MetalError::BadShape {
            kernel: "sigmoid_mul_gate_strided",
            detail: format!(
                "gate has {} elements, needs >= {gate_need}",
                gate.n_elements()
            ),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        head_dim: u32,
        gate_stride: u32,
        gate_offset: u32,
    }
    let pso = ctx.pipeline("kernel_sigmoid_mul_gate_strided_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            head_dim: head_dim as u32,
            gate_stride: gate_stride as u32,
            gate_offset: gate_offset as u32,
        },
    );
    enc.set_tensor(1, gate);
    enc.set_tensor(2, x);
    enc.set_tensor(3, out);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: (n as usize).div_ceil(tg_threads),
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

/// Scatter F32 source bytes into a F16 destination buffer at offset.
/// Used for KV cache append when the cache is F16. Counterpart of
/// `encode_scatter_offset_f32` (F32 → F32).
pub fn encode_scatter_offset_f32_to_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    dst_off: usize,
    n: usize,
) -> Result<(), MetalError> {
    if src.dtype != GgmlType::F32 || dst.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16",
            detail: format!("expected F32→F16, got {:?}→{:?}", src.dtype, dst.dtype),
        });
    }
    if src.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16",
            detail: format!("src.n={} != n={n}", src.n_elements()),
        });
    }
    if (dst_off + n) as u64 > dst.n_elements() {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16",
            detail: format!("dst_off+n={} > dst.n={}", dst_off + n, dst.n_elements()),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        dst_off: u32,
    }
    let pso = ctx.pipeline("kernel_scatter_offset_f32_to_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            dst_off: dst_off as u32,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, dst);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
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

/// Fused K+V scatter — writes K and V into their respective F16 caches at
/// `dst_off` in a single dispatch. K and V always share `n` and `dst_off`
/// at decode time (`n = kv_dim, dst_off = position * kv_dim`), so we
/// amortize one dispatch per attn layer.
///
/// Per Jeff & Sanjay (Bulk APIs / amortize boundary crossings).
/// Saves 16 dispatches/token for the 27B (16 attn layers).
pub fn encode_scatter_offset_f32_to_f16_kv(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    k_src: &MetalTensor,
    v_src: &MetalTensor,
    k_dst: &MetalTensor,
    v_dst: &MetalTensor,
    dst_off: usize,
    n: usize,
) -> Result<(), MetalError> {
    if k_src.dtype != GgmlType::F32 || v_src.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv",
            detail: format!(
                "expected F32 sources, got k={:?} v={:?}",
                k_src.dtype, v_src.dtype
            ),
        });
    }
    if k_dst.dtype != GgmlType::F16 || v_dst.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv",
            detail: format!(
                "expected F16 dests, got k={:?} v={:?}",
                k_dst.dtype, v_dst.dtype
            ),
        });
    }
    if k_src.n_elements() as usize != n || v_src.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv",
            detail: format!(
                "src lengths k={} v={} != n={n}",
                k_src.n_elements(),
                v_src.n_elements()
            ),
        });
    }
    let dst_end = dst_off.checked_add(n).ok_or_else(|| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_f16_kv",
        detail: format!("dst_off={dst_off} + n={n} overflows usize"),
    })?;
    if dst_end as u64 > k_dst.n_elements() || dst_end as u64 > v_dst.n_elements() {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv",
            detail: format!(
                "dst_off+n={} exceeds k.n={} or v.n={}",
                dst_end,
                k_dst.n_elements(),
                v_dst.n_elements()
            ),
        });
    }
    let n_u32 = u32::try_from(n).map_err(|_| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_f16_kv",
        detail: format!("n={n} does not fit u32 kernel args"),
    })?;
    let dst_off_u32 = u32::try_from(dst_off).map_err(|_| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_f16_kv",
        detail: format!("dst_off={dst_off} does not fit u32 kernel args"),
    })?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        dst_off: u32,
    }
    let pso = ctx.pipeline("kernel_scatter_offset_f32_to_f16_kv")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n_u32,
            dst_off: dst_off_u32,
        },
    );
    enc.set_tensor(1, k_src);
    enc.set_tensor(2, v_src);
    enc.set_tensor(3, k_dst);
    enc.set_tensor(4, v_dst);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
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

/// Fused K+V scatter plus transposed-V sidecar write. This is intentionally
/// narrow support for the experimental non-flash matrix attention path: it keeps
/// the canonical `[pos, kv]` V cache intact while also filling a fixed-stride
/// `[kvh, d, pos]` V_T bank for KQV.
pub fn encode_scatter_offset_f32_to_f16_kv_vt(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    k_src: &MetalTensor,
    v_src: &MetalTensor,
    k_dst: &MetalTensor,
    v_dst: &MetalTensor,
    v_t: &MetalTensor,
    dst_off: usize,
    n: usize,
    base_pos: usize,
    kv_dim: usize,
    head_dim: usize,
    vt_stride: usize,
) -> Result<(), MetalError> {
    if k_src.dtype != GgmlType::F32 || v_src.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!(
                "expected F32 sources, got k={:?} v={:?}",
                k_src.dtype, v_src.dtype
            ),
        });
    }
    if k_dst.dtype != GgmlType::F16 || v_dst.dtype != GgmlType::F16 || v_t.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!(
                "expected F16 dests, got k={:?} v={:?} vt={:?}",
                k_dst.dtype, v_dst.dtype, v_t.dtype
            ),
        });
    }
    if kv_dim == 0 || head_dim == 0 || !kv_dim.is_multiple_of(head_dim) {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!("bad kv_dim/head_dim: kv_dim={kv_dim} head_dim={head_dim}"),
        });
    }
    if n == 0 || !n.is_multiple_of(kv_dim) {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!("n={n} must be a positive multiple of kv_dim={kv_dim}"),
        });
    }
    if k_src.n_elements() as usize != n || v_src.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!(
                "src lengths k={} v={} != n={n}",
                k_src.n_elements(),
                v_src.n_elements()
            ),
        });
    }
    let dst_end = dst_off.checked_add(n).ok_or_else(|| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_f16_kv_vt",
        detail: format!("dst_off={dst_off} + n={n} overflows usize"),
    })?;
    if dst_end as u64 > k_dst.n_elements() || dst_end as u64 > v_dst.n_elements() {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!(
                "dst_off+n={} exceeds k.n={} or v.n={}",
                dst_end,
                k_dst.n_elements(),
                v_dst.n_elements()
            ),
        });
    }
    let n_rows = n / kv_dim;
    if vt_stride < base_pos + n_rows {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!(
                "vt_stride={vt_stride} < base_pos+n_rows={}",
                base_pos + n_rows
            ),
        });
    }
    let want_vt = kv_dim
        .checked_mul(vt_stride)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!("kv_dim={kv_dim} * vt_stride={vt_stride} overflows usize"),
        })?;
    if v_t.n_elements() < want_vt as u64 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_f16_kv_vt",
            detail: format!("v_t has {} elements, need >= {want_vt}", v_t.n_elements()),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        dst_off: u32,
        base_pos: u32,
        kv_dim: u32,
        head_dim: u32,
        vt_stride: u32,
    }
    let pso = ctx.pipeline("kernel_scatter_offset_f32_to_f16_kv_vt")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: u32::try_from(n).map_err(|_| MetalError::BadShape {
                kernel: "scatter_offset_f32_to_f16_kv_vt",
                detail: format!("n={n} does not fit u32"),
            })?,
            dst_off: u32::try_from(dst_off).map_err(|_| MetalError::BadShape {
                kernel: "scatter_offset_f32_to_f16_kv_vt",
                detail: format!("dst_off={dst_off} does not fit u32"),
            })?,
            base_pos: u32::try_from(base_pos).map_err(|_| MetalError::BadShape {
                kernel: "scatter_offset_f32_to_f16_kv_vt",
                detail: format!("base_pos={base_pos} does not fit u32"),
            })?,
            kv_dim: u32::try_from(kv_dim).map_err(|_| MetalError::BadShape {
                kernel: "scatter_offset_f32_to_f16_kv_vt",
                detail: format!("kv_dim={kv_dim} does not fit u32"),
            })?,
            head_dim: u32::try_from(head_dim).map_err(|_| MetalError::BadShape {
                kernel: "scatter_offset_f32_to_f16_kv_vt",
                detail: format!("head_dim={head_dim} does not fit u32"),
            })?,
            vt_stride: u32::try_from(vt_stride).map_err(|_| MetalError::BadShape {
                kernel: "scatter_offset_f32_to_f16_kv_vt",
                detail: format!("vt_stride={vt_stride} does not fit u32"),
            })?,
        },
    );
    enc.set_tensor(1, k_src);
    enc.set_tensor(2, v_src);
    enc.set_tensor(3, k_dst);
    enc.set_tensor(4, v_dst);
    enc.set_tensor(5, v_t);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
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

/// Fused K+V scatter into Q8_0 caches. Quantizes each 32-element block with
/// ggml's reference rule: `d = amax / 127`, `qs[j] = round(x[j] / d)`.
///
/// Constraints: `dst_off` and `n` must both be multiples of 32 so the append
/// lands on Q8_0 block boundaries.
pub fn encode_scatter_offset_f32_to_q8_0_kv(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    k_src: &MetalTensor,
    v_src: &MetalTensor,
    k_dst: &MetalTensor,
    v_dst: &MetalTensor,
    dst_off: usize,
    n: usize,
) -> Result<(), MetalError> {
    const QK8_0: usize = 32;
    if k_src.dtype != GgmlType::F32 || v_src.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_q8_0_kv",
            detail: format!(
                "expected F32 sources, got k={:?} v={:?}",
                k_src.dtype, v_src.dtype
            ),
        });
    }
    if k_dst.dtype != GgmlType::Q8_0 || v_dst.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_q8_0_kv",
            detail: format!(
                "expected Q8_0 dests, got k={:?} v={:?}",
                k_dst.dtype, v_dst.dtype
            ),
        });
    }
    if k_src.n_elements() as usize != n || v_src.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_q8_0_kv",
            detail: format!(
                "src lengths k={} v={} != n={n}",
                k_src.n_elements(),
                v_src.n_elements()
            ),
        });
    }
    if !dst_off.is_multiple_of(QK8_0) || !n.is_multiple_of(QK8_0) {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_q8_0_kv",
            detail: format!("dst_off={dst_off} and n={n} must both be multiples of {QK8_0}"),
        });
    }
    let dst_end = dst_off.checked_add(n).ok_or_else(|| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_q8_0_kv",
        detail: format!("dst_off={dst_off} + n={n} overflows usize"),
    })?;
    if dst_end as u64 > k_dst.n_elements() || dst_end as u64 > v_dst.n_elements() {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset_f32_to_q8_0_kv",
            detail: format!(
                "dst_off+n={} exceeds k.n={} or v.n={}",
                dst_end,
                k_dst.n_elements(),
                v_dst.n_elements()
            ),
        });
    }
    let n_blocks = n / QK8_0;
    let dst_block_off = dst_off / QK8_0;
    let n_blocks_u32 = u32::try_from(n_blocks).map_err(|_| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_q8_0_kv",
        detail: format!("n_blocks={n_blocks} does not fit u32 kernel args"),
    })?;
    let dst_block_off_u32 = u32::try_from(dst_block_off).map_err(|_| MetalError::BadShape {
        kernel: "scatter_offset_f32_to_q8_0_kv",
        detail: format!("dst_block_off={dst_block_off} does not fit u32 kernel args"),
    })?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_blocks: u32,
        dst_block_off: u32,
    }
    let pso = ctx.pipeline("kernel_scatter_offset_f32_to_q8_0_kv")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_blocks: n_blocks_u32,
            dst_block_off: dst_block_off_u32,
        },
    );
    enc.set_tensor(1, k_src);
    enc.set_tensor(2, v_src);
    enc.set_tensor(3, k_dst);
    enc.set_tensor(4, v_dst);
    enc.dispatch(
        MTLSize {
            width: n / QK8_0,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Row lookup: `y[r * n_cols + i] = source[ids[r] * n_cols + i]`.
/// The source may be flat or multidimensional; its logical element count
/// defines the row count. GGUF embeddings are normally `[n_cols, vocab]`.
pub fn encode_get_rows_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    embed: &MetalTensor,
    ids: &MetalTensor,
    y: &MetalTensor,
    n_rows: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!("n_rows={n_rows} and n_cols={n_cols} must be nonzero"),
        });
    }
    let output_elements = n_rows
        .checked_mul(n_cols)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "get_rows",
            detail: "n_rows*n_cols overflow".into(),
        })?;
    if y.n_elements() as usize != output_elements {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                "y.n={} != n_rows*n_cols={}",
                y.n_elements(),
                n_rows * n_cols
            ),
        });
    }
    if ids.n_elements() as usize != n_rows {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!("ids.n={} != n_rows={n_rows}", ids.n_elements()),
        });
    }
    if ids.dtype != GgmlType::I32
        || !ids
            .offset
            .is_multiple_of(std::mem::align_of::<i32>() as u64)
    {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                "ids must be aligned I32, got dtype={:?} offset={}",
                ids.dtype, ids.offset
            ),
        });
    }
    let ids_bytes = n_rows
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: "get_rows",
            detail: "ids byte size overflow".into(),
        })?;
    let ids_end = ids
        .offset
        .checked_add(ids_bytes as u64)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "get_rows",
            detail: "ids buffer range overflow".into(),
        })?;
    if ids_end > ids.buffer.length() as u64 {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                "ids range offset={} bytes={ids_bytes} exceeds buffer={}",
                ids.offset,
                ids.buffer.length()
            ),
        });
    }
    if y.dtype != GgmlType::F32
        || !y.is_writable()
        || !y.offset.is_multiple_of(std::mem::align_of::<f32>() as u64)
    {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                "output must be aligned writable F32, got dtype={:?} offset={} provenance={:?}",
                y.dtype,
                y.offset,
                y.provenance()
            ),
        });
    }
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: "get_rows",
            detail: "output byte size overflow".into(),
        })?;
    let output_end =
        y.offset
            .checked_add(output_bytes as u64)
            .ok_or_else(|| MetalError::BadShape {
                kernel: "get_rows",
                detail: "output buffer range overflow".into(),
            })?;
    if output_end > y.buffer.length() as u64 {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                "output range offset={} bytes={output_bytes} exceeds buffer={}",
                y.offset,
                y.buffer.length()
            ),
        });
    }
    let source_elements =
        usize::try_from(embed.n_elements()).map_err(|_| MetalError::BadShape {
            kernel: "get_rows",
            detail: format!("source element count {} exceeds usize", embed.n_elements()),
        })?;
    if source_elements == 0 || source_elements % n_cols != 0 {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                concat!(
                    "source shape {:?} with {} elements is not a nonempty ",
                    "collection of {}-element rows"
                ),
                embed.shape, source_elements, n_cols,
            ),
        });
    }
    let source_rows = source_elements / n_cols;
    let block_layout = match embed.dtype {
        GgmlType::F32 => (1usize, 4usize),
        GgmlType::F16 | GgmlType::BF16 => (1, 2),
        GgmlType::Q4_K => (256, 144),
        GgmlType::Q6_K => (256, 210),
        GgmlType::Q8_0 => (32, 34),
        GgmlType::IQ4_NL => (32, 18),
        other => {
            return Err(MetalError::BadShape {
                kernel: "get_rows",
                detail: format!("unsupported embedding dtype {other:?}"),
            });
        }
    };
    let (block_elements, block_bytes) = block_layout;
    let source_alignment = match embed.dtype {
        GgmlType::F32 => std::mem::align_of::<f32>(),
        GgmlType::F16
        | GgmlType::BF16
        | GgmlType::Q4_K
        | GgmlType::Q6_K
        | GgmlType::Q8_0
        | GgmlType::IQ4_NL => std::mem::align_of::<u16>(),
        _ => unreachable!(),
    } as u64;
    if !embed.offset.is_multiple_of(source_alignment) {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                "source {:?} offset {} is not {source_alignment}-byte aligned",
                embed.dtype, embed.offset
            ),
        });
    }
    if !n_cols.is_multiple_of(block_elements) {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                "n_cols={n_cols} is not divisible by {:?} block size {block_elements}",
                embed.dtype
            ),
        });
    }
    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| MetalError::BadShape {
        kernel: "get_rows",
        detail: format!("n_rows={n_rows} exceeds u32"),
    })?;
    let n_cols_u32 = u32::try_from(n_cols).map_err(|_| MetalError::BadShape {
        kernel: "get_rows",
        detail: format!("n_cols={n_cols} exceeds u32"),
    })?;
    let source_rows_u32 = u32::try_from(source_rows).map_err(|_| MetalError::BadShape {
        kernel: "get_rows",
        detail: format!("source row count {source_rows} exceeds u32"),
    })?;
    let expected_bytes = n_cols
        .checked_div(block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .and_then(|row_bytes| row_bytes.checked_mul(source_rows))
        .ok_or_else(|| MetalError::BadShape {
            kernel: "get_rows",
            detail: "embedding byte size overflow".into(),
        })?;
    let buffer_end = embed
        .offset
        .checked_add(expected_bytes as u64)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "get_rows",
            detail: "embedding buffer range overflow".into(),
        })?;
    if embed.n_bytes() as usize != expected_bytes || buffer_end > embed.buffer.length() as u64 {
        return Err(MetalError::BadShape {
            kernel: "get_rows",
            detail: format!(
                concat!(
                    "embedding bytes mismatch: logical={} expected={} ",
                    "offset={} buffer={}"
                ),
                embed.n_bytes(),
                expected_bytes,
                embed.offset,
                embed.buffer.length()
            ),
        });
    }
    let kernel_name = match embed.dtype {
        GgmlType::F32 => "kernel_get_rows_f32",
        GgmlType::F16 => "kernel_get_rows_f16",
        GgmlType::BF16 => "kernel_get_rows_bf16",
        GgmlType::Q4_K => "kernel_get_rows_q4_K_f32",
        GgmlType::Q6_K => "kernel_get_rows_q6_K_f32",
        GgmlType::Q8_0 => "kernel_get_rows_q8_0_f32",
        GgmlType::IQ4_NL => "kernel_get_rows_iq4_nl_f32",
        _ => unreachable!(),
    };
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GetRowsArgs {
        n_rows: u32,
        n_cols: u32,
        n_vocab: u32,
    }
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &GetRowsArgs {
            n_rows: n_rows_u32,
            n_cols: n_cols_u32,
            n_vocab: source_rows_u32,
        },
    );
    enc.set_tensor(1, embed);
    enc.set_tensor(2, ids);
    enc.set_tensor(3, y);

    // 2D grid: (n_cols, n_rows).
    enc.dispatch(
        MTLSize {
            width: n_cols.div_ceil(32),
            height: n_rows,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(crate) fn validate_compact_f32_tensor(
    kernel: &'static str,
    name: &str,
    tensor: &MetalTensor,
    shape: &[u64],
    writable: bool,
) -> Result<(), MetalError> {
    let (_, bytes) = checked_shape_bytes(shape, std::mem::size_of::<f32>())?;
    if tensor.dtype != GgmlType::F32
        || tensor.shape != shape
        || (writable && !tensor.is_writable())
        || !tensor_physical_range_valid(tensor, bytes, std::mem::align_of::<f32>() as u64)
    {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "{name} expected {}F32 {shape:?}, got {:?} {:?} writable={} offset={}",
                if writable { "writable " } else { "" },
                tensor.dtype,
                tensor.shape,
                tensor.is_writable(),
                tensor.offset
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;

    #[test]
    fn mps_two_pass_topk_matches_cpu_at_full_vocabulary_width() {
        // Exercise the rounded mask-dispatch tail as well as full vocabulary width.
        const ROWS: usize = 3;
        const COLUMNS: usize = 248_320;
        const PASS_K: usize = 16;
        const RESULT_K: usize = 25;
        const PRIME: usize = 1_000_003;
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("initialize Metal: {error}"),
        };
        let mut host = Vec::with_capacity(ROWS * COLUMNS);
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                host.push(((column * 48_271 + row * 17) % PRIME) as f32);
            }
        }
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&host),
            vec![ROWS as u64, COLUMNS as u64],
            GgmlType::F32,
        )
        .expect("input tensor");
        let first_ids =
            MetalTensor::zeros_i32(&ctx, vec![ROWS as u64, PASS_K as u64]).expect("first IDs");
        let first_values =
            MetalTensor::zeros_f32(&ctx, vec![ROWS as u64, PASS_K as u64]).expect("first values");
        let second_ids =
            MetalTensor::zeros_i32(&ctx, vec![ROWS as u64, PASS_K as u64]).expect("second IDs");
        let second_values =
            MetalTensor::zeros_f32(&ctx, vec![ROWS as u64, PASS_K as u64]).expect("second values");
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let started = std::time::Instant::now();
        encode_mps_topk16_f32(
            &ctx,
            &command,
            &input,
            &first_ids,
            &first_values,
            ROWS,
            COLUMNS,
        )
        .expect("first top-k pass");
        let encoder = KernelEncoder::begin(&command);
        encode_mask_row_indices_f32(&ctx, &encoder, &input, &first_ids, ROWS, COLUMNS, PASS_K)
            .expect("mask first-pass IDs");
        encoder.end();
        encode_mps_topk16_f32(
            &ctx,
            &command,
            &input,
            &second_ids,
            &second_values,
            ROWS,
            COLUMNS,
        )
        .expect("second top-k pass");
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        let elapsed = started.elapsed();
        assert_eq!(
            command.status(),
            objc2_metal::MTLCommandBufferStatus::Completed,
            "MPS top-k command failed: {:?}",
            command.error()
        );

        let read_ids = |tensor: &MetalTensor| unsafe {
            std::slice::from_raw_parts(
                tensor.buffer.contents().as_ptr().cast::<i32>(),
                ROWS * PASS_K,
            )
            .to_vec()
        };
        let first_ids = read_ids(&first_ids);
        let second_ids = read_ids(&second_ids);
        let first_values = read_back_f32(&first_values.buffer, ROWS * PASS_K);
        let second_values = read_back_f32(&second_values.buffer, ROWS * PASS_K);
        for row in 0..ROWS {
            let mut candidates = Vec::with_capacity(2 * PASS_K);
            for (ids, values) in [(&first_ids, &first_values), (&second_ids, &second_values)] {
                for column in 0..PASS_K {
                    let offset = row * PASS_K + column;
                    candidates.push((ids[offset] as usize, values[offset]));
                }
            }
            candidates.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let mut expected = Vec::<(usize, f32)>::with_capacity(RESULT_K);
            for column in 0..COLUMNS {
                let value = host[row * COLUMNS + column];
                let insertion = expected.partition_point(|&(id, score)| {
                    score > value || (score == value && id < column)
                });
                if insertion < RESULT_K {
                    expected.insert(insertion, (column, value));
                    if expected.len() > RESULT_K {
                        expected.pop();
                    }
                }
            }
            assert_eq!(&candidates[..RESULT_K], expected.as_slice(), "row {row}");
        }
        eprintln!(
            "MPS two-pass top-25 rows={ROWS} columns={COLUMNS} wall_ms={:.3}",
            elapsed.as_secs_f64() * 1e3
        );
    }

    #[test]
    fn get_rows_q6_k_gpu_matches_cpu_with_offsets_and_rejects_bad_inputs() {
        let ctx = match metal_test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        const N_VOCAB: usize = 3;
        const N_COLS: usize = 512;
        let mut weight_bytes = Vec::new();
        let mut decoded = vec![0.0f32; N_VOCAB * N_COLS];
        for row in 0..N_VOCAB {
            for block_index in 0..2 {
                let (block, values) = encode_q6_k_block(
                    if (row + block_index) % 2 == 0 {
                        0.5
                    } else {
                        -0.25
                    },
                    row * 2 + block_index,
                );
                weight_bytes.extend_from_slice(&block);
                let start = row * N_COLS + block_index * 256;
                decoded[start..start + 256].copy_from_slice(&values);
            }
        }
        let desc = TensorDesc {
            name: "q6_k_test".into(),
            shape: vec![N_COLS as u64, N_VOCAB as u64],
            dtype: GgmlType::Q6_K,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: weight_bytes.len() as u64,
        };
        let codec_decoded = crate::codec::dequant_to_f32(&desc, &weight_bytes)
            .expect("llama.cpp Q6_K reference dequantization");
        assert_eq!(decoded, codec_decoded);
        let decoded = codec_decoded;
        let embed = offset_tensor(
            &ctx,
            32,
            &weight_bytes,
            19,
            vec![N_COLS as u64, N_VOCAB as u64],
            GgmlType::Q6_K,
        );
        let row_ids = [2i32, -1, 0, 3];
        let ids = offset_tensor(
            &ctx,
            16,
            bytemuck::cast_slice(&row_ids),
            12,
            vec![row_ids.len() as u64],
            GgmlType::I32,
        );
        let output_bytes = vec![0u8; row_ids.len() * N_COLS * size_of::<f32>()];
        let y = offset_tensor(
            &ctx,
            32,
            &output_bytes,
            20,
            vec![(row_ids.len() * N_COLS) as u64],
            GgmlType::F32,
        );
        one_shot(&ctx, |enc| {
            encode_get_rows_f32(&ctx, enc, &embed, &ids, &y, row_ids.len(), N_COLS)
        })
        .expect("Q6_K get_rows");

        let actual = tensor_f32_at_offset(&y);
        for (lookup_row, &row_id) in row_ids.iter().enumerate() {
            let output_row = &actual[lookup_row * N_COLS..(lookup_row + 1) * N_COLS];
            if row_id >= 0 && (row_id as usize) < N_VOCAB {
                let expected = &decoded[row_id as usize * N_COLS..(row_id as usize + 1) * N_COLS];
                assert_eq!(output_row, expected);
            } else {
                assert!(output_row.iter().all(|&value| value == 0.0));
            }
        }

        let cmd = ctx
            .queue
            .commandBuffer()
            .expect("validation command buffer");
        let enc = KernelEncoder::begin(&cmd);
        assert!(encode_get_rows_f32(&ctx, &enc, &embed, &ids, &y, row_ids.len(), 255).is_err());
        let mut short_embed = embed.clone();
        short_embed.offset = short_embed.buffer.length() as u64 - 1;
        assert!(
            encode_get_rows_f32(&ctx, &enc, &short_embed, &ids, &y, row_ids.len(), N_COLS).is_err()
        );
        let mut misaligned_embed = embed.clone();
        misaligned_embed.offset += 1;
        assert!(
            encode_get_rows_f32(
                &ctx,
                &enc,
                &misaligned_embed,
                &ids,
                &y,
                row_ids.len(),
                N_COLS,
            )
            .is_err()
        );
        let mut short_ids = ids.clone();
        short_ids.offset = short_ids.buffer.length() as u64 - 1;
        assert!(
            encode_get_rows_f32(&ctx, &enc, &embed, &short_ids, &y, row_ids.len(), N_COLS).is_err()
        );
        let mut wrong_ids_dtype = ids.clone();
        wrong_ids_dtype.dtype = GgmlType::F32;
        assert!(
            encode_get_rows_f32(
                &ctx,
                &enc,
                &embed,
                &wrong_ids_dtype,
                &y,
                row_ids.len(),
                N_COLS,
            )
            .is_err()
        );
        enc.end();
    }

    #[test]
    fn get_rows_iq4_nl_gpu_matches_cpu_with_ple_width_and_offsets() {
        let ctx = match metal_test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        const N_VOCAB: usize = 3;
        const N_COLS: usize = 160;
        const BLOCKS_PER_ROW: usize = N_COLS / 32;
        let mut weight_bytes = Vec::new();
        let mut hand_decoded = vec![0.0f32; N_VOCAB * N_COLS];
        for row in 0..N_VOCAB {
            for block_index in 0..BLOCKS_PER_ROW {
                let ordinal = row * BLOCKS_PER_ROW + block_index;
                let sign = if ordinal.is_multiple_of(2) { 1.0 } else { -1.0 };
                let d = sign * (ordinal % 7 + 1) as f32 / 1024.0;
                let (block, values) = encode_iq4_nl_block(d, ordinal);
                weight_bytes.extend_from_slice(&block);
                let start = row * N_COLS + block_index * 32;
                hand_decoded[start..start + 32].copy_from_slice(&values);
            }
        }
        let desc = TensorDesc {
            name: "iq4_nl_ple_rows".into(),
            shape: vec![N_COLS as u64, N_VOCAB as u64],
            dtype: GgmlType::IQ4_NL,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: weight_bytes.len() as u64,
        };
        let codec_decoded = crate::codec::dequant_to_f32(&desc, &weight_bytes)
            .expect("llama.cpp IQ4_NL reference dequantization");
        assert!(
            hand_decoded
                .iter()
                .zip(&codec_decoded)
                .all(|(hand, codec)| hand.to_bits() == codec.to_bits()),
            "synthetic IQ4_NL encoder disagrees with llama.cpp"
        );

        let embed = offset_tensor(
            &ctx,
            30,
            &weight_bytes,
            19,
            vec![N_COLS as u64, N_VOCAB as u64],
            GgmlType::IQ4_NL,
        );
        let row_ids = [2i32, -1, 0, N_VOCAB as i32, 1, 1];
        let ids = offset_tensor(
            &ctx,
            12,
            bytemuck::cast_slice(&row_ids),
            13,
            vec![row_ids.len() as u64],
            GgmlType::I32,
        );
        let y = offset_tensor(
            &ctx,
            20,
            &vec![0u8; row_ids.len() * N_COLS * size_of::<f32>()],
            29,
            vec![(row_ids.len() * N_COLS) as u64],
            GgmlType::F32,
        );
        one_shot(&ctx, |enc| {
            encode_get_rows_f32(&ctx, enc, &embed, &ids, &y, row_ids.len(), N_COLS)
        })
        .expect("IQ4_NL get_rows");

        let actual = tensor_f32_at_offset(&y);
        for (lookup_row, &row_id) in row_ids.iter().enumerate() {
            let output_row = &actual[lookup_row * N_COLS..(lookup_row + 1) * N_COLS];
            if row_id >= 0 && (row_id as usize) < N_VOCAB {
                let expected =
                    &codec_decoded[row_id as usize * N_COLS..(row_id as usize + 1) * N_COLS];
                let max_abs = output_row
                    .iter()
                    .zip(expected)
                    .map(|(candidate, reference)| (candidate - reference).abs())
                    .fold(0.0f32, f32::max);
                assert!(max_abs <= 1e-6, "row {lookup_row} max_abs={max_abs}");
            } else {
                assert!(output_row.iter().all(|&value| value == 0.0));
            }
        }
        assert!(
            actual[4 * N_COLS..5 * N_COLS]
                .iter()
                .zip(&actual[5 * N_COLS..6 * N_COLS])
                .all(|(left, right)| left.to_bits() == right.to_bits())
        );
        assert_offset_guards(&embed, 30, 19);
        assert_offset_guards(&ids, 12, 13);
        assert_offset_guards(&y, 20, 29);

        let command = ctx.queue.commandBuffer().expect("validation command");
        let encoder = KernelEncoder::begin(&command);
        let malformed_embed = MetalTensor {
            shape: vec![31, N_VOCAB as u64],
            ..embed.clone()
        };
        let malformed_output = MetalTensor::zeros_f32(&ctx, vec![(row_ids.len() * 31) as u64])
            .expect("malformed output");
        assert!(
            encode_get_rows_f32(
                &ctx,
                &encoder,
                &malformed_embed,
                &ids,
                &malformed_output,
                row_ids.len(),
                31,
            )
            .is_err()
        );
        let mut misaligned_embed = embed.clone();
        misaligned_embed.offset += 1;
        assert!(
            encode_get_rows_f32(
                &ctx,
                &encoder,
                &misaligned_embed,
                &ids,
                &y,
                row_ids.len(),
                N_COLS,
            )
            .is_err()
        );
        encoder.end();
    }

    #[test]
    fn post_block_interventions_match_f32_cpu_oracles_at_realistic_hidden_size() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const H: usize = 6_656;
        let x_values: Vec<f32> = (0..H)
            .map(|i| ((i * 19 + 5) % 257) as f32 / 128.0 - 1.0)
            .collect();
        let direction_values: Vec<f32> = (0..H)
            .map(|i| ((i * 23 + 7) % 251) as f32 / 125.0 - 1.0)
            .collect();
        let source_values: Vec<f32> = (0..H)
            .map(|i| ((i * 29 + 11) % 241) as f32 / 120.0 - 1.0)
            .collect();
        let target_values: Vec<f32> = (0..H)
            .map(|i| ((i * 31 + 13) % 239) as f32 / 119.0 - 1.0)
            .collect();
        let direction = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&direction_values),
            vec![H as u64],
            GgmlType::F32,
        )
        .unwrap();
        let source = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&source_values),
            vec![H as u64],
            GgmlType::F32,
        )
        .unwrap();
        let target = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&target_values),
            vec![H as u64],
            GgmlType::F32,
        )
        .unwrap();
        let run = |intervention: PostBlockIntervention<'_>| {
            let x = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_values),
                vec![H as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_post_block_intervention_f32(&ctx, enc, &x, &intervention)
            })
            .unwrap();
            tensor_f32_at_offset(&x)
        };
        let coefficient = 0.375f32;
        let fixed = run(PostBlockIntervention::Fixed {
            layer: 0,
            direction: &direction,
            coefficient,
        });
        let residual_l2 = run(PostBlockIntervention::ResidualL2Relative {
            layer: 0,
            direction: &direction,
            coefficient,
        });
        let projection = run(PostBlockIntervention::Projection {
            layer: 0,
            direction: &direction,
            coefficient,
        });
        let source_to_target = run(PostBlockIntervention::SourceToTarget {
            layer: 0,
            source: &source,
            target: &target,
            coefficient,
        });
        let x_l2 = x_values
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let x_dot_direction: f32 = x_values
            .iter()
            .zip(&direction_values)
            .map(|(x, direction)| x * direction)
            .sum();
        let x_dot_source: f32 = x_values
            .iter()
            .zip(&source_values)
            .map(|(x, source)| x * source)
            .sum();
        let expected = [
            x_values
                .iter()
                .zip(&direction_values)
                .map(|(x, direction)| *x + coefficient * direction)
                .collect::<Vec<_>>(),
            x_values
                .iter()
                .zip(&direction_values)
                .map(|(x, direction)| *x + coefficient * x_l2 * direction)
                .collect::<Vec<_>>(),
            x_values
                .iter()
                .zip(&direction_values)
                .map(|(x, direction)| *x - coefficient * x_dot_direction * direction)
                .collect::<Vec<_>>(),
            x_values
                .iter()
                .zip(source_values.iter().zip(&target_values))
                .map(|(x, (source, target))| *x + coefficient * x_dot_source * (target - source))
                .collect::<Vec<_>>(),
        ];
        for (name, actual, expected) in [
            ("fixed", fixed, &expected[0]),
            ("residual-l2", residual_l2, &expected[1]),
            ("projection", projection, &expected[2]),
            ("source-to-target", source_to_target, &expected[3]),
        ] {
            let max_abs = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            let tolerance = if name == "residual-l2" { 1e-3 } else { 2e-5 };
            assert!(
                max_abs <= tolerance,
                "{name}: max|delta|={max_abs} exceeds F32 tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn elementwise_silu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let x: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let n = x.len();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let gpu = one_shot_f32_out(&ctx, n, |enc, y| encode_silu_f32(&ctx, enc, &x_t, y));
        for (i, &v) in x.iter().enumerate() {
            let expected = v / (1.0 + (-v).exp());
            assert!(
                (gpu[i] - expected).abs() < 1e-5,
                "silu[{i}] {v} -> {} vs {}",
                gpu[i],
                expected
            );
        }
    }

    #[test]
    fn elementwise_sigmoid_softplus() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let x: Vec<f32> = (-30..30).map(|i| i as f32 * 0.5).collect();
        let n = x.len();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();

        let sig = one_shot_f32_out(&ctx, n, |enc, y| encode_sigmoid_f32(&ctx, enc, &x_t, y));
        let sp = one_shot_f32_out(&ctx, n, |enc, y| encode_softplus_f32(&ctx, enc, &x_t, y));

        for (i, &v) in x.iter().enumerate() {
            let exp_sig = 1.0 / (1.0 + (-v).exp());
            assert!((sig[i] - exp_sig).abs() < 1e-5);
            let exp_sp = if v > 20.0 {
                v
            } else if v < -20.0 {
                v.exp()
            } else {
                (1.0 + v.exp()).ln()
            };
            assert!((sp[i] - exp_sp).abs() < 1e-5);
        }
    }

    /// Fused K+V scatter (one dispatch writes both caches) must produce
    /// identical bytes to the two-dispatch sequence.
    #[test]
    fn scatter_kv_fused_matches_unfused() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let kv_dim = 4 * 256; // n_kv_heads * head_dim for 27B
        let cap = 64usize;

        let k_src: Vec<f32> = (0..kv_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();
        let v_src: Vec<f32> = (0..kv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.07)
            .collect();
        let k_src_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&k_src),
            vec![kv_dim as u64],
            GgmlType::F32,
        )
        .unwrap();
        let v_src_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&v_src),
            vec![kv_dim as u64],
            GgmlType::F32,
        )
        .unwrap();

        for &dst_off_slot in &[0usize, 7, 31, 63] {
            let dst_off = dst_off_slot * kv_dim;

            // --- Reference: two unfused dispatches ---
            let k_ref = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_ref = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_f16(&ctx, enc, &k_src_t, &k_ref, dst_off, kv_dim)?;
                encode_scatter_offset_f32_to_f16(&ctx, enc, &v_src_t, &v_ref, dst_off, kv_dim)
            })
            .unwrap();

            // --- Fused: one dispatch ---
            let k_fused = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_fused = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_f16_kv(
                    &ctx, enc, &k_src_t, &v_src_t, &k_fused, &v_fused, dst_off, kv_dim,
                )
            })
            .unwrap();

            // Compare F16 bytes directly (must be byte-identical).
            let n_bytes = cap * kv_dim * 2;
            let k_ref_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(k_ref.buffer.contents().as_ptr() as *const u8, n_bytes)
            };
            let k_fused_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(k_fused.buffer.contents().as_ptr() as *const u8, n_bytes)
            };
            let v_ref_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(v_ref.buffer.contents().as_ptr() as *const u8, n_bytes)
            };
            let v_fused_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(v_fused.buffer.contents().as_ptr() as *const u8, n_bytes)
            };
            assert_eq!(
                k_ref_bytes, k_fused_bytes,
                "K mismatch at slot={dst_off_slot}"
            );
            assert_eq!(
                v_ref_bytes, v_fused_bytes,
                "V mismatch at slot={dst_off_slot}"
            );
            eprintln!("[scatter_kv_fused slot={dst_off_slot}] byte-identical to unfused");
        }
    }

    #[test]
    fn scatter_kv_q8_matches_ref_quant() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let kv_dim = 4 * 256usize;
        let cap = 8usize;
        let k_src: Vec<f32> = (0..kv_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.03125)
            .collect();
        let v_src: Vec<f32> = (0..kv_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.046875)
            .collect();
        let k_src_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&k_src),
            vec![kv_dim as u64],
            GgmlType::F32,
        )
        .unwrap();
        let v_src_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&v_src),
            vec![kv_dim as u64],
            GgmlType::F32,
        )
        .unwrap();

        for &dst_off_slot in &[0usize, 3, 7] {
            let dst_off = dst_off_slot * kv_dim;
            let k_q8 = MetalTensor::zeros_q8_0(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_q8 = MetalTensor::zeros_q8_0(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_q8_0_kv(
                    &ctx, enc, &k_src_t, &v_src_t, &k_q8, &v_q8, dst_off, kv_dim,
                )
            })
            .unwrap();

            let q8_block_bytes = 34usize;
            let blocks_per_row = kv_dim / 32;
            let total_bytes = cap * blocks_per_row * q8_block_bytes;
            let k_gpu: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    k_q8.buffer.contents().as_ptr() as *const u8,
                    total_bytes,
                )
            };
            let v_gpu: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    v_q8.buffer.contents().as_ptr() as *const u8,
                    total_bytes,
                )
            };

            let mut k_ref = vec![0u8; total_bytes];
            let mut v_ref = vec![0u8; total_bytes];
            let block_base = dst_off / 32;
            for (src, dst) in [(&k_src, &mut k_ref), (&v_src, &mut v_ref)] {
                for blk in 0..blocks_per_row {
                    let src_blk = &src[blk * 32..(blk + 1) * 32];
                    let amax = src_blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let d = amax / 127.0f32;
                    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                    let dst_blk = (block_base + blk) * q8_block_bytes;
                    let dh = half::f16::from_f32(d).to_bits().to_le_bytes();
                    dst[dst_blk..dst_blk + 2].copy_from_slice(&dh);
                    for j in 0..32 {
                        dst[dst_blk + 2 + j] = ((src_blk[j] * id).round() as i8) as u8;
                    }
                }
            }

            assert_eq!(
                k_gpu,
                k_ref.as_slice(),
                "Q8 K mismatch at slot={dst_off_slot}"
            );
            assert_eq!(
                v_gpu,
                v_ref.as_slice(),
                "Q8 V mismatch at slot={dst_off_slot}"
            );
            eprintln!("[scatter_kv_q8 slot={dst_off_slot}] byte-identical to ref quantization");
        }
    }

    #[test]
    fn elementwise_add_mul_silu_mul_sigmoid_mul() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let n = 17408usize; // FFN dim — exercise the realistic shape
        let a: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
        let a_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&a),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let b_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&b),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();

        let added = one_shot_f32_out(&ctx, n, |enc, y| encode_add_f32(&ctx, enc, &a_t, &b_t, y));
        for i in 0..n {
            assert!((added[i] - (a[i] + b[i])).abs() < 1e-5);
        }
        let muled = one_shot_f32_out(&ctx, n, |enc, y| encode_mul_f32(&ctx, enc, &a_t, &b_t, y));
        for i in 0..n {
            assert!((muled[i] - (a[i] * b[i])).abs() < 1e-5);
        }
        let silumul = one_shot_f32_out(&ctx, n, |enc, y| {
            encode_silu_mul_f32(&ctx, enc, &a_t, &b_t, y)
        });
        for i in 0..n {
            let silu_a = a[i] / (1.0 + (-a[i]).exp());
            assert!(
                (silumul[i] - silu_a * b[i]).abs() < 1e-5,
                "silu_mul[{i}] = {} vs {}",
                silumul[i],
                silu_a * b[i]
            );
        }

        let sigmul = one_shot_f32_out(&ctx, n, |enc, y| {
            encode_sigmoid_mul_f32(&ctx, enc, &a_t, &b_t, y)
        });
        for i in 0..n {
            let sig_a = 1.0 / (1.0 + (-a[i]).exp());
            assert!(
                (sigmul[i] - sig_a * b[i]).abs() < 1e-5,
                "sigmoid_mul[{i}] = {} vs {}",
                sigmul[i],
                sig_a * b[i]
            );
        }
    }

    #[test]
    fn elementwise_add_inplace() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let n = 5120usize;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-3).collect();
        let b: Vec<f32> = (0..n).map(|i| -(i as f32) * 2e-3).collect();
        let a_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&a),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let b_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&b),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        one_shot(&ctx, |enc| encode_add_inplace_f32(&ctx, enc, &a_t, &b_t)).unwrap();
        let result = read_back_f32(&a_t.buffer, n);
        for i in 0..n {
            assert!((result[i] - (a[i] + b[i])).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_matches_cpu() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        // Cover small (one warp) to large (multi-warp reduce) shapes.
        for &n in &[16usize, 256, 4096, 32768] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| encode_softmax_inplace_f32(&ctx, enc, &x_t)).unwrap();
            let gpu = read_back_f32(&x_t.buffer, n);

            // CPU reference.
            let mut cpu = x.clone();
            let m = cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0f32;
            for v in cpu.iter_mut() {
                *v = (*v - m).exp();
                s += *v;
            }
            for v in cpu.iter_mut() {
                *v /= s;
            }
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let total: f32 = gpu.iter().sum();
            assert!((total - 1.0).abs() < 1e-4, "softmax sum n={n}: {total}");
            assert!(max_abs < 1e-5, "softmax n={n} max|Δ|={max_abs}");
        }
    }

    #[test]
    fn get_rows_flat_f32_matches_cpu_and_guards_ids() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let vocab = 100usize;
        let n_cols = 64usize;
        let embed: Vec<f32> = (0..vocab * n_cols).map(|i| i as f32 * 0.001).collect();
        let ids: Vec<i32> = vec![3, 17, 42, 99, -1, vocab as i32];
        let n_rows = ids.len();

        let embed_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&embed),
            vec![(n_cols * vocab) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let ids_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![n_rows as u64],
            GgmlType::I32,
        )
        .unwrap();
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_rows as u64, n_cols as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_get_rows_f32(&ctx, enc, &embed_t, &ids_t, &y_t, n_rows, n_cols)
        })
        .unwrap();
        let gpu = read_back_f32(&y_t.buffer, n_rows * n_cols);
        for r in 0..n_rows {
            for i in 0..n_cols {
                let expected = if ids[r] < 0 || ids[r] as usize >= vocab {
                    0.0
                } else {
                    embed[ids[r] as usize * n_cols + i]
                };
                let got = gpu[r * n_cols + i];
                assert!(
                    (got - expected).abs() < 1e-7,
                    "get_rows row={r} ids={} col={i}: {got} vs {expected}",
                    ids[r]
                );
            }
        }
    }

    #[test]
    #[ignore = "requires local 27B Q4_K fixture"]
    fn get_rows_q4_k_matches_selected_cpu_rows() {
        quantized_get_rows_fixture(
            crate::test_fixtures::QWEN36_27B_Q4_K_M.path(),
            GgmlType::Q4_K,
            true,
        );
    }

    #[test]
    #[ignore = "requires local A3B Q8_0 embedding fixture"]
    fn get_rows_q8_0_matches_selected_cpu_rows() {
        quantized_get_rows_fixture(
            "/Users/tito/models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            GgmlType::Q8_0,
            true,
        );
    }

    #[test]
    #[ignore = "requires local Ridge 27B Q6_K embedding fixture"]
    fn get_rows_q6_k_matches_selected_cpu_rows() {
        quantized_get_rows_fixture(
            "/Users/tito/models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf",
            GgmlType::Q6_K,
            true,
        );
    }

    /// v0.432 equivalence gate: the fused strided gate epilogue
    /// (`out = x / (1 + e^-gate)`) vs the old split + sigmoid-into-temp +
    /// mul (`out = x * (1 / (1 + e^-gate))`). Different last-ulp rounding
    /// (division vs reciprocal-multiply), so tolerance-based, tight.
    #[test]
    fn sigmoid_mul_gate_strided_matches_split_path() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let (n_heads, head_dim) = (24usize, 256usize);
        let full: Vec<f32> = (0..n_heads * 2 * head_dim)
            .map(|i| ((i % 37) as f32 - 18.0) * 5e-2)
            .collect();
        let x: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 4e-2)
            .collect();
        let full_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&full),
            vec![(n_heads * 2 * head_dim) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_heads * head_dim) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let q_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let gate_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let sig_t = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let y_split = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let y_fused = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_split_q_gate_f32(&ctx, enc, &full_t, &q_t, &gate_t, n_heads, head_dim)?;
            encode_sigmoid_f32(&ctx, enc, &gate_t, &sig_t)?;
            encode_mul_f32(&ctx, enc, &x_t, &sig_t, &y_split)
        })
        .unwrap();
        one_shot(&ctx, |enc| {
            encode_sigmoid_mul_gate_strided_f32(
                &ctx,
                enc,
                &full_t,
                &x_t,
                &y_fused,
                n_heads,
                head_dim,
                2 * head_dim,
                head_dim,
            )
        })
        .unwrap();
        let a = read_back_f32(&y_split.buffer, n_heads * head_dim);
        let b = read_back_f32(&y_fused.buffer, n_heads * head_dim);
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 1e-6,
            "fused strided gate epilogue diverged beyond ulp scale: max|Δ|={max_abs:.3e}"
        );
    }

    #[test]
    fn argmax_rejects_f32_output_metadata() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let input = MetalTensor::zeros_f32(&ctx, vec![2]).unwrap();
        let wrong_output = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd);
        assert!(encode_argmax_f32(&ctx, &enc, &input, &wrong_output, 1, 2).is_err());
        assert!(encode_argmax_f32_greedy(&ctx, &enc, &input, &wrong_output, 1, 2).is_err());
        enc.end();
    }

    #[test]
    fn greedy_argmax_matches_sampler_total_order_contract() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        fn run(ctx: &MetalContext, x: &[f32], n_rows: usize, n: usize) -> Vec<i32> {
            assert_eq!(x.len(), n_rows * n);
            let xt = MetalTensor {
                buffer: ctx.buffer_from(x).expect("input buffer"),
                offset: 0,
                shape: vec![n_rows as u64, n as u64],
                dtype: crate::tensor::GgmlType::F32,
                provenance: MetalTensorProvenance::OwnedWritable,
            };
            let ot = MetalTensor::zeros_i32(ctx, vec![n_rows as u64]).expect("output buffer");
            let cmd = ctx.queue.commandBuffer().expect("command buffer");
            let enc = KernelEncoder::begin(&cmd);
            encode_argmax_f32_greedy(ctx, &enc, &xt, &ot, n_rows, n).expect("encode greedy argmax");
            enc.end();
            cmd.commit();
            wait_completed(&cmd).expect("Metal command buffer failed");
            unsafe {
                let ptr = ot.buffer.contents().as_ptr() as *const i32;
                (0..n_rows).map(|row| *ptr.add(row)).collect()
            }
        }

        fn cpu(row: &[f32]) -> i32 {
            if let Some(token) = row.iter().position(|value| value.is_nan()) {
                return !(token as i32);
            }
            let mut best = 0usize;
            for token in 1..row.len() {
                if row[token].total_cmp(&row[best]) == std::cmp::Ordering::Greater {
                    best = token;
                }
            }
            best as i32
        }

        let rows = [
            [1.0, 4.0, 2.0, 3.0, -1.0, -2.0, -3.0, -4.0],
            [5.0, 1.0, 5.0, 0.0, 5.0, 2.0, 5.0, 3.0],
            [-0.0, 0.0, -0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
            [
                f32::INFINITY,
                1.0,
                f32::INFINITY,
                0.0,
                -1.0,
                -2.0,
                -3.0,
                -4.0,
            ],
            [f32::NEG_INFINITY; 8],
            [0.0, f32::NAN, 2.0, f32::NAN, 4.0, 5.0, 6.0, 7.0],
            [f32::NAN; 8],
            [
                f32::from_bits(1),
                -f32::from_bits(1),
                0.0,
                -0.0,
                1.0,
                -1.0,
                2.0,
                -2.0,
            ],
        ];
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        let got = run(&ctx, &flat, rows.len(), rows[0].len());
        let expected: Vec<i32> = flat.chunks(rows[0].len()).map(cpu).collect();
        assert_eq!(got, expected);
        assert_eq!(got[1], 0, "finite ties choose the lowest token id");
        assert_eq!(got[2], 1, "+0 outranks -0 under total_cmp");
        assert_eq!(got[4], 0, "equal -inf chooses the lowest token id");
        assert_eq!(got[5], !1, "lowest NaN token is encoded");
        assert_eq!(got[6], !0, "all-NaN row reports token zero");

        for n in [1usize, 31, 32, 33, 1023, 1025] {
            let mut row = vec![-1.0f32; n];
            row[n / 2] = 3.0;
            row[n - 1] = 3.0;
            assert_eq!(
                run(&ctx, &row, 1, n),
                vec![(n / 2) as i32],
                "boundary row length {n}"
            );
        }

        let mut wide = vec![0.0f32; 4096];
        wide[100] = 9.0;
        wide[2500] = 9.0;
        wide[3999] = 9.0;
        assert_eq!(run(&ctx, &wide, 1, wide.len()), vec![100]);
        wide[2500] = f32::NAN;
        wide[100] = f32::NAN;
        assert_eq!(run(&ctx, &wide, 1, wide.len()), vec![!100]);

        let mut vocab = vec![0.0f32; 248_320];
        let mut state = 0xc0ffeeu32;
        for value in &mut vocab {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            *value = (state as i32) as f32 * 1e-9;
        }
        assert_eq!(run(&ctx, &vocab, 1, vocab.len()), vec![cpu(&vocab)]);
    }

    /// H5.3a GPU argmax — bit-exact match to CPU argmax with lowest-index
    /// tie-breaking, including the explicit edge cases:
    /// * tie at row start (idx 0 wins)
    /// * tie at row end
    /// * single-element row
    /// * row larger than 1024 (tests cross-simdgroup reduce path)
    /// * vocab-sized row (V=248320; the actual production shape)
    /// * negative-infinity entries (production lm_head won't have these,
    ///   but defensive)
    #[test]
    fn argmax_matches_cpu_with_tie_to_lowest_index() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        // Helper: encode-only argmax + readback for tests.
        fn run(ctx: &MetalContext, x: &[f32], n_rows: usize, n: usize) -> Vec<i32> {
            assert_eq!(x.len(), n_rows * n);
            let xb = ctx.buffer_from(x).expect("xb");
            let xt = MetalTensor {
                buffer: xb,
                offset: 0,
                shape: vec![n_rows as u64, n as u64],
                dtype: crate::tensor::GgmlType::F32,
                provenance: MetalTensorProvenance::OwnedWritable,
            };
            let ot = MetalTensor::zeros_i32(ctx, vec![n_rows as u64]).expect("ob");
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_argmax_f32(ctx, &enc, &xt, &ot, n_rows, n).expect("encode argmax");
            enc.end();
            cmd.commit();
            wait_completed(&cmd).expect("Metal command buffer failed");
            unsafe {
                let p = ot.buffer.contents().as_ptr() as *const i32;
                (0..n_rows).map(|i| *p.add(i)).collect()
            }
        }

        fn cpu_argmax_lowest_idx(row: &[f32]) -> i32 {
            let mut best = f32::NEG_INFINITY;
            let mut idx: i32 = 0;
            for (i, &v) in row.iter().enumerate() {
                if v > best {
                    best = v;
                    idx = i as i32;
                }
            }
            idx
        }

        // 1. Single-element row.
        {
            let x = vec![3.5f32];
            let got = run(&ctx, &x, 1, 1);
            assert_eq!(got, vec![0]);
        }

        // 2. Tie at row start: x = [5.0, 1.0, 5.0, 5.0, 0.0]. Lowest idx
        //    among matches = 0.
        {
            let x = vec![5.0f32, 1.0, 5.0, 5.0, 0.0];
            let got = run(&ctx, &x, 1, 5);
            assert_eq!(got, vec![0], "tie at start should pick idx 0");
        }

        // 3. Tie at row end (max only at the last position).
        {
            let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            let got = run(&ctx, &x, 1, 5);
            assert_eq!(got, vec![4]);
        }

        // 4. Tie spread across the row at multiple distant positions.
        {
            let mut x = vec![0.0f32; 4096];
            x[100] = 9.0;
            x[2500] = 9.0;
            x[3999] = 9.0;
            let got = run(&ctx, &x, 1, 4096);
            assert_eq!(got, vec![100], "spread tie should pick lowest idx");
        }

        // 5. Multi-row batch: argmax independently per row.
        {
            let n = 1024;
            let n_rows = 5;
            let mut x = vec![0.0f32; n_rows * n];
            for r in 0..n_rows {
                // Place the max for row r at idx (r * 137) % n.
                let idx = (r * 137) % n;
                x[r * n + idx] = 1.0 + (r as f32) * 0.1;
            }
            let got = run(&ctx, &x, n_rows, n);
            for r in 0..n_rows {
                let want = cpu_argmax_lowest_idx(&x[r * n..(r + 1) * n]);
                assert_eq!(got[r], want, "row {r}");
            }
        }

        // 6. Random fuzz at vocab-sized row (the production shape).
        {
            let n = 248_320usize;
            let n_rows = 16;
            let mut x = vec![0.0f32; n_rows * n];
            // Deterministic pseudo-random fill.
            let mut s: u32 = 0xc0ffeeu32;
            for v in x.iter_mut() {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
                *v = (s as i32) as f32 * 1e-9;
            }
            let got = run(&ctx, &x, n_rows, n);
            for r in 0..n_rows {
                let want = cpu_argmax_lowest_idx(&x[r * n..(r + 1) * n]);
                assert_eq!(got[r], want, "vocab row {r}");
            }
        }

        // 7. Negative-infinity entries (defensive — production lm_head
        //    won't produce these but the kernel must not get confused).
        {
            let mut x = vec![f32::NEG_INFINITY; 1024];
            x[42] = -1e9;
            x[500] = -1e10; // smaller than 42's value
            let got = run(&ctx, &x, 1, 1024);
            assert_eq!(got, vec![42]);
        }

        // 8. All-zero row (every position ties); lowest index = 0.
        {
            let x = vec![0.0f32; 1024];
            let got = run(&ctx, &x, 1, 1024);
            assert_eq!(got, vec![0], "all-tie should pick idx 0");
        }

        // 9. All -INFINITY row (degenerate but well-defined): every
        //    position ties at -inf, lowest idx wins. Per codex H5.3a
        //    review: this returns idx 0, NOT -1. Production lm_head
        //    cannot produce all -inf, but documenting the contract.
        {
            let x = vec![f32::NEG_INFINITY; 1024];
            let got = run(&ctx, &x, 1, 1024);
            assert_eq!(
                got,
                vec![0],
                "all -INFINITY: kernel ties at -inf, lowest idx 0 wins"
            );
        }

        // 10. All NaN row: IEEE comparison `a > b` is FALSE for any
        //     NaN operand, so the per-lane scan never updates from
        //     `best_val=-INF, best_idx=UINT_MAX`. simd_max also returns
        //     NaN; (NaN == NaN) is false, so lane_idx stays UINT_MAX
        //     for every lane, simd_min(UINT_MAX) = UINT_MAX, cast to
        //     i32 = -1.
        //
        //     Production lm_head does not produce NaN under correct
        //     numerics. Treat this as a "this kernel returns -1
        //     deterministically when the entire row is unranked";
        //     callers should not feed it NaN rows.
        {
            let x = vec![f32::NAN; 1024];
            let got = run(&ctx, &x, 1, 1024);
            assert_eq!(
                got,
                vec![-1],
                "all-NaN: kernel returns -1 (UINT_MAX cast); document only — production should never see this"
            );
        }
    }

    #[test]
    fn argmax_top2_matches_lowest_index_and_reports_exact_gap() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        fn run(ctx: &MetalContext, x: &[f32], n_rows: usize, n: usize) -> (Vec<i32>, Vec<f32>) {
            assert_eq!(x.len(), n_rows * n);
            let xb = ctx.buffer_from(x).expect("xb");
            let xt = MetalTensor {
                buffer: xb,
                offset: 0,
                shape: vec![n_rows as u64, n as u64],
                dtype: crate::tensor::GgmlType::F32,
                provenance: MetalTensorProvenance::OwnedWritable,
            };
            let ot = MetalTensor::zeros_i32(ctx, vec![n_rows as u64]).expect("ob");
            let gt = MetalTensor::zeros_f32(ctx, vec![n_rows as u64]).expect("gb");
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_argmax_top2_f32(ctx, &enc, &xt, &ot, &gt, n_rows, n)
                .expect("encode argmax top2");
            enc.end();
            cmd.commit();
            wait_completed(&cmd).expect("Metal command buffer failed");
            unsafe {
                let pi = ot.buffer.contents().as_ptr() as *const i32;
                let pg = gt.buffer.contents().as_ptr() as *const f32;
                let idx = (0..n_rows).map(|i| *pi.add(i)).collect();
                let gap = (0..n_rows).map(|i| *pg.add(i)).collect();
                (idx, gap)
            }
        }

        fn cpu_top2(row: &[f32]) -> (i32, f32) {
            let mut v1 = f32::NEG_INFINITY;
            let mut i1 = 0usize;
            let mut v2 = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > v1 {
                    v2 = v1;
                    v1 = v;
                    i1 = i;
                } else if v > v2 {
                    v2 = v;
                }
            }
            (i1 as i32, v1 - v2)
        }

        // Distinct max, duplicate max (gap 0), single element (+inf),
        // all-equal (gap 0), negative max, signed zero, boundary widths.
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 4.0, 2.0, 3.0, -1.0, -2.0, -3.0, -4.0],
            vec![5.0, 1.0, 5.0, 0.0, 5.0, 2.0, 5.0, 3.0],
            vec![7.0],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![-8.0, -9.0, -7.0, -9.5],
            vec![-0.0, 0.0, -0.0, -1.0],
        ];
        let n_rows = rows.len();
        let n = 8;
        let mut flat = Vec::with_capacity(n_rows * n);
        let mut expected = Vec::with_capacity(n_rows);
        for row in &rows {
            assert!(row.len() <= n);
            let mut padded = row.clone();
            padded.resize(n, f32::NEG_INFINITY);
            flat.extend_from_slice(&padded);
            expected.push(cpu_top2(&padded));
        }
        let (idx, gap) = run(&ctx, &flat, n_rows, n);
        for (r, &(ei, eg)) in expected.iter().enumerate() {
            assert_eq!(idx[r], ei, "row {r} idx");
            if eg.is_finite() {
                assert_eq!(gap[r], eg, "row {r} gap");
            } else {
                assert!(gap[r].is_infinite() && gap[r] > 0.0, "row {r} +inf gap");
            }
        }

        // Boundary widths with a duplicated max: lowest index wins, gap 0
        // (width 1 collapses to a single element: gap +inf).
        for width in [1usize, 31, 32, 33, 1023, 1025] {
            let mut row = vec![-1.0f32; width];
            row[width / 2] = 3.0;
            row[width - 1] = 3.0;
            let (idx, gap) = run(&ctx, &row, 1, width);
            assert_eq!(idx, vec![(width / 2) as i32], "boundary width {width} idx");
            if width == 1 {
                assert!(gap[0].is_infinite() && gap[0] > 0.0, "boundary width 1 gap");
            } else {
                assert_eq!(gap, vec![0.0], "boundary width {width} gap");
            }
        }
    }
}
