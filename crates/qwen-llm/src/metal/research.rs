//! Research-only probes: roofline and touch-bytes kernels (bench-only).

use super::*;

pub fn encode_touch_bytes_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    sink: &MetalTensor,
    stride_bytes: usize,
) -> Result<(), MetalError> {
    let sink_n = sink.n_elements() as usize;
    if sink_n == 0 {
        return Err(MetalError::BadShape {
            kernel: "touch_bytes",
            detail: "sink must have at least one element".into(),
        });
    }
    let n_bytes = src.n_bytes() as usize;
    if n_bytes == 0 || stride_bytes == 0 {
        return Err(MetalError::BadShape {
            kernel: "touch_bytes",
            detail: format!("n_bytes={n_bytes} stride_bytes={stride_bytes} must both be > 0"),
        });
    }
    let n_steps = n_bytes.div_ceil(stride_bytes);
    let pso = ctx.pipeline("kernel_touch_bytes_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_steps: u32,
        stride_bytes: u32,
        n_bytes: u64,
    }
    enc.set_bytes(
        0,
        &Args {
            n_steps: n_steps as u32,
            stride_bytes: stride_bytes as u32,
            n_bytes: n_bytes as u64,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, sink);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: sink_n.min(256),
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_roofline_stream_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
    alpha: f32,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if y.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "roofline_stream",
            detail: format!("y.n_elements={} != x.n_elements={n}", y.n_elements()),
        });
    }
    let pso = ctx.pipeline("kernel_roofline_stream_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        alpha: f32,
    }
    enc.set_bytes(0, &Args { n: n as u32, alpha });
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);
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

pub fn encode_roofline_fma_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
    iters: usize,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if y.n_elements() as usize != n || iters == 0 {
        return Err(MetalError::BadShape {
            kernel: "roofline_fma",
            detail: format!(
                "y.n_elements={} x.n_elements={n} iters={iters}",
                y.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_roofline_fma_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        iters: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            iters: iters as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);
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

fn glm_q6_solo_args(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    name: &str,
    n_in: usize,
    n_out: usize,
    clamp: f32,
    nsg: usize,
) -> Result<(), MetalError> {
    if !matches!(nsg, 2 | 4 | 8) {
        return Err(MetalError::BadShape {
            kernel: "glm_q6_solo",
            detail: format!("nsg={nsg} must be 2, 4 or 8"),
        });
    }
    let pso = ctx.pipeline(name)?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        clamp: f32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            clamp,
        },
    );
    Ok(())
}

/// Bench-only GLM Q6 split-K GEMV: one threadgroup per output row, K split
/// across `nsg` simdgroups. Fast-lineage only (different summation order).
pub fn encode_glm_q6_splitk_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    nsg: usize,
) -> Result<(), MetalError> {
    if weight.dtype != crate::tensor::GgmlType::Q6_K
        || weight.shape != [n_in as u64, n_out as u64]
        || x.dtype != crate::tensor::GgmlType::F32
        || x.n_elements() as usize != n_in
        || y.n_elements() as usize != n_out
        || !n_in.is_multiple_of(256)
    {
        return Err(MetalError::BadShape {
            kernel: "glm_q6_splitk",
            detail: format!(
                "weight {:?} x {:?} y {} n_in={n_in} n_out={n_out} nsg={nsg}",
                weight.shape,
                x.shape,
                y.n_elements()
            ),
        });
    }
    let name = format!("glm_q6_splitk_s{nsg}");
    glm_q6_solo_args(ctx, enc, &name, n_in, n_out, 10.0, nsg)?;
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.dispatch(
        MTLSize {
            width: n_out,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: nsg * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Bench-only fused same-input Q6 gate/up with clamped SwiGLU: one threadgroup
/// per output row, K split across `nsg` simdgroups.
#[allow(clippy::too_many_arguments)]
pub fn encode_glm_q6_pair_swiglu_splitk_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    nsg: usize,
    clamp: f32,
) -> Result<(), MetalError> {
    if gate.dtype != crate::tensor::GgmlType::Q6_K
        || up.dtype != crate::tensor::GgmlType::Q6_K
        || gate.shape != [n_in as u64, n_out as u64]
        || up.shape != [n_in as u64, n_out as u64]
        || x.dtype != crate::tensor::GgmlType::F32
        || x.n_elements() as usize != n_in
        || (y.n_elements() as usize) < n_out
        || !n_in.is_multiple_of(256)
    {
        return Err(MetalError::BadShape {
            kernel: "glm_q6_pair_swiglu_splitk",
            detail: format!(
                "gate {:?} up {:?} x {} y {} n_in={n_in} n_out={n_out} nsg={nsg}",
                gate.shape,
                up.shape,
                x.n_elements(),
                y.n_elements()
            ),
        });
    }
    let name = format!("glm_q6_pair_splitk_s{nsg}_swiglu1");
    glm_q6_solo_args(ctx, enc, &name, n_in, n_out, clamp, nsg)?;
    enc.set_tensor(1, gate);
    enc.set_tensor(2, up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, y);
    // The SwiGLU form leaves buffer 5 unused; bind the same output.
    enc.set_tensor(5, y);
    enc.dispatch(
        MTLSize {
            width: n_out,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: nsg * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Bench-only fused same-input Q6 triple GEMV (GLM KDA q/k/v): one threadgroup
/// per output row, K split across `nsg` simdgroups.
#[allow(clippy::too_many_arguments)]
pub fn encode_glm_q6_triple_splitk_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: [&MetalTensor; 3],
    x: &MetalTensor,
    ys: [&MetalTensor; 3],
    n_in: usize,
    n_out: usize,
    nsg: usize,
) -> Result<(), MetalError> {
    for weight in weights {
        if weight.dtype != crate::tensor::GgmlType::Q6_K
            || weight.shape != [n_in as u64, n_out as u64]
        {
            return Err(MetalError::BadShape {
                kernel: "glm_q6_triple_splitk",
                detail: format!(
                    "weight {:?} n_in={n_in} n_out={n_out} nsg={nsg}",
                    weight.shape
                ),
            });
        }
    }
    if x.dtype != crate::tensor::GgmlType::F32
        || x.n_elements() as usize != n_in
        || ys.iter().any(|y| y.n_elements() as usize != n_out)
        || !n_in.is_multiple_of(256)
    {
        return Err(MetalError::BadShape {
            kernel: "glm_q6_triple_splitk",
            detail: format!(
                "x {} ys {:?} n_in={n_in} n_out={n_out} nsg={nsg}",
                x.n_elements(),
                ys.iter().map(|y| y.n_elements()).collect::<Vec<_>>()
            ),
        });
    }
    let name = format!("glm_q6_triple_splitk_s{nsg}");
    glm_q6_solo_args(ctx, enc, &name, n_in, n_out, 10.0, nsg)?;
    enc.set_tensor(1, weights[0]);
    enc.set_tensor(2, weights[1]);
    enc.set_tensor(3, weights[2]);
    enc.set_tensor(4, x);
    enc.set_tensor(5, ys[0]);
    enc.set_tensor(6, ys[1]);
    enc.set_tensor(7, ys[2]);
    enc.dispatch(
        MTLSize {
            width: n_out,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: nsg * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
