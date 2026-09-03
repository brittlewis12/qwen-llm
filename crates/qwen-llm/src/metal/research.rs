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
