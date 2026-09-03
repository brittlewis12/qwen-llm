//! Prompt GEMM dispatchers across quant formats and mma8 variants.

use super::*;

pub(crate) fn encode_mat_mat_mxfp4_f32_mm64x32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_batch: usize,
) -> Result<(), MetalError> {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 17;
    const KERNEL: &str = "mat_mat_mxfp4_f32_mm64x32";
    if n_in == 0 || n_out == 0 || n_batch == 0 || !n_in.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "n_in={n_in} must be nonzero and divisible by {BLOCK_ELEMENTS}; n_out={n_out} and n_batch={n_batch} must be nonzero"
            ),
        });
    }
    let n_in_u32 = u32::try_from(n_in).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("n_in={n_in} exceeds u32"),
    })?;
    let n_out_u32 = u32::try_from(n_out).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("n_out={n_out} exceeds u32"),
    })?;
    let n_batch_u32 = u32::try_from(n_batch).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("n_batch={n_batch} exceeds u32"),
    })?;
    let row_bytes = n_in
        .checked_div(BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(BLOCK_BYTES))
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "MXFP4 row-byte calculation overflow".into(),
        })?;
    let row_bytes_u32 = u32::try_from(row_bytes).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("MXFP4 row bytes {row_bytes} exceeds u32"),
    })?;
    if weight.dtype != GgmlType::MXFP4 || x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "weight/x/y expected MXFP4/F32/F32, got {:?}/{:?}/{:?}",
                weight.dtype, x.dtype, y.dtype
            ),
        });
    }
    if weight.shape.as_slice() != [n_in as u64, n_out as u64]
        || x.shape.as_slice() != [n_in as u64, n_batch as u64]
        || y.shape.as_slice() != [n_out as u64, n_batch as u64]
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "weight/x/y expected shapes [{n_in},{n_out}]/[{n_in},{n_batch}]/[{n_out},{n_batch}], got {:?}/{:?}/{:?}",
                weight.shape, x.shape, y.shape
            ),
        });
    }
    if !x.offset.is_multiple_of(std::mem::align_of::<f32>() as u64)
        || !y.offset.is_multiple_of(std::mem::align_of::<f32>() as u64)
        || !y.is_writable()
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "x/y must be aligned F32 and y writable, got offsets={}/{} y_provenance={:?}",
                x.offset,
                y.offset,
                y.provenance()
            ),
        });
    }
    for (label, tensor) in [("weight", weight), ("x", x), ("y", y)] {
        let end =
            tensor
                .offset
                .checked_add(tensor.n_bytes())
                .ok_or_else(|| MetalError::BadShape {
                    kernel: KERNEL,
                    detail: format!("{label} buffer range overflow"),
                })?;
        if end > tensor.buffer.length() as u64 {
            return Err(MetalError::BadShape {
                kernel: KERNEL,
                detail: format!(
                    "{label} range offset={} bytes={} exceeds buffer={}",
                    tensor.offset,
                    tensor.n_bytes(),
                    tensor.buffer.length()
                ),
            });
        }
    }

    let pso = ctx.pipeline("kernel_mat_mat_mxfp4_f32_mm64x32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out_u32,
            n: n_batch_u32,
            k: n_in_u32,
            nb01: row_bytes_u32,
            stride_b: n_in_u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 12 * 1024);
    enc.dispatch(
        MTLSize {
            width: n_batch.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_mat_mat_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32",
            detail: format!("weight.dtype = {:?}, expected F32", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32",
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_out * n_query {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f32",
            detail: format!(
                "y.n_elements={} != n_out*n_query={}",
                y.n_elements(),
                n_out * n_query
            ),
        });
    }
    let pso = ctx.pipeline("kernel_mat_mat_f32_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_query: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_query: n_query as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 32 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out,
            height: n_query.div_ceil(32),
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

pub(crate) fn encode_mat_mat_16bit_weight_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    expected: GgmlType,
    kernel_name: &'static str,
) -> Result<(), MetalError> {
    if weight.dtype != expected {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("weight.dtype = {:?}, expected {expected:?}", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_out * n_query {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "y.n_elements={} != n_out*n_query={}",
                y.n_elements(),
                n_out * n_query
            ),
        });
    }
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_query: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_query: n_query as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 32 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out,
            height: n_query.div_ceil(32),
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

pub fn encode_mat_mat_f16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if mat_mat_f16_half_act_enabled() && n_in.is_multiple_of(32) && n_query >= 16 {
        return encode_mat_mat_f16_half_act_f32(ctx, enc, weight, x, y, n_in, n_out, n_query);
    }
    encode_mat_mat_16bit_weight_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::F16,
        "kernel_mat_mat_f16_f32",
    )
}

crate::env_flag!(default_on matmat_f16_half_act_env_default, "QWEN_MATMAT_F16_HALF_ACT");

pub(crate) fn mat_mat_f16_half_act_enabled() -> bool {
    if let Some(enabled) = MATMAT_F16_HALF_ACT_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    matmat_f16_half_act_env_default()
}

pub fn encode_mat_mat_f16_half_act_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f16_half_act_f32",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f16_half_act_f32",
            detail: format!("weight.dtype = {:?}, expected F16", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f16_half_act_f32",
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in || y.n_elements() as usize != n_out * n_query {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_f16_half_act_f32",
            detail: format!(
                "shape mismatch x={} y={} expected x={} y={}",
                x.n_elements(),
                y.n_elements(),
                n_query * n_in,
                n_out * n_query
            ),
        });
    }

    let pso = ctx.pipeline("kernel_mat_mat_f16_half_act_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_query: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_query: n_query as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_mat_mat_bf16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_16bit_weight_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::BF16,
        "kernel_mat_mat_bf16_f32",
    )
}

pub fn encode_mat_mat_bf16_bfloat_act_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_bf16_bfloat_act_f32",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::BF16 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_bf16_bfloat_act_f32",
            detail: format!("weight.dtype = {:?}, expected BF16", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_bf16_bfloat_act_f32",
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in || y.n_elements() as usize != n_out * n_query {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_bf16_bfloat_act_f32",
            detail: format!(
                "shape mismatch x={} y={} expected x={} y={}",
                x.n_elements(),
                y.n_elements(),
                n_query * n_in,
                n_out * n_query
            ),
        });
    }

    let pso = ctx.pipeline("kernel_mat_mat_bf16_bfloat_act_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_query: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_query: n_query as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(crate) fn encode_mat_mat_block32_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    expected: GgmlType,
    kernel_name: &'static str,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    encode_mat_mat_16bit_weight_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        expected,
        kernel_name,
    )
}

pub fn encode_mat_mat_q4_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if mat_mat_q4_legacy_mm_enabled() && n_query >= 16 {
        return encode_mat_mat_q4_legacy_mm_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            n_query,
            GgmlType::Q4_0,
            "kernel_mat_mat_q4_0_f32_mm",
            18,
        );
    }
    encode_mat_mat_block32_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::Q4_0,
        "kernel_mat_mat_q4_0_f32",
    )
}

pub fn encode_mat_mat_q4_1_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if mat_mat_q4_legacy_mm_enabled() && n_query >= 16 {
        return encode_mat_mat_q4_legacy_mm_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            n_query,
            GgmlType::Q4_1,
            "kernel_mat_mat_q4_1_f32_mm",
            20,
        );
    }
    encode_mat_mat_block32_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::Q4_1,
        "kernel_mat_mat_q4_1_f32",
    )
}

crate::env_flag!(default_on matmat_q4_legacy_mm_env_default, "QWEN_MATMAT_Q4_LEGACY_MM");

pub(crate) fn mat_mat_q4_legacy_mm_enabled() -> bool {
    if let Some(enabled) = MATMAT_Q4_LEGACY_MM_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    matmat_q4_legacy_mm_env_default()
}

pub(crate) fn encode_mat_mat_q4_legacy_mm_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    expected: GgmlType,
    kernel_name: &'static str,
    block_bytes: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != expected {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("weight.dtype = {:?}, expected {expected:?}", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01: ((n_in / 32) * block_bytes) as u32,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, mat_mat_qk_threadgroup_memory(n_out, n_query, 32));
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

crate::env_flag!(default_on matmat_iq4_nl_mm_enabled, "QWEN_MATMAT_IQ4_NL_MM");

pub fn encode_mat_mat_iq4_nl_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_iq4_nl_mm_enabled() {
        return encode_mat_mat_iq4_nl_f32_mm(ctx, enc, weight, x, y, n_in, n_out, n_query);
    }
    encode_mat_mat_block32_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::IQ4_NL,
        "kernel_mat_mat_iq4_nl_f32",
    )
}

pub(crate) fn encode_mat_mat_iq4_nl_f32_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_nl_mm",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::IQ4_NL {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_nl_mm",
            detail: format!("weight.dtype = {:?}, expected IQ4_NL", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_nl_mm",
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_nl_mm",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_nl_mm",
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let nb01 = ((n_in / 32) * 18) as u32;
    let pso = ctx.pipeline("kernel_mat_mat_iq4_nl_f32_mm")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(crate) fn encode_mat_mat_block256_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    expected: GgmlType,
    kernel_name: &'static str,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    encode_mat_mat_16bit_weight_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        expected,
        kernel_name,
    )
}

crate::env_flag!(default_on matmat_q3_k_mm_enabled, "QWEN_MATMAT_Q3_K_MM");

pub fn encode_mat_mat_q3_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_q3_k_mm_enabled() {
        return encode_mat_mat_q3_k_f32_mm(ctx, enc, weight, x, y, n_in, n_out, n_query);
    }
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::Q3_K,
        "kernel_mat_mat_q3_K_f32",
    )
}

pub(crate) fn encode_mat_mat_q3_k_f32_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_qk_lowbit_mm(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::Q3_K,
        "mat_mat_q3_k_mm",
        "kernel_mat_mat_q3_K_f32_mm",
        110,
    )
}

crate::env_flag!(default_on matmat_q2_k_mm_enabled, "QWEN_MATMAT_Q2_K_MM");

pub fn encode_mat_mat_q2_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_q2_k_mm_enabled() {
        return encode_mat_mat_q2_k_f32_mm(ctx, enc, weight, x, y, n_in, n_out, n_query);
    }
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::Q2_K,
        "kernel_mat_mat_q2_K_f32",
    )
}

pub(crate) fn encode_mat_mat_q2_k_f32_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_qk_lowbit_mm(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::Q2_K,
        "mat_mat_q2_k_mm",
        "kernel_mat_mat_q2_K_f32_mm",
        84,
    )
}

pub fn encode_mat_mat_iq2_xs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::IQ2_XS,
        "kernel_mat_mat_iq2_xs_f32",
    )
}

crate::env_flag!(default_on matmat_iq2_s_mm_enabled, "QWEN_MATMAT_IQ2_S_MM");

pub fn encode_mat_mat_iq2_s_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_iq2_s_mm_enabled() {
        return encode_mat_mat_qk_lowbit_mm(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            n_query,
            GgmlType::IQ2_S,
            "mat_mat_iq2_s_mm",
            "kernel_mat_mat_iq2_s_f32_mm",
            82,
        );
    }
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::IQ2_S,
        "kernel_mat_mat_iq2_s_f32",
    )
}

crate::env_flag!(default_on matmat_iq3_xxs_mm_enabled, "QWEN_MATMAT_IQ3_XXS_MM");

pub(crate) fn matmat_iq3_xxs_mm_is_enabled() -> bool {
    matmat_iq3_xxs_mm_enabled()
}

pub fn encode_mat_mat_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_iq3_xxs_mm_enabled() {
        return encode_mat_mat_qk_lowbit_mm(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            n_query,
            GgmlType::IQ3_XXS,
            "mat_mat_iq3_xxs_mm",
            "kernel_mat_mat_iq3_xxs_f32_mm",
            98,
        );
    }
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::IQ3_XXS,
        "kernel_mat_mat_iq3_xxs_f32",
    )
}

crate::env_flag!(default_on matmat_iq3_s_mm_enabled, "QWEN_MATMAT_IQ3_S_MM");

pub fn encode_mat_mat_iq3_s_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_iq3_s_mm_enabled() {
        return encode_mat_mat_qk_lowbit_mm(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            n_query,
            GgmlType::IQ3_S,
            "mat_mat_iq3_s_mm",
            "kernel_mat_mat_iq3_s_f32_mm",
            110,
        );
    }
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::IQ3_S,
        "kernel_mat_mat_iq3_s_f32",
    )
}

pub(crate) fn encode_mat_mat_qk_lowbit_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    expected: GgmlType,
    error_kernel: &'static str,
    metal_kernel: &'static str,
    block_bytes: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != expected {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!("weight.dtype = {:?}, expected {expected:?}", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let nb01 = ((n_in / 256) * block_bytes) as u32;
    let pso = ctx.pipeline(metal_kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

crate::env_flag!(default_on matmat_iq4_xs_mm_enabled, "QWEN_MATMAT_IQ4_XS_MM");

pub fn encode_mat_mat_iq4_xs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if matmat_iq4_xs_mm_enabled() {
        return encode_mat_mat_iq4_xs_f32_mm(ctx, enc, weight, x, y, n_in, n_out, n_query);
    }
    encode_mat_mat_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        GgmlType::IQ4_XS,
        "kernel_mat_mat_iq4_xs_f32",
    )
}

pub(crate) fn encode_mat_mat_iq4_xs_f32_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_xs_mm",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::IQ4_XS {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_xs_mm",
            detail: format!("weight.dtype = {:?}, expected IQ4_XS", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_xs_mm",
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_xs_mm",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_iq4_xs_mm",
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let nb01 = ((n_in / 256) * 136) as u32;
    let pso = ctx.pipeline("kernel_mat_mat_iq4_xs_f32_mm")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Small-N MMA experiment (v0.498 follow-up): 8-row x 8-padded-column
/// simdgroup_matrix tile at mat-vec-grade occupancy (`n_out/8`
/// single-simdgroup threadgroups). Caller ALWAYS provides 8 activation
/// columns (`x`: F32 `[8, n_in]`) and receives 8 output columns
/// (`y`: F32 `[8, n_out]`, `y[c*n_out + r]`); pad unused columns.
/// Q4_K | Q6_K. Exactness: E1 (half-staged weight dequant + MMA
/// accumulation order; activations NOT half-staged — loaded F32 direct).
/// Gate: cos >= 0.999 per column vs mv1, asserted by
/// `smalln_mma_micro_27b`.
pub fn encode_mat_mat_mma8_dispatch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8",
            detail: format!("n_in={n_in} not divisible by 256 (K-quant super-block)"),
        });
    }
    if !n_out.is_multiple_of(8) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8",
            detail: format!("n_out={n_out} not divisible by 8 (tile rows)"),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8",
            detail: format!("x/y dtype = {:?}/{:?}, expected F32/F32", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != 8 * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8",
            detail: format!("x.n_elements={} != 8*n_in={}", x.n_elements(), 8 * n_in),
        });
    }
    if y.n_elements() as usize != 8 * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8",
            detail: format!("y.n_elements={} != 8*n_out={}", y.n_elements(), 8 * n_out),
        });
    }
    if weight.n_elements() as usize != n_in * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8",
            detail: format!(
                "weight.n_elements={} != n_in*n_out={}",
                weight.n_elements(),
                n_in * n_out
            ),
        });
    }
    let name = match weight.dtype {
        GgmlType::Q4_K => "kernel_mat_mat_q4_K_mma8_f32",
        GgmlType::Q6_K => "kernel_mat_mat_q6_K_mma8_f32",
        other => {
            return Err(MetalError::BadShape {
                kernel: "mat_mat_mma8",
                detail: format!("unsupported dtype {other:?} (Q4_K | Q6_K)"),
            });
        }
    };
    let pso = ctx.pipeline(name)?;
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
            n_in: n_in as u32,
            n_out: n_out as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.dispatch(
        MTLSize {
            width: n_out / 8,
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

/// Dispatch a named small-N mma8v variant. `variant` ∈ {"r2c1k64",
/// "r1c1k128", "r1c2k64",
/// "r1c1k128_vec4", "r1c1k64_sg2", "r1c1k64_sg2_vec4",
/// "r2c1k64_vec4", "r2c2k64", "r2c1k128", "r4c1k64", "r2c2k128"};
/// column count = 8*CT (x/y must carry exactly that many columns),
/// rows per TG = 8*RT*SGS.
pub fn encode_mat_mat_mma8_variant(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    variant: &str,
) -> Result<(), MetalError> {
    let (rt, ct, sgs) = match variant {
        "r2c1k64" => (2usize, 1usize, 1usize),
        "r1c1k128" => (1, 1, 1),
        "r1c1k128_vec4" => (1, 1, 1),
        "r1c2k64" => (1, 2, 1),
        "r1c1k64_sg2" => (1, 1, 2),
        "r1c1k64_sg2_vec4" => (1, 1, 2),
        "r2c1k64_vec4" => (2, 1, 1),
        "r2c2k64" => (2, 2, 1),
        "r2c1k128" => (2, 1, 1),
        "r4c1k64" => (4, 1, 1),
        "r2c2k128" => (2, 2, 1),
        _ => {
            return Err(MetalError::BadShape {
                kernel: "mat_mat_mma8v",
                detail: format!("unknown variant {variant}"),
            });
        }
    };
    let cols = 8 * ct;
    let rows_per_tg = 8 * rt * sgs;
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8v",
            detail: format!("x/y dtype = {:?}/{:?}, expected F32/F32", x.dtype, y.dtype),
        });
    }
    if weight.n_elements() as usize != n_in * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8v",
            detail: format!(
                "weight.n_elements={} != n_in*n_out={}",
                weight.n_elements(),
                n_in * n_out
            ),
        });
    }
    if !n_in.is_multiple_of(256) || !n_out.is_multiple_of(rows_per_tg) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8v",
            detail: format!("n_in={n_in} % 256 or n_out={n_out} % rows_per_tg={rows_per_tg} != 0"),
        });
    }
    if x.n_elements() as usize != cols * n_in || y.n_elements() as usize != cols * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8v",
            detail: format!(
                "x/y elements {}/{} != cols({cols}) * n_in/n_out",
                x.n_elements(),
                y.n_elements()
            ),
        });
    }
    if variant.ends_with("vec4") && weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_mma8v",
            detail: format!("variant {variant} only supports Q4_K"),
        });
    }
    let dt = match weight.dtype {
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q6_K => "q6_K",
        // v0.77: Q5_K (packed-verify GDN out_proj) and Q8_0 (DFlash 2
        // drafter projections) join the N=8 tier — ct=1 variants only.
        GgmlType::Q5_K => "q5_K",
        GgmlType::Q8_0 => "q8_0",
        other => {
            return Err(MetalError::BadShape {
                kernel: "mat_mat_mma8v",
                detail: format!("unsupported dtype {other:?}"),
            });
        }
    };
    let name = format!("kernel_mat_mat_{dt}_mma8v_{variant}_f32");
    let pso = ctx.pipeline(&name)?;
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
            n_in: n_in as u32,
            n_out: n_out as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.dispatch(
        MTLSize {
            width: n_out / rows_per_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32 * sgs,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn encode_ffn_fused_swiglu_q4_k_mma8_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    x: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    const N: usize = 8;
    if gate.dtype != GgmlType::Q4_K
        || up.dtype != GgmlType::Q4_K
        || x.dtype != GgmlType::F32
        || inner.dtype != GgmlType::F32
        || !n_in.is_multiple_of(256)
        || !n_out.is_multiple_of(8)
        || gate.n_elements() as usize != n_in * n_out
        || up.n_elements() as usize != n_in * n_out
        || x.n_elements() as usize != N * n_in
        || inner.n_elements() as usize != N * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_k_mma8",
            detail: format!(
                "expected Q4_K gate/up [{n_in},{n_out}] and F32 [8,{n_in}] -> [8,{n_out}]"
            ),
        });
    }
    let pso = ctx.pipeline("kernel_ffn_fused_swiglu_q4_K_mma8_f32")?;
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
            n_in: n_in as u32,
            n_out: n_out as u32,
        },
    );
    enc.set_tensor(1, gate);
    enc.set_tensor(2, up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, inner);
    enc.dispatch(
        MTLSize {
            width: n_out / 8,
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

crate::env_flag!(default_off matmat_qk_llama_smem_enabled, "QWEN_MATMAT_QK_LLAMA_SMEM");

/// Q4_K mat-mat: `Y = W · X^T` where
///   * `W` is Q4_K `[n_out, n_in]` (row-major in Q4_K block bytes)
///   * `X` is F32 `[n_query, n_in]` row-major
///   * `Y` is F32 `[n_query, n_out]` row-major — equivalently
///     `Y[c * n_out + r]` for cell `(r, c)` of the kernel's
///     "[n_out, n_query] col-major" view (the bytes are bit-identical;
///     llama's notation just uses column-major framing).
///
/// **For layer-major H5.3b.4-5 plumbing, treat the output as
/// row-major `[n_query, n_out]`.** This means downstream consumers
/// (FFN silu_mul, residual_add, chained mat-mat with this output
/// as `srcB`) work without any transpose: the bytes ARE in the
/// row-major order the next mat-mat call expects as input. The
/// H5.3b.0 layout sanity test verified this equivalence at three
/// corner cells.
///
/// Lifts the 64×32×32 simdgroup_matrix tile from llama.cpp
/// `kernel_mul_mm_q4_K_f32` (classic non-MPS-tensor path,
/// ggml-metal.metal:9440-9648). Per H5.3b plan rev 6.
///
/// Constraints:
///   * `n_in % 256 == 0` (Q4_K super-block alignment)
///   * `n_in % 32 == 0` (kernel's NK_MM=32 K-step)
///   * The kernel internally tiles N to 32 (NR1_MM); host should
///     pass `n_query` directly (kernel handles N < 32 via partial-
///     output-tile path with threadgroup-mem buffered write).
///
/// Threadgroup memory: 5120/6144 bytes for full tiles, 8192 when edge-store
/// scratch is needed.
/// Threadgroup size: 128 threads (4 simdgroups × 32 lanes).
///
/// **NOT bit-exact** with N successive `encode_mat_vec_q4_k_f32`
/// (codex Q3 correction). The lifted kernel stages activations
/// through half before float accumulation; cosine ≥ 0.999 vs
/// scalar-float mat-vec is the gate (vs cos ≥ 0.9999 against a
/// CPU mat-mat oracle that uses the same staging).
pub(crate) fn mat_mat_qk_threadgroup_memory(n_out: usize, n_query: usize, nr1: usize) -> usize {
    mat_mat_qk_threadgroup_memory_with_policy(n_out, n_query, nr1, matmat_qk_llama_smem_enabled())
}

pub(crate) fn mat_mat_qk_threadgroup_memory_with_policy(
    n_out: usize,
    n_query: usize,
    nr1: usize,
    llama_smem: bool,
) -> usize {
    if !llama_smem {
        return 8192;
    }
    if n_out.is_multiple_of(64) && n_query.is_multiple_of(nr1) {
        if nr1 == 16 { 5120 } else { 6144 }
    } else {
        8192
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatMatQ4KN64Mode {
    Auto,
    ForceOn,
    ForceOff,
}

pub(crate) fn mat_mat_q4_k_n64_mode() -> MatMatQ4KN64Mode {
    static MODE: OnceLock<MatMatQ4KN64Mode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("QWEN_MATMAT_Q4_K_N64").as_deref() {
        Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO") => MatMatQ4KN64Mode::ForceOff,
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") => MatMatQ4KN64Mode::ForceOn,
        _ => MatMatQ4KN64Mode::Auto,
    })
}

#[cfg(test)]
pub(crate) fn mat_mat_q4_k_n64_enabled() -> bool {
    !matches!(mat_mat_q4_k_n64_mode(), MatMatQ4KN64Mode::ForceOff)
}

pub(crate) fn mat_mat_q4_k_use_n64(n_in: usize, n_out: usize, n_query: usize) -> bool {
    match mat_mat_q4_k_n64_mode() {
        MatMatQ4KN64Mode::ForceOff => false,
        MatMatQ4KN64Mode::ForceOn => true,
        MatMatQ4KN64Mode::Auto => !(n_query <= 512 && (n_in <= 2048 || n_out <= 2048)),
    }
}

crate::env_flag!(default_off mat_mat_n16_v2_enabled, "QWEN_MATMAT_N16_V2");

crate::env_flag!(default_on mat_mat_q5_k_n64_enabled, "QWEN_MATMAT_Q5_K_N64");

pub(crate) fn mat_mat_q5_k_n64_min_query() -> usize {
    static MIN_N: OnceLock<usize> = OnceLock::new();
    *MIN_N.get_or_init(|| {
        std::env::var("QWEN_MATMAT_Q5_K_N64_MIN_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 64)
            .unwrap_or(if cfg!(test) { 64 } else { 1024 })
    })
}

crate::env_flag!(default_on mat_mat_q6_k_n64_enabled, "QWEN_MATMAT_Q6_K_N64");

pub(crate) fn mat_mat_q6_k_n64_min_query() -> usize {
    static MIN_N: OnceLock<usize> = OnceLock::new();
    *MIN_N.get_or_init(|| {
        std::env::var("QWEN_MATMAT_Q6_K_N64_MIN_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 64)
            .unwrap_or(if cfg!(test) { 64 } else { 1024 })
    })
}

pub fn encode_mat_mat_q4_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor, // [n_query, n_in] row-major F32
    y: &MetalTensor, // F32 [n_out * n_query] flat. SEE LAYOUT NOTE BELOW.
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q4_k",
            detail: format!("n_in={n_in} not divisible by 256 (Q4_K super-block)"),
        });
    }
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q4_k",
            detail: format!("n_in={n_in} not divisible by 32 (NK_MM tile)"),
        });
    }
    if weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q4_k",
            detail: format!("weight.dtype = {:?}, expected Q4_K", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q4_k",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q4_k",
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let nb01 = ((n_in / 256) * 144) as u32;
    let stride_b = n_in as u32;

    let use_n64 = mat_mat_q4_k_use_n64(n_in, n_out, n_query)
        && n_query.is_multiple_of(64)
        && n_out.is_multiple_of(64);
    // H5.6 M2a: raw-block-staged N16 kernel (v2). The v1 A-path dequants
    // straight from device with ~2.7x byte amplification; v2 stages raw
    // super-blocks to threadgroup memory coalesced. K must cover whole
    // super-blocks. Rollback: QWEN_MATMAT_N16_V2=0.
    let use_n16_v2 =
        n_query == 16 && !use_n64 && n_in.is_multiple_of(256) && mat_mat_n16_v2_enabled();
    let kernel_name = if use_n64 {
        "kernel_mat_mat_q4_K_f32_n64"
    } else if use_n16_v2 {
        "kernel_mat_mat_q4_K_f32_n16_v2"
    } else if n_query == 16 {
        "kernel_mat_mat_q4_K_f32_n16"
    } else {
        "kernel_mat_mat_q4_K_f32"
    };
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    let nr1 = if use_n64 {
        64
    } else if n_query == 16 {
        16
    } else {
        32
    };
    let smem = if use_n64 {
        8192
    } else if use_n16_v2 {
        // raw 9216 + sa 4096 + sb 1024 (kernel doc block).
        14336
    } else {
        mat_mat_qk_threadgroup_memory(n_out, n_query, nr1)
    };
    enc.set_threadgroup_memory(0, smem);
    let n_tg_x = n_query.div_ceil(nr1);
    let n_tg_y = n_out.div_ceil(64);
    let threads = if use_n64 { 256 } else { 128 };
    enc.dispatch(
        MTLSize {
            width: n_tg_x,
            height: n_tg_y,
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

/// Bench-only production-grid Q4_K N64 MMA ceiling arms.
///
/// These are not model-forward implementations. They retain the exact v0.606
/// `[5120, 17408] x N=1024` grid, MMA count, FP32 accumulators, and stores while
/// removing all weight and activation traffic. `Tgm8CapMatched` makes both ends
/// of an 8192-byte dynamic threadgroup allocation live once before the MMA loop;
/// it matches that capacity constraint, not production occupancy.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q4MatMatMmaCeilingArm {
    Pure,
    Tgm8CapMatched,
}

impl Q4MatMatMmaCeilingArm {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pure => "e0_mma_only",
            Self::Tgm8CapMatched => "e8_mma_only_tgm8_cap_matched",
        }
    }

    pub(crate) fn kernel_name(self) -> &'static str {
        match self {
            Self::Pure => "kernel_mat_mat_q4_K_f32_n64_mma_ceiling",
            Self::Tgm8CapMatched => "kernel_mat_mat_q4_K_f32_n64_mma_ceiling_tgm8",
        }
    }
}

#[doc(hidden)]
pub fn encode_mat_mat_q4_k_mma_ceiling(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    y: &MetalTensor,
    arm: Q4MatMatMmaCeilingArm,
    nonce: u32,
) -> Result<(), MetalError> {
    const N_IN: usize = 5120;
    const N_OUT: usize = 17408;
    const N_QUERY: usize = 1024;

    if y.dtype != GgmlType::F32 || !y.is_writable() {
        return Err(MetalError::BadShape {
            kernel: arm.kernel_name(),
            detail: format!(
                "y must be writable F32, got dtype={:?} provenance={:?}",
                y.dtype,
                y.provenance()
            ),
        });
    }
    if y.n_elements() as usize != N_QUERY * N_OUT {
        return Err(MetalError::BadShape {
            kernel: arm.kernel_name(),
            detail: format!(
                "y.n_elements={} != fixed n_query*n_out={}",
                y.n_elements(),
                N_QUERY * N_OUT
            ),
        });
    }
    let y_end = y
        .offset
        .checked_add(y.n_bytes())
        .ok_or_else(|| MetalError::BadShape {
            kernel: arm.kernel_name(),
            detail: "y byte range overflows u64".to_string(),
        })?;
    if y_end > y.buffer.length() as u64 {
        return Err(MetalError::BadShape {
            kernel: arm.kernel_name(),
            detail: format!(
                "y byte end {y_end} exceeds buffer length {}",
                y.buffer.length()
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }

    let pso = ctx.pipeline(arm.kernel_name())?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: N_OUT as u32,
            n: N_QUERY as u32,
            k: N_IN as u32,
            nb01: ((N_IN / 256) * 144) as u32,
            stride_b: N_IN as u32,
        },
    );
    enc.set_tensor(3, y);
    enc.set_bytes(4, &nonce);
    if arm == Q4MatMatMmaCeilingArm::Tgm8CapMatched {
        enc.set_threadgroup_memory(0, 8192);
    }
    enc.dispatch(
        MTLSize {
            width: N_QUERY / 64,
            height: N_OUT / 64,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Bench-only production-grid Q4_K N64 no-dequant attribution arms.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q4MatMatNoDequantArm {
    SourceSegmentsLive,
    NoSource,
}

impl Q4MatMatNoDequantArm {
    pub fn label(self) -> &'static str {
        match self {
            Self::SourceSegmentsLive => "b_source_segments_live_no_dequant",
            Self::NoSource => "c_no_source_no_dequant",
        }
    }

    pub(crate) fn kernel_name(self) -> &'static str {
        match self {
            Self::SourceSegmentsLive => {
                "kernel_mat_mat_q4_K_f32_n64_source_segments_live_no_dequant"
            }
            Self::NoSource => "kernel_mat_mat_q4_K_f32_n64_no_source_no_dequant",
        }
    }
}

#[doc(hidden)]
pub fn encode_mat_mat_q4_k_no_dequant(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    arm: Q4MatMatNoDequantArm,
    nonce: u32,
) -> Result<(), MetalError> {
    const N_IN: usize = 5120;
    const N_OUT: usize = 17408;
    const N_QUERY: usize = 1024;

    if weight.dtype != GgmlType::Q4_K
        || weight.shape.as_slice() != [N_IN as u64, N_OUT as u64]
        || !weight.offset.is_multiple_of(16)
    {
        return Err(MetalError::BadShape {
            kernel: arm.kernel_name(),
            detail: format!(
                "weight must be aligned Q4_K [{N_IN},{N_OUT}], got dtype={:?} shape={:?} offset={}",
                weight.dtype, weight.shape, weight.offset
            ),
        });
    }
    if x.dtype != GgmlType::F32
        || x.shape.as_slice() != [N_QUERY as u64, N_IN as u64]
        || !x.offset.is_multiple_of(16)
    {
        return Err(MetalError::BadShape {
            kernel: arm.kernel_name(),
            detail: format!(
                "x must be 16-byte-aligned F32 [{N_QUERY},{N_IN}], got dtype={:?} shape={:?} offset={}",
                x.dtype, x.shape, x.offset
            ),
        });
    }
    if y.dtype != GgmlType::F32
        || !y.is_writable()
        || y.shape.as_slice() != [N_QUERY as u64, N_OUT as u64]
        || !y.offset.is_multiple_of(4)
    {
        return Err(MetalError::BadShape {
            kernel: arm.kernel_name(),
            detail: format!(
                "y must be aligned writable F32 [{N_QUERY},{N_OUT}], got dtype={:?} shape={:?} offset={} provenance={:?}",
                y.dtype,
                y.shape,
                y.offset,
                y.provenance()
            ),
        });
    }
    for (label, tensor) in [("weight", weight), ("x", x), ("y", y)] {
        let end =
            tensor
                .offset
                .checked_add(tensor.n_bytes())
                .ok_or_else(|| MetalError::BadShape {
                    kernel: arm.kernel_name(),
                    detail: format!("{label} byte range overflows u64"),
                })?;
        if end > tensor.buffer.length() as u64 {
            return Err(MetalError::BadShape {
                kernel: arm.kernel_name(),
                detail: format!(
                    "{label} byte end {end} exceeds buffer length {}",
                    tensor.buffer.length()
                ),
            });
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }

    let pso = ctx.pipeline(arm.kernel_name())?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: N_OUT as u32,
            n: N_QUERY as u32,
            k: N_IN as u32,
            nb01: ((N_IN / 256) * 144) as u32,
            stride_b: N_IN as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_bytes(4, &nonce);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: N_QUERY / 64,
            height: N_OUT / 64,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Q6_K mat-mat: same shape contract as [`encode_mat_mat_q4_k_f32`].
///
/// Lifts the same 64×32×32 simdgroup_matrix tile from llama.cpp,
/// templated on Q6_K dequant. Output is row-major `[n_query, n_out]`
/// (= bit-equivalent to llama's "[n_out, n_query] col-major" framing).
///
/// Used by H5.3b.6 to lift `ffn_down` and `lm_head` out of the
/// per-token mat-vec re-read loop.
pub fn encode_mat_mat_q6_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor, // [n_query, n_in] row-major F32
    y: &MetalTensor, // F32 [n_out * n_query] flat (row-major [n_query, n_out]).
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q6_k",
            detail: format!("n_in={n_in} not divisible by 256 (Q6_K super-block)"),
        });
    }
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q6_k",
            detail: format!("n_in={n_in} not divisible by 32 (NK_MM tile)"),
        });
    }
    if weight.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q6_k",
            detail: format!("weight.dtype = {:?}, expected Q6_K", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q6_k",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q6_k",
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    let use_n64 = mat_mat_q6_k_n64_enabled()
        && n_query >= mat_mat_q6_k_n64_min_query()
        && n_query.is_multiple_of(64)
        && n_out.is_multiple_of(64);
    let kernel_name = if use_n64 {
        "kernel_mat_mat_q6_K_f32_n64"
    } else if n_query == 16 {
        "kernel_mat_mat_q6_K_f32_n16"
    } else {
        "kernel_mat_mat_q6_K_f32"
    };
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);

    // Q6_K block bytes per row = (n_in / 256) * 210.
    let nb01 = ((n_in / 256) * 210) as u32;
    let stride_b = n_in as u32;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);

    let nr1 = if use_n64 {
        64
    } else if n_query == 16 {
        16
    } else {
        32
    };
    let smem = if use_n64 {
        8192
    } else {
        mat_mat_qk_threadgroup_memory(n_out, n_query, nr1)
    };
    enc.set_threadgroup_memory(0, smem);
    let n_tg_x = n_query.div_ceil(nr1);
    let n_tg_y = n_out.div_ceil(64);
    let threads = if use_n64 { 256 } else { 128 };
    enc.dispatch(
        MTLSize {
            width: n_tg_x,
            height: n_tg_y,
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

/// Q5_K mat-mat: same shape contract as [`encode_mat_mat_q4_k_f32`] /
/// [`encode_mat_mat_q6_k_f32`].
///
/// Lifts the same 64×32×32 simdgroup_matrix tile from llama.cpp,
/// templated on Q5_K dequant (adds the qh high-bit contribution to
/// the Q4_K nibble path; same scale/min decode as Q4_K).
///
/// Output is row-major `[n_query, n_out]` (= bit-equivalent to
/// llama's `[n_out, n_query] col-major` framing).
///
/// Used by v0.73a.1 to lift GDN `out_proj` (Q5_K [v_dim, hidden])
/// out of the per-token mat-vec re-read loop. Per-row cosine ≥ 0.999
/// vs N successive Q5_K mat-vec is the gate (same threshold as
/// Q4_K / Q6_K mat-mat — half-staging in lifted kernel).
pub fn encode_mat_mat_q5_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor, // [n_query, n_in] row-major F32
    y: &MetalTensor, // F32 [n_out * n_query] flat (row-major [n_query, n_out]).
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q5_k",
            detail: format!("n_in={n_in} not divisible by 256 (Q5_K super-block)"),
        });
    }
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q5_k",
            detail: format!("n_in={n_in} not divisible by 32 (NK_MM tile)"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q5_k",
            detail: format!("weight.dtype = {:?}, expected Q5_K", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q5_k",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q5_k",
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }

    let use_n64 = mat_mat_q5_k_n64_enabled()
        && n_query >= mat_mat_q5_k_n64_min_query()
        && n_query.is_multiple_of(64)
        && n_out.is_multiple_of(64);
    let kernel_name = if use_n64 {
        "kernel_mat_mat_q5_K_f32_n64"
    } else if n_query == 16 {
        "kernel_mat_mat_q5_K_f32_n16"
    } else {
        "kernel_mat_mat_q5_K_f32"
    };
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);

    // Q5_K block bytes per row = (n_in / 256) * 176.
    let nb01 = ((n_in / 256) * 176) as u32;
    let stride_b = n_in as u32;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);

    let nr1 = if use_n64 {
        64
    } else if n_query == 16 {
        16
    } else {
        32
    };
    let smem = if use_n64 {
        8192
    } else {
        mat_mat_qk_threadgroup_memory(n_out, n_query, nr1)
    };
    enc.set_threadgroup_memory(0, smem);
    let n_tg_x = n_query.div_ceil(nr1);
    let n_tg_y = n_out.div_ceil(64);
    let threads = if use_n64 { 256 } else { 128 };
    enc.dispatch(
        MTLSize {
            width: n_tg_x,
            height: n_tg_y,
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

/// Q8_0 mat-mat: same shape contract as [`encode_mat_mat_q4_k_f32`] /
/// [`encode_mat_mat_q5_k_f32`] / [`encode_mat_mat_q6_k_f32`].
///
/// CRITICAL difference: Q8_0 super-block is QK8_0=32 elements (vs
/// QK_K=256 for K-quants). The kernel still requires `n_in % 32 == 0`,
/// matching the K-step `NK_MM=32`. The pointer-advance specializes
/// to `Q8_0_NL=2` (one super-block per K-step per row).
///
/// Output is row-major `[n_query, n_out]` (= bit-equivalent to
/// llama's `[n_out, n_query] col-major` framing).
///
/// Used by v0.73b.0 to lift the DFlash drafter Q8_0 mat-mat path
/// (lm_head, FFN, projections) once the loader switches from
/// F32-dequant to native Q8_0. Per-row cosine ≥ 0.999 vs N successive
/// Q8_0 mat-vec is the gate (same threshold as Q4_K/Q5_K/Q6_K mat-mat
/// — half-staging in lifted kernel).
pub fn encode_mat_mat_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q8_0",
            detail: format!("n_in={n_in} not divisible by 32 (Q8_0 super-block / NK_MM tile)"),
        });
    }
    if weight.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q8_0",
            detail: format!("weight.dtype = {:?}, expected Q8_0", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q8_0",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_mat_q8_0",
            detail: format!(
                "y.n_elements={} != n_query*n_out={}",
                y.n_elements(),
                n_query * n_out
            ),
        });
    }
    // NR1=16 fast-path gate: same specialization as Q4_K/Q5_K/Q6_K mat-mat.
    // Only fires when n_query == 16 exactly; otherwise generic NR1=32 kernel.
    let kernel_name = if n_query == 16 {
        "kernel_mat_mat_q8_0_f32_n16"
    } else {
        "kernel_mat_mat_q8_0_f32"
    };
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);

    // Q8_0 block bytes per row = (n_in / 32) * 34.
    let nb01 = ((n_in / 32) * 34) as u32;
    let stride_b = n_in as u32;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);

    let nr1 = if n_query == 16 { 16 } else { 32 };
    enc.set_threadgroup_memory(0, mat_mat_qk_threadgroup_memory(n_out, n_query, nr1));
    let n_tg_x = n_query.div_ceil(nr1);
    let n_tg_y = n_out.div_ceil(64);
    enc.dispatch(
        MTLSize {
            width: n_tg_x,
            height: n_tg_y,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// **EXPERIMENTAL — FAILED A-LITE GATE — NOT WIRED INTO PRODUCTION (v0.73c.2)**
///
/// Layer-major fused SwiGLU FFN — Q4_K mat-mat × 2 + silu_mul, NR1=16.
///
/// Fuses the 3-dispatch sequence (gate_mm + up_mm + silu_mul) into one
/// kernel per FFN layer. Bit-exact with the unfused reference (cos =
/// 1.000000, max|Δ| = 0). Lifted from mat_mat_q4_k.metal NR1=16 with
/// doubled accumulators (mc_gate[4] + mc_up[4]) and shared sb tile.
///
/// **Why not in production:** A-lite bench at production 64-layer 27B
/// shape (n_in=5120, n_out=17408, N=16) measured 1.07× speedup vs
/// unfused — codex threshold was ≤ 0.7 (i.e. ≥ 30% speedup needed).
/// The unchanged W_gate + W_up weight reads dominate; fusion only saves
/// dispatch count and intermediate I/O, both of which Metal already
/// pipelines well within one command buffer. Codex's optimistic 5-15
/// ms/call savings estimate was ~30× too high (actual: ~0.5 ms/call).
///
/// See `kernels/ffn_fused_swiglu_q4_k_mm.metal` header for the full
/// negative-result writeup. Preserved as institutional memory; do NOT
/// plumb without re-running `ffn_fused_swiglu_q4_K_amortization_vs_unfused`
/// to confirm the regime has changed.
///
/// Constraints:
///   * `n_in % 256 == 0` (Q4_K super-block alignment)
///   * `n_query == 16` (NR1=16 fast path; host enforces)
///   * Both weight tensors must be Q4_K (host check)
///
/// Threadgroup memory: 16384 B (sa_g 4 KiB + sa_u 4 KiB + sb 1 KiB live)
#[allow(non_snake_case)]
pub fn encode_ffn_fused_swiglu_q4_K_mm_n16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor, // Q4_K [n_in, n_out]
    w_up: &MetalTensor,   // Q4_K [n_in, n_out]
    x: &MetalTensor,      // F32 [n_query=16, n_in] row-major
    inner: &MetalTensor,  // F32 [n_query, n_out] row-major
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm_n16",
            detail: format!("n_in={n_in} not divisible by 256 (Q4_K super-block)"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm_n16",
            detail: format!(
                "w_gate.dtype={:?} w_up.dtype={:?}, both must be Q4_K",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    const N: usize = 16;
    if x.n_elements() as usize != N * n_in {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm_n16",
            detail: format!("x.n_elements={} != N*n_in={}", x.n_elements(), N * n_in),
        });
    }
    if inner.n_elements() as usize != N * n_out {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm_n16",
            detail: format!(
                "inner.n_elements={} != N*n_out={}",
                inner.n_elements(),
                N * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_ffn_fused_swiglu_q4_K_mm_n16_f32")?;
    enc.set_pipeline(&pso);

    let nb01 = ((n_in / 256) * 144) as u32;
    let stride_b = n_in as u32;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: N as u32,
            k: n_in as u32,
            nb01,
            stride_b,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, inner);

    enc.set_threadgroup_memory(0, 16384);

    let n_tg_x = N.div_ceil(16);
    let n_tg_y = n_out.div_ceil(64);
    enc.dispatch(
        MTLSize {
            width: n_tg_x,
            height: n_tg_y,
            depth: 1,
        },
        MTLSize {
            width: 128, // 4 simdgroups × 32 lanes
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_ffn_fused_swiglu_q4_K_mm_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm",
            detail: format!("n_in={n_in} not divisible by 256 (Q4_K super-block)"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm",
            detail: format!(
                "w_gate.dtype={:?} w_up.dtype={:?}, both must be Q4_K",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_query * n_in {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm",
            detail: format!(
                "x.n_elements={} != n_query*n_in={}",
                x.n_elements(),
                n_query * n_in
            ),
        });
    }
    if inner.n_elements() as usize != n_query * n_out {
        return Err(MetalError::BadShape {
            kernel: "ffn_fused_swiglu_q4_K_mm",
            detail: format!(
                "inner.n_elements={} != n_query*n_out={}",
                inner.n_elements(),
                n_query * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_ffn_fused_swiglu_q4_K_mm_f32")?;
    enc.set_pipeline(&pso);

    let nb01 = ((n_in / 256) * 144) as u32;
    let stride_b = n_in as u32;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_query as u32,
            k: n_in as u32,
            nb01,
            stride_b,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, inner);
    enc.set_threadgroup_memory(0, 16384);

    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
