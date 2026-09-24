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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;

    #[test]
    fn iq2_xs_mat_vec_and_mat_mat_match_cpu_codec_with_offsets() {
        let ctx = match metal_test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        const N_IN: usize = 4_096;
        const N_OUT: usize = 9;
        let mut weight_bytes = Vec::new();
        for row in 0..N_OUT {
            for block in 0..N_IN / 256 {
                weight_bytes.extend_from_slice(&encode_iq2_xs_block(
                    0.001953125 * (1 + (row * 2 + block) % 7) as f32,
                    row * (N_IN / 256) + block,
                ));
            }
        }
        let desc = TensorDesc {
            name: "iq2_xs_test".into(),
            shape: vec![N_IN as u64, N_OUT as u64],
            dtype: GgmlType::IQ2_XS,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: weight_bytes.len() as u64,
        };
        let decoded = crate::codec::dequant_to_f32(&desc, &weight_bytes)
            .expect("llama.cpp IQ2_XS reference dequantization");
        let weight = offset_tensor(
            &ctx,
            32,
            &weight_bytes,
            19,
            vec![N_IN as u64, N_OUT as u64],
            GgmlType::IQ2_XS,
        );
        let input_values = (0..N_IN)
            .map(|index| ((index * 37 + 5) % 251) as f32 * 0.001 - 0.125)
            .collect::<Vec<_>>();
        let input = offset_tensor(
            &ctx,
            16,
            bytemuck::cast_slice(&input_values),
            17,
            vec![N_IN as u64],
            GgmlType::F32,
        );
        let output = offset_tensor(
            &ctx,
            32,
            &[0u8; N_OUT * size_of::<f32>()],
            23,
            vec![N_OUT as u64],
            GgmlType::F32,
        );
        let command = ctx.queue.commandBuffer().expect("IQ2_XS mat-vec command");
        let encoder = KernelEncoder::begin(&command);
        crate::metal_forward::encode_mat_vec_dispatch(
            &ctx, &encoder, &weight, &input, &output, N_IN, N_OUT,
        )
        .expect("IQ2_XS mat-vec dispatch");
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(command.error().is_none(), "IQ2_XS mat-vec command failed");
        let expected_row = |row: usize, input: &[f32]| {
            decoded[row * N_IN..(row + 1) * N_IN]
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
        };
        for (row, actual) in tensor_f32_at_offset(&output).into_iter().enumerate() {
            let expected = expected_row(row, &input_values);
            let tolerance = 3.0e-5 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "mat-vec row {row}: got {actual}, expected {expected}, tolerance {tolerance}"
            );
        }

        for n_query in [1usize, 2, 6, 16, 32, 128] {
            let inputs = (0..n_query * N_IN)
                .map(|index| ((index * 41 + index / N_IN * 17 + 3) % 509) as f32 * 0.0005 - 0.127)
                .collect::<Vec<_>>();
            let x = offset_tensor(
                &ctx,
                16,
                bytemuck::cast_slice(&inputs),
                13,
                vec![N_IN as u64, n_query as u64],
                GgmlType::F32,
            );
            let y = offset_tensor(
                &ctx,
                32,
                &vec![0u8; n_query * N_OUT * size_of::<f32>()],
                29,
                vec![N_OUT as u64, n_query as u64],
                GgmlType::F32,
            );
            let command = ctx.queue.commandBuffer().expect("IQ2_XS mat-mat command");
            let encoder = KernelEncoder::begin(&command);
            crate::metal_forward::encode_mat_mat_dispatch(
                &ctx, &encoder, &weight, &x, &y, N_IN, N_OUT, n_query,
            )
            .unwrap_or_else(|error| panic!("IQ2_XS mat-mat N={n_query}: {error}"));
            encoder.end();
            command.commit();
            crate::metal::wait_completed(&command).expect("command buffer completed");
            assert!(
                command.error().is_none(),
                "IQ2_XS mat-mat N={n_query} command failed"
            );
            let actual = tensor_f32_at_offset(&y);
            for query in 0..n_query {
                let input = &inputs[query * N_IN..(query + 1) * N_IN];
                for row in 0..N_OUT {
                    let expected = expected_row(row, input);
                    let got = actual[query * N_OUT + row];
                    let tolerance = 3.0e-5 * expected.abs().max(1.0);
                    assert!(
                        (got - expected).abs() <= tolerance,
                        "mat-mat N={n_query} query={query} row={row}: got {got}, expected {expected}, tolerance {tolerance}"
                    );
                }
            }
        }
    }

    #[test]
    fn read_only_mmap_backing_blits_and_outlives_rust_views() {
        use std::io::Write;
        use std::sync::atomic::AtomicUsize;

        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        let page_size = host_page_size().expect("host page size");
        let mut bytes = vec![0u8; page_size * 2];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = index.wrapping_mul(17) as u8;
        }
        let n_in = 32usize;
        let n_out = 16usize;
        let weights: Vec<f32> = (0..n_in * n_out)
            .map(|index| ((index % 23) as f32 - 11.0) * 0.01)
            .collect();
        bytes[32..32 + weights.len() * 4].copy_from_slice(bytemuck::cast_slice(&weights));
        let mut path = std::env::temp_dir();
        path.push(format!(
            "qwen-metal-no-copy-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::File::create(&path)
            .and_then(|mut file| file.write_all(&bytes))
            .expect("write mmap fixture");

        let callback_calls = Arc::new(AtomicUsize::new(0));
        let callback_mismatches = Arc::new(AtomicUsize::new(0));
        let (weak, source_probe) = objc2::rc::autoreleasepool(|_| {
            let file = std::fs::File::open(&path).expect("open mmap fixture");
            // SAFETY: the test retains the immutable file and does not mutate
            // or truncate it while the mapping exists.
            let mmap = Arc::new(unsafe { Mmap::map(&file).expect("map fixture") });
            let weak = Arc::downgrade(&mmap);
            let expected_pointer = mmap.as_ptr() as usize;
            let expected_length = bytes.len();
            let calls = Arc::clone(&callback_calls);
            let mismatches = Arc::clone(&callback_mismatches);
            let backing = ctx
                .gguf_no_copy_backing_with_observer(
                    Arc::clone(&mmap),
                    0,
                    32,
                    move |pointer, length| {
                        calls.fetch_add(1, Ordering::Relaxed);
                        if pointer.as_ptr() as usize != expected_pointer
                            || length != expected_length
                        {
                            mismatches.fetch_add(1, Ordering::Relaxed);
                        }
                    },
                )
                .expect("read-only no-copy backing");
            let probe = DiagnosticGgufBlitReleaseProbe {
                weak: Weak::from_retained(&backing.buffer),
                deallocator_calls: Arc::clone(&callback_calls),
                deallocator_mismatches: Arc::clone(&callback_mismatches),
            };
            let source = DiagnosticGgufBlitSourceWindow { backing, probe };
            let source_probe = source.release_probe();
            assert_eq!(source.backing.page_size(), page_size);
            assert_eq!(source.backing.mapped_len(), bytes.len());
            assert_eq!(source.exposed_len(), bytes.len());
            assert_eq!(
                source.backing.buffer.contents().as_ptr(),
                mmap.as_ptr().cast_mut().cast::<c_void>()
            );
            let prefault = source.backing.prefault_read();
            assert_eq!(prefault.page_count, 2);
            assert_eq!(prefault.covered_bytes, bytes.len());
            let expected_checksum =
                u64::from(bytes[0]).rotate_left(5) ^ u64::from(bytes[page_size]);
            assert_eq!(prefault.checksum, expected_checksum);
            drop(mmap);
            assert!(weak.upgrade().is_some());

            let desc = TensorDesc {
                name: "view".to_string(),
                shape: vec![n_in as u64, n_out as u64],
                dtype: GgmlType::F32,
                shard_idx: 0,
                data_offset: 32,
                n_bytes: (weights.len() * 4) as u64,
            };
            let (eligibility, tensor) = source.backing.tensor(&desc).expect("tensor view");
            assert_eq!(eligibility, GgufBackingEligibility::Eligible);
            let tensor = tensor.expect("eligible tensor");
            assert_eq!(tensor.offset, 32);

            let dst = MetalTensor::zeros_f32(&ctx, vec![16]).expect("destination");
            let command = ctx.queue.commandBuffer().expect("command buffer");
            let blit = BlitEncoder::begin(&command);
            assert!(
                source
                    .encode_copy_to(&blit, 1, 32, &dst.buffer, dst.offset, 64)
                    .is_err()
            );
            assert!(
                source
                    .encode_copy_to(&blit, 0, bytes.len() as u64, &dst.buffer, dst.offset, 64)
                    .is_err()
            );
            source
                .encode_copy_to(&blit, 0, 32, &dst.buffer, dst.offset, 64)
                .expect("diagnostic source blit");
            blit.end();
            command.commit();
            crate::metal::wait_completed(&command).expect("command buffer completed");
            assert!(command.error().is_none(), "no-copy blit command failed");
            let got = unsafe {
                std::slice::from_raw_parts(dst.buffer.contents().as_ptr().cast::<u8>(), 64)
            };
            assert_eq!(got, &bytes[32..96]);
            drop(source);
            assert!(weak.upgrade().is_some());

            let x: Vec<f32> = (0..n_in).map(|index| index as f32 * 0.02 - 0.3).collect();
            let expected = crate::forward::mat_vec_pub(&weights, n_in, n_out, &x);
            let x = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_in as u64],
                GgmlType::F32,
            )
            .expect("input");
            let y = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("output");
            let command = ctx.queue.commandBuffer().expect("matvec command");
            let encoder = KernelEncoder::begin(&command);
            encode_mat_vec_f32(&ctx, &encoder, &tensor, &x, &y, n_in, n_out)
                .expect("nonzero-offset matvec");
            encoder.end();
            command.commit();
            crate::metal::wait_completed(&command).expect("command buffer completed");
            assert!(command.error().is_none(), "no-copy matvec command failed");
            let got = unsafe {
                std::slice::from_raw_parts(y.buffer.contents().as_ptr().cast::<f32>(), n_out)
            };
            let max_abs = got
                .iter()
                .zip(expected.iter())
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs < 1e-5, "nonzero-offset matvec max error {max_abs}");
            (weak, source_probe)
        });
        assert!(
            weak.upgrade().is_none(),
            "MTLBuffer deallocator must release its Arc<Mmap> capture"
        );
        assert_eq!(callback_calls.load(Ordering::Relaxed), 1);
        assert_eq!(callback_mismatches.load(Ordering::Relaxed), 0);
        assert_eq!(
            source_probe.report(),
            DiagnosticGgufBlitReleaseReport {
                source_alive: false,
                deallocator_calls: 1,
                deallocator_mismatches: 0,
            }
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn diagnostic_blit_sources_release_after_completed_command() {
        use std::io::Write;

        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        let page_size = host_page_size().expect("host page size");
        let bytes = (0..page_size * 2)
            .map(|index| index.wrapping_mul(31) as u8)
            .collect::<Vec<_>>();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "qwen-metal-diagnostic-blit-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::File::create(&path)
            .and_then(|mut file| file.write_all(&bytes))
            .expect("write diagnostic blit fixture");

        let (probe, mmap_weak, staging_weak, window_out, staging_out) =
            objc2::rc::autoreleasepool(|_| {
                let file = std::fs::File::open(&path).expect("open diagnostic blit fixture");
                // SAFETY: the fixture remains immutable and untruncated through
                // command completion and source release.
                let mmap = Arc::new(unsafe { Mmap::map(&file).expect("map fixture") });
                let mmap_weak = Arc::downgrade(&mmap);
                let geometry =
                    GgufBackingGeometry::new_window(0, mmap.len(), 0, mmap.len(), page_size, 32)
                        .expect("diagnostic geometry");
                let calls = Arc::new(AtomicUsize::new(0));
                let mismatches = Arc::new(AtomicUsize::new(0));
                let observed_calls = Arc::clone(&calls);
                let observed_mismatches = Arc::clone(&mismatches);
                let expected_pointer = mmap.as_ptr() as usize;
                let expected_length = mmap.len();
                let backing = ctx
                    .gguf_no_copy_geometry_with_observer(
                        Arc::clone(&mmap),
                        geometry,
                        move |pointer, length| {
                            if pointer.as_ptr() as usize != expected_pointer
                                || length != expected_length
                            {
                                observed_mismatches.fetch_add(1, Ordering::Relaxed);
                            }
                            observed_calls.fetch_add(1, Ordering::Release);
                        },
                    )
                    .expect("diagnostic source backing");
                let probe = DiagnosticGgufBlitReleaseProbe {
                    weak: Weak::from_retained(&backing.buffer),
                    deallocator_calls: calls,
                    deallocator_mismatches: mismatches,
                };
                let source = DiagnosticGgufBlitSourceWindow {
                    backing,
                    probe: probe.clone(),
                };
                drop(mmap);

                let staging = ctx.buffer_from(&bytes[96..160]).expect("staging source");
                let staging_weak = Weak::from_retained(&staging);
                let window_out = ctx.buffer_uninit(64).expect("window destination");
                let staging_out = ctx.buffer_uninit(64).expect("staging destination");
                let command = ctx.queue.commandBuffer().expect("diagnostic command");
                assert!(command.retainedReferences());
                let blit = BlitEncoder::try_begin(&command).expect("diagnostic blit encoder");
                source
                    .encode_copy_to(&blit, 0, 32, &window_out, 0, 64)
                    .expect("window copy");
                blit.copy_buffer(&staging, 0, &staging_out, 0, 64);
                blit.end();
                command.commit();
                crate::metal::wait_completed(&command).expect("command buffer completed");
                assert_eq!(
                    command.status(),
                    objc2_metal::MTLCommandBufferStatus::Completed
                );
                assert!(command.error().is_none());
                drop(command);
                drop(staging);
                drop(source);
                (probe, mmap_weak, staging_weak, window_out, staging_out)
            });

        assert_eq!(
            probe.report(),
            DiagnosticGgufBlitReleaseReport {
                source_alive: false,
                deallocator_calls: 1,
                deallocator_mismatches: 0,
            }
        );
        assert!(mmap_weak.upgrade().is_none());
        assert!(staging_weak.load().is_none());
        let window_got =
            unsafe { std::slice::from_raw_parts(window_out.contents().as_ptr().cast::<u8>(), 64) };
        let staging_got =
            unsafe { std::slice::from_raw_parts(staging_out.contents().as_ptr().cast::<u8>(), 64) };
        assert_eq!(window_got, &bytes[32..96]);
        assert_eq!(staging_got, &bytes[96..160]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn overlapping_mmap_windows_retain_storage_independently() {
        use std::io::Write;
        use std::sync::atomic::AtomicUsize;

        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        let page_size = host_page_size().expect("host page size");

        for reverse_drop_order in [false, true] {
            let mut bytes = vec![0u8; page_size * 4];
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = index.wrapping_mul(29).wrapping_add(7) as u8;
            }
            let mut path = std::env::temp_dir();
            path.push(format!(
                "qwen-metal-window-{}-{}-{}.bin",
                std::process::id(),
                reverse_drop_order,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos()
            ));
            std::fs::File::create(&path)
                .and_then(|mut file| file.write_all(&bytes))
                .expect("write window fixture");

            let calls = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
            let mismatches = Arc::new(AtomicUsize::new(0));
            let weak = objc2::rc::autoreleasepool(|_| {
                let file = std::fs::File::open(&path).expect("open window fixture");
                // SAFETY: the fixture file remains immutable and untruncated
                // while any mapping-backed Metal buffer exists.
                let mmap = Arc::new(unsafe { Mmap::map(&file).expect("map window fixture") });
                let weak = Arc::downgrade(&mmap);
                let first = f32_desc("first", 0, 0, (page_size as u64 + 32) / 4);
                let second = f32_desc("second", 0, page_size as u64 + 64, page_size as u64 / 4);
                let requests = [&first, &second, &first];
                let plan =
                    plan_retained_storage(&[bytes.len()], &requests, page_size, page_size * 2, 32)
                        .expect("overlapping-window plan");
                assert_retained_plan_invariants(&plan, &requests, &[bytes.len()]);
                assert_eq!(plan.windows.len(), 2);
                assert_eq!(plan.windows[0].mmap_offset, 0);
                assert_eq!(plan.windows[0].length, page_size * 2);
                assert_eq!(plan.windows[1].mmap_offset, page_size as u64);
                assert_eq!(plan.windows[1].length, page_size * 2);
                assert_eq!(
                    plan.entries[2].disposition,
                    RetainedStorageDisposition::Alias {
                        source_request_index: 0,
                    }
                );

                let make_backing = |window_index: usize| {
                    let window = &plan.windows[window_index];
                    let expected_pointer =
                        unsafe { mmap.as_ptr().add(window.mmap_offset as usize) as usize };
                    let expected_length = window.length;
                    let calls = Arc::clone(&calls);
                    let mismatches = Arc::clone(&mismatches);
                    ctx.gguf_no_copy_window_with_observer(
                        Arc::clone(&mmap),
                        window.shard_idx,
                        window.mmap_offset as usize,
                        window.length,
                        32,
                        move |pointer, length| {
                            calls[window_index].fetch_add(1, Ordering::Relaxed);
                            if pointer.as_ptr() as usize != expected_pointer
                                || length != expected_length
                            {
                                mismatches.fetch_add(1, Ordering::Relaxed);
                            }
                        },
                    )
                    .expect("realize retained window")
                };
                let first_backing = make_backing(0);
                let second_backing = make_backing(1);
                assert_eq!(first_backing.mmap_offset(), 0);
                assert_eq!(second_backing.mmap_offset(), page_size);

                let (_, first_tensor) = first_backing.tensor(&first).expect("first view");
                let first_tensor = first_tensor.expect("eligible first view");
                let (_, first_alias) = first_backing.tensor(&first).expect("first alias view");
                let first_alias = first_alias.expect("eligible first alias view");
                let (_, second_tensor) = second_backing.tensor(&second).expect("second view");
                let second_tensor = second_tensor.expect("eligible second view");
                let shared = f32_desc("shared-page", 0, page_size as u64 + 128, 8);
                let (_, shared_first) = first_backing.tensor(&shared).expect("shared first view");
                let shared_first = shared_first.expect("eligible shared first view");
                let (_, shared_second) =
                    second_backing.tensor(&shared).expect("shared second view");
                let shared_second = shared_second.expect("eligible shared second view");
                assert_eq!(first_tensor.offset, 0);
                assert_eq!(second_tensor.offset, 64);
                assert_eq!(
                    first_tensor.provenance(),
                    MetalTensorProvenance::RetainedGgufReadOnly
                );
                assert!(!first_tensor.is_writable());
                let element_subview = first_tensor.view_subrange(0, vec![8]);
                let byte_subview = first_tensor.view_bytes(0, vec![8]);
                assert_eq!(
                    element_subview.provenance(),
                    MetalTensorProvenance::RetainedGgufReadOnly
                );
                assert_eq!(
                    byte_subview.provenance(),
                    MetalTensorProvenance::RetainedGgufReadOnly
                );
                assert_eq!(first_alias.offset, first_tensor.offset);
                assert_eq!(
                    Retained::as_ptr(&first_alias.buffer),
                    Retained::as_ptr(&first_tensor.buffer)
                );

                let first_out =
                    MetalTensor::zeros_f32(&ctx, first.shape.clone()).expect("first destination");
                let second_out =
                    MetalTensor::zeros_f32(&ctx, second.shape.clone()).expect("second destination");
                let shared_first_out =
                    MetalTensor::zeros_f32(&ctx, shared.shape.clone()).expect("shared destination");
                let shared_second_out =
                    MetalTensor::zeros_f32(&ctx, shared.shape.clone()).expect("shared destination");
                {
                    let command = ctx.queue.commandBuffer().expect("window blit command");
                    let blit = BlitEncoder::begin(&command);
                    blit.copy_tensor(&first_tensor, &first_out);
                    blit.copy_tensor(&second_tensor, &second_out);
                    blit.copy_tensor(&shared_first, &shared_first_out);
                    blit.copy_tensor(&shared_second, &shared_second_out);
                    blit.end();
                    command.commit();
                    crate::metal::wait_completed(&command).expect("command buffer completed");
                    assert!(command.error().is_none(), "window blit command failed");
                }
                let first_got = unsafe {
                    std::slice::from_raw_parts(
                        first_out.buffer.contents().as_ptr().cast::<u8>(),
                        first.n_bytes as usize,
                    )
                };
                let second_got = unsafe {
                    std::slice::from_raw_parts(
                        second_out.buffer.contents().as_ptr().cast::<u8>(),
                        second.n_bytes as usize,
                    )
                };
                assert_eq!(first_got, &bytes[..first.n_bytes as usize]);
                let second_start = second.data_offset as usize;
                assert_eq!(
                    second_got,
                    &bytes[second_start..second_start + second.n_bytes as usize]
                );
                let shared_expected = &bytes
                    [shared.data_offset as usize..(shared.data_offset + shared.n_bytes) as usize];
                let shared_first_got = unsafe {
                    std::slice::from_raw_parts(
                        shared_first_out.buffer.contents().as_ptr().cast::<u8>(),
                        shared.n_bytes as usize,
                    )
                };
                let shared_second_got = unsafe {
                    std::slice::from_raw_parts(
                        shared_second_out.buffer.contents().as_ptr().cast::<u8>(),
                        shared.n_bytes as usize,
                    )
                };
                assert_eq!(shared_first_got, shared_expected);
                assert_eq!(shared_second_got, shared_expected);

                let command = ctx.queue.commandBuffer().expect("write guard command");
                let blit = BlitEncoder::begin(&command);
                let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    blit.copy_tensor(&first_out, &first_tensor);
                }));
                assert!(
                    write_result.is_err(),
                    "retained destination must fail closed"
                );
                blit.end();
                let compute = KernelEncoder::begin(&command);
                let note_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    compute.note_write(&first_tensor);
                }));
                assert!(note_result.is_err(), "retained write note must fail closed");
                compute.end();

                drop(mmap);
                assert!(weak.upgrade().is_some());
                if reverse_drop_order {
                    drop(second_backing);
                    drop(first_backing);
                } else {
                    drop(first_backing);
                    drop(second_backing);
                }
                assert_eq!(calls[0].load(Ordering::Relaxed), 0);
                assert_eq!(calls[1].load(Ordering::Relaxed), 0);
                if reverse_drop_order {
                    drop(second_tensor);
                    drop(shared_second);
                    assert!(weak.upgrade().is_some());
                    drop(element_subview);
                    drop(byte_subview);
                    drop(first_alias);
                    drop(shared_first);
                    drop(first_tensor);
                } else {
                    drop(element_subview);
                    drop(byte_subview);
                    drop(first_alias);
                    drop(shared_first);
                    drop(first_tensor);
                    assert!(weak.upgrade().is_some());
                    drop(shared_second);
                    drop(second_tensor);
                }
                weak
            });
            assert!(
                weak.upgrade().is_none(),
                "last window must release the mmap"
            );
            assert_eq!(calls[0].load(Ordering::Relaxed), 1);
            assert_eq!(calls[1].load(Ordering::Relaxed), 1);
            assert_eq!(mismatches.load(Ordering::Relaxed), 0);
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn mat_mat_qk_threadgroup_memory_matches_full_tile_policy() {
        assert_eq!(
            mat_mat_qk_threadgroup_memory_with_policy(5120, 16, 16, false),
            8192
        );
        assert_eq!(
            mat_mat_qk_threadgroup_memory_with_policy(5120, 16, 16, true),
            5120
        );
        assert_eq!(
            mat_mat_qk_threadgroup_memory_with_policy(5120, 32, 32, true),
            6144
        );
        assert_eq!(
            mat_mat_qk_threadgroup_memory_with_policy(5120, 1024, 32, true),
            6144
        );
        assert_eq!(
            mat_mat_qk_threadgroup_memory_with_policy(5121, 32, 32, true),
            8192
        );
        assert_eq!(
            mat_mat_qk_threadgroup_memory_with_policy(5120, 31, 32, true),
            8192
        );
    }

    #[test]
    fn mat_vec_and_mat_mat_half_weights_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(path, dtype) in &[
            ("/Users/tito/models/Qwen3.5-0.8B.f16.gguf", GgmlType::F16),
            ("/Users/tito/models/Qwen3.5-0.8B-BF16.gguf", GgmlType::BF16),
        ] {
            if !std::path::Path::new(path).exists() {
                eprintln!("[half-weight] skipped missing fixture {path}");
                continue;
            }
            let g = crate::gguf::GgufFile::open(path).expect("open");
            let w = g
                .tensors
                .iter()
                .find(|t| {
                    t.name == "blk.0.ffn_gate.weight" && t.dtype == dtype && t.shape.len() == 2
                })
                .expect("missing half test tensor");
            let n_in = w.shape[0] as usize;
            let n_out = w.shape[1] as usize;
            eprintln!("[half-weight {dtype:?}] {} shape=[{n_in}, {n_out}]", w.name);

            let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
            let w_t =
                MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
                    .expect("weight tensor");

            let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
            let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
            one_shot(&ctx, |enc| match dtype {
                GgmlType::F16 => encode_mat_vec_f16_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
                GgmlType::BF16 => encode_mat_vec_bf16_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
                _ => unreachable!(),
            })
            .expect("mat_vec encode");
            let gpu = read_back_f32(&y_t.buffer, n_out);
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[half-weight {dtype:?} mat_vec] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-3, "{dtype:?} mat_vec max_abs={max_abs}");

            for &n_query in &[1usize, 16, 32, 33] {
                let x_pack: Vec<f32> = (0..n_query * n_in)
                    .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                    .collect();
                let mut cpu_pack = vec![0.0f32; n_query * n_out];
                for q in 0..n_query {
                    let row = &x_pack[q * n_in..(q + 1) * n_in];
                    let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                    cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
                }
                let x_pack_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&x_pack),
                    vec![n_query as u64, n_in as u64],
                    GgmlType::F32,
                )
                .expect("x pack tensor");
                let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                    .expect("y pack tensor");
                one_shot(&ctx, |enc| match dtype {
                    GgmlType::F16 => encode_mat_mat_f16_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    GgmlType::BF16 => encode_mat_mat_bf16_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    _ => unreachable!(),
                })
                .expect("mat_mat encode");
                let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
                let max_abs = gpu_pack
                    .iter()
                    .zip(cpu_pack.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "[half-weight {dtype:?} mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}"
                );
                assert!(max_abs < 1e-3, "{dtype:?} mat_mat max_abs={max_abs}");
            }
        }
    }

    #[test]
    fn mat_mat_bf16_bfloat_act_matches_rounded_cpu() {
        fn round_to_bf16_f32(x: f32) -> f32 {
            let bits = x.to_bits();
            let lsb = (bits >> 16) & 1;
            f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
        }

        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.5-0.8B-BF16.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[bf16-bfloat-act] skipped missing fixture {path}");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let w = g
            .tensors
            .iter()
            .find(|t| {
                t.name == "blk.0.ffn_gate.weight" && t.dtype == GgmlType::BF16 && t.shape.len() == 2
            })
            .expect("missing BF16 test tensor");
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
        let w_t = MetalTensor::from_bytes(
            &ctx,
            g.slice(w),
            vec![n_in as u64, n_out as u64],
            GgmlType::BF16,
        )
        .expect("weight tensor");

        for &n_out_case in &[70usize, n_out] {
            let weight_case = &weight_f32[..n_in * n_out_case];
            for &n_query in &[1usize, 16, 32, 33] {
                let x_pack: Vec<f32> = (0..n_query * n_in)
                    .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                    .collect();
                let x_bf16: Vec<f32> = x_pack.iter().copied().map(round_to_bf16_f32).collect();
                let mut cpu_pack = vec![0.0f32; n_query * n_out_case];
                for q in 0..n_query {
                    let row = &x_bf16[q * n_in..(q + 1) * n_in];
                    let out = crate::forward::mat_vec_pub(weight_case, n_in, n_out_case, row);
                    cpu_pack[q * n_out_case..(q + 1) * n_out_case].copy_from_slice(&out);
                }
                let x_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&x_pack),
                    vec![n_query as u64, n_in as u64],
                    GgmlType::F32,
                )
                .expect("x tensor");
                let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out_case) as u64])
                    .expect("y tensor");
                one_shot(&ctx, |enc| {
                    encode_mat_mat_bf16_bfloat_act_f32(
                        &ctx, enc, &w_t, &x_t, &y_t, n_in, n_out_case, n_query,
                    )
                })
                .expect("approx bf16 matmat encode");
                let gpu = read_back_f32(&y_t.buffer, n_query * n_out_case);
                let dot: f64 = gpu
                    .iter()
                    .zip(cpu_pack.iter())
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum();
                let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                let nc: f64 = cpu_pack.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
                let max_abs = gpu
                    .iter()
                    .zip(cpu_pack.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "[bf16-bfloat-act n_out={n_out_case} n_query={n_query}] \
                     cos={cos:.6} max|Delta|={max_abs:.2e}"
                );
                assert!(cos > 0.99999, "cos={cos}");
                assert!(max_abs < 1e-3, "max_abs={max_abs}");
            }
        }
    }

    #[test]
    fn mat_vec_and_mat_mat_q4_legacy_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(path, dtype) in &[
            ("/Users/tito/models/Qwen3.5-0.8B-Q4_0.gguf", GgmlType::Q4_0),
            ("/Users/tito/models/Qwen3.5-0.8B-Q4_1.gguf", GgmlType::Q4_1),
        ] {
            if !std::path::Path::new(path).exists() {
                eprintln!("[q4-legacy] skipped missing fixture {path}");
                continue;
            }
            let g = crate::gguf::GgufFile::open(path).expect("open");
            let w = g
                .tensors
                .iter()
                .find(|t| {
                    t.name == "blk.0.ffn_gate.weight"
                        && t.dtype == dtype
                        && t.shape.len() == 2
                        && t.shape[0] % 32 == 0
                })
                .expect("missing q4 legacy test tensor");
            let n_in = w.shape[0] as usize;
            let n_out = w.shape[1] as usize;
            eprintln!("[q4-legacy {dtype:?}] {} shape=[{n_in}, {n_out}]", w.name);

            let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
            let w_t =
                MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
                    .expect("weight tensor");

            let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
            let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
            one_shot(&ctx, |enc| match dtype {
                GgmlType::Q4_0 => encode_mat_vec_q4_0_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
                GgmlType::Q4_1 => encode_mat_vec_q4_1_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out),
                _ => unreachable!(),
            })
            .expect("mat_vec encode");
            let gpu = read_back_f32(&y_t.buffer, n_out);
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[q4-legacy {dtype:?} mat_vec] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "{dtype:?} mat_vec max_abs={max_abs}");

            for &n_query in &[1usize, 16, 32, 33] {
                let x_pack: Vec<f32> = (0..n_query * n_in)
                    .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                    .collect();
                let mut cpu_pack = vec![0.0f32; n_query * n_out];
                for q in 0..n_query {
                    let row = &x_pack[q * n_in..(q + 1) * n_in];
                    let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                    cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
                }
                let x_pack_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&x_pack),
                    vec![n_query as u64, n_in as u64],
                    GgmlType::F32,
                )
                .expect("x pack tensor");
                let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                    .expect("y pack tensor");
                one_shot(&ctx, |enc| match dtype {
                    GgmlType::Q4_0 => encode_mat_mat_q4_0_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    GgmlType::Q4_1 => encode_mat_mat_q4_1_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    _ => unreachable!(),
                })
                .expect("mat_mat encode");
                let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
                let max_abs = gpu_pack
                    .iter()
                    .zip(cpu_pack.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "[q4-legacy {dtype:?} mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}"
                );
                assert!(max_abs < 1e-2, "{dtype:?} mat_mat max_abs={max_abs}");
            }
        }
    }

    #[test]
    fn mat_vec_and_mat_mat_q3_k_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.5-0.8B-Q3_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[q3_k] skipped missing fixture {path}");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let w = g
            .tensors
            .iter()
            .find(|t| {
                t.name == "blk.0.ffn_gate.weight"
                    && t.dtype == GgmlType::Q3_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
            })
            .expect("missing q3_k test tensor");
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        eprintln!("[q3_k] {} shape=[{n_in}, {n_out}]", w.name);

        let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
        let w_t = MetalTensor::from_bytes(
            &ctx,
            g.slice(w),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q3_K,
        )
        .expect("weight tensor");

        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
        one_shot(&ctx, |enc| {
            encode_mat_vec_q3_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
        })
        .expect("mat_vec encode");
        let gpu = read_back_f32(&y_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q3_k mat_vec] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-2, "Q3_K mat_vec max_abs={max_abs}");

        for &n_query in &[1usize, 16, 32] {
            let x_pack: Vec<f32> = (0..n_query * n_in)
                .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                .collect();
            let mut cpu_pack = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row = &x_pack[q * n_in..(q + 1) * n_in];
                let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
            }
            let x_pack_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_pack),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x pack tensor");
            let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                .expect("y pack tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_q3_k_f32(&ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query)
            })
            .expect("mat_mat encode");
            let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
            let max_abs = gpu_pack
                .iter()
                .zip(cpu_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[q3_k mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "Q3_K mat_mat max_abs={max_abs}");
        }
    }

    #[test]
    fn mat_vec_and_mat_mat_dense_iq2_s_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.5-4B-UD-IQ2_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[dense-iq2_s] skipped missing fixture {path}");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let w = match g.tensors.iter().find(|t| {
            t.name.starts_with("blk.")
                && t.name.ends_with(".weight")
                && t.dtype == GgmlType::IQ2_S
                && t.shape.len() == 2
                && t.shape[0] % 256 == 0
        }) {
            Some(t) => t,
            None => {
                eprintln!("[dense-iq2_s] skipped missing IQ2_S tensor in {path}");
                return;
            }
        };
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        eprintln!("[dense-iq2_s] {} shape=[{n_in}, {n_out}]", w.name);

        let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
        let w_t = MetalTensor::from_bytes(
            &ctx,
            g.slice(w),
            vec![n_in as u64, n_out as u64],
            GgmlType::IQ2_S,
        )
        .expect("weight tensor");

        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
        one_shot(&ctx, |enc| {
            encode_mat_vec_iq2_s_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
        })
        .expect("mat_vec encode");
        let gpu = read_back_f32(&y_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[dense-iq2_s mat_vec] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-2, "IQ2_S mat_vec max_abs={max_abs}");

        for &n_query in &[1usize, 16] {
            let x_pack: Vec<f32> = (0..n_query * n_in)
                .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                .collect();
            let mut cpu_pack = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row = &x_pack[q * n_in..(q + 1) * n_in];
                let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
            }
            let x_pack_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_pack),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x pack tensor");
            let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                .expect("y pack tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_iq2_s_f32(
                    &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                )
            })
            .expect("mat_mat encode");
            let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
            let max_abs = gpu_pack
                .iter()
                .zip(cpu_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[dense-iq2_s mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "IQ2_S mat_mat max_abs={max_abs}");
        }
    }

    #[test]
    fn mat_vec_and_mat_mat_dense_iq3_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let fixtures = [
            (
                "/Users/tito/models/Qwen3.5-4B-UD-Q2_K_XL.gguf",
                GgmlType::IQ3_XXS,
            ),
            (
                "/Users/tito/models/Qwen3.5-4B-UD-Q2_K_XL.gguf",
                GgmlType::IQ3_S,
            ),
        ];
        for &(path, dtype) in &fixtures {
            if !std::path::Path::new(path).exists() {
                eprintln!("[dense-iq3 {dtype:?}] skipped missing fixture {path}");
                continue;
            }
            let g = crate::gguf::GgufFile::open(path).expect("open");
            let w = match g.tensors.iter().find(|t| {
                t.name.starts_with("blk.")
                    && t.name.ends_with(".weight")
                    && t.dtype == dtype
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
            }) {
                Some(t) => t,
                None => {
                    eprintln!("[dense-iq3 {dtype:?}] skipped missing dtype tensor in {path}");
                    continue;
                }
            };
            let n_in = w.shape[0] as usize;
            let n_out = w.shape[1] as usize;
            eprintln!("[dense-iq3 {dtype:?}] {} shape=[{n_in}, {n_out}]", w.name);

            let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
            let w_t =
                MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
                    .expect("weight tensor");

            let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
            let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
            one_shot(&ctx, |enc| match dtype {
                GgmlType::IQ3_XXS => {
                    encode_mat_vec_iq3_xxs_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
                }
                GgmlType::IQ3_S => {
                    encode_mat_vec_iq3_s_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
                }
                _ => unreachable!(),
            })
            .expect("mat_vec encode");
            let gpu = read_back_f32(&y_t.buffer, n_out);
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[dense-iq3 {dtype:?} mat_vec] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "{dtype:?} mat_vec max_abs={max_abs}");

            for &n_query in &[1usize, 16] {
                let x_pack: Vec<f32> = (0..n_query * n_in)
                    .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                    .collect();
                let mut cpu_pack = vec![0.0f32; n_query * n_out];
                for q in 0..n_query {
                    let row = &x_pack[q * n_in..(q + 1) * n_in];
                    let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                    cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
                }
                let x_pack_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&x_pack),
                    vec![n_query as u64, n_in as u64],
                    GgmlType::F32,
                )
                .expect("x pack tensor");
                let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                    .expect("y pack tensor");
                one_shot(&ctx, |enc| match dtype {
                    GgmlType::IQ3_XXS => encode_mat_mat_iq3_xxs_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    GgmlType::IQ3_S => encode_mat_mat_iq3_s_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    _ => unreachable!(),
                })
                .expect("mat_mat encode");
                let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
                let max_abs = gpu_pack
                    .iter()
                    .zip(cpu_pack.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "[dense-iq3 {dtype:?} mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}"
                );
                assert!(max_abs < 1e-2, "{dtype:?} mat_mat max_abs={max_abs}");
            }
        }
    }

    #[test]
    fn mat_vec_and_mat_mat_q2_k_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.5-0.8B.Q2_K.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[q2_k] skipped missing fixture {path}");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let w = g
            .tensors
            .iter()
            .find(|t| {
                t.name == "blk.0.ffn_gate.weight"
                    && t.dtype == GgmlType::Q2_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
            })
            .expect("missing q2_k test tensor");
        let n_in = w.shape[0] as usize;
        let n_out = w.shape[1] as usize;
        eprintln!("[q2_k] {} shape=[{n_in}, {n_out}]", w.name);

        let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
        let w_t = MetalTensor::from_bytes(
            &ctx,
            g.slice(w),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q2_K,
        )
        .expect("weight tensor");

        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
        one_shot(&ctx, |enc| {
            encode_mat_vec_q2_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
        })
        .expect("mat_vec encode");
        let gpu = read_back_f32(&y_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q2_k mat_vec] max|Delta|={max_abs:.2e}");
        assert!(max_abs < 1e-2, "Q2_K mat_vec max_abs={max_abs}");

        for &n_query in &[1usize, 16, 32] {
            let x_pack: Vec<f32> = (0..n_query * n_in)
                .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                .collect();
            let mut cpu_pack = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row = &x_pack[q * n_in..(q + 1) * n_in];
                let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
            }
            let x_pack_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_pack),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x pack tensor");
            let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                .expect("y pack tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_q2_k_f32(&ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query)
            })
            .expect("mat_mat encode");
            let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
            let max_abs = gpu_pack
                .iter()
                .zip(cpu_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[q2_k mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "Q2_K mat_mat max_abs={max_abs}");
        }
    }

    #[test]
    fn mat_vec_and_mat_mat_iq4_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(path, dtype) in &[
            (
                "/Users/tito/models/Qwen3.5-0.8B-IQ4_NL.gguf",
                GgmlType::IQ4_NL,
            ),
            (
                "/Users/tito/models/Qwen3.5-0.8B-IQ4_XS.gguf",
                GgmlType::IQ4_XS,
            ),
        ] {
            if !std::path::Path::new(path).exists() {
                eprintln!("[iq4] skipped missing fixture {path}");
                continue;
            }
            let g = crate::gguf::GgufFile::open(path).expect("open");
            let align = if dtype == GgmlType::IQ4_NL { 32 } else { 256 };
            let w = g
                .tensors
                .iter()
                .find(|t| {
                    t.name == "blk.0.ffn_gate.weight"
                        && t.dtype == dtype
                        && t.shape.len() == 2
                        && t.shape[0] % align == 0
                })
                .expect("missing iq4 test tensor");
            let n_in = w.shape[0] as usize;
            let n_out = w.shape[1] as usize;
            eprintln!("[iq4 {dtype:?}] {} shape=[{n_in}, {n_out}]", w.name);

            let weight_f32 = crate::codec::dequant_to_f32(w, g.slice(w)).expect("dequant");
            let w_t =
                MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
                    .expect("weight tensor");

            let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
            let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y tensor");
            one_shot(&ctx, |enc| match dtype {
                GgmlType::IQ4_NL => {
                    encode_mat_vec_iq4_nl_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
                }
                GgmlType::IQ4_XS => {
                    encode_mat_vec_iq4_xs_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
                }
                _ => unreachable!(),
            })
            .expect("mat_vec encode");
            let gpu = read_back_f32(&y_t.buffer, n_out);
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[iq4 {dtype:?} mat_vec] max|Delta|={max_abs:.2e}");
            assert!(max_abs < 1e-2, "{dtype:?} mat_vec max_abs={max_abs}");

            for &n_query in &[1usize, 16, 32] {
                let x_pack: Vec<f32> = (0..n_query * n_in)
                    .map(|i| ((i % 17) as f32 - 8.0) * 1e-2)
                    .collect();
                let mut cpu_pack = vec![0.0f32; n_query * n_out];
                for q in 0..n_query {
                    let row = &x_pack[q * n_in..(q + 1) * n_in];
                    let out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row);
                    cpu_pack[q * n_out..(q + 1) * n_out].copy_from_slice(&out);
                }
                let x_pack_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&x_pack),
                    vec![n_query as u64, n_in as u64],
                    GgmlType::F32,
                )
                .expect("x pack tensor");
                let y_pack_t = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_out) as u64])
                    .expect("y pack tensor");
                one_shot(&ctx, |enc| match dtype {
                    GgmlType::IQ4_NL => encode_mat_mat_iq4_nl_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    GgmlType::IQ4_XS => encode_mat_mat_iq4_xs_f32(
                        &ctx, enc, &w_t, &x_pack_t, &y_pack_t, n_in, n_out, n_query,
                    ),
                    _ => unreachable!(),
                })
                .expect("mat_mat encode");
                let gpu_pack = read_back_f32(&y_pack_t.buffer, n_query * n_out);
                let max_abs = gpu_pack
                    .iter()
                    .zip(cpu_pack.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!("[iq4 {dtype:?} mat_mat n_query={n_query}] max|Delta|={max_abs:.2e}");
                assert!(max_abs < 1e-2, "{dtype:?} mat_mat max_abs={max_abs}");
            }
        }
    }

    /// H5.3b.0 gate: lifted Q4_K mat-mat correctness against
    /// (a) CPU mat-mat oracle  (b) N successive mat-vec calls
    /// (c) col-major output layout sanity.
    ///
    /// Per codex H5.3b mid-impl review: lifted llama mat-mat is NOT
    /// bit-exact with N mat-vec because it stages activations through
    /// half before float accumulation. Gate thresholds:
    ///   * vs CPU mat-mat oracle (same half-staging math): cos ≥ 0.9999
    ///     and max|Δ| ≤ 0.01 (Q4_K dequant noise dominates the diff)
    ///   * vs N mat-vec: cos ≥ 0.999 per row (relaxed; half-vs-float
    ///     accumulation diff)
    ///   * layout: dst[row + col * M] stride explicitly probed
    ///
    /// Uses real Q4_K weight from the 27B GGUF; N_QUERY ∈ {1, 16, 32}
    /// to exercise both partial-tile path (N=1, N=16) and full-tile
    /// path (N=32).
    #[test]
    fn mat_mat_q4_k_matches_cpu_and_mat_vec() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
        if !std::path::Path::new(path).exists() {
            eprintln!("[mat_mat_q4_k] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        // Pick a Q4_K tensor with shape compatible with mat-mat tiling
        // (n_in % 32 == 0, n_out % 64 == 0 for the lifted tile).
        let q4k = g
            .tensors
            .iter()
            .find(|t| {
                t.name.starts_with("blk.0.")
                    && t.dtype == GgmlType::Q4_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
                    && t.shape[1] % 64 == 0
            })
            .expect("no Q4_K tensor with compatible shape");
        let n_in = q4k.shape[0] as usize;
        let n_out = q4k.shape[1] as usize;
        eprintln!(
            "[mat_mat_q4_k-test] tensor={} shape=[n_in={n_in}, n_out={n_out}]",
            q4k.name
        );

        let weight_f32 = crate::codec::dequant_to_f32(q4k, g.slice(q4k)).expect("dequant");
        let weight_bytes = g.slice(q4k);

        let n_queries: &[usize] = if mat_mat_q4_k_n64_enabled() {
            &[1, 16, 32, 64]
        } else {
            &[1, 16, 32]
        };
        for &n_query in n_queries {
            // Activation matrix [n_query, n_in] row-major, deterministic
            // pseudo-random fill.
            let mut x = vec![0.0f32; n_query * n_in];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i % 13) as f32 - 6.0) * 1e-2;
            }

            // -- CPU oracle: row-major output `[n_query, n_out]`
            //    y[q, o] = sum_i W[o, i] * x[q, i]
            //    We compute it via N successive mat_vec_pub calls.
            let mut cpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row_in = &x[q * n_in..(q + 1) * n_in];
                let row_out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row_in);
                cpu_row_major[q * n_out..(q + 1) * n_out].copy_from_slice(&row_out);
            }

            // -- GPU mat-mat: output `[n_out, n_query]` COL-major
            //    i.e. y[r + c * n_out]. Allocate raw n_out*n_query f32.
            let w_t = MetalTensor::from_bytes(
                &ctx,
                weight_bytes,
                vec![n_in as u64, n_out as u64],
                GgmlType::Q4_K,
            )
            .expect("weight tensor");
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_q4_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out, n_query)
            })
            .expect("mat_mat encode");

            let gpu_col_major = read_back_f32(&y_t.buffer, n_out * n_query);

            // -- Reshape: convert col-major [n_out, n_query] →
            //    row-major [n_query, n_out] for comparison.
            //    cell (q, o) lives at gpu_col_major[o + q * n_out]
            //                  vs   cpu_row_major[q * n_out + o].
            let mut gpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                for o in 0..n_out {
                    gpu_row_major[q * n_out + o] = gpu_col_major[o + q * n_out];
                }
            }

            // -- Per-row cosine + max|Δ|.
            let mut min_cos = f64::INFINITY;
            let mut max_abs = 0.0f32;
            for q in 0..n_query {
                let cpu_row = &cpu_row_major[q * n_out..(q + 1) * n_out];
                let gpu_row = &gpu_row_major[q * n_out..(q + 1) * n_out];
                let mut dot = 0.0f64;
                let mut np = 0.0f64;
                let mut nc = 0.0f64;
                for i in 0..n_out {
                    let p = gpu_row[i] as f64;
                    let c = cpu_row[i] as f64;
                    dot += p * c;
                    np += p * p;
                    nc += c * c;
                    let d = (gpu_row[i] - cpu_row[i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
                let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
                if cos < min_cos {
                    min_cos = cos;
                }
            }
            eprintln!(
                "[mat_mat_q4_k n_query={n_query}] min_cos={min_cos:.6} \
                 max|Δ|={max_abs:.3e}"
            );
            // Per H5.3b plan rev 6: cos ≥ 0.999 vs N mat-vec (relaxed
            // because half-staging in lifted kernel). max|Δ| ≤ 0.01
            // (Q4_K dequant + half-staging noise; same order as Q4_K
            // mat-vec test threshold).
            assert!(
                min_cos >= 0.999,
                "n_query={n_query}: min cos {min_cos} < 0.999"
            );
            assert!(
                max_abs < 1e-2,
                "n_query={n_query}: max|Δ| {max_abs} >= 1e-2"
            );

            // -- Layout sanity (codex Q7 failure-mode mitigation):
            //    explicitly assert col-major dst stride. Pick three
            //    cells (0,0), (1, n_query/2), (n_out-1, n_query-1) and
            //    check they live where the docs say they live.
            //    cell (r, c) at index `r + c * n_out` in gpu_col_major.
            for &(r, c) in &[
                (0usize, 0usize),
                (1usize, n_query / 2),
                (n_out - 1, n_query - 1),
            ] {
                let raw = gpu_col_major[r + c * n_out];
                let row_major_view = gpu_row_major[c * n_out + r];
                assert_eq!(
                    raw.to_bits(),
                    row_major_view.to_bits(),
                    "layout sanity: gpu_col_major[r={r}+c={c}*n_out={n_out}] should equal \
                     reshape→row_major[c={c}*n_out+r={r}]; got {raw} vs {row_major_view}"
                );
            }
        }
    }

    /// H5.3b.6 gate: Q6_K mat-mat parity vs N successive mat-vec.
    /// Includes N_QUERY=64/128 so the large-N N64 prompt tile is exercised.
    #[test]
    fn mat_mat_q6_k_matches_cpu_and_mat_vec() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[mat_mat_q6_k] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let q6k = g
            .tensors
            .iter()
            .find(|t| {
                t.name.starts_with("blk.0.")
                    && t.dtype == GgmlType::Q6_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
                    && t.shape[1] % 64 == 0
            })
            .expect("no Q6_K tensor with compatible shape");
        let n_in = q6k.shape[0] as usize;
        let n_out = q6k.shape[1] as usize;
        eprintln!(
            "[mat_mat_q6_k-test] tensor={} shape=[n_in={n_in}, n_out={n_out}]",
            q6k.name
        );

        let weight_f32 = crate::codec::dequant_to_f32(q6k, g.slice(q6k)).expect("dequant");
        let weight_bytes = g.slice(q6k);

        for &n_query in &[1usize, 16, 32, 64, 128] {
            let mut x = vec![0.0f32; n_query * n_in];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i % 13) as f32 - 6.0) * 1e-2;
            }

            // CPU oracle via N mat_vec_pub.
            let mut cpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row_in = &x[q * n_in..(q + 1) * n_in];
                let row_out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row_in);
                cpu_row_major[q * n_out..(q + 1) * n_out].copy_from_slice(&row_out);
            }

            let w_t = MetalTensor::from_bytes(
                &ctx,
                weight_bytes,
                vec![n_in as u64, n_out as u64],
                GgmlType::Q6_K,
            )
            .expect("weight tensor");
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_q6_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out, n_query)
            })
            .expect("mat_mat encode");

            let gpu_flat = read_back_f32(&y_t.buffer, n_out * n_query);

            // Reshape: bit-equivalent col-major [n_out, n_query] →
            // row-major [n_query, n_out] (same byte ordering trick as Q4_K).
            let mut gpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                for o in 0..n_out {
                    gpu_row_major[q * n_out + o] = gpu_flat[o + q * n_out];
                }
            }

            let mut min_cos = f64::INFINITY;
            let mut max_abs = 0.0f32;
            for q in 0..n_query {
                let cpu_row = &cpu_row_major[q * n_out..(q + 1) * n_out];
                let gpu_row = &gpu_row_major[q * n_out..(q + 1) * n_out];
                let mut dot = 0.0f64;
                let mut np = 0.0f64;
                let mut nc = 0.0f64;
                for i in 0..n_out {
                    let p = gpu_row[i] as f64;
                    let c = cpu_row[i] as f64;
                    dot += p * c;
                    np += p * p;
                    nc += c * c;
                    let d = (gpu_row[i] - cpu_row[i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
                let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
                if cos < min_cos {
                    min_cos = cos;
                }
            }
            eprintln!(
                "[mat_mat_q6_k n_query={n_query}] min_cos={min_cos:.6} \
                 max|Δ|={max_abs:.3e}"
            );
            assert!(
                min_cos >= 0.999,
                "n_query={n_query}: min cos {min_cos} < 0.999"
            );
            assert!(
                max_abs < 1e-2,
                "n_query={n_query}: max|Δ| {max_abs} >= 1e-2"
            );
        }
    }

    /// v0.73a.0 gate: Q5_K mat-mat parity vs N successive Q5_K mat-vec.
    /// Includes N_QUERY=64/128 so the large-N N64 prompt tile is exercised.
    #[test]
    fn mat_mat_q5_k_matches_cpu_and_mat_vec() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[mat_mat_q5_k] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let q5k = g
            .tensors
            .iter()
            .find(|t| {
                t.name.starts_with("blk.0.")
                    && t.dtype == GgmlType::Q5_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
                    && t.shape[1] % 64 == 0
            })
            .expect("no Q5_K tensor with compatible shape");
        let n_in = q5k.shape[0] as usize;
        let n_out = q5k.shape[1] as usize;
        eprintln!(
            "[mat_mat_q5_k-test] tensor={} shape=[n_in={n_in}, n_out={n_out}]",
            q5k.name
        );

        let weight_f32 = crate::codec::dequant_to_f32(q5k, g.slice(q5k)).expect("dequant");
        let weight_bytes = g.slice(q5k);

        for &n_query in &[1usize, 16, 32, 64, 128] {
            let mut x = vec![0.0f32; n_query * n_in];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i % 13) as f32 - 6.0) * 1e-2;
            }

            // CPU oracle via N mat_vec_pub.
            let mut cpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row_in = &x[q * n_in..(q + 1) * n_in];
                let row_out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row_in);
                cpu_row_major[q * n_out..(q + 1) * n_out].copy_from_slice(&row_out);
            }

            let w_t = MetalTensor::from_bytes(
                &ctx,
                weight_bytes,
                vec![n_in as u64, n_out as u64],
                GgmlType::Q5_K,
            )
            .expect("weight tensor");
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_q5_k_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out, n_query)
            })
            .expect("mat_mat encode");

            let gpu_flat = read_back_f32(&y_t.buffer, n_out * n_query);

            // Reshape: bit-equivalent col-major [n_out, n_query] →
            // row-major [n_query, n_out] (same byte ordering trick as Q4_K/Q6_K).
            let mut gpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                for o in 0..n_out {
                    gpu_row_major[q * n_out + o] = gpu_flat[o + q * n_out];
                }
            }

            let mut min_cos = f64::INFINITY;
            let mut max_abs = 0.0f32;
            for q in 0..n_query {
                let cpu_row = &cpu_row_major[q * n_out..(q + 1) * n_out];
                let gpu_row = &gpu_row_major[q * n_out..(q + 1) * n_out];
                let mut dot = 0.0f64;
                let mut np = 0.0f64;
                let mut nc = 0.0f64;
                for i in 0..n_out {
                    let p = gpu_row[i] as f64;
                    let c = cpu_row[i] as f64;
                    dot += p * c;
                    np += p * p;
                    nc += c * c;
                    let d = (gpu_row[i] - cpu_row[i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
                let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
                if cos < min_cos {
                    min_cos = cos;
                }
            }
            eprintln!(
                "[mat_mat_q5_k n_query={n_query}] min_cos={min_cos:.6} \
                 max|Δ|={max_abs:.3e}"
            );
            assert!(
                min_cos >= 0.999,
                "n_query={n_query}: min cos {min_cos} < 0.999"
            );
            assert!(
                max_abs < 1e-2,
                "n_query={n_query}: max|Δ| {max_abs} >= 1e-2"
            );

            // Layout sanity (codex Q7 mitigation): col-major dst stride
            // `dst[r + c*n_out]` must equal row-major view at three
            // corner cells. Catches a transposed write (which would
            // pass cosine within a single row but corrupt downstream
            // chained mat-mats).
            for &(r, c) in &[
                (0usize, 0usize),
                (1usize, n_query / 2),
                (n_out - 1, n_query - 1),
            ] {
                let raw = gpu_flat[r + c * n_out];
                let row_major_view = gpu_row_major[c * n_out + r];
                assert_eq!(
                    raw.to_bits(),
                    row_major_view.to_bits(),
                    "layout sanity n_query={n_query}: (r={r}, c={c})"
                );
            }
        }
    }

    /// v0.73b.0 gate: Q8_0 mat-mat correctness. Uses a real Q8_0
    /// weight from the spiritbuun DFlash drafter GGUF
    /// (`blk.0.ffn_down.weight`, shape `[17408, 5120]` — large weight,
    /// hits both the whole-M-tile and partial-N paths). Same playbook
    /// as the Q4_K (v0.63), Q6_K (v0.67), Q5_K (v0.73a.0) gates:
    /// per-row cosine ≥ 0.999 across N_QUERY ∈ {1, 16, 32, 64, 128},
    /// max|Δ| ≤ 1e-2, explicit col-major dst layout sanity probe.
    ///
    /// Q8_0's structurally-simpler dequant (`int8 * scale`) typically
    /// produces TIGHTER cosine than Q4_K/Q5_K/Q6_K mat-mat (which lose
    /// precision in nibble packing + scale folding). Expect cos very
    /// close to 1.000000.
    #[test]
    fn mat_mat_q8_0_matches_cpu_and_mat_vec() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = crate::test_fixtures::DFLASH_DRAFT_36_Q8_0.path();
        if !std::path::Path::new(path).exists() {
            eprintln!("[mat_mat_q8_0] skipped — drafter GGUF missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let q8 = g
            .tensors
            .iter()
            .find(|t| {
                t.name.starts_with("blk.0.")
                    && t.dtype == GgmlType::Q8_0
                    && t.shape.len() == 2
                    && t.shape[0] % 32 == 0
                    && t.shape[1] % 64 == 0
            })
            .expect("no Q8_0 tensor with compatible shape in drafter blk.0");
        let n_in = q8.shape[0] as usize;
        let n_out = q8.shape[1] as usize;
        eprintln!(
            "[mat_mat_q8_0-test] tensor={} shape=[n_in={n_in}, n_out={n_out}]",
            q8.name
        );

        let weight_f32 = crate::codec::dequant_to_f32(q8, g.slice(q8)).expect("dequant");
        let weight_bytes = g.slice(q8);

        for &n_query in &[1usize, 16, 32, 64, 128] {
            let mut x = vec![0.0f32; n_query * n_in];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i % 13) as f32 - 6.0) * 1e-2;
            }

            let mut cpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                let row_in = &x[q * n_in..(q + 1) * n_in];
                let row_out = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, row_in);
                cpu_row_major[q * n_out..(q + 1) * n_out].copy_from_slice(&row_out);
            }

            let w_t = MetalTensor::from_bytes(
                &ctx,
                weight_bytes,
                vec![n_in as u64, n_out as u64],
                GgmlType::Q8_0,
            )
            .expect("weight tensor");
            let x_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x tensor");
            let y_t =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y tensor");
            one_shot(&ctx, |enc| {
                encode_mat_mat_q8_0_f32(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out, n_query)
            })
            .expect("mat_mat encode");

            let gpu_flat = read_back_f32(&y_t.buffer, n_out * n_query);
            let mut gpu_row_major = vec![0.0f32; n_query * n_out];
            for q in 0..n_query {
                for o in 0..n_out {
                    gpu_row_major[q * n_out + o] = gpu_flat[o + q * n_out];
                }
            }

            let mut min_cos = f64::INFINITY;
            let mut max_abs = 0.0f32;
            for q in 0..n_query {
                let cpu_row = &cpu_row_major[q * n_out..(q + 1) * n_out];
                let gpu_row = &gpu_row_major[q * n_out..(q + 1) * n_out];
                let mut dot = 0.0f64;
                let mut np = 0.0f64;
                let mut nc = 0.0f64;
                for i in 0..n_out {
                    let p = gpu_row[i] as f64;
                    let c = cpu_row[i] as f64;
                    dot += p * c;
                    np += p * p;
                    nc += c * c;
                    let d = (gpu_row[i] - cpu_row[i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
                let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
                if cos < min_cos {
                    min_cos = cos;
                }
            }
            eprintln!(
                "[mat_mat_q8_0 n_query={n_query}] min_cos={min_cos:.6} \
                 max|Δ|={max_abs:.3e}"
            );
            assert!(
                min_cos >= 0.999,
                "n_query={n_query}: min cos {min_cos} < 0.999"
            );
            assert!(
                max_abs < 1e-2,
                "n_query={n_query}: max|Δ| {max_abs} >= 1e-2"
            );

            // Layout sanity: col-major dst at three corner cells.
            for &(r, c) in &[
                (0usize, 0usize),
                (1usize, n_query / 2),
                (n_out - 1, n_query - 1),
            ] {
                let raw = gpu_flat[r + c * n_out];
                let row_major_view = gpu_row_major[c * n_out + r];
                assert_eq!(
                    raw.to_bits(),
                    row_major_view.to_bits(),
                    "layout sanity n_query={n_query}: (r={r}, c={c})"
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn prompt_mat_mat_production_shapes() {
        use std::time::Instant;

        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
        if !std::path::Path::new(path).exists() {
            eprintln!("[prompt-matmat] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        const N_QUERY: usize = 321;
        let cases = [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.out_proj.weight",
            "blk.0.attn_qkv.weight",
        ];

        eprintln!("[prompt-matmat] {}", ctx.describe());
        for name in cases {
            let Some(t) = g.find(name) else {
                eprintln!("[prompt-matmat] skip missing {name}");
                continue;
            };
            if t.shape.len() != 2 {
                continue;
            }
            let n_in = t.shape[0] as usize;
            let n_out = t.shape[1] as usize;
            let w_t = MetalTensor::from_gguf_tensor(&ctx, t, g.slice(t)).expect("w");
            let x_vec: Vec<f32> = (0..n_in).map(|i| (i as f32 * 1e-3).sin()).collect();
            let x_vec_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_vec),
                vec![n_in as u64],
                GgmlType::F32,
            )
            .expect("x_vec");
            let y_vec_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y_vec");
            let x_mat: Vec<f32> = (0..N_QUERY * n_in)
                .map(|i| (i as f32 * 1e-3).sin())
                .collect();
            let x_mat_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_mat),
                vec![N_QUERY as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x_mat");
            let y_mat_t =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64, N_QUERY as u64]).expect("y_mat");

            match t.dtype {
                GgmlType::Q4_K => {
                    bench_q4_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("warm vec");
                    bench_q4_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("warm mm");
                    let t0 = Instant::now();
                    bench_q4_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("vec");
                    let vec_ms = t0.elapsed().as_secs_f64() * 1e3;
                    let t1 = Instant::now();
                    bench_q4_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("mm");
                    let mm_ms = t1.elapsed().as_secs_f64() * 1e3;
                    eprintln!(
                        "[prompt-matmat] {name:24} dtype={:?} n_in={n_in:>5} n_out={n_out:>6} N={N_QUERY:>3} vec321={vec_ms:>8.2} ms mm={mm_ms:>8.2} ms speedup={:>5.2}x",
                        t.dtype,
                        vec_ms / mm_ms.max(1e-9)
                    );
                }
                GgmlType::Q5_K => {
                    bench_q5_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("warm vec");
                    bench_q5_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("warm mm");
                    let t0 = Instant::now();
                    bench_q5_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("vec");
                    let vec_ms = t0.elapsed().as_secs_f64() * 1e3;
                    let t1 = Instant::now();
                    bench_q5_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("mm");
                    let mm_ms = t1.elapsed().as_secs_f64() * 1e3;
                    eprintln!(
                        "[prompt-matmat] {name:24} dtype={:?} n_in={n_in:>5} n_out={n_out:>6} N={N_QUERY:>3} vec321={vec_ms:>8.2} ms mm={mm_ms:>8.2} ms speedup={:>5.2}x",
                        t.dtype,
                        vec_ms / mm_ms.max(1e-9)
                    );
                }
                GgmlType::Q6_K => {
                    bench_q6_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("warm vec");
                    bench_q6_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("warm mm");
                    let t0 = Instant::now();
                    bench_q6_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("vec");
                    let vec_ms = t0.elapsed().as_secs_f64() * 1e3;
                    let t1 = Instant::now();
                    bench_q6_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("mm");
                    let mm_ms = t1.elapsed().as_secs_f64() * 1e3;
                    eprintln!(
                        "[prompt-matmat] {name:24} dtype={:?} n_in={n_in:>5} n_out={n_out:>6} N={N_QUERY:>3} vec321={vec_ms:>8.2} ms mm={mm_ms:>8.2} ms speedup={:>5.2}x",
                        t.dtype,
                        vec_ms / mm_ms.max(1e-9)
                    );
                }
                _ => {
                    eprintln!("[prompt-matmat] skip {name} dtype={:?}", t.dtype);
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn prompt_mat_mat_production_shapes_chained64() {
        use std::time::Instant;

        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
        if !std::path::Path::new(path).exists() {
            eprintln!("[prompt-matmat-chained] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let n_query: usize = std::env::var("QWEN_PROMPT_MATMAT_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(321);
        let n_dispatches: usize = std::env::var("QWEN_PROMPT_MATMAT_DISPATCHES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let cases = [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_qkv.weight",
        ];

        eprintln!(
            "[prompt-matmat-chained] {} N={n_query} dispatches={n_dispatches}",
            ctx.describe()
        );
        for name in cases {
            let Some(t) = g.find(name) else {
                eprintln!("[prompt-matmat-chained] skip missing {name}");
                continue;
            };
            let n_in = t.shape[0] as usize;
            let n_out = t.shape[1] as usize;
            let w_t = MetalTensor::from_gguf_tensor(&ctx, t, g.slice(t)).expect("w");
            let x_mat: Vec<f32> = (0..n_query * n_in)
                .map(|i| (i as f32 * 1e-3).sin())
                .collect();
            let x_mat_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x_mat),
                vec![n_query as u64, n_in as u64],
                GgmlType::F32,
            )
            .expect("x_mat");
            let y_mat_t =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64]).expect("y_mat");

            match t.dtype {
                GgmlType::Q4_K => {
                    bench_q4_k_mat_mat_chained(
                        &ctx,
                        &w_t,
                        &x_mat_t,
                        &y_mat_t,
                        n_in,
                        n_out,
                        n_query,
                        n_dispatches,
                    )
                    .expect("warm mm");
                    let t0 = Instant::now();
                    bench_q4_k_mat_mat_chained(
                        &ctx,
                        &w_t,
                        &x_mat_t,
                        &y_mat_t,
                        n_in,
                        n_out,
                        n_query,
                        n_dispatches,
                    )
                    .expect("mm");
                    let mm_ms = t0.elapsed().as_secs_f64() * 1e3 / n_dispatches as f64;
                    let weight_gib =
                        (t.n_bytes * n_dispatches as u64) as f64 / (1024.0 * 1024.0 * 1024.0);
                    let gib_s = weight_gib / (t0.elapsed().as_secs_f64());
                    eprintln!(
                        "[prompt-matmat-chained] {name:24} dtype={:?} N={n_query:>5} per-dispatch={mm_ms:>7.3} ms weight-throughput={gib_s:>7.1} GiB/s",
                        t.dtype
                    );
                }
                GgmlType::Q5_K => {
                    bench_q5_k_mat_mat_chained(
                        &ctx,
                        &w_t,
                        &x_mat_t,
                        &y_mat_t,
                        n_in,
                        n_out,
                        n_query,
                        n_dispatches,
                    )
                    .expect("warm mm");
                    let t0 = Instant::now();
                    bench_q5_k_mat_mat_chained(
                        &ctx,
                        &w_t,
                        &x_mat_t,
                        &y_mat_t,
                        n_in,
                        n_out,
                        n_query,
                        n_dispatches,
                    )
                    .expect("mm");
                    let mm_ms = t0.elapsed().as_secs_f64() * 1e3 / n_dispatches as f64;
                    let weight_gib =
                        (t.n_bytes * n_dispatches as u64) as f64 / (1024.0 * 1024.0 * 1024.0);
                    let gib_s = weight_gib / (t0.elapsed().as_secs_f64());
                    eprintln!(
                        "[prompt-matmat-chained] {name:24} dtype={:?} N={n_query:>5} per-dispatch={mm_ms:>7.3} ms weight-throughput={gib_s:>7.1} GiB/s",
                        t.dtype
                    );
                }
                GgmlType::Q6_K => {
                    bench_q6_k_mat_mat_chained(
                        &ctx,
                        &w_t,
                        &x_mat_t,
                        &y_mat_t,
                        n_in,
                        n_out,
                        n_query,
                        n_dispatches,
                    )
                    .expect("warm mm");
                    let t0 = Instant::now();
                    bench_q6_k_mat_mat_chained(
                        &ctx,
                        &w_t,
                        &x_mat_t,
                        &y_mat_t,
                        n_in,
                        n_out,
                        n_query,
                        n_dispatches,
                    )
                    .expect("mm");
                    let mm_ms = t0.elapsed().as_secs_f64() * 1e3 / n_dispatches as f64;
                    let weight_gib =
                        (t.n_bytes * n_dispatches as u64) as f64 / (1024.0 * 1024.0 * 1024.0);
                    let gib_s = weight_gib / (t0.elapsed().as_secs_f64());
                    eprintln!(
                        "[prompt-matmat-chained] {name:24} dtype={:?} N={n_query:>5} per-dispatch={mm_ms:>7.3} ms weight-throughput={gib_s:>7.1} GiB/s",
                        t.dtype
                    );
                }
                _ => {
                    eprintln!("[prompt-matmat-chained] skip {name} dtype={:?}", t.dtype);
                }
            }
        }
    }

    /// Q8_0 token-axis amortization bench. The defaults retain the original
    /// N=16 DFlash gate; environment overrides make the same harness useful for
    /// exact production shapes at other small-N operating points. It compares
    /// the generic mat-mat tile, one batched exact GEMV dispatch, and N exact
    /// singleton GEMVs in same-command chains.
    ///
    /// Run: `cargo test --release --lib -p qwen-llm
    /// q8_0_mat_mat_amortization_vs_n_mat_vec --ignored -- --nocapture`
    #[test]
    #[ignore]
    fn q8_0_mat_mat_amortization_vs_n_mat_vec() {
        use std::time::Instant;

        fn env_usize(name: &str, default: usize) -> usize {
            std::env::var(name)
                .ok()
                .map(|value| value.parse::<usize>().expect("invalid positive integer"))
                .unwrap_or(default)
        }

        fn median(values: &mut [f64]) -> f64 {
            values.sort_by(f64::total_cmp);
            let middle = values.len() / 2;
            if values.len().is_multiple_of(2) {
                (values[middle - 1] + values[middle]) * 0.5
            } else {
                values[middle]
            }
        }

        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = std::env::var("QWEN_Q8_AMORT_MODEL")
            .unwrap_or_else(|_| crate::test_fixtures::DFLASH_DRAFT_36_Q8_0.path().into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("[q8-amort] skipped - GGUF missing: {path}");
            return;
        }
        let tensor_name = std::env::var("QWEN_Q8_AMORT_TENSOR")
            .unwrap_or_else(|_| "blk.0.ffn_down.weight".into());
        let g = crate::gguf::GgufFile::open(&path).expect("open");
        let q8 = g
            .tensors
            .iter()
            .find(|tensor| tensor.name == tensor_name && tensor.dtype == GgmlType::Q8_0)
            .unwrap_or_else(|| {
                let available = g
                    .tensors
                    .iter()
                    .filter(|tensor| tensor.dtype == GgmlType::Q8_0)
                    .take(64)
                    .map(|tensor| tensor.name.as_str())
                    .collect::<Vec<_>>();
                panic!("{tensor_name} Q8_0 not found; first Q8_0 tensors: {available:?}")
            });
        let n_in = q8.shape[0] as usize;
        let n_out = q8.shape[1] as usize;
        let n_query = env_usize("QWEN_Q8_AMORT_N", 16);
        let n_layers = env_usize("QWEN_Q8_AMORT_LAYERS", 5);
        let warmup = env_usize("QWEN_Q8_AMORT_WARMUPS", 5);
        let iters = env_usize("QWEN_Q8_AMORT_ITERS", 30);
        assert!(n_query > 0 && n_layers > 0 && warmup > 0 && iters > 0);

        eprintln!(
            "[q8-amort] tensor={} shape=[n_in={n_in}, n_out={n_out}] N={n_query} layers={n_layers}",
            q8.name
        );

        let w_t = MetalTensor::from_bytes(
            &ctx,
            g.slice(q8),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q8_0,
        )
        .expect("weight tensor");
        let x = (0..n_query * n_in)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.002)
            .collect::<Vec<_>>();
        let x_packed = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_query * n_in) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let y_mat_mat = MetalTensor::zeros_f32(&ctx, vec![(n_out * n_query) as u64]).unwrap();
        let y_batch = MetalTensor::zeros_f32(&ctx, vec![(n_out * n_query) as u64]).unwrap();
        let y_sequential = MetalTensor::zeros_f32(&ctx, vec![(n_out * n_query) as u64]).unwrap();

        let bench_mat_mat = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                encode_mat_mat_q8_0_f32(
                    &ctx, &enc, &w_t, &x_packed, &y_mat_mat, n_in, n_out, n_query,
                )
                .unwrap();
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            wait_completed(&cmd).expect("Metal command buffer failed");
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        let bench_batch = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                encode_mat_vec_q8_0_batch_f32(
                    &ctx, &enc, &w_t, &x_packed, &y_batch, n_in, n_out, n_query,
                )
                .unwrap();
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            wait_completed(&cmd).expect("Metal command buffer failed");
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        let bench_n_mat_vec = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                for row in 0..n_query {
                    let x_row = x_packed.view_subrange((row * n_in) as u64, vec![n_in as u64]);
                    let y_row =
                        y_sequential.view_subrange((row * n_out) as u64, vec![n_out as u64]);
                    encode_mat_vec_q8_0_f32(&ctx, &enc, &w_t, &x_row, &y_row, n_in, n_out).unwrap();
                }
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            wait_completed(&cmd).expect("Metal command buffer failed");
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        for _ in 0..warmup {
            bench_mat_mat();
            bench_batch();
            bench_n_mat_vec();
        }

        let mut mm_wall = Vec::with_capacity(iters);
        let mut mm_gpu = Vec::with_capacity(iters);
        let mut batch_wall = Vec::with_capacity(iters);
        let mut batch_gpu = Vec::with_capacity(iters);
        let mut mv_wall = Vec::with_capacity(iters);
        let mut mv_gpu = Vec::with_capacity(iters);
        for iteration in 0..iters {
            let mut record = |kind: usize, (wall, gpu): (f64, f64)| match kind {
                0 => {
                    mm_wall.push(wall);
                    mm_gpu.push(gpu);
                }
                1 => {
                    batch_wall.push(wall);
                    batch_gpu.push(gpu);
                }
                2 => {
                    mv_wall.push(wall);
                    mv_gpu.push(gpu);
                }
                _ => unreachable!(),
            };
            match iteration % 3 {
                0 => {
                    record(0, bench_mat_mat());
                    record(1, bench_batch());
                    record(2, bench_n_mat_vec());
                }
                1 => {
                    record(1, bench_batch());
                    record(2, bench_n_mat_vec());
                    record(0, bench_mat_mat());
                }
                _ => {
                    record(2, bench_n_mat_vec());
                    record(0, bench_mat_mat());
                    record(1, bench_batch());
                }
            }
        }
        let mm_wall = median(&mut mm_wall);
        let mm_gpu = median(&mut mm_gpu);
        let batch_wall = median(&mut batch_wall);
        let batch_gpu = median(&mut batch_gpu);
        let mv_wall = median(&mut mv_wall);
        let mv_gpu = median(&mut mv_gpu);

        eprintln!("[q8-amort] {n_layers} layers x N={n_query} median over {iters} iters:");
        eprintln!(
            "  mat-mat (1 disp/layer):    wall={mm_wall:7.2} ms  gpu={mm_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            mm_gpu / n_layers as f64
        );
        eprintln!(
            "  batched GEMV (1/layer):    wall={batch_wall:7.2} ms  gpu={batch_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            batch_gpu / n_layers as f64
        );
        eprintln!(
            "  N={n_query} mat-vec ({n_query}/layer): wall={mv_wall:7.2} ms  gpu={mv_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            mv_gpu / n_layers as f64
        );
        let ratio_wall = mm_wall / mv_wall;
        let ratio_gpu = mm_gpu / mv_gpu;
        let batch_ratio_wall = batch_wall / mv_wall;
        let batch_ratio_gpu = batch_gpu / mv_gpu;
        eprintln!(
            "  ratios vs sequential: mat-mat wall/gpu={ratio_wall:.3}/{ratio_gpu:.3}; batched-GEMV wall/gpu={batch_ratio_wall:.3}/{batch_ratio_gpu:.3}"
        );

        one_shot(&ctx, |enc| {
            encode_mat_vec_q8_0_batch_f32(
                &ctx, enc, &w_t, &x_packed, &y_batch, n_in, n_out, n_query,
            )?;
            for row in 0..n_query {
                let x_row = x_packed.view_subrange((row * n_in) as u64, vec![n_in as u64]);
                let y_row = y_sequential.view_subrange((row * n_out) as u64, vec![n_out as u64]);
                encode_mat_vec_q8_0_f32(&ctx, enc, &w_t, &x_row, &y_row, n_in, n_out)?;
            }
            Ok(())
        })
        .unwrap();
        let batch = read_back_f32(&y_batch.buffer, n_query * n_out);
        let sequential = read_back_f32(&y_sequential.buffer, n_query * n_out);
        let bit_mismatches = batch
            .iter()
            .zip(&sequential)
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        eprintln!("  batched-GEMV bit mismatches vs sequential: {bit_mismatches}");
        assert_eq!(bit_mismatches, 0);

        if let Ok(value) = std::env::var("QWEN_Q8_AMORT_MAX_GPU_RATIO") {
            let max_ratio = value.parse::<f64>().expect("invalid GPU ratio");
            assert!(
                ratio_gpu <= max_ratio,
                "Q8 mat-mat GPU ratio {ratio_gpu:.3} exceeds {max_ratio:.3}"
            );
        } else if n_query == 16 && tensor_name == "blk.0.ffn_down.weight" {
            assert!(
                ratio_gpu <= 0.5,
                "original N=16 Q8 gate failed: GPU ratio {ratio_gpu:.3} > 0.5"
            );
        }
    }

    /// v0.73a.0 A-lite GO/NO-GO bench. Compares amortized weight-BW of
    /// Q5_K mat-mat (NR1=16 fast path) vs N=16 successive Q5_K mat-vec
    /// on production GDN out_proj (`blk.*.ssm_out.weight`) shape.
    ///
    /// Runs 48 chained dispatches (= 48 GDN layers) of each path in one
    /// command buffer; reports per-dispatch latency and the ratio. The
    /// hypothesis under test is that mat-mat amortizes the per-step
    /// weight reads N=16-fold, so the ratio should be substantially
    /// less than 1.0 (i.e. mat-mat much faster). Codex's framing:
    /// "if Q5_K mat-mat doesn't beat 16 mat-vecs by a large margin,
    /// stop and reassess." Threshold for proceed: ratio ≤ 0.5
    /// (mat-mat at LEAST 2× faster than 16-mat-vec equivalent work).
    /// Empirically Q4_K and Q6_K mat-mat at this shape achieve closer
    /// to 4-8× on M4 Max.
    ///
    /// Run: `cargo test --release --lib -p qwen-llm
    /// q5_k_mat_mat_amortization_vs_n_mat_vec --ignored -- --nocapture`
    #[test]
    #[ignore]
    fn q5_k_mat_mat_amortization_vs_n_mat_vec() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
        if !std::path::Path::new(path).exists() {
            eprintln!("[v0.73a.0-gate] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let q5k = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ssm_out.weight" && t.dtype == GgmlType::Q5_K)
            .expect("blk.0.ssm_out.weight Q5_K not found");
        let n_in = q5k.shape[0] as usize;
        let n_out = q5k.shape[1] as usize;
        let n_query = 16usize;
        let n_layers = 48usize; // GDN layers in 27B
        let warmup = 5usize;
        let iters = 30usize;

        eprintln!(
            "[v0.73a.0-gate] tensor={} shape=[n_in={n_in}, n_out={n_out}] N={n_query} layers={n_layers}",
            q5k.name
        );

        let w_t = MetalTensor::from_bytes(
            &ctx,
            g.slice(q5k),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q5_K,
        )
        .expect("weight tensor");
        let x_packed = MetalTensor::zeros_f32(&ctx, vec![(n_query * n_in) as u64]).unwrap();
        let y_packed = MetalTensor::zeros_f32(&ctx, vec![(n_out * n_query) as u64]).unwrap();
        let x_single = MetalTensor::zeros_f32(&ctx, vec![n_in as u64]).unwrap();
        let y_single = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();

        let bench_mat_mat = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                encode_mat_mat_q5_k_f32(
                    &ctx, &enc, &w_t, &x_packed, &y_packed, n_in, n_out, n_query,
                )
                .unwrap();
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            wait_completed(&cmd).expect("Metal command buffer failed");
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        // 16 mat-vec per layer = 16*48 = 768 dispatches per command buffer.
        // This mirrors the production per-token GDN out_proj fall-through
        // we'd be replacing.
        let bench_n_mat_vec = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                for _ in 0..n_query {
                    encode_mat_vec_q5_k_f32(&ctx, &enc, &w_t, &x_single, &y_single, n_in, n_out)
                        .unwrap();
                }
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            wait_completed(&cmd).expect("Metal command buffer failed");
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        // Warmup both.
        for _ in 0..warmup {
            bench_mat_mat();
            bench_n_mat_vec();
        }

        let mut sum_mm_wall = 0.0f64;
        let mut sum_mm_gpu = 0.0f64;
        let mut sum_mv_wall = 0.0f64;
        let mut sum_mv_gpu = 0.0f64;
        for _ in 0..iters {
            let (w, g) = bench_mat_mat();
            sum_mm_wall += w;
            sum_mm_gpu += g;
        }
        for _ in 0..iters {
            let (w, g) = bench_n_mat_vec();
            sum_mv_wall += w;
            sum_mv_gpu += g;
        }
        let mm_wall = sum_mm_wall / iters as f64;
        let mm_gpu = sum_mm_gpu / iters as f64;
        let mv_wall = sum_mv_wall / iters as f64;
        let mv_gpu = sum_mv_gpu / iters as f64;

        eprintln!("[v0.73a.0-gate] {n_layers} layers × N={n_query} avg over {iters} iters:");
        eprintln!(
            "  mat-mat (1 disp/layer):    wall={mm_wall:7.2} ms  gpu={mm_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            mm_gpu / n_layers as f64
        );
        eprintln!(
            "  N=16 mat-vec (16/layer):   wall={mv_wall:7.2} ms  gpu={mv_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            mv_gpu / n_layers as f64
        );
        let ratio_wall = mm_wall / mv_wall;
        let ratio_gpu = mm_gpu / mv_gpu;
        let speedup_wall = 1.0 / ratio_wall;
        let speedup_gpu = 1.0 / ratio_gpu;
        eprintln!(
            "  ratio mat-mat / 16×mat-vec: wall={ratio_wall:.3} (= {speedup_wall:.2}× speedup)  gpu={ratio_gpu:.3} (= {speedup_gpu:.2}× speedup)"
        );

        // GO/NO-GO: GPU ratio must be at most 0.5 (= 2× speedup). This
        // is a conservative bar relative to Q4_K/Q6_K experience (4-8×).
        // If we don't clear it, v0.73a.1 won't deliver the projected
        // win and we should reassess BEFORE shipping the orchestration
        // restructure.
        assert!(
            ratio_gpu <= 0.5,
            "v0.73a.0 GO/NO-GO failed: GPU ratio {ratio_gpu:.3} > 0.5 \
             (mat-mat must beat 16 mat-vec by at least 2×; \
             reassess before v0.73a.1)"
        );
    }

    #[test]
    fn ffn_fused_swiglu_q4_k_mma8_matches_unfused_n8() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let path = "/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let gate = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate.weight" && t.dtype == GgmlType::Q4_K)
            .expect("Q4_K gate");
        let up = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up.weight" && t.dtype == GgmlType::Q4_K)
            .expect("Q4_K up");
        assert_eq!(gate.shape, up.shape);
        let n_in = gate.shape[0] as usize;
        let n_out = gate.shape[1] as usize;
        const N: usize = 8;
        let x: Vec<f32> = (0..N * n_in)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.01)
            .collect();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(N * n_in) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let gate_w = MetalTensor::from_bytes(
            &ctx,
            g.slice(gate),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .unwrap();
        let up_w = MetalTensor::from_bytes(
            &ctx,
            g.slice(up),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .unwrap();

        let gate_ref = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
        let up_ref = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
        let inner_ref = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_mat_mat_mma8_variant(
                &ctx,
                enc,
                &gate_w,
                &x_t,
                &gate_ref,
                n_in,
                n_out,
                "r1c1k64_sg2",
            )?;
            encode_mat_mat_mma8_variant(
                &ctx,
                enc,
                &up_w,
                &x_t,
                &up_ref,
                n_in,
                n_out,
                "r1c1k64_sg2",
            )?;
            encode_silu_mul_f32(&ctx, enc, &gate_ref, &up_ref, &inner_ref)
        })
        .unwrap();

        let inner_fused = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_ffn_fused_swiglu_q4_k_mma8_f32(
                &ctx,
                enc,
                &gate_w,
                &up_w,
                &x_t,
                &inner_fused,
                n_in,
                n_out,
            )
        })
        .unwrap();
        let reference = read_back_f32(&inner_ref.buffer, N * n_out);
        let fused = read_back_f32(&inner_fused.buffer, N * n_out);
        let max_abs = reference
            .iter()
            .zip(&fused)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let bit_mismatches = reference
            .iter()
            .zip(&fused)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!("[ffn-fused-mma8-n8] max_abs={max_abs:.3e} bit_mismatches={bit_mismatches}");
        assert_eq!(bit_mismatches, 0, "fused N8 SwiGLU must be bitwise equal");

        let scalar = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
        let vec4 = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_mat_mat_mma8_variant(
                &ctx, enc, &gate_w, &x_t, &scalar, n_in, n_out, "r1c1k128",
            )?;
            encode_mat_mat_mma8_variant(
                &ctx,
                enc,
                &gate_w,
                &x_t,
                &vec4,
                n_in,
                n_out,
                "r1c1k128_vec4",
            )
        })
        .unwrap();
        let scalar = read_back_f32(&scalar.buffer, N * n_out);
        let vec4 = read_back_f32(&vec4.buffer, N * n_out);
        assert!(
            scalar
                .iter()
                .zip(&vec4)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "vectorized Q4_K dequant changed K128 matmul output"
        );
    }

    /// H5.3a foundation: verify `BlitEncoder` actually copies device-side
    /// buffers and that compute↔blit transitions on the same command
    /// buffer are visible. We write a known pattern via a compute kernel
    /// (`scatter_offset`), blit-copy into a destination buffer, then read
    /// the destination back. If the blit didn't fire, we'd read zeros.
    #[test]
    fn blit_encoder_copies_buffer_within_one_command_buffer() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        const N: usize = 1024;
        let pattern: Vec<f32> = (0..N).map(|i| (i as f32) * 0.125).collect();

        // Source: a freshly-uploaded MetalTensor holding `pattern`.
        let src_buf = ctx.buffer_from(&pattern).expect("src buf");
        let src = MetalTensor {
            buffer: src_buf,
            offset: 0,
            shape: vec![N as u64],
            dtype: crate::tensor::GgmlType::F32,
            provenance: MetalTensorProvenance::OwnedWritable,
        };

        // Destination: zero-initialized.
        let dst_buf = ctx.buffer_uninit(N * 4).expect("dst buf");
        // Zero it out via the host pointer (StorageModeShared).
        unsafe {
            let p = dst_buf.contents().as_ptr() as *mut f32;
            for i in 0..N {
                *p.add(i) = -1.0;
            }
        }
        let dst = MetalTensor {
            buffer: dst_buf,
            offset: 0,
            shape: vec![N as u64],
            dtype: crate::tensor::GgmlType::F32,
            provenance: MetalTensorProvenance::OwnedWritable,
        };

        // One command buffer. Compute pass (no-op trampoline to validate
        // that compute → blit transitions don't drop ordering), then blit
        // pass that performs the actual copy.
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        // Empty compute pass. We don't dispatch anything — we just want
        // to verify that an opened-and-immediately-closed compute encoder
        // doesn't break the subsequent blit.
        {
            let enc = KernelEncoder::begin(&cmd);
            enc.end();
        }
        {
            let blit = BlitEncoder::begin(&cmd);
            blit.copy_tensor(&src, &dst);
            blit.end();
        }
        cmd.commit();
        wait_completed(&cmd).expect("Metal command buffer failed");

        // Read back via host pointer.
        let got: Vec<f32> = unsafe {
            let p = dst.buffer.contents().as_ptr() as *const f32;
            (0..N).map(|i| *p.add(i)).collect()
        };
        for i in 0..N {
            assert!(
                (got[i] - pattern[i]).abs() < 1e-9,
                "blit mismatch at i={i}: got={} expected={}",
                got[i],
                pattern[i]
            );
        }

        // Also exercise `copy_buffer` with non-zero offsets: copy the
        // back half of `src` into the front half of `dst`.
        let cmd2 = ctx.queue.commandBuffer().expect("cmd2");
        {
            let blit = BlitEncoder::begin(&cmd2);
            let half_bytes = (N / 2) * 4;
            blit.copy_buffer(
                &src.buffer,
                half_bytes as u64,
                &dst.buffer,
                0,
                half_bytes as u64,
            );
            blit.end();
        }
        cmd2.commit();
        wait_completed(&cmd2).expect("Metal command buffer failed");
        let got2: Vec<f32> = unsafe {
            let p = dst.buffer.contents().as_ptr() as *const f32;
            (0..N).map(|i| *p.add(i)).collect()
        };
        // Front half of dst now equals back half of src.
        for i in 0..N / 2 {
            assert!(
                (got2[i] - pattern[N / 2 + i]).abs() < 1e-9,
                "offset blit front-half mismatch at i={i}: got={} expected={}",
                got2[i],
                pattern[N / 2 + i]
            );
        }
        // Back half of dst is unchanged from the previous full-blit copy.
        for i in N / 2..N {
            assert!(
                (got2[i] - pattern[i]).abs() < 1e-9,
                "offset blit back-half disturbed at i={i}: got={} expected={}",
                got2[i],
                pattern[i]
            );
        }
    }
}
