//! GEMV dispatchers across quant formats.

use super::*;

crate::env_flag!(default_on mat_vec_f32_lcpp_r2_enabled, "QWEN_MATVEC_F32_LCPP_R2");

#[cfg(test)]
pub(crate) fn mat_vec_f32_lcpp_r2_enabled_for_test() -> bool {
    mat_vec_f32_lcpp_r2_enabled()
}

/// F32 mat-vec: `y[o] = Σ_i W[o, i] * x[i]`, GGUF stride convention.
/// `W` has shape `[n_in, n_out]` (ne[0]=n_in fastest); `x` is `[n_in]`,
/// `y` is `[n_out]`.
///
/// CPU oracle: [`crate::forward::mat_vec_pub`].
pub fn encode_mat_vec_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    // Defense against the silent-NaN class of bug. If a quantized weight
    // tensor lands here by mistake (someone forgot to call
    // encode_mat_vec_dispatch), we'd reinterpret block bytes as floats
    // and produce garbage that propagates as NaNs through later layers.
    // Caught codex-review-style by adding this guard. v1.
    if weight.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_f32",
            detail: format!(
                "weight.dtype = {:?}, expected F32 — use encode_mat_vec_dispatch \
                 to handle quantized weights",
                weight.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_f32",
            detail: format!("x.n_elements={} != n_in={n_in}", x.n_elements()),
        });
    }
    if y.n_elements() as usize != n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_f32",
            detail: format!("y.n_elements={} != n_out={n_out}", y.n_elements()),
        });
    }
    let use_r2 = mat_vec_f32_lcpp_r2_enabled();
    let pso = ctx.pipeline(if use_r2 {
        "kernel_mat_vec_f32_f32_lcpp_r2"
    } else {
        "kernel_mat_vec_f32_f32"
    })?;
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

    if use_r2 {
        let nr0 = 2usize;
        let nsg = 4usize;
        enc.set_threadgroup_memory(0, 32 * nr0 * std::mem::size_of::<f32>());
        enc.dispatch(
            MTLSize {
                width: n_out.div_ceil(nr0),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: nsg * 32,
                height: 1,
                depth: 1,
            },
        );
        return Ok(());
    }

    const ROWS_PER_TG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(ROWS_PER_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: ROWS_PER_TG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// F32 mat-vec followed by sigmoid, kept fused for small GDN beta
/// projections. The input weights and activation must both be F32.
pub fn encode_mat_vec_f32_sigmoid(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::F32 || x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_f32_sigmoid",
            detail: format!(
                "weight/x/y expected F32, got {:?}/{:?}/{:?}",
                weight.dtype, x.dtype, y.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in || y.n_elements() as usize != n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_f32_sigmoid",
            detail: format!(
                "x/y expected {n_in}/{n_out} elements, got {}/{}",
                x.n_elements(),
                y.n_elements()
            ),
        });
    }
    if weight.n_elements() as usize != n_in * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_f32_sigmoid",
            detail: format!(
                "weight expected {} elements, got {}",
                n_in * n_out,
                weight.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_f32_f32_sigmoid")?;
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
            width: n_out.div_ceil(4),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 4 * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(crate) fn encode_mat_vec_16bit_weight_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
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
    if x.n_elements() as usize != n_in {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("x.n_elements={} != n_in={n_in}", x.n_elements()),
        });
    }
    if y.n_elements() as usize != n_out {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("y.n_elements={} != n_out={n_out}", y.n_elements()),
        });
    }
    let pso = ctx.pipeline(kernel_name)?;
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

    const ROWS_PER_TG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(ROWS_PER_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: ROWS_PER_TG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_mat_vec_f16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    encode_mat_vec_16bit_weight_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::F16,
        "kernel_mat_vec_f16_f32",
    )
}

pub fn encode_mat_vec_bf16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    encode_mat_vec_16bit_weight_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::BF16,
        "kernel_mat_vec_bf16_f32",
    )
}

/// MXFP4 mat-vec with F32 activation and output tensors. MXFP4 stores 32
/// values in each 17-byte block: one E8M0 scale followed by 16 packed nibbles.
pub fn encode_mat_vec_mxfp4_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    const BLOCK_ELEMENTS: usize = 32;
    const KERNEL: &str = "mat_vec_mxfp4";
    if n_in == 0 || n_out == 0 || !n_in.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "n_in={n_in} must be nonzero and divisible by {BLOCK_ELEMENTS}; n_out={n_out} must be nonzero"
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
    let weight_elements = n_in
        .checked_mul(n_out)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "n_in*n_out overflow".into(),
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
        || x.shape.as_slice() != [n_in as u64]
        || y.shape.as_slice() != [n_out as u64]
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "weight/x/y expected shapes [{n_in},{n_out}]/[{n_in}]/[{n_out}], got {:?}/{:?}/{:?}",
                weight.shape, x.shape, y.shape
            ),
        });
    }
    debug_assert_eq!(weight.n_elements() as usize, weight_elements);
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

    let pso = ctx.pipeline("kernel_mat_vec_mxfp4_f32")?;
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
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    const ROWS_PER_TG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(ROWS_PER_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: ROWS_PER_TG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(crate) fn encode_mat_vec_block32_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expected: GgmlType,
    kernel_name: &'static str,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    encode_mat_vec_16bit_weight_f32(ctx, enc, weight, x, y, n_in, n_out, expected, kernel_name)
}

pub fn encode_mat_vec_q4_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    encode_mat_vec_block32_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::Q4_0,
        "kernel_mat_vec_q4_0_f32",
    )
}

pub fn encode_mat_vec_q4_1_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    encode_mat_vec_block32_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::Q4_1,
        "kernel_mat_vec_q4_1_f32",
    )
}

pub fn encode_mtp_draft_affine_q4_gs64_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    scales: &MetalTensor,
    biases: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    const GROUP_SIZE: usize = 64;
    const PACK_FACTOR: usize = 8;
    const ROWS_PER_TG: usize = 8;
    let kernel_name = "kernel_mtp_draft_affine_q4_gs64_f32";
    if !n_in.is_multiple_of(GROUP_SIZE) {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("n_in={n_in} not divisible by {GROUP_SIZE}"),
        });
    }
    if weight.dtype != GgmlType::F32
        || scales.dtype != GgmlType::F16
        || biases.dtype != GgmlType::F16
    {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "weight/scales/biases expected packed-u32-as-F32/F16/F16, got {:?}/{:?}/{:?}",
                weight.dtype, scales.dtype, biases.dtype
            ),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_in || y.n_elements() as usize != n_out {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "x/y elements {}/{} do not match n_in/n_out {n_in}/{n_out}",
                x.n_elements(),
                y.n_elements()
            ),
        });
    }
    let packs_per_row = n_in / PACK_FACTOR;
    let n_groups = n_in / GROUP_SIZE;
    if weight.n_elements() as usize != n_out * packs_per_row
        || scales.n_elements() as usize != n_out * n_groups
        || biases.n_elements() as usize != n_out * n_groups
    {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!(
                "bad packed head sizes: weight={} scales={} biases={} expected {}/{}/{}",
                weight.n_elements(),
                scales.n_elements(),
                biases.n_elements(),
                n_out * packs_per_row,
                n_out * n_groups,
                n_out * n_groups
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
        n_groups: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_groups: n_groups as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, scales);
    enc.set_tensor(3, biases);
    enc.set_tensor(4, x);
    enc.set_tensor(5, y);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(ROWS_PER_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

crate::env_flag!(default_on matvec_iq4_nl_fast_enabled, "QWEN_MATVEC_IQ4_NL_FAST");

pub fn encode_mat_vec_iq4_nl_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_iq4_nl_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            32,
            GgmlType::IQ4_NL,
            "mat_vec_iq4_nl_fast",
            "kernel_mat_vec_iq4_nl_f32_fast",
            2,
            2,
            32 * std::mem::size_of::<f32>(),
        );
    }
    encode_mat_vec_block32_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ4_NL,
        "kernel_mat_vec_iq4_nl_f32",
    )
}

pub(crate) fn encode_mat_vec_block256_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expected: GgmlType,
    kernel_name: &'static str,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: kernel_name,
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    encode_mat_vec_16bit_weight_f32(ctx, enc, weight, x, y, n_in, n_out, expected, kernel_name)
}

crate::env_flag!(default_on matvec_q3_k_fast_enabled, "QWEN_MATVEC_Q3_K_FAST");

pub fn encode_mat_vec_q3_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_q3_k_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::Q3_K,
            "mat_vec_q3_k_fast",
            "kernel_mat_vec_q3_K_f32_fast",
            2,
            2,
            0,
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::Q3_K,
        "kernel_mat_vec_q3_K_f32",
    )
}

crate::env_flag!(default_on matvec_q2_k_fast_enabled, "QWEN_MATVEC_Q2_K_FAST");

pub fn encode_mat_vec_q2_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_q2_k_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::Q2_K,
            "mat_vec_q2_k_fast",
            "kernel_mat_vec_q2_K_f32_fast",
            4,
            2,
            0,
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::Q2_K,
        "kernel_mat_vec_q2_K_f32",
    )
}

crate::env_flag!(default_on matvec_iq2_xs_fast_enabled, "QWEN_MATVEC_IQ2_XS_FAST");

pub fn encode_mat_vec_iq2_xs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_iq2_xs_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::IQ2_XS,
            "mat_vec_iq2_xs_fast",
            "kernel_mat_vec_iq2_xs_f32_fast",
            4,
            2,
            0,
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ2_XS,
        "kernel_mat_vec_iq2_xs_f32",
    )
}

crate::env_flag!(default_on matvec_iq2_s_fast_enabled, "QWEN_MATVEC_IQ2_S_FAST");

pub fn encode_mat_vec_iq2_s_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_iq2_s_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::IQ2_S,
            "mat_vec_iq2_s_fast",
            "kernel_mat_vec_iq2_s_f32_fast",
            4,
            2,
            0,
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ2_S,
        "kernel_mat_vec_iq2_s_f32",
    )
}

pub(crate) fn encode_mat_vec_lowbit_nc2_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expected: GgmlType,
    error_kernel: &'static str,
    metal_kernel: &'static str,
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
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 || !y.is_writable() {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!(
                "x/y expected F32 with writable y, got {:?}/{:?} y_provenance={:?}",
                x.dtype,
                y.dtype,
                y.provenance(),
            ),
        });
    }
    let weight_shape = [
        u64::try_from(n_in).map_err(|_| MetalError::BadShape {
            kernel: error_kernel,
            detail: "n_in exceeds u64".into(),
        })?,
        u64::try_from(n_out).map_err(|_| MetalError::BadShape {
            kernel: error_kernel,
            detail: "n_out exceeds u64".into(),
        })?,
    ];
    if weight.shape != weight_shape {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!(
                "weight shape {:?} != expected {:?}",
                weight.shape, weight_shape
            ),
        });
    }
    let expected_x = n_in.checked_mul(2).ok_or_else(|| MetalError::BadShape {
        kernel: error_kernel,
        detail: "2*n_in overflow".into(),
    })?;
    let expected_y = n_out.checked_mul(2).ok_or_else(|| MetalError::BadShape {
        kernel: error_kernel,
        detail: "2*n_out overflow".into(),
    })?;
    let (x_elements, x_bytes) = checked_shape_bytes(&x.shape, std::mem::size_of::<f32>())?;
    let (y_elements, y_bytes) = checked_shape_bytes(&y.shape, std::mem::size_of::<f32>())?;
    if x_elements != expected_x || y_elements != expected_y {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!(
                "shape mismatch x={} y={} expected x={} y={}",
                x_elements, y_elements, expected_x, expected_y,
            ),
        });
    }
    let (_, weight_bytes) = checked_ggml_shape_bytes(&weight.shape, expected)?;
    let range_fits = |tensor: &MetalTensor, bytes: usize, alignment: u64| {
        tensor.offset.is_multiple_of(alignment)
            && u64::try_from(bytes)
                .ok()
                .and_then(|bytes| tensor.offset.checked_add(bytes))
                .is_some_and(|end| end <= tensor.buffer.length() as u64)
    };
    if !range_fits(weight, weight_bytes, 2)
        || !range_fits(x, x_bytes, std::mem::size_of::<f32>() as u64)
        || !range_fits(y, y_bytes, std::mem::size_of::<f32>() as u64)
    {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: "weight/x/y has an unaligned or out-of-buffer byte range".into(),
        });
    }
    if tensor_ranges_overlap(weight, y) || tensor_ranges_overlap(x, y) {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: "output must not overlap weight or input storage".into(),
        });
    }
    u32::try_from(expected_x).map_err(|_| MetalError::BadShape {
        kernel: error_kernel,
        detail: "2*n_in exceeds u32 indexing".into(),
    })?;
    u32::try_from(expected_y).map_err(|_| MetalError::BadShape {
        kernel: error_kernel,
        detail: "2*n_out exceeds u32 indexing".into(),
    })?;
    let n_in = u32::try_from(n_in).map_err(|_| MetalError::BadShape {
        kernel: error_kernel,
        detail: "n_in exceeds u32".into(),
    })?;
    let n_out_u32 = u32::try_from(n_out).map_err(|_| MetalError::BadShape {
        kernel: error_kernel,
        detail: "n_out exceeds u32".into(),
    })?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }

    let pso = ctx.pipeline(metal_kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_in,
            n_out: n_out_u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(8),
            height: 1,
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

pub fn encode_mat_vec_iq2_s_nc2_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    encode_mat_vec_lowbit_nc2_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ2_S,
        "mat_vec_iq2_s_nc2",
        "kernel_mat_vec_iq2_s_nc2_f32_fast",
    )
}

crate::env_flag!(default_on matvec_iq3_xxs_fast_enabled, "QWEN_MATVEC_IQ3_XXS_FAST");

pub fn encode_mat_vec_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_iq3_xxs_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::IQ3_XXS,
            "mat_vec_iq3_xxs_fast",
            "kernel_mat_vec_iq3_xxs_f32_fast",
            4,
            2,
            0,
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ3_XXS,
        "kernel_mat_vec_iq3_xxs_f32",
    )
}

crate::env_flag!(default_on matvec_iq3_s_fast_enabled, "QWEN_MATVEC_IQ3_S_FAST");

pub fn encode_mat_vec_iq3_s_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_iq3_s_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::IQ3_S,
            "mat_vec_iq3_s_fast",
            "kernel_mat_vec_iq3_s_f32_fast",
            4,
            2,
            0,
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ3_S,
        "kernel_mat_vec_iq3_s_f32",
    )
}

pub fn encode_mat_vec_iq3_s_nc2_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    encode_mat_vec_lowbit_nc2_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ3_S,
        "mat_vec_iq3_s_nc2",
        "kernel_mat_vec_iq3_s_nc2_f32_fast",
    )
}

crate::env_flag!(default_on matvec_iq4_xs_fast_enabled, "QWEN_MATVEC_IQ4_XS_FAST");

pub fn encode_mat_vec_iq4_xs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if matvec_iq4_xs_fast_enabled() {
        return encode_mat_vec_lowbit_fast_f32(
            ctx,
            enc,
            weight,
            x,
            y,
            n_in,
            n_out,
            256,
            GgmlType::IQ4_XS,
            "mat_vec_iq4_xs_fast",
            "kernel_mat_vec_iq4_xs_f32_fast",
            2,
            2,
            32 * std::mem::size_of::<f32>(),
        );
    }
    encode_mat_vec_block256_f32(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        GgmlType::IQ4_XS,
        "kernel_mat_vec_iq4_xs_f32",
    )
}

pub(crate) fn encode_mat_vec_lowbit_fast_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    block_multiple: usize,
    expected: GgmlType,
    error_kernel: &'static str,
    metal_kernel: &'static str,
    rows_per_simdgroup: usize,
    simdgroups: usize,
    threadgroup_bytes: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(block_multiple) {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!("n_in={n_in} not divisible by {block_multiple}"),
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
    if x.n_elements() as usize != n_in || y.n_elements() as usize != n_out {
        return Err(MetalError::BadShape {
            kernel: error_kernel,
            detail: format!(
                "shape mismatch x={} y={} expected x={} y={}",
                x.n_elements(),
                y.n_elements(),
                n_in,
                n_out
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }
    let pso = ctx.pipeline(metal_kernel)?;
    enc.set_pipeline(&pso);
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
    if threadgroup_bytes > 0 {
        enc.set_threadgroup_memory(0, threadgroup_bytes);
    }
    let rows_per_tg = rows_per_simdgroup * simdgroups;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(rows_per_tg),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: simdgroups * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Q4_K mat-vec on raw `block_q4_K` bytes. Same shape semantics as
/// [`encode_mat_vec_f32`]; `weight.dtype` must be `Q4_K`.
pub fn encode_mat_vec_q4_k_f32(
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
            kernel: "mat_vec_q4_k",
            detail: format!("n_in={n_in} not divisible by 256 (Q4_K super-block)"),
        });
    }
    if weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k",
            detail: format!("weight.dtype = {:?}, expected Q4_K", weight.dtype),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_q4_K_f32")?;
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

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Multi-column Q4_K mat-vec (H5.6 M2-nc skinny-GEMM experiment):
/// `Y = W · X^T` with mat-vec-grade occupancy (`n_out/4` threadgroups of
/// 128 threads: 2 row-pairs x 2 column-halves) running the mv1 body per
/// activation column — quant block reads are column-invariant and L1-hot
/// on re-reads (explicit register staging measured slower; see the kernel
/// header).
///
///   * `x`: F32 `[n_cols, n_in]` row-major (column c = `x + c*n_in`)
///   * `y`: F32 `[n_cols, n_out]` row-major (`y[c*n_out + r]`)
///   * `n_cols` ∈ {2, 4, 8} (compile-time instantiations)
///
/// Exactness: bit-identical per column to `encode_mat_vec_q4_k_f32`
/// (same accumulation order; E0 tier). Asserted by
/// `multicol_gemv_micro_27b` in tests/dflash_correctness.rs.
pub fn encode_mat_vec_q4_k_nc_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc",
            detail: format!("n_in={n_in} not divisible by 256 (Q4_K super-block)"),
        });
    }
    if weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc",
            detail: format!("weight.dtype = {:?}, expected Q4_K", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc",
            detail: format!("x/y dtype = {:?}/{:?}, expected F32/F32", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_cols * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc",
            detail: format!(
                "x.n_elements={} != n_cols*n_in={}",
                x.n_elements(),
                n_cols * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_cols * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc",
            detail: format!(
                "y.n_elements={} != n_cols*n_out={}",
                y.n_elements(),
                n_cols * n_out
            ),
        });
    }
    if weight.n_elements() as usize != n_in * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc",
            detail: format!(
                "weight.n_elements={} != n_in*n_out={}",
                weight.n_elements(),
                n_in * n_out
            ),
        });
    }
    let name = match n_cols {
        2 => "kernel_mat_vec_q4_K_nc2_f32",
        4 => "kernel_mat_vec_q4_K_nc4_f32",
        8 => "kernel_mat_vec_q4_K_nc8_f32",
        _ => {
            return Err(MetalError::BadShape {
                kernel: "mat_vec_q4_k_nc",
                detail: format!("n_cols={n_cols} not in {{2, 4, 8}}"),
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

    // TG = 4 simdgroups (128 threads): 2 row-pairs x 2 column-halves.
    // Same 4-rows-per-TG weight coverage as mv1 (grid = n_out/4), twice
    // the ALU per weight byte (columns split across the extra SGs).
    const ROWS_PER_TG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(ROWS_PER_TG),
            height: 1,
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

/// Q6_K companion to [`encode_mat_vec_q4_k_nc_f32`] — same layouts,
/// same 4-simdgroup (2 row-pairs x 2 column-halves) geometry, Q6_K
/// block decode. Covers the two largest verify-path tensors on
/// 27B-Q4_K_M (ffn_down, gdn_qkv).
pub fn encode_mat_vec_q6_k_nc_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_nc",
            detail: format!("n_in={n_in} not divisible by 256 (Q6_K super-block)"),
        });
    }
    if weight.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_nc",
            detail: format!("weight.dtype = {:?}, expected Q6_K", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_nc",
            detail: format!("x/y dtype = {:?}/{:?}, expected F32/F32", x.dtype, y.dtype),
        });
    }
    if x.n_elements() as usize != n_cols * n_in {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_nc",
            detail: format!(
                "x.n_elements={} != n_cols*n_in={}",
                x.n_elements(),
                n_cols * n_in
            ),
        });
    }
    if y.n_elements() as usize != n_cols * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_nc",
            detail: format!(
                "y.n_elements={} != n_cols*n_out={}",
                y.n_elements(),
                n_cols * n_out
            ),
        });
    }
    if weight.n_elements() as usize != n_in * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_nc",
            detail: format!(
                "weight.n_elements={} != n_in*n_out={}",
                weight.n_elements(),
                n_in * n_out
            ),
        });
    }
    let name = match n_cols {
        2 => "kernel_mat_vec_q6_K_nc2_f32",
        4 => "kernel_mat_vec_q6_K_nc4_f32",
        8 => "kernel_mat_vec_q6_K_nc8_f32",
        _ => {
            return Err(MetalError::BadShape {
                kernel: "mat_vec_q6_k_nc",
                detail: format!("n_cols={n_cols} not in {{2, 4, 8}}"),
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

    const ROWS_PER_TG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(ROWS_PER_TG),
            height: 1,
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

/// Dtype-routing wrapper for the multi-column mat-vec family (Q4_K and
/// Q6_K today). Mirrors `encode_mat_vec_dispatch` semantics for the
/// N-column case; returns `BadShape` for unsupported dtypes so callers
/// can fall back explicitly.
pub fn encode_mat_vec_nc_dispatch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    match weight.dtype {
        GgmlType::Q4_K => encode_mat_vec_q4_k_nc_f32(ctx, enc, weight, x, y, n_in, n_out, n_cols),
        GgmlType::Q6_K => encode_mat_vec_q6_k_nc_f32(ctx, enc, weight, x, y, n_in, n_out, n_cols),
        other => Err(MetalError::BadShape {
            kernel: "mat_vec_nc_dispatch",
            detail: format!("unsupported dtype {other:?} (Q4_K | Q6_K)"),
        }),
    }
}

/// v0.500 sweep B1 (bench-only): nc2 with 4 row-pairs per TG (8 SGs, 256
/// threads, 8 rows/TG). Same layouts and E0 body as
/// [`encode_mat_vec_q4_k_nc_f32`] at `n_cols = 2`.
pub fn encode_mat_vec_q4_k_nc2_rp4_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) || weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc2_rp4",
            detail: format!("n_in={n_in} % 256 != 0 or dtype {:?}", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc2_rp4",
            detail: format!("x/y dtype = {:?}/{:?}, expected F32/F32", x.dtype, y.dtype),
        });
    }
    if weight.n_elements() as usize != n_in * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc2_rp4",
            detail: format!(
                "weight.n_elements={} != n_in*n_out={}",
                weight.n_elements(),
                n_in * n_out
            ),
        });
    }
    if x.n_elements() as usize != 2 * n_in || y.n_elements() as usize != 2 * n_out {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q4_k_nc2_rp4",
            detail: format!(
                "x/y elements {}/{} != 2 * n_in/n_out",
                x.n_elements(),
                y.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_q4_K_nc2_rp4_f32")?;
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
            width: n_out.div_ceil(8),
            height: 1,
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

/// Fused SwiGLU FFN dispatch for Q4_K weights.
///
/// Replaces the 3-dispatch sequence:
///   mat_vec_q4_K(W_gate, x) -> gate
///   mat_vec_q4_K(W_up,   x) -> up
///   silu_mul(gate, up)      -> inner
/// with a single kernel that:
///   * reads x ONCE per super-block (shared across both gate and up paths)
///   * eliminates the n_out-element `gate` and `up` intermediate buffers
///   * fuses silu(gate) * up into the final lane-0 write
///
/// At the 27B FFN shape (n_in=5120, n_out=17408), this saves ~140 KB of
/// intermediate writes+reads per layer × 48 layers = ~6.5 MB/token.
///
/// CPU oracle: equivalent to the unfused 3-dispatch sequence (validated
/// by `ffn_swiglu_q4_K_matches_unfused`).
#[allow(non_snake_case)]
pub fn encode_ffn_swiglu_q4_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "ffn_swiglu_q4_K",
            detail: format!("n_in={n_in} not divisible by 256 (Q4_K super-block)"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "ffn_swiglu_q4_K",
            detail: format!(
                "expected Q4_K weights, got gate={:?} up={:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in {
        return Err(MetalError::BadShape {
            kernel: "ffn_swiglu_q4_K",
            detail: format!("x.n={} != n_in={n_in}", x.n_elements()),
        });
    }
    if inner.n_elements() as usize != n_out {
        return Err(MetalError::BadShape {
            kernel: "ffn_swiglu_q4_K",
            detail: format!("inner.n={} != n_out={n_out}", inner.n_elements()),
        });
    }

    let pso = ctx.pipeline("kernel_ffn_swiglu_q4_K_f32")?;
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
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, inner);

    // Same threadgroup shape as kernel_mat_vec_q4_K_f32: NR0=2 rows per
    // simdgroup, NSG=2 simdgroups per threadgroup.
    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Q5_K mat-vec, same API shape as [`encode_mat_vec_q4_k_f32`].
/// Block size 176 bytes / 256 elements.
pub fn encode_mat_vec_q5_k_f32(
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
            kernel: "mat_vec_q5_k",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q5_k",
            detail: format!("weight.dtype = {:?}, expected Q5_K", weight.dtype),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_q5_K_f32")?;
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

    // NR0=1, NSG=2 → 2 output rows per threadgroup.
    const NR0: usize = 1;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Q8_0 mat-vec, same API shape as [`encode_mat_vec_q4_k_f32`].
///
/// Q8_0 super-block is QK8_0=32 elements (vs QK_K=256 for Q4_K/Q5_K/Q6_K).
/// The kernel still requires `n_in % 32 == 0`. Used by v0.73b.0 to
/// switch the DFlash drafter from F32-dequant to native Q8_0 storage
/// (drafter weight footprint 7.4 GB → 1.85 GB, eliminates per-token
/// re-read of the dequant'd F32 weights at hot decode).
pub(crate) fn mat_vec_q8_0_lcpp_enabled() -> bool {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    let override_value =
        *OVERRIDE.get_or_init(|| match std::env::var("QWEN_MATVEC_Q8_0_LCPP").as_deref() {
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO") => Some(false),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") => Some(true),
            _ => None,
        });
    override_value.unwrap_or(true)
}

pub fn encode_mat_vec_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q8_0",
            detail: format!("n_in={n_in} not divisible by 32 (Q8_0 super-block)"),
        });
    }
    if weight.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q8_0",
            detail: format!("weight.dtype = {:?}, expected Q8_0", weight.dtype),
        });
    }
    let use_lcpp = mat_vec_q8_0_lcpp_enabled();
    let pso = ctx.pipeline(if use_lcpp {
        "kernel_mat_vec_q8_0_f32_lcpp"
    } else {
        "kernel_mat_vec_q8_0_f32"
    })?;
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

    let (nr0, nsg) = if use_lcpp { (2, 4) } else { (1, 2) };
    if use_lcpp {
        enc.set_threadgroup_memory(0, 32 * nr0 * std::mem::size_of::<f32>());
    }
    let rows_per_threadgroup = if use_lcpp { nr0 } else { nr0 * nsg };
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(rows_per_threadgroup),
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

/// Group-axis Q8_0 GEMV with the exact singleton `_lcpp` accumulation body.
/// Grid depth indexes `n_groups` consecutive weight blocks, input slices, and
/// output slices, so one dispatch replaces `n_groups` sequential singleton
/// dispatches with bitwise-identical per-group results.
pub fn encode_mat_vec_q8_0_grouped_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_groups: usize,
) -> Result<(), MetalError> {
    if n_groups == 0 || !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q8_0_grouped",
            detail: format!("n_groups={n_groups} must be nonzero and n_in={n_in} divisible by 32"),
        });
    }
    let expected_weight = n_groups
        .checked_mul(n_in)
        .and_then(|value| value.checked_mul(n_out));
    let expected_input = n_groups.checked_mul(n_in);
    let expected_output = n_groups.checked_mul(n_out);
    if weight.dtype != GgmlType::Q8_0
        || x.dtype != GgmlType::F32
        || y.dtype != GgmlType::F32
        || !y.is_writable()
        || expected_weight.is_none_or(|expected| weight.n_elements() as usize != expected)
        || expected_input.is_none_or(|expected| x.n_elements() as usize != expected)
        || expected_output.is_none_or(|expected| y.n_elements() as usize != expected)
        || u32::try_from(n_in).is_err()
        || u32::try_from(n_out).is_err()
        || u32::try_from(n_groups).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q8_0_grouped",
            detail: format!(
                "expected Q8_0 weight [{n_groups}x{n_in}x{n_out}] with F32 [{n_groups}x{n_in}] -> [{n_groups}x{n_out}], got {:?} w={} x={} y={}",
                weight.dtype,
                weight.n_elements(),
                x.n_elements(),
                y.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_q8_0_f32_lcpp_grouped")?;
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
    const NR0: usize = 2;
    const NSG: usize = 4;
    enc.set_threadgroup_memory(0, 32 * NR0 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0),
            height: 1,
            depth: n_groups,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn encode_ds4_compressor_pair_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kv_weight: &MetalTensor,
    score_weight: &MetalTensor,
    x: &MetalTensor,
    projected_score: &MetalTensor,
    ape: &MetalTensor,
    kv_state_row: &MetalTensor,
    score_state_row: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    let expected_weights = n_in.checked_mul(n_out);
    let expected_weight_bytes = n_in
        .checked_div(32)
        .and_then(|blocks| blocks.checked_mul(34))
        .and_then(|row_bytes| row_bytes.checked_mul(n_out));
    let valid_f32_input = |tensor: &MetalTensor, writable: bool| {
        tensor.dtype == GgmlType::F32
            && tensor.shape == [n_out as u64]
            && (!writable || tensor.is_writable())
            && tensor.n_elements() as usize == n_out
            && tensor_physical_range_valid(tensor, n_out.saturating_mul(4), 4)
    };
    if n_in == 0
        || n_out == 0
        || !n_in.is_multiple_of(32)
        || kv_weight.dtype != GgmlType::Q8_0
        || score_weight.dtype != GgmlType::Q8_0
        || kv_weight.shape != [n_in as u64, n_out as u64]
        || score_weight.shape != [n_in as u64, n_out as u64]
        || x.dtype != GgmlType::F32
        || x.shape != [n_in as u64]
        || x.n_elements() as usize != n_in
        || expected_weights.is_none_or(|expected| {
            kv_weight.n_elements() as usize != expected
                || score_weight.n_elements() as usize != expected
        })
        || !tensor_physical_range_valid(kv_weight, expected_weight_bytes.unwrap_or(usize::MAX), 2)
        || !tensor_physical_range_valid(
            score_weight,
            expected_weight_bytes.unwrap_or(usize::MAX),
            2,
        )
        || !tensor_physical_range_valid(x, n_in.saturating_mul(4), 4)
        || !valid_f32_input(projected_score, true)
        || !valid_f32_input(ape, false)
        || !valid_f32_input(kv_state_row, true)
        || !valid_f32_input(score_state_row, true)
        || [projected_score, kv_state_row, score_state_row]
            .iter()
            .any(|output| {
                tensor_ranges_overlap(output, kv_weight)
                    || tensor_ranges_overlap(output, score_weight)
                    || tensor_ranges_overlap(output, x)
                    || tensor_ranges_overlap(output, ape)
            })
        || tensor_ranges_overlap(projected_score, kv_state_row)
        || tensor_ranges_overlap(projected_score, score_state_row)
        || tensor_ranges_overlap(kv_state_row, score_state_row)
        || u32::try_from(n_in).is_err()
        || u32::try_from(n_out).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: "ds4_compressor_pair_q8_0",
            detail: format!(
                "expected Q8_0 KV/score [{n_in},{n_out}], F32 x={n_in}, and distinct writable F32 score/state rows of {n_out}; got {:?}/{:?} x={:?}/{} projected_score={:?} state={:?}/{:?}",
                kv_weight.dtype,
                score_weight.dtype,
                x.dtype,
                x.n_elements(),
                projected_score.shape,
                kv_state_row.shape,
                score_state_row.shape,
            ),
        });
    }

    let pso = ctx.pipeline("kernel_ds4_compressor_pair_q8_0_f32_lcpp")?;
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
    enc.set_tensor(1, kv_weight);
    enc.set_tensor(2, score_weight);
    enc.set_tensor(3, x);
    enc.set_tensor(4, projected_score);
    enc.set_tensor(5, ape);
    enc.set_tensor(6, kv_state_row);
    enc.set_tensor(7, score_state_row);
    const NR0: usize = 2;
    const NSG: usize = 4;
    enc.set_threadgroup_memory(0, 32 * NR0 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0),
            height: 1,
            depth: 2,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Token-axis Q8_0 GEMV with the exact singleton `_lcpp` accumulation body.
/// Each grid row owns one activation row; the weight traversal remains one
/// dispatch without half-staging persistent cache-producing projections.
pub fn encode_mat_vec_q8_0_batch_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if n_tokens == 0 || !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q8_0_batch",
            detail: format!("n_tokens={n_tokens} must be nonzero and n_in={n_in} divisible by 32"),
        });
    }
    let expected_weight = n_in.checked_mul(n_out);
    let expected_input = n_tokens.checked_mul(n_in);
    let expected_output = n_tokens.checked_mul(n_out);
    if weight.dtype != GgmlType::Q8_0
        || x.dtype != GgmlType::F32
        || y.dtype != GgmlType::F32
        || expected_weight.is_none_or(|expected| weight.n_elements() as usize != expected)
        || expected_input.is_none_or(|expected| x.n_elements() as usize != expected)
        || expected_output.is_none_or(|expected| y.n_elements() as usize != expected)
        || u32::try_from(n_in).is_err()
        || u32::try_from(n_out).is_err()
        || u32::try_from(n_tokens).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q8_0_batch",
            detail: format!(
                "expected Q8_0 weight and F32 [{n_tokens},{n_in}] -> [{n_tokens},{n_out}], got {:?} x={} y={}",
                weight.dtype,
                x.n_elements(),
                y.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_q8_0_f32_lcpp_batch")?;
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
    const NR0: usize = 2;
    const NSG: usize = 4;
    enc.set_threadgroup_memory(0, 32 * NR0 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0),
            height: n_tokens,
            depth: 1,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_shared_swiglu_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "shared_swiglu_q8_0",
            detail: format!("n_in={n_in} not divisible by 32 (Q8_0 super-block)"),
        });
    }
    if gate_weight.dtype != GgmlType::Q8_0 || up_weight.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "shared_swiglu_q8_0",
            detail: format!(
                "gate/up dtype = {:?}/{:?}, expected Q8_0/Q8_0",
                gate_weight.dtype, up_weight.dtype
            ),
        });
    }
    let pso = ctx.pipeline("kernel_shared_swiglu_q8_0_f32_lcpp")?;
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
    enc.set_tensor(1, gate_weight);
    enc.set_tensor(2, up_weight);
    enc.set_tensor(3, x);
    enc.set_tensor(4, y);

    let nr0 = 2usize;
    let nsg = 4usize;
    enc.set_threadgroup_memory(0, 32 * 2 * nr0 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(nr0),
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

pub fn encode_ds4_shared_swiglu_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    clamp: f32,
) -> Result<(), MetalError> {
    let expected_weights = n_in.checked_mul(n_out);
    let expected_weight_bytes = n_in
        .checked_div(32)
        .and_then(|blocks| blocks.checked_mul(34))
        .and_then(|row_bytes| row_bytes.checked_mul(n_out));
    if n_in == 0
        || n_out == 0
        || !n_in.is_multiple_of(32)
        || gate_weight.dtype != GgmlType::Q8_0
        || up_weight.dtype != GgmlType::Q8_0
        || gate_weight.shape != [n_in as u64, n_out as u64]
        || up_weight.shape != [n_in as u64, n_out as u64]
        || x.dtype != GgmlType::F32
        || x.shape != [n_in as u64]
        || y.dtype != GgmlType::F32
        || y.shape != [n_out as u64]
        || !y.is_writable()
        || expected_weights.is_none_or(|expected| {
            gate_weight.n_elements() as usize != expected
                || up_weight.n_elements() as usize != expected
        })
        || x.n_elements() as usize != n_in
        || y.n_elements() as usize != n_out
        || !tensor_physical_range_valid(gate_weight, expected_weight_bytes.unwrap_or(usize::MAX), 2)
        || !tensor_physical_range_valid(up_weight, expected_weight_bytes.unwrap_or(usize::MAX), 2)
        || !tensor_physical_range_valid(x, n_in.saturating_mul(4), 4)
        || !tensor_physical_range_valid(y, n_out.saturating_mul(4), 4)
        || tensor_ranges_overlap(y, gate_weight)
        || tensor_ranges_overlap(y, up_weight)
        || tensor_ranges_overlap(y, x)
        || !clamp.is_finite()
        || clamp <= 0.0
        || u32::try_from(n_in).is_err()
        || u32::try_from(n_out).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: "ds4_shared_swiglu_q8_0",
            detail: format!(
                "expected Q8_0 gate/up [{n_in},{n_out}], F32 x={n_in}, writable F32 y={n_out}, and positive finite clamp; got {:?}/{:?} w={}/{} x={:?}/{} y={:?}/{} writable={} clamp={clamp}",
                gate_weight.dtype,
                up_weight.dtype,
                gate_weight.n_elements(),
                up_weight.n_elements(),
                x.dtype,
                x.n_elements(),
                y.dtype,
                y.n_elements(),
                y.is_writable(),
            ),
        });
    }
    let pso = ctx.pipeline("kernel_ds4_shared_swiglu_q8_0_f32_lcpp")?;
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
    enc.set_tensor(1, gate_weight);
    enc.set_tensor(2, up_weight);
    enc.set_tensor(3, x);
    enc.set_tensor(4, y);
    const NR0: usize = 2;
    const NSG: usize = 4;
    enc.set_threadgroup_memory(0, 32 * 2 * NR0 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// One-shot Q8_0 mat-vec for tests.
pub fn mat_vec_q8_0_f32_readback_for_test(
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
        GgmlType::Q8_0,
    )?;
    let x_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(x),
        vec![n_in as u64],
        GgmlType::F32,
    )?;
    let y_t = MetalTensor::zeros_f32(ctx, vec![n_out as u64])?;
    one_shot(ctx, |enc| {
        encode_mat_vec_q8_0_f32(ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
    })?;
    Ok(read_back_f32(&y_t.buffer, n_out))
}

/// One-shot Q5_K mat-vec for tests.
pub fn mat_vec_q5_k_f32_readback_for_test(
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
        GgmlType::Q5_K,
    )?;
    let x_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(x),
        vec![n_in as u64],
        GgmlType::F32,
    )?;
    let y_t = MetalTensor::zeros_f32(ctx, vec![n_out as u64])?;
    one_shot(ctx, |enc| {
        encode_mat_vec_q5_k_f32(ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
    })?;
    Ok(read_back_f32(&y_t.buffer, n_out))
}

/// Q6_K mat-vec, same API shape as [`encode_mat_vec_q4_k_f32`].
pub fn encode_mat_vec_q6_k_f32(
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
            kernel: "mat_vec_q6_k",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k",
            detail: format!("weight.dtype = {:?}, expected Q6_K", weight.dtype),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_q6_K_f32")?;
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

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_ds4_shared_swiglu_q6_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    clamp: f32,
) -> Result<(), MetalError> {
    let expected_weights = n_in.checked_mul(n_out);
    let expected_weight_bytes = n_in
        .checked_div(256)
        .and_then(|blocks| blocks.checked_mul(210))
        .and_then(|row_bytes| row_bytes.checked_mul(n_out));
    let scratch_elements = n_out.checked_mul(3);
    if n_in == 0
        || n_out == 0
        || !n_in.is_multiple_of(256)
        || gate_weight.dtype != GgmlType::Q6_K
        || up_weight.dtype != GgmlType::Q6_K
        || gate_weight.shape != [n_in as u64, n_out as u64]
        || up_weight.shape != [n_in as u64, n_out as u64]
        || x.dtype != GgmlType::F32
        || x.shape != [n_in as u64]
        || y.dtype != GgmlType::F32
        || scratch_elements.is_none_or(|elements| y.shape != [elements as u64])
        || !y.is_writable()
        || expected_weights.is_none_or(|expected| {
            gate_weight.n_elements() as usize != expected
                || up_weight.n_elements() as usize != expected
        })
        || x.n_elements() as usize != n_in
        || scratch_elements.is_none_or(|elements| y.n_elements() as usize != elements)
        || !tensor_physical_range_valid(gate_weight, expected_weight_bytes.unwrap_or(usize::MAX), 2)
        || !tensor_physical_range_valid(up_weight, expected_weight_bytes.unwrap_or(usize::MAX), 2)
        || !tensor_physical_range_valid(x, n_in.saturating_mul(4), 4)
        || !tensor_physical_range_valid(
            y,
            scratch_elements.unwrap_or(usize::MAX).saturating_mul(4),
            4,
        )
        || tensor_ranges_overlap(y, gate_weight)
        || tensor_ranges_overlap(y, up_weight)
        || tensor_ranges_overlap(y, x)
        || !clamp.is_finite()
        || clamp <= 0.0
        || u32::try_from(n_in).is_err()
        || u32::try_from(n_out).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: "ds4_shared_swiglu_q6_k",
            detail: format!(
                "expected Q6_K gate/up [{n_in},{n_out}], F32 x={n_in}, writable F32 y>={}; got {:?}/{:?} w={}/{} x={:?}/{} y={:?}/{} writable={} clamp={clamp}",
                n_out.saturating_mul(3),
                gate_weight.dtype,
                up_weight.dtype,
                gate_weight.n_elements(),
                up_weight.n_elements(),
                x.dtype,
                x.n_elements(),
                y.dtype,
                y.n_elements(),
                y.is_writable(),
            ),
        });
    }
    let pso = ctx.pipeline("kernel_ds4_shared_swiglu_q6_K_f32")?;
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
    enc.set_tensor(1, gate_weight);
    enc.set_tensor(2, up_weight);
    enc.set_tensor(3, x);
    enc.set_tensor(4, y);
    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(crate) fn tensor_physical_range_valid(
    tensor: &MetalTensor,
    expected_bytes: usize,
    alignment: u64,
) -> bool {
    tensor.offset.is_multiple_of(alignment)
        && u64::try_from(expected_bytes)
            .ok()
            .and_then(|bytes| tensor.offset.checked_add(bytes))
            .is_some_and(|end| end <= tensor.buffer.length() as u64)
}

pub(crate) fn tensor_ranges_overlap(left: &MetalTensor, right: &MetalTensor) -> bool {
    if Retained::as_ptr(&left.buffer) != Retained::as_ptr(&right.buffer) {
        return false;
    }
    let left_end = left.offset.saturating_add(left.n_bytes());
    let right_end = right.offset.saturating_add(right.n_bytes());
    left.offset < right_end && right.offset < left_end
}

pub(crate) fn tensor_byte_ranges_overlap(
    left: &MetalTensor,
    left_bytes: usize,
    right: &MetalTensor,
    right_bytes: usize,
) -> bool {
    if Retained::as_ptr(&left.buffer) != Retained::as_ptr(&right.buffer) {
        return false;
    }
    let Ok(left_bytes) = u64::try_from(left_bytes) else {
        return true;
    };
    let Ok(right_bytes) = u64::try_from(right_bytes) else {
        return true;
    };
    let Some(left_end) = left.offset.checked_add(left_bytes) else {
        return true;
    };
    let Some(right_end) = right.offset.checked_add(right_bytes) else {
        return true;
    };
    left.offset < right_end && right.offset < left_end
}

/// Token-axis Q6_K GEMV with the exact singleton accumulation body.
pub fn encode_mat_vec_q6_k_batch_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    let expected_weight = n_in.checked_mul(n_out);
    let expected_input = n_tokens.checked_mul(n_in);
    let expected_output = n_tokens.checked_mul(n_out);
    if n_tokens == 0
        || n_in == 0
        || n_out == 0
        || !n_in.is_multiple_of(256)
        || weight.dtype != GgmlType::Q6_K
        || x.dtype != GgmlType::F32
        || y.dtype != GgmlType::F32
        || expected_weight.is_none_or(|expected| weight.n_elements() as usize != expected)
        || expected_input.is_none_or(|expected| x.n_elements() as usize != expected)
        || expected_output.is_none_or(|expected| y.n_elements() as usize != expected)
        || u32::try_from(n_in).is_err()
        || u32::try_from(n_out).is_err()
        || u32::try_from(n_tokens).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_q6_k_batch",
            detail: format!(
                "expected Q6_K weight and F32 [{n_tokens},{n_in}] -> [{n_tokens},{n_out}], got {:?} x={} y={}",
                weight.dtype,
                x.n_elements(),
                y.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_mat_vec_q6_K_f32_batch")?;
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
    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: n_tokens,
            depth: 1,
        },
        MTLSize {
            width: NSG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
