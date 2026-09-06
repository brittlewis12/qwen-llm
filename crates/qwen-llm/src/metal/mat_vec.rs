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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;

    #[test]
    fn mxfp4_mat_vec_dispatch_gpu_matches_reference_with_offsets() {
        let ctx = match metal_test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        const N_IN: usize = 64;
        const N_OUT: usize = 3;
        let mut all_indices = [0u8; 32];
        for (i, index) in all_indices.iter_mut().enumerate() {
            *index = (i % 16) as u8;
        }
        let zero_indices = [0u8; 32];
        let blocks = [
            encode_mxfp4_block(0, &all_indices),
            encode_mxfp4_block(127, &zero_indices),
            encode_mxfp4_block(1, &all_indices),
            encode_mxfp4_block(128, &zero_indices),
            encode_mxfp4_block(126, &zero_indices),
            encode_mxfp4_block(127, &all_indices),
        ];
        let weight_bytes = blocks.concat();
        let desc = TensorDesc {
            name: "mxfp4_test".into(),
            shape: vec![N_IN as u64, N_OUT as u64],
            dtype: GgmlType::MXFP4,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: weight_bytes.len() as u64,
        };
        let decoded = crate::codec::dequant_to_f32(&desc, &weight_bytes)
            .expect("llama.cpp MXFP4 reference dequantization");
        let weight = offset_tensor(
            &ctx,
            32,
            &weight_bytes,
            31,
            vec![N_IN as u64, N_OUT as u64],
            GgmlType::MXFP4,
        );
        let mut x_values = vec![0.0f32; N_IN];
        for (i, x) in x_values[..32].iter_mut().enumerate() {
            *x = if i % 2 == 0 {
                2.0f32.powi(120)
            } else {
                -2.0f32.powi(120)
            };
        }
        for (i, x) in x_values[32..].iter_mut().enumerate() {
            *x = (i as f32 - 15.5) / 8.0;
        }
        let x = offset_tensor(
            &ctx,
            16,
            bytemuck::cast_slice(&x_values),
            24,
            vec![N_IN as u64],
            GgmlType::F32,
        );
        let y = offset_tensor(
            &ctx,
            32,
            &[0u8; N_OUT * size_of::<f32>()],
            16,
            vec![N_OUT as u64],
            GgmlType::F32,
        );

        let cmd = ctx.queue.commandBuffer().expect("MXFP4 command buffer");
        let enc = KernelEncoder::begin(&cmd);
        crate::metal_forward::encode_mat_vec_dispatch(&ctx, &enc, &weight, &x, &y, N_IN, N_OUT)
            .expect("MXFP4 dispatch arm");
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        let mut expected = vec![0.0f32; N_OUT];
        for row in 0..N_OUT {
            expected[row] = decoded[row * N_IN..(row + 1) * N_IN]
                .iter()
                .zip(&x_values)
                .map(|(weight, x)| weight * x)
                .sum();
        }
        let actual = tensor_f32_at_offset(&y);
        for (row, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = 2.0e-5 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "row {row}: got {got}, want {want}"
            );
        }
        assert_eq!(mxfp4_scale(0).to_bits(), 0x0020_0000);
        assert_eq!(mxfp4_scale(1).to_bits(), 0x0040_0000);
        assert!(!crate::metal_forward::weight_dtype_kept_native(
            GgmlType::MXFP4
        ));
    }

    #[test]
    fn mxfp4_mat_vec_host_rejects_invalid_shapes_and_ranges() {
        let ctx = match metal_test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let weight =
            MetalTensor::from_bytes(&ctx, &[0u8; 34], vec![64, 1], GgmlType::MXFP4).unwrap();
        let x = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
        let y = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let cmd = ctx
            .queue
            .commandBuffer()
            .expect("validation command buffer");
        let enc = KernelEncoder::begin(&cmd);
        assert!(encode_mat_vec_mxfp4_f32(&ctx, &enc, &weight, &x, &y, 48, 1).is_err());
        let mut wrong_weight_shape = weight.clone();
        wrong_weight_shape.shape = vec![32, 1];
        assert!(encode_mat_vec_mxfp4_f32(&ctx, &enc, &wrong_weight_shape, &x, &y, 64, 1).is_err());
        let wrong_x = MetalTensor::zeros_f32(&ctx, vec![32]).unwrap();
        assert!(encode_mat_vec_mxfp4_f32(&ctx, &enc, &weight, &wrong_x, &y, 64, 1).is_err());
        for which in 0..3 {
            let mut bad_weight = weight.clone();
            let mut bad_x = x.clone();
            let mut bad_y = y.clone();
            match which {
                0 => bad_weight.offset = bad_weight.buffer.length() as u64 - 1,
                1 => bad_x.offset = bad_x.buffer.length() as u64 - 1,
                _ => bad_y.offset = bad_y.buffer.length() as u64 - 1,
            }
            assert!(
                encode_mat_vec_mxfp4_f32(&ctx, &enc, &bad_weight, &bad_x, &bad_y, 64, 1).is_err()
            );
        }
        enc.end();
    }

    #[test]
    fn mat_vec_f32_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        for &(n_in, n_out) in &[
            (1024usize, 248_320usize),
            (1024, 6144),
            (5120, 17408),
            (5120, 5120),
        ] {
            let w: Vec<f32> = (0..n_in * n_out)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-3)
                .collect();
            let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
            let cpu = crate::forward::mat_vec_pub(&w, n_in, n_out, &x);
            let gpu = mat_vec_f32_readback_for_test(&ctx, &w, &x, n_in, n_out).expect("gpu");
            let max_abs = gpu
                .iter()
                .zip(cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[mat_vec n_in={n_in} n_out={n_out}] max|Δ|={max_abs:.2e}");
            assert!(max_abs < 1e-3);
        }
    }

    #[test]
    fn mat_vec_f32_sigmoid_matches_cpu() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let n_in = 512usize;
        let n_out = 48usize;
        let w: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i % 29) as f32 - 14.0) * 2e-3)
            .collect();
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 17) as f32 - 8.0) * 3e-2).collect();
        let mat = crate::forward::mat_vec_pub(&w, n_in, n_out, &x);
        let cpu: Vec<f32> = mat.into_iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        let w_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&w),
            vec![n_in as u64, n_out as u64],
            GgmlType::F32,
        )
        .unwrap();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .unwrap();
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_mat_vec_f32_sigmoid(&ctx, enc, &w_t, &x_t, &y_t, n_in, n_out)
        })
        .unwrap();
        let gpu = read_back_f32(&y_t.buffer, n_out);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_abs < 1e-5, "fused beta sigmoid drift {max_abs}");
    }

    #[test]
    fn mat_vec_q4_k_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let q4k = g
            .tensors
            .iter()
            .find(|t| {
                t.name.starts_with("blk.0.")
                    && t.dtype == GgmlType::Q4_K
                    && t.shape.len() == 2
                    && t.shape[0] % 256 == 0
            })
            .expect("no Q4_K tensor");
        let n_in = q4k.shape[0] as usize;
        let n_out = q4k.shape[1] as usize;
        eprintln!("[q4_k-test] {} shape=[{n_in}, {n_out}]", q4k.name);

        let weight_f32 = crate::codec::dequant_to_f32(q4k, g.slice(q4k)).expect("dequant");
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let gpu =
            mat_vec_q4_k_f32_readback_for_test(&ctx, g.slice(q4k), &x, n_in, n_out).expect("gpu");
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q4_k] max|Δ|={max_abs:.2e}");
        assert!(max_abs < 1e-2);
    }

    #[test]
    fn mat_vec_trellis3_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        // Small shape with a partial last threadgroup (66 % 4 != 0) to
        // exercise the row guards, and multiple groups along n_in.
        let n_in = 512;
        let n_out = 66;
        let syn = trellis3_synthetic(n_in, n_out, 0x007E_1115);
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 29) as f32 - 14.0) * 3e-2).collect();
        for variant in [
            Trellis3Variant::ThreeInst,
            Trellis3Variant::ThreeInstV2,
            Trellis3Variant::Lut8x2,
            Trellis3Variant::HybV2,
            Trellis3Variant::ThreeInstG,
            Trellis3Variant::ThreeInstV2G,
            Trellis3Variant::ThreeInstGNsg4,
            Trellis3Variant::ThreeInstGNr4,
            Trellis3Variant::ThreeInstDG,
        ] {
            let cpu = trellis3_cpu_reference(variant, &syn, &x, n_in, n_out);
            let gpu = mat_vec_trellis3_f32_readback_for_test(&ctx, variant, &syn, &x, n_in, n_out)
                .expect("gpu");
            let mut dot = 0f64;
            let mut na = 0f64;
            let mut nb2 = 0f64;
            let mut max_delta = 0f32;
            for (a, b) in gpu.iter().zip(cpu.iter()) {
                dot += (*a as f64) * (*b as f64);
                na += (*a as f64) * (*a as f64);
                nb2 += (*b as f64) * (*b as f64);
                max_delta = max_delta.max((a - b).abs());
            }
            let cos = dot / (na.sqrt() * nb2.sqrt()).max(1e-30);
            let max_abs_y = cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
            eprintln!(
                "[trellis3 {}] max|Δ|={max_delta:.3e} max|y|={max_abs_y:.3e} cos={cos:.9}",
                variant.label()
            );
            assert!(cos >= 0.999999, "{} cosine {cos}", variant.label());
            assert!(
                max_delta <= 1e-3 * max_abs_y.max(1e-3),
                "{} max delta {max_delta} vs max|y| {max_abs_y}",
                variant.label()
            );
        }
    }

    #[test]
    fn mat_vec_lowbit_nc2_synthetic_matches_singletons_and_rejects_bad_views() {
        type Encoder = fn(
            &MetalContext,
            &KernelEncoder,
            &MetalTensor,
            &MetalTensor,
            &MetalTensor,
            usize,
            usize,
        ) -> Result<(), MetalError>;
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("metal context: {error}"),
        };
        const N_IN: usize = 256;
        const N_OUT: usize = 9;
        let cases: [(GgmlType, usize, Encoder, Encoder); 2] = [
            (
                GgmlType::IQ2_S,
                82,
                encode_mat_vec_iq2_s_nc2_f32,
                encode_mat_vec_iq2_s_f32,
            ),
            (
                GgmlType::IQ3_S,
                110,
                encode_mat_vec_iq3_s_nc2_f32,
                encode_mat_vec_iq3_s_f32,
            ),
        ];
        for (dtype, block_bytes, encode_nc2, encode_single) in cases {
            let mut weight_bytes = vec![0u8; N_OUT * block_bytes];
            for row in 0..N_OUT {
                let block = &mut weight_bytes[row * block_bytes..(row + 1) * block_bytes];
                for (index, byte) in block.iter_mut().enumerate().skip(2) {
                    *byte = ((row * 37 + index * 19 + 11) & 0xff) as u8;
                }
                block[..2].copy_from_slice(
                    &half::f16::from_f32(0.03125 + row as f32 * 0.001)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
            let weight = offset_tensor(
                &ctx,
                32,
                &weight_bytes,
                18,
                vec![N_IN as u64, N_OUT as u64],
                dtype,
            );
            let inputs = (0..2 * N_IN)
                .map(|index| ((index * 13 + index / N_IN * 7) % 97) as f32 * 1e-3 - 0.04)
                .collect::<Vec<_>>();
            let input = offset_tensor(
                &ctx,
                16,
                bytemuck::cast_slice(&inputs),
                12,
                vec![2, N_IN as u64],
                GgmlType::F32,
            );
            let output_bytes = vec![0u8; 2 * N_OUT * size_of::<f32>()];
            let nc2 = offset_tensor(
                &ctx,
                32,
                &output_bytes,
                20,
                vec![(2 * N_OUT) as u64],
                GgmlType::F32,
            );
            let sequential = offset_tensor(
                &ctx,
                32,
                &output_bytes,
                20,
                vec![(2 * N_OUT) as u64],
                GgmlType::F32,
            );

            one_shot(&ctx, |enc| {
                encode_nc2(&ctx, enc, &weight, &input, &nc2, N_IN, N_OUT)?;
                for row in 0..2 {
                    let input_row = input.view_subrange((row * N_IN) as u64, vec![N_IN as u64]);
                    let output_row =
                        sequential.view_subrange((row * N_OUT) as u64, vec![N_OUT as u64]);
                    encode_single(&ctx, enc, &weight, &input_row, &output_row, N_IN, N_OUT)?;
                }
                Ok(())
            })
            .unwrap_or_else(|error| panic!("{dtype:?} synthetic NC2: {error}"));
            let candidate = tensor_f32_at_offset(&nc2);
            let reference = tensor_f32_at_offset(&sequential);
            assert!(
                candidate
                    .iter()
                    .zip(&reference)
                    .all(|(left, right)| left.to_bits() == right.to_bits()),
                "{dtype:?} synthetic NC2 differs from singleton rows"
            );
            assert_offset_guards(&nc2, 32, 20);
            assert_offset_guards(&sequential, 32, 20);

            let command = ctx.queue.commandBuffer().expect("validation command");
            let encoder = KernelEncoder::begin(&command);
            let mut misaligned_weight = weight.clone();
            misaligned_weight.offset += 1;
            assert!(
                encode_nc2(
                    &ctx,
                    &encoder,
                    &misaligned_weight,
                    &input,
                    &nc2,
                    N_IN,
                    N_OUT,
                )
                .is_err()
            );
            let mut read_only_output = nc2.clone();
            read_only_output.provenance = MetalTensorProvenance::RetainedGgufReadOnly;
            assert!(
                encode_nc2(
                    &ctx,
                    &encoder,
                    &weight,
                    &input,
                    &read_only_output,
                    N_IN,
                    N_OUT,
                )
                .is_err()
            );
            let mut short_output = nc2.clone();
            short_output.offset = short_output.buffer.length() as u64 - 4;
            assert!(
                encode_nc2(&ctx, &encoder, &weight, &input, &short_output, N_IN, N_OUT,).is_err()
            );
            encoder.end();
        }
    }

    #[test]
    #[ignore = "requires local Ridge IQ2_S/IQ3_S fixtures"]
    fn mat_vec_lowbit_nc2_matches_two_singleton_rows_bit_exact() {
        type Encoder = fn(
            &MetalContext,
            &KernelEncoder,
            &MetalTensor,
            &MetalTensor,
            &MetalTensor,
            usize,
            usize,
        ) -> Result<(), MetalError>;
        let ctx = MetalContext::new().expect("metal context");
        let path = "/Users/tito/models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf";
        let g = crate::gguf::GgufFile::open(path).expect("open Ridge fixture");
        let cases: [(GgmlType, usize, usize, Encoder, Encoder); 4] = [
            (
                GgmlType::IQ2_S,
                5120,
                17408,
                encode_mat_vec_iq2_s_nc2_f32,
                encode_mat_vec_iq2_s_f32,
            ),
            (
                GgmlType::IQ2_S,
                17408,
                5120,
                encode_mat_vec_iq2_s_nc2_f32,
                encode_mat_vec_iq2_s_f32,
            ),
            (
                GgmlType::IQ3_S,
                5120,
                17408,
                encode_mat_vec_iq3_s_nc2_f32,
                encode_mat_vec_iq3_s_f32,
            ),
            (
                GgmlType::IQ3_S,
                17408,
                5120,
                encode_mat_vec_iq3_s_nc2_f32,
                encode_mat_vec_iq3_s_f32,
            ),
        ];
        for (dtype, expected_n_in, expected_n_out, encode_nc2, encode_single) in cases {
            let w = g
                .tensors
                .iter()
                .find(|tensor| {
                    tensor.dtype == dtype
                        && tensor.shape.len() == 2
                        && tensor.shape == [expected_n_in as u64, expected_n_out as u64]
                })
                .unwrap_or_else(|| {
                    panic!("Ridge {dtype:?} matrix [{expected_n_in}, {expected_n_out}]")
                });
            let n_in = w.shape[0] as usize;
            let n_out = w.shape[1] as usize;
            let weight =
                MetalTensor::from_bytes(&ctx, g.slice(w), vec![n_in as u64, n_out as u64], dtype)
                    .unwrap_or_else(|error| panic!("{dtype:?} weight: {error}"));
            let inputs = (0..2 * n_in)
                .map(|i| (((i * 17 + i / n_in * 11) % 101) as f32 - 50.0) * 1e-3)
                .collect::<Vec<_>>();
            let input = offset_tensor(
                &ctx,
                32,
                bytemuck::cast_slice(&inputs),
                16,
                vec![2, n_in as u64],
                GgmlType::F32,
            );
            let output_bytes = vec![0u8; 2 * n_out * size_of::<f32>()];
            let nc2 = offset_tensor(
                &ctx,
                64,
                &output_bytes,
                32,
                vec![(2 * n_out) as u64],
                GgmlType::F32,
            );
            let sequential = offset_tensor(
                &ctx,
                64,
                &output_bytes,
                32,
                vec![(2 * n_out) as u64],
                GgmlType::F32,
            );

            one_shot(&ctx, |enc| {
                encode_nc2(&ctx, enc, &weight, &input, &nc2, n_in, n_out)?;
                for row in 0..2 {
                    let input_row = input.view_subrange((row * n_in) as u64, vec![n_in as u64]);
                    let output_row =
                        sequential.view_subrange((row * n_out) as u64, vec![n_out as u64]);
                    encode_single(&ctx, enc, &weight, &input_row, &output_row, n_in, n_out)?;
                }
                Ok(())
            })
            .unwrap_or_else(|error| panic!("{dtype:?} NC2 and singleton rows: {error}"));

            let nc2_values = tensor_f32_at_offset(&nc2);
            let sequential_values = tensor_f32_at_offset(&sequential);
            let max_abs = nc2_values
                .iter()
                .zip(&sequential_values)
                .map(|(candidate, reference)| (candidate - reference).abs())
                .fold(0.0f32, f32::max);
            assert!(
                nc2_values
                    .iter()
                    .zip(&sequential_values)
                    .all(|(candidate, reference)| candidate.to_bits() == reference.to_bits()),
                "{dtype:?} NC2 differs from singleton rows; max_abs={max_abs:.3e}"
            );
            eprintln!("[lowbit-nc2] dtype={dtype:?} max_abs={max_abs:.3e}");
            assert_offset_guards(&nc2, 64, 32);
            assert_offset_guards(&sequential, 64, 32);
        }
    }

    #[test]
    fn mat_vec_q5_k_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
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
            })
            .expect("no Q5_K tensor in 27B layer 0");
        let n_in = q5k.shape[0] as usize;
        let n_out = q5k.shape[1] as usize;
        eprintln!("[q5_k-test] {} shape=[{n_in}, {n_out}]", q5k.name);

        let weight_f32 = crate::codec::dequant_to_f32(q5k, g.slice(q5k)).expect("dequant");
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let gpu = mat_vec_q5_k_f32_readback_for_test(&ctx, g.slice(q5k), &x, n_in, n_out)
            .expect("metal q5k");
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q5_k] max|Δ|={max_abs:.2e}");
        assert!(max_abs < 1e-2);
    }

    /// v0.73b.0 gate: Q8_0 mat-vec correctness. Uses a real Q8_0 weight
    /// from the spiritbuun DFlash drafter GGUF (`blk.0.attn_q.weight`,
    /// shape `[5120, 4096]`). Same threshold as Q4_K/Q5_K/Q6_K mat-vec
    /// (`max|Δ| < 1e-2`).
    #[test]
    fn mat_vec_q8_0_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[q8_0-test] skipped — drafter GGUF missing");
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
            })
            .expect("no Q8_0 tensor in drafter blk.0");
        let n_in = q8.shape[0] as usize;
        let n_out = q8.shape[1] as usize;
        eprintln!("[q8_0-test] {} shape=[{n_in}, {n_out}]", q8.name);

        let weight_f32 = crate::codec::dequant_to_f32(q8, g.slice(q8)).expect("dequant");
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let gpu = mat_vec_q8_0_f32_readback_for_test(&ctx, g.slice(q8), &x, n_in, n_out)
            .expect("metal q8_0");
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q8_0] max|Δ|={max_abs:.2e}");
        assert!(max_abs < 1e-2);
    }

    #[test]
    fn mat_vec_q6_k_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
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
            })
            .expect("no Q6_K tensor");
        let n_in = q6k.shape[0] as usize;
        let n_out = q6k.shape[1] as usize;
        eprintln!("[q6_k-test] {} shape=[{n_in}, {n_out}]", q6k.name);
        let weight_f32 = crate::codec::dequant_to_f32(q6k, g.slice(q6k)).expect("dequant");
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 1e-2).collect();
        let cpu = crate::forward::mat_vec_pub(&weight_f32, n_in, n_out, &x);
        let gpu =
            mat_vec_q6_k_f32_readback_for_test(&ctx, g.slice(q6k), &x, n_in, n_out).expect("gpu");
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[q6_k] max|Δ|={max_abs:.2e}");
        assert!(max_abs < 1e-2);
    }

    /// v0.73c.2 A-lite GO/NO-GO bench. Compares the fused
    /// `ffn_swiglu_q4_K_mm_n16` (one dispatch per FFN layer) against
    /// the unfused `mat_mat_q4_K + mat_mat_q4_K + silu_mul` 3-dispatch
    /// sequence at production 27B 64-layer FFN shape (n_in=5120,
    /// n_out=17408, N=16, 64 layers). Codex's threshold for proceed
    /// is ratio ≤ 0.7 (fused must beat unfused by at least ~30%).
    ///
    /// Run: `cargo test --release --lib -p qwen-llm
    /// ffn_fused_swiglu_q4_K_amortization_vs_unfused --ignored -- --nocapture`
    #[test]
    #[ignore]
    #[allow(non_snake_case)]
    fn ffn_fused_swiglu_q4_K_amortization_vs_unfused() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[v0.73c.2-gate] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let gate = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate.weight" && t.dtype == GgmlType::Q4_K)
            .expect("ffn_gate Q4_K not found");
        let up = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up.weight" && t.dtype == GgmlType::Q4_K)
            .expect("ffn_up Q4_K not found");
        let n_in = gate.shape[0] as usize;
        let n_out = gate.shape[1] as usize;
        const N: usize = 16;
        let n_layers = 64usize; // 27B has 64 transformer blocks (48 GDN + 16 attn; FFN runs on all)
        let warmup = 5usize;
        let iters = 30usize;

        eprintln!("[v0.73c.2-gate] shape=[n_in={n_in}, n_out={n_out}] N={N} layers={n_layers}");

        let w_gate = MetalTensor::from_bytes(
            &ctx,
            g.slice(gate),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .unwrap();
        let w_up = MetalTensor::from_bytes(
            &ctx,
            g.slice(up),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .unwrap();
        let x_packed = MetalTensor::zeros_f32(&ctx, vec![(N * n_in) as u64]).unwrap();
        let inner_packed = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
        let gate_packed = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();
        let up_packed = MetalTensor::zeros_f32(&ctx, vec![(N * n_out) as u64]).unwrap();

        let bench_fused = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                encode_ffn_fused_swiglu_q4_K_mm_n16_f32(
                    &ctx,
                    &enc,
                    &w_gate,
                    &w_up,
                    &x_packed,
                    &inner_packed,
                    n_in,
                    n_out,
                )
                .unwrap();
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        let bench_unfused = || {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..n_layers {
                encode_mat_mat_q4_k_f32(
                    &ctx,
                    &enc,
                    &w_gate,
                    &x_packed,
                    &gate_packed,
                    n_in,
                    n_out,
                    N,
                )
                .unwrap();
                encode_mat_mat_q4_k_f32(&ctx, &enc, &w_up, &x_packed, &up_packed, n_in, n_out, N)
                    .unwrap();
                encode_silu_mul_f32(&ctx, &enc, &gate_packed, &up_packed, &inner_packed).unwrap();
            }
            enc.end();
            let t = Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            (wall, gpu)
        };

        for _ in 0..warmup {
            bench_fused();
            bench_unfused();
        }
        let mut sum_f_wall = 0.0f64;
        let mut sum_f_gpu = 0.0f64;
        let mut sum_u_wall = 0.0f64;
        let mut sum_u_gpu = 0.0f64;
        for _ in 0..iters {
            let (w, g) = bench_fused();
            sum_f_wall += w;
            sum_f_gpu += g;
        }
        for _ in 0..iters {
            let (w, g) = bench_unfused();
            sum_u_wall += w;
            sum_u_gpu += g;
        }
        let f_wall = sum_f_wall / iters as f64;
        let f_gpu = sum_f_gpu / iters as f64;
        let u_wall = sum_u_wall / iters as f64;
        let u_gpu = sum_u_gpu / iters as f64;

        eprintln!("[v0.73c.2-gate] {n_layers} layers × N={N} avg over {iters} iters:");
        eprintln!(
            "  fused (1 disp/layer):       wall={f_wall:7.2} ms  gpu={f_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            f_gpu / n_layers as f64
        );
        eprintln!(
            "  unfused (3 disp/layer):     wall={u_wall:7.2} ms  gpu={u_gpu:7.2} ms  per-layer-gpu={:5.3} ms",
            u_gpu / n_layers as f64
        );
        let ratio_wall = f_wall / u_wall;
        let ratio_gpu = f_gpu / u_gpu;
        let speedup_wall = 1.0 / ratio_wall;
        let speedup_gpu = 1.0 / ratio_gpu;
        eprintln!(
            "  ratio fused / unfused: wall={ratio_wall:.3} (= {speedup_wall:.2}× speedup)  gpu={ratio_gpu:.3} (= {speedup_gpu:.2}× speedup)"
        );

        // v0.73c.2 RESULT: failed go/no-go. Measured ratio ≈ 0.94 on
        // production 27B Q4_K_M FFN shape — fusion only saves ~6%,
        // codex threshold was ≤ 0.7 (≥ 30% speedup). Kernel is
        // preserved as experimental institutional memory; the assertion
        // below allows the bench to run as a re-checkable "regime
        // still capped?" probe without panicking. If a future change
        // (e.g. larger N, different shape, different hardware) puts
        // the ratio under 0.7, this is where to flag it for plumbing.
        if ratio_gpu <= 0.7 {
            eprintln!(
                "[v0.73c.2-gate] REGIME CHANGE: ratio_gpu {ratio_gpu:.3} now ≤ 0.7. \
                 Reconsider plumbing fused FFN into layer-major path."
            );
        } else {
            eprintln!(
                "[v0.73c.2-gate] still capped (ratio_gpu {ratio_gpu:.3} > 0.7); \
                 fusion not worth plumbing. Same negative result as v0.73c.2."
            );
        }
    }

    /// v0.73c.2 gate: layer-major fused SwiGLU FFN at N=16 must match
    /// the unfused (mat_mat_q4_K + mat_mat_q4_K + silu_mul) reference
    /// within mat-mat half-staging tolerance. Per-row cosine ≥ 0.999,
    /// max|Δ| ≤ 1e-2 (mirrors Q4_K mat-mat gate).
    ///
    /// Uses real `blk.0.ffn_gate.weight` + `ffn_up.weight` from 27B Q4_K_M.
    #[test]
    #[allow(non_snake_case)]
    fn ffn_fused_swiglu_q4_K_mm_n16_matches_unfused() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[ffn-fused-mm-n16] skipped — fixture missing");
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");

        let gate = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate.weight" && t.dtype == GgmlType::Q4_K)
            .expect("no blk.0.ffn_gate.weight Q4_K");
        let up = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up.weight" && t.dtype == GgmlType::Q4_K)
            .expect("no blk.0.ffn_up.weight Q4_K");
        assert_eq!(gate.shape, up.shape, "gate/up shape mismatch");
        let n_in = gate.shape[0] as usize;
        let n_out = gate.shape[1] as usize;
        const N: usize = 16;
        eprintln!("[ffn-fused-mm-n16] n_in={n_in} n_out={n_out} N={N}");

        let w_gate = MetalTensor::from_bytes(
            &ctx,
            g.slice(gate),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .unwrap();
        let w_up = MetalTensor::from_bytes(
            &ctx,
            g.slice(up),
            vec![n_in as u64, n_out as u64],
            GgmlType::Q4_K,
        )
        .unwrap();

        // Activation: row-major [N, n_in] deterministic fill.
        let mut x = vec![0.0f32; N * n_in];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 1e-2;
        }
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![N as u64, n_in as u64],
            GgmlType::F32,
        )
        .unwrap();

        // --- Unfused reference: gate_mm + up_mm + silu_mul ---
        let gate_pack = MetalTensor::zeros_f32(&ctx, vec![N as u64 * n_out as u64]).unwrap();
        let up_pack = MetalTensor::zeros_f32(&ctx, vec![N as u64 * n_out as u64]).unwrap();
        let inner_ref_t = MetalTensor::zeros_f32(&ctx, vec![N as u64 * n_out as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_mat_mat_q4_k_f32(&ctx, enc, &w_gate, &x_t, &gate_pack, n_in, n_out, N)?;
            encode_mat_mat_q4_k_f32(&ctx, enc, &w_up, &x_t, &up_pack, n_in, n_out, N)?;
            encode_silu_mul_f32(&ctx, enc, &gate_pack, &up_pack, &inner_ref_t)
        })
        .unwrap();
        let inner_ref_flat = read_back_f32(&inner_ref_t.buffer, N * n_out);

        // --- Fused: 1 dispatch ---
        let inner_fused_t = MetalTensor::zeros_f32(&ctx, vec![N as u64 * n_out as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_ffn_fused_swiglu_q4_K_mm_n16_f32(
                &ctx,
                enc,
                &w_gate,
                &w_up,
                &x_t,
                &inner_fused_t,
                n_in,
                n_out,
            )
        })
        .unwrap();
        let inner_fused_flat = read_back_f32(&inner_fused_t.buffer, N * n_out);

        // Both buffers are bit-equivalently row-major [N, n_out] (= col-major [n_out, N]).
        // Reshape via the same indexing as Q4_K mat-mat tests.
        let mut min_cos = f64::INFINITY;
        let mut max_abs = 0.0f32;
        for q in 0..N {
            let mut dot = 0.0f64;
            let mut np = 0.0f64;
            let mut nc = 0.0f64;
            for o in 0..n_out {
                // dst[o + q * n_out] is the cell (m=o, n=q) in col-major
                // [n_out, N], which equals row-major [N, n_out][q][o].
                let p = inner_fused_flat[o + q * n_out] as f64;
                let c = inner_ref_flat[o + q * n_out] as f64;
                dot += p * c;
                np += p * p;
                nc += c * c;
                let d = (p - c).abs() as f32;
                if d > max_abs {
                    max_abs = d;
                }
            }
            let cos = dot / (np.sqrt() * nc.sqrt() + 1e-30);
            if cos < min_cos {
                min_cos = cos;
            }
        }
        eprintln!("[ffn-fused-mm-n16] min_cos={min_cos:.6} max|Δ|={max_abs:.3e}");
        assert!(min_cos >= 0.999, "fused FFN N=16 cos too low: {min_cos}");
        assert!(max_abs < 1e-2, "fused FFN N=16 diverged: max|Δ|={max_abs}");

        const N32: usize = 32;
        let mut x32 = vec![0.0f32; N32 * n_in];
        for (i, v) in x32.iter_mut().enumerate() {
            *v = ((i % 17) as f32 - 8.0) * 1e-2;
        }
        let x32_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x32),
            vec![N32 as u64, n_in as u64],
            GgmlType::F32,
        )
        .unwrap();
        let gate32 = MetalTensor::zeros_f32(&ctx, vec![N32 as u64 * n_out as u64]).unwrap();
        let up32 = MetalTensor::zeros_f32(&ctx, vec![N32 as u64 * n_out as u64]).unwrap();
        let inner32_ref = MetalTensor::zeros_f32(&ctx, vec![N32 as u64 * n_out as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_mat_mat_q4_k_f32(&ctx, enc, &w_gate, &x32_t, &gate32, n_in, n_out, N32)?;
            encode_mat_mat_q4_k_f32(&ctx, enc, &w_up, &x32_t, &up32, n_in, n_out, N32)?;
            encode_silu_mul_f32(&ctx, enc, &gate32, &up32, &inner32_ref)
        })
        .unwrap();
        let inner32_fused = MetalTensor::zeros_f32(&ctx, vec![N32 as u64 * n_out as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_ffn_fused_swiglu_q4_K_mm_f32(
                &ctx,
                enc,
                &w_gate,
                &w_up,
                &x32_t,
                &inner32_fused,
                n_in,
                n_out,
                N32,
            )
        })
        .unwrap();
        let inner32_ref_flat = read_back_f32(&inner32_ref.buffer, N32 * n_out);
        let inner32_fused_flat = read_back_f32(&inner32_fused.buffer, N32 * n_out);
        let mut min_cos32 = f64::INFINITY;
        let mut max_abs32 = 0.0f32;
        for q in 0..N32 {
            let mut dot = 0.0f64;
            let mut np = 0.0f64;
            let mut nc = 0.0f64;
            for o in 0..n_out {
                let p = inner32_fused_flat[o + q * n_out] as f64;
                let c = inner32_ref_flat[o + q * n_out] as f64;
                dot += p * c;
                np += p * p;
                nc += c * c;
                max_abs32 = max_abs32.max((p - c).abs() as f32);
            }
            min_cos32 = min_cos32.min(dot / (np.sqrt() * nc.sqrt() + 1e-30));
        }
        eprintln!("[ffn-fused-mm-n32] min_cos={min_cos32:.6} max|Δ|={max_abs32:.3e}");
        assert!(
            min_cos32 >= 0.999,
            "fused FFN N=32 cos too low: {min_cos32}"
        );
        assert!(
            max_abs32 < 1e-2,
            "fused FFN N=32 diverged: max|Δ|={max_abs32}"
        );
    }

    #[test]
    fn mat_vec_q6_k_batch_matches_singleton_bits() {
        let ctx = match MetalContext::new() {
            Ok(context) => context,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("init failed: {error}"),
        };
        let n_in = 768usize;
        let n_out = 7usize;
        let n_tokens = 16usize;
        let blocks_per_row = n_in / 256;
        let row_bytes = blocks_per_row * 210;
        let mut weight = vec![0u8; n_out * row_bytes];
        for row in 0..n_out {
            for block_index in 0..blocks_per_row {
                let start = row * row_bytes + block_index * 210;
                let block = &mut weight[start..start + 210];
                for (index, value) in block[..192].iter_mut().enumerate() {
                    *value = (index as u8)
                        .wrapping_mul(17)
                        .wrapping_add(row as u8)
                        .wrapping_add(block_index as u8 * 11);
                }
                for (index, value) in block[192..208].iter_mut().enumerate() {
                    *value = (index as i8 - 8 + row as i8 + block_index as i8) as u8;
                }
                block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
            }
        }
        let input = (0..n_tokens * n_in)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
            .collect::<Vec<_>>();
        let weight = MetalTensor::from_bytes(
            &ctx,
            &weight,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q6_K,
        )
        .unwrap();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input),
            vec![n_tokens as u64, n_in as u64],
            GgmlType::F32,
        )
        .unwrap();
        let batched = MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_out) as u64]).unwrap();
        let singleton = MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_out) as u64]).unwrap();
        one_shot(&ctx, |encoder| {
            encode_mat_vec_q6_k_batch_f32(
                &ctx, encoder, &weight, &input, &batched, n_in, n_out, n_tokens,
            )
        })
        .unwrap();
        one_shot(&ctx, |encoder| {
            for token in 0..n_tokens {
                let input_row = input.view_subrange((token * n_in) as u64, vec![n_in as u64]);
                let output_row =
                    singleton.view_subrange((token * n_out) as u64, vec![n_out as u64]);
                encode_mat_vec_q6_k_f32(
                    &ctx,
                    encoder,
                    &weight,
                    &input_row,
                    &output_row,
                    n_in,
                    n_out,
                )?;
            }
            Ok(())
        })
        .unwrap();
        let batched = read_back_f32(&batched.buffer, n_tokens * n_out);
        let singleton = read_back_f32(&singleton.buffer, n_tokens * n_out);
        for (index, (batched, singleton)) in batched.iter().zip(&singleton).enumerate() {
            assert_eq!(
                batched.to_bits(),
                singleton.to_bits(),
                "Q6 batch mismatch at {index}: {batched} != {singleton}"
            );
        }
    }
}
