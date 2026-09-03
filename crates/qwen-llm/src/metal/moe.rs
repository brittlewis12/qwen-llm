//! MoE routing, swiglu, down, and grouped/packed slot kernels.

use super::*;

pub fn encode_mat_mat_f32_router_e8p32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_f32_router_e8p32_kernel(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        "mat_mat_f32_router_e8p32",
        "kernel_mat_mat_f32_f32_router_e8p32",
        true,
    )
}

pub fn encode_mat_mat_f32_router_e8p32_strict(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_f32_router_e8p32_kernel(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        "mat_mat_f32_router_e8p32_strict",
        "kernel_mat_mat_f32_f32_router_e8p32_strict",
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mat_mat_f32_router_e8p32_kernel(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    error_name: &'static str,
    kernel_name: &'static str,
    require_float4_input: bool,
) -> Result<(), MetalError> {
    if n_in == 0
        || n_out == 0
        || n_query == 0
        || u32::try_from(n_in).is_err()
        || u32::try_from(n_out).is_err()
        || u32::try_from(n_query).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: error_name,
            detail: format!(
                "dimensions must be nonzero u32 values, got n_in={n_in} n_out={n_out} n_query={n_query}"
            ),
        });
    }
    let expected_weight = n_in
        .checked_mul(n_out)
        .ok_or_else(|| MetalError::BadShape {
            kernel: error_name,
            detail: format!("weight element count overflows for n_in={n_in} n_out={n_out}"),
        })?;
    let expected_x = n_query
        .checked_mul(n_in)
        .ok_or_else(|| MetalError::BadShape {
            kernel: error_name,
            detail: format!("input element count overflows for n_query={n_query} n_in={n_in}"),
        })?;
    let expected_y = n_query
        .checked_mul(n_out)
        .ok_or_else(|| MetalError::BadShape {
            kernel: error_name,
            detail: format!("output element count overflows for n_query={n_query} n_out={n_out}"),
        })?;
    if weight.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: error_name,
            detail: format!("weight.dtype = {:?}, expected F32", weight.dtype),
        });
    }
    if x.dtype != GgmlType::F32 || y.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: error_name,
            detail: format!("x/y expected F32, got {:?}/{:?}", x.dtype, y.dtype),
        });
    }
    if (require_float4_input && !n_in.is_multiple_of(4)) || !n_out.is_multiple_of(8) {
        return Err(MetalError::BadShape {
            kernel: error_name,
            detail: format!(
                "expected compatible n_in and n_out % 8 == 0, got n_in={n_in} n_out={n_out}"
            ),
        });
    }
    if weight.n_elements() as usize != expected_weight {
        return Err(MetalError::BadShape {
            kernel: error_name,
            detail: format!(
                "weight.n_elements={} != n_in*n_out={expected_weight}",
                weight.n_elements()
            ),
        });
    }
    if x.n_elements() as usize != expected_x {
        return Err(MetalError::BadShape {
            kernel: error_name,
            detail: format!(
                "x.n_elements={} != n_query*n_in={expected_x}",
                x.n_elements(),
            ),
        });
    }
    if y.n_elements() as usize != expected_y {
        return Err(MetalError::BadShape {
            kernel: error_name,
            detail: format!(
                "y.n_elements={} != n_out*n_query={expected_y}",
                y.n_elements(),
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
    enc.dispatch(
        MTLSize {
            width: n_out / 8,
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

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K",
            detail: format!(
                "expected Q4_K expert gate/up, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K",
            detail: format!("x.n_elements={} != n_in={n_in}", x.n_elements()),
        });
    }
    if topk_idx.n_elements() as usize != topk || inner.n_elements() as usize != topk * n_out {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K",
            detail: format!(
                "topk/inner mismatch: idx={} inner={} expected idx={topk} inner={}",
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q4_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_packed_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    topk_idx_pack: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_packed_slots",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_packed_slots",
            detail: format!(
                "expected Q4_K expert gate/up, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    let n_slots = n_tokens * topk;
    if x_pack.n_elements() as usize != n_tokens * n_in {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_packed_slots",
            detail: format!(
                "x_pack.n_elements={} != n_tokens*n_in={}",
                x_pack.n_elements(),
                n_tokens * n_in
            ),
        });
    }
    if topk_idx_pack.n_elements() as usize != n_slots
        || inner.n_elements() as usize != n_slots * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_packed_slots",
            detail: format!(
                "slot/inner mismatch: idx={} inner={} expected idx={n_slots} inner={}",
                topk_idx_pack.n_elements(),
                inner.n_elements(),
                n_slots * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q4_K_f32_packed_slots")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, topk_idx_pack);
    enc.set_tensor(5, inner);

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: n_slots,
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

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_hidden.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_n16",
            detail: format!(
                "expected Q4_K gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q4_K_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 144) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if min_count > max_count || max_count > i32::MAX as u32 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_grouped_slots_n16",
            detail: format!(
                "count range [{min_count}, {max_count}] must be ordered and fit signed kernel arguments"
            ),
        });
    }
    if !n_hidden.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::IQ3_XXS || w_up.dtype != GgmlType::IQ3_XXS {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_grouped_slots_n16",
            detail: format!(
                "expected IQ3_XXS gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 98) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_iq4_xs_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "moe_swiglu_iq4_xs_grouped_slots_n16";
    let dimensions = checked_moe_decode_args(KERNEL, n_hidden, n_ffn, n_expert, topk)?;
    if !n_hidden.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if topk > 16 {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("topk={topk} exceeds grouped kernel limit 16"),
        });
    }
    let n_tokens_u32 = u32::try_from(n_tokens).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("n_tokens={n_tokens} exceeds u32"),
    })?;
    if n_tokens == 0 {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "n_tokens must be nonzero".into(),
        });
    }
    for (name, value) in [
        ("n_ffn", n_ffn),
        ("n_expert", n_expert),
        ("n_tokens", n_tokens),
    ] {
        if i32::try_from(value).is_err() {
            return Err(MetalError::BadShape {
                kernel: KERNEL,
                detail: format!("{name}={value} exceeds signed shader indexing"),
            });
        }
    }

    let slot_count = checked_moe_product(KERNEL, "route slots", &[n_tokens, topk])?;
    if i32::try_from(slot_count).is_err() {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("route slot count {slot_count} exceeds signed shader indexing"),
        });
    }
    let bank_elements = checked_moe_product(KERNEL, "expert bank", &[n_hidden, n_ffn, n_expert])?;
    let input_elements = checked_moe_product(KERNEL, "packed input", &[n_tokens, n_hidden])?;
    let bucket_elements = checked_moe_product(KERNEL, "expert buckets", &[n_expert, n_tokens])?;
    let inner_elements = checked_moe_product(KERNEL, "inner output", &[slot_count, n_ffn])?;
    validate_moe_decode_tensor(
        KERNEL,
        "gate expert bank",
        w_gate,
        bank_elements,
        &[GgmlType::IQ4_XS],
        false,
        2,
    )?;
    validate_moe_decode_tensor(
        KERNEL,
        "up expert bank",
        w_up,
        bank_elements,
        &[GgmlType::IQ4_XS],
        false,
        2,
    )?;
    validate_moe_decode_tensor(
        KERNEL,
        "packed input",
        x_pack,
        input_elements,
        &[GgmlType::F32],
        false,
        16,
    )?;
    validate_moe_decode_tensor(
        KERNEL,
        "expert counts",
        counts,
        n_expert,
        &[GgmlType::I32],
        false,
        4,
    )?;
    validate_moe_decode_tensor(
        KERNEL,
        "expert slot buckets",
        ids,
        bucket_elements,
        &[GgmlType::I32],
        false,
        4,
    )?;
    validate_moe_decode_tensor(
        KERNEL,
        "inner output",
        inner,
        inner_elements,
        &[GgmlType::F32],
        true,
        4,
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let pso = ctx.pipeline("kernel_moe_swiglu_iq4_xs_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            ffn: dimensions.n_out,
            hidden: dimensions.n_in,
            n_expert: dimensions.n_expert,
            topk: dimensions.topk,
            n_tokens: n_tokens_u32,
            nb01: dimensions.n_in / 256,
            stride_b: dimensions.n_in,
            min_count: 0,
            max_count: n_tokens_u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16_384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_iq3_s_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_iq3_s_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_iq3_s_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_hidden.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::IQ3_S || w_up.dtype != GgmlType::IQ3_S {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s_grouped_slots_n16",
            detail: format!(
                "expected IQ3_S gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_iq3_s_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 110) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q5_K_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_q5_K_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q5_K_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_hidden.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q5_K_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q5_K || w_up.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q5_K_grouped_slots_n16",
            detail: format!(
                "expected Q5_K gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q5_K_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q5_K_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 176) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q6_K_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_q6_K_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q6_K_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_hidden.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q6_K_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q6_K || w_up.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q6_K_grouped_slots_n16",
            detail: format!(
                "expected Q6_K gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q6_K_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q6_K_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 210) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_swiglu_q8_0_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_q8_0_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

pub fn encode_moe_swiglu_q8_0_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_hidden.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q8_0_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 32"),
        });
    }
    if w_gate.dtype != GgmlType::Q8_0 || w_up.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q8_0_grouped_slots_n16",
            detail: format!(
                "expected Q8_0 gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q8_0_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q8_0_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 32) * 34) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_swiglu_f32_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if w_gate.dtype != GgmlType::F32 || w_up.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_f32_grouped_slots_n16",
            detail: format!(
                "expected F32 gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_f32_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_f32_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01: n_hidden as u32,
            stride_b: n_hidden as u32,
            min_count: 0,
            max_count: i32::MAX as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_swiglu_bf16_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_bf16_f32_grouped_slots_n16_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

pub fn encode_moe_swiglu_bf16_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if w_gate.dtype != GgmlType::BF16 || w_up.dtype != GgmlType::BF16 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_bf16_grouped_slots_n16",
            detail: format!(
                "expected BF16 gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_bf16_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_bf16_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01: n_hidden as u32,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_fused: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_hidden.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if !n_ffn.is_multiple_of(64) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n16",
            detail: format!("n_ffn={n_ffn} not divisible by 64"),
        });
    }
    if topk == 0 || n_expert == 0 || n_tokens == 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n16",
            detail: "topk/n_expert/n_tokens must all be > 0".into(),
        });
    }
    if w_fused.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n16",
            detail: format!("expected fused Q4_K expert bank, got {:?}", w_fused.dtype),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q4_K_f32_grouped_slots_fused_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 144) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_fused);
    enc.set_tensor(2, x_pack);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, inner);
    enc.set_threadgroup_memory(0, 12288);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_grouped_slots_n32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
        ctx,
        enc,
        w_gate,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_hidden.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_n32",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K || w_up.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_n32",
            detail: format!(
                "expected Q4_K gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_n32",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q4_K_f32_grouped_slots_n32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 144) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x_pack);
    enc.set_tensor(4, counts);
    enc.set_tensor(5, ids);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n32_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_fused: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_hidden.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n32",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if !n_ffn.is_multiple_of(64) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n32",
            detail: format!("n_ffn={n_ffn} not divisible by 64"),
        });
    }
    if topk == 0 || n_expert == 0 || n_tokens == 0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n32",
            detail: "topk/n_expert/n_tokens must all be > 0".into(),
        });
    }
    if w_fused.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n32",
            detail: format!("expected fused Q4_K expert bank, got {:?}", w_fused.dtype),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q4_K_grouped_slots_fused_n32",
            detail: format!(
                "shape mismatch x={} counts={} ids={} inner={} expected x={} counts={} ids={} inner={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                inner.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q4_K_f32_grouped_slots_fused_n32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 144) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, w_fused);
    enc.set_tensor(2, x_pack);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, inner);
    enc.set_threadgroup_memory(0, 16384);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_matmul_q4_K_f32_grouped_slots_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_matmul_q4_K_f32_grouped_slots_n16_range(
        ctx,
        enc,
        weight,
        x_pack,
        counts,
        ids,
        out,
        n_hidden,
        n_ffn,
        n_expert,
        topk,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_matmul_q4_K_f32_grouped_slots_n16_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_hidden.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_matmul_q4_K_grouped_slots_n16",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_matmul_q4_K_grouped_slots_n16",
            detail: format!("expected Q4_K expert bank, got {:?}", weight.dtype),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_matmul_q4_K_grouped_slots_n16",
            detail: format!(
                "shape mismatch x={} counts={} ids={} out={} expected x={} counts={} ids={} out={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_matmul_q4_K_f32_grouped_slots_n16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 144) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x_pack);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(16),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_matmul_q4_K_f32_grouped_slots_n32_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x_pack: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_hidden.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_matmul_q4_K_grouped_slots_n32",
            detail: format!("n_hidden={n_hidden} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_matmul_q4_K_grouped_slots_n32",
            detail: format!("expected Q4_K expert bank, got {:?}", weight.dtype),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: "moe_matmul_q4_K_grouped_slots_n32",
            detail: format!(
                "shape mismatch x={} counts={} ids={} out={} expected x={} counts={} ids={} out={}",
                x_pack.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                n_tokens * n_hidden,
                n_expert,
                n_expert * n_tokens,
                n_tokens * topk * n_ffn
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_matmul_q4_K_f32_grouped_slots_n32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ffn: u32,
        hidden: u32,
        n_expert: u32,
        topk: u32,
        n_tokens: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_hidden / 256) * 144) as u32;
    enc.set_bytes(
        0,
        &Args {
            ffn: n_ffn as u32,
            hidden: n_hidden as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
            n_tokens: n_tokens as u32,
            nb01,
            stride_b: n_hidden as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x_pack);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_ffn.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_fused_routed_q4q5_token_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    w_down: &MetalTensor,
    x_pack: &MetalTensor,
    topk_idx_pack: &MetalTensor,
    topk_w_pack: &MetalTensor,
    out: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if !n_hidden.is_multiple_of(256) || !n_ffn.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_fused_routed_q4q5_token",
            detail: format!("n_hidden={n_hidden} and n_ffn={n_ffn} must both be divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q4_K
        || w_up.dtype != GgmlType::Q4_K
        || w_down.dtype != GgmlType::Q5_K
    {
        return Err(MetalError::BadShape {
            kernel: "moe_fused_routed_q4q5_token",
            detail: format!(
                "expected gate/up/down dtypes Q4_K/Q4_K/Q5_K, got {:?}/{:?}/{:?}",
                w_gate.dtype, w_up.dtype, w_down.dtype
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || topk_idx_pack.n_elements() as usize != n_tokens * topk
        || topk_w_pack.n_elements() as usize != n_tokens * topk
        || out.n_elements() as usize != n_tokens * n_hidden
    {
        return Err(MetalError::BadShape {
            kernel: "moe_fused_routed_q4q5_token",
            detail: format!(
                "shape mismatch: x={} idx={} w={} out={} expected x={} idx={} w={} out={}",
                x_pack.n_elements(),
                topk_idx_pack.n_elements(),
                topk_w_pack.n_elements(),
                out.n_elements(),
                n_tokens * n_hidden,
                n_tokens * topk,
                n_tokens * topk,
                n_tokens * n_hidden
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_fused_routed_q4q5_token_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        hidden: u32,
        ffn: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            hidden: n_hidden as u32,
            ffn: n_ffn as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, w_down);
    enc.set_tensor(4, x_pack);
    enc.set_tensor(5, topk_idx_pack);
    enc.set_tensor(6, topk_w_pack);
    enc.set_tensor(7, out);
    enc.set_threadgroup_memory(0, n_ffn * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: n_tokens,
            depth: 1,
        },
        MTLSize {
            width: 512,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_down_q4_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q4_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q4_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q4_K",
            detail: format!("expected Q4_K expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q4_K",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q4_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_q5_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K",
            detail: format!("expected Q5_K expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q5_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);

    const NR0: usize = 1;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_q5_K_f32_grouped_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    expert_idx: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_rows: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_rows",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_rows",
            detail: format!("expected Q5_K expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != n_rows * n_in
        || expert_idx.n_elements() as usize != n_rows
        || out.n_elements() as usize != n_rows * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_rows",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={} out={}",
                inner.n_elements(),
                expert_idx.n_elements(),
                out.n_elements(),
                n_rows * n_in,
                n_rows,
                n_rows * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q5_K_f32_grouped_rows")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: 0,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, expert_idx);
    enc.set_tensor(4, out);

    const NR0: usize = 1;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: n_rows,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_q5_K_f32_grouped_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_down_q5_K_f32_grouped_slots_range(
        ctx,
        enc,
        weight,
        inner,
        counts,
        ids,
        out,
        n_in,
        n_out,
        n_expert,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

#[allow(non_snake_case)]
pub fn encode_moe_down_q5_K_f32_grouped_slots_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_slots",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_slots",
            detail: format!("expected Q5_K expert down, got {:?}", weight.dtype),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if inner.n_elements() as usize != slot_count * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != slot_count * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_slots",
            detail: format!(
                "shape mismatch: inner={} counts={} ids={} out={} expected inner={} counts={} ids={} out={}",
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                slot_count * n_in,
                n_expert,
                n_expert * n_tokens,
                slot_count * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q5_K_f32_grouped_slots")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_in / 256) * 176) as u32;
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            nb01,
            stride_b: n_in as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_down_q5_K_f32_grouped_slots_tiny8_r16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_slots_tiny8_r16",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_slots_tiny8_r16",
            detail: format!("expected Q5_K expert down, got {:?}", weight.dtype),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if inner.n_elements() as usize != slot_count * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != slot_count * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q5_K_grouped_slots_tiny8_r16",
            detail: format!(
                "shape mismatch: inner={} counts={} ids={} out={} expected inner={} counts={} ids={} out={}",
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                slot_count * n_in,
                n_expert,
                n_expert * n_tokens,
                slot_count * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q5_K_f32_grouped_slots_tiny8_r16")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        n_expert: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    let nb01 = ((n_in / 256) * 176) as u32;
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            n_expert: n_expert as u32,
            nb01,
            stride_b: n_in as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(16),
            height: n_expert.div_ceil(4),
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

#[allow(non_snake_case)]
pub fn encode_moe_down_q6_K_f32_grouped_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q6_K_grouped_slots",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q6_K_grouped_slots",
            detail: format!("expected Q6_K expert down, got {:?}", weight.dtype),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if inner.n_elements() as usize != slot_count * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != slot_count * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q6_K_grouped_slots",
            detail: format!(
                "shape mismatch: inner={} counts={} ids={} out={} expected inner={} counts={} ids={} out={}",
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                slot_count * n_in,
                n_expert,
                n_expert * n_tokens,
                slot_count * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q6_K_f32_grouped_slots")?;
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
    let nb01 = ((n_in / 256) * 210) as u32;
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            nb01,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_down_q8_0_f32_grouped_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q8_0_grouped_slots",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q8_0_grouped_slots",
            detail: format!("expected Q8_0 expert down, got {:?}", weight.dtype),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if inner.n_elements() as usize != slot_count * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != slot_count * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_q8_0_grouped_slots",
            detail: format!(
                "shape mismatch: inner={} counts={} ids={} out={} expected inner={} counts={} ids={} out={}",
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                slot_count * n_in,
                n_expert,
                n_expert * n_tokens,
                slot_count * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_q8_0_f32_grouped_slots")?;
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
    let nb01 = ((n_in / 32) * 34) as u32;
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            nb01,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct MoeDownIq4NlGroupedSlotsKernel {
    pub(crate) validation_name: &'static str,
    pub(crate) pipeline_name: &'static str,
    pub(crate) tile_m: usize,
    pub(crate) tile_n: usize,
    pub(crate) threadgroup_memory: usize,
}

pub(crate) const MOE_DOWN_IQ4_NL_GROUPED_SLOTS_M64_N32: MoeDownIq4NlGroupedSlotsKernel =
    MoeDownIq4NlGroupedSlotsKernel {
        validation_name: "moe_down_iq4_nl_grouped_slots",
        pipeline_name: "kernel_moe_down_iq4_nl_f32_grouped_slots",
        tile_m: 64,
        tile_n: 32,
        threadgroup_memory: 8_192,
    };

pub(crate) const MOE_DOWN_IQ4_NL_GROUPED_SLOTS_M128_N16: MoeDownIq4NlGroupedSlotsKernel =
    MoeDownIq4NlGroupedSlotsKernel {
        validation_name: "moe_down_iq4_nl_grouped_slots_m128_n16",
        pipeline_name: "kernel_moe_down_iq4_nl_f32_grouped_slots_m128_n16",
        tile_m: 128,
        tile_n: 16,
        threadgroup_memory: 9_216,
    };

pub fn encode_moe_down_iq4_nl_f32_grouped_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_down_iq4_nl_f32_grouped_slots_kernel(
        ctx,
        enc,
        weight,
        inner,
        counts,
        ids,
        out,
        n_in,
        n_out,
        n_expert,
        n_tokens,
        MOE_DOWN_IQ4_NL_GROUPED_SLOTS_M64_N32,
    )
}

pub fn encode_moe_down_iq4_nl_f32_grouped_slots_m128_n16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_down_iq4_nl_f32_grouped_slots_kernel(
        ctx,
        enc,
        weight,
        inner,
        counts,
        ids,
        out,
        n_in,
        n_out,
        n_expert,
        n_tokens,
        MOE_DOWN_IQ4_NL_GROUPED_SLOTS_M128_N16,
    )
}

pub(crate) fn encode_moe_down_iq4_nl_f32_grouped_slots_kernel(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
    config: MoeDownIq4NlGroupedSlotsKernel,
) -> Result<(), MetalError> {
    let kernel = config.validation_name;
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::IQ4_NL
        || inner.dtype != GgmlType::F32
        || out.dtype != GgmlType::F32
        || !matches!(counts.dtype, GgmlType::I32 | GgmlType::F32)
        || !matches!(ids.dtype, GgmlType::I32 | GgmlType::F32)
    {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "expected IQ4_NL/F32/I32-or-F32 metadata/F32, got {:?}/{:?}/{:?}/{:?}/{:?}",
                weight.dtype, inner.dtype, counts.dtype, ids.dtype, out.dtype
            ),
        });
    }
    if n_in == 0 || n_out == 0 || n_expert == 0 || n_tokens == 0 {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "n_in={n_in}, n_out={n_out}, n_expert={n_expert}, and n_tokens={n_tokens} must be nonzero"
            ),
        });
    }
    for (name, value) in [
        ("n_in", n_in),
        ("n_out", n_out),
        ("n_expert", n_expert),
        ("n_tokens", n_tokens),
    ] {
        if u32::try_from(value).is_err() {
            return Err(MetalError::BadShape {
                kernel,
                detail: format!("{name}={value} exceeds u32"),
            });
        }
    }
    for (name, value) in [("n_out", n_out), ("n_expert", n_expert)] {
        if i32::try_from(value).is_err() {
            return Err(MetalError::BadShape {
                kernel,
                detail: format!("{name}={value} exceeds signed shader indexing"),
            });
        }
    }
    let product = |name: &str, factors: &[usize]| {
        factors.iter().try_fold(1_usize, |value, &factor| {
            value
                .checked_mul(factor)
                .ok_or_else(|| MetalError::BadShape {
                    kernel,
                    detail: format!("{name} element count overflow"),
                })
        })
    };
    let bank_elements = product("expert bank", &[n_in, n_out, n_expert])?;
    let bucket_elements = product("expert buckets", &[n_expert, n_tokens])?;
    if !(out.n_elements() as usize).is_multiple_of(n_out) {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "output elements {} are not divisible by n_out={n_out}",
                out.n_elements()
            ),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if !slot_count.is_multiple_of(n_tokens) {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("slot count {slot_count} is not divisible by n_tokens={n_tokens}"),
        });
    }
    let topk = slot_count / n_tokens;
    if topk == 0 || topk > 16 || topk > n_expert || i32::try_from(slot_count).is_err() {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "expected 1 <= slots/tokens={topk} <= min(16, n_expert={n_expert}) and slot count {slot_count} to fit i32"
            ),
        });
    }
    let inner_elements = product("inner input", &[slot_count, n_in])?;
    let output_elements = product("expert output", &[slot_count, n_out])?;
    if weight.n_elements() as usize != bank_elements
        || inner.n_elements() as usize != inner_elements
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != bucket_elements
        || out.n_elements() as usize != output_elements
    {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "shape mismatch weight={} inner={} counts={} ids={} out={} expected {bank_elements}/{inner_elements}/{n_expert}/{bucket_elements}/{output_elements}",
                weight.n_elements(),
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements()
            ),
        });
    }

    let blocks_per_row = n_in / 32;
    let pso = ctx.pipeline(config.pipeline_name)?;
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
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            nb01: blocks_per_row as u32,
            stride_b: n_in as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, config.threadgroup_memory);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(config.tile_n),
            height: n_out.div_ceil(config.tile_m),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_down_bf16_f32_grouped_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    encode_moe_down_bf16_f32_grouped_slots_range(
        ctx,
        enc,
        weight,
        inner,
        counts,
        ids,
        out,
        n_in,
        n_out,
        n_expert,
        n_tokens,
        0,
        i32::MAX as u32,
    )
}

pub fn encode_moe_down_bf16_f32_grouped_slots_range(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_bf16_grouped_slots",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::BF16 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_bf16_grouped_slots",
            detail: format!("expected BF16 expert down, got {:?}", weight.dtype),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if inner.n_elements() as usize != slot_count * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != slot_count * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_bf16_grouped_slots",
            detail: format!(
                "shape mismatch: inner={} counts={} ids={} out={} expected inner={} counts={} ids={} out={}",
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                slot_count * n_in,
                n_expert,
                n_expert * n_tokens,
                slot_count * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_bf16_f32_grouped_slots")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            nb01: n_in as u32,
            stride_b: n_in as u32,
            min_count,
            max_count,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_down_iq4_xs_f32_grouped_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs_grouped_slots",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::IQ4_XS {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs_grouped_slots",
            detail: format!("expected IQ4_XS expert down, got {:?}", weight.dtype),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    if inner.n_elements() as usize != slot_count * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || out.n_elements() as usize != slot_count * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs_grouped_slots",
            detail: format!(
                "shape mismatch: inner={} counts={} ids={} out={} expected inner={} counts={} ids={} out={}",
                inner.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                slot_count * n_in,
                n_expert,
                n_expert * n_tokens,
                slot_count * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_iq4_xs_f32_grouped_slots")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        min_count: u32,
        max_count: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            nb01: (n_in / 256) as u32,
            stride_b: n_in as u32,
            min_count: 0,
            max_count: i32::MAX as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_out.div_ceil(64),
            depth: n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(non_snake_case)]
pub fn encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    topk_w: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q5_K_packed_slots",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q5_K_packed_slots",
            detail: format!("expected Q5_K expert down, got {:?}", weight.dtype),
        });
    }
    let n_slots = n_tokens * topk;
    if inner.n_elements() as usize != n_slots * n_in
        || topk_idx.n_elements() as usize != n_slots
        || topk_w.n_elements() as usize != n_slots
        || out.n_elements() as usize != n_tokens * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q5_K_packed_slots",
            detail: format!(
                "shape mismatch: inner={} idx={} w={} out={} expected inner={} idx={} w={} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                topk_w.n_elements(),
                out.n_elements(),
                n_slots * n_in,
                n_slots,
                n_slots,
                n_tokens * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_weighted_sum_q5_K_f32_packed_slots")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, topk_w);
    enc.set_tensor(5, out);

    const NR0: usize = 1;
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

#[allow(non_snake_case)]
pub fn encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    topk_w: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if n_in != 512 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q5_K_k512_r2",
            detail: format!("expected n_in=512, got {n_in}"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q5_K_k512_r2",
            detail: format!("expected Q5_K expert down, got {:?}", weight.dtype),
        });
    }
    let n_slots = n_tokens * topk;
    if inner.n_elements() as usize != n_slots * n_in
        || topk_idx.n_elements() as usize != n_slots
        || topk_w.n_elements() as usize != n_slots
        || out.n_elements() as usize != n_tokens * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q5_K_k512_r2",
            detail: format!(
                "shape mismatch: inner={} idx={} w={} out={} expected inner={} idx={} w={} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                topk_w.n_elements(),
                out.n_elements(),
                n_slots * n_in,
                n_slots,
                n_slots,
                n_tokens * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, topk_w);
    enc.set_tensor(5, out);

    const ROWS_PER_TG: usize = 4;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(ROWS_PER_TG),
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

#[allow(non_snake_case)]
pub fn encode_moe_mat_vec_q5_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_q5_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q5_K {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_q5_K",
            detail: format!("expected Q5_K expert weight, got {:?}", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_q5_K",
            detail: format!(
                "shape mismatch: x={} idx={} out={} expected x={n_in} idx={topk} out={}",
                x.n_elements(),
                topk_idx.n_elements(),
                out.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_mat_vec_q5_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, out);

    const NR0: usize = 1;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

pub fn encode_moe_mat_vec_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_iq3_xxs",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::IQ3_XXS {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_iq3_xxs",
            detail: format!("expected IQ3_XXS expert weight, got {:?}", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_iq3_xxs",
            detail: format!(
                "shape mismatch: x={} idx={} out={} expected x={n_in} idx={topk} out={}",
                x.n_elements(),
                topk_idx.n_elements(),
                out.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_mat_vec_iq3_xxs_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, out);

    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_mat_vec_iq3_s_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_iq3_s",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::IQ3_S {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_iq3_s",
            detail: format!("expected IQ3_S expert weight, got {:?}", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_iq3_s",
            detail: format!(
                "shape mismatch: x={} idx={} out={} expected x={n_in} idx={topk} out={}",
                x.n_elements(),
                topk_idx.n_elements(),
                out.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_mat_vec_iq3_s_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, out);

    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_swiglu_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::IQ3_XXS || w_up.dtype != GgmlType::IQ3_XXS {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs",
            detail: format!(
                "expected IQ3_XXS gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || inner.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs",
            detail: format!(
                "shape mismatch: x={} idx={} inner={} expected x={n_in} idx={topk} inner={}",
                x.n_elements(),
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_iq3_xxs_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_swiglu_iq3_xxs_f32_fast(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_fast",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::IQ3_XXS || w_up.dtype != GgmlType::IQ3_XXS {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_fast",
            detail: format!(
                "expected IQ3_XXS gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || inner.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_xxs_fast",
            detail: format!(
                "shape mismatch: x={} idx={} inner={} expected x={n_in} idx={topk} inner={}",
                x.n_elements(),
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_iq3_xxs_f32_fast")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NR0: usize = 4;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

pub fn encode_moe_swiglu_iq3_s_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::IQ3_S || w_up.dtype != GgmlType::IQ3_S {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s",
            detail: format!(
                "expected IQ3_S gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || inner.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s",
            detail: format!(
                "shape mismatch: x={} idx={} inner={} expected x={n_in} idx={topk} inner={}",
                x.n_elements(),
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_iq3_s_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_swiglu_iq3_s_f32_fast(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s_fast",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::IQ3_S || w_up.dtype != GgmlType::IQ3_S {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s_fast",
            detail: format!(
                "expected IQ3_S gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || inner.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_iq3_s_fast",
            detail: format!(
                "shape mismatch: x={} idx={} inner={} expected x={n_in} idx={topk} inner={}",
                x.n_elements(),
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_iq3_s_f32_fast")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NR0: usize = 4;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MoeDecodeArgs {
    pub(crate) n_in: u32,
    pub(crate) n_out: u32,
    pub(crate) n_expert: u32,
    pub(crate) topk: u32,
}

pub(crate) fn checked_moe_decode_args(
    kernel: &'static str,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<MoeDecodeArgs, MetalError> {
    if n_in == 0 || n_out == 0 || n_expert == 0 || topk == 0 || topk > n_expert {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "dimensions must satisfy n_in,n_out,n_expert > 0 and 0 < topk <= n_expert, got n_in={n_in} n_out={n_out} n_expert={n_expert} topk={topk}"
            ),
        });
    }
    let narrow = |name: &str, value: usize| {
        u32::try_from(value).map_err(|_| MetalError::BadShape {
            kernel,
            detail: format!("{name}={value} exceeds u32"),
        })
    };
    Ok(MoeDecodeArgs {
        n_in: narrow("n_in", n_in)?,
        n_out: narrow("n_out", n_out)?,
        n_expert: narrow("n_expert", n_expert)?,
        topk: narrow("topk", topk)?,
    })
}

pub(crate) fn checked_moe_product(
    kernel: &'static str,
    label: &str,
    factors: &[usize],
) -> Result<usize, MetalError> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| MetalError::BadShape {
                kernel,
                detail: format!("{label} element count overflow"),
            })
    })
}

pub(crate) fn validate_moe_decode_tensor(
    kernel: &'static str,
    name: &str,
    tensor: &MetalTensor,
    expected_elements: usize,
    allowed_dtypes: &[GgmlType],
    writable: bool,
    alignment: usize,
) -> Result<(), MetalError> {
    if !allowed_dtypes.contains(&tensor.dtype) {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "{name} has dtype {:?}, expected {allowed_dtypes:?}",
                tensor.dtype
            ),
        });
    }
    let (elements, bytes) = checked_ggml_shape_bytes(&tensor.shape, tensor.dtype)?;
    if elements != expected_elements {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("{name} has {elements} elements, expected {expected_elements}"),
        });
    }
    if writable && !tensor.is_writable() {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "{name} must be writable, provenance={:?}",
                tensor.provenance()
            ),
        });
    }
    if alignment == 0 || !tensor.offset.is_multiple_of(alignment as u64) {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "{name} offset {} is not {alignment}-byte aligned",
                tensor.offset
            ),
        });
    }
    let end = tensor
        .offset
        .checked_add(bytes as u64)
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: format!("{name} buffer range overflow"),
        })?;
    if end > tensor.buffer.length() as u64 {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "{name} range offset={} bytes={bytes} exceeds buffer={}",
                tensor.offset,
                tensor.buffer.length()
            ),
        });
    }
    Ok(())
}

pub fn encode_moe_swiglu_iq4_xs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "moe_swiglu_iq4_xs";
    let args = checked_moe_decode_args(KERNEL, n_in, n_out, n_expert, topk)?;
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    let bank_elements = checked_moe_product(KERNEL, "expert bank", &[n_in, n_out, n_expert])?;
    let inner_elements = checked_moe_product(KERNEL, "inner output", &[topk, n_out])?;
    validate_moe_decode_tensor(
        KERNEL,
        "gate expert bank",
        w_gate,
        bank_elements,
        &[GgmlType::IQ4_XS],
        false,
        2,
    )?;
    validate_moe_decode_tensor(
        KERNEL,
        "up expert bank",
        w_up,
        bank_elements,
        &[GgmlType::IQ4_XS],
        false,
        2,
    )?;
    validate_moe_decode_tensor(KERNEL, "input", x, n_in, &[GgmlType::F32], false, 4)?;
    validate_moe_decode_tensor(
        KERNEL,
        "top-k indices",
        topk_idx,
        topk,
        &[GgmlType::I32, GgmlType::F32],
        false,
        4,
    )?;
    validate_moe_decode_tensor(
        KERNEL,
        "inner output",
        inner,
        inner_elements,
        &[GgmlType::F32],
        true,
        4,
    )?;

    let pso = ctx.pipeline("kernel_moe_swiglu_iq4_xs_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NSG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_swiglu_q6_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q6_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if w_gate.dtype != GgmlType::Q6_K || w_up.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q6_K",
            detail: format!(
                "expected Q6_K gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || inner.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q6_K",
            detail: format!(
                "shape mismatch: x={} idx={} inner={} expected x={n_in} idx={topk} inner={}",
                x.n_elements(),
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q6_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

pub fn encode_moe_swiglu_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w_gate: &MetalTensor,
    w_up: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q8_0",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if w_gate.dtype != GgmlType::Q8_0 || w_up.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q8_0",
            detail: format!(
                "expected Q8_0 gate/up expert banks, got {:?}/{:?}",
                w_gate.dtype, w_up.dtype
            ),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || inner.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_swiglu_q8_0",
            detail: format!(
                "shape mismatch: x={} idx={} inner={} expected x={n_in} idx={topk} inner={}",
                x.n_elements(),
                topk_idx.n_elements(),
                inner.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_swiglu_q8_0_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, w_gate);
    enc.set_tensor(2, w_up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, topk_idx);
    enc.set_tensor(5, inner);

    let nr0 = 2usize;
    let nsg = 4usize;
    enc.set_threadgroup_memory(0, 32 * 2 * nr0 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(nr0),
            height: topk,
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

pub fn encode_moe_mat_vec_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_f32",
            detail: format!("expected F32 expert weight, got {:?}", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_f32",
            detail: format!(
                "shape mismatch: x={} idx={} out={} expected x={n_in} idx={topk} out={}",
                x.n_elements(),
                topk_idx.n_elements(),
                out.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_mat_vec_f32_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, out);

    const NSG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_down_f32_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::F32 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_f32",
            detail: format!("expected F32 expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_f32",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_f32_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);

    const NSG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_mat_vec_bf16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    topk_idx: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::BF16 {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_bf16",
            detail: format!("expected BF16 expert weight, got {:?}", weight.dtype),
        });
    }
    if x.n_elements() as usize != n_in
        || topk_idx.n_elements() as usize != topk
        || out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_mat_vec_bf16",
            detail: format!(
                "shape mismatch: x={} idx={} out={} expected x={n_in} idx={topk} out={}",
                x.n_elements(),
                topk_idx.n_elements(),
                out.n_elements(),
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_mat_vec_bf16_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, out);

    const NSG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_down_bf16_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::BF16 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_bf16",
            detail: format!("expected BF16 expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_bf16",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_bf16_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);

    const NSG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

pub fn encode_moe_down_iq4_nl_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    encode_moe_down_iq4_nl_f32_inner(
        ctx, enc, weight, inner, topk_idx, expert_out, n_in, n_out, n_expert, topk, false,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_moe_down_iq4_nl_f32_fast(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    encode_moe_down_iq4_nl_f32_inner(
        ctx, enc, weight, inner, topk_idx, expert_out, n_in, n_out, n_expert, topk, true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe_down_iq4_nl_f32_inner(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
    fast: bool,
) -> Result<(), MetalError> {
    let kernel = if fast {
        "moe_down_iq4_nl_fast"
    } else {
        "moe_down_iq4_nl"
    };
    let args = checked_moe_decode_args(kernel, n_in, n_out, n_expert, topk)?;
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    let bank_elements = checked_moe_product(kernel, "expert bank", &[n_in, n_out, n_expert])?;
    let inner_elements = checked_moe_product(kernel, "inner input", &[topk, n_in])?;
    let output_elements = checked_moe_product(kernel, "expert output", &[topk, n_out])?;
    validate_moe_decode_tensor(
        kernel,
        "down expert bank",
        weight,
        bank_elements,
        &[GgmlType::IQ4_NL],
        false,
        2,
    )?;
    validate_moe_decode_tensor(
        kernel,
        "inner input",
        inner,
        inner_elements,
        &[GgmlType::F32],
        false,
        if fast { 16 } else { 4 },
    )?;
    validate_moe_decode_tensor(
        kernel,
        "top-k indices",
        topk_idx,
        topk,
        &[GgmlType::I32, GgmlType::F32],
        false,
        4,
    )?;
    validate_moe_decode_tensor(
        kernel,
        "expert output",
        expert_out,
        output_elements,
        &[GgmlType::F32],
        true,
        4,
    )?;

    let pso = ctx.pipeline(if fast {
        "kernel_moe_down_iq4_nl_f32_fast"
    } else {
        "kernel_moe_down_iq4_nl_f32"
    })?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);
    if fast {
        enc.set_threadgroup_memory(0, 32 * std::mem::size_of::<f32>());
    }

    let (rows_per_simdgroup, simdgroups) = if fast { (2, 2) } else { (1, 4) };
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(rows_per_simdgroup * simdgroups),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_iq4_xs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::IQ4_XS {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs",
            detail: format!("expected IQ4_XS expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_iq4_xs_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);

    const NSG: usize = 4;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_iq4_xs_f32_fast(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs_fast",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::IQ4_XS {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs_fast",
            detail: format!("expected IQ4_XS expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || expert_out.n_elements() as usize != topk * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_iq4_xs_fast",
            detail: format!(
                "shape mismatch: inner={} idx={} out={} expected inner={} idx={topk} out={}",
                inner.n_elements(),
                topk_idx.n_elements(),
                expert_out.n_elements(),
                topk * n_in,
                topk * n_out
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_iq4_xs_f32_fast")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);
    enc.set_threadgroup_memory(0, 32 * std::mem::size_of::<f32>());

    const NR0: usize = 2;
    const NSG: usize = 2;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(NR0 * NSG),
            height: topk,
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

#[allow(non_snake_case)]
pub fn encode_moe_down_weighted_sum_q6_K_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    topk_w: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(256) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q6_K",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if weight.dtype != GgmlType::Q6_K {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q6_K",
            detail: format!("expected Q6_K expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || topk_w.n_elements() as usize != topk
        || out.n_elements() as usize != n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q6_K",
            detail: format!(
                "shape mismatch: inner={} idx={} w={} out={} expected inner={} idx={topk} w={topk} out={n_out}",
                inner.n_elements(),
                topk_idx.n_elements(),
                topk_w.n_elements(),
                out.n_elements(),
                topk * n_in
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_weighted_sum_q6_K_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, topk_w);
    enc.set_tensor(5, out);

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

pub fn encode_moe_down_weighted_sum_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    topk_w: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q8_0",
            detail: format!("n_in={n_in} not divisible by 32"),
        });
    }
    if weight.dtype != GgmlType::Q8_0 {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q8_0",
            detail: format!("expected Q8_0 expert down, got {:?}", weight.dtype),
        });
    }
    if inner.n_elements() as usize != topk * n_in
        || topk_idx.n_elements() as usize != topk
        || topk_w.n_elements() as usize != topk
        || out.n_elements() as usize != n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_down_weighted_sum_q8_0",
            detail: format!(
                "shape mismatch: inner={} idx={} w={} out={} expected inner={} idx={topk} w={topk} out={n_out}",
                inner.n_elements(),
                topk_idx.n_elements(),
                topk_w.n_elements(),
                out.n_elements(),
                topk * n_in
            ),
        });
    }

    let pso = ctx.pipeline("kernel_moe_down_weighted_sum_q8_0_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
            n_expert: n_expert as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, topk_w);
    enc.set_tensor(5, out);

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
    Ok(())
}

pub fn encode_moe_weighted_sum_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    expert_out: &MetalTensor,
    weights: &MetalTensor,
    out: &MetalTensor,
    n_out: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if expert_out.n_elements() as usize != topk * n_out
        || weights.n_elements() as usize != topk
        || out.n_elements() as usize != n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_weighted_sum",
            detail: format!(
                "shape mismatch: expert_out={} weights={} out={} expected {}/{}/{}",
                expert_out.n_elements(),
                weights.n_elements(),
                out.n_elements(),
                topk * n_out,
                topk,
                n_out
            ),
        });
    }
    let pso = ctx.pipeline("kernel_moe_weighted_sum_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_out: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_out: n_out as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, expert_out);
    enc.set_tensor(2, weights);
    enc.set_tensor(3, out);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(tg_threads),
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

pub fn encode_moe_weighted_sum_packed_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    expert_out: &MetalTensor,
    weights: &MetalTensor,
    out: &MetalTensor,
    n_out: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    if expert_out.n_elements() as usize != n_tokens * topk * n_out
        || weights.n_elements() as usize != n_tokens * topk
        || out.n_elements() as usize != n_tokens * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_weighted_sum_packed",
            detail: format!(
                "shape mismatch: expert_out={} weights={} out={} expected expert_out={} weights={} out={}",
                expert_out.n_elements(),
                weights.n_elements(),
                out.n_elements(),
                n_tokens * topk * n_out,
                n_tokens * topk,
                n_tokens * n_out
            ),
        });
    }
    let pso = ctx.pipeline("kernel_moe_weighted_sum_packed_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_out: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_out: n_out as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, expert_out);
    enc.set_tensor(2, weights);
    enc.set_tensor(3, out);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(32),
            height: n_tokens,
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

pub fn encode_moe_grouped_finalizer_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    expert_out: &MetalTensor,
    topk_w: &MetalTensor,
    shared_gate: &MetalTensor,
    shared_out: &MetalTensor,
    x_pack: &MetalTensor,
    n_out: usize,
    topk: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    let n_slots = n_tokens * topk;
    if expert_out.n_elements() as usize != n_slots * n_out
        || topk_w.n_elements() as usize != n_slots
        || shared_gate.n_elements() as usize != n_tokens
        || shared_out.n_elements() as usize != n_tokens * n_out
        || x_pack.n_elements() as usize != n_tokens * n_out
    {
        return Err(MetalError::BadShape {
            kernel: "moe_grouped_finalizer",
            detail: format!(
                "shape mismatch: expert_out={} topk_w={} shared_gate={} shared_out={} x_pack={} expected expert_out={} topk_w={} shared_gate={} shared_out={} x_pack={}",
                expert_out.n_elements(),
                topk_w.n_elements(),
                shared_gate.n_elements(),
                shared_out.n_elements(),
                x_pack.n_elements(),
                n_slots * n_out,
                n_slots,
                n_tokens,
                n_tokens * n_out,
                n_tokens * n_out
            ),
        });
    }
    let pso = ctx.pipeline("kernel_moe_grouped_finalizer_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_out: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_out: n_out as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, expert_out);
    enc.set_tensor(2, topk_w);
    enc.set_tensor(3, shared_gate);
    enc.set_tensor(4, shared_out);
    enc.set_tensor(5, x_pack);
    const THREADS_PER_TG: usize = 64;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(THREADS_PER_TG),
            height: n_tokens,
            depth: 1,
        },
        MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_moe_shared_accum_resid_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    shared_out: &MetalTensor,
    shared_gate: &MetalTensor,
    mixer_out: &MetalTensor,
    x: &MetalTensor,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if shared_gate.n_elements() != 1
        || shared_out.n_elements() as usize != n
        || mixer_out.n_elements() as usize != n
    {
        return Err(MetalError::BadShape {
            kernel: "moe_shared_accum_resid",
            detail: format!(
                "expected shared_gate[1] and shared_out/mixer_out/x n={n}, got gate={} shared={} mixer={}",
                shared_gate.n_elements(),
                shared_out.n_elements(),
                mixer_out.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_moe_shared_accum_resid_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
    }
    enc.set_bytes(0, &Args { n: n as u32 });
    enc.set_tensor(1, shared_out);
    enc.set_tensor(2, shared_gate);
    enc.set_tensor(3, mixer_out);
    enc.set_tensor(4, x);
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

pub fn encode_topk_logits_softmax_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if logits.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax",
            detail: format!("logits.n_elements={} != n={n}", logits.n_elements()),
        });
    }
    if out_idx.n_elements() as usize != k || out_w.n_elements() as usize != k {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax",
            detail: format!(
                "out_idx/out_w expected {k} elements, got {}/{}",
                out_idx.n_elements(),
                out_w.n_elements()
            ),
        });
    }
    if k == 0 || k > 16 {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax",
            detail: format!("k={k} must be in 1..=16"),
        });
    }
    let pso = ctx.pipeline("kernel_topk_logits_softmax_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        k: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            k: k as u32,
        },
    );
    enc.set_tensor(1, logits);
    enc.set_tensor(2, out_idx);
    enc.set_tensor(3, out_w);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_topk_logits_softmax_parallel_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if logits.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_parallel",
            detail: format!("logits.n_elements={} != n={n}", logits.n_elements()),
        });
    }
    if out_idx.n_elements() as usize != k || out_w.n_elements() as usize != k {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_parallel",
            detail: format!(
                "out_idx/out_w expected {k} elements, got {}/{}",
                out_idx.n_elements(),
                out_w.n_elements()
            ),
        });
    }
    if n == 0 || n > 256 || k == 0 || k > 16 || k > n {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_parallel",
            detail: format!("expected 1 <= k <= n <= 256 and k <= 16, got n={n} k={k}"),
        });
    }
    let pso = ctx.pipeline("kernel_topk_logits_softmax_parallel_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        k: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            k: k as u32,
        },
    );
    enc.set_tensor(1, logits);
    enc.set_tensor(2, out_idx);
    enc.set_tensor(3, out_w);
    const THREADS: usize = 256;
    enc.set_threadgroup_memory(0, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, THREADS * std::mem::size_of::<i32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
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

pub fn encode_topk_logits_softmax_dot_sigmoid_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    shared_weight: &MetalTensor,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    shared_out: &MetalTensor,
    n_expert: usize,
    topk: usize,
    hidden: usize,
) -> Result<(), MetalError> {
    if logits.dtype != GgmlType::F32
        || shared_weight.dtype != GgmlType::F32
        || x.dtype != GgmlType::F32
        || out_w.dtype != GgmlType::F32
        || shared_out.dtype != GgmlType::F32
    {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_dot_sigmoid",
            detail: format!(
                "expected F32 logits/shared_weight/x/out_w/shared_out, got {:?}/{:?}/{:?}/{:?}/{:?}",
                logits.dtype, shared_weight.dtype, x.dtype, out_w.dtype, shared_out.dtype
            ),
        });
    }
    if n_expert == 0 || n_expert > 256 || topk == 0 || topk > 16 || topk > n_expert {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_dot_sigmoid",
            detail: format!(
                "expected 1 <= topk <= n_expert <= 256 and topk <= 16, got n_expert={n_expert} topk={topk}"
            ),
        });
    }
    if logits.n_elements() as usize != n_expert
        || out_idx.n_elements() as usize != topk
        || out_w.n_elements() as usize != topk
        || shared_weight.n_elements() as usize != hidden
        || x.n_elements() as usize != hidden
        || shared_out.n_elements() != 1
    {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_dot_sigmoid",
            detail: format!(
                "shape mismatch logits={} idx={} w={} shared_weight={} x={} shared_out={} expected {n_expert}/{topk}/{topk}/{hidden}/{hidden}/1",
                logits.n_elements(),
                out_idx.n_elements(),
                out_w.n_elements(),
                shared_weight.n_elements(),
                x.n_elements(),
                shared_out.n_elements()
            ),
        });
    }

    let pso = ctx.pipeline("kernel_topk_logits_softmax_dot_sigmoid_f32")?;
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
            n_tokens: 1,
        },
    );
    enc.set_tensor(1, logits);
    enc.set_tensor(2, shared_weight);
    enc.set_tensor(3, x);
    enc.set_tensor(4, out_idx);
    enc.set_tensor(5, out_w);
    enc.set_tensor(6, shared_out);
    const THREADS: usize = 256;
    enc.set_threadgroup_memory(0, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, THREADS * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, THREADS * std::mem::size_of::<i32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
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

pub fn encode_topk_logits_softmax_dot_sigmoid_packed_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    shared_weight: &MetalTensor,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    shared_out: &MetalTensor,
    n_expert: usize,
    topk: usize,
    hidden: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "topk_logits_softmax_dot_sigmoid_packed";
    if logits.dtype != GgmlType::F32
        || shared_weight.dtype != GgmlType::F32
        || x.dtype != GgmlType::F32
        || !matches!(out_idx.dtype, GgmlType::I32 | GgmlType::F32)
        || out_w.dtype != GgmlType::F32
        || shared_out.dtype != GgmlType::F32
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected F32 logits/shared_weight/x/out_w/shared_out and I32-or-F32 indices, got {:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
                logits.dtype,
                shared_weight.dtype,
                x.dtype,
                out_idx.dtype,
                out_w.dtype,
                shared_out.dtype
            ),
        });
    }
    if n_expert == 0 || n_expert > 512 || topk == 0 || topk > 16 || topk > n_expert {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected 1 <= topk <= n_expert <= 512 and topk <= 16, got n_expert={n_expert} topk={topk}"
            ),
        });
    }
    if hidden == 0 || n_tokens == 0 {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("hidden={hidden} and n_tokens={n_tokens} must be nonzero"),
        });
    }
    for (name, value) in [
        ("n_expert", n_expert),
        ("topk", topk),
        ("hidden", hidden),
        ("n_tokens", n_tokens),
    ] {
        if u32::try_from(value).is_err() {
            return Err(MetalError::BadShape {
                kernel: KERNEL,
                detail: format!("{name}={value} exceeds u32"),
            });
        }
    }
    let checked_elements = |name: &str, left: usize, right: usize| {
        left.checked_mul(right).ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("{name} element count overflow: {left}*{right}"),
        })
    };
    let logits_elements = checked_elements("logits", n_tokens, n_expert)?;
    let topk_elements = checked_elements("top-k", n_tokens, topk)?;
    let input_elements = checked_elements("input", n_tokens, hidden)?;
    if logits.n_elements() as usize != logits_elements
        || out_idx.n_elements() as usize != topk_elements
        || out_w.n_elements() as usize != topk_elements
        || shared_weight.n_elements() as usize != hidden
        || x.n_elements() as usize != input_elements
        || shared_out.n_elements() as usize != n_tokens
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "shape mismatch logits={} idx={} w={} shared_weight={} x={} shared_out={} expected {}/{}/{}/{hidden}/{}/{}",
                logits.n_elements(),
                out_idx.n_elements(),
                out_w.n_elements(),
                shared_weight.n_elements(),
                x.n_elements(),
                shared_out.n_elements(),
                logits_elements,
                topk_elements,
                topk_elements,
                input_elements,
                n_tokens
            ),
        });
    }

    let pso = ctx.pipeline("kernel_topk_logits_softmax_dot_sigmoid_packed_f32")?;
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
    let threads = if n_expert <= 256 { 256 } else { 512 };
    if pso.maxTotalThreadsPerThreadgroup() < threads {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "512-expert route requires {threads} threads, pipeline exposes {}",
                pso.maxTotalThreadsPerThreadgroup()
            ),
        });
    }
    let dynamic_memory = threads
        .checked_mul(3 * std::mem::size_of::<u32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "threadgroup memory size overflow".into(),
        })?;
    let required_memory = pso
        .staticThreadgroupMemoryLength()
        .checked_add(dynamic_memory)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "threadgroup memory requirement overflow".into(),
        })?;
    if required_memory > ctx.device.maxThreadgroupMemoryLength() {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "route requires {required_memory} threadgroup bytes, device exposes {}",
                ctx.device.maxThreadgroupMemoryLength()
            ),
        });
    }
    enc.set_threadgroup_memory(0, threads * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, threads * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, threads * std::mem::size_of::<i32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: n_tokens,
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

pub fn encode_moe_route_bucket_slots_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    topk_idx: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    n_expert: usize,
    n_tokens: usize,
    topk: usize,
) -> Result<(), MetalError> {
    if topk_idx.n_elements() as usize != n_tokens * topk
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
    {
        return Err(MetalError::BadShape {
            kernel: "moe_route_bucket_slots",
            detail: format!(
                "shape mismatch idx={} counts={} ids={} expected idx={} counts={} ids={}",
                topk_idx.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                n_tokens * topk,
                n_expert,
                n_expert * n_tokens
            ),
        });
    }
    let pso = ctx.pipeline("kernel_moe_route_bucket_slots_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_expert: u32,
        n_tokens: u32,
        topk: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_expert: n_expert as u32,
            n_tokens: n_tokens as u32,
            topk: topk as u32,
        },
    );
    enc.set_tensor(1, topk_idx);
    enc.set_tensor(2, counts);
    enc.set_tensor(3, ids);
    enc.dispatch(
        MTLSize {
            width: n_expert.div_ceil(256),
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
