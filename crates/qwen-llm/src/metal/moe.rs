//! MoE routing, swiglu, down, and grouped/packed slot kernels.

use super::*;

#[cfg(test)]
mod route_parity_tests;

#[cfg(test)]
mod nonfinite_tests;

/// Qwen/Flash-Next route-status word bits written by the top-k and bucket
/// kernels (`kernels/moe.metal`). A ready row has bit 0 set and neither
/// failure bit.
pub const MOE_ROUTE_STATUS_READY: i32 = 1;
pub const MOE_ROUTE_STATUS_NONFINITE: i32 = 2;
pub const MOE_ROUTE_STATUS_OVERFLOW: i32 = 4;

#[cfg(test)]
thread_local! {
    static FAIL_NEXT_ROUTE_CHECK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test seam: the next route-status check on this thread reports a failed
/// route, so production completion paths can be exercised without a real
/// non-finite router input.
#[cfg(test)]
#[must_use = "the injection is disarmed when the guard drops"]
pub(crate) fn fail_next_route_status_check() -> FailNextRouteStatusCheck {
    FAIL_NEXT_ROUTE_CHECK.with(|flag| flag.set(true));
    FailNextRouteStatusCheck
}

#[cfg(test)]
pub(crate) struct FailNextRouteStatusCheck;

#[cfg(test)]
impl Drop for FailNextRouteStatusCheck {
    fn drop(&mut self) {
        FAIL_NEXT_ROUTE_CHECK.with(|flag| flag.set(false));
    }
}

/// Read and validate a route-status buffer after its command buffer has
/// completed. `rows` is the number of per-token words the dispatch could have
/// written; zero words are treated as "no route dispatched" and skipped.
pub fn check_moe_route_status(
    kernel: &'static str,
    status: &MetalTensor,
    rows: usize,
) -> Result<(), MetalError> {
    #[cfg(test)]
    if FAIL_NEXT_ROUTE_CHECK.with(|flag| flag.replace(false)) {
        return Err(MetalError::BadShape {
            kernel,
            detail: "injected route status failure".into(),
        });
    }
    if status.dtype != GgmlType::I32 || status.n_elements() < rows as u64 {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "route status must be I32[{rows}], got {:?} {:?}",
                status.dtype, status.shape
            ),
        });
    }
    let base = status.offset / 4;
    let words = unsafe {
        std::slice::from_raw_parts(
            (status.buffer.contents().as_ptr() as *const i32).add(base as usize),
            rows,
        )
    };
    for (row, &word) in words.iter().enumerate() {
        // Zero means no route was dispatched for this row (dense layer,
        // no-op route ablation, or a diagnostic probe that bypasses this
        // status word). Only rows that actually ran a route are judged.
        if word == 0 {
            continue;
        }
        if word & MOE_ROUTE_STATUS_READY != 0
            && word & (MOE_ROUTE_STATUS_NONFINITE | MOE_ROUTE_STATUS_OVERFLOW) == 0
        {
            continue;
        }
        let reason = if word & MOE_ROUTE_STATUS_OVERFLOW != 0 {
            "fixed-capacity expert bucket overflow"
        } else if word & MOE_ROUTE_STATUS_NONFINITE != 0 {
            "non-finite router logits"
        } else {
            "route did not complete"
        };
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("route row {row} status {word}: {reason}"),
        });
    }
    Ok(())
}

#[cfg(test)]
fn diagnostic_max_abs(a: &[f32], b: &[f32]) -> f32 {
    crate::compare::report_max_abs_diff_f32(a.iter(), b.iter())
}

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
    super::checks::check_disjoint(
        "moe_weighted_sum",
        out,
        &[(expert_out, "expert outputs"), (weights, "weights")],
    )?;
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
            n_out: super::checks::to_u32("moe_reduce", n_out, "n_out")?,
            topk: super::checks::to_u32("moe_reduce", topk, "topk")?,
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
    super::checks::check_disjoint(
        "moe_weighted_sum_packed",
        out,
        &[(expert_out, "expert outputs"), (weights, "weights")],
    )?;
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
            n_out: super::checks::to_u32("moe_reduce", n_out, "n_out")?,
            topk: super::checks::to_u32("moe_reduce", topk, "topk")?,
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
    super::checks::check_disjoint(
        "moe_grouped_finalizer",
        x_pack,
        &[
            (expert_out, "expert outputs"),
            (topk_w, "top-k weights"),
            (shared_gate, "shared gate"),
            (shared_out, "shared output"),
        ],
    )?;
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
            n_out: super::checks::to_u32("moe_reduce", n_out, "n_out")?,
            topk: super::checks::to_u32("moe_reduce", topk, "topk")?,
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
    // Same-index wobble: shared_out, mixer_out and x may alias only when they
    // cover exactly the same bytes; shared_gate is read by every thread and
    // must stay separate from both written streams.
    super::checks::check_disjoint_or_exact(
        "moe_shared_accum_resid",
        mixer_out,
        &[(shared_out, "shared output")],
    )?;
    super::checks::check_disjoint_or_exact(
        "moe_shared_accum_resid",
        x,
        &[(shared_out, "shared output")],
    )?;
    super::checks::check_disjoint_or_exact(
        "moe_shared_accum_resid",
        mixer_out,
        &[(x, "residual")],
    )?;
    super::checks::check_disjoint(
        "moe_shared_accum_resid",
        mixer_out,
        &[(shared_gate, "shared gate")],
    )?;
    super::checks::check_disjoint("moe_shared_accum_resid", x, &[(shared_gate, "shared gate")])?;
    let pso = ctx.pipeline("kernel_moe_shared_accum_resid_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: super::checks::to_u32("moe_shared_accum_resid", n, "n")?,
        },
    );
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

pub fn encode_topk_logits_softmax_f32_status(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    status: &MetalTensor,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if n == 0 || k > n {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax",
            detail: format!("need 1 <= k <= n, got n={n} k={k}"),
        });
    }
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
    if status.dtype != GgmlType::I32 || status.n_elements() < 1 {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax",
            detail: "status must be a nonempty I32 buffer".into(),
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
    enc.set_tensor(4, status);
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

pub fn encode_topk_logits_softmax_parallel_f32_status(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    status: &MetalTensor,
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
    if status.dtype != GgmlType::I32 || status.n_elements() < 1 {
        return Err(MetalError::BadShape {
            kernel: "topk_logits_softmax_parallel",
            detail: "status must be a nonempty I32 buffer".into(),
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
    enc.set_tensor(4, status);
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

pub fn encode_topk_logits_softmax_dot_sigmoid_f32_status(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    shared_weight: &MetalTensor,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    shared_out: &MetalTensor,
    status: &MetalTensor,
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
        || status.dtype != GgmlType::I32
        || status.n_elements() < 1
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
    enc.set_tensor(7, status);
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

pub fn encode_topk_logits_softmax_dot_sigmoid_packed_f32_status(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    shared_weight: &MetalTensor,
    x: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    shared_out: &MetalTensor,
    status: &MetalTensor,
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
        || status.dtype != GgmlType::I32
        || status.n_elements() < n_tokens as u64
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
    enc.set_tensor(7, status);
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

/// Buckets route ids by expert with one thread per expert scanning every
/// rows x topk id (`kernel_moe_route_bucket_slots_f32`).
pub fn encode_moe_route_bucket_slots_f32_status(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    topk_idx: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    n_expert: usize,
    n_tokens: usize,
    topk: usize,
) -> Result<(), MetalError> {
    encode_route_bucket(
        ctx, enc, topk_idx, counts, ids, status, n_expert, n_tokens, topk, false,
    )
}

/// Same counts and slot lists as [`encode_moe_route_bucket_slots_f32`]
/// (bitwise), with one 256-thread threadgroup per expert that counts, scans
/// and scatters its matches in parallel.
pub fn encode_moe_route_bucket_slots_parallel_f32_status(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    topk_idx: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    n_expert: usize,
    n_tokens: usize,
    topk: usize,
) -> Result<(), MetalError> {
    encode_route_bucket(
        ctx, enc, topk_idx, counts, ids, status, n_expert, n_tokens, topk, true,
    )
}

/// Shared fixed-capacity bucket layout validation for every route/bucket
/// encoder (standalone and fused).
pub(super) fn validate_bucket_layout(
    kernel: &'static str,
    n_expert: usize,
    n_tokens: usize,
    topk: usize,
) -> Result<(usize, usize), MetalError> {
    if n_expert == 0 || n_tokens == 0 || topk == 0 {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "degenerate bucket shape n_expert={n_expert} n_tokens={n_tokens} topk={topk}"
            ),
        });
    }
    let slot_count = n_tokens
        .checked_mul(topk)
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: "n_tokens * topk overflow".into(),
        })?;
    let id_count = n_expert
        .checked_mul(n_tokens)
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: "n_expert * n_tokens overflow".into(),
        })?;
    if i32::try_from(slot_count).is_err() {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("slot count {slot_count} exceeds signed bucket ids"),
        });
    }
    super::checks::to_u32(kernel, n_expert, "n_expert")?;
    super::checks::to_u32(kernel, n_tokens, "n_tokens")?;
    super::checks::to_u32(kernel, topk, "topk")?;
    super::checks::to_u32(kernel, slot_count, "n_tokens*topk")?;
    super::checks::to_u32(kernel, id_count, "n_expert*n_tokens")?;
    Ok((slot_count, id_count))
}

#[allow(clippy::too_many_arguments)]
fn encode_route_bucket(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    topk_idx: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    n_expert: usize,
    n_tokens: usize,
    topk: usize,
    parallel: bool,
) -> Result<(), MetalError> {
    let (slot_count, id_count) =
        validate_bucket_layout("moe_route_bucket_slots", n_expert, n_tokens, topk)?;
    if topk_idx.n_elements() as usize != slot_count
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != id_count
        || status.dtype != GgmlType::I32
        || status.n_elements() < n_tokens as u64
    {
        return Err(MetalError::BadShape {
            kernel: "moe_route_bucket_slots",
            detail: format!(
                "shape mismatch idx={} counts={} ids={} expected idx={} counts={} ids={}",
                topk_idx.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                slot_count,
                n_expert,
                id_count
            ),
        });
    }
    let pso = ctx.pipeline(if parallel {
        "kernel_moe_route_bucket_slots_parallel_f32"
    } else {
        "kernel_moe_route_bucket_slots_f32"
    })?;
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
            n_expert: super::checks::to_u32("moe_route_bucket_slots", n_expert, "n_expert")?,
            n_tokens: super::checks::to_u32("moe_route_bucket_slots", n_tokens, "n_tokens")?,
            topk: super::checks::to_u32("moe_route_bucket_slots", topk, "topk")?,
        },
    );
    enc.set_tensor(1, topk_idx);
    enc.set_tensor(2, counts);
    enc.set_tensor(3, ids);
    enc.set_tensor(4, status);
    enc.dispatch(
        MTLSize {
            width: if parallel {
                n_expert
            } else {
                n_expert.div_ceil(256)
            },
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

/// Legacy top-k convenience for tests/diagnostics: allocates a private
/// status word and does not return it. Production paths call
/// [`encode_topk_logits_softmax_f32_status`] and check the word.
pub fn encode_topk_logits_softmax_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    let status = MetalTensor::zeros_i32(ctx, vec![1])?;
    encode_topk_logits_softmax_f32_status(ctx, enc, logits, out_idx, out_w, &status, n, k)
}

/// See [`encode_topk_logits_softmax_f32`].
pub fn encode_topk_logits_softmax_parallel_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    out_idx: &MetalTensor,
    out_w: &MetalTensor,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    let status = MetalTensor::zeros_i32(ctx, vec![1])?;
    encode_topk_logits_softmax_parallel_f32_status(ctx, enc, logits, out_idx, out_w, &status, n, k)
}

/// See [`encode_topk_logits_softmax_f32`].
#[allow(clippy::too_many_arguments)]
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
    let status = MetalTensor::zeros_i32(ctx, vec![1])?;
    encode_topk_logits_softmax_dot_sigmoid_f32_status(
        ctx,
        enc,
        logits,
        shared_weight,
        x,
        out_idx,
        out_w,
        shared_out,
        &status,
        n_expert,
        topk,
        hidden,
    )
}

/// See [`encode_topk_logits_softmax_f32`].
#[allow(clippy::too_many_arguments)]
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
    let status = MetalTensor::zeros_i32(ctx, vec![n_tokens as u64])?;
    encode_topk_logits_softmax_dot_sigmoid_packed_f32_status(
        ctx,
        enc,
        logits,
        shared_weight,
        x,
        out_idx,
        out_w,
        shared_out,
        &status,
        n_expert,
        topk,
        hidden,
        n_tokens,
    )
}

/// See [`encode_topk_logits_softmax_f32`].
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
    let status = MetalTensor::zeros_i32(ctx, vec![n_tokens as u64])?;
    encode_moe_route_bucket_slots_f32_status(
        ctx, enc, topk_idx, counts, ids, &status, n_expert, n_tokens, topk,
    )
}

/// See [`encode_topk_logits_softmax_f32`].
pub fn encode_moe_route_bucket_slots_parallel_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    topk_idx: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    n_expert: usize,
    n_tokens: usize,
    topk: usize,
) -> Result<(), MetalError> {
    let status = MetalTensor::zeros_i32(ctx, vec![n_tokens as u64])?;
    encode_moe_route_bucket_slots_parallel_f32_status(
        ctx, enc, topk_idx, counts, ids, &status, n_expert, n_tokens, topk,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;

    #[test]
    fn parallel_route_bucket_matches_serial_bitwise() {
        let ctx = MetalContext::new().unwrap();
        for (n_expert, n_tokens, topk) in [
            (384usize, 4096usize, 6usize),
            (384, 3, 6),
            (8, 37, 2),
            (160, 2048, 8),
            (512, 512, 10),
            (512, 513, 10),
            (512, 1024, 10),
            (512, 2045, 10),
            (512, 2048, 10),
        ] {
            // Skewed, repeating expert choices (some experts empty, some hot).
            let idx: Vec<i32> = (0..n_tokens * topk)
                .map(|s| ((s * 7919 + s / topk * 31) % (n_expert * 3 / 4 + 1)) as i32)
                .collect();
            let top = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&idx),
                vec![(n_tokens * topk) as u64],
                GgmlType::I32,
            )
            .unwrap();
            let run = |parallel: bool| {
                let counts = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&vec![-7i32; n_expert]),
                    vec![n_expert as u64],
                    GgmlType::I32,
                )
                .unwrap();
                let ids = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&vec![-9i32; n_expert * n_tokens]),
                    vec![(n_expert * n_tokens) as u64],
                    GgmlType::I32,
                )
                .unwrap();
                let status = MetalTensor::zeros_i32(&ctx, vec![n_tokens as u64]).unwrap();
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                encode_route_bucket(
                    &ctx, &enc, &top, &counts, &ids, &status, n_expert, n_tokens, topk, parallel,
                )
                .unwrap();
                enc.end();
                crate::metal::commit_and_wait(&command).unwrap();
                let read = |t: &MetalTensor, n: usize| unsafe {
                    std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<i32>(), n)
                        .to_vec()
                };
                let counts_v = read(&counts, n_expert);
                let ids_v = read(&ids, n_expert * n_tokens);
                // Only each expert's filled prefix is defined output.
                let filled: Vec<Vec<i32>> = (0..n_expert)
                    .map(|e| ids_v[e * n_tokens..e * n_tokens + counts_v[e] as usize].to_vec())
                    .collect();
                (counts_v, filled)
            };
            let serial = run(false);
            let parallel = run(true);
            assert_eq!(
                parallel, serial,
                "experts={n_expert} tokens={n_tokens} topk={topk}"
            );
            assert_eq!(
                serial.0.iter().map(|&c| c as usize).sum::<usize>(),
                n_tokens * topk
            );
        }
    }

    #[test]
    fn topk_nonfinite_logits_stay_unique_and_raise_status() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let cases: [(&[f32], bool); 3] = [
            (&vec![f32::NAN; 32], true),
            (&vec![f32::NEG_INFINITY; 32], true),
            (&(0..32).map(|i| i as f32 * 0.25).collect::<Vec<_>>(), false),
        ];
        for (logits, nonfinite) in cases {
            let input = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(logits),
                vec![logits.len() as u64],
                GgmlType::F32,
            )
            .unwrap();
            for parallel in [false, true] {
                let ids = MetalTensor::zeros_i32(&ctx, vec![6]).unwrap();
                let weights = MetalTensor::zeros_f32(&ctx, vec![6]).unwrap();
                let status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                if parallel {
                    encode_topk_logits_softmax_parallel_f32_status(
                        &ctx,
                        &enc,
                        &input,
                        &ids,
                        &weights,
                        &status,
                        logits.len(),
                        6,
                    )
                    .unwrap();
                } else {
                    encode_topk_logits_softmax_f32_status(
                        &ctx,
                        &enc,
                        &input,
                        &ids,
                        &weights,
                        &status,
                        logits.len(),
                        6,
                    )
                    .unwrap();
                }
                enc.end();
                crate::metal::commit_and_wait(&command).unwrap();
                let read_i32 = |t: &MetalTensor, n: usize| unsafe {
                    std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<i32>(), n)
                        .to_vec()
                };
                let words = read_i32(&ids, 6);
                for (slot, &id) in words.iter().enumerate() {
                    assert!(
                        (0..logits.len() as i32).contains(&id),
                        "slot {slot} id {id}"
                    );
                    assert!(
                        !words[..slot].contains(&id),
                        "duplicate id {id} in {words:?}"
                    );
                }
                let word = read_i32(&status, 1)[0];
                assert_eq!(word & MOE_ROUTE_STATUS_READY, MOE_ROUTE_STATUS_READY);
                assert_eq!(
                    word & MOE_ROUTE_STATUS_NONFINITE != 0,
                    nonfinite,
                    "status {word} for parallel={parallel}"
                );
            }
        }
    }

    #[test]
    fn packed_route_nonfinite_is_unique_per_token_and_flagged() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N_EXPERT: usize = 8;
        const TOPK: usize = 3;
        const HIDDEN: usize = 4;
        const N_TOKENS: usize = 2;
        let mut logits = vec![f32::NAN; N_TOKENS * N_EXPERT];
        logits[N_EXPERT..].fill(f32::NEG_INFINITY);
        let logits_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&logits),
            vec![(N_TOKENS * N_EXPERT) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let shared_weight = MetalTensor::zeros_f32(&ctx, vec![HIDDEN as u64]).unwrap();
        let x = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * HIDDEN) as u64]).unwrap();
        let ids = MetalTensor::zeros_i32(&ctx, vec![(N_TOKENS * TOPK) as u64]).unwrap();
        let weights = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * TOPK) as u64]).unwrap();
        let shared_out = MetalTensor::zeros_f32(&ctx, vec![N_TOKENS as u64]).unwrap();
        let status = MetalTensor::zeros_i32(&ctx, vec![N_TOKENS as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        encode_topk_logits_softmax_dot_sigmoid_packed_f32_status(
            &ctx,
            &enc,
            &logits_t,
            &shared_weight,
            &x,
            &ids,
            &weights,
            &shared_out,
            &status,
            N_EXPERT,
            TOPK,
            HIDDEN,
            N_TOKENS,
        )
        .unwrap();
        enc.end();
        crate::metal::commit_and_wait(&command).unwrap();
        let read_i32 = |t: &MetalTensor, n: usize| unsafe {
            std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<i32>(), n).to_vec()
        };
        let ids = read_i32(&ids, N_TOKENS * TOPK);
        for token in 0..N_TOKENS {
            let row = &ids[token * TOPK..(token + 1) * TOPK];
            for (slot, &id) in row.iter().enumerate() {
                assert!((0..N_EXPERT as i32).contains(&id), "row {row:?}");
                assert!(!row[..slot].contains(&id), "duplicate in {row:?}");
            }
        }
        let words = read_i32(&status, N_TOKENS);
        assert!(words.iter().all(|w| w & MOE_ROUTE_STATUS_NONFINITE != 0));
    }

    #[test]
    fn bucket_overflow_is_bounded_and_flagged() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N_EXPERT: usize = 2;
        const N_TOKENS: usize = 4;
        const TOPK: usize = 3;
        // Every slot selects expert 0: 12 writes into a 4-slot bucket.
        let top_idx = vec![0i32; N_TOKENS * TOPK];
        let top = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&top_idx),
            vec![top_idx.len() as u64],
            GgmlType::I32,
        )
        .unwrap();
        for parallel in [false, true] {
            let counts = MetalTensor::zeros_i32(&ctx, vec![N_EXPERT as u64]).unwrap();
            let guard = 8usize;
            let ids_backing = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&vec![-77i32; N_EXPERT * N_TOKENS + guard]),
                vec![(N_EXPERT * N_TOKENS + guard) as u64],
                GgmlType::I32,
            )
            .unwrap();
            let ids = ids_backing.view_subrange(0, vec![(N_EXPERT * N_TOKENS) as u64]);
            let status = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&vec![MOE_ROUTE_STATUS_READY; N_TOKENS]),
                vec![N_TOKENS as u64],
                GgmlType::I32,
            )
            .unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            if parallel {
                encode_moe_route_bucket_slots_parallel_f32_status(
                    &ctx, &enc, &top, &counts, &ids, &status, N_EXPERT, N_TOKENS, TOPK,
                )
                .unwrap();
            } else {
                encode_moe_route_bucket_slots_f32_status(
                    &ctx, &enc, &top, &counts, &ids, &status, N_EXPERT, N_TOKENS, TOPK,
                )
                .unwrap();
            }
            enc.end();
            crate::metal::commit_and_wait(&command).unwrap();
            let read_i32 = |t: &MetalTensor, n: usize| unsafe {
                std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<i32>(), n).to_vec()
            };
            assert_eq!(read_i32(&counts, N_EXPERT), vec![12, 0]);
            let tail = read_i32(&ids_backing, N_EXPERT * N_TOKENS + guard);
            assert_eq!(&tail[N_EXPERT * N_TOKENS..], &vec![-77; guard]);
            // Expert 1's bucket is empty: a duplicate expert-0 write must not
            // spill into the adjacent expert row.
            assert_eq!(&tail[N_TOKENS..2 * N_TOKENS], &vec![-77; N_TOKENS]);
            let words = read_i32(&status, N_TOKENS);
            assert!(
                words.iter().any(|w| w & MOE_ROUTE_STATUS_OVERFLOW != 0),
                "status {words:?}"
            );
            let err = check_moe_route_status("test_route", &status, N_TOKENS).unwrap_err();
            assert!(err.to_string().contains("bucket overflow"), "{err}");
        }
    }

    #[test]
    fn route_status_names_both_failure_bits() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let skipped =
            MetalTensor::from_bytes(&ctx, bytemuck::cast_slice(&[0i32]), vec![1], GgmlType::I32)
                .unwrap();
        check_moe_route_status("test_route", &skipped, 1).expect("zero means no route dispatched");
        for (word, needle) in [
            (
                MOE_ROUTE_STATUS_READY | MOE_ROUTE_STATUS_NONFINITE,
                "non-finite",
            ),
            (
                MOE_ROUTE_STATUS_READY | MOE_ROUTE_STATUS_OVERFLOW,
                "overflow",
            ),
        ] {
            let status = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&[word]),
                vec![1],
                GgmlType::I32,
            )
            .unwrap();
            let err = check_moe_route_status("test_route", &status, 1).unwrap_err();
            assert!(err.to_string().contains(needle), "{err}");
        }
    }

    #[test]
    fn reduce_encoders_refuse_output_aliasing() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let backing = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[1.0f32; 12]),
            vec![12],
            GgmlType::F32,
        )
        .unwrap();
        let a = backing.view_subrange(0, vec![8]);
        let a_partial = backing.view_subrange(1, vec![8]);
        let b = MetalTensor::zeros_f32(&ctx, vec![8]).unwrap();
        let weights = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[1.0f32]),
            vec![1],
            GgmlType::F32,
        )
        .unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let err = encode_moe_weighted_sum_f32(&ctx, &enc, &a, &weights, &a, 8, 1).unwrap_err();
        assert!(err.to_string().contains("aliases expert outputs"), "{err}");
        let err =
            encode_moe_weighted_sum_packed_f32(&ctx, &enc, &a, &weights, &a, 8, 1, 1).unwrap_err();
        assert!(err.to_string().contains("aliases expert outputs"), "{err}");
        let gate = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[0.5f32]),
            vec![1],
            GgmlType::F32,
        )
        .unwrap();
        let err =
            encode_moe_shared_accum_resid_f32(&ctx, &enc, &a, &gate, &b, &a_partial).unwrap_err();
        assert!(err.to_string().contains("partially aliases"), "{err}");
        // Exact same-index aliases are allowed for this kernel.
        encode_moe_shared_accum_resid_f32(&ctx, &enc, &a, &gate, &a, &b).unwrap();
    }

    #[test]
    fn diagnostic_max_abs_marks_uncomparable_inputs_non_finite() {
        assert_eq!(diagnostic_max_abs(&[1.0, 3.0], &[1.5, 1.0]), 2.0);
        assert!(diagnostic_max_abs(&[1.0], &[1.0, 2.0]).is_nan());
        assert!(diagnostic_max_abs(&[1.0], &[f32::NAN]).is_nan());
        assert!(diagnostic_max_abs(&[], &[]).is_nan());
    }

    #[test]
    #[ignore]
    fn mxfp4_f32_matrix_tile_k216_n4096_all_expert_floor() {
        run_mxfp4_f32_matrix_tile_k216_bucket_floor(24_576, [216; 2], 114, 0.73);
    }

    #[test]
    fn mat_mat_f32_router_e8p32_strict_matches_generic_bits() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        for (n_in, n_out, queries) in [
            (4096usize, 160usize, &[1usize, 32, 128, 512, 2048, 4096][..]),
            (4096, 288, &[2048][..]),
            (7, 8, &[129][..]),
            (257, 24, &[129][..]),
        ] {
            let mut state = 0x8b8b_8b8b_u32;
            let mut sample = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 * (1.0 / 16_777_216.0) - 0.5) * 0.25
            };
            let weight = (0..n_in * n_out).map(|_| sample()).collect::<Vec<_>>();
            let weight_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&weight),
                vec![n_in as u64, n_out as u64],
                GgmlType::F32,
            )
            .expect("weight tensor");
            let short_weight = MetalTensor::zeros_f32(&ctx, vec![(n_in * n_out - 1) as u64])
                .expect("short weight tensor");
            let validation_x =
                MetalTensor::zeros_f32(&ctx, vec![n_in as u64]).expect("validation input");
            let validation_y =
                MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("validation output");
            let command = ctx.queue.commandBuffer().expect("validation command");
            let encoder = KernelEncoder::begin(&command);
            assert!(
                encode_mat_mat_f32_router_e8p32_strict(
                    &ctx,
                    &encoder,
                    &short_weight,
                    &validation_x,
                    &validation_y,
                    n_in,
                    n_out,
                    1,
                )
                .is_err()
            );
            encoder.end();

            // Both kernels handle scalar K tails and a partial final query group.
            for &n_query in queries {
                let x = (0..n_in * n_query).map(|_| sample()).collect::<Vec<_>>();
                let x_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&x),
                    vec![n_query as u64, n_in as u64],
                    GgmlType::F32,
                )
                .expect("input tensor");
                let generic = MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64])
                    .expect("generic output");
                let strict = MetalTensor::zeros_f32(&ctx, vec![n_out as u64, n_query as u64])
                    .expect("strict output");
                one_shot(&ctx, |enc| {
                    encode_mat_mat_f32(&ctx, enc, &weight_t, &x_t, &generic, n_in, n_out, n_query)?;
                    encode_mat_mat_f32_router_e8p32_strict(
                        &ctx, enc, &weight_t, &x_t, &strict, n_in, n_out, n_query,
                    )
                })
                .expect("router differential");

                let generic = read_back_f32(&generic.buffer, n_out * n_query);
                let strict = read_back_f32(&strict.buffer, n_out * n_query);
                if let Some((index, (expected, actual))) = generic
                    .iter()
                    .zip(&strict)
                    .enumerate()
                    .find(|(_, (expected, actual))| expected.to_bits() != actual.to_bits())
                {
                    panic!(
                        "n_in={n_in} n_out={n_out} n_query={n_query} output {index} differs: generic={expected:?} strict={actual:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn moe_grouped_finalizer_matches_cpu() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let n_out = 512usize;
        let topk = 4usize;
        let n_tokens = 2usize;
        let expert_out: Vec<f32> = (0..n_tokens * topk * n_out)
            .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
            .collect();
        let weights: Vec<f32> = (0..n_tokens * topk)
            .map(|i| 0.1 + (i % topk) as f32 * 0.1)
            .collect();
        let shared_gate = vec![0.65f32, 0.35];
        let shared_out: Vec<f32> = (0..n_tokens * n_out)
            .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
            .collect();
        let x_init: Vec<f32> = (0..n_tokens * n_out)
            .map(|i| ((i % 13) as f32 - 6.0) * 3e-2)
            .collect();
        let expected: Vec<f32> = (0..n_tokens * n_out)
            .map(|i| {
                let token = i / n_out;
                let routed: f32 = (0..topk)
                    .map(|slot| {
                        weights[token * topk + slot]
                            * expert_out[(token * topk + slot) * n_out + i % n_out]
                    })
                    .sum();
                x_init[i] + routed + shared_gate[token] * shared_out[i]
            })
            .collect();

        let expert_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&expert_out),
            vec![(n_tokens * topk * n_out) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let weights_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&weights),
            vec![(n_tokens * topk) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let gate_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&shared_gate),
            vec![n_tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let shared_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&shared_out),
            vec![(n_tokens * n_out) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x_init),
            vec![(n_tokens * n_out) as u64],
            GgmlType::F32,
        )
        .unwrap();
        one_shot(&ctx, |enc| {
            encode_moe_grouped_finalizer_f32(
                &ctx, enc, &expert_t, &weights_t, &gate_t, &shared_t, &x_t, n_out, topk, n_tokens,
            )
        })
        .unwrap();
        let gpu = read_back_f32(&x_t.buffer, n_tokens * n_out);
        let max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu.iter(),
            expected.iter(),
        );
        assert!(max_abs < 1e-6, "grouped finalizer drift {max_abs}");
    }

    #[test]
    fn moe_iq4_decode_matches_cpu_at_flash_next_geometry() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        run_iq4_moe_decode_case(&ctx, 2_560, 640, 2_560, 2, &[1, 0]);
    }

    #[test]
    fn moe_iq4_decode_addresses_all_512_experts_and_odd_output_tail() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        run_iq4_moe_decode_case(&ctx, 256, 32, 65, 512, &[511, 257, 0]);
    }

    #[test]
    fn moe_iq4_decode_rejects_unsafe_tensor_contracts() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N_IN: usize = 256;
        const N_FFN: usize = 32;
        const N_HIDDEN: usize = 65;
        const N_EXPERT: usize = 2;
        let gate_bytes = synthetic_iq4_xs_bank(N_IN, N_FFN, N_EXPERT, 17);
        let up_bytes = synthetic_iq4_xs_bank(N_IN, N_FFN, N_EXPERT, 31);
        let down_bytes = synthetic_iq4_nl_bank(N_FFN, N_HIDDEN, N_EXPERT, 47);
        let gate = MetalTensor::from_bytes(
            &ctx,
            &gate_bytes,
            vec![N_IN as u64, N_FFN as u64, N_EXPERT as u64],
            GgmlType::IQ4_XS,
        )
        .unwrap();
        let up = MetalTensor::from_bytes(
            &ctx,
            &up_bytes,
            vec![N_IN as u64, N_FFN as u64, N_EXPERT as u64],
            GgmlType::IQ4_XS,
        )
        .unwrap();
        let down = MetalTensor::from_bytes(
            &ctx,
            &down_bytes,
            vec![N_FFN as u64, N_HIDDEN as u64, N_EXPERT as u64],
            GgmlType::IQ4_NL,
        )
        .unwrap();
        let input = MetalTensor::zeros_f32(&ctx, vec![N_IN as u64]).unwrap();
        let indices =
            MetalTensor::from_bytes(&ctx, bytemuck::cast_slice(&[0i32]), vec![1], GgmlType::I32)
                .unwrap();
        let inner = MetalTensor::zeros_f32(&ctx, vec![N_FFN as u64]).unwrap();
        let output = MetalTensor::zeros_f32(&ctx, vec![N_HIDDEN as u64]).unwrap();
        let half_input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&vec![half::f16::ZERO; N_IN]),
            vec![N_IN as u64],
            GgmlType::F16,
        )
        .unwrap();
        let half_indices = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[half::f16::ZERO]),
            vec![1],
            GgmlType::F16,
        )
        .unwrap();

        let command = ctx.queue.commandBuffer().expect("validation command");
        let encoder = KernelEncoder::begin(&command);
        assert!(
            encode_moe_swiglu_iq4_xs_f32(
                &ctx,
                &encoder,
                &gate,
                &up,
                &half_input,
                &indices,
                &inner,
                N_IN,
                N_FFN,
                N_EXPERT,
                1,
            )
            .is_err()
        );
        assert!(
            encode_moe_swiglu_iq4_xs_f32(
                &ctx,
                &encoder,
                &gate,
                &up,
                &input,
                &half_indices,
                &inner,
                N_IN,
                N_FFN,
                N_EXPERT,
                1,
            )
            .is_err()
        );
        let mut read_only_inner = inner.clone();
        read_only_inner.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        assert!(
            encode_moe_swiglu_iq4_xs_f32(
                &ctx,
                &encoder,
                &gate,
                &up,
                &input,
                &indices,
                &read_only_inner,
                N_IN,
                N_FFN,
                N_EXPERT,
                1,
            )
            .is_err()
        );
        let mut short_input = input.clone();
        short_input.offset = 4;
        assert!(
            encode_moe_swiglu_iq4_xs_f32(
                &ctx,
                &encoder,
                &gate,
                &up,
                &short_input,
                &indices,
                &inner,
                N_IN,
                N_FFN,
                N_EXPERT,
                1,
            )
            .is_err()
        );
        let mut misaligned_gate = gate.clone();
        misaligned_gate.offset = 1;
        assert!(
            encode_moe_swiglu_iq4_xs_f32(
                &ctx,
                &encoder,
                &misaligned_gate,
                &up,
                &input,
                &indices,
                &inner,
                N_IN,
                N_FFN,
                N_EXPERT,
                1,
            )
            .is_err()
        );
        let overflow_dimension = u32::MAX as usize - 255;
        assert!(
            encode_moe_swiglu_iq4_xs_f32(
                &ctx,
                &encoder,
                &gate,
                &up,
                &input,
                &indices,
                &inner,
                overflow_dimension,
                overflow_dimension,
                N_EXPERT,
                1,
            )
            .is_err()
        );
        assert!(
            encode_moe_swiglu_iq4_xs_f32(
                &ctx,
                &encoder,
                &gate,
                &up,
                &input,
                &indices,
                &inner,
                N_IN,
                N_FFN,
                N_EXPERT,
                N_EXPERT + 1,
            )
            .is_err()
        );

        let mut read_only_output = output.clone();
        read_only_output.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        assert!(
            encode_moe_down_iq4_nl_f32(
                &ctx,
                &encoder,
                &down,
                &inner,
                &indices,
                &read_only_output,
                N_FFN,
                N_HIDDEN,
                N_EXPERT,
                1,
            )
            .is_err()
        );
        let padded_inner = MetalTensor::zeros_f32(&ctx, vec![(N_FFN + 1) as u64]).unwrap();
        let misaligned_inner = padded_inner.view_subrange(1, vec![N_FFN as u64]);
        assert!(
            encode_moe_down_iq4_nl_f32_fast(
                &ctx,
                &encoder,
                &down,
                &misaligned_inner,
                &indices,
                &output,
                N_FFN,
                N_HIDDEN,
                N_EXPERT,
                1,
            )
            .is_err()
        );
        let mut short_down = down.clone();
        short_down.offset = 2;
        assert!(
            encode_moe_down_iq4_nl_f32(
                &ctx,
                &encoder,
                &short_down,
                &inner,
                &indices,
                &output,
                N_FFN,
                N_HIDDEN,
                N_EXPERT,
                1,
            )
            .is_err()
        );
        encoder.end();
    }

    #[test]
    #[ignore]
    fn moe_mat_vec_iq3_xxs_matches_f32_dequant_fixture() {
        let Some(path) = crate::test_fixtures::A3B_Q3_K_M.path_or_skip() else {
            eprintln!(
                "[moe-iq3-oracle] skipped missing fixture {}",
                crate::test_fixtures::A3B_Q3_K_M.required().display()
            );
            return;
        };
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
            .expect("missing IQ3_XXS MoE gate tensor");
        let n_in = t.shape[0] as usize;
        let n_out = t.shape[1] as usize;
        let n_expert = t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let row_stride = (n_in / 256) * 98;
        let expert_stride = n_out * row_stride;
        let all_bytes = g.slice(t);
        let expert_bytes = &all_bytes[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_gate_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_out as u64],
            dtype: GgmlType::IQ3_XXS,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_bytes.len() as u64,
        };
        let w_f32 = crate::codec::dequant_to_f32(&expert_desc, expert_bytes).expect("dequant");
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.0075)
            .collect();
        let cpu = crate::forward::mat_vec_pub(&w_f32, n_in, n_out, &x);

        let w_t = MetalTensor::from_gguf_tensor(&ctx, t, all_bytes).expect("native weight");
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let expert_i = expert as i32;
        let topk_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[expert_i]),
            vec![1],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let out_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_mat_vec_iq3_xxs_f32(
                &ctx, enc, &w_t, &x_t, &topk_t, &out_t, n_in, n_out, n_expert, 1,
            )
        })
        .expect("gpu iq3 matvec");
        let gpu = read_back_f32(&out_t.buffer, n_out);
        let max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu.iter(),
            cpu.iter(),
        );
        eprintln!("[moe-iq3-oracle] max|delta|={max_abs:.3e}");
        assert!(max_abs < 2e-4, "max|delta|={max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_swiglu_iq3_xxs_matches_f32_dequant_fixture() {
        let Some(path) = crate::test_fixtures::A3B_Q3_K_M.path_or_skip() else {
            eprintln!(
                "[moe-iq3-direct-swiglu-oracle] skipped missing fixture {}",
                crate::test_fixtures::A3B_Q3_K_M.required().display()
            );
            return;
        };
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
            .expect("missing IQ3_XXS MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
            .expect("missing IQ3_XXS MoE up tensor");
        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let row_stride = (n_in / 256) * 98;
        let expert_stride = n_ffn * row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let gate_expert_bytes =
            &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::IQ3_XXS,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 =
            crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, &x);
        let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, &x);
        let mut cpu = vec![0.0f32; n_ffn];
        for i in 0..n_ffn {
            let g = gate[i];
            cpu[i] = (g / (1.0 + (-g).exp())) * up[i];
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let expert_i = expert as i32;
        let topk_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[expert_i]),
            vec![1],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let out_gpu = MetalTensor::zeros_f32(&ctx, vec![n_ffn as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_iq3_xxs_f32(
                &ctx, enc, &gate_gpu, &up_gpu, &x_gpu, &topk_gpu, &out_gpu, n_in, n_ffn, n_expert,
                1,
            )
        })
        .expect("gpu direct iq3 swiglu");
        let gpu = read_back_f32(&out_gpu.buffer, n_ffn);
        let max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu.iter(),
            cpu.iter(),
        );
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq3-direct-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");

        let out_fast_gpu =
            MetalTensor::zeros_f32(&ctx, vec![n_ffn as u64]).expect("fast out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_iq3_xxs_f32_fast(
                &ctx,
                enc,
                &gate_gpu,
                &up_gpu,
                &x_gpu,
                &topk_gpu,
                &out_fast_gpu,
                n_in,
                n_ffn,
                n_expert,
                1,
            )
        })
        .expect("gpu fast direct iq3 swiglu");
        let fast = read_back_f32(&out_fast_gpu.buffer, n_ffn);
        let fast_max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            fast.iter(),
            cpu.iter(),
        );
        let fast_dot: f64 = fast
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let nf: f64 = fast.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let fast_cos = fast_dot / (nf.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq3-fast-swiglu-oracle] cos={fast_cos:.6} max|delta|={fast_max_abs:.3e}");
        assert!(fast_cos > 0.999, "fast cos={fast_cos}");
        assert!(fast_max_abs < 2e-2, "fast max|delta|={fast_max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_grouped_swiglu_iq3_xxs_matches_f32_dequant_fixture() {
        let Some(path) = crate::test_fixtures::A3B_Q3_K_M.path_or_skip() else {
            eprintln!(
                "[moe-iq3-swiglu-oracle] skipped missing fixture {}",
                crate::test_fixtures::A3B_Q3_K_M.required().display()
            );
            return;
        };
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
            .expect("missing IQ3_XXS MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::IQ3_XXS)
            .expect("missing IQ3_XXS MoE up tensor");
        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let n_tokens = 32usize;
        let topk = 1usize;
        let row_stride = (n_in / 256) * 98;
        let expert_stride = n_ffn * row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let gate_expert_bytes =
            &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::IQ3_XXS,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 =
            crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
        let x: Vec<f32> = (0..n_tokens * n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let mut cpu = vec![0.0f32; n_tokens * n_ffn];
        for token in 0..n_tokens {
            let x_tok = &x[token * n_in..(token + 1) * n_in];
            let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, x_tok);
            let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, x_tok);
            for i in 0..n_ffn {
                let g = gate[i];
                cpu[token * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
            }
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_tokens * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let mut counts = vec![0i32; n_expert];
        counts[expert] = n_tokens as i32;
        let mut ids = vec![0i32; n_expert * n_tokens];
        for token in 0..n_tokens {
            ids[expert * n_tokens + token] = token as i32;
        }
        let counts_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&counts),
            vec![n_expert as u64],
            GgmlType::F32,
        )
        .expect("counts tensor");
        let ids_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![(n_expert * n_tokens) as u64],
            GgmlType::F32,
        )
        .expect("ids tensor");
        let out_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_ffn) as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16(
                &ctx,
                enc,
                &gate_gpu,
                &up_gpu,
                &x_gpu,
                &counts_gpu,
                &ids_gpu,
                &out_gpu,
                n_in,
                n_ffn,
                n_expert,
                topk,
                n_tokens,
            )
        })
        .expect("gpu grouped iq3 swiglu");
        let gpu = read_back_f32(&out_gpu.buffer, n_tokens * n_ffn);
        let max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu.iter(),
            cpu.iter(),
        );
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq3-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_mat_vec_iq3_s_matches_f32_dequant_fixture() {
        let Some(path) = crate::test_fixtures::A3B_IQ4_XS.path_or_skip() else {
            eprintln!(
                "[moe-iq3s-oracle] skipped missing fixture {}",
                crate::test_fixtures::A3B_IQ4_XS.required().display()
            );
            return;
        };
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_S)
            .expect("missing IQ3_S MoE gate tensor");
        let n_in = t.shape[0] as usize;
        let n_out = t.shape[1] as usize;
        let n_expert = t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let row_stride = (n_in / 256) * 110;
        let expert_stride = n_out * row_stride;
        let all_bytes = g.slice(t);
        let expert_bytes = &all_bytes[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_gate_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_out as u64],
            dtype: GgmlType::IQ3_S,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_bytes.len() as u64,
        };
        let w_f32 = crate::codec::dequant_to_f32(&expert_desc, expert_bytes).expect("dequant");
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.0075)
            .collect();
        let cpu = crate::forward::mat_vec_pub(&w_f32, n_in, n_out, &x);

        let w_t = MetalTensor::from_gguf_tensor(&ctx, t, all_bytes).expect("native weight");
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let expert_i = expert as i32;
        let topk_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[expert_i]),
            vec![1],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let out_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_mat_vec_iq3_s_f32(
                &ctx, enc, &w_t, &x_t, &topk_t, &out_t, n_in, n_out, n_expert, 1,
            )
        })
        .expect("gpu iq3s matvec");
        let gpu = read_back_f32(&out_t.buffer, n_out);
        let max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu.iter(),
            cpu.iter(),
        );
        eprintln!("[moe-iq3s-oracle] max|delta|={max_abs:.3e}");
        assert!(max_abs < 2e-4, "max|delta|={max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_swiglu_iq3_s_matches_f32_dequant_fixture() {
        let Some(path) = crate::test_fixtures::A3B_IQ4_XS.path_or_skip() else {
            eprintln!(
                "[moe-iq3s-direct-swiglu-oracle] skipped missing fixture {}",
                crate::test_fixtures::A3B_IQ4_XS.required().display()
            );
            return;
        };
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_S)
            .expect("missing IQ3_S MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::IQ3_S)
            .expect("missing IQ3_S MoE up tensor");
        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let row_stride = (n_in / 256) * 110;
        let expert_stride = n_ffn * row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let gate_expert_bytes =
            &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::IQ3_S,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 =
            crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, &x);
        let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, &x);
        let mut cpu = vec![0.0f32; n_ffn];
        for i in 0..n_ffn {
            let g = gate[i];
            cpu[i] = (g / (1.0 + (-g).exp())) * up[i];
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let expert_i = expert as i32;
        let topk_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[expert_i]),
            vec![1],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let out_gpu = MetalTensor::zeros_f32(&ctx, vec![n_ffn as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_iq3_s_f32(
                &ctx, enc, &gate_gpu, &up_gpu, &x_gpu, &topk_gpu, &out_gpu, n_in, n_ffn, n_expert,
                1,
            )
        })
        .expect("gpu direct iq3s swiglu");
        let gpu = read_back_f32(&out_gpu.buffer, n_ffn);
        let max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu.iter(),
            cpu.iter(),
        );
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq3s-direct-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");

        let out_fast_gpu =
            MetalTensor::zeros_f32(&ctx, vec![n_ffn as u64]).expect("fast out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_iq3_s_f32_fast(
                &ctx,
                enc,
                &gate_gpu,
                &up_gpu,
                &x_gpu,
                &topk_gpu,
                &out_fast_gpu,
                n_in,
                n_ffn,
                n_expert,
                1,
            )
        })
        .expect("gpu fast direct iq3s swiglu");
        let fast = read_back_f32(&out_fast_gpu.buffer, n_ffn);
        let fast_max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            fast.iter(),
            cpu.iter(),
        );
        let fast_dot: f64 = fast
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let nf: f64 = fast.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let fast_cos = fast_dot / (nf.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq3s-fast-swiglu-oracle] cos={fast_cos:.6} max|delta|={fast_max_abs:.3e}");
        assert!(fast_cos > 0.999, "fast cos={fast_cos}");
        assert!(fast_max_abs < 2e-2, "fast max|delta|={fast_max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_grouped_swiglu_iq3_s_matches_f32_dequant_fixture() {
        let Some(path) = crate::test_fixtures::A3B_IQ4_XS.path_or_skip() else {
            eprintln!(
                "[moe-iq3s-swiglu-oracle] skipped missing fixture {}",
                crate::test_fixtures::A3B_IQ4_XS.required().display()
            );
            return;
        };
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::IQ3_S)
            .expect("missing IQ3_S MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::IQ3_S)
            .expect("missing IQ3_S MoE up tensor");
        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let n_tokens = 32usize;
        let topk = 1usize;
        let row_stride = (n_in / 256) * 110;
        let expert_stride = n_ffn * row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let gate_expert_bytes =
            &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::IQ3_S,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 =
            crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
        let x: Vec<f32> = (0..n_tokens * n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let mut cpu = vec![0.0f32; n_tokens * n_ffn];
        for token in 0..n_tokens {
            let x_tok = &x[token * n_in..(token + 1) * n_in];
            let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, x_tok);
            let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, x_tok);
            for i in 0..n_ffn {
                let g = gate[i];
                cpu[token * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
            }
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_tokens * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let mut counts = vec![0i32; n_expert];
        counts[expert] = n_tokens as i32;
        let mut ids = vec![0i32; n_expert * n_tokens];
        for token in 0..n_tokens {
            ids[expert * n_tokens + token] = token as i32;
        }
        let counts_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&counts),
            vec![n_expert as u64],
            GgmlType::F32,
        )
        .expect("counts tensor");
        let ids_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![(n_expert * n_tokens) as u64],
            GgmlType::F32,
        )
        .expect("ids tensor");
        let out_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_ffn) as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_iq3_s_f32_grouped_slots_n16(
                &ctx,
                enc,
                &gate_gpu,
                &up_gpu,
                &x_gpu,
                &counts_gpu,
                &ids_gpu,
                &out_gpu,
                n_in,
                n_ffn,
                n_expert,
                topk,
                n_tokens,
            )
        })
        .expect("gpu grouped iq3s swiglu");
        let gpu = read_back_f32(&out_gpu.buffer, n_tokens * n_ffn);
        let max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu.iter(),
            cpu.iter(),
        );
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq3s-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_grouped_down_iq4_xs_matches_f32_dequant_fixture() {
        let Some(path) = crate::test_fixtures::A3B_IQ4_XS.path_or_skip() else {
            eprintln!(
                "[moe-iq4xs-down-oracle] skipped missing fixture {}",
                crate::test_fixtures::A3B_IQ4_XS.required().display()
            );
            return;
        };
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let down_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_down_exps.weight" && t.dtype == GgmlType::IQ4_XS)
            .expect("missing IQ4_XS MoE down tensor");
        let n_in = down_t.shape[0] as usize;
        let n_out = down_t.shape[1] as usize;
        let n_expert = down_t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let n_tokens = 32usize;
        let row_stride = (n_in / 256) * 136;
        let expert_stride = n_out * row_stride;
        let down_bytes_all = g.slice(down_t);
        let down_expert_bytes =
            &down_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_down_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_out as u64],
            dtype: GgmlType::IQ4_XS,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let down_f32 =
            crate::codec::dequant_to_f32(&expert_desc, down_expert_bytes).expect("down dequant");
        let x: Vec<f32> = (0..n_tokens * n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let mut cpu = vec![0.0f32; n_tokens * n_out];
        for token in 0..n_tokens {
            let x_tok = &x[token * n_in..(token + 1) * n_in];
            let y = crate::forward::mat_vec_pub(&down_f32, n_in, n_out, x_tok);
            cpu[token * n_out..(token + 1) * n_out].copy_from_slice(&y);
        }

        let down_gpu = MetalTensor::from_gguf_tensor(&ctx, down_t, down_bytes_all).expect("down");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_tokens * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let mut counts = vec![0i32; n_expert];
        counts[expert] = n_tokens as i32;
        let mut ids = vec![0i32; n_expert * n_tokens];
        for token in 0..n_tokens {
            ids[expert * n_tokens + token] = token as i32;
        }
        let counts_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&counts),
            vec![n_expert as u64],
            GgmlType::F32,
        )
        .expect("counts tensor");
        let ids_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![(n_expert * n_tokens) as u64],
            GgmlType::F32,
        )
        .expect("ids tensor");
        let out_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_out) as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_down_iq4_xs_f32_grouped_slots(
                &ctx,
                enc,
                &down_gpu,
                &x_gpu,
                &counts_gpu,
                &ids_gpu,
                &out_gpu,
                n_in,
                n_out,
                n_expert,
                n_tokens,
            )
        })
        .expect("gpu grouped iq4xs down");
        let gpu = read_back_f32(&out_gpu.buffer, n_tokens * n_out);
        let max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu.iter(),
            cpu.iter(),
        );
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-iq4xs-down-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");

        let topk_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[expert as i32]),
            vec![1],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let x_one = x_gpu.view_subrange(0, vec![n_in as u64]);
        let out_fast_gpu =
            MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("fast out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_down_iq4_xs_f32_fast(
                &ctx,
                enc,
                &down_gpu,
                &x_one,
                &topk_gpu,
                &out_fast_gpu,
                n_in,
                n_out,
                n_expert,
                1,
            )
        })
        .expect("gpu fast iq4xs down");
        let fast = read_back_f32(&out_fast_gpu.buffer, n_out);
        let fast_max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            fast.iter(),
            cpu[..n_out].iter(),
        );
        let fast_dot: f64 = fast
            .iter()
            .zip(cpu[..n_out].iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let nf: f64 = fast.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc_fast: f64 = cpu[..n_out].iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let fast_cos = fast_dot / (nf.sqrt() * nc_fast.sqrt()).max(1e-12);
        eprintln!("[moe-iq4xs-fast-down-oracle] cos={fast_cos:.6} max|delta|={fast_max_abs:.3e}");
        assert!(fast_cos > 0.999, "fast cos={fast_cos}");
        assert!(fast_max_abs < 2e-2, "fast max|delta|={fast_max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_swiglu_q6_k_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_Q6_MODEL").unwrap_or_else(|_| {
            crate::test_fixtures::required_filename("Qwen3.5-35B-A3B-Q6_K.gguf").into()
        });
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-q6-direct-swiglu-oracle] skipped missing fixture {path}");
            return;
        }
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::Q6_K)
            .expect("missing Q6_K MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::Q6_K)
            .expect("missing Q6_K MoE up tensor");
        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let topk = 2usize;
        let experts = [7usize.min(n_expert - 1), n_expert - 1];
        let row_stride = (n_in / 256) * 210;
        let expert_stride = n_ffn * row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let mut cpu = vec![0.0f32; topk * n_ffn];
        for (slot, expert) in experts.iter().copied().enumerate() {
            let gate_expert_bytes =
                &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
            let up_expert_bytes =
                &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
            let expert_desc = crate::tensor::TensorDesc {
                name: "blk.0.ffn_exps.weight.expert_oracle".into(),
                shape: vec![n_in as u64, n_ffn as u64],
                dtype: GgmlType::Q6_K,
                shard_idx: 0,
                data_offset: 0,
                n_bytes: expert_stride as u64,
            };
            let gate_f32 = crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes)
                .expect("gate dequant");
            let up_f32 =
                crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
            let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, &x);
            let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, &x);
            for i in 0..n_ffn {
                let g = gate[i];
                cpu[slot * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
            }
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let topk_i32 = [experts[0] as i32, experts[1] as i32];
        let topk_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&topk_i32),
            vec![topk as u64],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let out_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(topk * n_ffn) as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_q6_K_f32(
                &ctx, enc, &gate_gpu, &up_gpu, &x_gpu, &topk_gpu, &out_gpu, n_in, n_ffn, n_expert,
                topk,
            )
        })
        .expect("gpu direct q6 swiglu");
        let gpu = read_back_f32(&out_gpu.buffer, topk * n_ffn);
        let max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu.iter(),
            cpu.iter(),
        );
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-q6-direct-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_grouped_swiglu_q6_k_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_Q6_MODEL").unwrap_or_else(|_| {
            crate::test_fixtures::required_filename("Qwen3.5-35B-A3B-Q6_K.gguf").into()
        });
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-q6-swiglu-oracle] skipped missing fixture {path}");
            return;
        }
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::Q6_K)
            .expect("missing Q6_K MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::Q6_K)
            .expect("missing Q6_K MoE up tensor");
        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let expert = 7usize.min(n_expert - 1);
        let n_tokens = 16usize;
        let topk = 1usize;
        let row_stride = (n_in / 256) * 210;
        let expert_stride = n_ffn * row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let gate_expert_bytes =
            &gate_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let up_expert_bytes = &up_bytes_all[expert * expert_stride..(expert + 1) * expert_stride];
        let expert_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::Q6_K,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&expert_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 =
            crate::codec::dequant_to_f32(&expert_desc, up_expert_bytes).expect("up dequant");
        let x: Vec<f32> = (0..n_tokens * n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let mut cpu = vec![0.0f32; n_tokens * n_ffn];
        for token in 0..n_tokens {
            let x_tok = &x[token * n_in..(token + 1) * n_in];
            let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, x_tok);
            let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, x_tok);
            for i in 0..n_ffn {
                let g = gate[i];
                cpu[token * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
            }
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_tokens * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let mut counts = vec![0i32; n_expert];
        counts[expert] = n_tokens as i32;
        let mut ids = vec![0i32; n_expert * n_tokens];
        for token in 0..n_tokens {
            ids[expert * n_tokens + token] = token as i32;
        }
        let counts_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&counts),
            vec![n_expert as u64],
            GgmlType::F32,
        )
        .expect("counts tensor");
        let ids_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![(n_expert * n_tokens) as u64],
            GgmlType::F32,
        )
        .expect("ids tensor");
        let out_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_ffn) as u64]).expect("out tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_q6_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &gate_gpu,
                &up_gpu,
                &x_gpu,
                &counts_gpu,
                &ids_gpu,
                &out_gpu,
                n_in,
                n_ffn,
                n_expert,
                topk,
                n_tokens,
            )
        })
        .expect("gpu grouped q6 swiglu");
        let gpu = read_back_f32(&out_gpu.buffer, n_tokens * n_ffn);
        let max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu.iter(),
            cpu.iter(),
        );
        let dot: f64 = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc: f64 = cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-12);
        eprintln!("[moe-q6-swiglu-oracle] cos={cos:.6} max|delta|={max_abs:.3e}");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_abs < 2e-2, "max|delta|={max_abs}");
    }

    #[test]
    #[ignore]
    fn moe_q8_0_swiglu_down_weighted_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_Q8_MODEL").unwrap_or_else(|_| {
            crate::test_fixtures::required_filename("Qwen3.5-35B-A3B-Q8_0.gguf").into()
        });
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-q8-direct-oracle] skipped missing fixture {path}");
            return;
        }
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::Q8_0)
            .expect("missing Q8_0 MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::Q8_0)
            .expect("missing Q8_0 MoE up tensor");
        let down_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_down_exps.weight" && t.dtype == GgmlType::Q8_0)
            .expect("missing Q8_0 MoE down tensor");

        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let h = down_t.shape[1] as usize;
        let experts = [7usize.min(n_expert - 1), n_expert - 1];
        let topk = experts.len();
        let top_w = [0.35f32, 0.65f32];
        let gate_row_stride = (n_in / 32) * 34;
        let gate_expert_stride = n_ffn * gate_row_stride;
        let down_row_stride = (n_ffn / 32) * 34;
        let down_expert_stride = h * down_row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let down_bytes_all = g.slice(down_t);
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let gate_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_gate_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: gate_expert_stride as u64,
        };
        let down_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_down_exps.weight.expert_oracle".into(),
            shape: vec![n_ffn as u64, h as u64],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: down_expert_stride as u64,
        };
        let mut cpu_inner = vec![0.0f32; topk * n_ffn];
        let mut cpu_down = vec![0.0f32; h];
        for (slot, expert) in experts.iter().copied().enumerate() {
            let gate_expert_bytes =
                &gate_bytes_all[expert * gate_expert_stride..(expert + 1) * gate_expert_stride];
            let up_expert_bytes =
                &up_bytes_all[expert * gate_expert_stride..(expert + 1) * gate_expert_stride];
            let down_expert_bytes =
                &down_bytes_all[expert * down_expert_stride..(expert + 1) * down_expert_stride];
            let gate_f32 =
                crate::codec::dequant_to_f32(&gate_desc, gate_expert_bytes).expect("gate dequant");
            let up_f32 =
                crate::codec::dequant_to_f32(&gate_desc, up_expert_bytes).expect("up dequant");
            let down_f32 =
                crate::codec::dequant_to_f32(&down_desc, down_expert_bytes).expect("down dequant");
            let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, &x);
            let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, &x);
            for i in 0..n_ffn {
                let g = gate[i];
                cpu_inner[slot * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            let down = crate::forward::mat_vec_pub(
                &down_f32,
                n_ffn,
                h,
                &cpu_inner[slot * n_ffn..(slot + 1) * n_ffn],
            );
            for i in 0..h {
                cpu_down[i] += top_w[slot] * down[i];
            }
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let down_gpu = MetalTensor::from_gguf_tensor(&ctx, down_t, down_bytes_all).expect("down");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let topk_i32 = [experts[0] as i32, experts[1] as i32];
        let topk_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&topk_i32),
            vec![topk as u64],
            GgmlType::F32,
        )
        .expect("topk tensor");
        let topw_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&top_w),
            vec![topk as u64],
            GgmlType::F32,
        )
        .expect("top weights tensor");
        let inner_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(topk * n_ffn) as u64]).expect("inner tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_q8_0_f32(
                &ctx, enc, &gate_gpu, &up_gpu, &x_gpu, &topk_gpu, &inner_gpu, n_in, n_ffn,
                n_expert, topk,
            )
        })
        .expect("gpu direct q8 swiglu");
        let gpu_inner = read_back_f32(&inner_gpu.buffer, topk * n_ffn);
        let inner_max = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu_inner.iter(),
            cpu_inner.iter(),
        );
        let inner_dot: f64 = gpu_inner
            .iter()
            .zip(cpu_inner.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let inner_ng: f64 = gpu_inner.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let inner_nc: f64 = cpu_inner.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let inner_cos = inner_dot / (inner_ng.sqrt() * inner_nc.sqrt()).max(1e-12);
        eprintln!("[moe-q8-direct-swiglu-oracle] cos={inner_cos:.6} max|delta|={inner_max:.3e}");
        assert!(inner_cos > 0.999, "inner cos={inner_cos}");
        assert!(inner_max < 2e-2, "inner max|delta|={inner_max}");

        let down_gpu_out = MetalTensor::zeros_f32(&ctx, vec![h as u64]).expect("down out");
        one_shot(&ctx, |enc| {
            encode_moe_down_weighted_sum_q8_0_f32(
                &ctx,
                enc,
                &down_gpu,
                &inner_gpu,
                &topk_gpu,
                &topw_gpu,
                &down_gpu_out,
                n_ffn,
                h,
                n_expert,
                topk,
            )
        })
        .expect("gpu direct q8 down weighted sum");
        let gpu_down = read_back_f32(&down_gpu_out.buffer, h);
        let down_max = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu_down.iter(),
            cpu_down.iter(),
        );
        let down_dot: f64 = gpu_down
            .iter()
            .zip(cpu_down.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let down_ng: f64 = gpu_down.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let down_nc: f64 = cpu_down.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let down_cos = down_dot / (down_ng.sqrt() * down_nc.sqrt()).max(1e-12);
        eprintln!("[moe-q8-direct-down-oracle] cos={down_cos:.6} max|delta|={down_max:.3e}");
        assert!(down_cos > 0.999, "down cos={down_cos}");
        assert!(down_max < 2e-2, "down max|delta|={down_max}");
    }

    #[test]
    #[ignore]
    fn moe_grouped_q8_0_swiglu_down_matches_f32_dequant_fixture() {
        let path = std::env::var("QWEN_A3B_Q8_MODEL").unwrap_or_else(|_| {
            crate::test_fixtures::required_filename("Qwen3.5-35B-A3B-Q8_0.gguf").into()
        });
        if !std::path::Path::new(&path).exists() {
            eprintln!("[moe-q8-oracle] skipped missing fixture {path}");
            return;
        }
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let g = crate::gguf::GgufFile::open(&path).expect("open fixture");
        let gate_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate_exps.weight" && t.dtype == GgmlType::Q8_0)
            .expect("missing Q8_0 MoE gate tensor");
        let up_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight" && t.dtype == GgmlType::Q8_0)
            .expect("missing Q8_0 MoE up tensor");
        let down_t = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_down_exps.weight" && t.dtype == GgmlType::Q8_0)
            .expect("missing Q8_0 MoE down tensor");

        let n_in = gate_t.shape[0] as usize;
        let n_ffn = gate_t.shape[1] as usize;
        let n_expert = gate_t.shape[2] as usize;
        let h = down_t.shape[1] as usize;
        let expert = 7usize.min(n_expert - 1);
        let n_tokens = 16usize;
        let topk = 1usize;
        let gate_row_stride = (n_in / 32) * 34;
        let gate_expert_stride = n_ffn * gate_row_stride;
        let down_row_stride = (n_ffn / 32) * 34;
        let down_expert_stride = h * down_row_stride;
        let gate_bytes_all = g.slice(gate_t);
        let up_bytes_all = g.slice(up_t);
        let down_bytes_all = g.slice(down_t);
        let gate_expert_bytes =
            &gate_bytes_all[expert * gate_expert_stride..(expert + 1) * gate_expert_stride];
        let up_expert_bytes =
            &up_bytes_all[expert * gate_expert_stride..(expert + 1) * gate_expert_stride];
        let down_expert_bytes =
            &down_bytes_all[expert * down_expert_stride..(expert + 1) * down_expert_stride];
        let gate_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_gate_exps.weight.expert_oracle".into(),
            shape: vec![n_in as u64, n_ffn as u64],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: gate_expert_stride as u64,
        };
        let down_desc = crate::tensor::TensorDesc {
            name: "blk.0.ffn_down_exps.weight.expert_oracle".into(),
            shape: vec![n_ffn as u64, h as u64],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: down_expert_stride as u64,
        };
        let gate_f32 =
            crate::codec::dequant_to_f32(&gate_desc, gate_expert_bytes).expect("gate dequant");
        let up_f32 = crate::codec::dequant_to_f32(&gate_desc, up_expert_bytes).expect("up dequant");
        let down_f32 =
            crate::codec::dequant_to_f32(&down_desc, down_expert_bytes).expect("down dequant");
        let x: Vec<f32> = (0..n_tokens * n_in)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.00625)
            .collect();
        let mut cpu_inner = vec![0.0f32; n_tokens * n_ffn];
        let mut cpu_down = vec![0.0f32; n_tokens * h];
        for token in 0..n_tokens {
            let x_tok = &x[token * n_in..(token + 1) * n_in];
            let gate = crate::forward::mat_vec_pub(&gate_f32, n_in, n_ffn, x_tok);
            let up = crate::forward::mat_vec_pub(&up_f32, n_in, n_ffn, x_tok);
            for i in 0..n_ffn {
                let g = gate[i];
                cpu_inner[token * n_ffn + i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            let down = crate::forward::mat_vec_pub(
                &down_f32,
                n_ffn,
                h,
                &cpu_inner[token * n_ffn..(token + 1) * n_ffn],
            );
            cpu_down[token * h..(token + 1) * h].copy_from_slice(&down);
        }

        let gate_gpu = MetalTensor::from_gguf_tensor(&ctx, gate_t, gate_bytes_all).expect("gate");
        let up_gpu = MetalTensor::from_gguf_tensor(&ctx, up_t, up_bytes_all).expect("up");
        let down_gpu = MetalTensor::from_gguf_tensor(&ctx, down_t, down_bytes_all).expect("down");
        let x_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(n_tokens * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x tensor");
        let mut counts = vec![0i32; n_expert];
        counts[expert] = n_tokens as i32;
        let mut ids = vec![0i32; n_expert * n_tokens];
        for token in 0..n_tokens {
            ids[expert * n_tokens + token] = token as i32;
        }
        let counts_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&counts),
            vec![n_expert as u64],
            GgmlType::F32,
        )
        .expect("counts tensor");
        let ids_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![(n_expert * n_tokens) as u64],
            GgmlType::F32,
        )
        .expect("ids tensor");
        let inner_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_ffn) as u64]).expect("inner tensor");
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_q8_0_f32_grouped_slots_n16(
                &ctx,
                enc,
                &gate_gpu,
                &up_gpu,
                &x_gpu,
                &counts_gpu,
                &ids_gpu,
                &inner_gpu,
                n_in,
                n_ffn,
                n_expert,
                topk,
                n_tokens,
            )
        })
        .expect("gpu grouped q8 swiglu");
        let gpu_inner = read_back_f32(&inner_gpu.buffer, n_tokens * n_ffn);
        let dot_inner: f64 = gpu_inner
            .iter()
            .zip(cpu_inner.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng_inner: f64 = gpu_inner.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc_inner: f64 = cpu_inner.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos_inner = dot_inner / (ng_inner.sqrt() * nc_inner.sqrt()).max(1e-12);
        let max_inner = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu_inner.iter(),
            cpu_inner.iter(),
        );
        eprintln!("[moe-q8-swiglu-oracle] cos={cos_inner:.6} max|delta|={max_inner:.3e}");
        assert!(cos_inner > 0.999, "inner cos={cos_inner}");
        assert!(max_inner < 2e-2, "inner max|delta|={max_inner}");

        let cpu_inner_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&cpu_inner),
            vec![(n_tokens * n_ffn) as u64],
            GgmlType::F32,
        )
        .expect("cpu inner tensor");
        let down_out_gpu =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * h) as u64]).expect("down out");
        one_shot(&ctx, |enc| {
            encode_moe_down_q8_0_f32_grouped_slots(
                &ctx,
                enc,
                &down_gpu,
                &cpu_inner_gpu,
                &counts_gpu,
                &ids_gpu,
                &down_out_gpu,
                n_ffn,
                h,
                n_expert,
                n_tokens,
            )
        })
        .expect("gpu grouped q8 down");
        let gpu_down = read_back_f32(&down_out_gpu.buffer, n_tokens * h);
        let dot_down: f64 = gpu_down
            .iter()
            .zip(cpu_down.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng_down: f64 = gpu_down.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let nc_down: f64 = cpu_down.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let cos_down = dot_down / (ng_down.sqrt() * nc_down.sqrt()).max(1e-12);
        let max_down = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            gpu_down.iter(),
            cpu_down.iter(),
        );
        eprintln!("[moe-q8-down-oracle] cos={cos_down:.6} max|delta|={max_down:.3e}");
        assert!(cos_down > 0.999, "down cos={cos_down}");
        assert!(max_down < 2e-2, "down max|delta|={max_down}");
    }
}
