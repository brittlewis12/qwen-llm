//! Metal 4 tensor-ops twins of the generic grouped tiles
//! (`kernels/tensor/moe_grouped_slots_tensor.metal`).
//!
//! Same host contract as the generic `mm` kernel (128 threads, 8 KiB,
//! 32-slot x 64-row tiles, buffers 0..5, clamp at 6): only the library and
//! the pipeline name differ, so a caller that already drives the generic
//! tile can route to this one by name. The tier exists only where
//! [`MetalContext::tensor_ops_status`] is `Available` (M5-class GPUs on
//! macOS 26+); nothing here falls back silently, callers choose the route.
//!
//! Numerics: half operands staged exactly as the generic tile stages them,
//! F32 accumulation in a cooperative tensor; the reduction order differs,
//! so results match the generic tile closely but not bitwise.

use super::*;

/// Kernel roles of the tensor tile. `UpClamped` is the clamped up pass the
/// generic family instantiates as `kernel_moe_up_clamped_*_clamped`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeGroupedTensorRole {
    /// `dst[slot] = W_e * src[slot / b_div]` (down, or the gate pass).
    Down,
    /// `dst[slot] = silu(dst[slot]) * (W_e * x[slot / topk])`.
    UpSiluMul,
    /// `dst[slot] = silu(min(dst, c)) * clamp(W_e * x[slot / topk], -c, c)`.
    UpClamped,
}

impl MoeGroupedTensorRole {
    pub const ALL: [Self; 3] = [Self::Down, Self::UpSiluMul, Self::UpClamped];

    fn prefix(self) -> &'static str {
        match self {
            Self::Down => "kernel_moe_down_",
            Self::UpSiluMul => "kernel_moe_up_silu_mul_",
            Self::UpClamped => "kernel_moe_up_clamped_",
        }
    }
}

/// `kernel_moe_<role>_<suffix>_f32_grouped_slots_tensor` for every dtype
/// with a generic layout (the tensor file instantiates the same list; a
/// CPU test checks that).
pub fn moe_grouped_tensor_pipeline_name(
    role: MoeGroupedTensorRole,
    dtype: GgmlType,
) -> Option<String> {
    let l = moe_grouped_generic_layout(dtype)?;
    Some(format!(
        "{}{}_f32_grouped_slots_tensor",
        role.prefix(),
        l.suffix
    ))
}

/// True when the tensor tile can run on this context.
pub fn moe_grouped_tensor_available(ctx: &MetalContext) -> bool {
    ctx.tensor_ops_status().is_available()
}

crate::env_flag!(
    /// Routes grouped expert prefill tiles through the Metal 4 tensor-ops tile where the tier is available (M5-class GPU, macOS 26+); disabling keeps the simdgroup-MMA generic tile. Same operands and F32 sums, reduction order differs. Rollback lever and same-binary A/B switch.
    default_on expert_tensor_tile_enabled,
    "QWEN_EXPERT_TENSOR_TILE"
);

/// Route switch: the tensor tile replaces the generic half tile wherever
/// the tier is available unless `QWEN_EXPERT_TENSOR_TILE` is off.
pub fn moe_grouped_tensor_selected(ctx: &MetalContext) -> bool {
    expert_tensor_tile_enabled() && moe_grouped_tensor_available(ctx)
}

const TENSOR_THREADGROUP_BYTES: usize = 8192;
const TENSOR_THREADGROUP_BYTES_WIDE: usize = 16384;
const TENSOR_THREADGROUP_BYTES_COMBINED: usize = 32768;

#[allow(clippy::too_many_arguments)]
fn dispatch_tensor_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    name: &str,
    args: &GenericMmArgs,
    weight: &MetalTensor,
    src: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
    clamp: Option<f32>,
) -> Result<(), MetalError> {
    let pso = ctx.tensor_pipeline(name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, args);
    enc.set_tensor(1, weight);
    enc.set_tensor(2, src);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    if let Some(clamp) = clamp {
        enc.set_bytes(6, &clamp);
    }
    enc.set_threadgroup_memory(0, TENSOR_THREADGROUP_BYTES);
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

/// Tensor-tile down projection; contract of
/// [`encode_moe_down_f32_grouped_slots_generic`].
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_down_f32_grouped_slots_tensor(
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
    const KERNEL: &str = "moe_down_grouped_slots_tensor";
    let args = generic_mm_args(
        KERNEL,
        weight,
        inner,
        counts,
        ids,
        out,
        n_in,
        n_out,
        n_expert,
        n_tokens,
        1,
        0,
        i32::MAX as u32,
    )?;
    let name = moe_grouped_tensor_pipeline_name(MoeGroupedTensorRole::Down, weight.dtype)
        .expect("layout lookup succeeded in generic_mm_args");
    dispatch_tensor_mm(
        ctx, enc, &name, &args, weight, inner, counts, ids, out, n_out, n_expert, n_tokens, None,
    )
}

/// Tensor-tile gate/up: gate into `inner` (Down with `b_div = topk`), then
/// the up pass in place, with the clamped SwiGLU epilogue when `clamp` is
/// given (contract of [`encode_moe_swiglu_clamped_f32_grouped_slots_generic`])
/// and plain `silu(g) * u` otherwise. Two dependent dispatches, so the
/// encoder must be serial.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_swiglu_f32_grouped_slots_tensor(
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
    clamp: Option<f32>,
) -> Result<(), MetalError> {
    const KERNEL: &str = "moe_swiglu_grouped_slots_tensor";
    if let Some(clamp) = clamp
        && (!clamp.is_finite() || clamp <= 0.0)
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("clamp must be finite and positive, got {clamp}"),
        });
    }
    crate::metal::checks::require_serial(KERNEL, enc)?;
    let gate_args = generic_mm_args(
        KERNEL,
        w_gate,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        n_tokens,
        topk,
        0,
        i32::MAX as u32,
    )?;
    let up_args = generic_mm_args(
        KERNEL,
        w_up,
        x_pack,
        counts,
        ids,
        inner,
        n_hidden,
        n_ffn,
        n_expert,
        n_tokens,
        topk,
        0,
        i32::MAX as u32,
    )?;
    let gate = moe_grouped_tensor_pipeline_name(MoeGroupedTensorRole::Down, w_gate.dtype)
        .expect("layout lookup succeeded in generic_mm_args");
    let up_role = if clamp.is_some() {
        MoeGroupedTensorRole::UpClamped
    } else {
        MoeGroupedTensorRole::UpSiluMul
    };
    let up = moe_grouped_tensor_pipeline_name(up_role, w_up.dtype)
        .expect("layout lookup succeeded in generic_mm_args");
    // Resolve both pipelines before the first dispatch so a missing up
    // instantiation cannot leave the gate projection half applied.
    ctx.tensor_pipeline(&up)?;
    dispatch_tensor_mm(
        ctx, enc, &gate, &gate_args, w_gate, x_pack, counts, ids, inner, n_ffn, n_expert, n_tokens,
        None,
    )?;
    dispatch_tensor_mm(
        ctx, enc, &up, &up_args, w_up, x_pack, counts, ids, inner, n_ffn, n_expert, n_tokens, clamp,
    )
}

/// `kernel_mat_mat_<suffix>_f32_tensor`: the dense form of the tensor tile
/// (one weight matrix, slot == row, no routing buffers).
pub fn mat_mat_tensor_pipeline_name(dtype: GgmlType) -> Option<String> {
    let l = moe_grouped_generic_layout(dtype)?;
    Some(format!("kernel_mat_mat_{}_f32_tensor", l.suffix))
}

/// 64-slot dense form. Instanced only for Q8_0, the dtype of the V4.1
/// attention/output/shared-expert projections.
pub fn mat_mat_tensor_wide_pipeline_name(dtype: GgmlType) -> Option<String> {
    (dtype == GgmlType::Q8_0).then(|| "kernel_mat_mat_q8_0_f32_tensor_wide".to_string())
}

/// 64-slot x 128-row dense form; Q8_0 only.
pub fn mat_mat_tensor_combined_pipeline_name(dtype: GgmlType) -> Option<String> {
    (dtype == GgmlType::Q8_0).then(|| "kernel_mat_mat_q8_0_f32_tensor_combined".to_string())
}

/// Dense `y[row] = W x[row]` on the tensor tile with the `mat_mat`
/// contract: `weight` is `[n_in, n_out]` in any dtype with a generic
/// layout, `x` is `[n_in, n_query]` F32 and `y` is `[n_out, n_query]` F32
/// (row-major rows of `n_in` / `n_out`), `n_in` a multiple of 32 and of
/// the dtype's block. Half operands, F32 sums; output rows past `n_query`
/// are untouched. Callers gate on [`moe_grouped_tensor_selected`].
///
/// The 64-slot Q8_0 form is the default from 128 query rows (below that
/// the padded slot group loses; measured 64/96 rows are 12-14% slower).
/// `QWEN_V41_DENSE_NARROW=1` selects the 32-slot form for same-binary
/// A/B measurement. The arithmetic per output row is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_tensor_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    let combined = n_query >= 128
        && mat_mat_tensor_combined_pipeline_name(weight.dtype).is_some()
        && std::env::var_os("QWEN_V41_DENSE_COMBINED").is_some();
    let wide = !combined
        && n_query >= 128
        && mat_mat_tensor_wide_pipeline_name(weight.dtype).is_some()
        && std::env::var_os("QWEN_V41_DENSE_NARROW").is_none();
    encode_mat_mat_tensor_f32_impl(ctx, enc, weight, x, y, n_in, n_out, n_query, wide, combined)
}

/// Force the 64-slot dense form; test and probe entry point.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_tensor_f32_wide(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_tensor_f32_impl(ctx, enc, weight, x, y, n_in, n_out, n_query, true, false)
}

/// Force the 64-slot x 128-row dense form; test and probe entry point.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_tensor_f32_combined(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_tensor_f32_impl(ctx, enc, weight, x, y, n_in, n_out, n_query, false, true)
}

/// Force the 32-slot dense form; test and probe entry point.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_tensor_f32_narrow(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_tensor_f32_impl(ctx, enc, weight, x, y, n_in, n_out, n_query, false, false)
}

#[allow(clippy::too_many_arguments)]
fn encode_mat_mat_tensor_f32_impl(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    wide: bool,
    combined: bool,
) -> Result<(), MetalError> {
    const KERNEL: &str = "mat_mat_tensor";
    let l = moe_grouped_generic_layout(weight.dtype).ok_or_else(|| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("no tensor tile layout for {:?}", weight.dtype),
    })?;
    let bad = |detail: String| MetalError::BadShape {
        kernel: KERNEL,
        detail,
    };
    if n_in == 0 || n_out == 0 || n_query == 0 {
        return Err(bad(format!(
            "degenerate shape n_in={n_in} n_out={n_out} n_query={n_query}"
        )));
    }
    if !n_in.is_multiple_of(32) || !n_in.is_multiple_of(l.block_elems) {
        return Err(bad(format!(
            "n_in={n_in} must be a multiple of 32 and of the {:?} block ({})",
            weight.dtype, l.block_elems
        )));
    }
    let nb01 = (n_in / l.block_elems) * l.block_bytes;
    if weight.n_bytes() < (nb01 as u64) * n_out as u64
        || x.n_elements() < (n_in * n_query) as u64
        || y.n_elements() < (n_out * n_query) as u64
        || x.dtype != GgmlType::F32
        || y.dtype != GgmlType::F32
    {
        return Err(bad(format!(
            "operand extents: weight {} bytes (need {}), x {} ({:?}, need {}), y {} ({:?}, need {})",
            weight.n_bytes(),
            nb01 * n_out,
            x.n_elements(),
            x.dtype,
            n_in * n_query,
            y.n_elements(),
            y.dtype,
            n_out * n_query
        )));
    }
    let args = GenericMmArgs {
        m: n_out as u32,
        n: n_query as u32,
        k: n_in as u32,
        nb01: nb01 as u32,
        stride_b: n_in as u32,
        min_count: 0,
        max_count: i32::MAX as u32,
        b_div: 1,
        slot_limit: n_query as u32,
    };
    let combined = combined && mat_mat_tensor_combined_pipeline_name(weight.dtype).is_some();
    let wide = wide && !combined && mat_mat_tensor_wide_pipeline_name(weight.dtype).is_some();
    let name = if combined {
        mat_mat_tensor_combined_pipeline_name(weight.dtype).expect("layout found above")
    } else if wide {
        mat_mat_tensor_wide_pipeline_name(weight.dtype).expect("layout found above")
    } else {
        mat_mat_tensor_pipeline_name(weight.dtype).expect("layout found above")
    };
    let (slots, rows, threadgroup_bytes) = if combined {
        (64, 128, TENSOR_THREADGROUP_BYTES_COMBINED)
    } else if wide {
        (64, 64, TENSOR_THREADGROUP_BYTES_WIDE)
    } else {
        (32, 64, TENSOR_THREADGROUP_BYTES)
    };
    let pso = ctx.tensor_pipeline(&name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.set_threadgroup_memory(0, threadgroup_bytes);
    enc.dispatch(
        MTLSize {
            width: n_query.div_ceil(slots),
            height: n_out.div_ceil(rows),
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
