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

/// Activation operand storage the tensor tile reads: F32 rows cast to half
/// inside the tile, or rows already converted to half (`_f16act`). The
/// staged operand is the same half value either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorActivation {
    F32,
    F16,
}

impl TensorActivation {
    fn infix(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16act",
        }
    }

    fn of(src: &MetalTensor, kernel: &'static str) -> Result<Self, MetalError> {
        match src.dtype {
            GgmlType::F32 => Ok(Self::F32),
            GgmlType::F16 => Ok(Self::F16),
            other => Err(MetalError::BadShape {
                kernel,
                detail: format!("activations must be F32 or F16, got {other:?}"),
            }),
        }
    }
}

/// `kernel_moe_<role>_<suffix>_<f32|f16act>_grouped_slots_tensor` for
/// every dtype with a generic layout (the tensor file instantiates the
/// same list; a CPU test checks that).
pub fn moe_grouped_tensor_pipeline_name(
    role: MoeGroupedTensorRole,
    dtype: GgmlType,
    activation: TensorActivation,
) -> Option<String> {
    let l = moe_grouped_generic_layout(dtype)?;
    Some(format!(
        "{}{}_{}_grouped_slots_tensor",
        role.prefix(),
        l.suffix,
        activation.infix()
    ))
}

/// 64-slot gate/up probe pipeline for a generic dtype.
pub fn wide_probe_pipeline_name(dtype: GgmlType) -> Option<String> {
    moe_grouped_generic_layout(dtype).map(|l| format!("kernel_moe_wide_probe_{}_f16act", l.suffix))
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

crate::env_flag!(
    /// Count-adaptive IQ3_S down tensor panels (N16 for 1..=16 slots, N32 for
    /// 17..=32, N64 above) for the GLM-5.3 Flash 2048x4096x288 down bank at
    /// 128..=4096 rows. Same half operands and K32 F32 sums as the 32-slot
    /// tile; only the slot panel width changes. Zero restores the 32-slot tile.
    default_on iq3_down_panels_enabled,
    "QWEN_GLM53_IQ3_DOWN_PANELS"
);

const TENSOR_THREADGROUP_BYTES: usize = 8192;
const TENSOR_THREADGROUP_BYTES_WIDE: usize = 16384;

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_tensor_mm(
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
    let activation = TensorActivation::of(inner, KERNEL)?;
    let name =
        moe_grouped_tensor_pipeline_name(MoeGroupedTensorRole::Down, weight.dtype, activation)
            .expect("layout lookup succeeded in generic_mm_args");
    #[cfg(feature = "perf-ablations")]
    if weight.dtype == GgmlType::IQ3_S
        && crate::glm_append_price::before(ctx, enc, 1, out, weight, inner, Some(counts), n_expert)?
    {
        return Ok(());
    }
    let eligible =
        iq3_down_adaptive_eligible(weight.dtype, inner.dtype, n_in, n_out, n_expert, n_tokens);
    // Production uses the env default; the ablation probe overrides it so the
    // same binary can time both routes.
    #[cfg(feature = "perf-ablations")]
    let adaptive = eligible
        && crate::glm_append_price::down_mode_override()
            .map_or_else(iq3_down_panels_enabled, |mode| mode != 0);
    #[cfg(not(feature = "perf-ablations"))]
    let adaptive = eligible && iq3_down_panels_enabled();
    let encoded = if adaptive {
        #[cfg(feature = "perf-ablations")]
        crate::glm_append_price::note_down();
        encode_moe_down_iq3_adaptive(
            ctx, enc, weight, inner, counts, ids, out, n_in, n_out, n_expert, n_tokens,
        )?
    } else {
        false
    };
    if !encoded {
        dispatch_tensor_mm(
            ctx, enc, &name, &args, weight, inner, counts, ids, out, n_out, n_expert, n_tokens,
            None,
        )?;
    }
    #[cfg(feature = "perf-ablations")]
    if weight.dtype == GgmlType::IQ3_S {
        crate::glm_append_price::after(ctx, enc, 1, out)?;
    }
    Ok(())
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
    let activation = TensorActivation::of(x_pack, KERNEL)?;
    // The wide panel reads half storage only. F32 retains its typed narrow
    // path; at 4096 rows F16 gate/up uses the wide panel as before.
    if wide_swiglu_shape(n_tokens, x_pack.dtype, clamp)
        && let Some(limit) = clamp
        && moe_grouped_tensor_selected(ctx)
        && std::env::var_os("QWEN_V41_EXPERT_WIDE").is_none_or(|v| v != "0")
        && wide_probe_pipeline_name(w_gate.dtype).is_some()
        && wide_probe_pipeline_name(w_up.dtype).is_some()
    {
        return encode_moe_swiglu_wide_probe(
            ctx, enc, w_gate, w_up, x_pack, counts, ids, inner, n_hidden, n_ffn, n_expert, topk,
            n_tokens, limit,
        );
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
    let gate =
        moe_grouped_tensor_pipeline_name(MoeGroupedTensorRole::Down, w_gate.dtype, activation)
            .expect("layout lookup succeeded in generic_mm_args");
    let up_role = if clamp.is_some() {
        MoeGroupedTensorRole::UpClamped
    } else {
        MoeGroupedTensorRole::UpSiluMul
    };
    let up = moe_grouped_tensor_pipeline_name(up_role, w_up.dtype, activation)
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
pub fn mat_mat_tensor_pipeline_name(
    dtype: GgmlType,
    activation: TensorActivation,
) -> Option<String> {
    let l = moe_grouped_generic_layout(dtype)?;
    Some(format!(
        "kernel_mat_mat_{}_{}_tensor",
        l.suffix,
        activation.infix()
    ))
}

/// 64-slot dense form for the shared Q8_0-only wide-tile contract.
pub fn mat_mat_tensor_wide_pipeline_name(
    dtype: GgmlType,
    activation: TensorActivation,
) -> Option<String> {
    (dtype == GgmlType::Q8_0)
        .then(|| format!("kernel_mat_mat_q8_0_{}_tensor_wide", activation.infix()))
}

/// Explicit Q6_K capability; the shared automatic wide lookup stays Q8-only.
pub fn mat_mat_tensor_q6_wide_pipeline_name(
    dtype: GgmlType,
    activation: TensorActivation,
) -> Option<String> {
    (dtype == GgmlType::Q6_K)
        .then(|| format!("kernel_mat_mat_q6_K_{}_tensor_wide", activation.infix()))
}

#[derive(Clone, Copy)]
enum DenseTile {
    Narrow,
    WideQ8,
    ExplicitQuantWide,
    V41Wide,
}

/// Dense `y[row] = W x[row]` on the tensor tile with the `mat_mat`
/// contract: `weight` is `[n_in, n_out]` in any dtype with a generic
/// layout, `x` is `[n_in, n_query]` F32 (or F16, already converted) and
/// `y` is `[n_out, n_query]` F32 (row-major rows of `n_in` / `n_out`),
/// `n_in` a multiple of 32 and of the dtype's block. Half operands, F32
/// sums; output rows past `n_query` are untouched. Callers gate on
/// [`moe_grouped_tensor_selected`].
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
    let wide = weight.dtype == GgmlType::Q8_0
        && n_query >= 128
        && std::env::var_os("QWEN_V41_DENSE_NARROW").is_none();
    encode_mat_mat_tensor_f32_impl(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        if wide {
            DenseTile::WideQ8
        } else {
            DenseTile::Narrow
        },
    )
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
    encode_mat_mat_tensor_f32_impl(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        DenseTile::WideQ8,
    )
}

/// Explicit wide tile for the qualified Flash-Next projection caller. This
/// does not expand the shared Q8-only wide-tile capability lookup.
pub fn encode_mat_mat_tensor_f32_flash_next_wide(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    encode_mat_mat_tensor_f32_impl(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        DenseTile::ExplicitQuantWide,
    )
}

/// Explicit V4.1 Q5_K/cached-F16 projections with preconverted half input.
/// The caller owns device/geometry qualification; shared widening is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_tensor_f32_v41_wide(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if !matches!(weight.dtype, GgmlType::Q5_K | GgmlType::F16) || x.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "v41_dense_wide",
            detail: "Q5_K/F16 weights and F16 activations required".into(),
        });
    }
    encode_mat_mat_tensor_f32_impl(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        DenseTile::V41Wide,
    )
}

/// Explicit Q6_K wide tile for GLM's bounded short-append policy.
pub fn encode_mat_mat_tensor_f32_q6_wide(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::Q6_K {
        return Err(super::checks::bad_shape(
            "mat_mat_q6_wide",
            "Q6_K weights required",
        ));
    }
    encode_mat_mat_tensor_f32_impl(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        DenseTile::ExplicitQuantWide,
    )
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
    encode_mat_mat_tensor_f32_impl(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        DenseTile::Narrow,
    )
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
    tile: DenseTile,
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
    let activation = TensorActivation::of(x, KERNEL)?;
    let nb01 = (n_in / l.block_elems) * l.block_bytes;
    if weight.n_bytes() < (nb01 as u64) * n_out as u64
        || x.n_elements() < (n_in * n_query) as u64
        || y.n_elements() < (n_out * n_query) as u64
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
    let wide_name = match tile {
        DenseTile::Narrow => None,
        DenseTile::WideQ8 => mat_mat_tensor_wide_pipeline_name(weight.dtype, activation),
        DenseTile::V41Wide => Some(format!(
            "kernel_mat_mat_{}_{}_tensor_wide",
            l.suffix,
            activation.infix()
        )),
        DenseTile::ExplicitQuantWide => matches!(
            weight.dtype,
            GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::IQ4_XS | GgmlType::IQ4_NL
        )
        .then(|| {
            format!(
                "kernel_mat_mat_{}_{}_tensor_wide",
                l.suffix,
                activation.infix()
            )
        }),
    };
    let wide = wide_name.is_some();
    let name = wide_name.unwrap_or_else(|| {
        mat_mat_tensor_pipeline_name(weight.dtype, activation).expect("layout found above")
    });
    let slots = if wide { 64 } else { 32 };
    let threadgroup_bytes = if wide {
        TENSOR_THREADGROUP_BYTES_WIDE
    } else {
        TENSOR_THREADGROUP_BYTES
    };
    #[cfg(feature = "perf-ablations")]
    if weight.dtype == GgmlType::Q6_K
        && wide
        && crate::glm_append_price::before(
            ctx,
            enc,
            2,
            &y.view_subrange(0, vec![n_out as u64, n_query as u64]),
            weight,
            &x.view_subrange(0, vec![n_in as u64, n_query as u64]),
            None,
            1,
        )?
    {
        return Ok(());
    }
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
            height: n_out.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    #[cfg(feature = "perf-ablations")]
    if weight.dtype == GgmlType::Q6_K && wide {
        crate::glm_append_price::after(
            ctx,
            enc,
            2,
            &y.view_subrange(0, vec![n_out as u64, n_query as u64]),
        )?;
    }
    Ok(())
}

/// Natural row-major Q5_K -> F16 dequant for the dense preweight route.
/// Writes the same half values the Q5_K tensor tile would dequantize.
pub fn encode_v41_dequant_q5_k_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "v41_dequant_q5_k_f16";
    let bad = |detail: String| MetalError::BadShape {
        kernel: KERNEL,
        detail,
    };
    if src.dtype != GgmlType::Q5_K || dst.dtype != GgmlType::F16 {
        return Err(bad(format!(
            "requires Q5_K source and F16 destination, got {:?} -> {:?}",
            src.dtype, dst.dtype
        )));
    }
    if n_in == 0 || n_out == 0 || !n_in.is_multiple_of(256) {
        return Err(bad(format!("n_in={n_in} n_out={n_out}")));
    }
    let elements = n_in
        .checked_mul(n_out)
        .ok_or_else(|| bad("extent overflow".into()))?;
    if src.n_elements() < elements as u64 || dst.n_elements() < elements as u64 {
        return Err(bad(format!(
            "operand extents: src {} dst {} need {elements}",
            src.n_elements(),
            dst.n_elements()
        )));
    }
    let pso = ctx.tensor_pipeline(KERNEL)?;
    enc.set_pipeline(&pso);
    enc.set_tensor(0, src);
    enc.set_tensor(1, dst);
    enc.set_bytes(2, &[n_in as u32, n_out as u32]);
    enc.dispatch(
        MTLSize {
            width: n_in / 16,
            height: n_out,
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

/// Natural row-major Q6_K -> F16 dequant for the GLM dense prefill cache.
/// Writes the exact half values produced by the Q6_K tensor tile decoder.
pub fn encode_glm_dequant_q6_k_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "glm_dequant_q6_k_f16";
    let bad = |detail: String| MetalError::BadShape {
        kernel: KERNEL,
        detail,
    };
    if src.dtype != GgmlType::Q6_K || dst.dtype != GgmlType::F16 {
        return Err(bad(format!(
            "requires Q6_K source and F16 destination, got {:?} -> {:?}",
            src.dtype, dst.dtype
        )));
    }
    if n_in == 0 || n_out == 0 || !n_in.is_multiple_of(256) {
        return Err(bad(format!("n_in={n_in} n_out={n_out}")));
    }
    let elements = n_in
        .checked_mul(n_out)
        .ok_or_else(|| bad("extent overflow".into()))?;
    if src.n_elements() < elements as u64 || dst.n_elements() < elements as u64 {
        return Err(bad(format!(
            "operand extents: src {} dst {} need {elements}",
            src.n_elements(),
            dst.n_elements()
        )));
    }
    let pso = ctx.tensor_pipeline(KERNEL)?;
    enc.set_pipeline(&pso);
    enc.set_tensor(0, src);
    enc.set_tensor(1, dst);
    enc.set_bytes(2, &[n_in as u32, n_out as u32]);
    enc.dispatch(
        MTLSize {
            width: n_in / 16,
            height: n_out,
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

/// Probe-only 64-slot down dispatch with F16 activation storage. Same data and
/// arithmetic as the narrow tile; compares panel cost at 4096 routed tokens.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_down_wide_probe(
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
    const KERNEL: &str = "moe_down_wide_probe";
    require_wide_half(KERNEL, inner.dtype)?;
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
    let name = wide_probe_pipeline_name(weight.dtype).ok_or(MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("no wide probe pipeline for {:?}", weight.dtype),
    })?;
    let pso = ctx.tensor_pipeline(&name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_bytes(6, &0.0f32);
    enc.set_bytes(7, &0i32);
    enc.set_threadgroup_memory(0, TENSOR_THREADGROUP_BYTES_WIDE);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(64),
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

/// F16-activation-only 64-slot gate/up: gate into `inner` (EPI 0), then clamped up
/// pass in place (EPI 2). Selected from `encode_moe_swiglu_f32_grouped_slots_tensor`
/// at >=4096 rows; `QWEN_V41_EXPERT_WIDE=0` restores the narrow tile.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_swiglu_wide_probe(
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
    clamp_limit: f32,
) -> Result<(), MetalError> {
    encode_moe_swiglu_wide_impl(
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
        Some(clamp_limit),
    )
}

/// Explicit unclamped wide gate/up for qualified Flash-Next callers.
pub fn encode_moe_swiglu_wide_unclamped(
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
    encode_moe_swiglu_wide_impl(
        ctx, enc, w_gate, w_up, x_pack, counts, ids, inner, n_hidden, n_ffn, n_expert, topk,
        n_tokens, None,
    )
}

fn encode_moe_swiglu_wide_impl(
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
    clamp_limit: Option<f32>,
) -> Result<(), MetalError> {
    const KERNEL: &str = "moe_swiglu_wide_probe";
    require_wide_half(KERNEL, x_pack.dtype)?;
    crate::metal::checks::require_serial(KERNEL, enc)?;
    if let Some(limit) = clamp_limit
        && (!limit.is_finite() || limit <= 0.0)
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("clamp must be finite and positive, got {limit}"),
        });
    }
    let gate_name = wide_probe_pipeline_name(w_gate.dtype).ok_or(MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("no wide gate pipeline for {:?}", w_gate.dtype),
    })?;
    let up_name = wide_probe_pipeline_name(w_up.dtype).ok_or(MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("no wide up pipeline for {:?}", w_up.dtype),
    })?;
    // Resolve both before the first dispatch so a missing instantiation cannot
    // leave the gate pass half applied.
    ctx.tensor_pipeline(&gate_name)?;
    ctx.tensor_pipeline(&up_name)?;
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
    let dispatch = |name: &str,
                    args: &GenericMmArgs,
                    weight: &MetalTensor,
                    epi: i32,
                    limit: f32|
     -> Result<(), MetalError> {
        let pso = ctx.tensor_pipeline(name)?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, args);
        enc.set_tensor(1, weight);
        enc.set_tensor(2, x_pack);
        enc.set_tensor(3, counts);
        enc.set_tensor(4, ids);
        enc.set_tensor(5, inner);
        enc.set_bytes(6, &limit);
        enc.set_bytes(7, &epi);
        enc.set_threadgroup_memory(0, TENSOR_THREADGROUP_BYTES_WIDE);
        enc.dispatch(
            MTLSize {
                width: n_tokens.div_ceil(64),
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
    };
    dispatch(&gate_name, &gate_args, w_gate, 0, 0.0)?;
    dispatch(
        &up_name,
        &up_args,
        w_up,
        if clamp_limit.is_some() { 2 } else { 1 },
        clamp_limit.unwrap_or(0.0),
    )?;
    Ok(())
}

fn wide_swiglu_shape(n_tokens: usize, activation: GgmlType, clamp: Option<f32>) -> bool {
    n_tokens >= 4096 && activation == GgmlType::F16 && clamp.is_some()
}

fn require_wide_half(kernel: &'static str, activation: GgmlType) -> Result<(), MetalError> {
    if activation != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("wide panel reads F16 activations, got {activation:?}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod wide_policy_tests {
    use super::*;

    #[test]
    fn wide_policy_requires_physical_4k_half_clamped_rows() {
        for n in [1, 2047, 2048, 2049, 4095, 4096, 4097, 8192] {
            assert_eq!(wide_swiglu_shape(n, GgmlType::F16, Some(10.0)), n >= 4096);
            assert!(!wide_swiglu_shape(n, GgmlType::F32, Some(10.0)));
            assert!(!wide_swiglu_shape(n, GgmlType::F16, None));
        }
        require_wide_half("test", GgmlType::F16).unwrap();
        for dtype in [GgmlType::F32, GgmlType::BF16, GgmlType::I32] {
            assert!(require_wide_half("test", dtype).is_err());
        }
    }
}

/// Explicit Half-input tensor gate/up with N32 panels for buckets up to 32
/// slots and N64 panels for larger buckets. The two disjoint count ranges
/// preserve every slot's operands, clamp and K reduction. No buffer allocation.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_swiglu_adaptive_tensor(
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
    clamp: f32,
) -> Result<(), MetalError> {
    const KERNEL: &str = "moe_swiglu_adaptive_tensor";
    crate::metal::expert_tensor_down::check_encoder(ctx, enc)?;
    require_wide_half(KERNEL, x_pack.dtype)?;
    for t in [w_gate, w_up, x_pack, counts, ids, inner] {
        if t.buffer.device().registryID() != ctx.device.registryID() {
            return Err(crate::metal::checks::bad_shape(
                KERNEL,
                "binding belongs to another device",
            ));
        }
    }

    if !clamp.is_finite() || clamp <= 0.0 {
        return Err(crate::metal::checks::bad_shape(
            KERNEL,
            "clamp must be finite and positive",
        ));
    }
    let gate_args = generic_mm_args(
        KERNEL, w_gate, x_pack, counts, ids, inner, n_hidden, n_ffn, n_expert, n_tokens, topk, 0,
        32,
    )?;
    let gate_wide_args = generic_mm_args(
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
        33,
        i32::MAX as u32,
    )?;
    let up_args = generic_mm_args(
        KERNEL, w_up, x_pack, counts, ids, inner, n_hidden, n_ffn, n_expert, n_tokens, topk, 0, 32,
    )?;
    let up_wide_args = generic_mm_args(
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
        33,
        i32::MAX as u32,
    )?;
    let gate_name = moe_grouped_tensor_pipeline_name(
        MoeGroupedTensorRole::Down,
        w_gate.dtype,
        TensorActivation::F16,
    )
    .expect("validated gate layout");
    let up_name = moe_grouped_tensor_pipeline_name(
        MoeGroupedTensorRole::UpClamped,
        w_up.dtype,
        TensorActivation::F16,
    )
    .expect("validated up layout");
    let wide_gate_name = wide_probe_pipeline_name(w_gate.dtype).expect("validated gate layout");
    let wide_up_name = wide_probe_pipeline_name(w_up.dtype).expect("validated up layout");
    let psos = [
        ctx.tensor_pipeline(&gate_name)?,
        ctx.tensor_pipeline(&wide_gate_name)?,
        ctx.tensor_pipeline(&up_name)?,
        ctx.tensor_pipeline(&wide_up_name)?,
    ];
    let passes = [
        (&gate_args, w_gate, 0i32, false),
        (&gate_wide_args, w_gate, 0, true),
        (&up_args, w_up, 2, false),
        (&up_wide_args, w_up, 2, true),
    ];
    // Resolve and check every pass before the first encoder mutation.
    for (p, &(_, _, _, wide)) in psos.iter().zip(&passes) {
        let bytes = if wide {
            TENSOR_THREADGROUP_BYTES_WIDE
        } else {
            TENSOR_THREADGROUP_BYTES
        };
        if p.threadExecutionWidth() != 32
            || p.maxTotalThreadsPerThreadgroup() < 128
            || p.staticThreadgroupMemoryLength()
                .checked_add(bytes)
                .is_none_or(|n| n > ctx.device.maxThreadgroupMemoryLength())
        {
            return Err(crate::metal::checks::bad_shape(
                KERNEL,
                "unsupported tensor panel capacity",
            ));
        }
    }
    for (p, &(args, weight, epi, wide)) in psos.iter().zip(&passes) {
        enc.set_pipeline(p);
        enc.set_bytes(0, args);
        for (slot, t) in [(1, weight), (2, x_pack), (3, counts), (4, ids), (5, inner)] {
            enc.set_tensor(slot, t);
        }
        enc.set_bytes(6, &clamp);
        if wide {
            enc.set_bytes(7, &epi);
        }
        enc.set_threadgroup_memory(
            0,
            if wide {
                TENSOR_THREADGROUP_BYTES_WIDE
            } else {
                TENSOR_THREADGROUP_BYTES
            },
        );
        enc.dispatch(
            MTLSize {
                width: if wide { n_tokens.div_ceil(64) } else { 1 },
                height: n_ffn.div_ceil(64),
                depth: n_expert,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

/// Shape gate for the count-adaptive IQ3_S down panels: the GLM-5.3 Flash
/// down bank with F32 activation staging at append-sized row counts.
fn iq3_down_adaptive_eligible(
    weight: GgmlType,
    activation: GgmlType,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> bool {
    weight == GgmlType::IQ3_S
        && activation == GgmlType::F32
        && (n_in, n_out, n_expert) == (2048, 4096, 288)
        && (128..=4096).contains(&n_tokens)
}

/// Count-adaptive F32-storage / Half-operand IQ3_S down.
///
/// Dispatches one panel per count band: N16 slots for buckets of 1..=16, the
/// 32-slot tile for 17..=32, and an N64 tile above. Bands are disjoint, so
/// every slot is computed exactly once with the incumbent's half operands and
/// K32 F32 accumulation. Returns `Ok(false)` without touching the encoder when
/// a panel pipeline is missing or cannot fit its threadgroup allocation, so the
/// caller falls back to the 32-slot tensor tile.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_down_iq3_adaptive(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w: &MetalTensor,
    x: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    k: usize,
    m: usize,
    e: usize,
    n: usize,
) -> Result<bool, MetalError> {
    const K: &str = "iq3_adaptive_down";
    if crate::metal::expert_tensor_down::check_encoder(ctx, enc).is_err() {
        return Ok(false);
    }
    if w.dtype != GgmlType::IQ3_S || x.dtype != GgmlType::F32 {
        return Ok(false);
    }
    let narrow = moe_grouped_tensor_pipeline_name(
        MoeGroupedTensorRole::Down,
        GgmlType::IQ3_S,
        TensorActivation::F32,
    )
    .unwrap();
    let bands = [
        (16usize, 1u32, 16u32),
        (32, 17, 32),
        (64, 33, i32::MAX as u32),
    ];
    let mut passes = Vec::with_capacity(bands.len());
    for (rows, min, max) in bands {
        let args = generic_mm_args(K, w, x, counts, ids, out, k, m, e, n, 1, min, max)?;
        let name = match rows {
            16 => "kernel_glm_iq3_down_n16",
            32 => narrow.as_str(),
            _ => "kernel_glm_iq3_down_n64",
        };
        let bytes = match rows {
            16 => 5120,
            32 => 8192,
            _ => 16384,
        };
        let Ok(p) = ctx.tensor_pipeline(name) else {
            return Ok(false);
        };
        if p.threadExecutionWidth() != 32
            || p.maxTotalThreadsPerThreadgroup() < 128
            || p.staticThreadgroupMemoryLength()
                .checked_add(bytes)
                .is_none_or(|n| n > ctx.device.maxThreadgroupMemoryLength())
        {
            return Ok(false);
        }
        passes.push((p, args, rows, bytes));
    }
    for (p, args, rows, bytes) in passes {
        enc.set_pipeline(&p);
        enc.set_bytes(0, &args);
        for (i, t) in [(1, w), (2, x), (3, counts), (4, ids), (5, out)] {
            enc.set_tensor(i, t);
        }
        enc.set_threadgroup_memory(0, bytes);
        enc.dispatch(
            MTLSize {
                width: if rows == 64 { n.div_ceil(64) } else { 1 },
                height: m.div_ceil(64),
                depth: e,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(true)
}

#[cfg(test)]
mod iq3_down_panel_policy_tests {
    use super::*;

    #[test]
    fn adaptive_panels_require_the_glm_down_shape_and_row_band() {
        let glm = |rows| {
            iq3_down_adaptive_eligible(GgmlType::IQ3_S, GgmlType::F32, 2048, 4096, 288, rows)
        };
        for rows in [128, 400, 512, 2000, 2048, 4096] {
            assert!(glm(rows), "rows {rows}");
        }
        for rows in [0, 127, 4097, 8192] {
            assert!(!glm(rows), "rows {rows}");
        }
        assert!(!iq3_down_adaptive_eligible(
            GgmlType::IQ3_XXS,
            GgmlType::F32,
            2048,
            4096,
            288,
            400
        ));
        assert!(!iq3_down_adaptive_eligible(
            GgmlType::IQ3_S,
            GgmlType::F16,
            2048,
            4096,
            288,
            400
        ));
        for (n_in, n_out, experts) in [(2048, 4096, 287), (2048, 4095, 288), (2049, 4096, 288)] {
            assert!(!iq3_down_adaptive_eligible(
                GgmlType::IQ3_S,
                GgmlType::F32,
                n_in,
                n_out,
                experts,
                400
            ));
        }
    }
}
