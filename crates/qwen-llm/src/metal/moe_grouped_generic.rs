//! Generic grouped-slots MoE encoders: one kernel template per role
//! (`kernels/moe.metal` `kernel_moe_grouped_slots_mm_generic` /
//! `kernel_moe_swiglu_grouped_slots_n16_generic`), instantiated for every
//! weight dtype with a canonical tile dequant in `kernels/quant_tiles.h`.
//!
//! This is the *general* grouped MoE prefill path. Hand-tuned per-type
//! kernels (Q4_K n16/n32 SwiGLU, Q5_K tiny8_r16 down, ...) stay the
//! preferred fast paths and are selected first by the dispatcher in
//! `metal_dflash.rs`; anything they do not cover lands here instead of the
//! per-token fallback.
//!
//! Numerics: like llama.cpp's Metal `mul_mm`, tiles stage weights and
//! activations in half and accumulate in f32. Decoded weights and activations
//! must fit half for finite results, including native F32/BF16 banks. Storage
//! dtype support does not imply F32 operand precision; callers requesting it
//! must use a supported F32-operand tile or retain their per-row path.

use super::*;

/// Tile geometry of one generic instantiation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeGroupedGenericLayout {
    pub dtype: GgmlType,
    /// Kernel-name infix (`kernel_moe_down_<suffix>_f32_grouped_slots_generic`).
    pub suffix: &'static str,
    /// Logical elements per tile block (256 for K/I-quants, 64 for Q2_0, 32 otherwise).
    pub block_elems: usize,
    /// Bytes per tile block.
    pub block_bytes: usize,
}

const fn layout(
    dtype: GgmlType,
    suffix: &'static str,
    block_elems: usize,
    block_bytes: usize,
) -> MoeGroupedGenericLayout {
    MoeGroupedGenericLayout {
        dtype,
        suffix,
        block_elems,
        block_bytes,
    }
}

/// Every dtype with a generic grouped instantiation (down, up_silu_mul and
/// fused SwiGLU). Must stay in sync with the `host_name` list at the end of
/// `kernels/moe.metal`; `moe_grouped_generic_mapping_matches_metal_source`
/// enforces that.
pub const MOE_GROUPED_GENERIC_LAYOUTS: &[MoeGroupedGenericLayout] = &[
    layout(GgmlType::Q2_0, "q2_0", 64, 18),
    layout(GgmlType::Q2_K, "q2_K", 256, 84),
    layout(GgmlType::Q3_K, "q3_K", 256, 110),
    layout(GgmlType::Q4_K, "q4_K", 256, 144),
    layout(GgmlType::Q5_K, "q5_K", 256, 176),
    layout(GgmlType::Q6_K, "q6_K", 256, 210),
    layout(GgmlType::Q8_0, "q8_0", 32, 34),
    layout(GgmlType::Q4_0, "q4_0", 32, 18),
    layout(GgmlType::Q4_1, "q4_1", 32, 20),
    layout(GgmlType::IQ2_S, "iq2_s", 256, 82),
    layout(GgmlType::IQ3_XXS, "iq3_xxs", 256, 98),
    layout(GgmlType::IQ3_S, "iq3_s", 256, 110),
    layout(GgmlType::IQ4_NL, "iq4_nl", 32, 18),
    layout(GgmlType::IQ4_XS, "iq4_xs", 256, 136),
    layout(GgmlType::F32, "f32", 32, 128),
    layout(GgmlType::F16, "f16", 32, 64),
    layout(GgmlType::BF16, "bf16", 32, 64),
    layout(GgmlType::MXFP4, "mxfp4", 32, 17),
    layout(GgmlType::IQ2_XS, "iq2_xs", 256, 74),
    layout(GgmlType::IQ2_XXS, "iq2_xxs", 256, 66),
];

/// Kernel roles instantiated per dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeGroupedGenericRole {
    /// `dst[slot] = W_e * src[slot / b_div]` (down projection; also the gate
    /// pass of the mixed-dtype gate/up path).
    Down,
    /// `dst[slot] = silu(dst[slot]) * (W_e * x[slot / topk])` (up pass of the
    /// mixed-dtype gate/up path).
    UpSiluMul,
    /// Fused same-dtype gate+up SwiGLU.
    Swiglu,
}

impl MoeGroupedGenericRole {
    pub const ALL: [Self; 3] = [Self::Down, Self::UpSiluMul, Self::Swiglu];

    fn prefix(self) -> &'static str {
        match self {
            Self::Down => "kernel_moe_down_",
            Self::UpSiluMul => "kernel_moe_up_silu_mul_",
            Self::Swiglu => "kernel_moe_swiglu_",
        }
    }
}

pub fn moe_grouped_generic_layout(dtype: GgmlType) -> Option<MoeGroupedGenericLayout> {
    MOE_GROUPED_GENERIC_LAYOUTS
        .iter()
        .copied()
        .find(|l| l.dtype == dtype)
}

/// True when `dtype` has a generic grouped MoE instantiation for every role.
pub fn moe_grouped_generic_supported(dtype: GgmlType) -> bool {
    moe_grouped_generic_layout(dtype).is_some()
}

pub fn moe_grouped_generic_pipeline_name(
    role: MoeGroupedGenericRole,
    dtype: GgmlType,
) -> Option<String> {
    let l = moe_grouped_generic_layout(dtype)?;
    Some(format!(
        "{}{}_f32_grouped_slots_generic",
        role.prefix(),
        l.suffix
    ))
}

fn generic_layout_for(
    kernel: &'static str,
    weight: &MetalTensor,
    n_in: usize,
    n_rows: usize,
    n_expert: usize,
) -> Result<MoeGroupedGenericLayout, MetalError> {
    let l = moe_grouped_generic_layout(weight.dtype).ok_or_else(|| MetalError::BadShape {
        kernel,
        detail: format!("no generic grouped instantiation for {:?}", weight.dtype),
    })?;
    checked_layout(kernel, weight, n_in, n_rows, n_expert, l)
}

fn f32x_layout_for(
    kernel: &'static str,
    weight: &MetalTensor,
    n_in: usize,
    n_rows: usize,
    n_expert: usize,
) -> Result<MoeGroupedGenericLayout, MetalError> {
    match weight.dtype {
        GgmlType::MXFP4 => checked_layout(
            kernel,
            weight,
            n_in,
            n_rows,
            n_expert,
            layout(GgmlType::MXFP4, "mxfp4", 32, 17),
        ),
        GgmlType::IQ2_XXS => checked_layout(
            kernel,
            weight,
            n_in,
            n_rows,
            n_expert,
            layout(GgmlType::IQ2_XXS, "iq2_xxs", 256, 66),
        ),
        GgmlType::IQ2_XS => checked_layout(
            kernel,
            weight,
            n_in,
            n_rows,
            n_expert,
            layout(GgmlType::IQ2_XS, "iq2_xs", 256, 74),
        ),
        _ => generic_layout_for(kernel, weight, n_in, n_rows, n_expert),
    }
}

fn checked_layout(
    kernel: &'static str,
    weight: &MetalTensor,
    n_in: usize,
    n_rows: usize,
    n_expert: usize,
    l: MoeGroupedGenericLayout,
) -> Result<MoeGroupedGenericLayout, MetalError> {
    // The tile K-step is 32 elements; QK=256 formats additionally need whole
    // super-blocks per row.
    let k_align = l.block_elems.max(32);
    if n_in == 0 || !n_in.is_multiple_of(k_align) {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "n_in={n_in} not a positive multiple of {k_align} for {:?}",
                weight.dtype
            ),
        });
    }
    let row_bytes = (n_in / l.block_elems) * l.block_bytes;
    let need = (row_bytes as u64)
        .checked_mul(n_rows as u64)
        .and_then(|b| b.checked_mul(n_expert as u64))
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: "expert bank byte size overflows".into(),
        })?;
    if weight.n_bytes() < need {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "{:?} expert bank has {} bytes, need {need} for n_in={n_in} rows={n_rows} n_expert={n_expert}",
                weight.dtype,
                weight.n_bytes()
            ),
        });
    }
    if u32::try_from(row_bytes).is_err() {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("row stride {row_bytes} exceeds u32"),
        });
    }
    Ok(l)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GenericMmArgs {
    m: u32,
    n: u32,
    k: u32,
    nb01: u32,
    stride_b: u32,
    min_count: u32,
    max_count: u32,
    b_div: u32,
    slot_limit: u32,
}

#[allow(clippy::too_many_arguments)]
fn encode_generic_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    role: MoeGroupedGenericRole,
    kernel: &'static str,
    weight: &MetalTensor,
    src: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
    b_div: usize,
    min_count: u32,
    max_count: u32,
) -> Result<(), MetalError> {
    debug_assert!(role != MoeGroupedGenericRole::Swiglu);
    let args = generic_mm_args(
        kernel, weight, src, counts, ids, out, n_in, n_out, n_expert, n_tokens, b_div, min_count,
        max_count,
    )?;
    let name = moe_grouped_generic_pipeline_name(role, weight.dtype)
        .expect("layout lookup succeeded in generic_mm_args");
    dispatch_generic_mm(
        ctx, enc, &name, 8192, &args, weight, src, counts, ids, out, n_out, n_expert, n_tokens,
    )
}

/// Checked arguments of one grouped mat-mat (the generic and F32-operand
/// down tiles share them).
#[allow(clippy::too_many_arguments)]
fn generic_mm_args(
    kernel: &'static str,
    weight: &MetalTensor,
    src: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
    b_div: usize,
    min_count: u32,
    max_count: u32,
) -> Result<GenericMmArgs, MetalError> {
    let l = generic_layout_for(kernel, weight, n_in, n_out, n_expert)?;
    mm_args_with_layout(
        kernel, l, src, counts, ids, out, n_in, n_out, n_expert, n_tokens, b_div, min_count,
        max_count,
    )
}

#[allow(clippy::too_many_arguments)]
fn mm_args_with_layout(
    kernel: &'static str,
    l: MoeGroupedGenericLayout,
    src: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
    b_div: usize,
    min_count: u32,
    max_count: u32,
) -> Result<GenericMmArgs, MetalError> {
    if n_out == 0 || b_div == 0 || n_tokens == 0 || n_expert == 0 {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "degenerate shape n_out={n_out} b_div={b_div} n_tokens={n_tokens} n_expert={n_expert}"
            ),
        });
    }
    let slot_count = out.n_elements() as usize / n_out;
    let src_rows = slot_count.div_ceil(b_div);
    if out.n_elements() as usize != slot_count * n_out
        || src.n_elements() as usize != src_rows * n_in
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
    {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "shape mismatch: src={} counts={} ids={} out={} expected src={} counts={} ids={} out=k*{}",
                src.n_elements(),
                counts.n_elements(),
                ids.n_elements(),
                out.n_elements(),
                src_rows * n_in,
                n_expert,
                n_expert * n_tokens,
                n_out
            ),
        });
    }
    Ok(GenericMmArgs {
        m: n_out as u32,
        n: n_tokens as u32,
        k: n_in as u32,
        nb01: ((n_in / l.block_elems) * l.block_bytes) as u32,
        stride_b: n_in as u32,
        min_count,
        max_count,
        b_div: b_div as u32,
        slot_limit: slot_count as u32,
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_generic_mm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    name: &str,
    threadgroup_bytes: usize,
    args: &GenericMmArgs,
    weight: &MetalTensor,
    src: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    let pso = ctx.pipeline(name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, args);
    enc.set_tensor(1, weight);
    enc.set_tensor(2, src);
    enc.set_tensor(3, counts);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, out);
    enc.set_threadgroup_memory(0, threadgroup_bytes);
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

/// Generic grouped down projection. Same contract as
/// [`encode_moe_down_q5_K_f32_grouped_slots_range`]: `inner` holds one
/// `n_in` row per routed slot, `out` one `n_out` row per slot, `ids` is the
/// per-expert slot list (row stride `n_tokens`) and `counts` its lengths.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_down_f32_grouped_slots_generic_range(
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
    encode_generic_mm(
        ctx,
        enc,
        MoeGroupedGenericRole::Down,
        "moe_down_grouped_slots_generic",
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
        min_count,
        max_count,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_moe_down_f32_grouped_slots_generic(
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
    encode_moe_down_f32_grouped_slots_generic_range(
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

/// Generic grouped gate/up SwiGLU. Same contract as
/// [`encode_moe_swiglu_q5_K_f32_grouped_slots_n16_range`]:
/// `inner[slot] = silu(G_e x[slot / topk]) * (U_e x[slot / topk])`.
///
/// Same-dtype gate/up banks use the fused kernel. Mixed dtypes run two
/// dependent dispatches (gate into `inner`, then `inner = silu(inner) * up`),
/// so they require a serial encoder.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_swiglu_f32_grouped_slots_generic_range(
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
    const KERNEL: &str = "moe_swiglu_grouped_slots_generic";
    if topk == 0 || n_ffn == 0 || n_tokens == 0 || n_expert == 0 {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "degenerate shape topk={topk} n_ffn={n_ffn} n_tokens={n_tokens} n_expert={n_expert}"
            ),
        });
    }
    if x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
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

    if w_gate.dtype != w_up.dtype {
        if enc.is_concurrent() {
            return Err(MetalError::BadShape {
                kernel: KERNEL,
                detail: format!(
                    "mixed gate/up dtypes {:?}/{:?} need two dependent dispatches; \
                     a concurrent encoder cannot order them",
                    w_gate.dtype, w_up.dtype
                ),
            });
        }
        encode_generic_mm(
            ctx,
            enc,
            MoeGroupedGenericRole::Down,
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
            min_count,
            max_count,
        )?;
        return encode_generic_mm(
            ctx,
            enc,
            MoeGroupedGenericRole::UpSiluMul,
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
            min_count,
            max_count,
        );
    }

    encode_fused_swiglu(
        ctx, enc, KERNEL, w_gate, w_up, x_pack, counts, ids, inner, n_hidden, n_ffn, n_expert,
        topk, n_tokens, min_count, max_count, None,
    )
}

/// Same-dtype fused gate/up dispatch; `clamp` selects the clamped epilogue
/// `silu(min(g, c)) * clamp(u, -c, c)`.
#[allow(clippy::too_many_arguments)]
fn encode_fused_swiglu(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kernel: &'static str,
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
    clamp: Option<f32>,
) -> Result<(), MetalError> {
    let l = generic_layout_for(kernel, w_gate, n_hidden, n_ffn, n_expert)?;
    generic_layout_for(kernel, w_up, n_hidden, n_ffn, n_expert)?;
    let name = match clamp {
        None => moe_grouped_generic_pipeline_name(MoeGroupedGenericRole::Swiglu, w_gate.dtype)
            .expect("layout lookup succeeded above"),
        Some(_) => {
            clamped_swiglu_pipeline_name(w_gate.dtype).ok_or_else(|| MetalError::BadShape {
                kernel,
                detail: format!(
                    "no clamped grouped SwiGLU instantiation for {:?}",
                    w_gate.dtype
                ),
            })?
        }
    };
    dispatch_fused_swiglu(
        ctx, enc, &name, 16384, 16, l, w_gate, w_up, x_pack, counts, ids, inner, n_hidden, n_ffn,
        n_expert, topk, n_tokens, min_count, max_count, clamp,
    )
}

/// One fused gate/up dispatch of the half-staged or F32-operand grouped
/// SwiGLU (same arguments, buffers and grid).
#[allow(clippy::too_many_arguments)]
fn dispatch_fused_swiglu(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    name: &str,
    threadgroup_bytes: usize,
    tokens_per_tile: usize,
    l: MoeGroupedGenericLayout,
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
    clamp: Option<f32>,
) -> Result<(), MetalError> {
    let pso = ctx.pipeline(name)?;
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
            nb01: ((n_hidden / l.block_elems) * l.block_bytes) as u32,
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
    if let Some(clamp) = clamp {
        enc.set_bytes(7, &clamp);
    }
    enc.set_threadgroup_memory(0, threadgroup_bytes);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(tokens_per_tile),
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

/// Dtypes with a clamped fused grouped SwiGLU instantiation (the expert
/// dtypes of the GLM-5.3 release).
pub const MOE_GROUPED_CLAMPED_SWIGLU_DTYPES: &[GgmlType] = &[GgmlType::IQ2_S, GgmlType::IQ3_S];

pub fn clamped_swiglu_pipeline_name(dtype: GgmlType) -> Option<String> {
    MOE_GROUPED_CLAMPED_SWIGLU_DTYPES
        .contains(&dtype)
        .then(|| moe_grouped_generic_layout(dtype))
        .flatten()
        .map(|l| {
            format!(
                "kernel_moe_swiglu_clamped_{}_f32_grouped_slots_generic",
                l.suffix
            )
        })
}

/// Actual Half-operand recipe for clamped grouped gate/up. This does not
/// advertise F32 operands: even dense F32/BF16 banks are staged through half.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MoeGroupedClampedRecipe {
    /// Incumbent same-type IQ2_S/IQ3_S fused tile (optionally split n16/n32).
    Fused { pipeline: String },
    /// Down projects gate into inner, then UpClamped reads and updates inner.
    GateThenUpClamped { gate: String, up: String },
}

/// Independently select both native storage types from the generic layouts.
/// Capability queries and encoding share this recipe; no conversion is implied.
pub fn moe_grouped_clamped_recipe(gate: GgmlType, up: GgmlType) -> Option<MoeGroupedClampedRecipe> {
    if gate == up {
        if let Some(pipeline) = clamped_swiglu_pipeline_name(gate) {
            return Some(MoeGroupedClampedRecipe::Fused { pipeline });
        }
    }
    Some(MoeGroupedClampedRecipe::GateThenUpClamped {
        gate: moe_grouped_generic_pipeline_name(MoeGroupedGenericRole::Down, gate)?,
        up: format!(
            "kernel_moe_up_clamped_{}_f32_grouped_slots_clamped",
            moe_grouped_generic_layout(up)?.suffix,
        ),
    })
}

/// Grouped expert-major gate/up with the clamped SwiGLU epilogue:
/// `inner[slot] = silu(min(g, clamp)) * clamp(u, -clamp, clamp)`.
/// Independent native bank dtypes are supported by [`moe_grouped_clamped_recipe`].
/// Fused IQ2_S/IQ3_S pairs retain their existing dispatch; every other pair
/// needs a serial encoder for Down (b_div=topk), then UpClamped in place.
/// No activation or weight scratch is allocated. Operands are Half, sums F32.
/// Each bank's K must align to its generic layout, at least 32; M may tail.
/// Counts must be <= n_tokens, slots unique and in [0, n_tokens * topk).
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_swiglu_clamped_f32_grouped_slots_generic(
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
    const KERNEL: &str = "moe_swiglu_clamped_grouped_slots_generic";
    if !clamp.is_finite() || clamp <= 0.0 {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("clamp must be finite and positive, got {clamp}"),
        });
    }
    let bad = |detail: &str| MetalError::BadShape {
        kernel: KERNEL,
        detail: detail.into(),
    };
    let recipe = moe_grouped_clamped_recipe(w_gate.dtype, w_up.dtype)
        .ok_or_else(|| bad("no clamped grouped recipe for gate/up storage types"))?;
    let slots = n_tokens.checked_mul(topk);
    let x_elements = n_tokens.checked_mul(n_hidden);
    let id_elements = n_expert.checked_mul(n_tokens);
    let inner_elements = slots.and_then(|n| n.checked_mul(n_ffn));
    if [
        Some(n_hidden),
        Some(n_ffn),
        Some(n_expert),
        Some(n_tokens),
        Some(topk),
        slots,
        x_elements,
        id_elements,
        inner_elements,
    ]
    .iter()
    .any(|v| v.is_none_or(|v| v == 0 || v > i32::MAX as usize))
        || Some(x_pack.n_elements() as usize) != x_elements
        || counts.n_elements() as usize != n_expert
        || Some(ids.n_elements() as usize) != id_elements
        || Some(inner.n_elements() as usize) != inner_elements
    {
        return Err(bad(
            "degenerate, overflowing or mismatched x/counts/ids/inner shape",
        ));
    }
    if let MoeGroupedClampedRecipe::GateThenUpClamped { gate, up } = recipe {
        super::checks::require_serial(KERNEL, enc)?;
        // Check BOTH banks, bindings and pipelines before writing gate into
        // inner, so an unsupported/malformed up bank cannot leave half a recipe.
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
        for bank in [w_gate, w_up] {
            super::checks::check_physical(
                KERNEL,
                bank,
                if bank.dtype == GgmlType::F32 { 4 } else { 2 },
                false,
                "expert bank",
            )?;
        }
        for name in [&gate, &up] {
            check_f32x_bindings(
                ctx,
                KERNEL,
                name,
                8192,
                &[w_gate, w_up],
                x_pack,
                counts,
                ids,
                inner,
                &[(slots, "slot count"), (id_elements, "ids")],
            )?;
        }
        dispatch_generic_mm(
            ctx, enc, &gate, 8192, &gate_args, w_gate, x_pack, counts, ids, inner, n_ffn, n_expert,
            n_tokens,
        )?;
        // Buffer 6 is private to UpClamped; existing Down/UpSiluMul ABIs stay
        // unchanged. Serial dispatch order makes the gate projection visible.
        enc.set_bytes(6, &clamp);
        return dispatch_generic_mm(
            ctx, enc, &up, 8192, &up_args, w_up, x_pack, counts, ids, inner, n_ffn, n_expert,
            n_tokens,
        );
    }
    encode_fused_swiglu(
        ctx,
        enc,
        KERNEL,
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
        // Full count range; the kernel compares max_count as int.
        0,
        i32::MAX as u32,
        Some(clamp),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_moe_swiglu_f32_grouped_slots_generic(
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
    encode_moe_swiglu_f32_grouped_slots_generic_range(
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

/// Binding and device checks shared by the F32-operand grouped tiles
/// (`kernels/moe_grouped_f32.metal`): F32 activations, 16-byte aligned
/// (rows are read as `float2x4`); a writable F32 output overlapping no
/// input; I32 counts and ids; 2-byte aligned banks (block fields are read
/// as `half`/`ushort`); every binding inside its buffer; each dimension the
/// kernels index with `int` within `i32`; four 32-wide SIMD groups and
/// `threadgroup_bytes` of threadgroup memory.
#[allow(clippy::too_many_arguments)]
fn check_f32x_bindings(
    ctx: &MetalContext,
    kernel: &'static str,
    name: &str,
    threadgroup_bytes: usize,
    banks: &[&MetalTensor],
    activations: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    dims: &[(Option<usize>, &str)],
) -> Result<(), MetalError> {
    use super::checks::{bad_shape, check_disjoint, check_physical};
    for (tensor, dtype, label) in [
        (activations, GgmlType::F32, "activations"),
        (out, GgmlType::F32, "output"),
        (counts, GgmlType::I32, "counts"),
        (ids, GgmlType::I32, "ids"),
    ] {
        if tensor.dtype != dtype {
            return Err(bad_shape(
                kernel,
                format!("{label} must be {dtype:?}, got {:?}", tensor.dtype),
            ));
        }
    }
    for bank in banks {
        check_physical(kernel, bank, 2, false, "expert bank")?;
    }
    check_physical(kernel, activations, 16, false, "activations")?;
    check_physical(kernel, counts, 4, false, "counts")?;
    check_physical(kernel, ids, 4, false, "ids")?;
    check_physical(kernel, out, 4, true, "output")?;
    let mut inputs = vec![
        (activations, "activations"),
        (counts, "counts"),
        (ids, "ids"),
    ];
    inputs.extend(banks.iter().map(|bank| (*bank, "expert bank")));
    check_disjoint(kernel, out, &inputs)?;
    for (value, label) in dims {
        if value.is_none_or(|v| v > i32::MAX as usize) {
            return Err(bad_shape(kernel, format!("{label} exceeds i32")));
        }
    }
    if ctx.device.maxThreadgroupMemoryLength() < threadgroup_bytes {
        return Err(bad_shape(
            kernel,
            format!("needs {threadgroup_bytes} bytes of threadgroup memory"),
        ));
    }
    let pso = ctx.pipeline(name)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return Err(bad_shape(
            kernel,
            "needs four 32-thread SIMD groups per threadgroup",
        ));
    }
    Ok(())
}

const DOWN_F32X_THREADGROUP_BYTES: usize = 12 * 1024;
const SWIGLU_F32X_THREADGROUP_BYTES: usize = 18 * 1024;

/// Native-storage pipeline with F32 weight/activation operands and accumulation.
/// This is the source of truth for DOWN dtype capability; it does not imply
/// support for fused gate/up or validate geometry, bindings, or the device.
pub fn moe_down_f32x_grouped_slots_pipeline_name(dtype: GgmlType) -> Option<&'static str> {
    match dtype {
        GgmlType::MXFP4 => Some("kernel_moe_down_mxfp4_f32x_grouped_slots"),
        GgmlType::Q2_0 => Some("kernel_moe_down_q2_0_f32x_grouped_slots"),
        GgmlType::Q2_K => Some("kernel_moe_down_q2_K_f32x_grouped_slots"),
        GgmlType::Q3_K => Some("kernel_moe_down_q3_K_f32x_grouped_slots"),
        GgmlType::Q4_K => Some("kernel_moe_down_q4_K_f32x_grouped_slots"),
        GgmlType::Q5_K => Some("kernel_moe_down_q5_K_f32x_grouped_slots"),
        GgmlType::Q6_K => Some("kernel_moe_down_q6_K_f32x_grouped_slots"),
        GgmlType::Q8_0 => Some("kernel_moe_down_q8_0_f32x_grouped_slots"),
        GgmlType::Q4_0 => Some("kernel_moe_down_q4_0_f32x_grouped_slots"),
        GgmlType::Q4_1 => Some("kernel_moe_down_q4_1_f32x_grouped_slots"),
        GgmlType::IQ2_S => Some("kernel_moe_down_iq2_s_f32x_grouped_slots"),
        GgmlType::IQ2_XXS => Some("kernel_moe_down_iq2_xxs_f32x_grouped_slots"),
        GgmlType::IQ2_XS => Some("kernel_moe_down_iq2_xs_f32x_grouped_slots"),
        GgmlType::IQ3_XXS => Some("kernel_moe_down_iq3_xxs_f32x_grouped_slots"),
        GgmlType::IQ3_S => Some("kernel_moe_down_iq3_s_f32x_grouped_slots"),
        GgmlType::IQ4_NL => Some("kernel_moe_down_iq4_nl_f32x_grouped_slots"),
        GgmlType::IQ4_XS => Some("kernel_moe_down_iq4_xs_f32x_grouped_slots"),
        GgmlType::F32 => Some("kernel_moe_down_f32_f32x_grouped_slots"),
        GgmlType::F16 => Some("kernel_moe_down_f16_f32x_grouped_slots"),
        GgmlType::BF16 => Some("kernel_moe_down_bf16_f32x_grouped_slots"),
        _ => None,
    }
}

/// Whether a native F32-operand grouped DOWN pipeline exists for `dtype`.
/// K must be a positive multiple of max(32, storage block elements); use
/// `check_moe_down_f32x_grouped_slots` to preflight a concrete dispatch.
pub fn moe_down_f32x_grouped_slots_supported(dtype: GgmlType) -> bool {
    moe_down_f32x_grouped_slots_pipeline_name(dtype).is_some()
}

fn down_f32x_name(kernel: &'static str, dtype: GgmlType) -> Result<&'static str, MetalError> {
    moe_down_f32x_grouped_slots_pipeline_name(dtype).ok_or_else(|| MetalError::BadShape {
        kernel,
        detail: format!("no F32-operand grouped down for {dtype:?}"),
    })
}

/// Every check [`encode_moe_down_f32x_grouped_slots`] makes before encoding,
/// without encoding (so a composition can refuse before its first dispatch).
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_moe_down_f32x_grouped_slots(
    ctx: &MetalContext,
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
    const KERNEL: &str = "moe_down_f32x_grouped_slots";
    let name = down_f32x_name(KERNEL, weight.dtype)?;
    // Bound every dimension and product the shared argument builder
    // computes unchecked, before calling it.
    let within_i32 = |v: Option<usize>| v.is_some_and(|v| v > 0 && v <= i32::MAX as usize);
    let slots = (n_out > 0).then(|| out.n_elements() as usize / n_out);
    if !within_i32(Some(n_in))
        || !within_i32(Some(n_out))
        || !within_i32(Some(n_expert))
        || !within_i32(Some(n_tokens))
        || !within_i32(n_expert.checked_mul(n_tokens))
        || !within_i32(slots)
        || slots.and_then(|s| s.checked_mul(n_in)).is_none()
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "dimensions n_in={n_in} n_out={n_out} n_expert={n_expert} n_tokens={n_tokens} \
                 must be positive with every product within i32"
            ),
        });
    }
    mm_args_with_layout(
        KERNEL,
        f32x_layout_for(KERNEL, weight, n_in, n_out, n_expert)?,
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
        n_tokens as u32,
    )?;
    // The shared check covers half/ushort bank fields. Native F32 additionally
    // loads scalar floats, so its base must be aligned to four bytes.
    if weight.dtype == GgmlType::F32 {
        super::checks::check_physical(KERNEL, weight, 4, false, "F32 expert bank")?;
    }
    check_f32x_bindings(
        ctx,
        KERNEL,
        name,
        DOWN_F32X_THREADGROUP_BYTES,
        &[weight],
        inner,
        counts,
        ids,
        out,
        &[
            (Some(n_in), "n_in"),
            (Some(n_out), "n_out"),
            (slots, "slot count"),
            (n_expert.checked_mul(n_tokens), "ids"),
        ],
    )
}

/// F32-operand grouped down projection (`kernels/moe_grouped_f32.metal`,
/// map #12 accuracy lane): the contract of
/// [`encode_moe_down_f32_grouped_slots_generic`] with weights dequantized to
/// F32 (ggml's order), unrounded activations and F32 accumulation, for every
/// dtype accepted by [`moe_down_f32x_grouped_slots_supported`]. No repacking,
/// materialization, or scratch allocation. K is aligned to max(32, storage
/// block elements); M may have a partial tile. Counts outside 0..=n_tokens
/// and slot ids outside the output's slot range are skipped. Each slot's outputs
/// depend only on its own activation row: not on its expert's count or place in the
/// bucket. Each output slot must have a unique bucket owner. MXFP4 exponent-0/1
/// blocks use an edge-only, scaled F32 compensated reduction, merged with the
/// ordinary result before unscaling. Ordinary blocks retain the matrix reduction.
/// Qualification retains relative RMS <= 1e-5 and max error <= 2e-5 times the
/// reference peak for finite, normal-result witnesses; this is not a uniform
/// relative-error guarantee for arbitrary ill-conditioned dots. Ordinary-path
/// FTZ/overflow and subnormal final outputs are not repaired by the edge epilogue.
/// Bindings and device are checked before encoding
/// ([`check_moe_down_f32x_grouped_slots`]). 12 KiB of threadgroup memory.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_down_f32x_grouped_slots(
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
    const KERNEL: &str = "moe_down_f32x_grouped_slots";
    check_moe_down_f32x_grouped_slots(
        ctx, weight, inner, counts, ids, out, n_in, n_out, n_expert, n_tokens,
    )?;
    let args = mm_args_with_layout(
        KERNEL,
        f32x_layout_for(KERNEL, weight, n_in, n_out, n_expert)?,
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
        n_tokens as u32,
    )?;
    dispatch_generic_mm(
        ctx,
        enc,
        down_f32x_name(KERNEL, weight.dtype)?,
        DOWN_F32X_THREADGROUP_BYTES,
        &args,
        weight,
        inner,
        counts,
        ids,
        out,
        n_out,
        n_expert,
        n_tokens,
    )
}

/// Storage types with a fused F32-operand clamped gate/up matrix kernel.
pub fn moe_swiglu_clamped_f32x_grouped_slots_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::MXFP4
            | GgmlType::IQ2_S
            | GgmlType::IQ2_XXS
            | GgmlType::IQ2_XS
            | GgmlType::IQ3_S
            | GgmlType::IQ3_XXS
            | GgmlType::Q3_K
    )
}

/// Checks without encoding; returns the pipeline name.
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_moe_swiglu_clamped_f32x_grouped_slots(
    ctx: &MetalContext,
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
) -> Result<&'static str, MetalError> {
    const KERNEL: &str = "moe_swiglu_clamped_f32x_grouped_slots";
    let bad = |detail: String| MetalError::BadShape {
        kernel: KERNEL,
        detail,
    };
    if !clamp.is_finite() || clamp <= 0.0 {
        return Err(bad(format!(
            "clamp must be finite and positive, got {clamp}"
        )));
    }
    if w_gate.dtype != w_up.dtype {
        return Err(bad(format!(
            "gate/up dtypes differ: {:?}/{:?}",
            w_gate.dtype, w_up.dtype
        )));
    }
    let name = match w_gate.dtype {
        GgmlType::MXFP4 => "kernel_moe_swiglu_clamped_mxfp4_f32x_grouped_slots",
        GgmlType::IQ2_S => "kernel_moe_swiglu_clamped_iq2_s_f32x_grouped_slots",
        GgmlType::IQ2_XXS => "kernel_moe_swiglu_clamped_iq2_xxs_f32x_grouped_slots",
        GgmlType::IQ2_XS => "kernel_moe_swiglu_clamped_iq2_xs_f32x_grouped_slots",
        GgmlType::IQ3_S => "kernel_moe_swiglu_clamped_iq3_s_f32x_grouped_slots",
        GgmlType::IQ3_XXS => "kernel_moe_swiglu_clamped_iq3_xxs_f32x_grouped_slots",
        GgmlType::Q3_K => "kernel_moe_swiglu_clamped_q3_K_f32x_grouped_slots",
        other => return Err(bad(format!("no F32-operand grouped SwiGLU for {other:?}"))),
    };
    let slots = n_tokens.checked_mul(topk);
    let elements = |a: Option<usize>, b: usize| a.and_then(|a| a.checked_mul(b));
    if topk == 0
        || n_ffn == 0
        || n_tokens == 0
        || n_expert == 0
        || Some(x_pack.n_elements() as usize) != elements(Some(n_tokens), n_hidden)
        || counts.n_elements() as usize != n_expert
        || Some(ids.n_elements() as usize) != elements(Some(n_expert), n_tokens)
        || Some(inner.n_elements() as usize) != elements(slots, n_ffn)
    {
        return Err(bad(
            "degenerate shape or x/counts/ids/inner size mismatch".into()
        ));
    }
    f32x_layout_for(KERNEL, w_gate, n_hidden, n_ffn, n_expert)?;
    f32x_layout_for(KERNEL, w_up, n_hidden, n_ffn, n_expert)?;
    check_f32x_bindings(
        ctx,
        KERNEL,
        name,
        SWIGLU_F32X_THREADGROUP_BYTES,
        &[w_gate, w_up],
        x_pack,
        counts,
        ids,
        inner,
        &[
            (Some(n_hidden), "n_hidden"),
            (Some(n_ffn), "n_ffn"),
            (slots, "slot count"),
            (n_expert.checked_mul(n_tokens), "ids"),
        ],
    )?;
    Ok(name)
}

/// F32-operand grouped gate/up with the clamped SwiGLU epilogue
/// (`kernels/moe_grouped_f32.metal`, map #12 accuracy lane): the contract of
/// [`encode_moe_swiglu_clamped_f32_grouped_slots_generic`] with weights
/// dequantized to F32, unrounded activations and F32 accumulation, for the
/// same-type gate/up pairs MXFP4, IQ2_S, IQ2_XXS, IQ2_XS, IQ3_S, IQ3_XXS and
/// Q3_K. Counts outside
/// 0..=n_tokens and invalid slot IDs are skipped. Bindings and device are checked
/// before encoding ([`check_moe_swiglu_clamped_f32x_grouped_slots`]).
/// 18 KiB of threadgroup memory.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_swiglu_clamped_f32x_grouped_slots(
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
    const KERNEL: &str = "moe_swiglu_clamped_f32x_grouped_slots";
    let name = check_moe_swiglu_clamped_f32x_grouped_slots(
        ctx, w_gate, w_up, x_pack, counts, ids, inner, n_hidden, n_ffn, n_expert, topk, n_tokens,
        clamp,
    )?;
    let l = f32x_layout_for(KERNEL, w_gate, n_hidden, n_ffn, n_expert)?;
    dispatch_fused_swiglu(
        ctx,
        enc,
        name,
        SWIGLU_F32X_THREADGROUP_BYTES,
        16,
        l,
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
        n_tokens as u32,
        Some(clamp),
    )
}

/// Gate/up dtypes with a 32-token clamped grouped SwiGLU tile
/// (`kernels/moe_grouped_n32.metal`).
fn clamped_swiglu_n32_pipeline_name(dtype: GgmlType) -> Option<&'static str> {
    match dtype {
        GgmlType::IQ2_S => Some("kernel_moe_swiglu_clamped_iq2_s_f32_grouped_slots_n32"),
        GgmlType::IQ3_S => Some("kernel_moe_swiglu_clamped_iq3_s_f32_grouped_slots_n32"),
        _ => None,
    }
}

/// [`encode_moe_swiglu_clamped_f32_grouped_slots_generic`] with buckets of at
/// least `wide_min_count` slots on the 32-token tile and the rest on the
/// 16-token tile: two dispatches over disjoint, exhaustive count ranges.
/// The 32-token tile keeps the 16-token tile's dequantization and per-output
/// matrix-multiply sequence, so each slot's output is bitwise the same
/// (qualification-tested); a bucket's weight tiles are dequantized once per
/// 32 slots instead of 16. Dtypes without the wide tile, and spans shorter
/// than `wide_min_count` (no bucket can reach it), use the generic recipe
/// (fused n16 or Down + UpClamped). Both dispatches' bindings and pipelines are checked before either
/// is encoded.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_swiglu_clamped_f32_grouped_slots_split(
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
    wide_min_count: u32,
) -> Result<(), MetalError> {
    const KERNEL: &str = "moe_swiglu_clamped_grouped_slots_split";
    // A span shorter than the wide threshold has no bucket that large (a
    // bucket holds at most `n_tokens` slots): the incumbent alone.
    let wide = clamped_swiglu_n32_pipeline_name(w_gate.dtype)
        .filter(|_| w_gate.dtype == w_up.dtype && n_tokens >= wide_min_count as usize);
    let Some(wide) = wide else {
        return encode_moe_swiglu_clamped_f32_grouped_slots_generic(
            ctx, enc, w_gate, w_up, x_pack, counts, ids, inner, n_hidden, n_ffn, n_expert, topk,
            n_tokens, clamp,
        );
    };
    let bad = |detail: String| MetalError::BadShape {
        kernel: KERNEL,
        detail,
    };
    if wide_min_count == 0 || wide_min_count > i32::MAX as u32 {
        return Err(bad(format!(
            "wide_min_count {wide_min_count} outside 1..=i32::MAX"
        )));
    }
    if !clamp.is_finite() || clamp <= 0.0 || w_gate.dtype != w_up.dtype {
        return Err(bad(format!(
            "clamp {clamp}, gate/up {:?}/{:?}",
            w_gate.dtype, w_up.dtype
        )));
    }
    let slots = n_tokens.checked_mul(topk);
    let elements = |a: Option<usize>, b: usize| a.and_then(|a| a.checked_mul(b));
    if topk == 0
        || n_ffn == 0
        || n_tokens == 0
        || n_expert == 0
        || Some(x_pack.n_elements() as usize) != elements(Some(n_tokens), n_hidden)
        || counts.n_elements() as usize != n_expert
        || Some(ids.n_elements() as usize) != elements(Some(n_expert), n_tokens)
        || Some(inner.n_elements() as usize) != elements(slots, n_ffn)
    {
        return Err(bad(
            "degenerate shape or x/counts/ids/inner size mismatch".into()
        ));
    }
    let narrow = clamped_swiglu_pipeline_name(w_gate.dtype)
        .ok_or_else(|| bad(format!("no clamped grouped SwiGLU for {:?}", w_gate.dtype)))?;
    let l = generic_layout_for(KERNEL, w_gate, n_hidden, n_ffn, n_expert)?;
    generic_layout_for(KERNEL, w_up, n_hidden, n_ffn, n_expert)?;
    // Both pipelines and every binding checked before either dispatch:
    // F32 activations 16-byte aligned, a writable output aliasing no input,
    // I32 counts and ids, int-indexed dimensions within i32, 16 KiB of
    // threadgroup memory, four 32-wide SIMD groups.
    for name in [narrow.as_str(), wide] {
        check_f32x_bindings(
            ctx,
            KERNEL,
            name,
            16384,
            &[w_gate, w_up],
            x_pack,
            counts,
            ids,
            inner,
            &[
                (Some(n_hidden), "n_hidden"),
                (Some(n_ffn), "n_ffn"),
                (slots, "slot count"),
                (n_expert.checked_mul(n_tokens), "ids"),
            ],
        )?;
    }
    for (name, tokens_per_tile, min_count, max_count) in [
        (narrow.as_str(), 16, 0, wide_min_count - 1),
        (wide, 32, wide_min_count, i32::MAX as u32),
    ] {
        dispatch_fused_swiglu(
            ctx,
            enc,
            name,
            16384,
            tokens_per_tile,
            l,
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
            min_count,
            max_count,
            Some(clamp),
        )?;
    }
    Ok(())
}

mod tensor;
pub use tensor::*;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod clamped_tests {
    use super::*;
    use crate::metal::test_support::{
        assert_offset_guards, dequant_expert, metal_test_context, offset_tensor,
        tensor_backing_bytes, tensor_f32_at_offset,
    };

    const METAL: &str = include_str!("../../../../kernels/moe.metal");

    #[test]
    fn clamped_recipes_cover_independent_layouts_and_real_entry_points() {
        for gate in MOE_GROUPED_GENERIC_LAYOUTS {
            for up in MOE_GROUPED_GENERIC_LAYOUTS {
                let recipe = moe_grouped_clamped_recipe(gate.dtype, up.dtype).unwrap();
                assert!(grouped_clamped_gate_up_supported(gate.dtype, up.dtype));
                let names = match recipe {
                    MoeGroupedClampedRecipe::Fused { pipeline } => {
                        assert_eq!(gate.dtype, up.dtype);
                        assert!(MOE_GROUPED_CLAMPED_SWIGLU_DTYPES.contains(&gate.dtype));
                        vec![pipeline]
                    }
                    MoeGroupedClampedRecipe::GateThenUpClamped { gate: g, up: u } => {
                        assert!(g.contains(&format!("down_{}_", gate.suffix)));
                        assert!(u.contains(&format!("up_clamped_{}_", up.suffix)));
                        let entry = METAL
                            .lines()
                            .find(|line| line.contains(&format!("[[host_name(\"{u}\")]]")))
                            .unwrap();
                        assert!(entry.contains(&format!("qt_dequantize_{}, 2>", up.suffix)));
                        vec![g, u]
                    }
                };
                for name in names {
                    assert_eq!(
                        METAL.matches(&format!("[[host_name(\"{name}\")]]")).count(),
                        1
                    );
                }
            }
        }
        for unsupported in [GgmlType::IQ1_S, GgmlType::IQ1_M, GgmlType::Unknown] {
            assert!(moe_grouped_clamped_recipe(unsupported, GgmlType::Q2_K).is_none());
            assert!(moe_grouped_clamped_recipe(GgmlType::Q2_K, unsupported).is_none());
        }
    }

    // Arbitrary valid quant payloads; the independent CPU codec supplies the
    // decoded weights. No Metal decoder tables or indexing appear in the oracle.
    fn bank_bytes(dtype: GgmlType, k: usize, m: usize, experts: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as u8
        };
        if dtype == GgmlType::BF16 {
            return (0..k * m * experts)
                .flat_map(|_| {
                    half::bf16::from_f32((f32::from(next()) - 128.0) / 256.0).to_le_bytes()
                })
                .collect();
        }
        let (block, bytes) = dtype.storage_layout().unwrap();
        let scales: &[usize] = match dtype {
            GgmlType::Q2_K => &[80, 82],
            GgmlType::Q3_K => &[108],
            GgmlType::Q6_K => &[208],
            GgmlType::IQ2_S | GgmlType::IQ3_S => &[0],
            _ => unreachable!(),
        };
        let mut bank = Vec::new();
        for _ in 0..k * m * experts / block as usize {
            let mut data: Vec<u8> = (0..bytes).map(|_| next()).collect();
            for &offset in scales {
                let d = (1.0 + f32::from(next() % 7)) / 2048.0;
                data[offset..offset + 2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
            }
            bank.extend(data);
        }
        bank
    }

    fn view(ctx: &MetalContext, dtype: GgmlType, data: &[u8], shape: Vec<u64>) -> MetalTensor {
        offset_tensor(ctx, 32, data, 32, shape, dtype)
    }

    fn half_dot(w: &[f32], x: &[f32]) -> f32 {
        w.iter()
            .zip(x)
            .map(|(&a, &b)| {
                f64::from(half::f16::from_f32(a).to_f32())
                    * f64::from(half::f16::from_f32(b).to_f32())
            })
            .sum::<f64>() as f32
    }

    #[allow(clippy::too_many_arguments)]
    fn run_oracle(
        ctx: &MetalContext,
        gate_type: GgmlType,
        up_type: GgmlType,
        gb: &[u8],
        ub: &[u8],
        k: usize,
        m: usize,
        rows: usize,
        experts: usize,
        routes: &[i32],
        x: &[f32],
        clamp: f32,
    ) {
        let topk = routes.len() / rows;
        let gate = view(ctx, gate_type, gb, vec![k as u64, m as u64, experts as u64]);
        let up = view(ctx, up_type, ub, vec![k as u64, m as u64, experts as u64]);
        let input = view(
            ctx,
            GgmlType::F32,
            bytemuck::cast_slice(x),
            vec![x.len() as u64],
        );
        let route = view(
            ctx,
            GgmlType::I32,
            bytemuck::cast_slice(routes),
            vec![routes.len() as u64],
        );
        let counts = view(
            ctx,
            GgmlType::I32,
            bytemuck::cast_slice(&vec![0i32; experts]),
            vec![experts as u64],
        );
        let slots = view(
            ctx,
            GgmlType::I32,
            bytemuck::cast_slice(&vec![-1i32; experts * rows]),
            vec![(experts * rows) as u64],
        );
        let inner = view(
            ctx,
            GgmlType::F32,
            bytemuck::cast_slice(&vec![f32::NAN; routes.len() * m]),
            vec![(routes.len() * m) as u64],
        );
        for e in 0..experts {
            assert!(routes.iter().filter(|&&id| id == e as i32).count() <= rows);
        }
        let inputs = [&gate, &up, &input, &route];
        let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
        dispatch_census_begin();
        one_shot(ctx, |enc| {
            encode_moe_route_bucket_slots_f32(
                ctx, enc, &route, &counts, &slots, experts, rows, topk,
            )?;
            // Always exercise the public split dispatcher, including mixed IQ
            // pairs above its 17-slot threshold (must select the two-pass recipe).
            encode_moe_swiglu_clamped_f32_grouped_slots_split(
                ctx, enc, &gate, &up, &input, &counts, &slots, &inner, k, m, experts, topk, rows,
                clamp, 17,
            )
        })
        .unwrap();
        let dispatches = dispatch_census_take();
        let MoeGroupedClampedRecipe::GateThenUpClamped { gate: g, up: u } =
            moe_grouped_clamped_recipe(gate_type, up_type).unwrap()
        else {
            panic!("two-pass fixture")
        };
        assert_eq!(dispatches.len(), 3); // bucket, Down, UpClamped; no activation pass
        assert_eq!(dispatches[1].kernel, g);
        assert_eq!(dispatches[2].kernel, u);
        let decoded_g: Vec<_> = (0..experts)
            .map(|e| dequant_expert(gb, gate_type, k, m, e))
            .collect();
        let decoded_u: Vec<_> = (0..experts)
            .map(|e| dequant_expert(ub, up_type, k, m, e))
            .collect();
        let actual = tensor_f32_at_offset(&inner);
        for (slot, &expert) in routes.iter().enumerate() {
            let e = expert as usize;
            let x_row = &x[slot / topk * k..][..k];
            for channel in 0..m {
                let g = half_dot(&decoded_g[e][channel * k..][..k], x_row).min(clamp);
                let u = half_dot(&decoded_u[e][channel * k..][..k], x_row).clamp(-clamp, clamp);
                // f64 nonlinear oracle; Half operand rounding is explicit.
                let gf = f64::from(g);
                let expected = (gf / (1.0 + (-gf).exp()) * f64::from(u)) as f32;
                let got = actual[slot * m + channel];
                assert!(got.is_finite() && expected.is_finite());
                assert!(
                    (got - expected).abs() <= 0.003 * (1.0 + expected.abs()),
                    "{gate_type:?}/{up_type:?} rows={rows} M={m} slot={slot} channel={channel}: {got} != {expected}"
                );
            }
        }
        for t in [&counts, &slots, &inner] {
            assert_offset_guards(t, 32, 32);
        }
        for (t, bytes) in inputs.iter().zip(before) {
            assert_eq!(tensor_backing_bytes(t), bytes);
        }
    }

    #[test]
    fn mixed_clamped_independent_oracle_tails_routes_and_offsets() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let pairs = [
            (GgmlType::Q2_K, GgmlType::Q3_K),
            (GgmlType::IQ2_S, GgmlType::IQ3_S),
            (GgmlType::Q6_K, GgmlType::BF16),
        ];
        for (a, b) in pairs {
            for (g, u) in [(a, b), (b, a)] {
                for rows in [1, 3, 17, 129] {
                    for m in [1, 3, 17, 129] {
                        let (k, experts) = (256, 4);
                        let gb = bank_bytes(g, k, m, experts, 17);
                        let ub = bank_bytes(u, k, m, experts, 93);
                        // Expert 0 fills a whole bucket. The second route cycles
                        // among other experts; at rows >= 3 swap two routes to
                        // include a duplicate expert in a token without exceeding
                        // the existing rows-per-bucket storage contract.
                        let mut routes: Vec<i32> =
                            (0..rows).flat_map(|r| [0, 1 + (r % 3) as i32]).collect();
                        if rows >= 3 {
                            routes.swap(1, 2);
                        }
                        let x: Vec<f32> = (0..rows * k)
                            .map(|i| ((i * 17 % 83) as f32 - 41.0) / 64.0)
                            .collect();
                        run_oracle(&ctx, g, u, &gb, &ub, k, m, rows, experts, &routes, &x, 0.37);
                    }
                }
            }
        }
    }

    #[test]
    fn clamped_boundary_is_finite_and_gate_is_only_upper_clamped() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let values = [
            -65504.0, -80.0, -2.25, -2.0, -1.75, -0.0, 1.75, 2.0, 2.25, 80.0, 65504.0,
        ];
        let k = 32;
        let m = values.len() * values.len();
        let mut g = vec![0.0f32; k * m];
        let mut u = vec![0.0f32; k * m];
        for (i, &gv) in values.iter().enumerate() {
            for (j, &uv) in values.iter().enumerate() {
                g[(i * values.len() + j) * k] = gv;
                u[(i * values.len() + j) * k] = uv;
            }
        }
        let mut x = vec![0.0; k];
        x[0] = 1.0;
        run_oracle(
            &ctx,
            GgmlType::F32,
            GgmlType::F32,
            bytemuck::cast_slice(&g),
            bytemuck::cast_slice(&u),
            k,
            m,
            1,
            1,
            &[0],
            &x,
            2.0,
        );
    }

    #[test]
    fn clamped_two_pass_rejects_concurrency_and_bad_up_before_dispatch() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let bank = view(
            &ctx,
            GgmlType::F32,
            bytemuck::cast_slice(&[0.0f32; 32]),
            vec![32, 1, 1],
        );
        let x = view(
            &ctx,
            GgmlType::F32,
            bytemuck::cast_slice(&[0.0f32; 32]),
            vec![32],
        );
        let counts = view(&ctx, GgmlType::I32, bytemuck::cast_slice(&[1i32]), vec![1]);
        let ids = view(&ctx, GgmlType::I32, bytemuck::cast_slice(&[0i32]), vec![1]);
        let inner = view(
            &ctx,
            GgmlType::F32,
            bytemuck::cast_slice(&[123.0f32]),
            vec![1],
        );
        for case in 0..7 {
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = if case == 0 {
                KernelEncoder::begin_concurrent(&command)
            } else {
                KernelEncoder::begin(&command)
            };
            let mut up = bank.clone();
            let clamp = match case {
                3 => 0.0,
                4 => -1.0,
                5 => f32::NAN,
                6 => f32::INFINITY,
                _ => 2.0,
            };
            if case == 1 {
                up.shape = vec![1];
            }
            if case == 2 {
                up.offset += 2;
            }
            dispatch_census_begin();
            let result = encode_moe_swiglu_clamped_f32_grouped_slots_generic(
                &ctx, &enc, &bank, &up, &x, &counts, &ids, &inner, 32, 1, 1, 1, 1, clamp,
            );
            enc.end();
            assert!(result.is_err(), "case {case}");
            assert!(
                dispatch_census_take().is_empty(),
                "case {case}: partial encoding"
            );
            assert_eq!(tensor_f32_at_offset(&inner), [123.0]);
        }
    }
}
