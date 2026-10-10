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
//! activations in half and accumulate in f32. Quantized weights fit half by
//! construction (f16 block scales); F32/BF16 weights or activations beyond
//! +-65504 would overflow, a limit every quantized MMA prefill path shares
//! for activations and no trained weight approaches.

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
        ctx, enc, &name, 16384, l, w_gate, w_up, x_pack, counts, ids, inner, n_hidden, n_ffn,
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

/// Grouped expert-major fused gate/up with the clamped SwiGLU epilogue
/// (DeepSeek V4 / GLM-5.3): `inner[slot] = silu(min(g, clamp)) *
/// clamp(u, -clamp, clamp)` for every routed slot. Same layouts as
/// [`encode_moe_swiglu_f32_grouped_slots_generic`]; gate and up must share a
/// dtype listed in [`MOE_GROUPED_CLAMPED_SWIGLU_DTYPES`].
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
    if w_gate.dtype != w_up.dtype {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("gate/up dtypes differ: {:?}/{:?}", w_gate.dtype, w_up.dtype),
        });
    }
    if topk == 0
        || n_ffn == 0
        || n_tokens == 0
        || n_expert == 0
        || x_pack.n_elements() as usize != n_tokens * n_hidden
        || counts.n_elements() as usize != n_expert
        || ids.n_elements() as usize != n_expert * n_tokens
        || inner.n_elements() as usize != n_tokens * topk * n_ffn
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "degenerate shape or x/counts/ids/inner size mismatch".into(),
        });
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

fn down_f32x_name(kernel: &'static str, dtype: GgmlType) -> Result<&'static str, MetalError> {
    match dtype {
        GgmlType::IQ3_S => Ok("kernel_moe_down_iq3_s_f32x_grouped_slots"),
        GgmlType::IQ4_XS => Ok("kernel_moe_down_iq4_xs_f32x_grouped_slots"),
        other => Err(MetalError::BadShape {
            kernel,
            detail: format!("no F32-operand grouped down for {other:?}"),
        }),
    }
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
    generic_mm_args(
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
/// F32 (ggml's order), unrounded activations and F32 accumulation, for the
/// GLM-5.3 down types IQ3_S and IQ4_XS. Each slot's outputs depend only on
/// its own activation row: not on its expert's count nor on its place in the
/// bucket. Bindings and device are checked before encoding
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

/// Every check [`encode_moe_swiglu_clamped_f32x_grouped_slots`] makes before
/// encoding, without encoding; returns the pipeline name.
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
        GgmlType::IQ2_S => "kernel_moe_swiglu_clamped_iq2_s_f32x_grouped_slots",
        GgmlType::IQ3_S => "kernel_moe_swiglu_clamped_iq3_s_f32x_grouped_slots",
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
    generic_layout_for(KERNEL, w_gate, n_hidden, n_ffn, n_expert)?;
    generic_layout_for(KERNEL, w_up, n_hidden, n_ffn, n_expert)?;
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
/// GLM-5.3 gate/up types IQ2_S and IQ3_S. Bindings and device are checked
/// before encoding ([`check_moe_swiglu_clamped_f32x_grouped_slots`]). 18 KiB
/// of threadgroup memory.
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
    let l = generic_layout_for(KERNEL, w_gate, n_hidden, n_ffn, n_expert)?;
    dispatch_fused_swiglu(
        ctx,
        enc,
        name,
        SWIGLU_F32X_THREADGROUP_BYTES,
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
        i32::MAX as u32,
        Some(clamp),
    )
}

#[cfg(test)]
mod tests;
