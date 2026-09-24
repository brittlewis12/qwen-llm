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
    /// Logical elements per tile block (256 for K/I-quants, 32 otherwise).
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
    let name = moe_grouped_generic_pipeline_name(role, weight.dtype)
        .expect("layout lookup succeeded above");
    let pso = ctx.pipeline(&name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &GenericMmArgs {
            m: n_out as u32,
            n: n_tokens as u32,
            k: n_in as u32,
            nb01: ((n_in / l.block_elems) * l.block_bytes) as u32,
            stride_b: n_in as u32,
            min_count,
            max_count,
            b_div: b_div as u32,
            slot_limit: slot_count as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, src);
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

    let l = generic_layout_for(KERNEL, w_gate, n_hidden, n_ffn, n_expert)?;
    generic_layout_for(KERNEL, w_up, n_hidden, n_ffn, n_expert)?;
    let name = moe_grouped_generic_pipeline_name(MoeGroupedGenericRole::Swiglu, w_gate.dtype)
        .expect("layout lookup succeeded above");
    let pso = ctx.pipeline(&name)?;
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

#[cfg(test)]
mod tests;
