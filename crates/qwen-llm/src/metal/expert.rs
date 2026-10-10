//! Routed-expert decode projections over every selected slot in one dispatch,
//! shared by DeepSeek V4 and GLM-5.3-Flash.
//!
//! Banks are `[n_in, n_out, experts]` (contiguous per-expert matrices). Expert
//! ids (I32 `[top_k]`) and route status (I32 `[1]`) come from
//! [`encode_route_learned`]. Slot activations are slot-strided: gate/up outputs
//! and down inputs are `[ffn, top_k]`, down outputs `[hidden, top_k]`. With an
//! unready route the DS4 kernels write zero rows; the IQ4_XS kernel leaves its
//! output unspecified, so callers must check route status before using results
//! (both families validate route records after each token's command).

use super::checks::{
    bad_shape, check_alignment, check_disjoint, check_expert_bank, check_tensor, require_serial,
    to_u32,
};
use super::*;

/// Gate/up dtypes with a fused all-slot `silu(min(g, c)) * clamp(u, -c, c)`.
const GATE_UP_KERNELS: &[(GgmlType, &str)] = &[
    (
        GgmlType::IQ2_XS,
        "kernel_deepseek_v4_all_slots_swiglu_iq2_xs_f32_fast",
    ),
    (
        GgmlType::IQ2_S,
        "kernel_deepseek_v4_all_slots_swiglu_iq2_s_f32_fast",
    ),
    (
        GgmlType::IQ3_XXS,
        "kernel_deepseek_v4_all_slots_swiglu_iq3_xxs_f32_fast",
    ),
    (
        GgmlType::IQ3_S,
        "kernel_deepseek_v4_all_slots_swiglu_iq3_s_f32_fast",
    ),
    (
        GgmlType::Q3_K,
        "kernel_deepseek_v4_all_slots_swiglu_q3_K_f32",
    ),
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AllSlotsArgs {
    n_in: u32,
    n_out: u32,
    n_expert: u32,
    top_k: u32,
    clamp: f32,
}

fn check_route(
    kernel: &'static str,
    ids: &MetalTensor,
    status: &MetalTensor,
    top_k: usize,
) -> Result<(), MetalError> {
    if top_k == 0 || top_k > ROUTE_MAX_TOP_K {
        return Err(bad_shape(
            kernel,
            format!("top-k {top_k} outside 1..={ROUTE_MAX_TOP_K}"),
        ));
    }
    check_tensor(
        kernel,
        ids,
        GgmlType::I32,
        &[top_k as u64],
        false,
        "expert ids",
    )?;
    check_tensor(kernel, status, GgmlType::I32, &[1], false, "route status")
}

fn pipeline_for(
    ctx: &MetalContext,
    kernel: &'static str,
    threads: usize,
) -> Result<Pipeline, MetalError> {
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < threads {
        return Err(bad_shape(
            kernel,
            format!(
                "needs SIMD width 32 and {threads} threads, pipeline has {} and {}",
                pso.threadExecutionWidth(),
                pso.maxTotalThreadsPerThreadgroup()
            ),
        ));
    }
    Ok(pso)
}

/// `output[:, s] = silu(min(gate_s x, clamp)) * clamp(up_s x, -clamp, clamp)` for
/// each routed slot `s`, where `gate_s`/`up_s` are slot `s`'s expert matrices.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_gate_up_swiglu(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
    clamp: f32,
) -> Result<(), MetalError> {
    const K: &str = "all_slots_gate_up_swiglu";
    require_serial(K, enc)?;
    if !clamp.is_finite() || clamp <= 0.0 {
        return Err(bad_shape(K, "clamp must be finite and positive"));
    }
    check_route(K, ids, status, top_k)?;
    check_expert_bank(K, gate_bank, n_in, n_out, experts, "gate bank")?;
    check_expert_bank(K, up_bank, n_in, n_out, experts, "up bank")?;
    check_tensor(K, input, GgmlType::F32, &[n_in as u64], false, "input")?;
    let out_shape = [n_out as u64, top_k as u64];
    check_tensor(K, output, GgmlType::F32, &out_shape, true, "output")?;
    check_disjoint(
        K,
        output,
        &[
            (input, "input"),
            (gate_bank, "gate bank"),
            (up_bank, "up bank"),
            (ids, "expert ids"),
            (status, "route status"),
        ],
    )?;
    if gate_bank.dtype != up_bank.dtype {
        return Err(bad_shape(K, "gate and up banks must share a dtype"));
    }
    let kernel = GATE_UP_KERNELS
        .iter()
        .find(|(dtype, _)| *dtype == gate_bank.dtype)
        .map(|(_, kernel)| *kernel)
        .ok_or_else(|| bad_shape(K, format!("no all-slot kernel for {:?}", gate_bank.dtype)))?;
    if !n_in.is_multiple_of(256) {
        return Err(bad_shape(
            K,
            format!("input width {n_in} is not a multiple of 256"),
        ));
    }
    let pso = pipeline_for(ctx, kernel, 64)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AllSlotsArgs {
            n_in: to_u32(K, n_in, "n_in")?,
            n_out: to_u32(K, n_out, "n_out")?,
            n_expert: to_u32(K, experts, "experts")?,
            top_k: top_k as u32,
            clamp,
        },
    );
    enc.set_tensor(1, gate_bank);
    enc.set_tensor(2, up_bank);
    enc.set_tensor(3, input);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, status);
    enc.set_tensor(6, output);
    enc.set_threadgroup_memory(0, 16 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(8),
            height: top_k,
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

/// `output[:, s] = down_s input[:, s]` for each routed slot `s`.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_down(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
) -> Result<(), MetalError> {
    const K: &str = "all_slots_down";
    require_serial(K, enc)?;
    check_route(K, ids, status, top_k)?;
    check_expert_bank(K, bank, n_in, n_out, experts, "down bank")?;
    let in_shape = [n_in as u64, top_k as u64];
    let out_shape = [n_out as u64, top_k as u64];
    check_tensor(K, input, GgmlType::F32, &in_shape, false, "input")?;
    check_tensor(K, output, GgmlType::F32, &out_shape, true, "output")?;
    check_disjoint(
        K,
        output,
        &[
            (input, "input"),
            (bank, "down bank"),
            (ids, "expert ids"),
            (status, "route status"),
        ],
    )?;
    let args = AllSlotsArgs {
        n_in: to_u32(K, n_in, "n_in")?,
        n_out: to_u32(K, n_out, "n_out")?,
        n_expert: to_u32(K, experts, "experts")?,
        top_k: top_k as u32,
        clamp: 0.0,
    };
    if bank.dtype == GgmlType::IQ4_XS {
        // Qwen's all-slot IQ4_XS down: no status binding, 32-entry LUT, and
        // slot inputs read as float4.
        check_alignment(K, input, 16, "input")?;
        if !n_in.is_multiple_of(256) {
            return Err(bad_shape(
                K,
                format!("input width {n_in} is not a multiple of 256"),
            ));
        }
        let pso = pipeline_for(ctx, "kernel_moe_down_iq4_xs_f32_fast", 64)?;
        enc.set_pipeline(&pso);
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Iq4XsArgs {
            n_in: u32,
            n_out: u32,
            n_expert: u32,
            top_k: u32,
        }
        enc.set_bytes(
            0,
            &Iq4XsArgs {
                n_in: args.n_in,
                n_out: args.n_out,
                n_expert: args.n_expert,
                top_k: args.top_k,
            },
        );
        enc.set_tensor(1, bank);
        enc.set_tensor(2, input);
        enc.set_tensor(3, ids);
        enc.set_tensor(4, output);
        enc.set_threadgroup_memory(0, 32 * std::mem::size_of::<f32>());
        enc.dispatch(
            MTLSize {
                width: n_out.div_ceil(4),
                height: top_k,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        return Ok(());
    }
    let (kernel, rows_per_group, threads, block) = match bank.dtype {
        GgmlType::IQ3_S => (
            "kernel_deepseek_v4_all_slots_down_iq3_s_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::IQ3_XXS => (
            "kernel_deepseek_v4_all_slots_down_iq3_xxs_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::MXFP4 => ("kernel_deepseek_v4_all_slots_down_mxfp4_f32", 4, 128, 32),
        GgmlType::Q4_K => ("kernel_deepseek_v4_all_slots_down_q4_K_f32", 4, 64, 256),
        dtype => return Err(bad_shape(K, format!("no all-slot kernel for {dtype:?}"))),
    };
    if !n_in.is_multiple_of(block) {
        return Err(bad_shape(
            K,
            format!("input width {n_in} is not a multiple of {block}"),
        ));
    }
    let pso = pipeline_for(ctx, kernel, threads)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, ids);
    enc.set_tensor(4, status);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(rows_per_group),
            height: top_k,
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

/// Buffers for [`encode_grouped_routed_experts`] over `rows` tokens.
pub struct GroupedExperts<'a> {
    /// Routed expert banks; gate and up share a dtype.
    pub gate_bank: &'a MetalTensor,
    pub up_bank: &'a MetalTensor,
    pub down_bank: &'a MetalTensor,
    /// Normalized FFN input, F32 `[hidden, rows]`.
    pub input: &'a MetalTensor,
    /// Routes from [`encode_route_learned_rows`]: I32 ids and F32 weights
    /// `[top_k, rows]`.
    pub ids: &'a MetalTensor,
    pub weights: &'a MetalTensor,
    /// Scratch: I32 per-expert counts `[experts]` and expert-major slots
    /// `[experts * rows]`; F32 slot activations `[ffn, top_k * rows]` and slot
    /// outputs `[hidden, top_k * rows]`.
    pub counts: &'a MetalTensor,
    pub slots: &'a MetalTensor,
    pub inner: &'a MetalTensor,
    pub slot_out: &'a MetalTensor,
    /// Weighted routed output, F32 `[hidden, rows]`.
    pub output: &'a MetalTensor,
}

/// Expert-major routed experts for packed prefill: bucket routes by expert,
/// fused gate/up with the clamped SwiGLU epilogue, grouped down, and the
/// weighted sum over each row's slots. Expert weights are read once per tile
/// of routed tokens instead of once per routed token. Invalid route ids
/// (failed rows) are dropped by bucketing; callers must check per-row route
/// status after the command, as for decode.
#[allow(clippy::too_many_arguments)]
pub fn encode_grouped_routed_experts(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    b: &GroupedExperts<'_>,
    hidden: usize,
    ffn: usize,
    experts: usize,
    top_k: usize,
    rows: usize,
    clamp: f32,
) -> Result<(), MetalError> {
    encode_grouped_routed_experts_with_down_policy(
        ctx,
        enc,
        b,
        hidden,
        ffn,
        experts,
        top_k,
        rows,
        clamp,
        GroupedDownPolicy::Incumbent,
    )
}

/// The bindings [`encode_grouped_routed_experts_with_down_policy`] and the
/// F32-operand form ([`encode_grouped_routed_experts_f32x`]) share: shapes,
/// dtypes, expert banks, 16-byte activation alignment, and no output
/// aliasing an input or another output.
#[allow(clippy::too_many_arguments)]
fn check_grouped_routed_experts(
    kernel: &'static str,
    enc: &KernelEncoder,
    b: &GroupedExperts<'_>,
    hidden: usize,
    ffn: usize,
    experts: usize,
    top_k: usize,
    rows: usize,
) -> Result<(), MetalError> {
    require_serial(kernel, enc)?;
    if rows == 0 || top_k == 0 || top_k > ROUTE_MAX_TOP_K {
        return Err(bad_shape(
            kernel,
            "rows and top-k must be positive (top-k <= 16)",
        ));
    }
    let (h, f, e, k, r) = (
        hidden as u64,
        ffn as u64,
        experts as u64,
        top_k as u64,
        rows as u64,
    );
    check_tensor(kernel, b.input, GgmlType::F32, &[h, r], false, "input")?;
    check_tensor(kernel, b.ids, GgmlType::I32, &[k, r], false, "expert ids")?;
    check_tensor(kernel, b.weights, GgmlType::F32, &[k, r], false, "weights")?;
    check_tensor(kernel, b.counts, GgmlType::I32, &[e], true, "counts")?;
    check_tensor(kernel, b.slots, GgmlType::I32, &[e * r], true, "slots")?;
    check_tensor(kernel, b.inner, GgmlType::F32, &[f, k * r], true, "inner")?;
    check_tensor(
        kernel,
        b.slot_out,
        GgmlType::F32,
        &[h, k * r],
        true,
        "slot outputs",
    )?;
    check_tensor(kernel, b.output, GgmlType::F32, &[h, r], true, "output")?;
    check_expert_bank(kernel, b.gate_bank, hidden, ffn, experts, "gate bank")?;
    check_expert_bank(kernel, b.up_bank, hidden, ffn, experts, "up bank")?;
    check_expert_bank(kernel, b.down_bank, ffn, hidden, experts, "down bank")?;
    // The grouped tiles load activation rows as float2x4 / float4 (16 bytes).
    for (tensor, name) in [
        (b.input, "input"),
        (b.inner, "inner"),
        (b.slot_out, "slot outputs"),
    ] {
        check_alignment(kernel, tensor, 16, name)?;
    }
    let inputs = [
        (b.input, "input"),
        (b.ids, "expert ids"),
        (b.weights, "weights"),
        (b.gate_bank, "gate bank"),
        (b.up_bank, "up bank"),
        (b.down_bank, "down bank"),
    ];
    let scratch = [
        (b.counts, "counts"),
        (b.slots, "slots"),
        (b.inner, "inner"),
        (b.slot_out, "slot outputs"),
        (b.output, "output"),
    ];
    for (i, (written, name)) in scratch.iter().enumerate() {
        check_disjoint(kernel, written, &inputs)
            .map_err(|e| bad_shape(kernel, format!("{name}: {e}")))?;
        for (other, other_name) in &scratch[i + 1..] {
            if super::checks::overlaps(written, other) {
                return Err(bad_shape(kernel, format!("{name} aliases {other_name}")));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupedDownPolicy {
    Incumbent,
    Iq3SSmallCounts,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_grouped_routed_experts_with_down_policy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    b: &GroupedExperts<'_>,
    hidden: usize,
    ffn: usize,
    experts: usize,
    top_k: usize,
    rows: usize,
    clamp: f32,
    down_policy: GroupedDownPolicy,
) -> Result<(), MetalError> {
    const K: &str = "grouped_routed_experts";
    check_grouped_routed_experts(K, enc, b, hidden, ffn, experts, top_k, rows)?;
    let flat = |t: &MetalTensor| MetalTensor {
        shape: vec![t.n_elements()],
        ..t.clone()
    };
    let down_inner = flat(b.inner);
    let down_out = flat(b.slot_out);
    let prepared_down = match down_policy {
        GroupedDownPolicy::Incumbent => None,
        GroupedDownPolicy::Iq3SSmallCounts => super::iq3_s_down_retile::prepare_small_counts(
            ctx,
            enc,
            b.down_bank,
            &down_inner,
            b.counts,
            b.slots,
            &down_out,
            ffn,
            hidden,
            experts,
            rows,
        )?,
    };
    encode_moe_route_bucket_slots_f32(
        ctx,
        enc,
        &flat(b.ids),
        b.counts,
        b.slots,
        experts,
        rows,
        top_k,
    )?;
    encode_moe_swiglu_clamped_f32_grouped_slots_generic(
        ctx,
        enc,
        b.gate_bank,
        b.up_bank,
        &flat(b.input),
        b.counts,
        b.slots,
        &flat(b.inner),
        hidden,
        ffn,
        experts,
        top_k,
        rows,
        clamp,
    )?;
    if let Some(down) = prepared_down {
        down.encode()?;
    } else if b.down_bank.dtype == GgmlType::IQ4_XS {
        encode_moe_down_iq4_xs_f32_grouped_slots(
            ctx,
            enc,
            b.down_bank,
            &flat(b.inner),
            b.counts,
            b.slots,
            &flat(b.slot_out),
            ffn,
            hidden,
            experts,
            rows,
        )?;
    } else {
        encode_moe_down_f32_grouped_slots_generic(
            ctx,
            enc,
            b.down_bank,
            &flat(b.inner),
            b.counts,
            b.slots,
            &flat(b.slot_out),
            ffn,
            hidden,
            experts,
            rows,
        )?;
    }
    encode_moe_weighted_sum_packed_f32(
        ctx,
        enc,
        &flat(b.slot_out),
        &flat(b.weights),
        &flat(b.output),
        hidden,
        top_k,
        rows,
    )?;
    Ok(())
}

/// [`encode_grouped_routed_experts`] with F32 operands (map #12 accuracy
/// lane): the same bucketing, clamped SwiGLU, down projection and weighted
/// sum, with the gate/up and down tiles dequantizing weights to F32 and
/// reading activations (the input and the SwiGLU output) unrounded
/// (`kernels/moe_grouped_f32.metal`). Gate/up IQ2_S or IQ3_S; down IQ3_S
/// or IQ4_XS (the GLM-5.3 expert types); other types are refused. A
/// token's outputs depend on its own routes and activations only, not on
/// which other tokens share its experts' buckets.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_grouped_routed_experts_f32x(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    b: &GroupedExperts<'_>,
    hidden: usize,
    ffn: usize,
    experts: usize,
    top_k: usize,
    rows: usize,
    clamp: f32,
) -> Result<(), MetalError> {
    const K: &str = "grouped_routed_experts_f32x";
    check_grouped_routed_experts(K, enc, b, hidden, ffn, experts, top_k, rows)?;
    // Refuse before encoding anything (the bucketing would otherwise run).
    if b.gate_bank.dtype != b.up_bank.dtype
        || !matches!(b.gate_bank.dtype, GgmlType::IQ2_S | GgmlType::IQ3_S)
        || !matches!(b.down_bank.dtype, GgmlType::IQ3_S | GgmlType::IQ4_XS)
    {
        return Err(bad_shape(
            K,
            format!(
                "no F32-operand tiles for gate/up/down {:?}/{:?}/{:?}",
                b.gate_bank.dtype, b.up_bank.dtype, b.down_bank.dtype
            ),
        ));
    }
    if !clamp.is_finite() || clamp <= 0.0 {
        return Err(bad_shape(
            K,
            format!("clamp must be finite and positive, got {clamp}"),
        ));
    }
    let flat = |t: &MetalTensor| MetalTensor {
        shape: vec![t.n_elements()],
        ..t.clone()
    };
    // Both tiles' checks (bindings, geometry, pipelines) before the first
    // dispatch.
    super::moe_grouped_generic::check_moe_swiglu_clamped_f32x_grouped_slots(
        ctx,
        b.gate_bank,
        b.up_bank,
        &flat(b.input),
        b.counts,
        b.slots,
        &flat(b.inner),
        hidden,
        ffn,
        experts,
        top_k,
        rows,
        clamp,
    )?;
    super::moe_grouped_generic::check_moe_down_f32x_grouped_slots(
        ctx,
        b.down_bank,
        &flat(b.inner),
        b.counts,
        b.slots,
        &flat(b.slot_out),
        ffn,
        hidden,
        experts,
        rows,
    )?;
    encode_moe_route_bucket_slots_f32(
        ctx,
        enc,
        &flat(b.ids),
        b.counts,
        b.slots,
        experts,
        rows,
        top_k,
    )?;
    super::moe_grouped_generic::encode_moe_swiglu_clamped_f32x_grouped_slots(
        ctx,
        enc,
        b.gate_bank,
        b.up_bank,
        &flat(b.input),
        b.counts,
        b.slots,
        &flat(b.inner),
        hidden,
        ffn,
        experts,
        top_k,
        rows,
        clamp,
    )?;
    super::moe_grouped_generic::encode_moe_down_f32x_grouped_slots(
        ctx,
        enc,
        b.down_bank,
        &flat(b.inner),
        b.counts,
        b.slots,
        &flat(b.slot_out),
        ffn,
        hidden,
        experts,
        rows,
    )?;
    encode_moe_weighted_sum_packed_f32(
        ctx,
        enc,
        &flat(b.slot_out),
        &flat(b.weights),
        &flat(b.output),
        hidden,
        top_k,
        rows,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{assert_moe_oracle_close, dequant_expert, offset_tensor};
    use super::*;

    pub(super) const EXPERTS: usize = 12;
    pub(super) const TOP_K: usize = 8;
    pub(super) const HIDDEN: usize = 4096;
    pub(super) const FFN: usize = 2048;
    const IDS: [i32; TOP_K] = [11, 3, 7, 0, 9, 5, 10, 2];

    /// Arbitrary quant payload bytes with a small finite F16 scale per block
    /// (every IQ2_S/IQ3_S/IQ4_XS byte pattern decodes).
    pub(super) fn synthetic_bank(dtype: GgmlType, n_in: usize, n_out: usize, seed: u64) -> Vec<u8> {
        let (block, block_bytes) = dtype.storage_layout().unwrap();
        let blocks = EXPERTS * n_out * (n_in / block as usize);
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut bytes = vec![0u8; blocks * block_bytes as usize];
        for byte in bytes.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        for (i, chunk) in bytes.chunks_exact_mut(block_bytes as usize).enumerate() {
            let d = if i % 2 == 0 { 1.0 } else { -1.0 } * ((i % 7) + 1) as f32 / 4096.0;
            chunk[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        }
        bytes
    }

    pub(super) fn bank(
        ctx: &MetalContext,
        bytes: &[u8],
        dtype: GgmlType,
        n_in: usize,
        n_out: usize,
    ) -> MetalTensor {
        offset_tensor(
            ctx,
            256,
            bytes,
            64,
            vec![n_in as u64, n_out as u64, EXPERTS as u64],
            dtype,
        )
    }

    pub(super) fn f32_tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        offset_tensor(
            ctx,
            16,
            bytemuck::cast_slice(values),
            20,
            shape,
            GgmlType::F32,
        )
    }

    fn i32_tensor(ctx: &MetalContext, values: &[i32]) -> MetalTensor {
        offset_tensor(
            ctx,
            16,
            bytemuck::cast_slice(values),
            20,
            vec![values.len() as u64],
            GgmlType::I32,
        )
    }

    pub(super) fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
        super::super::test_support::tensor_f32_at_offset(tensor)
    }

    fn mat_vec(weights: &[f32], x: &[f32], n_out: usize) -> Vec<f32> {
        let n_in = x.len();
        (0..n_out)
            .map(|r| {
                weights[r * n_in..(r + 1) * n_in]
                    .iter()
                    .zip(x)
                    .map(|(w, v)| (*w as f64) * (*v as f64))
                    .sum::<f64>() as f32
            })
            .collect()
    }

    pub(super) fn values(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 31 + seed * 17) % 97) as f32 - 48.0) * 0.021)
            .collect()
    }

    pub(super) fn run(ctx: &MetalContext, encode: impl FnOnce(&KernelEncoder)) {
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder);
        encoder.end();
        command.commit();
        wait_completed(&command).expect("command buffer failed");
    }

    /// GLM-5.3 widths and top-8 over a reduced bank: fused gate/up clamp for
    /// IQ2_S and IQ3_S, against llama.cpp CPU dequantization.
    #[test]
    fn all_slots_gate_up_matches_dequantized_reference_at_glm_widths() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let x = values(HIDDEN, 1);
        for dtype in [GgmlType::IQ2_S, GgmlType::IQ3_S] {
            let gate_bytes = synthetic_bank(dtype, HIDDEN, FFN, 3);
            let up_bytes = synthetic_bank(dtype, HIDDEN, FFN, 5);
            let gate = bank(&ctx, &gate_bytes, dtype, HIDDEN, FFN);
            let up = bank(&ctx, &up_bytes, dtype, HIDDEN, FFN);
            let input = f32_tensor(&ctx, &x, vec![HIDDEN as u64]);
            let ids = i32_tensor(&ctx, &IDS);
            let ready = i32_tensor(&ctx, &[ROUTE_STATUS_READY]);
            let unready = i32_tensor(&ctx, &[ROUTE_STATUS_PENDING]);
            let out = f32_tensor(
                &ctx,
                &vec![7.0; FFN * TOP_K],
                vec![FFN as u64, TOP_K as u64],
            );
            let zeroed = f32_tensor(
                &ctx,
                &vec![7.0; FFN * TOP_K],
                vec![FFN as u64, TOP_K as u64],
            );
            run(&ctx, |enc| {
                encode_all_slots_gate_up_swiglu(
                    &ctx, enc, &gate, &up, &input, &ids, &ready, &out, HIDDEN, FFN, EXPERTS, TOP_K,
                    10.0,
                )
                .unwrap();
                encode_all_slots_gate_up_swiglu(
                    &ctx, enc, &gate, &up, &input, &ids, &unready, &zeroed, HIDDEN, FFN, EXPERTS,
                    TOP_K, 10.0,
                )
                .unwrap();
            });
            let mut expected = Vec::with_capacity(FFN * TOP_K);
            for &expert in &IDS {
                let g = mat_vec(
                    &dequant_expert(&gate_bytes, dtype, HIDDEN, FFN, expert as usize),
                    &x,
                    FFN,
                );
                let u = mat_vec(
                    &dequant_expert(&up_bytes, dtype, HIDDEN, FFN, expert as usize),
                    &x,
                    FFN,
                );
                expected.extend(g.iter().zip(&u).map(|(&g, &u)| {
                    let g = g.min(10.0);
                    g / (1.0 + (-g).exp()) * u.clamp(-10.0, 10.0)
                }));
            }
            assert_moe_oracle_close(&format!("{dtype:?} gate/up"), &read_f32(&out), &expected);
            assert!(
                read_f32(&zeroed).iter().all(|&v| v == 0.0),
                "{dtype:?} unready route"
            );
        }
    }

    /// GLM-5.3 down widths and top-8: the new all-slot IQ3_S kernel and the
    /// bridged IQ4_XS kernel against llama.cpp CPU dequantization.
    #[test]
    fn all_slots_down_matches_dequantized_reference_at_glm_widths() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let inner = values(FFN * TOP_K, 2);
        for dtype in [GgmlType::IQ3_S, GgmlType::IQ4_XS] {
            let bytes = synthetic_bank(dtype, FFN, HIDDEN, 7);
            let down = bank(&ctx, &bytes, dtype, FFN, HIDDEN);
            let input = f32_tensor(&ctx, &inner, vec![FFN as u64, TOP_K as u64]);
            let ids = i32_tensor(&ctx, &IDS);
            let status = i32_tensor(&ctx, &[ROUTE_STATUS_READY]);
            let out = f32_tensor(
                &ctx,
                &vec![7.0; HIDDEN * TOP_K],
                vec![HIDDEN as u64, TOP_K as u64],
            );
            run(&ctx, |enc| {
                encode_all_slots_down(
                    &ctx, enc, &down, &input, &ids, &status, &out, FFN, HIDDEN, EXPERTS, TOP_K,
                )
                .unwrap();
            });
            let mut expected = Vec::with_capacity(HIDDEN * TOP_K);
            for (slot, &expert) in IDS.iter().enumerate() {
                let w = dequant_expert(&bytes, dtype, FFN, HIDDEN, expert as usize);
                expected.extend(mat_vec(&w, &inner[slot * FFN..(slot + 1) * FFN], HIDDEN));
            }
            assert_moe_oracle_close(&format!("{dtype:?} down"), &read_f32(&out), &expected);
        }
        // The DS4 IQ3_S kernel writes zero rows for an unready route.
        let bytes = synthetic_bank(GgmlType::IQ3_S, FFN, HIDDEN, 7);
        let down = bank(&ctx, &bytes, GgmlType::IQ3_S, FFN, HIDDEN);
        let input = f32_tensor(&ctx, &inner, vec![FFN as u64, TOP_K as u64]);
        let ids = i32_tensor(&ctx, &IDS);
        let unready = i32_tensor(&ctx, &[ROUTE_STATUS_PENDING]);
        let out = f32_tensor(
            &ctx,
            &vec![7.0; HIDDEN * TOP_K],
            vec![HIDDEN as u64, TOP_K as u64],
        );
        run(&ctx, |enc| {
            encode_all_slots_down(
                &ctx, enc, &down, &input, &ids, &unready, &out, FFN, HIDDEN, EXPERTS, TOP_K,
            )
            .unwrap();
        });
        assert!(read_f32(&out).iter().all(|&v| v == 0.0));
    }

    /// Packed grouped experts over 24 rows (GLM widths, 12-expert banks,
    /// top-8, routes from the rows router) against the per-row all-slot decode
    /// composition on the same routes, for IQ3_S and IQ4_XS down.
    #[test]
    fn grouped_routed_experts_match_per_row_decode_composition() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const ROWS: usize = 24;
        let route = LearnedRoute {
            experts: EXPERTS,
            top_k: TOP_K,
            score: RouteScore::Sigmoid,
            routed_scale: 2.5,
        };
        let x = values(HIDDEN * ROWS, 3);
        let logits: Vec<f32> = (0..EXPERTS * ROWS)
            .map(|i| ((i * 29 + i / 7) % 53) as f32 * 0.09 - 2.0)
            .collect();
        let bias = vec![0.0f32; EXPERTS];
        let gate_bytes = synthetic_bank(GgmlType::IQ2_S, HIDDEN, FFN, 3);
        let up_bytes = synthetic_bank(GgmlType::IQ2_S, HIDDEN, FFN, 5);
        let gate = bank(&ctx, &gate_bytes, GgmlType::IQ2_S, HIDDEN, FFN);
        let up = bank(&ctx, &up_bytes, GgmlType::IQ2_S, HIDDEN, FFN);
        let r = ROWS as u64;
        let (h, f, k, e) = (HIDDEN as u64, FFN as u64, TOP_K as u64, EXPERTS as u64);
        let input = f32_tensor(&ctx, &x, vec![h, r]);
        let logits_t = f32_tensor(&ctx, &logits, vec![e, r]);
        let bias_t = f32_tensor(&ctx, &bias, vec![e]);
        let ids = offset_tensor(
            &ctx,
            16,
            bytemuck::cast_slice(&vec![0i32; TOP_K * ROWS]),
            16,
            vec![k, r],
            GgmlType::I32,
        );
        let weights = f32_tensor(&ctx, &vec![0.0; TOP_K * ROWS], vec![k, r]);
        let status = offset_tensor(
            &ctx,
            16,
            bytemuck::cast_slice(&[0i32; ROWS]),
            16,
            vec![r],
            GgmlType::I32,
        );
        for down_dtype in [GgmlType::IQ3_S, GgmlType::IQ4_XS] {
            let down_bytes = synthetic_bank(down_dtype, FFN, HIDDEN, 7);
            let down = bank(&ctx, &down_bytes, down_dtype, FFN, HIDDEN);
            let counts = offset_tensor(
                &ctx,
                16,
                bytemuck::cast_slice(&[0i32; EXPERTS]),
                16,
                vec![e],
                GgmlType::I32,
            );
            let slots = offset_tensor(
                &ctx,
                16,
                bytemuck::cast_slice(&vec![0i32; EXPERTS * ROWS]),
                16,
                vec![e * r],
                GgmlType::I32,
            );
            let inner = f32_tensor(&ctx, &vec![0.0; FFN * TOP_K * ROWS], vec![f, k * r]);
            let slot_out = f32_tensor(&ctx, &vec![0.0; HIDDEN * TOP_K * ROWS], vec![h, k * r]);
            let packed = f32_tensor(&ctx, &vec![0.0; HIDDEN * ROWS], vec![h, r]);
            run(&ctx, |enc| {
                encode_route_learned_rows(
                    &ctx, enc, &route, ROWS, &logits_t, &bias_t, &ids, &weights, &status,
                )
                .unwrap();
                encode_grouped_routed_experts(
                    &ctx,
                    enc,
                    &GroupedExperts {
                        gate_bank: &gate,
                        up_bank: &up,
                        down_bank: &down,
                        input: &input,
                        ids: &ids,
                        weights: &weights,
                        counts: &counts,
                        slots: &slots,
                        inner: &inner,
                        slot_out: &slot_out,
                        output: &packed,
                    },
                    HIDDEN,
                    FFN,
                    EXPERTS,
                    TOP_K,
                    ROWS,
                    10.0,
                )
                .unwrap();
            });
            // Per-row decode composition on the same routes.
            let all_ids = super::super::test_support::tensor_backing_bytes(&ids);
            let all_ids: &[i32] = bytemuck::cast_slice(&all_ids[16..16 + 4 * TOP_K * ROWS]);
            let all_weights = read_f32(&weights);
            let mut reference = Vec::with_capacity(HIDDEN * ROWS);
            for row in 0..ROWS {
                let row_ids = i32_tensor(&ctx, &all_ids[row * TOP_K..(row + 1) * TOP_K]);
                let ready = i32_tensor(&ctx, &[ROUTE_STATUS_READY]);
                let row_x = f32_tensor(&ctx, &x[row * HIDDEN..(row + 1) * HIDDEN], vec![h]);
                let row_inner = f32_tensor(&ctx, &vec![0.0; FFN * TOP_K], vec![f, k]);
                let row_out = f32_tensor(&ctx, &vec![0.0; HIDDEN * TOP_K], vec![h, k]);
                run(&ctx, |enc| {
                    encode_all_slots_gate_up_swiglu(
                        &ctx, enc, &gate, &up, &row_x, &row_ids, &ready, &row_inner, HIDDEN, FFN,
                        EXPERTS, TOP_K, 10.0,
                    )
                    .unwrap();
                    encode_all_slots_down(
                        &ctx, enc, &down, &row_inner, &row_ids, &ready, &row_out, FFN, HIDDEN,
                        EXPERTS, TOP_K,
                    )
                    .unwrap();
                });
                let slots_out = read_f32(&row_out);
                for d in 0..HIDDEN {
                    reference.push(
                        (0..TOP_K)
                            .map(|s| all_weights[row * TOP_K + s] * slots_out[s * HIDDEN + d])
                            .sum::<f32>(),
                    );
                }
            }
            assert_moe_oracle_close_loose(
                &format!("{down_dtype:?} grouped"),
                &read_f32(&packed),
                &reference,
            );
            // The tiles' 16-byte activation loads refuse a 4-byte offset.
            let shifted = |t: &MetalTensor| {
                let data = vec![0u8; t.n_elements() as usize * 4];
                offset_tensor(&ctx, 4, &data, 0, t.shape.clone(), GgmlType::F32)
            };
            let (input_4, inner_4, slot_out_4) =
                (shifted(&input), shifted(&inner), shifted(&slot_out));
            for (name, input, inner, slot_out) in [
                ("input", &input_4, &inner, &slot_out),
                ("inner", &input, &inner_4, &slot_out),
                ("slot outputs", &input, &inner, &slot_out_4),
            ] {
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                let err = encode_grouped_routed_experts(
                    &ctx,
                    &enc,
                    &GroupedExperts {
                        gate_bank: &gate,
                        up_bank: &up,
                        down_bank: &down,
                        input,
                        ids: &ids,
                        weights: &weights,
                        counts: &counts,
                        slots: &slots,
                        inner,
                        slot_out,
                        output: &packed,
                    },
                    HIDDEN,
                    FFN,
                    EXPERTS,
                    TOP_K,
                    ROWS,
                    10.0,
                )
                .unwrap_err();
                let message = err.to_string();
                assert!(
                    message.contains(&format!("{name} offset 4 is not 16-byte aligned")),
                    "{message}"
                );
                enc.end();
            }
        }
    }

    /// Grouped kernels stage activations in half: cosine and max-relative
    /// bounds as in the existing grouped-kernel tests.
    fn assert_moe_oracle_close_loose(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().all(|v| v.is_finite()), "{label}: non-finite");
        let dot: f64 = actual
            .iter()
            .zip(expected)
            .map(|(a, e)| *a as f64 * *e as f64)
            .sum();
        let na = actual
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let ne = expected
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cos = dot / (na * ne).max(1e-30);
        let ref_max = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let max_abs = super::super::test_support::max_abs_diff_finite(label, actual, expected);
        eprintln!("[{label}] cos={cos:.7} max|delta|={max_abs:.3e} ref_max={ref_max:.3e}");
        assert!(cos >= 0.9999, "{label}: cos {cos}");
        assert!(
            max_abs <= 1e-2 * ref_max + 1e-4,
            "{label}: max {max_abs} ref {ref_max}"
        );
    }

    /// Timing screen (not qualification): GPU time of one GLM-5.3 routed-expert
    /// decode block (288-expert banks: IQ2_S gate/up [4096 -> 2048], IQ3_S
    /// down [2048 -> 4096]; top-8 all-slot gate/up SwiGLU, all-slot down, then
    /// the weighted sum) as 42 serially encoded blocks per command: a repeated
    /// workload, not a model-state chain (every block reads the same input and
    /// no output feeds the next). Each block routes to a different 8 experts,
    /// so weights stream from DRAM. Random quant payloads with finite scales,
    /// which need not reproduce real lookup behavior (timing only). GB/s is
    /// logical-byte effective bandwidth. Refuses MTL_DEBUG_LAYER.
    #[test]
    #[ignore = "timing screen; run without MTL_DEBUG_LAYER"]
    fn routed_expert_block_dispatch_costs() {
        assert!(
            std::env::var_os("MTL_DEBUG_LAYER").is_none(),
            "timing runs must not enable MTL_DEBUG_LAYER"
        );
        let _lease = crate::metal::acquire_metal_benchmark_lease().expect("GPU lease");
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const E: usize = 288;
        const BLOCKS: usize = 42;
        let bank = |dtype: GgmlType, n_in: usize, n_out: usize, seed: u64| -> MetalTensor {
            let (block, block_bytes) = dtype.storage_layout().unwrap();
            let blocks = E * n_out * (n_in / block as usize);
            let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let mut bytes = vec![0u8; blocks * block_bytes as usize];
            for word in bytes.as_chunks_mut::<8>().0 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                word.copy_from_slice(&state.to_le_bytes());
            }
            for (i, chunk) in bytes.chunks_exact_mut(block_bytes as usize).enumerate() {
                let d = ((i % 7) + 1) as f32 / 4096.0;
                chunk[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            }
            offset_tensor(
                &ctx,
                0,
                &bytes,
                0,
                vec![n_in as u64, n_out as u64, E as u64],
                dtype,
            )
        };
        let gate = bank(GgmlType::IQ2_S, HIDDEN, FFN, 1);
        let up = bank(GgmlType::IQ2_S, HIDDEN, FFN, 2);
        let down = bank(GgmlType::IQ3_S, FFN, HIDDEN, 3);
        let f32_t = |values: &[f32], shape: Vec<u64>| {
            offset_tensor(
                &ctx,
                0,
                bytemuck::cast_slice(values),
                0,
                shape,
                GgmlType::F32,
            )
        };
        let i32_t = |values: &[i32], shape: Vec<u64>| {
            offset_tensor(
                &ctx,
                0,
                bytemuck::cast_slice(values),
                0,
                shape,
                GgmlType::I32,
            )
        };
        let input = f32_t(
            &(0..HIDDEN)
                .map(|i| ((i * 37) % 101) as f32 * 0.01 - 0.5)
                .collect::<Vec<_>>(),
            vec![HIDDEN as u64],
        );
        let routes: Vec<MetalTensor> = (0..BLOCKS)
            .map(|b| {
                let ids: Vec<i32> = (0..TOP_K).map(|j| ((b * 61 + j * 37) % E) as i32).collect();
                i32_t(&ids, vec![TOP_K as u64])
            })
            .collect();
        let status = i32_t(&[crate::metal::ROUTE_STATUS_READY], vec![1]);
        let weights = f32_t(&[0.125; TOP_K], vec![TOP_K as u64]);
        let inner = f32_t(&vec![0.0; FFN * TOP_K], vec![FFN as u64, TOP_K as u64]);
        let slots = f32_t(
            &vec![0.0; HIDDEN * TOP_K],
            vec![HIDDEN as u64, TOP_K as u64],
        );
        let routed = f32_t(&vec![0.0; HIDDEN], vec![HIDDEN as u64]);
        let gate_up = |enc: &KernelEncoder, b: usize| {
            encode_all_slots_gate_up_swiglu(
                &ctx, enc, &gate, &up, &input, &routes[b], &status, &inner, HIDDEN, FFN, E, TOP_K,
                7.0,
            )
            .unwrap();
        };
        let down_pass = |enc: &KernelEncoder, b: usize| {
            encode_all_slots_down(
                &ctx, enc, &down, &inner, &routes[b], &status, &slots, FFN, HIDDEN, E, TOP_K,
            )
            .unwrap();
        };
        let sum = |enc: &KernelEncoder, _: usize| {
            crate::metal::encode_moe_weighted_sum_f32(
                &ctx, enc, &slots, &weights, &routed, HIDDEN, TOP_K,
            )
            .unwrap();
        };
        let time = |encode: &dyn Fn(&KernelEncoder, usize)| -> f64 {
            let mut samples: Vec<f64> = (0..6)
                .map(|_| {
                    let command = ctx.queue.commandBuffer().expect("command buffer");
                    let enc = KernelEncoder::begin(&command);
                    for b in 0..BLOCKS {
                        encode(&enc, b);
                    }
                    enc.end();
                    command.commit();
                    crate::metal::wait_completed(&command).expect("command buffer failed");
                    (command.GPUEndTime() - command.GPUStartTime()) * 1e6 / BLOCKS as f64
                })
                .skip(1)
                .collect();
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        let whole = |enc: &KernelEncoder, b: usize| {
            gate_up(enc, b);
            down_pass(enc, b);
            sum(enc, b);
        };
        let warm = std::time::Instant::now();
        while warm.elapsed() < std::time::Duration::from_secs(2) {
            time(&whole);
        }
        let bytes = |dtype: GgmlType, n_in: usize, n_out: usize| {
            let (block, block_bytes) = dtype.storage_layout().unwrap();
            (n_in as u64 / block * block_bytes) as f64 * n_out as f64 * TOP_K as f64
        };
        let gate_up_bytes = 2.0 * bytes(GgmlType::IQ2_S, HIDDEN, FFN);
        let down_bytes = bytes(GgmlType::IQ3_S, FFN, HIDDEN);
        type Encode<'a> = &'a dyn Fn(&KernelEncoder, usize);
        let rows: [(&str, Encode<'_>, f64); 4] = [
            ("whole routed block", &whole, gate_up_bytes + down_bytes),
            ("gate/up SwiGLU (IQ2_S, 8 slots)", &gate_up, gate_up_bytes),
            ("down (IQ3_S, 8 slots)", &down_pass, down_bytes),
            ("weighted sum", &sum, 0.0),
        ];
        for (label, encode, bytes) in rows {
            let us = time(encode);
            eprintln!(
                "routed experts: {label:<34} {us:8.2} us/block {:7.1} GB/s",
                bytes / us / 1e3
            );
        }
    }
}

#[cfg(test)]
mod f32x_tests;
