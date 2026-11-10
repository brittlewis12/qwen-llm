use super::*;
use objc2_metal::{MTLComputePipelineState, MTLSize};

crate::env_flag!(
    /// Flash-Next one-token H2560/F640/E512/top10 on the tensor-capable tier combines router, shared gate/up and shared-scale projections in one dispatch. Q4_K/Q5_K/Q6_K/IQ4_XS shared pairs and F32 router only; reuses shared-output scratch until the shared down projection. Set 0 for the incumbent. M4 retains its prior route. Respects cooperative F32, Q6 r1, IQ4_XS fast and wide-scale rollbacks; unsupported PSOs/alignments retain the incumbent.
    default_on enabled,
    "QWEN_FN_FFN_PROJECTIONS_FUSED"
);

crate::env_flag!(
    /// Batch the fused shared-expert/router projections across independent members on one grid. The per-member arithmetic and output layout are unchanged; 0 keeps one dispatch per member.
    default_on members_enabled,
    "QWEN_FN_MOE_BATCH_PROJECTIONS"
);
fn members_selected() -> bool {
    #[cfg(feature = "perf-ablations")]
    {
        return crate::metal::fn_batch_probe::axis("moe_batch_projections", members_enabled());
    }
    #[cfg(not(feature = "perf-ablations"))]
    {
        members_enabled()
    }
}

fn tag(dtype: GgmlType) -> Option<usize> {
    Some(match dtype {
        GgmlType::Q4_K => 0,
        GgmlType::Q5_K => 1,
        GgmlType::Q6_K => 2,
        GgmlType::IQ4_XS => 3,
        _ => return None,
    })
}
const NAMES: [[&str; 4]; 4] = [
    [
        "kernel_qwen4exp_ffn_projections_0_0",
        "kernel_qwen4exp_ffn_projections_0_1",
        "kernel_qwen4exp_ffn_projections_0_2",
        "kernel_qwen4exp_ffn_projections_0_3",
    ],
    [
        "kernel_qwen4exp_ffn_projections_1_0",
        "kernel_qwen4exp_ffn_projections_1_1",
        "kernel_qwen4exp_ffn_projections_1_2",
        "kernel_qwen4exp_ffn_projections_1_3",
    ],
    [
        "kernel_qwen4exp_ffn_projections_2_0",
        "kernel_qwen4exp_ffn_projections_2_1",
        "kernel_qwen4exp_ffn_projections_2_2",
        "kernel_qwen4exp_ffn_projections_2_3",
    ],
    [
        "kernel_qwen4exp_ffn_projections_3_0",
        "kernel_qwen4exp_ffn_projections_3_1",
        "kernel_qwen4exp_ffn_projections_3_2",
        "kernel_qwen4exp_ffn_projections_3_3",
    ],
];
const NAMES_MEMBERS: [[&str; 4]; 4] = [
    [
        "kernel_qwen4exp_ffn_projections_members_0_0",
        "kernel_qwen4exp_ffn_projections_members_0_1",
        "kernel_qwen4exp_ffn_projections_members_0_2",
        "kernel_qwen4exp_ffn_projections_members_0_3",
    ],
    [
        "kernel_qwen4exp_ffn_projections_members_1_0",
        "kernel_qwen4exp_ffn_projections_members_1_1",
        "kernel_qwen4exp_ffn_projections_members_1_2",
        "kernel_qwen4exp_ffn_projections_members_1_3",
    ],
    [
        "kernel_qwen4exp_ffn_projections_members_2_0",
        "kernel_qwen4exp_ffn_projections_members_2_1",
        "kernel_qwen4exp_ffn_projections_members_2_2",
        "kernel_qwen4exp_ffn_projections_members_2_3",
    ],
    [
        "kernel_qwen4exp_ffn_projections_members_3_0",
        "kernel_qwen4exp_ffn_projections_members_3_1",
        "kernel_qwen4exp_ffn_projections_members_3_2",
        "kernel_qwen4exp_ffn_projections_members_3_3",
    ],
];

fn pipeline_named(
    ctx: &MetalContext,
    w: Qwen4ExpMoeMetalWeights<'_>,
    names: &[[&str; 4]; 4],
) -> Result<Option<(crate::metal::Pipeline, usize)>, MetalError> {
    let g = w.geometry;
    if !enabled()
        || !ctx.tensor_ops_status().is_available()
        || (
            g.hidden_size,
            g.shared_intermediate_size,
            g.expert_count,
            g.experts_per_token,
        ) != (2560, 640, 512, 10)
        || w.router.dtype != GgmlType::F32
        || w.shared_router.dtype != GgmlType::F32
        || !gate_wide()
        || !crate::metal::mat_vec_f32_cooperative_enabled()
    {
        return Ok(None);
    }
    let (Some(gt), Some(ut)) = (tag(w.shared_gate.dtype), tag(w.shared_up.dtype)) else {
        return Ok(None);
    };
    if !crate::metal::mat_vec_packet_quant_enabled(gt == 2 || ut == 2, gt == 3 || ut == 3) {
        return Ok(None);
    }
    let p = match ctx.pipeline(names[gt][ut]) {
        Ok(p)
            if p.threadExecutionWidth() == 32
                && p.maxTotalThreadsPerThreadgroup() >= 128
                && p.staticThreadgroupMemoryLength() + 256
                    <= ctx.device.maxThreadgroupMemoryLength() =>
        {
            p
        }
        Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let rows = |t| if t == 0 || t == 3 { 8 } else { 4 };
    Ok(Some((p, 640 / rows(gt) + 640 / rows(ut) + 257)))
}
fn pipeline(
    ctx: &MetalContext,
    w: Qwen4ExpMoeMetalWeights<'_>,
) -> Result<Option<(crate::metal::Pipeline, usize)>, MetalError> {
    pipeline_named(ctx, w, &NAMES)
}

fn pipeline_members(
    ctx: &MetalContext,
    w: Qwen4ExpMoeMetalWeights<'_>,
) -> Result<Option<(crate::metal::Pipeline, usize)>, MetalError> {
    pipeline_named(ctx, w, &NAMES_MEMBERS)
}

pub(super) fn preflight(
    ctx: &MetalContext,
    w: Qwen4ExpMoeMetalWeights<'_>,
) -> Result<(), MetalError> {
    pipeline(ctx, w)?;
    Ok(())
}

// Parent validates serial encoder custody, tensor types/shapes/ranges,
// disjointness and weight provenance before this leaf. Shared output is dead
// until SiLU consumes these two projections into shared_inner; shared down
// may then overwrite it. Routed down uses a different buffer.
pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    w: Qwen4ExpMoeMetalWeights<'_>,
    b: Qwen4ExpMoeSingletonBuffers<'_>,
) -> Result<bool, MetalError> {
    if [input, w.shared_gate, w.shared_up, w.router, w.shared_router]
        .iter()
        .any(|t| !t.offset.is_multiple_of(16))
    {
        return Ok(false);
    }
    let Some((p, groups)) = pipeline(ctx, w)? else {
        return Ok(false);
    };
    enc.set_pipeline(&p);
    for (slot, t) in [
        w.shared_gate,
        w.shared_up,
        w.router,
        w.shared_router,
        input,
        b.shared_output,
        b.router_logits,
        b.shared_scale,
    ]
    .into_iter()
    .enumerate()
    {
        enc.set_tensor(slot, t);
    }
    enc.set_threadgroup_memory(0, 256);
    enc.dispatch(
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(true)
}

/// Encode the fused shared/router projections for every member on one grid.
/// Returns false without encoding when the fused route, member layouts or
/// member-axis PSO are unavailable, so the caller keeps per-member dispatches.
pub(super) fn try_encode_members(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    inputs: &[&MetalTensor],
    w: Qwen4ExpMoeMetalWeights<'_>,
    members: &[&Qwen4ExpMoeMetalWorkspace],
) -> Result<bool, MetalError> {
    let b = inputs.len();
    if members.len() != b || !(2..=4).contains(&b) || !enabled() || !members_selected() {
        return Ok(false);
    }
    if [w.shared_gate, w.shared_up, w.router, w.shared_router]
        .iter()
        .any(|t| !t.offset.is_multiple_of(16))
    {
        return Ok(false);
    }
    let g = w.geometry;
    let n = g.shared_intermediate_size as u64;
    for (input, member) in inputs.iter().zip(members) {
        if member.geometry != g
            || !input.offset.is_multiple_of(16)
            || !member.shared_output.offset.is_multiple_of(4)
            || !member.router_logits.offset.is_multiple_of(4)
            || !member.shared_gate.offset.is_multiple_of(4)
            || member.shared_output.n_elements() < 2 * n
            || member.router_logits.n_elements() < g.expert_count as u64
            || member.shared_gate.n_elements() < 1
        {
            return Ok(false);
        }
    }
    let Some((p, groups)) = pipeline_members(ctx, w)? else {
        return Ok(false);
    };
    enc.set_pipeline(&p);
    for (slot, t) in [w.shared_gate, w.shared_up, w.router, w.shared_router]
        .into_iter()
        .enumerate()
    {
        enc.set_tensor(slot, t);
    }
    for i in 0..4 {
        let j = i.min(b - 1);
        enc.set_tensor(4 + i, inputs[j]);
        enc.set_tensor(8 + i, &members[j].shared_output);
        enc.set_tensor(12 + i, &members[j].router_logits);
        enc.set_tensor(16 + i, &members[j].shared_gate);
    }
    enc.set_bytes(20, &(groups as u32));
    enc.set_threadgroup_memory(0, 256);
    enc.dispatch(
        MTLSize {
            width: groups * b,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    #[cfg(feature = "perf-ablations")]
    crate::metal::fn_batch_probe::observe("moe_batch_projections", b);
    Ok(true)
}

#[cfg(test)]
mod tests;
