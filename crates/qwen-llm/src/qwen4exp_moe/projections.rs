use super::*;
use objc2_metal::{MTLComputePipelineState, MTLSize};

crate::env_flag!(
    /// Flash-Next one-token H2560/F640/E512/top10 combines router, shared gate/up and shared-scale projections in one dispatch. Q4_K/Q5_K/Q6_K/IQ4_XS shared pairs and F32 router only; reuses shared-output scratch until the shared down projection. Set 0 for the incumbent. Respects cooperative F32, Q6 r1, IQ4_XS fast and wide-scale rollbacks; unsupported PSOs/alignments retain the incumbent.
    default_on enabled,
    "QWEN_FN_FFN_PROJECTIONS_FUSED"
);

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
fn pipeline(
    ctx: &MetalContext,
    w: Qwen4ExpMoeMetalWeights<'_>,
) -> Result<Option<(crate::metal::Pipeline, usize)>, MetalError> {
    let g = w.geometry;
    if !enabled()
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
    let p = match ctx.pipeline(NAMES[gt][ut]) {
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
