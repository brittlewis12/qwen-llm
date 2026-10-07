//! Scoped Flash-Next IQ2_S singleton encoder. Included by qwen4exp_moe so
//! this helper needs no global metal module export.
use super::*;

pub(super) const KERNEL: &str = "kernel_qwen4exp_expert_compat_iq2_s_swiglu_f32";

#[cfg(test)]
#[path = "qwen4exp_expert_compat/tests.rs"]
mod tests;

pub(super) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    buffers: Qwen4ExpMoeSingletonBuffers<'_>,
) -> Result<(), Qwen4ExpMoeError> {
    // Geometry, bank bounds, dtype, disjointness and serial encoding are
    // validated by the parent MoE contract before any dispatch is encoded.
    let g = weights.geometry;
    enc.set_pipeline(&ctx.pipeline(KERNEL)?);
    enc.set_bytes_slice(
        0,
        &[
            g.hidden_size as u32,
            g.routed_intermediate_size as u32,
            g.expert_count as u32,
            g.experts_per_token as u32,
        ],
    );
    enc.set_tensor(1, weights.routed_gate);
    enc.set_tensor(2, weights.routed_up);
    enc.set_tensor(3, input);
    enc.set_tensor(4, buffers.topk_ids);
    enc.set_tensor(5, buffers.routed_inner);
    enc.set_threadgroup_memory(0, 16 * size_of::<f32>());
    enc.dispatch(
        objc2_metal::MTLSize {
            width: g.routed_intermediate_size.div_ceil(8),
            height: g.experts_per_token,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
