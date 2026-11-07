use super::*;
use objc2_metal::MTLComputePipelineState;

crate::env_flag!(
    /// Flash-Next one-token H2560/48-head GDN combines independent F32 alpha/beta projections and the decay epilogue. Reuses existing scratch and reduction orders. Set 0 for separate dispatches. Unsupported geometry, alignment or PSOs, and QWEN_MATVEC_F32_LCPP_R2=0, retain the incumbent.
    default_on enabled,
    "QWEN_FN_GDN_CONTROLS_FUSED"
);

const KERNEL: &str = "kernel_qwen4exp_gdn_controls_f32";

fn eligible(g: GatedDeltaNetMetalGeometry) -> bool {
    enabled()
        && crate::metal::mat_vec_f32_cooperative_enabled()
        && (g.hidden_size, g.value_heads, g.head_dim) == (2560, 48, 128)
}

fn pipeline(
    ctx: &MetalContext,
    g: GatedDeltaNetMetalGeometry,
) -> Result<Option<crate::metal::Pipeline>, MetalError> {
    if !eligible(g) {
        return Ok(None);
    }
    match ctx.pipeline(KERNEL) {
        Ok(p)
            if p.threadExecutionWidth() == 32
                && p.maxTotalThreadsPerThreadgroup() >= 128
                && p.staticThreadgroupMemoryLength() + 256
                    <= ctx.device.maxThreadgroupMemoryLength() =>
        {
            Ok(Some(p))
        }
        Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => Ok(None),
        Err(e) => Err(e),
    }
}

pub(super) fn preflight(
    ctx: &MetalContext,
    g: GatedDeltaNetMetalGeometry,
) -> Result<(), MetalError> {
    pipeline(ctx, g)?;
    Ok(())
}

// Parent validates types, writable ranges, disjointness and device custody
// before reserving or mutating the serial encoder. Decline before mutation.
pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: GatedDeltaNetMetalWeights<'_>,
    workspace: &GatedDeltaNetMetalWorkspace,
) -> Result<bool, MetalError> {
    if [input, weights.alpha, weights.beta]
        .iter()
        .any(|t| !t.offset.is_multiple_of(16))
    {
        return Ok(false);
    }
    let Some(p) = pipeline(ctx, workspace.geometry)? else {
        return Ok(false);
    };
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        heads: u32,
    }
    enc.set_pipeline(&p);
    enc.set_bytes(
        0,
        &Args {
            n_in: 2560,
            heads: 48,
        },
    );
    for (slot, tensor) in [
        input,
        weights.beta,
        weights.alpha,
        weights.dt_bias,
        weights.a,
        &workspace.beta,
        &workspace.alpha,
        &workspace.decay,
    ]
    .into_iter()
    .enumerate()
    {
        enc.set_tensor(slot + 1, tensor);
    }
    enc.set_threadgroup_memory(0, 256);
    enc.dispatch(
        MTLSize {
            width: 36,
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

#[cfg(test)]
mod tests;
