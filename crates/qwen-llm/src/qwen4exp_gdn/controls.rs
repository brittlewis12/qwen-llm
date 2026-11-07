use super::*;
use objc2_metal::MTLComputePipelineState;

crate::env_flag!(
    /// Flash-Next one-token H2560/48-head GDN on the Metal tensor-capable tier combines independent F32 alpha/beta projections and the decay epilogue. Reuses existing scratch and reduction orders. Set 0 for separate dispatches. M4 and unsupported geometry, alignment or PSOs, and QWEN_MATVEC_F32_LCPP_R2=0, retain the incumbent.
    default_on enabled,
    "QWEN_FN_GDN_CONTROLS_FUSED"
);

crate::env_flag!(
    /// Flash-Next one-token M5 GDN at H2560/key16/value48/HD128: combine controls, causal conv+SiLU and Q/K L2 normalization in one dispatch. Reuses existing state/scratch and reduction orders. Set 0 to keep the controls dispatch plus separate conv/norm. Also respects QWEN_FN_GDN_CONTROLS_FUSED=0 and QWEN_L2_PAIR_HD128_R4=0.
    default_on prep_enabled,
    "QWEN_FN_GDN_PREP_FUSED"
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fusion {
    Incumbent,
    Controls,
    ControlsAndPrep,
}

const PREP: &str = "kernel_qwen4exp_gdn_controls_prep_f32";
const KERNEL: &str = "kernel_qwen4exp_gdn_controls_f32";

fn eligible(g: GatedDeltaNetMetalGeometry) -> bool {
    enabled()
        && crate::metal::mat_vec_f32_cooperative_enabled()
        && (g.hidden_size, g.value_heads, g.head_dim) == (2560, 48, 128)
}

fn pipeline(
    ctx: &MetalContext,
    g: GatedDeltaNetMetalGeometry,
    prep: bool,
) -> Result<Option<crate::metal::Pipeline>, MetalError> {
    if !eligible(g) || !ctx.tensor_ops_status().is_available() {
        return Ok(None);
    }
    if prep
        && (!prep_enabled() || g.key_heads != 16 || !crate::metal::l2_pair_cooperative_enabled())
    {
        return Ok(None);
    }
    match ctx.pipeline(if prep { PREP } else { KERNEL }) {
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
    pipeline(ctx, g, false)?;
    pipeline(ctx, g, true)?;
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
) -> Result<Fusion, MetalError> {
    if [input, weights.alpha, weights.beta]
        .iter()
        .any(|t| !t.offset.is_multiple_of(16))
    {
        return Ok(Fusion::Incumbent);
    }
    let (p, prepared) = if let Some(p) = pipeline(ctx, workspace.geometry, true)? {
        (p, true)
    } else if let Some(p) = pipeline(ctx, workspace.geometry, false)? {
        (p, false)
    } else {
        return Ok(Fusion::Incumbent);
    };
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        heads: u32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct PrepArgs {
        n_in: u32,
        heads: u32,
        eps: f32,
    }
    enc.set_pipeline(&p);
    if prepared {
        enc.set_bytes(
            0,
            &PrepArgs {
                n_in: 2560,
                heads: 48,
                eps: workspace.geometry.eps,
            },
        );
    } else {
        enc.set_bytes(
            0,
            &Args {
                n_in: 2560,
                heads: 48,
            },
        );
    }
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
    if prepared {
        for (slot, tensor) in [
            &workspace.qkv,
            &workspace.conv_state,
            weights.conv,
            &workspace.qkv_conv,
            &workspace.query_norm,
            &workspace.key_norm,
        ]
        .into_iter()
        .enumerate()
        {
            enc.set_tensor(slot + 9, tensor);
        }
    }
    enc.set_threadgroup_memory(0, 256);
    enc.dispatch(
        MTLSize {
            width: if prepared { 56 } else { 36 },
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(if prepared {
        Fusion::ControlsAndPrep
    } else {
        Fusion::Controls
    })
}

#[cfg(test)]
mod tests;
