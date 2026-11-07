use super::*;

crate::env_flag!(
    /// Flash-Next one-token HC attention-to-FFN boundary: combine the prepared injection and the next RMS normalization in one dispatch. H2560/branches4 and incumbent 1024-lane norm only; reuses existing buffers. Set 0 for the two dispatches; unsupported PSOs retain them.
    default_on enabled,
    "QWEN_FN_HC_BOUNDARY_FUSED"
);

// The consumed HC read and the parent next-read validator establish device,
// range, dtype and disjointness custody before this private leaf is called.
pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    output: &MetalTensor,
    raw: &MetalTensor,
    residual: &MetalTensor,
    norm: &MetalTensor,
    scratch: &GatedResidualMetalScratch,
    eps: f32,
) -> Result<bool, MetalError> {
    if !enabled()
        || (scratch.branch_count, scratch.hidden_size) != (4, 2560)
        || ctx
            .pipeline("kernel_qwen4exp_hc_rms_norm_f32")?
            .maxTotalThreadsPerThreadgroup()
            < 1024
    {
        return Ok(false);
    }
    let p = match ctx.pipeline("kernel_qwen4exp_hc_inject_norm_f32") {
        Ok(p)
            if p.threadExecutionWidth() == 32
                && p.maxTotalThreadsPerThreadgroup() >= 1024
                && p.staticThreadgroupMemoryLength() + 128
                    <= ctx.device.maxThreadgroupMemoryLength() =>
        {
            p
        }
        Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => return Ok(false),
        Err(e) => return Err(e),
    };
    enc.set_pipeline(&p);
    enc.set_bytes(
        0,
        &HcNormArgs {
            branch_count: 4,
            hidden_size: 2560,
            eps,
        },
    );
    for (i, t) in [output, raw, residual, norm, &scratch.normalized]
        .into_iter()
        .enumerate()
    {
        enc.set_tensor(i + 1, t);
    }
    enc.set_threadgroup_memory(0, 128);
    enc.dispatch(
        MTLSize {
            width: 4,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
    Ok(true)
}
