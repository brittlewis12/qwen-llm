use super::*;

pub(super) fn eligible(scratch: &GatedResidualMetalScratch, dtype: GgmlType) -> bool {
    (scratch.branch_count, scratch.hidden_size, scratch.low_rank) == (4, 2560, 320)
        && dtype == GgmlType::Q8_0
}

pub(crate) fn preflight(ctx: &MetalContext) -> Result<(), Qwen4ExpMetalError> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_up_mix_q8_k320")?;
    if pipeline.threadExecutionWidth() != 32
        || pipeline.maxTotalThreadsPerThreadgroup() < 128
        || pipeline.staticThreadgroupMemoryLength() > ctx.device.maxThreadgroupMemoryLength()
    {
        return Err(invalid(
            "HC up-mix requires width32 and a128-thread cooperative pipeline",
        ));
    }
    Ok(())
}

pub(super) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    up: &MetalTensor,
    scratch: &GatedResidualMetalScratch,
) -> Result<(), MetalError> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_up_mix_q8_k320")?;
    enc.set_pipeline(&pipeline);
    for (index, tensor) in [
        up,
        &scratch.low,
        &scratch.normalized,
        &scratch.raw_gate,
        &scratch.mixed,
    ]
    .into_iter()
    .enumerate()
    {
        enc.set_tensor(index, tensor);
    }
    enc.dispatch(
        MTLSize {
            width: 640,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
