use super::*;

pub(crate) const KERNEL: &str = "kernel_qwen4exp_topk_guarded_f32";

pub(crate) fn supported(ctx: &MetalContext) -> Result<bool, Qwen4ExpMoeError> {
    let p = ctx.pipeline(KERNEL)?;
    Ok(p.threadExecutionWidth() == 32
        && p.maxTotalThreadsPerThreadgroup() >= 512
        && p.staticThreadgroupMemoryLength()
            .checked_add(6144)
            .is_some_and(|bytes| bytes <= ctx.device.maxThreadgroupMemoryLength()))
}

pub(super) fn eligible(n: usize, k: usize) -> bool {
    (n, k) == (512, 10)
}

pub(crate) fn preflight(ctx: &MetalContext) -> Result<(), Qwen4ExpMoeError> {
    require_pipeline_capacity(ctx, KERNEL, 512, 6144)?;
    if ctx.pipeline(KERNEL)?.threadExecutionWidth() != 32 {
        return invalid("guarded top-k requires SIMD width32");
    }
    Ok(())
}

pub(crate) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    ids: &MetalTensor,
    weights: &MetalTensor,
) -> Result<(), Qwen4ExpMoeError> {
    validate_encoder(ctx, enc)?;
    require_tensor("guarded top-k logits", logits, GgmlType::F32, &[512], false)?;
    require_tensor("guarded top-k IDs", ids, GgmlType::I32, &[10], true)?;
    require_tensor("guarded top-k weights", weights, GgmlType::F32, &[10], true)?;
    let tensors = [("logits", logits), ("IDs", ids), ("weights", weights)];
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)?;
    preflight(ctx)?;
    enc.set_pipeline(&ctx.pipeline(KERNEL)?);
    enc.set_bytes_slice(0, &[512u32, 10]);
    enc.set_tensor(1, logits);
    enc.set_tensor(2, ids);
    enc.set_tensor(3, weights);
    for slot in 0..3 {
        enc.set_threadgroup_memory(slot, 2048);
    }
    enc.dispatch(
        objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 512,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
