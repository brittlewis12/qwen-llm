use super::*;
crate::env_flag!(
    /// Flash-Next Q24/KV2/D256 decode logits at >=256 keys: reuse four F32 query heads across eight keys per SIMD. Same dimension/FMA/SIMD sum order and scalar invalid-ID/padding behavior, F16 cache, no new buffers. Default on; set 0 for the one-head decoder. Unsupported PSOs keep the incumbent.
    default_on enabled,
    "QWEN_FN_DECODE_QSA_REUSE"
);
pub(super) fn batch_enabled() -> bool {
    enabled()
}
pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w: &QwenSparseAttentionMetalWorkspace,
    ids: usize,
) -> Result<bool, MetalError> {
    let g = w.geometry;
    if !enabled() || !selected_value::geometry_qualified(g) || ids < 256 || ids > g.output_width() {
        return Ok(false);
    }
    let p = match ctx.pipeline("kernel_qwen4exp_qsa_decode_logits_reuse_f32") {
        Ok(p)
            if p.threadExecutionWidth() == 32
                && p.maxTotalThreadsPerThreadgroup() >= 128
                && p.staticThreadgroupMemoryLength() <= ctx.device.maxThreadgroupMemoryLength() =>
        {
            p
        }
        Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => return Ok(false),
        Err(e) => return Err(e),
    };
    enc.set_pipeline(&p);
    enc.set_bytes(0, &attention_args(g, ids));
    enc.set_tensor(1, &w.query);
    enc.set_tensor(2, &w.key_cache);
    enc.set_tensor(3, &w.token_ids);
    enc.set_tensor(4, &w.attention_logits);
    enc.dispatch(
        MTLSize {
            width: 3 * ids.div_ceil(32),
            height: 2,
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
