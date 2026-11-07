use super::*;
crate::env_flag!(
    /// Flash-Next M4 IQ2_S/IQ3_S gate/up at H2560/F640/E512/top10 and M2048: compact 64x32x32 half-operand/F32-accumulator tiles, skipping upper-half MMA work for buckets with <=16 slots. One bounded map dispatch in dead routed-down storage, no allocation. Set 0 for the original rectangular N16 route. Apple10 devices retain their existing routes.
    default_on enabled,
    "QWEN_FN_M4_EXPERT_TILE32"
);
pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    input: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    dead: &MetalTensor,
    g: Qwen4ExpMoeMetalGeometry,
    tokens: usize,
) -> Result<bool, MetalError> {
    if !enabled()
        || ctx.tensor_ops_status().is_available()
        || ctx
            .device
            .supportsFamily(objc2_metal::MTLGPUFamily::Apple10)
        || (
            g.hidden_size,
            g.routed_intermediate_size,
            g.expert_count,
            g.experts_per_token,
            tokens,
        ) != (2560, 640, 512, 10, 2048)
        || gate.dtype != up.dtype
        || !matches!(gate.dtype, GgmlType::IQ2_S | GgmlType::IQ3_S)
    {
        return Ok(false);
    }
    crate::metal::try_encode_moe_swiglu_mapped32(
        ctx,
        enc,
        gate,
        up,
        input,
        counts,
        ids,
        out,
        dead,
        g.hidden_size,
        g.routed_intermediate_size,
        g.expert_count,
        g.experts_per_token,
        tokens,
    )
}
