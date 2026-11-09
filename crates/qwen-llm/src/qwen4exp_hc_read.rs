use super::*;

crate::env_flag!(
    /// Flash-Next BF16 HC decode at branches4/H2560/rank320: fuse down projection with SiLU and prepare F32 injection before the block when its incumbent uses 1024 lanes. Reuses existing buffers and row/reduction order. Set 0 for the incumbent; unsupported PSOs/alignments retain it. Default on; this also respects QWEN_MAT_VEC_BF16_TG=0.
    default_on enabled,
    "QWEN_FN_HC_READ_FUSED"
);

pub(super) fn route_enabled() -> bool {
    enabled()
}

const DOWN: &str = "kernel_qwen4exp_hc_down_silu_bf16";
pub(super) fn incumbent_tg_enabled() -> bool {
    static TG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TG.get_or_init(|| std::env::var("QWEN_MAT_VEC_BF16_TG").as_deref() != Ok("0"))
}

const INJECT: &str = "kernel_qwen4exp_hc_down_silu_inject_bf16";

// All tensor/device/disjointness checks precede this private encoder in the
// parent HC contract. None means incumbent; Some(true) also prepared injection.
pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    down: &MetalTensor,
    inject: Option<&MetalTensor>,
    scratch: &GatedResidualMetalScratch,
) -> Result<Option<bool>, MetalError> {
    if !enabled()
        || !incumbent_tg_enabled()
        || (scratch.branch_count, scratch.hidden_size, scratch.low_rank) != (4, 2560, 320)
        || down.dtype != GgmlType::BF16
        || !down.offset.is_multiple_of(16)
        || !scratch.normalized.offset.is_multiple_of(16)
    {
        return Ok(None);
    }
    let prepared = inject.is_some_and(|w| w.dtype == GgmlType::F32)
        && hc_combine_wide()
        && ctx
            .pipeline("kernel_qwen4exp_hc_combine_f32")?
            .maxTotalThreadsPerThreadgroup()
            >= 1024;
    let pipeline = match ctx.pipeline(if prepared { INJECT } else { DOWN }) {
        Ok(p)
            if p.threadExecutionWidth() == 32
                && p.maxTotalThreadsPerThreadgroup() >= 128
                && p.staticThreadgroupMemoryLength() <= ctx.device.maxThreadgroupMemoryLength() =>
        {
            p
        }
        Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => return Ok(None),
        Err(e) => return Err(e),
    };
    enc.set_pipeline(&pipeline);
    enc.set_tensor(0, down);
    enc.set_tensor(1, &scratch.normalized);
    enc.set_tensor(2, &scratch.low);
    if prepared {
        enc.set_tensor(3, inject.unwrap());
        enc.set_tensor(4, &scratch.injection);
    }
    enc.set_threadgroup_memory(0, 0);
    enc.dispatch(
        MTLSize {
            width: if prepared { 324 } else { 320 },
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(Some(prepared))
}

#[cfg(test)]
#[path = "qwen4exp_hc_read/tests.rs"]
mod tests;
