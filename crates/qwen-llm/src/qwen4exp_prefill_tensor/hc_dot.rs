use crate::{
    metal::{KernelEncoder, MetalContext, MetalError, MetalTensor},
    metal_forward::MfError,
    tensor::GgmlType,
};
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLSize};
crate::env_flag!(
    /// Flash-Next HC injection at H10240/N4 and 128-aligned M512..M2048 on tensor-tier devices: coalesced F32 four-output dot products with SIMD reduction. No narrowing or new buffers. Set 0 to restore the original tensor tile. QWEN_FN_SHORT_SMALL_F32=0 or QWEN_FN_SMALL_F32_TILES=0 also disables this route.
    default_on enabled,
    "QWEN_FN_PREFILL_HC_DOT"
);
pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    n: usize,
    m: usize,
) -> Result<bool, MfError> {
    if !enabled()
        || !ctx.tensor_ops_status().is_available()
        || w.dtype != GgmlType::F32
        || !(512..=2048).contains(&m)
        || !m.is_multiple_of(128)
        || (k, n) != (10240, 4)
        || [w, x, y].iter().any(|t| !t.offset.is_multiple_of(16))
    {
        return Ok(false);
    }
    let p = match ctx.pipeline("kernel_qwen4exp_hc_inject_dot4_f32") {
        Ok(p)
            if p.threadExecutionWidth() == 32
                && p.maxTotalThreadsPerThreadgroup() >= 128
                && p.staticThreadgroupMemoryLength() <= ctx.device.maxThreadgroupMemoryLength() =>
        {
            p
        }
        Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => return Ok(false),
        Err(e) => return Err(e.into()),
    };
    use crate::metal::checks::{check_disjoint, check_tensor, require_serial};
    require_serial("fn_hc_dot", enc)?;
    check_tensor(
        "fn_hc_dot",
        w,
        GgmlType::F32,
        &[k as u64, n as u64],
        false,
        "weight",
    )?;
    check_tensor(
        "fn_hc_dot",
        x,
        GgmlType::F32,
        &[k as u64, m as u64],
        false,
        "input",
    )?;
    check_tensor(
        "fn_hc_dot",
        y,
        GgmlType::F32,
        &[n as u64, m as u64],
        true,
        "output",
    )?;
    check_disjoint("fn_hc_dot", y, &[(w, "weight"), (x, "input")])?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        k: u32,
        n: u32,
        m: u32,
    }
    enc.set_pipeline(&p);
    enc.set_bytes(
        0,
        &Args {
            k: k as u32,
            n: n as u32,
            m: m as u32,
        },
    );
    enc.set_tensor(1, w);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);

    enc.dispatch(
        MTLSize {
            width: m / 4,
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
