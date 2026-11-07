use crate::{
    metal::{KernelEncoder, MetalContext, MetalError, MetalTensor},
    metal_forward::MfError,
    tensor::GgmlType,
};
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLSize};
crate::env_flag!(
    /// Flash-Next narrow F32 prefill projections on non-tensor devices: use full-F32 SIMD matrix operations at H10240/N4 and H2560/N48, 128-aligned M512..M2048. Same operands, no narrowing or new buffers. Set 0 for the scalar-ordered incumbent. Also respects QWEN_FN_SHORT_SMALL_F32=0 and QWEN_FN_SMALL_F32_TILES=0.
    default_on enabled,
    "QWEN_FN_PREFILL_F32_MMA"
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
        || ctx.tensor_ops_status().is_available()
        || w.dtype != GgmlType::F32
        || !(512..=2048).contains(&m)
        || !m.is_multiple_of(128)
        || !matches!((k, n), (10240, 4) | (2560, 48))
        || [w, x, y].iter().any(|t| !t.offset.is_multiple_of(16))
    {
        return Ok(false);
    }
    let p = match ctx.pipeline("kernel_qwen4exp_small_f32_mma_k128_s4") {
        Ok(p)
            if p.threadExecutionWidth() == 32
                && p.maxTotalThreadsPerThreadgroup() >= 128
                && p.staticThreadgroupMemoryLength() + 4096
                    <= ctx.device.maxThreadgroupMemoryLength() =>
        {
            p
        }
        Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => return Ok(false),
        Err(e) => return Err(e.into()),
    };
    use crate::metal::checks::{check_disjoint, check_tensor, require_serial};
    require_serial("fn_f32_mma", enc)?;
    check_tensor(
        "fn_f32_mma",
        w,
        GgmlType::F32,
        &[k as u64, n as u64],
        false,
        "weight",
    )?;
    check_tensor(
        "fn_f32_mma",
        x,
        GgmlType::F32,
        &[k as u64, m as u64],
        false,
        "input",
    )?;
    check_tensor(
        "fn_f32_mma",
        y,
        GgmlType::F32,
        &[n as u64, m as u64],
        true,
        "output",
    )?;
    check_disjoint("fn_f32_mma", y, &[(w, "weight"), (x, "input")])?;
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
    enc.set_threadgroup_memory(0, 4096);
    enc.dispatch(
        MTLSize {
            width: m / 32,
            height: n.div_ceil(8),
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
