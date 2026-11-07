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

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

    #[test]
    fn hc_dot_refuses_unqualified_views_and_aliases_before_dispatch() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        if !enabled() || !ctx.tensor_ops_status().is_available() {
            return;
        }
        let (k, n, m) = (10240usize, 4usize, 512usize);
        let w = MetalTensor::zeros_f32(&ctx, vec![k as u64, n as u64]).unwrap();
        let x = MetalTensor::zeros_f32(&ctx, vec![k as u64, m as u64]).unwrap();
        let y = MetalTensor::zeros_f32(&ctx, vec![n as u64, m as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        for (bad_k, bad_n, bad_m) in [
            (k, n, 511),
            (k, n, 513),
            (k, n, 2176),
            (k, 5, m),
            (k - 128, n, m),
        ] {
            assert!(!try_encode(&ctx, &enc, &w, &x, &y, bad_k, bad_n, bad_m).unwrap());
        }
        let storage = MetalTensor::zeros_f32(&ctx, vec![(k * m + 1) as u64]).unwrap();
        let unaligned = storage.view_subrange(1, vec![k as u64, m as u64]);
        assert!(!try_encode(&ctx, &enc, &w, &unaligned, &y, k, n, m).unwrap());
        let alias = x.view_subrange(0, vec![n as u64, m as u64]);
        assert!(try_encode(&ctx, &enc, &w, &x, &alias, k, n, m).is_err());
        let short = y.view_subrange(0, vec![n as u64, (m - 1) as u64]);
        assert!(try_encode(&ctx, &enc, &w, &x, &short, k, n, m).is_err());
        assert!(crate::metal::dispatch_census_take().is_empty());
        enc.end();
        command.commit();
        crate::metal::wait_completed(&command).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin_concurrent(&command);
        crate::metal::dispatch_census_begin();
        assert!(try_encode(&ctx, &enc, &w, &x, &y, k, n, m).is_err());
        assert!(crate::metal::dispatch_census_take().is_empty());
        enc.end();
        command.commit();
        crate::metal::wait_completed(&command).unwrap();
    }
}
