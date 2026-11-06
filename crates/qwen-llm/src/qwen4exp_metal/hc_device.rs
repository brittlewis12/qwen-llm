use super::*;

crate::env_flag!(
    /// BF16 device-input HC prompt projections for Flash-Next at M2048 on the Metal 4 tier, default on. Preserves the incumbent BF16 activation conversion and F32 sums; reuses dead raw-gate/mixed storage. Set to 0 to restore the incumbent. QWEN_MATMAT_BF16_BFLOAT_ACT=0 disables this route too.
    default_on enabled,
    "QWEN_FN_PREFILL_HC_DEVICE"
);

pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    dead: &MetalTensor,
    k: usize,
    n: usize,
    m: usize,
) -> Result<bool, MfError> {
    if !enabled()
        || !ctx.tensor_ops_status().is_available()
        || !crate::metal_forward::matmat_bf16_bfloat_act_enabled()
        || weight.dtype != GgmlType::BF16
        || m != 2048
        || !matches!((k, n), (10240, 320) | (320, 10240))
        || !dead.offset.is_multiple_of(16)
        || !weight.offset.is_multiple_of(16)
    {
        return Ok(false);
    }
    let p = match ctx.tensor_pipeline("kernel_qwen4exp_hc_bf16_device") {
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
    let convert = ctx.pipeline("kernel_qwen4exp_hc_convert_bf16")?;
    use crate::metal::checks::{
        bad_shape, check_disjoint, check_physical, check_tensor, require_serial,
    };
    require_serial("hc_device", enc)?;
    check_tensor(
        "hc_device",
        weight,
        GgmlType::BF16,
        &[k as u64, n as u64],
        false,
        "weight",
    )?;
    check_tensor(
        "hc_device",
        input,
        GgmlType::F32,
        &[k as u64, m as u64],
        false,
        "input",
    )?;
    check_tensor(
        "hc_device",
        output,
        GgmlType::F32,
        &[n as u64, m as u64],
        true,
        "output",
    )?;
    check_physical("hc_device", dead, 16, true, "dead scratch")?;
    if dead.dtype != GgmlType::F32 || dead.n_elements() < (k * m / 2) as u64 {
        return Err(bad_shape("hc_device", "dead BF16 operand capacity").into());
    }
    let mut bf = dead.view_subrange(0, vec![(k * m / 2) as u64]);
    bf.dtype = GgmlType::BF16;
    bf.shape = vec![k as u64, m as u64];
    check_disjoint(
        "hc_device",
        &bf,
        &[(weight, "weight"), (input, "input"), (output, "output")],
    )?;
    check_disjoint("hc_device", output, &[(weight, "weight"), (input, "input")])?;
    enc.set_pipeline(&convert);
    enc.set_bytes(0, &((k * m) as u32));
    enc.set_tensor(1, input);
    enc.set_tensor(2, &bf);
    enc.dispatch(
        MTLSize {
            width: (k * m).div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
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
    enc.set_tensor(1, weight);
    enc.set_tensor(2, &bf);
    enc.set_tensor(3, output);
    enc.set_threadgroup_memory(0, 0);
    enc.dispatch(
        MTLSize {
            width: m / 128,
            height: n.div_ceil(64),
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
