//! Cooperative F32 GEMV shared across a small number of independent inputs.
use super::*;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLResource};

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode_mat_vec_f32_members(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    b: usize,
) -> Result<bool, MetalError> {
    use super::checks::{bad_shape, check_disjoint, require_serial};
    use super::moe::{checked_moe_product, validate_moe_decode_tensor};
    const K: &str = "mat_vec_f32_members";
    require_serial(K, enc)?;
    if k == 0 || !k.is_multiple_of(4) || m == 0 || !(2..=4).contains(&b) {
        return Err(bad_shape(
            K,
            "positive four-aligned K, positive M and two to four members required",
        ));
    }
    let ku = u32::try_from(k).map_err(|_| bad_shape(K, "K exceeds u32"))?;
    let mu = u32::try_from(m).map_err(|_| bad_shape(K, "M exceeds u32"))?;
    for (name, t, dims, writable, align) in [
        ("weight", w, [k, m], false, 16),
        ("input", x, [k, b], false, 16),
        ("output", y, [m, b], true, 4),
    ] {
        let n = checked_moe_product(K, name, &dims)?;
        validate_moe_decode_tensor(K, name, t, n, &[GgmlType::F32], writable, align)?;
    }
    check_disjoint(K, y, &[(w, "weight"), (x, "input")])?;
    let device = ctx.device.registryID();
    if enc.parent_command_buffer().device().registryID() != device
        || [w, x, y]
            .into_iter()
            .any(|t| t.buffer.device().registryID() != device)
    {
        return Err(bad_shape(
            K,
            "encoder and bindings must belong to the context device",
        ));
    }
    let p = ctx.pipeline(&format!("kernel_mat_vec_f32_members_b{b}"))?;
    if p.threadExecutionWidth() != 32
        || p.maxTotalThreadsPerThreadgroup() < 128
        || p.staticThreadgroupMemoryLength() > ctx.device.maxThreadgroupMemoryLength()
    {
        return Ok(false);
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        k: u32,
        m: u32,
        b: u32,
    }
    enc.set_pipeline(&p);
    enc.set_bytes(
        0,
        &Args {
            k: ku,
            m: mu,
            b: b as u32,
        },
    );
    enc.set_tensor(1, w);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.dispatch(
        MTLSize {
            width: m.div_ceil(2),
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
#[path = "mat_vec_members/tests.rs"]
mod tests;
