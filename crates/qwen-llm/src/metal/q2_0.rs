use super::moe::{
    MoeDecodeArgs, checked_moe_decode_args, checked_moe_product, validate_moe_decode_tensor,
};
use super::*;

#[allow(clippy::too_many_arguments)]
fn checked_q2_0_args(
    kernel: &'static str,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<MoeDecodeArgs, MetalError> {
    let args = checked_moe_decode_args(kernel, n_in, n_out, n_expert, topk)?;
    if !n_in.is_multiple_of(64) {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("n_in={n_in} not divisible by 64"),
        });
    }
    let bank = checked_moe_product(kernel, "weight bank", &[n_in, n_out, n_expert])?;
    let inputs = checked_moe_product(kernel, "input", &[n_in, topk])?;
    let outputs = checked_moe_product(kernel, "output", &[n_out, topk])?;
    validate_moe_decode_tensor(kernel, "weight", weight, bank, &[GgmlType::Q2_0], false, 2)?;
    validate_moe_decode_tensor(kernel, "input", input, inputs, &[GgmlType::F32], false, 4)?;
    validate_moe_decode_tensor(kernel, "output", output, outputs, &[GgmlType::F32], true, 4)?;
    Ok(args)
}

/// Dense Q2_0 matrix `[n_in, n_out]` times an F32 vector `[n_in]`.
/// Requires positive dimensions and `n_in % 64 == 0`; writes F32 `[n_out]`.
pub fn encode_mat_vec_q2_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    let args = checked_q2_0_args("mat_vec_q2_0", weight, x, y, n_in, n_out, 1, 1)?;
    let pso = ctx.pipeline("kernel_mat_vec_q2_0_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(4),
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

/// Q2_0 bank `[n_in, n_out, n_expert]` times F32 inputs `[n_in, topk]`,
/// producing F32 `[n_out, topk]` for every selected slot in one dispatch.
/// Route IDs are I32 or legacy F32-labelled integer bits, never float values.
/// Out-of-range route IDs produce zeros. Requires `n_in % 64 == 0` and
/// `0 < topk <= n_expert`. Tensor sizes, offsets and writability are checked.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_down_q2_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    topk_idx: &MetalTensor,
    expert_out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    topk: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "moe_down_q2_0";
    let args = checked_q2_0_args(
        KERNEL, weight, inner, expert_out, n_in, n_out, n_expert, topk,
    )?;
    validate_moe_decode_tensor(
        KERNEL,
        "top-k indices",
        topk_idx,
        topk,
        &[GgmlType::I32, GgmlType::F32],
        false,
        4,
    )?;
    let pso = ctx.pipeline("kernel_moe_down_q2_0_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, weight);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, topk_idx);
    enc.set_tensor(4, expert_out);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(4),
            height: topk,
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

#[cfg(test)]
mod tests;
