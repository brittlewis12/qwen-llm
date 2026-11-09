//! Shared mixer projections for independent native-decode states.
use super::*;
crate::env_flag!(
    /// Share Q6_K mixer projections across independent native-decode states, with separate MMA and transport rollbacks below.
    /// 0 uses ordinary member projections. Prefill and single decode retain their existing routes.
    default_on enabled, "QWEN_FN_NATIVE_DENSE_BATCH"
);
crate::env_flag!(
    /// Independent decode shared dense/head projections with full-F32 operand few-row MMA. Changes reduction order; 0 restores exact multicolumn Q6. Does not select half operands.
    default_on mma_flag,"QWEN_FN_BATCH_DENSE_MMA"
);
pub(crate) fn mma_enabled() -> bool {
    mma_flag()
}
crate::env_flag!(
    /// Gather/scatter independent dense projection columns with one member-axis launch each. 0 keeps ordinary per-member copies. Arithmetic, buffers and causal state are unchanged.
    default_on copy_enabled, "QWEN_FN_BATCH_PROJECTION_COPY"
);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CopyArgs {
    width: u32,
    rows: u32,
}

fn copy_columns(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    private: &[&MetalTensor],
    matrix: &MetalTensor,
    width: usize,
    rows: usize,
    gather: bool,
) -> Result<(), crate::metal_forward::MfError> {
    enc.set_pipeline(&ctx.pipeline(if gather {
        "kernel_fn_batch_gather_columns"
    } else {
        "kernel_fn_batch_scatter_columns"
    })?);
    enc.set_bytes(
        0,
        &CopyArgs {
            width: width as u32,
            rows: rows as u32,
        },
    );
    for i in 0..4 {
        enc.set_tensor(1 + i, private.get(i).unwrap_or(&private[0]));
    }
    enc.set_tensor(5, matrix);
    enc.dispatch(
        objc2_metal::MTLSize {
            width: width.div_ceil(256),
            height: rows,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
pub(crate) struct Scratch {
    pub inputs: MetalTensor,
    pub outputs: MetalTensor,
}
impl Scratch {
    pub fn project(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        w: &MetalTensor,
        inputs: &[&MetalTensor],
        outputs: &[&MetalTensor],
    ) -> Result<(), crate::metal_forward::MfError> {
        let k = w.shape[0] as usize;
        let m = w.shape[1] as usize;
        let b = inputs.len();
        let mma = mma_enabled();
        if !enabled() || !(2..=4).contains(&b) || w.dtype != GgmlType::Q6_K {
            for (x, y) in inputs.iter().zip(outputs) {
                encode_mat_vec_dispatch(ctx, enc, w, x, y, k, m)?;
            }
            return Ok(());
        }
        let columns = b.next_power_of_two();
        let x = self.inputs.view_subrange(0, vec![k as u64, columns as u64]);
        let y = self
            .outputs
            .view_subrange(0, vec![m as u64, columns as u64]);
        if copy_enabled() {
            copy_columns(ctx, enc, inputs, &x, k, if mma { b } else { columns }, true)?;
        } else {
            for (i, t) in inputs.iter().enumerate() {
                crate::metal::encode_copy_offset_f32(
                    ctx,
                    enc,
                    t,
                    0,
                    &x.view_subrange((i * k) as u64, vec![k as u64]),
                    k,
                )?;
            }
            if b == 3 && !mma {
                crate::metal::encode_copy_offset_f32(
                    ctx,
                    enc,
                    inputs[0],
                    0,
                    &x.view_subrange((3 * k) as u64, vec![k as u64]),
                    k,
                )?;
            }
        }
        if mma {
            crate::metal::encode_mat_mat_fewrow_gen_f32_weights(
                ctx,
                enc,
                w,
                &x.view_subrange(0, vec![k as u64, b as u64]),
                &y.view_subrange(0, vec![m as u64, b as u64]),
                k,
                m,
                b,
            )?;
        } else {
            crate::metal::encode_mat_vec_nc_dispatch(ctx, enc, w, &x, &y, k, m, columns)?;
        }
        if copy_enabled() {
            copy_columns(ctx, enc, outputs, &y, m, b, false)?;
        } else {
            for (i, t) in outputs.iter().enumerate() {
                crate::metal::encode_copy_offset_f32(
                    ctx,
                    enc,
                    &y.view_subrange((i * m) as u64, vec![m as u64]),
                    0,
                    t,
                    m,
                )?;
            }
        }
        Ok(())
    }
}
