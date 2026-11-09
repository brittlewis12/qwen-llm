//! Gather independent vectors for existing exact multicolumn Q6 projections.
use super::*;
crate::env_flag!(
    /// Share exact Q6_K dense projections across independent native-decode states.
    /// 0 uses ordinary member projections. Prefill and single decode are unchanged.
    default_on enabled, "QWEN_FN_NATIVE_DENSE_BATCH"
);
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
        if b == 3 {
            crate::metal::encode_copy_offset_f32(
                ctx,
                enc,
                inputs[0],
                0,
                &x.view_subrange((3 * k) as u64, vec![k as u64]),
                k,
            )?;
        }
        crate::metal::encode_mat_vec_nc_dispatch(ctx, enc, w, &x, &y, k, m, columns)?;
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
        Ok(())
    }
}
