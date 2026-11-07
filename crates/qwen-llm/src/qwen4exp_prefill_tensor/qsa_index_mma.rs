use crate::{
    metal::{KernelEncoder, MetalContext, MetalTensor},
    metal_forward::MfError,
    tensor::GgmlType,
};
use objc2_metal::MTLDevice;
crate::env_flag!(
    /// Flash-Next QSA BF16 index projections on non-tensor, pre-Apple10 devices: full-F32 SIMD matrix operations at K2560/N128 or N512 and M512..2048, including partial tiles. BF16 weights are exactly widened to F32; F32 activations and accumulation retain the selector-state precision contract. No new buffers or operand narrowing. Set 0 for scalar-ordered BF16/F32 matmul. Apple10 devices keep their original path even with tensor operations disabled.
    default_on enabled,
    "QWEN_FN_M4_QSA_INDEX_MMA"
);
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode(
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
        || ctx
            .device
            .supportsFamily(objc2_metal::MTLGPUFamily::Apple10)
        || w.dtype != GgmlType::BF16
        || k != 2560
        || !matches!(n, 128 | 512)
        || !(512..=2048).contains(&m)
        || [w, x, y].iter().any(|t| !t.offset.is_multiple_of(16))
    {
        return Ok(false);
    }
    super::small_f32_mma::try_encode_pipeline(
        ctx,
        enc,
        w,
        x,
        y,
        k,
        n,
        m,
        super::small_f32_mma::WeightStorage::Bf16,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

    #[test]
    fn m4_qsa_index_mma_keeps_f32_precision_and_partial_tile_guards() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        if !enabled()
            || ctx.tensor_ops_status().is_available()
            || ctx
                .device
                .supportsFamily(objc2_metal::MTLGPUFamily::Apple10)
        {
            let tiny = MetalTensor::from_bytes(&ctx, &[0, 0], vec![1], GgmlType::BF16).unwrap();
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            crate::metal::dispatch_census_begin();
            assert!(!try_encode(&ctx, &enc, &tiny, &tiny, &tiny, 2560, 512, 2045).unwrap());
            assert!(crate::metal::dispatch_census_take().is_empty());
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd).unwrap();
            return;
        }
        let k = 2560usize;
        let mut state = 0x6d4e3a21u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 20001) as f32 / 10000. - 1.
        };
        for (n, m) in [
            (128usize, 512usize),
            (128, 513),
            (512, 1024),
            (512, 2045),
            (128, 2048),
        ] {
            let w: Vec<half::bf16> = (0..k * n)
                .map(|_| half::bf16::from_f32(next() * 0.02))
                .collect();
            let x: Vec<f32> = (0..k * m).map(|_| next() * 4.).collect();
            let wt = offset_tensor(
                &ctx,
                16,
                bytemuck::cast_slice(&w),
                16,
                vec![k as u64, n as u64],
                GgmlType::BF16,
            );
            let xt = offset_tensor(
                &ctx,
                16,
                bytemuck::cast_slice(&x),
                16,
                vec![k as u64, m as u64],
                GgmlType::F32,
            );
            let out = || {
                offset_tensor(
                    &ctx,
                    16,
                    bytemuck::cast_slice(&vec![f32::NAN; n * m]),
                    16,
                    vec![n as u64, m as u64],
                    GgmlType::F32,
                )
            };
            let old = out();
            let new = out();
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            crate::metal::encode_mat_mat_bf16_f32(&ctx, &enc, &wt, &xt, &old, k, n, m).unwrap();
            crate::metal::dispatch_census_begin();
            assert!(try_encode(&ctx, &enc, &wt, &xt, &new, k, n, m).unwrap());
            let census = crate::metal::dispatch_census_take();
            assert_eq!(census.len(), 1);
            assert_eq!(census[0].kernel, "kernel_qwen4exp_qsa_index_bf16_f32_mma");
            assert_eq!(census[0].grid_width, m.div_ceil(32) as u64);
            assert_eq!(census[0].grid_height, (n / 8) as u64);
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd).unwrap();
            let a = tensor_f32_at_offset(&old);
            let b = tensor_f32_at_offset(&new);
            assert!(a.iter().chain(&b).all(|v| v.is_finite()));
            let mut errors = [0f64; 2];
            // Include every row in the shifted final tile, plus interior samples.
            for row in (m - 32..m).chain((0..64).map(|i| i * (m - 1) / 63)) {
                for col in [0, n / 2, n - 1] {
                    let r = (0..k)
                        .map(|j| f64::from(w[col * k + j].to_f32()) * f64::from(x[row * k + j]))
                        .sum::<f64>();
                    let v = f64::from(b[row * n + col]);
                    assert!(
                        (v - r).abs() <= 3e-5 + 3e-5 * r.abs(),
                        "M{m}/N{n}, row {row}, col {col}: {v} vs {r}"
                    );
                    errors[0] += (f64::from(a[row * n + col]) - r).powi(2);
                    errors[1] += (v - r).powi(2);
                }
            }
            assert!(errors[1] <= 2.25 * errors[0] + 1e-12);
            for t in [&wt, &xt, &old, &new] {
                assert_offset_guards(t, 16, 16);
            }
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            crate::metal::dispatch_census_begin();
            for (bk, bn, bm) in [(k, n, 511), (k, n, 2049), (k, 129, m), (k - 128, n, m)] {
                assert!(!try_encode(&ctx, &enc, &wt, &xt, &new, bk, bn, bm).unwrap());
            }
            let alias = xt.view_subrange(0, vec![n as u64, m as u64]);
            assert!(try_encode(&ctx, &enc, &wt, &xt, &alias, k, n, m).is_err());
            let short = new.view_subrange(0, vec![n as u64, (m - 1) as u64]);
            assert!(try_encode(&ctx, &enc, &wt, &xt, &short, k, n, m).is_err());
            let bad = xt.view_subrange(1, vec![k as u64, (m - 1) as u64]);
            assert!(!try_encode(&ctx, &enc, &wt, &bad, &new, k, n, m - 1).unwrap());
            assert!(crate::metal::dispatch_census_take().is_empty());
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd).unwrap();
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin_concurrent(&cmd);
            crate::metal::dispatch_census_begin();
            assert!(try_encode(&ctx, &enc, &wt, &xt, &new, k, n, m).is_err());
            assert!(crate::metal::dispatch_census_take().is_empty());
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd).unwrap();
        }
    }
}
