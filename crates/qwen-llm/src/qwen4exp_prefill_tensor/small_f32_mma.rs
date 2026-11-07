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
crate::env_flag!(
    /// Flash-Next F32 router projections on non-tensor, pre-Apple10 devices at H2560/E512 and qualified 512/1024/2048-row E8P32 widths: reuse full-F32 SIMD matrix operations, without narrowing or new buffers. Set 0 for the strict E8P32 incumbent. The existing strict-router off-switch retains priority; Apple10 devices keep their original route even with tensor operations disabled.
    default_on router_enabled,
    "QWEN_FN_M4_ROUTER_MMA"
);
#[cfg(test)]
thread_local! {
    static ROUTER_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Keep the strict incumbent differential independent of optional fast routes.
#[cfg(test)]
pub(crate) fn without_router_mma<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ROUTER_DISABLED.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(ROUTER_DISABLED.with(|s| s.replace(true)));
    f()
}

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
    let router = (k, n) == (2560, 512);
    #[cfg(test)]
    if router && ROUTER_DISABLED.with(|s| s.get()) {
        return Ok(false);
    }
    if (if router {
        !router_enabled()
    } else {
        !enabled()
    }) || (router
        && ctx
            .device
            .supportsFamily(objc2_metal::MTLGPUFamily::Apple10))
        || ctx.tensor_ops_status().is_available()
        || w.dtype != GgmlType::F32
        || !(512..=2048).contains(&m)
        || !m.is_multiple_of(128)
        || !matches!((k, n), (10240, 4) | (2560, 48 | 512))
        || [w, x, y].iter().any(|t| !t.offset.is_multiple_of(16))
    {
        return Ok(false);
    }
    try_encode_pipeline(ctx, enc, w, x, y, k, n, m, WeightStorage::F32)
}

pub(super) enum WeightStorage {
    F32,
    Bf16,
}

/// Call only after the family route has qualified dimensions and alignment.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_encode_pipeline(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    n: usize,
    m: usize,
    storage: WeightStorage,
) -> Result<bool, MfError> {
    let (name, dtype) = match storage {
        WeightStorage::F32 => ("kernel_qwen4exp_small_f32_mma_k128_s4", GgmlType::F32),
        WeightStorage::Bf16 => ("kernel_qwen4exp_qsa_index_bf16_f32_mma", GgmlType::BF16),
    };
    let p = match ctx.pipeline(name) {
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
        dtype,
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
            width: m.div_ceil(32),
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

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

    #[test]
    fn m4_router_mma_matches_f64_with_offset_guards() {
        use crate::metal::test_support::*;
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        if !router_enabled()
            || ctx.tensor_ops_status().is_available()
            || ctx
                .device
                .supportsFamily(objc2_metal::MTLGPUFamily::Apple10)
        {
            let tiny = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            crate::metal::dispatch_census_begin();
            assert!(!try_encode(&ctx, &enc, &tiny, &tiny, &tiny, 2560, 512, 512).unwrap());
            assert!(crate::metal::dispatch_census_take().is_empty());
            enc.end();
            command.commit();
            crate::metal::wait_completed(&command).unwrap();
            return;
        }
        let (k, n) = (2560usize, 512usize);
        let guarded = |v: &[f32], shape: Vec<u64>| {
            offset_tensor(&ctx, 16, bytemuck::cast_slice(v), 16, shape, GgmlType::F32)
        };
        let mut state = 0xf13b4e7a9cu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 20001) as f32 / 10000. - 1.
        };
        let w: Vec<f32> = (0..k * n).map(|_| next() * 0.02).collect();
        let wt = guarded(&w, vec![k as u64, n as u64]);
        for m in [512usize, 1024, 2048] {
            let x: Vec<f32> = (0..k * m).map(|_| next() * 4.).collect();
            let xt = guarded(&x, vec![k as u64, m as u64]);
            let old = guarded(&vec![f32::NAN; n * m], vec![n as u64, m as u64]);
            let new = guarded(&vec![f32::NAN; n * m], vec![n as u64, m as u64]);
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            crate::metal::encode_mat_mat_f32_router_e8p32_strict(
                &ctx, &enc, &wt, &xt, &old, k, n, m,
            )
            .unwrap();
            crate::metal::dispatch_census_begin();
            assert!(try_encode(&ctx, &enc, &wt, &xt, &new, k, n, m).unwrap());
            let census = crate::metal::dispatch_census_take();
            assert_eq!(census.len(), 1);
            assert_eq!(census[0].kernel, "kernel_qwen4exp_small_f32_mma_k128_s4");
            assert_eq!(census[0].grid_width, (m / 32) as u64);
            assert_eq!(census[0].grid_height, 64);
            enc.end();
            command.commit();
            crate::metal::wait_completed(&command).unwrap();
            let a = tensor_f32_at_offset(&old);
            let b = tensor_f32_at_offset(&new);
            assert!(a.iter().chain(&b).all(|v| v.is_finite()));
            let mut errors = [0f64; 2];
            for i in 0..256 {
                let row = i * (m - 1) / 255;
                let col = i * 7919 % n;
                let expected = (0..k)
                    .map(|j| f64::from(w[col * k + j]) * f64::from(x[row * k + j]))
                    .sum::<f64>();
                let actual = f64::from(b[row * n + col]);
                assert!((actual - expected).abs() <= 3e-5 + 3e-5 * expected.abs());
                errors[0] += (f64::from(a[row * n + col]) - expected).powi(2);
                errors[1] += (actual - expected).powi(2);
            }
            assert!(errors[1] <= 2.25 * errors[0] + 1e-12);
            for t in [&wt, &xt, &old, &new] {
                assert_offset_guards(t, 16, 16);
            }
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            crate::metal::dispatch_census_begin();
            for bad_m in [511, 513, 2049] {
                assert!(!try_encode(&ctx, &enc, &wt, &xt, &new, k, n, bad_m).unwrap());
            }
            let alias = xt.view_subrange(0, vec![n as u64, m as u64]);
            assert!(try_encode(&ctx, &enc, &wt, &xt, &alias, k, n, m).is_err());
            let short = new.view_subrange(0, vec![n as u64, (m - 1) as u64]);
            assert!(try_encode(&ctx, &enc, &wt, &xt, &short, k, n, m).is_err());
            assert!(crate::metal::dispatch_census_take().is_empty());
            enc.end();
            command.commit();
            crate::metal::wait_completed(&command).unwrap();
        }
    }

    #[test]
    fn f32_mma_refuses_unqualified_views_and_aliases_before_dispatch() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        if !enabled() || ctx.tensor_ops_status().is_available() {
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
