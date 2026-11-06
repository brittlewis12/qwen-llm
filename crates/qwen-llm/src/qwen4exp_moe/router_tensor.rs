use super::*;
use crate::metal::Pipeline;
use objc2_metal::MTLSize;

const KERNEL: &str = "kernel_qwen4exp_router_f32_tensor";

crate::env_flag!(
    /// F32-operand Metal 4 tensor router for the Flash-Next H2560/E512/top-10 packed motor at exactly 2048 rows, default on. No half operands or added GPU buffers. Set to 0 to retain the strict E8P32 router; also respects QWEN4EXP_PACKED_ROUTER_E8P32_STRICT=0.
    default_on configured_enabled,
    "QWEN_FN_PREFILL_ROUTER_TENSOR"
);

#[cfg(test)]
thread_local! {static ENABLED_OVERRIDE: std::cell::Cell<Option<bool>> = const {std::cell::Cell::new(None)};}

fn enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = ENABLED_OVERRIDE.with(|s| s.get()) {
        return value;
    }
    configured_enabled()
}

#[cfg(test)]
pub(super) fn with_enabled_override<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            ENABLED_OVERRIDE.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(ENABLED_OVERRIDE.with(|s| s.replace(Some(enabled))));
    f()
}

pub(super) fn pipeline(
    ctx: &MetalContext,
    g: Qwen4ExpMoeMetalGeometry,
    dtype: GgmlType,
    tokens: usize,
) -> Result<Option<Pipeline>, MetalError> {
    if !enabled()
        || !ctx.tensor_ops_status().is_available()
        || dtype != GgmlType::F32
        || (g.hidden_size, g.expert_count, g.experts_per_token, tokens) != (2560, 512, 10, 2048)
    {
        return Ok(None);
    }
    match ctx.tensor_pipeline(KERNEL) {
        Ok(p)
            if p.threadExecutionWidth() == 32
                && p.maxTotalThreadsPerThreadgroup() >= 128
                && p.staticThreadgroupMemoryLength() <= ctx.device.maxThreadgroupMemoryLength() =>
        {
            Ok(Some(p))
        }
        Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => Ok(None),
        Err(e) => Err(e),
    }
}

pub(super) fn encode(
    enc: &KernelEncoder,
    p: &Pipeline,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
) -> Result<(), MetalError> {
    use crate::metal::checks::{check_disjoint, check_tensor, require_serial};
    require_serial(KERNEL, enc)?;
    check_tensor(KERNEL, weight, GgmlType::F32, &[2560, 512], false, "weight")?;
    check_tensor(KERNEL, input, GgmlType::F32, &[2560, 2048], false, "input")?;
    check_tensor(KERNEL, output, GgmlType::F32, &[512, 2048], true, "output")?;
    check_disjoint(KERNEL, output, &[(weight, "weight"), (input, "input")])?;
    enc.set_pipeline(p);
    enc.set_tensor(0, weight);
    enc.set_tensor(1, input);
    enc.set_tensor(2, output);
    for index in 0..3 {
        enc.set_threadgroup_memory(index, 0);
    }
    enc.dispatch(
        MTLSize {
            width: 32,
            height: 8,
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
mod tests {
    use super::*;
    use objc2_metal::MTLCommandQueue;

    #[test]
    fn f32_router_tensor_matches_f64_with_guards() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        if !ctx.tensor_ops_status().is_available() {
            return;
        }
        let p = ctx.tensor_pipeline(KERNEL).unwrap();
        let mut state = 0x12345678abcdef01u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 20001) as f32 / 10000.0 - 1.0
        };
        let w: Vec<f32> = (0..2560 * 512).map(|_| next() * 0.05).collect();
        let x: Vec<f32> = (0..2560 * 2048).map(|_| next() * 4.0).collect();
        let guarded = |values: &[f32], shape: Vec<u64>| {
            let mut data = vec![f32::from_bits(0x45abcdef); values.len() + 32];
            data[16..16 + values.len()].copy_from_slice(values);
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&data),
                vec![data.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
            .view_subrange(16, shape)
        };
        let wt = guarded(&w, vec![2560, 512]);
        let xt = guarded(&x, vec![2560, 2048]);
        let yt = guarded(&vec![f32::NAN; 512 * 2048], vec![512, 2048]);
        let old = guarded(&vec![f32::NAN; 512 * 2048], vec![512, 2048]);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        encode_mat_mat_f32_router_e8p32_strict(&ctx, &enc, &wt, &xt, &old, 2560, 512, 2048)
            .unwrap();
        crate::metal::dispatch_census_begin();
        encode(&enc, &p, &wt, &xt, &yt).unwrap();
        let census = crate::metal::dispatch_census_take();
        assert_eq!(census.len(), 1);
        assert_eq!(census[0].kernel, KERNEL);
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).unwrap();
        let read = |t: &MetalTensor| unsafe {
            std::slice::from_raw_parts(
                t.buffer.contents().as_ptr().cast::<f32>().add(16),
                t.n_elements() as usize,
            )
            .to_vec()
        };
        let a = read(&old);
        let b = read(&yt);
        assert!(a.iter().chain(&b).all(|v| v.is_finite()));
        let mut errors = [0.0f64; 2];
        let mut maximum = 0.0f64;
        for cell in 0..256 {
            let row = cell * 2047 / 255;
            let column = cell * 173 % 512;
            let expected = (0..2560)
                .map(|k| f64::from(w[column * 2560 + k]) * f64::from(x[row * 2560 + k]))
                .sum::<f64>();
            let actual = f64::from(b[row * 512 + column]);
            maximum = maximum.max((actual - expected).abs());
            assert!(
                (actual - expected).abs() <= 2e-5 + 2e-5 * expected.abs(),
                "row{row} col{column}: {actual} vs {expected}"
            );
            errors[0] += (f64::from(a[row * 512 + column]) - expected).powi(2);
            errors[1] += (actual - expected).powi(2);
        }
        assert!(errors[1] <= 2.25 * errors[0] + 1e-12);
        for t in [&wt, &xt, &yt, &old] {
            let data = unsafe {
                std::slice::from_raw_parts(
                    t.buffer.contents().as_ptr().cast::<u32>(),
                    t.n_elements() as usize + 32,
                )
            };
            assert!(
                data[..16]
                    .iter()
                    .chain(&data[data.len() - 16..])
                    .all(|v| *v == 0x45abcdef)
            );
        }
        eprintln!(
            "F32 router tensor: sampled max_abs={maximum:.9e} RMS error ratio={:.9}",
            (errors[1] / errors[0]).sqrt()
        );
    }
}
