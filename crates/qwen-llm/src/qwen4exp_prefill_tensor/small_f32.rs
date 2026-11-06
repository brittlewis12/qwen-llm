use crate::{
    metal::{KernelEncoder, MetalContext, MetalError, MetalTensor},
    metal_forward::MfError,
    tensor::GgmlType,
};
use objc2_metal::MTLSize;

crate::env_flag!(
    /// Device-input F32 tensor HC injection and GDN alpha/beta projections at exactly 512/1024 rows, default on with the Metal 4 tier. Original F32 operands and sums, no relaxed precision, conversion or new buffers. Set to 0 for the scalar-ordered incumbent. Independent of QWEN_FN_SHORT_APPEND_TENSOR.
    default_on enabled,
    "QWEN_FN_SHORT_SMALL_F32"
);

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
        || !ctx.tensor_ops_status().is_available()
        || w.dtype != GgmlType::F32
        || !matches!(m, 512 | 1024)
        || !matches!((k, n), (10240, 4) | (2560, 48))
    {
        return Ok(false);
    }
    let name = if n == 4 {
        "kernel_qwen4exp_small_f32_n4"
    } else {
        "kernel_qwen4exp_small_f32_n48"
    };
    let p = match ctx.tensor_pipeline(name) {
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
    require_serial("fn_small_f32", enc)?;
    check_tensor(
        "fn_small_f32",
        w,
        GgmlType::F32,
        &[k as u64, n as u64],
        false,
        "weight",
    )?;
    check_tensor(
        "fn_small_f32",
        x,
        GgmlType::F32,
        &[k as u64, m as u64],
        false,
        "input",
    )?;
    check_tensor(
        "fn_small_f32",
        y,
        GgmlType::F32,
        &[n as u64, m as u64],
        true,
        "output",
    )?;
    check_disjoint("fn_small_f32", y, &[(w, "weight"), (x, "input")])?;
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
    enc.set_threadgroup_memory(0, 0);
    enc.dispatch(
        MTLSize {
            width: m / 128,
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
    use objc2_metal::MTLCommandQueue;
    #[test]
    fn small_f32_short_projections_match_f64_and_preserve_guards() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        if !enabled() || !ctx.tensor_ops_status().is_available() {
            return;
        }
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
        let read = |t: &MetalTensor| unsafe {
            std::slice::from_raw_parts(
                t.buffer.contents().as_ptr().cast::<f32>().add(16),
                t.n_elements() as usize,
            )
            .to_vec()
        };
        for (m, k, n) in [512usize, 1024]
            .into_iter()
            .flat_map(|m| [(m, 10240, 4), (m, 2560, 48)])
        {
            let mut state = 0x12345678abcdefffu64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 20001) as f32 / 10000.0 - 1.0
            };
            let w: Vec<f32> = (0..k * n).map(|_| next() * 0.02).collect();
            let x: Vec<f32> = (0..k * m).map(|_| next() * 4.0).collect();
            let wt = guarded(&w, vec![k as u64, n as u64]);
            let xt = guarded(&x, vec![k as u64, m as u64]);
            let yt = guarded(&vec![f32::NAN; n * m], vec![n as u64, m as u64]);
            let old = guarded(&vec![f32::NAN; n * m], vec![n as u64, m as u64]);
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            crate::metal::encode_mat_mat_f32(&ctx, &enc, &wt, &xt, &old, k, n, m).unwrap();
            crate::metal::dispatch_census_begin();
            assert!(try_encode(&ctx, &enc, &wt, &xt, &yt, k, n, m).unwrap());
            assert_eq!(crate::metal::dispatch_census_take().len(), 1);
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd).unwrap();
            let a = read(&old);
            let b = read(&yt);
            assert!(a.iter().chain(&b).all(|v| v.is_finite()));
            let mut errors = [0.0f64; 2];
            let mut maximum = 0.0f64;
            for i in 0..256 {
                let row = i * (m - 1) / 255;
                let col = i * 7919 % n;
                let r = (0..k)
                    .map(|j| f64::from(w[col * k + j]) * f64::from(x[row * k + j]))
                    .sum::<f64>();
                let v = f64::from(b[row * n + col]);
                maximum = maximum.max((v - r).abs());
                assert!(
                    (v - r).abs() <= 3e-5 + 3e-5 * r.abs(),
                    "M{m} K{k} N{n}: {v} vs {r}"
                );
                errors[0] += (f64::from(a[row * n + col]) - r).powi(2);
                errors[1] += (v - r).powi(2);
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
                "small F32 M{m} K{k} N{n}: F64 max_abs={maximum:.9e}, RMS error ratio={:.9}",
                (errors[1] / errors[0]).sqrt()
            );
        }
    }
}
