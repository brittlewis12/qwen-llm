use super::*;

crate::env_flag!(
    /// BF16 device-input HC prompt projections for Flash-Next at M2048 (or M512/M1024 with QWEN_FN_SHORT_APPEND_TENSOR) on the Metal 4 tier, default on. Preserves the incumbent BF16 activation conversion and F32 sums; reuses dead raw-gate/mixed storage. Set to 0 to restore the incumbent. QWEN_MATMAT_BF16_BFLOAT_ACT=0 disables this route too.
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
        || !crate::qwen4exp_prefill_tensor::rows_qualified(m)
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
    if convert.threadExecutionWidth() != 32 || convert.maxTotalThreadsPerThreadgroup() < 256 {
        return Ok(false);
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCommandQueue;

    #[test]
    fn hc_device_bf16_matches_f64_and_preserves_offset_guards() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        if !enabled()
            || !ctx.tensor_ops_status().is_available()
            || !crate::metal_forward::matmat_bf16_bfloat_act_enabled()
        {
            return;
        }
        let guarded = |bytes: &[u8], dtype: GgmlType, shape: Vec<u64>| {
            let element = if dtype == GgmlType::BF16 { 2 } else { 4 };
            let mut storage = vec![0xa5; bytes.len() + 128];
            storage[64..64 + bytes.len()].copy_from_slice(bytes);
            MetalTensor::from_bytes(
                &ctx,
                &storage,
                vec![(storage.len() / element) as u64],
                dtype,
            )
            .unwrap()
            .view_bytes(64, shape)
        };
        let read = |t: &MetalTensor| unsafe {
            std::slice::from_raw_parts(
                t.buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(t.offset as usize)
                    .cast::<f32>(),
                t.n_elements() as usize,
            )
            .to_vec()
        };
        for (m, k, n) in [512usize, 1024, 2048]
            .into_iter()
            .flat_map(|m| [(m, 10240, 320), (m, 320, 10240)])
        {
            if !crate::qwen4exp_prefill_tensor::rows_qualified(m) {
                continue;
            }
            let mut state = 0x12345678abcdefffu64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 20001) as f32 / 10000.0 - 1.0
            };
            let w: Vec<half::bf16> = (0..k * n)
                .map(|_| half::bf16::from_f32(next() * 0.02))
                .collect();
            let x: Vec<f32> = (0..k * m).map(|_| next() * 4.0).collect();
            let wt = guarded(
                bytemuck::cast_slice(&w),
                GgmlType::BF16,
                vec![k as u64, n as u64],
            );
            let xt = guarded(
                bytemuck::cast_slice(&x),
                GgmlType::F32,
                vec![k as u64, m as u64],
            );
            let poison = vec![f32::NAN; n * m];
            let yt = guarded(
                bytemuck::cast_slice(&poison),
                GgmlType::F32,
                vec![n as u64, m as u64],
            );
            let old = guarded(
                bytemuck::cast_slice(&poison),
                GgmlType::F32,
                vec![n as u64, m as u64],
            );
            let dead = guarded(
                bytemuck::cast_slice(&vec![f32::NAN; k * m / 2]),
                GgmlType::F32,
                vec![(k * m / 2) as u64],
            );
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            crate::metal::encode_mat_mat_bf16_bfloat_act_f32(&ctx, &enc, &wt, &xt, &old, k, n, m)
                .unwrap();
            crate::metal::dispatch_census_begin();
            assert!(try_encode(&ctx, &enc, &wt, &xt, &yt, &dead, k, n, m).unwrap());
            let census = crate::metal::dispatch_census_take();
            assert_eq!(
                census.iter().map(|r| r.kernel.as_str()).collect::<Vec<_>>(),
                [
                    "kernel_qwen4exp_hc_convert_bf16",
                    "kernel_qwen4exp_hc_bf16_device"
                ]
            );
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd).unwrap();
            let a = read(&old);
            let b = read(&yt);
            assert!(a.iter().chain(&b).all(|v| v.is_finite()));
            let mut errors = [0.0f64; 2];
            let mut maximum = 0.0f64;
            for i in 0..128 {
                let row = i * (m - 1) / 127;
                let column = i * 7919 % n;
                let mut original = 0.0f64;
                let mut rounded = 0.0f64;
                for j in 0..k {
                    let v = f64::from(w[column * k + j].to_f32());
                    original += v * f64::from(x[row * k + j]);
                    rounded += v * f64::from(half::bf16::from_f32(x[row * k + j]).to_f32());
                }
                let actual = f64::from(b[row * n + column]);
                maximum = maximum.max((actual - rounded).abs());
                assert!(
                    (actual - rounded).abs() <= 3e-5 + 3e-5 * rounded.abs(),
                    "HC K{k} N{n}: {actual} vs {rounded}"
                );
                errors[0] += (f64::from(a[row * n + column]) - original).powi(2);
                errors[1] += (actual - original).powi(2);
            }
            assert!(errors[1] <= 2.25 * errors[0] + 1e-12);
            for t in [&wt, &xt, &yt, &old, &dead] {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        t.buffer.contents().as_ptr().cast::<u8>(),
                        t.buffer.length(),
                    )
                };
                assert!(
                    bytes[..64]
                        .iter()
                        .chain(&bytes[64 + t.n_bytes() as usize..64 + t.n_bytes() as usize + 64])
                        .all(|b| *b == 0xa5)
                );
            }
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            crate::metal::dispatch_census_begin();
            for bad in [&xt, &yt] {
                assert!(try_encode(&ctx, &enc, &wt, &xt, &yt, bad, k, n, m).is_err());
            }
            assert!(crate::metal::dispatch_census_take().is_empty());
            enc.end();
            eprintln!(
                "HC device M{m} K{k} N{n}: BF16-operand F64 max_abs={maximum:.9e}, original-input RMS error ratio={:.9}",
                (errors[1] / errors[0]).sqrt()
            );
        }
    }
}
