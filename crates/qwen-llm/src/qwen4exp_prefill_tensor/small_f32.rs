use crate::{
    metal::{KernelEncoder, MetalContext, MetalError, MetalTensor},
    metal_forward::MfError,
    tensor::GgmlType,
};
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLSize};

crate::env_flag!(
    /// Device-input F32 HC injection and GDN alpha/beta short projections at aligned 512..2047 rows, default on; APPEND_SHAPES admits additional tensor-tier rows at 128..2048. QWEN_FN_SMALL_F32_TILES also includes M2048. Original F32 operands and sums, no conversion or new buffers. Zero restores the scalar-ordered incumbent. Independent of QWEN_FN_SHORT_APPEND_TENSOR.
    default_on enabled,
    "QWEN_FN_SHORT_SMALL_F32"
);

crate::env_flag!(
    /// Flash-Next F32 HC injection/GDN alpha-beta projections: prefer coalesced HC dots and 32-row tensor tiles, including APPEND_SHAPES tails. No narrowing or new buffers. Zero restores the prior aligned M128 tiles below 2048 and scalar-ordered M2048; SHORT_SMALL_F32=0 disables both. Missing tail PSOs retain the incumbent.
    default_on small_tiles_enabled,
    "QWEN_FN_SMALL_F32_TILES"
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
    let legacy_rows = (512..=2048).contains(&m) && m.is_multiple_of(128);
    let extra_rows = ctx.tensor_ops_status().is_available()
        && small_tiles_enabled()
        && super::shapes::selected(m);
    if !enabled()
        || w.dtype != GgmlType::F32
        || !(legacy_rows || extra_rows)
        || !matches!((k, n), (10240, 4) | (2560, 48))
    {
        return Ok(false);
    }
    if small_tiles_enabled() && super::hc_dot::try_encode(ctx, enc, w, x, y, k, n, m)? {
        return Ok(true);
    }
    // Extra N4 rows were qualified with the coalesced F32 dot. Disabling or
    // losing that PSO restores their incumbent, rather than an unscreened tile.
    if n == 4 && !legacy_rows {
        return Ok(false);
    }
    if small_tiles_enabled() && super::small_f32_mma::try_encode(ctx, enc, w, x, y, k, n, m)? {
        return Ok(true);
    }
    if !ctx.tensor_ops_status().is_available() || (m == 2048 && !small_tiles_enabled()) {
        return Ok(false);
    }
    let name = if n == 4 {
        "kernel_qwen4exp_small_f32_n4"
    } else {
        "kernel_qwen4exp_small_f32_n48"
    };
    let small_name = if n == 4 {
        "kernel_qwen4exp_small_f32_n4_m32"
    } else {
        "kernel_qwen4exp_small_f32_n48_m32"
    };
    let choices = [(small_name, 32), (name, 128)];
    let choices = if small_tiles_enabled() {
        &choices[..if m == 2048 || !legacy_rows { 1 } else { 2 }]
    } else {
        &choices[1..]
    };
    let mut chosen = None;
    for &(name, tile) in choices {
        let tail;
        let name = if m.is_multiple_of(tile) {
            name
        } else {
            tail = format!("{name}_tail");
            &tail
        };
        match ctx.tensor_pipeline(name) {
            Ok(p)
                if p.threadExecutionWidth() == 32
                    && p.maxTotalThreadsPerThreadgroup() >= 128
                    && p.staticThreadgroupMemoryLength()
                        <= ctx.device.maxThreadgroupMemoryLength() =>
            {
                chosen = Some((p, tile));
                break;
            }
            Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => {}
            Err(e) => return Err(e.into()),
        }
    }
    let Some((p, tile)) = chosen else {
        return Ok(false);
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
            width: m.div_ceil(tile),
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
    use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
    #[test]
    fn small_f32_short_projections_match_f64_and_preserve_guards() {
        let _shapes = super::super::shapes::scope(true);
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        if !enabled() {
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
        for (m, k, n) in [
            128usize, 129, 399, 400, 401, 511, 512, 513, 1009, 2000, 2047, 2048,
        ]
        .into_iter()
        .flat_map(|m| [(m, 10240, 4), (m, 2560, 48)])
        {
            let legacy_rows = (512..=2048).contains(&m) && m.is_multiple_of(128);
            if !legacy_rows
                && (!ctx.tensor_ops_status().is_available()
                    || !small_tiles_enabled()
                    || (n == 4 && !crate::env_flag::read_default_on("QWEN_FN_PREFILL_HC_DOT")))
            {
                continue;
            }
            if (!ctx.tensor_ops_status().is_available()
                && (!small_tiles_enabled()
                    || !crate::env_flag::read_default_on("QWEN_FN_PREFILL_F32_MMA")))
                || (m == 2048 && !small_tiles_enabled())
            {
                continue;
            }
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
            let census = crate::metal::dispatch_census_take();
            assert_eq!(census.len(), 1);
            assert_eq!(census[0].threads_width, 128);
            let tile = if census[0]
                .kernel
                .starts_with("kernel_qwen4exp_hc_inject_dot4_f32")
            {
                4
            } else if census[0].kernel.contains("_m32")
                || census[0].kernel == "kernel_qwen4exp_small_f32_mma_k128_s4"
            {
                32
            } else {
                128
            };
            assert_eq!(census[0].grid_width, m.div_ceil(tile) as u64);
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd).unwrap();
            let a = read(&old);
            let b = read(&yt);
            assert!(a.iter().chain(&b).all(|v| v.is_finite()));
            let mut errors = [0.0f64; 2];
            let mut maximum = 0.0f64;
            let mut maximum_pair = [0.0f64; 2];
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
                maximum_pair[0] = maximum_pair[0].max((f64::from(a[row * n + col]) - r).abs());
                maximum_pair[1] = maximum_pair[1].max((v - r).abs());
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
            eprintln!("F32_TAIL_F64 rows={m} k={k} n={n} sse={errors:?} max={maximum_pair:?}");
        }
    }
}
