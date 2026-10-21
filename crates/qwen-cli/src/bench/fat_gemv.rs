//! Production-shape Q8_0 fat-GEMV probe.
//!
//! Compares the V4.1 native single-row-per-simdgroup path, the tuned two-row
//! matvec, and the research 4-row/simdgroup fat kernel. Both a single-dispatch
//! command and a chained command are reported: the former includes launch and
//! drain, the latter is the steady-state kernel cost.

use super::*;

pub(crate) fn run(args: GemvFatArgs) -> Result<()> {
    let GemvFatArgs {
        dtype,
        n_in,
        n_out,
        rows_per_simdgroup,
        simdgroups,
        runs,
        output,
    } = args;
    if !dtype.eq_ignore_ascii_case("q8_0") {
        return Err(anyhow!(
            "gemv-fat currently supports --dtype q8_0, got {dtype:?}"
        ));
    }
    if n_in == 0 || !n_in.is_multiple_of(32) || n_out == 0 || runs == 0 {
        return Err(anyhow!(
            "n_in must be a positive multiple of 32, n_out positive, and runs > 0"
        ));
    }
    if !matches!(rows_per_simdgroup, 2 | 4 | 8) || !matches!(simdgroups, 1 | 2 | 4) {
        return Err(anyhow!(
            "unsupported geometry R={rows_per_simdgroup} SG={simdgroups}; R in 2/4/8, SG in 1/2/4"
        ));
    }

    let ctx = MetalContext::new()?;
    let (block, block_bytes) = GgmlType::Q8_0
        .storage_layout()
        .ok_or_else(|| anyhow!("Q8_0 has no storage layout"))?;
    if block != 32 || block_bytes != 34 {
        return Err(anyhow!(
            "unexpected Q8_0 layout block={block} bytes={block_bytes}"
        ));
    }
    let nb = n_in / 32;
    let row_stride = nb
        .checked_mul(34)
        .ok_or_else(|| anyhow!("weight row stride overflow"))?;
    let weight_bytes = row_stride
        .checked_mul(n_out)
        .ok_or_else(|| anyhow!("weight extent overflow"))?;

    // Deterministic valid Q8_0 blocks: half 1.0 (0x3C00 little-endian) and
    // signed 8-bit quants in [-3, 3].
    let mut weights = vec![0u8; weight_bytes];
    for row in 0..n_out {
        for b in 0..nb {
            let off = row * row_stride + b * 34;
            weights[off] = 0x00;
            weights[off + 1] = 0x3C;
            for i in 0..32 {
                weights[off + 2 + i] = (((i as i32) % 7) - 3) as i8 as u8;
            }
        }
    }
    let x_values: Vec<f32> = (0..n_in).map(|i| ((i as f32) * 0.001).sin()).collect();
    let zero_bytes = vec![0u8; n_out * std::mem::size_of::<f32>()];

    let weights_t = MetalTensor::from_bytes(
        &ctx,
        &weights,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q8_0,
    )?;
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x_values),
        vec![n_in as u64],
        GgmlType::F32,
    )?;
    let y_native = MetalTensor::from_bytes(&ctx, &zero_bytes, vec![n_out as u64], GgmlType::F32)?;
    let y_lcpp = MetalTensor::from_bytes(&ctx, &zero_bytes, vec![n_out as u64], GgmlType::F32)?;
    let y_fat = MetalTensor::from_bytes(&ctx, &zero_bytes, vec![n_out as u64], GgmlType::F32)?;
    let control = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[0i32, qwen_llm::metal::ROUTE_STATUS_READY]),
        vec![2],
        GgmlType::I32,
    )?;
    let ids = control.view_subrange(0, vec![1]);
    let status = control.view_subrange(1, vec![1]);
    let bank = weights_t.view_bytes(0, vec![n_in as u64, n_out as u64, 1]);
    let native_input = x_t.view_bytes(0, vec![n_in as u64, 1]);
    let native_output = y_native.view_bytes(0, vec![n_out as u64, 1]);

    let mut native = measure(&ctx, runs, |enc| {
        qwen_llm::metal::encode_all_slots_down_native(
            &ctx,
            enc,
            &bank,
            &native_input,
            &ids,
            &status,
            &native_output,
            n_in,
            n_out,
            1,
            1,
        )
        .map_err(|e| anyhow!("{e}"))
    })?;
    let mut lcpp = measure(&ctx, runs, |enc| {
        qwen_llm::metal::encode_mat_vec_q8_0_f32(&ctx, enc, &weights_t, &x_t, &y_lcpp, n_in, n_out)
            .map_err(|e| anyhow!("{e}"))
    })?;
    let mut fat = measure(&ctx, runs, |enc| {
        qwen_llm::metal::encode_fat_gemv_q8_0_f32(
            &ctx,
            enc,
            &weights_t,
            &x_t,
            &y_fat,
            n_in,
            n_out,
            rows_per_simdgroup,
            simdgroups,
        )
        .map_err(|e| anyhow!("{e}"))
    })?;

    let weight_bytes = weights_t.n_bytes() as f64;
    for arm in [&mut native, &mut lcpp, &mut fat] {
        arm.gb_s = weight_bytes / (arm.gpu_ms * 1e-3) / 1e9;
        arm.chained_gb_s = weight_bytes / (arm.chained_ms * 1e-3) / 1e9;
    }

    let native_values = qwen_llm::metal::read_f32_tensor(&y_native).map_err(|e| anyhow!("{e}"))?;
    let lcpp_values = qwen_llm::metal::read_f32_tensor(&y_lcpp).map_err(|e| anyhow!("{e}"))?;
    let fat_values = qwen_llm::metal::read_f32_tensor(&y_fat).map_err(|e| anyhow!("{e}"))?;
    let compare_native = compare(&native_values, &fat_values);
    let compare_lcpp = compare(&lcpp_values, &fat_values);

    let bytes = weight_bytes;
    let report = serde_json::json!({
        "schema_version": 1,
        "engine": "qwen-llm",
        "test": "gemv_fat",
        "test_time": utc_iso8601_now(),
        "device": ctx.device.name().to_string(),
        "dtype": "Q8_0",
        "n_in": n_in,
        "n_out": n_out,
        "rows_per_simdgroup": rows_per_simdgroup,
        "simdgroups": simdgroups,
        "rows_per_threadgroup": rows_per_simdgroup * simdgroups,
        "runs": runs,
        "weight_bytes": weights_t.n_bytes(),
        "arms": {
            "native": arm_json(&native, bytes, n_in, n_out),
            "lcpp": arm_json(&lcpp, bytes, n_in, n_out),
            "fat": arm_json(&fat, bytes, n_in, n_out),
        },
        "compare": {
            "fat_vs_native": compare_native,
            "fat_vs_lcpp": compare_lcpp,
        },
        "power": capture_power_snapshot(),
    });

    if matches!(output, OutputFormat::Json) {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "[gemv-fat] device={} dtype=Q8_0 n_in={} n_out={} R={} SG={} rows/TG={}",
            ctx.device.name(),
            n_in,
            n_out,
            rows_per_simdgroup,
            simdgroups,
            rows_per_simdgroup * simdgroups
        );
        for (name, arm) in [("native", &native), ("lcpp", &lcpp), ("fat", &fat)] {
            println!(
                "[gemv-fat] {name:<7} gpu={:.4} ms  chained={:.4} ms  {:.1} GB/s  chained {:.1} GB/s  wall={:.4} ms",
                arm.gpu_ms, arm.chained_ms, arm.gb_s, arm.chained_gb_s, arm.wall_ms
            );
        }
        println!(
            "[gemv-fat] fat vs native: max_abs={:.3e} rms={:.3e}; fat vs lcpp: max_abs={:.3e} rms={:.3e}",
            compare_native.max_abs, compare_native.rms, compare_lcpp.max_abs, compare_lcpp.rms
        );
    }
    Ok(())
}

#[derive(Debug)]
struct Arm {
    wall_ms: f64,
    gpu_ms: f64,
    chained_ms: f64,
    gb_s: f64,
    chained_gb_s: f64,
}

fn measure(
    ctx: &MetalContext,
    runs: usize,
    mut encode: impl FnMut(&KernelEncoder) -> Result<()>,
) -> Result<Arm> {
    let (wall_ms, gpu_ms) = time_gpu_reps(ctx, 2, runs, |enc| encode(enc))?;
    let cmd = ctx.queue.commandBuffer().context("gemv-fat chained cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    for _ in 0..runs {
        encode(&enc)?;
    }
    enc.end();
    cmd.commit();
    qwen_llm::metal::wait_completed(&cmd)?;
    let chained_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3 / runs as f64;
    Ok(Arm {
        wall_ms,
        gpu_ms,
        chained_ms,
        gb_s: 0.0,
        chained_gb_s: 0.0,
    })
}

fn arm_json(arm: &Arm, bytes: f64, n_in: usize, n_out: usize) -> serde_json::Value {
    let flops = 2.0 * (n_in as f64) * (n_out as f64);
    serde_json::json!({
        "wall_ms": arm.wall_ms,
        "gpu_ms": arm.gpu_ms,
        "chained_ms": arm.chained_ms,
        "gb_s": arm.gb_s,
        "chained_gb_s": arm.chained_gb_s,
        "tflops": flops / (arm.gpu_ms * 1e-3) / 1e12,
        "chained_tflops": flops / (arm.chained_ms * 1e-3) / 1e12,
        "weight_bytes": bytes as u64,
    })
}

#[derive(Debug, serde::Serialize)]
struct Compare {
    max_abs: f64,
    rms: f64,
}

fn compare(reference: &[f32], actual: &[f32]) -> Compare {
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for (&a, &b) in reference.iter().zip(actual) {
        let d = (a as f64) - (b as f64);
        max_abs = max_abs.max(d.abs());
        sum_sq += d * d;
    }
    Compare {
        max_abs,
        rms: (sum_sq / reference.len().max(1) as f64).sqrt(),
    }
}
