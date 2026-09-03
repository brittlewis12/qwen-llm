//! Roofline probe.

use super::*;

pub(crate) fn run_roofline(args: RooflineArgs) -> Result<()> {
    let RooflineArgs {
        stream_mib,
        fma_elements,
        fma_iters,
        mat_in,
        mat_out,
        mat_query,
        runs,
        output,
    } = args;
    if stream_mib == 0
        || fma_elements == 0
        || fma_iters == 0
        || mat_in == 0
        || mat_out == 0
        || mat_query == 0
        || runs == 0
    {
        return Err(anyhow!(
            "stream_mib, fma_elements, fma_iters, mat_in, mat_out, mat_query, and runs must all be > 0"
        ));
    }
    if mat_in % 256 != 0 || mat_out % 64 != 0 || mat_query % 64 != 0 {
        return Err(anyhow!(
            "Q4_K mat-mat calibration requires mat_in % 256 == 0, mat_out % 64 == 0, and mat_query % 64 == 0"
        ));
    }

    let ctx = MetalContext::new()?;
    let stream_bytes = stream_mib
        .checked_mul(1024 * 1024)
        .context("stream bytes overflow")?;
    let stream_elems = (stream_bytes / std::mem::size_of::<f32>()).max(1);
    let stream_x = MetalTensor::zeros_f32(&ctx, vec![stream_elems as u64])?;
    let stream_y = MetalTensor::zeros_f32(&ctx, vec![stream_elems as u64])?;
    let fma_x = MetalTensor::zeros_f32(&ctx, vec![fma_elements as u64])?;
    let fma_y = MetalTensor::zeros_f32(&ctx, vec![fma_elements as u64])?;
    let mat_x_elems = mat_query.checked_mul(mat_in).context("mat_x overflow")?;
    let mat_y_elems = mat_query.checked_mul(mat_out).context("mat_y overflow")?;
    let mat_w =
        MetalTensor::zeros_dtype(&ctx, vec![mat_in as u64, mat_out as u64], GgmlType::Q4_K)?;
    let mat_x = MetalTensor::zeros_f32(&ctx, vec![mat_x_elems as u64])?;
    let mat_y = MetalTensor::zeros_f32(&ctx, vec![mat_y_elems as u64])?;

    let init_cmd = ctx.queue.commandBuffer().context("roofline init cmd")?;
    let init_enc = KernelEncoder::begin(&init_cmd);
    encode_fill_f32(&ctx, &init_enc, &stream_x, 1.0)?;
    encode_fill_f32(&ctx, &init_enc, &stream_y, 2.0)?;
    encode_fill_f32(&ctx, &init_enc, &fma_x, 1.0)?;
    encode_fill_f32(&ctx, &init_enc, &fma_y, 0.0)?;
    encode_fill_f32(&ctx, &init_enc, &mat_x, 1.0)?;
    encode_fill_f32(&ctx, &init_enc, &mat_y, 0.0)?;
    init_enc.end();
    init_cmd.commit();
    init_cmd.waitUntilCompleted();

    fn timed_kernel(
        ctx: &MetalContext,
        runs: usize,
        mut encode: impl FnMut(&KernelEncoder) -> Result<()>,
    ) -> Result<Vec<f64>> {
        let mut samples = Vec::with_capacity(runs);
        for rep in 0..=runs {
            let cmd = ctx.queue.commandBuffer().context("roofline cmd")?;
            let enc = KernelEncoder::begin(&cmd);
            encode(&enc)?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            if rep > 0 {
                samples.push((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
            }
        }
        Ok(samples)
    }

    let stream_ms = timed_kernel(&ctx, runs, |enc| {
        encode_roofline_stream_f32(&ctx, enc, &stream_x, &stream_y, 1.0000001)?;
        Ok(())
    })?;
    let fma_ms = timed_kernel(&ctx, runs, |enc| {
        encode_roofline_fma_f32(&ctx, enc, &fma_x, &fma_y, fma_iters)?;
        Ok(())
    })?;
    let mat_ms = timed_kernel(&ctx, runs, |enc| {
        encode_mat_mat_dispatch(
            &ctx, enc, &mat_w, &mat_x, &mat_y, mat_in, mat_out, mat_query,
        )?;
        Ok(())
    })?;

    let stream_avg_ms = sample_mean(&stream_ms);
    let fma_avg_ms = sample_mean(&fma_ms);
    let mat_avg_ms = sample_mean(&mat_ms);
    let stream_nominal_bytes = (stream_elems as f64) * 3.0 * std::mem::size_of::<f32>() as f64;
    let fma_flops = (fma_elements as f64) * (fma_iters as f64) * 2.0;
    let mat_flops = (mat_in as f64) * (mat_out as f64) * (mat_query as f64) * 2.0;
    let stream_gb_s = stream_nominal_bytes / (stream_avg_ms * 1e-3) / 1e9;
    let fma_tflops = fma_flops / (fma_avg_ms * 1e-3) / 1e12;
    let mat_tflops = mat_flops / (mat_avg_ms * 1e-3) / 1e12;
    let power = capture_power_snapshot();

    if matches!(output, OutputFormat::Json) {
        let row = serde_json::json!({
            "schema_version": 1,
            "engine": "qwen-llm",
            "test": "roofline",
            "test_time": utc_iso8601_now(),
            "device": ctx.device.name().to_string(),
            "stream": {
                "elements": stream_elems,
                "nominal_bytes_per_rep": stream_nominal_bytes as u64,
                "avg_ms": stream_avg_ms,
                "stddev_ms": sample_stdev(&stream_ms),
                "samples_ms": stream_ms,
                "gb_s": stream_gb_s,
            },
            "fma": {
                "elements": fma_elements,
                "iters": fma_iters,
                "nominal_flops_per_rep": fma_flops as u64,
                "avg_ms": fma_avg_ms,
                "stddev_ms": sample_stdev(&fma_ms),
                "samples_ms": fma_ms,
                "tflops": fma_tflops,
            },
            "matmat_q4_k": {
                "n_in": mat_in,
                "n_out": mat_out,
                "n_query": mat_query,
                "dtype": "Q4_K",
                "nominal_flops_per_rep": mat_flops as u64,
                "avg_ms": mat_avg_ms,
                "stddev_ms": sample_stdev(&mat_ms),
                "samples_ms": mat_ms,
                "nominal_tflops": mat_tflops,
            },
            "power": power,
        });
        println!("{}", serde_json::to_string(&row)?);
    } else {
        println!("[roofline] device: {}", ctx.device.name());
        println!(
            "[roofline] power: {}",
            power_snapshot_summary(power.as_ref())
        );
        println!(
            "[roofline] stream elements={} nominal_bytes={} avg_ms={:.3} std_ms={:.3} GB/s={:.1}",
            stream_elems,
            stream_nominal_bytes as u64,
            stream_avg_ms,
            sample_stdev(&stream_ms),
            stream_gb_s
        );
        println!(
            "[roofline] fma elements={} iters={} nominal_flops={} avg_ms={:.3} std_ms={:.3} TFLOP/s={:.2}",
            fma_elements,
            fma_iters,
            fma_flops as u64,
            fma_avg_ms,
            sample_stdev(&fma_ms),
            fma_tflops
        );
        println!(
            "[roofline] matmat_q4_k n_in={} n_out={} n_query={} nominal_flops={} avg_ms={:.3} std_ms={:.3} nominal_TFLOP/s={:.2}",
            mat_in,
            mat_out,
            mat_query,
            mat_flops as u64,
            mat_avg_ms,
            sample_stdev(&mat_ms),
            mat_tflops
        );
    }
    Ok(())
}
