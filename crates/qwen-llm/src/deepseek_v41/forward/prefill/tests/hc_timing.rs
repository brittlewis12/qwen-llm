//! Opt-in timing of the prefill mHC pre-block kernels at the GSQ geometry
//! (hidden 5120, 4096 rows). Prints medians; asserts nothing.
//! `QWEN_V41_HC_TIMING=1`, under the Metal lease.
use super::*;

fn gpu_us(ctx: &MetalContext, f: &dyn Fn(&KernelEncoder)) -> f64 {
    let mut samples = Vec::new();
    for iteration in 0..12 {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        f(&enc);
        enc.end();
        metal::commit_and_wait(&command).unwrap();
        if iteration >= 2 {
            samples.push((command.GPUEndTime() - command.GPUStartTime()) * 1e6);
        }
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

#[test]
fn hc_rows_kernel_timing() {
    if std::env::var_os("QWEN_V41_HC_TIMING").is_none() {
        return;
    }
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = MetalContext::new().unwrap();
    let mut c = super::super::toy::config();
    c.hidden = 5120;
    let rows = 4096;
    let width = 4 * c.hidden;
    let fill = |n: usize, seed: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * seed) % 997) as f32 - 498.0) / 4096.0)
            .collect()
    };
    let x = upload(
        &ctx,
        &fill(width * rows, 13),
        vec![c.hidden as u64, 4, rows as u64],
    )
    .unwrap();
    let projection = Linear {
        ctx: &ctx,
        tensor: upload(&ctx, &fill(24 * width, 7), vec![width as u64, 24]).unwrap(),
        native_control: None,
        input: width,
        output: 24,
    };
    let constants = [
        upload(&ctx, &[0.7, 1.3, 0.4], vec![3]).unwrap(),
        upload(&ctx, &fill(24, 3), vec![24]).unwrap(),
        upload(&ctx, &fill(c.hidden, 5), vec![c.hidden as u64]).unwrap(),
    ];
    let incoming = upload(&ctx, &[0.25f32; 4].repeat(rows), vec![4, rows as u64]).unwrap();
    let hc = HcRows::new(&ctx, &c, rows).unwrap();
    // Precision of the fused MMA projection at the real K against the scalar
    // F32 projection + projected RMS, and against a CPU F64 reference.
    {
        let mixes = |fused: bool| {
            let out = HcRows::new(&ctx, &c, rows).unwrap();
            HC_ROWS_FUSED_OVERRIDE.with(|o| o.set(Some(fused)));
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            out.encode_with_arithmetic(
                &ctx,
                &enc,
                &c,
                &projection,
                &constants,
                &x,
                &incoming,
                RowsArithmetic::Ordinary,
            )
            .unwrap();
            enc.end();
            metal::commit_and_wait(&command).unwrap();
            HC_ROWS_FUSED_OVERRIDE.with(|o| o.set(None));
            f32_slice(&out.raw).unwrap()[..24 * rows].to_vec()
        };
        let (split, fused) = (mixes(false), mixes(true));
        let xs = f32_slice(&x).unwrap();
        let ws = f32_slice(&projection.tensor).unwrap();
        let (mut split_err, mut fused_err, mut max) = (0f64, 0f64, 0f64);
        for r in (0..rows).step_by(64) {
            let row = &xs[r * width..(r + 1) * width];
            let ss: f64 = row.iter().map(|&v| v as f64 * v as f64).sum();
            let scale = 1.0 / (ss / width as f64 + c.norm_eps as f64).sqrt();
            for o in 0..24 {
                let dot: f64 = row
                    .iter()
                    .zip(&ws[o * width..(o + 1) * width])
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                let reference = dot * scale;
                max = max.max(reference.abs());
                split_err = split_err.max((split[r * 24 + o] as f64 - reference).abs());
                fused_err = fused_err.max((fused[r * 24 + o] as f64 - reference).abs());
            }
        }
        eprintln!(
            "hc mix vs f64 reference (every 64th row): max |ref| {max:.4}, split max err {split_err:.3e}, fused max err {fused_err:.3e}"
        );
    }
    for arithmetic in [RowsArithmetic::Ordinary, RowsArithmetic::Packed] {
        let projection_us = gpu_us(&ctx, &|enc| {
            projection
                .encode_rows_with_arithmetic(enc, &x, &hc.raw, rows, arithmetic)
                .unwrap();
        });
        let rms_us = gpu_us(&ctx, &|enc| {
            enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_projected_rms").unwrap());
            enc.set_bytes(0, &(width as u32));
            enc.set_bytes(1, &c.norm_eps);
            enc.set_tensor(2, &x);
            enc.set_tensor(3, &hc.raw);
            enc.dispatch(
                objc2_metal::MTLSize {
                    width: rows,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        });
        let controls_us = gpu_us(&ctx, &|enc| {
            metal::encode_mhc4_controls_rows(
                &ctx,
                enc,
                rows,
                c.hc_eps,
                &hc.raw,
                &constants[0],
                &constants[1],
                &hc.pre,
                &hc.post,
                &hc.comb,
            )
            .unwrap();
        });
        let normalize_us = gpu_us(&ctx, &|enc| {
            hc.normalize_input(&ctx, enc, &c, &constants[2], &x, &incoming)
                .unwrap();
        });
        let full = |fused: bool| {
            HC_ROWS_FUSED_OVERRIDE.with(|o| o.set(Some(fused)));
            let us = gpu_us(&ctx, &|enc| {
                hc.encode_with_arithmetic(
                    &ctx,
                    enc,
                    &c,
                    &projection,
                    &constants,
                    &x,
                    &incoming,
                    arithmetic,
                )
                .unwrap();
            });
            HC_ROWS_FUSED_OVERRIDE.with(|o| o.set(None));
            us
        };
        let (full_us, fused_us) = (full(false), full(true));
        eprintln!(
            "hc rows={rows} hidden={} {arithmetic:?}: projection {projection_us:.1} us, projected_rms {rms_us:.1}, controls {controls_us:.1}, collapse+norm {normalize_us:.1}, full split {full_us:.1}, full fused {fused_us:.1}",
            c.hidden
        );
    }
}

#[test]
fn fused_hc_projection_rows_matches_split_passes() {
    let ctx = MetalContext::new().unwrap();
    let mut c = super::super::toy::config();
    c.hidden = 512;
    let width = 4 * c.hidden;
    for rows in [1usize, 31, 32, 100] {
        let fill = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * seed) % 997) as f32 - 498.0) / 512.0)
                .collect()
        };
        let x = upload(
            &ctx,
            &fill(width * rows, 13),
            vec![c.hidden as u64, 4, rows as u64],
        )
        .unwrap();
        let projection = Linear {
            ctx: &ctx,
            tensor: upload(&ctx, &fill(24 * width, 7), vec![width as u64, 24]).unwrap(),
            native_control: None,
            input: width,
            output: 24,
        };
        let constants = [
            upload(&ctx, &[0.7, 1.3, 0.4], vec![3]).unwrap(),
            upload(&ctx, &fill(24, 3), vec![24]).unwrap(),
            upload(&ctx, &fill(c.hidden, 5), vec![c.hidden as u64]).unwrap(),
        ];
        let incoming = upload(&ctx, &[0.25f32; 4].repeat(rows), vec![4, rows as u64]).unwrap();
        let run = |fused: bool| {
            let hc = HcRows::new(&ctx, &c, rows).unwrap();
            HC_ROWS_FUSED_OVERRIDE.with(|o| o.set(Some(fused)));
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            hc.encode_with_arithmetic(
                &ctx,
                &enc,
                &c,
                &projection,
                &constants,
                &x,
                &incoming,
                RowsArithmetic::Ordinary,
            )
            .unwrap();
            enc.end();
            metal::commit_and_wait(&command).unwrap();
            HC_ROWS_FUSED_OVERRIDE.with(|o| o.set(None));
            (
                f32_slice(&hc.raw).unwrap()[..24 * rows].to_vec(),
                f32_slice(&hc.post).unwrap()[..4 * rows].to_vec(),
                f32_slice(&hc.comb).unwrap()[..16 * rows].to_vec(),
                f32_slice(&hc.normalized).unwrap()[..c.hidden * rows].to_vec(),
            )
        };
        let split = run(false);
        let fused = run(true);
        let close = |name: &str, a: &[f32], b: &[f32]| {
            let scale = b.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() <= 2e-5 * scale,
                    "rows={rows} {name}[{i}]: {x} vs {y}"
                );
            }
        };
        close("mix", &fused.0, &split.0);
        close("post", &fused.1, &split.1);
        close("comb", &fused.2, &split.2);
        assert_eq!(
            fused.3, split.3,
            "rows={rows}: the collapse/norm does not read the mixes"
        );
    }
}
