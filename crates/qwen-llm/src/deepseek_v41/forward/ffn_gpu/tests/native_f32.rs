//! Synthetic routed-stage replay at V4.1 widths with reduced expert banks;
//! separate 384/top-6 geometry. This is not real-artifact qualification.
use super::*;

pub(super) fn bank(
    ctx: &MetalContext,
    dtype: GgmlType,
    k: usize,
    m: usize,
    e: usize,
    seed: usize,
) -> MetalTensor {
    let (block, size) = dtype.storage_layout().unwrap();
    let mut bytes = Vec::new();
    for ordinal in 0..k * m * e / block as usize {
        let mut b: Vec<_> = (0..size as usize)
            .map(|i| (ordinal * 73 + i * 29 + seed * 11 + ordinal / 7) as u8)
            .collect();
        if dtype == GgmlType::MXFP4 {
            b[0] = 116 + ((ordinal + seed) % 5) as u8;
        } else {
            let offset = if dtype == GgmlType::Q3_K { 108 } else { 0 };
            let sign = if (ordinal + seed) % 2 == 0 { 1.0 } else { -1.0 };
            let scale = sign * (1 + (ordinal + seed) % 7) as f32 / 65536.0;
            b[offset..offset + 2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        }
        bytes.extend(b);
    }
    MetalTensor::from_bytes(ctx, &bytes, vec![k as u64, m as u64, e as u64], dtype).unwrap()
}

fn write(t: &MetalTensor, values: &[f32]) {
    assert_eq!(t.n_elements() as usize, values.len());
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<f32>(),
            values.len(),
        );
    }
}

fn ids(t: &MetalTensor) -> Vec<i32> {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<i32>(),
            t.n_elements() as usize,
        )
        .to_vec()
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|v| v.to_bits()).collect()
}

fn assert_down_f64(actual: &[f32], exact: &[f64]) {
    assert!(actual.iter().all(|v| v.is_finite()));
    let error = actual
        .iter()
        .zip(exact)
        .map(|(&a, &b)| (f64::from(a) - b).powi(2))
        .sum::<f64>();
    let norm = exact.iter().map(|v| v * v).sum::<f64>();
    let peak = exact.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let max = crate::compare::assert_max_abs_diff_f64(
        "V41 frozen down",
        actual.iter().map(|&v| f64::from(v)),
        exact.iter(),
    );
    assert!((error / norm).sqrt() <= 1e-5 && max <= peak * 2e-5);
}

fn compare_native_rows(
    ctx: &MetalContext,
    ws: &Workspace,
    c: &Config,
    layer: &Layer<'_>,
    grouped: &Rows,
) {
    let n = grouped.hc.rows;
    let mut native = Rows::new(ctx, c, n, 1).unwrap();
    native.hc.normalized = grouped.hc.normalized.clone();
    for index in [0, 4] {
        native.buffers[index] = grouped.buffers[index].clone();
    }
    metal::dispatch_census_begin();
    with_native_rows(|| {
        submit(ctx, "v41_compare_native_rows", [0; 4], |enc| {
            ws.encode_routed_rows(ctx, enc, c, layer, 0, &native, None)
        })
        .unwrap();
    });
    let census = metal::dispatch_census_take();
    assert_eq!(census.len(), 6);
    for index in [1, 2, 4] {
        assert!(census[index].kernel.starts_with("kernel_expert_native_"));
        assert_eq!(census[index].grid_depth, n as u64);
    }
    native.validate_status().unwrap();
    assert_eq!(ids(&native.buffers[8]), ids(&grouped.buffers[8]));
    assert_eq!(
        bits(f32_slice(&native.buffers[9]).unwrap()),
        bits(f32_slice(&grouped.buffers[9]).unwrap())
    );
    for (index, stage) in [
        (12, "activation"),
        (13, "weighted"),
        (14, "down"),
        (5, "sum"),
    ] {
        let a = f32_slice(&grouped.buffers[index]).unwrap();
        let b = f32_slice(&native.buffers[index]).unwrap();
        close(a, b);
        let max = crate::compare::assert_max_abs_diff_f64(
            stage,
            a.iter().map(|&v| f64::from(v)),
            b.iter().map(|&v| f64::from(v)),
        );
        let error: f64 = a
            .iter()
            .zip(b)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum();
        let norm: f64 = b.iter().map(|&v| f64::from(v).powi(2)).sum();
        let changed = a
            .iter()
            .zip(b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "grouped/native {:?}/{:?} H={} F={} E={} top={} N={n} {stage}: changed={changed}/{} max={max:e} relative_l2={:e}",
            layer.banks[0].dtype,
            layer.banks[2].dtype,
            c.hidden,
            c.intermediate,
            c.experts,
            c.topk,
            a.len(),
            (error / norm).sqrt()
        );
    }
    // Same production composition, alternating order, one warmup and eight samples.
    // Synthetic GPU command timing is not an end-to-end model speed claim.
    let mut times = [Vec::new(), Vec::new()];
    for round in 0..9 {
        for which in [round % 2, 1 - round % 2] {
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            let encode = || {
                for _ in 0..4 {
                    ws.encode_routed_rows(
                        ctx,
                        &enc,
                        c,
                        layer,
                        0,
                        if which == 0 { &native } else { grouped },
                        None,
                    )
                    .unwrap();
                }
            };
            if which == 0 {
                with_native_rows(|| {
                    assert!(!grouped_selected(c, layer, n));
                    encode();
                });
            } else {
                assert!(grouped_selected(c, layer, n));
                encode();
            }
            enc.end();
            metal::commit_and_wait(&command).unwrap();
            let elapsed = command.GPUEndTime() - command.GPUStartTime();
            assert!(elapsed.is_finite() && elapsed > 0.0);
            if round > 0 {
                times[which].push(elapsed / 4.0);
            }
        }
    }
    let medians = times.map(|mut t| {
        t.sort_by(f64::total_cmp);
        (t[3] + t[4]) * 0.5
    });
    eprintln!(
        "grouped/native {:?}/{:?} H={} F={} E={} top={} N={n}: GPU median native={:.3}us grouped={:.3}us speedup={:.3}x",
        layer.banks[0].dtype,
        layer.banks[2].dtype,
        c.hidden,
        c.intermediate,
        c.experts,
        c.topk,
        medians[0] * 1e6,
        medians[1] * 1e6,
        medians[0] / medians[1]
    );
}

fn witness(
    ctx: &MetalContext,
    gate_type: GgmlType,
    down_type: GgmlType,
    h: usize,
    f: usize,
    e: usize,
    top: usize,
    n: usize,
) {
    let banks = [
        bank(ctx, gate_type, h, f, e, 3),
        bank(ctx, gate_type, h, f, e, 7),
        bank(ctx, down_type, f, h, e, 11),
    ];
    let toy_weights = toy::Weights::new(ctx);
    let mut model = toy_weights.model(ctx);
    model.layers.truncate(1);
    model.config.hidden = h;
    model.config.intermediate = f;
    model.config.experts = e;
    model.config.topk = top;
    model.config.clamp = 0.35;
    model.layers[0].bias = vec![0.0; e];
    model.layers[0].ffn_hc.norm = vec![1.0; h];
    model.layers[0].banks = [&banks[0], &banks[1], &banks[2]];
    let c = &model.config;
    let layer = &model.layers[0];
    assert!(grouped_supported(c, layer));
    let ws = Workspace::new(&model).unwrap();
    let x: Vec<_> = (0..h * n)
        .map(|i| ((i * 17 + i / h * 13) % 89) as f32 * 0.0317 - 1.4)
        .collect();
    let logits: Vec<_> = (0..e * n)
        .map(|i| {
            if i % e >= e - top {
                1.0 + (i % e) as f32 * 0.001
            } else {
                -5.0
            }
        })
        .collect();
    let shared: Vec<_> = (0..h * n)
        .map(|i| (i % 17) as f32 * 0.000137 - 0.001)
        .collect();
    let run_grouped = |start: usize, rows: usize| {
        assert!(grouped_selected(c, layer, rows));
        let scratch = Rows::new(ctx, c, rows, 1).unwrap();
        write(&scratch.hc.normalized, &x[start * h..(start + rows) * h]);
        write(&scratch.buffers[0], &logits[start * e..(start + rows) * e]);
        write(&scratch.buffers[4], &shared[start * h..(start + rows) * h]);
        metal::dispatch_census_begin();
        submit(ctx, "v41_frozen_grouped", [0; 4], |enc| {
            ws.encode_routed_rows(ctx, enc, c, layer, 0, &scratch, None)
        })
        .unwrap();
        let census = metal::dispatch_census_take();
        assert_eq!(census.len(), 6);
        let suffix = if gate_type == GgmlType::Q3_K {
            "q3_K"
        } else {
            "iq3_xxs"
        };
        assert_eq!(
            census[2].kernel,
            format!("kernel_moe_swiglu_clamped_{suffix}_f32x_grouped_slots")
        );
        let suffix = match down_type {
            GgmlType::IQ3_XXS => "iq3_xxs",
            GgmlType::IQ3_S => "iq3_s",
            GgmlType::Q3_K => "q3_K",
            GgmlType::MXFP4 => "mxfp4",
            _ => unreachable!(),
        };
        assert_eq!(
            census[4].kernel,
            format!("kernel_moe_down_{suffix}_f32x_grouped_slots")
        );
        scratch.validate_status().unwrap();
        scratch
    };
    let scratch = run_grouped(0, n);
    let b = &scratch.buffers;
    let frozen_ids = ids(&b[8]);
    let frozen_weights = f32_slice(&b[9]).unwrap();
    let activation = f32_slice(&b[12]).unwrap();
    let weighted = f32_slice(&b[13]).unwrap();
    let down = f32_slice(&b[14]).unwrap();
    let output = f32_slice(&b[5]).unwrap();
    let wrong_down = MetalTensor::zeros_f32(ctx, vec![h as u64, (top * n) as u64]).unwrap();
    submit(ctx, "v41_wrong_weight_after", [0; 4], |enc| {
        metal::encode_moe_down_f32x_grouped_slots(
            ctx,
            enc,
            &banks[2],
            &b[12],
            &b[10].view_subrange(0, vec![e as u64]),
            &b[11],
            &wrong_down,
            f,
            h,
            e,
            n,
        )
    })
    .unwrap();
    let wrong_down = f32_slice(&wrong_down).unwrap();
    let mut differs_from_weight_after = false;
    for row in 0..n {
        let expected_route = route(
            &logits[row * e..(row + 1) * e],
            &layer.bias,
            top,
            c.gate_temperature,
            c.route_scale,
        )
        .unwrap();
        assert_eq!(&frozen_ids[row * top..(row + 1) * top], &expected_route.0);
        assert!(
            frozen_ids[row * top..(row + 1) * top]
                .windows(2)
                .all(|v| v[0] < v[1])
        );
        close(
            &frozen_weights[row * top..(row + 1) * top],
            &expected_route.1,
        );
        if e == 384 {
            assert!(
                frozen_ids[row * top..(row + 1) * top]
                    .iter()
                    .all(|&id| id >= 272)
            );
        }
        // Scalar replay consumes the captured route, not a second routing decision.
        let route_ids = prefill::row(&b[8], top, row);
        let ready = prefill::row(&b[6], 1, row);
        let inner = MetalTensor::zeros_f32(ctx, vec![f as u64, top as u64]).unwrap();
        let scalar_down = MetalTensor::zeros_f32(ctx, vec![h as u64, top as u64]).unwrap();
        submit(ctx, "v41_frozen_scalar", [0; 4], |enc| {
            metal::encode_all_slots_gate_up_swiglu_native(
                ctx,
                enc,
                &banks[0],
                &banks[1],
                &prefill::row(&scratch.hc.normalized, h, row),
                &route_ids,
                &ready,
                &inner,
                h,
                f,
                e,
                top,
                c.clamp,
            )?;
            metal::encode_all_slots_down_native(
                ctx,
                enc,
                &banks[2],
                &prefill::row(&b[13], f * top, row).view_subrange(0, vec![f as u64, top as u64]),
                &route_ids,
                &ready,
                &scalar_down,
                f,
                h,
                e,
                top,
            )
        })
        .unwrap();
        close(
            &activation[row * top * f..(row + 1) * top * f],
            f32_slice(&inner).unwrap(),
        );
        close(
            &down[row * top * h..(row + 1) * top * h],
            f32_slice(&scalar_down).unwrap(),
        );
        for s in 0..top {
            let slot = row * top + s;
            let weight = frozen_weights[slot];
            assert_eq!(
                bits(&weighted[slot * f..(slot + 1) * f]),
                bits(
                    &activation[slot * f..(slot + 1) * f]
                        .iter()
                        .map(|v| v * weight)
                        .collect::<Vec<_>>()
                )
            );
            let decoded = decode_rows(&banks[2], frozen_ids[slot] as usize * h, h).unwrap();
            let exact: Vec<f64> = decoded
                .chunks_exact(f)
                .map(|w| {
                    w.iter()
                        .zip(&weighted[slot * f..(slot + 1) * f])
                        .map(|(&w, &x)| f64::from(w) * f64::from(x))
                        .sum()
                })
                .collect();
            assert_down_f64(&down[slot * h..(slot + 1) * h], &exact);
            differs_from_weight_after |= down[slot * h..(slot + 1) * h]
                .iter()
                .zip(&wrong_down[slot * h..(slot + 1) * h])
                .any(|(&a, &b)| a.to_bits() != (b * weight).to_bits());
        }
        let expected: Vec<f32> = (0..h)
            .map(|d| {
                (0..top).fold(0.0f32, |sum, s| sum + down[(row * top + s) * h + d])
                    + shared[row * h + d]
            })
            .collect();
        assert_eq!(bits(&output[row * h..(row + 1) * h]), bits(&expected));
    }
    assert!(
        differs_from_weight_after,
        "fixture must distinguish weight-before-down rounding"
    );
    let mut start = 0;
    for rows in [1, n - 1] {
        let part = run_grouped(start, rows);
        assert_eq!(
            ids(&part.buffers[8]),
            frozen_ids[start * top..(start + rows) * top]
        );
        for (index, width) in [
            (9, top),
            (12, top * f),
            (13, top * f),
            (14, top * h),
            (5, h),
        ] {
            assert_eq!(
                bits(f32_slice(&part.buffers[index]).unwrap()),
                bits(&f32_slice(&b[index]).unwrap()[start * width..(start + rows) * width]),
                "partition buffer {index}"
            );
        }
        start += rows;
    }
    // The decode entry remains the native one-row route/weight/down composition.
    metal::dispatch_census_begin();
    submit(ctx, "v41_decode_dispatch", [0; 4], |enc| {
        ws.encode_routed(
            ctx,
            enc,
            c,
            layer,
            &ws.constants[0][3],
            &prefill::row(&scratch.hc.normalized, h, 0),
            &prefill::row(&b[0], e, 0),
            &prefill::row(&b[4], h, 0),
            &ws.buffers[20],
            &ws.statuses[0],
        )
    })
    .unwrap();
    let census = metal::dispatch_census_take();
    assert_eq!(census.len(), 6);
    assert!(
        census[1..3]
            .iter()
            .all(|r| r.kernel.starts_with("kernel_expert_native_"))
    );
    assert!(census[4].kernel.starts_with("kernel_expert_native_"));
    close(f32_slice(&ws.buffers[20]).unwrap(), &output[..h]);
    let residual: Vec<_> = (0..4 * h * n)
        .map(|i| (i % 31) as f32 * 0.013 - 0.2)
        .collect();
    let residual_t = upload(ctx, &residual, vec![h as u64, 4, n as u64]).unwrap();
    let post = [0.3, 0.7, 1.1, 0.9];
    let comb: Vec<_> = (0..16).map(|i| (i % 5) as f32 * 0.07).collect();
    let post_t = upload(ctx, &post.repeat(n), vec![4, n as u64]).unwrap();
    let comb_t = upload(ctx, &comb.repeat(n), vec![4, 4, n as u64]).unwrap();
    let hc = MetalTensor::zeros_f32(ctx, vec![h as u64, 4, n as u64]).unwrap();
    submit(ctx, "v41_frozen_hc", [0; 4], |enc| {
        metal::encode_mhc4_post_rows(ctx, enc, h, n, &b[5], &residual_t, &post_t, &comb_t, &hc)
    })
    .unwrap();
    for row in 0..n {
        let mixes = op::Mixes {
            pre: vec![0.0; 4],
            post: post.to_vec(),
            comb: comb.clone(),
        };
        close(
            &f32_slice(&hc).unwrap()[row * 4 * h..(row + 1) * 4 * h],
            &op::merge(
                &output[row * h..(row + 1) * h],
                &residual[row * 4 * h..(row + 1) * 4 * h],
                &mixes,
            )
            .unwrap(),
        );
    }
    compare_native_rows(ctx, &ws, c, layer, &scratch);
    if h == 5120 && n == 3 {
        let repeated = Rows::new(ctx, c, 33, 1).unwrap();
        write(&repeated.hc.normalized, &x.repeat(11));
        write(&repeated.buffers[0], &logits.repeat(11));
        write(&repeated.buffers[4], &shared.repeat(11));
        metal::dispatch_census_begin();
        submit(ctx, "v41_grouped_repeated_rows", [0; 4], |enc| {
            ws.encode_routed_rows(ctx, enc, c, layer, 0, &repeated, None)
        })
        .unwrap();
        let census = metal::dispatch_census_take();
        assert_eq!(census.len(), 6);
        assert!(census[1].kernel.contains("bucket"));
        for index in [2, 4] {
            assert!(census[index].kernel.ends_with("_f32x_grouped_slots"));
        }
        repeated.validate_status().unwrap();
        compare_native_rows(ctx, &ws, c, layer, &repeated);
    }
}

#[test]
fn new_dtype_triples_v41_widths_frozen_composition_and_dispatch() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    for (gate, down) in [
        (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS),
        (GgmlType::IQ3_XXS, GgmlType::IQ3_S),
        (GgmlType::IQ3_XXS, GgmlType::MXFP4),
        (GgmlType::Q3_K, GgmlType::MXFP4),
        (GgmlType::Q3_K, GgmlType::Q3_K),
    ] {
        with_grouped_rows(|| witness(&ctx, gate, down, 5120, 2304, 3, 2, 3));
    }
}

#[test]
fn new_dtype_triples_384_top6_frozen_routes_and_partition() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    for gate in [GgmlType::IQ3_XXS, GgmlType::Q3_K] {
        with_grouped_rows(|| witness(&ctx, gate, GgmlType::MXFP4, 256, 256, 384, 6, 33));
    }
}
