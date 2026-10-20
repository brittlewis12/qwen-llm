use super::super::toy;
use super::*;

#[path = "tests/native_rows.rs"]
mod native_rows;

fn production_lease() -> impl Sized {
    crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease")
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a - b).abs() <= 4e-5 + 4e-5 * b.abs(),
            "index {i}: {a} != {b}"
        );
    }
}

#[test]
fn stages_lagged_pre_weight_before_down_and_nonalias() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let mut model = weights.model(&ctx);
    let projection: Vec<_> = (0..128 * 24)
        .map(|i| ((i * 19 % 101) as f32 - 50.0) / 512.0)
        .collect();
    model.layers[0].ffn_hc.projection.tensor = upload(&ctx, &projection, vec![128, 24]).unwrap();
    model.layers[0].ffn_hc.scale = [0.7, 1.3, 0.4];
    model.layers[0].ffn_hc.base = (0..24).map(|i| (i as f32 - 9.0) / 17.0).collect();
    let c = &model.config;
    let layer = &model.layers[0];
    let mut workspace = Workspace::new(&model).unwrap();
    let x: Vec<_> = (0..128)
        .map(|i| ((i * 17 % 43) as f32 - 21.0) / 7.0)
        .collect();
    let pre = [0.7, 0.1, 0.3, 0.9];
    let (mixes, input) = layer.ffn_hc.prepare(&x, &pre, c).unwrap();
    let output = Forward::moe(&ctx, c, layer, &input).unwrap();
    let (actual, next) = workspace.run(&ctx, c, layer, 0, &x, &pre).unwrap();
    close(&actual, &op::merge(&output, &x, &mixes).unwrap());
    close(&next, &mixes.pre);
    let b = &workspace.buffers;
    close(f32_slice(&b[0]).unwrap(), &x);
    close(f32_slice(&b[1]).unwrap(), &pre);
    let inv = (op::dot(&x, &x) / x.len() as f32 + c.norm_eps)
        .sqrt()
        .recip();
    let projected: Vec<_> = layer
        .ffn_hc
        .projection
        .forward(&x)
        .unwrap()
        .iter()
        .map(|v| v * inv)
        .collect();
    close(f32_slice(&b[2]).unwrap(), &projected);
    close(f32_slice(&b[4]).unwrap(), &mixes.post);
    close(f32_slice(&b[5]).unwrap(), &mixes.comb);
    close(f32_slice(&b[6]).unwrap(), &op::collapse(&x, &pre).unwrap());
    close(f32_slice(&b[7]).unwrap(), &input);
    let (ids, weights) = route(
        &layer.router.forward(&input).unwrap(),
        &layer.bias,
        c.topk,
        c.gate_temperature,
        c.route_scale,
    )
    .unwrap();
    let actual_ids = unsafe {
        std::slice::from_raw_parts(b[10].buffer.contents().as_ptr().cast::<i32>(), c.topk)
    };
    assert_eq!(actual_ids, ids);
    close(f32_slice(&b[11]).unwrap(), &weights);
    let expected: Vec<_> = f32_slice(&b[13])
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, v)| v * weights[i / c.intermediate])
        .collect();
    close(f32_slice(&b[14]).unwrap(), &expected);
    close(f32_slice(&b[20]).unwrap(), &output);
    assert!(f32_slice(&b[13]).unwrap().iter().any(|v| *v != 0.0));
    assert!(f32_slice(&b[15]).unwrap().iter().any(|v| *v != 0.0));
}

#[test]
fn routing_ties_bias_absorption_temperature_extremes_and_top1() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    for (logits, bias, topk, temperature, scale) in [
        (
            (0..384).map(|i| i as f32 / 384.0).collect(),
            vec![0.0; 384],
            6,
            1.0,
            1.5,
        ),
        (
            (0..272).map(|i| i as f32 / 272.0).collect(),
            vec![0.0; 272],
            6,
            1.0,
            1.5,
        ),
        (
            (0..320).map(|i| i as f32 / 320.0).collect(),
            vec![0.0; 320],
            6,
            1.0,
            1.5,
        ),
        (vec![0.0; 8], vec![0.0; 8], 6, 1.0, 1.5),
        (vec![-5.0, 9.0, 1.0, 4.0], vec![1e20; 4], 2, 1.0, 1.5),
        (
            vec![-100.0, -40.0, -20.0, 0.0, 40.0, 100.0],
            vec![0.0; 6],
            4,
            0.25,
            1.5,
        ),
        (vec![-90.0, -80.0, -70.0, -60.0], vec![0.0; 4], 3, 2.0, 0.75),
        (vec![-20.0, -20.0, -20.0], vec![0.0; 3], 1, 1.0, 1.5),
        (vec![3.0, 2.0, 1.0], vec![-10.0, 0.0, 10.0], 2, 3.0, 2.0),
        (vec![f32::NAN, 0.0], vec![0.0; 2], 1, 1.0, 1.5),
        (vec![0.0, 1.0], vec![0.0, f32::INFINITY], 1, 1.0, 1.5),
    ] {
        let n = logits.len();
        let input = upload(&ctx, &logits, vec![n as u64]).unwrap();
        let bias_gpu = upload(&ctx, &bias, vec![n as u64]).unwrap();
        let scores = MetalTensor::zeros_f32(&ctx, vec![n as u64]).unwrap();
        let ids =
            MetalTensor::zeros_dtype_unstaged(&ctx, vec![topk as u64], GgmlType::I32).unwrap();
        let weights = MetalTensor::zeros_f32(&ctx, vec![topk as u64]).unwrap();
        let status = MetalTensor::zeros_dtype_unstaged(&ctx, vec![1], GgmlType::I32).unwrap();
        submit(&ctx, "route_test", [0; 4], |enc| {
            enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_route")?);
            enc.set_bytes(
                0,
                &RouteArgs {
                    experts: n as u32,
                    topk: topk as u32,
                    temperature,
                    scale,
                },
            );
            for (slot, tensor) in [&input, &bias_gpu, &scores, &ids, &weights, &status]
                .iter()
                .enumerate()
            {
                enc.set_tensor(slot + 1, tensor);
            }
            dispatch(enc, 1, 1);
            Ok(())
        })
        .unwrap();
        let status = unsafe { *status.buffer.contents().as_ptr().cast::<i32>() };
        match route(&logits, &bias, topk, temperature, scale) {
            Ok((expected_ids, expected_weights)) => {
                assert_eq!(status, metal::ROUTE_STATUS_READY);
                let actual = unsafe {
                    std::slice::from_raw_parts(ids.buffer.contents().as_ptr().cast::<i32>(), topk)
                };
                assert_eq!(actual, expected_ids);
                if n > 272 {
                    assert!(actual.iter().all(|&id| id >= 272));
                }
                close(f32_slice(&weights).unwrap(), &expected_weights);
                for (&actual, &expected) in
                    f32_slice(&weights).unwrap().iter().zip(&expected_weights)
                {
                    if expected.abs() > 1e-30 {
                        assert!(
                            (actual - expected).abs() <= expected.abs() * 2e-5,
                            "router tail {actual} != {expected}"
                        );
                    }
                }
            }
            Err(_) => assert!(status < 0),
        }
    }
}

#[test]
fn asymmetric_clamps_preserve_negative_gate_tail() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let gate = [-20.0, -2.0, -0.2, 0.2, 2.0, 20.0];
    let up = [-20.0, 20.0, -2.0, 2.0, -20.0, 20.0];
    let g = upload(&ctx, &gate, vec![6]).unwrap();
    let u = upload(&ctx, &up, vec![6]).unwrap();
    let out = MetalTensor::zeros_f32(&ctx, vec![6]).unwrap();
    submit(&ctx, "clamp_test", [0; 4], |enc| {
        metal::encode_clamped_swiglu(&ctx, enc, &g, &u, &out, 0.35)
    })
    .unwrap();
    close(
        f32_slice(&out).unwrap(),
        &op::clamped_swiglu(&gate, &up, 0.35).unwrap(),
    );
    assert!(f32_slice(&out).unwrap()[0].abs() < 1e-6);
}

#[test]
fn ordered_slots_then_shared_sum() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let slots = upload(&ctx, &[1e20, -1e20, 3.0], vec![1, 3]).unwrap();
    let shared = upload(&ctx, &[4.0], vec![1]).unwrap();
    let output = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
    let status = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[metal::ROUTE_STATUS_READY]),
        vec![1],
        GgmlType::I32,
    )
    .unwrap();
    submit(&ctx, "sum_test", [0; 4], |enc| {
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_sum")?);
        enc.set_bytes(0, &1u32);
        enc.set_bytes(1, &3u32);
        enc.set_bytes(6, &1u32);
        for (slot, tensor) in [&slots, &shared, &output, &status].iter().enumerate() {
            enc.set_tensor(slot + 2, tensor);
        }
        dispatch(enc, 1, 1);
        Ok(())
    })
    .unwrap();
    assert_eq!(f32_slice(&output).unwrap(), &[7.0]);
}

#[test]
fn grouped_mxfp4_rows_match_scalar_weight_before_down() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let c = &model.config;
    let layer = &model.layers[0];
    assert!(grouped_supported(c, layer));
    let mut scalar = Workspace::new(&model).unwrap();
    for n in [2, 17, 32, 64] {
        let x: Vec<_> = (0..n * 4 * c.hidden)
            .map(|i| ((i * 19 % 101) as f32 - 50.0) / 17.0)
            .collect();
        let pre = [0.7, 0.1, 0.3, 0.9].repeat(n);
        let mut expected = Vec::new();
        for row in 0..n {
            expected.extend(
                scalar
                    .run(
                        &ctx,
                        c,
                        layer,
                        0,
                        &x[row * 4 * c.hidden..(row + 1) * 4 * c.hidden],
                        &pre[row * 4..(row + 1) * 4],
                    )
                    .unwrap()
                    .0,
            );
        }
        let scratch = Rows::new(&ctx, c, n, 1).unwrap();
        let input = upload(&ctx, &x, vec![c.hidden as u64, 4, n as u64]).unwrap();
        let incoming = upload(&ctx, &pre, vec![4, n as u64]).unwrap();
        let output = prefill::allocate(&ctx, 4 * c.hidden, n, GgmlType::F32)
            .unwrap()
            .view_bytes(0, vec![c.hidden as u64, 4, n as u64]);
        submit(&ctx, "grouped_mxfp4_test", [0; 4], |enc| {
            scalar.encode_rows(&ctx, enc, c, layer, 0, &input, &incoming, &output, &scratch)
        })
        .unwrap();
        scratch.validate_status().unwrap();
        close(f32_slice(&output).unwrap(), &expected);
    }
}
