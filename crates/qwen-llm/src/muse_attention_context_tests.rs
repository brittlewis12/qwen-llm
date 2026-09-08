fn context_oracle_f64(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    head: usize,
    start: usize,
    end: usize,
) -> Vec<f64> {
    let mut scores = Vec::with_capacity(end - start);
    for position in start..end {
        let offset = 256 + position * 256 + (head / 16) * 128;
        let dot: f64 = (0..128)
            .map(|dim| q[head * 128 + dim] as f64 * half::f16::from_bits(k[offset + dim]).to_f64())
            .sum();
        scores.push(dot / 128.0_f64.sqrt());
    }
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut result = vec![0.0; 128];
    let mut denominator = 0.0;
    for (position, score) in (start..end).zip(scores) {
        let weight = (score - maximum).exp();
        denominator += weight;
        let offset = 256 + position * 256 + (head / 16) * 128;
        for dim in 0..128 {
            result[dim] += weight * half::f16::from_bits(v[offset + dim]).to_f64();
        }
    }
    for value in &mut result {
        *value /= denominator;
    }
    result
}

fn context_numerical_check(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut dot = 0.0_f64;
    let mut aa = 0.0_f64;
    let mut bb = 0.0_f64;
    let mut max_abs = 0.0_f32;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        dot += a as f64 * b as f64;
        aa += (a as f64).powi(2);
        bb += (b as f64).powi(2);
        max_abs = max_abs.max((a - b).abs());
    }
    let cosine = dot / (aa * bb).sqrt();
    assert!(
        cosine >= 0.999_999 && max_abs <= 5e-4,
        "context attention cosine={cosine} abs={max_abs}"
    );
    max_abs
}

#[test]
#[ignore = "serial Metal, model-context attention correctness with independent F64 queries"]
fn attention_model_context_crosschecks() {
    use objc2_metal::MTLBuffer;
    const CONTEXT: usize = 131072;
    let ctx = MetalContext::new().unwrap();
    let query_values = |rows: usize, uniform: bool| -> Vec<f32> {
        (0..rows * 4096)
            .map(|i| {
                if uniform {
                    0.0
                } else {
                    ((i / 4096 * 11 + i / 128 % 32 * 17 + i % 128 * 7) % 127) as f32 * 0.0379
                        - 2.3877
                }
            })
            .collect()
    };
    let mut k = vec![0_u16; 256 + CONTEXT * 256];
    let mut v = k.clone();
    for i in 0..CONTEXT * 256 {
        k[256 + i] = half::f16::from_f32(((i * 13 % 103) as f32 - 51.0) * 0.03125).to_bits();
        let position = i / 256;
        let value = if position % 2 == 0 { 0.75 } else { -0.75 }
            + (i % 256 / 128) as f32 * 0.25
            + (i % 128 % 7) as f32 * 0.03125
            + (position % 17) as f32 * 0.00390625;
        v[256 + i] = half::f16::from_f32(value).to_bits();
    }
    for (position, rows) in [(32783, 16), (65535, 16), (131071, 128)] {
        let q = query_values(rows, false);
        for dim in 0..128 {
            k[256 + position * 256 + 128 + dim] =
                half::f16::from_f32(q[(rows - 1) * 4096 + 31 * 128 + dim] * 8.0).to_bits();
        }
    }
    let key_storage = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&k),
        vec![k.len() as u64],
        GgmlType::F16,
    )
    .unwrap();
    let value_storage = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&v),
        vec![v.len() as u64],
        GgmlType::F16,
    )
    .unwrap();
    let key = key_storage.view_subrange(256, vec![(CONTEXT * 256) as u64]);
    let value = value_storage.view_subrange(256, vec![(CONTEXT * 256) as u64]);
    let submit = |encode: &mut dyn FnMut(&KernelEncoder)| {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder);
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(
            command.status(),
            objc2_metal::MTLCommandBufferStatus::Completed
        );
        assert!(command.error().is_none());
    };
    let poison = |tensor: &MetalTensor| unsafe {
        (tensor.buffer.contents().as_ptr() as *mut u8)
            .add(tensor.offset as usize)
            .write_bytes(0xff, tensor.n_elements() as usize * 4);
    };
    let guard = |storage: &MetalTensor| {
        let data = read_f32(storage);
        assert!(
            data[..4]
                .iter()
                .chain(&data[data.len() - 4..])
                .all(|&value| value == -77.0)
        );
    };
    for (base, rows, window, uniform) in [
        (32768, 16, None, false),
        (65520, 16, None, false),
        (130944, 128, None, false),
        (130944, 128, Some(2048), false),
        (131056, 16, None, true),
        (131056, 16, Some(2048), true),
    ] {
        let key = key.view_subrange(0, vec![((base + rows) * 256) as u64]);
        let value = value.view_subrange(0, vec![((base + rows) * 256) as u64]);
        let values = query_values(rows, uniform);
        let mut padded = vec![-77.0; 4];
        padded.extend(&values);
        let query = tensor_from_f32(&ctx, &padded).view_subrange(4, vec![values.len() as u64]);
        let storage = tensor_from_f32(&ctx, &vec![-77.0; values.len() + 8]);
        let output = storage.view_subrange(4, vec![values.len() as u64]);
        poison(&output);
        submit(&mut |encoder| {
            encode_muse_glimmer_attn_prefill_with_online(
                &ctx, encoder, &query, &key, &value, &output, rows, base, 32, 2, 128, window, true,
            )
            .unwrap()
        });
        let actual = read_f32(&output);
        assert!(actual.iter().all(|value| value.is_finite()));
        guard(&storage);
        let reference = MetalTensor::zeros_f32(&ctx, vec![4096]).unwrap();
        let mut gpu_delta = 0.0_f32;
        let mut f64_delta = 0.0_f64;
        for row in [0, rows - 1] {
            let end = base + row + 1;
            let start = window.map(|window| end.saturating_sub(window)).unwrap_or(0);
            let q = query.view_subrange((row * 4096) as u64, vec![4096]);
            let k_view =
                key.view_subrange((start * 256) as u64, vec![((end - start) * 256) as u64]);
            let v_view =
                value.view_subrange((start * 256) as u64, vec![((end - start) * 256) as u64]);
            poison(&reference);
            submit(&mut |encoder| {
                encode_muse_glimmer_attn_decode_online_f16kv_f32(
                    &ctx,
                    encoder,
                    &q,
                    &k_view,
                    &v_view,
                    &reference,
                    32,
                    2,
                    128,
                    end - start,
                )
                .unwrap()
            });
            let actual = &actual[row * 4096..(row + 1) * 4096];
            gpu_delta = gpu_delta.max(context_numerical_check(actual, &read_f32(&reference)));
            for head in [0, 31] {
                let expected = context_oracle_f64(
                    &values[row * 4096..(row + 1) * 4096],
                    &k,
                    &v,
                    head,
                    start,
                    end,
                );
                for dim in 0..128 {
                    f64_delta =
                        f64_delta.max((actual[head * 128 + dim] as f64 - expected[dim]).abs());
                }
            }
        }
        assert!(
            f64_delta <= 5e-4,
            "prefill independent F64 error {f64_delta}"
        );
        eprintln!(
            "MUSE_CONTEXT_JSON {}",
            serde_json::json!({"kind":"prefill_oracle","base":base,"rows":rows,"window":window,"uniform":uniform,"gpu_max_abs":gpu_delta,"f64_max_abs":f64_delta,"guards":true})
        );
    }
    let partial_storage = tensor_from_f32(
        &ctx,
        &vec![-77.0; split_attention::PARTIAL_ELEMENTS as usize + 8],
    );
    let partial = partial_storage.view_subrange(4, vec![split_attention::PARTIAL_ELEMENTS]);
    for (positions, uniform) in [
        (32785, false),
        (65537, false),
        (131071, false),
        (131072, false),
        (131072, true),
        (17, false),
    ] {
        let values = query_values(1, uniform);
        let mut padded = vec![-77.0; 4];
        padded.extend(&values);
        let query = tensor_from_f32(&ctx, &padded).view_subrange(4, vec![4096]);
        let storage = tensor_from_f32(&ctx, &vec![-77.0; 4104]);
        let output = storage.view_subrange(4, vec![4096]);
        let reference = MetalTensor::zeros_f32(&ctx, vec![4096]).unwrap();
        let key = key.view_subrange(0, vec![(positions * 256) as u64]);
        let value = value.view_subrange(0, vec![(positions * 256) as u64]);
        poison(&partial);
        poison(&output);
        poison(&reference);
        submit(&mut |encoder| {
            split_attention::encode(
                &ctx, encoder, &query, &key, &value, &output, &partial, 32, 2, 128, positions,
            )
            .unwrap()
        });
        let actual = read_f32(&output);
        submit(&mut |encoder| {
            encode_muse_glimmer_attn_decode_online_f16kv_f32(
                &ctx, encoder, &query, &key, &value, &reference, 32, 2, 128, positions,
            )
            .unwrap()
        });
        let gpu_delta = context_numerical_check(&actual, &read_f32(&reference));
        let mut f64_delta = 0.0_f64;
        for head in [0, 31] {
            let expected = context_oracle_f64(&values, &k, &v, head, 0, positions);
            for dim in 0..128 {
                f64_delta = f64_delta.max((actual[head * 128 + dim] as f64 - expected[dim]).abs());
            }
        }
        assert!(f64_delta <= 5e-4, "split independent F64 error {f64_delta}");
        guard(&partial_storage);
        guard(&storage);
        eprintln!(
            "MUSE_CONTEXT_JSON {}",
            serde_json::json!({"kind":"split_oracle","positions":positions,"uniform":uniform,"gpu_max_abs":gpu_delta,"f64_max_abs":f64_delta,"scratch_driver_bytes":partial_storage.buffer.length(),"guards":true})
        );
    }
}
