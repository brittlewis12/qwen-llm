use super::*;
use crate::k2_horizon::K2HorizonConfig;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

fn request(capacity: u32, base: u32) -> K2ShortContextPlan {
    K2ShortContextPlan::new(
        K2HorizonConfig {
            layer_count: 36,
            context_length: 524288,
            hidden_size: 4096,
            feed_forward_size: 12288,
            vocab_size: 250624,
            query_head_count: 32,
            kv_head_count: 8,
            key_head_dim: 128,
            value_head_dim: 128,
            norm_groups: 4,
            rms_epsilon: 1e-6,
            rope_dimension_count: 128,
            rope_theta: 10_000_000.0,
        },
        base,
        capacity,
    )
    .unwrap()
}

fn view() -> View<'static> {
    View {
        allocation: 1,
        buffer_bytes: 32768,
        offset: 128,
        shape: &[128, 32],
        dtype: GgmlType::F32,
        writable: true,
    }
}

#[test]
fn host_descriptor_checks_dtype_shape_alignment_extent_and_access() {
    let mut v = view();
    assert_eq!(
        v.check(&[128, 32], GgmlType::F32, true).unwrap().bytes,
        128..16512
    );
    assert!(v.check(&[128, 8], GgmlType::F32, false).is_err());
    assert!(v.check(&[4096], GgmlType::F32, false).is_err());
    assert!(v.check(&[128, 32], GgmlType::F16, false).is_err());
    v.writable = false;
    assert!(v.check(&[128, 32], GgmlType::F32, false).is_ok());
    assert!(v.check(&[128, 32], GgmlType::F32, true).is_err());
    v.offset = 129;
    assert!(v.check(&[128, 32], GgmlType::F32, false).is_err());
    v.offset = 32768 - 16384 + 4;
    assert!(v.check(&[128, 32], GgmlType::F32, false).is_err());
    v.offset = u64::MAX - 3;
    assert!(v.check(&[128, 32], GgmlType::F32, false).is_err());
    v.offset = 0;
    v.shape = &[u64::MAX, 2];
    assert!(v.check(v.shape, GgmlType::F32, false).is_err());
    v.shape = &[0];
    assert!(v.check(v.shape, GgmlType::F32, false).is_err());
}

#[test]
fn host_checked_ranges_reject_aliases_but_accept_adjacent_views() {
    let left = view().check(&[128, 32], GgmlType::F32, true).unwrap();
    let mut right = view();
    assert!(
        disjoint(
            &left,
            &right.check(&[128, 32], GgmlType::F32, true).unwrap()
        )
        .is_err()
    );
    right.shape = &[128, 8];
    for offset in [128, 1024, 16508] {
        right.offset = offset;
        assert!(disjoint(&left, &right.check(&[128, 8], GgmlType::F32, true).unwrap()).is_err());
    }
    right.offset = 16512;
    assert!(disjoint(&left, &right.check(&[128, 8], GgmlType::F32, true).unwrap()).is_ok());
    right.allocation = 2;
    right.offset = 128;
    assert!(disjoint(&left, &right.check(&[128, 8], GgmlType::F32, true).unwrap()).is_ok());
}

#[test]
fn host_cache_descriptor_is_f16_only_and_preserves_nonzero_arena_offset() {
    let plan = request(5, 37);
    let shape = [plan.arena_bytes() / 2];
    let mut v = View {
        allocation: 7,
        buffer_bytes: plan.arena_bytes() + 256,
        offset: 256,
        shape: &shape,
        dtype: GgmlType::F16,
        writable: true,
    };
    let checked = v.check(&shape, GgmlType::F16, true).unwrap();
    assert_eq!(checked.bytes.end, v.buffer_bytes);
    for dtype in [GgmlType::F32, GgmlType::Q8_0, GgmlType::Q4_K] {
        v.dtype = dtype;
        assert!(v.check(&shape, GgmlType::F16, true).is_err());
    }
    v.dtype = GgmlType::F16;
    v.buffer_bytes -= 1;
    assert!(v.check(&shape, GgmlType::F16, true).is_err());
}

fn tensor(ctx: &MetalContext, values: &[f32], shape: &[u64]) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(values),
        shape.to_vec(),
        GgmlType::F32,
    )
    .unwrap()
}

fn execute(ctx: &MetalContext, f: impl FnOnce(&KernelEncoder) -> Result<()>) {
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let result = f(&encoder);
    encoder.end();
    result.unwrap();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(
        command.status(),
        objc2_metal::MTLCommandBufferStatus::Completed
    );
}

fn read(tensor: &MetalTensor) -> Vec<f32> {
    assert_eq!(tensor.dtype, GgmlType::F32);
    unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>(),
            tensor.n_elements() as usize,
        )
        .to_vec()
    }
}

#[test]
#[ignore = "GPU execution requires a separately authorized synthetic validation window"]
fn gpu_grouped_norm_and_full_neox_match_scalar_formulas() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let plan = request(1, 37);
    let append = plan.append(0, 37, 1).unwrap();
    let token = append.token(0).unwrap();
    let x = (0..4096)
        .map(|i| 0.1 + (i % 71) as f32 / 20.0 + (i / 1024) as f32)
        .collect::<Vec<_>>();
    let gamma = (0..4096)
        .map(|i| 0.3 + i as f32 / 4096.0)
        .collect::<Vec<_>>();
    let xt = tensor(&ctx, &x, &[4096]);
    let wt = tensor(&ctx, &gamma, &[4096]);
    let out = MetalTensor::zeros_f32(&ctx, vec![4096]).unwrap();
    execute(&ctx, |enc| {
        encode_grouped_norm(&ctx, enc, &plan, &xt, &wt, &out)
    });
    let actual = read(&out);
    for group in 0..4 {
        let start = group * 1024;
        let mean = x[start..start + 1024]
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            / 1024.0;
        for i in start..start + 1024 {
            let expected =
                f64::from(x[i]) / (mean + f64::from(1e-6_f32)).sqrt() * f64::from(gamma[i]);
            assert!((f64::from(actual[i]) - expected).abs() < 2e-6, "norm[{i}]");
        }
    }
    let q = (0..4096)
        .map(|i| (i % 37) as f32 / 37.0 - 0.5)
        .collect::<Vec<_>>();
    let k = q[..1024].to_vec();
    let qt = tensor(&ctx, &q, &[128, 32]);
    let kt = tensor(&ctx, &k, &[128, 8]);
    execute(&ctx, |enc| encode_full_rope(&ctx, enc, &token, &qt, &kt));
    for (source, actual) in [(&q, read(&qt)), (&k, read(&kt))] {
        for (head, values) in source.chunks_exact(128).enumerate() {
            for pair in 0..64 {
                let angle = 37.0 * 10_000_000_f64.powf(-2.0 * pair as f64 / 128.0);
                let (sin, cos) = angle.sin_cos();
                let a = f64::from(values[pair]);
                let b = f64::from(values[pair + 64]);
                for (channel, expected) in
                    [(pair, a * cos - b * sin), (pair + 64, b * cos + a * sin)]
                {
                    assert!(
                        (f64::from(actual[head * 128 + channel]) - expected).abs() < 2e-5,
                        "rope head={head} channel={channel}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "GPU execution requires a separately authorized synthetic validation window"]
fn gpu_f16_store_and_gqa4_attention_exclude_poisoned_future_rows() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let plan = request(5, 37);
    let append = plan.append(0, 37, 3).unwrap();
    for arena_offset in [0, 64] {
        let arena_elements = plan.arena_bytes() / 2;
        let poison = vec![half::f16::NAN; (arena_elements + arena_offset + 64) as usize];
        let backing = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&poison),
            vec![poison.len() as u64],
            GgmlType::F16,
        )
        .unwrap();
        let arena = backing.view_subrange(arena_offset, vec![arena_elements]);
        let q = (0..4096)
            .map(|i| (i % 23) as f32 / 23.0 - 0.4)
            .collect::<Vec<_>>();
        let query = tensor(&ctx, &q, &[128, 32]);
        let output = MetalTensor::zeros_f32(&ctx, vec![128, 32]).unwrap();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for position in 0..3 {
            let token = append.token(position).unwrap();
            let k = (0..1024)
                .map(|i| ((i + position as usize * 13) % 47) as f32 / 47.0 - 0.2)
                .collect::<Vec<_>>();
            let v = (0..1024)
                .map(|i| (i / 128) as f32 * 0.31 + (i % 17) as f32 * 0.037 + position as f32 * 0.17)
                .collect::<Vec<_>>();
            let kt = tensor(&ctx, &k, &[128, 8]);
            let vt = tensor(&ctx, &v, &[128, 8]);
            keys.push(
                k.iter()
                    .map(|&v| half::f16::from_f32(v).to_f32())
                    .collect::<Vec<_>>(),
            );
            values.push(
                v.iter()
                    .map(|&v| half::f16::from_f32(v).to_f32())
                    .collect::<Vec<_>>(),
            );
            execute(&ctx, |enc| {
                encode_store_kv(&ctx, enc, &token, 35, &arena, &kt, &vt)?;
                encode_short_attention(&ctx, enc, &token, 35, &arena, &query, &output)
            });
            let actual = read(&output);
            for head in 0..32 {
                let base = head / 4 * 128;
                let scores = keys
                    .iter()
                    .map(|key| {
                        (0..128)
                            .map(|d| f64::from(q[head * 128 + d]) * f64::from(key[base + d]))
                            .sum::<f64>()
                            / 128_f64.sqrt()
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probabilities = scores.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
                let sum = probabilities.iter().sum::<f64>();
                for d in 0..128 {
                    let expected = probabilities
                        .iter()
                        .zip(&values)
                        .map(|(&p, values)| p / sum * f64::from(values[base + d]))
                        .sum::<f64>();
                    assert!(
                        (f64::from(actual[head * 128 + d]) - expected).abs() < 2e-5,
                        "attention position={position} head={head} channel={d}"
                    );
                }
            }
        }
        let stored = unsafe {
            std::slice::from_raw_parts(
                backing.buffer.contents().as_ptr().cast::<half::f16>(),
                poison.len(),
            )
        };
        assert!(stored[..arena_offset as usize].iter().all(|v| v.is_nan()));
        assert!(
            stored[(arena_offset + arena_elements) as usize..]
                .iter()
                .all(|v| v.is_nan())
        );
    }
}
