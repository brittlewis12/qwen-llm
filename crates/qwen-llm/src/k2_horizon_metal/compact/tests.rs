use super::super::tests::{execute, read, tensor};
use super::*;

fn quantize(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len() % 32, 0);
    values
        .chunks_exact(32)
        .flat_map(|values| {
            let mut block = [0u8; 34];
            let maximum = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let d = maximum / 127.;
            let scale = half::f16::from_f32(d);
            if values.iter().any(|v| !v.is_finite()) || !scale.is_finite() {
                block[..2].copy_from_slice(&0x7e00u16.to_le_bytes());
            } else if scale.to_bits() != 0 {
                block[..2].copy_from_slice(&scale.to_le_bytes());
                let inverse = 1.0_f32 / d;
                for (&value, byte) in values.iter().zip(&mut block[2..]) {
                    *byte = (value * inverse).round().clamp(-127., 127.) as i8 as u8;
                }
            }
            block
        })
        .collect()
}

fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(34)
        .flat_map(|block| {
            let scale = half::f16::from_le_bytes(block[..2].try_into().unwrap()).to_f32();
            block[2..].iter().map(move |&q| f32::from(q as i8) * scale)
        })
        .collect()
}

fn plan(capacity: u32) -> K2ShortContextPlan {
    K2ShortContextPlan::with_storage(
        crate::k2_horizon::K2HorizonConfig {
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
            rope_theta: 10_000_000.,
        },
        37,
        capacity,
        K2KvStorage::Q8_0,
    )
    .unwrap()
}

fn cases() -> Vec<[f32; 32]> {
    let mut ties = [0.; 32];
    ties[..8].copy_from_slice(&[127., -127., 0.5, -0.5, 1.5, -1.5, -0., 0.]);
    let mut unrounded = [0.; 32];
    unrounded[..3].copy_from_slice(&[12.7, 6.349, -6.349]);
    let mut invalid = [1.; 32];
    invalid[17] = f32::NAN;
    vec![
        [0.; 32],
        [-0.; 32],
        ties,
        unrounded,
        [1e-8; 32],
        [f32::from_bits(1); 32],
        [65504. * 127.; 32],
        [f32::MAX; 32],
        [f32::INFINITY; 32],
        [f32::NEG_INFINITY; 32],
        invalid,
        std::array::from_fn(|i| (i as f32 - 15.0) * 0.0078125),
    ]
}

#[test]
fn quantizer_contract_has_exact_ties_canonical_zero_and_visible_failures() {
    let fixtures = cases();
    for index in [0, 1, 4, 5] {
        assert_eq!(quantize(&fixtures[index]), vec![0; 34]);
    }
    let ties = quantize(&fixtures[2]);
    assert_eq!(&ties[..2], &0x3c00u16.to_le_bytes());
    assert_eq!(&ties[2..10], &[127, 129, 1, 255, 2, 254, 0, 0]);
    assert_eq!(&quantize(&fixtures[3])[2..5], &[127, 63, 193]);
    for index in [7, 8, 9, 10] {
        let bytes = quantize(&fixtures[index]);
        assert_eq!(&bytes[..2], &0x7e00u16.to_le_bytes());
        assert!(bytes[2..].iter().all(|&b| b == 0));
    }
    for index in [0, 1, 2, 3, 4, 5, 6, 11] {
        let values = fixtures[index];
        let bytes = quantize(&values);
        let maximum = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let d = maximum / 127.;
        let d16 = half::f16::from_f32(d).to_f32();
        let bound = if d16 == 0. {
            // Canonical zero also covers F32 scale underflow, where d itself is zero.
            f64::from(maximum)
        } else {
            0.5 * f64::from(d) + 127. * (f64::from(d) - f64::from(d16)).abs()
        };
        for (&actual, &expected) in decode(&bytes).iter().zip(&values) {
            assert!((f64::from(actual) - f64::from(expected)).abs() <= bound + 1e-6 * f64::from(d));
        }
    }
    let good = vec![0; 1088];
    assert!(validate_row(&good).is_ok());
    for bytes in [vec![], vec![0; 1087], vec![0; 1089]] {
        assert!(validate_row(&bytes).is_err());
    }
    for bits in [0x7e00u16, 0x7c00, 0xfc00, 0x8000, 0xbc00] {
        let mut bad = good.clone();
        bad[34 * 31..34 * 31 + 2].copy_from_slice(&bits.to_le_bytes());
        assert!(validate_row(&bad).is_err());
    }
    for code in [1u8, 128] {
        let mut bad = good.clone();
        bad[1070] = code;
        assert!(validate_row(&bad).is_err());
    }
}

#[test]
fn raw_cache_view_rejects_mismatched_storage_dtype_alignment_and_extent() {
    let view = || View {
        allocation: 1,
        buffer_bytes: 1088 + 2,
        offset: 2,
        shape: &[1088],
        dtype: GgmlType::I8,
        writable: true,
    };
    assert!(cache_view(view(), 1088, K2KvStorage::Q8_0, true).is_ok());
    assert!(cache_view(view(), 1088, K2KvStorage::F16, true).is_err());
    for mode in 0..5 {
        let mut bad = view();
        match mode {
            0 => bad.dtype = GgmlType::F16,
            1 => bad.offset = 1,
            2 => bad.buffer_bytes -= 1,
            3 => bad.shape = &[544],
            _ => bad.writable = false,
        }
        assert!(cache_view(bad, 1088, K2KvStorage::Q8_0, true).is_err());
    }
}

#[test]
#[ignore = "GPU compact KV primitives; production lease and wired gate; no model or public promotion"]
fn gpu_q8_store_bytes_and_inline_attention_match_independent_controls() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let plan = plan(260);
    let prefix = 2usize;
    let length = plan.arena_bytes() as usize + prefix + 2;
    let price = [
        length as u64,
        4096 * 4 + 32,
        4096 * 4 + 32,
        1024 * 4,
        1024 * 4,
    ]
    .map(|b| ctx.price_shared_buffer_upper(b).unwrap().priced_upper_bytes)
    .iter()
    .sum();
    let _transaction = ctx.begin_allocation_transaction();
    let admission = crate::metal::evaluate_metal_memory_admission(
        price,
        256 * 1024 * 1024,
        ctx.memory_signals(),
        true,
    );
    assert!(admission.admitted, "{}", admission.reason.as_str());
    let before = ctx.current_allocated_size();
    let backing =
        MetalTensor::from_bytes(&ctx, &vec![0xff; length], vec![length as u64], GgmlType::I8)
            .unwrap();
    let arena = backing.view_bytes(prefix as u64, vec![plan.arena_bytes()]);
    let key = tensor(&ctx, &vec![0.; 1024], &[128, 8]);
    let value = tensor(&ctx, &vec![0.; 1024], &[128, 8]);
    let qb = tensor(&ctx, &vec![-77.; 4096 + 8], &[4096 + 8]);
    let ob = tensor(&ctx, &vec![-77.; 4096 + 8], &[4096 + 8]);
    let query = qb.view_subrange(4, vec![128, 32]);
    let output = ob.view_subrange(4, vec![128, 32]);
    assert!(ctx.current_allocated_size().saturating_sub(before) <= price);
    let snapshot = || unsafe {
        std::slice::from_raw_parts(backing.buffer.contents().as_ptr().cast::<u8>(), length).to_vec()
    };
    let upload = |tensor: &MetalTensor, values: &[f32]| unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>(),
            values.len(),
        );
    };
    let fixtures = cases();
    let k = (0..32)
        .flat_map(|block| fixtures[block % fixtures.len()])
        .collect::<Vec<_>>();
    let v = (0..32)
        .flat_map(|block| fixtures[(block + 3) % fixtures.len()])
        .collect::<Vec<_>>();
    upload(&key, &k);
    upload(&value, &v);
    let append = plan.append(0, 37, 1).unwrap();
    let token = append.token(0).unwrap();
    let mut expected_bytes = snapshot();
    let writes = token.write_ranges(35).unwrap();
    for (range, values) in [(writes.key, &k), (writes.value, &v)] {
        expected_bytes[prefix + range.start as usize..prefix + range.end as usize]
            .copy_from_slice(&quantize(values));
    }
    execute(&ctx, |enc| {
        encode_store(&ctx, enc, &token, 35, &arena, &key, &value)
    });
    assert!(
        snapshot() == expected_bytes,
        "curated quantizer bytes or guard mismatch"
    );
    assert!(
        read(&key)
            .iter()
            .zip(&k)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        read(&value)
            .iter()
            .zip(&v)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    for (count, gain) in [
        (1, 1.),
        (32, 1.),
        (33, 1.),
        (128, 1.),
        (256, 1.),
        (257, 1.),
        (257, 0.),
        (257, 16.),
    ] {
        let q = (0..4096)
            .map(|i| ((i * 13 % 113) as f32 - 56.) * 0.037 * gain)
            .collect::<Vec<_>>();
        upload(&query, &q);
        unsafe {
            std::slice::from_raw_parts_mut(backing.buffer.contents().as_ptr().cast::<u8>(), length)
                .fill(0xff);
        }
        let mut expected = snapshot();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        let append = plan.append(0, 37, count).unwrap();
        for p in 0..count {
            let k = (0..1024)
                .map(|i| (((i * 17 + p * 23) % 127) as f32 - 63.) * 0.041)
                .collect::<Vec<_>>();
            let v = (0..1024)
                .map(|i| (i / 128) as f32 * 0.31 + ((i * 11 + p * 7) % 37) as f32 * 0.017 - 0.4)
                .collect::<Vec<_>>();
            upload(&key, &k);
            upload(&value, &v);
            let token = append.token(p).unwrap();
            execute(&ctx, |enc| {
                encode_store(&ctx, enc, &token, 35, &arena, &key, &value)
            });
            assert_eq!(read(&key), k);
            assert_eq!(read(&value), v);
            let written = snapshot();
            let ranges = token.write_ranges(35).unwrap();
            for (range, source, decoded) in
                [(ranges.key, &k, &mut keys), (ranges.value, &v, &mut values)]
            {
                let bytes = &written[prefix + range.start as usize..prefix + range.end as usize];
                validate_row(bytes).unwrap();
                // General arithmetic may round at a tie differently across backends;
                // bound decoded error independently, then use actual stored bytes.
                let dequantized = decode(bytes);
                for ((original, actual), block) in source
                    .chunks_exact(32)
                    .zip(dequantized.chunks_exact(32))
                    .zip(bytes.chunks_exact(34))
                {
                    let d = original.iter().map(|v| v.abs()).fold(0.0_f32, f32::max) / 127.;
                    let d16 = half::f16::from_le_bytes(block[..2].try_into().unwrap()).to_f32();
                    let bound = 0.5 * f64::from(d)
                        + 127. * (f64::from(d) - f64::from(d16)).abs()
                        + 1e-4 * f64::from(d);
                    for (&a, &b) in original.iter().zip(actual) {
                        assert!((f64::from(a) - f64::from(b)).abs() <= bound);
                    }
                }
                expected[prefix + range.start as usize..prefix + range.end as usize]
                    .copy_from_slice(bytes);
                decoded.extend(dequantized);
            }
            assert!(
                written == expected,
                "store touched another layer/future/guard"
            );
        }
        let token = append.token(count - 1).unwrap();
        execute(&ctx, |enc| {
            encode_attention(&ctx, enc, &token, 35, &arena, &query, &output)
        });
        let actual = read(&output);
        let mut max_error = 0.0_f64;
        for head in 0..32 {
            let base = head / 4 * 128;
            let scores = (0..count as usize)
                .map(|p| {
                    (0..128)
                        .map(|d| {
                            f64::from(q[head * 128 + d]) * f64::from(keys[p * 1024 + base + d])
                        })
                        .sum::<f64>()
                        * f64::from(128.0_f32.sqrt().recip())
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores
                .iter()
                .map(|s| (s - maximum).exp())
                .collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            for d in 0..128 {
                let expected = weights
                    .iter()
                    .enumerate()
                    .map(|(p, w)| w * f64::from(values[p * 1024 + base + d]))
                    .sum::<f64>()
                    / sum;
                assert!(actual[head * 128 + d].is_finite());
                max_error = max_error.max((f64::from(actual[head * 128 + d]) - expected).abs());
            }
        }
        assert!(
            max_error < 2e-5,
            "count={count} gain={gain} max_error={max_error}"
        );
        assert!(snapshot() == expected);
        for t in [&qb, &ob] {
            let x = read(t);
            assert!(x[..4].iter().chain(&x[4100..]).all(|&v| v == -77.));
        }
        assert_eq!(read(&query), q);
        eprintln!("Q8 inline attention count={count} gain={gain} max_f64={max_error}");
    }
}
