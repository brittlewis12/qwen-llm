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

#[test]
fn host_batch_projection_checks_native_extent_shapes_and_aliases() {
    let w = || View {
        allocation: 1,
        buffer_bytes: 4096 * 1024 / 32 * 34,
        offset: 0,
        shape: &[4096, 1024],
        dtype: GgmlType::Q8_0,
        writable: false,
    };
    let x = || View {
        allocation: 2,
        buffer_bytes: 4096 * 2 * 4,
        offset: 0,
        shape: &[4096, 2],
        dtype: GgmlType::F32,
        writable: true,
    };
    let y = || View {
        allocation: 3,
        buffer_bytes: 1024 * 2 * 4,
        offset: 0,
        shape: &[1024, 2],
        dtype: GgmlType::F32,
        writable: true,
    };
    assert!(batch_projection_views(w(), x(), y(), 4096, 1024, 2).is_ok());
    for count in [0, 1, 3, 33, usize::MAX] {
        assert!(batch_projection_views(w(), x(), y(), 4096, 1024, count).is_err());
    }
    for mode in 0..4 {
        let mut bad = w();
        match mode {
            0 => bad.dtype = GgmlType::F16,
            1 => bad.offset = 1,
            2 => bad.buffer_bytes -= 1,
            _ => bad.shape = &[1024, 4096],
        }
        assert!(batch_projection_views(bad, x(), y(), 4096, 1024, 2).is_err());
    }
    let mut alias = y();
    alias.allocation = 2;
    assert!(batch_projection_views(w(), x(), alias, 4096, 1024, 2).is_err());
    let mut alias = x();
    alias.allocation = 1;
    assert!(batch_projection_views(w(), alias, y(), 4096, 1024, 2).is_err());
    let mut read_only = y();
    read_only.writable = false;
    assert!(batch_projection_views(w(), x(), read_only, 4096, 1024, 2).is_err());
}

#[test]
fn host_online_vector_alignment_is_stricter_than_scalar_alignment() {
    assert!(online_alignment(0, 8, 16, 32).is_ok());
    assert!(online_alignment(16, 24, 32, 48).is_ok());
    for offsets in [(4, 0, 0, 0), (0, 2, 0, 0), (0, 0, 4, 0), (0, 0, 0, 8)] {
        assert!(online_alignment(offsets.0, offsets.1, offsets.2, offsets.3).is_err());
    }
}

#[test]
#[ignore = "GPU primitive correctness; production lease and real wired-memory gate"]
fn gpu_online_attention_matches_f64_and_materialized_with_future_poison() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let prefix = 16u64;
    for (count, gain) in [
        (1, 1.),
        (32, 1.),
        (33, 1.),
        (128, 1.),
        (256, 1.),
        (257, 1.),
        (257, 0.),
        (257, 16.),
        (512, 1.),
        (512, -1.),
        (7168, 0.),
        (7168, 1.),
        (7168, 16.),
        (7169, 0.),
        (7169, 1.),
        (7169, 16.),
        (8192, 0.),
        (8192, 1.),
        (8192, 16.),
        (8192, -1.),
    ] {
        let plan = request(count as u32 + 3, 524288 - count as u32 - 3);
        let arena_elements = plan.arena_bytes() / 2;
        let buffer_bytes = (arena_elements + prefix + 16) * 2;
        let price = [buffer_bytes, 4096 * 4, 4096 * 4, 4096 * 4]
            .map(|bytes| {
                ctx.price_shared_buffer_upper(bytes)
                    .unwrap()
                    .priced_upper_bytes
            })
            .iter()
            .sum::<u64>();
        let q = (0..4096)
            .map(|i| ((i * 13 % 113) as f32 - 56.) * 0.037 * gain)
            .collect::<Vec<_>>();
        let _transaction = ctx.begin_allocation_transaction();
        let admission = crate::metal::evaluate_metal_memory_admission(
            price,
            256 * 1024 * 1024,
            ctx.memory_signals(),
            true,
        );
        assert!(admission.admitted, "{}", admission.reason.as_str());
        let mut bytes = vec![half::f16::NAN; (buffer_bytes / 2) as usize];
        let planes = plan.layer_planes(35).unwrap();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for p in 0..count {
            for i in 0..1024 {
                let k = half::f16::from_f32(if gain < 0. {
                    // Separated block maxima and spikes on either side of a merge.
                    q[(i / 128 * 4) * 128 + i % 128]
                        * if p == 255 || p == 256 || p == 511 || p == count - 1 {
                            8.
                        } else if (p / 256) % 2 == 0 {
                            -8.
                        } else {
                            0.
                        }
                } else {
                    (((i * 17 + p * 23) % 127) as f32 - 63.) * 0.041
                });
                let v = half::f16::from_f32(
                    (i / 128) as f32 * 0.31 + ((i * 11 + p * 7) % 37) as f32 * 0.017 - 0.4,
                );
                let index = p * 1024 + i;
                bytes[(prefix + planes.key.start / 2) as usize + index] = k;
                bytes[(prefix + planes.value.start / 2) as usize + index] = v;
                keys.push(k.to_f32());
                values.push(v.to_f32());
            }
        }
        let before = ctx.current_allocated_size();
        let backing = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&bytes),
            vec![bytes.len() as u64],
            GgmlType::F16,
        )
        .unwrap();
        let arena = backing.view_subrange(prefix, vec![arena_elements]);
        let query = tensor(&ctx, &q, &[128, 32]);
        let online = MetalTensor::zeros_f32(&ctx, vec![128, 32]).unwrap();
        let materialized = MetalTensor::zeros_f32(&ctx, vec![128, 32]).unwrap();
        assert!(ctx.current_allocated_size().saturating_sub(before) <= price);
        let append = plan.append(0, plan.start_position(), count as u32).unwrap();
        let token = append.token(count as u32 - 1).unwrap();
        execute(&ctx, |encoder| {
            encode_online_attention(&ctx, encoder, &token, 35, &arena, &query, &online)?;
            let control =
                encode_short_attention(&ctx, encoder, &token, 35, &arena, &query, &materialized);
            if count <= 7168 {
                control
            } else {
                assert!(control.unwrap_err().to_string().contains("7168-position"));
                Ok(())
            }
        });
        let actual = read(&online);
        let control = read(&materialized);
        let mut max_error = 0.0_f64;
        let mut max_control = 0.0_f64;
        for head in 0..32 {
            let base = head / 4 * 128;
            let scores = (0..count)
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
            let numerators = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = numerators.iter().sum::<f64>();
            for d in 0..128 {
                let expected = numerators
                    .iter()
                    .enumerate()
                    .map(|(p, probability)| probability * f64::from(values[p * 1024 + base + d]))
                    .sum::<f64>()
                    / denominator;
                let index = head * 128 + d;
                assert!(actual[index].is_finite() && control[index].is_finite());
                max_error = max_error.max((f64::from(actual[index]) - expected).abs());
                if count <= 7168 {
                    max_control = max_control
                        .max((f64::from(actual[index]) - f64::from(control[index])).abs());
                }
            }
        }
        eprintln!(
            "online positions={count} gain={gain} max_f64={max_error} max_materialized={:?}",
            (count <= 7168).then_some(max_control)
        );
        assert!(max_error < 2e-5, "positions={count} max_f64={max_error}");
        assert!(
            max_control < 2e-5,
            "positions={count} max_materialized={max_control}"
        );
        // The candidate must neither alter the visible cache nor touch guards/future rows.
        let after = unsafe {
            std::slice::from_raw_parts(
                backing.buffer.contents().as_ptr().cast::<half::f16>(),
                bytes.len(),
            )
        };
        assert!(
            after
                .iter()
                .zip(&bytes)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
}

pub(super) fn tensor(ctx: &MetalContext, values: &[f32], shape: &[u64]) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(values),
        shape.to_vec(),
        GgmlType::F32,
    )
    .unwrap()
}

#[test]
#[ignore = "GPU Q8 token-axis arithmetic/guard probe; production lease and real wired gate"]
fn gpu_q8_batch_projection_matches_singleton_bits_and_guards() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());
    let ctx = MetalContext::new().unwrap();
    for (n_in, n_out) in [
        (4096usize, 4096usize),
        (4096, 1024),
        (4096, 12288),
        (12288, 4096),
    ] {
        let weight_bytes = (n_in * n_out / 32 + 2) * 34;
        let input_bytes = (32 * n_in + 16) * 4;
        let output_bytes = (32 * n_out + 16) * 4;
        let _transaction = ctx.begin_allocation_transaction();
        let price = [weight_bytes, input_bytes, output_bytes, output_bytes]
            .iter()
            .map(|&b| {
                ctx.price_shared_buffer_upper(b as u64)
                    .unwrap()
                    .priced_upper_bytes
            })
            .sum::<u64>();
        let admission = crate::metal::evaluate_metal_memory_admission(
            price,
            256 * 1024 * 1024,
            ctx.memory_signals(),
            true,
        );
        assert!(admission.admitted, "{}", admission.reason.as_str());
        let before = ctx.current_allocated_size();
        let mut weights = vec![0xffu8; weight_bytes];
        for (block, bytes) in weights[34..weight_bytes - 34]
            .chunks_exact_mut(34)
            .enumerate()
        {
            bytes[..2].copy_from_slice(&half::f16::from_f32(0.015625).to_le_bytes());
            for (i, byte) in bytes[2..].iter_mut().enumerate() {
                *byte = (((block * 13 + i * 17) % 255) as i16 - 127) as i8 as u8;
            }
        }
        let wb = MetalTensor::from_bytes(
            &ctx,
            &weights,
            vec![(n_in * n_out + 64) as u64],
            GgmlType::Q8_0,
        )
        .unwrap();
        let weight = wb.view_bytes(34, vec![n_in as u64, n_out as u64]);
        let x = tensor(
            &ctx,
            &vec![-77.; input_bytes / 4],
            &[(input_bytes / 4) as u64],
        );
        let a = tensor(
            &ctx,
            &vec![-77.; output_bytes / 4],
            &[(output_bytes / 4) as u64],
        );
        let b = tensor(
            &ctx,
            &vec![-77.; output_bytes / 4],
            &[(output_bytes / 4) as u64],
        );
        assert!(ctx.current_allocated_size().saturating_sub(before) <= price);
        for count in [1usize, 2, 31, 32] {
            let mut input = vec![-77.; input_bytes / 4];
            input[8..8 + 32 * n_in].fill(f32::NAN);
            for (i, v) in input[8..8 + count * n_in].iter_mut().enumerate() {
                *v = ((i * 11 + i / n_in * 7) % 239) as f32 * 0.00390625 - 0.4;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    x.buffer.contents().as_ptr().cast::<f32>(),
                    input.len(),
                );
                for out in [&a, &b] {
                    std::slice::from_raw_parts_mut(
                        out.buffer.contents().as_ptr().cast::<f32>(),
                        output_bytes / 4,
                    )
                    .fill(-77.);
                }
            }
            let xv = checked_slice(&x, 8, vec![n_in as u64, count as u64], GgmlType::F32).unwrap();
            let av = checked_slice(&a, 8, vec![n_out as u64, count as u64], GgmlType::F32).unwrap();
            let bv = checked_slice(&b, 8, vec![n_out as u64, count as u64], GgmlType::F32).unwrap();
            assert!(checked_slice(&a, (output_bytes / 4) as u64, vec![1], GgmlType::F32).is_err());
            assert!(checked_slice(&a, u64::MAX, vec![2], GgmlType::F32).is_err());
            execute(&ctx, |enc| {
                encode_q8_projection_batch(&ctx, enc, &weight, &xv, &av, n_in, n_out, count)?;
                for row in 0..count {
                    crate::metal::encode_mat_vec_q8_0_f32(
                        &ctx,
                        enc,
                        &weight,
                        &checked_slice(&xv, (row * n_in) as u64, vec![n_in as u64], GgmlType::F32)?,
                        &checked_slice(
                            &bv,
                            (row * n_out) as u64,
                            vec![n_out as u64],
                            GgmlType::F32,
                        )?,
                        n_in,
                        n_out,
                    )?;
                }
                Ok(())
            });
            let actual = read(&av);
            let expected = read(&bv);
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits()),
                "shape={n_in}x{n_out} rows={count}"
            );
            for out in [&a, &b] {
                let values = read(out);
                assert!(
                    values[..8]
                        .iter()
                        .chain(&values[8 + count * n_out..])
                        .all(|&v| v == -77.)
                );
            }
            assert!(
                read(&x)
                    .iter()
                    .zip(&input)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            let weight_after = unsafe {
                std::slice::from_raw_parts(wb.buffer.contents().as_ptr().cast::<u8>(), weight_bytes)
            };
            assert_eq!(weight_after, weights);
            eprintln!("K2 Q8 batch {n_in}x{n_out} rows={count}: bitwise and guards pass");
        }
    }
}

pub(super) fn execute(ctx: &MetalContext, f: impl FnOnce(&KernelEncoder) -> Result<()>) {
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let result = f(&encoder);
    encoder.end();
    result.unwrap();
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert_eq!(
        command.status(),
        objc2_metal::MTLCommandBufferStatus::Completed
    );
}

pub(super) fn read(tensor: &MetalTensor) -> Vec<f32> {
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
