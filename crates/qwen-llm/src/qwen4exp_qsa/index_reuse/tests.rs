use super::*;
use objc2_metal::MTLCommandQueue;

fn guarded(ctx: &MetalContext, bytes: &[u8], dtype: GgmlType, shape: Vec<u64>) -> MetalTensor {
    let mut data = vec![0xa5; bytes.len() + 128];
    data[64..64 + bytes.len()].copy_from_slice(bytes);
    let size = if dtype == GgmlType::F16 { 2 } else { 4 };
    MetalTensor::from_bytes(ctx, &data, vec![(data.len() / size) as u64], dtype)
        .unwrap()
        .view_bytes(64, shape)
}
fn bits(t: &MetalTensor) -> Vec<u32> {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<u32>(),
            t.n_bytes() as usize / 4,
        )
        .to_vec()
    }
}
fn guards(t: &MetalTensor) {
    let b = unsafe {
        std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<u8>(), t.buffer.length())
    };
    assert!(
        b[..64]
            .iter()
            .chain(&b[64 + t.n_bytes() as usize..64 + t.n_bytes() as usize + 64])
            .all(|&v| v == 0xa5)
    );
}
fn next(s: &mut u64) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s % 20001) as f32 / 10000.0 - 1.0
}

#[test]
fn qsa_index_query_reuse_matches_incumbent_bits_f64_and_guards() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    if !ctx.tensor_ops_status().is_available() {
        return;
    }
    const Q: usize = 32;
    for start in [32765usize, 131069] {
        let capacity = (start + Q) / 4 + 5;
        let mut seed = 0x12345678abcdef01;
        let mut q: Vec<f32> = (0..512 * Q).map(|_| next(&mut seed)).collect();
        q[..512].fill(0.0);
        let k: Vec<half::f16> = (0..128 * capacity)
            .map(|_| half::f16::from_f32(next(&mut seed) * 0.5))
            .collect();
        let qt = guarded(
            &ctx,
            bytemuck::cast_slice(&q),
            GgmlType::F32,
            vec![128, 4, Q as u64],
        );
        let kt = guarded(
            &ctx,
            bytemuck::cast_slice(&k),
            GgmlType::F16,
            vec![128, capacity as u64],
        );
        let banks: Vec<_> = (0..3)
            .map(|_| {
                (
                    guarded(
                        &ctx,
                        bytemuck::cast_slice(&vec![0x7fc00001u32; capacity * Q]),
                        GgmlType::F32,
                        vec![capacity as u64, Q as u64],
                    ),
                    guarded(
                        &ctx,
                        bytemuck::cast_slice(&[-999i32; Q]),
                        GgmlType::I32,
                        vec![Q as u64],
                    ),
                )
            })
            .collect();
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        for (i, tile) in [1usize, 4, 8].into_iter().enumerate() {
            let name = match tile {
                1 => "kernel_qwen4exp_qsa_index_scores_packed_4x128_f16",
                4 => "kernel_qwen4exp_qsa_index_reuse4_f32",
                _ => "kernel_qwen4exp_qsa_index_reuse8_f32",
            };
            enc.set_pipeline(&ctx.pipeline(name).unwrap());
            enc.set_bytes(
                0,
                &PackedScoreArgs {
                    start_position: start as u32,
                    query_count: Q as u32,
                    ratio: 4,
                    block_capacity: capacity as u32,
                },
            );
            enc.set_tensor(1, &qt);
            enc.set_tensor(2, &kt);
            enc.set_tensor(3, &banks[i].0);
            enc.set_tensor(4, &banks[i].1);
            enc.dispatch(
                MTLSize {
                    width: ((start + Q) / 4).div_ceil(8),
                    height: Q.div_ceil(tile),
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).unwrap();
        let reference = bits(&banks[0].0);
        let visible = bits(&banks[0].1);
        for (i, (scores, counts)) in banks.iter().enumerate() {
            assert_eq!(bits(scores), reference, "start{start} tile{}", [1, 4, 8][i]);
            assert_eq!(bits(counts), visible);
            guards(scores);
            guards(counts);
        }
        let mut max_error = 0.0f64;
        for row in 0..Q {
            let visible = (start + row + 1) / 4;
            assert_eq!(bits(&banks[0].1)[row] as usize, visible);
            for block in [0, 1, 127, visible / 2, visible - 1] {
                let mut expected = 0.0f64;
                for head in 0..4 {
                    let dot = (0..128)
                        .map(|d| {
                            f64::from(q[row * 512 + head * 128 + d])
                                * f64::from(k[block * 128 + d].to_f32())
                        })
                        .sum::<f64>();
                    expected += dot.max(0.0);
                }
                expected *= f64::from(0.08838834764831845f32);
                let actual = f64::from(f32::from_bits(reference[row * capacity + block]));
                max_error = max_error.max((actual - expected).abs());
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= 2e-5 + 2e-5 * expected.abs()
                );
            }
            assert!(
                reference[row * capacity + visible..(row + 1) * capacity]
                    .iter()
                    .all(|&v| v == 0x7fc00001)
            );
        }
        guards(&qt);
        guards(&kt);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        crate::metal::dispatch_census_begin();
        let chosen = try_encode(
            &ctx,
            &enc,
            &qt,
            &kt,
            &banks[1].0,
            &banks[1].1,
            start,
            Q,
            4,
            capacity,
        )
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        assert_eq!(chosen, configured_tile() != 0);
        assert_eq!(census.len(), usize::from(chosen));
        enc.end();
        if chosen {
            cmd.commit();
            crate::metal::wait_completed(&cmd).unwrap();
            assert_eq!(bits(&banks[1].0), reference);
        }
        eprintln!(
            "index reuse start{start}: all score/visible bits equal for Q4/Q8; F64 max_abs={max_error:.9e}"
        );
    }
}

#[test]
fn qsa_selector_widths_preserve_cpu_ids_ties_errors_and_guards() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    const C: usize = 32768;
    const Q: usize = 32;
    const K: usize = 512;
    let mut seed = 0xcafebabef00d1234;
    let mut s: Vec<f32> = (0..C * Q)
        .map(|_| (next(&mut seed) * 32.0).round())
        .collect();
    let mut v = vec![C as i32; Q];
    for r in 0..C {
        s[r] = if r % 2 == 0 { 0.0 } else { -0.0 };
        s[C + r] = r as f32;
    }
    v[3] = 65;
    s[3 * C + 65..4 * C].fill(f32::NAN);
    v[4] = 0;
    v[5] = C as i32 + 1;
    v[6] = -1;
    s[7 * C + 17] = f32::INFINITY;
    s[8 * C + 23] = f32::NAN;
    s[9 * C + 29] = f32::NEG_INFINITY;
    let st = guarded(
        &ctx,
        bytemuck::cast_slice(&s),
        GgmlType::F32,
        vec![C as u64, Q as u64],
    );
    let vt = guarded(
        &ctx,
        bytemuck::cast_slice(&v),
        GgmlType::I32,
        vec![Q as u64],
    );
    let banks: Vec<_> = (0..4)
        .map(|_| {
            (
                guarded(
                    &ctx,
                    bytemuck::cast_slice(&vec![-999i32; K * Q]),
                    GgmlType::I32,
                    vec![K as u64, Q as u64],
                ),
                guarded(
                    &ctx,
                    bytemuck::cast_slice(&vec![-999i32; Q]),
                    GgmlType::I32,
                    vec![Q as u64],
                ),
                guarded(
                    &ctx,
                    bytemuck::cast_slice(&vec![-999i32; Q]),
                    GgmlType::I32,
                    vec![Q as u64],
                ),
            )
        })
        .collect();
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        query_count: u32,
        emit_ranked: u32,
    }
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    for (i, width) in [32usize, 64, 128, 256].into_iter().enumerate() {
        enc.set_pipeline(
            &ctx.pipeline("kernel_deepseek_v4_select_top_k_radix4_ids_f32")
                .unwrap(),
        );
        enc.set_bytes(
            0,
            &Args {
                row_capacity: C as u32,
                top_k: K as u32,
                query_count: Q as u32,
                emit_ranked: 0,
            },
        );
        enc.set_tensor(1, &st);
        enc.set_tensor(2, &vt);
        enc.set_tensor(3, &banks[i].0);
        enc.set_tensor(4, &banks[i].1);
        enc.set_tensor(5, &banks[i].2);
        enc.set_threadgroup_memory(0, width * 4);
        enc.set_threadgroup_memory(1, width * 4);
        enc.dispatch(
            MTLSize {
                width: Q,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
    enc.end();
    cmd.commit();
    crate::metal::wait_completed(&cmd).unwrap();
    let mut expected = vec![-1i32; K * Q];
    let mut counts = vec![0u32; Q];
    let mut status = vec![0u32; Q];
    for row in 0..Q {
        if v[row] <= 0 || v[row] as usize > C {
            status[row] = 1;
            continue;
        }
        let visible = v[row] as usize;
        let n = K.min(visible);
        counts[row] = n as u32;
        let mut ids: Vec<_> = (0..visible).collect();
        if s[row * C..row * C + visible].iter().any(|x| !x.is_finite()) {
            status[row] = 2;
            ids.truncate(n);
        } else {
            ids.sort_by(|&a, &b| {
                s[row * C + b]
                    .partial_cmp(&s[row * C + a])
                    .unwrap()
                    .then(a.cmp(&b))
            });
            ids.truncate(n);
            ids.sort_unstable();
        }
        for (i, id) in ids.into_iter().enumerate() {
            expected[row * K + i] = id as i32;
        }
    }
    let expected: Vec<u32> = expected.into_iter().map(|x| x as u32).collect();
    for (i, (ids, n, e)) in banks.iter().enumerate() {
        assert_eq!(bits(ids), expected, "width{}", [32, 64, 128, 256][i]);
        assert_eq!(bits(n), counts);
        assert_eq!(bits(e), status);
        guards(ids);
        guards(n);
        guards(e);
    }
    guards(&st);
    guards(&vt);
}
