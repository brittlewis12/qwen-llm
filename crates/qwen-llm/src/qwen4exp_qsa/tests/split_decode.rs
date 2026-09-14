use super::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SplitArgs {
    id_count: u32,
    cache_capacity: u32,
    splits: u32,
    keys_per_split: u32,
}

fn guarded(ctx: &MetalContext, count: usize) -> (MetalTensor, MetalTensor) {
    let storage = MetalTensor::zeros_f32(ctx, vec![(count + 8) as u64]).unwrap();
    write_f32_tensor(&storage, &vec![-777.0; count + 8]);
    let view = storage.view_subrange(4, vec![count as u64]);
    (storage, view)
}

fn check_guards(storage: &MetalTensor) {
    let values = read_f32(storage);
    assert_eq!(&values[..4], &[-777.0; 4]);
    assert_eq!(&values[values.len() - 4..], &[-777.0; 4]);
}

#[test]
#[ignore = "serial Metal; model-free Flash-Next split attention numerical and timing screen"]
fn split_decode_attention_screen() {
    let ctx = MetalContext::new().expect("real Metal required; no skipped screen");
    let config = Qwen4ExpConfig::flash_next_reference();
    let g = QwenSparseAttentionMetalGeometry::from_config(&config, 3, 4_100).unwrap();
    let workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, g).unwrap();
    let split_pso = ctx.pipeline("kernel_qwen4exp_qsa_split_f16").unwrap();
    let merge_pso = ctx.pipeline("kernel_qwen4exp_qsa_split_merge_f32").unwrap();
    let mut random = 0x4d595df4d0f33173_u64;
    let mut sample = || {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        ((random >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0
    };
    let keys: Vec<f32> = (0..g.capacity * g.kv_width())
        .map(|_| f16::from_f32(sample()).to_f32())
        .collect();
    let values: Vec<f32> = (0..keys.len())
        .map(|_| f16::from_f32(sample()).to_f32())
        .collect();
    write_f16_prefix(&workspace.key_cache, &keys);
    write_f16_prefix(&workspace.value_cache, &values);
    let frozen_keys = read_tensor_bytes(&workspace.key_cache);
    let frozen_values = read_tensor_bytes(&workspace.value_cache);
    let gates: Vec<f32> = (0..g.query_width())
        .map(|i| match i % 19 {
            0 => -100.0,
            1 => 100.0,
            _ => sample() * 4.0,
        })
        .collect();
    write_f32_tensor(&workspace.raw_gate, &gates);
    let (output_storage, output) = guarded(&ctx, g.query_width());
    let (partial_storage, partials) = guarded(&ctx, 24 * 64 * 258);
    let cases = [
        (0, 1.0, 0),
        (1, 1.0, 0),
        (31, 1.0, 0),
        (32, 1.0, 0),
        (33, 1.0, 0),
        (128, 1.0, 0),
        (2_048, 1.0, 0),
        (2_051, 1.0, 0),
        (2_051, 32.0, 0),
        (2_051, 0.0, 0),
        (2_051, 1.0, 1),
        (2_051, 1.0, 2),
    ];
    for (count, amplitude, id_mode) in cases {
        let query: Vec<f32> = (0..g.query_width()).map(|_| sample() * amplitude).collect();
        write_f32_tensor(&workspace.query, &query);
        let ids: Vec<i32> = (0..g.output_width())
            .map(|i| {
                if i >= count || id_mode == 2 || (id_mode == 1 && i % 37 == 9) {
                    -1
                } else if id_mode == 1 && i % 41 == 13 {
                    g.capacity as i32
                } else if id_mode == 1 {
                    ((i * 2_053 + 17) % g.capacity) as i32
                } else {
                    ((i / 4) * 8 + i % 4) as i32
                }
            })
            .collect();
        write_i32_tensor(&workspace.token_ids, &ids);
        let splits = count.div_ceil(32).clamp(1, 64);
        let args = SplitArgs {
            id_count: count as u32,
            cache_capacity: g.capacity as u32,
            splits: splits as u32,
            keys_per_split: count.div_ceil(splits).max(1) as u32,
        };
        let active_partials = partials.view_subrange(0, vec![(24 * splits * 258) as u64]);
        let run = |candidate: bool, repeats: usize| {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let started = std::time::Instant::now();
            for _ in 0..repeats {
                if candidate {
                    encoder.set_pipeline(&split_pso);
                    encoder.set_bytes(0, &args);
                    encoder.set_tensor(1, &workspace.query);
                    encoder.set_tensor(2, &workspace.key_cache);
                    encoder.set_tensor(3, &workspace.value_cache);
                    encoder.set_tensor(4, &workspace.token_ids);
                    encoder.set_tensor(5, &active_partials);
                    encoder.dispatch(
                        MTLSize {
                            width: splits,
                            height: 2,
                            depth: 1,
                        },
                        MTLSize {
                            width: 128,
                            height: 1,
                            depth: 1,
                        },
                    );
                    encoder.set_pipeline(&merge_pso);
                    encoder.set_bytes(0, &args);
                    encoder.set_tensor(1, &active_partials);
                    encoder.set_tensor(2, &workspace.raw_gate);
                    encoder.set_tensor(3, &output);
                    encoder.dispatch(
                        MTLSize {
                            width: 24,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 1,
                            depth: 1,
                        },
                    );
                } else {
                    encode_attention_logits(&ctx, &encoder, &workspace, count).unwrap();
                    encode_attention_softmax_value_tensors(
                        &ctx,
                        &encoder,
                        &workspace.raw_gate,
                        &workspace.value_cache,
                        &workspace.token_ids,
                        &workspace.attention_logits,
                        &output,
                        g,
                        count,
                    )
                    .unwrap();
                }
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert!(command.error().is_none(), "{:?}", command.error());
            let gpu = (command.GPUEndTime() - command.GPUStartTime()) * 1e3 / repeats as f64;
            assert!(gpu.is_finite() && gpu > 0.0);
            (gpu, started.elapsed().as_secs_f64() * 1e3 / repeats as f64)
        };

        let expected = oracle(g, &query, &keys, &values, &gates, &ids[..count]);
        let mut baseline = None;
        if count > 0 {
            write_f32_tensor(&output, &vec![f32::NAN; g.query_width()]);
            write_f32_tensor(
                &workspace.attention_logits,
                &vec![f32::NAN; g.query_heads * g.output_width()],
            );
            run(false, 1);
            let observed = read_f32(&output);
            compare(&observed, &expected, "incumbent/F64");
            baseline = Some(observed);
        }
        write_f32_tensor(&output, &vec![f32::NAN; g.query_width()]);
        write_f32_tensor(&partials, &vec![f32::NAN; 24 * 64 * 258]);
        run(true, 1);
        let candidate = read_f32(&output);
        let max_abs = compare(&candidate, &expected, "split/F64");
        if let Some(baseline) = &baseline {
            let delta = candidate
                .iter()
                .zip(baseline)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!("split_qsa diagnostic count={count} max_abs_incumbent={delta:.9e}");
        }
        assert!(read_f32(&active_partials).iter().all(|v| !v.is_nan()));
        assert!(
            read_f32(&partials)[24 * splits * 258..]
                .iter()
                .all(|v| v.is_nan())
        );
        check_guards(&output_storage);
        check_guards(&partial_storage);
        eprintln!(
            "split_qsa correctness count={count} amplitude={amplitude} id_mode={id_mode} splits={splits} max_abs_f64={max_abs:.9e}"
        );

        if matches!(count, 128 | 2_048 | 2_051) && amplitude == 1.0 && id_mode == 0 {
            for candidate_arm in [false, true, true, false] {
                run(candidate_arm, 12);
            }
            let mut times = Vec::new();
            for candidate_arm in [false, true, true, false] {
                let time = run(candidate_arm, 12);
                let expected = if candidate_arm {
                    &candidate
                } else {
                    baseline.as_ref().unwrap()
                };
                assert_eq!(read_f32(&output), *expected, "timed replay changed output");
                times.push(time);
                check_guards(&output_storage);
                check_guards(&partial_storage);
            }
            let a = (times[0].0 + times[3].0) * 0.5;
            let b = (times[1].0 + times[2].0) * 0.5;
            let spread = (times[0].0 - times[3].0).abs() / a;
            let useful = spread <= 0.05
                && b <= a * 0.8
                && times[1].0 <= times[0].0 * 0.8
                && times[2].0 <= times[3].0 * 0.8;
            let verdict = if spread > 0.05 {
                "INCONCLUSIVE"
            } else if useful {
                "USEFUL"
            } else {
                "HOLD"
            };
            eprintln!(
                "split_qsa timing count={count} gpu_encode_submit_wait_ms_abba={times:?} saved_pct={:.3} control_spread={spread:.6} verdict={verdict}",
                (1.0 - b / a) * 100.0
            );
        }
        assert_eq!(read_f32(&workspace.query), query);
        assert_eq!(read_i32(&workspace.token_ids), ids);
    }
    assert_eq!(read_tensor_bytes(&workspace.key_cache), frozen_keys);
    assert_eq!(read_tensor_bytes(&workspace.value_cache), frozen_values);
    assert_eq!(read_f32(&workspace.raw_gate), gates);
}

fn compare(actual: &[f32], expected: &[f64], label: &str) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f64;
    let mut error = 0.0;
    let mut energy = 0.0;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "{label} nonfinite at {i}");
        let delta = (f64::from(a) - b).abs();
        maximum = maximum.max(delta);
        error += delta * delta;
        energy += b * b;
        assert!(delta <= 3e-5 + 3e-5 * b.abs(), "{label} at {i}: {a} != {b}");
    }
    let relative = (error / energy.max(1e-30)).sqrt();
    assert!(relative <= 3e-5, "{label} relative RMS={relative}");
    maximum
}

fn oracle(
    g: QwenSparseAttentionMetalGeometry,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    gates: &[f32],
    ids: &[i32],
) -> Vec<f64> {
    let mut result = vec![0.0; g.query_width()];
    for head in 0..24 {
        let valid: Vec<usize> = ids
            .iter()
            .copied()
            .filter(|&id| id >= 0 && (id as usize) < g.capacity)
            .map(|id| (id as usize * 2 + head / 12) * 256)
            .collect();
        if valid.is_empty() {
            continue;
        }
        let scores: Vec<f64> = valid
            .iter()
            .map(|&base| {
                (0..256)
                    .map(|d| f64::from(query[head * 256 + d]) * f64::from(keys[base + d]))
                    .sum::<f64>()
                    / 16.0
            })
            .collect();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mass: Vec<f64> = scores.iter().map(|&s| (s - maximum).exp()).collect();
        let denominator: f64 = mass.iter().sum();
        for d in 0..256 {
            let numerator: f64 = valid
                .iter()
                .zip(&mass)
                .map(|(&base, &p)| f64::from(values[base + d]) * p)
                .sum();
            let index = head * 256 + d;
            result[index] = numerator / denominator / (1.0 + (-f64::from(gates[index])).exp());
        }
    }
    result
}
