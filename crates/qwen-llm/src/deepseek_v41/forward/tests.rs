use super::*;
use std::collections::BTreeMap;

#[test]
fn full_chunk_selector_keeps_large_residents_at_2048() {
    let gib = 1u64 << 30;
    assert_eq!(prefill_rows_for_resident(199 * gib), MAX_PREFILL_CHUNK_ROWS);
    assert_eq!(prefill_rows_for_resident(217 * gib), MAX_PREFILL_CHUNK_ROWS);
    assert_eq!(prefill_rows_for_resident(240 * gib), MAX_PREFILL_CHUNK_ROWS);
    assert_eq!(prefill_rows_for_resident(241 * gib), PREFILL_CHUNK_ROWS);
}

#[test]
fn folded_engram_storage_is_explicit() {
    let separate = EngramGate::Separate {
        q: vec![0.5; 16],
        k: vec![0.25; 16],
    };
    let folded = EngramGate::Folded(vec![0.125; 16]);
    let mut a = vec![0.75; 16];
    let mut b = a.clone();
    let kv = vec![0.25; 20];
    separate.inject(&mut a, &kv, 1e-6).unwrap();
    folded.inject(&mut b, &kv, 1e-6).unwrap();
    assert_eq!(a, b);
    assert!(folded.inject(&mut b, &kv[..19], 1e-6).is_err());
}

#[test]
fn gguf_dense_bf16_q8_bounded_rows_and_group_offsets() {
    let ctx = MetalContext::new().unwrap();
    let input = 32;
    let output = 8;
    let x: Vec<f32> = (0..input).map(|i| (i as f32 - 11.0) / 31.001).collect();
    for dtype in [GgmlType::BF16, GgmlType::Q8_0] {
        let mut bytes = Vec::new();
        for row in 0..16 {
            if dtype == GgmlType::BF16 {
                for j in 0..32 {
                    let v = ((row + j) as f32 - 17.0) / 32.0;
                    bytes.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
                }
            } else {
                bytes.extend_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
                bytes.extend((0..32).map(|j| ((row + j) as i8 - 17) as u8));
            }
        }
        let tensor = MetalTensor::from_bytes(&ctx, &bytes, vec![32, 16], dtype).unwrap();
        assert!(decode_rows(&tensor, 16, 1).is_err());
        let stride = row_bytes(&tensor).unwrap();
        let projections: Vec<_> = (0..2)
            .map(|group| Linear {
                native_control: native_control(&ctx, dtype).unwrap(),
                ctx: &ctx,
                tensor: tensor.view_bytes(group * 8 * stride, vec![32, 8]),
                input,
                output,
            })
            .collect();
        for (group, projection) in projections.iter().enumerate() {
            let actual = projection.forward(&x).unwrap();
            for (row, actual) in actual.into_iter().enumerate() {
                let values = decode_rows(&tensor, group * output + row, 1).unwrap();
                assert_eq!(values.len(), input);
                let expected = op::dot(&values, &x);
                assert!(
                    (actual - expected).abs() < 2e-5,
                    "{dtype:?}: {actual} != {expected}"
                );
            }
        }
        let grouped =
            <Linear<'_> as op::Projection>::forward_grouped(&projections, &x.repeat(2)).unwrap();
        let serial: Vec<_> = projections
            .iter()
            .flat_map(|p| p.forward(&x).unwrap())
            .collect();
        assert_eq!(grouped, serial);
    }
}

#[test]
fn gguf_quant_bank_real_expert_counts_high_ids() {
    let ctx = MetalContext::new().unwrap();
    for experts in [384, 272, 320] {
        let mut bytes = Vec::new();
        for expert in 0..experts {
            for row in 0..8 {
                bytes.extend_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
                bytes.extend((0..32).map(|j| (((expert + row + j) % 31) as i8 - 15) as u8));
            }
        }
        let bank =
            MetalTensor::from_bytes(&ctx, &bytes, vec![32, 8, experts as u64], GgmlType::Q8_0)
                .unwrap();
        let selected = [experts as i32 - 1, experts as i32 - 2];
        if experts > 272 {
            assert!(selected.iter().all(|&id| id >= 272));
        }
        let ids = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&selected),
            vec![2],
            GgmlType::I32,
        )
        .unwrap();
        let status = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[metal::ROUTE_STATUS_READY]),
            vec![1],
            GgmlType::I32,
        )
        .unwrap();
        let x: Vec<_> = (0..64).map(|i| (i as f32 - 20.0) / 63.001).collect();
        let input = upload(&ctx, &x, vec![32, 2]).unwrap();
        let output = MetalTensor::zeros_f32(&ctx, vec![8, 2]).unwrap();
        submit(&ctx, "quant_bank", [0; 4], |enc| {
            metal::encode_all_slots_down_native(
                &ctx, enc, &bank, &input, &ids, &status, &output, 32, 8, experts, 2,
            )
        })
        .unwrap();
        for (slot, &expert) in selected.iter().enumerate() {
            for row in 0..8 {
                let values = decode_rows(&bank, expert as usize * 8 + row, 1).unwrap();
                let expected = op::dot(&values, &x[slot * 32..(slot + 1) * 32]);
                assert!((f32_slice(&output).unwrap()[slot * 8 + row] - expected).abs() < 2e-5);
            }
        }
    }
}

#[test]
fn gguf_dense_extra_iq_formats_preserve_f32_operands() {
    let ctx = MetalContext::new().unwrap();
    for dtype in [GgmlType::IQ2_XXS, GgmlType::IQ1_S, GgmlType::IQ1_M] {
        assert!(dense_linear_supported(dtype));
        let (_, stride) = dtype.storage_layout().unwrap();
        let mut bytes = vec![0u8; stride as usize * 8];
        for (row, block) in bytes.chunks_exact_mut(stride as usize).enumerate() {
            for (i, byte) in block.iter_mut().enumerate() {
                *byte = (i * 11 + row * 7) as u8;
            }
            let scale = half::f16::from_f32((row + 1) as f32 / 1024.0).to_bits();
            if dtype == GgmlType::IQ1_M {
                for i in 0..4 {
                    let old = u16::from_le_bytes([block[48 + 2 * i], block[49 + 2 * i]]);
                    let value = (old & 0x0fff) | (((scale >> (4 * i)) & 15) << 12);
                    block[48 + 2 * i..50 + 2 * i].copy_from_slice(&value.to_le_bytes());
                }
            } else {
                block[..2].copy_from_slice(&scale.to_le_bytes());
            }
        }
        let tensor = MetalTensor::from_bytes(&ctx, &bytes, vec![256, 8], dtype).unwrap();
        let linear = Linear {
            ctx: &ctx,
            tensor,
            input: 256,
            output: 8,
            native_control: native_control(&ctx, dtype).unwrap(),
        };
        assert!(linear.native_control.is_none());
        let x: Vec<_> = (0..256).map(|i| 70000.125 + i as f32 / 31.001).collect();
        let actual = linear.forward(&x).unwrap();
        let input = upload(&ctx, &x, vec![64, 4]).unwrap();
        let output = MetalTensor::zeros_f32(&ctx, vec![4, 2]).unwrap();
        submit(&ctx, "iq_shaped_activation", [0; 4], |enc| {
            linear.encode(enc, &input, &output)
        })
        .unwrap();
        assert_eq!(actual, f32_slice(&output).unwrap());
        for (row, &actual) in actual.iter().enumerate() {
            let values = decode_rows(&linear.tensor, row, 1).unwrap();
            let expected: f64 = values
                .iter()
                .zip(&x)
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum();
            assert!(actual.is_finite(), "{dtype:?} narrowed F32 activations");
            assert!(
                (f64::from(actual) - expected).abs() <= 0.01 + 3e-5 * expected.abs(),
                "{dtype:?}: {actual} != {expected}"
            );
        }
    }
    for dtype in [GgmlType::Q5_0, GgmlType::Q5_1] {
        assert!(!dense_linear_supported(dtype));
        assert!(
            native_control(&ctx, dtype)
                .err()
                .unwrap()
                .contains("unsupported NativeF32")
        );
    }
}

#[test]
fn grouped_projection_default_validates_and_preserves_order() {
    use op::Projection;
    let p = vec![
        op::Linear {
            input: 2,
            output: 1,
            weight: vec![2.0, -1.0]
        };
        2
    ];
    assert_eq!(
        op::Linear::forward_grouped(&p, &[1.0, 3.0, 4.0, 2.0]).unwrap(),
        [-1.0, 6.0]
    );
    assert!(op::Linear::forward_grouped(&[], &[1.0]).is_err());
    assert!(op::Linear::forward_grouped(&p, &[]).is_err());
    assert!(op::Linear::forward_grouped(&p, &[1.0; 3]).is_err());
}

#[test]
fn grouped_linear_bitwise_serial_gpu_witness() {
    use op::Projection;
    let ctx = MetalContext::new().expect("Metal required");
    for (input, output) in [(32, 7), (4096, 1024)] {
        let projections: Vec<_> = (0..8)
            .map(|g| {
                let weights: Vec<_> = (0..input * output)
                    .map(|i| (((i * 17 + g * 31) % 257) as f32 - 128.0) / 257.0)
                    .collect();
                Linear {
                    native_control: None,
                    ctx: &ctx,
                    tensor: upload(&ctx, &weights, vec![input as u64, output as u64]).unwrap(),
                    input,
                    output,
                }
            })
            .collect();
        let x: Vec<_> = (0..8 * input)
            .map(|i| ((i * 13 % 127) as f32 - 63.0) / 127.0)
            .collect();
        metal::dispatch_census_begin();
        let scope = profile::Scope::begin().unwrap();
        let serial: Vec<_> = projections
            .iter()
            .zip(x.chunks_exact(input))
            .flat_map(|(p, x)| p.forward(x).unwrap())
            .collect();
        let serial_report = scope.finish();
        let serial_census = metal::dispatch_census_take();
        metal::dispatch_census_begin();
        let scope = profile::Scope::begin().unwrap();
        let grouped = Linear::forward_grouped(&projections, &x).unwrap();
        let report = scope.finish();
        let census = metal::dispatch_census_take();
        assert_eq!(serial_census.len(), 8);
        assert_eq!(census.len(), 8);
        for (a, b) in census.iter().zip(&serial_census) {
            assert!(a.kernel.starts_with("kernel_mat_vec_f32_f32"));
            assert_eq!(a.kernel, b.kernel);
            assert_eq!(a.encoder_ordinal, census[0].encoder_ordinal);
        }
        assert_eq!(serial_report.operations[0].completed_submits, 8);
        assert_eq!(report.operations[0].operation, "linear_grouped");
        assert_eq!(report.operations[0].dimensions, [input, output, 8, 1]);
        assert_eq!(report.operations[0].completed_submits, 1);
        assert!(serial.iter().chain(&grouped).all(|x| x.is_finite()));
        assert_eq!(
            serial.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            grouped.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert!(Linear::forward_grouped(&[], &x).is_err());
        assert!(Linear::forward_grouped(&projections, &x[..x.len() - 1]).is_err());
        println!(
            "grouped witness {input}->{output} x8: bitwise equal, 8 F32 dispatches, submits 8->1"
        );
    }
}

#[test]
fn profile_linear_completed_and_scope_cleanup() {
    assert!(profile::start().is_none());
    let ctx = MetalContext::new().expect("Metal required");
    let linear = Linear {
        native_control: None,
        ctx: &ctx,
        tensor: upload(&ctx, &[1.0; 64], vec![8, 8]).unwrap(),
        input: 8,
        output: 8,
    };
    let expected = linear.forward(&[1.0; 8]).unwrap();
    let scope = profile::Scope::begin().unwrap();
    assert!(profile::Scope::begin().is_err());
    for _ in 0..2 {
        assert_eq!(linear.forward(&[1.0; 8]).unwrap(), expected);
    }
    let report = scope.finish();
    assert!(report.token_wall_ms.is_finite() && report.token_wall_ms > 0.0);
    let op = &report.operations[0];
    assert_eq!(op.dimensions, [8, 8, 0, 1]);
    assert_eq!((op.count, op.completed_submits), (2, 2));
    assert_eq!(op.invalid_gpu_timestamps, 0);
    assert!(op.gpu_ms.unwrap().is_finite() && op.gpu_ms.unwrap() > 0.0);
    assert!(op.encode_ms.is_finite() && op.commit_wait_ms.is_finite());
    assert!(op.wall_ms >= op.encode_ms + op.commit_wait_ms);
    assert!(profile::start().is_none());
    let fail = || -> Result<()> {
        let _scope = profile::Scope::begin()?;
        linear.forward(&[])?;
        Ok(())
    };
    assert!(fail().is_err());
    assert!(profile::start().is_none());
    assert!(
        std::panic::catch_unwind(|| {
            let _scope = profile::Scope::begin().unwrap();
            panic!("scope cleanup");
        })
        .is_err()
    );
    let scope = profile::Scope::begin().unwrap();
    profile::submitted("invalid", [0; 4], 1.0, 2.0, true, 0.0, 0.0);
    let report = scope.finish();
    assert_eq!(report.operations[0].gpu_ms, None);
    assert_eq!(report.operations[0].invalid_gpu_timestamps, 1);
    assert_eq!(linear.forward(&[1.0; 8]).unwrap(), expected);
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 3e-5 + 2e-5 * e.abs(), "{a} != {e}");
    }
}

#[test]
fn router_bias_selects_but_does_not_weight() {
    let (ids, weights) = route(&[0.0, 2.0, -2.0], &[0.0, -100.0, 100.0], 2, 1.0, 1.5).unwrap();
    assert_eq!(ids, [0, 2]);
    let a = 2.0f32.ln().sqrt();
    let b = (-2.0f32).exp().ln_1p().sqrt();
    close(&weights, &[a / (a + b) * 1.5, b / (a + b) * 1.5]);
    assert!(route(&[f32::NAN], &[0.0], 1, 1.0, 1.0).is_err());
}

#[test]
fn router_exact_bias_absorption_ties_choose_lower_expert_ids() {
    let logits = [-34.0f32, -32.0, -30.0, -28.0];
    let scores: Vec<_> = logits.iter().map(|v| v.exp().ln_1p().sqrt()).collect();
    assert!(scores.windows(2).all(|s| s[0] < s[1]));
    assert!(scores.iter().all(|s| s + 32.0 == 32.0));
    let (ids, weights) = route(&logits, &[32.0; 4], 2, 1.0, 1.5).unwrap();
    assert_eq!(ids, [0, 1]);
    let total = scores[0] + scores[1] + 1e-20;
    assert_eq!(weights, [scores[0] / total * 1.5, scores[1] / total * 1.5]);
}

#[test]
fn engram_injection_normalizes_each_stream_and_preserves_gate_sign() {
    let mut x = vec![1.0, 2.0, -1.0, -2.0, 3.0, 6.0, -3.0, -6.0];
    let original = x.clone();
    let kv = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.5, -0.25];
    inject_engram(&mut x, &kv, &[1.0; 8], &[1.0; 8], 1e-20).unwrap();
    let positive = 1.0 / (1.0 + (-2.0f32.sqrt().sqrt()).exp());
    for s in 0..4 {
        let gate = if s % 2 == 0 { positive } else { 1.0 - positive };
        close(
            &x[s * 2..s * 2 + 2],
            &[
                original[s * 2] + gate * 0.5,
                original[s * 2 + 1] - gate * 0.25,
            ],
        );
    }
}

#[test]
fn compression_interns_in_id_order_and_falls_back_on_partial_utf8() {
    let decoded = [" The", "THE", "\u{fffd}", "", "the"];
    let (map, size) = build_compressed_token_map(
        5,
        |i| Ok(decoded[i as usize].into()),
        |i| Ok(format!("raw-{i}")),
        |s| Ok(s.trim().to_lowercase()),
    )
    .unwrap();
    assert_eq!(map, [0, 0, 1, 2, 0]);
    assert_eq!(size, 3);
}

fn fixture() -> BTreeMap<String, Vec<f32>> {
    include_str!("../operators/old_native_f32.json")
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(": ")?;
            let value = value
                .strip_prefix('[')?
                .trim_end_matches(',')
                .strip_suffix(']')?;
            let values = if value.is_empty() {
                Vec::new()
            } else {
                value
                    .split(',')
                    .map(|n| n.trim().parse().unwrap())
                    .collect()
            };
            Some((key.trim_matches('"').to_owned(), values))
        })
        .collect()
}

fn fixture_tensors(
    ctx: &MetalContext,
    data: &BTreeMap<String, Vec<f32>>,
) -> BTreeMap<String, MetalTensor> {
    data.iter()
        .filter_map(|(key, values)| {
            let key = key.strip_prefix("w.")?;
            if key.ends_with("frequencies") {
                return None;
            }
            let (_, name) = key.split_once('.')?;
            let input = if name.ends_with("_fn") {
                Some(32)
            } else if name.contains("norm")
                || name.ends_with("attn_sink")
                || !name.ends_with("weight")
            {
                None
            } else if name.ends_with("wq_a.weight") {
                Some(8)
            } else if name.ends_with("wq_b.weight") {
                Some(6)
            } else if name.ends_with("wo_a.weight") {
                Some(4)
            } else if name.ends_with("wo_b.weight") {
                Some(6)
            } else if name.ends_with("indexer.wk.weight") {
                Some(4)
            } else if name.ends_with("expert.w2.weight") {
                Some(10)
            } else {
                Some(8)
            };
            let shape = match input {
                Some(i) => vec![i as u64, (values.len() / i) as u64],
                None => vec![values.len() as u64],
            };
            Some((format!("layers.{key}"), upload(ctx, values, shape).unwrap()))
        })
        .collect()
}

#[test]
fn resident_metal_projections_match_old_source_all_40_layers() {
    let ctx = MetalContext::new().expect("Metal required for resident forward fixture");
    let data = fixture();
    let tensors = fixture_tensors(&ctx, &data);
    let b = bind::Binder {
        ctx: &ctx,
        weights: &tensors,
    };
    let c = Config {
        hidden: 8,
        heads: 2,
        head_dim: 4,
        rope_dim: 4,
        query_rank: 6,
        output_rank: 3,
        groups: 2,
        index_heads: 2,
        index_dim: 4,
        index_topk: 1,
        window: 3,
        candidate_block: 2,
        candidate_topk: 1,
        norm_eps: 1e-6,
        original_context: 16,
        rope_factor: 4.0,
        beta_fast: 4.0,
        ..Config::default()
    };
    let attention: Vec<_> = (0..40).map(|l| b.attention(l, &c).unwrap()).collect();
    let hc: Vec<_> = (0..40)
        .map(|l| [b.hc(l, "attn", &c).unwrap(), b.hc(l, "ffn", &c).unwrap()])
        .collect();
    let mut state = attention::Session::default();
    for pos in 0..7 {
        let mut x = data[&format!("p.{pos}.initial")].clone();
        let mut pre = vec![1.0, 0.0, 0.0, 0.0];
        for l in 0..40 {
            let p = format!("p.{pos}.l.{l}");
            let (mix, input) = hc[l][0].prepare(&x, &pre, &c).unwrap();
            close(&input, &data[&format!("{p}.attn_input")]);
            let out = state.attention(l, pos, &input, &attention[l]).unwrap();
            close(&out, &data[&format!("{p}.attention")]);
            let snapshot = state.cache_snapshot(l).unwrap();
            assert_eq!(snapshot.next_position, pos + 1);
            close(&snapshot.window.concat(), &data[&format!("{p}.window")]);
            if [2, 8, 14, 20].contains(&l) {
                assert_eq!(snapshot.published_owner, Some((l, pos)));
                for (name, actual) in [
                    ("compressed", snapshot.compressed),
                    ("keys", snapshot.keys),
                    ("pending_values", snapshot.pending_values),
                    ("pending_scores", snapshot.pending_scores),
                ] {
                    close(&actual.concat(), &data[&format!("{p}.{name}")]);
                }
            }
            if l >= 2 {
                let (_, stamp, indices) = snapshot.selection.unwrap();
                assert_eq!(stamp, pos);
                close(
                    &indices.iter().map(|&i| i as f32).collect::<Vec<_>>(),
                    &data[&format!("{p}.selection")],
                );
            }
            if l >= 20 {
                let (stamp, mask) = snapshot.candidates.unwrap();
                assert_eq!(stamp, pos);
                close(
                    &mask.iter().map(|&v| u8::from(v) as f32).collect::<Vec<_>>(),
                    &data[&format!("{p}.candidates")],
                );
            }
            x = op::merge(&out, &x, &mix).unwrap();
            pre = mix.pre;
            let (mix, input) = hc[l][1].prepare(&x, &pre, &c).unwrap();
            let linear = |part: &str, i, o| {
                b.linear(&format!("layers.{l}.ffn.expert.{part}.weight"), i, o)
                    .unwrap()
            };
            let act = op::clamped_swiglu(
                &linear("w1", 8, 10).forward(&input).unwrap(),
                &linear("w3", 8, 10).forward(&input).unwrap(),
                0.35,
            )
            .unwrap();
            let out = linear("w2", 10, 8).forward(&act).unwrap();
            x = op::merge(&out, &x, &mix).unwrap();
            pre = mix.pre;
            close(&x, &data[&format!("{p}.hidden")]);
            close(&pre, &data[&format!("{p}.pre")]);
        }
        close(
            &op::collapse(&x, &pre).unwrap(),
            &data[&format!("p.{pos}.head_input")],
        );
    }
}

#[test]
fn six_native_experts_plus_shared_match_cpu_f32() {
    let ctx = MetalContext::new().expect("Metal required for native expert fixture");
    let c = Config {
        hidden: 32,
        intermediate: 32,
        experts: 8,
        topk: 6,
        clamp: 0.35,
        ..Config::default()
    };
    let values = [
        0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let mut dense_banks = Vec::new();
    let mut banks = Vec::new();
    for seed in 0..3 {
        let mut bytes = Vec::new();
        let mut dense = Vec::new();
        for row in 0..8 * 32 {
            bytes.push(123);
            let codes: Vec<_> = (0..32)
                .map(|j| ((row * 3 + j * 7 + seed * 5) % 16) as u8)
                .collect();
            for j in 0..16 {
                bytes.push(codes[j] | codes[j + 16] << 4);
            }
            dense.extend(codes.iter().map(|&code| values[code as usize] / 16.0));
        }
        banks
            .push(MetalTensor::from_bytes(&ctx, &bytes, vec![32, 32, 8], GgmlType::MXFP4).unwrap());
        dense_banks.push(dense);
    }
    let linear = |input, output, phase| {
        let values: Vec<_> = (0..input * output)
            .map(|i| (((i * 7 + phase) % 31) as f32 - 15.0) / 70.0)
            .collect();
        Linear {
            native_control: None,
            ctx: &ctx,
            tensor: upload(&ctx, &values, vec![input as u64, output as u64]).unwrap(),
            input,
            output,
        }
    };
    let data = fixture();
    let tensors = fixture_tensors(&ctx, &data);
    let b = bind::Binder {
        ctx: &ctx,
        weights: &tensors,
    };
    let tiny = Config {
        hidden: 8,
        heads: 2,
        head_dim: 4,
        rope_dim: 4,
        query_rank: 6,
        output_rank: 3,
        groups: 2,
        ..Config::default()
    };
    let layer = Layer {
        attention: b.attention(0, &tiny).unwrap(),
        attn_hc: b.hc(0, "attn", &tiny).unwrap(),
        ffn_hc: b.hc(0, "ffn", &tiny).unwrap(),
        router: linear(32, 8, 9),
        bias: vec![0.2, 0.0, -0.8, 0.1, 1.4, 0.3, -0.2, 0.7],
        shared: [linear(32, 32, 2), linear(32, 32, 5), linear(32, 32, 11)],
        banks: [&banks[0], &banks[1], &banks[2]],
        engram: None,
    };
    let x: Vec<_> = (0..32).map(|i| (i as f32 - 16.0) / 8.0).collect();
    let cpu_linear = |l: &Linear<'_>, x: &[f32]| -> Vec<f32> {
        f32_slice(&l.tensor)
            .unwrap()
            .chunks_exact(l.input)
            .map(|row| op::dot(row, x))
            .collect()
    };
    let (ids, weights) = route(&cpu_linear(&layer.router, &x), &layer.bias, 6, 1.0, 1.5).unwrap();
    let mut expected = vec![0.0; 32];
    for (&id, &weight) in ids.iter().zip(&weights) {
        let projection = |part: usize, x: &[f32]| -> Vec<f32> {
            dense_banks[part][id as usize * 1024..(id as usize + 1) * 1024]
                .chunks_exact(32)
                .map(|row| op::dot(row, x))
                .collect()
        };
        let mut act = op::clamped_swiglu(&projection(0, &x), &projection(1, &x), c.clamp).unwrap();
        act.iter_mut().for_each(|v| *v *= weight);
        for (e, v) in expected.iter_mut().zip(projection(2, &act)) {
            *e += v;
        }
    }
    let shared = op::clamped_swiglu(
        &cpu_linear(&layer.shared[0], &x),
        &cpu_linear(&layer.shared[1], &x),
        c.clamp,
    )
    .unwrap();
    for (e, v) in expected
        .iter_mut()
        .zip(cpu_linear(&layer.shared[2], &shared))
    {
        *e += v;
    }
    let legacy = Forward::moe(&ctx, &c, &layer, &x).unwrap();
    close(&legacy, &expected);
}
