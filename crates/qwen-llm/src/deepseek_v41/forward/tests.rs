use super::*;
use std::collections::BTreeMap;

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
    close(&Forward::moe(&ctx, &c, &layer, &x).unwrap(), &expected);
}
