//! Compact channel-symmetric recipe shared with toy_oracle.py, not a mock step.
use super::*;
use std::collections::BTreeMap;

pub(super) fn config() -> Config {
    Config {
        hidden: 32,
        intermediate: 32,
        vocab: 8,
        heads: 2,
        head_dim: 4,
        rope_dim: 4,
        query_rank: 6,
        output_rank: 3,
        groups: 2,
        experts: 8,
        topk: 6,
        index_heads: 2,
        index_dim: 4,
        index_topk: 1,
        window: 3,
        candidate_block: 2,
        candidate_topk: 1,
        norm_eps: 1e-6,
        clamp: 0.35,
        original_context: 16,
        rope_factor: 4.0,
        beta_fast: 4.0,
        ..Config::default()
    }
}

pub(super) struct Weights {
    dense: BTreeMap<String, MetalTensor>,
    banks: Vec<MetalTensor>,
}
impl Weights {
    pub(super) fn new(ctx: &MetalContext) -> Self {
        let mut dense = BTreeMap::new();
        for l in 0..40 {
            let mut put = |suffix: &str, shape: &[u64], value| {
                let values = vec![value; shape.iter().product::<u64>() as usize];
                dense.insert(
                    format!("layers.{l}.{suffix}"),
                    upload(ctx, &values, shape.to_vec()).unwrap(),
                );
            };
            for (suffix, i, o) in [
                ("wq_a", 32, 6),
                ("wq_b", 6, 8),
                ("wkv", 32, 4),
                ("wo_a", 4, 6),
                ("wo_b", 6, 32),
            ] {
                put(
                    &format!("attn.{suffix}.weight"),
                    &[i, o],
                    (l % 3 + 1) as f32 / 128.0,
                );
            }
            put("attn.q_norm.weight", &[6], 1.0);
            put("attn.kv_norm.weight", &[4], 1.0);
            put("attn.attn_sink", &[2], 0.0);
            for kind in ["attn", "ffn"] {
                put(&format!("{kind}_norm.weight"), &[32], 1.0);
                put(&format!("hc_{kind}_fn"), &[128, 24], 1.0 / 1024.0);
            }
            if [2, 8, 14, 20].contains(&l) {
                put(
                    "attn.compressor.wkv.weight",
                    &[32, 4],
                    (l % 3 + 1) as f32 / 128.0,
                );
                if l < 20 {
                    put(
                        "attn.compressor.wgate.weight",
                        &[32, 4],
                        (l % 3 + 1) as f32 / 128.0,
                    );
                }
                put("attn.compressor.norm.weight", &[4], 1.0);
                put(
                    "attn.indexer.wk.weight",
                    &[4, 4],
                    (l % 3 + 1) as f32 / 128.0,
                );
                put("attn.indexer.k_norm.weight", &[4], 1.0);
            }
            if [2, 8, 14, 20, 24, 28, 32, 36].contains(&l) {
                put(
                    "attn.indexer.wq_b.weight",
                    &[6, 8],
                    (l % 3 + 1) as f32 / 128.0,
                );
                put(
                    "attn.indexer.weights_proj.weight",
                    &[32, 2],
                    (l % 3 + 1) as f32 / 128.0,
                );
            }
            for part in ["w1", "w3", "w2"] {
                put(
                    &format!("ffn.shared_experts.{part}.weight"),
                    &[32, 32],
                    1.0 / 64.0,
                );
            }
            if [1, 14].contains(&l) {
                put("engram.wkv.weight", &[6144, 160], 1.0 / 8192.0);
                put("engram.q_weight", &[32, 4], 1.0);
                put("engram.k_weight", &[32, 4], 1.0);
            }
            for kind in ["attn", "ffn"] {
                let base: Vec<_> = (0..24).map(|i| ((i % 7) as f32 - 3.0) / 16.0).collect();
                dense.insert(
                    format!("layers.{l}.hc_{kind}_base"),
                    upload(ctx, &base, vec![24]).unwrap(),
                );
                dense.insert(
                    format!("layers.{l}.hc_{kind}_scale"),
                    upload(ctx, &[0.25, 0.5, 0.75], vec![3]).unwrap(),
                );
            }
            let router: Vec<_> = (0..8)
                .flat_map(|i| vec![(i as f32 - 3.0) / 64.0; 32])
                .collect();
            let bias: Vec<_> = (0..8).map(|i| i as f32 / 32.0).collect();
            dense.insert(
                format!("layers.{l}.ffn.gate.weight"),
                upload(ctx, &router, vec![32, 8]).unwrap(),
            );
            dense.insert(
                format!("layers.{l}.ffn.gate.bias"),
                upload(ctx, &bias, vec![8]).unwrap(),
            );
        }
        for (name, shift, denom) in [("embed.weight", 2.0, 8.0), ("head.weight", 3.0, 32.0)] {
            let values: Vec<_> = (0..8)
                .flat_map(|i| vec![(i as f32 - shift) / denom; 32])
                .collect();
            dense.insert(name.into(), upload(ctx, &values, vec![32, 8]).unwrap());
        }
        let banks = [1, 3, 2]
            .into_iter()
            .map(|shift| {
                let mut bytes = Vec::new();
                for expert in 0..8 {
                    let code = (expert + shift) % 8;
                    for _ in 0..32 {
                        bytes.push(122);
                        bytes.extend([code | code << 4; 16]);
                    }
                }
                MetalTensor::from_bytes(ctx, &bytes, vec![32, 32, 8], GgmlType::MXFP4).unwrap()
            })
            .collect();
        Self { dense, banks }
    }

    pub(super) fn model<'a>(&'a self, ctx: &'a MetalContext) -> Forward<'a> {
        let c = config();
        let b = bind::Binder {
            ctx,
            weights: &self.dense,
        };
        let layers = (0..40)
            .map(|l| {
                let p = format!("layers.{l}");
                Layer {
                    attention: b.attention(l, &c).unwrap(),
                    attn_hc: b.hc(l, "attn", &c).unwrap(),
                    ffn_hc: b.hc(l, "ffn", &c).unwrap(),
                    router: b.linear(&format!("{p}.ffn.gate.weight"), 32, 8).unwrap(),
                    bias: f32_slice(&self.dense[&format!("{p}.ffn.gate.bias")])
                        .unwrap()
                        .to_vec(),
                    shared: ["w1", "w3", "w2"].map(|part| {
                        b.linear(&format!("{p}.ffn.shared_experts.{part}.weight"), 32, 32)
                            .unwrap()
                    }),
                    banks: [&self.banks[0], &self.banks[1], &self.banks[2]],
                    engram: [1, 14].contains(&l).then(|| Engram {
                        projection: b
                            .linear(&format!("{p}.engram.wkv.weight"), 6144, 160)
                            .unwrap(),
                        q: vec![1.0; 128],
                        k: vec![1.0; 128],
                    }),
                }
            })
            .collect();
        Forward {
            ctx,
            engram_rows: EngramRows::Toy(|layer, rows| {
                Ok(rows
                    .iter()
                    .flat_map(|row| {
                        vec![((row + layer as u64) % 17) as f32 / 128.0 - 8.0 / 128.0; 256]
                    })
                    .collect())
            }),
            config: c,
            token_map: vec![5, 3, 2, 7, 0, 1, 6, 4],
            pad: 2,
            layers,
            embedding: &self.dense["embed.weight"],
            norm: vec![1.0; 32],
            head: b.linear("head.weight", 32, 8).unwrap(),
            identity: Arc::new(()),
        }
    }
}

fn snapshot(state: &State) -> String {
    format!(
        "{:?}|{:?}|{}|{}",
        state.attention, state.addresses, state.max_context, state.poisoned
    )
}

#[test]
fn full_graph_python_parity_and_trace_is_observational() {
    let ctx = MetalContext::new().unwrap();
    let weights = Weights::new(&ctx);
    let model = weights.model(&ctx);
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("toy_expected.json")).unwrap();
    let mut traced = model.new_state(3).unwrap();
    let mut plain = model.new_state_with_expert_staging(3).unwrap();
    let close = |a: f32, e: f32| assert!((a - e).abs() <= 3e-5 + 3e-5 * e.abs(), "{a} != {e}");
    for (pos, token) in [0, 3, 1].into_iter().enumerate() {
        let mut count = 0;
        let mut callback = |trace: LayerTrace<'_>| {
            assert_eq!((trace.position, trace.layer), (pos, count));
            let row = &expected["traces"][pos * 40 + count];
            assert_eq!(trace.hidden.len(), 128);
            for s in 0..4 {
                for &value in &trace.hidden[s * 32..(s + 1) * 32] {
                    close(value, row[s].as_f64().unwrap() as f32);
                }
                close(trace.next_pre_mix[s], row[4 + s].as_f64().unwrap() as f32);
            }
            count += 1;
            Ok(())
        };
        let logits = model
            .step_with_trace(&mut traced, token, None, Some(&mut callback))
            .unwrap();
        assert_eq!(count, 40);
        let staged = model.step(&mut plain, token, None).unwrap();
        assert_eq!(
            logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            staged.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(snapshot(&traced), snapshot(&plain));
        assert_eq!(traced.position(), pos + 1);
        for (i, &value) in logits.iter().enumerate() {
            close(value, expected["logits"][pos][i].as_f64().unwrap() as f32);
        }
    }
    let before = snapshot(&plain);
    assert!(
        model
            .step(&mut plain, 0, None)
            .unwrap_err()
            .contains("maximum context")
    );
    assert_eq!(before, snapshot(&plain));
}

#[test]
fn trace_errors_and_mid_token_cancellation_poison_but_preflight_does_not() {
    let ctx = MetalContext::new().unwrap();
    let weights = Weights::new(&ctx);
    let model = weights.model(&ctx);
    let cancel = AtomicBool::new(true);
    let mut state = model.new_state(3).unwrap();
    let before = snapshot(&state);
    assert!(
        model
            .step(&mut state, 0, Some(&cancel))
            .unwrap_err()
            .contains("cancelled")
    );
    assert_eq!(before, snapshot(&state));
    assert!(model.step(&mut state, 8, None).is_err());
    assert_eq!(before, snapshot(&state));
    for at in [0, 14, 39] {
        let mut state = model.new_state(3).unwrap();
        cancel.store(false, Ordering::Relaxed);
        let mut count = 0;
        let mut callback = |trace: LayerTrace<'_>| {
            count += 1;
            if trace.layer == at {
                cancel.store(true, Ordering::Relaxed);
            }
            Ok(())
        };
        assert!(
            model
                .step_with_trace(&mut state, 0, Some(&cancel), Some(&mut callback))
                .unwrap_err()
                .contains("cancelled")
        );
        assert_eq!(count, at + 1);
        assert_eq!(state.position(), 0);
        assert!(state.is_poisoned());
        assert_eq!(state.addresses.history(), &[]);
        assert!(
            model
                .step(&mut state, 0, None)
                .unwrap_err()
                .contains("poisoned")
        );
    }
    let mut state = model.new_state(3).unwrap();
    model.step(&mut state, 0, None).unwrap();
    let addresses = state.addresses;
    let mut callback = |_: LayerTrace<'_>| Err("trace write failed".to_owned());
    assert_eq!(
        model
            .step_with_trace(&mut state, 3, None, Some(&mut callback))
            .unwrap_err(),
        "trace write failed"
    );
    assert!(state.is_poisoned());
    assert_eq!(state.addresses, addresses);
}
