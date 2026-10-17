use super::*;
use std::collections::BTreeMap;

// The generator writes flat numeric arrays one per line. This intentionally
// needs no serde dependency, so rustc --test can run outside the parent build.
fn fixture() -> BTreeMap<String, Vec<f32>> {
    include_str!("old_native_f32.json")
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(": ")?;
            let array = value
                .strip_prefix('[')?
                .trim_end_matches(',')
                .strip_suffix(']')?;
            let values = if array.is_empty() {
                vec![]
            } else {
                array
                    .split(',')
                    .map(|n| n.trim().parse().unwrap())
                    .collect()
            };
            Some((key.trim_matches('"').to_owned(), values))
        })
        .collect()
}

struct Weights<'a> {
    data: &'a BTreeMap<String, Vec<f32>>,
    layer: usize,
}
impl Weights<'_> {
    fn array(&self, name: &str) -> Vec<f32> {
        self.data[&format!("w.{}.{}", self.layer, name)].clone()
    }
    fn linear(&self, name: &str, input: usize, output: usize) -> Linear {
        Linear {
            input,
            output,
            weight: self.array(&format!("{name}.weight")),
        }
    }
    fn attention(&self) -> AttentionWeights {
        let owns = [2, 8, 14, 20].contains(&self.layer);
        AttentionWeights {
            query_a: self.linear("attn.wq_a", 8, 6),
            query_norm: self.array("attn.q_norm.weight"),
            query_b: self.linear("attn.wq_b", 6, 8),
            kv: self.linear("attn.wkv", 8, 4),
            kv_norm: self.array("attn.kv_norm.weight"),
            sink: self.array("attn.attn_sink"),
            output_a: self
                .array("attn.wo_a.weight")
                .chunks_exact(12)
                .map(|g| Linear {
                    input: 4,
                    output: 3,
                    weight: g.to_vec(),
                })
                .collect(),
            output_b: self.linear("attn.wo_b", 6, 8),
            compressor: owns.then(|| CompressorWeights {
                ratio: if self.layer < 20 { 2 } else { 1 },
                kv: self.linear("attn.compressor.wkv", 8, 4),
                gate: (self.layer < 20).then(|| self.linear("attn.compressor.wgate", 8, 4)),
                norm: self.array("attn.compressor.norm.weight"),
            }),
            index: [2, 8, 14, 20, 24, 28, 32, 36]
                .contains(&self.layer)
                .then(|| IndexWeights {
                    query: self.linear("attn.indexer.wq_b", 6, 8),
                    weights: self.linear("attn.indexer.weights_proj", 8, 2),
                    key: owns.then(|| self.linear("attn.indexer.wk", 4, 4)),
                    key_norm: if owns {
                        self.array("attn.indexer.k_norm.weight")
                    } else {
                        vec![]
                    },
                    heads: 2,
                    dim: 4,
                    topk: 1,
                }),
            frequencies: rotary_frequencies(
                4,
                if self.layer < 2 { 10000.0 } else { 160000.0 },
                if self.layer < 2 { 0 } else { 16 },
                4.0,
                4.0,
                1.0,
            )
            .unwrap(),
            head_dim: 4,
            window: 3,
            eps: 1e-6,
            candidate_block_size: 2,
            candidate_topk_blocks: 1,
        }
    }
    fn hc(&self, kind: &str) -> HcWeights {
        HcWeights {
            projection: Linear {
                input: 32,
                output: 24,
                weight: self.array(&format!("hc_{kind}_fn")),
            },
            scale: self.array(&format!("hc_{kind}_scale")).try_into().unwrap(),
            base: self.array(&format!("hc_{kind}_base")),
            streams: 4,
            iterations: 20,
            norm_eps: 1e-6,
            hc_eps: 1e-6,
        }
    }
    fn expert(&self, x: &[f32]) -> Result<Vec<f32>> {
        let gate = self.linear("ffn.expert.w1", 8, 10).forward(x)?;
        let up = self.linear("ffn.expert.w3", 8, 10).forward(x)?;
        self.linear("ffn.expert.w2", 10, 8)
            .forward(&clamped_swiglu(&gate, &up, 0.35)?)
    }
}

#[derive(Default)]
struct Errors(BTreeMap<&'static str, f32>);
impl Errors {
    fn compare(&mut self, kind: &'static str, key: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{key}");
        for (&a, &e) in actual.iter().zip(expected) {
            let error = (a - e).abs();
            assert!(
                error <= 1e-5 + 2e-6 * e.abs(),
                "{key}: actual={a} expected={e} error={error}"
            );
            let max = self.0.entry(kind).or_default();
            *max = max.max(error);
        }
    }
}
fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

#[test]
fn old_source_attention_outputs_and_every_cache_transition() {
    let data = fixture();
    let mut session = Session::default();
    let mut errors = Errors::default();
    for pos in 0..7 {
        for layer in 0..40 {
            let w = Weights { data: &data, layer }.attention();
            errors.compare(
                "frequencies",
                "frequencies",
                &w.frequencies,
                &data[&format!("w.{layer}.frequencies")],
            );
            let prefix = format!("p.{pos}.l.{layer}.");
            let output = session
                .attention(layer, pos, &data[&(prefix.clone() + "attn_input")], &w)
                .unwrap();
            errors.compare(
                "attention",
                &prefix,
                &output,
                &data[&(prefix.clone() + "attention")],
            );
            let snapshot = session.cache_snapshot(layer).unwrap();
            assert_eq!(snapshot.next_position, pos + 1);
            errors.compare(
                "window",
                &prefix,
                &flatten(&snapshot.window),
                &data[&(prefix.clone() + "window")],
            );
            if [2, 8, 14, 20].contains(&layer) {
                // In particular pos=2, layer=2 follows owner=20 from pos=1.
                assert_eq!(snapshot.published_owner, Some((layer, pos)));
                for (name, actual) in [
                    ("compressed", &snapshot.compressed),
                    ("keys", &snapshot.keys),
                    ("pending_values", &snapshot.pending_values),
                    ("pending_scores", &snapshot.pending_scores),
                ] {
                    errors.compare(
                        name,
                        &prefix,
                        &flatten(actual),
                        &data[&(prefix.clone() + name)],
                    );
                }
            }
            if layer >= 2 {
                let (producer, stamp, indices) = snapshot.selection.unwrap();
                assert_eq!(stamp, pos);
                assert_eq!(
                    producer,
                    [2, 8, 14, 20, 24, 28, 32, 36]
                        .into_iter()
                        .filter(|&i| i <= layer)
                        .last()
                        .unwrap()
                );
                assert_eq!(
                    indices.iter().map(|&i| i as f32).collect::<Vec<_>>(),
                    data[&(prefix.clone() + "selection")],
                    "{prefix}selection"
                );
            }
            if layer >= 20 {
                let (stamp, mask) = snapshot.candidates.unwrap();
                assert_eq!(stamp, pos);
                assert_eq!(
                    mask.iter().map(|&v| u8::from(v) as f32).collect::<Vec<_>>(),
                    data[&(prefix.clone() + "candidates")],
                    "{prefix}candidates"
                );
            }
        }
    }
    println!(
        "old-source independent-input max absolute errors: {:?}",
        errors.0
    );
    assert_ne!(
        data["negative.correct_selection"],
        data["negative.stale_selection"]
    );
    let negative_error = crate::compare::assert_max_abs_diff_f32(
        "stale-owner negative control",
        &data["negative.correct_attention"],
        &data["negative.stale_attention"],
    );
    assert!(negative_error > 1e-3);
    println!("uncorrected old-source first-divergence attention error: {negative_error}");
}

#[test]
fn old_source_lagged_block_assembly_numerical_parity() {
    let data = fixture();
    let mut session = Session::default();
    let mut errors = Errors::default();
    errors.compare(
        "clamp",
        "clamp",
        &clamped_swiglu(&data["clamp.input"], &data["clamp.input"], 0.35).unwrap(),
        &data["clamp.output"],
    );
    for pos in 0..7 {
        let mut x = data[&format!("p.{pos}.initial")].clone();
        let mut pre = vec![1.0, 0.0, 0.0, 0.0];
        for layer in 0..40 {
            let weights = Weights { data: &data, layer };
            let prefix = format!("p.{pos}.l.{layer}.");
            let hc = weights.hc("attn");
            let mixes = hc.mixes(&x).unwrap();
            for (name, actual) in [
                ("attn_pre", &mixes.pre),
                ("attn_post", &mixes.post),
                ("attn_comb", &mixes.comb),
            ] {
                errors.compare(name, &prefix, actual, &data[&(prefix.clone() + name)]);
            }
            (x, pre) =
                lagged_sublayer(&x, &pre, &hc, &weights.array("attn_norm.weight"), |input| {
                    errors.compare(
                        "attn_input",
                        &prefix,
                        input,
                        &data[&(prefix.clone() + "attn_input")],
                    );
                    let output = session.attention(layer, pos, input, &weights.attention())?;
                    errors.compare(
                        "assembled_attention",
                        &prefix,
                        &output,
                        &data[&(prefix.clone() + "attention")],
                    );
                    Ok(output)
                })
                .unwrap();
            (x, pre) = lagged_sublayer(
                &x,
                &pre,
                &weights.hc("ffn"),
                &weights.array("ffn_norm.weight"),
                |input| weights.expert(input),
            )
            .unwrap();
            errors.compare("hidden", &prefix, &x, &data[&(prefix.clone() + "hidden")]);
            errors.compare("pre", &prefix, &pre, &data[&(prefix.clone() + "pre")]);
        }
        errors.compare(
            "head_input",
            "head",
            &collapse(&x, &pre).unwrap(),
            &data[&format!("p.{pos}.head_input")],
        );
    }
    println!("old-source assembled max absolute errors: {:?}", errors.0);
}

#[test]
fn rollback_after_owner_append_preserves_completed_and_pending_rows() {
    let data = fixture();
    for layer in [2, 20] {
        let w = Weights { data: &data, layer }.attention();
        let mut session = Session::default();
        let first = &data[&format!("p.0.l.{layer}.attn_input")];
        let second = &data[&format!("p.1.l.{layer}.attn_input")];
        session.attention(layer, 0, first, &w).unwrap();
        let before = format!("{:?}", session.cache_snapshot(layer));
        let mut bad = w.clone();
        bad.output_b.weight.clear();
        assert!(session.attention(layer, 1, second, &bad).is_err());
        assert_eq!(before, format!("{:?}", session.cache_snapshot(layer)));
        let output = session.attention(layer, 1, second, &w).unwrap();
        Errors::default().compare(
            "retry",
            "retry",
            &output,
            &data[&format!("p.1.l.{layer}.attention")],
        );
    }
}
