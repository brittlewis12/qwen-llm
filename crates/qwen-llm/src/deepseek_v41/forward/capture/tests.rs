use super::*;

fn cpu(w: &op::AttentionWeights<Linear<'_>>) -> op::AttentionWeights {
    let p = |w: &Linear<'_>| op::Linear {
        input: w.input,
        output: w.output,
        weight: f32_slice(&w.tensor).unwrap().to_vec(),
    };
    op::AttentionWeights {
        query_a: p(&w.query_a),
        query_norm: w.query_norm.clone(),
        query_b: p(&w.query_b),
        kv: p(&w.kv),
        kv_norm: w.kv_norm.clone(),
        sink: w.sink.clone(),
        output_a: w.output_a.iter().map(p).collect(),
        output_b: p(&w.output_b),
        compressor: w.compressor.as_ref().map(|w| op::CompressorWeights {
            ratio: w.ratio,
            kv: p(&w.kv),
            gate: w.gate.as_ref().map(p),
            norm: w.norm.clone(),
        }),
        index: w.index.as_ref().map(|w| op::IndexWeights {
            query: p(&w.query),
            weights: p(&w.weights),
            key: w.key.as_ref().map(p),
            key_norm: w.key_norm.clone(),
            heads: w.heads,
            dim: w.dim,
            topk: w.topk,
        }),
        frequencies: w.frequencies.clone(),
        head_dim: w.head_dim,
        window: w.window,
        eps: w.eps,
        candidate_block_size: w.candidate_block_size,
        candidate_topk_blocks: w.candidate_topk_blocks,
    }
}

struct Replay {
    weights: Vec<op::AttentionWeights>,
    session: op::Session,
    calls: usize,
    checkpoints: usize,
}
impl Capture for Replay {
    fn wants_position(&self, p: usize) -> bool {
        p < 7
    }
    fn is_checkpoint(&self, p: usize) -> bool {
        [0, 1, 2, 5, 6].contains(&p)
    }
    fn layer(&mut self, t: AttentionCapture<'_>) -> Result<()> {
        assert_eq!(
            (t.position, t.layer),
            (self.calls / 8, LAYERS[self.calls % 8])
        );
        assert_eq!(t.input.len(), 32);
        let output =
            self.session
                .attention(t.layer, t.position, t.input, &self.weights[t.layer])?;
        assert_eq!(t.checkpoint.is_some(), self.is_checkpoint(t.position));
        if let Some(c) = t.checkpoint {
            self.checkpoints += 1;
            assert_eq!(c.output.len(), 32);
            for (&a, b) in c.output.iter().zip(output) {
                assert!(
                    (a - b).abs() < 3e-5 + 3e-5 * b.abs(),
                    "position {} layer {}: {a} != {b}",
                    t.position,
                    t.layer
                );
            }
            let snapshot = self.session.cache_snapshot(t.layer).unwrap();
            assert_eq!(snapshot.published_owner.unwrap(), (c.kv_owner, t.position));
            let (source, position, ids) = snapshot.selection.unwrap();
            assert_eq!((source, position), (c.index_source, t.position));
            assert_eq!(
                c.compressed_ids
                    .iter()
                    .map(|&i| i as usize)
                    .collect::<Vec<_>>(),
                ids
            );
            assert_eq!(
                ids.len(),
                ((t.position + 1) / if t.layer < 20 { 2 } else { 1 }).min(1)
            );
            if let Some(mask) = c.candidate_mask {
                let (_, expected) = snapshot.candidates.unwrap();
                assert_eq!(mask.len(), (t.position + 1).div_ceil(2));
                assert_eq!(mask.last(), Some(&1));
                assert_eq!(mask.iter().filter(|&&v| v != 0).count(), 1);
                for (i, &keep) in expected.iter().enumerate() {
                    assert_eq!(mask[i / 2] != 0, keep);
                }
            } else {
                assert!(t.layer < 20);
            }
        }
        self.calls += 1;
        Ok(())
    }
}

#[test]
fn capture_full_gpu_step_replays_cpu_phases_ids_and_pruning() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut replay = Replay {
        weights: model.layers.iter().map(|l| cpu(&l.attention)).collect(),
        session: op::Session::default(),
        calls: 0,
        checkpoints: 0,
    };
    let mut state = model.new_state_with_gpu_step(8).unwrap();
    let mut plain = model.new_state_with_gpu_step(8).unwrap();
    for token in [0, 3, 1, 4, 2, 5, 6, 7] {
        let logits = model
            .step_with_capture(&mut state, token, None, None, Some(&mut replay))
            .unwrap();
        assert_eq!(logits, model.step(&mut plain, token, None).unwrap());
    }
    assert_eq!((replay.calls, replay.checkpoints), (56, 40));
}

#[test]
fn batched_capture_replays_cpu_without_changing_logits() {
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let tokens = [0, 3, 1, 4, 2, 5, 6, 7];
    for chunk in [1, 2, 3, 7, 8] {
        let mut replay = Replay {
            weights: model.layers.iter().map(|l| cpu(&l.attention)).collect(),
            session: op::Session::default(),
            calls: 0,
            checkpoints: 0,
        };
        let mut state = model.new_state_with_gpu_step(tokens.len()).unwrap();
        let mut plain = model.new_state_with_gpu_step(tokens.len()).unwrap();
        for suffix in tokens.chunks(chunk) {
            let actual = model
                .prefill_with_capture(&mut state, suffix, None, &mut replay)
                .unwrap();
            assert_eq!(actual, model.prefill(&mut plain, suffix, None).unwrap());
        }
        assert_eq!((replay.calls, replay.checkpoints), (56, 40));
        assert_eq!(state.position(), tokens.len());
    }
}

#[test]
fn capture_requires_gpu_and_callback_errors_poison_state() {
    struct Fail;
    impl Capture for Fail {
        fn wants_position(&self, _: usize) -> bool {
            true
        }
        fn is_checkpoint(&self, _: usize) -> bool {
            false
        }
        fn layer(&mut self, _: AttentionCapture<'_>) -> Result<()> {
            Err("capture disk failure".into())
        }
    }
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut legacy = model.new_cpu_state(1).unwrap();
    assert!(
        model
            .step_with_capture(&mut legacy, 0, None, None, Some(&mut Fail))
            .unwrap_err()
            .contains("requires GPU")
    );
    assert!(!legacy.is_poisoned());
    let mut state = model.new_state_with_gpu_step(1).unwrap();
    assert!(
        model
            .step_with_capture(&mut state, 0, None, None, Some(&mut Fail))
            .unwrap_err()
            .contains("capture disk failure")
    );
    assert!(state.is_poisoned());
    assert_eq!(state.position(), 0);
    let mut batch = model.new_state_with_gpu_step(3).unwrap();
    assert!(
        model
            .prefill_with_capture(&mut batch, &[0, 1, 2], None, &mut Fail)
            .unwrap_err()
            .to_string()
            .contains("capture disk failure")
    );
    assert!(batch.is_poisoned());
    assert_eq!(batch.position(), 0);
}
