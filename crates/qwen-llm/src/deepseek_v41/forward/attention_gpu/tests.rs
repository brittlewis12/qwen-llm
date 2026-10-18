use super::*;
use std::collections::BTreeMap;

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() <= 2e-5 + 4e-6 * b.abs(), "{i}: {a} != {b}");
    }
}

fn execute(ctx: &MetalContext, f: impl FnOnce(&KernelEncoder)) {
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    f(&enc);
    enc.end();
    metal::commit_and_wait(&command).unwrap();
}

fn ids(t: &MetalTensor, n: usize) -> Vec<usize> {
    assert!(n <= t.n_elements() as usize);
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<u32>(),
            n,
        )
    }
    .iter()
    .map(|&i| i as usize)
    .collect()
}

fn fixture() -> BTreeMap<String, Vec<f32>> {
    let value: serde_json::Value =
        serde_json::from_str(include_str!("../../operators/old_native_f32.json")).unwrap();
    value
        .as_object()
        .unwrap()
        .iter()
        .filter_map(|(key, value)| {
            value.as_array().map(|array| {
                (
                    key.clone(),
                    array.iter().map(|v| v.as_f64().unwrap() as f32).collect(),
                )
            })
        })
        .collect()
}

fn cpu_weights(data: &BTreeMap<String, Vec<f32>>, layer: usize) -> op::AttentionWeights {
    let array = |name: &str| data[&format!("w.{layer}.{name}")].clone();
    let linear = |name: &str, input, output| op::Linear {
        input,
        output,
        weight: array(&format!("{name}.weight")),
    };
    let owns = [2, 8, 14, 20].contains(&layer);
    op::AttentionWeights {
        query_a: linear("attn.wq_a", 8, 6),
        query_norm: array("attn.q_norm.weight"),
        query_b: linear("attn.wq_b", 6, 8),
        kv: linear("attn.wkv", 8, 4),
        kv_norm: array("attn.kv_norm.weight"),
        sink: array("attn.attn_sink"),
        output_a: array("attn.wo_a.weight")
            .chunks_exact(12)
            .map(|w| op::Linear {
                input: 4,
                output: 3,
                weight: w.to_vec(),
            })
            .collect(),
        output_b: linear("attn.wo_b", 6, 8),
        compressor: owns.then(|| op::CompressorWeights {
            ratio: if layer < 20 { 2 } else { 1 },
            kv: linear("attn.compressor.wkv", 8, 4),
            gate: (layer < 20).then(|| linear("attn.compressor.wgate", 8, 4)),
            norm: array("attn.compressor.norm.weight"),
        }),
        index: [2, 8, 14, 20, 24, 28, 32, 36]
            .contains(&layer)
            .then(|| op::IndexWeights {
                query: linear("attn.indexer.wq_b", 6, 8),
                weights: linear("attn.indexer.weights_proj", 8, 2),
                key: owns.then(|| linear("attn.indexer.wk", 4, 4)),
                key_norm: if owns {
                    array("attn.indexer.k_norm.weight")
                } else {
                    vec![]
                },
                heads: 2,
                dim: 4,
                topk: 1,
            }),
        frequencies: array("frequencies"),
        head_dim: 4,
        window: 3,
        eps: 1e-6,
        candidate_block_size: 2,
        candidate_topk_blocks: 1,
    }
}

fn resident<'a>(ctx: &'a MetalContext, w: &op::AttentionWeights) -> Weights<'a> {
    let p = |w: &op::Linear| Linear {
        ctx,
        input: w.input,
        output: w.output,
        tensor: upload(ctx, &w.weight, vec![w.input as u64, w.output as u64]).unwrap(),
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

#[test]
fn pinned_seven_positions_forty_layers_outputs_caches_and_ids() {
    let ctx = MetalContext::new().unwrap();
    let data = fixture();
    let cpu: Vec<_> = (0..40).map(|id| cpu_weights(&data, id)).collect();
    let gpu: Vec<_> = cpu.iter().map(|w| resident(&ctx, w)).collect();
    let mut c = super::super::toy::config();
    c.hidden = 8;
    let mut workspace = Workspace::new(&ctx, &c, 7, &gpu.iter().collect::<Vec<_>>()).unwrap();
    let mut oracle = op::Session::default();
    for pos in 0..7 {
        for id in 0..40 {
            let prefix = format!("p.{pos}.l.{id}.");
            let x = &data[&(prefix.clone() + "attn_input")];
            let input = upload(&ctx, x, vec![8]).unwrap();
            execute(&ctx, |enc| {
                workspace
                    .encode(&ctx, enc, id, pos, &gpu[id], &input)
                    .unwrap();
            });
            let expected = oracle.attention(id, pos, x, &cpu[id]).unwrap();
            close(f32_slice(&workspace.output).unwrap(), &expected);
            close(&expected, &data[&(prefix.clone() + "attention")]);
            let snapshot = oracle.cache_snapshot(id).unwrap();
            let raw = f32_slice(&workspace.raw).unwrap();
            let count = (pos + 1).min(c.window);
            for j in 0..count {
                let start = (id * c.window + (pos + 1 - count + j) % c.window) * c.head_dim;
                close(&raw[start..start + c.head_dim], &snapshot.window[j]);
            }
            if let Some(owner) = [2, 8, 14, 20].iter().position(|&owner| owner == id) {
                assert_eq!(snapshot.published_owner, Some((id, pos)));
                let cache = &workspace.owners[owner];
                for (tensor, rows) in [
                    (&cache.compressed, &snapshot.compressed),
                    (&cache.keys, &snapshot.keys),
                    (&cache.pending_value, &snapshot.pending_values),
                    (&cache.pending_gate, &snapshot.pending_scores),
                ] {
                    let flat: Vec<_> = rows.iter().flatten().copied().collect();
                    close(&f32_slice(tensor).unwrap()[..flat.len()], &flat);
                }
            }
            if id >= 2 {
                let (_, _, selection) = snapshot.selection.unwrap();
                assert_eq!(
                    ids(&workspace.selected, selection.len()),
                    selection,
                    "{prefix}"
                );
            }
            if id >= 20 {
                let (_, mask) = snapshot.candidates.unwrap();
                assert_eq!(
                    (0..mask.len())
                        .map(|j| ids(
                            &workspace.candidates,
                            mask.len().div_ceil(c.candidate_block)
                        )[j / c.candidate_block])
                        .collect::<Vec<_>>(),
                    mask.iter().map(|&b| b as usize).collect::<Vec<_>>()
                );
            }
        }
    }
    assert_eq!((workspace.next_position, workspace.next_layer), (7, 0));
}

#[test]
fn stable_gpu_selection_long_prefix_and_candidate_boundary() {
    let ctx = MetalContext::new().unwrap();
    let weights = super::super::toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let mut c = model.config.clone();
    c.index_topk = 512;
    c.candidate_block = 8;
    c.candidate_topk = 2048;
    let mut bound: Vec<_> = model.layers.iter().map(|l| l.attention.clone()).collect();
    for w in &mut bound {
        w.candidate_block_size = 8;
        w.candidate_topk_blocks = 2048;
        if let Some(i) = &mut w.index {
            i.topk = 512;
        }
    }
    let workspace = Workspace::new(&ctx, &c, 16385, &bound.iter().collect::<Vec<_>>()).unwrap();
    for (n, keep_blocks) in [
        (513usize, 1usize),
        (513, 2048),
        (10000, 2048),
        (16384, 2048),
        (16385, 2048),
    ] {
        let scores: Vec<_> = (0..n).map(|i| ((i * 37) % 113) as f32 - 56.0).collect();
        let input = upload(&ctx, &scores, vec![n as u64]).unwrap();
        let blocks = n.div_ceil(8);
        execute(&ctx, |enc| {
            pipeline(&ctx, enc, "blocks").unwrap();
            enc.set_bytes(0, &[n as u32, 8u32]);
            enc.set_tensor(1, &input);
            enc.set_tensor(2, &workspace.block_scores);
            dispatch(enc, blocks);
            workspace
                .sort(&ctx, enc, &workspace.block_scores, blocks, false)
                .unwrap();
            pipeline(&ctx, enc, "keep").unwrap();
            enc.set_bytes(0, &[blocks as u32, blocks.min(keep_blocks) as u32]);
            enc.set_tensor(1, &workspace.sort_ids);
            enc.set_tensor(2, &workspace.candidates);
            dispatch(enc, blocks);
            workspace.sort(&ctx, enc, &input, n, true).unwrap();
            pipeline(&ctx, enc, "chronological").unwrap();
            enc.set_bytes(0, &512u32);
            enc.set_tensor(1, &workspace.sort_ids);
            enc.set_tensor(2, &workspace.selected);
            dispatch(enc, 512);
        });
        let mask = op::candidate_blocks(&scores, 8, keep_blocks).unwrap();
        let actual = ids(&workspace.candidates, blocks);
        for j in 0..n {
            assert_eq!(actual[j / 8], mask[j] as usize);
        }
        assert_eq!(
            mask.iter().filter(|&&b| !b).count(),
            if keep_blocks == 1 {
                512
            } else if n == 16385 {
                8
            } else {
                0
            }
        );
        let masked: Vec<_> = scores
            .iter()
            .zip(mask)
            .map(|(&s, m)| if m { s } else { f32::NEG_INFINITY })
            .collect();
        assert_eq!(ids(&workspace.selected, 512), op::topk(&masked, 512));
    }
    let input = upload(
        &ctx,
        &[f32::NAN, 1.0, -0.0, 0.0, f32::NEG_INFINITY],
        vec![5],
    )
    .unwrap();
    execute(&ctx, |enc| {
        workspace.sort(&ctx, enc, &input, 5, false).unwrap()
    });
    assert_eq!(ids(&workspace.sort_ids, 5), [1, 3, 2, 0, 4]);
    assert_eq!(ids(workspace.status(), 1), [1]);
}

#[test]
fn admission_and_order_bounds() {
    assert!(bytes(&Config::default(), 0).is_err());
    assert!(bytes(&Config::default(), usize::MAX).is_err());
    let mut c = Config::default();
    c.head_dim = usize::MAX;
    assert!(bytes(&c, 10).is_err());
    let ctx = MetalContext::new().unwrap();
    let weights = super::super::toy::Weights::new(&ctx);
    let model = weights.model(&ctx);
    let bound: Vec<_> = model.layers.iter().map(|l| &l.attention).collect();
    let mut workspace = Workspace::new(&ctx, &model.config, 1, &bound).unwrap();
    let input = upload(&ctx, &[0.0; 32], vec![32]).unwrap();
    execute(&ctx, |enc| {
        assert!(workspace.encode(&ctx, enc, 1, 0, bound[1], &input).is_err());
        assert!(workspace.encode(&ctx, enc, 0, 1, bound[0], &input).is_err());
        let bad = input.view_subrange(0, vec![31]);
        assert!(workspace.encode(&ctx, enc, 0, 0, bound[0], &bad).is_err());
        let mut bad = input.clone();
        bad.offset = bad.buffer.length() as u64;
        assert!(workspace.encode(&ctx, enc, 0, 0, bound[0], &bad).is_err());
        let mut bad_weights = bound[0].clone();
        bad_weights.query_a.tensor.offset = bad_weights.query_a.tensor.buffer.length() as u64;
        assert!(
            workspace
                .encode(&ctx, enc, 0, 0, &bad_weights, &input)
                .is_err()
        );
    });
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin_concurrent(&command);
    assert!(
        workspace
            .encode(&ctx, &enc, 0, 0, bound[0], &input)
            .is_err()
    );
    enc.end();
    assert_eq!((workspace.next_position, workspace.next_layer), (0, 0));
}

#[test]
fn native_512_dim_64_heads_sink_rope_and_invalid_ids() {
    let ctx = MetalContext::new().unwrap();
    let (dim, heads, window, k) = (512usize, 64usize, 128usize, 512usize);
    let make = |n, seed| {
        (0..n)
            .map(|i| (((i * seed + 13) % 101) as f32 - 50.0) / 103.0)
            .collect::<Vec<_>>()
    };
    let q = make(heads * dim, 37);
    let rows: Vec<_> = make((window + k) * dim, 19)
        .chunks_exact(dim)
        .map(|s| s.to_vec())
        .collect();
    let sinks = make(heads, 23);
    let frequencies = op::rotary_frequencies(64, 160000.0, 65536, 16.0, 32.0, 1.0).unwrap();
    let query = upload(&ctx, &q, vec![(heads * dim) as u64]).unwrap();
    let raw = upload(&ctx, &rows[..window].concat(), vec![(window * dim) as u64]).unwrap();
    let compressed = upload(&ctx, &rows[window..].concat(), vec![(k * dim) as u64]).unwrap();
    let sink = upload(&ctx, &sinks, vec![heads as u64]).unwrap();
    let freq = upload(&ctx, &frequencies, vec![32]).unwrap();
    let scores = tensor(&ctx, heads * (window + k), GgmlType::F32).unwrap();
    let output = tensor(&ctx, heads * dim, GgmlType::F32).unwrap();
    let status = tensor(&ctx, 1, GgmlType::I32).unwrap();
    let selection = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&(0..k as u32).collect::<Vec<_>>()),
        vec![k as u64],
        GgmlType::I32,
    )
    .unwrap();
    let encode_sink = |enc: &KernelEncoder, length| {
        pipeline(&ctx, enc, "sink").unwrap();
        enc.set_bytes(0, &[dim as u32, window as u32, 0, window as u32]);
        enc.set_bytes(1, &[k as u32, (window + k) as u32]);
        for (slot, t) in [
            (2, &query),
            (3, &raw),
            (4, &compressed),
            (5, &selection),
            (6, &sink),
            (7, &scores),
            (8, &output),
            (10, &status),
        ] {
            enc.set_tensor(slot, t);
        }
        enc.set_bytes(9, &length);
        dispatch(enc, heads);
    };
    execute(&ctx, |enc| {
        rope(&ctx, enc, &query, &freq, dim, heads, 32, 16384, false).unwrap();
        encode_sink(enc, k as u32);
        rope(&ctx, enc, &output, &freq, dim, heads, 32, 16384, true).unwrap();
    });
    let mut expected = Vec::new();
    for (h, query) in q.chunks_exact(dim).enumerate() {
        let mut query = query.to_vec();
        op::rotary(&mut query, &frequencies, 16384, false).unwrap();
        let mut out = op::sparse_attention(
            &query,
            &rows,
            &(0..window + k).collect::<Vec<_>>(),
            sinks[h],
        )
        .unwrap();
        op::rotary(&mut out, &frequencies, 16384, true).unwrap();
        expected.extend(out);
    }
    close(f32_slice(&output).unwrap(), &expected);
    assert_eq!(ids(&status, 1), [0]);
    execute(&ctx, |enc| encode_sink(enc, (k - 1) as u32));
    assert_eq!(ids(&status, 1), [2]);
    assert!(f32_slice(&output).unwrap().iter().all(|&x| x == 0.0));
}

#[test]
fn forty_layer_seven_token_chain_one_command() {
    let ctx = MetalContext::new().unwrap();
    let data = fixture();
    let cpu: Vec<_> = (0..40).map(|id| cpu_weights(&data, id)).collect();
    let gpu: Vec<_> = cpu.iter().map(|w| resident(&ctx, w)).collect();
    let mut c = super::super::toy::config();
    c.hidden = 8;
    let mut workspace = Workspace::new(&ctx, &c, 7, &gpu.iter().collect::<Vec<_>>()).unwrap();
    let inputs: Vec<_> = (0..7)
        .map(|p| upload(&ctx, &data[&format!("p.{p}.l.0.attn_input")], vec![8]).unwrap())
        .collect();
    let outputs: Vec<_> = (0..280)
        .map(|_| tensor(&ctx, 8, GgmlType::F32).unwrap())
        .collect();
    execute(&ctx, |enc| {
        for p in 0..7 {
            let mut input = inputs[p].clone();
            for id in 0..40 {
                let output = workspace
                    .encode(&ctx, enc, id, p, &gpu[id], &input)
                    .unwrap();
                copy(&ctx, enc, output, &outputs[p * 40 + id], 8).unwrap();
                input = output.clone();
            }
        }
    });
    let mut oracle = op::Session::default();
    for p in 0..7 {
        let mut input = data[&format!("p.{p}.l.0.attn_input")].clone();
        for id in 0..40 {
            input = oracle.attention(id, p, &input, &cpu[id]).unwrap();
            close(f32_slice(&outputs[p * 40 + id]).unwrap(), &input);
        }
    }
    assert_eq!(ids(workspace.status(), 1), [0]);
}

#[test]
fn native_index_32_heads_128_dim_signed_weights() {
    let ctx = MetalContext::new().unwrap();
    let (n, dim, heads) = (513usize, 128usize, 32usize);
    let make = |n, seed| {
        (0..n)
            .map(|i| (((i * seed) % 103) as f32 - 51.0) / 512.0)
            .collect::<Vec<_>>()
    };
    let query = make(heads * dim, 13);
    let keys = make(n * dim, 29);
    let weights = make(heads, 31);
    let q = upload(&ctx, &query, vec![(heads * dim) as u64]).unwrap();
    let k = upload(&ctx, &keys, vec![(n * dim) as u64]).unwrap();
    let w = upload(&ctx, &weights, vec![heads as u64]).unwrap();
    let output = tensor(&ctx, n, GgmlType::F32).unwrap();
    execute(&ctx, |enc| {
        pipeline(&ctx, enc, "index").unwrap();
        enc.set_bytes(0, &[n as u32, dim as u32, heads as u32, 0]);
        enc.set_tensor(1, &q);
        enc.set_tensor(2, &w);
        enc.set_tensor(3, &k);
        enc.set_tensor(4, &output);
        dispatch(enc, n);
    });
    let scale = (dim as f32).sqrt().recip() * (heads as f32).sqrt().recip();
    let expected: Vec<_> = keys
        .chunks_exact(dim)
        .map(|key| {
            query
                .chunks_exact(dim)
                .zip(&weights)
                .map(|(q, w)| op::dot(q, key).max(0.0) * (w * scale))
                .sum()
        })
        .collect();
    close(f32_slice(&output).unwrap(), &expected);
}
