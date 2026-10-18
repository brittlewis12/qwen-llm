use super::*;
use std::collections::BTreeMap;

fn scheduled_weights<'a>(
    ctx: &'a MetalContext,
    c: &Config,
    schedule: &Schedule,
    id: usize,
    dtype: GgmlType,
) -> Weights<'a> {
    let linear = |input: usize, output: usize| {
        let mut bytes = Vec::new();
        for row in 0..output {
            for block in 0..input / 32 {
                if dtype == GgmlType::Q8_0 {
                    bytes.extend_from_slice(&half::f16::from_f32(0.015625).to_bits().to_le_bytes());
                }
                for j in 0..32 {
                    let q = ((row * 3 + block * 7 + j * 11 + id) % 31) as i8 - 15;
                    let value = q as f32 * 0.015625;
                    match dtype {
                        GgmlType::Q8_0 => bytes.push(q as u8),
                        GgmlType::BF16 => bytes.extend_from_slice(
                            &half::bf16::from_f32(value).to_bits().to_le_bytes(),
                        ),
                        GgmlType::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
                        _ => unreachable!(),
                    }
                }
            }
        }
        Linear {
            ctx,
            tensor: MetalTensor::from_bytes(ctx, &bytes, vec![input as u64, output as u64], dtype)
                .unwrap(),
            native_control: native_control(ctx, dtype).unwrap(),
            input,
            output,
        }
    };
    let owner = schedule.owns_kv(id);
    op::AttentionWeights {
        query_a: linear(c.hidden, c.query_rank),
        query_norm: vec![1.0; c.query_rank],
        query_b: linear(c.query_rank, c.heads * c.head_dim),
        kv: linear(c.hidden, c.head_dim),
        kv_norm: vec![1.0; c.head_dim],
        sink: vec![-0.25; c.heads],
        output_a: (0..c.groups)
            .map(|_| linear(c.heads * c.head_dim / c.groups, c.output_rank))
            .collect(),
        output_b: linear(c.groups * c.output_rank, c.hidden),
        compressor: owner.then(|| op::CompressorWeights {
            ratio: schedule.ratio(id),
            kv: linear(c.hidden, c.head_dim),
            gate: (schedule.ratio(id) == 2).then(|| linear(c.hidden, c.head_dim)),
            norm: vec![1.0; c.head_dim],
        }),
        index: schedule.produces_index(id).then(|| op::IndexWeights {
            query: linear(c.query_rank, c.index_heads * c.index_dim),
            weights: linear(c.hidden, c.index_heads),
            key: owner.then(|| linear(c.head_dim, c.index_dim)),
            key_norm: if owner {
                vec![1.0; c.index_dim]
            } else {
                vec![]
            },
            heads: c.index_heads,
            dim: c.index_dim,
            topk: c.index_topk,
        }),
        frequencies: vec![],
        head_dim: c.head_dim,
        window: c.window,
        eps: c.norm_eps,
        candidate_block_size: c.candidate_block,
        candidate_topk_blocks: c.candidate_topk,
    }
}

#[test]
fn scheduled_bf16_q8_attention_and_capture_match_f32() {
    let ctx = MetalContext::new().unwrap();
    let c = Config {
        hidden: 32,
        heads: 2,
        head_dim: 32,
        query_rank: 32,
        output_rank: 32,
        groups: 1,
        index_heads: 1,
        index_dim: 32,
        index_topk: 2,
        window: 2,
        rope_dim: 0,
        candidate_block: 2,
        candidate_topk: 1,
        norm_eps: 1e-6,
        ..Config::default()
    };
    let schedules = [
        Schedule {
            layers: 7,
            kv_source_layers: vec![1, 3],
            index_source_layers: vec![1, 3, 5],
            compression_ratios: vec![0, 2, 2, 1, 1, 1, 1],
            candidate_source_layer: Some(3),
        },
        Schedule {
            layers: 3,
            kv_source_layers: vec![],
            index_source_layers: vec![],
            compression_ratios: vec![0; 3],
            candidate_source_layer: None,
        },
    ];
    for schedule in schedules {
        let oracle: Vec<_> = (0..schedule.layers)
            .map(|id| scheduled_weights(&ctx, &c, &schedule, id, GgmlType::F32))
            .collect();
        for dtype in [GgmlType::BF16, GgmlType::Q8_0] {
            let weights: Vec<_> = (0..schedule.layers)
                .map(|id| scheduled_weights(&ctx, &c, &schedule, id, dtype))
                .collect();
            let mut actual = Workspace::new_with_schedule(
                &ctx,
                &c,
                5,
                &weights.iter().collect::<Vec<_>>(),
                &schedule,
            )
            .unwrap();
            let mut expected = Workspace::new_with_schedule(
                &ctx,
                &c,
                5,
                &oracle.iter().collect::<Vec<_>>(),
                &schedule,
            )
            .unwrap();
            assert_eq!(
                actual.raw.n_elements(),
                (schedule.layers * c.window * c.head_dim) as u64
            );
            assert_eq!(actual.owners.len(), schedule.kv_source_layers.len());
            for (cache, &id) in actual.owners.iter().zip(&schedule.kv_source_layers) {
                assert_eq!(
                    cache.compressed.n_elements(),
                    (5usize.div_ceil(schedule.ratio(id)) * c.head_dim) as u64
                );
            }
            for position in 0..5 {
                let x: Vec<_> = (0..c.hidden)
                    .map(|i| (i as f32 - 15.0 + position as f32) / 31.0)
                    .collect();
                let input = upload(&ctx, &x, vec![c.hidden as u64]).unwrap();
                for id in 0..schedule.layers {
                    execute(&ctx, |enc| {
                        actual
                            .encode(&ctx, enc, id, position, &weights[id], &input)
                            .unwrap();
                        expected
                            .encode(&ctx, enc, id, position, &oracle[id], &input)
                            .unwrap();
                    });
                    close(
                        f32_slice(&actual.output).unwrap(),
                        f32_slice(&expected.output).unwrap(),
                    );
                    let count = schedule.owner(id).map_or(0, |_| {
                        ((position + 1) / schedule.ratio(id)).min(c.index_topk)
                    });
                    assert_eq!(actual.selected_count(id, position), count);
                    assert_eq!(ids(&actual.selected, count), ids(&expected.selected, count));
                    if schedule.produces_index(id) {
                        let snapshot =
                            capture::Snapshot::new(&ctx, &c, position, id, true, &schedule)
                                .unwrap();
                        execute(&ctx, |enc| {
                            actual
                                .capture(&ctx, enc, &snapshot, &input, id, position)
                                .unwrap()
                        });
                        struct Sink {
                            owner: usize,
                            count: usize,
                            mask: bool,
                        }
                        impl capture::Capture for Sink {
                            fn wants_position(&self, _: usize) -> bool {
                                true
                            }
                            fn is_checkpoint(&self, _: usize) -> bool {
                                true
                            }
                            fn layer(
                                &mut self,
                                captured: capture::AttentionCapture<'_>,
                            ) -> Result<()> {
                                let checkpoint = captured.checkpoint.unwrap();
                                assert_eq!(checkpoint.kv_owner, self.owner);
                                assert_eq!(checkpoint.index_source, captured.layer);
                                assert_eq!(checkpoint.compressed_ids.len(), self.count);
                                assert_eq!(checkpoint.candidate_mask.is_some(), self.mask);
                                if let Some(mask) = checkpoint.candidate_mask {
                                    assert_eq!(mask.last(), Some(&1));
                                }
                                Ok(())
                            }
                        }
                        snapshot
                            .deliver(
                                &c,
                                position,
                                id,
                                &mut Sink {
                                    owner: schedule.owner(id).unwrap(),
                                    count,
                                    mask: schedule
                                        .candidate_source_layer
                                        .is_some_and(|source| id >= source),
                                },
                            )
                            .unwrap();
                    }
                }
            }
            assert_eq!((actual.next_layer, actual.next_position), (0, 5));
        }
    }
}

#[test]
fn projection_checks_quantized_row_bounds_without_f32_upcasting() {
    let ctx = MetalContext::new().unwrap();
    let schedule = Schedule {
        layers: 1,
        kv_source_layers: vec![],
        index_source_layers: vec![],
        compression_ratios: vec![0],
        candidate_source_layer: None,
    };
    let c = Config {
        hidden: 32,
        heads: 1,
        head_dim: 32,
        query_rank: 32,
        groups: 1,
        output_rank: 32,
        ..Config::default()
    };
    for dtype in [GgmlType::BF16, GgmlType::Q8_0] {
        let mut w = scheduled_weights(&ctx, &c, &schedule, 0, dtype).query_a;
        projection(&w, 32, 32).unwrap();
        w.tensor.offset = w.tensor.buffer.length() as u64 - 16;
        assert!(projection(&w, 32, 32).is_err());
        w.tensor.offset = 0;
        w.tensor.shape[0] = 31;
        assert!(projection(&w, 32, 32).is_err());
    }
}

#[test]
fn candidate_mask_rejects_incompatible_compressed_coordinates() {
    let schedule = Schedule {
        layers: 4,
        kv_source_layers: vec![1, 3],
        index_source_layers: vec![1, 3],
        compression_ratios: vec![0, 1, 1, 2],
        candidate_source_layer: Some(1),
    };
    schedule.validate().unwrap();
    assert!(
        bytes_with_schedule(&super::super::toy::config(), 8, &schedule)
            .unwrap_err()
            .contains("ratio-one coordinates")
    );
}

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
        native_control: None,
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

#[test]
fn split_sink_real_geometry_same_run_witness() {
    let ctx = MetalContext::new().unwrap();
    let (heads, dim, window, k, stride) = (64usize, 512usize, 128usize, 512usize, 641usize);
    let make = |n, seed| {
        (0..n)
            .map(|i| ((i * seed % 101) as f32 - 50.0) / 103.0)
            .collect::<Vec<_>>()
    };
    let q = make(heads * dim, 37);
    let raw = make(window * dim, 19);
    let compressed = make(k * dim, 23);
    let sinks = make(heads, 29);
    let up = |v: &[f32]| upload(&ctx, v, vec![v.len() as u64]).unwrap();
    let query = up(&q);
    let raw_gpu = up(&raw);
    let compressed_gpu = up(&compressed);
    let sink = up(&sinks);
    let selection = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&(0..k as u32).rev().collect::<Vec<_>>()),
        vec![k as u64],
        GgmlType::I32,
    )
    .unwrap();
    let scores = tensor(&ctx, heads * stride, GgmlType::F32).unwrap();
    let split = up(&vec![f32::NAN; heads * dim]);
    let serial = up(&vec![f32::NAN; heads * dim]);
    let status = tensor(&ctx, 1, GgmlType::I32).unwrap();
    let encode =
        |enc: &KernelEncoder, parallel, nr: usize, nk: usize, length: u32, query: &MetalTensor| {
            enc.set_bytes(0, &[dim as u32, nr as u32, 17u32, window as u32]);
            enc.set_bytes(1, &[nk as u32, stride as u32]);
            for (slot, t) in [
                (2, query),
                (3, &raw_gpu),
                (4, &compressed_gpu),
                (5, &selection),
                (6, &sink),
                (7, &scores),
                (8, if parallel { &split } else { &serial }),
                (10, &status),
            ] {
                enc.set_tensor(slot, t);
            }
            enc.set_bytes(9, &length);
            if parallel {
                sink_passes(&ctx, enc, heads, dim, nr + nk).unwrap();
            } else {
                pipeline(&ctx, enc, "sink").unwrap();
                dispatch(enc, heads);
            }
        };
    for (nr, nk) in [(0, 0), (1, 0), (128, 0), (128, 512)] {
        metal::dispatch_census_begin();
        execute(&ctx, |enc| {
            encode(enc, false, nr, nk, k as u32, &query);
            encode(enc, true, nr, nk, k as u32, &query);
        });
        let census = metal::dispatch_census_take();
        let expected_passes = if nr + nk == 0 {
            vec![("sink", heads), ("softmax", heads), ("output", heads * dim)]
        } else {
            vec![
                ("sink", heads),
                ("score", heads * (nr + nk)),
                ("softmax", heads),
                ("output", heads * dim),
            ]
        };
        assert_eq!(census.len(), expected_passes.len());
        for (row, (name, count)) in census.iter().zip(expected_passes) {
            assert_eq!(row.kernel, format!("kernel_deepseek_v41_attention_{name}"));
            assert_eq!(row.grid_tgs * row.tg_threads, count as u64);
            assert_eq!(row.encoder_ordinal, census[0].encoder_ordinal);
            assert!(!row.encoder_concurrent);
        }
        let actual = f32_slice(&split).unwrap();
        let incumbent = f32_slice(&serial).unwrap();
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            incumbent.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "serial mismatch {nr}+{nk}"
        );
        let rows: Vec<_> = (0..nr)
            .map(|i| raw[((17 + i) % window) * dim..((17 + i) % window + 1) * dim].to_vec())
            .chain((0..nk).map(|i| compressed[(k - 1 - i) * dim..(k - i) * dim].to_vec()))
            .collect();
        for h in 0..heads {
            let expected = op::sparse_attention(
                &q[h * dim..(h + 1) * dim],
                &rows,
                &(0..nr + nk).collect::<Vec<_>>(),
                sinks[h],
            )
            .unwrap();
            close(&actual[h * dim..(h + 1) * dim], &expected);
        }
        assert_eq!(ids(&status, 1), [0]);
        // Alternate incumbent/split ordering; warm up both before timed samples.
        let mut times = [Vec::new(), Vec::new()];
        for round in 0..9 {
            for which in [round % 2, 1 - round % 2] {
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                for _ in 0..4 {
                    encode(&enc, which == 1, nr, nk, k as u32, &query);
                }
                enc.end();
                metal::commit_and_wait(&command).unwrap();
                if round > 0 {
                    times[which].push(command.GPUEndTime() - command.GPUStartTime());
                }
            }
        }
        for t in &mut times {
            t.sort_by(f64::total_cmp);
        }
        assert!(times.iter().all(|t| t[4] > 0.0));
        eprintln!(
            "split sink {heads}x{dim} {nr}+{nk}: same-run median speedup {:.3}x, bitwise incumbent match",
            times[0][4] / times[1][4]
        );
    }
    execute(&ctx, |enc| {
        encode(enc, true, window, k, (k - 1) as u32, &query)
    });
    assert_eq!(ids(&status, 1), [2]);
    assert!(f32_slice(&split).unwrap().iter().all(|&v| v == 0.0));
    execute(&ctx, |enc| {
        encode(enc, false, window, k, (k - 1) as u32, &query)
    });
    assert_eq!(f32_slice(&split).unwrap(), f32_slice(&serial).unwrap());
    execute(&ctx, |enc| encode(enc, true, window, k, k as u32, &query));
    assert_eq!(ids(&status, 1), [2]);
    assert!(f32_slice(&split).unwrap().iter().any(|&v| v != 0.0));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bad_query = up(&vec![bad; heads * dim]);
        execute(&ctx, |enc| {
            encode(enc, true, 1, 0, k as u32, &bad_query);
        });
        assert_eq!(ids(&status, 1), [6]);
        execute(&ctx, |enc| {
            encode(enc, false, 1, 0, k as u32, &bad_query);
        });
        assert!(f32_slice(&split).unwrap().iter().all(|v| !v.is_finite()));
        assert!(f32_slice(&serial).unwrap().iter().all(|v| !v.is_finite()));
        assert_eq!(ids(&status, 1), [6]);
    }
}
